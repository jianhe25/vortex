// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValuesRef;

use super::chunked_indices;
use super::take::UNPACK_CHUNK_THRESHOLD;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;

/// The threshold over which it is faster to fully unpack the entire [`BitPackedArray`](crate::BitPackedArray) and then
/// filter the result than to unpack only specific bitpacked values into the output buffer.
pub const fn unpack_then_filter_threshold(ptype: PType) -> f64 {
    // TODO(connor): Where did these numbers come from? Add a public link after validating them.
    // These numbers probably don't work for in-place filtering either.
    match ptype.byte_width() {
        1 => 0.03,
        2 => 0.03,
        4 => 0.075,
        _ => 0.09,
        // >8 bytes may have a higher threshold. These numbers are derived from a GCP c2-standard-4
        // with a "Cascade Lake" CPU.
    }
}

/// Kernel to execute filtering directly on a bit-packed array.
impl FilterKernel for BitPacked {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let values = match mask {
            Mask::AllTrue(_) | Mask::AllFalse(_) => {
                return Ok(None);
            }
            Mask::Values(values) => values,
        };

        // Preserve two-bit compression for selective filters. Dense masks use the existing
        // decode-and-filter path below, since repacking all selected values is more expensive.
        if array.dtype().as_ptype() == PType::U8
            && array.bit_width() == 2
            && values.density() <= unpack_then_filter_threshold(PType::U8)
            && array.patches().is_none()
            && array.validity()?.execute_mask(array.len(), ctx)?.all_true()
        {
            let packed = filter_two_bit_u8(array.data(), values);
            let validity = match array.validity()? {
                Validity::NonNullable => Validity::NonNullable,
                _ => Validity::AllValid,
            };
            return Ok(Some(
                BitPacked::try_new(
                    BufferHandle::new_host(packed.freeze().into_byte_buffer()),
                    PType::U8,
                    validity,
                    None,
                    2,
                    values.true_count(),
                    0,
                )?
                .into_array(),
            ));
        }

        // If the density is high enough, then we would rather decompress the whole array and then apply
        // a filter over decompressing values one by one.
        if values.density() > unpack_then_filter_threshold(array.dtype().as_ptype()) {
            return Ok(None);
        }

        // Filter and patch using the correct unsigned type for FastLanes, then cast to signed if needed.
        let primitive =
            match_each_unsigned_integer_ptype!(array.dtype().as_ptype().to_unsigned(), |U| {
                let (buffer, validity) = filter_primitive_without_patches::<U>(array, values)?;
                // reinterpret_cast for signed types.
                let primitive = PrimitiveArray::new(buffer, validity);
                if array.dtype().as_ptype().is_signed_int() {
                    PrimitiveArray::from_buffer_handle(
                        primitive.buffer_handle().clone(),
                        array.dtype().as_ptype(),
                        primitive.validity()?,
                    )
                } else {
                    primitive
                }
            });

        let patches = array
            .patches()
            .map(|patches| patches.filter(&Mask::Values(MaskValuesRef::clone(values)), ctx))
            .transpose()?
            .flatten();

        if let Some(patches) = patches {
            let mut prim_array = primitive;
            prim_array = prim_array.patch(&patches, ctx)?;
            return Ok(Some(prim_array.into_array()));
        }

        Ok(Some(primitive.into_array()))
    }
}

fn filter_two_bit_u8(array: &BitPackedData, selection: &MaskValuesRef) -> BufferMut<u8> {
    let mut packed = BufferMut::with_capacity(selection.true_count().div_ceil(1024) * 256);
    let mut unpacked_block = [0u8; 1024];
    let mut output_block = [0u8; 1024];
    let mut packed_block = [0u8; 256];
    let mut output_count = 0;
    let mut current_block = usize::MAX;
    let mut selected_in_block = 0;
    let source = array.packed_slice::<u8>();
    let offset = array.offset() as usize;

    selection.bit_buffer().for_each_set_index(|index| {
        let physical = index + offset;
        let block_index = physical / 1024;
        if block_index != current_block {
            current_block = block_index;
            selected_in_block = 0;
        }
        let block = &source[block_index * 256..][..256];
        let value = if selected_in_block < UNPACK_CHUNK_THRESHOLD {
            // SAFETY: the selected index falls inside a complete, padded FastLanes block.
            unsafe { u8::unchecked_unpack_single(2, block, physical % 1024) }
        } else {
            if selected_in_block == UNPACK_CHUNK_THRESHOLD {
                // SAFETY: both input and scratch cover one complete FastLanes block.
                unsafe { u8::unchecked_unpack(2, block, &mut unpacked_block) };
            }
            unpacked_block[physical % 1024]
        };
        selected_in_block += 1;
        output_block[output_count] = value;
        output_count += 1;
        if output_count == 1024 {
            // SAFETY: both buffers have the full lengths required by the two-bit packer.
            unsafe { u8::unchecked_pack(2, &output_block, &mut packed_block) };
            packed.extend_from_slice(&packed_block);
            output_count = 0;
        }
    });

    if output_count != 0 {
        output_block[output_count..].fill(0);
        // SAFETY: the tail is padded to a complete FastLanes block.
        unsafe { u8::unchecked_pack(2, &output_block, &mut packed_block) };
        packed.extend_from_slice(&packed_block);
    }

    packed
}

/// Specialized filter kernel for primitive bit-packed arrays.
///
/// Because the FastLanes bit-packing kernels are only implemented for unsigned types, the provided
/// `U` should be promoted to the unsigned variant for any target bit width.
/// For example, if the array is bit-packed `i16`, this function should be called with `U = u16`.
///
/// This function fully decompresses the array for all but the most selective masks because the
/// FastLanes decompression is so fast and the bookkeepping necessary to decompress individual
/// elements is relatively slow.
///
/// Returns a tuple of (values buffer, validity mask).
fn filter_primitive_without_patches<U: UnsignedPType + BitPacking>(
    array: ArrayView<'_, BitPacked>,
    selection: &MaskValuesRef,
) -> VortexResult<(Buffer<U>, Validity)> {
    let values = filter_with_indices(array.data(), selection.indices());
    let validity = array
        .validity()?
        .filter(&Mask::Values(MaskValuesRef::clone(selection)))?;

    Ok((values.freeze(), validity))
}

fn filter_with_indices<T: NativePType + BitPacking>(
    array: &BitPackedData,
    indices: &[usize],
) -> BufferMut<T> {
    let offset = array.offset() as usize;
    let bit_width = array.bit_width() as usize;
    let mut values = BufferMut::with_capacity(indices.len());

    // Some re-usable memory to store per-chunk indices.
    let mut unpacked = [const { MaybeUninit::<T>::uninit() }; 1024];
    let packed_bytes = array.packed_slice::<T>();

    // Group the indices by the FastLanes chunk they belong to.
    let chunk_size = 128 * bit_width / size_of::<T>();

    chunked_indices(
        indices.iter().copied(),
        offset,
        |chunk_idx, indices_within_chunk| {
            let packed = &packed_bytes[chunk_idx * chunk_size..][..chunk_size];

            if indices_within_chunk.len() == 1024 {
                // Unpack the entire chunk.
                unsafe {
                    let values_len = values.len();
                    values.set_len(values_len + 1024);
                    BitPacking::unchecked_unpack(
                        bit_width,
                        packed,
                        &mut values.as_mut_slice()[values_len..],
                    );
                }
            } else if indices_within_chunk.len() > UNPACK_CHUNK_THRESHOLD {
                // Unpack into a temporary chunk and then copy the values.
                unsafe {
                    let dst: &mut [MaybeUninit<T>] = &mut unpacked;
                    let dst: &mut [T] = std::mem::transmute(dst);
                    BitPacking::unchecked_unpack(bit_width, packed, dst);
                }
                values.extend_trusted(
                    indices_within_chunk
                        .iter()
                        .map(|&idx| unsafe { unpacked.get_unchecked(idx).assume_init() }),
                );
            } else {
                // Otherwise, unpack each element individually.
                values.extend_trusted(indices_within_chunk.iter().map(|&idx| unsafe {
                    BitPacking::unchecked_unpack_single(bit_width, packed, idx)
                }));
            }
        },
    );

    values
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray as _;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::filter::FilterKernel;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::BitPacked;
    use crate::BitPackedData;
    use crate::bitpacking::array::BitPackedArrayExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn take_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();

        let mask = Mask::from_indices(bitpacked.len(), vec![0, 125, 2047, 2049, 2151, 2790]);

        let primitive_result = bitpacked.filter(mask).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([0u8, 62, 31, 33, 9, 18]),
            &mut ctx
        );
    }

    #[test]
    fn sparse_two_bit_filter_stays_packed() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..4096).map(|idx| (idx % 3) as u8));
        let packed = BitPackedData::encode(&values.into_array(), 2, &mut ctx)?;
        let mask = Mask::from_indices(4096, [1, 1023, 1024, 2049, 4095]);
        let filtered = BitPacked::filter(packed.as_view(), &mask, &mut ctx)?
            .expect("selective two-bit filter has a packed kernel");
        assert!(filtered.is::<BitPacked>());
        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter([1u8, 0, 1, 0, 0]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn sparse_two_bit_filter_handles_sliced_nullable_input() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = PrimitiveArray::from_option_iter((0..5000).map(|idx| Some((idx % 3) as u8)))
            .into_array();
        let packed = BitPackedData::encode(&source, 2, &mut ctx)?;
        let sliced = packed.slice(123..4123)?;
        let mask = Mask::from_indices(4000, [0, 900, 901, 3999]);
        let filtered = BitPacked::filter(
            sliced
                .as_opt::<BitPacked>()
                .expect("slice stays bit-packed"),
            &mask,
            &mut ctx,
        )?
        .expect("selective two-bit filter has a packed kernel");
        assert!(filtered.is::<BitPacked>());
        assert_eq!(filtered.dtype(), source.dtype());
        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_option_iter(
                [123, 1023, 1024, 4122].map(|idx| Some((idx % 3) as u8))
            ),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn dense_two_bit_filter_uses_decode_fallback() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..4096).map(|idx| (idx % 3) as u8));
        let packed = BitPackedData::encode(&values.into_array(), 2, &mut ctx)?;
        let mask = Mask::from_iter((0..4096).map(|idx| idx % 2 == 0));
        assert!(BitPacked::filter(packed.as_view(), &mask, &mut ctx)?.is_none());
        Ok(())
    }

    #[test]
    fn take_sliced_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();
        let sliced = bitpacked.slice(128..2050).unwrap();

        let mask = Mask::from_indices(sliced.len(), vec![1919, 1921]);

        let primitive_result = sliced.filter(mask).unwrap();
        assert_arrays_eq!(
            primitive_result,
            PrimitiveArray::from_iter([31u8, 33]),
            &mut ctx
        );
    }

    #[test]
    fn filter_bitpacked() {
        let mut ctx = SESSION.create_execution_ctx();
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 6, &mut ctx).unwrap();
        let filtered = bitpacked.filter(Mask::from_indices(4096, 0..1024)).unwrap();
        let filtered_prim = filtered.execute::<PrimitiveArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            filtered_prim,
            PrimitiveArray::from_iter((0..1024).map(|i| (i % 63) as u8)),
            &mut ctx
        );
    }

    #[test]
    fn filter_bitpacked_signed() {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Buffer<i64> = (0..500).collect();
        let unpacked = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 9, &mut ctx).unwrap();
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), 0..250))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter(values[0..250].iter().copied()),
            &mut ctx
        );
    }

    #[test]
    fn test_filter_bitpacked_conformance() {
        let mut ctx = SESSION.create_execution_ctx();
        // Test with u8 values
        let unpacked = buffer![1u8, 2, 3, 4, 5].into_array();
        let bitpacked = BitPackedData::encode(&unpacked, 3, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array(), &mut ctx);

        // Test with u32 values
        let unpacked = buffer![100u32, 200, 300, 400, 500].into_array();
        let bitpacked = BitPackedData::encode(&unpacked, 9, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array(), &mut ctx);

        // Test with nullable values
        let unpacked = PrimitiveArray::from_option_iter([Some(1u16), None, Some(3), Some(4), None]);
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 3, &mut ctx).unwrap();
        test_filter_conformance(&bitpacked.into_array(), &mut ctx);
    }

    /// Regression test for signed integers with patches.
    ///
    /// When filtering signed integers that have patches (exceptions), the patches
    /// are stored with the signed type but FastLanes uses unsigned types internally.
    /// This test ensures that the type handling is correct.
    #[test]
    fn filter_bitpacked_signed_with_patches() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create signed integer values where some exceed the bit width (causing patches).
        // Values 0-127 fit in 7 bits, but 1000 and 2000 do not.
        let values: Vec<i32> = vec![0, 10, 1000, 20, 30, 2000, 40, 50, 60, 70];
        let unpacked = PrimitiveArray::from_iter(values.clone());
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 7, &mut ctx).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Expected patches for values exceeding bit width"
        );

        // Filter to include some patched and some non-patched values.
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), vec![0, 2, 5, 9]))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter([0i32, 1000, 2000, 70]),
            &mut ctx
        );
    }

    /// Regression test for signed integers with patches using low selectivity.
    ///
    /// This test uses a low selectivity filter which takes a different code path
    /// that doesn't fully decompress the array first.
    #[test]
    fn filter_bitpacked_signed_with_patches_low_selectivity() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a larger array with signed integers and some patches.
        let values: Vec<i32> = (0..1000)
            .map(|i| {
                if i % 100 == 0 {
                    10000 + i // These will be patches (exceed 7 bits)
                } else {
                    i % 128 // These fit in 7 bits
                }
            })
            .collect();
        let unpacked = PrimitiveArray::from_iter(values.clone());
        let bitpacked = BitPackedData::encode(&unpacked.into_array(), 7, &mut ctx).unwrap();
        assert!(
            bitpacked.patches().is_some(),
            "Expected patches for values exceeding bit width"
        );

        // Use low selectivity (only select 2% of values) to avoid full decompression.
        let indices: Vec<usize> = (0..20).collect();
        let filtered = bitpacked
            .filter(Mask::from_indices(values.len(), indices))
            .unwrap()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        let expected: Vec<i32> = values[0..20].to_vec();
        assert_arrays_eq!(filtered, PrimitiveArray::from_iter(expected), &mut ctx);
    }

    #[test]
    fn dense_genotype_filter_preserves_values() {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<u8> = (0..20_000).map(|idx| (idx % 3) as u8).collect();
        let source = PrimitiveArray::from_iter(values.clone()).into_array();
        let packed = BitPackedData::encode(&source, 2, &mut ctx).unwrap();
        let mask = Mask::from_indices(values.len(), (0..values.len()).filter(|idx| idx % 2 == 0));
        let filtered = packed
            .filter(mask)
            .unwrap()
            .execute::<vortex_array::ArrayRef>(&mut ctx)
            .unwrap()
            .into_array();
        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter(values.into_iter().step_by(2)),
            &mut ctx
        );
    }

    #[test]
    fn nullable_dense_genotype_filter_preserves_dtype() {
        let mut ctx = SESSION.create_execution_ctx();
        let source = PrimitiveArray::from_option_iter((0..4096).map(|idx| Some((idx % 3) as u8)))
            .into_array();
        let packed = BitPackedData::encode(&source, 2, &mut ctx).unwrap();
        let mask = Mask::from_indices(4096, (0..4096).filter(|idx| idx % 2 == 0));
        let filtered = packed
            .filter(mask)
            .unwrap()
            .execute::<vortex_array::ArrayRef>(&mut ctx)
            .unwrap()
            .into_array();
        assert_eq!(filtered.dtype(), source.dtype());
        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_option_iter((0..4096).step_by(2).map(|idx| Some((idx % 3) as u8))),
            &mut ctx
        );
    }

    #[test]
    fn dense_genotype_filter_handles_sliced_offset() {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<u8> = (0..5000).map(|idx| (idx % 3) as u8).collect();
        let packed = BitPackedData::encode(
            &PrimitiveArray::from_iter(values.clone()).into_array(),
            2,
            &mut ctx,
        )
        .unwrap();
        let sliced = packed.slice(123..4123).unwrap();
        let mask = Mask::from_indices(4000, (0..4000).filter(|idx| idx % 2 == 1));
        let filtered = sliced
            .filter(mask)
            .unwrap()
            .execute::<vortex_array::ArrayRef>(&mut ctx)
            .unwrap()
            .into_array();
        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_iter(
                (123..4123)
                    .filter(|idx| (idx - 123) % 2 == 1)
                    .map(|idx| values[idx])
            ),
            &mut ctx
        );
    }

    #[test]
    fn dense_genotype_filter_sliced_nullable_all_valid() {
        let mut ctx = SESSION.create_execution_ctx();
        let source = PrimitiveArray::from_option_iter((0..5000).map(|idx| Some((idx % 3) as u8)))
            .into_array();
        let packed = BitPackedData::encode(&source, 2, &mut ctx).unwrap();
        let sliced = packed.slice(123..4123).unwrap();
        let mask = Mask::from_indices(4000, (0..4000).filter(|idx| idx % 2 == 1));
        let filtered = sliced
            .filter(mask)
            .unwrap()
            .execute::<vortex_array::ArrayRef>(&mut ctx)
            .unwrap()
            .into_array();
        assert_eq!(filtered.dtype(), source.dtype());
        assert_arrays_eq!(
            filtered,
            PrimitiveArray::from_option_iter(
                (123..4123)
                    .filter(|idx| (idx - 123) % 2 == 1)
                    .map(|idx| Some((idx % 3) as u8))
            ),
            &mut ctx
        );
    }
}
