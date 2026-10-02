// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::PatchSlotIndices;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_sequence::Sequence;

pub mod bitpack_compress;
pub mod bitpack_decompress;
pub mod unpack_iter;

#[cfg(test)]
mod tests;

use crate::BitPackedArray;
use crate::FL_CHUNK_SIZE;
use crate::bitpack_compress::bitpack_encode;
use crate::unpack_iter::BitPacked as BitPackedIter;
use crate::unpack_iter::BitUnpackedChunks;

#[array_slots(crate::BitPacked)]
pub struct BitPackedSlots {
    /// The indices of exception values that don't fit in the bit-packed representation.
    #[slot(0)]
    pub patch_indices: Option<ArrayRef>,
    /// The exception values that don't fit in the bit-packed representation.
    #[slot(1)]
    pub patch_values: Option<ArrayRef>,
    /// Chunk offsets for the patch indices/values.
    #[slot(2)]
    pub patch_chunk_offsets: Option<ArrayRef>,
    /// The validity bitmap indicating which elements are non-null.
    #[slot(3)]
    pub validity_child: Option<ArrayRef>,
    /// Byte boundaries of the packed blocks as non-nullable unsigned integers, including one
    /// trailing boundary.
    /// Block `i` is packed at `(block_offsets[i + 1] - block_offsets[i]) / 128` bits. When every
    /// block has the same width, this is a [`SequenceArray`](vortex_sequence::SequenceArray) with
    /// step `128 * bit_width`.
    #[slot(4)]
    pub block_offsets: ArrayRef,
}

pub(crate) const PATCH_SLOTS: PatchSlotIndices = PatchSlotIndices {
    indices: BitPackedSlots::PATCH_INDICES,
    values: BitPackedSlots::PATCH_VALUES,
    chunk_offsets: BitPackedSlots::PATCH_CHUNK_OFFSETS,
};

/// Byte boundaries for `num_chunks` chunks that are all packed at `bit_width`.
pub(crate) fn block_offsets_from_constant_bit_width(
    bit_width: u8,
    num_chunks: usize,
) -> VortexResult<ArrayRef> {
    let step = 128 * u64::from(bit_width);
    vortex_ensure!(
        num_chunks < usize::MAX,
        "Block offsets length does not fit in usize"
    );
    vortex_ensure!(
        u64::try_from(num_chunks).is_ok_and(|chunks| chunks.checked_mul(step).is_some()),
        "Uniform block offsets do not fit in u64"
    );

    // SAFETY: The sequence has at least one entry, an integer base of zero, and a nonnegative
    // integer step. Its final value fits u64 by the check above, so every boundary does too.
    let offsets = unsafe {
        Sequence::new_unchecked(
            0u64.into(),
            step.into(),
            PType::U64,
            Nullability::NonNullable,
            num_chunks + 1,
        )
    };
    Ok(offsets.into_array())
}

/// Check that `offsets` holds `num_blocks + 1` boundaries spanning `packed_len` bytes, each block a
/// whole number of 128-byte rows with a bit width supported by `ptype`.
///
/// Boundaries are only inspected when they are a sequence or materialized on the host.
pub(crate) fn validate_block_offsets(
    offsets: &ArrayRef,
    ptype: PType,
    num_blocks: usize,
    packed_len: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        offsets.dtype().is_unsigned_int() && !offsets.dtype().is_nullable(),
        "Expected non-nullable unsigned integer block offsets, got {}",
        offsets.dtype()
    );
    vortex_ensure!(
        offsets.len() == num_blocks + 1,
        "Expected {} block boundaries, got {}",
        num_blocks + 1,
        offsets.len()
    );
    let max_bit_width = ptype.bit_width() as u64;
    if let Some(sequence) = offsets.as_opt::<Sequence>() {
        validate_sequence_offsets(sequence, max_bit_width, num_blocks, packed_len)
    } else if let Some(primitive) = offsets.as_opt::<Primitive>()
        && primitive.buffer_handle().is_on_host()
    {
        match_each_unsigned_integer_ptype!(primitive.ptype(), |T| {
            validate_primitive_offsets(primitive.as_slice::<T>(), max_bit_width, packed_len)
        })
    } else {
        Ok(())
    }
}

/// Check that a sequence of `num_blocks + 1` boundaries steps by whole 128-byte rows of at most
/// `max_bit_width` bits and spans `packed_len` bytes.
fn validate_sequence_offsets(
    offsets: ArrayView<'_, Sequence>,
    max_bit_width: u64,
    num_blocks: usize,
    packed_len: usize,
) -> VortexResult<()> {
    let step = offsets.multiplier().cast::<u64>()?;
    vortex_ensure!(
        step % 128 == 0 && step / 128 <= max_bit_width,
        "Block offsets step {step} is not a supported bit width (at most {max_bit_width} bits)"
    );
    let span = step * num_blocks as u64;
    vortex_ensure!(
        span == packed_len as u64,
        "Block offsets span {span} bytes, but the packed buffer has {packed_len}"
    );
    Ok(())
}

/// Check that each block between `boundaries` is a whole number of 128-byte rows of at most
/// `max_bit_width` bits, and that the boundaries span `packed_len` bytes.
fn validate_primitive_offsets<T: Copy + Display>(
    boundaries: &[T],
    max_bit_width: u64,
    packed_len: usize,
) -> VortexResult<()>
where
    u64: From<T>,
{
    for pair in boundaries.windows(2) {
        let size = u64::from(pair[1]).checked_sub(u64::from(pair[0]));
        vortex_ensure!(
            size.is_some_and(|size| size % 128 == 0 && size / 128 <= max_bit_width),
            "Block boundaries {} and {} do not hold a supported bit width (at most {max_bit_width} bits)",
            pair[0],
            pair[1]
        );
    }
    let span = match boundaries {
        [first, .., last] => u64::from(*last) - u64::from(*first),
        _ => 0,
    };
    vortex_ensure!(
        span == packed_len as u64,
        "Block offsets span {span} bytes, but the packed buffer has {packed_len}"
    );
    Ok(())
}

/// The packed payload and children extracted from a [`BitPackedArray`].
pub struct BitPackedDataParts {
    /// The position of the first logical value within the first packed block.
    pub offset: u16,
    /// Byte boundaries of the packed blocks, including the trailing end boundary.
    pub block_offsets: ArrayRef,
    /// The number of logical values in the array.
    pub len: usize,
    /// The buffer containing the packed blocks.
    pub packed: BufferHandle,
    /// Exception values and their positions in the array.
    pub patches: Option<Patches>,
    /// The validity of the logical values.
    pub validity: Validity,
}

#[derive(Clone, Debug)]
pub struct BitPackedData {
    /// The offset within the first block (created with a slice).
    /// 0 <= offset < 1024
    pub(super) offset: u16,
    pub(super) packed: BufferHandle,
    /// Patch metadata for reconstructing Patches from slots.
    pub(super) patches_data: Option<PatchesData>,
}

impl Display for BitPackedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}", self.offset)
    }
}

impl BitPackedData {
    /// Create the packed payload and patch metadata for a [`BitPackedArray`].
    ///
    /// Returns an error if `offset` is outside the first 1024-value block. The dtype, length,
    /// validity, patches, and block boundaries are validated when the payload is assembled into
    /// an array, for example by [`crate::BitPacked::try_new_with_block_offsets`].
    pub fn try_new(
        packed: BufferHandle,
        patches: Option<Patches>,
        offset: u16,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            offset < 1024,
            "Offset must be less than the full block i.e., 1024, got {offset}"
        );

        Ok(Self {
            offset,
            packed,
            patches_data: patches.as_ref().map(PatchesData::from_patches),
        })
    }

    pub(crate) fn validate(
        ptype: PType,
        validity: &Validity,
        patches: Option<&Patches>,
        length: usize,
    ) -> VortexResult<()> {
        vortex_ensure!(ptype.is_int(), MismatchedTypes: "integer", ptype);

        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == length,
                "BitPackedArray validity length {validity_len} != array length {length}",
            );
        }

        // Validate patches
        if let Some(patches) = patches {
            Self::validate_patches(patches, ptype, length)?;
        }

        Ok(())
    }

    fn validate_patches(patches: &Patches, ptype: PType, len: usize) -> VortexResult<()> {
        // Ensure that array and patches have same ptype
        vortex_ensure!(
            patches.dtype().eq_ignore_nullability(ptype.into()),
            "Patches DType {} does not match BitPackedArray dtype {}",
            patches.dtype().as_nonnullable(),
            ptype
        );

        vortex_ensure!(
            patches.array_len() == len,
            "BitPackedArray patches length {} != expected {len}",
            patches.array_len(),
        );

        Ok(())
    }

    pub fn ptype(&self, dtype: &DType) -> PType {
        dtype.as_ptype()
    }

    /// Underlying bit packed values as byte array
    #[inline]
    pub fn packed(&self) -> &BufferHandle {
        &self.packed
    }

    /// Access the slice of packed values as an array of `T`
    #[inline]
    pub fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        let packed_bytes = self.packed().as_host();
        let packed_ptr: *const T = packed_bytes.as_ptr().cast();
        // Return number of elements of type `T` packed in the buffer
        let packed_len = packed_bytes.len() / size_of::<T>();

        // SAFETY: as_slice points to buffer memory that outlives the lifetime of `self`.
        //  Unfortunately Rust cannot understand this, so we reconstruct the slice from raw parts
        //  to get it to reinterpret the lifetime.
        unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) }
    }

    /// Accessor for bit unpacked chunks
    pub(crate) fn unpacked_chunks<'a, T: BitPackedIter>(
        &'a self,
        dtype: &DType,
        bit_width: u8,
        len: usize,
        scratch: &'a mut [MaybeUninit<T>; FL_CHUNK_SIZE],
    ) -> VortexResult<BitUnpackedChunks<'a, T>> {
        assert_eq!(
            T::PTYPE,
            self.ptype(dtype),
            "Requested type doesn't match the array ptype"
        );
        BitUnpackedChunks::try_new(self, bit_width, len, scratch)
    }

    #[inline]
    pub fn offset(&self) -> u16 {
        self.offset
    }

    /// Bit-pack an array of primitive integers down to the target bit-width using the FastLanes
    /// SIMD-accelerated packing kernels.
    ///
    /// # Errors
    ///
    /// If the provided array is not an integer type, an error will be returned.
    ///
    /// If the provided array contains negative values, an error will be returned.
    ///
    /// If the requested bit-width for packing is larger than the array's native width, an
    /// error will be returned.
    pub fn encode(
        array: &ArrayRef,
        bit_width: u8,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<BitPackedArray> {
        let parray: PrimitiveArray = array
            .clone()
            .try_downcast::<Primitive>()
            .map_err(|a| vortex_err!(InvalidArgument: "Bitpacking can only encode primitive arrays, got {}", a.encoding_id()))?;
        bitpack_encode(&parray, bit_width, None, ctx)
    }
}

pub trait BitPackedArrayExt: BitPackedArraySlotsExt {
    #[inline]
    fn packed(&self) -> &BufferHandle {
        BitPackedData::packed(self)
    }

    /// The bit width shared by every block, or `None` if block offsets are not a sequence.
    #[inline]
    fn constant_bit_width_opt(&self) -> Option<u8> {
        let step = self
            .block_offsets()
            .as_opt::<Sequence>()?
            .multiplier()
            .cast::<u64>()
            .ok()?;
        u8::try_from(step / 128).ok()
    }

    /// The bit width shared by every block, or an error if block offsets are not a sequence.
    #[inline]
    fn constant_bit_width(&self) -> VortexResult<u8> {
        self.constant_bit_width_opt().ok_or_else(|| {
            vortex_err!("BitPacked block offsets are not a constant-width sequence")
        })
    }

    #[inline]
    fn offset(&self) -> u16 {
        BitPackedData::offset(self)
    }

    #[inline]
    fn patches(&self) -> Option<Patches> {
        PatchesData::patches_from_slots(
            self.patches_data.as_ref(),
            self.as_ref().len(),
            self.as_ref().slots(),
            PATCH_SLOTS,
        )
    }

    #[inline]
    fn validity(&self) -> Validity {
        child_to_validity(self.validity_child(), self.as_ref().dtype().nullability())
    }

    #[inline]
    fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        BitPackedData::packed_slice::<T>(self)
    }

    #[inline]
    fn unpacked_chunks<'a, T: BitPackedIter>(
        &'a self,
        scratch: &'a mut [MaybeUninit<T>; FL_CHUNK_SIZE],
    ) -> VortexResult<BitUnpackedChunks<'a, T>> {
        BitPackedData::unpacked_chunks::<T>(
            self,
            self.as_ref().dtype(),
            self.constant_bit_width()?,
            self.as_ref().len(),
            scratch,
        )
    }
}

impl<T: TypedArrayRef<crate::BitPacked>> BitPackedArrayExt for T {}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::Buffer;
    use vortex_session::VortexSession;

    use crate::BitPackedData;
    use crate::bitpacking::array::BitPackedArrayExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_encode() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = [
            Some(1u64),
            None,
            Some(1),
            None,
            Some(1),
            None,
            Some(u64::MAX),
        ];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let packed = BitPackedData::encode(&uncompressed.into_array(), 1, &mut ctx).unwrap();
        let expected = PrimitiveArray::from_option_iter(values);
        let packed_primitive = packed
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(packed_primitive, expected, &mut ctx);
    }

    #[test]
    fn test_encode_too_wide() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = [Some(1u8), None, Some(1), None, Some(1), None];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let _packed = BitPackedData::encode(&uncompressed.clone().into_array(), 8, &mut ctx)
            .expect_err("Cannot pack value into the same width");
        let _packed = BitPackedData::encode(&uncompressed.into_array(), 9, &mut ctx)
            .expect_err("Cannot pack value into larger width");
    }

    #[test]
    fn signed_with_patches() {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Buffer<i32> = (0i32..=512).collect();
        let parray = values.clone().into_array();

        let packed_with_patches = BitPackedData::encode(&parray, 9, &mut ctx).unwrap();
        assert!(packed_with_patches.patches().is_some());
        let packed_primitive = packed_with_patches
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            packed_primitive,
            PrimitiveArray::new(values, vortex_array::validity::Validity::NonNullable),
            &mut ctx
        );
    }
}
