// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the block offsets child of uniformly bit-packed arrays.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::cast::CastKernel;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedArraySlotsExt;
use crate::bitpacking::array::block_offsets_from_constant_bit_width;
use crate::bitpacking::bitpack_compress::bitpack_to_best_bit_width;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

fn encode(values: &[u32]) -> VortexResult<BitPackedArray> {
    let mut ctx = SESSION.create_execution_ctx();
    bitpack_to_best_bit_width(&PrimitiveArray::from_iter(values.iter().copied()), &mut ctx)
}

fn sequence(base: u64, step: u64, len: usize) -> VortexResult<ArrayRef> {
    Ok(Sequence::try_new_typed(base, step, Nullability::NonNullable, len)?.into_array())
}

fn uniform() -> VortexResult<BitPackedArray> {
    encode(&(0..3000u32).map(|i| i % 128).collect::<Vec<_>>())
}

fn with_block_offsets(array: &BitPackedArray, offsets: ArrayRef) -> VortexResult<BitPackedArray> {
    BitPacked::try_new_with_block_offsets(
        array.packed().clone(),
        array.dtype().as_ptype(),
        array.validity()?,
        array.patches(),
        offsets,
        array.len(),
        array.offset(),
    )
}

#[test]
fn uniform_block_offsets_are_a_sequence() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let uniform = uniform()?;
    assert_eq!(uniform.constant_bit_width_opt(), Some(7));
    assert_eq!(uniform.constant_bit_width()?, 7);
    assert!(uniform.block_offsets().is::<Sequence>());
    assert_arrays_eq!(
        uniform.block_offsets(),
        buffer![0u64, 896, 1792, 2688].into_array(),
        &mut ctx
    );
    let rebased = with_block_offsets(&uniform, sequence(128, 896, 4)?)?;
    assert_eq!(rebased.constant_bit_width_opt(), Some(7));
    assert_arrays_eq!(uniform, rebased, &mut ctx);
    Ok(())
}

#[rstest]
#[case::equal_steps(buffer![0u64, 896, 1792, 2688])]
#[case::different_widths(buffer![0u64, 384, 1408, 2688])]
fn materialized_block_offsets_have_no_constant_width(
    #[case] offsets: vortex_buffer::Buffer<u64>,
) -> VortexResult<()> {
    let array = with_block_offsets(&uniform()?, offsets.into_array())?;
    assert_eq!(array.constant_bit_width_opt(), None);
    assert!(array.constant_bit_width().is_err());
    // Decoding per-block widths is not supported yet.
    assert!(
        array
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .is_err()
    );
    Ok(())
}

#[rstest]
#[case::equal_steps(buffer![0u64, 896, 1792, 2688])]
#[case::different_widths(buffer![0u64, 384, 1408, 2688])]
fn casts_with_materialized_block_offsets_decline(
    #[case] offsets: vortex_buffer::Buffer<u64>,
    #[values(
        DType::Primitive(PType::U32, Nullability::Nullable),
        DType::Primitive(PType::U64, Nullability::NonNullable)
    )]
    dtype: DType,
) -> VortexResult<()> {
    let array = with_block_offsets(&uniform()?, offsets.into_array())?;
    let mut ctx = SESSION.create_execution_ctx();

    assert!(<BitPacked as CastReduce>::cast(array.as_view(), &dtype)?.is_none());
    assert!(<BitPacked as CastKernel>::cast(array.as_view(), &dtype, &mut ctx)?.is_none());
    Ok(())
}

#[rstest]
#[case::too_few_boundaries(sequence(0, 896, 3))]
#[case::step_not_multiple_of_128(sequence(0, 895, 4))]
#[case::step_disagrees_with_packed_len(sequence(0, 768, 4))]
#[case::unaligned_block(Ok(buffer![0u64, 896, 1791, 2688].into_array()))]
#[case::decreasing(Ok(buffer![0u64, 896, 768, 2688].into_array()))]
#[case::span_disagrees_with_packed_len(Ok(buffer![0u64, 768, 1536, 2304].into_array()))]
#[case::wrong_ptype(Ok(Sequence::try_new_typed(0u32, 896, Nullability::NonNullable, 4)?.into_array()))]
#[case::nullable(Ok(Sequence::try_new_typed(0u64, 896, Nullability::Nullable, 4)?.into_array()))]
fn invalid_block_offsets_are_rejected(#[case] offsets: VortexResult<ArrayRef>) -> VortexResult<()> {
    assert!(with_block_offsets(&uniform()?, offsets?).is_err());
    Ok(())
}

#[rstest]
fn block_width_must_fit_ptype(
    #[values(
        PType::U8, PType::I8, PType::U16, PType::I16, PType::U32, PType::I32, PType::U64,
        PType::I64
    )]
    ptype: PType,
    #[values(false, true)] materialized: bool,
    #[values(false, true)] too_wide: bool,
) -> VortexResult<()> {
    let bit_width = u8::try_from(ptype.bit_width())? + u8::from(too_wide);
    let step = 128 * u64::from(bit_width);
    let offsets = if materialized {
        buffer![0u64, step].into_array()
    } else {
        sequence(0, step, 2)?
    };
    let result = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(ByteBuffer::zeroed(128 * usize::from(bit_width))),
        ptype,
        Validity::NonNullable,
        None,
        offsets,
        1024,
        0,
    );
    assert_eq!(result.is_err(), too_wide);
    Ok(())
}

#[rstest]
#[case::equal_steps(buffer![0u64, 512, 1024])]
#[case::different_widths(buffer![0u64, 384, 1024])]
fn unsupported_offsets_leave_builder_unchanged(
    #[case] offsets: vortex_buffer::Buffer<u64>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(ByteBuffer::zeroed(1024)),
        PType::U32,
        Validity::AllValid,
        None,
        offsets.into_array(),
        2048,
        0,
    )?;
    let mut builder = PrimitiveBuilder::<u32>::with_capacity_in(
        Nullability::Nullable,
        array.len() + 2,
        ctx.allocator(),
    );
    builder.append_null();
    assert!(array.append_to_builder(&mut builder, &mut ctx).is_err());
    builder.append_value(7);
    assert_arrays_eq!(
        builder.finish_into_primitive(),
        PrimitiveArray::from_option_iter([None, Some(7u32)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn construct_blocks_without_a_uniform_width() -> VortexResult<()> {
    // Two blocks at widths 3 and 4 occupy 896 bytes, which no constant width can represent.
    let array = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(ByteBuffer::zeroed(896)),
        PType::U32,
        Validity::NonNullable,
        None,
        buffer![0u64, 384, 896].into_array(),
        2048,
        0,
    )?;
    assert_eq!(array.constant_bit_width_opt(), None);
    Ok(())
}

#[test]
fn empty_and_zero_width_offsets() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let empty = encode(&[])?;
    assert_arrays_eq!(empty.block_offsets(), buffer![0u64].into_array(), &mut ctx);
    let zeros = encode(&vec![0u32; 2049])?;
    assert_arrays_eq!(
        zeros.block_offsets(),
        buffer![0u64, 0, 0, 0].into_array(),
        &mut ctx
    );
    assert_arrays_eq!(zeros, PrimitiveArray::from_iter(vec![0u32; 2049]), &mut ctx);
    Ok(())
}

#[rstest]
#[case::sequence(sequence(128, 512, 3))]
#[case::equal_steps(Ok(buffer![128u64, 640, 1152].into_array()))]
#[case::different_widths(Ok(buffer![128u64, 512, 1152].into_array()))]
fn into_parts_preserves_block_offsets(#[case] offsets: VortexResult<ArrayRef>) -> VortexResult<()> {
    let offsets = offsets?;
    let array = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(ByteBuffer::zeroed(1024)),
        PType::U32,
        Validity::NonNullable,
        None,
        offsets.clone(),
        1500,
        17,
    )?;
    let parts = BitPacked::into_parts(array);
    let rebuilt = BitPacked::try_new_with_block_offsets(
        parts.packed,
        PType::U32,
        parts.validity,
        parts.patches,
        parts.block_offsets,
        parts.len,
        parts.offset,
    )?;
    assert!(ArrayRef::ptr_eq(&offsets, rebuilt.block_offsets()));
    assert_eq!(rebuilt.len(), 1500);
    assert_eq!(rebuilt.offset(), 17);
    Ok(())
}

#[cfg(target_pointer_width = "64")]
#[test]
fn block_offsets_from_constant_bit_width_reject_end_overflow() {
    assert!(block_offsets_from_constant_bit_width(64, usize::MAX / (128 * 64) + 1).is_err());
}

#[test]
fn block_offsets_from_constant_bit_width_reject_length_overflow() {
    assert!(block_offsets_from_constant_bit_width(0, usize::MAX).is_err());
}
