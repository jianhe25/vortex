// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison of primitive arrays with specialized bitmap packing for 8-bit inputs, and a
//! binary-search path for arrays whose cached statistics mark them as sorted.

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::collect_bool_word;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::compare::bit_buffer_from_words;
use crate::scalar_fn::fns::binary::compare::collect_bits;
use crate::scalar_fn::fns::binary::compare::collect_zip_bits;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::binary::primitive_operand::PrimitiveOperand;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

/// Compare two primitive arrays of the same [`PType`].
///
/// Floats compare with Vortex's total ordering: `NaN` is the largest value, `-0.0 < +0.0`, and
/// equality is bitwise.
pub(super) fn compare_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ptype = PType::try_from(lhs.dtype())?;
    match_each_native_ptype!(ptype, |T| {
        compare_primitive_typed::<T>(lhs, rhs, op, nullability, ctx)
    })
}

fn compare_primitive_typed<T: NativePType>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let lhs_sorted = is_sorted_cached(lhs);
    let rhs_sorted = is_sorted_cached(rhs);
    let lhs = PrimitiveOperand::<T>::try_new(lhs, ctx)?;
    let rhs = PrimitiveOperand::<T>::try_new(rhs, ctx)?;
    if lhs.len() != rhs.len() {
        vortex_bail!(
            "compare operator requires equal lengths, got {} and {}",
            lhs.len(),
            rhs.len()
        );
    }

    let validity = compare_validity(lhs.validity(), rhs.validity(), nullability)?;

    let bits = match (&lhs, &rhs) {
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => compare_slices(lhs, rhs, op, ctx.allocator()),
        (
            PrimitiveOperand::Array {
                values: lhs,
                validity: lhs_validity,
            },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => {
            if lhs_sorted {
                compare_sorted_constant(lhs, lhs_validity, *rhs, op, ctx)?
            } else {
                compare_slice_constant(lhs, *rhs, op, ctx.allocator())
            }
        }
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Array {
                values: rhs,
                validity: rhs_validity,
            },
        ) => {
            if rhs_sorted {
                compare_sorted_constant(rhs, rhs_validity, *lhs, op.swap(), ctx)?
            } else {
                compare_slice_constant(rhs, *lhs, op.swap(), ctx.allocator())
            }
        }
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => {
            // Unreachable through `execute_compare` (constant-constant is folded there), but
            // cheap to answer anyway.
            BitBuffer::full_in(apply_op(*lhs, *rhs, op), len, ctx.allocator().clone())
        }
        (PrimitiveOperand::Null(_), _) | (_, PrimitiveOperand::Null(_)) => {
            return Ok(
                ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), len)
                    .into_array(),
            );
        }
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

/// Whether the array is known to be sorted from its cached statistics.
///
/// Sortedness is never computed here: that is a full pass, which the sorted path exists to avoid.
fn is_sorted_cached(array: &ArrayRef) -> bool {
    let stats = array.statistics();
    matches!(stats.get_as::<bool>(Stat::IsSorted), Precision::Exact(true))
        || matches!(
            stats.get_as::<bool>(Stat::IsStrictSorted),
            Precision::Exact(true)
        )
}

/// Compare a sorted slice against a constant by binary searching for the run of values equal to
/// the constant. Every comparison operator selects at most two contiguous runs of positions, so
/// the result is written with range fills instead of one predicate evaluation per element.
///
/// Sortedness orders nulls first and values with the same total ordering the linear kernel uses,
/// so the valid values are the sorted suffix after the leading nulls. Result bits at null
/// positions are masked by the validity and left unset.
fn compare_sorted_constant<T: NativePType>(
    values: &[T],
    validity: &Validity,
    constant: T,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    let len = values.len();
    let first_valid = match validity {
        Validity::NonNullable | Validity::AllValid => 0,
        Validity::AllInvalid => len,
        Validity::Array(_) => len - validity.execute_mask(len, ctx)?.true_count(),
    };

    let sorted = &values[first_valid..];
    let lower = first_valid + sorted.partition_point(|v| v.is_lt(constant));
    let upper = first_valid + sorted.partition_point(|v| v.is_le(constant));

    let (first, second) = match op {
        CompareOperator::Eq => (lower..upper, 0..0),
        CompareOperator::NotEq => (first_valid..lower, upper..len),
        CompareOperator::Lt => (first_valid..lower, 0..0),
        CompareOperator::Lte => (first_valid..upper, 0..0),
        CompareOperator::Gt => (upper..len, 0..0),
        CompareOperator::Gte => (lower..len, 0..0),
    };

    let mut bits = BitBufferMut::with_capacity_in(len, ctx.allocator().clone());
    bits.append_n(false, len);
    bits.fill_range(first.start, first.end, true);
    bits.fill_range(second.start, second.end, true);
    Ok(bits.freeze())
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn apply_op<T: NativePType>(lhs: T, rhs: T, op: CompareOperator) -> bool {
    match op {
        CompareOperator::Eq => lhs.is_eq(rhs),
        CompareOperator::NotEq => !lhs.is_eq(rhs),
        CompareOperator::Gt => lhs.is_gt(rhs),
        CompareOperator::Gte => lhs.is_ge(rhs),
        CompareOperator::Lt => lhs.is_lt(rhs),
        CompareOperator::Lte => lhs.is_le(rhs),
    }
}

fn compare_slices<T: NativePType>(
    lhs: &[T],
    rhs: &[T],
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    // Dispatch the operator outside the lane loop so each instantiation vectorizes a single
    // branch-free predicate.
    match op {
        CompareOperator::Eq => {
            collect_zip_bits_dispatch(lhs, rhs, |a: T, b: T| a.is_eq(b), allocator)
        }
        CompareOperator::NotEq => {
            collect_zip_bits_dispatch(lhs, rhs, |a: T, b: T| !a.is_eq(b), allocator)
        }
        CompareOperator::Gt => collect_zip_bits_dispatch(lhs, rhs, T::is_gt, allocator),
        CompareOperator::Gte => collect_zip_bits_dispatch(lhs, rhs, T::is_ge, allocator),
        CompareOperator::Lt => collect_zip_bits_dispatch(lhs, rhs, T::is_lt, allocator),
        CompareOperator::Lte => collect_zip_bits_dispatch(lhs, rhs, T::is_le, allocator),
    }
}

fn compare_slice_constant<T: NativePType>(
    lhs: &[T],
    rhs: T,
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    match op {
        CompareOperator::Eq => collect_bits_dispatch(lhs, |a: T| a.is_eq(rhs), allocator),
        CompareOperator::NotEq => collect_bits_dispatch(lhs, |a: T| !a.is_eq(rhs), allocator),
        CompareOperator::Gt => collect_bits_dispatch(lhs, |a: T| a.is_gt(rhs), allocator),
        CompareOperator::Gte => collect_bits_dispatch(lhs, |a: T| a.is_ge(rhs), allocator),
        CompareOperator::Lt => collect_bits_dispatch(lhs, |a: T| a.is_lt(rhs), allocator),
        CompareOperator::Lte => collect_bits_dispatch(lhs, |a: T| a.is_le(rhs), allocator),
    }
}

fn collect_bits_dispatch<T: NativePType>(
    values: &[T],
    f: impl Fn(T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    // This type check folds away during monomorphization. Wider masks keep the lane kernel:
    // byte packing regresses 64-bit comparisons on AVX2.
    if matches!(T::PTYPE, PType::I8 | PType::U8) {
        collect_bits_narrow(values, f, allocator)
    } else {
        collect_bits(values, f, allocator)
    }
}

fn collect_zip_bits_dispatch<T: NativePType>(
    lhs: &[T],
    rhs: &[T],
    f: impl Fn(T, T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    if matches!(T::PTYPE, PType::I8 | PType::U8) {
        collect_zip_bits_narrow(lhs, rhs, f, allocator)
    } else {
        collect_zip_bits(lhs, rhs, f, allocator)
    }
}

fn collect_bits_narrow<T: Copy>(
    values: &[T],
    f: impl Fn(T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let (chunks, tail) = values.as_chunks::<64>();
    let mut words = BufferMut::<u64>::zeroed_in(values.len().div_ceil(64), allocator.clone());
    // Fixed-size chunks let the compiler prove the predicate's indexing stays in bounds.
    for (word, chunk) in words.iter_mut().zip(chunks) {
        *word = collect_bool_word(64, |i| f(chunk[i]));
    }
    if !tail.is_empty() {
        words[chunks.len()] = collect_bool_word(tail.len(), |i| f(tail[i]));
    }
    bit_buffer_from_words(words, values.len())
}

fn collect_zip_bits_narrow<T: Copy>(
    lhs: &[T],
    rhs: &[T],
    f: impl Fn(T, T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    assert_eq!(lhs.len(), rhs.len());
    let (left_chunks, left_tail) = lhs.as_chunks::<64>();
    let (right_chunks, right_tail) = rhs.as_chunks::<64>();
    let mut words = BufferMut::<u64>::zeroed_in(lhs.len().div_ceil(64), allocator.clone());
    for ((word, left), right) in words.iter_mut().zip(left_chunks).zip(right_chunks) {
        *word = collect_bool_word(64, |i| f(left[i], right[i]));
    }
    if !left_tail.is_empty() {
        words[left_chunks.len()] =
            collect_bool_word(left_tail.len(), |i| f(left_tail[i], right_tail[i]));
    }
    bit_buffer_from_words(words, lhs.len())
}
