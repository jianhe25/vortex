// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexResult;

use super::Dict;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ConstantArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar::Scalar;
use crate::stats::StatsSet;
use crate::validity::Validity;

pub trait TakeReduce: VTable {
    /// Take elements from an array at the given indices without reading buffers.
    ///
    /// This trait is for take implementations that can operate purely on array metadata and
    /// structure without needing to read or execute on the underlying buffers. Implementations
    /// should return `None` if taking requires buffer access.
    ///
    /// # Preconditions
    ///
    /// The indices are guaranteed to be non-empty.
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
}

pub trait TakeExecute: VTable {
    /// Take elements from an array at the given indices, potentially reading buffers.
    ///
    /// Unlike [`TakeReduce`], this trait is for take implementations that may need to read
    /// and execute on the underlying buffers to produce the result.
    ///
    /// # Preconditions
    ///
    /// The indices are guaranteed to be non-empty.
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Short-circuits take for the inputs that need no encoding-specific work.
///
/// Returns `Some(result)` when the answer is already known, or `None` when take must proceed
/// normally.
fn short_circuit<V: VTable>(array: ArrayView<'_, V>, indices: &ArrayRef) -> Option<ArrayRef> {
    // Fast-path for empty indices.
    if indices.is_empty() {
        let result_dtype = array
            .dtype()
            .clone()
            .union_nullability(indices.dtype().nullability());
        return Some(Canonical::empty(&result_dtype).into_array());
    }

    // Fast-path for empty arrays: all indices must be null, return all-invalid result.
    if array.is_empty() {
        return Some(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                .into_array(),
        );
    }

    None
}

#[derive(Default, Debug)]
pub struct TakeReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for TakeReduceAdaptor<V>
where
    V: TakeReduce,
{
    type Parent = Dict;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ArrayView<'_, Dict>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle the values child (index 1), not the codes child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        if let Some(result) = short_circuit::<V>(array, parent.codes()) {
            return Ok(Some(result));
        }
        let result = <V as TakeReduce>::take(array, parent.codes())?;
        if let Some(taken) = &result {
            propagate_take_stats(array.array(), taken, parent.codes())?;
        }
        Ok(result)
    }
}

#[derive(Default, Debug)]
pub struct TakeExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for TakeExecuteAdaptor<V>
where
    V: TakeExecute,
{
    type Parent = Dict;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle the values child (index 1), not the codes child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        if let Some(result) = short_circuit::<V>(array, parent.codes()) {
            return Ok(Some(result));
        }
        if let Some(result) = take_strict_sorted_as_slice(array.array(), parent.codes(), ctx)? {
            return Ok(Some(result));
        }
        let result = <V as TakeExecute>::take(array, parent.codes(), ctx)?;
        if let Some(taken) = &result {
            propagate_take_stats(array.array(), taken, parent.codes())?;
        }
        Ok(result)
    }
}

/// Execute a take whose indices are a strictly sorted contiguous run as a slice.
///
/// Strictly increasing indices whose first and last are `len - 1` apart leave no gaps, so they
/// select exactly `first..=last`. A slice is zero-copy for most encodings, where a take gathers
/// every selected element. Only the first and last index are read, so the check costs two scalar
/// reads whatever the encoding of the indices.
///
/// Strictly sorted indices with gaps are left to the take kernels: filtering with a mask built
/// from them measured slower than taking for most canonical encodings (see the
/// `take_strict_sorted` benchmark).
///
/// Only a cached [`Stat::IsStrictSorted`] is consulted; computing sortedness would cost a pass
/// over the indices. Returns `None` when the indices are not known to be strictly sorted, are not
/// contiguous, contain nulls, or are out of bounds, leaving those cases to the take kernels.
pub(crate) fn take_strict_sorted_as_slice(
    values: &ArrayRef,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if indices.is_empty()
        || !matches!(
            indices.statistics().get_as::<bool>(Stat::IsStrictSorted),
            Precision::Exact(true)
        )
    {
        return Ok(None);
    }

    // Strict sortedness allows a null only in the first position, so a valid first index means
    // every index is valid. `as_opt` is `None` for a negative index and `Some(None)` for a null.
    let index_at = |position: usize, ctx: &mut ExecutionCtx| -> VortexResult<Option<usize>> {
        Ok(indices
            .execute_scalar(position, ctx)?
            .as_primitive()
            .as_opt::<usize>()
            .flatten())
    };
    let Some(first) = index_at(0, ctx)? else {
        return Ok(None);
    };
    let Some(last) = index_at(indices.len() - 1, ctx)? else {
        return Ok(None);
    };
    if last >= values.len() || last - first + 1 != indices.len() {
        return Ok(None);
    }

    let result_dtype = values
        .dtype()
        .union_nullability(indices.dtype().nullability());
    let sliced = values.slice(first..last + 1)?;
    if sliced.dtype() == &result_dtype {
        Ok(Some(sliced))
    } else {
        sliced.cast(result_dtype).map(Some)
    }
}

pub(crate) fn propagate_take_stats(
    source: &ArrayRef,
    target: &ArrayRef,
    indices: &ArrayRef,
) -> VortexResult<()> {
    let indices_all_valid = matches!(
        indices.validity()?,
        Validity::NonNullable | Validity::AllValid
    );
    target.statistics().with_mut_typed_stats_set(|mut st| {
        if indices_all_valid {
            let is_constant = source.statistics().get_as::<bool>(Stat::IsConstant);
            if matches!(is_constant, Precision::Exact(true)) {
                // Any combination of elements from a constant array is still const
                st.set(Stat::IsConstant, Precision::exact(true));
            }
        }
        let inexact_min_max = [Stat::Min, Stat::Max]
            .into_iter()
            .filter_map(|stat| match source.statistics().get(stat).into_inexact() {
                Precision::Exact(scalar) | Precision::Inexact(scalar) => {
                    scalar.into_value().map(|sv| (stat, Precision::Inexact(sv)))
                }
                Precision::Absent => None,
            })
            .collect::<SmallVec<_>>();
        st.combine_sets(
            &(unsafe { StatsSet::new_unchecked(inexact_min_max) }).as_typed_ref(source.dtype()),
        )
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;

    fn mark_strict_sorted(indices: ArrayRef) -> ArrayRef {
        indices
            .statistics()
            .set(Stat::IsStrictSorted, Precision::exact(true));
        indices
    }

    fn primitive_values() -> ArrayRef {
        PrimitiveArray::from_option_iter((0..10i32).map(|i| (i % 4 != 1).then_some(i * 10)))
            .into_array()
    }

    fn chunked_values() -> ArrayRef {
        ChunkedArray::from_iter([
            buffer![0i64, 1, 2].into_array(),
            buffer![3i64, 4, 5, 6].into_array(),
            buffer![7i64, 8, 9].into_array(),
        ])
        .into_array()
    }

    fn utf8_values() -> ArrayRef {
        VarBinViewArray::from_iter_str((0..10).map(|i| format!("value-{i}"))).into_array()
    }

    fn varbin_values() -> ArrayRef {
        VarBinArray::from_iter_nonnull(
            (0..10).map(|i| format!("v{i}")),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array()
    }

    fn struct_values() -> ArrayRef {
        StructArray::from_fields(&[
            ("a", buffer![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array()),
            ("b", utf8_values()),
        ])
        .expect("struct")
        .into_array()
    }

    /// Taking with indices marked strictly sorted must match taking with the same indices
    /// without the statistic.
    #[rstest]
    fn strict_sorted_take_matches_take(
        #[values(
            primitive_values as fn() -> ArrayRef,
            chunked_values,
            utf8_values,
            varbin_values,
            struct_values
        )]
        values: fn() -> ArrayRef,
        #[values(
            || buffer![0u32, 2, 3, 7, 9].into_array(),
            || buffer![3u8, 4, 5, 6].into_array(),
            || buffer![0i64, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
            || buffer![9i16].into_array(),
            || PrimitiveArray::from_option_iter([Some(1u16), Some(4), Some(8)]).into_array()
        )]
        indices: fn() -> ArrayRef,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let expected = values().take(indices())?;
        let actual = values().take(mark_strict_sorted(indices()))?;
        assert_eq!(actual.dtype(), expected.dtype());
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    /// Strict sortedness allows a single leading null index, which a slice cannot express.
    #[test]
    fn leading_null_index_falls_back_to_take() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let indices = || PrimitiveArray::from_option_iter([None, Some(2u32), Some(5)]).into_array();
        let expected = primitive_values().take(indices())?;
        let actual = primitive_values().take(mark_strict_sorted(indices()))?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn negative_strict_sorted_indices_error() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let result = primitive_values()
            .take(mark_strict_sorted(buffer![-1i32, 2, 5].into_array()))
            .and_then(|taken| taken.execute::<Canonical>(&mut ctx));
        assert!(result.is_err());
        Ok(())
    }
}
