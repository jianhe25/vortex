// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Answer a comparison against a constant from cached statistics, without reading the values.
//!
//! When the constant lies outside the `[min, max]` range of the array, every valid position
//! compares the same way, so the result is a constant boolean carrying the array's validity.
//! Only statistics that are already cached are consulted: computing them would cost a full pass
//! over the data, which is exactly what this is trying to avoid.

use std::cmp::Ordering;

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

/// Try to answer `lhs op rhs` from the cached min/max statistics of the non-constant operand.
///
/// Returns `None` when neither operand is a non-null constant, when the statistics are missing,
/// or when they cannot decide the comparison for every row.
pub(super) fn compare_from_stats(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let (array, constant, op) = if let Some(constant) = rhs.as_constant() {
        (lhs, constant, op)
    } else if let Some(constant) = lhs.as_constant() {
        (rhs, constant, op.swap())
    } else {
        return Ok(None);
    };

    // The kernels reject mismatched dtypes and compare extensions through their storage; leave
    // both to them rather than answering from statistics of a different type.
    if constant.is_null()
        || array.is::<Constant>()
        || array.dtype().is_extension()
        || !array.dtype().eq_ignore_nullability(constant.dtype())
    {
        return Ok(None);
    }

    let Some(value) = decide_from_min_max(array, &constant, op) else {
        return Ok(None);
    };

    let nullability = array.dtype().nullability() | constant.dtype().nullability();
    constant_result(array, value, nullability, ctx).map(Some)
}

/// Decide `array[i] op constant` for every valid `i`, if the cached min/max bounds allow it.
fn decide_from_min_max(array: &ArrayRef, constant: &Scalar, op: CompareOperator) -> Option<bool> {
    let stats = array.statistics();

    // Min and max skip NaNs, but the kernels order NaN above every other value, so float bounds
    // only describe every row when the array is known to be NaN-free.
    if array.dtype().is_float()
        && !matches!(stats.get_as::<u64>(Stat::NaNCount), Precision::Exact(0))
    {
        return None;
    }

    // An inexact min is a lower bound and an inexact max an upper bound, which is all the range
    // checks need. Only the all-equal case requires both to be exact.
    let (min, min_exact) = bound(stats.get(Stat::Min));
    let (max, max_exact) = bound(stats.get(Stat::Max));
    // `None` when the statistic is absent or the scalars do not order against each other.
    let min_cmp = min.and_then(|min| min.partial_cmp(constant));
    let max_cmp = max.and_then(|max| max.partial_cmp(constant));

    match op {
        CompareOperator::Eq => eq_from_bounds(min_cmp, max_cmp, min_exact && max_exact),
        CompareOperator::NotEq => {
            eq_from_bounds(min_cmp, max_cmp, min_exact && max_exact).map(|eq| !eq)
        }
        CompareOperator::Lt => match (min_cmp, max_cmp) {
            (_, Some(Ordering::Less)) => Some(true),
            (Some(Ordering::Greater | Ordering::Equal), _) => Some(false),
            _ => None,
        },
        CompareOperator::Lte => match (min_cmp, max_cmp) {
            (_, Some(Ordering::Less | Ordering::Equal)) => Some(true),
            (Some(Ordering::Greater), _) => Some(false),
            _ => None,
        },
        CompareOperator::Gt => match (min_cmp, max_cmp) {
            (Some(Ordering::Greater), _) => Some(true),
            (_, Some(Ordering::Less | Ordering::Equal)) => Some(false),
            _ => None,
        },
        CompareOperator::Gte => match (min_cmp, max_cmp) {
            (Some(Ordering::Greater | Ordering::Equal), _) => Some(true),
            (_, Some(Ordering::Less)) => Some(false),
            _ => None,
        },
    }
}

fn eq_from_bounds(
    min_cmp: Option<Ordering>,
    max_cmp: Option<Ordering>,
    exact: bool,
) -> Option<bool> {
    match (min_cmp, max_cmp) {
        (Some(Ordering::Greater), _) | (_, Some(Ordering::Less)) => Some(false),
        (Some(Ordering::Equal), Some(Ordering::Equal)) if exact => Some(true),
        _ => None,
    }
}

fn bound(stat: Precision<Scalar>) -> (Option<Scalar>, bool) {
    match stat {
        Precision::Exact(value) => (Some(value), true),
        Precision::Inexact(value) => (Some(value), false),
        Precision::Absent => (None, false),
    }
}

/// A boolean result that is `value` at every valid position of `array`.
fn constant_result(
    array: &ArrayRef,
    value: bool,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = array.len();
    match array.validity()?.union_nullability(nullability) {
        Validity::NonNullable | Validity::AllValid => {
            Ok(ConstantArray::new(Scalar::bool(value, nullability), len).into_array())
        }
        Validity::AllInvalid => {
            Ok(ConstantArray::new(Scalar::null(DType::Bool(nullability)), len).into_array())
        }
        validity @ Validity::Array(_) => {
            let bits = BitBuffer::full_in(value, len, ctx.allocator().clone());
            Ok(BoolArray::try_new(bits, validity)?.into_array())
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::Nullability;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::operators::CompareOperator;

    const OPS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    const STATS: [Stat; 5] = [
        Stat::Min,
        Stat::Max,
        Stat::IsSorted,
        Stat::IsStrictSorted,
        Stat::NaNCount,
    ];

    /// Compare an array carrying cached statistics against a constant, in both operand orders,
    /// and check the result against the same values compared without any statistics.
    fn assert_matches_linear(
        build: impl Fn() -> ArrayRef,
        constants: &[Scalar],
        with_stats: impl Fn(&ArrayRef),
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let plain = build();
        let stats = build();
        with_stats(&stats);

        for constant in constants {
            let constant = ConstantArray::new(constant.clone(), plain.len()).into_array();
            for op in OPS {
                assert_arrays_eq!(
                    stats.binary(constant.clone(), op.into())?,
                    plain.binary(constant.clone(), op.into())?,
                    &mut ctx
                );
                assert_arrays_eq!(
                    constant.binary(stats.clone(), op.into())?,
                    constant.binary(plain.clone(), op.into())?,
                    &mut ctx
                );
            }
        }
        Ok(())
    }

    fn compute_stats(array: &ArrayRef) {
        let mut ctx = array_session().create_execution_ctx();
        array
            .statistics()
            .compute_all(&STATS, &mut ctx)
            .vortex_expect("compute stats");
    }

    fn i32_constants(values: impl IntoIterator<Item = i32>) -> Vec<Scalar> {
        values
            .into_iter()
            .map(|v| Scalar::primitive(v, Nullability::NonNullable))
            .collect()
    }

    #[rstest]
    #[case::non_nullable(|| PrimitiveArray::from_iter([1i32, 3, 3, 3, 5, 8, 8, 13]).into_array())]
    #[case::leading_nulls(|| {
        PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(3), Some(3), Some(8)])
            .into_array()
    })]
    #[case::all_null(|| PrimitiveArray::from_option_iter([None::<i32>, None]).into_array())]
    #[case::single(|| PrimitiveArray::from_iter([3i32]).into_array())]
    fn sorted_primitive(#[case] build: fn() -> ArrayRef) -> VortexResult<()> {
        assert_matches_linear(
            build,
            &i32_constants([-1, 0, 1, 2, 3, 4, 8, 13, 14]),
            compute_stats,
        )
    }

    #[test]
    fn unsorted_primitive_min_max() -> VortexResult<()> {
        assert_matches_linear(
            || {
                PrimitiveArray::from_option_iter([Some(7i32), None, Some(2), Some(9), Some(2)])
                    .into_array()
            },
            &i32_constants([0, 1, 2, 5, 9, 10]),
            compute_stats,
        )
    }

    #[test]
    fn inexact_bounds() -> VortexResult<()> {
        assert_matches_linear(
            || PrimitiveArray::from_iter([4i32, 5, 6]).into_array(),
            &i32_constants([0, 2, 4, 5, 6, 8, 10]),
            |array| {
                array.statistics().set(Stat::Min, Precision::inexact(2i32));
                array.statistics().set(Stat::Max, Precision::inexact(8i32));
            },
        )
    }

    #[test]
    fn sorted_floats_with_nan() -> VortexResult<()> {
        assert_matches_linear(
            || PrimitiveArray::from_iter([-1.5f64, -0.0, 0.0, 2.0, f64::NAN]).into_array(),
            &[-2.0f64, -0.0, 0.0, 1.0, 2.0, 3.0, f64::NAN]
                .map(|v| Scalar::primitive(v, Nullability::NonNullable)),
            compute_stats,
        )
    }

    /// Min and max skip NaNs, so without a NaN count they must not prune: `NaN > 5.0` holds.
    #[test]
    fn float_bounds_without_nan_count() -> VortexResult<()> {
        assert_matches_linear(
            || PrimitiveArray::from_iter([1.0f64, f64::NAN]).into_array(),
            &[0.0f64, 1.0, 5.0].map(|v| Scalar::primitive(v, Nullability::NonNullable)),
            |array| {
                array.statistics().set(Stat::Min, Precision::exact(1.0f64));
                array.statistics().set(Stat::Max, Precision::exact(1.0f64));
            },
        )
    }

    #[test]
    fn utf8_min_max() -> VortexResult<()> {
        assert_matches_linear(
            || VarBinViewArray::from_iter_str(["pear", "apple", "fig"]).into_array(),
            &["a", "apple", "banana", "pear", "zebra"].map(Scalar::from),
            compute_stats,
        )
    }

    #[test]
    fn out_of_range_constant_skips_the_scan() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([4i32, 5, 6]).into_array();
        compute_stats(&array);
        let constant = ConstantArray::new(10i32, array.len()).into_array();
        let mut ctx = array_session().create_execution_ctx();

        let result = array
            .binary(constant, CompareOperator::Gte.into())?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Constant>());
        assert_arrays_eq!(result, ConstantArray::new(false, 3), &mut ctx);
        Ok(())
    }
}
