// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::AggregateCacheMode;
use super::ArrayInput;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::make_minmax_dtype;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::array::VTable;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;
use crate::validity::Validity;

#[rstest]
#[case(AggregateCacheMode::Array)]
#[case(AggregateCacheMode::Input)]
#[case(AggregateCacheMode::Disabled)]
fn selected_store_and_clones(#[case] mode: AggregateCacheMode) -> VortexResult<()> {
    let array = buffer![1i32, 2, 3].into_array();
    let input = ArrayInput::new(array.clone()).with_cache_mode(mode);
    let clone = input.clone();
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(input.get_result(&sum), Precision::Absent);
    assert_eq!(i64::try_from(&input.compute_result(&sum, &mut ctx)?)?, 6);
    let cached = Precision::Exact(Scalar::primitive(6i64, Nullability::Nullable));
    match mode {
        AggregateCacheMode::Array => {
            assert_eq!(clone.get_result(&sum), cached);
            assert_eq!(array.aggregations().get_result(&sum), cached);
        }
        AggregateCacheMode::Input => {
            assert_eq!(clone.get_result(&sum), cached);
            assert_eq!(array.aggregations().get_result(&sum), Precision::Absent);
            assert_eq!(ArrayInput::new(array).get_result(&sum), Precision::Absent);
        }
        AggregateCacheMode::Disabled => {
            assert_eq!(clone.get_result(&sum), Precision::Absent);
            assert_eq!(array.aggregations().get_result(&sum), Precision::Absent);
        }
    }
    Ok(())
}

#[rstest]
#[case(false)]
#[case(true)]
fn numerical_options_do_not_alias(#[case] include_first: bool) -> VortexResult<()> {
    let input = ArrayInput::new(buffer![1.0f64, f64::NAN].into_array());
    let skip = Sum.bind(NumericalAggregateOpts::skip_nans());
    let include = Sum.bind(NumericalAggregateOpts::include_nans());
    let order = if include_first {
        [&include, &skip]
    } else {
        [&skip, &include]
    };
    let mut ctx = array_session().create_execution_ctx();
    input.compute_result(order[0], &mut ctx)?;
    assert_eq!(input.get_result(order[1]), Precision::Absent);
    input.compute_result(order[1], &mut ctx)?;
    assert_eq!(f64::try_from(&input.compute_result(&skip, &mut ctx)?)?, 1.0);
    assert!(f64::try_from(&input.compute_result(&include, &mut ctx)?)?.is_nan());
    Ok(())
}

#[rstest]
#[case(AggregateCacheMode::Array)]
#[case(AggregateCacheMode::Input)]
#[case(AggregateCacheMode::Disabled)]
fn producer_indices_handle_empty_all_and_sparse(
    #[case] mode: AggregateCacheMode,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    for (mask, expected) in [
        (Mask::AllFalse(4), vec![]),
        (Mask::AllTrue(4), vec![0u64, 1, 2, 3]),
        (Mask::from_indices(8, [1, 4, 6]), vec![1u64, 4, 6]),
    ] {
        let input = ArrayInput::from_mask_indices_with_cache_mode(&mask, mode)?;
        assert_arrays_eq!(input.array(), PrimitiveArray::from_iter(expected), &mut ctx);
        let sorted = IsSorted.bind(IsSortedOptions { strict: true });
        if mode == AggregateCacheMode::Disabled {
            assert_eq!(input.get_result(&sorted), Precision::Absent);
        } else {
            assert_eq!(input.get_result(&sorted), Precision::Exact(true.into()));
        }
    }
    Ok(())
}

#[test]
fn stable_subsets_keep_bounds_and_positive_sortedness() -> VortexResult<()> {
    let input = ArrayInput::from_mask_indices(&Mask::from_indices(10, [1, 4, 6, 9]))?;
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let sorted = IsSorted.bind(IsSortedOptions { strict: true });
    let mut ctx = array_session().create_execution_ctx();
    input.compute_result(&sum, &mut ctx)?;
    let sliced = input.slice(1..3)?;
    assert!(matches!(sliced.get_result(&min), Precision::Inexact(_)));
    assert!(matches!(sliced.get_result(&max), Precision::Inexact(_)));
    assert_eq!(sliced.get_result(&sorted), Precision::Exact(true.into()));
    assert_eq!(sliced.get_result(&sum), Precision::Absent);
    assert_eq!(u64::try_from(&sliced.compute_result(&min, &mut ctx)?)?, 4);
    assert_eq!(u64::try_from(&input.compute_result(&min, &mut ctx)?)?, 1);
    let filtered = input.filter(Mask::from_indices(4, [0, 2]))?;
    assert!(matches!(filtered.get_result(&max), Precision::Inexact(_)));
    assert_eq!(filtered.get_result(&sorted), Precision::Exact(true.into()));
    assert_arrays_eq!(filtered.array(), buffer![1u64, 6].into_array(), &mut ctx);
    Ok(())
}

#[test]
fn negative_sortedness_and_noninteger_facts_do_not_propagate() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    let input = ArrayInput::new(buffer![3i32, 1, 2].into_array());
    assert_eq!(input.compute_result(&sorted, &mut ctx)?, false.into());
    let subset = input.slice(1..3)?;
    assert_eq!(subset.get_result(&sorted), Precision::Absent);
    assert_eq!(subset.compute_result(&sorted, &mut ctx)?, true.into());
    let float = ArrayInput::new(buffer![1.0f64, 2.0].into_array());
    float.compute_result(&sorted, &mut ctx)?;
    assert_eq!(float.slice(0..1)?.get_result(&sorted), Precision::Absent);
    Ok(())
}

#[test]
fn sortedness_uses_nulls_first_and_preserves_strictness() -> VortexResult<()> {
    let input = ArrayInput::new(
        PrimitiveArray::from_option_iter([None, Some(1i32), Some(1), Some(2)]).into_array(),
    );
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    let strict = IsSorted.bind(IsSortedOptions { strict: true });
    let mut ctx = array_session().create_execution_ctx();
    assert_eq!(input.compute_result(&sorted, &mut ctx)?, true.into());
    assert_eq!(input.compute_result(&strict, &mut ctx)?, false.into());
    let subset = input.filter(Mask::from_indices(4, [0, 1, 3]))?;
    assert_eq!(subset.get_result(&sorted), Precision::Exact(true.into()));
    assert_eq!(subset.get_result(&strict), Precision::Absent);
    assert_eq!(subset.compute_result(&strict, &mut ctx)?, true.into());
    Ok(())
}

#[test]
fn scoped_contexts_retain_owners_and_leave_parent_unchanged() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let outer = ArrayInput::from_mask_indices(&Mask::AllTrue(3))?;
    let array = outer.array().clone();
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    let parent = ctx.with_aggregate_input(&outer);
    drop(outer);
    assert_eq!(
        parent.aggregate_result(&array, &sorted),
        Precision::Exact(true.into())
    );
    let different = ArrayInput::new(buffer![9u64, 2, 1].into_array());
    let child = parent.with_aggregate_input(&different);
    assert_eq!(
        child.aggregate_result(&array, &sorted),
        Precision::Exact(true.into())
    );
    assert_eq!(
        child.aggregate_result(different.array(), &sorted),
        Precision::Absent
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _child = parent.with_aggregate_input(&different);
            panic!("scope unwind");
        }))
        .is_err()
    );
    assert_eq!(
        parent.aggregate_result(&array, &sorted),
        Precision::Exact(true.into())
    );
    assert!(different.execute_cast(PType::U8.into(), &mut ctx).is_ok());
    let invalid = ArrayInput::new(buffer![-1i32].into_array());
    assert!(invalid.execute_cast(PType::U32.into(), &mut ctx).is_err());
    assert_eq!(ctx.aggregate_cache_mode(), AggregateCacheMode::Array);
    Ok(())
}

#[derive(Debug)]
struct IncorrectMinMax;

impl DynAggregateKernel for IncorrectMinMax {
    fn aggregate(
        &self,
        _: &AggregateFnRef,
        batch: &ArrayRef,
        _: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        Ok(Some(Scalar::struct_(
            make_minmax_dtype(batch.dtype()),
            vec![
                Scalar::primitive(0i16, Nullability::NonNullable),
                Scalar::primitive(1i16, Nullability::NonNullable),
            ],
        )))
    }
}

#[test]
fn safe_aggregate_results_cannot_bypass_decimal_precision() -> VortexResult<()> {
    let session = array_session();
    session.aggregate_fns().register_aggregate_kernel(
        Primitive.id(),
        Some(MinMax.id()),
        &IncorrectMinMax,
    );
    let mut ctx = session.create_execution_ctx();
    let input = ArrayInput::new(buffer![1000i16].into_array());
    input.compute_result(&MinMax.bind(NumericalAggregateOpts::skip_nans()), &mut ctx)?;
    assert!(input.verified_bounds().is_none());
    let decimal = DType::Decimal(DecimalDType::new(3, 0), Nullability::NonNullable);
    assert!(input.execute_cast(decimal.clone(), &mut ctx).is_err());
    input.validate_integer_bounds(&mut ctx)?;
    assert!(
        !input
            .verified_bounds()
            .vortex_expect("direct validation retained proof")
            .fits(&decimal)
    );
    assert!(input.execute_cast(decimal, &mut ctx).is_err());
    Ok(())
}

#[test]
fn direct_proof_ignores_null_slots_and_checks_nullability() -> VortexResult<()> {
    let array = PrimitiveArray::new(buffer![1000i16, 12], Validity::from_iter([false, true]));
    let input = ArrayInput::new(array.into_array());
    let mut ctx = array_session().create_execution_ctx();
    input.validate_integer_bounds(&mut ctx)?;
    let dtype = DType::Decimal(DecimalDType::new(3, 0), Nullability::Nullable);
    assert!(
        input
            .verified_bounds()
            .vortex_expect("direct validation retained proof")
            .fits(&dtype)
    );
    assert!(input.execute_cast(dtype, &mut ctx).is_ok());
    assert!(
        input
            .execute_cast(
                DType::Decimal(DecimalDType::new(3, 0), Nullability::NonNullable),
                &mut ctx
            )
            .is_err()
    );
    assert!(
        ArrayInput::new(ConstantArray::new(1i32, 4).into_array())
            .validate_integer_bounds(&mut ctx)
            .is_err()
    );
    assert!(
        ArrayInput::new(buffer![1.0f32].into_array())
            .validate_integer_bounds(&mut ctx)
            .is_err()
    );
    Ok(())
}

#[test]
fn input_subsets_bypass_array_result_inheritance() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = buffer![1i32, 2, 3].into_array();
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    array.aggregations().compute_result(&sorted, &mut ctx)?;
    let input = ArrayInput::new(array);
    let sliced = input.slice(0..2)?;
    assert_eq!(sliced.get_result(&sorted), Precision::Absent);
    assert_eq!(
        sliced.array().aggregations().get_result(&sorted),
        Precision::Absent
    );
    Ok(())
}

#[rstest]
#[case(AggregateCacheMode::Input)]
#[case(AggregateCacheMode::Disabled)]
fn nullable_cast_does_not_populate_validity_array_cache(
    #[case] mode: AggregateCacheMode,
) -> VortexResult<()> {
    let validity = BoolArray::from_iter([true, true]).into_array();
    let input = ArrayInput::new(
        PrimitiveArray::new(buffer![12i16, 34], Validity::Array(validity.clone())).into_array(),
    )
    .with_cache_mode(mode);
    let mut ctx = array_session().create_execution_ctx();
    input.validate_integer_bounds(&mut ctx)?;
    input.execute_cast(
        DType::Decimal(DecimalDType::new(3, 0), Nullability::NonNullable),
        &mut ctx,
    )?;
    assert_eq!(
        validity
            .aggregations()
            .get_result(&Min.bind(NumericalAggregateOpts::skip_nans())),
        Precision::Absent
    );
    Ok(())
}

#[rstest]
#[case(AggregateCacheMode::Array)]
#[case(AggregateCacheMode::Input)]
#[case(AggregateCacheMode::Disabled)]
fn an_empty_slice_does_not_inherit_constantness(
    #[case] mode: AggregateCacheMode,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = ArrayInput::new(buffer![7i32, 7].into_array()).with_cache_mode(mode);
    let constant = IsConstant.bind(EmptyOptions);
    assert_eq!(input.compute_result(&constant, &mut ctx)?, true.into());
    let empty = input.slice(0..0)?;
    assert_eq!(empty.get_result(&constant), Precision::Absent);
    assert_eq!(empty.compute_result(&constant, &mut ctx)?, false.into());
    Ok(())
}
