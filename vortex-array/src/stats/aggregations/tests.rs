// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;
use crate::stats::Aggregations;

#[test]
fn known_results_remain_distinct_from_missing() -> VortexResult<()> {
    let array = buffer![1i32, 2].into_array();
    let store = Aggregations::default();
    let cache = store.to_ref(&array);
    let count = NullCount.bind(EmptyOptions);
    let constant = IsConstant.bind(EmptyOptions);
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());

    assert_eq!(cache.get_result(&count), Precision::Absent);
    cache.insert_result(count.clone(), Precision::Exact(0u64.into()))?;
    cache.insert_result(constant.clone(), Precision::Exact(false.into()))?;
    cache.insert_result(
        sum.clone(),
        Precision::Exact(Scalar::null(sum.return_dtype(array.dtype()).unwrap())),
    )?;

    assert_eq!(cache.get_result(&count), Precision::Exact(0u64.into()));
    assert_eq!(cache.get_result(&constant), Precision::Exact(false.into()));
    assert!(cache.get_result(&sum).as_exact().unwrap().is_null());
    assert_eq!(cache.snapshot_results().iter().count(), 3);
    Ok(())
}

#[test]
fn numerical_options_are_distinct_cache_keys() -> VortexResult<()> {
    let array = buffer![1.0f64, f64::NAN].into_array();
    let store = Aggregations::default();
    let cache = store.to_ref(&array);
    let skipped = Sum.bind(NumericalAggregateOpts::skip_nans());
    let included = Sum.bind(NumericalAggregateOpts::include_nans());
    let mut ctx = array_session().create_execution_ctx();

    assert_eq!(cache.compute_as::<f64>(&skipped, &mut ctx)?, 1.0);
    assert_eq!(cache.get_result(&included), Precision::Absent);
    assert!(cache.compute_as::<f64>(&included, &mut ctx)?.is_nan());
    assert_eq!(cache.compute_as::<f64>(&skipped, &mut ctx)?, 1.0);
    Ok(())
}

#[test]
fn errors_do_not_populate_the_cache() {
    let array = ConstantArray::new("text", 3).into_array();
    let store = Aggregations::default();
    let cache = store.to_ref(&array);
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let mut ctx = array_session().create_execution_ctx();

    assert!(cache.compute_result(&sum, &mut ctx).is_err());
    assert_eq!(cache.get_result(&sum), Precision::Absent);
    assert!(cache.snapshot_results().iter().next().is_none());
}

#[test]
fn insertion_checks_result_dtype() {
    let array = buffer![1i32].into_array();
    let store = Aggregations::default();
    let cache = store.to_ref(&array);
    let count = NullCount.bind(EmptyOptions);

    assert!(
        cache
            .insert_result(
                count.clone(),
                Precision::Exact(Scalar::primitive(1u64, Nullability::Nullable))
            )
            .is_err()
    );
    assert_eq!(cache.get_result(&count), Precision::Absent);
}
