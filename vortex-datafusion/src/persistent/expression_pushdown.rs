// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A physical optimizer rule that moves more expressions into Vortex scans.

use std::sync::Arc;

use datafusion_common::Result as DFResult;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::Transformed;
use datafusion_common::tree_node::TransformedResult;
use datafusion_common::tree_node::TreeNode;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::aggregates::AggregateMode;
use datafusion_physical_plan::expressions::Column;
use datafusion_physical_plan::projection::ProjectionExec;

use crate::VortexSource;

/// Moves expressions that DataFusion evaluates directly above a Vortex scan into the scan.
///
/// DataFusion leaves two kinds of expressions above the scan, where they run on fully decoded
/// columns instead of on the encoded data:
///
/// - Projections that reference a computed column of the scan, which DataFusion creates when it
///   eliminates common sub-expressions. This rule merges such a projection into the scan.
/// - Aggregate arguments, such as the `array_sum(xs)` in `SUM(array_sum(xs))`. This rule computes
///   them in a projection that it pushes into the scan.
///
/// With projection pushdown enabled the Vortex scan then evaluates the expressions it supports and
/// returns the rest to DataFusion. Plans are left unchanged when the input is not a Vortex scan or
/// the scan can't absorb the projection.
///
/// Register it after DataFusion's default rules, e.g. with
/// `SessionStateBuilder::with_physical_optimizer_rule`.
#[derive(Debug, Default)]
pub struct VortexExpressionPushdown;

impl VortexExpressionPushdown {
    /// Creates the rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for VortexExpressionPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|plan| {
            let rewritten = if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
                merge_projection(projection)?
            } else if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
                push_aggregate_arguments(aggregate)?
            } else {
                None
            };
            Ok(match rewritten {
                Some(rewritten) => Transformed::yes(rewritten),
                None => Transformed::no(plan),
            })
        })
        .data()
    }

    fn name(&self) -> &str {
        "VortexExpressionPushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn merge_projection(projection: &ProjectionExec) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
    push_projection_into_scan(projection.input(), projection)
}

/// Merges `projection` into the Vortex scan `input`, returning `None` if `input` isn't one.
///
/// Unlike DataFusion's projection pushdown, this also inlines computed scan columns referenced
/// several times. The Vortex scan computes repeated convertible sub-expressions once.
fn push_projection_into_scan(
    input: &Arc<dyn ExecutionPlan>,
    projection: &ProjectionExec,
) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
    let Some(config) = input
        .downcast_ref::<DataSourceExec>()
        .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>())
    else {
        return Ok(None);
    };
    let Some(source) = config.file_source().downcast_ref::<VortexSource>() else {
        return Ok(None);
    };
    let Some(file_source) = source.try_pushdown_projection(projection.projection_expr())? else {
        return Ok(None);
    };
    let mut config = config.clone();
    config.file_source = file_source;
    Ok(Some(DataSourceExec::from_data_source(config)))
}

fn push_aggregate_arguments(aggregate: &AggregateExec) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
    // Only these modes evaluate the aggregate arguments against the input rows.
    if !matches!(
        aggregate.mode(),
        AggregateMode::Partial | AggregateMode::Single | AggregateMode::SinglePartitioned
    ) {
        return Ok(None);
    }
    let input = aggregate.input();

    // Pass every input column through unchanged, so the group by and filter expressions keep their
    // column indices, and append one column per computed argument.
    let input_schema = input.schema();
    let mut projection: Vec<ProjectionExpr> = input_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, field)| ProjectionExpr {
            expr: Arc::new(Column::new(field.name(), idx)) as Arc<dyn PhysicalExpr>,
            alias: field.name().clone(),
        })
        .collect();

    let mut aggr_exprs = Vec::with_capacity(aggregate.aggr_expr().len());
    for aggr_expr in aggregate.aggr_expr() {
        let args = aggr_expr
            .expressions()
            .into_iter()
            .map(|arg| {
                if arg.downcast_ref::<Column>().is_some() || arg.children().is_empty() {
                    return arg;
                }
                if let Some(existing) = projection.iter().position(|p| p.expr.eq(&arg)) {
                    return Arc::new(Column::new(&projection[existing].alias, existing))
                        as Arc<dyn PhysicalExpr>;
                }
                let alias = format!("__vortex_aggregate_arg_{}", projection.len());
                let column =
                    Arc::new(Column::new(&alias, projection.len())) as Arc<dyn PhysicalExpr>;
                projection.push(ProjectionExpr { expr: arg, alias });
                column
            })
            .collect::<Vec<_>>();
        let order_bys = aggr_expr
            .order_bys()
            .iter()
            .map(|sort| Arc::clone(&sort.expr))
            .collect();
        let Some(rewritten) = aggr_expr.with_new_expressions(args, order_bys) else {
            return Ok(None);
        };
        aggr_exprs.push(Arc::new(rewritten));
    }

    if projection.len() == input_schema.fields().len() {
        return Ok(None);
    }

    let projection = ProjectionExec::try_new(projection, Arc::clone(input))?;
    let Some(new_input) = push_projection_into_scan(input, &projection)? else {
        return Ok(None);
    };

    let new_aggregate = AggregateExec::try_new(
        *aggregate.mode(),
        aggregate.group_expr().clone(),
        aggr_exprs,
        aggregate.filter_expr().to_vec(),
        Arc::clone(&new_input),
        new_input.schema(),
    )?
    .with_limit_options(aggregate.limit_options());

    // The rewrite must not change what the aggregate produces.
    if new_aggregate.schema() != aggregate.schema() {
        return Ok(None);
    }
    Ok(Some(Arc::new(new_aggregate)))
}
