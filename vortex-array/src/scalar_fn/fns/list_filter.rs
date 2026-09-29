// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter each list using a parallel list of boolean predicates.

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::GroupedArray;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFnArray;
use crate::dtype::DType;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::validity::Validity;

/// Select list elements using a parallel boolean list.
///
/// Both inputs must have the same row count and matching list lengths where both outer lists are
/// valid. Null predicate elements are false. If either outer list is null, the output list is null.
/// The result is a variable-length list even when both inputs are fixed-size lists.
#[derive(Clone)]
pub struct ListFilter;

impl ListFilter {
    /// Creates a lazy list filter from parallel value and boolean lists.
    pub fn try_new(input: ArrayRef, predicate: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(ListFilter.bind(EmptyOptions), vec![input, predicate])
    }
}

impl ScalarFnVTable for ListFilter {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.filter");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            1 => ChildName::from("predicate"),
            _ => unreachable!("Invalid child index {child_idx} for list_filter()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let (DType::List(elements, input_nullable)
        | DType::FixedSizeList(elements, _, input_nullable)) = &arg_dtypes[0]
        else {
            vortex_bail!(
                "list_filter() requires a List or FixedSizeList input, got {}",
                arg_dtypes[0]
            );
        };
        let (DType::List(predicate, predicate_nullable)
        | DType::FixedSizeList(predicate, _, predicate_nullable)) = &arg_dtypes[1]
        else {
            vortex_bail!(
                "list_filter() requires a List or FixedSizeList predicate, got {}",
                arg_dtypes[1]
            );
        };
        vortex_ensure!(
            matches!(predicate.as_ref(), DType::Bool(_)),
            "list_filter() requires boolean predicate elements, got {predicate}"
        );
        Ok(DType::List(
            elements.clone(),
            *input_nullable | *predicate_nullable,
        ))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let predicate = args.get(1)?;
        let output_nullable = input.dtype().is_nullable() || predicate.dtype().is_nullable();
        vortex_ensure!(
            input.len() == predicate.len(),
            "list_filter() inputs have different row counts"
        );
        let input = grouped(input.execute::<Canonical>(ctx)?)?;
        let predicate = grouped(predicate.execute::<Canonical>(ctx)?)?;
        let input_ranges = input.group_ranges(ctx)?;
        let predicate_ranges = predicate.group_ranges(ctx)?;
        let input_validity = input.group_validity(ctx)?;
        let predicate_validity = predicate.group_validity(ctx)?;
        let predicate_bits = predicate.elements().clone().null_as_false().execute(ctx)?;

        let mut ranges = Vec::with_capacity(input.len());
        let mut flat = input.elements().len() == predicate.elements().len();
        let mut next_offset = 0;
        for ((input_range, predicate_range), (input_valid, predicate_valid)) in input_ranges
            .iter()
            .zip(predicate_ranges.iter())
            .zip(input_validity.iter().zip(predicate_validity.iter()))
        {
            if input_valid && predicate_valid {
                vortex_ensure!(
                    input_range.1 == predicate_range.1,
                    "list_filter() requires matching list lengths"
                );
            }
            flat &= input_range == predicate_range && input_range.0 == next_offset;
            next_offset = input_range.0 + input_range.1;
            ranges.push((input_range, predicate_range, input_valid && predicate_valid));
        }
        flat &= next_offset == input.elements().len();

        let (elements, offsets) = if flat {
            let ends: Vec<_> = ranges.iter().map(|(range, ..)| range.0 + range.1).collect();
            let counts = predicate_bits.valid_counts_for_indices(&ends);
            let mut offsets = Vec::with_capacity(input.len() + 1);
            offsets.push(0u64);
            offsets.extend(counts.into_iter().map(|count| count as u64));
            (input.elements().filter(predicate_bits)?, offsets)
        } else {
            // List views may share or reorder child ranges. Build an ordered gather only for
            // those cases; the common contiguous representation filters the child once above.
            let mut indices = Vec::<u64>::new();
            let mut offsets = Vec::with_capacity(input.len() + 1);
            offsets.push(0u64);
            for ((input_start, size), (predicate_start, _), valid) in ranges {
                if valid {
                    match predicate_bits.slice(predicate_start..predicate_start + size) {
                        Mask::AllTrue(_) => indices
                            .extend((input_start..input_start + size).map(|index| index as u64)),
                        Mask::AllFalse(_) => {}
                        Mask::Values(values) => values.bit_buffer().for_each_set_index(|offset| {
                            indices.push((input_start + offset) as u64);
                        }),
                    }
                }
                offsets.push(indices.len() as u64);
            }
            let elements = input
                .elements()
                .take(PrimitiveArray::from_iter(indices).into_array())?;
            (elements, offsets)
        };

        let validity = if output_nullable {
            Validity::from_iter(
                input_validity
                    .iter()
                    .zip(predicate_validity.iter())
                    .map(|(a, b)| a && b),
            )
        } else {
            Validity::NonNullable
        };
        Ok(
            ListArray::try_new(elements, Buffer::from_iter(offsets).into_array(), validity)?
                .into_array(),
        )
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }
}

fn grouped(canonical: Canonical) -> VortexResult<GroupedArray> {
    match canonical {
        Canonical::List(list) => Ok(list.into()),
        Canonical::FixedSizeList(list) => Ok(list.into()),
        other => vortex_bail!(
            "list_filter() requires List or FixedSizeList, got {}",
            other.into_array().dtype()
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use prost::Message;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::list_filter;
    use crate::expr::list_sum;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::proto::expr as pb;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::list_filter::ListFilter;
    use crate::validity::Validity;

    #[test]
    fn fixed_size_list_filter_preserves_nulls_and_empty_rows() -> VortexResult<()> {
        let values = FixedSizeListArray::new(
            buffer![1u8, 2, 3, 4, 5, 6].into_array(),
            2,
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
            3,
        )
        .into_array();
        let predicate = FixedSizeListArray::new(
            BoolArray::from_iter([
                Some(true),
                Some(false),
                Some(true),
                Some(true),
                None,
                Some(false),
            ])
            .into_array(),
            2,
            Validity::NonNullable,
            3,
        )
        .into_array();
        let result = ListFilter::try_new(values, predicate)?.into_array();
        let expected = ListArray::try_new(
            PrimitiveArray::from_iter([1u8, 3, 4]).into_array(),
            buffer![0u64, 1, 3, 3].into_array(),
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        )?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn variable_list_filter_checks_each_list_length() -> VortexResult<()> {
        let values = ListArray::try_new(
            buffer![1u8, 2, 3, 4].into_array(),
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let predicate = ListArray::try_new(
            BoolArray::from_iter([true, false, true]).into_array(),
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let result = ListFilter::try_new(values, predicate)?.into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert!(result.execute::<crate::Canonical>(&mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn null_outer_row_does_not_require_matching_child_lengths() -> VortexResult<()> {
        let values = ListArray::try_new(
            buffer![1u8, 2, 3, 4].into_array(),
            buffer![0u32, 2, 4].into_array(),
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        )?
        .into_array();
        let predicate = ListArray::try_new(
            BoolArray::from_iter([true, false, true]).into_array(),
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let result = ListFilter::try_new(values, predicate)?.into_array();
        let expected = ListArray::try_new(
            buffer![1u8].into_array(),
            buffer![0u64, 1, 1].into_array(),
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        )?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn variable_list_filter_selects_elements() -> VortexResult<()> {
        let values = ListArray::try_new(
            buffer![1u8, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 2, 5].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let predicate = ListArray::try_new(
            BoolArray::from_iter([true, false, false, true, true]).into_array(),
            buffer![0u32, 2, 2, 5].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let result = ListFilter::try_new(values, predicate)?.into_array();
        let expected = ListArray::try_new(
            buffer![1u8, 4, 5].into_array(),
            buffer![0u64, 1, 1, 3].into_array(),
            Validity::NonNullable,
        )?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn list_view_filter_handles_reordered_overlapping_ranges() -> VortexResult<()> {
        let values = ListViewArray::try_new(
            buffer![1u8, 2, 3, 4, 5].into_array(),
            buffer![2u32, 0, 1].into_array(),
            buffer![2u32, 2, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let predicate = ListViewArray::try_new(
            BoolArray::from_iter([true, false, true, false, true]).into_array(),
            buffer![0u32, 2, 1].into_array(),
            buffer![2u32, 2, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let result = ListFilter::try_new(values, predicate)?.into_array();
        let expected = ListArray::try_new(
            buffer![3u8, 1, 3].into_array(),
            buffer![0u64, 1, 2, 3].into_array(),
            Validity::NonNullable,
        )?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn sliced_fixed_size_lists_compose_with_list_sum() -> VortexResult<()> {
        let values = FixedSizeListArray::new(
            buffer![1u8, 2, 3, 4, 5, 6].into_array(),
            2,
            Validity::NonNullable,
            3,
        )
        .into_array()
        .slice(1..3)?;
        let predicate = FixedSizeListArray::new(
            BoolArray::from_iter([false, false, true, true, true, false]).into_array(),
            2,
            Validity::NonNullable,
            3,
        )
        .into_array()
        .slice(1..3)?;
        let filtered = ListFilter::try_new(values, predicate)?.into_array();
        let summed = filtered.apply(&list_sum(root()))?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(
            summed,
            PrimitiveArray::from_option_iter::<u64, _>([Some(7), Some(5)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn zero_size_fixed_size_list_filter() -> VortexResult<()> {
        let values = FixedSizeListArray::new(
            PrimitiveArray::from_iter(Vec::<u8>::new()).into_array(),
            0,
            Validity::NonNullable,
            3,
        )
        .into_array();
        let predicate = FixedSizeListArray::new(
            BoolArray::from_iter(Vec::<bool>::new()).into_array(),
            0,
            Validity::NonNullable,
            3,
        )
        .into_array();
        let result = ListFilter::try_new(values, predicate)?.into_array();
        let expected = ListArray::try_new(
            PrimitiveArray::from_iter(Vec::<u8>::new()).into_array(),
            buffer![0u64, 0, 0, 0].into_array(),
            Validity::NonNullable,
        )?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn constant_lists_filter() -> VortexResult<()> {
        let values = ListArray::try_new(
            buffer![1u8, 2].into_array(),
            buffer![0u32, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let predicate = ListArray::try_new(
            BoolArray::from_iter([false, true]).into_array(),
            buffer![0u32, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        let values = ConstantArray::new(values.execute_scalar(0, &mut ctx)?, 3).into_array();
        let predicate = ConstantArray::new(predicate.execute_scalar(0, &mut ctx)?, 3).into_array();
        let result = ListFilter::try_new(values, predicate)?.into_array();
        let expected = ListArray::try_new(
            buffer![2u8, 2, 2].into_array(),
            buffer![0u64, 1, 2, 3].into_array(),
            Validity::NonNullable,
        )?;
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn rejects_non_boolean_predicate() {
        let dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            2,
            Nullability::NonNullable,
        );
        assert!(
            ListFilter
                .return_dtype(&EmptyOptions, &[dtype.clone(), dtype])
                .is_err()
        );
    }

    #[test]
    fn proto_round_trip() -> VortexResult<()> {
        let expr = list_filter(root(), root());
        let proto = expr.serialize_proto()?;
        let decoded = pb::Expr::decode(proto.encode_to_vec().as_slice())?;
        assert_eq!(expr, Expression::from_proto(&decoded, &array_session())?);
        Ok(())
    }
}
