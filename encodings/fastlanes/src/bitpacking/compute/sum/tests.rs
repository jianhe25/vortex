// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::GroupedArray;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::aggregate_fn::fns::sum_v2::sum_v2;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::aggregate_fn::kernels::DynGroupedAggregateKernel;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::expr::list_sum;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::list_filter::ListFilter;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::BitPackedSumKernel;
use super::FilteredBitPackedSumKernel;
use super::interval_mask;
use super::packed_position;
use super::packed_selection_mask;
use super::sum_masked_block;
use super::sum_range;
use super::sum_selected_block;
use crate::BitPacked;
use crate::BitPackedData;

#[test]
fn packed_sum_handles_slices_and_padding() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values: Vec<u8> = (0..31_000).map(|idx| (idx % 3) as u8).collect();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter(values.clone()).into_array(),
        2,
        &mut ctx,
    )?;
    for (start, end) in [
        (0, 0),
        (0, 1),
        (0, 1023),
        (0, 1024),
        (0, 1025),
        (0, 10_000),
        (37, 10_037),
        (1000, 11_000),
        (2048, 12_048),
        (30_999, 31_000),
    ] {
        let sliced = encoded.as_array().clone().slice(start..end)?;
        if start != end {
            assert!(sliced.is::<BitPacked>());
        }
        let expected: u64 = values[start..end]
            .iter()
            .map(|&value| u64::from(value))
            .sum();
        assert_eq!(
            sum(&sliced, &mut ctx)?.as_primitive().as_::<u64>(),
            Some(expected)
        );
        let sql_sum = sum_v2(&sliced, &mut ctx)?;
        if start == end {
            assert!(sql_sum.is_null());
        } else {
            assert_eq!(sql_sum.as_primitive().as_::<u64>(), Some(expected));
        }
    }
    Ok(())
}

#[test]
fn packed_sum_matches_boundary_and_random_ranges() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let mut value_seed = 19u64;
    let values: Vec<u8> = (0..5000)
        .map(|_| {
            value_seed = value_seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (value_seed >> 39) as u8 & 3
        })
        .collect();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter(values.clone()).into_array(),
        2,
        &mut ctx,
    )?;
    let sliced = encoded.as_array().clone().slice(123..4123)?;
    let packed = sliced
        .as_opt::<BitPacked>()
        .expect("slice must remain packed");
    for start in (0..=4000).step_by(16) {
        for size in [0, 1, 7, 16, 127, 1024] {
            let end = (start + size).min(4000);
            let expected: u64 = values[123 + start..123 + end]
                .iter()
                .map(|&value| u64::from(value))
                .sum();
            assert_eq!(sum_range(&packed, start, end - start), expected);
        }
    }
    let mut seed = 1729usize;
    for _ in 0..1000 {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let start = seed % 4001;
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let end = start + seed % (4001 - start);
        let expected: u64 = values[123 + start..123 + end]
            .iter()
            .map(|&value| u64::from(value))
            .sum();
        assert_eq!(sum_range(&packed, start, end - start), expected);
    }
    Ok(())
}

#[test]
fn packed_position_matches_fastlanes_values() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..1024).map(|idx| (idx % 3) as u8));
    let encoded = BitPackedData::encode(&values.into_array(), 2, &mut ctx)?;
    let block = &encoded.packed().as_host()[..256];
    for index in 0..1024 {
        let (word, shift) = packed_position(index);
        let bytes: [u8; 8] = block[word * 8..word * 8 + 8].try_into().unwrap();
        let direct = (u64::from_le_bytes(bytes) >> shift) as u8 & 3;
        // SAFETY: block contains 1024 two-bit values and index is in 0..1024.
        let fastlanes = unsafe { u8::unchecked_unpack_single(2, block, index) };
        assert_eq!(direct, fastlanes, "index={index}");
    }
    Ok(())
}

#[test]
fn packed_sum_handles_nullable_all_valid_and_falls_back_for_nulls_and_patches() -> VortexResult<()>
{
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let all_valid =
        PrimitiveArray::from_option_iter((0..3000).map(|idx| Some((idx % 3) as u8))).into_array();
    let encoded = BitPackedData::encode(&all_valid, 2, &mut ctx)?.into_array();
    assert_eq!(
        sum_v2(&encoded, &mut ctx)?.as_primitive().as_::<u64>(),
        Some(3000)
    );

    let nullable = PrimitiveArray::from_option_iter((0..3000).map(|idx| {
        if idx % 7 == 0 {
            None
        } else {
            Some((idx % 3) as u8)
        }
    }))
    .into_array();
    let null_encoded = BitPackedData::encode(&nullable, 2, &mut ctx)?.into_array();
    let sum_aggregate = Sum.bind(NumericalAggregateOpts::default());
    assert!(
        BitPackedSumKernel
            .aggregate(&sum_aggregate, &null_encoded, &mut ctx)?
            .is_none()
    );
    assert_eq!(
        sum_v2(&null_encoded, &mut ctx)?,
        sum_v2(&nullable, &mut ctx)?
    );

    let patched = PrimitiveArray::from_iter(
        (0..3000).map(|idx| if idx == 127 { 9u8 } else { (idx % 3) as u8 }),
    )
    .into_array();
    let patch_encoded = BitPackedData::encode(&patched, 2, &mut ctx)?.into_array();
    assert!(
        BitPackedSumKernel
            .aggregate(&sum_aggregate, &patch_encoded, &mut ctx)?
            .is_none()
    );
    assert_eq!(
        sum_v2(&patch_encoded, &mut ctx)?,
        sum_v2(&patched, &mut ctx)?
    );
    Ok(())
}

#[test]
fn grouped_packed_sum_preserves_groups() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values: Vec<u8> = (0..30_000).map(|idx| (idx % 3) as u8).collect();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter(values.clone()).into_array(),
        2,
        &mut ctx,
    )?;
    let groups =
        FixedSizeListArray::try_new(encoded.into_array(), 10_000, Validity::NonNullable, 3)?;
    let aggregate = SumV2.bind(NumericalAggregateOpts::default());
    let partial = BitPackedSumKernel
        .grouped_aggregate(&aggregate, &GroupedArray::from(groups), &mut ctx)?
        .expect("two-bit nonnullable input must use the packed kernel")
        .downcast::<Struct>();
    let sums = partial
        .unmasked_field_by_name("sum")?
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let expected = PrimitiveArray::from_iter(
        values
            .chunks_exact(10_000)
            .map(|group| group.iter().map(|&value| u64::from(value)).sum::<u64>()),
    );
    vortex_array::assert_arrays_eq!(sums, expected, &mut ctx);
    Ok(())
}

#[test]
fn fixed_size_list_sum_uses_packed_elements() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values: Vec<u8> = (0..30_000).map(|idx| (idx % 3) as u8).collect();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter(values.clone()).into_array(),
        2,
        &mut ctx,
    )?;
    let groups =
        FixedSizeListArray::try_new(encoded.into_array(), 10_000, Validity::NonNullable, 3)?
            .into_array();
    let result = groups
        .apply(&list_sum(root()))?
        .execute::<PrimitiveArray>(&mut ctx)?;
    let expected = PrimitiveArray::from_option_iter(
        values
            .chunks_exact(10_000)
            .map(|group| Some(group.iter().map(|&value| u64::from(value)).sum::<u64>())),
    );
    vortex_array::assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn fixed_size_list_filter_then_sum_matches_reference() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values: Vec<u8> = (0..30_000).map(|idx| (idx % 3) as u8).collect();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter(values.clone()).into_array(),
        2,
        &mut ctx,
    )?;
    let groups =
        FixedSizeListArray::try_new(encoded.into_array(), 10_000, Validity::NonNullable, 3)?
            .into_array();
    let mut mask_seed = 23u64;
    let predicate_values: Vec<bool> = (0..30_000)
        .map(|_| {
            mask_seed = mask_seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            mask_seed >> 63 != 0
        })
        .collect();
    let predicates = FixedSizeListArray::try_new(
        BoolArray::from_iter(predicate_values.iter().copied()).into_array(),
        10_000,
        Validity::NonNullable,
        3,
    )?
    .into_array();
    let filtered = ListFilter::try_new(groups, predicates)?.into_array();
    let result = filtered
        .apply(&list_sum(root()))?
        .execute::<PrimitiveArray>(&mut ctx)?;
    let expected = PrimitiveArray::from_option_iter(
        values
            .chunks_exact(10_000)
            .zip(predicate_values.chunks_exact(10_000))
            .map(|(group, selected)| {
                Some(
                    group
                        .iter()
                        .zip(selected)
                        .filter(|(_, keep)| **keep)
                        .map(|(&value, _)| u64::from(value))
                        .sum::<u64>(),
                )
            }),
    );
    vortex_array::assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn fixed_size_list_filter_sum_preserves_nulls_and_empty_selection() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values =
        PrimitiveArray::from_iter([0u8, 0, 0, 0, 1, 2, 3, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
            .into_array();
    let packed = BitPackedData::encode(&values, 2, &mut ctx)?.into_array();
    let groups = FixedSizeListArray::try_new(
        packed,
        4,
        Validity::Array(BoolArray::from_iter([true, true, false, true, true]).into_array()),
        5,
    )?
    .into_array();
    let predicates = FixedSizeListArray::try_new(
        BoolArray::from_iter([
            Some(true),
            None,
            Some(false),
            None,
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            None,
            Some(true),
        ])
        .into_array(),
        4,
        Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array()),
        5,
    )?
    .into_array();
    let result = ListFilter::try_new(groups, predicates)?
        .into_array()
        .apply(&list_sum(root()))?
        .execute::<PrimitiveArray>(&mut ctx)?;
    let expected = PrimitiveArray::from_option_iter([Some(0u64), None, None, None, Some(6)]);
    vortex_array::assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn fused_filtered_sum_handles_empty_null_and_trailing_groups() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values: Vec<u8> = (0..10_003).map(|idx| ((idx * 7 + 1) % 4) as u8).collect();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter(values.clone()).into_array(),
        2,
        &mut ctx,
    )?;
    let mut seed = 0x0123_4567_89ab_cdefu64;
    let indices: Vec<usize> = (0..values.len())
        .filter(|&idx| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (seed >> 63) != 0 || idx == values.len() - 1
        })
        .collect();
    let selected: Vec<u8> = indices.iter().map(|&idx| values[idx]).collect();
    let filtered = FilterArray::new(
        encoded.into_array(),
        Mask::from_indices(values.len(), indices),
    )
    .into_array();
    let first = selected.len() / 3;
    let second = selected.len() / 3;
    let third = selected.len() - first - second;
    let groups = ListViewArray::try_new(
        filtered,
        buffer![0u64, 0, first as u64, (first + second) as u64].into_array(),
        buffer![0u64, first as u64, second as u64, third as u64].into_array(),
        Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array()),
    )?;
    let aggregate = SumV2.bind(NumericalAggregateOpts::default());
    let partial = FilteredBitPackedSumKernel
        .grouped_aggregate(&aggregate, &GroupedArray::from(groups), &mut ctx)?
        .expect("eligible filtered packed values must use the fused kernel")
        .downcast::<Struct>();
    let sums = partial
        .unmasked_field_by_name("sum")?
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let expected = PrimitiveArray::from_iter([
        0u64,
        0,
        selected[first..first + second]
            .iter()
            .map(|&value| u64::from(value))
            .sum(),
        selected[first + second..]
            .iter()
            .map(|&value| u64::from(value))
            .sum(),
    ]);
    vortex_array::assert_arrays_eq!(sums, expected, &mut ctx);
    assert_eq!(
        partial
            .validity()?
            .execute_mask(4, &mut ctx)?
            .iter()
            .collect::<Vec<_>>(),
        vec![true, false, true, true]
    );
    Ok(())
}

#[test]
fn fused_filtered_sum_defers_unsupported_shapes() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter((0..2048).map(|idx| (idx % 4) as u8)).into_array(),
        2,
        &mut ctx,
    )?
    .into_array();
    let aggregate = SumV2.bind(NumericalAggregateOpts::default());

    let all_false = FilterArray::new(encoded.clone(), Mask::AllFalse(2048)).into_array();
    let empty_groups = FixedSizeListArray::try_new(all_false, 0, Validity::NonNullable, 3)?;
    assert!(
        FilteredBitPackedSumKernel
            .grouped_aggregate(&aggregate, &GroupedArray::from(empty_groups), &mut ctx)?
            .is_none()
    );

    let selected = FilterArray::new(
        encoded.clone(),
        Mask::from_indices(2048, (0..2048).step_by(2)),
    )
    .into_array();
    let irregular = ListViewArray::try_new(
        selected,
        buffer![0u64, 5].into_array(),
        buffer![2u64, 2].into_array(),
        Validity::NonNullable,
    )?;
    assert!(
        FilteredBitPackedSumKernel
            .grouped_aggregate(&aggregate, &GroupedArray::from(irregular), &mut ctx)?
            .is_none()
    );

    let sliced = encoded.slice(17..2017)?;
    let selected =
        FilterArray::new(sliced, Mask::from_indices(2000, (0..2000).step_by(2))).into_array();
    let groups = FixedSizeListArray::try_new(selected, 500, Validity::NonNullable, 2)?;
    assert!(
        FilteredBitPackedSumKernel
            .grouped_aggregate(&aggregate, &GroupedArray::from(groups), &mut ctx)?
            .is_none()
    );
    Ok(())
}

#[test]
fn early_filtered_sum_handles_sliced_ranges_and_bit_offset_mask() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values: Vec<u8> = (0..5000).map(|idx| ((idx * 7 + 1) % 4) as u8).collect();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter(values.clone()).into_array(),
        2,
        &mut ctx,
    )?;
    let sliced = encoded.as_array().clone().slice(123..4123)?;
    let groups = FixedSizeListArray::try_new(
        sliced,
        1000,
        Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array()),
        4,
    )?;
    let selected = Mask::from_indices(
        4013,
        (13..4013).filter(|index| index % 7 == 0 || index % 11 == 0),
    )
    .slice(13..4013);
    let aggregate = SumV2.bind(NumericalAggregateOpts::default());
    let partial = BitPackedSumKernel
        .filtered_grouped_aggregate(&aggregate, &GroupedArray::from(groups), &selected, &mut ctx)?
        .expect("sliced two-bit array should use the early fused kernel")
        .downcast::<Struct>();
    let sums = partial
        .unmasked_field_by_name("sum")?
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let expected = PrimitiveArray::from_iter((0..4).map(|group| {
        if group == 1 {
            return 0u64;
        }
        (group * 1000..(group + 1) * 1000)
            .filter(|&index| selected.value(index))
            .map(|index| u64::from(values[123 + index]))
            .sum::<u64>()
    }));
    vortex_array::assert_arrays_eq!(sums, expected, &mut ctx);
    Ok(())
}

#[test]
fn early_filtered_sum_distinguishes_zero_from_empty() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter([0u8; 2048]).into_array(),
        2,
        &mut ctx,
    )?;
    let groups = FixedSizeListArray::try_new(encoded.into_array(), 1024, Validity::NonNullable, 2)?;
    let selected = Mask::from_indices(2048, [17, 100, 1023]);
    let aggregate = SumV2.bind(NumericalAggregateOpts::default());
    let partial = BitPackedSumKernel
        .filtered_grouped_aggregate(&aggregate, &GroupedArray::from(groups), &selected, &mut ctx)?
        .expect("two-bit array should use the early fused kernel")
        .downcast::<Struct>();
    let sums = partial
        .unmasked_field_by_name("sum")?
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    vortex_array::assert_arrays_eq!(sums, PrimitiveArray::from_iter([0u64, 0]), &mut ctx);
    let empty = partial
        .unmasked_field_by_name("is_empty")?
        .clone()
        .execute::<BoolArray>(&mut ctx)?;
    vortex_array::assert_arrays_eq!(empty, BoolArray::from_iter([false, true]), &mut ctx);
    Ok(())
}

#[test]
fn selected_block_matches_portable_reference() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let values: Vec<u8> = (0..1024)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (seed >> 32) as u8 & 3
        })
        .collect();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter(values.clone()).into_array(),
        2,
        &mut ctx,
    )?;
    let block = &encoded.packed().as_host()[..256];
    for trial in 0..80 {
        let mut selection = [0u8; 128];
        for byte in &mut selection {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *byte = if trial == 0 { 0 } else { (seed >> 32) as u8 };
        }
        let start = (trial * 37) % 1024;
        let end = (start + (trial * 113) % (1025 - start)).min(1024);
        let interval = interval_mask(start, end);
        let portable = sum_masked_block(block, &packed_selection_mask(&selection), Some(&interval));
        let direct = sum_selected_block(block, &selection, Some(&interval));
        assert_eq!(direct, portable, "trial={trial}, range={start}..{end}");
        let expected: u64 = (start..end)
            .filter(|&index| selection[index / 8] & (1 << (index % 8)) != 0)
            .map(|index| u64::from(values[index]))
            .sum();
        assert_eq!(direct.0, expected, "trial={trial}, range={start}..{end}");
        assert_eq!(
            direct.1,
            (start..end).any(|index| selection[index / 8] & (1 << (index % 8)) != 0)
        );
    }
    Ok(())
}

#[test]
fn selected_block_maps_every_predicate_bit_to_packed_value() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let blocks: Vec<_> = (0..5)
        .map(|digit| {
            let values = PrimitiveArray::from_iter(
                (0..1024).map(|index| ((index >> (digit * 2)) & 3) as u8),
            );
            BitPackedData::encode(&values.into_array(), 2, &mut ctx)
        })
        .collect::<VortexResult<_>>()?;
    let mut selection = [0u8; 128];
    for index in 0..1024 {
        selection[index / 8] = 1 << (index % 8);
        let mut recovered = 0usize;
        for (digit, encoded) in blocks.iter().enumerate() {
            let block = &encoded.packed().as_host()[..256];
            let (value, selected) = sum_selected_block(block, &selection, None);
            assert!(selected, "index={index}, digit={digit}");
            recovered |= (value as usize) << (digit * 2);
        }
        assert_eq!(recovered, index);
        selection[index / 8] = 0;
    }

    let block = &blocks[0].packed().as_host()[..256];
    assert_eq!(sum_selected_block(block, &[0; 128], None), (0, false));
    let total = (0..1024).map(|index| (index & 3) as u64).sum::<u64>();
    assert_eq!(sum_selected_block(block, &[0xff; 128], None), (total, true));
    Ok(())
}

#[test]
fn early_filtered_sum_all_false_and_overlapping_ranges() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values: Vec<u8> = (0..2048).map(|index| (index % 4) as u8).collect();
    let encoded = BitPackedData::encode(
        &PrimitiveArray::from_iter(values.clone()).into_array(),
        2,
        &mut ctx,
    )?;
    let groups = ListViewArray::try_new(
        encoded.into_array(),
        buffer![1500u64, 0, 100].into_array(),
        buffer![400u64, 1000, 300].into_array(),
        Validity::NonNullable,
    )?;
    let aggregate = SumV2.bind(NumericalAggregateOpts::default());
    let grouped = GroupedArray::from(groups);
    let all_false = BitPackedSumKernel
        .filtered_grouped_aggregate(&aggregate, &grouped, &Mask::AllFalse(2048), &mut ctx)?
        .expect("all-false predicate should use early packed kernel")
        .downcast::<Struct>();
    let empty = all_false
        .unmasked_field_by_name("is_empty")?
        .clone()
        .execute::<BoolArray>(&mut ctx)?;
    vortex_array::assert_arrays_eq!(empty, BoolArray::from_iter([true, true, true]), &mut ctx);

    let predicate = Mask::from_indices(2048, (0..2048).filter(|&index| index % 7 < 3));
    let partial = BitPackedSumKernel
        .filtered_grouped_aggregate(&aggregate, &grouped, &predicate, &mut ctx)?
        .expect("overlapping ranges should use early packed kernel")
        .downcast::<Struct>();
    let sums = partial
        .unmasked_field_by_name("sum")?
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let expected =
        PrimitiveArray::from_iter([(1500, 400), (0, 1000), (100, 300)].map(|(start, len)| {
            (start..start + len)
                .filter(|&index| predicate.value(index))
                .map(|index| u64::from(values[index]))
                .sum::<u64>()
        }));
    vortex_array::assert_arrays_eq!(sums, expected, &mut ctx);
    Ok(())
}
