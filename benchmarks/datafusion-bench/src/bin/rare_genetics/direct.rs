// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-memory reductions: both measured paths start with the same packed genotypes.

use std::hint::black_box;
use std::time::Instant;

use anyhow::ensure;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::expr::get_item;
use vortex::array::expr::list_filter;
use vortex::array::expr::list_sum;
use vortex::array::expr::root;
use vortex::array::validity::Validity;
use vortex::encodings::fastlanes::BitPackedData;
use vortex::session::VortexSession;

use super::DirectOperation;
use super::DirectRepresentation;

pub(super) fn run(
    genotypes: ArrayRef,
    samples: usize,
    variants: usize,
    iterations: usize,
    operation: DirectOperation,
    representation: DirectRepresentation,
) -> anyhow::Result<()> {
    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();
    let primitive = genotypes.execute::<PrimitiveArray>(&mut ctx)?;
    if matches!(operation, DirectOperation::FilterSum) {
        let mut selected_count = 0usize;
        let expected = primitive
            .as_slice::<u8>()
            .chunks_exact(samples)
            .enumerate()
            .map(|(row_index, row)| {
                let mut sum = 0u64;
                let mut any_selected = false;
                for (sample_index, &genotype) in row.iter().enumerate() {
                    if random_selected(row_index * samples + sample_index) {
                        sum += u64::from(genotype);
                        selected_count += 1;
                        any_selected = true;
                    }
                }
                any_selected.then_some(sum)
            })
            .collect::<Vec<_>>();
        ensure!(expected.len() == variants);
        let packed = BitPackedData::encode(&primitive.into_array(), 2, &mut ctx)?.into_array();
        return run_single_filter_sum(
            &session,
            &packed,
            samples,
            variants,
            iterations,
            &expected,
            selected_count,
            representation,
        );
    }
    let expected: Vec<u64> = primitive
        .as_slice::<u8>()
        .chunks_exact(samples)
        .map(|row| row.iter().map(|&value| u64::from(value)).sum())
        .collect();
    let expected_cases: Vec<u64> = primitive
        .as_slice::<u8>()
        .chunks_exact(samples)
        .map(|row| row.iter().step_by(2).map(|&value| u64::from(value)).sum())
        .collect();
    let expected_controls: Vec<u64> = expected
        .iter()
        .zip(&expected_cases)
        .map(|(all, cases)| all - cases)
        .collect();
    ensure!(expected.len() == variants);
    let packed = BitPackedData::encode(&primitive.into_array(), 2, &mut ctx)?.into_array();
    let list_size = u32::try_from(samples)?;
    let packed_lists =
        FixedSizeListArray::try_new(packed.clone(), list_size, Validity::NonNullable, variants)?
            .into_array();
    let expression = list_sum(root());

    for decode in [true, false].into_iter().filter(|&decode| {
        matches!(operation, DirectOperation::All | DirectOperation::Sum)
            && selected_representation(representation, decode)
    }) {
        let name = if decode {
            "vortex_list_sum_decode_u8"
        } else {
            "vortex_list_sum_packed_2bit"
        };
        let mut timings = Vec::with_capacity(iterations);
        for iteration in 0..=iterations {
            let mut ctx = session.create_execution_ctx();
            let start = Instant::now();
            let lists = if decode {
                let decoded = packed
                    .clone()
                    .execute::<PrimitiveArray>(&mut ctx)?
                    .into_array();
                FixedSizeListArray::try_new(decoded, list_size, Validity::NonNullable, variants)?
                    .into_array()
            } else {
                packed_lists.clone()
            };
            let sums = lists
                .apply(&expression)?
                .execute::<PrimitiveArray>(&mut ctx)?;
            black_box(&sums);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            ensure!(
                sums.as_slice::<u64>() == expected,
                "{name}: per-row sums differ from generated truth"
            );
            ensure!(
                sums.validity()?
                    .execute_mask(variants, &mut ctx)?
                    .all_true(),
                "{name}: unexpected null sum"
            );
            if iteration > 0 {
                timings.push(elapsed);
                println!(
                    "{{\"mode\":\"{name}\",\"iteration\":{iteration},\"elapsed_ms\":{elapsed:.6}}}"
                );
            }
        }
        timings.sort_by(f64::total_cmp);
        let median = median(&timings);
        let checksum: u64 = expected.iter().sum();
        println!(
            "{{\"mode\":\"{name}\",\"median_ms\":{median:.6},\"genotypes_per_second\":{:.0},\"checksum\":{checksum},\"scope\":\"in_memory_all_samples_per_variant\"}}",
            (samples * variants) as f64 / (median / 1000.0)
        );
    }
    if matches!(
        operation,
        DirectOperation::All | DirectOperation::CaseControl
    ) {
        run_filtered(
            &session,
            &packed,
            samples,
            variants,
            iterations,
            [&expected_cases, &expected_controls],
            representation,
        )?;
    }
    Ok(())
}

fn run_filtered(
    session: &VortexSession,
    packed: &ArrayRef,
    samples: usize,
    variants: usize,
    iterations: usize,
    expected: [&[u64]; 2],
    representation: DirectRepresentation,
) -> anyhow::Result<()> {
    let list_size = u32::try_from(samples)?;
    let masks: Vec<ArrayRef> = [true, false]
        .into_iter()
        .map(|case| {
            let bits = BoolArray::from_iter(
                (0..samples * variants).map(|i| ((i % samples) % 2 == 0) == case),
            );
            FixedSizeListArray::new(
                bits.into_array(),
                list_size,
                Validity::NonNullable,
                variants,
            )
            .into_array()
        })
        .collect();
    let expression = list_sum(list_filter(
        get_item("values", root()),
        get_item("predicate", root()),
    ));
    for decode in [true, false]
        .into_iter()
        .filter(|&decode| selected_representation(representation, decode))
    {
        let name = if decode {
            "vortex_case_control_decode_u8"
        } else {
            "vortex_case_control_packed_2bit"
        };
        let mut timings = Vec::with_capacity(iterations);
        for iteration in 0..=iterations {
            let mut ctx = session.create_execution_ctx();
            let start = Instant::now();
            let elements = if decode {
                packed
                    .clone()
                    .execute::<PrimitiveArray>(&mut ctx)?
                    .into_array()
            } else {
                packed.clone()
            };
            let lists =
                FixedSizeListArray::new(elements, list_size, Validity::NonNullable, variants)
                    .into_array();
            let mut results = Vec::with_capacity(2);
            for mask in &masks {
                let input = StructArray::from_fields(&[
                    ("values", lists.clone()),
                    ("predicate", mask.clone()),
                ])?
                .into_array();
                results.push(
                    input
                        .apply(&expression)?
                        .execute::<PrimitiveArray>(&mut ctx)?,
                );
            }
            black_box(&results);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            for (result, expected) in results.iter().zip(expected) {
                ensure!(
                    result.as_slice::<u64>() == expected,
                    "{name}: case/control per-row mismatch"
                );
                ensure!(
                    result
                        .validity()?
                        .execute_mask(variants, &mut ctx)?
                        .all_true(),
                    "{name}: unexpected null sum"
                );
            }
            if iteration > 0 {
                timings.push(elapsed);
                println!(
                    "{{\"mode\":\"{name}\",\"iteration\":{iteration},\"elapsed_ms\":{elapsed:.6}}}"
                );
            }
        }
        timings.sort_by(f64::total_cmp);
        println!(
            "{{\"mode\":\"{name}\",\"median_ms\":{:.6},\"scope\":\"in_memory_case_control_per_variant\"}}",
            median(&timings)
        );
    }
    Ok(())
}

fn run_single_filter_sum(
    session: &VortexSession,
    packed: &ArrayRef,
    samples: usize,
    variants: usize,
    iterations: usize,
    expected: &[Option<u64>],
    selected_count: usize,
    representation: DirectRepresentation,
) -> anyhow::Result<()> {
    let list_size = u32::try_from(samples)?;
    let predicate = FixedSizeListArray::try_new(
        BoolArray::from_iter((0..samples * variants).map(random_selected)).into_array(),
        list_size,
        Validity::NonNullable,
        variants,
    )?
    .into_array();
    let expression = list_sum(list_filter(
        get_item("values", root()),
        get_item("predicate", root()),
    ));
    let checksum: u64 = expected.iter().flatten().sum();
    let null_rows = expected.iter().filter(|sum| sum.is_none()).count();
    for decode in [true, false]
        .into_iter()
        .filter(|&decode| selected_representation(representation, decode))
    {
        let name = if decode {
            "vortex_filter_sum_decode_u8"
        } else {
            "vortex_filter_sum_packed_2bit"
        };
        let mut timings = Vec::with_capacity(iterations);
        for iteration in 0..=iterations {
            let mut ctx = session.create_execution_ctx();
            let start = Instant::now();
            let elements = if decode {
                packed
                    .clone()
                    .execute::<PrimitiveArray>(&mut ctx)?
                    .into_array()
            } else {
                packed.clone()
            };
            let lists =
                FixedSizeListArray::try_new(elements, list_size, Validity::NonNullable, variants)?
                    .into_array();
            let input = StructArray::from_fields(&[
                ("values", lists),
                ("predicate", predicate.clone()),
            ])?
            .into_array();
            let sums = input
                .apply(&expression)?
                .execute::<PrimitiveArray>(&mut ctx)?;
            black_box(&sums);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let validity = sums.validity()?.execute_mask(variants, &mut ctx)?;
            let values = sums.as_slice::<u64>();
            for (row, expected_sum) in expected.iter().enumerate() {
                match expected_sum {
                    Some(expected_sum) => ensure!(
                        validity.value(row) && values[row] == *expected_sum,
                        "{name}: wrong selected sum for row {row}"
                    ),
                    None => ensure!(
                        !validity.value(row),
                        "{name}: expected a null sum for row {row}"
                    ),
                }
            }
            if iteration > 0 {
                timings.push(elapsed);
                println!(
                    "{{\"mode\":\"{name}\",\"iteration\":{iteration},\"elapsed_ms\":{elapsed:.6}}}"
                );
            }
        }
        timings.sort_by(f64::total_cmp);
        println!(
            "{{\"mode\":\"{name}\",\"median_ms\":{:.6},\"checksum\":{checksum},\"selected_count\":{selected_count},\"null_rows\":{null_rows},\"mask_setup_excluded\":true,\"scope\":\"in_memory_single_random_filter_sum_per_variant\"}}",
            median(&timings)
        );
    }
    Ok(())
}

fn random_selected(index: usize) -> bool {
    let mut value = (index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) & 1) != 0
}

fn selected_representation(representation: DirectRepresentation, decode: bool) -> bool {
    matches!(representation, DirectRepresentation::Both)
        || (decode && matches!(representation, DirectRepresentation::Decoded))
        || (!decode && matches!(representation, DirectRepresentation::Packed))
}

fn median(sorted: &[f64]) -> f64 {
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
}
