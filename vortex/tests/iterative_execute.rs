// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executing a compressed chunk runs one scheduler loop. Encodings hand their children back to
//! the scheduler through `ExecuteSlot` rather than executing them inline, so the trace of a
//! `Shared` chunk shows exactly one `execute_until` to canonical. A `Chunked` stack appends each
//! chunk through its builder, which executes the chunk in a loop of its own, so it shows one
//! loop plus one per chunk.
//!
//! Validity masks are read off already canonical `Bool` arrays through `AnyColumnar` loops that
//! return on their first iteration, so only loops to `AnyCanonical` are counted.

use std::sync::LazyLock;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::SharedArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::assert_arrays_eq;
use vortex::array::validity::Validity;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DecimalDType;
use vortex::error::VortexResult;
use vortex_array::test_harness::trace::trace_op;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::string::FSSTScheme;
use vortex_btrblocks::schemes::string::OnPairScheme;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

const CHUNK_LEN: usize = 2048;
const NUM_CHUNKS: usize = 2;

#[derive(Clone, Copy, Debug)]
enum Column {
    Ints,
    Runs,
    FsstStrings,
    OnPairStrings,
    Decimals,
}

fn canonical_chunk(column: Column, chunk: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(chunk as u64);
    match column {
        Column::Ints => PrimitiveArray::from_option_iter((0..CHUNK_LEN).map(|_| {
            (rng.random_range(0u32..20) != 0).then(|| 1_000_000i64 + rng.random_range(0i64..4096))
        }))
        .into_array(),
        Column::Runs => {
            let mut values = Vec::with_capacity(CHUNK_LEN);
            while values.len() < CHUNK_LEN {
                let value = rng.random_range(0u32..1000);
                let run = rng.random_range(1usize..64).min(CHUNK_LEN - values.len());
                values.extend(std::iter::repeat_n(value, run));
            }
            PrimitiveArray::from_iter(values).into_array()
        }
        Column::FsstStrings | Column::OnPairStrings => {
            VarBinViewArray::from_iter_str((0..CHUNK_LEN).map(|_| {
                format!(
                    "https://example.com/{}/items?page={}&sort={}",
                    ["catalog", "users", "orders", "search"][rng.random_range(0..4)],
                    rng.random_range(0u32..500),
                    ["asc", "desc"][rng.random_range(0..2)],
                )
            }))
            .into_array()
        }
        Column::Decimals => DecimalArray::new(
            (0..CHUNK_LEN)
                .map(|_| i128::from(rng.random_range(-5_000_000i64..5_000_000)))
                .collect(),
            DecimalDType::new(38, 2),
            Validity::NonNullable,
        )
        .into_array(),
    }
}

/// `Chunked` over `Shared` over the compressor's output, and the canonical data it encodes.
fn stack(column: Column) -> VortexResult<(ArrayRef, ArrayRef)> {
    let mut ctx = SESSION.create_execution_ctx();
    let builder = BtrBlocksCompressorBuilder::from_session(&SESSION).unrestricted();
    let builder = match column {
        Column::FsstStrings => builder.exclude_schemes([OnPairScheme.id()]),
        Column::OnPairStrings => builder.exclude_schemes([FSSTScheme.id()]),
        _ => builder,
    };
    let compressor = builder.build();
    let mut chunks = Vec::with_capacity(NUM_CHUNKS);
    let mut canonical = Vec::with_capacity(NUM_CHUNKS);
    for chunk in 0..NUM_CHUNKS {
        let data = canonical_chunk(column, chunk);
        chunks.push(SharedArray::new(compressor.compress(&data, &mut ctx)?).into_array());
        canonical.push(data);
    }
    let dtype = chunks[0].dtype().clone();
    Ok((
        ChunkedArray::try_new(chunks, dtype.clone())?.into_array(),
        ChunkedArray::try_new(canonical, dtype)?.into_array(),
    ))
}

/// The number of `execute_until` loops to canonical that executing `array` runs.
fn canonical_loops(array: ArrayRef, expected: &ArrayRef) -> VortexResult<(usize, String)> {
    let mut ctx = SESSION.create_execution_ctx();
    let traced = trace_op(|| {
        array
            .execute::<Canonical>(&mut ctx)
            .map(IntoArray::into_array)
    })?;
    assert_arrays_eq!(traced.output, expected, &mut ctx);
    let trace = traced.trace.to_string();
    let loops = trace.matches("execute_until target=AnyCanonical").count();
    Ok((loops, trace))
}

fn assert_one_scheduler_loop(column: Column) -> VortexResult<()> {
    let (stack, expected) = stack(column)?;
    let chunks = stack.as_::<vortex::array::arrays::Chunked>().chunks();
    let expected_chunks = expected.as_::<vortex::array::arrays::Chunked>().chunks();
    for (chunk, expected) in chunks.into_iter().zip(expected_chunks) {
        let (loops, trace) = canonical_loops(chunk, &expected)?;
        assert_eq!(
            loops, 1,
            "{column:?}: a Shared chunk should execute in one loop:\n{trace}"
        );
    }
    let (loops, trace) = canonical_loops(stack, &expected)?;
    assert_eq!(
        loops,
        1 + NUM_CHUNKS,
        "{column:?}: a Chunked stack should execute in one loop plus one per appended chunk:\n{trace}"
    );
    Ok(())
}

#[test]
fn for_over_bitpacked_executes_in_one_loop() -> VortexResult<()> {
    assert_one_scheduler_loop(Column::Ints)
}

#[test]
fn runend_executes_in_one_loop() -> VortexResult<()> {
    assert_one_scheduler_loop(Column::Runs)
}

#[test]
fn fsst_executes_in_one_loop() -> VortexResult<()> {
    assert_one_scheduler_loop(Column::FsstStrings)
}

#[test]
fn onpair_executes_in_one_loop() -> VortexResult<()> {
    assert_one_scheduler_loop(Column::OnPairStrings)
}

#[test]
fn decimal_byte_parts_executes_in_one_loop() -> VortexResult<()> {
    assert_one_scheduler_loop(Column::Decimals)
}
