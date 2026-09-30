// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonicalising the array stacks a file scan hands out: `Chunked` over `Shared` over the
//! compressor's output for each column. Cases are `(column, shape)`: `Stack` executes the whole
//! chunked column, `Chunk` one `Shared` chunk of it.
//!
//! Set `VORTEX_BENCH_PRINT_TREE=1` to print each column's encoding tree before the runs.

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
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
use vortex::array::validity::Validity;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DecimalDType;
use vortex::error::VortexExpect;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::string::FSSTScheme;
use vortex_btrblocks::schemes::string::OnPairScheme;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    if std::env::var_os("VORTEX_BENCH_PRINT_TREE").is_some() {
        for column in COLUMNS {
            eprintln!(
                "{column}:\n{}",
                stack(*column).display_tree_encodings_only()
            );
        }
    }
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

const CHUNK_LEN: usize = 8192;
const NUM_CHUNKS: usize = 8;

#[derive(Clone, Copy, Debug)]
enum Column {
    /// Nullable `i64` clustered around one value: FoR over BitPacked.
    Ints,
    /// `u32` in long runs: RunEnd.
    Runs,
    /// URL-like strings: FSST.
    FsstStrings,
    /// The same strings compressed with OnPair.
    OnPairStrings,
    /// `decimal(38, 2)` values that fit in a narrow MSP: DecimalByteParts.
    Decimals,
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

const COLUMNS: &[Column] = &[
    Column::Ints,
    Column::Runs,
    Column::FsstStrings,
    Column::OnPairStrings,
    Column::Decimals,
];

#[derive(Clone, Copy, Debug)]
enum Shape {
    /// The whole chunked column.
    Stack,
    /// One `Shared` chunk of it.
    Chunk,
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

const CASES: &[(Column, Shape)] = &[
    (Column::Ints, Shape::Stack),
    (Column::Ints, Shape::Chunk),
    (Column::Runs, Shape::Stack),
    (Column::Runs, Shape::Chunk),
    (Column::FsstStrings, Shape::Stack),
    (Column::FsstStrings, Shape::Chunk),
    (Column::OnPairStrings, Shape::Stack),
    (Column::OnPairStrings, Shape::Chunk),
    (Column::Decimals, Shape::Stack),
    (Column::Decimals, Shape::Chunk),
];

/// One chunk of the column's canonical data, `chunk` picking the seed.
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

/// The compressor's output for each chunk, each wrapped in `Shared` as a scan hands it out.
fn chunks(column: Column) -> Vec<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let builder = BtrBlocksCompressorBuilder::from_session(&SESSION).unrestricted();
    let builder = match column {
        Column::FsstStrings => builder.exclude_schemes([OnPairScheme.id()]),
        Column::OnPairStrings => builder.exclude_schemes([FSSTScheme.id()]),
        _ => builder,
    };
    let compressor = builder.build();
    (0..NUM_CHUNKS)
        .map(|chunk| {
            let canonical = canonical_chunk(column, chunk);
            let compressed = compressor
                .compress(&canonical, &mut ctx)
                .vortex_expect("compress chunk");
            SharedArray::new(compressed).into_array()
        })
        .collect()
}

/// `Chunked` over the column's chunks.
fn stack(column: Column) -> ArrayRef {
    let chunks = chunks(column);
    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype)
        .vortex_expect("chunked stack")
        .into_array()
}

#[divan::bench(args = CASES)]
fn execute_stack(bencher: Bencher, (column, shape): (Column, Shape)) {
    let array = match shape {
        Shape::Stack => stack(column),
        Shape::Chunk => chunks(column).swap_remove(0),
    };
    bencher
        // `Shared` caches its result, so each iteration executes a fresh copy of the tree
        // rather than the cache.
        .with_inputs(|| (SESSION.create_execution_ctx(), fresh_copy(&array)))
        .bench_values(|(mut ctx, array)| {
            array
                .execute::<Canonical>(&mut ctx)
                .vortex_expect("execute stack")
        });
}

/// A copy of `array` whose `Shared` nodes have empty caches.
fn fresh_copy(array: &ArrayRef) -> ArrayRef {
    if let Some(shared) = array.as_opt::<vortex::array::arrays::Shared>() {
        use vortex::array::arrays::shared::SharedArraySlotsExt;
        return SharedArray::new(fresh_copy(shared.source())).into_array();
    }
    if let Some(chunked) = array.as_opt::<vortex::array::arrays::Chunked>() {
        use vortex::array::arrays::chunked::ChunkedArrayExt;
        let chunks = chunked.iter_chunks().map(fresh_copy).collect::<Vec<_>>();
        return ChunkedArray::try_new(chunks, array.dtype().clone())
            .vortex_expect("chunked copy")
            .into_array();
    }
    array.clone()
}
