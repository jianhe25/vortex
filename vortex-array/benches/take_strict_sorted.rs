// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks take with strictly sorted indices, with and without the cached
//! [`Stat::IsStrictSorted`] statistic that lets a contiguous run of indices execute as a slice.
//!
//! Each case takes the same indices from the same values and differs only in whether the indices
//! carry the statistic (`strict_sorted`) or not (`plain`):
//!
//! - `run`: the contiguous middle half, which becomes a zero-copy slice.
//! - `sparse`: every 64th position, and `alternate`: every other position. These have gaps, so
//!   they still take; they measure the cost of checking for a run.
//!
//! Results are executed to [`RecursiveCanonical`] so lazy results pay for materialization.

#![expect(clippy::unwrap_used)]

use std::fmt;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_buffer::Buffer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const LEN: usize = 65_536;
const CHUNK_LEN: usize = 1_024;

#[derive(Clone, Copy)]
enum Pattern {
    Sparse,
    Alternate,
    Run,
}

#[derive(Clone, Copy)]
struct Case {
    pattern: Pattern,
    strict_sorted: bool,
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pattern = match self.pattern {
            Pattern::Sparse => "sparse",
            Pattern::Alternate => "alternate",
            Pattern::Run => "run",
        };
        let mode = if self.strict_sorted {
            "strict_sorted"
        } else {
            "plain"
        };
        write!(f, "{pattern}/{mode}")
    }
}

const CASES: [Case; 6] = {
    const fn case(pattern: Pattern, strict_sorted: bool) -> Case {
        Case {
            pattern,
            strict_sorted,
        }
    }
    [
        case(Pattern::Sparse, false),
        case(Pattern::Sparse, true),
        case(Pattern::Alternate, false),
        case(Pattern::Alternate, true),
        case(Pattern::Run, false),
        case(Pattern::Run, true),
    ]
};

fn indices(case: Case) -> ArrayRef {
    let len = u32::try_from(LEN).unwrap();
    let indices: Buffer<u32> = match case.pattern {
        Pattern::Sparse => (0..len).step_by(64).collect(),
        Pattern::Alternate => (0..len).step_by(2).collect(),
        Pattern::Run => (len / 4..3 * len / 4).collect(),
    };
    let indices = indices.into_array();
    if case.strict_sorted {
        indices
            .statistics()
            .set(Stat::IsStrictSorted, Precision::exact(true));
    }
    indices
}

fn primitive_values() -> ArrayRef {
    (0..LEN as i64).collect::<Buffer<_>>().into_array()
}

fn chunked_values() -> ArrayRef {
    ChunkedArray::from_iter(
        (0..LEN / CHUNK_LEN)
            .map(|c| ((c * CHUNK_LEN) as i64..((c + 1) * CHUNK_LEN) as i64).collect::<Buffer<_>>())
            .map(IntoArray::into_array),
    )
    .into_array()
}

fn utf8_values() -> ArrayRef {
    VarBinViewArray::from_iter_str((0..LEN).map(|i| format!("a-longer-string-value-{i}")))
        .into_array()
}

fn struct_values() -> ArrayRef {
    StructArray::from_fields(&[
        ("a", primitive_values()),
        (
            "b",
            (0..LEN)
                .map(|i| u8::try_from(i % 256).unwrap())
                .collect::<Buffer<_>>()
                .into_array(),
        ),
        ("c", utf8_values()),
    ])
    .unwrap()
    .into_array()
}

fn bench_take(bencher: Bencher, values: ArrayRef, case: Case) {
    let session = array_session();
    let indices = indices(case);

    bencher
        .counter(ItemsCount::new(indices.len()))
        .with_inputs(|| (&values, &indices, session.create_execution_ctx()))
        .bench_refs(|(values, indices, ctx)| {
            values
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = CASES)]
fn primitive(bencher: Bencher, case: Case) {
    bench_take(bencher, primitive_values(), case);
}

#[divan::bench(args = CASES)]
fn chunked(bencher: Bencher, case: Case) {
    bench_take(bencher, chunked_values(), case);
}

#[divan::bench(args = CASES)]
fn utf8(bencher: Bencher, case: Case) {
    bench_take(bencher, utf8_values(), case);
}

#[divan::bench(args = CASES)]
fn struct_(bencher: Bencher, case: Case) {
    bench_take(bencher, struct_values(), case);
}
