// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for comparisons against a constant that cached statistics can short-circuit.
//!
//! Every case compares the same sorted `i64` column against a constant, and differs only in the
//! statistics cached on the column:
//!
//! - `none`: no statistics, so the comparison is a linear scan.
//! - `min_max`: exact min and max, which answer the comparison without a scan when the constant
//!   falls outside the column's range.
//! - `sorted`: sortedness, which replaces the scan with a binary search and range fills.
//!
//! Statistics are computed before timing starts. The result is executed to [`Canonical`], so a
//! constant result still pays for materializing its bitmap.

#![expect(clippy::unwrap_used)]

use std::fmt;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::Buffer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const SIZES: [usize; 2] = [8_192, 65_536];

#[derive(Clone, Copy, Debug)]
enum Stats {
    None,
    MinMax,
    Sorted,
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Stats::None => "none",
            Stats::MinMax => "min_max",
            Stats::Sorted => "sorted",
        })
    }
}

const STATS: [Stats; 3] = [Stats::None, Stats::MinMax, Stats::Sorted];

/// A sorted column where every value repeats four times: `0, 0, 0, 0, 1, 1, 1, 1, ...`.
fn sorted_column(len: usize, stats: Stats) -> ArrayRef {
    let array = (0..len as i64)
        .map(|i| i / 4)
        .collect::<Buffer<_>>()
        .into_array();
    let cached: &[Stat] = match stats {
        Stats::None => &[],
        Stats::MinMax => &[Stat::Min, Stat::Max],
        Stats::Sorted => &[Stat::IsSorted],
    };
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    array.statistics().compute_all(cached, &mut ctx).unwrap();
    array
}

fn bench_compare(bencher: Bencher, len: usize, stats: Stats, constant: i64, op: Operator) {
    let session = vortex_array::array_session();
    let array = sorted_column(len, stats);
    let constant = ConstantArray::new(constant, len).into_array();

    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| (&array, &constant, session.create_execution_ctx()))
        .bench_refs(|(array, constant, ctx)| {
            array
                .clone()
                .binary((*constant).clone(), op)
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

/// The constant splits the column in half: min/max cannot decide it, sortedness can.
#[divan::bench(consts = SIZES, args = STATS)]
fn gte_in_range<const N: usize>(bencher: Bencher, stats: Stats) {
    bench_compare(bencher, N, stats, (N / 8) as i64, Operator::Gte);
}

/// The constant is above the column's max: both min/max and sortedness decide it.
#[divan::bench(consts = SIZES, args = STATS)]
fn gte_out_of_range<const N: usize>(bencher: Bencher, stats: Stats) {
    bench_compare(bencher, N, stats, N as i64, Operator::Gte);
}

/// Equality selects one run of four values in the middle of the column.
#[divan::bench(consts = SIZES, args = STATS)]
fn eq_in_range<const N: usize>(bencher: Bencher, stats: Stats) {
    bench_compare(bencher, N, stats, (N / 8) as i64, Operator::Eq);
}

/// The constant is below the column's min, so every row is not equal to it.
#[divan::bench(consts = SIZES, args = STATS)]
fn not_eq_out_of_range<const N: usize>(bencher: Bencher, stats: Stats) {
    bench_compare(bencher, N, stats, -1, Operator::NotEq);
}
