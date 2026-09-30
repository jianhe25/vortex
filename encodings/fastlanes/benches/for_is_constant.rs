// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks checking whether a BitPacked FoR array is constant, three ways:
//!
//! - `unblocked_kernel`: a single reference, through the FoR is_constant kernel.
//! - `blocked_kernel`: a reference per 1024-element chunk, through the FoR is_constant kernel.
//! - `blocked_decode`: a reference per chunk, decoded first and then checked, as happens without
//!   a kernel.
//!
//! `drifting` values climb by a million every chunk, so the references differ. `uniform` values
//! share one range, so every chunk has the same reference. `constant` values are all equal.
//!
//! The result is cached in the array's statistics, so every iteration builds a fresh array.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench for_is_constant`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArray;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

const LEN: u32 = 128 * 1024;

const DATASETS: &[&str] = &["drifting", "uniform", "constant"];

fn values(dataset: &str) -> Buffer<u32> {
    (0..LEN)
        .map(|i| match dataset {
            "drifting" => (i / 1024) * 1_000_000 + (i * 7919) % 100,
            "uniform" => 1000 + (i * 7919) % 100,
            _ => 1000,
        })
        .collect()
}

/// Bit-pack `for_array`'s encoded child at the width of its largest value.
fn bitpack(for_array: &FoRArray) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = for_array
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let max = encoded.as_slice::<u32>().iter().copied().max().unwrap();
    // BitPacking needs a non-zero width.
    let bit_width = u8::try_from(u32::BITS - max.leading_zeros())
        .unwrap()
        .max(1);
    BitPacked::encode(&encoded.into_array(), bit_width, &mut ctx)
        .unwrap()
        .into_array()
}

fn unblocked(dataset: &str) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(values(dataset), Validity::NonNullable);
    let for_array = FoR::encode(array, &mut ctx).unwrap();
    let reference = for_array.constant_reference().unwrap();
    FoR::try_new(bitpack(&for_array), reference)
        .unwrap()
        .into_array()
}

fn blocked(dataset: &str) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::new(values(dataset), Validity::NonNullable);
    let for_array = FoR::encode_chunked(array, &mut ctx).unwrap();
    FoR::try_new_chunked(bitpack(&for_array), for_array.references().clone(), 0)
        .unwrap()
        .into_array()
}

fn run(bencher: Bencher, make: fn(&str) -> ArrayRef, dataset: &str, decode_first: bool) {
    bencher
        .counter(ItemsCount::new(LEN as usize))
        .with_inputs(|| (make(dataset), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            let array = if decode_first {
                array
                    .execute::<PrimitiveArray>(&mut ctx)
                    .unwrap()
                    .into_array()
            } else {
                array
            };
            is_constant(&array, &mut ctx).unwrap()
        });
}

#[divan::bench(args = DATASETS)]
fn unblocked_kernel(bencher: Bencher, dataset: &str) {
    run(bencher, unblocked, dataset, false);
}

#[divan::bench(args = DATASETS)]
fn blocked_kernel(bencher: Bencher, dataset: &str) {
    run(bencher, blocked, dataset, false);
}

#[divan::bench(args = DATASETS)]
fn blocked_decode(bencher: Bencher, dataset: &str) {
    run(bencher, blocked, dataset, true);
}
