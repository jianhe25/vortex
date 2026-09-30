// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks filtering a BitPacked FoR array with a random mask, three ways:
//!
//! - `unblocked_kernel`: a single reference, through the FoR filter kernel.
//! - `blocked_kernel`: a reference per 1024-element chunk, through the FoR filter kernel.
//! - `blocked_decode`: a reference per chunk, decoded first and then filtered, as happens
//!   without a kernel.
//!
//! `drifting` values climb by a million every chunk, so per-chunk references pack narrower than a
//! single one. `uniform` values share one range, so both pack to the same width.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench for_filter`.

#![expect(clippy::unwrap_used)]
// The benchmarks are kept out of CodSpeed, which leaves their helpers unused there.
#![cfg_attr(codspeed, allow(dead_code, unused_imports))]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArray;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_mask::Mask;
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

/// (dataset, percentage of rows selected)
const ARGS: &[(&str, f64)] = &[
    ("drifting", 0.1),
    ("drifting", 1.0),
    ("drifting", 5.0),
    ("drifting", 50.0),
    ("uniform", 0.1),
    ("uniform", 1.0),
    ("uniform", 5.0),
    ("uniform", 50.0),
];

fn values(dataset: &str) -> Buffer<u32> {
    (0..LEN)
        .map(|i| match dataset {
            "drifting" => (i / 1024) * 1_000_000 + (i * 7919) % 100,
            _ => 1000 + (i * 7919) % 100,
        })
        .collect()
}

/// A pseudo-random mask selecting about `percent` of the rows.
fn mask(percent: f64) -> Mask {
    let mut state = 0x2545_f491_u32;
    Mask::from_iter((0..LEN).map(|_| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        f64::from(state % 100_000) < percent * 1000.0
    }))
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
    let bit_width = u8::try_from(u32::BITS - max.leading_zeros()).unwrap();
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
    assert!(for_array.constant_reference().is_none());
    FoR::try_new_chunked(bitpack(&for_array), for_array.references().clone(), 0)
        .unwrap()
        .into_array()
}

fn run(bencher: Bencher, array: ArrayRef, percent: f64, decode_first: bool) {
    let mask = mask(percent);
    bencher
        .counter(ItemsCount::new(mask.true_count()))
        .with_inputs(|| (&array, &mask, SESSION.create_execution_ctx()))
        .bench_refs(|(array, mask, ctx)| {
            let array = if decode_first {
                (*array)
                    .clone()
                    .execute::<PrimitiveArray>(ctx)
                    .unwrap()
                    .into_array()
            } else {
                (*array).clone()
            };
            array
                .filter((*mask).clone())
                .unwrap()
                .execute::<PrimitiveArray>(ctx)
                .unwrap()
        });
}

#[cfg(not(codspeed))]
#[divan::bench(args = ARGS)]
fn unblocked_kernel(bencher: Bencher, (dataset, percent): (&str, f64)) {
    run(bencher, unblocked(dataset), percent, false);
}

#[cfg(not(codspeed))]
#[divan::bench(args = ARGS)]
fn blocked_kernel(bencher: Bencher, (dataset, percent): (&str, f64)) {
    run(bencher, blocked(dataset), percent, false);
}

#[cfg(not(codspeed))]
#[divan::bench(args = ARGS)]
fn blocked_decode(bencher: Bencher, (dataset, percent): (&str, f64)) {
    run(bencher, blocked(dataset), percent, true);
}
