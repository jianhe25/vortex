// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executions that re-enter a parent's `execute` through `ExecuteSlot`.
//!
//! Each re-entry repeats the parent's `require_*` checks for slots that already completed. These
//! benchmarks are the ones a resume-slot / state-index optimisation of that path would move.

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::patches::Patches;
use vortex_array::session::ArraySession;
use vortex_array::stats::StatsSession;
use vortex_array::memory::MemorySession;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex_fastlanes::bitpack_compress::test_harness::make_array;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

/// No execute-parent kernels at all, so every parent goes through `ExecuteSlot` re-entry.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<ArraySession>()
        .with_some(KernelSession::empty())
        .with::<StatsSession>()
        .with::<MemorySession>()
});

const LENS: &[usize] = &[8, 64, 1024, 16384];

fn bitpacked_u32(n: usize, k: usize, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let codes = PrimitiveArray::from_iter((0..n).map(|i| (i % k) as u32));
    Ok(bitpack_encode(&codes, 16, None, ctx)?.into_array())
}

/// BitPacked whose patch indices and patch values are themselves BitPacked, so
/// `BitPacked::execute` re-enters twice (indices, then values) before unpacking.
fn bitpacked_with_bitpacked_patches(n: usize, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let mut rng = StdRng::seed_from_u64(0);
    let array = make_array(&mut rng, n, 0.25, 0.0, ctx)?;
    let view = array.as_::<BitPacked>();
    let patches = view.patches().expect("make_array produced patches");
    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    let patches = Patches::new(
        patches.array_len(),
        patches.offset(),
        bitpack_to_best_bit_width(&indices, ctx)?.into_array(),
        bitpack_to_best_bit_width(&values, ctx)?.into_array(),
        None,
    )?;
    Ok(BitPacked::try_new(
        view.packed().clone(),
        view.ptype(view.dtype()),
        view.validity()?,
        Some(patches),
        view.bit_width(),
        view.len(),
        view.offset(),
    )?
    .into_array())
}

fn bench<F: Fn(&mut ExecutionCtx) -> VortexResult<ArrayRef> + Sync>(bencher: Bencher, build: F) {
    let ctx = std::sync::Mutex::new(SESSION.create_execution_ctx());
    bencher
        .with_inputs(|| build(&mut SESSION.create_execution_ctx()).expect("build"))
        .bench_values(|array| {
            array
                .execute::<Canonical>(&mut ctx.lock().expect("ctx"))
                .expect("execute")
        });
}

/// Dict re-entered once: codes are BitPacked, values are already Primitive.
#[divan::bench(args = LENS)]
fn dict_one_reentry(bencher: Bencher, n: usize) {
    let k = (n / 4).max(2);
    bench(bencher, |ctx| {
        let values = PrimitiveArray::from_iter((0..k).map(|i| i as u32 * 3)).into_array();
        Ok(DictArray::try_new(bitpacked_u32(n, k, ctx)?, values)?.into_array())
    });
}

/// Dict re-entered twice: both codes and values are BitPacked.
#[divan::bench(args = LENS)]
fn dict_two_reentries(bencher: Bencher, n: usize) {
    let k = (n / 4).max(2);
    bench(bencher, |ctx| {
        Ok(DictArray::try_new(bitpacked_u32(n, k, ctx)?, bitpacked_u32(k, k, ctx)?)?.into_array())
    });
}

/// BitPacked re-entered twice through `require_patches!`.
#[divan::bench(args = LENS)]
fn bitpacked_patch_reentries(bencher: Bencher, n: usize) {
    bench(bencher, |ctx| bitpacked_with_bitpacked_patches(n, ctx));
}

/// Dict over a BitPacked with BitPacked patches: four re-entries in the tree.
#[divan::bench(args = LENS)]
fn dict_over_patched_bitpacked(bencher: Bencher, n: usize) {
    bench(bencher, |ctx| {
        let codes = bitpacked_with_bitpacked_patches(n, ctx)?;
        let values = PrimitiveArray::from_iter((0..(1 << 14)).map(|i| i as u32)).into_array();
        Ok(DictArray::try_new(codes, values)?.into_array())
    });
}

/// Control: no re-entry, a plain BitPacked with no patches or validity child.
#[divan::bench(args = LENS)]
fn bitpacked_no_reentry(bencher: Bencher, n: usize) {
    bench(bencher, |ctx| bitpacked_u32(n, 16, ctx));
}
