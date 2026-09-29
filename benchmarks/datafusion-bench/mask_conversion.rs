// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Standalone diagnostic for converting a logical 1024-bit selection bitmap
//! into the byte mask for one 2-bit FastLanes block. Build directly with rustc.

#[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
compile_error!("mask_conversion.rs requires little-endian AArch64 with NEON");

use std::arch::aarch64::vandq_u8;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::vdupq_n_u16;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vorrq_u8;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vreinterpretq_u8_u16;
use std::arch::aarch64::vst1q_u8;
use std::arch::aarch64::vtstq_u8;
use std::hint::black_box;
use std::time::Instant;

#[cfg(target_os = "macos")]
mod counters {
    use std::ffi::c_void;
    use std::mem::MaybeUninit;
    use std::mem::offset_of;
    use std::mem::size_of;

    // sys/resource.h rusage_info_v4: 29 u64 fields precede ri_instructions;
    // ri_cycles follows it. Keep the complete 296-byte C layout for libproc.
    #[repr(C)]
    struct RusageInfoV4 {
        uuid: [u8; 16],
        fields_before: [u64; 29],
        instructions: u64,
        cycles: u64,
        fields_after: [u64; 4],
    }

    #[link(name = "proc")]
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut c_void) -> i32;
    }

    pub(super) fn read() -> Option<(u64, u64)> {
        const RUSAGE_INFO_V4: i32 = 4;
        assert_eq!(size_of::<RusageInfoV4>(), 296);
        assert_eq!(offset_of!(RusageInfoV4, instructions), 248);
        assert_eq!(offset_of!(RusageInfoV4, cycles), 256);
        let mut info = MaybeUninit::<RusageInfoV4>::uninit();
        // SAFETY: the buffer has the complete C struct layout and this process's
        // PID is valid. A successful return initializes every field read below.
        let status = unsafe {
            proc_pid_rusage(
                std::process::id() as i32,
                RUSAGE_INFO_V4,
                info.as_mut_ptr().cast(),
            )
        };
        (status == 0).then(|| {
            // SAFETY: proc_pid_rusage returned success and initialized the struct.
            let info = unsafe { info.assume_init() };
            (info.instructions, info.cycles)
        })
    }
}

#[cfg(not(target_os = "macos"))]
mod counters {
    pub(super) fn read() -> Option<(u64, u64)> {
        None
    }
}

#[path = "mask_conversion_transpose.rs"]
mod transpose;

const PREDICATE_BYTES: usize = 128;
const MASK_BYTES: usize = 256;
const VALUES_PER_BLOCK: usize = 1024;

/// Expand one full-block predicate into FastLanes' 2-bit byte order.
///
/// The mask construction preserves the original broadcast-based FastLanes
/// sum kernel as a baseline for comparison with the bulk transpose.
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "neon")]
pub fn mask_conversion_block(selection: &[u8; PREDICATE_BYTES], output: &mut [u8; MASK_BYTES]) {
    let lane_weights = [1u8, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128];
    let lane_indices = [0u8, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15];
    // SAFETY: both source arrays have exactly the 16 bytes loaded by NEON.
    let lane_bits = unsafe { vld1q_u8(lane_weights.as_ptr()) };
    let lane_order = unsafe { vld1q_u8(lane_indices.as_ptr()) };
    for pair in 0..16 {
        let row_group = pair / 8;
        let lane_group = (pair % 8) * 2;
        let mut mask = vdupq_n_u8(0);
        for row_in_word in 0..4 {
            let row = row_group * 4 + row_in_word;
            let selected_pair = u16::from_le_bytes([
                selection[row * 16 + lane_group],
                selection[row * 16 + lane_group + 1],
            ]);
            let selected = vreinterpretq_u8_u16(vdupq_n_u16(selected_pair));
            let row_mask = vdupq_n_u8(3 << (row_in_word * 2));
            mask = vorrq_u8(mask, vandq_u8(vtstq_u8(selected, lane_bits), row_mask));
        }
        mask = vqtbl1q_u8(mask, lane_order);
        // SAFETY: pair ranges over 16 sixteen-byte chunks in a 256-byte output.
        unsafe { vst1q_u8(output.as_mut_ptr().add(pair * 16), mask) };
    }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn scalar_mask(selection: &[u8; PREDICATE_BYTES]) -> [u8; MASK_BYTES] {
    let mut mask = [0u8; MASK_BYTES];
    for logical in 0..VALUES_PER_BLOCK {
        if (selection[logical / 8] >> (logical % 8)) & 1 == 0 {
            continue;
        }
        let row = logical / 128;
        let lane = logical % 128;
        let packed_byte = (row / 4) * 128 + lane;
        mask[packed_byte] |= 3 << ((row % 4) * 2);
    }
    mask
}

struct Options {
    blocks: usize,
    iterations: usize,
    passes: usize,
    check_only: bool,
    algorithm: String,
}

fn options() -> Result<Options, String> {
    let mut options = Options {
        blocks: 100_000,
        iterations: 30,
        passes: 1,
        check_only: false,
        algorithm: "broadcast".to_owned(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--algorithm" {
            options.algorithm = args.next().ok_or("missing value for --algorithm")?;
            if !matches!(
                options.algorithm.as_str(),
                "broadcast" | "transpose" | "transpose-store"
            ) {
                return Err(
                    "--algorithm must be broadcast, transpose, or transpose-store".to_owned(),
                );
            }
            continue;
        }
        if argument == "--check-only" {
            options.check_only = true;
            continue;
        }
        if argument == "--help" {
            println!(
                "Usage: mask_conversion [--algorithm broadcast|transpose|transpose-store] [--blocks N] [--iterations N] [--passes N] [--check-only]"
            );
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;
        let number = value
            .parse::<usize>()
            .map_err(|_| format!("invalid value for {argument}: {value}"))?;
        if number == 0 {
            return Err(format!("{argument} must be positive"));
        }
        match argument.as_str() {
            "--blocks" => options.blocks = number,
            "--iterations" => options.iterations = number,
            "--passes" => options.passes = number,
            _ => return Err(format!("unknown option: {argument}")),
        }
    }
    options
        .blocks
        .checked_mul(options.passes)
        .ok_or("blocks * passes overflows usize")?;
    Ok(options)
}

fn validate(input: &[[u8; PREDICATE_BYTES]], output: &[[u8; MASK_BYTES]]) -> usize {
    let mut selected = 0usize;
    for (index, (selection, mask)) in input.iter().zip(output).enumerate() {
        assert_eq!(
            *mask,
            scalar_mask(selection),
            "mask mismatch at block {index}"
        );
        selected += selection
            .iter()
            .map(|byte| byte.count_ones() as usize)
            .sum::<usize>();
    }
    selected
}

// Check every source-bit position, both extremes, and all sixteen four-row
// combinations at each column, independently of the randomized timed input.
fn validate_edge_cases(convert: unsafe fn(&[u8; 128], &mut [u8; 256])) {
    let mut output = [0u8; MASK_BYTES];
    let mut check = |selection: &[u8; PREDICATE_BYTES]| {
        // SAFETY: this executable is restricted to AArch64 with NEON.
        unsafe { convert(selection, &mut output) };
        assert_eq!(output, scalar_mask(selection));
    };
    check(&[0; PREDICATE_BYTES]);
    check(&[255; PREDICATE_BYTES]);
    for logical in 0..VALUES_PER_BLOCK {
        let mut selection = [0u8; PREDICATE_BYTES];
        selection[logical / 8] = 1 << (logical % 8);
        check(&selection);
    }
    for group in 0..2 {
        for column in 0..128 {
            for combination in 0..16 {
                let mut selection = [0u8; PREDICATE_BYTES];
                for row in 0..4 {
                    if combination & (1 << row) != 0 {
                        selection[group * 64 + row * 16 + column / 8] |= 1 << (column % 8);
                    }
                }
                check(&selection);
            }
        }
    }
}

fn main() -> Result<(), String> {
    let options = options()?;
    let convert = match options.algorithm.as_str() {
        "broadcast" => mask_conversion_block,
        "transpose" => transpose::mask_conversion_transpose,
        "transpose-store" => transpose::mask_conversion_structure_store,
        _ => unreachable!(),
    };
    validate_edge_cases(convert);
    let mut input = vec![[0u8; PREDICATE_BYTES]; options.blocks];
    let mut output = vec![[0u8; MASK_BYTES]; options.blocks];
    for (block, selection) in input.iter_mut().enumerate() {
        for (byte, value) in selection.iter_mut().enumerate() {
            *value = mix64(0x9e37_79b9_7f4a_7c15 ^ ((block as u64) << 32 | byte as u64)) as u8;
        }
    }

    for (selection, mask) in input.iter().zip(&mut output) {
        // SAFETY: this executable is restricted to AArch64 with NEON.
        unsafe { convert(selection, mask) };
    }
    let selected = validate(&input, &output);
    if options.check_only {
        println!(
            "validated {} blocks and {} selected values",
            options.blocks, selected
        );
        return Ok(());
    }

    let conversions = options.blocks * options.passes;
    let mut timings = Vec::with_capacity(options.iterations);
    let mut instructions_per_block = Vec::with_capacity(options.iterations);
    let mut cycles_per_block = Vec::with_capacity(options.iterations);
    let mut ipc = Vec::with_capacity(options.iterations);
    for _ in 0..options.iterations {
        let before = counters::read();
        let start = Instant::now();
        for _ in 0..options.passes {
            for (selection, mask) in input.iter().zip(&mut output) {
                // SAFETY: this executable is restricted to AArch64 with NEON.
                unsafe { convert(black_box(selection), black_box(mask)) };
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let after = counters::read();
        black_box(&output);
        timings.push(elapsed);
        if let (
            Some((before_instructions, before_cycles)),
            Some((after_instructions, after_cycles)),
        ) = (before, after)
            && let (Some(instructions), Some(cycles)) = (
                after_instructions.checked_sub(before_instructions),
                after_cycles.checked_sub(before_cycles),
            )
            && cycles > 0
        {
            instructions_per_block.push(instructions as f64 / conversions as f64);
            cycles_per_block.push(cycles as f64 / conversions as f64);
            ipc.push(instructions as f64 / cycles as f64);
        }
    }
    assert_eq!(validate(&input, &output), selected);
    timings.sort_by(f64::total_cmp);
    let median = (timings[(options.iterations - 1) / 2] + timings[options.iterations / 2]) / 2.0;
    let conversions_per_sec = conversions as f64 / median;
    println!(
        "algorithm={} blocks={} passes={} iterations={} selected_values={} median_ms={:.3} ns_per_block={:.3} masks_per_sec={:.0} input_gb_per_sec={:.3} output_gb_per_sec={:.3}",
        options.algorithm,
        options.blocks,
        options.passes,
        options.iterations,
        selected,
        median * 1000.0,
        median * 1e9 / conversions as f64,
        conversions_per_sec,
        conversions_per_sec * PREDICATE_BYTES as f64 / 1e9,
        conversions_per_sec * MASK_BYTES as f64 / 1e9,
    );
    if instructions_per_block.len() == options.iterations {
        instructions_per_block.sort_by(f64::total_cmp);
        cycles_per_block.sort_by(f64::total_cmp);
        ipc.sort_by(f64::total_cmp);
        let median_of = |values: &[f64]| {
            (values[(options.iterations - 1) / 2] + values[options.iterations / 2]) / 2.0
        };
        println!(
            "retired_instructions_per_block={:.3} cycles_per_block={:.3} ipc={:.3} counter_scope=process",
            median_of(&instructions_per_block),
            median_of(&cycles_per_block),
            median_of(&ipc),
        );
    } else {
        println!("retired_instruction_counters=unavailable");
    }
    Ok(())
}
