// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bulk four-row bit transpose followed by two-bit mask expansion.

use std::arch::aarch64::uint8x16_t;
use std::arch::aarch64::uint32x4x2_t;
use std::arch::aarch64::vandq_u8;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::veorq_u8;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vreinterpretq_u8_u16;
use std::arch::aarch64::vreinterpretq_u8_u32;
use std::arch::aarch64::vreinterpretq_u16_u8;
use std::arch::aarch64::vreinterpretq_u32_u8;
use std::arch::aarch64::vshlq_n_u8;
use std::arch::aarch64::vshrq_n_u8;
use std::arch::aarch64::vst1q_u8;
use std::arch::aarch64::vst2q_u32;
use std::arch::aarch64::vzip1q_u8;
use std::arch::aarch64::vzip1q_u16;
use std::arch::aarch64::vzip1q_u32;
use std::arch::aarch64::vzip2q_u8;
use std::arch::aarch64::vzip2q_u16;
use std::arch::aarch64::vzip2q_u32;

#[inline(always)]
unsafe fn swap<const S: i32>(
    a: uint8x16_t,
    b: uint8x16_t,
    mask: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    // SAFETY: caller executes on AArch64 with NEON.
    unsafe {
        let t = vandq_u8(veorq_u8(vshrq_n_u8::<S>(a), b), mask);
        (veorq_u8(a, vshlq_n_u8::<S>(t)), veorq_u8(b, t))
    }
}

/// Convert a logical 1,024-bit predicate into a 256-byte two-bit FastLanes mask.
///
/// # Safety
/// The caller must execute on AArch64 with NEON.
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "neon")]
pub unsafe fn mask_conversion_transpose(selection: &[u8; 128], output: &mut [u8; 256]) {
    // SAFETY: caller guarantees NEON; fixed-size arrays bound all accesses.
    unsafe { convert::<false>(selection, output) }
}

/// Use an interleaving structure store to emit the same two-bit FastLanes mask.
///
/// # Safety
/// The caller must execute on AArch64 with NEON.
#[unsafe(no_mangle)]
#[inline(never)]
#[target_feature(enable = "neon")]
pub unsafe fn mask_conversion_structure_store(selection: &[u8; 128], output: &mut [u8; 256]) {
    // SAFETY: caller guarantees NEON; fixed-size arrays bound all accesses.
    unsafe { convert::<true>(selection, output) }
}

#[inline(always)]
unsafe fn convert<const STRUCTURE_STORE: bool>(selection: &[u8; 128], output: &mut [u8; 256]) {
    // SAFETY: all input/output addresses stay within the fixed-size arrays;
    // the lookup table contains the sixteen bytes loaded. NEON is enabled.
    unsafe {
        let m1 = vdupq_n_u8(0x55);
        let m2 = vdupq_n_u8(0x33);
        let m4 = vdupq_n_u8(0x0f);
        let table = vld1q_u8(
            [
                0u8, 3, 12, 15, 48, 51, 60, 63, 192, 195, 204, 207, 240, 243, 252, 255,
            ]
            .as_ptr(),
        );
        for group in 0..2 {
            let p = selection.as_ptr().add(group * 64);
            let a = vld1q_u8(p);
            let b = vld1q_u8(p.add(16));
            let c = vld1q_u8(p.add(32));
            let d = vld1q_u8(p.add(48));
            // Exchange the row index with the low two column-index bits. Each
            // byte then holds two four-row predicate nibbles, four columns apart.
            let (a, b) = swap::<1>(a, b, m1);
            let (c, d) = swap::<1>(c, d, m1);
            let (a, c) = swap::<2>(a, c, m2);
            let (b, d) = swap::<2>(b, d, m2);
            // Interleave row vectors into consecutive four-column groups.
            let ab0 = vzip1q_u8(a, b);
            let ab1 = vzip2q_u8(a, b);
            let cd0 = vzip1q_u8(c, d);
            let cd1 = vzip2q_u8(c, d);
            let chunks = [
                vreinterpretq_u8_u16(vzip1q_u16(
                    vreinterpretq_u16_u8(ab0),
                    vreinterpretq_u16_u8(cd0),
                )),
                vreinterpretq_u8_u16(vzip2q_u16(
                    vreinterpretq_u16_u8(ab0),
                    vreinterpretq_u16_u8(cd0),
                )),
                vreinterpretq_u8_u16(vzip1q_u16(
                    vreinterpretq_u16_u8(ab1),
                    vreinterpretq_u16_u8(cd1),
                )),
                vreinterpretq_u8_u16(vzip2q_u16(
                    vreinterpretq_u16_u8(ab1),
                    vreinterpretq_u16_u8(cd1),
                )),
            ];
            for (i, x) in chunks.into_iter().enumerate() {
                // Lookup duplicates each nibble bit: abcd -> aabbccdd. ZIP32
                // joins adjacent four-column groups into FastLanes byte order.
                let lo = vqtbl1q_u8(table, vandq_u8(x, m4));
                let hi = vqtbl1q_u8(table, vshrq_n_u8::<4>(x));
                if STRUCTURE_STORE {
                    // NEON structure stores permit unaligned addresses. This
                    // interleaves four-byte groups exactly like the ZIP32 pair.
                    vst2q_u32(
                        output.as_mut_ptr().add(group * 128 + i * 32).cast::<u32>(),
                        uint32x4x2_t(vreinterpretq_u32_u8(lo), vreinterpretq_u32_u8(hi)),
                    );
                } else {
                    let y0 = vreinterpretq_u8_u32(vzip1q_u32(
                        vreinterpretq_u32_u8(lo),
                        vreinterpretq_u32_u8(hi),
                    ));
                    let y1 = vreinterpretq_u8_u32(vzip2q_u32(
                        vreinterpretq_u32_u8(lo),
                        vreinterpretq_u32_u8(hi),
                    ));
                    vst1q_u8(output.as_mut_ptr().add(group * 128 + i * 32), y0);
                    vst1q_u8(output.as_mut_ptr().add(group * 128 + i * 32 + 16), y1);
                }
            }
        }
    }
}
