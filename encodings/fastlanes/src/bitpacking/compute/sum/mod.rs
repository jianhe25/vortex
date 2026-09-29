// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sum two-bit unsigned integers without expanding FastLanes blocks.

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::uint8x16_t;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vaddlvq_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vaddq_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vandq_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vcntq_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vdupq_n_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::veorq_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vld1q_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vmaxvq_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vorrq_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vqtbl1q_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vreinterpretq_u8_u16;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vreinterpretq_u8_u32;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vreinterpretq_u16_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vreinterpretq_u32_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vshlq_n_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vshrq_n_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vzip1q_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vzip1q_u16;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vzip1q_u32;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vzip2q_u8;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vzip2q_u16;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::vzip2q_u32;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::GroupedArray;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::aggregate_fn::kernels::DynGroupedAggregateKernel;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::filter::FilterArraySlotsExt;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::BitPacked;
use crate::BitPackedArrayExt;

const BLOCK_VALUES: usize = 1024;
const BLOCK_BYTES: usize = BLOCK_VALUES / 4;
const WORDS_PER_BLOCK: usize = BLOCK_BYTES / 8;
const LOW_BITS: u64 = 0x5555_5555_5555_5555;
const HIGH_BITS: u64 = 0xaaaa_aaaa_aaaa_aaaa;
static PREFIX_MASKS: [[u64; WORDS_PER_BLOCK]; 65] = build_prefix_masks();

// For u8 FastLanes uses 128 lanes and eight rows. Bitpacking at width two
// places four rows into each byte, preserving the lane within each half-block.
const fn packed_position(index: usize) -> (usize, usize) {
    let lane = index % 128;
    let row = index / 128;
    let byte = (row / 4) * 128 + lane;
    (byte / 8, (byte % 8) * 8 + (row % 4) * 2)
}

const fn build_prefix_masks() -> [[u64; WORDS_PER_BLOCK]; 65] {
    let mut masks = [[0u64; WORDS_PER_BLOCK]; 65];
    let mut boundary = 1;
    while boundary <= 64 {
        masks[boundary] = masks[boundary - 1];
        let mut index = (boundary - 1) * 16;
        while index < boundary * 16 {
            let (word, shift) = packed_position(index);
            masks[boundary][word] |= 3u64 << shift;
            index += 1;
        }
        boundary += 1;
    }
    masks
}

fn toggle_position(mask: &mut [u64; WORDS_PER_BLOCK], index: usize) {
    let (word, shift) = packed_position(index);
    mask[word] ^= 3u64 << shift;
}

#[inline(always)]
fn sum_packed_block(block: &[u8], start: usize, end: usize) -> u64 {
    if start == 0 && end == BLOCK_VALUES {
        let (words, remainder) = block.as_chunks::<8>();
        debug_assert!(remainder.is_empty());
        return words
            .iter()
            .map(|bytes| {
                let word = u64::from_le_bytes(*bytes);
                u64::from((word & LOW_BITS).count_ones())
                    + 2 * u64::from((word & HIGH_BITS).count_ones())
            })
            .sum();
    }

    // Tiny ranges touch too few values to pay for a 256-byte mask and packed scan.
    if end - start < 16 {
        return (start..end)
            // SAFETY: block is one complete two-bit block and the range lies within it.
            .map(|index| u64::from(unsafe { u8::unchecked_unpack_single(2, block, index) }))
            .sum();
    }

    let mask = interval_mask(start, end);

    let (words, remainder) = block.as_chunks::<8>();
    debug_assert!(remainder.is_empty());
    words
        .iter()
        .zip(mask)
        .map(|(bytes, bits)| {
            let word = u64::from_le_bytes(*bytes) & bits;
            u64::from((word & LOW_BITS).count_ones())
                + 2 * u64::from((word & HIGH_BITS).count_ones())
        })
        .sum()
}

/// Scalar and grouped sum kernel for two-bit unsigned integers.
#[derive(Debug)]
pub(crate) struct BitPackedSumKernel;

/// Grouped sum for a filtered two-bit unsigned array.
#[derive(Debug)]
pub(crate) struct FilteredBitPackedSumKernel;

impl DynGroupedAggregateKernel for FilteredBitPackedSumKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !aggregate_fn.is::<SumV2>() {
            return Ok(None);
        }
        try_fused_filter_grouped_sum(groups, ctx)
    }
}

impl DynAggregateKernel for BitPackedSumKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<Sum>() && !aggregate_fn.is::<SumV2>() {
            return Ok(None);
        }
        let Some(array) = eligible_array(batch, ctx)? else {
            return Ok(None);
        };
        let sum = sum_range(&array, 0, array.len());
        let sum = Scalar::primitive(sum, Nullability::Nullable);
        if aggregate_fn.is::<SumV2>() {
            Ok(Some(SumV2::partial_from_sum(sum, array.is_empty())?))
        } else {
            Ok(Some(sum))
        }
    }
}

impl DynGroupedAggregateKernel for BitPackedSumKernel {
    fn supports_filtered_grouped_aggregate(&self) -> bool {
        true
    }

    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !aggregate_fn.is::<SumV2>() {
            return Ok(None);
        }
        let Some(array) = eligible_array(groups.elements(), ctx)? else {
            return Ok(None);
        };
        let ranges = groups.group_ranges(ctx)?;
        let group_validity = groups.group_validity(ctx)?;
        let mut empty = BitBufferMut::new_unset(ranges.len());
        let sums =
            PrimitiveArray::from_iter(ranges.iter().enumerate().map(|(idx, (start, len))| {
                if !group_validity.value(idx) {
                    return 0u64;
                }
                if len == 0 {
                    // SAFETY: idx is an index of ranges, and empty has ranges.len() bits.
                    unsafe { empty.set_unchecked(idx) };
                }
                sum_range(&array, start, len)
            }));
        let false_bits = BitBuffer::new_unset(ranges.len());
        let fields = FieldNames::from_iter([
            FieldName::from("sum"),
            FieldName::from("is_overflow"),
            FieldName::from("is_empty"),
        ]);
        let partial = StructArray::try_new(
            fields,
            [
                sums.into_array(),
                BoolArray::new(false_bits, Validity::NonNullable).into_array(),
                BoolArray::new(empty.freeze(), Validity::NonNullable).into_array(),
            ],
            ranges.len(),
            Validity::from_mask(group_validity, Nullability::Nullable),
        )?;
        Ok(Some(partial.into_array()))
    }

    fn filtered_grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        predicate: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !aggregate_fn.is::<SumV2>() {
            return Ok(None);
        }
        try_early_filtered_grouped_sum(groups, predicate, ctx)
    }
}

fn eligible_array<'a>(
    batch: &'a ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayView<'a, BitPacked>>> {
    let Some(array) = batch.as_opt::<BitPacked>() else {
        return Ok(None);
    };
    if array.dtype().as_ptype() != PType::U8
        || array.bit_width() != 2
        || array.patches().is_some()
        || !array.validity()?.execute_mask(array.len(), ctx)?.all_true()
    {
        return Ok(None);
    }
    Ok(Some(array))
}

fn sum_range(array: &ArrayView<'_, BitPacked>, start: usize, len: usize) -> u64 {
    let packed = array.packed().as_host();
    let start = start + array.offset() as usize;
    let end = start + len;
    let mut position = start;
    let mut sum = 0u64;
    while position < end {
        let block_index = position / BLOCK_VALUES;
        let block = &packed[block_index * BLOCK_BYTES..(block_index + 1) * BLOCK_BYTES];
        let block_offset = position % BLOCK_VALUES;
        let block_end = end.min((block_index + 1) * BLOCK_VALUES);
        sum += sum_packed_block(block, block_offset, block_offset + block_end - position);
        position = block_end;
    }
    sum
}

const EXPAND8: [u64; 256] = build_expand8();

const fn build_expand8() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut value = 0;
    while value < 256 {
        let mut bit = 0;
        while bit < 8 {
            if (value & (1 << bit)) != 0 {
                table[value] |= 1u64 << (8 * bit);
            }
            bit += 1;
        }
        value += 1;
    }
    table
}

// Map each logical 128-lane row of the selection bitmap to the matching
// four-row packed value byte. Each set selection bit becomes two set bits.
fn packed_selection_mask(selection: &[u8; 128]) -> [u64; WORDS_PER_BLOCK] {
    let mut packed = [0u64; WORDS_PER_BLOCK];
    for row_group in 0..2 {
        for lane_group in 0..16 {
            let mut lanes = 0u64;
            for row_in_word in 0..4 {
                let row = row_group * 4 + row_in_word;
                let selected_lanes = selection[row * 16 + lane_group];
                lanes |= EXPAND8[selected_lanes as usize] << (row_in_word * 2);
            }
            packed[row_group * 16 + lane_group] = lanes | (lanes << 1);
        }
    }
    packed
}

fn sum_masked_block(
    block: &[u8],
    selection: &[u64; WORDS_PER_BLOCK],
    interval: Option<&[u64; WORDS_PER_BLOCK]>,
) -> (u64, bool) {
    let (words, tail) = block.as_chunks::<8>();
    debug_assert!(tail.is_empty());
    let mut sum = 0u64;
    let mut selected = 0u64;
    for (index, bytes) in words.iter().enumerate() {
        let mut mask = selection[index];
        if let Some(interval) = interval {
            mask &= interval[index];
        }
        selected |= mask;
        let bits = u64::from_le_bytes(*bytes) & mask;
        sum += u64::from((bits & LOW_BITS).count_ones());
        sum += 2 * u64::from((bits & HIGH_BITS).count_ones());
    }
    (sum, selected != 0)
}

fn sum_selected_block(
    block: &[u8],
    selection: &[u8; 128],
    interval: Option<&[u64; WORDS_PER_BLOCK]>,
) -> (u64, bool) {
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        // SAFETY: NEON is available on all AArch64 targets. The caller passes a complete
        // 256-byte packed block and a complete 128-byte predicate block.
        unsafe { sum_selected_block_neon(block, selection, interval) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
    {
        let packed_mask = packed_selection_mask(selection);
        sum_masked_block(block, &packed_mask, interval)
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline(always)]
unsafe fn transpose_swap<const SHIFT: i32>(
    left: uint8x16_t,
    right: uint8x16_t,
    mask: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    // SAFETY: callers enable NEON and only instantiate this with valid shifts 1 and 2.
    unsafe {
        let bits = vandq_u8(veorq_u8(vshrq_n_u8::<SHIFT>(left), right), mask);
        (
            veorq_u8(left, vshlq_n_u8::<SHIFT>(bits)),
            veorq_u8(right, bits),
        )
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn sum_selected_block_neon(
    block: &[u8],
    selection: &[u8; 128],
    interval: Option<&[u64; WORDS_PER_BLOCK]>,
) -> (u64, bool) {
    debug_assert_eq!(block.len(), BLOCK_BYTES);
    let duplicate_bits = [
        0u8, 3, 12, 15, 48, 51, 60, 63, 192, 195, 204, 207, 240, 243, 252, 255,
    ];
    // SAFETY: duplicate_bits contains exactly the 16 bytes loaded by vld1q_u8.
    let table = unsafe { vld1q_u8(duplicate_bits.as_ptr()) };
    let one_bit_mask = vdupq_n_u8(0x55);
    let two_bit_mask = vdupq_n_u8(0x33);
    let nibble_mask = vdupq_n_u8(0x0f);
    let low_bits = vdupq_n_u8(0x55);
    let high_bits = vdupq_n_u8(0xaa);
    let mut low_count = vdupq_n_u8(0);
    let mut high_count = vdupq_n_u8(0);
    let mut any_selected = vdupq_n_u8(0);

    // Each group contains four rows of 128 logical lanes, matching 128 packed bytes.
    for group in 0..2 {
        let predicate_start = group * 64;
        // SAFETY: each group contains 64 predicate bytes and each load stays within it.
        let (mut a, mut b, mut c, mut d) = unsafe {
            let ptr = selection.as_ptr().add(predicate_start);
            (
                vld1q_u8(ptr),
                vld1q_u8(ptr.add(16)),
                vld1q_u8(ptr.add(32)),
                vld1q_u8(ptr.add(48)),
            )
        };
        // Exchange the row index with the low two column-index bits, producing
        // two four-row predicate nibbles per byte.
        // SAFETY: this AArch64 NEON kernel invokes only the valid shifts 1 and 2.
        unsafe {
            (a, b) = transpose_swap::<1>(a, b, one_bit_mask);
            (c, d) = transpose_swap::<1>(c, d, one_bit_mask);
            (a, c) = transpose_swap::<2>(a, c, two_bit_mask);
            (b, d) = transpose_swap::<2>(b, d, two_bit_mask);
        }
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
        for (chunk_idx, packed_predicate) in chunks.into_iter().enumerate() {
            let low = vqtbl1q_u8(table, vandq_u8(packed_predicate, nibble_mask));
            let high = vqtbl1q_u8(table, vshrq_n_u8::<4>(packed_predicate));
            let masks = [
                vreinterpretq_u8_u32(vzip1q_u32(
                    vreinterpretq_u32_u8(low),
                    vreinterpretq_u32_u8(high),
                )),
                vreinterpretq_u8_u32(vzip2q_u32(
                    vreinterpretq_u32_u8(low),
                    vreinterpretq_u32_u8(high),
                )),
            ];
            for (half, mut mask) in masks.into_iter().enumerate() {
                let pair = group * 8 + chunk_idx * 2 + half;
                if let Some(interval) = interval {
                    // SAFETY: interval contains 32 words, and pair is within 0..16.
                    let interval_bytes =
                        unsafe { vld1q_u8(interval.as_ptr().add(pair * 2).cast::<u8>()) };
                    mask = vandq_u8(mask, interval_bytes);
                }
                any_selected = vorrq_u8(any_selected, mask);
                // SAFETY: pair ranges over 16 sixteen-byte chunks in a complete packed block.
                let packed = unsafe { vld1q_u8(block.as_ptr().add(pair * 16)) };
                let selected_values = vandq_u8(packed, mask);
                low_count = vaddq_u8(low_count, vcntq_u8(vandq_u8(selected_values, low_bits)));
                high_count = vaddq_u8(high_count, vcntq_u8(vandq_u8(selected_values, high_bits)));
            }
        }
    }
    // Each byte lane counts at most four low or high bits per pair, so 16 pairs cannot wrap u8.
    (
        u64::from(vaddlvq_u8(low_count)) + 2 * u64::from(vaddlvq_u8(high_count)),
        vmaxvq_u8(any_selected) != 0,
    )
}

fn interval_mask(start: usize, end: usize) -> [u64; WORDS_PER_BLOCK] {
    let lower = start / 16;
    let upper = end / 16;
    let mut mask = [0u64; WORDS_PER_BLOCK];
    for (index, word) in mask.iter_mut().enumerate() {
        *word = PREFIX_MASKS[upper][index] ^ PREFIX_MASKS[lower][index];
    }
    for index in lower * 16..start {
        toggle_position(&mut mask, index);
    }
    for index in upper * 16..end {
        toggle_position(&mut mask, index);
    }
    mask
}

// Nth set bit of a padded 1024-bit selection block. Called only when a group
// boundary cuts through a block, not on every group or every selected value.
fn select_within_block(selection: &[u8; 128], mut nth: usize) -> usize {
    for (byte_index, &byte) in selection.iter().enumerate() {
        let count = byte.count_ones() as usize;
        if nth < count {
            let mut remaining = byte;
            for _ in 0..nth {
                remaining &= remaining - 1;
            }
            return byte_index * 8 + remaining.trailing_zeros() as usize;
        }
        nth -= count;
    }
    unreachable!("nth set bit must be within selection block")
}

/// Align one logical predicate block with a physical BitPacked block.
///
/// Ordinary unsliced arrays borrow the bitmap bytes directly. Sliced arrays and bit-offset
/// predicates shift into the reusable scratch block, with out-of-range bits cleared.
fn predicate_block<'a>(
    bits: &'a BitBuffer,
    physical_start: usize,
    source_offset: usize,
    scratch: &'a mut [u8; 128],
) -> &'a [u8; 128] {
    if let Some(logical_start) = physical_start.checked_sub(source_offset)
        && logical_start + BLOCK_VALUES <= bits.len()
        && logical_start.is_multiple_of(8)
        && let Some(bytes) = bits.byte_aligned_bytes()
    {
        let start_byte = logical_start / 8;
        return bytes[start_byte..start_byte + 128]
            .try_into()
            .expect("complete aligned predicate block");
    }

    scratch.fill(0);
    let raw = bits.inner().as_slice();
    for (byte_index, output) in scratch.iter_mut().enumerate() {
        let physical = physical_start + byte_index * 8;
        if let Some(logical) = physical.checked_sub(source_offset)
            && logical + 8 <= bits.len()
        {
            let source_bit = bits.offset() + logical;
            let byte = source_bit / 8;
            let shift = source_bit % 8;
            *output = raw[byte] >> shift;
            if shift != 0 {
                *output |= raw.get(byte + 1).copied().unwrap_or(0) << (8 - shift);
            }
        } else {
            for bit in 0..8 {
                let Some(logical) = (physical + bit).checked_sub(source_offset) else {
                    continue;
                };
                if logical < bits.len() && bits.value(logical) {
                    *output |= 1 << bit;
                }
            }
        }
    }
    scratch
}

fn try_early_filtered_grouped_sum(
    groups: &GroupedArray,
    predicate: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some(source) = eligible_array(groups.elements(), ctx)? else {
        return Ok(None);
    };
    if predicate.len() != source.len() {
        return Ok(None);
    }
    let ranges = groups.group_ranges(ctx)?;
    let group_validity = groups.group_validity(ctx)?;
    let mut sums = Vec::with_capacity(ranges.len());
    let mut empty = BitBufferMut::new_unset(ranges.len());
    let packed = source.packed().as_host();
    let source_offset = source.offset() as usize;
    let mut scratch = [0u8; 128];

    for (group_idx, (start, size)) in ranges.iter().enumerate() {
        let Some(end) = start.checked_add(size).filter(|&end| end <= source.len()) else {
            return Ok(None);
        };
        if !group_validity.value(group_idx) {
            sums.push(0u64);
            continue;
        }

        let mut sum = 0u64;
        let mut selected = false;
        let mut physical = start + source_offset;
        let physical_end = end + source_offset;
        while physical < physical_end {
            let block_idx = physical / BLOCK_VALUES;
            let block_start = block_idx * BLOCK_VALUES;
            let block = &packed[block_idx * BLOCK_BYTES..(block_idx + 1) * BLOCK_BYTES];
            let range_start = physical - block_start;
            let range_end = (block_start + BLOCK_VALUES).min(physical_end) - block_start;
            match predicate {
                Mask::AllTrue(_) => {
                    sum += sum_packed_block(block, range_start, range_end);
                    selected = true;
                }
                Mask::AllFalse(_) => {}
                Mask::Values(values) => {
                    let bits = predicate_block(
                        values.bit_buffer(),
                        block_start,
                        source_offset,
                        &mut scratch,
                    );
                    let interval = (range_start != 0 || range_end != BLOCK_VALUES)
                        .then(|| interval_mask(range_start, range_end));
                    let (block_sum, block_selected) =
                        sum_selected_block(block, bits, interval.as_ref());
                    sum += block_sum;
                    selected |= block_selected;
                }
            }
            physical = block_start + range_end;
        }
        sums.push(sum);
        if !selected {
            // SAFETY: group_idx comes from enumerating ranges, and empty has ranges.len() bits.
            unsafe { empty.set_unchecked(group_idx) };
        }
    }

    Ok(Some(
        StructArray::try_new(
            FieldNames::from_iter([
                FieldName::from("sum"),
                FieldName::from("is_overflow"),
                FieldName::from("is_empty"),
            ]),
            [
                PrimitiveArray::from_iter(sums).into_array(),
                BoolArray::new(BitBuffer::new_unset(ranges.len()), Validity::NonNullable)
                    .into_array(),
                BoolArray::new(empty.freeze(), Validity::NonNullable).into_array(),
            ],
            ranges.len(),
            Validity::from_mask(group_validity, Nullability::Nullable),
        )?
        .into_array(),
    ))
}

fn try_fused_filter_grouped_sum(
    groups: &GroupedArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some(filtered) = groups.elements().as_opt::<Filter>() else {
        return Ok(None);
    };
    let Some(source) = eligible_array(filtered.child(), ctx)? else {
        return Ok(None);
    };
    if source.offset() != 0 {
        return Ok(None);
    }
    let Mask::Values(selected) = filtered.filter_mask() else {
        return Ok(None);
    };
    let Some(selection_bytes) = selected.bit_buffer().byte_aligned_bytes() else {
        return Ok(None);
    };
    let ranges = groups.group_ranges(ctx)?;
    let spans: Vec<(usize, usize)> = ranges.iter().collect();
    let mut expected_start = 0;
    for &(start, size) in &spans {
        if start != expected_start {
            return Ok(None);
        }
        let Some(next_start) = expected_start.checked_add(size) else {
            return Ok(None);
        };
        expected_start = next_start;
    }
    if expected_start != filtered.len() {
        return Ok(None);
    }
    let group_validity = groups.group_validity(ctx)?;
    let mut sums = vec![0u64; spans.len()];
    let mut empty = BitBufferMut::new_unset(spans.len());
    for (idx, &(_, size)) in spans.iter().enumerate() {
        if size == 0 && group_validity.value(idx) {
            // SAFETY: idx comes from enumerating spans, and empty has spans.len() bits.
            unsafe { empty.set_unchecked(idx) };
        }
    }
    let mut group_idx = 0;
    let mut group_remaining = spans.first().map_or(0, |(_, size)| *size);
    let packed = source.packed().as_host();
    for block_idx in 0..selection_bytes.len().div_ceil(128) {
        let start = block_idx * 128;
        let end = (start + 128).min(selection_bytes.len());
        let mut selection = [0u8; 128];
        selection[..end - start].copy_from_slice(&selection_bytes[start..end]);
        if block_idx == selection_bytes.len().div_ceil(128) - 1 {
            let remainder = selected.bit_buffer().len() % 8;
            if remainder != 0 {
                selection[end - start - 1] &= (1 << remainder) - 1;
            }
        }
        let selected_count: usize = selection
            .iter()
            .map(|byte| byte.count_ones() as usize)
            .sum();
        if selected_count == 0 {
            continue;
        }
        let packed_mask = packed_selection_mask(&selection);
        let block = &packed[block_idx * BLOCK_BYTES..(block_idx + 1) * BLOCK_BYTES];
        let mut rank = 0;
        let mut logical_start = 0;
        while rank < selected_count {
            while group_remaining == 0 {
                group_idx += 1;
                group_remaining = spans[group_idx].1;
            }
            let take = (selected_count - rank).min(group_remaining);
            let logical_end = if rank + take == selected_count {
                BLOCK_VALUES
            } else {
                select_within_block(&selection, rank + take - 1) + 1
            };
            let partial = logical_start != 0 || logical_end != BLOCK_VALUES;
            if group_validity.value(group_idx) {
                let interval = partial.then(|| interval_mask(logical_start, logical_end));
                sums[group_idx] += sum_masked_block(block, &packed_mask, interval.as_ref()).0;
            }
            rank += take;
            group_remaining -= take;
            logical_start = logical_end;
        }
    }
    let partial = StructArray::try_new(
        FieldNames::from_iter([
            FieldName::from("sum"),
            FieldName::from("is_overflow"),
            FieldName::from("is_empty"),
        ]),
        [
            PrimitiveArray::from_iter(sums).into_array(),
            BoolArray::new(BitBuffer::new_unset(spans.len()), Validity::NonNullable).into_array(),
            BoolArray::new(empty.freeze(), Validity::NonNullable).into_array(),
        ],
        spans.len(),
        Validity::from_mask(group_validity, Nullability::Nullable),
    )?;
    Ok(Some(partial.into_array()))
}

#[cfg(test)]
mod tests;
