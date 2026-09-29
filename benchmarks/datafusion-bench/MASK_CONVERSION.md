<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# FastLanes predicate-mask conversion

A 1,024-bit logical selection bitmap becomes a 256-byte mask for one two-bit FastLanes
`u8` block. Each selected genotype needs `11` in its packed two-bit field, and FastLanes'
physical byte order differs from logical sample order. The conversion is part of the
packed `LIST_SUM(LIST_FILTER(values, predicate))` kernel.

The [standalone benchmark](mask_conversion.rs) compares the previous broadcast conversion
with the current [four-row bit transpose](mask_conversion_transpose.rs). It runs on
little-endian AArch64 with NEON and has no Cargo dependencies:

```sh
rustc --edition=2024 -O -C codegen-units=1 -g \
  benchmarks/datafusion-bench/mask_conversion.rs -o /private/tmp/mask-conversion
/private/tmp/mask-conversion --algorithm broadcast --blocks 100000 --iterations 30
/private/tmp/mask-conversion --algorithm transpose --blocks 100000 --iterations 30
/private/tmp/mask-conversion --algorithm transpose --blocks 64 --passes 1600 --iterations 30
/private/tmp/mask-conversion --algorithm transpose --blocks 103 --check-only
rustc --edition=2024 -O -C codegen-units=1 --emit=asm \
  benchmarks/datafusion-bench/mask_conversion.rs -o /private/tmp/mask-conversion.s
```

Each block receives a distinct deterministic random predicate. Both implementations
write all 256 output bytes, and the benchmark checks every byte against a scalar mapping
before and after timing. It also checks every one-hot source bit, all-zero and all-one
masks, and all sixteen four-row bit combinations at every column. Input generation,
allocation, and validation are excluded from the timer. The named conversion functions
are intentionally not inlined so their assembly can be inspected; their call and constant
setup costs differ from the fused production kernel. There is no genotype loading or
summation in this diagnostic. `--algorithm transpose-store` selects an alternative
interleaved structure-store implementation with the same output.

## Algorithm and measurements

The current conversion loads four 128-bit predicate rows at a time and exchanges the
two-bit row index with low column-index bits using two bit-swap stages. Byte and halfword
ZIPs arrange four-row selection nibbles in column order. A sixteen-entry table turns
each nibble into a byte of doubled bits, then word ZIPs produce FastLanes byte order.
The process repeats for the other four predicate rows. The fused kernel consumes these
mask vectors immediately; it does not write or reread an expanded mask buffer.

On an Apple M3 Max, macOS 15.8, September 29, 2026, 100,000 full blocks and 30 timed
iterations gave these warm-cache medians:

| Standalone conversion | Time/block | Retired instructions/block | Cycles/block |
| --- | ---: | ---: | ---: |
| Previous broadcasts | 22.040 ns | 548.112 | 82.971 |
| Current transpose | 8.456 ns | 135.060 | 30.867 |

The transpose is 2.61× faster for conversion alone and retires about 75% fewer
instructions. A 64-block cache-resident input repeated 1,600 times measured 8.120 ns/block
for transpose and 22.324 ns/block for broadcast. The small and large results are close,
which argues against bulk memory traffic dominating these warm measurements. It does
not rule out load-use latency or execution-resource pressure. Reported input/output GB/s
are logical byte rates, not measured DRAM traffic.

The named transpose function has 126 instructions in the generated assembly, including
loads, stores, constants, and return; the caller contributes to the approximately 135
retired instructions per block. The function is unrolled and has no stack spills. The
`transpose-store` variant uses fewer instructions but measured about 8.292 ns/block,
without a clear speed advantage. It remains a diagnostic option, while ordinary ZIP
interleaves are used in the production fused kernel.

macOS `proc_pid_rusage` `RUSAGE_INFO_V4` supplies the optional retired-instruction and
cycle counts. They are process-wide differences between counter reads around each timed
trial and include loop and timer overhead. Failed counter reads are reported as unavailable.
Instruction count and IPC are evidence about the complete single-threaded conversion,
not direct measurements of a particular NEON execution unit's occupancy.

## Fused filtered-sum result

The [rare-genetics benchmark](RARE_GENETICS.md) runs one filter and one per-list sum over
10,000 lists of 10,000 genotypes. It starts both paths from the same two-bit input,
decompresses inside the decoded timer, and prepares the random predicate outside both
timers. Each of 10,000 outputs is checked against an independent scalar reference.

| Direct single-filter path | 30-run median |
| --- | ---: |
| Decode to `u8`, filter, sum | 7.632 ms |
| Previous broadcast-based fused packed kernel | 3.907 ms |
| Current transpose-based fused packed kernel | 2.393 ms |

The current fused path is 1.63× faster than the previous packed path and 3.19× faster
than decoding for this isolated operation. The predicate selects 49,995,133 genotypes;
all paths produce checksum 9,142,796. A three-sample smoke workload exercises empty
selections and null sums. These are in-memory reductions, not DataFusion query times.
The normal decoder and portable fallback are unchanged. A standalone full-block fused
diagnostic also favored the transpose, but its wrapper and data differ from the list
benchmark, so its absolute time is not substituted here.

## Profiling limits

Production disassembly confirms that the full-block mask conversion and sum use NEON
instructions directly without mask-buffer stores or register spills. A 1 kHz Samply
profile of the current packed single-filter path attributed approximately 70% of its
selected execution CPU samples to the NEON block kernel, 12% to group/block handling,
12% to predicate population counting, and 5% to boundary interval masks. Those sampled
shares are not separately timed stages, and profiling increased the median from 2.393
to 2.767 ms. The unprofiled median is the latency comparison.

A CPU Counters recording showed about 54% useful retired work, 45% instruction-processing
bottleneck, and less than 1% combined delivery and discarded work on performance cores.
These are whole-core instruction-bandwidth categories. The processing category can reflect
execution resources, dependencies, or memory latency; it does **not** mean NEON units are
idle for 45% of cycles. Per-unit NEON occupancy was not measured, so saturation remains
unknown. Apple's [CPU bottleneck guide](https://developer.apple.com/documentation/xcode/addressing-cpu-bottlenecks)
describes those categories.

The optimized benchmark build, scalar reference checks, and small null-result smoke run
passed. A unit regression test covers every predicate bit's FastLanes mapping. Focused
unit tests and Clippy were not rerun after the final transpose change.
