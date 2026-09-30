<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Executor loop overhead on TPC-H SF1 chunks

Everything below was measured with `benchmarks/execute-overhead-bench` on the chunks that the
default Vortex file writer produces for TPC-H SF1 (`lineitem`, `orders`, `part`, `partsupp`,
`customer`, `supplier`), read back through the normal file scan one column at a time with
`SplitBy::Layout`, so every input is a real on-disk chunk with the encodings the compressor chose.
Machine: 4 vCPU cloud container, release profile, `--iters 200`, medians unless noted.

## Summary

- **Whole-chunk decode is not loop-bound.** Executing a full chunk (8K to 512K rows) to
  canonical costs 2.1% loop overhead over the sum of kernel time, weighted by how often each shape
  occurs in SF1 (172ms of kernel work, 3.7ms of loop). A chunk needs 4.0 loop iterations on
  average; the compressed encodings (FoR, FSST, OnPair, RunEnd, DecimalByteParts, Dict values,
  Shared) mostly decode their children with nested `execute` calls inside the kernel rather than
  yielding `ExecuteSlot`, so the loop rarely runs.
- **The overhead is in the small-array / expression regime.** The same shapes sliced to 32 rows
  spend 45 to 85% of the wall clock outside kernels; at 1024 rows it is 35 to 60%; a scalar
  function over a 1024-row slice is 40 to 70% loop. Per `execute_until` call the fixed cost is
  66ns (already canonical input) to 141ns (`Extension`), and each loop iteration adds 250 to
  350ns before any kernel runs. Nested `execute` calls from kernels pay that fixed cost again,
  one to four times per chunk.
- **Kernel lookup is cheap but mostly wasted.** 92% of `(parent, child)` registry probes miss
  (3835 lookups, 311 hits) because no compression encoding is registered as a *parent*; the
  hits are almost all `Dict` parents, where 4 of 5 invocations decline (`TakeExecuteAdaptor`
  registered per child encoding is called for the codes slot and returns `None`). A probe costs
  24ns, a decline 30 to 50ns; `optimize_ctx` after an applied kernel costs 74 to 130ns of which
  64ns is the session variable lookup.
- **The per-iteration fixed costs were (before this change):** two `AnyCanonical::matches`
  scans of up to 12 downcasts (10 to 34ns each), one registry probe per child, a `DType` clone
  and stats `Arc` clone before every `execute`, a `StatsSet` clone plus two lock acquisitions
  on every `Done`, plus 600ns per `take_slot` when the parent `Arc` is shared.
- **Loop changes implemented here** (all in `vortex-array`, no kernel touched): dtype-directed
  `AnyCanonical` matching (one downcast instead of twelve), a parent-id prefilter that skips the
  per-child kernel lookups for encodings that own no kernels, the kernel registry held in
  `ExecutionCtx` so post-kernel `optimize` skips the session lookup, stats transfer without
  cloning the set, and the dtype clone gated to debug builds. Measured on interleaved
  before/after runs: 32-row slices 18% faster (40 of 42 shapes), 1024-row slices 11%, a
  comparison over a 1024-row slice 6%, whole chunks unchanged (+0.2% on the sum, noise);
  `AnyCanonical::matches` 12 to 31ns down to 5ns; registry probes per full decode 3835 down
  to 497; loop share of whole-chunk decode 2.1% down to 1.4%.
- **What the loop cannot fix** (kernel-side suggestions, section 7): kernels that decode whole
  children for a 32-row slice (`RunEnd`, `FSST`, `OnPair`: 3 to 13µs per 32 rows), `Filter`
  over `FSST`/`OnPair` decoding the entire chunk first (30 to 90ns per surviving row), `Between`
  on `DecimalByteParts` not pushed into the bit-packed part (3 to 4 ns/row versus 0.1 ns/row for
  `<`), `Like` over `Dict` values not fusing for 3 of the 6 dictionary shapes (13 to 18 ns/row
  versus 2 ns/row), `Extension` counted as canonical so `execute::<Canonical>` on `l_shipdate`
  returns the still-compressed storage untouched (0 work, 141ns), and `Dict` kernels registered
  for the codes slot that always decline.

## 1. Method

- `gen` writes the six tables with `session.write_options().write(..)` (default strategy: 8192-row
  blocks coalesced towards 1MiB, dictionary encoding at the layout level, BtrBlocks compression).
- `harvest` scans each column with `SplitBy::Layout` and keeps the first chunk of every distinct
  encoding-tree *shape* (encoding ids and primitive types, ignoring lengths and metadata). 42
  distinct shapes came out of 60 columns.
- `bench` times `array.execute::<Canonical>(&mut ctx)` per shape, once with the chunk `Arc`
  shared (a clone is kept) and once on a deep copy whose every node is uniquely owned (what a
  scan hands to the executor).
- `workloads` builds, for every shape, the inputs a query engine actually executes: a 32-row and
  a 1024-row slice, a 10% filter, a comparison with a constant (`<` or `=`), the comparison over
  the slice and over the filter, `BETWEEN` for numeric/decimal columns and `LIKE '%special%'`
  for strings. Inputs are built outside the timed region.
- `micro` times the primitives the loop is built from.
- The `profile` feature adds thread-local timers around every phase of `execute_until`; timers
  are placed so that nested `execute_until` calls made *inside* a kernel are subtracted, so the
  phase numbers are exclusive. Each timer costs about 60ns (`Instant::now` + `elapsed`), which
  lands in the `untracked` bucket, so counts are exact but phase times are upper bounds. Wall
  clock numbers always come from the default build.
- The `trace` feature uses the existing `test_harness::trace` recorder to print the executor's
  step sequence, including declined kernel attempts.

## 2. The shapes

| idx | chunks | rows (all columns with this shape) | example rows | first source | encoding tree |
|---:|---:|---:|---:|---|---|
| 0 | 1 | 131072 | 131072 | lineitem.l_orderkey | `runend<i64>[fastlanes.for<u32>[fastlanes.bitpacked<u32>],fastlanes.bitpacked<i64>]` |
| 1 | 115 | 14593502 | 131072 | lineitem.l_orderkey | `fastlanes.bitpacked<i64>` |
| 2 | 24 | 3145728 | 131072 | lineitem.l_orderkey | `fastlanes.for<i64>[fastlanes.bitpacked<i64>[primitive<u32>,primitive<i64>,primitive<u8>]]` |
| 3 | 17 | 2228224 | 131072 | lineitem.l_orderkey | `runend<i64>[fastlanes.for<u32>[fastlanes.bitpacked<u32>],fastlanes.for<i64>[fastlanes.bitpacked<i64>]]` |
| 4 | 2 | 262144 | 131072 | lineitem.l_orderkey | `fastlanes.for<i64>[fastlanes.bitpacked<i64>[primitive<u32>,constant<i64>,primitive<u8>]]` |
| 5 | 14 | 1616543 | 102975 | lineitem.l_orderkey | `fastlanes.for<i64>[fastlanes.bitpacked<i64>]` |
| 6 | 28 | 7001215 | 262144 | lineitem.l_linenumber | `fastlanes.bitpacked<i32>` |
| 7 | 46 | 6001215 | 131072 | lineitem.l_quantity | `decimal_byte_parts.v2[fastlanes.bitpacked<i16>]` |
| 8 | 65 | 8301215 | 131072 | lineitem.l_extendedprice | `decimal_byte_parts.v2[fastlanes.bitpacked<i32>]` |
| 9 | 92 | 12002430 | 131072 | lineitem.l_discount | `decimal_byte_parts.v2[fastlanes.bitpacked<i8>]` |
| 10 | 17 | 7901215 | 524288 | lineitem.l_returnflag | `dict[fastlanes.bitpacked<u16>,shared[fsst[constant<u8>,sequence<u8>]]]` |
| 11 | 12 | 6001215 | 524288 | lineitem.l_linestatus | `dict[fastlanes.bitpacked<u16>,shared[fsst[primitive<u8>,sequence<u8>]]]` |
| 12 | 75 | 19503645 | 262144 | lineitem.l_shipdate | `ext<vortex.date[days](i32)>[fastlanes.for<i32>[fastlanes.bitpacked<i32>]]` |
| 13 | 16 | 7651215 | 524288 | lineitem.l_shipinstruct | `dict[fastlanes.bitpacked<u16>,shared[varbinview]]` |
| 14 | 13 | 6201215 | 524288 | lineitem.l_shipmode | `dict[fastlanes.bitpacked<u16>,shared[fsst[primitive<u8>,primitive<u8>]]]` |
| 15 | 731 | 5988352 | 8192 | lineitem.l_comment | `fsst[fastlanes.bitpacked<u8>,fastlanes.bitpacked<u32>]` |
| 16 | 1 | 8192 | 8192 | lineitem.l_comment | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,primitive<u16>,fastlanes.bitpacked<u8>]` |
| 17 | 26 | 204671 | 4671 | lineitem.l_comment | `fsst[fastlanes.bitpacked<u8>,primitive<u16>]` |
| 18 | 3 | 1500000 | 524288 | orders.o_clerk | `dict[fastlanes.bitpacked<u16>,shared[onpair[primitive<u16>,primitive<u8>,fastlanes.bitpacked<u16>,constant<u8>]]]` |
| 19 | 6 | 1500000 | 262144 | orders.o_shippriority | `constant<i32>` |
| 20 | 196 | 1607440 | 8192 | orders.o_comment | `fsst[fastlanes.for<u8>[fastlanes.bitpacked<u8>],fastlanes.bitpacked<u32>]` |
| 21 | 10 | 81920 | 8192 | orders.o_comment | `fsst[fastlanes.for<u8>[fastlanes.bitpacked<u8>],fastlanes.bitpacked<u32>[primitive<u16>,primitive<u32>,primitive<u8>]]` |
| 22 | 12 | 90976 | 8192 | orders.o_comment | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,primitive<u16>,fastlanes.for<u8>[fastlanes.bitpacked<u8>]]` |
| 23 | 9 | 73728 | 8192 | orders.o_comment | `onpair[fastlanes.bitpacked<u16>,fastlanes.bitpacked<u16>,primitive<u16>,fastlanes.for<u8>[fastlanes.bitpacked<u8>]]` |
| 24 | 5 | 360000 | 131072 | part.p_partkey | `sequence<i64>` |
| 25 | 2 | 5936 | 3392 | part.p_name | `fsst[fastlanes.for<u8>[fastlanes.bitpacked<u8>],primitive<u16>]` |
| 26 | 1 | 200000 | 200000 | part.p_type | `dict[fastlanes.bitpacked<u16>,shared[fsst[primitive<u8>,primitive<u16>]]]` |
| 27 | 5 | 360000 | 131072 | part.p_retailprice | `decimal_byte_parts.v2[fastlanes.for<i32>[fastlanes.bitpacked<i32>]]` |
| 28 | 6 | 786432 | 131072 | partsupp.ps_partkey | `runend<i64>[sequence<u32>,sequence<i64>]` |
| 29 | 96 | 786432 | 8192 | partsupp.ps_comment | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,fastlanes.bitpacked<u32>,primitive<u8>]` |
| 30 | 1 | 8192 | 8192 | partsupp.ps_comment | `onpair[primitive<u16>,fastlanes.bitpacked<u16>[primitive<u32>,primitive<u16>,primitive<u16>],fastlanes.bitpacked<u32>,primitive<u8>]` |
| 31 | 1 | 5376 | 5376 | partsupp.ps_comment | `onpair[fastlanes.bitpacked<u16>,fastlanes.bitpacked<u16>,fastlanes.bitpacked<u32>,primitive<u8>]` |
| 32 | 19 | 155648 | 8192 | customer.c_name | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,primitive<u16>,constant<u8>]` |
| 33 | 1 | 2544 | 2544 | customer.c_name | `onpair[primitive<u16>,primitive<u8>,fastlanes.bitpacked<u16>,constant<u8>]` |
| 34 | 2 | 16384 | 8192 | customer.c_phone | `onpair[primitive<u16>,fastlanes.for<u16>[fastlanes.bitpacked<u16>],primitive<u16>,constant<u8>]` |
| 35 | 16 | 125424 | 8192 | customer.c_phone | `fsst[constant<u8>,primitive<u16>]` |
| 36 | 12 | 98304 | 8192 | customer.c_comment | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,fastlanes.bitpacked<u32>,fastlanes.bitpacked<u8>]` |
| 37 | 6 | 49152 | 8192 | customer.c_comment | `onpair[fastlanes.bitpacked<u16>,fastlanes.bitpacked<u16>,fastlanes.bitpacked<u32>,fastlanes.bitpacked<u8>]` |
| 38 | 1 | 2544 | 2544 | customer.c_comment | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,primitive<u16>,primitive<u8>]` |
| 39 | 1 | 10000 | 10000 | supplier.s_name | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,fastlanes.bitpacked<u16>,constant<u8>]` |
| 40 | 1 | 10000 | 10000 | supplier.s_phone | `fsst[constant<u8>,fastlanes.bitpacked<u32>]` |
| 41 | 2 | 10000 | 8192 | supplier.s_comment | `onpair[fastlanes.bitpacked<u16>,fastlanes.bitpacked<u16>,primitive<u16>,fastlanes.bitpacked<u8>]` |

### 2.1 What each family does under `execute::<Canonical>`

Traces come from `trace --attempts`; "nested" means the kernel called `execute` on a child
itself instead of returning `ExecuteSlot`, so a second `execute_until` loop runs inside the
kernel. Row counts are per chunk.

**A. Bit-packed integers, no wrapper (idx 1, 6; 143 chunks, `l_orderkey`/`l_partkey`/...,
`l_linenumber`).** 2 iterations: `execute` unpacks straight to `Primitive`, next iteration sees a
canonical array and returns. 0 lookups (BitPacked owns no kernels as parent). This is the floor
of the loop for a real kernel: 45µs of unpacking for 131072 i64, 0.3µs of loop.

**B. FoR over BitPacked (idx 5, 12-storage; 14 + 75 chunks).** FoR's `execute` does not yield the
child. The unsigned-reference fast path fuses unpack and add; signed references (all TPC-H keys)
call `encoded().execute::<PrimitiveArray>` (nested loop, 2 iterations) and then add the
reference in place. 4 iterations total, 1 wasted lookup `(FoR, BitPacked)`.

**C. FoR over BitPacked with patches (idx 2, 4; 26 chunks).** Same as B but BitPacked's
`execute` first asserts patches are `Primitive` via `require_patches!`; in idx 4 the patch
values are `Constant`, so the nested loop yields `ExecuteSlot(patch_values)`, decodes a 4-row
constant, pops, and re-enters BitPacked: 7 iterations, 1 push/pop, 8 lookups all missing. The
one `take_slot`/`put_slot` pair runs on a uniquely owned parent, so no allocation.

**D. RunEnd over FoR/BitPacked (idx 0, 3; 18 chunks of `l_orderkey`) and RunEnd over Sequence
(idx 28, `ps_partkey`).** `run_end_canonicalize` executes `ends` and `values` with two nested
loops (each 2 to 4 iterations), then expands runs. 6 to 8 iterations, 2 to 4 lookups, all
misses. Decoding runs is the slowest integer shape at 3.15 ns/row.

**E. DecimalByteParts over BitPacked or FoR (idx 7, 8, 9, 27; 208 chunks of `l_quantity`,
`l_extendedprice`, `l_discount`, `l_tax`, `p_retailprice`, ...).** `assemble_decimal` executes the
most-significant part with a nested loop and widens it to the decimal width. 4 to 6 iterations.
Because unpacking i8/i16 is so fast (5 to 12µs per 131072 rows), this is the family where loop
overhead is most visible for whole chunks: 10 to 18% in the profiled build.

**F. Dict over BitPacked codes and Shared values (idx 10, 11, 13, 14, 18, 26; 62 chunks of the
low-cardinality strings `l_returnflag`, `l_linestatus`, `l_shipinstruct`, `l_shipmode`,
`o_clerk`, `p_type`).** The only family that exercises the stack and the kernel registry:
`Dict::execute` returns `ExecuteSlot(codes)`, the loop unpacks the codes (iterations 1 to 2),
pops, `Dict::execute` returns `ExecuteSlot(values)`, `Shared::execute` decodes the dictionary
once and caches it (its `OnceLock`), pops, and finally the registered `(Dict, VarBinView)`
`TakeExecuteAdaptor` gathers the strings. 8 iterations, 2 pushes/pops, 9 lookups of which 5
find kernels, 4 decline: `(Dict, BitPacked)` before the push, again from the stack after the
push, `(Dict, Primitive)` after the pop and once more after the values pop. Every decline is the
same `child_idx != 1` early return. One `optimize_ctx` after the applied kernel.

**G. FSST comments (idx 15, 17, 20, 21, 25, 35, 40; ~980 chunks of `l_comment`, `o_comment`,
`p_name`, `c_phone`, `s_phone`).** These columns were written as 8192-row chunks (the writer's
byte-size coalescing did not merge them), so they are the most numerous chunks in the data set.
`canonicalize_fsst` executes `uncompressed_lengths` with a nested loop (BitPacked or FoR over
BitPacked), decodes symbols into one buffer and builds views. 4 iterations, 2 to 3 lookups.
12 to 15 ns/row.

**H. OnPair comments and names (idx 16, 22, 23, 29-34, 36-39, 41; ~170 chunks).** Four children
(`dict_offsets`, `codes`, `codes_offsets`, `uncompressed_lengths`); `canonicalize_onpair`
executes two or three of them with nested loops, and casts `dict_offsets` to u32 through a
`Cast` scalar-fn whose `(Cast, Primitive)` kernel applies inside a nested loop. 4 to 6
iterations, 4 to 7 lookups. 10 to 20 ns/row.

**I. Constant (idx 19, `o_shippriority`) and Sequence (idx 24, `p_partkey`).** 2 iterations,
one kernel materialising the buffer. 0 lookups.

**J. Extension over FoR (idx 12, 75 chunks of `l_shipdate`, `l_commitdate`, `l_receiptdate`,
`o_orderdate`).** `AnyCanonical` includes `Extension`, so `execute::<Canonical>` returns after
one done check without touching the storage: 141ns and no decoding. Whoever consumes the
array (Arrow export, a comparison) decodes the storage later through another `execute`.

## 3. Whole-chunk regime

Median wall clock of `execute::<Canonical>` per chunk in the default build, and the chunk-weighted
loop profile from the `profile` build (counts exact, phase times upper bounds).

| Metric (chunk-weighted over the 60 columns) | Value |
|---|---:|
| `execute_until` calls (top level) | 1708 |
| loop iterations | 6778 (3.97 per call) |
| registry lookups | 3835 |
| lookups that found a kernel | 311 (8%) |
| kernel invocations that declined | 248 |
| kernel invocations that applied | 63 |
| `optimize_ctx` calls | 63 |
| total | 172.3ms |
| kernel work (`Done` executes, applied kernels, builder appends) | 168.7ms (97.9%) |
| loop overhead | 3.7ms (2.1%) |
| of which: done checks | 0.99ms |
| of which: `finalize_done` stats transfer | 0.66ms |
| of which: registry lookups | 0.30ms |
| of which: dtype/stats clone before `execute` | 0.22ms |
| of which: `execute` calls that returned `ExecuteSlot` | 0.05ms |
| of which: kernels that declined | 0.05ms |
| of which: `optimize_ctx` | 0.03ms |
| of which: take/put slot | 0.02ms |
| of which: untracked glue (mostly the timers themselves) | 1.3ms |

The shared-versus-unique tree distinction made no measurable difference at this size: only
shape 4 does a `take_slot` at all, and the 600ns allocation on a shared parent is lost in 68µs.

## 4. Small-array and expression regime

Means over all shapes of the median execute time (default build), and the exact loop counts from
the profile build.

| workload | shapes | mean ns | mean iterations | mean lookups | mean declines | mean nested calls | work share (profile build) |
|---|---:|---:|---:|---:|---:|---:|---:|
| slice32 | 42 | 2759 | 5.2 | 3.7 | 0.6 | 1.1 | 48% |
| slice1024 | 42 | 9700 | 5.2 | 3.7 | 0.6 | 1.1 | 67% |
| slice1024 > `<`/`=` const | 42 | 5500 | 6.1 | 3.9 | 0.7 | 1.4 | 58% |
| filter 10% | 42 | 78843 | 9.6 | 7.8 | 2.7 | 1.6 | 87% |
| filter 10% > `<`/`=` const | 42 | 84500 | 11.7 | 9.9 | 2.9 | 2.5 | 88% |
| `<`/`=` const (full chunk) | 42 | 78400 | 6.0 | 4.0 | 0.8 | 1.4 | 90% |
| between (full chunk) | 14 | 209610 | 6.5 | 5.0 | 0.0 | 1.8 | 93% |
| like '%special%' (full chunk) | 27 | 995049 | 6.9 | 6.2 | 0.9 | 1.8 | 98% |

Selected shapes, default build, median ns (32-row slice / 1024-row slice / `<` or `=` over the
1024-row slice):

| shape | slice32 | slice1024 | slice1024 > cmp |
|---|---:|---:|---:|
| 1 bitpacked i64 | 622 | 897 | 2258 |
| 6 bitpacked i32 (3 bits) | 548 | 710 | 1010 |
| 19 constant | 414 | 492 | 979 |
| 24 sequence | 454 | 609 | 2263 |
| 9 decimal_byte_parts[bitpacked i8] | 991 | 1093 | 1852 |
| 5 for[bitpacked] | 1190 | 1619 | 3624 |
| 2 for[bitpacked+patches] | 3642 | 3162 | 5465 |
| 12 ext[for[bitpacked]] | 160 | 186 | 4487 |
| 10 dict[bitpacked, shared[fsst]] | 1751 | 2484 | 3566 |
| 13 dict[bitpacked, shared[varbinview]] | 1890 | 2743 | 3463 |
| 0 runend[for[bitpacked], bitpacked] | 9274 | 17240 | 20908 |
| 28 runend[sequence, sequence] | 4539 | 5384 | 7380 |
| 15 fsst[bitpacked, bitpacked] | 2498 | 14169 | 3902 |
| 16 onpair[...] | 3086 | 13618 | 5080 |
| 29 onpair[..., bitpacked u32 offsets, ...] | 3215 | 22235 | 8561 |

Reading these: a 32-row bit-packed slice costs 622ns of which the unpack kernel (including its
256-byte allocation and `PrimitiveArray` construction) is about 450ns; the loop is the rest. An
already-canonical input costs 66ns, an `Extension` 141ns: that is the fixed price of entering
`execute_until` (kernel snapshot `Arc` clone, `max_iterations` read, two matcher scans, the
final `as_opt::<AnyCanonical>` + `Canonical::from`). Every nested `execute` from a kernel pays
it again.

The `runend` and string shapes show the other regime: 9µs for 32 rows of `runend` because the
`(Slice, RunEnd)` kernel builds two `SliceArray`s (each `SliceArray::try_new` + `optimize` runs
the static parent rules, visible in the trace) and `run_end_canonicalize` then runs two nested
loops; `fsst` at 1024 rows costs 14µs because slicing FSST keeps the whole 8192-row code buffer
and the kernel decodes from the slice offset; `onpair` slices decode all four children.

## 5. Cost of the loop's primitives (`micro`, default build, ns per operation)

| primitive | ns |
|---|---:|
| `AnyCanonical::matches` on `Primitive` (3rd of 12 downcasts) | 9.4 |
| `AnyCanonical::matches` on `VarBinView` (10th) | 28.7 |
| `AnyCanonical::matches` on `RunEnd` (all 12 fail) | 33.9 |
| `Primitive::matches` (one downcast) | 2.7 |
| `encoding_id() == Primitive.id()` | 0.3 |
| `kernels.has_execute_parent(p, c)` (hash of two ids + probe, miss) | 23.8 |
| `session.kernels()` (ArcSwap load + TypeId map probe + guard) | 64.3 |
| `create_execution_ctx()` | 82.7 |
| `optimize_ctx` on a canonical array (no rewrite) | 74.2 |
| `optimize_ctx` on `RunEnd` (no rewrite, 2 children) | 130.2 |
| `optimize` static rules only, canonical | 38.9 |
| stats: `to_array_stats` + `set_iter(StatsSet::from(..))` | 91.0 |
| stats: `inherit_from` | 35.0 |
| `DType::clone` (primitive) | 2.6 |
| `ArrayRef` clone (Arc) | 15.1 |
| `with_slots` on a shared root (what `take_slot_unchecked` does when the Arc is shared) | 620.2 |
| deep copy of a 2-node tree (`with_slots` + validate per node) | 1391.3 |
| `builder_with_capacity_in(i64, 8192)` | 91.7 |
| `execute::<Canonical>` on an already canonical `Primitive` | 65.9 |
| `Instant::now()` + `elapsed()` (cost of one profiler timer) | 60.4 |
| `slice(5..37)` on bitpacked / for[bitpacked] / dict / fsst / onpair / runend[sequence] | 371 / 589 / 657 / 1648 / 1077 / 185 |
| `Buffer<i64>(32).into_array()` (`PrimitiveArray` construction) | 116.6 |

Per loop iteration before this change, with a non-canonical current array and N children, the
loop therefore spent roughly: 2 × 34ns matcher scans + N × 24ns lookups + 3ns dtype clone + 15ns
stats Arc clone + (on `Done`) 91ns stats transfer + (after an applied kernel) 74 to 130ns
`optimize_ctx`, plus the `Vec` push/pop and `Option` shuffling: 150 to 350ns per iteration,
matching the 250 to 350ns per iteration the profile build attributes to non-kernel time.

## 6. Optimizing the loop

### 6.1 Implemented in this change (vortex-array only, kernels untouched)

1. **Dtype-directed `AnyCanonical` matching** (`canonical.rs`). Each logical dtype has exactly one
   canonical encoding and every canonical encoding validates its dtype kind, so `matches` and
   `try_match` now switch on `array.dtype()` and do one downcast instead of up to twelve. The
   done check runs twice per iteration (target matcher plus canonical check) and again on return,
   so this removes 20 to 70ns per iteration and 30ns per call. Semantics are unchanged: a
   `Primitive` array cannot carry a non-primitive dtype, and so on for the other eleven.
2. **Parent-id prefilter for kernel lookups** (`kernels.rs`, `executor.rs`). The registry now also
   records the set of parent ids that have any execute-parent kernel; `ExecutionCtx` snapshots it
   next to the kernel map, and steps 2a/2b check `parents.contains_key(parent.encoding_id())`
   before probing per child. Pure compression parents (`FoR`, `BitPacked`, `FSST`, `OnPair`,
   `RunEnd`, `DecimalByteParts`, `Extension`, ...) own no kernels, so the 92% of probes that
   missed now cost one probe per parent visit instead of one per child. The single-step
   executor (`Executable for ArrayRef`) applies the same prefilter.
3. **Kernel registry held in `ExecutionCtx`** (`optimizer/mod.rs`, `executor.rs`). The
   `optimize_ctx` after an applied kernel called `session.kernels()` (64ns of ArcSwap load and
   TypeId lookup) on every call; the context now clones the `ArrayKernels` handle once at creation
   and the executor calls `optimize_with_kernels`. `optimize_ctx` keeps its public behaviour.
4. **Stats transfer without cloning** (`stats/array.rs`, `executor.rs`, `array/mod.rs`).
   `finalize_done` used `set_iter(StatsSet::from(stats).into_iter())`: a read lock, a clone of the
   whole set, then a write lock. `transfer_from` reads the source under its lock, returns early
   when it is empty or shared with the target, and writes the entries directly.
5. **Debug-only dtype clone.** The `DType` cloned before every `execute` was only compared in the
   debug postcondition; it is now gated on `cfg!(debug_assertions)`.

### Measured effect

Before/after runs use the two release binaries interleaved (before, after, before, after) with
200 iterations each; every cell is the best of the two medians, which removes the drift this
shared 4-vCPU container shows between runs (unrelated primitives such as a deep tree copy moved
by up to 1.5x between rounds, so single-run comparisons are not meaningful here).

### Workloads (best-of-2 medians, geometric mean of per-shape ratios)

| workload | shapes | before mean ns | after mean ns | geomean ratio | shapes faster / slower (>3%) |
|---|---:|---:|---:|---:|---:|
| slice32 | 42 | 2683 | 2332 | 0.821 | 40 / 1 |
| slice1024 | 42 | 9661 | 9319 | 0.893 | 31 / 4 |
| slice1024>lt | 15 | 4752 | 4639 | 0.943 | 12 / 2 |
| slice1024>eq | 27 | 5691 | 5435 | 0.934 | 18 / 5 |
| filter10pct | 42 | 77306 | 77315 | 0.985 | 15 / 14 |
| filter10pct>lt | 15 | 120426 | 119586 | 1.012 | 4 / 3 |
| filter10pct>eq | 27 | 62784 | 62854 | 0.981 | 8 / 7 |
| lt_mid | 15 | 104174 | 102863 | 0.985 | 5 / 2 |
| eq_mid | 27 | 62642 | 63412 | 1.014 | 6 / 11 |
| between | 14 | 206050 | 199844 | 0.964 | 7 / 3 |
| like_%special% | 27 | 997684 | 1011313 | 1.005 | 4 / 5 |


Small arrays gained 18% (32 rows), 11% (1024 rows) and 6% (a comparison over 1024 rows), with
40 of 42 shapes faster at 32 rows. Whole chunks and the full-chunk expression workloads are
unchanged within noise, as the 2% overhead measurement predicted: the sum over all 42
whole-chunk decodes moved +0.2%. `AnyCanonical::matches` went from 12/27/31ns (primitive /
varbinview / miss) to 5ns in all three cases; `create_execution_ctx` grew from 98 to 170ns because
it now also clones the `ArrayKernels` handle and the parent-id snapshot (once per scan task).
The profiled whole-chunk decode shows the kernel lookups drop from 3835 to 497 (the 311 that find
a kernel plus one probe per `Dict`/scalar-fn parent visit), done checks halve, stats transfer
drops by a third, and total loop overhead goes from 2.1% to 1.4%.

### Per-shape small workloads (best-of-2 medians, ns)

| idx | shape | slice32 before | after | slice1024 before | after | slice1024>cmp before | after |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `runend<i64>[fastlanes.for<u32>[fastlanes.bitpacked<u32>],fastlanes.bit` | 8899 | 8329 | 12745 | 12232 | 13367 | 14456 |
| 1 | `fastlanes.bitpacked<i64>` | 569 | 512 | 829 | 783 | 2081 | 1987 |
| 2 | `fastlanes.for<i64>[fastlanes.bitpacked<i64>[primitive<u32>,primitive<i` | 2681 | 2556 | 3092 | 2968 | 4779 | 5319 |
| 3 | `runend<i64>[fastlanes.for<u32>[fastlanes.bitpacked<u32>],fastlanes.for` | 9919 | 9249 | 13781 | 13266 | 16659 | 15784 |
| 4 | `fastlanes.for<i64>[fastlanes.bitpacked<i64>[primitive<u32>,constant<i6` | 2864 | 2763 | 3335 | 3213 | 5457 | 5323 |
| 5 | `fastlanes.for<i64>[fastlanes.bitpacked<i64>]` | 1177 | 927 | 1596 | 1311 | 3591 | 3303 |
| 6 | `fastlanes.bitpacked<i32>` | 572 | 434 | 730 | 584 | 1125 | 1023 |
| 7 | `decimal_byte_parts.v2[fastlanes.bitpacked<i16>]` | 1063 | 781 | 1220 | 915 | 1921 | 1782 |
| 8 | `decimal_byte_parts.v2[fastlanes.bitpacked<i32>]` | 1077 | 784 | 1294 | 985 | 2081 | 1721 |
| 9 | `decimal_byte_parts.v2[fastlanes.bitpacked<i8>]` | 996 | 713 | 1102 | 819 | 1804 | 1646 |
| 10 | `dict[fastlanes.bitpacked<u16>,shared[fsst[constant<u8>,sequence<u8>]]]` | 1977 | 1255 | 2839 | 2133 | 3469 | 2754 |
| 11 | `dict[fastlanes.bitpacked<u16>,shared[fsst[primitive<u8>,sequence<u8>]]` | 1896 | 1267 | 2732 | 2207 | 3408 | 2750 |
| 12 | `ext<vortex.date[days](i32)>[fastlanes.for<i32>[fastlanes.bitpacked<i32` | 160 | 86 | 158 | 76 | 4286 | 3768 |
| 13 | `dict[fastlanes.bitpacked<u16>,shared[varbinview]]` | 1818 | 1282 | 2721 | 2115 | 3408 | 2800 |
| 14 | `dict[fastlanes.bitpacked<u16>,shared[fsst[primitive<u8>,primitive<u8>]` | 1893 | 1315 | 2712 | 2184 | 3407 | 2911 |
| 15 | `fsst[fastlanes.bitpacked<u8>,fastlanes.bitpacked<u32>]` | 2375 | 2273 | 14244 | 14263 | 4018 | 3704 |
| 16 | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,primitive<u16>,fastlane` | 3171 | 2602 | 14580 | 14483 | 5826 | 5417 |
| 17 | `fsst[fastlanes.bitpacked<u8>,primitive<u16>]` | 2490 | 1946 | 14964 | 12810 | 2881 | 2422 |
| 18 | `dict[fastlanes.bitpacked<u16>,shared[onpair[primitive<u16>,primitive<u` | 1927 | 1340 | 2802 | 2335 | 8069 | 7097 |
| 19 | `constant<i32>` | 420 | 309 | 487 | 398 | 973 | 850 |
| 20 | `fsst[fastlanes.for<u8>[fastlanes.bitpacked<u8>],fastlanes.bitpacked<u3` | 2760 | 2558 | 18227 | 17635 | 3609 | 3634 |
| 21 | `fsst[fastlanes.for<u8>[fastlanes.bitpacked<u8>],fastlanes.bitpacked<u3` | 3924 | 4302 | 17807 | 20531 | 5123 | 5858 |
| 22 | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,primitive<u16>,fastlane` | 3018 | 2874 | 16284 | 15866 | 6226 | 5932 |
| 23 | `onpair[fastlanes.bitpacked<u16>,fastlanes.bitpacked<u16>,primitive<u16` | 3261 | 2889 | 16256 | 15852 | 6814 | 6709 |
| 24 | `sequence<i64>` | 425 | 311 | 571 | 448 | 2310 | 2171 |
| 25 | `fsst[fastlanes.for<u8>[fastlanes.bitpacked<u8>],primitive<u16>]` | 2482 | 2510 | 15088 | 15715 | 2919 | 2904 |
| 26 | `dict[fastlanes.bitpacked<u16>,shared[fsst[primitive<u8>,primitive<u16>` | 1711 | 1284 | 2495 | 2202 | 3345 | 3030 |
| 27 | `decimal_byte_parts.v2[fastlanes.for<i32>[fastlanes.bitpacked<i32>]]` | 1735 | 1255 | 1924 | 1529 | 3962 | 3799 |
| 28 | `runend<i64>[sequence<u32>,sequence<i64>]` | 4177 | 3736 | 4941 | 4472 | 6888 | 6651 |
| 29 | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,fastlanes.bitpacked<u32` | 3093 | 2697 | 21812 | 20894 | 8726 | 9061 |
| 30 | `onpair[primitive<u16>,fastlanes.bitpacked<u16>[primitive<u32>,primitiv` | 7073 | 6472 | 26454 | 24452 | 11245 | 12554 |
| 31 | `onpair[fastlanes.bitpacked<u16>,fastlanes.bitpacked<u16>,fastlanes.bit` | 2735 | 2586 | 19757 | 21501 | 9011 | 8745 |
| 32 | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,primitive<u16>,constant` | 2668 | 2355 | 11748 | 10447 | 5890 | 5652 |
| 33 | `onpair[primitive<u16>,primitive<u8>,fastlanes.bitpacked<u16>,constant<` | 2903 | 2495 | 11939 | 11515 | 6168 | 5875 |
| 34 | `onpair[primitive<u16>,fastlanes.for<u16>[fastlanes.bitpacked<u16>],pri` | 3253 | 2733 | 14293 | 14681 | 5766 | 5984 |
| 35 | `fsst[constant<u8>,primitive<u16>]` | 2194 | 2025 | 12706 | 12375 | 4417 | 3748 |
| 36 | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,fastlanes.bitpacked<u32` | 3566 | 2530 | 18764 | 15699 | 8123 | 7179 |
| 37 | `onpair[fastlanes.bitpacked<u16>,fastlanes.bitpacked<u16>,fastlanes.bit` | 3415 | 2850 | 18548 | 17823 | 7836 | 7258 |
| 38 | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,primitive<u16>,primitiv` | 2418 | 2166 | 17147 | 17293 | 7651 | 8012 |
| 39 | `onpair[primitive<u16>,fastlanes.bitpacked<u16>,fastlanes.bitpacked<u16` | 2556 | 2166 | 11159 | 11549 | 5585 | 5180 |
| 40 | `fsst[constant<u8>,fastlanes.bitpacked<u32>]` | 2193 | 2012 | 13006 | 12564 | 4165 | 3535 |
| 41 | `onpair[fastlanes.bitpacked<u16>,fastlanes.bitpacked<u16>,primitive<u16` | 3172 | 2416 | 16860 | 16258 | 6562 | 6050 |


### Whole-chunk `execute::<Canonical>` (best-of-2 medians, ns)

| idx | source | rows | before | after | delta |
|---:|---|---:|---:|---:|---:|
| 0 | lineitem.l_orderkey | 131072 | 447706 | 481953 | +7.6% |
| 1 | lineitem.l_orderkey | 131072 | 43824 | 46306 | +5.7% |
| 2 | lineitem.l_orderkey | 131072 | 67406 | 71263 | +5.7% |
| 3 | lineitem.l_orderkey | 131072 | 460378 | 475371 | +3.3% |
| 4 | lineitem.l_orderkey | 131072 | 70069 | 69470 | -0.9% |
| 5 | lineitem.l_orderkey | 102975 | 52632 | 52303 | -0.6% |
| 6 | lineitem.l_linenumber | 262144 | 42075 | 41717 | -0.9% |
| 7 | lineitem.l_quantity | 131072 | 12088 | 11848 | -2.0% |
| 8 | lineitem.l_extendedprice | 131072 | 22179 | 21852 | -1.5% |
| 9 | lineitem.l_discount | 131072 | 6287 | 6029 | -4.1% |
| 10 | lineitem.l_returnflag | 524288 | 503617 | 521280 | +3.5% |
| 11 | lineitem.l_linestatus | 524288 | 474286 | 497695 | +4.9% |
| 12 | lineitem.l_shipdate | 262144 | 140 | 83 | -40.7% |
| 13 | lineitem.l_shipinstruct | 524288 | 507355 | 499267 | -1.6% |
| 14 | lineitem.l_shipmode | 524288 | 491559 | 489406 | -0.4% |
| 15 | lineitem.l_comment | 8192 | 104244 | 93781 | -10.0% |
| 16 | lineitem.l_comment | 8192 | 96514 | 84302 | -12.7% |
| 17 | lineitem.l_comment | 4671 | 56794 | 57221 | +0.8% |
| 18 | orders.o_clerk | 524288 | 543852 | 526842 | -3.1% |
| 19 | orders.o_shippriority | 262144 | 24186 | 22945 | -5.1% |
| 20 | orders.o_comment | 8192 | 138095 | 132594 | -4.0% |
| 21 | orders.o_comment | 8192 | 135551 | 133183 | -1.7% |
| 22 | orders.o_comment | 8192 | 110205 | 106998 | -2.9% |
| 23 | orders.o_comment | 8192 | 104562 | 106883 | +2.2% |
| 24 | part.p_partkey | 131072 | 21703 | 23757 | +9.5% |
| 25 | part.p_name | 3392 | 46173 | 45902 | -0.6% |
| 26 | part.p_type | 200000 | 163873 | 165894 | +1.2% |
| 27 | part.p_retailprice | 131072 | 33474 | 34080 | +1.8% |
| 28 | partsupp.ps_partkey | 131072 | 129682 | 109656 | -15.4% |
| 29 | partsupp.ps_comment | 8192 | 158563 | 156999 | -1.0% |
| 30 | partsupp.ps_comment | 8192 | 163318 | 158354 | -3.0% |
| 31 | partsupp.ps_comment | 5376 | 108953 | 105397 | -3.3% |
| 32 | customer.c_name | 8192 | 77227 | 76701 | -0.7% |
| 33 | customer.c_name | 2544 | 22868 | 24953 | +9.1% |
| 34 | customer.c_phone | 8192 | 96017 | 101317 | +5.5% |
| 35 | customer.c_phone | 8192 | 90932 | 89548 | -1.5% |
| 36 | customer.c_comment | 8192 | 125659 | 124503 | -0.9% |
| 37 | customer.c_comment | 8192 | 125384 | 122791 | -2.1% |
| 38 | customer.c_comment | 2544 | 42291 | 42341 | +0.1% |
| 39 | supplier.s_name | 10000 | 93594 | 91856 | -1.9% |
| 40 | supplier.s_phone | 10000 | 107238 | 111924 | +4.4% |
| 41 | supplier.s_comment | 8192 | 119731 | 117179 | -2.1% |
| | **sum** | | 6242284 | 6253744 | +0.2% |


### Micro (best-of-2, ns)

| primitive | before | after |
|---|---:|---:|
| deep_clone(example tree) [with_slots + validate] | 1362.0 | 1385.6 |
| optimize_ctx(canonical example) [no rewrite] | 79.8 | 79.9 |
| AnyCanonical::matches(primitive) [3rd branch] | 12.1 | 4.7 |
| Arc clone of ArrayRef | 16.3 | 15.5 |
| builder_with_capacity_in(i64, 8192) | 105.0 | 103.7 |
| encoding_id() == Primitive.id() | 0.7 | 0.3 |
| with_slots(shared root) [take_slot on shared arc] | 622.7 | 693.7 |
| dtype().clone() (primitive) | 2.7 | 2.7 |
| session.kernels() (ArcSwap load + TypeId probe) | 66.8 | 56.7 |
| optimize(primitive) [static rules only] | 41.9 | 40.9 |
| AnyCanonical::matches(varbinview) [10th branch] | 27.4 | 5.0 |
| Primitive::matches(primitive) | 3.1 | 2.6 |
| AnyCanonical::matches(vortex.runend) [all 12 fail] | 30.7 | 4.9 |
| create_execution_ctx() | 98.4 | 169.8 |
| kernels.has_execute_parent(vortex.runend, fastlanes.for) [session] | 24.8 | 23.0 |
| stats: to_array_stats + set_iter (canonical example) | 86.1 | 85.6 |
| execute::<Canonical>(already canonical primitive) | 68.9 | 72.7 |
| optimize_ctx(primitive) [no rewrite] | 84.5 | 79.4 |
| optimize_ctx(vortex.runend) [no rewrite] | 146.8 | 131.2 |
| stats: inherit_from (canonical <- example) | 33.9 | 33.6 |


### 6.2 Further loop changes worth doing (not implemented)

- **Slot-aware kernel registration.** `TakeExecuteAdaptor` is registered for
  `(Dict, <every encoding>)` and declines whenever `child_idx != 1`; in the dictionary shapes 4 of
  the 5 kernel invocations per chunk are that decline (from step 2b before the push, from step 2a
  after the push, and twice after pops). Letting a registration name the slot(s) it applies to
  (`register_execute_parent_kernel_for_slots`) and keying the map by `(parent, child, slot)` turns
  those into misses at 24ns instead of dyn calls at 40 to 50ns, and removes the need for every
  kernel to defend against the wrong slot. This is a registry change, not a kernel change.
- **Do not retry step 2a on the iteration right after `ExecuteSlot`.** Step 2b just probed the
  same `(parent, child, slot)` and every kernel declined; the child is unchanged when the frame is
  pushed. The retry is only redundant when `execute` returned the parent unmodified, so guard it
  with `Arc::ptr_eq` on the parent before and after `execute`. Saves one probe (and in the Dict
  case one decline) per push. It is *not* safe to skip 2a on later iterations: after a grandchild
  is decoded the same `(parent, child)` pair can newly match (the Dict-RLE case in the docs).
- **Reusable stack for nested calls.** Kernels that execute children internally start a fresh
  `execute_until` (fresh `Vec`, kernel snapshot `Arc` clone, `max_iterations` read, final
  `as_opt` + `Canonical::from`). Keeping a scratch `Vec<StackFrame>` and the snapshots in
  `ExecutionCtx` and adding an internal `execute_child::<M>` entry that reuses them would cut the
  66 to 141ns fixed cost of the 1 to 4 nested calls per chunk to a few tens of ns.
- **Single done check.** When `M` is `AnyCanonical` the target and canonical checks are the same
  call; a `const IS_CANONICAL: bool` on `Matcher` (or a `TypeId` compare) lets the loop do one.
  After change 1 each check is a few ns, so this is only worth it together with the reusable stack.
- **Cheaper registry keys.** Keys are `hash_one((Id, Id))` through `DefaultHashBuilder`, then a
  hashbrown probe. Ids are interned `u32`s; a key of `(parent << 32) | child` with an identity
  hasher, or a per-parent small vector of `(child, kernels)` scanned linearly, would make a probe
  a few ns. Most probes are now avoided by change 2, so this matters only for parents that do own
  kernels (`Dict`, the scalar functions).
- **Skip `optimize_ctx` when the kernel result is canonical.** After an applied kernel the loop
  runs the reduce/reduce-parent fixpoint on the result; for a canonical result with no children
  the pass is a no-op costing 40 to 75ns. A cheap `AnyCanonical::matches && slots().is_empty()`
  check avoids it.
- **Lazy stats capture.** `to_array_stats()` is an `Arc` clone per iteration even when `execute`
  returns `ExecuteSlot`; capturing it only on the `Done` path needs the encoding to hand the
  original stats back, or the loop to keep a borrowed copy of the `ArrayStats` handle (which is
  just an `Arc`, so the saving is the 15ns refcount pair).

### 6.3 Where the time really goes for small arrays

Even with every item above, `execute_until` on a 32-row bit-packed slice will not drop far below
500ns, because the kernel itself allocates a buffer and constructs a `PrimitiveArray` (about
120ns), and `slice()` on the input already cost 370ns before execution started (`SliceArray::try_new`,
`optimize` with static parent rules, stats inheritance). The loop overhead is real but it is
one third of a small decode, not most of it. The larger constant is the *number* of
`execute_until` activations per chunk: nested kernel calls, `Shared` values, `Cast` inside
`OnPair`, each paying the fixed entry cost and a fresh set of done checks.

## 7. Kernel-side suggestions (slow kernels and meta kernels)

These are the places where the measurements say the loop is executing far more than it needs to,
but the fix lives in a kernel or a registration, not in the loop.

1. **`RunEnd` slice of 32 rows costs 9 to 13µs.** `(Slice, RunEnd)` slices `ends` and `values`
   through `ArrayRef::slice` (each a `SliceArray::try_new` + static optimize, 185 to 590ns), then
   `run_end_canonicalize` executes both children with nested loops and expands. A meta kernel
   for `(Slice, RunEnd)` that binary-searches the run range and directly materialises the
   32 output values from the (already tiny) sliced values would be 20x cheaper. Same for the
   `runend[sequence, sequence]` shape of `ps_partkey`.
2. **`Filter` over `FSST`/`OnPair` decodes the entire chunk.** No `(Filter, FSST)` or
   `(Filter, OnPair)` kernel is registered, so a 10% filter over an 8192-row comment costs 24 to
   75µs (30 to 90ns per surviving row) versus 5µs for the 820 rows of a bit-packed column. A
   filter kernel that gathers `codes_offsets`/`uncompressed_lengths` for the selected rows and
   decodes only those codes (FSST decode is per-string) is the single largest saving in the
   workload table.
3. **`Between` on `DecimalByteParts` is not pushed into the packed part.** `l_quantity BETWEEN`
   costs 520µs per 131072 rows (4 ns/row) while `l_quantity <` costs 16µs (0.12 ns/row) because
   the comparison is pushed through to the bit-packed i16/i32 part but `Between` decodes to a
   128-bit decimal first. Registering the `Between` parent rule/kernel for `DecimalByteParts`
   (or reducing `Between` to two comparisons when the child is `DecimalByteParts`) recovers the
   30x.
4. **`Like` over `Dict` values only fuses for some dictionaries.** For `l_returnflag`,
   `l_linestatus`, `l_shipmode` (`shared[fsst]` values) `LIKE` costs 1.9 ns/row, but for
   `l_shipinstruct` (`shared[varbinview]`, 18 ns/row), `o_clerk` (`shared[onpair]`, 12 ns/row) and
   `p_type` (13 ns/row) it evaluates the pattern per row instead of per dictionary entry. The
   `Dict` `like` rule should apply to any values child; worth a look at why it declines here.
5. **`Extension` is canonical, so `execute::<Canonical>` on dates does nothing.** Shape 12
   (`ext[for[bitpacked]]`, 75 chunks of every date column) returns in 141ns without decoding; the
   storage is decoded later by whoever unwraps it, through a second `execute`. A `Canonical`
   target that also requires canonical storage (or an `Extension` `execute` that yields
   `ExecuteSlot(storage)`) would make this family decode once, in the same loop.
6. **Kernels that recurse instead of yielding.** `FoR`, `RunEnd`, `FSST`, `OnPair`,
   `DecimalByteParts`, `ZigZag`, `Zstd` and `Shared` call `execute` on children inside the kernel.
   Each nested call is a full `execute_until` (66 to 141ns fixed) and hides the child from the
   parent-kernel machinery (a `(Slice, BitPacked)` kernel cannot fire on `FoR`'s child because
   `FoR` never yields it). Converting these to `require_child!`/`ExecuteSlot` (as `BitPacked`,
   `ALP`, `Dict`, `DateTimeParts` already do) makes the loop's own optimisations apply to them
   and removes the nested entry cost. This is a mechanical change per kernel.
7. **`Dict` kernels registered for the codes slot.** See 6.2: `TakeExecuteAdaptor(BitPacked)`,
   `(Primitive)` etc. are invoked for slot 0 and always decline. Registering them for slot 1 only
   removes 4 dyn calls per dictionary chunk.
8. **`Cast` inside `OnPair`.** `canonicalize_onpair` casts `dict_offsets` to u32 by building a
   `Cast` scalar-fn array and executing it (a nested loop whose `(Cast, Primitive)` kernel
   applies). A direct widening of the offsets buffer avoids one activation and one `optimize_ctx`.
9. **`Shared` caching hides cost from benchmarks and from itself.** The first `execute` of a
   `Shared` dictionary decodes and caches; every later `execute` returns the cached `ArrayRef`
   after one done check. In a scan that is the intended win, but it means the first chunk of a
   column pays for the whole dictionary decode and any per-chunk measurement of `Dict` shapes is
   bimodal (the `max` in the timing distribution is 4 to 5x the median).

## 8. Files

- `benchmarks/execute-overhead-bench/` — the harness (`gen`, `harvest`, `bench`, `workloads`,
  `micro`, `trace`), see its README.
- `vortex-array/src/exec_profile.rs` + `exec-profile` feature — the thread-local phase profiler
  used by `--features profile`; compiled out by default.
- `vortex-array/src/canonical.rs`, `executor.rs`, `optimizer/mod.rs`, `optimizer/kernels.rs`,
  `stats/array.rs`, `array/mod.rs` — the loop changes of section 6.1.

## 9. Real TPC-H queries (DataFusion, SF1, vortex files)

`datafusion-bench tpch --formats vortex` on the same 4-vCPU container, baseline built from
`develop` (50c3f9f) and the branch (0186706), three interleaved rounds (5 + 10 + 10 iterations,
25 samples per query), all 22 queries. Data generated with `data-gen tpch --formats vortex`.

| | baseline | branch | delta |
|---|---:|---:|---:|
| sum of per-query medians | 1713ms | 1724ms | +0.6% |
| sum of per-query minimums | 1514ms | 1497ms | -1.1% |

No query moved outside the run-to-run noise of this machine: the first 5-iteration round alone
read -3.7% on the median sum with single queries swinging between -32% and +45%, and merging
the two 10-iteration rounds pulled every query back to within about ±6%, both directions. This
matches section 3: whole-chunk decode spends about 2% of its time in the loop, DataFusion scans
execute whole chunks, and a 30% cut of that 2% is invisible at query level. The loop savings
only show where `execute_until` runs on small arrays or many activations (section 4), which the
TPC-H scan path does not do. The kernel-side items in section 7 (filter over FSST/OnPair,
`Between` on decimals, `Like` over `Dict`, `RunEnd` slices) are the ones that would move queries.

## 10. Existing `vortex-array` divan benches

Ten of the repository's own benches that call `execute` were run on `develop` and on the branch
(`cargo bench -p vortex-array --bench <name>`, same target dir). `binary_ops`, `compare`,
`take_primitive` and `filter_fixed_width` were run twice interleaved; those cells are the best of
the two medians. The single-run cells for the other six carry the usual ±5% noise of this box.

| bench | develop (best of 2 medians) | branch | delta |
|---|---:|---:|---:|
| binary_ops/add_decimal_i64_nonnull | 7.20 µs | 7.14 µs | -0.8% |
| binary_ops/add_decimal_i128_nullable | 24.09 µs | 18.19 µs | -24.5% ** |
| binary_ops/add_i32_nonnull | 13.48 µs | 13.56 µs | +0.6% |
| binary_ops/add_i64_nonnull | 15.36 µs | 15.33 µs | -0.2% |
| binary_ops/add_i64_nullable | 16.44 µs | 16.30 µs | -0.9% |
| binary_ops/add_u32_nonnull | 12.86 µs | 12.78 µs | -0.6% |
| binary_ops/and_bool_nullable | 1.85 µs | 1.79 µs | -3.7% |
| binary_ops/div_decimal_i64_nonnull | 12.59 µs | 12.44 µs | -1.2% |
| binary_ops/div_decimal_i128_nullable | 59.38 µs | 58.16 µs | -2.1% |
| binary_ops/div_i64_nonnull | 69.71 µs | 69.65 µs | -0.1% |
| binary_ops/div_i64_nullable | 56.56 µs | 56.63 µs | +0.1% |
| binary_ops/lt_i64_nullable | 5.70 µs | 5.57 µs | -2.3% |
| binary_ops/mul_decimal_i64_nonnull | 2.44 µs | 2.28 µs | -6.5% ** |
| binary_ops/mul_decimal_i128_nullable | 16.26 µs | 15.86 µs | -2.5% |
| binary_ops/mul_i8_nonnull | 54.33 µs | 54.06 µs | -0.5% |
| binary_ops/mul_i16_nonnull | 27.13 µs | 27.16 µs | +0.1% |
| binary_ops/mul_i32_nonnull | 47.01 µs | 47.12 µs | +0.2% |
| binary_ops/mul_i32_nullable | 48.22 µs | 47.81 µs | -0.9% |
| binary_ops/mul_i64_nonnull | 24.75 µs | 24.41 µs | -1.4% |
| binary_ops/mul_u8_nonnull | 40.04 µs | 39.78 µs | -0.6% |
| binary_ops/mul_u16_nonnull | 12.93 µs | 12.77 µs | -1.2% |
| binary_ops/mul_u32_nonnull | 24.22 µs | 24.15 µs | -0.3% |
| binary_ops/mul_u64_nonnull | 18.55 µs | 18.51 µs | -0.2% |
| binary_ops/subtract_shapes/32768, | 15.67 µs | 16.22 µs | +3.5% |
| compare/compare_bool | 1.39 µs | 1.24 µs | -10.8% ** |
| compare/compare_bool_constant | 1.18 µs | 1.02 µs | -13.7% ** |
| compare/compare_bool_nullable | 1.75 µs | 1.54 µs | -11.7% ** |
| compare/compare_decimal | 8.47 µs | 8.33 µs | -1.7% |
| compare/compare_f32 | 9.46 µs | 9.39 µs | -0.8% |
| compare/compare_float | 13.26 µs | 13.18 µs | -0.6% |
| compare/compare_int | 7.03 µs | 6.90 µs | -1.8% |
| compare/compare_int_constant | 6.39 µs | 6.29 µs | -1.6% |
| compare/compare_int_eq | 7.09 µs | 6.92 µs | -2.4% |
| compare/compare_int_nullable | 7.46 µs | 7.34 µs | -1.7% |
| compare/compare_string_constant | 7.33 µs | 7.19 µs | -1.9% |
| compare/compare_string_eq | 7.26 µs | 7.14 µs | -1.7% |
| compare/compare_string_lt | 10.13 µs | 7.56 µs | -25.4% ** |
| compare/compare_struct_eq | 33.18 µs | 33.76 µs | +1.7% |
| compare/compare_struct_lt | 33.26 µs | 35.38 µs | +6.4% ** |
| compare/compare_u8 | 1.45 µs | 1.31 µs | -9.6% ** |
| compare/compare_u64 | 7.03 µs | 6.93 µs | -1.4% |
| take_primitive/primitive_take_u32 | 12.18 µs | 12.23 µs | +0.4% |
| take_primitive/1000 | 997 ns | 855 ns | -14.2% ** |
| take_primitive/10000 | 3.15 µs | 3.04 µs | -3.6% |
| take_primitive/25000 | 6.61 µs | 6.51 µs | -1.5% |
| filter_fixed_width/0.01 | 1.03 µs | 913 ns | -11.3% ** |
| filter_fixed_width/0.5 | 3.41 µs | 3.31 µs | -2.8% |
| filter_fixed_width/0.8 | 4.72 µs | 4.59 µs | -2.8% |
| filter_fixed_width/0.95 | 3.25 µs | 3.11 µs | -4.5% |

Single-run benches:

| bench | develop median | branch median | delta |
|---|---:|---:|---:|
| cast_primitive/8192 | 7.78 µs | 7.66 µs | -1.5% |
| dict_compare/bench_compare_varbinview/10000, | 10.63 µs | 11.86 µs | +11.6% ** |
| dict_compare/bench_compare_varbinview/50000, | 26.29 µs | 27.64 µs | +5.1% ** |
| dict_mask/bench_dict_mask/0.1, | 28.39 µs | 29.57 µs | +4.2% |
| dict_mask/bench_dict_mask/0.01, | 27.91 µs | 25.64 µs | -8.1% ** |
| dict_mask/bench_dict_mask/0.5, | 27.11 µs | 30.20 µs | +11.4% ** |
| dict_mask/bench_dict_mask/0.9, | 31.06 µs | 29.02 µs | -6.6% ** |
| scalar_subtract/scalar_subtract | 10.75 µs | 10.06 µs | -6.4% ** |
| slice_dict_primitive/10000 | 45.17 µs | 41.79 µs | -7.5% ** |
| take_patches/take_search_chunked/0.1, | 57.78 µs | 55.61 µs | -3.8% |
| take_patches/take_search_chunked/0.01, | 44.63 µs | 39.88 µs | -10.6% ** |
| take_patches/take_search_chunked/0.005, | 36.38 µs | 35.45 µs | -2.6% |

Reading: benches that execute a small array or expression per iteration (about 1µs) gain 10 to
14% (`compare_bool*`, `compare_u8`, `take_primitive/1000`, `filter_fixed_width/0.01`,
`scalar_subtract`, `slice_dict_primitive`, `take_patches/0.01`); benches dominated by a
multi-microsecond kernel move 1 to 3%. Two cells stayed outside noise in both rounds without an
obvious loop-side explanation (`add_decimal_i128_nullable` -24%, `compare_string_lt` -25%) and one
went the other way (`compare_struct_lt` +6%, plausibly the 70ns heavier `create_execution_ctx`
if the bench creates a context per field); they are worth a dedicated rerun before being quoted.
