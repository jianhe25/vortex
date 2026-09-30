<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Executor overhead work: what was left out of scope

Companion to `REPORT.md`. Everything below was measured or observed but deliberately not changed,
either because it lives in a kernel (out of bounds for this change) or because the gain did not
justify the risk yet. Numbers are from TPC-H SF1 chunks on a 4-vCPU container.

## Loop-side (in `execute_until`, not done)

- **Slot-aware kernel registration.** `TakeExecuteAdaptor` is registered for `(Dict, <every child
  encoding>)` and declines whenever `child_idx != 1`. Per dictionary chunk that is 4 of the 5 kernel
  invocations (step 2b before the push, step 2a after it, and twice after pops). Keying the registry
  by `(parent, child, slot)` turns them into 24ns misses instead of 40 to 50ns dyn calls.
- **Redundant step 2a right after `ExecuteSlot`.** Step 2b just probed the same
  `(parent, child, slot)`; the retry on the next iteration is wasted when `execute` returned the
  parent unmodified. Safe only with an `Arc::ptr_eq` guard on the parent; not safe on later
  iterations, where a decoded grandchild can make the same pair match (the Dict-RLE case).
- **Reusable stack for nested calls.** Every kernel that executes a child internally starts a fresh
  `execute_until` (66 to 141ns: `Vec`, kernel snapshot clone, `max_iterations`, final
  `as_opt` + `Canonical::from`). A scratch stack in `ExecutionCtx` plus an internal
  `execute_child::<M>` entry would cut this to tens of ns; 1 to 4 nested calls per chunk.
- **Single done check when `M` is `AnyCanonical`.** Target and canonical checks are the same call.
  Worth a few ns per iteration after the dtype-directed matcher; only meaningful with the item
  above.
- **Cheaper registry keys.** Keys are `hash_one((Id, Id))` plus a hashbrown probe (24ns). A `u32`
  pair with an identity hasher, or a per-parent small vector, would be a few ns. Only matters for
  parents that own kernels (`Dict`, scalar functions) now that the prefilter skips the rest.
- **Skip `optimize_ctx` for canonical, childless kernel results.** The reduce fixpoint is a 40 to
  75ns no-op there.
- **Lazy stats capture.** `to_array_stats()` is an `Arc` clone (15ns) on every iteration, even when
  `execute` returns `ExecuteSlot`.
- **`create_execution_ctx` got heavier** (98 to 170ns) because it now clones the `ArrayKernels`
  handle and the parent-id snapshot. Once per scan task, but `compare_struct_lt` moved +6% and may
  be creating one per field; not investigated.
- **Shared-parent `take_slot`.** When the parent `Arc` is shared, `take_slot_unchecked` allocates a
  new node with a stats clone (600ns). Rare in scans (2 of 6778 iterations on SF1); left alone.

## Kernel-side (measured, not changed)

- **`RunEnd` slice of 32 rows costs 9 to 13µs.** `(Slice, RunEnd)` slices `ends` and `values`
  through `ArrayRef::slice` (each a `SliceArray::try_new` + static optimize), then
  `run_end_canonicalize` runs two nested loops. A meta kernel that materialises the few output
  values directly would be roughly 20x cheaper. Same for `runend[sequence, sequence]`
  (`ps_partkey`).
- **`Filter` over `FSST`/`OnPair` decodes the whole chunk.** No `(Filter, FSST)` or
  `(Filter, OnPair)` kernel exists, so a 10% filter over an 8192-row comment costs 24 to 75µs (30 to
  90ns per surviving row) versus 5µs for a bit-packed column. Largest single saving in the workload
  table.
- **`Between` on `DecimalByteParts` is not pushed into the packed part.** `l_quantity BETWEEN`
  costs 520µs per 131072 rows (4 ns/row); `l_quantity <` costs 16µs (0.12 ns/row). Hits Q6/Q19
  style predicates.
- **`Like` over `Dict` values only fuses for some dictionaries.** `l_returnflag`, `l_linestatus`,
  `l_shipmode` (`shared[fsst]` values): 1.9 ns/row. `l_shipinstruct` (`shared[varbinview]`):
  18 ns/row; `o_clerk` (`shared[onpair]`): 12 ns/row; `p_type`: 13 ns/row. The `Dict` like rule
  declines for those value encodings; cause not chased.
- **`Extension` counts as canonical.** `execute::<Canonical>` on every date column
  (`ext[for[bitpacked]]`, 75 chunks) returns the compressed storage untouched in 141ns; the
  storage is decoded later by whoever unwraps it, in a second loop.
- **Kernels that recurse instead of yielding.** `FoR`, `RunEnd`, `FSST`, `OnPair`,
  `DecimalByteParts`, `ZigZag`, `Zstd`, `Shared` call `execute` on children inside the kernel. Each
  is a full `execute_until` and hides the child from parent kernels (a `(Slice, BitPacked)` kernel
  cannot fire under `FoR`). `BitPacked`, `ALP`, `Dict`, `DateTimeParts` already use
  `require_child!`/`ExecuteSlot`; converting the rest is mechanical.
- **`Cast` inside `OnPair`.** `canonicalize_onpair` widens `dict_offsets` by building a `Cast`
  scalar-fn array and executing it (a nested loop plus `optimize_ctx`) instead of widening the
  buffer.
- **`Shared` caching.** The first `execute` of a `Shared` dictionary decodes and caches in a
  `OnceLock`; per-chunk timings of `Dict` shapes are bimodal (max 4 to 5x the median). Intended,
  but it means the first chunk of a column pays for the whole dictionary.

## Writer-side observation

- Comment columns (`l_comment`, `o_comment`, `ps_comment`, `c_comment`) were written as 8192-row
  chunks; byte-size coalescing did not merge them, so they are the most numerous chunks in SF1
  (about 1000) and pay every per-chunk fixed cost most often.

## Measurement gaps

- `add_decimal_i128_nullable` (-24%) and `compare_string_lt` (-25%) held across two divan rounds
  but are too large for a loop change; not explained.
- TPC-DS was not run: its data generation needs DuckDB, which is not in this container.
- Profiler phase times carry about 60ns of timer cost each; counts are exact, times are upper
  bounds. Wall-clock numbers are always from the uninstrumented build.
- No Samply profile of the DataFusion runs was taken, since the query-level delta was inside noise.
