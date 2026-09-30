# Review of the "left out of scope" claims

Each claim from the executor-overhead out-of-scope notes was checked against the code on this
branch (`ji/relaxed-hopper-atxql4`, which is at the same commit as `develop`; the modified
`execute_until` and `REPORT.md` are not on it) and, where a number was given, re-measured with
`vortex/examples/probe_kernels.rs` and `vortex/examples/probe_writer.rs` (release build, 4 vCPU
container, synthetic TPC-H-shaped data, median of 30 runs). The loop-side section was skipped at
the user's request.

Legend: **Confirmed** = mechanism and number hold. **Partly** = the observation holds but the
stated cause does not. **Disproved** = the statement is wrong on this tree.

## Kernel-side

### 1. `RunEnd` slice of 32 rows costs 9 to 13µs — Confirmed (number and path), one wording nit

- Path: `(Slice, RunEnd)` is an execute-parent kernel, `encodings/runend/src/kernel.rs:33-86`. It
  slices `ends` and `values` with `ArrayRef::slice` (lines 79-80), and `ArrayRef::slice`
  (`vortex-array/src/array/erased.rs:213-250`) builds `SliceArray::try_new(..).optimize()`,
  where `optimize()` is the static-rules-only fixpoint (`optimizer/mod.rs:62`). The result is then
  canonicalised by `run_end_canonicalize` (`encodings/runend/src/array.rs:479-517`).
- Measured, 32-row slice at row 65000 of a 131072-row, 887-run u8 column:

  | shape | slice + execute |
  |---|---|
  | `runend[primitive, primitive]` | 2.98µs |
  | `runend[bitpacked, bitpacked]` (what the compressor emits) | 13.1µs |
  | `dict[runend[bitpacked, bitpacked]]` (full compressor output) | 14.6µs |
  | primitive slice, same 32 rows (baseline) | 0.34µs |
  | whole 131072-row `runend` to canonical | 7.4µs |

  So the 9 to 13µs figure is reproduced, and a 32-row slice costs about twice a full-chunk decode.
- Nit: `run_end_canonicalize` has no two nested loops. It is one loop over runs with a bulk
  `push_n_unchecked` per run (`encodings/runend/src/compress.rs:206-271`). The cost is in slicing
  and decoding the two bit-packed children through separate `Slice` executions, not in the decode
  loop.

### 2. `Filter` over `FSST`/`OnPair` decodes the whole chunk; no kernel exists — Disproved on the cause, cost confirmed

- Both kernels exist and are registered by default:
  `encodings/fsst/src/kernel.rs:27` and `encodings/onpair/src/kernel.rs:22`, wired in through
  each crate's `initialize()`, which `vortex_file::register_default_encodings`
  (`vortex-file/src/lib.rs:171-174`) calls from `VortexSession::default()`
  (`vortex/src/lib.rs:355`). The probe confirms
  `has_execute_parent(Filter, FSST)` and `(Filter, OnPair)` are both true in a default session.
- Neither kernel decodes first. FSST filters the VarBin codes and keeps the symbol table
  (`encodings/fsst/src/compute/filter.rs:18-45`); OnPair filters codes as a `ListArray` and keeps
  `dict_bytes`/`dict_offsets` (`encodings/onpair/src/compute/filter.rs:33-77`).
- Measured, 8192-row comment column, 10% mask (786 survivors):

  | path | time | per surviving row |
  |---|---|---|
  | `fsst.filter(mask)` then execute | 22.2µs | 28ns |
  | decode whole FSST chunk, then filter canonical | 91.5µs | 116ns |
  | compressor output (`onpair[..]`) filter then execute | 39.4µs | 50ns |
  | `bitpacked` filter then execute | 3.2µs | 4ns |
  | canonical `varbinview` filter | 1.5µs | 2ns |

  The 24 to 75µs / 30 to 90ns-per-row observation is reproduced (22 to 39µs, 28 to 50ns), but
  the kernel is firing: full decode would be four times slower. The cost is the per-row VarBin
  code copy plus decoding the survivors, so the proposed saving is real but smaller than "avoid
  a full decode" suggests.
- The only way to hit the described full-decode path is a session without the FSST/OnPair
  `initialize()` calls (for example a bare `array_session()`), in which case Filter's own
  `execute` canonicalises the child (`vortex-array/src/arrays/filter/vtable.rs:179`).

### 3. `Between` on `DecimalByteParts` is not pushed into the packed part — Confirmed

- The encoding registers only `CompareExecuteAdaptor`
  (`encodings/decimal-byte-parts/src/decimal_byte_parts/compute/kernel.rs:12-18`); its static
  rules are Cast, Filter, Mask, Slice, Take (`rules.rs:13-19`). There is no Between adaptor.
- `Between::execute` canonicalises a non-canonical input before evaluating
  (`vortex-array/src/scalar_fn/fns/between/mod.rs:308-327`), and nothing rewrites Between into
  two compares before that point (`as_two_compares` is only used for a null bound and as the
  post-canonicalise fallback).
- Measured on a 131072-row decimal(15,2) column compressed to
  `decimal_byte_parts[dict[bitpacked]]`:

  | expression | time | ns/row |
  |---|---|---|
  | `BETWEEN 5.00 AND 20.00` | 636µs | 4.85 |
  | `< 20.00` | 65µs | 0.50 |
  | `>= 5.00 AND <= 20.00` as two compares | 128µs | 0.98 |
  | canonical decimal `BETWEEN` (baseline) | 538µs | 4.10 |

  Matches the reported 4 ns/row vs 0.12 ns/row ordering; the Between cost is the full decode
  plus the canonical decimal kernel. The compare kernel also declines when lower parts are
  present (`compute/compare.rs:47-49`), so any Between pushdown would inherit that limit.

### 4. `Like` over `Dict` only fuses for some dictionaries because the rule declines for those value encodings — Cause disproved, observation not reproduced

- `LikeReduce for Dict` (`vortex-array/src/arrays/dict/compute/like.rs:19-49`) declines only when
  `values.len() > codes.len()` or the pattern is not constant. It never inspects the values
  encoding, and neither does the generic `DictionaryScalarFnValuesPushDownRule`
  (`dict/compute/rules.rs:96-208`: Pack, Cast, values longer than codes, fallible fn with
  unreferenced values, non-constant siblings, non-strict fn with null codes).
- Measured, 8192 codes, `LIKE '%AIR%'` on a 7-value dictionary and `LIKE '%PERSON%'` on a
  4-value dictionary, values wrapped four ways:

  | values encoding | l_shipmode | l_shipinstruct |
  |---|---|---|
  | `varbinview` | 0.74 ns/row | 0.74 ns/row |
  | `shared[varbinview]` | 0.90 | 0.79 |
  | `fsst` | 1.05 | 1.12 |
  | `shared[fsst]` | 0.79 | 0.79 |
  | full compressor output | 1.09 | 0.88 |

  Every variant optimises to `dict[codes, like[values]]` and runs at about 1 ns/row. The 12 to
  18 ns/row seen for `l_shipinstruct`, `o_clerk` and `p_type` cannot come from the values encoding.
  Candidates that are consistent with the code: the chunk's codes being shorter than the shared
  dictionary (the `values.len() > codes.len()` decline fires on short or sliced chunks), or the
  expression reaching the array in a shape other than a constant-pattern `Like` directly over the
  `Dict`. Neither was checked against the SF1 files, which are not in this container.

### 5. `Extension` counts as canonical — Confirmed

- `impl Matcher for AnyCanonical` ends with `|| array.is::<Extension>()`
  (`vortex-array/src/canonical.rs:1251`) and never looks at the storage child. The Extension
  vtable's `execute` is `ExecutionResult::done(array)` (`arrays/extension/vtable/mod.rs:187-189`).
  Only `CanonicalValidity` and `RecursiveCanonical` execute the storage.
- Measured: `execute::<Canonical>` on `ext[for[bitpacked]]` returns the same tree in 176ns;
  executing the storage child separately costs 2.9µs for 8192 rows, paid later by whoever
  unwraps it.
- Knock-on: `Dict` requires its values to be `AnyCanonical` via `require_child!`, so an Extension
  values child is accepted undecoded.

### 6. Kernels that recurse instead of yielding — Confirmed

| encoding | style | evidence |
|---|---|---|
| FoR | direct `execute` on children | `encodings/fastlanes/src/for/array/for_decompress.rs:84,108,113` |
| RunEnd | direct | `encodings/runend/src/array.rs:483-513` (`execute_as` on ends and values) |
| FSST | direct | `encodings/fsst/src/canonical.rs:81-83` |
| OnPair | direct | `encodings/onpair/src/canonical.rs:70,113` and `decode.rs:24-33` |
| DecimalByteParts | direct | `encodings/decimal-byte-parts/src/decimal_byte_parts/assemble.rs:92,109,114-117` |
| ZigZag | direct | `encodings/zigzag/src/array.rs:139-143` |
| Zstd | direct, single-step `execute::<ArrayRef>` | `encodings/zstd/src/array.rs:270-280` |
| Shared | direct, cached | `vortex-array/src/arrays/shared/vtable.rs:121-125` |
| BitPacked | `require_patches!` / `require_validity!` | `encodings/fastlanes/src/bitpacking/vtable/mod.rs:279-291` |
| ALP | `require_child!` + `require_patches!` | `encodings/alp/src/alp/array.rs:178-190` |
| Dict | `require_child!` | `vortex-array/src/arrays/dict/vtable/mod.rs:202,211` |
| DateTimeParts | `require_child!` | `encodings/datetime-parts/src/array.rs:189-200` |

None of the eight recursive crates contains `require_child!` or `execute_slot`. The claim that
converting them "is mechanical" was not tested; Shared's `OnceLock` cache and the FSST/OnPair
decode plans (which read children as buffers) are the parts that are not a one-line swap.

### 7. `Cast` inside `OnPair` — Confirmed, with two caveats

- `collect_widened` (`encodings/onpair/src/decode.rs:24-33`) does
  `arr.cast(dtype)?.execute::<PrimitiveArray>(ctx)`, and `ArrayRef::cast`
  (`vortex-array/src/builtins.rs:172-177`) builds a `Cast` scalar-fn array and runs static
  `optimize()` on it, so the nested loop plus optimize is real.
- Caveats: `dict_offsets` is widened once and cached in a `OnceLock`
  (`encodings/onpair/src/array.rs:263-277`), so only the first decode of a dictionary pays it;
  `ArrayRef::cast` returns `self` when the dtype already matches, so an already-u32
  `dict_offsets` builds no `Cast`. The `codes` child is widened to u16 the same way on every
  decode (`canonical.rs:113`), which is the recurring cost.

### 8. `Shared` caching — Confirmed

`SharedData` holds `Arc<OnceLock<..>>` (`vortex-array/src/arrays/shared/array.rs:35-38`),
`get_or_compute` fills it once (lines 55-63, errors are cached too), and the vtable's `execute`
canonicalises the source inside that closure (`vtable.rs:121-125`). First-chunk-pays-for-all
follows directly; the bimodal per-chunk timing was not re-measured.

## Writer-side

### Comment columns written as 8192-row chunks; byte-size coalescing did not merge them — Confirmed, and the cause is identified

- The coalescing pass is the second `RepartitionStrategy` in `vortex-file/src/strategy.rs:192-206`
  with `block_size_minimum` = 1MiB. Its `ChunksBuffer` sizes each chunk with `chunk.nbytes()`
  (`vortex-layout/src/layouts/repartition.rs:250-262`), and `nbytes()` sums every buffer
  reachable from the array (`vortex-array/src/array/erased.rs:474-481`).
- A `VarBinView` slice keeps the whole parent data buffers, so an 8192-row slice of a large input
  batch reports the entire batch's bytes. `vortex-bench` feeds 524288-row batches
  (`vortex-bench/src/tpch/tpchgen.rs:66`, `batch_size: 8192 * 64`), the first repartition pass
  slices them into 8192-row blocks (`repartition.rs:145`), and every block then trips the 1MiB
  minimum on its own. Fixed-width columns are unaffected because a primitive slice narrows its
  buffer.
- Probe (`probe_writer`, 524288 rows of 27-byte distinct strings, default write options):

  | input batches | chunks in the column's `chunked` layout |
  |---|---|
  | one 524288-row batch | 64 (one per 8192 rows) |
  | 64 slices of that batch, 8192 rows each | 64 |
  | 64 independently built 8192-row batches | 22 (about 24K rows each) |

  The coalescing works as designed when the per-chunk byte count is honest. A fix would be to
  size by the sliced view range (or by a per-slice estimate) rather than by whole buffers.

## Measurement gaps

- `add_decimal_i128_nullable` and `compare_string_lt` exist as divan benches
  (`vortex-array/benches/binary_ops.rs:280`, `vortex-array/benches/compare.rs:266`). Their deltas
  depend on the loop change, which is not on this branch, so they could not be re-run.
- TPC-DS generation needs DuckDB: confirmed. `vortex-bench/src/tpcds/duckdb.rs:38` shells out to a
  `duckdb` binary and `vortex-sqllogictest/slt/tpcds/generate_data.sh` uses `uvx --with duckdb`.
  Neither the binary nor the Python module is present in this container.
- The 60ns timer cost, the "counts are exact" statement, and the absence of a Samply profile
  describe the profiler build, which is not on this branch, so they were not checked.

## Not checked

The loop-side section (`execute_until` items) was skipped at the user's request. The current
`vortex-array/src/executor.rs` also does not contain the prefilter, parent-id snapshot, or
`ArrayKernels` clone that those items refer to, so they describe a tree other than this one.
