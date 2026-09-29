<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Rare-disease allele-count benchmark

This synthetic workload counts alternate alleles in cases and controls by gene. It is a
performance model, not a disease-association analysis: it has no missing calls, ancestry,
relatedness, or pathogenicity annotation. The default input has 10,000 variants × 10,000
samples, represented as `fixed_size_list<struct<genotype: u8, is_case: bool>>`. Case status
alternates by sample and stays fixed across variants. One quarter of variants have 20–50%
alternate-allele frequency; the others have 0.1–1%. Two deterministic Bernoulli draws
produce each diploid genotype dosage. Twenty disjoint gene intervals partition the variants.

The [benchmark source](src/bin/rare_genetics.rs) generates the input, writes a Vortex file,
and runs DataFusion's native `list_filter`, `array_transform`, and `list_sum` functions.
The query joins variants to genes by contig and half-open position interval, then groups
case and control dosage sums by gene. Summing dosage counts alternate alleles, rather than
carriers. DataFusion coerces the list-sum input to floating-point values in this baseline.

## Run

```sh
cargo build -p datafusion-bench --bin rare-genetics --profile release_debug
target/release_debug/rare-genetics --mode all --iterations 5 --explain
target/release_debug/rare-genetics --mode direct --direct-operation filter-sum --iterations 30
```

The default Databricks SQL dialect enables lambda syntax. `--variants`, `--samples`,
`--genes`, `--threads`, and `--path` adjust the workload. `--mode sql-sum` isolates native
DataFusion per-variant list sum. An equivalent query that preaggregates each variant
before the gene join is in [preaggregate.sql](src/bin/rare_genetics/preaggregate.sql):

```sh
target/release_debug/rare-genetics --mode sql \
  --query="$(cat benchmarks/datafusion-bench/src/bin/rare_genetics/preaggregate.sql)"
```

Run timing commands serially on an idle machine. Generation, writing, planning, one warmup,
physical-plan reset, and result validation are outside the timed SQL executions. Direct
Vortex modes prepare the two-bit packed array and predicate lists outside the timer and
validate every result after it. In the decoded direct paths, decompression occurs inside
each timed execution. The SQL and direct measurements have **different scopes**: direct
modes exclude file I/O, nested struct extraction, gene join, and final group-by. The
file writer chooses encodings adaptively; only direct modes force two-bit packing.
Expression projection pushdown is disabled for SQL, while ordinary column pruning and
predicate pushdown remain enabled. This checkout has opt-in computed projection pushdown,
but its converter does not yet translate these higher-order list expressions.

## Measured results

Local warm-cache measurements on an Apple M3 Max running macOS 15.8, September 29, 2026.
DataFusion 55.1.0 used 14 target partitions and the `release_debug` Cargo profile.
They were recorded on the branch originally based at `48985d564`; performance should be
remeasured after rebasing onto newer `develop` commits.
The SQL rows are five-run medians; the single-filter rows are 30-run medians. All generated
counts matched independent reference values. The Vortex file occupied 37,488,988 bytes.

| Path | Median | Scope |
| --- | ---: | --- |
| Original DataFusion case/control SQL | 53.805 ms | Vortex file through gene grouping |
| Preaggregate before gene join | 37.536 ms | Same SQL result, different plan |
| DataFusion per-variant list sum | 18.486 ms | Vortex file, genotype extraction, sum |
| Vortex decode to `u8`, then list sum | 3.586 ms | In-memory per-variant sum |
| Vortex two-bit packed list sum | 1.332 ms | Same direct sum, 2.69× faster |

The original and preaggregated SQL queries both produced 9,141,589 case and 9,143,803
control alternate alleles. The packed list-sum rate is about 75 billion genotypes/s on
one thread; the decoded direct rate is about 28 billion genotypes/s. These are in-memory
reduction rates, not end-to-end query rates. A 1 kHz profile of the original SQL attributed
about 15% of sampled CPU to DataFusion `array_sum`, 35% to Arrow filtering, 27% to Arrow
gathers, and 10% to numeric casts. The shares are inclusive across threads and may overlap;
they are not wall-time fractions. This suggests pushing down the complete per-variant
filter-and-reduce expression, not only `LIST_SUM`. The preaggregated SQL is a useful
comparison because it already avoids gathering sample lists at the gene join.

The explicit `--direct-operation filter-sum` evaluates one
`LIST_SUM(LIST_FILTER(values, predicate))` over the same 100 million genotypes. Its
deterministic approximately 50% selection mask varies at every flat element position,
rather than repeating the case cohort across rows. Each iteration has one reduction over the
same prebuilt predicate list. Input generation, mask construction, and scalar
reference validation are excluded. Runtime predicate conversion, filtering, and summation
are included.

| Single-filter direct path | Median |
| --- | ---: |
| Decode to `u8`, filter, then sum | 7.632 ms |
| Previous broadcast-based fused packed kernel | 3.907 ms |
| Current transpose-based fused packed kernel | 2.393 ms |

The current packed path is 3.19× faster than the decoded path for this isolated operation.
It selected 49,995,133 genotypes and produced checksum 9,142,796. Every row matched the
scalar reference; a 103-row × three-sample run also exercised nine empty selections with
null sums. The [mask-conversion analysis](MASK_CONVERSION.md) explains the transpose and
its standalone diagnostic. Two separate case/control reductions, measured before the
transpose change, took 13.152 ms decoded and 7.793 ms with the fused packed path. Those
are **not** measurements of the final transpose implementation.

## Implementation and validation

`ListSum` already dispatches grouped aggregates on encoded children. The new FastLanes
kernel sums patchless, all-valid two-bit `u8` data directly; other widths, element nulls,
and patched arrays use the existing fallback. `list_filter(values, predicates)` accepts
parallel fixed-size or variable lists, treats null predicates as false, and propagates
null outer lists. `ListSum` detects a lazy `ListFilter` before materialization and offers
the original ranges and mask to an encoding-specific grouped hook. For aligned packed
inputs, it sums selected bit planes without a filtered genotype buffer. Sliced packed
sources and bit-offset predicates are supported. Standalone packed filtering uses bounded
scratch and emits compressed output only at selectivity up to 3%; denser filters retain
the decoded fallback.

Earlier focused checks passed for list functions and FastLanes sum kernels, including slices,
null/empty groups, fallback encodings, and randomized ranges. Unit tests were not rerun after
the final transpose and cleanup changes; the added exhaustive bit-mapping and sliced-filter
regressions have not been run. The focused commands are:

```sh
cargo test -p vortex-array scalar_fn::fns::list_ --lib
cargo test -p vortex-fastlanes bitpacking::compute::sum::tests --lib
```

The optimized benchmark and small-workload reference checks passed. Workspace-wide tests
and Clippy were not run. The measurements here isolate a projection-pushdown opportunity;
they do not demonstrate an end-to-end DataFusion query with this expression pushed down.
