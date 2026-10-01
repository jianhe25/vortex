# Aggregate Statistics

Arrays retain finalized aggregate results in `Aggregations`. This replaces the fixed runtime
`ArrayStats`, `StatsSet`, and `Stat` APIs with the same aggregate functions used by execution.

## Cache API

`array.aggregations()` returns an `AggregationsRef` bound to that immutable array. The store does not
own the array, so it cannot form an ownership cycle. The key is the full `AggregateFnRef`, including
its options.

```rust
let aggregate = Sum.bind(NumericalAggregateOpts::skip_nans());
let known: Precision<Scalar> = array.aggregations().get_result(&aggregate);
let result: Scalar = array.aggregations().compute_result(&aggregate, ctx)?;
let snapshot: AggregateResults = array.aggregations().snapshot_results();
```

`get_result` reads metadata without execution. `compute_result` returns an exact cached result or
computes one outside the cache lock. Concurrent misses can compute the same result. Failed
computations leave the cache unchanged.

`Precision::Absent` means unknown. `Precision::Exact` includes null, zero, and false results.
`Precision::Inexact` carries a bound defined by the aggregate. An inexact value cannot answer an
exact computation.

`AggregateResults` is an immutable snapshot for file summaries and serialization. Its public
constructor checks unique keys and result dtypes. It does not establish that those values describe
an array. Producers that avoid scanning must prove that their facts describe the exact immutable
input before using the documented unsafe `seed_result` boundary.

## Results and Partials

An accumulator owns the state needed to merge chunks. A finalized scalar can replace a partial only
when the aggregate implements `partial_from_result`. The default declines this conversion, even
when the result and state dtypes match.

Min, max, counts, and the original sum support proven conversions. Sum's null result denotes
saturated overflow, while its empty state is zero. Sortedness and constantness decline because their
booleans omit boundary or representative values. Rich states such as `SumV2` also decline.

`StatFn` reads metadata and converts results through this explicit aggregate contract. It does not
scan the input. Missing metadata stays unknown, and supported bounds retain their precision.
Streaming file summaries accumulate typed partials and finalize them only after merging chunks.

## Array Rewrites

A new representation receives a fresh store. An aggregate opts into transfer with
`is_representation_invariant` when its result depends only on logical values, order, and dtype.
Custom aggregates keep their results on the original representation by default.

`UncompressedSizeInBytes` does not opt in. Some implementations count retained backing buffers, so
equal logical values can have different sizes in different representations. Slice and subset
propagation use separate rules for facts that remain valid, such as known constantness or sortedness.

Generic aggregate results do not prove unchecked constructor invariants. Decimal precision, list
offsets, and list-view trimming validate physical values independently before unchecked construction.

## Wire Compatibility

The node, footer, and legacy zone formats retain their field identifiers and scalar encodings.
`stats::compat` translates them into finalized aggregate results and validates historical scalar
types. Legacy extrema use the input dtype on the wire and the nullable aggregate dtype in memory.
Numerical legacy fields always use NaN-skipping options.

Node serialization projects cache snapshots into representable fields. Footer selections reject
functions or options that the existing format cannot represent. Variable-length truncation changes
only a snapshot, leaving exact live cache entries intact. Legacy sortedness and constantness flags
remain final results and never become aggregate partials.
