# execute-overhead-bench

Measures the overhead of the iterative executor loop (`ArrayRef::execute_until`) on the
compressed chunks the default Vortex file writer produces for TPC-H.

```bash
# 1. Write TPC-H SF1 (lineitem, orders, part, partsupp, customer, supplier) as .vortex files.
cargo run --release -p execute-overhead-bench -- gen --sf 1

# 2. One chunk per distinct encoding tree shape, with the tree printed.
cargo run --release -p execute-overhead-bench -- harvest --trees

# 3. Time execute::<Canonical> on every shape (shared and uniquely-owned trees).
cargo run --release -p execute-overhead-bench -- bench --iters 200

# 4. Small slices, filters and scalar-function trees over every shape.
cargo run --release -p execute-overhead-bench -- workloads --iters 200

# 5. Primitives the loop is built from (matchers, registry lookups, optimize_ctx, stats, ...).
cargo run --release -p execute-overhead-bench -- micro

# Per-phase breakdown of the loop (thread-local timers inside vortex-array).
cargo run --release -p execute-overhead-bench --features profile -- bench
cargo run --release -p execute-overhead-bench --features profile -- workloads --details

# Step-by-step executor traces.
cargo run --release -p execute-overhead-bench --features trace -- trace --attempts --only 10
cargo run --release -p execute-overhead-bench --features trace -- workloads --trace --only 10
```

Data is written to `benchmarks/execute-overhead-bench/data` by default (`--data-dir` overrides
it); `.vortex` files are git-ignored.

The `profile` feature enables `vortex-array/exec-profile`, which wraps every phase of
`execute_until` in an `Instant` timer. Each timed region costs roughly 60ns of timer overhead
that lands in the `untracked` bucket, so use the default build for wall-clock numbers and the
profile build for counts and relative phase costs. `REPORT.md` holds the measurements and
conclusions.
