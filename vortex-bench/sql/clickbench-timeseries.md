# ClickBench time-series benchmark

A time-series query suite over the real ClickBench `hits` table. It reuses the data of the
[sorted ClickBench variant](./clickbench.md#sorted-variant): the table is sorted by
`EventTime` and split into 100 shards whose filenames are shuffled, so engines cannot rely
on file order.

Stock ClickBench is an analytics workload and only a handful of its queries touch time. The
queries in [`clickbench_timeseries.sql`](./clickbench_timeseries.sql) are written in the
shape of time-series workloads instead:

- Narrow and wide time-range scans and counts.
- Newest and oldest rows, overall and per series.
- Filters combined with a limit, with and without an ordering.
- Time bounds and counts that file statistics can answer.
- Hourly, daily and per-minute buckets with `DATE_TRUNC`.
- Predicates on functions of time, `EXTRACT`, and interval arithmetic.

Queries are numbered from Q0 in file order. The harness lives in
[`src/clickbench`](../src/clickbench).

## Running locally

```bash
vx-bench run clickbench-timeseries --engine datafusion,duckdb --format parquet,vortex
```

The suite shares its data directory with `clickbench-sorted`, so preparing either one
prepares both.
