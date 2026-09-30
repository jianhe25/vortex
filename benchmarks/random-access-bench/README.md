# Random Access benchmark

Measures point-lookup latency: fetching individual rows by index from a file, rather than
scanning it. This is the workload behind the `Random Access` PR comment.

[Arrow IPC](https://arrow.apache.org/docs/format/Columnar.html#ipc-file-format) is Apache
Arrow's built-in file format, formerly called Feather V2. This suite writes it without
optional buffer compression, so it provides the established constant-time access reference.
Parquet provides the established reference for a compressed columnar representation. Together,
they let the suite compare Vortex and Lance against both ends of the storage trade-off.

## Keeping the comparison fair

Every format is read the way an expert would read it from local disk, and each `take` returns
fully decoded Arrow-equivalent rows:

- **Arrow IPC** memory-maps the file and decodes record batches zero-copy with
  `FileDecoder`, then takes the requested rows. Only the pages holding those rows are touched.
  The file carries per-batch row offsets in its custom metadata so lookups go straight to the
  right batch.
- **Parquet** loads the footer and page index once, then reads with a `RowSelection` so the
  offset index fetches and decodes only the pages that hold the requested rows, never a whole
  row group. Page reads use `pread` on a shared descriptor, the same syscall profile Vortex uses.
  The synthetic inputs are written with zstd level 3, the repository's convention for generated
  Parquet, in 32Ki-row row groups with 1024-row data pages, so a lookup decompresses one small
  page per column. Selecting by page rather than by row group is what keeps the cost of a
  lookup proportional to the rows fetched, not to the row group size: without it, a
  single-row-group file decodes the whole file per lookup, and smaller zstd row groups still
  decode every row group a pattern touches.
- **Lance** uses `Dataset::take` on a v2.1 dataset, its native point-lookup API.
- **Vortex** runs a scan restricted to the requested row indices, then canonicalizes the result so
  it is as decoded as the Arrow batches the other formats return.

The Lance, Arrow IPC, and Vortex files are all converted from the same Parquet source, so every
format sees identical data. Cached mode reuses one open handle per format and warms it for a
second before timing; reopen mode pays each format's footer or manifest parse on every iteration.

Two access patterns are generated with a fixed seed (see [`src/main.rs`](./src/main.rs)):

- **correlated**: several clusters of consecutive indices scattered across the dataset,
  simulating lookups with spatial locality;
- **uniform**: indices drawn from a Poisson process spread uniformly across the dataset,
  simulating lookups with no locality.

Each pattern runs over four datasets (`taxi`, `feature-vectors`, `nested-lists`,
`nested-structs`) in Arrow IPC, Parquet, Lance, and Vortex. Cached mode performs a one-second
untimed warm-up, then reuses the open file handle. Reopen mode includes file open and metadata
work in each timed iteration. CI drives the full matrix via
[`scripts/random-access-split.py`](../../scripts/random-access-split.py).

## Running locally

```bash
cargo run -p random-access-bench --profile release_debug --features lance
```
