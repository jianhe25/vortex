// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures the overhead of the iterative executor loop (`execute_until`) on the compressed
//! chunks a default Vortex file writer produces for TPC-H.
//!
//! Workflow:
//!
//! 1. `gen` writes TPC-H tables as Vortex files with the default write strategy.
//! 2. `harvest` reads every column back chunk by chunk and keeps one chunk per distinct encoding
//!    tree shape, so the examples are the real encodings the writer chose.
//! 3. `trace` (feature `trace`) prints the executor's step sequence for each example.
//! 4. `bench` (optionally feature `profile`) times `execute::<Canonical>` on each example and, with
//!    the profile feature, breaks the time down per loop phase.
//! 5. `micro` times the individual primitives the loop relies on.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use tokio::task::spawn_blocking;
use tpchgen::generators::CustomerGenerator;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen::generators::PartGenerator;
use tpchgen::generators::PartSuppGenerator;
use tpchgen::generators::SupplierGenerator;
use tpchgen_arrow::RecordBatchIterator;
use vortex::VortexSessionDefault;
use vortex::array::AnyCanonical;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VTable;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::Primitive;
use vortex::array::arrays::Struct;
use vortex::array::arrays::VarBinView;
use vortex::array::builders::builder_with_capacity_in;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::display::DisplayOptions;
use vortex::array::dtype::DType;
use vortex::array::dtype::Nullability;
use vortex::array::dtype::PType;
use vortex::array::expr::col;
use vortex::array::matcher::Matcher;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::array::optimizer::kernels::ArrayKernelsExt;
use vortex::array::scalar::Scalar;
use vortex::array::scalar_fn::fns::between::BetweenOptions;
use vortex::array::scalar_fn::fns::between::StrictComparison;
use vortex::array::scalar_fn::fns::like::Like;
use vortex::array::scalar_fn::fns::like::LikeOptions;
use vortex::array::scalar_fn::fns::operators::Operator;
use vortex::array::stats::StatsSet;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::split_by::SplitBy;
use vortex::mask::Mask;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
#[cfg(feature = "profile")]
use vortex_array::exec_profile;
use vortex_arrow::ArrowSessionExt;

/// Nanoseconds since `start`, saturating rather than truncating.
fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Human-readable duration from nanoseconds.
fn fmt_ns(ns: f64) -> String {
    if ns >= 1e9 {
        format!("{:.2}s", ns / 1e9)
    } else if ns >= 1e6 {
        format!("{:.1}ms", ns / 1e6)
    } else if ns >= 1e3 {
        format!("{:.1}µs", ns / 1e3)
    } else {
        format!("{ns:.0}ns")
    }
}

const TABLES: &[&str] = &[
    "lineitem", "orders", "part", "partsupp", "customer", "supplier",
];

#[derive(Parser)]
#[command(about = "Executor loop overhead benchmark on TPC-H chunk encodings")]
struct Cli {
    /// Directory holding the generated `.vortex` files.
    #[arg(long, default_value = "benchmarks/execute-overhead-bench/data")]
    data_dir: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate TPC-H tables as Vortex files using the default writer.
    Gen {
        #[arg(long, default_value_t = 1.0)]
        sf: f64,
        #[arg(long, value_delimiter = ',')]
        tables: Option<Vec<String>>,
    },
    /// List distinct chunk encoding trees per column.
    Harvest {
        /// Print the full tree for each distinct shape.
        #[arg(long)]
        trees: bool,
    },
    /// Print the executor trace for each distinct shape (needs feature `trace`).
    Trace {
        /// Only trace the shape with this index from `harvest`.
        #[arg(long)]
        only: Option<Vec<usize>>,
        /// Record kernel attempts that declined too.
        #[arg(long)]
        attempts: bool,
    },
    /// Time `execute::<Canonical>` on each distinct shape.
    Bench {
        #[arg(long, default_value_t = 200)]
        iters: usize,
        #[arg(long)]
        only: Option<Vec<usize>>,
    },
    /// Time the primitives the loop is built from.
    Micro,
    /// Time small slices, filters and scalar-function trees over each distinct shape.
    Workloads {
        #[arg(long, default_value_t = 200)]
        iters: usize,
        #[arg(long)]
        only: Option<Vec<usize>>,
        /// Print the executor trace of each workload (needs feature `trace`).
        #[arg(long)]
        trace: bool,
        /// Print the per-phase overhead breakdown of each workload (needs feature `profile`).
        #[arg(long)]
        details: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let _guard = rt.enter();
    let session = VortexSession::default().with_tokio();
    match cli.cmd {
        Cmd::Gen { sf, tables } => {
            let tables = tables.unwrap_or_else(|| TABLES.iter().map(|s| s.to_string()).collect());
            rt.block_on(generate(&session, &cli.data_dir, sf, &tables))
        }
        Cmd::Harvest { trees } => {
            let examples = rt.block_on(harvest(&session, &cli.data_dir))?;
            print_examples(&examples, trees);
            Ok(())
        }
        Cmd::Trace { only, attempts } => {
            let examples = rt.block_on(harvest(&session, &cli.data_dir))?;
            trace(&session, &examples, only.as_deref(), attempts)
        }
        Cmd::Bench { iters, only } => {
            let examples = rt.block_on(harvest(&session, &cli.data_dir))?;
            bench(&session, &examples, iters, only.as_deref())
        }
        Cmd::Micro => {
            let examples = rt.block_on(harvest(&session, &cli.data_dir))?;
            micro(&session, &examples)
        }
        Cmd::Workloads {
            iters,
            only,
            trace,
            details,
        } => {
            let examples = rt.block_on(harvest(&session, &cli.data_dir))?;
            run_workloads(&session, &examples, iters, only.as_deref(), trace, details)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------------------------

fn batch_iter(table: &str, sf: f64) -> Result<Box<dyn RecordBatchIterator>> {
    let bs = 8192 * 64;
    Ok(match table {
        "lineitem" => Box::new(
            tpchgen_arrow::LineItemArrow::new(LineItemGenerator::new(sf, 1, 1)).with_batch_size(bs),
        ),
        "orders" => Box::new(
            tpchgen_arrow::OrderArrow::new(OrderGenerator::new(sf, 1, 1)).with_batch_size(bs),
        ),
        "part" => Box::new(
            tpchgen_arrow::PartArrow::new(PartGenerator::new(sf, 1, 1)).with_batch_size(bs),
        ),
        "partsupp" => Box::new(
            tpchgen_arrow::PartSuppArrow::new(PartSuppGenerator::new(sf, 1, 1)).with_batch_size(bs),
        ),
        "customer" => Box::new(
            tpchgen_arrow::CustomerArrow::new(CustomerGenerator::new(sf, 1, 1)).with_batch_size(bs),
        ),
        "supplier" => Box::new(
            tpchgen_arrow::SupplierArrow::new(SupplierGenerator::new(sf, 1, 1)).with_batch_size(bs),
        ),
        other => anyhow::bail!("unknown table {other}"),
    })
}

async fn generate(session: &VortexSession, dir: &Path, sf: f64, tables: &[String]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for table in tables {
        let path = dir.join(format!("{table}.vortex"));
        if path.exists() {
            println!("{} exists, skipping", path.display());
            continue;
        }
        let start = Instant::now();
        let iter = batch_iter(table, sf)?;
        let schema = Arc::clone(iter.schema());
        let dtype = session.arrow().from_arrow_schema(&schema)?;
        let (mut tx, rx) = mpsc::channel::<vortex::error::VortexResult<ArrayRef>>(2);
        let gen_session = session.clone();
        let producer = spawn_blocking(move || -> Result<()> {
            for batch in iter {
                let array = gen_session
                    .arrow()
                    .from_arrow_record_batch(batch, &schema)?;
                futures::executor::block_on(tx.send(Ok(array)))?;
            }
            Ok(())
        });
        let mut file = tokio::fs::File::create(&path).await?;
        session
            .write_options()
            .write(&mut file, ArrayStreamAdapter::new(dtype, rx))
            .await?;
        producer.await??;
        let bytes = std::fs::metadata(&path)?.len();
        println!(
            "wrote {} ({:.1} MiB) in {:.1}s",
            path.display(),
            bytes as f64 / (1 << 20) as f64,
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Harvest
// ---------------------------------------------------------------------------------------------

struct Example {
    idx: usize,
    table: String,
    column: String,
    signature: String,
    array: ArrayRef,
    /// Number of chunks across all columns that share this shape.
    chunks: usize,
    /// Rows across all chunks sharing this shape.
    rows: usize,
}

fn signature(array: &ArrayRef) -> String {
    let mut sig = array.encoding_id().to_string();
    sig = sig.trim_start_matches("vortex.").to_string();
    match array.dtype() {
        DType::Primitive(ptype, nullability) => sig.push_str(&format!(
            "<{ptype}{}>",
            if bool::from(*nullability) { "?" } else { "" }
        )),
        DType::Struct(..) => sig.push_str("<struct>"),
        DType::Utf8(_) | DType::Binary(_) | DType::Bool(_) | DType::Decimal(..) => {}
        other => sig.push_str(&format!("<{other}>")),
    }
    let children: Vec<String> = array.children_iter().map(signature).collect();
    if !children.is_empty() {
        sig.push('[');
        sig.push_str(&children.join(","));
        sig.push(']');
    }
    sig
}

async fn harvest(session: &VortexSession, dir: &Path) -> Result<Vec<Example>> {
    let mut by_sig: HashMap<String, Example> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for table in TABLES {
        let path = dir.join(format!("{table}.vortex"));
        if !path.exists() {
            continue;
        }
        let file = session.open_options().open_path(&path).await?;
        let DType::Struct(fields, _) = file.dtype().clone() else {
            anyhow::bail!("expected struct dtype");
        };
        for name in fields.names().iter() {
            let projection = col(name.clone()).bind(file.dtype())?.optimize_recursive()?;
            let mut stream = file
                .scan()?
                .with_projection(projection)
                .with_split_by(SplitBy::Layout)
                .into_array_stream()?;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.context("reading chunk")?;
                let sig = signature(&chunk);
                match by_sig.get_mut(&sig) {
                    Some(ex) => {
                        ex.chunks += 1;
                        ex.rows += chunk.len();
                    }
                    None => {
                        order.push(sig.clone());
                        by_sig.insert(
                            sig.clone(),
                            Example {
                                idx: 0,
                                table: table.to_string(),
                                column: name.to_string(),
                                signature: sig,
                                array: chunk.clone(),
                                chunks: 1,
                                rows: chunk.len(),
                            },
                        );
                    }
                }
            }
        }
    }
    let mut examples: Vec<Example> = order
        .into_iter()
        .filter_map(|sig| by_sig.remove(&sig))
        .collect();
    for (i, ex) in examples.iter_mut().enumerate() {
        ex.idx = i;
    }
    Ok(examples)
}

fn print_examples(examples: &[Example], trees: bool) {
    println!(
        "{:>3} {:>7} {:>10} {:>8} {:<22} signature",
        "idx", "chunks", "rows", "ex_len", "source"
    );
    for ex in examples {
        println!(
            "{:>3} {:>7} {:>10} {:>8} {:<22} {}",
            ex.idx,
            ex.chunks,
            ex.rows,
            ex.array.len(),
            format!("{}.{}", ex.table, ex.column),
            ex.signature
        );
        if trees {
            println!(
                "{}",
                ex.array.display_as(DisplayOptions::TreeDisplay {
                    buffers: true,
                    metadata: true,
                    stats: false
                })
            );
        }
    }
}

fn select<'a>(examples: &'a [Example], only: Option<&[usize]>) -> Vec<&'a Example> {
    match only {
        Some(only) => only.iter().filter_map(|i| examples.get(*i)).collect(),
        None => examples.iter().collect(),
    }
}

// ---------------------------------------------------------------------------------------------
// Trace
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "trace")]
fn trace(
    session: &VortexSession,
    examples: &[Example],
    only: Option<&[usize]>,
    attempts: bool,
) -> Result<()> {
    use vortex_array::test_harness::trace::TraceOptions;
    use vortex_array::test_harness::trace::TraceResolution;
    use vortex_array::test_harness::trace::trace_op_with;
    let options = TraceOptions {
        resolution: if attempts {
            TraceResolution::Attempts
        } else {
            TraceResolution::ExecutedOnly
        },
    };
    for ex in select(examples, only) {
        println!(
            "===== [{}] {}.{}  {}",
            ex.idx, ex.table, ex.column, ex.signature
        );
        println!(
            "{}",
            ex.array.display_as(DisplayOptions::TreeDisplay {
                buffers: false,
                metadata: true,
                stats: false
            })
        );
        let mut ctx = session.create_execution_ctx();
        let array = ex.array.clone();
        let traced = trace_op_with(options, || array.execute::<Canonical>(&mut ctx))?;
        println!("{}", traced.trace);
    }
    Ok(())
}

#[cfg(not(feature = "trace"))]
fn trace(_: &VortexSession, _: &[Example], _: Option<&[usize]>, _: bool) -> Result<()> {
    anyhow::bail!("rebuild with `--features trace`")
}

// ---------------------------------------------------------------------------------------------
// Bench
// ---------------------------------------------------------------------------------------------

/// Rebuild the tree so every node is a fresh, uniquely owned `Arc` (buffers stay shared).
fn deep_clone(array: &ArrayRef) -> Result<ArrayRef> {
    let slots = array
        .slots()
        .iter()
        .map(|slot| slot.as_ref().map(deep_clone).transpose())
        .collect::<Result<_>>()?;
    // SAFETY: slots are logically identical copies.
    Ok(unsafe { array.clone().with_slots(slots) }?)
}

fn median(v: &mut [u64]) -> u64 {
    v.sort_unstable();
    v[v.len() / 2]
}

struct Timing {
    median_ns: u64,
    min_ns: u64,
    mean_ns: u64,
    max_ns: u64,
}

fn time_execute(ctx: &mut ExecutionCtx, inputs: Vec<ArrayRef>) -> Result<Timing> {
    let mut samples = Vec::with_capacity(inputs.len());
    for input in inputs {
        let start = Instant::now();
        let out = input.execute::<Canonical>(ctx)?;
        samples.push(elapsed_ns(start));
        drop(out);
    }
    let min_ns = samples.iter().copied().min().unwrap_or(0);
    let max_ns = samples.iter().copied().max().unwrap_or(0);
    let mean_ns = samples.iter().sum::<u64>() / samples.len() as u64;
    Ok(Timing {
        median_ns: median(&mut samples),
        min_ns,
        mean_ns,
        max_ns,
    })
}

fn bench(
    session: &VortexSession,
    examples: &[Example],
    iters: usize,
    only: Option<&[usize]>,
) -> Result<()> {
    let mut ctx = session.create_execution_ctx();
    println!(
        "{:>3} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}  {:<20} signature",
        "idx", "len", "shared_ns", "unique_ns", "min_ns", "ns/row", "share%", "source"
    );
    #[cfg(feature = "profile")]
    let mut profiles: Vec<(usize, exec_profile::ExecProfile)> = Vec::new();
    let total_rows: usize = examples.iter().map(|e| e.rows).sum();
    let mut weighted_ns = 0f64;
    let mut rows_of_selected = 0usize;
    for ex in select(examples, only) {
        // Warm up.
        for _ in 0..3 {
            ex.array.clone().execute::<Canonical>(&mut ctx)?;
        }
        let shared = time_execute(&mut ctx, (0..iters).map(|_| ex.array.clone()).collect())?;
        let uniques: Vec<ArrayRef> = (0..iters)
            .map(|_| deep_clone(&ex.array))
            .collect::<Result<_>>()?;
        #[cfg(feature = "profile")]
        exec_profile::reset();
        let unique = time_execute(&mut ctx, uniques)?;
        #[cfg(feature = "profile")]
        profiles.push((ex.idx, exec_profile::snapshot()));
        let per_row = unique.median_ns as f64 / ex.array.len() as f64;
        let est_total_ns = per_row * ex.rows as f64;
        weighted_ns += est_total_ns;
        rows_of_selected += ex.rows;
        println!(
            "{:>3} {:>8} {:>10} {:>10} {:>10} {:>10.2} {:>8.1}  {:<20} {}",
            ex.idx,
            ex.array.len(),
            shared.median_ns,
            unique.median_ns,
            unique.min_ns,
            per_row,
            100.0 * ex.rows as f64 / total_rows as f64,
            format!("{}.{}", ex.table, ex.column),
            ex.signature
        );
    }
    println!(
        "\nestimated full-decode time for the selected shapes over {} rows: {} ({:.2} ns/row)",
        rows_of_selected,
        fmt_ns(weighted_ns),
        weighted_ns / rows_of_selected as f64
    );
    #[cfg(feature = "profile")]
    print_profiles(examples, &profiles, iters);
    Ok(())
}

#[cfg(feature = "profile")]
fn print_profiles(
    examples: &[Example],
    profiles: &[(usize, exec_profile::ExecProfile)],
    iters: usize,
) {
    let iters_f = iters as f64;
    println!(
        "\n=== per-call executor profile (unique-tree runs, averaged over {iters} iterations) ==="
    );
    println!(
        "{:>3} {:>5} {:>4} {:>5} {:>5} {:>5} {:>5} {:>4} | {:>8} {:>8} {:>6} | {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "idx",
        "iters",
        "pops",
        "lkup",
        "found",
        "decl",
        "appl",
        "opt",
        "total",
        "work",
        "ovh%",
        "done",
        "pop",
        "lookup",
        "declin",
        "optim",
        "preexe",
        "exslot",
        "take",
        "bcreat",
        "final",
        "untrk",
        "shared"
    );
    // Chunk-weighted totals, accumulated in f64 to avoid per-shape rounding.
    let mut weighted: HashMap<&'static str, f64> = HashMap::default();
    for (idx, p) in profiles {
        let ex = &examples[*idx];
        let d = |v: u64| v as f64 / iters_f;
        println!(
            "{:>3} {:>5.1} {:>4.1} {:>5.1} {:>5.1} {:>5.1} {:>5.1} {:>4.1} | {:>8.0} {:>8.0} {:>6.1} | {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.1}",
            idx,
            d(p.iterations),
            d(p.pops),
            d(p.ep_lookups),
            d(p.ep_lookup_found),
            d(p.ep_declined),
            d(p.ep_applied),
            d(p.optimize_calls),
            d(p.total_ns),
            d(p.work_ns()),
            100.0 * p.overhead_ns() as f64 / p.total_ns.max(1) as f64,
            d(p.done_check_ns),
            d(p.pop_ns),
            d(p.ep_lookup_ns),
            d(p.ep_declined_ns),
            d(p.optimize_ns),
            d(p.pre_execute_ns),
            d(p.execute_slot_ns),
            d(p.take_slot_ns),
            d(p.builder_create_ns),
            d(p.finalize_ns),
            d(p.untracked_ns()),
            d(p.take_slot_shared + p.put_slot_shared),
        );
        // Weight each shape by how many chunks of TPC-H SF1 it represents.
        let weight = ex.chunks as f64 / iters_f;
        for (name, value) in [
            ("calls", p.calls),
            ("iterations", p.iterations),
            ("ep_lookups", p.ep_lookups),
            ("ep_lookup_found", p.ep_lookup_found),
            ("ep_declined", p.ep_declined),
            ("ep_applied", p.ep_applied),
            ("total_ns", p.total_ns),
            ("work_ns", p.work_ns()),
            ("overhead_ns", p.overhead_ns()),
            ("done_check_ns", p.done_check_ns),
            ("pop_ns", p.pop_ns),
            ("ep_lookup_ns", p.ep_lookup_ns),
            ("ep_declined_ns", p.ep_declined_ns),
            ("optimize_ns", p.optimize_ns),
            ("pre_execute_ns", p.pre_execute_ns),
            ("execute_slot_ns", p.execute_slot_ns),
            ("append_child_execute_ns", p.append_child_execute_ns),
            ("take_slot_ns", p.take_slot_ns),
            ("builder_create_ns", p.builder_create_ns),
            ("finalize_ns", p.finalize_ns),
            ("untracked_ns", p.untracked_ns()),
            ("shared", p.take_slot_shared + p.put_slot_shared),
        ] {
            *weighted.entry(name).or_default() += value as f64 * weight;
        }
    }
    println!(
        "\n=== chunk-weighted totals (what a full SF1 decode of these columns would spend) ==="
    );
    let get = |name: &str| weighted.get(name).copied().unwrap_or_default();
    let total = get("total_ns").max(1.0);
    let pct = |v: f64| 100.0 * v / total;
    println!("calls              {:>12.0}", get("calls"));
    println!(
        "iterations         {:>12.0}   ({:.2} per call)",
        get("iterations"),
        get("iterations") / get("calls").max(1.0)
    );
    println!(
        "kernel lookups     {:>12.0}   found {:.0} declined {:.0} applied {:.0}",
        get("ep_lookups"),
        get("ep_lookup_found"),
        get("ep_declined"),
        get("ep_applied")
    );
    println!("total              {:>12}", fmt_ns(total));
    println!(
        "  work (done+applied+append) {:>12}  {:5.1}%",
        fmt_ns(get("work_ns")),
        pct(get("work_ns"))
    );
    println!(
        "  overhead                   {:>12}  {:5.1}%",
        fmt_ns(get("overhead_ns")),
        pct(get("overhead_ns"))
    );
    for (label, name) in [
        ("done checks", "done_check_ns"),
        ("pop/put_slot", "pop_ns"),
        ("kernel lookup (hash+probe)", "ep_lookup_ns"),
        ("kernels that declined", "ep_declined_ns"),
        ("optimize_ctx after kernel", "optimize_ns"),
        ("pre-execute dtype/stats clone", "pre_execute_ns"),
        ("execute returning ExecuteSlot", "execute_slot_ns"),
        ("execute returning AppendChild", "append_child_execute_ns"),
        ("take_slot", "take_slot_ns"),
        ("builder create", "builder_create_ns"),
        ("finalize (stats transfer)", "finalize_ns"),
        ("untracked loop glue", "untracked_ns"),
    ] {
        println!(
            "    {:<32} {:>12}  {:5.1}%",
            label,
            fmt_ns(get(name)),
            pct(get(name))
        );
    }
    println!("  shared-arc take/put allocations {:>8.0}", get("shared"));
}

// ---------------------------------------------------------------------------------------------
// Micro benchmarks
// ---------------------------------------------------------------------------------------------

fn time_n<R>(n: usize, mut f: impl FnMut() -> R) -> f64 {
    let start = Instant::now();
    for _ in 0..n {
        std::hint::black_box(f());
    }
    start.elapsed().as_nanos() as f64 / n as f64
}

fn micro(session: &VortexSession, examples: &[Example]) -> Result<()> {
    let n = 200_000;
    let mut ctx = session.create_execution_ctx();
    let prim = vortex::buffer::buffer![1i64, 2, 3, 4].into_array();
    let first_non_canonical = examples
        .iter()
        .find(|e| !AnyCanonical::matches(&e.array))
        .map(|e| e.array.clone())
        .context("need a non-canonical example")?;
    let canonical = first_non_canonical
        .clone()
        .execute::<Canonical>(&mut ctx)?
        .into_array();
    let varbin = examples
        .iter()
        .find(|e| matches!(e.dtype(), DType::Utf8(_)))
        .map(|e| e.array.clone().execute::<Canonical>(&mut ctx))
        .transpose()?
        .map(|c| c.into_array());

    println!("{:<52} {:>10}", "primitive", "ns/op");
    println!(
        "{:<52} {:>10.1}",
        "AnyCanonical::matches(primitive) [3rd branch]",
        time_n(n, || AnyCanonical::matches(&prim))
    );
    if let Some(v) = &varbin {
        println!(
            "{:<52} {:>10.1}",
            "AnyCanonical::matches(varbinview) [10th branch]",
            time_n(n, || AnyCanonical::matches(v))
        );
    }
    println!(
        "{:<52} {:>10.1}",
        format!(
            "AnyCanonical::matches({}) [all 12 fail]",
            first_non_canonical.encoding_id()
        ),
        time_n(n, || AnyCanonical::matches(&first_non_canonical))
    );
    println!(
        "{:<52} {:>10.1}",
        "Primitive::matches(primitive)",
        time_n(n, || Primitive::matches(&prim))
    );
    println!("{:<52} {:>10.1}", "encoding_id() == Primitive.id()", {
        let id = Primitive.id();
        time_n(n, || prim.encoding_id() == id)
    });

    let kernels = session.kernels();
    let p_id = first_non_canonical.encoding_id();
    let c_id = first_non_canonical
        .children_iter()
        .next()
        .map(|c| c.encoding_id())
        .unwrap_or(p_id);
    println!(
        "{:<52} {:>10.1}",
        format!("kernels.has_execute_parent({p_id}, {c_id}) [session]"),
        time_n(n, || kernels.has_execute_parent(p_id, c_id))
    );
    println!(
        "{:<52} {:>10.1}",
        "session.kernels() (ArcSwap load + TypeId probe)",
        time_n(n, || session.kernels().has_execute_parent(p_id, c_id))
    );
    println!(
        "{:<52} {:>10.1}",
        "create_execution_ctx()",
        time_n(n, || session.create_execution_ctx())
    );

    println!(
        "{:<52} {:>10.1}",
        "optimize_ctx(primitive) [no rewrite]",
        time_n(n, || prim.optimize_ctx(session))
    );
    println!(
        "{:<52} {:>10.1}",
        "optimize_ctx(canonical example) [no rewrite]",
        time_n(n, || canonical.optimize_ctx(session))
    );
    println!(
        "{:<52} {:>10.1}",
        format!(
            "optimize_ctx({}) [no rewrite]",
            first_non_canonical.encoding_id()
        ),
        time_n(n, || first_non_canonical.optimize_ctx(session))
    );
    println!(
        "{:<52} {:>10.1}",
        "optimize(primitive) [static rules only]",
        time_n(n, || prim.optimize())
    );

    println!(
        "{:<52} {:>10.1}",
        "stats: to_array_stats + set_iter (canonical example)",
        {
            time_n(n, || {
                let stats = first_non_canonical.statistics().to_array_stats();
                canonical
                    .statistics()
                    .set_iter(StatsSet::from(stats).into_iter());
            })
        }
    );
    println!(
        "{:<52} {:>10.1}",
        "stats: inherit_from (canonical <- example)",
        {
            time_n(n, || {
                canonical
                    .statistics()
                    .inherit_from(first_non_canonical.statistics())
            })
        }
    );
    println!(
        "{:<52} {:>10.1}",
        "dtype().clone() (primitive)",
        time_n(n, || prim.dtype().clone())
    );

    println!(
        "{:<52} {:>10.1}",
        "deep_clone(example tree) [with_slots + validate]",
        time_n(n / 10, || deep_clone(&first_non_canonical))
    );
    println!(
        "{:<52} {:>10.1}",
        "with_slots(shared root) [take_slot on shared arc]",
        {
            time_n(n / 10, || {
                let slots = first_non_canonical.slots().to_vec().into();
                unsafe { first_non_canonical.clone().with_slots(slots) }
            })
        }
    );
    println!(
        "{:<52} {:>10.1}",
        "Arc clone of ArrayRef",
        time_n(n, || first_non_canonical.clone())
    );
    println!("{:<52} {:>10.1}", "builder_with_capacity_in(i64, 8192)", {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        time_n(n / 10, || {
            builder_with_capacity_in(&dtype, 8192, ctx.allocator())
        })
    });
    println!(
        "{:<52} {:>10.1}",
        "execute::<Canonical>(already canonical primitive)",
        { time_n(n, || prim.clone().execute::<Canonical>(&mut ctx)) }
    );
    println!(
        "{:<52} {:>10.1}",
        "Instant::now() + elapsed()",
        time_n(n, || Instant::now().elapsed())
    );
    for idx in [1usize, 2, 10, 15, 16, 28] {
        if let Some(ex) = examples.get(idx) {
            let a = ex.array.clone();
            println!(
                "{:<52} {:>10.1}",
                format!(
                    "slice(0..32) [{}] {}",
                    idx,
                    ex.signature.chars().take(28).collect::<String>()
                ),
                time_n(n / 10, || a.slice(5..37))
            );
        }
    }
    let buf = vortex::buffer::Buffer::<i64>::from(vec![1i64; 32]);
    println!(
        "{:<52} {:>10.1}",
        "PrimitiveArray from Buffer<i64>(32).into_array()",
        time_n(n, || buf.clone().into_array())
    );
    let _ = (Struct::matches(&prim), VarBinView::matches(&prim));
    Ok(())
}

impl Example {
    fn dtype(&self) -> &DType {
        self.array.dtype()
    }
}

// ---------------------------------------------------------------------------------------------
// Workloads: small slices, filters and scalar-function trees over the real chunks
// ---------------------------------------------------------------------------------------------

/// Builds a fresh workload input from the deep-cloned chunk.
type BuildFn = Box<dyn Fn(&ArrayRef) -> Result<ArrayRef>>;

struct Workload {
    name: &'static str,
    build: BuildFn,
}

fn mid_scalar(ex: &Example, ctx: &mut ExecutionCtx) -> Result<Scalar> {
    Ok(ex.array.execute_scalar(ex.array.len() / 2, ctx)?)
}

fn workloads(ex: &Example, ctx: &mut ExecutionCtx) -> Result<Vec<Workload>> {
    let len = ex.array.len();
    let mid = mid_scalar(ex, ctx)?;
    let is_utf8 = matches!(ex.dtype(), DType::Utf8(_));
    let cmp_op = if is_utf8 { Operator::Eq } else { Operator::Lt };
    let mut out: Vec<Workload> = Vec::new();

    let lt = {
        let mid = mid.clone();
        move |a: &ArrayRef| -> Result<ArrayRef> {
            Ok(a.binary(
                ConstantArray::new(mid.clone(), a.len()).into_array(),
                cmp_op,
            )?)
        }
    };
    let filter_mask = |n: usize| Mask::from_indices(n, (0..n).step_by(10));

    out.push(Workload {
        name: "slice32",
        build: Box::new(move |a| Ok(a.slice(len / 3..len / 3 + 32.min(len - len / 3))?)),
    });
    out.push(Workload {
        name: "slice1024",
        build: Box::new(move |a| Ok(a.slice(len / 3..len / 3 + 1024.min(len - len / 3))?)),
    });
    out.push(Workload {
        name: "filter10pct",
        build: Box::new(move |a| Ok(a.filter(filter_mask(a.len()))?)),
    });
    {
        let lt = lt.clone();
        out.push(Workload {
            name: if is_utf8 { "eq_mid" } else { "lt_mid" },
            build: Box::new(move |a| lt(a)),
        });
    }
    {
        let lt = lt.clone();
        out.push(Workload {
            name: if is_utf8 {
                "slice1024>eq"
            } else {
                "slice1024>lt"
            },
            build: Box::new(move |a| lt(&a.slice(len / 3..len / 3 + 1024.min(len - len / 3))?)),
        });
    }
    {
        let lt = lt.clone();
        out.push(Workload {
            name: if is_utf8 {
                "filter10pct>eq"
            } else {
                "filter10pct>lt"
            },
            build: Box::new(move |a| lt(&a.filter(filter_mask(a.len()))?)),
        });
    }
    if is_utf8 {
        out.push(Workload {
            name: "like_%special%",
            build: Box::new(move |a| {
                let pattern =
                    ConstantArray::new(Scalar::utf8("%special%", a.dtype().nullability()), a.len())
                        .into_array();
                Ok(Like::try_new(a.clone(), pattern, LikeOptions::default())?.into_array())
            }),
        });
    }
    if let DType::Decimal(..) | DType::Primitive(..) = ex.dtype() {
        let lo = mid;
        out.push(Workload {
            name: "between",
            build: Box::new(move |a| {
                Ok(a.clone().between(
                    ConstantArray::new(lo.clone(), a.len()).into_array(),
                    ConstantArray::new(lo.clone(), a.len()).into_array(),
                    BetweenOptions {
                        lower_strict: StrictComparison::NonStrict,
                        upper_strict: StrictComparison::NonStrict,
                    },
                )?)
            }),
        });
    }
    Ok(out)
}

#[cfg(feature = "trace")]
fn trace_one(label: &str, array: ArrayRef, ctx: &mut ExecutionCtx) -> Result<()> {
    use vortex_array::test_harness::trace::TraceOptions;
    use vortex_array::test_harness::trace::TraceResolution;
    use vortex_array::test_harness::trace::trace_op_with;
    let options = TraceOptions {
        resolution: TraceResolution::Attempts,
    };
    println!("----- {label}");
    let traced = trace_op_with(options, || array.execute::<Canonical>(ctx))?;
    println!("{}", traced.trace);
    Ok(())
}

#[cfg(not(feature = "trace"))]
fn trace_one(_: &str, _: ArrayRef, _: &mut ExecutionCtx) -> Result<()> {
    anyhow::bail!("rebuild with `--features trace`")
}

fn run_workloads(
    session: &VortexSession,
    examples: &[Example],
    iters: usize,
    only: Option<&[usize]>,
    trace: bool,
    details: bool,
) -> Result<()> {
    let mut ctx = session.create_execution_ctx();
    println!(
        "{:>3} {:<16} {:>8} {:>7} {:>10} {:>10} {:>8} | {:>5} {:>5} {:>5} {:>5} {:>5} {:>4} {:>4} | {:>6} {:>7} {:>8}",
        "idx",
        "workload",
        "in_len",
        "out_len",
        "build_ns",
        "exec_ns",
        "ns/row",
        "iters",
        "lkup",
        "found",
        "decl",
        "appl",
        "opt",
        "nest",
        "work%",
        "ovh_ns",
        "ovh/iter"
    );
    for ex in select(examples, only) {
        for w in workloads(ex, &mut ctx)? {
            // Build inputs first so the timed region is execution only.
            let unique = deep_clone(&ex.array)?;
            let build_start = Instant::now();
            let inputs: Vec<ArrayRef> = (0..iters)
                .map(|_| (w.build)(&unique))
                .collect::<Result<_>>()?;
            let build_ns = elapsed_ns(build_start) / iters as u64;
            let out_len = inputs[0].len();
            let input_sig = signature(&inputs[0]);
            // Warm up.
            (w.build)(&unique)?.execute::<Canonical>(&mut ctx)?;
            if trace {
                trace_one(
                    &format!("[{}] {} {}", ex.idx, w.name, input_sig),
                    (w.build)(&unique)?,
                    &mut ctx,
                )?;
            }
            #[cfg(feature = "profile")]
            exec_profile::reset();
            let timing = time_execute(&mut ctx, inputs)?;
            let summary = summarize_profile();
            let per_call = |v: u64| v as f64 / iters as f64;
            println!(
                "{:>3} {:<16} {:>8} {:>7} {:>10} {:>10} {:>8.2} | {:>5.1} {:>5.1} {:>5.1} {:>5.1} {:>5.1} {:>4.1} {:>4.1} | {:>6.1} {:>7.0} {:>8.0}  {}",
                ex.idx,
                w.name,
                ex.array.len(),
                out_len,
                build_ns,
                timing.median_ns,
                timing.median_ns as f64 / out_len.max(1) as f64,
                per_call(summary.iterations),
                per_call(summary.ep_lookups),
                per_call(summary.ep_lookup_found),
                per_call(summary.ep_declined),
                per_call(summary.ep_applied),
                per_call(summary.optimize_calls),
                per_call(summary.nested_calls),
                100.0 * summary.work_ns as f64 / summary.total_ns.max(1) as f64,
                per_call(summary.overhead_ns),
                summary.overhead_ns as f64 / summary.iterations.max(1) as f64,
                input_sig,
            );
            if details {
                println!(
                    "      timing ns: median {} mean {} min {} max {}",
                    timing.median_ns, timing.mean_ns, timing.min_ns, timing.max_ns
                );
                print_details(iters);
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct Summary {
    iterations: u64,
    ep_lookups: u64,
    ep_lookup_found: u64,
    ep_declined: u64,
    ep_applied: u64,
    optimize_calls: u64,
    nested_calls: u64,
    work_ns: u64,
    total_ns: u64,
    overhead_ns: u64,
}

#[cfg(feature = "profile")]
fn summarize_profile() -> Summary {
    let p = exec_profile::snapshot();
    Summary {
        iterations: p.iterations,
        ep_lookups: p.ep_lookups,
        ep_lookup_found: p.ep_lookup_found,
        ep_declined: p.ep_declined,
        ep_applied: p.ep_applied,
        optimize_calls: p.optimize_calls,
        nested_calls: p.nested_calls,
        work_ns: p.work_ns(),
        total_ns: p.total_ns,
        overhead_ns: p.overhead_ns(),
    }
}

#[cfg(not(feature = "profile"))]
fn summarize_profile() -> Summary {
    Summary::default()
}

#[cfg(feature = "profile")]
fn print_details(iters: usize) {
    let profile = exec_profile::snapshot();
    let per_call = |v: u64| v as f64 / iters as f64;
    println!(
        "      phases ns/call: calls {:.1} total {:.0} | done_check {:.0} | pop {:.0} (shared {:.1}) | lookup {:.0} | declined {:.0} | applied {:.0} | optimize {:.0} | pre_execute {:.0} | exec_slot {:.0} | exec_append {:.0} | exec_done {:.0} | take {:.0} (shared {:.1}) | builder_create {:.0} | append {:.0} | finalize {:.0} | untracked {:.0} | nested calls {:.1} nested_ns {:.0} | max_depth {}",
        per_call(profile.calls),
        per_call(profile.total_ns),
        per_call(profile.done_check_ns),
        per_call(profile.pop_ns),
        per_call(profile.put_slot_shared),
        per_call(profile.ep_lookup_ns),
        per_call(profile.ep_declined_ns),
        per_call(profile.ep_applied_ns),
        per_call(profile.optimize_ns),
        per_call(profile.pre_execute_ns),
        per_call(profile.execute_slot_ns),
        per_call(profile.append_child_execute_ns),
        per_call(profile.done_execute_ns),
        per_call(profile.take_slot_ns),
        per_call(profile.take_slot_shared),
        per_call(profile.builder_create_ns),
        per_call(profile.builder_append_ns),
        per_call(profile.finalize_ns),
        per_call(profile.untracked_ns()),
        per_call(profile.nested_calls),
        per_call(profile.nested_ns),
        profile.max_depth,
    );
}

#[cfg(not(feature = "profile"))]
fn print_details(_: usize) {}
