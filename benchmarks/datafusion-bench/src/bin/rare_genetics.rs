// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic case/control allele-count benchmark on a Vortex variant matrix.
//!
//! File generation and planning happen outside the timed query. The SQL path uses
//! DataFusion's native list_filter, array_transform, and list_sum functions.

#[path = "rare_genetics/direct.rs"]
mod direct;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::ensure;
use clap::Parser;
use clap::ValueEnum;
use datafusion::arrow::array::Array;
use datafusion::arrow::array::ArrayRef as ArrowArrayRef;
use datafusion::arrow::array::BooleanArray;
use datafusion::arrow::array::FixedSizeListArray;
use datafusion::arrow::array::StringArray;
use datafusion::arrow::array::StructArray as ArrowStructArray;
use datafusion::arrow::array::UInt8Array;
use datafusion::arrow::array::UInt64Array;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::datatypes::Field;
use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::file::WriteOptionsSessionExt;
use vortex_arrow::ArrowSessionExt;
use vortex_bench::SESSION;
use vortex_datafusion::VortexFormat;
use vortex_datafusion::VortexTableOptions;

const QUERY: &str = r#"
SELECT g.gene_name,
       SUM(list_sum(array_transform(list_filter(v.samples, s -> s['is_case']), s -> s['genotype']))) AS n_case_mutations,
       SUM(list_sum(array_transform(list_filter(v.samples, s -> NOT s['is_case']), s -> s['genotype']))) AS n_control_mutations
FROM variant_sample_matrix v
JOIN genes g ON v.contig = g.contig
            AND v.position >= g.start_position
            AND v.position < g.end_position
GROUP BY g.gene_name
ORDER BY g.gene_name
"#;

const SUM_QUERY: &str = r#"
SELECT position, list_sum(array_transform(samples, s -> s['genotype'])) AS n_mutations
FROM variant_sample_matrix
"#;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Sql,
    SqlSum,
    Direct,
    All,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DirectOperation {
    All,
    Sum,
    CaseControl,
    FilterSum,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DirectRepresentation {
    Both,
    Decoded,
    Packed,
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 10_000)]
    variants: usize,
    #[arg(long, default_value_t = 10_000)]
    samples: usize,
    #[arg(long, default_value_t = 20)]
    genes: usize,
    #[arg(long, default_value_t = 3)]
    iterations: usize,
    #[arg(long, value_enum, default_value_t = Mode::All)]
    mode: Mode,
    #[arg(long, value_enum, default_value_t = DirectOperation::All)]
    direct_operation: DirectOperation,
    #[arg(long, value_enum, default_value_t = DirectRepresentation::Both)]
    direct_representation: DirectRepresentation,
    #[arg(long, default_value = "/private/tmp/rare_genetics.vortex")]
    path: PathBuf,
    #[arg(long)]
    explain: bool,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value = "databricks", value_parser = ["databricks", "generic"])]
    dialect: String,
    /// Override the SQL in --mode sql or --mode sql-sum for planner diagnostics.
    #[arg(long)]
    query: Option<String>,
    /// Plan SQL and print its explanation without running the query.
    #[arg(long)]
    plan_only: bool,
}

struct Generated {
    schema: Arc<Schema>,
    batches: Vec<ArrayRef>,
    genotypes: Option<ArrayRef>,
    case_counts: Vec<u64>,
    control_counts: Vec<u64>,
    variant_counts: Vec<u64>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ensure!(args.variants > 0 && args.samples >= 2 && args.genes > 0 && args.iterations > 0);
    ensure!(
        args.samples <= i32::MAX as usize,
        "Arrow fixed list size exceeds i32"
    );
    ensure!(
        args.genes <= args.variants,
        "genes must not exceed variants"
    );
    ensure!(
        args.threads.is_none_or(|threads| threads > 0),
        "threads must be positive"
    );
    ensure!(
        args.query.is_none() || matches!(args.mode, Mode::Sql | Mode::SqlSum),
        "--query requires --mode sql or --mode sql-sum"
    );
    let keep_genotypes = matches!(args.mode, Mode::Direct | Mode::All);

    let generation_start = Instant::now();
    let generated = generate(&args, keep_genotypes)?;
    println!(
        "setup_generation_ms={:.3}",
        generation_start.elapsed().as_secs_f64() * 1e3
    );

    if matches!(args.mode, Mode::Sql | Mode::SqlSum | Mode::All) {
        let write_start = Instant::now();
        write_file(&args.path, &generated.batches).await?;
        println!(
            "setup_write_ms={:.3} file_bytes={} path={}",
            write_start.elapsed().as_secs_f64() * 1e3,
            std::fs::metadata(&args.path)?.len(),
            args.path.display()
        );
        if matches!(args.mode, Mode::Sql | Mode::All) {
            run_sql(
                &args,
                &generated,
                args.query.as_deref().unwrap_or(QUERY),
                "datafusion_sql",
            )
            .await?;
        }
        if matches!(args.mode, Mode::SqlSum | Mode::All) {
            run_sql(
                &args,
                &generated,
                args.query.as_deref().unwrap_or(SUM_QUERY),
                "datafusion_sql_sum",
            )
            .await?;
        }
    }

    if let Some(genotypes) = generated.genotypes {
        direct::run(
            genotypes,
            args.samples,
            args.variants,
            args.iterations,
            args.direct_operation,
            args.direct_representation,
        )?;
    }

    Ok(())
}

fn generate(args: &Args, keep_genotypes: bool) -> anyhow::Result<Generated> {
    let sample_fields = vec![
        Arc::new(Field::new("genotype", DataType::UInt8, false)),
        Arc::new(Field::new("is_case", DataType::Boolean, false)),
    ];
    let sample_type = DataType::Struct(sample_fields.clone().into());
    let schema = Arc::new(Schema::new(vec![
        Field::new("contig", DataType::Utf8, false),
        Field::new("position", DataType::UInt64, false),
        Field::new("reference", DataType::Utf8, false),
        Field::new("alternate", DataType::Utf8, false),
        Field::new(
            "samples",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", sample_type, false)),
                args.samples as i32,
            ),
            false,
        ),
    ]));

    let mut batches = Vec::new();
    let mut genotype_chunks = Vec::new();
    let mut case_counts = vec![0_u64; args.genes];
    let mut control_counts = vec![0_u64; args.genes];
    let mut variant_counts = Vec::with_capacity(args.variants);
    // Limit each Arrow batch to about a million sample structs.
    let variants_per_batch = (1_000_000 / args.samples).max(1);

    for start in (0..args.variants).step_by(variants_per_batch) {
        let end = (start + variants_per_batch).min(args.variants);
        let mut genotypes = Vec::with_capacity((end - start) * args.samples);
        let mut is_case = Vec::with_capacity((end - start) * args.samples);
        for variant in start..end {
            let gene = variant * args.genes / args.variants;
            let mut variant_count = 0_u64;
            // One common variant in four; the rest have 0.1-1% alternate allele frequency.
            let freq_per_million = if variant % 4 == 0 {
                200_000 + (mix64(variant as u64) % 300_001) as u32
            } else {
                1_000 + (mix64(variant as u64) % 9_001) as u32
            };
            for sample in 0..args.samples {
                let seed = ((variant as u64) << 32) ^ sample as u64;
                let genotype = ((mix64(seed ^ 0x9e37_79b9_7f4a_7c15) % 1_000_000
                    < u64::from(freq_per_million)) as u8)
                    + ((mix64(seed ^ 0xbf58_476d_1ce4_e5b9) % 1_000_000
                        < u64::from(freq_per_million)) as u8);
                let case = sample % 2 == 0;
                genotypes.push(genotype);
                is_case.push(case);
                variant_count += u64::from(genotype);
                if case {
                    case_counts[gene] += u64::from(genotype);
                } else {
                    control_counts[gene] += u64::from(genotype);
                }
            }
            variant_counts.push(variant_count);
        }
        let genotype_array: ArrowArrayRef = Arc::new(UInt8Array::from(genotypes));
        if keep_genotypes {
            genotype_chunks.push(
                SESSION
                    .arrow()
                    .from_arrow_array(genotype_array.clone(), false)?,
            );
        }
        let sample_array: ArrowArrayRef = Arc::new(ArrowStructArray::new(
            sample_fields.clone().into(),
            vec![genotype_array, Arc::new(BooleanArray::from(is_case))],
            None,
        ));
        let list_array: ArrowArrayRef = Arc::new(FixedSizeListArray::try_new(
            Arc::new(Field::new("item", sample_array.data_type().clone(), false)),
            args.samples as i32,
            sample_array,
            None,
        )?);
        let n = end - start;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["1"; n])),
                Arc::new(UInt64Array::from_iter_values(
                    (start..end).map(|v| v as u64),
                )),
                Arc::new(StringArray::from(vec!["A"; n])),
                Arc::new(StringArray::from(vec!["C"; n])),
                list_array,
            ],
        )?;
        batches.push(SESSION.arrow().from_arrow_record_batch(batch, &schema)?);
    }

    let genotypes = if keep_genotypes {
        let dtype = genotype_chunks[0].dtype().clone();
        Some(ChunkedArray::try_new(genotype_chunks, dtype)?.into_array())
    } else {
        None
    };
    Ok(Generated {
        schema,
        batches,
        genotypes,
        case_counts,
        control_counts,
        variant_counts,
    })
}

async fn write_file(path: &PathBuf, batches: &[ArrayRef]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let dtype = batches[0].dtype().clone();
    let chunked = ChunkedArray::try_new(batches.to_vec(), dtype)?.into_array();
    let file = tokio::fs::File::create(path).await?;
    SESSION
        .write_options()
        .write(file, chunked.to_array_stream())
        .await?;
    Ok(())
}

async fn run_sql(
    args: &Args,
    generated: &Generated,
    query: &str,
    label: &str,
) -> anyhow::Result<()> {
    let threads = args.threads.unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    });
    let config = SessionConfig::new()
        .with_target_partitions(threads)
        .set_str("datafusion.sql_parser.dialect", &args.dialect);
    let session = SessionContext::new_with_config(config);
    println!(
        "mode={label} dialect={} target_partitions={threads} projection_pushdown=false predicate_pushdown=true",
        args.dialect
    );
    let mut options = VortexTableOptions::default();
    options.projection_pushdown = false;
    options.predicate_pushdown = true;
    let format = Arc::new(VortexFormat::new_with_options(SESSION.clone(), options));
    let path = args.path.canonicalize()?;
    let url = ListingTableUrl::parse(format!("file://{}", path.display()))?;
    let listing = ListingTable::try_new(
        ListingTableConfig::new(url)
            .with_listing_options(ListingOptions::new(format))
            .with_schema(generated.schema.clone()),
    )?;
    session.register_table("variant_sample_matrix", Arc::new(listing))?;
    register_genes(&session, args)?;

    let dataframe = session.sql(query).await.context("plan genetics SQL")?;
    if args.explain || args.plan_only {
        println!("query=\n{query}");
        println!(
            "plan=\n{}",
            dataframe
                .clone()
                .explain(false, false)?
                .collect()
                .await?
                .iter()
                .map(|batch| format!("{batch:?}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    if args.plan_only {
        return Ok(());
    }
    let mut plan = dataframe.create_physical_plan().await?;
    let warmup = datafusion::physical_plan::collect(plan.clone(), session.task_ctx()).await?;
    if label == "datafusion_sql_sum" {
        validate_sum_result(&warmup, generated)?;
    } else {
        validate_result(&warmup, generated)?;
    }
    let mut timings = Vec::with_capacity(args.iterations);
    for iteration in 0..args.iterations {
        plan = reset_plan_states(plan)?;
        let start = Instant::now();
        let batches = datafusion::physical_plan::collect(plan.clone(), session.task_ctx()).await?;
        let duration = start.elapsed();
        if label == "datafusion_sql_sum" {
            validate_sum_result(&batches, generated)?;
        } else {
            validate_result(&batches, generated)?;
        }
        timings.push(duration);
        println!(
            "mode={label} iteration={} elapsed_ms={:.3} total_case={} total_control={}",
            iteration + 1,
            duration.as_secs_f64() * 1e3,
            generated.case_counts.iter().sum::<u64>(),
            generated.control_counts.iter().sum::<u64>()
        );
    }
    print_summary(label, &mut timings);
    Ok(())
}

fn register_genes(session: &SessionContext, args: &Args) -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("contig", DataType::Utf8, false),
        Field::new("start_position", DataType::UInt64, false),
        Field::new("end_position", DataType::UInt64, false),
        Field::new("gene_name", DataType::Utf8, false),
    ]));
    let names = (0..args.genes)
        .map(|i| format!("GENE_{i:05}"))
        .collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["1"; args.genes])),
            Arc::new(UInt64Array::from_iter_values(
                (0..args.genes).map(|i| (i * args.variants).div_ceil(args.genes) as u64),
            )),
            Arc::new(UInt64Array::from_iter_values(
                (1..=args.genes).map(|i| (i * args.variants).div_ceil(args.genes) as u64),
            )),
            Arc::new(StringArray::from(names)),
        ],
    )?;
    session.register_table(
        "genes",
        Arc::new(MemTable::try_new(schema, vec![vec![batch]])?),
    )?;
    Ok(())
}

fn validate_result(batches: &[RecordBatch], generated: &Generated) -> anyhow::Result<()> {
    let mut seen = 0;
    for batch in batches {
        let names = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .context("gene_name is not Utf8")?;
        let cases = batch
            .column(1)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Float64Array>()
            .context("n_case_mutations is not Float64")?;
        let controls = batch
            .column(2)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Float64Array>()
            .context("n_control_mutations is not Float64")?;
        for row in 0..batch.num_rows() {
            ensure!(
                !names.is_null(row) && !cases.is_null(row) && !controls.is_null(row),
                "null case/control SQL result at row {row}"
            );
            let gene: usize = names
                .value(row)
                .strip_prefix("GENE_")
                .context("unexpected gene name")?
                .parse()?;
            ensure!(
                gene < generated.case_counts.len() && gene == seen,
                "unexpected or duplicate gene {gene}"
            );
            ensure!(
                cases.value(row) == generated.case_counts[gene] as f64,
                "wrong case count for gene {gene}: {} versus {}",
                cases.value(row),
                generated.case_counts[gene]
            );
            ensure!(
                controls.value(row) == generated.control_counts[gene] as f64,
                "wrong control count for gene {gene}: {} versus {}",
                controls.value(row),
                generated.control_counts[gene]
            );
            seen += 1;
        }
    }
    ensure!(
        seen == generated.case_counts.len(),
        "result contains {seen} genes"
    );
    Ok(())
}

fn validate_sum_result(batches: &[RecordBatch], generated: &Generated) -> anyhow::Result<()> {
    let mut seen = vec![false; generated.variant_counts.len()];
    for batch in batches {
        let positions = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .context("position is not UInt64")?;
        let sums = batch
            .column(1)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Float64Array>()
            .context("n_mutations is not Float64")?;
        for row in 0..batch.num_rows() {
            ensure!(
                !positions.is_null(row) && !sums.is_null(row),
                "null list_sum SQL result at row {row}"
            );
            let position = positions.value(row) as usize;
            ensure!(
                position < seen.len() && !seen[position],
                "unexpected variant {position}"
            );
            ensure!(
                sums.value(row) == generated.variant_counts[position] as f64,
                "wrong mutation count for variant {position}"
            );
            seen[position] = true;
        }
    }
    ensure!(seen.into_iter().all(|found| found), "missing variant sums");
    Ok(())
}

fn print_summary(mode: &str, durations: &mut [Duration]) {
    durations.sort();
    let midpoint = durations.len() / 2;
    let median = if durations.len() % 2 == 0 {
        (durations[midpoint - 1].as_secs_f64() + durations[midpoint].as_secs_f64()) * 500.0
    } else {
        durations[midpoint].as_secs_f64() * 1e3
    };
    println!(
        "mode={mode} median_ms={median:.3} iterations={}",
        durations.len()
    );
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
