// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Probe: time the kernel-side claims from the executor overhead notes.

use std::time::Duration;
use std::time::Instant;

use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VTable;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::Filter;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::SharedArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::dtype::DecimalDType;
use vortex::array::dtype::Nullability;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::array::optimizer::kernels::ArrayKernelsExt;
use vortex::array::scalar::DecimalValue;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::compressor::BtrBlocksCompressor;
use vortex::encodings::fsst::FSST;
use vortex::encodings::fsst::fsst_compress;
use vortex::encodings::fsst::fsst_train_compressor;
use vortex::extension::datetime::TimeUnit;
use vortex::mask::Mask;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::session::VortexSession;
use vortex_onpair::OnPair;

fn time<F: FnMut() -> ArrayRef>(label: &str, rows: usize, mut f: F) {
    // warm up
    for _ in 0..3 {
        std::hint::black_box(f());
    }
    let mut samples = Vec::new();
    for _ in 0..30 {
        let start = Instant::now();
        std::hint::black_box(f());
        samples.push(start.elapsed());
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let max = samples[samples.len() - 1];
    println!(
        "  {label:<58} median {:>9.2?}  min {:>9.2?}  max {:>9.2?}  ({:.2} ns/row)",
        median,
        min,
        max,
        median.as_nanos() as f64 / rows as f64
    );
}

fn comment(i: usize) -> String {
    let words = ["carefully", "final", "deposits", "haggle", "blithely", "quick", "furious", "pending"];
    format!("{} {} {} {i:08}", words[i % 8], words[(i / 8) % 8], words[(i / 64) % 8])
}

fn compress(session: &VortexSession, arr: &ArrayRef) -> ArrayRef {
    let mut ctx = session.create_execution_ctx();
    BtrBlocksCompressor::from_session(session)
        .compress(arr, &mut ctx)
        .unwrap()
}

fn exec(ctx: &mut ExecutionCtx, arr: ArrayRef) -> ArrayRef {
    arr.execute::<Canonical>(ctx).unwrap().into_array()
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed >> 33
}

fn mask_10pct(len: usize) -> Mask {
    let mut seed = 42u64;
    let indices: Vec<usize> = (0..len).filter(|_| lcg(&mut seed) % 10 == 0).collect();
    Mask::from_indices(len, indices)
}

fn section_runend(session: &VortexSession) {
    println!("\n== RunEnd: slice of 32 rows then canonicalize ==");
    let mut ctx = session.create_execution_ctx();
    let n = 131072usize;
    // ~100-row runs of u8 values (l_linestatus-like, but many runs).
    let mut seed = 7u64;
    let mut vals = Vec::with_capacity(n);
    let mut v = 0u8;
    while vals.len() < n {
        v = (lcg(&mut seed) % 3) as u8;
        let run = 50 + (lcg(&mut seed) % 100) as usize;
        for _ in 0..run.min(n - vals.len()) {
            vals.push(v);
        }
    }
    let _ = v;
    let prim = PrimitiveArray::new(Buffer::copy_from(&vals), Validity::NonNullable).into_array();
    let compressed = compress(session, &prim);
    println!("  compressor picked: {}", compressed.display_tree_encodings_only());
    let re = vortex::encodings::runend::RunEnd::encode(prim.clone(), &mut ctx).unwrap().into_array();
    println!("  built: {}", re.display_tree_encodings_only());
    let start = 65_000usize;
    time("ArrayRef::slice(32) only (no execute)", 32, || re.slice(start..start + 32).unwrap());
    let sliced = re.slice(start..start + 32).unwrap();
    println!("  sliced tree: {}", sliced.display_tree_encodings_only());
    time("slice(32) + execute::<Canonical>", 32, || {
        exec(&mut ctx, re.slice(start..start + 32).unwrap())
    });
    time("execute::<Canonical> of the pre-sliced array", 32, || {
        exec(&mut ctx, sliced.clone())
    });
    time("compressor shape dict[runend[bp,bp]]: slice(32) + execute", 32, || {
        exec(&mut ctx, compressed.slice(start..start + 32).unwrap())
    });
    let inner = compressed.slots()[0].clone().unwrap();
    println!("  inner runend: {}", inner.display_tree_encodings_only());
    time("runend[bitpacked, bitpacked]: slice(32) + execute", 32, || {
        exec(&mut ctx, inner.slice(start..start + 32).unwrap())
    });
    // For comparison: a 32-row slice of a plain primitive array.
    time("primitive slice(32) + execute::<Canonical> (baseline)", 32, || {
        exec(&mut ctx, prim.slice(start..start + 32).unwrap())
    });
    // And a run-end slice that lands inside one run (constant fast path).
    time("slice(32) inside one run + execute", 32, || {
        exec(&mut ctx, re.slice(10..42).unwrap())
    });
    // Whole-chunk canonicalize for scale.
    time("execute::<Canonical> whole 131072-row RunEnd", n, || exec(&mut ctx, re.clone()));
}

fn section_filter(session: &VortexSession) {
    println!("\n== Filter (10% mask) over FSST / OnPair / BitPacked, 8192 rows ==");
    let mut ctx = session.create_execution_ctx();
    let n = 8192usize;
    let kernels = session.kernels();
    println!(
        "  session has (Filter, FSST) kernel: {}   (Filter, OnPair) kernel: {}",
        kernels.has_execute_parent(Filter.id(), FSST.id()),
        kernels.has_execute_parent(Filter.id(), OnPair.id())
    );
    let strings = VarBinViewArray::from_iter_str((0..n).map(comment)).into_array();
    let compressed = compress(session, &strings);
    println!("  compressor picked for comments: {}", compressed.display_tree_encodings_only());
    let compressor = fsst_train_compressor(&strings, &mut ctx).unwrap();
    let fsst = fsst_compress(&strings, &compressor, &mut ctx).unwrap().into_array();
    println!("  built FSST: {}", fsst.display_tree_encodings_only());
    let mask = mask_10pct(n);
    println!("  mask true count: {}", mask.true_count());
    let kept = mask.true_count();
    time("FSST: filter(mask) + execute::<Canonical>", kept, || {
        exec(&mut ctx, fsst.filter(mask.clone()).unwrap())
    });
    let filtered = fsst.filter(mask.clone()).unwrap();
    println!("  filtered (before execute) tree: {}", filtered.display_tree_encodings_only());
    time("FSST: execute whole chunk, then filter canonical", kept, || {
        let whole = exec(&mut ctx, fsst.clone());
        exec(&mut ctx, whole.filter(mask.clone()).unwrap())
    });
    time("FSST: execute::<Canonical> whole 8192-row chunk", n, || exec(&mut ctx, fsst.clone()));
    time("compressor output: filter(mask) + execute", kept, || {
        exec(&mut ctx, compressed.filter(mask.clone()).unwrap())
    });
    time("VarBinView (canonical): filter(mask) + execute", kept, || {
        exec(&mut ctx, strings.filter(mask.clone()).unwrap())
    });

    // OnPair: build through the compressor by giving it repeated word pairs.
    let words = ["carefully", "final", "deposits", "haggle", "blithely", "quick", "furious", "pending"];
    let mut seed = 9u64;
    let onpairish = VarBinViewArray::from_iter_str((0..n).map(|_| {
        let k = lcg(&mut seed) as usize;
        format!("{} {} {} {}", words[k % 8], words[(k / 8) % 8], words[(k / 64) % 8], words[(k / 512) % 8])
    }))
    .into_array();
    let onpair = compress(session, &onpairish);
    println!("  compressor picked for word strings: {}", onpair.display_tree_encodings_only());
    time("compressed word strings: filter(mask) + execute", kept, || {
        exec(&mut ctx, onpair.filter(mask.clone()).unwrap())
    });
    time("compressed word strings: execute whole chunk", n, || exec(&mut ctx, onpair.clone()));

    let ints = PrimitiveArray::new(
        Buffer::from_iter((0..n).map(|i| (i % 1000) as u32)),
        Validity::NonNullable,
    )
    .into_array();
    let packed = compress(session, &ints);
    println!("  compressor picked for ints: {}", packed.display_tree_encodings_only());
    time("BitPacked: filter(mask) + execute::<Canonical>", kept, || {
        exec(&mut ctx, packed.filter(mask.clone()).unwrap())
    });
}

fn section_between(session: &VortexSession) {
    println!("\n== Between vs Lt on decimal(15,2) l_quantity-like column, 131072 rows ==");
    let mut ctx = session.create_execution_ctx();
    let n = 131072usize;
    let mut seed = 3u64;
    let vals: Vec<i64> = (0..n).map(|_| ((lcg(&mut seed) % 50) as i64 + 1) * 100).collect();
    let dt = DecimalDType::new(15, 2);
    let dec = DecimalArray::new(Buffer::copy_from(&vals), dt, Validity::NonNullable).into_array();
    let compressed = compress(session, &dec);
    println!("  compressor picked: {}", compressed.display_tree_encodings_only());
    let lit = |v: i64| ConstantArray::new(
        Scalar::decimal(DecimalValue::I64(v), dt, Nullability::NonNullable),
        n,
    )
    .into_array();
    let opts = BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::NonStrict,
    };
    let between = compressed.clone().between(lit(500), lit(2000), opts.clone()).unwrap();
    println!("  between tree after optimize: {}", between.display_tree_encodings_only());
    let lt = compressed.binary(lit(2000), Operator::Lt).unwrap();
    println!("  lt tree after optimize: {}", lt.display_tree_encodings_only());
    time("compressed BETWEEN 5.00 AND 20.00 -> execute", n, || {
        exec(&mut ctx, compressed.clone().between(lit(500), lit(2000), opts.clone()).unwrap())
    });
    time("compressed < 20.00 -> execute", n, || {
        exec(&mut ctx, compressed.binary(lit(2000), Operator::Lt).unwrap())
    });
    time("compressed >= 5.00 AND <= 20.00 (two compares) -> execute", n, || {
        let a = compressed.binary(lit(500), Operator::Gte).unwrap();
        let b = compressed.binary(lit(2000), Operator::Lte).unwrap();
        exec(&mut ctx, a.binary(b, Operator::And).unwrap())
    });
    time("canonical decimal BETWEEN -> execute (baseline)", n, || {
        exec(&mut ctx, dec.clone().between(lit(500), lit(2000), opts.clone()).unwrap())
    });
}

fn section_like(session: &VortexSession) {
    println!("\n== Like over Dict, 8192 rows, by values encoding ==");
    let mut ctx = session.create_execution_ctx();
    let n = 8192usize;
    let sets: Vec<(&str, Vec<&str>, &str)> = vec![
        ("l_shipmode", vec!["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"], "%AIR%"),
        (
            "l_shipinstruct",
            vec!["DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"],
            "%PERSON%",
        ),
    ];
    for (name, values, pattern) in sets {
        let mut seed = 11u64;
        let codes = PrimitiveArray::new(
            Buffer::<u8>::from_iter((0..n).map(|_| (lcg(&mut seed) % values.len() as u64) as u8)),
            Validity::NonNullable,
        )
        .into_array();
        let plain = VarBinViewArray::from_iter_str(values.iter().copied()).into_array();
        let full = VarBinViewArray::from_iter_str((0..n).map(|i| values[i % values.len()])).into_array();
        let compressed = compress(session, &full);
        println!("  {name}: compressor picked {}", compressed.display_tree_encodings_only());
        let compressor = fsst_train_compressor(&plain, &mut ctx).unwrap();
        let fsst_values = fsst_compress(&plain, &compressor, &mut ctx).unwrap().into_array();
        let pat = ConstantArray::new(Scalar::utf8(pattern, Nullability::NonNullable), n).into_array();
        let variants: Vec<(&str, ArrayRef)> = vec![
            ("dict[u8, varbinview]", plain.clone()),
            ("dict[u8, shared[varbinview]]", SharedArray::new(plain.clone()).into_array()),
            ("dict[u8, fsst]", fsst_values.clone()),
            ("dict[u8, shared[fsst]]", SharedArray::new(fsst_values.clone()).into_array()),
        ];
        for (label, vals) in variants {
            let dict = DictArray::try_new(codes.clone(), vals).unwrap().into_array();
            let like = Like::try_new(dict.clone(), pat.clone(), LikeOptions::default())
                .unwrap()
                .into_array()
                .optimize()
                .unwrap();
            println!("    {label}: after optimize -> {}", like.display_tree_encodings_only());
            time(&format!("{name} {label} LIKE {pattern}"), n, || {
                exec(
                    &mut ctx,
                    Like::try_new(dict.clone(), pat.clone(), LikeOptions::default())
                        .unwrap()
                        .into_array()
                        .optimize()
                        .unwrap(),
                )
            });
        }
        let pat2 = pat.clone();
        time(&format!("{name} compressor output LIKE {pattern}"), n, || {
            exec(
                &mut ctx,
                Like::try_new(compressed.clone(), pat2.clone(), LikeOptions::default())
                    .unwrap()
                    .into_array()
                    .optimize()
                    .unwrap(),
            )
        });
    }
}

fn section_extension(session: &VortexSession) {
    println!("\n== Extension (date) execute::<Canonical> ==");
    let mut ctx = session.create_execution_ctx();
    let n = 8192usize;
    let mut seed = 5u64;
    let days = PrimitiveArray::new(
        Buffer::from_iter((0..n).map(|_| 9000 + (lcg(&mut seed) % 2500) as i32)),
        Validity::NonNullable,
    )
    .into_array();
    let date = TemporalArray::new_date(days, TimeUnit::Days).into_array();
    let compressed = compress(session, &date);
    println!("  compressor picked: {}", compressed.display_tree_encodings_only());
    let out = exec(&mut ctx, compressed.clone());
    println!("  after execute::<Canonical>: {}", out.display_tree_encodings_only());
    time("execute::<Canonical> on ext[...]", n, || exec(&mut ctx, compressed.clone()));
    time("execute::<Canonical> on the storage child", n, || {
        exec(&mut ctx, compressed.slots()[0].clone().unwrap())
    });
}

fn main() {
    let session = VortexSession::default();
    let which = std::env::args().nth(1).unwrap_or_default();
    let all = which.is_empty();
    if all || which == "runend" {
        section_runend(&session);
    }
    if all || which == "filter" {
        section_filter(&session);
    }
    if all || which == "between" {
        section_between(&session);
    }
    if all || which == "like" {
        section_like(&session);
    }
    if all || which == "extension" {
        section_extension(&session);
    }
    let _ = Duration::ZERO;
}
