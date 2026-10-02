// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Equality and prefix predicates evaluated on FSST codes, comparing the kernels against
//! alternative strategies on the same data.
//!
//! Equality: the registered compare kernel, the codes routed through the VarBin compare, and two
//! direct loops that differ only in whether the uncompressed lengths are read to reject rows.
//! Each runs with the uncompressed lengths stored plain and bit-packed.
//!
//! Prefix: the registered LIKE kernel on short and long prefixes. Short prefixes run the DFA over
//! the codes; long ones compare the codes every matching row opens with as raw bytes.

#![expect(clippy::unwrap_used)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArray;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::FSSTArraySlotsExt;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;
use vortex_fsst::test_utils::make_fsst_emails;
use vortex_fsst::test_utils::make_fsst_file_paths;
use vortex_fsst::test_utils::make_fsst_json_strings;
use vortex_fsst::test_utils::make_fsst_log_lines;
use vortex_fsst::test_utils::make_fsst_short_urls;
use vortex_fsst::test_utils::make_fsst_urls;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fsst::initialize(&session);
    session
});

const N: usize = 1_000_000;

fn with_bitpacked_lengths(fsst: &FSSTArray) -> FSSTArray {
    let mut ctx = SESSION.create_execution_ctx();
    let lengths = fsst
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let packed = bitpack_to_best_bit_width(&lengths, &mut ctx).unwrap();
    FSST::try_new_with_symbol_table(
        fsst.dtype().clone(),
        fsst.symbol_table(),
        fsst.codes(),
        packed.into_array(),
        &mut ctx,
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Equality
// ---------------------------------------------------------------------------

static EQ_PLAIN: LazyLock<FSSTArray> =
    LazyLock::new(|| make_fsst_urls(N, &mut SESSION.create_execution_ctx()));
static EQ_PACKED: LazyLock<FSSTArray> = LazyLock::new(|| with_bitpacked_lengths(&EQ_PLAIN));

/// A value present in the data.
static EQ_NEEDLE: LazyLock<String> = LazyLock::new(|| {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = EQ_PLAIN
        .clone()
        .into_array()
        .slice(N / 2..N / 2 + 1)
        .unwrap()
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_varbinview();
    String::from_utf8(canonical.bytes_at(0).to_vec()).unwrap()
});

#[derive(Clone, Copy)]
enum Lengths {
    Plain,
    BitPacked,
}

impl fmt::Display for Lengths {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plain => f.write_str("plain_lengths"),
            Self::BitPacked => f.write_str("bitpacked_lengths"),
        }
    }
}

impl Lengths {
    fn array(self) -> &'static FSSTArray {
        match self {
            Self::Plain => &EQ_PLAIN,
            Self::BitPacked => &EQ_PACKED,
        }
    }
}

const LENGTHS: [Lengths; 2] = [Lengths::Plain, Lengths::BitPacked];

/// Codes equality, optionally rejecting by uncompressed length before the code length.
fn eq_direct(fsst: &FSSTArray, needle: &[u8], read_lengths: bool) -> BitBuffer {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = fsst.compressor().compress(needle);
    let offsets = fsst
        .codes_offsets()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let bytes = fsst.codes_bytes().as_slice();
    if !read_lengths {
        return match_each_integer_ptype!(offsets.ptype(), |O| {
            codes_eq::<O>(offsets.as_slice::<O>(), bytes, &encoded)
        });
    }
    let lengths = fsst
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    match_each_integer_ptype!(lengths.ptype(), |L| {
        match_each_integer_ptype!(offsets.ptype(), |O| {
            lengths_then_codes_eq::<L, O>(
                lengths.as_slice::<L>(),
                offsets.as_slice::<O>(),
                bytes,
                needle.len(),
                &encoded,
            )
        })
    })
}

fn codes_eq<O: IntegerPType>(offsets: &[O], bytes: &[u8], encoded: &[u8]) -> BitBuffer {
    BitBuffer::collect_bool(offsets.len() - 1, |i| {
        let start: usize = offsets[i].as_();
        let end: usize = offsets[i + 1].as_();
        end - start == encoded.len() && &bytes[start..end] == encoded
    })
}

fn lengths_then_codes_eq<L: IntegerPType, O: IntegerPType>(
    lengths: &[L],
    offsets: &[O],
    bytes: &[u8],
    needle_len: usize,
    encoded: &[u8],
) -> BitBuffer {
    BitBuffer::collect_bool(lengths.len(), |i| {
        let len: usize = lengths[i].as_();
        if len != needle_len {
            return false;
        }
        let start: usize = offsets[i].as_();
        let end: usize = offsets[i + 1].as_();
        end - start == encoded.len() && &bytes[start..end] == encoded
    })
}

/// The registered FSST compare kernel.
#[divan::bench(args = LENGTHS)]
fn eq_kernel(bencher: Bencher, lengths: Lengths) {
    let fsst = lengths.array().clone().into_array();
    let rhs = ConstantArray::new(EQ_NEEDLE.as_str(), N).into_array();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            fsst.clone()
                .binary(rhs.clone(), Operator::Eq)
                .unwrap()
                .execute::<BoolArray>(ctx)
                .unwrap()
        });
}

/// The literal compressed once and the codes compared through the VarBin compare kernel.
#[divan::bench(args = LENGTHS)]
fn eq_codes_via_varbin(bencher: Bencher, lengths: Lengths) {
    let fsst = lengths.array();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            let encoded = ByteBuffer::from(fsst.compressor().compress(EQ_NEEDLE.as_bytes()));
            let rhs = ConstantArray::new(Scalar::binary(encoded, fsst.dtype().nullability()), N);
            fsst.codes()
                .into_array()
                .binary(rhs.into_array(), Operator::Eq)
                .unwrap()
                .execute::<BoolArray>(ctx)
                .unwrap()
        });
}

/// Code-length rejection from the offsets only.
#[divan::bench(args = LENGTHS)]
fn eq_codes_only(bencher: Bencher, lengths: Lengths) {
    let fsst = lengths.array();
    bencher.bench(|| eq_direct(fsst, EQ_NEEDLE.as_bytes(), false));
}

/// Uncompressed-length rejection first, then code length and code bytes.
#[divan::bench(args = LENGTHS)]
fn eq_lengths_then_codes(bencher: Bencher, lengths: Lengths) {
    let fsst = lengths.array();
    bencher.bench(|| eq_direct(fsst, EQ_NEEDLE.as_bytes(), true));
}

/// Decompress to VarBinView and compare there.
#[divan::bench(args = LENGTHS)]
fn eq_decompress(bencher: Bencher, lengths: Lengths) {
    let fsst = lengths.array().clone().into_array();
    let rhs = ConstantArray::new(EQ_NEEDLE.as_str(), N).into_array();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            fsst.clone()
                .execute::<Canonical>(ctx)
                .unwrap()
                .into_array()
                .binary(rhs.clone(), Operator::Eq)
                .unwrap()
                .execute::<BoolArray>(ctx)
                .unwrap()
        });
}

// ---------------------------------------------------------------------------
// Prefix
// ---------------------------------------------------------------------------

static PREFIX_URLS: LazyLock<FSSTArray> =
    LazyLock::new(|| make_fsst_short_urls(N, &mut SESSION.create_execution_ctx()));
static PREFIX_CB: LazyLock<FSSTArray> =
    LazyLock::new(|| make_fsst_clickbench_urls(N, &mut SESSION.create_execution_ctx()));
static PREFIX_LOG: LazyLock<FSSTArray> =
    LazyLock::new(|| make_fsst_log_lines(N, &mut SESSION.create_execution_ctx()));
static PREFIX_JSON: LazyLock<FSSTArray> =
    LazyLock::new(|| make_fsst_json_strings(N, &mut SESSION.create_execution_ctx()));
static PREFIX_PATH: LazyLock<FSSTArray> =
    LazyLock::new(|| make_fsst_file_paths(N, &mut SESSION.create_execution_ctx()));
static PREFIX_EMAIL: LazyLock<FSSTArray> =
    LazyLock::new(|| make_fsst_emails(N, &mut SESSION.create_execution_ctx()));

#[derive(Clone, Copy)]
enum Dataset {
    Urls,
    Cb,
    Log,
    Json,
    Path,
    Email,
}

impl Dataset {
    fn array(self) -> &'static FSSTArray {
        match self {
            Self::Urls => &PREFIX_URLS,
            Self::Cb => &PREFIX_CB,
            Self::Log => &PREFIX_LOG,
            Self::Json => &PREFIX_JSON,
            Self::Path => &PREFIX_PATH,
            Self::Email => &PREFIX_EMAIL,
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    dataset: Dataset,
    prefix: &'static str,
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

const CASES: &[Case] = &[
    Case {
        name: "urls_short",
        dataset: Dataset::Urls,
        prefix: "https",
    },
    Case {
        name: "urls_long",
        dataset: Dataset::Urls,
        prefix: "https://github.com/",
    },
    Case {
        name: "cb_short",
        dataset: Dataset::Cb,
        prefix: "https://www.",
    },
    Case {
        name: "cb_long",
        dataset: Dataset::Cb,
        prefix: "https://www.google.com/catalog/",
    },
    Case {
        name: "log_short",
        dataset: Dataset::Log,
        prefix: "192.168",
    },
    Case {
        name: "log_long",
        dataset: Dataset::Log,
        prefix: "203.0.113.50 - - [15/Mar/2024:10:",
    },
    Case {
        name: "json_short",
        dataset: Dataset::Json,
        prefix: r#"{"id"#,
    },
    Case {
        name: "json_long",
        dataset: Dataset::Json,
        prefix: r#"{"id":5000"#,
    },
    Case {
        name: "path_short",
        dataset: Dataset::Path,
        prefix: "/home",
    },
    Case {
        name: "path_long",
        dataset: Dataset::Path,
        prefix: "/home/user/target/release/",
    },
    Case {
        name: "email_short",
        dataset: Dataset::Email,
        prefix: "john",
    },
    Case {
        name: "email_long",
        dataset: Dataset::Email,
        prefix: "john.doe@",
    },
];

fn like_prefix(fsst: &FSSTArray, prefix: &str, ctx: &mut vortex_array::ExecutionCtx) -> BoolArray {
    let pattern = ConstantArray::new(format!("{prefix}%"), fsst.len()).into_array();
    Like::try_new(fsst.clone().into_array(), pattern, LikeOptions::default())
        .unwrap()
        .into_array()
        .execute::<BoolArray>(ctx)
        .unwrap()
}

/// The registered LIKE kernel.
#[divan::bench(args = CASES)]
fn prefix_like(bencher: Bencher, case: &Case) {
    let fsst = case.dataset.array();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| like_prefix(fsst, case.prefix, ctx));
}
