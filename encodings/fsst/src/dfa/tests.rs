// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::LazyLock;

use fsst::Compressor;
use fsst::ESCAPE_CODE;
use fsst::Symbol;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::FsstMatcher;
use super::LikeKind;
use super::MatcherInner;
use super::flat_contains::FlatContainsDfa;
use super::prefix::FlatPrefixDfa;
use crate::FSSTArray;
use crate::fsst_compress;
use crate::fsst_train_compressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// Helper: make a Symbol from a byte string (up to 8 bytes, zero-padded).
fn sym(bytes: &[u8]) -> Symbol {
    let mut buf = [0u8; 8];
    buf[..bytes.len()].copy_from_slice(bytes);
    Symbol::from_slice(&buf)
}

/// A compressor with an empty symbol table, which escapes every byte.
fn no_symbols() -> Compressor {
    Compressor::rebuild_from(&[] as &[Symbol], &[] as &[u8])
}

fn escaped(bytes: &[u8]) -> Vec<u8> {
    let mut codes = Vec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        codes.push(ESCAPE_CODE);
        codes.push(b);
    }
    codes
}

fn assert_borrowed_prefix(pattern: &[u8], expected: &[u8]) {
    let Some(LikeKind::Prefix(actual)) = LikeKind::parse(pattern) else {
        panic!("expected borrowed prefix pattern");
    };
    assert!(matches!(actual, Cow::Borrowed(_)));
    assert_eq!(actual.as_ref(), expected);
}

fn assert_owned_prefix(pattern: &[u8], expected: &[u8]) {
    let Some(LikeKind::Prefix(actual)) = LikeKind::parse(pattern) else {
        panic!("expected owned prefix pattern");
    };
    assert!(matches!(actual, Cow::Owned(_)));
    assert_eq!(actual.as_ref(), expected);
}

fn assert_borrowed_contains(pattern: &[u8], expected: &[u8]) {
    let Some(LikeKind::Contains(actual)) = LikeKind::parse(pattern) else {
        panic!("expected borrowed contains pattern");
    };
    assert!(matches!(actual, Cow::Borrowed(_)));
    assert_eq!(actual.as_ref(), expected);
}

fn assert_owned_contains(pattern: &[u8], expected: &[u8]) {
    let Some(LikeKind::Contains(actual)) = LikeKind::parse(pattern) else {
        panic!("expected owned contains pattern");
    };
    assert!(matches!(actual, Cow::Owned(_)));
    assert_eq!(actual.as_ref(), expected);
}

#[test]
fn test_like_kind_parse_plain_patterns() {
    assert_borrowed_prefix(b"http%", b"http");
    assert_borrowed_contains(b"%needle%", b"needle");
    assert_borrowed_prefix(b"%", b"");
}

#[test]
fn test_like_kind_parse_escaped_patterns() {
    assert_owned_prefix(br"\%%", b"%");
    assert_owned_prefix(br"\_%", b"_");
    assert_owned_prefix(br"\\%", b"\\");
    assert_owned_prefix(br"has\%middle%", b"has%middle");
    assert_owned_contains(br"%\%%", b"%");
    assert_owned_contains(br"%\_%", b"_");
    assert_owned_contains(br"%\\%", b"\\");
    assert_owned_contains(br"%has\%middle%", b"has%middle");
}

#[test]
fn test_like_kind_parse_unsupported_patterns() {
    assert!(LikeKind::parse(b"%suffix").is_none());
    assert!(LikeKind::parse(b"a_c").is_none());
    assert!(LikeKind::parse(br"%\%").is_none());
    assert!(LikeKind::parse(br"foo\%bar").is_none());
}

/// No symbols — all bytes escaped. Simplest case to see the two tables.
#[test]
fn test_prefix_dfa_no_symbols() -> VortexResult<()> {
    let dfa = FlatPrefixDfa::new(&[], &[], b"ab")?;

    assert!(dfa.matches(&escaped(b"abx")));
    assert!(dfa.matches(&escaped(b"ab")));
    assert!(!dfa.matches(&escaped(b"a")));
    assert!(!dfa.matches(&escaped(b"ax")));
    assert!(!dfa.matches(&escaped(b"ba")));
    assert!(!dfa.matches(&[]));

    Ok(())
}

/// With symbols — shows how multi-byte symbols interact with prefix matching.
///
/// Symbol table: code 0 = "ht", code 1 = "tp"
/// Prefix: "http"
///
/// The string "http" can be encoded as:
///   [0, 1]           — two symbols: "ht" + "tp"
///   [ESC,h, ESC,t, ESC,t, ESC,p] — all escaped
///   [0, ESC,t, ESC,p]            — symbol "ht" + escaped "t" + escaped "p"
#[test]
fn test_prefix_dfa_with_symbols() -> VortexResult<()> {
    let symbols = [sym(b"ht"), sym(b"tp")];
    let lengths = [2u8, 2];
    let dfa = FlatPrefixDfa::new(&symbols, &lengths, b"http")?;

    // "http" via two symbols: code 0 ("ht") + code 1 ("tp") → accept
    assert!(dfa.matches(&[0, 1]));

    // "http" all escaped
    assert!(dfa.matches(&escaped(b"http")));

    // "http" mixed: symbol "ht" + escaped "tp"
    assert!(dfa.matches(&[0, ESCAPE_CODE, b't', ESCAPE_CODE, b'p']));

    // "htxx" via symbol "ht" + escaped "xx" → fail after "ht" advances to state 2,
    // then 'x' doesn't match 't'
    assert!(!dfa.matches(&[0, ESCAPE_CODE, b'x', ESCAPE_CODE, b'x']));

    // "tp" alone → symbol "tp" from state 0 feeds 't','p' through byte table:
    // state 0 wants 'h', sees 't' → fail
    assert!(!dfa.matches(&[1]));

    Ok(())
}

/// Longer prefix showing more progress states.
#[test]
fn test_prefix_dfa_longer() -> VortexResult<()> {
    // code 0 = "tp" (2 bytes), code 1 = "htt" (3 bytes), code 2 = "p:/" (3 bytes)
    let symbols = [sym(b"tp"), sym(b"htt"), sym(b"p:/")];
    let lengths = [2u8, 3, 3];
    let dfa = FlatPrefixDfa::new(&symbols, &lengths, b"http://")?;

    // "http://e" via symbols: "htt"(1) + "p:/"(2) + escaped "/" + escaped "e"
    // "htt" = states 0→1→2→3, "p:/" = states 3→4→5→6, "/" = state 6→accept
    assert!(dfa.matches(&[1, 2, ESCAPE_CODE, b'/', ESCAPE_CODE, b'e']));

    // "http:/" — 6 chars, missing the 7th '/'
    assert!(!dfa.matches(&[1, ESCAPE_CODE, b'p', ESCAPE_CODE, b':', ESCAPE_CODE, b'/',]));

    // "http://" all escaped — 7 chars, exact match
    assert!(dfa.matches(&escaped(b"http://")));

    // "tp" alone (code 0) from state 0: feeds 't','p' → state 0 wants 'h', sees 't' → fail
    assert!(!dfa.matches(&[0]));

    // "htt" + "tp" = "httpp"? No — "htt" → states 0→1→2→3, then "tp":
    // state 3 wants 'p', sees 't' → fail immediately
    assert!(!dfa.matches(&[1, 0]));

    Ok(())
}

#[test]
fn test_prefix_pushdown_len_13_with_escapes() {
    let matcher = FsstMatcher::try_new(&no_symbols(), b"abcdefghijklm%")
        .unwrap()
        .unwrap();

    assert!(matcher.matches(&escaped(b"abcdefghijklm")));
    assert!(!matcher.matches(&escaped(b"abcdefghijklx")));
}

#[test]
fn test_prefix_pushdown_len_14_now_handled() {
    // 14-byte prefix is now handled by FlatPrefixDfa (was rejected by shift-packed).
    assert!(
        FsstMatcher::try_new(&no_symbols(), b"abcdefghijklmn%")
            .unwrap()
            .is_some()
    );
}

#[test]
fn test_prefix_pushdown_long_prefix() -> VortexResult<()> {
    let prefix = "a".repeat(FlatPrefixDfa::MAX_PREFIX_LEN);
    let pattern = format!("{prefix}%");
    let matcher = FsstMatcher::try_new(&no_symbols(), pattern.as_bytes())?.unwrap();

    assert!(matcher.matches(&escaped(prefix.as_bytes())));

    let mut mismatch = prefix.into_bytes();
    mismatch[FlatPrefixDfa::MAX_PREFIX_LEN - 1] = b'b';
    assert!(!matcher.matches(&escaped(&mismatch)));

    Ok(())
}

#[test]
fn test_prefix_pushdown_beyond_dfa_limit_uses_shared_codes() -> VortexResult<()> {
    debug_assert_eq!(FlatPrefixDfa::MAX_PREFIX_LEN, 253);
    let prefix = "a".repeat(254);
    let pattern = format!("{prefix}%");
    let matcher = FsstMatcher::try_new(&no_symbols(), pattern.as_bytes())?.unwrap();
    assert!(matches!(matcher.inner, MatcherInner::SharedCodesPrefix(_)));

    assert!(matcher.matches(&escaped(prefix.as_bytes())));
    assert!(!matcher.matches(&escaped(&prefix.as_bytes()[..253])));

    let mut mismatch = prefix.into_bytes();
    mismatch[253] = b'b';
    assert!(!matcher.matches(&escaped(&mismatch)));
    Ok(())
}

#[test]
fn test_contains_pushdown_len_254_with_escapes() {
    let needle = "a".repeat(FlatContainsDfa::MAX_NEEDLE_LEN);
    let pattern = format!("%{needle}%");
    let matcher = FsstMatcher::try_new(&no_symbols(), pattern.as_bytes())
        .unwrap()
        .unwrap();

    assert!(matcher.matches(&escaped(needle.as_bytes())));

    let mut mismatch = needle.into_bytes();
    mismatch[FlatContainsDfa::MAX_NEEDLE_LEN - 1] = b'b';
    assert!(!matcher.matches(&escaped(&mismatch)));
}

#[test]
fn test_contains_pushdown_rejects_len_255() {
    let needle = "a".repeat(FlatContainsDfa::MAX_NEEDLE_LEN + 1);
    let pattern = format!("%{needle}%");
    assert!(
        FsstMatcher::try_new(&no_symbols(), pattern.as_bytes())
            .unwrap()
            .is_none()
    );
}

// ---------------------------------------------------------------------------
// End-to-end edge cases: FSST compress → LIKE → compare booleans
// ---------------------------------------------------------------------------

fn make_fsst_str(strings: &[Option<&str>]) -> FSSTArray {
    let array = VarBinArray::from_iter(
        strings.iter().copied(),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressor = fsst_train_compressor(&array, &mut ctx).unwrap();
    fsst_compress(&array, &compressor, &mut ctx).unwrap()
}

fn run_like(array: FSSTArray, pattern_arr: ArrayRef) -> VortexResult<BoolArray> {
    let arr: ArrayRef = array.into_array();
    let result = Like::try_new(arr, pattern_arr, LikeOptions::default())?
        .into_array()
        .execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
    Ok(result.into_bool())
}

#[rstest]
// Empty strings
#[case(&[""], "aaaa%", &[false])]
#[case(&[""], "%aaaa%", &[false])]
#[case(&[""], "%", &[true])]
#[case(&[""], "%%", &[true])]
#[case(&["", "", ""], "%", &[true, true, true])]
#[case(&["", "abc", ""], "%%", &[true, true, true])]
// Single-char patterns
#[case(&["a", "b", ""], "a%", &[true, false, false])]
#[case(&["a", "b", ""], "%a%", &[true, false, false])]
// Needle longer than every input string
#[case(&["ab", "abc", ""], "%abcd%", &[false, false, false])]
#[case(&["ab", "abc", ""], "abcd%", &[false, false, false])]
// Exact match (prefix pattern = entire string + %)
#[case(&["abc", "abcd", "ab"], "abc%", &[true, true, false])]
#[case(&["abc", "abcd", "ab"], "%abc%", &[true, true, false])]
// Repeated characters — KMP overlap
#[case(&["aa", "aaa", "aaaa", "aba"], "%aaa%", &[false, true, true, false])]
#[case(&["aab", "aaab", "a"], "aaa%", &[false, true, false])]
// Needle at different positions
#[case(&["xxabcyy", "abcyy", "xxabc", "abc", "xabx"], "%abc%", &[true, true, true, true, false])]
// All identical strings
#[case(&["aaa", "aaa", "aaa"], "%aaa%", &[true, true, true])]
#[case(&["aaa", "aaa", "aaa"], "bbb%", &[false, false, false])]
// Single element arrays
#[case(&["hello"], "hello%", &[true])]
#[case(&["hello"], "hellx%", &[false])]
#[case(&["hello"], "%ello%", &[true])]
#[case(&["hello"], "%ellx%", &[false])]
// Overlapping KMP pattern "abab"
#[case(&["ababab", "abab", "aba", "xababx"], "%abab%", &[true, true, false, true])]
// Prefix that shares chars with rest of string
#[case(&["abab", "abba", "abcd"], "ab%", &[true, true, true])]
#[case(&["abab", "abba", "abcd", "ba"], "ab%", &[true, true, true, false])]
// The string "aabaabaabaab" requires multi-level KMP fallback at the 'a' after "aabaabaab"
#[case(&["aabaabaabaab", "aabaabaax", "xaabaabaab"], "%aabaabaab%", &[true, false, true])]
#[case(&["café latte", "naïve approach", "café noir"], "café%", &[true, false, true])]
#[case(&["日本語テスト", "日本語データ", "英語テスト"], "%日本語%", &[true, true, false])]
// 10-byte needle, contains: match at start, middle, end, exact, and near-miss
#[case(
    &["abcdefghijxxx", "xxxabcdefghij", "xxabcdefghijxx", "abcdefghij", "abcdefghxx"],
    "%abcdefghij%",
    &[true, true, true, true, false]
)]
// 10-byte prefix: same needle but anchored at the start of the string
#[case(
    &["abcdefghijxxx", "abcdefghij", "xabcdefghij", "abcdefghxx"],
    "abcdefghij%",
    &[true, true, false, false]
)]
// 9-byte needle with KMP-relevant overlap ("abcabcabc"):
// failure table = [0,0,0,1,2,3,4,5,6], so a partial match of "abcabcab"
// followed by a mismatch must fall back to state 5 ("abcab"), not restart.
// This exercises multi-level KMP backtracking across symbol boundaries.
#[case(
    &["xxabcabcabcxx", "abcabcabc", "abcabcabx", "abcabcxx"],
    "%abcabcabc%",
    &[true, true, false, false]
)]
fn test_like_edge_cases(
    #[case] strings: &[&str],
    #[case] pattern: &str,
    #[case] expected: &[bool],
) -> VortexResult<()> {
    let opts: Vec<Option<&str>> = strings.iter().map(|s| Some(*s)).collect();
    let fsst_arr = make_fsst_str(&opts);
    let result = run_like(
        fsst_arr,
        ConstantArray::new(pattern, opts.len()).into_array(),
    )?;
    let expected_arr = BoolArray::from_iter(expected.iter().copied());
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(&result, &expected_arr, &mut ctx);
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared-codes prefix matching
// ---------------------------------------------------------------------------

fn url_compressor() -> Compressor {
    let urls = [
        "https://www.example.com/catalog/electronics/phones",
        "https://www.example.com/catalog/home/garden",
        "https://www.example.com/search?q=phones",
        "https://docs.example.org/guide/getting-started",
        "http://www.example.com/catalog/electronics/laptops",
    ];
    let lines: Vec<&[u8]> = urls
        .iter()
        .cycle()
        .take(500)
        .map(|u| u.as_bytes())
        .collect();
    Compressor::train(&lines)
}

#[test]
fn test_prefix_matcher_selection() -> VortexResult<()> {
    let compressor = url_compressor();

    // Fewer than 8 bytes leave no position with a full symbol window: always the DFA.
    let short = FsstMatcher::try_new(&compressor, b"https%")?.unwrap();
    assert!(matches!(short.inner, MatcherInner::Prefix(_)));

    let long = FsstMatcher::try_new(&compressor, b"https://www.example.com/catalog/%")?.unwrap();
    let MatcherInner::SharedCodesPrefix(shared) = &long.inner else {
        panic!("expected the shared-codes matcher for a long prefix");
    };
    assert!(shared.shared_len() >= 2);
    Ok(())
}

#[rstest]
#[case::urls("https://www.example.com/catalog/")]
#[case::exact_row("https://www.example.com/search?q=phones")]
#[case::longer_than_rows("https://www.example.com/catalog/electronics/phones/and/more")]
#[case::untrained_bytes("https://www.example.com/ñandú/")]
#[case::near_miss("https://www.example.com/catalog/electronicz")]
fn test_shared_codes_prefix_like(#[case] prefix: &str) -> VortexResult<()> {
    let strings = [
        "https://www.example.com/catalog/electronics/phones",
        "https://www.example.com/catalog/home/garden",
        "https://www.example.com/search?q=phones",
        "https://www.example.com/ñandú/pájaro",
        "https://www.example.com/catalo",
        "http://www.example.com/catalog/electronics/laptops",
        "https://docs.example.org/guide/getting-started",
        "",
    ];
    let opts: Vec<Option<&str>> = strings.iter().map(|s| Some(*s)).collect();
    let fsst_arr = make_fsst_str(&opts);

    let pattern = format!("{prefix}%");
    let matcher = FsstMatcher::try_new(fsst_arr.compressor(), pattern.as_bytes())?.unwrap();
    assert!(matches!(matcher.inner, MatcherInner::SharedCodesPrefix(_)));

    let result = run_like(
        fsst_arr,
        ConstantArray::new(pattern.as_str(), opts.len()).into_array(),
    )?;
    let expected = BoolArray::from_iter(strings.iter().map(|s| s.starts_with(prefix)));
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(&result, &expected, &mut ctx);
    Ok(())
}

/// The shared-codes matcher relies on FSST compressing a row that starts with the prefix the
/// same way it compresses the prefix alone. Check it against `starts_with` on random rows over a
/// small alphabet, so symbols overlap heavily, with prefixes cut from rows and then perturbed.
#[test]
fn test_shared_codes_prefix_matches_starts_with() {
    let mut rng = StdRng::seed_from_u64(7);
    let alphabet = b"abcab/.-_";
    let rows: Vec<Vec<u8>> = (0..400)
        .map(|_| {
            let len = rng.random_range(0..48);
            (0..len)
                .map(|_| alphabet[rng.random_range(0..alphabet.len())])
                .collect()
        })
        .collect();
    let lines: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
    let compressor = Compressor::train(&lines);
    let codes: Vec<Vec<u8>> = rows.iter().map(|row| compressor.compress(row)).collect();

    let mut checked = 0;
    for _ in 0..300 {
        let source = &rows[rng.random_range(0..rows.len())];
        if source.len() < 8 {
            continue;
        }
        let mut prefix = source[..rng.random_range(8..=source.len())].to_vec();
        if rng.random_bool(0.3) {
            let at = rng.random_range(0..prefix.len());
            prefix[at] = alphabet[rng.random_range(0..alphabet.len())];
        }

        let Some(matcher) = super::SharedCodesPrefix::try_new(&compressor, &prefix) else {
            continue;
        };
        checked += 1;
        for (row, row_codes) in rows.iter().zip(&codes) {
            assert_eq!(
                matcher.matches(row_codes),
                row.starts_with(&prefix),
                "prefix {:?} on row {:?}",
                String::from_utf8_lossy(&prefix),
                String::from_utf8_lossy(row),
            );
        }
    }
    assert!(
        checked > 100,
        "only {checked} prefixes used the shared-codes matcher"
    );
}
