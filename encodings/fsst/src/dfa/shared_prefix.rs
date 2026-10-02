// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prefix matching (`LIKE 'prefix%'`) by comparing the codes every matching row opens with.
//!
//! FSST compresses greedily, and the code it picks at a position depends only on the (at most 8)
//! bytes starting there. So at every position of the prefix with at least 8 prefix bytes from it,
//! a row starting with the prefix is compressed exactly as the prefix alone is. The codes the
//! prefix compresses to at those positions are therefore a prefix of every matching row's codes
//! and are compared as raw bytes. Fewer than 8 prefix bytes remain after them; those are decoded
//! from the row and compared.
//!
//! Like code-space equality, this relies on the codes having been produced by the array's own
//! compressor.

use fsst::Compressor;
use fsst::ESCAPE_CODE;

/// The longest FSST symbol, in bytes.
const MAX_SYMBOL_LEN: usize = 8;

/// The fewest shared code bytes for which this matcher beats the prefix DFA.
///
/// Rows that open with the shared codes still decode a tail, so the shared codes have to reject
/// most rows on their own. On the `fsst_code_predicates` benchmark, 2 or more shared code bytes
/// measured 1.0-2.3x faster than the DFA, while 0 or 1 measured 1.3-7x slower.
const MIN_SHARED_CODE_BYTES: usize = 2;

pub(crate) struct SharedCodesPrefix {
    /// The prefix's leading codes, which every row starting with the prefix opens with.
    shared: Vec<u8>,
    /// The prefix bytes after `shared`, fewer than [`MAX_SYMBOL_LEN`].
    rest: Vec<u8>,
    /// Symbol bytes by code, zero-padded to 8 bytes.
    symbols: Box<[[u8; MAX_SYMBOL_LEN]; 256]>,
    /// Symbol lengths by code; zero for unused codes.
    lengths: Box<[u8; 256]>,
}

impl SharedCodesPrefix {
    /// Build the matcher, or `None` when the prefix yields too few shared codes for this to beat
    /// the DFA.
    pub(crate) fn try_new(compressor: &Compressor, prefix: &[u8]) -> Option<Self> {
        let symbol_lengths = compressor.symbol_lengths();
        let codes = compressor.compress(prefix);

        let mut consumed = 0;
        let mut shared_len = 0;
        while shared_len < codes.len() && consumed + MAX_SYMBOL_LEN <= prefix.len() {
            if codes[shared_len] == ESCAPE_CODE {
                consumed += 1;
                shared_len += 2;
            } else {
                consumed += usize::from(symbol_lengths[usize::from(codes[shared_len])]);
                shared_len += 1;
            }
        }
        if shared_len < MIN_SHARED_CODE_BYTES {
            return None;
        }

        let mut symbols = Box::new([[0u8; MAX_SYMBOL_LEN]; 256]);
        let mut lengths = Box::new([0u8; 256]);
        for (code, (symbol, &len)) in compressor
            .symbol_table()
            .iter()
            .zip(symbol_lengths)
            .enumerate()
        {
            symbols[code] = symbol.to_u64().to_le_bytes();
            lengths[code] = len;
        }

        Some(Self {
            shared: codes[..shared_len].to_vec(),
            rest: prefix[consumed..].to_vec(),
            symbols,
            lengths,
        })
    }

    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let Some(tail) = codes.strip_prefix(self.shared.as_slice()) else {
            return false;
        };

        // Fewer than 8 bytes are decoded before the last symbol, which writes at most 8 more.
        let mut decoded = [0u8; 2 * MAX_SYMBOL_LEN];
        let mut n = 0;
        let mut pos = 0;
        while n < self.rest.len() {
            let Some(&code) = tail.get(pos) else {
                return false;
            };
            if code == ESCAPE_CODE {
                let Some(&byte) = tail.get(pos + 1) else {
                    return false;
                };
                decoded[n] = byte;
                n += 1;
                pos += 2;
            } else {
                let code = usize::from(code);
                decoded[n..n + MAX_SYMBOL_LEN].copy_from_slice(&self.symbols[code]);
                n += usize::from(self.lengths[code]);
                pos += 1;
            }
        }
        decoded[..self.rest.len()] == *self.rest
    }

    #[cfg(test)]
    pub(crate) fn shared_len(&self) -> usize {
        self.shared.len()
    }
}
