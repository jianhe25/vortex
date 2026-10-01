// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Explicit ownership of aggregate results and producer guarantees for one input.
//!
//! An [`ArrayInput`] retains its array and the same aggregate store used by array-owned caching.
//! Its clones share both. Subsets get fresh stores: positive integer sortedness survives stable
//! selection, while exact extrema become bounds. Facts belong to the exact array handle and do
//! not travel implicitly through execution rewrites or array serialization.

mod bounds;
use self::bounds::VerifiedIntegerBounds;

mod execution;
pub(crate) use self::execution::AggregateInputFrame;

mod guarantees;
mod owner;
mod partial;

use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::OnceLock;

use crate::ArrayRef;
use crate::stats::Aggregations;

/// Selects result ownership for the input experiment.
///
/// This changes lookup and population for the exercised aggregate paths. All modes retain the
/// array's existing cache storage; disabling access does not remove that storage allocation.
/// Explicitly retained typed partial states always belong to the input owner.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AggregateCacheMode {
    /// Retain finalized results on the underlying array.
    Array,
    /// Retain finalized results on this input owner and bypass the array cache.
    #[default]
    Input,
    /// Bypass both result stores, including producer guarantees.
    Disabled,
}

struct ArrayInputInner {
    array: ArrayRef,
    aggregations: Aggregations,
    // Generic aggregate plugins are safe to implement. Their results cannot mint the proof used
    // to bypass an unsafe decimal constructor's value-precision precondition.
    verified_bounds: OnceLock<VerifiedIntegerBounds>,
}

/// An input array and its explicitly retained aggregate results.
///
/// Construction accepts any encoding. Clones share one store. Independently wrapping the same
/// array creates a new store, and cloning out an [`ArrayRef`] does not retain this owner.
#[derive(Clone)]
pub struct ArrayInput {
    inner: Arc<ArrayInputInner>,
    cache_mode: AggregateCacheMode,
}

impl Debug for ArrayInput {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayInput")
            .field("array", &self.inner.array)
            .field("cache_mode", &self.cache_mode)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
