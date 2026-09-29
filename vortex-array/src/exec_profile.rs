// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Thread-local profiling counters for the iterative executor.
//!
//! Enabled with the `exec-profile` feature. Every timed region in
//! [`execute_until`](crate::ArrayRef::execute_until) accumulates into the thread-local
//! [`ExecProfile`], which benchmarks read back with [`snapshot`] after calling [`reset`].
//! Without the feature the macros expand to the bare expression and cost nothing.

use std::cell::RefCell;

/// Accumulated counters and nanosecond timers for one thread's executor activity.
#[derive(Clone, Debug, Default)]
pub struct ExecProfile {
    /// Number of top-level `execute_until` calls.
    pub calls: u64,
    /// `execute_until` calls made from inside a kernel (`FoR`, `ZigZag`, `Zstd`, ... execute
    /// their children recursively instead of yielding `ExecuteSlot`).
    pub nested_calls: u64,
    /// Wall time spent inside nested calls (each counted once). Every timed region subtracts
    /// the nested time it contains, so the phase timers below are exclusive.
    pub nested_ns: u64,
    /// Wall time of top-level `execute_until` calls.
    pub total_ns: u64,
    /// Current nesting depth (bookkeeping).
    pub depth: u64,
    /// Loop iterations.
    pub iterations: u64,
    /// Time spent in the done-predicate and `AnyCanonical` checks.
    pub done_check_ns: u64,
    /// Frames popped because the child reached its target.
    pub pops: u64,
    /// Time in `pop_frame` (put_slot).
    pub pop_ns: u64,
    /// `put_slot_unchecked` calls that had to allocate a fresh parent because the `Arc` was shared.
    pub put_slot_shared: u64,
    /// Step 2a attempts (stacked parent).
    pub stack_ep_attempts: u64,
    /// Step 2a hits.
    pub stack_ep_hits: u64,
    /// Step 2b attempts (one per parent visit, not per child).
    pub child_ep_attempts: u64,
    /// Step 2b hits.
    pub child_ep_hits: u64,
    /// Registry lookups (`kernels.get(&key)`), across 2a and 2b.
    pub ep_lookups: u64,
    /// Lookups that found at least one registered kernel.
    pub ep_lookup_found: u64,
    /// Time spent hashing and probing the registry.
    pub ep_lookup_ns: u64,
    /// Kernel invocations that returned `None`.
    pub ep_declined: u64,
    /// Time spent inside kernels that declined.
    pub ep_declined_ns: u64,
    /// Kernel invocations that produced a result.
    pub ep_applied: u64,
    /// Time spent inside kernels that applied.
    pub ep_applied_ns: u64,
    /// `optimize_ctx` calls made after a parent kernel applied.
    pub optimize_calls: u64,
    /// Time spent in `optimize_ctx`.
    pub optimize_ns: u64,
    /// Time spent cloning the dtype / stats snapshot before `execute`.
    pub pre_execute_ns: u64,
    /// `V::execute` calls that returned `ExecuteSlot`.
    pub execute_slot_steps: u64,
    /// Time inside `V::execute` for calls that returned `ExecuteSlot`.
    pub execute_slot_ns: u64,
    /// `V::execute` calls that returned `AppendChild`.
    pub append_child_steps: u64,
    /// Time inside `V::execute` for calls that returned `AppendChild`.
    pub append_child_execute_ns: u64,
    /// `V::execute` calls that returned `Done`.
    pub done_steps: u64,
    /// Time inside `V::execute` for calls that returned `Done`.
    pub done_execute_ns: u64,
    /// Time in `take_slot_unchecked`.
    pub take_slot_ns: u64,
    /// `take_slot_unchecked` calls that had to allocate because the `Arc` was shared.
    pub take_slot_shared: u64,
    /// Time creating builders.
    pub builder_create_ns: u64,
    /// Time in `append_to_builder`.
    pub builder_append_ns: u64,
    /// Time in `finalize_done` (builder finish plus stats transfer).
    pub finalize_ns: u64,
    /// Maximum stack depth reached.
    pub max_depth: u64,
}

impl ExecProfile {
    /// Time attributed to encoding kernels doing real work: successful parent kernels, `Done`
    /// executes and builder appends.
    pub fn work_ns(&self) -> u64 {
        self.ep_applied_ns + self.done_execute_ns + self.builder_append_ns
    }

    /// Everything else inside `execute_until`.
    pub fn overhead_ns(&self) -> u64 {
        self.total_ns.saturating_sub(self.work_ns())
    }

    /// Time not covered by any timed region (loop control flow, stack pushes, etc).
    pub fn untracked_ns(&self) -> u64 {
        let tracked = self.done_check_ns
            + self.pop_ns
            + self.ep_lookup_ns
            + self.ep_declined_ns
            + self.ep_applied_ns
            + self.optimize_ns
            + self.pre_execute_ns
            + self.execute_slot_ns
            + self.append_child_execute_ns
            + self.done_execute_ns
            + self.take_slot_ns
            + self.builder_create_ns
            + self.builder_append_ns
            + self.finalize_ns;
        self.total_ns.saturating_sub(tracked)
    }
}

thread_local! {
    static PROFILE: RefCell<ExecProfile> = RefCell::new(ExecProfile::default());
}

/// Reset the current thread's counters.
pub fn reset() {
    PROFILE.with(|p| *p.borrow_mut() = ExecProfile::default());
}

/// Copy the current thread's counters.
pub fn snapshot() -> ExecProfile {
    PROFILE.with(|p| p.borrow().clone())
}

#[doc(hidden)]
pub fn with_mut(f: impl FnOnce(&mut ExecProfile)) {
    PROFILE.with(|p| f(&mut p.borrow_mut()));
}

/// Nanoseconds since `start`, saturating rather than truncating.
#[doc(hidden)]
pub fn elapsed_ns(start: std::time::Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Nested time accumulated so far on this thread.
#[doc(hidden)]
pub fn nested_ns() -> u64 {
    PROFILE.with(|p| p.borrow().nested_ns)
}

/// Bracket a (possibly nested) `execute_until` call: returns the depth on entry.
#[doc(hidden)]
pub fn enter_call() -> u64 {
    PROFILE.with(|p| {
        let mut p = p.borrow_mut();
        p.depth += 1;
        p.depth
    })
}

/// `nested_during` is the nested time accumulated while this call ran, so a nested call adds
/// only its exclusive time: summed over a subtree that equals the elapsed time of the subtree's
/// root, which is what the enclosing timed region subtracts.
#[doc(hidden)]
pub fn exit_call(depth: u64, elapsed_ns: u64, nested_during: u64) {
    PROFILE.with(|p| {
        let mut p = p.borrow_mut();
        p.depth -= 1;
        if depth == 1 {
            p.calls += 1;
            p.total_ns += elapsed_ns;
        } else {
            p.nested_calls += 1;
            p.nested_ns += elapsed_ns.saturating_sub(nested_during);
        }
    });
}

/// Time an expression into a nanosecond field of the thread-local profile, excluding any
/// nested `execute_until` calls made while it ran.
#[cfg(feature = "exec-profile")]
macro_rules! prof_time {
    ($field:ident, $e:expr) => {{
        let __nested = $crate::exec_profile::nested_ns();
        let __start = ::std::time::Instant::now();
        let __r = $e;
        let __ns = $crate::exec_profile::elapsed_ns(__start);
        let __inner = $crate::exec_profile::nested_ns() - __nested;
        $crate::exec_profile::with_mut(|p| p.$field += __ns.saturating_sub(__inner));
        __r
    }};
}

#[cfg(not(feature = "exec-profile"))]
macro_rules! prof_time {
    ($field:ident, $e:expr) => {
        $e
    };
}

/// Increment a counter field of the thread-local profile.
#[cfg(feature = "exec-profile")]
macro_rules! prof_count {
    ($field:ident) => {
        $crate::exec_profile::with_mut(|p| p.$field += 1)
    };
    ($field:ident, $n:expr) => {
        $crate::exec_profile::with_mut(|p| p.$field += ($n) as u64)
    };
}

#[cfg(not(feature = "exec-profile"))]
macro_rules! prof_count {
    ($field:ident) => {};
    ($field:ident, $n:expr) => {};
}

/// Run `$e`, then add elapsed time to one of two nanosecond fields depending on the outcome
/// predicate `$is_a`, and bump the matching counter.
#[cfg(feature = "exec-profile")]
macro_rules! prof_time_branch {
    (
        $e:expr, |
        $r:ident |
        $is_a:expr,($a_ns:ident, $a_count:ident),($b_ns:ident, $b_count:ident)
    ) => {{
        let __nested = $crate::exec_profile::nested_ns();
        let __start = ::std::time::Instant::now();
        let $r = $e;
        let __ns = $crate::exec_profile::elapsed_ns(__start)
            .saturating_sub($crate::exec_profile::nested_ns() - __nested);
        let __a = $is_a;
        $crate::exec_profile::with_mut(|p| {
            if __a {
                p.$a_ns += __ns;
                p.$a_count += 1;
            } else {
                p.$b_ns += __ns;
                p.$b_count += 1;
            }
        });
        $r
    }};
}

#[cfg(not(feature = "exec-profile"))]
macro_rules! prof_time_branch {
    (
        $e:expr, |
        $r:ident |
        $is_a:expr,($a_ns:ident, $a_count:ident),($b_ns:ident, $b_count:ident)
    ) => {
        $e
    };
}

pub(crate) use prof_count;
pub(crate) use prof_time;
pub(crate) use prof_time_branch;
