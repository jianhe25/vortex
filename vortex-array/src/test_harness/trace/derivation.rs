// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Renders a [`TraceDisplay`] as a walk through the small-step reduction semantics that the
//! optimizer and executor implement.
//!
//! Every rule, kernel, and encoding execution is a single step `a -> a'` on one node of the
//! array tree. The rendering prints the whole tree once, with each child labelled by its slot
//! index, and then one numbered line per step naming the node it rewrote by path (`@root`,
//! `@0`, `@0.1`):
//!
//! ```text
//! [execute_until AnyCanonical]
//! vortex.filter(i32, len=2)
//! └─(0) vortex.filter(i32, len=4)
//!    └─(0) vortex.primitive(i32, len=6)
//!
//! 1. execute vortex.filter @root: vortex.filter(i32, len=2) -> vortex.slice(i32, len=2)
//! 2. execute vortex.filter @0: vortex.filter(i32, len=4) -> vortex.primitive(i32, len=4)
//!    = vortex.slice(i32, len=2)
//!      └─(0) vortex.primitive(i32, len=4)
//! 3. execute vortex.slice @root: vortex.slice(i32, len=2) -> vortex.primitive(i32, len=2)
//!    │ [optimize]
//!    │ vortex.slice(i32, len=2)
//!    │ └─(0) vortex.primitive(i32, len=4)
//!    │ 1. reduce_parent static:SliceReduceAdaptor(Primitive) slot=0 @root: ...
//!    = vortex.primitive(i32, len=2)
//! ```
//!
//! Lines in a step's `│` gutter happened inside that step: a nested `optimize` or
//! `execute_until` pass, builder appends, and at
//! [`TraceResolution::Attempts`](super::TraceResolution::Attempts) the rules and kernels that
//! were tried and declined (`x ...`). The `=` block is the whole tree after the step; it is
//! printed only when the step changed the shape beneath the rewritten node, not just the node
//! itself. A nested pass whose root is a node of the enclosing tree is named by path; otherwise
//! its own tree is printed.

use std::fmt;
use std::fmt::Display;

use vortex_error::VortexExpect;

use super::TraceDisplay;
use super::TraceEvent;
use crate::ArrayRef;

/// A step-by-step rendering of a [`TraceDisplay`]; see the module docs.
pub struct DerivationDisplay<'a> {
    pub(super) trace: &'a TraceDisplay,
}

impl Display for DerivationDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = String::new();
        for (idx, pass) in build(self.trace).iter().enumerate() {
            if idx > 0 {
                out.push('\n');
            }
            pass.render(&mut out, "", true);
        }
        f.write_str(out.trim_end_matches('\n'))
    }
}

/// The array tree as displayed: one summary per node, children in slot order.
#[derive(Clone)]
struct Tree {
    array: ArrayRef,
    children: Vec<(usize, Tree)>,
}

impl Tree {
    fn of(array: &ArrayRef) -> Self {
        Self {
            array: array.clone(),
            children: array
                .slots()
                .iter()
                .enumerate()
                .filter_map(|(idx, slot)| slot.as_ref().map(|child| (idx, Tree::of(child))))
                .collect(),
        }
    }

    fn same_shape(&self, other: &Tree) -> bool {
        self.children.len() == other.children.len()
            && self.children.iter().zip(&other.children).all(|(a, b)| {
                a.0 == b.0 && a.1.array.to_string() == b.1.array.to_string() && a.1.same_shape(&b.1)
            })
    }

    /// Finds `array` by identity and returns its slot path.
    fn find(&self, array: &ArrayRef) -> Option<Vec<usize>> {
        if ArrayRef::ptr_eq(&self.array, array) {
            return Some(Vec::new());
        }
        self.children.iter().find_map(|(idx, child)| {
            child.find(array).map(|mut path| {
                path.insert(0, *idx);
                path
            })
        })
    }

    fn render(&self, out: &mut String, prefix: &str) {
        out.push_str(&self.array.to_string());
        out.push('\n');
        let last = self.children.len().saturating_sub(1);
        for (pos, (idx, child)) in self.children.iter().enumerate() {
            let (branch, extend) = if pos == last {
                ("└─", "   ")
            } else {
                ("├─", "│  ")
            };
            out.push_str(&format!("{prefix}{branch}({idx}) "));
            child.render(out, &format!("{prefix}{extend}"));
        }
    }
}

fn path_text(path: &[usize]) -> String {
    if path.is_empty() {
        "root".to_string()
    } else {
        path.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(".")
    }
}

/// Something that happened inside a step, before its result was known.
enum Inner {
    Note(String),
    Pass(Pass),
}

struct Step {
    title: String,
    inner: Vec<Inner>,
    after: Option<Tree>,
}

/// An `execute_until`, `optimize`, or single-step pass.
struct Pass {
    label: String,
    /// The path of the root within the enclosing tree, when it is a node of that tree.
    at: Option<Vec<usize>>,
    start: Tree,
    steps: Vec<Step>,
    /// Inner items that no step consumed.
    trailing: Vec<Inner>,
}

impl Pass {
    fn render(&self, out: &mut String, prefix: &str, top_level: bool) {
        match &self.at {
            Some(path) if !top_level => {
                out.push_str(&format!("{prefix}[{} @{}]\n", self.label, path_text(path)));
            }
            _ => {
                out.push_str(&format!("{prefix}[{}]\n", self.label));
                out.push_str(prefix);
                self.start.render(out, prefix);
            }
        }
        if top_level && !self.steps.is_empty() {
            out.push('\n');
        }
        for (idx, step) in self.steps.iter().enumerate() {
            out.push_str(&format!("{prefix}{}. {}\n", idx + 1, step.title));
            let gutter = format!("{prefix}   │ ");
            for inner in &step.inner {
                render_inner(inner, out, &gutter);
            }
            if let Some(after) = &step.after {
                out.push_str(&format!("{prefix}   = "));
                after.render(out, &format!("{prefix}     "));
            }
        }
        for inner in &self.trailing {
            render_inner(inner, out, &format!("{prefix}   "));
        }
    }
}

fn render_inner(inner: &Inner, out: &mut String, prefix: &str) {
    match inner {
        Inner::Note(text) => out.push_str(&format!("{prefix}{text}\n")),
        Inner::Pass(pass) => pass.render(out, prefix, false),
    }
}

enum FrameKind {
    Root,
    Pass {
        label: String,
        root: ArrayRef,
        /// The path of `root` within the enclosing pass's tree, when it is a node of it.
        at: Option<Vec<usize>>,
        single_step: bool,
        steps: Vec<Step>,
        pending: Vec<Inner>,
    },
    /// A child slot focused by `ExecuteSlot`; closed by `pop_frame` or a stack kernel.
    Slot {
        parent: ArrayRef,
        slot_idx: usize,
    },
    /// A builder driven by `AppendChild`; closed by `builder finish`.
    Builder,
}

struct Frame {
    kind: FrameKind,
    /// The array the next step in this frame rewrites.
    current: Option<ArrayRef>,
}

fn build(trace: &TraceDisplay) -> Vec<Pass> {
    let hidden = trace.hidden_events();
    let mut builder = Builder {
        frames: vec![Frame {
            kind: FrameKind::Root,
            current: None,
        }],
        passes: Vec::new(),
    };
    for (idx, event) in trace.events.iter().enumerate() {
        if !hidden[idx] {
            builder.event(event);
        }
    }
    // Passes left open by an error path are closed with what was recorded.
    while builder.frames.len() > 1 {
        builder.close_top();
    }
    builder.passes
}

struct Builder {
    frames: Vec<Frame>,
    passes: Vec<Pass>,
}

impl Builder {
    fn top(&mut self) -> &mut Frame {
        self.frames
            .last_mut()
            .vortex_expect("root frame is never popped")
    }

    fn push(&mut self, kind: FrameKind, current: &ArrayRef) {
        self.frames.push(Frame {
            kind,
            current: Some(current.clone()),
        });
    }

    fn pop(&mut self) -> Option<Frame> {
        (self.frames.len() > 1).then(|| self.frames.pop()).flatten()
    }

    fn current(&self) -> Option<&ArrayRef> {
        self.frames.last().and_then(|frame| frame.current.as_ref())
    }

    /// The index of the innermost pass frame.
    fn pass_index(&self) -> Option<usize> {
        self.frames
            .iter()
            .rposition(|frame| matches!(frame.kind, FrameKind::Pass { .. }))
    }

    /// The slot path from the innermost pass root to the current array.
    fn path(&self) -> Vec<usize> {
        let start = self.pass_index().map_or(0, |idx| idx + 1);
        self.frames[start..]
            .iter()
            .filter_map(|frame| match &frame.kind {
                FrameKind::Slot { slot_idx, .. } => Some(*slot_idx),
                _ => None,
            })
            .collect()
    }

    /// The whole tree of the innermost pass, with `leaf` in place of the current array.
    fn compose(&self, leaf: &ArrayRef) -> Tree {
        let start = self.pass_index().map_or(0, |idx| idx + 1);
        let mut tree = Tree::of(leaf);
        for frame in self.frames[start..].iter().rev() {
            let FrameKind::Slot { parent, slot_idx } = &frame.kind else {
                continue;
            };
            let children = parent
                .slots()
                .iter()
                .enumerate()
                .filter_map(|(idx, slot)| {
                    if idx == *slot_idx {
                        Some((idx, tree.clone()))
                    } else {
                        slot.as_ref().map(|child| (idx, Tree::of(child)))
                    }
                })
                .collect();
            tree = Tree {
                array: parent.clone(),
                children,
            };
        }
        tree
    }

    fn pending(&mut self) -> Option<&mut Vec<Inner>> {
        let idx = self.pass_index()?;
        match &mut self.frames[idx].kind {
            FrameKind::Pass { pending, .. } => Some(pending),
            _ => None,
        }
    }

    fn note(&mut self, text: String) {
        if let Some(pending) = self.pending() {
            pending.push(Inner::Note(text));
        }
    }

    /// The whole tree of the innermost pass as it stands now.
    fn before(&self) -> Option<Tree> {
        self.current().map(|current| self.compose(current))
    }

    /// Records the step `current -> output` justified by `rule` at the current path.
    fn step(&mut self, rule: &str, output: &ArrayRef) {
        let before = self.before();
        self.step_from(before, rule, output);
    }

    /// Records a step whose tree `before` it was captured earlier, when the frames that
    /// produced it have since been popped.
    fn step_from(&mut self, before: Option<Tree>, rule: &str, output: &ArrayRef) {
        let lhs_text = self
            .current()
            .map_or_else(|| "?".to_string(), ToString::to_string);
        let path = path_text(&self.path());
        let title = format!("{rule} @{path}: {lhs_text} -> {output}");
        let after_tree = self.compose(output);
        let after = match before {
            Some(before) if before.same_shape(&after_tree) => None,
            _ => Some(after_tree),
        };
        let Some(idx) = self.pass_index() else { return };
        if let FrameKind::Pass { steps, pending, .. } = &mut self.frames[idx].kind {
            steps.push(Step {
                title,
                inner: std::mem::take(pending),
                after,
            });
        }
        self.top().current = Some(output.clone());
    }

    fn close_top(&mut self) {
        let Some(frame) = self.frames.last() else {
            return;
        };
        match &frame.kind {
            FrameKind::Root => {}
            FrameKind::Pass { .. } => self.close_pass(),
            FrameKind::Slot { .. } => self.close_slot(None),
            FrameKind::Builder => {
                let output = frame.current.clone();
                self.close_builder(output.as_ref());
            }
        }
    }

    fn open_pass(&mut self, label: String, root: &ArrayRef, single_step: bool) {
        let at = self
            .current()
            .map(|current| self.compose(current))
            .and_then(|tree| tree.find(root));
        self.push(
            FrameKind::Pass {
                label,
                root: root.clone(),
                at,
                single_step,
                steps: Vec::new(),
                pending: Vec::new(),
            },
            root,
        );
    }

    fn close_pass(&mut self) {
        let Some(frame) = self.pop() else { return };
        let FrameKind::Pass {
            label,
            root,
            at,
            steps,
            pending,
            ..
        } = frame.kind
        else {
            return;
        };
        let pass = Pass {
            label,
            at,
            start: Tree::of(&root),
            steps,
            trailing: pending,
        };
        match self.pending() {
            Some(pending) => pending.push(Inner::Pass(pass)),
            None => self.passes.push(pass),
        }
    }

    fn close_if_single_step(&mut self) {
        if matches!(
            self.top().kind,
            FrameKind::Pass {
                single_step: true,
                ..
            }
        ) {
            self.close_pass();
        }
    }

    /// Leaves a focused slot. With `output` (from `pop_frame`) the parent is restored to it.
    fn close_slot(&mut self, output: Option<&ArrayRef>) {
        let Some(frame) = self.pop() else { return };
        let FrameKind::Slot { .. } = frame.kind else {
            return;
        };
        if let Some(output) = output {
            self.top().current = Some(output.clone());
        }
    }

    fn close_builder(&mut self, output: Option<&ArrayRef>) {
        let Some(_) = self.pop() else { return };
        match output {
            Some(output) => self.step("builder", output),
            None => self.note("builder: unfinished".to_string()),
        }
        self.close_if_single_step();
    }

    fn event(&mut self, event: &TraceEvent) {
        match event {
            TraceEvent::OptimizeStart { root, session } => {
                let session = if *session { " session" } else { "" };
                self.open_pass(format!("optimize{session}"), &root.0, false);
            }
            TraceEvent::OptimizeLoopStart { array } => self.top().current = Some(array.0.clone()),
            TraceEvent::OptimizeLoopEnd
            | TraceEvent::ExecuteUntilDoneCheck { .. }
            | TraceEvent::PhaseNone { .. }
            | TraceEvent::ExecuteEncoding { .. } => {}
            TraceEvent::OptimizeDone { .. } => {
                if matches!(self.top().kind, FrameKind::Pass { .. }) {
                    self.close_pass();
                }
            }
            TraceEvent::OptimizeRecursiveStart { root } => {
                self.top().current = Some(root.0.clone());
            }
            TraceEvent::OptimizeRecursiveSlot {
                slot_idx,
                input,
                output,
            } => {
                let path = path_text(&[self.path(), vec![*slot_idx]].concat());
                self.note(format!("optimize_recursive @{path}: {input} -> {output}"));
            }
            TraceEvent::ReduceAttempt { rule, outcome, .. } => {
                self.note(format!("x reduce {rule}: {outcome}"));
            }
            TraceEvent::ReduceApplied { rule, output, .. } => {
                self.step(&format!("reduce {rule}"), &output.0);
            }
            TraceEvent::ParentReduceAttempt {
                slot_idx,
                source,
                rule,
                outcome,
                ..
            } => self.note(format!(
                "x reduce_parent {source}:{rule} slot={slot_idx}: {outcome}"
            )),
            TraceEvent::ParentReduceApplied {
                slot_idx,
                source,
                rule,
                output,
                ..
            } => self.step(
                &format!("reduce_parent {source}:{rule} slot={slot_idx}"),
                &output.0,
            ),
            TraceEvent::ExecuteUntilStart { target, root } => {
                self.open_pass(format!("execute_until {target}"), &root.0, false);
            }
            TraceEvent::ExecuteUntilIteration { current, .. } => {
                self.top().current = Some(current.0.clone());
            }
            TraceEvent::ExecuteUntilReturn { .. } => {
                if matches!(self.top().kind, FrameKind::Pass { .. }) {
                    self.close_pass();
                }
            }
            TraceEvent::ExecuteUntilPopFrame { output, .. } => self.close_slot(Some(&output.0)),
            TraceEvent::ExecuteParentAttempt {
                phase,
                slot_idx,
                source,
                kernel,
                outcome,
                ..
            } => self.note(format!(
                "x {phase} {source}:{kernel} slot={slot_idx}: {outcome}"
            )),
            TraceEvent::ExecuteParentApplied {
                phase,
                slot_idx,
                source,
                kernel,
                output,
                ..
            } => {
                let before = self.before();
                if *phase == "stack_execute_parent"
                    && matches!(self.top().kind, FrameKind::Slot { .. })
                {
                    // The kernel rewrote the suspended parent, consuming the focused slot.
                    let Some(frame) = self.pop() else { return };
                    if let FrameKind::Slot { parent, .. } = frame.kind {
                        self.top().current = Some(parent);
                    }
                }
                self.step_from(
                    before,
                    &format!("{phase} {source}:{kernel} slot={slot_idx}"),
                    &output.0,
                );
            }
            TraceEvent::ExecuteOptimized {
                output, changed, ..
            } => {
                if *changed {
                    self.step("optimize_ctx", &output.0);
                } else if let Some(pending) = self.pending() {
                    // An optimize pass that changed nothing justifies nothing; keep only
                    // items that carry information of their own.
                    pending.retain(|inner| match inner {
                        Inner::Pass(pass) => !pass.steps.is_empty(),
                        Inner::Note(_) => true,
                    });
                }
            }
            TraceEvent::SlotTransition {
                step,
                slot_idx,
                parent,
                child,
            } => match *step {
                "ExecuteSlot" => self.push(
                    FrameKind::Slot {
                        parent: parent.0.clone(),
                        slot_idx: *slot_idx,
                    },
                    &child.0,
                ),
                _ => self.note(format!("append ({slot_idx}) {child}")),
            },
            TraceEvent::BuilderEvent { action, array, .. } => match *action {
                "start" => self.push(FrameKind::Builder, &array.0),
                "finish" => self.close_builder(Some(&array.0)),
                _ => {}
            },
            TraceEvent::ExecuteDone { array } => {
                if matches!(self.top().kind, FrameKind::Builder) {
                    // `builder finish` follows and carries the real output.
                    return;
                }
                let encoding = self
                    .current()
                    .map_or_else(|| "?".to_string(), |a| a.encoding_id().to_string());
                self.step(&format!("execute {encoding}"), &array.0);
                self.close_if_single_step();
            }
            TraceEvent::SingleStepStart { array } => {
                self.open_pass("execute_step".to_string(), &array.0, true);
            }
            TraceEvent::SingleStepApplied { phase, output, .. } => {
                self.step(phase, &output.0);
                self.close_if_single_step();
            }
        }
    }
}
