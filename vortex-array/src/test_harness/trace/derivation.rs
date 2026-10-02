// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Renders a [`TraceDisplay`] as a set of derivations in the small-step reduction semantics that
//! the optimizer and executor implement.
//!
//! A derivation is one reduction sequence on one array: its starting tree, with each child
//! labelled by its slot name, followed by its steps. Each step is a pair, the array before and
//! the array after, under the rule that justified it. A step that needed another reduction
//! first, such as a child the parent asked to have reduced or an optimize pass run inside an
//! encoding's execute, cites it as `Dn`. Derivation `Dn` is written out in full further down, in
//! the order it is first cited, the way a proof names a subderivation and gives it separately.
//!
//! ```text
//! [execute_until AnyCanonical]
//! vortex.filter(i32, len=2)
//! └─child: vortex.filter(i32, len=4)
//!    └─child: vortex.primitive(i32, len=6)
//!
//! 1. execute vortex.filter
//!    vortex.filter(i32, len=2)
//!    -> vortex.slice(i32, len=2)
//! 2. child reduced, by D1
//!    vortex.slice(i32, len=2)
//!    -> vortex.slice(i32, len=2)
//!       └─child: vortex.primitive(i32, len=4)
//! 3. execute vortex.slice, by D2
//!    ...
//!
//! D1: child of vortex.slice(i32, len=2)
//! vortex.filter(i32, len=4)
//! └─child: vortex.primitive(i32, len=6)
//!
//! 1. execute vortex.filter
//!    ...
//! ```
//!
//! After a step, the whole tree below the array is printed only when the step changed it. When
//! only the array itself changed, the right side is its one-line summary. Rules that act on a
//! parent through one of its children name that child: `reduce_parent ... from codes`. At
//! [`TraceResolution::Attempts`](super::TraceResolution::Attempts), rules and kernels that were
//! tried and declined before a step are listed under it as `x ...`.

use std::collections::VecDeque;
use std::fmt;
use std::fmt::Display;

use vortex_error::VortexExpect;

use super::TraceDisplay;
use super::TraceEvent;
use crate::ArrayRef;

/// A rendering of a [`TraceDisplay`] as derivations; see the module docs.
pub struct DerivationDisplay<'a> {
    pub(super) trace: &'a TraceDisplay,
}

impl Display for DerivationDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (arena, roots) = build(self.trace);
        let mut out = String::new();
        let mut numbers = vec![None; arena.len()];
        let mut next = 1usize;
        for root in roots {
            let mut queue = VecDeque::new();
            if !out.is_empty() {
                out.push('\n');
            }
            arena[root].render(&mut out, None, &mut numbers, &mut next, &mut queue);
            while let Some(id) = queue.pop_front() {
                out.push('\n');
                let number = numbers[id];
                arena[id].render(&mut out, number, &mut numbers, &mut next, &mut queue);
            }
        }
        f.write_str(out.trim_end_matches('\n'))
    }
}

/// The array tree as displayed: one summary per node, children in slot order.
struct Tree {
    array: ArrayRef,
    /// Children in slot order, labelled by slot name.
    children: Vec<(String, Tree)>,
}

impl Tree {
    fn of(array: &ArrayRef) -> Self {
        Self {
            array: array.clone(),
            children: array
                .slots()
                .iter()
                .enumerate()
                .filter_map(|(idx, slot)| {
                    slot.as_ref()
                        .map(|child| (array.slot_name(idx), Tree::of(child)))
                })
                .collect(),
        }
    }

    /// Whether the trees below the two roots print identically.
    fn same_below(&self, other: &Tree) -> bool {
        self.children.len() == other.children.len()
            && self.children.iter().zip(&other.children).all(|(a, b)| {
                a.0 == b.0 && a.1.array.to_string() == b.1.array.to_string() && a.1.same_below(&b.1)
            })
    }

    fn render(&self, out: &mut String, prefix: &str) {
        out.push_str(&self.array.to_string());
        out.push('\n');
        let last = self.children.len().saturating_sub(1);
        for (pos, (name, child)) in self.children.iter().enumerate() {
            let (branch, extend) = if pos == last {
                ("└─", "   ")
            } else {
                ("├─", "│  ")
            };
            out.push_str(&format!("{prefix}{branch}{name}: "));
            child.render(out, &format!("{prefix}{extend}"));
        }
    }
}

enum Rhs {
    /// Only the array itself changed.
    Summary(String),
    /// The tree below the array changed too.
    Tree(Tree),
}

struct Step {
    rule: String,
    /// Derivations this step depends on, by arena index.
    refs: Vec<usize>,
    /// Declined attempts and other events that happened before the step.
    notes: Vec<String>,
    lhs: String,
    rhs: Rhs,
}

/// One reduction sequence on one array.
struct Derivation {
    label: String,
    start: Tree,
    steps: Vec<Step>,
    /// Notes and derivations recorded after the last step.
    trailing_notes: Vec<String>,
    trailing_refs: Vec<usize>,
}

impl Derivation {
    fn render(
        &self,
        out: &mut String,
        number: Option<usize>,
        numbers: &mut [Option<usize>],
        next: &mut usize,
        queue: &mut VecDeque<usize>,
    ) {
        let mut cite = |id: usize| -> String {
            let number = *numbers[id].get_or_insert_with(|| {
                queue.push_back(id);
                let number = *next;
                *next += 1;
                number
            });
            format!("D{number}")
        };

        match number {
            Some(number) => out.push_str(&format!("D{number}: {}\n", self.label)),
            None => out.push_str(&format!("[{}]\n", self.label)),
        }
        self.start.render(out, "");

        if !self.steps.is_empty() {
            out.push('\n');
        }
        for (idx, step) in self.steps.iter().enumerate() {
            out.push_str(&format!("{}. {}", idx + 1, step.rule));
            if !step.refs.is_empty() {
                let refs: Vec<String> = step.refs.iter().map(|id| cite(*id)).collect();
                out.push_str(&format!(", by {}", refs.join(", ")));
            }
            out.push('\n');
            for note in &step.notes {
                out.push_str(&format!("   {note}\n"));
            }
            out.push_str(&format!("   {}\n", step.lhs));
            match &step.rhs {
                Rhs::Summary(rhs) => out.push_str(&format!("   -> {rhs}\n")),
                Rhs::Tree(tree) => {
                    out.push_str("   -> ");
                    tree.render(out, "      ");
                }
            }
        }
        for note in &self.trailing_notes {
            out.push_str(&format!("   {note}\n"));
        }
        if !self.trailing_refs.is_empty() {
            let refs: Vec<String> = self.trailing_refs.iter().map(|id| cite(*id)).collect();
            out.push_str(&format!("   then {}\n", refs.join(", ")));
        }
    }
}

enum FrameKind {
    Root,
    /// An `execute_until`, `optimize`, or single-step pass.
    Pass {
        single_step: bool,
    },
    /// A child slot the parent asked for with `ExecuteSlot`.
    Slot {
        parent: ArrayRef,
        name: String,
    },
    /// A builder driven by `AppendChild`; closed by `builder finish`.
    Builder,
}

struct Frame {
    kind: FrameKind,
    /// The derivation this frame records into; `None` for the root and builder frames.
    derivation: Option<usize>,
    /// The array the next step in this frame rewrites.
    current: Option<ArrayRef>,
    pending_notes: Vec<String>,
    pending_refs: Vec<usize>,
}

impl Frame {
    fn new(kind: FrameKind, derivation: Option<usize>, current: Option<&ArrayRef>) -> Self {
        Self {
            kind,
            derivation,
            current: current.cloned(),
            pending_notes: Vec::new(),
            pending_refs: Vec::new(),
        }
    }
}

/// The name of `parent`'s slot `idx`, e.g. `codes` or `values`.
fn slot(parent: &ArrayRef, idx: usize) -> String {
    parent.slot_name(idx)
}

fn build(trace: &TraceDisplay) -> (Vec<Derivation>, Vec<usize>) {
    let hidden = trace.hidden_events();
    let mut builder = Builder {
        frames: vec![Frame::new(FrameKind::Root, None, None)],
        arena: Vec::new(),
        roots: Vec::new(),
    };
    for (idx, event) in trace.events.iter().enumerate() {
        if !hidden[idx] {
            builder.event(event);
        }
    }
    // Frames left open by an error path, or by a single step that recursed into a child, are
    // closed with what was recorded.
    while builder.frames.len() > 1 {
        builder.close_top();
    }
    (builder.arena, builder.roots)
}

struct Builder {
    frames: Vec<Frame>,
    arena: Vec<Derivation>,
    roots: Vec<usize>,
}

impl Builder {
    fn top(&mut self) -> &mut Frame {
        self.frames
            .last_mut()
            .vortex_expect("root frame is never popped")
    }

    /// The innermost frame that records into a derivation.
    fn recording(&mut self) -> Option<&mut Frame> {
        self.frames
            .iter_mut()
            .rev()
            .find(|frame| frame.derivation.is_some())
    }

    fn set_current(&mut self, array: &ArrayRef) {
        if let Some(frame) = self.recording() {
            frame.current = Some(array.clone());
        }
    }

    fn note(&mut self, text: String) {
        if let Some(frame) = self.recording() {
            frame.pending_notes.push(text);
        }
    }

    fn open(&mut self, kind: FrameKind, label: String, start: &ArrayRef) {
        let id = self.arena.len();
        self.arena.push(Derivation {
            label,
            start: Tree::of(start),
            steps: Vec::new(),
            trailing_notes: Vec::new(),
            trailing_refs: Vec::new(),
        });
        self.frames.push(Frame::new(kind, Some(id), Some(start)));
    }

    /// Pops the top frame and finishes its derivation. Returns the derivation's id when it
    /// recorded anything.
    fn close(&mut self) -> Option<(usize, Frame)> {
        if self.frames.len() <= 1 {
            return None;
        }
        let mut frame = self.frames.pop()?;
        let id = frame.derivation?;
        let derivation = &mut self.arena[id];
        derivation.trailing_notes = std::mem::take(&mut frame.pending_notes);
        derivation.trailing_refs = std::mem::take(&mut frame.pending_refs);
        let empty = derivation.steps.is_empty()
            && derivation.trailing_notes.is_empty()
            && derivation.trailing_refs.is_empty();
        (!empty).then_some((id, frame))
    }

    /// Records the step `current -> output` justified by `rule` in the innermost derivation,
    /// citing `extra` derivations after any pending ones.
    fn step(&mut self, rule: String, output: &ArrayRef, extra: Vec<usize>) {
        let Some(frame) = self.recording() else {
            return;
        };
        let lhs = frame.current.replace(output.clone());
        let mut refs = std::mem::take(&mut frame.pending_refs);
        refs.extend(extra);
        let notes = std::mem::take(&mut frame.pending_notes);
        let id = frame
            .derivation
            .vortex_expect("recording frame has a derivation");
        let after = Tree::of(output);
        let rhs = match &lhs {
            Some(lhs) if Tree::of(lhs).same_below(&after) => Rhs::Summary(output.to_string()),
            _ if after.children.is_empty() => Rhs::Summary(output.to_string()),
            _ => Rhs::Tree(after),
        };
        self.arena[id].steps.push(Step {
            rule,
            refs,
            notes,
            lhs: lhs.map_or_else(|| "?".to_string(), |lhs| lhs.to_string()),
            rhs,
        });
    }

    fn close_pass(&mut self) {
        let Some((id, _)) = self.close() else { return };
        match self.recording() {
            Some(frame) => frame.pending_refs.push(id),
            None => self.roots.push(id),
        }
    }

    fn close_if_single_step(&mut self) {
        if matches!(self.top().kind, FrameKind::Pass { single_step: true }) {
            self.close_pass();
        }
    }

    /// Leaves a slot the parent asked for. With `output` (from `pop_frame`) the parent takes the
    /// reduced child back, which is a congruence step on the parent citing the child's
    /// derivation. Without, a stack kernel consumed the parent and returns the parent array the
    /// next step rewrites, plus the child's derivation for that step to cite.
    fn close_slot(&mut self, output: Option<&ArrayRef>) -> (Option<ArrayRef>, Vec<usize>) {
        if !matches!(
            self.frames.last().map(|frame| &frame.kind),
            Some(FrameKind::Slot { .. })
        ) {
            return (None, Vec::new());
        }
        let Some(mut popped) = self.frames.pop() else {
            return (None, Vec::new());
        };
        let (parent, name) = match &popped.kind {
            FrameKind::Slot { parent, name } => (parent.clone(), name.clone()),
            _ => return (None, Vec::new()),
        };
        let id = popped
            .derivation
            .vortex_expect("slot frame has a derivation");
        let derivation = &mut self.arena[id];
        derivation.trailing_notes = std::mem::take(&mut popped.pending_notes);
        derivation.trailing_refs = std::mem::take(&mut popped.pending_refs);
        let recorded = !(derivation.steps.is_empty()
            && derivation.trailing_notes.is_empty()
            && derivation.trailing_refs.is_empty());
        let refs: Vec<usize> = recorded.then_some(id).into_iter().collect();

        match output {
            Some(output) => {
                self.set_current(&parent);
                self.step(format!("{name} reduced"), output, refs);
                (None, Vec::new())
            }
            None => (Some(parent), refs),
        }
    }

    fn close_top(&mut self) {
        match self.frames.last().map(|frame| &frame.kind) {
            Some(FrameKind::Pass { .. }) => self.close_pass(),
            Some(FrameKind::Slot { .. }) => {
                drop(self.close_slot(None));
            }
            Some(FrameKind::Builder) => {
                self.frames.pop();
                self.note("builder: unfinished".to_string());
            }
            Some(FrameKind::Root) | None => {}
        }
    }

    fn event(&mut self, event: &TraceEvent) {
        match event {
            TraceEvent::OptimizeStart { root, session } => {
                let session = if *session { " session" } else { "" };
                self.open(
                    FrameKind::Pass { single_step: false },
                    format!("optimize{session}"),
                    &root.0,
                );
            }
            TraceEvent::OptimizeLoopStart { array } => self.set_current(&array.0),
            TraceEvent::OptimizeLoopEnd
            | TraceEvent::ExecuteUntilDoneCheck { .. }
            | TraceEvent::PhaseNone { .. }
            | TraceEvent::ExecuteEncoding { .. } => {}
            TraceEvent::OptimizeDone { .. } | TraceEvent::ExecuteUntilReturn { .. } => {
                if matches!(self.top().kind, FrameKind::Pass { .. }) {
                    self.close_pass();
                }
            }
            TraceEvent::OptimizeRecursiveStart { root } => self.set_current(&root.0),
            TraceEvent::OptimizeRecursiveSlot {
                slot_idx,
                input,
                output,
            } => {
                let name = self
                    .recording()
                    .and_then(|frame| frame.current.as_ref())
                    .map_or_else(|| slot_idx.to_string(), |parent| slot(parent, *slot_idx));
                self.note(format!("optimize_recursive {name}: {input} -> {output}"));
            }
            TraceEvent::ReduceAttempt { rule, outcome, .. } => {
                self.note(format!("x reduce {rule}: {outcome}"));
            }
            TraceEvent::ReduceApplied { rule, output, .. } => {
                self.step(format!("reduce {rule}"), &output.0, Vec::new());
            }
            TraceEvent::ParentReduceAttempt {
                parent,
                slot_idx,
                source,
                rule,
                outcome,
                ..
            } => self.note(format!(
                "x reduce_parent {source}:{rule} from {}: {outcome}",
                slot(&parent.0, *slot_idx)
            )),
            TraceEvent::ParentReduceApplied {
                parent,
                slot_idx,
                source,
                rule,
                output,
                ..
            } => self.step(
                format!(
                    "reduce_parent {source}:{rule} from {}",
                    slot(&parent.0, *slot_idx)
                ),
                &output.0,
                Vec::new(),
            ),
            TraceEvent::ExecuteUntilStart { target, root } => self.open(
                FrameKind::Pass { single_step: false },
                format!("execute_until {target}"),
                &root.0,
            ),
            TraceEvent::ExecuteUntilIteration { current, .. } => self.set_current(&current.0),
            TraceEvent::ExecuteUntilPopFrame { output, .. } => {
                drop(self.close_slot(Some(&output.0)));
            }
            TraceEvent::ExecuteParentAttempt {
                phase,
                parent,
                slot_idx,
                source,
                kernel,
                outcome,
                ..
            } => self.note(format!(
                "x {phase} {source}:{kernel} from {}: {outcome}",
                slot(&parent.0, *slot_idx)
            )),
            TraceEvent::ExecuteParentApplied {
                phase,
                parent,
                slot_idx,
                source,
                kernel,
                output,
                ..
            } => {
                let mut refs = Vec::new();
                if *phase == "stack_execute_parent"
                    && matches!(self.top().kind, FrameKind::Slot { .. })
                {
                    // The kernel rewrote the suspended parent, consuming the slot it asked for.
                    let (parent, slot_refs) = self.close_slot(None);
                    if let Some(parent) = parent {
                        self.set_current(&parent);
                    }
                    refs = slot_refs;
                }
                self.step(
                    format!(
                        "{phase} {source}:{kernel} from {}",
                        slot(&parent.0, *slot_idx)
                    ),
                    &output.0,
                    refs,
                );
            }
            TraceEvent::ExecuteOptimized {
                output, changed, ..
            } => {
                if *changed {
                    self.step("optimize_ctx".to_string(), &output.0, Vec::new());
                } else {
                    // An optimize pass that changed nothing justifies nothing.
                    let arena = &self.arena;
                    if let Some(frame) = self
                        .frames
                        .iter_mut()
                        .rev()
                        .find(|frame| frame.derivation.is_some())
                    {
                        frame.pending_refs.retain(|id| !arena[*id].steps.is_empty());
                    }
                }
            }
            TraceEvent::SlotTransition {
                step,
                slot_idx,
                parent,
                child,
            } => match *step {
                "ExecuteSlot" => {
                    let name = slot(&parent.0, *slot_idx);
                    self.open(
                        FrameKind::Slot {
                            parent: parent.0.clone(),
                            name: name.clone(),
                        },
                        format!("{name} of {parent}"),
                        &child.0,
                    );
                }
                _ => self.note(format!("append {}: {child}", slot(&parent.0, *slot_idx))),
            },
            TraceEvent::BuilderEvent { action, array, .. } => match *action {
                "start" => self.frames.push(Frame::new(FrameKind::Builder, None, None)),
                "finish" => {
                    if matches!(self.top().kind, FrameKind::Builder) {
                        self.frames.pop();
                    }
                    self.step("builder".to_string(), &array.0, Vec::new());
                    self.close_if_single_step();
                }
                _ => {}
            },
            TraceEvent::ExecuteDone { array } => {
                if matches!(self.top().kind, FrameKind::Builder) {
                    // `builder finish` follows and carries the real output.
                    return;
                }
                let encoding = self
                    .recording()
                    .and_then(|frame| frame.current.as_ref())
                    .map_or_else(|| "?".to_string(), |a| a.encoding_id().to_string());
                self.step(format!("execute {encoding}"), &array.0, Vec::new());
                self.close_if_single_step();
            }
            TraceEvent::SingleStepStart { array } => self.open(
                FrameKind::Pass { single_step: true },
                "execute_step".to_string(),
                &array.0,
            ),
            TraceEvent::SingleStepApplied { phase, output, .. } => {
                self.step((*phase).to_string(), &output.0, Vec::new());
                self.close_if_single_step();
            }
        }
    }
}
