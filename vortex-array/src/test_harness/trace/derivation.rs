// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Renders a [`TraceDisplay`] as a derivation of the small-step reduction semantics that the
//! optimizer and executor implement.
//!
//! Every rule, kernel, and encoding execution is a single step `a -> a'`. `ExecuteSlot` focuses
//! a child slot as an evaluation context: steps on the child are the premises of one congruence
//! step `P -> P[i <- c']` on the parent. Nested `execute_until` and `optimize` passes are
//! likewise premises of the step that invoked them.
//!
//! Every line is a complete judgement, `lhs -> rhs  [rule]` for a step or `lhs ->* rhs  [pass]`
//! for a whole pass, and the tree connectors attach each step's premises beneath it:
//!
//! ```text
//! vortex.filter(i32, len=2) ->* vortex.primitive(i32, len=2)  [execute_until AnyCanonical]
//! ├─ vortex.filter(i32, len=2) -> vortex.slice(i32, len=2)  [execute vortex.filter]
//! ├─ vortex.slice(i32, len=2) -> vortex.slice(i32, len=2)[0 <- vortex.primitive(i32, len=4)]  [slot 0]
//! │  └─ vortex.filter(i32, len=4) -> vortex.primitive(i32, len=4)  [execute vortex.filter]
//! └─ vortex.slice(i32, len=2) -> vortex.primitive(i32, len=2)  [execute vortex.slice]
//!    └─ vortex.slice(i32, len=2) ->* vortex.primitive(i32, len=2)  [optimize]
//!       └─ vortex.slice(i32, len=2) -> vortex.primitive(i32, len=2)  [reduce_parent ...]
//! ```
//!
//! Consecutive steps under one pass chain: each step's `lhs` is the previous step's `rhs`. A
//! step whose summary is unchanged names the child slot it replaced, as `a[i <- new_child]`.
//! At [`TraceResolution::Attempts`](super::TraceResolution::Attempts), rules and kernels that
//! were tried and declined before a step appear as `x ...` leaves under that step.

use std::fmt;
use std::fmt::Display;

use vortex_error::VortexExpect;

use super::ArraySummary;
use super::TraceDisplay;
use super::TraceEvent;

/// A derivation-tree rendering of a [`TraceDisplay`]; see the module docs.
pub struct DerivationDisplay<'a> {
    pub(super) trace: &'a TraceDisplay,
}

impl Display for DerivationDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (idx, node) in build(self.trace).iter().enumerate() {
            if idx > 0 {
                writeln!(f)?;
            }
            render(f, node, "")?;
        }
        Ok(())
    }
}

/// A judgement (a step or a whole pass) with the premises that justify it, or a bare note.
struct Node {
    text: String,
    children: Vec<Node>,
}

impl Node {
    fn leaf(text: String) -> Self {
        Self {
            text,
            children: Vec::new(),
        }
    }
}

fn render(f: &mut fmt::Formatter<'_>, node: &Node, prefix: &str) -> fmt::Result {
    f.write_str(&node.text)?;
    let last = node.children.len().saturating_sub(1);
    for (idx, child) in node.children.iter().enumerate() {
        let (branch, extend) = if idx == last {
            ("└─ ", "   ")
        } else {
            ("├─ ", "│  ")
        };
        write!(f, "\n{prefix}{branch}")?;
        render(f, child, &format!("{prefix}{extend}"))?;
    }
    Ok(())
}

enum FrameKind {
    Root,
    /// An `execute_until`, `optimize`, or single-step pass over `lhs`.
    Chain {
        lhs: ArraySummary,
        label: String,
        single_step: bool,
    },
    /// A child slot focused by `ExecuteSlot`; closed by `pop_frame` or a stack kernel.
    Slot {
        parent: ArraySummary,
        slot_idx: usize,
    },
    /// A builder driven by `AppendChild`; closed by `builder finish`.
    Builder,
}

struct Frame {
    kind: FrameKind,
    /// Steps taken so far in this frame.
    nodes: Vec<Node>,
    /// Nested blocks and declined attempts waiting to become premises of the next step.
    pending: Vec<Node>,
    /// The array the next step in this frame rewrites.
    current: Option<ArraySummary>,
    /// How many steps this frame has recorded.
    steps: usize,
}

impl Frame {
    fn new(kind: FrameKind, current: Option<&ArraySummary>) -> Self {
        Self {
            kind,
            nodes: Vec::new(),
            pending: Vec::new(),
            current: current.cloned(),
            steps: 0,
        }
    }
}

/// Describes `rhs` as the result of rewriting `lhs`.
///
/// Array summaries hide children, so a rewrite that only replaced a child slot would print as
/// `a -> a`. When the encoding and summary are unchanged, the differing slots are listed
/// instead, in evaluation-context notation: `a[i <- new_child]`.
fn describe_rewrite(lhs: Option<&ArraySummary>, rhs: &ArraySummary) -> String {
    let Some(lhs) = lhs else {
        return rhs.to_string();
    };
    let (lhs, rhs_array) = (&lhs.0, &rhs.0);
    if lhs.encoding_id() != rhs_array.encoding_id()
        || lhs.slots().len() != rhs_array.slots().len()
        || lhs.to_string() != rhs_array.to_string()
    {
        return rhs.to_string();
    }
    let changed: Vec<String> = lhs
        .slots()
        .iter()
        .zip(rhs_array.slots())
        .enumerate()
        .filter_map(|(idx, (before, after))| {
            let after = after.as_ref()?;
            let before = before.as_ref().map(ToString::to_string);
            (before.as_deref() != Some(&after.to_string())).then(|| format!("{idx} <- {after}"))
        })
        .collect();
    if changed.is_empty() {
        rhs.to_string()
    } else {
        format!("{rhs}[{}]", changed.join(", "))
    }
}

struct Builder {
    frames: Vec<Frame>,
}

fn build(trace: &TraceDisplay) -> Vec<Node> {
    let hidden = trace.hidden_events();
    let mut builder = Builder {
        frames: vec![Frame::new(FrameKind::Root, None)],
    };
    for (idx, event) in trace.events.iter().enumerate() {
        if !hidden[idx] {
            builder.event(event);
        }
    }
    // Blocks left open by an error path are closed with what was recorded.
    while builder.frames.len() > 1 {
        builder.close_top();
    }
    let mut root = builder.frames.pop().vortex_expect("root frame");
    root.nodes.append(&mut root.pending);
    root.nodes
}

impl Builder {
    fn top(&mut self) -> &mut Frame {
        self.frames
            .last_mut()
            .vortex_expect("root frame is never popped")
    }

    fn push(&mut self, kind: FrameKind, current: &ArraySummary) {
        self.frames.push(Frame::new(kind, Some(current)));
    }

    fn pop(&mut self) -> Option<Frame> {
        (self.frames.len() > 1).then(|| self.frames.pop()).flatten()
    }

    fn current_encoding(&mut self) -> String {
        self.top()
            .current
            .as_ref()
            .map_or_else(|| "?".to_string(), |a| a.0.encoding_id().to_string())
    }

    /// Records a declined attempt or note that will justify the next step.
    fn note(&mut self, text: String) {
        self.top().pending.push(Node::leaf(text));
    }

    /// Records the step `current -> output` justified by `rule`, taking pending nodes as its
    /// leading premises followed by `extra`.
    fn step(&mut self, rule: String, output: &ArraySummary, mut extra: Vec<Node>) {
        let frame = self.top();
        let lhs = frame
            .current
            .as_ref()
            .map_or_else(|| "?".to_string(), ToString::to_string);
        let rhs = describe_rewrite(frame.current.as_ref(), output);
        let mut children = std::mem::take(&mut frame.pending);
        children.append(&mut extra);
        frame.nodes.push(Node {
            text: format!("{lhs} -> {rhs}  [{rule}]"),
            children,
        });
        frame.current = Some(output.clone());
        frame.steps += 1;
    }

    fn close_top(&mut self) {
        let Some(frame) = self.frames.last() else {
            return;
        };
        match &frame.kind {
            FrameKind::Root => {}
            FrameKind::Chain { .. } => {
                let output = frame.current.clone();
                self.close_chain(output.as_ref());
            }
            FrameKind::Slot { .. } => self.close_slot(None),
            FrameKind::Builder => {
                let output = frame.current.clone();
                self.close_builder(output.as_ref());
            }
        }
    }

    fn close_chain(&mut self, output: Option<&ArraySummary>) {
        let Some(mut frame) = self.pop() else { return };
        let FrameKind::Chain { lhs, label, .. } = frame.kind else {
            return;
        };
        let rhs = output
            .or(frame.current.as_ref())
            .map_or_else(|| "?".to_string(), ToString::to_string);
        frame.nodes.append(&mut frame.pending);
        self.top().pending.push(Node {
            text: format!("{lhs} ->* {rhs}  [{label}]"),
            children: frame.nodes,
        });
    }

    fn close_if_single_step(&mut self) {
        if matches!(
            self.top().kind,
            FrameKind::Chain {
                single_step: true,
                ..
            }
        ) {
            let output = self.top().current.clone();
            self.close_chain(output.as_ref());
        }
    }

    /// Closes a focused slot. With `output` (from `pop_frame`) the parent is rewritten to it by
    /// congruence over the child's steps. Without (a stack kernel consumed the frame), the
    /// child's steps so far still justify a congruence step; if there were none the focus is
    /// only noted.
    fn close_slot(&mut self, output: Option<&ArraySummary>) {
        let Some(mut frame) = self.pop() else { return };
        let FrameKind::Slot { parent, slot_idx } = frame.kind else {
            return;
        };
        let child = frame
            .current
            .as_ref()
            .map_or_else(|| "?".to_string(), ToString::to_string);
        if frame.steps == 0 && output.is_none() {
            self.note(format!("focus slot={slot_idx} {child}"));
            self.top().pending.append(&mut frame.pending);
            return;
        }
        frame.nodes.append(&mut frame.pending);
        let rule = format!("slot {slot_idx}");
        match output {
            Some(output) => {
                self.top().current = Some(parent);
                self.step(rule, output, frame.nodes);
            }
            None => {
                // The child's steps are known but the rewritten parent was never materialized.
                let top = self.top();
                let mut children = std::mem::take(&mut top.pending);
                children.extend(frame.nodes);
                top.nodes.push(Node {
                    text: format!("{parent} -> {parent}[{slot_idx} <- {child}]  [{rule}]"),
                    children,
                });
                top.current = Some(parent);
                top.steps += 1;
            }
        }
    }

    /// Attaches nested work recorded after an `append` to that append rather than the next one.
    fn attach_pending_to_last(&mut self) {
        let frame = self.top();
        if frame.pending.is_empty() {
            return;
        }
        match frame.nodes.last_mut() {
            Some(node) => node.children.append(&mut frame.pending),
            None => frame.nodes.append(&mut frame.pending),
        }
    }

    fn close_builder(&mut self, output: Option<&ArraySummary>) {
        self.attach_pending_to_last();
        let Some(frame) = self.pop() else { return };
        match output {
            Some(output) => self.step("builder".to_string(), output, frame.nodes),
            None => {
                let top = self.top();
                let mut children = std::mem::take(&mut top.pending);
                children.extend(frame.nodes);
                top.nodes.push(Node {
                    text: "? -> ?  [builder]".to_string(),
                    children,
                });
            }
        }
        self.close_if_single_step();
    }

    fn event(&mut self, event: &TraceEvent) {
        match event {
            TraceEvent::OptimizeStart { root, session } => {
                let session = if *session { " session" } else { "" };
                self.push(
                    FrameKind::Chain {
                        lhs: root.clone(),
                        label: format!("optimize{session}"),
                        single_step: false,
                    },
                    root,
                );
            }
            TraceEvent::OptimizeLoopStart { array } => self.top().current = Some(array.clone()),
            TraceEvent::OptimizeLoopEnd
            | TraceEvent::ExecuteUntilDoneCheck { .. }
            | TraceEvent::PhaseNone { .. }
            | TraceEvent::ExecuteEncoding { .. } => {}
            TraceEvent::OptimizeDone { output, .. } => {
                if matches!(self.top().kind, FrameKind::Chain { .. }) {
                    self.close_chain(Some(output));
                }
            }
            TraceEvent::OptimizeRecursiveStart { root } => {
                self.top().current = Some(root.clone());
            }
            TraceEvent::OptimizeRecursiveSlot {
                slot_idx, output, ..
            } => {
                // There is no event for the rewritten parent; describe the slot directly.
                let parent = self.top().current.clone();
                let parent_text = parent
                    .as_ref()
                    .map_or_else(|| "?".to_string(), ToString::to_string);
                let text = format!(
                    "{parent_text} -> {parent_text}[{slot_idx} <- {output}]  [slot {slot_idx}]"
                );
                let top = self.top();
                let children = std::mem::take(&mut top.pending);
                top.nodes.push(Node { text, children });
                top.steps += 1;
            }
            TraceEvent::ReduceAttempt { rule, outcome, .. } => {
                self.note(format!("x reduce {rule}: {outcome}"));
            }
            TraceEvent::ReduceApplied { rule, output, .. } => {
                self.step(format!("reduce {rule}"), output, Vec::new());
            }
            TraceEvent::ParentReduceAttempt {
                child,
                slot_idx,
                source,
                rule,
                outcome,
                ..
            } => self.note(format!(
                "x reduce_parent {source}:{rule} slot={slot_idx} child={child}: {outcome}"
            )),
            TraceEvent::ParentReduceApplied {
                child,
                slot_idx,
                source,
                rule,
                output,
                ..
            } => self.step(
                format!("reduce_parent {source}:{rule} slot={slot_idx} child={child}"),
                output,
                Vec::new(),
            ),
            TraceEvent::ExecuteUntilStart { target, root } => self.push(
                FrameKind::Chain {
                    lhs: root.clone(),
                    label: format!("execute_until {target}"),
                    single_step: false,
                },
                root,
            ),
            TraceEvent::ExecuteUntilIteration { current, .. } => {
                self.top().current = Some(current.clone());
            }
            TraceEvent::ExecuteUntilReturn { output } => {
                if matches!(self.top().kind, FrameKind::Chain { .. }) {
                    self.close_chain(Some(output));
                }
            }
            TraceEvent::ExecuteUntilPopFrame { output, .. } => self.close_slot(Some(output)),
            TraceEvent::ExecuteParentAttempt {
                phase,
                child,
                slot_idx,
                source,
                kernel,
                outcome,
                ..
            } => self.note(format!(
                "x {phase} {source}:{kernel} slot={slot_idx} child={child}: {outcome}"
            )),
            TraceEvent::ExecuteParentApplied {
                phase,
                child,
                slot_idx,
                source,
                kernel,
                output,
                ..
            } => {
                if *phase == "stack_execute_parent"
                    && matches!(self.top().kind, FrameKind::Slot { .. })
                {
                    self.close_slot(None);
                }
                self.step(
                    format!("{phase} {source}:{kernel} slot={slot_idx} child={child}"),
                    output,
                    Vec::new(),
                );
            }
            TraceEvent::ExecuteOptimized {
                output, changed, ..
            } => {
                if *changed {
                    self.step("optimize_ctx".to_string(), output, Vec::new());
                } else {
                    // An optimize pass that changed nothing justifies nothing; keep only
                    // premises that carry information of their own.
                    let frame = self.top();
                    let pending = std::mem::take(&mut frame.pending);
                    frame.nodes.extend(
                        pending.into_iter().filter(|node| {
                            !(node.text.contains(" ->* ") && node.children.is_empty())
                        }),
                    );
                }
            }
            TraceEvent::SlotTransition {
                step,
                slot_idx,
                parent,
                child,
            } => match *step {
                "ExecuteSlot" => {
                    self.push(
                        FrameKind::Slot {
                            parent: parent.clone(),
                            slot_idx: *slot_idx,
                        },
                        child,
                    );
                }
                _ => {
                    self.attach_pending_to_last();
                    self.top()
                        .nodes
                        .push(Node::leaf(format!("append slot={slot_idx} {child}")));
                }
            },
            TraceEvent::BuilderEvent { action, array, .. } => match *action {
                "start" => self.push(FrameKind::Builder, array),
                "finish" => self.close_builder(Some(array)),
                _ => {}
            },
            TraceEvent::ExecuteDone { array } => {
                if matches!(self.top().kind, FrameKind::Builder) {
                    // `builder finish` follows and carries the real output.
                    return;
                }
                let encoding = self.current_encoding();
                self.step(format!("execute {encoding}"), array, Vec::new());
                self.close_if_single_step();
            }
            TraceEvent::SingleStepStart { array } => self.push(
                FrameKind::Chain {
                    lhs: array.clone(),
                    label: "execute_step".to_string(),
                    single_step: true,
                },
                array,
            ),
            TraceEvent::SingleStepApplied { phase, output, .. } => {
                self.step(phase.to_string(), output, Vec::new());
                self.close_if_single_step();
            }
        }
    }
}
