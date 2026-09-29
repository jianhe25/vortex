// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Renders a [`TraceDisplay`] as a derivation of the small-step reduction semantics that the
//! optimizer and executor implement.
//!
//! Every rule, kernel, and encoding execution is a single step `a -> a'`. `ExecuteSlot` focuses
//! a child slot as an evaluation context: steps on the child are the premises of one congruence
//! step `P[i <- c] -> P[i <- c']` on the parent. Nested `execute_until` and `optimize` passes are
//! likewise premises of the step that invoked them.
//!
//! A block is rendered as a chain: the starting array on its own line, then one `-> rhs  [rule]`
//! line per step, with each step's premises indented beneath it:
//!
//! ```text
//! vortex.filter(i32, len=2)  [execute_until AnyCanonical]
//! -> vortex.slice(i32, len=2)  [execute vortex.filter]
//! -> vortex.slice(i32, len=2)[0 <- vortex.primitive(i32, len=4)]  [slot 0]
//!     vortex.filter(i32, len=4)
//!     -> vortex.primitive(i32, len=4)  [execute vortex.filter]
//! -> vortex.primitive(i32, len=2)  [execute vortex.slice]
//!     vortex.slice(i32, len=2)  [optimize]
//!     -> vortex.primitive(i32, len=2)  [reduce_parent static:SliceReduceAdaptor(Primitive) ...]
//! ```
//!
//! At [`TraceResolution::Attempts`](super::TraceResolution::Attempts), rules and kernels that
//! were tried and declined before a step appear as `x ...` premises of that step.

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
        let mut first = true;
        for node in build(self.trace) {
            render(f, &node, 0, &mut first)?;
        }
        Ok(())
    }
}

enum Node {
    /// A reduction chain: a starting array followed by its steps.
    Chain { header: String, nodes: Vec<Node> },
    /// A single line, usually a step, with the premises that justify it.
    Item { text: String, premises: Vec<Node> },
}

fn render(f: &mut fmt::Formatter<'_>, node: &Node, depth: usize, first: &mut bool) -> fmt::Result {
    let line = |f: &mut fmt::Formatter<'_>, text: &str, first: &mut bool| -> fmt::Result {
        if *first {
            *first = false;
        } else {
            writeln!(f)?;
        }
        for _ in 0..depth {
            f.write_str("    ")?;
        }
        f.write_str(text)
    };
    match node {
        Node::Chain { header, nodes } => {
            line(f, header, first)?;
            for node in nodes {
                render(f, node, depth, first)?;
            }
        }
        Node::Item { text, premises } => {
            line(f, text, first)?;
            for premise in premises {
                render(f, premise, depth + 1, first)?;
            }
        }
    }
    Ok(())
}

enum FrameKind {
    Root,
    /// An `execute_until`, `optimize`, or single-step block.
    Chain {
        header: String,
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
        self.top().pending.push(Node::Item {
            text,
            premises: Vec::new(),
        });
    }

    /// Records the step `current -> output` justified by `rule`, taking pending nodes as its
    /// leading premises followed by `extra`.
    fn step(&mut self, rule: String, output: &ArraySummary, mut extra: Vec<Node>) {
        let frame = self.top();
        let rhs = describe_rewrite(frame.current.as_ref(), output);
        let mut premises = std::mem::take(&mut frame.pending);
        premises.append(&mut extra);
        frame.nodes.push(Node::Item {
            text: format!("-> {rhs}  [{rule}]"),
            premises,
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
            FrameKind::Chain { .. } => self.close_chain(),
            FrameKind::Slot { .. } => self.close_slot(None),
            FrameKind::Builder => {
                let output = frame.current.clone();
                self.close_builder(output.as_ref());
            }
        }
    }

    fn close_chain(&mut self) {
        let Some(mut frame) = self.pop() else { return };
        let FrameKind::Chain { header, .. } = frame.kind else {
            return;
        };
        frame.nodes.append(&mut frame.pending);
        self.top().pending.push(Node::Chain {
            header,
            nodes: frame.nodes,
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
            self.close_chain();
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
                let rhs = format!("{parent}[{slot_idx} <- {child}]");
                let top = self.top();
                let mut premises = std::mem::take(&mut top.pending);
                premises.extend(frame.nodes);
                top.nodes.push(Node::Item {
                    text: format!("-> {rhs}  [{rule}]"),
                    premises,
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
            Some(Node::Item { premises, .. }) => premises.append(&mut frame.pending),
            _ => frame.nodes.append(&mut frame.pending),
        }
    }

    fn close_builder(&mut self, output: Option<&ArraySummary>) {
        self.attach_pending_to_last();
        let Some(frame) = self.pop() else { return };
        match output {
            Some(output) => self.step("builder".to_string(), output, frame.nodes),
            None => {
                let top = self.top();
                let mut premises = std::mem::take(&mut top.pending);
                premises.extend(frame.nodes);
                top.nodes.push(Node::Item {
                    text: "-> ?  [builder]".to_string(),
                    premises,
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
                        header: format!("{root}  [optimize{session}]"),
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
            TraceEvent::OptimizeDone { .. } => {
                if matches!(self.top().kind, FrameKind::Chain { .. }) {
                    self.close_chain();
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
                let text = format!("-> {parent_text}[{slot_idx} <- {output}]  [slot {slot_idx}]");
                let top = self.top();
                let premises = std::mem::take(&mut top.pending);
                top.nodes.push(Node::Item { text, premises });
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
                    header: format!("{root}  [execute_until {target}]"),
                    single_step: false,
                },
                root,
            ),
            TraceEvent::ExecuteUntilIteration { current, .. } => {
                self.top().current = Some(current.clone());
            }
            TraceEvent::ExecuteUntilReturn { .. } => {
                if matches!(self.top().kind, FrameKind::Chain { .. }) {
                    self.close_chain();
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
                    frame.nodes.extend(pending.into_iter().filter(
                        |node| !matches!(node, Node::Chain { nodes, .. } if nodes.is_empty()),
                    ));
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
                    self.top().nodes.push(Node::Item {
                        text: child.to_string(),
                        premises: Vec::new(),
                    });
                }
                _ => {
                    self.attach_pending_to_last();
                    self.top().nodes.push(Node::Item {
                        text: format!("append slot={slot_idx} {child}"),
                        premises: Vec::new(),
                    });
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
                    header: format!("{array}  [execute_step]"),
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
