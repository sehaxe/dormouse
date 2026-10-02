//! The in-place optimizer step and the device-fed scalars: what makes a
//! capture window that contains `opt + retr + ema` ADVANCE instead of
//! re-reading its capture values.
//!
//! Two blockers stand between the capture window and the optimizer's tail,
//! and only one of them is the named blocker (`8dd8398`):
//!
//! 1. **Addresses** (the named blocker). burn's functional optimizers
//!    allocate every tensor they touch, each step
//!    (`burn-optim-0.22.0-pre.4/src/optim/adam.rs:88`,
//!    `(tensor - delta, Some(state))`). A replay runs no host code, so a
//!    graph captured at step *k* would forever compute the update out of the
//!    buffers the tensors lived in at capture time — read two-step-old
//!    weights, write where the parameter no longer lives, and error nowhere
//!    (the `8fa5d4c` shape reproduced in `graph_step.rs` on a toy).
//! 2. **Kernel args are baked** (found and settled here, cheaply). Every
//!    per-step host value an optimizer uses reaches its kernel as an
//!    argument: `mul_scalar(lr)` in `adam.rs:88`, and Adam's bias
//!    correction `combined_factor` is computed from a HOST usize counter in
//!    `burn-optim/src/optim/momentum.rs:176` — a capture bakes those values
//!    into the recorded launches, so a replay would apply the capture step's
//!    LR and bias correction on every replay, forever. Not the pin's fault;
//!    nothing about in-place writes would have fixed it.
//!
//! Both share one escape, and it is small: **everything that crosses a
//! replay boundary lives in a pinned buffer whose content is read
//! dynamically at every kernel run.** Per-step host scalars (LR, the bias
//! factor) are copied into pinned device tensors by the host between
//! windows. Step temporaries — the Newton-Schulz chain, the moments'
//! intermediates — do NOT cross the boundary (each replay rewrites each one
//! before anything reads it), so they stay out of place and burn's math is
//! used verbatim everywhere else.
//!
//! # The warmup is not a step
//!
//! `graph_prepare`'s priming pass must run the whole window to grow the
//! retained pool, but the tail that enters the window mutates state — an
//! un-snapshotted priming pass is an extra optimizer step and the weight
//! trajectory is off by two after every capture. So the seam snapshots every
//! pinned buffer before priming and restores them before `start_capture`
//! (both plain copies, outside the window). The Adam `time` counters live on
//! the host, advanced once per step by the feed path, so capture and replay
//! advance the state exactly once per step — the same accounting the
//! control's walk has.
//!
//! # Scope, named (ADR-0011)
//!
//! The composite covers the live `mix` recipe exactly: AdamW fallback (the
//! "rest" group), plain Adam on the n-gram tables, Muon+ ColRow on the small
//! linear maps, and head-wise Muon+ on Q/K. `--opt` other than `mix`,
//! `--retract-every != 1`, `--retract-batched` and `--grad-clip > 0` each
//! change the window's shape (or its numerical path) and are refused
//! loudly in [`super::graph::check`], each naming the arm.

use std::collections::HashMap;

use burn::{
    module::{Module, ModuleMapper, Param, ParamId},
    tensor::{ElementConversion, Tensor},
};

use crate::graph::{as_constant_float, copy_into, Pins};

/// A host-computed, per-step scalar pinned at a device buffer that kernels
/// inside a captured window read fresh on every run.
///
/// Kernel arguments are baked at capture; a device buffer read is not. The
/// buffer is allocated once at construction and never moves: the `Tensor`
/// object lives in the optimizer for its whole lifetime, and nothing re-points
/// it, so a graph can hold its raw address across any number of replays.
pub struct ScalarPin {
    dst: Tensor<2>,
}

impl ScalarPin {
    pub fn new<B: burn::backend::Backend>(device: &burn::tensor::Device<B>) -> Self {
        Self { dst: Tensor::zeros([1, 1], device) }
    }

    /// `dst := value`, one copy launch. Host code, always outside the window.
    pub fn put<B: burn::backend::Backend>(
        &mut self,
        value: f32,
        device: &burn::tensor::Device<B>,
    ) -> Result<(), String> {
        let src: Tensor<2, _> =
            Tensor::from_data(burn::tensor::TensorData::new(vec![value], [1, 1]), device);
        copy_into(&as_constant_float(&src), &as_constant_float(&self.dst))
    }

    /// The pinned buffer, for use inside the window. Every kernel that reads
    /// this sees the value written by the most recent `put`.
    pub fn east(&self) -> Tensor<2> {
        self.dst.clone()
    }

    #[allow(dead_code)]
    fn rank(&self) -> usize {
        2
    }
}

/// The pinned state of one parameter's optimizer: the moments and, for 2D
/// Muon-routed parameters, the step-size constants.
/// `mm` is the Muon momentum buffer; `m1`/`m2` the adaptive moments; `time`
/// lives on the host, so it never crosses a replay boundary as a tensor.
pub struct ParamState {
    mm: Option<Pins>,
    m1: Option<Pins>,
    m2: Option<Pins>,
}

// NOTE: `Pins` is a map keyed by ParamId — one module's pins, not one tensor.
// One parameter's pinned state wraps each field in a plain `Master` instead.
