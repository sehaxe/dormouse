//! Training stability stress test (Qwen3.8-Flash-Next §3.3).
//!
//! The report reproduces production-scale instabilities at small scale by
//! holding the LR constant at a multiple of the optimum: loss spikes
//! (step above a 201-step rolling median + 0.1), p99.9 pre-clip grad norm
//! and activation outliers separate a stable recipe from a fragile one
//! within a few hundred steps. Enable with `DM_STRESS=1` (constant LR,
//! default 1x, multiplier via `DM_STRESS_LR`), log cadence via
//! `DM_STRESS_EVERY` (default 50).

use std::collections::VecDeque;

use burn::module::{ModuleVisitor, Param};
use burn::tensor::Tensor;

/// Rolling-window spike counter + pre-clip grad norm distribution.
pub struct StressMonitor {
    /// Constant LR multiplier (the report: 2x and 4x the optimal LR).
    pub lr_mult: f64,
    /// Loss history for the 201-step rolling median.
    history: VecDeque<f32>,
    /// Steps whose loss exceeded median + 0.1.
    pub spikes: u64,
    /// Pre-clip gradient norms (p99.9 reported).
    grad_norms: Vec<f32>,
    log_every: usize,
}

impl StressMonitor {
    /// Build from env: `DM_STRESS=1` enables, `DM_STRESS_LR` sets the LR
    /// multiplier, `DM_STRESS_EVERY` the report cadence.
    pub fn from_env() -> Option<Self> {
        let on = std::env::var("DM_STRESS").map(|v| v != "0").unwrap_or(false);
        if !on {
            return None;
        }
        let lr_mult = std::env::var("DM_STRESS_LR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        let log_every: usize = std::env::var("DM_STRESS_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);
        Some(Self {
            lr_mult,
            history: VecDeque::with_capacity(201),
            spikes: 0,
            grad_norms: Vec::new(),
            log_every,
        })
    }

    /// Constant LR (no WSD decay): the stress protocol's defining switch.
    pub fn lr(&self, base: f64) -> f64 {
        base * self.lr_mult
    }

    /// Feed one step: loss spike detection + grad norm record.
    pub fn observe(&mut self, ce: f32, grad_norm: f32) {
        self.grad_norms.push(grad_norm);
        self.history.push_back(ce);
        if self.history.len() > 201 {
            self.history.pop_front();
        }
        if self.history.len() == 201 {
            let mut sorted: Vec<f32> = self.history.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted[100];
            if ce > median + 0.1 {
                self.spikes += 1;
            }
        }
    }

    /// Report line at cadence.
    pub fn report(&self, step: u64) -> Option<String> {
        if step as usize % self.log_every != 0 {
            return None;
        }
        let p999 = if self.grad_norms.len() >= 1000 {
            let mut s = self.grad_norms.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            s[(s.len() as f64 * 0.999) as usize]
        } else {
            self.grad_norms.iter().copied().fold(0.0f32, f32::max)
        };
        Some(format!(
            "stress: lr={}x spikes={} p99.9_grad={:.3} grad_last={:.3}",
            self.lr_mult,
            self.spikes,
            p999,
            self.grad_norms.last().copied().unwrap_or(0.0)
        ))
    }
}

/// L2 norm of all parameter gradients (pre-clip). Syncs the device; call at
/// monitor cadence, not per step. Non-destructive: reads via `Param::grad`.
pub fn grad_norm<M: burn::module::Module>(model: &M, grads: &burn::tensor::Gradients) -> f32 {
    struct NormVisitor<'a> {
        grads: &'a burn::tensor::Gradients,
        sq: f64,
    }
    impl ModuleVisitor for NormVisitor<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if let Some(g) = param.grad(self.grads) {
                let s: f32 = g.powf_scalar(2.0).sum().into_scalar();
                self.sq += s as f64;
            }
        }
    }
    let mut v = NormVisitor { grads, sq: 0.0 };
    model.visit(&mut v);
    v.sq.sqrt() as f32
}