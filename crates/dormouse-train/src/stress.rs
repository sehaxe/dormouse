//! Training stability stress test (Qwen3.8-Flash-Next §3.3).
//!
//! The report reproduces production-scale instabilities at small scale by
//! holding the LR constant at a multiple of the optimum: loss spikes
//! (step above a 201-step rolling median + 0.1), p99.9 pre-clip grad norm
//! and activation outliers separate a stable recipe from a fragile one
//! within a few hundred steps. Enabled via `TrainCfg::stress` (constant LR
//! at `stress_lr`x), report cadence via `stress_every`.

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
    /// Non-finite observations refused (a NaN loss cannot be a median and a
    /// NaN grad norm cannot be a percentile; both used to sort as `Equal`
    /// and silently poison the window, ADR-0019).
    refused: u64,
    log_every: usize,
}

impl StressMonitor {
    /// Constant-LR monitor at `lr_mult`x the base LR, reporting every
    /// `log_every` steps.
    pub fn new(lr_mult: f64, log_every: usize) -> Self {
        Self {
            lr_mult,
            history: VecDeque::with_capacity(201),
            spikes: 0,
            grad_norms: Vec::new(),
            refused: 0,
            log_every,
        }
    }

    /// Constant LR (no WSD decay): the stress protocol's defining switch.
    pub fn lr(&self, base: f64) -> f64 {
        base * self.lr_mult
    }

    /// Feed one step: loss spike detection + grad norm record. Non-finite
    /// values are REFUSED and counted, never sorted into the statistics: the
    /// old `partial_cmp(..).unwrap_or(Equal)` treated a NaN loss as equal to
    /// its neighbours, so a window containing one NaN reported a median and a
    /// spike count that were both fiction.
    pub fn observe(&mut self, ce: f32, grad_norm: f32) {
        if !ce.is_finite() || !grad_norm.is_finite() {
            self.refused += 1;
            return;
        }
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
            "stress: lr={}x spikes={} p99.9_grad={:.3} grad_last={:.3} refused_nonfinite={}",
            self.lr_mult,
            self.spikes,
            p999,
            self.grad_norms.last().copied().unwrap_or(f32::NAN),
            self.refused
        ))
    }
}

/// L2 norm of all parameter gradients (pre-clip). The squares accumulate in
/// one device tensor and the host reads a single scalar, so this costs one
/// device sync regardless of parameter count. Call at monitor cadence, not
/// per step. Non-destructive: reads via `Param::grad`.
///
/// NaN - never 0.0 - when NO gradient was found at all. The hot loop reads an
/// exactly-zero norm as "the NaN firewall fired this window", so returning
/// 0.0 for "no gradient / readback failed" made a broken gradient set
/// indistinguishable from a masked NaN step, and a masked step
/// indistinguishable from a healthy one (ADR-0019, the class of the
/// `ce=0.000` firewall bug).
pub fn grad_norm<M: burn::module::Module>(model: &M, grads: &burn::tensor::Gradients) -> f32 {
    struct NormVisitor<'a> {
        grads: &'a burn::tensor::Gradients,
        acc: Option<Tensor<1>>,
    }
    impl ModuleVisitor for NormVisitor<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if let Some(g) = param.grad(self.grads) {
                let sq = g.powf_scalar(2.0).sum();
                self.acc = Some(match self.acc.take() {
                    Some(a) => a + sq,
                    None => sq,
                });
            }
        }
    }
    let mut v = NormVisitor { grads, acc: None };
    model.visit(&mut v);
    v.acc
        .map(|t| t.sqrt().into_scalar())
        .unwrap_or(f32::NAN) // no gradient at all: NaN, not a fake clean 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The firewall's accounting rests on ONE number: an exactly-zero
    /// gradient norm means "every gradient on this step was non-finite and
    /// got zeroed". `grad_norm` used to return 0.0 when it found no gradient
    /// at all, so "no gradient / readback failed" said "the firewall fired"
    /// - and a masked step then looked like a healthy one in the same log
    /// line. ADR-0019's class: a value that means two things, both of them
    /// about health.
    #[test]
    fn no_gradient_is_nan_not_a_clean_zero() {
        use burn::nn::LinearConfig;
        let dev = burn::tensor::Device::ndarray().autodiff();
        let lin = LinearConfig::new(4, 4).init(&dev);
        // A gradient set that shares no parameter with `lin` at all.
        let unrelated = burn::tensor::Tensor::<1>::zeros([1], &dev)
            .require_grad()
            .sum()
            .backward();
        let gn = grad_norm(&lin, &unrelated);
        assert!(
            gn.is_nan(),
            "no gradient found must be NaN, not 0.0 (a 0.0 is read as 'the NaN firewall fired'): got {gn}"
        );
        // And a REAL gradient set still reports a real norm, so the change
        // did not turn the monitor off.
        let real = lin.weight.val().clone().require_grad().sum().backward();
        let gn2 = grad_norm(&lin, &real);
        assert!(gn2 > 0.0 && gn2.is_finite(), "a real gradient must report its norm, got {gn2}");
        // The stress monitor refuses a non-finite observation and COUNTS it,
        // instead of sorting NaN as Equal and reporting a fiction.
        let mut m = StressMonitor::new(2.0, 1);
        m.observe(f32::NAN, 1.0);
        m.observe(1.0, 2.0);
        let line = m.report(0).expect("report at every step");
        assert!(line.contains("refused_nonfinite=1"), "{line}");
        assert!(line.contains("grad_last=2.000"), "{line}");
    }
}