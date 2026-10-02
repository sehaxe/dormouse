//! CUDA-graph capture of ONE stage: the EMA teacher's forward (`--graph-stage`).
//!
//! # Why THIS stage and not the main loop's KDA forward
//!
//! The lane brief's first choice was the fused KDA forward of the main loop
//! (the attention arm is 65.5% of the step's launches per the launch atlas,
//! `docs/reviews/launch-atlas-2026-10-02.md`). A captured region replays raw
//! kernel launches and runs NO host code — so the stage's burn autodiff nodes
//! are not rebuilt on a replay step. The main-loop KDA forward feeds
//! `loss.backward()`, which needs a fresh node graph every step: replaying
//! that forward would leave the attention arm without gradients — the `8fa5d4c`
//! defect shape (a frozen arm behind a healthy loss curve), which this project
//! spent a month digging out of its own history. That stage is therefore not
//! honestly capturable in training.
//!
//! The teacher's forward is. `forward_latent` runs on the trainer's autodiff
//! backend but builds no node graph at all: the teacher's params are
//! `no_grad`-frozen (`aux::ema_update` ends in `.no_grad()`), so every op is
//! `UnTracked` and the latents come back leaves — the JEPA loss detaches them
//! anyway. Gradient-free, sync-free, and the same kernel sequence every step
//! is exactly what a captured graph can stand in for. Its population (atlas
//! row 4): **4401.7 launches/step at the atlas recipe**, of which the KDA copy
//! alone (row 5) is 3099.4 — more KDA-arm launches than the main loop's own
//! forward row (3146.1). One replay dispatch replaces them.
//!
//! # The contract
//!
//! One capture for the life of the run (not one per ungraphed step — the
//! whole-step seam must re-capture because its window holds the gradients the
//! optimizer consumes; this seam's window holds none):
//!
//! - the teacher's parameters are PINNED to master buffers ([`graph::Pins`]);
//!   `ema_update` rebuilds every teacher `Param` each step, so the EMA'd
//!   values are copied back into the masters after every advance — the same
//!   `copy_into` the whole-step seam's pin makes;
//! - the batch `x` is pinned: built fresh from host bytes every step, copied
//!   into one fixed Int buffer (one launch);
//! - the capture runs `forward_latent` once to prime the pool, once inside
//!   the recording (recorded, not executed), then REPLAYS — one replay
//!   executes the recording and fills the output buffer with the real latents
//!   (the seam-measured semantics: a capture records, it does not run);
//! - every later graphed step copies the batch in and replays; the latents
//!   tensor handed to the JEPA loss is THE SAME buffer every step;
//! - steps that read anything back (log/eval/500-step cadences) run the
//!   teacher's forward normally and do not disturb the graph: the graph owns
//!   only pinned masters and slices `capture_end` retained, which neither the
//!   allocator nor the 500-step `memory_cleanup` may hand out again
//!   (`PersistentPool::cleanup` frees free slices only).
//!
//! A capture that is refused is refused LOUDLY under the same budget as the
//! whole-step seam: two consecutive failures disable the stage for the run
//! ([`graph::Stats`]) instead of growing the pool toward ILLEGAL_ADDRESS.

use burn::tensor::Tensor;
use dormouse_core::DormouseModel;

pub use crate::graph::Stats;

/// The teacher-forward stage of one run.
#[derive(Default)]
pub struct StageSeam {
    #[cfg(feature = "cuda")]
    client: Option<cubecl_runtime::client::Client>,
    graph: Option<std::sync::Arc<dyn crate::graph::GraphReplay>>,
    pins: crate::graph::Pins,
    /// The pinned batch buffer, allocated from the first batch's shape.
    x: Option<Tensor<2, burn::tensor::Int>>,
    /// The stage's output buffer: the tensor the recorded pass produced, which
    /// every replay refills. `None` until the first capture.
    out: Option<Tensor<3>>,
    pub stats: Stats,
    /// Batches copied into the pin (the report line).
    feeds: u64,
}

impl StageSeam {
    pub fn new(client: Option<cubecl_runtime::client::Client>) -> Self {
        #[cfg(feature = "cuda")]
        {
            Self { client, ..Default::default() }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = client;
            Self::default()
        }
    }

    /// Pin the teacher's parameters. The returned model is the one to keep
    /// feeding `ema_update`; its params live at the masters the graph baked.
    pub fn arm(&mut self, teacher: DormouseModel) -> DormouseModel {
        crate::graph::arm_pins(teacher, &mut self.pins)
    }

    /// Whether the pin is usable (loud at arm time otherwise).
    pub fn armed(&self) -> bool {
        self.param_copies() > 0 && self.unsupported_rank().is_none()
    }

    /// Pinned parameter count (the armed line).
    pub fn param_copies(&self) -> usize {
        self.pins.copies()
    }

    /// The rank the pin could not carry, if any.
    pub fn unsupported_rank(&self) -> Option<usize> {
        self.pins.unsupported_rank
    }

    /// Copy the EMA-updated teacher back into its masters and re-point it.
    /// Same price as the whole-step seam's pin: one copy per parameter.
    pub fn refresh(&mut self, teacher: DormouseModel) -> (DormouseModel, Result<(), String>) {
        crate::graph::refresh_pins(teacher, &mut self.pins)
    }

    /// The pinned batch: allocated from the first batch's shape, copied every
    /// step. An error names what could not be pinned — a graph fed a buffer
    /// that moves is the silent failure this module exists to prevent.
    fn feed(&mut self, x: &Tensor<2, burn::tensor::Int>) -> Result<Tensor<2, burn::tensor::Int>, String> {
        match &self.x {
            None => {
                let pin = Tensor::zeros(x.dims(), &x.device());
                self.x = Some(pin);
            }
            Some(pin) => {
                if pin.dims() != x.dims() {
                    return Err(format!(
                        "graph stage: the batch shape changed ({:?} vs the pinned {:?}) — a \
                         captured stage has one shape",
                        x.dims(),
                        pin.dims()
                    ));
                }
            }
        }
        let pin = self.x.as_ref().expect("just ensured");
        crate::graph::copy_into_int(
            &crate::graph::as_constant_int(x),
            &crate::graph::as_constant_int(pin),
        )
        .map_err(|e| format!("graph stage: could not pin x: {e}"))?;
        self.feeds += 1;
        Ok(self.x.as_ref().expect("just ensured").clone())
    }

    /// One step's teacher forward: replay the captured stage, run `body` and
    /// capture it, or run `body` ungraphed. Returns the latents tensor the
    /// JEPA loss consumes — the same buffer on every replayed step.
    ///
    /// `ungraphed` is the caller's cadence verdict (a step that reads anything
    /// back). The graph is NOT destroyed on those steps: unlike the whole-step
    /// window this stage holds no gradients, so nothing an ungraphed step does
    /// can invalidate the recorded launches.
    pub fn step(
        &mut self,
        ungraphed: bool,
        x: &Tensor<2, burn::tensor::Int>,
        body: impl Fn(Tensor<2, burn::tensor::Int>) -> Tensor<3>,
    ) -> Result<Tensor<3>, String> {
        let xin = self.feed(x)?;
        if !ungraphed {
            if let Some(g) = &self.graph {
                g.replay()
                    .map_err(|e| format!("graph stage replay failed: {e} — a run cannot continue on a graph it cannot drive"))?;
                self.stats.replays += 1;
                return Ok(self.out.clone().expect("the capture stored its output"));
            }
        }
        #[cfg(feature = "cuda")]
        {
            if !ungraphed && self.stats.capture_allowed() {
                if let Some(client) = &self.client {
                    if let Err(e) = client.graph_prepare() {
                        let out = body(xin.clone());
                        self.stats.refused(format!("graph_prepare: {e:?}"));
                        return Ok(out);
                    }
                    // Priming: one executed pass populates the persistent pool
                    // with the stage's whole working set, so the recorded pass
                    // allocates nothing (an allocation inside the window is a
                    // memory node, and a memory node refuses the capture).
                    let _ = body(xin.clone());
                    if let Err(e) = client.start_capture() {
                        let out = body(xin.clone());
                        self.stats.refused(format!("start_capture: {e:?}"));
                        return Ok(out);
                    }
                    // The recorded pass does not execute; the replay below does.
                    let out = body(xin.clone());
                    return match client.stop_capture() {
                        Ok(g) => {
                            let g: std::sync::Arc<dyn crate::graph::GraphReplay> =
                                std::sync::Arc::new(g);
                            self.out = Some(out.clone());
                            self.graph = Some(g);
                            self.stats.captures += 1;
                            self.stats.captured();
                            if let Some(g) = &self.graph {
                                g.replay().map_err(|e| {
                                    format!("graph stage replay after capture failed: {e}")
                                })?;
                                self.stats.replays += 1;
                            }
                            Ok(out)
                        }
                        Err(e) => {
                            let out = body(xin.clone());
                            self.stats.refused(format!("stop_capture: {e:?}"));
                            Ok(out)
                        }
                    };
                }
            }
        }
        Ok(body(xin))
    }

    pub fn report(&self) -> String {
        format!(
            "stage: captured {} replayed {} refused {}{} fed {}{}",
            self.stats.captures,
            self.stats.replays,
            self.stats.refusals,
            if self.stats.disabled { " DISABLED" } else { "" },
            self.feeds,
            match &self.stats.last_refusal {
                Some(r) => format!(" last refusal: {r}"),
                None => String::new(),
            }
        )
    }
}

/// Which arms `--graph-stage` cannot run beside, named. Pure so it is testable
/// without a GPU.
pub fn check(cfg: &crate::RunCfg) -> Result<(), String> {
    if !cfg.train.graph_stage {
        return Ok(());
    }
    if cfg.train.graph_capture {
        return Err(
            "--graph-stage and --graph-capture both own the one capture a device can run: \
             pick one (the stage captures the teacher's forward, the window the whole step)."
                .into(),
        );
    }
    if cfg.train.engram_ram {
        return Err(
            "--graph-stage refuses --engram-ram: the teacher's forward would read host_rows, \
             gathered and uploaded per step, which v1 does not pin."
                .into(),
        );
    }
    if cfg.train.jepa_targets.is_some() {
        return Err(
            "--graph-stage refuses --jepa-targets: both supply the JEPA latents, one from a \
             sidecar file, one from the captured teacher — the run would measure neither."
                .into(),
        );
    }
    if cfg.model.jepa_weight <= 0.0 {
        return Err(
            "--graph-stage supplies the JEPA teacher's latents, but this run has \
             jepa_weight = 0: nothing consumes them. Turn JEPA on or drop the flag."
                .into(),
        );
    }
    if cfg.model.use_engram {
        return Err(
            "--graph-stage v1 does not pin the Engram key tensor the teacher's forward would \
             read (use_engram = true); run it with --no-engram, the launch-atlas recipe."
                .into(),
        );
    }
    #[cfg(not(feature = "cuda"))]
    return Err(
        "--graph-stage needs the cuda feature: there is no graph on another backend, and \
         silently running without one would be a flag that does nothing."
            .into(),
    );
    #[cfg(feature = "cuda")]
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(stage: bool) -> crate::RunCfg {
        crate::RunCfg {
            source: "small".into(),
            model: dormouse_core::DormouseConfig::default(),
            train: crate::TrainCfg { graph_stage: stage, ..Default::default() },
        }
    }

    #[test]
    fn stage_check_is_inert_when_off() {
        assert!(check(&cfg(false)).is_ok());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn stage_check_passes_on_the_plain_recipe() {
        let mut c = cfg(true);
        c.model.jepa_weight = 0.05;
        assert!(check(&c).is_ok(), "{:?}", check(&c).err());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn stage_check_refuses_the_arms_it_cannot_stand_in_for() {
        let mut c = cfg(true);
        c.model.jepa_weight = 0.05;
        c.train.graph_capture = true;
        assert!(check(&c).is_err(), "--graph-capture must be refused");

        let mut c = cfg(true);
        c.model.jepa_weight = 0.05;
        c.train.engram_ram = true;
        assert!(check(&c).is_err(), "--engram-ram must be refused");

        let mut c = cfg(true);
        c.model.jepa_weight = 0.05;
        c.train.jepa_targets = Some("t.bin".into());
        assert!(check(&c).is_err(), "--jepa-targets must be refused");

        let mut c = cfg(true);
        c.model.jepa_weight = 0.0;
        assert!(check(&c).is_err(), "no JEPA consumer must be refused");

        let mut c = cfg(true);
        c.model.jepa_weight = 0.05;
        c.model.use_engram = true;
        assert!(check(&c).is_err(), "the keyed Engram arm must be refused");
    }

    #[test]
    fn report_names_the_counters() {
        let mut s = StageSeam::new(None);
        s.stats.captures = 1;
        s.stats.replays = 7;
        s.stats.refusals = 1;
        s.stats.last_refusal = Some("stop_capture: 3 memory node(s)".into());
        let line = s.report();
        assert!(line.contains("captured 1 replayed 7 refused 1"), "{line}");
        assert!(line.contains("stop_capture: 3 memory node(s)"), "{line}");
    }
}
