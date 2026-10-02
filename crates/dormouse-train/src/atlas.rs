//! Per-stage kernel-launch atlas (`DM_LAUNCH_ATLAS=1`).
//!
//! The launch-bound attack needs to know WHERE the 21 434 launches/step
//! come from before any fusion work is ranked. This wraps the stage timers
//! that already exist in `train_loop` (fwd / bwd / opt / retr / ema) and
//! reads `cubecl_launches()` at the same five boundaries, so each stage's
//! population is the counter delta across it. Per-ARM attribution comes
//! from differential runs (`--no-kda`, `--jepa-weight 0`, `--set
//! n_experts=1`, `--set max_iter=2`): arm off = its launches gone, same
//! stage marks in both runs.
//!
//! Counting only, §1.3-clean: the counter is a host-side atomic incremented
//! on the device thread, so a mark is one atomic load and one push - no
//! sync, no host branch on a device value. Disabled (no env var) it costs
//! one env read per process and nothing per step. Known bound: the device
//! thread may lag the enqueue, so a stage boundary can smear into its
//! neighbour by a fraction of a step at startup; steady-state per-step
//! totals are exact and per-stage deltas settle once the host runs ahead.

/// The 24 us/launch host-enqueue estimate the atlas' ms column is quoted at
/// (launch-bound host cost, release-measured on this box). Not a constant
/// in code because it is arithmetic for the report, not a knob.
pub const US_PER_LAUNCH: f64 = 24.0;

pub struct LaunchAtlas {
    on: bool,
    step: u64,
    last: u64,
    /// This step's (stage, launches, ms), in mark order.
    rows: Vec<(&'static str, u64, f32)>,
    /// Every (step, stage, launches, ms) of the run, for the warm summary.
    all: Vec<(u64, &'static str, u64, f32)>,
}

impl LaunchAtlas {
    pub fn from_env() -> Self {
        Self::new(std::env::var("DM_LAUNCH_ATLAS").is_ok_and(|v| v == "1"))
    }

    pub fn new(on: bool) -> Self {
        Self { on, step: 0, last: 0, rows: Vec::new(), all: Vec::new() }
    }

    /// Record one stage boundary. `ms` is the stage timer that already
    /// exists; the launch delta comes from the global counter.
    pub fn mark(&mut self, step: u64, stage: &'static str, ms: f32) {
        if !self.on {
            return;
        }
        let now = crate::cubecl_launches();
        let d = now - self.last;
        self.last = now;
        self.step = step;
        self.rows.push((stage, d, ms));
        self.all.push((step, stage, d, ms));
    }

    /// Flush the step's rows as TSV. Call once per step, after the last
    /// stage mark - so a run that dies mid-way keeps its rows.
    pub fn step_done(&mut self) {
        if !self.on {
            return;
        }
        for (stage, l, ms) in self.rows.drain(..) {
            println!("ATLAS\t{}\t{stage}\t{l}\t{ms:.1}", self.step);
        }
    }

    /// Warm-window summary: mean launches/step and ms/step per stage over
    /// the last third of the requested steps (steps 20+ of a 30-step run),
    /// descending by launches. Cold steps are autotune garbage; the warm
    /// window is the reading (AGENTS.md 2.2).
    pub fn finish(&self, steps: u64) {
        if !self.on {
            return;
        }
        let warm = steps * 2 / 3;
        let mut stages: Vec<(&'static str, u64, u32, f32)> = Vec::new();
        for &(step, stage, l, ms) in &self.all {
            if step < warm {
                continue;
            }
            match stages.iter_mut().find(|(s, ..)| *s == stage) {
                Some((_, sl, n, sm)) => {
                    *sl += l;
                    *n += 1;
                    *sm += ms;
                }
                None => stages.push((stage, l, 1, ms)),
            }
        }
        stages.sort_by(|a, b| (b.1 / u64::from(b.2)).cmp(&(a.1 / u64::from(a.2))));
        println!(
            "ATLAS-WARM\tfrom_step={warm}\tstage\tlaunches_per_step\tms_per_step\tms_at_24us"
        );
        for (stage, sl, n, sm) in stages {
            let lps = sl as f64 / f64::from(n);
            println!(
                "ATLAS-WARM\t{stage}\t{lps:.1}\t{:.1}\t{:.1}",
                sm / n as f32,
                lps * US_PER_LAUNCH / 1000.0
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_atlas_is_inert() {
        let mut a = LaunchAtlas::new(false);
        a.mark(1, "fwd", 1.0);
        a.step_done();
        a.finish(30);
        assert!(a.all.is_empty());
    }

    #[test]
    fn marks_accumulate_and_flush_in_order() {
        // Off CUDA the counter reads 0, so every delta is 0 - the test pins
        // the plumbing (rows accumulate, flush clears, summary runs), not
        // the counter. The counter itself is gated on CUDA hardware
        // (cubecl-cuda/tests/graph_step.rs).
        let mut a = LaunchAtlas::new(true);
        a.mark(7, "fwd", 10.0);
        a.mark(7, "bwd", 20.0);
        assert_eq!(a.rows.len(), 2);
        assert_eq!(a.all.len(), 2);
        a.step_done();
        assert!(a.rows.is_empty());
        a.finish(30); // warm = 20, so both rows sit below the window
        assert_eq!(a.all.len(), 2);
    }
}
