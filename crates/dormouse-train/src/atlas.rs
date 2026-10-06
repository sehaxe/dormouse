//! Per-stage timing atlas (`DM_LAUNCH_ATLAS=1`).
//!
//! The launch-bound attack needs to know WHERE a step's time goes before any
//! fusion work is ranked. This wraps the stage timers that already exist in
//! `train_loop` (fwd / bwd / opt / retr / ema) and flushes them as TSV.
//!
//! The LAUNCH COLUMNS are gone: the pre-0.22 fork counted at the single choke
//! point every launch passed through, stock cubecl 0.11 exposes no such hook,
//! and burn 0.22's native graph capture (#5146) shifts the question from
//! counting launches to capturing them. Per-arm attribution still comes from
//! differential runs (`--no-kda`, `--jepa-weight 0`, `--set n_experts=1`,
//! `--set max_iter=2`): arm off = its time gone, same stage marks in both
//! runs.
//!
//! Counting only, §1.3-clean: a mark is one timer read and one push — no
//! sync, no host branch on a device value. Disabled (no env var) it costs
//! one env read per process and nothing per step.

pub struct LaunchAtlas {
    on: bool,
    step: u64,
    /// This step's (stage, ms), in mark order.
    rows: Vec<(&'static str, f32)>,
    /// Every (step, stage, ms) of the run, for the warm summary.
    all: Vec<(u64, &'static str, f32)>,
}

impl LaunchAtlas {
    pub fn from_env() -> Self {
        Self::new(std::env::var("DM_LAUNCH_ATLAS").is_ok_and(|v| v == "1"))
    }

    pub fn new(on: bool) -> Self {
        Self { on, step: 0, rows: Vec::new(), all: Vec::new() }
    }

    /// Record one stage boundary. `ms` is the stage timer that already
    /// exists in `train_loop`.
    pub fn mark(&mut self, step: u64, stage: &'static str, ms: f32) {
        if !self.on {
            return;
        }
        self.step = step;
        self.rows.push((stage, ms));
        self.all.push((step, stage, ms));
    }

    /// Flush the step's rows as TSV. Call once per step, after the last
    /// stage mark - so a run that dies mid-way keeps its rows.
    pub fn step_done(&mut self) {
        if !self.on {
            return;
        }
        for (stage, ms) in self.rows.drain(..) {
            println!("ATLAS\t{}\t{stage}\t{ms:.1}", self.step);
        }
    }

    /// Warm-window summary: mean ms/step per stage over the last third of
    /// the requested steps (steps 20+ of a 30-step run), descending by time.
    /// Cold steps are autotune garbage; the warm window is the reading
    /// (AGENTS.md 2.2).
    pub fn finish(&self, steps: u64) {
        if !self.on {
            return;
        }
        let warm = steps * 2 / 3;
        let mut stages: Vec<(&'static str, u32, f32)> = Vec::new();
        for &(step, stage, ms) in &self.all {
            if step < warm {
                continue;
            }
            match stages.iter_mut().find(|(s, ..)| *s == stage) {
                Some((_, n, sm)) => {
                    *n += 1;
                    *sm += ms;
                }
                None => stages.push((stage, 1, ms)),
            }
        }
        stages.sort_by(|a, b| (b.2 / b.1 as f32).total_cmp(&(a.2 / a.1 as f32)));
        println!("ATLAS-WARM\tfrom_step={warm}\tstage\tms_per_step");
        for (stage, n, sm) in stages {
            println!("ATLAS-WARM\t{stage}\t{:.1}", sm / n as f32);
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
