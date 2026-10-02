//! # The fused CUDA kernel, against the SAME tier-(a) fixture as the tensor path.
//!
//! `rmsnorm_oracle.rs` names its own gap in its module docs: *"The fused CUDA
//! kernel is NOT covered by any of this ... closing it needs a GPU window."*
//! This file is that window, and it also settles the question the gap was
//! standing in the way of: **is the kernel reachable at all?**
//!
//! ## Reachability, measured, and the answer is "yes, on a bare device"
//!
//! `RMSNorm::forward` asks the fused kernel on every call under the `cuda`
//! feature (`src/lib.rs:46`) and takes the tensor path when
//! `fused::rmsnorm_cuda` returns `None`. The decline happens one line into it,
//! at `x.try_into_primitive::<burn_cubecl::CubeBackend>()`
//! (`src/fused.rs:110`), and what declines is **the autodiff context**:
//! `DispatchKindConversion::try_into_backend` refuses any dispatch tensor whose
//! `autodiff` field is not `Disabled`
//! (burn-dispatch-0.22.0-pre.4 `src/tensor.rs:481-487`, verbatim:
//! `"Expected concrete Cube backend with disabled autodiff context, got {:?}"`).
//!
//! A bare `Device::cuda(0)` has `autodiff == Disabled`, and it is the *same*
//! backend type — `burn_cuda::Cuda` is a re-export of `burn_cubecl::Cube`
//! (burn-cuda-0.22.0-pre.4 `src/lib.rs:10`), not a second runtime. So the
//! kernel RUNS there. Two shapes, two tests, one row each:
//!
//! | shape | asked | ran |
//! |---|---|---|
//! | bare `Device::cuda(0)` | +1 | **+1** — the kernel's own output |
//! | `Device::cuda(0).autodiff()` | +1 | **+1** — the autodiff node arm, since 2026-10-02 |
//!
//! The second row WAS the trainer reading `norm=0/N` (`~/logs/train_nokda.log:13`,
//! `norm=0/1569` at step 500, `norm=0/3129` at step 1000 — 1560 asks per 500
//! steps, zero runs, on the real binary on a real GPU). The autodiff node
//! (`src/ops.rs`) ended that: the same ask now runs the kernel with a graph
//! attached. `the_fused_kernel_runs_on_an_autodiff_tensor` keeps it that way,
//! so the claim is a gate and not a doc string.
//!
//! ## The numbers
//!
//! Every case of `tests/fixtures/rmsnorm_oracle.txt`, on the bare device, at
//! the sibling file's `TOL_REL` (1e-5, unloosened) against **both** upstream
//! columns. The fixture is tier (a) and unchanged:
//!
//! | # | what was run | version | pinned by |
//! |---|---|---|---|
//! | 1 | `torch.nn.functional.rms_norm` -> `torch.rms_norm`, PyTorch's own ATen operator; the arithmetic is compiled C++ (`aten/src/ATen/native/layer_norm.cpp::rms_norm` in [pytorch/pytorch](https://github.com/pytorch/pytorch) at tag `v2.14.0`) | `torch==2.14.0+cpu` | `tests/oracle/upstream/torch_rms_norm.py`, a verbatim `inspect.getsource()` transcript |
//! | 2 | `fla.modules.layernorm.rms_norm_ref`, flash-linear-attention's own torch-level reference | `flash-linear-attention==0.5.2` from [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | `tests/oracle/upstream/fla_rms_norm_ref.py`, verbatim, plus `sha256(fla/modules/layernorm.py) = e78b729bcba29b30d8d6ddb6ce17d465261a6462be1363a04a6b3dc51c6d5c6f` in the fixture header |
//!
//! Both were executed 2026-09-29 on this box, CPU only, and emitted at 9
//! significant digits by `tests/oracle/gen_rmsnorm_oracle.py`.
//!
//! What that does **not** license is stated in the sibling file and is not
//! restated as licence here: the authors of arXiv:1910.07467 ship no code, so
//! "the authors' own implementation" does not exist and cannot be run. The two
//! upstreams agree with each other to 0.0 relative on all 12 cases, so the
//! expected column is not a compromise between two opinions.
//!
//! ## The counter assertion is the test
//!
//! A comparison against the fixture passes just as happily on the tensor
//! path, which is the sibling's job. So every case here asserts the seam
//! counters moved — `asked +1` **and** `ran +1` — around its own forward. If
//! the kernel ever starts declining, this file goes red with "the fused kernel
//! did not run" instead of quietly re-testing the tensor path under a file
//! whose name says otherwise.
//!
//! ## Seen red
//!
//! `tests/oracle/mutate_fused_kernel.sh` perturbs `src/fused.rs` — five wrong
//! kernels, the CUDA twins of the tensor path's M1..M5 — and each turns at
//! least one test here red by many orders of magnitude. Measured output is in
//! the script's own header.
//!
//! Run: `cargo test -p burn-rmsnorm --test rmsnorm_kernel_cuda --features cuda`

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::sync::Mutex;

use burn::module::{Module, Param};
use burn::tensor::{Device, Tensor, TensorData};

use burn_rmsnorm::RMSNorm;

/// **Unloosened, and the sibling's value**: `TOL_REL` in `rmsnorm_oracle.rs`,
/// whose module docs derive it (~20x the f32 sum bound of ~5e-7). A kernel
/// that runs on a different machine, with a different reduction order, is not
/// entitled to a wider bound than the one that runs everywhere else.
const TOL_REL: f64 = 1e-5;

/// The SAME fixture bytes the tensor-path oracle reads. Re-read here rather
/// than shared through a module: a `tests/common/` module would have to edit
/// the sibling file (which has a live mutation script and a committed
/// five-mutant table), and two readers of a flat `key: v v v` file is a
/// smaller risk than two owners of one file.
const FIXTURE: &str = include_str!("fixtures/rmsnorm_oracle.txt");

/// The counters are process-global (`fused.rs`), and cargo runs the tests in
/// one binary on several threads. Every test that reads them holds this.
static SEAM: Mutex<()> = Mutex::new(());

/// `(ran, asked)`: how many of the times the kernel was asked it actually ran.
/// `ran == 0` is the trainer's shape, and is what the eval line prints as the
/// first half of `norm=0/N`.
fn seam() -> (u64, u64) {
    let (asked, skipped) = burn_rmsnorm::fused::calls();
    (asked.saturating_sub(skipped), asked)
}

/// The trainer's device: an autodiff one. Every training forward, and the
/// `model.valid()` eval snapshot, run on this.
fn autodiff_cuda() -> Device {
    Device::cuda(0).autodiff()
}

/// A bare CUDA device: `autodiff == Disabled`, so the dispatch conversion at
/// `fused.rs:110` is allowed through and the kernel's own arithmetic is what
/// gets compared.
fn bare_cuda() -> Device {
    Device::cuda(0)
}

struct Fx {
    map: HashMap<String, Vec<f64>>,
    meta: HashMap<String, String>,
}

impl Fx {
    fn load() -> Self {
        let (mut map, mut meta): (HashMap<String, Vec<f64>>, HashMap<String, String>) =
            (HashMap::new(), HashMap::new());
        for line in FIXTURE.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, body)) = line.split_once(':') else {
                panic!("fixture line without ':' — {line:?}");
            };
            // `meta.*` is provenance (versions, a path, a sha256), not numbers.
            if key.starts_with("meta.") {
                meta.insert(key.to_string(), body.trim().to_string());
                continue;
            }
            map.entry(key.to_string())
                .or_default()
                .extend(body.split_whitespace().map(|v| {
                    v.parse::<f64>()
                        .unwrap_or_else(|_| panic!("bad number {v:?}"))
                }));
        }
        assert!(
            map.keys().any(|k| k.ends_with("out_torch")),
            "the fixture has no expected columns at all"
        );
        Self { map, meta }
    }
    fn cases(&self) -> Vec<String> {
        self.meta
            .get("meta.cases")
            .expect("fixture has no meta.cases")
            .split_whitespace()
            .map(str::to_string)
            .collect()
    }
    fn text(&self, key: &str) -> &str {
        self.meta
            .get(key)
            .map(String::as_str)
            .unwrap_or_else(|| panic!("no {key:?}"))
    }
    fn get(&self, key: &str) -> &[f64] {
        self.map
            .get(key)
            .unwrap_or_else(|| panic!("fixture key {key:?} missing"))
    }
    fn one(&self, key: &str) -> f64 {
        let v = self.get(key);
        assert_eq!(v.len(), 1, "{key} should be a scalar, got {}", v.len());
        v[0]
    }
    fn dims(&self, case: &str) -> Vec<usize> {
        self.get(&format!("case.{case}.dims"))
            .iter()
            .map(|v| *v as usize)
            .collect()
    }
    /// The sibling's `rel_diff`, unchanged: max |a - b| / max(|b|, 1), and a
    /// non-finite on EITHER side is maximally separated rather than skipped.
    fn rel_diff(&self, case: &str, ours: &[f32], col: &str) -> f64 {
        let want = self.get(&format!("case.{case}.{col}"));
        assert_eq!(
            ours.len(),
            want.len(),
            "{case}/{col}: {} vs {}",
            ours.len(),
            want.len()
        );
        ours.iter()
            .zip(want)
            .map(|(a, b)| {
                if !a.is_finite() || !b.is_finite() {
                    return f64::INFINITY;
                }
                (f64::from(*a) - b).abs() / b.abs().max(1.0)
            })
            .fold(0.0f64, f64::max)
    }
}

/// A module carrying the fixture's own gain — never all-ones, so a broken
/// per-feature broadcast cannot hide.
fn norm_for(fx: &Fx, case: &str, dev: &Device) -> RMSNorm {
    let d = fx.dims(case)[2];
    let eps = fx.one(&format!("case.{case}.eps")) as f32;
    let w = fx.get(&format!("case.{case}.w"));
    let mut norm = RMSNorm::new(d, eps, dev);
    norm.weight = Param::from_tensor(Tensor::<1>::from_data(
        TensorData::new(w.iter().map(|v| *v as f32).collect::<Vec<_>>(), [d]),
        dev,
    ));
    norm
}

fn run_case(fx: &Fx, case: &str, norm: &RMSNorm, dev: &Device) -> Vec<f32> {
    let dims = fx.dims(case);
    let xs = fx.get(&format!("case.{case}.x"));
    assert_eq!(xs.len(), dims.iter().product::<usize>());
    let x = Tensor::<3>::from_data(
        TensorData::new(xs.iter().map(|v| *v as f32).collect::<Vec<_>>(), dims),
        dev,
    );
    norm.forward(x)
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

// ── 1. the kernel, on a bare device, against both upstreams ───────────────

/// Every case, on a bare CUDA device, against `out_torch` and `out_fla` — and
/// with the seam counters pinned around each case, so the thing being compared
/// is the KERNEL's output and not the tensor path's.
#[test]
fn the_fused_kernel_matches_both_upstreams() {
    let _g = SEAM.lock().unwrap_or_else(|e| e.into_inner());
    let fx = Fx::load();
    let cases = fx.cases();
    assert_eq!(cases.len(), 12, "the fixture grew/shrank: {:?}", cases);

    let dev = bare_cuda();
    let (mut worst_torch, mut worst_fla) = (0.0f64, 0.0f64);
    for case in &cases {
        let (ran0, asked0) = seam();
        let ours = run_case(&fx, case, &norm_for(&fx, case, &dev), &dev);
        let (ran1, asked1) = seam();

        assert_eq!(
            asked1 - asked0,
            1,
            "case {case}: the fused kernel was asked {} times, not once. The path \
             under test changed; this file is the gate that says so.",
            asked1 - asked0
        );
        // The ENVELOPE, not "always". Measured 2026-09-30: the launch's trailing
        // cubes do not execute when the feature width is below
        // `MIN_FUSED_D`, so at d=1 and d=2 the fused arm returns a PARTLY
        // UNWRITTEN tensor - 5.245e-1 relative on this file's own case `d2`,
        // where the tensor path is 7.29e-8. `fused.rs` now refuses that whole
        // class, and this assertion is what keeps the refusal honest in BOTH
        // directions: d below the floor must DECLINE (a pass must never be the
        // silently-wrong answer), and d at or above it must RUN (or this file
        // stops testing the kernel and says so).
        let d = fx.dims(case)[2];
        let expected_ran = usize::from(d >= burn_rmsnorm::fused::MIN_FUSED_D);
        assert_eq!(
            ran1 - ran0,
            expected_ran as u64,
            "case {case} (d={d}): the fused kernel's {} count is {expected_ran} at the \
             claimed envelope (MIN_FUSED_D={}). asked={asked1} ran={ran1}. Either the \
             guard has moved without the underlying launch being fixed, or the \
             envelope is being violated. See tests/d2_isolate.rs.",
            if expected_ran == 1 { "ran" } else { "DECLINE" },
            burn_rmsnorm::fused::MIN_FUSED_D
        );

        for (col, worst) in [("out_torch", &mut worst_torch), ("out_fla", &mut worst_fla)] {
            let d = fx.rel_diff(case, &ours, col);
            assert!(
                d <= TOL_REL,
                "the fused arm on case {case} (d={d}) differs from the reference's \
                 `{col}` column by {d:e} relative (> {TOL_REL:e}). Fixture: \
                 tests/fixtures/rmsnorm_oracle.txt"
            );
            *worst = worst.max(d);
        }
    }
    eprintln!(
        "fused CUDA kernel vs upstream, {} cases: max rel diff {worst_torch:e} against \
         torch.nn.functional.rms_norm (torch {}, ATen) and {worst_fla:e} against \
         fla.modules.layernorm.rms_norm_ref (flash-linear-attention {}, sha256 {}), \
         at TOL_REL {TOL_REL:e}",
        cases.len(),
        fx.text("meta.upstream_torch_version"),
        fx.text("meta.upstream_fla_version"),
        &fx.text("meta.upstream_fla_sha256")[..8],
    );
}

/// eps is a PARAMETER of the call, on this arm too. `micro` at eps = 1e-5 and
/// `micro_eps0` at eps = 0 are the SAME tensor, so any difference between the
/// two kernel runs is attributable to eps and to nothing else — and a kernel
/// with eps baked in makes the two identical. This is the test that owns "eps
/// reached the KERNEL"; the sibling owns it for the tensor path.
#[test]
fn eps_reaches_the_fused_kernel() {
    let _g = SEAM.lock().unwrap_or_else(|e| e.into_inner());
    let fx = Fx::load();
    assert_eq!(
        fx.get("case.micro.x"),
        fx.get("case.micro_eps0.x"),
        "`micro` and `micro_eps0` are documented as the same tensor; they are not, \
         and every gap below would then be a measurement of the DATA"
    );
    assert_ne!(
        fx.get("case.micro.eps")[0],
        fx.get("case.micro_eps0.eps")[0],
        "the two cases are meant to differ ONLY in eps"
    );

    let dev = bare_cuda();
    let a = run_case(&fx, "micro", &norm_for(&fx, "micro", &dev), &dev);
    let b = run_case(&fx, "micro_eps0", &norm_for(&fx, "micro_eps0", &dev), &dev);
    let gap = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs() / f64::from(*y).abs().max(1.0))
        .fold(0.0f64, f64::max);
    eprintln!(
        "the FUSED KERNEL at eps 1e-5 vs eps 0 on the SAME `micro` tensor moves the \
         output by {gap:e} relative (analytic sqrt(1 + e/r^2) - 1 at r = 9.08e-5: 33.8)"
    );
    assert!(
        gap > 2.0 * TOL_REL,
        "changing eps from 1e-5 to 0 moved the FUSED KERNEL's output by only {gap:e}; \
         the eps is not reaching the implementation (or the two fixture cases are not \
         the same shape)"
    );
}

// ── 2. the trainer's shape, and the gate on `norm=0/N` ────────────────────

/// The trainer runs on an autodiff device, and the kernel declines there —
/// `norm=0/N` on every eval line on record
/// (`~/logs/train_nokda.log:13`, `norm=0/1569`). This is that claim as a test,
/// because the claim is about the FUTURE as much as the past: if someone wires
/// the kernel into a training forward, the numerics change under every reader
/// of those logs and this goes red instead.
///
/// The output is checked against the same fixture, so the decline is pinned as
/// a CORRECT fallback and not merely as a decline: the tensor path agreeing
/// with the two upstreams is the sibling's claim, restated on the arm that
/// would be taken if this kernel ever engaged.
#[test]
fn the_fused_kernel_runs_on_an_autodiff_tensor() {
    let _g = SEAM.lock().unwrap_or_else(|e| e.into_inner());
    let fx = Fx::load();
    let dev = autodiff_cuda();
    assert!(dev.is_autodiff(), "the trainer's device is an autodiff one");

    let (ran0, asked0) = seam();
    let ours = run_case(&fx, "main", &norm_for(&fx, "main", &dev), &dev);
    let (ran1, asked1) = seam();

    assert_eq!(asked1 - asked0, 1, "the kernel was not asked once");
    assert_eq!(
        ran1 - ran0,
        1,
        "the fused kernel did NOT run on an autodiff tensor (asked={asked1} ran={ran1}). \
         Before 2026-10-02 that was the whole story — the node arm now carries the \
         graph the bare launch never had, so norm= must count a run."
    );
    assert!(
        fx.rel_diff("main", &ours, "out_torch") <= TOL_REL,
        "the node arm's output disagrees with the reference"
    );
    eprintln!(
        "autodiff device: asked {asked1}, ran {ran1} (delta asked {} ran {}) — the \
         node arm engaged",
        asked1 - asked0,
        ran1 - ran0
    );
}

/// The trainer's EVAL is `model.valid()` on a model built on the autodiff
/// device (`crates/dormouse-train/src/lib.rs:1416`). Whether that snapshot
/// strips the autodiff context decides whether the eval line's `norm=` field
/// could ever be non-zero, so it is measured rather than assumed.
#[test]
fn the_fused_kernel_runs_on_the_trainers_eval_snapshot() {
    let _g = SEAM.lock().unwrap_or_else(|e| e.into_inner());
    let fx = Fx::load();
    let dev = autodiff_cuda();
    // Exactly the trainer's shape: built and trained on the autodiff device,
    // then snapshotted with `valid()`, then fed a tensor built on that same
    // device (which is what the trainer hands it).
    let eval_model = norm_for(&fx, "main", &dev).valid();

    let (ran0, asked0) = seam();
    let ours = run_case(&fx, "main", &eval_model, &dev);
    let (ran1, asked1) = seam();

    assert_eq!(asked1 - asked0, 1, "the kernel was not asked once");
    assert_eq!(
        ran1 - ran0,
        1,
        "the fused kernel did NOT run inside a `valid()` snapshot \
         (asked={asked1} ran={ran1}). The eval line's `norm=` would read 0/N \
         while the kernel ships."
    );
    assert!(
        fx.rel_diff("main", &ours, "out_torch") <= TOL_REL,
        "the eval snapshot's output disagrees with the reference"
    );
    eprintln!(
        "`valid()` snapshot on an autodiff device: asked {asked1} ran {ran1} \
         (delta asked {} ran {})",
        asked1 - asked0,
        ran1 - ran0
    );
}
