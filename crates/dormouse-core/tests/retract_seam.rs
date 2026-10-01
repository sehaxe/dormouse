//! Differential test for the TSCT retraction: the per-factor path the trainer
//! has always run (`LinearLike::retract` -> `SpectralLinear::retract` ->
//! `polar_retracked`, one host-syncing `polar_orthogonalize` per factor) and
//! the grouped one (`burn_spectral::retract_batched`, factors stacked by shape
//! into sync-free batched Newton-Schulz) must produce THE SAME masters from THE
//! SAME inputs, on a real model's real factor set.
//!
//! Why a test and not a bench: the batched arm exists to remove ~50 ms of
//! per-step cost, and a speed arm is exactly the kind that can be wired in and
//! compute a subtly different matrix (a wrong transpose, a stale slot, a
//! dropped `require_grad`) while every counter still reads clean. This pins the
//! VALUES, the SHAPES, the TRAINING STATE and that the comparison is not
//! trivially true because neither arm did anything.
//!
//! The test-local `batched_retract` below is written independently of the
//! model's own `retract_tsct_batched`, so the test says something about the
//! library function even before (and independently of) the production entry
//! point it then pins too.
//!
//! Default run: `cargo test -p dormouse-core --test retract_seam`

use burn::module::{Param, ParamId, ParamMapper};
use burn::tensor::{Device, Tensor, TensorData};
use dormouse_core::param::LinearLikeInner;
use dormouse_core::{DormouseConfig, DormouseModel};

fn device() -> Device {
    Device::flex().autodiff()
}

/// Newton-Schulz iterations the trainer runs (`--retract-iters`, default 3).
const ITERS: usize = 3;

/// Same width cut as `model_seam`'s mini-nano: every property that defines
/// the retraction's factor set (TSCT on, 3 experts, 4 loop iterations, rank)
/// stays the preset's; only the widths shrink, because this is a numerical
/// agreement test and not a capacity one.
fn mini_nano() -> DormouseConfig {
    let preset = dormouse_core::config::load_config(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/nano.toml"
    ))
    .expect("configs/nano.toml loads");
    DormouseConfig {
        d_model: 128,
        n_heads: 4,
        head_dim: 32,
        d_ffn: 256,
        rank: 32,
        engram_rows: 4096,
        ..preset
    }
}

/// Every TSCT master the model retracts, in a fixed walk order (`u`, `v` per
/// linear, loop_block experts -> readout -> lm_head). Same order for both
/// clones, which is what makes the zip below a per-factor comparison.
fn masters(model: &DormouseModel) -> Vec<Tensor<2>> {
    let mut out: Vec<Tensor<2>> = Vec::new();
    let mut push = |l: &dormouse_core::LinearLike| {
        if let LinearLikeInner::Tsct(s) = &l.inner {
            out.push(s.u.val());
            out.push(s.v.val());
        }
    };
    for f in &model.loop_block.expert_ffns {
        push(&f.gate_up);
        push(&f.down);
    }
    push(&model.loop_block.out_proj);
    push(&model.lm_head);
    out
}

/// The same factors as writable `Param` slots.
fn master_slots(model: &mut DormouseModel) -> Vec<&mut Param<Tensor<2>>> {
    fn push<'a>(l: &'a mut dormouse_core::LinearLike, out: &mut Vec<&'a mut Param<Tensor<2>>>) {
        if let LinearLikeInner::Tsct(s) = &mut l.inner {
            out.push(&mut s.u);
            out.push(&mut s.v);
        }
    }
    let mut out: Vec<&mut Param<Tensor<2>>> = Vec::new();
    for f in &mut model.loop_block.expert_ffns {
        push(&mut f.gate_up, &mut out);
        push(&mut f.down, &mut out);
    }
    push(&mut model.loop_block.out_proj, &mut out);
    push(&mut model.lm_head, &mut out);
    out
}

/// Take every master's value out of its `Param`, retract them all in ONE
/// grouped call, and write them back - the bookkeeping the production entry
/// point has to do, written here against `burn_spectral::retract_batched`
/// directly.
///
/// Two invariants are copied from the scalar path on purpose, because they are
/// silent failures rather than wrong numbers: the `ParamId`/mapper survive the
/// round trip (the optimizer's records are keyed by id - a retraction that
/// handed back fresh ids would silently reset every factor's momentum), and
/// the `require_grad` state is MIRRORED, not forced (the polar output is a
/// non-leaf on an autodiff backend, which burn-optim downgrades to a frozen
/// master; forcing the flag would equally un-freeze a master the caller had
/// frozen).
fn batched_retract(model: &mut DormouseModel, iters: usize) -> usize {
    let mut slots = master_slots(model);
    let mut ids: Vec<ParamId> = Vec::with_capacity(slots.len());
    let mut maps: Vec<ParamMapper<Tensor<2>>> = Vec::with_capacity(slots.len());
    let mut tracked: Vec<bool> = Vec::with_capacity(slots.len());
    let mut vals: Vec<Tensor<2>> = Vec::with_capacity(slots.len());
    for p in slots.iter() {
        let was_tracked = p.val().is_require_grad();
        let (id, val, map) = Param::clone(p).consume();
        ids.push(id);
        maps.push(map);
        tracked.push(was_tracked);
        vals.push(val);
    }
    let n = vals.len();
    {
        let mut refs: Vec<&mut Tensor<2>> = vals.iter_mut().collect();
        burn_spectral::retract_batched(&mut refs, iters);
    }
    for (slot, (((id, map), was_tracked), val)) in slots
        .iter_mut()
        .zip(ids.into_iter().zip(maps).zip(tracked).zip(vals))
    {
        let val = val.detach();
        let val = if was_tracked {
            val.set_require_grad(true)
        } else {
            val
        };
        **slot = Param::from_mapped_value(id, val, map);
    }
    n
}

fn maxdiff(a: &Tensor<2>, b: &Tensor<2>) -> f32 {
    a.clone().sub(b.clone()).abs().max().into_scalar::<f32>()
}

/// Push every master AWAY from orthonormal, deterministically, before either
/// arm runs.
///
/// This is not decoration, it is the test. `SpectralLinear::new` initialises
/// `u`/`v` orthonormal (Householder QR), so at initialisation the retraction is
/// very nearly the identity: measured on mini-nano, the masters move by
/// 3.0e-7 over 3 Newton-Schulz iterations, and BOTH arms then agree to 0.0 -
/// bit-for-bit, on 16 factors. A differential test run on that input compares
/// two identity maps and passes even if one of them is `x -> x`. A row-wise
/// scale in [1.0, 1.35] breaks the orthogonality (it scales singular values
/// non-uniformly, so the polar factor is genuinely different from the input)
/// while staying pure arithmetic - both clones get the same perturbation, and
/// the retraction then has real work to do.
fn perturb_masters(model: &mut DormouseModel) {
    let mut slots = master_slots(model);
    for p in slots.iter_mut() {
        let (id, val, map) = Param::clone(p).consume();
        let [rows, _] = val.dims();
        let scales: Vec<f32> = (0..rows)
            .map(|i| 1.0 + 0.35 * (((i * 7) % 13) as f32 / 12.0))
            .collect();
        let scale = Tensor::<2>::from_data(TensorData::new(scales, [rows, 1]), &val.device());
        // `.detach().set_require_grad(true)` and not a bare `val.mul(scale)`:
        // on this backend an op with ONE untracked operand returns an
        // UNTRACKED tensor (measured here - the perturbation silently
        // untracked all 16 masters), which would make the retraction's own
        // tracking assertion vacuous.
        let perturbed = val.mul(scale).detach().set_require_grad(true);
        assert!(
            perturbed.is_require_grad(),
            "the perturbation must leave the master tracked"
        );
        **p = Param::from_mapped_value(id, perturbed, map);
    }
}

/// The two arms, on one real factor set, must agree.
///
/// TOLERANCE: 2e-5 absolute, on entries whose post-retraction magnitude is
/// ~1/sqrt(k) (the masters are orthonormal, so |entry| ~ 0.18 at k=32).
/// Measured on this backend: **0.0** - bit-for-bit, all 16 factors - because
/// the CPU/Flex path reduces the same [c,c]x[c,k] products in the same order
/// whether or not they are stacked. The bound is NOT written as 0.0 for two
/// reasons, and both were measured rather than assumed:
/// - the bound is derived from fp32 accumulation, not from the run: a length-c
///   dot product carries ~sqrt(c)*eps of noise (eps = 1.19e-7, c <= 128 here),
///   and the NS cubic amplifies a relative perturbation by |a|+|b|+|c| = 3.5
///   per iteration, so 3 iterations put the bound near 1e-5;
/// - this test does not run on CUDA, where a batched GEMM and a single GEMM
///   have different reduction trees and bit-identical agreement is not
///   available. A 0.0 assertion would be a claim about Flex that CUDA refutes.
///
/// What the bound does and does not catch, both demonstrated by breaking the
/// batched path on 2026-09-29 and restoring it:
/// - a stale slot in the write-back (`slice([0..1])` for every factor) turns
///   the test RED at 5.3e-1 / 5.7e-1 - the silent bug this gate exists for;
/// - a 5% change in the polar pre-scale (1.05 -> 1.00) leaves it GREEN at
///   1.8e-7, and that is correct rather than a weak gate: the polar factor is
///   invariant to positive scaling, so that input change is not an output
///   change. A tolerance cannot be "tight" against a quantity that does not
///   depend on what was changed.
#[test]
fn batched_matches_the_per_factor_path() {
    let dev = device();
    let model = DormouseModel::new(&mini_nano(), &dev);
    let (mut per_factor, mut batched) = (model.clone(), model.clone());
    perturb_masters(&mut per_factor);
    perturb_masters(&mut batched);
    let before = masters(&per_factor);
    assert!(before.len() >= 2, "mini-nano must own TSCT masters");

    per_factor.retract_tsct(ITERS);
    let n = batched_retract(&mut batched, ITERS);

    let a = masters(&per_factor);
    let b = masters(&batched);
    assert_eq!(n, before.len(), "both arms must see the same factor count");
    assert_eq!(a.len(), b.len());

    let (mut worst, mut worst_i) = (0.0f32, 0usize);
    let mut moved = 0.0f32;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            x.dims(),
            y.dims(),
            "factor {i}: the batched arm changed a shape"
        );
        let d = maxdiff(x, y);
        if d > worst {
            worst = d;
            worst_i = i;
        }
        moved = moved.max(maxdiff(&before[i], x));
    }
    println!(
        "retraction differential: {} factors, worst |per_factor - batched| = {worst:.3e} \
         at factor {worst_i} {:?}, max |before - after| = {moved:.3e}",
        a.len(),
        a[worst_i].dims()
    );

    // 1. The values agree. This is the whole point of the arm.
    assert!(
        worst < 2e-5,
        "batched retraction diverged from the per-factor path: worst {worst:.3e} at factor {worst_i} {:?}",
        a[worst_i].dims()
    );
    // 2. The comparison is not two no-ops agreeing. A retraction that changed
    // nothing by less than 1e-3 would make assertion 1 free.
    assert!(
        moved > 1e-3,
        "neither arm moved the masters by more than 1e-3 ({moved:.3e}) - the \
         comparison above would be vacuous"
    );
    // 3. Training state survives on BOTH arms (the ADR-0011 class: a master
    // silently frozen is a correct run that stops training).
    assert!(
        b.iter().all(|t| t.is_require_grad()),
        "the batched arm left a master untracked; burn-optim downgrades a \
         non-leaf to a frozen leaf"
    );
    // 4. The retraction is real: both arms land at the orthonormality the plan
    // calls good enough to keep the factor-quant forward engaged (per-entry
    // 1e-3 drift bound, fresh-init floor ~1.4e-4).
    let (ortho_pf, ortho_bt) = (per_factor.max_ortho(), batched.max_ortho());
    println!("max_ortho: per-factor {ortho_pf:.3e}, batched {ortho_bt:.3e}");
    assert!(
        ortho_bt < 1e-3,
        "batched retraction left the masters non-orthonormal: {ortho_bt:.3e}"
    );
    assert!(
        (ortho_pf - ortho_bt).abs() < 1e-4,
        "the two arms disagree about how orthonormal they are: {ortho_pf:.3e} vs {ortho_bt:.3e}"
    );
}

/// The PRODUCTION entry point (`DormouseModel::retract_tsct_batched`, the arm
/// `--retract-batched` selects) must produce the same masters as the
/// per-factor default, and it must be countable - the two arms are the same
/// mathematics, so nothing but the counter tells a reader which one ran.
#[test]
fn production_batched_arm_matches_and_counts() {
    use dormouse_core::probe;
    let dev = device();
    let model = DormouseModel::new(&mini_nano(), &dev);
    let (mut per_factor, mut batched) = (model.clone(), model.clone());
    perturb_masters(&mut per_factor);
    perturb_masters(&mut batched);

    probe::reset();
    per_factor.retract_tsct(ITERS);
    let (pf, bt) = (
        probe::count(probe::RETRACT_FACTOR),
        probe::count(probe::RETRACT_BATCHED),
    );
    batched.retract_tsct_batched(ITERS);
    let (pf2, bt2) = (
        probe::count(probe::RETRACT_FACTOR),
        probe::count(probe::RETRACT_BATCHED),
    );

    // 1. One call, one count, on the right arm.
    assert_eq!(
        (pf, pf2),
        (1, 1),
        "retract_tsct must count once per call on the per-factor arm"
    );
    assert_eq!(
        (bt, bt2),
        (0, 1),
        "retract_tsct_batched must count once per call on the grouped arm"
    );

    let a = masters(&per_factor);
    let b = masters(&batched);
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(&b)
        .enumerate()
        .map(|(i, (x, y))| {
            assert_eq!(x.dims(), y.dims(), "factor {i}: shape changed");
            maxdiff(x, y)
        })
        .fold(0.0f32, f32::max);
    println!("production arm vs per-factor: worst |d| = {worst:.3e}");
    // 2. Same bound as the library-level differential: same math, so the same
    // fp32 reduction-order noise.
    assert!(
        worst < 2e-5,
        "the production batched arm diverged from the per-factor path: {worst:.3e}"
    );
    // 3. Training state survives the round trip through the grouped call (a
    // non-leaf stored in a Param is a master burn-optim freezes silently).
    assert!(
        b.iter().all(|t| t.is_require_grad()),
        "the production batched arm left a master untracked"
    );
    // 4. The masters are still orthonormal afterwards, so the flag does not
    // quietly trade the factor-quant forward's precondition for speed.
    assert!(
        batched.max_ortho() < 1e-3,
        "max_ortho after the production batched arm: {:.3e}",
        batched.max_ortho()
    );
}
