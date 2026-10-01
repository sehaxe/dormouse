//! The zero-gradient rule: a step with no signal must not move the weights.
//!
//! This exists because the trainer's NaN firewall zeroes gradients on device,
//! and Muon+'s Newton-Schulz normalization would otherwise rescale the DECAYED
//! MOMENTUM back to unit Frobenius norm - turning a masked no-op step into a
//! full-magnitude step in the stale direction. That is how a run the firewall
//! was protecting still got its weights moved. The gate is a device-side
//! `mask_fill` on a float tensor, with no host synchronization, because the
//! project rule is to never build a numeric indicator from a bool tensor on
//! device and count on the host instead (ADR-0018 rule 2) - not because the
//! cast is broken, which was believed until ADR-0016 measured it correct on
//! both backends. The rule outlived the reason.
//!
//! The rule lives in [`burn_muon_plus::signal_mask`], which
//! `MuonPlus::step` applies to its update, so this drives that impl directly
//! (`MuonPlusConfig::build()` + an explicit state slot) rather than
//! `MuonPlusConfig::init()`. `init()` hands back a `ModuleOptimizer`,
//! whose `step` takes a whole `Module` and a `GradientsParams`, so a
//! tensor-in/tensor-out call against it does not compile - and the per-tensor
//! state is precisely what this test needs to thread by hand.
//!
//! The same rule has a second implementation, because the Muon update has a
//! second implementation: `dormouse-train`'s `HeadWiseMuon` re-derives the 2D
//! branch to slice the attention Q/K weights per head. It did not have this,
//! and `qk_heads` is always resolved, so the Q/K weights took a full step in
//! the stale direction on every masked step. It now calls this crate's
//! `signal_mask`, and `headwise_zero_gradient_does_not_move_the_parameter`
//! in `dormouse-train` pins it.
#![allow(deprecated)]
use burn::optim::Optimizer;
use burn::tensor::{Distribution, Tensor};
use burn_muon_plus::{MuonPlusConfig, NormDir};

fn dev() -> burn::tensor::Device {
    burn::tensor::Device::ndarray().autodiff()
}

fn opt() -> MuonPlusConfig {
    MuonPlusConfig::new()
        .with_norm_dir(Some(NormDir::ColRow))
        .with_weight_decay(0.0)
}

/// Step with an all-zero gradient leaves the parameter bit-identical.
#[test]
fn zero_gradient_does_not_move_the_parameter() {
    let device = dev();
    for dims in [[8usize, 4usize], [4, 4]] {
        let mut w: Tensor<2> = Tensor::random(dims, Distribution::Default, &device);
        let opt = opt().build();
        // SEED THE MOMENTUM WITH A REAL GRADIENT FIRST. Stepping twice with a
        // zero gradient does not test this rule: the momentum is `0.95·0 = 0`
        // and stays 0, so `orthogonalize(0)` is 0 and the step is a no-op with
        // or without the mask. That is why the version of this test that used a
        // zero gradient in both steps stayed GREEN with the mask deleted - it
        // was not evidence the rule was needed. (Verified: delete the
        // `mask_fill` in `signal_mask` and that version still passes.)
        //
        // The leak needs a LIVE momentum that the NS normalization rescales
        // back to unit Frobenius norm, which is exactly the poisoned case: a
        // masked step must not inherit a direction the firewall meant to
        // suppress.
        let seed = Tensor::<2>::random(dims, Distribution::Default, &device);
        let zero = Tensor::<2>::zeros(dims, &device);
        let (w1, s1) = opt.step(1e-3f64, w, seed, None);
        let after_seed = w1.clone().into_data();
        // Two masked steps: the momentum decays but stays non-zero, so both
        // would move the weight if the gate were not there.
        let (w2, s2) = opt.step(1e-3f64, w1, zero.clone(), s1);
        let (w3, _) = opt.step(1e-3f64, w2, zero, s2);
        w = w3;
        let after = w.clone().into_data();
        assert_eq!(
            after_seed.into_bytes(),
            after.into_bytes(),
            "a zero gradient moved the {dims:?} parameter - the stale momentum \
             leaked through the Newton-Schulz normalization"
        );
    }
}

/// And a non-zero gradient still moves it, so the rule is not just "Muon+ is
/// broken now".
#[test]
fn non_zero_gradient_still_moves_the_parameter() {
    let device = dev();
    let dims = [8usize, 4usize];
    let w: Tensor<2> = Tensor::random(dims, Distribution::Default, &device);
    let before = w.clone().into_data();
    let opt = opt().build();
    let g = Tensor::<2>::random(dims, Distribution::Default, &device);
    let (w, _) = opt.step(1e-3f64, w, g, None);
    assert_ne!(
        before.into_bytes(),
        w.into_data().into_bytes(),
        "a non-zero gradient did not move the parameter"
    );
}
