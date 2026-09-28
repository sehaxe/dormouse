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
//! The rule lives in `impl Optimizer for MuonPlus::step`, so this drives that
//! impl directly (`MuonPlusConfig::build()` + an explicit state slot) rather
//! than `MuonPlusConfig::init()`. `init()` hands back a `ModuleOptimizer`,
//! whose `step` takes a whole `Module` and a `GradientsParams`, so a
//! tensor-in/tensor-out call against it does not compile - and the per-tensor
//! state is precisely what this test needs to thread by hand.
#![allow(deprecated)]
use burn::optim::Optimizer;
use burn::tensor::{Distribution, Tensor};
use burn_muon_plus::{MuonPlusConfig, MuonPlusState, NormDir};

fn dev() -> burn::tensor::Device {
    burn::tensor::Device::ndarray().autodiff()
}

fn opt() -> MuonPlusConfig {
    MuonPlusConfig::new()
        .with_norm_dir(Some(NormDir::ColRow))
        .with_weight_decay(0.0)
}

/// One step with an all-zero gradient leaves the parameter bit-identical.
#[test]
fn zero_gradient_does_not_move_the_parameter() {
    let device = dev();
    for dims in [[8usize, 4usize], [4, 4]] {
        let mut w: Tensor<2> = Tensor::random(dims, Distribution::Default, &device);
        let before = w.clone().into_data();
        let opt = opt().build();
        // Step twice: the FIRST step seeds the momentum, the SECOND is the one
        // that must be a no-op, because with a live momentum the stale
        // direction is what used to leak through.
        let g = Tensor::<2>::zeros(dims, &device);
        let mut state: Option<MuonPlusState<2>> = None;
        let (w1, s1) = opt.step(1e-3f64, w, g.clone(), state.take());
        let (w2, _) = opt.step(1e-3f64, w1, g.clone(), s1);
        w = w2;
        let after = w.clone().into_data();
        assert_eq!(
            before.into_bytes(),
            after.into_bytes(),
            "a zero gradient moved the {dims:?} parameter - the momentum leaked"
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
