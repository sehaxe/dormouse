//! The zero-gradient rule: a step with no signal must not move the weights.
//!
//! This exists because the trainer's NaN firewall zeroes gradients on device,
//! and Muon+'s Newton-Schulz normalization would otherwise rescale the DECAYED
//! MOMENTUM back to unit Frobenius norm - turning a masked no-op step into a
//! full-magnitude step in the stale direction. That is how a run the firewall
//! was protecting still got its weights moved. The gate is a device-side
//! `mask_fill` on a float tensor, with no host synchronization, because the
//! `Bool -> float` cast is broken on this backend (returns 0.0 for `true`).
#![allow(deprecated)]
use burn::optim::Optimizer;
use burn::tensor::{Distribution, Tensor};
use burn_muon_plus::{MuonPlus, MuonPlusConfig, NormDir};

fn dev() -> burn::tensor::Device {
    burn::tensor::Device::ndarray().autodiff()
}

/// One step with an all-zero gradient leaves the parameter bit-identical.
#[test]
fn zero_gradient_does_not_move_the_parameter() {
    let device = dev();
    for dims in [[8usize, 4usize], [4, 4]] {
        let mut w: Tensor<2> = Tensor::random(dims, Distribution::Default, &device);
        let before = w.clone().into_data();
        let mut opt = MuonPlusConfig::new()
            .with_norm_dir(Some(NormDir::ColRow))
            .with_weight_decay(0.0)
            .init();
        // Step twice: the FIRST step seeds the momentum, the SECOND is the one
        // that must be a no-op, because with a live momentum the stale
        // direction is what used to leak through.
        let g = Tensor::<2>::zeros(dims, &device);
        w = opt.step(1e-3f64, w, g.clone());
        w = opt.step(1e-3f64, w, g.clone());
        let after = w.clone().into_data();
        assert_eq!(
            before.to_bytes::<f32>(),
            after.to_bytes::<f32>(),
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
    let mut w: Tensor<2> = Tensor::random(dims, Distribution::Default, &device);
    let before = w.clone().into_data();
    let mut opt = MuonPlusConfig::new()
        .with_norm_dir(Some(NormDir::ColRow))
        .with_weight_decay(0.0)
        .init();
    let g = Tensor::<2>::random(dims, Distribution::Default, &device);
    w = opt.step(1e-3f64, w, g);
    assert_ne!(
        before.to_bytes::<f32>(),
        w.into_data().to_bytes::<f32>(),
        "a non-zero gradient did not move the parameter"
    );
}
