//! The production wiring: `KdaModule::forward_train_state` on the trainer's
//! backend (`Autodiff<CudaBare, BalancedCheckpointing>`) must run the fused
//! kernels. Before the gate fix it returned `None`, and the trainer silently
//! ran the tensor-ops chunk loop for every step — which computes the same
//! function, so only a launch counter can tell the two apart.
#![cfg(all(feature = "cuda", feature = "autodiff"))]

use burn::backend::AutodiffBackend;
use burn::tensor::{Device, Distribution, Tensor};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_gdn2::CudaBare;
use burn_kda::{DecayFn, KdaConfig, KdaModule};

type AdBal = Autodiff<CudaBare, BalancedCheckpointing>;

fn rel_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
    let a = a.into_data();
    let b = b.into_data();
    let mut max_abs = 0.0f32;
    let mut scale = 0.0f32;
    for (x, y) in a.bytes.chunks_exact(4).zip(b.bytes.chunks_exact(4)) {
        let x = f32::from_le_bytes(x.try_into().unwrap());
        let y = f32::from_le_bytes(y.try_into().unwrap());
        max_abs = max_abs.max((x - y).abs());
        scale = scale.max(x.abs()).max(y.abs());
    }
    max_abs / scale.max(1e-30)
}

#[test]
fn kda_forward_train_state_runs_the_fused_kernels() {
    let dev = Device::cuda(0);
    let cfg = KdaConfig {
        hidden_size: 128,
        num_heads: 4,
        head_dim: 32,
        num_v_heads: Some(4),
        use_short_conv: false,
        decay_fn: DecayFn::Sigmoid,
        chunk_size: 16,
        ..Default::default()
    };
    let km = KdaModule::new(&cfg, 0.9, &dev);
    let bare = Tensor::<3>::random([2, 64, 128], Distribution::Normal(0.0, 1.0), &dev);
    let x = Tensor::from_primitive::<AdBal>(<AdBal as AutodiffBackend>::from_inner(
        bare.try_into_primitive::<CudaBare>().unwrap(),
    ))
    .require_grad();

    burn_gdn2::reset_fused_calls();
    let (y, s) = km.forward_train_state::<AdBal>(x.clone(), None);
    let (fwd, _) = burn_gdn2::fused_calls();
    assert!(
        fwd > 0,
        "KDA ran the tensor-ops chunk path: the fused gate is closed"
    );

    let loss = y.clone().powf_scalar(2.0).sum() + s.clone().powf_scalar(2.0).sum();
    let grads = loss.backward();
    let (_, bwd) = burn_gdn2::fused_calls();
    assert!(bwd > 0, "the fused adjoint kernel never launched");
    assert!(
        x.grad(&grads).unwrap().abs().max().into_scalar::<f32>() > 0.0,
        "no gradient reached the input"
    );

    // The numbers: the chunked WY form (fused kernels) against the exact
    // per-token recurrence. Same function, different f32 reassociation.
    let mut st = None;
    let y_ref = km.forward_recurrent(x, &mut st, true);
    let err = rel_diff(y, y_ref);
    println!("KDA fused chunk vs per-token reference: rel_diff = {err:.3e}");
    assert!(err < 1e-3, "chunked fused drift vs reference: {err:.3e}");
}
