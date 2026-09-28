#![cfg(all(feature = "cuda", feature = "autodiff"))]
#![allow(deprecated)]

//! The GPU assertion this library was advertising and never running, at the
//! shape dormouse actually trains.
//!
//!   tools/gpu-gate.sh     (the gate — see there for why it is local, not CI)
//!
//! `configs/small.toml`: `d_model 768`, `n_heads 12`, `head_dim 64`,
//! `max_seq_len 512`; `AGENTS.md`: batch 10 is the VRAM-validated shape for
//! that preset. `Gdn2Config::chunk_size` defaults to 64, "matching reference".
//! So the shape is B=10, H=12, T=512, K=V=64, 8 chunks of 64.
//!
//! Why the shape is the assertion, and not the launch counters alone: every
//! existing fused test in this library runs at a toy shape — T=64, 80, 128,
//! 256 — and the two that are long (T=2048, 4096) are B=1, 2. The trainer's
//! own configuration was never run through a fused kernel by any job that
//! executes.
//!
//! Why the launch counters, not the numbers: a fallback computes the SAME
//! function, so a fused/tensor disagreement cannot detect that the fused path
//! was never taken. That is not hypothetical — the dispatch gate for
//! `Autodiff<CudaBare, BalancedCheckpointing>` was closed for a year while
//! every test that "covered" the fused path went through the tensor-ops
//! fallback and reported PASS. `fused_calls()` is incremented inside the kernel
//! launch, never on the decision to consider it, so `> 0` means it ran.
//!
//! It lives in the facade rather than in `crates/burn-gdn2/tests/` because that
//! crate is under concurrent edit; it calls only burn-gdn2's public API
//! (`chunk_dispatch`, `fused_calls`, `reset_fused_calls`), so it is a
//! consumer of the crate, not a second copy of its test. Delete it when the
//! same case lands in `fused_chunk_verify.rs`'s shape table.
//!
//! Tolerances are pre-registered from the numbers this tree already argues
//! for, NOT chosen to make this pass (`research/2026-09-27-oracle-audit.md`
//! §3, checklist item 3):
//!   - forward/state 1e-3 — fp32 reassociation across chunk boundaries;
//!     `fused_chunk_verify.rs:124` uses 1e-3 for the same K=V=64 family at
//!     T=2048, i.e. 4x more boundaries than T=512.
//!   - grads 1e-2, and 1e-1 for `k` only — k's chunk-boundary rows divide by
//!     E≈glast (~1e-3), which amplifies fp32 noise to a few percent and is
//!     data-dependent. `fused_chunk_verify.rs:201-204` states this exact
//!     reason and carries the same exception.
//! If this test goes red, that is a finding about the kernel, not a tolerance
//! to be widened.

use burn::backend::AutodiffBackend;
use burn::tensor::{Device, Distribution, Tensor};
use burn_fused::burn_autodiff::Autodiff;
use burn_fused::burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_fused::burn_gdn2::{
    backend_matches, chunk_dispatch, chunk_wy_forward, fused_calls, reset_fused_calls, CudaBare,
    Fused,
};

/// The trainer's backend: burn-gdn2's fused op is reached from an autodiff
/// graph over bare CUDA with balanced checkpointing. dormouse trains on this
/// exact type; a test on `CudaBare` alone cannot catch a balanced-checkpointing
/// bug (ADR-0020 checklist item 7).
type AdBal = Autodiff<CudaBare, BalancedCheckpointing>;

/// Lift a bare CUDA tensor onto `Autodiff<CudaBare, BalancedCheckpointing>` —
/// what a training graph hands the module. The lift is the point: a tensor
/// that never went through it carries a DISABLED autodiff context, which is
/// what every fused test in this library used to build.
macro_rules! lift {
    ($t:expr) => {{
        let bare = $t
            .clone()
            .try_into_primitive::<CudaBare>()
            .expect("bare cuda tensor");
        let node = <AdBal as AutodiffBackend>::from_inner(bare);
        Tensor::from_primitive::<AdBal>(node).require_grad()
    }};
}

/// Max |a-b| / max|a| — the measure the crate's other fused tests use.
fn rel_diff(a: Tensor<4>, b: Tensor<4>) -> f32 {
    let a = a.into_data();
    let b = b.into_data();
    let (mut max_abs, mut scale) = (0.0f32, 0.0f32);
    for (x, y) in a.bytes.chunks_exact(4).zip(b.bytes.chunks_exact(4)) {
        let x = f32::from_le_bytes(x.try_into().unwrap());
        let y = f32::from_le_bytes(y.try_into().unwrap());
        max_abs = max_abs.max((x - y).abs());
        scale = scale.max(x.abs()).max(y.abs());
    }
    max_abs / scale.max(1e-30)
}

fn finite(t: &Tensor<4>) -> bool {
    t.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .all(|x| x.is_finite())
}

#[test]
fn gated_delta_chunk_path_runs_at_the_production_shape() {
    let device = Device::cuda(0);
    device.seed(42); // a gate that flakes is worse than no gate
    let (batch, heads, time, k_dim, v_dim, chunk) = (10usize, 12usize, 512usize, 64, 64, 64);
    let scale = k_dim as f64.powf(-0.5);
    println!(
        "production shape: B={batch} H={heads} T={time} k={k_dim} v={v_dim} chunk={chunk} \
         ({} chunks), backend {}",
        time / chunk,
        <AdBal as burn::backend::Backend>::name(&device),
    );

    // The gate must be open for the backend we ship on, or the rest of this
    // test is measuring the fallback and calling it a pass.
    assert!(
        backend_matches::<AdBal>(),
        "the trainer's backend fell off the fused path"
    );

    let inp = [
        lift!(Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(0.0, 0.3),
            &device
        )),
        lift!(Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(0.0, 0.3),
            &device
        )),
        lift!(Tensor::<4>::random(
            [batch, heads, time, v_dim],
            Distribution::Normal(0.0, 0.3),
            &device
        )),
        // negative gates (decay), like the model produces
        lift!(Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(-0.5, 0.2),
            &device
        )),
        lift!(Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(0.0, 0.3),
            &device
        )),
        lift!(Tensor::<4>::random(
            [batch, heads, time, v_dim],
            Distribution::Normal(0.0, 0.3),
            &device
        )),
        lift!(Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(0.0, 0.3),
            &device
        )),
    ];
    // The state is [B, H, K, V] and starts at zero, as it does in training.
    let state = lift!(Tensor::<4>::zeros(
        [batch, heads, k_dim, v_dim],
        &device
    ));
    let inputs = [
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        state,
    ];

    reset_fused_calls();
    let (out, st) = match chunk_dispatch::<AdBal>(
        inputs[0].clone(),
        inputs[1].clone(),
        inputs[2].clone(),
        inputs[3].clone(),
        inputs[4].clone(),
        inputs[5].clone(),
        inputs[6].clone(),
        scale,
        chunk,
    ) {
        Fused::Fused(r) => r,
        Fused::Fallback(why) => panic!("the fused path did not run: {why:?}"),
    };
    let (fwd, _) = fused_calls();
    println!("fused forward kernel launches: {fwd}");
    assert!(
        fwd > 0,
        "the fused forward kernel never launched: the gate is closed again"
    );

    // Backward through the checkpointed graph: the fused ADJOINT kernel, not
    // the tensor-ops adjoint. Without this the backward half of the claim is
    // unasserted, which is how the dead gate survived a test suite.
    let loss = out.clone().powf_scalar(2.0).sum() + st.clone().powf_scalar(2.0).sum();
    let grads = loss.backward();
    let (_, bwd) = fused_calls();
    println!("fused adjoint kernel launches: {bwd}");
    assert!(
        bwd > 0,
        "the fused adjoint kernel never launched: the gate is closed again"
    );

    for (i, t) in inputs.iter().enumerate() {
        let g = t
            .grad(&grads)
            .unwrap_or_else(|| panic!("input {i} got no gradient"));
        assert!(finite(&g), "input {i} got a non-finite gradient");
        assert!(
            g.abs().max().into_scalar::<f32>() > 0.0,
            "input {i} got a zero gradient"
        );
    }

    // And the numbers are the chunked WY form: the fused kernels against the
    // tensor-ops chunk path, which is what the trainer ran while the gate was
    // closed. Different algorithm (f32 reassociation over chunks), same
    // function.
    let (plain_out, plain_st) = chunk_wy_forward(
        inputs[0].clone(),
        inputs[1].clone(),
        inputs[2].clone(),
        inputs[3].clone(),
        inputs[4].clone(),
        inputs[5].clone(),
        inputs[6].clone(),
        scale,
        chunk,
    );
    let d_out = rel_diff(out.clone(), plain_out.clone());
    let d_state = rel_diff(st.clone(), plain_st.clone());
    println!("fused vs tensor-ops chunk path: out {d_out:.3e}, state {d_state:.3e}");
    assert!(d_out < 1e-3, "forward drift vs tensor path: {d_out:.3e}");
    assert!(d_state < 1e-3, "state drift vs tensor path: {d_state:.3e}");

    // Same comparison for the gradients: the fused adjoint against the
    // tensor-ops adjoint of the identical loss, off the forward already
    // computed above.
    let ref_grads = (plain_out.powf_scalar(2.0).sum() + plain_st.powf_scalar(2.0).sum()).backward();
    for (name, t) in ["q", "k", "v", "g", "b", "w", "s"].iter().zip(inputs.iter()) {
        let rel = rel_diff(
            t.grad(&grads).unwrap().clone(),
            t.grad(&ref_grads).unwrap().clone(),
        );
        println!("{name}: grad rel={rel:.2e}");
        let tol = if *name == "k" { 1e-1 } else { 1e-2 };
        assert!(rel < tol, "{name}: grad mismatch rel={rel:.2e} (tol {tol:.0e})");
    }
}
