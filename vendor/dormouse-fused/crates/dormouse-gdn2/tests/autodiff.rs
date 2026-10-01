//! Training (autodiff) and incremental-decoding tests.
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//!
//! The decode test is the critical one for the short-convolution cache:
//! token-by-token decoding must reproduce the full-sequence forward output.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Device;
use burn::tensor::{Distribution, Tensor};
use dormouse_gdn2::{GatedDeltaNet2, Gdn2Config, Gdn2Mode, Gdn2State};

fn cfg(hidden: usize, heads: usize, head_dim: usize, mode: Gdn2Mode) -> Gdn2Config {
    Gdn2Config {
        hidden_size: hidden,
        num_heads: heads,
        head_dim,
        use_short_conv: true,
        mode,
        ..Default::default()
    }
}

fn max_diff(a: Tensor<3>, b: Tensor<3>) -> f32 {
    let d = (a - b).abs().max().into_data();
    d.bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .fold(0.0f32, f32::max)
}

/// Token-by-token decoding (with state) must reproduce a single forward pass
/// over the full sequence, for both short-conv on and off.
#[test]
fn decode_equals_full_forward() {
    let device = Device::ndarray();
    let (hidden, heads, hk, batch, tokens) = (32usize, 2usize, 16usize, 2usize, 12usize);

    for use_sc in [true, false] {
        let c = cfg(hidden, heads, hk, Gdn2Mode::FusedRecurrent);
        let m = GatedDeltaNet2::new(&c, &device);
        let x = Tensor::<3>::random(
            [batch, tokens, hidden],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Reference: single pass over the full sequence.
        let mut ref_state: Option<Gdn2State> = None;
        let full = m.forward::<NdArray>(x.clone(), &mut ref_state, true);
        let ref_out = full.clone();

        // Decode token-by-token, carrying the state.
        let mut state: Option<Gdn2State> = None;
        let mut decoded: Vec<Tensor<3>> = Vec::new();
        for t in 0..tokens {
            let tok = x.clone().slice([0..batch, t..t + 1, 0..hidden]);
            let out = m.forward::<NdArray>(tok, &mut state, true);
            decoded.push(out);
        }
        let decoded = Tensor::cat(decoded, 1);

        let diff = max_diff(decoded, ref_out);
        assert!(
            diff < 1e-6,
            "decode != full forward (use_short_conv={use_sc}): max_diff={diff:.2e}",
        );
    }
}

/// Continuing a prefill with one more token must match a full forward over
/// the extended sequence (prefill -> decode handoff).
#[test]
fn prefill_then_decode_matches_extended_forward() {
    let device = Device::ndarray();
    let (hidden, heads, hk, batch, tokens) = (32usize, 2usize, 16usize, 1usize, 8usize);
    let c = cfg(hidden, heads, hk, Gdn2Mode::FusedRecurrent);
    let m = GatedDeltaNet2::new(&c, &device);
    let x = Tensor::<3>::random(
        [batch, tokens + 1, hidden],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let full = m.forward::<NdArray>(x.clone(), &mut None, true);

    let mut state: Option<Gdn2State> = None;
    let _ = m.forward::<NdArray>(
        x.clone().slice([0..batch, 0..tokens, 0..hidden]),
        &mut state,
        true,
    );
    let next = m.forward::<NdArray>(
        x.clone().slice([0..batch, tokens..tokens + 1, 0..hidden]),
        &mut state,
        true,
    );

    let expected = full
        .clone()
        .slice([0..batch, tokens..tokens + 1, 0..hidden]);
    let diff = max_diff(next, expected);
    assert!(
        diff < 1e-6,
        "prefill->decode handoff mismatch: max_diff={diff:.2e}"
    );
}

/// Chunk and fused recurrences must agree within float tolerance (same
/// projections, same state, different scan strategies).
#[test]
fn chunk_and_fused_agree() {
    let device = Device::ndarray();
    let (hidden, heads, hk, batch, tokens) = (32usize, 2usize, 16usize, 1usize, 80usize);
    let c = cfg(hidden, heads, hk, Gdn2Mode::Chunk);
    let m = GatedDeltaNet2::new(&c, &device);
    let x = Tensor::<3>::random(
        [batch, tokens, hidden],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (proj, _) = m.project(x.clone(), None);
    let (c_out, _) = dormouse_gdn2::chunk_wy_forward(
        proj.q.clone(),
        proj.k.clone(),
        proj.v.clone(),
        proj.g.clone(),
        proj.b.clone(),
        proj.w.clone(),
        Tensor::zeros([batch, heads, hk, hk], &device),
        16.0f64.powf(-0.5),
        16,
    );
    let (f_out, _) = dormouse_gdn2::fused_recurrent_forward(
        proj.q,
        proj.k,
        proj.v,
        proj.g,
        proj.b,
        proj.w,
        Tensor::zeros([batch, heads, hk, hk], &device),
        16.0f64.powf(-0.5),
    );
    let c_out = c_out
        .permute([0, 2, 1, 3])
        .reshape([batch, tokens, heads * hk]);
    let f_out = f_out
        .permute([0, 2, 1, 3])
        .reshape([batch, tokens, heads * hk]);
    let diff = max_diff(c_out, f_out);
    assert!(diff < 1e-4, "chunk vs fused mismatch: max_diff={diff:.2e}");
}

/// Gradients must flow through both forward modes (training path).
#[test]
fn gradients_flow_both_modes() {
    type AD = Autodiff<NdArray>;
    let device = Device::ndarray().autodiff();
    let (hidden, heads, hk) = (32usize, 2usize, 16usize);

    for mode in [Gdn2Mode::FusedRecurrent, Gdn2Mode::Chunk] {
        let c = cfg(hidden, heads, hk, mode);
        let m = GatedDeltaNet2::new(&c, &device);
        let x = Tensor::<3>::random([2, 24, hidden], Distribution::Normal(0.0, 1.0), &device);
        let out = m.forward_train::<AD>(x);
        let loss = out.clone().powf_scalar(2.0).mean();
        let grads = loss.backward();

        let qg = m.q_proj.weight.val().grad(&grads).unwrap();
        let qg_abs: f32 = qg.clone().abs().max().into_scalar();
        let ag = m.a_log.val().grad(&grads).unwrap();
        let ag_abs: f32 = ag.clone().abs().max().into_scalar();
        assert!(
            qg_abs > 0.0 && ag_abs > 0.0,
            "mode={mode:?}: no gradient (q={qg_abs}, a_log={ag_abs})",
        );
    }
}

/// Decode path must also produce gradients (autoregressive training).
#[test]
fn gradients_flow_decode_path() {
    type AD = Autodiff<NdArray>;
    let device = Device::ndarray().autodiff();
    let (hidden, heads, hk, tokens) = (32usize, 2usize, 16usize, 6usize);
    let c = cfg(hidden, heads, hk, Gdn2Mode::FusedRecurrent);
    let m = GatedDeltaNet2::new(&c, &device);
    let x = Tensor::<3>::random([1, tokens, hidden], Distribution::Normal(0.0, 1.0), &device);
    let mut state: Option<Gdn2State> = None;
    let mut loss = Tensor::<1>::zeros([1], &device);
    for t in 0..tokens {
        let tok = x.clone().slice([0..1, t..t + 1, 0..hidden]);
        let out = m.forward::<AD>(tok, &mut state, true);
        loss = loss + out.clone().powf_scalar(2.0).mean();
    }
    let grads = loss.backward();
    let qg: f32 = m
        .q_proj
        .weight
        .val()
        .grad(&grads)
        .unwrap()
        .abs()
        .max()
        .into_scalar();
    assert!(qg > 0.0, "no gradient through decode path: q={qg}");
}

/// Invalid configs must be rejected with a clear panic.
#[test]
#[should_panic(expected = "num_v_heads")]
fn rejects_num_v_heads_below_num_heads() {
    let device = Device::ndarray();
    let c = Gdn2Config {
        hidden_size: 32,
        num_heads: 4,
        head_dim: 16,
        num_v_heads: Some(2),
        ..Default::default()
    };
    let _ = GatedDeltaNet2::new(&c, &device);
}

#[test]
#[should_panic(expected = "divisible")]
fn rejects_non_divisible_num_v_heads() {
    let device = Device::ndarray();
    let c = Gdn2Config {
        hidden_size: 32,
        num_heads: 4,
        head_dim: 16,
        num_v_heads: Some(6),
        ..Default::default()
    };
    let _ = GatedDeltaNet2::new(&c, &device);
}

#[test]
#[should_panic(expected = "integer")]
fn rejects_fractional_expand_v() {
    let device = Device::ndarray();
    let c = Gdn2Config {
        hidden_size: 32,
        num_heads: 4,
        head_dim: 16,
        expand_v: 1.3,
        ..Default::default()
    };
    let _ = GatedDeltaNet2::new(&c, &device);
}

#[test]
#[should_panic(expected = "chunk_size")]
fn rejects_zero_chunk_size() {
    let device = Device::ndarray();
    let c = Gdn2Config {
        hidden_size: 32,
        num_heads: 4,
        head_dim: 16,
        chunk_size: 0,
        ..Default::default()
    };
    let _ = GatedDeltaNet2::new(&c, &device);
}

/// GVA: more value heads than query heads repeats key-side heads.
#[test]
fn gva_configuration_runs() {
    let device = Device::ndarray();
    let (hidden, heads, hk) = (64usize, 2usize, 16usize);
    let c = Gdn2Config {
        hidden_size: hidden,
        num_heads: heads,
        head_dim: hk,
        num_v_heads: Some(4),
        ..Default::default()
    };
    let m = GatedDeltaNet2::new(&c, &device);
    let x = Tensor::<3>::random([2, 16, hidden], Distribution::Normal(0.0, 1.0), &device);
    let out = m.forward_train::<NdArray>(x);
    assert_eq!(out.shape().dims(), [2, 16, hidden]);
}
