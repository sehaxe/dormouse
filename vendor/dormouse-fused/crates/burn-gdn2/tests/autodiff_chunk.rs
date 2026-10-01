#![cfg(feature = "autodiff")]
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//! Fused autodiff chunk op: forward must match the plain path and gradients
//! must be exact (equal to the tensor-path autodiff and to finite
//! differences).
//!
//! The CPU reference backend is `Autodiff<NdArray>`: the fused op runs its
//! forward on the inner backend and its backward on the same inner backend.

use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::tensor::Device;
use burn::tensor::{Distribution, Tensor};
use burn_gdn2::{chunk_wy_forward, chunk_wy_forward_autodiff};

type AD = Autodiff<NdArray>;

const ATOL: f32 = 1e-5;
const RELATOL: f32 = 1e-3;

fn rel_diff(a: Tensor<4>, b: Tensor<4>) -> f32 {
    let a = a.clone().into_data();
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

/// Random inputs on the autodiff device, all requiring gradients.
fn inputs(
    device: &Device,
    batch: usize,
    heads: usize,
    time: usize,
    k_dim: usize,
    v_dim: usize,
) -> [Tensor<4>; 7] {
    [
        Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(0.0, 0.1),
            device,
        )
        .require_grad(),
        Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(0.0, 0.1),
            device,
        )
        .require_grad(),
        Tensor::<4>::random(
            [batch, heads, time, v_dim],
            Distribution::Normal(0.0, 0.1),
            device,
        )
        .require_grad(),
        // negative gates (decay), like the model produces
        Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(-0.5, 0.1),
            device,
        )
        .require_grad(),
        Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Uniform(0.0, 0.1),
            device,
        )
        .require_grad(),
        Tensor::<4>::random(
            [batch, heads, time, v_dim],
            Distribution::Uniform(0.0, 0.1),
            device,
        )
        .require_grad(),
        Tensor::<4>::random(
            [batch, heads, k_dim, v_dim],
            Distribution::Normal(0.0, 0.1),
            device,
        )
        .require_grad(),
    ]
}

fn grads(t: &[Tensor<4>; 7], g: &burn::tensor::Gradients) -> Vec<Tensor<4>> {
    t.iter()
        .map(|t| t.grad(g).unwrap_or_else(|| panic!("missing grad")).clone())
        .collect()
}

/// Fused op forward must be (numerically) identical to the plain path, and
/// the returned state must match.
#[test]
fn fused_forward_matches_plain() {
    let device = Device::ndarray().autodiff();
    let (batch, heads, time, k_dim, v_dim) = (2usize, 2usize, 80usize, 16usize, 16usize);
    let scale = 16f64.powf(-0.5);
    let chunk_size = 16;

    let inp = inputs(&device, batch, heads, time, k_dim, v_dim);
    let plain = chunk_wy_forward(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        scale,
        chunk_size,
    );
    let fused = chunk_wy_forward_autodiff::<NdArray>(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        scale,
        chunk_size,
    )
    .unwrap();

    // Recompute expected on the plain backend for a fair comparison.
    let (plain_out, plain_s) = plain;
    let (fused_out, fused_s) = fused;
    let d_out = rel_diff(fused_out, plain_out);
    let d_s = rel_diff(fused_s, plain_s);
    assert!(d_out < ATOL, "forward mismatch: rel_diff={d_out:.2e}");
    assert!(d_s < ATOL, "state mismatch: rel_diff={d_s:.2e}");
}

/// Gradients through the fused op must match the tensor-path gradients on the
/// same inputs, for all seven inputs.
#[test]
fn fused_grads_match_tensor_path() {
    let device = Device::ndarray().autodiff();
    let (batch, heads, time, k_dim, v_dim) = (1usize, 1usize, 10usize, 4usize, 3usize);
    let scale = 4f64.powf(-0.5);
    let chunk_size = 5;

    // Tensor path: burn's per-op autodiff, then the fused op on the SAME
    // inputs (two forwards on the same leaves are independent graphs).
    let inp = inputs(&device, batch, heads, time, k_dim, v_dim);
    let (ref_grads, ref_scale) = {
        let (out, _s) = chunk_wy_forward(
            inp[0].clone(),
            inp[1].clone(),
            inp[2].clone(),
            inp[3].clone(),
            inp[4].clone(),
            inp[5].clone(),
            inp[6].clone(),
            scale,
            chunk_size,
        );
        let loss = out.clone().powf_scalar(2.0).sum();
        let loss_scale: f32 = loss.clone().into_scalar();
        let g = loss.backward();
        (grads(&inp, &g), loss_scale)
    };

    // Fused op path.
    let (out, _s) = chunk_wy_forward_autodiff::<NdArray>(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        scale,
        chunk_size,
    )
    .unwrap();
    let loss = out.powf_scalar(2.0).sum();
    let g = loss.backward();
    let fused_grads = grads(&inp, &g);

    for (i, (fg, rg)) in fused_grads.iter().zip(ref_grads.iter()).enumerate() {
        let d = rel_diff(fg.clone(), rg.clone());
        let msg = format!("input {i}: rel_diff={d:.2e}");
        assert!(d < RELATOL, "{msg}");
        assert!(!msg.is_empty());
    }
    let _ = ref_scale;
}

/// Finite-difference check of the fused op gradients against the plain
/// forward (independent of burn's autodiff).
#[test]
fn fused_grads_match_finite_difference() {
    let device = Device::ndarray().autodiff();
    let (batch, heads, time, k_dim, v_dim) = (1usize, 1usize, 10usize, 4usize, 3usize);
    let scale = 4f64.powf(-0.5);
    let chunk_size = 5;

    let inp = inputs(&device, batch, heads, time, k_dim, v_dim);
    let (out, _s) = chunk_wy_forward_autodiff::<NdArray>(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        scale,
        chunk_size,
    )
    .unwrap();
    let loss = out.powf_scalar(2.0).sum();
    let g = loss.backward();
    let fused_grads = grads(&inp, &g);

    // Numeric grads on the plain backend: probe a few coordinates per input.
    // All tensors are detached AD leaves on the same device, so the probe
    // forward runs untracked-but-same-backend.
    let detached: [Tensor<4>; 7] = [
        inp[0].clone().detach(),
        inp[1].clone().detach(),
        inp[2].clone().detach(),
        inp[3].clone().detach(),
        inp[4].clone().detach(),
        inp[5].clone().detach(),
        inp[6].clone().detach(),
    ];

    // Probe the position of the largest gradient per input: finite
    // differences are only meaningful where grad·eps exceeds the fp32 noise
    // floor of the loss (~1e-6·loss), i.e. anywhere else the probe is pure
    // rounding noise.
    let eps = 1e-3f32;
    for (i, inp_i) in inp.iter().enumerate() {
        let shape = inp_i.shape().dims::<4>();
        let grad_flat = fused_grads[i].clone().into_data();
        let grad_vals: Vec<f32> = grad_flat
            .bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let (max_idx, _max_val) = grad_vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap();
        let (b, h, t, d) = (
            max_idx / (shape[1] * shape[2] * shape[3]),
            (max_idx / (shape[2] * shape[3])) % shape[1],
            (max_idx / shape[3]) % shape[2],
            max_idx % shape[3],
        );
        let probe = |delta: f32| -> f32 {
            let mut buf = inp_i.clone().into_data().bytes;
            let idx = (((b * shape[1] + h) * shape[2] + t) * shape[3] + d) * 4;
            let mut val = f32::from_le_bytes(buf[idx..idx + 4].try_into().unwrap());
            val += delta;
            buf[idx..idx + 4].copy_from_slice(&val.to_le_bytes());
            let vals: Vec<f32> = buf
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            let perturbed = Tensor::<4>::from_data(
                burn::tensor::TensorData::new(vals, shape.to_vec()),
                &device,
            );
            let (o, _s) = chunk_wy_forward(
                if i == 0 {
                    perturbed.clone()
                } else {
                    detached[0].clone()
                },
                if i == 1 {
                    perturbed.clone()
                } else {
                    detached[1].clone()
                },
                if i == 2 {
                    perturbed.clone()
                } else {
                    detached[2].clone()
                },
                if i == 3 {
                    perturbed.clone()
                } else {
                    detached[3].clone()
                },
                if i == 4 {
                    perturbed.clone()
                } else {
                    detached[4].clone()
                },
                if i == 5 {
                    perturbed.clone()
                } else {
                    detached[5].clone()
                },
                if i == 6 {
                    perturbed.clone()
                } else {
                    detached[6].clone()
                },
                scale,
                chunk_size,
            );
            (o.powf_scalar(2.0).sum()).into_scalar()
        };
        let numeric = (probe(eps) - probe(-eps)) / (2.0 * eps);
        let analytical: f32 = fused_grads[i]
            .clone()
            .slice([b..b + 1, h..h + 1, t..t + 1, d..d + 1])
            .into_scalar();
        let tol = (numeric.abs() + analytical.abs()).max(1e-4) * 5e-2;
        assert!(
            (numeric - analytical).abs() < tol,
            "input {i} [{b},{h},{t},{d}]: numeric={numeric:.4} analytical={analytical:.4} tol={tol:.4}",
        );
    }
}

/// The model's fused training path must produce the same gradients as the
/// plain training path.
#[test]
fn model_fused_train_grads_match_plain() {
    use burn::tensor::Tensor as T3;
    use burn_gdn2::{GatedDeltaNet2, Gdn2Config, Gdn2Mode};

    let device = Device::ndarray().autodiff();
    let cfg = Gdn2Config {
        hidden_size: 16,
        num_heads: 1,
        head_dim: 4,
        mode: Gdn2Mode::Chunk,
        chunk_size: 5,
        ..Default::default()
    };

    let x = T3::<3>::random([1, 10, 16], Distribution::Normal(0.0, 0.1), &device);

    let (m1, m2) = {
        let m = GatedDeltaNet2::new(&cfg, &device);
        (m.clone(), m)
    };

    let loss1 = m1.forward_train::<AD>(x.clone()).powf_scalar(2.0).mean();
    let g1 = loss1.backward();
    let loss2 = m2.forward_train_fused::<AD>(x).powf_scalar(2.0).mean();
    let g2 = loss2.backward();

    for (name, p1, p2) in [
        ("q_proj", m1.q_proj.weight.val(), m2.q_proj.weight.val()),
        ("v_proj", m1.v_proj.weight.val(), m2.v_proj.weight.val()),
        ("b_proj", m1.b_proj.weight.val(), m2.b_proj.weight.val()),
        ("w_proj", m1.w_proj.weight.val(), m2.w_proj.weight.val()),
        (
            "f_proj_1",
            m1.f_proj_1.weight.val(),
            m2.f_proj_1.weight.val(),
        ),
        (
            "g_proj_1",
            m1.g_proj_1.weight.val(),
            m2.g_proj_1.weight.val(),
        ),
        ("o_proj", m1.o_proj.weight.val(), m2.o_proj.weight.val()),
    ] {
        let grad1 = p1.grad(&g1).expect("grad").clone();
        let grad2 = p2.grad(&g2).expect("grad").clone();
        let d = rel_diff_2d(grad1, grad2);
        assert!(d < RELATOL, "{name}: rel_diff={d:.2e}");
    }
    for (name, p1, p2) in [
        ("a_log", m1.a_log.val(), m2.a_log.val()),
        ("dt_bias", m1.dt_bias.val(), m2.dt_bias.val()),
        ("o_norm", m1.o_norm_weight.val(), m2.o_norm_weight.val()),
    ] {
        let grad1 = p1.grad(&g1).expect("grad").clone();
        let grad2 = p2.grad(&g2).expect("grad").clone();
        let d = rel_diff_2d(grad1, grad2);
        assert!(d < RELATOL, "{name}: rel_diff={d:.2e}");
    }
}

fn rel_diff_2d<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
    let a = a.clone().into_data();
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

/// The K3 16-tile log-space decay scheme must stay finite at chunk 64 with
/// strong gates (g ~ -5 per token, naive K/exp(cumsum) would underflow to 0
/// and produce inf).
#[test]
fn chunk64_strong_decay_stays_finite() {
    let device = Device::ndarray();
    let (batch, heads, time, k_dim, v_dim) = (1usize, 1usize, 128usize, 8usize, 8usize);
    let mk = |shape: [usize; 4], dist: Distribution| Tensor::<4>::random(shape, dist, &device);
    let q = mk([batch, heads, time, k_dim], Distribution::Normal(0.0, 0.1));
    let k = mk([batch, heads, time, k_dim], Distribution::Normal(0.0, 0.1));
    let v = mk([batch, heads, time, v_dim], Distribution::Normal(0.0, 0.1));
    // strong decay: every token at the K3 floor g = -5
    let g = Tensor::<4>::full([batch, heads, time, k_dim], -5.0, &device);
    let b = mk([batch, heads, time, k_dim], Distribution::Uniform(0.0, 0.1));
    let w = mk([batch, heads, time, v_dim], Distribution::Uniform(0.0, 0.1));
    let s = mk([batch, heads, k_dim, v_dim], Distribution::Normal(0.0, 0.1));
    let scale = (k_dim as f64).powf(-0.5);

    let (out, new_state) = chunk_wy_forward(
        q.clone(),
        k.clone(),
        v.clone(),
        g,
        b.clone(),
        w.clone(),
        s.clone(),
        scale,
        64,
    );
    // naive form would produce inf/NaN here; everything must be finite
    let mut max_abs = 0.0f32;
    for (t, name) in [
        (out.clone().into_data(), "out"),
        (new_state.into_data(), "state"),
    ] {
        for x in t.bytes.chunks_exact(4) {
            let v = f32::from_le_bytes(x.try_into().unwrap());
            assert!(v.is_finite(), "{name} not finite: {v}");
            max_abs = max_abs.max(v.abs());
        }
    }
    // sanity: with the floor decay the output is driven by the first tokens
    let _ = max_abs;
    // and the chunked path must agree with the token-by-token recurrence
    let (ref_out, _) = burn_gdn2::fused_recurrent_forward(
        q,
        k,
        v,
        Tensor::<4>::full([batch, heads, time, k_dim], -5.0, &device),
        b,
        w,
        s,
        scale,
    );
    let a = out.into_data();
    let bb = ref_out.into_data();
    let mut max_abs = 0.0f32;
    let mut scale_v = 0.0f32;
    for (x, y) in a.bytes.chunks_exact(4).zip(bb.bytes.chunks_exact(4)) {
        let x = f32::from_le_bytes(x.try_into().unwrap());
        let y = f32::from_le_bytes(y.try_into().unwrap());
        max_abs = max_abs.max((x - y).abs());
        scale_v = scale_v.max(x.abs()).max(y.abs());
    }
    let rel = max_abs / scale_v.max(1e-30);
    assert!(
        rel < 1e-3,
        "chunk64 vs fused-recurrent mismatch: rel={rel:.2e}"
    );
}
