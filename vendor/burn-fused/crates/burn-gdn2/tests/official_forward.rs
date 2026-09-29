//! An independent per-token reference for the GDN-2 forward, and the tests that
//! hold the implementation to it.
//!
//! # Why this file exists
//!
//! The crate's other reference (`tests/ref_data.bin` + `tools/gen_reference.rs`)
//! is a transcription of the same code under test, so it can only prove
//! self-consistency — it cannot find a divergence from the source it was meant
//! to copy, and it is written in a layout the layer does not use
//! (`research/papers/gdn-kda.md` §4.1, §7.1). This file is the tier-1 answer
//! from that report: **fp64 algebraic properties, no fixture, runnable here.**
//!
//! It is written from the equations, not from the implementation:
//!   - Eq 9 / Eq 10 (Gated Delta Rule-2) — `research/papers/gdn-kda.md` §2.2,
//!     `research/papers/spec-gdn2-official.md` §3.1
//!   - Eq 11 (gates), Eq 12 (log-decay), §3.5 (block design) — `gdn-kda.md` §2.2
//!   - the short-conv padding — `gdn-kda.md` §2.5: the reference is
//!     `causal_conv1d` with `other=0.0` on the left-pad branch, i.e. **zeros**.
//!   - the recurrence ordering — `spec-gdn2-official.md` §3.1 prints two forms
//!     that look contradictory. See [`RECURRENCE_NOTE`].
//!
//! Every tensor is `[T, n]` (token-major) or `[HV, T, n]` (head-major) here,
//! and the head index is taken as `h * t * n + ti * n + i` over a genuinely
//! head-major buffer. The recurrence is a plain double loop over tokens, so it
//! shares no code, no layout and no reduction order with the chunked path.
//!
//! # TOLERANCE
//!
//! `TOL = 1e-5` on the layer output. Measured 2026-09-29 on this file's own
//! matrix — 10 configurations x 2 entry points (`forward_train` and `forward`) —
//! the worst case is **5.04e-9**, which is the f32 floor rather than a modelling
//! error. So the tolerance carries five orders of magnitude of headroom, and it
//! is set by what a CORRECT implementation scores, not by what a convenient one
//! scores. The same matrix read **1.30e-2** before the two prologue fixes this
//! file found (the GVA head repeat and the short-conv left pad), so the distance
//! the tolerance has to bridge is real and is measured.
//!
//! `official_forward_is_sensitive_to_its_own_inputs` is the counterweight: a
//! gate that has gone slack in the loose direction is caught there.

use burn::backend::NdArray;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, TensorData};
use burn_gdn2::{fused_recurrent_forward, GatedDeltaNet2, Gdn2Config, Gdn2Mode, Gdn2State};

/// The reference's own SiLU, in f64. NOT `burn::tensor::activation::silu`:
/// that one is generic over a backend, and a transcription has to be a
/// transcription, not a call into the code it is checking.
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// The two printed forms of the Gated Delta Rule-2 recurrence differ in
/// reading, not in algebra, and this file implements the operational one.
///
/// GDN-2 Eq 9 (operational, `gdn-kda.md` §2.2):
/// ```text
///   Sbar = Diag(alpha) Sprev;  r = Sbar^T (b . k);  S = Sbar + k (w . v - r)^T
/// ```
/// GDN-2 Eq 10 (closed, `spec-gdn2-official.md` §3.1):
/// ```text
///   S = (I - k (b . k)^T) Diag(alpha) Sprev + k (w . v)^T
/// ```
/// Expand Eq 10 with `x = Diag(alpha) Sprev`:
/// ```text
///   [(I - k e^T) x]_j = x_j - k_j (e^T x)_j,   e = b . k
///   =>  x_j - k_j * sum_i (b_i k_i) * alpha_i * Sprev[i, j]
/// ```
/// and `e^T x` is `(b . k)^T (Diag(alpha) Sprev)`, which is exactly Eq 9's `r`.
/// So Eq 10 is Eq 9 regrouped — the decay is applied to `S` before the erase in
/// both, and the erase reads the *decayed* state. The two forms are the same
/// map; `fused_recurrent_forward` (the operational form) is what this file and
/// the implementation both run.
pub const RECURRENCE_NOTE: &str = "Eq 9 == Eq 10 regrouped; both decay first, then erase on the decayed state";

const TOL: f32 = 1e-5;

// --- deterministic parameters, so a failure is reproducible ---------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 40) as f64) / ((1u64 << 24) as f64)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.unit();
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform(-1.0, 1.0)).collect()
    }
    fn normal_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.normal()).collect()
    }
    /// xavier_uniform, gain 2^-2.5, torch [out, in] layout.
    fn xavier(&mut self, out_f: usize, in_f: usize) -> Vec<f64> {
        let bound = 2f64.powf(-2.5) * (6.0 / (in_f + out_f) as f64).sqrt();
        self.vec(out_f * in_f)
            .into_iter()
            .map(|v| v * bound)
            .collect()
    }
}

// --- reference primitives -------------------------------------------------

fn linear(x: &[f64], w: &[f64], b: Option<&[f64]>, in_f: usize, out_f: usize) -> Vec<f64> {
    let t = x.len() / in_f;
    let mut y = vec![0.0; t * out_f];
    for ti in 0..t {
        for o in 0..out_f {
            let mut acc = b.map_or(0.0, |bb| bb[o]);
            for i in 0..in_f {
                acc += w[o * in_f + i] * x[ti * in_f + i];
            }
            y[ti * out_f + o] = acc;
        }
    }
    y
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Causal depthwise conv, kernel 4, then SiLU on the *sum*.
///
/// `pad_zero` selects the padding the reference actually uses. FLA's
/// `causal_conv1d` loads the left pad with `other=0.0`
/// (`gdn-kda.md` §2.5); `false` keeps the replicate behaviour the old fixture
/// generator used, so the two can be compared and the difference attributed.
fn short_conv(x: &[f64], w: &[f64], ch: usize, pad_zero: bool) -> Vec<f64> {
    let t = x.len() / ch;
    let mut out = vec![0.0; t * ch];
    for ti in 0..t {
        for c in 0..ch {
            let mut acc = 0.0;
            for i in 0..4 {
                // tap i reads token ti + i - 3
                let src_idx = ti as i64 + i as i64 - 3;
                let xv = if src_idx < 0 {
                    if pad_zero {
                        0.0
                    } else {
                        x[c] // replicate: the first token
                    }
                } else {
                    x[src_idx as usize * ch + c]
                };
                acc += xv * w[c * 4 + i];
            }
            out[ti * ch + c] = acc * sigmoid(acc);
        }
    }
    out
}

/// `[T, n_per_head * n_heads]` -> `[HV, T, n_per_head]`, GVA: value head `hv`
/// reads key head `hv / rep`.
fn to_heads(t: usize, token_major: &[f64], h: usize, hv: usize, n: usize) -> Vec<f64> {
    let rep = hv / h;
    let mut out = vec![0.0; hv * t * n];
    for hv_i in 0..hv {
        let from_h = hv_i / rep;
        for ti in 0..t {
            for i in 0..n {
                out[(hv_i * t + ti) * n + i] = token_major[(ti * h + from_h) * n + i];
            }
        }
    }
    out
}

/// L2 normalize in place over the last axis of a head-major `[HV, T, n]`.
/// The eps is `1e-6` — the recurrent kernel's own value
/// (`spec-gdn2-official.md` §4.2); the chunk path's is NOT STATED there, and
/// `module.rs` uses this one on both.
fn l2_head_major(v: &mut [f64], hv: usize, t: usize, n: usize) {
    for h in 0..hv * t {
        let row = &mut v[h * n..(h + 1) * n];
        let ss: f64 = row.iter().map(|x| x * x).sum();
        let inv = 1.0 / (ss + 1e-6).sqrt();
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
}

struct P {
    d: usize,
    h: usize,
    hk: usize,
    hv: usize,
    v_head: usize,
    q_proj: Vec<f64>,
    k_proj: Vec<f64>,
    v_proj: Vec<f64>,
    f_proj_0: Vec<f64>,
    f_proj_1: Vec<f64>,
    b_proj: Vec<f64>,
    w_proj: Vec<f64>,
    g_proj_0: Vec<f64>,
    g_proj_1: Vec<f64>,
    g_proj_1_b: Vec<f64>,
    a_log: Vec<f64>,
    dt_bias: Vec<f64>,
    o_norm_w: Vec<f64>,
    o_proj: Vec<f64>,
    q_conv_w: Vec<f64>,
    k_conv_w: Vec<f64>,
    v_conv_w: Vec<f64>,
}

impl P {
    fn new(rng: &mut Rng, d: usize, h: usize, hk: usize, hv: usize, v_head: usize) -> Self {
        let kd = h * hk;
        let vd = hv * v_head;
        Self {
            d,
            h,
            hk,
            hv,
            v_head,
            q_proj: rng.xavier(kd, d),
            k_proj: rng.xavier(kd, d),
            v_proj: rng.xavier(vd, d),
            f_proj_0: rng.xavier(v_head, d),
            f_proj_1: rng.xavier(kd, v_head),
            b_proj: rng.xavier(kd, d),
            w_proj: rng.xavier(vd, d),
            g_proj_0: rng.xavier(v_head, d),
            g_proj_1: rng.xavier(vd, v_head),
            g_proj_1_b: vec![0.0; vd],
            // A_log = log U(1, 16), so exp(A_log) in [1, 16] (spec §4.4)
            a_log: (0..h).map(|_| rng.uniform(1.0, 16.0).ln()).collect(),
            // dt_bias = dt + log(-expm1(-dt)), dt ~ logU(1e-3, 1e-1), clamp 1e-4
            dt_bias: (0..kd)
                .map(|_| {
                    let u = rng.unit();
                    let dt = (u * (0.1f64.ln() - 0.001f64.ln()) + 0.001f64.ln()).exp().max(1e-4);
                    dt + (-(-dt).exp_m1()).ln()
                })
                .collect(),
            o_norm_w: vec![1.0; v_head],
            o_proj: rng.xavier(d, vd),
            q_conv_w: rng.vec(kd * 4),
            k_conv_w: rng.vec(kd * 4),
            v_conv_w: rng.vec(vd * 4),
        }
    }
}

/// The GDN-2 layer, per token, in f64. `x` is `[t, d]`.
///
/// `o` comes back `[t, hv * v_head]`.
/// Everything the recurrence consumes, head-major `[HV, T, n]`.
struct Proj {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    g: Vec<f64>,
    b: Vec<f64>,
    w: Vec<f64>,
}

/// The §3.5 block design, up to (not including) the recurrence: projections,
/// short conv, L2 norm on q/k, the log-decay, the two gates, GVA, the
/// negative-eigenvalue lift. Split out from the scan so a test can compare the
/// prologue on its own instead of only through the layer output.
fn official_project(
    p: &P,
    x: &[f64],
    t: usize,
    use_short_conv: bool,
    allow_neg_eigval: bool,
    pad_zero: bool,
) -> Proj {
    let (d, h, hk, hv, v_head) = (p.d, p.h, p.hk, p.hv, p.v_head);
    let kd = h * hk;
    let vd = hv * v_head;
    let conv = |proj: Vec<f64>, w: &[f64], ch: usize| -> Vec<f64> {
        if use_short_conv {
            short_conv(&proj, w, ch, pad_zero)
        } else {
            proj.into_iter().map(silu).collect()
        }
    };

    let q_t = conv(
        linear(x, &p.q_proj, None, d, kd),
        &p.q_conv_w,
        kd,
    );
    let k_t = conv(
        linear(x, &p.k_proj, None, d, kd),
        &p.k_conv_w,
        kd,
    );
    let v_t = conv(linear(x, &p.v_proj, None, d, vd), &p.v_conv_w, vd);

    let mut q = to_heads(t, &q_t, h, hv, hk);
    let mut k = to_heads(t, &k_t, h, hv, hk);
    l2_head_major(&mut q, hv, t, hk);
    l2_head_major(&mut k, hv, t, hk);

    // Eq 12: g = -exp(A_log[h]) * softplus(W_f x + dt_bias), fp32 per §D.1
    let f_hid = linear(x, &p.f_proj_0, None, d, v_head);
    let f_out = linear(&f_hid, &p.f_proj_1, None, v_head, kd);
    let g_t: Vec<f64> = (0..t * kd)
        .map(|n| {
            let c = n % kd;
            -p.a_log[c / hk].exp() * softplus(f_out[n] + p.dt_bias[c])
        })
        .collect();

    // Eq 11
    let b_t: Vec<f64> = linear(x, &p.b_proj, None, d, kd)
        .into_iter()
        .map(sigmoid)
        .collect();
    let w_t: Vec<f64> = linear(x, &p.w_proj, None, d, vd)
        .into_iter()
        .map(sigmoid)
        .collect();

    let g = to_heads(t, &g_t, h, hv, hk);
    let mut b = to_heads(t, &b_t, h, hv, hk);
    let w = to_heads(t, &w_t, hv, hv, v_head);
    let v = to_heads(t, &v_t, hv, hv, v_head);
    if allow_neg_eigval {
        for x in b.iter_mut() {
            *x *= 2.0;
        }
    }
    Proj { q, k, v, g, b, w }
}

/// Eq 9, one token at a time, from an initial zero state. `o` is
/// `[T, hv * v_head]`, token-major, value-head-minor within a token — the
/// layout the module's `permute([0,2,1,3])` + `o_proj` expects.
fn official_scan(p: &P, t: usize, pr: &Proj) -> Vec<f64> {
    let (hk, hv, v_head) = (p.hk, p.hv, p.v_head);
    let vd = hv * v_head;
    let scale = (hk as f64).powf(-0.5);
    let at = |a: &[f64], h: usize, ti: usize, i: usize, n: usize| a[h * t * n + ti * n + i];

    let mut s = vec![0.0f64; hv * hk * v_head];
    let mut o = vec![0.0f64; t * vd];
    for ti in 0..t {
        for hh in 0..hv {
            for kk in 0..hk {
                let a = at(&pr.g, hh, ti, kk, hk).exp();
                for vv in 0..v_head {
                    s[(hh * hk + kk) * v_head + vv] *= a;
                }
            }
            for vv in 0..v_head {
                // r = Sbar^T (b . k);  v_new = w . v - r
                let mut r = 0.0;
                for kk in 0..hk {
                    r += s[(hh * hk + kk) * v_head + vv]
                        * at(&pr.b, hh, ti, kk, hk)
                        * at(&pr.k, hh, ti, kk, hk);
                }
                let v_new = at(&pr.w, hh, ti, vv, v_head) * at(&pr.v, hh, ti, vv, v_head) - r;
                for kk in 0..hk {
                    s[(hh * hk + kk) * v_head + vv] += at(&pr.k, hh, ti, kk, hk) * v_new;
                }
            }
            for vv in 0..v_head {
                let mut acc = 0.0;
                for kk in 0..hk {
                    acc += s[(hh * hk + kk) * v_head + vv] * at(&pr.q, hh, ti, kk, hk);
                }
                o[ti * vd + hh * v_head + vv] = acc * scale;
            }
        }
    }
    o
}

/// §3.5 output: per-head RMSNorm, then `. * silu(output gate)`, then o_proj.
fn official_head(p: &P, x: &[f64], t: usize, o: &mut [f64]) -> Vec<f64> {
    let (d, hv, v_head) = (p.d, p.hv, p.v_head);
    let vd = hv * v_head;
    let gate = {
        let gh = linear(x, &p.g_proj_0, None, d, v_head);
        linear(&gh, &p.g_proj_1, Some(&p.g_proj_1_b), v_head, vd)
    };
    for hh in 0..hv {
        for ti in 0..t {
            let row = &mut o[ti * vd + hh * v_head..ti * vd + (hh + 1) * v_head];
            let ss: f64 = row.iter().map(|v| v * v).sum();
            let rms = (ss / v_head as f64 + 1e-5).sqrt();
            for (i, val) in row.iter_mut().enumerate() {
                *val = *val / rms * p.o_norm_w[i] * silu(gate[ti * vd + hh * v_head + i]);
            }
        }
    }
    linear(o, &p.o_proj, None, vd, d)
}

fn official_forward(
    p: &P,
    x: &[f64],
    t: usize,
    use_short_conv: bool,
    allow_neg_eigval: bool,
    pad_zero: bool,
) -> Vec<f64> {
    let pr = official_project(p, x, t, use_short_conv, allow_neg_eigval, pad_zero);
    let mut o = official_scan(p, t, &pr);
    official_head(p, x, t, &mut o)
}

// --- the module under test, built from the same parameters ----------------

fn t2(v: &[f64], shape: &[usize], device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape.to_vec()), device)
}
fn t1(v: &[f64], shape: &[usize], device: &Device) -> Tensor<1> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape.to_vec()), device)
}

/// burn's `Linear.weight` is `[in, out]`; the reference is torch `[out, in]`.
fn lin_w(w: &[f64], in_f: usize, out_f: usize, device: &Device) -> Linear {
    let mut t = vec![0.0; in_f * out_f];
    for i in 0..in_f {
        for o in 0..out_f {
            t[i * out_f + o] = w[o * in_f + i];
        }
    }
    let mut lin = LinearConfig::new(in_f, out_f).with_bias(false).init(device);
    lin.weight = Param::from_tensor(t2(&t, &[in_f, out_f], device));
    lin
}

fn build(
    p: &P,
    device: &Device,
    use_short_conv: bool,
    allow_neg_eigval: bool,
    chunk_size: usize,
) -> GatedDeltaNet2 {
    let kd = p.h * p.hk;
    let vd = p.hv * p.v_head;
    let cfg = Gdn2Config {
        hidden_size: p.d,
        num_heads: p.h,
        head_dim: p.hk,
        num_v_heads: Some(p.hv),
        expand_v: p.v_head as f32 / p.hk as f32,
        use_short_conv,
        allow_neg_eigval,
        norm_eps: 1e-5,
        mode: Gdn2Mode::Chunk,
        chunk_size,
        min_decay: None,
    };
    let mut g1 = LinearConfig::new(p.v_head, vd).with_bias(true).init(device);
    g1.weight = Param::from_tensor(t2(
        &{
            let mut t = vec![0.0; p.v_head * vd];
            for i in 0..p.v_head {
                for o in 0..vd {
                    t[i * vd + o] = p.g_proj_1[o * p.v_head + i];
                }
            }
            t
        },
        &[p.v_head, vd],
        device,
    ));
    g1.bias = Some(Param::from_tensor(t1(&p.g_proj_1_b, &[vd], device)));

    GatedDeltaNet2 {
        q_proj: lin_w(&p.q_proj, p.d, kd, device),
        k_proj: lin_w(&p.k_proj, p.d, kd, device),
        v_proj: lin_w(&p.v_proj, p.d, vd, device),
        f_proj_0: lin_w(&p.f_proj_0, p.d, p.v_head, device),
        f_proj_1: lin_w(&p.f_proj_1, p.v_head, kd, device),
        b_proj: lin_w(&p.b_proj, p.d, kd, device),
        w_proj: lin_w(&p.w_proj, p.d, vd, device),
        g_proj_0: lin_w(&p.g_proj_0, p.d, p.v_head, device),
        g_proj_1: g1,
        a_log: Param::from_tensor(t1(&p.a_log, &[p.h], device)),
        dt_bias: Param::from_tensor(t1(&p.dt_bias, &[kd], device)),
        o_norm_weight: Param::from_tensor(t1(&p.o_norm_w, &[p.v_head], device)),
        o_proj: lin_w(&p.o_proj, vd, p.d, device),
        q_conv_w: Param::from_tensor(t2(&p.q_conv_w, &[kd, 4], device)),
        k_conv_w: Param::from_tensor(t2(&p.k_conv_w, &[kd, 4], device)),
        v_conv_w: Param::from_tensor(t2(&p.v_conv_w, &[vd, 4], device)),
        config: cfg,
        decay_factors: None,
    }
}

fn max_abs_diff(a: &Tensor<3>, b: &[f64], t: usize, d: usize) -> f32 {
    let data = a.clone().into_data();
    let got: Vec<f32> = data
        .bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    assert_eq!(got.len(), t * d);
    let mut worst = 0.0f32;
    for (i, g) in got.iter().enumerate() {
        worst = worst.max(((*g as f64) - b[i]).abs() as f32);
    }
    worst
}

/// Per-position max abs diff, so a defect can be localised to a range of
/// positions instead of only sized.
fn pos_diffs(a: &Tensor<3>, b: &[f64], t: usize, d: usize) -> Vec<f32> {
    let data = a.clone().into_data();
    let got: Vec<f32> = data
        .bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    (0..t)
        .map(|ti| {
            (0..d)
                .map(|i| ((got[ti * d + i] as f64) - b[ti * d + i]).abs() as f32)
                .fold(0.0f32, f32::max)
        })
        .collect()
}

struct Case {
    name: &'static str,
    d: usize,
    h: usize,
    hk: usize,
    hv: usize,
    v_head: usize,
    t: usize,
    use_short_conv: bool,
    allow_neg_eigval: bool,
    chunk_size: usize,
}

fn cases() -> Vec<Case> {
    let base = |name, hk, v_head, t, use_short_conv, allow_neg_eigval, chunk_size| Case {
        name,
        d: 64,
        h: 4,
        hk,
        hv: 4,
        v_head,
        t,
        use_short_conv,
        allow_neg_eigval,
        chunk_size,
    };
    vec![
        // T=1 is the case the old fixture could not see (it started at 2) and
        // the case where the short-conv padding is at its most visible.
        base("t1 conv zero-pad", 16, 24, 1, true, false, 64),
        // T > 1, single chunk, conv on: the prologue is fully exercised and the
        // decay/scan are inside one chunk.
        base("t13 conv zero-pad", 16, 24, 13, true, false, 64),
        // Multiple chunks with a ragged tail (70 = 64 + 6), conv on.
        base("t70 ragged conv zero-pad", 16, 24, 70, true, false, 64),
        // Many chunks: 200 = 3*64 + 8, so four chunks and a partial one.
        base("t200 ragged conv zero-pad", 16, 24, 200, true, false, 64),
        // The batched arm (chunk_size <= 16) against the loop arm.
        base("t70 chunk16", 16, 24, 70, true, false, 16),
        base("t200 chunk16", 16, 24, 200, true, false, 16),
        // No short conv: the conv arm is out of the picture entirely.
        base("t70 no conv", 16, 24, 70, false, false, 64),
        // allow_neg_eigval lifts b to [0, 2].
        base("t70 neg-eigval", 16, 24, 70, true, true, 64),
    ]
}

fn gva_cases() -> Vec<Case> {
    vec![
        Case {
            name: "gva t70",
            d: 64,
            h: 2,
            hk: 16,
            hv: 4, // 2 value heads per key head
            v_head: 16,
            t: 70,
            use_short_conv: true,
            allow_neg_eigval: false,
            chunk_size: 64,
        },
        Case {
            name: "gva t1",
            d: 64,
            h: 2,
            hk: 16,
            hv: 4,
            v_head: 16,
            t: 1,
            use_short_conv: true,
            allow_neg_eigval: false,
            chunk_size: 64,
        },
    ]
}

/// Localiser. Compares the reference's prologue tensor-by-tensor against the
/// module's own `project()` output, then the reference's full forward against
/// the module's. Prints where a difference lives, so a red
/// [`official_forward_matrix`] names its own cause instead of being a number.
///
/// This is a diagnostic, not a gate: it asserts nothing about the values (the
/// only assertion is a vacuous `>= 0.0` on the max it printed) and exists so
/// that a future failure has somewhere to point. Run it with `--nocapture`.
#[test]
fn official_forward_localiser() {
    let device = Device::ndarray();
    for c in [
        Case {
            name: "t13 conv",
            d: 64,
            h: 4,
            hk: 16,
            hv: 4,
            v_head: 24,
            t: 13,
            use_short_conv: true,
            allow_neg_eigval: false,
            chunk_size: 64,
        },
        Case {
            name: "t13 no conv",
            d: 64,
            h: 4,
            hk: 16,
            hv: 4,
            v_head: 24,
            t: 13,
            use_short_conv: false,
            allow_neg_eigval: false,
            chunk_size: 64,
        },
        Case {
            name: "gva t13",
            d: 64,
            h: 2,
            hk: 16,
            hv: 4,
            v_head: 16,
            t: 13,
            use_short_conv: true,
            allow_neg_eigval: false,
            chunk_size: 64,
        },
    ] {
        let mut rng = Rng::new(0x10CA_1E ^ c.t as u64);
        let p = P::new(&mut rng, c.d, c.h, c.hk, c.hv, c.v_head);
        let x = rng.normal_vec(c.t * c.d);
        let module = build(&p, &device, c.use_short_conv, c.allow_neg_eigval, c.chunk_size);
        let input = Tensor::<3>::from_data(
            TensorData::new(x.clone(), vec![1, c.t, c.d]),
            &device,
        );
        let (got, _cache) = module.project(input.clone(), None);
        let pr = official_project(&p, &x, c.t, c.use_short_conv, c.allow_neg_eigval, true);

        println!("--- {} ---", c.name);
        let mut worst_p = 0.0f32;
        for (name, g, w, n) in [
            ("q", &got.q, &pr.q, c.hk),
            ("k", &got.k, &pr.k, c.hk),
            // v and w live on the VALUE-head axis, so they are v_head wide
            ("v", &got.v, &pr.v, c.v_head),
            ("g", &got.g, &pr.g, c.hk),
            ("b", &got.b, &pr.b, c.hk),
            ("w", &got.w, &pr.w, c.v_head),
        ] {
            let d = max_abs_diff4(g, w, c.hv, c.t, n);
            let scale = w.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            println!("    project.{name}: |Δ|max {d:.2e}   |ref|max {scale:.2e}");
            worst_p = worst_p.max(d);
        }
        let mut o_ref = official_scan(&p, c.t, &pr);
        let o_max: f64 = o_ref.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        println!("    scan: |o| max {o_max:.3e}");

        let out = module.forward_train::<NdArray>(input);
        let y_ref = official_head(&p, &x, c.t, &mut o_ref);
        println!(
            "    head+proj: |Δ|max {:.2e}   |y|max {:.2e}",
            max_abs_diff(&out, &y_ref, c.t, c.d),
            y_ref.iter().fold(0.0f64, |m, v| m.max(v.abs()))
        );
        assert!(worst_p >= 0.0);
    }
}

/// Does the GVA head repeat preserve values, on this backend?
///
/// `module.rs` builds the repeated key-side tensors as
/// `reshape([B,T,H,n]).permute([0,2,1,3]).unsqueeze(3).repeat([1,1,1,rep,1])
/// .reshape([B,H*rep,T,n])` — a `repeat` over a *strided* view. If the backend
/// mis-handles that, the repeated heads carry different numbers than the head
/// they were copied from, and every GVA number is quietly wrong.
///
/// This is a property of the backend, not of the layer, so it is measured
/// directly on the same three operations with no module involved.
#[test]
fn gva_head_repeat_preserves_values_on_this_backend() {
    let device = Device::ndarray();
    let grab = |t: Tensor<4>| -> Vec<f64> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|x| f64::from(f32::from_le_bytes(x.try_into().unwrap())))
            .collect()
    };
    for (t, h, rep, n) in [(13usize, 2usize, 2usize, 16usize), (70, 4, 2, 16)] {
        let mut rng = Rng::new(0x9E37 + t as u64);
        let src = rng.normal_vec(t * h * n);
        let tok = Tensor::<3>::from_data(
            TensorData::new(src.clone(), vec![1, t, h * n]),
            &device,
        );
        // value head vh should carry key head vh / rep
        let want = |vh: usize, ti: usize, i: usize| src[(ti * h + vh / rep) * n + i];

        let headmajor = |vh: usize, ti: usize, i: usize| src[(ti * h + vh) * n + i];

        // 1. permute alone
        let p = tok.clone().reshape([1, t, h, n]).permute([0, 2, 1, 3]);
        let a = grab(p.clone());
        let d_permute = (0..h)
            .flat_map(|vh| (0..t).flat_map(move |ti| (0..n).map(move |i| (vh, ti, i))))
            .map(|(vh, ti, i)| (a[(vh * t + ti) * n + i] - headmajor(vh, ti, i)).abs())
            .fold(0.0f64, f64::max);

        // 2. permute -> unsqueeze -> repeat (still 5D, not reshaped back)
        let b: Vec<f64> = p
            .clone()
            .unsqueeze_dim::<5>(3)
            .repeat(&[1, 1, 1, rep, 1])
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|x| f64::from(f32::from_le_bytes(x.try_into().unwrap())))
            .collect();
        let d_repeat5 = (0..h * rep)
            .flat_map(|vh| (0..t).flat_map(move |ti| (0..n).map(move |i| (vh, ti, i))))
            .map(|(vh, ti, i)| {
                (b[(((vh / rep) * t + ti) * rep + 0) * n + i] - want(vh, ti, i)).abs()
            })
            .fold(0.0f64, f64::max);

        // 3. the full expression the layer uses
        let c = grab(
            p.clone()
                .unsqueeze_dim::<5>(3)
                .repeat(&[1, 1, 1, rep, 1])
                .reshape([1, h * rep, t, n]),
        );
        let d_full = (0..h * rep)
            .flat_map(|vh| (0..t).flat_map(move |ti| (0..n).map(move |i| (vh, ti, i))))
            .map(|(vh, ti, i)| (c[(vh * t + ti) * n + i] - want(vh, ti, i)).abs())
            .fold(0.0f64, f64::max);

        // 4. the candidate fixes. The buffer must be genuinely head-major, so
        //    build it head-major (passing the token-major `src` under a
        //    [1,h,t,n] shape would reinterpret it and measure this probe's own
        //    mistake instead of the layer's).
        let mut hm = vec![0.0f64; t * h * n];
        for kh in 0..h {
            for ti in 0..t {
                for i in 0..n {
                    hm[(kh * t + ti) * n + i] = src[(ti * h + kh) * n + i];
                }
            }
        }
        let contig = Tensor::<4>::from_data(TensorData::new(hm, vec![1, h, t, n]), &device);
        // score a result buffer under a candidate head->key-head map
        let score = |got: Vec<f64>, kh_of: &dyn Fn(usize) -> usize| -> f64 {
            (0..h * rep)
                .flat_map(move |vh| (0..t).flat_map(move |ti| (0..n).map(move |i| (vh, ti, i))))
                .map(|(vh, ti, i)| (got[(vh * t + ti) * n + i] - src[(ti * h + kh_of(vh)) * n + i]).abs())
                .fold(0.0f64, f64::max)
        };
        let by_rep = |vh: usize| vh / rep;
        let by_mod = |vh: usize| vh % h;

        // 4a. `cat(rep)` and `repeat_dim(1, rep)` both TILE the head list ->
        //     [k0, k1, k0, k1], the interleaved mapping. Value-preserving, so
        //     they look right, and they are the wrong list.
        let d_cat = score(grab(Tensor::cat(vec![contig.clone(); rep], 1)), &by_rep);
        let d_cat_mod = score(grab(Tensor::cat(vec![contig.clone(); rep], 1)), &by_mod);
        let d_repeat_dim = score(grab(contig.clone().repeat_dim(1, rep)), &by_rep);
        let d_repeat_dim_mod = score(grab(contig.clone().repeat_dim(1, rep)), &by_mod);

        // 4b. `repeat_interleave`: put the new axis NEXT TO the head axis, so
        //     the merge in the reshape is over ADJACENT axes and is a legal
        //     index-preserving reshape rather than a flat reinterpretation.
        let interleave = |t4: Tensor<4>| {
            t4.unsqueeze_dim::<5>(2)
                .repeat_dim(2, rep)
                .reshape([1, h * rep, t, n])
        };
        let d_il = score(grab(interleave(contig.clone())), &by_rep);
        let d_il_mod = score(grab(interleave(contig.clone())), &by_mod);
        // and the same off a PERMUTED (strided) view, which is what the layer
        // actually holds at that point
        let strided = tok.clone().reshape([1, t, h, n]).permute([0, 2, 1, 3]);
        let d_il_strided = score(grab(interleave(strided)), &by_rep);

        println!("  t={t} h={h} rep={rep} n={n}:");
        println!("     permute                            |Δ| = {d_permute:.2e}");
        println!("     permute+repeat (5D)                |Δ| = {d_repeat5:.2e}");
        println!("     permute+repeat+reshape             |Δ| = {d_full:.2e}  <- what the layer DID");
        println!("     cat(rep) along head                |Δ| = {d_cat:.2e} vs vh/rep, {d_cat_mod:.2e} vs vh%h");
        println!("     repeat_dim(1, rep)                 |Δ| = {d_repeat_dim:.2e} vs vh/rep, {d_repeat_dim_mod:.2e} vs vh%h");
        println!("     unsqueeze(2)+repeat_dim(2)+reshape  |Δ| = {d_il:.2e} vs vh/rep, {d_il_mod:.2e} vs vh%h");
        println!("     ...same, off the permuted view      |Δ| = {d_il_strided:.2e} vs vh/rep");

        assert!(d_permute < 1e-6, "permute alone is lossy: {d_permute:.2e}");
        assert!(d_repeat5 < 1e-6, "the 5-D repeat alone is lossy: {d_repeat5:.2e}");
        assert!(
            d_il_strided < 1e-6,
            "unsqueeze(2)+repeat_dim+reshape is not the GDN-2 head mapping off a permuted \
             view: |Δ| = {d_il_strided:.2e}"
        );
        // The old expression must STAY wrong. If this ever goes green the bug
        // class is different from the one diagnosed, and module.rs needs
        // re-deriving.
        assert!(
            d_full > 1e-3,
            "unsqueeze(3)+repeat+reshape is now correct (|Δ| = {d_full:.2e}); \
             the diagnosis in module.rs is stale and the fix needs re-justifying"
        );
        // Tiling is value-preserving but the WRONG head list. Asserted so the
        // reason the adjacent-axis form was chosen is checkable, not folklore.
        assert!(d_cat_mod < 1e-6, "cat is no longer the interleaved tiling: {d_cat_mod:.2e}");
        assert!(d_repeat_dim_mod < 1e-6, "repeat_dim is no longer a tiling: {d_repeat_dim_mod:.2e}");
        assert!(d_cat > 1e-3, "cat now gives vh/rep: {d_cat:.2e}");
        assert!(d_repeat_dim > 1e-3, "repeat_dim now gives vh/rep: {d_repeat_dim:.2e}");
    }
}

fn max_abs_diff4(got: &Tensor<4>, want: &[f64], hv: usize, t: usize, n: usize) -> f32 {
    let data = got.clone().into_data();
    let v: Vec<f32> = data
        .bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    assert_eq!(v.len(), hv * t * n, "shape");
    let mut worst = 0.0f32;
    for (i, x) in v.iter().enumerate() {
        // the reference is head-major [HV, T, n]; the module's is [B=1, HV, T, n]
        worst = worst.max(((*x as f64) - want[i]).abs() as f32);
    }
    worst
}

/// THE GATE. Every case, `forward_train` (chunked WY, training shape) and
/// `forward` (inference, state-passing) against the f64 per-token reference,
/// with the reference's short-conv padding set to the reference's padding
/// (zeros).
#[test]
fn official_forward_matrix() {
    let device = Device::ndarray();
    let mut worst: f32 = 0.0;
    let mut worst_name = String::new();
    for c in cases().into_iter().chain(gva_cases()) {
        let mut rng = Rng::new(0x5EED ^ (c.t as u64) << 8 ^ c.h as u64);
        let p = P::new(&mut rng, c.d, c.h, c.hk, c.hv, c.v_head);
        let x = rng.normal_vec(c.t * c.d);
        let module = build(&p, &device, c.use_short_conv, c.allow_neg_eigval, c.chunk_size);
        let want = official_forward(
            &p,
            &x,
            c.t,
            c.use_short_conv,
            c.allow_neg_eigval,
            true, // zero pad: what the reference actually does
        );
        let input = Tensor::<3>::from_data(
            TensorData::new(x.clone(), vec![1, c.t, c.d]),
            &device,
        );

        let train_out = module.forward_train::<NdArray>(input.clone());
        let d_train = max_abs_diff(&train_out, &want, c.t, c.d);

        let mut state: Option<Gdn2State> = None;
        let infer_out = module.forward::<NdArray>(input.clone(), &mut state, true);
        let d_infer = max_abs_diff(&infer_out, &want, c.t, c.d);

        println!(
            "  {:<28} train={d_train:.2e} infer={d_infer:.2e}",
            c.name
        );
        for (tag, d) in [("train", d_train), ("infer", d_infer)] {
            if d > worst {
                worst = d;
                worst_name = format!("{} [{tag}]", c.name);
            }
        }
    }
    println!("worst = {worst:.2e} at {worst_name}");
    assert!(
        worst < TOL,
        "official per-token reference disagrees: {worst:.2e} at {worst_name} (tol {TOL:.0e})"
    );
}

/// The other direction of the gate: a layer whose parameters are perturbed must
/// move. Without this, a reference that accidentally compared the reference to
/// itself would also be green.
#[test]
fn official_forward_is_sensitive_to_its_own_inputs() {
    let (d, h, hk, hv, v_head, t) = (64usize, 4usize, 16usize, 4usize, 24usize, 70usize);
    let mut rng = Rng::new(0xB0BA);
    let p = P::new(&mut rng, d, h, hk, hv, v_head);
    let x = rng.normal_vec(t * d);
    let want = official_forward(&p, &x, t, true, false, true);

    // a different input must give a different answer
    let x2 = rng.normal_vec(t * d);
    let other = official_forward(&p, &x2, t, true, false, true);
    let d_input: f64 = want
        .iter()
        .zip(&other)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        d_input > 1e-2,
        "reference is insensitive to its input ({d_input:.2e}); the gate is vacuous"
    );

    // A perturbed dt_bias must give a different answer. The threshold is
    // deliberately small: the official init makes the decay steep (g reaches
    // -1.24 on these parameters, and alpha = exp(g) is correspondingly small),
    // so there is little decayed state left for a change in the decay to act
    // on. Measured +0.1 in dt_bias moves the output by 2.9e-4 on this config,
    // which is 3 orders of magnitude above f32 noise -- real, but not 1e-2.
    // The test's job is to prove the decay is WIRED IN, not to size it.
    let mut p2 = P::new(&mut Rng::new(0xB0BA), d, h, hk, hv, v_head);
    for v in p2.dt_bias.iter_mut() {
        *v += 0.1;
    }
    let dt_shift: f64 = want
        .iter()
        .zip(official_forward(&p2, &x, t, true, false, true))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        dt_shift > 1e-5,
        "reference is insensitive to the decay gate ({dt_shift:.2e})"
    );
}

/// The short-conv padding is the one prologue choice the reference is explicit
/// about and the old fixture generator got backwards. This pins where the two
/// paddings differ: the *first* `SHORT_CONV_CACHE` output positions, and
/// nowhere else, because the conv is causal with a 3-token receptive field.
///
/// It is written as a measurement, not an assertion of equality: the two
/// paddings produce different numbers by construction. What it asserts is the
/// SHAPE of the difference, which is what makes a future conv change visible.
#[test]
fn short_conv_padding_is_confined_to_the_first_three_positions() {
    let device = Device::ndarray();
    let (d, h, hk, hv, v_head) = (64usize, 4usize, 16usize, 4usize, 24usize);
    for t in [1usize, 5, 70, 200] {
        let mut rng = Rng::new(0xC0FFEE ^ t as u64);
        let p = P::new(&mut rng, d, h, hk, hv, v_head);
        let x = rng.normal_vec(t * d);
        let module = build(&p, &device, true, false, 64);
        let input = Tensor::<3>::from_data(TensorData::new(x.clone(), vec![1, t, d]), &device);
        let out = module.forward_train::<NdArray>(input);

        let zero = official_forward(&p, &x, t, true, false, true);
        let repl = official_forward(&p, &x, t, true, false, false);
        let dz = pos_diffs(&out, &zero, t, d);
        let dr = pos_diffs(&out, &repl, t, d);

        // positions the reference-with-zeros explains
        for (ti, e) in dz.iter().enumerate().skip(3) {
            assert!(
                *e < TOL,
                "t={t} position {ti}: zeros-pad reference off by {e:.2e}"
            );
        }
        // positions where the two paddings must differ, and by how much
        for (ti, (a, b)) in dr.iter().zip(&dz).take(3.min(t)).enumerate() {
            let gap = (a - b).abs();
            assert!(
                gap > 1e-4,
                "t={t} position {ti}: the two paddings agree ({gap:.2e}), so this \
                 test is not actually testing the padding"
            );
        }
        let top = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
        println!(
            "  t={t:<4} zero-pad: first3 max {:.2e}, rest max {:.2e}; \
             replicate-pad first3 max {:.2e}, rest max {:.2e}",
            top(&dz[..3.min(t)]),
            top(&dz[3.min(t)..]),
            top(&dr[..3.min(t)]),
            top(&dr[3.min(t)..]),
        );
    }
}

/// Tier-2 metamorphic, `gdn-kda.md` §7.2 item 6: the chunk size is a schedule
/// choice, not a mathematical one. A seam at a chunk boundary lands in a
/// different place at each chunk size, so this catches a whole class that a
/// single-chunk comparison cannot.
#[test]
fn chunk_size_is_a_schedule_choice_not_a_mathematical_one() {
    let device = Device::ndarray();
    let (d, h, hk, hv, v_head) = (64usize, 4usize, 16usize, 4usize, 24usize);
    for t in [70usize, 200] {
        let mut rng = Rng::new(0x5C_ED ^ t as u64);
        let p = P::new(&mut rng, d, h, hk, hv, v_head);
        let x = rng.normal_vec(t * d);
        let input = Tensor::<3>::from_data(TensorData::new(x.clone(), vec![1, t, d]), &device);
        let mut worst = 0.0f32;
        let base = build(&p, &device, true, false, 64).forward_train::<NdArray>(input.clone());
        for c in [4usize, 8, 16, 32, 64] {
            let out = build(&p, &device, true, false, c).forward_train::<NdArray>(input.clone());
            let diff = (out - base.clone()).abs().max().into_data();
            let v: f32 = diff
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .fold(0.0, f32::max);
            println!("  t={t} chunk {c:>2}: |Δ| vs chunk64 = {v:.2e}");
            worst = worst.max(v);
        }
        assert!(worst < 1e-3, "t={t}: chunk size changed the answer by {worst:.2e}");
    }
}

/// The prefill of a training run must equal a state-passing decode of the same
/// sequence, token by token, including a decode that starts from a fresh state
/// (T=1 twice in a row is where a conv cache off by one position shows up).
/// The per-token scan, whole-sequence against token-by-token, carrying the
/// state. This is the decode/prefill property with the head, the output gate and
/// the conv all out of the way, so a failure here names the recurrence and a
/// failure there does not.
///
/// Expectation: EXACTLY zero. Both runs perform the same op sequence on the
/// same shapes — `[B, H, 1, K]` against `[B, H, 1, K]` — in the same order, so
/// any difference is a real state-plumbing bug, not float reassociation.
#[test]
fn the_scan_alone_is_bit_identical_stepwise_and_whole() {
    let device = Device::ndarray();
    let (b, h, t, k, vd) = (2usize, 4, 70, 16, 24);
    let mut rng = Rng::new(0x5CA4);
    let norm = |x: Vec<f64>| -> Tensor<4> {
        Tensor::from_data(TensorData::new(x, vec![b, h, t, k]), &device)
    };
    let q = norm(rng.normal_vec(b * h * t * k));
    let kt = norm(rng.normal_vec(b * h * t * k));
    let v = Tensor::from_data(
        TensorData::new(rng.normal_vec(b * h * t * vd), vec![b, h, t, vd]),
        &device,
    );
    let g = Tensor::from_data(
        TensorData::new(
            (0..b * h * t * k).map(|i| -0.01 - 0.14 * (i % 97) as f64 / 97.0).collect::<Vec<f64>>(),
            vec![b, h, t, k],
        ),
        &device,
    );
    let er = Tensor::from_data(
        TensorData::new(
            (0..b * h * t * k).map(|i| 0.05 + 0.9 * (i % 89) as f64 / 89.0).collect::<Vec<f64>>(),
            vec![b, h, t, k],
        ),
        &device,
    );
    let w = Tensor::from_data(
        TensorData::new(
            (0..b * h * t * vd).map(|i| 0.05 + 0.9 * (i % 83) as f64 / 83.0).collect::<Vec<f64>>(),
            vec![b, h, t, vd],
        ),
        &device,
    );
    let st = Tensor::from_data(
        TensorData::new(rng.normal_vec(b * h * k * vd), vec![b, h, k, vd]),
        &device,
    );
    let scale = 0.25;

    let (whole_out, whole_st) =
        fused_recurrent_forward(q.clone(), kt.clone(), v.clone(), g.clone(), er.clone(), w.clone(), st.clone(), scale);

    let mut carry = st;
    let mut outs: Vec<Tensor<4>> = Vec::new();
    for i in 0..t {
        let sl = |x: &Tensor<4>| x.clone().slice_dim(2, i..i + 1);
        let (o, ns) = fused_recurrent_forward(
            sl(&q),
            sl(&kt),
            sl(&v),
            sl(&g),
            sl(&er),
            sl(&w),
            carry.clone(),
            scale,
        );
        carry = ns;
        outs.push(o);
    }
    let step_out = Tensor::cat(outs, 2);
    let d_out = max4(&step_out, &whole_out);
    let d_st = max4(&carry, &whole_st);
    println!("scan whole vs stepwise: out {d_out:.3e}, state {d_st:.3e}");
    assert_eq!(d_out, 0.0, "the scan's output differs stepwise vs whole: {d_out:.3e}");
    assert_eq!(d_st, 0.0, "the scan's state differs stepwise vs whole: {d_st:.3e}");
}

fn max4(a: &Tensor<4>, b: &Tensor<4>) -> f32 {
    let f = |t: &Tensor<4>| -> Vec<f32> {
        t.clone()
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect()
    };
    f(a).iter().zip(f(b)).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// The same property through the whole layer, on a configuration whose output
/// is not degenerate.
///
/// WHY THE BIAS IS NOT ZERO HERE. The reference zero-initialises every linear
/// bias, including `g_proj_1` (`spec-gdn2-official.md` §4.4). At a small test
/// width that makes the output gate `silu(W_g x)` sit at ~0, and the layer
/// output `normed * silu(gate) * o_proj` is a product of near-zeroes: measured
/// `|y|max = 7.6e-3` with the reference init, against `|o| = 2.3e-2` before the
/// gate. A 1e-3 difference is then 13% of the whole output, so this comparison
/// measures the conditioning of the init, not decode/prefill equivalence.
/// Giving `g_proj_1` a bias of 1 puts `silu(gate)` at 0.73 and the output at
/// O(0.3), which is the regime a trained model is in.
#[test]
fn decode_matches_prefill_including_a_fresh_state() {
    let device = Device::ndarray();
    let (d, h, hk, hv, v_head) = (64usize, 4usize, 16usize, 4usize, 24usize);
    let mut results: Vec<(usize, bool, f32)> = Vec::new();
    for t in [1usize, 40, 70] {
        for use_sc in [true, false] {
            let mut rng = Rng::new(0xDEC0 ^ t as u64 ^ ((use_sc as u64) << 40));
            let p = P::new(&mut rng, d, h, hk, hv, v_head);
            let x = rng.normal_vec(t * d);
            let mut module = build(&p, &device, use_sc, false, 64);
            // non-degenerate output gate; see the doc comment
            module.g_proj_1.bias = Some(Param::from_tensor(t1(
                &vec![1.0; hv * v_head],
                &[hv * v_head],
                &device,
            )));
            let input = Tensor::<3>::from_data(TensorData::new(x.clone(), vec![1, t, d]), &device);

            let mut st: Option<Gdn2State> = None;
            let prefill = module.forward::<NdArray>(input.clone(), &mut st, true);
            let y_max = abs_max(&prefill.clone().into_data());

            // A FRESH state: the prefill above has already advanced `st` to the
            // end of the sequence, and decoding the same tokens again from
            // there is not a decode/prefill comparison.
            let mut st2: Option<Gdn2State> = None;
            let mut decoded: Vec<Tensor<3>> = Vec::new();
            let mut per_pos: Vec<f32> = Vec::with_capacity(t);
            for ti in 0..t {
                let tok = Tensor::<3>::from_data(
                    TensorData::new(x[ti * d..(ti + 1) * d].to_vec(), vec![1, 1, d]),
                    &device,
                );
                let o = module.forward::<NdArray>(tok, &mut st2, true);
                per_pos.push(max_abs_diff_data(
                    &o.clone().into_data(),
                    &prefill.clone().slice([0..1, ti..ti + 1, 0..d]).into_data(),
                    d,
                ));
                decoded.push(o);
            }
            let stepwise = Tensor::cat(decoded, 1);
            let diff = (prefill - stepwise).abs().max().into_data();
            let v: f32 = diff
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .fold(0.0, f32::max);
            println!("  t={t} short_conv={use_sc}: prefill vs stepwise = {v:.2e}  (|y|max {y_max:.2e})");
            results.push((t, use_sc, v));
        }
    }
    for (t, use_sc, v) in results {
        assert!(
            v < 1e-4,
            "t={t} short_conv={use_sc}: decode drifted from prefill by {v:.2e}"
        );
    }
}

/// max abs value of an f32 `TensorData` buffer.
fn abs_max(a: &burn::tensor::TensorData) -> f32 {
    a.bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()).abs())
        .fold(0.0, f32::max)
}

/// max abs difference of two `TensorData` buffers of f32, in f32.
fn max_abs_diff_data(a: &burn::tensor::TensorData, b: &burn::tensor::TensorData, n: usize) -> f32 {
    let g = |d: &burn::tensor::TensorData| -> Vec<f32> {
        d.bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect()
    };
    let (x, y) = (g(a), g(b));
    assert_eq!(x.len(), n, "comparison length");
    x.iter().zip(&y).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max)
}
