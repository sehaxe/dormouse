#!/usr/bin/env -S rustc --edition 2021 -O
//! Regenerate `tests/ref_data.bin`, the reference fixture behind
//! `tests/bit_exact.rs` and `tests/test_chunk.rs`.
//!
//! WHAT THIS IS. A line-for-line port of `tests/gen_reference.py`, which is
//! itself a pure-PyTorch transcription of the original authors' layer:
//! NVlabs/GatedDeltaNet-2, `lit_gpt/gdn2.py`, with the Triton `fused_recurrent`
//! kernel replaced by an equivalent per-token scan. The Python script stays in
//! the tree because it is the reviewable statement of the math; this file is
//! the executable one, because a fixture nobody can regenerate is not a
//! reproducibility claim (see `tests/bit_exact.rs`).
//!
//! WHY RUST AND NOT THE PYTHON. The project has no torch dependency, torch
//! cannot be pinned for a stranger, and a 7 MB fixture produced by an unpinned
//! torch is exactly the "reproducible on one machine only" failure this is
//! meant to fix. This program is std-only: `rustc` builds it, no cargo
//! workspace, no lockfile, no Python.
//!
//! PRECISION AND REPRODUCIBILITY. All arithmetic is f32, sequential (no
//! parallel reductions, no fused-multiply-add reassociation, no thread-count
//! dependence). The RNG is splitmix64 + Box-Muller, seeded 1337, so the output
//! is BYTE-IDENTICAL on any platform, any core count, any rustc version. The
//! fixture is a transcription, not the original authors' bytes, so it is not
//! bit-equal to what torch would emit; the honest statement is that the
//! transcription noise is ~1e-6 and the test tolerance is 5e-4 (see the doc
//! comment on `tests/bit_exact.rs`).
//!
//! USAGE.
//!     cd crates/burn-gdn2
//!     rustc --edition 2021 -O tools/gen_reference.rs -o /tmp/gen_ref
//!     /tmp/gen_ref                       # writes tests/ref_data.bin
//!     git diff --exit-code -- tests/ref_data.bin   # must be empty
//!
//! The last two lines are a CI job: if regenerating ever changes a byte, the
//! fixture and the code that produced it have drifted apart, which is the one
//! failure this file exists to make impossible to miss.

use std::io::{BufWriter, Write};

// --- Configuration (fixed; matches the burn-gdn2 test matrix) ---------------
const D: usize = 64;
const H: usize = 4;
const HK: usize = 16;
const HV: usize = 4;
const EXPAND_V: f32 = 1.5;
const USE_SHORT_CONV: bool = true;
const ALLOW_NEG_EIGVAL: bool = false;
const N_CASES: usize = 1000;
const EPS: f32 = 1e-5;

const KD: usize = H * HK;
const V_HEAD: usize = (HK as f32 * EXPAND_V) as usize; // int(HK * EXPAND_V)
const VD: usize = HV * V_HEAD;

// --- RNG: splitmix64 + Box-Muller. No deps, no platform dependence. --------
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
    /// Uniform in [0, 1), 24 bits of entropy.
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    /// Standard normal, Box-Muller. `1.0 - unit()` keeps u1 in (0, 1].
    fn normal(&mut self) -> f32 {
        let u1 = 1.0 - self.unit();
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn normal_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal()).collect()
    }
    /// torch's `xavier_uniform_(w, gain)` for a torch-layout [out, in] weight.
    fn xavier(&mut self, out_f: usize, in_f: usize, gain: f32) -> Vec<f32> {
        let bound = gain * (6.0 / (in_f + out_f) as f32).sqrt();
        (0..out_f * in_f).map(|_| self.uniform(-bound, bound)).collect()
    }
}

// --- Tensor helpers. Torch layout [out, in] for Linear weights. -------------
/// y[t, o] = sum_i w[o, i] * x[t, i] + b[o];  x is [t, in_f], y is [t, out_f].
fn linear(x: &[f32], w: &[f32], b: Option<&[f32]>, in_f: usize, out_f: usize) -> Vec<f32> {
    let t = x.len() / in_f;
    let mut y = vec![0f32; t * out_f];
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    // torch's softplus: log1p(exp(x)) with the 20.0 threshold
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
/// Causal depthwise conv, kernel 4, replicate padding (first token), then SiLU.
/// `x` is [t, ch], `w` is [ch, 4] (the torch [ch, 1, 4] weight flattened).
fn short_conv(x: &[f32], w: &[f32], ch: usize) -> Vec<f32> {
    let t = x.len() / ch;
    let mut out = vec![0f32; t * ch];
    for ti in 0..t {
        for c in 0..ch {
            let mut acc = 0f32;
            for i in 0..4 {
                let src = if ti + i < 3 { 0 } else { ti + i - 3 };
                acc += x[src * ch + c] * w[c * 4 + i];
            }
            out[ti * ch + c] = silu(acc);
        }
    }
    out
}
/// F.normalize(x, p=2, dim=-1) along the last axis of a head-major [t, n] slice.
fn l2_normalize_last(x: &mut [f32], n: usize) {
    for row in x.chunks_mut(n) {
        let ss: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / ss.max(1e-12).sqrt();
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

// --- The reference layer. One forward, batch 1. ----------------------------
struct Params {
    q_proj: Vec<f32>,   // [KD, D]
    k_proj: Vec<f32>,   // [KD, D]
    v_proj: Vec<f32>,   // [VD, D]
    f_proj_0: Vec<f32>, // [V_HEAD, D]
    f_proj_1: Vec<f32>, // [KD, V_HEAD]
    b_proj: Vec<f32>,   // [KD, D]
    w_proj: Vec<f32>,   // [VD, D]
    g_proj_0: Vec<f32>, // [V_HEAD, D]
    g_proj_1: Vec<f32>, // [VD, V_HEAD]
    g_proj_1_b: Vec<f32>, // [VD]
    a_log: Vec<f32>,    // [H]
    dt_bias: Vec<f32>,  // [KD]
    o_norm_w: Vec<f32>, // [V_HEAD]
    o_proj: Vec<f32>,   // [D, VD]
    q_conv_w: Vec<f32>, // [KD, 4]
    k_conv_w: Vec<f32>, // [KD, 4]
    v_conv_w: Vec<f32>, // [VD, 4]
}

fn init_params(rng: &mut Rng) -> Params {
    // torch module order: q, k, v, f_proj.0, f_proj.1, b, w, g_proj.0, g_proj.1, o
    let lin = |rng: &mut Rng, out_f: usize, in_f: usize| rng.xavier(out_f, in_f, 2f32.powf(-2.5));
    let q_proj = lin(rng, KD, D);
    let k_proj = lin(rng, KD, D);
    let v_proj = lin(rng, VD, D);
    let f_proj_0 = lin(rng, V_HEAD, D);
    let f_proj_1 = lin(rng, KD, V_HEAD);
    let b_proj = lin(rng, KD, D);
    let w_proj = lin(rng, VD, D);
    let g_proj_0 = lin(rng, V_HEAD, D);
    let g_proj_1 = lin(rng, VD, V_HEAD);
    let o_proj = lin(rng, D, VD);
    // nn.init.zeros_ on the only bias in the module
    let g_proj_1_b = vec![0f32; VD];

    let a_log: Vec<f32> = (0..H).map(|_| rng.uniform(1.0, 16.0).ln()).collect();

    // dt = exp(u * (ln 0.1 - ln 0.001) + ln 0.001).clamp(min=1e-4)
    let dt_bias: Vec<f32> = (0..KD)
        .map(|_| {
            let u = rng.unit();
            let dt = (u * (0.1f32.ln() - 0.001f32.ln()) + 0.001f32.ln()).exp().max(1e-4);
            dt + (-(-dt).exp_m1()).ln() // dt + log(-expm1(-dt))
        })
        .collect();

    let o_norm_w = vec![1f32; V_HEAD];
    let conv = |rng: &mut Rng, ch: usize| (0..ch * 4).map(|_| rng.uniform(-0.5, 0.5)).collect::<Vec<f32>>();
    let q_conv_w = conv(rng, KD);
    let k_conv_w = conv(rng, KD);
    let v_conv_w = conv(rng, VD);

    Params {
        q_proj,
        k_proj,
        v_proj,
        f_proj_0,
        f_proj_1,
        b_proj,
        w_proj,
        g_proj_0,
        g_proj_1,
        g_proj_1_b,
        a_log,
        dt_bias,
        o_norm_w,
        o_proj,
        q_conv_w,
        k_conv_w,
        v_conv_w,
    }
}

/// The reference forward: x is [T, D], y is [T, D].
fn forward(p: &Params, x: &[f32]) -> Vec<f32> {
    let t = x.len() / D;

    // project, short-conv (or bare SiLU), then per-head L2 normalize
    let mut q = linear(x, &p.q_proj, None, D, KD);
    let mut k = linear(x, &p.k_proj, None, D, KD);
    let mut v = linear(x, &p.v_proj, None, D, VD);
    for t_ in q.iter_mut() {
        *t_ = silu(*t_);
    }
    for t_ in k.iter_mut() {
        *t_ = silu(*t_);
    }
    for t_ in v.iter_mut() {
        *t_ = silu(*t_);
    }
    if USE_SHORT_CONV {
        q = short_conv(&q, &p.q_conv_w, KD);
        k = short_conv(&k, &p.k_conv_w, KD);
        v = short_conv(&v, &p.v_conv_w, VD);
    }
    l2_normalize_last(&mut q, HK);
    l2_normalize_last(&mut k, HK);

    // channel-wise log-decay (f32), as in the official layer
    let f = linear(x, &p.f_proj_0, None, D, V_HEAD);
    let f = linear(&f, &p.f_proj_1, None, V_HEAD, KD);
    let a_exp: Vec<f32> = p.a_log.iter().map(|a| a.exp()).collect(); // [H]
    let g: Vec<f32> = f
        .iter()
        .enumerate()
        .map(|(n, fx)| {
            let c = n % KD; // channel index within [T, KD]
            -a_exp[c / HK] * softplus(fx + p.dt_bias[c])
        })
        .collect();

    let b_raw = linear(x, &p.b_proj, None, D, KD);
    let b: Vec<f32> = b_raw.iter().map(|x| sigmoid(*x)).collect();
    let w_raw = linear(x, &p.w_proj, None, D, VD);
    let w_gate: Vec<f32> = w_raw.iter().map(|x| sigmoid(*x)).collect();

    // GVA: repeat the key-side heads when there are more value heads than
    // query heads (HV % H == 0, as in the official layer).
    let rep = HV / H;
    assert!(HV % H == 0, "GVA needs HV % H == 0");
    // head-major [H, T, n] -> [HV, T, n], value head hv reads query head hv / rep
    let expand = |src: &[f32], n_per_head: usize| -> Vec<f32> {
        if rep == 1 {
            return src.to_vec();
        }
        let mut out = vec![0f32; t * HV * n_per_head];
        for hv in 0..HV {
            let from_h = hv / rep;
            for ti in 0..t {
                for i in 0..n_per_head {
                    out[(hv * t + ti) * n_per_head + i] = src[(from_h * t + ti) * n_per_head + i];
                }
            }
        }
        out
    };
    let q = expand(&q, HK);
    let k = expand(&k, HK);
    let g = expand(&g, HK);
    let mut b = expand(&b, HK);

    // heads: [HV, T, HK] and [HV, T, V_HEAD], head-major
    let at = |a: &[f32], h: usize, ti: usize, i: usize, n: usize| a[h * t * n + ti * n + i];

    if ALLOW_NEG_EIGVAL {
        for x in b.iter_mut() {
            *x *= 2.0;
        }
    }

    let scale = (HK as f32).powf(-0.5);
    let mut s = vec![0f32; HV * HK * V_HEAD]; // [HV, HK, V_HEAD]
    let mut o = vec![0f32; t * VD]; // [T, HV, V_HEAD]

    for ti in 0..t {
        for h in 0..HV {
            for kk in 0..HK {
                let decay = at(&g, h, ti, kk, HK).exp();
                for vv in 0..V_HEAD {
                    s[(h * HK + kk) * V_HEAD + vv] *= decay;
                }
            }
            for vv in 0..V_HEAD {
                let mut erased = 0f32;
                for kk in 0..HK {
                    erased += s[(h * HK + kk) * V_HEAD + vv] * at(&b, h, ti, kk, HK) * at(&k, h, ti, kk, HK);
                }
                let v_new = at(&w_gate, h, ti, vv, V_HEAD) * at(&v, h, ti, vv, V_HEAD) - erased;
                for kk in 0..HK {
                    s[(h * HK + kk) * V_HEAD + vv] += at(&k, h, ti, kk, HK) * v_new;
                }
            }
            for vv in 0..V_HEAD {
                let mut acc = 0f32;
                for kk in 0..HK {
                    acc += s[(h * HK + kk) * V_HEAD + vv] * at(&q, h, ti, kk, HK);
                }
                o[ti * VD + h * V_HEAD + vv] = acc * scale;
            }
        }
    }

    // SiLU-gated RMS norm per head, then the output projection
    let gate = linear(x, &p.g_proj_0, None, D, V_HEAD);
    let gate = linear(&gate, &p.g_proj_1, Some(&p.g_proj_1_b), V_HEAD, VD);
    for h in 0..HV {
        for ti in 0..t {
            let row = &mut o[ti * VD + h * V_HEAD..ti * VD + (h + 1) * V_HEAD];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let rms = (ss / V_HEAD as f32 + EPS).sqrt();
            for (i, val) in row.iter_mut().enumerate() {
                *val = *val / rms * p.o_norm_w[i] * silu(gate[ti * VD + h * V_HEAD + i]);
            }
        }
    }
    linear(&o, &p.o_proj, None, VD, D)
}

// --- Serialization (the format consumed by tests/bit_exact.rs) -------------
fn w_i32(f: &mut impl Write, v: i32) {
    f.write_all(&v.to_le_bytes()).unwrap();
}
fn w_name(f: &mut impl Write, name: &str) {
    w_i32(f, name.len() as i32);
    f.write_all(name.as_bytes()).unwrap();
}
/// Named tensor. `linear_w` tensors are torch-layout [out, in] and are stored
/// transposed ([in, out]) because burn's `Linear` weight is [in, out].
fn w_tensor(f: &mut impl Write, name: &str, t: &[f32], shape: &[usize]) {
    w_name(f, name);
    w_i32(f, shape.len() as i32);
    w_i32(f, t.len() as i32);
    for s in shape {
        w_i32(f, *s as i32);
    }
    f.write_all(bytemuck_f32(t)).unwrap();
}
fn w_linear(f: &mut impl Write, name: &str, w: &[f32], out_f: usize, in_f: usize) {
    // stored [in, out]: element (i, o) = w[o, i]
    let mut t = vec![0f32; out_f * in_f];
    for i in 0..in_f {
        for o in 0..out_f {
            t[i * out_f + o] = w[o * in_f + i];
        }
    }
    w_tensor(f, name, &t, &[in_f, out_f]);
}
fn w_raw(f: &mut impl Write, t: &[f32], shape: &[usize]) {
    w_i32(f, shape.len() as i32);
    w_i32(f, t.len() as i32);
    for s in shape {
        w_i32(f, *s as i32);
    }
    f.write_all(bytemuck_f32(t)).unwrap();
}
fn bytemuck_f32(t: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(t.as_ptr() as *const u8, t.len() * 4) }
}

fn main() {
    let out_path = std::env::args().nth(1).unwrap_or_else(|| "tests/ref_data.bin".to_string());
    let f = std::fs::File::create(&out_path).unwrap_or_else(|e| panic!("{out_path}: {e}"));
    let mut f = BufWriter::new(f);

    let mut rng = Rng::new(1337);
    let p = init_params(&mut rng);

    w_i32(&mut f, D as i32);
    w_i32(&mut f, H as i32);
    w_i32(&mut f, HK as i32);
    w_i32(&mut f, HV as i32);
    w_i32(&mut f, (EXPAND_V * 10.0) as i32);
    f.write_all(&[USE_SHORT_CONV as u8, ALLOW_NEG_EIGVAL as u8]).unwrap();

    w_linear(&mut f, "q_proj", &p.q_proj, KD, D);
    w_linear(&mut f, "k_proj", &p.k_proj, KD, D);
    w_linear(&mut f, "v_proj", &p.v_proj, VD, D);
    w_linear(&mut f, "f_proj_0", &p.f_proj_0, V_HEAD, D);
    w_linear(&mut f, "f_proj_1", &p.f_proj_1, KD, V_HEAD);
    w_linear(&mut f, "b_proj", &p.b_proj, KD, D);
    w_linear(&mut f, "w_proj", &p.w_proj, VD, D);
    w_linear(&mut f, "g_proj_0", &p.g_proj_0, V_HEAD, D);
    w_linear(&mut f, "g_proj_1_w", &p.g_proj_1, VD, V_HEAD);
    w_tensor(&mut f, "g_proj_1_b", &p.g_proj_1_b, &[VD]);
    w_tensor(&mut f, "A_log", &p.a_log, &[H]);
    w_tensor(&mut f, "dt_bias", &p.dt_bias, &[KD]);
    w_tensor(&mut f, "o_norm_w", &p.o_norm_w, &[V_HEAD]);
    w_linear(&mut f, "o_proj", &p.o_proj, D, VD);
    w_tensor(&mut f, "q_conv_w", &p.q_conv_w, &[KD, 4]);
    w_tensor(&mut f, "k_conv_w", &p.k_conv_w, &[KD, 4]);
    w_tensor(&mut f, "v_conv_w", &p.v_conv_w, &[VD, 4]);

    w_i32(&mut f, N_CASES as i32);
    for i in 0..N_CASES {
        let seq_len = (1usize << (i % 6)) + (i % 7); // 2..70, varied
        let x = rng.normal_vec(seq_len * D);
        let y = forward(&p, &x);
        w_raw(&mut f, &x, &[1, seq_len, D]);
        w_raw(&mut f, &y, &[1, seq_len, D]);
    }
    f.flush().unwrap();
    println!("wrote {out_path} ({N_CASES} cases, d={D} h={H} hk={HK} hv={HV} expand_v={EXPAND_V})");
}
