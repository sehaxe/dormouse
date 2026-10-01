// The FUSED KERNEL adjoint against the f64 gradient oracle.
//
// # WHAT THIS IS, AND WHY IT IS NOT `fused_adjoint_vs_ops.rs`
//
// `tests/fused_adjoint_vs_ops.rs` compares the fused kernels' gradients with
// burn's autograd over our own ops path.  That crosses an algorithm boundary
// and it is a real test, but BOTH sides are our forward, differentiated by the
// same framework, so a misreading of the specification is symmetric across them
// and cancels (`docs/protocols/ORACLE.md` §2).  Its reference is tier (d).
//
// This file's references are `tests/ref_bwd_f64.bin` (one chunk) and
// `tests/ref_bwd_f64_carry.bin` (two chunks): forward-mode AD in f64 over a
// transcription of the forward, cross-checked inside the generator against
// full-tensor central finite differences.  Those two agree to 1.2e-12 and
// 2.3e-12 respectively, and the forward itself is asserted against the same
// fixtures by `tests/autodiff_bwd_f64.rs`.  A wrong adjoint, a missing gradient,
// a dropped term, a transposed contraction, a sign error and a gradient of a
// different function ALL fail this and none of them fails an arm-vs-arm
// comparison.
//
// # THE DEFECT THIS FILE IS POINTED AT
//
// `f737710` moved the `note_fused_backward` counter from the first statement of
// `fused_chunk_backward` to after every gate, so the CUDA gate test's `bwd > 0`
// stopped being satisfiable by a backward that returned `None` on the next
// line — and it was left RED on purpose until a real comparison landed.  This
// is that comparison.  If the fused adjoint is right, the counter it asserts is
// finally earned; if it is wrong, the number is here.
//
// # SCOPE: BK1 AND BK2, AND WHY IT WASN'T ALWAYS
//
// This file runs TWICE, at one chunk and at two, from two committed f64
// fixtures.  At one chunk (`ref_bwd_f64.bin`) it exercises BK1 — the
// token-parallel intra-chunk adjoint — on all seven gradients, on all
// coordinates.  At two chunks (`ref_bwd_f64_carry.bin`, T=32) it also exercises
// BK2, the sequential reverse recurrence over the chunks, and the
// `d_k_bptt` / `d_e_bptt` / `d_s_shift` glue.  Those are exactly the terms
// `2a430cc` measured at rel 3.3e-1 (d_k) and 6.2e-1 (d_g) before its fix.
//
// It used to run at one chunk only, and the header said why: the two-chunk
// forward "does not match the f64 transcription at all (measured 8.3e-1)".  That
// reason was WRONG and was retracted by `c305ec5` (09-30 18:17), which located
// the fault in `tools/gen_bwd_f64.py:258` — `np.exp(g_last - G)` where
// `g_last` was already `exp(G_last)`, a double exp in the REFERENCE, no
// production code involved.  After the fix: chunk 1 5.603e-01 -> 3.327e-08,
// final state 8.930e-01 -> 2.621e-07.  `c305ec5` regenerated the two-chunk
// fixture and used it for the state gate; nobody pointed this file at it, so
// the ungated half of the backward kept a green gate and a scope note that read
// like a measurement.
//
// **Do not read a green run of the ONE-chunk arm as "the fused backward is
// correct"** — that is the mistake `d8fa449` inherited, filing a 16-chunk
// disagreement against a kernel that had only ever been checked at one chunk.
//
// # RUN
//
//   cargo test -p burn-gdn2 --features cuda,autodiff --test fused_adjoint_f64 -- --nocapture
#![cfg(all(feature = "cuda", feature = "autodiff"))]
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]

use std::io::{Cursor, Read};

use burn::tensor::{Distribution, Tensor, TensorData};
use burn_gdn2::kernel::chunk_adjoint_cube::cuda::{fused_chunk_backward, FusedBackwardInputs};
use burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward_scratch;
use burn_gdn2::CudaBare;

const SCALE: f64 = 0.353_553_390_593_273_73;
const CHUNK: usize = 16;
const NAMES: [&str; 7] = ["q", "k", "v", "g", "b", "w", "state"];

/// Identical to `autodiff_bwd_f64.rs::GRAD_BAR`, and for the same reason: the
/// oracle's own error is 1.2e-12 and the code under test is f32, so the bar is
/// set by the f32 side. Measured worst input for the fused kernels, on this
/// box at this shape: see the table the test prints.
const GRAD_BAR: f32 = 1e-3;

struct Fixture {
    blocks: Vec<(String, Vec<usize>, Vec<f64>)>,
}

fn rd_u32(c: &mut Cursor<&[u8]>) -> u32 {
    let mut b = [0u8; 4];
    c.read_exact(&mut b).unwrap();
    u32::from_le_bytes(b)
}
fn rd_name(c: &mut Cursor<&[u8]>) -> String {
    let n = rd_u32(c) as usize;
    let mut b = vec![0u8; n];
    c.read_exact(&mut b).unwrap();
    String::from_utf8(b).unwrap()
}

fn load(data: &[u8]) -> Fixture {
    // `include_bytes!` yields `&[u8; N]`; `Cursor::new` over the SLICE is what
    // gives the `&[u8]` the rd_* readers take. Same shape as ref_f64.rs.
    let mut c = Cursor::new(data);
    let mut m = [0u8; 8];
    c.read_exact(&mut m).unwrap();
    assert_eq!(&m, b"GDN2BFD\0", "not a GDN2BFD fixture");
    let n = rd_u32(&mut c) as usize;
    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let name = rd_name(&mut c);
        let ndim = rd_u32(&mut c) as usize;
        let shape: Vec<usize> = (0..ndim).map(|_| rd_u32(&mut c) as usize).collect();
        let numel: usize = shape.iter().product();
        let mut v = vec![0f64; numel];
        let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, numel * 8) };
        c.read_exact(bytes).unwrap();
        blocks.push((name, shape, v));
    }
    let mut sp = [0u8; 8];
    c.read_exact(&mut sp).unwrap();
    Fixture { blocks }
}

impl Fixture {
    fn get(&self, name: &str) -> (Vec<usize>, Vec<f64>) {
        let (_, s, v) = self
            .blocks
            .iter()
            .find(|(n, _, _)| n == name)
            .unwrap_or_else(|| panic!("fixture has no block {name:?}"));
        (s.clone(), v.clone())
    }
    fn input(&self, name: &str, dev: &burn::tensor::Device) -> Tensor<4> {
        let (shape, v) = self.get(name);
        let v: Vec<f32> = v.iter().map(|x| *x as f32).collect();
        Tensor::<4>::from_data(TensorData::new(v, shape), dev)
    }
}

fn vals(t: &Tensor<4>) -> Vec<f64> {
    t.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
        .collect()
}

fn rel(a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "shape mismatch in the comparison");
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        worst = worst.max((x - y).abs());
        scale = scale.max(x.abs()).max(y.abs());
    }
    (worst / scale.max(1e-300), scale)
}

/// The seven fused-kernel gradients against the f64 oracle, per term, on every
/// coordinate, at ONE chunk and at TWO.
///
/// Run: `cargo test -p burn-gdn2 --features cuda,autodiff --test fused_adjoint_f64 -- --nocapture`
///
/// TWO arms, from two committed fixtures, because they are two different
/// functions and the difference is the whole point:
///
/// * `ref_bwd_f64.bin` — T=16, chunk=16, ONE chunk. This is BK1, the
///   token-parallel intra-chunk adjoint, and nothing else. It has been green
///   since `814847b`.
/// * `ref_bwd_f64_carry.bin` — T=32, chunk=16, TWO chunks. This is BK1 **and
///   BK2**, the sequential reverse recurrence over the state, **and the
///   `d_k_bptt` / `d_e_bptt` / `d_s_shift` glue**. That is the half where the
///   terms `2a430cc` measured at rel 3.3e-1 (d_k) and 6.2e-1 (d_g) lived.
///
/// The second arm is what this file could not have when it was written: its own
/// header gave the reason — "the forward's chunk carry does not match the f64
/// transcription on a two-chunk sequence at all (measured 8.3e-1)". That reason
/// was RETRACTED by `c305ec5` (09-30 18:17), which found the double-`exp` in
/// `tools/gen_bwd_f64.py:258` — one line of the REFERENCE, no production code
/// changed — and took chunk 1 from 5.603e-01 to 3.327e-08. The scope was never
/// widened afterwards, so for 5h38m the ungated half was described as blocked
/// while its oracle sat on disk regenerated and unused.
#[test]
fn the_fused_kernels_agree_with_the_f64_oracle_term_by_term() {
    for (label, data) in [
        (
            "ONE CHUNK (BK1 only)",
            &include_bytes!("ref_bwd_f64.bin")[..],
        ),
        (
            "TWO CHUNKS (BK1 + BK2 + the d_s_shift glue)",
            &include_bytes!("ref_bwd_f64_carry.bin")[..],
        ),
    ] {
        run(label, data);
    }
}

fn run(label: &str, data: &[u8]) {
    let f = load(data);
    let dev: burn::tensor::Device = Default::default();
    let inp: [Tensor<4>; 7] = std::array::from_fn(|i| f.input(NAMES[i], &dev));
    let (_, d_out) = f.get("d_out");
    let d_out = Tensor::<4>::from_data(
        TensorData::new(d_out.iter().map(|x| *x as f32).collect(), f.get("d_out").0),
        &dev,
    );

    // The FORWARD first: if the fused forward is a different function from the
    // f64 transcription, every gradient below is a statement about nothing.
    // `autodiff_bwd_f64.rs` proves that for the ops path; this proves it for
    // the kernels, and it is the arm whose exports the adjoint consumes.
    let (fused_out, _state, io) = fused_chunk_forward_scratch::<CudaBare>(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        SCALE,
        CHUNK,
    )
    .expect("the fused forward must engage on the bare CUDA backend");
    let (_, out_ref) = f.get("out");
    let (fwd, fwd_scale) = rel(&vals(&fused_out), &out_ref);
    println!("\n=== {label} ===\nfused forward vs the f64 transcription: rel={fwd:.3e} (output scale {fwd_scale:.3e})");
    assert!(
        fwd < 1e-3,
        "[{label}] the fused forward is {fwd:.3e} from the f64 transcription. The adjoint below \
         differentiates the exported buffers, so a disagreement here means the two sides \
         are different functions and the gradient numbers would be meaningless."
    );

    let fbi = FusedBackwardInputs {
        m_inv: io.m_inv,
        aqk: io.aqk,
        qgt: io.qgt,
        glast: io.glast,
        v_new: io.v_new,
        states: io.states,
        w: io.w,
        u: io.u,
        gexp: io.gexp,
    };
    let g = fused_chunk_backward::<CudaBare>(
        &fbi, &inp[1], &inp[2], &inp[4], &inp[5], &d_out, SCALE, CHUNK,
    )
    .expect("the fused adjoint must engage on the bare CUDA backend");
    let fused = [
        g.d_q.clone(),
        g.d_k.clone(),
        g.d_v.clone(),
        g.d_g.clone(),
        g.d_b.clone(),
        g.d_w.clone(),
        g.d_s.clone(),
    ];

    println!(
        "\nfused KERNEL adjoint (f32) against the f64 oracle, all coordinates\n\
         {:>6}  {:>13}  {:>13}  {:>11}  {:>9}",
        "input", "|oracle|max", "|fused|max", "rel", "verdict"
    );
    let mut worst = (0.0f64, "");
    for (i, name) in NAMES.iter().enumerate() {
        let (_, want) = f.get(&format!("d{name}"));
        let (r, scale) = rel(&vals(&fused[i]), &want);
        println!(
            "{name:>6}  {scale:>13.6e}  {:>13.6e}  {r:>11.3e}  {:>9}",
            vals(&fused[i]).iter().fold(0.0f64, |m, x| m.max(x.abs())),
            if r < GRAD_BAR as f64 { "ok" } else { "WRONG" }
        );
        if r > worst.0 {
            worst = (r, name);
        }
    }
    println!(
        "\noracle self-consistency: forward-mode AD vs central differences, ~1e-12 relative\n\
         our bar: {GRAD_BAR:.0e} (the f32 side).  SCOPE of this arm: {label}."
    );
    assert!(
        worst.0 < GRAD_BAR as f64,
        "[{label}] the fused kernel adjoint is wrong on input {}: rel={:.3e} against the f64 \
         oracle (BAR {GRAD_BAR:.0e}). This is a KERNEL number against a gradient computed by \
         a different method, not a tolerance question.",
        worst.1,
        worst.0
    );
    let _ = (Distribution::Normal(0.0, 1.0),);
}
