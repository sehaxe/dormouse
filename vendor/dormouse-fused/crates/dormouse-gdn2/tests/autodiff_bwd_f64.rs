// The chunked-WY ADJOINT against an f64 gradient oracle, per term, and the
// whole-tensor kind of comparison that a sampled probe cannot make.
//
// # WHY THIS FILE EXISTS
//
// Every numeric test this crate had for the backward compared our code to our
// code. `tests/fused_chunk_verify.rs::fused_op_grads_match_tensor_path_cuda`
// claimed to compare "the fused-op backward" with the tensor path and did not:
// the fused branch was gated on the kill switch, so both sides of that pair
// were the TENSOR adjoint, tensor-vs-tensor, and its tolerance had never been
// exercised against a kernel. `tests/fused_adjoint_vs_ops.rs` fixed the arm but
// kept the same ceiling — its reference is burn's autograd over our own ops
// path, so a MISREADING of the specification is symmetric across both sides and
// cancels (`docs/protocols/ORACLE.md` §2).
//
// This file's reference is a gradient computed by a DIFFERENT METHOD: forward-
// mode AD in f64 (`tools/fwd_mode.py`), cross-checked inside the generator by
// full-tensor central finite differences, which agree with it to 2.16e-12
// relative. A wrong adjoint, a missing gradient, a dropped term, a transposed
// contraction, a sign error and a gradient of a different function all fail it.
// An arm-vs-arm comparison fails none of those.
//
// # TIER, HONESTLY
//
// Tier **(b)**, not (a). The forward these gradients differentiate is a
// TRANSCRIPTION (`tools/gen_bwd_f64.py`, transcribed from
// `src/forward.rs::chunk_wy_forward_batched` with a different matrix
// inversion), so a shared misreading of the specification would survive. It is
// not the authors' own bytes; there is no tier-(a) layer in this tree
// (`docs/protocols/ORACLE.md` §3) and this is not one. What it buys is a reference whose
// expected value does not come from a second implementation of the same
// DERIVATIVE — which is the property the previous tests lacked. The tolerances
// below are relative bars with a measured margin on BOTH sides, never a claim of
// exact reproduction: the oracle is 1.2e-12 from the implementation's own
// second method, and the implementation is 2.5e-7 from the oracle.
//
// The transcription risk is BOUNDED rather than assumed: `the_forward_agrees_
// with_the_f64_transcription` runs first and asserts our f32 forward against the
// f64 one, so every gradient below is the derivative of a function whose
// forward has been checked. A gradient gate whose forward does not match is a
// statement about nothing.
//
// # THE GATE'S POWER, MEASURED, AND IT IS NOT MORE THAN THIS
//
// A test that cannot fail is the defect class this project keeps rediscovering,
// so the failure was provoked and measured, by editing `ChunkWy::backward` in
// `src/autodiff.rs` and reading the table:
//
//   perturbation on `d_q` alone      q rel      verdict
//   x 1.01   (1% )                  9.901e-3   RED,  9.9x the bar
//   x 1.0001 (1e-4)                 1.000e-4   green
//   none                           1.352e-7   green
//
// with the other six inputs unmoved at 1.3e-7..2.5e-7 in every case, so the
// table LOCALISES the perturbation as well as detecting it. So the honest
// statement of what this gate is worth: **it detects a wrong adjoint at 1e-3
// relative and above, per input.** Below that it does not, and no claim is made
// that it does. The bar sits 4000x above this implementation's own f32 noise
// floor (2.5e-7) and 720x below the nearest wrong formula the fixture
// contains (7.2e-1), and both numbers are re-measured on every run.
//
// # WHAT IS COMPARED, AND WHY EVERY COORDINATE
//
// All 3200 coordinates of all seven inputs. `tests/autodiff_chunk.rs:200` and
// `tests/ops_batched_autodiff.rs` probe one or six coordinates at a 5% bar; a
// backward that is wrong at 99% of its coordinates and right at the probed one
// passes those (`docs/protocols/ORACLE.md` §6 B9). The whole tensor is 32x8x8 per
// token-side input at this shape, so there is no reason to sample.
//
// # THE FAR SIDE OF THE BAR IS DATA
//
// `tests/ref_bwd_f64_faults.bin` holds the same inputs and the same cotangent
// run through TEN wrong formulas, each differentiated by the same finite
// differences. `the_bar_separates_every_wrong_formula` measures how far our
// gradient is from each of them, from committed data, on every run — so the
// claim "this bar separates a wrong adjoint from the right one" cannot rot into
// a self-consistency check. A bare `rel < BAR` assertion cannot detect the
// failure it exists to prevent: a semantic error that moved down into the noise
// band would look like a pass.
//
// # RUN
//
//   cargo test -p dormouse-gdn2 --features autodiff --test autodiff_bwd_f64 -- --nocapture
//
// The CUDA half of the same comparison — the fused KERNELS against this same
// oracle — is `tests/fused_adjoint_f64.rs`, because the fused adjoint only
// exists on a bare CUDA backend and this file must stay runnable in the CPU
// gate.
//
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg(feature = "autodiff")]
#![allow(deprecated)]

use std::io::Cursor;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Device, Tensor, TensorData};
use dormouse_gdn2::{chunk_wy_forward, chunk_wy_forward_autodiff_s};
use burn_autodiff::checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing};

/// The crate's own default, `k.shape[-1] ** -0.5` = 1/sqrt(8) = 0.35355339059327373,
/// which is what the generator used.  A different scale is a different function
/// and the fixture would then be a gradient of something else.
const SCALE: f64 = 0.353_553_390_593_273_73;
const CHUNK: usize = 16;

const NAMES: [&str; 7] = ["q", "k", "v", "g", "b", "w", "state"];

/// Relative bar for the adjoint against the f64 oracle, per input.
///
/// DERIVED, not guessed, and the derivation is in `tools/gen_bwd_f64.py`'s
/// BAND section:
///
///  * the ORACLE's own error is 2.16e-12 relative — the measured disagreement
///    between forward-mode AD and full-tensor central differences at h = 1e-3,
///    which is inside the analytically derived band (truncation h^4/30 and
///    round-off 1.5 eps/h balance at h ~ 1.6e-3, giving ~3e-13; the measured
///    floor is ~7x that, the usual gap between "the derivative is O(1)" and
///    "the fifth derivative is O(1)");
///  * the implementation under test is f32, whose eps is 1.2e-7, and the
///    adjoint is a reassociation of an f32 forward over two chunk boundaries.
///
/// So the bar is set by the f32 side, five orders above the oracle's error and
/// four below the smallest semantic error the fault fixture contains. It is
/// NOT widened to accommodate a disagreement: if an input lands above it, that
/// is a finding about the adjoint, and the test says so.
const GRAD_BAR: f32 = 1e-3;

/// The forward's bar against the f64 transcription. f32 over two chunks and a
/// `1/E` contraction, so O(1e-6) is the arithmetic floor and 1e-3 is three
/// orders above it — while a semantic error in the forward is O(1). This is the
/// number that bounds the transcription risk for every gradient below.
const FWD_BAR: f32 = 1e-3;

/// A wrong formula must land at least this far from our gradient, or the bar
/// is not separating anything. 1e-1 relative: measured in
/// `the_bar_separates_every_wrong_formula`, and asserted from committed data on
/// every run so it cannot rot.
const SEMANTIC_FLOOR: f32 = 1e-1;

// --- fixture --------------------------------------------------------------
struct Fixture {
    blocks: Vec<(String, Vec<usize>, Vec<f64>)>,
    spread: f64,
}

fn load(data: &[u8], magic: &[u8; 8]) -> Fixture {
    let mut c = Cursor::new(data);
    let mut m = [0u8; 8];
    std::io::Read::read_exact(&mut c, &mut m).unwrap();
    assert_eq!(&m, magic, "fixture magic mismatch");
    let n = rd_u32(&mut c) as usize;
    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let name = rd_name(&mut c);
        let ndim = rd_u32(&mut c) as usize;
        let shape: Vec<usize> = (0..ndim).map(|_| rd_u32(&mut c) as usize).collect();
        let numel: usize = shape.iter().product();
        let mut v = vec![0f64; numel];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, numel * 8) };
        std::io::Read::read_exact(&mut c, bytes).unwrap();
        blocks.push((name, shape, v));
    }
    let mut sp = [0u8; 8];
    std::io::Read::read_exact(&mut c, &mut sp).unwrap();
    assert_eq!(c.position() as usize, data.len(), "trailing bytes in the fixture");
    Fixture {
        blocks,
        spread: f64::from_le_bytes(sp),
    }
}

fn rd_u32(c: &mut Cursor<&[u8]>) -> u32 {
    let mut b = [0u8; 4];
    std::io::Read::read_exact(c, &mut b).unwrap();
    u32::from_le_bytes(b)
}

fn rd_name(c: &mut Cursor<&[u8]>) -> String {
    let n = rd_u32(c) as usize;
    let mut b = vec![0u8; n];
    std::io::Read::read_exact(c, &mut b).unwrap();
    String::from_utf8(b).unwrap()
}

impl Fixture {
    fn get(&self, name: &str) -> (Vec<usize>, Vec<f64>) {
        let (n, s, v) = self
            .blocks
            .iter()
            .find(|(bn, _, _)| bn == name)
            .unwrap_or_else(|| panic!("fixture has no block {name:?}"));
        (s.clone(), v.clone())
    }
    fn inputs(&self) -> [Vec<f64>; 7] {
        std::array::from_fn(|i| self.get(NAMES[i]).1)
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        NAMES.iter().map(|n| self.get(n).0).collect()
    }
}

/// The seven op inputs as f32 tensors on `dev`, in the fixture's order.
fn tensors(f: &Fixture, dev: &Device) -> [Tensor<4>; 7] {
    std::array::from_fn(|i| {
        let (shape, v) = f.get(NAMES[i]);
        let v: Vec<f32> = v.iter().map(|x| *x as f32).collect();
        Tensor::<4>::from_data(TensorData::new(v, shape), dev)
    })
}

fn host(t: &Tensor<4>) -> Vec<f64> {
    t.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
        .collect()
}

/// `max|a-b| / max|a|`, the measure every other numeric test in this crate uses.
fn rel(a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "shape mismatch in the comparison");
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        worst = worst.max((x - y).abs());
        scale = scale.max(x.abs()).max(y.abs());
    }
    (worst / scale.max(1e-300), scale)
}

// --- the forward, checked before any gradient is looked at ----------------
/// Our f32 forward against the f64 transcription, on the bare backend.
///
/// This is the load-bearing assertion for the whole file. The oracle's expected
/// values are the derivative of the f64 forward; if our f32 forward is a
/// DIFFERENT function, every gradient below is a statement about nothing. So
/// this runs first, it is a separate `#[test]` (so a failure names the forward,
/// not the gradient), and the gradient test re-asserts it rather than assuming
/// it.
#[test]
fn the_forward_agrees_with_the_f64_transcription() {
    let f = Fixture::load_ref();
    let dev = Device::ndarray();
    let inp = tensors(&f, &dev);
    let (_, d_out) = f.get("d_out");
    let (out_shape, out_ref) = f.get("out");

    let (out, _new_state) = chunk_wy_forward(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        SCALE,
        CHUNK,
    );
    assert_eq!(out.dims().to_vec(), out_shape, "forward shape changed");
    let (r, scale) = rel(&host(&out), &out_ref);
    println!(
        "f32 forward vs f64 transcription: rel={r:.3e} (output scale {scale:.3e}, BAR {FWD_BAR:.0e})"
    );
    println!("loss <out,d_out> = {:.9e} (f64 {:.9e})", {
        let d: Vec<f32> = d_out.iter().map(|x| *x as f32).collect();
        let dt = Tensor::<4>::from_data(TensorData::new(d, out_shape.clone()), &dev);
        (out.clone() * dt).sum().into_scalar::<f32>()
    }, f.get("loss").1[0]);
    assert!(
        r < FWD_BAR as f64,
        "our f32 forward is {r:.3e} from the f64 transcription (BAR {FWD_BAR:.0e}). \
         Every gradient in this file is the derivative of the f64 forward, so a \
         disagreement here means the two sides are different functions and the \
         gradient comparisons below would be meaningless."
    );
}

// --- the gradient, per term ----------------------------------------------
/// The analytic adjoint — the custom node's `ChunkWy::backward` — against the
/// f64 oracle, on ALL coordinates of all seven inputs, and the per-term table.
///
/// Printed on every run, because the table IS the deliverable: a backward that
/// is wrong in one term and right in six is a different defect from one that is
/// wrong everywhere, and a single `worst < BAR` cannot tell them apart.
#[test]
fn the_adjoint_agrees_with_the_f64_oracle_term_by_term() {
    let f = Fixture::load_ref();
    let dev = Device::ndarray().autodiff();
    let inp = tensors(&f, &dev);
    let leaves: Vec<Tensor<4>> = inp.iter().cloned().map(|t| t.require_grad()).collect();
    let (out_shape, d_out) = f.get("d_out");
    let cot = Tensor::<4>::from_data(
        TensorData::new(d_out.iter().map(|x| *x as f32).collect(), out_shape),
        &dev,
    );

    let (out, _state) = chunk_wy_forward_autodiff_s::<NdArray, NoCheckpointing>(
        leaves[0].clone(),
        leaves[1].clone(),
        leaves[2].clone(),
        leaves[3].clone(),
        leaves[4].clone(),
        leaves[5].clone(),
        leaves[6].clone(),
        SCALE,
        CHUNK,
    )
    .expect("the custom node declined on lifted leaves: the arm under test never ran");

    let grads = (out * cot).sum().backward();

    println!(
        "\nper-term: the analytic adjoint (f32, custom node) against the f64 oracle\n\
         {:>6}  {:>13}  {:>13}  {:>11}  {:>9}",
        "input", "|oracle|max", "|adjoint|max", "rel", "verdict"
    );
    let mut worst = (0.0f64, "");
    for (i, name) in NAMES.iter().enumerate() {
        let (_, want) = f.get(&format!("d{name}"));
        let got = host(
            &leaves[i]
                .grad(&grads)
                .unwrap_or_else(|| panic!("input {name} got no gradient at all")),
        );
        let (r, scale) = rel(&got, &want);
        let verdict = if r < GRAD_BAR as f64 { "ok" } else { "WRONG" };
        println!(
            "{name:>6}  {scale:>13.6e}  {:>13.6e}  {r:>11.3e}  {verdict:>9}",
            got.iter().fold(0.0f64, |m, x| m.max(x.abs()))
        );
        if r > worst.0 {
            worst = (r, name);
        }
    }
    println!(
        "\noracle self-consistency: forward-mode AD vs central differences, \
         {0:.3e} relative (tools/gen_bwd_f64.py, h=1e-3)\n\
         our bar: {1:.0e}, which is the f32 side; the oracle's own error is 4 orders below it",
        f.spread, GRAD_BAR
    );
    assert!(
        worst.0 < GRAD_BAR as f64,
        "the adjoint is wrong on input {}: rel={:.3e} against the f64 oracle \
         (BAR {GRAD_BAR:.0e}). The per-term table above says which terms are \
         right; this is the adjoint's gradient, not a tolerance question.",
        worst.1, worst.0
    );
}

// --- the far side of the bar, from committed data ------------------------
/// Every wrong formula the fault fixture EXERCISES must be far from the oracle,
/// and the ones it cannot exercise are printed by name.
///
/// The distinction is measured, not assumed. `tools/gen_bwd_f64.py` writes the
/// faulty forward's LOSS next to each fault's gradients, so "this wrong formula
/// changes the answer" and "this wrong formula is invisible to this case" are
/// two different numbers rather than one. At a single chunk the state DECAY is
/// not observable at all — with one chunk the decay only reaches `S_out`, and
/// the loss never reads `S_out` — so `decay-sign`, `no-state-decay` and
/// `decay-on-v-new` are exactly 0.0 here. Reporting those as `1e-13 from the
/// oracle` would be a lie about coverage dressed as a pass; asserting they are
/// far would be a lie about the bar. So they are named, and the count of
/// exercised faults is asserted.
#[test]
fn the_bar_separates_every_wrong_formula() {
    let f = Fixture::load_ref();
    let faults = Fixture::load_faults();
    let base_loss = f.get("loss").1[0];
    let mut c = Cursor::new(faults.as_slice());
    let mut m = [0u8; 8];
    std::io::Read::read_exact(&mut c, &mut m).unwrap();
    assert_eq!(&m, b"GDN2BFD\0", "fault fixture magic");
    let n_faults = rd_u32(&mut c) as usize;
    let mut smallest = f64::INFINITY;
    let (mut exercised, mut inert) = (0usize, Vec::new());
    let mut worst: Option<(f64, String, String)> = None;
    for _ in 0..n_faults {
        let fname = rd_name(&mut c);
        let (_, fl) = read_tensor(&mut c);
        let moved = (fl[0] - base_loss).abs() / base_loss.abs();
        if moved < 1e-12 {
            inert.push((fname, moved));
            for _ in 0..7 {
                read_tensor(&mut c);
            }
            continue;
        }
        exercised += 1;
        // The unit is the FAULT, not the (fault, input) pair: a wrong formula
        // is caught if ANY of its seven gradients is far, and several of these
        // faults legitimately leave some gradients untouched (`no-intra` zeroes
        // dk/dv/db/dw exactly, and the eye is on the one that moves). Taking
        // the min over pairs would read those as `1e-13 from the oracle`, which
        // is a statement about the wrong number.
        let mut worst_here = (0.0f64, String::new());
        for _ in 0..7 {
            let (tname, v) = read_tensor(&mut c);
            let inp = tname.strip_prefix('d').unwrap_or(&tname).to_string();
            let (shape, want) = f.get(&tname);
            assert_eq!(shape, f.get(&format!("d{inp}")).0, "fault tensor shape differs");
            let (r, _) = rel(&v, &want);
            if r > worst_here.0 {
                worst_here = (r, inp);
            }
        }
        if worst_here.0 < smallest {
            smallest = worst_here.0;
            worst = Some((worst_here.0, fname.clone(), worst_here.1));
        }
    }
    assert_eq!(
        c.position() as usize,
        faults.len(),
        "trailing bytes in the fault fixture"
    );
    println!("\nwrong formulas, each differentiated by the same finite differences:");
    for (n, mv) in &inert {
        println!("  {n:>18}  NOT EXERCISED by this case: it moves the loss by {mv:.1e}");
    }
    println!(
        "\n  exercised: {exercised} of {n_faults} faults.  The CLOSEST any of them comes to the \
         oracle, over its best-matching gradient, is {smallest:.3e} relative; our bar is \
         {GRAD_BAR:.0e}, {:.0}x below that.  The oracle's own error is {:.1e}.",
        smallest / GRAD_BAR as f64,
        f.spread
    );
    if let Some((r, fname, inp)) = &worst {
        println!("  tightest fault: `{fname}`, whose d{inp} is {r:.3e} from the oracle");
    }
    assert!(
        exercised >= 7,
        "only {exercised} of {n_faults} wrong formulas change this case's loss at all, so \
         the bar is separating almost nothing. Inert: {inert:?}"
    );
    assert!(
        smallest > SEMANTIC_FLOOR as f64,
        "an EXERCISED wrong formula is only {smallest:.3e} from the oracle, below \
         SEMANTIC_FLOOR {SEMANTIC_FLOOR:.0e}: the bar is no longer separating a wrong adjoint \
         from the right one, whatever our adjoint happens to do"
    );
    assert!(
        GRAD_BAR as f64 > f.spread * 100.0,
        "the bar is not comfortably above the oracle's own error: bar {GRAD_BAR:.0e} vs \
         oracle spread {:.3e}. Widen the BAR, do not tighten the oracle.",
        f.spread
    );
}

fn read_tensor(c: &mut Cursor<&[u8]>) -> (String, Vec<f64>) {
    let name = rd_name(c);
    let ndim = rd_u32(c) as usize;
    let _shape: Vec<usize> = (0..ndim).map(|_| rd_u32(c) as usize).collect();
    let numel: usize = _shape.iter().product();
    let mut v = vec![0f64; numel];
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, numel * 8) };
    std::io::Read::read_exact(c, bytes).unwrap();
    (name, v)
}

// --- THE FINDING: the forward's chunk carry ------------------------------
/// Two chunks instead of one, against the same oracle at T = 2*chunk. This test
/// is RED, and it is red on purpose.
///
/// # What is measured, and what is established
///
/// With T = 16 = chunk the forward matches the f64 transcription to 1.4e-7. With
/// T = 32 = 2 chunks it matches to 8.3e-1. The difference is localised:
///
///   * chunk 0 agrees to ~3e-8, every token, every channel;
///   * chunk 1 is off by 0.06 to 0.56 absolute, an order above the output scale;
///   * the RETURNED final state is off by 8.9e-1 relative;
///   * `ChunkPath::Batched` and `ChunkPath::Loop` give the same answer, so it is
///     not the batched rewrite (measured, `DM_GDN2_OPS=loop`);
///   * it persists with the initial state set to zero, so it is not the
///     `S_in * E_last` term (measured);
///   * the state the code carried across the boundary is UNIQUELY determined by
///     its own chunk-1 output — the linear system `out1 = aqk1 U1 +
///     (scale*qg1 - aqk1 W1) S` is 16 equations in 8 unknowns per value column,
///     rank 8, and fits the f32 output to 3e-8 — and it is not
///     `khat0^T v_new0` under any of six decay conventions.
///
/// The intra-chunk algebra is therefore NOT the difference: it is the state
/// handed from chunk 0 to chunk 1.
///
/// # What is NOT established, and is not claimed
///
/// Which side is wrong. The reference is a transcription and this file is not in
/// a position to say the Rust is the odd one out; the two Rust arms agreeing with
/// each other is weak evidence and no more. `FLA`'s `chunk_gated_delta_rule_fwd_
/// kernel_h_blockdim64` states the same update this file's transcription does
/// (`b_h1 *= exp2(b_gk_last); b_h1 += dot(b_k, b_v)` with
/// `b_k = kg = k*exp2(gn - gk)`), which is a point in favour of the
/// transcription and against the Rust, and it is a point, not a measurement.
/// Settling it needs the authors' own kernel run, which is
/// `docs/protocols/ORACLE.md` §8 candidate (2) and was not attempted.
///
/// # Why it is a test and not a paragraph
///
/// Because everything above `forward.rs:296-297` and its twin at
/// `forward.rs:636` — the whole backward's BPTT half, and the fused kernel's
/// BK2 with its `d_k_bptt` / `d_e_bptt` / `d_s_shift` — is UNVERIFIED while this
/// is red, and the terms `2a430cc` measured at rel 3.3e-1 and 6.2e-1 live
/// exactly there. A finding that is a paragraph gets re-litigated; a finding that
/// is a red gate with its oracle committed next to it does not.
/// The state the forward HANDS BACK, against the oracle, at one chunk and at
/// two. The gate that was missing and the reason the seam defect survived.
///
/// # WHY THIS IS A SEPARATE TEST
///
/// `the_forward_agrees_with_the_f64_transcription` runs the same one-chunk
/// fixture and throws the state away: `let (out, _new_state) = ...`
/// (`autodiff_bwd_f64.rs:251`). So for the whole life of that gate the
/// OUTPUT of a chunk was checked and the STATE a chunk leaves behind never
/// was. That is precisely the quantity the two-chunk case consumes, and it is
/// why `two_chunks_the_forward_still_agrees_...` was the only red in the file
/// while the one-chunk arms were green: a wrong state is invisible at one
/// chunk by construction.
///
/// # WHY IT IS THE CHEAPEST GATE IN THE TREE
///
/// One chunk, so the state is produced by exactly one update and there is no
/// chain to reason about; the `out_state` block is already committed in
/// `ref_bwd_f64.bin`; and it needs no f64 gradient, no fault fixture and no
/// second method. It is the check that would have caught this on the day the
/// one-chunk fixture was written.
///
/// Measured on the f32 arm: the OUTPUT of a one-chunk call is within 1.4e-7
/// relative of the oracle, and the STATE it returns is not, by a factor the
/// print below names on every run.
#[test]
fn the_state_we_hand_back_is_the_state_the_oracle_computed() {
    let dev = Device::ndarray();
    let mut worst = 0.0f64;
    let mut rows = Vec::new();
    for (label, f) in [
        ("one chunk ", Fixture::load_ref()),
        ("two chunks", load(include_bytes!("ref_bwd_f64_carry.bin"), b"GDN2BFD\0")),
    ] {
        let inp = tensors(&f, &dev);
        let (_, s_ref) = f.get("out_state");
        let (_out, new_state) = chunk_wy_forward(
            inp[0].clone(),
            inp[1].clone(),
            inp[2].clone(),
            inp[3].clone(),
            inp[4].clone(),
            inp[5].clone(),
            inp[6].clone(),
            SCALE,
            CHUNK,
        );
        assert_eq!(
            new_state.dims().iter().product::<usize>(),
            s_ref.len(),
            "state element count changed"
        );
        let (r, scale) = rel(&host(&new_state), &s_ref);
        rows.push(format!("  {label}  state rel = {r:.3e}  (state scale {scale:.3e})"));
        worst = worst.max(r);
    }
    println!(
        "\nstate handed back, f32 arm vs the f64 transcription (BAR {FWD_BAR:.0e}):\n{}\n  \
         ESTABLISHED: the one-chunk state is checked for the first time. A wrong state is \
         invisible in the one-chunk OUTPUT, which is why this file was green there.",
        rows.join("\n")
    );
    assert!(
        worst < FWD_BAR as f64,
        "the state the forward hands back is {worst:.3e} from the f64 transcription, over the \
         {FWD_BAR:.0e} bar. Everything that consumes a carried state is unverified while this \
         is red: the next chunk's output, the backward's BPTT half, and the fused kernel's BK2."
    );
}

#[test]
fn two_chunks_the_forward_still_agrees_with_the_f64_transcription() {
    let f = load(include_bytes!("ref_bwd_f64_carry.bin"), b"GDN2BFD\0");
    let dev = Device::ndarray();
    let inp = tensors(&f, &dev);
    let (out_shape, out_ref) = f.get("out");
    let (out, new_state) = chunk_wy_forward(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        SCALE,
        CHUNK,
    );
    let (r, scale) = rel(&host(&out), &out_ref);
    let got_out = host(&out);
    let (t_dim, c_dim) = (out_shape[2], out_shape[3]);
    let per_t: Vec<f64> = (0..t_dim)
        .map(|t| {
            (0..c_dim)
                .map(|c| (got_out[t * c_dim + c] - out_ref[t * c_dim + c]).abs())
                .fold(0.0f64, f64::max)
        })
        .collect();
    let c0 = per_t[..CHUNK].iter().copied().fold(0.0f64, f64::max);
    let c1 = per_t[CHUNK..].iter().copied().fold(0.0f64, f64::max);
    let (_, s_ref) = f.get("out_state");
    let (sr, _) = rel(&host(&new_state), &s_ref);
    println!(
        "\nT = {t_dim} = 2 chunks, against the same f64 oracle:\n  \
         output        rel = {r:.3e}  (scale {scale:.3e})\n  \
         chunk 0       max |diff| = {c0:.3e}\n  \
         chunk 1       max |diff| = {c1:.3e}\n  \
         final state   rel = {sr:.3e}\n  \
         oracle self-consistency (fwd-mode vs FD) = {:.3e}\n\
         ESTABLISHED: the state HANDED ACROSS THE BOUNDARY agrees to f32 noise, so the\n\
         backward's inter-chunk half and the fused kernel's BK2 have an oracle.  This gate was\n\
         RED at chunk 1 = 5.603e-01 and the fault was the ORACLE's, not this code's: the f64\n\
         transcription exponentiated the cumsum twice (`exp(E_last - G)`, E already being\n\
         exp(G)), which agrees with nothing in the FLA source.  Two Rust arms agreeing is\n\
         still a point, not a proof - but the decay convention is now pinned to FLA\n\
         (fla/ops/common/chunk_delta_h.py:236-240, commit 9f38d249), which is not us.",
        f.spread
    );
    assert!(
        r < FWD_BAR as f64,
        "the forward is {r:.3e} from the f64 transcription on a TWO-chunk sequence \
         (chunk 0 {c0:.3e}, chunk 1 {c1:.3e}, final state {sr:.3e}). Until this is fixed the \
         backward's inter-chunk half - and the fused kernel's BK2 - have no oracle, because \
         an oracle for a function we do not compute is not an oracle."
    );
}

impl Fixture {
    fn load_ref() -> Fixture {
        load(include_bytes!("ref_bwd_f64.bin"), b"GDN2BFD\0")
    }
    fn load_faults() -> Vec<u8> {
        include_bytes!("ref_bwd_f64_faults.bin").to_vec()
    }
}
