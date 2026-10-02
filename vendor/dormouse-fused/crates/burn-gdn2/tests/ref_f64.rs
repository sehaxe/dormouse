// The f64 CPU reference layer for Gated DeltaNet-2.
//
// WHAT THIS IS. `tests/ref_f64.bin` holds the output of an **f64** transcription
// of arXiv:2605.22791's Eq. 9/10, written out one token at a time, for 18 fixed
// cases (T = 1..70, straddling every chunk size this crate tests). The generator
// is `tools/gen_reference_f64.py`; its module docstring cites the source of
// every line, including the three that are NOT in the paper (the `1/sqrt(K)`
// scale, the short conv's zero padding, and the exact form of the L2 norm).
//
// WHY IT EXISTS. Every other numeric test in this crate compares our code to
// our code. `docs/protocols/ORACLE.md` §2 shows why that is not enough: a bug above the
// fused/ops branch point — i.e. inside `project` or `output` — moves both arms
// together and the difference is exactly zero. `tests/ref_data.bin` was the one
// layer that could see `project`, and it is RED (1.38e-2, 976/1000) behind a
// non-default feature. This file is the replacement, and it is the first layer
// here whose expected value does not come from a second implementation of the
// same code.
//
// TIER, HONESTLY. This is **(b)**, not (a). It is a transcription, so a shared
// *misreading* of the paper would survive it. It is not the authors' own bytes;
// those need NVlabs' Triton kernel actually run (`docs/protocols/ORACLE.md` §8 candidate
// (2), not attempted). Neither "bit-exact" nor "bit-for-bit" is used
// about anything here and must not be.
//
// THE BAR, AND WHY IT IS FAKEABLE. A semantic error is O(1) relative; f32
// reassociation over 8 chunk boundaries is O(1e-6). The bar is 1e-3 relative to
// the case's own output scale, so it sits ~3 orders of magnitude above the
// noise and hundreds of times below the smallest semantic error. Nothing rests
// on the bar alone: `the_bar_bites_a_wrong_formula` measures the other side of
// the gap on every run, from committed data, so the claim cannot rot into a
// self-consistency check.
//
// ===================== STATUS: the RED was the FIXTURE =====================
//
// `gdn2_f32_agrees_with_the_f64_reference` was RED, and it was this file's
// reference that was wrong. Not a tolerance problem, not f32 noise, and not the
// kernel. It is recorded here at length because the shape of the mistake is the
// reusable part.
//
// THE DEFECT, ONE LINE. `tools/gen_reference_f64.py` split the six per-head
// tensors like this:
//
//     q = q.reshape(t, H, HK).transpose(1, 0, 2)   # [T, KD] -> [T, H, HK] -> [H, T, HK]
//     k = k.reshape(t, H, HK).transpose(1, 0, 2)
//     g = g.T.reshape(H, t, HK)                   # <- head-major. WRONG.
//     b = b.reshape(t, H, HK).transpose(1, 0, 2)
//
// Every one of those is a [T, KD] token-major buffer. `g` alone was reshaped
// head-major, so for t > 1 the head axis and the token axis are mixed. With
// H=4, HK=16, t=2 and input g[kd, i], `g.T.reshape(H, t, HK)` puts
// out[1, 0, 0] = g[32, 0] where out[1, 0, 0] must be g[16, 0].
//
// AT t == 1 THE TWO FORMS COINCIDE ON EVERY ELEMENT. So case 0 (T=1) was green,
// case 1 (T=2) was 9.227e-2 against a 1e-3 bar, and everything after it was
// worse. A canary that passes for a structural reason is not a canary.
//
// WHY IT SURVIVED A FIXTURE THAT REPRODUCES. The committed `.bin`
// regenerates to the identical bytes from the generator - checked, `cmp` clean -
// so reproducibility was never going to catch it. Only comparing the fixture
// against an INDEPENDENT computation of the same quantity does, which is what
// the arithmetic below is.
//
// THE EVIDENCE, AND IT IS CLOSED. The failing number is not merely the right
// order of magnitude, it is the right number:
//
//   observed   9.227e-02 rel * 1.124e-01 scale = 1.0371e-02 absolute
//   predicted  max |ref_broken - ref_fixed| at T=2     = 1.037104e-02
//   predicted  the same at T=1                         = 0.000000e+00 exactly
//
// So our f32 output agrees with the CORRECTED reference, and the entire
// deviation the test reported was the fixture's own error. That is a
// measurement, not an inference: if the kernel had a defect of its own the
// deviation would be the sum of two errors and would not land on the predicted
// value to five significant figures. The fix is the transpose, in the
// generator, to the idiom the same function already used four lines above.
//
// SAME DISEASE AS `ff7cd57`, IN A FILE WRITTEN AFTER THAT FIX.
// `tools/gen_reference.rs` read token-major scan inputs with head-major offsets;
// there it produced a false RED (1000 cases, 1.38e-2) and here it produced a
// false FIXTURE. Both were invisible at T=1, and in both the single-token cases
// were the ones that passed. A canary that only tests the degenerate case of
// its own indexing rule is a comment, not a gate.
//
// ------------------------- WHAT WAS TRUE, AND IS NOT -------------------
//
// History, corrected rather than deleted, because a reader comparing this file
// against `8162672`'s message needs to know which claims died and why.
//
// 1. "The short conv pads by REPLICATING; the authors zero-fill." TRUE of the
//    code between `0a6998a` and `ff7cd57`, and FIXED: `src/short_conv.rs:44-58`
//    now pads with `Tensor::zeros` and names the three sources. `conv-padding`
//    is still in the fault list, and it is still a wrong formula - it is just no
//    longer what we do.
// 2. "The stage table shows `q_conv`/`k_conv`/`v_conv` disagreeing by 0.49..3.01
//    while every other 2-D stage of `project` agrees to <= 3e-07." That table
//    was measured WHILE the pad was wrong. It is not evidence about the current
//    code and is not repeated here.
// 3. "`bit_exact` is RED at 1.38e-2 and the cause is elsewhere." FIXED by
//    `ff7cd57`: the generator, not the crate. 1000 cases, max_diff 2.32e-7.
//    The paragraph in this crate's `README.md` that still says "currently RED on
//    both, by 1.38e-2" is corrected in this commit; `0a6998a`'s message cannot
//    be edited and now names a defect that had been fixed four commits earlier.
// 4. "NOT FULLY EXPLAINED: at T=1, where no state carry exists, the gap between
//    our output and a replicate-padded f64 forward is 1.15e+00, so a SECOND
//    difference exists." THERE IS NO SECOND DIFFERENCE. The probe that printed
//    it compared case 0's output against the FIRST `d` VALUES OF CASE 17 - a
//    T=70 output, computed from a DIFFERENT random input (the fixtures draw one
//    input per case from a single stream). Two unrelated inputs cannot agree
//    below O(1) and it never could, so the probe had no power to read below
//    O(1) and its O(1) output was not evidence of anything. At HEAD it printed
//    9.410e-01, and an independent Python computation of the same mis-measured
//    quantity gives 9.410e-01. The probe is DELETED, because a measurement with
//    no power to discriminate is worse than no measurement: it was quoted as an
//    open defect, and an unexplained 1.15 would have implied a third bug that
//    does not exist. What actually answers the question it asked is the main
//    test: case 0 passes, so at T=1 our output IS the f64 answer to within
//    1e-3, and the whole of any T=1 disagreement is nothing.
//
// ------------------------------ THE MARGIN ------------------------------
//
// BAR = 1e-3 relative; f32 reassociation noise is ~1e-6; SEMANTIC_FLOOR = 1e-1.
// EIGHT wrong formulas are committed in `ref_f64_faults.bin` on the same
// weights and the same input, and `the_bar_bites_a_wrong_formula` measures the
// far side of the gap on every run. Measured on the committed fixture:
//
//     output-gate-sigmoid  2.090e-01   209x BAR   2.1x SEMANTIC_FLOOR
//     no-write-gate        2.584e-01   258x BAR   2.6x
//     no-erase-gate        3.850e-01   385x BAR   3.8x
//     no-scale             4.171e-01   417x BAR   4.2x
//     conv-padding         5.170e-01   517x BAR   5.2x
//     read-before-write    6.005e-01   600x BAR   6.0x
//     decay-sign           1.345e+00  1345x BAR  13.4x
//     transposed-proj      1.389e+00  1389x BAR  13.9x
//
// The margin is 209x, and the nearest wrong formula is now the OUTPUT GATE -
// the one line where two credible upstreams disagree and this crate has picked
// one without an A/B behind it. That is the honest place for the tightest
// number in the table to be.
//
// THE OLD "460x" WAS THE DENOMINATOR, NOT THE MARGIN. `0a6998a`/`8162672` and
// this file both quoted "460x" from 4.6e-01, which is `max|ref_y|` - the
// quantity every relative deviation is DIVIDED by. The nearest wrong formula
// was 2.71e-01 away, i.e. 272x. The claim was conservative (it overstated the
// margin rather than hiding a tight one) and it was still not what the
// arithmetic says. The test now prints `max|ref_y|` separately, on its own
// line, so the two can never be conflated again.
//
// ------------------------- THE OUTPUT GATE ------------------------------
//
// The 8th fault is new and is the point of it. `src/module.rs` applies
// `silu(gate)`. `fla/layers/gdn2.py:197` (fla-org/flash-linear-attention @
// 9f38d249, fetched 2026-09-29) applies `sigmoid`. Both branches live in ONE
// upstream kernel - `fla/modules/fused_norm_gate.py:101-104`, selected by
// `ACTIVATION` - and the line the reference always cited (`:102`) is the SWISH
// one, so the reference and the kernel agree on this line and have done since
// the fixture was written.
//
// A claim that this oracle "transcribes the FLA file and therefore picks
// sigmoid" is wrong, and wrong in a way worth writing down: `fla/modules/` is
// the KERNEL and `fla/layers/` is the LAYER, and only the layer selects
// `activation="sigmoid"`. The oracle is NOT blind to a kernel flip - if
// `silu(gate)` became `sigmoid(gate)` the main test would fail loudly, because
// the reference is swish. What it did not do is MEASURE the choice, so the
// choice rested on a docstring sentence. It is now a committed wrong formula
// with a measured distance, and the `saw_output_gate` assert in
// `the_bar_bites_a_wrong_formula` fails the suite if a regeneration ever drops
// it. The choice itself is deliberately not
// changed: it is a technology A/B arm, and the neighbouring `burn-kda` already
// picks the other side (arXiv:2607.24653 §2.1.1 Eq. 6).
//
// END-TO-END DEMONSTRATION THAT THE TEST FAILS ON A WRONG FORMULA, 2026-09-29,
// ndarray, `--release`. Regenerate the fixture with a wrong formula and re-run
// THIS test (`--fault` skips writing `ref_f64_faults.bin`, which stays correct):
//     python3 tools/gen_reference_f64.py --fault decay-sign
//     cargo test -p burn-gdn2 --release --test ref_f64     -> FAILED
//     python3 tools/gen_reference_f64.py --fault transposed-proj
//     cargo test -p burn-gdn2 --release --test ref_f64     -> FAILED
//     python3 tools/gen_reference_f64.py                   # restore
// The clean, attribution-valid measurement is the committed one:
// `the_bar_bites_a_wrong_formula` runs on every invocation and puts the nearest
// wrong formula 209x above the bar, from data in the tree.
//
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the
// burn-flex migration.
#![allow(dead_code, deprecated)]

// `Cursor`/`Read` (and everything else the fixture loader needs) arrive with
// the `include!` below, which is a textual include into this module.
include!("common/ref_f64.rs");

/// Relative to the case's own max |output|. Do NOT widen this to make something
/// green: a discrepancy that a wider bar would hide is the whole reason this
/// layer exists.
const BAR: f64 = 1e-3;

/// A wrong formula must land at least this far out, or the bar is not
/// separating anything. 1e-1 relative: 100x above BAR and ~3 orders below the
/// largest measured semantic error.
const SEMANTIC_FLOOR: f64 = 1e-1;

/// The fault entry covering the unresolved output-gate technology choice.
///
/// `src/module.rs` applies `silu(gate)`; `fla/layers/gdn2.py:197` selects
/// `sigmoid`. Both branches are in ONE upstream kernel
/// (`fla/modules/fused_norm_gate.py:101-104`), so this is a configuration
/// choice between two credible implementations, not a transcription error, and
/// it is deliberately NOT resolved here. What this constant does is make the
/// choice MEASURED instead of assumed: the fixture carries the sigmoid answer
/// for the same weights and the same input, so the distance between the branch
/// we implement and the branch the other upstream picks is a number in the
/// tree. The `saw_output_gate` assert below fails loudly if a
/// regeneration ever drops it, which is how this class would silently go back
/// to being uncovered.
const OUTPUT_GATE_FAULT: &str = "output-gate-sigmoid";

/// The main assertion: our f32 output, on both dispatch arms, is the f64
/// reference's answer to within f32 arithmetic.
#[test]
fn gdn2_f32_agrees_with_the_f64_reference() {
    let t = load();
    let device = Device::ndarray();
    let v_head = (t.hk as f32 * t.expand_v) as usize;
    println!(
        "fixture: d={} h={} hk={} hv={} expand_v={} -> value head {v_head}, VD={}, \
         {} cases, {} tensors",
        t.d,
        t.h,
        t.hk,
        t.hv,
        t.expand_v,
        t.hv * v_head,
        t.cases.len(),
        t.tensors.len()
    );
    for (label, module) in [
        (
            "FusedRecurrent",
            build(&t, Gdn2Mode::FusedRecurrent, 64, &device),
        ),
        ("Chunk", build(&t, Gdn2Mode::Chunk, 64, &device)),
    ] {
        let mut worst = 0.0f64;
        let mut worst_t = 0usize;
        for (i, (seq, x, ref_y)) in t.cases.iter().enumerate() {
            let input = Tensor::<3>::from_data(TensorData::new(x.clone(), [1, *seq, t.d]), &device);
            let mut state: Option<burn_gdn2::Gdn2State> = None;
            let out = module.forward::<NdArray>(input, &mut state, true);
            let got = to_f32_vec(&out);
            let scale = max_abs(ref_y).max(f64::MIN_POSITIVE);
            let rel = got
                .iter()
                .zip(ref_y.iter())
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0.0f64, f64::max)
                / scale;
            if rel > worst {
                worst = rel;
                worst_t = *seq;
            }
            assert!(
                rel < BAR,
                "{label} case {i} (T={seq}): max relative deviation {rel:.3e} \
                 >= BAR {BAR:.0e} (output scale {scale:.3e})"
            );
        }
        println!(
            "{label:>14} vs f64 reference: max rel = {worst:.3e} at T={worst_t} (BAR {BAR:.0e})"
        );
    }
}

/// The demonstration that the bar can fail, kept in the tree so it cannot rot.
///
/// `ref_f64_faults.bin` holds the SAME weights and the SAME input run through
/// seven wrong formulas. For each we assert our f32 output is at least
/// `SEMANTIC_FLOOR` away from it. A bare `rel < BAR` assertion cannot detect the
/// failure this guards: a semantic error that moved down into the noise band
/// would look like a pass.
#[test]
fn the_bar_bites_a_wrong_formula() {
    let t = load();
    let data = include_bytes!("ref_f64_faults.bin");
    let mut c = Cursor::new(data.as_slice());
    let mut magic = [0u8; 8];
    c.read_exact(&mut magic).unwrap();
    assert_eq!(
        &magic, b"GDN2FLT\0",
        "ref_f64_faults.bin is not this format"
    );
    let case_idx = rd_u32_pub(&mut c) as usize;
    let n_faults = rd_u32_pub(&mut c) as usize;
    let seq = rd_u32_pub(&mut c) as usize;
    let x_hash = rd_u64_pub(&mut c);

    let (ref_t, x, ref_y) = &t.cases[case_idx];
    assert_eq!(*ref_t, seq, "fault fixture and ref fixture disagree on T");

    // The fault fixture must have been built from THIS case's input. The first
    // version of the generator re-drew the RNG for the fault case instead of
    // consuming the stream up to it, so the wrong-formula outputs belonged to a
    // different input and every fault read O(1) for a reason that had nothing
    // to do with the formula. Check it rather than trust it.
    let got_hash = fnv1a64(x);
    assert_eq!(
        got_hash, x_hash,
        "ref_f64_faults.bin was generated from a different input than \
         ref_f64.bin case {case_idx}: fnv1a64 {got_hash:#018x} vs {x_hash:#018x}. \
         Re-run tools/gen_reference_f64.py so both come from make_inputs()."
    );

    let device = Device::ndarray();
    let module = build(&t, Gdn2Mode::FusedRecurrent, 64, &device);
    let input = Tensor::<3>::from_data(TensorData::new(x.clone(), [1, seq, t.d]), &device);
    let mut state: Option<burn_gdn2::Gdn2State> = None;
    let got = to_f32_vec(&module.forward::<NdArray>(input, &mut state, true));
    let scale = max_abs(ref_y).max(f64::MIN_POSITIVE);

    let mut devs = Vec::new();
    let mut saw_output_gate = false;
    for _ in 0..n_faults {
        let name = rd_name_pub(&mut c);
        let faulty = rd_f64_pub(&mut c, seq * t.d);
        let rel = got
            .iter()
            .zip(faulty.iter())
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0.0f64, f64::max)
            / scale;
        assert!(
            rel > SEMANTIC_FLOOR,
            "a wrong formula (`{name}`) is only {rel:.3e} from our output, below \
             SEMANTIC_FLOOR {SEMANTIC_FLOOR:.0e} — the bar is no longer separating \
             semantic error from arithmetic noise"
        );
        saw_output_gate |= name == OUTPUT_GATE_FAULT;
        devs.push((name, rel));
    }
    assert_eq!(
        c.position() as usize,
        data.len(),
        "trailing bytes in the fault fixture"
    );
    // The output-gate class must stay covered. The loop above only proves the
    // bar separates what IS in the fixture; this proves the fixture still
    // contains the one entry that keeps the SiLU-vs-sigmoid choice honest.
    assert!(
        saw_output_gate,
        "ref_f64_faults.bin has no `{OUTPUT_GATE_FAULT}` entry, so the output-gate \
         technology choice is uncovered again. Re-run tools/gen_reference_f64.py."
    );
    for (name, rel) in &devs {
        println!("  wrong formula {name:>20}: {rel:>10.3e} rel  (BAR {BAR:.0e})");
    }
    let smallest = devs.iter().map(|(_, r)| *r).fold(f64::INFINITY, f64::min);
    // The margin is `smallest / BAR` and NOTHING ELSE. It used to be quoted as
    // `max|ref_y| / BAR` (4.6e-01/1e-3 = "460x"), which is the DENOMINATOR of
    // the relative deviation, not the numerator: the nearest wrong formula was
    // 2.71e-01 away, i.e. 272x. The number was conservative - it overstated the
    // margin rather than hiding a tight one - and it was still not what the
    // arithmetic says. Printed separately from here on so the two can never be
    // confused again.
    println!(
        "margin: the nearest wrong formula is {smallest:.3e} and the bar is {BAR:.0e} \
         — {:.0}x below it, against f32 noise at the 1e-6 scale",
        smallest / BAR
    );
    println!(
        "  (for contrast: max|ref_y| on this case is {scale:.3e}, which is the \
         DENOMINATOR of every rel figure above and is NOT the margin)"
    );
}

// The fixture reader in ref_f64_common.rs is private; these are thin public
// wrappers so the fault fixture (a second file) can be read with the same code
// path rather than a second copy of the format.
fn rd_u32_pub(c: &mut Cursor<&[u8]>) -> u32 {
    let mut b = [0u8; 4];
    c.read_exact(&mut b).unwrap();
    u32::from_le_bytes(b)
}
fn rd_u64_pub(c: &mut Cursor<&[u8]>) -> u64 {
    let mut b = [0u8; 8];
    c.read_exact(&mut b).unwrap();
    u64::from_le_bytes(b)
}
/// FNV-1a 64 over the little-endian f32 bytes. The same function as
/// `fnv1a64` in `tools/gen_reference_f64.py`; it exists so the two fixtures can
/// prove they were built from the same input.
fn fnv1a64(data: &[f32]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for v in data {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3);
        }
    }
    h
}
fn rd_name_pub(c: &mut Cursor<&[u8]>) -> String {
    let n = rd_u32_pub(c) as usize;
    let mut b = vec![0u8; n];
    c.read_exact(&mut b).unwrap();
    String::from_utf8(b).unwrap()
}
fn rd_f64_pub(c: &mut Cursor<&[u8]>, n: usize) -> Vec<f64> {
    let mut v = vec![0f64; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 8) };
    c.read_exact(bytes).unwrap();
    v
}
