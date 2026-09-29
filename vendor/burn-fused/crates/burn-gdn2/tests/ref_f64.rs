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
// our code. `docs/ORACLE.md` §2 shows why that is not enough: a bug above the
// fused/ops branch point — i.e. inside `project` or `output` — moves both arms
// together and the difference is exactly zero. `tests/ref_data.bin` used to be
// the one layer that could see `project`; it is DELETED (2026-09-29) because it
// was an f32 transcription of our OWN algorithm, so all it could ever prove was
// self-consistency, and it replicate-padded the short conv exactly as the kernel
// wrongly did. This file is its replacement, and it is the first layer here
// whose expected value does not come from a second implementation of the same
// code. The two `binary-tests` targets it absorbed are `tests/oracle_breadth.rs`
// (1000-case breadth sweep, `tests/ref_f64_broad.bin`) and
// `tests/oracle_chunk.rs` (the same sweep at five chunk sizes); both are RED, on
// the pre-existing defect the STATUS block below measures.
//
// TIER, HONESTLY. This is **(b)**, not (a). It is a transcription, so a shared
// *misreading* of the paper would survive it. It is not the authors' own bytes;
// those need NVlabs' Triton kernel actually run (`docs/ORACLE.md` §8 candidate
// (2), not attempted). The words "bit-exact" and "bit-for-bit" are not used
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
// ============================ STATUS: RED ================================
//
// `gdn2_f32_agrees_with_the_f64_reference` FAILS. Not a tolerance problem and
// not f32 noise: the deviation is 9.227e-02 relative at T=2, two orders above
// the 1e-3 bar. Reproduce with
//     cargo test --release -p burn-gdn2 --test ref_f64
//
// THE BLOCK BELOW IS THE 2026-09-28 TEXT AND IT IS NOW MEASURED WRONG IN
// THREE PLACES. Kept as a record of the search, corrected where measurement
// moved it, and corrected measurements are marked CORRECTED. The three:
//
// 1. THE CONV PAD IS FIXED AND THE FIX IS RIGHT. `0a6998a` changed
//    `src/short_conv.rs` from replicate to zero padding. Re-measured against
//    this layer's f64 ZERO-pad oracle, `cargo run --release -p burn-gdn2
//    --example ref_f64_stages` + `python3 tools/gen_reference_f64.py
//    --diff-stages /tmp/rust_stages.txt`:
//
//        stage       T=1 (was)   T=1 (now)    T=70 (was)  T=70 (now)
//        q_conv      1.8e-01     1.683e-07    1.6e-01     1.418e-07
//        k_conv      1.4e-01     2.884e-07    1.3e-01     1.742e-07
//        v_conv      1.5e-01     2.371e-07    1.9e-01     1.867e-07
//
//    The old table's numbers are relative to each stage's own scale; the
//    absolute columns it quotes are unchanged in kind. Every conv row is now
//    f32 noise at both lengths. "OUR `q_conv` MATCHES A f64 REPLICATE-PAD TO
//    1.06e-07" is a statement about a bug that no longer exists.
//
// 2. `project` IS CLEAN, INCLUDING EVERY PER-HEAD TENSOR. The old text says the
//    second difference "is not in `project`'s 2-D stages", which was all the
//    diagnostic could see: `diff_stages` compared only 2-D and 1-D rows
//    because a permuted 4-D view reads back unstably on ndarray. It does, but
//    pushing the view through an elementwise op first forces a dense readback
//    (burn 0.22.0-pre.4 has no `contiguous()`), and with those rows added every
//    per-head tensor agrees too: q4d 1.7e-07/3.4e-07, k4d 5.2e-07/2.5e-07,
//    g4d 2.1e-07/3.9e-07, b4d 1.7e-07/2.1e-07, v4d 4.6e-07/1.9e-07, w4d
//    1.9e-07/2.1e-07, gate4d 1.3e-07/1.6e-07 (T=1 / T=70). So the fault is
//    DOWNSTREAM of `project`.
//
// 3. THE FAULT IS THE STATE CARRY, NOT AN UNEXPLAINED DIFFERENCE. Per token,
//    over this 18-case matrix: token 0 is right at every length (3.07e-07 at
//    T=1, 3.49e-08 at T=64) and every token from 1 on is O(1) wrong. `S`
//    starts at zero, so T=1 has no carry - which is also why exactly 24 of the
//    1000 cases in the breadth sweep (`tests/oracle_breadth.rs`, same fixture
//    family) pass and they are exactly the T=1 ones.
//
// NOT YET LOCALISED TO A LINE, and one number I do not trust. Dumping the
// carry's four sub-steps by hand at T=2 (decay, erase, `v_new`, and the
// resulting `S`) has all four agreeing to <= 4.6e-07 at every step, final `S`
// included, which leaves the readout as the only suspect BY ELIMINATION and not
// by measurement. A separate probe calling the crate's own
// `fused_recurrent_forward` reported a final state 1.34e-01 off on the same
// input, which contradicts that and which I could not reproduce. Neither number
// is claimed. The next measurement is the readout
// `(state * q_t.swap_dims(2, 3)).sum_dim(2)` with a RAW (not squared) readback
// of `q_t`: the probe written to settle it squared and then took `sqrt`, which
// recovers `|x|` and not `x`, so every signed row in it read O(1). That is a
// bug in the probe and it is the reason this header claims no line.
//
// WHAT THE OLD TEXT GOT RIGHT AND STILL STANDS. The pad was a real defect and
// both reference generators replicate-padded it too, which is why the
// self-transcription agreed with the bug - `docs/ORACLE.md` §2 in a concrete
// instance. `tests/ref_data.bin`, `tools/gen_reference.rs` and
// `tests/gen_reference.py` are DELETED (2026-09-29) rather than regenerated;
// `tests/bit_exact.rs` and `tests/test_chunk.rs` are retargeted at the f64
// layer as `tests/oracle_breadth.rs` and `tests/oracle_chunk.rs`.
//
// WHY T=1 IS A WEAK CANARY, MEASURED. `S` starts at zero, so `S̄ = Diag(a) S = 0`
// and `r = S̄^T e = 0`: neither the decay nor the erase gate appears at T=1.
// Injected as faults, `decay-sign` and `no-erase-gate` change the T=1 output by
// EXACTLY 0.0, while the other five change it by 0.8 to 2.3. The old fixture's
// 24 passing single-token cases were therefore blind to two of the seven
// semantic error classes, and to the entire state carry - and it turns out the
// state carry is where this crate's live defect is.
//
// OWNER'S NOTE, the 2026-09-28 text that is now DONE. The padding fix is no
// longer deferred to another lane; `0a6998a` landed it, and the current
// numbers above are the check that it was right.
//
// CONDITIONING, because it changed the numbers. `lit_gpt/gdn2.py` zeroes every
// bias, which puts the output gate at `silu(0) = 0` exactly, so a freshly
// initialised layer's output is ~7.5e-04 where its pre-gate value is O(1). A
// RELATIVE bar through a near-null quantity is meaningless - it read 1e+0 for
// differences that were f32 noise. The fixture therefore gives `g_proj_1` a
// non-zero bias (see `make_params`), which changes no formula and no projection,
// only the scale the output is measured against: max|out| goes 7.5e-04 -> 1.1e-01.
//
// END-TO-END DEMONSTRATION THAT THE TEST FAILS ON A WRONG FORMULA, 2026-09-29,
// ndarray, `--release`. Regenerate the fixture with a wrong formula and re-run
// THIS test (`--fault` skips writing `ref_f64_faults.bin`, which stays correct):
//     python3 tools/gen_reference_f64.py --fault decay-sign
//     cargo test -p burn-gdn2 --release --test ref_f64     -> FAILED
//     python3 tools/gen_reference_f64.py --fault transposed-proj
//     cargo test -p burn-gdn2 --release --test ref_f64     -> FAILED
//     python3 tools/gen_reference_f64.py                   # restore
// CAVEAT, stated because it matters: at T=1 the `decay-sign` and
// `no-erase-gate` fixtures are bit-identical to the correct one (see above), so
// the assert trips on T=1 for the pre-existing reason rather than for the
// injected fault. The clean, attribution-valid measurement is the committed one:
// `the_bar_bites_a_wrong_formula` runs on every invocation and puts the nearest
// wrong formula 460x above the bar, from data in the tree.
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

/// The fault-fixture entry that `src/short_conv.rs` actually implements: it
/// replicates the first token into the short conv's left pad where the authors'
/// kernel zero-fills. At T=1, where nothing else can disagree, our output
/// coincides with this entry — which is what identifies the pad as a real
/// defect rather than a transcription difference. See the STATUS block.
const CONV_FAULT: &str = "conv-padding";

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
        t.d, t.h, t.hk, t.hv, t.expand_v, t.hv * v_head, t.cases.len(), t.tensors.len()
    );
    for (label, module) in [
        ("FusedRecurrent", build(&t, Gdn2Mode::FusedRecurrent, 64, &device)),
        ("Chunk", build(&t, Gdn2Mode::Chunk, 64, &device)),
    ] {
        let mut worst = 0.0f64;
        let mut worst_t = 0usize;
        for (i, (seq, x, ref_y)) in t.cases.iter().enumerate() {
            let input =
                Tensor::<3>::from_data(TensorData::new(x.clone(), [1, *seq, t.d]), &device);
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
    assert_eq!(&magic, b"GDN2FLT\0", "ref_f64_faults.bin is not this format");
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
        devs.push((name, rel));
    }
    assert_eq!(
        c.position() as usize,
        data.len(),
        "trailing bytes in the fault fixture"
    );
    for (name, rel) in &devs {
        println!("  wrong formula {name:>18}: {rel:>10.3e} rel  (BAR {BAR:.0e})");
    }
    let smallest = devs.iter().map(|(_, r)| *r).fold(f64::INFINITY, f64::min);
    println!(
        "margin: the nearest wrong formula is {smallest:.3e} and the bar is {BAR:.0e} \
         — {:.0}x below it, against f32 noise at the 1e-6 scale",
        smallest / BAR
    );
    // The `conv-padding` row above is O(1), NOT because the whole of our
    // disagreement is the conv pad. It is O(1) at T=70 for two reasons at once:
    // we really do pad wrongly (confirmed stage-by-stage, STATUS block), AND
    // something between `project`'s output and the layer output disagrees too
    // (see "NOT FULLY EXPLAINED" in the header). T=1 has no state carry, so it
    // isolates the pad: measured there, our output and the replicate-padded f64
    // output are the same number to four significant figures, which is what
    // "the pad is a real and sufficient explanation at T=1" means. Nothing at
    // T>1 is explained by it.
    let (t0, x0, _) = &t.cases[0];
    assert_eq!(*t0, 1, "case 0 is meant to be the single-token canary");
    let m0 = build(&t, Gdn2Mode::FusedRecurrent, 64, &device);
    let inp0 = Tensor::<3>::from_data(TensorData::new(x0.clone(), [1, 1, t.d]), &device);
    let mut st0: Option<burn_gdn2::Gdn2State> = None;
    let got0 = to_f32_vec(&m0.forward::<NdArray>(inp0, &mut st0, true));
    let mut c0 = Cursor::new(data.as_slice());
    let mut m8 = [0u8; 8];
    c0.read_exact(&mut m8).unwrap();
    let _ = rd_u32_pub(&mut c0); // case_idx
    let _ = rd_u32_pub(&mut c0); // n_faults
    let _ = rd_u32_pub(&mut c0); // T
    let _ = rd_u64_pub(&mut c0); // input hash
    let mut ident = None;
    for _ in 0..n_faults {
        let name = rd_name_pub(&mut c0);
        let faulty = rd_f64_pub(&mut c0, seq * t.d);
        if name != CONV_FAULT {
            continue;
        }
        // The fault fixture holds T=70 outputs. Its first `d` values are that
        // case's t=0 output, which is the single-token quantity we want here.
        let first = &faulty[..t.d];
        ident = Some(
            got0.iter()
                .zip(first.iter())
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0.0f64, f64::max)
                / max_abs(first).max(f64::MIN_POSITIVE),
        );
    }
    let ident = ident.expect("the fault fixture has no `conv-padding` entry");
    // Printed, NOT asserted. The hypothesis this measures is "at T=1, where no
    // state carry exists, the whole of our disagreement is the conv pad, so our
    // output should equal the replicate-padded f64 output". IT DOES NOT: the
    // measured value is O(1), not ~1e-6, even though our `q_conv` matches a f64
    // replicate-pad to 1.06e-07 and every other 2-D stage of `project` agrees
    // to ~2e-07. So a SECOND difference exists between this crate's output and
    // the f64 reference that is not in `project`'s 2-D stages and not in
    // `fused_recurrent_forward` (which was read against Eq. 9 and matches it).
    // Asserting the identification would be asserting something measured to be
    // false; asserting its negation would be asserting an open question. So it
    // is reported, and left open. See "NOT FULLY EXPLAINED" in the header.
    println!(
        "T=1 probe: our output vs the {CONV_FAULT} f64 output is {ident:.3e} rel \
         (expected ~1e-6 if the pad were the ONLY difference; it is not)"
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
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 8) };
    c.read_exact(bytes).unwrap();
    v
}
