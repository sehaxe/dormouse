// The chunk-size sweep, against the f64 external oracle, plus one self-
// consistency test that earns its place for a different reason.
//
// WHY THIS IS THE SAME ORACLE AS `tests/oracle_breadth.rs`. Both used to read
// `tests/ref_data.bin`, an **f32 transcription of burn-gdn2's own algorithm**
// produced by `tools/gen_reference.rs`. `docs/protocols/ORACLE.md` §2: a self-
// transcription can only prove self-consistency, because two arms that share a
// misunderstanding agree. It was not hypothetical - the short conv's
// replicate-pad (`0a6998a`, fixed) was replicate-padded by BOTH reference
// generators too, so the fixture agreed with the bug and the test was green.
// `ref_data.bin`, `tools/gen_reference.rs` and `tests/gen_reference.py` are
// deleted in the commit that retargets this test; the provenance, the tier
// ("(b) transcription of arXiv:2605.22791 §3.1 Eq. 8-12", NOT the authors'
// bytes, no "bit-exact" claim) and the full source citations are in
// `tools/gen_reference_f64.py`'s module docstring and in
// `tests/oracle_breadth.rs`'s header. This header does not repeat them; the two
// files read one fixture through one loader and one bar.
//
// WHAT THE SWEEP ADDS OVER `oracle_breadth.rs`. The same 1000 cases, the same
// weights, the same f64 expected values, but the CHUNKED dispatch arm at
// chunk_size 4 / 8 / 16 / 32 / 64 instead of the fused-recurrent arm. That is
// the one arm in this crate whose arithmetic is NOT a re-ordering of the same
// loop: it is a different factorisation of the delta rule (chunked WY
// representation, `Ut`, `kw`, intra-chunk correction), so its rounding error
// has a different shape and it needed its own arm rather than being assumed to
// follow the other one. T in 1..=38 straddles 4, 8, 16 and 32 many times; 64
// exceeds the sweep, which is why the 18-case matrix in `tests/ref_f64.rs`
// (T = 64, 65, 70, both arms) is not redundant with it.
//
// THE BAR. `1e-3` relative to each case's own `max|reference output|`, the same
// value and the same meaning as `BAR` in `tests/ref_f64.rs` and
// `tests/oracle_breadth.rs`. The old bound on this test was `1e-3` per case
// ABSOLUTE plus a `1e-2` global ABSOLUTE, against a fixture whose output scale
// was ~1e-2 - so the old per-case bound was effectively 1e-1 RELATIVE. The new
// one is 1e-3 relative, i.e. 100x tighter in relative terms and 13x tighter in
// absolute terms on the worst-conditioned case of this sweep (per-case output
// scale 3.93e-02 min, 3.29e-01 median, 5.49e-01 max; 1e-3 relative on the
// smallest of those is 3.9e-05 absolute). Nothing was widened. The far side of
// the bar is committed data, not prose: `tests/ref_f64_faults.bin` holds the
// eight wrong formulas and `ref_f64.rs::the_bar_bites_a_wrong_formula` asserts
// on every run that our output is at least 1e-1 relative from each of them;
// re-measured over all 1000 cases of this sweep the nearest is 4.084390e-01
// (`output-gate-sigmoid` at T=2), 408x above the bar.
//
// REGENERATE - run the checker, not a byte-diff.
//     cd vendor/dormouse-fused/crates/burn-gdn2
//     python3 tools/check_f64_fixtures.py
// `git diff --exit-code` on the fixture is NOT the check and never was: the f64
// outputs are numpy reductions whose summation order depends on array layout,
// so a correct regeneration agrees to ~7e-16 rather than byte-for-byte, and the
// byte-diff fails on a good fixture. `tests/oracle_breadth.rs`'s header carries
// the measurements.
//
// STATUS, AND THE DEMONSTRATION THAT THIS TEST CAN FAIL. The status is in
// `tests/oracle_breadth.rs`, and the short version is that both of these tests
// were red because `ref_f64_broad.bin` had stopped matching its generator
// (`ca45600` fixed a head-major `g` layout bug in the generator and did not
// regenerate this one fixture of three), NOT because of any kernel defect.
// They share the root exactly: same file, same loader (`load_broad()`), same
// weights, same `BAR`. What was measured, and what each number is:
//
//   * measured, then: the first failing case was chunk_size=4 case 1 (T=3) at
//     1.851e-01, and 976/1000 cases over the bar. Those are the STALE FIXTURE's
//     numbers, reproduced to five figures by
//     `python3 tools/check_f64_fixtures.py` - the chunked arm was never the
//     subject, it was reading the same stale bytes through the same bar.
//   * measured: with the fixture regenerated from the current generator, both
//     arms are inside the bar, at all five chunk sizes; the run prints each.
//   * NOT measured, and not claimed: whether the chunked arm's OWN rounding sits
//     comfortably inside 1e-3 at every chunk size, as a question separate from
//     the fixture. That is what the printed per-size worst is for; read it.
//
// One note that IS specific to this file, and it is the reason the arm exists:
// the chunked arm's rounding is a re-ordering of a DIFFERENT factorisation (the
// chunked WY solve), not of the same loop, so its error shape is its own and
// assuming the two arms follow each other would have been an assumption, not a
// result.
#![allow(dead_code, deprecated)]

include!("common/ref_f64.rs");

/// See `tests/oracle_breadth.rs` for the arithmetic behind this value. It is
/// the same constant, deliberately, so the two arms cannot drift apart on how
/// much error they tolerate.
const BAR: f64 = 1e-3;

#[test]
#[cfg(feature = "binary-tests")]
fn chunk_sizes_match_the_f64_oracle() {
    let t = load_broad();
    let device = Device::ndarray();
    let mut global_worst = 0.0f64;
    let mut n_fail = 0usize;

    for chunk_size in [4usize, 8, 16, 32, 64] {
        let module = build(&t, Gdn2Mode::Chunk, chunk_size, &device);
        let mut case_worst = 0.0f64;
        let mut case_worst_t = 0usize;
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
            if rel > case_worst {
                case_worst = rel;
                case_worst_t = *seq;
            }
            let abs = rel * scale;
            assert!(
                rel < BAR,
                "chunk_size={chunk_size} case {i} (T={seq}): max relative deviation \
                 {rel:.3e} >= BAR {BAR:.0e} (output scale {scale:.3e}, i.e. \
                 {abs:.3e} absolute)"
            );
            if rel >= BAR {
                n_fail += 1;
            }
        }
        global_worst = global_worst.max(case_worst);
        println!(
            "  chunk_size={chunk_size:>2}: worst {case_worst:.3e} rel at T={case_worst_t} \
             (BAR {BAR:.0e})"
        );
    }

    assert_eq!(n_fail, 0, "{n_fail} case/chunk-size pairs at or above the bar");
    println!(
        "Chunk, all 5 chunk sizes x {} cases: worst {global_worst:.3e} rel (BAR {BAR:.0e})",
        t.cases.len()
    );
}

/// NOT AN ORACLE TEST, and it is kept for a different reason. It compares the
/// chunked arm against the fused-recurrent arm - our code against our code -
/// so it cannot see a bug both share, and it makes no external claim. It stays
/// because it is the only thing in the crate that exercises a LONG state carry
/// (T=130, i.e. two full chunks plus a partial) against a STRONG non-uniform
/// per-channel decay, which is the regime `oracle_breadth.rs`'s T in 1..=38 and
/// the T=1 blind spot documented there both under-cover.
#[test]
#[cfg(feature = "binary-tests")]
fn test_chunk_matches_fused_with_real_decay() {
    use burn::tensor::Device;
    use burn::tensor::{Distribution, Tensor};
    use burn_gdn2::{chunk_wy_forward, fused_recurrent_forward};

    let device = Device::ndarray();
    let (b, h, t, k, vd, c) = (2usize, 3usize, 130usize, 16usize, 8usize, 64usize);

    // Strong, non-uniform per-channel decay: log-decay in [-0.15, -0.01]
    // (per-channel, per-position). The old channel-mean of exp(G_rj-G_sj)
    // is ~30-90% wrong in this regime; the factorized (Q.Gamma)(K/Gamma)^T
    // must match the fused recurrence to ~1e-4.
    // Note: keys must be L2-normalized (as the module does) or the delta-rule
    // operator (I - (b*k)k^T) is unstable for raw Gaussian keys.
    let g = Tensor::<4>::random([b, h, t, k], Distribution::Uniform(-0.15, -0.01), &device);
    let q = Tensor::<4>::random([b, h, t, k], Distribution::Normal(0.0, 1.0), &device);
    let kt_raw = Tensor::<4>::random([b, h, t, k], Distribution::Normal(0.0, 1.0), &device);
    let kt = kt_raw.clone() / kt_raw.powf_scalar(2.0).sum_dim(3).sqrt();
    let v = Tensor::<4>::random([b, h, t, vd], Distribution::Normal(0.0, 1.0), &device);
    let erase = Tensor::<4>::random([b, h, t, k], Distribution::Uniform(0.0, 1.0), &device);
    let write = Tensor::<4>::random([b, h, t, vd], Distribution::Normal(0.0, 1.0), &device);
    let state = Tensor::<4>::random([b, h, k, vd], Distribution::Normal(0.0, 1.0), &device);

    let (chunk_out, chunk_state) = chunk_wy_forward(
        q.clone(),
        kt.clone(),
        v.clone(),
        g.clone(),
        erase.clone(),
        write.clone(),
        state.clone(),
        0.125,
        c,
    );
    let (fused_out, fused_state) = fused_recurrent_forward(q, kt, v, g, erase, write, state, 0.125);

    let diff = (chunk_out - fused_out).abs().max().into_data();
    let state_diff = (chunk_state - fused_state).abs().max().into_data();
    let to_f32 = |d: burn::tensor::TensorData| {
        d.bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .fold(0.0f32, f32::max)
    };
    let d_out = to_f32(diff);
    let d_state = to_f32(state_diff);
    println!("chunk vs fused with real decay: out={d_out:.2e} state={d_state:.2e}");
    assert!(d_out < 1e-4, "output mismatch {d_out:.2e}");
    assert!(d_state < 1e-4, "state mismatch {d_state:.2e}");
}
