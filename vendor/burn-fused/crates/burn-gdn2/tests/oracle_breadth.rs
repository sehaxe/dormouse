// The 1000-case breadth test, against the f64 external oracle.
//
// WHAT IT COMPARES. `tests/ref_f64_broad.bin`: 1000 cases, one weight set,
// T in 1..=38, the T sweep of the fixture this file USED to read
// (`tools/gen_reference.rs:482`, `let seq_len = (1usize << (i % 6)) + (i % 7);`),
// with the expected values computed in **f64** by `tools/gen_reference_f64.py`
// instead of in f32 by our own algorithm. The breadth is preserved exactly;
// only the arm the answer comes from changed.
//
// WHAT IT USED TO BE, AND WHY THAT WAS WORTH NOTHING. It read
// `tests/ref_data.bin`, produced by `tools/gen_reference.rs` - an **f32
// transcription of burn-gdn2's own algorithm**. `docs/ORACLE.md` §2 is the
// reason that is not a reference: two independent transcriptions of one
// recurrence agree when they are both right AND when they share a
// misunderstanding, so the test could only ever prove self-consistency. It was
// not hypothetical. `src/short_conv.rs` replicate-padded the depthwise conv's
// left edge where the authors' kernel zero-fills
// (`fla/modules/conv/triton/kernels.py::causal_conv1d_fwd_kernel`, the
// `mask=((o_x >= 0) & (o_x < T))` load with `other=0.0`), and BOTH reference
// generators replicate-padded too - so the fixture agreed with the bug and the
// test was green. `0a6998a` fixed the kernel; the fixture was left stale and
// went red at max_diff 7.90e-3 against `EPSILON = 5e-4`. Regenerating it would
// have been the wrong move: it would have re-committed the bug as the truth.
// `ref_data.bin`, `tools/gen_reference.rs` and `tests/gen_reference.py` are
// DELETED in the same commit that retargets this test, rather than left in the
// tree as a fixture nothing reads.
//
// PROVENANCE, and the words that are therefore NOT used here. The oracle is an
// **f64 transcription of arXiv:2605.22791 §3.1, Eq. 8-12** ("Gated DeltaNet-2"),
// written out one token at a time, plus a committed fixture of its outputs. It
// is **tier (b)**: a transcription, so a shared *misreading* of the paper would
// survive it. It is NOT the original authors' output bytes - those need
// NVlabs/GatedDeltaNet-2's Triton kernel actually run, which is
// `docs/ORACLE.md` §8 candidate (2) and has not been attempted. The words
// "bit-exact" and "bit-for-bit" are not used about this test and must not be;
// that is why the target is named `oracle_breadth` and not `bit_exact`, which
// is what this file was called until 2026-09-29 and was never.
//
// The generator's module docstring cites the source of every non-obvious line,
// including the three that are NOT in the paper: the `K**-0.5` scale applied to
// the WHOLE readout (`lit_gpt/gdn2_ops/chunk_gdn2.py`, `if scale is None: scale
// = k.shape[-1] ** -0.5`, with `fla/ops/gla/chunk.py::chunk_gla_fwd_kernel_o`
// doing `b_o *= scale` on the inter-chunk term only); the short conv's ZERO
// padding; and eps INSIDE the L2 norm's root (`fla/modules/l2norm.py`,
// `1 / tl.sqrt(tl.sum(b_x * b_x) + eps)`) against burn's `sqrt(ss + 1e-6)`.
//
// THE BAR, AND BOTH OF ITS SIDES MEASURED. `BAR = 1e-3` RELATIVE to each case's
// own `max|reference output|` - unchanged from the f64 layer's bar in
// `tests/ref_f64.rs`, so it is not a number invented to pass this test:
//
//   * f32 noise, i.e. what two implementations of the SAME maths in f32 and
//     f64 differ by. MEASURED where it is still re-runnable from the tree, and
//     deliberately NOT quoted from a throwaway probe: the per-stage diff of
//     `project` against this generator's own f64 recomputation, 21 rows at T=1
//     and T=70, whose worst relative disagreement is 5.2e-07 (`k4d`) and whose
//     short-conv rows are 1.4e-07 .. 2.9e-07. Run it with
//         cargo run  -p burn-gdn2 --release --example ref_f64_stages
//         python3 tools/gen_reference_f64.py --diff-stages /tmp/rust_stages.txt
//     The bar is 3.3 orders above that. (An earlier draft of this header told
//     the reader to read the number out of `ref_f64.rs`'s own run output. It
//     cannot: that test is RED and panics at case 1, so the line is never
//     printed. A claim a reader cannot get to is not a measurement.)
//   * the nearest wrong formula. `tests/ref_f64_faults.bin` carries the SEVEN
//     wrong formulas for the same weights and input, and
//     `ref_f64.rs::the_bar_bites_a_wrong_formula` asserts our output is at
//     least `SEMANTIC_FLOOR = 1e-1` from every one of them on every run - so
//     the bar cannot rot into a self-consistency check. Re-measured over THIS
//     1000-case sweep (all seven faults, all 1000 cases, `python3 -
//     ` equivalent of `gdn2_forward(x, P, fault)`), the smallest margin is
//     3.909822e-01 relative - `no-erase-gate` at T=8 - i.e. 391x above the bar.
//     Per fault, worst over the sweep: decay-sign 2.34e+00, transposed-proj
//     3.30e+00, no-write-gate 1.02e+00, no-erase-gate 3.91e-01,
//     read-before-write 1.54e+00, conv-padding 4.94e+00, no-scale 2.69e+00.
//
// IS 1e-3 RELATIVE A TIGHTENING? Yes, and the old number for the record:
// `EPSILON = 5e-4` ABSOLUTE against `ref_data.bin`, whose output scale was
// ~1e-2 - about 5% of the signal. This sweep's per-case output scale runs
// 3.93e-02 (min) / 3.21e-01 (median) / 5.35e-01 (max), so 1e-3 relative is
// 3.9e-05 ABSOLUTE on the worst-conditioned case - about 13x tighter than the
// 5e-4 it replaces, and ~3200x tighter in relative terms. No tolerance was
// widened anywhere in this change. The conditioning is not luck: the fixture
// gives `g_proj_1` a non-zero bias because `lit_gpt/gdn2.py` zeroes every bias,
// which puts the output gate at `silu(0) = 0` exactly and makes a RELATIVE bar
// through a near-null quantity meaningless.
//
// WHAT THIS SWEEP IS BLIND TO, MEASURED, NOT ASSUMED. At T=1 `S` starts at
// zero, so `S̄ = Diag(α)S = 0` and `r = S̄^T e = 0`: neither the decay nor the
// erase gate appears. Injected as faults, `decay-sign` and `no-erase-gate`
// change the T=1 output by EXACTLY 0.0, while the other five change it by
// 0.8 to 2.3 (measured, `tests/ref_f64.rs` header). This sweep contains 24
// T=1 cases (`i % 42 == 0`), so 24 of its 1000 rows are blind to two of the
// seven semantic error classes and to the entire state carry. What covers those
// is the 18-case matrix in `tests/ref_f64.rs` (T = 1..70, both dispatch arms),
// its committed fault fixture, and `test_chunk_matches_fused_with_real_decay`
// in `tests/oracle_chunk.rs` (T=130, chunk vs fused, with a strong non-uniform
// per-channel decay). Three layers, each with a different blind spot; this one
// is the breadth, not the coverage.
//
// REGENERATE, AND HOW TO KNOW IT IS THE SAME BYTES.
//     cd vendor/burn-fused/crates/burn-gdn2
//     python3 tools/gen_reference_f64.py --broad
//     git diff --exit-code -- tests/ref_f64_broad.bin    # must be empty
// The generator is f64 NumPy with a splitmix64 + Box-Muller RNG, so it needs
// numpy but no torch and no GPU; the committed bytes were produced by
// `python3 tools/gen_reference_f64.py --broad` at numpy 2.5.3 and reproduced
// byte-identically on a re-run. Unlike the deleted f32 generator this one CAN
// be regenerated only against an f64 implementation, which is the point.
//
// ======================= STATUS: RED, AND IT IS NOT THE RETARGET ============
//
// The retarget is done. The suite is RED because it found a REAL, PRE-EXISTING
// defect the moment it was pointed at an oracle that is not our own algorithm,
// and that is the layer doing its job. Nothing here is skipped, ignored or
// widened, and the bar is the f64 layer's own 1e-3, unchanged.
//
// MEASURED, this tree, `cargo test --release -p burn-gdn2 --features
// binary-tests --test oracle_breadth --test oracle_chunk`:
//     FusedRecurrent: worst 8.9402e-01 rel at T=3, 976/1000 cases over the bar
//     Chunk (4/8/16/32/64): the same, worst 8.9402e-01 at T=3
//     first failing case: 1 (T=3), 1.851e-01 rel
//
// PRE-EXISTING, PROVEN BY REBUILDING THE BASE. `git stash` of every change in
// this commit, same worktree, then
// `cargo test --release -p burn-gdn2 --test ref_f64`:
//     FAILED - FusedRecurrent case 1 (T=2): max relative deviation 9.227e-02
// That is `dbfa4ca` with none of this work in it. The f64 layer's own test was
// already red before this commit, and its header said so ("STATUS: RED"); the
// two `binary-tests` tests were red for the separate, already-explained reason
// in `0a6998a` (a stale fixture), which is what this commit removes.
//
// WHERE THE DEFECT IS, TO THE MEASUREMENT. `tools/gen_reference_f64.py
// --diff-stages` against `examples/ref_f64_stages.rs`, case 0 (T=1) and case 17
// (T=70), 21 stage rows, all relative:
//
//     stage            T=1         T=70
//     raw_q/k/v_proj   1.4-2.2e-07 2.2-3.3e-07
//     q/k/v_conv       1.7-2.9e-07 1.4-1.9e-07   <- the pad, see below
//     f0 f1 gp0 gp1    1.0-2.7e-07 1.0-2.7e-07
//     g_recomputed     1.2e-07     2.9e-07
//     a_exp dt_bias A_log 1.5e-08 - 2.7e-07
//     q4d k4d g4d b4d v4d w4d gate4d  1.3-5.2e-07  1.6-3.9e-07
//
// So EVERY stage of `project` - all seven per-head 4-D tensors included - is
// right, and the defect is downstream of it. The per-head rows did not exist
// before this commit: `diff_stages` compared only 2-D and 1-D stages, so the
// diagnostic was blind to every per-head tensor, which is exactly where a
// head-layout or GVA-repeat defect would live. The reason is in the example's
// header - burn 0.22.0-pre.4 has no `contiguous()`, and an elementwise op is
// the only way to force a dense readback out of a permuted view.
//
// THE CONV PAD FIX IS CONFIRMED CORRECT, which was the open question.
// `0a6998a` changed `src/short_conv.rs` from replicate to zero padding and
// `ref_f64.rs`'s header still carries the pre-fix numbers. Re-measured against
// the f64 ZERO-pad oracle at both lengths: q_conv 1.683e-07 / 2.884e-07
// (k_conv), v_conv 2.371e-07 at T=1, where the old table read 3.0e+00, 1.4e+00
// and 2.0e+00. The fix is right; the fixture that disagreed with it was wrong.
//
// NOT YET LOCALISED, AND ONE NUMBER I DO NOT TRUST. Per token, over the 18-case
// matrix: token 0 is always right (3.07e-07 at T=1 down to 3.49e-08 at T=64) and
// every token from 1 on is O(1) wrong - so the fault is in the state carry, not
// in `project` and not in the per-token output stage (which is pointwise in T,
// and token 0 exercises it). Exactly 24 of the 1000 cases pass and they are
// exactly the T=1 ones (`i % 42 == 0`), where `S` starts at zero and no carry
// exists.
//
// Dumping the carry's four sub-steps by hand for T=2 (decay, erase, `v_new`,
// and the resulting `S`), all four agree to <= 4.6e-07 at EVERY step, the
// final `S` included - which leaves the readout as the only remaining suspect
// and is by elimination, not by measurement. A separate probe that called the
// crate's own `fused_recurrent_forward` reported a final state 1.34e-01 off on
// the same input, which contradicts that and which I could NOT REPRODUCE. I am
// not claiming either number. The next measurement a follow-up should make is
// the readout `(state * q_t.swap_dims(2, 3)).sum_dim(2)` with a raw (NOT
// squared) readback of `q_t`, because the probe that was supposed to settle it
// squared before `sqrt` - which recovers `|x|`, not `x`, and made every signed
// row read O(1). That is the bug in the probe, and it is why this header claims
// no line.
//
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the
// burn-flex migration.
//
// ================== THIS TEST CAN FAIL: MEASURED, BOTH DIRECTIONS =============
//
// A test nobody has seen fail is the defect class this whole layer exists to
// kill, so here is the number from each side, from commands in this repo.
//
// 1. REINTRODUCE THE PAD BUG. In `src/short_conv.rs`, the `None =>` arm of the
//    cache match, replace the zero pad with the pre-`0a6998a` replicate:
//        let pad = Tensor::zeros([b, SHORT_CONV_CACHE, c], &x.device());
//    ->
//        let pad = x.clone().slice([0..b, 0..1, 0..c])
//            .expand([b, SHORT_CONV_CACHE, c]);
// 2. MEASURE, worst relative deviation over all 1000 cases, FusedRecurrent and
//    Chunk/4, on the same fixture:
//
//        correct zero pad (this tree)   worst 8.9402e-01 at T=3   976/1000 over
//        replicate pad injected         worst 4.9448e+00 at T=1  1000/1000 over
//
//    and the tests' own first failure moves with it:
//        correct   -> case 1 (T=3), 1.851e-01 >= BAR 1e-3
//        injected  -> case 0 (T=1), 1.164e+00  >= BAR 1e-3
//
//    Two things to notice, because both are the point. The magnitude goes up
//    5.5x, and the FIRST FAILING CASE MOVES TO T=1 - the 24 cases the
//    pre-existing state-carry defect leaves passing are exactly the ones that
//    catch the pad, so a sweep that only ever reported "976/1000" would have
//    hidden a whole error class at T=1. `4.9448e+00` also matches, to five
//    figures, what the f64 generator predicts for its `conv-padding` fault
//    measured over the same 1000 cases in Python (4.944795e+00), which is an
//    independent check that the fault model and the test agree.
// 3. RESTORE. `git checkout -- src/short_conv.rs` (or keep a copy before step 1;
//    do not leave the injected pad in a commit).
//
// The same demonstration for `tests/oracle_chunk.rs` needs no second run: it
// reads the same fixture with the same bar through the same `build`, and the
// Chunk numbers above were taken in the same two builds.
#![allow(dead_code, deprecated)]

include!("common/ref_f64.rs");

/// Relative to the case's own max |reference output|. Do NOT widen this to
/// make something green: a discrepancy that a wider bar would hide is the whole
/// reason this layer exists. Same value, same meaning, as `BAR` in
/// `tests/ref_f64.rs`.
const BAR: f64 = 1e-3;

#[test]
#[cfg(feature = "binary-tests")]
fn gdn2_1000_cases_match_the_f64_oracle() {
    let t = load_broad();
    let device = Device::ndarray();
    let v_head = (t.hk as f32 * t.expand_v) as usize;
    println!(
        "fixture ref_f64_broad.bin: d={} h={} hk={} hv={} expand_v={} -> value head \
         {v_head}, VD={}, {} cases, {} tensors",
        t.d, t.h, t.hk, t.hv, t.expand_v, t.hv * v_head, t.cases.len(), t.tensors.len()
    );

    let module = build(&t, Gdn2Mode::FusedRecurrent, 64, &device);
    let mut worst = 0.0f64;
    let mut worst_t = 0usize;
    let mut worst_scale = 0.0f64;
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
            worst_scale = scale;
        }
        let abs = rel * scale;
        assert!(
            rel < BAR,
            "case {i} (T={seq}): max relative deviation {rel:.3e} >= BAR {BAR:.0e} \
             (output scale {scale:.3e}, i.e. {abs:.3e} absolute)"
        );
    }
    let worst_abs = worst * worst_scale;
    println!(
        "FusedRecurrent vs the f64 oracle, {} cases: worst {worst:.3e} rel at T={worst_t} \
         (scale {worst_scale:.3e}, {worst_abs:.3e} abs) against BAR {BAR:.0e}",
        t.cases.len()
    );
}

// --- ndarray timings, carried over from the file this target replaces ---------
// `test_targets.py` counts a target as live on the default feature set if ANY
// `#[test]` survives there, and these two do - so this target deliberately has
// NO `[[test]]` block in Cargo.toml. A block with
// `required-features = ["binary-tests"]` would make cargo SKIP the target
// entirely on the default cell and these benchmarks would never run again.
struct BenchCfg {
    d: usize,
    h: usize,
    hk: usize,
}

const BENCH_MODELS: &[BenchCfg] = &[BenchCfg {
    d: 256,
    h: 4,
    hk: 64,
}];

fn bench_model(label: &str, device: &Device, seq_lens: &[usize]) {
    for bc in BENCH_MODELS {
        for mode in [Gdn2Mode::FusedRecurrent, Gdn2Mode::Chunk] {
            let cfg = Gdn2Config {
                hidden_size: bc.d,
                num_heads: bc.h,
                head_dim: bc.hk,
                num_v_heads: Some(bc.h),
                expand_v: 1.5,
                use_short_conv: true,
                allow_neg_eigval: false,
                norm_eps: 1e-5,
                mode,
                chunk_size: 64,
                min_decay: None,
            };
            let module = GatedDeltaNet2::new(&cfg, device);

            for &seq_len in seq_lens {
                let n_iters = if seq_len >= 4096 {
                    2
                } else if seq_len >= 1024 {
                    5
                } else {
                    20
                };
                let input = Tensor::<3>::zeros([1, seq_len, bc.d], device);
                let mut state: Option<burn_gdn2::Gdn2State> = None;

                for _ in 0..3 {
                    let _ = match mode {
                        Gdn2Mode::FusedRecurrent => {
                            module.forward::<NdArray>(input.clone(), &mut state, true)
                        }
                        Gdn2Mode::Chunk => module.forward_train::<NdArray>(input.clone()),
                    };
                }

                let start = std::time::Instant::now();
                for _ in 0..n_iters {
                    let _ = match mode {
                        Gdn2Mode::FusedRecurrent => {
                            module.forward::<NdArray>(input.clone(), &mut state, true)
                        }
                        Gdn2Mode::Chunk => module.forward_train::<NdArray>(input.clone()),
                    };
                }
                let elapsed = start.elapsed();

                let tok_s = (n_iters * seq_len) as f64 / elapsed.as_secs_f64();
                let per_fwd = elapsed / n_iters as u32;
                let tag = match mode {
                    Gdn2Mode::FusedRecurrent => "FR",
                    Gdn2Mode::Chunk => "CK",
                };

                println!(
                    "{label:>5}/{tag}  d={:>4} h={:>2} S={:>5}  {:>8.0} tok/s  [{:.2?}/fwd]",
                    bc.d, bc.h, seq_len, tok_s, per_fwd,
                );
            }
        }
    }
}

#[test]
fn bench_ndarray_short() {
    bench_model("ND", &Device::ndarray(), &[64, 256]);
}

#[test]
fn bench_ndarray_single() {
    bench_model("ND", &Device::ndarray(), &[64]);
}
