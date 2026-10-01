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
// transcription of dormouse-gdn2's own algorithm**. `docs/protocols/ORACLE.md` §2 is the
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
// `docs/protocols/ORACLE.md` §8 candidate (2) and has not been attempted. The words
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
//         cargo run  -p dormouse-gdn2 --release --example ref_f64_stages
//         python3 tools/gen_reference_f64.py --diff-stages /tmp/rust_stages.txt
//     The bar is 3.3 orders above that. (An earlier draft of this header told
//     the reader to read the number out of `ref_f64.rs`'s own run output. It
//     cannot: that test is RED and panics at case 1, so the line is never
//     printed. A claim a reader cannot get to is not a measurement.)
//   * the nearest wrong formula. `tests/ref_f64_faults.bin` carries the EIGHT
//     wrong formulas for the same weights and input, and
//     `ref_f64.rs::the_bar_bites_a_wrong_formula` asserts our output is at
//     least `SEMANTIC_FLOOR = 1e-1` from every one of them on every run - so
//     the bar cannot rot into a self-consistency check. Re-measured over THIS
//     1000-case sweep (all eight faults, all 1000 cases, `python3 -
//     ` equivalent of `gdn2_forward(x, P, fault)`), the smallest margin is
//     4.084390e-01 relative - `output-gate-sigmoid` at T=2 - i.e. 408x above
//     the bar. Per fault, worst over the sweep: conv-padding 4.94e+00 (T=1),
//     transposed-proj 3.30e+00 (T=1), no-scale 2.69e+00 (T=1), decay-sign
//     2.06e+00 (T=36), read-before-write 1.51e+00 (T=2), no-write-gate
//     1.02e+00 (T=1), no-erase-gate 5.27e-01 (T=35), output-gate-sigmoid
//     4.08e-01 (T=2).
//
// IS 1e-3 RELATIVE A TIGHTENING? Yes, and the old number for the record:
// `EPSILON = 5e-4` ABSOLUTE against `ref_data.bin`, whose output scale was
// ~1e-2 - about 5% of the signal. This sweep's per-case output scale runs
// 3.93e-02 (min) / 3.29e-01 (median) / 5.49e-01 (max), so 1e-3 relative is
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
// REGENERATE, AND HOW TO KNOW IT IS THE SAME BYTES - BY RUNNING THE CHECKER,
// NOT BY `git diff`. The recipe this file used to print was
//     python3 tools/gen_reference_f64.py --broad
//     git diff --exit-code -- tests/ref_f64_broad.bin    # must be empty
// and it was WRONG. The generator is f64 NumPy with a splitmix64 + Box-Muller
// RNG (numpy but no torch, no GPU) and the f32 weights and inputs DO reproduce
// bit-exactly, so that half of the claim was fine. The f64 OUTPUTS do not: they
// are numpy reductions whose summation order depends on the array's memory
// layout, so a correct regeneration lands within ~7e-16 relative of the
// committed bytes rather than on them. Measured against all three committed
// f64 fixtures: `ref_f64.bin` <= 9.7e-16, `ref_f64_faults.bin` <= 4.4e-15. So
// `git diff --exit-code` FAILS on a correct regeneration, and the person
// following it learns to ignore it - which is one of the two ways this fixture
// went stale for a day without anyone noticing.
//
//     cd vendor/dormouse-fused/crates/dormouse-gdn2
//     python3 tools/check_f64_fixtures.py
//
// That is the check, and it runs in `tools/lib_gate.sh`. It regenerates all
// three f64 fixtures into a temp dir and compares: weights and inputs must be
// bit-identical (they come off a deterministic stream, so any difference is a
// changed input), outputs to `OUT_TOL = 1e-12` - 3.6x above the worst observed
// round-off and 4.1e+11 below the smallest wrong formula, i.e. with margin on
// both sides. It is not a byte-diff because a byte-diff is not a measurement
// of anything.
//
// ======================== STATUS: IT WAS THE FIXTURE ==========================
//
// This suite was red, and the reason recorded here for a day was WRONG. The
// previous version of this block said it was red because it had "found a REAL,
// PRE-EXISTING defect" in the state carry, and it pointed at a hand-dumped T=2
// carry and at a probe of the readout. There is no such defect. The 1000-case
// fixture had stopped being the output of its own generator, and the deviation
// the whole investigation was measuring was the FIXTURE's, not the kernel's.
//
// `ca45600` changed one line of the f64 generator's maths:
//
//     -    g = g.T.reshape(H, t, HK)                     # head-major
//     +    g = g.T.reshape(t, H, HK).transpose(1, 0, 2)  # token-major, then [H,T,HK]
//
// That is the `ff7cd57` defect described in the same commit's own comment
// block - a head-major reshape of a token-major buffer, identical at t == 1 and
// wrong at every t > 1. The commit regenerated `ref_f64.bin` and
// `ref_f64_faults.bin` and did NOT regenerate `ref_f64_broad.bin`. One fixture
// of the three was left holding the bug. Nothing in the tree could notice:
// there was no check that a fixture matches its generator, and the only recipe
// on record (`git diff --exit-code`) fails on a CORRECT regeneration too.
//
// PROOF, both directions, neither needing a GPU or a build.
//   * Re-run the current generator and diff against the committed bytes: the
//     WEIGHTS and INPUTS are bit-identical (max |A-B| = 0.0 across all 1000
//     cases) and only the outputs move.
//   * Re-run the generator with that ONE line reverted and diff again: all 1000
//     cases agree to 4.1e-16 .. 9.7e-16 RELATIVE, i.e. f64 round-off. The
//     committed `ref_f64_broad.bin` IS the pre-`ca45600` generator's output, to
//     the last bit it can be pinned to.
//   * `python3 tools/check_f64_fixtures.py` names it unaided: `ref_f64.bin` OK,
//     `ref_f64_faults.bin` OK, `ref_f64_broad.bin` STALE - "OUTPUTS DIFFER:
//     worst 8.940234e-01 rel at case 715 (T=3)".
//
// THE PREVIOUS BLOCK'S OWN NUMBERS WERE THE FIXTURE'S, TO FIVE FIGURES, AND
// THAT IS WHAT SOLD THE MISDIAGNOSIS. Current generator vs the stale committed
// bytes, all 1000 cases:
//
//     cases over BAR 1e-3:  976/1000                  <- its "976/1000 over the bar"
//     worst                  8.940234e-01 at T=3      <- its "worst 8.9402e-01 at T=3"
//     first failing case     1 (T=3), 1.851e-01 rel   <- its "case 1 (T=3), 1.851e-01"
//     cases that pass        exactly the 24 with i % 42 == 0, all T=1
//                                                          <- "exactly 24 of the 1000
//                                                              cases pass and they are
//                                                              exactly the T=1 ones"
//
// All four are properties of the stale fixture, and a throwaway harness reading
// that same fixture reproduced all four while being read as a kernel
// measurement. The T=1 coincidence is the fingerprint of the defect CLASS: a
// head-major and a token-major split index identically at t == 1, so precisely
// the degenerate cases agreed - which is what made a layout bug present as a
// state-carry bug, and why "the fault is in the carry, `project` is clean" was
// consistent with the evidence available and wrong anyway.
//
// PRODUCTION IS RIGHT. `src/module.rs`'s `to_4d` is
// `reshape([b, tt, n, d]).permute([0, 2, 1, 3])` - token-major split, then
// permute - i.e. the FIXED form, and the whole recurrence
// (`src/kernel/fused_recurrent.rs`) transcribes Eq. 9/10 with the readout from
// the state AFTER the write. Against the regenerated fixture this file is green.
//
// `0a6998a` IS EXONERATED AND WAS NEVER INVOLVED, and both facts are readable
// in the source without running anything. Its GVA half lives inside
// `if hv > h` and this fixture is H=4, HV=4, so that block never executes; its
// conv-pad half moved production TOWARD this fixture (replicate -> zeros, and
// the generator has zero-padded since `8162672`), so reverting it moves away.
// The revert-scratch that would have settled it was never needed, and the
// "T < kernel width" reading of the T=3 failure was a coincidence: T=3 is simply
// the first case in the sweep, and the failing region is every T > 1.
//
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the
// burn-flex migration.
//
// ================== THIS TEST CAN FAIL: MEASURED, BOTH DIRECTIONS =============
//
// A test nobody has seen fail is the defect class this whole layer exists to
// kill, so here is the number from each side, from commands in this repo.
//
// 1. STALE FIXTURE - the side that was actually exercised, and the one that
//    went unnoticed for a day. Revert the one generator line named in the
//    STATUS block (`g.T.reshape(H, t, HK)`) and re-run
//    `python3 tools/check_f64_fixtures.py`; it reports
//        ref_f64.bin        OK
//        ref_f64_faults.bin OK
//        ref_f64_broad.bin  STALE - worst 8.940234e-01 rel at case 715 (T=3)
//    and the test's own first failure reproduces at its documented value:
//        case 1 (T=3): max relative deviation 1.851e-1 >= BAR 1e-3
//                     (output scale 1.201e-1, i.e. 2.223e-2 absolute)
//    976 of 1000 cases over the bar, and the 24 that pass are exactly the
//    T=1 ones. Restoring the line and regenerating turns it green, and prints:
//
//        FusedRecurrent vs the f64 oracle, 1000 cases: worst 8.512e-07 rel at
//        T=3 (scale 1.110e-1, 9.451e-08 abs) against BAR 1e-3
//
//    i.e. 1.851e-01 -> 8.512e-07, a factor of 2.2e5, and the remaining 8.5e-07 is
//    f32 rounding (the same band the stage diffs in `ref_f64.rs` measure).
//
// 2. REINTRODUCE THE PAD BUG - the direction that proves the pad is still
//    covered. In `src/short_conv.rs`, the `None =>` arm of the cache match,
//    replace the zero pad with the pre-`0a6998a` replicate:
//        let pad = Tensor::zeros([b, SHORT_CONV_CACHE, c], &x.device());
//    ->
//        let pad = x.clone().slice([0..b, 0..1, 0..c])
//            .expand([b, SHORT_CONV_CACHE, c]);
//    then measure, worst relative deviation over all 1000 cases:
//        correct zero pad       worst 8.512e-07 at T=3        0/1000 over the bar
//        replicate pad injected worst 4.9448e+00 at T=1      1000/1000 over
//    and the test's own first failure moves to case 0 (T=1) at 1.164e+00, i.e.
//    T=1 - the 24 degenerate cases a layout bug cannot touch are exactly the
//    ones that catch the pad, so a sweep that only ever reported a count would
//    have hidden a whole error class. `4.9448e+00` matches, to five figures,
//    what the f64 generator predicts for its `conv-padding` fault over the same
//    1000 cases in Python (4.944795e+00), which is an independent check that
//    the fault model and the test agree.
// 3. RESTORE. `git checkout -- src/short_conv.rs` (or keep a copy before step 2;
//    do not leave the injected pad in a commit).
//
// The same demonstration for `tests/oracle_chunk.rs` needs no second run: it
// reads the same fixture with the same bar through the same `build`, and the
// Chunk numbers above were taken in the same builds.
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
        let mut state: Option<dormouse_gdn2::Gdn2State> = None;
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
                let mut state: Option<dormouse_gdn2::Gdn2State> = None;

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
