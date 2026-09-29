#!/usr/bin/env python3
"""Regenerate docs/ORACLE-TIERS.tsv from the table below.

Kept as a script because the TSV is the one artifact in this lane that must be
line-exact, and a hand-edited tab-separated file loses rows.
"""
import os

ROWS = [
    # ---------------------------------------------------------------- burn-kda
    ("burn-kda/tests/cuda_gate.rs", "d",
     "in-crate `KdaModule::forward_recurrent` (per-token scan) + `burn_gdn2::seam_counts`",
     "chunked-WY algebra vs the exact scan; SEAM (which arm ran); nonzero parameter grad on AdBal",
     "anything in `project`/`output` (shared by all three arms); the fused CUDA kernel (this fixture DECLINES it on purpose); paper semantics", ""),
    ("burn-kda/tests/ops_grad_cuda.rs", "d",
     "central finite differences of the ops forward (independent METHOD, not an arm)",
     "the chunk op's adjoint on the trainer's backend; a real `q_proj.weight` derivative; 8 coordinates/tensor",
     "the fused CUDA kernel; PAPER semantics", ""),
    ("burn-kda/tests/fused_cuda.rs", "d",
     "in-crate `forward_recurrent` + `burn_gdn2::chunk_wy_forward`",
     "the fused chunk CUDA kernel forward; the chunked algebra; both DecayFn forms",
     "`project`/`output`; the fused gradient (asserts only nonzero); ragged T; the fused DECODE kernel (KDA dispatches to the tensor scan, not a kernel)", ""),
    ("burn-kda/tests/bench_cuda.rs", "x", "none", "(nothing asserted)", "-", ""),
    ("burn-kda/examples/bitforbit.rs", "d",
     "in-crate `forward_recurrent` + `chunk_wy_forward`",
     "dumps tensors for an off-tree FLA cross-check; asserts nothing itself",
     "any claim of bit-exactness - both sides are our own arms",
     "R5 (the FILENAME says bit-for-bit) is waived, not fixed: renaming collides with a `[[example]] name =` entry in burn-kda/Cargo.toml, two dated research logs that quote `cargo run --example bitforbit`, and a generated graphify report. The docstring is corrected in this lane and now says so in its first line; the RENAME is a follow-up for whoever owns burn-kda."),
    ("burn-kda/examples/bitforbit_cuda.rs", "d",
     "in-crate `forward_recurrent`, same tensors as the CPU dump",
     "fused chunk kernel vs the CPU-dumped scan inputs; asserts nothing itself",
     "any claim of bit-for-bit - both sides are our own arms",
     "R5 (the FILENAME) waived for the same reason as bitforbit.rs, plus Cargo.toml:35 declares `name = \"bitforbit_cuda\"` explicitly, so the rename is a Cargo edit and not a `git mv`. Docstring corrected here."),
    ("burn-kda/examples/kda_bench.rs", "x", "none", "(nothing asserted)", "-", ""),
    ("burn-kda/examples/kda_step_probe.rs", "x", "none", "(nothing asserted)", "-", ""),
    # ---------------------------------------------------------------- burn-gdn2
    ("burn-gdn2/tests/alloc_probe.rs", "x", "none",
     "all 4 tests `#[ignore]`d; allocator counters only", "any correctness claim", ""),
    ("burn-gdn2/tests/autodiff_chunk.rs", "d",
     "in-crate `chunk_wy_forward` (ops) + central finite differences",
     "CHUNK (custom-node forward vs per-op); GRAD (analytic adjoint vs FD, but 1 argmax coordinate per input at 5% rel)",
     "the fused CUDA kernel (CPU build); `project`/`output`; every coordinate the 7 probes do not land on", ""),
    ("burn-gdn2/tests/autodiff_cuda_gate.rs", "d",
     "in-crate `chunk_wy_forward` + seam counters",
     "backend gate decision; the fused kernel reaching a Balanced graph; fused-vs-ops numerics; a wrong `d_k`/`d_g`",
     "`project`/`output`; any non-CUDA backend", ""),
    ("burn-gdn2/tests/autodiff_nested_balanced.rs", "d",
     "in-crate, NoCheckpointing vs BalancedCheckpointing",
     "the checkpoint-strategy transposition; that the op declines a nested graph",
     "the numerics; the fused kernel", ""),
    ("burn-gdn2/tests/autodiff.rs", "d",
     "in-crate `fused_recurrent_forward` (scan) vs `chunk_wy_forward`; decode vs prefill",
     "CHUNK; STATE carry; prefill->decode handoff; 5 config-refusal tests",
     "`project`/`output` (used as FIXED input); the fused CUDA kernel", ""),
    ("burn-gdn2/tests/b5_seam_probe.rs", "d",
     "hand-written host rank-5 references in the same file",
     "LAYOUT: swap_dims(3,4)/matmul/cumsum/slice/permute on rank 5",
     "anything about the recurrence", ""),
    ("burn-gdn2/tests/bench_cuda.rs", "d",
     "in-crate `fused_recurrent_forward`; `fused_step` directly at :204",
     "the fused DECODE CUDA kernel, per token, state and output. NOTE: the file is named bench but :204 is a real assert!; only `bench_cuda` at :87 is `#[ignore]`d",
     "`project`/`output`; the fused CHUNK kernel", ""),
    ("burn-gdn2/tests/bench_fused_bwd.rs", "x", "none",
     "times the fused adjoint and discards the result", "-", ""),
    ("burn-gdn2/tests/bench_tracks.rs", "x", "none", "(nothing asserted)", "-", ""),
    ("burn-gdn2/tests/bench_train_cuda.rs", "x", "none", "(nothing asserted)", "-", ""),
    ("burn-gdn2/tests/bit_exact.rs", "c",
     "`tests/ref_data.bin`, emitted by `tools/gen_reference.rs`, itself a transcription of NVlabs/GatedDeltaNet-2 `lit_gpt/gdn2.py` with the Triton kernel replaced by a per-token scan - the ONLY external-ish anchor in the two crates",
     "PAPER-SEMANTICS; `project` (layout, L2 norm, decay parameterisation, GVA repeat); `short_conv_1d`; `output` (gate, o_norm, o_proj); STATE carry; both modes, 1000 cases",
     "NOTHING today: the suite is RED (976/1000, 1.38e-2 vs 5e-4) and `binary-tests` is not a default feature, so it catches nothing in any normal run. The fused CUDA kernel",
     "line 1 says 'Bit-exact reference tests' while lines 8-10 say it is NOT bit-for-bit. STALE AS OF 2026-09-29 in its REASON, corrected here: the layout defect this waiver blamed (token-major buffers read with head-major offsets) was FIXED by `ff7cd57` - 1000 cases, max_diff 2.32e-7, 0 failures - and `gen_reference.rs:310-315` now points at the fix. The `cannot` column's red is therefore no longer the state of that suite. The WAIVER stands and the FILE is not edited from this lane, because `bit_exact.rs` is owned by the `wt/gdn2f64` lane; what is left is the tier-(a) wording in the filename and the line-1 header, which is a rename that lane should make."),
    ("burn-gdn2/tests/fused_adjoint_vs_ops.rs", "d",
     "in-crate `chunk_wy_forward` per-op autograd, on bare tensors",
     "the fused CUDA ADJOINT, on two shapes chosen to localise a wrong term",
     "the fused forward; the state OUTPUT (a leaf on both paths, so `d_s` there is the INPUT state's gradient)", ""),
    ("burn-gdn2/tests/fused_chunk_verify.rs", "d", "in-crate `chunk_wy_forward`",
     "the fused CUDA chunk forward; zero-key-row gradient finiteness",
     "its own `fused_op_grads_match_tensor_path_cuda`, which compared the TENSOR adjoint to the tensor path (see that file's header)", ""),
    ("burn-gdn2/tests/fused_launch_count.rs", "d",
     "two hand-written CUDA kernels in the SAME file (spec-flashkda.md §1: fwd_kernel1.cuh + fwd_kernel2.cuh)",
     "the launch COUNT of the module path vs the scratch path, i.e. that a call that grew an extra kernel is visible; nothing numeric",
     "any correctness claim - this is a counter, and a count cannot see a wrong answer",
     "registered (d) and not (b): the two kernels it compares against are in this file, so arm-vs-arm is the whole comparison. Requires `cuda,autodiff` and does not run in the default cell."),
    ("burn-gdn2/tests/fused_permuted_view.rs", "d", "in-crate, strided vs contiguous views",
     "the seam on NON-CONTIGUOUS input (the stack overflow); contiguity left alone",
     "the numerics of the recurrence", ""),
    ("burn-gdn2/tests/gen_reference.py", "c",
     "NVlabs/GatedDeltaNet-2 `lit_gpt/gdn2.py`, read by hand",
     "is the transcription itself, and the fixture it emits",
     "anything about the Rust tree", ""),
    ("burn-gdn2/tests/lowp_bf16_cuda.rs", "b",
     "IEEE-754 round-to-nearest-even, via the third-party `half` crate - the one reference in the tree that is neither our code nor our transcription",
     "bf16 STORAGE as u16 bit patterns + f32 accumulation; the f32-only dtype gate",
     "the recurrence; anything else",
     "'bit-exact' is CORRECT here and the one legitimate use of the word outside tier (a): the expected value is `half::bf16::from_f32`, a third-party implementation of IEEE-754, not another arm and not our transcription. Registered (b) not (a) because the `half` crate is not the mechanism's authors' code, so ADR-0020's AUTHORS label would be a different kind of false claim."),
    ("burn-gdn2/tests/ops_batched_autodiff.rs", "d",
     "central finite differences of the plain path + cross-arm",
     "RAGGED tail (T not divisible by chunk) in the custom node's backward; both arms",
     "`project`/`output`; the fused kernel", ""),
    ("burn-gdn2/tests/ops_batched_bench_cuda.rs", "x", "none", "(nothing asserted)", "-", ""),
    ("burn-gdn2/tests/ops_batched_diff.rs", "d",
     "in-crate `chunk_wy_forward_loop` (the untouched production arm)",
     "CHUNK: the finite-Neumann rewrite, 9 cases incl. a hostile one and a ragged tail; TILE routing past 32",
     "`project`/`output`; the fused kernel; gradients", ""),
    ("burn-gdn2/tests/ops_batched_grad_cuda.rs", "d",
     "central finite differences of the same forward on `Autodiff<CudaBare, BalancedCheckpointing>`",
     "GRAD for BOTH arms on the trainer's backend; 8 coordinates/tensor at a 5% rel bar",
     "`project`/`output`; the fused kernel", ""),
    ("burn-gdn2/tests/official_forward.rs", "b",
     "arXiv:2605.22791 Eq 9/10/11/12 and §3.5, re-derived in f64 per token in THIS file - a transcription, and a no-fixture one, so it is runnable rather than committed",
     "PAPER-SEMANTICS across 10 configs x 2 entry points; its own short conv, its own head-major layout, its own reduction order (it shares no code with the chunked path); TOL = 1e-5, measured 5.04e-09",
     "the authors' own Triton kernel (never run); the 4-D/strided stages the example dumps",
     "the f64 layer (ref_f64.rs) is the fixture-based sibling of this one; the two disagree in TIER, not in verdict - this file has no committed bytes, so a reader can run it, and ref_f64.rs is the one CI can diff. `research/papers/gdn-kda.md` carries the arXiv id because the file cites the equations through that report rather than the paper directly."),
    ("burn-gdn2/tests/ref_f64.rs", "b",
     "`tests/ref_f64.bin`, emitted by `tools/gen_reference_f64.py`, itself an f64 transcription of arXiv:2605.22791 Eq 9-12 with the three non-paper lines (the 1/sqrt(K) scale, the short conv's ZERO left pad, the L2-norm eps inside the sqrt) each citing the file it came from",
     "PAPER-SEMANTICS; `project`; `short_conv_1d`; `output`; the STATE carry; both dispatch arms; 18 cases T=1..70",
     "the authors' own Triton kernel (never run - ORACLE.md §8 candidate (2), not attempted). A shared MISREADING of the paper survives a transcription, which is why this is (b) and not (a)",
     "RED until 2026-09-29 and the cause was THIS LAYER, not the kernel: the generator reshaped `g` head-major while q/k/b/v used the token-major idiom, and the two coincide at T=1 only, so case 0 passed and case 1 read 9.227e-02. The whole episode is in the file's STATUS block. It is a second instance of the `ff7cd57` generator-layout defect, in a file written after that fix."),
    ("burn-gdn2/tests/common/ref_f64.rs", "b",
     "the same `ref_f64.bin` + `tools/gen_reference_f64.py` as ref_f64.rs; it is the shared reader, not a test of its own (a leading-underscore-free file in tests/ would be)",
     "the FIXTURE FORMAT: magic, header, and the 'no trailing bytes' assertion. Asserts nothing about any number",
     "everything numeric - it contains no comparison at all, only the loader and the module builder the two callers share",
     "registered as a file because R1 scans tests/ mechanically and an `include!`d helper is not a test target. Waiving it would leave the loader undeclared, and the loader is where a silent format change would land."),
    ("burn-gdn2/examples/ref_f64_stages.rs", "b",
     "the same f64 transcription as ref_f64.rs, but per-STAGE: `tools/gen_reference_f64.py --diff-stages` reads this example's dump and compares each stage separately",
     "the DIAGNOSTIC, not a verdict: it names the FIRST stage of `project` that disagrees with the f64 transcription, which is how the 2026-09-29 `g`-reshape defect was localised in minutes rather than by bisecting the layer",
     "the 4-D permuted stages, which it does NOT compare: this backend's `into_data()` on a strided view does not read in a stable order, so a 4-D row would be a false positive (see the example's header)",
     "it is an example, not a test: `cargo test` never runs it, so it can never be the thing that keeps a regression out. It is the instrument, and the gate that uses it is the one in the tier (b) sibling."),
    ("burn-gdn2/tests/test_chunk.rs", "c",
     "`tests/ref_data.bin` (the same transcription as bit_exact.rs)",
     "PAPER-SEMANTICS at 5 chunk sizes; scan vs chunk under REAL decay",
     "NOTHING today: behind the same RED `binary-tests` feature", ""),
    ("burn-gdn2/tools/gen_reference.rs", "c",
     "NVlabs/GatedDeltaNet-2 `lit_gpt/gdn2.py`, read by hand",
     "is the runnable generator; CI diffs its output against the committed fixture",
     "anything about the Rust tree", ""),
    # ---- files outside tests/ that still make a fidelity claim -----------
    ("burn-gdn2/src/lib.rs", "c",
     "`tests/ref_data.bin` (our transcription, same as bit_exact.rs)",
     "a feature list",
     "its `binary-tests` bullet said 'bit-exact reference tests', which is a tier-(a) claim about a tier-(c) fixture. ADR-0020 listed the same defect in the README; the doc comment was missed and is fixed in this lane",
     ""),
    ("burn-gdn2/tests/gen_reference.py", "c",
     "NVlabs/GatedDeltaNet-2 `lit_gpt/gdn2.py`, read by hand",
     "is the readable transcription, and the fixture it emits",
     "anything about the Rust tree",
     "docstring line 2 says 'Regenerate the bit-exact reference data', a tier-(a) claim about a tier-(c) generator. NOT edited from this lane: the .py is the readable twin of the .rs, and both belong to the `wt/gdn2f64` lane. Rename to 'element-exact' or 'reference data' when that lane lands. The waiver's original reason - that the .rs had a layout defect - is STALE as of 2026-09-29: `ff7cd57` fixed it."),
    ("burn-kda/src/lib.rs", "d", "n/a - no fidelity claim in the file",
     "(documents the Kimi Linear / K3 equations, which is provenance, not a comparison)",
     "any numeric claim: no test in this crate compares against either paper",
     ""),
]

HEADER = """# ORACLE-TIERS.tsv - what every reference comparison in burn-kda / burn-gdn2 is
# ACTUALLY compared against. Read by tools/oracle_gate.py; the prose form is
# docs/ORACLE.md. Tiers are ADR-0020's: (a) AUTHORS (b) TRANSCRIPTION
# (c) TRANSCRIPTION-OF-TRANSCRIPTION (d) NOTHING. (x) = not a correctness test.
#
# "target" is the thing the expected value came from. If it names a function in
# this fork, the comparison is arm-vs-arm and the tier is (d), whatever the
# test's own name says. That single column is the whole point of the file.
#
# tab-separated. Waive a real violation by putting a reason in the LAST column;
# an empty waiver is a FAIL. Waived rows print as DEBT on every run, so the debt
# shows up in a diff instead of in somebody's memory.
#
# Generated by tools/gen_oracle_tiers.py - do not hand-edit; tab-separated files
# lose rows.
"""


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = [HEADER, "file\ttier\ttarget\tcatches\tcannot\twaiver\n"]
    for rel, tier, target, catches, cannot, waiver in ROWS:
        for cell in (rel, tier, target, catches, cannot, waiver):
            assert "\t" not in cell and "\n" not in cell, cell[:40]
        full = "vendor/burn-fused/crates/" + rel
        out.append("\t".join((full, tier, target, catches, cannot, waiver)) + "\n")
    p = os.path.join(root, "docs", "ORACLE-TIERS.tsv")
    with open(p, "w", encoding="utf-8") as fh:
        fh.writelines(out)
    print("wrote %s: %d rows" % (p, len(ROWS)))


if __name__ == "__main__":
    main()
