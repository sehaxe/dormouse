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
     "line 1 says 'Bit-exact reference tests' while lines 8-10 say it is NOT bit-for-bit. NOT editable from this lane: gen_reference.rs / ref_data.bin / bit_exact.rs are owned by another worktree (layout defect: token-major buffers read with head-major offsets, which is why exactly the 24 single-token cases pass). Tracked as FINDING 0 in vendor/burn-fused/TEST-AUDIT.md."),
    ("burn-gdn2/tests/fused_adjoint_vs_ops.rs", "d",
     "in-crate `chunk_wy_forward` per-op autograd, on bare tensors",
     "the fused CUDA ADJOINT, on two shapes chosen to localise a wrong term",
     "the fused forward; the state OUTPUT (a leaf on both paths, so `d_s` there is the INPUT state's gradient). SUPERSEDED IN STRENGTH, not removed: `fused_adjoint_f64.rs` compares the same kernels against a gradient computed by a different METHOD, which this cannot do", ""),
    ("burn-gdn2/tests/autodiff_bwd_f64.rs", "b",
     "f64 forward-mode AD + full-tensor central differences over a TRANSCRIPTION of `chunk_wy_forward_batched` (tools/gen_bwd_f64.py, tools/fwd_mode.py), committed as tests/ref_bwd_f64.bin",
     "the analytic adjoint (custom node, f32) on ALL 1792 coordinates of all seven inputs; the FORWARD against the f64 transcription first, which is what bounds the transcription risk; the bar's far side from 10 wrong formulas in ref_bwd_f64_faults.bin, three of which this shape provably cannot exercise and are named",
     "a shared MISREADING of the specification, which survives any transcription; the fused CUDA kernel (CPU build); the inter-chunk half of the adjoint (one chunk, so no BK2 and no d_k_bptt/d_e_bptt/d_s_shift); any two-chunk shape, where the forward disagrees with the transcription at 8.3e-1",
     "TIER (b) NOT (a): the two methods are independent of each other, so the DERIVATIVE is not a transcription, but the FORWARD is. The words bit-exact/bit-for-bit are not used about it."),
    ("burn-gdn2/tests/fused_adjoint_f64.rs", "b",
     "the same f64 oracle (tests/ref_bwd_f64.bin): forward-mode AD cross-checked by central differences over a transcription of the forward",
     "the fused CUDA KERNEL adjoint (BK1) on all seven gradients, and the fused forward against the f64 transcription first",
     "BK2 and the d_k_bptt/d_e_bptt/d_s_shift glue - one chunk, and those are the terms 2a430cc measured at rel 3.3e-1 and 6.2e-1; a shared misreading of the specification; any multi-chunk shape",
     "TIER (b) for the same reason as autodiff_bwd_f64.rs. The file header states the one-chunk scope in its first screen so a green run cannot be read as 'the fused backward is correct'."),
    ("burn-gdn2/tests/fused_chunk_verify.rs", "d", "in-crate `chunk_wy_forward`",
     "the fused CUDA chunk forward; zero-key-row gradient finiteness",
     "its own `fused_op_grads_match_tensor_path_cuda`, which compared the TENSOR adjoint to the tensor path (see that file's header)", ""),
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
     "docstring line 2 says 'Regenerate the bit-exact reference data', a tier-(a) claim about a tier-(c) generator. NOT edited from this lane: the .py is the readable twin of the .rs, and both are owned by the worktree fixing the layout defect. Rename to 'element-exact' or 'reference data' when that lane lands."),
    ("burn-kda/src/lib.rs", "d", "n/a - no fidelity claim in the file",
     "(documents the Kimi Linear / K3 equations, which is provenance, not a comparison)",
     "any numeric claim: no test in this crate compares against either paper",
     ""),
    # -------------------------------------------------------------- burn-rmsnorm
    # The first tier-(a) row in the table. `target` names the two upstreams and
    # WHERE the expected value came from, because that column is the point.
    ("burn-rmsnorm/tests/rmsnorm_oracle.rs", "a",
     "RAN, not transcribed: `torch.nn.functional.rms_norm` (torch==2.14.0+cpu, "
     "pytorch/pytorch v2.14.0, ATen `torch::rms_norm`) and "
     "`fla.modules.layernorm.rms_norm_ref` (flash-linear-attention==0.5.2, "
     "fla-org/flash-linear-attention, sha256(fla/modules/layernorm.py)="
     "e78b729b..c6d6f), both EXECUTED 2026-09-29 on CPU and emitted to "
     "`tests/fixtures/rmsnorm_oracle.txt` at 9 significant digits. Both are "
     "byte-pinned as transcripts under `tests/oracle/upstream/`, so the gate "
     "needs no network. The two upstreams agree with each other to 0.0 "
     "relative on all 12 cases (identical f32), so the expected column is not "
     "a compromise between two opinions. Regeneration command: "
     "tests/oracle/gen_rmsnorm_oracle.py",
     "the eps INSIDE-vs-OUTSIDE-vs-DROPPED question (the decisive cases sit at "
     "rms 5e-5..2e-3, where the gaps are O(0.1..1); the worst is 44538x the "
     "tolerance); the LayerNorm-style centring misreading (1962x); the "
     "reduction axis (122173x on a square cube); eps as a parameter of the "
     "call, checked by running the kernel at eps=1e-5 and eps=0 on ONE tensor; "
     "the per-feature gain broadcast (non-constant gains on 11 of 12 cases)",
     "the fused CUDA kernel (norm_cuda never engages on the trainer's backend - "
     "AGENTS.md 3.3, eval line norm=0/N); WHETHER either upstream faithfully "
     "transcribes arXiv:1910.07467, whose AUTHORS SHIP NO CODE, so tier (a) here "
     "means 'the two public reference implementations of the mechanism', not "
     "'the authors of the paper'. A shared misreading of Zhang & Sennrich "
     "survives this test. The bf16 and f64 paths",
     ""),
    ("burn-rmsnorm/tests/oracle/gen_rmsnorm_oracle.py", "a",
     "the same two upstreams it runs (torch 2.14.0+cpu, flash-linear-attention "
     "0.5.2). It is the GENERATOR, and it REFUSES to write a fixture on which "
     "the upstreams disagree by more than 1e-6, or on which any of the crate's "
     "own guards would be unsatisfiable",
     "is the fixture's provenance: the sha256 of the fla file it read, the "
     "torch tag, and the 9-significant-digit encoding that round-trips f32",
     "anything about the Rust tree - it writes no Rust",
     ""),
    ("burn-rmsnorm/tests/oracle/mutate_kernel.sh", "x",
     "n/a - it runs the crate's own tests against deliberately wrong kernels",
     "that the oracle CAN FAIL: five mutants of `RMSNorm::forward`'s tensor path "
     "(eps outside the sqrt / dropped / hardcoded, gain collapsed to its mean, "
     "reduction over dim 1), each of which turns at least one test red. It "
     "asserts nothing about the numerics itself",
     "-", ""),
    ("burn-rmsnorm/tests/oracle/upstream/torch_rms_norm.py", "a",
     "verbatim `inspect.getsource()` of the two PyTorch entry points that were "
     "executed, from torch 2.14.0+cpu (pytorch/pytorch v2.14.0). TRANSCRIPT of "
     "upstream source, re-extractable with `gen_rmsnorm_oracle.py --dump`",
     "which upstream code was run, so the version in the fixture header is not "
     "an unbacked claim",
     "the arithmetic: it dispatches to COMPILED C++ "
     "(aten/src/ATen/native/layer_norm.cpp::rms_norm), which is not quotable "
     "here and is not pinned by this file",
     ""),
    ("burn-rmsnorm/tests/oracle/upstream/fla_rms_norm_ref.py", "a",
     "verbatim `inspect.getsource()` of `fla.modules.layernorm.rms_norm_ref` "
     "from flash-linear-attention 0.5.2 (fla-org/flash-linear-attention), the "
     "file whose sha256 (e78b729b..c6d6f) is in the fixture header. TRANSCRIPT, "
     "re-extractable with `--dump`",
     "which upstream code was run",
     "any revision handle: the wheel carries NO git revision, so the sha256 is "
     "the only one available and this fixture is tied to that hash rather than "
     "to a commit",
     ""),
    ("burn-rmsnorm/tests/fixtures/rmsnorm_oracle.txt", "a",
     "GENERATED by tests/oracle/gen_rmsnorm_oracle.py from the two upstreams "
     "above; 9 significant digits of the f32 they produced. Do not hand-edit - "
     "a hand-edited golden is exactly what tier (a) exists to replace",
     "carries the right answers AND the four WRONG formulas as separate "
     "columns, so every margin in rmsnorm_oracle.rs is measured rather than "
     "assumed",
     "-", ""),
    ("burn-rmsnorm/src/lib.rs", "d",
     "in-file scalar f64 loop over host memory in `forward_matches_scalar_reference` "
     "(an independent formulation, not the tensor chain restated)",
     "the tensor path against a host-side loop on 3x4x8 with a non-constant gain",
     "the fused CUDA arm (the scalar loop is the CPU path's oracle only); the "
     "tolerance there (1e-4 relative) is ~1000x looser than the tier-(a) "
     "test's 1e-5, which is the reason the tier-(a) test exists",
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
