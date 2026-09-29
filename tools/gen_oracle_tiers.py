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
    ("burn-gdn2/tests/bit_exact.rs", "-",
     "DELETED 2026-09-29 along with tests/ref_data.bin and tools/gen_reference.rs. Was tier (c): an f32 transcription of burn-gdn2's OWN algorithm, so it could only prove self-consistency, and both reference generators replicate-padded the short conv exactly as the kernel wrongly did. See the replacement entries below.",
     "-", "nothing; the file does not exist", ""),
    ("burn-gdn2/tests/oracle_breadth.rs", "b",
     "`tests/ref_f64_broad.bin`, emitted by `tools/gen_reference_f64.py`: an f64 transcription of arXiv:2605.22791 3.1 Eq. 8-12, with the three details not in the paper each cited to the authors' own source (`lit_gpt/gdn2_ops/chunk_gdn2.py` for the K**-0.5 scale on the whole readout, `fla/modules/conv/triton/kernels.py` for the ZERO pad, `fla/modules/l2norm.py` for eps inside the sqrt)",
     "PAPER-SEMANTICS; `project` (layout, L2 norm, decay parameterisation, GVA repeat); `short_conv_1d`; `output` (gate, o_norm, o_proj); the STATE carry; 1000 cases at T in 1..=38, the T sweep of the fixture it replaced",
     "the fused CUDA kernel, and the chunked arm (that is oracle_chunk.rs). RED today on a pre-existing state-carry defect, measured worst 8.9402e-01 relative with 976/1000 cases over the 1e-3 bar, and every stage of `project` clean to <= 5.2e-07", ""),
    ("burn-gdn2/tests/fused_adjoint_vs_ops.rs", "d",
     "in-crate `chunk_wy_forward` per-op autograd, on bare tensors",
     "the fused CUDA ADJOINT, on two shapes chosen to localise a wrong term",
     "the fused forward; the state OUTPUT (a leaf on both paths, so `d_s` there is the INPUT state's gradient)", ""),
    ("burn-gdn2/tests/fused_chunk_verify.rs", "d", "in-crate `chunk_wy_forward`",
     "the fused CUDA chunk forward; zero-key-row gradient finiteness",
     "its own `fused_op_grads_match_tensor_path_cuda`, which compared the TENSOR adjoint to the tensor path (see that file's header)", ""),
    ("burn-gdn2/tests/fused_permuted_view.rs", "d", "in-crate, strided vs contiguous views",
     "the seam on NON-CONTIGUOUS input (the stack overflow); contiguity left alone",
     "the numerics of the recurrence", ""),
    ("burn-gdn2/tests/gen_reference.py", "-",
     "DELETED 2026-09-29 with the .rs twin and the fixture they emitted",
     "-", "nothing; the file does not exist", ""),    ("burn-kda/src/lib.rs", "d", "n/a - no fidelity claim in the file",
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
