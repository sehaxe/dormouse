# f16 tensor-core through burn: measured 2026-10-01

Lane: verify the f16 tensor-core path now that the `SizedType` fix is in the
vendored cubecl-ir (`src/types/scalar.rs:240`, root `[patch.crates-io]`,
AGENTS §2.1/§2.5 reconciled in `79762a2`). The gate that answers it:
`crates/backend-parity/tests/f16_gemm_perf.rs` — four `#[ignore]`d arms, one
dtype per process (an unpatched f16 run poisons the device runner; see the
file's module docs).

## Results

All four arms green on a quiet card, 2026-10-01 21:49-21:56 (min of 3
repeats × 10 drained iterations each; logs `~/logs/f16tc-*.log`; worktree
`wt/f16tc` off `a0a1dce`, target `/mnt/.../target-f16tc`):

| shape | dtype | ms | TFLOP/s | reference |
|---|---|---|---|---|
| 5120×768×2048 | fp32 | 1.96 | 8.24 | POC measured 2.27 ms / 7.1 |
| 5120×768×2048 | **f16** | **0.43 / 0.40** (two runs) | **37.7 / 40.1** | cuBLAS f16 41.4 → **91-97 % of ceiling** |
| 5120×2048×8192 | fp32 | 19.33 | 8.89 | pre-patch 16.60 / 10.35 |
| 5120×2048×8192 | **f16** | **4.32** | **39.75** | cuBLAS ceiling 3.93 ms / 43.7 → **91 %**; pre-patch this arm was 20.02 ms / 8.58, i.e. SLOWER than fp32 |

**Verdict: the f16 tensor-core path is alive through burn.** The `SizedType`
panic appears in none of the run logs (grep: 0 hits). The wins over the
sibling fp32 arm are 4.5-4.8×, over the pre-patch f16 arm 4.6-7×, and the
absolute rate sits at ~91-97 % of what the raw cuBLAS route (zero-copy
`crates/cublas-poc`) measured on the same shapes. Correctness (1e-2 relative
against the fp32 result of the same product, asserted before the clock)
passed on both shapes; the small-shape class stays green in
`backend_parity.rs` (8 passed / 2 ignored, the two ignored are the bf16
design gap, ADR-0016 bug 2).

What 34-40 TFLOP/s also proves: the winner is a tensor-core kernel (CUDA-core
fp16 on this card tops out near the fp32 rate, ~9 TF), so a cmma/mma candidate
compiles, wins the autotune, and executes - the whole chain the patch was
supposed to unblock.

## The one defect found and fixed: a gate that fails on success

The file's poisoned-runner guard was a wall-clock floor (`ms > 1.0`), written
pre-patch when f16 was the slow arm. The genuine 0.474 ms win tripped it and
the first f16-model run FAILED on success - the mirror image of "a gate that
cannot fail" (.bulba/memory.md:54). Rewritten as a physical ceiling
(TFLOP/s > 60 f16 / 25 fp32 ⇒ broken runner; cuBLAS measured 43.7 / 13.2 and
burn has never beaten cuBLAS), with the f16 correctness assert as the other
half: a poisoned runner returns a wrong answer and dies there, not at the
clock. After the fix the arm passed twice: 37.7 and 40.1 TFLOP/s.

## Now visible: the autotune candidates that never could run

The vendored COUNTED warn (`Candidate::fail`) fired 28 times per f16-model
run across 20 unique candidate names - `*_cmma`/`*_mma` tile variants refusing
with "No tile size is available for the problem" and `*_tma` variants with
"TMA is not available". Identical sets in the fp32 arms: a shape/tile-coverage
limitation of those candidate families, not an f16 problem. Before this
commit every one of those skips was silent - the exact seam that let the f16
fallback masquerade as a working path for the life of the project. A
follow-up lane that wants more of the cuBLAS gap has a concrete target list
in any run log.

## Scope correction first: which graph carries the fix

The lane brief said `cargo check -p burn-gdn2 --features cuda`. That check
runs in the `vendor/dormouse-fused` workspace, and **that workspace does not
carry the patch**: its root manifest has no `[patch.crates-io]` and its
`Cargo.lock` resolves `cubecl-ir 0.11.0-pre.4` from the registry (verified
2026-10-01), which does not contain the `SizedType` impl. Any f16 test built
inside `dormouse-fused` would test the UNPATCHED stack and report the old
silent fallback as the answer. The fix is only visible through the **root**
workspace graph, which is where `backend-parity` (and `cublas-poc`) live. All
measurements below are root-graph. This is worth knowing for every future
"test a vendored-cubecl change" lane: the vendor patch table and the library
workspace are two different dependency graphs.

## Results

<!-- MEASURE: table filled from the runs below -->

## Method

Per arm: `cargo test --release -p backend-parity --features cuda --test
f16_gemm_perf -- --nocapture --ignored --exact <name>`. Deterministic LCG
operands shared by both dtypes; correctness checked before the clock for f16
(1e-2 relative, the `backend_parity.rs` band); warmup outside the clock (JIT +
autotune land there); the figure is the minimum of 3 repeats × 10 drained
iterations (`sum().into_scalar()` drain — execution time, not submission
time). Shapes: PROD 5120×2048×8192 (the pre-patch 16.60/20.02 ms rows) and
MODEL 5120×768×2048 (the shape the cuBLAS 41.4 / burn-f16 5.6 TFLOP/s
comparison was taken on, `2026-09-27-cublas-integration-poc.md`).

Accelerated-vs-fallback is decided by three signals, in order of trust: the
TFLOP/s ratio against the sibling fp32 arm; stderr (a dying candidate now
prints `autotune: candidate '...' failed and was skipped: ...` — see the
COUNTED-warn commit — and/or the original `SizedType` panic text); the
poisoned-runner guard (`ms > 1.0`, because an unpatched f16 warmup panic once
made later ops report 0.04 ms).

## What changed in the tree

1. **COUNTED fallback marker** (ADR-0019): vendored
   `cubecl-fix/cubecl-runtime/src/tune/schedule.rs` `Candidate::fail` now
   `log::warn!`s the candidate name and error. Before this, a candidate that
   died while another won was invisible: the total-failure path warns, the
   "some candidate won anyway" path logged nothing, which is exactly how the
   f16 tensor-core candidate lived its whole life. `fail` sets `live = false`,
   so the warn fires once per failed candidate, never in a loop.
2. `f16_gemm_perf.rs`: the MODEL shape added (fp32 + f16 arms), a stderr
   Warn-level logger installed by the tests (a test binary otherwise has no
   logger, and the vendor warn above would go nowhere), `log` as a
   dev-dependency. Existing two arms unchanged in behavior.
