# 2026-09-27 — Optimization: what actually costs a step, and what 1B on this box needs

All numbers measured on THIS box (RTX 5060 Ti 16 GB, 64 GB RAM, burn 0.22.0-pre.4 +
vendored cubecl), not estimated. Harnesses: `crates/dormouse-core/examples/gemm_probe.rs`
(GEMM shapes), the trainer's own `--timers` (per-phase CPU-side).

## The reference point: LLMQ on our exact GPU

LLMQ (arXiv 2512.15306, IST Austria, Dec 2025) is the only paper found that
benchmarks a **single RTX 5060 Ti** — our card. Table 1, single GPU, 500k tokens/step:

| size | BF16 tok/s | BF16 MFU | FP8 tok/s | FP8 MFU |
|------|-----------|----------|-----------|----------|
| 0.5B | 13.0k | 85% | 16.5k | 70% |
| 1.5B | 3.9k | 78% | 5.7k | 67% |
| 3B | 2.0k | 80% | 3.1k | 69% |
| 7B | 0.9k | 79% | 1.4k | 70% |

Their recipe for fitting a model on 16 GB (the part we can copy):
1. **BF16 compute** as the base (fp8 optional, +27…55% only for big models).
2. **Reduced-precision optimizer states**: fp32 AdamW is 8 B/param (12 GB at
   1.5B); bf16 momentum+variance halves it, with **stochastic rounding** for
   unbiasedness. Master weights in bf16 too.
3. **Offload optimizer states to host**, explicitly *double-buffered*: they
   measured **zero-copy as bad on gaming cards** (5060Ti/4090) and good on
   L40S — the opposite of the datacenter intuition.
4. **Activation recomputation**, selective: non-GEMM ops only, up to whole
   blocks; absmax statistics from the forward are kept so recompute needs no
   global reduction.
5. **Chunked lm_head** (vocab-sized logits + the cuDNN deterministic workspace
   are the two tensors that explode with batch) and a fused CE fwd+bwd that
   never materializes a per-token loss tensor. Irrelevant for us: vocab 256.
6. All allocations at startup: "if the program does not run out of memory
   before the first step, it will never run out".

## Our measured reality

`small` preset, batch 10 x seq 512 = 5120 tokens/step, fp32, no-engram:

- **1.8-2.1 s/step steady state** (v4/v5 logs, `--timers`).
- CPU-side phase timers sum to the whole step: fwd ~800 ms + bwd ~1000 ms +
  opt ~80 ms + retract ~87 ms. The CPU never runs ahead of the GPU ⇒ the GPU
  is idle waiting for work ⇒ **launch/dispatch-bound**.

### The decisive experiment (seq 128, same launch count, 4x the tokens)

| arm | tokens/step | step time |
|-----|-------------|-----------|
| batch 10 | 1 280 | 551 ms |
| batch 40 | 5 120 | 808 ms |

Fitting `t = F + c*tokens`: **c = 0.067 ms/token, F = ~465 ms fixed per step.**
(Measured with v5 training concurrently, so contention inflates the
GPU-bound arm — the true fixed share is if anything larger.)

### Where the arithmetic actually sits

`gemm_probe` on the shapes the preset multiplies (20 iters, release, real GPU):

| shape | f32 ms | TFLOP/s |
|-------|--------|---------|
| 5120x768x768 (attn) | 79.1* | 0.1* |
| 5120x768x2048 (ffn up) | 3.05 | 5.3 |
| 5120x768x256 (lm_head) | 0.54 | 3.8 |
| 5120x768x2304 (kda qkv) | 2.37 | 7.6 |
| 5120x2048x8192 (1B ffn) | 49.1 | 3.5 |

\* first shape measured ⇒ includes the LLVM JIT/autotune compile; ignore it.

Mature fp32 GEMMs run at **3.5-7.6 TFLOP/s**. One step contains roughly 40 ms
of GEMM against a 1900 ms step. Even counting the 4 loop iterations and the
aux heads, the arithmetic is **~2-5% of the step**. The rest is elementwise
traffic over [10,512,768] activations (15.7 MB each), op dispatch, the
optimizer, TSCT retraction, and pool/launch overhead.

## Consequences, ranked by what they are worth

1. **At our scale the lever is per-step overhead, not precision.** bf16
   tensor cores cannot give LLMQ's 4-8x here: their runs are 50-85%
   GEMM-bound, ours is ~4%. The levers are CUDA-graph capture of the step,
   op fusion (fewer elementwise passes), and killing fixed costs.
2. **Cheap fixed-cost cuts available today (no new code):**
   - `--retract-every 5` instead of 1: the polar retraction exists to keep
     the *quantized* factor forward faithful; under `--quant fp32` it is
     87 ms/step of pure overhead. A/B on held-out BPB.
   - `--bf16` storage halves elementwise traffic (the per-token term). Flag
     only; the AGENTS.md note that it was slower predates the current
     autotune/fused state and must be re-measured.
3. **At 1B the ranking inverts.** Per-step FLOPs scale with N: 4 iterations x
   6 x 1e9 x 5120 tokens ≈ 123 TFLOP/step. At our 7.6 TFLOP/s that is 16 s/step;
   at a bf16-tensor-core 50 TFLOP/s it is 2.5 s/step. So for 1B, GEMM
   efficiency IS the lever, and it is currently **blocked**.
4. **The bf16 GEMM blocker is now localized, not vague.** `bf16_matmul`
   (burn-spectral, the true tensor-core path) **fails its own tests on
   pre.4 + cuda**: both `bf16_matmul_matches_fp32_with_grads` and
   `bf16_matmul_ffn_sizes_finite` panic in
   `burn-cubecl-0.22.0-pre.4/src/ops/tensor.rs:150`, and the probe's bf16
   variant dies in `cubecl-llvm .../constant.rs:72` (`float_attr(...).unwrap()`
   on a non-f32 constant). The llvm backend cannot lower bf16 mma by design;
   the cpp/NVRTC path was already measured broken (150 s/step canary). The
   clean bypass: a **cuBLAS bf16 GEMM primitive** (cubecl-cuda already links
   cuBLAS; `cublasGemmEx` with bf16 in / fp32 out sidesteps the dialect
   entirely and needs no JIT). This is the one piece of real systems work
   that unlocks the 1B regime.
5. **Memory at 1B is an offload problem, not an arithmetic one.** fp32 AdamW
   states are 8 B/param; bf16 m/v halves that; host offload with double
   buffering removes it. We already ship the host-offload machinery
   (`offload.rs`, CPU Adam, D2H row grads) for the Engram tables — extending
   it to the Adam states of the big matrices is the enabler, and LLMQ's
   "zero-copy is bad on gaming cards" is the design note to respect.
6. **All allocations at startup** (LLMQ) is the antidote to our cubecl pool
   high-water behaviour; `memory_cleanup()` mid-run is a workaround for a
   design the reference never needed.

## What this means for the 1B goal, honestly

LLMQ's own numbers put a 1.5B at 3.9k tok/s (bf16) on this card. A
Chinchilla-minimum 1B (10B tokens) would then be **~30 days of continuous
training**, and that is the *optimized-stack* figure, not ours. A 7.5M model
trained on 100k steps x 5120 tokens = 512M tokens is the regime where our
current stack is actually competitive, because the per-step overhead is a
fixed cost that a small model amortizes over few tokens.

So the honest program order: (a) cut fixed overhead now (it compounds into
every future run), (b) unlock bf16 GEMM via cuBLAS before any 1B attempt,
(c) only then scale N, with optimizer offload as the memory enabler.
