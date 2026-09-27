# 2026-09-27 — The scale ladder this box can actually afford

The owner wants 1B+ parameters trained in a low amount of time on this
machine. This file is the arithmetic, from MEASURED throughput, so the plan is
a plan and not a wish. Every number below is either measured on this box
(2026-09-27) or taken from a cited source; the derivations are shown so they can
be checked.

## What we measure today

| quantity | value | how |
|----------|-------|-----|
| step time, `small` (7.5M), batch 10 x seq 512 | ~1.6-2.0 s | `--timers`, official_v5 |
| throughput | ~2.7-3.2k tokens/s | 5120 tokens / step |
| share of the step that is the KDA **backward** | **~80%** | `--no-kda` ablation: 956 -> 188 ms at batch 4 / seq 256 |
| arithmetic share of a step | 2-5% | `gemm_probe`: fp32 GEMMs 3.5-7.6 TFLOP/s, ~40 ms of GEMM per 1900 ms step |
| fixed cost per step | ~465 ms | 4x tokens at fixed launch count costs only 1.47x time |
| the same GPU with cuBLAS f16 GEMM (5120x2048x8192) | 43.7 TFLOP/s | `research/cublas_probe/`, 3.3-4x cuBLAS fp32, 4.1e-4 max rel err |

Reference point for the ceiling: LLMQ (arXiv 2512.15306) benchmarks this exact
card - 0.5B at 13.0k tok/s bf16 (85% MFU), 1.5B at 3.9k (78%), 7B at 0.9k (79%),
and reports 7B pretraining on one 16 GB card with optimizer offload.

## The ladder, with wall-clock cost

Tokens needed = 20x parameters (Chinchilla, 2022). At our measured cost per
token, and assuming the fixed 465 ms/step stays fixed while the arithmetic grows
with N (measured: the optimizer, the retraction and the EMA do NOT grow with N -
84 -> 85 ms from 7.5M to 12.2M - they are launch-bound):

| params | Chinchilla tokens | est. step | est. wall clock | affordable here? |
|--------|-------------------|-----------|-----------------|------------------|
| **7.5M (now)** | 150M | 1.6 s | **9 h** | yes, today |
| 20M | 400M | 1.7 s | 2.3 days | yes, this week |
| 30M | 600M | 1.8 s | 3.4 days | yes |
| 124M (nanochat d12) | 2.5B | 2.6 s | 36 days | NO |
| 1B | 20B | 16 s (fp32) / 2.5 s (f16) | 30-90 days | only with every fix |
| 1.5B (LLMQ's number) | 10B (their run) | - | ~30 days at their 3.9k tok/s | needs bf16 + offload |

So the honest answer to "1B in low time on my hardware": **1B is a 1-3 month
continuous-training project, and it is gated on two things we do not have yet** -
a working tensor-core GEMM (cuBLAS f16, 3.3-4x, blocked on a device-pointer
handshake with cubecl) and optimizer-state offload (LLMQ-verified on this card;
we already ship the host-offload machinery for the n-gram tables, so the
mechanism exists). With both, 1B at 2.5 s/step x 4M steps (20B tokens) is ~11
days; without them it is 3 months and the arithmetic is 6x slower.

The regime where this box is genuinely competitive is **small N trained long**:
7.5M for 9 hours, 20-30M for a few days. That is also the regime where every
architectural question is still answerable, because held-out BPB moves visibly
at that scale. Chasing 1B before the 7.5M model beats a 24-line 5-gram counter
(2.572 BPB on our fixed held-out window) is optimizing the wrong term.

## What multiplies the whole table

1. **KDA backward (80% of a step now).** If the agent's work lands and the
   backward drops to a sane share, the 7.5M 150M-token run goes from 9 h to
   ~3.5 h and every A/B arm gets cheaper by the same factor. This is the
   single highest-leverage optimization we have, and it is not about precision -
   it is about one op's backward.
2. **CUDA graphs** for the ~465 ms fixed cost (capture/replay exists in burn
   0.22.0-pre.4: `Backend::graph_prepare/start_capture/stop_capture/replay`).
   Prerequisite: in-place parameter updates in the optimizer, because burn's
   optimizers are functional (they return new tensors every step, so a captured
   graph would write into freed buffers).
3. **cuBLAS f16 GEMM** - irrelevant at 7.5M (2-5% of the step), decisive at 1B.
4. **Per-step token count.** At 5120 tokens/step our batch is small; the byte
   research's "fixed working set, 2-4 epochs" recipe also implies we could
   afford a larger batch per step, which amortizes the fixed cost. Worth one
   measured A/B: same tokens/s, half the steps, double the batch.

## The milestone ladder (what "done" means at each rung)

1. **7.5M beats the 5-gram counter** (< 2.572 BPB on the fixed 100 KB
   held-out window). Today: 6.49 and falling. Until this, no capability claim
   about the model is meaningful.
2. **7.5M beats the unigram counter** (< 5.170 BPB) - the "it is learning
   language" line. Extrapolating the current curve: a few thousand steps.
3. **20-30M model, same recipe, beats the 7.5M curve at equal tokens** - the
   scaling check that says the architecture is not the bottleneck.
4. **Domain arms**: genomics (< 2.01 BPB, its 5-gram bar), code (its own bar to
   be computed with the `anchors` tool), physics/math by mixture weight.
5. **Post-training**: SFT floor -> execution-grounded RLVR -> distillation
   (see the post-training verdict in PLAN.md; nanochat is the template).
6. **1B**: only after 1-5, with tensor cores and offload in place.
