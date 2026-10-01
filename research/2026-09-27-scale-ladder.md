# 2026-09-27 — The scale ladder this box can actually afford

> ## ⚠ CORRECTION 2026-09-29 — the step-time column is withdrawn; the ladder's
> ## conclusions are re-derived from it and move.
> The owner wants 1B+ parameters trained in a low amount of time on this
> machine. This file is the arithmetic, from MEASURED throughput, so the plan is
> a plan and not a wish. Every number below is either measured on this box
> (2026-09-27) or taken from a cited source; the derivations are shown so they can
> be checked.
>
> **The 2026-09-27 step-time numbers in "What we measure today" are all
> step-0 readings and are struck.** Re-measured warm, release, quiet card,
> `CUBECL_AUTOTUNE_LEVEL=3`, `--timers`, at a step index past 50
> (`benches/history.tsv`, 2026-09-29):
>
> | quantity | 2026-09-27 (withdrawn) | 2026-09-29 (warm) |
> |---|---|---|
> | step time, batch 8 x seq 512 | ~1.6-2.0 s | **244 ms** (steps 50/100/150: 249/240/250) |
> | step 0, same config | not separated | **5549 ms** — **23x** the warm step |
> | throughput | ~2.7-3.2k tok/s | **16.8k tok/s** at batch 8, 19.8k at batch 32 |
> | share of the step that is the KDA backward | ~80 % | **RETRACTED** — see below |
> | arithmetic share of a step | 2-5 % | **~16 %** (~40 ms of GEMM against a 245 ms step) |
> | fixed cost per step | ~465 ms | **not a measured constant**; the warm step is 244 ms |
>
> **Two rules, both learned the expensive way:**
> 1. **A step-time reading with no step index is not a measurement of a step.**
>    `--timers` used to print on `step % 50 == 0` only and was not tied to
>    `--log-every`, so a short run could only ever see step 0 (**fixed in
>    `b8a47ee`** - the cadence now follows `--log-every`, and a warm step is
>    reached by step 2). A step-0 step is
>    23x a warm one; `opt` alone warms 94x (4053 -> 43 ms) because the cubecl
>    autotune cache is cold for the first steps.
> 2. **The `10809 ms` reading promoted to the rulebook as a headline was a
>    step-0 reading.** Struck in `benches/history.tsv`.
>
> **What is confirmed, and it is the load-bearing assumption of this file:** the
> workload is **launch-bound**. That was inferred here from a step-time fit; it
> is now measured — **13.3 % mean GPU utilisation, 142 of 180 samples at <=5 %**
> over a 150-step warm run at batch 32 (`nvidia-smi` at 2 Hz, 2026-09-29). Too
> many small kernels to fill the SMs. The 465 ms fixed cost is the right
> *order* but larger than the warm step it was supposed to bound, so do not
> quote it as a measured fixed cost.
>
> **What is retracted:** "KDA backward is ~80 % of a step" (`--no-kda` ablation,
> 956 -> 188 ms at batch 4 / seq 256). That ablation was measured while the
> attention arm executed **no backward** (`fused kda=<f>/0`, `8fa5d4c`), so it
> is 80 % of a step that skipped the backward. The warm forward is 46-48 ms,
> i.e. **19 %** of a warm step. The ablation is also the only step-time
> measurement of removing the attention arm and it prices neither the arm as it
> was nor as it now is.

## What we measure today

| quantity | value | how |
|----------|-------|-----|
| step time, `small` (7.5M), batch 10 x seq 512 | ~~~1.6-2.0 s~~ **WITHDRAWN — step-0 reading** | `--timers`, official_v5. Warm at batch 8 is **244 ms**; step 0 is 5549 ms (`benches/history.tsv` 2026-09-29) |
| throughput | ~~~2.7-3.2k tokens/s~~ **superseded: 16.8k tok/s at batch 8, 19.8k at batch 32** | warm, 2026-09-29 |
| share of the step that is the KDA **backward** | ~~**~80%**~~ **RETRACTED** — the arm ran no backward when this was measured | `--no-kda` ablation: 956 -> 188 ms at batch 4 / seq 256. Warm forward is 46-48 ms of a 244 ms step = 19 % |
| arithmetic share of a step | ~16 % (was quoted as 2-5 % on a step-0 denominator) | `gemm_probe`: fp32 GEMMs 3.5-7.6 TFLOP/s, ~40 ms of GEMM against a **245 ms** warm step |
| fixed cost per step | ~~~465 ms~~ **not a measured constant** — the warm step is 244 ms | 4x tokens at fixed launch count costs only 1.47x time; warm ladder is 244/440/826 ms at batch 8/16/32 |
| GPU utilisation, warm | **13.3 % mean, 79 % of samples at <=5 %** | `nvidia-smi` at 2 Hz over 150 warm steps at batch 32, 2026-09-29. Launch-bound, **measured** |
| the same GPU with cuBLAS f16 GEMM (5120x2048x8192) | 43.7 TFLOP/s | `research/cublas_probe/`, 3.3-4x cuBLAS fp32, 4.1e-4 max rel err — **unaffected by the step-time retraction** |

Reference point for the ceiling: LLMQ (arXiv 2512.15306) benchmarks this exact
card - 0.5B at 13.0k tok/s bf16 (85% MFU), 1.5B at 3.9k (78%), 7B at 0.9k (79%),
and reports 7B pretraining on one 16 GB card with optimizer offload.

## The ladder, with wall-clock cost

Tokens needed = 20x parameters (Chinchilla, 2022).

**⚠ The `est. step` and `est. wall clock` columns below are WITHDRAWN
(2026-09-29).** They are the output of `t = F + c·tokens` fitted to two
step-0 readings, and the warm measurement shows the model is wrong at both
ends. The table is kept, struck, because the *shape* of the argument (small N
trained long is the affordable regime; 1B is a multi-month project) is what
this file was for, and deleting it would lose the derivation.

| params | Chinchilla tokens | est. step | est. wall clock | affordable here? |
|--------|-------------------|-----------|-----------------|------------------|
| **7.5M (now)** | 150M | ~~1.6 s~~ | ~~9 h~~ | yes, today |
| 20M | 400M | ~~1.7 s~~ | ~~2.3 days~~ | yes, this week |
| 30M | 600M | ~~1.8 s~~ | ~~3.4 days~~ | yes |
| 124M (nanochat d12) | 2.5B | ~~2.6 s~~ | ~~36 days~~ | NO |
| 1B | 20B | **16 s (fp32) / 2.5 s (f16) — SURVIVES, see below** | ~~30-90 days~~ | only with every fix |
| 1.5B (LLMQ's number) | 10B (their run) | — | ~30 days at their 3.9k tok/s | needs bf16 + offload |

**What can be re-derived, and what cannot.**

- **Survives: the 1B step estimate.** 16 s (fp32) / 2.5 s (f16) comes from
  FLOPs and the measured GEMM rate (4 iterations x 6 x 1e9 x 5120 tokens ≈
  123 TFLOP/step at 7.6 TFLOP/s measured, or 50 TFLOP/s on a bf16 tensor core)
  — a roofline, not a step timer, so the step-0 retraction does not touch it.
- **Does not survive: the wall-clock column.** It was `est. step` x
  Chinchilla-steps, and `est. step` is gone. **I am not replacing it with a
  guessed number.** The one clean re-derivation available is at the measured
  operating point: 150M tokens at the warm **16.8k tok/s** (batch 8) is
  **~2.5 h**, against the withdrawn 9 h — i.e. the true figure is *smaller*,
  but it is 4x the tokens per step different from the row's own assumption, so
  treat it as an order-of-magnitude statement, not a replacement.
- **An internal inconsistency worth naming rather than fixing silently:** the
  withdrawn "30-90 days" for 1B does not follow from the 16 s/step in the same
  row. 20B tokens at 5120/step is 3.9M steps, which at 16 s is **~2 years**.
  The 30-90 days must therefore have come from LLMQ's 3.9k tok/s on their own
  optimized stack, not from our fp32 path — two different things under one
  number. Whoever re-derives this column must state which one is meant.
- **The scaling premise is unverified.** "The optimizer, the retraction and the
  EMA do NOT grow with N (84 -> 85 ms from 7.5M to 12.2M)" came from the same
  step-0 reading. Warm, those three are 43-47 / 52-61 / 0.0 ms — a different
  split, and no warm measurement at two model sizes exists, so **the premise
  that the fixed cost is flat in N is currently untested.** Every row above
  depends on it.

So the honest answer to "1B in low time on my hardware" is unchanged in
conclusion and now has a named source: **1B is a multi-month
continuous-training project, and it is gated on two things we do not have yet** —
a working tensor-core GEMM (cuBLAS f16, 43.7 TFLOP/s measured on this card,
blocked on a device-pointer handshake with cubecl) and optimizer-state offload
(LLMQ-verified on this card; we already ship the host-offload machinery for the
n-gram tables, so the mechanism exists). The 1-3 month range is **LLMQ's own
figure for this GPU at 1.5B and 3.9k tok/s**, not ours; our own fp32 path is
several times slower and has never been run at that size.

The regime where this box is genuinely competitive is **small N trained long**:
~~7.5M for 9 hours~~ (**withdrawn with the step-time column; the warm
arithmetic gives ~2.5 h for 150M tokens at 16.8k tok/s, batch 8**), 20-30M for
a few days. That is also the regime where every architectural question is still
answerable, because held-out BPB moves visibly at that scale. Chasing 1B before
the 7.5M model beats a 24-line 5-gram counter (2.572 BPB — and note that bar was
**not measured on a trainer eval window**; see `docs/protocols/AB-PROTOCOL.md` and
`anchors.rs:22-35`) is optimizing the wrong term.

## What multiplies the whole table

1. ~~**KDA backward (80% of a step now).** If the agent's work lands and the
   backward drops to a sane share, the 7.5M 150M-token run goes from 9 h to
   ~3.5 h and every A/B arm gets cheaper by the same factor. This is the
   single highest-leverage optimization we have, and it is not about precision -
   it is about one op's backward.~~
   **RETRACTED 2026-09-29, twice over.** The 80 % was measured while the
   attention arm ran **no backward** (`fused kda=<f>/0`, `8fa5d4c`) — it is
   80 % of a step that skipped the backward — and the warm forward is 46-48 ms
   of a 244 ms step, i.e. **19 %**, not 80 %. The 9 h -> 3.5 h saving is
   therefore withdrawn without replacement: the attention backward's true cost
   **has never been measured**, because no run on record has executed one.
2. **CUDA graphs** for the ~~465 ms fixed cost~~ *(withdrawn as a measured
   constant; the warm step is 244 ms)* (capture/replay exists in burn
   0.22.0-pre.4: `Backend::graph_prepare/start_capture/stop_capture/replay`).
   Prerequisite: in-place parameter updates in the optimizer, because burn's
   optimizers are functional (they return new tensors every step, so a captured
   graph would write into freed buffers).
   **STRENGTHENED 2026-09-29, and this is the one item on the list that got
   better.** The 465 ms is withdrawn as a measured constant, but the *reason*
   CUDA graphs are the lever is now an instrument rather than a fit: the GPU is
   **13.3 % utilised, 79 % of samples at <=5 %**. The step is launch-bound, so
   removing launches is the only large win available at this scale. Capture and
   replay is **confirmed working on this GPU** (`vendor/cubecl-fix/cubecl-cuda/
   tests/graph.rs`, 5/5 green) and **not yet wired**; the documented blocker is
   that a capture window refuses stream reads, syncs and handle writes
   (`client.rs:1288-1296`), and a training step must do all three — so the first
   version captures ONE stage that needs no host round-trip, not the whole step.
3. **cuBLAS f16 GEMM** - irrelevant at 7.5M (2-5% of the step), decisive at 1B.
   **The 2-5 % is a step-0 denominator; warm it is ~16 %** (~40 ms of GEMM
   against a 245 ms step). Still the smallest item at 7.5M, still decisive at
   1B, and still blocked on the same pointer handshake.
4. **Per-step token count.** At 5120 tokens/step our batch is small; the byte
   research's "fixed working set, 2-4 epochs" recipe also implies we could
   afford a larger batch per step, which amortizes the fixed cost. Worth one
   measured A/B: same tokens/s, half the steps, double the batch.
   **MEASURED 2026-09-29, and the answer is "a bigger batch buys gradient
   quality, not speed":** 244 / 440 / 826 ms at batch 8 / 16 / 32 is
   16.8k / 18.6k / 19.8k tok/s — **+18 % throughput for 4x the memory, and
   diminishing** (+11 %, then +6.5 %). This also corrects the older
   "throughput is flat from batch 8 to 16" claim: it is **+11 %**. Batch 64 OOMs
   at `server.rs:144`'s 100 MB reserve.

## The milestone ladder (what "done" means at each rung)

1. **7.5M beats the 5-gram counter** (< 2.572 BPB on the fixed 100 KB
   held-out window). Today: 6.49 and falling. Until this, no capability claim
   about the model is meaningful.
   **Two corrections, both about the bar, neither about the model:** the
   "fixed 100 KB window" is not a constant — the window is
   `eval_batches × batch × seq_len`, i.e. **102 400 B at batch 10 and 20 480 B
   at batch 2**, and is printed on every eval line; and 2.572 is one of four
   5-gram readings in circulation, **none of them on a trainer eval window**.
   As of 2026-09-28 the best valid held-out number is **4.997** — above every
   5-gram reading by 2.1-2.4 BPB, so the rung is not close.
2. **7.5M beats the unigram counter** (< 5.170 BPB) - the "it is learning
   language" line. Extrapolating the current curve: a few thousand steps.
3. **20-30M model, same recipe, beats the 7.5M curve at equal tokens** - the
   scaling check that says the architecture is not the bottleneck.
4. **Domain arms**: genomics (< 2.01 BPB, its 5-gram bar), code (its own bar to
   be computed with the `anchors` tool), physics/math by mixture weight.
5. **Post-training**: SFT floor -> execution-grounded RLVR -> distillation
   (see the post-training verdict in PLAN.md; nanochat is the template).
6. **1B**: only after 1-5, with tensor cores and offload in place.
