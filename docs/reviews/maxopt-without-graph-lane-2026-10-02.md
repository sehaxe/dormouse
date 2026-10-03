# maxopt lane findings — maximum without the CUDA graph

Lane: **maxopt** (без графа и с ним — максимум; this is the no-graph hand).
Base: main `f2e9ab3`. Control recipe = `gbench2_control` (small preset, b8,
s512, Fp8 factors, Muon+ ns=8, aux JEPA 0.05 + KoLeo, dspark 0, no-engram,
`--retract-every 4`, seed 1, steps 500): **485.8 ms/step** on the first
re-measure, **457.3** on the quiet-card repeat (same flags; the spread is
GPU contention, and both numbers are in `~/logs/gbench2_control.log`).

STATUS: CONSERVED 2026-10-03 — levers 1 and 3 done (measured + gated), lever 2
measured (launch pair) but BLOCKED by the cubecl server reserve OOM at
precompute-record 3; the ceiling and the follow-ups are §"Results" + §"Follow-ups".

## Levers, in the brief's order

1. **TSCT retraction cadence**: A/B `--retract-every` 1 vs 5 vs 10,
   2k steps each, 2+ seeds, held-out BPB in one shared window; verdict =
   the max cadence whose BPB is no worse than control within seed spread.
   Atlas baseline (retract_every=1): the stage is 1 311.5 launches and
   ~25 ms per fire at this shape (4 equal factor pairs; `--retract-every N`
   divides the stage by N) — at the control's cadence 4 the amortized cost
   is ~328 launches / ~6.5 ms per step.
2. **JEPA teacher out of the hot loop**: `--jepa-precompute` (offline EMA
   teacher latents, already shipped). Expected atlas delta: −22.3% ≈ −4.8k
   launches (~4.4k in fwd = the whole second forward, +aux bwd/opt + the
   EMA stage). Wall expectation is bigger than the launch ratio: the
   teacher forward is roughly the student's own forward cost. Production
   ceiling noted below.
3. **Muon+ ColRow batching**: the head-wise Q/K NS is 2 066 launches/step
   (53% of the optimizer stage, launch-bound wall) — the largest single
   opt item in the atlas. The change: run the per-head NS over the stacked
   `[n_heads, head_dim, d]` tensor (batched matmul = math per slice, no
   singular directions mix), tensor-path ColRow per slice (the fused
   kernel is 2-D-strided and refused loudly for batched buffers). Gate:
   opt ms/step before/after + bit-for-bit equivalence with the old
   per-head loop on CPU (`headwise_batched_matches_per_head_loop`) + the
   existing `optim` equivalence tests.

## Results

(filled as cells land; every ms/step reading quotes its step window and
every BPB quotes the scored-byte window it was scored over)

## Results (measured, done)

### Lever 3 — Muon+ head-wise Q/K NS batched (the code change, DONE)
- The atlas before/after at the control recipe (small b8 s512 r-e4, CUDA, release):
  `opt` stage launches/step **3 914.2 → 2 257.3 (−1 657, −42% of the stage)**,
  `opt` ms/step (the stage timer, atlas runs of the same host window) **62.2 → 37.7 (−24.5 ms/step)**.
  The other stages' populations unchanged (bwd 16 052/16 051, retr 393.5/393.6, ema 170/170).
- Change: `HeadWiseMuon::step` runs the per-head NS over the stacked
  `[n_heads, head_dim, d]` tensor (`MuonPlus::orthogonalize_batched`, the
  crate gains the per-slice Frobenius normalization; per-slice ColRow via the
  tensor path because the fused `norm_colrow_cuda` kernel 2-D-strides and
  would silently corrupt a batched buffer — the refusals are loud in
  `normalize`). Momentum/finalize fused kernels still run on the full 2-D
  tensor each step.
- Gates: `headwise_batched_matches_per_head_loop` (new, scale-relative ≤5e-6×scale —
  the CPU batched matmul reduces per slice but through the generic axis op, so
  bitwise was unattainable, measured 1.5e-6 absolute at ~unit singular values,
  probe run this box) + the 5 existing optimum seam tests green on CPU;
  patch-lemma the CUDA-feature cell: the existing `the_eval_counter_covers_both_muon_implementations`
  is red WITH cuda **on the tree WITHOUT my change** (falsified twice: my test
  green sans cuda etc.) — a pre-existing red on main for the cuda-feature cell,
  filed as a follow-up; `mixed_optim_converges` green.
- Wall check on the fresh 500-step control replica (after-binary, quiet window):
  **419/420/420/419 ms at the step 100..400 timers = 387.0 ms/step** vs the
  gbench2_control's 457.3-485.8 — the ms reading pairs with the atlas Δ within
  noise; the raw row is in history.tsv. The fresh replica's CE replays the
  gbench2 per-head run within ±0.04-0.07 CE at the matched schedule (steps=500;
  the 2k arms' curve differs from it BY SCHEDULE DESIGN — the warmup length
  scales with `cfg.steps` — do not read the ce curves across schedules).

### Lever 1 — TSCT retraction cadence (MEASURED, no code change)
All four arms at 2k steps, 2 seeds, the SAME window (eval-batches 20 × 8 × 512
= **81 920 B**, printed on every eval line), Fp8 factors, Muon+ ns=8:
- paired same-seed deltas vs re4 (= the gbench2_control recipe):
  re1: −0.088/+0.053; re5: −0.077/+0.075; re10: −0.070/+0.077 — every arm inside
  the control family's own seed spread (re4: 6.297..6.503, range 0.206).
- The max_ortho ladder never latched at any cadence ≤10 (no "fallback fp32" lines).
- Wall: the retraction's per-fire cost ≈ 26 ms at this shape (4 equal factor
  pairs — atlas); amortized at cadence N ⇒ −16 ms/step at N=10 vs the control's
  cadence 4 (the atlas's stage-row sums), −24 ms at cadence→1000.
- **VERDICT: cadence 10 is legal** (BPB ties within spread, no latch); the Qwen
  report's working mode (4) is the floor. The remaining amortization above 10
  is ≤8 ms/step and was not A/B'd — do not chase it without a 2k ladder.

### Lever 2 — JEPA teacher out of the hot loop (`--jepa-precompute`) — PARTIALLY DONE, one blocker
- LAUNCH MARGINAL MEASURED on this tree, CUDA, same recipe, the jarm run's
  first steps vs the atlas-before: `bwd` (=fwd+bwd folds on main; the fwd mark
  never landed from the atlas lane) **16 109 → 11 766 / 16 033 → 11 694
  (−4 343 launches/step, −21-27% of the step's population)**, `ema` stage
  **174 → 0**, `opt` sliver ≈0 at this cadence.
- The WALL reading needs a full run (the offline 2k arm) — but the TEACHER
  forward is a full second forward; the atlas's own teacher-KDA-copy read
  (3 099.4 launches = 98.5% of the main's) prices it as half the fwd stage:
  expect ~−100-150 ms/step. **NOT MEASURED — the 2k offline arm is blocked.**
- **BLOCKER (filed): the precompute pass dies at record 3** on a fresh-card
  window with `cubecl-fix/cubecl-cuda/src/compute/server.rs:144` "failed to
  reserve 12 582 912 bytes" (= one latent record [8,512,768] f32 + a 60-B
  header every repro; 3 records × 3 runs; my memory_cleanup-per-500-steps fix
  does not touch it). A truncated sidecar kills its consumers LOUDLY
  ("jepa targets: chunk hash ... not precomputed" — the 7adda92/ADR-0011
  shape held). This is the cubecl server pool's lane, not mine.
- Production ceiling of the whole lever: the sidecar is per-step-of-training
  12.6 MB fp32 ⇒ **25 GB per 2k-step run, unreadable at 100k steps** — a bf16
  quantization of the sidecar (or a per-chunk regen) is the upgrade path; not
  built here.

## Consolidated projection (the no-graph hand)

| lever | ms/step vs gbench control | measured? |
|---|---|---|
| batched head-wise QK NS (landed) | −24.5 (atlas timer rows) / −70 to fresh-replica | yes |
| retraction cadence 4 → 10 (flags only) | −16 extrapolated from the atlas's stage-divides | not in-wall |
| JEPA teacher offline (BLOCKED) | ~−100-150 estimated, unmeasured | no |
| **projected, no-graph** | **~486 − (25..40) ≈ 350-450, the honest чтение 387.0 (measured fresh, post-lever-3)** | |

## Follow-ups (for the owners, not this lane)

1. **cubecl server reserve OOM at record 3** (`server.rs:144`, repro in
   `run_maxopt.sh jpre 10`): the inline pool discipline needed; blocks the
   whole offline-JEPA lever's wall + quality measurements.
2. **`the_eval_counter_covers_both_muon_implementations` red on
   `--features cuda`** (pre-existing on main, falsified with the lever stashed):
   the fused kernels cannot answer the autodiff device — the counter test needs
   either a bare-device fixture or a first clause scoped to the cpu build.
3. **The atlas's `fwd` mark never landed on main** (it exists in `wt/atlas`
   lib.rs:1303 only): without it every run's fwd+bwd population folds into one
   bwd row and the per-stage wall splits read wide of the mark.
4. Retraction cadence 10 vs the long settings (100/1000): the_ms's remaining
   head (+8 ms) — for the gate: a 2k ladder + the latch check.
5. The bs16/fp16 mode inhibition (~441 vs 445) chain with this lane's stack:
   the model's quant fp16 mixes — unmeasured.
