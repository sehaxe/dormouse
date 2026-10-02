# Optimization map — ranked under-optimizations, 2026-10-02

Lane: **optim-map** (read-only audit). Head audited: `354a738`. Owner-quoted
facts restated, not re-derived: warm step **485.8 ms/step** (gbench2_control,
2026-10-02, batch 8 s512 small 9 197 454 params aux on quant fp8 — second
identical run in the same log read **457.3 ms**: run-to-run spread 6%, that IS
the instrument noise floor); GPU idle **87%** (mean util 13.3%); launch-bound
at **~21.4–22.2 K launches/step**.

**The model this map is priced at.** The atlas' own host-enqueue constant is
**24 µs/launch** (`crates/dormouse-train/src/atlas.rs:23`). At the measured
warm launch count 21 427/step (`atlas_base.log`, steps 20+) that predicts
**514 ms ≈ the measured step**. So the whole warm step is the launch tax, and
**ms saved ≈ launches removed × 24 µs** — every "expected ms" below is that
arithmetic from a counted launch population, not a guess. Two instrument
caveats, both quoted, neither hidden: (a) per-stage launch *deltas* are exact
in steady state but per-stage *ms* smear across boundaries (atlas.rs:16-18);
(b) the ~22 248/step and 21 427/step figures come from two runs one
system-shape apart. Nothing in this map rests on a stage-ms reading alone.

**Measurement source discipline:** launch populations = ATLAS rows with step
index (`~/logs/atlas_base.log`, warm window) + eval-line counters at current
HEAD (`~/logs/ab2000_*.log`, commit af9def1) + code-path op counts marked as
such. No new GPU runs were taken: a train run (`pgrep -x train` PID 2149814,
2.3 GB) was live for the whole audit, and §1.5 forbids two GPU processes.
**Therefore: nothing on this map was re-measured; every row cites the log or
the counter it stands on.**

Per-step warm launch populations (atlas_base.log, 30-step run, warm window):

| stage | launches/step | ms (stage timer, smeared) |
|---|---|---|
| fwd (window = fwd + loss + mask + bwd + sanitize, `lib.rs:1389-1451`) | 9 002 | 237.5 → ~390 in gbench2 |
| bwd marker (after the window: loss read + host-adam reads) | 7 026 | ~24 |
| opt | 3 921 | 54–70 |
| retr (per-factor NS, default) | 1 311 | 25–27 |
| ema | 165 | 2.4–2.9 |

The differential attribution in hand: **KDA 65.5% of launches · JEPA teacher
22.3% · TSCT rest**. Note the *fwd stage carries the whole window*, i.e.
fwd + bwd are one timer (`lib.rs:1454` stamps `fwd_ms` after `seam.step` which
contains `loss.backward()` at `lib.rs:1448`); the separate `bwd` marker measures
the log-step reads. The stage labels in `benches/history.tsv` rows from
2026-09-29 ("bwd 221 ms of 485") refer to the same window.

---

## Ranked top-10 (expected ms × cheapness)

| # | lever | file:line | measured share / evidence | expected ms/step | LOC | gate |
|---|---|---|---|---|---|---|
| 1 | **KDA fused adjoint: fix numerics, wire it at the trainer's strategy** | `vendor/…/burn-gdn2/src/autodiff.rs:511-722`, `src/kernel/chunk_adjoint_cube.rs` | KDA = 65.5% ≈ 14 572 launches/step (`docs/reviews/kda-gradflow-2026-09-30.md` + ab2000 counters `ops=4096 node_bwd=0`) | **~250–300** | big (adjoint rewrite + numeric gate) | `burn-kda/tests/kda_param_grads_cuda.rs` must go GREEN on the fused arm (FD + CPU-ref), `custom_node_bwd > 0` |
| 2 | **Batched retraction — already measured, still default OFF** | `cfg.rs:331`, `train/src/lib.rs:192` (`retract_batched: false`), arm at `core/src/model.rs:641-696` | committed row `feaf0da`: 265 → **221 ms/step** (−44 ms, −16.6%) | **−44** | ~1 line | greedy: `probe::RETRACT_BATCHED` count > 0 + the 3-seed A/B the flag is designed for |
| 3 | **JEPA teacher forward — the second full forward every step** | `core/src/model.rs:319-337` (`forward_with_hidden` runs `teacher.forward_latent`); shipped escape `--jepa-precompute`/`--jepa-targets` (`train/src/jepa_targets.rs`) | 22.3% launches ≈ 4 960/step ⇒ ~119 ms at 24 µs; BUT differential `arm-A0-pure-CE` read the same ~480 ms warm on coarse timers (`0e9817b` row) — the ms response is NOT confirmed | **~20–119 (unresolved bound)** | 0 to use precompute (objective changes: EMA teacher frozen vs advancing) | `jepa=` counters, then a launch-delta-matched differential |
| 4 | **Discarded fused forward at HEAD — half the KDA calls run fused kernels AND the tensor ops path** | `burn-gdn2/src/cuda_dispatch.rs:~449` + the trainer's Backend strategy naming; fix already coded in `wt/dispatch-guard` (`feece22`, unmerged) | `~/logs/ab2000_attnres_s1.log:130`: `fused kda=2012/0 asked=4096 … ops=4096 node_bwd=0` — on main, ~1 wasted fused-KDA fwd per step | additional ~20–40 (inside #1's ceiling) | tiny | same counters: `fused_fwd == 0 && ops == asked` (no waste) or `node_bwd>0` (fused wins) |
| 5 | **Head-wise Q/K NS batching** | `wt/maxopt2` `c442060`, `train/src/optim.rs` (+109) | worktree commit: "12 separate matmul chains per step, the opt stage's largest launch block"; opt = 3 921 launches/step | ~15 | 0 — merge the worktree | `headwise_batched_ns_matches_per_head` green |
| 6 | **Muon ns 8 → 5** | optimizer line `lib.rs` printout (`muon=15 … ns=8`), burn-muon-plus | no direct ms log; launch arithmetic: 15 groups × 3 fewer NS iters | ~10–15 (est., launch-counted) | config + A/B | quality A/B mandatory (ns=8 is the report's stability choice, §3.5) |
| 7 | **KDA allocator traffic** — 17 fresh tensors / 248 MB scratch per iteration | `burn-gdn2/src/forward.rs:195+` (batched fold), `loop_block.rs:842-855` | §2.2: ~1 GB/step allocator churn; cubecl pool is high-water, `memory_cleanup` every 500 steps | warm ms ≈ 0; buys VRAM headroom & stall-free 500-step boundaries | medium | `--memlog` pool_stats before/after |
| 8 | **Fused RMSNorm in training** | `burn-rmsnorm/src/fused.rs`; `wt/rmsnorm-kernel` (324d2c5 lowering fix) + `wt/dispatch-guard` (`ec438ec`) ready | `norm=0/4619` asks over 500 steps = ~9.2 asks/step, zero runs | ~1 (small — 3 elementwise launches → 1 kernel per norm) | small | `norm=ran/asked` healthy, `==` is the regression signal (`8182e69`) |
| 9 | **`mul_scalar(1.0)` fold copies in batched chunk fwd** | `forward.rs ~225` (`fold` materializes contiguity via mul + cat per tensor) | code-path: 6 tensors × 2 copies per KDA call × 2 calls + teacher ≈ ~24 redundant launches/step | ~0.6 | small | value-preservation is already the batched arm's own test |
| 10 | **Step-0 autotune warmup** — 17.7 s (gbench2 step 0), fwd 9.9 s / opt 6.7 s | cubecl autotune cache; `CUBECL_AUTOTUNE_LEVEL` | measured 23× at 2026-09-29 (§3.1) | 0 on warm steps; only matters for short A/B runs and throughput denominators | instrument hygiene only | step-index discipline (§3.1 rule) |

**Ceiling if 1–5 all land:** ≈ **−330 to −440 ms/step** off 485.8 ⇒ warm step
plausibly **~100–150 ms/step** while it stays launch-bound; competing route is
the CUDA-graph lane (its `wt/graphstage` v1 covers ~85% of launches, but its
own measured 500-step A/B is **1299.3 vs 460.6 ms/step — 3× SLOWER**
(`~/logs/gs500_stage.log` vs `gs500_plain.log`) — pool-retention/recapture
problem, theirs to fix, excluded from this map as instructed).

---

## Layer 1 — vendor/dormouse-fused, crate by crate

- **burn-gdn2 / burn-kda** (the 65.5%): rows #1, #4, #9 above. Shape of the
  defect chain as measured at HEAD: `asked=4096, fused_fwd=2012, declined=10276,
  ops=4096, node_bwd=0` per 2000 train steps at depth 2 — the fused arm
  half-runs and never carries a gradient; the ops path carries it through burn's
  tape whose backward multiplies launches (7026 bwd-stage launches/step).
  The adjoint's measured wrongness is per-input only (post-recurrence groups
  agree at 1e-6; gate inputs off 2–25%, decay amax moves 7–27% between identical
  runs) — `docs/reviews/kda-gradflow-2026-09-30.md` tables.
- **burn-spectral**: row #2. The batched retract (`retract_batched`,
  `lib.rs:365`) computes identical numbers, is COUNTED, prefix-gate-checked in
  the flag plumbing — and the measured 44 ms win sits unused in history.tsv.
  NS per factor is 1 311 launches/step at retract_every=1; not amortized by
  batch (§3.1).
- **burn-muon-plus**: rows #5–6. The opt stage is 3 921 launches ≈ 94 ms at
  the atlas constant (real timer 54–70 ms); the muon share is NS chains per
  group. Note the crate's own doc: `ns_combine_cuda` has NO production caller
  (its guard was deleted as unsatisfiable) — the fused-NS arm exists as a
  tested kernel with no route; routing it is the same class of edit as row #2
  but for opt. Not separately ranked; it rides under #5/#6's A/B if attempted.
- **burn-rmsnorm**: row #8. Never produced a number on ANY device before
  `34c5631`; the lane's gate lives at `norm=ran/asked`. Its ms value is small;
  its lane value is killing a SILENT-fallback site (ADR-0019).
- **burn-jepa**: row #3 — the teacher is not a harness artifact, it is a full
  second model forward on every training step, 22.3% of launches; 92% of the
  JEPA cost IS that second forward (atlas). Cheapest shipped escape is
  `--jepa-train-precompute`, which changes the objective (teacher frozen at
  step-0 weights) — needs its own A/B, not a silent swap.
- **burn-attnres / future-byte (burn-engram? fb arm)**: NOT levers — both arms
  went TIE → DELETED per §1.2 in wave 3 (`history.tsv`, `af9def1`), and the 3-seed
  spread of attnres (0.262) exceeded the control's (0.0730): an optimization move
  here is a *deletion*, already done.
- **burn-byteflow**: quality arm behind `--byteflow`, offline default
  (`d30c9f5`); off in the production recipe, not on the ms map.

## Layer 2 — dormouse-train hot loop

- **Already clean (verify-only, do not touch):** loss/aux clones on log cadence
  only (`lib.rs:1412-1427`); NaN firewall masks + sanitizes on device, sync
  rides the grad-norm read at log cadence (`lib.rs:1428-1500`);
  prefetch build-ahead (`lib.rs:1287-1297`); data stage = 0.1 ms every run.
- **Remaining sync points are cadence-declared:** loss scalar read (log),
  host-table grads D2H (host-adam every step on engram-ram runs — the flagged
  trade `--host-adam-every`), `max_ortho` + `memory_cleanup` every 500 steps.
  None touches the 485 ms recipe (no host tables).
- **`--timers` is not on the hot path** (cadence-gated). The one artifact left:
  stage-ms smear (instrument note (a)) — per-stage ms conclusions must ride on
  launch deltas.
- **Allocator**: row #7 — the highest-value VRAM item, 0 ms on warm steps.
- **Checkpoint/EMA**: ema 165 launches / 2.4–2.9 ms — below the map's floor.

## Layer 3 — atlas: where the top-100 launches live

Top-100 *by count* are all inside the KDA tensor-ops path (65.5% ≈ 14.5 K
launches, dominated by the batched chunk forward's ~30-op body ×2 calls, its
burn-tape backward at ~3.5 K launches/KDA-pass, and the projection stack's
silu/sigmas/exp/log/repeat chains in `project`). Of these, **batchable without
a graph**: the projection elementwise chains (one fused elementwise-lambda
kernel per chain, row #9-class edits), and the fold copies. **Not batchable
without a graph**: anything reading the running state recurrence (the fold's
irreducible per-chunk state loop).

## What is NEW in this map vs the known three (bwd / retr / jepa)

1. **Row #2 is measured money left on the table**: the batched retraction's
   −44 ms/step (265→221, commit `feaf0da`) is committed in `benches/history.tsv`
   and the arm's default is still `false` — a one-line flip, gated by an
   existing counter. Not in any of the known-lane briefs.
2. **The discarded half-forward at HEAD** (`fused kda=2012/0` in the Oct-2
   ab2000 counters on main): the waste is not "fused never engages" (§3.3
   wording) — it engages on roughly HALF the KDA calls and its result is
   thrown away, while the ops path recomputes the same forward. Quantified on
   main, causal fix already coded unmerged in `wt/dispatch-guard`.
3. **The JEPA ms bound is genuinely unresolved** — 22.3% of launches vs an
   arm-A0 differential that moved nothing on coarse timers. The map prices it
   honestly as a ceiling, and names the missing instrument (a launch-delta
   differential, not a `~480 ms` timer comparison).
4. **The launch-tax identity**: launches × 24 µs ≈ the warm step to within 6%
   (514 vs 485.8) — turning the atlas constant into the pricing rule every row
   above uses.
5. **The known-lane risk flags**: `wt/graphstage`'s current implementation is a
   measured 3× regression (1299.3 ms/step), and `wt/maxopt2`'s +109-line NS
   batching is finished and gated but unmerged — the two cheapest calendar
   actions in the whole space are a lane merge and a flag default.

## Excluded by instruction

GEMM rewrites (cuBLAS ceiling reached, `f16` core banked: night run
`e55a4c2`, 440–445 ms/step at 9 288 tok/s fp16 = the −5.2% already shipped);
CUDA graphs (wt/graphstage, wt/cuda-graph lanes); bf16 simulation (slower
than fp32 on this stack, §2.1).
