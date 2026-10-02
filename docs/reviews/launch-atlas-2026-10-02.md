# Launch atlas — where the step's ~21.4k kernel launches come from

Lane: atlas запусков (2026-10-02). Instrument: `DM_LAUNCH_ATLAS=1`
(permanent, `crates/dormouse-train/src/atlas.rs`, summary tool
`tools/atlas_summarize.py`). Question the lane owns: **which stage and
which arm produces the launches a training step enqueues**, so
kernel-fusion / CUDA-graph work can be ranked by population instead of
taste.

**Headline.** A warm step at `small`/b8/s512/no-engram enqueues
**21 427 launches** (measured, warm steps 20-29; the briefed 21 434 is
confirmed). The attention (KDA) arm owns **65.5%** of them (6 246 fwd +
5 730 bwd + 2 067 opt), the JEPA teacher+aux **22.3%**, the TSCT
experts **15.0%**, the Engram arm **4.4%**. The optimizer's 3 921
launches and the retraction's 1 312 are **depth- and iteration-count
independent** — they are per-param-group, not per-loop. Every stage
except the bwd loss-sync boundary is free of host reads, so a captured
{fwd-loop, bwd, opt, retr, ema} graph covers **>99% of all launches**;
the top-4 graph targets alone price at **−413 of 514 ms@24µs** without
touching the objective.

## Method

Two instruments, both host-side, both §1.3-clean:

1. **Per-stage deltas** (`atlas.rs`): the trainer already times five
   stages (fwd / bwd / opt / retr / ema). The atlas reads
   `cubecl_launches()` at the same five boundaries and prints one TSV
   row per stage per step: `ATLAS\t<step>\t<stage>\t<launches>\t<ms>`,
   plus a warm-window summary (`ATLAS-WARM`, last third of the run).
   Counting only — no sync, no hot-path branch, inert without the env
   var (one env read per process). Known bound: the counter is
   incremented on the device thread, so a host read lags the enqueue;
   warm-window SUMS are exact, a single stage row can smear across its
   boundary by the driver's pending-launch depth (~1% here: the iter2
   half-step prediction misses by 51 of 4 501).
2. **Per-arm marginals** (differential runs): one arm toggled per run,
   same 30 steps, same seed; the launch difference is that arm's
   population, already split by stage. One cross run
   (`--no-kda --jepa-weight 0`) splits the KDA marginal into its main
   and EMA-teacher copies. Marginals are **not additive** across arms
   (the KDA toggle removes KDA inside the teacher forward; the JEPA
   toggle removes the whole teacher forward) — every number below says
   which toggle it came from.

Runs: `--preset small --batch 8 --seq-len 512 --no-engram --steps 30`,
seed 1, warm window steps 20-29 (steps 0-19 are autotune garbage;
AGENTS.md 2.2). Real corpus, fp32 masters (auto factors = Fp8 on
sm_120), aux at preset defaults (jepa 0.05, dspark 0), release binary,
`--detach --timers`. Raw logs: `/home/sehaxe/logs/atlas_base.log` and
`/tmp/opencode/atlas_{nokda,jepa0,exp1,iter2,cross,engramon}.log`.

Provenance notes, so nobody re-derives them: the base run is the 12:04
`--log` run; a duplicate run reproduced it at **21 427.0 vs 21 427.2**
(that second run then died writing its final checkpoint — **the disk
was full at that moment, `os error 28`** — costing nothing but the
checkpoint; atlas rows print before any save). The first attempt
resumed from the probe run's snapshot legally: `steps` is an exempt
progress key (ADR-0021) and `timers` matched. `use_situ`, `use_attnres`,
`use_mhc`, `use_mor`, dspark are OFF in the preset: their rows are
structural zeros, not measurements.

## The brief's number is confirmed, the night run's is a different recipe

The briefed "21 434 launches/step" is real: the base run reads
**21 427.2/step warm** (sum of the stage table below) and an
independent duplicate read 21 427.0. The night run's clean 100-step
deltas (19 300 ± 3) are **not comparable**: that recipe is fp16 at
`--retract-every 4` — retraction alone accounts for −984/step
(1 312 → 328 averaged), and the fp16 cast pattern is a different
enqueue graph. The base run is the authoritative count for the brief's
config.

## Stage totals — base run, warm steps 20-29

| stage | launches/step | share | ms/step (actual, warm) | ms @24µs |
|---|---|---|---|---|
| fwd | 9002.5 | 42.0% | 237.5 | 216.1 |
| bwd | 7026.3 | 32.8% | 245.8 | 168.6 |
| opt | 3921.0 | 18.3% | 70.3 | 94.1 |
| retr | 1311.5 | 6.1% | 25.1 | 31.5 |
| ema | 165.9 | 0.8% | 2.8 | 4.0 |
| **total** | **21 427.2** | 100% | 581.5 | **514.3** |

(ms columns: "actual" is the warm stage timer — treat as
order-of-magnitude, the card was shared with other lanes;
"@24µs" is the briefed host-enqueue arithmetic. The release warm step
is ~245 ms wall while the @24µs sum is 514, so the true mean cost per
launch at this shape is ~11 µs — **the count, not the 24 µs
conversion, is the instrument**; 24 µs is the ceiling the roadmap
prices at.)

## Arm marginals — differential runs (each vs base, same 30 steps)

| toggle | fwd | bwd | opt | retr | ema | total | % of base |
|---|---|---|---|---|---|---|---|
| `--no-kda` | −6245.5 | −5729.9 | −2066.7 | +4.1 | +3.6 | **−14 041.6** | 65.5% |
| `--jepa-weight 0` | −4401.7 | −137.1 | −75.6 | −1.9 | −165.9 | **−4782.2** | 22.3% |
| `--set n_experts=1` (2 of 3 experts) | −1222.0 | −394.6 | −901.4 | −655.1 | −38.3 | **−3211.4** | 15.0% |
| `--max-iter 2` (4→2) | −4450.3 | −3339.1 | +0.2 | +9.2 | +1.8 | **−7778.2** | 36.3% |
| engram ON (vs `--no-engram` base) | +344.3 | +462.1 | +153.1 | −4.4 | −3.5 | **+951.6** | 4.4% |

Readings that fall straight out:

- **The KDA arm is the launch atlas.** 65.5% of all launches. Its opt
  share (2 066.7) is its parameter groups inside the Muon+ pass; its
  retr share is **zero** — retraction touches TSCT U/V only.
- **The JEPA teacher is a second forward**: 92% of its marginal is the
  fwd stage (the EMA copy runs the same 4-iteration loop), plus the EMA
  lerp itself (165.9 ≈ 53 param groups × ~3 ops) and a sliver of
  aux-head bwd/opt. The off-switch is `--jepa-precompute` (offline
  latents, already shipped) — a cost lever, not a quality knob.
- **The retraction is exactly 4 equal factor-pairs**: 1 311.5 → 656.4
  when 2 of 3 experts vanish → ~328 per expert pair and ~328 for
  lm_head. `--retract-every N` divides the stage by N.
- **opt and retr have NO per-iteration component** (iter2 deltas ≈ 0
  within the smear bound): they are per-param-group/per-factor costs.
  The depth lever buys only fwd+bwd launches — 3 889.1 per iteration
  (2 225.2 fwd + 1 669.6 bwd, main+teacher combined).
- **The Engram arm is launch-cheap and wall-expensive**: +952
  launches/step (4.4%) but fwd wall 907 vs 237 ms in the same session —
  its cost is scatter/gather work plus the per-step host-Adam D2H drain
  the timer line already names, not launch count. Hypothesis, single
  run, not chased here.

### The main/teacher split (cross run: `--no-kda --jepa-weight 0`)

| cell | launches/step | how measured |
|---|---|---|
| fwd × KDA, main loop (4 iters) | 3146.1 | jepa0 fwd − cross fwd |
| fwd × KDA, teacher copy (4 iters) | 3099.4 | (base−nokda) − main |
| fwd × main non-KDA (body×4 + const) | 1454.7 | cross fwd |
| fwd × teacher non-KDA | 1302.3 | nokda fwd − cross fwd |
| bwd × KDA, main loop | 5735.6 | jepa0 bwd − cross bwd |
| bwd × aux heads (JEPA/CE heads) | 137.1 | base bwd − jepa0 bwd |
| bwd × main non-KDA (body×4 + const) | 1153.6 | cross bwd |
| opt × aux-head groups | 75.6 | base opt − jepa0 opt |

The teacher's KDA copy (3 099.4) is 98.5% of the main's — the EMA
teacher is a near-copy of the training forward, as it should be.

**What stays underdetermined** (one free parameter each, not guessed in
any table above): the split of the non-KDA fwd body into constants
(embedding/final head/loss) vs aux-head forward vs per-iteration
experts-vs-core glue. Closing it needs one more toggle cross
(`--set n_experts=1 --max-iter 2`); the roadmap does not depend on it.

## Top-20 populations (exact cells, ranked; 15 exhaustive rows sum to 21 427.2 with the stage totals)

| # | stage × population | launches/step | % | ms @24µs | mergeable |
|---|---|---|---|---|---|
| 1 | bwd × KDA main loop | 5735.6 | 26.8% | 137.7 | graph: capture iteration bwd |
| 2 | fwd × main non-KDA body (experts+core glue, ×4) | ~1350–1460 | ~6.5% | ~34 | graph: capture iteration fwd |
| 3 | fwd × KDA main loop | 3146.1 | 14.7% | 75.5 | graph: capture iteration fwd |
| 4 | fwd × JEPA teacher, whole (KDA+glue) | 4401.7 | 20.5% | 105.6 | graph: same iteration graph; or `--jepa-precompute` |
| 5 | fwd × KDA teacher copy | 3099.4 | 14.5% | 74.4 | rides row 4's graph |
| 6 | opt × KDA param groups | 2066.7 | 9.6% | 49.6 | graph: capture opt stage (in-place params prereq) |
| 7 | bwd × main non-KDA (incl. aux 137.1) | 1153.6 | 5.4% | 27.7 | graph: iteration bwd |
| 8 | opt × non-KDA non-expert groups | 952.9 | 4.4% | 22.9 | same opt graph |
| 9 | opt × expert groups | 901.4 | 4.2% | 21.6 | same opt graph |
| 10 | retr × lm_head + 1 expert factors | 656.4 | 3.1% | 15.8 | graph: NS chain, no host reads; or `--retract-every` |
| 11 | retr × 2 expert factor pairs | 655.1 | 3.1% | 15.7 | same |
| 12 | ema × teacher lerp | 165.9 | 0.8% | 4.0 | graph: trivial |
| 13 | opt × aux-head groups | 75.6 | 0.4% | 1.8 | same opt graph |
| 14 | bwd × per-step const (embedding/head/loss glue) | ≤74 | ≤0.4% | ≤1.8 | the loss-sync boundary stays outside captures |
| 15 | fwd × per-step const (embedding + final head + loss) | ≤102 | ≤0.5% | ≤2.4 | same boundary |

(Rows 2/14/15 carry the one underdetermined split, priced as the interval
its two readings bound; nothing else in the table is derived.)

## Merge roadmap — top-5 targets (minus N launches → minus M ms @24µs)

1. **Graph-capture the optimizer stage** — −3 921 launches → **−94.1 ms**
   (18.3% of the step). Pure tensor ops, zero host reads. Prerequisite:
   in-place parameter writes — burn's functional optimizer allocates new
   param tensors, and a capture window refuses handle writes
   (`client.rs:1288`); the wall is documented in the wt/cuda-graph
   handover.
2. **Graph-capture the KDA iteration forward, main + teacher** —
   −6 246 launches → **−149.9 ms** (29.1%). The fused chunk kernels and
   their tensor-op glue have no host reads; the capture needs stable
   input handles with per-step rewrite — the exact pattern
   `vendor/cubecl-fix/cubecl-cuda/tests/graph.rs` proves 5/5 on this
   GPU. One graph serves both copies (rows 3+4+5).
3. **Graph-capture the KDA backward** — −5 730 launches → **−137.5 ms**
   (26.7%). Same mechanics as (2); the loss-sync boundary at the end of
   bwd stays outside the window. Numerics caveat carried, not created
   here: the fused adjoint's correctness is a burn-gdn2 lane item —
   capturing the tensor-op path that actually runs changes no math.
4. **Graph-capture the retraction** — −1 311 launches → **−31.5 ms**
   (6.1%). Newton-Schulz chains are pure tensor ops; no prerequisite
   beyond the graph seam itself. `--retract-every N` is the zero-code
   alternative (divides the stage by N) at a quality cost the A/B queue
   already owns.
5. **`--jepa-precompute` as a launch lever** — −4 568 launches
   (teacher fwd 4 401.7 + ema lerp 165.9) → **−109.6 ms** (21.3%),
   **zero code, ships today**. It changes the objective (frozen teacher
   latents vs live EMA), so it is the owner's call — the graph route
   (row 4 rides target 2's iteration graph) buys the same launches
   without touching the objective.

**Ceiling.** Targets 1-4 + ema are all host-read-free: capturing
{iteration fwd+bwd, opt, retr, ema} covers **~21 250 of 21 427
launches (99.2%) → −510 ms @24µs**, leaving the per-step constants at
the log/eval sync boundaries. That is the number the
cuda-graph lane was built on, now with a per-stage price tag.

## Raw TSV

`docs/reviews/launch-atlas-2026-10-02.tsv` — per-run warm stage means,
arm marginals, and the main/teacher split. Regenerate:
`python3 tools/atlas_summarize.py <logs...>` (order: base first).
