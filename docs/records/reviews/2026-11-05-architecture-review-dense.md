# 2026-11-05 — architecture review: the dense arm, what is overhead, and what to test next

**Scope.** Read-only audit of `port/` at `38d1583` + the working tree; every
parameter count below was produced by instantiating the resolved config on CPU
(`CUDA_VISIBLE_DEVICES=""`, `torch.set_num_threads(4)`) and counting the live
module tree (`param_count`, `model.py:1193`); every MAC number is the project's
own `scripts/capacity_math.py` ledger. No GPU command was run and no code was
modified.

**Tags.** `[M]` measured on this box, source named. `[D]` derived arithmetic
from measured numbers. `[L]` literature result with an arXiv id. `[U]`
unmeasured — an explicit absence, not a zero.

---

## 0. The finding, re-audited — four numbers do not match the table

The direction of `docs/records/reviews/2026-11-05-dense-beats-bank.md` survives. The
table's magnitudes, and two of its labels, do not.

| claim in the finding | what the run actually contains | evidence |
|---|---|---|
| dense = **81,126,404** params | **66,910,751** | run banner, `~/logs/abdense/dense_s1.log:1-3`; re-instantiated from the driver's flags: 66,910,751 |
| "equal active compute", 28.44M MAC/B | dense = **33,687,040** MAC/B, **+18.44 %** | `capacity_math.active_macs_per_token` on both resolved configs |
| "dense, f=4096, share_u off" is the 600-step winner | the 2k arm is **rank 64, f 4096, topk 1**; the 600-step winner is **rank 1536, f 1603, topk 0** | `/tmp/opencode/ab_dense.sh` vs `ab_capacity.py:ARMS["D"]`; `D_s10/resolved.config.toml` |
| "31 bytes/param" | the run that produced 2.3195 saw **16.38 MB**, i.e. **0.245 B/param**; 36.7 B/param is the planned 300k-step run | 2000 × 8 × 1024 = 16.38 MB |

### 0.1 The 81M is a hand sum, not a model

No log, config or checkpoint in the tree contains 81,126,404. The two nearest
models this repo can instantiate are **77,464,991** (`rank=d`, `f=4096`,
latent-256 attention, `T=3`) and **84,936,095** (`rank=64`, `f=4096`, no
latent). Neither has 28.44M MAC/B: the first is 44,238,336, the second
66,651,648. `[M/D]` The run that produced 2.3195 is 66,910,751 params with
33,687,040 MAC/B. The 81,126,404 figure is a hand sum for a hypothetical
full-attention dense model; no run on record produced it.

### 0.2 The 2000-step pair is not equal-compute; the 600-step sweep is

The 600-step sweep is honest: `D_s10` is 69,327,839 params, 28,442,688 MAC/B
against A's 28,442,624 (+0.0002 %), and the driver refuses an arm outside 5 %
(`ab_capacity.py:COMPUTE_TOL`). `[M]`

The 2000-step follow-up silently moved four axes at once:

| axis | 600-step winner (D) | 2000-step "confirmation" |
|---|---|---|
| expert rank | **1536** (= d, full rank) | **64** |
| `d_ffn` | 1603 (solved for equal MAC) | 4096 |
| `moe_topk` | **0** (dense blend, `moe_dense_banked`) | **1** (routed path, `moe_dropless`) |
| LR | 3e-4 (battery protocol) | 5e-4 |
| MAC/B | 28,442,688 | 33,687,040 |
| params | 69,327,839 | 66,910,751 |
| body without Engram | 16,536,031 | 14,118,943 |

`[M]` — `D_s10/run.log` prints `moe_dense_banked=14745600`; the 2k log prints
`moe_dropless=1500` and no `moe_dense_banked`. The 2k arm pays the full MoE
dispatch (router GEMM, softmax, top-k, gather, scatter) to route to **one of
one** expert. That is not the dense arm; it is a routed arm with one expert.

So the 600→2000 comparison measures a *different* model twice: the 2k arm has
a 64-dimensional FFN up-path where the winner had a 1536-dimensional one, a
2.6× wider `f` in the `v`/`down` matrices only, and 18 % more arithmetic. The
BPB difference (-0.067 → -0.046) is confounded on all three. `[M/D]`

### 0.3 79 % of the "dense model" is one hash table

The live tree of the 2k arm (my count, CPU instantiation):

| component | params | share |
|---|---:|---:|
| Engram tables (3 × 524,288 × 32) | 50,331,648 | 75.2 % |
| Engram `mem_dense` + engram router/gate | 2,460,160 | 3.7 % |
| FFN (`down` 6.29M + shared 1.38M + one rank-64 expert 0.46M + router) | 8,128,512 | 12.1 % |
| attention (latent 256) | 2,002,944 | 3.0 % |
| CSA | 786,432 | 1.2 % |
| `out_proj` + norms + other | 2,473,988 | 3.7 % |
| embed + head + mHC + AttnRes + LayerRoPE | 0.826M | 1.2 % |
| **total** | **66,910,751** | |
| **non-table body** | **14,118,943** | **21.1 %** |

`[M]` — `D_noengram` instantiated at 14,118,943. The 600-step winner's
non-table body is 16,536,031. The run that reached 1.7292 is
**102,678,943** params (`flagship_resume.log`), not the 105,035,148 its label
says (that banner belongs to a different `flagship1.log` run); of the 102.68M,
50.33M is Engram tables (49.0 %), 1.05M `mem_dense`, and 48.87M FFN, of which
≈44.7M is the top-2-of-256 routed bank (`E=256`, `rank=64`, `f=2730`,
`share_u`). `[M/D]`

The sentence "dense 81M beats MoE 199M" is therefore really "a **14.1M**
active transformer + a 50.3M n-gram table beats a 199M model that activates
top-2 of 512 experts". That is a stronger statement about routing, and a much
weaker statement about dense capacity. `[D]`

The active-per-token comparison is the one that matters, and it points the
other way: the MoE's active path is ≈13.4M params (2 experts + shared + one
`f→d` down + latent attention + embed/head), the dense 2k arm's non-table body
is 14.1M, and the 600-step winner's is 16.5M. `[D]` At equal MAC the dense layout is not smaller — it
puts **more useful parameters into the active path per MAC**. That is the
result worth keeping; "2.5× fewer parameters" is a statement about storage,
not about the compute path.

### 0.4 The ledger was measured at 0.03–0.25 B/param, not 31

This is the largest transfer problem in the whole memo.

| run | params | bytes seen | B/param |
|---|---:|---:|---:|
| mHC / AttnRes battery (`medium`, 600 steps) | 141.9M | 4.92 MB | **0.035** |
| dense capacity sweep, 600 steps | 69.3M | 4.92 MB | **0.071** |
| dense 2000-step pair | 66.9M | 16.38 MB | **0.245** |
| planned final, 300k steps | 66.9M | 2.46 GB | **36.7** |
| flagship resume (4.2 GB, 170k steps) | 102.7M | 4.20 GB | **40.9** (stored) / ≈82 (non-table) |

`[M/D]` — bytes = `steps × batch × seq`, the same arithmetic as
`capacity_math.py`. The brief's "31 B/param" is ≈30 (2.46 GB / 81M, rounded up in the
brief); the run that exists is 36.7 B/param. The difference is the parameter mislabel, not the data;
at either number the screen is ~0.25 B/param, and that is the transfer problem. The 2k screen is **150×** less data per parameter than the
final run; the 600-step screens are **500–1000×** less. Every mechanism
verdict in the ledger is an early-training measurement. The 102.7M flagship resume's per-iteration
curve shows why that matters: at step 10k the third iteration is flat
(1.4711 / 1.3396 / 1.3386), and at step 122k it contributes 0.048 CE
(1.3895 / 1.1707 / 1.1228). `[M]` A mechanism that is flat at 0.07 B/param can
pay at 36 B/param; the ledger does not distinguish those cases.

### 0.5 Eval windows differ across the ledger

The dense sweep and the 2k pair score **65,536 B** (`8 × 8 × 1024`). The
2026-10-23 mHC/AttnRes verdict scores **40,960 B** (`10 × 8 × 512`); the
battery document says **81,920 B** (`20 × 8 × 512`). `[M]` The project's own
rule (`AGENTS.md` §2.6) says a BPB is only comparable within one window. The
mHC −0.078 and AttnRes −0.050 were not measured on the window the dense arm
uses, so the ledger's effects and the dense BPBs cannot be subtracted from each
other without a re-run. The direction (mechanism beats its removal on its own
window) still stands; the magnitudes are window-specific.

### What survives

At **equal active MAC**, 3 seeds, one corpus, one window: **dense 69.3M
2.4714 vs E=512 199.3M 2.5384, Δ = −0.0670, control spread 0.0201.** At 2k,
dense 66.9M 2.3195 vs E=512 2.3655, Δ = −0.046, but with +18.4 % MAC. `[M]`
The routing bank is dead capacity at this budget (187–193 of 512 live at 600
steps; 8 of 512 on the m400 150k checkpoint, per the 2026-11-04 audit). The
conclusion "do not put capacity in a routed bank at 22 GB" is measured and
robust. The conclusions "81M", "equal compute" and "31 B/param" are not.

One caveat on the 600-step win: at equal MAC the dense arm's active path is
16.5M vs the MoE's ≈13.4M, so that arm is also a *larger active path* win
(+23 %). The 2k pair is the cleaner routing test — 14.1M active vs 13.4M, +18 %
MAC — and it still wins by 0.046. `[D]`

### Ranked top five

1. **Re-run the 600-step winner at 2k** — `rank=d_model`, `f` solved
   (~4096), `T=2`: 77.5M total / 24.7M non-table / 33.62M MAC. Config only,
   25 GPU-min. §2.1.
2. **Engram off/on**; if it ties, move the 52.8M table params into the active
   body (28.2M non-table, 39.3M MAC). 10 + 25 GPU-min. §1.3, §4.
3. **mHC + AttnRes 4-arm on dense** (mHC only / AttnRes only / both /
   neither), 3 seeds, 600 steps. 40 GPU-min. §1.1–1.2.
4. **Exact attention at 1 KB** — the control the project wrote
   (`ab_attn.py`) and never ran; +22.4M MAC/B (+66 %) and no O(1) state. 10
   GPU-min + profile. §5.1.
5. **Distillation, data filtering, EMA** — the three quality levers with no
   arm; the cheapest is EMA (25 GPU-min), the largest is data filtering
   (20 GPU-min + data prep), the most standard at 81M is distillation (1 day
   of code). §5.2–5.4.

---

## 1. Which mechanisms are now overhead on a dense 66.9M

Costs below are per **step** at the dense run's shape (`b8 × s1024`, 177.6 ms
= 8192 / 46,106 B/s). `[M]` The ms/step attribution comes from the one kernel
profile on record (`docs/records/reviews/2026-11-05-kernel-profile.md`, m400b, same
`d=1536`, same `b8 × s1024`); the regions are per-element/per-position and
transfer across the expert bank, so they are quoted against the dense step.
`[D]`

**Launches and risk, which the table cannot hold.** Per-mechanism launch counts
are not in the profile except mHC (**139 launches per mix**, graph verdict);
the whole step is 4,692 launches (kernel profile). Launch count is not the cost
model here: CUDA graph took 9,325 launches to 59 and the step got **4 % slower**
(graph verdict), and `opt_graph` was +0. `[M]` The risks rank: **Engram** (79 %
of the parameter budget, no evidence), **mHC** (13 % of the step plus +40 %
VRAM for two full-width lanes), **the loop** (46 % more MAC than T=2, and the
only mechanism whose value is demonstrably budget-dependent), **CSA/LayerRoPE**
(low: ties, small).

| mechanism | params | MAC/B | ms/step | % of dense step | measured effect | expected on dense |
|---|---:|---:|---:|---:|---|---|
| **Engram** (3 orders + `mem_dense`) | 52,791,808 | ≈0 (gather) | 5.5 + unmeasured share of "other" | 3.1 % + | **unmeasured** — no clean on/off on this corpus; the old "lost by 1.46" was the broken eval | **unmeasured**; prior ±0.03, decides 79 % of params |
| **mHC** (2 lanes, Sinkhorn 20) | 24,587 | ≈0 | **23.2** | **13.1 %** | −0.078 @142M MoE 600 (spread 0.038, 2.05×); −0.028 @2k (0.74×) | **unmeasured**; prior −0.03 ± 0.03 |
| **AttnRes** (composition, S=2) | 9,216 | S·d/pos | **6.5** | **3.7 %** | −0.050 @142M MoE 600 (spread 0.015, 3.3×) | **unmeasured**; prior −0.02 ± 0.02 |
| **T=3 loop** (vs T=2) | +4,609 | **+10,616,832 (+46 % over T=2)** | ≈1/3 of body | ~30 % | 600: iter3 +0.0000 CE; 2k: +0.0001 (worse); 122k MoE: −0.048 CE | **unmeasured at 36 B/param**; likely pays at 300k |
| **CSA** (fine+coarse) | 786,435 | ≈2.4M (derived) | **7.1** | **4.0 %** | −0.011 (ledger; no spread quoted) | **unmeasured**; at 1 KB prior 0 ± 0.01 |
| **LayerRoPE** (paper) | ≈6,160 | ≈0 | in norm chain | <1 % | ±0 at 5e-4; widens LR band ×12 at 1e-3 | **0 at the recipe LR** |
| **latent attn 256** | −7.47M vs full | **−22.4M** (−34 % of total) | not isolated | — | −0.021 on the dense-readout screen (`ab_width.log`); −0.002 (tie) on the shipped recipe (`ab_width_clean`) | keep |
| **GQA 2** | −0.72M | −2.16M (−4.9 %) | not isolated | — | +0.045 **LOSS** (`ab_width.log`); −0.002 tie (`ab_width_clean`, n=2) | do not adopt without a 3-seed dense test |
| **share_u** | — | MoE-only | — | — | −0.013 on MoE; **inert on dense** (refused by `experts.py:895`) | delete (already off) |
| **prelude/coda** | +1.6M/+3.1M | + | + | + | tie (2.5110 vs 2.5080) | not taken (already 0) |
| **ByteFlow** | +90M fp32 | + | +31 % before codec | — | −0.37 mini, then **−1.5 worse** as full recipe | delete |
| **MTP / JEPA / energy / halt** | + | + | + | + | ties | off |
| **fp8 / muP / compile / graph** | 0 | 0 | 0 / −4 % | — | no gain / tie | closed |

### 1.1 mHC — the biggest single line item, on the weakest evidence

**Cost.** +24,587 params (0.037 % of the model), 23.2 ms/step = 13.1 % of the
dense step, +40 % VRAM for the two full-width residual lanes (verdict,
2026-10-23). The Sinkhorn is 20 iterations (`mhc.py:72`) and the fused write
launches 139 kernels per mix (graph verdict). `[M]`

**Evidence.** −0.078 at 142M MoE / 600 steps, control spread 0.038, so the
effect is 2.05× the spread; the 2k confirmation is −0.028, i.e. 0.74× the
spread — a tie by the project's own rule. `[M]` The 600-step test is
10 GPU-min. On dense, the same residual operator has the same job; the
mechanism is architecture-agnostic in principle, but it has never been ablated
on dense. The correct statement is **unmeasured on dense**, with a prior of
−0.03 ± 0.03, not "−0.078 transfers".

**Verdict.** Test it first (10 min). If it ties, delete 13 % of the step and
40 % of the residual VRAM; the money is in the VRAM, not the BPB. If it wins
≥0.02, keep it and cut the Sinkhorn to 5 iterations (the profile prices
−10…−15 ms; BPB unmeasured).

### 1.2 AttnRes — cheap, small, and measured on the wrong architecture

**Cost.** +9,216 params (0.014 %), 6.5 ms/step (3.7 %). In the shipped
composition the source set is **S=2** (`loopblock.py:113`: `u_0 = 0`, `u_1 =
body`); the "3 sources" in the brief is the AttnRes-only history at the last
iteration (`loopblock.py:836`), which is not the arm the 2k run used. The
third source costs one extra RMSNorm + dot per position and the stored
history; it is not a parameter cost. `[M/D]`

**Evidence.** −0.050 at 142M MoE / 600, spread 0.015 (3.3×). `[M]` On dense:
**unmeasured**, prior −0.02 ± 0.02. The zero-query init makes the composition
bit-identical to mHC-only at step 0, so the arm has to learn to deviate; at
600 steps it did in the MoE.

**Verdict.** Test with mHC in one 4-arm wave (mHC only / AttnRes only /
both / neither), 3 seeds, 600 steps = ~40 GPU-min. Do not test them one at a
time; the composition is the shipped form.

### 1.3 Engram — 79 % of the parameters, zero clean evidence

**Cost.** 50,331,648 table params + 2,360,832 `mem_dense` + 99,328 engram
router/gate = **52,791,808**,
78.9 % of the 66.9M model. 5.5 ms/step of attributed kernels (3.1 %) plus an
unmeasured share of the 18 ms "other" region (gather/scatter). At the final
2.46 GB, each slot is touched on average 2.46e9 × 3 / 1.57e6 ≈ **4,700 times** — the
table is not cold, so "it never trains" is false; "it earns 79 % of the
parameter budget" is the open question. `[M/D]`

**Evidence.** There is **no on/off Engram arm on the current corpus**. The
only on-record comparison is `engram_ram` vs VRAM (2.649 vs 3.476), which
isolates the host-update formula, not the mechanism. The old "Engram lost by
1.46 BPB" is invalid because the eval ran with `hashed_ids=None` (retracted,
`AGENTS.md` §3.2). `[M]`

**Verdict.** This is the highest-value 10-minute experiment in the project.
If off ties (|Δ| ≤ 0.01), the final model should be a **28M active dense
transformer** (the `T2_d2048_f3200_noeng` config I instantiated: 28,212,638
params, 39.35M MAC/B) instead of a 14M transformer with a 50M table. If it
wins ≥0.02, keep it and grow the active body beside it (total ~80M, active
~25M). Either way the answer changes the final model more than any kernel.

### 1.4 The 3-iteration loop — right trick, wrong budget to judge it

**Cost.** The loop is shared, so depth is nearly free in parameters: +4,609
params for the third iteration's slots. It is **not** free in compute:
`T=3` is 33,687,040 MAC/B, `T=2` is 23,070,208, so the third pass is
**+10,616,832 MAC/B = +46 % over T=2**, ~30 % of the whole step. `[M/D]`

**Evidence.** At 600 steps the dense winner's per-iteration CE is
**1.7319 / 1.7097 / 1.7097**; at 2k it is **1.6100 / 1.6078 / 1.6079**. The
third iteration is exactly flat at 600 and 0.0001 *worse* at 2k. `[M]` On the
199M MoE at 2k it is also flat (1.6419 / 1.6383 / 1.6386). On the 102.7M
flagship resume at 122k it is **−0.048 CE** (1.3895 / 1.1707 / 1.1228). `[M]` So the
loop's depth is a **long-budget** effect, invisible at every screening budget
the ledger uses. Deleting T=3 on the 2k evidence would be the same error as
keeping it on the 600-step evidence.

**Verdict.** Do **not** delete the third iteration for the final 300k run on
2k data. Do stop pricing depth at 2k: any depth A/B that ends before ~20k
steps measures zero. The cheap version is a 20k-step T=2 vs T=3 pair (3 seeds,
~6 GPU-h); adopt T=2 only if the 20k delta is within 0.01. At 36 B/param the
third pass is likely worth keeping (the 102.7M resume curve says so).

### 1.5 CSA, LayerRoPE, latent, GQA, share_u

- **CSA** is 786,435 params, 7.1 ms/step (4.0 %), and −0.011 on record with no
  quoted spread — a tie under the project's rule. At 1 KB context the context
  A/B is already a tie (2.605 vs 2.613), so the long-range motivation is gone;
  the remaining job is exact local recall beside KDA. **Delete at 1 KB; keep
  the flag for long context.** `[M]`
- **LayerRoPE paper** is ±0 at 5e-4, the recipe's LR; it buys a ×12 LR band at
  1e-3. At 5e-4 it is pure overhead (4 sites × rotation per iteration, inside
  the norm chain). **Delete from the dense recipe** or raise the LR to the
  band where it pays. `[M]`
- **latent 256** is −0.021 BPB and −35.9 % MAC on the *dense-readout* screen
  (`ab_width.log`, 81,920 B), but −0.0017 — a tie — on the shipped recipe
  (`ab_width_clean.log`, 65,536 B). My instantiation says −22.4M MAC/B at
  `f=4096` (−34 %). It is already on. **Keep**, but the −0.021 in the ledger is
  a different config, not the shipped one. `[M]`
- **GQA 2** is a **LOSS** on the dense-readout screen (+0.0451, `ab_width.log`)
  and a tie on the shipped recipe (`ab_width_clean.log`: 2.5305 vs 2.5320,
  control n=2 because one seed died). It saves 0.72M params and 2.16M MAC/B
  (−4.9 % at `f=4096`). **Do not adopt without a 3-seed dense test**; the
  evidence is contradictory, and the tie rests on a 2-seed control. `[M]`
- **share_u** is MoE-only and refused on a dense body (`experts.py:895`);
  already off. **Delete from the dense config surface.** `[M]`

**Verdict on Q1.** On the evidence, the only mechanisms that are *pure*
overhead on dense are **CSA at 1 KB** (a tie, 4 % of the step) and
**LayerRoPE at 5e-4** (a tie, <1 %); the **third iteration** is pure overhead
at the 2k screen (+0.0001 CE) but pays at 122k, so it is a budget effect, not
overhead; **Engram** is the largest unknown, not a proven overhead; **mHC and
AttnRes** have measured wins on a 142M MoE and no dense ablation at all.
Delete CSA and LayerRoPE now; test Engram, mHC and AttnRes before deleting
them.

### 1.6 Ranked by cost-to-test × chance it pays

Cost is GPU time at 46.1 kB/s: 600 steps = 1.78 min/run; 3 seeds ≈ 6 min,
~10 min with startup/eval. 2k steps = 5.9 min/run; 3 seeds ≈ 18 min, ~25 min
with overhead. `[D]`

| rank | test | cost | P(pays) | expected ΔBPB | why this rank |
|---:|---|---:|---:|---:|---|
| 1 | **Engram on/off**, 3 seeds, 600 + 2k | 10 + 25 min | 0.5 | ±0.03, and 52.8M params at stake | decides 79 % of the parameter budget; no evidence exists |
| 2 | **mHC + AttnRes 4-arm wave**, 3 seeds, 600 | 40 min | 0.5 | −0.05 combined | 16.8 % of the step; the 600-step effects are 2–3× spread |
| 3 | **Full-rank FFN at 2k** (rank=d, f solved), 3 seeds | 25 min | 0.6 | −0.02…−0.05 | the 2k arm is not the 600-step winner; this is a config fix, not an invention |
| 4 | **T=2 vs T=3 at 20k**, 3 seeds | 6 h | 0.3 | 0…−0.05 | the only way to price depth; 2k is blind to it |
| 5 | **CSA off**, 3 seeds, 600 | 10 min | 0.2 | 0 | −0.011 is a tie; 4 % of the step |
| 6 | **LayerRoPE off**, 3 seeds, 600 | 10 min | 0.1 | 0 | ±0 at 5e-4 |
| 7 | **mHC Sinkhorn 20→5**, 3 seeds, 600 | 10 min | 0.7 (speed only) | 0 BPB, −10…−15 ms | recovers 6–8 % of the step; quality risk unmeasured |

The ordering is not "cheapest first" — all the 600-step tests are ~10–40 min.
It is "parameter budget at stake per test", because at 36 B/param the final
number is set by capacity and data, not by step time (see §4).

---

## 2. What would most raise quality per parameter at 66.9M / 36 B/param

### 2.1 The FFN rank bottleneck is the biggest unmeasured confound

The "dense" 2k arm still runs `DualLatentGLU` with **rank 64** on the routed
expert and **192** on the shared expert (`experts.py:520`, `:991`,
`:1173`): the input→hidden map is
`x → gate_u/up_u [d,64] → v [64,f]`, with `f=4096`. The hidden width is 4096,
but the map into it is the sum of a rank-64 and a rank-192 map, i.e. a
256-dimensional subspace. A full-rank GLU would be `[d,f]` = 4096-dimensional.
`[M]`

The 600-step winner did **not** have this bottleneck: `ab_capacity.py` sets
`rank=d_model=1536` for arm D, and its resolved config confirms
`rank=1536, f=1603`. It won by −0.0670. The 2k "confirmation" reverted to
rank 64 and widened `f` to 4096, which puts the extra width behind the same
64-dimensional door. `[M]` This is a copy-paste-class experiment defect, and
it means the dense win is either understated or misattributed.

At equal active MAC to the 2k arm (33.7M), my instantiation gives:

| arm | total params | non-table body | MAC/B | T |
|---|---:|---:|---:|---:|
| 2k arm as run (rank 64, f 4096) | 66,910,751 | 14,118,943 | 33,687,040 | 3 |
| **rank=d, f 4096, T=2** | **77,460,382** | **24,668,574** | **33,621,504** | 2 |
| rank=d, f 1603, T=3 (600 winner) | 69,327,839 | 16,536,031 | 28,442,688 | 3 |
| rank=d, f 1603, T=2 | 69,323,230 | 16,531,422 | 21,655,104 | 2 |
| rank=d, f 4096, T=3 | 77,464,991 | 24,673,183 | 44,238,336 | 3 |

`[M]` (CPU instantiation, `capacity_math`). The second row is the clean
"same compute, 1.75× the active body" arm: 77.5M total, 24.7M non-table,
33.62M MAC/B — the same MAC as the model that ran, with one fewer iteration
and a full-rank FFN. That is the strongest config-only candidate in the queue.
It is **unmeasured** at 2k.

The literature agrees that the gate is not the issue: GLU variants beat plain
FFNs in Shazeer's ablations (`2002.05202`), and this model already has a
SwiGLU. The open axis is the **rank of the up-projection**, not GLU vs plain.

### 2.2 Depth vs width: the loop is the right *param-free* depth, but it is not the capacity lever

At fixed parameters, the loop is the cheapest depth available: a shared body
run T times costs `T × MAC` and ~0 params (`+4,609` for the third slot). The
alternative `loop_blocks=2` doubles the body's parameters (medium:
141.9M → 283.5M, `tests/test_loop_blocks.py:MEDIUM_TOTALS`) and runs the same
compute; at 36 B/param it halves bytes/param. The pre-registered prediction in
`ab_multiblock.py` is a tie, and the driver **has not been run**. `[M/U]` At
66.9M/36 B/param, a 2× parameter body is the wrong direction.

But the loop is not where the unspent capacity is. The 2k arm's non-table body
is 14.1M; the 600-step winner's is 16.5M; the full-rank/T=2 arm's is 24.7M.
The data budget at 2.46 GB and the Chinchilla anchor (20 tokens/param ≈ 80
bytes/param, `2203.15556`) would support ~30M non-table params — **2.1× the
current body**. `[D/L]` The loop adds depth without adding that capacity; width
and FFN rank do. So the answer to "is the loop the right trick at 81M?" is:
**it is the right trick for depth per parameter, but at 66.9M the binding
constraint is the 14M active body, not depth.** Keep the loop for the final
run; spend the parameter budget on the body.

### 2.3 Norm placement, QK-norm, RoPE, tying, head, dropout

- **RMSNorm placement.** The block is pre-norm with one shared RMSNorm before
  the body (`loopblock.py` §1); RMSNorm is the normalization on record
  (`1910.07467`). Pre-LN is the stable choice and does not need
  the warmup post-LN does (`2002.04745`); at 3 iterations the depth is small,
  so post-LN's final-layer advantage is not worth the instability. **No
  change.** `[L]`
- **QK-norm.** KDA already applies an L2 norm to q/k inside the FLA kernel
  (`use_qk_l2norm_in_kernel=True`, `kda.py` module docstring), so QK-norm is
  already present on the main attention arm. CSA has **no** q/k normalize
  (`csa.py` has no `normalize`/`qk` call). `[M]` If CSA stays, add QK-norm
  there; if CSA is deleted at 1 KB, the question disappears. Literature:
  `2010.04245`, `2302.05442`.
- **RoPE vs KDA's implicit position.** The model has no explicit positional
  encoding: KDA's per-channel decay and causal recurrence provide order, and
  LayerRoPE is a depth-wise rotation, not a position. At 1 KB the 1 KB vs 8 KB
  A/B is a tie (2.605 vs 2.613), so position is not the measured problem.
  `[M]` NoPE is a legitimate choice at short context (`2305.19466`); adding
  RoPE to KDA is not a config flag and is not justified by the data. **No
  change.**
- **Tied embeddings.** The arm exists (`config.tie_embeddings`; `ab_train4.py`
  `+tie`), but there is no verdict in the ledger. It saves 393,216 params
  (0.59 %) and the head matmul, and tying is a measured win at small data
  (`1608.05859`). `[L/U]` Price: one 600-step arm, 10 min. Low priority, but
  free if it ties.
- **256-way byte head vs a wider patch head.** The byte head is 393,216
  params and 393,216 MAC/byte (1.2 % of the ledger); it is not the lever. A
  patch head only matters as part of the patching change in §3.
- **Dropout vs weight decay.** No dropout arm exists; `wd=0.01`. At 36 B/param
  the model is not overfitting in the measured sense (the 2k train loss 1.254
  vs eval 1.609 is a held-out gap, not a capacity wall; the flagship "plateau" at
  4.2 GB (102.7M, 170k steps) was measured under full LR decay, which is what a converged run looks
  like, so it is not evidence of a capacity wall either). `[M/D]` Dropout would
  cost capacity at a budget where capacity is the constraint. **Do not add
  it.** The cheaper regularizer test is EMA/SWA (§5).

### 2.4 Ranked answer for Q2

1. **Fix the FFN rank before anything else**: rank = `d_model` (1536), `f`
   solved for the MAC budget, `T=2`. At the same 33.6M MAC/B as the 2k arm this
   is 77.5M total / 24.7M non-table / 33.62M MAC — 1.75× the active body.
   **Unmeasured at 2k; 25 min to test.** `[D]`
2. **Reallocate the Engram budget** if its on/off ties: 28.2M non-table at
   `d=2048, f=3200, T=2`, 39.3M MAC/B (the `T2_d2048_f3200_noeng` arm). This is
   the only change that can add ~2× active capacity at the same data. `[D]`
3. **Keep T=3 for the final 300k run** unless a 20k test says otherwise; the
   102.7M resume's per-iteration curve is the evidence that depth pays late. `[M]`
4. **No GLU, RoPE, norm-placement or dropout change**: GLU is already in,
   QK-norm is already in KDA, position is not the measured problem, pre-LN is
   the stable choice, and dropout is the wrong regularizer at this budget.
5. **Tied embeddings**: one cheap arm, low expected effect.

---

## 3. Is byte-level still right at 66.9M / 36 B/param?

### 3.1 The arithmetic

One byte = one position = one token (vocab 256). The 2k arm runs 8192
positions/step at 177.6 ms = 46,106 B/s. A fixed k-byte patch model runs
`8192/k` positions for the same text. The head arithmetic is the part that
decides whether patching is cheap or impossible:

| head | params | head MAC/byte | body MAC/byte | total MAC/byte | vs byte |
|---|---:|---:|---:|---:|---:|
| 256-way byte (current) | 393,216 | 393,216 | 33.29M | **33.69M** | — |
| k=2, factorised 2×256 | 786,432 | 393,216 | 16.65M | **17.04M** | **−49 %** |
| k=4, factorised 4×256 | 1,572,864 | 393,216 | 8.32M | **8.72M** | **−74 %** |
| k=2, joint 65536-way, rank 256 | 17,170,432 | 8.59M | 16.65M | **25.24M** | −25 % |
| k=2, joint 65536-way, full | 100.7M | 50.3M | 16.65M | 67.0M | +99 % |

`[D]` — head params = `d×vocab` for factorised `k×256`; joint = `d×65536`;
body = the measured 33.29M **divided by k** (the per-position body share). The
factorised head is the key: **k conditional 256-way softmaxes cost exactly the
same head params and MAC per byte as one byte head**, because each byte still
gets one 256-way prediction. The body's MAC per byte falls by k. The joint
65536-way head is not viable at 66.9M: 17.2M params, and its head alone adds
8.59M MAC/B (+26 % over the current body) even at rank 256; the total is still
−25 % only because positions halve, and the full 100.7M-param version doubles
the model's MAC/byte.

### 3.2 The step-time gain is 1.6–2.5×, not 2–4×

The dense step is not all per-position work. The fixed part is the optimizer:
35.4 ms at 199M scales to ~11.9 ms at 66.9M (derived), plus launch/setup
overhead; the per-position part is the rest. `[M/D]` If ~85 % of the dense
step scales with positions, k=2 gives `1/(0.15 + 0.85/2) = 1.74×`, k=4 gives
`1/(0.15 + 0.85/4) = 2.48×`; if 70 % scales, the range is 1.54–2.11×. `[D]`
The honest claim is **1.6–2.5× bytes/s**, and only if the patch encoder/decoder
adds no per-byte work. ByteFlow's conv codec measured 5.4 % of the step at
8 B/patch (0.84 + 0.91 ms of 32.6 ms), so a conv codec is affordable — but the
conv codec **leaked** (the patch latent was built from the bytes it predicted;
`2026-11-03-three-verdicts-and-a-leak.md` §3), and the causal attention codec
cost 34.8 % of the step. `[M]`

### 3.3 Quality: literature says it works at 1B–8B, not at 81M

- BLT: byte-level with entropy patches **matches token-based LLMs at 8B** in a
  FLOP-controlled scaling study up to 8B/4T bytes (`2412.09871`). `[L]`
- SpaceByte: byte-level with word-boundary patches **roughly matches tokenized
  Transformers at fixed training/inference compute** (`2404.14408`). `[L]`
- ByT5: byte-level is competitive but pays in sequence length/compute at equal
  params (`2105.13626`). `[L]`
- MegaByte: multi-scale patching makes million-byte sequences tractable
  (`2305.07185`). `[L]`

None of these is an 81M-parameter, 10 GB result. The project's own ByteFlow
history is the closest evidence: −0.37 BPB on a mini-polygon, then **−1.5 BPB
worse** as a full recipe (it displaced mHC/CSA/LayerRoPE/Engram), +31 % step
before the conv codec. `[M]` At 36 B/param the recipe is capacity-limited, and
a patcher spends capacity on a local decoder that does not add long-range
modeling. The prior should be **neutral-to-negative at 66.9M**, not the 1B
result.

### 3.4 The minimum experiment

One arm, not a recipe:

1. Fixed 2-byte patches (no dynamic boundaries, no entropy model). Patch state
   = causal local encoder over the 2 bytes (1 layer, `d_local=256`, causal
   conv/attention with a **future-byte invariance gate**: changing bytes
   `[:, q:]` must not change logits `[:, :q]`).
2. Global model unchanged (same `d`, `f`, `T`, same 8192 bytes/step → 4096
   positions/step).
3. Head = two conditional 256-way softmaxes (`P(b1|h)·P(b2|h,b1)`), 786K
   params.
4. Control = the byte arm, same bytes/step, same seeds, 600 steps, 3 seeds,
   65,536 B eval. Primary: BPB at equal bytes. Secondary: B/s.
5. Pre-registered rule: **adopt only if BPB ≤ control + 0.005 and B/s ≥
   1.25×**; then run the equal-wall-clock arm (patch model sees 1.25× the
   bytes). If BPB is worse by >0.01, delete the lane for this scale.

Price: ~300–400 LOC + 6 runs × 2 min ≈ 1 day of engineering + 15 GPU-min.
`[D]` Do **not** reuse the conv codec without the causality gate; that is the
defect that already produced a fake −3 BPB.

---

## 4. The single cheapest change with the largest expected effect

**Name it: stop spending 79 % of the parameter budget on a lookup table.
Run Engram off/on, and if it ties, spend the 52.8M params on the active body.**

Why this one:

- The Engram is 52,791,808 params (78.9 %) and 5.5 ms/step + gather; its
  on/off effect on the current corpus is **unmeasured**. `[M/U]`
- If it ties, the final model can be a **28.2M non-table dense transformer**
  (`d=2048, f=3200, T=2`, 39.35M MAC/B) instead of a 14.1M body behind a 50M
  table. That is 2× the active capacity at a 17 % higher MAC, and the MAC is
  not the binding constraint (MoE GEMM is 1.3 ms of 151 ms of kernels = 0.9 % of kernel time, 0.4 % of the wall step; MFU ~2–7 %). `[M/D]`
- If it wins ≥0.02, the final model should keep it and still grow the body to
  ~25M non-table (total ~80M), because 36 B/param total is 174 B/param
  non-table — 2.2× the Chinchilla anchor. `[D/L]`
- No other single change has this range: the full-rank FFN fix is a prior of
  −0.02…−0.05, the loop 0…−0.05, the mHC/AttnRes pair −0.05 combined, and all
  are conditional on a 10–40 min test too. `[D/U]`

**Price.** Engram on/off: 3 seeds × 600 steps = **10 GPU-min** (then 25 min at
2k to confirm). Reallocation arm: one config, 3 seeds × 2k = **25 GPU-min**.
Total < 1 GPU-hour.

**Pre-registered decision rule (write it before the first run):**

1. **VOID** if any arm has a NaN-skipped step, an eval window ≠ 65,536 B, or
   <3 seeds; a one-seed control cannot promote (the `collect_ab` defect).
2. Engram is a **TIE** iff `|mean_off − mean_on| ≤ 0.01` at 2k, 3 seeds, and
   the off arm's spread overlaps the on arm's. A tie **adopts off** (the
   smaller parameter count wins, `ab_capacity.py` rule).
3. If off ties, instantiate `d=2048, f=3200, T=2, no engram` (28.2M/39.35M
   MAC) and adopt it iff it beats the engram control by **≥0.02 BPB** at 2k
   and its B/s is **≥0.85×** the control (the final 2.46 GB run must still fit
   the wall clock).
4. If Engram wins by **≥0.02**, keep it; grow the active body beside it to
   ~25M non-table (`d=2048, f=2400, T=2`: 79,851,934 total, 25,191,838
   non-table, 34.69M MAC/B — +3 % MAC over the current arm) and adopt iff it
   beats the current arm by ≥0.02 at 2k.
5. Any result between 0.01 and 0.02 is a **hold**: re-run at 20k before
   changing the final recipe.

The companion change (do it in the same wave, because it fixes the arm that
was actually measured): **rank = `d_model` and `T=2`** at the same 33.6M
MAC/B. If the full-rank/T=2 arm and the no-engram arm both win, the final
model is ~28M non-table at 39M MAC/B — the first configuration in this project
whose parameter count is mostly compute.

---

## 5. What we are missing entirely

Ranked by expected effect per unit of work. "No arm" means no code path; "arm
never run" means the code exists and has no number.

1. **Exact causal attention at 1 KB — arm exists, never run.**
   `attn_kind="full"` builds `model/fullattn.py`; `ab_attn.py` says in its own
   docstring **"THIS DRIVER HAS NOT BEEN RUN."** `[M]` At 1 KB the O(1) KDA
   state buys nothing (1 KB vs 8 KB is a tie, 2.605 vs 2.613), KDA (`2510.26692`) costs
   15 ms/step (8.4 %) at 2–4 % MFU, and exact attention over 1024 bytes is a
   `1024²` problem this card can do. The compute delta is real: full attention's
   projection bill is 9.44M MAC/token/iteration vs latent KDA's 1.97M — a
   +7.47M/iteration delta, i.e. +22.4M MAC/B at T=3 (+66 % of the 33.7M
   ledger) — and full attention also pays a `2·t·d` score term
   (3.15M/token/iteration) that the KDA recurrence does not. `[D]` Whether that is a net win is
   **unmeasured**; the arm is 100 LOC already written and a 600-step A/B.
   Literature: transformers beat fixed-state models at copying
   (`2402.01032`); pure Mamba lags a Transformer on MMLU while a hybrid
   matches (`2406.07887`). At 1 KB, run the control before adding another
   compression scheme.
2. **Distillation — no arm.** A 124M GPT-2-small teacher over the same 22 GB
   at 66.9M student: the literature's closest result is DistilBERT (40 %
   smaller, 97 % of GLUE, 60 % faster, `1910.01108`); on-policy distillation
   improves further (`2306.08543`). At 81M, distillation is the standard way
   to buy quality per parameter. Price: teacher forward over 2.46 GB (top-k
   logits on the fly, no 2.5 TB fp32 / 630 GB fp8 logit cache) adds ≈0.6× a
   student step (a 124M forward against a 66.9M student's forward+backward);
   1 day of code. **Unmeasured.**
3. **Data filtering / dedup at fixed bytes — no arm in the current recipe.**
   The corpus is 22.2 GB; a filtered copy exists. DCLM shows model-based
   filtering is the key lever and reaches 64 % MMLU at 7B/2.6T (`2406.11794`);
   dedup removes >1 % verbatim copying and saves steps (`2107.06499`); FineWeb
   documents the filter stack (`2406.17557`). `[L]` At 36 B/param, a 0.05–0.2
   BPB data-quality effect is plausible (`[L/U]`) and costs no architecture.
   Price: two 600-step arms on two corpora, 20 GPU-min + data prep.
   **Unmeasured on this recipe.**
4. **Weight EMA / SWA / model soup — `ema_decay=0.0`, no arm.** One shadow
   copy, one update per step, no inference cost; SWA and soups are the
   zero-training-cost version (`1703.01780`, `1803.05407`, `2203.05482`).
   `[L]` At 2.5 GB and a small model this is the cheapest regularizer
   available; the config field already exists. Price: 3 seeds × 2k with
   `ema_decay=0.999`, 25 GPU-min. **Unmeasured.**
5. **LR / batch sweep at 66.9M — no arm.** The 5e-4 LR was validated at 105M
   and 314M, not at 66.9M. Muon's update scale is width-dependent and its
   compute-efficiency claim is ~2× over AdamW at compute-optimal training
   (`2502.16982`); muP is the formal version (`2203.03466`). At `d=1536`,
   5e-4 may be 2× off. Price: 4 LRs × 2 seeds × 600 steps = 8 runs, 15
   GPU-min. **Unmeasured.**
6. **z-loss / logit softcap / QK-norm on CSA — fields exist, all 0.0 or
   absent.** `z_loss=0.0`, `logit_softcap=0.0` in `m400b.toml`; both have code
   (`model.py:902`, `:1009`); neither has an arm. PaLM uses z-loss for
   stability (`2204.02311`); Gemma 2 uses a final logit softcap
   (`2408.00118`); QK-norm stabilizes and allows higher LR (`2010.04245`,
   `2302.05442`). At 3 layers and 5e-4 the expected effect is small, but the
   z-loss/softcap pair is the enabler for the LR sweep in item 5.
7. **Tied embeddings — arm exists, no verdict.** 393,216 params, `1608.05859`.
   Price: 10 GPU-min.
8. **Batch size at fixed tokens — no arm.** b8 is small; the batch ladder
   measured +18 % throughput at b32, and the eval window is
   `eval_batches × batch × seq`, so a batch change moves the window unless
   `eval_batches` is re-scaled. The gradient-quality question is unmeasured at
   66.9M. `[M/U]`

Not missing, measured and closed: MoE capacity (the bank), MTP/JEPA/energy/
halt (ties), fp8/MuP/torch.compile/CUDA graph (no gain), ByteFlow as a full
recipe (worse), 8 KB context (tie), prelude/coda (tie), ternary (cost).

---

## 6. Pre-registered queue (cheap → expensive)

All arms: 3 seeds, one corpus (`mix_v7`), eval window **65,536 B**
(`8 × 8 × 1024`), one batch size, no NaN-skipped steps. A tie adopts the
smaller parameter count; a one-seed control voids the arm.

| # | arm | steps | GPU time | decision |
|---:|---|---:|---:|---|
| 1 | Engram off vs on | 600 + 2k | 35 min | tie → off; win ≥0.02 → keep |
| 2 | mHC × AttnRes 4-arm | 600 | 40 min | keep only pairs that beat the no-op control by >2× spread |
| 3 | Full-rank FFN (rank=d, f solved) at T=2 vs the 2k arm | 2k | 25 min | adopt if ≤ −0.01 BPB and B/s ≥1.0 |
| 4 | CSA off | 600 | 10 min | tie → off at 1 KB |
| 5 | LayerRoPE off | 600 | 10 min | tie → off at 5e-4 |
| 6 | mHC Sinkhorn 20→5 | 600 | 10 min | BPB within 0.01 → 5 iters |
| 7 | Exact attention (`attn_kind=full`) | 600 | 10 min + profile | beat KDA by >spread → replace at 1 KB |
| 8 | Tied embeddings | 600 | 10 min | tie → tie |
| 9 | EMA 0.999 | 2k | 25 min | adopt if ≤ −0.005 |
| 10 | LR 2.5e-4 / 5e-4 / 1e-3 at 66.9M | 600 | 15 min | adopt the best mean; re-confirm at 2k |
| 11 | T=2 vs T=3 | 20k | 6 h | adopt T=2 only if within 0.01 |
| 12 | 2-byte patch head | 600 | 15 min + 1 day code | §3.4 rule |

The first seven are < 2.5 GPU-h total. They answer every question in §1 and
§2 except the long-budget depth one.

---

## 7. What I would delete

- **LayerRoPE paper form at 5e-4** — ±0 at the recipe LR; it only buys a
  wider band at 1e-3. Keep the flag, off.
- **CSA at 1 KB** — −0.011 is a tie, 786K params, 4 % of the step, and the
  8 KB context A/B is a tie. Keep the flag for long context.
- **share_u** from the dense path — refused on a dense body, and the MoE win
  does not transfer. Already off; remove it from the dense recipe surface.
- **The "81M" and "equal active compute" claims** — they are not what ran.
- **The 2000-step dense arm as a confirmation** — it is a different model from
  the 600-step winner. Re-run the winner before citing the 2k number.
- **The rank-64 + f=4096 FFN combination** — it is the worst of both: a
  64-dimensional up-path with a 4096-wide down matrix. Replace with
  rank=d_model and f solved.
- **MTP / JEPA / energy / halt as pretraining losses** — ties, off.
- **ByteFlow as a full recipe** — measured worse by 1.5 BPB; keep the code
  flag for the patch experiment, not the recipe.
- **The "capacity wall" conclusion from the flagship plateau** — the run ended
  under full LR decay; a decayed plateau is not a capacity measurement. The
  size ladder (`ab_dense_size.py`, whose 81M label actually instantiates 66,898,463, and which omits mHC/AttnRes) has only one arm on disk.
  Do not cite the wall until the ladder runs.

---

## 8. Unmeasured — the honest list

- mHC, AttnRes, Engram, CSA on a **dense** body: all four.
- Any mechanism at **>0.25 B/param**: the entire ledger is a 150–1000×
  extrapolation to the final budget.
- The 2k dense arm's **full-rank** counterpart (the 600-step winner at 2k).
- T=2 vs T=3 on dense at **≥20k** steps.
- The **size ladder** (66.9M vs 120M vs 160M) at fixed bytes.
- **Distillation**, **data filtering**, **EMA/SWA**, **LR at 66.9M**,
  **exact attention at 1 KB**, **patch head at 81M**.
- The flagship plateau's cause (schedule vs capacity).
- The eval resolution: the AB protocol's 0.002–0.005 BPB is "estimated, not
  verified"; at 65,536 B and a per-byte CE sd of ~1.5 nats the byte-sampling
  standard error is ~0.008 BPB, so a 0.01 effect is ~1.2 SE. A 0.01 "win" at
  this window is not a number until the paired-eval resolution is measured.
  `[D/U]`

---

## Appendix A — parameter/MAC table (my instantiation)

Command: `CUDA_VISIBLE_DEVICES="" .venv/bin/python` with
`load_run(configs/m400.toml, model_overrides=...)` + `DormouseModel(device="cpu")`
+ `capacity_math.active_macs_per_token`; the "body" column subtracts
`loop.engram.*` + `loop.mem_dense.*`.

| config | total | non-table body | MAC/B | d | f | T | rank | topk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m400b MoE E=512 | 199,275,551 | — | 28,442,624 | 1536 | 4096 | 3 | 64 | 2 |
| 600-step winner D | 69,327,839 | 16,536,031 | 28,442,688 | 1536 | 1603 | 3 | 1536 | 0 |
| 2k arm as run | 66,910,751 | 14,118,943 | 33,687,040 | 1536 | 4096 | 3 | 64 | 1 |
| 2k minus Engram | 14,118,943 | 14,118,943 | 33,687,040 | 1536 | 4096 | 3 | 64 | 1 |
| rank=d, f=4096, T=2 | 77,460,382 | 24,668,574 | 33,621,504 | 1536 | 4096 | 2 | 1536 | 0 |
| rank=d, f=4096, T=3 | 77,464,991 | 24,673,183 | 44,238,336 | 1536 | 4096 | 3 | 1536 | 0 |
| rank=d, f=1603, T=2 | 69,323,230 | 16,531,422 | 21,655,104 | 1536 | 1603 | 2 | 1536 | 0 |
| T=2, d=1792, f=1603 | 72,950,750 | 19,290,334 | 25,720,128 | 1792 | 1603 | 2 | 1536 | 0 |
| T=2, d=2048, f=1600 | 76,831,134 | 22,171,038 | 30,029,824 | 2048 | 1600 | 2 | 1536 | 0 |
| T=2, d=2048, f=3200, no Engram | 28,212,638 | 28,212,638 | 39,348,224 | 2048 | 3200 | 2 | 1536 | 0 |
| no Engram, d=2048, f=1600 | 22,171,038 | 22,171,038 | 30,029,824 | 2048 | 1600 | 2 | 1536 | 0 |

The mHC/AttnRes/CSA/LayerRoPE param deltas (from the same instantiation):
mHC +24,587; AttnRes +9,216; CSA +786,435; LayerRoPE ≈ +6,160; T=3 vs T=2
+4,609; latent 256 saves 7.47M params and 22.4M MAC/B; GQA 2 saves 0.72M
params and 2.16M MAC/B.

## Appendix B — evidence

- Dense 2k run: `~/logs/abdense/dense_s{1,2}.log` and `.jsonl`;
  `~/logs/abdense/moe_s{1,2}.log`; driver `/tmp/opencode/ab_dense.sh`;
  `bpb=2.321 / 2.318`, `bpb=2.363 / 2.368`, `tokens_per_s 43 216 / 46 124`.
- 600-step sweep: `/tmp/ab_capacity/{A,C,D}_s*/run.log`,
  `/tmp/ab_capacity/D_s10/resolved.config.toml`; verdict printed by
  `ab_capacity.py`: A 2.5384, C 2.5382, D 2.4714, bar 2.4982.
- Kernel profile: `docs/records/reviews/2026-11-05-kernel-profile.md` (m400b, b8×s1024,
  320 ms wall, 151 ms kernels, 4 692 launches; optimizer 35.4, MoE dispatch
  31.6, mHC 23.2, AttnRes 6.5, Engram 5.5, CSA 7.1, KDA 15.0, MoE GEMM 1.3).
- MFU/step: `docs/records/research/2026-11-05-small-model-mfu.md` (GPU-busy, not
  host-bound; 1.7–7.3 % depending on ledger).
- Graph: `docs/records/reviews/2026-11-05-graph-verdict.md` (eager 332.8 ms, graph
  348.3 ms, 9 325 launches; the graph is not the lever).
- Flagship resume: `~/logs/flagship_resume.log`, **102,678,943** params
  (`flagship1.log` banners 105,035,148 — a different run), bpb 1.734 at 122k,
  `periter=1.3895,1.1707,1.1228`; bpb 1.729 at 170k/171k,
  `periter=1.3973,1.1663,1.1150`; 4.20 GB at 170k
  (`docs/records/reviews/2026-11-03-three-verdicts-and-a-leak.md` §1 says 105M).
- mHC/AttnRes/Engram verdicts: `docs/records/reviews/2026-10-23-architecture-verdict.md`
  §1.1, §3.1, §5; `docs/records/reviews/2026-10-21-battery.md`.
- Width arms: `~/logs/ab_width.log` (dense readout, 81,920 B, 3 seeds:
  latent −0.0214, gqa **+0.0451 LOSS**, latent+gqa +0.0696) and
  `~/logs/ab_width_clean.log` (shipped recipe, 65,536 B: dense 2.5320,
  gqa 2.5305, latent 2.5303, latent+gqa 2.5340; control n=2, one seed failed).
- Dead capacity / LFB: `docs/records/reviews/2026-11-04-dead-capacity-audit.md` §1,
  §3 (8/512 live at m400 150k; MSA indexer no gradient).
- ByteFlow leak: `docs/records/reviews/2026-11-03-three-verdicts-and-a-leak.md` §3
  (conv codec BPB 3.6532 → 0.4992, a leak; attention codec 3.2427 vs 3.6152).
- `ab_attn.py` docstring: "THIS DRIVER HAS NOT BEEN RUN."
- `ab_multiblock.py` docstring: "THIS DRIVER HAS NOT BEEN RUN."
- `ab_cheapwidth.py` docstring: "THIS DRIVER HAS NOT BEEN RUN." (the
  `ab_width_clean` run is the one on disk).

## Appendix C — literature cited

`1608.05859` tied embeddings · `1703.01780` mean teacher · `1803.05407` SWA ·
`1910.01108` DistilBERT · `1910.07467` RMSNorm · `2002.04745` pre/post-LN ·
`2002.05202` GLU variants · `2010.04245` QK normalization · `2105.13626` ByT5 ·
`2107.06499` dedup · `2203.03466` muP · `2203.05482` model soups ·
`2203.15556` Chinchilla · `2204.02311` PaLM z-loss · `2302.05442` ViT-22B
QK-norm · `2305.07185` MegaByte · `2305.13245` GQA · `2305.19466` positional
encoding/length generalization · `2306.08543` MiniLLM · `2402.01032` copying vs
state-space models · `2404.14408` SpaceByte · `2406.07887` Mamba empirical
study · `2406.11794` DCLM · `2406.17557` FineWeb · `2408.00118` Gemma 2 softcap ·
`2412.09871` BLT · `2502.16982` Muon · `2510.26692` Kimi Linear/KDA.
