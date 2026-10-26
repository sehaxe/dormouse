# MoD / token-skipping — does it skip ~50% of tokens AND give better quality, and does dormouse adopt it at b1?

**Date** 2026-10-16 · **lane** owner's question, one file · **verdict metric** held-out BPB @ matched steps + recurrence-φ, not BPB alone (AB-PROTOCOL §"verdict metric", added 2026-10-11).
**Ask.** "MoD (mixture-of-depths) — that token-skipping tech, was it skipping ~50% of tokens and giving quality BETTER too? Verify against 2026 sources and decide whether dormouse adopts it at b1."
Every arXiv ID below was re-resolved and its abstract/HTML read this session via the arXiv API; the repo numbers are quoted from the files named. Anything no source measures is marked **NOT VERIFIED** and separated at the bottom as **SPECULATION**.

**Method.** Five angles, each with ≥2 sources: (1) MoD primary + full text; (2) 2026 MoD successors (`abs:"Mixture-of-Depths"`, date-sorted, 30 hits); (3) MoR primary + successors (`ti:"Mixture-of-Recursions"`, 64 hits); (4) the refutation side (`abs:"Mixture-of-Depths" AND degrad/worse`, `abs:"adaptive computation" AND small`, plus the null-result and early-exit surveys); (5) the repo's own plan/ADR/protocol numbers. Candidate IDs from the brief were checked and two were wrong (§1).

---

## TL;DR

**Half-right, and the wrong half is the load-bearing one.** MoD's real, measured claim is *"match baseline at a fraction of the FLOPs; be **upwards of 50% faster to step** during post-training sampling"* — the 50% is a **speed** number, not a token-skip fraction, and the headline quality claim is **parity**, with a separate **+1.5% on the training log-prob objective** at isoFLOP (arXiv:2404.02258, read in full). Skips of **~87.5% on the sparse blocks (~44% averaged over all token–block slots)** come at a **12.5% capacity, every-other-block** configuration, and beyond that "performance degrad[es]". The quality-better half is real only at the training-objective level and only for the isoFLOP-optimal variants, at **60M–3B** (paper), and it is a **retrofit-hostile, stability-fragile** recipe (MoDification 2410.14268; GateSkip 2510.13876). dormouse is a **weight-shared 2–4 iteration loop, launch-bound at 87% GPU idle** — the one regime where FLOP-skipping buys the least wall-clock — and the plan already rules: **NO-GO as a b1 default**; keep exactly one cheap arm, gated on the depth curve.

---

## 0. The dormouse facts that constrain every design (read from the repo, cited)

* **Shape.** b1 = 1,066,354,113 total / **~57M active**, `max_iter 2`, topk-2 over 1024 TSCT experts + Engram (`configs/b1.toml`; `docs/decisions/2026-10-15-1b-readiness-plan.md` §2.2). Port `small` = **9,476,993** params, `max_iter 4` in the file, **operating depth 2** on the burn side (`configs/small.toml`; `docs/protocols/AB-PROTOCOL.md:179`).
* **The loop shares weights per iteration.** `loop_block.rs:381` resets `h = h_ctx` each iteration; the readout is the **mean of per-iteration outputs**, and every iteration's own logits are in the loss (recurrence-equivalence, `docs/research/2026-09-27-adaptive-depth-safe.md` §0). So "skip an iteration for a token" is **not** "skip a layer": it changes the readout function, and there are only 2–4 slots.
* **It is launch-bound, not FLOP-bound.** Warm step ~245 ms at 9.2M; **mean GPU utilisation 13.3%, 79% of samples ≤5%**; GEMMs ≈ **16% of a warm step** (AGENTS.md §3.1, D1 correction). Card ceiling bf16 **44.06 TFLOP/s**; effective **0.6–1.2 TFLOP/s → 40–70× headroom, all in software** (plan §2.1).
* **The plan's own verdict, quoted.** ADR-0013: *"the loop runs at fixed max_iter=4 with an honest unweighted CE… MoR-style routers (2507.10524) are the only sanctioned re-entry path for adaptive depth — REJECTED for now: MoR underperforms vanilla at 135M (we are 7.5M) and its per-depth gathers are sm_120-hostile."* Plan §5 trap list, verbatim: *"Carrying losing arms into 1B (GR, AttnRes, mHC, MoR, full-arms `small` — all ties/losses at 9.2M). They re-enter only if the 100M rung revives them."* Token sparsity is costed as **plan §2.4 item 2**: "per-position adaptive depth via the in-tree MoR arm or an ALoDLM-style learned schedule".
* **Queue rows that already exist:** AB-PROTOCOL **row 4** (`--rand-depth`) and **row 4b** (`--max-iter 2`), both "unknown", **3 seeds per arm mandatory**. The port has **no adaptive-depth arm** yet.

---

## 1. VERIFIED (sourced): the IDs, and the two the brief got wrong

| brief said | truth (fetched 2026-10-16) |
|---|---|
| MoD = "2604.00232?" | **2604.00232 is a quantum-dot spectroscopy paper** (Yoo et al., cond-mat). **MoD = arXiv:2404.02258** — Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro (2024-04-02), *"Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"*. ✅ |
| "Mixture-of-Depths-and-Order" | **NOT FOUND**: 0 arXiv hits for the exact phrase; Semantic Scholar returned 429 (rate-limited) — **NOT VERIFIED** as a real title. The plausible real neighbour is MoD's own §4.3 **"Mixture-of-Depths-and-Experts (MoDE)"** (2404.02258). |
| MoR = "2507.xxxx, verify" + "ours = 2504.13372" | **2504.13372 is a robot path-planner paper.** **MoR = arXiv:2507.10524** — Bae et al. (2025-07-14, v3), *"Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"*; the burn arm (`burn-mor`/`burn-mod`, `tools/mor_ab.sh`) is this paper. ✅ |

---

## 2. Comparison table

`evidence` = the strength of the artifact (peer review / preprint / null result). `measured-by` = who produced the number: **paper** / **ours** / **nobody**. "Fit" = transferability to a **byte-level, weight-shared 2–4-iteration, topk-2 MoE loop at ≤1.07B / 57M-active / 9.2M A/B polygon**.

| candidate | real arXiv id | fetched | evidence type | scale tested | claimed speedup | claimed quality Δ | measured-by | fit to dormouse | plan-item ref |
|---|---|---|---|---|---|---|---|---|---|
| **MoD** | **2404.02258** (v1, 2024-04-02) | 2026-10-16 | preprint, DeepMind, heavily cited | 60M–3B, isoFLOP 6e18–1e20 | 220M model #3: **66% faster train step**; "+**upwards of 50%** faster to step" sampling | **parity** at reduced FLOPs; **+1.5%** final-logprob at isoFLOP | paper | **weak**: per-layer gates + distinct weights; our loop has ONE block. Its 12.5%-cap/other-block shape is the only transferable one | AB-PROTOCOL rows 4/4b; plan §2.4.2; ADR-0013 |
| **MoR** | **2507.10524** (v3, 2025-07-14) | 2026-10-16 | **peer-reviewed, NeurIPS 2025** | 135M–1.7B | recursion-wise KV + depth-wise batching; "higher throughput" | **new Pareto frontier** at isoFLOP; **loses to vanilla at 135M** | paper | **native**: MoR's router *is* per-recursion routing for a shared stack; blocked in burn by sm_120 gathers | ADR-0013; plan §5; README `burn-mor` |
| **X-MoD** | **2609.34212** (v1, 2026-09-28) | 2026-10-16 | preprint (freshest successor) | pretraining sweeps (scale n/a in abstract) | decouples sparsity from anchor stride | **scaling law** decomposing capacity-gain / context / stride; beats Dense, MoD, MoE | paper | read for the law, not for a ≤1B byte claim | new |
| **NAG** | **2606.16112** (v1, 2026-06-15) | 2026-10-16 | preprint (Millidge et al.) | n/a in abstract | "substantially reduc[ed]… forward-pass FLOPs" | **~20–25% MoD rates MATCH full-depth baseline** | paper | **the safe-rate number**: 20–25%, parity, *needs a norm-agnostic residual first* | new |
| **TriRoute** | **2607.06601** (v1, 2026-07-07) | 2026-10-16 | preprint | 160M–1.3B, compute-optimal | Pareto at matched inference FLOPs/memory | **Pareto-dominates** MoD+MoE+KV-quant; names a **routing-collapse cascade** | paper | **warning**: joint routing collapses; needs per-axis norm + coupling loss | new |
| **GateSkip** | **2510.13876** (v3, 2025-10-13) | 2026-10-16 | preprint | pretrained, fine-tuned | 15% compute saved on long-form reasoning; "near **50% savings**" on instruct | **>90% accuracy at 15% savings**; parity near 50% **only on instruction-tuned larger models** | paper | **the scale caveat**: 50%-at-parity is a big-model number | new |
| **MoDification** | **2410.14268** (v1, 2024-10-18) | 2026-10-16 | preprint | 3B–70B (retrofit) | ~1.2× latency, ~1.8× memory | **"MoD can barely transform existing LLMs without costly training"**; top-k → **threshold-p** | paper | **the retrofit trap**; only bites if we bolt routing onto a trained ckpt | new |
| **SEWN** | **2608.20632** (v1, 2026-08-21) | 2026-10-16 | preprint, **null result** | two-stream Transformer | none | routing gives **negligible accuracy change** vs parameter-matched baselines | paper | **the honest counterweight** | tech-super-research |
| **Diminishing Returns of Early-Exit** | **2603.23701** (v1, 2026-03-24) | 2026-10-16 | preprint | many modern LLMs | n/a | effectiveness **falls across generations**; higher **>20B**, dense, base | paper | **prior against us** on scale | adaptive-depth §3.1 |
| **Do Transformers Use Depth Adaptively?** | **2604.12426** (v1, 2026-04-14) | 2026-10-16 | preprint | 120M–14B | n/a | pretrained: **"some limited evidence"**; clearer only after finetuning | paper | base-pretrained regime = thin evidence | adaptive-depth §4.2 |
| **MoR-MLLM** | **2610.08830** (v1, 2026-09-27) | 2026-10-16 | preprint | MLLM | adaptive recursion depths | efficiency/effectiveness trade | paper | successor, not our modality | new |
| **Soft-MoR** | **2607.00774** (v1, 2026-07-01) | 2026-10-16 | preprint | vision transformers | recursion for capacity | deeper effective nets | paper | successor, vision | new |
| **bioMoR** | **2608.06727** (v2, 2026-08-07) | 2026-10-16 | preprint | genomic omics | token/expert-choice routing | efficiency | paper | successor, other domain | new |

---

## 3. The QUALITY question, with names — at what scale does depth-skipping stop hurting?

1. **MoD (paper, 60M–3B).** The optimum is **12.5% capacity on every *other* block**; "routing every *other* block was crucial for strong performance", and "performance degrad[es] beyond" 12.5%. At that setting the paper reports **parity at reduced FLOPs** and, for the isoFLOP-optimal variants, **+1.5% on the final log-prob objective** — a *training objective*, not a downstream benchmark. The smallest measured "does not hurt" point is **~60M params** — but on **per-layer, distinct-weight** transformers.
2. **NAG (2606.16112, 2026).** "Moderate MoD rates of **~20–25% match full-depth baseline** performance under equal training compute" — again **parity**, and NAG needed a new residual **architecture** to get it. Best current answer to "what rate is safe": **≈20–25%**, not 50%.
3. **GateSkip (2510.13876).** The **50%-savings-at-parity** point exists **only on instruction-tuned models**, and "for increasingly larger models, this tradeoff improves drastically"; at small scale it is **15% compute for >90% accuracy** on long-form reasoning.
4. **TriRoute (2607.06601).** At **160M–1.3B** learned joint routing Pareto-dominates — but only after fixing a **routing-collapse cascade** with per-axis normalization + a coupling-aware balancing loss.
5. **The refuters.** MoDification (3B–70B): you **cannot retrofit** MoD onto a pretrained LLM cheaply. SEWN (2608.20632): routing is a **null result** vs parameter-matched baselines. 2603.23701: early-exit effectiveness **falls** across generations, **higher only >20B**. 2604.12426: adaptivity in **base pretrained** models is "limited".
6. **The looped-LM number that calibrates us best** (`2026-09-27-adaptive-depth-safe.md` §3.2, RecurTrace 2609.03379, 1.7B): fixed T=2 = 54.71% MathQA; the whole adaptive win is **+53 items from 2.7% of inputs**; ACT/PonderNet **collapse to 1 loop**; confidence exit **overspends** (margin: 5.63 loops, −14 items). Gain "grow[s] with model size from 0.6 to 3.4 points" over 0.6B→8B — **monotone increasing with scale**.

### 3.5 Scale-vs-quality in one view (who says what, at what size)

| study | smallest "no hurt" scale | rate that is safe | verdict word used |
|---|---|---|---|
| MoD 2404.02258 | ~60M (per-layer) | 12.5% on every-other block | "match"/"equal or better" |
| NAG 2606.16112 | n/a (needs NAG residual) | **~20–25%** | "match" |
| GateSkip 2510.13876 | "larger models" (scale-conditional) | 15% (reasoning) / ~50% (instruct) | "match"/">90%" |
| TriRoute 2607.06601 | 160M | learned, joint | "Pareto-dominates" |
| MoR 2507.10524 | **135M loses; Pareto at 1.7B** | learned, per-recursion | "Pareto frontier" |
| 2603.23701 / 2604.12426 | **>20B / finetuned** | — | "diminishing"/"limited" |

**Skilled details that separate wins from losses** (all sourced above): (a) **learned, rank-based expert-choice top-k**, *never* stochastic routing — MoD: stochastic routing is "drastically worse"; (b) **route every *other* block**, keeping frequent full-capacity blocks; (c) **capacity ≈12.5% (MoD) / 20–25% (NAG safe rate)** — 50% is not the safe point; (d) **causality patch** — top-k is non-causal, so sampling needs a learned predictor or an aux-BCE on the router (97–99% accurate; the aux-loss variant costs 0.2–0.3% LM loss); (e) **retrofit needs top-k → threshold-p** (MoDification); (f) **aux-free balancing** is the point of expert-choice (MoR instead *needs* the aux BCE; un-anchored routers hit 66.7% dead tokens). **Distill-from-full is NOT evidenced** in any of these papers — **SPECULATION** if proposed.

---

## 4. The "≤50% of tokens" claim and the speedup math on the FEATURE-SHARED loop

**What 50% actually is.** MoD's abstract says *"**upwards of 50% faster to step** during post-training sampling"*; the training-config number for the 220M variant is **66% faster** (2404.02258 §4.1). The token-skip fraction is separate: **87.5% of tokens route *around* the sparse blocks** at the 12.5% optimum, which averages to **43.75% of token–block slots skipped** ((0.125 + 1.0)/2). So "~50%" is approximately right *as an average*, and it is a *speed* claim, not a quality claim.

**FLOP model, on dormouse's own numbers** (plan §2.1: `FLOPs/token ≈ 6 × active_params_per_pass × max_iter`; §2.2: ~0.40 GFLOP/token at b1, max_iter 2; AGENTS §3.1: GEMMs ≈16% of a warm step, 44 TFLOP/s ceiling, 13.3% util).
Let `F` = one shared-block pass, `T` = iterations, `f` = fraction of tokens skipped at each skipped slot, `s` = slots skipped. Train FLOPs/token = `6·F·T`; saving = `f·s/T`.

| loop | skip rule | FLOP saved | wall-clock saved (GEMMs = 16% of step) |
|---|---|---|---|
| depth 2 (b1/operating) | iteration 2 skipped for f=0.5 | **25%** | **≤4%** |
| depth 2 | f=0.875 (MoD sparse-block rate) | **43.75%** | **≤7%** |
| depth 4 (`small.toml`) | last iteration skipped, f=0.5 | **12.5%** | **≤2%** |
| depth 4 | f=0.875 | **21.9%** | **≤3.5%** |

**The point.** Because our block is **shared**, skipping is **per-token-per-iteration, not per-layer**: at depth 2 there are only two slots, so one skip = 50% of that token's loop FLOPs — but 2–4 slots means coarse granularity, and the whole step is **launch-bound** (13.3% util), so a FLOP cut lands on the ~16% GEMM slice → **≤4–7% wall-clock at depth 2**. That is the repo doctrine ("FLOP savings buy ~no wall-clock at our scale"), now with an instrument behind it. MoD's own 66% is on a **30-layer** model where each route-around removes a whole layer's launches **and** FLOPs; our shared loop has neither the layer count nor the FLOP-bound step.

---

## 5. DECISION DIGEST (≤6 bullets)

* **(a) Does MoD REPLACE the plan's fixed-depth claim? NO.** The plan's own verdict is ADR-0013 — *"fixed max_iter… honest unweighted CE"* — and plan §5 lists **MoR** in the "do not carry losing arms into 1B" set. MoD is a *sibling* of that arm, not a replacement for the default. Fixed depth stays the default; MoD/MoR stay arms.
* **(b) Is MoR (ours) cheaper on our loop than MoD-à-la-layer? Yes.** MoR routes **per recursion of a shared stack** — 1:1 with our block. MoD is **per-layer with distinct weights** and collapses to "per-iteration capacity" on a shared loop *anyway*; MoR is the native, cheaper form. Burn's blocker (per-depth token-subset gathers = sm_120 `ILLEGAL_ADDRESS`) is a **burn-stack** issue, not a port one.
* **(c) Minimal arm to test the idea.** A `--max-depth-routing` arm on the port: `Linear(d→1)` router on the shared block (`768×1 ≈ 768` params, **+0.008%**), **expert-choice top-k, capacity 50%, iteration >1 only, every token runs iteration 1**; carry `h` unchanged on a skip; readout stays the mean over **executed** iterations. **~30–50 LOC**. Budget = AB-PROTOCOL's own number: `tools/mor_ab.sh preflight`, `small`+mor, b20×s512 = **1068 ms/step → 2k-step arm ≈ 36 min, 3 seeds × 2 arms ≈ 3.6 GPU-h** (burn); the port runs byteflow `small` at **14,048 tok/s** (`.bulba/memory.md:310`), so faster.
* **(d) Pre-registered prediction at 9.2M (3-seed rule).** **TIE.** Evidence: RecurTrace's adaptive gain grows with scale (0.6B→8B), smallest tested 0.6B; every depth/residual arm measured at 9.2M so far is a **tie or loss** (plan §5); 2608.20632's null result; the depth budget is 2–4. Predict **within the 3-seed spread on BPB** — no quality win, at most a depth-robustness/inference-knob side effect. If it *does* move BPB, suspect the **readout-function change** (§4), not a routing gain.
* **(e) Best published recipe compacted to our arm.** MoD's **learned expert-choice top-k, capacity 12.5–50%, applied at every *other* iteration, with an aux-BCE router for causality** — the aux-free/rank-based form keeps it collapse-proof (no loss-mass knob). MoR's recursion-wise variant is the better *form factor* but needs the gather. **Recommendation: test MoD-shape first, keep MoR as the fallback.**
* **Gate.** Run it **only after** AB-PROTOCOL **4b (`--max-iter 2`)** and the `--eval-depths` curve say depth *pays*; if depth 4 ≈ depth 2, the whole adaptive family is a compute-only lever and the 3.6 GPU-h go to scale.

---

## 6. NEW ids found and verified this session (id + date fetched = 2026-10-16)

1. **2609.34212** — X-MoD: Practical Scaling Laws for Sparse-Depth Routing Beyond MoD (2026-09-28).
2. **2606.16112** — Scaling Adaptive Depth with Norm-Agnostic Residual Networks (2026-06-15); the **20–25% parity** number.
3. **2607.06601** — TriRoute: joint adaptive attention/experts/KV routing, 160M–1.3B (2026-07-07); routing-collapse cascade.
4. **2610.08830** — MoR-MLLM: Mixture of Recursions for Efficient MLLMs (2026-09-27).
5. **2607.00774** — Soft Mixture-of-Recursions (2026-07-01).
6. **2608.06727** — bioMoR (2026-08-07) — plus re-verified live: **2404.02258** (MoD), **2507.10524** (MoR), **2410.14268** (MoDification), **2510.13876** (GateSkip), **2608.20632** (SEWN/null), **2603.23701**, **2604.12426**.

---

## 7. Open questions (what would change the verdict)

* **Does depth pay at all at 9.2M byte level?** `--eval-depths 1..4` on one checkpoint (AB-PROTOCOL 4b, row A0'). If `BPB@4 − BPB@2` ≤ seed noise, the MoD/MoR arm is closed — the honest answer is "iterations don't help bytes here".
* **Is the readout-function confound real on our mean-of-iterations readout?** Compare prefix-mean-1..k vs iteration-k logits alone before crediting any routing win.
* **Does the sm_120 gather blocker survive the port?** The port is PyTorch, not burn/cubecl, so MoR's per-recursion gather may be feasible there — untested.
* **Does X-MoD's 2026 scaling law predict a ≤100M byte-level regime?** The abstract gives a law, not a head-to-head at our scale — **NOT VERIFIED** for us.
* **Is a byte-level (256-vocab) token-difficulty signal even smooth?** MoD's router correlates with output entropy; the paper is subword — nobody has measured whether byte-level "easy" tokens exist as a stable class. **NOT VERIFIED.**
* **Would the arm's router need its own warmup/seed sweep?** Un-anchored routers hit 66.7% dead tokens (MoR); our 3-seed budget must cover a router init as a config difference.

---

## VERIFIED (sourced) vs SPECULATION (my inference)

**VERIFIED.** All arXiv IDs, dates, scale ranges and quoted claims in §1–§6 are from abstracts/HTML fetched 2026-10-16. MoD's 12.5%/every-other-block optimum, the +1.5% logprob, the "upwards of 50% faster" and "66% faster" lines are direct quotes from 2404.02258. MoR's 135M–1.7B Pareto and the 135M deficit are from 2507.10524 (and the isoFLOP table in `2026-09-27-adaptive-depth-safe.md` §2.2). The dormouse FLOPs, launch-bound 13.3%/16%-GEMM numbers and the ADR-0013/plan-§5 quotes are cited to their files.

**SPECULATION.** (i) The **wall-clock ≤4–7%** column in §4 assumes MoD-shape routing's gather adds no launches and the 16% GEMM share is unchanged — **no source measures adaptive depth on a launch-bound hand-rolled loop**; based on AGENTS §3.1's 13.3% util I read the saving as ≤ the GEMM slice. (ii) The **TIE at 9.2M** (§5d) is an inference from the scale-monotone trend (RecurTrace) + the all-ties-at-9.2M record, **not a measured result**. (iii) "Distill-from-full separates wins from losses" — **no source in this body of work supports it**; do not adopt it on my word. (iv) **"Mixture-of-Depths-and-Order" is unverified** (0 arXiv hits; likely a conflation with MoDE). (v) The exact port tok/s at the A/B shape is extrapolated from a byteflow smoke config, not a 9.2M training log.

## Sources
arXiv (fetched 2026-10-16): 2404.02258 · 2507.10524 · 2609.34212 · 2606.16112 · 2607.06601 · 2610.08830 · 2607.00774 · 2608.06727 · 2410.14268 · 2510.13876 · 2608.20632 · 2603.23701 · 2604.12426 · (guessed-and-rejected: 2604.00232 quantum dots, 2504.13372 robot planner).
Repo: `docs/decisions/2026-10-15-1b-readiness-plan.md` §2.1–2.4, §5 · `docs/adr/0013-fixed-depth.md` · `docs/protocols/AB-PROTOCOL.md:78-84,179-200,220-221` · `AGENTS.md` §3.1 · `docs/research/2026-09-27-adaptive-depth-safe.md` · `docs/research/2026-10-11-alodlm-fang-et-al.md` · `docs/research/2026-10-07-technology-super-research.md` · `configs/{b1,small}.toml` · `.bulba/memory.md:310`.
