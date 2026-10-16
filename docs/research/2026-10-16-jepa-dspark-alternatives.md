# JEPA/DSpark alternatives — what the 2026 record says about aux heads and draft heads

**Date:** 2026-10-16. Read-only: no config touched, nothing committed.
**A:** are JEPA-style latent-prediction aux heads the *right* aux for a small byte-LM (9.2M–1B), or does a cheaper signal win?
**B:** does a DeepSeek-style DSpark draft head (MTP + spec-decode) buy SPEED at train or generation time at ≤1B, given our byte level?
**Builds on, does not repeat:** `docs/papers/jepa.md` (J1-J13), `docs/decisions/2026-09-30-do-we-need-jepa.md` (RETUNE verdict), `docs/papers/dspark.md` (D1-D16), `docs/protocols/AB-PROTOCOL.md` rows 1/1b, `docs/decisions/2026-10-15-1b-readiness-plan.md` §7.
**Method:** ~14 `search_papers` queries (cs.LG/CL/AI/CV, 2025-01→2026-10) + ~30 `get_abstract` reads; every ID below resolved individually through the API. Marks: **(A)** from the paper's own abstract · **(M)** abstract claim, not reproduced here · **(ours)** our measurement, source named · **(S)** SPECULATION, no source.

## TL;DR
1. **[A] No 2026 paper supports a latent-prediction aux on a byte-level or ≤100M AR LM, and the two on-regime studies point the other way:** ProteinJEPA `2605.07554` (**A/M**) gains downstream "without improving MLM loss", and from random init replicates at **p=0.059**; `2604.17228` (**A/M**) removes a JEPA-style gate and *improves* LM loss **3/3 seeds** at 157.5M. Our own `d8062d1` (**ours**) says the opposite (6.343 vs 6.425 BPB, 3/3). Both are true; they do not reconcile, and ours is the only direct AR byte-LM measurement either way.
2. **[A] The signal to trust is NOT the aux loss.** `2606.05753` (**A/M**): cosine/MSE alignment is **negatively** correlated with accuracy, **r = -0.94**, and the aux "reshapes the model via shared parameters rather than via the latent variable it nominally optimizes"; its prescription is a **linear probe + corruption test** — the instrument our protocol lacks.
3. **[A] The cheapest competitor is already pre-registered at 2.1% of the model:** the **future-byte head** (`aux_fb_weight`, AB-PROTOCOL row 1b, source `2404.19737` **verified real**), 197 120 params, **no extra forward** — so it does not pay JEPA's measured price (the teacher **halves the legal batch**, `benches/history.tsv:51-56`, **ours**).
4. **[B] The best ≤1B draft-head number in 2026 is 1.48×, on a different architecture family** (IronLLM-0.6B, `2609.36860` **A/M**); the cheapest *recipe* is post-hoc (`2610.00888` **A/M**: plain CE on ~2.5B tokens on a frozen target ≈ joint pretraining, **10³–10⁴×** fewer MTP tokens). Neither is measured at our scale.
5. **[B] Our exact topology has a published NEGATIVE.** `2605.01106` (**A/M**): **sequential** hybrids (interleaved linear+attention = our KDA interleave) give draft acceptance **α = 0.038** vs **α = 0.68** parallel — 18×, **scale-invariant** (0.5B and 3B match).
6. **[B] Our byte level is the one sourced advantage:** `2610.05978` (**A/M**) — byte LMs learn segmentation-like structure, and exploiting it for speculative decoding yields **3.4× more accepted tokens** than subword models.
7. **Three brief items do not survive verification** (§5): `2602.00232` is **not** "DeepSeek M2"; **no LAUSE/LAUSE-N exists anywhere in the repo**; "next-latency prediction" is not a term in use. `2607.05147` (DSpark) **is** real.
8. **Nothing published shows train-time speedup from a draft head at ≤1B** (§6) — the closest are RL rollouts (`2606.12370`, ≥1.5B-class) and AR *image* generation (`2608.25386`, 39% faster training).

## 1. A — the aux question
### 1.1 WHEN it helps
The 2026 record gives three necessary conditions, plus one diagnostic. Our arm satisfies none:

| condition | source | do we have it? |
|---|---|---|
| a **second view** of the same content — its own authors call this "crucial" | `2509.14252`, restated as a substrate property by `2606.00091` (**M**) | **No.** A byte corpus has no text↔code / paraphrase / Q-A pairing. Unavailable, not merely unattractive. |
| an **input bottleneck**: student fed something the target is not | `2212.07525` (**M**) — teacher on the *unmasked* sample, student on a masked one | **No.** Our mask selects loss *positions*; the predictor sees the full sequence (`aux.rs:177`, **ours**). |
| a target **more abstract** than the student's own state | ProteinJEPA ablation: "adding shallower targets removes most of the task gain" (**M**) | **No.** Target = teacher's T-average readout at the student's own depth. |
| the target is **load-bearing**, not decorative | `2606.05753` PRISM corruption test (**M**) | **Never measured.** No probe in the protocol. |

**2024-25 lore** (EMA teacher + predictor ⇒ self-supervised win, imported from data2vec-2.0/DINOv2) is what the crate header inherited. **2026 evidence** narrows it to: works with two views or a real information bottleneck; otherwise it degrades to **lagged self-distillation**. `2607.23531` (**M**) supplies the theory — MSE prediction of an *arbitrary* continuation is pulled toward the conditional mean of admissible continuations. `2608.05806` HiLP (**M**) adds that plain MTP (limited horizon) and plain NextLat (compounding rollout error) are each weak alone; its fix is a **new mechanism** (a hierarchical abstract latent), not a retune.
**The gap that matters:** this pass again found **no autoregressive natural-text pretraining A/B below 100M params, loss metric, 3 seeds, random init**. That is why `d8062d1` can neither be corroborated nor refuted from the literature.

### 1.2 WHAT signal tells you it worked
The literature's answer changed in 2026 — **the aux loss is explicitly not it**: `2606.05753` (**A/M**) alignment **r = -0.94** vs accuracy, latents "largely bypassed", corrupting them shifts accuracy **≤4 points**; ProteinJEPA wins downstream "**without improving MLM loss**"; `2604.17228` (**A/M**) measured at the loss and the aux *lost* 3/3.
Our protocol already says the same from the other side (`AB-PROTOCOL.md:397`, **ours**): *"Убывание aux-loss без выигрыша BPB или code-задач — не основание возвращать ненулевой вес."*
**Practical rule for dormouse:** report held-out **BPB** (the protocol's primary) **plus a probe** — a linear probe on `out_acc` for the target byte, and a **corruption test** (zero the aux path, re-score). If the corruption test says the contribution is not load-bearing, the term is ornamental whatever its weight or loss curve.

### 1.3 The CHEAPEST competitor, per published study

| candidate | params added | per-step | per-token | VRAM | source |
|---|---|---|---|---|---|
| **future-byte head** (`aux_fb_weight`) | **197 120 = 2.1% of `small`** (**ours**) | one untied `d_model→256` linear + CE, **no extra forward** | +1 shallow head | 1 logits block | AB-PROTOCOL 1b; `2404.19737` (**A**) |
| **LightMTP** (latent MTP from own hidden states) | **≤1%** (**M**) | no conventional-MTP overhead, no external teacher | latent, no token head | not reported | `2610.06031` (**A/M**) |
| **FSP** (future summaries) | an aux head (size unreported) | +1 head; reverse LM for the learned variant | summary target | not reported | `2510.14751` (**A/M**) |
| **JEPA + KoLeo** (current default) | EMA teacher copy + predictor | EMA update **0.14–0.21%** of a step (**ours**) | **+1 full teacher forward** | **halves the legal batch: aux off = batch 16 @ 5908 MiB; teacher on ⇒ batch 4 OOMs** (**ours**) | `~/logs/ab8m_ab8m_iter4.log`; `benches/history.tsv:51-56` |
| **KoLeo alone** | 0 | ~0 | ~0 | 0 | but over **10 rows** (`aux.rs:178`, **ours**) — not the published estimator |

The ranking is decisive because **VRAM is our binding constraint** (readiness plan §2.3). `2404.19737`'s own abstract carries the caveat: *"no overhead in training time"* **and** *"increasingly useful for larger model sizes"* (induction-head evidence at 1–30M algorithmic tasks; LM gains at 7B/13B). The direct negative is `2505.22757` (**A/M**): *"smaller language models (SLMs) struggle with the MTP objective"* — its fix is a **forward curriculum**, not a bigger head. The byte-level anchor is `2608.16671` (**A/M**): on byte-level WikiText-2, 5 paired seeds, *"Tested auxiliary feedback routes do not beat tuned backpropagation."*

## 2. B — the DSpark / MTP draft-head question
### 2.1 What is published at or near ≤1B

| study | scale | speedup | assembly | VRAM |
|---|---|---|---|---|
| **IronLLM-0.6B** `2609.36860` (**A/M**) | **654M** | **1.48× decode** | X-MTP: shared-KV MTP + lightweight verification head, rollback-free; 6.2T-token pretrain then multi-domain on-policy distillation | not reported |
| `2610.00888` (**A/M**) | frozen **Qwen3-8B**, K=3 heads | **+12–16%** (chain-aware verification), **+11–14%** (adaptive head-count controller) | **post-hoc**: plain CE on **~2.5B** target-generated CoT tokens = **10³–10⁴×** fewer than joint pretraining (MiMo-7B baseline) | not reported |
| `2602.06019` (**A/M**) | not given | **>3× decode** at **<5%** GSM8K drop | self-distillation converts a pretrained NTP model in place; **no auxiliary verifier** | not reported |
| `2602.01469` P-EAGLE (**A/M**) | GPT-OSS 120B/20B, Qwen3-Coder 30B | 1.10–1.36× | parallel drafting; gradient accumulation within a sequence | not reported |
| `2606.12370` Bebop (**A/M**) | Qwen3.5/3.6/3.7 | **up to 1.8× end-to-end async RL training** | MTP + rejection sampling; **end-to-end TV loss** on the acceptance rate; **pre-RL** MTP training suffices | not reported |
| `2607.24434` DraftExpert (**A/M**) | DeepSeek-V2-Lite, Moonlight-16B-A3B | **1.45×** decode | one draft expert/layer, self-distilled from the frozen MoE; expert-offload | drafter accelerator-resident by design |

Three rows matter more than that column:
- `2502.09419` (**A/M**): on a frozen NTP model, hidden layers are *"strongly specialized for NTP, making adaptation non-trivial"*, and joint training **"cannot fully overcome this barrier"** — dormouse plans exactly that frozen-logits arrangement (`aux.rs:259` `frozen = logits.detach()`, **ours**).
- `2605.01106` (**A/M**): **the architecture-matched negative** — isolating the linear-attention/SSM subgraph as a zero-cost draft gives **α = 0.68 parallel / 0.038 sequential**, 18×, scale-invariant; and it offers a **free gate**: a perplexity-degradation ratio predicts speculative viability without running SD (3.15× ⇒ α=0.37 at k=4; 81.96× ⇒ α=0.019).
- `2609.17184` LoopSpec (**A/M**): for **looped** transformers — our shape — *"intermediate recurrent states can directly provide draft predictions without an auxiliary draft model"*, **up to 6.83×, training-free**, optimal proposal depths in closed form. The strongest published analogue of our LoopBlock, and it needs **no draft head at all**.
- `2607.21535` Windowed-MTP (**A/M**): the draft's full-context KV read *"comes to dominate the draft cost"* and *"sharpens under hybrid/linear-attention targets"*; windowing the draft's attention only gives **+28–44%** per-decode-step at 1M, lossless. Relevant before any context extension.

### 2.2 Is "our byte level makes it exact replace-striking" true?
**VERIFIED:** the byte level removes tokenizer mismatch, and byte LMs are *better* speculation substrates — `2610.05978` (**A/M**): **3.4× more accepted tokens** than subword. Also verified: in dormouse draft and target are one network (`dspark.md` D15: *"the draft and target are the same network here"*), so this is **self**-speculation, not a drafter/target pair.
**SPECULATION (S2):** that this makes the speedup "exact" or lets the draft replace-strike the target at ≤1B in our shape. The published self-speculation result for our *topology* is α = 0.038; the published ≤1B number is 1.48× on a different model; **not one** study in §2.1 reports VRAM.

### 2.3 Our own blockers, before any DSpark A/B deserves GPU time
From `docs/papers/dspark.md` (**ours**, tree `eeb3b73`): **D1** the window is one position shifted (`model.rs:204` passes `targets` where Eq 4/5/6 need `input_ids`) — one-line fix, open; **D9** the confidence head is wired *without* the Markov feature the paper makes load-bearing (`aux.rs:311` passes `None`; `with_markov` has zero callers); **D2** `DSPARK_GAMMA = 4.0` is decoupled from `dspark_k = 4` where the paper ties one γ=7 to both. Its own verdict: *"Any A/B of the DSpark head right now would move three things at once and attribute the result to none of them."* Nothing in this pass changes that, and the port has no head at all (`port/src/dormouse/config.py:163-165` is `_unsupported(0.0, "DSpark draft head")`, **ours**).

## 3. Table — candidate | cost | claimed gain | measured by | fit to our byte-MoE | plan-item ref

| # | candidate | cost | claimed gain | measured by study? | fit to our byte-MoE | ref |
|---|---|---|---|---|---|---|
| 1 | **pure CE** (control; what the port runs today) | 0 | — | **ours**: BPB 3.001 @5k, window 81 920 B, 0 NaN | the baseline | AB-PROTOCOL row 1 |
| 2 | **future-byte head** (`aux_fb_weight=0.1`, horizon 2) | **197 120 params = 2.1%**; no extra forward | induction-head dev at 1–30M; LM gains at 7–13B | `2404.19737` (**A/M**): "no overhead in training time", but "increasingly useful for larger model sizes" | **best cost/gain available here**; counter-evidence at 9M from `2505.22757` | AB-PROTOCOL **row 1b** |
| 3 | **JEPA + KoLeo** (burn default, `jepa_weight 0.05`) | teacher copy + full extra forward; **halves legal batch** (**ours**) | ours: **6.343 vs 6.425** BPB, 3/3 seeds | **ours** `d8062d1`; literature: `2604.17228` **negative 3/3**, ProteinJEPA **p=0.059** from random init (**A/M**) | works on our data, **uncorroborated**; pays the binding resource | AB-PROTOCOL 1; do-we-need-jepa |
| 4 | JEPA **retuned** (R1 norm + R2 input mask) | same as #3 | none measured | **no study**; prescribed by `2212.07525` (**M**) | the honest version of #3 — but its metric is the one `2606.05753` flags | do-we-need-jepa §6.2 A2 |
| 5 | **KoLeo alone** | ~0 | DINOv2 retrieval **+8.3 mAP** (**A**) | yes, but **vision, n=thousands of patches** | our estimator runs on **10 rows** — **not the same estimator** | jepa.md **J8** |
| 6 | **LightMTP** (latent MTP, own hidden states) | **≤1% params**, no external teacher | parameter-efficient MTP | `2610.06031` (**A/M**) | avoids a 256-way head; untested ≤1B | new §5 |
| 7 | **FSP** (future summary head) | aux head + reverse LM (learned variant) | beats **both** NTP and MTP at 3B/8B | `2510.14751` (**A/M**) | "predict something *further*" at the head, not the latent | new §5 |
| 8 | **HiLP** (hierarchical abstract latent) | extra abstract latent + head | longer-horizon belief state; better SD efficiency | `2608.05806` (**A/M**) | a **new mechanism**, competing with NextLat and MTP | new §5 |
| 9 | **DSpark draft head** (as wired today) | RNN head + `W₁`/`W₂`; `dspark_weight 0.1` | γ=7 blocks; acceptance (paper's 4B+ targets) | `2607.05147` (**A**) at Qwen3-4B/8B/14B + Gemma4-12B — **not ≤1B** | **3 stacked defects (D1/D2/D9, ours) + α=0.038 for our topology** | dspark.md; AB-PROTOCOL 1 |
| 10 | **X-MTP** (shared-KV MTP + verify head) | one head set | **1.48× decode** | `2609.36860` (**A/M**) at **654M** | only ≤1B draft-head datapoint; different attention family | new §5 |
| 11 | **post-hoc MTP head** (CE on target CoT) | ~2.5B extra tokens **once**, frozen backbone | joint-pretraining throughput at **10³–10⁴×** fewer MTP tokens; +12–16% | `2610.00888` (**A/M**) at frozen Qwen3-8B | cheapest *recipe* — but 8B and on a frozen NTP-specialized trunk | new §5 |
| 12 | **LoopSpec** (draft from early recurrent states) | **0 params, training-free** | **up to 6.83×** inference | `2609.17184` (**A/M**), looped transformers | **closest published analogue of our LoopBlock**; no draft head needed | new §5 |
| 13 | **component-aware self-spec** (linear-attn subgraph as draft) | training-free | α=0.68 parallel / **0.038 sequential** | `2605.01106` (**A/M**), scale-invariant 0.5B→3B | **NEGATIVE for our sequential KDA interleave** | new §5 |
| 14 | **Windowed-MTP** | training-free, drop-in | **+28–44%** per-decode-step @1M | `2607.21535` (**A/M**) | long-context only; *sharpens* for hybrid/linear targets | new §5 |
| 15 | **EntMTP** (entropy-guided draft depth) | training-free scheduler | throughput across entropy regimes | `2606.27550` (**A/M**) | serving-side; we have no serving scheduler | dspark.md D13 |
| 16 | **MTP via self-distillation** | one short pass on a pretrained ckpt | **>3× decode at <5%** GSM8K drop | `2602.06019` (**A/M**) | the "no extra verifier" property is the interesting one | new §5 |
| 17 | **Bebop MTP for RL rollouts** | MTP head + TV loss + rejection sampling | **up to 1.8×** end-to-end async RL | `2606.12370` (**A/M**) | post-train lane only, not pretraining | new §5 |
| 18 | **EB-JEPA** (energy-based JEPA) | 256 candidate scores/position/iter | none in LM | `2602.03604` (**A/M**) — a **library** (CV/video/world-models) | rejected: new objective, zero LM-pretraining evidence | do-we-need-jepa §3.2(d) |
| 19 | **SOD** (step-wise on-policy distillation) | teacher + per-step reweighting | fixes OPD's cascade failure for TIR agents | `2605.07725` (**A/M**) — post-training, not an aux loss | not a pretraining aux; belongs to the RLVR lane | agentic-post-training §TL;DR 3 |
| 20 | **circuit probe** `2606.02378` | 0 (measurement) | maps induction/prev-token/BOS head *emergence*, 3 × 1B-class, 10 revisions each | `2606.02378` (**A/M**) | **an INSTRUMENT, not a signal — it cannot replace an aux**; it is what our protocol lacks | do-we-need-jepa §2.1(3) |
| 21 | **MTP curriculum** (NTP→MTP ramp) | 0 extra params | lets **SLMs** use MTP at all | `2505.22757` (**A/M**) | the published precondition if any MTP head is tried at 9M | new §5 |
| 22 | **PRISM diagnostics** (probe + corruption) | 0 (measurement) | alignment vs accuracy **r = -0.94**; latents largely bypassed | `2606.05753` (**A/M**) | **the missing instrument for rows 3/4/5** | new §5 |

## 4. Decision-digest
### 4.1 What goes into the port's tests NEXT
**Item 1 — the A/B: `aux_fb_weight` (future-byte head) vs the port's current pure-CE recipe.**
Why this and not the JEPA retune: it is **pre-registered** (row 1b), **cheapest** by every costing we have (#2 vs #3: 2.1% params and no extra forward, against a term whose price is a halved batch), and the port **already runs the control** (pure CE, BPB 3.001 @5k, `1b-readiness-plan.md:479`). Implementing the arm is the smaller diff.
- **Budget:** 3 seeds × **2 000 steps**, the port's landed WSD recipe. **Matched:** one `batch . seq_len` across both arms = the port's current operating point, so the window (81 920 B) is identical; a different batch voids the table (`AB-PROTOCOL` §"Одинаковое число шагов ≠ matched-compute").
- **Fairness:** equal processed **bytes** for the smoke, equal **tokens** for the verdict; print GPU-seconds per arm.
- **Metrics in order:** (1) held-out **BPB** at matched steps, window printed; (2) **`fb=<ran>/<asked>`** — `0/<n>` means the horizon was past the sequence length and nothing trained (head did not run ⇒ void); (3) **a probe** — linear probe on `out_acc` for the byte at `t+2` **plus a corruption test** (zero the head's contribution, re-score) — row 22's instrument, because the loss cannot answer the question.
- **Decision rule (unchanged):** win = arm mean BPB below the pure-CE mean by **more than the spread across the control's own 3 seeds**; inside the spread = **delete** (ADR-0002).
- **Pre-registered prediction, so it can be falsified:** on `2505.22757` (**A/M**) and `2404.19737`'s own "increasingly useful for larger model sizes" (**A/M**), the head lands **inside the spread or worse** at 9.2M. Write that down before running.
- **Named asymmetry:** the 100M ladder was dropped (owner, 2026-10-15 — b1 directly), so the only remaining rung is b1. A 2 000-step arm at 9.2M can **falsify** cheaply but cannot **confirm** a 7–13B effect. A loss or tie is evidence against carrying the arm into b1; a win is a licence to keep it, not proof it scales.

**Item 2 — the test, not a race: the DSpark D1 window gold vector.** `dspark.md` §7 test 1 (`x[i] = i`; assert the RNN head at `(p, s)` is fed `x[p+s]` and supervised on `x[p+s+1]`) **fails on the current tree**, needs no GPU, and gates any future draft-head decision; D9's width assertion (`d_model + markov_rank`) rides the same file. Cost: CPU only. **Do not** spend a seed set on the DSpark A/B while D1/D2/D9 are live (§2.3).

### 4.2 What rides "measured later"
| deferred measure | why deferred | trigger |
|---|---|---|
| **JEPA retune arm** (R1+R2) | #3 already won on our data; the retune's prescribed metric (cosine) is the one `2606.05753` flags at **r = -0.94**; its cost is the binding resource | only if the future-byte head **loses** and the aux family is still wanted — then run it **with** the probe, not the loss |
| **KoLeo re-pooling** (per-position, not `mean_dim(1)`) | not the same estimator as the published one at 10 rows | only alongside the JEPA arm; never standalone |
| **`correct@budget(bytes)`** for the draft head | the post-train lane does not exist yet; it is the metric that would show "replace-striking" is real | first RLVR report after SFT (`post-training.md`; readiness plan §7.2) |
| **byte-level acceptance rate α on our decode path** | no serving path yet; `2605.01106` gives a **free predictor** worth measuring first | when `port/src/dormouse/model/decode.py` grows a verified-draft path |
| **LoopSpec-style drafting from recurrent states** | needs the loop's intermediates exposed as a draft; **training-free**, cheap later | after b1, if *generation* speed becomes binding (not training) |
| **Windowed-MTP** | long-context only | only if context exceeds the current window |

## 5. New IDs verified (5 + 2, and the corrections)
| id | date | authors | one line |
|---|---|---|---|
| **2606.05753** | 2026-06-04 | Zhang, Fang, Liang | "Cosine Misleads" — cosine/MSE aux alignment **inverted** vs accuracy (r=-0.94); PRISM probe + corruption test; the aux reshapes the model via shared params, not the latent. |
| **2610.00888** | 2026-10-01 | Badarayani, Jay, Zhou, Julienne | Post-hoc MTP heads: plain CE on **~2.5B** tokens on a frozen target matches joint pretraining at **10³–10⁴×** fewer MTP tokens. |
| **2605.01106** | 2026-05-01 | Borobia, Seguí-Mas, Tormo-Carbó | Component-aware self-speculation: linear-attention subgraph as a zero-cost draft ⇒ **α=0.68 parallel / 0.038 sequential**, scale-invariant. **Our topology is the loser.** |
| **2609.17184** | 2026-09-15 | Cho, Cui, Kim | LoopSpec: looped transformers draft from early recurrent states — **training-free, up to 6.83×**, closed-form optimal proposal depth. |
| **2610.05978** | 2026-10-05 | (cs.CL) | Byte LMs learn segmentation-like abstractions; exploiting them gives **3.4× more accepted tokens** than subword in speculative decoding. |
| +**2610.06031** | 2026-10-05 | Czinczoll, Kallini, de Melo et al. | LightMTP: latent MTP bootstrapped from the model's **own** hidden states, **≤1%** added params, no external teacher. |
| +**2505.22757** | 2025-05-28 | Aynetdinov & Akbik | The direct negative for #2: **SLMs struggle with the MTP objective**; a forward NTP→MTP curriculum is the published fix. |

**Corrections to the brief (verified, not assumed):** **`2602.00232` is NOT "the DeepSeek M2 paper"** — it resolves to *"Complexity of Quantum Trajectories"* (quant-ph, 2026-01-30), is absent from `docs/papers/`, and has no `provenance.tsv` row. **LAUSE / LAUSE-N: nothing.** A repo-wide grep (md/tsv/toml + `.bulba/`) returns nothing and arXiv has no match — **the ID is not real, as suspected**. **"next-latency prediction" returns zero** results; the real family is **next-latent / NextLat** (`2608.05806`, `2607.00958`, and the **negative** `2609.36227`, "One-Step Next-Latent Prediction Is Not a World Model"). `2606.02378`, `2605.07725`, `2602.03604`, `2604.17228`, `2404.19737` **all resolve** and are as described.
**MTP sets, by author (the 2511–2610 wave and its roots):** *(pretraining-aux)* `2404.19737` Gloeckle, Youbi Idrissi, Rozière et al. · `2505.22757` Aynetdinov & Akbik · `2510.14751` Mahajan, Goyal, Youbi Idrissi et al. · `2610.06031` Czinczoll, Kallini, de Melo · `2610.05978`. *(draft head / self-spec)* `2502.09419` Mehra, Alonso Garcia, Mauch · `2602.06019` Kirchenbauer, Hans, Bartoldson et al. · `2602.01469` Hui et al. · `2606.27550` Chen · `2607.21535` Valliappan · `2610.00888` Badarayani et al. *(RL rollouts)* `2606.12370` Li, Jiang, Xu · `2605.28184`. *(looped / hybrid self-spec)* `2609.17184` Cho, Cui, Kim · `2605.01106` Borobia et al. · `2607.24434` Han. *(DeepSeek line)* `2607.05147` Cheng et al., DeepSeek-AI + PKU (DSpark) · `2606.26744` HyperDFlash (DeepSeek-V4 MTP).

## 6. VERIFIED (sourced) vs SPECULATION (my inference)
**VERIFIED.** Every ID in §3/§5 resolved through the API with the date shown; numbers are the authors' abstract-level claims unless marked **(ours)**. Ours: `d8062d1` 6.343 vs 6.425 BPB, 3/3 seeds (`docs/guides/ab-testing.md:138`); JEPA EMA 0.14–0.21% and the halved legal batch (`benches/history.tsv:51-56`); future-byte head 197 120 params = 2.1% of `small` (AB-PROTOCOL 1b); port k5000 BPB 3.001 @ 81 920 B, CE 1.46 (`1b-readiness-plan.md:479`); port JEPA/DSpark/aux_fb all `_unsupported` (`port/src/dormouse/config.py:163-165`); DSpark D1/D2/D9 open (`docs/papers/dspark.md`); `2602.00232` is quant-ph.
**SPECULATION (mine, no source).** **S1** that the future-byte head is the right *first* aux A/B — an inference from cost and pre-registration, **not** from a scaled result; the one on-regime citation predicts it *underperforms* at 9M. **S2** that our α would land near `2605.01106`'s 0.038 — our loop is weight-shared with a per-iteration router, *closer* to a sequential hybrid than to a looped transformer's per-depth states, but that is a topology argument, not a measurement. **S3** that byte-vocabulary identity makes the draft/target pair cheaper here; supported in spirit by `2610.05978` (**A/M**), never measured on our decode path. **S4** that a probe + corruption test would show our JEPA term is not load-bearing (the reading do-we-need-jepa §3.1 argues for) — the instrument is published, the result on our model is not.
**I cannot answer on available information:** (i) whether *any* 2026 study shows **train-time** speedup from a draft head at **≤1B** — the closest are `2606.12370` (RL rollouts, ≥1.5B-class) and `2608.25386` (AR **image** generation, 39% faster training, **A/M**); (ii) **per-study VRAM cost** for any draft head — **not one** study in §2.1 reports it; (iii) independent support for `d8062d1` at our regime — **none exists**, and ours remains the only direct evidence either way.

## 7. Open questions
1. Does a **corruption test** show our JEPA+KoLeo contribution is load-bearing on `out_acc`? Cheapest possible experiment: one checkpoint, zero the aux path, re-score BPB.
2. What is **our** perplexity-degradation ratio (the `2605.01106` free predictor)? If it is bad, the DSpark lane closes without a GPU run.
3. Does **LoopSpec**'s draft-from-early-states beat a trained draft head **here**? Training-free, so nearly free to test once the loop exposes intermediates.
4. Is `2610.00888`'s post-hoc recipe the right shape for a **byte** drafter, where "CoT" is not a natural unit? **No source. (S)**
5. `docs-site/tools/manifest.mjs` needs a line for this document before `npm run check` is green — not done here (one-file mandate, no commit).

## 8. Sources
arXiv, fetched 2026-10-16 via the API (id · date as returned): `2404.19737` 2024-04-30 Gloeckle et al. · `2502.09419` 2025-02-13 Mehra et al. · `2505.10518` 2025-05-15 · `2505.22757` 2025-05-28 Aynetdinov & Akbik · `2510.14751` 2025-10-16 Mahajan et al. · `2602.01469` 2026-02-01 Hui et al. · `2602.03604` 2026-02-03 · `2602.06019` 2026-02-05 Kirchenbauer et al. · `2603.19714` 2026-03-20 Tang et al. · `2604.17228` 2026-04-19 · `2605.01106` 2026-05-01 Borobia et al. · `2605.07725` 2026-05-08 · `2605.07554` v2 ProteinJEPA (via do-we-need-jepa) · `2606.02378` 2026-06-01 · `2606.05753` 2026-06-04 Zhang, Fang, Liang · `2606.12370` 2026-06-10 Li, Jiang, Xu · `2606.27550` 2026-06-25 Chen · `2607.21535` 2026-07-23 Valliappan · `2607.23531` 2026-07-26 · `2607.24434` 2026-07-27 Han · `2608.05806` 2026-08-06 Shi, Pearce, Tomar, Sen · `2608.16671` 2026-08-17 Murugan · `2608.25386` 2026-08-26 Niu et al. · `2609.17184` 2026-09-15 Cho, Cui, Kim · `2609.36860` 2026-09-29 Yang et al. · `2610.00888` 2026-10-01 Badarayani et al. · `2610.05978` 2026-10-05 · `2610.06031` 2026-10-05 Czinczoll et al. · `2602.00232` 2026-01-30 **not DeepSeek M2** (quant-ph).
Repo: `docs/papers/jepa.md`, `docs/papers/dspark.md`, `docs/decisions/2026-09-30-do-we-need-jepa.md`, `docs/protocols/AB-PROTOCOL.md:216-217,384-397`, `docs/decisions/2026-10-15-1b-readiness-plan.md:477-479`, `docs/guides/ab-testing.md:138`, `benches/history.tsv:51-56`, `port/src/dormouse/config.py:163-165`, `port/src/dormouse/model/model.py:112-122`.
