# Do we need JEPA? — keep / retune / replace / delete

**Date:** 2026-09-30 · **Tree:** read-only pass, nothing committed, no code touched ·
**Prior pass:** `docs/papers/jepa.md` (2026-09-29) verified the teacher-input fix and
catalogued code-vs-paper deltas. This brief is the **decision** and it corrects two of that
pass's claims (§7).

---

## 0. TL;DR VERDICT

> ## **RETUNE. Do not keep as-is, do not delete yet — and set `jepa_weight = 0` on every preset until an arm wins.**
>
> 1. **Our arm is not JEPA.** It is a data2vec-2.0 EMA teacher + a predictor, driving a
>    **loss-position mask that does not mask the input**, on **raw, un-normalised** latents,
>    with a KoLeo term computed over **10 points**. Two of those four are the *load-bearing*
>    parts of the mechanism and both are missing.
> 2. **Evidence for latent-prediction aux objectives in byte-level or ≤100M-parameter
>    autoregressive LM pretraining is one paper, weakly positive, and it does not move the
>    LM loss.** ProteinJEPA (arXiv:2605.07554v2, 2026-05-08) is the only scale-matched
>    3-seed pretraining A/B I found: 78/114 wins, median +0.0106 on structure tasks,
>    **"wins 81 of 114 comparisons … without improving MLM loss"**, and **from random init
>    the gain "replicates inconsistently across seeds (p=0.059)"**. That is our exact
>    regime (small model, pretraining) and our exact instrument (a loss). Read it plainly:
>    *the one on-regime study found the gain downstream and not in the loss.*
> 3. **The clearest negative result in the closest published architecture says the JEPA
>    aux is net-negative for LM loss.** arXiv:2604.17228v1 (2026-04-19), 157.5M params,
>    decoder-only **depth-routed** LM, 3 seeds: *"jointly removing util/rank improves
>    best/avg LM … in 3/3 seeds for both gates, and the early-to-mid advantage of G3 over G1
>    disappears"* — and removing it cut FLOPs 1.53× → 1.07×. Its G3 gate is a JEPA-style
>    latent-space predictor. This is the nearest published analogue to a LoopBlock with a
>    controller, and it is a negative one.
> 4. **The price is measured, and it is not the unsourced "26% of the step" in
>    `docs/protocols/AB-PROTOCOL.md:216`.** The real, repo-sourced cost is **VRAM, not time**: with
>    aux off the card takes **batch 16 in 5908 MiB = 37% of 16 GB**; with the JEPA teacher
>    on, **batch 4 OOMs** (`benches/history.tsv:51-56`). The EMA update itself is
>    **3.9–5.4 ms of a 2601–2851 ms warm step = 0.14–0.21%**
>    (`~/logs/ab8m_ab8m_iter4.log:10-16`). The teacher forward is not separately timed.
> 5. **Retune = two deltas, both with a named source, before any A/B step is spent:**
>    **(a) normalise the target** (data2vec 2.0: *"activations are normalized using instance
>    normalization"* — verified verbatim, §3.2) **and (b) either mask the input or stop
>    calling it masked.** Then run the 4-arm A/B in §6. **The measurement, not the tuning,
>    is the deliverable** — per AGENTS §1.2 (A/B or death) the arm has never been measured and
>    loses by default.

Confidence: **high** on the implementation findings (read from our own code), **high** on
the negative-transfer finding (three independent primary sources), **low-to-medium** on
"JEPA cannot help a 9M byte LM" (absence of evidence, and the mechanism is not implausible —
see §8 for what would change my mind).

---

## 1. Q1 — Is there any evidence for latent-prediction aux objectives in **byte-level or
small-scale LM pretraining**?

### 1.1 Byte-level: **none. Not one paper.**

Searched: arXiv API metadata across `byte-level + language model + auxiliary`,
`character-level + language model + self-supervised + pretraining`, `JEPA + byte level
language model`. Returned work on tokenization (`2510.16987` Back to Bytes, `2507.12720`
FlexiTokens), byte-latent transformers (`2605.08044`), and unrelated aux heads. **No
paper runs a JEPA/data2vec-style latent-prediction objective on a byte-level LM, at any
scale.** VERIFIED by exhaustive search, 2026-09-30.

This is a genuine gap, and it is explicable: a 256-symbol alphabet leaves no room for a
*separate* input view. JEPA needs two things our regime structurally lacks — a second
**view** of the same content, or a **mask that removes information from the student's
input**. §3 is about how that plays out in each variant.

### 1.2 Small-scale LM pretraining: **one paper, and it is ProteinJEPA.**

`arXiv:2605.07554v2` (2026-05-08, upd 2026-09-23), "ProteinJEPA: Latent prediction improves
protein language model pretraining". Not text, but it is the only study I found that is
**pretraining + small + 3 seeds + compute-matched and step-matched controls**, which is
precisely the shape of our protocol. Its design is also the closest published relative of
ours: an **aux** latent-prediction term on top of a token-level objective, with an EMA
teacher, at 8M–150M params (our band is 7–12M).

What it reports, verbatim from the abstract:

| claim | number | bearing on us |
|---|---|---|
| MLM+JEPA vs compute-matched MLM-only, 3 seeds, 19 tasks | **78 of 114** wins (76 step-matched); 14 losses, 22 ties | a real but **2-in-3** effect, not a sweep |
| median compute-matched gain, structure/homology tasks | **+$0.0106** (vs +0.0041 elsewhere) | the effect is **task-shaped**, concentrated in representation probes |
| vs an off-the-shelf checkpoint | 81/114 wins, median +0.0068, **"without improving MLM loss"** | ⚠️ **the gain is not in the pretraining loss.** Our only instrument is held-out BPB, a loss. |
| **from random initialisation** | gain "is smaller and **replicates inconsistently across seeds (p=0.059)**)" | ⚠️ we always train from random init. This is the row that matters most and it is the weakest. |
| JEPA-only (no MLM) | **"collapses downstream performance"** | matches LLM-JEPA's γ=0 → 0.00% (§2.3). Aux complements, never replaces. |
| causal model control (ProGen3, autoregressive) | beats compute-matched next-token-prediction on **12 of 16** tasks | the one positive causal-LM result that exists |
| ablations | **cosine > MSE**; **"adding shallower targets removes most of the task gain"** | two concrete, free design levers — see §3.2 and §3.5 |

**Read together:** the mechanism is real, small, task-shaped, invisible in the pretraining
loss, and unreliable from scratch at our scale. That is the whole case for keeping it *on the
table* and against keeping it *on by default*.

### 1.3 The transfer limits of the rest of the evidence base, stated plainly

| source | regime | does it transfer to us? |
|---|---|---|
| I-JEPA `2301.08243v3` (2023-01-19, upd 2023-04-13) | ViT-H/14, ImageNet, **bidirectional** block masks, 16 A100 | **No.** Needs a non-causal encoder to mask a block and predict a disjoint block. |
| data2vec 2.0 `2212.07525v2` (2022-12-14, upd 2023-06-15) | vision / speech / **BERT-scale masked LM**, GLUE fine-tuning | **Partly — as a recipe, not as a result.** See §1.4. |
| DINOv2 `2304.07193v2` | 142M images, ViT-g/14, 1B params | **No.** KoLeo's evidence is entirely here. |
| LLM-JEPA `2509.14252v2` (2025-09-11, upd 2025-10-07) | **fine-tuning** 1B–8B on paired-view datasets; pretraining only on synthetic NL-RX-SYNTH from random init | **No** (see §1.4). |
| DLLM-JEPA `2606.00091v1` (2026-05-24) | **fine-tuning** 7–8B diffusion LMs | **No** for the objective; **Yes** for the cost accounting (§2.2). |
| ER-JEPA `2609.36952v1` (2026-09-29) | fine-tuning, NL-RX/GSM8K/Spider/NQ-Open | **No.** Newest in the family, still fine-tuning-only, and it is a *delta on LLM-JEPA*, not new evidence. |
| STP `2602.22617v1` (2026-02-26) | "16× less training data" on **NL-RX-SYNTH** | **No.** Synthetic; the "16×" is on a dataset of generated regexes. |

### 1.4 Two claims I will not repeat, both of which the primary sources contradict

- **data2vec 2.0 does not beat RoBERTa on language.** Its own abstract: it *"matches a
  retrained RoBERTa model **in half the time**"*, and §4.1: *"data2vec 2.0 performs four
  epochs compared to 32 for RoBERTa."* It is a **speed** claim against a masked-LM baseline.
  It was never an autoregressive next-token-prediction comparison, and its NLP result is
  GLUE after fine-tuning a BERT-Base-sized model on 16 A100s. **The paper our crate names as
  its parent does not contain a single result that speaks to our regime.** VERIFIED, §1
  and §4.1 of `2212.07525v2`.
- **LLM-JEPA's pretraining evidence is a synthetic-corpus result.** From the full text:
  pretraining is (i) Llama-3.2-1B **from randomly initialised weights** on NL-RX-SYNTH
  (a synthetic regex dataset), evaluated with a *modified* criterion because the model
  *"fails to reliably learn how to terminate generation"* — 54.38±1.70 → 60.59±1.01, 5 seeds,
  p=2.94e-4; and (ii) `cestwc/paraphrase`, 4 epochs, then fine-tuned to Rotten Tomatoes/Yelp.
  Everything else in that paper — 4 model families, 4 datasets, 1B/3B/7B/8B — is
  **fine-tuning of already-pretrained instruct models**. The authors say so: *"While our
  experiments mostly focus on finetuning, preliminary pretraining experiment are
  promising which we plan to scale."* **Its own authors call the pretraining evidence
  preliminary.** VERIFIED, §4.1 + §6 Conclusion.

---

## 2. Q2 — What does the literature claim it buys, and what does it cost?

### 2.1 The claimed benefits, with the strength of each

1. **"A good next-token predictor is not a good JEPA."** LLM-JEPA §3.3, on
   Llama-3.2-1B + NL-RX-SYNTH: training with `L_LLM` alone leaves the JEPA objective
   *unminimised* (red vs green in their Fig. 3), and adding the JEPA term does **not** move
   `L_LLM` (blue and yellow overlap). Accuracy 51.95% → 71.10%. **This is the single most
   useful claim for us and it is exactly the hypothesis our A/B tests.** But: fine-tuning,
   synthetic corpus, and a *tied-weight* predictor with `[PRED]` tokens. VERIFIED as a
   claim about that setting.
2. **Regularises against overfitting.** LLM-JEPA §4.1: *"we observe that LLM-JEPA resists
   overfitting, whereas standard fine-tuning does not"*, and again in LoRA (§A.5). Our
   binding problem is a curve that reaches 4.997 at step 6500 and regresses to 5.450 by
   19500 (`~/logs/train_nokda.log:91`) — so this is the one benefit that would be *worth*
   real money to us. **Fine-tuning evidence only.** Note the model is at 9M params on a
   2 GB corpus, i.e. overfitting is mostly a small-data artefact, not a small-model one.
3. **Representation structure.** LLM-JEPA §4.2: t-SNE structure appears, and the top-100
   singular values of `Enc(Text) − Enc(Code)` drop "a few magnitudes". **The only way we
   could observe this is a probe, and we have no probe in the A/B protocol.**
4. **Data efficiency.** STP `2602.22617v1`: 16× less data on NL-RX-SYNTH. Synthetic
   dataset; **do not carry this number forward.**
5. **Sample-complexity theory.** `2605.27734v1` (2026-05-26): on a tractable PCFG, latent
   prediction recovers the latent tree with samples *constant in depth* where token-level
   SSL needs samples *exponential in depth*. Also concludes *"explicit stacking such as
   H-JEPA is largely redundant."* **This is a theorem about recovering a synthetic
   compositional tree, not about language quality, and I weight it near zero for the
   decision.** Named so nobody re-cites it as support.

### 2.2 Cost — the number that actually matters for us

**Time.** LLM-JEPA's own conclusion: *"the primary bottleneck at present is the 2-fold
increase in compute cost during training"*. Their fix — **random JEPA-loss dropout (LD)** —
is directly actionable and directly relevant to the `--jepa-precompute` flag we already
ship: *"LLM-JEPA tolerates aggressive loss dropout rates (e.g., 0.5 or
0.75), which leads to higher accuracy under the same compute budget. Moreover, increasing λ
in proportion to the dropout rate can further improve performance. Empirically, we observe
that keeping λ(1−α) approximately constant provides a useful guideline."*
**VERIFIED, §5.2 + Table 6.** Their LD=0.75/λ=4 row is 73.08% vs the LD=0/λ=1 row's
63.57% at equal PFLOPs — the same budget buys ~9.5 pp more when the aux is applied
*stochastically and stronger* rather than *everywhere and weaker*.

DLLM-JEPA `2606.00091v1` prices the same thing from the other side and names our substrate
as the cause: LLM-JEPA *"inherited two steep costs from the causal-attention substrate: it
demands explicit multi-view data (e.g., text-code pairs), and it requires two
gradient-carrying forward passes per step."* Their diffusion substrate removes both, cutting
training FLOPs **33%** vs LLM-JEPA. **The multi-view requirement is the load-bearing half
of that sentence for us — see §3.1.**

**Our own cost, measured, in this repo:**

| what | number | source |
|---|---|---|
| EMA teacher update | **3.9 / 4.2 / 5.4 / 4.3 ms** of 2665 / 2851 / 2601 / 2729 ms warm steps → **0.14–0.21%** | `~/logs/ab8m_ab8m_iter4.log:10-16` (aux on; `ema` timer non-zero) |
| **VRAM, the real price** | aux off: **batch 16 = 5908 MiB = 37% of 16 GB**. With the JEPA teacher on: **batch 4 OOMs.** | `benches/history.tsv:51-56`, 2026-09-28, `422414c` |
| teacher forward | **not separately timed.** SPECULATION: ≈ +1 fwd. On the 9.2M warm profile fwd is 46–48 ms of 245 ms (19%), so ≈ **+19% step time** if it is counted in `fwd`, or **+0** if the timer excludes it. **Unresolved — do not quote a step-cost number until `--timers` is read on an aux-on run.** | — |
| "26% of the step" in `docs/protocols/AB-PROTOCOL.md:216` | **UNSOURCED.** No derivation in the repo. Under AGENTS §1.4 this number should be struck or derived. | — |

The VRAM row is the finding that changes the decision. **At 16 GB, the JEPA teacher costs us
batch size, and batch size is the knob that buys gradient quality** (AGENTS §3.1: the batch
ladder buys +18% throughput for 8→32, but its real product is variance reduction across
seeds — and §1.2 requires 3 seeds per arm). A second full forward that halves the legal
batch is not a rounding error in a 3-seed A/B.

### 2.3 Negative results, collected

- **LLM-JEPA Appendix A.3, Table 10.** A `γ` weight on the LM term: γ=0 → **0.00±0.00%**
  ("it generate only empty output"); γ=0.1 → 45.80; γ=λ → 70.42. *"L_LLM remains essential
  … the JEPA component primarily serves as a regularization term, complementing the
  generative loss."* **The aux is a regulariser, never a replacement.**
- **LLM-JEPA §4.3, Table 3 — the metric ablation, and the most useful table in the paper.**
  Same LR/λ/k, NL-RX-SYNTH: baseline 57.29±5.32 · **cosine 71.46±1.34** · MSE 70.64±2.05 ·
  prepend 68.07 · reversed direction 65.70 · **ℓ2-norm 2.22±0.07 (collapse)** ·
  **InfoNCE 34.40±6.10 (worse than baseline, double the variance)**. A **scale-sensitive
  regressor on embeddings is where this family dies.** We use raw **L1** — §3.2.
- **ProteinJEPA** — from random init, p=0.059; JEPA-only collapses.
- **2604.17228v1** — the closest architecture, and negative: aux removal improved LM in
  **3/3 seeds** and deleted the JEPA gate's advantage.
- **2508.19228v2** ("Predicting the Order of Upcoming Tokens", 2025-08-26, upd 2026-02-16):
  MTP as an LM-pretraining aux *"shows inconsistent improvements, underperforming in standard
  NLP benchmarks"*; the authors found exact-future-token prediction *"too difficult as an
  auxiliary loss"* and replaced it with a *much easier* rank-ordering aux (TOP), which does
  beat NTP at 340M/1.8B/7B. **General lesson for us: aux difficulty is the tuning knob, and
  an aux that is too hard is worse than none.** This is also the closest read on our DSpark
  decision, which was already made.

---

## 3. Q3 — Which variant would *our* shape want?

### 3.0 What our shape actually is (read from the tree, 2026-09-30)

- Student latent = `out_acc`, the **readout average over T=2–4 iterations** of one
  weight-shared `LoopBlock` (`model.rs:103`).
- Teacher = **the same architecture, same depth, EMA 0.999**, run on the **same
  unmasked input** (`model.rs:150-156`). The teacher-input defect is fixed and gated
  (`tests/jepa_teacher_seam.rs`).
- The mask (`burn-jepa/src/mask.rs`, 15% span-8) is threaded into
  `jepa_l1_loss` as a **loss-position selector**. `pred.forward(student_latent)` runs the
  **full, unmasked** sequence (`aux.rs:177`). **Nothing is hidden from the model.**
- Loss is **L1 on raw latents**; there is no normalisation anywhere on the path.
- KoLeo runs on `student_latent.mean_dim(1)` → **`[b, d]`** = **10 rows at batch 10**
  (`aux.rs:178`), weight 0.1 × 0.05 = 0.005.
- `lejepa_loss` (SIGReg) ships **uncalled**.

### 3.1 The structural problem, stated once

A JEPA term only carries information if the target is **not already available to the
predictor**. Three things can supply that, and our regime has exactly one:

1. **A second view of the same content.** LLM-JEPA's entire language result rests on this:
   *"being able to obtain non-trivial views … is crucial to the success of JEPA
   objectives"*; its data is (text, code), (question, answer), (context, continuation),
   (paraphrase, paraphrase). DLLM-JEPA restates it as a property of the causal substrate.
   **A 46 GB byte corpus has no second view.** There is no text-code pairing, no
   paraphrase groups, no question-answer structure. **Variants that depend on a second view
   are unavailable to us, not merely unattractive.**
2. **An input mask** (data2vec 2.0, I-JEPA, BEiT). Available to us, and **currently absent**.
3. **A deeper/abstracted target** (data2vec 2.0's top-K layer average; ProteinJEPA's
   half-depth target, whose ablation says shallower targets delete most of the gain).
   Available to us via the loop's per-iteration states. **Currently absent** — our target is
   the same-depth readout average.

**With 2 and 3 both absent, our term is self-distillation of a lagged copy of the student's
own visible input through a 2-layer head.** The optimal predictor is ≈ identity, the loss
is dominated by tracking the EMA's lag rather than by any prediction, and the gradient it
injects into the backbone is a small pull toward the teacher's recent past. That is a
plausible recipe for *mild smoothing* and an implausible recipe for *representation
learning*. It also explains why no one has ever measured a win: there may not be one to find
in the current shape.

### 3.2 Variant comparison against our constraints

| variant | information bottleneck available to us? | cost | evidence at our scale | verdict |
|---|---|---|---|---|
| **(a) ours: data2vec-2.0 EMA teacher + mask-as-loss-weight + raw L1** | **none** (mask doesn't touch input, target is same depth) | +1 forward; halves legal batch (measured) | none | **retune, do not keep** |
| **(a′) data2vec 2.0 as written: EMA teacher over the FULL unmasked sequence, target = mean of top-K blocks after instance norm, student input masked** | **yes, both** | +1 forward, amortisable by multi-mask M (their M=8/16) | weak-positive at 8–150M (ProteinJEPA) | **RECOMMENDED REPLACEMENT** — this is the one variant that is both correct and has on-regime evidence |
| **(b) I-JEPA** | needs a **bidirectional** encoder: context block and target block must be disjoint | n/a | vision only | **reject** — impossible under causal attention without bolting on a second non-causal encoder, which is a different project |
| **(c) predictor-less / stop-grad variants (BYOL-style, LeJEPA/SIGReg)** | no teacher and no predictor ⇒ **no bottleneck at all** ⇒ collapse is the only fixed point unless the regulariser carries it | **cheapest arm in the family — zero extra forward** | LeJEPA `2511.08544v3`; `2606.01443v1` UR-JEPA shows the isotropic-Gaussian target is in tension with the manifold hypothesis | **keep `lejepa_loss` as the cheap arm (e) — it is already written and uncalled** |
| **(d) EBM-flavoured (EB-JEPA `2602.03604v3`, 2026-04-08, ICLR world-models workshop)** | different mechanism: energy head over (context, candidate) | 256 candidate scores/position/iteration | **zero** at our scale; AGENTS §3.6 already lists it as a research-playbook item | **reject for this decision.** It is a *new objective*, not a JEPA variant, and it has no LM-pretraining evidence. Queue it separately, do not bundle it into this A/B |
| **(e) nothing / KoLeo alone** | n/a | free | KoLeo alone: see §4 | **the control.** Must be in the sweep |

### 3.3 The two mandatory retunes (both sourced, both cheap)

**R1 — normalise the target.** data2vec 2.0, §3.1, verbatim: *"Before averaging,
activations are normalized using **instance normalization** (Ulyanov et al., 2016). Layer
normalization of the averaged targets can be useful for some modalities such as speech and
vision."* Their per-modality tables list `IN → AVG → LN` for both vision and NLP
(Appendix A, Tables 8-10). Ours has no normalisation at all.
**Why it is load-bearing:** LLM-JEPA Table 3 shows an un-normalised/scale-sensitive regressor
on embeddings is the exact failure mode (the ℓ2-normalised variant collapsed to 2.22%), and
ProteinJEPA's ablation says **cosine beats MSE**. On raw latents the magnitude of the JEPA
term is a free parameter set by the backbone's output scale, which under our LoRA/TSCT
quantisation and the `--bf16` path is not a constant.
**Cheapest faithful form:** cosine on the normalised pair (satisfies data2vec's IN, LLM-JEPA's
metric ablation, and ProteinJEPA's cosine>MSE in one change). One line in `losses.rs`.

**R2 — make the mask bite, or rename it.** Two sub-options, and this is the one real
**design fork** for the owner:

- **R2-input (recommended).** Feed the student a masked *input* (span-8 already covers
  exactly the right structure: a masked span of bytes is the byte-level analogue of a
  masked patch block, and unlike a "second view" it is available from a raw corpus). Teacher
  keeps the full unmasked sequence — verbatim data2vec 2.0. This makes the term a
  **prediction** and it is the only change that gives the arm a reason to exist.
- **R2-drop.** If input masking is judged too invasive, delete `mask.rs` and the `(seed,
  step)` mask stream (~50 lines that exist only to serve it), run the arm unmasked, and let
  the target be the loss over all positions. Honest, cheap, and it is the honest reading of
  what the code does today. **It is a weaker arm but it is not a fake one** — and a weak arm
  that is honestly named beats a strong-sounding one that is not (AGENTS §1.4).

**Do not do R2 and R1 in separate A/B arms.** That is a 4-arm × 3-seed sweep on a cost
whose step-time is currently unmeasured. Bundle them as one "retuned" arm.

### 3.4 Does the weight-shared loop change what a latent target should be? — YES, and this
### is the most interesting finding for our shape

`out_acc` is the **readout average over T iterations**. The teacher is the EMA of a student
that also averages T iterations, at the same T. So the target is a lagged copy of the same
depth-averaged function — while at eval, `--eval-depths` scores the model at depths 1..=T,
i.e. **the artifact we evaluate is a function of T while the target never was.**

ProteinJEPA's ablation is the transferable finding here: *"adding **shallower** targets
removes most of the task gain."* The target's abstraction depth is load-bearing, and our
target is the *shallowest* available one. The loop gives us three cheaper options than any
of the papers do:

- **T-iteration target:** regress the teacher's latent at the *final* iteration rather than
  the T-average — strictly more abstract, and the per-iteration states already exist
  (`loop_block.forward_full_state` builds them; `L_Rec` is accumulated per iteration).
- **k-iteration target with k > T** (extra teacher-only iterations): the direct analogue of
  data2vec 2.0's "average the top-K blocks" and of ProteinJEPA's half-depth target.
- **leave it at T.**

**This is a second A/B fork, and it is the more interesting one** — but it is also the more
expensive, and §6 budgets for exactly one fork. Recommendation: **fold it into the long gate
(§6.3), not the confirm.** It is a hypothesis, not a default, and there is no evidence for
any of the three.

### 3.5 And the thing that is cheapest of all

ProteinJEPA's other ablation: **cosine beats MSE**, and the target must be *abstract*. Both
point the same way as R1. Combine: **cosine on the instance-normalised target, regressed at
the deepest teacher state available.** That is data2vec 2.0 + ProteinJEPA + LLM-JEPA's
metric ablation converging on one line of code, and it is the strongest single-line
recommendation in this brief. SPECULATION that all three transfer to a 9M byte LM — none of
them was measured at that scale.

---

## 4. Q4 — KoLeo specifically

### 4.1 What the evidence actually is

**There is no evidence for KoLeo on language models, and I could not find a study that
applies it to one.** VERIFIED by search across the arXiv metadata index (title/abstract/
authors) on 2026-09-30 for `KoLeo`, `KoLeo + language`, `uniformity regularizer + text`.
Caveat I must state: the arXiv `searchtype=all` index did **not** return DINOv2 itself for
the query `KoLeo`, so it is not a true full-text index — **I am reporting absence in the
index, not proof of absence in the literature.** Anyone with a real full-text tool
(Semantic Scholar / Google Scholar / Connected Papers) should re-run this before the
delete is executed.

### 4.2 What DINOv2 actually measured (the only primary source)

`2304.07193v2` §6.4, Table 3(a). ViT-L/16, ImageNet-22k, same iterations:

| | INet-1k (linear) | INet-1k-A | ADE-20k (seg) | Oxford-M (retrieval mAP) |
|---|---|---|---|---|
| without KoLeo | 85.3 | 70.6 | 47.2 | 55.6 |
| with KoLeo | **85.8 (+0.5)** | **72.8 (+2.2)** | 47.1 (−0.1) | **63.9 (+8.3)** |

The paper's own reading: *"the instance retrieval performance improves by more than 8%,
confirming that this term helps spread features in the output space. At the same time, the
other metrics do not suffer."* DINOv2's §6.1 recipe table also credits KoLeo with +2.3 k-NN /
+0.6 linear.

**Two consequences for us, and both are structural:**

1. **The measurable effect lives in retrieval.** Classification moves +0.5. **Our instrument
   is held-out BPB.** A +0.5-point classification effect and a loss metric are different
   quantities; DINOv2 does not report a loss.
2. **DINOv2's KoLeo is over n = batch × patches (thousands of points). Ours is over b=10
   rows.** `aux.rs:178` pools the whole 512-position sequence to one vector per sequence.
   `-mean(log(min_dist))` over 10 points on a d=768 sphere is a function of 10 pairwise
   distances with enormous variance, and it enters the loss at weight **0.005**. **Whatever
   DINOv2's ablation says, it does not apply to a term computed on 10 samples.** The prior
   pass called this "statistically vacuous" (`jepa.md` J8); the sharper statement is that
   **it is not the same estimator**, so the +8.3 retrieval number is not a prior for it.

### 4.3 Does it survive without JEPA?

**It does not need to — it is not a JEPA term.** KoLeo is an independent uniformity penalty
on whatever tensor you hand it; `koleo_loss(pooled)` is callable on the CE hidden state with
no teacher, no predictor and no second forward. So the question is not "does KoLeo survive
JEPA's death" but "is a 10-point uniformity penalty on a mean-pooled byte-LM latent worth
0.005 of the loss". **On the evidence: no, and it should not be promoted to a standalone
arm.** Its correct status in the A/B is: **it rides inside the JEPA arm, unchanged, and the
JEPA arm is judged as a whole.** If the JEPA arm loses, KoLeo dies with it — do not spend a
seed set separating them. If someone wants a KoLeo-alone number later, the honest version
needs the pooling fixed to per-position latents first, which is a different arm with a
different (also unevidenced) premise.

---

## 5. What is VERIFIED vs SPECULATION

**VERIFIED (sourced, primary):**
- Our implementation's four properties in §3.0 — read from `model.rs`, `aux.rs`,
  `burn-jepa/src/{losses,mask,predictor}.rs` on this tree.
- data2vec 2.0 normalises targets with instance norm; target = mean of top-K teacher blocks
  over the **unmasked** sample; **student input is masked**; teacher momentum ramps
  `τ₀→τ_e` over `τ_n` updates; multi-mask M amortises the teacher; L2 loss
  (`2212.07525v2` §3.1, §3.3, App. A).
- data2vec 2.0's language result is *"matches a retrained RoBERTa … in half the time"*,
  4 epochs vs 32 — a speed claim, not a quality claim, and not against AR LM pretraining.
- I-JEPA **uses an EMA target encoder** (momentum 0.996 → 1.0) — see §7.
- LLM-JEPA's numbers in §2.1-2.3, all read from the v2 HTML full text.
- ProteinJEPA's numbers in §1.2, read from the v2 abstract.
- DINOv2's KoLeo ablation in §4.2, read from the v2 HTML.
- `2604.17228v1`, `2508.19228v2`, `2509.24317v1`, `2606.00091v1`, `2602.22617v1`,
  `2605.27734v1`, `2602.03604v3`, `2511.08544v3`, `2609.36952v1`: abstracts via the arXiv API
  with version and date recorded in §8.
- Our cost numbers in §2.2, read from `benches/history.tsv:51-56` and
  `~/logs/ab8m_ab8m_iter4.log:10-16`.

**SPECULATION (my inference, explicitly not measured):**
- **S1.** That our term currently behaves as lagged self-distillation rather than prediction
  (§3.1). Inferred from reading the call graph, **not measured**. A one-off check would
  settle it: log the masked-vs-unmasked JEPA loss on one batch at step 0 and step 500 of a
  20-step run. If they are within a few percent, the term is a copy task and the arm is dead
  on arrival. **This is the cheapest possible test of the whole thesis and it should run
  before any A/B step is spent.**
- **S2.** That the teacher forward costs ≈ +19% of step time. The `fwd` timer does not
  decompose it; the EMA update is measured at 0.14-0.21%.
- **S3.** That cosine-on-normalised-target transfers from protein/Masked-LM/synthetic-text to
  a 9M byte AR LM. No paper measured it there.
- **S4.** That the three loop-target variants in §3.4 differ at all.
- **S5.** That no KoLeo-for-LM study exists. §4.1 states the search instrument's limitation.

**I cannot answer, on available information:** what a JEPA term would buy *our* model. The
first real measurement is still ahead — the pre-2026-09-27 archive contains no valid JEPA
result, and this brief changes that only by naming the arms.

---

## 6. The A/B that would settle it

Consistent with `docs/protocols/AB-PROTOCOL.md` (3 seeds, 2k steps, one batch size across all arms so
the eval window matches, window printed, pure CE baseline, 200-500 step smoke first).

### 6.1 Preconditions — run these first, they are free and they gate everything

| # | check | pass criterion | why it gates |
|---|---|---|---|
| P0 | **re-baseline the control first** (AGENTS §3.4). Read `fused kda=<f>/<b>` on the eval line. | the control is a run whose attention arm received a gradient | every arm below is undefined against a frozen-attention control |
| P1 | **`--timers` on an aux-on run, warm step.** | the teacher forward is either inside `fwd` or it is not; record which, and the ms | §2.2's cost is currently unresolved; do not price an arm you cannot cost |
| P2 | **S1: masked vs unmasked JEPA loss on one batch, step 0 and 500.** | if they are within ~5%, the mask is decorative — say so in the log and skip R2-input | one 20-step run settles whether the arm is a prediction task |
| P3 | **VRAM at the arm's batch size with the teacher on.** | peak MiB printed, from the ladder | the teacher halves the legal batch (§2.2); the whole A/B must run at one batch |

### 6.2 The four arms (confirm tier)

All at **one** batch size, one depth, pure CE elsewhere, `--dspark-weight 0`, 3 seeds,
2000 steps, window printed. Control is re-baselined in the same sweep.

| arm | flags | what it decides |
|---|---|---|
| **A0 control** | `--jepa-weight 0` | the number everything is compared to |
| **A1 aux as-is** | preset default (`jepa_weight 0.05`) | is the current arm worth anything *as it is*? Per ADR-0002 a tie deletes it. |
| **A2 aux retuned** | A1 + R1 (cosine on instance-normalised target) + R2 (input mask, **or** mask machinery removed) | does the corrected version beat A0? This is the arm that can justify the cost. |
| **A3 aux retuned, no KoLeo** | A2 with `KOLEO_WEIGHT = 0` | separates the 10-point uniformity penalty from the L1 term. **Only run this if A2 beats A0** — otherwise it is a seed set spent on a loser. |

**λ grid, pre-committed and small.** LLM-JEPA's headline cost result (§2.2) says the right
move under a fixed budget is a *stronger, stochastically applied* term: their guideline is
keep **λ(1−α) ≈ const**. With `--jepa-precompute` we get α for free (precompute every N-th
batch). Pre-commit to **λ ∈ {0.05, 0.2} at α = 0.5** and nothing else. LLM-JEPA's own
limitation section admits *"the optimal configuration may occur at any point in a grid"* —
a λ sweep is how that paper burned its budget, and with 3 seeds and an unmeasured step cost
we cannot afford it. **A2 wins or it does not.**

**Decision rule (unchanged from the protocol):** an arm wins if its mean held-out BPB at
2000 steps is below A0's mean by more than the spread across A0's own 3 seeds. Inside that
spread = no difference = **delete** (AGENTS §1.2). A2 losing to A1 by more than the spread
means the retune is the problem, not the family.

**One prediction to write down before running, so it can be falsified:** if the JEPA term is
a *regulariser* (LLM-JEPA §A.3, §4.1), the curve shape should differ, not just the endpoint
— A2 should show a **higher** step-2000 BPB with a **flatter or later** regression, or
better best-of-curve. If A2 only shifts the curve down uniformly, that is a speed effect,
not a regularisation effect, and it is a different (and smaller) claim. **Report best-of-curve
and step-2000 separately.** This matters because our best held-out number on record (4.997)
is a best-of-an-overfitting-curve reading, so a pure endpoint comparison is the wrong
instrument for a term whose claimed benefit is anti-overfitting.

### 6.3 Long gate — only if A2 wins

Then, and only then, one seed per rung, no 3-seed requirement:
- **loop-target depth** (§3.4): readout-average (current) vs final-iteration vs k>T teacher-only.
- **α = loss-dropout** sweep at fixed λ(1−α), per LLM-JEPA §5.2 — this is how the term pays
  for its own VRAM cost, and it is the arm that would let us keep the teacher *and* the
  batch size.

### 6.4 Budget, honestly

Per-arm cost is **still unknown** (`docs/protocols/AB-PROTOCOL.md:202-203` withdrew the 1.6 s/step
figure, and the 25.8 s/step replacement has no committed log). The one in-shape datapoint is
1068 ms/step at `small`+`mor`, batch 20, pure CE — but that run printed `fused kda=64/0`, so
it is a floor, not a budget. **4 arms × 3 seeds × 2000 steps, plus P1-P3.** If the teacher's
VRAM cost forces the batch down, the sweep gets slower *and* noisier at once, which is the
worst combination; that is itself an argument for measuring P3 first.

---

## 7. Corrections to `docs/papers/jepa.md` (2026-09-29 pass)

Both of these are load-bearing for the variant taxonomy, and both are wrong in the existing
document. Verified against primary sources today.

| # | claim in `jepa.md` | what the source says | effect |
|---|---|---|---|
| **C1** | §5: *"I-JEPA (2301.08243) — has context-prediction with a target encoder but **no EMA momentum**; using it as the citation would have been wrong. The crate did not."* (also row J-list §0) | I-JEPA §3: *"The parameters of the predictor ϕ and context encoder θ are learned through gradient-based optimization, while the parameters of the target encoder θ̄ are updated via an **exponential moving average** of the context-encoder parameters. The use of an exponential moving average target-encoder has proven essential for training JEAs with Vision Transformers, **we find the same to be true for I-JEPA**."* App. A: *"We use a momentum value of 0.996, and linearly increase this value to 1.0 throughout pretraining."* | **I-JEPA and data2vec 2.0 agree on the EMA teacher.** The real distinction is (i) target = average-pooled **target blocks** vs average of the top-K blocks, and (ii) the mask pattern. The variant taxonomy in the owner's brief ("I-JEPA-style target-encoder … vs ours") rests on a distinction that is not the EMA. |
| **C2** | row **J4** / §2.3: *"data2vec 2.0 does not mask … **BUG — citation misattribution.** A BEiT/I-JEPA-style masked loss sold as data2vec 2.0."* Verdict: the crate "is marketing data2vec 2.0's name over a loss data2vec 2.0 does not have." | The abstract's *"We do not encode masked tokens"* refers to the **teacher** encoding the **unmasked** sample. The method section is explicit: *"we create latent contextualized representations with a teacher model based on **unmasked** training examples which are regressed by a student model whose input is a **masked version** of the sample"*; *"The training task is for the student network to regress these targets based on the masked version of the sample"*; and the multi-mask mechanism reuses one target across **M different masked versions**. | **The naming is right; the reasoning was wrong, and it reached the opposite conclusion.** data2vec 2.0 masks the student input. So the real defect is not "we mask, the paper doesn't" — it is **"we do not mask the input and the paper does"** (§3.3 R2). This *strengthens* the retune case and removes a misattribution. |
| **C3** | row **J5**: instance normalization is the largest delta, marked **SPECULATION** because the paper body was not read | Now **VERIFIED**: *"Before averaging, activations are normalized using instance normalization (Ulyanov et al., 2016)."* NLP/vision tables list `IN → AVG → LN`. Also: the loss is **L2**, and *"This is a simplification compared to the Smooth L1 loss used in Baevski et al. (2022)"* — so `jepa.md`'s "smooth-L1 (Huber, β=2.0)" is the **v1** loss, not 2.0's. | R1 is no longer a hypothesis. Also `jepa.md` mis-transcribed the loss family: 2.0 uses **L2**, ours is **L1**, neither is smooth-L1. |
| **C4** | `docs/protocols/AB-PROTOCOL.md:216` "do the aux losses earn their 26% of the step?" | No derivation anywhere in the repo | **Strike or derive** (AGENTS §1.4). The measured costs are 0.14-0.21% (EMA) and a **halved legal batch** (VRAM). |

---

## 8. References

All fetched 2026-09-30. Version and date as returned by the arXiv API.

| id | version | date | title | used for |
|---|---|---|---|---|
| `2605.07554` | v2 | 2026-05-08 (upd 2026-09-23) | ProteinJEPA: Latent prediction improves protein language model pretraining | **§1.2 — the only scale-matched 3-seed pretraining A/B**; cosine>MSE; shallower-target ablation; JEPA-only collapse; p=0.059 |
| `2604.17228` | v1 | 2026-04-19 | Revisiting Auxiliary Losses for Conditional Depth Routing: An Empirical Study | **§2.3 — the closest published architecture (157.5M, depth-routed decoder-only, 3 seeds) and a negative result** |
| `2509.14252` | v2 | 2025-09-11 (upd 2025-10-07) | LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures | §2.1 claims; §2.2 LD/λ(1−α) guideline and 2× compute; §2.3 metric ablation (Table 3), γ ablation (Table 10), §3.3 "a good NTP is not a good JEPA"; §1.4 pretraining is synthetic + called preliminary by its authors |
| `2212.07525` | v2 | 2022-12-14 (upd 2023-06-15) | data2vec 2.0 (Baevski et al., ICML 2023) | §3.2 instance norm (**verbatim**); §3.3 masked student / unmasked teacher, multi-mask M, momentum ramp; §1.4 the RoBERTa speed claim; L2 loss |
| `2301.08243` | v3 | 2023-01-19 (upd 2023-04-13) | I-JEPA (Assran et al., ICCV 2023) | §7 C1 — **EMA target encoder, momentum 0.996→1.0**; §3.2 representation-space loss is load-bearing (Table 7) |
| `2304.07193` | v2 | 2023-04-14 (upd 2024-02-02) | DINOv2 (Oquab et al.) | §4.2 the KoLeo ablation, Table 3(a) and the §6.1 recipe row |
| `2606.00091` | v1 | 2026-05-24 | DLLM-JEPA (SPIGM @ ICML 2026) | §2.2 — causal substrate demands multi-view data + 2 forwards; 33% FLOP cut |
| `2609.36952` | v1 | 2026-09-29 | ER-JEPA | §1.3 — newest in the family, still fine-tuning-only |
| `2602.22617` | v1 | 2026-02-26 | Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA | §1.3 — synthetic-corpus data-efficiency claim, not transferable |
| `2605.27734` | v1 | 2026-05-26 | Learn from your own latents and not from tokens: A sample-complexity theory | §2.1 — named and down-weighted; PCFG theory, not LM quality |
| `2508.19228` | v2 | 2025-08-26 (upd 2026-02-16) | Predicting the Order of Upcoming Tokens Improves Language Modeling | §2.3 — aux-too-hard is worse than none; also the read on the DSpark decision |
| `2509.24317` | v1 | 2025-09-29 | Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers (SALT) | §3.2(c) — frozen teacher suffices in video; student quality robust to teacher quality. **Video-only; named as the cheapest future variant, not adopted** |
| `2511.08544` | v3 | 2025-11-11 (upd 2025-11-14) | LeJEPA: Provable and Scalable SSL Without the Heuristics (SIGReg) | §3.2(c) — the predictor-less/no-teacher arm; our `lejepa_loss` is already written and uncalled |
| `2606.01443` | v1 | 2026-05-31 | UR-JEPA: Uniform Rectifiability as a Regularizer for JEPAs | §3.2(c) — the isotropic-Gaussian target is in tension with the manifold hypothesis; vision |
| `2602.03604` | v3 | 2026-02-03 (upd 2026-04-08) | A Lightweight Library for Energy-Based JEPAs | §3.2(d) — the EBM variant; rejected for this decision, queue separately |

**Repo sources cited:** `crates/dormouse-core/src/{model.rs:86-106,150-156}`, `aux.rs:25,29,176-179,224-232`;
`vendor/burn-fused/crates/burn-jepa/src/{losses.rs:11-20,63-97, mask.rs:9-25, predictor.rs}`;
`benches/history.tsv:51-56`; `~/logs/ab8m_ab8m_iter4.log:10-16`; `~/logs/train_nokda.log:91`;
`docs/protocols/AB-PROTOCOL.md:185-230`; `docs/archive/audit-2026-09-25.md:29,62,64,87`; `docs/papers/jepa.md`.

---

## 9. What would change this verdict

- A real full-text search (Semantic Scholar / Connected Papers) finding a KoLeo-for-LM
  study → §4 changes; a 10-point estimator still does not, but the "no evidence" claim dies.
- A JEPA arm on an **autoregressive** model **below 100M params** on **natural text** with a
  **loss** metric, 3 seeds, random init → ProteinJEPA's p=0.059 row is the weakest load-bearing
  number here and it is the one most likely to move.
- P2 (S1) showing the masked and unmasked JEPA losses *differ materially* → §3.1's structural
  argument weakens, because it would mean the loss-position mask is doing something the call
  graph does not obviously support.
- A `--timers` reading putting the teacher forward **outside** `fwd` → the cost argument in
  §2.2 collapses to ~0.2% and the VRAM argument becomes the *only* one left, which is still
  enough at 16 GB but changes the recommended batch size.

---

## Кросс-проверка вторым исследователем (alphaxiv-библиотека, 2026-09-30 ~19:20)

Независимый агент по той же постановке вопроса сошёлся с этим вердиктом в главном
и добавил три предмета. Сходимость двух независимых проходов — само по себе
свидетельство, что вердикт RETUNE/вес-0 устойчив.

**Сошлось:**
- Генеративный CE остаётся обязательно (LLM-JEPA 2509.14252v2 p.17: полная
  абляция генеративного лосса → модель выдаёт пустой результат).
- LLM-JEPA — миллиардный масштаб + **парные представления** (парафраз,
  описание→код, вопрос→ответ); у байтового корпуса таких пар нет, перенос
  не доказан.
- Решает matched-compute A/B по BPB и задачам, а не по снижению JEPA-loss —
  тот же инструмент, что наш §A/B (3 сида, один батч, окно напечатано).

**Новое (добавлено в досье):**
1. **JEPA Paradox в языке** — arXiv:2607.23531v1 (2026-07): предсказание
   MSE-вектором произвольного продолжения тянет цель к условному среднему
   допустимых продолжений. Прямое теоретическое обоснование того, почему
   наш текущий target (T-усреднённый readout произвольного будущего) — слабая
   постановка, и почему ретун должен либо нормировать цель, либо брать
   парные/структурированные вью.
2. **BLT** (arXiv:2412.09871) — иерархическая байтовая архитектура
   (локальный энкодер → патчи → ядро → локальный декодер). Это НЕ JEPA
   (латентные патчи — ради частоты запуска ядра, не ради абстракции), и это
   technology-REPLACE, а не aux-лосс: отдельная entropy-модель в их конфиге
   весит 100M — больше нашего всего бюджета. Регистрируется как отдельный
   дальний кран, не как JEPA-решение.
3. **Приоритет для маленькой модели: данные + дистилляция от сильного
   учителя**, а не сложность aux — подтверждает docs/architecture/post-training.md
   (SFT→RLVR→distill), который у нас уже и есть план.

**Ответы на вопросы агента (наши факты):** GPU = RTX 5060 Ti 16 GB + 64 GB
RAM; окна — ночь; цель = код-домен корпус → SFT → RLVR (docs/architecture/post-training.md),
русский чат + код. Модель 9.2M (пресет small) — значит «40–60M как
эксперимент» агента — над нашими 9M, но BLT-иерархия и парные вью — решения
не сегодняшнего дня: сначала контроль на 9M, потом очередь A/B, и только
потом — архитектурные замены.

### Корректировка от автора cross-check (2026-09-30, вечер) — принимается целиком

1. **Сходимость двух обзоров — согласованный вердикт, не независимое
   экспериментальное подтверждение.** Оба прохода опирались на одну и ту же
   публикационную базу; доказательство у нас одно — matched-compute A/B на
   9.2M, и оно будет первым ПРЯМЫМ свидетельством для этой модели. Выше по
   тексту формулировку «само по себе свидетельство» читать с этой поправкой.
2. **Нормировка target не решает условное среднее.** R1 чинит масштаб и
   устойчивость лосса; несколько семантически разных допустимых продолжений
   остаются несколькими продолжениями. Парные вью меняют постановку задачи,
   нормировка — геометрию ошибки. И если пар в корпусе нет, ретун — это ещё и
   ИЗМЕНЕНИЕ ДАННЫХ (построение пар), не только коэффициента aux. Для очереди
   это значит: рука «JEPA-ретун» должна либо прийти с планом построения пар,
   либо честно называться «нормированный target», без претензии на решение
   Paradox.
3. **«CE обязателен» — инженерное решение этого проекта,** не закон природы:
   пустой вывод без CE — результат одной абляции LLM-JEPA, а не доказательство
   невозможности генерации с другим objective. Для нашей AR байтовой модели
   оснований отключать CE нет — но формулировка в досье смягчена.
