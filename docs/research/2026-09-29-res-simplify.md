# What can be DELETED from a modern LLM block — evidence for removal being free

**Question.** In a modern LLM block, which components can be deleted with no measurable
loss of capability? Not which is best — which is dead weight.
**Subject:** the six arms dormouse has accumulated and never A/B'd (adaptive depth, MoR
routing, hashed n-gram memory, TSCT spectral factors, JEPA + DSpark auxiliary heads,
Gated Residual).
**Method:** arXiv API title/field searches + full-text reads of 15 papers; 1 leaderboard
README; 1 read-only pass over `crates/dormouse-core/src/aux.rs`. No GPU, no cargo, no
other file touched. `websearch` returned zero results for every query, so **discovery ran
entirely through the arXiv API, arXiv HTML, direct HTTP and the Semantic Scholar graph
API** — a real limitation on coverage, noted in §7.
**Fetch date:** 2026-09-29. **Tree:** `08ee0a2`, worktree `/home/sehaxe/dormouse-wt/res-simplify`.

---

## 1. Verdict (three sentences)

Two of our arms have published evidence of being removable, and both are removal
candidates **because their own source papers report the removal or the null**: the
**MoR token-level router** is negative at 135M (−vs vanilla, the authors' own words),
+0.8 few-shot points at 360M-base, and **negative again at 1.7B** where the routed model
sees 30% more data and still loses to vanilla [2507.10524v3]; and the **DSpark draft
head** is *bits-per-byte-neutral in DeepSeek's own MTP ablation* (0.729→0.729 at 15.7B,
0.658→0.657 at 228.7B) while moving downstream benchmarks by up to +9.2 points — i.e.
the only metric our A/B protocol measures is the one MTP does not move [2412.19437v2].
A third, **JEPA**, is defensible-to-delete on cost grounds (an EMA teacher is a second
full forward per step) but I found **no published ablation of a data2vec-style latent
objective removed from a jointly-trained LM at any scale** — that is an absence of
evidence, not evidence of absence. The three arms with **no** published evidence of
free removal are the hashed n-gram memory, the TSCT factors, and Gated Residual, and for
the first two the best available sources are respectively a single-author preprint whose
own numbers are inside their own standard deviation, and our own unpublished invention.

---

## 2. Deletion-candidate table

Transferability is stated per row. "**BPB**" = our metric (held-out bits per byte);
"**downstream**" = MMLU/HumanEval/etc. A row only transfers to us on the metric it was
measured on.

| # | Candidate (our arm) | Evidence that removal is free | Scale measured | Metric it was measured on | Transferability to a 9.2 M byte-level model |
|---|---|---|---|---|---|
| 1 | **MoR token-level router** (the `use_mor` arm; the per-iteration controller blend) | 2507.10524v3 §3.2: "**it underperforms the vanilla model at the smallest model size (135M)**—likely due to a recursive capacity bottleneck". App. Table 7 @1.7B/68.5e18 FLOPs: Vanilla 1.61 B, 20 B tok, few-shot avg **48.9** vs MoR Expert+Cache Nr=2 (0.87 B, **26 B tok**) **48.4** and Nr=3 (0.67 B, 27 B tok) **46.7**. Authors: the vanilla "performs slightly better… it might also signal that the current architecture design of MoR is **not suitable for scaling**." At 360M-base, the *positive* number is MoR 43.1 vs vanilla 42.3 = **+0.8 few-shot points** | 135 M / 360 M / 730 M / 1.7 B, FineWeb-Edu, 49 K vocab, 2 K ctx | validation NLL + 6-benchmark few-shot accuracy | **Strongest row in the table.** Negative at the two ends of the scale range and +0.8 pp in the middle, on a metric we do not measure. At 9.2 M — 15× *below* the paper's own smallest model and inside the regime where it says MoR loses — the sign of the effect is, if anything, more favourable to deletion. **Caveat that cuts the other way:** MoR's rows are never token-matched to their Recursive baselines (MoR 27 B tok vs Recursive 20 B at equal FLOPs, because recursion-wise caching frees FLOPs), so **the paper contains no row that isolates the router**; the isolated router effect is bounded by the 360M MoR-vs-Recursive gap, NLL 2.7511 vs 2.8079. |
| 2 | **Learned gate / halting head** (the read-out-and-halt machinery; the reason `use_mor` needs a router at all) | 2607.20519v1: "**simple post-hoc confidence readouts often match or outperform learned linear and MLP gates**"; "the failure… stems mainly from the trajectory induced by joint gate training rather than from limited gate expressivity"; on real checkpoints, "pretrained ponder gates are **competitive but not uniformly Pareto-optimal**". 2607.14427v1 (135 M, FineWeb-Edu): a **training-free** convergence exit (halt when successive outputs stop moving) matches uniform-depth-8 quality (3.189) at **4.94 average loops — a 38% depth reduction** — and beats it at the shallow end (3.231 vs 3.272 at avg depth 3), within 0.002–0.003 nats at avg depth 4–7. Abstract: "reading it outperforms learning it" | 135 M + 31.8 M (TinyStories), FineWeb-Edu, randomized recursion; 2607.20519 additionally at Ouro-1.4B/2.6B | validation loss (nats) vs average loop count | **Best scale match in this table** — 135 M is 15× ours and the *same random-depth* training regime we run. The 0.002–0.003 nat margin at matched depth is inside the 0.002–0.005 BPB paired-eval resolution `docs/AB-PROTOCOL.md` estimates for us, i.e. the training-free rule is free *by our instrument's resolution*. **Limits:** S0 in 2607.14427 has 2 of 3 planned seeds complete (paper's own note); the comparison is inference-depth policy, not a pretraining A/B. |
| 3 | **DSpark draft head** (aux loss, default 0.1) | DeepSeek-V3 Table 4, the closest true ablation of a next-K draft objective: **Pile-test BPB 0.729 → 0.729** (15.7 B MoE, 1.33 T tok) and **0.658 → 0.657** (228.7 B MoE, 540 B tok) — i.e. **0.000 and 0.001 BPB, on our exact metric** — while BBH 39.0→41.4, MMLU 50.0→53.3, HumanEval 20.7→26.8, GSM8K 25.4→31.4 move. GrowMTP 2609.16648v1: "**policy quality stays on par** with the autoregressive baseline on all six benchmarks, with Mean@16 and Pass@4 differing by at most 2.29 points" | 15.7 B / 228.7 B MoE, 1.33 T / 540 B tok; Qwen3-4B, 500 RL steps | **BPB (DeepSeek) — the transfer is on metric**; pass@1/EM (GrowMTP) | **Transfers on metric, and the structural difference makes the case for us weaker, not stronger.** DeepSeek's MTP module **shares the embedding and the output head with the main model** (2412.19437v2 §2.4, verbatim), so its loss reshapes the model. Ours does not: `crates/dormouse-core/src/aux.rs:234-236` states the DSpark loss corrects **frozen (detached) backbone logits** and reaches the backbone only "`hidden` carries gradients into the backbone". A head that cannot touch the output head is structurally less able to move BPB than one that can. **Counter-evidence to weigh:** modded-nanogpt record #53 (2025-12-22) "Multi-token prediction, untie embed/lm_head at 2/3 training" was *accepted* at 124 M — but that leaderboard's metric is wall-clock to a fixed val-loss target (§6), so acceptance means "not harmful and faster", not "loss-neutral". |
| 4 | **JEPA / data2vec-style masked-latent head** (default 0.05) + KoLeo | **NONE FOUND.** The nearest controlled study, 2604.17228v1, is a *different* JEPA (a routing-gate predictor, no EMA teacher) and it reports: ablation A3, "**simultaneously removing util/rank improves best/average LM and threshold-hit speed in 3/3 seeds for both gate architectures**", and the JEPA gate's advantage over a plain MLP gate "disappears". 2509.14252v2 (LLM-JEPA) is the only LM-scale JEPA, and its finding is *neutrality*, not benefit: "the next token prediction capability is **not hindered** by the presence of the JEPA term" — it reports gains on *task* metrics, never on next-token loss | 157.5 M (controller-only, 50% budget, 3 seeds); Llama-3.2-1B-Instruct **fine-tuning** (not pretraining) | LM loss (routing study); EM/accuracy on NL-RX/GSM8K/Spider (LLM-JEPA) | **Weak, and I say so plainly.** 2604.17228's own scope note: conclusions are "strictly bounded to the ~157.5 M controller-only 50%-budget regime" and "**not extrapolated to larger scales, joint backbone training**" — which is the opposite of our setting (we jointly train the backbone with a real EMA teacher). LLM-JEPA is fine-tuning, not pretraining, and costs "two forward passes instead of three… still a substantial slowdown". **Verdict: unevidenced in both directions.** |
| 5 | **Auxiliary losses generally** (the pattern, not one arm) | DeepSeek-V3 Table 5, load-balancing aux loss removed at both scales: BPB **0.727 → 0.724** (15.7 B) and **0.656 → 0.652** (228.7 B) — removal *improved* the pretraining loss. BBH 37.3→39.3 and 66.7→67.9; HumanEval 22.0→22.6 and 40.2→46.3. Counter-example in the same table: MMLU at 228.7 B **drops** 68.3 → 67.2 | 15.7 B / 228.7 B MoE, 1.33 T / 578 B tok | BPB + downstream | **Transfers in sign, not in mechanism.** This is a MoE load-balancing term, not a representation term; the useful generalisation is narrow and real: *an auxiliary loss is a term competing with CE for the same gradients, and DeepSeek's own 228.7 B run got a better pretraining loss by deleting one.* It says nothing about a JEPA head, but it is the cleanest published data point that "delete the aux loss and the loss goes down". |
| 6 | **Hashed n-gram memory (Engram)** | **NONE.** The direct Engram study, 2601.16531v2, reports its own best configuration as **not significant**: best val loss 4.4799 "…its advantage over Hash-500K is only 0.001, far smaller than the measurement standard deviation (0.008–0.012), and is **not statistically significant**". Slot curve 300 K 4.4825 / 500 K 4.4809 / 800 K 4.4961 — **flat to within noise across the whole usable range** | ~185 M GPT-2 backbone inflated to 313.6 M (128,815 vocab) + 128 M shared Engram embeddings; single author, no venue | validation loss | **The load-bearing point is the absence.** Not one source measures a hashed n-gram memory arm against its own removal on a loss metric. The strong-looking FOR source is 2412.09764v2 (Memory Layers at Scale, Meta, 1.3 B base / 128 B memory params / 1 T tokens), which reports the memory model "outperform dense models with more than twice the computation budget" — but that is **factual-QA-weighted** ("gains are especially pronounced for factual tasks") and uses *learned product-quantized keys*, not FNV hashing, so it is a different mechanism. **Our arm's supporting literature is weaker than the arm.** |
| 7 | **TSCT spectral factors + per-step polar retraction** | **NONE — this is ours; there is no TSCT paper and I did not go looking for a surrogate.** The cost is the argument, not the evidence: `AGENTS.md` §3.1 measures `retr` at **52.8 / 53.3 / 64.6 ms** at batch 8 / 16 / 32 against steps of 244 → 826 ms — i.e. **22% of a step at batch 8, 7.8% at batch 32**, and it does not amortise (8× the data for 1.2× the retraction) | ours, 7.5 M–9.2 M, measured 2026-09-29 | step time (not quality) | **Not a deletion finding; a cost finding.** No published evidence in either direction, and none will arrive. If the owner deletes it, the justification is the 22% of a step, and it must be stated as a cost argument, not a quality one. |
| 8 | **Gated Residual** | **NONE for removal; there is evidence *against* removing it**, and the project already holds it: the Qwen3.8-Flash-Next report's −0.026 loss at 276 B tokens (`AGENTS.md` §3.5). `use_gr = false` in all eight `configs/*.toml`, so no run of ours has ever measured it either way | 125 B MoE / 6 B active, 276 B tokens (vendor report) | pretraining loss | **A transposition, not a transfer.** Our placement is one GR per loop *iteration* of a weight-shared block; the report's is one per attention and MLP sublayer of a 56-sublayer stack. `AGENTS.md` §3.5 records that the implementation did not match Eq. 31/32/33 until 2026-09-28. Leave it off; do not spend an A/B on deleting something that is already off and has evidence for. |

---

## 3. What must NOT be deleted

Equally load-bearing, and mostly *not* the same shape as our arms.

| Component | Evidence that removal costs capability | Scale | What it means for us |
|---|---|---|---|
| **Whole layers beyond ~50% of depth** | 2403.17887v2: after a flat MMLU/BoolQ region, "performances are quite robust until 20%-55% pruning fractions… at which point they transition to **random guessing**" — Llama-2 45–55%, Mistral-7B 35%, Phi-2 25%, Qwen family 20%. The transition is **sharp**, not gradual | 2.7 B–70 B, 32–80 layers, healed with QLoRA on 164 M/328 M C4 tokens | Depth is not free to cut. But note *which* metric: their own §4.2 finds the healed C4 **loss rises "slowly and linearly"** across exactly the fractions where MMLU collapses to random — i.e. "no loss" here is a property of an easy multiple-choice metric, not of the objective. **We measure the objective.** |
| **Generative capability under layer removal** | 2403.03853v3's own unexplained observation: "the negative effect of layer removal is more significant on **generative** tasks compared to multiple-choice tasks. When we remove 25% layers from Llama2-7B or Baichuan2-7B, the performance in generative tasks such as **XSum and C3 decreases to nearly zero**". Table 2 at 24.2% removal: Baichuan2-7B XSum 20.82 → **0.04**; Llama2-13B XSum 23.45 → **0.67**, while MMLU only falls 53.87 → 45.77 and 55.00 → 52.11 | 7 B / 13 B, no retraining | **The single most important warning in this document.** Every "we removed 25% of the layers and it barely moved" claim in the layer-pruning literature is measured on multiple-choice benchmarks. On a generative metric the same 25% removal is annihilation. Our metric (held-out BPB on a fixed window) is closer to the generative side than to MMLU. |
| **Normalization — not deletable, only replaceable** | 2503.10622v2 (CVPR 2025, Zhu/Chen/He/LeCun/Liu): LLaMA 7 B/13 B/34 B/70 B with **DyT** replacing RMSNorm — "DyT performs **on par** with RMSNorm across all four model sizes", loss curves "closely aligned". The claim is a *drop-in replacement by a scalar function*, not a deletion | 7 B–70 B, pretraining | No published source found showing a norm can simply be **removed**. If the owner wants fewer norm ops, DyT is the cited route; it does not save the parameter or the op count. |
| **Routing's auxiliary/balancing losses** (a *mechanism* inside a deletion candidate) | 2507.10524v3: the expert-choice router needs a causality patch or it fails — aux-router with tanh and α=1.0 → **66.7% dead tokens**; un-normalised configs → **NaN NLL**. Final recipe needs aux coeff 0.001 | 360 M, 3 recursions | Not a reason to keep the router — a reason that if anyone *does* keep it, it is not free. Cuts both ways in the same paper as row 1. |
| **The randomization of depth, if you want depth robustness** | 2607.14427v1: fixed-depth-4 control "is sharply peaked at its training depth: loss 1.97 at [its own depth], 6.01 at [another], 2.50 at [another]", while the randomized-depth model is "flat and low across depths (1.886, 1.883, 1.883)" — and "slightly better than the control **even at the control's own specialized depth**" | 31.8 M TinyStories + 135 M FineWeb-Edu | This is the one adaptive-depth finding that argues *for* keeping something. It argues for keeping **randomized depth** (our `--rand-depth`, a pure function of step index, zero parameters) and against keeping a **learned router**. Those are different arms and the project has them conflated in its own queue. |

---

## 4. Rejected candidates (searched, not used)

| Candidate | Why rejected |
|---|---|
| DeepSeek-V3-Lite, Mini, Qwen small, OLMo | No A/B their own removal was found; the queue ran out of budget for per-model reports |
| "Byte-level / ByT5 / character models need more arms" | ByT5's finding is about *data efficiency* at 4× less text, not component deletion, and I did not re-resolve its arXiv id in this pass — so I am not citing it here. The project's own `research/2026-09-26-small-lm-dynamics.md` already carries it with a verified id. Not a deletion source |
| 2604.24938 "Rethinking Layer Redundancy: Calibration Matters More Than Search" | Real and relevant (layer-pruning search is not where the win is; calibration is) but it re-prunes *existing* 7 B-class models. Same post-hoc-pruning regime as row 1 of §3, and it inherits the generative-task caveat |
| TALE 2510.22767, LayerChop 2305.14864, Reassessing Layer Pruning 2411.15558, Just CHOP | Four more post-hoc layer-pruning papers. All require fine-tuning/healing to recover, all evaluated on the same QA metrics. Adding four more rows of the same evidence class would not change a single conclusion |
| 2605.30202 "A Dual-Path Architecture" | Correct and on-topic ("at fixed FLOPs a looped model has strictly less capacity than a baseline transformer") but it is a *proposal* of a new block, not an ablation of an existing one |
| 2606.06574 "Skip a Layer or Loop It?" | Training-free inference-time program-of-layers; "substantially shorter program executions can achieve the same or better accuracy" is an inference claim. We A/B training, not inference |
| The whole flat-minima / loss-landscape family (2001.01678, 2207.02628, 2403.00574) | These are about generalization under SGD noise, not about "can a smaller model match". Searched and set aside — see §5 for the framing that actually answers the owner's 99% question |

---

## 5. The "99% of intelligence" question, answered in the only terms that apply

The owner wants a smaller architecture at ~99% of the capability. The literature answer is
**not** about flatness. It is about **which metric the 1% is measured on**, and the two
deepest sources here agree:

1. **DeepSeek-V3's MTP ablation is the cleanest instance of the split.** On BPB, the
   next-K objective moves the objective by **0.000–0.001**. On HumanEval it moves it by
   **+6.1 to +9.2 points**. Same model, same data, same compute. A "99% of intelligence"
   target is therefore **unanswerable until the metric is named**: on our BPB, deleting
   MTP costs 0.0%; on HumanEval it costs 9 points. Our A/B protocol measures BPB, so it
   will report `use_dspark` as worthless — and that result would be **correct and
   misleading at the same time**.
2. **Our own capacity position.** By Chinchilla's ≈20 tokens/param (the figure the
   project already carries in `research/2026-09-26-small-lm-dynamics.md`), 9.2 M params
   wants ≈184 M tokens ≈ **184 MB** of single-pass data. At 19,500 steps × 5,120 B the
   best recorded run had seen ≈**100 MB**, i.e. ~54% of the compute-optimal budget —
   and its curve had already regressed (4.997 at step 6,500 → 5.450 at 19,500).
   *SPECULATION: this is arithmetic from two sourced constants, not a measurement, and
   the run's own overfitting curve is the direct evidence that we are at or past the
   optimum — which is the argument for simplifying capacity, and for nothing else.*

**The honest reading:** "99% of the intelligence" is a benchmark-selection decision, not
a scientific one. Nothing in this document licenses deleting a component whose value is
visible only on a benchmark we do not run. That is the whole disagreement between rows 3
and 6 of §2, and it is the owner's call, not the literature's.

---

## 6. What modded-nanogpt contributes, and its one hard limitation

The closest thing to our scale in this document is not a paper: it is a public ablation
leaderboard at **GPT-2 small scale (124 M non-embedding)**, 8×H100, target ≤3.28 mean val
loss on FineWeb — "3.28 was selected to match Andrej Karpathy's GPT-2 (small)
reproduction" (`github.com/KellerJordan/modded-nanogpt`, README fetched 2026-09-29; track
2 is 350 M against llm.c's 2.92). Accepted records include outright **deletions**:
"**Drop first MLP layer**" (#30, 2025-09-05), "**Drop first attn layer**" (#35,
2025-09-21), "Sparsify value embeddings, improve rotary embeddings, **drop an attn
layer**" (#17, 2024-12-17), "**removed post attention lambdas**" (#74, 2026-02-16). It
also records three explicit **"Backout"** entries (#40 2025-10-04, #44 2025-11-16, track
2 #15 2025-10-04) — a leaderboard that publishes its reverts is more trustworthy than one
that does not.

Two things it hands us and one it does not:

- **It hands us the variance scale.** Rule 2: "Due to inter-run variance, submissions
  must provide enough run logs to attain a statistical significance level of **p<0.01**
  that their mean val loss is ≤3.28." At 124 M, one run cannot settle a val-loss
  question. Discretionary reason 2: "The current record is intentionally kept roughly
  **0.001-0.002 loss** below 3.28." At 124 M the material band is 0.001–0.002 — the
  same order as our own unverified 0.002–0.005 paired-eval resolution.
- **It hands us two of our own arms as *accepted* entries**: "**Multi-token
  prediction**, untie embed/lm_head at 2/3 training" (#53, 2025-12-22) and "**Bigram Hash
  Embedding**" (#62, 2026-01-19). So at 124 M, both of the arms §2 rows 3 and 6 point at
  were tried and kept.
- **It does not hand us a loss delta, and this is why I did not put it in the table.**
  The leaderboard's metric is **wall-clock time to a fixed val-loss target**. An accepted
  deletion means "did not prevent reaching 3.28 and ran faster" — it is a *survivability*
  result, not a *neutrality* result. Reading it as evidence that removing an attention
  layer is free would be exactly the kind of over-read this project has retracted before.

---

## 7. The minimum experiment that would settle it

**The cheapest arm to retire is the DSpark draft head**, on three counts: the removal is
predicted by the only source that measured the same thing on our metric (0.000–0.001 BPB);
our head is structurally less able to move the model than DeepSeek's (`aux.rs:234-236`,
detached logits); and it deletes a per-step K-step draft forward plus three loss terms.

**Design — the cheapest form that is not a lie:**

- **Two arms, not three**: (A) control = current recipe with `--dspark-weight 0`, all
  else identical; (B) current recipe as shipped. `--set use_dspark=false` or the existing
  weight flag — the arm is a **weight zero**, not a config deletion, so nothing else in
  the graph moves.
- **Fix `batch` and `seq_len`** and hold them across both arms, or the eval windows differ
  and neither number is comparable (`AGENTS.md` §2.6: the window is
  `eval_batches × batch × seq_len`; 20,480 B at batch 2, 102,400 B at batch 10).
- **3 seeds**, per `AGENTS.md` §1.2 — and note the standing caveat that after `4b42b6d`
  the seed governs init but ~4% of the model is still process entropy, so three seeds
  are *nearly* the same three runs.
- **Read two counter fields before believing the number**: `fused kda=<f>/<b>` must show
  `b>0`, and `engram=<rows>/<arms>` must be non-zero if `use_engram` is on. A control that
  fails either is not a control (`AGENTS.md` §3.2/§3.3).
- **Read `--retract-every`** into the cost accounting, not the quality one.

**Cost — two bounds, and the gap between them is the project's open question:**

| basis | ms/step | 2 k steps × 3 seeds | both arms |
|---|---|---|---|
| measured warm step, 9.2 M, batch 8×512, depth 2, fp32, `--no-kda --no-engram` (`AGENTS.md` §3.1, 2026-09-29) | 245 | **24.5 min** | **≈0.8 GPU-h** |
| `docs/AB-PROTOCOL.md` pricing (1.6 s/step) — derived on a run **whose attention backward did not execute** | 1600 | 160 min | ≈5.4 GPU-h |

**SPECULATION:** the true cost sits in that range and I cannot narrow it, because
`AGENTS.md` §3.3 states the cost of one A/B arm is *currently unknown* — the 25.8 s/step
batch-8 attention-backward figure that would replace the pricing has no committed log and
no `benches/history.tsv` row. **Budget the 0.8 GPU-h figure and treat it as a lower
bound, not a plan.** If the arm does not resolve at 2 k steps, the correct reading is
"unresolved", not "tie" — a tie deletes the mechanism (`AGENTS.md` §1.2), and an
unresolved comparison does not.

**A cheaper first move, at zero GPU-h:** log the DSpark loss value and its gradient norm
per step for 200 steps of the existing recipe. If the aux gradient's share of the total
backbone gradient is already inside the 0.002–0.005 BPB resolution band, the A/B is
predicted to be unresolvable *before it is run*, and that is worth knowing. This is a
SPECULATION-grade heuristic, not a substitute for the A/B.

---

## 8. VERIFIED (sourced) vs SPECULATION (my inference)

**VERIFIED — read from the primary source, quoted or tabled above:**
- MoR underperforms vanilla at 135 M; the authors' "not suitable for scaling" reading at
  1.7 B; the exact Table 3 and Table 7 few-shot averages and token counts [2507.10524v3].
- DeepSeek-V3 Table 4: MTP changes Pile-test BPB by 0.000 (15.7 B) and 0.001 (228.7 B)
  while moving BBH/MMLU/HumanEval/GSM8K; Table 5: removing the load-balancing aux loss
  improved BPB 0.727→0.724 and 0.656→0.652; MTP shares embedding and output head with the
  main model [2412.19437v2].
- Gromov: flat region to 20–55% then sharp transition to random guessing; the healed C4
  loss is continuous and "slowly and linearly" increasing across that same region
  [2403.17887v2].
- ShortGPT: Llama2-13B MMLU 55.00→52.11 at 24.6% removal; Baichuan2-7B XSum 20.82→0.04
  and Llama2-13B XSum 23.45→0.67 at ~24% removal; "redundancy in depth than in width"
  [2403.03853v3].
- A training-free convergence exit matches uniform-depth-8 quality (3.189) at 4.94 average
  loops (38% reduction) on a 135 M FineWeb-Edu model; the fixed-depth-4 control is peaked
  and degrades off its training depth [2607.14427v1].
- Post-hoc confidence readouts match or beat learned gates [2607.20519v1].
- Removing util/rank aux losses improved LM in 3/3 seeds and erased the JEPA gate's
  advantage, at 157.5 M, controller-only, 50% budget [2604.17228v1].
- GrowMTP: policy quality on par with no-draft-head autoregressive decoding on all six
  benchmarks, max difference 2.29 points, Qwen3-4B, 500 RL steps [2609.16648v1].
- LLM-JEPA: the JEPA term does not hinder next-token prediction; it costs an extra forward
  pass; it is a fine-tuning result at 1 B [2509.14252v2].
- DyT matches RMSNorm on LLaMA 7 B–70 B — a replacement, not a deletion [2503.10622v2].
- Engram: the best configuration's advantage is 0.001 against a 0.008–0.012 standard
  deviation and "not statistically significant"; ~185 M backbone; single author
  [2601.16531v2].
- Memory Layers at Scale: 1.3 B base, up to 128 B memory params, 1 T tokens, beats dense
  models with >2× the compute budget, gains concentrated on factual tasks
  [2412.09764v2].
- modded-nanogpt: the four accepted deletion records, the three "Backout" records, the
  p<0.01 run-log rule, the 0.001–0.002 material-loss band, and the two accepted records
  matching our MTP and hashed-n-gram arms.
- Our own DSpark loss structure: detached backbone logits, gradients reaching the
  backbone only through hidden states (`crates/dormouse-core/src/aux.rs:234-236, 305-306`).

**SPECULATION — my inference, flagged as such:**
- That the 9.2 M regime is *more* deletion-friendly than 135 M for the router. Basis: the
  sign of MoR's effect at 135 M, plus the compute-budget arithmetic. No source measures
  routing below 135 M.
- That our token budget sits at ~54% of Chinchilla-optimal. Arithmetic from two sourced
  constants, not a measurement.
- The 0.8 GPU-h lower bound for the DSpark A/B, and that a gradient-share log could
  predict unresolvability before spending the GPU-h. Heuristic, not evidence.

**UNRESOLVED — I could not answer these from the sources, and I am not going to guess:**
- Whether *any* published study removes a data2vec/JEPA objective from a **jointly-trained**
  LM and reports the effect on next-token loss. Searched; found none.
- Whether *any* published study ablates a hashed n-gram memory arm against its own removal
  on a loss metric. Searched; found none. (The FOR evidence is QA-weighted and uses
  learned keys, not hashing.)
- Whether MoR's router helps *at matched tokens*. Its own paper has no such row.
- The GPU-hour cost of one A/B arm with a working attention backward. Open in
  `AGENTS.md` §3.3 before this document existed.

## 9. Was the search thorough?

**Thorough on depth, incomplete on breadth, and the gap is a tool failure you should
know about.** `websearch` returned "No search results found" for every query I tried,
including single-word ones, so I never had a general web index: discovery ran through the
arXiv API only (title, abstract-field and ID-list queries), plus direct HTTP to
`arxiv.org/html`, `raw.githubusercontent.com` and the Semantic Scholar graph API. That
reaches essentially all of cs.CL and the Semantic Scholar citation graph, and it found
every paper above — but it cannot see workshop papers, non-arXiv venue write-ups, blogs,
or anything behind a paywall, and Semantic Scholar returned 100 of MoR's citations
(truncated, not complete). I read 15 papers in full text, not abstracts, and every number
in §2 and §3 is quoted or tabulated from a document I opened. **Two arXiv IDs I recalled
from memory before checking turned out to be wrong** (2403.00577 is a chemistry paper and
2508.06201 is a QCD paper; the real deeper-layers paper is 2403.17887) — which is the
concrete reason every ID in this document was resolved through the API rather than
trusted, and the reason I would not trust a shorter version of this report.
