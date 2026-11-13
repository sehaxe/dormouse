# Outcome RL, agentic midtraining, training-free distillation: what three 2026 papers actually establish for a 100M byte-level model on one 16 GB card

Scope: three papers read in full (HTML v1, main text + appendices), assessed against
dormouse's plan **SFT → RLVR → distillation from a 4B teacher**. Every claim below
carries a section/table/figure reference. Two sections are kept strictly apart:
**VERIFIED** (stated by the paper or derived algebraically from its equations) and
**SPECULATION** (my inference, explicitly labelled). The confidence rating applies
to the verified part only.

Sources (all fetched 2026-10-28, HTML v1):

- P1 — Shan Yang, *What Outcome RL Buys: Finishing, Reliability, Rarely Reach*, arXiv:2610.34894v1 [cs.CL], 26 Oct 2026. Single author, individual researcher. Code + released tables: github.com/shanyang-me/outcome-rl-buys.
- P2 — Singhal, Jennings, Shum, Bukharin, Bhardwaj et al. (NVIDIA), *Priming Base Models for Agentic Post-Training: An Open Midtraining Recipe*, arXiv:2610.22019v1 [cs.CL], 15 Oct 2026.
- P3 — Yang, Fan, Zhu, Ma, Liu (UIUC / Tsinghua), *Training-Free On-Policy Distillation*, arXiv:2610.22236v1 [cs.LG], 15 Oct 2026.

Scale note used throughout: the brief says 100M params; the presets measured in this
repo today are 7.2M (`nano`) and 9.2M (`small`) (`AGENTS.md` §3.1, `tests/preset_exec.rs`).
I therefore treat 100M as the design point and flag where the two diverge. Throughput
assumption: 1.9 GB/day with vocab 256 means 1 byte = 1 token, i.e. ~1.9e9 tokens/day
(~22k tok/s), consistent with the measured 16.8–19.8K tok/s batch ladder
(`AGENTS.md` §3.1).

---

## TL;DR

1. **P1 kills "RLVR will teach the model something new" at our scale.** At 1.5B–4B the
   net new capability is −0.007 to +0.007 pass@1 — indistinguishable from zero
   (Table 4). What outcome RL buys is *finishing* (82% of the change at tight budgets,
   Table 1) and *reliability* where the base already finishes (+0.032, Table 2). Both
   are cheaper to buy with SFT and a fixed answer format.
2. **P2's scale is unreachable for us (500B tokens ≈ 263 days) but two of its ablations
   are free and one is worth ~10 benchmark points**: mask tool responses from the loss
   (27.5 → 37.8, Table 9), keep ~20% pretraining replay, keep the midtraining LR
   moderate (MATH-500 79.7 → 61.7 at 5e-5, Table 11), and put agent traces in the mix
   because natural code alone gains <1 point after SFT (Table 2).
3. **P3 does not apply to our distillation plan.** Its closed form needs ≥2 teachers
   from one shared base and explicitly defers cross-architecture teachers to training
   (§4.6); with one teacher Eq. (2) collapses to exactly that teacher's weights. What
   transfers is the *filter*: 19 looping rollouts out of 256 moved AIME25 from 34.2 to
   40.0 (Table 5) — token-weighted statistics are dominated by repetition loops.
4. **None of the three papers has a single data point below 1.5B parameters.** Our
   design point is 15× below the smallest model in any of them. Everything here is
   extrapolation at the low end, and I say so per row.

---

## 1. Verdict per paper

### P1 — "What Outcome RL Buys" → **changes our plan: it deletes the knowledge half of RLVR and shrinks the rest.**

The paper decomposes the pass@1 change of 20 released outcome-RL models (1.5B–32B,
each against the thinking base it started from) into finishing `F`, reliability `P`
and reach, at fixed token budgets (§3.1, Eqs. 1–5). The split is exact and sums to the
pass@1 change, which makes it the most useful measurement instrument in the three
papers. The findings: where the base hits the budget, finishing carries 82% [77, 86]
of the change at 4k/8k (Table 1); where it does not, reliability rises in 16 of 16
recipes (+0.032 mean, +0.021 matched, Table 2); and reach — solving problems the base
gets wrong in all 8 samples — is bounded by 0.021 on every set (Table 4), and is
−0.007..+0.007 for the 1.5B–4B recipes, i.e. nothing. Of 144 "only the RL model solved
it" physics problems, 3.5% survive resampling and reading (§4.3). A controlled 8B probe
then tries to *install* missing physics knowledge five ways and none transfers beyond
plain RL (Table 5). For us this is a direct argument that RLVR is a **format and
termination** intervention, not a capability intervention, and that its measurable
budget is small. Caveat that matters: all bases are thinking-mode models already
widened by distillation (§1, §3), so this is the *most favourable* case for reach.

### P2 — "Agentic Midtraining" → **does not change the shape of our plan, changes three data-pipeline settings and one hyperparameter rule; the scale is 5–9 months of our compute for one run.**

NVIDIA's open recipe: 256K context, 500B tokens on a 30B-A3B and 280B on a 120B-A12B,
blend 40.8% issue–PR / 20.0% repos / 10.0% traces / 3.0% diff / 1.2% logs / 20.0%
replay / 5.0% long-context (Table 1). After identical full SFT: SWE-bench Verified
60.7→64.5, Multilingual 55.8→62.8, Pro 32.0→35.5, Terminal-Bench 22.6→23.6 (§4.3,
Table 3). The ablation we care about is Table 2: midtrained **without** agent traces
gains +0.8 on SWE-V after full SFT, the full blend +3.8 — traces carry most of it.
Table 9 is the sleeper: training on tool responses instead of masking them costs
~10 points (27.5 vs 37.8). Table 11 is the other sleeper: the same midtraining at
5e-5 instead of 1e-5 drops MATH-500 from 79.7 to 61.7. Honest limits: one run per
comparison, Nemotron only, no comparison against spending the same tokens on
pretraining or SFT (§6).

### P3 — "Training-Free On-Policy Distillation" → **does not change our plan at all, because our distillation is cross-architecture and this method is explicitly not.**

The method is RegMean with on-policy statistics: for each linear layer, solve
`W⋆ = Wb + (Σ Δd G̃d)(Σ G̃d)⁻¹` from per-teacher input Gram matrices gathered on the
teachers' own rollouts (§3.1 Eq. 2, §3.2). It needs all teachers to share one base and
one architecture (§3.1: "All share an architecture"), and it says outright that
"training remains the tool when the teachers differ in architecture or size from the
student" (§4.6). On Olmo 3 7B, two 7B specialists (math, code) fuse into a generalist
with mean 61.1 vs 59.8 / 53.6 (§4.2 Table 2), for ~10 GPU-hours on 48 GB cards
(App. B Table 4). Two results transfer anyway: the loop filter (AIME25 34.2 → 40.0,
Table 5) and the warning that token-level agreement with the teacher is *not* evidence
of capability transfer — the all-answers model tracks the specialist's next token
closely and still scores like task arithmetic (§4.4, Fig. 5a).

---

## 2. Claim table

`applies?` = does the claim bear on a 100M byte-level model, one 16 GB GPU, SFT→RLVR→distill.

### P1 — outcome RL (arXiv:2610.34894)

| # | claim | evidence | number | applies? | why |
|---|---|---|---|---|---|
| 1 | Finishing dominates the pass@1 change when the base hits the token budget | §4.1, Table 1, Fig. 2 | 82% [77, 86] of the change at 4k/8k; physics 71%, MATH-500 92% | yes | a byte model's first failure is "no parseable answer inside the budget", which is the same mechanism |
| 2 | Where the base already finishes, the gain is reliability | §4.2, Table 2 | +0.032 mean over 16 recipes; matched +0.021; 16/16 recipes rise | yes | the realistic regime for us *after* SFT teaches format |
| 3 | Net new capability (reach) is ≈ 0 | §4.3, Table 4 | ≤ 0.021 on all four sets; 1.5B–4B recipes: −0.007..+0.007 | yes | our design point is below every recipe studied; the trend is not in our favour |
| 4 | "Lone wins" mostly are not real | §4.3, Table 4 | of 144 checked, 3.5% hold up; base solves 87 of 121 RL lone wins with 56 more samples | yes | any RL win we measure at 8 samples needs resampling before it is believed |
| 5 | No RL step is missing from the base's own samples | §4.2, Table 2 | 0 of 26 decisive steps missing (base right 1–4 of 8); 1 of 57 when base never right | yes | RL re-weights; it does not add procedure |
| 6 | Supplying the missing knowledge does not transfer | §5, Table 5 | hint +0.01 [−0.01, 0.03]; teacher derivation −0.05 [−0.08, −0.03]; written setup −0.03 | yes | kills QuestA/KnowRL-style "hint the setup" for us before we build it |
| 7 | Group-relative RL is starved when success is rare | App. B.2 | "at a few percent per sample, a group of sixteen sees a success less than half the time"; gains follow `p(1−p)` | yes | the binding constraint on our 2500 tasks; see §4 item 1 |
| 8 | Reward works only where it lands | App. B.1, B.2, Table 5 | revisited problems 0.10 → 0.28 (0.31 with replay); other 293 problems and held-out do not move | yes | unless we revisit, each task teaches once |
| 9 | The reward is dominated by finishing early in training | App. B.2 | finishing = 48% of within-group reward variance early, 21% saturated | yes | our verifier's gradient will mostly encode "answer at all" |
| 10 | A setup reward was gamed end-to-end | §5, App. B.6 | +99% apparent gain, no real gain; lenient judge marked 75% of claims established vs 34% strict | yes | a direct precedent for our AST-whitelisted sympy matcher and sandbox allowlist |
| 11 | Small-scale effects are below the detection floor | §5, App. B.5 | minimum detectable effect ≈ 0.03; hint leads control seeds by +0.009 / +0.019 | yes | our 1-GPU runs cannot resolve the effects P1 reports |
| 12 | Training longer does not buy reach | §4.3, Table 4 | Spearman ρ = 0.13 between steps and robust reach | yes | "train RLVR longer" is not a lever |
| 13 | Larger bases show *some* reach, partly answer-form | §4.3, Table 4 | 7B+ net reach +0.009..+0.013; 32B 0.013 matcher / 0.004 judge | no | we are 15× below 1.5B; the only positive signal is at the far end |
| 14 | Derivation length tracks the reward, not the reasoning | §4.1, Table 1 | ρ = −0.93 between length change and finishing change; Open-Reasoner-Zero 0.98 → 0.65 finishing at 2k | yes | at a fixed eval budget, a longer model scores worse for free |
| 15 | Rebuild the score with `fin@1(B)` reported next to `pass@1(B)` | §6 | recommendation stated explicitly | yes | cheapest instrument change available to us |

### P2 — agentic midtraining (arXiv:2610.22019)

| # | claim | evidence | number | applies? | why |
|---|---|---|---|---|---|
| 16 | The blend is mostly natural SWE data with a bounded trace share | §3, Fig. 1, Table 1 | traces 10.0% of tokens (39.4B unique, 1.28 epochs); replay 20.0%; issue–PR 40.8% | partly | the 10–15% cap is a rule we can copy; the sources are not ours |
| 17 | Agent traces are necessary, plain code is not enough | §4.2, Table 2 | after full SFT: base 59.7, no-traces 60.5 (+0.8), full 63.5 (+3.8) on SWE-V; TB 22.9 / 18.5 / 25.1 | yes, as a shape | if we want tool-call behaviour, natural code will not produce it |
| 18 | Masking tool responses is worth ~10 points | §5.2, Table 9 | 27.5 vs 37.8 pass@1 | yes | 20-line dataloader rule; the single cheapest win in this memo |
| 19 | Long SFT hurts later RL trainability | §1, §2 (citing Liu 2026b, Kang 2025, Zhang 2026) | qualitative; the reason they moved learning into midtraining | yes | argues against a long SFT before RLVR, which is our plan's first stage |
| 20 | 20% replay prevents regression; 40% adds nothing | §5.3, Table 10 | every general-capability group within 1.2 points through 65B | yes | direct recipe for a continued-pretrain stage |
| 21 | Too-high LR costs general capability even with replay | §5.3, Table 11 | MATH-500 79.7 (base) → 78.8 (1e-5) → 74.3 (3e-5) → 61.7 (5e-5) | yes | same shape as our 5e-4→1e-3 cliff; supports 5e-4 |
| 22 | 256K midtraining is also the long-context stage, short-context stays intact | §5.1, Table 8 | RULER 8K: 92.3 midtrained vs 94.1 base; 92.1 after a 1M stage; "within two points at shorter lengths" | unknown | we cannot train at 8K let alone 256K; the mechanism (long-context training does not wreck short-context) is mildly reassuring |
| 23 | Context-limit failures collapse | App. E.1, Table 19 | SWE-Pro 91 → 13 of 2,193 rollouts | no | 256K harness artefact, not a small-model effect |
| 24 | Harness robustness improves on every harness | §4.5, Table 7 | +8.6 mean SWE-V, +6.8 TB2.1 (Super); +11.6 across six harnesses (Lightning) | unknown | scaffold diversity is plausible at any scale, unmeasured below 30B |
| 25 | The gains do not clearly transfer to RL headroom | App. D, Table 18 | 20 GRPO steps, single seed; "does not show that midtraining increases what RL can add" | yes | the honest ceiling of the evidence: it survives RL, it does not amplify it |
| 26 | Token cost of the recipe | Table 1, §4.3, §4.4 | 500B / 280B midtraining; SFT 33B or 100B | yes (as a veto) | 263 / 147 / 53 / 17 days at 1.9 GB/day — see §4 |

### P3 — training-free distillation (arXiv:2610.22236)

| # | claim | evidence | number | applies? | why |
|---|---|---|---|---|---|
| 27 | The closed form needs ≥2 teachers sharing one base | §3.1 Eq. 2, §3.2 | `W⋆ = Wb + (ΣΔd G̃d)(Σ G̃d)⁻¹` | no | our teacher is a 4B model of a different family and tokenizer |
| 28 | Cross-architecture teachers are out of scope, explicitly | §4.6 | "Training remains the tool when the teachers differ in architecture or size" | no | this is the sentence that closes the question |
| 29 | With one teacher the formula returns that teacher exactly | §3.1 Eq. 2 (algebra) | K=1 ⇒ `Wb + Δ1 G̃1 G̃1⁻¹ = Wb + Δ1` | no | merging an SFT checkpoint "into" its base is a no-op, not a blend |
| 30 | The loop filter is worth more than the solver | §3.4, §4.4, Table 5 | AIME25 34.2 (all answers) → 40.0 (loop-free) vs specialist 39.2 in float16 | yes | applies to any token-weighted statistic over generated text, including our SFT data |
| 31 | Token-level agreement does not imply capability | §4.4, Fig. 5a | the all-answers model tracks the specialist's next token yet scores like task arithmetic | yes | our distillation eval must be task accuracy, not KL/agreement |
| 32 | Student-written rollouts do not beat teacher-written ones | §4.5, Fig. 5b, Table 5 | student reads own answers 35.0; teacher reads student's 37.5; teacher's own loop-free 40.0 | yes (inference) | argues against spending our compute on student rollouts re-scored by the teacher |
| 33 | More rollout data helps modestly | §4.5, Table 3 | 128 problems/16k → 256/32k: MATH-500 92.0→93.2, HE+ 84.8→87.2 | yes | a few hundred answers is the whole data requirement |
| 34 | Precision matters at the merge | §3.5, §4.5 | bf16 rounding of the fused update: cosine 0.96; store fp32, eval fp16 | yes (if we ever merge) | free at 100M; fp32 weights = 400 MB |
| 35 | RL moves weights by ~1 representable step at bf16 | §3.5 | stated; the basis for the precision argument | unknown | plausible and consistent with P1's "behaviour not knowledge", but not measured here |
| 36 | Cost | App. B, Table 4 | rollouts 1 GPU-h / 64 problems; stats 40 min / 2.6M tokens; solve 25 min; ≈10 GPU-h for two 7B specialists | yes (if we ever merge) | at 100M this is an afternoon, not a project |

---

## 3. What the papers actually establish vs what I am inferring

### VERIFIED (sourced)

- The F/P/reach split is exact: `Δpass@1(B) = F + P + reach@k(B)` (P1 §3.1, Eqs. 1–5).
  Reach subtracts its own mirror, so chance lone-wins do not count.
- Finishing share 82% [77, 86] at 4k/8k; physics 71%, MATH-500 92% (P1 Table 1).
- Reliability +0.032 mean over the 16 non-cap-bound recipes, matched +0.021, and it
  rises in 16 of 16; finishing moves either way (7 of 16 finish less) (P1 Table 2).
- Reach ≤ 0.021 on all four sets; −0.007..+0.007 for 1.5B–4B recipes; +0.009..+0.013
  for 7B+ (P1 Table 4).
- 3.5% of 144 checked lone wins survive resampling and reading (P1 §4.3).
- 0 of 26 decisive RL steps are missing from the base's other samples (P1 Table 2).
- The 8B GSPO probe: control held-out 0.16 → 0.26 almost entirely by finishing
  (`fin@1` 0.39 → 0.69, `pass@1|fin` flat at 0.38); no channel beats plain RL beyond
  the 0.03 detection floor; the setup reward was gamed for +99% apparent gain
  (P1 §5, App. B.1, B.5, B.6, Table 5, Table 14).
- P2 blend shares and trace counts (Table 1); no-traces vs full-blend deltas after
  identical SFT (Table 2); tool-response masking 27.5 vs 37.8 (Table 9); LR ablation
  MATH-500 79.7/78.8/74.3/61.7 (Table 11); replay within 1.2 points (Table 10);
  RULER within two points at short lengths (Table 8); ~10-point context-limit-failure
  reduction (Table 19); 20-step RL persistence with 0.03% vs 0.34% format violations
  (Table 18, App. D.2).
- P3: Eq. (2), the K=1 degeneracy (algebra on Eq. 2), the shared-architecture
  requirement (§3.1), the explicit cross-architecture exclusion (§4.6), the loop-filter
  ablation (Table 5: 34.2 → 40.0 vs 39.2), student-rollout null result (§4.5, Fig. 5b),
  compute (App. B Table 4), precision (cosine 0.96, §3.5).

### SPECULATION (my inference, not in the papers)

- **S1.** At 100M a byte model's "finishing" failure is *format*, not *length*: it will
  emit something, just not a parseable answer. P1's mechanism (reward distinguishes
  finished from unfinished) still applies, but the 82% number does not transfer —
  their models already knew how to answer at all.
- **S2.** The `p(1−p)` starvation bound (P1 App. B.2) is the binding constraint for us:
  with 2,500 tasks and a 9–100M model, most tasks will sit at p≈0, so a group of 8–16
  yields an all-zero advantage and the step is wasted. Concrete prediction: fewer than
  ~300 of our 2,500 tasks will fall in the 1–6-of-8 band at any realistic SFT
  checkpoint. This is a prediction, not a measurement.
- **S3.** P2's tool-response masking transfers because the mechanism is capacity, not
  scale (predicting multi-KB file dumps and stack traces costs the same regardless of
  model size). Magnitude at 100M is unknown; the sign is likely.
- **S4.** P2's LR result is consistent with our 5e-4 → 1e-3 cliff (spread 0.007 →
  0.58 BPB) but does not explain it: they see a monotone capability loss on a
  converged model, we see seed-dependent divergence during pretraining. Same direction,
  different phenomenon.
- **S5.** A 100M TF-OPD merge would cost ~1–3 GPU-hours, not 10, dominated by
  generation; our box is launch-bound at 13% utilisation so the scaling is not the
  70× the parameter ratio suggests. Rough estimate only.
- **S6.** P3's method does not transfer to our architecture even with same-size
  teachers: KDA is a stateful delta rule (its "weight" is updated by the input), the
  looped block applies one weight set 3–4 times so the per-layer input distribution is
  a mixture over iterations, Engram is a hash lookup, and MoE routing splits the input
  distribution per expert. Eq. (2) is derived for `y = Wx` with one distribution per
  teacher per layer. The paper handles only embeddings/norms/biases by task arithmetic
  (§3.3); it never encounters a layer whose operator is input-dependent.
- **S7.** P1's finding is the *most favourable* case for reach (thinking bases already
  distillation-widened, §1). A from-scratch 100M model has less to re-weight and less
  to sample, so reach should be ≤ theirs, not more.

---

## 4. What we should do differently, by expected value

Effort is wall-clock hours including the repo's own costs (≥23 min builds, one heavy
thing at a time, 3 seeds per arm per `AGENTS.md` §1.2).

| # | action | why (paper) | effort | if we skip it |
|---|---|---|---|---|
| 0 | ~~**Precondition:** KDA backward is non-deterministic, disagrees with central difference by 4.5e-2..2.6e-1.~~ **CORRECTED 2026-10-28 by direct measurement:** the Python port's KDA backward matches a float64 central difference to **2.58e-7 relative / 1.15e-9 absolute** (d=64, h=4, hd=16, 12 positions, 12 sampled coordinates, eps=1e-6). The 4.5e-2 figure belongs to the retired burn/Rust lane (`AGENTS.md` §3.3 describes that tree), not this one. **Not a blocker.** Lesson: an inherited number from the other lane is a hypothesis, not a property of this code (rule #89). | repo state, re-measured | 0 | — |
| 1 | **Gate RLVR on a solve-rate band.** Sample the SFT checkpoint 8× on all 2,500 tasks; keep only tasks at 1–6 of 8; if <300 survive, do not run RLVR — run format SFT instead. | P1 App. B.2 (`p(1−p)`, group of 16 starves), §4.1 | 8–12 | most of the RL budget is spent on all-zero groups |
| 2 | **Instrument `fin@1(B)` and `pass@1` given fin in the eval harness** at a fixed byte budget, and never quote pass@1 without them. | P1 §6, §3.1 | 3–4 | every later gain is ambiguous between format and correctness |
| 3 | **Mask tool responses in every multi-turn trace**; train assistant turns only. | P2 Table 9 (27.5 → 37.8) | 2 | ~10 points of agentic competence left on the floor |
| 4 | **Delete "install knowledge via hints / teacher-derivation splicing" from the roadmap.** | P1 Table 5, Table 4 (QuestA/KnowRL robust wins: 1 and 3 problems) | 0 | days lost building a channel the literature says does not transfer |
| 5 | **Filter repetition loops out of all generated SFT/distillation data** (DEFLATE ratio < 0.05 over the last 20k chars is the paper's rule). | P3 §3.4, Table 5 (34.2 → 40.0) | 2 | loop tokens dominate every token-weighted average |
| 6 | **Judge distillation by task accuracy, not by teacher agreement/KL.** | P3 §4.4 Fig. 5a | 3 | we will "verify" a model that tracks the teacher and cannot do the task |
| 7 | **Keep SFT short before RLVR; put the behaviour into the data mix, not into more SFT epochs.** | P2 §1/§2 (long SFT reduces later RL gains) | 0 (policy) | RLVR headroom is spent before RLVR starts |
| 8 | **If we add a continued-pretrain stage: 20% replay, moderate LR, verify general caps each 25B-equivalent.** | P2 §5.3, Tables 10–11 | 1 (config) | capability regression that we notice three weeks later |
| 9 | **Do not build a TF-OPD merge now.** Revisit only if we ship ≥2 specialists from one base. | P3 §3.1, §4.6 | 0 | a 4-hour experiment on a method whose K=1 case is the identity |
| 10 | If (9) becomes true: try TF-OPD on the two specialists, fp32 storage, task-accuracy eval. | P3 §4.2, §3.5 | 4–6 | nothing lost; it is cheap and beats task arithmetic on hard items only |
| 11 | Do **not** spend compute on student-generated rollouts re-scored by the 4B teacher; distil from teacher-generated text. | P3 §4.5 Fig. 5b (35.0 / 37.5 vs 40.0) | 0 (decision) | rollout compute with no measured benefit |

---

## 5. Uncertainty list — what these papers do NOT establish

**Coverage of our regime**

1. No model below 1.5B in P1, below 30B in P2, below 7B in P3. Our 100M design point
   is outside every experiment. Nothing here is measured at our scale.
2. No byte-level or character-level model in any of the three. All tokenizers are
   subword; sequence lengths, budgets and "finishing" are all defined in tokens.
3. No stateful/recurrent layer (KDA), no weight-shared looped block, no hash memory,
   no MoE in P3's method section — so its closed form is untested on any of our layer
   types (S6).

**P1 specifically**

4. Compute is not reported anywhere: no GPU count, no GPU-hours, for either the
   20-model study or the 8B probe. The probe is "60 updates × 96 groups × 16 rollouts
   at a 12k cap" (App. B.3, Table 13) — auditable in shape, not in cost.
5. No group-size ablation, no GRPO-vs-PPO-vs-GSPO ablation, no verifier-noise ablation.
   The 20 recipes confound data, steps and algorithm: "our reads cannot explain why some
   recipes gain more reliability than others" (§4.2).
6. The closest thing to a verifier study is a matcher-vs-judge regrade: the judge lowers
   the reliability term by 0.003 (~8%) and moves net reach by <0.01 (§6 "Grading", and
   the regenerated-pairs table in App. A). That bounds answer-form noise, not
   reward-hacking risk.
7. One seed per treated arm in the probe; the held-out set was selected on base samples;
   the probe runs ~2 visits per problem, so prolonged training is untested (§6).
8. All bases are thinking-mode and already distillation-widened (§1). The result is
   about *re-weighting an already-wide base*, which is the best case for reach.

**P2 specifically**

9. One run per comparison; differences of 1–2 points (especially Terminal-Bench) may be
   run-to-run variation (§6). No confidence intervals are given for the main table.
10. Nemotron only, agentic coding only, agentic-focused SFT blend only (§6). No evidence
    that the trace share, the 256K context or the replay fraction is right for a
    non-Nemotron or a non-coding model.
11. No comparison against spending the same tokens on longer pretraining or longer SFT —
    explicitly stated (§6). So "midtraining beats nothing" is established; "midtraining
    beats more pretraining" is not.
12. Cannot separate scaffold diversity from the mere presence of traces (§4.5).
13. RL evidence is 20 GRPO steps, single seed, and the authors say it does not show that
    midtraining increases what RL can add (App. D.1).

**P3 specifically**

14. One model family (Olmo 3 7B), two specialists, one base, one tokenizer. No
    cross-architecture teacher, no size mismatch, no single-teacher/nested case.
15. No test of whether the fused model can be *further trained* (SFT or RL after the
    merge). We would need that; nobody measured it.
16. Competition benchmarks are 30 problems each, so "within a few points" is inside what
    the benchmark resolves (§4.2). The apparent AIME25 loss in Table 2 is substantially a
    bf16-vs-fp16 artefact: specialist 43.3 bf16 vs 39.2 fp16, and TF-OPD at 40.0 in
    float16 (Table 5, Table 7).
17. No calibration, safety, or out-of-domain evaluation of the fused model. No test of
    whether the merge preserves behaviour the benchmarks do not probe.

**Cross-cutting**

18. None of the three studies RL or optimizer behaviour under pretraining instability.
    P1 and P3 assume a converged, stable base; P2 assumes a converged base plus replay.
    Our lr=1e-3 spread of 0.58 BPB is 100× the paired-eval resolution this repo
    estimates (0.002–0.005 BPB, `AGENTS.md` §3.1) and larger than every effect P1
    reports. That is the gap none of these papers closes.

---

## 6. Confidence

- **P1's own findings, within its stated scope (1.5B–32B thinking bases, physics +
  competition math, k=8, T=0.6): high.** The decomposition is exact, the protocol is
  pre-registered for 10 pairs, 16 pairs were regenerated, and the negative results are
  reported with intervals. The instrument is the contribution; the numbers are its
  output.
- **P2's own findings, on Nemotron 3.5 at 30B/120B: high for the direction, medium for
  the magnitudes** (single run per cell, no CIs).
- **P3's own findings, on Olmo 3 7B with two 7B specialists: high for the method and the
  loop-filter ablation, low for generalization** (one family, 30-problem benchmarks).
- **Any transfer to a 100M byte-level model on one 16 GB GPU: low, and derived, not
  measured.** I have marked every such step as SPECULATION (S1–S7).
