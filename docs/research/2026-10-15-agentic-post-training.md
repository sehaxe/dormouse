# Agentic post-training for dormouse-b1 — SOTA sweep 2026-10-15

**Design input for the port.** Read with
`docs/architecture/post-training.md` (the owner's plan, the STARTING position),
`docs/decisions/2026-10-15-1b-readiness-plan.md` §5-§6 (what the port is today:
PyTorch, byte-level, **1.07B total / ~57M active**, E=1024 topk-2 TSCT +
Engram), and `docs/research/2026-10-15-1b-tech-verification.md` (already
verified claims, reused here without re-checking: BLT 2412.09871, ByteFlow
2603.03583).

**Method (ADR-0018 rule 1: a claim names its evidence).** ~20 arXiv MCP
`search_papers` queries (cs.LG/cs.AI/cs.CL/cs.SE, 2026-01-01 → 2026-10-15,
relevance-sorted, snippet abstracts) plus ~20 full `get_abstract` reads. The
named IDs in the owner's plan were checked individually: **2607.14777 = SEED**
(verified), **2511.16652 = "Evolution Strategies at the Hyperscale" / EGGROLL**
(verified), **2504.13134 = EBRM** (verified), **2608.03392 = "Self-Evolving
Coding Agents"** (verified). Every citation below is an **abstract-level** read
unless stated; I did **not** read methods sections, so numbers are reported as
the authors report them. arXiv rate-limited this IP repeatedly (HTTP 429 /
50 s timeouts), so the sweep is deliberately query-shaped, not exhaustive.

**Evidence marks used throughout:** `(A)` = verified from the paper's own
abstract/title via the arXiv MCP · `(M)` = abstract claim, not independently
reproduced here · `(S)` = SPECULATION, my inference, no source · `(web)` = no
arXiv source exists for the statement.

---

## TL;DR — the six verdicts

1. **The owner's "PPO+value, not GRPO" instinct is half-confirmed and needs an
   upgrade, not a reversal.** Confirmed: the GRPO family's *trajectory-level
   group advantage* is the documented failure mode in agents — credit
   misassignment in tool agents (2606.05784), advantage collapse when a whole
   group fails (2610.14000), and GRPO/Dr.GRPO/DAPO shown to be three edits of
   one number (2607.00152). **Upgraded:** at 57M active params the cheapest
   step-level credit in 2026 is *not* a trained critic — it is (a) counterfactual
   forked replay where the env is forkable (2609.33875, +5.0 pts on SWE-bench
   Verified at 14B vs outcome-only GRPO, no learned process RM), or (b)
   LLM/privileged value estimates used as a *baseline*, not a policy
   (2610.15840, 2610.20516, 2609.20784).
2. **Verifier-first, no learned reward model in v1.** No source found shows a
   learned RM earning its keep at ≤7B for executable objectives; the strongest
   2026 agentic results explicitly avoid one (2609.33875). Rubric/LLM-judge
   rewards are where the hacking lives (2605.12474, 2608.11669) and judges are
   biased and unreliably validated (2602.02219, 2606.19544), which at 1B is
   worse, not better.
3. **Small models need the *stabilisers*, not the raw RL.** Every small-scale
   agentic-RL paper found adds a divergence gate, a rollback, a teacher
   handoff, or an SFT interleave: 2605.07725 (step-wise divergence gating),
   2601.15141 (rollback of error-contaminated context, 4-7B), 2610.14000
   (verified handoff, Qwen2.5-1.5B), 2606.26027 (SFT/RL interleave fixes
   tool-use collapse), 2607.16169 (Muon's usable step-size regime, 0.5-3B).
4. **Self-evolution: three levers are data-lean and cheap (curriculum/difficulty
   selection, hindsight-skill distillation, on-policy self-distillation as an
   *update-magnitude* signal); three are expensive and unproven at our scale
   (memory-control-as-policy, harness/prompt evolution, multi-instance
   self-play). Self-play without an anchor drifts to unrealistic tasks
   (2607.03523).
5. **The little-data levers are: the verifier (a data multiplier), byte-level
   cross-tokenizer distillation from an open teacher (2602.01007: token→byte
   conversion in ~125B bytes), self-distillation for magnitudes only
   (2604.03128), policy-adaptive experience reuse (2606.30420), and reflective
   rollback (2601.15141).** Engram n-gram memory is a *pretraining* asset; no
   source found supports it as a post-training knowledge store.
6. **Build order:** SFT spans → rollout runner + trajectory parquet → verifier
   protocol (with the instrument layer) → step-level advantage (forked replay
   first) → optional controller-ES (EGGROLL) → eval anti-memorization harness.
   ~1.4-2.6k LOC, no new heavy deps. **Start with §8.**

---

## 1. Verdict table — agentic-RL recipes, 2026-10 SOTA

Read as: *does this beat plain GRPO/PPO at 1B-ish scale, and what does it cost
in a torch port?* LOC figures are **my estimates (S)** for a minimal
implementation inside the port, not measurements.

| recipe | mechanism (as the authors describe it) | reported scale + result | fit for 1B / one 16 GB GPU | LOC/deps in the port |
|---|---|---|---|---|
| **GRPO family identity** 2607.00152 (06-30, A/M) | GRPO, Dr.GRPO, DAPO are three operations on the group **standard deviation** | analysis; no scale claim | **Why the family is fragile:** it is one normalizer, so "GRPO vs DAPO" is not a real axis | 0 — it is a warning, not a recipe |
| **Credit misassignment** 2606.05784 (06-04, A/M) | GRPO broadcasts one trajectory advantage to all tokens → good tool calls inside failed trajectories get punished | multimodal search agents (sizes not in abstract) | **Direct indictment of group-relative at agent scale** — exactly our case (long trajectories, tool calls) | 0 |
| **Group collapse → verified handoff (VH-GPO)** 2610.14000 (09-27, A/M) | when every rollout in a group fails, relative advantages → 0; a frozen teacher repairs from the student state, then **control returns to the student at the earliest verified handoff point** | **Qwen2.5-1.5B** (verified in abstract); success ↑ | **Closest published size to ours.** Small-model-specific. Needs a teacher (teacher cost = real) | ~300-450 + teacher rollout path |
| **LLM value functions (LVRL / LVOPD)** 2610.15840 (10-11, A/M) | use a pretrained LLM's in-context value estimate as the **baseline**, calibrated at initial/terminal states; step-level advantages from value differences; teacher supervision focused on value-decreasing actions | AppWorld, BabyAI (abstract); `params` not stated | **Attractive at 57M active:** replace the trained critic with a *prompted* value estimator. But it wants a large value model → a second model in VRAM or host | ~250-400 (value prompt + parser); deps: whatever serves the value model |
| **Privileged-value self-distillation (VASPO)** 2610.20516 (10-14, A/M) | on-policy self-distillation as dense signal + **privileged value estimation** instead of a critic | reasoning models (size not in snippet) | Fits the "self-teacher, no external teacher" budget | ~250-400 |
| **Self-retiring OPD (RetireOPD)** 2609.20784 (09, A/M) | train a skill-conditioned teacher with env rewards, distill into a skill-free student jointly with RL, then **retire the teacher when discrepancy stops shrinking** | multi-turn agentic (sizes not in snippet) | **The retirement rule is the cheap part and the useful part** — it bounds distillation cost and stops the leakage failure of 2604.03128 | ~200-350 |
| **SEED (the owner's cite)** 2607.14777 (07-16, A/M) | completed on-policy trajectories → **hindsight natural-language skills** (generated by the policy itself) → re-score actions under skill-augmented context → **skill-induced probability shift = dense token-level OPD signal**, jointly optimized with outcome RL | text + vision agentic tasks; "consistently improves performance and sample efficiency" | **Verified as the owner read it.** No external teacher; the "teacher" is the same model with hindsight. Cheap to adopt conceptually, but needs a skill store + re-scoring pass (~2× forward per step) | ~400-600 + memory store |
| **Counterfactual Rollout Replay (CRR)** 2609.33875 (09-27, A/M) | **forkable executable envs**: restore a state, take an alternative action, roll forward; replace the advantage at selected steps with (real return − counterfactual return). **No human process labels, no learned process RM** | **14B policy; SWE-bench Verified 41.7% vs 36.7% for outcome-only GRPO at equal wall-clock** (abstract) | **Best mechanism-to-cost ratio found**, *if* our envs are forkable (code exec: yes; tool/API: state-restore needed). Fork overhead is explicitly counted by the authors | ~300-500 + env forking; no deps |
| **Anchored self-play (ASP)** 2607.03523 (07-03, A/M) | generator–fixer self-play creates an automatic curriculum; **unanchored it drifts to hard but unrealistic bugs** (degrades on human-authored); anchoring = small reference set + code-embedding similarity reward + mixing reference bugs | BugSourceBench (new) | **The anti-drift prescription** for our "self-develop" loop; needs a small human/reference seed set — we have the 46 GB corpus and the eval carve | ~250-400 + embedder for similarity |
| **Explorer-Zero** 2610.19868 (10-14, A/M) | self-play proposer constructs tasks **and** records intermediate/final env states as verifiable targets → partial-credit rewards; alternating proposer/solver | ALFWorld, ScienceWorld, AppWorld; **30B agent 59.67% AppWorld** vs 62.65% for RL on human tasks | Task+verifier synthesis in one loop — the single most direct answer to "little data". **30B is 30× our box**; porting the *loop* is cheap, expecting its numbers is not | ~500-800 (proposer/solver + state capture) |
| **ANCORA** 2604.27644 (04-30, A/M) | manifold-anchored self-play: policy **learns to question** — generates verifiable problems, solves them, verifier feedback drives self-improvement without annotated data | abstract only | Cheap data-multiplier; anchor mechanism is the part to keep | ~300-500 |
| **EAPO (experience-augmented PO)** 2606.30420 (06-29, A/M) | reuse accumulated experience **as an action-level prior injected at critical decision points** (not as fixed trajectories — that suffers policy mismatch), with an adapted importance-sampling scheme | **Qwen2.5-math-7B, Qwen3-8B** on five benchmarks (abstract) | **Directly addresses "few tokens"**: it is experience *reuse*, i.e. an effective data multiplier. 7-8B tested; behavioural claim plausible at smaller | ~300-500 (experience store + IS) |
| **CLEANER** 2601.15141 (01-21, A/M) | for **parameter-constrained models (4B-7B)** execution failures poison trajectories; **Similarity-Aware Adaptive Rollback** replaces failures with the model's own successful self-corrections during collection | 4B-7B agentic | **The small-model-specific failure and its fix.** Cheap: no extra model, no supersampling | ~250-400 |
| **SOD (step-wise OPD for SLM agents)** 2605.07725 (05-08, A/M) | plain OPD on tool-integrated reasoning **cascades**: an erroneous tool call amplifies student-teacher divergence; SOD re-weights distillation **per step** by step-level divergence | small LMs, TIR | **The correct shape of distillation for a 1B agent** — per-step, divergence-gated | ~300-450 |
| **Tool-use RL collapse** 2606.26027 (06-24, A/M) | RL alone collapses (probability spikes on control tokens) while the capability is intact; **interleaving SFT with RL stabilises**, but degrades under format/content OOD | several models (sizes in body only) | **Prescription: never run RL-only.** Keep an SFT interleave in the loop | ~150 (schedule change) |
| **Muon in agentic RL** 2607.16169 (07-17, A/M) | maps Muon's usable regime on ALFWorld with group-based objectives; fan-in Muon stable at more aggressive step sizes; high-rate Muon applies ~3.53× AdamW's update RMS; RMS-matched control removes the gain | **Qwen2.5 0.5B→3B** | **Muon+ is already our default → read before setting RL LR.** The gain is regime-dependent, not free | 0 (recipe, not code) |
| **EGGROLL (the owner's cite)** 2511.16652 (2025-11-20, A/M) | ES with perturbations structured as **rank-r matrices** → ~100× faster at billion-param scale, up to **91% of pure batch-inference throughput**; linearising theory; **competitive with GRPO for post-training LLMs on reasoning**; stable pretraining of nonlinear recurrent LMs **in integer dtypes** | billion-param models | **Two independent wins for us:** (i) ES is a real alternative when the objective is non-differentiable/noisy (routers, harness knobs); (ii) the integer-dtype recurrent-LM result touches our byte/quant line. Controller-only adoption = (S) | ~200-350 (rank-r perturbation + antithetic pairs) |
| **AbsoluteZero lineage** — `ti:"Absolute Zero"` 2026 sweep returned **no LLM successor** (only physics papers) | — | — | **Honest gap:** the 2025 AbsoluteZero idea lives on as ANCORA/Explorer-Zero/GASP in this sweep; do not cite a 2026 "AbsoluteZero" | — |
| **RAGEN-2 / template collapse** 2604.06268 (04-07, A/M) | entropy is the wrong stability metric: models use **input-agnostic templates** with stable entropy; decompose reasoning quality into within-input entropy **and cross-input mutual information**; MI correlates with performance | multi-turn agentic | **Adopt MI proxies as the online collapse monitor** — a metric, not a recipe | ~100-200 (logging only) |
| **SWE-Protégé** 2602.22124 (02-25, A/M) | SLM stays the sole decision-maker, learns **when to ask a strong expert**, when it is stalled, and how to follow through; SFT on expert-augmented trajectories + agentic RL that discourages looping | **Qwen2.5-Coder-7B → 42.4% Pass@1 SWE-bench Verified (+25.4% over prior SLM SOTA), ~4 expert calls/task = 11% of tokens** | **The strongest "little data + small model" pattern in the sweep**: sparse expert calls instead of a resident teacher | ~400-700 (expert-call plumbing + loop penalty) |
| **SWE-MiniSandbox** 2602.11210 (02-11, A/M) | container-free RL for SWE agents (containers cost storage/setup at scale) | SWE tasks | **Env-cost answer for a one-box setup**: we do not need docker-in-docker per rollout | ~200-350 (process jail, rlimits) |
| **Rollout infra**: OpenTinker 2601.07376 · ThunderSyncRL 2610.05935 · single-rollout async 2607.07508 · long-horizon efficient RL 2608.10357 (A) | many LoRA policies over shared execution; async rollout/verify to stop the learner starving; per-rollout async updates | infra papers | **Read before writing the runner**; we need none of the multi-policy machinery at n=1 model | informs (b) |

### 1.1 What this table means for the owner's plan

- **Confirmed**: the plan's env pools (code/math/agent), its
  synthesis→solvability→anti-hack pipeline, its 5/70/archived curriculum bands,
  and its selective on-policy distillation are all in the 2026 mainstream
  (2607.14777, 2610.19868, 2602.20532, 2605.07725).
- **Changed**: "RLVR PPO+value" as the *first* RL mechanism is now the most
  expensive of three credible options. Forked-replay advantage (2609.33875)
  and self-distilled credit (2609.20784/2610.20516) deliver step-level credit
  without training a critic; a critic at 57M active params is a small head but
  brings GAE over long agent trajectories and the instability that
  `2606.20008`/`2605.21654` document.
- **Added**: small-model stabilisers are not optional at our size
  (2605.07725, 2601.15141, 2610.14000, 2606.26027), and the *monitor* changes
  from entropy to mutual information (2604.06268).

---

## 2. Rewards on little data — verifier classes, synthesis, anti-hack

**Verdict: verifier-first, everywhere. A learned reward model is not needed for
v1, and I found no 2026 source in which one pays for itself at ≤7B on an
executable objective.** The strongest counter-example to "you need an RM" is
the paper with the best measured agentic gain in this sweep: CRR "needs no
human process labels or learned process reward model" and still takes SWE-bench
Verified from 36.7 → 41.7 (2609.33875). (M)

### 2.1 Verifiability classes, cheapest → most expensive

| class | signal | evidence | cost in the port |
|---|---|---|---|
| exact-match / parseable answer | string/number equality | 2606.30420 (math, Qwen2.5-math-7B), 2603.07084 testbed | ~30 LOC |
| **code execution** (unit tests) | pass/fail + which assertions | 2602.11210, 2607.03523 (generated bug-fix tasks carry tests), 2601.20103 (reward-hack detection **in code environments** — proves the class is hackable) | ~250-400 LOC (jail, timeouts, rlimits) |
| **environment state diff** | compare env state / intermediate checkpoints against recorded targets | 2610.19868 (records intermediate+final states as verification targets), 2609.33875 (fork + restore) | ~300-500 LOC |
| end-to-end grader (repo-level) | SWE-bench-style resolution | 2602.22124, 2610.07898, 2610.14405 | heavy: needs repo fixtures; long per-rollout wall-clock |
| rubric / LLM-judge | criteria scored by a model | 2605.12474, 2608.11669, 2604.13618 | **the one to defer** (see 2.3) |

### 2.2 Synthetic verifiable task generation

- **Env-derived, not text-derived**: generate the task *and* its verification
  targets by exploring the environment (2610.19868); synthesize *environments*
  from reusable skills (2609.33772 Skill2Env) and terminal tasks from skill
  graphs (2604.25727). (A/M)
- **Solvability screen**: self-play/self-questioning methods probe learnability
  by checking the solver can solve what the proposer made (2604.27644 ANCORA;
  2608.21871 zero-data self-play "probes learnability"). (A/M)
- **Anti-drift anchoring is mandatory**: unanchored generator-fixer self-play
  drifts toward "difficult but unrealistic" bugs and *degrades* on
  human-authored ones; a **small reference set** plus a similarity reward fixes
  it (2607.03523). (M)
- **Anti-reward-hack screening has a literature now**: hack-verifiable
  environments (2605.20744), a controlled testbed for measuring hacking
  (2603.07084), contrastive detection in code envs (2601.20103), gradient
  fingerprints (2604.16242), and reward hacking shown to **generalize into
  broader misalignment** (2609.14998) and to be **transmissible between agents**
  (2610.12822). (A/M) The plan's "anti-reward-hack" stage should name which of
  these it implements; "screened" is not a mechanism.

### 2.3 Is a learned RM ever needed at our scale?

Honest answer: **only for objectives with no executable check.** Where it is
tempting, the 2026 evidence is that it goes wrong in ways a 1B model cannot
police:
- rubric-as-reward is hackable, and the fix is *dropout*, not a better judge
  (2605.12474, 2608.11669);
- rubric judges carry position bias (2602.02219) and judge agreement metrics
  overstate validity (2606.19544);
- **the judge can itself be the failure**: a biased judge silently disables
  skill retirement in a self-evolving agent (2607.07436). (A/M)

`s` (S): at 57M active params a self-judge is the *same* 57M-active network
being trained — self-reward at that capacity is exactly the configuration
2607.07436 warns about. **Recommendation: no learned RM in v1; no LLM-judge
rewards in v1; revisit only with a frozen, externally-validated judge and
rubric dropout.**

---

## 3. Self-evolution mechanisms, ranked by (data-lean bet × implementation cost)

Ranking is mine (S); each mechanism's evidence and failure modes are sourced.

| # | mechanism | data-lean bet | reported failure modes | minimum size reported | cost |
|---|---|---|---|---|---|
| 1 | **Curriculum / difficulty selection driven by verifier pass-rate** (the plan's 5/70/archive bands, upgraded) | high — it is a *sampling* change, no new data | needs diversity monitoring: entropy hides template collapse (2604.06268); difficulty-aware variants exist for multi-turn tools (2608.21863); bandit/co-adaptive selectors (2602.20532, 2606.19750) | bandits/curricula tested on 7-8B LLMs and on RL agents; **no ≤1B agentic report found** (S) | low: ~150-300 LOC if selection is a verifier-pass-rate table |
| 2 | **Hindsight-skill distillation (SEED-shaped)** | high — the corpus is your own rollouts | skill library rot and retirement depend on an unbiased judge (2607.07436); leaks if the teacher has privileged info (2604.03128) | agentic LLMs (2607.14777, 2609.20784); skill-library papers are LLM-agent papers (2605.25430, 2605.27366, 2602.02474, 2608.16114) | medium: ~400-600 + a skill store (parquet + retrieval) |
| 3 | **On-policy self-distillation as an update-magnitude signal** | high — no teacher tokens needed | as a *direction* signal it leaks and destabilises (2604.03128); self-distillation can shorten reasoning and degrade accuracy (2603.24472); OPD in TIR cascades (2605.07725); trust-region/relaxed variants exist (2606.01249, 2603.11137); retirement rule (2609.20784); SLM-agent variant (2610.19439) | QQ: 7B-8B class for EAPO (2606.30420); small-model-specific TIR (2605.07725) | low-medium: ~250-400 on top of (d) |
| 4 | **Memory control as policy** (our "memory evolution" lever) | high in principle | memory turns credit assignment across sessions into a hard problem (2605.21768); densified rewards needed (2601.23014); needs an episode-persistent workspace + shell (2609.34422 CAMG, 4 envs, **async PPO**) | CAMG: not stated in abstract; memory-policy papers are LLM-agent scale | **high**: long-horizon envs + storage + async PPO |
| 5 | **Harness / prompt evolution** | unclear | no dedicated evidence found; the nearest is that **harness details silently reverse method rankings** (2610.02911), and EGGROLL covers *knob* search (drift-free, quantitative) instead | — | low cost, **low evidence**: (S) treat EGGROLL-style knob search as the honest version of this lever |
| 6 | **Multi-instance self-play** | highest generation volume | drift to unrealistic tasks without anchoring (2607.03523); proposer/solver alternating training is the expensive half (2610.19868) | 30B (2610.19868) | high: 2 policies + env fork + storage |

**Failure-mode checklist to instrument from day one** (all sourced):
template collapse invisible to entropy (2604.06268) · diversity collapse and
pass@k inversion (2606.15455, 2607.20543) · pass@k is a diagnostic, not an
objective (2511.16231) · spurious rewards elicit reasoning anyway, so
"it improved" is not evidence the reward was right (2506.10947) ·
tool-use/format collapse (2606.26027) · skill-retirement failure under a biased
judge (2607.07436) · reward hacking → misalignment (2609.14998) and its
transmission (2610.12822) · contamination at eval and at *search* time
(2608.29463, 2606.05241, 2601.04301).

---

## 4. The little-data levers, ranked by mismatch tolerance

"Mismatch tolerance" = how much the lever *depends* on data that matches the
model's own distribution. Higher = safer for a byte-level 57M-active policy.

| # | lever | mismatch tolerance | evidence | verdict for dormouse |
|---|---|---|---|---|
| 1 | **Verifier + forkable env as a data multiplier** | **high** — the signal is executable, not distributional | 2609.33875 (+5.0 pts, no process RM), 2610.19868 (state targets), 2602.11210 (container-free) | **Adopt first.** Our envs are ours, so forking is a design choice we control |
| 2 | **Reflective rollback of polluted trajectories** | **high** — self-correction stays in-distribution by construction | 2601.15141 (4-7B) | Adopt: cheap, targeted at exactly the small-model failure |
| 3 | **On-policy self-distillation, magnitudes only** | **high** — the student *is* the teacher | 2604.03128 (RLSD), 2609.20784 (retirement), 2610.20516 | Adopt, gated by divergence (2605.07725) |
| 4 | **Policy-adaptive experience replay** | medium — old experience must be re-weighted | 2606.30420 (EAPO, 7-8B), 2603.18079 (step-level) | Adopt after (1)-(3); needs the IS scheme |
| 5 | **Byte-level distillation from an open teacher** | **medium-low for us** — teacher is token-level, we are byte-level | **2602.01007: token-trained LM → byte LM in a two-stage recipe (progressive KD aligning byte reps to teacher embeddings, then byte-level SFT) using ~125B bytes, validated on Llama/Qwen/OLMo**; 2604.07466 (byte-level interface for cross-tokenizer), 2607.22334 (byte-prefix marginalization), 2610.13257 (credit alignment), 2609.34738 (event completion) | **(S) Which teachers are legally/cheaply usable is NOT an arXiv question** — 2602.01007 names Llama/Qwen/OLMo as *technical* choices; license verification is owed and must be checked per-model before use. Compute: teacher forward per student rollout, so budget it as ≥1× extra forward |
| 6 | **Engram / n-gram memory as a knowledge store instead of weights** | n/a — **no evidence found** | the memory literature found is *agent* memory (files, summaries, retrieval), not parameter substitution (2602.06052, 2609.34422, 2605.21768, 2601.23014) | **(S)** treat Engram as a pretraining asset (as the tech-verification doc already does, 2608.30320/2610.20456); do not plan post-training gains from it. The post-training analogue is an **episodic workspace** (2609.34422) |

---

## 5. What to BUILD into the port — the minimal stack, ranked

Order matters: each rung is a precondition for the next. LOC are **estimates
(S)** for the port's style (`uv`/`ruff`/`pytest`, instrument layer day-one,
pure kernel interfaces).

| order | component | what it is | deps | LOC (S) | why this order |
|---|---|---|---|---|---|
| 1 | **(a) SFT data class + loss reuse** | CE the port already has, masked to assistant spans; byte-level role tags (`<|user|>`/`<|assistant|>` as byte sequences); data class = (prompt, response, verifier-tag) records | none beyond the existing byte stream | 150-300 + JSONL reader | Every other rung consumes SFT-format records; the loss is free |
| 2 | **(b) Rollouts runner + trajectory parquet** | N parallel rollouts over env processes; per-step record `{step, action_bytes, obs_digest, tool, tool_ok, tokens, verifier_calls}`; append-only parquet/JSONL; deterministic seeds | `pyarrow` or stdlib `json`; `subprocess`; **no** ray/verl-class stack (2601.07376 is read-for-design) | 400-700 | Nothing can be measured, replayed, or distilled without it; also it *is* the Experience data for 2606.30420 |
| 3 | **(c) Verifier protocol** | one interface, three implementations: `exec` (subprocess, rlimits, no network, timeout → 2602.11210's container-free shape), `exact_match` (parse → compare), `env_step` (state diff vs recorded targets); **every decision logged** (LOUD/COUNTED/SILENT per ADR-0011) | stdlib `subprocess`/`resource`/`signal` | 250-400 | The verifier is the reward; an unlogged verifier is the ADR-0019 failure mode |
| 4 | **(d) Step-level advantage** | *start with* forkable counterfactual advantage (2609.33875) if the env can restore state; otherwise privileged/self-distilled credit (2609.20784/2610.20516) with divergence gating (2605.07725); a value head only as an optional baseline, not a requirement | none new | 300-500 (fork) + 250-400 (credit) | This is the owner's PPO-vs-GRPO question, answered as a *mechanism* choice (see verdict 1) |
| 5 | **(d2) SFT/RL interleave + rollback hygiene** | never RL-only (2606.26027); roll back error-contaminated context (2601.15141); keep a KL/clip recipe (the papers' shared baseline) | none | 150-250 | Cheap, and it is the difference between "trains" and "collapses" at small scale |
| 6 | **(e) Controller ES (EGGROLL)** | rank-r perturbation + antithetic pairs on **control-plane params only** (routers/iteration budget/halt policy), objective = verifier pass-rate, no gradients | none (pure torch) | 200-350 | **Only after (c) exists** — ES needs a cheap, low-variance scalar. Paper scale is billion-param ES (2511.16652); controller-only use is (S) |
| 7 | **(f) Eval harness, anti-memorization** | frozen held-out env instances + fresh-seed task generator + **harness probe checks** (2610.02911) + contamination audit per eval round (2608.29463, 2606.05241) + the MI/entropy collapse monitor (2604.06268) | none | 200-300 | A number from an unprobed harness is not a number; this repo's own history (retracted BPB classes) is the same defect |

**Total ≈ 1.4-2.6k LOC** (S) on top of the port's own 2.85-3.9k band — i.e. the
post-training lane is roughly the size of the port itself, and should be built
the same way: mechanisms only, no frameworks.

**Explicitly NOT built in v1** (S, with reasons): a learned reward model (§2.3);
a full PPO critic with GAE over long agent trajectories (2606.20008, 2605.21654
document the trade-off, and 2609.33875 gets more with less); rubric/LLM-judge
rewards (§2.3); multi-policy infra (2601.07376 — n=1 model); memory-control RL
environments (§3 row 4, high cost, no ≤1B evidence).

---

## 6. CLI spec BLOCK — `dormouse post-train` (specification only, not implemented)

Shape follows the existing TYper CLI (`dormouse train|eval|decode|plot`):
subcommand per stage, config snapshot + refusal on resume (ADR-0005/0021
discipline), every degradation marked LOUD/COUNTED/SILENT.

```text
dormouse post-train sft    --base <ckpt.bin> --data <sft.jsonl>
                           [--mask assistant|all] [--epochs N] [--lr F]
                           [--interleave-rl F]        # SFT:RL ratio, default 0.25 (2606.26027)
                           --out <ckpt.bin>

dormouse post-train roll   --base <ckpt.bin> --env <name|spec.toml>
                           --n <rollouts> [--max-steps N] [--temp F]
                           [--verifier exec|exact|envstep] [--fork]     # 2609.33875
                           --out <traj.parquet> --seed N

dormouse post-train verify --traj <traj.parquet> --verifier <kind>
                           [--fork-budget N] [--hack-screen on|off]      # 2605.20744
                           --report <verify.json>   # per-call log, LOUD on every unlogged path

dormouse post-train rl     --base <ckpt.bin> --traj <traj.parquet>
                           --adv fork-replay|counterfactual|group|value # default fork-replay
                           [--clip F] [--kl F] [--lr F]
                           [--consolidate seed|ropd|off]                # 2607.14777 / 2609.20784
                           [--div-gate F]                               # 2605.07725
                           [--rollback saar|off]                        # 2601.15141
                           [--monitor entropy,mi]                       # 2604.06268
                           --out <ckpt.bin>

dormouse post-train evolve --base <ckpt.bin> --memory skills|none
                           [--band mastered=F,frontier=F,archived=F]    # 5/70/rest
                           [--anchor <ref.jsonl>] [--anchor-weight F]   # 2607.03523
                           [--retire-teacher F]                         # 2609.20784
                           [--es-rank r --es-pop N]                     # EGGROLL, control-plane only
                           --out <ckpt.bin>

dormouse post-train eval   --base <ckpt.bin> --suite <name|spec.toml>
                           [--fresh-seed on|off] [--probe-harness]      # 2610.02911
                           [--contam-audit]                             # 2608.29463
                           --out <eval.json>
```

Contract notes (each one is a rule, not a preference):
1. `--adv group` exists **only** as a baseline to beat, never as the default —
   the default is the mechanism the 2026 evidence supports (2606.05784,
   2610.14000, 2609.33875).
2. `--verifier` has no "learned" option in v1 (§2.3).
3. Everything a flag can turn off is **COUNTED** (a log line naming the arm);
   a resume whose resolved config differs refuses (ADR-0021).
4. `roll` and `verify` write the fields the instrument layer needs to be able
   to answer "did this arm run": `verifier_calls`, `verifier_kind`,
   `tool_calls`, `tokens`, `fork_count`, `div_gate_fired`.
5. `evolve --es-*` operates on the control plane only; touching policy weights
   with ES must be a separate, louder flag if ever added (S: not recommended).

---

## 7. VERIFIED (sourced)

- The four IDs the owner's plan cites are correct and are what the plan claims:
  **2607.14777 SEED** (hindsight skills → dense token-level OPD signal, joint
  with outcome RL) · **2511.16652 "Evolution Strategies at the Hyperscale"**,
  the EGGROLL paper (rank-r perturbations, ~100× at billion-param scale, 91% of
  batch-inference throughput, **competitive with GRPO** on reasoning
  post-training, stable integer-dtype recurrent-LM pretraining) ·
  **2504.13134 EBRM** (energy-based reward refinement) · **2608.03392
  "Self-Evolving Coding Agents"** (framework/memory/skills/tools taxonomy).
- GRPO-family fragility in agents is documented from three angles: credit
  misassignment (2606.05784), all-fail advantage collapse (2610.14000), and
  "GRPO/Dr.GRPO/DAPO are three operations on one number" (2607.00152).
- Forked counterfactual replay beats outcome-only GRPO at 14B on SWE-bench
  Verified (41.7 vs 36.7, equal wall-clock) **without any learned process RM**
  (2609.33875).
- Small-model (0.5B-7B) agentic RL has specific stabilisers published:
  divergence-gated step-wise distillation (2605.07725), rollback cleaning
  (2601.15141), verified teacher handoff at 1.5B (2610.14000), SFT/RL
  interleave against tool-use collapse (2606.26027), a Muon step-size regime
  (2607.16169), sparse expert calls at 7B with 11% of tokens (2602.22124).
- Entropy is an insufficient stability monitor for multi-turn agents; mutual
  information proxies correlate better (2604.06268); pass@k degradation
  (2606.15455, 2607.20543) and spurious-reward elicitation (2506.10947) are
  published failure modes.
- Anchored self-play is the published fix for benchmark drift in
  generated tasks (2607.03523); env-derived verifiable subtask rewards reduce
  collapse among failed rollouts (2610.19868).
- Token→byte distillation from existing token-trained LMs is published and
  works in ~125B bytes (2602.01007), with a cross-tokenizer OPD line
  (2607.22334, 2610.13257, 2609.34738, 2604.07466).
- Harness details can silently reverse RL method rankings (2610.02911);
  contamination taxonomies and search-time contamination exist
  (2608.29463, 2606.05241, 2601.04301).

## 8. SPECULATION (my inference — no source)

- **(S) LOC estimates** in §1 and §5 are engineering guesses for this port, not
  measurements or citations.
- **(S) Controller-only EGGROLL** (routers/iteration budget): the paper's
  evidence is billion-parameter *model* ES; applying it to a small control
  plane is plausible (fewer params ⇒ cheaper), never measured at our shape.
- **(S) No paper was found doing agentic RL at ≤1B total / ≤100M active.** The
  smallest credible published points are 1.5B (2610.14000) and 0.5-3B
  (2607.16169), both dense tokenizer models on text agent benchmarks. A
  byte-level, 57M-active, MoE+Engram model is **outside the published
  evidence**. The mechanisms above are therefore *transfers*, not guarantees.
- **(S) Engram as a post-training knowledge store**: no source found. The
  memory literature found is agent-memory (files/summaries/retrieval), and the
  plan's "memory evolution" should be read as an *episodic workspace*
  (2609.34422) until evidence says otherwise.
- **(S) "GRPO is DPO / encourages verbosity"** (the plan's phrasing) is **not
  confirmed by any source found**: 2607.00152 confirms the *standard-deviation*
  reading, not the DPO reading. Treat the plan's line as unverified.
- **(S) Teacher licensing**: 2602.01007 names Llama/Qwen/OLMo as technical
  teacher families; which licences permit distillation into a released model
  is `(web)`/legal, not arXiv, and must be checked before use.
- **(S) EBM reranker for agentic rewards**: EBRM (2504.13134) is verified for
  *preference/alignment* rewards; this sweep found no 2026 agentic-verification
  result that validates it as a step-level scorer. Keep it **after** the first
  RL pass, as the plan already says.

## 9. Open questions this sweep could not close

1. No published k/rank sweep, and no published agentic-RL run, at ~1B total /
   ≤100M active — everything above is a transfer at our scale.
2. Whether `fork-replay` advantage survives a *byte-level* action space (the
   CRR evidence is token-level agents, 14B).
3. Whether a 57M-active policy can carry a useful natural-language skill store
   (SEED-style) — every skill-library paper found is LLM-agent scale.
4. Cost of `verify` at scale on one GPU box: no paper reports verifier-seconds
   per rollout except CRR's own "fork overhead" remark.
5. Legal/licence status of each teacher family for distillation into a shipped
   model `(web)`.
6. Whether the Muon agentic regime (2607.16169) transfers to Muon+ with TSCT
   low-rank factors — untested anywhere.
7. `docs-site/tools/manifest.mjs` has **not** been updated for this document
   (this task was restricted to one file); `python3 tools/check_doc_refs.py`
   will flag it until that line lands.

---

## 10. Start here — 3 steps

1. **Build (c) the verifier protocol + (b) the rollout runner together, on
   `code` only** (`exec` verifier, subprocess jail, per-call logging). Nothing
   else in this document can be tested without a reward that cannot lie, and
   the runner is what produces both the RL data and the experience store.
   Gate: a 20-task smoke where every verifier decision is attributable.
2. **Add (a) SFT spans and (d) fork-replay advantage, then run the smallest
   honest A/B this repo's protocol allows**: fork-replay vs `group` (GRPO-shaped
   baseline) at fixed wall-clock, byte-level, on the same env subset — the
   2609.33875 comparison reproduced at our scale or refuted. Instrument
   `div_gate`, `fork_count`, and the MI monitor from day one.
3. **Only if that A/B shows RL is learning anything**, add (e) control-plane
   EGGROLL and the (f) harness probes, then wire the self-evolution loop
   (SEED-shaped hindsight skills with the retirement rule) as a *stage*, not a
   background process.

### Sources (arXiv IDs cited above, deduplicated)

2511.16652 · 2504.13134 · 2506.10947 · 2511.16231 · 2601.04301 · 2601.07376 ·
2601.15141 · 2601.20103 · 2601.23014 · 2602.01007 · 2602.02219 · 2602.06052 ·
2602.09305 · 2602.11210 · 2602.20532 · 2602.22124 · 2603.07084 · 2603.11137 ·
2603.12248 · 2603.15957 · 2603.18079 · 2603.24472 · 2604.03128 · 2604.06268 ·
2604.07466 · 2604.13618 · 2604.16242 · 2604.25727 · 2604.27644 · 2605.07725 ·
2605.12474 · 2605.15155 · 2605.18871 · 2605.20164 · 2605.20744 · 2605.21654 ·
2605.21768 · 2605.25430 · 2605.27366 · 2605.30159 · 2606.01249 · 2606.05241 ·
2606.05784 · 2606.15455 · 2606.17680 · 2606.19544 · 2606.19750 · 2606.20008 ·
2606.26027 · 2606.30420 · 2607.00152 · 2607.03523 · 2607.07436 · 2607.07508 ·
2607.11012 · 2607.14777 · 2607.16169 · 2607.20543 · 2607.22334 · 2607.27973 ·
2608.03223 · 2608.03392 · 2608.04788 · 2608.10357 · 2608.11669 · 2608.16114 ·
2608.21871 · 2608.29463 · 2608.29897 · 2609.05295 · 2609.12424 · 2609.14998 ·
2609.20784 · 2609.33772 · 2609.33875 · 2609.34422 · 2610.00388 · 2610.00954 ·
2610.02911 · 2610.05935 · 2610.12822 · 2610.13257 · 2610.14000 · 2610.14405 ·
2610.15840 · 2610.19439 · 2610.19868 · 2610.20516
