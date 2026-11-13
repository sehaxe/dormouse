# Agentic post-training at ~1B on one 16 GiB card: which objective, which data

**Date:** 2026-11-09. **Mode:** read-only research; this file is the only write.
**Extends:** `2026-11-08-post-training-at-1b.md` (SFT/RLVR at 1B, eval, byte-level) and
`2026-11-08-test-time-iteration.md`. It does not re-derive them; it cites them and
contradicts them where newer numbers say so (§7.1).

**Evidence marks.** `[T]` = stated in the cited paper's abstract, table or body text, read
here. `[C]` = my arithmetic on cited numbers, formula printed. `[M]` = measured in this
repo. `[S]` = SPECULATION, my inference, no measurement. Every arXiv ID below was resolved
through the arXiv MCP (`get_abstract`, title checked against the claim) or through
`download_paper` + `search_paper_text`; IDs I could not resolve were dropped, not guessed.

---

## 0. TL;DR (RU)

1. **Порядок другой, чем в отчёте от 08.11.** Тогда первым шёл общий SFT. Для *агентной*
   способности дешевле всего оказался не SFT на большом миксе, а **экспертная итерация с
   верификатором**: 491 траектория SWE-Gym даёт 7B-модели +9.0 пункта resolve (1.0 → 10.0)
   [T, 2412.21139 Table 3]. Обучение на таком объёме — минуты, не часы.
2. **Самое сильное измеренное число по «слабый учитель + верификатор» — 1B-класс:** SEAL на
   Llama-3.2-1B-Instruct решает 72.5% подмножества ARC против 20% у того же self-edit цикла
   без RL и 0% у in-context обучения [T, 2506.10943 Table 1]; на 7B тот же метод поднимает
   SQuAD без контекста 33.5% → 47.0% и обгоняет синтетику от GPT-4.1 [T, §4.2]. Награда там
   внешняя (успех задачи), а не самооценка — это и есть работающая часть.
3. **Самосуждение измеренно ломается.** Селф-плей с reference-free судьёй разгоняет
   pass-rate судьи 0.72 → 0.94 при истинной точности 0.20 на трёх сидах [T, 2607.05904
   abstract]. «Модель сама себя улучшает» без внешнего верификатора — не гипотеза, а
   измеренный negative result и здесь, и в 2310.01798 (без внешней обратной связи
   self-correction ухудшает ответ).
4. **Данные: качество стоит +2.8, количество +8.0.** На 32B P2T: outcome-filtered SFT 39.6 →
   size-matched 42.4 (+2.8) → full 50.4 (+10.8) на SWE-bench Verified [T, 2605.21996 Table 1].
   То есть после появления верификатора фильтрация — не главный рычаг, объём ещё важен.
5. **Разнообразие сред — отдельная ось масштабирования.** AWM: 10 сред → деградация, 100 →
   рост, 526 → ещё рост; 8B на BFCLv3 53.83 → 65.94 (+12.1) [T, 2602.10090 §5.2, §6.4].
   Правилами сгенерированные среды (без LLM) дают 1.7–2.2× на multi-hop поиске при 3B/7B
   [T, 2609.40221 §1].
6. **Байты: опубликованного byte-level агентного пост-тренинга нет.** BLT не делал SFT
   [T, 2412.09871]; UTF8Tokenizer задаёт протокол C0-байтов для tool calling, но без
   измерений [T, 2510.16987 §3.1]; ByteFlow — только претрейн [T, 2603.03583]. Любой наш
   результат здесь — первое измерение, не репликация.
7. **Цена формата в байтах считается явно:** JSON-вызов `{"tool":"calc","args":{"expr":"17*23"}}`
   — 39 байт, из них 34 — синтаксис [C]; компактный протокол `\x02calc\t17*23\n` — 12 байт,
   в 3.25 раза меньше [C]. При живом `max_seq_len=1024` и `byte_patch=false` это разница
   между 26 и 85 вызовами в окне [M/C].
8. **План на этой коробке — 3 эксперимента, ~15 GPU-ч (+50% запас):** протокол-данные (4 ч),
   skill-library без обновления весов (3 ч), экспертная итерация (8 ч). Все три дешевле
   одного 2 GB прогона базы (7 ч при 79.5 kB/s) [M]. Пре-регистрированные abort-правила — §6.

**Confidence:** high on the measured rows (each number read from the named table); medium on
cost estimates for this box (generation rate is assumed, §6.0); low on transfer to 95M — no
published agentic post-training exists at that scale or at byte level.

---

## 1. The constraint set that decides every recommendation

| Constraint | Value | Consequence |
|---|---|---|
| GPU | RTX 5070 Ti, 16 GiB, sm_120, one card | one model in VRAM at a time; no resident critic + policy + PRM stack |
| Live recipe | 95M dense, **vocab 256, byte-level, `max_seq_len=1024`, `byte_patch=false`** [M, `port/configs/flagship_base.toml:8`, `config.py:406,1009`] | 1 token = 1 byte = 1 position; a 1024-byte window holds ~25 JSON tool calls |
| Throughput | **79.5 kB/s** at batch 24 + `torch.compile` static max-autotune [M] | 1 GB = 3.49 h; 0.309 s/step at 24×1024 B; 11,646 steps/h [C] |
| Base data | 2–20 GB bytes, not trillions of tokens | post-training data is ≤0.5% of pretraining; training is never the bottleneck |
| Post code | `post/{rl,rl_train,rollouts,tasks,verifiers,agent}.py`, `selfplay/*` (4.1k LOC) [M] | the loop exists; the open question is objective + data, not infrastructure |
| Eval budget | held-out BPB + IFEval-class format + custom suite; GSM8K at floor | a 200-item suite has SE ≈ 2.8 pp at p=0.8 [C from 2411.00640], so ≥500 items or 3 seeds |

**Generation rate assumption.** All rollout costs below use **0.5 kB/s** batch-1 byte-level
decode (range 0.3–1.0 from the prior report, unchanged). This is the single largest
uncertainty in §6 and is marked `[S]` wherever it enters a cost.

---

## 2. Ranked methods: measured delta, what it needs, whether it survives here

Ranked by *measured capability delta per GPU-hour of training* — but the honest ordering has
to separate two things the literature conflates: **training cost** (which is negligible for
every method at 95M) and **rollout/label cost** (which is the real budget). The "cost class"
column is that second thing.

| # | Method | Needs | Best measured evidence (model · data · delta) | Compute as published | Cost class on this box | Survives at 1B / byte? |
|---|---|---|---|---|---|---|
| 1 | **Verifier-filtered trace SFT / expert iteration** (STaR, RFT, SWE-Gym) | prompts with a programmatic check; a sampler (self or teacher); SFT | 7B · **491 trajectories** · SWE-bench Lite 1.0 → 10.0 (**+9.0**) [T, 2412.21139 Table 3]; LLaMA-7B · GSM8K · SFT 35.9 → RFT 49.3 (**+13.4**) [T, 2308.01825 §1]; GPT-J 6B · STaR [T, 2203.14465] | not reported | **S** — training minutes; generation dominates | **yes** — the cheapest thing with a measured agentic delta; byte cost is in §5 |
| 2 | **Memory / skill library, no weight update** (Voyager, ExpeL, AWM, AutoFyn) | task suite; store of verified traces/skills; retrieval | Voyager (GPT-4, Minecraft) 3.3× unique items, 2.3× distance, 15.3× faster tech milestones [T, 2305.16291 abstract]; AWM web +24.6% / +51.1% relative [T, 2409.07429 abstract] | none (inference only) | **XS** — eval-only | **qualitatively yes, quantitatively unmeasured ≤1B**; needs a store of *correct* traces, which is method 1's output |
| 3 | **Synthetic tool-call data from a stronger model, execution-verified** (APIGen, ToolACE, P2T) | frontier teacher + executable API/DB + verifier | xLAM-**1B** trained on 60k APIGen entries beats GPT-3.5-Turbo on BFCL, rank 24 [T, 2406.18518 abstract, §5]; P2T 32B +10.8 [T, 2605.21996 Table 1] | not reported | **M–L** — teacher API cost, not GPU | **yes for data, no for teacher on-box**; the 1B data point is the closest published to our target |
| 4 | **Environment synthesis + multi-turn RL** (AWM, PhantomEnvironments) | executable envs, reward hooks, RL loop | AWM 8B BFCLv3 53.83 → 65.94 (**+12.1**), τ²-bench 26.44 → 33.45, 526 envs/3,315 tasks [T, 2602.10090 §5.2]; Phantom (rule-based, no LLM) 3B/7B +1.7× / +2.2× relative on multi-hop [T, 2609.40221 §1] | not reported ("limited computation budget", 526 of 1,000 envs) [T, §5.1] | **XL** — RL over 4B–14B for many steps | **envs yes, RL no** at 95M; env count is a real scaling axis (§4.3) |
| 5 | **RLVR with execution verifier** (code, search, tool) | on-policy rollouts, unit tests/EM, solve-rate band | Qwen3-0.6B MBPP 0.273 → 0.417 (**+14.4 pp**, 10 runs) [T, 2605.30478 Table 1]; Qwen3.5-0.8B search EM 0.092 → 0.352 (**+26 pp**, 3 seeds) [T, 2609.28765 §6.1]; 1.5B math avg 48.9 → 56.3 (**+7.4**) at **96 A40-h** [T, 2503.16219 Tables 1–2] | 96 A40-h = the only published number in this class | **L** — \(10^2\) GPU-h | **only above ~10% base solve rate**; at 95M the one measured attempt went 1.82 → 1.59 [T, 2606.22189] |
| 6 | **Process reward vs outcome reward** (PRM/PPM, step labels) | step labels (human, MC, or privileged info) + a second model | PRM 78.2% vs ORM 72.4% vs majority 69.6, best-of-1860 on MATH (7B-class) [T, 2305.20050 §1, Fig. 3]; process-graph curation +2.8 size-matched / +10.8 full at 32B [T, 2605.21996 Table 1]; decaying process rewards +45.4% relative over outcome-only GRPO on >60-turn tasks [T, 2610.35497 abstract] | not reported | **L** — labels or a second 95M model | **no learned PRM at 95M**; auto-labeled PRMs are measurably noisy [T, 2501.07301] |
| 7 | **Self-play with a verifier, no external data** (AZR) | code executor as both task-validator and answer-verifier | AZR on Qwen2.5-7B/7B-Coder, 3B/8B/14B variants, no external data, SOTA among zero-setting models [T, 2505.03335 abstract, §4] | not reported | **XL** | **mechanism yes, budget no**; the executor is our `verifiers.py`, the training is \(10^3\) GPU-h |
| 8 | **Self-rewarding / LLM-judge self-play** | a judge (same or frozen model) | **negative, measured:** judge pass 0.72 → 0.94 while true accuracy stays 0.20, 3 seeds [T, 2607.05904 abstract]; self-rewarding at 70B reports leaderboard wins [T, 2401.10020 abstract] but the judge is the same family being trained | — | — | **rejected** — do not build |
| 9 | **DPO / preference tuning** | preference pairs | 8B Tulu 3 +4.0 avg [T, 2411.15124 Table 4]; no ≤1B ablation found (prior report, §2.1 row 7) | not reported | **S** | **unmeasured at 1B**; style, not agency |

### 2.1 Rate, where a rate can be computed at all

| Method | Best delta | Published compute | Rate | Note |
|---|---|---|---|---|
| Expert iteration (SWE-Gym) | +9.0 resolve at 7B | not reported | cannot compute; **491 traces** is the denominator that matters | SFT on 491 trajectories is O(1) GPU-hour even at 7B [C] |
| Sweep RLVR (Open-RS) | +7.4 avg at 1.5B | 96 A40-h | **7.7 pp / 100 A40-h** [C] | needs an R1-distilled base |
| ProRL | +14.7 math at 1.5B | 16,000 H100-h | **0.09 pp / 100 H100-h** [C] | boundary expansion, not sharpening |
| AWM env RL | +12.1 BFCLv3 at 8B | not reported | cannot compute | 526 envs × 3,315 tasks |
| PRM best-of-N | +5.8 over ORM (78.2 vs 72.4) at 7B | not reported | cannot compute | needs 1,860 samples/problem at eval — inference cost, not training |

**The ordering that survives scrutiny:** for *agentic* capability at ≤1B, the methods with a
programmatic verifier and a small number of verified traces (1, 3) dominate everything that
trains a reward model or runs on-policy RL (5, 6, 7) by one to two orders of magnitude in
cost per point. That is the opposite of the 2026-11-08 report's ordering for *math*, and the
difference is real: math needs many samples per prompt to find a correct path; tool calls are
verified per step and per trace, so one correct trace is worth a thousand failed rollouts.

### 2.2 Per-method notes (the part the table cannot hold)

**#1 Expert iteration.** The loop is: sample K, keep what the verifier accepts, fine-tune,
repeat. STaR needs the model to already succeed sometimes (it trains only on rationales that
reached the right answer) [T, 2203.14465 abstract]. RFT shows the gain grows with the number
of *distinct* reasoning paths and is **larger for weaker models** [T, 2308.01825 §1] — the
one published scaling statement that favours a small base. ReST-EM adds iterations and beats
human-data-only fine-tuning, but is measured on PaLM-2 S/S*/L, **not below 1B** [T, 2312.06585
abstract, §4]. SWE-Gym is the strongest agentic point: 491 sampled trajectories, 7B +9.0
resolve, and the failure modes it fixes are the ones we would have (empty patch 40.3 → 29.7%,
stuck-in-loop 47.0 → 31.0%) [T, 2412.21139 Table 3].

**#2 Memory without weights.** Voyager's skill library is executable code, retrieved by
description, and it compounds because skills compose [T, 2305.16291 abstract]. ExpeL extracts
natural-language insights and needs no parameter updates at all [T, 2308.10144 abstract]. AWM
induces reusable workflows and reports +24.6% / +51.1% relative on Mind2Web / WebArena, with
online induction beating offline [T, 2409.07429 abstract]. AutoFyn is the 2026 formalisation:
a frozen model, a persistent state, a task-grounded verifier, and the reward distilled into
the state instead of the weights [T, 2609.05446 abstract]. **All of these run on a frontier
model.** At 95M the store can still be retrieved and pasted into a 1024-byte window, but the
model's ability to *use* an instruction is the binding constraint, not the store — unmeasured
[S].

**#3 Synthetic verified data.** APIGen's 1B result is the most relevant single number in this
report: 60,000 execution-verified entries, 3,673 APIs, and a 1B model that outranks
GPT-3.5-Turbo on BFCL [T, 2406.18518 abstract, §5]. ToolACE reaches 26,507 APIs through a
self-evolution synthesis loop with dual rule+model verification [T, 2409.00920 abstract].
P2T's decomposition is the useful methodological result: at 32B, **quality +2.8, quantity
+8.0** [T, 2605.21996 Table 1] — so "curate harder" is not a substitute for "collect more",
once a verifier exists.

**#4 Environment synthesis.** AWM generates 1,000 code-driven MCP environments with database
state, averaging ~2,000 lines and 35 tools each, and uses GPT-5 for synthesis [T,
2602.10090 §4.3, §5.1]. PhantomEnvironments shows the opposite end: **rule-generated fictional
worlds, zero LLM, zero marginal cost**, still transfer 1.7–2.2× [T, 2609.40221 §1]. The
lesson for a 16 GiB card is that environment *diversity* is purchasable with code, not with a
teacher. What is not purchasable is the RL that consumes them: 4B–14B policies, 526 envs,
unreported compute [T, 2602.10090 §5.1].

**#5 RLVR.** Already settled in the prior report: works as a sharpener at 0.6–1.5B when the
base solves 10–50%, costs \(10^2\)–\(10^4\) GPU-h, and at 95M has one measured negative
[2606.22189]. New in this pass: the 0.8B search result is the cleanest evidence that the
*tool schema* is the learnable part (+26 pp EM), and that an EM-only reward is the worst
variant at every seed [T, 2609.28765 §6.1]. Nothing here changes the verdict for 95M.

**#6 Process vs outcome.** The pro-PRM evidence is real but narrow: PRM 78.2 vs ORM 72.4 vs
majority 69.6 at best-of-1860 on MATH [T, 2305.20050 Fig. 3], and decaying process rewards
+45.4% relative on tasks over 60 turns [T, 2610.35497 abstract]. The anti-PRM evidence is
equally real: Monte-Carlo step labels are *inferior* to LLM-judge and human labels because a
weak completer assigns credit from futures it cannot reliably produce [T, 2501.07301 §1];
PRMs drift from process to outcome assessment, concentrating minimum scores on final-answer
steps [T, 2501.07301 abstract]; and general/open PRMs struggle to exceed **55%** average step
accuracy on tool-use trajectories while tool-specialised PRMs beat them [T, 2601.12294 §5.3].
PRInTS argues the same from the information-seeking side: short-reasoning binary PRMs cannot
score tool interactions or long contexts [T, 2511.19314 abstract]. **Verdict: process signal
helps when it is grounded in privileged ground truth (the reference patch, the executed
state), and hurts when it is guessed.**

**#7 Self-play.** AZR's contribution is architectural, not numerical: a code executor serves
as both task validator and answer verifier, so no human data enters [T, 2505.03335 abstract].
It is trained at 7B (plus 3B/8B/14B variants) with batch 64×6 [T, §4]. At 95M the proposer is
the same weak model; the prior self-play report's determinism gate is the cheap part to keep
(`selfplay/loop.py`).

**#8 Self-rewarding.** Rejected on measurement, not taste. The 70B claim [T, 2401.10020
abstract] and the 0.72→0.94-vs-0.20 audit [T, 2607.05904 abstract] are not in conflict: the
first measures a judge's *preference*, the second measures *correctness* against a hidden
anchor. At 95M the judge is the network being trained.

---

## 3. Self-improvement reality check

### 3.1 What replicates

| Claim | Replicates? | Evidence |
|---|---|---|
| "Train on your own verified successes" improves the model | **yes, repeatedly** | STaR 6B [2203.14465]; RFT 7B +13.4 [2308.01825]; SWE-Gym 7B +9.0 with 491 traces [2412.21139]; SEAL 1B ARC 20% → 72.5% [2506.10943 Table 1] |
| "A stronger teacher's traces are better than your own" | **sometimes; teacher strength buys trajectory validity, not transfer** | P2T: two teachers (Qwen3-C-480B, GLM-5-FP8) give +10.8 / +10.6 at 32B — nearly identical [2605.21996 Table 1] |
| "Self-play without data discovers new capability" | **only with an executable verifier** | AZR (code executor) [2505.03335]; Phantom (rule worlds) [2609.40221]; SAGE (AppWorld tests) [2512.17102] |
| "The model can judge itself and improve" | **no** | judge pass 0.72→0.94 at true accuracy 0.20 [2607.05904]; intrinsic self-correction degrades reasoning [2310.01798] |
| "Skill libraries compound without weight updates" | **yes, at frontier-model scale** | Voyager 3.3× items [2305.16291]; AWM +24.6/+51.1% relative [2409.07429]; AutoFyn qualitative [2609.05446] |
| "Memory replaces learning" | **no; it replaces forgetting** | Voyager's own framing: skills "alleviate catastrophic forgetting" [2305.16291 abstract] |

### 3.2 Which claims are teacher artifacts

Three distinct artifacts, and they are easy to confuse:

1. **Frontier teacher in the loop.** Voyager is GPT-4 via black-box queries with no weight
   updates [T, 2305.16291 abstract]. AutoFyn is a harness over provider models [T,
   2609.05446 abstract]. SAGE's SFT stage uses expert experience generated by Claude, and the
   paper says plainly that "merely imitating expert behaviors is insufficient" — the RL stage
   is what beats GRPO [T, 2512.17102 §4.2]. If you remove the teacher and keep the loop, the
   published delta does not survive.
2. **Frontier teacher in the *data*.** AWM synthesises environments with GPT-5 [T,
   2602.10090 §5.1]; APIGen's 1B model is trained on data from stronger models [T,
   2406.18518]. The artifact is real but *transferable*: the dataset is the product, and a 1B
   student can keep it.
3. **Frontier judge in the *metric*.** Self-Rewarding reports AlpacaEval wins judged by
   GPT-4 [T, 2401.10020 §3.2]. Any "self-improvement" measured by a same-family judge is
   measuring preference drift, not capability — the 0.72→0.94 audit is the proof.

**The test to apply before believing any self-evolving claim:** name the external signal.
If it is a program's exit code, an exact-match anchor, or a held-out benchmark, the claim can
be true. If it is the model's own verdict, it is measured false.

### 3.3 The smallest honest loop

**Weak model + hard verifier + expert iteration, where "self" appears only in sampling.**

```
for round in 1..3:
    sample K=4..8 completions per prompt from the current policy
    keep completions the verifier accepts (exit code 0 / exact match / parse OK)
    if accepted < 20% of prompts: stop (no signal)
    SFT 1-2 epochs on accepted only, prompt masked
```

Measured both ways at the smallest scales published:

| Point | Setup | Result | Source |
|---|---|---|---|
| **works, 1B** | Llama-3.2-1B-Instruct, self-edits, reward = downstream success on an 11+8-task ARC subset | ARC success **20% → 72.5%** (ICL 0%, oracle 100%) | [T, 2506.10943 Table 1] |
| **works, 7B** | Qwen2.5-7B, SEAL self-edits, 2 ReST rounds, 50 contexts per batch | SQuAD no-context **33.5 → 47.0**, beats GPT-4.1 synthetic after 2 iterations | [T, 2506.10943 §4.2] |
| **works, 7B, agentic** | Qwen2.5-Coder-7B, **491** SWE-Gym trajectories | SWE-bench Lite **1.0 → 10.0** | [T, 2412.21139 Table 3] |
| **works, 7B, math** | LLaMA-7B, RFT from own samples | GSM8K **35.9 → 49.3** | [T, 2308.01825 §1] |
| **works, 3.8B, hard** | rStar-Math, MCTS + PPM, 747k problems, 4 rounds | MATH **41.4 → 86.4** | [T, 2501.04519 abstract] |
| **fails, 135M** | L20-Edu-135M, GRPO on GSM8K, single run | **1.82 → 1.59 / 1.21** | [T, 2606.22189 abstract] |
| **fails, 1B** | OLMo-2-1B, 13 checkpoints of RLVR | first-try +3.3 pp, pass@256 94.5 → 96.9, error probes invariant | [T, 2611.02574 abstract] |
| **fails, any size** | self-judge self-play | true accuracy flat at 0.20 while judge pass → 0.94 | [T, 2607.05904 abstract] |

**Where the boundary sits.** The loop works when the verifier is a program and the policy can
already pass sometimes. It stops working when either condition fails. The 1B SEAL point works
because the reward is a *held-out accuracy check*, not a self-verdict. The 135M point fails
because pass@1 ≈ 2%: with K=8, `0.98^8 = 0.85` of prompts yield no accepted sample [C], so
85% of the batch is discarded and the gradient is a handful of traces per epoch.

**Minimum viable numbers, from the cited work:** 491–2,000 prompts with a verifier
[2412.21139; 2308.01825], K = 4–8, 1–3 rounds, accepted fraction ≥ 20%, 1–2 epochs. Below
that, the honest answer is "no loop yet — build the verifier and the prompt set first".

---

## 4. Data: what actually moves the number

### 4.1 Agent traces vs raw text

Agent traces are not "more text". They are *supervision on a policy*, and the measured
difference is that a small number of **verified** traces beats a large amount of unverified
text for the same token budget. The cleanest comparison is within P2T: same teacher, same
scaffold, same student, only the curation differs — outcome-filtered 39.6 vs process-curated
42.4 size-matched vs 50.4 full [T, 2605.21996 Table 1].

Two counterweights, both measured:

- **Agent-only data destroys general ability.** AgentTuning: training on agent instructions
  alone improves the held-in set but "struggles to generalize"; mixing general instructions
  is what keeps held-out performance [T, 2310.12823 §5, Table 5]. Their dataset is only
  **1,866 trajectories** across six tasks [T, §2, Table 1].
- **Raw text is still what teaches the language the traces are written in.** At byte level
  this is stronger, not weaker: the model has no tokenizer prior for `{`, `"`, `\t`. The
  pretraining mix is the only place it learns them.

**Practical ratio for this repo:** keep 70–85% general text / 15–30% agent-format in the SFT
mix, and *measure* held-out BPB on plain text as the guard. The number is my inference from
AgentTuning's ablation [S]; the direction is `[T]`.

### 4.2 Per-task verifiable data: how many traces

| Source | Count | Measured effect | Note |
|---|---|---|---|
| SWE-Gym | **491 trajectories** (from 2,438 instances) | 7B +9.0 resolve; 32B +12.3 | the highest delta-per-trace published |
| AgentTuning | 1,866 trajectories, 6 tasks | generalisation needs the general-data mix | [2310.12823] |
| APIGen | 60,000 verified entries, 3,673 APIs | 1B beats GPT-3.5-Turbo on BFCL | [2406.18518] |
| ToolACE | 26,507 APIs | dual-layer verification is the load-bearing part | [2409.00920] |
| Explorer (web) | 94k trajectories, 49k URLs, 720k screenshots | largest web set; multimodal | [2502.11357] |
| AWM | 1,000 envs, 3,315 tasks (526 used) | env count scaling, §4.3 | [2602.10090] |

**No published study gives "N traces → Δ" as a curve at ≤1B.** The closest is RFT's
augmented-data-count ablation, which says gains grow with *distinct reasoning paths* and
diminish with more of the same [T, 2308.01825 §3.3]. Treat the counts above as existence
proofs of "hundreds are enough to move a 7B", not as a scaling law.

### 4.3 Environment synthesis: the count is the knob

AWM's scaling curve is the most actionable data number in this report: **10 environments
degrades performance** (overfitting to the distribution), **100 gives substantial gains**, and
**526 continues to improve** [T, 2602.10090 §6.4]. PhantomEnvironments adds that hop count
(not constraints or comparisons) drives transfer, and that the simplest rule-generated worlds
already work [T, 2609.40221 §1]. SPADE (prior report, 2608.19197) reaches the same conclusion
from a 30B env-designer: code-as-environment exceeds three hand-written templates.

**For a 16 GiB box:** 50–100 code-generated environments with deterministic state is a
weekend of Python, not a training problem. Below ~30, the AWM curve says you are more likely
to overfit than to learn.

### 4.4 Dedup and quality filters

| Lever | Measured effect | Source |
|---|---|---|
| Near-dup dedup (MinHash/LSH) | verbatim-memorised tokens in unprompted generation **>1% → 0.1%** | [T, 2107.06499 Table 4] |
| Educational-quality classifier | "dramatically better" MMLU/ARC at 350B-token scale (direction, no delta in abstract) | [T, 2406.17557 abstract] |
| Model-based filtering | "key to assembling a high-quality corpus" across 412M–7B | [T, 2406.11794 abstract] |
| Benchmark decontamination | required; 13-gram rule at SmolLM2, LCS at Tulu 3 (prior report §3.3) | inherited |
| Recursive synthetic without real data | tail collapse; irreversible forgetting of low-probability events | [T, 2305.17493 abstract] |

The gap: **nobody has measured dedup or quality filtering on *agent traces*.** The pretraining
numbers transfer in direction only. At 6k traces dedup is nearly free, so do it and do not
budget for it.

### 4.5 Synthetic:real ratio

The only clean measured ratio is phi-1, and it is a pretraining result: **6B filtered real
code + <1B synthetic textbooks + 180M synthetic exercises** (synthetic ≈ 16% of tokens) →
1.3B params, HumanEval 50.6%, MBPP 55.5% [T, 2306.11644 abstract, §2]. ReST-EM is the
post-training extreme: 100% model-generated, verified by a checker, and it beats
human-data-only fine-tuning on MATH/APPS [T, 2312.06585 abstract] — but at PaLM-2 scale, not
1B.

**Honest state:** there is no 1B post-training study of synthetic:real ratio, and none at all
for byte-level models. The defensible rule is the one the evidence supports: synthetic is safe
when every sample is *verified by something outside the model*, and the mix keeps real text
for the language prior. Mark any specific percentage as `[S]`.

### 4.6 Where the evidence is thin (explicit list)

1. **No agentic post-training at ≤1B with multi-turn environments.** The floor is 0.6B
   single-turn code [2605.30478], 0.8B search [2609.28765], 3B Phantom [2609.40221].
2. **No PRM below 7B.** ToolPRMBench evaluates PRMs as judges; nothing trains a 95M–1B PRM.
3. **No GPU-hour reporting** in AWM, P2T, SAGE, SWE-Gym, APIGen, AZR. Rates in §2.1 are
   therefore sparse; the ranking leans on cost class, not on measured hours.
4. **No byte-level agentic anything** (§5).
5. **No trace-count scaling law.** Only RFT's distinct-path ablation.
6. **No study of how many *distinct tools* a small model can hold.** Our task suite design
   has no citation to lean on.

---

## 5. The byte-level constraint, priced

### 5.1 What the live model actually is

`vocab=256`, `max_seq_len=1024`, `byte_patch=false` in the flagship config [M,
`configs/flagship_base.toml:8`, `config.py:406,1009`]. So **1 position = 1 byte**, and a
window holds 1,024 bytes. `byteflow_patch_bytes=8` exists but is only active when
`byte_patch=true` (e.g. `configs/arch_dense_byteflow8.toml:50`).

### 5.2 The arithmetic of a tool call

| Encoding | Bytes | Positions at 1024 B | Positions with 8-byte patches |
|---|---|---|---|
| JSON `{"tool":"calc","args":{"expr":"17*23"}}` | **39** | 39 | 5 |
| Compact `\x02calc\t17*23\n` | **12** | 12 | 2 |
| Same call, BPE (~3.5 B/token) | 39 | ~11 tokens | n/a |

[C, counted by hand]. **34 of the 39 JSON bytes are syntax** — braces, quotes, the key names.
The model spends ~87% of that call's loss learning punctuation it already emits correctly,
and one wrong byte makes the call unparseable. The compact protocol cuts positions by 3.25×
and syntax share from 87% to ~58% (`\x02`, `\t`, `\n` are 3 of 12).

This also changes the earlier report's "3.5–4.5× longer" claim: that is true for the
**unpatched** byte stack we run today, but with `byte_patch=8` a byte stream is roughly
*shorter* than BPE for punctuation-heavy text (5 patches vs ~11 tokens above). The fix is a
config, not a research program — but only after the patch path is trusted.

### 5.3 What the published byte-level work did

| Work | Post-training | Protocol for structure | Numbers |
|---|---|---|---|
| BLT [2412.09871] | **none** — pretraining + zero/few-shot | entropy patches; no chat/tool format | FLOP-matched up to 8B, 4T bytes; no SFT |
| UTF8Tokenizer [2510.16987] | chat template defined, **no SFT results** | **C0 control bytes** repurpose ASCII control codes for message blocks, tool calls, thinking spans; no token 256 | 14× faster tokenization; 8× less host-device transfer |
| ByteFlow [2603.03583] | none | adaptive compression-driven segmentation | pretraining only |
| Broken Tokens [2506.19004] | robustness study | non-canonical tokenization | 90.8% retained at char level; char-level +14% on string manipulation/code |

**The one usable protocol is UTF8Tokenizer's C0 table.** Bytes `0x00–0x08` are invalid as
standalone UTF-8 and essentially absent from natural text, so they are free delimiters in a
256-vocab model — no new embedding rows, no tokenizer, no claim broken. That is the
recommendation: one control byte per role/turn boundary, tab-separated arguments, newline
terminator, strict parser.

**What it costs, honestly:** (i) syntax loss share is high until the protocol is compact;
(ii) there are no atomic tokens, so a beam or a constrained decoder cannot enforce JSON
grammar — the verifier must reject, not repair; (iii) at 1,024 bytes the trace must be
*short*: a 39-byte call plus a 200-byte observation is 6 calls per window, and a real file
listing will not fit. Either extend `max_seq_len` to 4,096 for post-training, or cap
observations at ~300 bytes. This is a data-loader decision, not a model change.

---

## 6. What to run on this box

### 6.0 Cost basis

| Quantity | Value | Derivation |
|---|---|---|
| Training throughput | 79.5 kB/s | measured [M] |
| Step | 24 × 1,024 B = 24,576 B, 0.309 s | [C] |
| 1 GB of training | 40,690 steps = **3.49 h** | [C] |
| Generation (rollouts) | **0.5 kB/s** batch-1 (range 0.3–1.0) | [S], inherited assumption |
| 1 MB of generation | 33 min at 0.5 kB/s | [C] |
| 1B-parameter multiplier | ~8× training hours, ~1.5–2× generation | [S] — params ×10, throughput ~÷8 |
| **Budget for the three experiments** | **15 GPU-h + 50% contingency = 22 h** | [C] — about three 2 GB base runs |

Every experiment below is sized so that **training is minutes and the budget is generation +
eval**. That is the correct shape at 95M: 6,000 traces × 600 B × 2 epochs = 7.2 MB = 91 s of
training.

### 6.1 E1 — Byte-protocol agent traces (the format experiment)

| Field | Value |
|---|---|
| Question | Can a 95M byte model learn to emit a *parseable, compact* tool call at all? |
| Data | 6,000 traces, avg 600 B: 4,000 scripted (50 code-generated tasks × 4 turns × 20 paraphrases), 2,000 from a local 0.5–1.5B teacher. Compact protocol only (`\x02` role bytes). 15% general text mixed in. |
| Cost | generation 3.6 MB @ 0.5 kB/s = **2.0 h**; training 7.2 MB = **0.03 h**; eval 500 prompts × 8 samples × 200 B × 3 seeds = 1.4 h → **~4 GPU-h** |
| Metric that decides | **exact-parse rate** (strict parser, no repair) and task pass@1 on a held-out 500-prompt suite |
| PASS | parse ≥ 95%, pass@1 ≥ +8 pp over the pretrain checkpoint, held-out BPB ≤ +0.05, 3/3 seeds same sign |
| ABORT | parse < 80% after 2 epochs, or BPB > +0.10 → the protocol is too expensive for 256-way byte prediction; drop to single-byte commands (`calc 17*23`) before spending anything else |

### 6.2 E2 — Skill library with no weight updates (the cheap self-improvement test)

| Field | Value |
|---|---|
| Question | Does retrieving *verified past traces* beat the same model without retrieval, at zero training? |
| Data | the E1 trace pool, deduplicated, indexed by task signature; retrieve top-1 into the 1,024-byte window |
| Cost | 500 tasks × 4 attempts × 400 B = 0.8 MB = 27 min per arm; 2 arms × 3 seeds = **2.7 h**; plus harness → **~3 GPU-h** |
| Metric | pass@1 with retrieval vs without, same checkpoint, same seeds |
| PASS | ≥ +5 pp and non-overlapping 95% CI across 3 seeds |
| ABORT | < +2 pp → a 95M model cannot use an in-context trace; this closes the "memory without weights" lane for this scale and is worth knowing before building a store |

This is the cheapest experiment in the report and the only one that can *fail informatively*
in three hours. Run it before E3.

### 6.3 E3 — Expert iteration on the E1 checkpoint (the real loop)

| Field | Value |
|---|---|
| Precondition | E1 parse ≥ 95%; sample 500 prompts × 8; **keep prompts with 1–7 of 8 correct**; if < 100 survive, abort |
| Data | 3 rounds × 2,000 prompts × 8 samples × 200 B = 3.2 MB per round; keep verified-correct only; 1 epoch |
| Cost | per round: generation 1.8 h + training 80 s + eval 0.5 h = 2.4 h; 3 rounds = 7.2 h; prompt mining 1 h → **~8 GPU-h** |
| Metric | pass@1 on the held-out suite (primary); **pass@8 coverage** (secondary — the 1B pencil result shows first-try and coverage diverge [2611.02574]); parse rate; BPB |
| PASS | pass@1 ≥ +5 pp over E1, 3/3 seeds, parse ≥ 95%, BPB ≤ +0.05 |
| ABORT | round-2 delta ≤ 0, or accepted fraction < 20%, or pass@8 < 5% (no signal — the 135M failure regime [2606.22189]) |

**Total:** 4 + 3 + 8 = **15 GPU-h**, 22 h with contingency. At 1B, ×8 on training but training
is 0.03 h, so the real 1B cost is generation ×1.5–2 ≈ **25–30 GPU-h** [S].

**What is deliberately not in the plan:** RLVR (prior report's E3, expected to fail at 95M),
a learned PRM (no ≤1B evidence; §4.6), env-synthesis RL (XL cost, §2 row 4), and any
self-judge loop (measured negative, §3).

### 6.4 The one thing to measure regardless of which arm wins

Report **parse rate on held-out syntax** and **BPB on plain text** together, every time. The
failure mode unique to this stack is a model that gets better at emitting delimiters while
getting worse at English, and only the pair detects it. `[M]` precedent: the repo's prior SFT
report §3.4 documents exactly this divergence at 1.7B (SmolLM2 ARC 60.5 → 51.7).

---

## 7. Where this contradicts the prior reports

### 7.1 `2026-11-08-post-training-at-1b.md`

1. **Ranking.** That report ranks SFT > re-distillation > code RLVR on pp/GPU-h for *math
   capability*. For *agentic* capability the order changes: verifier-filtered trace SFT is
   first by a wide margin because 491 traces move a 7B by +9.0 [2412.21139] while RLVR needs
   \(10^2\) GPU-h for a comparable move [2503.16219]. Both are right in their own objective —
   the objective, not the method, decides the ordering.
2. **Throughput.** 79.5 kB/s, not 60: every cost in that report is 25% too high [M].
3. **Byte length.** "3.5–4.5× longer" holds only while `byte_patch=false`. With 8-byte
   patches a punctuation-heavy stream is *shorter* than BPE (§5.2). The report's §6.1 table
   should be read as "unpatched byte stack".
4. **"Selection is the lever."** True for SFT mixtures at 1.7B (28-point GSM8K swing from data
   choice alone). Not true once a verifier exists: P2T decomposes quality +2.8 vs quantity
   +8.0 at 32B [2605.21996]. Selection is the lever *when the signal is absent*; with a
   verifier, volume still pays.
5. **"Teacher 0.5–1.5B with a verifier, not a 70B API."** Half right. A weak teacher's traces
   are *valid* if a verifier filters them, so the teacher can be weak — but every strong
   agentic dataset in this pass uses a frontier teacher for *trajectory validity* [2602.10090,
   2605.21996, 2406.18518]. The cheap substitute is not a weak teacher; it is a **scripted
   environment + verifier** (Phantom: zero LLM, 1.7–2.2× [2609.40221]).

### 7.2 `2026-11-08-test-time-iteration.md`

No contradiction; it is orthogonal (test-time depth ≠ post-training). One implication: its
D7 rule ("if BPB is flat while reasoning improves, stop") applies here too — an agentic gain
with a BPB regression is not a win for this repo's stated objective.

### 7.3 `2026-10-15-agentic-post-training.md` and `2026-10-18-self-play.md`

Those sweeps are abstract-level and 2026-10-dated. Two of their theses are confirmed with
full-text numbers here: the gate/verifier beats reward shaping (SAGE's RL beats its own
expert-imitation SFT [2512.17102 §4.2]; ToolPRMBench's general PRMs < 55% [2601.12294]) and
self-rewarding is broken (their 2607.05904, re-verified here). One is narrowed: their "small
models need stabilisers" list is about RL; for expert iteration the stabiliser is just the
accepted-fraction floor, which is cheaper.

---

## 8. VERIFIED (sourced)

Every line below is a number or claim read from the cited source in this pass.

**Expert iteration / rejection sampling**

1. STaR: iterative rationale bootstrapping; trains only on rationales that reached the correct
   answer; GPT-J 6B; "comparable to fine-tuning a 30× larger model" on CommonsenseQA —
   [T] 2203.14465 abstract.
2. RFT: LLaMA-7B GSM8K SFT 35.9 → 49.3 (+13.4) with samples from multiple models; gains larger
   for weaker models; gains grow with distinct reasoning paths — [T] 2308.01825 §1, §3.3.
3. ReST-EM: generate → filter by binary feedback → fine-tune → repeat; PaLM-2-S/S*/L; beats
   human-data-only fine-tuning on MATH and APPS; ablates number of solutions, problems,
   iterations — [T] 2312.06585 abstract, §4.
4. SWE-Gym: 2,438 instances, 11 repos; 491 sampled trajectories; 7B SWE-bench Lite resolve
   1.0 ± 1.0 → 10.0 ± 2.4; empty patch 40.3 → 29.7%; stuck-in-loop 47.0 → 31.0%; 32B 3.0 →
   15.3; verifier-augmented 32.0% / 26.0% on Verified/Lite — [T] 2412.21139 abstract, Table 3.
5. SAGE: Qwen2.5-32B-Instruct, AppWorld; SFT 41.7 SGC → SAGE 60.7 (+19.0) on Test Normal;
   GRPO 51.8; SGC +8.9 over GRPO; −59% tokens, −26% steps; expert SFT data from Claude —
   [T] 2512.17102 Table 1, §4.2.
6. SEAL, two domains. Few-shot ARC: Llama-3.2-1B-Instruct, success 72.5% (SEAL) vs 20%
   (TTT + self-edit without prior RL) vs 0% (ICL) vs 100% (oracle) — [T] 2506.10943 Table 1,
   §A.1. Knowledge incorporation: Qwen2.5-7B, 2 ReST rounds, batch of 50 contexts; SQuAD
   no-context 33.5% → 47.0%, beating GPT-4.1 synthetic data after two iterations — [T] §4.2,
   Fig. 4.
7. rStar-Math: MATH 58.8 → 90.0 (Qwen2.5-Math-7B), 41.4 → 86.4 (Phi3-mini-3.8B); 747k
   problems, 4 rounds, MCTS + process preference model — [T] 2501.04519 abstract.

**Synthetic data / tool calls**

8. APIGen: 60,000 entries, 3,673 APIs, 21 categories; xLAM-1B ranks 24th on BFCL and beats
   GPT-3.5-Turbo-0125, Claude-3 Haiku, Mistral-large; 7B beats several GPT-4 variants —
   [T] 2406.18518 abstract, §5.
9. ToolACE: 26,507 APIs; multi-agent dialog synthesis; dual-layer rule+model verification —
   [T] 2409.00920 abstract.
10. P2T: 1.8k curated SWE-Gym instances; 32B 39.6 → 42.4 (size-matched, +2.8) → 50.4 (full,
    +10.8); 14B 36.0 → 37.6 (+1.6) → 43.2 (+7.2); cost per instance −$0.14; two teachers give
    +10.8 / +10.6 — [T] 2605.21996 Table 1.
11. phi-1: 1.3B, 6B filtered real code + <1B synthetic textbooks + ~180M synthetic exercises
    (<7B tokens total), 8×A100 4 days; HumanEval 50.6%, MBPP 55.5% — [T] 2306.11644 abstract,
    §2.

**Environments**

12. AWM: 1,000 code-driven envs, ~2,000 LOC and 35 tools each, DB-backed; GPT-5 synthesis;
    trained on 526 envs / 3,315 tasks; BFCLv3 4B 54.92 → 64.50, 8B 53.83 → 65.94, 14B 61.25 →
    70.18; τ²-bench Pass@1 8B 26.44 → 33.45 (EnvScaler 39.39); MCP-Universe 8B 6.70 → 11.17;
    env scaling 10 → degrade, 100 → gains, 526 → more — [T] 2602.10090 §4.3, §5.2, §6.4.
13. PhantomEnvironments: rule-generated fictional worlds, no LLM, zero marginal cost;
    Qwen2.5-3B/7B, Llama-3.2-3B, Phi-4-mini; +1.7× on HotpotQA/2Wiki/MuSiQue and +2.2× on
    SynthWorlds/FRAMES relative to base; hop count drives transfer — [T] 2609.40221 §1.
14. AZR: one model proposes tasks and solves them, code executor validates both; Qwen2.5-7B and
    7B-Coder main runs, 3B/8B/14B variants; batch 64×6; no external data — [T] 2505.03335
    abstract, §4.

**Reward models / credit assignment**

15. Let's Verify Step by Step: PRM 78.2% vs ORM 72.4% vs majority voting 69.6% solved,
    best-of-1860 on a MATH subset — [T] 2305.20050 §1, Fig. 3.
16. Qwen PRM lessons: MC-estimated step labels inferior to LLM-judge and human labels because a
    weak completer assigns credit from unreliable futures; PRMs drift process → outcome, with
    minimum scores concentrated on final-answer steps; consensus filtering fixes part of it —
    [T] 2501.07301 abstract, §1.
17. ToolPRMBench: general/open PRMs often cannot exceed 55% average step accuracy on tool-use
    trajectories; tool-specialised PRMs (ToolPRM-GRPO) beat all non-API models and are far
    cheaper than API judges — [T] 2601.12294 §5.3, §5.4.
18. Fine-grained credit: decaying process rewards +45.4% relative over outcome-only GRPO on
    tasks averaging >60 turns; benefit is task-horizon dependent — [T] 2610.35497 abstract.
19. PRInTS: short-reasoning binary PRMs cannot score tool interactions or long contexts;
    generative multi-dimensional PRM proposed — [T] 2511.19314 abstract.

**Self-improvement limits**

20. Self-judge hacking: GSM8K/Qwen3, three seeds, judge pass 0.72 → 0.94 while true accuracy
    stays 0.20; errors transfer across judge families; an answer-committing judge cuts false
    positives 0.719 → 0.012 — [T] 2607.05904 abstract.
21. Intrinsic self-correction: LLMs cannot reliably self-correct reasoning without external
    feedback; performance sometimes degrades — [T] 2310.01798 abstract.
22. Self-Rewarding LMs: Llama-2-70B, three Iterative-DPO rounds, judged by GPT-4, reported to
    beat Claude 2 / Gemini Pro / GPT-4-0613 on AlpacaEval 2.0 — [T] 2401.10020 abstract, §3.2.
23. Gödel Agent: self-referential framework that modifies its own logic; claims recursive
    self-improvement without predefined routines — [T] 2410.04444 abstract. (No benchmark
    number read; treat as a claim, not evidence.)

**Data**

24. AgentTuning: 1,866 verified trajectories, 6 tasks; agent-only training improves held-in but
    generalises poorly; adding general instructions restores both — [T] 2310.12823 §2, Table 1,
    §5, Table 5.
25. Dedup: >1% of unprompted generated tokens are part of a 50-token training substring;
    drops to 0.1% after dedup (100k samples, 512 tokens) — [T] 2107.06499 Table 4.
26. FineWeb-Edu: 1.3T-token educational subset; "dramatically better" MMLU/ARC; dedup and
    filtering ablated individually — [T] 2406.17557 abstract, §3.
27. DCLM: 240T-token corpus, 412M–7B scale grid, 53 tasks; model-based filtering "key" to a
    high-quality corpus — [T] 2406.11794 abstract.
28. The Curse of Recursion: training on generated data causes irreversible defects, tails of the
    distribution disappear first — [T] 2305.17493 abstract.
29. τ-bench: gpt-4o succeeds on <50% of tasks and is inconsistent, **pass^8 < 25%** in retail;
    pass^k proposed as the reliability metric — [T] 2406.12045 abstract.
30. Explorer: 94k successful multimodal web trajectories, 49k URLs, 720k screenshots, 33M
    elements — [T] 2502.11357 abstract.
31. Voyager: GPT-4 black-box, no fine-tuning; 3.3× unique items, 2.3× distance, 15.3× faster
    milestones; skill library is executable code and "alleviates catastrophic forgetting" —
    [T] 2305.16291 abstract.
32. ExpeL: no parameter updates; natural-language insights extracted from experience and
    recalled at inference — [T] 2308.10144 abstract.
33. AWM (web): +24.6% Mind2Web, +51.1% WebArena relative; online induction generalises +8.9 to
    +14.0 absolute as distribution gap widens — [T] 2409.07429 abstract.
34. AutoFyn: frozen model, persistent state, task-grounded verifier; reward distilled into state
    not weights; qualitative results (IMO 2026, Spider 2.0, 16 advisories) — [T] 2609.05446
    abstract. **No benchmark table read.**

**RLVR at small scale (re-verified from the prior report)**

35. Code RLVR: Qwen3-0.6B MBPP 0.273 → 0.417 (+14.4 pp), 10 runs — [T] 2605.30478 Table 1.
36. Search RLVR: Qwen3.5-0.8B EM 0.092 → 0.352, 3 seeds; EM-only worst at every seed —
    [T] 2609.28765 §6.1.
37. Open-RS: 1.5B, 7k samples, 6 rollouts, 4×A40 24 h = 96 A40-h; avg 48.9 → 56.3 —
    [T] 2503.16219 Tables 1–2.
38. L20-Edu-135M: GSM8K 1.82% → 1.59% / 1.21%, single run, author calls it a failure mode —
    [T] 2606.22189 abstract.
39. OLMo-2-1B pencil: first-try 58.0 → 61.3, pass@256 94.5 → 96.9, error probes invariant —
    [T] 2611.02574 abstract.
40. Priming base models for agentic post-training: 30B, 500B-token midtraining, SWE-V +3.8 with
    traces — [T] 2610.22019 Table 2 (via the prior report).
41. `p(1−p)` starvation of outcome RL: [T] 2610.34894 App. B.2 (via the prior report).

**Byte-level**

42. BLT: byte patches 4.4–6.1 bytes, FLOP-matched scaling to 8B / 4T bytes; **no SFT, no
    instruction experiments** — [T] 2412.09871 abstract (prior report's grep of the HTML).
43. UTF8Tokenizer: 256 IDs, no ID 256; all structure (padding, boundaries, conversation, tool
    calling, thinking spans) via **C0 control bytes**; 14× faster tokenization, 8× less
    host-device transfer; **no SFT results** — [T] 2510.16987 abstract, §3.1.
44. ByteFlow: adaptive compression-driven segmentation; pretraining — [T] 2603.03583 abstract.
45. Broken Tokens: instruction-tuned models retain up to 93.4% under random tokenization and
    90.8% under character-level; char-level improves string manipulation/code by up to +14% —
    [T] 2506.19004 abstract.

**Repo (measured)**

46. Throughput 79.5 kB/s at batch 24 + static max-autotune; 2 GB ≈ 7 h; 1 GB = 3.49 h;
    0.309 s/step at 24×1024 B; 11,646 steps/h — [M] + [C].
47. `vocab=256` (`config.py:406`), `max_seq_len=1024` (`configs/flagship_base.toml:8`),
    `byte_patch=false` default (`config.py:1009`), `byteflow_patch_bytes=8`
    (`config.py:1077`), patch arm enabled only in `configs/arch_dense_byteflow8.toml:50`.
48. Post/self-play code size: `post/` 2,707 LOC across 8 files; `selfplay/` 1,420 LOC across 5
    files — [M, `wc -l`].

---

## 9. SPECULATION (my inference, explicitly marked)

- **S1. At 95M, parseable tool-call emission is achievable; two-turn planning is not.**
  Rationale: format compliance is capacity-independent [T, 2502.02737's second-epoch +10.4
  IFEval], while the 134.5M GSM8K point (1.82%) says multi-step arithmetic is absent. I expect
  E1 to pass its parse bar and fail any bar that requires choosing between two tools
  conditionally. **Not measured anywhere.**
- **S2. The skill library will give < +2 pp at 95M.** Rationale: AWM/ExpeL/Voyager all run on
  models that can follow a multi-line instruction; a 95M byte model has ~1,024 bytes of
  context and no demonstrated instruction-following beyond IFEval-class constraints. If E2
  fails, that is the first measurement of the "memory without weights" floor. **Unmeasured.**
- **S3. Expert iteration will produce +3–6 pp at 95M if pass@8 ≥ 10%, and ≈0 otherwise.**
  Derived from the accepted-fraction arithmetic (`0.9^8 = 0.43` of prompts still yield a
  correct sample at p=0.1, vs `0.98^8 = 0.85` dry at p=0.02) [C]. **Unmeasured at 95M.**
- **S4. A compact byte protocol is worth 2–4× more than a JSON protocol at fixed compute.**
  From §5.2's 3.25× position reduction: the same 1,024-byte window then holds ~85 calls
  instead of ~26, so the same token budget sees 3× more task structure. **No paper compares
  protocols; this is arithmetic, not a benchmark.**
- **S5. The 1B multiplier is ~8× on training and ~1.7× on generation.** From 10× params and
  the observed memory-bound behaviour at 95M. **No measurement.**
- **S6. Data above ~20k verified traces will not help at 95M.** Rationale: capacity, not data,
  binds below ~200M (the L20-Edu and pencil results). **Unmeasured; would need a 3-point
  curve.**

---

## 10. Sources

Primary, all resolved in this pass (title checked):

1. 2203.14465 — Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning*.
2. 2308.01825 — Yuan et al., *Scaling Relationship on Learning Mathematical Reasoning with LLMs* (RFT).
3. 2312.06585 — Singh et al., *Beyond Human Data* (ReST-EM).
4. 2412.21139 — Pan et al., *SWE-Gym*.
5. 2512.17102 — *RL for Self-Improving Agent with Skill Library* (SAGE).
6. 2506.10943 — Zweiger, Pari, Agrawal, *Self-Adapting LLMs* (SEAL).
7. 2501.04519 — *rStar-Math*.
8. 2406.18518 — *APIGen*.
9. 2409.00920 — *ToolACE*.
10. 2605.21996 — *From Patches to Trajectories* (P2T).
11. 2306.11644 — Gunasekar et al., *Textbooks Are All You Need* (phi-1).
12. 2602.10090 — *Agent World Model* (AWM).
13. 2609.40221 — *PhantomEnvironments*.
14. 2505.03335 — Zhao et al., *Absolute Zero* (AZR).
15. 2305.20050 — Lightman et al., *Let's Verify Step by Step*.
16. 2501.07301 — *The Lessons of Developing Process Reward Models in Mathematical Reasoning*.
17. 2601.12294 — *ToolPRMBench*.
18. 2610.35497 — *When Fine-Grained Credit Assignment Helps and Hurts*.
19. 2511.19314 — *PRInTS*.
20. 2607.05904 — *More Convincing, Not More Correct* (self-play reward hacking).
21. 2310.01798 — Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*.
22. 2401.10020 — Yuan et al., *Self-Rewarding Language Models*.
23. 2410.04444 — Yin et al., *Gödel Agent*.
24. 2310.12823 — Zeng et al., *AgentTuning*.
25. 2107.06499 — Lee et al., *Deduplicating Training Data Makes Language Models Better*.
26. 2406.17557 — Penedo et al., *The FineWeb Datasets*.
27. 2406.11794 — Li et al., *DataComp-LM*.
28. 2305.17493 — Shumailov et al., *The Curse of Recursion*.
29. 2406.12045 — Yao et al., *τ-bench*.
30. 2502.11357 — *Explorer: Scaling Exploration-driven Web Trajectory Synthesis*.
31. 2305.16291 — Wang et al., *Voyager*.
32. 2308.10144 — Zhao et al., *ExpeL*.
33. 2409.07429 — Wang et al., *Agent Workflow Memory*.
34. 2609.05446 — *AutoFyn Technical Report*.
35. 2412.09871 — Pagnoni et al., *Byte Latent Transformer*.
36. 2510.16987 — *Back to Bytes: Revisiting Tokenization Through UTF-8*.
37. 2603.03583 — *ByteFlow*.
38. 2506.19004 — *Broken Tokens?*.
39. 2503.16219, 2605.30478, 2609.28765, 2610.22019, 2610.34894, 2606.22189, 2611.02574 —
    RLVR/scale rows re-verified from the 2026-11-08 report; used only for the numbers that
    report cites with the same table references.
40. 2502.02737 — Allal et al., *SmolLM2* (second-epoch IFEval +10.4); inherited from the prior
    report, not re-read in this pass.

Prior reports (extended, not duplicated):

- `docs/research/2026-11-08-post-training-at-1b.md`
- `docs/research/2026-11-08-test-time-iteration.md`
- `docs/research/2026-10-15-agentic-post-training.md`
- `docs/research/2026-10-18-self-play.md`

Dropped (could not resolve / wrong paper, therefore not cited): 2505.17988 (re-distillation
claim inherited from the prior report), 2508.05004 (R-Zero — not found), 2503.08764 (resolves
to a protein-SAE paper, not AgentGym), Cosmopedia (ID did not resolve to it — 2402.00838 is
OLMo).

---

## 11. Open questions

1. **What is the smallest base that can emit a parseable multi-byte tool call?** E1 answers it
   for 95M/byte/256-vocab. No published point exists below 1B.
2. **Does expert iteration beat RLVR at fixed GPU-hours on the same task?** No head-to-head
   published. E1→E3 gives one at 95M; the ordering may invert at 1B.
3. **How many distinct tools can a 1B model hold?** APIGen's 1B result is on 3,673 APIs but
   only evaluates function calling, not multi-turn composition.
4. **Is a 95M PRM worth training?** ToolPRMBench says general PRMs are weak; it does not test
   whether a *small specialised* PRM beats a rule-based verifier. Cheap to test (E2 arm).
5. **What is the right byte protocol?** C0 control bytes are the only published design
   [2510.16987] and have no SFT numbers. E1 is the first comparison (JSON vs compact).
6. **Does the AWM environment-scaling curve hold below 1B?** 10 → 100 → 526 is measured at
   4B–14B with GPT-5-generated envs; a 95M curve is unmeasured.
