# MiMo RL — what transfers to dormouse's agent-RL lane

**2026-10-15.** Owner directive this session: *"make the model as smart as possible self-taught, few tokens, byte-thrifty — RD: use the research to fine-tune the post-training plan."* Read with `docs/architecture/post-training.md` (the owner's plan — the STARTING position) and `docs/research/2026-10-15-agentic-post-training.md` (the 2026 sweep: CRR fork-replay 41.7 vs GRPO 36.7, VH-GPO 1.5B, verifier-first, EGGROLL, SEED). This doc adds **one family read at the source**, not a new sweep; where it contradicts the earlier doc it says so.

**Read-only, one file.** Only this file was written; `.bulba/plan.md` untouched, nothing committed. `docs-site/tools/manifest.mjs` has **not** been updated for this document and `tools/check_doc_refs.py` will flag it until that line lands (same debt as the sibling doc).

**Marks:** `(A)` read in the source's own text this session · `(L)` read from the live run's raw JSON (path quoted) · `(M)` authors' claim, not reproduced here · `(S)` SPECULATION, my inference, no source · `(web)` non-arXiv source.

**Where the sources are.** `https://mimo.xiaomi.com/rl/` **is a JavaScript SPA** — `webfetch` returns only `[overview] [metrics] [about]` + `reconnecting…`. I read it with Playwright, then read the raw JSON the page itself fetches: `https://mimo.xiaomi.com/rl/data.04135c89/{runs,status-pro,status-flash,series-pro,series-flash,notices,benchmarks}.json` `(L)`. The alphaxiv link in the task (`alphaxiv.org/overview/2609.mimo`) **404s** — verified this session. The Sept-2026 MiMo RL paper is **arXiv 2609.32577 (GAGAR, 2026-09-26)**; the run the page streams is **arXiv 2610.11959 (MiMo-V2.6, 2026-10-08)**. Neither is "2609.mimo".

---

## 0. TL;DR — six verdicts

1. **The 2026 MiMo RL recipe is GRPO with no value critic** — at 1.02T total / 42B active (2610.11959 §5.1; `A_i = R_i − R̄`, Eq. 3, no value head). The owner's plan says "RLVR PPO+value (not GRPO)"; the largest lab in this space ships the opposite. It does **not** rescue GRPO-as-is: MiMo's advantage is still one sequence-level number broadcast to every token (Eq. 1) — exactly the credit-misassignment criticism the earlier sweep documented. **PPO+value vs fork-replay stays open and is measured later.**
2. **What they add instead of a critic is grader compute — 12.7% of a $2.6M run** (§4.1). Groupwise agentic grading ranks *passing* trajectories and redistributes positive advantage among them (GAGAR, 2609.32577). That mechanism is the source of the token-thrift claim: the abstract says it "steers the model towards shorter, more token-efficient solutions"; GAGAR's abstract says "reduced trajectory-length growth" `(A)`.
3. **Honest counter-measurement: growth slowed, it did not stop.** Raw series: Pro `ctx_total_length/mean` **72.2K → 137K tokens over 30 steps (1.90×)**, Flash **71.6K → 148K (2.07×)**, `agg_turn/mean` roughly flat (47.5→43.6 / 47.3→68.5), step time 9.6Ks→23.2Ks (Pro) `(L)`. Any plan promising "byte-thrifty" from a length penalty alone is promising what MiMo did not achieve.
4. **Router freezing is a measured, free fix for MoE load collapse in RL** (§5.4): trainable router → CV 0.78→2.0, peak load 6×→16×, cold experts 0.5%→22% in 20 steps; restoring only the router's weights recovers balance **with benchmark performance unchanged**. Most directly transferable item for our `E=1024 topk-2 TSCT`.
5. **Nothing in this family was measured below 9B dense / 15B active.** Smallest RL-tested artifact is MiMo-V2.6-Distill-Qwen-9B (MIT, Qwen3.5-9B base, 25,582 downloads, modified 2026-09-22) `(web)`. Our lane is **57M active — 260× smaller** and byte-level: everything in §3 is a *shape* transfer, never an effect-size transfer.
6. **Build order this adds:** reward-shape node (§3.1) and router freeze (§3.2) are cheap and land before anything needing a grader; the grader itself is **deferred** — at 57M active the judge is either the network under training (the failure the earlier doc sources) or a second model this box cannot hold.

---

## 1. What the series did, exactly

**Family, in order.** `2505.07608` MiMo-7B (2025-05-12): 25T pretrain, **130K verifiable math+code problems**, a *test-difficulty-driven code reward* for sparse reward, strategic resampling `(A)`. `2506.03569` MiMo-VL-7B (2025-06-04): 2.4T tokens, **Mixed On-policy RL (MORL)**; the paper flags "challenges in simultaneous multi-domain optimization" `(A)`. `2601.02780` MiMo-V2-Flash (2026-01-06): 309B/15B active MoE, hybrid SWA 128-token window at 5:1, 27T tokens, **MOPD** (multi-teacher on-policy distillation) — the ancestor of our "selective OPD (SEED)" node `(A/M)`. `2609.32577` **GAGAR** (2026-09-26): quality-aware, sum-preserving advantage redistribution for code-agent GRPO, evaluated on Flash and Pro pre-RL checkpoints `(A)`. `2610.11959` **MiMo-V2.6** (2026-10-08): Pro 1.02T/42B active, Flash 310B/15B active, text 26T + omni 22T stage (Flash) `(A)` — the RL report, streamed at `mimo.xiaomi.com/rl/` `(A/L)`.

**Before RL** `(A)`: two-stage pretrain 32K→256K ctx; mid-training on an agent-centric mixture at 256K then 1M, **switching AdamW → Muown** (Muon + explicit row-norm control) on hidden matrices because AdamW loses data efficiency past the critical batch size in mixed-task RL — stated as *necessary to prepare the model for large-batch RL*, no loss spike reported; **MXFP4 QAT**; embeddings/head/router stay on AdamW. **SFT** `(A)`: for the 9B distill, **77.4B total tokens, only 27.2B loss tokens** (code 23.2 / cyber 11.0 / general 22.0 / visual 21.2 total; loss 7.3 / 4.8 / 5.7 / 9.4) — Table 4.

**RL objective** `(A)`: GRPO, `L = −E[Σ r_{i,t} M_{i,t} A_i log π_θ]`, length-normalised, `A_i = R_i − R̄`.

| knob | value (§5.1 unless noted) |
|---|---|
| algorithm | GRPO; **async partial rollout, staleness 4**; **prompt-mean** loss aggregation |
| batch | **1,568 prompts × G=16 = 25K sequences/step**; 2.7–3.7B tokens/step; ~110–150K tok/sequence; 1M context |
| optimizer | Muown lr **3e-6**, **no weight decay, no warmup**, grad-clip 1.0; Muon momentum 0.95 + Nesterov, **10 Newton–Schulz iters**, update scale 0.5; Adam β0.95/0.95, ε1e-8 |
| stabilisers | carry over **FP32 masters + row state** from SFT for MXFP4; **freeze the router** |
| IS clip | per-token `sg[π_θ/μ_old]`; **four decoupled bounds**, init [0.2, 5.0], retuned at runtime from **policy entropy** (widen positive/narrow negative when entropy is low; reverse when high); per-direction clip rate logged |
| sampling | **dynamic sampler drops all-pass and all-fail groups**; Sample Mixer holds the mixture under 90× token / 66× duration heterogeneity across 25 sources |
| reward | `R_i = R_i^test · S_i^sol · S_i^beh` — **multiplicative**, so a failed trajectory is exactly 0 and only passes get ranked (Eq. 2) |
| advantage fix | `λ = Σ_P A_j / Σ_P f_j A_j`; `A'_i = λ f_i A_i` (passes), unchanged (fails), `λ` capped, then subtract group mean → zero-mean group (Eq. 3) |
| length penalty | group-relative: `R̃_i = R_i − 1[i∈P]·X·[clip((ℓ_i/ℓ*_q − 1 − δ)/(s − δ),0,1)]^γ`, `ℓ*_q` = B-th percentile of **successful** lengths, term gated on group pass rate > A (Eq. 4) |
| behaviour penalty | token-level: flagged tokens (malformed tool call/markup) mask positive advantage and amplify negative by κ>1, `α`/`β` renormalised so each sign's advantage mass is conserved (Eq. 5) |
| hack defence | confirmed-hack reward **set to 0 before group statistics are recomputed**; logged confirmed-hack share **<2%** all run |

**Cost** `(A/L)`: **$2.6M (Pro) + $0.9M (Flash)**; split **rollout 43.8% / training 43.5% / grader 12.7%**. 30 steps, 75.0B / 81.4B cumulative tokens, 753k trajectories, **14 restarts** on Pro (DBEs, K8s, grader network, MoE-imbalance OOM, CPU OOM in packing) `(L)` → `(S)` ≈**$35 per million RL tokens** on Pro, ≈$10.5 on Flash.

**Live-run series** `(L)` (the paper's Figure 9 as raw data, not a rendered chart):

| Pro / Flash | step 1 | step 30 |
|---|---|---|
| `dynsam/avg@n` (mean pass rate) | 0.565 / 0.514 | **0.633 / 0.644** (+0.068 / +0.130) |
| `critic/rewards/mean` (reward incl. penalties) | 0.552 / 0.517 | 0.596 / 0.601 — the gap *is* the penalty |
| `ctx_total_length/mean` | 72.2K / 71.6K | 137K / 148K (**1.90× / 2.07×**) |
| `dynsam/agg_turn/mean` | 47.5 / 47.3 | 43.6 / 68.5 (flat-ish) |
| `perf/total_num_tokens` per step | 1.80B / 1.79B | 3.43B / 3.70B |
| `timing_s/step` | 9.6Ks / 6.5Ks | 23.2Ks / 12.4Ks |
| `actor/agentic/entropy_loss` | 0.399 / 0.417 | 0.479 / 0.484 (rising, no collapse) |
| `partial/avg_staleness` (cap 4) | 0 | 0.01–2.2 |
| `train_infer_diff/new_infer/kl` | 0.0022 | 0.0028 |
| `dynsam/passrate/zero` / `one` | 0.146 / 0.178 | 0.132 / 0.262 |
| `dynsam/infra_error/seq_rate` | 0.0073 | 0.0052 |

Curriculum in production, verbatim notice `(L)`: **"we filtered out tasks that are relatively easy for the current pro model."**

**Final published table** (Table 3, `(A/M)`) Pro / Flash, previous gen in brackets: DeepSWE v1.1 **71.9 / 67.9** (19.0) · ProgramBench 26.5 / 26.0 (12.5) · MiMo Code Bench 63.2 / 61.2 (40.4) · AutomationBench 53.1 / 52.3 (16.0) · Toolathlon-Verified 76.9 / 73.6 (49.1) · Agents' Last Exam 31.6 / 27.6 (13.2) · Terminal-Bench 4.0 34.9 / 28.8 (1.5) · Terminal-Bench 2.1 89.9 / 87.6 (65.2) · OSWorld-Verified 82.0 / 80.8 · JobBench 62.0 / 61.2 (25.0) · CyberGym 94.0 / 95.1 (40.0) · ExploitGym 17.8 / 6.0 · SEC Bench Pro 66.3 / 47.5 · MiMo Visual Coding 72.3 / 71.5. **Discrepancy I will not paper over:** the live page reports Flash DeepSWE v1.1 as **65.68** (`mini-swe-agent`, avg@3) against Table 3's **67.9**; harness and aggregation differ and neither source states the mapping `(S)`.

**The 9B distill + open environments** (§7, `(A/M/web)`) — the only size we can reason about: SWE-bench Verified 60.0 (Qwen3.5-9B) → 61.1 (SFT) → **66.2** (GRPO) · SWE-bench Pro 32.0 → 44.6 → 47.6 · MiMo Code Bench mini 19.5 → 51.6 → **59.9** · Cyber mini 5.7 → 31.3 → 47.0 · Terminal-Bench 2.1 27.0 → 37.1 → 52.8 · AutomationBench 5.0 → 30.3 → 33.1 · JobBench 2.6 → 18.3 → 25.2 · Visual Coding mini 61.7 → 64.0 → 72.4. Multi-harness RL, same checkpoint: mean over 7 harnesses SWE-bench Verified **62.3 → 65.7**, MiMo Code Bench **53.1 → 59.0**, per-harness gains 1.8–9.3 pts, **3 of 7 harnesses never trained on**. Released: ~**7k RL tasks** (code 3k executable tests · cyber 1k rule checks · general 1k rubric judging · visual 2k visual grading) + ~1k music + an end-to-end RL framework + mini-harnesses. `(S)` **The SFT share is the real headline: 27.2B loss tokens for a 9B model ≈ 3k supervised loss tokens per parameter, then only 30 RL steps** — RL is short because SFT was long, not because RL is cheap.

---

## 2. Transfers to 1.07B total / 57M active byte-level, one 16 GB GPU

| mechanism | transfers? | why |
|---|---|---|
| multiplicative reward `R_test·S_sol·S_beh` | **shape only** | the gate-times-bounded-factors shape is hack-resistant by construction (a failed trajectory has no quality path to reward) and free to port; the *factors* need a judge we do not have (§3.1) |
| group-relative length penalty (Eq. 4) | **yes** | 6 scalars, no model, no VRAM; replaces the plan's flat `−λ_tokens·T` with a per-prompt quantile reference and a pass-rate gate. Byte-level makes it easier — our "length" is already bytes |
| prompt-mean aggregation | **yes** | one normaliser; they state it "prevents response length from growing too quickly"; their own A/B used token-mean as the control (Fig. 8) |
| **freeze the router in RL** | **yes — highest value** | measured fix, zero cost, and `E=1024 topk-2` + controller is exactly the fragile object (risk below) |
| dynamic sampler (drop all-pass/all-fail) | **yes** | selection on numbers we can log; their zero/one shares are 13–16% / 26–29% of prompts `(L)`, i.e. **40%+ of the batch carries no group signal** |
| env-prep + hack agent + rollout audit | **yes** | deterministic, no LLM judge: scrub build artefacts/logs/patches/caches, truncate git at the base commit, network isolation, then an adversary loop until no leak remains |
| segment-level behaviour penalty (Eq. 5) | **later** | needs per-token flags; our tool-call grammar is small so flags are cheap — defer until §3.1 is measured |
| instrument set (§3.5) | **yes** | names exactly what a byte-thrift claim must log |
| multi-harness / mini-harness training | **partly** | mechanism yes (2–3 harness configs, evaluate on a held-out one); the 1.8–9.3 pt gains are 9B-scale and need 7 real harnesses |
| async partial rollout, staleness 4, KV pools, Ray harness pool, decoupled data/control plane | **no** | needs an inference engine distinct from the trainer; we are one process on one GPU and §1.5 forbids two GPU processes |
| MXFP4 QAT + Humming GEMM, R3 routing replay, top-p candidate-set replay | **no** | cluster-serving concerns; our quant line is TSCT-factor fp8/bf16 with its own latched fallback |
| Muown (Muon + row-norm control) | **partly** | our Muon+ already does ColRow (AGENTS §3.5 item 3); the *optimizer switch to enable large-batch RL* is the interesting half and is unmeasured here `(S)` |
| MOPD2 / prefix-conditioned OPD | **partly** | the new idea is reusing teacher-trajectory *prefixes* instead of full student rollouts; cheap, but needs a teacher |
| 1M context, 25K sequences/step, 2.7B tokens/step, $2.6M | **no** | 260× our active params; warm step is ~245 ms at 9.2M params batch 8 and the eval window is `batch × seq_len` bytes (§2.6) |

**Two failure modes, stated before they are run** `(S)`. (a) At 42B active MiMo has 384 experts *dense per token*; we have 1024 experts with **topk-2**, an Engram arm, and a controller whose softmax blend *is* the per-iteration routing. Freezing the controller may remove the loop block's only adaptive capacity while the TSCT selection is the part that actually risks load collapse → split the freeze (`--freeze-router experts|controller|both`), default unset until measured. (b) The length penalty's gate `A` can silence the term entirely at a low pass rate; if it does, it must be **COUNTED** (ADR-0011) so a no-op cannot look like a mechanism — `fused kda=0/0` is this repo's own precedent.

---

## 3. Adoptable items for `docs/architecture/post-training.md`

**3.1 REWARD-NODE `token-thrift`** — replaces the plan's `reward = verifier_pass − λ_tokens·T` (additive, flat token price) with a *gate × bounded quality factors* plus a *group-relative* deduction. Add to the `post-train rl` spec:

```text
[--grade mult|add|none]                  # default mult   (2610.11959 Eq. 2)
[--quality heuristic|judge]              # default heuristic = NO judge
[--len-penalty X,B,A,delta,s,gamma]      # 2610.11959 Eq. 4
[--agg prompt-mean|token-mean]           # default prompt-mean
```

**ARGUMENTS TO FILL** (placeholders in the plan, not defaults — nobody has measured them at our scale): `X` 0.2–0.5 · `B` 50–75 · `A` 0.3–0.5 · `δ` 0.0–0.1 · `s` 0.5 · `γ` 1–2. `--quality` **must** default to a deterministic proxy (patch size / bytes written / tool-call count within the verifier's accepted set); `(S)` no LLM judge in v1 — at 57M active the judge is the network under training, the exact configuration the earlier doc sources as a failure. Files: `docs/architecture/post-training.md` §"RLVR" (reward line + `token-thrift` sub-node); the flags slot beside `--adv/--clip/--kl/--monitor` in the CLI block of the earlier doc's §6.

**3.2 STABILITY-NODE `router-freeze`** (before any RL run). Promote the freeze from a footnote to a rule with its own A/B: run 1 freezes nothing, run 2 the TSCT router, run 3 controller+router. Metrics are §5.4's three load statistics — CV of expert load, peak load factor (max/mean), fraction of cold experts below 0.1× mean — plus held-out BPB. `--freeze-router experts|controller|both|none`, counted. Files: `docs/architecture/post-training.md`; a queue row in `docs/protocols/AB-PROTOCOL.md`; the counters belong in `crates/dormouse-core/src/probe.rs` (a counter change, not a training change).

**3.3 CURRICULUM-NODE `difficulty-refresh`** (upgrades the 5/70/archived bands). MiMo's bands are numbers it watches, not a policy: it rejects groups at **both** ends (`passrate/zero` ≈14%, `passrate/one` ≈26% `(L)`) and periodically deletes whole tasks that are "relatively easy for the current model" `(L)`. Adopt: bands recomputed from a logged pass-rate histogram (their `dynsam/passrate/hist9_ratio`, nine bins none→all solved) at a declared cadence, and `archived` is **deleted from the pool**, not parked. Arguments: `--band-refresh N`, `--dynsam-reject zero>F,one>F` (F initialised to measured shares), `--retire-easy` off by default. File: `docs/architecture/post-training.md` §"Curriculum" (the existing `--band mastered=F,frontier=F,archived=F` keeps its shape).

**3.4 VERIFIER-HYGIENE node** (extends "anti-reward-hack screening", which today names no mechanism). MiMo's leak list is enumerable: remove build logs, verifier outputs, residual patches and generated binaries/bytecode; scrub caches *outside* the repo; keep git history only to the base commit; enforce container-level network isolation; then run an adversarial **hack agent in a loop until it finds no exploit**, re-running after each cleanup change. Two audit procedures ride along: every task attempted **4×** with an auditor comparing judged correctness against the observed reward (false positive = incomplete tests; false negative = over-restrictive tests), and F2P/P2P outcome stability required across **8 reruns** before a task is admitted. This is where the earlier doc's "the anti-reward-hack stage should name which of these it implements" gets a primary-source answer. Arguments: `--hack-screen rounds=N,net=isolate,git=base-only`, `--audit 4`, `--rerun-stable 8`. Files: `docs/architecture/post-training.md` (env pools / anti-hack); the `verify` contract in the CLI spec.

**3.5 INSTRUMENT set** (log these names or the A/B is not a measurement). From `runs.json` `descriptions` `(L)`: `ctx_total_length/mean` (prompt+response tokens), a response-only sibling, `agg_turn/mean`, `total_num_tokens`/step, `avg@n`, `rewards/mean`, `entropy_loss`, `passrate/zero`, `passrate/one`, `passrate/hist9_ratio/*`, `infra_error/seq_rate`, `avg_staleness`, and a train↔infer divergence number (`train_infer_diff/*/kl`). **For dormouse the unit is bytes** — log `ctx_total_bytes/mean` and `bytes/step` too, so a byte-thrift claim is a number and not an adjective. Files: `docs/architecture/post-training.md` contract note; same discipline as `engram=<rows>/<arms>`.

---

## 4. The honest A/B gate, and what it costs here

Replicate the **comparison, not the scale**. MiMo's GAR A/B (GAGAR; Fig. 8 in 2610.11959) is: code-only RL, fixed batch (128 prompts), report **pass rate + mean turns + total token length**, with and without online groupwise grading. Three things transfer:

1. **Fixed batch, equal steps, equal wall-clock, and token length reported next to pass rate.** Our gate therefore reads: *avg@n **and bytes per solved task**, at equal steps and equal wall-clock* — never pass rate alone.
2. **The control is what we already planned, not a strawman.** Their control is GRPO + token-mean + no grading; ours is `--adv group --agg token-mean --len-penalty off --grade none`. The plan's PPO+value A/B is untouched by this document and remains open.
3. **The cheapest honest gate needs no judge.** Four single-knob A/Bs: (a) `--agg prompt-mean` vs `token-mean`; (b) `--len-penalty` on/off; (c) `--dynsam-reject` on/off; (d) `--freeze-router` none/experts/both — each a few dozen lines in the runner, each measurable in bytes and pass rate. The grader-dependent arm (groupwise quality ranking) is **deferred** with the judge.

**Cost, with the caveat.** The protocol is 3 seeds × 2k steps per arm, and **the per-arm cost is currently unknown — `docs/protocols/AB-PROTOCOL.md` says so itself** (AGENTS §3.3: the 53 min figure came from a run whose attention backward never executed, and the 25.8 s/step replacement has no committed log). `(S)` at the measured warm ~245 ms one arm is ~8 min of pure steps; at the retracted 1.6 s/step, ~53 min. Three seeds × four A/Bs × two arms = **24 runs ≈ 3–21 GPU-hours**, sequential under §1.5 (`MemoryMax=40G`, `free -g ≥ 25`, one heavy thing at a time), with matching eval windows (§2.6) and on the pre-4 stack, since 0.22.0 corrupts runs 2-in-3 (AGENTS §3.3). **Order of operations: measure one arm's per-step cost at the target shape first, then cost the gate.**

---

## 5. VERIFIED (sourced)

- **2610.11959** (2026-10-08): Pro 1.02T/42B active, Flash 310B/15B; GRPO, no value head; 1,568×16 batch; 2.7–3.7B tokens/step; 1M context; staleness-4 partial rollout; prompt-mean aggregation; four decoupled entropy-adaptive IS clip bounds; dynamic sampler; frozen router; Muown lr 3e-6, no warmup; reward `R_test·S_sol·S_beh`; GAR redistribution; length penalty (Eq. 4); behaviour penalty (Eq. 5); $2.6M/$0.9M; cost split 43.8/43.5/12.7% `(A)`.
- **Router freeze is measured** at 384 experts: CV 0.78→2.0, peak 6×→16×, cold experts 0.5%→22% in 20 steps; restoring only router weights recovers balance with performance unchanged `(A/M)`.
- **Confirmed-hack share <2%** all run, with hack rewards zeroed before group statistics recompute `(A)`.
- **Live series** (raw JSON, the page the owner named): token length 72.2K→137K (Pro) / 71.6K→148K (Flash); turns flat-ish; step time 9.6Ks→23.2Ks (Pro); pass rate +0.068 / +0.130; zero/one shares ≈14% / ≈26%; 14 restarts (Pro); 75.0B / 81.4B cumulative tokens; 753k trajectories `(L)`; the easy-task notice quoted verbatim `(L)`.
- **Open release**: MiMo-V2.6-Distill-Qwen-9B (MIT, Qwen3.5-9B base) + ~7k RL tasks with named verifier classes + RL framework + mini-harnesses; the 9B GRPO and multi-harness gains are tabulated (Tables 6/7) `(A/web)`.
- **The task's alphaxiv link 404s**; the Sept-2026 MiMo RL paper is **2609.32577 (GAGAR)**, whose abstract states sum-preserving quality-aware advantage redistribution for code-agent GRPO with reduced trajectory-length growth `(A)`.
- Earlier anchors: **2505.07608** (130K verifiable problems, test-difficulty code reward, resampling) · **2506.03569** (MORL) · **2601.02780** (MOPD, 309B/15B, SWA 5:1) `(A)`.

## 6. SPECULATION (my inference — no source)

- `(S)` **Every effect size here is a transfer.** No MiMo result exists below 9B dense / 15B active, and none for a byte-level, 57M-active, weight-shared loop block with Engram.
- `(S)` **Freezing the controller may cost us more than it cost MiMo** (§2) — split the flag, measure, never adopt `both` by default.
- `(S)` **The length penalty may be dead code at our pass rate** (§2b) — then log it COUNTED.
- `(S)` **The multiplicative shape with a deterministic proxy is untested anywhere I read**; MiMo's factors come from an LLM grader. This is the one place I recommend a shape whose implementation I explicitly cannot port.
- `(S)` **The SFT share matters more than the RL recipe** (§1) — if 27.2B loss tokens for a 9B model is the ratio, our plan's SFT node (which states percentages, not volume) is the under-specified half, and "few tokens" means *few RL tokens*.
- `(S)` **PPO+value is out-competed, not refuted** — MiMo shows a frontier loop with no critic; our critic's cost at 57M active is still unmeasured and fork-replay is still the deciding test.

## 7. Open questions this read could not close

1. **Our pass-rate distribution at 57M active** — without it, the length-penalty gate `A` and the dynsam thresholds cannot be set. First measurement the lane owes.
2. **Whether a router freeze at TSCT+controller helps or lobotomises the loop block** — no published instance at our shape.
3. **Grader cost and proxy validity** — 12.7% of a $2.6M run is not amortisable here, and MiMo publishes no correlation table for `S_sol`/`S_beh` against anything.
4. **67.9 vs 65.68 DeepSWE** (Table 3 vs live page) — unresolved, and it is the harness-attributable gap class the earlier doc sources (2610.02911).
5. **Usability of their released stack** — a verl-derived cluster framework and Python/TS harnesses against a Rust byte-level trainer. `(S)` read for design, import nothing.
