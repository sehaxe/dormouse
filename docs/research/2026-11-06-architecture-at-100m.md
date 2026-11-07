# What beats a vanilla transformer at 20–120M params, byte-level, one 16 GB card

**Date:** 2026-11-06. **Read-only:** this file only; no config, code or `port/src` touched, nothing run on the GPU.
**Question (owner):** the 12-layer plain transformer beats our KDA+CSA+mHC+Engram block by **0.13–0.37 BPB at equal bytes** [ours, owner measurement]. What, if anything, beats a vanilla transformer *at our scale* — 20–120M params, 0.5–22 GB text, 1–8 KB context, one 16 GB card — not at 7B.
**Method:** 20 arXiv queries, 40+ `get_abstract` resolutions, 28 full-text HTML fetches, every number read out of the paper's own table (table number named in each row). Repo facts from `AGENTS.md` §3.1 and `docs/protocols/AB-PROTOCOL.md`. Dates are arXiv first-submission dates.
**Marks:** **[T]** table read · **[A]** abstract only · **[ours]** repo measurement · **[S1]** SPECULATION with a sourced mechanism · **[S2]** SPECULATION with no direct source.

## Correction (2026-11-07, after the frontier teardown)

Point 1 below — "not one sequence-mixer candidate has a published compute-matched
win over a dense transformer at ≤200M" — is **wrong for linear attention** and is
kept only as the record of what we believed for one day. `docs/research/2026-11-07-frontier-teardown.md`
finds Olmo Hybrid (**2604.03444v4**, Table 5): at **60M params**, a 3:1 Gated
DeltaNet/attention hybrid reaches **1.077 BPB against a transformer's 1.155** —
a 0.078 BPB win, and the closest published scale to ours (95M). Kimi Linear
(2510.26692v2, Table 1) shows the same sign at 653M active with equal FLOPs
(PPL 5.65 vs 5.77).

What survives unchanged: MoE, loops, MTP, BLT and sparse attention have no
compute-matched win at our scale, and sparse attention is arithmetically worse at
s=1024 (+5.7% because the indexer costs more than it saves). What changes: the
linear-attention arm is not a dead end — it is the one mechanism with a
published win close to our size, and our `configs/arch_hybrid_kda.toml` (12
unique KDA blocks, 95.9M) is the arm that tests it.

## TL;DR

1. **At ≤200M params, not one sequence-mixer candidate in this list has a published compute-matched win over a dense transformer on loss/BPB.** The only ≤200M wins I can verify are (a) **sparse memory layers on a 134M backbone** (iso-FLOPs, 7× total params; NQ 0.91 → 2.10) and (b) a **byte-patching rule at 50M** (computed-result accuracy 21% → 57%), neither of which is a BPB win against a dense control.
2. **The closest scale-matched negative is depth recurrence:** Mixture-of-Recursions at **135M/2.0e18 FLOPs is worse than vanilla** (NLL 3.0922 → 3.1077, +0.0155) and only wins at **360M** (3.3785 → 3.1026, −0.2759) [T, 2507.10524 Table 15]. That is the shape of this whole literature: the crossover is above us.
3. **MTP is negative at 300M** (MBPP@1 1.8 → 1.0 with 4 heads) and positive at 3B (11.1 → 12.7) [T, 2404.19737 Table S7]. **BLT is worse than BPE at 1B** and overtakes at 7B [A/T, 2412.09871 §5.1, Table 13]. **Mamba-2 at 350M is a 0.9 % ppl tie** (8.68 → 8.60); its hybrid is −4.8 % at params-matched, not FLOP-matched [T, 2405.21060 Table 2].
4. **Your 0.13–0.37 BPB loss is what the literature predicts for a mechanism stack whose parts were validated at ≥1B.** KDA/GDN's clean win is at 1.3B (Wiki ppl 18.53 → 15.91 for the hybrid) [T, 2412.06464 Table 3]; Engram's is at 27B (val loss 1.634 → 1.622) [T, 2601.07372 Table 1]. Nothing transfers down for free.
5. **Recommendation:** keep the dense 12L/768 control; spend the A/B on **Engram-style hash memory** (only arm with a plausible win), a **3:1 GDN:full hybrid widened to d=960 at iso-FLOPs** (best mechanistic case, but predicted near-tie at 100M), and a **top-2-of-8 MoE at iso-active-FLOPs** (the arm that decides whether the b1 MoE lane lives). **300M bytes is 3.5 bytes/param** — a capacity screen, not a quality test; 2 seeds cannot resolve effects below ~0.06 BPB.

## 1. Candidate table — what was actually measured

`Scale` is the model the *number* comes from, not the paper's largest. `Matched?` distinguishes the three things papers conflate: **isoFLOP** (same training FLOPs), **params+tokens** (same size and data, attention FLOPs differ), and **none** (no dense control).

| # | Mechanism | Source (arXiv, date) | Scale (params / tokens) | Baseline → arm (metric) | Matched? | Note |
|---|---|---|---|---|---|---|
| 1 | **MoE, fine-grained, frontier** (DeepSeek-V3 style, 256 routed + 1 shared, top-8) | 2412.19437, 2024-12-27 [T] | 671B total / 37B active, 14.8T tok | no dense control; MTP ablation at 15.7B/1.33T shows wins on most benchmarks (Table 4) | **no** | frontier MoE paper never runs a dense baseline; MTP ablation is 15.7B, not 100M |
| 2 | **MoE, small-scale scaling law** (granularity G) | 2402.07871, 2024-02-12 [T] | **3M–100M active** (64×3M … 64×100M; total 192M–6.4B), 8–66B tok | fitted law: MoE curve below dense at *every* FLOP budget (Fig. 1); at 2.95e18 FLOPs, 64×100M loss **3.133** (Table 2) | law-level yes; **no printed matched pair** | smallest MoE is 64×3M = **192M total / 3M active**; claim is from the fit, not a head-to-head table |
| 3 | **MoE, 1.3B-active study** (OLMoE) | 2409.02060, 2024-09-03 [T] | 1.3B active / 6.9B total, 5.1T tok | "MoEs train **2× faster** than dense LMs with equivalent active parameters" (abstract) | active-param matched | 1.3B ≫ 200M; no 100M rung |
| 4 | **Mamba-2, pure** | 2405.21060, 2024-05-31 [T] | 350M, 48L, 7B tok Pile | Transformer++ **8.68** → Mamba-2 **8.60** ppl (Table 2) | params+tokens; FLOPs not matched | **−0.08 ppl (−0.9 %)** — a tie |
| 5 | **Mamba-2 + attention ratio sweep** | same [T] | 350M, 48L, 7B tok | 8.68 → **8.38** (1 attn layer), **8.29** (3), **8.26** (6), 8.50 (24) | params+tokens | best at **6/48 = 12.5 % attention**, −0.42 ppl (−4.8 %); 50 % attention recovers only −0.18 |
| 6 | **Hybrid at 8B** (Mamba-2 43 % + attention 7.1 % + MLP 50 %) | 2406.07887, 2024-06-12 [T] | 8B, 3.5T tok | Transformer → hybrid: **+2.65 pts** avg over 12 NL tasks (Table 7) | params+tokens | hybrid is **worse than Transformer until 1.5T tokens** (Fig. 6); irrelevant to a 0.3 GB run |
| 7 | **Gated DeltaNet / DeltaNet** | 2412.06464, 2024-12-09 [T] | **1.3B, 100B tok** FineWeb-Edu | Transformer++ Wiki ppl **18.53** → DeltaNet **17.71**, GDN **16.42**, GDN+SWA **16.07**, GDN+Mamba2+SWA **15.91** (Table 3) | params+tokens; FLOPs not matched | pure GDN −2.11 ppl (−11.4 %); hybrid −2.62 (−14.1 %); no 100M rung |
| 8 | **KDA / Qwen-style 3:1 GDN:attention** | Qwen3.8-Flash-Next tech report, PDF dated 2026-08-29 [T] | ablation on **28-layer 25B-A3B** MoE, 80B tok, 32K ctx (main model 125B/6B active) | Full attention **49.87** → SWA hybrid **51.15** → GDN hybrid **53.81** avg over 9 benchmarks (Table 1) | params+tokens; no FLOP match | **+3.94 pts** for 3:1 GDN at 25 % attention; the ratio is a frontier recipe, not a small-scale one |
| 9 | **KDA / linear-attention survey** | 2607.07953, 2026-07-08 [T] | 350M, 15B tok | **no Transformer row**: best KDA hybrid+Muon **2.273**, DeltaNet hybrid 2.299, pure GDN+AdamW 2.433 | **no dense control** | the only controlled KDA-vs-family sweep at 350M; cannot answer "vs vanilla" |
| 10 | **RWKV-7** | 2503.14456, 2025-03-18 [T] | 0.1B–2.9B, 3.1T tok | no matched dense baseline; compares to Llama3.2/Qwen2.5 on different data (Tables 4–5) | **no** | 0.1B and 0.4B checkpoints exist, but no controlled Transformer comparison is published |
| 11 | **Looped / recurrent depth** (Mixture-of-Recursions) | 2507.10524, 2025-07-14 [T] | **135M base / 42M unique**, 2.0e18 FLOPs (vanilla 6.5B tok, MoR 9.2B tok) | Vanilla **3.0922** → Recursive 3.2058 → MoR **3.1077** NLL (Table 15) | **YES isoFLOP** | **+0.0155 NLL worse at 135M**; same direction at 5e18 (+0.0728) and 16.5e18 (+0.1058) |
| 12 | **Looped depth, 360M** | same [T] | 360M base / 118M unique, 2.0e18 | Vanilla **3.3785** → MoR **3.1026** | **YES isoFLOP** | **−0.2759 NLL** at 360M; at 16.5e18 the NLL tie is 2.7824 vs 2.7924 (MoR slightly worse) |
| 13 | **Looped MoE with stabilization** (LOOM) | 2610.01153, 2026-10-01 [T] | 100M–1.7B, up to 60B tok | near-isoFLOP: "best near 5 loops… consistently lower loss" (abstract, §1) | near-isoFLOP | win reported across **100M–1.7B**, but the HTML numbers are LaTeX placeholders; **delta not verifiable** from the text |
| 14 | **Recurrent depth, test-time** (Huginn) | 2502.05171, 2025-02-07 [T] | 3.5B, 800B tok | matched to "what a 32B transformer would consume" in pretraining FLOPs | spends compute; no dense control | **NOT MEASURED** as a delta |
| 15 | **Multi-token prediction** | 2404.19737, 2024-04-30 [T] | 0.3B–13B, 91–209B tok code | MBPP@1: **0.3B 1.8 → 1.0** (n=4), 1.3B 6.8 → 7.4, 3B **11.1 → 12.7** (Table S7) | same compute (params reallocated) | **negative at 300M**, tie at 1.3B, positive at 3B; DeepSeek-V3's own ablation is 15.7B |
| 16 | **Byte-level patching** (BLT) | 2412.09871, 2024-12-12 [T] | 400M–8B; 1B and 8B reported | 8B FLOP-matched avg: Llama 3 **60.0**, BLT-Space **58.0**, BLT-Entropy **61.1** (Table 1); at **1B BLT-8 is worse than BPE Llama 2**, crossover at 7B (§5.1) | isoFLOP at 8B | **−2.0 / +1.1 pts** at 8B; the 470M entry needs **150B bytes** to cross BPE (Table 13) — our whole corpus is 0.5–22 GB |
| 17 | **Byte-level multi-scale** (MegaByte) | 2305.07185, 2023-05-12 [T] | 320M Transformer control vs 758M global/local, 80B bytes | PG-19 bpb **1.057 → 1.000**; Code **0.575 → 0.411** (Table 2) | **YES compute+data** | **−0.057 bpb** on prose, −0.164 on code, at 320M control — the one clean byte-level win, but >200M and 80B bytes |
| 18 | **Byte patching rule** (novelty patching) | 2610.22100, 2026-10-15 [T] | **50M**, 1.9 GB math, 3 seeds, ~11 % patch rate | vs entropy patching: computed-result **21 % → 57 %**, final answer **60 % → 90 %** | matched patch rate | **+36 pp** at 50M; not a BPB number and not vs a dense transformer |
| 19 | **Engram / conditional memory** | 2601.07372, 2026-01-12 [T] | 27B total / 3.8B active, 262B tok | MoE-27B val **1.634** → Engram-27B **1.622**; Pile **1.960 → 1.950** (Table 1) | **YES iso-param + iso-FLOPs** | **−0.012 val loss** (−0.73 %); 5.7B of the 26.7B params moved from experts to a lookup table |
| 20 | **Engram small-scale controlled** | 2601.16531, 2026-01-23 [T] | **185M backbone + 128M table**, 100M tok | iso-parameter memory variants only: Hash-500K **4.4809**, Nine-100/400K **4.4799**, Hash-800K 4.4961 (Table 12) | **no dense/no-memory control** | capacity knee at **500K slots/order**; collision elimination is a null (Δ 0.0010) |
| 21 | **Sparse memory layers** | 2412.09764, 2024-12-16 [T] | **134M base / 937M total** (token count not in Table 1; paper's runs to ~1T) | NQ **0.91 → 2.10**; TQA **7.7 → 16.31**; HotPot **5.18 → 9.28** (Table 1) | **iso-FLOPs** (params 7×) | the only ≤200M loss/QA win in this table; memory is learned KV, not hashed n-grams |
| 22 | **Sliding-window + global hybrid** | 2606.15378, 2026-06-13 [T] | **71M–665M**, 16K ctx, 100B tok | 0.22B short-ctx avg **38.13 → 38.03**; 0.66B **40.46 → 41.31**; 16K long-ctx 0.22B **25.09 → 35.33** (Tables 2, 21) | params+tokens | **no short-context win**; the gain exists only at 16K+, i.e. outside our 1–8 KB window |
| 23 | **SWA at 2B–27B** (Gemma 2, 1:1 local:global, window 4096) | 2408.00118, 2024-07-31 [T] | 2B / 9B / 27B, >50× Chinchilla tokens | no isolated local-vs-global ablation in the report | **no** | architecture shipped, effect not isolated |
| 24 | **SWA at 7B** (Mistral) | 2310.06825, 2023-10-02 [A] | 7B | no matched dense baseline | **no** | commonly cited, never measured as a controlled delta |

## 2. The one number per row

`delta vs dense at matched compute` — the column the decision turns on. "NM" = not measured at that scale with a dense control.

| Mechanism | One number | Scale of that number | Matched compute? |
|---|---|---|---|
| MoE fine-grained (frontier) | **NM** | 671B/37B | no dense control |
| MoE small-scale law | **NM** (law: MoE below dense at all budgets; 64×100M loss 3.133 @2.95e18) | 3M–100M active | law-level only |
| MoE (OLMoE) | **2×** training efficiency at equal active params | 1.3B active | active-param |
| Mamba-2 pure | **−0.08 ppl** (8.68 → 8.60, −0.9 %) | 350M / 7B tok | params+tokens |
| Hybrid ratio (Mamba-2) | **−0.42 ppl** (8.68 → 8.26, −4.8 %) at 12.5 % attention | 350M / 7B tok | params+tokens |
| Hybrid (8B) | **+2.65 pts** avg (12 tasks), but worse until 1.5T tok | 8B / 3.5T | params+tokens |
| Gated DeltaNet | **−2.11 ppl** pure, **−2.62** hybrid (18.53 → 15.91) | 1.3B / 100B | params+tokens |
| KDA survey | **NM vs dense** (best 2.273 vs 2.433 within family) | 350M / 15B | no dense control |
| RWKV-7 | **NM** | 0.1B–2.9B | no |
| Looped depth (MoR, 135M) | **+0.0155 NLL worse** (3.0922 → 3.1077) | 135M / 2e18 FLOPs | **YES isoFLOP** |
| Looped depth (MoR, 360M) | **−0.2759 NLL** (3.3785 → 3.1026) | 360M / 2e18 FLOPs | **YES isoFLOP** |
| LOOM | win reported at 100M; **number not extractable** | 100M–1.7B | near-isoFLOP |
| Huginn | **NM** | 3.5B | no |
| MTP | **−0.8 MBPP@1 at 0.3B**, **+1.6 at 3B** | 0.3B / 3B | same compute |
| BLT | **worse at 1B**, crossover at 7B; 8B **−2.0 (space) / +1.1 (entropy) pts** | 1B / 8B | isoFLOP at 8B |
| MegaByte | **−0.057 bpb** (1.057 → 1.000, PG-19) | 320M control / 80B bytes | **YES compute+data** |
| Novelty patching | **+36 pp** computed-result accuracy (21 → 57) | 50M | matched patch rate |
| Engram | **−0.012 val loss** (1.634 → 1.622, −0.73 %) | 27B / 3.8B act | **YES iso-param+iso-FLOPs** |
| Engram small-scale | **NM vs dense** (memory variants 4.4799–4.4961) | 185M + 128M | no dense control |
| Memory layers | **+1.19 NQ pts** (0.91 → 2.10), +8.6 TQA pts | 134M base / 937M total | **YES iso-FLOPs** |
| SWA+global hybrid | **−0.10 / +0.85 pts** short-context (0.22B / 0.66B); +10.2 pts at 16K | 71M–665M | params+tokens |
| Gemma 2 / Mistral SWA | **NM** | 2B–27B / 7B | no |

## 3. What the evidence says works at ≤200M params

Only these have a measured win *at or below 200M*; the number is stated with its metric, and the caveat is in the row.

1. **Sparse memory layers at a 134M backbone — the one real ≤200M win.** Dense 134M vs Memory 937M total (iso-FLOPs by construction): NQ **0.91 → 2.10**, TQA **7.7 → 16.31**, HotPot **5.18 → 9.28** [T, 2412.09764 Table 1]. Caveats: 7× total params; learned key-value memory, not hashed n-grams; QA accuracy, no BPB; the paper's own token count for that table is not printed (runs go to ~1T tokens).
2. **A byte-patching *rule*, not an architecture, at 50M.** Novelty patching vs entropy patching at a matched ~11 % patch rate: computed-result **21 % → 57 %**, final answer **60 % → 90 %**, 3 seeds [T, 2610.22100 abstract/§1]. Caveat: math accuracy on a 1.9 GB math corpus, not held-out BPB; the gain shrinks to "ties entropy" on code and synthetic traces.
3. **Nothing else.** The scale-matched tests that exist are **negative**: MoR at 135M (+0.0155 NLL), MTP at 300M (−0.8 MBPP@1), BLT at 1B (worse than BPE), SWA hybrid at 0.22–0.66B (no short-context win), Engram at 185M (no dense control), Mamba-2 at 350M (−0.9 %, a tie). The MoE scaling law is the only *positive* ≤200M-active claim, and it is a fitted curve, not a matched pair [T, 2402.07871 Fig. 1, Table 2].

**The honest summary:** at 20–120M params the published evidence does not support any of KDA/CSA/mHC sequence mixing, or Engram-style memory, *as a BPB win over dense at this scale*. Capacity that is cheap per FLOP (lookup memory) is the one mechanism with a ≤200M win, and it is parameter-heavy, not compute-heavy.

## 4. What is only shown at ≥1B — SPECULATION, not advice

These are real wins, at scales 10–250× ours. Listing them is not a recommendation; the transfer is unmeasured and in three cases the paper itself shows the effect reversing below 1B.

| Mechanism | Win at its scale | Why it does **not** transfer down |
|---|---|---|
| **GDN / KDA 3:1 hybrid** | 1.3B: ppl 18.53 → 15.91 (−14.1 %) [T, 2412.06464]; Qwen 25B-class: 49.87 → 53.81 avg (+3.94) [T, Qwen3.8 report Table 1] | at 350M the same family is a **0.9 % tie** (Mamba-2) and at 8B the hybrid is **behind until 1.5T tokens**; 25 % attention at 100M is not where the 350M sweep's optimum sits (12.5 %) |
| **Engram** | 27B: val 1.634 → 1.622 (−0.73 %) [T, 2601.07372 Table 1] | the win is bought with **5.7B memory params** (21 % of the model); at 85M backbone the table would be ~96M params, and the only 185M study finds the capacity knee at 500K slots and **no dense control** |
| **MoE fine-grained** | law fitted over 3M–100M active, claim "always more efficient" [T, 2402.07871]; 1.3B active 2× data efficiency [T, 2409.02060] | Clark et al.'s earlier result (cited by 2402.07871 §2) is that routed models are efficient **only up to a model size**; on a launch-bound 16 GB card the expert gathers cost step time, and our own MoE arm is a measured null [ours, `2026-10-16-token-skipping.md`] |
| **Looped depth** | 360M isoFLOP −0.2759 NLL [T, 2507.10524]; 730M matched-FLOPs looped ≈ 112-layer FFN [A, 2610.23036] | **135M isoFLOP is a loss** (+0.0155); the crossover in the only sweep with both rungs is between 135M and 360M |
| **MTP** | 3B +1.6 MBPP@1, 13B larger; DeepSeek-V3 15.7B [T, 2404.19737 Table S7; T, 2412.19437 Table 4] | **300M is negative** (1.8 → 1.0) and 1.3B is a tie; the paper says so in its own figure caption |
| **BLT / byte patching** | 8B FLOP-matched: Llama 3 60.0, BLT-Space 58.0, BLT-Entropy 61.1 [T, 2412.09871 Table 1]; MegaByte −0.057 bpb at a 320M control [T, 2305.07185 Table 2] | BLT is **worse at 1B** and needs 150B bytes to cross BPE at 470M; our 300M-byte screen is 500× shorter |
| **SWA + global** | 16K long-context: 25.09 → 35.33 at 0.22B [T, 2606.15378 Table 21] | at 1–8 KB the window is the whole context; the same table shows **no short-context gain** (38.13 → 38.03 / 40.46 → 41.31) |

## 5. Recommendation — three arms, one control, 300M bytes

**Assumptions for the arithmetic** (state yours if different): byte vocab 256, tied head, FFN 4d, 4 KB context, fwd+bwd FLOPs/byte = `6N + 12·n_attn·L·d`, L=4096. `MFLOP/byte` is training FLOPs, not inference.

| Arm | Config | Params | MFLOP/byte | Predicted ΔBPB vs control | Confidence |
|---|---|---|---|---|---|
| **C0 control** | 12L, d=768, 12 full-attn, 4 KB | **85.1M** | **964** | — | — |
| **C1 Engram** | C0 + 3 orders × 500K slots × 64 dim, host-RAM tables, convex gate | 85.1M GPU + **96M host** | **964** (+3 gathers/byte) | **−0.02 [−0.05, +0.01]** | low–medium |
| **C2 hybrid** | 12L, **3 GDN : 1 full-attn**, widened to d=960 to match FLOPs | **133.0M** | **939** (−2.7 %) | **−0.03 [−0.10, +0.03]** | low |
| **C3 MoE** | C0 with 8 experts/layer, d_ff=1536, top-2, dropless | **255M total / 85M active** | **964** | **0.00 [−0.05, +0.05]** | low |

**Why these three.**
- **C1 is the only arm with a ≤200M win in the literature** (memory layers at 134M; Engram at 27B) and it costs almost no FLOPs. If it cannot move BPB at 85M, the entire memory lane is dead at our scale — a cheap, decisive kill.
- **C2 is the strongest ≥1B mechanism that is architecturally compatible with the block we already have** (KDA is a GDN variant). Widening to d=960 is the compute-match: 133M params at 939 MFLOP/byte vs 85M at 964. At 4 KB the FLOP saving from linear attention is only ~35 % of the attention term, so this buys ~56 % more parameters, not 4×.
- **C3 decides the b1 lane.** At iso-active-FLOPs it must beat the dense control on BPB, not just loss-per-parameter, or the 1.07B-total/57M-active plan has no measured basis at 100M. The literature is split (law says yes; Clark says only up to a size; our own run says null).

**Protocol (pre-registered).**
- 300M bytes, 4 KB, batch 8, bf16, same data order and eval window (`eval_batches × batch × seq_len` printed), 2 seeds minimum, 3 if any arm is within 0.05 BPB.
- **Resolution warning, stated before the run:** 300M bytes at 85M params = **3.5 bytes/param**, i.e. 18 % of the Chinchilla budget. This is a capacity *screen*, not a quality test: C1 and C3 are parameter-heavy and will be under-trained by construction. The repo's own 3-seed spread at the 9.2M preset is **0.048–0.073 BPB** [ours, AGENTS §3.1]; even if it halves at 85M, a 2-seed difference must exceed **~0.06 BPB** to be credible. C1's expected −0.02 is *below* that: the 300M screen can kill C1, not crown it.
- **Confirm gate:** any arm within 0.03 BPB of control gets **1.5 GB bytes, 3 seeds** (≈5× the compute, ~26 h/arm at 15 TFLOP/s effective). Anything that loses by >0.03 at 300M is deleted.
- **FLOP accounting:** 0.964 GFLOP/byte × 0.3 GB = **2.9e17 FLOPs/seed** ≈ 5.4 h/seed at 15 TFLOP/s, ≈ 11 h per 2-seed arm, ≈ 44 h for C0–C3. That is one card for two days — the honest price of four arms.

**Kill criteria.** C1: no BPB gain at 1.5 GB ⇒ delete the Engram arm (not "keep for long context"). C2: worse than control at iso-FLOPs ⇒ the 3:1 recipe is a ≥1B recipe; keep full attention. C3: tie ⇒ delete the MoE lane, because a tie at 100M active with 3× total params is the best case it will ever get on this box.

## 6. VERIFIED (sourced)

- **MoR 135M is worse than vanilla at equal FLOPs; 360M is better at 2e18 and a tie at 16.5e18.** Vanilla 3.0922 / MoR 3.1077 at 135M/2e18; 3.3785 / 3.1026 at 360M/2e18; 2.7824 / 2.7924 at 360M/16.5e18 [T, 2507.10524 Table 15, 2025-07-14].
- **MTP is negative at 0.3B and positive at 3B on MBPP@1**: 1.8 → 1.0 (n=4) and 11.1 → 12.7 [T, 2404.19737 Table S7, 2024-04-30].
- **BLT is worse than BPE at 1B and overtakes at 7B**; 8B FLOP-matched average: Llama 3 60.0, BLT-Space 58.0, BLT-Entropy 61.1 [T/A, 2412.09871 Table 1 and §5.1, 2024-12-12].
- **Mamba-2 at 350M/7B is a ppl tie** (8.68 → 8.60); the 12.5 %-attention hybrid is −0.42 ppl, params+tokens matched, FLOPs not [T, 2405.21060 Table 2, 2024-05-31].
- **GDN at 1.3B/100B**: 18.53 → 16.42 pure, 15.91 hybrid, params+tokens matched [T, 2412.06464 Table 3, 2024-12-09].
- **Engram at 27B is iso-param and iso-FLOPs**: val 1.634 → 1.622, Pile 1.960 → 1.950 [T, 2601.07372 Table 1, 2026-01-12].
- **Memory layers at a 134M backbone**: NQ 0.91 → 2.10, TQA 7.7 → 16.31 with 937M total params, iso-FLOPs [T, 2412.09764 Table 1, 2024-12-16].
- **The 185M Engram study finds the capacity knee at 500K slots/order** (4.4809) and 800K is worse (4.4961); collision elimination is a null (4.4809 vs 4.4799) [T, 2601.16531 Table 12, 2026-01-23].
- **SWA hybrid has no short-context win at 0.22–0.66B**: 38.13 → 38.03 and 40.46 → 41.31; the 16K gain is 25.09 → 35.33 [T, 2606.15378 Tables 2/21, 2026-06-13].
- **Novelty patching at 50M**: 21 % → 57 % computed-result, 60 % → 90 % final answer, 3 seeds, matched ~11 % patch rate [T, 2610.22100, 2026-10-15].
- **Fine-grained MoE law**: models from 64×3M to 64×100M active; 64×100M at 2.95e18 FLOPs has loss 3.133; the "always more efficient" claim is from the fitted curves, not a matched pair [T, 2402.07871 Table 2/Fig. 1, 2024-02-12].
- **Qwen's 3:1 GDN recipe**: full 49.87 vs GDN hybrid 53.81 over 9 benchmarks, 80B tokens, 32K ctx [T, Qwen3.8-Flash-Next PDF Table 1, 2026-08-29].
- **Repo context**: 3-seed spread 0.048–0.073 BPB at the 9.2M preset; eval window = `eval_batches × batch × seq_len` [ours, AGENTS §3.1/§2.6].

## 7. SPECULATION (my inference — no direct measurement)

- **S1 — C1's −0.02 BPB is an extrapolation, not a measurement.** It assumes the 27B Engram relative loss gain (−0.73 %) survives at 85M with a 96M table. The 185M study's flat curve (4.4799–4.4825 across 300K–500K slots) suggests the capacity effect at that size is ≤0.002 val loss, i.e. **an order of magnitude smaller than my prediction.** If the 185M number is the right anchor, C1 is a **tie**, not a win. I keep the range wide because the 134M memory-layer result is large and because hashed lookup is cheaper per param than learned KV.
- **S2 — C2's −0.03 BPB assumes a 350M params-matched gain survives a FLOP match.** Mamba-2's −4.8 % at 350M is not FLOP-matched; the hybrid adds attention FLOPs. At iso-FLOPs I discount it by ~70 %, then by another factor for 100M. The +0.03 upper bound covers the possibility that 25 % attention is simply the wrong ratio at 100M (the 350M sweep's optimum is 12.5 %).
- **S3 — C3's 0.00 assumes the MoE law's small-scale regime is real but unmeasurable on a launch-bound card.** If the law holds *and* the card were compute-bound, C3 would be the most likely winner; on this box the expert gathers keep the step launch-bound, so the FLOP saving does not become wall-clock, and a BPB win at equal steps would be the only thing that counts.
- **S4 — "0.13–0.37 BPB is the price of ≥1B mechanisms at 100M" is my reading, not a published claim.** No paper measures KDA+CSA+mHC+Engram at 100M; the closest published facts are the 135M MoR loss, the 300M MTP loss, the 1B BLT loss, and the 350M Mamba-2 tie. Four independent negatives at the same scale band make the pattern credible, but it is a pattern, not a controlled experiment.

## 8. Rejected candidates (and why)

| Rejected | Why |
|---|---|
| **NSA (2502.11089, 2025-02-16)** | 27B MoE; the sparse-attention win is at 27B and long context. No ≤200M dense-controlled delta. |
| **Zamba2 (2411.15242), Samba (2406.07522), Nemotron-H (2504.03624)** | 1.2B–56B hybrid reports; comparisons are to other open models, not a controlled dense baseline at our scale. |
| **Falcon-H1 (2507.22448, 2025-07-30)** | ablations on 300M–1.5B *proxy* models, but the published comparisons are final models vs other families; no isolated dense-control delta at 300M in the tables I read. |
| **Ouro / looped LM checkpoints** | no resolvable arXiv ID for the 1.4B/2.6B models; every claim is second-hand (see `2026-10-21-adaptivity.md` §6.4). |
| **MoD (2404.02258)** | isoFLOP parity, not a win; token-level skipping is launch-bound on this card [ours, `2026-10-16-token-skipping.md`]. |
| **CSA / compressed sparse attention (our arm)** | no external paper with a ≤200M dense-controlled delta; our own measurement is the 0.13–0.37 BPB loss. |
| **mHC / multi-head controller** | no published compute-matched evidence at any scale that I could verify. |

## 9. Open questions

1. **Does the MoR crossover sit between 135M and 360M, or between 135M and 200M?** Only a 200M rung would tell; nobody has run it. This is the single most decision-relevant missing point.
2. **Is the Engram capacity curve flat at 85M too?** The 185M study says 300K–500K slots differ by 0.0016 val loss; if that holds at 85M, the 96M-param table is pure RAM cost.
3. **What is the FLOP-matched hybrid ratio at 100M?** The 350M sweep says 12.5 %, the 8B model says 7.1 %, Qwen says 25 %. The trend is upward with scale, which means 25 % is probably wrong for us and 12.5 % is the better arm.
4. **Can a 2-seed 300M-byte run resolve anything?** Our own spread says no below ~0.06 BPB. Either accept a screen-only role or budget 1.5 GB × 3 seeds.
5. **Does byte-level change the crossover?** Every mechanism above was measured on BPE tokens. Byte sequences are 3–4× longer, so attention is 3–4× more expensive per byte and linear attention's FLOP saving is larger — the hybrid case may be *better* at 8 KB than the token-level numbers suggest. Nobody has published that comparison.

## Sources

**Full text [T], table numbers named above:** 2412.19437 (Table 4) · 2402.07871 (Tables 1–2, Fig. 1) · 2409.02060 (§4.1.1, abstract) · 2405.21060 (Table 2) · 2406.07887 (Tables 6–7, Fig. 6) · 2412.06464 (Tables 3, 29) · 2503.14456 (Tables 4–5) · 2507.10524 (Tables 3, 15–16) · 2404.19737 (Table S7) · 2412.09871 (Tables 1, 13) · 2305.07185 (Table 2) · 2412.09764 (Table 1) · 2408.00118 (Table 1) · 2601.07372 (Table 1) · 2601.16531 (Table 12) · 2606.15378 (Tables 1–2, 21) · 2607.07953 (Table 2) · 2610.22100 (abstract, §1) · 2610.01153 (abstract, §1) · 2502.05171 (abstract, §6.1) · 2610.23036 (abstract).

**Abstract only [A]:** 2310.06825 · 2502.11089 · 2411.15242 · 2406.07522 · 2504.03624 · 2507.22448.

**Tech report, not peer-reviewed:** Qwen3.8-Flash-Next, local PDF `~/Downloads/tech_report.pdf`, mtime 2026-08-29, Table 1 (architecture ablation) and §2.1 (3 GDN : 1 full attention).

**Repo [ours]:** `AGENTS.md` §2.6 (eval window), §3.1 (3-seed spread 0.048–0.073), §3.2 (retracted Engram/KDA numbers — not used here) · `docs/research/2026-10-16-token-skipping.md` (MoD/MoR/MoE nulls) · `docs/research/2026-10-21-adaptivity.md` §1 (MoR/LOOM/Huginn scale table).

**Manifest line owed:** `docs-site/tools/manifest.mjs` is not updated for this document (task restricted to one file); `python3 tools/check_doc_refs.py` will flag it until that line lands.
