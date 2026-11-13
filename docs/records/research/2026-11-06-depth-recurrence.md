# Depth recurrence at 20–200M params: does reusing weights across depth ever pay?

**Date:** 2026-11-06. **Read-only:** this file only; no config, code or `port/src` touched, nothing run on the GPU.
**Question (owner):** we train byte-level LMs at 20–200M params on one 16 GB card. Our block runs **one unique layer three times** (`loop_blocks=1`, `max_iter=3`, `prelude_layers=0`, `coda_layers=0` — `port/configs/flagship_base.toml`). A plain 12-layer transformer beat it by **0.13–0.37 BPB at equal bytes**. Can depth recurrence be made to pay at our scale?
**Method:** 14 arXiv queries, 20 abstract resolutions, 11 full-text reads through the arXiv MCP server, plus the local PDF of arXiv:2604.21106 (the paper this question is literally about). Every number below is read out of the paper's own table/figure or is marked **[ours]**/**[S]**.
**Marks:** **[T]** table read · **[F]** figure/prose number, no table · **[A]** abstract only · **[ours]** repo document · **[S]** SPECULATION (my inference, no measurement behind it).

---

> **CORRECTION 2026-11-08** (from `docs/records/research/2026-11-08-hyperconnections.md`):
> the claim below that no ≤200M iso-FLOP loop win exists was true of the papers
> read on 2026-11-06 and is FALSE as of 2026-07: **Hyperloop Transformers**
> (arXiv:2604.21254v3, read in full on 2026-11-08) reports a **135.7M looped
> model at 14.40 PPL against a 238.0M dense at 14.65** — same tokens, same 16
> layer-applications — and against a dense model carrying the same
> hyperconnections (14.55). It is the first real iso-FLOP win for loops at our
> scale.
>
> **CORRECTION 2026-11-08, part 2** (from `docs/records/research/2026-11-08-mor-cycle2.md`):
> the "MoR cycle-2 135M" row this file cites is **not MoR** — it is a plain
> recursive transformer with a *Cycle* scheme in Table 10 (no router, no
> early-exit); the router in that paper's iso-FLOP grid LOSES to vanilla at
> 135M. Read correctly, the row is +0.0251 NLL = **0.009–0.012 BPB** — about
> 1σ at 120 MB — and the paper's own 360M row flips the sign (+0.0016). The
> 116-run iso-depth law (+0.03…+0.06 nats at r=2) is the better-supported
> number, and it points the other way.

> **It does not revive OUR loop.** Hyperloop's cell is a prelude + a 4L middle
> repeated + a coda = **8 unique layers** (its own shape ablation: 4L 14.40,
> 3L 14.62, 2L 15.06; **1L is outside the table**). Our `loop_blocks=1,
> max_iter=3` is 1 unique layer, and no connection scheme adds the missing
> blocks: even φ=1 gives 3 layer-equivalents of 12. The fair next step is an
> 8-unique-layer Hyperloop-shaped arm at equal FLOPs, not a hyperconnection on
> the 1×3.

## TL;DR

1. **The decisive experiment exists and it is at our scale.** Schwethelm et al., *How Much Is One Recurrence Worth?* (arXiv:2604.21106v3, 2026-05-07), sweeps `r ∈ {1,2,4,8}` loops at **20 effective layers**, six compute budgets from **4.64e17 to 2.15e19 FLOPs** (= 30–400M params), 116 runs, and fits a **recurrence-equivalence exponent φ = 0.46** (95% CI **[0.41, 0.53]**). Each recurrence is worth **0.46 of a unique block**. At matched FLOPs the looped frontier trails by **+0.03…0.06 nats (r=2)**, **+0.05…0.08 (r=4)**, **+0.09…0.12 (r=8)**, and **the gap widens at lower budgets** [T/F].
2. **Our config is at the far extreme of the losing axis.** `N_eff = r^φ·N_rec` with one recurrent layer and r=3 gives **3^0.46 = 1.66 layer-equivalents** vs the control's 12 — **14% of the capacity** at the same 12 layer-passes. The best measured intervention (hyperconnections) lifts φ to **0.65**, i.e. **2.04 layer-equivalents**. No published fix reaches φ=1 [T].
3. **Loops win at ≤200M only when they keep most of the parameters.** The one compute-matched ≤200M win in the literature is MoR's **cycle-2 at 135M**: 15 unique layers looped twice, **53M of 106M non-emb params kept**, NLL **3.0323 → 3.0071 (−0.0251)** at iso-FLOPs and iso-tokens [T, 2507.10524v3 Table 10]. Three loops on the same table **lose 0.0725** (3.0323 → 3.1048). Our 1-layer × 3 keeps **8%** of the control's layer count.
4. **The published tying penalty at our scale is ~0.015–0.045 BPB — 3–10× smaller than our measured 0.13–0.37 BPB.** So our gap is **not primarily weight tying**: it is the block (KDA/Engram/256-expert MoE) plus a 1-layer effective depth. Making the tie "cheaper" (per-iteration LoRA, norms, iteration embeddings) can recover at most ~0.01–0.02 BPB and never flips the sign.
5. **Recommendation: do not spend the 1.5 GB × 3-seed A/B on a looped arm.** Predicted best case for the least-bad config is **+0.02 BPB [−0.01, +0.03]** vs the dense control — a tie at best, and a tie is worthless here because **parameters are not our binding resource; FLOPs and bytes are**. Spend that GPU on an arm that can win.

---

## 0. The arithmetic that decides it

The iso-depth law is `L = E + A·(N_once + r^φ·N_rec)^−α + B·D^−β`. Two consequences, both from [T, 2604.21106v3 Table 2, §3.2]:

| Quantity | Formula | Our flagship (`max_iter=3`, 1 block) | A 12-layer dense control |
|---|---|---|---|
| Unique body layers | — | **1** | **12** |
| Layer-passes per token | — | **3** | **12** |
| Effective params under φ=0.46 | `N_once + r^0.46·N_rec` | **1.66 layer-equivalents** | **12** |
| Same, best published fix (φ=0.65) | `r^0.65` | **2.04** | **12** |
| Capacity vs control | — | **14%** (φ=.46) / **17%** (φ=.65) | 100% |

Ratios are exact in the paper's own units (non-embedding transformer layers at equal width); applying them to a 1-layer block is **[S]** — the paper's own r=8 arm is 2 unique layers, the smallest it measures.

Sanity check of the fit on the paper's own worked example: r=4, `(4 + 4^0.46·4)/20 = 0.58`, i.e. a **410M r=4 looped model ties a 580M dense one but costs the FLOPs of a 1B dense one** [T, Appendix F.4]. That is the whole trade in one sentence: loops buy parameters, and spend FLOPs to get them.

**Why this matters for us specifically:** at 85M params on a 16 GB card, parameters are not scarce. A mechanism that converts FLOPs into parameter savings is buying a currency we do not need and paying in the one we are short of.

---

## 1. Every family of "reuse the same weights across depth", and what each actually measures

### 1.1 Universal Transformer / ACT (2018) — the original, and it won on MT

One block applied L times, with a per-step position embedding and optional Adaptive Computation Time (a halting scalar per token, so depth varies per position). Measured: **UT base 28.9 BLEU vs Transformer base 28.0** on WMT14 En-De, at **6 steps** ("UT base, 6 steps"), with the caption "all base results have the same number of parameters" [T, 1807.03819v3 Table 7]. UT small 26.8. Caveat on that caption: UT ties attention *and* FFN across steps, so a literal same-parameter match would need a wider block; the paper does not print the parameter breakdown — **[S]**, treat "iso-params" as the authors' claim, not a verified count. Depth and per-token FLOPs *are* matched by construction (6 steps = 6 layers).

UT also reports a language-modelling perplexity win — LAMBADA, UT base 6 steps **279 dev / 319 test** vs Transformer **5122 / 7321** [T, Table 3] — but that Transformer baseline is degenerate (no usable long context in the 2018 setup), so it is **not** evidence about LM loss at our scale. Do not cite it as one.

So the credible UT result is a genuine iso-depth WMT14 win at ~65M params — the strongest ≤200M win in this document. Two caveats that matter: it is **MT BLEU, not LM loss**, and the win has **not replicated on language modelling** — MoEUT's own abstract states that "no previous work has succeeded in proposing a shared-layer Transformer design that is competitive in parameter count-dominated tasks such as language modeling" [A, 2405.16039v2]. Later UT-lineage work adds MoE to recover (MoEUT: "slightly outperforms parameter-matched" models, "large margin" at equal compute — figure only, no table [F, 2405.16039v2 §3]).

### 1.2 Looped / recurrent-depth transformers — measured on LM loss, and they lose

| Paper | Setup | Result |
|---|---|---|
| **iso-depth** 2604.21106v3 | 20 effective layers, r∈{1,2,4,8}, 4.64e17–2.15e19 FLOPs, FineWeb-Edu, 116 runs | **φ=0.46**; looped frontier trails **+0.03…0.12 nats**, growing with r, **widening at low budgets**; extrapolation to 4e20 FLOPs: r=4 still **+0.061 nats** [T, §4.2, Table 2] |
| **Ouro** 2510.25741v6 | 1.4B/2.6B, 7.7T tokens | Matches "up to 12B" — **by unique-parameter count, at 4× the effective depth**; Sparse Layers calls this out explicitly: "they compare performance by the number of unique parameters, but give their models much more compute" [T, 2605.09165v2 §7] |
| **Huginn** (Geiping et al.) 2502.05171 | 3.5B, 800B tokens, 4 recurrences | Consumes pretraining FLOPs "close to a 32B transformer", test-time scaling to 50B-equivalent; **no dense control at matched FLOPs** — a test-time-compute result, not a BPB result [T, §5.1] |
| **Parcae** 2604.12946 | 1.3B, fixed **params and data** | **+2.99 CORE points** over strong Transformer baselines; but the looped model gets more FLOPs per token, so this is a *parameter-efficiency* win, not a compute-matched one. Their own law: "looping and data should be increased in tandem, given a fixed FLOP budget" [T, abstract, §1] |
| **DeepLoop** 2607.13491 | GPT-2 small **124M**, 50B tokens, R∈{1,3,5,7} | Fixes residual scaling for tied depth: R=3 **2.8077 → 2.7917**, R=5 **2.7910 → 2.7679**, R=7 **2.7700 → 2.7514**; R=1 neutral (+0.0004) [T, Table 5.1]. Within-looped only: the R=7 row burns **7× the FLOPs** of R=1, so it is not a win over dense |

### 1.3 Mixture-of-Recursions (MoR) and Mixture-of-Depths (MoD)

**MoR** = shared block + a router that gives each token its own recursion depth (plus KV sharing across recursions). This is the only paper with a **full isoFLOP grid that includes both 135M and 360M** [T, 2507.10524v3 Table 9/10]:

| Scale | Budget | Vanilla (unique) | Plain recursive | MoR |
|---|---|---|---|---|
| 135M | 2.0e18 | **3.0922** (106M) | 3.2058 (42M, 1+10+1 ×3) | 3.1077 |
| 360M | 2.0e18 | 3.3785 (315M) | 3.4864 (118M) | **3.1026** |
| 360M | 16.5e18 | 2.7824 | 2.8466 | 2.7924 (tie) |

Plain recursion loses at **all four scales** (135M +0.114, 360M +0.108, 730M +0.097, 1.7B +0.193 at 2e18). MoR's routing only crosses the baseline at **360M and above**, and at 16.5e18 even MoR is a tie. **The recursion crossover is above us.** Table 10 is the more interesting table for our question because it has parameter-matched dense controls — see §3.2.

**MoD** is not weight reuse: every layer keeps its own weights, and a router sends only some tokens through each block. It measures compute allocation, not tying. IsoFLOP: "**improves upon vanilla transformers by as much as 1.5%** on final log probability for equivalent training FLOPs, and taking an equivalent amount of wall-clock time", or alternatively "**training loss parity with an isoFLOP optimal vanilla** while using upward of 50% fewer FLOPs per forward pass"; 60M–3B params [T, 2404.02258 §1, §4.1]. Verdict at isoFLOP: **small win**; at iso-step: parity for half the FLOPs. Nothing here reuses weights, so it does not rescue a loop.

### 1.4 Weight-tied / ALBERT-style sharing, and partial tying (`loop_blocks`-style)

ALBERT is the cleanest controlled measurement of *what* to tie, at 12 layers and 108M params [T, 1909.11942v2 Table 4, Avg over SQuAD1.1/2.0/MNLI/SST-2/RACE]:

| Sharing | Params | Avg | Δ vs not-shared |
|---|---|---|---|
| not-shared (BERT-style) | 108M | **82.3** | — |
| attention shared only | 83M | 81.6 | **−0.7** |
| FFN shared only | 57M | 79.5 | **−2.8** |
| all shared (ALBERT) | 31M | 79.8 | **−2.5** |

Two facts to take away: all-layer tying costs **−2.5 points** at this size, and **the FFN is where tying hurts** (sharing only the FFN ≈ all-shared; sharing only attention nearly recovers the untied model). This is the same diagnosis as Sparse Layers (§3.1): the dense FFN is the expressivity bottleneck under looping.

MoR's Table 10 ablates *which* layers to tie at 135M/360M and 10B tokens: **Middle-Cycle** (unique first + last layer, loop the middle) is consistently best; **Sequence** sharing is worst in all four settings (e.g. 135M r=3: M-Cyc 3.1048 vs Seq 3.1637 vs Vanilla 30L 3.0323) [T]. That is the closest thing to evidence for a `prelude/coda` + `loop_blocks` topology: keeping the first and last layer unique is worth **~0.06 NLL** over tying everything in sequence.

### 1.5 LoRA-per-iteration / iteration embeddings (cheap differentiation of a shared block)

Three independent measurements, and they disagree in an informative way:

- **Layer-wise LoRA on a converted model works** (Relaxed Recursive Transformers): Gemma 2B → 1B recursive (two blocks), rank-512 depth-wise LoRA with SVD init reaches **58.4 vs 58.6** for the full 2B model; SVD initialisation is worth **up to +6.5 points** over zero-init; average-init of the tied layers beats stepwise-init under relaxation [T, 2410.20672v3 §3.3, §3.5]. But this is **uptraining a pretrained model**, not from-scratch isoFLOP, and it is recovering a model that already had the capacity.
- **Per-step LoRA is a null at 1.7B** (Closing the Loop): "step-specific LoRA adapters are optional… they provide little benefit: training losses nearly match the shared model, and benchmark differences stay within two points with neither variant consistently ahead" [T, 2610.00673v1 §4.2–4.4].
- **Iteration/time conditioning helps only under depth extrapolation**: at Qwen3-0.6B, branch-gating at inference depth 56 scores **38.46 vs 36.59** for the unconditioned loop — but the same scheme is "much smaller or negative" for other depth allocations, and gains at training depth are not guaranteed [T, 2609.36636v1 §5.2]. We run a **fixed** 3 iterations, i.e. exactly the regime where this buys nothing.

### 1.6 Papers that report looping WINNING at ≤200M params, with the baseline number

Exhaustive, and short:

| # | Paper | Baseline | Looped | Delta | Matched? |
|---|---|---|---|---|---|
| 1 | UT, 1807.03819 Table 7 | Transformer base **28.0 BLEU** (WMT14) | UT base **28.9** | **+0.9 BLEU** | iso-depth (6 steps = 6 layers); params matched per caption, not printed |
| 2 | MoR, 2507.10524 Table 10 | Vanilla 135M/30L **3.0323 NLL** (10B tok) | Cycle-2, 15L ×2, 53M **3.0071** | **−0.0251** | **iso-FLOP + iso-token** |
| 3 | MoR, Table 10 | same | M-Cyc-2 (1+14+1), 57M **3.0330** | +0.0007 | iso-FLOP + iso-token |
| 4 | DeepLoop, 2607.13491 Table 5.1 | pre-LN loop, 124M, R=3 **2.8077** | DeepLoop R=3 **2.7917** | **−0.0160** | iso-token, **not iso-FLOP**; no dense control |

Nothing else. Every other ≤200M number in the literature is a loss or has no dense control. Note what #1 and #2 have in common: **the loop keeps 50–100% of the unique layers** (UT: 6 passes of a 6-layer block; MoR: 15 unique layers looped twice). Neither is "one layer, three times".

---

## 2. Master table

`Matched?` separates the three things papers conflate: **isoFLOP** (same training FLOPs), **params+tokens** (same size and data, but the looped model spends more FLOPs), **none** (no dense control).

| Paper (arXiv, date) | Params | Tokens | Baseline (dense depth) | Looped variant | Metric | Δ | Compute-matched? | Param-matched? | iso-FLOP verdict |
|---|---|---|---|---|---|---|---|---|---|
| UT 1807.03819v3, 2018-07-10 [T] | base, d=512, 6 steps | WMT14 En-De | Transformer base 6L **28.0** | UT 1 block ×6 | BLEU | **+0.9** | yes (6 steps = 6 layers) | claimed (not printed) | **WIN** (MT only) |
| MoR 2507.10524v3, 2025-07-14 [T] | 135M (106M non-emb) | 10B | Vanilla 30L **3.0323** | Cycle-2, 15L ×2, 53M | NLL | **−0.0251** | yes | half | **WIN** (marginal) |
| MoR same [T] | 135M | 10B | 3.0323 | M-Cyc-3, 1+9+1 ×3, 39M | NLL | **+0.0725** | yes | 37% | **LOSS** |
| MoR same [T] | 135M | 6.5B | Vanilla 30L **3.0922** | Recursive 1+10+1 ×3, 42M | NLL | **+0.1136** | yes (2e18) | 40% | **LOSS** |
| MoR same [T] | 360M | 2.4B | Vanilla 32L **3.3785** | Recursive 1+10+1 ×3, 118M | NLL | **+0.1079** | yes (2e18) | 37% | **LOSS** |
| MoR same [T] | 360M | 2.4B | 3.3785 | MoR (routed), 118M | NLL | **−0.2759** | yes (2e18) | 37% | **WIN** (routing, ≥360M) |
| iso-depth 2604.21106v3, 2026-05-07 [T] | 30–400M, 20 eff. layers | 6 budgets 4.64e17–2.15e19 | r=1 same width/depth | r=2 / r=4 / r=8 | nats | **+0.03…0.06 / +0.05…0.08 / +0.09…0.12** | **yes, by construction** | 61/41/31% | **LOSS at every r** |
| iso-depth same [T] | — | 4e20 (extrapolation) | r=1, s=34 | r=4, s=34 | nats | **+0.061** | ~yes | 40% | **LOSS** |
| Sparse Layers 2605.09165v2, 2026-05-09 [T] | 168M looped / 246M base, 16 eff. layers, d=704 | 1e18 FLOPs | Base Core9 **38.7** | Looped **37.4** | OLMES Core9 | **−1.3 pts** | yes | 68% | **LOSS** |
| Sparse Layers same [T] | 216M | 1e18 | 38.7 | Looped-MoE **39.6** | Core9 | **+0.9 pts** | yes | 88% | **WIN** (MoE, not loop) |
| LOOM 2610.01153v1, 2026-10-01 [T] | **700M total** (10L, d=512, E=80) | 10B | 1L top-26 **18.36 ppl** | 5 loops top-4 | ppl | **−1.82 (−9.9%)** | **near-isoFLOP** f=90 vs 84 (+7%) | yes | **WIN** (needs MoE + per-loop routers) |
| LOOM same [T] | ~100M | ~5B | 6 eff. **26.26** | native loop 3× **27.61** | ppl | **+1.35** | no (18 vs 6 depth) | — | **LOSS** (native loop) |
| Parcae 2604.12946v1, 2026-04-14 [T] | 1.3B | — | Transformer baseline | Parcae looped | CORE | **+2.99** | **no** (fixed params+data) | yes | **WIN on params, not FLOPs** |
| DeepLoop 2607.13491v2, 2026-07-15 [T] | 124M / 355M | 50B | pre-LN loop R=3 **2.8077** | DeepLoop R=3 **2.7917** | nats | **−0.0160** (R=7 −0.0186) | no (same tokens, 3–7× FLOPs) | yes | **within-loop fix, not a dense win** |
| ALBERT 1909.11942v2, 2019-09-26 [T] | 108M, 12L | BERT corpus | not-shared **82.3** | all-shared 31M **79.8** | GLUE-ish avg | **−2.5** | no (params only) | — | **LOSS**; attn-only 83M **81.6** (−0.7) |
| Relaxed Recursive 2410.20672v3, 2024-10-28 [T] | Gemma 1B recursive (from 2B) | 60B uptrain | full 2B **58.6** | rank-512 LoRA **58.4** | few-shot avg | **−0.2** | no (conversion) | yes | **recovery, not a from-scratch win** |
| Closing the Loop 2610.00673v1, 2026-09-30 [T] | 1.7B (Qwen3) | 60–84B | continued dense same steps | loop T=3 | train loss / bench | naive **+0.01/+0.02 nats** on steps 2/3; scalar fix **−0.02** | no dense isoFLOP | yes | **fix, not a win** |
| MoD 2404.02258v1, 2024-04-02 [T] | 60M–3B | 6e18–1e20 | isoFLOP-optimal vanilla | MoD 12.5% capacity | log-prob | **+1.5%** (or parity at 50% FLOPs) | yes | no | **small WIN** (not weight reuse) |
| MoEUT 2405.16039v2, 2024-05-25 [F] | 44M–719M | 1e5 steps C4 | same-param dense | MoEUT | ppl | "slightly outperforms"; "large margin at equal compute" | partially | yes | **WIN** (MoE + loop, figures only) |
| Huginn 2502.05171v1, 2025-02-07 [T] | 3.5B | 800B | — | 4 recurrences | reasoning | no dense control | **no** | — | **NM** |

**Count of the verdict column at ≤200M with a dense control: 2 wins (UT on MT; MoR cycle-2 by 0.025 NLL), 5+ losses.** Both wins keep ≥50% of the unique layers.

---

## 3. Why loops lose at small scale

The literature does say so, and it gives four distinct mechanisms — not one.

### 3.1 Capacity: `r^φ` < `r`, so per-FLOP capacity falls as you loop more

This is the quantitative core, and it is measured at our exact budgets. φ=0.46 (CI [0.41, 0.53]) means the recurrent block contributes `r^0.46` times its unique parameters: **1.9× at r=4, 2.6× at r=8**, while costing **4× and 8× the FLOPs** [T, 2604.21106v3 §4.3]. Consequence: the looped compute-optimal model must **widen** to compensate, which raises per-token FLOPs and **lowers the optimal token count** — the looped optimum is wider, sees fewer tokens, and still lands above the dense frontier (Fig. 3). And "**the gap widens at lower budgets and flattens at our largest**" [T, §4.2] — i.e. small scale is the worst case for loops, not the best.

Same mechanism seen from the parameter side in MoR: at 135M, cutting unique params 106M → 42M at fixed effective depth costs **+0.11 nats**, and the deficit is *not* recovered by routing until 360M [T, 2507.10524 Table 9]. And in ALBERT at 108M, cutting 108M → 31M costs **−2.5 points** [T, Table 4]. Parameter cuts at small N are expensive because the `AN^−α` term is steep exactly there.

### 3.2 The dense FFN is the expressivity bottleneck under tying

Sparse Layers' hypothesis, with a controlled test at 1e18 FLOPs and 16 effective layers: the FFN "applies the same operation to every token, an invariance that compounds when looped". Swap the dense FFN for a top-2-of-8 MoE **and nothing else**, and the isoFLOP ordering flips: Looped **37.4** < Base **38.7** < Looped-MoE **39.6** Core9, at 168M / 246M / 216M stored params [T, 2605.09165v2 Table 4]. Mechanism check: across passes, **25–53% of tokens get fully disjoint expert sets** and only 4–14% get identical ones — the shared layer does different work on each pass [T, §7.1]. That is also why ALBERT's "share attention only" (81.6) is so much cheaper than "share FFN only" (79.5): **tying the FFN is the expensive half**.

This is the strongest argument that *our* loop is the wrong shape: we tie a block whose FFN sits in a 256-expert MoE bank (`n_experts=256`) run 3×, so the router can in principle differentiate passes — but the routing is per-token and the *weights* are shared, so the capacity argument in §3.1 still binds.

### 3.3 Optimization: tied depth breaks residual scaling, and truncation breaks the loop's learning signal

- **Residual scaling.** Under tying, "one shared update aggregates gradients from repeated visits and is read back by those same visits"; the first-order bound requires the DeepNorm exponent to go **from 1/4 to 1/2 as loop count grows at fixed physical depth**, and the fix is worth **−0.016 / −0.023 / −0.019 nats** at R=3/5/7 on GPT-2 small, **−0.015 / −0.020 / −0.028** on GPT-2 medium, with R=1 neutral (+0.0004/+0.0011) [T, 2607.13491v2 Table 5.1]. A correctly-scaled loop is better than a badly-scaled one — it still is not better than dense at isoFLOP.
- **Variance growth.** LOOM: "hidden-state variance grows with each iteration as residual updates accumulate, which destabilizes deep recurrence"; unmodified loops at ~100M hit **ppl 1311 at 6 loops, 1312 at 9, 1587 at 12**, and residual scaling alone still spikes at 9+ [T, 2610.01153v1 Table 3]. Stabilisation is a prerequisite, not a win.
- **Gradient interference / truncation.** Truncated BPTT (`r_bwd = ⌈r/2⌉`) *lowers validation loss* but **drops φ from 0.45 to 0.38**: the early loops run forward-only and get no accurate signal, so each recurrence becomes worth less [T, 2604.21106v3 Table 3, §5.1]. Conversely, a training recipe that keeps every loop supervised raises the value of each loop. **Our loop is full BPTT — `loopblock.py:1568` states "not truncated BPTT: nothing is detached" — so we are not paying this tax.** That is one thing already right.

### 3.4 Effective depth is not the same thing as unique parameters, and the difference is measurable

The cleanest demonstration is MoR Table 10, which is the only table in the literature with **both** a param-matched and a FLOP-matched dense control at 135M:

| Model at 135M, 10B tokens | Unique params | Effective depth | NLL |
|---|---|---|---|
| Vanilla 30L | 106M | 30 | **3.0323** |
| Vanilla 15L | 53M | 15 | 3.0818 |
| Vanilla 10L | 35M | 10 | 3.1582 |
| Cycle-2 (15L ×2) | 53M | 30 | **3.0071** |
| M-Cyc-3 (1+9+1 ×3) | 39M | 32 | 3.1048 |

Read the last two rows against the first three. **Cycle-2 beats both the 30L dense (its FLOP twin) and the 15L dense (its param twin)** — depth reuse at r=2 is a real, if tiny, win. **M-Cyc-3 lands between the 10L and 15L dense models in quality while running 32 layer-passes**: effective depth 32 buys the quality of ~12–13 unique layers. That is φ≈0.46 visible in a single table, and it is the exact config family we run (3 iterations), one step more extreme (1 unique layer instead of 10).

So: unique parameters set the ceiling, loops only approach it. "Effective depth alone is insufficient to predict behavior" is also the conclusion of the controlled 0.6B/1B study, which finds recurrence helps reasoning beyond the training horizon but **degrades knowledge**, and that the *placement* of recurrent vs non-recurrent layers matters more than the loop count [T, 2609.36636v1 abstract, §4–5].

---

## 4. The cheap fixes that make tied depth behave more like untied

All deltas below are measured, in the paper's own units. **None closes the gap; the best one halves it.**

| Fix | Mechanism | Measured delta | Scale / match | Source |
|---|---|---|---|---|
| **Input re-injection (1 scalar)** | `z_t = w·e + (1−w)·h_{t−1}`, w=σ(a), **one parameter** | naive loop: steps 2/3 are **+0.01/+0.02 nats worse** than step 1; with the scalar they are **−0.02 nats better** | Qwen3-1.7B, T=3, conversion | [T] 2610.00673 §4.1 |
| **Per-depth residual scale (DeepLoop)** | α=(2N)^½, β=(8N)^−½, N = unrolled depth | **−0.016/−0.023/−0.019 nats** (R=3/5/7) at GPT-2 small; **−0.015/−0.020/−0.028** at medium; R=1 neutral | 124M/355M, 50B tok | [T] 2607.13491 Table 5.1 |
| **Hyperconnections** | K parallel residual lanes mixed across loops | **φ 0.45 → 0.65**; r=2 "matches or beats r=1 at some budgets"; still <1, so the frontier still trails | 20 eff. layers, 4.64e17–4.64e18 | [T] 2604.21106 Table 3, §5.2 |
| **Per-loop routers / loop-specific MoE routing** | different experts per pass | LOOM component ablation at 350M/9 loops: **18.62 ppl** full recipe → **19.83** w/o routing refresh, **19.25** w/o looping residual, **24.63** w/o residual scale, **26.05** w/o embed injection | 350M, 5k steps | [T] 2610.01153 Table 5 |
| **Iteration embeddings / timestep gating** | LG/BG/AdaLN on the iteration index | at inference depth 56: BG **38.46** vs BaseLoop **36.59**, AdaLN 38.01; "much smaller or negative" at other depth allocations; gains are at **extrapolation**, not training depth | Qwen3-0.6B | [T] 2609.36636 §5.2 |
| **Per-iteration LoRA** | depth-wise low-rank adapters | **null**: "training losses nearly match the shared model… within two points, neither ahead" | Qwen3-1.7B, T=3 | [T] 2610.00673 §4.4 |
| **Per-iteration LoRA, SVD-initialised** | rank-r adapters approximating the untied weights | **+6.5 points** over zero-init at rank 512; rank-512 recovers **58.4 vs 58.6** of the full model | Gemma 2B→1B, uptrain | [T] 2410.20672 §3.5 |
| **Partial tying (unique first/last)** | Middle-Cycle topology | 135M r=3: M-Cyc **3.1048** vs Sequence **3.1637**; 360M r=3: **2.8760** vs **3.0245** | 135M/360M, 10B tok | [T] 2507.10524 Table 10 |
| **Truncated BPTT** (anti-fix) | detach all but last ⌈r/2⌉ loops | **φ 0.45 → 0.38** (lower val loss, but each loop is worth less — a hidden inference-cost trade) | same grid | [T] 2604.21106 Table 3 |
| **Stochastic depth per iteration** | randomly drop loop passes | **no measurement found** — the nearest published mechanisms are layer dropout (LayerSkip) and MoD's token routing, neither of which is per-iteration depth dropout in a tied loop. Repo has a `rand_depth` flag; its effect is unmeasured here. **[S]** | — | — |

**What this table says for us.** The two fixes that are already in the repo and free — `prelude/coda` topology and LayerRoPE-style per-depth norm gain (`config.py:504-510`) — map to the *partial tying* row (−0.06 NLL vs all-in-sequence) and the *per-depth scale* row (−0.02 nats). Neither flips the sign. The two fixes with the largest published effect (hyperconnections +0.2 φ; MoE routing) require new modules, and hyperconnections still lands at φ=0.65, i.e. **2.04 layer-equivalents for our 1×3 block against 12 for the control**.

And the cheapest fix of all — one scalar that re-injects the embedding — is the one that turns a *degrading* loop into a *refining* one. If we ever run a loop again, that scalar is the first thing to add; it costs one parameter.

---

## 5. Recommendation

### 5.1 The plain answer

**Loops lose at our scale. Do not spend the 1.5 GB × 3-seed A/B on a depth-recurrent arm.**

Three independent reasons, in decreasing order of strength:

1. **The compute-matched evidence at exactly our scale is negative and unanimous.** iso-depth (20 layers, 30–400M params, 116 runs): φ=0.46, looped frontier trails at every r, gap **widens at low budgets** [T]. MoR at 135M: **+0.11 nats** for 3 loops, and routing only crosses at 360M [T]. Sparse Layers at 1e18 FLOPs: looped **37.4 vs base 38.7** [T]. There is no published iso-FLOP BPB win for depth recurrence in 20–200M.
2. **A tie would not help us.** The loop's only product is parameter efficiency, at a FLOP premium. At 85M params on 16 GB, parameters are not the binding resource; bytes and FLOPs are. Buying a currency we don't need with the one we're short of is a bad trade even at parity.
3. **Our own 0.13–0.37 BPB is 3–10× larger than the published tying penalty, so it is not a tying problem.** At our scale the literature's loop penalty is ~1–2.5% relative ≈ **0.015–0.045 BPB** on a ~1.4 BPB model. The rest of our gap is the block (KDA/Engram/256-expert MoE at 17M active params) and the 1-layer effective depth — the subject of `2026-11-06-architecture-at-100m.md`, not of this one. **Fixing the tie cannot recover a gap that tying did not create.**

### 5.2 If the owner overrules this, here is the only arm worth running

Not "one layer × 3". The least-bad looped config the literature supports is **r=2 with a deep recurrent block and unique prelude/coda** (MoR's cycle-2 shape, which is the only iso-FLOP ≤200M win). All fields already exist in `config.py`: `prelude_layers`, `coda_layers`, `loop_blocks`, `max_iter`.

| Arm | Config | Layer-passes | Unique layers | Params | MFLOP/byte | Predicted ΔBPB vs C0 | Confidence |
|---|---|---|---|---|---|---|---|
| **C0 control** | 12L, d=768, 4 KB | 12 | 12 | **85.1M** | **964** | — | — |
| **L2** | prelude 2 + **4 × 2** + coda 2 | 12 | 8 | 56.7M | ~996 | **+0.020 [+0.014, +0.032]** | medium |
| **L3** | prelude 1 + **3 × 3** + coda 2 | 12 | 6 | 42.6M | ~1012 | **+0.028 [+0.020, +0.042]** | medium |
| **L2+HC** | L2 + hyperconnections K=2 | 12 | 8 | ~57M | ~1010 | **0.000 [−0.010, +0.015]** | low |

Params = C0 × unique/12; MFLOP/byte = 964 × passes/12 × (1 + r/120) injection overhead, the same accounting as [T, 2604.21106v3 §3.2]; BPB prediction = the paper's **relative** gap band (1.0–2.4% at r=2, 1.7–2.8% between r=2 and r=4) applied to our control's ~1.4 BPB, because nats/token does not transfer across tokenizers (their Llama-2 subword vs our bytes) — **[S]**, stated as such. The absolute-nats reading would predict 0.04–0.09 BPB, the relative reading 0.015–0.04; I use the relative one because our BPB per byte is not their 2.5–2.9 nats per token.

**Cost:** 0.964 GFLOP/byte × 1.5 GB = 1.45e18 FLOPs/seed ≈ **27 h/seed at 15 TFLOP/s**, ×3 seeds = **80 h per arm**. Two arms is a week of the card for a predicted loss.

**Pre-registered falsifier (write it before the run):**
> L2+HC is the only arm that can survive: the φ model with hyperconnections predicts a tie. **If its 3-seed mean BPB is ≥ 0.03 better than C0** (our resolution), the φ=0.46–0.65 model is wrong at byte level and a 4-loop arm becomes worth testing. **If it lands in [−0.03, +0.05] — tie to loss — the loop lane is closed permanently at this scale.** If it is > 0.05 worse, the block, not the tying, is confirmed as the defect.

My prediction: it lands at **+0.00 to +0.02**, i.e. a tie, and the 80 h teaches us nothing we can ship. That is why the recommendation is to not run it.

### 5.3 What would change the verdict

- **If the card becomes memory-bound** (weights or KV bind before FLOPs — e.g. serving ≥1B on 16 GB), loops regain value as a *parameter* trick, and φ=0.46 says a 4-loop model is worth 58% of a dense model at the same width. That is a deployment argument, not a training-BPB argument.
- **If the objective switches to reasoning at test time** (Huginn/Ouro/LOOM/Closing-the-Loop axis), loops are the right structure and the relevant metric is pass@k under a depth budget, not BPB. Ouro and Huginn are 1.4–3.5B; the controlled 0.6B study finds recurrence helps reasoning beyond the training horizon and **hurts knowledge** [T, 2609.36636v1].
- **If an intervention with φ > 0.8 appears at ≤200M.** None exists today; the best measured is 0.65.

---

## 6. VERIFIED (sourced)

1. φ = 0.46, CI [0.41, 0.53], 20 effective layers, r∈{1,2,4,8}, 6 budgets 4.64e17–2.15e19, 116 runs, gap +0.03…0.12 nats growing with r and widening at low budgets — [T] 2604.21106v3 Table 2, §4.2–4.3.
2. At r=4, 410M looped ties 580M dense but costs the FLOPs of 1B dense — [T] 2604.21106v3 Appendix F.4.
3. Truncated BPTT lowers φ to 0.38; hyperconnections raise it to 0.65 and r=2 can match/beat r=1 at some budgets — [T] 2604.21106v3 Table 3.
4. MoR 135M/2e18: 3.0922 vanilla vs 3.2058 recursive (+0.1136); 360M: 3.3785 vs 3.1026 MoR (−0.2759) — [T] 2507.10524v3 Table 9.
5. Cycle-2 at 135M/10B: 3.0071 vs vanilla 30L 3.0323 (−0.0251); M-Cyc-3 3.1048 (+0.0725); Sequence sharing worst — [T] 2507.10524v3 Table 10.
6. Sparse Layers at 1e18 FLOPs, 16 effective layers, d=704: Looped 168M Core9 37.4 < Base 246M 38.7 < Looped-MoE 216M 39.6; 25–53% disjoint expert routing across passes — [T] 2605.09165v2 Table 4, §7.1.
7. LOOM: 700M total, 10B tokens, near-isoFLOP f=90 vs 84, 5 loops 18.36 → 16.54 ppl; native loop at ~100M degrades 26.26 → 27.61 at 3× and spikes at 6+ (1311) — [T] 2610.01153v1 Tables 2, 3.
8. DeepLoop at GPT-2 small/medium: −0.016/−0.023/−0.019 and −0.015/−0.020/−0.028 nats at R=3/5/7, R=1 neutral — [T] 2607.13491v2 Table 5.1.
9. ALBERT 12L: not-shared 108M 82.3, all-shared 31M 79.8, attention-only 83M 81.6, FFN-only 57M 79.5 — [T] 1909.11942v2 Table 4.
10. Relaxed Recursive: rank-512 LoRA 58.4 vs 58.6 full Gemma 2B; SVD init +6.5 points — [T] 2410.20672v3 §3.3, §3.5.
11. Closing the Loop: naive T=3 degrades steps 2/3 by +0.01/+0.02 nats; one scalar makes them −0.02; per-step LoRA-32 is a null within 2 points — [T] 2610.00673v1 §4.1–4.4.
12. UT base 28.9 vs Transformer base 28.0 BLEU, same params, WMT14 — [T] 1807.03819v3 Table 7.
13. MoD: +1.5% log-prob at isoFLOP, or parity at ≤50% FLOPs — [T] 2404.02258v1 §1.
14. Parcae: +2.99 CORE at 1.3B at fixed params and data, not fixed FLOPs; "looping and data should be increased in tandem" — [T] 2604.12946v1 abstract/§1.
15. Our loop is full BPTT (`loopblock.py:1568`: "not truncated BPTT: nothing is detached") — [ours].
16. Flagship: d=1024, `max_iter=3`, `loop_blocks=1`, `prelude_layers=0`, `coda_layers=0`, 102.7M total / 17.26M active MAC/token / 137.42 MFLOP/B; 13.5M dense control at 135.36 MFLOP/B (−1.5%), BPB 1.7423→1.7292 flagship vs 1.492 control — [ours] `docs/records/reviews/2026-11-06-control-transformer.md`.

## 7. SPECULATION (my inference)

- **[S1]** Applying φ=0.46 to a **1-layer** recurrent block (the paper's smallest is 2 unique layers) gives 3^0.46 = 1.66 layer-equivalents. The exponent is fit on 4–16-layer blocks; a 1-layer block may be worse than the fit, not better, because it cannot host the depth-wise specialisation the fit averages over. Direction: our 14% capacity estimate is, if anything, optimistic.
- **[S2]** The BPB transfer uses the paper's **relative** gap (1–2.5%) because nats per subword token does not convert to nats per byte. If the mechanism is per-token rather than per-unit-of-capacity, our byte-level model could see 2–4× that absolute number (0.04–0.09 BPB). I use the relative one; the recommendation does not change under either.
- **[S3]** Our measured 0.13–0.37 BPB is dominated by the block (KDA/Engram/256-expert MoE) and by the 1-layer depth, not by tying, because the published tying penalty at our scale is an order of magnitude smaller. Consistent with `2026-11-06-architecture-at-100m.md`, but not proven by it.
- **[S4]** L2's ~996 and L3's ~1012 MFLOP/byte assume each layer-pass costs the same as a control layer at d=768 and that injection adds r/120; the control's 964 is itself a closed-form estimate, not a measured `FlopCounterMode` number for that arm. Measure before quoting.
- **[S5]** Per-iteration stochastic depth: no published measurement found in a tied loop; the nearest evidence (LayerSkip's layer dropout, MoD's token routing) does not transfer. Untested here.

## 8. Open questions

1. Does φ hold for a **1-layer** block at 3 passes, or does it collapse further? A 3-point sweep (1×3, 2×3, 4×3 at 20M params, 300M bytes) would settle it for ~6 h of card — cheaper than the A/B, and it answers the mechanism rather than the config.
2. Does the **byte-level** objective change φ? All φ evidence is subword. Byte-level models have 4× the tokens per unit text and a much flatter loss surface; the tying penalty could be relatively smaller (more tokens per parameter) or larger (no lexical prior to lean on). **Unmeasured anywhere.**
3. Would **MoE-per-loop routing inside our existing 256-expert bank** raise φ at 85M? LOOM says yes at 700M with per-loop routers; Sparse Layers says MoE is the fix at 1e18 FLOPs. This is the one loop-adjacent arm with a published positive, and it is really a routing experiment.
4. Is our 0.13–0.37 gap reproducible at **iso-FLOP with a widened loop** (e.g. d=1536, 3 passes, ~85M params)? If the gap collapses to ~0.02, our loop was never the problem and the block is; if it stays >0.1, the 1-layer depth is doing damage the φ law underestimates.

---

## Sources

1. Dehghani et al., *Universal Transformers*, arXiv:1807.03819v3, 2018-07-10. https://arxiv.org/abs/1807.03819
2. Bae et al., *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*, arXiv:2507.10524v3, 2025-07-14. https://arxiv.org/abs/2507.10524
3. Schwethelm, Rückert, Kaissis, *How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models*, arXiv:2604.21106v3, 2026-05-07. https://arxiv.org/abs/2604.21106
4. Lee, Biloki, Hu, May, *Sparse Layers are Critical to Scaling Looped Language Models*, arXiv:2605.09165v2, 2026-05-09. https://arxiv.org/abs/2605.09165
5. He et al., *Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts* (LOOM), arXiv:2610.01153v1, 2026-10-01. https://arxiv.org/abs/2610.01153
6. Prairie, Novack, Berg-Kirkpatrick, Fu, *Parcae: Scaling Laws For Stable Looped Language Models*, arXiv:2604.12946v1, 2026-04-14. https://arxiv.org/abs/2604.12946
7. Li et al., *DeepLoop: Depth Scaling for Looped Transformers*, arXiv:2607.13491v2, 2026-07-15. https://arxiv.org/abs/2607.13491
8. Lan et al., *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*, arXiv:1909.11942v2, 2019-09-26. https://arxiv.org/abs/1909.11942
9. Bae et al., *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA*, arXiv:2410.20672v3, 2024-10-28. https://arxiv.org/abs/2410.20672
10. Marchenko et al., *Closing the Loop: Practical Training Recipes for Looped Language Models*, arXiv:2610.00673v1, 2026-09-30. https://arxiv.org/abs/2610.00673
11. Zhuang, Wang, Razzak, Liu, *What Makes Recurrence Effective in Looped Language Models?*, arXiv:2609.36636v1, 2026-09-29. https://arxiv.org/abs/2609.36636
12. Raposo et al., *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*, arXiv:2404.02258v1, 2024-04-02. https://arxiv.org/abs/2404.02258
13. Csordás et al., *MoEUT: Mixture-of-Experts Universal Transformers*, arXiv:2405.16039v2, 2024-05-25. https://arxiv.org/abs/2405.16039
14. Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn), arXiv:2502.05171v1, 2025-02-07. https://arxiv.org/abs/2502.05171
15. Zhu et al., *Scaling Latent Reasoning via Looped Language Models* (Ouro), arXiv:2510.25741v6, 2025-10-29. https://arxiv.org/abs/2510.25741
16. Repo: `docs/records/reviews/2026-11-06-control-transformer.md`, `docs/records/research/2026-11-06-architecture-at-100m.md`, `port/configs/flagship_base.toml`, `port/src/dormouse/config.py:370-520,910-913`, `port/src/dormouse/model/loopblock.py:1568`.
