# FLOPs per byte at 100M: what the record measures, and what it does not

**Date:** 2026-11-06. **Scope:** one question — which mechanisms cut compute per byte at equal or better quality, and which reach <=80 MFLOP/byte at 100M stored params on one 16 GB card. **Read-only:** arXiv MCP only; no GPU run, no `port/src` edit. Every number below was read from the named paper's table or text; the paper's version and first-submission date are in the row.

**Marks:** **[T]** table read · **[A]** abstract/text only · **[C]** computed from the stated baseline, not from a paper · **[S]** SPECULATION, my inference. `NOT MEASURED` means the paper does not report that quantity; it is not zero.

**Accounting box (read this before the ratios).** The given 137.4 MFLOP/byte is exactly `6 × 102.7M / 4.485` [C]. I therefore treat it as the standard `6N` training-FLOPs accounting at ~4.485 bytes/patch, with the attention-core term (`4·L·d·n_layers` per token) **not** inside that number. If your 137.4 is a profiled total that already includes attention, the ratios below apply to the `6N` part only; the attention line in Table 1 is then the missing headroom, not an add-on. This assumption is the single biggest source of error in the arm arithmetic and is marked [S] wherever it is used.

**Baseline for scale:** 102.7M params, 8.5 TFLOP/s achieved, 17.9% MFU on a 47.7 TFLOP/s card. Target: 100M–1B stored, 1–22 GB text, 1–8 KB context, one 16 GB GPU.

---

## TL;DR

1. **The target is a 0.58× cut (137.4 → 80).** Under `6N` accounting the only three levers are active params per patch, bytes per patch, and the attention-core term at long context. At 100M and 4–8 KB, attention is not negligible: with 12 layers, `d=768`, the attention core is ~33.7 MFLOP/byte at 4 KB and ~67.3 at 8 KB [C]; that is 20–33% of the total once added to 137.4.
2. **Fine-grained MoE has the strongest equal-quality FLOPs evidence in the 2B–145B band, not at 100M.** DeepSeekMoE 2B reaches the Pile loss of a dense 1.89B model at **0.175× the FLOPs** (4.3T vs 24.6T per 2K tokens, loss 1.808 vs 1.806) [T, 2401.06066v1 Table 3]; 16B reaches dense-7B BPB at **0.405×** (0.74 vs 0.75) [T, Table 4]; 145B reaches 67B-dense loss at **0.285×** [T, Table 6]. The fine-grained scaling law claims MoE beats dense at every budget and needs **20× fewer FLOPs at 1e20**, >40× at 1e25 [T/A, 2402.07871v1 Fig. 1] — but its smallest measured MoE is **192M total / 3M active with 16–66B tokens**, not 100M/1–22 GB.
3. **At <=1B, Mixture-of-Depths is the only per-token adaptive-compute mechanism with a measured equal-quality FLOPs saving:** 0.60× FLOPs per forward at equal loss, 66% faster steps at 6e18, over 60M–3B params [T, 2404.02258v1 §5.1]. **Mixture-of-Recursions is negative at 135M** (+0.0155 NLL at iso-FLOPs) and only wins at 360M (−0.2759) [T, 2507.10524v3 Table 9]. Early exit works at task level (CALM up to 3×; LayerSkip 1.34–2.16× decode) but has **no FLOPs/byte number** [A, 2207.07061v2; A, 2404.16710v4].
4. **Attention hybrids are a memory/bandwidth story at 4–8 KB, not a compute story, until the context is long.** MLA cuts KV cache 93.3% and raises throughput 5.76× but reports no FLOPs/token [T, 2405.04434v5]. Kimi Linear's 3:1 KDA:MLA is 1.16× more compute-efficient at equal loss in its scaling law (0.862× FLOPs) and −0.12 PPL at equal FLOPs in its 16-layer ablation [T, 2510.26692v2 Table 1/§5.3]. SWA+global 1:1 is a **short-context tie** (38.13→38.03 at 0.22B; 40.46→41.31 at 0.66B) and a 16K win (25.09→35.33 at 0.22B) [T, 2606.15378v1 Tables 2/13].
5. **I found no paper that verifies <=40 MFLOP/byte at >=100M stored with quality matched to a 100M dense baseline at 1–22 GB.** The arithmetic candidate is a 192M-total/3M-active MoE (≈4.5 MFLOP/byte at 4 B/token), but its quality at our data budget is unmeasured and the law's runs used 16–66B tokens [S]. BLT's 0.5× is at 8B (≈69 if applied to 137.4, still not <=40), and BLT is **worse than BPE at 1B**; the 470M ps=8 model needs 150B bytes to cross [T, 2412.09871v1 Table 2/§5.1]. Scratchpad's 2.9× is off a **pure byte-level** baseline, not off a 4.5 B/patch baseline (at 100M, 616/2.9 ≈ 212, above 137.4), so it is not a candidate either [T/C, 2605.09630v1]. This is a negative search result, not a proof.
6. **Recommendation:** three arms, all forecasts, none proven at 100M. C1 (MoD 12.5% + patch 4.67 + SWA 1:1) = 79.4 MFLOP/byte, medium-low confidence; C2 (fine-grained MoE 170M total/60M active) = 80.3, low; C3 (MoE + MoD capacity-recovery, 280M/100M active) = 80.5, low. **The 35% MFU target is systems-bound, not architecture-bound:** every arm that shrinks FLOPs at 100M also shrinks the GEMMs or adds launches, so MFU is expected to stay at 10–18%, not rise to 35%, unless CUDA-graph capture/fusion cuts launch count [S].

---

## 1. Where the 137.4 goes, and what “equal quality” means

### 1.1 The three levers

`MFLOP/byte ≈ 6·N_active / patch_bytes + attention_core/byte` [C].

- **Active params per patch** — MoE, MoD, early exit, low-rank weights.
- **Bytes per patch** — byte patching (MegaByte, BLT, H-Net, ByteFlow, Scratchpad).
- **Attention core per byte** — SWA/global hybrids, linear/SSM layers; only matters at long `L`.

At 102.7M and 4.485 bytes/patch the `6N` term is 137.4. The attention-core line, computed for 12 layers and `d=768`, is [C]:

| Context | attention core, MFLOP/byte | share of 6N (137.4) | share of 6N+attention |
|---|---:|---:|---:|
| 1 KB (1024) | 8.4 | 6.1% | 5.8% (145.8 total) |
| 4 KB (4096) | 33.7 | 24.5% | 19.7% (171.1 total) |
| 8 KB (8192) | 67.3 | 49.0% | 32.9% (204.7 total) |

This table is arithmetic, not a paper result. It is the reason a “137.4 → 80” target is easy on paper at 1 KB and hard at 8 KB if the profiler counts attention.

### 1.2 Three comparisons papers conflate

| Match | What is held equal | What it answers |
|---|---|---|
| **iso-FLOP** | training/inference FLOPs | “at the same compute, which is better?” |
| **iso-param+tokens** | params and tokens; attention FLOPs differ | “at the same model size and data, which is better?” |
| **iso-quality** | loss/BPB | “at the same quality, how many FLOPs?” |

A row below says which one it is. A ratio of 0.5× is only meaningful with the match named: 0.5× iso-quality is a compute win; 0.5× iso-FLOP with a worse loss is not.

---

## 2. Mechanism table

### 2.1 Mixture-of-Experts: fine-grained, shared, top-k, capacity, balancing

| Mechanism | Paper (id, version, date) | Scale (stored / active, tokens) | Match | FLOPs metric | Quality delta | FLOPs/byte ratio | Evidence |
|---|---|---|---|---|---|---|---|
| **DeepSeekMoE 2B** (1 shared + 63 routed, top-7; expert size 0.25× FFN) | 2401.06066v1, 2024-01-11, Table 3 [T] | 2.0B / 0.3B, 100B tok | **iso-quality** vs Dense×16 (1.89B) | 4.3T vs 24.6T per 2K tok | Pile 1.808 vs 1.806 (+0.002 worse) | **0.175×** | table, equal-loss pair |
| **DeepSeekMoE 16B** | 2401.06066v1 Table 4 [T] | 16.4B / 2.8B, 2T tok | **iso-quality** vs DeepSeek 7B dense | 74.4T vs 183.5T per 4K tok | Pile BPB 0.74 vs 0.75 (−0.01 better) | **0.405×** | table, equal-quality pair |
| **DeepSeekMoE 145B** | 2401.06066v1 Table 6 [T] | 144.6B / 22.2B, 245B tok | **iso-quality** vs DeepSeek 67B dense | 585.6T vs 2057.5T per 4K tok | Pile 1.876 vs 1.905 (−0.029 better) | **0.285×** | table, equal-quality pair |
| **Fine-grained MoE scaling law** (granularity G) | 2402.07871v1, 2024-02-12, Table 2/Fig. 1 [T/A] | 64×3M (192M/3M) up to 64×100M (6.4B/100M); 16–66B tok | **law-level**; no printed matched pair | 64×100M: 4.37B tok, 2.95e18 FLOPs, loss 3.133 | law: MoE curve below dense at every budget | **0.05×** at 1e20 (20× dense budget); >0.025× at 1e25 | fitted law, not a table pair |
| **OLMoE** | 2409.02060v2, 2024-09-03, abstract [A] | 6.9B / 1.3B, 5T tok | active-param matched (no dense control) | NOT MEASURED | beats models with similar active params | NOT MEASURED | abstract only |
| **DeepSeek-V3** (1 shared + 256 routed, top-8, max 4 nodes, no token drop) | 2412.19437v2, v1 2024-12-27, §4.1.2/§6.2 [T] | 671B / 37B, 14.8T tok | no dense control | no FLOPs/token printed | strongest open base at its date | NOT MEASURED | architecture report |
| **Aux-loss-free balancing** (bias update γ=0.001) | 2412.19437v2 Table 5 [T] | 15.7B / 2.4B, 1.33T tok | same FLOPs, aux-loss vs aux-free | same | Pile BPB 0.727→0.724 (−0.003); BBH 37.3→39.3 | **1.0×** (quality gain) | ablation table |

**What the MoE evidence actually says.** The 2B/16B/145B rows are equal-quality pairs, and the ratio is the dense FLOPs divided by the MoE FLOPs at that equality: 5.7×, 2.5×, 3.5× fewer FLOPs. The mechanism is real at 2B+. The 100M question is not answered: the law's smallest total is 192M with **3M active** and **16B+ tokens**; our corpus is 1–22 GB (≈0.25–5.5B tokens at 4 B/token), 3–64× below the law's smallest runs. The paper itself says MoE “needs longer training to perform competitively but scales better after reaching that point” [T, 2402.07871v1 §5.5]. At 100M/1–22 GB the expected direction is **worse, not better**, unless the MoE is only mildly sparse.

**Capacity/top-k.** DeepSeek-V3 uses top-8 of 256, 1 shared, and **does not drop tokens** because the bias balancing holds load [T, 2412.19437v2 §4.1.2]. So “capacity factor” is not a quality knob in that recipe; the classic drop-token capacity trade is a GShard-era artifact, not a V3 result. The aux-loss-free ablation buys 0.003 BPB at 15.7B — a quality gain at fixed FLOPs, not a FLOPs cut [T, Table 5].

### 2.2 Adaptive compute per token: MoD, MoR, early exit, layer skipping

| Mechanism | Paper (id, version, date) | Scale | Match | FLOPs metric | Quality delta | FLOPs/byte ratio | Evidence |
|---|---|---|---|---|---|---|---|
| **Mixture-of-Depths** (top-k 256 = 12.5% capacity, route every other block) | 2404.02258v1, 2024-04-02, §5.1/Fig. 4 [T] | 60M–3B; 6e18, 2e19, 1e20 FLOPs | **iso-FLOPs at equal loss** | model #3: equal loss to isoFLOP-optimal baseline, 66% faster step | 0 at 6e18 (equal performance) | **0.60×** at 6e18; **1.0×** at 1e20 (optimal MoD uses same FLOPs/forward, more params, lower loss) | iso-FLOPs table/curves |
| **Mixture-of-Recursions, 135M** | 2507.10524v3, 2025-07-14, Table 9 [T] | 135M base / 42M unique, 2.0e18, 6.5B vs 9.2B tok | **iso-FLOPs** | NLL 3.0922 → 3.1077 | **+0.0155 worse** | **>1×** (equal-quality FLOPs not measured) | iso-FLOPs table |
| **Mixture-of-Recursions, 360M** | 2507.10524v3 Table 9 [T] | 360M base / 118M unique, 2.0e18, 2.4B vs 3.6B tok | **iso-FLOPs** | NLL 3.3785 → 3.1026 | **−0.2759 better** | **<1×** (not measured) | iso-FLOPs table |
| **CALM** (early exit, confidence-calibrated) | 2207.07061v2, 2022-07-14, abstract/Table 2 [A/T] | 8-layer T5; CNN/DM, WMT, SQuAD | task-level; no LM BPB | 5.73/8 decoder layers at δ=0.1 on CNN/DM; 1.41× end-to-end; up to 3× | “minor performance loss”, provable consistency | **NOT MEASURED** (FLOPs/byte) | task-level speedup |
| **LayerSkip** (layer dropout + early-exit loss + self-speculative decode) | 2404.16710v4, 2024-04-25, abstract [A] | Llama2 7B/13B; Llama3 8B/1B | task-level | 1.34–2.16× decode speedup; up to 2.16× CNN/DM, 1.82× coding | early exit alone has accuracy cost; self-speculative preserves | **NOT MEASURED** | abstract/task tables |

**The scale shape is the finding.** MoR is the cleanest controlled test: at 135M it **loses** at iso-FLOPs, at 360M it wins by 0.276 NLL, with one-third the unique params. MoD is positive at 60M–3B but its own paper says the optimal MoD at 1e20 uses **as many FLOPs per forward as the baseline** and spends the budget on parameters instead; the 0.60× is a 6e18-budget result, not a universal constant [T, 2404.02258v1 §5.1]. Early exit has task-level speedups but no per-byte language-model number, and its speedup is decode-bound, not training-FLOP-bound.

### 2.3 Attention: MLA, low-rank KV, SWA+global, linear/SSM hybrids

| Mechanism | Paper (id, version, date) | Scale | Match | FLOPs metric | Quality delta | FLOPs/byte ratio | Evidence |
|---|---|---|---|---|---|---|---|
| **MLA** (low-rank KV joint compression; decoupled RoPE) | 2405.04434v5, 2024-05-07, abstract/§2.1.4 [T] | 236B / 21B, 8.1T tok | vs DeepSeek 67B dense (MoE confound) | no FLOPs/token; KV cache −93.3%; throughput 5.76×; training cost −42.5% | stronger than 67B | **NOT MEASURED for FLOPs**; memory 0.067× | deployment numbers, not arithmetic |
| **SWA + global, 1:1** (window 128) | 2606.15378v1, 2026-06-13, Tables 1/2/13 [T] | 0.22B (217M total/104M non-emb) and 0.66B (665M/477M), ~100B tok | **params+tokens** | no FLOPs; 1:1 full:efficient | ShortAvg 38.13→38.03 (S4), 40.46→41.31 (S5); 16K RULER 25.09→35.33 (S4), 47.17→46.13 (S5) | **NOT MEASURED** | ablation table |
| **Mamba-2** (SSD), pure and hybrid | 2405.21060v1, 2024-05-31, Table 2 [T] | 350M, 48L, 7B tok Pile | **params+tokens** | no FLOPs; best hybrid 6/48 = 12.5% attention layers | Transformer++ 8.68 → Mamba-2 8.60 → 6-attn 8.26 ppl (−0.42, −4.8%) | **NOT MEASURED** | ppl table |
| **Gated DeltaNet** (GDN), pure + H1/H2 hybrids | 2412.06464v3, 2024-12-09, Table 3 [T] | 1.3B, 100B tok FineWeb-Edu | **params+tokens** | no FLOPs; H1 = GDN+SWA, H2 = Mamba2+GDN+SWA; **attention fraction not stated** (figure shows N× GDN + 1 SWA at 2K window) | Transformer++ 18.53 → GDN 16.42 → H1 16.07 → H2 15.91 ppl (−2.11 pure, −2.62 hybrid) | **NOT MEASURED** | ppl table |
| **Kimi Linear / KDA**, 3:1 KDA:MLA | 2510.26692v2, 2025-10-30, Table 1/§5.3 [T] | 48B / 3B active, 5.7T tok; scaling 653M–1.7B active | **iso-FLOPs** (Table 1); **iso-loss** (scaling) | attention FLOPs formula: KDA `6Td_h²+3TCd_h+TC²` vs full `2T²d_h`; 3:1 reduces KV 75%, decode 6.3× at 1M | Table 1: 3:1 5.65 vs 0:1 5.77 val PPL (−0.12); scaling: 1.16× compute efficiency | **0.862×** at equal loss (law); −0.12 PPL at equal FLOPs | iso-FLOPs ablation + scaling law |

**Fractions are layer fractions, not FLOP fractions.** Mamba-2's 12.5% is 6 of 48 layers; Kimi's 3:1 is 25% full-attention layers; the SWA hybrid is 1:1 (50% full layers). None of the papers prints the attention FLOP share, so these are not directly comparable to the 137.4 arithmetic. **Separate the two savings.** MLA, GQA/MQA and Kimi's KV reduction are **memory/bandwidth** wins: the KV cache shrinks, decode throughput rises, and the paper quantifies no FLOPs/token; MLA adds down/up projections, so any arithmetic change is small but unmeasured [S]. At 1–8 KB on a 100M model, the KV cache is small and decode is not the training bottleneck, so **MLA does not materially move the 137.4 number**. SWA/GDN/KDA do cut the attention-core term — at 4 KB the whole attention line is ~20% of the total, so a full SWA can save at most that — but the controlled short-context tables show a tie, not a win. Kimi's own scaling law is the honest magnitude: **1.16× compute efficiency**, i.e. 0.862× FLOPs at equal loss — not 2×.

### 2.4 Low-rank weights and training-time low-rank

| Mechanism | Paper (id, version, date) | Scale | Match | FLOPs metric | Quality delta | FLOPs/byte ratio | Evidence |
|---|---|---|---|---|---|---|---|
| **GaLore** (gradient low-rank projection) | 2403.03507v2, 2024-03-06, abstract/§9.4 [T] | LLaMA 1B/7B, 19.7B tok | full-parameter training, memory-only | optimizer memory −65.5%; throughput −17% | maintains performance | **NOT MEASURED** (forward FLOPs unchanged) | memory/throughput table |
| **PiSSA** (principal singular values LoRA init) | 2404.02948v4, 2024-04-03, abstract [A] | PEFT; no pretraining-scale control | fine-tuning | no FLOPs; faster convergence / better than LoRA | PEFT-level | **NOT MEASURED** | abstract |
| **ReLoRA** (low-rank updates + restarts/merging) | 2307.05695v4, 2023-07-11, §5 [T] | 1B/1.3B, rank 128 | full-rank baseline | 21% wall-clock speedup on 6×A6000 (33 h saved on 1B) | comparable | **NOT MEASURED** (forward FLOPs unchanged) | wall-clock table |
| **SLTrain** (sparse + low-rank parameterization) | 2406.02214v2, 2024-06-04, abstract [A] | LLaMA 7B | full-rank pretraining | memory −73% with quantization/per-layer updates | comparable to full-rank | **NOT MEASURED** (forward FLOPs not reported) | memory table |

**The class does not do what the target needs.** GaLore, PiSSA, ReLoRA and SLTrain are **training-memory** methods: they keep or reconstruct full-rank weights, so the forward `6N` per token is unchanged. SLTrain's factorized parameterization could in principle cut forward FLOPs, but the paper reports memory, not FLOPs, and no matched-quality 100M run exists. **Inference-time** weight factorization is a different class; I found no 100M-scale, BPB-matched FLOPs/byte table for it, so it is `NOT MEASURED`, not a candidate.

### 2.5 Byte-level patching

| Mechanism | Paper (id, version, date) | Scale | Match | FLOPs metric | Quality delta | FLOPs/byte ratio | Evidence |
|---|---|---|---|---|---|---|---|
| **MegaByte** (fixed stride P=8, global/local) | 2305.07185v2, 2023-05-12, Table 2/§3.1 [T] | 320M control vs 758M global/262M local, 80B bytes | **compute+data controlled** | `2T(m_g/P + m_l)`; printed config ≈713.5 vs 640 MFLOP/byte forward (1.11×) | PG-19 BPB 1.057→1.000; Code 0.575→0.411 | **1.11× at the printed config**; equal-quality ratio NOT MEASURED | controlled table |
| **BLT** (entropy/space patches) | 2412.09871v1, 2024-12-13, Tables 1/2/§5.1 [T] | 470M–8B, 1T tok / 4.5T bytes | **iso-FLOPs** at 8B | 8B: Llama3 60.0 vs BLT-Space 58.0 / BLT-Entropy 61.1 avg; bytes/patch 4.4/6.1/4.5 | +1.1 pts (entropy), −2.0 (space) at equal FLOPs | **0.72×** at equal quality (space, −2.0 pts); **~0.5×** nominal at ps=8 vs BPE 4.4, but ps=8 is worse at 1B | iso-FLOP table + scaling |
| **H-Net** (2-stage dynamic chunking) | 2507.07955v2, 2025-07-10, Table 2 [T] | 760M and 1.3B, 100B tok | **data/FLOP-matched** | Transformer 0.42/0.69 GFLOPs/byte; H-Net 2-stage 0.43/0.69 | BPB 0.756→0.743 (Large), 0.730→0.715 (XL); avg 53.3→55.5, 55.5→58.2 | **~1.0× at same FLOPs** (0.43 vs 0.42, better quality); **0.62×** at equal downstream avg 55.5 (Large 0.43 vs XL 0.69, cross-scale) | matched table |
| **Scratchpad Patching** (SP; larger patches + intra-patch compute) | 2605.09630v1, 2026-05-10, Table 1 [T] | ~2B, ~400B bytes | **fixed data; FLOPs/byte measured vs pure byte-level (1 B/patch)** | byte 54.1; p=4 52.9 (3.1×), p=8 52.2 (4.5×), p=16 48.0 (5.7×); +SP 56.2 (2.1×), 53.8 (2.6×), 54.2 (2.9×) | at equal quality 54.1, p=16+SP matches with **2.9× fewer FLOPs** | **0.345× vs byte-level** at 2B | measured FLOPs/byte + accuracy |
| **ByteFlow** (learned byte compression) | 2603.03583v1, 2026-03-03, Table 6 [T] | 600M/50B tok, 1.3B/500B tok | fixed data; compression sweep | no FLOPs/byte table | 600M: 2.0× 51.74, 2.56× 50.89, 3.41× 50.13, 5.12× 48.48; 1.3B: 64.44→60.91 at 5.12× | **NOT MEASURED** | accuracy vs compression |

**What patching does to FLOPs per byte.** The nominal saving is `patch_bytes`, but the measured saving is smaller because the local encoder/decoder runs per byte and does not shrink with the patch. Scratchpad is the cleanest measurement: at ~2B, nominal 16× at p=16 becomes **5.7×** for the same model, and the quality-matched point is **2.9×** (p=16+SP) **relative to a pure byte-level baseline (1 B/patch)** [T, 2605.09630v1 Table 1]. That matters for our baseline: 137.4 is already at ~4.5 B/patch, so 2.9× off a 1 B/patch baseline is not 137.4/2.9; the pure-byte 100M point would be ~616/2.9 ≈ 212, which is *above* 137.4 [C]. BLT's own text says patch 8 gives “nearly 50% inference FLOP savings” versus Llama 3's 4.4 bytes/token [T, 2412.09871v1 §5.1], but the same paper says ps=8 starts worse at 1B and only overtakes at 7B; at 470M the ps=8 crossover needs **150B bytes** [T, Table 2]. Our 1–22 GB is 1–22B bytes, 7–150× short of that crossover. **This is the strongest negative result in the report for byte patching at 100M.**

MegaByte is a **parameter-efficiency** win at fixed compute, not a FLOPs/byte cut at equal quality: its printed 758M/262M/P=8 model uses ~1.11× the control's FLOPs/byte and wins 0.057 BPB on PG-19 [T, Table 2]. H-Net's 2-stage is a quality win at the same FLOPs/byte (0.743 vs 0.756 BPB at 760M); the 0.62× equal-downstream ratio is a cross-scale comparison (Large H-Net vs XL Transformer) and is not a BPB match [T, Table 2]. ByteFlow reports compression vs accuracy, no FLOPs, so it cannot be placed on the ratio axis [T, Table 6].

### 2.6 The <=40 MFLOP/byte question

| Candidate | Arithmetic | Why it is not a verified answer |
|---|---|---|
| **MoE 64×3M** (192M stored / 3M active) | `6×3M = 18 MFLOP/token`; at 4 B/token ≈ **4.5 MFLOP/byte** training [C] | No matched-quality pair at 100M; the law's runs use **16–66B tokens**, 3–64× our corpus; the 3M-active model's absolute quality at 1–5B tokens is unmeasured [S, 2402.07871v1 Table 3] |
| **BLT ps=8** | ~0.5× inference FLOPs at 8B [T] | Worse than BPE at 1B; crossover 7B; 470M needs 150B bytes; our scale is below the crossover [T, 2412.09871v1 Table 2/§5.1] |
| **Scratchpad p=16+SP** | 2.9× fewer FLOPs at equal quality at 2B, **vs pure byte-level (1 B/patch)** [T] | At 100M the pure-byte point is `6N/1 ≈ 616`, /2.9 ≈ **212**, above 137.4; applying 0.345 to 137.4 is invalid because 137.4 is already ~4.5× patched. The 2B run is 400B bytes, not 1–22 GB [C/S] |
| **H-Net 2-stage** | 0.62× at equal downstream avg, 760M vs 1.3B [T] | Cross-scale, BPB not matched (0.743 vs 0.730); absolute FLOPs/byte is 0.43–0.69 GFLOP/byte, not <=40 |
| **MoD + SWA** | 0.60× × attention saving; 79.4 at 102.7M [C/S] | Not <=40; and the combination is unmeasured |
| **MoE + MoD, 280M/100M active** | 0.602 × (100/102.7) ≈ 0.586 → 80.5 [C/S] | Not <=40; capacity-recovery arm, not a verified win |

**Verdict on <=40:** I cannot answer with a verified paper. In this survey, the only configuration that is arithmetically <=40 at >=100M stored is the 192M-total/3M-active MoE, and its quality at 1–22 GB is **SPECULATION**: no direct benchmark exists at that data budget. BLT's 0.5× at 8B lands at ~69 if applied to 137.4 — below 80, not <=40; Scratchpad's 2.9× is off a pure byte-level baseline and lands at ~212 at 100M; H-Net's 0.62× lands at ~85. None of the three is a verified <=40 at our scale; each is either above 40, above 80, or a cross-scale extrapolation.

---

## 3. The one number per mechanism

`Ratio` is FLOPs/byte versus dense **at equal quality** where the paper prints an equal-quality pair; `NOT MEASURED` where it prints only tokens/s, memory, or a different match. `Δ at equal FLOPs` is the quality delta when FLOPs are held equal.

| Mechanism | Ratio at equal quality | Δ at equal FLOPs | Scale of the number | Match |
|---|---:|---|---|---|
| DeepSeekMoE 2B | **0.175×** | +0.002 Pile loss (worse) | 2.0B/0.3B, 100B tok | iso-quality |
| DeepSeekMoE 16B | **0.405×** | −0.01 BPB (better) | 16.4B/2.8B, 2T tok | iso-quality |
| DeepSeekMoE 145B | **0.285×** | −0.029 Pile loss (better) | 144.6B/22.2B, 245B tok | iso-quality |
| Fine-grained MoE law | **0.05×** at 1e20 | curve below dense | 3M–100M active, 16–66B tok | law-level |
| MoD 12.5% | **0.60×** at 6e18 | 0 (equal loss) | 60M–3B | iso-FLOPs |
| MoR 135M | **>1×** (not measured) | +0.0155 NLL (worse) | 135M/42M unique, 2e18 | iso-FLOPs |
| MoR 360M | **<1×** (not measured) | −0.2759 NLL (better) | 360M/118M unique, 2e18 | iso-FLOPs |
| CALM | **NOT MEASURED** | task-level; up to 3× speedup | 8-layer T5 | task |
| LayerSkip | **NOT MEASURED** | 1.34–2.16× decode | Llama 1B–13B | task |
| MLA | **NOT MEASURED** | memory 0.067× KV; 5.76× throughput | 236B/21B, 8.1T tok | deployment |
| SWA+global 1:1 | **NOT MEASURED** | short −0.10 (S4) / +0.85 (S5) pts; 16K +10.24 (S4) / −1.04 (S5) | 0.22B/0.66B, 100B tok | params+tokens |
| Mamba-2 12.5% attn | **NOT MEASURED** | −0.42 ppl (−4.8%) | 350M, 7B tok | params+tokens |
| GDN hybrid | **NOT MEASURED** | −2.62 ppl (−14.1%) | 1.3B, 100B tok | params+tokens |
| Kimi 3:1 KDA:MLA | **0.862×** (law) | −0.12 PPL | 48B/3B, 5.7T tok; 1.3B scaling | iso-FLOPs/law |
| GaLore | **NOT MEASURED** | memory −65.5%, −17% throughput | 1B/7B, 19.7B tok | memory |
| PiSSA / ReLoRA / SLTrain | **NOT MEASURED** | memory/wall-clock only | 1B–7B | memory/PEFT |
| MegaByte | **1.11×** at printed config | +0.057 BPB (better) | 320M control, 80B bytes | compute+data |
| BLT 8B | **0.72×** (space, −2.0 pts) | +1.1 pts (entropy) | 470M–8B, 1T tok/4.5T bytes | iso-FLOPs vs BPE 4.4 B/token |
| H-Net 2-stage | **~1.0×** (better quality) | −0.013/−0.015 BPB | 760M/1.3B, 100B tok | data/FLOP-matched |
| Scratchpad p=16+SP | **0.345× vs byte-level** | −0.1 pts (equal) | ~2B, 400B bytes | measured FLOPs/byte, 1 B/patch baseline |
| ByteFlow 5.12× | **NOT MEASURED** | −2.41 pts (600M), −3.53 (1.3B) | 600M/50B, 1.3B/500B | compression |

### 3.1 Credibility and effort

`Credibility` is about the *number in the row*, not the paper's importance: a table-read with a dense control is stronger than an abstract or a fitted law. `Effort` is what it costs to reproduce at 100M/1–22 GB in this repo.

| Candidate | Source/date | Key fact | Credibility | Effort to adopt/verify |
|---|---|---|---|---|
| DeepSeekMoE 2B/16B/145B | 2401.06066v1, 2024-01 | equal-loss **0.175–0.405×** FLOPs | industry lab, table-read, dense controls; no independent replication | **high**: routing/gather + 1–22 GB training run |
| Fine-grained MoE law | 2402.07871v1, 2024-02 | **20×** at 1e20, >40× at 1e25 | academic, fitted law; no printed 100M pair | **medium**: fit at 100M, 1–5B tokens |
| DeepSeek-V3 | 2412.19437v2, 2024-12 | 671B/37B, aux-free −0.003 BPB | industry report, table; **no dense control** | **low** for 100M (too big; only the balancing recipe transfers) |
| MoD | 2404.02258v1, 2024-04 | **0.60×** at 6e18, 60M–3B | industry (DeepMind), table; only 6e18 has the equal-loss pair | **medium**: router/top-k + 3 seeds |
| MoR | 2507.10524v3, 2025-07 | **+0.0155** at 135M, **−0.2759** at 360M | academic, clean isoFLOP table | **medium**: loop block already exists in the repo |
| MLA | 2405.04434v5, 2024-05 | KV **−93.3%**, throughput **5.76×** | industry, deployment numbers | **high**: kernel work; no FLOPs win at 8 KB |
| SWA 1:1 | 2606.15378v1, 2026-06 | short-context **tie**, 16K win | academic, table-read | **low**: window mask, no new kernel |
| Mamba-2 / GDN | 2405.21060v1 / 2412.06464v3 | **−4.8% / −14.1%** ppl | academic/industry, ppl only, no FLOPs | **high**: kernels; FLOPs/byte unmeasured |
| Kimi KDA | 2510.26692v2, 2025-10 | **1.16×** compute, 3:1 | industry, table + law; 48B proof | **high**: kernel; the 1.16× is the ceiling |
| GaLore / PiSSA / ReLoRA / SLTrain | 2403.03507v2 etc. | memory only | mixed; **no FLOPs** | **low** for this target (out of scope) |
| MegaByte / BLT | 2305.07185v2 / 2412.09871v1 | quality at fixed compute / **0.5×** at 8B | industry, table; scale mismatch | **high**: local/global + 150B bytes at 470M |
| H-Net / Scratchpad | 2507.07955v2 / 2605.09630v1 | same-FLOPs better BPB / **2.9×** at 2B | academic/industry, table; 0.76–2B | **high**: chunking + 400B bytes |

---

## 4. What is measured where

### 4.1 At <=1B params (only these)

| Mechanism | <=1B evidence | Number | Source |
|---|---|---|---|
| MoD | 60M–1B subset of the 60M–3B sweep | 0.60× FLOPs at equal loss, 6e18 | [T] 2404.02258v1 §5.1 |
| MoR 135M | 135M base / 42M unique | +0.0155 NLL at iso-FLOPs (worse) | [T] 2507.10524v3 Table 9 |
| MoR 360M | 360M base / 118M unique | −0.2759 NLL at iso-FLOPs (better) | [T] 2507.10524v3 Table 9 |
| Mamba-2 | 350M, 48L, 7B tok | 8.68→8.60 pure (tie); 8.26 at 12.5% attention | [T] 2405.21060v1 Table 2 |
| SWA+global 1:1 | 0.22B and 0.66B, 100B tok | short 38.13→38.03 / 40.46→41.31; 16K 25.09→35.33 / 47.17→46.13 | [T] 2606.15378v1 Tables 2/13 |
| H-Net | 760M, 100B tok | same ~0.42–0.43 GFLOPs/byte, BPB 0.756→0.743 | [T] 2507.07955v2 Table 2 |
| BLT | 470M and 1B | ps=8 worse at 1B; 470M ps=8 crossover 150B bytes | [T] 2412.09871v1 Table 2/§5.1 |
| MegaByte | 320M control | 1.057→1.000 BPB at 1.11× FLOPs/byte | [T] 2305.07185v2 Table 2 |
| CALM | 8-layer T5 | 5.73/8 decoder layers, 1.41× end-to-end, up to 3× | [T/A] 2207.07061v2 Table 2 |
| LayerSkip | Llama3.2 1B (and 7B/13B) | 1.34–2.16× decode; no FLOPs/byte | [A] 2404.16710v4 |
| MoE law | 192M total/3M active is the smallest | law only; 16–66B tokens | [T] 2402.07871v1 Table 3 |

### 4.2 Only >=3B (transfer is SPECULATION, not advice)

| Mechanism | Scale where the win is shown | Win | Why it does not transfer down for free |
|---|---|---|---|
| DeepSeek-V3 | 671B/37B, 14.8T tok | no dense control; 180K H800 hours/1T tok; aux-free −0.003 BPB at 15.7B | no 100M rung; the paper's own ablation is 15.7B |
| DeepSeek-V2 / MLA | 236B/21B, 8.1T tok | KV −93.3%, throughput 5.76×, training cost −42.5% | memory/bandwidth win; no FLOPs/token; at 8 KB the KV cache is small |
| DeepSeekMoE 145B | 144.6B/22.2B, 245B tok | 0.285× FLOPs at equal loss vs 67B | 100M row unmeasured; 145B has 22B active, ~220× our active budget |
| Kimi Linear | 48B/3B, 5.7T tok | 3:1 KDA:MLA, 1.16× compute efficiency, 6.3× decode at 1M, KV −75% | the 1.16× is a scaling-law result at 653M–1.7B active; at 4 KB the attention core is a small fraction |
| BLT | 8B, 1T tok / 4.5T bytes | 0.5× inference FLOPs at ps=8; +1.1 pts at iso-FLOPs | worse at 1B; crossover 7B; 470M needs 150B bytes |

**What this means:** the literature's positive results cluster at 2B+ for MoE and 7B+ for byte patching. The <=1B positive results are MoD (60M–3B), Mamba-2 hybrid (350M, −4.8% ppl at 12.5% attention; pure Mamba-2 is a 0.9% tie), H-Net (760M, a same-FLOPs quality win), and SWA at 16K (0.22B only; 0.66B is slightly worse). None of them is a 100M/1–22 GB iso-quality FLOPs win.

---

## 5. Recommendation: three arms to 80 MFLOP/byte

**Common protocol.** 1.5 GB bytes, 3 seeds, 4 KB and 8 KB, same eval window (`eval_batches × batch × seq_len`, printed), same data order, pure CE, 2k steps minimum. The gate is **ΔBPB <= +0.02** at both contexts. The resolution warning is real: the repo's 3-seed spread at 9.2M was 0.048–0.073 BPB [ours, AGENTS §3.1]; at 100M it may halve, but 0.02 is at or below the noise floor. A 2-seed result cannot decide this. All MFLOP/byte below use the `6N` accounting of the 137.4 baseline; attention-core is a separate line.

| Arm | Config | Params total / active | MFLOP/byte | Why it might hold quality | MFU impact (launch count, TC shapes) | Falsifying A/B | Confidence |
|---|---|---|---:|---|---|---|---|
| **C0** | current dense, 4.485 B/patch (12L assumed only for the attention diagnostic) | 102.7M / 102.7M | **137.4** | — | 17.9% measured | — | — |
| **C1** | **MoD 12.5% + patch 4.67 + SWA 1:1** (4.67 = 4.485/0.960, the smallest patch increase that clears 80 after 0.602) | 102.7M / 102.7M | **79.4** (`0.960×0.602`) | MoD equal loss at 60M–3B, 6e18 [T]; SWA short-context tie at 0.22B/0.66B [T]; patch +4% is inside the published 4.5–6.1 B/patch band [T] | MoD adds 1 router/top-k per block and shrinks token dim to 12.5% (M≈256 at 2K ctx); SWA makes attention L=128; TC-valid but low occupancy; **MFU 12–16% expected** [S] | 3 seeds, 1.5 GB, 4 KB and 8 KB; kill if ΔBPB > +0.02 at either; kill if step time > 0.85× control (FLOPs not translating) | **medium-low** |
| **C2** | **fine-grained MoE, 1 shared + 7 routed, top-1, patch 4.485** | 170M / 60M | **80.3** (`60/102.7`) | DeepSeekMoE 2B equal loss at 0.175× FLOPs; 16B at 0.405× [T]; the law says MoE is better at every budget, but its smallest run is 192M/3M with 16–66B tokens [T] | +3–4 launches/layer (router, top-k, gather, scatter); 8 expert GEMMs M≈256, N≈1536, K=768; **MFU 8–12% expected** [S] | 3 seeds, 1.5 GB; kill if ΔBPB > +0.02 at 4 KB or 8 KB; kill if step time >= control (launch-bound) | **low** |
| **C3** | **MoE + MoD capacity-recovery**: 1 shared + 7 routed, top-1, MoD 12.5%, patch 4.485 | 280M / 100M | **80.5** (`0.602×100/102.7`) | Same MoE + MoD evidence, but active width stays at 100M (≈control) while stored params are 2.7×; MoD pays the 0.602×, so the FLOPs target is met without narrowing the active stream [S] | Same expert gathers as C2 plus MoD's 12.5% token dim; **MFU 8–12% expected** [S] | 3 seeds, 1.5 GB; kill if ΔBPB > +0.02 at 4 KB or 8 KB; if it passes, it is the capacity-recovery proof; if it loses, MoE+MoD at 100M is dead | **low** |

**Reading the arm numbers.** C1's 79.4 is the `6N` term after MoD and the +4% patch; the SWA 1:1 is an **additional** attention-core saving not counted in 79.4 (at 8 KB the whole attention line is ~67 MFLOP/byte, so half of it is ~34). C2 and C3 keep full attention: their 80.3/80.5 are `6N` only, and at 8 KB their real total is higher by the attention line. If your profiler's 137.4 already includes attention, **C2/C3 do not reach 80 at 8 KB**; C1's SWA is the only arm that attacks that line, and its `6N` number is already 79.4. This is the single largest uncertainty in the table and the reason the A/B must print the FLOPs split, not just the total.

**Why these three and not byte patching.** C1 is the only arm built from mechanisms with **<=1B** iso-FLOPs evidence (MoD) and **<=0.66B** short-context evidence (SWA). C2 is the strongest **equal-quality FLOPs** evidence among the surveyed mechanisms, but at 2B+, so it is a bet on transfer. C3 is the only arm that can plausibly be **equal or better** quality at 80 because it keeps the active width at ~100M and buys 2.7× stored capacity, with MoD paying the FLOPs bill; it is also the most expensive and the most likely to be launch-bound. A byte-patch arm (patch 8 + local model) is **not** recommended: BLT is worse at 1B, the 470M ps=8 crossover needs 150B bytes, and a 10M local model running per byte would add ~60 MFLOP/byte and erase the nominal 0.56× [T, 2412.09871v1; C].

**MFU: the honest forecast.** The closest measured point is launch-bound: at 9.2M, 13.3% mean utilization, 79% of samples <=5%, GEMMs 2–5% of a warm step [ours, AGENTS §2.2/§3.1]; the 100M point is not measured. Every arm above **reduces** the arithmetic per launch or **adds** launches: MoD shrinks the token dimension to 12.5%, MoE adds router/gather/scatter, patching halves the global M. None of that raises MFU; the expected direction is 8–16%, not 35%. The 35% target is a **systems** target: CUDA-graph capture of the largest launch-bound stage (confirmed working on this GPU, not built), fused optimizer/retention, and larger effective GEMMs. Architecture choice can stop making it worse; it cannot make it 35% by itself [S, AGENTS §3.1].

**What would falsify the recommendation as a whole.** Run C0 and C1 first. If C1 loses more than 0.02 BPB at either 4 KB or 8 KB, the MoD+SWA lane is dead at 100M, and the only remaining architecture bet is C2/C3 MoE. If C1 passes but C2 and C3 lose, the conclusion is that 100M/1–22 GB is below the MoE crossover, and the 137.4 baseline is already on the efficient frontier for this data budget; the remaining lever is systems MFU, not architecture. If all three pass, the next step is a 5B-byte confirm at 3 seeds, not a bigger model.

---

## 6. Open questions

1. **I found no 100M-scale iso-quality FLOPs/byte table** for MoE, MoD, or byte patching. Every ratio here is either at >=2B or is a law extrapolation. This is the central gap.
2. **The data budget is the binding constraint for MoE.** The fine-grained law's smallest run is 192M total/3M active with 16–66B tokens; our 1–22 GB is 0.25–5.5B tokens. A 100M MoE at 1–5B tokens is undertrained by the paper's own standard.
3. **Attention accounting is unresolved.** The 137.4 equals `6N/patch` exactly; if the profiler includes attention, the real total at 8 KB is ~205 MFLOP/byte by arithmetic, and the 80 target is much harder. Measure the split before choosing C1.
4. **BPB resolution.** A 0.02 BPB gate may be below the 3-seed noise at 100M. The repo's 9.2M spread was 0.048–0.073; if 100M does not halve it, the gate cannot be measured with two seeds, and "equal quality" needs a paired eval with more bytes.
5. **MFU is not a model property.** No paper reports MFU for these mechanisms at 100M; the 35% target needs a measured launch/GEMM breakdown per arm, not an architecture claim.
6. **Low-rank weights are a memory story.** No paper reports forward FLOPs/byte at matched quality for GaLore/PiSSA/ReLoRA/SLTrain; if the target is FLOPs, this class is out of scope until someone prints that table.

---

## 7. Sources (all arXiv, version and first-submission date)

| ID | Title | Used for |
|---|---|---|
| 2401.06066v1 (2024-01-11) | DeepSeekMoE: Towards Ultimate Expert Specialization in MoE LMs | 2B/16B/145B equal-quality FLOPs, Tables 2/3/4/6 |
| 2402.07871v1 (2024-02-12) | Scaling Laws for Fine-Grained Mixture of Experts | 20×/40× law, 64×3M–64×100M, Table 2/3 |
| 2412.19437v2 (v1 2024-12-27) | DeepSeek-V3 Technical Report | 671B/37B, 1+256 top-8, no token drop, aux-free Table 5 |
| 2404.02258v1 (2024-04-02) | Mixture-of-Depths | 0.60× at 6e18, 60M–3B, 12.5% capacity |
| 2507.10524v3 (2025-07-14) | Mixture-of-Recursions | 135M +0.0155, 360M −0.2759, Table 9 |
| 2405.21060v1 (2024-05-31) | Transformers are SSMs (Mamba-2) | 350M, 8.68→8.60 pure, 8.26 at 12.5% attention |
| 2412.06464v3 (2024-12-09) | Gated Delta Networks | 1.3B/100B, 18.53→16.42/15.91, Table 3 |
| 2510.26692v2 (2025-10-30) | Kimi Linear | 3:1 KDA:MLA, 1.16× compute, Table 1, 48B/3B |
| 2405.04434v5 (2024-05-07) | DeepSeek-V2 | MLA KV −93.3%, 5.76× throughput, 236B/21B |
| 2606.15378v1 (2026-06-13) | Rethinking the Role of Efficient Attention in Hybrid Architectures | SWA 1:1, 0.22B/0.66B, short/16K tables |
| 2403.03507v2 (2024-03-06) | GaLore | −65.5% optimizer memory, −17% throughput |
| 2404.02948v4 (2024-04-03) | PiSSA | PEFT, no FLOPs |
| 2307.05695v4 (2023-07-11) | ReLoRA | rank 128, 21% wall-clock speedup (33 h saved on 1B) |
| 2406.02214v2 (2024-06-04) | SLTrain | −73% memory at LLaMA 7B |
| 2404.16710v4 (2024-04-25) | LayerSkip | 1.34–2.16× decode speedup |
| 2207.07061v2 (2022-07-14) | Confident Adaptive Language Modeling | 5.73/8 layers, 1.41×, up to 3× |
| 2305.07185v2 (2023-05-12) | MegaByte | 320M control, 1.057→1.000 BPB, FLOPs formula |
| 2412.09871v1 (2024-12-13) | Byte Latent Transformer | 8B iso-FLOPs, ps=8 worse at 1B, 150B-byte crossover |
| 2507.07955v2 (2025-07-10) | Dynamic Chunking / H-Net | 760M/1.3B, same GFLOPs/byte, BPB −0.013/−0.015 |
| 2605.09630v1 (2026-05-10) | Scratchpad Patching | 2B, p=16+SP 2.9× at equal quality |
| 2603.03583v1 (2026-03-03) | ByteFlow | compression 2.0–5.12×, accuracy deltas |
| 2409.02060v2 (2024-09-03) | OLMoE | 6.9B/1.3B, 5T tokens |

## 8. Evidence separation

### VERIFIED (sourced)

- Every number in the mechanism tables (§2), the summary table (§3), and the <=1B / >=3B tables (§4) carries an arXiv ID, version, date, and table/section.
- The arithmetic identity `137.4 = 6·102.7M / 4.485` and the attention-core table in §1.1 are marked [C]; they are arithmetic on the stated baseline, not paper measurements.
- The direction of each mechanism **at its own scale** is verified: MoE wins at 2B+, MoD wins at 6e18 over 60M–3B, MoR loses at 135M and wins at 360M, SWA is a short-context tie and a 16K win, BLT is worse at 1B, MLA is a KV-memory win, low-rank training is a memory win.
- The negative result "I found no <=40 MFLOP/byte configuration at 100M/1–22 GB with matched quality" is a **search result**, not a proof of impossibility.

### SPECULATION (my inference)

- The transfer of any paper ratio to 100M/1–22 GB; every arm MFLOP/byte assumes the paper ratios multiply and the `6N` accounting holds.
- The MFU forecasts (8–16%); the measured 13.3% utilization is at 9.2M, not 100M.
- BLT's 0.5× applied to 137.4 (≈69) and H-Net's 0.62× applied to 137.4 (≈85) are arithmetic extrapolations, not 100M results.
- The capacity-recovery argument for C3: it is the only arm that keeps the active width at ~100M while buying 2.7× stored capacity, but no paper tests that combination at 100M.
- The claim that the 35% MFU target is systems-bound is an inference from the repo's launch-bound measurements, not a measured MFU result for these arms.

**Confidence:** high in the verified table (numbers read from tables); low in the 100M transfer, as stated per arm. The confidence rating applies only to the verified part; the arm confidences are forecasts.
