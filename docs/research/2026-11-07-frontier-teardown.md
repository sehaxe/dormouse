# Frontier teardown: 8 open architectures, what each component buys, and what survives at 95M

**Date:** 2026-11-07. **Mode:** read-only research. No config, code, checkpoint or GPU was touched.
**Question:** tear down the strongest open models as of 2026-11 and answer, with arithmetic, which of their components can exist at 95M parameters / 1 KB context / 22 GB corpus / one 16 GB card.

**Tags:** **[T]** — the number is read from a table, figure or section of the cited source · **[C]** — computed here from published or our own numbers (formula shown) · **[S]** — speculation, no direct measurement · **[ours]** — measured in this repository.

**Citation rule.** Each model section states the report id, version and date once; every row then cites `§`/`Table`. Sources outside the section cite `id vX (date) §/Table`. All ids, versions and dates are repeated in §7. Where a component is not published, the table says **not published** — it is not inferred.

---

## TL;DR (RU)

1. **Из восьми имён в брифе реально существуют семь.** `DeepSeek-V4.1-Flash` — ровно arXiv:2609.19969v1 (2026-09-17). `Qwen3.8` — это открымый `Qwen3.8-Flash-Next` 125B-A6B (2608.30320v1) плюс `Qwen3.8-Omni`; `Qwen3.8-Max` встречается только колонкой в чужой таблице и публичного отчёта/весов не имеет. `OLMoE 2` **не существует**: после OLMoE-1B-7B (2024) AllenAI выпустила плотный Olmo 3 и гибрид Olmo Hybrid, но не второе поколение MoE. `GLM-5.3` — это пост-тренинг на базе 5.2, а `GLM-5.3-Flash` — новая гибридная база 321B/18B; это разные архитектуры под одним именем.
2. **Сошлись пять вещей, и все они про ёмкость, а не про внимание.** MoE — 9 из 11 чекпойнтов; мелкие эксперты (E≥64) — 9 из 9 MoE; shared expert — 7 из 9; top-k≥2 — 8 из 9; словарь ≥128K — 9 из 11.
3. **По вниманию фронтир расколот, и это не «конвергенция», а живая ставка.** 5 из 11 — линейные гибриды (KDA/GDN), 4 из 11 — разрежённое внимание (DSA/CSA2/QSA), 3 из 11 — только полное внимание. Опубликованные абляции противоречат друг другу: GLM на 9B видит у GDN **−8.59 RULER@64K**, Kimi на 3B видит у KDA **+3.0 RULER@128K**, MiniMax на 229.9B видит у SWA **−4.5 SWE-verified**. Победитель зависит от режима, а не от «правильной архитектуры».
4. **Удаления — тоже данные.** OLMoE → Olmo 3 убрал MoE на 7B; MiniMax убрал линейное внимание из Text-01; DeepSeek убрал HCA из V4.1; Qwen убрал SWA в пользу GDN; Kimi убрал полный MLA из 75% слоёв; Llama 4 Maverick убрал QK-norm, который есть у Scout.
5. **При 95M / 1KB / 16GB оси MEMORY и FLOPs мертвы арифметически.** KV — 39.0 MiB на последовательность, 312 MiB на batch 8 = **1.9% карты**; внимание — **17.7%** счёта; переход 3:1 на линейное внимание экономит **13.3% FLOPs**, что при MFU 2% даёт **≈0 секунд wall-clock**. Разрежённое внимание при s=1024 **дороже плотного на 5.7%**: индексер стоит 100.7 MFLOP/байт, экономия top-512 — 61.3.
6. **Ось CAPACITY при 22GB, наоборот, жива — и это исправляет наш прежний вывод, сделанный для 1GB.** 22e9 байт-токенов дают **8.25e9 токенов на эксперта** при E=64, top-2 (2.06e9 при E=256) — это не беднее опубликованных прогонов. 4× total params (380M) занимают **4.56 GB** весов+градиентов+Adam = 28.6% карты. Связывающим становится не данные, а launch-bound шаг и VRAM под активации.
7. **Единственный опубликованный положительный результат ≤100M — линейное внимание.** Olmo Hybrid, Table 5: BPB **1.077** (GDN 3:1) против **1.155** (transformer) при 60M; линейная интерполяция на 95M даёт **−0.054 BPB**. MoE измерен на **1.3B active** (OLMoE, 3× меньше токенов/FLOPs); head-to-head на 95M нет ни у кого.
8. **Вердикт для 95M:** плотная база + GDN на всех слоях (при 1K глобальное внимание не окупается), никакого разрежённого внимания, никакого словаря 128K+, MoE — единственная ставка, стоящая A/B на ~31 GPU-ч. Sparse attention, FP4 KV, CED, DSpark и Engram при наших осях — ноль по арифметике, а не по вкусу.

---

## 0. Verification of the brief's names

| Name in brief | What actually exists (2026-11) | Primary artifact | Version / date | Status |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash, "arXiv 2609.19969 series" | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | arXiv:2609.19969 | v1, 2026-09-17 | **verified, exact id** |
| Qwen3.8 | Qwen3.8-Flash-Next 125B-A6B (+ Qwen3.8-Omni, Qwen3.8-27B dense) | arXiv:2608.30320 (architecture/ablations); arXiv:2609.25611 (Omni) | v1, 2026-08-31; v1, 2026-09-22 | **verified; the open flagship is Flash-Next, not a dense "Qwen3.8"** |
| Qwen3.8-Max | only a benchmark column in the GLM-5.3 model card; no report, no weights | — | — | **not documented publicly** |
| Kimi K3 | Kimi K3: Open Frontier Intelligence | arXiv:2607.24653 | v2, 2026-07-27 (v2 2026-08-07) | **verified** |
| Kimi Linear | Kimi Linear: An Expressive, Efficient Attention Architecture | arXiv:2510.26692 | v2, 2025-10-30 | **verified** |
| GLM-5.3 | GLM-5.3 model card + config; "same base model as GLM-5.2 — every gain comes from post-training" | HF `zai-org/GLM-5.3`; arXiv:2602.15763 (GLM-5 report) | card 2026-08-25; report v2 2026-02-17 | **verified, but no standalone 5.3 report** |
| GLM-5.3-Flash | 320B total / 18B active, new hybrid base (KDA + DSA + mHC) | HF `zai-org/GLM-5.3-Flash` + card | card 2026-08-25 | **verified; benchmark published as image only** |
| Llama 4 | Scout / Maverick / Behemoth (preview) | Meta blog + `meta-llama/llama4/MODEL_CARD.md` | blog 2025-04-05 | **verified; no arXiv report** |
| Gemma 3 | Gemma 3 Technical Report | arXiv:2503.19786 | v1, 2025-03-25 | **verified** |
| OLMoE 2 | **does not exist**; successors are OLMoE-1B-7B (2024), Olmo 3 (dense, 2025), Olmo Hybrid (2026) | arXiv:2409.02060; 2512.13961; 2604.03444 | see §7 | **no such release** |
| MiniMax-M2 | The MiniMax-M2 Series | arXiv:2605.26494 | v2, 2026-05-26 (v2 2026-07-30) | **verified** |

---

## 1. One table per model

### 1.1 DeepSeek-V4.1-Flash

**Identity.** arXiv:2609.19969v1 (2026-09-17). 552B backbone + 196B Engram; 8B active on prefill, 16B on decode; 40 layers = 20 causal encoder + 20 decoder; d=5120; 64 query heads × 512; 1 shared + 384 routed experts, top-6, expert d_ff 2304; 129,280 vocab (HF config, **not published in the report**); 45T tokens; 1M context. HF checkpoint 763.2B total [T HF safetensors].

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| CED (Causal Encoder-Decoder) | decoder global KV is projected from the final encoder hidden states; upper half reuses lower-half KV | FLOPs / LATENCY | **8B active prefill vs 16B decode**; 4K→1M context raises decode FLOPs by only **1/4** [T §1, Fig. 2] | 552B backbone, 1M ctx |
| CSA2 (Compressed Sparse Attention 2) | cross-layer reuse of main KV + indexer K + top-k indices; three modes Full/Reindex/Reuse | MEMORY | **890 bytes/token** global KV = **1/4** of V4-Flash; 437× less than V1 [T abstract, Fig. 1b] | 552B, 1M ctx |
| FP4 main KV cache | 4-bit KV during training, marginal degradation | MEMORY | included in the 890 B/token figure [T §2.4.4] | same |
| SWA Bounded Replay | replay only the last `n_win` tokens instead of `L×n_win` to reconstruct SWA KV | MEMORY | persistent KV = **1/8** of V4-Flash [T abstract, §3.2.2] | deployment |
| DeepSeekMoE 1+384, top-6 | fine-grained routed experts + one shared | CAPACITY | 552B total / 16B active ⇒ **sparsity 34.5** [C 552/16]; expert d_ff 2304 [T §4.2.1] | 45T tokens |
| Engram | conditional n-gram memory, 2 modules at layers 1 and 14; orders {2,3,4}, 8 hash heads, dim 2048/order, 16M vocab | CAPACITY | **196B parameters outside the backbone** [T §2.1, §2.4.2] | 552B backbone |
| Single-Pass mHC | manifold-constrained hyper-connections, expansion 4, 20 Sinkhorn iterations | STABILITY | **−50% activation traffic** vs the original 4-kernel implementation [T §2.4.1] | Mega-mHC kernel |
| DSpark | semi-autoregressive speculative draft, 128 experts top-3, block 5 | LATENCY | not quantified in the read sections (**not published**) [T §2.4.3] | deployment |
| Muon + Sinkhorn-balanced embeddings | optimizer; Sinkhorn on embedding/head updates | STABILITY | 20 Sinkhorn iterations [T §4.2.2] | 45T run |
| MTP, 3 layers | multi-token prediction | CAPACITY | config `num_nextn_predict_layers=3` [T HF config]; no ablation in the report | 45T run |

**Published benchmarks.**

| Benchmark (metric, shots) | V4-Flash-Base | V4-Pro-Base | **V4.1-Flash-Base** |
|---|---:|---:|---:|
| MMLU-Pro (EM, 5) | 68.3 | 73.5 | **74.1** |
| HumanEval (Pass@1, 0) | 69.5 | 76.8 | **79.4** |
| GSM8K (EM, 8) | 90.8 | 92.6 | **93.0** |
| MATH (EM, 4) | 57.4 | 64.5 | **61.1** |
| LongBench-V2 (EM, 1) | 44.7 | 51.5 | **45.2** |
| MMMU-Pro (EM, 4) | — | — | **56.5** |

*Source: 2609.19969v1 §4.3.2 Table 1.*

| Post-trained (Max effort) | V4.1-Flash | Best closed in table |
|---|---:|---:|
| GPQA Diamond | **90.9** | 94.1 (GPT-5.6 Sol) |
| HLE (text-only †) | **36.8 (39.1 †)** | 56.3 (Opus-5) |
| Terminal-Bench 2.1 | **90.6** | 89.1 (Opus-5) |
| DeepSWE v1.1 | **74.2** | 74.0 (Opus-5) |

*Source: 2609.19969v1 §5.3.2 Table 3.*

**Verdict (RU).** Почти вся работа этой модели — про стоимость обслуживания 1M контекста: 890 Б/токен, FP4 KV, bounded replay, CED. Из качества-релевантного при 95M переносимы только fine-grained MoE, Engram и mHC; остальное — ноль при 1K.

---

### 1.2 Kimi K3

**Identity.** arXiv:2607.24653v2 (2026-07-27, v2 2026-08-07). 2.78T total / 104.2B active; 93 layers; d=7168; 96 heads; latent MoE width 3584 (0.5×d); 896 routed experts, 16 active, 2 shared, expert d_ff 3072; 69 KDA + 24 Gated MLA (3:1 + final MLA); 160K vocab (config 163,840); 1M context; **training token count not published**. HF checkpoint 2.779T [T HF safetensors].

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| KDA : Gated MLA = 3:1 | 69 Kimi Delta Attention layers + 24 global MLA layers; NoPE on MLA | FLOPs / MEMORY | 3:1 ratio; 1M context [T Table 1, §2.1] | 2.78T |
| AttnRes (Attention Residuals) | block-wise attention over embedding + preceding block outputs; 8 blocks of 12 layers | STABILITY / CAPACITY | memory/comm **O(Nd) instead of O(Ld)**; N≈8 recovers most benefit [T §2.2] | 2.78T |
| Stable LatentMoE | routed experts operate in ℓ=3584 (0.5×d), 896 experts top-16, 2 full-width shared | CAPACITY | **sparsity 56**; 896 experts [T §2.3, Table 1] | 2.78T |
| RMSNorm before up-projection | normalizes aggregated routed representation | STABILITY | improves val loss and downstream (no single number) [T §2.3.1] | 2.78T |
| SiTU-GLU | bounded GLU: β1·tanh(x/β1)·σ(x) · β2·tanh(x/β2) | STABILITY | **|f(x)| ≤ β1β2 = 100** with β1=4, β2=25 [T §2.3.2, Fig. 4] | 2.78T |
| Quantile Balancing (QB) | bias update from a single forward pass at the (1−k/n)-quantile of routing margins | STABILITY | balances **~10³ experts**, beyond aux-loss-free regime [T §2.3.3] | 2.78T |
| NoPE on global MLA | no positional encoding in the 24 global layers | LATENCY / length | 1M ctx; KDA carries position [T §2.1] | 2.78T |
| 2.5× scaling efficiency | aggregate architecture+data recipe gain vs K2 | overall | **~2.5×** over Kimi K2 [T abstract, Fig. 7] | scaling-law study |

**Published benchmarks.**

| Benchmark | Kimi K3 (max) | Claude Fable 5 | GPT-5.6 Sol | GLM-5.2 |
|---|---:|---:|---:|---:|
| GPQA Diamond | **93.5** | 92.6 | 94.1 | 91.2 |
| HLE-Full (no tools / tools) | **43.5 / 56.0** | 53.3 / 63.0 | 44.5 / 58.0 | — |
| DeepSWE | **67.5** | 70.0 | 73.0 | 46.2 |
| Terminal-Bench 2.1 | **88.3** | 88.0 | 88.8 | 82.7 |
| FrontierSWE | **81.2** | 86.6 | 71.3 | 67.3 |
| BrowseComp | **91.2** | 88.0 | 90.4 | — |
| GDPval-AA v2 (Elo) | **1686** | 1747 | 1736 | 1510 |

*Source: 2607.24653v2 §5 Table 2. Note: V4.1's Table 3 reports K3 GPQA 92.9 — a 0.6 cross-report difference; use each report's own table.*

**Verdict (RU).** K3 — это «ёмкость любой ценой»: 896 экспертов, latent-ширина 0.5d, 2 shared, bounded activation и QB, чтобы это не развалилось. Ни один из этих механизмов не измерен ниже 2.78T; перенос на 95M — только гипотеза.

---

### 1.3 Kimi Linear

**Identity.** arXiv:2510.26692v2 (2025-10-30). 48B total / 3B active; 27 layers; d=2304; 32 heads; 256 experts top-8 + 1 shared, expert d_ff 1024; 20 KDA + 7 MLA; vocab 163,840; 1.4T tokens (5.7T extended); 1M context.

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| KDA : MLA = 3:1 | 75% linear layers, 25% full MLA | FLOPs / MEMORY | val PPL **5.65 (3:1)** vs **5.77 (0:1 full)** at equal FLOPs [T §5.2 Table 1] | **653M active / 38.8B tokens** |
| KV cache reduction | linear state replaces KV in 75% of layers | MEMORY | **−75% KV** at long generation [T abstract, §1] | 48B / 1M ctx |
| Decode throughput | fixed-size state vs growing KV | LATENCY | **6.3×** TPOT at 1M: 1.84 ms vs 11.48 ms [T Fig. 1b] | 48B / 1M ctx |
| NoPE on MLA layers | position carried by KDA | length | RULER 128K **84.3** vs MLA **81.3**; MMLU-Pro 51.0 vs 47.2 [T Fig. 1a] | 48B / 1.4T |
| Fine-grained MoE 256 top-8 + 1 | sparse FFN | CAPACITY | 48B total / 3B active [T abstract] | 1.4T tokens |

**Published benchmarks.**

| Benchmark | Kimi Linear | Full MLA baseline |
|---|---:|---:|
| MMLU-Pro (4K) | **51.0** | 47.2 |
| RULER (128K) | **84.3** | 81.3 |
| TPOT at 1M (ms) | **1.84** | 11.48 |

*Source: 2510.26692v2 Fig. 1a/1b; Table 3 for the full comparison.*

**Verdict (RU).** Самый чистый опубликованный аргумент за линейное внимание: равный FLOPs-бюджет, 653M active, 38.8B токенов, 3:1 лучше полного MLA и по PPL, и по RULER. Это же — самый близкий к нам масштаб, где механизм ещё выигрывает.

---

### 1.4 GLM-5.3 (text)

**Identity.** No standalone 5.3 report. HF card `zai-org/GLM-5.3` (2026-08-25): "uses the same base model as GLM-5.2 — every gain comes from post-training." Architecture documented in arXiv:2602.15763v2 (2026-02-17) for GLM-5 and in the config. 753.3B checkpoint total [T HF safetensors]; 40B active documented for the GLM-5 base [T 2602.15763v2 §2.1], ≈37.7B rough recomputation from the 5.3 config with estimated attention terms [C]; 78 layers; d=6144; 256 routed experts top-8 + 1 shared, moe_int 2048; vocab 154,880; 1M context; **GLM-5.2/5.3 base token count not published** (GLM-5's figure was 28.5T).

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| DSA (DeepSeek Sparse Attention) | lightning indexer selects top-k tokens; top-k 2048, 32 index heads × 128 | MEMORY / FLOPs | **1.5–2×** less attention compute at 128K, "half the GPU cost" [T 2602.15763v2 §2.1.1] | GLM-5, 744B |
| IndexCache / IndexShare | one indexer shared by every 4 sparse layers | MEMORY | indexer runs **1/4** as often; config pattern full/shared×3 [T HF config; 2603.12201v1] | GLM-5.2/5.3, 1M |
| MLA-256 | 256-dim qk (192 NoPE + 64 RoPE), kv_lora 512, q_lora 2048 | MEMORY | MLA-256 matches GQA-8 while cutting decode attention FLOPs by ~1/3 [T 2602.15763v2 §2.1, Table 1] | 9B-class ablation (MMLU ~61–62) |
| MoE 256 top-8 + 1 shared | fine-grained FFN | CAPACITY | 744B total / 40B active in GLM-5; sparsity 18.6 [C 744/40] | 28.5T tokens |
| MTP with shared 3 heads | one MTP layer predicting 2 tokens, sharing 3 layers' params | LATENCY | higher accept length at constant draft memory [T 2602.15763v2 §2.1, Table 2] | GLM-5 |
| Muon split | separate Muon preconditioner per head group | STABILITY | MMLU **62.5** vs 61.5 (MLA), 61.2 (GQA-8) [T Table 1] | 9B-class |

**Published benchmarks (GLM-5.3 card).**

| Benchmark | GLM-5.3 | GLM-5.2 | Kimi K3 | Opus 4.8 | GPT-5.6 Sol |
|---|---:|---:|---:|---:|---:|
| Terminal Bench 2.1 | 88.2 | 81.0 | 88.3 | 85.0 | **88.8** |
| Terminal Bench 3.0 | 28.3 | 4.6 | 17.4 | 21.1 | **34.6** |
| DeepSWE v1.1 | 66.9 | 46.2 | 67.5 | 58.0 | **72.7** |
| CyberGym | **84.5** | 77.2 | 80.0 | 78.1 | 83.6 |
| HLE w/ tools | 62.5 | 54.7 | 59.8 | 57.9 | **64.5** |
| GDPval-AA v2 | **1769** | 1508 | 1682 | 1588 | 1730 |

*Source: HF `zai-org/GLM-5.3` model card table (2026-08-25).*

**Verdict (RU).** Это не новая архитектура, а пост-тренинг на уже существующей. Для teardown важно: заявленный прирост 5.2→5.3 — целиком RL/SFT, а не компонент. Архитектурная часть — DSA + MLA-256 + Muon split, унаследованная от GLM-5.

---

### 1.5 GLM-5.3-Flash

**Identity.** HF card `zai-org/GLM-5.3-Flash` (2026-08-25). 320B total (checkpoint 321.3B) / 18B active; 45 layers; d=4096; 288 routed experts top-8 + 1 shared, moe_int 2048; 34 KDA + 11 DSA layers (3:1); mHC expansion 4, 20 Sinkhorn iterations; MLA NoPE, kv_lora 512, q_lora 1536; vocab 154,880; 30T multimodal tokens; 1M context.

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| KDA 34 : DSA 11 | first GLM hybrid of linear and sparse attention | FLOPs / MEMORY | 3:1; "sharply reducing long-context serving costs" [T card] | 321B / 30T |
| DSA every 4th layer | global sparse attention | MEMORY | index_topk 2048, 32×128 indexer [T HF config] | 321B / 1M |
| mHC | manifold-constrained hyper-connections, 4 branches | STABILITY | expansion 4, 20 iterations [T HF config] | 321B / 30T |
| MoE 288 top-8 + 1 | fine-grained FFN | CAPACITY | 321B total / 18B active ⇒ **sparsity 17.9** [C 321/18] | 30T tokens |
| 30T multimodal corpus | native multimodal pretraining | CAPACITY | 30T tokens [T card] | 321B |
| NoPE MLA | no positional encoding in global layers | length | 1M ctx [T HF config] | 321B |

**Published benchmarks.** The card publishes a **chart only** (`bench_53.png`); no numeric table was extracted. The text claims "outperforms GLM-5.2 across benchmarks … approaching Claude Opus 4.8 on coding and agentic". **Not tabulated — do not cite numbers.**

**Verdict (RU).** Самая «конвергентная» модель набора: KDA + DSA + mHC + MoE в одном чекпойнте. Но её бенчмарк-таблица не опубликована числами, поэтому как доказательство качества она слабее, чем GLM-5.3-text.

---

### 1.6 Qwen3.8-Flash-Next

**Identity.** arXiv:2608.30320v1 (2026-08-31). 125B backbone + 51B n-gram embedding tables off-accelerator (180B checkpoint total [T HF safetensors]); 6B active; 48 layers; d=2560; 24 heads × 256; 512 experts top-10, moe_int 640, shared expert 640; 36 GDN + 12 full-attention layers (3:1), QSA on the full layers at CPT; Gated Residual 4 branches; vocab 248,320; 256K context in the config, 1M in Omni; **absolute training tokens not published** ("1/3 the predecessor's").

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| GDN 36 : full 12 | gated delta-rule linear layers, one full layer per 4 | FLOPs / MEMORY | avg over 9 benchmarks: full **49.87**, SWA hybrid **51.15**, GDN **53.81** (+3.94) [T §2.1 Table 1] | **25B-A3B, 480B tokens** |
| QSA (Qwen Sparse Attention) | micro-block indexer, 4 index heads × 128, budget 2048, compress ratio 4 | MEMORY | **7.6×** prefill and **4.9×** decode vs dense at 1M, kernel level [T §1] | 25B / 125B, 256K CPT |
| Gated Residual (GR) | 4 residual branches, elementwise gate, low-rank 320 | STABILITY | **−0.026 loss** at 276B tokens vs pre-GR [T §4, Fig. 13] | 25B-A3B, 276B tokens |
| n-gram embedding tables | single layer, n=3, 20M base vocab, 51B params on host | CAPACITY | 51B params outside the accelerator; vocabulary ↑ lowers loss monotonically, downstream saturates [T abstract, §2.3] | 125B |
| MoE 512 top-10 + 1 shared | fine-grained FFN | CAPACITY | 125B / 6B active ⇒ **sparsity 20.8** [C 125/6] | 1/3 predecessor tokens |
| 1/9 training FLOPs | aggregate efficiency | overall | leads 397B-A17B on **8 of 14** benchmarks at 1/3 active params, 1/3 tokens, ~1/9 FLOPs [T abstract] | 125B |

**Published benchmarks.**

| Benchmark | Qwen3.8-Flash-Next-Base (125B/6B) | Qwen3.8-27B-Base (27B dense) | Qwen3.7-Plus-Base (397B/17B) |
|---|---:|---:|---:|
| MMLU | **90.36** | 87.51 | 90.43 |
| MMLU-Pro | **73.23** | 68.60 | 70.90 |
| SuperGPQA | **51.36** | 44.86 | 48.42 |
| GPQA | 51.42 | 45.01 | **51.52** |
| MATH | 72.78 | 60.54 | **74.38** |
| EvalPlus | **78.76** | 76.05 | 78.06 |
| SWEBench-Pretrain | **50.99** | 41.66 | 49.24 |
| MMMLU | **84.86** | 79.74 | 84.53 |

*Source: 2608.30320v1 §5 Table 11.*

**Verdict (RU).** Qwen — единственный, кто публикует рядом плотную 27B и MoE 125B-A6B. При 4.5× меньшем active 125B выигрывает 2.9 пункта MMLU-Pro, но проигрывает MATH 1.6 пункта плотной 27B. Это не «MoE всегда лучше», а «MoE лучше по knowledge, dense лучше по math на этом бюджете».

---

### 1.7 Llama 4 (Maverick / Scout)

**Identity.** No arXiv report. Meta blog 2025-04-05 + `meta-llama/llama4/MODEL_CARD.md`. Maverick: 400B total / 17B active; 48 layers; d=5120; 40 heads, 8 KV; 128 routed experts + 1 shared, **top-1**; alternating dense and MoE layers (`interleave_moe_layer_step=2`); vocab 202,048; ~22T tokens; 1M context. Scout: 109B / 17B active; 16 experts; ~40T tokens; 10M context (pretrained/post-trained at 256K).

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| MoE 128 + 1 shared, top-1 | each token → shared + exactly one routed expert | CAPACITY / LATENCY | 400B / 17B ⇒ **sparsity 23.5** [C 400/17]; top-1 minimizes dispatch [T blog] | 22T tokens |
| Alternating dense / MoE | every second layer dense | LATENCY | `interleave_moe_layer_step=2` [T HF config]; "for inference efficiency" [T blog] | 400B |
| iRoPE | interleaved attention layers **without positional embeddings** + inference-time attention temperature scaling | LATENCY / length | 10M context (Scout), 1M (Maverick) [T blog] | 40T / 22T |
| Early fusion | text+image+video tokens in one backbone from pretraining | CAPACITY | 200 languages; up to 48 images in pretraining [T blog + model card] | 400B |
| Distillation from Behemoth | 288B-active, 16-expert, ~2T teacher | CAPACITY | Behemoth not released; teacher for both students [T blog] | 2T teacher |
| QK-norm | query-key normalization | STABILITY | **Scout `use_qk_norm=true`, Maverick `false`** [T HF configs] | 400B / 109B |

**Published benchmarks.** The blog publishes **bar charts, not a numeric table**. Verifiable numbers in text: LMArena Elo **1417** for an experimental Maverick chat version; Scout fits on one H100 with int4; Maverick on one H100 host [T blog]. **No MMLU/GPQA table exists in the primary artifact** — do not cite one.

**Verdict (RU).** Llama 4 — ставка на top-1 и latency, а не на качество на параметр. Top-1 против top-8 у всех остальных — единственный такой случай в наборе; именно поэтому Maverick дешевле в обслуживании, но не выигрывает по benchmark/param.

---

### 1.8 Gemma 3 27B

**Identity.** arXiv:2503.19786v1 (2025-03-25). 27B nominal: 25.6B non-embedding + 1.416B embedding + 417M vision; 62 layers; d=5376; 32 heads, 16 KV, head_dim 128; FFN 21504; vocab 262,208; 128K context (1B: 32K); 14T tokens for 27B / 12T / 4T / 2T; **dense, no MoE**.

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| 5 local : 1 global, window 1024 | only every 6th layer attends globally | MEMORY | KV overhead at 32K: **60%** global-only → **<15%** with 5:1 + sw=1024 [T §5.2, Fig. 5] | 27B, 32K |
| QK-norm | normalizes q/k before attention | STABILITY | no isolated number in the report [T §3] | 27B |
| Dense FFN | no experts | — | 25.6B non-embedding for 27B nominal [T Table 1] | 14T tokens |
| 262K vocab | large subword vocabulary | CAPACITY | embedding 1.416B = **5.5%** of non-embedding [C 1.416/25.6] | 27B |
| 128K context | RoPE base 1M on global layers | length | 1B capped at 32K; 4B+ at 128K [T §2] | 27B |

**Published benchmarks (IT, zero-shot).**

| Benchmark | Gemma 3 27B | Gemma 3 12B | Gemma 2 27B |
|---|---:|---:|---:|
| MMLU-Pro | **67.5** | 60.6 | 56.9 |
| MATH | **89.0** | 83.8 | 55.6 |
| GPQA Diamond | **42.4** | 40.9 | 34.3 |
| LiveCodeBench | **29.7** | 24.6 | 20.4 |
| MMMU (val) | **64.9** | 59.6 | — |

*Source: 2503.19786v1 §5 Table 6.*

**Verdict (RU).** Контрпример всему остальному набору: плотная модель, без MoE, без линейного внимания, без MLA. 27B — тот масштаб, где Google всё ещё не платит за гетерогенность. Это не аргумент против архитектуры, а верхняя граница «dense достаточен».

---

### 1.9 OLMoE-1B-7B

**Identity.** arXiv:2409.02060v2 (2024-09-03; v2 2025-03-03). 6.9B total / 1.3B active; 16 layers; d=2048; 16 heads; 64 experts top-8, expert d_ff 1024; **no shared expert**; QK-norm; vocab 50,304; 5.133T tokens; 4096 context.

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| 64 experts top-8 | fine-grained, dropless token-choice | CAPACITY | 6.9B / 1.3B [T Table 1] | 5.133T |
| No shared expert | all capacity routed | CAPACITY | explicit ablation choice [T Table 1, §4.1.3] | 1.3B active |
| QK-norm | stability | STABILITY | §4.2.5 | 1.3B |
| MoE vs dense | 1.3B dense vs 6.9B/1.3B MoE, 130B tokens | CAPACITY | **~3× fewer tokens/FLOPs**; 2× wall-clock; 23.6k vs 37.5k tok/s/GPU [T Fig. 4] | **1.3B active, 130B tokens** |
| 5.133T tokens | 1.3 epochs | — | 5.133T [T §3] | 6.9B |

**Published benchmarks.**

| Model | Active | MMLU | HellaSwag | ARC-C | ARC-E | PIQA | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLMoE-1B-7B | 1.3B | **54.1** | **80.0** | **62.1** | **84.2** | **79.8** | **70.2** |
| OLMo-1B | 1.3B | 32.1 | 67.5 | 36.4 | 53.5 | 74.0 | 62.9 |
| DCLM-1B | 1.4B | 48.5 | 75.1 | 57.6 | 79.5 | 76.6 | 68.1 |
| Llama3.2-1B | 1.2B | 38.2 | 67.3 | 43.5 | 71.6 | 73.7 | 62.5 |

*Source: 2409.02060v2 §3 Table 4.*

**Verdict (RU).** Единственный полностью открытый MoE с контролируемым dense-близнецом. Это и есть измеренная точка crossover для MoE: 1.3B active, 130B токенов, 3× по FLOPs. Ниже неё published head-to-head нет.

---

### 1.10 Olmo Hybrid 7B

**Identity.** arXiv:2604.03444v4 (2026-04-03; v4 2026-06-15). 7B; Olmo 3 7B blueprint with 75% of attention layers replaced by Gated DeltaNet, 3 GDN : 1 attention; trained up to 6T tokens.

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| GDN 3:1 | 75% linear layers | FLOPs / quality | BPB at 100M: transformer **1.037** vs hybrid **0.986** [T Table 5] | **60M–1B sweep** |
| 6T tokens | long pretraining | CAPACITY | matches Olmo 3 7B on MMLU with **49% fewer tokens** [T abstract] | 7B |
| RULER 64K | long-context retrieval | length | **+14.1%** over Olmo 3 [T abstract] | 7B |
| Scaling law | Chinchilla fit, 35 points per architecture | overall | hybrid curve below transformer at every scale 60M–1B; projected **1.3–1.9× token savings at 1B–70B** [T §4.1, Fig. 8] | 60M–1B |

**Published benchmark (BPB, OlmoBaseEval average).**

| Architecture | 60M | 100M | 190M | 370M | 600M | 760M | 1B |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 1.155 | 1.037 | 0.950 | 0.839 | 0.781 | 0.747 | 0.682 |
| GDN (0:1) | **1.080** | 0.990 | 0.895 | 0.795 | 0.761 | 0.722 | 0.677 |
| GDN 3:1 hybrid | **1.077** | **0.986** | **0.891** | **0.799** | **0.754** | **0.717** | **0.669** |

*Source: 2604.03444v4 §5.3 Table 5. Blueprint-matched (same layers/heads/d per nominal scale), each at 8× Chinchilla-optimal tokens; actual parameter counts differ (Table 22), so the row is **not** a strict iso-FLOP pair — the fitted law in §4.1 is.*

**Verdict (RU).** Это самый важный источник для нас: линейный гибрид выигрывает уже при **60M**, а не «выше 1B». Но при 60–190M чистая GDN (0:1) практически равна гибриду (1.080 vs 1.077), то есть глобальное внимание на нашем масштабе не окупается.

---

### 1.11 MiniMax-M2

**Identity.** arXiv:2605.26494v2 (2026-05-26; v2 2026-07-30). 229.9B total / 9.8B active; 62 layers; d=3072; 48 query heads / 8 KV (GQA); 256 experts top-8, sigmoid gating + learnable bias; **full attention in all layers**; MTP 1→3; vocab 200,064; 29.2T tokens; 192K context.

| Component | What it is | Axis | The number that justifies it | Measured at |
|---|---|---|---|---|
| Full attention, all 62 layers | GQA 48/8, RoPE | LATENCY / quality | SWE-verified **54.7** vs **50.2** with SWA hybrid; RULER 128K CWE **90.0 → 72.0** [T Table 2/3] | **229.9B, 29.2T** |
| 256 experts top-8 | fine-grained + sigmoid + bias | CAPACITY | 229.9B / 9.8B ⇒ **sparsity 23.5** [C 229.9/9.8] | 29.2T |
| MTP 1→3 | weight-copied expansion for speculative decode | LATENCY | 3 draft modules; accept length table [T §2.3] | 29.2T |
| 192K context | multi-stage extension 8K→32K→192K | length | decay phase 9.3T tokens [T §3] | 29.2T |

**Published benchmarks (M2.7 checkpoint).**

| Benchmark | M2.7 |
|---|---:|
| SWE-bench Pro | **56.2** |
| SWE-bench Multilingual | **76.5** |
| Multi-SWE-bench | **52.7** |
| Terminal-Bench 2.0 | **57.0** |
| AIME 2026 | **94.2** |
| GPQA-Diamond | **89.8** |

*Source: 2605.26494v2 §1.*

**Verdict (RU).** Единственный в наборе, кто **отказался** от линейного/разрежённого внимания после собственных абляций на полном масштабе. Это не «отсталость»: их SWA-варианты теряли 4.5 пункта SWE и 18 пунктов RULER@128K. Прямое противоречие с Kimi/Qwen, и это надо держать в голове, а не выбирать удобную сторону.

---

## 2. Cross-model synthesis

### 2.1 Component counts (11 checkpoints)

| Component | Count | Which models | Verdict |
|---|---:|---|---|
| MoE FFN | **9/11** | DS-V4.1, K3, Kimi Linear, GLM-5.3, GLM-5.3-Flash, Qwen3.8, Llama 4, OLMoE, MiniMax-M2 | **converged (all ≥6.9B); dominant above 100B** |
| Fine-grained experts (E ≥ 64) | **9/9 MoE** | same nine | **converged** |
| Shared expert(s) ≥ 1 | **7/9 MoE** | DS, K3 (2), KL, GLM-5.3, GLM-F, Qwen, Llama 4 | **converged** |
| top-k ≥ 2 | **8/9 MoE** | all except Llama 4 (top-1) | **converged** |
| Vocabulary ≥ 128K | **9/11** | DS, K3, KL, GLM-5.3, GLM-F, Qwen, Llama 4, Gemma 3, MiniMax | **converged** |
| Linear-attention hybrid | **5/11** | K3, Kimi Linear, GLM-5.3-Flash, Qwen3.8, Olmo Hybrid | **split bet** |
| Sparse attention (top-k indexer) | **4/11** | DS CSA2, GLM-5.3 DSA, GLM-F DSA, Qwen QSA | **split bet** |
| Latent / compressed KV (MLA-family) | **5/11** | DS, K3, Kimi Linear, GLM-5.3, GLM-F | **converged for serving** |
| Residual widening (mHC / AttnRes / GR) | **4/11** | DS mHC, K3 AttnRes, GLM-F mHC, Qwen GR | **bet, rising** |
| MTP | **5/11** | DS (3), K3 (1), GLM-5.3 (1), Qwen (1), MiniMax (1→3) | **converged for serving** |
| QK-norm | **3/11** | Gemma 3, OLMoE, Llama 4 **Scout only** | **neither converged nor dead** |
| NoPE in global layers | **4/11** | K3, Kimi Linear, GLM-5.3-Flash (`mla_use_nope=true`), Llama 4 iRoPE | **bet** |
| Full attention only | **3/11** | Gemma 3, OLMoE, MiniMax-M2 | **still shipped** |
| Native multimodal | **6/11** | DS, K3, GLM-F, Qwen (Omni), Llama 4, Gemma 3 | **converged above 100B** |

### 2.2 What appears once (the bets)

| Component | Only in | What it bets on | Published number |
|---|---|---|---|
| CED encoder-decoder split | DS-V4.1 | prefill-heavy agent workloads | 8B prefill vs 16B decode; decode FLOPs +1/4 over 256× context [T 2609.19969v1 §1] |
| Engram (196B conditional memory) | DS-V4.1 | capacity outside the backbone | 196B params at layers 1,14 [T §2.4.2] |
| FP4 main KV | DS-V4.1 | HBM/SSD bandwidth | inside 890 B/token [T §2.4.4] |
| DSpark | DS-V4.1 | decode latency | not quantified [T §2.4.3] |
| AttnRes | K3 | depth information flow | O(Nd) vs O(Ld), N≈8 [T §2.2] |
| Stable LatentMoE (ℓ=0.5d + RMSNorm) | K3 | communication vs expert width | 896 experts at 0.5× width, sparsity 56 [T §2.3] |
| SiTU-GLU | K3 | bounded activations at 2.8T | |f|≤100 [T §2.3.2] |
| Quantile Balancing | K3 | load balance at ~10³ experts | one-pass quantile update [T §2.3.3] |
| n-gram tables off-accelerator | Qwen3.8 | capacity without VRAM | 51B params [T abstract] |
| Gated Residual, 4 branches | Qwen3.8 | residual read/write | −0.026 loss at 276B tokens [T §4] |
| iRoPE | Llama 4 | length without positional extrapolation | 10M context (Scout) [T blog] |
| top-1 routing | Llama 4 | dispatch latency | 128 experts, top-1 [T blog] |
| Early fusion | Llama 4 | multimodal in one backbone | 48 images pretraining [T blog] |
| MLA-256 + Muon split | GLM-5.3 | decode compute at constant params | matches GQA-8; −1/3 decode attention in a **9B-class ablation** [T Table 1] |
| IndexCache (indexer shared by 4 layers) | GLM-5.2/5.3 | indexer O(L²) cost | 4× fewer indexer runs [T HF config; 2603.12201v1] |
| Hybrid KDA + DSA | GLM-5.3-Flash | both linear and sparse at once | 34 KDA + 11 DSA [T HF config] |

### 2.3 What was removed (a removal is evidence)

| Removed | From → to | Evidence (number) | What it proves |
|---|---|---|---|
| **MoE at 7B** | OLMoE-1B-7B (2024) → Olmo 3 7B / Olmo Hybrid 7B (2025–26), both dense | OLMoE 6.9B/1.3B; Olmo 3 dense 7B [T 2512.13961v2]; Olmo Hybrid dense+GDN [T 2604.03444v4] | the most open lab stopped paying for MoE below 10B |
| **Linear attention at 230B** | MiniMax-Text-01 (Lightning hybrid) → M2 (full attention) | SWE 54.7 vs 50.2; RULER 128K CWE 90.0 vs 72.0 [T 2605.26494v2 Tables 2–3] | linear/sparse attention lost at production scale in their regime |
| **HCA** | DeepSeek-V4 (CSA+HCA hybrid) → V4.1 (pure CSA2) | "different from V4 … uses pure CSA2" [T 2609.19969v1 §2.3] | a whole attention type deleted in one generation |
| **SWA** | Qwen3.5 SWA hybrid → Qwen3.8 GDN hybrid | full 49.87, SWA 51.15, GDN 53.81 [T 2608.30320v1 Table 1] | SWA was tried and dropped, not ignored |
| **SWA / GDN at 9B** | GLM-5 considered both, shipped DSA | GDN −8.59 RULER@64K, SimpleGDN −3.59; DSA lossless [T 2602.15763v2 Tables 5–6] | same mechanism, opposite conclusion to Kimi/Qwen |
| **Full MLA in 75% of layers** | Kimi K2 (61 MLA) → K3 (69 KDA + 24 MLA) | Table 1 layer composition [T 2607.24653v2] | full attention retained only 25% of depth |
| **QK-norm** | Llama 4 Scout (`true`) → Maverick (`false`) | HF configs [T] | even within one family the component is optional |

**RU-вывод по синтезу.** Конвергенция есть только в ёмкости: MoE + мелкие эксперты + shared + top-k≥2 + большой словарь. Внимание — поле боя трёх несовместимых лагерей, и у каждого есть опубликованное измерение в свою пользу. Любой, кто говорит «фронтир использует X», выбирает один лагерь и молчит о двух других.

---

## 3. The transfer question: 95M / 1 KB / 22 GB / one 16 GB card

### 3.0 The stand

| Quantity | Value | Source |
|---|---:|---|
| Parameters | **95,035,200** | `port/configs/flagship3_dense.toml` [ours] |
| Layers / d_model / heads × dim / FFN | 12 / 832 / 13×64 / 3072 | same [ours] |
| Vocabulary | **256** (byte-level, 1 byte = 1 token) | same [ours] |
| Context / batch / grad_accum | 1024 / 8 / 3 ⇒ **24,576 bytes per optimizer step** | same [ours] |
| FLOPs per byte | **692.9 MFLOP** = 6N (570.2M) + attention (122.7M) | [C], formula below |
| Measured step time | **0.462 s** ⇒ **53.2 kB/s** (`night_opt/mix_s1.jsonl`, `flagship3_dense`) | `docs/reviews/2026-11-07-muon-audit.md` §0 [ours] |
| Sensitivity | the Rust trainer at 9.2M / batch 8 × seq 512 reads 245 ms = **16.7 kB/s**; if that rate holds at 95M, every cost below is **3.2× higher** | `AGENTS.md` §3.1 [ours] |
| Corpus | **22 GB** = 22e9 byte-tokens | brief; byte-level ⇒ 1 byte = 1 token [C] |
| Steps for 22 GB | 22e9 / 24,576 = **895,182** | [C] |
| Wall-clock for 22 GB | 895,182 × 0.462 s = **114.9 h = 4.8 days** | [C] |
| Bytes per parameter | 22e9 / 95.0M = **231.5** | [C] |
| GPU | one 16 GB card, MFU ≈ 2%, launch-bound | `docs/research/2026-11-07-frontier-vs-vanilla.md` §2.0 [ours] |

Formulas: attention FLOPs/byte = `12·L·d·s`; dense block = `6N`; KV bytes/token = `2·L·n_kv·d_head·2` (bf16 K and V).

**What stand is this.** The numbers are for `flagship3_dense`, the repo's **dense control arm** (12 unique layers, plain transformer). The shipped model is a weight-shared loop block with KDA as its only attention arm (`AGENTS.md` §1, §3.5). For that model the MEMORY axis is smaller still — KDA keeps a fixed-size state, not a per-token KV — and the attention FLOPs share is the same to first order. So the arithmetic below is an **upper bound on MEMORY** and a fair estimate of the FLOPs axis. The one number that does not transfer is the step rate: the dense control is the slowest arm per byte, and every experiment cost in §4 is quoted at 53.2 kB/s with the 3.2× sensitivity above.

### 3.1 MEMORY: dead at 1 KB

| Quantity | Formula | Value |
|---|---|---:|
| KV per token, all 12 layers, bf16 | 2 · 12 · 13 · 64 · 2 | **39,936 B ≈ 39.0 KB** |
| KV per sequence, s=1024 | ×1024 | **39.0 MiB** |
| KV per batch 8 | ×8 | **312 MiB = 0.305 GiB** |
| Share of 16 GB card | 312 / 16,384 | **1.90%** |
| KV with GQA 4:1 (4 KV heads) | 2·12·4·64·2·1024·8 | **96 MiB (0.59%)** |
| KV with MLA d_c=256 + rope 64 | 2·(256+64)·12·1024·8 | **60 MiB (0.37%)** |
| Weights bf16 | 95.0M × 2 | **181.3 MiB** |

**RU.** Ось MEMORY не существует при обучении: KV-кэш вообще не аллоцируется, K/V живут как активации до backward. Даже при сервинге 312 MiB на batch 8 — это 1.9% карты. CSA2, FP4 KV, SWA bounded replay, MLA, GQA — все эти механизмы оптимизируют величину, которая у нас равна 0.3 GiB. **Мёртвая ось, без эксперимента.**

### 3.2 FLOPs: almost dead at 1 KB

| Quantity | Formula | Value |
|---|---|---:|
| Attention FLOPs/byte | 12 · 12 · 832 · 1024 | **122.7 MFLOP** |
| Total FLOPs/byte | 570.2 + 122.7 | **692.9 MFLOP** |
| Attention share | 122.7 / 692.9 | **17.7%** |
| 3:1 linear saving | 0.75 × 122.7 | **92.0 MFLOP = 13.3%** |
| Iso-FLOPs parameter equivalent | 92.0 / 6 | **+15.3M params (+16%)**, d 832 → ≈890 |
| Wall-clock saving at MFU 2% | 13.3% of a launch-bound step | **≈0 s** |
| Sparse attention, top-512 of 1024: saved | 0.5 × 122.7 | **61.3 MFLOP** |
| Sparse attention indexer cost (frontier shape 32 heads × 128) | 12 · 2·32·128·1024 | **100.7 MFLOP** |
| **Net sparse cost** | 100.7 − 61.3 | **+39.3 MFLOP = +5.7% worse** |
| SWA window 128 on 75% of layers: saved | 0.75·122.7·(1−128/1024) | **80.5 MFLOP = 11.6%** |

**RU.** Три вывода, все арифметические. (1) Полный переход на линейное внимание экономит 13.3% счёта — это +16% параметров при iso-FLOPs, но ноль секунд при нашем launch-bound шаге. (2) Разрежённое внимание при s=1024 **дороже** плотного на 5.7%: индексер стоит больше, чем экономит top-512. (3) SWA-128 экономит 11.6% счёта, но оставляет каждому токену только 128 предыдущих в 75% слоёв — при 1K это 1/8 контекста; качество придётся мерить, а не предполагать.

### 3.3 CAPACITY: alive at 22 GB (this corrects the 1 GB conclusion)

**MoE data arithmetic.** At 22e9 byte-tokens, 12 layers:

| Experts E | top-k | Expert-token assignments | Tokens per expert |
|---:|---:|---:|---:|
| 8 | 2 | 22e9·12·2 = 528e9 | **66.0e9** |
| 64 | 2 | 528e9 | **8.25e9** |
| 256 | 2 | 528e9 | **2.06e9** |
| 512 | 2 | 528e9 | **1.03e9** |
| 64 | 4 | 1,056e9 | **16.5e9** |

Compare at the same L=12, k=2: the fine-grained MoE law's smallest run, 16B tokens at 129M total [T 2402.07871v1 §3, §5.4], gives `16e9·12·2/64 = 6.0e9` assignments/expert at E=64 — the **same order** as our 8.25e9; at E=512 ours is 1.03e9. The exact comparison depends on their L/k/E, which the law parameterises by granularity rather than listing per run, so this is an order-of-magnitude statement [C], not an exact ratio. Conclusion: at 22 GB, **data is not obviously the constraint on expert granularity** — the earlier "MoE needs data" verdict was a 1 GB artifact, but this does not prove 22 GB is enough for E=512.

**MoE memory arithmetic.** Weights bf16 + grads bf16 + AdamW fp32 (m, v) = 12 bytes/param (lower bound; an fp32 master copy adds 4 bytes/param):

| Total params | ×active | Weights+grads+Adam | Share of 16 GB |
|---:|---:|---:|---:|
| 190M | 2× | **2.28 GB** | 13.3% |
| 380M | 4× | **4.56 GB** | 26.6% |
| 760M | 8× | **9.12 GB** | 53.1% |

**Vocabulary arithmetic.** Embedding params = `V · d`:

| Vocabulary | Embedding params | Share of 95M model |
|---:|---:|---:|
| 256 (ours) | **0.21M** | **0.22%** |
| 32,000 | 26.6M | 28.0% |
| 50,304 (OLMoE) | 41.9M | 44.0% |
| 129,280 (DS-V4.1) | 107.6M | **113.2%** |
| 163,840 (Kimi) | 136.3M | **143.4%** |
| 248,320 (Qwen) | 206.6M | **217.4%** |

**RU.** Словарь ≥128K при d=832 — это больше параметров, чем вся модель: 107.6M только на эмбеддинги при 95M. Фронтир может себе это позволить при d=4096–7168; мы — нет, если не факторизовать/не отвязать эмбеддинги. Это не «вкус», это деление.

### 3.4 Per-component verdict at 95M / 1K / 22 GB / 16 GB

| Component | Axis | Arithmetic at our scale | Alive? | Smallest test |
|---|---|---|---|---|
| Fine-grained MoE + shared | CAPACITY | 8.25e9 tok/expert (E=64); 4.56 GB at 4× total | **YES**, quality unproven | 3 arms × 1 GB × 2 seeds = **31.3 h** |
| Linear attention (GDN/KDA) | FLOPs | saves 13.3% FLOPs; ≈0 wall-clock at 2% MFU; Olmo predicts −0.054 BPB | **YES (quality), NO (speed)** | 2 arms × 1 GB × 2 seeds = **20.9 h** |
| Sparse attention (DSA/CSA2/QSA) | MEMORY/FLOPs | net **+5.7%** FLOPs at s=1024 | **NO** | arithmetic only, 0 h |
| SWA-128 | FLOPs | saves 11.6%; covers 1/8 of context | **UNPROVEN** | 2 arms × 1 GB = 10.4 h |
| mHC / AttnRes / GR | STABILITY | 4× residual traffic; 20 Sinkhorn iters; our 9M tie | **UNPROVEN** | 2 arms × 1 GB = 10.4 h |
| High vocab (≥128K) | CAPACITY | 107.6M embedding params > model | **NO** | arithmetic only |
| n-gram tables off-accelerator | CAPACITY | 20M×8 table = 160M host params; host bandwidth per token | **MAYBE** | 2 arms × 1 GB = 10.4 h |
| MTP | CAPACITY/LATENCY | +1 layer ≈ +8% params and step FLOPs | **UNPROVEN** | 2 arms × 1 GB = 10.4 h |
| Load balancing (QB / noaux bias) | STABILITY | ~0 runtime cost; prevents dead experts | **YES, cheapest** | 2 arms × 0.5 GB = 5.2 h |
| FP4/FP8 KV, CED, DSpark, Engram-196B | MEMORY/LATENCY | KV 0.3 GiB; no serving; Engram = 196B/552B scale | **NO** | none |

*(Costs use 1 GB = 5.22 h measured at 53.2 kB/s; 0.5 GB = 2.61 h.)*

**RU-вердикт по переносу.** При 95M/1K/22GB живы ровно две вещи: **ёмкость через MoE** (данных хватает, памяти хватает, но quality-выигрыш не измерен ниже 1.3B active) и **линейное внимание** (единственный опубликованный положительный результат ≤100M). Разрежённое внимание и большой словарь убиваются делением, а не экспериментом. mHC/MTP/Engram — платить реальную цену за недоказанный выигрыш.

---

## 4. What the frontier knows that we do not

Each row: the mechanism, the frontier evidence, the specific unknown at 95M, and the smallest experiment that would settle it. Costs at 53.2 kB/s.

| # | Mechanism | Frontier evidence | Unknown at 95M | Smallest experiment | Cost | Kill criterion |
|---|---|---|---|---|---|---|
| 1 | Load balancing without aux loss (QB / noaux bias) | K3 QB at 896 experts [T 2607.24653v2 §2.3.3]; DS/V3 bias correction [T 2412.19437v2] | does it prevent dead experts at 64 experts / 12 layers? | 2 arms: aux-loss vs bias/QB, 64 experts top-2, 0.5 GB, 1 seed | **2.6 h** | dead-expert count > 2/64 or BPB worse |
| 2 | Latent-space experts (ℓ=0.5d) | K3 ℓ=3584 at d=7168, sparsity 56 [T Table 1] | does ℓ=416 preserve capacity at d=832? | 4 arms: dense / full-width MoE / LatentMoE / LatentMoE+RMSNorm, 1 GB, 1 seed | **20.9 h** | LatentMoE > full-width by >0.02 BPB |
| 3 | Attention ratio 0:1 vs 3:1 vs 7:1 at 1K | Olmo: 7:1 best at 60–190M, 3:1 best at 600M–1B [T 2604.03444v4 §5.3] | where is the ratio optimum at 95M/1K? | 3 arms: GDN 0:1, 3:1, 7:1, 1 GB, 1 seed | **15.7 h** | none of the hybrids beats dense by ≥0.02 BPB |
| 4 | Residual widening (mHC/AttnRes/GR) | DS −50% traffic [T §2.4.1]; Qwen −0.026 loss at 276B [T §4]; our 9M tie [ours] | does 4-branch residual help at 12 layers? | 2 arms: plain vs 4-branch gated, 1 GB, 1 seed | **10.4 h** | <0.02 BPB gain |
| 5 | Sparse attention indexer | DS 890 B/token [T]; Qwen 7.6× prefill at 1M [T] | can indexer cost < saving at s=1024? | arithmetic: 100.7 vs 61.3 MFLOP/byte | **0 h** | already killed |
| 6 | Off-accelerator n-gram capacity | Qwen 51B tables [T abstract]; DS Engram 196B [T] | does it generalize, or memorize like our Engram? | 2 arms: baseline vs n=3 hash table, 1 GB, held-out BPB | **10.4 h** | train CE ↓ but held-out flat |
| 7 | Bounded activations (SiTU-GLU / clamp) | K3 |f|≤100 [T]; DS SwiGLU clamp 10 [T §4.2.1] | does it stop our NaN/spike class at 95M? | 2 arms: SwiGLU vs SiTU, 1 GB, log max-logit | **10.4 h** | no reduction in spike rate |
| 8 | High-vocab tokenization | 128K–262K in 9/11 | can a 95M byte model use subwords without embedding blowup? | arithmetic + 1 arm: tied 32K vocab with low-rank adapter, 1 GB | **5.2 h** | params > 95M or BPB worse |
| 9 | MTP | DS 3 layers, GLM 1, MiniMax 1→3, Qwen 1, K3 1 | does a second-token loss help byte-level 95M? | 2 arms: CE vs CE+MTP, 1 GB, 1 seed | **10.4 h** | next-1 BPB worse or no gain |
| 10 | Depth recurrence / loops | MoR: 135M worse (+0.0155), 360M better (−0.2759) [T 2507.10524v3 Table 15] | is 95M below the loop crossover? | 2 arms: 12 unique vs 6 unique × 2 passes at iso-FLOPs | **10.4 h** | loop arm worse by >0.02 BPB |

**RU.** Фронтир знает, как **не разваливать** очень большие разрежённые модели (QB, bounded GLU, RMSNorm перед up-projection, Sinkhorn). Он не знает, переносится ли это на 64 эксперта и 12 слоёв, потому что никто не публиковал такой прогон. Наш самый дешёвый способ узнать — не копировать 896 экспертов, а проверить три вещи, которые стоят 2.6–20.9 h каждая: балансировку, latent-ширину и ratio 0:1.

---

## 5. The honest gap: where these architectures actually start to beat dense

### 5.1 Published crossovers, mechanism by mechanism

| Mechanism | First scale where it beats dense (published) | Matched figure | Match type | Source |
|---|---|---|---|---|
| GDN 3:1 hybrid | **60M params** (smallest measured point) | BPB **1.077 vs 1.155** at 60M; **0.986 vs 1.037** at 100M | blueprint-matched (same layers/heads/d), each at 8× Chinchilla; law fit is FLOP-based | 2604.03444v4 §5.3 Table 5, §4.1 |
| Pure GDN (0:1) | **60M** | 1.080 vs 1.155 at 60M; 0.677 vs 0.682 at 1B | same | 2604.03444v4 Table 5 |
| MoE (64 experts top-8) | **1.3B active / 6.9B total**, 130B tokens | **~3× fewer tokens/FLOPs**; 2× wall-clock (23.6k vs 37.5k tok/s/GPU) | **active-param and FLOP matched** (same 1.3B active, same 130B tokens) | 2409.02060v2 §4.1.2, Fig. 4 |
| Fine-grained MoE law | **129M total**, 16B tokens (smallest measured) | the paper's fitted law says MoE consistently outperforms dense and the gap widens with scale [T abstract, §5.4 Table 1] | fitted law vs exact N,D; FLOPs from active params | 2402.07871v1 §3, §5.4 |
| KDA 3:1 | **653M active**, 38.8B tokens | val PPL **5.65 vs 5.77**; train PPL 9.23 vs 9.45 | **equal FLOPs** (stated), same data | 2510.26692v2 §5.2 Table 1 |
| GDN 3:1 at 1.3B | **1.3B**, 100B tokens | PPL **15.91 vs 18.53** | param+token matched; GDN has fewer FLOPs (not iso-FLOP) | 2412.06464v3 Table 3 |
| GDN 3:1 at 25B-A3B | **25B total / 3B active**, 480B tokens | avg **53.81 vs 49.87** (+3.94) | active-param + token matched; GDN cheaper FLOPs | 2608.30320v1 §2.1 Table 1 |
| Sparse attention (DSA/CSA/QSA) | **≥64K–128K context**, not a parameter scale | NSA 11.6× fewer memory accesses at 64K; QSA 7.6× prefill at 1M | kernel/system level, not iso-FLOP loss | 2502.11089v2 Table 4; 2608.30320v1 §1 |
| Residual widening (mHC / xHC / AttnRes) | **18B** (xHC +4.0 pts) / **9M tie** [ours] | no 95M published win | downstream, not iso-FLOP | 2607.14530v1 abstract; our 2026-11-07 reports |

### 5.2 What this means (RU)

1. **Единого порога нет.** «Эти архитектуры начинают бить dense при N параметров» — ложная постановка. Для линейного внимания порог **≤60M** (измерен), для MoE — **1.3B active** (измерен) или 129M total (закон), для разрежённого внимания порог задаётся **длиной контекста**, а не параметрами.
2. **Наш масштаб попадает в зону, где линейное внимание уже выигрывает, а MoE — ещё нет.** Это переворачивает приоритет: GDN — не «фронтирная игрушка», а единственный механизм с положительным числом на 60M. MoE — ставка с ближайшим измерением на 1.3B, то есть в 13.7× выше нас.
3. **Предыдущий вывод репозитория «ничто не бьёт dense ниже 200M» верен только для MoE/loops.** Он не учитывал Olmo Hybrid Table 5, опубликованную 2026-04: линейный гибрид бьёт dense при 60M на 0.078 BPB. Это надо исправить в `2026-11-06-architecture-at-100m.md`, а не защищать.
4. **Честная цена сравнения.** Olmo Hybrid — blueprint-matched, не param-matched: при одном «номинальном размере» фактическое число параметров разное (Table 22), а закон фитит по точному N. Поэтому 0.078 BPB — это оценка для FLOP-кривой, а не измерение на паре «95M vs 95M». Единственные строгие измерения — OLMoE (active-param + FLOP matched) и Kimi Linear (равный FLOPs); Qwen и GDN — param+token-matched, но не iso-FLOP.
5. **Sparse attention не имеет crossover по параметрам вообще.** При 1K она отрицательна арифметически; при 64K+ выигрывает по памяти/латентности, но не по loss на параметр. Сравнивать её с dense по BPB при 1K — категориальная ошибка.

---

## 6. Not published / not verified

| Item | Status | Why it matters |
|---|---|---|
| DeepSeek-V4.1 vocabulary | **not published in the report**; 129,280 only in HF config | report-level table cannot be cited for vocab |
| Kimi K3 training tokens | **not published** (checked §3.1, §3.2, Table 1) | cannot compute tokens/param or Chinchilla ratio |
| Kimi K3 MTP | Table 1 says 1 layer; HF config `num_nextn_predict_layers=0` | released checkpoint may omit MTP |
| GLM-5.3 active params | **not restated** on the 5.3 card; 40B is the GLM-5 base figure; ≈37.7B rough recomputation from config [C] | exact active count unverified |
| GLM-5.3-Flash benchmark numbers | **image only** (`bench_53.png`) | no citable table |
| Qwen3.8-Flash-Next absolute tokens | **not published**; only "1/3 of predecessor" | cannot compute tokens/param |
| Qwen3.8-Flash-Next context | config 262,144; Omni extends to 1M | 1M is the Omni model, not the base |
| Llama 4 numeric benchmark table | **not published**; blog uses bar charts | no MMLU/GPQA numbers to cite |
| Llama 4 iRoPE layer count | blog describes interleaving; configs show `no_rope_layers: []` in mirrors | exact NoPE layer pattern unverified |
| Gemma 3 full layer config | tech report gives parameter classes, not per-layer d/h; config from a mirror | layer-level numbers are config-derived |
| OLMoE 2 | **does not exist** | no data at all |
| Qwen3.8-Max | no report, no weights | only a competitor's benchmark column |
| MiniMax-M2 full benchmark suite | §8 not extracted; only §1 headline numbers used | fine-grained per-benchmark table not checked |

---

## 6.5 VERIFIED (sourced) vs SPECULATION (my inference)

**VERIFIED (sourced).** These are read from the cited artifact, not inferred:
1. The identities, parameter counts, layer counts, vocabularies, token budgets and contexts in §1 — from the arXiv reports and HF configs/safetensors listed in §7.
2. The component counts in §2.1 and the removals in §2.3 — each row carries a source and a number.
3. Our stand numbers in §3.0 (95,035,200 params, 12 layers, d=832, vocab 256, 24,576 bytes/step, 0.462 s/step) — from `port/configs/flagship3_dense.toml` and the measured review.
4. All arithmetic in §3.1–§3.3 and Appendix A — formulas are shown; the inputs are the two rows above.
5. The published crossovers in §5.1 — each with the match type stated explicitly (FLOP / active-param / blueprint / context).
6. The non-publication list in §6 — each item was searched in the named artifact and not found.

**SPECULATION (my inference).** These are marked [S] in place and must not be quoted as measurements:
1. **[S]** That MoE at 95M active / 380M total will improve BPB at 22 GB. Basis: data and memory arithmetic are sufficient (§3.3), but the nearest measured MoE-vs-dense win is 1.3B active (OLMoE), 13.7× above us.
2. **[S]** That GDN 0:1 will beat the transformer by ≈0.054 BPB at 95M. Basis: linear interpolation of Olmo Hybrid Table 5 between 60M and 100M, assuming our corpus/eval preserves the delta. It is a published-curve extrapolation to a different corpus, not a measurement on our data.
3. **[S]** That sparse attention is a net loss at 1K in wall-clock, not only in FLOPs. Basis: the +5.7% FLOP arithmetic (§3.2) plus launch-bound step; no measurement of an indexer on our trainer.
4. **[S]** That mHC/AttnRes/GR costs more than it returns at 12 layers. Basis: 4× residual traffic and our 9M tie; no 95M measurement.
5. **[S]** That high-vocab tokenization cannot be made to work at 95M via tying/low-rank adapters. Basis: the raw `V·d` arithmetic; a factored embedding could change the conclusion, and no experiment was run.
6. **[S]** That the GLM-5.3 active count is ~40B. Basis: GLM-5's published 40B plus a rough config recomputation (37.7B); the 5.3 card does not restate it.

---

## 7. Sources

| # | Id / URL | Version | Date | Used for |
|---|---|---|---|---|
| S1 | arXiv:2609.19969 | v1 | 2026-09-17 | DeepSeek-V4.1-Flash: architecture, KV, MoE, benchmarks |
| S2 | HF `deepseek-ai/DeepSeek-V4.1-Flash` config + safetensors | — | 2026-09-10 (repo) | vocab 129,280; 763.2B checkpoint; MTP/DSpark config |
| S3 | arXiv:2607.24653 | v2 | 2026-08-07 (v1 2026-07-27) | Kimi K3: 2.78T/104.2B, 896 experts, KDA, AttnRes, QB |
| S4 | HF `moonshotai/Kimi-K3` config + safetensors | — | 2026-06-13 | vocab 163,840; 69 KDA + 24 MLA; MTP 0 in release |
| S5 | arXiv:2510.26692 | v2 | 2025-10-30 | Kimi Linear: 48B/3B, 3:1, −75% KV, 6.3× TPOT, Table 1 |
| S6 | HF `moonshotai/Kimi-Linear-48B-A3B-Instruct` config | — | 2025 | 27 layers, 256 experts, vocab 163,840 |
| S7 | arXiv:2602.15763 | v2 | 2026-02-17 | GLM-5: 744B/40B, DSA, MLA-256, SWA/GDN ablations |
| S8 | HF `zai-org/GLM-5.3` card + config + safetensors | — | 2026-08-25 | 753.3B checkpoint; 78 layers; vocab 154,880; bench table |
| S9 | HF `zai-org/GLM-5.3-Flash` card + config + safetensors | — | 2026-08-25 | 321.3B/18B; 34 KDA + 11 DSA; mHC; 30T tokens |
| S10 | arXiv:2608.30320 | v1 | 2026-08-31 | Qwen3.8-Flash-Next: 125B/6B + 51B n-gram; GDN/QSA/GR; Tables 1, 11 |
| S11 | HF `Qwen/Qwen3.8-Flash-Next` config | — | 2026-08-24 | vocab 248,320; 48 layers; 512 experts top-10; 262K ctx |
| S12 | arXiv:2609.25611 | v1 | 2026-09-22 | Qwen3.8-Omni inherits Flash-Next; 1M context |
| S13 | arXiv:2605.26494 | v2 | 2026-07-30 (v1 2026-05-26) | MiniMax-M2: 229.9B/9.8B, full attention, SWA ablation |
| S14 | arXiv:2604.03444 | v4 | 2026-06-15 (v1 2026-04-03) | Olmo Hybrid: 3:1 GDN, 60M–1B scaling law, Table 5 |
| S15 | arXiv:2512.13961 | v2 | 2025-12-15 | Olmo 3: dense 7B/32B (MoE removed from OLMoE line) |
| S16 | arXiv:2409.02060 | v2 | 2024-09-03 (v2 2025-03-03) | OLMoE: 6.9B/1.3B, 64 experts, MoE-vs-dense 3× |
| S17 | HF `allenai/OLMoE-1B-7B-0924` config | — | 2024-07-20 | 16 layers, 64 experts top-8, vocab 50,304, ctx 4096 |
| S18 | arXiv:2503.19786 | v1 | 2025-03-25 | Gemma 3: 5:1 local/global, KV 60%→<15%, Table 1/6 |
| S19 | HF `unsloth/gemma-3-27b-it-GGUF` config (mirror of gated Google repo) | — | 2025-03 | 62 layers, d 5376, 32/16 heads, vocab 262,208 |
| S20 | Meta AI blog, "The Llama 4 herd" | — | 2025-04-05 | Maverick/Scout/Behemoth, top-1, iRoPE, early fusion, 10M ctx |
| S21 | `meta-llama/llama4/MODEL_CARD.md` | — | 2025-04-05 | 17B/109B/400B, 40T/22T tokens, 10M/1M ctx |
| S22 | HF `unsloth/Llama-4-{Scout,Maverick}` configs (mirrors of gated Meta repos) | — | 2025-04 | 48 layers, 16/128 experts, top-1, interleave step, vocab 202,048, QK-norm flags |
| S23 | arXiv:2402.07871 | v1 | 2024-02-12 | Fine-grained MoE scaling law, 129M–3.7B, 16B–130B tokens |
| S24 | arXiv:2401.06066 | v1 | 2024-01-11 | DeepSeekMoE: fine-grained + shared; 16B at 40.5% compute of 7B |
| S25 | arXiv:2405.04434 | v5 | 2024-05-07 | MLA: −93.3% KV vs DeepSeek 67B; 21B active/236B total |
| S26 | arXiv:2412.19437 | v2 | 2024-12-27 | DeepSeek-V3: 671B/37B, 256 experts top-8, aux-loss-free, MTP |
| S27 | arXiv:2412.06464 | v3 | 2024-12-09 | Gated DeltaNet: 1.3B/100B, Table 3 (15.91 vs 18.53) |
| S28 | arXiv:2502.11089 | v2 | 2025-02-16 | NSA: 11.6× memory, 9.0× fwd / 6.0× bwd at 64K |
| S29 | arXiv:2202.01169 | v2 | 2022-02-02 | Unified scaling laws for routed LMs (effective parameter count) |
| S30 | arXiv:2603.12201 | v1 | 2026-03-12 (updated 2026-11-07) | IndexCache: cross-layer index reuse for DSA (GLM-5.2's "IndexShare") |
| S31 | arXiv:2607.14530 | v1 | 2026-07-16 | xHC: +4.0 pts over mHC at 18B; 1.50×/1.19× compute for vanilla/mHC |
| S32 | arXiv:2507.10524 | v3 | 2025-07-14 | Mixture-of-Recursions: 135M worse, 360M better at iso-FLOPs |
| S33 | `port/configs/flagship3_dense.toml` + `docs/reviews/2026-11-07-muon-audit.md` + `docs/research/2026-11-07-frontier-vs-vanilla.md` + `AGENTS.md` §3.1 | — | 2026-11-07 | our 95M stand: 12 layers, d 832, vocab 256, 0.462 s/step at 53.2 kB/s, 245 ms at 9.2M = 16.7 kB/s sensitivity, MFU 2% |
| S34 | `docs/research/2026-11-06-architecture-at-100m.md` | — | 2026-11-06 | prior claim being corrected: "nothing beats dense ≤200M" |
| S35 | `docs/research/2026-11-07-architecture-frontier.md` | — | 2026-11-07 | our 9M mHC/AttnRes tie, quoted in §3.4/§4/§5.1 |

**Count: 35 distinct sources** — 21 arXiv papers, 9 HF configs/model cards (including 3 mirrors of gated repos), 1 vendor blog + 1 model card (Llama 4), 3 repo documents.

---

## Appendix A — arithmetic definitions

- `KV bytes/token = 2 · L · n_kv · d_head · bytes_per_elem` (K and V; bf16 = 2).
- `attention FLOPs/byte = 12 · L · d · s` (QK^T and AV, 2 FLOPs per MAC, 2 matmuls).
- `dense block FLOPs/byte = 6N` (fwd+bwd).
- `MoE tokens/expert = corpus_tokens · L · k / E`.
- `optimizer memory/param = 2 (bf16 weights) + 2 (bf16 grads) + 8 (fp32 Adam m,v)`.
- `embedding params = V · d` (untied, no factorisation).
- Cost conversion: `1 GB / 53.2 kB/s = 5.22 h`; `0.5 GB = 2.61 h`.
- Interpolation at 95M: linear between the measured 60M and 100M rows of Olmo Hybrid Table 5: `1.155 + (1.037−1.155)·(95−60)/40 = 1.0518` (transformer); `1.077 + (0.986−1.077)·35/40 = 0.9974` (hybrid); Δ = **0.0544 BPB** [C].
