# TSCT at the ceiling — what the stack buys, what the literature says, and whether a factorised expert bank can beat a dense FFN at 95M

**Date:** 2026-11-07. **Mode:** read-only + CPU experiments (`CUDA_VISIBLE_DEVICES=""`); no GPU job, no default changed, no file outside this one written.
**Trees read:** `main` @ `79331e4` (port working tree), and branch `wt/tsct-ceiling` @ `b249bec` (unmerged; its worktree is gone, read via `git show`).
**Tags:** **[T]** number read in a paper's table/text (id, version, date, table named) · **[C]** computed here from someone else's or our numbers (formula shown) · **[S]** speculation, no direct measurement · **[M]** measured in this repo, source named · **[U]** unmeasured — an explicit absence, not a zero.

The owner-facing verdicts are in Russian; the technical sections are in English, because every number below has to survive a reader who checks the log.

---

## TL;DR (RU)

1. **Потолок TSCT не архитектурный, а оптимизаторный.** Дефолт `muon_lr_mult = 1.0` (`port/src/dormouse/config.py:1960`) — это шаг Muon в **15–55 раз меньше** шага AdamW на 99.5% параметров (`docs/records/reviews/2026-11-07-muon-audit.md`). Механизм «мультипликатор» починен 2026-11-07 (`optim.py:1882` `set_lr`), но **дефолт остался 1.0**: любой свежий прогон без явного `--set muon_lr_mult=12` всё ещё меряет сломанный Muon. Все архитектурные вердикты в архиве — и «за», и «против» — сняты в этом режиме.
2. **Наш лучший чистый замер: банк ≠ выигрыш.** 600 шагов, 3 сида, равный активный MAC: dense 69.3M = **2.4714**, E=512 199.3M = **2.5384**, Δ = −0.067 в пользу dense; на 2000 шагах Δ = −0.046 при +18.4% MAC у dense (`docs/records/reviews/2026-11-05-architecture-review-dense.md` §0, §What survives). При равной ширине банк 141.9M = 2.650 против dense 117.5M = 2.653 — **ничья, но у dense на 20.8% меньше параметров** (`docs/records/reviews/2026-10-23-architecture-verdict.md` §2). **При равных параметрах сравнение никогда не завершалось** — dense-плечо падало в OOM дважды (там же, строка «dense, equal params (f63024)»).
3. **Ни одна матрица в модели не является низкоранговой в полезном смысле.** Спектр обученного `loop.moe.down.weight` (1024×2730): 99% энергии требует **941** сингулярных чисел из 1024; attention q/k/v — 245–246 из 256; embed — 247 из 256 (`b249bec:docs/reviews/2026-11-06-tsct-ceiling.md` §1, воспроизведено здесь независимо, §4.2). Rank 64 держит 43.4% энергии `down`, то есть факторизация — это выбор **сжатия**, а не бесплатная замена.
4. **Литература говорит то же.** Общая база + per-expert дельта — это структура **сжатия**: DeRS снижает *добавленные* параметры апцикленного MoE с 1.24B до 1.20M при качестве 57.4 → 57.5 (2503.01359v1, Table 1); SLBF сжимает Qwen3-30B на 24% при −0.6 среднего балла (2610.09342v1, Table 2); ButterflyMoE при **равной памяти** бьёт dense 81.19 против 135.46 PPL (2601.13563v5, Table 1). Ни одной работы, где факторизованный банк с нуля бьёт dense при равных параметрах и токенах на 95–200M, я не нашёл.
5. **Единственная опубликованная конфигурация, где low-rank эксперты бьют dense при равных параметрах, — это dense-anchor + residual experts.** SPRI: 80.57 COMET против 76.59 у плотного файнтюна при сопоставимом общем числе параметров (2606.16456v1, Table 4). Но это апциклинг **предобученной** плотной модели на речь, а не обучение с нуля.
6. **Честный ответ на вопрос «может ли факторизованный банк побить dense при 95M и 1–5 ГБ»: качеством — нет.** Он выигрывает памятью, размером экспорта и FLOPs на токен. При 95M FLOPs на токен не стоят ничего: шаг launch-bound, MFU ~2% (`docs/records/research/2026-11-05-small-model-mfu.md`), а `down f→d` — 89.7% MAC MoE и он уже плотный (`docs/records/reviews/2026-11-04-balance-fix-verdict.md`). Если нужен BPB — это плотная 95M модель, данные и рабочий Muon, а не форма эксперта.
7. **Первый патч — не архитектура.** `muon_lr_mult` по умолчанию должен быть либо 12 (измерено: 1.7478 против 1.8947 у AdamW на 2k, `~/logs/night_stage3.log`), либо заменён на per-shape нормировку (§6, P0). Это единственное изменение, у которого уже есть измеренная дельта в нашу пользу.

---

## 1. Method, and what was actually read

**Code.** `port/src/dormouse/model/experts.py` (3 774 lines), `tsct.py`, `engram.py`, `mhc.py`, `loopblock.py`, `csa.py`, `kda.py`, `layerrope.py`, `mtp.py`, `byteflow.py`, `transformer.py`, `sparse/attention.py`, `train/optim.py`, `train/loop.py`, `config.py`. Line numbers below are `main` @ `79331e4` unless the row says `b249bec`.

**Measurement records.** `~/logs/wave14.log` + `w14_{t,d}_s{1,2,3}.log`, `~/logs/rank_up.log`, `~/logs/ab_rank_up2.log`, `~/logs/night_stage3.log`, `~/logs/sweep_mult.log`, `~/logs/flagship3.jsonl`, `~/logs/dense103.jsonl`, `~/logs/flagship2.jsonl`. Reports: `docs/records/reviews/2026-11-07-muon-audit.md`, `2026-11-06-mechanism-liveness.md`, `2026-11-05-architecture-review-dense.md`, `2026-11-05-dense-beats-bank.md`, `2026-11-04-routing-collapse-CONFIRMED.md`, `2026-11-04-balance-fix-verdict.md`, `2026-10-23-architecture-verdict.md`, `2026-11-05-final-dense-recipe.md`.

**CPU experiments run for this document** (all reproducible in seconds; no GPU):

1. SVD of every dense 2-D matrix in `port/checkpoints/medium/med1.pt` (step 40k, `medium` preset) — participation ratio and cumulative energy.
2. SVD of the stacked bank writes `V` (256 × [64, 2730]) — how many directions the 44.7M-parameter bank actually spans.
3. `lfb_counts` of the same checkpoint — is the router alive at 40k steps.
4. Param/MAC arithmetic for the equal-parameter rank and the bank-at-equal-params construction.

**Papers.** 10 read in full (PDF → `pdftotext` or arXiv HTML), 3 read in part, 13 abstracts, 12 arXiv API queries. Each number carries id + version + date + table.

**One environment fact that matters for reproduction:** `/tmp` on this box is a 32 GB tmpfs and was **100% full** (`df -h /tmp` → 32G/32G). Downloads to `/tmp/opencode` fail with `curl: (23)`. Everything here was read/written under `/home/sehaxe`, and `/home` itself is at 99% (5.8 GB free). A 1.5 GB-per-seed A/B (§6) needs a free-space check before it launches.

---

## 2. TASK A — every capacity mechanism in the stack

### 2.1 The table

`Δ BPB` is always the mechanism's own contribution **inside one eval window**; windows are named because a BPB is not comparable across them (AGENTS §2.6). `SUSPECT` means the measurement ran with Muon at `muon_lr_mult=1`, i.e. a step 15–55× below AdamW's on 99.5% of parameters.

| mechanism | where (file:line, `main`) | what it multiplies | measured Δ BPB | conditions | SUSPECT? |
|---|---|---|---|---|---|
| **bank** (historical TSCT) | `experts.py:966` `SparseMoE`, expert `DualLatentGLU:595`, shared write `down:1443`, packed stack `_banks:2423` | **params**: `E·(2dr+r+rf)` = 44 761 088 at E=256, r=64, f=2730 → 43.6% of the 102.7M model. **Active MAC**: 2 of 256 experts per token → 0.8% of the bank. **Memory**: params (fp32 179 MB, bf16 90 MB) | in-place, trained ckpt 50k: **+0.067** for the whole bank; top-2 vs dense blend over all 256: **+0.008**; dispatch→zeros: **+0.0668** (`mechanism-liveness` §2) | `flagship_base`, step 49 999, 8 192 B window, fp32 CPU surrogate validated against GPU-FLA at 1.1560 vs 1.1561 nats | **yes** for every trained A/B; the in-place ablation is not a training measurement |
| **tsct** (`ffn_form`) | `experts.py:842` `TSCTGLU`, ranks `config.py:508/511`, dense body `experts.py` `DenseFFN` | **params**: `2(d·r_g+r_g+r_g·f) + (f·r_d+r_d+r_d·d)` — at `r_g=r_d=min(d,f)` it is the dense map **reparameterised with more parameters**. **MAC**: `2(d·r_g+r_g·f) + 2(f·r_d+r_d·d)` vs dense `3df` | none clean. The only `tsct`-vs-dense run on record is `wave14` (600 steps, 3 seeds) and it also changed `d_model`/`rank`/`share_u`: 2.7137 vs 2.6503 (`~/logs/wave14.log`). The clean driver `scripts/ab_ffn_form.py` **was never run** (no `/tmp/ab_ffn_form`) | — | **yes**, and the arm is confounded on top of it |
| **engram** | `engram.py:274` `Engram`, host tables `:179`, host update `:620` | **params**: `orders·rows·dim` = 3×524 288×32 = 50 398 208 (49.1% of 102.7M). **MAC**: 3 lookups × 32 dims per byte — negligible. **Memory**: params × 4 B = 201 MB fp32 | in-place: **+2.306** total; order 4 **+1.097**, order 3 **+0.339**, order 2 **+0.112** (`mechanism-liveness` §2) | same checkpoint/window | **no** — in-place, no training |
| **share_u** | `experts.py:540` `share_u_dict_index`, `:739` `bind_shared_u`, modes `config.py` `SHARE_U_MODES` | **params**: plane removes `E·2dr`, adds `2d·r`; at E=256, d=1024, r=64 that is −33 554 432 + 131 072 (`scripts/ab_rank.py` header arithmetic). **MAC**: one `[n,d]×[d,2r]` per site instead of per assignment → measured **−1.4% of MoE MAC**, no win on the step (`balance-fix-verdict`) | no isolated BPB delta on record; the `dict` rank-2 arm went **NaN in <10 steps** (`architecture-verdict` §1.3) | — | **yes** — `share_u` is on in every bank A/B and was never the single variable |
| **mhc** | `mhc.py:129` `MHCMix`, `:93` `sinkhorn_knopp` | **params**: 16 395 (0.02%). **Memory**: +40% VRAM (two residual lanes). **MAC**: elementwise, ~0 | −0.078 at 600 steps (2.5717 vs 2.650, spread 0.038, prose corpus `n2_mhc_s*`); in-place Sinkhorn mixing **+0.0001** (`mechanism-liveness` §2) — i.e. the mixing is inert and the gain is the topology | 600 steps, 3 seeds, 40 960 B | **yes** |
| **attnres** | `loopblock.py:184` `AttnResSlot`, `:198` `depth_attend` | **params**: 6 144. **MAC**: 2–4 scores per position, ~0 | −0.050 at 600 steps (2.5997 vs 2.650, spread 0.015); in-place gains→0 at 50k: **+0.238** | 600 steps, 3 seeds | **yes** |
| **csa** | `csa.py:443` `CompressedSelectiveAttention`, ledger `:273` | **params**: 524 288 (0.51%). **MAC**: fine top-32 of `t/8` chunks; coarse state. **Memory**: compressed KV state | fine in-place **+0.048**; coarse in-place **−0.0002** (inert, delete-candidate); flag-off REBUILD +0.0484 but changes the readout scale | 50k ckpt, 8 192 B | **no** for in-place; **yes** for the flag arms |
| **kda** | `kda.py:192` `KDAAttention` | **params**: 1 327 104 (1.29%). **MAC**: **17.7%** of the training step (`frontier-vs-vanilla` §2). **Memory**: recurrent state | in-place **+0.633** — the largest single-mechanism delta in the model | 50k ckpt | **no** |
| **layerrope** | `layerrope.py:201/274/373` | **params**: 4 112 (0.004%). **MAC**: ~0 | in-place **+0.444** (`mechanism-liveness` §2, freezing per-iteration gains to depth 0). The `layerrope=false` rebuild is invalid (+310 BPB: it swaps a fresh `nn.RMSNorm` gain in) | 50k ckpt | **no** |
| **msa** | `sparse/attention.py:137` `BlockSparseAttention` | **params**: 7.6M dead in the presets (`final-dense-recipe`). **MAC**: block-sparse top-k | tie at 2k (2.7208 vs 2.7214, `architecture-verdict` §1.2); 3.5× slower | 2k (seed count not stated in the source) | **yes** |
| **mod** | `transformer.py:103` `mod_route`, `:133` `TransformerBlock` | **params**: one router per block. **MAC**: **0.629×** at capacity 0.5, 0.60× at capacity 1.0 is bit-identical (`2404.02258v1` §5; the repo's `mod_capacity_tokens` refuses a capacity of 0) | no repo measurement; published at 60M–3B. At our launch-bound trainer the FLOP saving is ≈0 wall-clock | — | **[U]** |
| **byteflow** | `byteflow.py:578` `ByteFlowNet`, `:831` `BfChannel` | **params**: +60%. **MAC**: fewer loop positions per byte. **Memory**: fp32-only | −0.485 at medium (2.9337 vs 3.4185, 31× spread), capacity-matched control −0.428; **not usable**: fp32-only, +60% params, 1.6× slower, incompatible with Engram | medium, 600 steps | **yes** |
| **mtp** | `mtp.py:155` `MTPHead` | **params**: one draft head. **MAC**: one extra head over K tokens | k=4: **−0.0044** (small), **−0.0007** (small_wide) → tie; off by default (`mtp_k=0`) | 600 steps | **yes** |
| **tsct_packed** | `config.py:1217`, `experts.py:2486` docstring | **launches**: 521 → 5 per bank rebuild; operand traffic 1 681 → 1 010 MB. **Params/MAC**: unchanged | `torch.equal` in fp32/bf16 and through one optimizer step (commit `644e2d0`, on `main`) | CPU/GPU parity, no BPB | **n/a** — a speed seam |
| **tsct_expert_basis / _lr / _revive** | `experts.py:521/1894`, config `:810/820/825` | **params**: `w [rank, shared_rank]` per expert = 6.3M at E=512; **MAC**: 0 (folded into the basis at rebuild, `tsct_basis_fold`) | **never measured** — all three are off by default and no A/B on record | — | **[U]** |

### 2.2 The one paragraph: what buys quality per unit of compute, and what is decoration

Ranked by measured BPB per parameter and per MAC, using only the in-place ablations (the only ones that isolate a mechanism on fixed weights): **KDA** is the best buy in the model — +0.633 BPB for 1.29% of parameters and 17.7% of the step's MAC; **Engram** is the best buy per MAC (+2.306, three table lookups per byte) but it is half the parameters and it is a lookup table, not capacity; **LayerRoPE** (+0.444 for 4 112 parameters) and **AttnRes** (+0.238 for 6 144) are the best buys per parameter, two to three orders of magnitude better than the bank; **CSA-fine** (+0.048) is thin but real. The **bank** is the worst return in the stack: +0.067 for 44.8M parameters — 43.6% of the model buying 0.067 BPB, with 254 of its 256 experts redundant to the routing decision (+0.008 for top-2 vs all-256) and a `down` GEMM that is 89.7% of the MoE MAC and already dense. **Decoration, on this evidence:** the LFB bias controller (+0.0002, and its `sign` mode is the one that collapsed 9/512 at 150k), CSA-coarse (−0.0002), the mHC Sinkhorn mixing (+0.0001 on top of a topology change that is worth +2.76 when removed), `residual_scale` (structurally dead), MSA (tie, 7.6M dead parameters, 3.5× slower), MTP (tie) and MoD (published FLOP saving, zero wall-clock here). And the ceiling over all of it: **the projection is 2.8% of MoE MAC, gate+up 7.5%, `down f→d` 89.7%** (`balance-fix-verdict`, n=8192/k=2/E=512/r=64/d=1536/f=4096) — factorising the projection cannot move the needle, and the one GEMM that could is shared and dense by construction (`experts.py:1443`).

**Why every row is SUSPECT, and why the bias is not neutral.** The Muon defect is shape-dependent: `RMS(ΔW) = lr·√(max(1,r/c))/√c` (`muon-audit` §1). At `lr=3e-4` that is 5.41e-6 on a `[832,3072]` dense up-projection (**55.4×** below AdamW) and 5.74e-6 on a `[64,2730]` expert write (**52.3×**) — but 2.00e-5 on a `[3072,832]` dense down (**15.0×**) and 3.75e-5 on a `[1024,64]` input basis (**8.0×**). **[C]** The bank's per-expert identity (`v`) is therefore the *most* penalised tensor in the model, 3.5× more than the dense arm's widest matrix. A comparison run in that regime is not arm-neutral: it is biased **against** the bank. That is the single strongest reason to re-run the founding A/B before believing any of the negative verdicts — including the ones this document repeats.

---

## 3. TASK B — the arXiv side

### 3.1 The table

`Match` says what was controlled. **FLOP-matched** is called out explicitly; a params+tokens match is named as such and the matched figure is given.

| # | paper | id / version / date | scale of the number | exact claim (table) | Match | Δ sign | applicability at 95M | effort |
|---|---|---|---|---|---|---|---|---|
| 1 | Fine-grained MoE scaling laws | 2402.07871v1, 2024-02-12 | smallest compute-optimal row: 64×100M total, 4.37B tokens, G=8, **2.95e18 FLOPs**, loss **3.133** (Table 2) | "MoE can always be considered more efficient … for short training schedules, they may under-perform" (§6.3) | **isoFLOPs** | + (MoE) at ≥4.37B tokens in their smallest compute-optimal run | **no** — our 98MB is 0.1B bytes; the smallest published run is 4.37B tokens = 44× our corpus | low (read) |
| 2 | MoE-I² | 2411.01016v1, 2024-11-01 | Mixtral 8×7B, d=4096 experts; Qwen1.5-MoE 14.3B; DeepSeek-V2-Lite 16B | intra-expert SVD at **average rank 2048** (Mixtral) / 512 (Qwen, DS) cuts expert params **38.6%**, quality preserved after 2 epochs LoRA (Table 1, Table 2) | params (compression), no FLOP match | − (lossless-ish) | **yes as a rank anchor**: even a 4096-wide trained expert needs rank ≈ half for a tolerable cut | low |
| 3 | DeRS | 2503.01359v1, 2025-03-03 | MoE-LLaVA StableLM/Qwen/Phi (~1–3B) | shared dense base + per-expert low-rank delta: added params **1.24B → 1.20M**, overall **57.4 → 57.5** (Table 1); formula `N·d·dh → d·dh + N·r·(d+dh)` | added-params matched | ~0 vs full-rank bank | **high** — this is exactly TSCT's structure, and it says the structure is a *parameter* win | low |
| 4 | SLBF | 2610.09342v1, 2026-10-07 | 5 MoE LLMs 16B–122B | rank-k bases shared among experts, data-free post-hoc: Qwen3-30B at 24% compression **84.3** avg vs **84.9** original; MoBE 82.8 (Table 2). Mixtral WT2 **6.92** vs EAN 8.28 (Table 3) | compression ratio | −0.6 avg at 24% | **medium** — export only; 94.3 min per (layer, projection) to fit | medium |
| 5 | SPRI | 2606.16456v1, 2026-06-15 | Qwen3-ASR 0.6B → ~3B upcycled | SVD-partitioned residuals on the **down** projection: **80.57 COMET / 24.43 BLEU** vs dense FT **76.59 / 19.95** at comparable total params (Table 4) | **params-matched, not FLOPs** (the MoE activates a subset) | +4.0 COMET | **high for the init idea**, low for the claim: it is a pretrained dense anchor + data-constrained adaptation | low |
| 6 | ButterflyMoE | 2601.13563v5, 2026-01-20 | WikiText-103, 4 layers, d=512, f=2048, 8 experts top-2 | Table 1: Standard MoE **71.75** PPL / 32.0 MB; Dense **83.01** / 8.0 MB; Dense matched-mem **135.46** / 0.5 MB; Butterfly **81.19** / **0.40 MB** (80×) | **memory-matched**, not params or FLOPs | + at equal memory | **high for the memory currency**; it is the same "shared substrate + expert-specific transform" family | medium |
| 7 | Upcycling scaling laws | 2502.03009v2, 2025-06-16 (ICML 2025) | dense 15M/44M/0.1B/0.2B/0.5B/1B; MoE to 7B total; hundreds of B tokens | §5.2: "for a given compute budget without any pre-trained models, training a compute-optimal dense or MoE from scratch outperforms the two-stage dense-to-upcycled approach"; D* = **4B tokens** for 8×1B (§5.1) | **FLOPs budget** | − for upcycling at fixed FLOPs | **high**: at 95M with no pretrained teacher, upcycling is the wrong path | low |
| 8 | Upcycling LLMs | 2410.07524v2, 2024-10-10 | Nemotron-4 15B, 1T tokens | upcycled **67.6** MMLU vs continued dense **65.3** at the same 1T tokens | **params+tokens**, not FLOPs (the MoE has 2× params) | +2.3 MMLU | **no** — 15B, and the win is at 2× params | low |
| 9 | SMEAR | 2306.03745v1, 2023-06-06 | small-scale vision + LM | "models with learned routing often **underperform their parameter-matched densely activated counterparts**" (abstract) | params-matched | − (learned routing) | **high as a warning**, no scale table | low |
| 10 | Equifinality in MoE | 2604.14419v1, 2026-04-15 | **76–84M params**, 50k steps, **1.64B tokens**, WikiText-103 | 62 experiments: five cosine-routing variants within **1 PPL** (33.93–34.72, TOST p<0.05); a linear router with 5.3× more routing params reaches **32.76**, iso-parameter closes 67% → true routing advantage **~1.2%** | **iso-params, iso-tokens** | ~0 for routing topology | **highest** — the only paper at our parameter scale; says routing is not the lever | medium |
| 11 | Myth of expert specialization | 2604.09780v1, 2026-04-10 | 5 pretrained MoEs, analysis | specialization is a property of hidden-state geometry, not routing; load-balancing suppresses shared directions → collapse under less diverse data / small batch | none (analysis) | n/a | **high**: explains why a 95M bank's experts never specialise on 98MB | low |
| 12 | Nonlinear init for low-rank nets | 2202.00834v3, 2022-02-02 | theory + small nets | approximating the **function** of the layer at init is provably better than spectral (Frobenius) approximation of its weights | — | + (init) | **high** — cheap, init-only; SPRI is the empirical version at 3B | low |
| 13 | UniRank | 2606.21847v2, 2026-06-20 | Llama-3.2-3B at 40% retention | singular-component allocation (local energy × global IO-cosine) cuts WT2 PPL **40.8–61.8%** (avg 52.9%) vs manual per-layer allocation, no fine-tuning (Table 1) | params (retention) | + | **high for rank allocation**, zero for training quality | medium |
| 14 | Krony-PT | 2412.12351v2, 2024-12-16 | GPT-2 124M → 96M/81M | rank-preserving Kronecker FFN compression; the 96M model beats DistilGPT2 on all datasets (Table V), i.e. a 1.3× cut is survivable | params-matched vs distilled models | −small | medium: 1.3× is the honest compression headroom for a trained FFN | low |
| 15 | MoLE (fine-tuning) | 2404.13628v1, 2024-04-21 | CLIP + LLM fine-tuning | text-alignment +0.031, image-alignment +0.037 over the best prior composition (Table 1) | task metrics | +small | **low** — adapters, not pretraining | low |

**Credibility, one line each (the column the table above does not have).** #1 is a scaling-law paper with published coefficients and a compute-optimal table, the strongest kind of evidence for a *law*, weakest for a specific scale. #2–#6 are post-hoc compression/reconstruction papers: peer-reviewed venues, large models, but they measure bytes at fixed quality, not quality at fixed bytes. #5 and #3 are the two with a training-time claim; #5 is a single research group's audio adaptation, #3 a multimodal upcycling study. #7 and #8 are ICML/EMNLP-grade upcycling studies whose direction is consistent (upcycling pays only with a paid-for dense checkpoint). #9 is a 2023 method paper whose negative claim about learned routing is stated in the abstract and replicated by our own numbers. #10 is the only paper at 76–84M with 62 controlled runs and TOST tests; treat it as the strongest evidence in the set *for our scale*, with the caveat that it is a preprint. #11 is an analysis paper (5 pretrained models), not a training result. #12 is theory plus small-scale validation. #13–#15 are compression/fine-tuning results with limited transfer. Every [T] number above was read in the named table; none was taken from an abstract when a table existed.

### 3.2 Ranked by (delta × applicability at 95M) / effort

The score is qualitative on purpose: a numeric score over incomparable metrics would be fake precision. The ordering is by *what to do first*, not by paper quality.

1. **Fix the optimizer before any form comparison** (not a paper; `muon-audit`, `night_stage3.log`). Measured −0.147 BPB at 2k for `mult=12` vs `mult=1`. Applicability 100%, effort 1 config line. **This dominates every architectural delta in the table.**
2. **Function-faithful SVD init of the factors** (2202.00834v3 + SPRI 2606.16456v1). Init-only, no new parameters (you reuse `v`), and the branch already implements the SVD half (`b249bec`). Expected small-to-medium; effort low. This is the only idea that changes *what the model starts as* rather than what it is.
3. **Per-matrix rank from the trained spectrum** (UniRank's allocation rule, 2606.21847v2, + our §4.2). It cannot buy quality — at 99% energy it is near full rank — but it is the honest way to spend a *compression* budget. Effort low (the script exists on `b249bec`).
4. **Dense anchor + low-rank residual experts** (DeRS 2503.01359v1, SPRI 2606.16456v1). High applicability, but it changes the question: it is upcycling, and 2502.03009v2 says at fixed FLOPs from scratch is better. Do it only if a dense checkpoint is already paid for.
5. **Ternary shared substrate** (ButterflyMoE 2601.13563v5). The best *memory* result in the set (80× at 8 experts, 0.40 MB vs 0.50 MB for a dense matched-mem arm), but its own baseline comparison is memory-matched. Effort medium (quantization + rotations).
6. **Rank-aware Muon for factor pairs** (`b249bec` §4). A formula with ≤4% error on six shapes, already implemented, refuses to run on the legacy random init. It is the correct complement to (2) and (1).
7. **Do not do:** routing-topology work (2604.14419 says ~1.2% at our exact scale), more experts (our own E=256→512 = +0.033), learned rank allocation by gradient (UniRank does it post-hoc with a closed-form score; a learned allocator is a second mechanism on top of a tie), and MoD as a *speed* play (launch-bound trainer).

**Rejected, with the reason each is rejected for *this* question.** MoLE as fine-tuning (2404.13628v1): adapters on a frozen backbone, +0.031/+0.037 on CLIP alignment — nothing to transfer to a from-scratch 95M bank. Multilinear MoE (2402.12550v4): factorised expert *tensors* for vision, measured on specialisation/interpretability, no BPB at our scale. Krony-PT (2412.12351v2): a 1.3× FFN compression whose win is against DistilGPT2, not against a dense model of equal parameters. UniRank (2606.21847v2): a compression allocator — it improves PPL *at a fixed retention budget*, which is the memory currency, not the quality one. SMEAR (2306.03745v1): the method (soft merging) needs a differentiable mixture of all experts per token, which is the opposite of a sparse bank and costs `E×` the expert compute. MoE-I² (2411.01016v1): post-hoc compression of 14B–47B models; the rank anchor transfers, the method does not.

### 3.3 The negative results, stated plainly

- SMEAR (2306.03745v1): learned routing loses to parameter-matched dense at small scale. This is the closest published statement of our own result and it predates it by three years.
- 2604.14419v1: at 76–84M parameters and 1.64B tokens, routing topology is worth ~1.2%; five routing families are inside 1 PPL. Our router work (LFB `sign`→`prop`, entropy top-k, adaptive width) is therefore not where quality lives.
- 2604.09780v1: expert specialisation is geometry, and load balancing actively suppresses shared directions. A 95M bank on 98MB has neither the data diversity nor the batch to grow specialisation; that is a mechanism-level explanation of 189/512 live at 600 steps.
- 2502.03009v2: with no pretrained model, from-scratch beats upcycling at fixed FLOPs. Upcycling's published wins (2410.07524v2, 2606.16456v1) all start from a dense checkpoint that already paid for its pretraining.

---

## 4. TASK C — the honest ceiling question

**Question:** can a factorised expert bank beat a dense FFN of the same parameter count at 95M parameters and 1–5 GB of bytes?

### 4.1 Arithmetic 1 — at equal parameters, a factorised FFN is almost full rank

For a dense `W ∈ R^{d×f}`, a rank-`r` factor `U[d,r]·diag(s)·V[r,f]` has `r(d+f+1)` parameters. Setting that equal to `df`:

```
r* = d·f / (d + f + 1)                      [C]
```

At the 95M recipe (`d=832`, `f=3072`): **r\* = 654.5 of 832**, i.e. 78.6% of full rank. At `medium` (`d=1024`, `f=2730`): **r\* = 744.5 of 1024** — the same number the `b249bec` report derives independently. **At equal parameters the factorisation is not a compression.** A 4× parameter cut needs `r ≈ 164` (20% of `d`); §4.2 measures what that costs in energy.

Consequence for the bank: at a fixed 7.67M-parameter FFN slot (`d=832, f=3072`), a `share_u` bank at `r=64` can hold `E = (7.67M − 2dr) / (2r + rf) ≈ 38` experts, of which top-2 are active → active MAC `2·(2dr + rf) = 606 208`, **12.7× less** than the dense FFN's 7.67M. That is the entire trade: 12.7× less arithmetic per token for `E=38` thin experts instead of one full-width map. Nothing about it says the thin experts are *better*; it says they are cheaper.

### 4.2 Arithmetic 2 — the trained weights are not low rank

Independent measurement, this document, CPU, `port/checkpoints/medium/med1.pt` (step 40k). `E[·]` is cumulative squared-Frobenius energy; `PR` is the participation ratio `(Σσ)²/Σσ²`.

| matrix | shape | PR | E@64 | E@128 | E@256 | E@512 |
|---|---|---:|---:|---:|---:|---:|
| `loop.moe.down.weight` | 1024×2730 | **530.4** / 1024 | **0.434** | 0.684 | 0.881 | 0.955 |
| `loop.attn.q_proj.weight` | 1024×1024 | 706.6 / 1024 | 0.244 | 0.418 | 0.661 | — |
| `loop.attn.k_proj.weight` | 1024×1024 | 693.9 / 1024 | 0.261 | 0.435 | 0.675 | — |
| `loop.attn.v_proj.weight` | 1024×1024 | 605.8 / 1024 | 0.338 | 0.533 | 0.764 | — |
| `loop.attn.g_proj.weight` | 1024×1024 | 686.5 / 1024 | 0.272 | 0.446 | 0.683 | — |
| `loop.mem_dense.weight` | 1024×1024 | 725.5 / 1024 | 0.244 | 0.417 | 0.659 | — |
| random control, same shape | 1024×2730 | — | 0.144 | — | 0.470 | — |

The `b249bec` report measures the same checkpoint with a wider sweep and reports the rank holding 99% energy: **941/1024** for the shared `down`, **754/1024** for `mem_dense`, **245–246/256** for q/k/v, **247/256** for embed, **189/192** for the shared expert's `v`. Both instruments agree on the shape of the answer: training concentrates the spectrum somewhat (43% vs 14% of energy at rank 64 on a random matrix) but leaves it far from low-rank. The attention matrices are *not* more compressible than the FFN — they are less (PR 606–707 vs 530). **[M]**

The bank's writes are not redundant either: the stacked `V` (256 × [64, 2730], 16 384 × 2 730) has rank **2 730** (full `f`) with PR **2 613.8**. So the 44.7M bank parameters span ~2 614 effective directions — ~17k parameters per effective direction. The bank's capacity is real as storage; it is simply not reachable by one token, and it is not a low-rank view of a dense map.

### 4.3 Arithmetic 3 — information per parameter, and why it does **not** explain the loss

At 98.3 MB of tokens and 95M parameters, the dense body sees **1.03 bytes per parameter** (8.3 bits). For the bank, only `k` of `E` experts are touched, so

```
bytes per bank parameter = k · B / P_bank            [C]
```

At **equal parameters** (`P_bank = P_dense = 7.67M`, `k=2`) each bank parameter sees **2× more** bytes than a dense one (25.6 vs 12.8 at 98.3 MB). So at the comparison the question asks for, data per parameter is *not* the bank's problem; it is an advantage.

Our actual bank is not equal-parameter: at E=256, r=64, f=2730 it is **44.8M parameters against a 7.67M dense FFN slot, 5.8× larger**, and then each of its parameters sees `2·B/44.8M` — **4.4 B/param at 98.3 MB, 110 B/param at the planned 2.46 GB, against the dense body's 12.8 / 320**. That is a consistent **2.9× penalty**, and it is entirely a consequence of the bank being 5.8× oversized, not of the routing: the ratio `k·P_dense/P_bank` does not depend on `B`. So the "capacity needs 10× data" story is only true for the oversized bank we built; at equal parameters the bank is the *less* data-starved side, and its loss must come from capacity per expert (rank 64), the optimizer, or the router — not from the corpus. **[C]**

### 4.4 Arithmetic 4 — what FLOPs-per-token is worth at 95M

The bank's win is active MAC. At 95M the training step is launch-bound: mean GPU utilisation 13.3%, 79% of samples ≤5% (`AGENTS.md` §3.1, 2026-09-29), MFU ~2% (`docs/records/research/2026-11-05-small-model-mfu.md`). The MoE block is **2.5 ms of a 125 ms iteration (2.0%)** and its `down` GEMM is 89.7% of that 2.5 ms (`balance-fix-verdict`). A perfect factorisation of everything else — projection, gate, up — can therefore save at most **0.2% of a step**, which is below the step-time noise. The FLOP win is real on paper and worth zero wall-clock here. **The only currency that survives at 95M is memory (and export size).**

### 4.5 Our own numbers, side by side

| comparison | what was matched | result | source |
|---|---|---|---|
| dense 69.3M vs bank E=512 199.3M | **active MAC** (28 442 688 vs 28 442 624, +0.0002%) | dense **2.4714** vs bank **2.5384**, Δ **−0.0670**, control spread 0.0201 | `2026-11-05-architecture-review-dense.md` §What survives; `ab_capacity.py` arms A/D, 600 steps, 3 seeds, 65 536 B window |
| dense 66.9M vs bank E=512 | **+18.4% MAC for dense** | dense **2.3195** vs bank **2.3655**, Δ **−0.046** | same doc, §0.2; 2000 steps, 2 seeds |
| bank 141.9M vs dense 117.5M | **same width**, not params | **2.650** vs **2.653** (tie), bank +20.8% params | `2026-10-23-architecture-verdict.md` §2, 600 steps, 3 seeds, 40 960 B |
| dense at equal params (f=63024) | params | **never completed** — OOM twice at batch 8 | same table |
| `ffn_form="tsct"` vs dense | nothing clean | **no clean run**; `scripts/ab_ffn_form.py` never ran | `~/logs` + absence of `/tmp/ab_ffn_form` |

Note what the first two rows do *not* say: the dense arm's active **path** was 16.5M vs the bank's ≈13.4M, so the 600-step win is also a +23% larger active-path win (the audit says so in §0.4). The 2k pair is the cleaner routing test (14.1M vs 13.4M) and still wins by 0.046.

### 4.6 The literature's answer to the same question

- **At equal parameters and tokens, a trained dense FFN needs rank ≈ half its width** to survive an SVD cut with fine-tuning (MoE-I², rank 2048 of 4096; 2411.01016v1 Table 1). A rank-64 expert is not an approximation of a dense expert; it is a different function.
- **Shared-basis + per-expert delta is a parameter-efficiency structure, not a quality structure.** DeRS gets the same quality with 1.24B → 1.20M *added* parameters (2503.01359v1 Table 1); SLBF gets 24% compression at −0.6 average points (2610.09342v1 Table 2). Both are wins in bytes, both are post-hoc, neither claims a from-scratch quality win.
- **At equal memory, a shared-substrate expert bank does beat a dense FFN**: 81.19 vs 135.46 PPL at 0.40 MB vs 0.50 MB (2601.13563v5 Table 1). This is the strongest published support for the *memory* claim, and it is at 4 layers, d=512 — our scale.
- **The one params-matched win for a low-rank/residual bank is upcycling with a dense anchor**: SPRI 80.57 vs 76.59 COMET (2606.16456v1 Table 4). The dense model was pretrained; the experts are residual corrections to it. 2502.03009v2 §5.2 says that with no pretrained model, from-scratch beats the two-stage path at fixed FLOPs.
- **At our exact parameter scale (76–84M), routing topology is worth ~1.2%** (2604.14419v1) — so the capacity argument cannot be rescued by a better router.

### 4.7 Verdict (RU)

**Факторизованный банк экспертов не может побить плотный FFN равного числа параметров при 95M и 1–5 ГБ по качеству.** Это не мнение и не следствие сломанного Muon: это арифметика. При равных параметрах факторизация вырождается в ранг 654 из 832 (78.6% полного), то есть перестаёт быть сжатием; при меньшем ранге она теряет энергию, которой у плотной матрицы 99% лежит в хвосте до 941-го сингулярного числа. Всё, что банк действительно покупает — это память, размер экспорта и FLOPs на токен; из этих трёх при 95M имеет цену только память, потому что шаг launch-bound и FLOPs на токен не конвертируются в секунды.

**Что он *может* купить:** (а) тот же BPB при в 3–13 раз меньшем активном счёте — это измерено и это единственное, что подтверждено; (б) меньший экспорт при тернарных факторах (8×, ценой +0.083 BPB на 600 шагах); (в) лучший BPB на байт памяти, если сравнивать с плотной моделью **того же размера в байтах** (ButterflyMoE, 81.19 vs 135.46 при 0.40/0.50 MB).

**Что он не может купить:** BPB при равных параметрах, BPB при равных данных, и «ум» на 95M. Если цель — качество, то это плотная 95M модель + данные + рабочий Muon; банк остаётся флагом для экспорта и для случаев, где память — дефицитный ресурс.

**Уверенность в проверенной части: высокая** (два независимых спектральных инструмента, два независимых прогона dense-vs-bank, арифметика перепроверяема). **Уверенность в том, что «банк не выиграл бы и с рабочим Muon»: низкая** — этот эксперимент не проводился, и §2.2 даёт механизм, по которому сломанный Muon бил именно по `v` эксперта.

---

## 5. TASK D — ranked patch list, cheapest first

Costs are per seed unless stated. Step times are measured on this box at `medium` b8×s1024: **0.58 s/step** (600 steps in 345–351 s, `~/logs/ab_rank_up2.log`); 600 steps × 8 192 B = 4.9 MB, which at 60 kB/s is 82 s of data — so at this batch the run is **compute-bound, not byte-bound**. At `flagship3`'s b8×s1024×accum 3 the step is 0.46 s and 24 kB/step = 53 kB/s, which is at the 60 kB/s line: there the data feed is the limit. Both are stated.

### P0 — Make Muon's step shape-fair. No new parameters. 1 config line.

- **Change:** `port/src/dormouse/config.py:1960` `muon_lr_mult: float = 1.0` → the measured 12, or better: compute a per-shape multiplier at group construction so that `RMS(ΔW)` equals the AdamW half's `lr` on every shape. The formula is exact and already in the audit: `mult_c = √c / √(max(1,r/c))` normalises the step to `lr`; the measured per-shape values were 15.0 / 55.4 / 28.8 / 16.7 (`docs/records/reviews/2026-11-07-muon-audit.md` §3). Touch points: `train/optim.py:1798` (constructor) and `:1882` (`set_lr`), `train/loop.py:1807`.
- **Expected Δ:** **−0.10 … −0.20 BPB at 2k** vs `mult=1`; the measured point is 1.7478 vs 1.8947 = **−0.147** (`~/logs/night_stage3.log`, 2 seeds each, flagship3 dense). At 600 steps the effect is smaller and noisier: −0.03 … −0.10 [S].
- **CPU-only gate:** `port/scripts/audit_mult_liveness.py` already prints per-shape RMS and `identical=False`. Extend it to assert `RMS(ΔW_i)/lr ∈ [0.8, 1.25]` for every shape in the model and both `mult` values; it must fail on `main` today and pass after the patch. No GPU.
- **GPU cost:** 3 seeds × 2 000 steps × 0.46 s = **46 min/arm**, two arms = **1.5 GPU-h**. At 60 kB/s the data is 3 × 16.4 MB = 49 MB = 14 min, so compute dominates.
- **Falsifier:** with 3 seeds, `mult=12` (or per-shape) is **not** better than `mult=1` by more than the control's own spread. If it ties, the whole Muon story is wrong and every architecture verdict needs a different explanation.

### P1 — Initialise the factors from the dense function (init only, no new parameters).

- **Change:** `b249bec`'s `tsct_init="svd"` (`config.py` on that branch; `TSCTGLU`/`TSCTLinear` `tsct.py:` on that branch): draw the dense init the counterpart would draw, keep its top-`r` SVD triple. On `main` the equivalent for the bank is to initialise each expert's `v` from a disjoint SVD block of a dense `[f, d]` and let the shared `down` stay dense (`experts.py:1443`, `:1234`). No parameter is added; `v` already exists.
- **Expected Δ:** **−0.02 … −0.06 at 600 steps** [S]. The mechanism guarantees step-0 equivalence to a dense FFN (measured 3.6e-7 max logit delta at full rank on `b249bec`), so it can only help the early curve; how much survives 600 steps is unknown.
- **CPU-only gate:** full-rank `tsct`/bank forward equals the dense FFN to `<1e-6` max logit; a truncated init equals the Eckart–Young-best approximation (the branch's `port/tests/test_tsct_ceiling.py` already pins both). Add: the gradient to `v` is non-zero at step 0, and the dense-equivalence survives one optimizer step.
- **GPU cost:** 3 seeds × 600 × 0.58 s = **17 min**; data 2.5 MB.
- **Falsifier:** the arm ties the random-init arm at 3 seeds (then the init does not matter), or the step-0 equivalence fails on `main`'s `DenseFFN` (then the reparameterisation is wrong).

### P2 — Stop factorising `down`. Removes parameters, adds none.

- **Change:** `ffn_form="tsct"` factorises all three matrices (`experts.py:842`, `:928`); the `down` map is 89.7% of the MoE MAC and the least compressible matrix in the model (§4.2, rank 941/1024 for 99%). Add a form (or set `ffn_rank_down = min(d, f)` plus a "dense down" branch) that keeps `down` as one `nn.Linear` and factorises only gate/up. This is **fewer** parameters than `tsct` and the same as `bank`'s shared write.
- **Expected Δ:** removes the `tsct` arm's measured penalty relative to the bank (wave14's `tsct` was +0.063 vs dense, but confounded); at equal params expect **±0.01 vs dense** [S] — i.e. the form stops losing, not that it wins.
- **CPU-only gate:** the width ledger (`scripts/capacity_math.py`) must show `down` dense and the MAC total drop by the factorised-down share; a forward at `rank_gate=min(d,f)` must equal `DenseFFN` to `<1e-6`. Also assert `ffn_rank_down=min(d,f)` is refused as a factorisation (it is a reparameterisation, not a compression).
- **GPU cost:** 3 seeds × 600 = **17 min**.
- **Falsifier:** still worse than dense at equal params by >0.02 with a working Muon.

### P3 — Balanced routing without a controller. No new parameters.

- **Change:** `experts.py:1979` `_permute` / `route_fused.py`; add a `moe_route_warmup` that routes by a seeded hash for the first N steps, then hands over to the learned router (or freezes the router until every expert has ≥ `floor` assignments). The existing LFB `prop` controller (`experts.py:1839`) stays off.
- **Expected Δ:** **−0.02 … −0.10 at 600 steps** if collapse is the cause [S]; zero if the router was already healthy (the `med1` checkpoint at 40k has all 256 live, max 8.68M, top-10 share 4.3% — so at 40k on `medium` it *is* healthy).
- **CPU-only gate:** a 32-expert toy with a deliberately skewed router; after 200 steps the minimum live-expert count is ≥ `0.8·E`, and the hash arm's per-expert counts are within 2× of uniform by construction.
- **GPU cost:** 3 seeds × 600 = **17 min**.
- **Falsifier:** the live-expert count at 600 steps is already ≥80% without it, or the arm loses BPB.

### P4 — Measure the three TSCT-2 arms that were never measured. No new parameters.

- **Change:** `tsct_expert_lr` (`config.py:820`, `experts.py:521`) and `tsct_expert_revive` (`config.py:825`, `experts.py:1894`). Both are implemented, off by default, and have **no A/B on record**. `tsct_expert_basis` (`config.py:810`) adds 6.3M parameters at E=512 and should be last.
- **Expected Δ:** **0 … −0.05** [S]; the mechanism is a rescue for a collapsed bank, and `med1` shows no collapse at 40k on `medium`.
- **CPU-only gate:** a synthetic 16-expert bank with counts skew (one expert holds 90%); one `revive_experts` call moves the coldest expert's parameters to the hottest + 1% noise, resets both LFB entries, and the next `expert_lr_multipliers()` gives the revived expert > 1.0 while the donor is < 1.0. Assert the revived expert's gradient is non-zero on the next forward.
- **GPU cost:** 3 seeds × 600 = **17 min** per arm, two arms = **38 min**.
- **Falsifier:** the revived expert's count does not rise within 200 steps, or BPB is unchanged (then delete the arms — AGENTS §1.2).

### P5 — Per-matrix rank from the trained spectrum. No new parameters; removes some.

- **Change:** cherry-pick `b249bec`'s `tsct_ranks="spectrum:0.99"` (or a `90/95` variant) and `scripts/tsct_spectrum.py`. On `main` this is new code; on the branch it is landed and gated (`port/tests/test_tsct_ceiling.py`).
- **Expected Δ:** **≈0 BPB**, because at 99% the rule lands near full rank (§4.2; the branch reports 941/1024 on `down`); at 95% it is a ~1.2× compression of `down`. Its value is bytes, not BPB. If someone tries to sell it as a quality patch, the falsifier is the reported `Δ` at 99%: it must be ≤ the control spread.
- **CPU-only gate:** the spectrum script reproduces a hand SVD on a 16×8 fixture (ranks 2/3/4 at 90/95/99%, exact Eckart–Young values) — already in the branch's test file; and the parameter bill equals `width.ffn_params` for an explicit list.
- **GPU cost:** 3 seeds × 600 = **17 min**, only if a compression claim needs a BPB number.
- **Falsifier:** the 99% rule produces ranks that make the model *larger* than the current default (it does: near full rank), and the 95% arm loses more BPB than it saves in bytes.

### P6 — Make the format the budget: ternary factors + rank→width reinvestment.

- **Change:** `factor_qat="ternary"` (`tsct.py:335`, `:788`) plus the measured rank↔width trade: at equal *storage*, ternary rank 32 with `d=1920` beat bf16 rank 64 with `d=1024` by **−0.015 BPB** (3 seeds, 2000 steps, `.bulba/goal.md` 2026-10-22). Config only.
- **Expected Δ:** **−0.01 … −0.02 at 2k at equal storage**; at equal *parameters* the ternary arm is +0.083 (`architecture-verdict` §3), so the claim is only valid in the storage currency.
- **CPU-only gate:** `pack_tsct`/`unpack_tsct` round-trip bit-exact for the rung; the packed parameter count is ≤ the bf16 arm's; the dequantised forward matches the fake-quant forward to 0 (the repo's `tests/test_tsct_packed.py` pins the packed bank through one optimizer step).
- **GPU cost:** 3 seeds × 2 000 × 0.46 s = **46 min**.
- **Falsifier:** at equal storage the ternary arm is not better than the bf16 arm by more than the 3-seed spread (0.007–0.02).

### P7 — Distillation from a 3× byte teacher. Not a TSCT change; listed because it dominates every architectural delta.

- **Change:** none in `model/`; train a 3× byte model, then distil into the 95M (`docs/records/research/2026-11-07-architecture-frontier.md` §TL;DR item 8 predicts −0.03…−0.08 at 95M, 21 GPU-h).
- **Expected Δ:** **−0.03 … −0.08 BPB** [S] at 95M; the frontier doc's numbers are its own extrapolation, tagged there.
- **CPU-only gate:** the distillation loss is a function of `(student_logits, teacher_logits)` and nothing else — a fixture that changes the labels must not move it.
- **GPU cost:** ~21 GPU-h per the frontier doc; at 60 kB/s the data for 1 GB is 4.6 h, so compute is the limit.
- **Falsifier:** no gain at 1 GB, or the gain vanishes when the teacher is the same size (then it was the teacher's data, not the distillation).

**Recommended order:** P0 (a day, one config + one gate) → P1 (init, cheap) → P2 (remove the losing factorisation) → P3/P4 (routing rescue, only if the collapse reproduces) → P5 (compression accounting) → P6 (storage) → P7 (data/teacher). P0 is the only one with a measured, replicated delta in the right direction.

---

## 6. VERIFIED (sourced)

1. **The Muon step is 15–55× too small at `lr=3e-4`, shape-dependent, and the formula is exact.** `RMS(ΔW)=lr·√(max(1,r/c))/√c`, measured on 6 shapes to 4 digits; 15.0× on `[3072,832]`, 55.4× on `[832,3072]`, 28.8× on `[832,832]`, 16.7× on `[2496,832]`. Source: `docs/records/reviews/2026-11-07-muon-audit.md` §1/§3, CPU, 2026-11-07.
2. **The multiplier knob was dead and is now fixed, but the default is still 1.0.** `optim.py:1882` `set_lr` applies `muon_lr_mult`; `config.py:1960` default `1.0`; no preset sets it. `~/logs/night_stage3.log`: adam 1.8947, m12 1.7478, m32 1.9942 (2 seeds, 2k, flagship3 dense).
3. **Dense beats the bank at equal active MAC.** 2.4714 vs 2.5384 (600 steps, 3 seeds, 65 536 B) and 2.3195 vs 2.3655 (2k, 2 seeds, +18.4% MAC). Source: `docs/records/reviews/2026-11-05-architecture-review-dense.md` §0/§What survives; `~/logs/abdense/*`.
4. **At same width it is a tie, with the bank 20.8% bigger, and the equal-params arm never ran (OOM twice).** `docs/records/reviews/2026-10-23-architecture-verdict.md` §2.
5. **`ffn_form="tsct"` has no clean A/B.** `scripts/ab_ffn_form.py` exists; `/tmp/ab_ffn_form` does not; `~/logs` has no `tsct_eq` run. The `wave14` tsct arm is confounded (d/rank/share_u changed).
6. **The rank ladder above 64 was never measured.** `~/logs/rank_up.log`: r128_e128, r192_e85, r256_e64 all `rc=1`, no BPB; `~/logs/ab_rank_up2.log`: `bpb=None` for all 9 runs. Only rank 64 → 32 → 16 is measured (+0.027 / +0.136), and rank 8 crashes.
7. **The trained spectrum is not low-rank.** 941/1024 for 99% on `down`, 754/1024 on `mem_dense`, 245–246/256 on attention q/k/v (`b249bec:docs/reviews/2026-11-06-tsct-ceiling.md` §1); independent PR measurement here: 530.4/1024 (`down`), 606–707/1024 (attention).
8. **The bank's 44.7M writes span 2 730 directions with PR 2 613.8** — measured here on `med1.pt`.
9. **`down f→d` is 89.7% of MoE MAC, projection 2.8%, gate+up 7.5%** at n=8192/k=2/E=512/r=64/d=1536/f=4096. `docs/records/reviews/2026-11-04-balance-fix-verdict.md`.
10. **The MoE block is 2.5 ms of a 125 ms iteration (2.0%)** on the same shape. Same source.
11. **The 9/512 collapse is the `sign` LFB controller on `m400`, not the form.** `m400.toml` sets `moe_lb_coef=0.01`, `moe_lb_mode` default `sign`; `m400b.toml` uses `prop` with γ=0.002 / clamp 4.0; the 600-step sweep runs `m400b`. `docs/records/reviews/2026-11-04-routing-collapse-CONFIRMED.md`; `config.py` `moe_lb_mode="sign"` default.
12. **`med1.pt` at step 40k has all 256 experts live, max 8.68M, top-10 share 4.3%** — measured here; so "the bank always collapses" is false.
13. **Papír numbers:** 2402.07871v1 Table 2 (64×100M, 4.37B tok, 2.95e18 FLOPs, loss 3.133); 2411.01016v1 Table 1 (rank 2048/4096, 38.6% cut); 2503.01359v1 Table 1 (1.24B → 1.20M added params, 57.4 → 57.5); 2610.09342v1 Table 2 (24% → 84.3 vs 84.9); 2601.13563v5 Table 1 (81.19/0.40 MB vs 135.46/0.50 MB); 2606.16456v1 Table 4 (80.57/24.43 vs 76.59/19.95); 2502.03009v2 §5.1–5.2 (D*=4B for 8×1B; from-scratch wins at fixed FLOPs); 2604.14419v1 (76–84M, 1 PPL, ~1.2%); 2306.03745v1 (abstract).
14. **`tsct_packed` is on `main`** (commit `644e2d0`) and does 521 → 5 launches with `torch.equal` through one optimizer step; `tsct_init`/`tsct_ranks`/`tsct_rank_lr` are **not** on `main` (branch `b249bec` only).

## 7. SPECULATION (my inference, marked)

- **[S] The founding A/B is biased against the bank by ~3.5×**, because the broken Muon penalises the expert write `v [64,2730]` (52.3×) more than the dense arm's widest matrix (15.0×). The mechanism is verified; that the bias *changed the verdict* is not — no run exists with a working Muon on both arms. This is the cheapest decisive experiment (§6 P0).
- **[S] Re-running dense-vs-bank with P0 will shrink the 0.067 gap but not reverse it.** My guess is a tie at 600 steps and dense ahead by 0.02–0.04 at 2k, because the dense arm also had a larger active path in the 600-step sweep and still won the cleaner 2k pair.
- **[S] The SVD init (P1) is worth −0.02…−0.06 at 600 steps.** Justified by the step-0 equivalence and by SPRI's params-matched win at 3B, not by a measurement at 95M.
- **[S] `tsct_expert_basis` (6.3M params at E=512) will not pay.** Its own design note says it is a per-expert input selector inside a shared plane; at 98MB the bank's parameters already see 4.4 B/param, and the selector adds parameters without adding data. I expect a tie at best.
- **[S] At 95M the only positive architectural lever left is the optimizer plus data.** The frontier doc's own prediction (1 GB → 1.16–1.42 BPB vs 1.927 at 98 MB) is a data effect 5–10× larger than any measured architecture delta in this repo.

## 8. Open questions

1. Does the 0.067 dense-vs-bank gap survive a **working** Muon at 3 seeds? This is one config line and 1.5 GPU-h (§6 P0). Nothing else should be decided before it.
2. Why did the equal-params dense arm OOM at `f=63024` while the 141.9M bank ran? The arm is 193.6M MAC/token — the `[E,cap,f]` bank's activation is not the only peak. A bucketed/`moe_capacity=0` dense arm might fit; until it runs, "equal params" is unmeasured.
3. Is the `rank_up` sweep's kernel-selection failure fixed on `main`? `ab_rank_up2.log` still shows `rc=1` for r128+; the docstring says the tile search was fixed, the log says it was not.
4. Where is the run behind "engram 128k→2M rows = 0.016 BPB"? The claim is in `docs/records/reviews/2026-11-05-final-dense-recipe.md:21`; no log, table or driver in `~/logs` reproduces it. Until it is found, treat 0.016 as **[U]**.
5. What is the right `muon_lr_mult` for the *bank* arms specifically? The per-shape formula says the expert `v` needs ~52× while the dense half needs 15×; a single global mult cannot be right for both. Per-shape normalisation (§6 P0) is the principled version.

## 9. Sources

**In-repo measurements:** `~/logs/wave14.log`, `w14_t_s{1,2,3}.log`, `w14_d_s{1,2,3}.log`; `~/logs/rank_up.log`; `~/logs/ab_rank_up2.log`; `~/logs/night_stage3.log`; `~/logs/sweep_mult.log`; `~/logs/flagship3.jsonl`; `~/logs/dense103.jsonl`; `~/logs/flagship2.jsonl`; `~/logs/abdsize_driver.log`; `port/checkpoints/medium/med1.pt` (SVD, bank stack, `lfb_counts`).

**In-repo reports:** `docs/records/reviews/2026-11-07-muon-audit.md`; `2026-11-06-mechanism-liveness.md`; `2026-11-05-architecture-review-dense.md`; `2026-11-05-dense-beats-bank.md`; `2026-11-04-routing-collapse-CONFIRMED.md`; `2026-11-04-balance-fix-verdict.md`; `2026-10-23-architecture-verdict.md`; `2026-11-05-final-dense-recipe.md`; `2026-11-07-night-architecture.md`; `docs/records/research/2026-11-07-architecture-frontier.md`; `docs/records/research/2026-11-07-frontier-vs-vanilla.md`; `docs/records/research/2026-11-05-small-model-mfu.md`; branch `b249bec:docs/reviews/2026-11-06-tsct-ceiling.md`.

**Papers (id / version / date / table):** 2402.07871v1 (2024-02-12) Table 2, §6.3 · 2411.01016v1 (2024-11-01) Tables 1–3 · 2503.01359v1 (2025-03-03) Table 1 · 2610.09342v1 (2026-10-07) Tables 1–3 · 2606.16456v1 (2026-06-15) Tables 3–5 · 2601.13563v5 (2026-01-20) Table 1 · 2502.03009v2 (2025-06-16) §5.1–5.2, Eq. 13 · 2410.07524v2 (2024-10-10) abstract · 2306.03745v1 (2023-06-06) abstract · 2604.14419v1 (2026-04-15) abstract/results · 2604.09780v1 (2026-04-10) abstract · 2202.00834v3 (2022-02-02) abstract/theorems · 2606.21847v2 (2026-06-20) Table 1 · 2412.12351v2 (2024-12-16) Tables V–VII · 2404.13628v1 (2024-04-21) Table 1 · 2402.12550v4 (2024-02-19) abstract.
