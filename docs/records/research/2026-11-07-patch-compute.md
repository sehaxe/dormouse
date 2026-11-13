# Patch/token compute units at 95M–1B: what is published, what our failure was, and the arithmetic of 8 bytes/position

**Date:** 2026-11-07. **Mode:** read-only — arXiv MCP (abstracts + full-text sections + tables), repo docs and run logs; no GPU run, no code edit.
**Marks:** **[T]** number read from a paper table · **[A]** abstract/text only · **[C]** computed here from stated numbers (formula shown) · **[M]** measured in this repo (path or log named) · **[S]** SPECULATION, no direct measurement.

**Scope of the question.** The owner's goal, verbatim: *"much less VRAM, much faster training, quality not lost — so that one card can train intelligence comparable to a small frontier model"* [`.bulba/goal.md`]. The candidate unit is the byte patch: one position = 8 bytes instead of 1. This report answers four things: what is published on patch/token-level units 2023–2026, what those papers say about our measured codec failure, the arithmetic of the 8× unit on one 16 GiB card, and three ranked experiments.

---

## 0. TL;DR (RU)

1. **Ни один опубликованный метод в диапазоне 95M–1B не бьёт обычный байтовый патчинг при равном качестве и равном счёте.** Единственные положительные equal-compute результаты — TST (мешок из 4 токенов, 2.5× на 10B) и Scratchpad (2.9× на 2B, но против *чисто байтовой* базы). BLT при 1B **хуже** BPE; его ps=8 пересекает BPE только на 7B или на 2.5–3× выше compute-optimal бюджета [T, 2412.09871v1 §5.1/§5.3]. MegaByte — выигрыш в качестве при 1.11× счёта и 3.2× параметров, не в счёте [T, 2305.07185v2 Table 2].
2. **Наш failure mode — не «слепой кодек», а утечка или отставание.** Наш conv-кодек строил латент патча i из байтов патча i и предсказывал те же байты → фальшивый BPB 0.4992 [M, `2026-11-03-three-verdicts-and-a-leak.md` §3]. Если просто замаскировать свои байты, кодек становится слепым (то самое «receptive field 5–12 Б»). Опубликованное решение — **сдвиг цели на один патч**: энкодер видит весь патч j, декодер предсказывает патч j+1 (MegaByte §3.2; BLT §3.3 «decode the next patch»). Это стоит ноль дополнительных параметров и позволяет сохранить кодек за 5.4%, если его поле не меньше патча; иначе нужен structure C/D.
3. **Арифметика.** При 95M и p=8: счёт/байт падает 691.4 → 73.0 MFLOP (**9.47×**), активации/байт — примерно в 8×. Но это два разных режима. **Равные байты на шаг:** потолок ускорения `1/(f+(1−f)/8)`; при измеренной доле не-позиционной работы f=0.64 это **1.46×**, и никакая цена кодека не даёт 3×. **Равные позиции на шаг:** байтов на шаг ×8 при том же VRAM и том же времени шага; там выигрыш `8/(1+c)`, и кодек может стоить до **167%** основного шага, всё ещё давая 3×. Измеренные 5.4% далеки от порога — узкое место не кодек, а качество на байт.
4. **«VRAM ÷4» верно только для активаций.** При 95M фиксированная часть (веса+оптимизатор+градиенты) — 1.0–2.3 GiB из ~7 GiB; при равных байтах общий VRAM падает в **2.3–4.2×**, а не в 8×. При 1B фиксированная часть — 16.8 GiB во fp32, и патчинг сам по себе не спасает: 1B на 16 ГиБ — это 8-битный оптимизатор (11.2 GiB), а патчинг лишь возвращает батч (≈25 kB против 3 kB на микро-шаг) [C].
5. **Вердикт:** патчинг — рычаг скорости/памяти и данных, не рычаг качества при 95M–1B. Как база он впереди только в data-starved режиме (≤1–2 ГБ на 95M). Сначала — 14-часовой контроль «равные GPU-часы, 8× байтов»; он решает, покупают ли патчи качество или только скорость.

---

## 0.1 Сначала аудит: три числа из брифа не совпадают с артефактами

| Утверждение брифа | Что в артефакте | Вердикт |
|---|---|---|
| «−0.3725 BPB **на 199M** на mini-polygon» | `~/logs/night2/ab_byteflow.log:133`: control 3.6152 ± 0.0202, channel 3.2427 ± 0.0099, delta −0.3725. Шапка: **1.9M params, b4 s512, 600 шагов, eval 40,960 B**; channel **медленнее**: 70.1 vs 53.6 ms/step | **[M] число верное, масштаб неверный.** Это 1.9M, не 199M; и это attention-кодек, не conv |
| «conv codec lost 1.5 BPB at 199M» | `~/logs/ab_bf400/bf8_s1.log`: 4.135 BPB, 150.8M params, 800 шагов; `full_s1.log`: 2.684 BPB, 199.3M params. Разница 1.45 BPB, но у conv-плеча **нет Engram/mHC/CSA/LayerRoPE** и на 48M меньше параметров | **[M] число верное, причинность не установлена.** Это конфаундированный full-recipe дельта, а не эффект рецептивного поля |
| «2.04× скорость шага после bf16-фикса» | Не найдено ни в одном логе/доке. Ближайшее: 47.6 → 32.6 ms = **×1.46** [M, `2026-11-03` §3]; bf4→bf8 при 150M: 23.1k → 24.3k поз/с = **×1.05** [M, `~/logs/ab_bf400/*.jsonl`] | **Не подтверждено.** Число 2.04 в артефактах отсутствует |
| «conv codec 357k params at 5.4% of the step» | encode 0.84 + decode 0.91 ms из 32.6 ms = 5.37% [M, `2026-11-03` §3]; 357,120 против 93,909,248 параметров, 1.27–1.90% MAC | **Подтверждено** (но шаг launch-bound: 2.9% по FLOPs, 5.4% по wall-clock) |

**Отсюда правило для всего отчёта:** «−1.5 BPB» — это не измерение слепоты кодека. Чистого A/B «causal conv vs byte» при 95M–199M в репозитории **нет**. Это и есть эксперимент E2.

---

## 1. PART 1 — published patch/token compute units, 2023–2026

### 1.1 ID corrections (the brief's own citations)

| Brief | Resolves to | Correct ID |
|---|---|---|
| MegaByte `2305.15011` | Bactrian-X, multilingual LoRA instruction tuning (2023-05-24) | **`2305.07185`** — MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers (2023-05-12) |
| «TST/Token-Superposition `2610.05978`» | *Byte Language Models: Scaling, Emergent Abstractions, and Information Allocation* (2026-10-05) — it **uses** TST, it is not the TST paper | TST is **`2605.06546`** — Efficient Pre-Training with Token Superposition (2026-05-07) |

### 1.2 The comparison table

`Codec cost` is the fraction of the training step (FLOPs, or wall clock where the paper prints it) paid by the byte→patch/local modules. `Match` names what is held equal: **iso-FLOP**, **iso-param+tokens**, **iso-quality**, or **fixed data**. No paper reports the codec as a wall-clock fraction except ours; where the number is derived from the paper's own hyper-parameters it is marked [C].

| # | Candidate (id, version, date) | Unit | Codec cost, share of the step | Match | Measured delta | Crossover / scale | Credibility | Effort to adopt/verify |
|---|---|---|---|---|---|---|---|---|
| 1 | **BLT** — `2412.09871v1` (2024-12-13) | entropy patches, 4.5 / 6.1 / 8 B | **35.3% @400M, 36.0% @1B, 9.0% @8B** [C: (enc+dec params) ÷ (that + global/p), Table 10]; at 1B·ps=8 with 3 encoder layers **54% at fixed global size**, ≈41% in BLT's own run because it grows the global 1.7× [C] | **iso-FLOP** (compute-optimal BPE vs BLT) | 8B avg: Llama3 60.0, BLT-Space 58.0, BLT-Entropy **61.1** [T, via `2026-11-06-flops-per-byte.md` §2.5]; ps=8 **worse at 1B** | ps=8 beats BPE at **7B**; 470M–1B needs **2.5–3× compute-optimal** budget [T, §5.3] | industry (Meta), table + scaling curves; no independent replication; scale 4–80× ours | **high**: local/global stack + 150B bytes at 470M |
| 2 | **MegaByte** — `2305.07185v2` (2023-05-12) | fixed 8 B | **73.4%** [C: local 262M ÷ (local + global/8 = 94.75M)]; total **713.5 vs 640 MFLOP/byte** forward = 1.11× [C from §2.1 formula] | **compute+data controlled** (forward time per byte matched), **params NOT matched**: 1.02B vs 320M | PG-19 BPB **1.057→1.000**; Code **0.575→0.411**; arXiv 0.816→0.678 [T, Table 2] | one operating point (320M control, 80B bytes); no scaling study | industry (Meta), table-read; the "win" is params-per-FLOP, not FLOPs-per-byte | **high**: two-level model |
| 3 | **SpaceByte** — `2404.14408v3` (2024-04-22) | space boundaries, avg **6–8 B** | local blocks = global blocks in count, split half before/after [T, §4.1] → ≈50% of layers [C] | **fixed training+inference compute** | "roughly matches tokenized" [A]; BLT's reproduction: space patching **"remains far from Llama 3"** [T, 2412.09871 §5.1] | 1e18/1e19 FLOPs, ≤~1B | academic, compute-controlled; contradicted by BLT's reimplementation | **medium**: block insertion, no new codec |
| 4 | **Scratchpad Patching** — `2605.09630v1` (2026-05-10) | 4–16 B + intra-patch scratchpad | SP costs **1.97×** the vanilla patched FLOPs (5.7×→2.9× reduction); **49% of the final step** [C] | **fixed data**, measured FLOPs/byte vs **pure byte (1 B/patch)** | byte 54.1; p=16 48.0 (5.7×); **p=16+SP 54.2 (2.9×)**; p=8+SP 53.8 (2.6×); p=4+SP 56.2 (2.1×) [T, Table 1] | ~2B params, ~400B bytes; no ≤200M sweep | academic/industry, **strongest equal-quality FLOPs number in the set**; baseline is pure byte | **medium-high**: scratchpad mechanism + entropy trigger |
| 5 | **ByteFlow Net** — `2603.03583v1` (2026-03-03) | learned compression 2.0–5.12× (≈2–5 B) | **NOT MEASURED** (no FLOPs table); 6-layer local encoder + symmetric decoder [T, App. C.2] | fixed data | 600M: 50.89 vs LLaMA 49.15 (**+1.74**); 1.3B: **+3.04** [T, §5.3]; compression sweep 2.0× 51.74 / 2.56× 50.89 / 3.41× 50.13 / 5.12× 48.48 [T, Table 6 via repo doc] | 600M crosses LLaMA at **~25B tokens** [T, §5.3] | academic; params not matched; no FLOPs | **high**: coding-rate chunking + codec |
| 6 | **H-Net** — `2507.07955v2` (2025-07-10) | learned 4.8–5.3 B chunks | inside **0.43 vs 0.42 GFLOP/byte** at 760M; not separable [T, Table 2] | **data + FLOP matched** | BPB **0.756→0.743** (Large), 0.730→0.715 (XL); avg 53.3→55.5, 55.5→58.2 [T, Table 2] | 760M/1.3B, 100B tokens | academic, clean matched table | **high**: hierarchical chunker |
| 7 | **TST (Token Superposition)** — `2605.06546v2` (2026-05-07) | bag of **4 tokens** (not bytes) | **0** (embedding average + multi-hot CE; no codec) | **equal loss** (wall-clock time to a loss) | **2.5× less total pre-training time at 10B A1B**; validated at 270M, 600M, 3B [A] | 270M–10B | industry, equal-loss timing; strongest positive equal-compute result | **low**: drop-in objective |
| 8 | **Byte-LM scaling** — `2610.05978v1` (2026-10-05) | 1 B/position + TST(4) + hash n-grams | 0 | **iso-FLOP** and **iso-param** | **at equal FLOPs, Subword is always better**; at equal optimal params, Byte is better; Byte allocates **3.2× bytes/param** [T, §3.2/Fig. 1] | 200M–3B | academic, isoFLOP sweep — the cleanest negative for "fewer positions wins" | **low**: TST + hash + wider conv |
| 9 | **EntropyMoE** — `2608.06398v1` (2026-07-31) | BLT patches + Top-2 experts | no compute win claimed | matched data, updates, **active params** | lowest held-out BPB vs dense BLT and hidden-state routers; **"does not establish a compute-matched advantage"** [T, §1] | continuation training, scale not stated | academic, self-limiting claim | **medium** |
| 10 | **ConceptLM / NCP** — `2602.08984v1` (2026-02-09) | VQ over hidden states (concepts) — **still BPE at input** | theoretical **4.69% FLOPs** cut; measured **−11.3% train time, −13.1% inference** at 8B [T, §6.5] | **params matched** (PM baseline) | GPT2-124M avg PPL 101.94→**84.75**; +0.9 acc; Pythia-70M **+3.6**; Pythia-410M +0.8 [T, Tables 1–2] | 70M–1.5B, 300B tokens | academic, params-matched, not a byte unit | **medium**: VQ module + concept head |
| 11 | **BPE-dropout** — `1910.13267v1` (2019-10-29); pretraining study `2605.13436v1` (2026-05-13) | subword segmentation diversity | **0** (tokenizer-level) | none at LM pretraining scale | 2026 study is **BERT, low-resource** (XNLI/PAWS-X/PAN-X/MasakhaNER); no LM BPB at 95M–1B [A] | — | academic; wrong regime | **low**, but it is not a compute-unit lever |
| 12 | **RVQ / EnCodec-style discrete codes for text** | — | — | — | **NO PAPER FOUND** in this pass. `all:"residual vector quantization" AND all:"language model"` returns KV-cache and audio codecs; `all:"text codec"` returns a lossy-compression paper. The audio analogy does not transfer: text bytes are already discrete, there is no continuous waveform to reconstruct | — | **negative search result, not proof** | — |
| 13 | **Fast BLT** — `2605.08044v1` (2026-05-08) | same patches | inference-only (diffusion/self-speculation) | generation speed | >50% lower memory-bandwidth cost at generation [A] | inference | industry | **medium**, but **no training FLOPs** |

### 1.3 What the table says (RU)

1. **Единственный чистый equal-compute выигрыш в качестве — не патчинг, а TST.** 2.5× меньше времени до того же лосса на 10B A1B [A, 2605.06546v1]. Механизм: 4 токена усредняются в одну позицию на 30% обучения, потом обычный NTP. Это ровно «одна позиция = 4 единицы», но **на уровне токенов, без кодеков и без потери байтовой информации**. При 95M это самый дешёвый способ проверить сам принцип «меньше позиций на единицу текста».
2. **Патчинг выигрывает в параметрах, а не в счёте.** MegaByte: 1.02B параметров против 320M при равном времени forward — это +0.057 BPB [T, Table 2], но 1.11× счёта/байт. BLT: равный счёт, +1.1 пункта на 8B, но **хуже на 1B**. Ни одна работа не показывает «8 B/патч лучше 4.5 B/патч при равном счёте и равных параметрах» — наоборот, BLT ps=8 стартует хуже и догоняет только на 7B [T, §5.1].
3. **Цена кодекa растёт, когда патч растёт, если глобальная модель фиксирована.** У BLT на 1B локальные модули — 36% счёта при p=4.5 и 54% при p=8 с тремя слоями энкодера [C из Table 10]. Это не «лёгкий кодек»: это треть-половина шага. Наш attention-кодек измерил 34.8% [M] — совпадение не случайное, это одна и та же архитектурная цена. Наш conv-кодек в 5.4% в 6–7 раз дешевле именно потому, что делает меньше работы (и поэтому слеп/течёт).
4. **Crossover по масштабу есть у всех, и он выше нашего.** BLT — 7B; MegaByte — без скейлинга; Scratchpad — 2B; ByteFlow — 600M с 25B токенов; H-Net — 760M. В 95M–1B положительных equal-quality результатов **нет**.
5. **Наш режим — data-starved, и это меняет знак.** При 95M и 98.3 МБ корпуса мы на **1.03 Б/параметр** против оптимума ≈50 Б/пар [C из T, 2610.05978v1 Table 8]. В этом режиме 8× байтов на шаг может выиграть даже при потере 0.05–0.15 BPB на байт — потому что байты сейчас дороже качества. Это единственная формулировка, при которой цель владельца не противоречит литературе.

### 1.4 Negative results (what I did not find)

- **No paper uses RVQ/VQ codes as the pretraining unit for text.** The discrete-code literature is audio (EnCodec, SpeechTokenizer), KV-cache compression (`2410.15704v1`), and embedding compression (`2510.12721v1`). ConceptLM quantizes *hidden states* but still reads BPE tokens.
- **No paper reports the codec as a wall-clock fraction of the training step.** BLT/MegaByte/Scratchpad report FLOPs or nothing. Our 5.4% is the only wall-clock measurement in this comparison; every other codec share here is [C] from published hyper-parameters.
- **No BLT patch-size ablation at 95M–470M.** Table 10 gives 400M–8B configs; the ps=8 vs ps=6 study is at 400M–3.6B with the global model grown 1.6–1.7× [T, §5.3], so it cannot be read as "8 B is better at fixed global size".
- **No text result above 16 B/patch.** Largest measured is Scratchpad p=16 (48.0 vanilla, 54.2 with SP) [T, Table 1].

---

## 2. PART 2 — what the methods say about our failure mode

### 2.1 The three artifacts, separated

| Artifact | What was run | Number | What it proves |
|---|---|---|---|
| `docs/records/reviews/2026-11-03-three-verdicts-and-a-leak.md` §3 | conv codec, 314M shape, b8 s1024 | eval@400 **0.4992**, train loss 0.7851 | **Leak, not learning.** `encode_chunks` pooled the patch's own bytes with a non-causal conv and the decoder predicted them. The "−3 BPB" is fake |
| same doc | causal attention codec A/B | 3.2427 vs 3.6152, **−0.3725** | A causal encoder is plausible; measured at **1.9M/40,960 B** (`~/logs/night2/ab_byteflow.log`) |
| `~/logs/ab_bf400/{bf8_s1,full_s1}.log` | conv codec 150.8M vs full 199.3M, 800 steps | 4.135 vs 2.684, **+1.45 BPB** | A **confounded** full-recipe gap: the codec arm refuses Engram/mHC/CSA/LayerRoPE and is 48M smaller |

**The brief's causal story ("a 5–12 byte receptive field is blind to the patch it encodes") is not what these artifacts measure.** The docs measure (a) a leak when the encoder sees its own patch, and (b) a confounded recipe gap when it doesn't. The blindness is a plausible mechanism, not a measured cause.

### 2.2 The taxonomy of patch-decode structures — and which one leaks

| Structure | Encoder sees | Decoder predicts | Leak? | Blind? | Published example |
|---|---|---|---|---|---|
| **A. Same-patch decode** (our conv codec) | patch j (pooled) | bytes of patch j | **YES** — target is inside the input | no | our `byteflow_codec=conv` |
| **A′. Same-patch decode, masked** | bytes < patch j only | bytes of patch j | no | **YES** — the latent is built from the previous patch; a 5–12 B conv sees only the tail of it | the naive "fix" the brief describes |
| **B. Shift by one patch** | whole patch j (causal or not) | bytes of patch **j+1** | **no** | **no** — the encoder may see all of patch j because every byte it predicts is strictly later | **MegaByte §3.2** (global input is patch k−1; local model predicts patch k); **BLT §3.3/Fig. 2** ("decode the next patch of bytes") |
| **C. Causal encoder + per-byte residual + boundary latent** | bytes ≤ boundary s_i | bytes ≥ s_i | no | **partly** — patch lag: bytes inside the chunk condition on the chunk's *first* byte | **ByteFlow §4.1/4.4/4.5** (SWA + Canon, `chunk(t)=argmax{s_i≤t}`); lag formalized in **Scratchpad §3.2** |
| **D. Causal encoder + intra-patch scratchpad** | bytes so far, refreshed inside the patch | next byte | no | **no** — the patch context is refreshed at entropy-triggered points | **Scratchpad §4.1**; cost 1.97× the vanilla patched FLOPs [C] |

**The single most important line in the literature for us** is Scratchpad §3.2: *"only the final byte of each patch can condition on the current patch's trunk output, while all earlier bytes must instead rely on the output of the previous patch… we refer to the gap … as patch lag."* That is exactly the failure the brief attributes to the receptive field — and it is a property of **causal patch decoding**, not of convolution. A conv codec with a 5–12 B field makes it worse, but a transformer codec has the same lag unless it adds structure C or D.

### 2.3 The specific architectural fix, with the paper's evidence

**Fix: structure B — align the target one patch ahead.** The encoder may see the whole patch j (so the 5–12 B field is no longer a blindness problem, as long as the latent is taken after the patch's last byte or pooled over causal states); the decoder predicts patch j+1. This is MegaByte's exact construction:

> `h_global_in_k = embed(x[(k−1)P : kP])` and the local model at patch k predicts patch k autoregressively from `h_global_out_k` plus the bytes of the next patch offset by one [T, 2305.07185v2 §3.2, eqs. (2)–(4)].

BLT uses the same shift at the patch level ("decode the next patch of bytes", §3.3/Fig. 2), with a block-causal global mask. **Cost: zero extra parameters, zero extra FLOPs** — the same encoder/decoder passes, just a one-patch offset between the latent and the target. The future-byte invariance gate (changing `bytes[:, q:]` must not change `logits[:, :q]`) then passes by construction: every predicted byte is strictly after every encoded byte.

**What it costs in quality:** one patch of latency in the *target alignment*, which is the normal next-patch LM objective, not an extra loss. MegaByte's numbers are the evidence that this structure trains: 1.057→1.000 BPB on PG-19 at 1.11× FLOPs/byte [T, Table 2]. BLT's numbers are the evidence that it scales: matches BPE at 8B, +1.1 avg points at iso-FLOPs [T, §5.1 + repo table read].

**Why not simply mask the patch's own bytes (A′)?** Because then the latent for patch j is a function of bytes < j only, and the decoder must predict all of patch j from the previous patch — the model has to carry the entire intra-patch content in one latent, with no per-byte residual. That is the "blind" regime, and it is strictly weaker than B, where the decoder also gets the bytes of patch j+1 already generated (MegaByte eq. 4) and the encoder's per-byte states (ByteFlow eq. 17).

**The honest cost of the quality-preserving alternatives:** BLT pays 36% of FLOPs at 1B; MegaByte pays 73%; Scratchpad pays ~2× the vanilla patched step. Our conv codec at 5.4% cannot be doing the same work. The choice is: (a) keep 5.4% and accept B's one-patch shift (cheap, untested at 95M), or (b) pay 30–50% for C/D (published, measured at 2B). E2 tests (a).

---

## 3. PART 3 — the arithmetic of the goal

### 3.1 FLOPs per byte

Baseline is the measured dense arm, `port/configs/flagship3_dense.toml`: 95,035,200 params, 12 layers, d=832, f=3072, V=256, T=1024 bytes, fwd+bwd. The ledger is from `docs/records/reviews/2026-11-07-metrics.md` §4 [T/M]:

| Term | Formula (fwd+bwd) | MFLOP/byte at p=1 |
|---|---|---:|
| block (qkv+proj+MLP) | `3·2·L·(4d² + 2df)` | 567.410 |
| attention core | `3·4·L·T·d` | 122.683 |
| head | `3·2·d·V` | 1.278 |
| **total** | | **691.371** |

At `p` bytes/patch, with the **same** global model and the same 1024-byte window:

```
FLOPs/byte(p) = (block + head)/p + attention/p²
```

because positions/byte = 1/p and the attention core per position is ∝ T_pos = T/p [C].

| p (B/patch) | block+head | attention | total MFLOP/byte | speedup vs p=1 | + conv codec (2.14) |
|---:|---:|---:|---:|---:|---:|
| 1 | 568.69 | 122.68 | **691.37** | 1.00× | 693.5 |
| 2 | 284.34 | 30.67 | **315.01** | 2.19× | 317.2 |
| 4 | 142.17 | 7.67 | **149.84** | 4.61× | 152.0 |
| 8 | 71.09 | 1.92 | **73.00** | **9.47×** | 75.1 |
| 16 | 35.54 | 0.48 | **36.02** | 19.19× | 38.2 |

The codec is `6 × 357,120 = 2.143 MFLOP/byte` [C; 357,120 from `2026-11-03` §3] — **2.9% of the p=8 total**, not 5.4%; the 5.4% is wall clock on a launch-bound step, where the codec's launches cost more than its arithmetic.

### 3.2 Activation memory and max batch on 16 GiB

Per-position activation for the dense block, no checkpointing, fp32 saves (calibrated against the measured 7.2 GiB peak at b8·s1024, `docs/records/reviews/2026-11-03-three-verdicts-and-a-leak.md` and `model/act_mem.py`):

```
A_pos = L·(8d + 2f)·4 B = 12·(6656 + 6144)·4 = 614,400 B = 0.586 MiB/position
```

At `p` bytes/patch, `A/byte = 0.586/p MiB`. The conv codec adds ≈0.014 MiB/byte (d_local=128, f=384, 2 layers, [C]); for the attention codec I use 0.05–0.10 MiB/byte as a **[S] estimate** — its activation set was never measured, only its 34.8% wall-clock share.

| Fixed cost on card | p=1: max micro-batch | p=8: max micro-batch | ratio |
|---|---:|---:|---:|
| 1.0 GiB (measured "fixed" from `act_mem.py`) | 25,865 B | 174,370 B | **6.7×** |
| 2.3 GiB (fp32 master+grads+AdamW+bf16 weights+ctx) | 23,593 B | 159,054 B | **6.7×** |

At **equal bytes per micro-batch (8192 B)** the total VRAM is:

| Fixed | p=1 | p=8 | ratio |
|---|---:|---:|---:|
| 1.0 GiB | 5.69 GiB | 1.70 GiB | **3.35×** |
| 2.3 GiB | 6.99 GiB | 3.00 GiB | **2.33×** |

So **"VRAM ÷4" is the optimistic edge, not the middle**: against the analytic baseline it is **2.3–3.4×** at equal bytes with a real optimizer state, and only if the codec is nearly free. Against the **measured** 7.2 GiB peak the fixed=1.0 row is 7.2/1.70 = **4.2×**, because the measured peak carries ~1.5 GiB beyond the four analytic terms; that is where the owner's "÷4" comes from. The honest range across both baselines is **2.3–4.2×**. At equal *positions* (8× bytes per step) the VRAM is unchanged — that is the regime that buys throughput, not memory.

### 3.3 Two speed regimes, and what the codec may cost

Let `f` be the share of the step that does **not** scale with positions (optimizer, launches, host gap). Measured at 314M: 47.6 → 32.6 ms for p=8 ⇒ `f = 0.64` [M, `2026-11-03` §3]. Measured at 150M, p=4→8: 23.1k → 24.3k positions/s ⇒ `f ≈ 0.87` [M, `~/logs/ab_bf400/*.jsonl`]. At 60 kB/s, one optimizer step is 24,576 B in 0.41 s.

**Regime A — equal bytes per step (positions ÷ p):**

```
S_A = 1 / (f + (1−f)/p) · (1−c)
```

| f | p=8, c=0 | p=8, c=5.4% | 3× possible? |
|---|---:|---:|---|
| 0.64 (measured 314M) | **1.46×** | 1.38× | **No** — even a free codec caps at 1.46× |
| 0.15 (compute-bound hope) | 5.45× | 5.16× | yes, for c ≤ 45% |

**This is the central arithmetic result: at equal bytes, our measured fixed share caps the win at 1.46×, and the codec is irrelevant to whether 3× is reached.** The 2.04× in the brief is above this ceiling.

**Regime B — equal positions per step (bytes × p, same VRAM, same step time):**

```
S_B = p / (1 + c),   c = codec time / main-model time
```

| codec | c | S_B at p=8 |
|---|---:|---:|
| our conv (5.4% of step) | 0.057 | **7.57×** |
| our attention (34.8% of step) | 0.534 | **5.21×** |
| BLT-like (36% of FLOPs at 1B) | 0.56 | 5.13× |
| MegaByte-like (73%) | 2.70 | 2.16× |

For `S_B ≥ 3×`: `c ≤ 8/3 − 1 = 1.67`, i.e. **the codec may cost up to 167% of the main model's step** and still clear 3×. At 60 kB/s baseline, Regime B with c=5.7% gives **454 kB/s**, and 1 GB trains in **0.61 h instead of 4.63 h**. **The codec's 5.4% is nowhere near the binding constraint in Regime B. The binding constraint is per-byte quality.**

### 3.4 The 1B case — patching alone does not put 1B on 16 GiB

| State | 1B dense, fp32 AdamW | 1B dense, 8-bit AdamW |
|---|---:|---:|
| bf16 weights | 1.86 GiB | 1.86 GiB |
| fp32 master | 3.73 | 3.73 |
| grads | 3.73 (fp32) | 1.86 (bf16) |
| AdamW m+v | 7.45 (fp32) | 3.73 (8-bit) |
| **fixed total** | **16.77 GiB** | **11.18 GiB** |
| left for activations on 16 GiB | **−0.8** | **4.6 GiB** |
| max micro-batch, p=1 (d≈2048, A/byte≈1.5 MiB) | — | **≈3.1 kB** |
| max micro-batch, p=8 (A/byte≈0.19 MiB + codec) | — | **≈23.4 kB** |

[C; A/byte for 1B scales as d, `12·(8·2048+2·8192)·4 = 1.57 MB/position`; no activation checkpointing]. **Patching is what turns b3 into b23 at 1B; the 8-bit optimizer is what makes the fixed state fit at all.** Full activation checkpointing multiplies both batches by ~4–8× (the repo's ledger reads 1B dense b1 → b13 with full checkpointing, `memory.md` 2026-11-06), so the honest statement is: patching and the 8-bit optimizer are two independent levers, and neither alone reaches a useful batch at 1B.

### 3.5 Verdict (RU)

- **Счёт:** p=8 режет FLOPs/байт в 9.47× — это правда и это много.
- **Память:** активации в 8×, общий VRAM при равных байтах в 2.3–4.2×. «÷4» — только при фиксированной части 1.0 GiB и почти бесплатном кодеке.
- **Скорость:** при равных байтах потолок 1.46× (измерено), при равных позициях — 7.6× с нашим кодеком. Цель «×4» живёт **только** во втором режиме.
- **Качество:** ни одна работа не показывает equal-quality выигрыш при 95M–1B. В data-starved режиме (наш) 8× байтов может перевесить потерю на байте — это гипотеза E1, не факт.

---

## 4. PART 4 — three experiments, ranked

Protocol for all three: dense 95M (`flagship3_dense.toml`), pure CE, `eval_batches` printed, same data order, 3 seeds, held-out window ≥40,960 B, and the pre-registered rule "win only if the delta beats the control's own seed spread" (AGENTS §1.2). Costs at **60 kB/s** = **4.63 h/GB**.

### E1 — the equal-GPU-hours control (rank 1; decides the thesis)

**Question:** do patches buy quality, or only speed? Spend the same wall clock on **more bytes** instead of fewer positions.

| Arm | Unit | Bytes seen in 4.63 h | Steps | Expected BPB | Basis |
|---|---|---|---:|---|---|
| C0 byte | 1 B/pos | 1.0 GB | 40,690 | **baseline** | measured slope: 98.3→268 MB = −0.408 BPB [M, `2026-11-07-architecture-frontier.md` §TL;DR] |
| P8 + attention codec | 8 B/patch, c=34.8% | **5.2 GB** | 26,500 | **−0.5…+0.1 vs C0** [S] | 5.2× data at 55 B/param vs 10.5; per-byte penalty 0.05–0.15 at 8 B/patch [T, 2412.09871 §5.1] |
| P8 + conv codec (after E2) | 8 B/patch, c=5.4% | **7.6 GB** | 38,400 | **−0.55…+0.1 vs C0** [S] | 7.6× data at 80 B/param; same penalty |

**Config change:** none in the model; `use_byteflow=true`, `byteflow_codec=attention` (the causal one that already measured 3.2427), `byteflow_k_tokens=64` for a 512-byte window (measured mean 4.0 B/patch at K=128; K=64 was a tie, 2.9431 vs 2.9410 [M, `2026-10-21-battery.md` §2]). Train the byte arm on 1 GB and the patch arm for the same wall clock — the byte count is an output of the codec cost, not an input.

**GPU-hours:** 3 × (4.63 + 4.63) = **27.8 h** for the byte/attention-codec pair. Screening version at 500 MB byte vs 2.6 GB patch: **13.9 h**.

**Falsifier:** if P8's BPB at 8 GB is **worse** than C0's at 1 GB, patching is a speed-only lever and must not be the base of the recipe. If P8 wins by >0.1 BPB, the data-efficiency thesis holds in the data-starved regime; then repeat at 22 GB where the sign should flip.

**arXiv basis:** BLT §5.1 (ps=8 worse at 1B), `2610.05978v1` §3.2 (subword always better at equal FLOPs; byte better at equal params), MegaByte Table 2 (quality at fixed compute, 3.2× params).

### E2 — the causal-codec fix (rank 2; cheapest, unblocks E1's cheap arm)

**Question:** does the one-patch shift (structure B) remove the leak **and** the blindness at zero cost?

| Arm | Encoder | Target | Expected BPB vs byte @ equal bytes | Basis |
|---|---|---|---:|---|
| C0 | — | byte t | baseline | — |
| B8-shift | conv, 357k, sees whole patch j | bytes of patch **j+1** | **+0.05…+0.20** [S] | BLT ps=8 worse at 1B; our conv capacity is 357k |
| B8-shift + gate | same + future-byte invariance gate | same | must equal B8-shift bit-for-bit | gate definition from `2026-11-03` §3 |

**Config change:** in `encode_chunks`/the patch head, offset the latent by one patch (MegaByte eq. 2–4); add the gate `bytes[:, q:] → logits[:, :q]` on CPU; keep `byteflow_codec=conv`, 8 B/patch.

**GPU-hours:** 200 MB × 3 seeds × (0.93 h byte + 0.64 h patch) = **4.7 h**; 500 MB confirm = 11.7 h.

**Falsifier:** any BPB **below** the byte control by >0.3 at equal bytes is a leak signature, not a win (the 2026-11-03 failure was 3 BPB below). A loss >0.2 BPB kills the conv codec for this scale.

**arXiv basis:** MegaByte §3.2 (shifted patch input), BLT §3.3/Fig. 2 ("decode the next patch"), Scratchpad §3.2 (patch lag definition).

### E3 — the learned latent unit (rank 3; lowest prior)

**Question:** does a learned VQ/RVQ code over 4-byte spans beat fixed 8-byte patches?

| Arm | Unit | Codec | Expected BPB vs byte @ equal bytes | Basis |
|---|---|---|---:|---|
| C0 | 1 B | none | baseline | — |
| VQ4 | 4 B → 1 of 8192 codes, 2 RVQ stages | VQ encoder + straight-through, factorised 4×256 head | **+0.05…+0.30** [S] | no text-VQ pretraining precedent; ConceptLM's VQ is over hidden states, not bytes, and is params-matched [T, 2602.08984v1] |

**Config change:** 4-byte spans, codebook 8192, commitment 0.25, 2 residual stages; head = 4 conditional 256-way softmaxes (same head MAC/byte as the byte head, `2026-11-05-architecture-review-dense.md` §3.1); positions/byte = 1/4.

**GPU-hours:** 500 MB × 3 seeds × (2.31 h byte + 2.08 h VQ) = **13.2 h**.

**Falsifier:** codebook usage <50% (collapse) or BPB > +0.1 vs byte — then a learned unit at 95M is dead; fixed patches remain the only cheap unit.

**arXiv basis:** `2602.08984v1` §3/§5 (VQ concept module, +0.8–3.6 downstream at 70M–1.5B, params-matched); **no RVQ-as-text-unit paper exists** — this is the honest negative that makes E3 a bet, not a transfer.

### Ranking (RU)

| Rank | Experiment | GPU-h | Why this rank |
|---|---|---:|---|
| **1** | **E1** equal-GPU-hours control | 27.8 (screening 13.9) | Единственный тест метрики владельца — BPB на (ГиБ×час). Работает на существующем attention-кодеке, не требует нового кода |
| **2** | **E2** one-patch shift + gate | 4.7 | Самый дешёвый; если выигрывает, 5.4%-кодек становится законным, и тот же 4.63-часовой бюджет E1 покрывает 7.6 ГБ вместо 5.2 ГБ (+46% байтов, не ×8 — байтовая рука не меняется) |
| **3** | **E3** VQ-4 | 13.2 | Наименьший prior: ни одной публикации text-VQ как единицы; но единственный путь к *learned* unit |

If only one runs: **E1**. If E1 is negative, stop the patch lane and spend the quarter on data (the frontier report's #1: 1 GB → −0.51…−0.77 BPB, 4.6 h [C]).

---

## 5. Open questions

1. **Where is the 2.04×?** No artifact in the repo prints it. If it is the bf4→bf8 bytes/s ratio at 150M, the logs say 1.05× (23.1k→24.3k positions/s) — the discrepancy matters because the owner's plan assumes it.
2. **Does the one-patch shift actually cost nothing in quality?** MegaByte and BLT both use it, but neither ablates "same-patch decode with a causal encoder" vs "shifted target". E2 is the first clean test at 95M.
3. **What is the per-byte penalty of 8 B/patch at 95M?** BLT's ps=8 is worse at 1B; no 95M number exists. The whole Regime-B win depends on this being <0.2 BPB.
4. **Is the byte-level slope really 50 B/param?** The frontier report reads it from `2610.05978v1` Table 8; the paper's own optimum is a fitted curve over 200M–3B, not a measured 95M point [T/C].
5. **Can the codec be made causal *and* cheap?** The 5.4% conv codec and the 34.8% attention codec bracket the design space; no published design lands at 10% with quality parity at 1B.

---

## 6. VERIFIED / SPECULATION

### VERIFIED (sourced)

- **BLT**: Table 10 hyper-parameters (encoder 7M/12M/20M, decoder 50M/113M/120M, global 470M/1B/6.4B); §5.1 ps=8 worse at 1B, better at 7B; §5.3 crossover 2.5–3× compute-optimal; §3.3 "decode the next patch"; §7 Table 6/7/8 ablations. `2412.09871v1`, 2024-12-13.
- **MegaByte**: Table 2 BPB numbers, 320M control / 758M+262M / P=8 / 80B bytes; §3.2 eqs. 2–4 (shifted patch input); 713.5 vs 640 MFLOP/byte [C]. `2305.07185v2`, 2023-05-12.
- **Scratchpad**: Table 1 full row set (byte 54.1; p=16 48.0/5.7×; p=16+SP 54.2/2.9×); §3.2 patch-lag definition; SP cost 1.97× [C]. `2605.09630v1`, 2026-05-10.
- **ByteFlow**: §5.3 (+1.74 at 600M, +3.04 at 1.3B, crossover ~25B tokens); Table 6 compression sweep via `2026-11-06-flops-per-byte.md`. `2603.03583v1`, 2026-03-03.
- **TST**: 2.5× less total pre-training time at 10B A1B at equal loss; 270M/600M/3B. `2605.06546v2`, 2026-05-07 [A].
- **Byte-LM scaling**: §3.1–3.2, Fig. 1 — subword always better at equal FLOPs, byte better at equal optimal params, 3.2× bytes/param. `2610.05978v1`, 2026-10-05.
- **ConceptLM**: Tables 1–2 (PPL 101.94→84.75 at GPT2-124M; +3.6 at Pythia-70M), §6.5 (−11.3% train time, 4.69% theoretical FLOPs at 8B). `2602.08984v1`, 2026-02-09.
- **EntropyMoE**: "does not establish … a compute-matched advantage over the dense backbone". `2608.06398v1`, 2026-07-31.
- **Our measurements**: 5.4% codec share, 47.6→32.6 ms, 0.4992 leak, 3.2427 vs 3.6152, 4.135 vs 2.684, 70.1 vs 53.6 ms, 23.1k→24.3k positions/s, 0.462 s/step at 24,576 B. Paths in the tables above.
- **The FLOPs/byte and activation arithmetic** in §3 is [C]: every formula is shown, and the 0.586 MiB/position is calibrated against the measured 7.2 GiB peak.

### SPECULATION (my inference, no direct benchmark)

- **S1** E1's range −0.5…+0.1 BPB (attention codec, 5.2× data) / −0.55…+0.1 (conv codec, 7.6× data). It applies the measured data exponent (0.408 BPB over a 2.73× data increase at 98→268 MB ⇒ β≈0.19) to 5.2–7.6× and subtracts a 0.05–0.15 per-byte patch penalty. No paper measures a 95M/8 B-patch/5–8 GB point.
- **S2** E2's +0.05…+0.20 at equal bytes. BLT's ps=8 is worse at 1B, but our codec is 357k params and our scale is 10× smaller; the direction is a transfer, not a measurement.
- **S3** E3's +0.05…+0.30. There is **no** text-VQ pretraining paper; ConceptLM's VQ is over hidden states and params-matched. This is the weakest prior in the report.
- **S4** "Patching wins in the data-starved regime and loses in the capacity-limited regime" is my synthesis of the 50 B/param law and BLT's scale trend. No paper states or tests it.
- **S5** The 1B/8-bit-optimizer/b23 arithmetic assumes d≈2048, f≈4d, and a 0.5 GiB context; a real 1B config may differ by ±30%.

**Confidence.** **HIGH** on: the measurement audit (§0.1); the FLOPs/byte and activation arithmetic (§3.1–3.2); the leak/lag taxonomy and the MegaByte/BLT shifted-target structure (§2.2–2.3); the published direction at each paper's own scale (BLT worse at 1B, MegaByte params-not-FLOPs, Scratchpad 2.9× vs pure byte, TST 2.5×). **MEDIUM** on: the codec-share computations from Table 10 (they assume f=4h and ignore cross-attention); the transfer of any of it to 95M/1 GB. **LOW** on: every expected delta in §4 (S1–S3) and the 1B batch arithmetic (S5). The confidence rating applies only to the VERIFIED part; the experiment forecasts are explicitly [S].
