# Long context at 95M: what this model can prove, and the cheapest honest protocol

**Date:** 2026-11-08 · **Mode:** read-only, CPU only; no GPU job, no code edit, no retraining.
**Scope:** the owner's question — *what can a 95M byte-level model actually prove about long
context, and what is the cheapest honest protocol?* — with the project's 1M design (KDA +
CSA fine 8-byte top-k + HCA coarse 256-byte, ~34 MB per CSA slot, segment carry).

**Tags.** **[T]** — the paper's own table/text, with arXiv id, version and table/figure.
**[C]** — arithmetic done here, formula printed. **[M]** — measured in this repo, path/log
named. **[S]** — SPECULATION: no direct measurement; the inference is named as such.
Every number below carries one of these; no number is untagged.

**The short answer.** The 95M checkpoint that exists on this disk **cannot make a 1M claim at
all**: it is a 12-layer, 1K-context, dense transformer with `use_csa=false`, and the 1M design
is a different block form (`LoopBlock`) that the dense form refuses. The literature has **no
long-context benchmark result at 95M**; the closest measured points are 300M (Infini-attention,
32K passkey), 0.5B (SRRM, 4K retrieval vs reconstruction), and 1.24B (Llama-3.2-1B in HELMET,
128K). At 95M the only defensible 1M claim is **exact retrieval at 1M**, not reasoning; the
cheapest honest protocol that could support it is four gates, 100 items, exact match and
same-weights controls, at roughly **3–12 GPU-hours**. It cannot start until a CSA-trained
checkpoint exists.

---

## 0. TL;DR — вердикт по-русски

1. **На диске нет модели, которая может пройти 1M-тест.** `checkpoints/flagship3` — это
   95 035 200 параметров, 12 слоёв, `d_model=832`, `max_seq_len=1024`, `use_csa=false`,
   `block_form="transformer"`; обучение — 245,76 МБ, последний eval — **BPB 1.5185** на
   196 608 Б, но это **192 независимых окна по 1 КиБ**, а не 196-килобайтный контекст.
   [M] `checkpoints/flagship3/flagship3.config.toml`; `~/logs/flagship3.jsonl`.
2. **1M-дизайн — это другая архитектура.** `use_csa` — флаг KDA-loop-формы; на
   `block_form="transformer"` он **отвергается** (`_KDA_ONLY_FORM_FLAGS`,
   `port/src/dormouse/config.py:3228-3235`). Ближайший обученный CSA-артефакт — `longctx1`
   (102,7M, KDA+CSA+HCA, 8K, 523 МБ, 512K BPB 1.7634/1.6882), но его checkpoint на диске
   **отсутствует**. [M] `~/logs/longctx1.sh`, `~/logs/longctx1.log`; каталога
   `checkpoints/longctx1` нет.
3. **Литература не даёт 1M-результата при 95M.** Ближайшие измеренные точки: **300M**
   Infini-attention — passkey 32K: 48/0/0/0/1 по глубинам, zero-shot за 4K почти нуль
   [T, 2512.23862v1, Table 1]; **0.5B** Qwen2.5 в SRRM — retrieval 65,1 %, reconstruction
   29,7 % на 4K [T, 2611.05386v1, Tables 14/16]; **1,24B** Llama-3.2-1B в HELMET — ∞Bench
   2,0, HELMET 21,2/24,6 (base/instruct), JSON KV 34,0, MSMARCO 0,3 на 128K [T, 2410.02694v3, Tables 6/8/9].
   Ни одной статьи с long-context benchmark на 95M в этом проходе не найдено.
4. **Байтовые модели тоже не дают 1M.** MegaByte-1B измерен на **8 192 байтах** (PG19 test
   BPB 1.000 / PPL 36.4) [T, 2305.07185v2, Tables 2–3]; MambaByte-972M — те же 8 192 Б
   (PG19 PPL 33.0) [T, 2401.13660v1, Table 3]; BLT 1B/8B не оценивает 1M-retrieval
   [T, 2412.09871v1, Table 10/§5]. 1M / 8K = **128×** дальше самого длинного измеренного
   байтового контекста.
5. **Retrieval-only evidence стоит ровно одного: механизмной заявки.** NIAH-проход на 1M
   доказывает, что селектор CSA нашёл 8-байтовый ключ и модель его скопировала; он **не**
   доказывает понимание. HELMET: корреляция NIAH с реальными задачами ρ = 0,44–0,76, среднее
   0,68, ни одна synthetic-задача не выше 0,8 [T, 2410.02694v3, Figure 3]; Llama-3.1-8B:
   NIAH 100,0 vs HELMET 47,0 [T, Table 5]; NoLiMa: 11 из 13 моделей падают ниже 50 % от
   short-context базы на 32K [T, 2502.05167v3, abstract/Table 3].
6. **Честный протокол — 4 гейта для retrieval (+ опциональный E для reasoning), 100
   айтемов, exact match, без LLM-judge.** Artifact → state-carry → rank probe → retrieval +
   controls. Проход: **≥95/100** exact на 1M (Wilson 95 % lower bound 88,8 %), no-needle
   ≤3/100 (upper 3,7 %), CSA-off ≤10/100, carry ≤1e-3 (bf16) / 1e-5 (fp32), argmax 100 %.
   Оценка стоимости: **3–12 GPU-часов** на одной карте; формула и попилотный порог — §4.5.
   [C]
7. **Три заявки по убыванию защищаемости:** (1) exact retrieval at 1M — единственная
   честная 1M-заявка; (2) constant-state 1M (396 MiB на 12 слоёв, 33 MiB на слой) —
   инженерная, не про интеллект; (3) reasoning at 1M — **не защищаема** при 95M, ожидание
   около нуля. [C]/[S]
8. **«34 MB total state» — неверно для 12 слоёв.** 33 MiB — это один CSA-модуль (один слой,
   один iteration slot); 12 слоёв дают **396 MiB** (bf16 ring) или 792 MiB (fp32), если CSA
   стоит в каждом слое. [C] `state_bytes(1M, 64, 8) = 131 072 × 2 × 64 × 2 = 33 554 432 B`;
   `+ coarse 1 048 576 B`; ×12 = 396 MiB. Это надо исправить в заголовке 1M-заявки.

---

## 1. Аудит артефакта: что именно лежит на диске

The question "what can this model prove" is an artifact question first. The repo has three
different 100M-class models; two are on disk, and neither on-disk model is a 1M model. The
only trained CSA artifact is the third one, and its checkpoint is missing.

| artifact | config (source) | training context | CSA | bytes seen / eval | verdict |
|---|---|---|---:|---|---|
| **flagship3** (95M) | `d_model=832`, `n_layers=12`, `max_seq_len=1024`, `block_form="transformer"`, `use_kda=false`, `use_csa=false` [M, `checkpoints/flagship3/flagship3.config.toml:2,7,9,10,38,137`] | **1 024 B** | **none** | 245 760 000 B at step 10k; BPB **1.5185** over 196 608 B = 24×8×1024 = 192 windows of 1K [M, `~/logs/flagship3.jsonl`; `~/logs/flagship3.log:31`] | **Cannot run 1M.** No CSA, no KDA, RoPE at 1K, full attention. Any 1M NIAH here measures a 1K model plus a randomly initialized branch (if one were bolted on). |
| **med1** (102.7M) | `d_model=1024`, `max_seq_len=1024`, `use_kda=true`, no `use_csa` [M, `port/checkpoints/medium/med1.config.toml`] | 1 024 B | none | — | A 1K KDA/Engram model, not the 1M design. |
| **longctx1** (102.7M) | `use_csa=true`, `csa_chunk=8`, `csa_coarse_chunk=256`, `seq_len=8192`, 16k steps [M, `~/logs/longctx1.sh`] | 8 192 B | **trained** | 523 MB; 512K BPB 1.7634 headline / 1.6882 fin [M, `docs/records/reviews/2026-10-27-longctx-night.md:3,9,33`] | The closest real CSA artifact; **checkpoint not on disk** (`checkpoints/longctx1/` missing). |
| **95M + 1M design** | no config; `use_csa` is refused on the dense form [M, `port/src/dormouse/config.py:3228-3235`] | — | — | — | **Does not exist.** The 1M design lives in `LoopBlock` (`flagship_base`, `m400b`), not in the 12-layer dense control. |

**Вердикт (RU).** На сегодня 1M-заявку нельзя измерить ни на одном чекпоинте: 95M — это
1K-плотный контроль без CSA; 102,7M `longctx1` — единственный обученный CSA-артефакт, но его
файла нет; 95M-с-CSA не существует как форма. Поэтому любой «1M NIAH» на `flagship3` был бы
измерением другой сети. Первый честный шаг — либо восстановить `longctx1`, либо обучить
95M/105M CSA-модель на ≥8K; это уже не «inference only», и это надо сказать владельцу прямо.

Two further artifact facts that change the interpretation of every number:

- **The 1.49 BPB in the brief is not the number in the log.** The log's last eval is
  **1.5185** over 196 608 B at 245.76 MB [M, `~/logs/flagship3.jsonl`]. A 1.49 needs its own
  checkpoint/window; I cannot verify it from the artifact. [M]/[S]
- **The 196 608 B eval is not a 196K context.** `eval_bytes=196608`, `batch=8`, `seq_len=1024`,
  so `eval_batches_for = 196608/(8×1024) = 24` batches; the model scores 192 independent
  1K windows [M, `port/src/dormouse/train/data.py:870-889`; log line "24 x 8 x 1024"]. It is
  a 1K-context BPB averaged over 196 KB, not a long-context number.

---

## 2. What the literature says a small model can do at long context

The table separates **measured size** from **claimed size** and **measured context** from
**claimed context**. No row is a 95M model; the closest measured points are 300M, 0.5B and
1.24B.

| # | candidate (id/version) | measured size | measured context | key metric | task | credibility / effort |
|---:|---|---:|---:|---|---|---|
| 1 | **RULER** `2404.06654v3` (COLM 2024) | 7B–104B, 17 models | 4K–128K tokens | Llama3.1-8B 95.5 → 87.4 (32K) → **77.0** (128K); Mistral-v0.2-7B 93.6 → 75.4 (32K) → **13.8** (128K); NIAH near-perfect while RULER drops [T, Table 3] | retrieval + multi-hop tracing + aggregation + QA | high (NVIDIA, 13 tasks, 17 models); reading is cheap, reproducing is expensive |
| 2 | **LongBench** `2308.14508v2` (ACL 2024) | 6B–7B (ChatGLM2-6B, Llama2-7B, …) | ~8K, max 16K tokens | ChatGLM2-6B-32k overall **41.4**; Llama2-7B-chat-4k **26.8** [T, Tables 2–3] | single/multi-doc QA, summarization, few-shot, code | high; not a 100K+ benchmark |
| 3 | **∞Bench** `2402.13718v2` (ACL 2024) | 7B/13B + proprietary | ~200K tokens average | YaRN-Mistral-7B average **19.96**; PassKey **92.71**, Number 56.61, **KV 0.00**, En.Sum 9.09, En.QA 9.55 [T, Table 3] | retrieval + reasoning + code + math + novel | high; the retrieval subtasks already diverge by 90 points |
| 4 | **HELMET** `2410.02694v3` (ICLR 2025) | **1.24B** Llama-3.2-1B; 59 models total | 128K tokens | Llama-3.2-1B: ∞Bench **2.0**; HELMET rank **21.2 base / 24.6 instruct**; JSON KV **34.0**, NQ 25.1, **MSMARCO 0.3**, MC 23.0, QA 18.9 [T, Tables 6/8/9] | retrieval + RAG + ICL + re-rank + citation + summarization | high; the closest 1B-class full benchmark |
| 5 | **NoLiMa** `2502.05167v3` (ICML 2025) | 7B–405B, 13 models | 32K tokens | Llama-3.1-8B base **76.7** → **14.2** at 32K; Command R+ 90.9 → 7.4; GPT-4o 99.3 → 69.7; **11/13 below 50 % of base at 32K** [T, Table 3, abstract] | associative retrieval without literal match | high; the cleanest "literal NIAH is a shortcut" result |
| 6 | **LongBench v2** `2412.15204v1` | 7B–70B+ | 8K–2M words | Llama-3.1-8B **30.0**; GPT-4o 50.1; o1-preview 57.7; human 53.7; random **25** [T, Table 2] | deep understanding / reasoning MCQ | high; 503 questions, 6 categories |
| 7 | **Lost in the Middle** `2307.03172v3` (TACL 2024) | 13B–30B + proprietary | 2K–16K tokens | U-shaped curve: best at start/end, worst in the middle; 75/140/300 key-value pairs [T, Figures 1/5/7, §3.1] | retrieval / QA | high; position bias is a separate failure axis |
| 8 | **SRRM** `2611.05386v1` | **0.5B / 1.5B / 3B** Qwen2.5 | **4 096 tokens**, 10–100 facts | 0.5B retrieval mean **65.1 %**, reconstruction mean **29.7 %**; at 100 facts 40.0 / **4.0**; 1.5B 86.7/45.7; 3B 72.2/18.5 [T, Tables 14/16] | targeted retrieval vs global reconstruction | low-medium (single-author preprint); the retrieval/reconstruction gap is the useful part |
| 9 | **Infini-attention small-scale** `2512.23862v1` | **300M** LLaMA | 32K tokens | zero-shot passkey: perfect <2K, 1/0/0/3/0 at 4K, **0** at 8K/16K, 0/0/0/0/1 at 32K; after FT at 16K **45/0/0/0/0**, at 32K **48/0/0/0/1** [T, Table 1] | passkey retrieval | medium (Huawei, arXiv); the smallest measured long-context retrieval |
| 10 | **HyLo / Long-Context Aware Upcycling** `2604.24715v1` | **1.24B** Llama-3.2-1B | 64K tokens | baseline MambaInLlama-1B: RULER 8K/16K/32K/64K = **18.9/3.0/1.0/0.0**; HyLo with 64K training = **61.5/53.7/48.1/41.6**; with 8K training = 60.3/0.5/0.1/0.1 [T, Table 2] | RULER (retrieval + reasoning) | medium-high (arXiv 2026); shows training context, not architecture alone, is the lever |
| 11 | **MegaByte** `2305.07185v2` (ICML 2023) | 1.02B (758M global + 262M local) | **8 192 bytes** | PG19 test BPB **1.000**, word PPL **36.4**; title says million-byte, measurement is 8K [T, Tables 2–3] | byte-level LM, no retrieval | high; the closest byte-level 1B measurement |
| 12 | **MambaByte** `2401.13660v1` (NeurIPS 2024) | 972M | **8 192 bytes** | PG19 test PPL **33.0**; 353M variant PG19 BPB 0.930 [T, Tables 2–3] | byte-level LM, no retrieval | high; same 8K wall |
| 13 | **BLT** `2412.09871v1` (Meta) | 1B / 8B | not 1M; byte-patch scaling | 1B/8B scaling study; no 1M NIAH/RULER [T, Table 10, §5] | byte-level LM | high; does not answer the 1M question |

**The closest points to 95M.** In measured long-context retrieval, **300M** (Infini) is 3.2×
larger; in measured retrieval-vs-reconstruction, **0.5B** (SRRM) is 5.3× larger but only 4K;
in measured RULER/HELMET, **1.24B** (HyLo/HELMET) is 13× larger and 64K–128K. There is **no
published long-context benchmark at 95M** in the papers read here. This is a negative search
result, not a proof of absence; it is still the honest state of the evidence. The only
small-scale 1M sequence result found is Mamba on DNA: 1.3M–1.4M parameters, 1M DNA bases,
pretraining perplexity improving with context [T, `2312.00752v2`, §4.3.2] — not a language or
retrieval result, so it does not transfer to this question.

**What the 1B ceiling looks like.** At 1.24B, **without long-context training**, Llama-3.2-1B
scores 2.0 on ∞Bench and 21.2/24.6 on HELMET at 128K (base/instruct) [T, 2410.02694v3, Tables 6/9]. **With**
long-context training, HyLo lifts the same backbone to 41.6 RULER at 64K [T, 2604.24715v1,
Table 2]. That is the size-adjusted ceiling: a 1B model can reach ~40 RULER at 64K if trained
for it, and is near-floor without it. A 95M model is 13× smaller than that ceiling; a 1M
context is 16× longer than 64K. The two extrapolations do not point to a reasoning claim.

---

## 3. What retrieval-only evidence is worth

A 1M NIAH pass is **not a toy at 95M**: it is the first direct evidence that the CSA selector
and the segment carry work at the target distance, and it is falsifiable with same-weights
controls. It is also **not a capability claim**: the literature is consistent that NIAH is a
necessary but weak predictor of real long-context behaviour.

| claim | evidence | number | source |
|---|---|---|---|
| NIAH is nearly saturated at 7B+, while RULER collapses | RULER: 17 models, 13 tasks; NIAH near-perfect, RULER drops; only half maintain at 32K | Llama3.1-8B 95.5 → 77.0 at 128K; Mistral-7B 93.6 → 13.8 | [T] `2404.06654v3`, Table 3, abstract |
| NIAH correlates weakly with real tasks | HELMET: Spearman ρ across 35 instruction-tuned models; no synthetic task >0.8 | NIAH ρ = **0.44–0.76**, mean **0.68**; Llama-3.1-8B NIAH **100.0** vs HELMET **47.0** | [T] `2410.02694v3`, Figure 3, Table 5 |
| Literal matching is the shortcut | NoLiMa removes lexical overlap; 11/13 drop below 50 % of base at 32K | Llama-3.1-8B 76.7 → **14.2**; Command R+ 90.9 → 7.4; GPT-4o 99.3 → 69.7 | [T] `2502.05167v3`, Table 3, abstract |
| Retrieval subtasks diverge wildly | ∞Bench: same 7B model, same 200K context | PassKey **92.71** vs KV **0.00**; average 19.96 | [T] `2402.13718v2`, Table 3 |
| NIAH is not deep understanding | LongBench v2: 503 MCQ, 8K–2M words; random 25 % | Llama-3.1-8B **30.0**; GPT-4o 50.1; o1-preview 57.7; human 53.7 | [T] `2412.15204v1`, Table 2 |
| Retrieval ≠ reconstruction even at 4K | SRRM: 0.5B/1.5B/3B, 10–100 facts | 0.5B: retrieval **65.1 %**, reconstruction **29.7 %**; 3B: 72.2 / 18.5 | [T] `2611.05386v1`, Tables 14/16 |
| Position matters independently of retrieval | Lost in the Middle: key-value retrieval at 75/140/300 pairs | U-shaped; middle is worst; same model, different position | [T] `2307.03172v3`, Figures 1/5/7 |

**Вердикт (RU).** На 95M NIAH-проход — это **механизмная** заявка: «селектор CSA находит
уникальный 8-байтовый ключ на расстоянии 1M, и модель его копирует». Это ровно то, что
проект обещает архитектурой, и это стоит измерять. Но это **не** заявка «модель понимает
1M»: при 1B NIAH 100 у Llama-3.1-8B сосуществует с HELMET 47; при 300M passkey на 32K
держит только нулевая глубина; при 0.5B retrieval 65 % сосуществует с reconstruction 30 %.
Поэтому в любой формулировке 1M-заявки должно стоять слово **retrieval**, а не «context
understanding», «reasoning» или «long-context ability».

One nuance that matters for honesty: the "NIAH is saturated" critique is a **7B+** statement.
At 95M, NIAH is not saturated — Llama-3.2-1B gets 34.0 on JSON KV and 2.0 on ∞Bench at 128K
[T, `2410.02694v3`, Tables 6/8]. So a 95M NIAH pass is a **real** capability, not a trivial
one; it is simply a narrow one. The right claim is "exact retrieval at 1M", not "NIAH is
meaningless" and not "long-context understanding".

---

## 4. The cheapest honest protocol that is not a toy

The protocol is written for **the model the owner thinks they have**: a 95M-class byte-level
model with a trained CSA fine level (8-byte chunks, top-k 32) and HCA coarse (256-byte), with
segment carry. It is **pre-registered**: the pass rules and the stop conditions are fixed
before the numbers exist. The first gate is the artifact gate; if it fails, the protocol stops
and the honest answer is "no 1M claim".

### 4.1 Task: S-NIAH-8, an exact 8-byte copy

- **Haystack:** 1 048 576 bytes of held-out natural English, one disjoint window per item.
- **Needle:** a 16-byte random cue `c` followed by an 8-byte random key `k`, aligned to an
  8-byte boundary so the key is exactly one fine chunk. Both are drawn from printable ASCII
  (64-symbol alphabet); `k` is unique per item.
- **Query:** the same 16-byte cue `c` at the end of the 1M window. The target is the 8 bytes
  that followed the first `c`.
- **Score:** teacher-forced argmax at the 8 target positions; **exact match of all 8 bytes**.
  No sampling, no judge, deterministic.
- **Distances:** d = 1K (positive control), 64K, 256K, 1M. The 1M point is the claim; the
  shorter points are the curve that says whether a 1M failure is specific to 1M.

Why this task and not "The key is: ..." instruction following: the model is a byte-level
next-byte predictor trained on raw text, not an instruction-tuned chat model. A cue→copy task
is native to next-byte prediction and does not require instruction following. It is also exactly
the task the CSA mechanism is supposed to solve: one unique 8-byte chunk among 131 072.

### 4.2 Gates and pass rules

| gate | what is measured | n / cost | pass rule | stop rule |
|---|---|---|---|---|
| **A. Artifact** | checkpoint has CSA weights; gate live; training context ≥ 8K | 1 load + 1 forward; ~1 min | `use_csa=true`, `|sigmoid(csa_gate)| > 0.01` in ≥1 layer, `max_seq_len ≥ 8192` [M] | if not, **stop**: no 1M claim is measurable |
| **B. Carry** | 16K one-shot vs 4×4K segments; 1M 4K vs 8K segments; last 4K logits | 3 passes; 2–5 min | 16K: max\|Δ\| ≤ 1e-5 fp32 or argmax 100 %; 1M: max\|Δ\| ≤ 1e-3 bf16 and argmax 100 % | if fail, the 1M number is a different function; **stop** |
| **C. Rank probe** | rank of the needle chunk in the **flat** fine score row and in the **hierarchical** candidate set, per layer/query, at 1M | 10 items; 5–10 min | flat best-layer rank ≤ 32 in ≥ 90 % of items; hierarchical rank ≤ 32 in ≥ 90 %; median ≤ 32 | if flat rank is chance (~65 536), the selector is the bottleneck; if flat ≤32 but hierarchical >32, the hierarchy is; report both, do not run 100 items |
| **D. Retrieval** | exact 8-byte match at d=1M, plus controls | 100 items + controls; 2–10 h | **≥95/100** exact; per-byte ≥ 99 %; controls below their bars | if <95, claim the measured number, not "1M retrieval" |
| **E. Reasoning (negative)** | RULER VT (2 chains × 2 hops) and CWE (K=3) at 1M, exact set match | 100 items; 1–3 h | ≥50/100 to claim *any* reasoning | if ~0, report "retrieval-only"; do not claim reasoning |

The carry gate is deliberately split: at 16K a one-shot pass fits (16K × 2K chunks = 32M
scores = 128 MiB), so the "2 segments == 1 pass" equality can be tested directly; at 1M a
one-shot fine score is **512 GiB** [C: 1 048 576 × 131 072 × 4 B], so the 1M check is
4K-segmented vs 8K-segmented, not vs one-shot. The carry contract is length-independent, but
the 1M check is a consistency check, not an identity proof. Say that in the claim.

### 4.3 Controls

| control | what it rules out | n | expected / bar |
|---|---|---:|---|
| **Positive d=1K** | the task is unlearnable at short context | 100 | ≥95/100; if <95, stop and redesign |
| **No-needle** | hallucinating a key from the cue | 100 | ≤3/100 (Wilson upper 3.7 %) |
| **Shuffled-key** | following the nearest key instead of the cue | 100 | ≥95/100 on the cued key |
| **CSA-off** (same weights, gate 0) | the CSA branch is not load-bearing | 100 | ≤10/100; if it passes, the 1M claim is false |
| **Coarse-only** (`csa_fine=false`) | the fine 8-byte level is unnecessary | 100 | ≤50/100; if it passes, the exact-8-byte claim is weaker |
| **Flat selection** (`csa_hierarchical=false`) | the hierarchy is the bottleneck | 100 | ≥90/100; if it fails and flat passes, the hierarchy is the problem |
| **Flat dense 95M at 1K** | the task is learnable by a 95M model at all | 100 | ≥50/100; if it fails, the task is too hard and the 1M test is uninformative |

The **flat-attention control** needs a precise statement. The 95M `flagship3` is a 1K full-
attention transformer; at 1M its attention matrix would be 1 048 576² × 12 × 13 × 4 B ≈
6.9e14 B (690 TB) [C], so it cannot run at 1M. Its only honest role is the **positive control
at 1K**: it shows whether a 95M model can do the cue→copy task when the needle is inside its
training window. At 1M the mechanism control is the **same CSA model with the fine branch
off** (or the gate zeroed); a different flat model at 1M is not runnable and a sliding-window
flat model fails by construction (the needle is outside the window), which proves nothing.
This distinction must be in the claim: "flat-attention control at 1K" ≠ "flat-attention
control at 1M".

### 4.4 Sample size and the arithmetic of the pass rule

The pass rule is a confidence statement, not a point estimate. Wilson 95 % intervals [C,
`p ± z√(p(1−p)/n + z²/4n²) / (1+z²/n)`, z=1.96]:

| observed | n | Wilson 95 % | what can be claimed |
|---|---:|---|---|
| 90/100 | 100 | [0.826, 0.945] | "≥82 % exact" — **not** "≥90 %" |
| **95/100** | 100 | **[0.888, 0.978]** | **"≥88 % exact"** — the protocol's bar |
| 190/200 | 200 | [0.910, 0.973] | "≥91 % exact" — use if the owner wants a 90 % lower bound |
| 0/100 | 100 | [0.000, 0.037] | "≤3.7 %" for the no-needle control |
| 3/100 | 100 | [0.010, 0.085] | "≤8.5 %" |
| 10/100 | 100 | [0.055, 0.174] | "≤17.4 %" for CSA-off |
| 0/30 | 30 | [0.000, 0.114] | "≤11.4 %" — the cheap version if a 1M pass is slow |

The **chance rate** of the fine selector is the other half of the argument. At 1M there are
`ceil(1 048 576/8) = 131 072` fine chunks; top-k = 32; a random selector hits the needle chunk
with probability `32/131 072 = 2.44e-4` per query, and `1 − (1 − 2.44e-4)^12 = 0.0029` over 12
layers if independent [C]. So 95/100 is not chance by any reasonable model; even one hit is
p ≈ 2.4e-4 per layer. The control that matters is not "is it chance" but "is the branch
load-bearing" (CSA-off) and "is the task learnable at 1K" (positive control).

### 4.5 Cost arithmetic

| quantity | formula | value |
|---|---|---:|
| fine ring at 1M, one layer | `ceil(1 048 576/8) × 2 × 64 × 2 B` | **32 MiB** |
| coarse ring at 1M, one layer | `ceil(1 048 576/256) × 2 × 64 × 2 B` | **1 MiB** |
| CSA state, 12 layers | `12 × 33 MiB` | **396 MiB** (fp32 ring: 792 MiB) |
| KDA state, 12 layers | `12 × 13 × 64 × 64 × 4 B` | **2.6 MB** |
| one-shot 1M fine score | `1 048 576 × 131 072 × 4 B` | **512 GiB** → cannot one-shot |
| 4K-segment fine score | `4 096 × 131 072 × 4 B` | **2 GiB** → fits |
| 8K-segment fine score | `8 192 × 131 072 × 4 B` | 4 GiB → fits |
| 1M score FLOPs, 12 layers, causal half | `0.5 × 1 048 576 × 131 072 × 64 × 2 × 12` | **1.06e14 FLOP** |
| time at 5 / 20 TFLOP/s | `1.06e14 / 5e12` / `1.06e14 / 20e12` | **21 s / 5 s** per pass |
| memory traffic, 4K segments | `256 × 12 × 2 GiB × 2` (write+read) | **12.9 TB** → ~13 s at 1 TB/s |
| 100 passes at 30 s | `100 × 30 s` | **50 min** |
| 5 **1M** controls × 100 passes | `500 × 30 s` | **4.2 h** |
| 4 distances × 100 passes | `400 × 30 s` | **3.3 h** (upper bound; 1K/64K/256K are cheaper) |
| **total, planning number** | 100 + 500 + 400 passes | **3–12 h** on one 16 GB card |

The 1K positive control and the 1K flat-dense control are 1K forwards, not 1M forwards; they
are negligible in this budget. The "5 1M controls" are no-needle, shuffled-key, CSA-off,
coarse-only and flat-selection; all five run on the same 1M items, so they can reuse the
haystack and the tokenized input.

The range is dominated by the CSA score kernel's MFU and by the number of CSA layers. **Pilot
rule:** run one 1M pass first; if it takes > 2 min, cut the retrieval set to 30 items (0/30
upper bound 11.4 %) and report a weaker claim; if it takes < 30 s, use 200 items for the
"≥91 %" bound. The protocol never spends the full budget before the rank probe and the carry
gate have passed.

**Memory fit.** Peak = weights 190 MB (bf16) + CSA 396 MiB + 2 GiB score + activations
(~0.3–1.4 GiB at 4K segments) + KDA 2.6 MB ≈ **3.5–4.5 GiB**; fits 16 GB with ~3× headroom
[C]. The score tensor, not the state, is the peak; that is why the 4K segment is the right
unit. At 16K segments the score is 8 GiB and the headroom is gone; at 32K it does not fit.

**No LLM judge.** Every gate is exact match or exact set match. A judge is rejected for three
reasons: (1) a 1M-context judge is itself the capability under test; (2) the common 128K
judges truncate the 1M input; (3) at published 2024 list prices a 100-item 1M-token judge pass
is roughly $15 for a 128K model that cannot see the input, or $125–250 for a 1M-context model,
before the judge's own errors [S, list prices not re-verified]. The protocol deliberately
avoids the paid API.

### 4.6 What would kill the claim

- **Artifact gate fails:** no CSA weights, gate ≈ 0, training context < 8K → no 1M claim.
- **Carry gate fails:** 4K and 8K segmented passes disagree → the 1M number is not the model.
- **Rank probe is chance:** best-layer rank ~65 536 → the selector does not find the needle;
  the failure is architectural, not a readout issue.
- **CSA-off passes:** the branch is not load-bearing → the 1M claim is false even if the
  retrieval number is high.
- **Positive control fails at 1K:** the task is too hard for 95M → the 1M result is
  uninterpretable; redesign the task, do not claim 1M.
- **No-needle passes:** the model hallucinates the key → the task is not measuring retrieval.

---

## 5. What a 1M claim should say — three candidates, ranked

| rank | claim | exact wording | measurement that supports it | measurement that kills it | confidence | what it does **not** say |
|---:|---|---|---|---|---|---|
| **1** | **Exact retrieval at 1M** | **Conditional on Gate A:** "On 100 disjoint 1M-byte windows, a 95M CSA checkpoint retrieves a unique 8-byte key at distance 1 048 576 with 95/100 exact match (Wilson 95 % lower bound 88.8 %); the same weights with the CSA fine branch disabled score ≤10/100; the no-needle control scores ≤3/100." | Gate D + controls; rank probe best-layer ≤32 in ≥90/100 | <95/100, CSA-off passes, no-needle passes, carry fails | **high** for the measurement, **medium** for the extrapolation to other keys/corpora | reasoning, summarization, multi-hop, "understanding" |
| **2** | **Constant-state 1M** | "The model processes 1M bytes with 396 MiB of CSA state (33 MiB per layer × 12) and a 2 GiB peak score tensor at 4K segments; the 4K- and 8K-segmented passes agree to 1e-3 on the last 4K logits with 100 % argmax." | Gate B + VRAM ledger; §4.5 arithmetic | carry mismatch, OOM, state > 396 MiB | **high** (arithmetic + carry), but it is an **engineering** claim | quality, retrieval, intelligence |
| **3** | **Long-context reasoning at 1M** | "The model answers RULER VT/CWE at 1M with ≥50/100 exact set match, above the 1K control." | Gate E; 100 items; exact set match | ~0/100; at 95M the prior is near-floor | **low**; not defensible today | — |

**Вердикт (RU).** Честная 1M-заявка — **№1, exact retrieval at 1M**. №2 — это заявка про
память и сегментацию, полезная инженерно, но не про интеллект. №3 при 95M **не защищаема**:
ближайшие точки (300M Infini, 0.5B SRRM, 1.24B HELMET/HyLo) дают reasoning около нуля на
32K–128K; 1M в 8–32 раза дальше. Если модель не проходит №1, единственная оставшаяся 1M-
заявка — №2, и её надо называть именно так.

The numbers in the claim table are restatements of §4.4–4.5: 95/100 Wilson [0.888, 0.978]
[C]; 396 MiB = 12 × 33 MiB [C]; 2 GiB at 4K [C]; 50/100 is the pre-registered reasoning bar,
not a measurement [S]. The claim's evidence tags are the ones in §4.

### What the 1M claim should say

- **Name the task and the distance:** "exact retrieval of a unique 8-byte key at 1 048 576
  bytes", not "1M context".
- **Name the controls:** CSA-off, no-needle, positive 1K, flat selection.
- **Name the confidence:** 95/100 and the Wilson lower bound; 0/100 and the upper bound.
- **Name the training context and the extrapolation:** "the model was trained at X bytes; 1M
  is an extrapolation of the content-based selector, not of the LM's training distribution."
- **Name the state per layer and total:** 33 MiB per CSA slot, 396 MiB for 12 layers.
- **Say what it is not:** "This is exact retrieval, not reasoning."

### What the 1M claim should not say

- **"Supports 1M context"** without a task and a score. Memory is not capability.
- **"Understands 1M" / "long-context reasoning at 1M"** — no measurement supports it at 95M.
- **"34 MB total state"** — 34 MB is one slot; 12 layers are 396 MiB. The comparison to V4.1's
  890 MB/token must use the total, or say "per layer".
- **"One-pass 1M"** — the one-shot fine score is 512 GiB; the claim is segmented.
- **"Beats 7B/8B models at 1M"** — different size, training context, tokenizer and task; no
  such comparison is run.
- **"NIAH at 1M proves long-context ability"** — HELMET/NoLiMa/LongBench v2 say otherwise.
- **"Passes at 1M"** from a multiple-choice task with 25 % chance — use exact match.

---

## 6. VERIFIED (sourced) vs SPECULATION (my inference)

### VERIFIED

- **No 95M long-context benchmark exists in the papers read.** The closest measured sizes are
  300M (Infini, 32K passkey), 0.5B (SRRM, 4K), 1.24B (HELMET/HyLo, 128K/64K). [T]
  `2512.23862v1` Table 1; `2611.05386v1` Tables 14/16; `2410.02694v3` Tables 6/8/9;
  `2604.24715v1` Table 2.
- **At 1B, long-context training is the lever:** Llama-3.2-1B baseline RULER-64K 0.0; HyLo
  with 64K training 41.6. [T] `2604.24715v1` Table 2.
- **NIAH is a weak predictor:** HELMET ρ 0.44–0.76, mean 0.68; Llama-3.1-8B NIAH 100 vs
  HELMET 47; NoLiMa 11/13 below 50 % at 32K. [T] `2410.02694v3` Figure 3/Table 5;
  `2502.05167v3` Table 3/abstract.
- **The 95M checkpoint is 1K dense, no CSA:** `max_seq_len=1024`, `use_csa=false`,
  `block_form="transformer"`, BPB 1.5185 over 192×1K windows. [M]
  `checkpoints/flagship3/flagship3.config.toml`; `~/logs/flagship3.jsonl`.
- **The 1M design is a different block form:** `use_csa` is refused on `block_form=
  "transformer"`; the closest CSA artifact `longctx1` is not on disk. [M]
  `port/src/dormouse/config.py:2873, :3228-3235`; `~/logs/longctx1.sh`; missing
  `checkpoints/longctx1/`.
- **State arithmetic:** 33 MiB per CSA slot at 1M; 396 MiB for 12 layers; 512 GiB one-shot
  fine score; 2 GiB at 4K segment; 1.06e14 FLOPs per 1M pass. [C] §4.5 formulas.
- **Pass-rule arithmetic:** 95/100 Wilson [0.888, 0.978]; 0/100 upper 3.7 %; top-k chance
  2.44e-4 per query. [C] §4.4.

### SPECULATION (my inference, no direct measurement)

- **A 95M CSA model would retrieve at 1M.** No such model exists on disk; the 300M Infini
  result collapses beyond 8K, so the prior is not favourable. [S] based on [T]
  `2512.23862v1` Table 1 and [C] the 1M/8K gap.
- **The protocol costs 3–12 GPU-hours.** The FLOPs and memory traffic are computed, but the
  CSA kernel's MFU and the per-segment Python overhead are not measured; the pilot decides.
  [S] based on [C] §4.5.
- **A 1M reasoning gate would score ~0/100 at 95M.** The closest measured reasoning at 1B is
  near-floor; 95M is 13× smaller and 16× longer. [S] based on [T] `2410.02694v3` Table 6 and
  `2604.24715v1` Table 2.
- **The 1.49 BPB in the brief is a different checkpoint/window.** The log says 1.5185 at
  245.76 MB; I cannot verify 1.49. [S] based on [M] `~/logs/flagship3.jsonl`.

**Confidence.** High for the literature numbers and the artifact audit; medium for the cost;
low for the 95M 1M retrieval outcome. The confidence rating applies **only** to the verified
part; the speculation is labelled and not used as a premise for a claim.

---

## 7. Sources

**Papers (id, version, table/figure used).**

1. `2404.06654v3` — RULER: What's the Real Context Size of Your Long-Context Language Models?
   COLM 2024. Table 3, abstract, §3.1–3.3.
2. `2308.14508v2` — LongBench: A Bilingual, Multitask Benchmark for Long Context
   Understanding. ACL 2024. Tables 2–3.
3. `2402.13718v2` — ∞Bench: Extending Long Context Evaluation Beyond 100K Tokens. ACL 2024.
   Tables 1–3.
4. `2410.02694v3` — HELMET: How to Evaluate Long-Context Language Models Effectively and
   Thoroughly. ICLR 2025. Tables 5, 6, 8, 9; Figure 3; §3.1.
5. `2502.05167v3` — NoLiMa: Long-Context Evaluation Beyond Literal Matching. ICML 2025.
   Table 3, abstract.
6. `2412.15204v1` — LongBench v2: Towards Deeper Understanding and Reasoning on Realistic
   Long-context Multitasks. Table 2, abstract.
7. `2307.03172v3` — Lost in the Middle: How Language Models Use Long Contexts. TACL 2024.
   Figures 1, 5, 7; §3.1.
8. `2611.05386v1` — SRRM: A Retrieval-Reconstruction Benchmark for Evaluating Long-Context
   Memory in Small Language Models. Tables 13–16, abstract.
9. `2512.23862v1` — Probing the Limits of Compressive Memory: A Study of Infini-Attention in
   Small-Scale Pretraining. Table 1, abstract.
10. `2604.24715v1` — Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling.
    Table 2, §3.
11. `2305.07185v2` — MEGABYTE: Predicting Million-byte Sequences with Multiscale
    Transformers. ICML 2023. Tables 2–3.
12. `2401.13660v1` — MambaByte: Token-free Selective State Space Model. NeurIPS 2024.
    Tables 2–3.
13. `2412.09871v1` — Byte Latent Transformer: Patches Scale Better Than Tokens. Table 10,
    §5.
14. `2312.00752v2` — Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
    §4.3.2 (1M DNA, 1.3M params; not a language/retrieval result).

**Project files.**

- `checkpoints/flagship3/flagship3.config.toml` — 95M dense 1K config (`d_model=832`,
  `n_layers=12`, `max_seq_len=1024`, `use_csa=false`).
- `~/logs/flagship3.jsonl`, `~/logs/flagship3.log:31` — BPB 1.5185 over `24 x 8 x 1024`.
- `~/logs/longctx1.sh`, `~/logs/longctx1.log` — 102.7M CSA run, 8K, 16k steps; checkpoint
  missing.
- `port/src/dormouse/model/csa.py:186-220` — `state_bytes` / `total_state_bytes` formulas;
  `:775-805` — parent-mask fix (math audit §6).
- `port/src/dormouse/config.py:2873, :3228-3235` — `_KDA_ONLY_FORM_FLAGS` refuses `use_csa` on
  `block_form="transformer"`.
- `port/src/dormouse/train/data.py:870-889` — `eval_batches_for` = window / (batch × seq).
- `docs/records/reviews/2026-11-08-segment-csa.md:113-118` — 2.0 GiB score at 4K; "34 MB is one
  slot"; 66 MiB × 3 = 198 MiB for one loop block.
- `docs/records/reviews/2026-11-08-math-audit.md` §6 — hierarchical CSA parent mask; scratch fix
  verified to 3.0e-8; repo now carries the fix at `csa.py:775`.
- `docs/records/reviews/2026-10-27-longctx-night.md:3,9,33` — longctx1 102.7M, 512K BPB 1.7634/1.6882.
- `docs/records/reviews/2026-11-05-context-ab.md:8-12` — 8K vs 1K tie, Δ = 0.008 BPB.
- `docs/records/architecture/2026-11-08-the-design.md:19,39,50-51` — "34 MB" headline; needs the
  per-layer/total correction.

**Method.** arXiv API + PDF extraction (`pdftotext -layout`) of the papers above, read for
tables and model sizes; repo docs, configs and logs read directly. Negative search: no arXiv
paper in this pass reports a long-context benchmark at 95M; no byte-level paper reports a 1M
retrieval at ≤1B.

---

## 8. Open questions

1. **Does a 95M CSA checkpoint exist anywhere?** `longctx1` is 102.7M and its file is missing;
   `flagship3` is 1K dense. Without one, the protocol's Gate A fails and no 1M claim is
   measurable. The first decision is restore vs retrain.
2. **Does the CSA selector's top-k generalize from 1K to 1M?** The content-based score is
   position-agnostic, but the score distribution at 131 072 chunks is not the 1K distribution.
   The rank probe answers this in 10 items; it is the cheapest informative measurement.
3. **Is the hierarchical parent-mask fix tested at 1M?** The fix is in `csa.py:775-805`; the
   tests are at 128–600 bytes and 1e-5. The 1M carry gate would be its first large-scale
   check.
4. **Does the 1M forward fit with fine-on?** The arithmetic says 3.5–4.5 GiB at 4K segments;
   the 2 GiB score tensor is the peak. A pilot is needed before the 100-item run.
5. **What is the model's training context?** The 95M is 1K. A 1M claim on a 1K-trained model
   is an extrapolation of the selector, not of the LM. The claim must say so.
6. **Is a 1M BPB a useful number?** The segmented forward can compute it, but it is a language-
   modeling metric, not a capability metric; it would be dominated by the 1K training
   distribution and would not support a 1M claim either way.

---

*End of report. The honest 1M claim is exact retrieval at 1M, not reasoning. The current 95M
checkpoint cannot make it; the first step is a CSA-trained 95M/105M artifact, then the four
gates above.*
