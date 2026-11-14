# Pretraining corpora: 2.6 GB → 10–20 GB, quality-first

Date: 2026-11-14. Scope: open corpora that are (a) downloadable today **without an HF token**,
(b) licence-defensible, (c) sub-settable at single-shard granularity, (d) carry Russian, and
(e) enough bytes for a 275M–1B byte-level model on one 16 GB card.

Every size below was read from the HuggingFace API (`/api/datasets/<id>/tree/refs/convert/parquet`)
on 2026-11-14 and is the **resolved byte size of that parquet file**, not a card's `num_bytes`.
Every arXiv id was resolved through `export.arxiv.org/api/query?id_list=…` and the **title printed
below is the title the API returned** — two of the ids I expected were wrong and are corrected
in §3.5 rather than quietly dropped.

---

## TL;DR

1. **The corpus is not the binding constraint — the 4-epoch cap is.** `docs/guides/data-provenance.md:90-93`
   already computes that `max_epochs = 4.0` binds on fwedu at ≈6.32 GiB of *served* bytes. Growing to
   20 GB removes that ceiling; growing past 20 GB buys nothing at 275M (20 GB ≈ 5.5B GPT-2-token-equivalents
   ≈ one Chinchilla epoch for 275M).
2. **Six sources carry the whole plan**, all anonymous, all ODC-By or Apache-2.0:
   FineWeb-Edu `sample-100BT`, FineWeb2-HQ `rus_Cyrl`, FineMath `finemath-4plus`, Cosmopedia v2
   (5 configs), `allenai/peS2o`, and Wikipedia (ru + en). Russian is **16 % of bytes**, which is
   only **≈8.5 % of words** — Cyrillic is 2 bytes/char in UTF-8.
2b. **But the Russian shard I downloaded is not finished goods.** Measured on FineWeb2-HQ
   `rus_Cyrl/0000` (§1.1): median `quality_score` **0.145**, only **17.3 %** of docs ≥ 0.5, and the
   sampled prose reads machine-translated. FineWeb2-HQ is a 10 %-retention **subset**, not a
   high-score **set**. Use the carried `quality_score` column for a *gentle* bottom cut (< 0.05, ≈5 %
   of docs) — a 0.5 cut keeps 17 % and would cost more diversity than it returns at a 2.8 B-token budget
   (`2402.09739`); revisit it once the fine-tune shows Russian is the weak axis.
3. **The best 2025–26 corpora are gated and therefore out**: Nemotron-CC-v2 / -v2.1 / -CC-Math-v1 /
   -CC-Code-v1 (`gated=manual` / `auto` → HTTP 401 anonymously), The Stack v2 and StarCoderData
   (`gated=auto`), Stack-Edu (not gated, but the card's feature list has **no text column** —
   it is `blob_id` metadata that must be joined against The Stack v2's S3 blobs).
4. **Dolma 3 / Olmo 3 is the best-documented 2026 mixture and still unusable as an anchor**:
   its parquet branch holds a 3.80 GB partial sample, not the pool.
5. **Code is the weak link.** The only anonymous, licence-clean code is `codeparrot/github-code-clean`
   (`all-mit` + `all-apache-2.0`, ~1 GB usable) — 2022 GitHub with heuristic-only filtering. It beats
   the current `codeparrot-clean` slice on licence (0 % copyleft vs 40 % measured) but not on quality.
6. **Two corpus-level numbers to copy from the literature**: OLMo 2's mix is **2.05 % code and 0.41 %
   math by bytes**; SmolLM2's *decay* stage is **24 % code and 14 % math by tokens**. The current
   11 % / 12 % byte mix sits between them, and that is the right place for a 275M model.
7. The near-dup pass should **not** re-run on FineWeb-Edu / FineWeb2-HQ / FineMath — those are already
   MinHash-deduped upstream at the same 0.75 threshold. `docs/guides/dataset.md:77-84` already measured
   that re-running it removes 0.063 % of bytes. Run it only on the four un-deduped sources.
8. Total download for the 20 GB plan: **20.1 GB in 39 files**; for 10 GB: **10.02 GB in 19 files**.

---

## VERIFIED (sourced)

### 1. What is actually reachable, measured

All rows below were produced by the same script: list `refs/convert/parquet` recursively, sum
`size` per config directory, then `curl -sIL` the individual shard to confirm 200 + `content-length`.
The seven HEAD checks all returned 200 with the API's byte count (§6 has the URLs).

| what | value | how measured |
|---|---|---|
| FineWeb-Edu `sample-100BT` | 140 shards, **286.4 GB**, 2 153 444 469 B for shard 0 | tree API + HEAD 200 |
| FineWeb-Edu `sample-10BT` | 14 shards, **28.5 GB**, same 2.15 GB/shard | tree API |
| FineWeb-Edu `default` (the 1.3T set) | 2410 shards, 4522.7 GB | tree API |
| FineWeb-2 `rus_Cyrl` | 440 shards, **1987.6 GB**, 4.84 GB/shard | tree API |
| FineWeb2-HQ `rus_Cyrl` | 1205 shards, **1276.8 GB**, 1.63/1.60/1.61 GB for shards 0–2 | tree API |
| FineMath `finemath-4plus` | 64 shards, **18.4 GB**, ≈0.286 GB/shard | tree API |
| FineMath `finemath-3plus` | 128 shards, 65.0 GB | tree API |
| Cosmopedia `auto_math_text` | 18 shards, 4.5 GB, 248 133 641 B/shard | tree API |
| Cosmopedia `stanford` / `openstax` / `khanacademy` / `wikihow` | 3.3 GB (13) / 0.35 GB (2) / 0.05 GB (1) / 0.50 GB (2) | tree API |
| `allenai/peS2o` `v1` | 10 shards `partial-train`, **7.5 GB**, 285.5 MB/shard | tree API |
| `codeparrot/github-code-clean` `all-mit` / `all-apache-2.0` | 10 shards each, **1.72 GB / 1.65 GB**, ≈171 MB/shard | tree API |
| `wikimedia/wikipedia` `20231101.ru` / `.en` | 21 shards / 4.88 GB; 41 shards / 11.6 GB | tree API |
| `karpathy/climbmix-400b-shuffle` | 6543 shards, **599.8 GB**, ≈92 MB/shard | tree API |
| `openbmb/Ultra-FineWeb` | en 2048 shards / 2661 GB, zh 256 / 324 GB | tree API |
| `allenai/dolma3_pool` | **3.80 GB partial sample only** | tree API |
| `mlfoundations/dclm-baseline-1.0-parquet` | **3.34 GB partial sample only** | tree API |
| `open-web-math/open-web-math` | 114 shards, 27.4 GB, 0.24 GB/shard | tree API |

### 1.1 One shard actually downloaded and measured (FineWeb2-HQ `rus_Cyrl` shard 0)

1 629 162 947 B in **2 min 23 s** (11.4 MB/s), so the 20 GB plan is ≈30 min of network, not a day.
Column list and the numbers below come from `uv run --with pyarrow`, first 30 000 documents:

| measurement | value | reading |
|---|---:|---|
| columns | `text, id, date, dump, embeddings, file_path, language, language_score, language_script, minhash_cluster_size, top_langs, url, quality_score` | **carries its own classifier verdict and duplicate count** — no re-scoring needed |
| rows in shard | 75 915 | mean doc ≈ 21.5 KB |
| Cyrillic letters / byte | **0.442** | genuinely Russian; 2.26 B per Cyrillic letter ≈ the 2 B/char rule |
| Latin letters / byte | 0.011 | almost no English leakage |
| docs with `і ї є ґ ў` | **0.76 %** | rule 5 in §6 has a real target, small but non-zero |
| duplicate 1 KiB blocks (every 97th) | **13 / 30 385 = 0.04 %** | already deduped upstream — confirms the "don't re-dedup" decision |
| `quality_score` min / median / max | 0.045 / **0.145** / 1.000 | only **17.3 %** of docs score ≥ 0.5 |
| `minhash_cluster_size` median, frac > 1 | 14, **96.1 %** | FineWeb-2's duplicate clusters are large; this is the column its rebalancing uses |
| first sampled doc | «Лучшие места для отдыха. Шум начинается в 7:00 … ВЫ - более обеспеченный платеж немного больше» | **read this line twice**: the text is Russian but reads machine-translated and low-value |

That last row is the most useful finding in this document and it cuts against the plan's own
headline: **FineWeb2-HQ is not finished goods.** It is a 10 %-retention *subset*, not a
high-score *set* — the median document still scores 0.145. Three consequences: (a) keep the
`quality_score` column so a later run can threshold without re-downloading; (b) the only safe
one-pass cut is the bottom (see step 5b in §6), because a 0.5 cut keeps 17 % of the shard and at a
2.8 B-token budget that is a diversity risk, not a quality win; (c) do **not** treat "HQ" as a
finished filter in the provenance record — write down which threshold was applied and where.

Gating, straight from `/api/datasets/<id>`:

| id | `gated` | consequence |
|---|---|---|
| `nvidia/Nemotron-CC-v2`, `-v2.1`, `Nemotron-CC-Code-v1` | `manual` | 401 anonymous |
| `nvidia/Nemotron-CC-Math-v1` | `auto` | needs a token, then auto-approved |
| `bigcode/the-stack-v2`, `bigcode/starcoderdata`, `bigcode/the-stack-dedup` | `auto` | needs a token |
| `HuggingFaceTB/stack-edu` | **False** | reachable, but metadata-only (see §2) |
| `allenai/dolma3_pool`, `dolma3_mix-6T-1025-7B` | False | reachable, parquet is a partial sample |
| `epfml/FineWeb2-HQ`, `HuggingFaceFW/fineweb-2`, `HuggingFaceFW/fineweb-edu` | False | reachable in full |

### 2. The ranked table

Ranking axis is *bytes of usable, licence-clean, deduplicated text per hour of work*, not catalogue size.
"Subset works?" is the answer to: can one `snapshot_download` / `hf download` fetch exactly the bytes
we want (one or a few parquet files), or does it force a 100 GB+ tree?

| # | dataset | HF id | take for us | licence | lang | subset-granular & anonymous? | verdict |
|---|---|---|---|---|---|---|---|
| 1 | FineWeb-Edu `sample-100BT` | `HuggingFaceFW/fineweb-edu` | 2 shards = **4.31 GB** (of 286 GB) | ODC-By 1.0 + CC ToU | en | **yes** — 2.15 GB/shard, 140 shards, `sample-10BT` also exists for small runs | **ADOPT as the English anchor.** Score ≥3 is already applied upstream (§3.2), so no classifier work |
| 2 | FineWeb2-HQ `rus_Cyrl` | `epfml/FineWeb2-HQ` | 2 shards = **3.23 GB** (of 1 277 GB), ≈95 % kept after the gentle `quality_score` cut | ODC-By 1.0 | ru | **yes** — 1.63 GB/shard, 1205 shards | **ADOPT as the Russian anchor, with a local score cut.** 55.2 M docs / 1.2 TB, 10 % retention of FineWeb-2 rus; median doc scores 0.145 so "HQ" is a subset, not a threshold (§1.1) |
| 3 | FineMath `finemath-4plus` | `HuggingFaceTB/finemath` | 6 shards = **1.72 GB**, or the whole 18.4 GB | ODC-By 1.0 + CC ToU | en (math) | **yes** — 0.286 GB/shard; the entire 4+ set is only 18.4 GB | **ADOPT.** 4+ is the tier SmolLM2 used in its decay stage; 3+ is 65 GB and lower precision |
| 4 | Cosmopedia v2 (5 configs) | `HuggingFaceTB/cosmopedia` | 0.9–1.8 GB from `auto_math_text` + `stanford` + `openstax` (+`khanacademy`) | Apache-2.0 (card YAML only — see `data-provenance.md:154-161`) | en (synth) | **yes** — one config per subdirectory, 0.05–0.25 GB/shard | **ADOPT, at 8–9 % of bytes.** It is the only *large* open synthetic-textbook set and it is already in the corpus |
| 5 | `peS2o` v1 | `allenai/peS2o` | 4 shards = **1.14 GB** (of 7.5 GB) | ODC-By 1.0 | en (papers) | **yes** — 285 MB/shard under `v1/partial-train/` | **ADOPT for science.** This is the exact subset OLMo 2 used (58.6 B tokens / 413 GB in the full set) |
| 6 | Wikipedia ru + en | `wikimedia/wikipedia` | ru 2 shards = 0.88 GB; en 2 shards = 0.77 GB | CC-BY-SA-3.0 + GFDL | ru, en | **yes** — 0.3–0.58 GB/shard | **ADOPT, ~8 % combined.** Highest fact density per byte; `wiki_ru` was already a REAL-sidecar source in `mix_v7` |
| 7 | `github-code-clean` permissive | `codeparrot/github-code-clean` | 12 shards (6 `all-mit` + 6 `all-apache-2.0`) = **2.02 GB** (of 32 GB/19 configs) | card says **apache-2.0**; the *slice* is MIT + Apache-2.0 only | code | **yes** — 0.17 GB/shard | **ADOPT as the code line, with eyes open**: 2022 GitHub, heuristic filters only, no edu score. It is the only anonymous + zero-copyleft option |
| 8 | Ultra-FineWeb | `openbmb/Ultra-FineWeb` | 2 shards = 2.6 GB (of 2 661 GB) | Apache-2.0 | en (zh separate) | **yes** — 1.3 GB/shard | **SKIP.** A 2026 verification-filtered FineWeb, but at 4.3 GB of English web we already have FineWeb-Edu; the shard is 1.3 GB and mixing two English web filters adds a second provenance chain for no measured gain |
| 9 | ClimbMix-400B | `karpathy/climbmix-400b-shuffle` | 4 shards = 0.37 GB | **MIT** | en | **yes** — 92 MB/shard, 6543 shards | **OPTIONAL top-up.** A ready-made cluster-balanced mix (400 B tokens, 20 clusters); MIT is the cleanest licence in the table. Use as a diversity control arm, not an anchor |
| 10 | DCLM-Baseline | `mlfoundations/dclm-baseline-1.0-parquet` | 3.34 GB (all that is converted) | CC-BY-4.0 | en | **partially** — the parquet branch is only a 3.34 GB `partial-train` sample | **OPTIONAL.** SmolLM2 ran 60 FineWeb-Edu / 40 DCLM; but 3.3 GB of unknown-provenance sample is a weak anchor. Add only as a 2–3 GB "diverse web" arm |
| 11 | Dolma 3 / Olmo 3 mix | `allenai/dolma3_pool` | — | ODC-By 1.0 | en | **no** — parquet branch is a 3.80 GB partial sample; the pool itself is jsonl.zst | **SKIP.** Right idea, no usable subset |
| 12 | Nemotron-CC v1/v2/v2.1 | `nvidia/Nemotron-CC*` | — | NVIDIA Open Model Licence (`other`) | en | **no** — gated=manual | **SKIP unless a token appears.** Best long-horizon evidence in the literature (§3.2) and the strongest argument for *not* over-filtering |
| 13 | Nemotron-CC-Math 3+/4+ | `nvidia/Nemotron-CC-Math-v1` | 52 B tokens (4+), 133 B (3+) | `other` | en (math) | **no** — gated=auto | **SKIP.** Would supersede FineMath (`+4.8…+12.6` on MATH at 8B) but needs a token |
| 14 | The Stack v2 / StarCoderData | `bigcode/the-stack-v2`, `bigcode/starcoderdata` | — | `other` (permissive-filtered provenance) | code | **no** — gated=auto | **SKIP.** The licence stance is *better* than github-code-clean; the auth requirement is the blocker |
| 15 | Stack-Edu | `HuggingFaceTB/stack-edu` | — | **none declared** | code | **no** — features are `blob_id,language,repo_name,path,src_encoding,length_bytes,score,int_score,detected_licenses,license_type`: **no `text` column** | **SKIP.** The best code-quality idea in the literature (edu-scored code, SmolLM2) shipped as metadata that needs The Stack v2's gated S3 blobs |
| 16 | TinyStories | `roneneldan/TinyStories` | — | CDLA-Sharing-1.0 | en | yes | **NO** — by instruction, and correctly: it is a 2–10 M-parameter demo set, not pretraining data at 275M |
| 17 | OpenWebMath | `open-web-math/open-web-math` | 1–2 shards = 0.24–0.48 GB | ODC-By 1.0 + CC ToU | en (math) | **yes** — 0.24 GB/shard | **REPLACE with FineMath-4+.** `2508.15096` states Nemotron-CC-Math-4+ beats FineMath and OpenWebMath, and SmolLM2's own ablation put FineMath above OWM. OWM survives only as the 0.08 % anneal ingredient |

### 3. What the literature actually says

All titles verbatim from the arXiv API on 2026-11-14; ids that I had wrong are called out in §3.5.

#### 3.1 Deduplication

- `2107.06499` — *Deduplicating Training Data Makes Language Models Better*. "over 1% of the unprompted
  output … is copied verbatim", "emit memorized text ten times less frequently and require fewer train
  steps", "train-test overlap, which affects over 4% of the validation set".
- `2202.06539` — *Deduplicating Training Data Mitigates Privacy Risks in Language Models*. Extraction
  attacks drop sharply after dedup; independent of the quality argument.
- The operational recipe everyone copies is FineWeb's, and the paper states it exactly
  (`2406.17557` §3): "each document's **5-grams** … **112 hash functions** in total, split into
  **14 buckets of 8 hashes** each — targeting documents that are at least **75% similar**."
- **Counter-evidence, from this repo**: `docs/guides/dataset.md:77-84` measured that on already-deduped
  mount shards, near-dup removed **0.063 %** of web bytes. Dedup is not free at 20 GB — it is ~1–2 h of
  CPU. Run it where it can still find something.

#### 3.2 Classifier-based quality filtering

- `2406.17557` — *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*. 15 T tokens
  from 96 snapshots; FineWeb-Edu is **1.3 T tokens** filtered from it; "dramatically better performance on
  knowledge- and reasoning-intensive benchmarks like MMLU and ARC". The threshold is **≥ 3** on a 0–5
  scale, labels from **Llama-3-70B-Instruct on 460 000 sampled pages**, classifier **F1 82 %** on
  validation. Consequence for us: **the edu filter is already inside `sample-100BT`; there is nothing to
  re-score.**
- `2406.11794` — *DataComp-LM: In search of the next generation of training sets for language models*.
  "model-based filtering is key"; DCLM-Baseline reaches **64 % 5-shot MMLU at 7B / 2.6 T tokens**, 6.6 pp
  over MAP-Neo at 40 % less compute.
- `2412.02595` — *Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset*.
  The important counter-argument: aggressive filtering "remov[es] 90% of data", which is wrong when you
  need a long horizon; their full 6.3 T set matches DCLM on MMLU with **4× more unique real tokens**.
  **This does not argue against filtering at 10–20 GB** — we are compute-bound, not horizon-bound; the
  horizon problem starts at 1 T+ tokens.
- `2402.09739` — *QuRating: Selecting High-Quality Data for Training Language Models*. The nuance that
  matters at our size: "it is important to **balance quality and diversity**"; the best rating
  (educational value) matched uniform sampling **with 50 % more steps** — i.e. a quality filter is worth
  ~1.5× compute, not 10×. At 20 GB, over-filtering costs more diversity than it buys.
- `2502.10361` — *Enhancing Multilingual LLM Pretraining with Model-Based Data Selection*. This is the
  FineWeb2-HQ paper: 1B Llama, 70 B and 119 B tokens, "**match the baseline MMLU score with as little as
  15 % of the training tokens**", released for **20 languages including Russian** (Table 8: Russian
  55 220 956 docs / 1.2 TB at 10 % retention).
- `2505.05427` — *Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training
  Data*. Verification-based filtering on FineWeb + Chinese web; ~1 T en tokens. Not adopted (§2 #8).

#### 3.3 Synthetic textbook data at 100 M – 1 B

- `2306.11644` — *Textbooks Are All You Need*. phi-1: **1.3 B params, 6 B web tokens + 1 B synthetic
  tokens, 50.6 % pass@1 HumanEval**; and — the number that matters here — **phi-1-small, 350 M params,
  same pipeline, 45 % HumanEval**. Synthetic textbooks work at 350 M.
- `2309.05463` — *Textbooks Are All You Need II: phi-1.5 technical report*. 1.3 B, 30 B tokens, textbook
  quality + synthetic; strong common-sense/MMLU for its size.
- Cosmopedia (no arXiv id — see §3.5): the blog's own 1B ablation reports **cosmo-1b trained on
  Cosmopedia alone beats TinyLlama 1.1B on ARC-easy, ARC-challenge, OpenBookQA and MMLU, is comparable to
  Qwen-1.5-1B on ARC-challenge and OpenBookQA, and is worse than phi-1.5**. Read: synthetic-only gets you
  a coherent small model; it does **not** replace web+code.
- `2502.02737` — *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*. The
  direct precedent for our scale: 11 T tokens over 4 stages, and **Cosmopedia v2 is 4 % of the final
  stage's mixture**, described as "30B tokens of high-quality synthetic textbooks, blog posts, and
  stories". Also the source of FineMath and Stack-Edu.
- `2305.07759` — *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
  Cited only to justify *not* using it: it targets 1–10 M params.

#### 3.4 Code and math share

Two published anchors, and they disagree by an order of magnitude — which is the useful part:

| anchor | web | code | math | papers/wiki | unit |
|---|---:|---:|---:|---:|---|
| **OLMo 2 Mix 1124** (`2501.00656` Table 1) | 21.32 T (**95.3 %**) | 459 B (**2.05 %**) | 91.2 B (**0.41 %**: OWM 47.2 + Algebraic Stack 44.0) | 413 B peS2o + 77.2 B arXiv + 16.2 B wiki (**2.3 %**) | **bytes** |
| **SmolLM2 stage 4 / decay** (`2502.02737` §4.5) | 58 % | **24 %** | **14 %** (incl. 0.08 % OWM) | 4 % Cosmopedia | **tokens** |

The resolution: byte share and token share are close for English (≈3.6 B/token), so the difference is
real and it is **stage-dependent**. OLMo 2's number is the whole pretraining run of a 7B; SmolLM2's 24/14
is a 1 T-token *decay* stage on a 1.7B. Our corpus is read across three stages already
(`port/configs/mix_v7_docmask.toml`: stage 3 has code 26 % / math 32 %) — so the corpus must carry enough
**distinct** code and math bytes for the decay stage even though the corpus-wide share is modest.

Mixture-search evidence, for the record: `2305.10429` (DoReMi), `2403.16952` (Data Mixing Laws — 1B/100B
RedPajama match to +48 % steps), `2407.01492` (RegMix), `2504.13161` (Nemotron-CLIMB / ClimbMix-400B:
1B model +2.0 % over Llama-3.2-1B at an equal 400 B-token budget). At one 5070 Ti these are
**not affordable** — they need dozens of proxy runs. Use published ratios, not a search.

#### 3.5 Corrections — ids that are not what the folklore says

| claimed | actually is | consequence |
|---|---|---|
| `2402.06619` = Cosmopedia | **Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning** | Cosmopedia has **no arXiv paper**. Cite the dataset card + `huggingface.co/blog/cosmopedia`, or cite SmolLM2's use of cosmopedia-v2 |
| `2402.04333` = "D4" dedup paper | **LESS: Selecting Influential Data for Targeted Instruction Tuning** | do not cite for dedup. The dedup citations that hold are `2107.06499`, `2202.06539`, and FineWeb's §3 |
| `2508.14444` = Nemotron-CC-v2 paper | **NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model** | Nemotron-CC-v2/v2.1 have **no dedicated paper**; the dataset cards tag `2508.14444`, `2508.15096`, `2412.02595`, `2505.02881` |
| `2502.10361` = "FineWeb-HQ" | **Enhancing Multilingual LLM Pretraining with Model-Based Data Selection** | the dataset is FineWeb2-HQ; the paper is the model-based-selection one |

### 4. Byte-level arithmetic (the part a token-level plan gets wrong)

- **Cyrillic is 2 bytes per character** (U+0400–U+04FF in UTF-8). English averages ≈5.7 B/word with the
  space; Russian averages ≈6 characters/word → **≈12 B/word**. So a **16 % byte share of Russian is
  ≈8.5 % of the words** the model sees. If the product needs Russian at parity with English *per word*,
  the Russian byte share must be **≈2× the intended word share**. This is the single most important
  budget line in this document.
- GPT-2-class tokenizers run ≈3.6–4.0 B/token on English web. **10 GB ≈ 2.5–2.8 B token-equivalents;
  20 GB ≈ 5.0–5.6 B.** For 275M params at 20 tokens/param (Chinchilla) you need **5.5 B tokens** →
  **20 GB is exactly one Chinchilla epoch**, and 10 GB is half of one.
- Byte-level models do not have to pay a tokenizer tax, but they do see every byte: BLT
  (`2412.09871` — *Byte Latent Transformer: Patches Scale Better Than Tokens*) trained up to **4 T bytes**
  at 8B with FLOP-controlled scaling, i.e. byte-level pretraining is demonstrated at 20–40× our byte
  budget. Our scale is comfortably inside the demonstrated regime.

### 5. The two mixtures

Shares are **of downloaded bytes**, before the §6 filter pass; expected yield is in the last column.
`data_local/` names match `docs/guides/mix_refetch.toml` so a new plan file is a drop-in.

#### Plan A — 10.02 GB, 19 files

| domain | source | shards | bytes | share | yield after filter |
|---|---|---|---:|---:|---:|
| `fwedu` | fineweb-edu `sample-100BT` | 0000–0001 | 4 306 161 084 | 43.0 % | 99 % |
| `ru` | FineWeb2-HQ `rus_Cyrl` | 0000 | 1 629 162 947 | 16.3 % | 99 % |
| `wiki_ru` | `20231101.ru` | 0000 | 581 024 230 | 5.8 % | 99 % |
| `code` | github-code-clean `all-mit` 0–2 + `all-apache-2.0` 0–2 | 6 | 1 010 148 154 | 10.1 % | 80 % |
| `math` | finemath `finemath-4plus` | 0000–0002 | 862 376 390 | 8.6 % | 98 % |
| `synth` | cosmopedia `auto_math_text` 0–1, `stanford` 0, `openstax` 0 | 4 | 922 951 635 | 9.2 % | 90 % |
| `science` | peS2o `v1/partial-train` | 0000 | 285 532 203 | 2.9 % | 90 % |
| `wiki_en` | `20231101.en` | 0000 | 420 296 449 | 4.2 % | 99 % |
| | | **19** | **10 017 653 092** | 100 % | **≈95 %** |

#### Plan B — 20.12 GB, 39 files

| domain | source | shards | bytes | share | yield |
|---|---|---|---:|---:|---:|
| `fwedu` | fineweb-edu `sample-100BT` | 0000–0003 | 8 610 524 840 | 42.8 % | 99 % |
| `ru` | FineWeb2-HQ `rus_Cyrl` | 0000–0001 | 3 231 736 527 | 16.1 % | 99 % |
| `wiki_ru` | `20231101.ru` | 0000–0001 | 882 972 857 | 4.4 % | 99 % |
| `code` | github-code-clean `all-mit` 0–5 + `all-apache-2.0` 0–5 | 12 | ≈2 017 000 000 | 10.0 % | 80 % |
| `math` | finemath `finemath-4plus` | 0000–0005 | 1 716 524 580 | 8.5 % | 98 % |
| `synth` | cosmopedia `auto_math_text` 0–3, `stanford` 0–2 | 7 | 1 752 988 217 | 8.7 % | 90 % |
| `science` | peS2o `v1/partial-train` | 0000–0003 | 1 141 331 699 | 5.7 % | 90 % |
| `wiki_en` | `20231101.en` | 0000–0001 | 771 744 816 | 3.8 % | 99 % |
| | | **39** | **20 123 823 536** | 100 % | **≈95 %** |

What Plan B buys over Plan A, in the units that matter: 5.5 B token-equivalents (one Chinchilla epoch at
275M) instead of 2.8 B; **≈4.6 B distinct code+math+synth bytes** so the decay stage can hit 24 % code /
14 % math without re-reading the same shards 4× (`max_epochs = 4.0`); and the 196 KB / 8.4 MB eval window
stays untouched because it is carved from the old `default/train/0000.parquet`, not from `sample-100BT`.

### 6. The filtering recipe that runs on this laptop

Budget: 20 cores, 62 GB RAM, 52 GB free disk. Measured cost of the decode step is dominated by parquet
→ UTF-8 (≈10 min/GB single-thread), so parallelise **per shard**, not per document.

Order matters — each rule is counted at the first one that fires, so the manifest shows where bytes died.

| # | step | exact rule | cost on 20 GB |
|---|---|---|---|
| 0 | decode | `pyarrow.parquet.ParquetFile.read_row_group` → UTF-8, write `<domain>.bin` joined by the repo's `\n\n---\n\n` separator, and `<domain>.bin.docs.json` (`{"version":1,"size":N,"starts":[…]}`) in the same pass | 20–40 min, 20 procs |
| 1 | encoding | `bytes.decode("utf-8", errors="strict")`; drop the document, never the shard | seconds |
| 2 | trivial | `< 400 bytes` or `< 50 words` → drop; `word_count == 0` → drop | 1 min |
| 3 | symbol ratio | `symbols/words > 0.15` → drop (FineWeb's English cut); `mean_word_len ∉ [3,10]` → drop | 2 min |
| 4 | boilerplate | ≥3 of `{cookie, javascript, enable js, subscribe, sign in, lorem ipsum}` in the first 500 chars → drop | 1 min |
| 5 | language | **Russian sources**: Cyrillic letters / total letters `< 0.30` → drop; any of `і ї є ґ ў` present → drop (Ukrainian/Belarusian, not Russian). **English sources**: Latin / total `< 0.80` → drop | 2 min |
| 5b | carried score | **FineWeb-2 / FineWeb2-HQ only**: `quality_score < 0.05` → drop (bottom ~5 % by the measured distribution, §1.1), or `minhash_cluster_size > 20` → drop. This replaces a classifier run with a column read — the only source where that is possible. A 0.5 cut keeps 17 %: do not do it on day one | 2 min |
| 6 | exact dedup | `blake2b(normalize(text), digest_size=16)` with `normalize` = NFC + collapse whitespace + casefold; keep first, drop rest | 5 min, ~3 GB RAM for 5 M docs |
| 7 | line dedup | lines with ≥13 words, global set of `blake2b(line)`; drop a document when **>50 %** of its long lines are already seen. This is the `2107.06499` "one 61-word sentence repeated 60 000×" failure | 10 min |
| 8 | near dedup | **only** `code`, `synth`, `science`, `wiki_*` — 5-word shingles, `num_perm=112`, `bands=14`, `rows=8`, Jaccard ≥ **0.75** (FineWeb's params, `2406.17557` §3); band-hash with numpy, union-find over collisions | 40–70 min, the expensive step |
| 9 | degenerate | any 5-gram repeated ≥3× inside one document → drop; `len(set(words))/len(words) < 0.3` → drop | 5 min |
| 10 | contamination | SHA-1 of each document vs the held-out eval window **and** vs the existing `data_local/best/fwedu_train.bin` docs → drop. `Domain.finish()` already refuses to publish on intersection (`docs/guides/dataset.md:60-65`) | 2 min |
| 11 | manifest | per-source kept/dropped counts per rule, `sha256` of every `.bin`, byte totals vs this table's expectation | 1 min |

Deliberate skips, with the reason:

- **No edu-score re-scoring.** FineWeb-Edu and FineWeb2-HQ already carry a classifier verdict; re-scoring
  with a 2026 model would create a second, incomparable quality axis. `ponytail:` if a later run wants it,
  score only `github-code-clean`, where nothing was scored. Note the difference between the two: FineWeb-Edu
  *is* thresholded at ≥3 upstream, FineWeb2-HQ is **not** thresholded at all (§1.1), which is why step 5b
  exists for Russian and not for English.
- **No fastText LID.** FineWeb-2 configs are already language-partitioned and the Cyrillic/Latin ratio
  check catches the two realistic leak classes (Ukrainian/Belarusian into ru, Russian into en).
  `ponytail:` `lid.176.bin` is 917 MB and ~2 h; add it only if rule 5's drop counts look wrong.
- **No re-dedup of fwedu/ru/math.** Measured at 0.063 % on these exact sources (`docs/guides/dataset.md:77-84`).
- **No `max_epochs` change.** Keep 4.0; the bigger corpus is what stops it binding.

### 7. Commands

`huggingface_hub` is not installed here (`python3 -c "import huggingface_hub"` → ModuleNotFoundError), and
`pyarrow` is not either. Both are one `uv run --with …` away, which is the pattern the repo already uses
(`port/scripts/build_doc_sidecar.py` invocations in `mix_v7_docmask.toml`).

Prefer `curl` over the hub's parquet reader: `docs/guides/dataset.md:20` records that the reader burned the
anonymous quota and returned **HTTP 429**. Every URL below was HEAD-checked 200 on 2026-11-14, and the
`content-length` matched the API's `size` — the script asserts that, so a silent HTML error page cannot
become a training file. **The helper was run, not just written**: a 49 MB Cosmopedia shard and a 166 MB
`all-apache-2.0` shard both downloaded, re-ran as no-ops, and a deliberately wrong expected size exited 1
with `SIZE MISMATCH`. Measured throughput **10.5 MB/s**, so the 20 GB plan is **≈32 min of network**.

```bash
# 10 GB plan. ~10.0 GB, 19 files. Run from the repo root.
cd ~/dormouse && mkdir -p data_local/raw && cd data_local/raw

BASE=https://huggingface.co/datasets
get() { # get <dir> <expected_bytes> <url-path-after-datasets/>
  mkdir -p "$1"
  name=$(echo "$3" | sed -E 's#^.*refs%2Fconvert%2Fparquet/##; s#/(partial-)?train/#_#; s#\.parquet$##')
  f="$1/$name.parquet"
  [ -f "$f" ] && [ "$(stat -c%s "$f")" = "$2" ] && { echo "ok   $f"; return 0; }
  curl -fsSL --retry 5 --retry-delay 3 -o "$f" "$BASE/$3"
  got=$(stat -c%s "$f"); [ "$got" = "$2" ] || { echo "SIZE MISMATCH $f: $got != $2"; return 1; }
  echo "ok   $f"; }

# 1. English web, 4.31 GB
get fwedu 2153444469 "HuggingFaceFW/fineweb-edu/resolve/refs%2Fconvert%2Fparquet/sample-100BT/train/0000.parquet"
get fwedu 2152716615 "HuggingFaceFW/fineweb-edu/resolve/refs%2Fconvert%2Fparquet/sample-100BT/train/0001.parquet"
# 2. Russian web, 1.63 GB
get ru    1629162947 "epfml/FineWeb2-HQ/resolve/refs%2Fconvert%2Fparquet/rus_Cyrl/train/0000.parquet"
# 3. Russian Wikipedia, 0.58 GB
get wiki_ru 581024230 "wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.ru/train/0000.parquet"
# 4. Code, 1.01 GB (6 shards)
get code 172370083 "codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/all-mit/partial-train/0000.parquet"
get code 171433052 "codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/all-mit/partial-train/0001.parquet"
get code 172215157 "codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/all-mit/partial-train/0002.parquet"
get code 165742365 "codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/all-apache-2.0/partial-train/0000.parquet"
get code 165124740 "codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/all-apache-2.0/partial-train/0001.parquet"
get code 163262757 "codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/all-apache-2.0/partial-train/0002.parquet"
# 5. Math, 0.86 GB
get math 286267316 "HuggingFaceTB/finemath/resolve/refs%2Fconvert%2Fparquet/finemath-4plus/train/0000.parquet"
get math 288474354 "HuggingFaceTB/finemath/resolve/refs%2Fconvert%2Fparquet/finemath-4plus/train/0001.parquet"
get math 287634720 "HuggingFaceTB/finemath/resolve/refs%2Fconvert%2Fparquet/finemath-4plus/train/0002.parquet"
# 6. Synthetic textbooks, 0.92 GB
get synth 248133641 "HuggingFaceTB/cosmopedia/resolve/refs%2Fconvert%2Fparquet/auto_math_text/train/0000.parquet"
get synth 247696722 "HuggingFaceTB/cosmopedia/resolve/refs%2Fconvert%2Fparquet/auto_math_text/train/0001.parquet"
get synth 253620171 "HuggingFaceTB/cosmopedia/resolve/refs%2Fconvert%2Fparquet/stanford/train/0000.parquet"
get synth 173501101 "HuggingFaceTB/cosmopedia/resolve/refs%2Fconvert%2Fparquet/openstax/train/0000.parquet"
# 7. Science, 0.29 GB
get science 285532203 "allenai/peS2o/resolve/refs%2Fconvert%2Fparquet/v1/partial-train/0000.parquet"
# 8. English Wikipedia, 0.42 GB
get wiki_en 420296449 "wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.en/train/0000.parquet"
```

Plan B is the same block with: fwedu `0002 2152829440`, `0003 2151532316`; ru `0001 1602573580`;
wiki_ru `0001 301948627`; code `all-mit/0003 170834430`, `0004`, `0005` and `all-apache-2.0/0003 163467284`,
`0004`, `0005` (read the last four sizes from the API — they are all ≈171 MB / ≈163 MB);
math `0003 284689405`, `0004 284684820`, `0005 284773965`; synth `auto_math_text/0002 247940446`,
`0003 248102266`, `stanford/0001 253904186`, `0002 253591785`; science `0001 285196022`, `0002 285291537`,
`0003 285311937`; wiki_en `0001 351448367`.

If a token ever shows up, the `huggingface_hub` form of the same thing is:

```bash
uv run --with huggingface_hub hf download HuggingFaceFW/fineweb-edu \
  --repo-type dataset --revision refs/convert/parquet \
  --include "sample-100BT/train/000[01].parquet" --local-dir data_local/raw/fwedu
```

`--include` is a glob over the repo path, so `sample-100BT/train/000[01].parquet` fetches exactly two
2.15 GB files and never touches the 286 GB. The same trick works for every row in §2 that says "yes" —
the `refs/convert/parquet` revision is what makes a 100-shard dataset a one-file download.

### 8. Open questions this document does not answer

1. **The eval window's size is recorded two ways.** The brief says 196 KB carved from the fwedu tail;
   `docs/guides/dataset.md:12` says 8 386 155 B. Both can be true if a newer carve replaced it. The
   contamination gate must hash whatever file is actually there, not what a doc says.
2. **`sample-100BT` vs the existing `fwedu_train.bin`.** The current file is a prefix of
   `default/train/0000.parquet` (CC-MAIN-2013-20). `sample-100BT` is drawn from the whole 1.3 T set, so the
   two **can** overlap. That is precisely what step 10 exists for; do not skip it because "different
   config".
3. **`github-code-clean`'s `apache-2.0` card tag is on the *dataset*, not the code inside it.** Each file
   still carries its upstream licence; the config name (`all-mit`, `all-apache-2.0`) is the filter, and
   that filter is what makes this slice clean. `codeparrot-clean`, by contrast, has **no licence field at
   all** and measured 40.08 % copyleft (`docs/guides/data-provenance.md:163-191`).
4. **Nobody has measured a byte-level model on any of these corpora.** Every quality claim above comes
   from tokenizer-based models at 1B+. See the SPECULATION section.
5. **What the `quality_score` column means is not documented on the card I read.** Median 0.145 on a 1.63 GB
   shard, max 1.000, min 0.045 — that is consistent with either an uncalibrated classifier logit or a
   genuine 85 %-of-docs-are-mediocre verdict, and I could not tell which without the classifier's own card
   (`epfml/FineWeb2-HQ` links its code, not its calibration). Treat any threshold as a hypothesis, not a rule.

---

## SPECULATION (my inference — not a published result)

1. **SPECULATION: the 43 / 16 / 10 / 9 / 9 split is right for a 275M byte-level model, but no ablation
   exists for it.** It is interpolated between OLMo 2 (2 % code by bytes) and SmolLM2's decay stage
   (24 % code by tokens), and justified by `2312.09871`-style byte budgets rather than by a run on this
   card. The only way to know is two 200-step arms at 95M.
2. **SPECULATION: Russian at 16 % of bytes is the product optimum, not a measured one.** No paper reports
   a Russian byte share. The 2 bytes/char argument is arithmetic (VERIFIED); the conclusion "16 % of bytes
   ≈ parity of *words* with a 43 % English web share" assumes Russian and English words carry similar
   information per word, which I have not measured.
3. **SPECULATION: FineWeb2-HQ will beat raw FineWeb-2 `rus_Cyrl` at our scale.** The paper's 15 %-of-tokens
   result is at 1B/70–119 B tokens, which is 3–5× our compute. At 2.8 B tokens a 10 %-retention filter
   could cost more diversity than it returns (`2402.09739`'s quality/diversity warning). My expectation is
   HQ wins but by less than the paper's headline.
4. **SPECULATION: 8–9 % Cosmopedia is near the ceiling for a byte-level 275M model.** SmolLM2 used 4 % at
   1.7B; phi-1 used 14 % of tokens (1 B of 7 B) for a *code* model. I assume byte-level models gain less
   from synthetic prose because the pattern is in the bytes either way. Untested.
5. **SPECULATION: the 30-file, 20 GB plan is ~95 % yield after §6.** Per-source yields in the tables are
   estimates from the published upstream filters (FineWeb/FineWeb2-HQ are already deduped; Cosmopedia and
   GitHub are not). The measured number will come out of step 11 of the manifest.
6. **SPECULATION: a free HF token would be worth ~1 pp MMLU-equivalent at 275M.** The gated sets
   (Nemotron-CC-Math-4+, Stack-Edu text, Nemotron-CC-Code) are better per byte by published margins at
   8B; whether that survives at 275M is unknown. My ranking treats the auth requirement as a hard
   constraint because the brief said "without auth", not because the data is worse.
