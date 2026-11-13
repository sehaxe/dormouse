# SFT corpora for a 60M byte-level Russian-capable chatbot

Date: 2026-11-14. Scope: open SFT/instruction datasets that are (a) actually downloadable today,
(b) legally usable, (c) small enough for a single 16 GB GPU, (d) carry Russian.

All row counts / sizes below were read from the HF API (`/api/datasets/.../tree`,
`datasets-server.huggingface.co/size`) on 2026-11-14. Russian row counts for `oasst1`,
`oasst2` and `aya_dataset` were measured by downloading the parquet/jsonl and counting
(`lang == "ru"`), not copied from cards.

---

## TL;DR

1. **Russian coverage is the binding constraint, and the popular "multilingual" sets are a trap.**
   `CohereLabs/aya_dataset` has **423 Russian rows / 0.12 MB** out of 202,362. Verified by download.
2. The only large, legally clean, natively-Russian multi-turn SFT sets are the
   **IlyaGusev Saiga line** (CC-BY-4.0) and **oasst1/oasst2 ru** (Apache-2.0, human-written).
   Together they are ~15 MB of human dialogue + ~22 MB of synthetic dialogue — the right scale already.
3. Everything famous at 7B scale (`tulu-3-sft-mixture` 1.35 GB, `smoltalk` 3.9 GB,
   `OpenOrca` 2.7 GB) is 20-60x over budget and mostly English. Take subsets or skip.
4. **For a 60M model the useful axis is not "curated 1k vs 50k" — it is short + deduplicated +
   format-consistent, with repetition over a small pool.** Direct small-model evidence: SmolLM2
   filtered its own SFT set for the 135M/360M models by *removing complex tasks and hard examples*
   "to better align with the models' capacity" (2502.02737 §6).
5. Licences that fail the "usable" test: `alpaca` and `no_robots` and all Magpie data are
   **CC-BY-NC-4.0** (non-commercial); `LIMA` is **gated behind HF login**; `saiga_scored` and
   `smoltalk` have **no licence field at all**.

---

## VERIFIED (sourced)

### 1. Academic corpora

| Candidate | HF id / URL | Rows | Size (download) | Lang | Licence | Auth needed | Verdict for a 60M byte-level RU model |
|---|---|---|---|---|---|---|---|
| LIMA | `GAIR/lima` | ~1k (2.9 MB repo, 2 jsonl) | 2.9 MB | en | card tag `other` (commonly cited CC-BY-NC-4.0; card unreadable — gated) | **YES, gated: HTTP 401 without login** | SKIP as a download (auth + non-commercial + English). Use as a *selection philosophy*, not as data. |
| Self-Instruct | `yizhongw/self_instruct` | **0 rows on HF** — repo holds only `self_instruct.py` | 0.01 MB | en | Apache-2.0 | no | SKIP: no data on HF; the 175 seed tasks live in the GitHub repo. |
| WizardLM Evol-Instruct | `WizardLMTeam/WizardLM_evol_instruct_70k` | 70,000 | 65 MB | en | MIT | no | SKIP for RU; takes 65 MB of budget for English complexity you cannot afford at 60M. |
| Orca | paper 2306.02707 only | — | — | en | — | — | SKIP: **no official dataset release** (`microsoft/orca-*` is math only). Open stand-in is `Open-Orca/OpenOrca`. |
| OpenOrca | `Open-Orca/OpenOrca` | 2,942,029 | 2,737 MB | en | MIT | no | SUBSET ONLY (e.g. 5k shortest) — 2.7 GB is 20x budget. |
| SlimOrca | `Open-Orca/SlimOrca` | ~518k | ~0.9 GB | en | MIT | no | SUBSET ONLY, same reason. |
| phi-1 "textbooks" | 2306.11644 | — | — | en | **never released** | — | SKIP: data is not public in any form. Cosmopedia is the open re-implementation. |
| Cosmopedia | `HuggingFaceTB/cosmopedia` (338 files) | ~30M docs | tens of GB | en | Apache-2.0 | no | SKIP: pre-training corpus, not SFT, English, huge. |
| Tulu 3 SFT mixture | `allenai/tulu-3-sft-mixture` | 939,343 | 1,347 MB | mostly en (rus present via aya etc.) | ODC-BY | no | SKIP as-is; it is the reference "big curated mixture" and a good *source list* to mine. |
| Magpie | `Magpie-Align/Magpie-Qwen2-Pro-300K-Filtered` | ~300k | ~1 GB | en (+ separate zh) | **CC-BY-NC-4.0** (repo badge) | no | SKIP: non-commercial + English + hard/long examples (exactly what SmolLM2 stripped for small models). |
| Deita 6k/10k | `hkust-nlp/deita-10k-v0` | 10,000 | 370 MB repo | en | MIT | no | SKIP: English; the *method* (complexity+quality+diversity scoring) is worth copying for RU filtering. |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | 14M | 12,043 MB | en | CC-BY-4.0 | no | SKIP: math, English, 12 GB. Zero value for chat format at 60M. |
| No Robots | `HuggingFaceH4/no_robots` | 9,500 train + 500 test | 11 MB (17 MB raw) | en | **CC-BY-NC-4.0** | no | OPTIONAL: the cleanest human-written LIMA-style set, but non-commercial. 11 MB, good short answers. |
| UltraChat 200k | `HuggingFaceH4/ultrachat_200k` | 515,311 | 1,549 MB | en | MIT | no | SUBSET ONLY (shortest 5-10k turns). |
| SmolTalk | `HuggingFaceTB/smoltalk` | 2,197,730 | 3,960 MB | en | **no licence field in card** | no | SKIP as-is; use per-config subsets (`everyday-conversations` = 2,379 rows / 1 MB). |
| Smol-SmolTalk | `HuggingFaceTB/smol-smoltalk` | 460,341 train | 926 MB (1.8 GB raw) | en | **Apache-2.0** | no | TAKE A SUBSET: the only set here explicitly filtered for sub-1B models. 5-10 MB of short turns is enough. |
| TinyStories-Instruct | `roneneldan/TinyStoriesInstruct` | 21,974,061 | 1,317 MB | en | **none stated** (base TinyStories is CDLA-Sharing-1.0; instruct card 404) | no | FLAG: right shape for tiny models, but no licence and no card. Verify before use. |

### 2. Russian / multilingual instruction sets

| Candidate | HF id | Rows | Size (download) | Licence | Auth | Verdict |
|---|---|---|---|---|---|---|
| **OASST1 ru** | `OpenAssistant/oasst1` | 88,838 msgs total, **7,611 ru (4,382 assistant)** | 40 MB parquet / 32.6 MB ready jsonl.gz | Apache-2.0 | no (verified: plain curl works) | **TAKE full ru filter.** 5.29 MB of human multi-turn Russian. Highest-quality RU dialogue available. |
| **OASST2 ru** | `OpenAssistant/oasst2` | 135,174 msgs, **13,336 ru** | 64 MB parquet | Apache-2.0 | no | **TAKE full ru filter.** 9.84 MB, extends oasst1 with more trees + reviews. |
| **Saiga chats (v1)** | `IlyaGusev/ru_turbo_saiga` | 37,731 | 21.7 MB (87.3 MB raw) | **CC-BY-4.0** | no | **TAKE full.** ChatGPT-generated multi-turn Russian, Baize-style. The single best synthetic RU chat base. |
| Saiga scored (2025) | `IlyaGusev/saiga_scored` | 41,609 | 110 MB (257 MB raw) | **none (no licence field)** | no | SKIP until licence is clarified — better data, unusable licence. |
| RuTurboAlpaca | `IlyaGusev/ru_turbo_alpaca` | 29,822 | 14.6 MB | CC-BY-4.0 | no | TAKE optional: single-turn, same generator family as Saiga, high overlap risk. |
| ru_instruct_gpt4 | `lksy/ru_instruct_gpt4` | 15,056 | 22.4 MB | CC-BY-4.0 | no | **TAKE full.** Native Russian GPT-4 instructions (not machine-translated). Short answers. |
| Russian instructions 2 | `Den4ikAI/russian_instructions_2` | 237,281 | 112 MB | MIT | no | **TAKE A SUBSET (~60k).** Mixed translated + >50k human QA; biggest clean RU bulk. |
| ru-instruct (translated bundle) | `d0rj/ru-instruct` | 753,650 | 839 MB (1.87 GB raw) | Apache-2.0 | no | SUBSET ONLY: machine-translated OpenOrca/OpenHermes/dolphin; stilted Russian. Prefer its sub-datasets. |
| alpaca-cleaned-ru | `d0rj/alpaca-cleaned-ru` | 51,760 | 36.6 MB | CC-BY-4.0 | no | TAKE ~20k rows: translated Alpaca, mediocre but cheap format diversity. |
| Russian dialogues 2 | `Den4ikAI/russian_dialogues_2` | 1,701,649 | 476 MB | MIT | no | **TAKE A SUBSET (~150k).** Very short everyday turns — ideal length profile for a 60M model. |
| Tiny-Russian-Dialogs | `GlayAI/Tiny-Russian-Dialogs` | 125,000 | 8.4 MB | CC0-1.0 | no | SAMPLE-FIRST (low trust: 2026 upload, 205 downloads, synthetic). Right shape, unproven quality. |
| Russian colloquial SFT 50k | `pavelfedortsov/russian-colloquial-sft-50k` | 50,000 | 23 MB | **CC-BY-NC-SA-4.0** | no | SKIP: non-commercial *and* it is a style-transfer task, not chat. |
| **Aya Dataset** | `CohereLabs/aya_dataset` | 202,362 total, **ru = 423 rows / 0.12 MB** | 132 MB | Apache-2.0 | no | **SKIP for Russian.** Verified by download — 0.2% Russian. Great for other languages, useless here. |
| Infinity-Instruct-RU | `KirillR/Infinity-Instruct-RU-Synthetic` | 6,160,896 | 6,593 MB | not stated in card | no | SKIP: 6.6 GB, unverified provenance. |
| LMSYS-Chat-1M | `LMSYS/lmsys-chat-1m` | 1M convs | 1,420 MB | custom LMSYS licence | **YES, gated** | SKIP: gated + custom research terms + privacy-restricted content. |

### 3. Practical chat corpora (non-academic)

| Candidate | HF id | Rows | Size | Lang | Licence | Auth | Verdict |
|---|---|---|---|---|---|---|---|
| Dolly 15k | `databricks/databricks-dolly-15k` | 15,011 | 12.5 MB | en | CC-BY-SA-3.0 | no | OPTIONAL: human-written, genuinely free for commercial use. English only. |
| Alpaca | `tatsu-lab/alpaca` | 52,002 | 23 MB | en | **CC-BY-NC-4.0** ("research use only") | no | SKIP: non-commercial, English, and 2023-era translated-flavour answers. |
| OpenHermes-2.5 | `teknium/OpenHermes-2.5` | ~1M | ~1.5 GB | en | **none stated** | no | SKIP: no licence + size. |

---

## Quality vs quantity, for a *small* model — what the evidence actually says

| Claim | Evidence | Strength |
|---|---|---|
| Small curated sets match/beat 10x larger ones | LIMIT (2311.13133): 1k-6k samples sufficient on MPT-7B/30B for both perplexity benchmarks and model-based eval; mixing textbook-style + open-ended QA is best | 7B/30B, not 60M |
| Filtering beats volume | AlpaGasus (2307.08701): 9k filtered from 52k *beats* full 52k, 5.7x faster training. Deita (2312.15685): 6k selected ≥ baselines using >10x more data | 7B-13B |
| Field consensus that quality > quantity | Survey (2402.05123) states it as the premise of the whole data-selection literature | survey |
| **Small models need *simpler and shorter*, not fewer** | SmolLM2 (2502.02737 §6): for 135M/360M they used `smol-smoltalk`, produced by *removing* complex instruction-following tasks (function calling) and hard MagPie examples "to better align with the models' capacity" | 135M-360M — closest available scale |
| Generic big synthetic sets underperform a curated mixture even at 1.7B | SmolLM2 (2502.02737 §5.1): base model SFT'd on MagPie-Pro or OpenHermes2.5 alone scored *lower* than the post-trained peers; that motivated building SmolTalk | 1.7B |
| Under a fixed update budget, repetition beats more unique data | (2602.11149): 128 epochs on 400 samples beats 1 epoch on 51,200 samples by 12-26 pts; gains plateau at memorisation | 7B, reasoning SFT |
| Very small models learn from simple, short text | TinyStories (2305.07759): 1-10M-parameter models produce coherent English on simple short stories | 1-10M — closest architecture scale |
| Answer density is the best cheap quality proxy | (2610.33969): response-ratio (answer tokens / total tokens) predicts SFT utility better than common alternatives | 2026, recent |
| Mixture allocation matters more than per-sample ranking | (2610.14028): cross-domain complementarity — "pick the mixture, not the samples" | 2026, recent |

**Verdict on the 1k-vs-50k question (verified part):** at 7B+, 1k-6k curated wins. At 135M-360M
the demonstrated lever is *removing complex/long examples*, not *reducing count*. Nothing published
measures a 60M byte-level model, so the 1k-vs-50k framing does not transfer as stated.

---

## SPECULATION (my inference, no direct benchmark)

- **SPECULATION:** for a 60M byte-level model, 1k examples is too few to learn the chat template
  from scratch — the template itself (role markers, turn boundaries, EOS placement) is a
  mechanical pattern that needs thousands of exposures. Based on 2602.11149 (repetition dominates
  under a fixed budget) and SmolLM2 §6 (capacity-matched filtering), I assume **5-15k examples
  × 2-4 epochs** is the sweet spot, and that beyond ~50k unique examples the marginal gain is
  near zero while the risk of teaching long, rambling answers rises.
- **SPECULATION:** quality here means *format consistency and short answers*, not "reasoning
  depth". A 60M model cannot absorb knowledge from SFT; it absorbs style. So "LIMA quality"
  should be re-defined as: one template, answers ≤ ~250 bytes, no markdown, no lists, deduplicated.
- **SPECULATION:** because the model is byte-level, Cyrillic costs 2 bytes/char in UTF-8, so a
  Russian SFT corpus occupies ~2x the byte budget of the same English text. A 100 MB Russian SFT
  set is therefore closer to 200 MB of "English-equivalent" training signal. Budget in bytes, never
  in tokens.

---

## Ranked shortlist (5-8 datasets) + mixture

Rank by (Russian coverage × licence clarity × length-fit for 60M), not by fame.

| # | Dataset | Why it is on the list | Take |
|---|---|---|---|
| 1 | `OpenAssistant/oasst2` (+`oasst1`) | Only human-written, Apache-2.0, genuinely multi-turn Russian at usable scale (21k ru msgs / 15 MB). Real human style, short turns. | full `lang=="ru"` filter |
| 2 | `IlyaGusev/ru_turbo_saiga` | 37.7k CC-BY-4.0 multi-turn Russian chats; the licence is unambiguous and the turns are short. | full (21.7 MB) |
| 3 | `lksy/ru_instruct_gpt4` | 15k native-Russian (not translated) short instruction/answer pairs, CC-BY-4.0. | full (22.4 MB) |
| 4 | `Den4ikAI/russian_dialogues_2` | 1.7M ultra-short everyday exchanges, MIT — the best length profile for a tiny model. | ~150k rows (~40 MB) |
| 5 | `Den4ikAI/russian_instructions_2` | 237k MIT rows of mixed human+translated Russian instructions — cheap topical variety. | ~60k rows (~25 MB) |
| 6 | `d0rj/alpaca-cleaned-ru` | CC-BY-4.0 translated Alpaca; adds task-shape diversity. Stilted Russian — keep the share low. | ~20k rows (~14 MB) |
| 7 | `HuggingFaceTB/smol-smoltalk` | Apache-2.0, explicitly filtered for sub-1B models; small English anchor that keeps the template stable and stops the model from over-fitting to Russian-only phrasing. | 5-10 MB of shortest turns |
| 8 | `GlayAI/Tiny-Russian-Dialogs` (CC0) or `HuggingFaceH4/no_robots` (CC-BY-NC) | Wildcard: CC0 short RU Q→A / human-written EN. Both need a manual quality read before use. | 5-8 MB sample |

**Suggested mixture (total ≈ 120-160 MB raw, one epoch, then 2 more epochs on the merged set):**

- 40% human multi-turn dialogue — oasst1+oasst2 ru (15 MB)
- 25% synthetic multi-turn — ru_turbo_saiga (21.7 MB)
- 20% short everyday exchanges — russian_dialogues_2 subset (40 MB)
- 10% single-turn instructions — lksy/ru_instruct_gpt4 + russian_instructions_2 subset + alpaca-cleaned-ru (≈50 MB)
- 5% English anchor — smol-smoltalk subset (5 MB)

Minimum viable core if you want to start today: **items 1+2+3 = 59 MB, fully CC-BY/Apache,
zero auth, ~60k turns.** That alone teaches format + short answers in Russian.

**Filtering rules to apply before training (inference from the evidence table):**
1. Drop assistant turns longer than ~400 bytes (response-ratio intuition, 2610.33969; capacity,
   SmolLM2 §6).
2. Exact + near dedup (MinHash) across sources — Saiga and RuTurboAlpaca overlap heavily.
3. Drop turns with code blocks, tables, markdown lists — a 60M model reproduces the syntax and
   nothing else.
4. Balance roles: drop conversations whose last turn is a user turn.
5. Cap per-source share at 40% so one generator's style does not dominate.

---

## Exact download commands

Verified-working direct URLs (tested 2026-11-14, no auth):

```bash
# oasst1 — Russian filter afterwards (5.29 MB of ru text)
curl -L -o oasst1_ready.jsonl.gz \
  "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.messages.jsonl.gz"

# oasst2 parquet (9.84 MB of ru text)
curl -L -o oasst2.parquet \
  "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/main/data/train-00000-of-00001-88ba0162028a73fc.parquet"

# aya — NOT recommended for Russian (423 ru rows / 0.12 MB), kept for the record
curl -L -o aya.parquet \
  "https://huggingface.co/datasets/CohereLabs/aya_dataset/resolve/main/data/train-00000-of-00001.parquet"
```

Everything else via `huggingface_hub` (handles filenames, no hardcoded shas):

```bash
pip install -U "huggingface_hub[cli]"

hf download IlyaGusev/ru_turbo_saiga          --repo-type dataset --local-dir data/saiga
hf download lksy/ru_instruct_gpt4             --repo-type dataset --local-dir data/ru_gpt4
hf download Den4ikAI/russian_instructions_2   --repo-type dataset --local-dir data/ru_instr2
hf download Den4ikAI/russian_dialogues_2      --repo-type dataset --local-dir data/ru_dialog2
hf download d0rj/alpaca-cleaned-ru            --repo-type dataset --local-dir data/alpaca_ru
hf download HuggingFaceTB/smol-smoltalk       --repo-type dataset --local-dir data/smoltalk_small
hf download GlayAI/Tiny-Russian-Dialogs       --repo-type dataset --local-dir data/tiny_ru
```

Parquet-only, Russian, one pass:

```python
from datasets import load_dataset
ru = load_dataset("OpenAssistant/oasst2", split="train").filter(lambda r: r["lang"] == "ru")
```

> `GAIR/lima` and `LMSYS/lmsys-chat-1m` return **HTTP 401** without a token and accepted terms —
> verified. Do not plan a pipeline around them unless you have an HF account and accept the gate.

---

## Rejected candidates and why (explicit)

| Rejected | Reason |
|---|---|
| `GAIR/lima` | Gated (401 verified) + `license: other` (widely reported CC-BY-NC-4.0) + English + 1k only. |
| `yizhongw/self_instruct` | Contains no data on HF — only the generator script. |
| Orca / `microsoft/orca-1m` | Never released. |
| phi-1 textbooks | Never released. |
| `HuggingFaceTB/cosmopedia` | Pre-training synthetic corpus, not SFT, tens of GB, English. |
| `allenai/tulu-3-sft-mixture` | 1.35 GB, overwhelmingly English; good as a source *list*, not as data. |
| `Magpie-Align/*` | CC-BY-NC-4.0 + deliberately hard/long examples — the opposite of what a 60M model needs. |
| `tatsu-lab/alpaca` | CC-BY-NC-4.0, "research use only" per its own DATA_LICENSE. |
| `HuggingFaceH4/no_robots` | CC-BY-NC-4.0 → kept only as optional wildcard. |
| `nvidia/OpenMathInstruct-2` | 12 GB, English, math-only. |
| `LMSYS/lmsys-chat-1m` | Gated + custom research licence. |
| `IlyaGusev/saiga_scored` | No licence field anywhere in the card. |
| `HuggingFaceTB/smoltalk` | No licence field in card + 3.9 GB; superseded by `smol-smoltalk` (Apache-2.0). |
| `KirillR/Infinity-Instruct-RU-Synthetic` | 6.6 GB, no licence/provenance. |
| `pavelfedortsov/russian-colloquial-sft-50k` | CC-BY-NC-SA-4.0, and it is style transfer, not chat. |

---

## arXiv ID verification (as requested)

| Given ID | Actual title | Match? |
|---|---|---|
| 2305.11206 | LIMA: Less Is More for Alignment | ✅ |
| 2212.10560 | Self-Instruct: Aligning Language Models with Self-Generated Instructions | ✅ |
| 2304.12244 | WizardLM: Empowering large pre-trained language models to follow complex instructions | ✅ |
| 2306.02707 | Orca: Progressive Learning from Complex Explanation Traces of GPT-4 | ✅ |
| 2306.11644 | Textbooks Are All You Need | ✅ |
| **2402.00838** | **OLMo: Accelerating the Science of Language Models** | ❌ **not Cosmopedia** — no arXiv paper found for Cosmopedia (HF blog + dataset card only) |
| 2411.15124 | Tulu 3: Pushing Frontiers in Open Language Model Post-Training | ✅ |
| 2406.08464 | Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing | ✅ |
| **2411.12308** | **SNN-Based Online Learning of Concepts and Action Laws in an Open World** | ❌ OpenMathInstruct-2 is **2410.01560** |
| **2402.07827** | **Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model** | ❌ Aya *Dataset* paper is **2402.06619** |
| 2312.15685 | What Makes Good Data for Alignment? ... | ✅ |
| **2502.05127** | **Self-supervised Conformal Prediction for Uncertainty Quantification in Imaging Problems** | ❌ no such survey; the data-selection survey is **2402.05123** |
| 2305.07759 | TinyStories: How Small Can Language Models Be and Still Speak Coherent English? | ✅ |
| 2502.02737 | SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model | ✅ |
| 2402.04333 | LESS: Selecting Influential Data for Targeted Instruction Tuning | ✅ |
| 2402.00530 | Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning | ✅ |
| 2307.08701 | AlpaGasus: Training A Better Alpaca with Fewer Data | ✅ |
| 2310.05492 | How Abilities in LLMs are Affected by Supervised Fine-tuning Data Composition | ✅ |
| 2311.13133 | LIMIT: Less Is More for Instruction Tuning Across Evaluation Paradigms | ✅ (added) |
| 2402.05123 | A Survey on Data Selection for LLM Instruction Tuning | ✅ (added) |
| 2602.11149 | Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning | ✅ (added) |
| 2610.33969 | "High-Quality" Data for Early-Stage Instruction Tuning: An Empirical Study of Data Quality Metrics | ✅ (added, 2026) |
| 2610.14028 | Pick the Mixture, Not the Samples: Combinatorial Data Selection for LLMs | ✅ (added, 2026) |

## Open questions

- `GAIR/lima` licence: card says `other`; I could not read it (gated). Non-commercial assumed.
- `HuggingFaceTB/smoltalk` licence: absent from the card. The SmolLM2 *paper* is CC-BY 4.0 and the
  *weights* are Apache-2.0 — neither licenses the dataset. Verify with HF before commercial use;
  `smol-smoltalk` (Apache-2.0) is the safe substitute.
- `roneneldan/TinyStoriesInstruct`: 21.9M rows, no card, no licence. Right shape, unverifiable terms.
- No published SFT-size ablation exists below 100M parameters; the 5-15k recommendation is inference.
- Whether byte-level models need a *different* chat template density than BPE models is unmeasured.
