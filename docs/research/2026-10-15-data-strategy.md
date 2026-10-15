# Data strategy for the 1B dormouse pretrain — best quality-per-byte on a small budget (2026-10-15)

Scope: what data to train `b1` (1.07B total / 57M active, byte-level, 5–20B byte
budget on one 5060 Ti) on, and what to *add*. Read-only pass; **no other file was
modified.** Evidence tags follow AGENTS §1.4: every number carries an arXiv id or
a `[web]` URL, and every derived/assumed line is marked **DERIVED** or
**SPECULATION**. Over-training citations marked `[sib]` are **reused** from
`docs/research/2026-10-15-1b-tech-verification.md`, not re-derived (its §6).

## TL;DR

1. **The owner is not data-poor for this budget — he is data-poor relative to
   labs, and that is the wrong yardstick.** The filtered corpus is 19.7 GB
   (`AGENTS.md` §2.6) = 19.7e9 bytes. A 5–20B *byte* budget is **0.25–1.01 passes
   over the filtered set** and 0.11–0.43 passes over the 46.2 GB raw set
   (**DERIVED**, from the plan's §2.2 byte count). The run is **epoch-limited,
   not volume-limited**: at ≤1 epoch the marginal value of each byte is at its
   maximum, so *filtering and mixture* dominate, and "add 100 TB" is not this
   run's problem.
2. **The regime is not Chinchilla.** Chinchilla puts a 1B at ~20 B tokens
   (2203.15556); SmolLM2 (2502.02737) trains 1.7B on ~11T tokens, MiniCPM
   (2404.06395) derives "much higher compute-optimal data-model ratio than
   Chinchilla", and 2403.08540 shows over-training is *predictable and useful*
   when inference matters. Our 5–20B bytes at 0.40 GFLOP/byte (plan §2.2) is
   **2–8e18 FLOPs → 1.1–4.3 days at 20 TFLOP/s** (plan §2.2 table, **DERIVED**).
3. **Mixture, not volume.** Port SmolLM2's stage shape (web→+code→+math→decay
   upsample; exact fractions in §2) and Nemotron-CC's "4× more unique tokens +
   synthetic rephrasing" (2412.02595). Byte-level adds one wrinkle: patches
   shrink sequence positions 3–5× (plan §2.4; BLT avg patch 8 B, 2412.09871) but
   the *bytes seen* are unchanged — budget in bytes, spend compute in patches.
4. **The pipeline mostly exists.** `crates/dormouse-data` already ships a
   DCLM-classifier (`bin/filter.rs`, 1723 LOC), FNV shard/anchors
   (`bin/shard.rs`, `bin/anchors.rs`) and a byte-level ChatML SFT packer
   (`src/sft.rs`, 396 LOC). **No new filtering logic is needed** — the work is a
   `dormouse data build` CLI (~200–400 LOC of wiring, §6).
5. **Start here** (§6 order): carve eval → run the 5B smoke on filtered+code →
   sweep the web:code ratio at 9.2M → 3-stage decay curriculum for b1.

---

## 0. The corpus ledger (facts, no interpretation)

| item | fact | source |
|---|---|---|
| raw corpus | 46.2 GB UTF-8, `/mnt/…/aria_data/pretrain/real/corpus.bin`, unfiltered | `AGENTS.md` §2.6 |
| filtered corpus | 19.7 GB, DCLM-classifier 0.5 + exact dedup, 272M docs → 104.2M kept | `AGENTS.md` §2.6 |
| eval tail | 500 MB carve, sibling `real_eval/eval_tail.bin`; old 30 MB kept as `.30m.bak` | `AGENTS.md` §2.6 |
| eval window rule | `eval_batches × batch × seq_len` **bytes**; a BPB is comparable only within one window | `AGENTS.md` §2.6 |
| the port's day-1 stream | `port/src/dormouse/train/data.py` (180 LOC): mmap `.bin`, 64 MB ring, byte-range split, `rewind`/`skip_bytes`; no shuffle/parquet | `port/…/data.py:1-180` |
| the crate's extras | `filter.rs` 1723 · `sft.rs` 396 · `anchors.rs` 334 · `shard.rs` 113 LOC (total `dormouse-data/src` = 3845) | `wc -l crates/dormouse-data` |
| Engram n-gram orders | `ORDERS = [2,3,4]`, signed-FNV keys, raw (unreduced) | `crates/dormouse-data/src/lib.rs:84,105` |
| burn-side A/B history | ByteFlow 3.466 vs plain-bytes 3.947 at equal params; ByteFlow 100k = 2.255 BPB; Engram "20–25%" law **refuted**; allocation swept | plan §1.1; `[sib]` §4 |

**Consequence (DERIVED):** with 19.7e9 filtered bytes, a 20B-byte run is ~1.01
passes; a 5B run is 0.25 passes. There is **no data-constrained regime** until
the budget exceeds 19.7 GB or the mixture wants more of a scarce domain.

---

## 1. How much data does a 1.07B need in the correct regime?

| candidate | source / date | key facts & metrics | credibility | effort to adopt |
|---|---|---|---|---|
| Chinchilla ratio | 2203.15556 (2022); replication 2404.10102 | ~20 tokens/param at compute-optimum; the replication re-derives and finds the published third-method estimates "inconsistent", needing >600k experiments | HIGH (canonical) | none — reference only |
| Data-constrained scaling | 2305.16264 (Muennighoff) | ≤4 epochs of repeats ≈ negligible loss change; half-life R*_D ≈ 15; gains up to ~16 epochs; when constrained, spend compute on **epochs not params** | HIGH | informs our ≤1-epoch decision |
| Over-training | 2403.08540 (Gadre) | over-training is predictable; predicts a 1.4B/900B-token (32×) run and a 6.9B/138B run; perplexity→downstream power law | HIGH | none — reference |
| SmolLM2 | 2502.02737 | 1.7B on **~11T tokens (~2 epochs)**, 4 stages, WSD; ~$250k compute; small (135M/360M) models did **single-stage** | HIGH | the stage template (§2) |
| MiniCPM | 2404.06395 | 1.2B/2.4B non-embed; WSD; "much higher compute-optimal data-model ratio than Chinchilla" | HIGH | WSD + wind-tunnel |
| Data-scarce/over-train bias | 2608.07222 `[sib]` | standard loss forms systematically mis-estimate at both extremes; one capacity–data interaction exponent cuts MAPE 1.5–3× | MED | "fit our own law" (sib verdict) |
| Data-constrained 2026 | 2606.06888 | MIR (masked-input reg.) ≈ **1.3× more unique data**; SoftQ couples N and D (Chinchilla's additive form is misspecified) | MED | optional reg. trick |
| Byte-level cost | 2105.13626 (ByT5); 2412.09871 (BLT) | byte sequences are ~4× longer for English (ByT5); BLT is the first FLOP-controlled byte scaling study to 8B/4T bytes, avg patch 8 B | HIGH | budget in **bytes**, not BPE tokens |

**The two readings, stated honestly (DERIVED):**
- *Against total capacity (1.07B params)* — Chinchilla's ~20 tok/param says ~20B
  **bytes** is optimal; 5–20B bytes is 0.25–1.0× Chinchilla. **Not over-trained.**
- *Against active compute (57M params)* — 5–20B bytes is ~88–350× the
  active-param Chinchilla point.
- **FLAG — an inconsistency to fix, not propagate:** `[sib]` §6 labels a
  "1B on ~10B tokens" run "10× overtraining". 10B bytes / 1.07B params = **9.3
  bytes/param, below Chinchilla's 20**, so that label holds only under the
  active-param reading. Quote the parameter count with the label.
- **For a fixed model, more data monotonically lowers loss** (power law) until
  4-epoch decay (2305.16264) or the irreducible floor — so the question is not
  "how much is optimal" but "when does the eval curve stop paying". §6 gives
  the stop conditions.

**ByteFlow wrinkle (DERIVED from plan §2.4 + 2412.09871):** patching cuts
*sequence positions* 3–5× (our measured criterion) or to an 8-byte average (BLT),
but every byte is still fed. A "10B-token" byte budget is 10B **bytes** of data
seen at 2–3.3B patch positions — the data volume is unchanged; only the compute
per byte falls. Log both numbers on every eval line.

---

## 2. Mixture design for a byte-level model on a small budget

### 2.1 What the reference recipes actually did (read off the sources)

**SmolLM2 stage table (2502.02737, §4; HIGH credibility — primary HTML read):**

| stage | tokens | web | code | math |
|---|---|---|---|---|
| 1 | 0–6T | 60% FineWeb-Edu / 40% DCLM | 10% StarCoderData | 0% |
| 2 | 6–8T | 75% (same 60/40) | 20% | 5% OWM |
| 3 | 8–10T | FW-Edu/DCLM → **40/60** | Stack-Edu | ~10% (InfiMM-WebMath + OWM) |
| 4 (decay, 10% of steps) | 10–11T | 58% + 4% Cosmopedia | 24% Stack-Edu | 14% (FineMath4+ + InfiWebMath-3+ + 0.08% OWM + 0.02% AugGSM8K) |

SmolLM2's own benchmark deltas across stages: math 3.21→3.7→7.27→**22.07**, code
8.87→10.56→16.75→**23.21** (Table 3). The stage-4 upsample is where math and code
pay. Their 60/40 and 40/60 web ablations are at Table 1 (FineWeb-Edu wins
MMLU/ARC/OpenBookQA; DCLM wins HellaSwag/CommonsenseQA; the mix gets both). The
135M/360M models did **single-stage, high-quality-only** (DCLM filtered with the
FineWeb-Edu classifier, drop score-0, downsample 1–2) — the small-model lesson.

**Nemotron-CC (2412.02595; HIGH):** ~**4× more unique tokens** than FineWebEdu-2
and DCLM because those only sharded-dedup; Nemotron-CC-HQ raises MMLU **+5.6**
over DCLM at similar size; validated on a 15T-token horizon; 6.3T tokens total,
with a **synthetic-rephrasing** subset. `[web]` Nemotron-CC-v2.1 (HF) = 2.5T
tokens of organic + translated + synthetically-rephrased English.

**Other candidate sources (for the add-list):**

| source | tokens / size | license | note |
|---|---|---|---|
| FineWeb-Edu (2406.17557 lineage; 1.3T tok) | 1.3T | ODC-By | educational classifier |
| DCLM-baseline (2406.11794) | 3.8T | CC-BY-4.0 [web] | fastText classifier; 7B/2.6T→64% MMLU 5-shot |
| Nemotron-CC (2412.02595) | 6.3T | CC-BY-4.0 [web] | 4× unique tokens; HQ +5.6 MMLU |
| The Stack v2 / StarCoder2Data (2402.19173) | 3.3–4.3T tok; ~40% of repos permissive [web] | permissive subset only | **licence filter is mandatory** |
| FineMath4+/3+ (2502.02737 §3.3) | 10B / 34B | ODC-By [web] | best small-budget math source in the sweep |
| CulturaX (2309.09400) | 6.3T, 167 langs | [web] check per-lang | RU+EN source |
| FineWeb2 (2506.20920) | 20 TB / 5B docs / 1000+ langs | **ODC-By** | RU source if a native pipeline is wanted |
| Cosmopedia (2402.09739) | 25B synthetic | Apache-2.0 [web] | synthetic textbooks |

### 2.2 Multilingual (RU+EN heavy by owner default)

Adding a second language to a fixed-capacity 1B buys RU at a cost in EN — the
"curse of multilinguality" (Conneau et al., ACL 2020, aclanthology.org/2020.acl-main.747).
The 2025 re-examination (2510.25947) finds the penalty is **not inevitable when
the mixture is balanced**. **SPECULATION:** no source gives a RU:EN ratio at
1B/5–20B; based on 2510.25947 (balance matters) and 2305.16284-style domain
mixtures, I assume an EN:RU split of ~90:10 late in training (RU as a decay
upsample, like math/code) costs less than mixing RU from step 0. The cheap RU
route is CulturaX's Russian shard + a Wikipedia RU dump, since both are already
filtered and need no new classifier. **Verify** by scoring a RU eval carve — we
have none today, so a bilingual claim is currently unmeasurable.

### 2.3 Mixture as an optimization, not a guess

Two published methods predict the best mixture from small runs: **Data Mixing
Laws** (2403.16952) optimizes a 1B/100B-token RedPajama mixture "comparable to
48% more steps on the default mixture"; **RegMix** (2407.01492) regresses many
small runs. Both want many small runs — which is exactly the 9.2M smoke we can
afford. **SPECULATION:** given our 1-GPU budget, the realistic use is a *coarse*
grid (web:code ∈ {90:10, 80:20} × math ∈ {0, 5%, 10%} late), not a fitted law.

---

## 3. The dedup + filter stack mapped to our infra

| layer | what the literature says | what we already have | verdict |
|---|---|---|---|
| exact dedup | removes repeated docs; models emit memorized text **10× less often** and need fewer steps (Lee et al., ACL 2022, aclanthology.org/2022.acl-long.577) | `bin/filter.rs`: 128-bit-hash exact dedup, `dedup_cap` default **48M** entries | **ship as-is** |
| fuzzy dedup (MinHash) | single-band MinHash LSH, 10 hashes (SmolLM2's FineMath pipeline, 2502.02737 §3.3.2) | not implemented | add later; exact dedup covers most of a ≤1-epoch run |
| quality classifier | fastText beats LLM-prompt ("AskLLM") filtering (2406.11794; `[web]` CS294 slides) | `bin/filter.rs` ports the DCLM classifier (signed-FNV) | **ship as-is** |
| perplexity gate | KenLM/CCNet doc-perplexity filter is one of three canonical signals (Wenzek 2020 LREC; 2409.09613) | none | optional; a tiny byte n-gram scorer is ~50 LOC |
| PII / toxicity | Dolma runs PII + toxicity filters (2402.00159) | none in-tree | add if the corpus is redistributed; not needed for a private run |
| dedup metric | SoftDedup *reweights* duplicates instead of deleting (2407.06654) | none | skip; exact dedup suffices |
| at-rest compression | zstd would cut disk; `[web]` FineWeb2 ships 20 TB as ~8 TB compressed (~2.5×) | `bin/anchors.rs` already gunzips; the trainer reads **plain bytes only** | **SPECULATION:** keep `.bin` plain; the drive is 188 GB free and mmap wants plain bytes |

**Dedup math at 5–10B bytes (DERIVED):** at ~2 KB/doc, 10e9 bytes ≈ 5M docs; a
128-bit key each = **80 MB** of set — trivial against the 48M-entry cap. Exact
dedup is not the bottleneck; **it is already done on the 19.7 GB filtered copy**
(`AGENTS.md` §2.6). Do not re-dedup the filtered stream; re-run only if new data
is mixed in.

**"Many passes on fewer bytes" vs "one pass on more" (the owner's question):**
2305.16264 answers it directly — ≤4 epochs of repeats cost ≈ nothing, so at a
*fixed* compute you may legitimately reuse a good 5 GB four times. But 2510.01631
and 2402.09739 both show the *quality* of the repeated source then dominates.
For us: **one pass over the filtered 19.7 GB beats four passes over the raw 46.2
GB's best 5 GB**, because the filtered copy is the classifier+dedup output and a
20B-byte budget never needs a repeat.

---

## 4. SFT + agent-trace data (post-train lane)

**Byte-level packing is already solved in-tree.** `crates/dormouse-data/src/sft.rs`
(396 LOC) renders a conversation as
`<|im_start|>{role}\n{content}<|im_end|>\n` with a **byte-indexed loss mask**
(1 on assistant content + `<|im_end|>`) and the byte-shift
(`target_mask`) that matches the trainer's `targets[q]=bytes[q+1]` indexing.
There is **no tokenizer and no chat template to borrow** — the "tokens ≈ bytes/4"
penalty (ByT5, 2105.13626) applies to the *cost* of the shell, not to the data.

| dataset | size | licence | fit for 1B+Engram |
|---|---|---|---|
| SWE-Gym (2412.21139) | 2,438 instances; +19% resolve when used for training | HF card `[web]` (check repo) | SWE agent traces + verifier training |
| Open-SWE-Traces (2606.16038) | 207,489 trajectories, 9 languages | NVIDIA HF `[web]` | broadest SWE trace set found |
| Nemotron-SWE-v1 / SWE-Hero OpenHands (HF) | 59k trajectories; MIT/Apache-2.0/BSD-2/3 filter | commercial-safe [web] | repo-grounded OpenHands traces |
| ToolACE | function-calling SFT | **Apache-2.0** [web] | tool-use shell; verifier-friendly |
| AgentInstruct (2407.03502) | 25M synthetic prompt/response pairs | HF `microsoft/orca-agentinstruct-1M-v1` | generative-teaching shell |
| Qwen-AgentWorld (2606.24597) | world model / env simulator, 7 domains | **Apache-2.0** [web] | roll-out env for the post-train loop |

**Post-train SFT mix (from `docs/architecture/post-training.md`):** 30–50% expert
+ 30–50% synthetic-verified + 10–20% repair + 10–25% chat. **Caution:** a 1B
byte-level model has ~no headroom for multi-turn chat *and* tool use; SmolLM2's
1.7B needed a filtered SmolTalk (2402.02737 §5) and its own 135M/360M cut
function-calling entirely. **SPECULATION:** for b1, ship the *trajectory* shell
(SWE + tool) first and defer chat.

---

## 5. Synthetic data on a small compute budget

| candidate | source | finding | evidence level |
|---|---|---|---|
| Synthetic-in-pretrain study | 2510.01631 (EMNLP 2025) | >1000 LLMs, >100k GPU-h: **rephrased-synthetic alone is not faster** than natural web; **1/3 rephrased + 2/3 natural speeds up 5–10×** at larger budgets; good ratio **~30%**; **textbook-style alone is worse at small budgets**; generators >8B params are no better; rephrased shows no collapse, textbook-mixtures do | HIGH (large-scale) |
| Cosmopedia | 2402.09739 | 25B synthetic-textbook tokens from Mixtral | MED |
| phi "Textbooks" | 2306.11644 | textbook-quality synthetic corpus for a small model | MED (narrow) |
| AgentInstruct | 2407.03502 | 25M synthetic pairs via agentic flows for **post-training** | HIGH for SFT |
| model collapse | 2305.17493 (Shumailov); 2404.01413 (Gerstgrasser) | recursion erodes tails; **accumulating** data instead of replacing prevents collapse | HIGH |

**Verdict for us (DERIVED from 2510.01631):** at a **5–20B-byte** budget the
"larger data budgets" condition where rephrasing pays is **not met** — 2510.01631
says textbook-alone *hurts* at small budgets. So: **do not synthesize pretrain
bulk.** Spend any synthetic budget on the **SFT/agent shell**, where AgentInstruct
and ToolACE are the proven pattern, and on a **verifier loop** (the K2-style
rubric + executable check from `docs/archive/research/2026-09-26-long-context-agentic.md:58`).
The self-evolution taxonomy is 2608.03392 (`docs/architecture/post-training.md:36`;
`[sib]` reuse) — framework/memory/skills/tools/workflow/environment; it is a
post-train structure, not a pretrain data source.

---

## 6. Recommended plan

### 6.1 Decision table (sized for the existing corpus + the minimum to add)

| decision | choice | why (source) | confidence |
|---|---|---|---|
| **corpus** | filtered 19.7 GB as the base; **add** nothing for a ≤19.7 GB byte budget | 0.25–1.01 passes; `AGENTS.md` §2.6 | HIGH |
| **web mix** | start 60/40 FineWeb-Edu/DCLM-equivalent, shift to 40/60 in stage 3 | SmolLM2 Table 1 + stage 3 (2502.02737) | HIGH |
| **code share** | **10%** early → **20%** late, from **Stack-Edu-style filtered** code | SmolLM2 stages; Stack-Edu (~125B tok) | HIGH |
| **code licence** | permissive subset only; The Stack v2's ~40%-permissive filter is mandatory | 2402.19173; HF card `[web]` | HIGH |
| **math share** | **0 early → 5–10% mid → ~14% decay**; finest-filtered math only | SmolLM2 stage table; FineMath 2×/6× GSM8K/MATH over InfiMM | HIGH |
| **multilingual** | EN default; RU as a **≤10% decay upsample**, not from step 0 | 2510.25947; Conneau 2020 | **SPECULATION** |
| **dedup** | exact (already done on filtered); **skip re-dedup**; MinHash only if new data is mixed | Lee 2022; `bin/filter.rs` | HIGH |
| **PII/toxicity** | not needed for a private run; add only if redistributing | Dolma 2402.00159 | MED |
| **curriculum** | **single-stage** if ≤5B; **3-stage WSD** (web → +code → math/code decay) if ≥10B | SmolLM2 small vs large; 2404.06395 | MED |
| **zstd at rest** | keep plain `.bin` (mmap); do not compress the training stream | only the anchors tool reads gzip; `[web]` FineWeb2 ~2.5× | **SPECULATION** |
| **eval carve** | carve a **RU + code + math** eval tail now; one window for every arm | `AGENTS.md` §2.6 (a BPB is window-specific) | HIGH |
| **stop condition** | stop adding data when: (a) >4 epochs on the added source, or (b) eval-BPB improvement <0.005 over 2 evals, or (c) budget > 4× the unique bytes | 2305.16264 (4-epoch rule); plateau protocol | MED |

### 6.2 Token budget by wall-clock (DERIVED from plan §2.2: 0.40 GFLOP/byte)

| bytes | at 5 TFLOP/s | at 10 | at 20 |
|---|---|---|---|
| 5B | 4.3 d | 2.1 d | 1.1 d |
| 10B | 8.6 d | 4.3 d | 2.1 d |
| 20B | 17.3 d | 8.6 d | 4.3 d |

Decision: **5B first checkpoint review → extend to 10B → only then 20B.** The 20B
endpoint crosses one epoch on the filtered set (needs the raw copy or a repeat).

### 6.3 What to ADD (only if the budget exceeds 19.7 GB, all permissively licensed)

1. **Nemotron-CC-v2.1** (CC-BY-4.0) — 2.5T tokens, 4× unique-token density.
2. **The Stack v2 permissive subset** (2402.19173) — code, licence-filtered.
3. **FineMath3+/4+** (ODC-By) — the code/math decay upsample.
4. **CulturaX Russian / FineWeb2 RU shard** (check per-language licence) — the RU slice.
5. **SWE-Gym + ToolACE + Nemotron-SWE-v1** (Apache-2.0 / MIT) — SFT only.

### 6.4 `dormouse data build` CLI — LOC estimate

The mechanisms already exist in Rust (`bin/filter.rs` DCLM classifier + exact
dedup, `bin/shard.rs` FNV sharding, `src/sft.rs` byte ChatML). Per plan §7 the
data pipeline **stays in Rust** (PyO3), so:

| option | LOC | note |
|---|---|---|
| A — Python orchestration calling the Rust bins | **200–400** | subprocess `filter`→`shard`→pack; matches the spike's data-pipeline band (300–800) |
| B — port DCLM classifier to Python (`fasttext-wheel`) | 600–1200 | re-implements `filter.rs`; rejected by plan §7 |
| C — add MinHash + perplexity gate in Rust | +150–250 | only if new data is mixed |

**Recommended: A.** Add `data build` as: `filter → dedup → shard → eval-carve →
pack`, with the `eval=` window printed and the no-leak split enforced
(`data.py:134-173` already implements the byte-range split).

### 6.5 Start here (order)

1. **Carve the eval tail** (RU + code + math) and print its window — nothing below
   is measurable without it (`AGENTS.md` §2.6).
2. **Land `dormouse data build`** (option A) so the mixture is a config, not a script.
3. **9.2M smoke**: sweep web:code ∈ {90:10, 80:20}, math ∈ {0, 5%} — 3 seeds each
   (`AGENTS.md` §1.2).
4. **5B-byte b1 run** on filtered+code (stage-1 shape), log BPB + `engram=` +
   `fused kda=` every eval.
5. **If the 5B curve is still falling steeply**, extend to 10B with the stage-2/3
   decay mix; do **not** add bulk data before that decision.

---

## VERIFIED (sourced)

- Chinchilla ~20 tok/param (2203.15556); the replication questions its third method (2404.10102).
- ≤4 epochs of repeats ≈ free; half-life R*_D≈15 (2305.16264). Over-training is predictable (2403.08540).
- SmolLM2 = 1.7B/~11T tokens/4 stages with the exact fractions in §2.1 (2502.02737). MiniCPM derives a >Chinchilla ratio (2404.06395).
- Nemotron-CC: 4× unique tokens, HQ +5.6 MMLU, synthetic rephrasing (2412.02595). DCLM 7B/2.6T→64% MMLU (2406.11794). FineWeb-Edu 1.3T, DCLM 3.8T (2502.02737 §3.2).
- Byte models: ~4× longer sequences (2105.13626); BLT first FLOP-controlled byte scaling, avg patch 8 B (2412.09871).
- Exact dedup cuts memorization 10× (Lee et al. ACL 2022); Dolma runs PII/toxicity filters (2402.00159); CCNet/KenLM is a canonical quality gate (Wenzek 2020).
- Synthetic: 1/3 rephrased + 2/3 natural = 5–10× at large budgets; textbook-alone hurts at small budgets; ~30% optimal; >8B generator no better (2510.01631). Cosmopedia 25B (2402.09739). Collapse is prevented by accumulating data (2305.17493; 2404.01413).
- Agent traces: SWE-Gym 2,438 instances/+19% (2412.21139); Open-SWE-Traces 207,489/9 langs (2606.16038); ToolACE + Qwen-AgentWorld Apache-2.0 [web]; AgentInstruct 25M pairs (2407.03502).
- The pipeline exists: `filter.rs` 1723 / `sft.rs` 396 / `shard.rs` 113 / `anchors.rs` 334 LOC.

## SPECULATION (my inference, marked inline)

- **EN:RU = 90:10 as a decay upsample** — no source gives a ratio at 1B/5–20B; inferred from 2510.25947 (balance) + the SmolLM2 stage shape.
- **Keep `.bin` plain, skip zstd** — inferred from the mmap read path and a ~2.5× `[web]` compression figure; not benchmarked here.
- **`dormouse data build` = 200–400 LOC** — inherited from the torch-spike's data-pipeline band, not measured.
- **"One pass over filtered beats four passes over raw"** — DERIVED from 2305.16264 + the fact that the filtered copy is already classifier+dedup output; our own BPB has not confirmed it.

## Confidence

**HIGH** on: the regime framing (Chinchilla vs over-training), the SmolLM2 stage
template, the dedup/licence facts, and the "budget ≤1 epoch of the filtered
corpus" arithmetic. **MED** on: the exact stage fractions for a *byte-level*
model (SmolLM2 is BPE; patch rates change effective positions), the RU ratio, and
any per-source token count cited from a dataset card. **LOW/SPECULATION** on:
patch-rate→token-count translation and the zstd/LOC estimates.

## Open questions the sweep could not close

1. **No published mixture for a byte-level model at 1B/5–20B.** SmolLM2/BLT are the nearest, and neither publishes byte-level stage fractions.
2. **No RU:EN ratio** for a 1B bilingual; and **we have no RU eval carve** — the claim is unmeasurable today.
3. **No head-to-head of ByteFlow-patched vs plain bytes on a quality-filtered corpus** — the 3.466/3.947 win is at 9.2M on the old stream (plan §1.1).
4. **The `[sib]` "10× overtraining" label is inconsistent** with a 20-tok/param reading; the parameter count must travel with it.

## Sources (read this session)

`arxiv.org/abs/{2203.15556, 2404.10102, 2305.16264, 2403.08540, 2502.02737, 2404.06395,
2606.06888, 2105.13626, 2412.09871, 2406.11794, 2412.02595, 2402.19173, 2506.20920,
2309.09400, 2403.16952, 2407.01492, 2510.25947, 2510.01631, 2402.09739, 2306.11644,
2407.03502, 2412.21139, 2606.16038, 2606.24597, 2305.17493, 2404.01413, 2409.09613}`
plus `[web]`: aclanthology.org/2022.acl-long.577 (Lee dedup), aclanthology.org/2020.acl-main.747
(curse of multilinguality), huggingface.co/datasets/nvidia/Nemotron-CC-v2.1,
huggingface.co/datasets/HuggingFaceFW/fineweb-2, huggingface.co/SWE-Gym,
huggingface.co/datasets/nvidia/Open-SWE-Traces, github.com/QwenLM/Qwen-AgentWorld,
`docs/decisions/2026-10-15-1b-readiness-plan.md`, `docs/research/2026-10-15-1b-tech-verification.md`,
`docs/archive/research/2026-09-26-byte-lm-recipes.md`, `crates/dormouse-data/src/*`.

*Owed (not done — read-only pass): a line in `docs-site/tools/manifest.mjs`; the
task forbade touching other files.*
