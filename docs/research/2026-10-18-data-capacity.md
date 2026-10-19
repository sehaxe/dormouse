# How much data can dormouse usefully swallow — saturation budgets for `medium` (138M) and `b1` (1.07B) (2026-10-18)

Scope: (a) what the 2022–2026 scaling literature says about token budgets and saturation,
(b) what the best small models actually trained on, (c) a measurement protocol that pins
**our** knee, with costs. Read-only pass: only this file was created, nothing committed,
no GPU work. **Source note:** every arXiv id below was verified this session through the
arXiv MCP `get_abstract` (no 429 this time; the 2026-10-16 sibling doc was rate-limited and
said so). Two sources are `[web]` (HF cards / blog) and are marked. Where a number comes
from our own docs or logs it is cited as such.

## TL;DR

1. **Our own medium curve is still steep and our points are not comparable to each other.**
   `med1` (batch 8 × s768, 24 576 B window, 141.9M params, MSA on): 49.2 MB → 2.877,
   122.9 MB → 2.711, 245.8 MB → 2.541. `medbig2` (batch 24 × s1024, 196 608 B window,
   138.4M params, no `msa_*` counters): 61.4 MB → 2.863. Two different networks, two
   windows — the §1 table is the only honest way to hold them.
2. **The marginal is ≈0.17 BPB per doubling over 0.05–0.25 GB.** Where it drops below the
   noise floor depends on the exponent α, which three points over 5× cannot pin: with
   Chinchilla's α ≈ 0.37 the knee is ≈ 82 GB, with α = 0.5 it is 18 GB, with α = 0.2 it is
   ~11 TB. **The measurement exists to pin α, not to find a number we already know.**
3. **The corpus, not the architecture, is the binding constraint.** Best-in-class small
   models train at 2 000–60 000 BPE-tokens/param (§3). Our `b1` at 4 epochs of the 19.7 GB
   corpus reaches ≈ 18 BPE-tok/param — ~100× below SmolLM2-1.7B. `medium` at 4 epochs
   reaches 143 BPE-tok/param.
4. **Cost (our measured rates):** medium 39.5 kB/s → **7.0 GPU-h/GB**; b1 4–8 kB/s →
   **37–69 GPU-h/GB**. A 5 GB medium ladder is ~35 h; a b1 ladder is 1.5–3 days per GB.
5. **MoE-specific:** repetition degradation starts at **4×** and scales with *total* params
   (2609.11917) — our epoch cap of 4 is the onset, not a free zone, and `b1` (1.07B total)
   is far more exposed than `medium` (138M).
6. **Pre-registered prediction:** medium's knee at **4–10 GB** (point: 6 GB); b1's at
   **25–80 GB** (unaffordable: 30–57 days per epoch). The 4-epoch cap adds ≈ 3.7× effective
   unique data for a *dense* model (2305.16264, R\*≈15) but less for our MoE.

## 1. What we measured (facts, with windows)

| preset | params | geometry (config) | notes |
|---|---|---|---|
| `small` | 9 476 993 (port; burn 9 197 390) | d768 · E=3 dense blend (topk 0) · engram 25k rows/order | the A/B polygon (`port/configs/small.toml`) |
| `medium` | 138 435 969 (`medbig2`) / 141 910 405 (`med1`) | d1024 · E=256 topk 2 · rank 64 · d_ffn 2730 · engram 524 288 rows/order | two nets: `med1.config.toml` has `use_msa = true`; `medbig2`'s log has no `msa_*` counters |
| `b1` | 1 066 358 209 | d2048 · E=1024 topk 2 · rank 64 · d_ffn 5461 · engram 4M rows/order | `port/configs/b1.toml`; active set ≈ 37.9M/iteration (config comment, derived not measured) |

**Corpus:** 19.7 GB DCLM-filtered English (`AGENTS.md` §2.6); code/math/RU slices being
added. **Epoch cap 4** (MoE degradation, `docs/research/2026-10-16-data-mixing.md`, sourced
to 2609.11917 below).

| run (geometry) | eval window | bytes seen → BPB | source |
|---|---|---|---|
| `med1` b8×s768 = 6 144 B/step, lr 1e-3, no warmup | 24 576 B (4×8×768) | 49.2 MB → 2.877 · 73.7 → 2.764 · 122.9 → 2.711 · 196.6 → 2.621 · 245.8 → **2.541** | `~/logs/medium_long.log` (steps 8000…40000, eval_every 2000) |
| `medbig2` b24×s1024 = 24 576 B/step, in flight | 196 608 B (8×24×1024) | 61.4 MB → 2.863 (step 2500); last log 3 800 = 93.4 MB | `~/logs/medbig2.jsonl` |
| `b1first` b8×s512 = 4 096 B/step, lr 3e-4 | 32 768 B (8×8×512) | 53.2 MB → 2.840 (13k) · 122.9 → 2.687 (30k) · 204.8 → **2.557** (50k) | `~/logs/b1first.log`, `b1first_resume2.log` |
| `overnight1` b1 b8×s512, lr 3e-4, opt_graph | 32 768 B | 88.1 MB → 2.764 (21 500) | `~/logs/overnight1.log` |

Three cautions that decide the protocol: **(1) a BPB is only comparable within one window**
(`AGENTS.md` §2.6) — none of the four rows above may be plotted on one axis as-is;
**(2) `med1` is noisy**: on a 24.5 KB window the curve reads 2.731 (18k), 2.831 (16k),
2.627 (30k), 2.635 (34k) — ±0.05 bounce, the same order as one doubling's gain;
**(3) throughput is contention-dependent**: `medbig2` logs 39.5 kB/s early and 19–27 kB/s
late; `overnight1` logs 3.3–3.8 kB/s for b1 while `b1first`'s own resume2 implies ≈7.5 kB/s
(143 MB over ~5.3 h of file timestamps — DERIVED). The brief's "b1 8k, newest arms
10–15k" is **not in any log I could read**; treat b1 as 4–8 kB/s until measured.

**Marginal, derived from `med1` (same window, same net):** 49.2→122.9 MB (×2.50) = −0.166
BPB; 122.9→245.8 MB (×2.00) = −0.170 BPB. ≈ 0.17 BPB/doubling, i.e. ≈ 0.5 BPB per 10×,
with the noise caveat above.

## 2. Literature: laws that give a token budget or a saturation criterion

| candidate | source/date | law form / criterion | measured on | what it predicts for us |
|---|---|---|---|---|
| Chinchilla | 2203.15556 (2022) | L = E + A/N^α + B/D^β; N\*∝C^0.5; **~20 tok/param** at optimum | 70M–16B params, 5B–500B tok | 138M → 2.8B BPE tok (≈11 GB bytes); b1 → 21B BPE tok (≈86 GB). Both above our whole budget |
| Chinchilla replication | 2404.10102 (2024) | method-3 re-fit | Hoffmann's plots | published estimates are inconsistent and CIs implausibly narrow — the 20× is a coarse anchor, not a target; 3-point fits are worse |
| Data-constrained | 2305.16264 (JMLR 2025) | **≤4 epochs ≈ free**, value decays to zero beyond; D' = U + U·R\*(1−e^(−R/R\*)) (form re-quoted in 2606.01155 Eq. 5) | ≤900B tok, ≤9B params | one epoch of 19.7 GB is nowhere near the cap; 4 epochs ≈ 3.7× unique if R\*≈15 (**secondary**, our own 2026-10-15 doc, not re-read) |
| Over-training | 2403.08540 (2024) | over-training is predictable; ppl → downstream power law | 0.011B–6.9B params | our runs are under-trained, not over; the loss keeps paying — no knee in the law |
| Beyond Chinchilla | 2401.00448 (2024) | inference-aware optimum: D\* grows | Chinchilla fits + cost model | if b1 is ever served, spend on data/epochs, not params |
| MobileLLM | 2402.14905 (2024) | sub-billion, deep-and-thin | 125M/350M at 1T tok | 125M-class = 8 000 TPP; `medium` at 1.8 B/param is ~1000× below |
| MoE sparsity | 2501.12370 (ICML 2025) | L(N_a, D) + sparsity; optimal sparsity ↑ with compute | 294M–26B total | our 98%-sparse bank is the right direction; **budget against active params**, which makes our effective TPP high |
| Greater Leverage | 2507.17702 (ICLR 2026) | unified MoE efficiency-leverage law | 0.85B active MoE-mini vs 6.1B dense | validates the geometry class; 30–40% of FLOPs to attention is the sweet spot; no token budget |
| **MoE × repetition** | 2609.11917 (Sep 2026) | MoE degrades at **4×**, dense at 8×; effect scales with **total** params | 80M–1B active, 8.5B total | our cap of 4 is the *onset*; b1 is the exposed one |
| Sparse × repetition | 2606.01155 (ICML 2026) | law in (N_a, U_d, R_d, sparsity); sparse training **delays** saturation | ≤1.92B, 2.6B unique, 16 epochs | **conflicts with the row above** — two 2026 papers disagree on sparsity's sign; do not lean on either alone |
| Mixture under constraints | 2605.12715 (May 2026) | D_eff = (1−h)D_total + τ·D_T(1+ρ(r)); scarce domains tolerate **15–20** repeats inside a mix | 143M model, 50M German + English | licenses aggressive decay-phase upsampling of code/math/RU — exactly the plan in `2026-10-16-data-mixing.md` §6.1 |
| Compute- vs data-optimal | 2607.25271 (2026) | effectiveness η(strategy) saturates; depends on N, TPP, D' | **14M–600M**, Dolma-3 | at our exact scale: paraphrasing beats repetition for **<600M** and sub-Chinchilla; repetition wins for large models — so `medium` should prefer paraphrase/synthetic, `b1` repetition |
| Wild AI text | 2609.40295 (2026) | benefit+harm law, reduces to Chinchilla | 800 models, 19.9M–973M | AI tokens help only at low human budgets then harm — keep synthetic out of pretrain bulk |
| Repetition vs dataset size | 2511.13421 (2025) | E(K,N) ≈ K for small K, saturating at Θ(log N) | theory + LLM exps | larger datasets tolerate more repeats; our 19.7 GB is "small N" → few repeats are safe |
| Domain repetition | 2608.14071 (2026) | optimal repeat rises mildly with model size, tracks the domain's own loss | many domains | decide code/math/RU repeats **per domain from domain BPB**, not by a global epoch rule |
| Skaling | 2608.07222 (2026) | capacity–data interaction exponent | — | standard forms mis-estimate at both extremes; fit our own law |
| Byte-level | 2105.13626 (ByT5) · 2412.09871 (BLT) | bytes ≈ 4× tokens for English; BLT is the only FLOP-controlled byte scaling study (to 8B/4T bytes, avg patch 8 B) | — | **no byte-level saturation law exists.** Quote every budget in bytes *and* BPE-equivalents |
| Quantization model | 2303.13506 (2023) | loss = sum of learned quanta → power law + plateaus | theory + small LMs | plateaus are expected; never over-read a single Δ |

**Reading of the table (DERIVED).** Nothing in 2024–2026 gives a *saturation point* for a
byte-level MoE at 10⁸–10⁹ params; the field's nearest statements are (a) dense repetition
laws, (b) MoE repetition degradation, (c) small-model over-training curves. All three say
the same practical thing: **the loss keeps falling, the constraint is compute, and the only
hard wall we own is the 4-epoch MoE onset.**

## 3. What the best small models actually trained on

| model | params | pretrain tokens | TPP | stages / notes | source |
|---|---|---|---|---|---|
| SmolLM2-135M | 135M | **2T** | 14 800 | single-stage, high-quality-only | HF card `[web]`, 2502.02737 |
| SmolLM2-360M | 360M | 4T | 11 100 | — | **not verified this session** (135M's 2T is on the card; 360M's number is from the paper) |
| SmolLM2-1.7B | 1.7B | **11T** | 6 470 | 4 stages 6/2/2/1T, WSD; S4 decay = 24% code + 14% math | 2502.02737 |
| SmolLM3-3B | 3B | 11.2T + 100B long-ctx + 140B reasoning | 3 730 | 3 stages 8/2/1.1T; then 4 epochs × 35B reasoning, SFT, APO | HF blog `[web]` (no arXiv) |
| Qwen3-0.6B / 1.7B | 0.6B / 1.7B | **36T** (all sizes) | 60 000 / 21 000 | 30T S1 + 5T S2 + ~0.1T long-ctx | 2505.09388 |
| Llama-3.2-1B / 3B | 1.23B / 3.21B | **9T** | 7 300 / 2 800 | + logit distillation from Llama-3.1 8B/70B during pretrain | HF card `[web]` |
| Gemma-3-1B | 1B (0.7B non-emb) | **2T** | 2 000 | all Gemma 3 sizes trained with knowledge distillation | 2503.19786 |
| Gemma-3-270M | 270M | **6T** | 22 000 | distilled | Kaggle model card `[web]` |
| Puro-2B | 2B | 1.4T (438.8B + 960B) | 700 | FP8 on RTX 5090s, **$6.9K, 22 514 GPU-h** — closest cost-budget sibling to us | 2608.27370 |
| TinyLlama-1.1B | 1.1B | ~1T unique × ~3 epochs | ~2 700 | ≈3T tokens seen | 2401.02385 |
| MobileLLM-125M / 350M | 125M / 350M | 1T | 8 000 / 2 860 | deep-and-thin | 2402.14905 |
| MiniCPM-2.4B | 2.4B | 1.1T | 459 | WSD, "much higher ratio than Chinchilla" | 2404.06395 + survey 2505.19529 |
| Hunyuan-0.5B | 0.5B | **not published** | — | the release page gives benchmarks only | GitHub `[web]` |

**Our position (DERIVED, byte-level → BPE at ~4 B/token for English, ByT5):**

| | seen now | 1 epoch (19.7 GB) | 4 epochs (78.8 GB) |
|---|---|---|---|
| `medium` 138M stored | 0.245 GB = 1.8 B/param | 143 B/param = 36 BPE-tok/param | 571 B/param = 143 BPE-tok/param |
| `b1` 1.07B stored | 0.205 GB = 0.2 B/param | 18 B/param = **4.6 BPE-tok/param** | 74 B/param = 18 BPE-tok/param |

So: `b1` **with the whole corpus at the MoE-safe cap is a Chinchilla-optimal model, nothing
more**; `medium` can be over-trained to ~7× Chinchilla and still sits 45× below SmolLM2-1.7B.
The byte-level tax (4 positions per token, 2105.13626) means every byte buys ~¼ of a BPE
token's information while costing a full byte of compute.

## 4. The measurement protocol

**4.1 What the curve must be.** One geometry for the whole ladder — the in-flight
`medbig2` shape (batch 24 × s1024, `eval_batches` 8 → **196 608 B window**), stable LR (as
`med1`: `warmup_tokens = 0`, lr 1e-3, no decay keys in the snapshot — so the existing
medium curve is a stable-LR curve and stays comparable). Every eval line prints
`bytes = step × batch × seq_len`. Evals every 1 000 steps. Never mix windows; never plot
`med1` and `medbig2` on one axis without saying which is which.

**4.2 Fit form.** BPB(D) = BPB_inf + A·D^(−α) (Chinchilla's data term, D in bytes). Plot
**two** panels: (i) BPB vs log₁₀ D; (ii) the marginal ΔBPB per doubling of D vs D — panel
(ii) is the decision instrument, because BPB_inf is not identifiable from a few points.

**4.3 Saturation criterion (pre-registered).** Measure the noise floor first: 3 repeat evals
of one checkpoint on the fixed window. The knee is the first D where the smoothed (3-point)
marginal reads **< 0.02 BPB/doubling** — 4–10× the `AB-PROTOCOL.md` estimate of 0.002–0.005
BPB, which is itself unmeasured. Stop the ladder when two consecutive doublings both read
below it.

**4.4 The ladder and its cost** (medium, 39.5 kB/s = 7.03 h/GB; today at 0.093 GB):

| rung | bytes | GPU-h from today | what it decides |
|---|---|---|---|
| R1 | 1.25 GB (the planned budget) | +8.2 | is Δ/doubling at 0.6→1.25 GB inside [0.02, 0.15]? |
| R2 | 2.5 GB | +8.7 | pins α to ±0.1 (4 points over 20×) |
| R3 | **5 GB** | +17.6 | the decision rung: ≥0.02/doubling → continue; else stop |
| R4 (conditional) | 10 GB | +35.2 | only if R3 says continue |
| — | 19.7 GB (1 epoch) | +137 | only if R4 continues |
| — | 78.8 GB (4 epochs) | +553 (23 d) | the MoE cap — almost certainly not worth it |

**Unconditional cost: ~35 h (1.5 days).** One cheap add-on: if a 0-step/eval-only path
exists, re-score `med1.pt` (245.8 MB, 2.541) under `medbig2`'s window — a minutes-long run
that bridges the two nets. **No eval-only flag exists in `port/src` (grep, this session)**, so
treat this as optional and build it only if the bridge matters.

**4.5 b1: only one point, and only if it changes a decision.** b1 costs 37–69 h/GB. It
already has 53 MB → 2.840 and 205 MB → 2.557 — the slope is as steep as medium's. Run a
third b1 point at **0.5 GB (≈19–35 h)** *only if* the medium knee lands below 2 GB (i.e. the
knee is capacity-insensitive, which would be surprising). Otherwise spend that time on the
b1 run itself and on the mixture. The decision b1 actually needs is the queued
**working-set A/B** (4 epochs × 4.8 GB vs one pass × 19 GB, `AB-PROTOCOL.md`), not a ladder.

**4.6 What would falsify the protocol.** If the three repeat evals show σ > 0.05 BPB, the
current single-eval curve is not an instrument: raise `eval_batches` (window ↑) before any
ladder spend, because every Δ in §1 sits inside that noise.

## 5. Pre-registered prediction (mine, before R1 lands)

1. **Marginal curve:** Δ/doubling ≈ 0.06–0.12 at 1 GB, 0.04–0.08 at 2.5 GB, 0.02–0.05 at
   5 GB, 0.015–0.035 at 10 GB.
2. **`medium` knee (first doubling < 0.02): 4–10 GB, point estimate 6 GB** — i.e. ~30% of one
   epoch. With α = 0.37 (Chinchilla's data exponent) the same fit puts it at ~82 GB; with
   α = 0.2 it never arrives. R2's four points decide which world we are in.
3. **`medium` at one epoch (19.7 GB): 1.35–1.70 BPB** (from L = 1.0–1.3 + A·D^−0.37 anchored
   at 0.246 GB → 2.541; SPECULATION with the arithmetic shown). At 5 GB: 1.5–1.75.
4. **The corpus stops paying** when the marginal hits the floor — predicted **before one
   epoch for `medium`**, i.e. we will never need the 4-epoch cap for medium. For `b1` the
   knee is 25–80 GB (4–8× medium, scaled by the stored-param ratio 7.7×, discounted for the
   ~38–76M active set — DERIVED), which is 1.3–4 epochs: **b1 is the model that needs
   repetition, and cannot afford it** (30–57 days/epoch at 4–8 kB/s).
5. **What the 4-epoch cap adds:** for a dense model ≈ 3.7× unique data (2305.16264, R\*≈15,
   secondary). For our MoE I predict **1.5–2.5×** for `medium` and **≤1.3×** for `b1`,
   because degradation starts at 4× and scales with total params (2609.11917). The queued
   working-set A/B is the direct falsifier.

## 6. Ranked shortlist — what to launch after the current run

1. **R0 — noise floor + window hygiene (0 GPU-h, do now).** 3 repeat evals of the current
   checkpoint on the fixed 196 608 B window; print `bytes` on every eval line. *Falsifies:*
   "0.17 BPB/doubling is real" if σ > 0.05.
2. **R1 — resume `medbig2` to 1.25 GB (≈8 h).** Same geometry, `eval_every 1000`.
   *Falsifies:* the α range [0.2, 0.5] if Δ/doubling at 0.6→1.25 GB falls outside
   [0.02, 0.15].
3. **R2 — extend to 2.5 GB (≈9 h).** Four points over 20×. *Falsifies:* the 4–10 GB knee if
   the marginal is still > 0.10/doubling (knee is further out) or < 0.02 (knee is behind us
   — stop buying data, buy mixture).
4. **R3 — extend to 5 GB (≈18 h, the decision rung).** *Falsifies:* prediction 5.3 if
   `medium` at 5 GB is outside 1.5–1.75 BPB.

Total to a decided medium knee: **~35 h**. R4 (10 GB, +35 h) only if R3 says continue.
The b1 point (§4.5) is last, and only conditionally.

## VERIFIED (sourced)

- Every arXiv id in §2/§3 was resolved via the arXiv MCP `get_abstract` this session, no 429:
  2203.15556 · 2404.10102 · 2305.16264 · 2403.08540 · 2401.00448 · 2402.14905 · 2501.12370 ·
  2507.17702 · 2609.11917 · 2606.01155 · 2605.12715 · 2607.25271 · 2609.40295 · 2511.13421 ·
  2608.14071 · 2608.07222 · 2606.06888 · 2510.01631 · 2603.18534 · 2410.05192 · 2508.01483 ·
  2507.17634 · 2404.06395 · 2505.09388 · 2503.19786 · 2407.21783 · 2401.02385 · 2105.13626 ·
  2412.09871 · 2303.13506 · 2502.02737.
- Token budgets/stages: SmolLM2-135M 2T (HF card), SmolLM2-1.7B 11T/4 stages (2502.02737),
  SmolLM3 11.2T + 100B + 140B (HF blog), Qwen3 36T = 30T+5T+~0.1T (2505.09388), Llama-3.2
  9T + distillation (HF card), Gemma-3-1B 2T / 270M 6T (2503.19786, Kaggle card), Puro-2B
  1.4T/$6.9K/22 514 GPU-h (2608.27370), TinyLlama ~1T unique × ~3 epochs (2401.02385),
  MobileLLM 1T (2402.14905), Hunyuan-0.5B **no token count published** (GitHub).
- Our data points, windows and param counts: `~/logs/medium_long.log`, `~/logs/medbig2.jsonl`,
  `~/logs/b1first*.log`, `~/logs/overnight1.log`; `port/checkpoints/medium/med1.config.toml`;
  `port/configs/{small,medium,b1}.toml`; `AGENTS.md` §2.6.
- The b1 log timestamps imply ≈7.5 kB/s while `overnight1.jsonl` logs 3.3–3.8 kB/s — the
  discrepancy is contention (AGENTS §1.5), reported, not resolved.

## SPECULATION (my inference, marked inline)

- **The knee arithmetic** (§5.1–5.3): anchored on three noisy, window-incomparable points and
  on α borrowed from BPE dense models; the byte-level translation (4 B/token) is ByT5's, not
  a byte-MoE measurement.
- **b1's knee at 4–8× medium's bytes**: scaled by stored params, discounted for the active
  set; no source measures a byte-level weight-shared-loop MoE.
- **4 epochs ≈ 3.7× unique (dense)** rests on R\* ≈ 15, which I did **not** re-verify this
  session (it comes from our own `2026-10-15-data-strategy.md` §1, secondary); **the MoE
  discount (1.5–2.5× / ≤1.3×)** is inferred from 2609.11917's onset-at-4× plus that dense
  form — no paper combines the two for a looped MoE.
- **b1's 8–15 kB/s** (the brief's figure) is unverified by any log; my costs use 4–8 kB/s.

## Confidence

**HIGH** on: the corpus/repetition arithmetic (§3 position table), the window-incomparability
of our points, the MoE 4× onset and its total-param dependence (2609.11917), the small-model
token budgets, the cost model at 39.5 kB/s. **MED** on: the applicability of BPE-exponent
scaling laws to a byte-level model, the mixture-under-constraints licence for 15–20× domain
repeats. **LOW** on: the exact knee (4–10 GB), b1's knee, and any number that depends on α.

## Could not verify

1. **SmolLM2-360M = 4T** — read from the paper in a prior session, not re-checked today.
2. **Muennighoff's R\*_D ≈ 15** — abstract-level only this session.
3. **b1's real throughput at the newest arms** — no log shows 8–15 kB/s.
4. **`medbig2`'s config snapshot** — not on disk; its param count (138 435 969) and its
   counter set (no `msa_*`) are the evidence that it differs from `med1` (141 910 405, MSA on).
5. **Whether an eval-only path exists** to re-score old checkpoints on a new window.
6. **Hunyuan-0.5B's token count** — the release publishes none.
7. **A byte-level saturation law** — none exists in 2022–2026; BLT (2412.09871) is the only
   FLOP-controlled byte scaling study and it does not give a token budget.

*Owed: a line in `docs-site/tools/manifest.mjs` — this task forbade touching other files.*
