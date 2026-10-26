# The dormouse model — an honest scorecard

**2026-10-19.** Written from the record on disk: training logs in `~/logs/`,
checkpoints + config snapshots in `/home/sehaxe/dormouse/port/checkpoints/`,
presets in `configs/`, this repo's `docs/`. Every number names its source;
anything not verifiable from a file or a primary external source is marked
**unverified**. This is a snapshot of a live project: `sw1_long` is training
right now, `mix1_long` is stopped at ~84.8k of 260k steps, and the b1 run
finished its planned 50k steps.

Repo note: the canonical Python tree is this worktree (`main`, `7f0b6569`); the
runs were made from a separate checkout (`/home/sehaxe/dormouse/port`). Where
the two disagree (b1 param count, depth verdict) both readings are printed.

## 1. What the model is

Four presets exist; two have real training behind them at the sizes that matter.

| preset | stored params | geometry | vocab / context | iterations | experts | Engram | pinned by |
|---|---|---:|---|---:|---|---|---|
| `b1` | **1,066,358,209** | d 2048 · H 32×64 · d_ffn 5461 · rank 64 | 256 / 2048 | 4 | 1024 routed, topk 2 + shared rank 192 | 3 orders × 4,194,304 rows × dim 32 = **402,653,184 rows** | `tests/test_param_count.py:23`; hand-sum in `configs/b1.toml`; banner in `~/logs/b1first.log:3` |
| `medium` | **141,910,405** (MSA on, `med1`) / **138,435,969** (MSA off, `medbig2`/`mix1`) | d 1024 · H 16×64 · d_ffn 2730 · rank 64 | 256 / 1024 | 3–4 | 256 routed, topk 2 + shared rank 192 | 524,288 rows/order | `tests/test_param_count.py:172`; banners in `medium_run.log`, `medbig2.log`, `mix1.log`; the +3.47M is the MSA arm (`docs/research/2026-10-18-data-capacity.md` §1) |
| `small` | **9,476,993** (port; burn's own was 9,197,390, +3.0% structural) | d 768 · H 12×64 · d_ffn 2048 · rank 64 | 256 / 512 | 4 | 3 experts, **dense blend** (`moe_topk=0`) + shared rank 192 | 25,000 rows/order | `tests/test_param_count.py:22`; `configs/small.toml` |
| `small_wide` | **68,850,949** | small's body (d 768, rank 64, 25k rows/order) with medium's bank: 256 routed, topk 2 | 256 / 512 | 4 | 256 routed, topk 2 + shared rank 192 | 25,000 rows/order | banner in `~/logs/sw1_long.log:2`; `configs/small_wide.toml` (untracked in the canonical repo) |

**Stored** is the checkpoint: at b1, 402.65M of the 1.066B (37.8%) are Engram
lookup rows, 626.46M are routed-expert factors (U·diag(s)·Vᵀ, rank 64), and only
~37M are the dense per-token machinery (KDA 16.8M, shared write 11.2M, mem 4.2M,
shared expert 1.8M, router 2.1M, controller/out/head the rest) — `configs/b1.toml`
"PARAM COUNT" block.

**Active-compute per byte** at b1 is the touched-parameter set: **~37.9M per
iteration** (`configs/b1.toml`, "the touched parameter set is ~37.9M per
iteration", derived not measured). The readiness plan's number is **~57M active
per token** and its own note says the two may not be the same quantity
(`docs/decisions/2026-10-15-1b-readiness-plan.md:82-83`). medium's touched set
is ~10.0M/iteration (derived from the preset's own component sum, not pinned
anywhere); small_wide's stays "in small's class" per its preset comment.

**"Dense equivalent" is a statement about function shape, not information
capacity.** The project's own phrasing is `small_wide.toml` ("the
dense-equivalent shape of the bank grows ~8× against `small`'s") and `b1.toml`'s
active-compute note: the loop applies one block `max_iter` times and the MoE
bank is stored at E× but read at k×. A "×36.6 dense equivalent" for b1 was
**not found verbatim in any file** (searched `docs/`, `port/`, `README*`,
`.bulba/`, `~/logs/`). Derived ratios from pinned numbers: 1.066B/37.9M = ×28.1
(per-iteration touched), 1.066B/57M = ×18.7 (plan's active), 1.066B/(KDA+routed
≈18.1M) = ×59. Treat any single multiplier as **derived/unverified**; the
statement that it is about shape and not capacity is the project's position and
is correct as far as it goes — the same weights are read 4× per byte and the
bank's capacity is addressed, not touched.

The demo page (`docs-site/src/components/Demo.astro`, `i18n/demo.ts`) ships the
`small` polygon as a plain-JS engine: ternary 2-bit factors unpacked to fp32 at
load, `moe_topk = 0`, one thread, ~30M MAC/byte, and a gate that the engine's
logits match the Python reference to max |Δlogit| **1.058e-6** (tolerance 1e-4)
with identical greedy bytes.

## 2. Every measured BPB, with its window

A BPB is only a number with its window. The window is `eval_batches × batch ×
seq_len` bytes (`src/dormouse/train/loop.py`, printed on every eval line).

| run (checkpoint) | params | geometry | data → eval file | window | steps |
|---|---:|---|---|---|---:|
| `k5000` (`real_small/k5000.pt`) | 9,476,993 | b8 × s512, iter 4, dense blend | `corpus_bin_filtered.bin` → `eval_tail.bin` | **81,920 B** (20×8×512) | 5,000 |
| `real1500` | 9,476,993 | b8 × s512, iter 4 | same | 81,920 B | 1,500 |
| `med1` (`medium/med1.pt`) | 141,910,405 | b8 × s768, iter 4, E256 topk2, **MSA on**, padded MoE (`moe_dropped_assignments` non-zero on every log line) | same → `eval_tail.bin` | **24,576 B** (4×8×768) | 6,000 |
| `med1` long (resume) | 141,910,405 | same | same | 24,576 B | 40,000 |
| `medbig2` (`medium_big2/medbig2.pt`) | 138,435,969 | b24 × s1024, **iter 3**, MSA off, dropless | same → `eval_tail.bin` | **196,608 B** (8×24×1024) | 50,000 |
| `mix1` + `mix1_long` (`mix1/mix1.pt`) | 138,435,969 | b24 × s1024, iter 3, dropless, web+wiki+ru+math mix | `corpus_bin_filtered.bin` + mix plan → **`eval_mix/eval_mix.bin`** | 196,608 B | 84.8k (planned 260k) |
| `b1first` (`b1run/b1first.pt`) | 1,066,358,209 | b8 × s512, iter 4, E1024 topk2, dropless, engram in host RAM | filtered → `eval_tail.bin` | **32,768 B** (8×8×512) | 50,000 (204.8 MB) |
| `overnight1` (`b1night/overnight1.pt`) | 1,066,358,209 | b8 × s512, lr 3e-4, `opt_graph` | filtered → `eval_tail.bin` | 32,768 B | 21,500 of 30,000 |
| `sw1_long` (`checkpoints/sw1`) | 68,850,949 | b32 × s512, iter 4, E256 topk2, ternary QAT + int4 activations + LOOM + DSpark on | mix → `eval_mix.bin` | **131,072 B** (8×32×512) | 3,800 of 300,000 (live) |
| `mixall_long` | 143,221,383 | b24 × s1024, iter 3 | mix → (not reached) | — | banner only, **no evals yet** |

Curves (BPB; `fin` = CE of the mean readout, the depth-comparable metric):

- **small** (`port_small_k5000.log`): 3.642@500 → 3.295@1k → 3.104@2k → 3.019@4k
  → **3.001@5k** (81920 B). `real1500`: 3.611@1k → **2.981@1500** (81920 B).
  No `fin` on these lines (older format).
- **medium** (`medium_run.log` + `medium_long.log`): 3.670@500 → 3.140@2.5k →
  2.845@6k; resumed: 2.877@8k → 2.764@12k → 2.711@20k → 2.621@32k → **2.541@40k**
  (fin 2.377). All on the 24,576 B window, `eval_tail.bin`.
- **medbig2** (`medbig2.log`): 2.862@2.5k → 2.715@5k → 2.556@12.5k → 2.472@22.5k
  → 2.441@32.5k → 2.398@42.5k → **2.374@50k** (fin **2.162**), 196,608 B.
- **mix1** (`mix1.log` + `mix1_long.log`): 2.368@2.5k → 2.150@7.5k → 2.008@22.5k
  → 1.980@32.5k → 1.963@37.5k → 1.960@57.5k → **1.936@60k** (fin 1.758); resumed:
  **1.941@65k → 1.941@70k → 1.923@75k → 1.917@80k** (fin **1.738**), 196,608 B.
  README's "BPB 1.936 after 1.47 GB seen" = step 60,000 exactly
  (60,000 × 24,576 B = 1.474 GB).
- **b1first** (`b1first.log` + `b1first_resume2.log`): 3.714@1k → 3.010@8k →
  2.687@30k → 2.557@49k → **2.557@50k** (fin **2.399**), 32,768 B. Note the
  README/plan's "eval every 1000 (16 384 B window)" is wrong for this run: the
  actual lines read 32,768 B (8×8×512).
- **overnight1** (`overnight1.log`): 2.764@21.5k (fin 2.634), 32,768 B.
- **sw1_long** (`sw1_long.log`): one eval, **2.685@2.5k** (fin 2.630), 131,072 B;
  depth curve on the same line: fin at depths 1–4 = 2.0222 / 1.8616 / 1.8314 /
  1.8233.

**Not comparable, and why** — none of these may be plotted on one axis:

1. **Five different windows**: 24,576 / 32,768 / 81,920 / 131,072 / 196,608 B.
2. **Two different eval files**: everything except `mix1`/`sw1` scores
   `eval_tail.bin`; those two score `eval_mix.bin`. Different file = different
   text, and no anchor exists on `eval_mix.bin` (searched `~/logs/` — the
   5-gram bar 2.588–2.911 is a fit on the filtered corpus / `eval_tail` only).
   So mix1's 1.917 cannot be compared to the published 5-gram bar, and the
   night report's "2.54 beats the 5-gram anchors" claim belongs to `med1`
   (eval_tail), not to mix1.
3. **Different depth**: `medbig2`/`mix1` train iter 3; the others iter 4.
4. **Different nets**: `med1` has MSA on and runs padded MoE with dropped
   assignments (`moe_dropped_assignments` non-zero on every line of
   `medium_run.log`); `medbig2`/`mix1` are dropless.
5. **Different batch × seq**: 8×768 vs 8×512 vs 24×1024 vs 32×512 — batch is a
   throughput knob that silently changes how much text the eval scores.

## 3. Benchmarks (ARC-Easy / HellaSwag / synth math)

Two bench runs exist, both on the `mix1` checkpoint, both n = 1,000 (capped from
the full sets: ARC-Easy test = 2,376, HellaSwag val = 10,042), `max_seq_len`
1024, dropped = 0.

| checkpoint (step) | bytes seen | ARC-Easy acc_mean / acc_raw | HellaSwag acc_mean / acc_raw | synth_math acc_mean / acc_raw | source |
|---|---|---:|---:|---:|---|
| mix1 @ 60k | 1.47 GB | **0.272** / 0.244 | **0.305** / 0.278 | **0.240** / 0.201 | `mix1_bench.jsonl` / `.log` (chain3) |
| mix1 @ 80k | 1.97 GB | **0.295** / 0.271 | **0.263** / 0.290 | **0.214** / 0.186 | `mix1long_bench.jsonl` / `.log` |

The 80k step of the same run is **worse on 2 of 3 tasks** (HellaSwag −4.2 pt,
synth −2.6 pt; ARC +2.3 pt); macro acc_mean 0.2723 → 0.2573. At n = 1,000 the
binomial 1σ is ≈1.4–1.6 pt, so the HellaSwag move is real and the others are
not individually resolvable. This is the honest state: **one checkpoint, one
seed, and the curve does not monotonically improve on downstream MC.**

`medbig2_bench.log` is a **crash** (`RuntimeError: Expected all tensors to be on
the same device` in the bench gather) — no medbig2 bench numbers exist. No bench
has been run on b1, medium, small_wide, or small.

**Protocol** (`src/dormouse/bench/{tasks,score,runner}.py`): byte-level MC,
no tokenizer. Each choice is scored by one teacher-forced forward on
`context + choice`; score = Σ log p(byte) over the choice. `acc_raw` picks
argmax of that sum (rewards short choices); `acc_mean` picks argmax of the
per-byte mean (the headline; length-fair). Random = 25%. Templates are pinned in
`tasks.py`: ARC `"Question: {q}\nAnswer:"` + `" choice"`; HellaSwag the
dataset's own `ctx` + ending; synth_math `"Q: {a} op {b} = ?\nA:"` + `" n"`
(256 deterministic items, four distinct integer choices).

## 4. Comparison table: published models at the same scale

Same benchmarks, different protocols — the gap between the two protocols is
**not measured** and could move numbers by several points either way. Our column
is byte-level `acc_mean`, n = 1,000, context 512–1024 **bytes** (≈128–256 BPE
tokens); lm-eval/lighteval are token-level, full sets, `acc_norm` for
HellaSwag/ARC.

| model | params | HellaSwag | ARC-Easy | protocol | source |
|---|---:|---:|---:|---|---|
| **dormouse mix1 @60k** | 138.4M | **30.5** | **27.2** | byte-level acc_mean, n=1000, 1.47 GB seen | `~/logs/mix1_bench.jsonl` |
| **dormouse mix1 @80k** | 138.4M | **26.3** | **29.5** | same, 1.97 GB seen | `~/logs/mix1long_bench.jsonl` |
| SmolLM2-135M | 135M | 42.1 | ARC avg 43.9 | lighteval, zero-shot, 2T tokens | HF card `HuggingFaceTB/SmolLM2-135M` |
| SmolLM2-360M | 360M | 54.5 | ARC avg 53.0 | lighteval, zero-shot, 4T tokens | HF card `HuggingFaceTB/SmolLM2-360M` |
| Qwen2.5-0.5B | 0.49B | 51.2 | ARC avg 45.4 | lighteval re-run reported on the SmolLM2-360M card | same card (HF's own re-run, not Qwen's) |
| Pythia-160M | 162.3M total | not in EleutherAI's suite | **43.5** acc / 39.6 acc_norm | lm-eval 0-shot, step143000 | `EleutherAI/pythia` `evals/pythia-v1/pythia-160m/zero-shot/160m_step143000.json` |
| GPT-2 small | 124M | "31.4%" claimed in this repo's README | — | **unverified** | no primary source located this session; the GPT-2 paper predates HellaSwag, and the HellaSwag paper reports *OpenAI GPT (2018)* 41.9, a different model |
| L20-Edu-135M | 134.5M | 32.4 | **49.6** acc_norm | lm-eval, full zero-shot sets, 13B tokens | HF card `AliceYin/l20-edu-135m` |

How to read this honestly:

- The training-budget gap dwarfs everything: we have seen **1.5–2.0 GB of bytes
  ≈ 0.4–0.5B BPE-equivalent tokens** (bytes/4, ByT5 rule the repo itself uses);
  SmolLM2-135M saw 2T tokens, L20-Edu 13B, Pythia 300B. At 0.4B tokens our
  HellaSwag is 30.5; L20-Edu at 13B gets 32.4; SmolLM2 at 2T gets 42.1.
- Protocol differences are real but second-order next to that: byte vs token
  scoring, acc_mean vs acc_norm, n=1000 vs full set, our 512–1024-byte context
  (HellaSwag items longer than the window are *dropped*, not truncated — 0
  dropped at 1024), and our templates vs the datasets' defaults.
- The README's "ahead of Pythia-160M (30.3%) on ~800× fewer tokens" uses a
  Pythia HellaSwag number that **is not in Pythia's own eval suite**; the
  verifiable comparison on the one shared task (ARC-Easy) is 27.2–29.5 vs 43.5
  (acc) at ~750× the data. The claim should be restated or sourced.

## 5. The A/B verdict ledger

All port-lane arms: 3 seeds, 2,000 steps, `small` b8×s512, 81,920 B window,
unless noted. Rule: an arm is kept iff its mean BPB beats the control mean by
more than the **control's own seed spread**; a tie deletes it (AGENTS.md §1.2).
"delta" is arm − control in BPB (negative = arm better).

| arm | control → arm (mean BPB) | delta | verdict | source |
|---|---|---:|---|---|
| **ColRow 3-D fix** | frobenius 3.0542 (spread 0.0049) → colrow 3.0113 | **−0.0428** | **ADMITTED**, flipped to default; 8.7× the control spread, zero seed overlap | `ab_colrow_driver.log` |
| **depth 2/3/4 (old metric)** | i2 3.0563 → i3 3.0495 → i4 3.0537 | i3 best (−0.0068 vs i2, −0.0042 vs i4) | **ADMITTED as "3"** at the time; see the fin re-run below | `ab_depth_*.jsonl` |
| **depth (fin metric re-run)** | train@1 2.1068 / @2 2.0893 / @3 2.0890 / **@4 2.0810** at eval depth 1; @4 also best at depth 2 (2.0948) | train@4 best, full seed separation | **max_iter = 4** in `b1.toml`; the README's "three passes … four was a tie" is the **old-metric** reading and is stale | `ab_fin_i{1..4}_*.log`; plan §7.3 |
| **LOOM** | control 3.0559 (spread 0.0038) → loom 3.0522 | −0.0037 (inside spread = tie) | **TIE on BPB; REJECTED on `fin`** (loom 2.1005 vs control 2.0810; flattens the depth curve 2.1007→2.1012). Off in b1 | `ab_loom_*.jsonl`, `ab_fin_i4loom_*.log` |
| **MSA (short context)** | colrow control 3.0113 → msaon 3.0103 (round 2, clean) | −0.0010 | **TIE → OFF** in b1; stays a flag for long context | `ab_msa2_*.jsonl`, `2026-10-17-port-night-report.md` |
| **aux heads: future-byte** | control 3.0517 (spread 0.0123) → fb 3.0522 | +0.0005 | **TIE → not carried** (optional flag) | `ab_aux_*.jsonl` |
| **aux heads: DSpark** | → dspark 3.0545 (spread 0.0018) | +0.0028 | **TIE → not carried** | same |
| **NorMuon** | control 2.1713 → normuon 2.2435 | **+0.0722** | **REJECTED** | `ab_muon_variants/normuon_*.jsonl` |
| **schedule-free Muon** | → sf_muon 2.2161 (seed 2 diverged to 4.06, excluded) | **+0.0448** | **REJECTED** (one seed is a failed run — report both) | `ab_muon_variants/sf_muon_*.jsonl` |
| **schedule-free NorMuon** | → sf_normuon 2.2138 | **+0.0426** | **REJECTED** | same |
| **SIGReg (LeJEPA regulariser)** | control 2.2288 (spread 0.0846) → sigreg 2.3768 | **+0.1480** | **REJECTED** (speed +3.3% not net-positive by the owner's rule) | `ab_sigreg_run.log` |
| **act_quant rungs** (int8 / int4 / fp4) | control 2.2027 (spread 0.0124) → 2.2022 / 2.2071 / 2.2042 | −0.0004 / +0.0045 / +0.0016 | **all tie/loss → deleted** at this scale (quantised arms also slower: 34.9–40.4k vs 44.0k tok/s) | `ab_act_quant_run.log` |
| **expert activation: situ** | silu 3.0584 → situ 3.0534 (arm spread 0.0110) | −0.0050 | **no verdict recorded**; not in any preset (default `silu`) | `ab_act_*.jsonl`, `ab_act_driver.log` |
| **expert activation: memglu** | — | NaN on 3/3 seeds at ~25 s | **dead** (assertion `L_Rec + final CE must be finite`) | `ab_act_driver.log` |
| **ternary weights from scratch** | none 3.0540 → annealed 3.0936 / constant 3.1036 / binary 3.1414 | **+0.0396 / +0.0497 / +0.0874** | **REJECTED for training**; QAT belongs to post-base finetuning | `ab_qat_*.jsonl` |
| **ns_steps 5 vs 3** | ns5 3.0555 → ns3 3.0613 | +0.0058 | **5 kept** (the cheaper arm loses) | `ab_ns_*.jsonl` |
| **KDA recompute flag** | false vs true, same loss within noise (2.8037 vs 2.7954 at step 150) | tie | **default false kept**; `true` costs +960 MiB peak (13,164 vs 12,204 MiB) | `ab_kda_rec*.log` |
| **Gram-NS** (optimizer) | — | **1.2–1.4× slower** here (hands back *more* calls, not fewer) | **stays off**; `ns_kind="triton"` won instead (below) | `docs/research/2026-10-17-muon-variants.md`; `src/dormouse/train/ns_kernels.py` docstring |
| **Triton Newton–Schulz** | classic 68.1 ms/step → triton 18.6 ms/step for medium's Muon group (848 → 408 kernels; a b1-shape factor: 52 dispatches → 1 launch) | 3.7× faster optimizer stage | **ADMITTED** (`ns_kind=triton` in the live runs) | `fix2_bench_ns_triton.log` |
| **optimizer CUDA graph** | — | **≈ −12 ms/step** ("the measured opt-stage capture gave −12 ms/step"; README says −11.7) | **ADMITTED** (`opt_graph=true`) | `docs/research/2026-10-18-throughput-leftovers.md:138`; README |
| **dropless routing** | cap 2.0 dropped **77%** of b1 assignments at step 1 (cap 32: 12.5%); Triton grouped GEMM fwd 16.5 ms vs naive 626 ms | 37.9× kernel, 0% drops | **ADMITTED** (`moe_capacity=0`, `moe_kernel=triton`) | `docs/reviews/2026-10-17-port-audit.md` CRITICAL 3; `bench_dropless.log` |
| **bf16 Muon momentum** | — | **−1.23 GiB** (README figure; the exact log was not located this session) | **ADMITTED** (`muon_state_dtype=bf16` in medbig2/mix1/sw1) | README; `src/dormouse/train/optim.py:36` |
| **Engram host RAM** | — | bit-identical to the in-VRAM forward (`torch.equal` on logits and loss) | **ADMITTED** (`engram_ram=true`) | `tests/test_engram_host.py:39-65` |

Not in the ledger because they are not port-lane measurements: the burn-lane A/B
verdicts (JEPA+KoLeo 3/3 seeds, AttnRes, mHC, GR, MoE-dev…) — the burn rulebook
itself declares every A/B verdict in its `AB-PROTOCOL.md` void (AGENTS.md §3.2),
and the port rewrites the mechanisms from scratch, so those results are history,
not evidence for this model.

## 6. Honest position

In league terms: this is a **byte-level, from-scratch research model with a
genuinely measured architecture and no genuinely measured language ability**.
The engineering record is unusually strong for its scale — ~2,000-step 3-seed
A/Bs decided every mechanism, losers are recorded not deleted, the eval window
is printed, and a 1.07B-parameter model trains on one 16 GB consumer card. The
quality record is thin: the best BPB (mix1, 1.917 on a 196,608 B window of
`eval_mix.bin`) is a number on a private window with no external anchor, and the
only downstream numbers are one n=1,000 checkpoint whose HellaSwag/ARC sit at
25–30% (random = 25%) — GPT-2-small-class guesses, three to four orders of
magnitude of training data short of the models it is tabulated against. It is
**not** a chat model (SFT is ~500-6,000 pairs and learns format, not knowledge),
**not** benchmark-validated, **not** comparable to SmolLM2/L20/Pythia numbers
without a protocol bridge, and its flagship b1 run has seen 204.8 MB of bytes —
≈0.003% of what SmolLM2-135M saw. What would change the verdict: (1) a
**full-set bench** (n = 2,376 / 10,042) plus a token-level re-scoring of the
same checkpoint, to bound the protocol gap; (2) a **5-gram anchor on
`eval_mix.bin` at the exact window**, so the mixture BPB is comparable to
something outside this repo; (3) a **≥10× data run** on the same recipe (the
measured marginal is ≈0.17 BPB per doubling over 0.05–0.25 GB — if that slope
holds to 10 GB, the model stops being a toy; if it flattens, the architecture
bet is the story and the model is not). Until then, the honest headline is:
**the instrument is the product; the model is a demonstration.**

## 7. Unverified / open items

- **×36.6 dense equivalent** — not found in any file; derived ratios in §1.
- **GPT-2 small HellaSwag 31.4%** (README) — no primary source located.
- **Pythia-160M HellaSwag 30.3%** (README) — Pythia's own suite has no HellaSwag.
- **−1.23 GiB (bf16 Muon momentum)** and **−11.7 ms/step (opt graph)** — README
  figures; the throughput doc says −12 ms/step, the exact −1.23 GiB log was not
  located.
- **b1 param count** — 1,066,358,209 (test + banner) vs 1,066,354,113 (port
  README), a 4,096 gap; the test is the authority.
- **`mixall_long`** — banner only, no eval yet; not a measurement.
- **`medbig2` bench** — crashed; no numbers.
- **`sw1_long`** — live; one eval point and a `fin` depth curve only.
- **Port docs vs burn docs** — this repo's AGENTS.md status section describes the
  burn lane; the port numbers above come from `port/` and `~/logs/`, and the two
  lanes must not be mixed in one table.
