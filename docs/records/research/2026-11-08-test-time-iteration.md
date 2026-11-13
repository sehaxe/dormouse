# Test-time iteration at 95M: can a model trained with `r` loops get better at `r' > r`?

**Date:** 2026-11-08. **Mode:** read-only + this one file. No GPU job, no code edit, no commit.
**Question (owner):** the dense 95M is the live recipe; loops lose at iso-pretraining-FLOPs
(φ = 0.46, `docs/records/research/2026-11-06-depth-recurrence.md`). But the loop literature's strongest
results are *test-time*: Huginn (3.5B, scaling to 50B-equivalent), Parcae (1.3B, +2.99 CORE),
Ouro (1.4B/2.6B). What is the measured quality-vs-iteration curve, at which scales, and can a
model **trained** with `r` iterations get better by running `r' > r` at inference?

**Tags.** **[T]** — table or a number stated in the paper's text. **[F]** — figure or prose curve
with no table. **[M]** — measured in this repo, file/log named. **[C]** — arithmetic done here,
formula printed. **[S]** — SPECULATION: my inference, no measurement behind it.
Every measurement below carries one of these; code line counts and file paths do not.

---

## 0. TL;DR — вердикт по-русски

1. **Литература не утверждает, что модель, обученная на `r`, становится лучше при `r' > r`.
   Она утверждает почти обратное.** Ouro обучен с максимумом 4 шагов и на 5–8 шагах падает
   (6-бенч среднее 72.39 → 71.32 / 71.14 / 70.49 / 69.87) [T, 2510.25741v6 Table 10].
   Parcae: потолок test-time качества задаётся *глубиной обучения* μ_rec, а не inference
   [T, 2604.12946v1 §5.3].
2. **Положительные результаты `r' > r` есть, но все — при ≥0.6B, на reasoning-метриках, и
   только когда обучение специально сделано глубинно-обобщающим** (случайная глубина у
   Huginn/Parcae, curriculum у retrofit, timestep-conditioning у 2609.36636). На validation
   loss при ≤1.3B положительного результата в этом проходе найти не удалось.
3. **Наш loop — carry-loop с фиксированным `r = 3`** (prompt входит один раз, дальше только
   состояние + `iter_embed[i]`; `loopblock.py:467`). Это ровно тот класс, где каждое измерение
   даёт пик на обученной глубине: Ouro (фикс. 4) и обзор восьми looped-моделей
   [T, 2610.18127v1 §4.3, Table 1].
4. **При ≤200M мне не удалось найти ни одного test-time эксперимента, кроме Parcae 140M/370M,
   и он насыщается на μ_rec** [T, 2604.12946v1 §5.3]. BabyLM 97M — это sweep по *обучаемой* K,
   K=8 против K=4 даёт +0.19 BLiMP на одном seed [T, 2611.06206v1 Table 10].
5. **Наша собственная кривая (102.7M, 2.11 GB, `max_iter=3`): 1.3968 / 1.1762 / 1.1140
   nats/byte.** Предельная отдача 3-й итерации — **0.062 nats = 0.090 BPB**, отношение
   шагов 0.28. Геометрическая экстраполяция: 4-я итерация даёт **0.017 nats = 0.025 BPB**,
   все оставшиеся вместе — 0.035 BPB [M/C, `~/logs/flagship2_stdout.log` step 86000].
6. **30 MB — не тот эксперимент.** На 49 MB (step 2 000) третья итерация даёт **0.0000 nats**:
   модель ещё не научилась пользоваться глубиной. Нулевой результат на 30 MB ничего не
   докажет — он измеряет незрелый блок, а не test-time scaling [M].
7. **Дешёвый решающий эксперимент — не обучение, а eval существующего 5.75 GB чекпоинта
   `flagship1.pt` с расширенными до 6 слотами** (минуты, §4.3). Если нужен новый прогон —
   1.2 GB и ~5 ч, а не 30 MB и 40 мин (§4.2). 40-минутный прогон оставить как проверку
   сантехники.
8. **Вердикт: не тратить карту на test-time итерации при 95M.** Пре-регистрированное правило
   и критерии «don't bother» — §5; там же — что именно должно случиться, чтобы я передумал.

**Confidence:** high on the literature (every number read from the named table); medium on the
transfer to our byte-level BPB (I found no published byte-level test-time curve at any scale).

---

## 1. The distinction the whole literature turns on: three training contracts

"Trained with `r` iterations" names three different things, and they have opposite test-time
behaviour. Conflating them is why the field looks contradictory.

| # | contract | training | examples | what `r' > r` does | source |
|---|---|---|---|---|---|
| C1 | **fixed-`r` carry loop** | one depth, fixed, every step; prompt enters once and is carried in the state | Ouro (max 4); **our flagship (`max_iter=3`, one block, `iter_embed`)** | peaks at `r`, degrades past it | [T] 2510.25741v6 Tables 10–13; [M] `port/configs/flagship_base.toml:9`; `loopblock.py:467` |
| C2 | **depth-distribution re-read loop** | `r` sampled per step from a Poisson-lognormal (Huginn mean 32, Parcae μ_rec 2–12, retrofit mean 4–32); prompt re-supplied every iteration | Huginn, Parcae, Rec-OLMo/Retrofit | improves across the sampled range, then plateaus at μ_rec | [T] 2502.05171v2 §4.2, 2604.12946v1 §5.3, 2511.07384v1 §4.3 |
| C3 | **depth-conditioned / progress-aligned** | iteration index is an input (timestep embedding, FiLM, progress target) or a curriculum | 2609.36636 (LG/BG/AdaLN), PAR, LARM | extrapolation can work, on reasoning; knowledge often degrades | [T] 2609.36636v1 §5.2; 2610.27628v1; 2606.04678v2 |

**Our model is C1, and it is a *carry* loop, not a re-read loop.** `loopblock.py` adds
`iter_embed[i]` to the state each pass (`loopblock.py:467, 1038, 1068`) but never re-injects the
input embedding; the prompt is the initial state and is then carried. The monitoring study of
eight looped LMs separates exactly these two families and finds:

- carry loops (Ouro, LoopUS): **"Best depth ≤ training depth"**, "the loop has learned a
  four-step pipeline, not a process that keeps improving" [T, 2610.18127v1 Table 1, §4.3];
- re-read loops (Huginn, Rec-OLMo, Parcae): **"Monotonous improvement, a plateau"** [T,
  2610.18127v1 Table 1].

That table is the single most transferable result in this report: our block is in the first row.

---

## 2. Every measured quality-vs-iteration result I can source

### 2A. Beyond the trained depth (`r' > r_train`) — the actual question

Δ is the change *from the trained depth*, in the metric's own points; negative = worse.
"Inf FLOPs" is relative to the trained-depth forward. Rows A12–A13 are annotated because those
papers report a *comparison* (conditioning, or vs a non-loop baseline) rather than a clean
depth delta; they are here because they are the only positives beyond a horizon.

| # | paper (arXiv version, table) | params | training contract | `r_train` | `r'` | metric | at `r_train` | at `r'` | Δ | Inf FLOPs | mark |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Ouro, 2510.25741v6, Table 10 | 1.4B | C1 fixed 4 | 4 | 5 / 6 / 7 / 8 | 6-bench avg (ARC-C, ARC-E, C-QA, HS, MMLU, Wino) [C over the table] | **72.39** | 71.32 / 71.14 / 70.49 / 69.87 | **−1.08 / −1.26 / −1.90 / −2.53** | 1.25–2× | [T] |
| A2 | Ouro, 2510.25741v6, Table 10 | 1.4B | C1 | 4 | 8 | MMLU | 67.45 | 64.49 | **−2.96** | 2× | [T] |
| A3 | Ouro-1.4B-Thinking, 2510.25741v6, Table 12 | 1.4B | C1 + SFT | 4 | 5 / 6 / 8 | AIME 2024 | 65.00 | 60.67 / 50.67 / 38.67 | **−4.33 / −14.33 / −26.33** | 1.25–2× | [T] |
| A4 | Ouro-2.6B base, 2510.25741v6, Table 11 | 2.6B | C1 | 4 | 8 | 6-bench avg [C] | **77.45** | 76.44 | **−1.00** | 2× | [T] |
| A5 | Ouro-2.6B-Thinking, 2510.25741v6, Table 13 | 2.6B | C1 + SFT | 4 (peak 3) | 8 | AIME 2025 | 50.67 | 24.33 | **−26.34** | 2.7× | [T] |
| A6 | Monitoring looped LMs, 2610.18127v1 §4.3 | 1.4B | C1 | 4 | 6 | next-token accuracy (5 sets) | 0.605 | 0.580 | **−2.5 pp**; ECE 0.057→0.087 | 1.5× | [T] |
| A7 | Parcae, 2604.12946v1 §5.3, Fig. 7 | 140M | C2 μ_rec 2–12 | μ_rec | ≤24 | val loss | — | plateaus **at T ≈ μ_rec** | no tabulated delta | 2–12× | [F] |
| A8 | Parcae, 2604.12946v1 §5.3, Fig. 7 | 370M | C2 | μ_rec | ≤24 | val loss | — | plateaus at μ_rec | no tabulated delta | 2–12× | [F] |
| A9 | Parcae, 2604.12946v1 Table 5, Fig. 7 | 1.3B | C2 μ_rec=8 | 8 | ≤16 | CORE | **28.44** | plateau at 8; 2× in Fig. 7 | no tabulated delta | 2× | [T/F] |
| A10 | Retrofit, 2511.07384v1 Table 3 | TinyLlama 1.1B | C2 mean 4 | 4 | 8 / 32 | GSM8K | 38.0 | 37.7 / 37.9 | **−0.3 / −0.1** | 2–8× | [T] |
| A11 | Retrofit, 2511.07384v1 Table 4 | OLMo-1B | C2 mean 4 | 4 | 8 / 32 | GSM8K | 38.7 | 39.5 / 40.6 | **+0.8 / +1.9** | 2–8× | [T] |
| A12 | What Makes Recurrence Effective, 2609.36636v1 §3, Fig. 2 | Llama3.1-1B, 5 layers × 4 loops | C1 fixed 4 loops (effective depth 20) | 4 loops (L=20) | 6 loops (L=30) | reasoning score | — (peaks at 30) | **31.88** vs NonLoop 30.25 | **+1.63 vs non-loop**; peaks at 30 | 1.5× | [F] |
| A13 | What Makes Recurrence Effective, 2609.36636v1 §5.2, Fig. 6 | Qwen3-0.6B, 7 × 4 | C1 + timestep cond. | 4 loops (L=28) | 8 loops (L=56) | overall at 2× horizon | 36.59 (BaseLoop, same depth) | **38.46** (BG conditioning) | **+1.87 for conditioning**, not a depth delta | 2× | [F] |
| A14 | STARS, 2605.26733v1 Table 2 | Ouro-1.4B-SFT | C1 + SFT | 4 | 8 | 5-math avg | 70.46 | 52.97 | **−17.49** | 2× | [T] |
| A15 | STARS, 2605.26733v1 Table 2 | Ouro-1.4B-STARS | C1 + JSRR | 4 | 8 | 5-math avg | **74.18** | 65.55 | **−8.63** | 2× | [T] |

**What table 2A says.** Every *fixed-depth* model (C1) gets worse beyond its trained depth.
The only positives are C2/C3 models at 0.6–1.1B, on reasoning/accuracy metrics, and they are
either small (+1.9 GSM8K on one 1B retrofit) or accompanied by explicit knowledge loss
[2609.36636v1 abstract: "recurrence can improve reasoning beyond the training horizon while
degrading knowledge performance"]. At ≤370M the only loss-based test-time curve (Parcae)
saturates at the trained depth.

### 2B. Within the trained depth, and depth-distribution training — why Huginn and Parcae look positive

| # | paper (arXiv version, table) | params | training contract | `r` | metric | numbers | Δ | mark |
|---|---|---|---|---|---|---|---|---|
| B1 | Huginn, 2502.05171v2 Table 1 | 3.5B | C2, mean r̄ = **32** | 4 / 8 / 16 / 32 | ARC-C | 27.99 / 35.15 / 37.71 / 38.23 | **+10.24** from 4→32 | [T] |
| B2 | Huginn, 2502.05171v2 Table 1 | 3.5B | C2 | 4 / 8 / 16 / 32 | HellaSwag | 43.46 / 58.54 / 64.67 / 65.21 | **+21.75** from 4→32 | [T] |
| B3 | Huginn, 2502.05171v2 Table 4 | 3.5B | C2 | 1 / 32 | ARC-E | 34.89 / 69.91 | **+35.02** | [T] |
| B4 | Ouro, 2510.25741v6 Table 10 | 1.4B | C1 | 1 / 2 / 3 / 4 | 6-bench avg [C] | 49.93 / 66.90 / 71.55 / 72.39 | **+22.46** from 1→4 | [T] |
| B5 | Retrofit, 2511.07384v1 Table 4 | OLMo-1B | C2 mean 8 | 4 / 8 / 32 | GSM8K | 43.1 / 44.4 / 44.6 | +1.3 / +0.2 | [T] |
| B6 | Think-at-Hard, 2511.08577v4 Table 1 | 1.7B | two separate loop fine-tunes | 1 / 2 | NTP acc | 73.1 / 79.7; oracle 81.8 | **+6.6** (oracle +8.7); training comparison, not one model at two depths | [T] |
| B7 | BabyLM loops, 2611.06206v1 Table 10 | 97M | C2-ish training-K | 1 / 2 / 4 / 8 | BLiMP | 75.21 / 75.28 / 76.16 / 76.35 | +0.95 from 1→4; +0.19 4→8 | [T] |
| B8 | **ours**, `~/logs/flagship2_stdout.log` step 86000 | 102.7M | C1 fixed 3 | 1 / 2 / 3 | per-iteration CE (nats/B) | **1.3968 / 1.1762 / 1.1140** | −0.2206 / −0.0622 | [M] |

**Read B1–B3 carefully.** Huginn is not "trained on 4, extrapolated to 32". It is trained with a
Poisson-lognormal recurrence distribution whose mean is **32** (`r̄ = 32`, §4.2), and the paper
evaluates at 4/8/16/32 — i.e. *inside* the training distribution, mostly below the mean. The
same is true of Parcae (μ_rec sampled from {2,4,6,8,10,12}, evaluated to 24) and of the retrofit models
(mean 4/8/16/32, evaluated 1–32). Their positive curves are evidence that **depth-distribution
training works**, not that a fixed-depth model extrapolates.

**B6 is the one honest counterweight, with a caveat.** At 1.7B, the *always-1* and *always-2*
fine-tunes differ by +6.6 NTP (73.1 → 79.7), and an oracle that iterates only on the 12–19% of
tokens that were mispredicted scores 81.8 (+2.1 over always-2). That is a comparison of two
separately trained models, not one model at two inference depths. The same paper measures the
cost: the second pass **corrects 8.7% of predictions and flips 2.1% of correct ones into
errors** [T, 2511.08577v4 §1]. Extra iterations are a net positive at 2 passes, with a real
overthinking tax — and that model was trained to loop, unlike ours.

### 2C. At which scales

| params | published `r' > r_train` result | source |
|---|---|---|
| **97M** | none. K=4 vs K=8 is a *training* sweep: +0.19 BLiMP, 1 seed | [T] 2611.06206v1 Table 10 |
| **140M / 370M** | loss plateaus at T ≈ μ_rec; training depth sets the ceiling | [T/F] 2604.12946v1 §5.3 |
| **0.6B** | +1.87 overall at 2× horizon for BG conditioning (reasoning +1.76) | [F] 2609.36636v1 Fig. 6 |
| **1.0–1.1B** | GSM8K −0.1 (TinyLlama, mean 4) to +1.9 (OLMo, mean 4) at r'=8–32 | [T] 2511.07384v1 Tables 3–4 |
| **1.3B** | CORE 28.44 at T=8 (trained μ_rec=8); Fig. 7 to 2× | [T/F] 2604.12946v1 Table 5 |
| **1.4B / 2.6B** | peak at 4, degrade 5–8; SFT collapses harder | [T] 2510.25741v6 Tables 10–13 |
| **1.7B** | 1 → 2 passes +6.6 NTP (two fine-tunes); 2.1% correct→wrong flips | [T] 2511.08577v4 Table 1 |
| **3.5B** | improves to 32 (trained mean 32; Fig. 6 evaluates 1–64) | [T] 2502.05171v2 Table 1, Fig. 6 |

**I found no published test-time-iteration result below 140M. At 140M the curve is a plateau at
the trained depth, i.e. no gain from `r' > r`.** That is the answer to question 3; it is a search
result, not a proof of impossibility.

---

## 3. The mechanism claims, and what each predicts for a model not trained that way

| # | mechanism | claim | prediction for our C1 carry loop (`r=3`, fixed) | evidence |
|---|---|---|---|---|
| M1 | **latent serial computation** | each iteration is one more step of a recurrent computation; tasks needing `k` sequential operations need `k` steps | extra steps can help *if* the model learned a `k`-step algorithm that has not terminated; but the readout was trained on 3-step states, so `r'=4` is off-policy. Task-dependent: reasoning may gain, knowledge should not | [T] 2609.36636v1 abstract + Fig. 2; [T] 2511.08577v4 §1 |
| M2 | **depth extrapolation** ("train short, deploy deep") | with a depth distribution, progress-aligned targets, or iteration conditioning, index `i` maps to computation progress and extrapolates | **no prediction of gain**: our per-iteration slots for 4–6 rows are untrained, and the model never saw a 4-step state. This is the mechanism Huginn/Parcae have and we do not | [T] 2502.05171v2 §4.2; [T] 2604.12946v1 §5.3; [T] 2610.27628v1 (progress-aligned targets lift extrapolation from <2.2× to 16×) |
| M3 | **fixed-point convergence** | iterating a contracting map converges to `h*`; the readout improves as `‖h_T − h*‖` falls; a non-contracting map diverges | gain is bounded by the residual and decays geometrically. **Our measured ratio is 0.28/step**, so the 4th iteration can add at most ~0.017 nats (0.025 BPB), and *all* remaining iterations at most 0.035 BPB [C, §4.1]. STARS shows unconstrained loops collapse instead | [T] 2605.26733v1 §4.2, Table 2; [C] §4.1 |
| M4 | **training-depth ceiling** (Parcae law) | `L(T) = L∞ + Z·exp(−zT)`, with `z ∝ 1/μ_rec`; `L∞` is set by the training law at `T = μ_rec` | the ceiling is a property of the trained weights, not of the inference budget. For a fixed-3 model, `L∞` is reached around T=3; `r'=6` buys at most the residual | [T] 2604.12946v1 §5.3, Appendix L (Huber loss 2.5e-7 at 140M, 1.8e-7 at 370M) |
| M5 | **readout dilution / overthinking** | our readout is the **mean over iterations** (`out_acc/ran`); adding a worse-than-mean iteration makes the mean worse. Separately, later passes flip correct tokens | the 3rd iteration's own CE is 1.1140 while the mean readout is 1.1123 — the average is already almost the last iteration, so an extra iteration moves `fin` by roughly its own CE difference, and a *worse* 4th pass moves it up | [M] `loop.py:1140-1142`; [T] 2511.08577v4 §1 (8.7% fixed vs 2.1% broken) |

**The one mechanism that could rescue the idea** is M3 with a *non*-geometric tail: if the 4th
iteration happened to cross a discrete improvement (a "phase change" in the latent computation),
the geometric fit would underpredict. Nothing in the literature shows such a phase change for a
fixed-depth carry loop; Ouro's curves are smooth and monotone after the peak.

### 3.1 The honest boundary at 95M with 120 MB–2 GB of training

**The answer is "only at 1B+ with dedicated training" — and on validation loss, not even there
below 1.3B.** The pieces:

- **At 95M I found no published test-time iteration experiment.** The only ≤100M loop result is
  BabyLM's *training* K sweep (K=4 vs K=8: +0.19 BLiMP, one seed) [T, 2611.06206v1].
- **The smallest published test-time curve is Parcae at 140M/370M**, and it *saturates at the
  trained μ_rec*: "training depth determines the test-time scaling ceiling" [T, 2604.12946v1
  §5.3]. A plateau is not a gain.
- **The iso-depth study at 20–400M**, the one paper with a compute-matched grid at our scale,
  explicitly refuses to treat recurrence as a test-time axis: "the mean recurrence in training
  caps the test-time frontier" [T, 2604.21106v4 §6, Appendix A].
- **The positives start at 0.6–1.1B and are conditional**: on reasoning accuracy, and only with
  depth-distribution/curriculum/conditioning training (Table 2A rows A11–A13). The same papers
  report knowledge loss or flat loss.
- **The one clean loss-based positive is Huginn at 3.5B**, and it is trained with mean recurrence
  32 and evaluated mostly below that mean; its Fig. 6 evaluates 1–64 [T, 2502.05171v2 §4.2].

So the precise statement is: *at ≥0.6B, on reasoning metrics, with training that exposes the
model to a depth distribution or conditions on the iteration index, `r' > r` can help; on
validation loss at ≤1.3B the only measured law saturates at the trained depth; at ≤200M there is
no positive result at all.* Our recipe is a fixed-3 carry loop at 95M — the intersection of the
two negative regimes.

---

## 4. The experiment

### 4.0 First, the arithmetic that kills the 30 MB design — measured, not asserted

The proposal is 30 MB of training, then eval at depths 1/2/3/4/6. Two facts make that
uninformative before it starts.

**(a) A `max_iter=3` model cannot be evaluated at depth 4 or 6.** The per-iteration parameters
are sized `max_iter`: `iter_embed` is `[max_iter, d]` (`loopblock.py:467`), `attnres` is a
`ModuleList(max_iter)` (`loopblock.py:472`), the gates are `[max_iter]`
(`loopblock.py:409, 420, 490`). `forward(depth=T)` asserts `1 <= T <= max_iter`
(`loopblock.py:608-609`); `--eval-depths` sweeps `1..=max_iter` (`loop.py:1259`); the halt
escape hatch, when `halt_mode` is on, is refused above `max_iter` with the message "the
`iter_embed`/gate rows stop at max_iter" (`config.py:4454-4461`). **The run as specified raises
`AssertionError`.** The model must be *built* with `max_iter=6`; that is a different snapshot,
not a flag on the 3-iteration one.

**(b) At 30 MB the loop is depth-blind.** Our own 102.7M loop, same `max_iter=3`, evals at
step `k` (bytes = k × 8 × 1024 × 3):

| bytes seen | step | CE iter 1 | CE iter 2 | CE iter 3 | marginal gain of iter 3 | source |
|---|---|---|---|---|---|---|
| 49 MB | 2 000 | 1.5482 | 1.5468 | 1.5468 | **0.0000 nats** | [M] `flagship2_stdout.log` |
| 246 MB | 10 000 | 1.4687 | 1.3421 | 1.3375 | 0.0046 | [M] |
| 442 MB | 18 000 | 1.4516 | 1.2892 | 1.2747 | 0.0145 | [M] |
| 836 MB | 34 000 | 1.4103 | 1.2171 | 1.1861 | 0.0310 | [M] |
| 1.23 GB | 50 000 | 1.3970 | 1.1933 | 1.1503 | 0.0430 | [M] |
| 2.11 GB | 86 000 | 1.3968 | 1.1762 | 1.1140 | **0.0622** | [M] |

The marginal value of the *third* iteration grows monotonically with log-bytes and is **zero at
49 MB**. At 30 MB the model has not learned to use two iterations, let alone three; a null at
depth 4 there measures an untrained block, not a ceiling. The cheapest run that reaches the
regime where depth matters is ~1.2 GB (5.3 h at the measured 63.2 kB/s), not 30 MB.

**This is the single most important correction in this report: the proposed experiment is
underpowered by 40–70× (30 MB against the 1.2–2.1 GB regime), and its null would be
uninterpretable.**

### 4.1 The fit that already answers the question (zero cost)

From the 2.11 GB row, fit `CE(T) = L∞ + Z·q^(T−1)` on the three measured points:

```
q  = (CE3 − CE2) / (CE2 − CE1) = 0.0622 / 0.2206 = 0.2819
Z  = (CE1 − CE2) / (1 − q)     = 0.2206 / 0.7181 = 0.3072
L∞ = CE1 − Z                  = 1.3968 − 0.3072 = 1.0896
CE4 = 1.0896 + 0.3072·0.2819³ = 1.0965     → Δ(3→4) = 0.0175 nats = 0.0253 BPB
CE6 = 1.0896 + 0.3072·0.2819⁵ = 1.0901     → Δ(3→6) = 0.0239 nats = 0.0345 BPB
```

[C, arithmetic on M numbers]. The deployable metric is `fin`, the mean readout: at depth 3 it is
1.1123, only 0.0017 nats below the 3rd iteration's own CE, so the mean tracks the last iteration
rather than acting as an ensemble — **the 4th iteration's contribution to `fin` is at most its
own 0.017 nats, and likely a little less** [S, the mean's depth-2 value is not in the log].

**Prediction: Δ(3→4) ≈ −0.02 BPB, against a pass bar of −0.03.** The geometric fit puts the effect
0.005 BPB (17%) under the bar, on the mature model, before any new training.

### 4.2 If the owner wants a new run: the cheapest *decision-grade* arm (1.2 GB, ~5 h)

This is the smallest run that reaches the regime where the loop's depth curve is non-trivial
(§4.0), and it uses only the public CLI. It trains at a **fixed depth 3** while the model is
*built* with 6 slots, so depths 4/6 are legal at eval and their rows are at init — the exact
"trained at 3, run 4/6" condition. Requires the 3-line `train_depth` seam below; without it,
the trainer runs `max_iter` (`loop.py:2162`).

**The 3-line seam** (only if you want the literal fixed-3 arm; `rand_depth` is the zero-code
alternative, but its uniform 1..6 sampling spends 1/6 of the steps at depth 1 and is a different
contract):

```
# 1) port/src/dormouse/config.py, ModelConfig, next to max_iter (line ~403):
#      train_depth: int = 0          # 0 = max_iter; the depth TRAINED at when slots exceed it
# 2) port/src/dormouse/train/loop.py:2162:
#      depth = (_sample_depth(cfg.seed, step, run.model.max_iter) if cfg.rand_depth
#               else (run.model.train_depth or None))
# 3) port/src/dormouse/config.py validate: refuse train_depth < 0 or > max_iter
```

Commands (the trainer reports 63.2 kB/s as `tokens_per_s` at b8/s1024/ga3,
`flagship2_stdout.log` step 87200; at 24,576 B per optimizer step that is ~2.6 steps/s;
1.2 GB = 4 883 optimizer steps ≈ 5.3 h; 30 MB = 1 221 steps ≈ 8 min):

```bash
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
STEPS=$((1200*1000*1000 / (8*1024*3)))          # 4883 optimizer steps = 1.2 GB
systemd-run --user --scope -p MemoryMax=40G --working-directory=$PWD \
  env PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u -m dormouse.cli train flagship_base \
    --data ../data_local/best/fwedu_train.bin \
    --eval ../data_local/best/fwedu_eval.bin \
    --set max_iter=6 --set train_depth=3 --set eval_depths=true \
    --set batch=8 --set seq_len=1024 --set grad_accum=3 \
    --set steps=$STEPS --set lr=5e-4 --set warmup_tokens=10000000 --set decay_frac=0.1 \
    --set eval=true --set eval_every=$((STEPS/10)) --set eval_bytes=196608 --set eval_offset=0 \
    --set seed=1 --set opt=mix --set moe_kernel=triton --set ns_kind=triton \
    --set ckpt_every=$STEPS --set ckpt_name=tt_fix3 --set out_dir=/tmp/opencode/tt_fix3 \
    --json-log /home/sehaxe/logs/tt_fix3.jsonl 2>&1 | tee /home/sehaxe/logs/tt_fix3.log
```

The eval line prints the whole curve as `depths=1:...(fin),2:...,...,6:...`
(`loop.py:1318-1323`), so one run yields all six points on one window.

**The 30 MB version** is the same command with `STEPS=1221`, `--set eval_every=122`, and
`ckpt_name=tt30`. **Run it only as a plumbing check** (does the depth-6 forward run, does the
`depths=` field appear). Its depth curve is not decision-grade (§4.0b). **Zero-code variant:**
replace `--set train_depth=3` with `--set rand_depth=true`; the trainer then samples `T` in 1..6
uniformly per step (`loop.py:974-986, 2162`), which is the C2 contract, not the literal fixed-3
one.

**The 40-minute version** (what the brief asked for, kept for completeness): `STEPS=$((150*1000*1000/(8*1024*3)))`
= 610 steps = 150 MB ≈ 40 min at 63.2 kB/s. Still eight times below the 1.2 GB regime, so the
same underpowering caveat applies.

### 4.3 The cheapest real measurement: expand the existing 5.75 GB checkpoint, no training

`port/checkpoints/flagship/flagship1.pt` is the mature C1 loop: 105,035,148 params, `max_iter=3`,
234,000 steps × 24 × 1024 = **5.75 GB** seen, `engram_ram=false` (so no `.ngram` sidecar is
needed — the missing sibling trap does not apply). Its per-iteration parameters are two kinds of
arrays: `loop.iter_embed [3,1024]` and `loop.attnres.{0,1,2}.query/.gain`. `AttnResSlot` is
zero-query / ones-gain (`loopblock.py:184-195`), so rows 4–6 at init are a well-defined
"no iteration identity" continuation.

**The measurement:** build the model with `max_iter=6`, append three zero rows to `iter_embed`,
three fresh `attnres` slots, load every other tensor unchanged, and call
`evaluate(..., depths=True)` on the 196,608 B window. Cost: one 997 MB load + six forward passes
of 196 KB — **minutes, no optimizer, no training**. This is the literal question ("trained at 3,
run 4/6") on the most mature model we own.

The script is in Appendix A. It is **untested** [S]: I did not run it (read-only, and this pass
is CPU-only). Two failure points to check before trusting it: (1) `RunConfig.from_snapshot` must
accept `max_iter=6` for a snapshot recorded at 3 (it is a model key, not a progress key, so a
*resume* would refuse — this is a fresh eval, not a resume); (2) the unpacked state dict's type
(`load_model_half` vs `unpack_state_dict`) must accept new keys. If either fails, the 1.2 GB arm
in §4.2 is the fallback.

Expected result, from §4.1: depth 4 ≈ 0.017 nats better than depth 3, depth 6 ≈ 0.024 nats
better, both **below the 0.03 BPB pass bar**; with the untrained slots at uniform readout, I
expect depth 4 to be *worse* than the fit [S].

### 4.4 Pre-registered pass rule (write this before the run)

Applies to §4.2 or §4.3; the same rule, same window, same weights.

- **Primary metric:** `fin` (CE of the mean readout) in BPB = `fin / ln 2`, from the `depths=`
  field, on the 196,608 B window at `off=0`, all depths on the same bytes.
- Let `Δ34 = BPB(d=4) − BPB(d=3)` and `Δ46 = BPB(d=6) − BPB(d=4)`.
- **PASS** iff `Δ34 ≤ −0.030` **and** `Δ46 ≤ +0.010` **and** the sign of `Δ34` reproduces on a
  second window (`--eval-offset 4000000`).
- **FAIL** iff `Δ34 ≥ 0`.
- **INCONCLUSIVE** for `−0.030 < Δ34 < 0`: treat as FAIL for the recipe. The effect is below the
  resolution of one run, and §5 explains why a sub-0.03 effect is not worth a 3-seed campaign.
- A PASS **nominates**, it does not establish: rerun the same arm with seeds 1/2/3 at the same
  budget; require mean `Δ34 ≤ −0.030` and 3/3 negative before any config changes.

**Why 0.030.** (i) The documented cross-window interval is **±0.19 BPB at 64 KB**
(`config.py:1945-1953`), ±0.11 at 196 KB; a paired same-window comparison cancels that term.
(ii) The residual paired noise on our own late-training curve is ~0.0045 nats ≈ **0.0065 BPB**
(per-iteration CE spread over steps 66k–86k, `flagship2_stdout.log`); 0.030 is ~4.6× that.
(iii) The repo's cross-seed spread at 9.2M was **0.048–0.073 BPB** (AGENTS §3.1); 0.030 is below
1σ of a *cross-run* claim, which is why a single seed can only nominate. (iv) §4.1 predicts
0.025 — deliberately, the bar is set just above the prediction.

---

## 5. What would make the answer "don't bother" — written before the run

D1. **The free fit already says no.** With the measured per-step ratio `q ≈ 0.28`, clearing the
0.03 BPB bar at `r+1` needs a marginal of `0.0208 / 0.28 = 0.074 nats` (0.107 BPB) at `r`. Our
measured marginal at `r = 3` is **0.062 nats**, which predicts 0.017 nats (0.025 BPB) for the
4th — below the bar. **If the fit on the final checkpoint gives < 0.02 BPB for `r' = r+1`, stop;
do not train anything.**

D2. **The 30 MB / 40-min run is not decision-grade.** If its `periter` shows the 3rd iteration
adding < 0.005 nats, the model is depth-blind; a null there must not be quoted as evidence.
This is a pre-registered *interpretation* rule, not a result.

D3. **Any `Δ34 ≥ 0` on the 1.2 GB arm or the expanded checkpoint closes the lane.** No
"maybe more data will fix it": the literature's fixed-depth models do not recover with scale;
they peak at the trained depth at 1.4B and 2.6B too.

D4. **`−0.030 < Δ34 < 0` also closes it for the recipe.** A 0.01–0.02 BPB effect, at +33%
inference FLOPs, is below our own resolution and below the effect sizes that have ever
survived a 3-seed gate here.

D5. **A non-monotone curve (4 better, 5/6 worse) closes it.** That is a tuning artifact at one
depth, not a scaling axis; Parcae's law is monotone with a single ceiling.

D6. **If the gain appears only with `rand_depth` and not with `train_depth=3`, stop calling it
test-time scaling.** It is then a training-time depth-distribution effect, and the correct
comparison is a deeper dense model at the same training FLOPs — which the 2026-11-06 dense
control already wins by 0.126 BPB.

D7. **If BPB is flat or worse while a reasoning benchmark improves, stop for the pretraining
recipe.** Our objective is held-out BPB; the only consistent positive in the literature is
reasoning accuracy with knowledge loss (2609.36636v1), which is a post-training question, not
this one.

D8. **If the expanded 5.75 GB checkpoint gets worse at 4 and the 1.2 GB arm cannot be afforded,
stop.** The mature-model measurement is the cheapest evidence we will ever have; a worse `fin`
there is the answer.

---

## 6. VERIFIED (sourced)

1. Ouro-1.4B base, trained max 4: 6-bench avg 72.39 at T=4; 71.32 / 71.14 / 70.49 / 69.87 at
   T=5/6/7/8; MMLU 67.45 → 64.49 at 8 — [T] 2510.25741v6 Table 10.
2. Ouro-1.4B-Thinking AIME24 65.00 at T=4 → 60.67 / 50.67 / 38.67 at T=5/6/8; Ouro-2.6B-Thinking
   AIME25 50.67 at T=3 → 24.33 at T=8 — [T] 2510.25741v6 Tables 12–13.
3. Ouro-1.4B run at 6 loops: accuracy 0.605 → 0.580, ECE 0.057 → 0.087 — [T] 2610.18127v1 §4.3.
4. Parcae: trained 140M/370M with μ_rec ∈ {2,4,6,8,10,12}, evaluated to T=24; "gains plateau near
   μ_rec, suggesting training depth determines the test-time scaling ceiling"; law
   `L(T)=L∞+Z e^{−zT}`, `z ∝ 1/μ_rec` — [T] 2604.12946v1 §5.3, Appendix L.
5. Parcae 1.3B, μ_rec=8: CORE 28.44 at T=8, Table 5; Fig. 7 evaluates to 2×μ_rec — [T/F]
   2604.12946v1.
6. Huginn: 4-layer recurrent block, **trained mean recurrence r̄ = 32** (Poisson-lognormal,
   truncated BPTT k=8), evaluated at r=4/8/16/32: ARC-C 27.99/35.15/37.71/38.23, HellaSwag
   43.46/58.54/64.67/65.21; r=1 ARC-E 34.89 vs r=32 69.91 — [T] 2502.05171v2 §4.2, Tables 1/4.
7. Huginn's "50B-equivalent" is a FLOP-budget prose claim, not a quality table: it "can
   continuously improve in performance with test-time scaling" to FLOP budgets equivalent to a
   standard 50B fixed-depth transformer — [T] 2502.05171v2 §5.1.
8. Retrofit: TinyLlama 1.1B mean-4 GSM8K 38.0 at r=4 → 37.7 / 37.9 at r=8/32; OLMo-1B mean-4
   38.7 → 39.5 / 40.6; OLMo mean-8 44.4 at 8 → 44.6 at 32 — [T] 2511.07384v1 Tables 3–5.
9. 2609.36636: BaseLoop 5×4 reaches reasoning 31.88 at L(r)=30 (training horizon L(K)=20);
   Qwen3-0.6B 7×4 at L(r)=56: BG 38.46 vs BaseLoop 36.59; recurrence improves reasoning beyond
   the horizon while degrading knowledge — [F/T] 2609.36636v1 Fig. 2, Fig. 6, abstract.
10. STARS: Ouro-1.4B-SFT 5-math avg 70.46 at 4 → 52.97 at 8; STARS 74.18 → 65.55 — [T]
    2605.26733v1 Table 2.
11. BabyLM: 97M (12L × K), BLiMP K=1/2/4/8 = 75.21 / 75.28 / 76.16 / 76.35; K=2 and K=8 are
    one seed each — [T] 2611.06206v1 Table 10.
12. Think-at-Hard, 1.7B: Always1 → Always2 NTP 73.1 → 79.7; oracle 81.8; second pass corrects
    8.7% and flips 2.1% of correct tokens — [T] 2511.08577v4 Table 1, §1.
13. Iso-depth (20–400M) explicitly declines to use recurrence as a test-time axis: "the mean
    recurrence in training caps the test-time frontier"; cites Parcae's saturating law and
    Ouro's no-gain-beyond-4 — [T] 2604.21106v4 §6 (Discussion), Appendix A.
14. Our model is a carry loop: `iter_embed[i]` added per pass, no input re-injection —
    [M] `port/src/dormouse/model/loopblock.py:467, 1038, 1068`.
15. Our loop cannot be evaluated above its built `max_iter`: `forward` asserts `T <= max_iter`
    (`loopblock.py:608-609`), `--eval-depths` sweeps `1..=max_iter` (`loop.py:1259`), and the
    halt cap is refused above it (`config.py:4454-4461`).
16. Our 102.7M loop at 2.11 GB: per-iteration CE 1.3968 / 1.1762 / 1.1140, `fin` 1.1123, BPB
    (fin) 1.605 over 65,536 B — [M] `~/logs/flagship2_stdout.log` eval step 86000; run summary
    `steps=171000, batch=8, seq_len=1024, grad_accum=3` (`flagship2.jsonl`).
17. Marginal value of the third iteration vs bytes: 0.0000 nats at 49 MB, 0.0046 at 246 MB,
    0.0145 at 442 MB, 0.0310 at 836 MB, 0.0430 at 1.23 GB, 0.0622 at 2.11 GB — [M] same log.
18. Eval window arithmetic: `eval_batches × batch × seq_len`; default `eval_bytes=196608`;
    documented 95% interval ±0.19 BPB at 64 KB — [M] `config.py:1945-1953`, `loop.py:1283`.
19. The mature loop checkpoint `port/checkpoints/flagship/flagship1.pt` is 105,035,148 params,
    `max_iter=3`, 234,000 steps at b24×s1024, `engram_ram=false`; its state dict has
    `loop.iter_embed [3,1024]` and `loop.attnres.{0,1,2}.{query,gain}` and no `.ngram` sidecar
    — [M] `flagship1.config.toml`; torch inspection of the state dict.
20. `AttnResSlot` is zero-query / ones-gain, so untrained slots start as an equal-weight average
    — [M] `loopblock.py:184-195`.

## 7. SPECULATION (my inference)

- **[S1] The 4th iteration will be worse than the geometric fit for a fixed-3 model.** The fit
  assumes the new iteration is exchangeable with the trained ones; its `iter_embed` and `attnres`
  rows are at init, so its readout is a uniform mixture the model never trained. Direction: the
  real `Δ34` is more negative than −0.025 BPB. Confidence: medium.
- **[S2] A byte-level model may have a *flatter* iteration curve than subword models.** Bytes
  are 4–5× more tokens per unit text and the loss surface is flatter; the same 3 passes may
  carry less serial computation per byte. If true, our marginal gains are smaller than the
  subword literature at the same parameter count, which strengthens the negative verdict.
  No published byte-level test-time curve exists at any scale.
- **[S3] Depth-distribution training at 95M might turn the curve from negative to flat**, because
  Parcae's 140M result is a *flat plateau* rather than a collapse. That would still fail D4
  (no detectable gain), but it would explain Ouro-style degradation as a training-contract
  artifact. Untested at 95M.
- **[S4] The 30 MB run's null is predetermined by §4.0b, not by a property of loops.** If someone
  runs it and reports "test-time iteration does not help at 95M", that conclusion is not
  supported: the run never entered the regime where the third iteration itself has value.
- **[S5] The `attnres` uniform readout at depth 4 may cost more than the extra computation
  gains.** The slot's query is zero, so the mixture is equal-weight over {embedding, body}; at
  depth 3 the trained slot has learned a non-uniform weighting. This is a second reason the
  expanded-checkpoint test is a lower bound, not an unbiased estimate.

---

## 8. Open questions

1. **Does a 95M model trained with `rand_depth` over 1..6 show any `fin` gain at 4–6?** The
   zero-code arm (§4.2, `--set rand_depth=true`) answers the depth-distribution contract at our
   scale in one 1.2 GB run. No published measurement below 140M.
2. **Does the `attnres` slot need training for depth 4 to be usable?** The expanded-checkpoint
   test confounds "more iterations" with "untrained readout slot". A 2-arm comparison (expand vs
   train-6-with-rand-depth) would separate them: the expand arm costs minutes, the trained arm
   ~5 h.
3. **Is the byte-level iteration curve flatter than the subword one?** A 3-point sweep at 9M
   params, 30–300 MB, depths 1–4, would measure the shape for ~2 h; the φ literature is all
   subword.
4. **Does the optimum shift if the readout is `last` instead of `mean`?** Our `fin` is the mean
   over iterations; a last-iteration readout would make depth 4's value exactly its own CE, not
   a diluted average. This is a 1-line ablation and it changes the mechanism (M5), not the
   weights.

---

## Appendix A. Untested script for the zero-training measurement (§4.3)

**Read this as a proposal, not a verified procedure.** It was not run. The two risky calls are
marked. If either fails, use §4.2.

```python
# port/scripts/expand_depth.py  (throwaway; do not commit)
import dataclasses, torch
from pathlib import Path
from dormouse.config import RunConfig
from dormouse.model import DormouseModel
from dormouse.train.data import open_streams, eval_batches_for
from dormouse.train.loop import evaluate, load_model_state, load_weights_only, unpack_state_dict

OUT = Path("checkpoints/flagship")
run = RunConfig.from_snapshot((OUT / "flagship1.config.toml").read_text())
model_cfg = dataclasses.replace(run.model, max_iter=6)          # 6 slots, 3 trained
train_cfg = dataclasses.replace(
    run.train, eval=True, eval_data="/home/sehaxe/dormouse/data_local/best/fwedu_eval.bin",
    eval_bytes=196608, eval_offset=0, batch=8, seq_len=1024,
)
run = dataclasses.replace(run, model=model_cfg, train=train_cfg)

ck = load_weights_only(OUT / "flagship1.pt")                     # safe-pickle loader
sd = unpack_state_dict(ck["model"], ck.get("model_quant_rung"))   # RISK 1: API name
d = sd["loop.iter_embed"].shape[1]
sd["loop.iter_embed"] = torch.cat([sd["loop.iter_embed"], torch.zeros(3, d)], 0)
for i in range(3, 6):
    sd[f"loop.attnres.{i}.query"] = torch.zeros(d)
    sd[f"loop.attnres.{i}.gain"] = torch.ones(d)

model = DormouseModel(model_cfg)
load_model_state(model, train_cfg, sd)                            # RISK 2: strict key set
model = model.to("cuda").eval()

_, eval_stream = open_streams(run, eval_only=True)
curve = evaluate(model, eval_stream, eval_batches_for(train_cfg),
                 device="cuda", depths=True)
for dep in sorted(curve):
    loss, scored, fin, per_iter = curve[dep]
    print(f"depth={dep} fin={fin:.4f} bpb_fin={fin/0.6931471805599453:.4f} "
          f"per_iter={['%.4f' % v for v in per_iter]} over {scored} B")
```

Two checks before believing a number: (1) the depth-3 `fin` must equal a **non-expanded** load of
the same checkpoint on the same window and the same eval file — run the same script once with
`max_iter=3` and no expansion, and compare; if it moves, the expansion perturbed the trained rows
and the script is wrong. (2) Do not compare the depth-3 `fin` with the snapshot's recorded 1.605
BPB: that number was taken on `mix4/web_eval.bin` over 65,536 B, and this script scores
`fwedu_eval.bin` over 196,608 B. Only the *within-run* depth differences are comparable here.

---

## Sources

1. Zhu et al., *Scaling Latent Reasoning via Looped Language Models* (Ouro), arXiv:2510.25741v6,
   2025-10-29 (v6 2026-10-11). Tables 10–13, §5.3.
2. Prairie, Novack, Berg-Kirkpatrick, Fu, *Parcae: Scaling Laws For Stable Looped Language
   Models*, arXiv:2604.12946v1, 2026-04-14. §5.3, Table 5, Figs. 7–8, Appendix L.
3. Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth
   Approach* (Huginn), arXiv:2502.05171v2, 2025-02-07 (v2 2025-02-17). §4.2, Tables 1/4, Fig. 6.
4. McLeish et al., *Teaching Pretrained Language Models to Think Deeper with Retrofitted
   Recurrence*, arXiv:2511.07384v1, 2025-11-10. §4.2–4.3, Tables 3–5.
5. Zhuang, Wang, Razzak, Liu, *What Makes Recurrence Effective in Looped Language Models?*,
   arXiv:2609.36636v1, 2026-09-29. §3, §5.2, Figs. 1–2, 6.
6. *Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language
   Models* (STARS), arXiv:2605.26733v1, 2026-05-26. §4–6, Table 2.
7. Hotta et al., *Loop Before You Leap: What Recurrent Depth Can and Cannot Buy in Data-Limited
   Pretraining*, arXiv:2611.06206v1, 2026-11-04. Tables 1, 10, Appendix J.
8. *Monitoring latent reasoning in looped language models*, arXiv:2610.18127v1, 2026-10-13.
   Table 1, Table 2, §4.3.
9. Schwethelm, Rückert, Kaissis, *How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for
   Looped Language Models*, arXiv:2604.21106v4, 2026-04-22 (v4 2026-10-28). §6 Discussion,
   Appendix A.
10. Fu et al., *Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning*,
    arXiv:2511.08577v4, 2025-11-11 (v4 2026-08-28). Table 1, §1.
11. Zhang et al., *PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking*,
    arXiv:2610.27628v1, 2026-10-20. §1/abstract (extrapolation 16× with progress-aligned
    targets).
12. *Test-Time Compute Scaling for ASR with Depth-Conditioned Looped Transformers* (LARM),
    arXiv:2606.04678v2, 2026-06-03 (v2 2026-06-15). Abstract ("naive looping does not fully
    exploit additional recurrent compute").
13. Repo: `port/configs/flagship_base.toml`; `port/checkpoints/flagship/flagship1.config.toml`;
    `port/src/dormouse/model/loopblock.py:184-195,409,420,467,472,490,608-609,1038,1068`;
    `port/src/dormouse/train/loop.py:1127-1142,1259,1283,1318-1323,2162`;
    `port/src/dormouse/config.py:403,1945-1953,2378-2384,4454-4461`;
    `~/logs/flagship2_stdout.log` (step 86000), `~/logs/flagship2.jsonl`,
    `~/logs/flagship1.log`.
14. Prior report: `docs/records/research/2026-11-06-depth-recurrence.md` (φ = 0.46, iso-FLOP verdict);
    `docs/records/reviews/2026-11-06-control-transformer.md` (dense 95M / 102.7M comparison).
