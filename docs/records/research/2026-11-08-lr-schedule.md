# 2026-11-08 — LR schedule, warmup and min-lr for the 2 GB dense-95M run

**Question.** The live recipe is WSD (`port/src/dormouse/train/lr.py`) with `lr=3e-4`,
`decay_frac` from the preset (0.1 everywhere — no preset overrides it) and `min_lr_frac`
default 0.1. The next big spend is the 1.9 GB dense-95M run (`port/scripts/train_budget.sh`,
231 934 steps at b8×s1024, ~9.4 h at the script's 56 kB/s). What schedule, warmup and
min-lr should it use, and what is the evidence — ours and the literature's?

**Tags.** `[M]` measured in this repo, with the artifact named · `[C]` computed here, formula
shown · `[S]` speculation, no direct measurement · `[T]` read from a paper, table or section.

**Scope.** CPU only. No config, checkpoint or GPU was touched. No commit.

---

## TL;DR (RU)

1. **Наш репозиторий не знает ответа про warmup.** Ни один прогон на живой форме (dense 95M,
   b8×1024) не имел `warmup_tokens>0`: у всех `warmup_tokens=0` (`train_budget.sh`,
   `recipe_ladder.sh`, `night_opt`, `flagship3`). Единственный прогон с warmup — `lr400` —
   провёл 733 из 800 шагов в warmup и недействителен как LR-тест. Единственное сравнение
   пика LR (3e-4 vs 6e-4 на 49 МБ, −0.25 BPB) искажено отсутствием warmup, и репозиторий
   сам это признаёт (`port/scripts/warmup_lr_ab.sh:1-20`).
2. **Единственное измерение min-lr — не результат.** `cooldown (min_lr_frac=0.0)` против
   control на 120 МБ: −0.0036 BPB, один сид. Это внутри 1σ≈0.011 на 73.7 МБ и ниже правила
   самого репозитория («arm wins if … more than the spread across the control's own 3 seeds»,
   `docs/protocols/AB-PROTOCOL.md`). До-decay контрольные точки дают прямую оценку
   воспроизводимости: те же конфиги, тот же сид, разница 0.0002–0.0073 BPB.
3. **Ни одна измеренная разница между WSD и cosine не превышает 0.06 aggregate score.**
   На 1B/100B это 0.03–0.06 (2405.18392v3 §3.3, Fig. 8b); K3 (2026) заявляет преимущество
   cosine при независимой настройке, но величины не публикует (2607.24653v2 §3.2).
   10% decay достаточно на нашем масштабе (MiniCPM, 0.036–0.17B); 5% достаточно на 200k шагах
   (2405.18392v3 Fig. 6). min-lr=0 улучшает loss, но на 1B портит downstream на 0.38
   (46.26 → 45.88, 2405.18392v3 §3.3).
4. **Рекомендация:** WSD, peak **3e-4**, warmup **1% (19 МБ / 2 320 шагов)**,
   decay **10% (190 МБ / 23 193 шага)**, min_lr **0.1 (3e-5)**, batch **8 × accum 1**.
   Цена ошибки в каждом из этих чисел ≤0.01 BPB; единственный риск масштаба 0.25 BPB —
   instability от LR без warmup — снимается warmup'ом.
5. **Дешёвый план:** 30 МБ/арм, 1σ≈0.02–0.03 (на 30 МБ не измерено; ближайшее — 0.051 на
   36.9 МБ, 0.019–0.022 на 49.2 МБ, 0.011 на 73.7 МБ). 30 МБ решает только warmup×LR
   (эффект 0.25) и batch (эффект неизвестен, литература заявляет 50× gap). min-lr и
   decay_frac (эффекты 0.004–0.01) на 30 МБ не видны — их решает только fork с общего
   чекпойнта: 28 МБ хвостов + 37 мин префикс, нулевая seed-компонента.
6. **Не трогать:** `muon_lr_mult=12`, `ns_steps=5`, bf16 NS/state, `opt=mix`,
   `decay_frac=0.1`, форму b8×s1024. Единственное исключение — `ema_decay`: скрипт ставит
   0.999, а ладдер шёл с 0.0; eval читает shadow (`loop.py:2335`), поэтому 0.999 даёт
   другое число и ломает сравнимость с 1.4906 на 120 МБ. Для этого прогона — **0.0**.

---

## 0. The recipe as it stands

| knob | value | where | note |
|---|---:|---|---|
| schedule | WSD: linear warmup → constant → **linear** decay | `train/lr.py:24-68` | the decay is linear, not cosine/1-sqrt |
| peak `lr` | 3e-4 | `config.py:1795`, `train_budget.sh` | Muon matrices get `lr × muon_lr_mult` = 3.6e-3 |
| `warmup_tokens` | **0** (default) | `config.py:2265` | every run on record inherited this |
| `decay_frac` | 0.1 | `config.py:2273` | no preset overrides it |
| `min_lr_frac` | 0.1 | `config.py:2280` | last step = 3e-5 |
| batch | 8 × 1024 × accum 1 = 8 192 B/step | `config.py:1793-1794` | 1 byte = 1 token |
| resume rules | `decay_frac`/`min_lr_frac` may change on resume; `warmup_tokens` may **not** | `config.py:56-77`, `2545-2615` | the tail can be re-planned; warmup cannot be added to an existing run |

The 2 GB run's arithmetic: `BYTES=1900000000` → 231 934 steps → 9.4 h at 56 kB/s. A 1%
warmup is 19 MB (5.7 min of wall clock). A 10% decay is 190 MB (~56 min).

---

## 1. Our data

### 1.1 The only LR comparison at the live shape

`night_opt`, dense 95M, b8×s1024, **AdamW** (not the live `opt=mix`), `grad_accum=3`
(24 576 B/step), `warmup_tokens=0` (the log's first LR at step 1000 is already the peak,
3e-4 / 6e-4 [M]), `decay_frac=0.1` (the step-2000 LR is exactly 0.1× peak), eval
`fwedu_eval.bin` 196 608 B at offset 0.

| run | bytes | peak lr | warmup | tail | min_lr | seeds | BPB | source |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `lr3e-4_s1` | 49.152 MB | 3e-4 | 0 | 10% → 0.1× | 0.1 | 1 | 1.9090 | `~/logs/night_opt/lr3e-4_s1.jsonl` [M] |
| `lr3e-4_s2` | 49.152 MB | 3e-4 | 0 | 10% → 0.1× | 0.1 | 2 | 1.8822 | `~/logs/night_opt/lr3e-4_s2.jsonl` [M] |
| `lr6e-4_s1` | 49.152 MB | 6e-4 | 0 | 10% → 0.1× | 0.1 | 1 | **2.1460** | `~/logs/night_opt/lr6e-4_s1.jsonl` [M] |

Mean 3e-4 = **1.8956**; Δ(6e-4 − 3e-4) = **+0.2504 BPB**, i.e. **11–13σ** against the pooled
1σ = 0.019–0.022 at 49.152 MB (`docs/records/reviews/2026-11-08-data-eval-integrity.md` §6). The
effect is real *as run*; what it does **not** separate is "6e-4 is bad" from "no warmup is
bad" — the repo's own script says exactly that, and `warmup_lr_ab.sh` was written and never
run (`~/logs/warmup` does not exist).

The only other LR probe near our scale is `lr400` (199M, `port/configs/m400.toml`, 800 steps, **warmup 6M
tokens** = 733 of 800 steps, `decay_frac=0.4`, `min_lr_frac=0.01`): all three arms spent 92%
of the run inside warmup and never held a peak. 3e-4 / 5e-4 / 1e-3 → 2.723 / 2.657 / 2.750
(two seeds each, spread 0.01–0.13). **Void as an LR ranking** [M].

### 1.2 The ladder: the only schedule arm on record

7 arms × 120.0 MB, b8×s1024, `opt=mix`, `muon_lr_mult=12`, `warmup_tokens=0`,
`decay_frac=0.1`, `mix_refetch`, eval 196 608 B at offset 0, one seed each.

| arm | @30 MB | @60 MB | @90 MB | @120 MB | Δ vs control @120 MB |
|---|---:|---:|---:|---:|---:|
| all (vr+gate+clip+nope+cooldown) | 1.7373 | 1.6443 | 1.5797 | 1.4778 | −0.0128 |
| **cooldown (`min_lr_frac=0.0`)** | 1.7818 | 1.6681 | 1.6040 | **1.4870** | **−0.0036** |
| value_residual | 1.7806 | 1.6668 | 1.5953 | 1.4878 | −0.0028 |
| ffn_clip=7 | 1.7710 | 1.6624 | 1.5994 | 1.4889 | −0.0017 |
| **control** | 1.7745 | 1.6680 | 1.6002 | **1.4906** | — |
| attn_gate | 1.7758 | 1.6675 | 1.6036 | 1.4937 | +0.0031 |
| nope_layers=2 | 1.7486 | 1.6556 | 1.5966 | 1.4958 | +0.0052 |

Source: `docs/records/reviews/2026-11-08-night.md` §5; artifacts `~/logs/ladder/*_refetch.jsonl` [M].

**The pre-decay columns are a free reproducibility experiment.** The decay starts at step
13 183 (90% of 14 648) = 108 MB, so at 30/60/90 MB the control and cooldown arms are the
same computation with the same seed and the same data order — they *should* be identical:

| bytes | control | cooldown | Δ (should be 0) |
|---:|---:|---:|---:|
| 30.0 MB | 1.7745 | 1.7818 | **+0.0073** |
| 60.0 MB | 1.6680 | 1.6681 | +0.0002 |
| 90.0 MB | 1.6002 | 1.6040 | **+0.0039** |

RMS = **0.0048 BPB** [M]. That is the run-to-run floor of the *same* configuration (CUDA
non-determinism), not the seed spread. It is the resolution limit of any one-seed A/B at
this budget: a 0.0036 "win" is below it.

### 1.3 Long runs that are not the live shape (context only)

| run | params | bytes | peak lr | warmup | tail | BPB | source |
|---|---:|---:|---:|---|---:|---:|---|
| `flagship3` | 95.0M | 245.76 MB | 3e-4 | 0 | 10%, never reached (killed at 10k of 40k steps) | 1.5185 | `~/logs/flagship3.jsonl` [M] |
| `f3short` | 95.0M | 98.30 MB | 3e-4 | 0 | 10% → 0.1× | 1.9272 | `~/logs/f3short.jsonl` [M]; **pre-Muon-fix, retracted class** |
| `m400` | 199.3M | 1.229 GB | 5e-4 | 0 | 10%, never reached | 1.7622 | `~/logs/m400.jsonl` [M] |
| `flagship2` | 102.7M | 2.114 GB | 5e-4 | 0 | 10%, never reached | 1.7309 | `~/logs/flagship2.jsonl` [M] |
| `flagship_resume` | 102.7M | 4.178 GB | 5e-4 | unknown | **40% → 0.01×**, inferred from the LR trace | 1.7292 | `~/logs/flagship_resume.jsonl` [M] |

`flagship_resume`'s tail: from step 122 000 (3.00 GB) to 170 000 (4.18 GB) the LR fell
3.60e-4 → 1.22e-5 and BPB moved 1.7338 → 1.7292 = **−0.0046 over 1.18 GB** [M]. The decay
start is step 102 600 (LR trace slope 7.237e-9/step, landing at 5e-6 = 0.01×5e-4 at step
171 000 [C]). This is not an A/B — continued training is in the same interval — but it is
our only long-run observation of a decay tail, and it says the tail is worth ~0.005 BPB at
4 GB, not 0.05.

What the repo lacks is not a constant-LR curve but a **matched pair**: `flagship3` (95M, 3e-4)
is a stable-LR stretch to 245.76 MB whose decay was never reached, `m400` (199M, 5e-4) is one
to 1.229 GB, `flagship2` (102.7M, 5e-4) to 2.114 GB. None has a sibling that ran the same
bytes with a decay, and none is scored on the ladder's 196 608 B / `mix_refetch` window. So the
decay's contribution cannot be read off our logs; it is bounded only by the `flagship_resume`
tail (0.0046 over 1.18 GB, confounded) and by MiniCPM/WSD.

### 1.4 What our logs cannot answer, and what the answer costs

| question | what exists at 95M / b8×s1024 | gap | price of a real answer |
|---|---|---|---|
| does warmup matter? | **zero** runs with `warmup_tokens>0` | no measurement at any budget | 3 arms × 3 seeds × 30 MB = 81 min |
| 3e-4 vs 6e-4 with warmup | one 6e-4 run, warmup=0, AdamW | the confound the repo names | 2 arms × 2 seeds × 30 MB = 36 min |
| is 3e-4 right at 2 GB? | longest 3e-4 run = 245.76 MB (killed) | 8× extrapolation in bytes | **2 full runs = 18.8 h** |
| decay_frac | **zero** arms; 0.1 everywhere | no 5/10/20 comparison | fork test: 6/12/24 MB tails = 28 min + one 120 MB prefix with checkpoints (37 min) |
| min_lr 0 vs 0.1 | one arm, one seed, −0.0036 | inside noise | fork test, 3 tails × 12 MB = 11 min (on a 120 MB checkpoint) |
| batch × LR | **zero** loss measurements; `bsweep*.log` is a throughput driver only, `sweep_blr/` does not exist | totally unmeasured | 3 arms × 2 seeds × 30 MB = 54 min |
| LR at 1B / 22 GB in our code | nothing — the 1B recipe has never run | — | not priced: no measured byte rate exists for that model |

No 95M long-context or long-budget LR result exists in the literature either: the closest
small-scale schedule ablations are 124M/210M (2405.18392) and 36M–170M (2404.06395).

---

## 2. Literature

Every row names the paper id, the setting, the number, and what it can support. Where a
number is not published, the cell says **NOT PUBLISHED**.

| claim | paper (id v) | setting | number | supports | tag |
|---|---|---|---|---|---|
| constant LR + short cooldown matches cosine | 2405.18392v3 §3.1, Fig. 3 | 210M, SlimPajama 6B tok, seq 512, batch 200, AdamW | "almost perfect match"; the cooldown's best peak LR sits "slightly below … half of the optimal cosine maximum LR" (Fig. 3 caption) | WSD is a legitimate default | [T] |
| how long the cooldown must be | 2405.18392v3 §3.1, Figs. 5-6 | 124M, 20k-200k steps | benefits plateau at ~20%; 10-20% surpasses cosine; on a 200k-step run **10k steps (~5%)** with (1-sqrt) "almost perfectly match cosine" | 10% is enough at 232k steps; 5% is defensible | [T] |
| warmup length | 2405.18392v3 §A.1, Table 2 | 33M-360M | 300 steps for most runs (= 30.7M tok, 0.15-0.5% of 6-20B tok); **1000-3000 steps** for runs above 100k steps (0.5-1.5% of a 20B-token run); batch 200×512 = 102 400 tok/step | 1% is inside the published band | [T] |
| cosine's endpoint | 2405.18392v3 §3.1 | — | cosine decays to **10%** of peak in their baseline | 0.1 is the standard floor | [T] |
| **min-lr 0 vs 0.1, downstream** | 2405.18392v3 §3.3, Fig. 8b | **1B, 100B tokens, FineWeb**, lr 8e-4, batch 1.8M | aggregate score: cosine→10% **46.26**, cosine→0 **45.88**, linear-20% **46.20**, (1-sqrt)-20% **46.23**; "cosine to zero improves the loss … leads to a saturation before the end of training" | decay-to-zero buys loss and costs downstream at 1B | [T] |
| decay fraction at small scale | 2404.06395v3 §4.3 "10% Steps are Enough", Fig. 6 | 0.036B, 0.17B, 40N-80N tokens | decay of **10% of total tokens** is sufficient; **2.5% falls short** | 10% at 95M is the published minimum | [T] |
| decay below 10% of peak | 2404.06395v3 §6.4, Fig. 12 | 1.2B MiniCPM | "the loss still drops after the learning rate drops below 10% of the max" | a 0.1 floor is not loss-optimal | [T] |
| cooldown shape | 2508.01483v1 §4, Figs. 4-7 | 210M, 33k-step cooldowns, 3.4B tok | shape changes final perplexity; sqrt / lowered-linear-0.7 best; β2 tuning during cooldown is worth as much as the shape | our linear tail is the weakest of the good shapes, by a small margin | [T] |
| normalized optimizers need η→0 | 2607.12360v1 Props. 1-3 | theory, strongly convex + affine-variance noise | SGD self-anneals and the optimal cooldown fraction is 0; sign/normalized methods have an **η²** floor and "reach the minimizer only as the learning rate is driven to zero" | for Muon/AdamW, a low final LR should help *training loss* | [T] |
| optimal schedule shape | 2602.06797v3 abstract | power-law kernel regression | hard-task regime → WSD-like; decay exponent 2q−1; "overly slow terminal decay leads to schedule-induced capacity saturation" | a long, slow tail can hurt | [T] |
| WSD vs cosine at 2026 frontier | 2607.24653v2 §3.2 | Kimi K3, 2.78T total | "cosine decay consistently achieves a lower final loss than WSD" **under independently tuned hyperparameters**; magnitude **NOT PUBLISHED** | the strongest contrary evidence to WSD | [T] |
| warmup at frontier | 2607.24653v2 §3.3 | K3 | **1% linear warmup** | 1% is a frontier default | [T] |
| cosine + a floor at frontier | 2602.15763v2 App. A | GLM-5, 744B, Muon | warmup 0 → 2e-4, decay to **4e-5 = 0.2× peak** | a nonzero floor is what a 2026 Muon run ships | [T] |
| warmup + cosine at 7B | 2501.00656v3 Table 3, §4.1 | OLMo 2 7B/13B/32B | **2000 steps** warmup (0.17-0.34% of 5T tok); peak 3e-4 / 9e-4 / 6e-4; cosine to **10%**; the last cosine part can be replaced by "a linear decay to zero with little loss of performance" | short warmup, 10% floor, zero at mid-training | [T] |
| does the LR optimum move with D? | 2501.00656v3 §4.1, Fig. 11 | OLMo 2 7B, 3e-4 vs 6e-4 | higher LR is better early; **3e-4 overtakes 6e-4 "well past 200B tokens"** | at 2B tokens, a higher LR may still be winning | [T] |
| LR vs N, D fit | 2608.28308v2 Eq. 7 | 0.13-1.7B, 6-300B tok, WSD 20% → 1e-5, warmup 2000 steps | **η\* = 0.371·N^−0.417·D^0.0862** (D exponent p=0.0055) | no published law says to *lower* the peak LR for 16× more bytes; this one says slightly raise it | [T] |
| LR vs batch, adaptive | 2006.09092v6 abstract | theory + VGG/WideResNet | SGD → linear scaling; **Adam → square-root scaling** | sqrt is the adaptive default | [T] |
| critical batch | 1812.06162v1 abstract | many domains | gradient noise scale predicts the largest useful batch | our batch may be far below it | [T] |
| batch instead of LR decay | 1711.00489v2 abstract | SGD/momentum/Adam, ImageNet | increasing batch can replace LR decay; B ∝ ε | a batch change is a schedule change | [T] |
| WSD-specific batch | 2601.05034v2 §3.3 Fig. 3, §5.1 | 122M-1B, Qwen3 MoE ablation | B_opt ≈ 4-6×10^5 tokens at 122M-1B; **"scaling the learning rate offers no performance improvement"** when growing batch 2M→6M | our 8 192-token batch is far below B_opt, but do not sqrt-scale the LR on faith | [T] |
| small-scale LR stability | 2309.14322v2 abstract | small Transformers | high LR reproduces large-scale instabilities; warmup/weight-decay/μParam mitigate | warmup is a stability instrument, not a quality knob | [T] |

### 2.1 The citation in the brief is wrong

The brief cites **arXiv:2404.14219** for WSD-vs-cosine-vs-linear. That id is **"Phi-3
Technical Report"** (v4, 2024-08-30) — a model report with no LR-schedule ablation
[verified against the arXiv API, 2026-11-08]. The WSD paper is **arXiv:2405.18392**
(Hägele et al., "Scaling Laws and Compute-Optimal Training Beyond Fixed Training
Durations"); the small-model WSD paper is **arXiv:2404.06395** (MiniCPM). Any claim in this
repo that cites 2404.14219 for a schedule result is mis-attributed and should be repaired.

### 2.2 WSD vs cosine: a tie at ≤360M, and an unpublished cosine win at frontier

- 2405.18392v3 trains 33M-360M with both and reports an "almost perfect match" in loss
  (Fig. 3) and 46.26 vs 46.20-46.23 aggregate at 1B/100B (§3.3, Fig. 8b). The gap is
  **0.03-0.06 aggregate score**, i.e. noise for a decision.
- 2607.24653v2 (K3) says the opposite at frontier scale, and is careful about the confound:
  the two schedules have different optimal peak LR and batch, so a shared-hyperparameter
  comparison is unfair. Under independent tuning, cosine wins. **No number is published**,
  so the size of the win cannot be quoted — only its direction.
- 2602.06797v3 says the *optimal* shape depends on task hardness: hard tasks → WSD-like,
  easy tasks → power decay from step 0. Our byte-level web mixture is not characterised on
  that axis.

**Reading:** at 95M neither schedule has a measured or published BPB gap. The published 1B
gap is 0.03-0.06 aggregate score. Switching to cosine costs a code path, invalidates every
curve in the repo, and buys an unpublished amount. Keep WSD; the honest reason is
comparability and the branch property (§4), not superiority.

### 2.3 Decay fraction: 10% is the consensus minimum, 20% the plateau

- 2404.06395v3: **10% of tokens is enough; 2.5% is not** (0.036B and 0.17B — our scale).
- 2405.18392v3: the benefit plateaus at ~20%; 10-20% surpasses cosine; on a 200k-step run
  **10k steps (5%)** with a (1-sqrt) shape nearly matches cosine (Fig. 6).
- 2508.01483v1: 20% is the common choice and the shape matters more than the endpoint.
- 2608.28308v2: 20% decay to an absolute 1e-5 on every rung, 6-300B tokens.

Our 10% sits at the low end of the supported range and at the exact value MiniCPM measured
as sufficient. Going to 20% costs 10% of the budget (190 MB, ~56 min) for a gain the
literature bounds at "plateau", i.e. ≤0.005. Do not.

Two theory papers say why the tail is not substitutable by more stable-phase steps.
2410.05192v3 ("river valley") argues the stable phase moves *along* the valley and the decay
descends *into* it, so a stable-phase checkpoint has not seen what its decay sees. 2503.12811v1
fits the final loss as a function of the summed LR plus a **separate additive decay term**
`L_D`, whose size depends on the decay's shape — decay steps and stable-phase steps are not
interchangeable at equal summed LR.

### 2.4 Warmup: 0.2-1.5% of tokens, and 0 is the outlier

| source | warmup | as a fraction of the run |
|---|---|---|
| 2405.18392v3 §A.1 | 300 steps at 102 400 tok/step = 30.7M tok | 0.5% of 6B |
| 2405.18392v3 §A.1 | 1000-3000 steps for >100k-step runs | 0.5-1.5% |
| 2501.00656v3 Table 3 | 2000 steps at 4.2-8.4M tok/step | 0.17-0.34% of 5T |
| 2607.24653v2 §3.3 | 1% linear | 1% |
| 2608.28308v2 §3.1 | 2000 steps fixed; runs with warmup >30% of D excluded as degenerate | 0.35-17% |

No paper reports a no-warmup arm at our scale. 2309.14322v2 is the mechanism: a high LR
without warmup reproduces the large-scale instability class, and warmup is one of the
mitigations. Our 6e-4 datum is consistent with that failure mode [S], which is why it cannot
be read as an LR optimum.

### 2.5 The min-lr floor: the one place with a real published number

2405.18392v3 §3.3, Fig. 8b, **1B parameters, 100B tokens, FineWeb, lr 8e-4, batch 1.8M**:

| tail | aggregate score |
|---|---:|
| cosine → 10% | **46.26** |
| cosine → 0 | **45.88** |
| linear cooldown 20% → 0 | 46.20 |
| (1-sqrt) cooldown 20% → 0 | 46.23 |

The text is explicit: decaying to zero *improves the loss* but "leads to a saturation before
the end of training, hurting overall performance". So the answer depends on the metric:

- **BPB (our only metric):** 0 is weakly better. Our own arm agrees in sign (−0.0036, one
  seed, inside noise); 2607.12360v1 predicts it for normalized optimizers (Muon/AdamW).
- **Downstream (any later SFT/RL use):** 0.1 is better by 0.38 aggregate at 1B — 6× larger
  than any schedule effect we can measure at 95M.

There is no published 95M number for this. It is measured at 1B and inferred below.

### 2.6 Batch × LR: linear, sqrt, or neither

- 2006.09092v6: linear for SGD, **sqrt for Adam**, derived from the batch-Hessian spectrum
  and validated on VGG/WideResNet (not an LLM).
- 2601.05034v2 §5.1: on Qwen3 MoE with WSD, growing the batch 2M→6M with **sqrt scaling vs
  constant LR** gave "no performance improvement" from the LR scaling. Their B_opt at
  122M-1B is ~4-6×10^5 tokens (Fig. 3), 50-75× our 8 192-token step.
- 1711.00489v2: for SGD, increasing the batch can replace the LR decay — the two knobs are
  not independent.
- 2501.00656v3: at 7B, 3e-4 beats 6e-4 only after 200B tokens; before that the higher LR wins.

Our batch/LR is **entirely unmeasured** (`sweep_blr/` absent; `bsweep.log` is a throughput
driver: b8 rc=0, b16 rc=1, b32 rc=1). The literature disagrees with itself, so no scaling
rule can be imported. Hold the batch fixed for the 2 GB run.

---

## 3. Recommendation for the 2 GB run

Run: `BYTES=1900000000 BATCH=8 ACCUM=1 LR=3e-4`, 231 934 steps, ~9.4 h.

| knob | value | reason | falsifier |
|---|---:|---|---|
| schedule | **WSD, linear tail** | 2405.18392v3 matches cosine at 33M-360M (Fig. 3) and differs by 0.03-0.06 aggregate at 1B/100B; K3's cosine win is unpublished in magnitude; WSD is the only schedule the code has and the only one our curves use | a cosine-tail arm at 120 MB, 3 seeds (2 × 37 min); if cosine beats WSD by >0.01, switch |
| peak `lr` | **3e-4** | the only peak with a measured win at our shape (3e-4 vs 6e-4 = −0.25 BPB at 49 MB, 11σ); no published law lowers η\* with D (2608.28308v2 Eq. 7 has D^+0.086); 5e-4 was stable to 1.2-4.2 GB on 102.7M-199M [M] | 30 MB, 3 arms × 3 seeds, warmup 1%: if 6e-4 beats 3e-4 by >2σ (~0.03), raise to 5e-4 |
| `warmup_tokens` | **19 000 000** (1.0% = 2 320 steps = 19 MB) | 2607.24653v2 uses 1%; 2405.18392v3's >100k-step band is 0.5-1.5%; 2501.00656v3 uses 2000 steps (0.2-0.3%); 0 is the outlier and is the confound in our only LR datum | 30 MB, 3 arms × 3 seeds {0, 0.3%, 3%}: if 0 ties 1%, drop it; if 3% wins, lengthen |
| `decay_frac` | **0.1** (190 MB, 23 193 steps, ~56 min) | 2404.06395v3: 10% sufficient at 0.036-0.17B, 2.5% not; 2405.18392v3: 10-20% matches cosine, plateau at 20%; 5% nearly matches at 200k steps | fork at 120 MB {5%, 10%, 20%}: if 20% wins by >0.01, lengthen; if 5% ties, shorten |
| `min_lr_frac` | **0.1** (3e-5) | 2405.18392v3 §3.3: decay-to-0 improves loss but saturates downstream at 1B (46.26 vs 45.88); our 0-vs-0.1 delta is −0.0036, inside 1σ and below the repo's own spread rule; the checkpoint keeps its optionality | fork {0, 0.05, 0.1}: if 0 wins by >0.01 BPB **and** the run is only a BPB artifact, switch to 0 |
| batch/accum | **8 / 1** (8 192 B/step) | every number in the repo is at this shape; no batch-loss measurement exists; 2601.05034v2 shows LR scaling with batch gave no gain; changing it invalidates the 120 MB curve | 30 MB, 3 arms × 2 seeds {a1, a4, a16} at constant LR: if a4 wins by >0.03, use a4 |
| `ema_decay` | **0.0** for this run (the script's 0.999 would report a better, incomparable number) | the eval scores the shadow (`loop.py:2335`); the ladder ran 0.0, so 0.999 makes the 2 GB curve a *new baseline* at every matched byte, and the shadow's gain is unmeasured | eval raw vs shadow at 120 MB; if the gap is <0.005, either is fine |

**Wall clock.** 9.4 h + a 56-minute tail is already inside the 9.4 h (the decay is part of
the budget). The 1% warmup adds ~6 min of low-LR time.

**Launch line.** `train_budget.sh` has no `WARMUP`/`EMA` env var and does not forward `"$@"`,
so the run needs two lines in the script (or a direct `dormouse train` call):

```
--set warmup_tokens=19000000 --set ema_decay=0.0
```

Everything else in `train_budget.sh` stays as it is: `lr=3e-4`, `seq_len=1024`, `seed=1`,
`opt=mix`, `muon_lr_mult=12`, `ns_dtype=bf16`, `muon_state_dtype=bf16`, `decay_frac` 0.1
(never passed, so the default), `min_lr_frac` 0.1 (same), `ckpt_every=1000`.

**The three things that would make me change the recommendation**

1. The 30 MB warmup test shows 6e-4 with warmup is not worse. Then 5e-4 is the candidate,
   and the 2 GB run becomes the first test of it at 245+ MB — a risk, because no 95M run at
   5e-4 exists beyond 0 MB.
2. The 120 MB fork shows decay-to-zero beats 0.1 by >0.01 BPB. Then `min_lr_frac=0.0` for a
   BPB artifact, 0.1 for a checkpoint.
3. The batch test shows a4 wins by >0.03. Then accum 4, and the LR stays 3e-4 (2601.05034v2
   found no gain from scaling it).

**What the schedule is worth versus the data.** From 120 MB to 1.9 GB is 3.99 doublings; the
repo's measured marginal is ≈0.17 BPB/doubling over 0.05-0.25 GB
(`docs/records/research/2026-10-18-data-capacity.md:18`), which extrapolates to ≈0.68 BPB at 1.9 GB
[S]. The entire schedule decision space in the table above is 0.004-0.01 BPB — **1.5% of the
run's expected gain**. The one exception is the instability risk: 6e-4 with no warmup cost
0.25 BPB, which is 37% of the expected gain. That is why warmup is the one non-negotiable
change, and why nothing else should be swept on the expensive run.

---

## 4. The cheapest test plan

### 4.1 The noise floor, named

| budget | 1σ (pooled over arms) | source |
|---:|---:|---|
| 36.864 MB | **0.051** | 3 arms × 2 seeds, `data-eval-integrity.md` §6 [M] |
| 49.152 MB | **0.019-0.022** | 3-6 arms × 2 seeds, same §6 [M] |
| 73.728 MB | **0.011** | 3 arms × 2 seeds, same §6 [M] |
| 120.0 MB | **not measured** | no multi-seed arm at this budget |
| 30.0 MB | **not measured** | nearest is 0.051 at 36.9 MB; 1/√D interpolation gives 0.028 from 49.2 MB and 0.017 from 73.7 MB [C] |
| 1.9 GB | **not measured** | 1/√D from 73.7 MB gives 0.0021 [C], unverified over 26× |
| same-config, same-seed, same-data (30-90 MB) | **RMS 0.0048** (0.0002-0.0073) | ladder control vs cooldown before decay, §1.2 [M] |

**Honest answer for 30 MB/arm:** 1σ ≈ **0.02-0.03**. That is an interpolation, not a
measurement. A 2-arm, 1-seed A/B at 30 MB needs a **≥0.06** delta to be a result at 2σ; a
3-seed arm needs **≥0.035**. **I cannot scale the 73.7 MB number down with confidence** —
the only sub-49 MB measurement in the repo is 0.051, and it is inflated by one arm with a
0.117 spread. Any plan that claims a 0.01 effect at 30 MB is claiming 0.3σ.

### 4.2 Tests that 30 MB *can* settle (the top-2 by expected value)

Cost: 30 MB at 56 kB/s = **9.1 min/arm**, b8×1024, `mix_refetch`, eval 196 608 B.

**Test A — warmup × peak LR (settles the confound in our only LR datum).**
Arms, 3 seeds each: `3e-4 / warmup 0`, `3e-4 / warmup 0.3 MB (1%)`, `6e-4 / warmup 0.3 MB`.
9 arms × 9.1 min = **82 min**. Decision rule: if 6e-4+warmup is within 0.035 of 3e-4+warmup,
the 0.25 BPB gap was the missing warmup and the peak re-opens — 5e-4 goes to a 120 MB confirm,
it does not go straight into the 2 GB run; if 6e-4 stays worse by >0.035, keep 3e-4 and close
the question.

**Test B — batch × LR (the least-explored axis, and the only one with a literature claim of
a 50× gap).** Arms, 2 seeds each: `b8 a1 lr3e-4`, `b8 a4 lr3e-4`, `b8 a4 lr6e-4` (the last
one tests whether the LR should move with the effective batch). 6 arms × 9.1 min =
**55 min**. Decision rule: if a4 at constant LR beats a1 by >0.035, the 2 GB run uses accum 4
and the same 3e-4; if a4+6e-4 wins, the LR moves with the batch, contrary to 2601.05034v2.

Total for A+B: **~2.3 h**, before a 9.4 h spend. Both tests are sized to detect effects above
the 30 MB floor, not to chase the 0.004-0.01 effects that live in the tail.

### 4.3 Tests that 30 MB *cannot* settle — use a fork instead

min-lr and decay fraction change only the last 5-20% of a run. A 30 MB arm that ends with a
3 MB tail cannot resolve a 0.004-0.01 effect against 0.02-0.03 noise. The WSD structure
gives a better instrument: **branch from one common checkpoint**, so the pre-tail trajectory
is identical and the seed/init/data-order variance cancels. The remaining difference is
CUDA non-determinism, RMS 0.0048 at 30-90 MB [M].

**Test C — tail fork at 120 MB.** One `control`-style prefix run with `ckpt_every=1000` and
`decay_frac=0` (constant LR after warmup, 37 min) gives checkpoints every 1000 steps = 8.2 MB.
Fork at a checkpoint `C` and set `decay_frac = (steps − C) / steps`; the decay then starts
exactly at `C` (both keys are progress keys). Arms: `min_lr ∈ {0, 0.05, 0.1}` over a fixed
10% tail (12 MB), and `decay_frac ∈ {5%, 10%, 20%}` (6, 12 and 24 MB tails, forking at
≈114.7, 106.5 and 98.3 MB). Total tail compute = 3×12 + 6 + 12 + 24 = **78 MB ≈ 28 min**,
plus the 37 min prefix. Everything but the tail is shared, so a 0.005 difference is
measurable. Decision rule: pick the best tail, and note that the answer is measured at
120 MB, not 1.9 GB.

### 4.4 The definitive version, on the real run

`train_budget.sh` already sets `ckpt_every=1000` and `decay_frac`/`min_lr_frac`/`steps` are
progress keys (`config.py:56-77`), so a tail branch on the 2 GB run is a legal resume with no
code change:

1. run the stable phase with `--set steps=208000 --set decay_frac=0` (constant LR after warmup;
   1.704 GB);
2. branch three resumes with `--set steps=231934 --set decay_frac=0.10319
   --set min_lr_frac=...` — the 23 934-step (196 MB, ~58 min) tail starts exactly at step
   208 000 because `decay_steps = round(0.10319 × 231 934) = 23 934`. Arms:
   `min_lr_frac ∈ {0, 0.05, 0.1}`;
3. keep the winner. Extra cost: **2 × 58 min = 1.9 h** on top of 9.4 h, and the min-lr
   question is answered at 2 GB with a paired design.

This is the only design that answers the question at the scale it is asked. The 120 MB fork
answers it at 120 MB, where the effect is known to be ~0.004 and possibly scale-dependent.

---

## 5. What NOT to change

| knob | keep | evidence it is settled |
|---|---|---|
| `muon_lr_mult` | **12** | m8 vs m12 at 73.7 MB: 0.0036 = 0.3σ (inside); m16 is +0.022 (1.95σ, not clean). 12 is the plateau choice, not a measured optimum (`data-eval-integrity.md` §6; `~/logs/sweep_mult/`) [M] |
| `ns_steps` | **5** | 3 steps costs **+0.274 BPB** at 36.9 MB = 5.4σ — a real regression (`docs/architecture/2026-11-08-the-16x.md`, `data-eval-integrity.md` §6) [M] |
| `ns_dtype` / `muon_state_dtype` | **bf16 / bf16** | +0.006 / +0.008 BPB, both inside 16 MB noise; the state rung measured +18% step (`2026-11-05-opt-region.md`) [M] |
| `opt` | **mix** (Muon + AdamW), `muon_lr_mult=12` | `muon_lr_mult=12` vs AdamW: 0.147 BPB better at 49.2 MB = 6.6σ, 2 seeds vs 2 (`~/logs/night_opt3/m12_*` vs `adam_*`) [M] |
| `decay_frac` | **0.1** | §2.3: 10% is the published minimum at 0.036-0.17B; the flagship's 0.4 tail moved 0.0046 over its last 1.18 GB, confounded with continued training [M] |
| shape | **b8 × s1024 × accum 1** | the only shape with a measured curve; every comparison in §1 is at it |
| `doc_masking` | **false** | the `mix/` sources have no document boundaries (`2026-11-08-night.md` §1) |
| `z_loss`, `logit_softcap` | **absent** | they were proposed only as a patch for the 6e-4 instability and have never been A/B'd at 95M; adding them to the 2 GB run makes it a 2-variable experiment |
| `use_msa`, `use_tsct` | **off** | dense won at equal bytes (`2026-11-05-dense-beats-bank.md`, `2026-11-05-final-dense-recipe.md`) [M] |

The only knob outside this table that I would change is `ema_decay`: it is not a training
knob, but it changes the number the run reports and breaks comparability with the ladder's 0.0.
Recommendation for this run: 0.0.

---

## 6. Verdict

### VERIFIED (sourced)

1. **The repo has no warmup measurement at the live shape.** Every 95M/b8×s1024 run on
   record has `warmup_tokens=0`; the one warmup run (`lr400`) spent 733 of 800 steps inside
   the ramp and never held a peak. [M] `train_budget.sh`, `recipe_ladder.sh`,
   `~/logs/night_opt/*`, `~/logs/lr400/*`.
2. **3e-4 vs 6e-4 at 49.152 MB is +0.2504 BPB (11-13σ), measured with `warmup_tokens=0` and
   AdamW.** It does not separate "6e-4 is bad" from "no warmup is bad". [M]
   `~/logs/night_opt/lr3e-4_s{1,2}.jsonl`, `lr6e-4_s1.jsonl`.
3. **min_lr 0.0 vs 0.1 at 120 MB is −0.0036 BPB, one seed, inside 1σ≈0.011 and below the
   repo's own spread rule.** The same-seed reproducibility floor at 30-90 MB is RMS 0.0048. [M]
   `~/logs/ladder/{control,cooldown}_refetch.jsonl`.
4. **At 1B/100B, cosine→10% scores 46.26 and cosine→0 scores 45.88 on the aggregate
   benchmark; cosine→0 improves the loss but saturates downstream.** [T] 2405.18392v3 §3.3,
   Fig. 8b.
5. **10% decay is sufficient at 0.036-0.17B; 2.5% is not.** [T] 2404.06395v3 §4.3, Fig. 6.
6. **On a 200k-step run, 10k cooldown steps (5%) with a (1-sqrt) shape nearly match cosine;
   the benefit plateaus at ~20%.** [T] 2405.18392v3 §3.1, Figs. 5-6.
7. **Warmup in the literature is 0.2-1.5% of tokens; K3 uses 1%; no paper reports a
   no-warmup arm at our scale.** [T] 2405.18392v3 §A.1; 2501.00656v3 Table 3;
   2607.24653v2 §3.3; 2608.28308v2 §3.1.
8. **The optimal-LR fits do not agree on the sign of the D dependence**: 2608.28308v2 gets
   D^+0.086, Porian et al. (their Table 2) gets no D term, Li et al. gets D^+0.307; OLMo 2
   finds 3e-4 overtakes 6e-4 only after 200B tokens. No published law justifies lowering the
   peak LR for a 16× longer run at our scale. [T]
9. **K3 reports cosine > WSD under independently tuned hyperparameters and does not publish
   the magnitude; Hägele et al. report a 0.03-0.06 aggregate difference at 1B/100B.** [T]
   2607.24653v2 §3.2; 2405.18392v3 §3.3.
10. **arXiv:2404.14219 is the Phi-3 Technical Report, not a schedule paper.** [T]
11. **Our batch/LR is unmeasured**: `sweep_blr/` does not exist, `bsweep*.log` is a
    throughput driver, and `ab_batch_lr.sh` has never run. [M]

### SPECULATION (my inference)

1. **SPECULATION:** at 95M/1.9 GB the peak-LR optimum is within ±0.05 BPB of 3e-4. Basis:
   the 3e-4 run was stable to 245.76 MB with no decay, 5e-4 was stable to 1.2-4.2 GB on
   102.7M-199M, and 2608.28308v2's D exponent is +0.086. No 95M measurement exists above
   245.76 MB. Confidence: medium.
2. **SPECULATION:** the min-lr 0 vs 0.1 gap at 2 GB will be ≤0.005 BPB. Basis: the 120 MB
   arm (−0.0036), the 1B/100B loss-vs-downstream split, and the 4.18 GB tail's 0.0046 over
   1.18 GB. Confidence: medium.
3. **SPECULATION:** the expected 2 GB BPB is ≈0.81, from 1.4906 at 120 MB and 0.17
   BPB/doubling over 3.99 doublings. The marginal was measured over 0.05-0.25 GB on a
   different mix; this is a 4-doubling extrapolation. Confidence: low.
4. **SPECULATION:** a 30 MB warmup test will show 6e-4+warmup within 0.035 BPB of
   3e-4+warmup. Basis: the 0.25 gap is far outside every published LR-sensitivity curve at
   this scale, and the repo itself attributes it to the missing warmup. Confidence: medium.

### Confidence

- **High** for everything in VERIFIED. The measurements are artifacts with paths; the paper
  numbers are table or section references.
- **The recommendation** (WSD, 3e-4, 1% warmup, 10% decay, min 0.1, b8×a1) is a
  *defensible default*, not a measured optimum at 2 GB. Its expected cost against the best
  schedule is ≤0.01 BPB; its expected protection against the 0.25 BPB no-warmup failure is
  the reason to adopt it.
- **Low** for the batch/accum choice: it is the one knob with no local measurement and a
  literature that contradicts itself.

---

## Sources

**Papers** (all fetched from the arXiv API / PDF on 2026-11-08):
2405.18392v3 · 2404.06395v3 · 2410.05192v3 · 2508.01483v1 · 2607.12360v1 · 2602.06797v3 ·
2601.05034v2 · 2608.28308v2 · 2503.12811v1 · 2309.14322v2 · 1812.06162v1 · 1711.00489v2 ·
2006.09092v6 · 2501.00656v3 · 2607.24653v2 · 2602.15763v2 · 2404.14219v4 (Phi-3, cited in
error).

**Repo artifacts:**
`port/src/dormouse/train/lr.py:24-68` · `port/src/dormouse/config.py:56-77,1793-1795,2265-2280,2545-2615` ·
`port/src/dormouse/train/loop.py:2335` · `port/scripts/train_budget.sh` · `port/scripts/recipe_ladder.sh` ·
`port/scripts/warmup_lr_ab.sh` · `port/scripts/ab_batch_lr.sh` · `port/scripts/batch_sweep.sh` ·
`~/logs/night_opt/{lr3e-4_s1,lr3e-4_s2,lr6e-4_s1}.jsonl` · `~/logs/ladder/*_refetch.jsonl` ·
`~/logs/{flagship2,flagship3,flagship_resume,f3short,f3muon8}.jsonl` ·
`~/logs/m400.jsonl` · `~/logs/medium_long.log` (med1, window not comparable) · `~/logs/lr400/*`.

**Docs:**
`docs/records/reviews/2026-11-08-night.md` §5 · `docs/records/reviews/2026-11-08-data-eval-integrity.md` §6 ·
`docs/records/reviews/2026-11-08-ripple-65b.md` · `docs/records/reviews/2026-11-05-final-dense-recipe.md` ·
`docs/records/research/2026-11-08-efficiency-frontier.md` §5 · `docs/protocols/AB-PROTOCOL.md` ·
`docs/records/research/2026-10-18-data-capacity.md:18,130`.

---

## Open questions

1. **Does the 6e-4-with-warmup arm close the 0.25 BPB gap?** 82 min at 30 MB. This is the
   single highest-value schedule experiment in the queue because it decides whether 3e-4 is a
   real optimum or an artifact of the missing warmup.
2. **Is our 8 192-token batch 50× below B_opt, and does it cost BPB at fixed bytes?** 55 min
   at 30 MB. If yes, the 2 GB run should use accum 4 and every future run re-baselines.
3. **Does decay-to-zero still win at 2 GB, and does the 1B downstream penalty reproduce on a
   byte-level model?** The tail branch answers the first for 1.9 h; the second needs an SFT
   or benchmark eval that the repo does not have at 95M.
4. **Would a (1-sqrt) tail beat our linear tail?** 2508.01483v1 and 2405.18392v3 both say the
   shape matters by ~0.03 aggregate at 1B/100B. Our code has only linear. A 1-line variant +
   the 120 MB fork (28 min of tails) would price it; the expected gain is ≤0.01.
5. **Is `ema_decay=0.999` worth more than comparability?** The eval reads the shadow, so the
   2 GB run's number is not the ladder's number even at matched bytes. One eval of both at
   120 MB settles it.
6. **Corpus caveat, not a schedule one:** at 1.9 GB the 0.55 fwedu weight needs 1.05 GB
   against 933.7 MB on disk (`mix_refetch.toml`; fwedu_train 933 743 230 B [M]), so fwedu
   wraps ~1.12× inside the run. `max_epochs=4.0` allows it; the curve is still a new number.
