# 2026-11-08 — Is the loop-vs-dense verdict fair? What the A/B must be

**Read-only except this file.** No code, config, checkpoint or corpus modified. **CPU
only**, no GPU run. The surviving artifacts were re-read, not re-generated: every
checkpoint the 2026-11-06 report scored (`checkpoints/flagship`, `checkpoints/flagship2`)
and its eval file (`data_local/mix4/web_eval.bin`) were deleted by the 2026-11-08 merge
(`docs/reviews/2026-11-08-merge-incident.md:24-31`), so nothing below is a re-measurement.

**Question.** `docs/reviews/2026-11-06-control-transformer.md` says a plain 13.5M dense
transformer beats the 102.7M KDA/CSA/mHC/Engram loop by **0.126–0.37 BPB**. Since that
measurement: the Muon LR defect was found (2026-11-07), the fast optimizer rung landed,
the mixture changed twice, and the eval window changed. Is the gap a comparison of
*blocks*, or is some of it the optimizer / the mixture / the budget?

---

## 0. Вердикт (RU)

**Замер 2026-11-06 — не сравнение блоков.** Это сравнение двух *рецептов*, и у одного
из них (loop) оптимизатор был сломан на всех матрицах: Muon-половина получала
`lr=5e-4` с шагом `lr·√(max(1,r/c))/√c` на элемент, то есть **0.7 %–17 %** от
первого порядка AdamW-половины (`docs/reviews/2026-11-06-optimizer-audit.md:137-156`).
Контроль — чистый AdamW 6e-4 (`scripts/control_transformer.py:261-263`). Цена этого
дефекта измерена на *другой* модели того же тренера: **0.248 BPB** на 95M dense,
2 сида, разброс ≤0.007 (`~/logs/night_opt/{mix,adam}_s{1,2}.jsonl`, 3000 шагов).
Это **вдвое больше** «измеренной нижней границы» 0.126 и примерно равна
«матч-по-байтам» 0.258. Знак вердикта («плотный лучше») почти наверняка остаётся,
но **величина и приписывание блоку — нет**.

**Три вещи, которые нельзя честно приписать блоку на этих данных.** (1) Ни один
loop-прогон не запускался с **живым** Muon-множителем: два loop-прогона после фикса
(`~/logs/csa8k.log`, 2026-11-08 10:49; `~/logs/csa8k_prop.log`, 13:29) шли с
`muon_lr_mult=1.0` + `muon_lr_mode="uniform"`
(`checkpoints/csa8k/csa8k.config.toml:197-198`), а это, по слову самого поля,
«reproduces the pre-fix step bit-for-bit» (`config.py:2084`). (2) 4.2 GB endpoint — это прогон с
**разошедшимся роутером** (`lfb_bias` −117…+47, `control-transformer.md:256-262`) и с
**перепланированным на резюме микстурным расписанием** (§6, F1). (3) Депт-эффект
(1 уникальный слой ×3 против 12) в обзоре не отделён от блока — ни одного плеча,
где меняется только одно.

**A/B стоит ≈8 карточных часов** (500 MB, 2 плеча × 2 сида + LR-брекет), 15–17 ч на
1 GB. Это одна ночь. **Тратить надо**, потому что решение — «выкинуть loop из
рецепта навсегда» — сейчас опирается на замер сломанного рецепта, а это ровно тот
класс ошибки, который ADR-0020 называет «неверно про то, что измерялось».

**Правило, записанное до прогона:** loop проходит, только если его среднее по 2
сидам **лучше** dense на **≥ 0.03 BPB** И оба сида лучше среднего dense. Всё
остальное — закрытие lane (ничья удаляет механизм, `AGENTS.md` §1.2).

---

## 1. Восемь прогонов (девять строк), как они реально шли

Controls = `scripts/control_transformer.py` (чистый torch, **AdamW β=(0.9,0.95), wd 0.1,
lr 6e-4, cosine→0.1×, warmup 2%, clip 1.0, bf16 autocast, seed 1**; в скрипте нет
Muon — `scripts/control_transformer.py:261-263,272-276`). «Ours» = `port/` trainer,
`opt=mix`.

| # | arm | command / flags (as run) | Muon LR actually applied | bytes | mixture | eval window | seeds | BPB |
|---|---|---|---|---|---:|---|---|---:|---:|
| A1 | `flagship2` @50k | `--preset configs/flagship_base.toml --set steps=171000 --set batch=8 --set grad_accum=3 --set lr=5e-4 --set mix_plan=configs/mix_v7.toml` (banner `~/logs/flagship2_stdout.log:2-3`; run event `~/logs/flagship2.jsonl:1`) | **`muon_lr_mult` did not exist yet** (field landed 2026-11-07, `port/src/dormouse/train/optim.py:1726`); Muon group = **5e-4** → per-element 9.57e-6…2.50e-4 by shape | 1.2288 GB | `mix_v7`, stage 1 only (stage 1 ends at micro 215,460 = 71,820 opt steps) | 65,536 B @ off 0 | **1** | headline **1.7629**, fin 1.6552 |
| A2 | live @66k | same | same | 1.6220 GB | stage 1 | 65,536 | 1 | headline 1.7446 |
| A3 | `flagship.pt` finished | same, **resumed at 120k with `steps` 320000→171000** (`~/logs/flagship.log:2`, `~/logs/flagship_resume.log:2-4`) | same | 4.2025 GB | `mix_v7`, **re-planned at the resume** (§6 F1) | 65,536 | 1 | headline 1.7349 (log 1.7292 @170k), fin 1.6166 |
| B1 | `c50` | `control_transformer.py --bytes 50000000 --tag c50` | **AdamW only** | 49,995,776 | `mix_v7` stage 1 | 65,536 | 1 | 1.9991 |
| B2 | `c200` | `--bytes 200000000` | AdamW | 199,999,488 | stage 1 | 65,536 | 1 | 1.6828 |
| B3 | `c200r` | B2 re-run, same seed | AdamW | 199,999,488 | stage 1 | 65,536 | 1 (same) | 1.6399 |
| B4 | `c500` | `--bytes 500000000` | AdamW | 499,998,720 | stage 1 | 65,536 | 1 | 1.4673 |
| B5 | `c50ga3` | `--bytes 50000000 --grad-accum 3` | AdamW | 49,987,584 | stage 1 | 65,536 (JSON says 196,608 — **wrong**, §6 F2) | 1 | 2.1151 |
| B6 | `c500ga3` | `--bytes 500000000 --grad-accum 3` | AdamW | 499,998,720 | stage 1 | 65,536 (JSON says 196,608) | 1 | **1.4918** |

Artifacts: `/tmp/opencode/control/{c50,c50ga3,c200,c200r,c500,c500ga3}.json` + `.log`
(+ `c500ga3.pt`, `c200.pt`). **`/tmp` is tmpfs** — the `.pt` files and the JSONs die on
reboot; the numbers above are copied out of them and out of the logs.

The 8-window scan (`§6` of the report) is `window_scan.py` / `scan_control.py`, both
`ByteStream(web_eval.bin, 8, 1024, lo=off)` × 8 batches = **65,536 B per window**, offsets
`0 … size-70000` — so the flagship and the control *were* scored on the same geometry.
The scan stdout was not saved; only the report's table survives.

Verified against the logs: `~/logs/flagship2.jsonl` step 50000 → `bpb=1.762936` (the
report's "the checkpoint measured through this path reads 1.7629" is true); step 66000 →
1.744583; step 20000 (=491.5 MB) → 1.864460, which is the interpolation point behind the
report's "ours 1.8622" (`control-transformer.md:135`). `~/logs/flagship_resume.jsonl`
step 170000 → 1.729189 (the report's "log 1.729").

---

## 2. Fairness: five things differ, one of them is larger than the gap

### 2.1 The optimizer — the largest single term, and it is not the block

The loop's Muon half ran at the AdamW LR with a step whose size is a constant of the
*shape* (`optim.py:1122-1134`): `RMS(ΔW) = lr·√(max(1,r/c))/√c`. On the loop's own
shapes at `lr=5e-4` (`2026-11-06-optimizer-audit.md:111-148`):

| parameter | shape | Muon RMS | AdamW RMS | deficit | ⟨G,ΔW⟩ ratio |
|---|---|---:|---:|---:|---:|
| `attn.q_proj` | `[1024,256]` | 6.25e-5 | 4.98e-4 | 8.0× | 0.053 |
| `attn.latent_down` | `[256,1024]` | 1.56e-5 | 4.99e-4 | 32× | 0.009 |
| `mem_dense` | `[1024,1024]` | 1.56e-5 | 4.99e-4 | 32× | 0.013 |
| `moe.down` | `[1024,2730]` | 9.57e-6 | 4.96e-4 | **52×** | **0.007** |
| `moe.experts.0.v` | `[64,2730]` | 9.57e-6 | 4.32e-4 | 45× | — |
| `moe.shared.v` | `[192,2730]` | 9.57e-6 | 4.89e-4 | 51× | — |
| `csa.W_o` / `moe.gate_u` | `[1024,64]` | 2.50e-4 | 4.9e-4 | 2.0× | 0.168 |
| `mhc.phi_pre` | `[2048,2]` | 1.13e-2 | 3.06e-4 | **0.03× (33× hot)** | 22.75 |

The expert bank is **86.5 % of the Muon group** (`2026-11-06-optimizer-audit.md:43-45`)
and ran at **1/45 of the AdamW step's per-element RMS**; the nearest measured first-order
ratio, on `moe.down` (same RMS), is **0.7 %**. That is the component the depth-recurrence
report names as part of "the block" (`2026-11-06-depth-recurrence.md:216`) — so the
report's own suspect was the most starved part of the model.

**Measured cost of the same defect, same trainer, on the dense 95M** (`flagship3_dense`,
`opt=mix` vs `opt=adamw`, `lr=3e-4`, b8×1024×accum 3, 196,608-B window, 2 seeds each):

| steps | bytes | mix, `muon_lr_mult=1` | adamw | Δ | seed spread |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 49.15 MB | 2.1813 | 1.9354 | **0.246** | 0.006 / 0.022 |
| 3,000 | 73.73 MB | 2.0185 | 1.7707 | **0.248** | 0.007 / 0.005 |

Sources: `~/logs/night_opt/mix_s1.jsonl` 2.014967, `mix_s2` 2.022008, `adam_s1`
1.773200, `adam_s2` 1.768139 (3000); the 2000-step rows are the same files. (The muon
audit's `:390` says "one seed per arm, `adam_s2` never ran" — **stale**: both second
seeds exist and completed at 03:42/03:58 on 2026-11-07.)

**And the fix flips the sign**, 2 seeds, same window, 2,000 steps / 49.15 MB
(`~/logs/night_opt3/`): `m12` 1.7513 / 1.7442 → **1.7478**; `adam` 1.9109 / 1.8784 →
1.8947; `m32` 2.0158 / 1.9726 → 1.9942. So at a live multiplier the mixed optimizer
**beats** AdamW by 0.147, and the swing from broken to fixed is 0.43 BPB.

That is a 2-seed, same-bytes, same-window, same-trainer measurement. It is on a
*different block* (dense 12L/832 at 691.37 MFLOP/B, `port/configs/flagship3_dense.toml:25`)
— so it transfers in *kind*, not in exact size — but it is the only term in this whole
comparison that was measured on this equipment, and it is **0.25 BPB**, twice the 0.126
and equal to the 0.258.

### 2.2 The router at the 4.2 GB endpoint is collapsed

`control-transformer.md:256-262`: `lfb_bias` range −117.277…+47.138 and the repo's own
loader prints "this controller DIVERGED … the file's routing is a collapsed one". The
report counts it as a caveat; it is a **second defect on the loop's side, independent of
Muon**, and it is unquantified. The 0.126 and the 0.258 are both measured against it.

### 2.3 The LR schedule is not matched at the matched-bytes point

The control is **fully annealed** at every budget (cosine to 0.1×, `control_transformer.py:272-276`);
the loop at 20,345 steps of 171,000 is still at **peak 5e-4** (decay starts at 102,600,
`2026-11-06-optimizer-audit.md:330-334`). The review's own two measurements of the anneal
are **+0.066 at 50M and −0.019 at 200M** (`control-transformer.md:199-204`) — no
consistent sign, unmeasured at 500M. The 0.258 carries no error bar for it.

### 2.4 The byte budget is 8.4× in the control's disfavour — that one is honest

The 0.126 is control@500M vs loop@4.2GB. That biases **against** the control and is the
report's stated reason for calling 0.126 a lower bound. Fine. But the *matched-bytes*
0.258 (`:142`) is not measured at all: no checkpoint exists at 20,345 steps, so "ours 1.75"
is `log-interpolated headline − 0.108`, and the 0.108 offset was measured at step 50,000
while the 4.2 GB offset is 0.118 (`:151-152`) — it drifts with training, so the 500M
extrapolation is ±0.01 at best before the window and seed terms.

### 2.5 One seed each, and the same-seed rerun moved 0.043

`c200` 1.6828 vs `c200r` 1.6399, identical flags and seed (`/tmp/opencode/control/{c200,c200r}.json`).
The per-window mean is stable to 0.009 across *windows* (`control-transformer.md:219-224`),
which is a different quantity from across *seeds*. There is no second seed for the loop
at all.

### 2.6 What *is* fair

* **Same bytes, exactly**: the control builds the trainer's own `MixtureStream` with the
  flagship's plan and `steps=513000` (`control_transformer.py:55`, `:195-197`), and the
  verified banner `~/logs/flagship2_stdout.log:3` says `mix_v7 … steps=513000 micro-batches
  (171000 x grad_accum=3)`. `source_for_step` is a pure function of `(seed, step, steps)`
  (`port/src/dormouse/train/mixture.py:374-384`), so for the control's 61,035 micro steps
  the draw is identical. **Confirmed** by the realized stage-1 shares (fwedu 0.5866 vs
  0.58 target, `control-transformer.md:106-109`).
* **Same eval geometry**: 65,536 B @ off 0, `rewind()` first, `BPB = CE/ln 2`
  (`control_transformer.py:139-152` vs `port/src/dormouse/train/loop.py:1127-1158`).
* **Equal FLOPs/byte**: 135.36 vs 137.42 MFLOP/B, −1.5 % — the only valid match here
  (see §4.2 on why equal-params is not).

### 2.7 The two loop runs that exist *after* the fix still ran the broken default

`~/logs/csa8k.log` (2026-11-08 10:49) and `~/logs/csa8k_prop.log` (13:29) are real loop
runs: `medium` 102,678,928 params, `block_form="kda"`, `max_iter=4`, `loop_blocks=1`,
256 experts, seq 8192, lr 1e-3, `mix_refetch`, 8,000 steps × 32,768 B = **262.1 MB**
(`checkpoints/csa8k/csa8k.config.toml:2-19,167-170,186`). Their eval window is
**32,768 B** (4 × 1 × 8192, `~/logs/csa8k.log:107`), not 196,608 and not 65,536.

| run | moe_lb_mode | bpb_fin @262 MB |
|---|---|---:|
| `csa8k` | `sign` | 1.900 |
| `csa8k_prop` | `prop` | 1.881 |

Both ran `muon_lr_mult=1.0` + `muon_lr_mode="uniform"` — the **bit-identical pre-fix
step** — and `ns_dtype=fp32`. So the post-fix loop runs are not evidence about the
fixed loop either; and there is **no dense arm at this context/window/corpus**, so no
comparison of any kind exists on the current lane. A 1.900 at 2.6 bytes/param is a
capacity screen, not a quality number.

### 2.8 The 0.37 end of the quoted range is the frame the report itself retracts

The report's own largest measured Δ is **−0.3949** at 500M bytes — but that is against the
**headline** readout (`control-transformer.md:135`), and §7.2 of the same report says the
headline is `(Σ CE_i + CE_final)/(T+1)`, **depth-confounded**, and charges the architecture
for three intermediate readouts it never serves (`:237-242`). The 0.37 in
`docs/research/2026-11-06-architecture-at-100m.md:4` ("0.13–0.37 BPB") therefore mixes the
fin frame (0.126, 0.258) with a frame its own source rejects. Use 0.126 (measured, 8/8
windows, different byte budgets) and 0.258 (estimated, matched bytes) — **0.37 has no
defensible reading** as a block cost.

---

## 3. Block vs depth: the review **cannot** separate them

What differs between the two arms of the 0.126:

| axis | loop | 13.5M control | measured separately? |
|---|---|---|---|
| block | KDA + CSA + mHC + AttnRes + Engram + 256-expert MoE | plain attention + GELU FFN | **no** |
| unique layers | **1**, run 3× | **12** | **no** |
| width | d=1024 | d=304 | **no** |
| active params/token | 17.26M MAC | 13.48M (all active) | equal-FLOPs only |
| optimizer | broken Muon on 51.7M matrix params | AdamW on everything | **yes, on a third model** (§2.1) |
| params | 102.7M total (50.3M is the Engram table) | 13.5M | — |

No arm holds the block fixed and varies depth, and no arm holds depth fixed and varies
the block. The report's own numbers therefore contain **zero** information that separates
the two; the split in `2026-11-06-depth-recurrence.md:216` ("the block … plus a 1-layer
effective depth") is an inference from the literature, not a result of this comparison.

What the literature term can be, and what it cannot:

* **Tying alone** (unique params held): 0.015–0.045 BPB by the report's own transfer
  (`depth-recurrence.md:15`). Small.
* **Depth alone**: the φ=0.46 law gives `3^0.46 = 1.66` layer-equivalents against 12
  (`depth-recurrence.md:12-13`, marked **[S]** by its own author at `:32`, `:267` because
  the paper's smallest arm is 2 unique layers and our block is 1). Applying the paper's
  *relative* gap (1–2.5 % at r=2) to a ~1.4 BPB model is 0.015–0.035 BPB; applying the
  *absolute* nats reading is 0.04–0.09. Either way, **0.02–0.09**, not 0.13.
* **The block**: the remainder — and if the optimizer term (0.25) and the router term are
  subtracted first, the remainder is **not distinguishable from zero** on these arms.

So: **how much is the block, how much the depth — unidentified.** What the 2026-11-06
numbers *do* establish is narrower and still useful: a 13.5M dense model trained by AdamW
on 8.4× less data beats this loop on 8/8 windows. That is a statement about the recipe as
run. It is not a statement about KDA, Engram, the MoE, or recurrence.

---

## 4. The fair A/B

### 4.1 Design

Same trainer (`port/`), same data, same eval, same optimizer policy, same schedule,
**only the block differs** — the shape `port/configs/mirror_control.toml:1-9` was written
for. Both arms:

* `mirror_control` (dense, 12×304, **13,478,448 params** [M], meta-device count) vs
  `flagship_base` (loop, 102.7M total, 1 unique × 3);
* `steps=20345` = **499,998,720 B** (500 MB), b8 × s1024 × **accum 3** (the flagship's own cadence);
* `mix_refetch` + `fwedu_train.bin` / `fwedu_eval.bin`, `eval_bytes=196608` @ off 0
  (`port/scripts/train_budget.sh:36-38,53`) — **a new corpus, so these are new numbers**;
* the fast rung: `ns_dtype=bf16`, `muon_state_dtype=bf16`, `ns_steps=5`
  (`the-16x.md:30-33`), `doc_masking=false`;
* the Muon fix: **`muon_lr_mode=shape_fair`**, `muon_lr_mult=1.0`, `lr=3e-4` for the
  AdamW half. `shape_fair` is the only mode where one `lr` means one thing on every shape
  (`optim.py:1122-1134`), which is exactly what a two-block comparison needs; `uniform`
  punishes `[64,2730]` 3.5× harder than a dense `[3072,832]` (`tests/test_muon_shape_fair.py:1-12`). *Caveat:* the
  end-to-end winner on record is `uniform+mult=12` (`night_opt3`), not `shape_fair` — so
  bracket it (§4.3).
* `seed ∈ {1,2}`, same schedule, same warmup (`warmup_tokens=19e6`).

### 4.2 Equal FLOPs, and why equal-params is meaningless here

Equal-params at equal-FLOPs is arithmetically impossible: the loop is 102.7M total /
17.26M active MAC/token at **137.42 MFLOP/B**; a 102.7M **dense** model is `6N` = 616
MFLOP/B before attention — the report's own `:58` — i.e. 4.5× the loop's bill, and 906
MFLOP/B with the T-dependent term at d=1024/12L. You can match one or the other, never
both. Parameters are **not** the binding resource here (16 GB card, 100M-class models);
card-hours and bytes are. So the match is **equal FLOPs/byte**, which is what the 13.5M
control was sized for (135.36 vs 137.42, −1.5 %). Equal-params would hand the loop a
4.5–6× compute advantage and answer a deployment question, not this one.

### 4.3 The commands (copy-paste; one arm = one line)

```bash
cd /home/sehaxe/dormouse/port
COMMON=( --set steps=20345 --set batch=8 --set seq_len=1024 --set grad_accum=3
         --set lr=3e-4 --set opt=mix --set muon_lr_mode=shape_fair --set muon_lr_mult=1.0
         --set ns_steps=5 --set ns_dtype=bf16 --set muon_state_dtype=bf16
         --set warmup_tokens=19000000
         --set mix_plan=configs/mix_refetch.toml --set doc_masking=false
         --set data=../data_local/best/fwedu_train.bin
         --set eval_data=../data_local/best/fwedu_eval.bin
         --set eval=true --set eval_every=2000 --set eval_bytes=196608
         --set ckpt_every=20345 )
# D — dense control (the control's block, in this trainer)
PYTHONPATH=src .venv/bin/dormouse train configs/mirror_control.toml "${COMMON[@]}" \
  --set seed=1 --set ckpt_name=dense_s1 --set out_dir=/home/sehaxe/dormouse/checkpoints/loopab \
  --json-log /home/sehaxe/logs/loopab/dense_s1.jsonl
# L — the loop
PYTHONPATH=src .venv/bin/dormouse train configs/flagship_base.toml "${COMMON[@]}" \
  --set seed=1 --set ckpt_name=loop_s1  --set out_dir=/home/sehaxe/dormouse/checkpoints/loopab \
  --json-log /home/sehaxe/logs/loopab/loop_s1.jsonl
```

**LR bracket first** (before spending 7 h), 2,000 steps per arm:
`muon_lr_mult ∈ {0.42, 0.84}` — 0.42·3e-4 = 1.26e-4 is the same *per-element* step the
validated `uniform+12` gives on the modal `[832,832]` shape (`config.py:2076-2084`).
Pick the better multiplier per arm; run the 500M A/B at that arm's own winner. A
"best recipe per block" comparison is what the owner will ship, and it kills the
"you under-LR'd the loop" objection — the exact objection that invalidates 2026-11-06.

### 4.4 Pre-flight gates (all cheap; VOID if any fails)

| gate | command / check | why |
|---|---|---|
| FLOPs matched ≤10 % | `port/scripts/bench_compile.py --preset <arm>` (`--flops-per-byte` auto) on each arm | the 135.36/137.42 pair was measured on the *torch* control; the port arm must be re-measured |
| the fix is live | one 200-step run per arm prints a step RMS ≠ the old one; `tests/test_muon_shape_fair.py` green | `muon_lr_mult=1.0` + `uniform` **is** the broken arm (`config.py:2084`) |
| params / depth | `dormouse info` on each arm: dense 13,478,448 / 12 layers; loop 102,678,943 / 1×3 | a preset typo silently runs the wrong arm |
| window | the eval line prints `over 196608 B` | a batch change moves the window (`AGENTS.md` §2.6) |
| no NaN skips | `nan_skipped=0` on every log row | a skipped step is a different run |

### 4.5 Pre-registered pass rule (write it before the first step)

```
Define  D = BPB(dense) - BPB(loop)   [positive D = the loop is better].

H1: at equal FLOPs/byte (<= 10% drift), equal bytes (499,998,720), the same
optimizer policy, the same data stream and the same 196,608-B window, D >= 0.03
BPB, 2 seeds.

VOID if: FLOPs drift > 10%; eval window != 196,608 B; any nan_skipped>0; a
params/depth row that is not the arm's; < 2 seeds; the two arms' realized byte
counts differ.

PASS (loop lane survives): D_mean >= +0.03 AND D_seed1 > 0 AND D_seed2 > 0.
FAIL (lane closed at this scale): everything else. A tie (|D| < 0.03) deletes the
   mechanism (AGENTS.md 1.2).

Secondary, reported not gating: wall clock per byte (the loop is launch-bound:
8.2 TFLOP/s vs the dense 95M's 38.8 TFLOP/s at 691 MFLOP/B), and the block/depth
split is NOT identified by this A/B -- it holds depth fixed at each arm's own
value.
```

**Resolution.** Two seeds buy ~0.03–0.04 BPB at this size: the paired 2-seed spreads
on record are 0.005–0.043 (`night_opt`, `night_opt3`) and the control's same-seed rerun
moved 0.043. 0.03 is the smallest defensible pass bar; if the owner needs 0.02, pay for
3 seeds (+3.5 h) — the repo's own rule is 3 anyway (`AGENTS.md` §1.2).

### 4.6 Cost, in card-hours

Measured rates, same trainer: loop **59,728 B/s** (median warm, `~/logs/flagship2.jsonl`,
Nov 4-6) → 0.411 s/step at 24,576 B; dense 95M **56,181 B/s** at 691 MFLOP/B
(`docs/reviews/2026-11-08-mfu-first-look.md:74-77`); the 13.5M dense at 1/5.1 of those
FLOPs is ~130–250 kB/s (the torch twin measured 131 kB/s, `/tmp/opencode/control/c500ga3.log:12`).

| item | runs | h |
|---|---|---:|
| LR bracket, 2k steps, 2 multipliers × 2 arms | 4 | 0.9 |
| pre-flight (FLOPs, 200-step smoke) | 2 | 0.4 |
| 500M A/B: loop 2.33 h/seed × 2 | 2 | 4.7 |
| 500M A/B: dense 0.7–1.1 h/seed × 2 | 2 | 1.4–2.1 |
| **500M total** | **10** | **7.4–8.1 → say 8** |

1 GB instead of 500M: **15–17 h**. Adding the two depth arms that *would* separate block
from depth (`loop_blocks=1,max_iter=12` and `loop_blocks=4,max_iter=3`, both at 12
layer-passes) is +2 arms × 2 seeds ≈ +9–12 h; only worth it if the loop passes.

Note the fast rung does **not** give the 1.89× here: that was measured at
`b8×s1024×accum=1` / 8,192 B per step (`the-16x.md:30-33`), where the optimizer is 43 % of
a 283-ms step. At accum 3 the optimizer is a smaller share — the surviving runs go
47.0 → 51.2 kB/s (`~/logs/f3short.jsonl` vs `~/logs/flagship3.jsonl`, ~1.09×, and even
that also changes the mixture). Cost the A/B at the measured 59.7 kB/s, not at 2×.

---

## 5. What cannot be recovered

| lost | why it matters | replacement |
|---|---|---|
| `checkpoints/flagship/flagship.pt`, `checkpoints/flagship2/flagship2.pt` | every `fin`, the 8-window scan, the router state, a fixed-Muon resume | none; re-train |
| `data_local/mix4/web_eval.bin` | the 65,536-B window is gone; `mix_refetch` + `fwedu_eval.bin` is a **new** number | `port/configs/mix_refetch.toml:3-7` says so itself |
| `data_local/{best,mix4}/*` (mix_v7 sources) | the exact byte stream cannot be re-created (refetch caps are 1.2 GB vs 3 GB originals) | `refetch_pretrain.py` slices |
| `flagship*.config.toml` snapshots | the exact `--set` list (ns/muon/opt_graph/eval_batches) of the 4.2 GB run | closest surviving: `port/checkpoints/flagship/flagship1.config.toml` (mix_v4, not the run) |
| the `window_scan.py` stdout | the "correlate 0.97 / 1σ 0.009" pair is unrecomputable | none |
| the 4.62-nat CPU replay contradiction | §6 F3 | never resolved; now unresolvable |

---

## 6. Findings in code (unfixed lines; each is a defect independent of the verdict)

**F1 — `steps` is a resume-legal progress key AND an input to the data schedule.**
`PROGRESS_KEYS` contains `"steps"` (`port/src/dormouse/config.py:61,78`) and its docstring
says progress keys "decide WHEN we look, never what the model computes" (`config.py:45-46`).
But the mixture's stage boundaries are `round(until_frac × steps)`
(`port/src/dormouse/train/mixture.py:341-356`), and `source_for_step` reads them
(`:374-377`). So `--set steps=` on a resume **re-plans the whole remaining data stream**,
silently. The run under review did exactly this: `~/logs/flagship.log:2-3` says
`steps=320000 … 960000 micro-batches`, `~/logs/flagship_resume.log:2-3` says
`steps=171000 … 513000 micro-batches` at `resume: step=120000` (`:4`). Stage 1's end
moved from micro 403,200 to 215,460 — the resumed segment jumped into stage 2 at micro
360,000 where the original plan had 43,200 more stage-1 micro steps. The control is not
affected (its 61,035 micro steps are stage 1 under both plans, and `_unit(seed, step)` is
independent of `steps`), so the report's "literal prefix" claim holds for the arms it
compares — but the 4.2 GB endpoint's *last third* is not the same curriculum as its first
two-thirds, and nothing in the log says so. Fix: make `steps` non-exempt when a
`mix_plan` is set, or freeze the stage bounds into the checkpoint snapshot.

**F2 — the control JSONs misreport the scored window on the `grad_accum=3` arms.**
`scripts/control_transformer.py:336` writes `window_bytes = eval_batches × bytes_per_step`
(= 196,608 for ga3), but `eval_bpb` scores `batches × batch × seq_len` = **65,536**
(`:139-152`) — `grad_accum` never enters the eval stream. The report text is right
(`:111-115`), the artifact is wrong; anyone re-deriving from `c500ga3.json` will conclude
the control was scored on 196,608 B and "fix" a comparison that is already fair. One-line
fix in the dict.

**F3 — the step-50k checkpoint has two contradictory replay readings, both live.**
`control-transformer.md:113-115` says the checkpoint measures **1.7629** through the
hosted eval (CUDA) and matches the JSONL. `2026-11-06-optimizer-audit.md:362-373` says the
same file measures **fin 4.62 nats** through the official loader on **CPU**. Both
documents are current; the checkpoints are deleted; the disagreement can never be settled.
The CUDA reading is corroborated by `~/logs/flagship2.jsonl` (step 50000, 1.762936), so
the CPU path is the suspect — and if it is wrong, every *absolute* number in that audit's
§8 is contaminated (its §2.2/2.3 ratios are not: they are ratios on one gradient and NS is
scale-invariant).

**F4 — the preset alone is the broken arm, and nothing in the preset says so.**
`port/configs/flagship3_dense.toml:11-19` says the file "is the MODEL, not the run": a
bare `--preset` resolves to `muon_lr_mult=1.0`, `ns_dtype=fp32`, `grad_accum=1`. Train
keys cannot live in the preset, so the recipe exists only in shell
(`port/scripts/train_budget.sh:47-58`). Any future A/B that forgets the six `--set`s
silently reproduces the 2026-11-06 defect — and this already happened: `csa8k` ran on
2026-11-08 with `muon_lr_mult=1.0` in its snapshot
(`checkpoints/csa8k/csa8k.config.toml:197-199`). The A/B must be **committed as a
script**, not described in a doc.

**F5 — `MIN_EVAL_BYTES = 65_536` is exactly the window the verdict was taken on, and the
constant's own docstring calls that window ±0.19 BPB at 95 %** (`config.py:5059-5062`).
The paired-difference design survives that (the 1σ of the difference across windows is
0.009, `control-transformer.md:219-224`), but the absolute numbers do not, and the default
is now 196,608 (`config.py:1952`). Any extension of the 2026-11-06 table must not mix the
two.

---

## Appendix — what was actually re-read

```sh
# the arms
cat /tmp/opencode/control/{c50,c50ga3,c200,c200r,c500,c500ga3}.{json,log}
cat scripts/control_transformer.py            # eval_bpb, adamw, the window dict
cat /tmp/opencode/control/{eval_ours,window_scan,scan_control}.py

# the live side
head -4 ~/logs/flagship2_stdout.log; head -4 ~/logs/flagship_resume.log; head -4 ~/logs/flagship.log
python3 -c 'import json; [print(d["step"], d["bpb"]) for d in map(json.loads, open(".../flagship2.jsonl")) if d.get("event")=="eval"]'

# the optimizer terms
cat ~/logs/night_opt/{mix,adam}_s{1,2}.jsonl          # 3000 steps, 196,608
cat ~/logs/night_opt3/{m12,m32,adam}_s{1,2}.jsonl     # 2000 steps

# code claims
port/src/dormouse/train/optim.py:1122-1134,1726-1742,1908-1942
port/src/dormouse/config.py:45-78,1952,2059-2084,5059-5062
port/src/dormouse/train/mixture.py:341-384
port/src/dormouse/train/loop.py:1127-1158,2125-2145

# params [M], meta device, CPU
PYTHONPATH=src .venv/bin/python -c '... param_count ...'   # 13,478,448 / 102,678,943 / 95,035,200
```

Owed and **not** done here (one-file rule): the `docs-site/tools/manifest.mjs` line for
this document. `python3 tools/check_doc_refs.py` was run after the file landed: 1,314 files,
3,608 references, the same single pre-existing dead reference as before (no regression).

Status: complete
Integrity: clean
Contract: aligned
