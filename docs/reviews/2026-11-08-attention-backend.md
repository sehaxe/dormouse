# 2026-11-08 — attention backend at s1024 on sm_120: the default is already flash

Closes §5/§7 item 3 of [`docs/reviews/2026-11-08-kernel-audit.md`](2026-11-08-kernel-audit.md)
("бэкенд SDPA — единственный оставшийся неизвестный на 0–12%"). Measured on the
card, at the live shape, with the trainer itself. **Nothing was committed; no
model file was edited** — the backend is forced from a driver
(`torch.nn.attention.sdpa_kernel`), which is the whole patch (§4).

Marks as in the audit: **[M]** measured, artifact named · **[C]** computed from a
named code path or from measured inputs · **[S]** inference, with the check that
settles it. A number without an artifact is a defect.

---

## 0. Verdict (RU)

**Дефолт уже flash. Менять нечего, и «оставить как есть» — это ответ, а не
отсутствие ответа.**

1. **`F.scaled_dot_product_attention` на b8×s1024 (h13×64, bf16, `is_causal`)
   уходит в `aten::_scaled_dot_product_flash_attention`** — 12 вызовов на
   forward, `pytorch_flash::flash_fwd_kernel` + `flash_bwd_dq_dk_dv_…`
   в профиле, и `can_use_flash_attention(SDPAParams) == True` на тех самых
   тензорах, которые отдаёт блок. Три инструмента, один ответ (§2).
2. **Принудительный EFFICIENT — +6.2% времени шага** (157.1 против 147.9 мс
   [M]), **принудительный MATH — ×2.07** (306.2 мс [M]). Ядро attention
   стоит 9.2 мс/микробатч, и дельта шага (9.19 мс) совпадает с дельтой
   device-времени SDPA (9.75 мс) — это не шум, это тот же самый механизм.
3. **`sdpa_mem_efficient=true` на этой форме — SILENT no-op** (как
   `residual_dtype` в C7): 12 `TransformerBlock`, 0 `FullAttention`, параметры
   бит-в-бит те же. Флаг читает только `FullAttention` (`fullattn.py:101`).
   А там, где он подключён, на этой карте он **в 1.9 раза медленнее**: его
   доккомментарий («EFFICIENT 0.91 мс против flash 1.91 мс», 5060 Ti,
   2026-10-27) не переносится на 5070 Ti — измерено 9.17 → 18.92 мс на
   12 блоков, а исполняется sm80-ядро `fmha_cutlass*_sm80`, не sm_120 (§4).
4. **`attn_window` на s1024 не меняет время шага вообще: 0 мс.** Измерено
   W=512 и W=1024 — 146.8/146.7 мс против 147.9 у дефолта, то есть в пределах
   ±1 мс. Причина не в том, что окно дешёвое: **этот блок его не читает** —
   поле принимается валидацией и никуда не идёт (`transformer.py` не содержит
   `attn_window`; `_model` строит 12 `TransformerBlock` и ноль `FullAttention`,
   число параметров идентично). Рычага контекста у плотной формы нет — ни
   включённого, ни выключенного (§5).
5. **Единственное, что здесь стоит денег, — не бэкенд, а то, что окно
   контекста вообще не подключено.** Если локальная полоса когда-нибудь
   понадобится, это новый код в `transformer.py` (полоса маской или чанками),
   а не флаг; и на s1024 маска сделает attention *дороже*, а не дешевле,
   потому что не-каузальная маска не является кандидатом для flash.

---

## 1. What ran (protocol)

| | |
|---|---|
| shape | `batch=8`, `seq_len=1024`, `grad_accum=1` → 8 192 B/step |
| recipe | `port/configs/flagship3_dense.toml` (12 layers, d=832, 13 heads × 64, dense GELU FFN 3072), `block_form="transformer"` |
| data | `--set mix_plan=configs/mix_refetch.toml`, `doc_masking=false`, `eval=false` |
| optimizer | `opt=mix`, `muon_lr_mult=12`, `lr=3e-4`, **FAST rung**: `ns_dtype=bf16`, `muon_state_dtype=bf16`, `ns_steps=5` |
| steps | 150 timed per arm; the first 25 steps (5 log windows) are dropped as warmup |
| instrument | the trainer's own JSONL: `tokens_per_s` = `log_every·batch·seq_len / window_seconds` (`train/loop.py:2369`), so `ms/step = 8192 / rate · 1000`. `log_every=5` ⇒ 25 surviving windows per arm |
| model | 95 035 200 params, bf16, identical init per arm (`train.seed=1`) |
| card | RTX 5070 Ti, sm_120, torch 2.13.0+cu129, 16 303 MiB |
| commit | `354520f`, `port/` worktree dirty (`config.py` only; `transformer.py` clean, so its line numbers are `HEAD`'s) |
| exclusion | every arm ran under `port/scripts/gpu_lock.sh` (one GPU process at a time); a lost lock (`exit 75`) or timeout (`124`) was retried, never quoted |

Two complete passes of all seven arms were taken, each pass a fresh process per
arm. **Pass 1 was the first sweep and its `default` arm is invalid** — the card
had just been released by another lane and the arm swings 150 → 338 ms between
windows. Pass 2 ran all seven back-to-back on a warm card. The report uses the
**floor** (that arm's best 5-step window) as the estimator and the median as the
"what a run pays" figure; §6 says why the floor, not the median, is the one that
reproduces.

Artifacts: `/tmp/opencode/sdpa/probe.json` (backend identity, profiler),
`/tmp/opencode/sdpa/` and `/tmp/opencode/sdpa/pass1/` (both passes, per arm
`.jsonl` + `.log` + `.meta.json`). The scripts are
`/tmp/opencode/sdpa_{probe,arm,sweep,table}.py|sh`, not repo files.

---

## 2. Which backend the default actually picks

### 2.1 The aten op (the dispatch choice)

Profile of one fwd+bwd of the whole model, 12 blocks, at the live shape. The
aten op is the choice; the kernel names are the proof it ran.

| mode | SDPA aten op | calls | SDPA device ms (fwd+bwd, 12 blocks) | kernels seen |
|---|---|---:|---:|---|
| **default** | `aten::_scaled_dot_product_flash_attention` (+ `_backward`) | 12+12 | **9.20** (fwd 2.456, bwd 6.743) | `pytorch_flash::flash_fwd_kernel<…64,128,128,4…>`, `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`, `flash_bwd_dot_do_o_kernel`, `flash_bwd_convert_dq_kernel` |
| flash forced | same | 12+12 | 9.17 (fwd 2.403, bwd 6.745) | same four |
| efficient forced | `aten::_scaled_dot_product_efficient_attention` (+ `_backward`) | 12+12 | **18.92** (fwd 4.508, bwd 14.410) | `fmha_cutlassF_bf16_aligned_64x64_rf_sm80`, `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80` (`AttentionBackwardKernel<cutlass::arch::Sm80, …>`) |
| math forced | `aten::_scaled_dot_product_attention_math` | 12 | ≥ 93.0 | `aten::bmm` ×72 = 57.45, `softmax_warp_forward` 14.15, `softmax_warp_backward` 21.39 |

`[M]` `probe.json`. Notes: the default and the forced-flash arms are the same
program (same 12 aten calls, same four kernels, device times within 0.03 ms).
Math's device number is a floor — it under-counts the fp32 score tensor
`[8,13,1024,1024]` (435 MB per block) that the op materialises, because the
casts/copies live outside the aten op; the step-time delta in §3 is the better
number.

**The math bmm count is a free cross-check on the audit's census**: the audit
counted 72 `bmm` for the math-backend SDPA in its meta-tensor dispatch census
(§2 of the audit) and the live math arm here runs exactly 72 (12 blocks × 6 =
2 fwd + 4 bwd). That is the confirmation that the census was measuring the math
path, as the audit said it was.

### 2.2 `can_use_flash_attention`, on the block's own tensors

The probe patches `F.scaled_dot_product_attention` for one forward, so the
parameters below are the *captured* ones, not a synthetic reconstruction:

```
shape   [8, 13, 1024, 64]   q stride [851968, 65536, 64, 1]   dtype bfloat16
kwargs  ['is_causal', 'scale']   is_causal=True   scale=0.125   attn_mask=None
is_flash_attention_available()      True
can_use_flash_attention(params)     True
can_use_efficient_attention(params) True
torch.backends.cuda: flash=True mem_efficient=True math=True cudnn=True
arch list: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
```

`[M]` `probe.json:feasibility`. So flash is not merely "available in the wheel"
(the audit's C5 had that from `get_arch_list`): it is **eligible for this exact
call and is what the dispatcher selects**.

---

## 3. What each backend costs per step

150 steps, b8×s1024, FAST rung, mix_refetch, eval off. `ms/step` is a 5-step
window mean from the trainer's JSONL; floor = that arm's minimum window.

| arm | floor ms/step (pass 2) | floor (pass 1) | × vs default | median ms/step (pass 2) | p95 (pass 2) | attention device ms (from §2.1) |
|---|---:|---:|---:|---:|---:|---:|
| **default (flash)** | **147.95** | 150.78 | **1.000** | 148.79 | 153.84 | 9.20 |
| flash forced | 147.89 | 148.92 | 0.999 | 167.41 | 189.50 | 9.17 |
| efficient forced | **157.14** | 157.66 | **1.062** | 168.27 | 186.27 | 18.92 |
| math forced | **306.18** | 304.83 | **2.070** | 307.50 | 308.26 | ≥ 93.0 |
| `attn_window=512` | 148.06 | 146.80 | 1.001 | 149.41 | 150.89 | — |
| `attn_window=1024` | 147.68 | 146.70 | 0.998 | 148.76 | 151.03 | — |
| `sdpa_mem_efficient=true` | 147.03 | 146.58 | 0.994 | 148.48 | 170.04 | — |

`[M]` `/tmp/opencode/sdpa/pass1/`, `/tmp/opencode/sdpa/`. Reading it:

- **default = flash**, and forcing flash changes nothing (floor 147.89 vs 147.95
  — 0.06 ms apart, i.e. the same program; the median column for the forced arm
  is inflated by host excursions, see §6).
- **efficient is +9.19 ms/step (+6.2%)** (157.14 − 147.95), and the *device*
  delta of the attention core is +9.75 ms (18.92 − 9.17) — the step pays
  exactly what the kernel costs. This is the only arm where the two independent
  instruments agree to 0.6 ms, which is what makes the row believable.
- **math is +158 ms/step (×2.07)**. Fully outside noise: its own median is
  307.5 with sd 0.68, and its floor reproduces to 1.4 ms across passes.
- **the two `attn_window` arms and `sdpa_mem_efficient` are flat**: 146.6–148.1
  floors, i.e. a ±1.2 ms band around the default with no sign of a cost, which
  is what §5's code proof predicts.

Attention is therefore **6.2% of the step** (9.2 / 147.9). The audit's C5 priced
a real math path at **18–25 ms (12–17%)** from a bandwidth argument (35.3 GB of
`bmm` + 26 GB of softmax at 1.9 TB/s); measured, it is **+158 ms**, because the
math path is not bandwidth-bound — it is fp32 CUDA-core compute-bound
(72 `aten::bmm` on `cutlass_80_simt_sgemm` plus two full softmax passes over a
materialised `[8,13,1024,1024]` score tensor). **The 0–12% unknown is 0%: the
default is already the fastest backend, and the only alternative that could have
been faster is 6.2% slower.**

---

## 4. The exact patch

No model edit; the backend is a scoped override around the trainer's own
`train()`:

```python
from contextlib import nullcontext
from torch.nn.attention import SDPBackend, sdpa_kernel
from dormouse.train.loop import train as _train

FORCED = {"flash": SDPBackend.FLASH_ATTENTION,
          "efficient": SDPBackend.EFFICIENT_ATTENTION,
          "math": SDPBackend.MATH}

ctx = sdpa_kernel([FORCED[arm]]) if arm in FORCED else nullcontext()
with ctx:                      # one scope around the whole run; per-call also works
    _train(model, run, device=dev)
```

- A forced mode that the card cannot run **raises** (`RuntimeError: No available
  kernel`), so a successful arm is itself evidence the backend ran — the arm
  cannot silently fall through to another kernel.
- `model/transformer.py:267` (`y = F.scaled_dot_product_attention(q, k, v,
  is_causal=True, scale=self.scale)`, identical at `HEAD 354520f`) is untouched.
- Process-global equivalents exist (`torch.backends.cuda.enable_flash_sdp(False)`
  etc. — the probe read them, all `True`), but the context manager is the
  scoped one and it is what the audit's C5 named.
- **There is no counter for the backend on this path.** `TransformerBlock`
  counts nothing at the attention call; `FullAttention` counts
  `sdpa_default_backend`/`sdpa_mem_efficient` (`fullattn.py:177-178`), but that
  module is not built by `block_form="transformer"`. So a run's log cannot say
  which kernel ran — the profiler is the only instrument, which is why this took
  a GPU run rather than one line in a log. Adding `counters.count("sdpa_default_backend")`
  next to `transformer.py:267` is a two-line fix in the audit's ADR-0019
  vocabulary (**COUNTED**), and it is the cheapest way to make this question
  answerable without a profiler in future.

### 4.1 `sdpa_mem_efficient` does not transfer to this card

`fullattn.py:100-101` says "measured 2.1x on sm_120" and its docstring records
flash 1.91 ms / EFFICIENT 0.91 ms / math 19.05 ms at b8 h16 t1024 hd64 bf16
causal on a **5060 Ti**. On this box at h13 (h16 would be 1.23× more work, so
the expected flash fwd is 1.91 × 1.23 ≈ 2.35 ms):

| | 5060 Ti, 2026-10-27 (h16) | 5070 Ti, this run (h13) |
|---|---:|---:|
| flash fwd | 1.91 ms | 2.40 ms (≈ 1.23× — consistent) |
| efficient fwd | 0.91 ms | **4.51 ms** (5.0× worse, not 2.1× better) |
| fwd+bwd, 12 blocks | — | flash 9.17, efficient 18.92 |

`[C]` for the 5060 Ti column (the docstring's own numbers), `[M]` for this one.
Mechanism, visible in the kernel name rather than inferred: this build's
memory-efficient attention runs `fmha_cutlass*_sm80` /
`AttentionBackwardKernel<cutlass::arch::Sm80, …>` — sm_80 code on an sm_120
card — while flash gets `pytorch_flash` kernels built for the arch.
**[S]** that the sm80 instantiation is the reason for the 2× loss; the check is
a torch build with sm_120 fmha instantiations, which does not exist in this
wheel. What is certain is the outcome on this box: the flag's premise is false
here, so leaving it `False` is right, and its docstring should name the device
(that is a documentation fix in `fullattn.py`, not a code change).

---

## 5. `attn_window` at W=512/1024: accepted, then ignored

**Measured: no change in step time** (146.8 / 146.7 ms floors vs 147.9 for the
default — a ±1.2 ms band, and slightly *below* the default, i.e. no cost at all,
not even the cost of building a mask). `[M]` §3.

**Why — the flag never reaches this block:**

| check | result |
|---|---|
| `grep attn_window src/dormouse/model/transformer.py` | no match |
| `load_run(..., {"attn_window": 512})` | accepted, `model.attn_window == 512` (`config.py:3370-3408` only refuses negatives, `attn_kind='full'`, `segment_len`, `use_byteflow`, query-chunk misuse) |
| instantiated model, `attn_window=512` | 12 `TransformerBlock`, **0** `FullAttention`, 95 035 200 params — identical to `attn_window=0` |
| who reads it | `LoopBlock._window` (`loopblock.py:421-440`), `decode.recurrent_refusal` (`decode.py:269`). `block_form="transformer"` builds neither |

So this is the **C7 defect in a second field**: `attn_window` (and
`attn_window_query_chunk`) are accepted on `block_form="transformer"` and read by
nobody. A run can print `attn_window=512` and train full causal attention. That
is **SILENT**, which ADR-0019 classes as a defect, not a degradation — the config
lies to the reader, and the only instrument that would notice is a step time that
does not move (§3) or a profiler.

**And the band is not free even where it IS wired**, which is the part the flag's
own docstring does not say out loud: `fullattn.py:190-200` builds the band as an
**explicit boolean mask** (this torch has no `window_size=`), and a non-causal
`attn_mask` is not a flash candidate — the masked call goes to the
memory-efficient kernel, which on this card is the 18.92 ms one. At s1024 a
W=512 band would therefore *lose* the 9.2 ms flash call and buy at most half of
the 18.9 ms kernel's work back: **a window at this context length is a
regression, not a lever**. **[S]**, and the check that settles it is three lines
(`can_use_flash_attention(params_with_mask)`, then time the banded call) — the
probe that would have run it is
`/tmp/opencode/sdpa_window_probe.py`, written but **not run**: the card was held
by other lanes for the whole tail of this session and the probe is not worth a
sixth queue position. Until it runs, treat "the band costs ≥ the causal flash
call" as inference, not measurement.

**What this means for the context lever.** The dense arm has exactly one
context-cost lever and it is not connected: attention is `t²` with no window, no
segment, no compression, and at s1024 that is 6% of the step. The lever becomes
worth building when `t` grows (the band is flat in `t`, the dense call is not) —
but the shape of the win is a *chunked* window that avoids materialising the
band, not a mask on a full `t×t` call, and it has to be written in
`transformer.py` because `FullAttention` belongs to the loop form.

---

## 6. What could skew this (read before quoting a number)

1. **The window mean is host-measured, and the box is shared.** `tokens_per_s`
   divides by a `time.perf_counter()` interval (`train/loop.py:2369`), so a host
   descheduled between two log windows inflates that window. The box carried four
   to eleven queued GPU jobs from other lanes all session (long CPU jobs too:
   `niah.py`, `compile_matrix`), and the *median* moves 10–20 ms between passes
   while the **floor reproduces to ≤1.3 ms** (default 147.95/150.78; efficient
   157.14/157.66; math 306.18/304.83). That is why §3 quotes floors for the
   ranking and medians only as "what a run pays". Pass 1's `default` median
   (221 ms) is not a step time — it is a card that had just changed hands.
2. **`log_every=5` is not the live cadence.** It adds one host read per 5 steps
   (`_host_scalar(last_loss_dev)` + counter drain); the live recipe logs every
   10–50. All arms share this, so the ranking is unaffected, but the absolute
   number is not the recipe's. The floor (147.9 ms) sits 1.4% above the audit's
   145.8 ms, which came from a `log_every`-window reading of a different run —
   consistent, not identical.
3. **`torch.profiler` inflates the totals, not the ratios.** One profiled
   fwd+bwd sums to 193 ms of device time against a real ~148 ms step, so only
   per-op comparisons *within* the profile are used (flash vs efficient attention
   is the one that is cross-checked against the step).
4. **Two passes, not three.** §1.2 asks for three seeds for a *quality* verdict;
   this is a speed measurement with a mechanism that reproduces at the kernel
   level, so two passes + the device-time agreement were taken as sufficient.
   The two further passes that were queued (`rep3`/`rep4`, reverse order) never
   got the card inside this session.
5. **Efficient and flash are different numerics.** Flash and memory-efficient
   both keep bf16 tiled online softmax, but they reduce in a different order;
   math upcasts q/k to fp32 and materialises the score tensor. "Same step,
   different kernel" is therefore not "same model": the recommendation does not
   need the comparison, but a future arm that *wants* to switch backends owes a
   parity gate, not just a time.
6. **One GPU, one torch build.** All of this is torch 2.13.0+cu129 on an RTX
   5070 Ti; the 5060 Ti measurement in `fullattn.py` shows the same flag can
   invert on a different die of the same arch (§4.1). A backend claim must name
   the device.

---

## 7. Recommendation

**Leave it alone.** Concretely:

1. **Do not force a backend.** The default is `FLASH_ATTENTION`, it is the
   fastest of the four on this card, and the two alternatives cost +6.2% and
   +107% of a step. There is no version of this lever worth a config flag.
2. **Do not flip `sdpa_mem_efficient`.** It is dead on `block_form="transformer"`
   and harmful where it is live on this GPU (§4.1). Fix the docstring to name
   the 5060 Ti, or delete the flag (§4.1 is the evidence that it is not a
   portable win) — but that is a different commit in a file this audit does not
   own.
3. **Close the audit's C5 as 0%, with this document as the artifact.** The
   "0–12%" band collapses to 0: attention is 9.2 ms of a 147.9 ms step and it is
   already the flash kernel.
4. **Add the two counters** (`sdpa_default_backend` at `transformer.py:267`, and
   one for the window path when it exists). Not for speed — so the next reader
   can answer §2 from a log line instead of a profiler.
5. **Either wire `attn_window`/`attn_window_query_chunk` into
   `TransformerBlock` or refuse them on this block form** (`validate_block_form`,
   the fix C7 proposes for `residual_dtype`). Two fields now accept a value and
   change nothing; a third makes it a pattern.
6. **The context lever is unbuilt, and that is the real finding** — but it is not
   a 5–10% step win at s1024 (attention is 6% of the step, and the banded
   alternative at this length is slower). It becomes a win only where `t` grows,
   and only as a chunked band, not as a mask.

---

## 8. Reproduction

```bash
cd /home/sehaxe/dormouse/port
# 1. which backend the default picks + the four kernels (one GPU process)
GPU_TIMEOUT=900 bash scripts/gpu_lock.sh .venv/bin/python /tmp/opencode/sdpa_probe.py
#    -> /tmp/opencode/sdpa/probe.json: aten op, kernel names, can_use_* on the
#       captured SDPAParams, per-mode device ms

# 2. the timed arms (fresh process per arm; resumable; skips finished arms)
OUT=/tmp/opencode/sdpa FORCE=1 ARM_LIST="default flash efficient math window512 window1024 memeff_flag" \
    bash scripts/gpu_lock.sh bash /tmp/opencode/sdpa_sweep.sh
#    -> <arm>.jsonl (the trainer's telemetry) + <arm>.meta.json (resolved config)

# 3. the table
.venv/bin/python /tmp/opencode/sdpa_table.py          # floors/medians/p95, one dir
```

`exit 75` (lock busy) and `exit 124` (timeout) mean nothing was measured and are
never quoted — `gpu_lock.sh` also refuses a card held above 2 000 MiB, and every
arm above passed that gate.

`docs-site/tools/manifest.mjs` has **no line for this document** — that file is
held by another lane this session, so adding it is the one owed step here
(`npm run check` in `docs-site/` after it lands). `python3
tools/check_doc_refs.py` is green with this file present (its single dead path is
pre-existing, in `port/src/dormouse/eval_window.py:7`).

---

## 9. One line (RU)

**Дефолт — flash, и это самый быстрый из четырёх: EFFICIENT +6.2% шага (и его
ядро — sm80-код на sm_120, поэтому «2.1× на sm_120» из `fullattn.py` на этой
карте неверно в другую сторону), MATH ×2.07. `attn_window` на s1024 = 0 мс,
потому что плотный блок его не читает (второй SILENT no-op после C7), а сама
полоса сделана маской, которая на этом контексте отняла бы flash. Неизвестный
из аудита закрыт как 0: рычага здесь нет, и «оставить как есть» — это результат.**
