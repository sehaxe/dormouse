# 2026-11-06 — BPB accounting for the 102.7M flagship (`flagship2`)

**Scope.** Read-only audit of the held-out BPB instrument: the formula, the
bytes it scores, the depth normalization, and the window. Independent forward
passes were run on the live checkpoint
`checkpoints/flagship2/flagship2.pt` (**step 49 999** — the run only saves
every 25k; it was at step ~68 600 when this was written) and on the previous
run's checkpoint `checkpoints/flagship/flagship.pt` (**step 149 999**). No file
under `port/`, `crates/` or the live run's output directory was modified; no
commit, no push. GPU work was capped at 2 GB
(`torch.cuda.set_per_process_memory_fraction(0.12)`); the live job was not
stopped. The probes did run alongside it and slowed it (`tokens_per_s` fell to
~28k in the logged window), so **no timing claims are made here**; the BPB
values are not timing-sensitive.

**Tags.** `[M]` measured on this box, instrument named. `[D]` derived arithmetic
from measured numbers. `[U]` unmeasured — an explicit absence, not a zero.

---

## 0. Verdict

**DEFECTIVE as a BPB instrument; CLEAN as arithmetic.**

The arithmetic is right: an independent forward over the trainer's exact window
reproduces the logged number to **0.0004 BPB** (depth-average) and
**0.0007 BPB** (`fin`). The denominator is exact, there is no padding, and
there is no double division by bytes.

The number is still not the model's BPB, for two independent reasons:

1. **The headline is the depth-averaged training objective, not the model's
   predictive distribution.** `bpb` is
   `(Σ_i CE_i + CE_final)/(T+1) / ln 2` over the T deep-supervision readouts and
   the final readout. At T=3 it reads **+0.106 to +0.120 BPB** worse than the
   final readout, which is the readout decode actually uses. `bpb_fin` is the
   model's BPB.
2. **The eval is one fixed 65 536-byte window** — 1.04 % of the 6 294 307-byte
   eval file and 0.004 % of the bytes the run has consumed — restarted at
   offset 0 on every eval. Across 10 disjoint 65 KB windows the final-readout
   BPB moves **0.316 BPB** (1.5382 → 1.8539), 1σ ≈ 0.09–0.11. A single 65 KB
   absolute BPB therefore has a **95 % interval of about ±0.20 BPB**. The
   window actually scored is **0.073–0.077 BPB easier** than the whole eval
   file, and that bias is stable across two checkpoints.

Net at step 64 000 (extrapolated from the 50k checkpoint; see §7): the logged
**1.7423 is +0.039 BPB above** the honest final-readout BPB over the whole
eval file (**1.703**). It is not 0.4. The rest
of the apparent gap is the eval distribution (`mix4` **raw web**, which v7 does
not train on, costs **+0.17 BPB** vs a mostly-unseen FineWeb-Edu slice) plus
whatever the model/optimizer/mechanism lanes find.

The current run is also **not** 0.4 behind the previous one. On the same
window, at step 64k it is **0.017 BPB behind the previous run's final** (171k)
with 107k steps still to run.

---

## 1. The exact definition

Let `T = model.cfg.max_iter` (this run: **3**), `batch = 8`, `seq_len = 1024`,
`eval_batches = 8`.

For each eval batch `b` (8 batches of 8 rows × 1024 targets):

* `CE_i,b` for `i = 1..T` is the mean CE of iteration `i`'s readout over the
  batch's `8 × 1024` scored positions (`loopblock.py:1395-1430`, `_iter_ce`,
  ending in `ce.mean()`).
* `CE_fin,b` is the mean CE of the **mean readout** — the tensor the model
  returns for inference and the one decode uses (`loopblock.py:967`,
  `recurrent_step` returns the same `out = out_acc / ran`).

The model computes (`model.py:1007`):

```
loss_b   = (Σ_{i=1..T} CE_i,b + CE_fin,b) / (T + 1)
loss     = (1/B) Σ_b loss_b                       # B = eval_batches
bpb      = loss / ln 2                            # loop.py:55-57
bpb_fin  = ((1/B) Σ_b CE_fin,b) / ln 2            # printed, loop.py:1000-1001
```

`_eval_loss` (`loop.py:876-927`) calls `stream.rewind()`, then for each of the
8 batches does `total += float(loss)`, `scored += int(x.numel())`, and returns
`total / max(batches,1)`, `scored`. There is **no division by `scored`**: each
`CE` is already a per-position mean, and `scored` is printed, not used as a
divisor. This is correct arithmetic; the defect is what `loss` is.

**Bytes in the denominator.** `x` is `[8, 1024]`, so `scored = 8 × 8192 =
65 536` target positions. The model is byte-level (`vocab = 256`, the target is
the next byte), so one position is one byte. The source span is 65 600 bytes
because each 1024-target row consumes `seq_len + 1 = 1025` bytes
(`data.py:394-397`, `_windows`). The 64 bytes at offsets `k × 1025` are never
targets; counting the 65 600 source bytes instead would lower BPB by
`65 536 / 65 600 = 0.99902`, i.e. **0.0017 BPB** at 1.74. Negligible.

**Depth divisor.** The eval path applies the same `/(T+1)` as training, because
it calls the same `model(x, y, pure=True)` and the division is inside
`DormouseModel.forward`. This is **not** a per-byte normalization bug: each
`CE_i` is already a mean over bytes. It is an **objective** choice: the number
is the training objective's value, not the deployed model's NLL. `pure=True`
does not rescue it — `pure` skips the aux terms and z-loss (`model.py:1008-1140`),
not the per-iteration deep-supervision terms. The code comment at
`loop.py:894` ("the held-out number is the PREDICTIVE CE alone") is false for
`loss`; `fin` is the predictive CE alone. In this run `ce_fixed=false`,
`logit_softcap=0`, `z_loss=0`, `moe_lb_mode="prop"` (not `aux`), so `loss`
contains no other term.

**Is 65 536 bytes the entire eval?** Yes. The eval is
`eval_batches × batch × seq_len = 8 × 8 × 1024 = 65 536` target bytes
(`data.py:720-722`), printed as `over 65536 B`. The eval file is 6 294 307 bytes
= 6 140 full 1025-byte rows = 6 287 360 scoreable targets (plus 807 trailing
bytes that do not form a row), so **1.04 % of the file is ever scored**.

---

## 2. Independent recomputation

Load the checkpoint with `load_model_half`, build `DormouseModel` from the
snapshot, run `model(x, y, pure=True, breakdown=True)` on CPU-built windows,
and compute the CE from the returned logits with plain torch
(`-log_softmax(logits).gather(1, y)`). Batch 1 rows reproduce the trainer's
`b8 × s1024` positions exactly: the trainer's 8 batches are 64 consecutive
1025-byte rows, and row grouping does not move a byte.

At the step-49 999 checkpoint, on rows 0..63 (the trainer's exact window):

| quantity | logged at step 50 000 | independent | Δ |
|---|---:|---:|---:|
| `loss` (depth-average CE) | 1.221974 | 1.222270 | +0.000296 |
| `bpb` = loss/ln2 | 1.762936 | 1.763363 | **+0.000427** |
| `fin` (final readout CE) | 1.1473 | 1.147773 | +0.000473 |
| `bpb_fin` = fin/ln2 | 1.655 | 1.655887 | **+0.000682** |

`[M]` Both deltas are below the 0.01 finding threshold; the arithmetic and the
checkpoint loading are clean. The b1-vs-b8 residue is ~0.001 BPB (bf16
reduction order), not a bug.

**CPU is not a valid substitute for this model.** The same checkpoint and the
same 1024-byte row read **fin CE 4.68 (6.75 BPB)** on CPU and **1.04
(1.50 BPB)** on CUDA, with `bf16=true` and `bf16=false` both 4.68 on CPU. A
CPU eval would report a ~5.2 BPB error. Do not use CPU for an independent BPB
of this checkpoint; the
cause is outside this review (mechanism-liveness lane).

---

## 3. The eval window: same 65 KB every time, and it is easy

`_eval_loss` calls `stream.rewind()` unconditionally (`loop.py:876`), and
`ByteStream.rewind` sets `_pos = lo` (`data.py:400-408`). With `eval_data` set,
`open_streams` builds `ByteStream(eval_data, batch, seq_len)` with `lo = 0`
(`data.py:787-793`), so **every eval of the run scores bytes 0..65 599 of
`web_eval.bin`**. Every eval point of `flagship2` (2k, 4k, … 68k) is a
measurement of the same 65 536 bytes. This was confirmed by exact
reproduction, not inferred: the step-50k logged number is the number my forward
gets on rows 0..63.

Ten disjoint 65 536-target windows, step-49 999 checkpoint (`[M]`):

| window | source bytes | `fin` BPB | depth-avg BPB | premium |
|---:|---:|---:|---:|---:|
| 0 | 0..65 600 | 1.6559 | 1.7634 | 0.1075 |
| 1 | 65 600..131 200 | 1.7106 | 1.8132 | 0.1026 |
| 2 | 131 200..196 800 | 1.7675 | 1.8711 | 0.1036 |
| 3 | 196 800..262 400 | 1.8539 | 1.9650 | 0.1111 |
| 4 | 262 400..328 000 | 1.6696 | 1.7794 | 0.1098 |
| 5 | 328 000..393 600 | 1.7920 | 1.8993 | 0.1073 |
| 6 | 393 600..459 200 | 1.7143 | 1.8215 | 0.1072 |
| 7 | 459 200..524 800 | 1.6944 | 1.7898 | 0.0955 |
| 8 | 524 800..590 400 | 1.7330 | 1.8418 | 0.1088 |
| 9 | 590 400..656 000 | 1.5382 | 1.6427 | 0.1045 |
| **mean** | | **1.7129** | **1.8187** | **0.1057** |

* `fin` window SD = **0.0853**, range **0.3156**; depth-avg SD = 0.0867,
  range 0.3223.
* On the full 4 MiB slice (rows 0..4095, 4 194 304 targets): `fin` **1.71852**,
  depth-avg **1.82441**.
* On the whole file (rows 0..6139, 6 287 360 targets): `fin` **1.73303**,
  depth-avg **1.83963**. 95 blocks of 64 KB: SD 0.1073, SEM 0.0110, 95 % CI of
  the mean **±0.0216**.

**Confidence interval of a 65 KB BPB.** The naive byte-sampling SE is wrong
here. The per-byte CE SD is 1.498 nats and lag-1 autocorrelation inside a row
is only 0.087, but rows are correlated through documents: 64 independent rows
would give 0.033 BPB, the observed window/block SD is **0.085–0.107 BPB**.
Therefore a single 65 KB window has a **95 % interval of ±0.19 BPB** (t₉ × SD
from the 10 windows) to **±0.21 BPB** (1.96 × the 95-block SD). The project's
`docs/reviews/2026-11-05-architecture-review-dense.md:723` estimate
("~0.008 BPB" from a per-byte sd of 1.5 nats) is the independent-bytes number
and is **~12× too small** for this corpus. Use the block/window unit.

**The trainer's window is easier than the file.** Whole file minus window 0:
`fin` **+0.0771**, depth-avg **+0.0763** at step 50k; **+0.0731 / +0.0719** at
the previous run's step 150k. The bias is stable across checkpoints, so it is a
property of the slice, not noise. The run has been training and being read on a
window that is ~0.075 BPB easier than the file it is supposed to represent.

**Consequence for the curve.** Every eval on this run is a repeated
measurement of one fixed sample. The eval-to-eval smoothness is partly the same-window offset
cancelling, not corpus-level smoothness. Same-window paired comparisons are
still precise (numerical residue ~0.001 BPB, the window offset is common to
both arms); the *generalization* of a measured effect to the corpus is not
measured by this window, and a 0.01 "win" on it is not a corpus win.

---

## 4. Padding and the denominator

There is **no padding** in this eval:

* All 8 batches are full (`8 × 1024`); the file is 6.29 MB, far larger than the
  65 600-byte window, so `next_batch` never returns a short row.
* `scored` is `Σ x.numel() = 65 536`, exactly the number of target positions.
  No padded position is counted because none exists.
* The eval stream is flat even when training uses `doc_masking`: `open_streams`
  deliberately leaves the eval flat (`data.py:774-779`) and `_eval_loss` passes
  no `loss_mask`, so every target counts. If a mask were passed, `scored` would
  still count masked positions (`x.numel()`), which would be a denominator bug —
  it is not on this path.

**Latent defect if the window is grown past the file.** For an explicit
`eval_data` file, `open_streams` never sets `eval_tail_capped` (`data.py:787-793`;
the cap is only in the eval-tail-of-train branch, `data.py:815-837`).
`ByteStream._refill` wraps at `hi` (`data.py:372-384`), and `_eval_loss` only
recomputes `scored` when `eval_tail_capped` is not `None` (`loop.py:912-918`).
So `eval_batches > 767` on this 6 294 307-byte file would silently re-score the
head of the file and still print the duplicated count as `over N B`. The
current `eval_batches=8` is safe. Any "score the whole file" change must use
**767** batches, not 768, or fix the wrap counter first.

---

## 5. The 4.2 GB claim and eval disjointness

**The step consumes 24 576 bytes, not 8 192.** `loop.py:1251`:
`tokens_per_step = batch × seq_len × grad_accum = 8 × 1024 × 3 = 24 576`. The
mix plan banner agrees: `steps=513000 micro-batches (171000 x grad_accum=3)`.

| point | arithmetic | bytes |
|---|---:|---:|
| planned end (171k) | 171 000 × 24 576 | **4.202 GB** |
| at step 64 000 | 64 000 × 24 576 | **1.573 GB** |
| at step 67 600 (latest log) | 67 600 × 24 576 | **1.661 GB** |
| the "0.53 GB" figure | 64 000 × 8 192 | 0.524 GB |

The previous run finished 171 000 steps, so it did consume **4.20 GB**; the
current run is at **39.5 %** of that, not 12.6 %.

`[M]` The checkpoint at step 49 999 was written ~5 h 44 min after the run
started (00:26 → 06:10), i.e. 50 000 × 24 576 / 20 640 s = **59.5 kB/s**,
matching the logged `tokens_per_s` ~57–60 kB/s. If a step consumed 8 192 B the
rate would be ~19.8 kB/s. **The run has seen 3× the bytes the brief assumed.**
Any "it should be at 1.2–1.35 by now" argument must use 1.57 GB at step 64k,
not 0.53 GB.

**Eval disjointness.** `web_eval.bin` is not a training source: `mix_v7.toml`
lists 12 sources and none is `web_eval.bin` or `mix4/web.bin`. The `mix4`
manifest records it as a whole-shard carve (`carve: web_04108.txt`,
`method: doc_sha1_disjoint`, `eval_units: 1`, `train_units: 373`,
`overlap: 0`). That proof is against **mix4 web**, which v7 does not train on;
it does not cover `fwedu` (FineWeb-Edu), the largest v7 source. A spot check of
32 × 64-byte anchors spread across `web_eval.bin` found **0 exact hits** in the
consumed prefix of all eight stage-1 sources (1.03 GB fwedu, 0.27 cosmo,
0.34 ru, 0.16 wiki_en, 0.11 wiki_ru, 0.09 rudialog, 0.09 instruct,
0.11 reasoning). That is a spot check, not a document-level proof; the honest
gate is a full shingle/doc-hash sweep across `fwedu` before the next run.

**The eval is out of distribution.** v7 trains on FineWeb-Edu (the 42 % target
share; 58 % of stage-1 steps) plus cosmo/ru/code/math/science/…;
`web_eval.bin` is `mix4` **raw web**, which
v7 never trains on. At the previous run's step 150k, the same checkpoint scores
**1.6894 `fin`** on the whole `web_eval.bin` and **1.5245 / 1.5188** on two
2 MB samples of `fwedu.bin` (mid and tail, both beyond the run's consumed
prefix) — a **+0.168 BPB** domain penalty for the eval file. At step 50k the
same comparison is 1.7330 vs 1.5533 = **+0.180**. This is the largest single
term in the apparent 0.4, and it is an eval-choice defect, not arithmetic.

---

## 6. `bpb` vs `loss / ln 2`

They are the same function: `bpb(loss) = loss / math.log(2)` (`loop.py:55-57`),
and every `bpb` in the JSONL is `bpb(ev)` where `ev` is the depth-average CE
(`loop.py:1936`). The relation is exact:

```
bpb     = ((Σ_i CE_i + CE_fin)/(T+1)) / ln 2
bpb_fin = CE_fin / ln 2
bpb     - bpb_fin = (Σ_i (CE_i - CE_fin) / (T+1)) / ln 2  > 0 after the loop specializes
```

At step 50k: `bpb` 1.76294, `bpb_fin` 1.655, premium **0.108**.
At step 64k: `bpb` 1.74231, `bpb_fin` 1.626, premium **0.116**.
At the previous run's 150k/171k: premium **0.118 / 0.120**.
Across the 10 windows the premium is 0.0955–0.1111 (mean 0.1057); across the
64 blocks of the 4 MiB slice it is 0.091–0.128 (mean 0.1059); the whole file's
global premium is 0.1066. It is a stable function of depth, not noise.

The JSONL eval event carries only `loss` and `bpb` (`loop.py:1936`); `fin` and
`bpb_fin` are printed to stdout only (`loop.py:999-1003`). A JSONL/dashboard
consumer therefore never sees the model's BPB, only the objective's value.

---

## 7. What the model honestly scores

| checkpoint / protocol | bytes | `fin` BPB | depth-avg BPB |
|---|---:|---:|---:|
| `flagship2` 50k, trainer window 0 | 65 536 | 1.6559 | 1.7634 |
| `flagship2` 50k, 10 disjoint 65 KB windows, mean | 655 360 | 1.7129 | 1.8187 |
| `flagship2` 50k, 4 MiB slice | 4 194 304 | 1.7185 ±0.024 | 1.8244 |
| `flagship2` 50k, whole eval file | 6 287 360 | **1.7330 ±0.022** | **1.8396** |
| `flagship2` 64k (logged window 0; whole-file extrapolation) | 6 287 360 | **1.7034** | 1.8186 |
| `flagship` 150k, trainer window 0 | 65 536 | 1.6164 | 1.7348 |
| `flagship` 150k, whole eval file | 6 287 360 | **1.6894 ±0.020** | **1.8066** |
| `flagship` 171k final (logged; whole-file extrapolation) | 6 287 360 | **~1.6826** | ~1.8011 |
| `flagship` 150k, `fwedu` 2 MB mid + tail | 4 194 304 | **1.5217** | 1.6438 |

`[M]` for the rows with a checkpoint; `[D]` for the 64k and 171k whole-file
rows (the window-0 delta from the previous checkpoint carried over, using the
measured 0.073–0.077 window bias). There is no 64k checkpoint (the newest is
49 999); the 171k weights are not on disk (the last save is 149 999).

**The 64k number is not a regression against the finished run.** On the same
window the logged `bpb_fin` is 1.626 at 64k vs 1.609 at 171k: **0.017 BPB
behind the previous final with 107k steps left**, not 0.4. The previous run's
first 120k steps used a different window (`b24 × 1024 = 196 608 B` in
`flagship1.log`), so its early curve is not comparable anyway. This is an
instrument-level comparison, not an A/B: the training recipe differs (fused
residual/epilogue, mHC log-softmax Sinkhorn, LFB `prop`).

**The 1.7292 "wall" is the depth-average on the easy 65 KB window.** The
previous run's honest final-readout number on the full eval file is
**~1.6826**, not 1.7292. The relative flatness over the last 41k steps
(−0.0008 BPB on the same window) is unaffected; the absolute number is ~0.047
BPB above the honest `fin` (1.6826) and ~0.072 BPB below the honest whole-file
depth-average (1.8011). The 0.12 is the depth premium at 171k, not the
accounting error. Do not compare 1.7292 to a literature BPB.

---

## 8. Decomposing the apparent ~0.4 BPB gap at step 64k

| step | BPB | what changed |
|---|---:|---|
| logged `bpb` (window 0, depth-avg) | 1.7423 | the number under discussion |
| − depth premium on window 0 | −0.116 | depth-avg objective → final readout |
| = `bpb_fin` on window 0 | 1.6263 | deployed-model BPB, still one window |
| + window-0 ease vs whole file | +0.077 | the scored 65 KB is easier than the file |
| = `fin` on the whole eval file | **1.7034** | honest BPB on `web_eval.bin` |
| − mix4-web vs fwedu domain gap | −0.170 | eval is raw web; training is FineWeb-Edu + mix |
| = in-distribution estimate (`fwedu` samples) | **~1.53** | same checkpoint, mostly-unseen training domain |
| 124M vanilla FineWeb-Edu @ ~1B tokens | 1.20–1.35 | the anchor in the brief |

So: **the BPB instrument explains +0.039 of the ~0.4** (net, at 64k; +0.116 if
you compare only same-window `bpb` vs `bpb_fin`; −0.076 if you compare to the
whole-file depth-average). **The eval domain explains ~0.17.** The remaining
**~0.18 (vs 1.35) to ~0.33 (vs 1.20)** is a real model/data/training deficit on
in-distribution text. It is not a counting error, and the current run is not
behind the previous one on the same clock.

Caveats on the last row: the `fwedu` numbers are 2 MB samples of the **training
source** (mid and tail, beyond the consumed prefix), not a held-out FineWeb-Edu
validation set; the 1.20–1.35 anchor is for a 124M vanilla at 1B tokens, while
this is a 102M mixed-domain recurrent/MoE model at 1.57 GB at step 64k. The
vanilla control is the other lane's job. But the direction and size of the
domain term are measured, not assumed.

---

## 9. The honest protocol

**For an absolute number** (a claim, a wall, a literature comparison):

1. Report **`fin / ln2`**, never the depth-average. If the objective value is
   wanted for training diagnostics, label it `objective/ln2`, not BPB.
2. Score at least **4 MiB**; for the current file, score
   **6 283 264 of the 6 287 360 targets** with `eval_batches=767`
   (767 × 8 × 1025 = 6 289 400 source bytes ≤ 6 294 307; 767 × 8 × 1024 =
   6 283 264 targets = 99.93 %; 768 would wrap — see §4).
3. Report the mean **and** a block spread: 64 KB blocks, SD ≈ 0.10 BPB; the
   whole-file mean's 95 % CI is **±0.022**, the 4 MiB mean's is **±0.024**.
4. Do not quote a single 65 KB number as an absolute BPB: its 95 % interval is
   **±0.19–0.21 BPB**. The current 0.002–0.005 "paired-eval resolution" in
   `docs/protocols/AB-PROTOCOL.md` is only meaningful as a same-window figure
   (the window offset cancels); it is not an absolute resolution.

**For the training curve** (cheap, every 2k steps): score **10 disjoint
65 KB windows** (655 KB) and report mean ± spread. At step 50k that is
`fin` **1.7129**, 95 % CI of the mean **±0.061**, window spread **0.316**.
A 1 MiB tier (128 batches) is 16× the current eval and already gives a ~±0.05
95 % interval; the full file is 96× and belongs on a 10k–20k cadence, not 2k.

**For an A/B** (two arms, same checkpoint budget): keep both arms on the
**same** window so the offset cancels — that part of the current protocol is
sound — but confirm the effect on a ≥4 MiB disjoint set before believing it
generalizes. The window-to-window variation of the *effect* is unmeasured; a
0.01 win on one 65 KB slice is not a corpus win.

**The eval file itself is too small and one-shard.** 6.29 MB is one
`mix/web` shard (`web_04108.txt`); scoring all of it removes the file-internal
sampling error but not shard-to-shard variation. For absolute claims, carve a
multi-shard, multi-domain held-out file of ≥100 MB from the **v7** mix (or keep
this file as a legacy comparable and add a v7-matched one), and score it in
4 MiB disjoint blocks.

---

## 10. Reproduction

The probe is `/tmp/opencode/bpb_probe.py` (plain torch; loads the snapshot with
`RunConfig.from_snapshot`, the weights with `load_model_half`, runs
`model(x, y, pure=True, breakdown=True)` at 1 row per forward, and computes the
CE from the returned logits itself). GPU cap:
`torch.cuda.set_per_process_memory_fraction(0.12)`.

```bash
cd /home/sehaxe/dormouse/port
# exact trainer window (rows 0..63) at the step-49 999 checkpoint
.venv/bin/python /tmp/opencode/bpb_probe.py exact
# ten disjoint 65 KB windows (rows 0..639)
.venv/bin/python /tmp/opencode/bpb_probe.py windows
# 4 MiB, then the rest of the file (6 140 full rows in total)
.venv/bin/python /tmp/opencode/bpb_probe.py big 0 4096
.venv/bin/python /tmp/opencode/bpb_probe.py big 4096 2043
.venv/bin/python /tmp/opencode/bpb_probe.py big 6139 1
# previous run's 150k checkpoint, whole file
BPB_CKPT=/home/sehaxe/dormouse/checkpoints/flagship/flagship.pt \
BPB_SNAP=/home/sehaxe/dormouse/checkpoints/flagship/flagship.config.toml \
  .venv/bin/python /tmp/opencode/bpb_probe.py big 0 6139
BPB_CKPT=... BPB_SNAP=... .venv/bin/python /tmp/opencode/bpb_probe.py big 6139 1
# fwedu samples (2 MiB, mid and tail)
BPB_CKPT=... BPB_SNAP=... .venv/bin/python /tmp/opencode/bpb_probe.py \
  file /home/sehaxe/dormouse/data_local/best/fwedu.bin 5853658 2048
BPB_CKPT=... BPB_SNAP=... .venv/bin/python /tmp/opencode/bpb_probe.py \
  file /home/sehaxe/dormouse/data_local/best/fwedu.bin 11705268 2048
```

Raw row-level results are in `/tmp/opencode/bpb_{exact,windows,big_*,file_*}.json`.
The live run was not modified, paused or restarted.

---

## 11. What this review does not cover

* Why the CPU forward reads fin CE 4.68 (6.75 BPB) instead of 1.04 (1.50 BPB).
  It is a separate numerical defect; any CPU-based BPB in the project is
  suspect for this model.
* Why the model is ~0.18–0.33 BPB above the 124M vanilla anchor on
  in-distribution text. That is the mechanism/control/optimizer/data lanes.
* Whether `fwedu` is representative of FineWeb-Edu's own validation split.
  Only two 2 MB samples were scored (they agree to 0.006 BPB).
* A document-level disjointness proof between `web_eval.bin` and `fwedu`. The
  32-anchor exact-match spot check found 0 hits; that is not a proof.
