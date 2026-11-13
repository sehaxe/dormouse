# 2026-11-08 — Activation VRAM at b8 × s1024: what holds the step, and what the levers buy

Lane `port/`, HEAD `5c37420` (`port/src/dormouse/config.py` dirty in the worktree;
line numbers there are this tree's, the other cited files are HEAD-clean).
Card: RTX 5070 Ti, 15.56 GiB usable, torch 2.13.0+cu129. Every GPU number below
was taken under `scripts/gpu_lock.sh` (one process on the card), with the
trainer's own model + `MixOptimizer` and a synthetic `randint(0,256)` batch:
this file prices **memory** and **step time**, never quality, and the data loader
is not part of either.

Marks: **[M]** measured here (artifact named) · **[C]** computed from a named
code path or from measured inputs · **[S]** inference, with the check that
settles it. A number without an artifact is a defect.

**Artifacts.** `/home/sehaxe/logs/vram/run.log` (every attempt's console),
`kda95_extra.json` (the two KDA combos), and four `*polluted*.json` files that
are kept as the record of a harness defect (a caught exception's frame pinned
the previous arm's graph: every arm after a refused/failed one inherited
**+4.114 GiB**) — §8 says exactly what each file is trustworthy for. The
instrument is `port/scripts/bench_activation_vram.py` (new, untracked); it
reuses `dormouse.cli._model`, `dormouse.train.optim.MixOptimizer` and the
presets, and it is the only place a per-arm map was taken.

---

## 0. TL;DR

1. **The dense 95M (`configs/flagship3_dense.toml`, 95 035 200 params, 12 unique
   transformer blocks) peaks at 4.49 GiB ALLOCATED at b8 × s1024** [M], not
   6.2. The 6.2 GiB figure is `nvidia-smi` **residency**: allocated + the
   allocator's reserved-but-inactive blocks + the CUDA context (§2.3).
2. **Of the 4.49 GiB, only 1.06 is the model**: params 0.354 + grads 0.354 +
   Muon momentum 0.352 + AdamW m/v 0.003 [C from the meta tree]. **3.04 GiB is
   the autograd save set** (391 tensors) [M], and **~0.4 GiB** is the backward's
   gradient/transient gap [M]. There is **no whale**: the largest single live
   block is 48 MiB, and the peak is the sum of 12 identical layers.
3. **The save set is 12 copies of the same block**: the fp32 residual stream and
   its two LayerNorm reads dominate (`[8,1024,832]` fp32 × 27 = 0.69 GiB),
   then the bf16 attention tensors (`[8,13,1024,64]` × 48 = 0.61 GiB) and the
   FFN's `[8192,3072]` bf16 × 24 = 1.15 GiB. Half of the fp32 residual stream
   would come back as bf16 (**0.34 GiB, −7.7 % of the peak**) with no recompute
   at all — the second-largest lever on this form after a checkpoint that does
   not exist here.
4. **On this form `segment_len` and `activation_recompute`/`iter_checkpoint` do
   not exist.** They are LOUD-refused (`block_form="transformer"` has no state
   to carry; there is no loop block to checkpoint) [M]. The only activation
   lever that reaches the dense form is `activation_offload`: **−50.2 % peak
   (4.49 → 2.24 GiB) for ×4.49 step time** [M], and a 4 GiB budget **refuses
   b12** (the selected set exceeds it by 0.27 GiB) [M].
5. **On the KDA loop form (`configs/arch_hybrid_kda.toml`, 95.9M, 12 loop
   passes) the ordering is the opposite**: `activation_recompute=full` is
   **−65.2 % peak (6.79 → 2.36 GiB) for +11 % step time** [M], and it beats
   every other lever on both axes: `segment512` −38.4 %/+53 %,
   `segment256` −58.2 %/+142 %, `selective` −26.1 %/+6.7 %, `iter_checkpoint`
   is the same arm as `full` (−65.3 %). `activation_offload=4` is **refused** on
   this arm at b8 (needs ≥ 4.99 GiB) [M]. Stacking offload on top of
   checkpointing buys **0.006 GiB** and costs **+78 %** [M]: after a checkpoint
   the outer hook sees only the checkpoint's inputs, so there is nothing large
   left to park. Stacking `muon_state_dtype=bf16` on the checkpoint does pay:
   **−0.179 GiB** [M].
6. **The dense 95M batch ceiling is the card**: b8 4.57, b16 7.72, b20 9.37,
   b24 10.96, **b28 12.58 GiB — the largest that fits** [M]; b32 **OOM** [M]
   needing >15.4 GiB, ~1.3 GiB above the 0.40-GiB-per-batch linear fit, so the
   curve steepens there (kernel fallback or a larger transient; not isolated).
   b28 runs at **54.8 kB/s, 1.9× the b8 rate**. The offload does not raise this
   ceiling for the 95M: its budget must cover the whole selected set and its cost
   scales with bytes moved at **~6.3 GiB/s** measured (§4) [M], so a bigger
   budget buys batches at 4-6× the step time.
7. **Segmentation is a memory decision, not a model change, to 1e-6 on the
   loss**: on the repo's 64-byte CPU window and `tiny_model`, `|Δloss|` is
   **0.0** at 4 segments and **4.77e-7** at 2, while `max|Δlogit|` is 1.16e-6
   absolute (**2.1e-7 relative** to a 5.6 logit scale) [M]. The existing gate
   (`tests/test_longctx.py`, 1e-4 abs) is right for logits and too loose for
   the scalar; `tests/test_segment_parity_strict.py` now pins the loss at 1e-6.
8. **For the 1B plan**: the `b1` preset is the KDA loop form, so
   `activation_recompute=full` is available there and is the lever to ship
   (measured multiplier −64…65 % peak / +7…16 % time on a 95M KDA; the ledger's
   arithmetic puts it at **11.798 GiB** of the 1B's activation branch
   [C]). If the 1B lands on the **dense transformer** form instead, the
   checkpoint must be **ported** (40 lines around `TransformerBlock._block`)
   before any 1B dense run, because on that form the only shipped lever is a
   4.6×-slower offload.

---

## 1. The target(s) and the shape

Two 95M models, because "95M dense" is ambiguous in this repo and the answer
depends on the block form:

| preset | params | form | loop | d / f / heads | why it is here |
|---|---:|---|---|---|---|
| `configs/flagship3_dense.toml` | 95 035 200 | `transformer` | 12 unique blocks, `max_iter=1` | 832 / 3072 / 13×64 | the ladder's dense control (`bench_engine`'s default) |
| `configs/arch_hybrid_kda.toml` | 95 875 404 | `kda` | `loop_blocks=12`, `max_iter=12` | 832 / 1792 / 13×64 | the KDA arm at the same budget; the ONLY 95M with a segmented/checkpointable forward |

Both were measured at `b8 × s1024` (8×1024 = 8 192 bytes/step). The step is
`forward + backward + opt.step()`; the recipe is the **preset's own defaults**
— `ns_dtype=fp32`, `ns_steps=5`, `muon_state_dtype=fp32`, `muon_lr_mult=1.0`
— i.e. the **slow arm**, not the throughput ladder's `--set`-overridden fast arm
(`docs/records/reviews/2026-11-08-kernel-audit.md` §1 measured 289.9 ms on the slow arm
against 145.8 ms on the fast one; this run's dense base is **272.9 ms**, the
same program on a quieter host). A time in this document is therefore
comparable to another row **in this document**, and to a ladder row only after
the recipe's `--set`s are matched. Memory is recipe-insensitive except for the
optimizer state (fp32 vs bf16: 0.35 vs 0.18 GiB at 95M).

**What the harness is.** `_model(run)` + `MixOptimizer(...)` from the trainer,
`model.configure_actmem(...)` once (as `train/loop.py` does), then 3 warm steps
and 7 timed ones; `peak` is `torch.cuda.max_memory_allocated` after a reset;
`ms/step` is the median, `kB/s = batch·seq/1e3 / s`. Two badges of the harness
are printed in every JSON: `counters.counters["loop_segments"]` proves the
segmented path ran (4 at `segment_len=256`, 0 elsewhere) and
`iter_checkpoint_saved=12` proves the per-iteration checkpoint ran.

**Not in the measurement** (all of them a few hundred MB or less at this size,
none of them a lever): the NaN firewall's `nan_to_num_`/`_foreach_mul_`
(in-place), grad clip (`_foreach_norm` + fused multiply), the data stream, and
the EMA shadow (off: `ema_decay=0.0`). A run with `ema_decay>0` carries one
extra full model copy (+0.35 GiB at 95M).

---

## 2. The map — where the 4.5 GiB actually sits

### 2.1 `memory_stats()` [M, `f3_b8s1024.json`, `run.log`]

| quantity | GiB | note |
|---|---:|---|
| `allocated_bytes.all.peak` (b8 arm) | **4.489** | this run; 4.547 in the earlier one — 1.3 % cell-to-cell |
| `allocated_bytes.all.peak` (b8 ladder row) | 4.504 | a separate build of the same model |
| `residual_gib` | 0.062 | bytes the failed map pinned; the harness SUBTRACTS it and prints it, so the row is this cell's own |

The KDA illustration, from `kda95_extra.json` (`segment256`, b8 × s1024):
allocated peak **2.878** GiB, reserved peak **4.207** GiB, inactive split 0.606,
device free 10.04 of 15.56 — i.e. the card reports **4.21 GiB "used"** for a
2.88 GiB step: the allocator's cache is 1.3 GiB of it, and the CUDA context +
other processes (this box runs a TUI, a browser and an agent) make up the rest.
That is the whole mystery of "6.2 GiB" vs 4.55 (§2.3).

### 2.2 The composition

| component | GiB | source |
|---|---:|---|
| parameters (fp32 masters) | 0.354 | meta tree, 95 035 200 × 4 B |
| gradients (fp32) | 0.354 | one per parameter |
| Muon momentum (fp32) | 0.352 | 94 568 448 2-D params × 4 B |
| AdamW `exp_avg`+`exp_avg_sq` | 0.003 | 466 752 1-D params × 8 B |
| **autograd save set** | **3.043** | [M] `saved_tensors_hooks`, 391 tensors, same forward |
| backward gradient/transient gap | ~0.4 | [M] peak − (fixed + saves) |
| **allocator peak** | **4.49** | [M] `max_memory_allocated` |

The save set is **device-independent** on this form: the CPU census at the same
shape agrees with the GPU's counter to 3 % (`C + k·b·s` fit on `t=256/512/1024`:
constant **180.94 MiB** of bf16 weight copies + **378.68 MiB per 1024 tokens**;
at 8 192 tokens = **3.135 GiB** [C] against the GPU's **3.043 GiB** [M]). The
KDA form does **not** transfer: on CPU it runs the SDPA fallback, which saves
`[13,1024,64]` fp32 score-side tensors that FLA's chunk kernel does not keep
(9.80 GiB census vs 6.79 GiB measured [M]) — quote the GPU number for KDA only.

### 2.3 The top of the peak, and why `nvidia-smi` says ~6.2

**How the map was taken.** `torch.cuda.memory_snapshot()` after a warmed step,
with `_record_memory_history(enabled="state")` so every live block carries the
stack that allocated it, grouped by that frame; plus `saved_tensors_hooks` on
the same forward for the SAVED set's shapes (the snapshot carries stacks, not
shapes). The device-side snapshot's own per-block list is **not** in this
report: the harness's map computed its tables and then raised a stale key in its
printout (`KeyError: 'trace_peak_gib'`), and the card did not free again for a
re-run before this report was written (the fixed harness is
`scripts/bench_activation_vram.py --arms base --skip-ladder`). What follows is
the same census taken on the CPU at the identical shapes, **cross-checked
against the device's own saved-bytes total to 3 %** (`saved 3.043 GiB in 391
tensors` [M]) — the shapes, sizes and call sites are the Python frames on either
side; only the numbering differs.

At the forward→backward boundary every layer's save is live and no gradient has
been allocated yet: the peak is `params + opt state + saves + transients`. The
top live blocks are **not** whales — they are 12 copies of six shapes:

| shape | dtype | bytes each | live count | total GiB | who saved it |
|---|---:|---:|---:|---:|---|
| `[8,1024,832]` | fp32 | 26.0 MiB | 27 | 0.686 | `layer_norm` backward input (the residual stream and both LN reads, 2.25/layer) |
| `[8,13,1024,64]` | bf16 | 13.0 MiB | 48 | 0.610 | SDPA's q/k/v/out (4/layer, `transformer.py:267`) |
| `[8192,3072]` | bf16 | 48.0 MiB | 24 | 1.125 | the FFN's gelu input + `fc2` input (2/layer) |
| `[8192,832]` | bf16 | 13.0 MiB | 37 | 0.470 | `mm_mixed` inputs/outputs (3/layer) |
| `[832,3072]`, `[3072,832]`, `[832,2496]`, `[832,832]` | bf16 | 4.9/4.9/4.0/1.3 MiB | 48 | 0.176 | the bf16 **weight copies** `mm_mixed` casts per forward (constant in batch) |

The CPU census at **b1 × s1024** (shapes scale by `b`; the totals are that
forward's), top rows by total bytes — the GPU's 391-tensor save set is the same
set, and the `C + k·b·s` fit above comes from this table:

| shape | dtype | total MiB | × | at |
|---|---:|---:|---:|---|
| `[832,3072]` | bf16 | 58.5 | 12 | `tsct.py:292 mm_mixed` (fc weight copy) |
| `[3072,832]` | bf16 | 58.5 | 12 | `mm_mixed` (fc2 weight copy) |
| `[832,2496]` | bf16 | 47.5 | 12 | `mm_mixed` (qkv weight copy) |
| `[1,1024,832]` | fp32 | 87.8 | 27 | `functional.py:2987 layer_norm` |
| `[1,13,1024,64]` | bf16 | 78.0 | 48 | `transformer.py:267 _block` (SDPA) |
| `[1,1024,3072]` | bf16 | 72.0 | 12 | `transformer.py:276 _block` (gelu) |
| `[1024,3072]` | bf16 | 72.0 | 12 | `mm_mixed` (fc2 input) |
| `[1024,832]` | bf16 | 60.1 | 37 | `mm_mixed` |

**The one-line diagnosis**: this block stores its residual stream in **fp32**
(`x` arrives in `qkv.weight.dtype`, and the masters are fp32 — `transformer.py`
casts only at the GEMM seam), so every `[b,t,d]` save costs 27.3 MB where a
bf16 stream would cost 13.6. `residual_dtype="bf16"` exists as a config field
and is **inert on this form** (it lives in `LoopBlock`; `TransformerBlock` has
no such attribute — kernel audit C7). It is worth **0.34 GiB (−7.7 % of the
peak)** at zero recompute: a real lever, but an order of magnitude below the
checkpoint the form does not have.

**`nvidia-smi` vs the allocator.** `nvidia-smi` reports the process's resident
context: the allocator's peak plus reserved-but-inactive blocks plus the CUDA
context. The recorded "~6.2 GiB" is therefore consistent with the 4.49-4.55 GiB
allocated peak + ~1 GiB of cache/context + the other processes on the box (§2.1
shows the same gap measured on the KDA cell: 4.21 GiB "used" for a 2.88 GiB
step). The harness prints `reserved_*` and `mem_get_info` in every row, so a
reader can close the gap instead of guessing — quote `max_memory_allocated` for
"does it fit", and `nvidia-smi` only for "what the card shows".

---

## 3. The levers at b8 × s1024

### 3.1 The dense 95M — three levers exist, four arms are refused

| arm | `config.py` field | peak GiB | Δ peak | ms/step | Δ time | kB/s | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| base | — | **4.489** | — | 282.1 | — | 29.0 | control |
| `segment_len=512` | 2430 | — | — | — | — | — | **REFUSED** (`block_form="transformer"` has no carried state) |
| `activation_recompute="full"` | 2462 | — | — | — | — | — | **REFUSED** (no loop block) |
| `activation_recompute="selective"` | 2462 | — | — | — | — | — | **REFUSED** (no loop block) |
| `iter_checkpoint=true` | 2451 | — | — | — | — | — | **REFUSED** (alias of `full`) |
| `activation_offload=4.0` | 2483 | **2.236** | **−50.2 %** | 1266.1 | ×4.49 | 6.5 | works; budget-limited |
| `activation_offload=8.0` | 2483 | 2.235 | −50.2 % | 1278.6 | ×4.53 | 6.4 | **no gain**: the set is 3.04 GiB, both budgets cover it |
| `muon_state_dtype="bf16"` | 2046 | 4.303 | −4.1 % | 277.4 | ×0.98 | 29.5 | memory-only lever (time spread is host noise; ledger measured 0.99×) |

The dense rows carry `residual_gib=0.062` (62 MiB pinned by the arm before
them); added back, the process-held peaks are 4.55 / 2.30 / 4.37 — which is
exactly what the earlier run's fresh process measured (4.547 / 2.298 / 4.392).
Both readings are in the JSON; the table quotes the cell's own.

The refusals are the **measured** answer to "what does segmentation/checkpointing
buy on the dense 95M": **nothing, because neither reaches the form**. The
message names the escape, e.g.:

> `segment_len=512 with block_form='transformer': the segmented path carries the
> KDA recurrence's state across cuts, and the vanilla decoder has no state to
> carry (its K/V context is the window). Escape: segment_len=0, or
> block_form='kda'.`
>
> `activation_recompute='full' but this model builds no loop block (use_byteflow
> Option A returns before the loop), so the arm would recompute nothing while the
> log line reported it on. Escape: use_byteflow=false, or
> activation_recompute='none'.`

This is rule 1.1 working: a memory flag that reached no code would have been a
silent no-op; here it stops the run. `activation_offload` is the exception
because its hook pair is model-agnostic.

**The offload's budget is all-or-nothing at the shape it profiles.** At b8 the
selected set is 3.04 GiB < 4 GiB and the whole set is parked (peak 2.30 GiB);
at b12 the set is 4.26 GiB and the arm **refuses** (`exceeded by 0.2739 GiB;
escape: activation_offload >= 4.2583 GiB`) [M]. It cannot partially park, so a
fixed budget has a hard max batch — the opposite of a capacity lever.

### 3.2 The KDA 95M — checkpointing dominates, and the levers nest

| arm | peak GiB | Δ peak | ms/step | Δ time | kB/s | counters |
|---|---:|---:|---:|---:|---:|---|
| base | **6.791** | — | 398.9 | — | 20.5 | `iter=12` |
| `segment_len=512` | 4.182 | −38.4 % | 612.0 | +53 % | 13.4 | `loop_segments=2` |
| `segment_len=256` | 2.836 | −58.2 % | 963.6 | +142 % | 8.5 | `loop_segments=4` |
| `activation_recompute="full"` | **2.360** | **−65.2 %** | **442.7** | **+11.0 %** | 18.5 | `iter_checkpoint_saved=12` |
| `activation_recompute="selective"` | 5.018 | −26.1 % | 425.7 | +6.7 % | 19.2 | `iter_checkpoint_saved=12` |
| `iter_checkpoint=true` | 2.359 | −65.3 % | 535.7 | +34 % | 15.3 | `iter_checkpoint_saved=12` (median inflated by one slow step; min 454 → +14 %) |
| `activation_offload=4.0` | — | — | — | — | — | **REFUSED** (needs ≥ 4.9941 GiB) |
| `full` + `muon_state_dtype=bf16` | **2.181** | **−67.9 %** | 467.7 | +17 % | 17.5 | the momentum halves: −0.179 GiB over `full` |
| `full` + `activation_offload=4.0` | 2.416 | −64.4 % | 795.4 | +78 % (vs base) | 10.3 | `offload_tensors=66`, 0.39 GiB parked |

**Read the memory column as exact and the time column as a two-run range.** The
peaks agree within **0.06 GiB** across three runs (checkpoint 2.360 / 2.422 /
2.422, segment512 4.182 / 4.244 / 4.244, base 6.791 three times) — the 0.06 is
exactly the `residual_gib` the later harness subtracts, so the cells are the same
size and only the accounting differs. The checkpoint cell's own time is stable to
1 % (442.7 / 446.3 / 446.4 ms). The BASE cell is the variable one — 383.9 →
417.0 → 398.9 ms — so a ratio against it inherits ±7 %. Hence "+11.0 %" here and
the bracketing "+7…16 %" in §0: the checkpoint costs a reproducible ~445 ms per
step; whether that reads as +7 % or +16 % over the one-shot path depends on which
base cell you divide by. The qualitative ordering does not: **checkpointing is
cheaper than every segmentation cut, and far cheaper than the offload.**

**Why `full` wins.** `checkpoint(use_reentrant=False)` around each of the 12
loop passes keeps only that pass's INPUT state and recomputes the body in the
backward; the save set collapses from 1.44 GiB to 0.10 GiB at b1/s1024 [M, CPU
census], i.e. one pass instead of twelve. `segment_len` does the same thing per
SEGMENT, so its saving is `1/N` of the window rather than `1/12` of the depth —
which is why `segment512` (−37 %) is weaker than `full` (−64 %) even though
both cost one extra forward. `selective` keeps the READ half (the q/k/v/g
projection outputs and the carried state) and recomputes only the WRITE half:
on this block the write half is most of the arithmetic but only a quarter of
the bytes, so it is dominated on both axes — 5.08 GiB against 2.42 for 403 ms
against 446, and the 43 ms it saves is the recompute it does not do.

**Composition is not additive.** `full + offload4` parks 0.39 GiB — after the
checkpoint, the outer hook sees only the checkpoint's inputs, and the largest is
32 KiB — for **+349 ms** and **−0.006 GiB**. The two levers are alternatives,
not rungs.

**Segmentation scales with the number of cuts, and so does its cost**:
512→256 halves the memory again (4.24→2.90) and takes the step from 601 to
997 ms (+66 %). The loop calls the arm once per iteration per segment, so `N`
cuts multiply the launch count and the state-carry work; the `loop_segments`
counter in the row is the proof the cuts are real.

### 3.3 What the optimizer state lever is worth

`muon_state_dtype="bf16"` halves the Muon momentum (0.352 → 0.176 GiB at 95M).
It is a **memory** lever: the ledger's speed reading is 0.99× (no launch change),
and the two runs here bracket the same conclusion — 328.6 ms in one, 279.9 ms in
the other, against a 273 ms base in both. The 17 % spread is host load (this box
was running a 5.3 GiB training job for part of the window), not the arm; read
the row as "no measurable time change". On the 1B it is the second-largest free
saving (1.875 GiB [C], `docs/reports/2026-11-06-vram-ledger.md` §4).

---

## 4. The batch ceiling at s1024

Base (no lever), dense 95M, all rows MEASURED in one process (`f3_b8s1024.json`,
ladder `base`, `residual_gib=0.062` added back so the column is process-held):

| batch | peak GiB | ms/step | kB/s | status |
|---:|---:|---:|---:|---|
| 8 | 4.57 | 281.8 | 29.1 | ok |
| 16 | **7.72** | 373.3 | 43.9 | ok |
| 20 | 9.37 | 433.6 | 47.2 | ok |
| 24 | 10.96 | 479.6 | 51.3 | ok |
| 28 | **12.58** | **522.9** | **54.8** | **ok — the ceiling** |
| 32 | — | — | — | **OOM** (`Tried to allocate 104.00 MiB`, 147 MiB free of 15.56) |

The slope is **0.400 GiB per batch unit** and four measured points sit on it to
0.05 GiB (4.57, 7.72, 9.37, 10.96, 12.58 = 4.57 + 0.400·(b−8)), which projects
b32 at **14.18 GiB** — yet b32 OOM'd with the card holding **>15.4 GiB**. So the
curve steepens between b28 and b32 by ~1.3 GiB, i.e. **b32 crosses a cliff that
is not arithmetic**: a kernel fallback (an unfused attention path allocating the
score matrix near `b·h = 416`) or a larger transient is the prime suspect, and
this run does not isolate which. The practical answer is the measured one:

**The largest batch that fits is b28, at 54.8 kB/s — 1.89× the b8 rate** (the
per-byte cost falls from 34.4 ms/kB at b8 to 18.2 ms/kB at b28; the fixed cost
is amortised, and b28 leaves 3.0 GiB of headroom over the 15.56 GiB card). If a
run must share the card, b24 (10.96 GiB, 51.3 kB/s) is the safe pick; b32 is not
reachable at s1024 on this card without a memory lever, and the only lever that
reaches this form (`activation_offload`) costs ×4.5 and caps at the budget.

The offload's ceiling is its own budget, and its rate is the PCIe link:

| cell | parked GiB | peak GiB | ms/step | implied round-trip rate |
|---|---:|---:|---:|---:|
| b8, budget 4.0 | 3.04 | 2.298 | 1266.1 | 6.08 GiB / 0.98 s = **6.2 GiB/s** |
| b8, budget 8.0 | 3.04 | 2.297 | 1278.6 | 6.2 GiB/s |
| b20, budget 8.0 (M*) | 7.4 | 2.293 | 2692.0 | 14.8 GiB / 2.26 s = **6.5 GiB/s** |

so parking `x` GiB costs ≈ `2x/6.4` seconds on this box. At b20 that is 2.3 s
against a 0.42 s compute step: the offload is a **capacity** lever for a model
that does not otherwise fit, never a throughput one. (The b12/§4 rows the
offload refused — `exceeded by 0.2739 GiB` at 4 GiB, `1.0137 GiB` at 4 GiB for
the KDA arm — are the budget doing its job: a partial park is not the
mechanism.)

---

## 5. Quality: segmentation is a memory decision, and the loss proves it

**What the repo already gates** (`tests/test_longctx.py`, all CPU, all green in
this tree):

1. `segment_len=0` is **bit-identical** to the one-shot forward (`torch.equal`
   on logits and loss, and the segmented branch is not entered).
2. `segment_len=t` (one segment) is **bit-identical** to the one-shot forward.
3. Segmented logits == one-shot logits at **1e-4 absolute**, on a 64-byte window
   at 4 and 2 segments.
4. The carried state is **load-bearing**: blanking segment 1 changes its logits
   when the state is carried vs when it is dropped (1e-3), and a segment-0-only
   change moves segment 1's logits through the state (1e-4).
5. Segment 0 receives a non-zero gradient, and the loss moves when its bytes
   move (1e-6).
6. The CSA ring pair (cuts inside a chunk) is equal at 1e-5 on logits
   (`tests/test_segment_csa.py`: 1.281e-6 and 1.550e-6 abs on its two
   coarse rows, **Δloss = 0** on both).
7. The trainer passes `train.segment_len` and the counters are exact.

**The gap is the scalar's tolerance**: the load-bearing number is the LOSS (the
trainer, the eval and every BPB read it), and the only pin was 1e-4. Measured
here on the same fixture (CPU, fp32, seed 0, 64-byte window, batch 2):

| segments | `max|Δlogit|` | relative to 5.615 logit scale | `|Δloss|` |
|---:|---:|---:|---:|
| 4 (`segment_len=16`) | 1.155e-6 | 2.06e-7 | **0.0** |
| 2 (`segment_len=32`) | 1.073e-6 | 1.91e-7 | **4.768e-7** |

So **the loss is identical to 1e-6** (0 and 4.8e-7) and **the logits are not
identical to 1e-6 absolute** (1.16e-6) — the logit delta is reassociation
round-off (BLAS order), 2e-7 relative, and tightening that gate would make a
flake generator, not an invariant. `tests/test_segment_parity_strict.py` adds
exactly the missing pin (`|Δloss| <= 1e-6` at 2 and 4 segments, with the
`loop_segments` counter asserted so a silently un-entered branch cannot pass).

---

## 6. Recommendation for the 1B plan

**The 1B target is the KDA loop form** (`configs/b1.toml`: 1 066 358 209 params,
`max_iter=4`, MoE bank 1024×top-2, Engram), so the levers that exist are the
KDA ones:

1. **Ship `activation_recompute="full"`.** Measured on a 95M KDA:
   **−65 % peak for +11 % step time** (+7…16 % across the three runs), exact
   values (1e-6 gates in `tests/test_actmem.py` / `tests/test_vram_arms.py`). The
   ledger's 1B arithmetic prices it at **11.798 GiB** of the 1B's activation
   branch and raises the max batch from **1 → 13** [C]. Its TIME model says
   **+33.3 %** ("3 forward units → 4", `docs/reports/2026-11-06-vram-ledger.md:226`);
   the 95M measurement is **+11 %**, so the ledger is conservative by ~3× — a 1B
   ETA built on it will be pessimistic, not optimistic, which is the right
   direction. There is no second candidate close to it (`selective` saves a
   quarter for +6.7 %, and `segment_len` is for windows longer than one forward,
   not batch).
2. **Keep `muon_state_dtype="bf16"` on** (it is already in the fast recipe):
   **−1.875 GiB** [C] at no measurable time cost (0.99×; the clean runs measure
   277.4 and 279.9 ms against a 282/273 ms base, and the checkpoint+bf16 combo
   gives another **−0.179 GiB** for +5.6 % over the checkpoint alone).
3. **Do not plan batches around `segment_len`.** It is a long-context enabler
   (a window longer than one forward), it costs +53…+142 % at 95M, and its
   saving is `1/N` of the window while `full` is `1/12` of the depth. At
   `s1024` it is strictly worse than the checkpoint.
4. **If the 1B lands on the dense transformer form** (`flagship3_dense` scaled
   up — the form the 95M control in this report uses), **the checkpoint does not
   exist there**. Two honest options: (a) port
   `activation_recompute` into `TransformerBlock._block` — wrap the block body in
   `checkpoint(use_reentrant=False)` and thread the mode through
   `configure_actmem` (the KDA wiring is the template; ~40 lines + one gate),
   then the same −60 %…−70 % applies; or (b) pay the offload's **×4.5** time for
   **−50 %**. (a) is the only one that makes a 1B dense step trainable at a
   useful batch.
5. **The next lever on the dense form is the fp32 residual stream**, not a new
   checkpoint: `residual_dtype="bf16"` is already a field and is **inert** on
   `TransformerBlock` (kernel audit C7). Wiring it changes the biggest single
   save class (fp32 `[b,t,d]` × 27) and is cheaper than any recompute. It does
   move numerics, so it needs the A/B the ledger's `[C]` can only price.

**Where the numbers come from, and their error bars.** The 95M multipliers are
[M] (one run, one card, one shape, median of 7). The 1B GiB are the 2026-11-06
ledger's arithmetic [C], which is **−9.4 %** from its own measured anchor
(`docs/reports/2026-11-06-vram-ledger.md` §2.3), so read them as ±10 %, not to
the third digit. The 1B has never been run on this card (the preset says so).

---

## 7. What this does not measure

1. **`s8192` (the long-context ladder).** `segment_len` exists for windows
   longer than one forward; this file measures it at `s1024`, where the
   checkpoint is the better lever. The 8k/32k curve is the next run.
2. **The PCIe overlap.** The offload's 6.3 GiB/s (~6.8 GB/s) is a round-trip
   rate measured against `torch.cuda.synchronize()`; whether the D2H copy
   overlaps the next block's compute (SlackSwap's claim) needs a timeline, not a
   wall clock.
3. **`residual_dtype=bf16` on the dense form** — it is inert, so there is
   nothing to measure until it is wired.
4. **A real data-loader/trainer step** (the harness's step is the model +
   optimizer only; the firewall/clip/data are ≤ a few hundred MB at 95M).
5. **`nvidia-smi` residency at the peak.** The harness records
   `max_memory_allocated` (the allocator's own peak), `max_memory_reserved` and
   `mem_get_info`; a 2 Hz `nvidia-smi` sampler would close the last ~1 GiB and
   is not in this run.

---

## 8. Reproduction

```bash
cd /home/sehaxe/dormouse/port
# cell 1 — the dense map + arm table + batch ladder (one card acquisition, ~4 min)
GPU_TIMEOUT=2400 bash scripts/gpu_lock.sh .venv/bin/python scripts/bench_activation_vram.py \
  --preset configs/flagship3_dense.toml \
  --arms base,segment512,ckpt_full,ckpt_selective,iter_ckpt,muonstate_bf16,offload4,offload8 \
  --batch 8 --seq 1024 --batches 8,16,20,24,28,32,36 --ladder-arms base \
  --json /home/sehaxe/logs/vram/f3_b8s1024.json
# cell 2 — the KDA lever table (offload LAST: its refusal is a row, not a stop)
GPU_TIMEOUT=2400 bash scripts/gpu_lock.sh .venv/bin/python scripts/bench_activation_vram.py \
  --preset configs/arch_hybrid_kda.toml \
  --arms base,segment512,segment256,ckpt_full,ckpt_selective,iter_ckpt,ckpt_full+bf16state,offload4 \
  --batch 8 --seq 1024 --skip-map --skip-ladder \
  --json /home/sehaxe/logs/vram/kda95_b8s1024.json
# cell 3 — the two combos the ladder cannot reach
GPU_TIMEOUT=2400 bash scripts/gpu_lock.sh .venv/bin/python scripts/bench_activation_vram.py \
  --preset configs/arch_hybrid_kda.toml --arms segment256,ckpt_full+offload4 \
  --batch 8 --seq 1024 --skip-map --skip-ladder \
  --json /home/sehaxe/logs/vram/kda95_extra.json
# the quality gate (CPU, no card)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_longctx.py tests/test_segment_parity_strict.py tests/test_segment_csa.py -q
```

**The instrument.** `port/scripts/bench_activation_vram.py` — **new, untracked**
(this report's only code deliverable besides one test). It builds the trainer's
own model + `MixOptimizer` from a preset, applies one arm's overrides, and
records per cell: `peak_gib` (allocator peak, minus the `residual_gib` a previous
refused arm may have pinned — printed in the row so a leak is visible instead of
silent), median/min ms, kB/s, `saved_gib` and the top saved shapes with the
allocating frame, the forward-end live set by frame, and the trainer's arm
counters. It deliberately does **not** replace `scripts/bench_vram.py` (the four
trainer arms) or `scripts/probe_vram.py` (the real trainer's batch curve); if the
map is wanted long-term it belongs in `bench_vram.py` as a `--map` mode, not as
a third script.

**The gate I ran.** `tests/test_actmem.py` (the two activation arms' exactness
and refusals), `tests/test_vram_arms.py` (the three recompute modes) and
`tests/test_vram_ledger.py` are the ledger's own gates and are unchanged here;
the ones this report touches and ran to completion are
`tests/test_segment_parity_strict.py`, `tests/test_longctx.py` and
`tests/test_segment_csa.py` — **42 passed, 1 skipped** (the CUDA+FLA gate), plus
the two new loss pins. The full 1993-test suite was **not** run to completion:
the CPU-only run outlived the lane's window and was interrupted; no `src/` file
was modified by this task, so the risk it would have covered is the repo's
existing state, not this change.

**The artifacts, and what each is trustworthy for.**

| file | what it is | trust |
|---|---|---|
| `f3_b8s1024.json` | the clean dense cell: 8 arms + the b8..b32 ladder + the map's ERROR | **everything but `map`**; the map computed its tables and then raised `KeyError: 'trace_peak_gib'` in its own printout (a stale key in `main`), so only the error string survives |
| `kda95_b8s1024.json` | the clean KDA cell: 8 arms (no map, no ladder) | clean; `residual_gib=0.062` on every arm after the first |
| `kda95_extra.json` | `segment256` + `ckpt_full+offload4` | clean (no refusal before them) |
| `run.log` | every attempt's console | the first map's `saved 3.043 GiB in 391 tensors`, and the rows of the runs whose JSON was lost |
| `*.polluted*.json` | four runs (dense ×2, KDA ×2) that leaked a constant **+4.114 GiB** after a map/refusal raised | **deltas and timings only**; every absolute peak in them is +4.114 (e.g. base 8.654 instead of 4.547) |

**The harness defect, for the record.** A Python exception caught in the harness
pinned its frame — and that frame holds the model, the loss and the whole
autograd save set — so every arm that ran AFTER a refused/failed one inherited
**+4.1 GiB**. `e.__traceback__ = None` + `gc.collect()` + `empty_cache()` did
**not** release it in this torch build; sampling `memory_allocated()` before the
build and subtracting it DID (and the row prints the subtracted value, so a leak
is visible instead of silent). Any future arm harness should do that and run
refusing arms last. The `*.polluted*.json` files are evidence, not data.
