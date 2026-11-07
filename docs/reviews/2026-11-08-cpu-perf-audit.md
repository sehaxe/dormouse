# 2026-11-08 — CPU-perf audit: the Python side of the step is measured, and it is free

Read-only audit of `port/src/dormouse/`: what the CPU does per training step, per
eval and per checkpoint, and what it costs. **No GPU work was run** (a live
training run holds the card); every number below is either CPU-measured, read out
of this repo's own logs, or arithmetic on those two, and is tagged.

**Question asked:** the step is 136 ms at b=8 s=1024 (model 104 / optimizer 32,
per `port/scripts/batch_sweep.sh:4-6`). Everything outside the kernels is
unmeasured. What is it, and what does it cost?

**Short verdict.** The Python side of the step is **~2.2 ms of pure-Python work
plus 5–11 ms of aten dispatch per step (6–10 % of 136 ms; ≤25 ms contended), and it does not appear
on the wall clock at all** — the non-report step takes zero host reads
(`[M]`, now verified on the *production* architecture, not only on the kda
fixture the gate uses), so the CPU runs ahead and its work is buried under the
GPU's 136 ms. There is **no second hot loop to find**. The only two CPU items
that actually cost wall-clock seconds per run are the **cadence policies**
(`ckpt_every`, `eval_every`): 2.1 s per checkpoint (1.4 s on the bf16 state
rung) and ~0.95 s per eval window.
The one real waste found in the step path is the data reader, `_windows`'s
`torch.tensor(list(raw))` — 0.74 ms/batch at b=8, 6.0 ms at b=32, a 9–18×
fix (40× at b=32) — and it is currently *hidden*, so it is insurance, not a win.

Machine during the audit: 20 cores, load avg 25–51 (other agents' `rustc`
builds), CPU pressure `some` 5.5–54 %, 10–28 GB RAM available, `/home` 91 % full
(42 GB free). Every run: `nice -n 19 ionice -c3 taskset -c 0,1`,
`torch.set_num_threads(2)`, ≤60 s. Absolute CPU times are therefore inflated
~1.5–2× against a quiet box; counts, ratios and byte sizes are not.

---

## 0. Result first: ranked by seconds per hour of training

Baseline = the live E1 config (`arch_dense_byteflow8`, b=32 s=1024,
127.4 ms/step, 28 346 steps/h `[M]` from `~/logs/e1v4/patch_s1.log`, 257 234 tok/s)
and the b=8 premise (136 ms/step, 26 471 steps/h).

| # | item | s/h at the named cadence | s/h at schema defaults | status |
|---|---|---:|---:|---|
| 1 | **checkpoint write** (`ckpt_every=5000`) | **11** | **~500** (`ckpt_every=100`) | real, serial, not overlapped |
| 2 | **eval window** (`eval_every=1907`) | **14** | **~500** (`eval_every=50`) | real, serial, not overlapped |
| 3 | data reader (`_windows`) | 0 (hidden) | 0 (hidden) | **9–40× fixable**, buy insurance |
| 4 | module-tree walks (`parameters()`/`modules()`) | 0 | 0 | measured 0.8 ms/step, hidden |
| 5 | counters / invalidation / `set_lr` / `zero_grad` | 0 | 0 | measured ≤0.15 ms/step, hidden |
| 6 | firewall's 117 `nan_to_num_` launches | 0 | 0 | no `_foreach` form exists in torch 2.13 |

The current E1 run sets `ckpt_every=1000000`, so item 1 is **0 s/h** there and
item 2 is the only real per-run CPU cost: ~14 s/h. The default values are
hazards for anyone launching a 95 M run without the script's overrides, and
switching either default is a **policy** decision (a checkpoint cadence sets the
crash-loss window) that needs its own test, not an A/B of the step.

---

## 1. Host syncs and scalar reads, per file and line

Census: AST walk over the step path plus a live run of the repo's own
`stepaudit` (`port/src/dormouse/stepaudit.py`) on the **production
architecture** (byteflow8 codec + 12 dense `TransformerBlock`s). The gate
`port/tests/test_step_audit.py:141` only exercises the tiny `kda`-loop fixture,
so this is the first run of the 0-sync check on the path E1 actually trains.

**`[M]` result, 4 steps, `log_every=2`:**

```
step 1 (non-report)  reads=0 (device 0)  allocs=229  req=3.1M
step 2 (report)      reads=2 (device 0)  allocs= 36  req=771K   ← drain + _host_scalar
step 3 (non-report)  reads=0 (device 0)  allocs= 32  req=771K
step 4 (report)      reads=2 (device 0)  allocs= 36  req=771K
```

The claim in the module docstring is **true on the production path**, and true
a fortiori on CUDA: the CPU run found *zero* reads even including the
`torch.optim` CPU fallback the audit excuses (this model never takes it).

| path | reads | file:line | note |
|---|---:|---|---|
| training step, non-report | **0** | — | `[M]` above; `.item/.tolist/.cpu/float()/int()/bool()` all absent |
| report step | **2** | `instrument.py:127` (one `.tolist()` for the whole window), `loop.py:77` (`float(value)`), called at `loop.py:2027` | batched drain; one sync, not N (`absorb` at `loop.py:2018` avoids the second transfer) |
| eval, per batch | **2** (flat) / **3** (masked) | `model.py:986` (`float(loss.detach())` when `breakdown=True`), `loop.py:953` (`float(loss)`), `loop.py:949` (`int(mask.sum())`) | `[C]` 24 batch drains per eval (the second read finds the queue empty) |
| eval, other | 0 | `loop.py:947` `int(x.numel())` is shape metadata, not a device read | |
| checkpoint | one blocking D2H per storage | `loop.py:465` (`torch.save`) | `[C]`: `torch.save` copies each CUDA storage to a host buffer synchronously; 762 MB ≈ 30–60 ms at 12–25 GB/s, plus the 2.1 s host serialize |
| logging (`mtp=`, `halt=`, `mod=`) | 1 each, report cadence | `loop.py:127`, `:140`, `:152` | all `None`/0 for this config (`mtp_weight=0`, `mod_aux_weight=0`) |
| grad-norm / clip | **0** | `optim.py:2337`, `:2402` | device-only `_foreach_norm` + `_foreach_mul_` |
| `mix.sf` (schedule-free Muon) | **1 per step** | `optim.py:2052` `float(flag)` | the one declared exception, `muon_sf` only; off by default and off here |
| `engram_ram` host tables | 1 per step | `loop.py:1994` `_host_scalar(flag_dev)` | arm off in this config |
| MoE naive dispatch | 1 per forward | `experts.py:576`, `:583` `row.tolist()` | arm off (`n_experts=1`, `moe_topk=0`) |

```python
# the only two reads a report step takes, both inside the declared cadence
drained = window.drain_if_due(step + 1, cfg.log_every)   # loop.py:2014
last_loss = _host_scalar(last_loss_dev)                  # loop.py:2027
```

One broken invariant found here: `stepaudit` reports **`noop_contig=2` per
step** on this architecture, while `port/tests/test_step_audit.py:229` asserts
`== 0`. The site is `data.py:86` — with `batch=1` both `ids[:, :-1]` and
`ids[:, 1:]` satisfy PyTorch's contiguity test (size-1 dims ignored), so the two
`.contiguous()` calls are no-ops. At b>1 they are real copies. Cost: ~2 µs;
the gate's invariant is batch-dependent, which is the finding, not the time.

---

## 2. Per-step Python overhead

### 2.1 Call and op census

`[M]`, clean steps (no report, no eval, no ckpt), cProfile + a
`TorchDispatchMode` counter, on a d=64/12-block twin that runs the **same code
path** as production:

| metric | value | where |
|---|---:|---|
| Python calls / step | **12 500–17 500** | run-to-run spread is generator resumption counting |
| … of them in `port/src/dormouse/` | **~934** | 715 `train/optim.py`, 147 `model/`, 34 `stepaudit`, 28 `instrument`, 6 `train/data`, 4 `train/loop` |
| … the rest | torch's own per-op frames | 2 688 aten ops × 2–4 Python frames |
| **aten ops / step** | **2 688** | `[M]` dispatch mode |
| allocations / step | 32–36, 771 KB requested | `[M]` stepaudit, tiny twin |

Top of the op census (per step): `view` 285, `t` 270, `mul.Tensor` 230, `mm` 192,
`add.Tensor` 184, `detach` 159, `transpose` 144, **`nan_to_num_` 117** (the
firewall, one per gradient), `slice` 116, `bmm` 90 (the Muon NS chain),
`zeros_like` 63, `_to_copy` 48, `cat` 37, `linalg_vector_norm` 29.
2 688 ops × 2–4 µs/op = **5.4–10.8 ms of host dispatch per step** at a typical
dispatch cost; the per-op costs this loaded box measured are 0.9–9.3 µs
(`detach` 0.88, `add` 2.94, `t` 4.02, `view` 4.29, `mul` 7.95, `where` 9.31), so
a fully contended host could pay up to ~25 ms.

### 2.2 Host-only work, measured at production size (97 584 896 params)

| site | what | `[M]` med | p95 |
|---|---|---:|---:|
| `data.py:393` `_read` | mmap slice + ring append + `bytes()` | 0.002 ms | — |
| `data.py:77-86` `next_batch` | `torch.tensor(list(raw), long)` + 2 copies | **0.74 ms** | 0.92 ms |
| `experts.py:3717` `invalidate_banks_all` | `model.modules()` walk | 0.09 ms | 0.23 ms |
| `qat_cache.py:221` `invalidate_qat_cache_all` | 0 cached modules | 0.00 ms | 0.00 ms |
| `model.parameters()` walk | 104 modules / 117 tensors | 0.18 ms ×4–5 | 0.26 ms |
| grads list (`loop.py:1647`, `:1585`) | 2 builds | 0.20 ms ×2 | 0.24 ms |
| `opt.zero_grad(set_to_none=True)` | 117 `.grad = None` | 0.04 ms | 0.08 ms |
| `MixOptimizer.set_lr` | 2 groups | 0.004 ms | — |
| `model.state_dict()` / `opt.state_dict()` | views only, ckpt cadence | 0.40 / 0.04 ms | 0.78 ms |
| `Counters` report step | 9 counters, 1 drain | 0.08 ms | 0.11 ms |
| `_ema_update` host part (`loop.py:865`) | 117 params, 1 dtype group | 0.18 ms + 1.7 ms GPU traffic | — |

**Sum of host-only work: ~2.2 ms/step = 1.6 % of 136 ms.** With dispatch, the
whole host side is **8–13 ms/step (6–10 %)**, ~27 ms on the worst per-op
reading — still 5–10× below the GPU's 136 ms, which is why it does not show.

### 2.3 Where the loop touches each tensor

Per optimizer step, for a 117-tensor model (measured walks, not estimated):

| tensor | host touches / step |
|---|---|
| each parameter | ~11: 4–5 `parameters()` walks (`loop.py:1647`, `:1652`, `:1585`, `_ema_update` when on, `experts.py:3717` for `modules()`), 2 grad-list builds, 3 foreach passes, optimizer state dict lookups |
| each gradient | 3 foreach passes (`grad_norm`, `_gate_grads` mul, `clip_grads` mul) + 1 `nan_to_num_` launch each (`loop.py:90`) |
| `flag_dev` | 5: `fill_`, `copy_`, `1 - flag_dev` ×2 (`loop.py:1944-1945`), multiplicand in `_gate_grads` |
| `loss_ok_dev` | 3: `fill_`, `copy_`, `torch.where` (`loop.py:1929`) |
| the loss scalar | 6: `detach`, `isfinite`, `all`, `where`, `backward`, `detach` for the accumulator |
| small allocations for logging | 5–6 scalar tensors/step: `new_zeros(())` (`loop.py:1894`), `1 - flag_dev` ×2, `gate.to(dtype)`, counter clone |

### 2.4 Top-20 by cumulative time, classified

`[M]` cProfile, 6 steps of the d=64 twin (CPU, so the *seconds* are CPU math and
do **not** transfer; the classification and the call sites do). 2 evals are
inside the window.

| # | function | cum s / step | class |
|---:|---|---:|---|
| 1 | `torch.bmm` | 1.687 | optimizer (bf16 NS on CPU — inflated) |
| 2 | `run_backward` | 0.604 | model |
| 3 | `torch._foreach_mul` | 0.279 | optimizer (NS poly) |
| 4 | `torch.mm` | 0.238 | model + optimizer prescale |
| 5 | `torch._foreach_add` | 0.196 | optimizer |
| 6 | `tsct.py:259 mm_mixed` | 0.164 | model |
| 7 | `Tensor.to` | 0.159 | model (codec casts) + optimizer (bf16→fp32) |
| 8 | `torch.conv1d` | 0.079 | model (codec) |
| 9 | `linalg_vector_norm` | 0.079 | step firewall + clip |
| 10 | `torch.layer_norm` | 0.060 | model |
| 11 | `scaled_dot_product_attention` | 0.060 | model |
| 12 | `Tensor.mul_` | 0.056 | optimizer |
| 13 | `gelu` | 0.055 | model |
| 14 | `sin` (rope) | 0.050 | model |
| 15 | `_foreach_mul_` | 0.045 | step firewall |
| 16 | `Tensor.div_` | 0.036 | optimizer |
| 17 | `Tensor.contiguous` | 0.033 | data + model |
| 18 | `Tensor.copy_` | 0.027 | optimizer (bf16 momentum store) |
| 19 | `cos` (rope) | 0.027 | model |
| 20 | `F.pad` | 0.020 | model (codec convs) |

Data = 0.74 ms/step (0.7 % of the row-1 total on CPU; `_windows` is `#17`'s
neighbour and is 759 µs of it). Logging = 0.08 ms per *report* step. Eval =
43 ms per 2-batch eval on this twin → 0.33–0.44 of a step per eval batch (§3).
The GPU-side split stays the measured one: model 104 ms, optimizer 32 ms.

---

## 3. The eval path

`_eval_loss` (`loop.py:874`) — `stream.rewind()` once, then
`eval_batches_for(cfg)` batches (`data.py:866`), each: `next_batch` → `.to(device)`
→ one `no_grad` forward with `breakdown=True` → **2 host reads** → host-side
accumulation. `evaluate` (`loop.py:981`) is a thin wrapper; `depths=True` runs
the whole window once per depth (`1..max_iter`), i.e. ×T cost.

E1 logs: `over 196608 B (24 x 8 x 1024)`
(`~/logs/e1v4/bytes_s1.log`) — 24 batches at b=8. The live b=32 run uses the
**same 196,608-byte window split into 6 batches** (`6 x 32 x 1024`,
`~/logs/e1v4/patch_s1.log`): same scored bytes, 4× fewer host drains and 6
larger GEMMs, so 6 batches is the cheaper of the two and the b=8 figure below is
the conservative one.

Cost of one eval, two independent estimates:

| source | value |
|---|---|
| `[M]` CPU twin: eval batch = 0.33–0.44 of `fwd+bwd` | 24 × 0.37 × 104 ms = **0.92 s** |
| `[M]` GPU log delta, `~/logs/overnight1.log` (1.07 B model, 8 batches of b8×s512): step 1094.0 → 1105.6 ms across the eval window | 11.6 ms × 100 steps = **1.16 s** per eval; scaled to our 24 batches × 2× bytes × 1/10.9 params → **0.63 s** |
| adopted | **0.8–1.1 s per eval** |

Fraction of a run (step 136 ms at b=8, 127 ms at b=32):

| cadence | eval per run | % of run | s/h |
|---|---:|---:|---:|
| `eval_every=7629` (the b=8 bytes run) | 4 evals / 30 517 steps | 0.08 % | 3 |
| `eval_every=1907` (**live E1**) | 4 / 7 629 | 0.37 % | 14 |
| `eval_every=500` | 15 / 7 629 | 1.4 % | 50 |
| `eval_every=50` (**schema default**) | 153 / 7 629 | **14 %** | 500 |

Per-batch Python overhead is small (2 reads + a dict read of `last_breakdown`);
the cost is the 24 forward passes themselves plus the queue drain each read
forces. **`[C]` the 24 drains per eval are the reason the eval cannot overlap
with the next training step** — this is the one place where a Python read
converts into wall-clock: ~24 × 2–5 ms of ramp-down = 0.05–0.12 s, included in
the 0.8–1.1 s above.

---

## 4. Checkpoint cost

`save_ckpt` (`loop.py:417`), inline in the loop at `loop.py:2110-2115`
**between** steps, so the GPU is idle for its whole duration.

`[M]` anatomy of a real file (`/tmp/opencode/ck/flagship3.pt`, step 5000,
95 035 200 params, loaded with `mmap=True`, nothing materialised):

| part | bytes | per param |
|---|---:|---:|
| `model` (fp32, 99 tensors) | 380 100 000 | 4.00 B |
| `opt.muon` momentum (fp32, 94.6 M params) | 378.3 MB | 3.98 B |
| `opt.adamw` (`exp_avg`+`exp_avg_sq`, 0.47 M params) | 3.8 MB | 0.04 B |
| file total | **762 231 325** | **8.02 B** |

The optimizer state is **50.1 %** of the file — not dominant, but half. On the
live rung (`muon_state_dtype=bf16`) the momentum is 189 MB and the file is
**~573 MB (6.03 B/param)**. With `ema_decay>0` a second model half is written
(`model_ema`, `loop.py:462-463`): **+380 MB (+4.0 B/param)** — 762→1142 MB on
the fp32 rung, 573→953 MB on bf16.

`[M]` serialize cost, synthetic payload of the same sizes, 2 nice'd cores while
the box was loaded:

| payload | time | rate |
|---|---:|---:|
| 566.6 MB (fp32 momentum) | **1.574 s** | 360 MB/s |
| 475 MB (bf16 momentum) | **1.247 s** | 381 MB/s |
| real 762 MB file, `torch.save` from a cold mmap | 3.161 s | 241 MB/s (includes reading 762 MB off disk) |

`[C]` therefore 762 MB ≈ **2.1 s**, bf16 rung ≈ **1.4 s**, plus 30–60 ms of D2H
and the disk write; the `.pt.tmp` + rename (`loop.py:464-466`) is atomic but needs
one extra file's worth of space at the moment of writing.

| cadence | s per checkpoint | % of run | s/h |
|---|---:|---:|---:|
| `ckpt_every=1000000` (E1 throughput runs) | — | 0 | 0 |
| `ckpt_every=5000` (night scripts) | 2.1 | 0.31 % | **11** |
| `ckpt_every=100` (**schema default**, `config.py:1918`) | 2.1 | **13.4 %** | **500** |

On this box `/home` has 42 GB free = 55 fp32 checkpoints (73 on the bf16 rung);
`checkpoints/` already holds 14 GB. The `.tmp`-then-rename needs one extra file's
worth of space at the moment of writing.

---

## 5. Data path

`ByteStream` (`data.py:321`): `mmap` at `data.py:356`, 64 MB ring / 8 MB refill
(`data.py:67-68`), `_read` at `data.py:393` = mmap slice → `bytearray +=` →
`bytes(self._buf[:n])` → `del self._buf[:n]`; `_windows` (`data.py:77`) then
converts to a Python list of ints and a long tensor, and takes two
`.contiguous()` slices.

| step of the path | `[M]` per 8 200 B window (b=8) |
|---|---:|
| `_read` (mmap + ring + `del`) | **0.002 ms** |
| `_windows` (list + tensor + 2 copies) | **0.759 ms** (0.68–1.57 across runs) |
| total `next_batch` | **0.739 ms** med / 0.92 p95 (quiet run); **1.06 ms** med / 1.53 p95 (loaded run) |
| `MixtureStream.source_for_step` (`mixture.py:374`) | 0.0067 ms |

The `del self._buf[:n]` is **not** the O(ring) memmove it looks like — CPython's
bytearray advances an internal offset, measured 1.7 µs. **The whole cost is
`torch.tensor(list(raw), dtype=torch.long)`**: 9 216 PyLong objects (~260 KB of
allocations) and 82 ns per byte. The cheapest fix is one line and removes the
list:

```python
ids = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long().view(batch, seq_len + 1)
return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
```

`[M]` 85 µs vs 1 567 µs at b=8 (**18.4×**), 150 µs vs 6 030 µs at b=32
(**40×**). Saving: **0.66–1.48 ms/step at b=8** (1.1 % of the step on the
unloaded reading, 0.74→0.08 ms) **and ~5.9 ms at b=32 (4.6 %)**. `[M]` end-to-end `mmap + frombuffer` = **81 µs** per batch, i.e. the
whole data path becomes ~0.1 % of a step.

**Flatness across batch sizes — the sanity check in the brief.** Strictly, it
cannot be run today: there are no same-config multi-batch logs
(`/home/sehaxe/logs/batch/` is empty; `port/scripts/batch_sweep.sh` was never
executed), and the GPU is held. What the existing logs say:
`~/logs/e1v4/bytes_s1.log` = 51 373 tok/s → **50.2 kB/s** (b=8,
`flagship3_dense`); `~/logs/e1v4/patch_s1.log` = 257 234 tok/s → **251 kB/s**
(b=32, `arch_dense_byteflow8`). Different presets, so **not a batch A/B** —
but 4× the batch buying 5.0× the bytes/s is strictly superlinear, so no fixed
loader ceiling exists at these rates. And the loader cannot be the bottleneck by
construction: 0.74 ms of a 136 ms step.

**Cheaper still (`[S]`, not measured):** the loop copies `x`/`y` to the device as
int64 (`[8,1024]` = 65 KB/step). Sending `uint8` (8 KB) and casting on the
device would cut the host-side bytes 8×. The transfer is ~3 µs today, so this is
not worth a change without a reason.

---

## 6. Allocation churn: full-size temporaries per step

| site | temporary | size/step (b=8) | cheapest change |
|---|---|---:|---|
| `data.py:85` | `list(raw)` of 9 216 PyLong + `[8,1025]` int64 | ~260 KB + 74 KB | `frombuffer` (above), −0.66 ms |
| `convcodec.py:311-312` | identity `gather` when `k == p`: `[8,128,2048]` fp32 | **8.4 MB** | `if k == p: logits = u.reshape(b,t,VOCAB) + …` else gather (bit-identical: `chunk` is all-zeros) |
| `model.py:1517` | `log_softmax` buffer `[8192,256]` fp32 behind `gather(1)` | **8.4 MB** | `F.cross_entropy` — same math, **not bit-identical**, needs an A/B |
| `optim.py:852` | bf16 momentum → fp32 chunk temp | 2–3 chunks, ≤265 MB at a time | none; the fp32 arithmetic is the rung's contract |
| `model.py:1405-1406` | `h.to(float32)` + `reshape.to(float32)` | 3.4 MB | **not on this config's path** (kda/transformer form only) |
| `loop.py:1894` etc. | 5–6 scalar tensors (`new_zeros`, `1-flag_dev` ×2, `gate.to`, counter clone) | ~40 B | reuse one buffer; ~0 s |

Only the codec gather is structural, and it is GPU-side memory traffic
(8.4 MB/step at b=8, 67 MB at b=32), not CPU.

---

## 7. Null results — suspected hot spots that are free

Stated explicitly because a null result is a result:

* **The 117 per-tensor `nan_to_num_` calls are free on the host** and cannot be
  batched: torch 2.13 has no `_foreach_nan_to_num_` (checked; only the non-list
  `nan_to_num_` exists). Cost is GPU launch+bandwidth, hidden.
* **Module-tree walks** (`parameters()` 4–5×, `modules()` 1× per step): 0.6–0.9 ms
  measured, hidden. Optimising them would buy 0 s/h.
* **`invalidate_banks_all` / `invalidate_qat_cache_all`**: 0.09 + 0.00 ms/step.
  The QAT call visits an empty list every step (`qat.invalidate=1x0` in the
  audit) — one attribute lookup, free.
* **`set_lr`, `zero_grad`, counter merge**: ≤0.05 ms each.
* **`_ema_update`** (arm off here): host part 0.18 ms + 1.7 ms of GPU traffic
  `[C at 672 GB/s]` = 1.4 % of the step if switched on; the per-dtype `_foreach_lerp_`
  is already the fused form.
* **The bf16 momentum rung is not cheaper in traffic** (read bf16 → fp32 temp →
  mul → add → cast back ≈ 4 passes vs 2 in place); it is a memory lever, as
  documented. `[M]` 285 vs 283 ms/step, `docs/architecture/2026-11-08-the-16x.md`.
* **`noop_contig=2`**: the invariant is broken but the cost is ~2 µs (§1).

---

## 8. What to do, in order (and what needs an A/B first)

1. **Data reader, one line** (`data.py:85`): `torch.frombuffer(bytearray(raw),
   torch.uint8).long()`. Zero semantic risk (same bytes, same int64 view, same
   target shift), 9–40× on the function, 0.66–5.9 ms/step. It changes no
   default and no number. **Expected wall-clock saving today: 0 s/h** (hidden);
   it buys headroom, and it is the difference between 5 % and 0.4 % of the step
   when the step gets faster.
2. **Checkpoint cadence**: the current scripts (5000 / 1e6) are fine (11 / 0 s/h).
   The schema default `ckpt_every=100` costs **13 % of a 95 M run**. Changing it
   is a crash-loss policy decision — it needs a test of the recovery window, not
   an A/B of the step. Do not change it silently.
3. **Eval cadence**: same picture, `eval_every=1907` costs 14 s/h, the default 50
   costs 14 %. `eval_depths` multiplies it by `max_iter`.
4. **Codec identity gather** (`convcodec.py:311`): remove the `[b,p,2048]` copy
   when `k == p`. Bit-identical; GPU-side only; 8.4 MB/step at b=8.
5. **Do not** touch the firewall, the counters, the invalidation calls or the
   optimizer's host loops: all measured at ≤0.1 ms/step and hidden.
6. **Do not** switch `_byte_ce` to `F.cross_entropy` without an A/B: it removes
   an 8.4 MB buffer but is not bit-identical to the `log_softmax + gather`
   arithmetic every number on record was produced by.

---

## 9. Reproduction, and what is not measured

Harnesses: `/tmp/opencode/pa_{census,ckpt,ckpt2,micro1,micro2,profile,components2,audit,last,data2,calls}.py`
(python = `port/.venv/bin/python`, `PYTHONPATH=port/src`, always
`torch.set_num_threads(2)` and `taskset -c 0,1`). The production-size model is
built from `port/configs/arch_dense_byteflow8.toml` with
`model_overrides=dict(bf16=False)` (CPU); the twin for ratios is the same preset
at `d_model=64, n_heads=1, head_dim=64, d_ffn=128, n_layers=12`.

Not measured, and why:

* **GPU wall-clock for eval** — the card is held; the number is a CPU ratio
  (0.33–0.44 of a step per eval batch) cross-checked against a log delta from a
  different preset. Both agree within 1.7×, hence the 0.8–1.1 s band.
* **D2H during `torch.save`** — `[C]` from PCIe bandwidth; not executed.
* **Batch-scaling of the loader on the card** — no logs exist and the GPU is
  unavailable; the CPU cost is linear in bytes (82 ns/B), which is the whole
  claim.
* **Any CUDA launch/overlap behaviour** — inferred from the 0-sync result plus
  the 136 ms vs 8–27 ms host budget; `[C]`, not measured on the device.

The one thing this audit could not do is prove the *overlap* directly (a CUDA
event timeline). The 0-sync census plus the 5–10× host/GPU budget margin is the
evidence; if the step is ever re-instrumented, a `torch.cuda` timeline on a
50-step run would close it in one line.
