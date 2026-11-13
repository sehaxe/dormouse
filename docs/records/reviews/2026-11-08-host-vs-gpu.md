# 2026-11-08 — host vs GPU: the launch census and the CPU/GPU split of one step

Follow-up to [`2026-11-08-kernel-audit.md`](2026-11-08-kernel-audit.md), which
accounts for 58 % of a `flagship3_dense` step as arithmetic and leaves 42 %
unnamed. This report measures the other side of the question: **is the step
limited by the GPU or by the host, and what is the 42 %?**

Marks: **[M]** measured on the card with the instrument named · **[C]** computed
from measured inputs, arithmetic shown · **[S]** inference with the check that
settles it.

Instrument: `port/scripts/host_vs_gpu.py` (new, four modes; every command below
is reproducible). Artifacts: `/tmp/opencode/hvg/{phases,census,loader,sync}.json`
and `census_trace.json` (122 MB chrome trace, 20 recorded steps).

Companion, measured the same day on the same shape:
[`2026-11-08-pointwise-fusion.md`](2026-11-08-pointwise-fusion.md) fuses the
largest pointwise chain and A/Bs it. Where the two touch — how much of the
pointwise bill is removable — **that file's A/B wins over this file's kernel
totals**, and §4/§6.2 are written against its result.

---

## 0. Verdict (RU)

**Шаг ограничен GPU, а не хостом — но запас у хоста почти нулевой, и его
съедает один настоящий дефект.**

1. **GPU занят 149.2 мс из 157.2 мс шага (94.9 %)**: 1741 ядро на шаг, суммарный
   простой карты — 8.1 мс (5 %). Этот простой создаёт хост, который на
   загруженной машине (`load 42`) еле успевает за картой.
2. **Хост тратит 156.4 мс на шаг, из них 28.7–44 мс — это блокировка** на трёх
   `cudaStreamSynchronize` внутри `opt.step()`: `torch.as_tensor(1.0 - lr*wd,
   device=cuda)` (`train/optim.py:2165`, `:2329`). Это единственная
   *структурная* проблема хоста, и она не зависит от загрузки CPU.
3. **42 % — это не launch overhead и не Python.** Это 40.6 мс memory-bound
   pointwise (касты bf16↔fp32, add/mul остатка, RoPE, LN) + 14.3 мс мелких ядер
   оптимизатора/норм + ~8 мс простоя GPU, вызванного хостом. Ядра-GEMM идут на
   **90 % пика** (79.7 TFLOP/s из 88.7), NS — на **98 %** (86.9 TFLOP/s):
   арифметику ускорять нечем. Но и «убрать» этот пул слиянием нельзя: байты идут
   из L2 (48 МБ), и замеренное слияние самой большой цепочки (residual+LN+каст)
   даёт **+1.48 % шага**, а не 15 % — `2026-11-08-pointwise-fusion.md` §0.2.
4. **Загрузчик не при чём**: `MixtureStream.next_batch` = 0.169 мс CPU, а
   инъекция +5/+20/+40 мс в него стоит +9.4/+23.5/+41.9 мс шага — запас хоста
   сегодня почти исчерпан, и всё же сам загрузчик в него влезает с запасом.
   Prefetch-поток вернёт 0.169 мс и 0 мс wall.

**EN, one line:** 95 % of the step is kernels; the 42 % is bytes (pointwise) and
small optimizer kernels, not launches; the host's only real defect is three
blocking scalar copies inside the AdamW write-back.

---

## 1. What was measured, on what

| | |
|---|---|
| preset | `configs/flagship3_dense.toml` — 95 035 200 params, 12 dense `TransformerBlock`s, d 832 / ffn 3072, b **8** × s **1024** (8 192 B/step) |
| recipe | lane C: `opt=mix`, `muon_lr_mult=12`, `ns_dtype=bf16`, `muon_state_dtype=bf16`, `ns_steps=5`, `mix_refetch`, `doc_masking=false` |
| pinned away | `eval=false`, `ckpt_every=1e6`, `log_every=1000` — no eval, checkpoint or report step lands in a window |
| card | RTX 5070 Ti 16 GB, driver 615.78.08, torch 2.13.0+cu129, exclusive (`gpu_lock.sh`) |
| host | 20 cores, **load average 39.5–42.6**, ~20 other agents' CPU jobs running; a tiny aten op measured **19.7 µs** on this box against the 0.9–9.3 µs the CPU-perf audit measured [M]. Every host number below is an **upper bound** inflated ~5×. |
| step wall (median of 25 clean steps) | **157.25 ms** (p95 176.0) — the run under measurement |
| same-shape audit for comparison | 145.8 ms (`kernel-audit` §3, quiet box) |

Protocol per mode (in the script's docstring): `phases` = 30 steps, 25 kept, no
profiler; `loader` = 4 × 15 steps with injected CPU work; `sync` =
`torch.cuda.set_sync_debug_mode("warn")` for 3 steps; `census` = **20 warm + 20
recorded** steps under `torch.profiler`, started after a `synchronize()` at a step
boundary and drained before the last `p.step()`, so the trace contains exactly 20
steps of kernels and no leakage.

---

## 2. Launch census (20 recorded steps, b8 × s1024)

| metric | value |
|---|---:|
| executed kernels / step | **1 741.0** |
| host launches / step | **1 741.0** — 1 534 `cudaLaunchKernel` + **207 `cuLaunchKernel`** (cuBLAS/cutlass go through the driver API; a runtime-API-only count under-reports by exactly the GEMM count) |
| GPU busy (Σ kernel duration) / step | **149.17 ms** |
| wall, unprofiled / step | 157.25 ms → **GPU 94.9 % busy**, 8.1 ms idle |
| streams with kernels | **1** (stream 7); no second stream anywhere in the step |
| memcpy / step | 6 `cudaMemcpyAsync` calls (3 scalar + 2 window + 1 in `aten::all`) → 3 HtoD **pageable** (0.009 ms) + 3 DtoD (0.003 ms) on the device, + 10 memset (0.009 ms) |
| host syncs / step (trace) | **3 `cudaStreamSynchronize`** + 0.1 `cudaDeviceSynchronize` |
| kernels ≤ 20 µs | **884 / step (50.8 % of launches) = 4.7 % of GPU time** |

The last row is the census' main answer to "cost of a launch": **half the
launches cost 7.0 ms of a 149 ms bill.** The launch count is not where the step
goes. **81 % of the GPU time is in 354 kernels ≥ 100 µs.**

### 2.1 Top 30 by count (per step)

Labels are decoded from the mangled signature; the full names are in
`/tmp/opencode/hvg/census2.json` (`by_count`), and `[M]` is the 20-step mean.

| # | kernel (decoded) | n / step | ms / step | µs each |
|---:|---|---:|---:|---:|
| 1 | `elementwise<128,4>` BF16 **mul** (`BinaryFunctor<BFloat16,MulFunctor>`) | 192.0 | 3.316 | 17.3 |
| 2 | `vectorized<4>` BF16 **add** (`CUDAFunctor_add<BFloat16>`) | 160.0 | 5.649 | 35.3 |
| 3 | `vectorized<4>` **cast → bf16** (`bfloat16_copy_kernel`) | 131.0 | 3.001 | 22.9 |
| 4 | `unrolled_elementwise` **cast-copy** (LoadWithCast/StoreWithCast) | 117.0 | 4.638 | 39.6 |
| 5 | `elementwise<128,2>` fp32 **mul** | 115.0 | 1.945 | 16.9 |
| 6 | `vectorized<4>` **nan_to_num** | 99.0 | 0.898 | 9.1 |
| 7 | `elementwise<128,4>` **cast-copy → bf16** | 98.0 | 3.600 | 36.7 |
| 8 | `vectorized<4>` BF16 **mul (scalar)** (`AUnaryFunctor`) | 60.0 | 3.714 | 61.9 |
| 9 | `vectorized<4>` fp32 **add** | 49.0 | 3.500 | 71.4 |
| 10 | cutlass bf16 GEMM `128x128_32x4_nt` | 49.0 | **21.669** | 442.2 |
| 11 | cutlass bf16 GEMM `256x128_32x3_tn` | 48.0 | **19.903** | 414.6 |
| 12 | cutlass bf16 GEMM `256x128_32x3_nn` | 48.0 | **21.386** | 445.5 |
| 13 | `vectorized<4>` BF16 **fill** | 48.0 | 0.440 | 9.2 |
| 14 | cutlass bf16 GEMM `64x64_32x6_nn` | 41.0 | **12.400** | 302.4 |
| 15 | `vectorized_layer_norm` | 24.0 | 0.951 | 39.6 |
| 16 | `arange` | 24.0 | 0.024 | 1.0 |
| 17 | `CatArrayBatchedCopy` (RoPE `cat`) | 24.0 | 1.303 | 54.3 |
| 18 | `layer_norm_grad_input` | 24.0 | 2.266 | 94.4 |
| 19 | `GammaBetaBackward` | 24.0 | 1.801 | 75.0 |
| 20 | `elementwise<128,4>` BF16 **neg** | 24.0 | 0.465 | 19.4 |
| 21 | `vectorized<4>` fp32 **mul (scalar)** | 21.0 | 1.827 | 87.0 |
| 22 | cutlass bf16 GEMM `64x64_32x6_tn` | 21.0 | **8.996** | 428.4 |
| 23 | `multi_tensor_apply` binary-add (optimizer) | 14.0 | 3.125 | 223.2 |
| 24 | `reciprocal` | 13.0 | 0.017 | 1.3 |
| 25 | `CUDAFunctorOnSelf_add<float>` | 13.0 | 0.017 | 1.3 |
| 26 | `BUnaryFunctor` fp32 mul | 12.0 | 0.015 | 1.3 |
| 27 | `FillFunctor<long>` | 12.0 | 0.013 | 1.1 |
| 28 | `pow(tensor,tensor)` | 12.0 | 0.024 | 2.0 |
| 29 | `cos` | 12.0 | 0.017 | 1.4 |
| 30 | `sin` | 12.0 | 0.017 | 1.4 |

### 2.2 Top 30 by CUDA time (per step)

| # | kernel | ms / step | share | n / step | µs each |
|---:|---|---:|---:|---:|---:|
| 1 | cutlass bf16 GEMM `128x128_32x4_nt` | 21.669 | 14.5 % | 49.0 | 442.2 |
| 2 | cutlass bf16 GEMM `256x128_32x3_nn` | 21.386 | 14.3 % | 48.0 | 445.5 |
| 3 | cutlass bf16 GEMM `256x128_32x3_tn` | 19.903 | 13.3 % | 48.0 | 414.6 |
| 4 | cutlass bf16 GEMM `64x64_32x6_nn` | 12.400 | 8.3 % | 41.0 | 302.4 |
| 5 | cutlass bf16 GEMM `64x64_32x6_tn` | 8.996 | 6.0 % | 21.0 | 428.4 |
| 6 | `flash_bwd_dq_dk_dv_loop_seqk_parallel` | 6.462 | 4.3 % | 12.0 | 538.5 |
| 7 | BF16 **add** | 5.649 | 3.8 % | 160.0 | 35.3 |
| 8 | **cast-copy** (unrolled) | 4.638 | 3.1 % | 117.0 | 39.6 |
| 9 | BF16 **mul (scalar)** | 3.714 | 2.5 % | 60.0 | 61.9 |
| 10 | **cast-copy → bf16** (elementwise) | 3.600 | 2.4 % | 98.0 | 36.7 |
| 11 | fp32 **add** | 3.500 | 2.3 % | 49.0 | 71.4 |
| 12 | BF16 **mul** | 3.316 | 2.2 % | 192.0 | 17.3 |
| 13 | `multi_tensor_apply` binary-add | 3.125 | 2.1 % | 14.0 | 223.2 |
| 14 | **cast → bf16** (`bfloat16_copy`) | 3.001 | 2.0 % | 131.0 | 22.9 |
| 15 | fp32 **div** (optimizer) | 2.725 | 1.8 % | 12.0 | 227.1 |
| 16 | `flash_fwd_kernel` | 2.532 | 1.7 % | 12.0 | 211.0 |
| 17 | `layer_norm_grad_input` | 2.266 | 1.5 % | 24.0 | 94.4 |
| 18 | `gelu_backward` | 2.229 | 1.5 % | 12.0 | 185.7 |
| 19 | fp32 **mul** | 1.945 | 1.3 % | 115.0 | 16.9 |
| 20 | fp32 **mul (scalar)** | 1.827 | 1.2 % | 21.0 | 87.0 |
| 21 | `GammaBetaBackward` | 1.801 | 1.2 % | 24.0 | 75.0 |
| 22 | `CatArrayBatchedCopy` (RoPE) | 1.303 | 0.9 % | 24.0 | 54.3 |
| 23 | `gelu` | 1.281 | 0.9 % | 12.0 | 106.8 |
| 24 | `direct_copy` fp32 | 1.118 | 0.7 % | 2.0 | 559.2 |
| 25 | `multi_tensor_apply` mul | 1.090 | 0.7 % | 7.0 | 155.7 |
| 26 | `multi_tensor_apply` LpNorm | 1.004 | 0.7 % | 10.0 | 100.4 |
| 27 | `CatArrayBatchedCopy_vectorized` | 0.981 | 0.7 % | 12.0 | 81.8 |
| 28 | `vectorized_layer_norm` | 0.951 | 0.6 % | 24.0 | 39.6 |
| 29 | `nan_to_num` | 0.898 | 0.6 % | 99.0 | 9.1 |
| 30 | `reduce_kernel` (norm) | 0.862 | 0.6 % | 10.0 | 86.2 |

**Family bill** (sum of all 1741 kernels, 149.17 ms):

| family | kernels / step | ms / step | share | µs / kernel |
|---|---:|---:|---:|---:|
| GEMM (cutlass; model `mm` + Muon NS `bmm`) | 255.0 | **94.34** | 63.2 % | 370 |
| pointwise / cast / copy | 1 298.0 | **40.55** | 27.2 % | 31 |
| multi_tensor (optimizer foreach) | 40.0 | 5.32 | 3.6 % | 133 |
| LayerNorm (fwd + grad_input) | 50.0 | 3.32 | 2.2 % | 66 |
| cat (RoPE) | 40.0 | 2.30 | 1.5 % | 58 |
| reduce / norm | 16.0 | 1.39 | 0.9 % | 87 |
| other (γβ-backward 1.80 + 0.15) | 42.0 | 1.95 | 1.3 % | 46 |
| **total** | **1 741.0** | **149.17** | 100 % | 86 |

### 2.3 Top dispatchers (kernel → CPU op, exact join on `External id`)

Not `key_averages`: every kernel event carries the `External id` of the CPU op
that issued it, so the sum below has **no parent/child double count** (the
profiler's own table counts `aten::mm` inside `aten::matmul` twice). `aten::copy_`
shows fewer kernels than calls because 60 of its calls are the blocking memcpy
(scalars, x/y), not kernels.

| CPU op | calls / step | CUDA ms / step |
|---|---:|---:|
| `aten::mm` | 148.0 | **63.05** |
| `aten::bmm` (Muon NS only — the model's attention is flash) | 60.0 | **21.31** |
| `aten::copy_` (casts, clones, data windows) | 350.0 | 12.36 |
| `aten::_flash_attention_backward` | 36.0 | 7.45 |
| `aten::mul` | 279.0 | 7.06 |
| `aten::add` | 102.0 | 6.61 |
| `aten::native_layer_norm_backward` | 48.0 | 4.07 |
| `aten::mul_` | 111.0 | 3.74 |
| `aten::_foreach_add_` | 15.0 | 3.13 |
| `aten::_flash_attention_forward` | 12.0 | 2.53 |
| `aten::add_` | 98.0 | 2.42 |
| `aten::cat` | 38.0 | 2.29 |
| `aten::gelu_backward` | 12.0 | 2.23 |
| `aten::div_` | 8.0 | 1.83 |
| `aten::linalg_vector_norm` | 14.0 | 1.38 |
| `aten::gelu` | 12.0 | 1.28 |
| `aten::_foreach_mul_` | 8.0 | 1.10 |
| `aten::_foreach_norm` | 12.0 | 1.01 |
| `aten::native_layer_norm` | 24.0 | 0.95 |
| `aten::div` | 17.0 | 0.91 |
| `aten::nan_to_num` | 99.0 | 0.90 |
| `aten::fill_` | 71.0 | 0.47 |
| `aten::neg` | 26.0 | 0.47 |
| the 7 ops below the cut (sub, rms-norm bwd, embedding bwd, gather, arange, …) | 71 | 0.46 |

**Dispatcher verdict:** the three big rows are the only arithmetic
(`mm+bmm` = 84.4 ms). Everything else is pointwise/cast/norm traffic, and no
single op below `aten::mm` owns more than 8 % of the step — there is no
"one slow op" left to find, which is why the audit's remaining 42 % needed a
census rather than a bisect.

---

## 3. CPU wall vs GPU busy

`phases` mode, no profiler, 25 steps kept (the brief asked for 5; at 157 ms/step
the extra 20 are free and the medians stop moving), medians. Host = `perf_counter` around
the real seams (patched on `MixOptimizer.zero_grad/step`, `MixtureStream.next_batch`,
`Tensor.backward`, `loop.clip_grads`). GPU-on-stream = `torch.cuda.Event` pairs
recorded at the same seams — the stream's own timeline, so it includes any bubble
the host opens in front of a phase.

| phase | host med (ms) | host p95 | GPU-on-stream med (ms) | host share |
|---|---:|---:|---:|---:|
| `zero_grad` | 0.26 | 0.33 | — | 0.2 % |
| `next_batch` (loader) | **0.17** | 5.07 | — | 0.1 % |
| forward (incl. `x.to(dev)`) | 26.10 | 33.91 | 36.28 | 16.7 % |
| backward | 32.17 | 50.18 | **67.88** | 20.6 % |
| firewall (`grad_norm` + sanitize + gate) | 4.43 | 7.27 | 2.67 | 2.8 % |
| clip (`clip_grads`) | 2.02 | 5.85 | — | 1.3 % |
| `opt.step()` | **89.20** | 114.09 | 48.83 | **57.0 %** |
| **sum / step span** | **156.37** | | **156.77** | |
| **period between step starts** | **157.25** | 176.0 | | |

The honest reading, in order:

1. **GPU busy is 149.17 ms of the 157.25 ms period (94.9 %)** [M census]. The
   remaining 8.1 ms is GPU idle — and it is host-induced, because on this box the
   host's *work* is the same order as the GPU's.
2. **Host work is ~112–127 ms/step, not 156.** The 89.2 ms inside `opt.step()` is
   not work: **28.7–44 ms of it is `cudaStreamSynchronize`** (§3.1). Subtract and
   the CPU's own cost is 112–127 ms — still an upper bound, because the box
   measured 19.7 µs per tiny aten op against 0.9–9.3 µs on the quiet box the
   CPU-perf audit used [M].
3. **The pipeline is not overlapped; it oscillates.** Forward and backward run
   *ahead* of the GPU (host 62.7 ms vs GPU 106.8 ms — the GPU falls 44 ms
   behind), then the optimizer's sync drains that lead in one stall, and the
   GPU's 48.8 ms of optimizer work runs while the CPU starts the next step. That
   is why `host ≈ GPU ≈ wall`: the sync makes the CPU wait instead of running
   ahead.
4. **`torch.get_num_threads() = 10`** on a 20-core box with ~20 other processes —
   part of why the host is slow here; not a code choice I changed.

### 3.1 The step blocks on the device three times, inside `opt.step()`

Not a device *read* — the CPU-perf audit's `reads=0 (device 0)` is correct and
`stepaudit` still measures it. What its AST walk does not classify is a blocking
**host→device write** of a scalar, and the step has three of them.

`torch.cuda.set_sync_debug_mode("warn")`, 3 measured steps: **0 warnings**, and
that reading is **wrong** — `warnings` dedupes by (message, category, module,
line), and all three sync sites had already fired before the window opened. The
authority is the trace and the run's own warning stream:

| evidence | [M] |
|---|---|
| `cudaStreamSynchronize` in the trace | **3.0 / step** |
| enclosing CPU ops (innermost → outermost) | `aten::copy_` → `aten::_to_copy` → **`aten::to`** |
| where the warning fires | `torch.as_tensor(1.0 - lr*wd, dtype=…, device=cuda)` — `train/optim.py:2165` (`_foreach_adamw_step`) and the same pattern at `train/optim.py:2329` (`_gate_identity`, Muon write-back) |
| host time inside those `aten::to` calls | 28.7 ms/step profiled; median 0.1 ms, **max 46.3 ms** — the stall is the drain, not the copy |
| the copy itself | 3 blocking HtoD copies (the three scalars); 0.009 ms of traffic |

Each sync drains a queue holding ~44 ms of already-queued work; the host then
refills it. That is the entire difference between "the CPU is 46 ms ahead" (what
the forward/backward rows show) and "the CPU is exactly at the GPU's pace" (the
period). The fix, and the one trap in it. The **bit-exact** version is to keep a
persistent device buffer and `copy_(src, non_blocking=True)` into it from a
**pinned** host scalar: the value is still formed in float64 and rounded to fp32
exactly as today (`torch.as_tensor(1.0 - lr*wd, dtype=fp32)`), so the healthy
step's bytes do not move, and the copy stops blocking. The cheaper device-only
form — `one - flag.to(dtype) * (lr*wd)` — rounds in fp32 instead of float64→fp32
and can differ by **one ulp**, which is a numerics change needing its own A/B,
not a free edit (the firewall's healthy-step sha256 is pinned). Either way it
removes the only *destructive* serialization in the step.

### 3.2 GPU idle: 8.1 ms, and the gap instrument that cannot be used naively

| | profiled (20 steps) | unprofiled |
|---|---:|---:|
| Σ kernel time | 149.17 ms | 149.17 ms (the same kernels: the five cutlass GEMMs measure 302–445 µs here, 440–471 µs standalone in the audit) |
| inter-kernel gaps on stream 7 | **32.06 ms** (17.7 % of span), 164 gaps > 50 µs | **8.1 ms** (wall 157.25 − busy 149.17) |
| wall | 181.27 ms | 157.25 ms |

So the profiler's own CPU cost reappears as inter-kernel gaps (181.3 − 149.2 =
32.1 ms ≈ the 32.06 ms gap total). **The gap column of a profiled trace measures
the profiler; only `wall − Σkernel` from an unprofiled run is the GPU's idle
time.** In the unprofiled timeline the gap is 5.1 %, which bounds the total cost
of host stalls (including the three syncs) at 8.1 ms.

### 3.3 Environment caveat that changes the verdict's *margin*, not its direction

The same step on a quiet host: host work ~15–25 ms **[C]** — the CPU-perf
audit's 5.4–10.8 ms of aten dispatch + 2.2 ms Python + module walks and foreach
loops, scaled by the 5× contention this box measured; GPU busy 149.2 ms → GPU 100 % busy, wall ~149–150 ms. The host would
then have ~125 ms of slack and every host win below is worth 0 wall-clock. What
is **not** load-dependent is the three syncs: they stall the host regardless of
CPU speed.

---

## 4. Naming the 42 %

The audit's bill: 7 516 GFLOP/step (5 664 model + 1 852 NS), 145.8 ms →
51.5 TFLOP/s = 58 % of the card's 88.7 bf16/FP32-accumulate spec. Its 42 % gap =
61.2 ms. That 58 % is the **kernel-busy** number: on today's wall (157.25 ms) the
same bill reads 47.8 TFLOP/s = **53.9 %**, and the 4 points of difference are the
8.1 ms of GPU idle. Measured here, on this box:

| component | ms / step | share of wall | what it is | can it go faster? |
|---|---:|---:|---|---|
| GEMM `mm` + `bmm` (model + NS) | **94.34** | 60.0 % | 7 516 GFLOP → **79.7 TFLOP/s = 89.8 % of spec** | no — at the tensor-core roof |
| … of which Muon NS (`aten::bmm`) | 21.31 | 13.6 % | 1 852 GFLOP → **86.9 TFLOP/s = 98 % of spec** | no (and it is work, not waste) |
| pointwise / cast / copy | **40.55** | 25.8 % | bf16↔fp32 casts (301 `_to_copy`), residual/gate/rope add+mul (381 ops), LN, cat | **only by moving fewer bytes** — mean kernel 31 µs, 6–15× a launch bubble |
| optimizer + norms + cat + reduce (non-GEMM) | 14.28 | 9.1 % | momentum/foreach (5.3), LayerNorm (3.3), RoPE cat (2.3), reduce/norm (1.4), other incl. γβ-backward (2.0) | some — see §6 |
| **GPU idle (host-induced)** | **8.10** | 5.1 % | the three syncs + a host that barely keeps up | yes — §6 host win 1 |
| **total** | **157.25** | 100 % | | |

**The 42 % is 40.6 + 14.3 + 8.1 = 63.0 ms (40.1 % of the wall).** It closes the
audit's 61.2 ms within 3 %. The composition is the finding:

- **It is not host dispatch** (≤ 8.1 ms, and 0 on a quiet box).
- **It is not launch count**: 884 of 1 741 launches are ≤ 20 µs and carry 4.7 % of
  the time; 50.8 % of the launches cost 7.0 ms.
- **It is not arithmetic inefficiency**: the GEMM kernels run at 89.8 % of the
  card's bf16/FP32-accumulate peak, the NS at 98 %. `94.34 ms` of the step is
  already the fastest this card does FLOPs.
- **It is bytes**, and those bytes are served from L2, so the roof is higher than
  HBM and the removable part is much smaller than the pool. Two measurements:
  (a) the 24 forward LayerNorm kernels take 39.6 µs each over `[8,1024,832]` fp32
  = 27.2 MB in + 27.2 MB out → **1.38 TB/s effective**, above the card's 896 GB/s
  HBM spec; (b) the companion review measured the *marginal* price of the largest
  fused chain at **2.59 TB/s** (5.56 GB removed for 2.15 ms) and the whole fusion's
  payoff at **+1.48 % of the step** [M, `2026-11-08-pointwise-fusion.md` §0.2–0.3].
  So the 40.6 ms pointwise bill is **not** 40.6 ms of recoverable time: it is the
  price of the **fp32 residual stream under bf16 compute** (the port's deliberate
  mixed-dtype rule, `tsct.py:259-292`), and a fusion buys ~1.5 %, not 15 %.

So the MFU framing is the wrong denominator for the remainder: 40.6 ms of that
step can never appear as MFU because it moves bytes, not FLOPs. If the analytic
bill is 7 516 GFLOP, the *achievable* ceiling for this program is
`94.34 ms + (pointwise bytes / roofline)`, not `7 516 / 88.7 = 84.7 ms`.

---

## 5. Does the dataloader overlap with the GPU? And a prefetch thread?

`MixtureStream.next_batch` → `_windows` → `torch.frombuffer(bytearray(raw))`
(`train/data.py:89`, 85 µs at b8) + an mmap ring read: **0.169 ms median**,
p95 5.07. The tail is a stall (mmap page-in / scheduler), not bytes — the window
is 8 193 B, so there is no size effect to have.

It overlaps trivially, because it is CPU work that runs while the GPU executes
the previous step's queue: **there is no second stream and none is needed.**
The two window copies are 8 KB each and the third is a 4-byte scalar; the only
stream in the whole trace is the default one (stream 7), so no overlap machinery
exists or is being missed.

The decisive test is not "how long is the loader" but "does the step notice extra
host work". Injected sleep inside `next_batch`, 15 steps per arm, same process:

| injected CPU work | loader measured | step period (median) | Δ vs baseline | p95 |
|---:|---:|---:|---:|---:|
| 0 ms | 0.169 ms | 155.42 ms | — | 167.2 |
| **+5 ms** | 5.74 ms | 164.83 ms | **+9.4 ms** | 172.5 |
| **+20 ms** | 20.33 ms | 178.88 ms | **+23.5 ms** | 186.8 |
| **+40 ms** | 40.35 ms | 197.28 ms | **+41.9 ms** | 200.7 |

Above ~2 ms, extra host work costs the step ~1:1: **the host has ~2 ms of
headroom per step today** (0.4 ms by the phases medians; the injection says the
crossover is a few ms). At 0.169 ms the loader is well inside it, so:

- **A prefetch thread buys 0 ms of wall.** It would remove 0.169 ms of host work
  that the GPU absorbs anyway. Do not build it.
- **A separate stream for the loader is meaningless** — the loader never touches
  the device.
- The injection number is also the cheapest regression gate for host work: add
  the 20 ms arm to any future step-time A/B and a host regression shows up as a
  period change instead of hiding inside "GPU-bound".

---

## 6. Wins, ranked, each with the measurement that would prove it

### 6.1 Top 3 host-side wins

Expected value at b8 × s1024 **today: 0 ms of wall for all three** (the GPU is
94.9 % busy and the host has ~2 ms of headroom; the injection curve proves extra
host work is free below that). They buy **headroom and robustness**, and the
first one buys it on any box, loaded or not.

| # | win | site | measured today | the measurement that proves it |
|---:|---|---|---|---|
| 1 | **Delete the 3 blocking scalar copies in the AdamW/Muon write-back.** `torch.as_tensor(1.0-lr*wd, device=cuda)` → pinned host scalar + `copy_(non_blocking=True)` into a persistent device buffer (bit-exact; the device-only `one - flag*(lr*wd)` form is 1 ulp off and needs an A/B) | `train/optim.py:2165`, `:2329` | 3 `cudaStreamSynchronize`/step; **28.7 ms** of host blocking (profiled) / **40.4 ms** (opt host 89.2 − opt GPU 48.8); max single stall 46.3 ms | re-run `--mode phases` + `--mode census`: `opt` host 89.2 → ~48 ms, sync count 3 → 0, **period unchanged ±2 ms** (that is the pass condition — it must not change the step, only the headroom) |
| 2 | **Stop blocking on the input window.** Pin the host buffer in `_windows` + `non_blocking=True` on `x.to(device), y.to(device)` | `train/data.py:89`, `train/loop.py:2224` | 2 pageable HtoD/step (3 total), 0.009 ms of traffic, driver-level stall | trace: `Memcpy HtoD (Pageable -> Device)` 3.0 → 0/step; loader-injection curve shifts: the +20 ms arm should stay flat instead of +23.5 ms |
| 3 | **Fold the firewall into one multi-tensor pass.** 99–117 `nan_to_num_` launches + a 117-element Python loop, done twice per step (fwd-tail and clip) | `train/loop.py:85-104` (`_gate_grads`) | host **4.43 ms** (p95 7.27); GPU 0.90 ms (`aten::nan_to_num`) + 1 kernel for the gate | phases: `firewall` host 4.43 → ~0.5 ms; census: `aten::nan_to_num` 99 → 0–1 launches, `unrolled_elementwise` count drops accordingly; `torch.equal` on the grads after one step |

Nothing else on the host is worth a diff: the loader is 0.17 ms (§5), the
counter/invalidation/module walks are ≤ 0.2 ms each (CPU-perf audit), and the
Python step body is ~1.5 ms of 157.

### 6.2 Top 3 launch-count wins

Ranked by **launches removed per step**, with the measured milliseconds those
launches own. Half the launches are free, so the ranking is *not* the same as the
time ranking — say which one you are buying, and do not read the "time in those
kernels" column as recoverable time (row 1 is the cautionary case: 23.7 ms of
kernels, 2.3 ms measured payoff).

| # | fusion | launches removed (of 1741) | measured time in those kernels | note |
|---:|---|---:|---:|---|
| 1 | **One bf16 epilogue per block edge**: cast + residual add + LayerNorm in one kernel (fwd and VJP), i.e. stop materialising fp32 copies of the bf16 GEMM output | **~300–400** (`_to_copy` 301 + bf16/fp32 add 209 + LN 50, minus what the epilogue keeps) | those kernels cost **23.7 ms** — but the measured payoff of exactly this fusion is **+1.48 % of the step (≈2.3 ms)**, not 23.7 [M, `2026-11-08-pointwise-fusion.md` §0.2] | the launch win is real, the **time** win is ~2 ms: the chain is L2-served, so the fusion removes one pass of a 4.57 GB chain, not the pool. Parity gate: `fused_residual.PARITY_RTOL/GRTOL` (1e-5/5e-5) |
| 2 | **One multi-tensor sanitize+gate** for the gradient list (fuse `nan_to_num_` and the flag multiply) | **99–117** | 0.90 ms (`aten::nan_to_num`); the gate itself is already one `_foreach_mul_` launch | cheap, and it removes 4.43 ms of host too; correctness-critical, gate on bit-exact grads |
| 3 | **Preallocate the RoPE tables**: `arange` 24 + `cos` 12 + `sin` 12 + `cat` 38 = 86 launches | **86** | **2.35 ms** (`cat` 2.29 + trig 0.06) | the tables are already cached per `t`, but `apply_rope` still `cat`s them into the block's dtype every forward (`transformer.py:266`); a persistent `[2,1,t,d]` buffer removes both `cat`s and the per-step trig |

Deliberately **not** in the list: NS `bmm` (60 launches, 21.3 ms) is at 98 % of
peak — fusing it saves launches that cost nothing; the optimizer's `foreach` is
already fused (40 multi_tensor kernels, 5.3 ms); `torch.compile`/CUDA graphs are
measured zeros on this arm (`kernel-audit` §5, C4).

---

## 7. What could skew this result

1. **The box was loaded** (load 39–42, ~20 agents' CPU jobs, 10 torch threads).
   This is the biggest caveat and it inflates every host number ~5×
   (19.7 µs/tiny aten op vs 0.9–9.3 µs on the audit's box). It is also why the
   host is at the edge at all: on a quiet host the GPU would be 100 % busy and
   the host would have ~125 ms of slack. The three syncs are the only
   load-independent host fact.
2. **The profiler changes CPU wall, not kernel duration.** The census' gaps
   (32.06 ms) and its 181.27 ms wall are profiler artifacts; only the kernel sum
   (149.17) and the launch counts are used here, and a GEMM measures 442 µs here
   vs 440–471 µs standalone in the audit — so the kernel bill transfers.
3. **Different day, different wall from the audit** (157.25 vs 145.8, +7.8 %). The
   kernel bill is within 2.3 % (149.17 vs 145.8); the rest is host idle (8.1 ms)
   plus clock/thermal. The *split* is the transferable result, not the absolute.
4. **The MFU denominator is still the spec 88.7 TFLOP/s**, which
   `2026-11-08-mfu-first-look.md` argues is right for bf16/FP32-accumulate and
   ~2× above the *measured* GEMM ceiling (40–49 TFLOP/s). Nothing here depends on
   which one you pick: the GEMM kernels are at 89.8 % of the spec, i.e. already
   at the ceiling the whole-program MFU is compared against.
5. **One shape.** b8 × s1024 is GPU-bound by 95 %. At b2 the **optimizer's
   48.8 ms and the 1 741 launches do not shrink with the batch**, while the
   model's GEMM and pointwise bill quarter (the residual tensors are 3.4 MB, not
   13.6 MB) — so the fixed ~50 ms optimizer stage plus a 112–127 ms host makes a
   small-batch step host-bound. That crossover is **not measured here** and is
   the first check for anyone optimizing a small-batch arm.
6. **Line numbers are the dirty worktree's** (HEAD `570546b` + uncommitted
   `loop.py`/`config.py` edits by another lane), same convention as the audit.

---

## 8. Reproduction

```bash
cd port
# one GPU process at a time; the queue can be long
GPU_TIMEOUT=2400 bash scripts/gpu_lock.sh bash /tmp/opencode/hvg/run_all.sh
# or one mode at a time
bash scripts/gpu_lock.sh .venv/bin/python scripts/host_vs_gpu.py --mode phases --steps 30 --skip 5
bash scripts/gpu_lock.sh .venv/bin/python scripts/host_vs_gpu.py --mode loader
bash scripts/gpu_lock.sh .venv/bin/python scripts/host_vs_gpu.py --mode sync --measure-from 6
bash scripts/gpu_lock.sh .venv/bin/python scripts/host_vs_gpu.py --mode census --warm 20 --active 20
```

`--mode sync` **should** print a non-zero count now (`warnings.simplefilter("always")`
inside the window). The run behind this report returned 0 because of warning
dedupe; the trace is what settled it (3 syncs/step), and the fix has **not been
re-run on the card** — treat its first run as the check.
