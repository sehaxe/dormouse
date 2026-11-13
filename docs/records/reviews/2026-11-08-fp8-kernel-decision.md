# 2026-11-08 — FP8 kernel decision: Triton already reaches the hardware, and the win is not in the forward

**Lane:** `port/`. **Card:** RTX 5070 Ti, sm_120, 70 SMs. **Stack:** torch 2.13.0+cu129, triton 3.7.1. **Not committed.**

**Question (owner + `docs/records/research/2026-11-08-fp8-on-sm120.md`).** That note measured `torch._scaled_mm` fp8 at 1.97× bf16 on the live GEMM `[8192, 832] @ [3072, 832]^T`, but the per-tensor quantize ate the win (1.09× end to end), per-row scales were refused, and per-tensor scaling was called numerically unusable (0.81 abs error with a 40× outlier row). Which kernel should be written, is it worth it, and should part of this be C++?

**Verdict.** `triton.language.dot_scaled` works on this card: it lowers to the native `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0`, confirmed in the generated PTX. But the block-scaled GEMM is not worth writing:

1. **There is no GEMM headroom.** A plain Triton fp8 GEMM on pre-quantized operands runs at **0.250 ms = 2.13× bf16** (0.533 ms), above `torch._scaled_mm`'s 2.06× (0.259 ms) — the fastest fp8 rate anything reached at this shape. The hardware ratio is ~2×; no kernel language can beat it.
2. **The numerical premise is false as measured.** On the note's own 40×-outlier-row pattern, per-tensor, per-row and 32-block scaling all land at **relFro 3.55–3.57%**. That is e4m3's 3-bit noise floor, 21× bf16's 0.166%; the 0.96 absolute error is the format's error **on the outlier row**, not cross-row contamination. A CPU sweep shows per-tensor separates from per-row only around **10 000×** dynamic range, two to three orders of magnitude beyond the note's 40×.
3. **The win is not in the forward.** Measured at the same shape, bf16 forward:backward = 0.533 : 1.337 ms, so replacing only the forward gives **1.14× on the GEMM triple**, and with the repo's 82% block-GEMM FLOP share that is a **~1.08× step** — not the note's 1.37×. The 1.37× assumed fp8 on most GEMM work; two-thirds of it is the backward.

The only kernel worth owning is the **quantizer**: torch's quantize (both operands) is 0.409 ms, a Triton activation amax+cast pair is **0.0515 ms** (the weight half is 37.5% of that and hoistable in a training step, so the per-step cost is the activation pass alone). It is ~50 lines. C++ buys nothing at this shape: the same instruction is one `tl.dot_scaled` call away, and the port has zero C++ sources and no `CUDAExtension` (verified), so C++ would add build infrastructure for at most the ~5% between Triton and the theoretical 2×.

---

## 1. The live shape and the measurement setup

Live step: `port/configs/flagship3_dense.toml` — 12 dense layers, `d_model=832`, `d_ffn=3072`, b=8, s=1024. The FFN GEMM is `[8192, 832] @ [3072, 832]^T` = 41.87 GFLOP. The step is 136 ms at b=8/s=1024 (model fwd+bwd 104 ms, optimizer 32 ms; `docs/architecture/2026-11-08-training-step.md:340-358`), and the FLOP ledger puts 567.41 of 691.37 MFLOP/byte in the block — **82% of the step's FLOPs are block GEMMs** (`port/configs/flagship3_dense.toml:24-27`).

Probes: `port/scripts/probe_fp8_kernel.py` (pass 1) and `port/scripts/probe_fp8_kernel2.py` (pass 2). Warmup 5, 30 iters, best of 3 reps, CUDA events. `gpu_lock` entered at 1738 MiB (idle); other lanes share the box, so treat absolute ms as ±10% and same-run ratios as the evidence.

**The note's absolute ms were contention-inflated** — it says so in its own header ("E1 held the card at 100%"). Under an idle card: bf16 **0.533 ms**, fp8 0.259 ms, not 1.037/0.526. The **ratio reproduces**: 2.06× against the note's 1.97×.

---

## 2. Does `tl.dot_scaled` work on sm_120? Yes — and it is the native instruction

Triton 3.7.1 has an sm_120-specific `ScaledBlockedToMMA` pattern (`lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`: `computeCapability != 120 -> failure()`), which converts both scales to `getSM120DotScaledScaleLayout` and lowers through MMAv2. The probe's generated PTX is the direct proof:

```
mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0
  { ... }, { ... }, { ... }, { ... }, %scaleA, {0, 0}, %scaleB, {0, 0};
```

That is the **hardware** block-scaled MMA, not a bf16-upcast emulation. Per the PTX ISA 9.7.16.3 (Target ISA Notes):

- `.kind` + `.block_scale` + `.scale_vec_size` **requires sm_120a**, supported on sm_120f and later in the family (PTX ISA 8.7+). This is a consumer-Blackwell feature; the `tcgen05` block-scaled path is SM100-family only, which is why DeepGEMM (SM90/SM100) cannot run here.
- `scale_A` is `M x 1`, `scale_B` is `1 x N` (`.scale_vec::1X`): **one e8m0 scale per 32-element K chunk per row**, applied by the tensor core. A per-row scale is the degenerate case (same byte in every K chunk); a per-128 block is the same byte across 4 consecutive 32-chunks.
- The scale bytes are selected per thread with `{byte-id, thread-id}`: one thread-pair per quad contributes A's scale, one thread per quad contributes B's.

Triton adds only `out_dtype=float32` and `numCTAs == 1`. The probe's inner loop is the whole kernel:

```python
ar = tl.reshape(a, (BM, BK // 32, 32))            # a: bf16 tile [BM, BK]
amax = tl.max(tl.abs(ar), axis=2)                 # [BM, BK//32]
e = tl.clamp(tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0)), -127.0, 127.0)
s = tl.exp2(e)
qa = tl.clamp(a.to(tl.float32) /
              tl.reshape(tl.broadcast_to(s[:, :, None], (BM, BK // 32, 32)), (BM, BK)),
              -448.0, 448.0).to(tl.float8e4nv)
sa8 = (e.to(tl.int32) + 127).to(tl.uint8)         # e8m0 bytes
acc = tl.dot_scaled(qa, sa8, "e4m3", tl.trans(b), sb8, "e4m3", acc=acc)
```

The scale rule (`ceil(log2(amax/448))`, e8m0 code = exponent+127) is the one `port/src/dormouse/model/mxfp.py` already implements for the MX codec — one source of truth, not a second convention.

**The `torch._scaled_mm` picture is different, and worth stating precisely:**

| path | result on this card |
|---|---|
| per-tensor (scalar scales) | runs; GEMM-only 2.06× |
| **per-row** (`[M,1]` / `[1,N]` scales) | **runs** (0.285 ms) and is **numerically correct**: 0.166% against the exact dequantized product (bf16 output rounding), unit/row/column scale cases isolated. The note's "refused" was almost certainly its calling shape (a 1-D `[M]` scale vs the documented `[M,1]`), not the kernel |
| 1x128 / 128x128 (DeepSeek) | refused: `_check_deepseek_support()` requires `major == 9` — **SM90 only**, hard-coded (`aten/src/ATen/native/cuda/ScaledBlas.cpp`) |
| MX 1x32 e8m0 (blockwise) | meta path exists; the compute path is SM100-only in inductor |

**A correction to this review's own first measurement:** pass 1 reported 0.349 relFro for the per-row path. That was a **probe bug** — B was quantized with a per-tensor scale and then handed per-row scales (a ~35% multiplicative error), not a torch defect. Pass 2 isolated the scale application with unit/row/column cases and the result is exact (0.166% vs the dequantized product, i.e. bf16 output rounding). The corrected error vs the fp32 reference is the quantization floor measured for the identical `q_row` on the same tensors in the Triton per-row kernel: **3.554%**. Do not cite the 0.349 number.

The per-row path is real: `ScaledBlas.cpp` dispatches `major >= 10` + 2-D scales to `f8f8bf16_rowwise`, and `RowwiseScaledMM.cu` has a dedicated **Sm120** branch. It is available today. What remains unavailable on this card is only the **blockwise** 1x128/128x128 scheme (§4 shows it is not needed either).

---

## 3. Speed: the GEMM is already at the ceiling; the quantize is the only headroom

All at `[8192, 832] @ [3072, 832]^T`, same run, idle card:

| path | ms (med) | vs bf16 | notes |
|---|---:|---:|---|
| bf16 cuBLAS | 0.533 | 1.00 | 78.6 TFLOP/s |
| `_scaled_mm` per-tensor, GEMM only | 0.259 | **2.06×** | 161.8 TFLOP/s |
| `_scaled_mm` per-tensor + torch quantize | 0.676 | **0.79×** | torch quantize 0.409 ms |
| `_scaled_mm` rowwise, GEMM only | 0.285 | 1.87× | correct (verified §2), ~10% slower than per-tensor |
| Triton bf16 `tl.dot` | 0.507 | 1.05× | codegen is 5% *faster* than cuBLAS at this shape |
| Triton fp8 `tl.dot`, pre-quantized A | **0.250** | **2.13×** | `mma.sync...f32.e4m3.e4m3.f32` |
| Triton fp8 per-row, A quantized **in-kernel** | 0.394 | 1.35× | fused version — 58% slower than prequant |
| Triton `dot_scaled` MXFP8, A scales **in-kernel** | 0.316 | 1.69× | native `mxf8f6f4.block_scale` |
| Triton row-amax kernel | 0.0249 | — | 8192 rows × 832 |
| Triton row-quantize kernel | 0.0266 | — | read bf16, write e4m3 |
| **quantize pair (amax+cast)** | **0.0515** | — | activation half; torch's both-operands quantize is 0.409 → ~5.8× for both, and the weight half hoists |
| **lane: prequant Triton fp8 + quantize pair** | **0.3015** | **1.77×** | the honest per-GEMM lane |
| lane: `_scaled_mm` + Triton quantize pair | 0.3105 | 1.72× | same hardware, torch GEMM |
| lane: fused per-row + amax | 0.419 | 1.27× | worse than prequant |
| lane: fused MX + amax | 0.341 | 1.56× | block scaling costs 26% vs plain fp8 |
| backward bf16 dX | 0.670 | — | `dY @ W` |
| backward bf16 dW | 0.667 | — | `dY^T @ X` |
| **forward+backward bf16** | **1.870** | 1.00 | 0.533 + 0.670 + 0.667 |

Three consequences:

1. **There is no GEMM kernel to win.** Triton's plain fp8 GEMM (0.250) beats `_scaled_mm` (0.259) and is the fastest fp8 rate measured at this shape; the ceiling is the hardware ratio, not the codegen. A hand-written CUDA kernel can only tie it or lose to it.
2. **Fusing the quantize into the GEMM is the wrong design at this shape.** With `BN=128` there are 24 N-tiles, so the in-kernel A-quantize recomputes 24× and re-reads A as bf16 each time: 0.394 vs 0.250+0.0515 = 0.302. One separate quantize pass is ~5.8× cheaper than torch's both-operands quantizer (8× on the activation half) and wins.
3. **The block-scaled instruction costs ~26%** (0.316 vs 0.250) — the price of the scale-layout conversion and the block-scale MMA on this part. It is not a precision win (§4).

---

## 4. Numerics: the scaling scheme is not the lever; the format is

`[8192,832]@[3072,832]^T`, error against an fp32 (TF32-off) reference. `row_outlier` is the note's pattern (row 0, first 64 columns ×40); the reference's RMS element is 0.634.

| path | max abs | mean abs | relFro | row-0 mean abs |
|---|---:|---:|---:|---:|
| bf16 cuBLAS | 0.068 | 0.00065 | **0.166%** | 0.0076 |
| `_scaled_mm` per-tensor | 0.963 | 0.01723 | **3.565%** | 0.176 |
| Triton per-row (correct kernel) | 0.964 | 0.01718 | **3.554%** | 0.182 |
| Triton MX 32-block (`dot_scaled`) | 0.882 | 0.01728 | **3.573%** | 0.173 |

The torch rowwise path uses the same `q_row` quantization and applies its scales exactly (0.166% vs the dequantized product, above), so its error vs fp32 is the same 3.554% floor as the Triton per-row kernel — not the 34.9% pass 1 printed, which was the probe bug corrected in §2.

**The bf16 row and the three fp8 rows are the finding.** The note's 40× outlier row does **not** make per-tensor scaling unusable: per-tensor, per-row and 32-block scaling are within **0.6% relative** of each other in relFro (3.554–3.573%), and within 10% in max abs. All are ~21× bf16, because e4m3 has 3 mantissa bits against bf16's 8 — 2^5 ≈ 32× fewer, minus the RMS factor. e4m3 is floating-point: a larger amax costs *absolute* resolution in proportion to the value; the *relative* error stays the format's ~3.5% until values fall below the subnormal floor (`scale·2^-9`). A 40× range is 2^5.3, far inside e4m3's ~2^17 range. The note's "0.81 abs on a signal of ~0.58" is the fp8 error **on the outlier row itself** (whose values are 40× larger), not contamination of the other rows.

A CPU sweep of the same quantizer (512×832 × 512×832, one row's 64 columns × mult; the trend is the point, the absolute level differs from the GPU run) shows where per-tensor actually separates from per-row:

| outlier mult | per-tensor relFro | per-row relFro | per-tensor other rows | per-row other rows | values below e4m3 subnormal |
|---:|---:|---:|---:|---:|---:|
| 1× | 2.644% | 2.628% | 2.644% | 2.628% | 0.02% |
| 10× | 2.644% | 2.631% | 2.642% | 2.628% | 0.14% |
| **40× (the note's pattern)** | **2.659%** | **2.650%** | **2.642%** | **2.628%** | **0.54%** |
| 100× | 2.683% | 2.678% | 2.644% | 2.628% | 1.37% |
| 1 000× | 2.696% | 2.696% | 2.648% | 2.628% | 13.6% |
| 10 000× | 2.696% | 2.696% | **5.422%** | **2.628%** | 87.0% |
| 100 000× | 2.696% | 2.696% | **52.3%** | **2.628%** | 100% |

At 40×, per-tensor and per-row differ by **0.3% relative**; the non-outlier rows by 0.5%. The schemes only separate around **10 000×**, and the mechanism is the subnormal floor, not the mantissa. The note's "per-tensor unusable at 40×" is measurably false.

The channel-outlier pattern (8 columns ×40) tells the same story with the actual kernels: bf16 0.166%, Triton per-row **3.318%**, Triton MX **3.732%** — the same floor, and here the finer per-32 blocks are slightly *worse* than per-row, consistent with e8m0's power-of-two scale giving up to 1 bit of headroom while the per-row scale is exact. The scaling scheme does not buy accuracy on either pattern; the format is the whole error.

**Practical reading.** If 3.5% relFro fp8 GEMMs are acceptable, per-tensor `_scaled_mm` + a fast quantizer is enough, and no block-scaled GEMM is needed. If they are not acceptable, **no block-scaling scheme fixes it — the format does.** Per-tensor with a fresh amax per batch (what we measured) has no saturation failure mode at all; block scales mainly buy a cheaper scale-reduction cadence, which our 0.025 ms amax pass already makes moot.

---

## 5. What the step actually gains: the backward is two-thirds of the GEMM work

Measured at the same shape: bf16 forward 0.533, dX 0.670, dW 0.667 → fwd+bwd **1.870 ms**. Replacing only the forward with the measured fp8 lane (0.302) gives **1.638 ms = 1.14×** on the GEMM triple. Replacing all three at ~2.1× gives 0.302 + 0.637 = **0.939 ms = 2.0×** (backward lane not measured).

Step projection from the measured 82% block-GEMM share of the 104 ms model time, leaving the 32 ms optimizer alone:

| scope | GEMM time | step time | step speedup |
|---|---:|---:|---:|
| bf16 (measured) | 85 ms | 136 ms | 1.00 |
| fp8 forward only, bf16 backward (measured ratios, forward quantize included) | 74.7 ms | 125.4 ms | **1.08×** |
| fp8 fwd + bwd, 2.0× on all GEMMs (derived) | 43 ms | 94 ms | **1.45×** |
| fp8 fwd + bwd, minus backward quantize (~2.4 ms, derived) | — | ~96 ms | **1.42×** |

**The forward-only project is worth ~1.08× on the step, not 1.37×.** The note's parenthetical "(GEMMs are 104 ms of the step)" is the misreading: 104 ms is the **whole model fwd+bwd** (`docs/architecture/2026-11-08-training-step.md:340-358`), not the GEMMs, so 136→99 ms would require halving the entire model including the backward. `[derived]` — from the measured per-GEMM ratios and the repo's 82% FLOP share; no end-to-end fp8 training step was run.

---

## 6. The CUDA C++ fallback: what it would need, and why it is not the answer

If a hand-written kernel were wanted, the instruction set is documented:

- **Instruction:** `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0` (PTX ISA 8.7+, **sm_120a**; the dense `mma` block-scale section, not `tcgen05`).
- **Scales:** e8m0 bytes, one per 32-element K chunk per row, selected per thread with `{byte-id, thread-id}`: one thread-pair per quad for A, one thread per quad for B. The scale operand is a `b32` register holding 4 bytes (4 consecutive K chunks), reused across 4 MMA issues; the selector picks the byte.
- **Data:** m16n8k32 fragments, e4m3 in 8-bit containers, A row-major / B column-major (`.row.col`).
- **Epilogue:** fp32 accumulators (4 `f32` per thread per m16n8 tile); multiply by the per-row/column scale if the scheme is coarser than 1x32; cast to bf16; store. The block scales are applied by the hardware.
- **For 1x128 or 128x128** (DeepSeek's), fill the same e8m0 value across the 4 K chunks of the 128-block (and across the 128 rows for 128x128). The hardware is more granular than the scheme, so a coarser scheme is expressible, not the reverse.

**CUTLASS already has the scaffold.** CUTLASS 4.8.1 ships `CollectiveBuilder<arch::Sm120, arch::OpClassBlockScaledTensorOp, ...>` with `KernelScheduleMxf8f6f4Sm120` (`include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`), SM120 block-scale MMA atoms in `include/cute/arch/mma_sm120.hpp`, and `sm120_blockscaled_mma_tma.hpp`. Constraints visible in the builder: TN layout only, cluster size 1 (no multicast on this part), `N` tile ≥ 8, `PermTileK=32` for MXF8F6F4. **DeepGEMM cannot run** (README: SM90 or SM100; SM100's scale format is packed UE8M0 and the kernel family is tcgen05-based, which sm_120 does not have). The CUTLASS README also states that SM100 arch-conditional kernels are **not** compatible with RTX 50 (SM120).

**Effort, for someone who has this codebase:**

| route | work | realistic time |
|---|---|---|
| Triton forward lane: quantizer + scales + autograd Function + shape/counter gates | ~150–250 lines, no build system | **3–5 days** |
| Triton fp8 **backward** lane: 3 GEMM shapes, transposed layouts, quantize of dY (25.2M elements), autotune per shape, gates | ~400–600 lines | **1.5–2.5 weeks** |
| C++/CUTLASS SM120 blockscaled: new nvcc extension (the port has **zero** `.cu`/`.cpp` and no `CUDAExtension` — verified), SF-layout conversion, fused quantize, wheel/CI | CUTLASS instantiation + build infra | **2–4 weeks** |
| hand-written PTX/mma pipeline (no CUTLASS) | shared-memory pipeline, swizzle, scale selectors, epilogue | **4–8 weeks** |

The expected ceiling is the measured one: **2.1× GEMM-only, ~1.77× lane**; C++ can at best recover the small gap between Triton and the hardware's fp8 rate at this shape (~5%), which is not worth 2–4 weeks of build infrastructure — and certainly not worth hand-written PTX when the same instruction is one `tl.dot_scaled` call away.

---

## 7. The decision rule

The owner's example rule was "build iff end-to-end ≥ 1.25× and error ≤ bf16's". The first half is met; the second half is **unachievable by construction** for any fp8 format on this GEMM (3 mantissa bits vs 8; measured 3.55% vs 0.166%). Use this instead:

```
FORWARD-ONLY LANE (quantizer + per-tensor fp8 GEMM):
  build iff  lane >= 1.25x bf16 at the live shape          [measured 1.77x  -> PASS]
        and  relFro <= 1.5x the e4m3 noise floor (3.5%)     [measured 3.55%  -> PASS]
        and  projected step >= 1.10x                        [derived 1.05-1.10x -> FAIL at the margin]
  -> do NOT build. The lane is fast enough; the step is not.

FULL FP8 LANE (forward + backward):
  build iff  each of the 3 GEMM shapes >= 1.8x bf16          [fwd 2.13x measured; dX/dW not measured]
        and  the full step >= 1.25x MEASURED (not derived)   [derived 1.40-1.45x]
        and  200 steps on real data, loss within the bf16 seed spread, no NaN
  -> the only version that pays; 2-3x the forward-only work; needs the numerical A/B first.

C++: never for this lane. The same instruction is reachable from Triton at ~95% of the
     fp8 rate; the ceiling is the hardware ratio, not the language.
```

**Recommendation.** Do not write the block-scaled GEMM. The note's premise ("per-tensor unusable → per-row/block kernel needed") is refuted by the measurement: the scaling scheme changes relFro by 0.6% relative, while the format changes it 21×. If fp8 is pursued, it is a **backward-lane** project with a 200-step numerical gate; the forward GEMM is a solved problem (`_scaled_mm` or a 35-line Triton kernel) and the only kernel worth owning is the **0.05 ms quantizer**. Otherwise keep the cheaper levers (batch sweep, NS dtype, optimizer) the note already ranked above this.

---

## 8. Reproduce

```
cd port
bash scripts/gpu_lock.sh .venv/bin/python -u scripts/probe_fp8_kernel.py    # pass 1
bash scripts/gpu_lock.sh .venv/bin/python -u scripts/probe_fp8_kernel2.py   # pass 2
bash scripts/gpu_lock.sh .venv/bin/python -u scripts/probe_rowwise_error.py # corrected rowwise
```

Pass 1 prints the torch paths, the Triton kernels, the PTX MMA lines and the lane arithmetic; pass 2 prints the rowwise scale isolation, both outlier patterns and the multiplier sweep; the third re-measures the corrected rowwise call. Full JSON: `/tmp/opencode/fp8_kernel_results.json`; logs under `/tmp/opencode/`.

---

## VERIFIED (sourced / measured)

- `tl.dot_scaled` compiles and runs on sm_120 with triton 3.7.1; the PTX contains `mma.sync...kind::mxf8f6f4.block_scale.scale_vec::1X...ue8m0` — a **measured** generated instruction. Triton has a dedicated `ScaledBlockedToMMA` sm_120 pattern (`AccelerateMatmul.cpp`, `computeCapability != 120` guard).
- PTX ISA 9.7.16.3: dense `mma` block scaling requires **sm_120a**, sm_120f+ in the family; `tcgen05` block scaling is SM100-family only. CUTLASS 4.8.1 ships SM120 blockscaled builders/atoms; SM100 arch-conditional kernels are **not** compatible with SM120 (CUTLASS README).
- DeepGEMM requires SM90 or SM100 (README, requirements) — not runnable here.
- torch 2.13.0+cu129: per-row `_scaled_mm` **does run** on sm_120 (v1 and v2) and is **numerically correct** — 0.166% against the exact dequantized product with unit/row/column scale cases isolated; its error vs fp32 is the 3.554% fp8 floor (the identical `q_row` measured on the Triton row kernel). The 0.349 number from this review's pass 1 was a probe bug (B quantized per-tensor, handed per-row scales) and is **retracted**. The 1x128/128x128 path is hard-refused by `_check_deepseek_support()` (`major == 9`); the rowwise arch dispatch has a dedicated Sm120 branch (`RowwiseScaledMM.cu`).
- Measured at `[8192,832]@[3072,832]^T`: bf16 0.533 ms; `_scaled_mm` per-tensor GEMM 0.259 (2.06×); Triton fp8 prequant 0.250 (2.13×); torch quantize 0.409 vs Triton amax+cast 0.0515; lane 0.3015 (1.77×); MX `dot_scaled` 0.316; bwd 0.670+0.667.
- Measured errors: bf16 relFro 0.166%; per-tensor 3.565%; Triton per-row 3.554%; MX 32-block 3.573% on the 40× row-outlier pattern; on the channel-outlier pattern bf16 0.166%, per-row 3.318%, MX 3.732%. The three fp8 schemes are within 0.6% relative of each other on the row pattern. CPU and GPU sweeps: per-tensor vs per-row separate only at ~10 000× dynamic range (the GPU sweep's non-outlier rows: 2.655% → 2.658% through 1 000×, 3.804% at 10 000×; per-row 2.645% throughout).
- The port has no C++ extension: zero `.cu`/`.cuh`/`.cpp`/`.cc` outside `.venv`/`target`, no `CUDAExtension` in `src`/`tests`/`scripts`.

## SPECULATION (my inference, not measured)

- **Step projections** (1.05–1.10× forward-only, 1.40–1.45× full fp8) are derived from the measured per-GEMM ratios, the repo's 82% block-FLOP share and the 104 ms model time. No end-to-end fp8 training step was run.
- **Effort estimates** (3–5 days forward Triton, 1.5–2.5 weeks backward, 2–4 weeks C++/CUTLASS) are judgment, not measurements.
- **Trainability of a 3.5% relFro GEMM for this model is unverified.** DeepSeek's fp8 recipe ran on a different architecture, scale and kernel; the repo has no fp8 training A/B. A 200-step loss gate is the minimum before any training claim.

## Open questions

1. Per-row `_scaled_mm` is correct but ~10% slower than per-tensor at this shape (0.285 vs 0.259) and buys no accuracy on the patterns measured; the open question is whether any **real** activation distribution separates them. A measured amax histogram over 200 steps (question 3) would answer both.
2. Are the backward GEMM shapes (`[8192,3072]@[3072,832]` and `[3072,8192]@[8192,832]`) also at 2× when fp8? Not measured; they are K-heavy and may tune differently.
3. What is the real activation dynamic range of this model? The 40× synthetic outlier is far inside e4m3's range; the failure mode starts around 10 000× (CPU sweep). A measured amax histogram over 200 steps would decide whether block scaling has any real work to do.
4. Does the 200-step loss move under fp8 GEMMs (forward only, then fwd+bwd)? That is the gate that decides the project, and it is cheap relative to the kernel work.
5. A channel-pattern run of the Triton per-row and MX kernels (pass 1 measured the torch per-tensor path only on that pattern). The CPU trend says the schemes will be within 0.5%; the GPU number would close it.
