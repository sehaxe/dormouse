"""FP8 kernel decision probe — sm_120, torch 2.13.0+cu129, triton 3.7.1.

Question: at the live dense-stack GEMM `[8192, 832] @ [3072, 832]^T`, can a
Triton kernel with a FUSED quantize beat the bf16 cuBLAS GEMM end-to-end, and
is its error on the repo's outlier pattern no worse than bf16's?

Paths measured, each isolated so one failure does not kill the run:

  torch   1. bf16 cuBLAS (baseline)
          2. `_scaled_mm` per-tensor + torch quantize (the 2026-11-08 note's lane)
          3. `_scaled_mm` per-tensor, fp8 inputs pre-quantized (GEMM-only)
          4. `_scaled_mm` rowwise   scale_a [M,1], scale_b [1,N]        (does it run?)
          5. `_scaled_mm` 1x128 blockwise fp32 scales                   (DeepSeek layout)
          6. `_scaled_mm_v2` RowWise / BlockWise1x128                   (new API)
  triton  7. plain fp8 `tl.dot`, per-row scales, A quantized in-kernel
          8. `tl.dot_scaled` MXFP8 (e8m0, 32-element K blocks), A quantized in-kernel
          9. bf16 `tl.dot` reference (is Triton codegen itself near cuBLAS?)
  bwd     10. the two bf16 backward GEMMs, for the step-level ceiling

Every number: warmup 5, iters 30, best of 3 reps, CUDA events. Error is
measured against an fp32 (TF32 off) reference on two synthetic patterns —
the repo's row-outlier pattern and a per-channel outlier pattern (which is
what LayerNorm outputs actually look like).

Run (GPU protocol):
  cd port && bash scripts/gpu_lock.sh .venv/bin/python scripts/probe_fp8_kernel.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0
M, K, N = 8192, 832, 3072
DEV = "cuda"

RESULTS: dict = {"meta": {}, "paths": {}, "triton": {}, "patterns": {}}


# --------------------------------------------------------------------------- #
# tensors / reference
# --------------------------------------------------------------------------- #
def make_pattern(kind: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(M, K, device=DEV, generator=g, dtype=torch.float32)
    x = x * (0.5 + torch.rand(M, 1, device=DEV, generator=g))
    if kind == "row_outlier":  # the 2026-11-08 note's pattern
        x[0, :64] *= 40.0
    elif kind == "channel_outlier":  # per-channel outliers: the real LayerNorm shape
        x[:, :8] *= 40.0
    elif kind != "plain":
        raise ValueError(kind)
    return x.to(torch.bfloat16)


def fp32_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    torch.backends.cuda.matmul.allow_tf32 = False
    return (x.float() @ w.float().t()).contiguous()


def err_stats(c: torch.Tensor, ref: torch.Tensor, row: int | None = None) -> dict:
    e = (c.float() - ref).abs()
    denom = ref.abs().clamp_min(1e-3)
    out = {
        "max_abs": e.max().item(),
        "mean_abs": e.mean().item(),
        "rel_fro": (
            torch.linalg.vector_norm(c.float() - ref) / torch.linalg.vector_norm(ref)
        ).item(),
        "mean_rel_pct": (e / denom).mean().item() * 100,
    }
    if row is not None:
        out["row0_mean_abs"] = e[row].mean().item()
        out["row0_max_abs"] = e[row].max().item()
    return out


# --------------------------------------------------------------------------- #
# torch quantizers
# --------------------------------------------------------------------------- #
def q_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.detach().float().abs().amax().clamp_min(1e-12)
    s = (amax / E4M3_MAX).reshape(1)
    q = (x.float() / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).contiguous()
    return q, s


def q_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.detach().float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    s = amax / E4M3_MAX
    q = (x.float() / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).contiguous()
    return q, s.contiguous()


def q_block128(x: torch.Tensor, blk: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 + fp32 amax inverse scales for 1x128 blocks along the last dim.

    The live K=832 is NOT a multiple of 128, so the fp8 operand is padded with
    zero columns to 896 (and the last block's scale is a tiny constant whose
    product with 0 is 0). The GEMM then does 7.7% more MACs; that is charged to
    the 1x128 row honestly.
    """
    rows, cols = x.shape
    nb = (cols + blk - 1) // blk
    xp = torch.zeros(rows, nb * blk, device=x.device, dtype=torch.float32)
    xp[:, :cols] = x.float()
    xb = xp.reshape(rows, nb, blk)
    amax = xb.abs().amax(dim=2).clamp_min(1e-12)
    s = amax / E4M3_MAX
    q = (xb / s[:, :, None]).clamp(-E4M3_MAX, E4M3_MAX).reshape(rows, nb * blk)
    return q.to(torch.float8_e4m3fn).contiguous(), s.contiguous()


# --------------------------------------------------------------------------- #
# triton kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _row_amax_kernel(x_ptr, s_ptr, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * COLS + offs, mask=offs < COLS, other=0.0).to(tl.float32)
    tl.store(s_ptr + row, tl.max(tl.abs(x)) / 448.0)


@triton.jit
def _quant_rows_kernel(x_ptr, s_ptr, q_ptr, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    s = tl.load(s_ptr + row)
    x = tl.load(x_ptr + row * COLS + offs, mask=offs < COLS, other=0.0).to(tl.float32)
    q = tl.clamp(x / s, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(q_ptr + row * COLS + offs, q, mask=offs < COLS)


@triton.jit
def _quant_blocks32_kernel(x_ptr, q_ptr, s_ptr, COLS: tl.constexpr):
    """fp8 + e8m0 byte scales for 32-element K blocks (MXFP8). COLS % 32 == 0."""
    row = tl.program_id(0)
    offs = tl.arange(0, 32)
    for b in range(0, COLS // 32):
        x = tl.load(x_ptr + row * COLS + b * 32 + offs).to(tl.float32)
        amax = tl.max(tl.abs(x))
        e = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0))
        e = tl.clamp(e, -127.0, 127.0)
        s = tl.exp2(e)
        q = tl.clamp(x / s, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(q_ptr + row * COLS + b * 32 + offs, q)
        tl.store(s_ptr + row * (COLS // 32) + b, (e.to(tl.int32) + 127).to(tl.uint8))


@triton.jit
def _gemm_row_kernel(
    a_ptr,
    sa_ptr,
    b_ptr,
    sb_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C = (A/row_scale -> e4m3) @ B^T * row_scale * B_scale. A quantized in-kernel."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]
    sa = tl.load(sa_ptr + offs_m)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K // BK):
        a = tl.load(a_ptrs)  # bf16
        b = tl.load(b_ptrs)  # fp8, [BN, BK]
        qa = tl.clamp(a.to(tl.float32) / sa[:, None], -448.0, 448.0).to(tl.float8e4nv)
        acc = tl.dot(qa, tl.trans(b), acc, out_dtype=tl.float32)
        a_ptrs += BK
        b_ptrs += BK
    sb = tl.load(sb_ptr + offs_n)
    acc = acc * sa[:, None] * sb[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16))


@triton.jit
def _gemm_ds_kernel(
    a_ptr,
    b_ptr,
    sb_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C = MXFP8 dot_scaled; A's 32-block e8m0 scales built in-kernel from bf16."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_s = tl.arange(0, BK // 32)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]
    sb_ptrs = sb_ptr + offs_n[:, None] * (K // 32) + (offs_s[None, :] + 0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _kb in range(0, K // BK):
        a = tl.load(a_ptrs)  # bf16 [BM, BK]
        b = tl.load(b_ptrs)  # fp8  [BN, BK]
        ar = tl.reshape(a, (BM, BK // 32, 32))
        amax = tl.max(tl.abs(ar), axis=2)  # [BM, BK//32]
        e = tl.clamp(tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0)), -127.0, 127.0)
        s = tl.exp2(e)
        qa = tl.clamp(
            a.to(tl.float32)
            / tl.reshape(tl.broadcast_to(s[:, :, None], (BM, BK // 32, 32)), (BM, BK)),
            -448.0,
            448.0,
        ).to(tl.float8e4nv)
        sa8 = (e.to(tl.int32) + 127).to(tl.uint8)
        sb8 = tl.load(sb_ptrs)
        acc = tl.dot_scaled(qa, sa8, "e4m3", tl.trans(b), sb8, "e4m3", acc=acc)
        a_ptrs += BK
        b_ptrs += BK
        sb_ptrs += BK // 32
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16))


@triton.jit
def _gemm_fp8_kernel(
    a_ptr,
    sa_ptr,
    b_ptr,
    sb_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Pure fp8 dot on pre-quantized operands; scales applied in the epilogue."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K // BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)
        a_ptrs += BK
        b_ptrs += BK
    sa = tl.load(sa_ptr + offs_m)
    sb = tl.load(sb_ptr + offs_n)
    acc = acc * sa[:, None] * sb[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16))


@triton.jit
def _gemm_bf16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K // BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)
        a_ptrs += BK
        b_ptrs += BK
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16))


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #
def bench(fn, iters: int = 30, warmup: int = 5, reps: int = 3) -> dict:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) / iters)
    times.sort()
    return {"min_ms": times[0], "med_ms": times[len(times) // 2]}


def best_of(fn, cfg, kernel, iters=30):
    """Time one tile config; return (ms, cfg, compiled_kernel)."""
    grid = (triton.cdiv(M, cfg[0]) * triton.cdiv(N, cfg[1]),)

    def run():
        kernel[grid](*fn(cfg))

    t = bench(run, iters=iters)
    return t["med_ms"], t["min_ms"], cfg


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    prop = torch.cuda.get_device_properties(0)
    RESULTS["meta"] = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": prop.name,
        "cap": list(torch.cuda.get_device_capability()),
        "sm_count": prop.multi_processor_count,
        "torch_cuda": torch.version.cuda,
        "M": M,
        "K": K,
        "N": N,
        "shape": "[8192,832] @ [3072,832]^T",
    }
    print(json.dumps(RESULTS["meta"], indent=1))

    torch.manual_seed(0)
    w = (torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02).contiguous()
    w.float()

    # ---------------- torch paths, per pattern ---------------- #
    patterns = ["row_outlier", "channel_outlier"]
    for kind in patterns:
        x = make_pattern(kind)
        ref = fp32_reference(x, w)
        entry = {"fp32_ref_norm": torch.linalg.vector_norm(ref).item()}

        # 1. bf16 cuBLAS
        yb = x @ w.t()
        tb = bench(lambda x=x: x @ w.t())
        entry["bf16_cublas"] = {"ms": tb, **err_stats(yb, ref, row=0)}

        # 2. _scaled_mm per-tensor + torch quantize (the note's lane)
        qx, sx = q_tensor(x)
        qw, sw = q_tensor(w)

        def mm_pt(qx=qx, qw=qw, sx=sx, sw=sw):
            return torch._scaled_mm(qx, qw.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)

        try:
            y8 = mm_pt()
            torch.cuda.synchronize()
            t8 = bench(mm_pt)
            tq = bench(lambda x=x: (q_tensor(x), q_tensor(w)))
            entry["scaled_mm_pertensor"] = {
                "ms": t8,
                "quant_ms": tq,
                "end_to_end_ms": t8["med_ms"] + tq["med_ms"],
                "speedup_vs_bf16": tb["med_ms"] / (t8["med_ms"] + tq["med_ms"]),
                **err_stats(y8, ref, row=0),
            }
        except Exception as e:  # noqa: BLE001
            entry["scaled_mm_pertensor"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

        # 3. per-tensor GEMM only (fp8 inputs prequantized) -> is 1.97x real now?
        try:
            t8g = bench(mm_pt)
            entry["scaled_mm_pertensor_gemm_only"] = {
                "ms": t8g,
                "speedup_vs_bf16": tb["med_ms"] / t8g["med_ms"],
            }
        except Exception as e:  # noqa: BLE001
            entry["scaled_mm_pertensor_gemm_only"] = {
                "error": f"{type(e).__name__}: {str(e)[:300]}"
            }

        # 4. rowwise v1: scale_a [M,1], scale_b [1,N]. B must be quantized with
        # the SAME per-row scales it is given (a per-tensor B asked to dequantize
        # with per-row scales is ~35% wrong — a probe bug, not a torch bug).
        qxr, sxr = q_row(x)
        qwr, swr = q_row(w)
        swr = swr.reshape(1, N).contiguous()

        def mm_row(qxr=qxr, qwr=qwr, sxr=sxr, swr=swr):
            return torch._scaled_mm(
                qxr, qwr.t(), scale_a=sxr, scale_b=swr, out_dtype=torch.bfloat16
            )

        try:
            yr = mm_row()
            torch.cuda.synchronize()
            tr = bench(mm_row)
            entry["scaled_mm_rowwise"] = {"ms": tr, **err_stats(yr, ref, row=0)}
        except Exception as e:  # noqa: BLE001
            entry["scaled_mm_rowwise"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

        # 5. v1 1x128 blockwise fp32 scales (DeepSeek layout, K padded to 896)
        try:
            qxb, sxb = q_block128(x)
            qwb, swb = q_block128(w)

            def mm_128(qxb=qxb, qwb=qwb, sxb=sxb, swb=swb):
                return torch._scaled_mm(
                    qxb, qwb.t(), scale_a=sxb, scale_b=swb, out_dtype=torch.bfloat16
                )

            yblk = mm_128()
            tblk = bench(mm_128)
            entry["scaled_mm_block128"] = {
                "ms": tblk,
                "K_padded": qxb.shape[1],
                **err_stats(yblk, ref, row=0),
            }
        except Exception as e:  # noqa: BLE001
            entry["scaled_mm_block128"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

        # 6. _scaled_mm_v2 RowWise / BlockWise1x128
        from torch.nn.functional import ScalingType, SwizzleType

        for label, recipe in [
            ("v2_rowwise", ScalingType.RowWise),
            ("v2_block128", ScalingType.BlockWise1x128),
        ]:
            try:
                if recipe == ScalingType.RowWise:
                    qa_, sa = qxr, sxr
                    qb_, sb = qwr, swr
                else:
                    qa_, sa = q_block128(x)
                    qb_, sb = q_block128(w)

                def mm_v2(qa_=qa_, qb_=qb_, sa=sa, sb=sb, recipe=recipe):
                    return torch.ops.aten._scaled_mm_v2.default(
                        qa_,
                        qb_.t(),
                        [sa],
                        [int(recipe)],
                        [int(SwizzleType.NO_SWIZZLE)],
                        [sb],
                        [int(recipe)],
                        [int(SwizzleType.NO_SWIZZLE)],
                        None,
                        torch.bfloat16,
                        [],
                        False,
                    )

                yv = mm_v2()
                torch.cuda.synchronize()
                tv = bench(mm_v2)
                entry[label] = {"ms": tv, **err_stats(yv, ref, row=0)}
            except Exception as e:  # noqa: BLE001
                entry[label] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

        RESULTS["patterns"][kind] = entry
        print(f"\n== pattern {kind} ==")
        for k, v in entry.items():
            print(f"  {k}: {json.dumps(v, default=str)[:400]}")

    # ---------------- Triton kernels (row_outlier pattern) ---------------- #
    x = make_pattern("row_outlier")
    ref = fp32_reference(x, w)
    qxr, sxr = q_row(x)
    qw, sw = q_tensor(w)
    # B prequantized per row (weight static in training)
    qwr, swr = q_row(w)
    # B prequantized per 32-block (MXFP8)
    qwb32 = torch.empty(N, K, device=DEV, dtype=torch.float8_e4m3fn)
    swb32 = torch.empty(N, K // 32, device=DEV, dtype=torch.uint8)
    try:
        _quant_blocks32_kernel[(N,)](w, qwb32, swb32, K)
        torch.cuda.synchronize()
        RESULTS["triton_block_quant"] = "ok"
    except Exception as e:  # noqa: BLE001
        RESULTS["triton_block_quant"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  block-32 quant kernel FAILED: {e}")

    bf16_ms = RESULTS["patterns"]["row_outlier"]["bf16_cublas"]["ms"]["med_ms"]
    cfgs = [(128, 128, 64, 8, 3), (128, 64, 64, 4, 3), (64, 128, 64, 4, 3), (64, 64, 64, 4, 3)]
    tri = {}

    # 7. plain fp8 dot per-row, A fused in-kernel
    out = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    best = None
    for cfg in cfgs:
        BM, BN, BK, nw, ns = cfg
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        try:

            def f(c=cfg, g=grid):
                return _gemm_row_kernel[g](
                    x,
                    sxr.view(-1),
                    qwr,
                    swr.view(-1),
                    out,
                    M,
                    N,
                    K,
                    c[0],
                    c[1],
                    c[2],
                    8,
                    num_warps=c[3],
                    num_stages=c[4],
                )

            t = bench(f, iters=20)
            if best is None or t["med_ms"] < best[0]:
                best = (t["med_ms"], t["min_ms"], cfg)
            print(f"  gemm_row cfg={cfg}: med {t['med_ms']:.3f} min {t['min_ms']:.3f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"  gemm_row cfg={cfg}: FAILED {type(e).__name__}: {str(e)[:200]}")
    if best:
        BM, BN, BK, nw, ns = best[2]
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _gemm_row_kernel[grid](
            x,
            sxr.view(-1),
            qwr,
            swr.view(-1),
            out,
            M,
            N,
            K,
            BM,
            BN,
            BK,
            8,
            num_warps=nw,
            num_stages=ns,
        )
        torch.cuda.synchronize()
        tri["fp8_dot_rowwise_fused"] = {
            "cfg": best[2],
            "med_ms": best[0],
            "min_ms": best[1],
            "speedup_vs_bf16": bf16_ms / best[0],
            **err_stats(out, ref, row=0),
        }

    # 7b. pure fp8 dot on pre-quantized A: the separate-quantize design.
    # Fusing the quantize into the GEMM recomputes it once per N-tile (24x at
    # BN=128) and re-reads A as bf16 every time; one quantize pass costs a
    # single read+write instead. This pair of numbers decides which is cheaper.
    qA = torch.empty(M, K, device=DEV, dtype=torch.float8_e4m3fn)
    sA = torch.empty(M, device=DEV, dtype=torch.float32)
    try:
        _row_amax_kernel[(M,)](x, sA, K, 1024)
        _quant_rows_kernel[(M,)](x, sA, qA, K, 1024)
        torch.cuda.synchronize()
        out4 = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
        best = None
        for cfg in cfgs:
            BM, BN, BK, nw, ns = cfg
            grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
            try:

                def f(c=cfg, g=grid):
                    return _gemm_fp8_kernel[g](
                        qA,
                        sA,
                        qwr,
                        swr.view(-1),
                        out4,
                        M,
                        N,
                        K,
                        c[0],
                        c[1],
                        c[2],
                        8,
                        num_warps=c[3],
                        num_stages=c[4],
                    )

                t = bench(f, iters=20)
                if best is None or t["med_ms"] < best[0]:
                    best = (t["med_ms"], t["min_ms"], cfg)
                print(
                    f"  gemm_fp8_prequant cfg={cfg}: med {t['med_ms']:.3f} min {t['min_ms']:.3f} ms"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  gemm_fp8_prequant cfg={cfg}: FAILED {type(e).__name__}: {str(e)[:200]}")
        if best:
            BM, BN, BK, nw, ns = best[2]
            grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
            kern = _gemm_fp8_kernel[grid](
                qA, sA, qwr, swr.view(-1), out4, M, N, K, BM, BN, BK, 8, num_warps=nw, num_stages=ns
            )
            torch.cuda.synchronize()
            tri["fp8_gemm_prequant"] = {
                "cfg": best[2],
                "med_ms": best[0],
                "min_ms": best[1],
                "speedup_vs_bf16": bf16_ms / best[0],
                **err_stats(out4, ref, row=0),
            }
            try:
                ptx = kern.asm["ptx"]
                lines = [ln.strip() for ln in ptx.splitlines() if "mma" in ln]
                tri["fp8_gemm_prequant"]["mma_lines"] = sorted(set(lines))[:6]
            except Exception as e:  # noqa: BLE001
                tri["fp8_gemm_prequant"]["mma_lines"] = [f"ptx unavailable: {e}"]
    except Exception as e:  # noqa: BLE001
        tri["fp8_gemm_prequant"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"  prequant path FAILED: {e}")

    # 8. dot_scaled MXFP8, A scales built in-kernel
    out2 = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    best = None
    for cfg in cfgs:
        BM, BN, BK, nw, ns = cfg
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        try:

            def f(c=cfg, g=grid):
                return _gemm_ds_kernel[g](
                    x,
                    qwb32,
                    swb32,
                    out2,
                    M,
                    N,
                    K,
                    c[0],
                    c[1],
                    c[2],
                    8,
                    num_warps=c[3],
                    num_stages=c[4],
                )

            t = bench(f, iters=20)
            if best is None or t["med_ms"] < best[0]:
                best = (t["med_ms"], t["min_ms"], cfg)
            print(f"  dot_scaled cfg={cfg}: med {t['med_ms']:.3f} min {t['min_ms']:.3f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"  dot_scaled cfg={cfg}: FAILED {type(e).__name__}: {str(e)[:300]}")
    if best:
        BM, BN, BK, nw, ns = best[2]
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        kern = _gemm_ds_kernel[grid](
            x, qwb32, swb32, out2, M, N, K, BM, BN, BK, 8, num_warps=nw, num_stages=ns
        )
        torch.cuda.synchronize()
        tri["dot_scaled_mxfp8_fused"] = {
            "cfg": best[2],
            "med_ms": best[0],
            "min_ms": best[1],
            "speedup_vs_bf16": bf16_ms / best[0],
            **err_stats(out2, ref, row=0),
        }
        # which MMA did it lower to?
        try:
            ptx = kern.asm["ptx"]
            lines = [
                ln.strip()
                for ln in ptx.splitlines()
                if ("mma" in ln or "wgmma" in ln or "tcgen05" in ln) and "//" not in ln[:4]
            ]
            tri["dot_scaled_mxfp8_fused"]["mma_lines"] = sorted(set(lines))[:8]
        except Exception as e:  # noqa: BLE001
            tri["dot_scaled_mxfp8_fused"]["mma_lines"] = [f"ptx unavailable: {e}"]

    # 9. bf16 triton (codegen reference)
    out3 = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    best = None
    for cfg in cfgs:
        BM, BN, BK, nw, ns = cfg
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        try:

            def f(c=cfg, g=grid):
                return _gemm_bf16_kernel[g](
                    x, w, out3, M, N, K, c[0], c[1], c[2], 8, num_warps=c[3], num_stages=c[4]
                )

            t = bench(f, iters=20)
            if best is None or t["med_ms"] < best[0]:
                best = (t["med_ms"], t["min_ms"], cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  triton bf16 cfg={cfg}: FAILED {type(e).__name__}: {str(e)[:200]}")
    if best:
        tri["triton_bf16"] = {
            "cfg": best[2],
            "med_ms": best[0],
            "min_ms": best[1],
            "speedup_vs_bf16": bf16_ms / best[0],
        }

    # 10. separate quantize kernels, and the fused-lane end-to-end numbers
    s_ = torch.empty(M, device=DEV, dtype=torch.float32)
    q_ = torch.empty(M, K, device=DEV, dtype=torch.float8_e4m3fn)
    t_amax = bench(lambda: _row_amax_kernel[(M,)](x, s_, K, 1024), iters=50)
    t_quant = bench(lambda: _quant_rows_kernel[(M,)](x, s_, q_, K, 1024), iters=50)
    tri["row_amax_kernel"] = t_amax
    tri["quant_rows_kernel"] = t_quant
    tri["quant_pair_ms"] = t_amax["med_ms"] + t_quant["med_ms"]
    if "fp8_gemm_prequant" in tri and "med_ms" in tri["fp8_gemm_prequant"]:
        lane = tri["fp8_gemm_prequant"]["med_ms"] + t_amax["med_ms"] + t_quant["med_ms"]
        tri["lane_prequant_separate_e2e_ms"] = lane
        tri["lane_prequant_separate_speedup"] = bf16_ms / lane
    if "fp8_dot_rowwise_fused" in tri:
        lane = tri["fp8_dot_rowwise_fused"]["med_ms"] + t_amax["med_ms"]
        tri["lane_rowwise_e2e_ms"] = lane
        tri["lane_rowwise_speedup"] = bf16_ms / lane
    if "dot_scaled_mxfp8_fused" in tri:
        lane = tri["dot_scaled_mxfp8_fused"]["med_ms"] + t_amax["med_ms"]
        tri["lane_dot_scaled_e2e_ms"] = lane
        tri["lane_dot_scaled_speedup"] = bf16_ms / lane

    # 11. backward GEMMs (bf16), for the step-level ceiling
    dy = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)
    torch.empty(M, K, device=DEV, dtype=torch.bfloat16)
    torch.empty(N, K, device=DEV, dtype=torch.bfloat16)
    t_dx = bench(lambda: torch.matmul(dy, w))
    t_dw = bench(lambda: torch.matmul(dy.t(), x))
    tri["bwd_bf16"] = {
        "dx_ms": t_dx,
        "dw_ms": t_dw,
        "fwd_bwd_ms": t_dx["med_ms"] + t_dw["med_ms"] + bf16_ms,
    }
    RESULTS["triton"] = tri

    print("\n== triton / lane ==")
    for k, v in tri.items():
        print(f"  {k}: {json.dumps(v, default=str)[:500]}")

    out_path = Path("/tmp/opencode/fp8_kernel_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
