"""Pass 2: rowwise `_scaled_mm` diagnosis + Triton error on BOTH outlier patterns.

Pass 1 (scripts/probe_fp8_kernel.py) measured speeds at the live shape; this
script answers the two questions it did not:

  1. torch's RowWise `_scaled_mm` runs on sm_120 (pass 1) but read 0.35 relative
     error — is the kernel wrong, or is the scale convention wrong? Isolated
     against the exact dequantized reference with unit / row / column scales.
  2. error of (a) the per-row fp8 dot and (b) the MXFP8 `dot_scaled` on the
     per-CHANNEL outlier pattern, which per-row scales cannot fix and 32-blocks
     can. Pass 1 only measured the per-ROW pattern.

Run: cd port && bash scripts/gpu_lock.sh .venv/bin/python scripts/probe_fp8_kernel2.py
"""

from __future__ import annotations

import json
import sys

import torch
import triton

sys.path.insert(0, "scripts")
import probe_fp8_kernel as P  # noqa: E402

torch.manual_seed(0)
E4 = 448.0
M, K, N = P.M, P.K, P.N


def q_row(x):
    amax = x.detach().float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    s = amax / E4
    q = (x.float() / s).clamp(-E4, E4).to(torch.float8_e4m3fn).contiguous()
    return q, s.contiguous()


def rowwise_diag():
    out = {}
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.8
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    qx, sx = q_row(x)
    qw, sw = q_row(w)
    swT = sw.reshape(1, N).contiguous()
    ones_m = torch.ones(M, 1, device="cuda")
    ones_n = torch.ones(1, N, device="cuda")

    def ref(xs, ws):
        # qx/qw are already quantized; multiply by the scales the kernel is told.
        # ws is per output column (N), so it scales rows of qw [N, K].
        return (qx.float() * xs) @ (qw.float() * ws.reshape(N, 1)).t()

    def call(xs, ws):
        return torch._scaled_mm(qx, qw.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)

    for name, xs, ws in [
        ("ones_ones", ones_m, ones_n),
        ("xs_ones", sx, ones_n),
        ("ones_ws", ones_m, swT),
        ("xs_ws", sx, swT),
    ]:
        try:
            y = call(xs, ws)
            r = ref(xs, ws)
            e = (y.float() - r).abs()
            rel = (torch.linalg.vector_norm(y.float() - r) / torch.linalg.vector_norm(r)).item()
            entry = {"rel_fro": rel, "max_abs": e.max().item(), "mean_abs": e.mean().item()}
            if name == "xs_ones":
                ratio = y.float() / r.abs().clamp_min(1e-6)
                entry["ratio_rows_0_5"] = [round(v, 4) for v in ratio[:6, 0].tolist()]
                entry["ratio_mean"] = ratio.mean().item()
            out[name] = entry
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    return out


def triton_errors():
    """Per-row fp8 dot vs MXFP8 dot_scaled on both outlier patterns, live shape."""
    out = {}
    for kind in ("row_outlier", "channel_outlier"):
        x = P.make_pattern(kind)
        w = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
        ref = P.fp32_reference(x, w)
        bf = x @ w.t()
        qxr, sxr = q_row(x)
        qwr, swr = q_row(w)
        qwb32 = torch.empty(N, K, device="cuda", dtype=torch.float8_e4m3fn)
        swb32 = torch.empty(N, K // 32, device="cuda", dtype=torch.uint8)
        P._quant_blocks32_kernel[(N,)](w, qwb32, swb32, K)
        torch.cuda.synchronize()

        res = {"bf16": P.err_stats(bf, ref, row=0)}
        cfg = (64, 128, 64, 4, 3)
        BM, BN, BK, nw, ns = cfg
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

        o1 = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        P._gemm_row_kernel[grid](
            x,
            sxr.view(-1),
            qwr,
            swr.view(-1),
            o1,
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
        res["triton_row_fused"] = P.err_stats(o1, ref, row=0)

        o2 = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        P._gemm_ds_kernel[grid](
            x, qwb32, swb32, o2, M, N, K, BM, BN, BK, 8, num_warps=nw, num_stages=ns
        )
        torch.cuda.synchronize()
        res["triton_mx_fused"] = P.err_stats(o2, ref, row=0)
        out[kind] = res
    return out


def outlier_sweep():
    """When does per-tensor scaling actually break? Scale ONE row's 64 columns.

    e4m3 is floating point, so a big max costs relative precision only for
    values that fall below the subnormal floor (`scale * 2^-9`). Sweeps the
    outlier multiplier and reports relFro of the dequantized product for
    per-tensor and per-row scaling.
    """
    out = {}
    w = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    g = torch.Generator(device="cuda").manual_seed(0)
    base = torch.randn(M, K, device="cuda", generator=g, dtype=torch.float32)
    base = base * (0.5 + torch.rand(M, 1, device="cuda", generator=g))
    ref = base @ w.float().t()
    refn = torch.linalg.vector_norm(ref)
    for mult in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        x = base.clone()
        x[0, :64] *= mult
        x = x.to(torch.bfloat16)
        # per-tensor
        amax = x.float().abs().amax().clamp_min(1e-12)
        s = amax / E4
        q = (x.float() / s).clamp(-E4, E4).to(torch.float8_e4m3fn)
        y_t = (q.float() * s) @ w.float().t()
        # per-row
        am = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        sr = am / E4
        qr = (x.float() / sr).clamp(-E4, E4).to(torch.float8_e4m3fn)
        y_r = (qr.float() * sr) @ w.float().t()
        out[f"x{int(mult)}"] = {
            "pertensor_rel_fro": (torch.linalg.vector_norm(y_t - ref) / refn).item(),
            "perrow_rel_fro": (torch.linalg.vector_norm(y_r - ref) / refn).item(),
            "other_rows_pertensor_rel": (
                torch.linalg.vector_norm((y_t - ref)[1:]) / torch.linalg.vector_norm(ref[1:])
            ).item(),
            "other_rows_perrow_rel": (
                torch.linalg.vector_norm((y_r - ref)[1:]) / torch.linalg.vector_norm(ref[1:])
            ).item(),
        }
    return out


if __name__ == "__main__":
    res = {
        "rowwise_diag": rowwise_diag(),
        "triton_errors": triton_errors(),
        "outlier_sweep": outlier_sweep(),
    }
    print(json.dumps(res, indent=1))
