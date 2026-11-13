"""Does FP8 actually run on THIS card, and is it faster? — a 5-second probe.

Why: DeepSeek's one transferable training-infra contribution is FP8 (their V3
report validates it at 671B, 2.664M H800-hours, no rollbacks). Their kernel
library (DeepGEMM) requires **SM90 or SM100** and this card is **SM120**
(`torch.cuda.get_device_capability() == (12, 0)`), so DeepGEMM itself cannot run
here. What can run is `torch._scaled_mm` — which the repo already uses in
`mm_mixed`'s `mm_dtype=fp8` lane, forward only (`docs/records/architecture/
2026-10-23-1b-in-a-week.md:78`: "blocked — torch has no `_scaled_mm` backward").

This probe answers three questions in order, and stops at the first NO:

1. does `_scaled_mm` accept e4m3 at our training shape `[8192, 832] @ [832, 3072]`?
2. is it FASTER than the bf16 GEMM it would replace, and by how much?
3. how much error does per-tensor amax scaling introduce on real activations?

A custom autograd Function is what turns (1)+(2) into a training lane: the
forward runs `_scaled_mm`, the backward runs the bf16 GEMM on the unquantized
masters (the standard straight-through-with-scale recipe). That is ~15 lines, and
it is worth writing only if this probe says the GEMM runs and wins.
"""

from __future__ import annotations

import time

import torch

from dormouse.instrument import loud

M, K, N = 8192, 832, 3072  # the dense stack's real shape at b=8, s=1024
E4M3_MAX = 448.0


def quant_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor amax e4m3 (the rule `model/tsct.py:_quant_e4m3` already uses)."""
    amax = x.detach().abs().amax().to(torch.float32).clamp_min(1e-12)
    scale = (amax / E4M3_MAX).reshape(1)
    q = (x.float() / scale).to(torch.float8_e4m3fn).contiguous()
    return q, scale


def bench(fn, iters: int = 50) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main() -> None:
    if not torch.cuda.is_available():
        loud("no CUDA device: this probe measures a GPU lane, nothing else")
    cap = torch.cuda.get_device_capability()
    print(f"gpu={torch.cuda.get_device_name(0)} cap={cap} torch={torch.__version__}")

    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    x = x * (0.5 + torch.rand(M, 1, device="cuda", dtype=torch.bfloat16))  # per-row spread
    x[0, :64] *= 40.0  # one outlier row: the fp8 failure mode

    # --- 1. does the kernel accept the call? ---------------------------------
    try:
        qx, sx = quant_e4m3(x)
        qw, sw = quant_e4m3(w)
        y8 = torch._scaled_mm(qx, qw.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - the probe's whole point is to report this
        print(f"RESULT 1: _scaled_mm REFUSED this shape on sm_{cap[0]}{cap[1]}: {e}")
        return
    print(f"RESULT 1: _scaled_mm ran, out={tuple(y8.shape)} {y8.dtype}")

    # --- 2. speed against the bf16 GEMM it replaces --------------------------
    y_bf16 = x @ w.t()
    t8 = bench(
        lambda: torch._scaled_mm(qx, qw.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
    )
    tq = bench(lambda: (quant_e4m3(x), quant_e4m3(w)))
    tb = bench(lambda: x @ w.t())
    print(
        f"RESULT 2: bf16 {tb:.3f} ms | fp8 gemm {t8:.3f} ms | quantize {tq:.3f} ms | "
        f"end-to-end {t8 + tq:.3f} ms -> speedup {(tb / (t8 + tq)):.2f}x"
    )

    # --- 3. accuracy of per-tensor scaling ----------------------------------
    err = (y8.float() - y_bf16.float()).abs()
    denom = y_bf16.float().abs().clamp_min(1e-3)
    print(
        f"RESULT 3: max abs err {err.max():.4f} | mean rel err "
        f"{(err / denom).mean() * 100:.2f}% | outlier-row rel err "
        f"{(err[0].mean() / denom[0].mean()) * 100:.2f}%"
    )

    # --- 4. is the backward the blocker this repo says it is? ---------------
    xr = x.detach().clone().requires_grad_(True)
    try:
        y = torch._scaled_mm(*quant_e4m3(xr), qw.t(), scale_b=sw, out_dtype=torch.bfloat16)
        y.sum().backward()
        print("RESULT 4: _scaled_mm HAS a backward in this torch (the 2026-10-23 note is stale)")
    except Exception as e:  # noqa: BLE001
        print(
            f"RESULT 4: no backward ({type(e).__name__}: {str(e)[:90]}) -> "
            f"a custom autograd Function is required"
        )


if __name__ == "__main__":
    main()
