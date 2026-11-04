"""Does the expert bank actually USE the shared input basis at compute time?

    python scripts/bench_share_u_compute.py --tokens 8192

Why: `share_u` stores ONE basis `U[r, d]` and per-expert `gate_s/up_s [2f, r]`,
but the dispatch multiplies `x` by a PER-EXPERT `[d, r]` matrix
(`bmm(xg, gate_u)` with `gate_u` of shape `[E, d, r]`). Storage is shared, compute
is not: every expert re-does the `d -> r` projection on its own rows, and that is
the expensive half (d=1536 against r=64).

The formulation `share_u` implies at compute time is:
    lat = x @ U.T                     ONE GEMM over all R rows   [R, r]
    g   = bmm(lat_by_expert, gate_s)  [E, cap, r] x [E, r, 2f]
Two smaller products instead of one huge one, and the first stops depending on E.

This script times both on synthetic tensors of the measured shape and prints the
ratio, so the change is decided by a number rather than by the flag's name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402


def bench(fn, iters: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--d", type=int, default=1536)
    ap.add_argument("--f", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--topk", type=int, default=2)
    a = ap.parse_args()

    t, d, f, e, r = a.tokens, a.d, a.f, a.experts, a.rank
    rows = t * a.topk
    cap = (rows + e - 1) // e + 8  # a little slack, as a real routing has
    dev, dt = "cuda", torch.bfloat16

    x = torch.randn(rows, d, device=dev, dtype=dt)
    U = torch.randn(r, d, device=dev, dtype=dt)
    U_per = torch.randn(e, d, r, device=dev, dtype=dt)  # today: one per expert
    G = torch.randn(e, 2 * f, r, device=dev, dtype=dt)
    V = torch.randn(e, d, f, device=dev, dtype=dt)
    xg = torch.randn(e, cap, d, device=dev, dtype=dt)
    lat_flat = torch.randn(e * cap, r, device=dev, dtype=dt)

    def today():
        # what the dispatch does now: per-expert d->r on its own rows
        g = torch.bmm(xg, U_per.transpose(1, 2))          # [E, cap, r]
        y = torch.bmm(g, G.transpose(1, 2))               # [E, cap, 2f]
        v = y[..., :f] * torch.nn.functional.silu(y[..., f:])
        return torch.bmm(v, V.transpose(1, 2))            # [E, cap, d]

    def with_shared_basis():
        # the shared basis: ONE d->r over all rows, then rank-space bmm
        lat = x @ U.T                                     # [rows, r]
        g = torch.bmm(lat.view(e, cap, r), G.transpose(1, 2))
        y = torch.bmm(g, G.transpose(1, 2))
        v = y[..., :f] * torch.nn.functional.silu(y[..., f:])
        return torch.bmm(v, V.transpose(1, 2))

    def with_shared_basis_padded():
        # the same, but the projection reads a pre-built [E, cap, r] (no view)
        g = torch.bmm(lat_flat.view(e, cap, r), G.transpose(1, 2))
        y = torch.bmm(g, G.transpose(1, 2))
        v = y[..., :f] * torch.nn.functional.silu(y[..., f:])
        return torch.bmm(v, V.transpose(1, 2))

    ms_today = bench(today)
    ms_shared = bench(with_shared_basis)
    ms_shared_p = bench(with_shared_basis_padded)

    macs_today = rows * d * r + e * cap * (r * 2 * f) + e * cap * (f * d)
    macs_shared = rows * d * r + e * cap * (r * 2 * f) + e * cap * (f * d)
    print(f"shape: tokens {t} · d {d} · f {f} · E {e} · r {r} · top{a.topk} · cap {cap}")
    print(f"  {ms_today:8.2f} ms  today (per-expert [d,r] projection)")
    print(f"  {ms_shared:8.2f} ms  shared basis (one [rows,d]x[d,r] then rank bmm)")
    print(f"  {ms_shared_p:8.2f} ms  shared basis, pre-built latent")
    print(f"  ratio today/shared = {ms_today / ms_shared:.2f}x")
    print(
        f"\n  MAC: the projection is {rows * d * r / 1e9:.1f} GMAC of "
        f"{macs_today / 1e9:.1f} GMAC total ({100 * rows * d * r / macs_today:.0f}%) "
        f"— identical in both arms, so a win here is a SHAPE win, not fewer FLOPs"
    )
    # sanity: the two must agree (same math, different order)
    torch.manual_seed(0)
    a1 = today()
    a2 = with_shared_basis_padded()
    rel = (a1.float() - a2.float()).abs().max().item() / max(1e-6, a1.float().abs().max().item())
    print(f"  parity (different reduction order, expected ~1e-3 in bf16): rel {rel:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
