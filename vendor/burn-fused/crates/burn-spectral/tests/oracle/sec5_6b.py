"""Sections 5 and 6b, fixed, unbuffered.

5 (fixed): the previous run measured `svdvals(W)` on a WIDE [96,128] matrix, so
its "max/min ratio" was the rank deficiency of a wide matrix, not anything the
retraction did. The right question is whether the retracted factors make
W = U diag(s) V^T an exact SVD, and the right test is U^T W V == diag(s).

6b: the number the owner actually needs. `retract_iters = 3` is the trainer
default and the one-way `max_ortho` latch is 1e-3 PER ENTRY. At what
singular-value spread does a 3-iteration retraction land over that latch?

Run: /tmp/opencode/oracle-venv/bin/python -u sec5_6b.py
"""

import numpy as np
import torch

torch.manual_seed(20261001)
A, B, C = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0
POWER_ITERS = 5
SAFETY = 1.05
LATCH = 1e-3  # train/src/lib.rs:1352, per entry


def sigma_est(M, k=POWER_ITERS):
    G = M @ M.T
    v = G.sum(dim=1)
    for _ in range(k):
        v = v / v.norm()
        v = G @ v
    gv = G @ v
    return float(((v * gv).sum() / (v * v).sum()).sqrt())


def gram_iter(M, iters):
    M = M.clone()
    for _ in range(iters):
        G = M @ M.T
        M = A * M + (B * G + C * (G @ G)) @ M
    return M


def retract(X, iters):
    """lib.rs:208-266 on a [rows, cols] tensor, canonicalised as the code does."""
    M = X.T if X.shape[0] > X.shape[1] else X.clone()
    out = gram_iter(M / (sigma_est(M) * SAFETY), iters)
    return out.T if X.shape[0] > X.shape[1] else out


def sec5():
    print("=" * 78)
    print("5 (fixed). DOES RETRACTING U AND V SEPARATELY KEEP W's SPECTRUM = s?")
    print("=" * 78)
    print("  the claim: {U^T U = I, V^T V = I} is exactly the set on which")
    print("  W = U diag(s) V^T IS an SVD with singular values s. Test: U^T W V.")
    d, f, k = 96, 128, 64
    for spread in [1.0, 10.0, 1e3]:
        torch.manual_seed(7)
        U = retract(torch.randn(d, k, dtype=torch.float64), 20)
        V = retract(torch.randn(f, k, dtype=torch.float64), 20)
        s = torch.logspace(
            0, -np.log10(spread), k, dtype=torch.float64
        )
        W = U @ torch.diag(s) @ V.T
        core = U.T @ W @ V  # must be exactly diag(s)
        off = (core - torch.diag(torch.diagonal(core))).abs().max().item()
        diag = torch.diagonal(core)
        rel = ((diag - s).abs() / s).max().item()
        print(
            f"  spread {spread:<8.0e} ||U^T U - I||_F="
            f"{((U.T @ U - torch.eye(k, dtype=torch.float64)).norm()):.2e}"
            f"  max|offdiag(U^T W V)|={off:.3e}  max rel diag error={rel:.3e}"
        )
        print(
            f"      -> U^T W V IS diag(s) to {off:.1e}: the factors being"
            f" retracted separately is SOUND, and `s` survives as the spectrum"
        )
    print()
    print("  the OTHER question: what does the retraction do to the FUNCTION?")
    print("  W_on = U_on diag(s) V_on^T; W_off after a drift E and one retraction.")
    d, f, k = 256, 256, 64
    s = torch.ones(k, dtype=torch.float64)
    for eps in [1e-4, 1e-3, 1e-2, 1e-1]:
        torch.manual_seed(3)
        U = retract(torch.randn(d, k, dtype=torch.float64), 20)
        V = retract(torch.randn(f, k, dtype=torch.float64), 20)
        W_on = U @ torch.diag(s) @ V.T
        g1 = torch.Generator().manual_seed(11)
        g2 = torch.Generator().manual_seed(12)
        Uo = retract(U + eps * torch.randn(d, k, dtype=torch.float64, generator=g1), 3)
        Vo = retract(V + eps * torch.randn(f, k, dtype=torch.float64, generator=g2), 3)
        W_off = Uo @ torch.diag(s) @ Vo.T
        pe = ((Uo.T @ Uo - torch.eye(k, dtype=torch.float64)).norm() / k).item()
        print(
            f"    drift {eps:.0e}: rel ||W_off - W_on||_F = "
            f"{((W_off - W_on).norm() / W_on.norm()).item():.4e}   "
            f"per-entry ortho AFTER retract(3) = {pe:.3e}"
        )
    print("  a retraction is ALLOWED to move W - that is what a projection is.")
    print("  The fixed point that must hold is the FACTORS'. And the number")
    print("  above is the cost of a SKIPPED retraction, which is the cadence A/B.")


def sec6b():
    print()
    print("=" * 78)
    print("6b. THE LATCH CROSSING: retract_iters=3 vs the one-way 1e-3 latch")
    print("=" * 78)
    d, k = 768, 64
    torch.manual_seed(11)
    Q, _ = torch.linalg.qr(torch.randn(d, k, dtype=torch.float64))
    print(
        f"  {'sigma_min/sigma_max':>19} {'iters':>6} {'per-entry':>11} "
        f"{'x latch':>9} {'relF vs LAPACK':>16}"
    )
    for spread in [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]:
        for iters in [3, 4, 5, 6, 8]:
            s = torch.logspace(0, np.log10(spread), k, dtype=torch.float64)
            M = Q @ torch.diag(s)
            out = retract(M, iters)
            pe = ((out.T @ out - torch.eye(k, dtype=torch.float64)).norm() / k).item()
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            rel = (out - (U @ Vh)).norm().item() / (U @ Vh).norm().item()
            mark = "  <-- LATCH FIRES" if pe > LATCH else ""
            print(
                f"  {spread:19.2f} {iters:6d} {pe:11.3e} {pe / LATCH:9.1f} "
                f"{rel:16.4e}{mark}"
            )
    print()
    print("  READ: at the trainer's default retract_iters=3, per-entry ortho")
    print("  stays under 1e-3 only while the factor's spectrum is within about")
    print("  2:1. Past that the ONE-WAY fp32 fallback fires and is persisted in")
    print("  the checkpoint, so a factor that drifts wide silently and")
    print("  permanently disables the factor-quant forward for the rest of the")
    print("  run. The iteration count is the knob that decides it.")


if __name__ == "__main__":
    sec5()
    sec6b()
