"""Section 4 alone, fixed: can the sigma_max estimate land OUTSIDE the basin?

The bug in the first attempt: `cholesky(G)` returns R with G = R^T R, so
`M = R` gives `M M^T = R R^T != G`. The eigenbasis I reasoned about was not
the one being measured. This version uses L (M = L, so M M^T = L L^T = G) and
asserts the reconstruction.

Run: /tmp/opencode/oracle-venv/bin/python sec4_fixed.py
"""

import numpy as np
import torch

torch.manual_seed(20261001)
A, B, C = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0
POWER_ITERS = 5
SAFETY = 1.05
BASIN = (7.0 / 3.0) ** 0.5
THR = 1.0 / (SAFETY * BASIN)  # est/sigma_1 below this => prescale outside basin


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


def ortho_basis_with_first_orthogonal_to_ones(k, rng):
    """An orthonormal basis of R^k whose FIRST vector is exactly orthogonal
    to the all-ones vector, completed by Gram-Schmidt (verified orthonormal)."""
    Q, _ = torch.linalg.qr(torch.randn(k, k, dtype=torch.float64, generator=rng))
    one = torch.ones(k, 1, dtype=torch.float64)
    e1 = Q[:, 0:1] - one * (one.T @ Q[:, 0:1]) / (one.T @ one)
    e1 = e1 / e1.norm()
    cols = [e1]
    for j in range(1, k):
        v = Q[:, j : j + 1].clone()
        for c in cols:
            v = v - c * (c.T @ v)
        v = v / v.norm()
        cols.append(v)
    Q2 = torch.cat(cols, dim=1)
    err = float((Q2.T @ Q2 - torch.eye(k, dtype=torch.float64)).abs().max())
    assert err < 1e-10, f"basis not orthonormal: {err}"
    # and the first vector really is orthogonal to 1
    assert float((e1.T @ one).abs()) < 1e-12
    return Q2, e1


def main():
    rng = torch.Generator().manual_seed(20261001)
    k = 64
    one = torch.ones(k, 1, dtype=torch.float64)
    print("=" * 78)
    print("4 (fixed). CAN THE ESTIMATE LAND OUTSIDE THE BASIN?")
    print("=" * 78)
    print(f"  basin edge sqrt(7/3) = {BASIN:.6f}")
    print(f"  divergence threshold est/sigma_1 < {THR:.6f}")
    print("  the start vector is G*1 (lib.rs:230), so a top eigenvector")
    print("  ORTHOGONAL TO 1 is exactly the case the start vector cannot see.")
    print()
    print(
        f"  {'l2/l1':>8} {'cos(start,e1)':>15} {'est/sigma_1':>13} {'prescaled':>11} "
        f"{'OUTSIDE':>8} {'max|out|':>12} {'finite':>7}"
    )
    for l2 in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 1e-2, 1e-4]:
        Q2, e1 = ortho_basis_with_first_orthogonal_to_ones(k, rng)
        lam = torch.full((k,), 0.05, dtype=torch.float64)
        lam[0], lam[1] = 1.0, l2
        G = Q2 @ torch.diag(lam) @ Q2.T
        L = torch.linalg.cholesky(G)  # G = L L^T
        M = L.clone()  # M M^T = L L^T = G   <-- the fix
        assert float((M @ M.T - G).abs().max()) < 1e-10
        M = M / float(torch.linalg.matrix_norm(M, 2))  # sigma_1 = 1
        true = float(torch.linalg.matrix_norm(M, 2))
        start = G @ one
        cosang = float((e1.T @ start).abs() / start.norm())
        est = sigma_est(M)
        pre = true / (est * SAFETY)
        out = gram_iter(M / (est * SAFETY), 3)
        print(
            f"  {l2:8.4f} {cosang:15.2e} {est / true:13.6f} {pre:11.4f} "
            f"{str(pre > BASIN):>8} {out.abs().max().item():12.3e} "
            f"{str(bool(torch.isfinite(out).all())):>7}"
        )

    print()
    print("  random adversarial search: 20000 draws, top eigvec in 1-perp,")
    print("  spectrum log-uniform on [1e-6, 1] apart from the forced sigma_1=1.")
    found = []
    for trial in range(20000):
        Q2, _ = ortho_basis_with_first_orthogonal_to_ones(k, rng)
        lam = torch.exp(
            torch.rand(k, dtype=torch.float64, generator=rng) * -13.8
        )
        lam[0] = 1.0
        G = Q2 @ torch.diag(lam) @ Q2.T
        G = G + 1e-14 * torch.eye(k, dtype=torch.float64)  # cholesky needs PD
        try:
            M = torch.linalg.cholesky(G)
        except Exception:
            continue
        true = float(torch.linalg.matrix_norm(M, 2))
        est = sigma_est(M)
        pre = true / (est * SAFETY)
        if pre > BASIN:
            found.append((trial, true, est, pre, lam[1].item()))
    print(f"    outside the basin: {len(found)}/20000")
    for t, true, est, pre, l2 in found[:5]:
        print(
            f"      trial {t}: sigma_1={true:.4f} est={est:.4e} "
            f"({est / true * 100:.2f}% of truth) prescaled={pre:.4f} "
            f"l2/l1={l2:.3e}"
        )
        M = None
    if found:
        # reproduce the first one and show what the 3-iteration retraction does
        pass

    print()
    print("  the same search but with the top eigvec NOT constrained (ordinary")
    print("  factors), to show the constraint is what does the work:")
    out_n = 0
    worst = 0.0
    for _ in range(4000):
        M = torch.randn(k, k, dtype=torch.float64, generator=rng)
        true = float(torch.linalg.matrix_norm(M, 2))
        est = sigma_est(M)
        pre = true / (est * SAFETY)
        worst = max(worst, pre)
        out_n += pre > BASIN
    print(f"    outside the basin: {out_n}/4000, worst prescaled {worst:.4f}")

    print()
    print("  REACHABILITY: the drift sweep. Start from an ORTHONORMAL factor")
    print("  (what the retraction actually receives, up to one optimizer step of")
    print("  drift) and add adversarial drift that kills the start vector.")
    print(f"  {'drift':>8} {'kind':>12} {'est/sigma_1':>13} {'prescaled':>11} "
          f"{'OUTSIDE':>8}")
    for eps in [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 0.5, 1.0, 3.0]:
        for kind in ["isotropic", "kills-1"]:
            Q2, _ = ortho_basis_with_first_orthogonal_to_ones(k, rng)
            base = Q2  # orthonormal columns, sigma_i = 1
            E = torch.randn(k, k, dtype=torch.float64, generator=rng)
            if kind == "kills-1":
                # drift whose Gram has ~zero ROW SUMS: E with column sums
                # cancelling, which is what makes G*1 vanish.
                E = E - E.mean(dim=0, keepdim=True)
            M = base + eps * E
            true = float(torch.linalg.matrix_norm(M, 2))
            est = sigma_est(M)
            pre = true / (est * SAFETY)
            print(
                f"  {eps:8.1e} {kind:>12} {est / true:13.6f} {pre:11.4f} "
                f"{str(pre > BASIN):>8}"
            )


if __name__ == "__main__":
    main()
