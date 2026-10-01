"""4c (v2). REACHABILITY: the tolerance, by direct construction.

No eigenbasis search. G is built as a rank-2 perturbation of a multiple of the
identity, so its spectrum is known exactly:

    G = lam_rest*I + (1 - lam_rest) u1 u1^T + (l2 - lam_rest) u2 u2^T
      eigenvalues: 1 (u1), l2 (u2), lam_rest (the other 62)
      u1 has a controlled relative overlap `delta` with the all-ones direction
      u2 is orthogonal to both u1 and 1

Then M = chol(G) satisfies M M^T = G, sigma_1(M) = 1, and the power
iteration's start G*1 has (1 - lam_rest)*delta along the top eigenvector.

Run: /tmp/opencode/oracle-venv/bin/python -u sec4c_tolerance.py
"""

import torch

torch.manual_seed(20261001)
A, B, C = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0
POWER_ITERS = 5
SAFETY = 1.05
BASIN = (7.0 / 3.0) ** 0.5
THR = 1.0 / (SAFETY * BASIN)
K = 64
F32MAX = 3.4028235e38


def sigma_est(M, kk=POWER_ITERS):
    G = M @ M.T
    v = G.sum(dim=1)
    for _ in range(kk):
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


def make(l2, delta, lam_rest=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    one = torch.ones(K, 1, dtype=torch.float64)
    n1 = one / one.norm()
    # a unit vector p orthogonal to 1
    p = torch.randn(K, 1, dtype=torch.float64, generator=g)
    p = p - one * (one.T @ p) / (one.T @ one)
    p = p / p.norm()
    # a unit vector q orthogonal to both 1 and p
    q = torch.randn(K, 1, dtype=torch.float64, generator=g)
    q = q - one * (one.T @ q) / (one.T @ one)
    q = q - p * (p.T @ q)
    q = q / q.norm()
    u1 = delta * n1 + (1 - delta**2) ** 0.5 * p
    u1 = u1 / u1.norm()
    I = torch.eye(K, dtype=torch.float64)
    G = (
        lam_rest * I
        + (1.0 - lam_rest) * (u1 @ u1.T)
        + (l2 - lam_rest) * (q @ q.T)
    )
    M = torch.linalg.cholesky(G)
    assert float((M @ M.T - G).abs().max()) < 1e-10
    M = M / float(torch.linalg.matrix_norm(M, 2))
    return M, float((u1.T @ one).item() / one.norm().item())


def main():
    print("=" * 78)
    print("4c. REACHABILITY OF THE DIVERGENCE (direct construction)")
    print("=" * 78)
    print(f"  divergence needs est/sigma_1 < 1/(1.05*sqrt(7/3)) = {THR:.6f}")
    print(f"  the wrong eigenvalue dominates when delta < (l2/l1)^{POWER_ITERS}")
    print()
    print(
        f"  {'l2/l1':>7} {'delta':>10} {'(l2/l1)^5':>11} {'cos(u1,1)':>11} "
        f"{'est/sigma_1':>13} {'prescaled':>11} {'DIVERGES':>10} {'max|out|':>12}"
    )
    for l2 in [0.7, 0.5, 0.3, 0.1]:
        tol = l2**POWER_ITERS
        for mult in [0.01, 0.3, 1.0, 3.0, 8.0]:
            delta = min(tol * mult, 0.9)
            M, c = make(l2, delta, seed=int(l2 * 100) + int(mult * 10))
            true = float(torch.linalg.matrix_norm(M, 2))
            est = sigma_est(M)
            pre = true / (est * SAFETY)
            out = gram_iter(M / (est * SAFETY), 3)
            print(
                f"  {l2:7.2f} {delta:10.2e} {tol:11.3e} {c:11.2e} "
                f"{est / true:13.6f} {pre:11.4f} {str(pre > BASIN):>10} "
                f"{out.abs().max().item():12.3e}"
            )
    print()
    print("  F32: the trainer's masters are f32, max|x| = 3.4028e+38, so a")
    print("  'diverge to 1e42' is an INF, not a large number.")
    print(f"  {'l2/l1':>7} {'delta':>10} {'f64 max|out|':>14} {'f32 max|out|':>14} "
          f"{'f32 finite':>11}")
    for l2 in [0.5, 0.3, 0.1, 0.01]:
        for delta in [1e-9, 1e-4]:
            M, _ = make(l2, delta, seed=5)
            o64 = gram_iter(M / (sigma_est(M) * SAFETY), 3)
            M32 = M.float()
            est32 = torch.tensor(sigma_est(M), dtype=torch.float32)
            o32 = gram_iter(M32 / (est32 * SAFETY), 3)
            print(
                f"  {l2:7.2f} {delta:10.1e} {o64.abs().max().item():14.3e} "
                f"{o32.abs().max().item():14.3e} "
                f"{str(bool(torch.isfinite(o32).all())):>11}"
            )
    print()
    print("  THE FROBENESS ALTERNATIVE, on the same inputs. Cauchy-Schwarz gives")
    print("  sigma_max <= ||X||_F always, so a Frobenius prescale CANNOT land")
    print("  outside the basin - which is why all three reference")
    print("  implementations use it. Measured here, same factors:")
    print(f"  {'l2/l1':>7} {'delta':>10} {'frob prescaled':>15} {'frob 3-iter':>13} "
          f"{'finite':>8}")
    for l2 in [0.5, 0.3, 0.1, 0.01]:
        for delta in [1e-9, 1e-4]:
            M, _ = make(l2, delta, seed=5)
            frob = float(M.norm())
            out = gram_iter(M / (frob * SAFETY), 3)
            print(
                f"  {l2:7.2f} {delta:10.1e} {1.0 / SAFETY:15.6f} "
                f"{out.abs().max().item():13.3e} "
                f"{str(bool(torch.isfinite(out).all())):>8}"
            )


if __name__ == "__main__":
    main()
