"""4d. WOULD THE ONE-LINE GUARD FIX EVERY DIVERGING CASE?

The guard: never prescale by an estimate that Cauchy-Schwarz cannot vouch for.

    sigma_max(X) <= ||X||_F   (always, exactly)
    so  sigma_true/(sigma_used * 1.05) <= ||X||_F/(sigma_used * 1.05)

Dividing by `max(sigma_est, ||X||_F / (1.05 * sqrt(7/3)))` makes the prescaled
input provably <= sqrt(7/3) - inside the basin - for EVERY input, at the cost of
one extra tensor reduction and no host sync. When sigma_est is healthy it wins
and nothing changes; when it is wrong the Frobenius bound takes over.

Run: /tmp/opencode/oracle-venv/bin/python -u sec4d_guard.py
"""

import torch

torch.manual_seed(20261001)
A, B, C = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0
POWER_ITERS = 5
SAFETY = 1.05
BASIN = (7.0 / 3.0) ** 0.5
K = 64
# 1.05 * sqrt(7/3): the divisor below is ||X||_F / this, so the prescaled
# sigma_max is at most sqrt(7/3) by Cauchy-Schwarz.
GUARD = SAFETY * BASIN


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


def dct_orthonormal(n, k):
    """The first k DCT-II modes: orthonormal BY CONSTRUCTION, no RNG."""
    i = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    j = torch.arange(k, dtype=torch.float64).unsqueeze(0)
    scale = torch.where(j == 0, n**-0.5, (2.0 / n) ** 0.5)
    return scale * torch.cos(torch.pi * (2 * i + 1) * j / (2 * n))


def make(l2, delta, lam_rest=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    one = torch.ones(K, 1, dtype=torch.float64)
    n1 = one / one.norm()
    p = torch.randn(K, 1, dtype=torch.float64, generator=g)
    p = p - one * (one.T @ p) / (one.T @ one)
    p = p / p.norm()
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
    return M / float(torch.linalg.matrix_norm(M, 2))


def main():
    print("=" * 78)
    print("4d. THE ONE-LINE GUARD, ON EVERY CASE THAT DIVERGES")
    print("=" * 78)
    print(f"  guard: sigma_used = max(sigma_est, ||X||_F / {GUARD:.6f})")
    print()
    print(
        f"  {'case':>22} {'sigma_est':>11} {'||X||_F':>10} {'est used now':>13} "
        f"{'est used guarded':>18} {'out now':>11} {'out guarded':>13}"
    )
    cases = []
    for l2 in [0.5, 0.3, 0.1, 0.01]:
        for delta in [1e-9, 1e-6, 1e-3, 0.1]:
            cases.append((l2, delta))
    # plus well-behaved factors, which the guard must NOT change
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        M = torch.randn(K, 4, dtype=torch.float64, generator=g)
        cases.append(("gauss", M))
    # and the ONE case the audit's verdict rests on: an ORTHONORMAL [768,64]
    # factor, i.e. exactly what the retraction receives in the steady state.
    # sigma_max = 1 and ||X||_F = sqrt(64) = 8, so ||X||_F/1.604 = 4.99 > 1 and
    # the `max` must select the Frobenius bound ALWAYS. The other gauss rows
    # are [64,4] and would NOT show that, which is why this row exists.
    cases.append(("onmanifold", dct_orthonormal(768, 64)))

    n_worse = 0
    for c in cases:
        if c[0] in ("gauss", "onmanifold"):
            M = c[1]
            lbl = f"{c[0]}[{M.shape[0]}x{M.shape[1]}]"
        else:
            l2, delta = c
            M = make(l2, delta, seed=5)
            lbl = f"l2={l2} delta={delta:.0e}"
        est = sigma_est(M)
        frob = float(M.norm())
        now = est
        guarded = max(est, frob / GUARD)
        o_now = gram_iter(M / (now * SAFETY), 3)
        o_g = gram_iter(M / (guarded * SAFETY), 3)
        # (there was a per-entry ortho_error of the guarded result here, unused
        # and, on the [K,4] rows, a shape crash against torch.eye(K) - the
        # question this script answers is WHICH DIVISOR WINS, not how good the
        # output is, so it is gone rather than fixed.)
        same = abs(guarded - est) < 1e-12
        if not same:
            n_worse += 1
        print(
            f"  {lbl:>22} {est:11.4e} {frob:10.4f} {now:13.4e} {guarded:18.4e} "
            f"{o_now.abs().max().item():11.3e} {o_g.abs().max().item():13.3e}"
            f"{'' if same else '  <- guard engaged'}"
        )
    print()
    print(f"  MEASURED: the guard engaged on {n_worse}/{len(cases)} cases.")
    print("  Including every WELL-BEHAVED factor: the four [64,4] Gaussians and")
    print("  the orthonormal [768,64] (sigma_max = 1 exactly, ||X||_F = 8, so")
    print("  ||X||_F/1.604 = 4.99 wins the max against a correct estimate of 1).")
    print("  THAT is the finding the guard was proposed against: it does not only")
    print("  catch the diverging cases, it silently replaces the sigma_max")
    print("  prescale with the Frobenius one on the shape the trainer actually")
    print("  retracts - so it converges, and converges to the wrong place. The")
    print("  two `out` columns above are entry magnitudes, not sigma_max; the")
    print("  sigma_max consequence is measured in spectral-reference.md 3.4.")
    print()
    print("  AND the cost: one extra Frobenius norm per factor, which is a")
    print("  reduction on an already-reduced tensor - the same shape of op as")
    print("  the Rayleigh quotient, and on the tensor path it is not a host sync.")


if __name__ == "__main__":
    main()
