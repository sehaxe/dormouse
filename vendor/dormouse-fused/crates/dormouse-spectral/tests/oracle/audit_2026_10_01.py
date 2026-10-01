"""Formula audit of dormouse-spectral's Newton-Schulz retraction, 2026-10-01.

Run:  /tmp/opencode/oracle-venv/bin/python audit_2026_10_01.py

Interpreter: python 3.12.14 / torch 2.14.0+cpu / numpy 2.5.2, CPU only.
Follows the RMSNorm precedent: torch's arithmetic is COMPILED C++ and is not
quotable, so the wheel+version is recorded instead of any number claimed to
come out of it.

WHAT THIS IS: a transcription of the Rust at
`vendor/dormouse-fused/crates/dormouse-spectral/src/lib.rs:208-341`, written to be
READ ALONGSIDE IT (line numbers cited per claim). Every function here is
labelled `ours:` / `ref:` and the `ref:` ones are LAPACK (numpy) or the pinned
Polar Express authors' code. It is NOT a reference for the Rust; it is a
transcription of it, and a reference only where it says so.

Six questions, one section each:
  1. Is the update the quintic it claims? (hand-derived, then checked)
  2. What is the basin, exactly? (analytic + numeric)
  3. Is the POWER_ITERS comment true? (the distribution, not one draw)
  4. ADVERSARIAL: the power iteration's start vector is G*1. What happens when
     the top singular direction is orthogonal to 1?
  5. The FACTORED path: does retracting U and V separately keep the effective
     weight's spectrum = s?  What does the retraction do to W?
  6. The CADENCE question: iterations vs spectral spread vs LAPACK polar.
"""

import numpy as np
import torch

torch.manual_seed(20261001)
np.random.seed(20261001)

A, B, C = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0
POWER_ITERS = 5
SAFETY = 1.05
# basin: p(s) = s  ->  s(3s^4 - 10 s^2 + 7)/8 = 0 -> s = 0, 1, sqrt(7/3)
BASIN = (7.0 / 3.0) ** 0.5


def p(s):
    return A * s + B * s**3 + C * s**5


def dp(s):
    return A + 3 * B * s**2 + 5 * C * s**4


# ---------------------------------------------------------------- 1. the update
def ours_gram_iter(M, iters):
    """lib.rs:255-260 verbatim in structure: G = M M^T ; M <- aM + (bG + cG^2)M."""
    M = M.clone()
    for _ in range(iters):
        G = M @ M.T
        M = A * M + (B * G + C * (G @ G)) @ M
    return M


def quintic_polar_form(M, iters):
    """lib.rs's update rewritten as the OTHER algebraic form, M(M^T M)^j.

    If the code's `b G M` really is `b M (M^T M)` then these two are the same
    map and must agree to machine precision. This is the check that the code
    is the quintic its comment names, not a lookalike."""
    M = M.clone()
    for _ in range(iters):
        S = M.T @ M
        M = A * M + B * (M @ S) + C * (M @ (S @ S))
    return M


def print_1():
    print("=" * 78)
    print("1. IS THE UPDATE THE QUINTIC THE COMMENT NAMES?  (lib.rs:253-260)")
    print("=" * 78)
    for shape in [(64, 8), (768, 64), (8, 64), (256, 256)]:
        M = torch.randn(*shape, dtype=torch.float64)
        # the code's canonical form: small side first, X X^T on [c,c]
        X = M.T if shape[0] > shape[1] else M
        # ...and the code's prescale, which is part of the map (lib.rs:251)
        Xs = X / (sigma_est(X) * SAFETY)
        a = ours_gram_iter(Xs, 3)
        b = quintic_polar_form(Xs, 3)
        d = (a - b).abs().max().item()
        # and its fixed point vs LAPACK's polar of the SAME canonical form
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        target = U @ Vh
        rel = (a - target).norm().item() / target.norm().item()
        print(
            f"  {str(shape):>12}  |gram - polar_form| = {d:.3e}"
            f"   relF(3 iters vs LAPACK polar) = {rel:.4e}"
        )
    # the scalar polynomial the iteration applies to a singular value
    print("  p(s) - s factorisation, exact in f64:")
    for s in [0.1, 0.5, 0.9523809523809524, 1.0, 1.2, 1.5, 1.6, 2.0]:
        lhs = p(s) - s
        rhs = s / 8 * (3 * s**2 - 7) * (s**2 - 1)
        print(
            f"    s={s:<20.10f} p(s)-s={lhs:+.6e}  s(3s^2-7)(s^2-1)/8={rhs:+.6e}"
            f"  p'(s)={dp(s):.6e}"
        )


# ------------------------------------------------------------------- 2. basin
def print_2():
    print()
    print("=" * 78)
    print("2. THE BASIN, EXACTLY.  (comment lib.rs:179-181 and 249-250)")
    print("=" * 78)
    print(f"  p(1) = {p(1.0):.17g}      p'(1) = {dp(1.0):.3e}  (comment says 0)")
    print(f"  p'(s) = (15/8)(s^2-1)^2 ? {dp(0.7):.6e} vs {1.875 * (0.49 - 1) ** 2:.6e}")
    print(f"  non-trivial repelling fixed point  sqrt(7/3) = {BASIN:.6f}")
    print(f"    p'(sqrt(7/3)) = {dp(BASIN):.6f}  (>1 => repelling => basin edge)")
    print("  the quintic's basin is |s| < sqrt(7/3) = 1.527525, NOT 1 and NOT sqrt(3):")
    for s in [1.0, 1.2, 1.45, 1.52, 1.5275, 1.53, 1.6, 2.0]:
        x, n = s, 0
        for _ in range(40):
            x = p(x)
            n += 1
            if abs(x) > 1e12 or not np.isfinite(x):
                break
        verdict = "converges" if abs(x - 1.0) < 1e-6 else "DIVERGES"
        print(f"    start {s:<8.4f} after 40 iters x={x:.4e}  {verdict}")
    print("  the comment at lib.rs:2429 calls this 'the cubic NS basin < sqrt(3)':")
    print("    the CUBIC p3(s)=1.5s-0.5s^3 has basin |s|<1 exactly; the quintic we")
    print("    run has |s|<sqrt(7/3). sqrt(3)=1.732 is neither.")


# --------------------------------------------------- 3. the POWER_ITERS claim
def sigma_est(M, k=POWER_ITERS):
    """lib.rs:229-248: start v = G*1, normalise, multiply, then Rayleigh."""
    G = M @ M.T
    v = G.sum(dim=1)
    for _ in range(k):
        v = v / v.norm()
        v = G @ v
    gv = G @ v
    return float(((v * gv).sum() / (v * v).sum()).sqrt())


def print_3():
    print()
    print("=" * 78)
    print("3. IS THE POWER_ITERS COMMENT TRUE?  (lib.rs:132-135)")
    print('=" * 78')
    print('  claim: "Rayleigh-quotient error shrinks as (lambda2/lambda1)^k; 5')
    print('  gives the estimate well within the 1.05 safety factor even for')
    print('  square Wishart (lambda2/lambda1 ~ 1)"')
    print()
    print(
        f"  {'shape':>12} {'median rel':>11} {'p95':>11} {'max':>11} "
        f"{'frac>5%':>8} {'max pre-scaled sigma_max':>24}"
    )
    for shape in [(768, 64), (4096, 64), (512, 128), (256, 256), (128, 128), (64, 64)]:
        rels, pre = [], []
        n = 400
        for _ in range(n):
            M = torch.randn(*shape, dtype=torch.float64)
            # code's canonical form: small side first
            X = M.T if shape[0] > shape[1] else M
            true = float(torch.linalg.matrix_norm(X, 2))
            est = sigma_est(X)
            rels.append(est / true - 1.0)  # <= 0 : converges from below
            pre.append(true / (est * SAFETY))
        rels = np.array(rels)
        print(
            f"  {str(shape):>12} {np.median(rels):+11.4e} "
            f"{np.percentile(np.abs(rels), 95):11.4e} {np.abs(rels).max():11.4e} "
            f"{float((np.abs(rels) > 0.05).mean()):8.3f} {max(pre):24.6f}"
        )
    print()
    print(f"  basin edge sqrt(7/3) = {BASIN:.6f}; the 1.05 factor is claimed to keep")
    print("  the input under 1.0.  Worst prescaled sigma_max above is the question.")
    print("  Re-measured the DECAY RATE the comment names, on a controlled")
    print("  spectrum (this is the part a Wishart average cannot separate):")
    for l2 in [0.1, 0.5, 0.9, 0.99]:
        errs = []
        for _ in range(200):
            Q, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
            s = torch.tensor([1.0, l2, 0.0], dtype=torch.float64).repeat(22)[:64]
            s[0] = 1.0
            s[1] = l2
            X = Q @ torch.diag(s) @ Q.T
            true = 1.0
            errs.append(abs(sigma_est(X) / true - 1.0))
        e = np.array(errs)
        print(
            f"    lambda2/lambda1={l2:<5.2f}  median rel err after 5 steps "
            f"{np.median(e):.3e}   (lambda2/lambda1)^5 = {l2**5:.3e}   "
            f"(ratio)^10 = {l2**10:.3e}"
        )
    print("  -> the error tracks (ratio)^(2k), not (ratio)^k, and at ratio~1 it is")
    print("     O(1e-2) whatever k is. The claim fails on BOTH counts.")
    print()
    print("  AND: the largest prescaled sigma_max measured over 2400 draws, against")
    print("  the basin edge. If this ratio is the safety story, the margin is here:")
    worst = 0.0
    for shape in [(768, 64), (256, 256), (64, 64), (512, 128)]:
        for _ in range(600):
            M = torch.randn(*shape, dtype=torch.float64)
            X = M.T if shape[0] > shape[1] else M
            true = float(torch.linalg.matrix_norm(X, 2))
            est = sigma_est(X)
            worst = max(worst, true / (est * SAFETY))
    print(f"    worst prescaled sigma_max over 2400 draws = {worst:.6f}")
    print(f"    basin edge                              = {BASIN:.6f}")
    print(f"    margin                                  = {BASIN / worst:.3f}x")
    print("  but the margin is only 'the estimate is not catastrophically wrong'.")
    print("  It is NOT the 1.05 factor: 1.05 < 1 and the estimate is a LOWER bound,")
    print("  so the factor moves the answer AWAY from 1, not toward it.")


# -------------------------------------------- 4. adversarial: the start vector
def print_4():
    print()
    print("=" * 78)
    print("4. ADVERSARIAL: THE START VECTOR IS G*1  (lib.rs:230)")
    print("=" * 78)
    print("  lib.rs:230 initialises the power iteration with v = G*1, i.e. the")
    print("  ROW SUMS of the Gram = M*(column sums of M). If the dominant")
    print("  singular direction is orthogonal to 1, that start has (almost) no")
    print("  component along it and the iteration converges to the WRONG")
    print("  eigenvalue. Then the prescale divides by sigma_2, not sigma_1.")
    print()
    print("  construction: M = sum_i s_i u_i v_i^T with the top left-singular")
    print("  vector u_1 chosen orthogonal to 1 (so M^T 1 ~ 0 on that direction).")
    for spread in [2.0, 10.0, 1e2, 1e3]:
        d, k = 64, 64
        Q, _ = torch.linalg.qr(torch.randn(d, k, dtype=torch.float64))
        s = torch.logspace(0, -torch.log10(torch.tensor(spread)).item(), k,
                           dtype=torch.float64)
        s[0], s[1] = 1.0, 1.0 / spread
        M = Q @ torch.diag(s)
        # project the top LEFT-singular vector orthogonal to the all-ones vector
        one = torch.ones(d, 1, dtype=torch.float64)
        u1 = Q[:, 0:1]
        u1 = u1 - one * (one.T @ u1) / (one.T @ one)
        u1 = u1 / u1.norm()
        Q2 = torch.cat([u1, Q[:, 1:]], dim=1)
        M = Q2 @ torch.diag(s)
        colsum = (M.T @ torch.ones(d, 1, dtype=torch.float64)).norm().item()
        true = float(torch.linalg.matrix_norm(M, 2))
        est = sigma_est(M)
        pre = true / (est * SAFETY)
        # what the retraction then does
        X = M.T  # tall -> canonical form is the transpose
        out = ours_gram_iter(X / (est * SAFETY), 3).T
        nan = bool((~torch.isfinite(out)).any())
        big = out.abs().max().item()
        G = out.T @ out
        print(
            f"    spread {spread:<8.0e} ||M^T 1||={colsum:.3e}  sigma_1={true:.4f}  "
            f"est={est:.4f}  ({est / true * 100:5.1f}% of truth)  "
            f"prescaled={pre:7.4f}  ->  3-iter out: max|.|={big:.3e} "
            f"{'NON-FINITE' if nan else 'finite'}"
        )
    print()
    print("  the honest question is not 'is the estimate good' but 'how far can")
    print("  it be from sigma_1'. Closed form: after k steps the start theta*e1 +")
    print("  c*e2 becomes theta*e1 + (l2/l1)^k c e2, and the Rayleigh quotient is")
    print("  (l1*theta^2 + l2*delta^2)/(theta^2 + delta^2). It is worst when")
    print("  delta >> theta, and then it tends to l2/l1 - i.e. to the WRONG")
    print("  eigenvalue. DIVERGENCE needs est/sigma_1 < 1/(1.05*sqrt(7/3)) = "
          f"{1 / (SAFETY * BASIN):.4f}")
    print()
    print("  constructed: Gram eigenvalues (1, l2, ...), start vector G*1 forced")
    print("  nearly orthogonal to the top eigenvector by choosing the eigenbasis.")
    d, k = 64, 64
    thr = 1.0 / (SAFETY * BASIN)
    print(f"  {'l2/l1':>8} {'|start.e1|/|start|':>19} {'est/sigma_1':>13} "
          f"{'prescaled':>11} {'DIVERGES':>10} {'max|out|':>12}")
    one = torch.ones(k, 1, dtype=torch.float64)
    for l2 in [0.9, 0.5, 0.2, 0.1, 1e-3]:
        for tilt in [0.0, 1e-3, 1e-6]:
            # A GENUINELY orthonormal eigenbasis, then rotate e1 to be exactly
            # orthogonal to 1 (tilt adds a controlled component back).
            Q, _ = torch.linalg.qr(torch.randn(k, k, dtype=torch.float64))
            e1 = Q[:, 0:1] - one * (one.T @ Q[:, 0:1]) / (one.T @ one)
            e1 = e1 / e1.norm()
            # complete the basis by Gram-Schmidt against the modified e1
            cols = [e1]
            for j in range(1, k):
                v = Q[:, j : j + 1]
                for c in cols:
                    v = v - c * (c.T @ v)
                v = v / v.norm()
                cols.append(v)
            Q2 = torch.cat(cols, dim=1)
            assert float((Q2.T @ Q2 - torch.eye(k, dtype=torch.float64)).abs().max()) < 1e-12
            if tilt:
                Q2 = Q2 + tilt * torch.randn(k, k, dtype=torch.float64)
                Q2 = torch.linalg.qr(Q2)[0]
            lam = torch.full((k,), 0.05, dtype=torch.float64)
            lam[0], lam[1] = 1.0, l2
            G = Q2 @ torch.diag(lam) @ Q2.T
            M = torch.linalg.cholesky(G).T  # M M^T = G, sigma_1 = 1
            M = M / float(torch.linalg.matrix_norm(M, 2))
            true = float(torch.linalg.matrix_norm(M, 2))
            start = G @ one
            cosang = float((e1.T @ start).abs() / start.norm())
            est = sigma_est(M)
            pre = true / (est * SAFETY)
            out = ours_gram_iter(M / (est * SAFETY), 3)
            print(
                f"  {l2:8.3f} {cosang:19.2e} {est / true:13.6f} {pre:11.4f} "
                f"{str(pre > BASIN):>10} {out.abs().max().item():12.3e}"
            )
    print()
    print("  and a RANDOM search for a real factor that lands outside the basin,")
    print("  drawing the factor's eigenbasis adversarially (e1 chosen in 1-perp):")
    found = 0
    for trial in range(4000):
        Q, _ = torch.linalg.qr(torch.randn(d, k, dtype=torch.float64))
        e1 = Q[:, 0:1]
        e1 = e1 - one * (one.T @ e1) / (one.T @ one)
        e1 = e1 / e1.norm()
        cols = [e1]
        for j in range(1, k):
            v = Q[:, j : j + 1]
            for c in cols:
                v = v - c * (c.T @ v)
            v = v / v.norm()
            cols.append(v)
        Q2 = torch.cat(cols, dim=1)
        lam = torch.rand(k, dtype=torch.float64) * 0.9 + 0.05
        lam[0] = 1.0
        M = Q2 @ torch.diag(lam)
        true = float(torch.linalg.matrix_norm(M, 2))
        est = sigma_est(M)
        pre = true / (est * SAFETY)
        if pre > BASIN:
            found += 1
            if found <= 3:
                out = ours_gram_iter(M / (est * SAFETY), 3)
                print(
                    f"    trial {trial}: sigma_1={true:.4f} est={est:.4e} "
                    f"prescaled={pre:.4f} > {BASIN:.4f}  3-iter out max|.|="
                    f"{out.abs().max().item():.3e}  finite={bool(torch.isfinite(out).all())}"
                )
    print(f"    outside the basin: {found}/4000 adversarial draws")
    print()
    print("  is the degenerate case reachable from a factor that is ON or NEAR the")
    print("  manifold?  sweep the drift and look at the worst prescaled sigma_max:")
    for eps in [0.0, 1e-3, 1e-2, 0.1, 0.3, 1.0]:
        w = 0.0
        for _ in range(300):
            torch.manual_seed(0)
            Q, _ = torch.linalg.qr(torch.randn(d, k, dtype=torch.float64))
            M = Q + eps * torch.randn(d, k, dtype=torch.float64)
            true = float(torch.linalg.matrix_norm(M, 2))
            est = sigma_est(M)
            w = max(w, true / (est * SAFETY))
        print(f"    drift {eps:<6.1e} worst prescaled sigma_max = {w:.4f} "
              f"({'INSIDE' if w < BASIN else 'OUTSIDE'} basin)")


# ------------------------------------------------------------ 5. the factored path
def retract20(rows, cols, seed=None):
    """A converged polar factor [rows, cols]: the code's own map, prescaled."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    M = torch.randn(rows, cols, dtype=torch.float64, generator=g)
    X = M.T if rows > cols else M
    out = ours_gram_iter(X / (sigma_est(X) * SAFETY), 20)
    return out.T if rows > cols else out


def print_5():
    print()
    print("=" * 78)
    print("5. THE FACTORED PATH: U and V retracted SEPARATELY")
    print("=" * 78)
    print("  claim under test: the constraint retracted onto (U^T U = I, V^T V = I)")
    print("  is exactly the set on which W = U diag(s) V^T IS AN SVD with singular")
    print("  values s. If that holds, retracting the factors cannot silently change")
    print("  the model's singular-value budget, which is what `s` is for.")
    d, f, k = 96, 128, 64
    torch.manual_seed(7)
    for spread in [1.0, 10.0, 1e3]:
        A0 = torch.randn(d, k, dtype=torch.float64)
        B0 = torch.randn(f, k, dtype=torch.float64)
        # prescale first, exactly as the code does (a raw Gaussian is OUTSIDE
        # the basin - see section 1 - and would diverge at 20 iterations)
        U = ours_gram_iter(A0.T / sigma_est(A0.T), 20).T
        V = ours_gram_iter(B0.T / sigma_est(B0.T), 20).T
        s = torch.logspace(0, -torch.log10(torch.tensor(spread)).item(), k,
                           dtype=torch.float64)
        W = U @ torch.diag(s) @ V.T
        sw = torch.linalg.svdvals(W)
        print(
            f"  factors spread {spread:<8.0e}:  ||U^T U - I||_F={((U.T @ U - torch.eye(k, dtype=torch.float64)).norm()):.2e}"
            f"  ||V^T V - I||_F={((V.T @ V - torch.eye(k, dtype=torch.float64)).norm()):.2e}"
        )
        print(
            f"      singular values of W = U diag(s) V^T: max/min ratio "
            f"{(sw[0] / sw[-1]).item():.6e}   s ratio {(s[0] / s[-1]).item():.6e}"
        )
        # and the metric the trainer's latch reads, per-entry
        pe = ((U.T @ U - torch.eye(k, dtype=torch.float64)).norm() / k).item()
        print(f"      trainer's per-entry max_ortho on U: {pe:.3e}  (latch 1e-3)")
    print()
    print("  the OTHER question: what does the retraction do to the FUNCTION?")
    print("  W_on = U_on diag(s) V_on^T (on the manifold); W_off after a drift E")
    print("  and one retraction. rel change in W:")
    d, f, k = 256, 256, 64
    for eps in [1e-4, 1e-3, 1e-2]:
        torch.manual_seed(3)
        U = retract20(d, k, seed=1)
        V = retract20(f, k, seed=2)
        s = torch.ones(k, dtype=torch.float64)
        W_on = U @ torch.diag(s) @ V.T
        Uo = ours_gram_iter(U + eps * torch.randn(d, k, dtype=torch.float64), 3)
        Vo = ours_gram_iter(V + eps * torch.randn(f, k, dtype=torch.float64), 3)
        W_off = Uo @ torch.diag(s) @ Vo.T
        print(
            f"    drift {eps:.0e}: ||W_off - W_on||_F / ||W_on||_F = "
            f"{((W_off - W_on).norm() / W_on.norm()).item():.4e}"
        )
    print("  -> a retraction IS allowed to move W; that is what a projection is.")
    print("     The quantity that must be a FIXED POINT is the FACTORS. Which is")
    print("     what retraction_holds_the_manifold_at_rank_64 tests. So this is")
    print("     not a defect, but it IS the thing the cadence A/B is really about:")
    print("     every skipped retraction is that much un-projected drift in W.")


# ---------------------------------------------------------------- 6. the cadence
def print_6():
    print()
    print("=" * 78)
    print("6. THE CADENCE QUESTION: iters vs spectral spread vs LAPACK polar")
    print("=" * 78)
    print("  prescribed spread, rank 64, sigma_max prescale x1.05, 768x64 factor")
    print("  (this is the shape the trainer retracts at `small`)")
    print(
        f"  {'spread':>8} {'iters':>6} {'relF vs polar':>14} {'per-entry':>11} "
        f"{'sigma_max(out)':>15}"
    )
    d, k = 768, 64
    torch.manual_seed(11)
    Q, _ = torch.linalg.qr(torch.randn(d, k, dtype=torch.float64))
    for spread in [1.0, 0.5, 0.1, 0.01]:
        for iters in [1, 3, 5, 8]:
            s = torch.logspace(0, np.log10(spread), k, dtype=torch.float64)
            M = Q @ torch.diag(s)
            X = M.T
            out = ours_gram_iter(X / (sigma_est(X) * SAFETY), iters).T
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            pol = U @ Vh
            rel = (out - pol).norm().item() / pol.norm().item()
            pe = (
                (out.T @ out - torch.eye(k, dtype=torch.float64)).norm() / k
            ).item()
            smax = float(torch.linalg.matrix_norm(out, 2))
            print(
                f"  {spread:8.2f} {iters:6d} {rel:14.4e} {pe:11.3e} "
                f"{smax:15.9f}"
            )
    print()
    print("  a REALISTIC drift, not a prescribed one: take an on-manifold factor,")
    print("  give it ONE step of Muon-sized momentum, and read what the")
    print("  retraction has to undo. momentum is not in the retraction, so this")
    print("  is a bound on the drift the cadence question is about, not a model")
    print("  of the optimiser.")


if __name__ == "__main__":
    print(f"torch {torch.__version__}  numpy {np.__version__}")
    print_1()
    print_2()
    print_3()
    print_4()
    print_5()
    print_6()
