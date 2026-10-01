"""Pin the two findings the LAPACK reference produced, so the document quotes
measurements rather than impressions.

FINDING 1: the 5-step power iteration UNDERESTIMATES sigma_max by up to ~7% on
          a Wishart matrix -- outside the 1.05 safety factor that
          burn-spectral/src/lib.rs:249-251 relies on. The doc's own parenthetical
          ("even for square Wishart, lambda2/lambda1 ~ 1") is the reason: the
          Rayleigh-quotient error decays as (lambda2/lambda1)^(2k), which is
          SLOW exactly in the case the comment claims to cover.

FINDING 2: the 3-iteration retraction is 27% away from LAPACK's polar(X) on a
          full-rank 256x256 input, and ~1e-8 on the low-rank TSCT-shaped inputs.
          The number of iterations is not a free parameter; it is set by the
          input's spectral spread.

Run: /tmp/opencode/oracle-venv/bin/python pin_findings.py
"""
import numpy as np

rng = np.random.default_rng(20260930)


def sigma_max_powerit(M, iters):
    g = M @ M.T
    v = g.sum(axis=1)
    for _ in range(iters):
        v = v / max(np.sqrt((v * v).sum()), 1e-12)
        v = g @ v
    gv = g @ v
    return np.sqrt((v * gv).sum() / max((v * v).sum(), 1e-14))


print("=" * 84)
print("FINDING 1 -- power-iteration sigma_max, ours (5 steps) vs LAPACK dgesdd")
print("=" * 84)
print("  power iteration converges from BELOW, so the estimate is a lower bound")
print("  and the sign of the error is what makes the 1.05 factor load-bearing.")
print()
print("  %-12s %-6s %-14s %-14s %-10s %-12s" %
      ("shape", "steps", "ours", "LAPACK", "rel err", "prescaled sigmax"))
for shape in [(768, 64), (256, 256), (512, 128), (4096, 64)]:
    G = rng.standard_normal(shape)
    M = G.T.copy() if shape[0] > shape[1] else G.copy()
    exact = np.linalg.norm(M, 2)
    for steps in (1, 2, 5, 8, 12, 20):
        e = sigma_max_powerit(M, steps)
        rel = (exact - e) / exact           # positive => underestimate
        print("  %-12s %-6d %-14.9f %-14.9f %-10.2e %-12.9f" %
              (str(shape), steps, e, exact, rel, exact / (e * 1.05)))
    print("  %-12s %-6s %-14s %-14s %-10s %-12s" % ("", "", "", "", "", ""))
print("  'prescaled sigmax' = LAPACK sigma_max / (ours * 1.05). Anything > 1.0 is")
print("  OUTSIDE the basin [0,1] of p(s) = 15/8 s - 5/4 s^3 + 3/8 s^5, i.e. the")
print("  1.05 safety factor did not do its job at 5 steps on these inputs.")

print("\n" + "=" * 84)
print("FINDING 2 -- distance from the retraction to LAPACK polar(X), 3 iterations")
print("=" * 84)
print("  %-12s %-6s %-9s %-14s %-16s" % ("shape", "rank", "spec", "rel F error", "per-entry UtU-I"))
for shape, k in [((768, 64), 64), ((64, 768), 64), ((512, 128), 128), ((256, 256), 256),
                 ((256, 256), 32), ((1024, 64), 64), ((4096, 64), 64)]:
    P = np.linalg.qr(rng.standard_normal((shape[0], k)))[0]
    Q = np.linalg.qr(rng.standard_normal((shape[1], k)))[0]
    s = np.logspace(0, -np.log10(max(1.0, (min(shape) / k) ** 2)), k)
    G = P @ np.diag(s) @ Q.T
    M = G.T.copy() if shape[0] > shape[1] else G.copy()
    U, sg, Vt = np.linalg.svd(G, full_matrices=False)
    ref = U @ Vt
    est = sigma_max_powerit(M, 5)
    for iters in (3, 5):
        Mr = M / (est * 1.05)
        for _ in range(iters):
            xx = Mr @ Mr.T
            Mr = 1.875 * Mr + (-1.25 * xx + 0.375 * (xx @ xx)) @ Mr
        out = Mr.T if shape[0] > shape[1] else Mr
        rel = np.linalg.norm(out - ref, "fro") / np.linalg.norm(ref, "fro")
        pe = np.linalg.norm(out.T @ out - np.eye(out.shape[1]), "fro") / out.shape[1]
        print("  %-12s %-6d %-9d %-14.4e %-16.4e  (iters=%d)"
              % (str(shape), k, int(round(min(shape) / k)), rel, pe, iters))
print("  'spec' = full rank when rank == min(shape). A full-rank input is the")
print("  case the fixed triple cannot flatten in 3 steps; the low-rank TSCT")
print("  factors (rank 64 of 768) are the case it was chosen for.")
