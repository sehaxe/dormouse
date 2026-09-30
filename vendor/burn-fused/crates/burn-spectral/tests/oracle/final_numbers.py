"""Two corrections to the previous run, then the final numbers.

FIX A: the per-entry orthonormality metric must be applied on the side the
       factor spans. A tall [64,768] factor has orthonormal COLUMNS, so the
       metric is ||X X^T - I_64||, not ||X^T X - I_768||. The previous run
       printed 3.45e-2 for a result whose distance to LAPACK's polar was
       1.15e-15 -- the metric was misapplied, not the retraction.

FIX B: the power-iteration error is a RANDOM VARIABLE, so one draw is not a
       measurement. Sampled over 200 independent Wishart matrices per shape.

Run: /tmp/opencode/oracle-venv/bin/python final_numbers.py
"""
import numpy as np

A, B, C = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0


def sigma_max_powerit(M, iters=5):
    g = M @ M.T
    v = g.sum(axis=1)
    for _ in range(iters):
        v = v / max(np.sqrt((v * v).sum()), 1e-12)
        v = g @ v
    gv = g @ v
    return np.sqrt((v * gv).sum() / max((v * v).sum(), 1e-14))


def retract(M, iters=3):
    R = M / (sigma_max_powerit(M) * 1.05)
    for _ in range(iters):
        xx = R @ R.T
        R = A * R + (B * xx + C * (xx @ xx)) @ R
    return R


def ortho_on_span(X):
    """Correct on either shape: measure on the side the factor spans."""
    rows, cols = X.shape
    if rows >= cols:
        G = X.T @ X
        k = cols
    else:
        G = X @ X.T
        k = rows
    return np.linalg.norm(G - np.eye(k), "fro") / k


print("=" * 88)
print("FINDING 1 (fixed) -- power-iteration sigma_max, 5 steps, vs LAPACK dgesdd")
print("200 independent draws per shape. Underestimate is positive by construction.")
print("=" * 88)
print("  %-12s %-11s %-11s %-11s %-11s %-13s" %
      ("shape", "med rel", "p95 rel", "MAX rel", "frac>5%", "max prescaled"))
for shape in [(768, 64), (4096, 64), (512, 128), (256, 256), (128, 128), (64, 64)]:
    rels, pres = [], []
    for _ in range(200):
        G = np.random.default_rng(20260930 + _ * 7919).standard_normal(shape)
        M = G.T.copy() if shape[0] > shape[1] else G.copy()
        exact = np.linalg.norm(M, 2)
        e = sigma_max_powerit(M)
        rels.append((exact - e) / exact)
        pres.append(exact / (e * 1.05))
    r = np.array(rels)
    print("  %-12s %-11.3e %-11.3e %-11.3e %-11.3f %-13.6f" %
          (str(shape), np.median(r), np.percentile(r, 95), r.max(),
           (r > 0.05).mean(), max(pres)))
print("  'frac>5%' is the fraction of draws where the 1.05 safety factor does NOT")
print("  cover the error, i.e. where the prescaled sigma_max lands above 1 and")
print("  strictly outside the basin of p on [0,1].")

print("\n" + "=" * 88)
print("FINDING 2 (fixed) -- retraction vs LAPACK polar, as a function of the")
print("input's spectral spread. 768x64 rank 64, sigma spread set explicitly.")
print("=" * 88)
print("  %-14s %-9s %-16s %-18s %-18s" %
      ("sigma_min/max", "iters=3", "rel F vs LAPACK", "per-entry (correct)", "sig_max(out)"))
for spread in (1.0, 0.5, 0.1, 0.01, 1e-3, 1e-4):
    s = np.logspace(0, np.log10(spread), 64)
    P = np.linalg.qr(np.random.default_rng(20260930).standard_normal((768, 64)))[0]
    Q = np.linalg.qr(np.random.default_rng(7).standard_normal((64, 64)))[0]
    G = P @ np.diag(s) @ Q.T
    U, sg, Vt = np.linalg.svd(G, full_matrices=False)
    ref = U @ Vt
    out = retract(G, 3)
    rel = np.linalg.norm(out - ref, "fro") / np.linalg.norm(ref, "fro")
    print("  %-14.0e %-9d %-16.4e %-18.4e %-18.9f" %
          (spread, 3, rel, ortho_on_span(out), np.linalg.norm(out, 2)))
print("  A TSCT factor is 768x64 with rank 64 and arrives on the manifold, so the")
print("  top row is the case the retraction is actually for; the rest is the")
print("  regime where the fixed terminal triple stops being enough, and where the")
print("  Polar Express schedule would be the right answer instead.")
