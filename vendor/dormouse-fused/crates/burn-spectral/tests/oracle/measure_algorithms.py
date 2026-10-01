"""Measure, in float64, three orthogonalisation procedures. NOTHING here is a
golden value -- every number is printed by running this file.

  A. our algorithm, transcribed from burn-spectral/src/lib.rs:208-266
       (power-iteration sigma_max prescale x1.05, then `iters` applications of
        the FIXED triple (15/8, -5/4, 3/8))
  B. the same, with the Frobenius prescale instead of the power iteration
  C. Polar Express, the schedule the paper's reference implementation uses

Run:  /tmp/opencode/oracle-venv/bin/python measure_algorithms.py
"""
import numpy as np
from math import sqrt

rng = np.random.default_rng(20260930)
POWER_ITERS = 5
SAFETY = 1.05
A, B, C = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0


def polar_exact(X):
    """The answer the iteration is approximating, from numpy's SVD."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ Vt, s


def orient(X):
    if X.shape[0] > X.shape[1]:
        return X.T, True
    return X, False


def sigma_max_powerit(M):
    """Transcription of burn-spectral lib.rs:229-248."""
    g = M @ M.T
    v = g.sum(axis=1)
    for _ in range(POWER_ITERS):
        v = v / max(sqrt((v * v).sum()), 1e-12)
        v = g @ v
    gv = g @ v
    vgv = (v * gv).sum()
    vv = (v * v).sum()
    return sqrt(max(vgv / max(vv, 1e-14), 0.0)) if vv > 0 else 1.0


def fixed_triple(M, iters):
    """Transcription of burn-spectral lib.rs:254-260."""
    for _ in range(iters):
        xx = M @ M.T
        xx2 = xx @ xx
        M = A * M + (B * xx + C * xx2) @ M
    return M


def ortho_per_entry(U):
    k = U.shape[1]
    return np.linalg.norm(U.T @ U - np.eye(k), "fro") / k


def fro(M):
    return np.sqrt((M * M).sum())


def report(name, out, exact, s0, X):
    k = out.shape[1]
    s_out = np.linalg.svd(out, compute_uv=False)
    # sigma_max of the OUTPUT: is it at the fixed point 1?
    print("  %-34s per-entry||UtU-I||=%.4e  ||out-U Vt||_F=%.4e  sig_max(out)=%.9f"
          % (name, ortho_per_entry(out), np.linalg.norm(out - exact, "fro"), s_out[0]))
    return s_out[0]


def on_manifold(rows, cols, rank):
    """A factor that starts EXACTLY on the manifold: U @ Vt with U,Vt orth."""
    A0 = np.linalg.qr(rng.standard_normal((rows, rank)))[0]
    B0 = np.linalg.qr(rng.standard_normal((cols, rank)))[0]
    return A0 @ B0.T


for (rows, cols, iters) in [(768, 64, 3), (768, 64, 5), (512, 64, 3), (64, 64, 3), (4096, 64, 3)]:
    X0 = on_manifold(rows, cols, 64)
    exact, s0 = polar_exact(X0)
    print("\n=== %dx%d on-manifold (rank 64), %d iterations ===" % (rows, cols, iters))
    print("  sigma_max(X0)=%.9f  sigma_min(X0)=%.3e  fro=%.6f  fro/sig_max=%.6f"
          % (s0[0], s0[-1], fro(X0), fro(X0) / s0[0]))

    M, tr = orient(X0)
    sig = sigma_max_powerit(M)
    print("  power-iteration sigma_max estimate: %.9f  (rel err %.2e)" % (sig, abs(sig - s0[0]) / s0[0]))

    a = fixed_triple(M / (sig * SAFETY), iters)
    report("A: sigma_max prescale, fixed triple", a.T if tr else a, exact, s0, X0)

    b = fixed_triple(M / (fro(M) * SAFETY), iters)
    report("B: Frobenius prescale, fixed triple", b.T if tr else b, exact, s0, X0)

    c = fixed_triple(M / (fro(M) * 1.01), iters)
    report("B2: Frobenius 1.01, fixed triple", c.T if tr else c, exact, s0, X0)
