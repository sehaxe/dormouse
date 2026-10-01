"""RUN the pinned Polar Express reference -- both stages, their code, unedited.

Stage 1 (offline)  : polar_express.optimal_composition  -> the coefficient list
Stage 2 (online)   : polar_express.PolarExpress         -> their torch kernel

Compared against the transcription of our own procedure
(burn-spectral/src/lib.rs:208-266) on the SAME matrices.

Run: /tmp/opencode/oracle-venv/bin/python run_polar_express.py
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "upstream")
import polar_express as pe

A, B, C = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0
rng = np.random.default_rng(20260930)


def orient(X):
    return (X.T, True) if X.shape[0] > X.shape[1] else (X, False)


def sigma_max_powerit(M):
    g = M @ M.T
    v = g.sum(axis=1)
    for _ in range(5):
        v = v / max(np.sqrt((v * v).sum()), 1e-12)
        v = g @ v
    gv = g @ v
    return np.sqrt((v * gv).sum() / max((v * v).sum(), 1e-14))


def fro(M):
    return np.sqrt((M * M).sum())


def ours(X, iters):
    """Transcription of burn-spectral polar_orthogonalize: sigma_max prescale
    x1.05, then `iters` applications of the FIXED triple."""
    M, tr = orient(X)
    M = M / (sigma_max_powerit(M) * 1.05)
    for _ in range(iters):
        xx = M @ M.T
        M = A * M + (B * xx + C * (xx @ xx)) @ M
    return M.T if tr else M


def schedule_numpy(X, steps, coeffs):
    """The online loop of polar_express.PolarExpress, in numpy, so it can be
    compared against their torch result on the same input."""
    M, tr = orient(X)
    M = M / (fro(M) * 1.01 + 1e-7)
    from itertools import repeat
    hs = list(coeffs[:steps]) + list(repeat(coeffs[-1], max(0, steps - len(coeffs))))
    for a, b, c in hs:
        Y = M @ M.T
        M = a * M + (b * Y + c * (Y @ Y)) @ M
    return M.T if tr else M


def per_entry(U):
    k = U.shape[1]
    return np.linalg.norm(U.T @ U - np.eye(k), "fro") / k


def on_manifold(rows, cols, rank):
    P = np.linalg.qr(rng.standard_normal((rows, rank)))[0]
    Q = np.linalg.qr(rng.standard_normal((cols, rank)))[0]
    return P @ Q.T


print("=" * 78)
print("STAGE 1 -- their offline stage, module default (as shipped at HEAD)")
print("=" * 78)
for i, c in enumerate(pe.coeffs_list):
    print("  t=%-2d (%.17g, %.17g, %.17g)" % (i + 1, c[0], c[1], c[2]))
print("  note t=1 coefficients are large; the terminal triple is the dyadic")
print("  (15/8, -10/8, 3/8).  exact in binary: %s"
      % all(x == y for x, y in zip(pe.coeffs_list[-1], (1.875, -1.25, 0.375))))

PAPER8 = [(8.28721201814563, -23.595886519098837, 17.300387312530933),
          (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
          (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
          (3.3184196573706015, -2.488488024314874, 0.51004894012372),
          (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
          (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
          (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
          (1.875, -1.25, 0.375)]
print("\n  HEAD list vs the list PRINTED in 2505.16932 App. A / 2602.21545v3 D.3:")
for i in range(8):
    g, p = pe.coeffs_list[i], PAPER8[i]
    print("    t=%d  head=%-22.15g paper=%-22.15g  reldiff=%.2e"
          % (i + 1, g[0], p[0], abs(g[0] - p[0]) / abs(p[0])))

print("\n  PROOF the terminal triple is the analytic dyadic branch, not a decimal:")
for l in (1.0, 0.999999, 0.999995, 0.99999, 0.999, 0.9):
    q = pe.optimal_quintic(l, 1.0)
    print("    optimal_quintic(l=%-9g, u=1) -> (%.17g, %.17g, %.17g)  == (15/8,-10/8,3/8): %s"
          % (l, q[0], q[1], q[2], q == (1.875, -1.25, 0.375)))

print("\n" + "=" * 78)
print("STAGE 2 -- their ONLINE stage, RUN on this box")
print("=" * 78)
for shape in [(768, 64), (64, 64), (512, 64)]:
    X = on_manifold(*shape, 64)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    exact = U @ Vt
    print("\n--- %dx%d ON the manifold, rank 64 (sigma_max=1, fro=%.4f) ---"
          % (shape[0], shape[1], fro(X)))
    for steps in (3, 5):
        ref = pe.PolarExpress(torch.tensor(X), steps).float().numpy()  # HEAD returns bf16
        nps = schedule_numpy(X, steps, pe.coeffs_list)
        print("  PolarExpress(steps=%d)  per-entry||UtU-I||=%.4e  sig_max=%.9f  "
              "||out-UVt||_F=%.3e   [torch vs numpy %.2e]"
              % (steps, per_entry(ref), np.linalg.svd(ref, compute_uv=False)[0],
                 np.linalg.norm(ref - exact, "fro"), np.abs(ref - nps).max()))
        o = ours(X, steps)
        print("  OURS  (fixed triple)    per-entry||UtU-I||=%.4e  sig_max=%.9f  "
              "||out-UVt||_F=%.3e"
              % (per_entry(o), np.linalg.svd(o, compute_uv=False)[0],
                 np.linalg.norm(o - exact, "fro")))

# an off-manifold, realistic input: a Wishart gradient
print("\n" + "=" * 78)
print("OFF-manifold input (Wishart 768x64, i.e. what a real gradient looks like)")
print("=" * 78)
W = rng.standard_normal((768, 64))
U, s, Vt = np.linalg.svd(W, full_matrices=False)
exact = U @ Vt
print("  sigma_max=%.6f sigma_min=%.3e fro=%.4f fro/sig_max=%.4f"
      % (s[0], s[-1], fro(W), fro(W) / s[0]))
for steps in (3, 5, 8):
    ref = pe.PolarExpress(torch.tensor(W), steps).float().numpy()
    o = ours(W, steps)
    print("  steps=%d  PolarExpress per-entry=%.4e sig_max=%.9f | OURS per-entry=%.4e sig_max=%.9f"
          % (steps, per_entry(ref), np.linalg.svd(ref, compute_uv=False)[0],
             per_entry(o), np.linalg.svd(o, compute_uv=False)[0]))
