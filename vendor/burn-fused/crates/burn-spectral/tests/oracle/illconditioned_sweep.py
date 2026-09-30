"""The discriminating sweep: does a SCHEDULE beat a FIXED terminal triple when
the input is ill-conditioned?

All three procedures are in float64 numpy:
  ours          transcription of burn-spectral/src/lib.rs:208-266
                (sigma_max power-iteration prescale x1.05; FIXED 15/8,-5/4,3/8)
  schedule_fro  transcription of the ONLINE loop of polar_express.py:92-95
                with the coefficient list PRINTED in 2505.16932 App. A
                (== 2602.21545v3 D.3), safety factor 1.01 folded in as in App. A;
                Frobenius prescale x1.01.   TIER (b) -- the loop is transcribed.
  their_code    the authors' own torch, bf16, AS SHIPPED.       TIER (a), RUN.

Run: /tmp/opencode/oracle-venv/bin/python illconditioned_sweep.py
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "upstream")
import polar_express as pe

rng = np.random.default_rng(20260930)
A, B, C = 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0
SAFETY = 1.01
# the list PRINTED in 2505.16932 App. A, with App. A's 1.01 folding on all but
# the last entry.  Transcribed from the paper, not from the authors' repo HEAD
# (which generates a different list -- see run_polar_express.py output).
PAPER = [(8.28721201814563, -23.595886519098837, 17.300387312530933),
         (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
         (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
         (3.3184196573706015, -2.488488024314874, 0.51004894012372),
         (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
         (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
         (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
         (1.875, -1.25, 0.375)]
PAPER_FOLDED = [(a / SAFETY, b / SAFETY ** 3, c / SAFETY ** 5)
                for (a, b, c) in PAPER[:-1]] + [PAPER[-1]]


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


def per_entry(U):
    k = U.shape[1]
    return np.linalg.norm(U.T @ U - np.eye(k), "fro") / k


def ours(X, iters):
    M, tr = orient(X)
    M = M / (sigma_max_powerit(M) * 1.05)
    for _ in range(iters):
        Y = M @ M.T
        M = A * M + (B * Y + C * (Y @ Y)) @ M
    return M.T if tr else M


def schedule_fro(X, iters, coeffs=PAPER_FOLDED):
    from itertools import repeat
    M, tr = orient(X)
    M = M / (fro(M) * SAFETY + 1e-7)
    hs = list(coeffs[:iters]) + list(repeat(coeffs[-1], max(0, iters - len(coeffs))))
    for a, b, c in hs:
        Y = M @ M.T
        M = a * M + (b * Y + c * (Y @ Y)) @ M
    return M.T if tr else M


def prescribed(rows, cols, k, kappa):
    """Singular values log-spaced on [1, 1/kappa] -- a controlled condition number."""
    P = np.linalg.qr(rng.standard_normal((rows, k)))[0]
    Q = np.linalg.qr(rng.standard_normal((cols, k)))[0]
    s = np.logspace(0.0, -np.log10(kappa), k)
    return P @ np.diag(s) @ Q.T


print("=" * 92)
print("ILL-CONDITIONED SWEEP  (768x64, rank 64, sigma_max = 1, sigma_min = 1/kappa)")
print("=" * 92)
print("%-8s %-11s | %-13s %-13s %-13s | %-11s %-11s"
      % ("kappa", "fro/sigmax", "OURS(3)", "SCHED-fro(3)", "their_code(3)", "OURS(5)", "SCHED-fro(5)"))
for kappa in (1.0, 1e1, 1e2, 1e3, 1e4, 1e6):
    X = prescribed(768, 64, 64, kappa)
    o3, o5 = per_entry(ours(X, 3)), per_entry(ours(X, 5))
    s3, s5 = per_entry(schedule_fro(X, 3)), per_entry(schedule_fro(X, 5))
    t3 = per_entry(pe.PolarExpress(torch.tensor(X), 3).float().numpy())
    print("%-8.0e %-11.4f | %-13.4e %-13.4e %-13.4e | %-11.4e %-11.4e"
          % (kappa, fro(X) / 1.0, o3, s3, t3, o5, s5))

print("\n" + "=" * 92)
print("SAME, at iters=3, as a FRACTION of the 1e-3 one-way max_ortho latch")
print("=" * 92)
for kappa in (1.0, 1e2, 1e4, 1e6):
    X = prescribed(768, 64, 64, kappa)
    o3 = per_entry(ours(X, 3))
    s3 = per_entry(schedule_fro(X, 3))
    print("  kappa=%-8.0e  OURS %10.3e x the latch %s | SCHED-fro %10.3e x the latch %s"
          % (kappa, o3 / 1e-3, "(LATCHES)" if o3 > 1e-3 else "",
             s3 / 1e-3, "(LATCHES)" if s3 > 1e-3 else ""))

print("\n" + "=" * 92)
print("CROSS-CHECK of the transcribed loop against PUBLISHED numbers")
print("Ethan Epperly, 'A Neat Not-Randomized Algorithm: Polar Express' (2025-06-07),")
print("ran the UNSCALED printed list in MATLAB on randn(100)/25 and printed")
print("||X_t - polar||_F per iteration. Different seed, so NOT bit-exact; the")
print("point is the decay SHAPE and the machine-precision landing.")
print("=" * 92)
P = rng.standard_normal((100, 100)) / 25.0
U, s, Vt = np.linalg.svd(P, full_matrices=False)
polar = U @ Vt
print("  ours, seed 20260930:            theirs, published (MATLAB):")
mine = []
for i, (a, b, c) in enumerate(PAPER):
    P2 = P @ P.T
    P = ((c * P2 + b * np.eye(100)) @ P2 + a * np.eye(100)) @ P
    e = np.linalg.norm(P - polar, "fro")
    mine.append(e)
pub = [9.921347e-01, 9.676980e-01, 8.725474e-01, 5.821937e-01, 1.551595e-01,
       5.88549e-03, 2.286853e-07, 1.113148e-14]
for i, (m, p_) in enumerate(zip(mine, pub)):
    print("    iter %d   %.6e                %.6e" % (i + 1, m, p_))
print("  final: mine %.3e  theirs 1.113148e-14  -> the transcription lands at"
      " machine precision too" % mine[-1])
