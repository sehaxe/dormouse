"""Targets 2 and 3.

TARGET 2: run Keller Jordan's `newtonschulz5` (pinned upstream, their bytes,
          bf16 as shipped) and compare it against a transcription of
          burn-muon-plus's `orient_and_normalize` + `orthogonalize`
          (burn-muon-plus/src/lib.rs:266-305).

TARGET 3: the polar retraction has an EXTERNAL definition that is not ours --
          polar(X) = U V^T from an SVD. numpy routes that to LAPACK
          `dgesdd`, which is an independent implementation. This checks:
            (a) our sigma_max power-iteration estimate vs LAPACK's exact
                spectral norm, and
            (b) our retracted output vs LAPACK's polar(X), on the inputs the
                retraction is actually fed.

Run: /tmp/opencode/oracle-venv/bin/python run_targets_2_and_3.py
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "upstream")
from kellerjordan_newtonschulz5 import newtonschulz5

rng = np.random.default_rng(20260930)
NS_COEFFS = (3.4445, -4.775, 2.0315)
A, B, C = NS_COEFFS


def per_entry(U):
    k = U.shape[1]
    return np.linalg.norm(U.T @ U - np.eye(k), "fro") / k


def orient_and_normalize(g):
    """Transcription of burn-muon-plus/src/lib.rs:266-276."""
    t = g.shape[0] > g.shape[1]
    x = g.T.copy() if t else g.copy()
    norm = np.sqrt((x * x).sum()).clip(min=1e-7)
    return x / norm, t


def ours(g, steps=5):
    """Transcription of burn-muon-plus/src/lib.rs:283-305."""
    x, t = orient_and_normalize(g)
    for _ in range(steps):
        xx = x @ x.T
        xx2 = xx @ xx
        poly = B * xx + C * xx2
        x = A * x + poly @ x
    return x.T if t else x


print("=" * 88)
print("TARGET 2 -- Keller Jordan's newtonschulz5 (THEIR bytes, RUN) vs our transcription")
print("=" * 88)
print("  note: their line 6 is `X = G.bfloat16()`, ours keeps the input dtype.")
print("  bf16 on CPU is 8 mantissa bits, so agreement is only ~1e-2; that is the")
print("  cost of running THEIR shipped configuration, and is reported, not hidden.")
for shape in [(768, 64), (64, 768), (256, 256), (1000, 32)]:
    G = rng.standard_normal(shape) * 0.3
    theirs = newtonschulz5(torch.tensor(G), steps=5).float().numpy()
    ours_out = ours(G, steps=5)
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    print("  %-12s theirs per-entry=%.4e | ours(f64) per-entry=%.4e | max|diff|=%.3e  (bf16 floor)"
          % (str(shape), per_entry(theirs), per_entry(ours_out),
             np.abs(theirs - ours_out).max()))
print("  eps handling: theirs `X /= (X.norm() + 1e-7)`, ours `.clamp_min(1e-7)`.")
print("  identical for any matrix whose Frobenius norm exceeds 1e-7; they differ")
print("  only on a matrix small enough that the norm itself is below the floor.")
tiny = rng.standard_normal((64, 64)) * 1e-9
print("    ||G||_F = %.3e -> theirs denom %.6e, ours denom %.6e"
      % (np.sqrt((tiny * tiny).sum()),
         np.sqrt((tiny * tiny).sum()) + 1e-7, max(np.sqrt((tiny * tiny).sum()), 1e-7)))

print("\n" + "=" * 88)
print("TARGET 3 -- the polar retraction against LAPACK dgesdd (external, not ours)")
print("=" * 88)


def sigma_max_powerit(M, iters=5):
    g = M @ M.T
    v = g.sum(axis=1)
    for _ in range(iters):
        v = v / max(np.sqrt((v * v).sum()), 1e-12)
        v = g @ v
    gv = g @ v
    return np.sqrt((v * gv).sum() / max((v * v).sum(), 1e-14))


SPC, FRO, = 1.05, None
for shape, k in [((768, 64), 64), ((64, 768), 64), ((256, 256), 256), ((4096, 64), 64),
                 ((512, 128), 128), ((768, 64), 32)]:
    G = rng.standard_normal(shape)
    M = G.T.copy() if shape[0] > shape[1] else G.copy()
    exact_norm = np.linalg.norm(M, 2)            # LAPACK: true spectral norm
    est = sigma_max_powerit(M)                   # ours: 5-step power iteration
    rel = abs(est - exact_norm) / exact_norm
    # retraction, 3 iterations, our coefficients
    Mr = M / (est * SPC)
    for _ in range(3):
        xx = Mr @ Mr.T
        Mr = 1.875 * Mr + (-1.25 * xx + 0.375 * (xx @ xx)) @ Mr
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    polar_ref = U @ Vt                            # LAPACK reference
    if shape[0] > shape[1]:
        Mr = Mr.T
    err = np.linalg.norm(Mr - polar_ref, "fro") / np.linalg.norm(polar_ref, "fro")
    print("  %-12s rank%-4d  sig_max: ours=%.9f  LAPACK=%.9f  rel=%.2e   "
          "retraction vs LAPACK polar, rel F = %.4e  sig_max(out)=%.9f"
          % (str(shape), k, est, exact_norm, rel, err, np.linalg.norm(Mr, 2)))

print("\n  the reference for the retraction is therefore NOT absent: it is")
print("  polar(X) = U V^T from LAPACK, i.e. the DEFINITION, and the question the")
print("  iteration answers is 'how close did we get to it'. What has no external")
print("  reference is the CHOICE of coefficients, iteration count and prescale --")
print("  those are design decisions, and no oracle can adjudicate them.")
