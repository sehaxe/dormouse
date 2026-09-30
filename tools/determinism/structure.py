"""Separate the two candidate mechanisms inside qr_householder:

  (a) the host-side `sign` branch (lib.rs:701-702) flipping because a 1-ULP
      reduction difference moved `first` across zero  -> a DISCRETE event,
      visible as whole negated columns: delta ~= -2*a, ratio a/b ~ -1.
  (b) plain reduction-order rounding amplified over the 64 sequential
      iterations -> SCATTER of a few ULP, sign-preserving, no structure.

If (a), a nonzero fraction of columns has large |delta| relative to |a| and
opposite sign. If (b), every column is a small random perturbation.
"""
import sys, numpy as np, dmck
R = "/home/sehaxe/cache/determinism/runs"

def cols(x, y, name):
    a = dmck.load(f"{R}/{x}/m.bin")[name].astype(np.float64)
    b = dmck.load(f"{R}/{y}/m.bin")[name].astype(np.float64)
    print(f"\n{name}  shape={a.shape}")
    print(f"  {'col':>4s} {'ndiff':>7s} {'max|d|':>11s} {'max|d|/|a|':>11s} "
          f"{'min a*b':>11s} {'cols fully negated':>19s}")
    neg = 0
    for j in range(a.shape[1]):
        u, v = a[:, j], b[:, j]
        d = np.abs(u - v)
        nd = int((d > 0).sum())
        rel = d.max() / max(1e-30, np.abs(u).max())
        dot = float((u * v).sum() / max(1e-30, np.linalg.norm(u) * np.linalg.norm(v)))
        if dot < -0.9:
            neg += 1
        if j < 8 or nd == a.shape[0]:
            print(f"  {j:4d} {nd:7d} {d.max():11.3e} {rel:11.3e} {dot:11.6f} "
                  f"{'YES' if dot < -0.9 else '':>19s}")
    print(f"  => columns with cosine < -0.9 (fully negated): {neg}/{a.shape[1]}")

for pair in (("A", "B"), ("D", "E")):
    print("=" * 78)
    print(f"{pair[0]} vs {pair[1]}")
    for nm in ("lm_head.inner.Tsct.v", "loop_block.out_proj.inner.Tsct.v"):
        cols(pair[0], pair[1], nm)
