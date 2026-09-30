"""The number that decides ADR-0002: does a ~5-ULP perturbation of the TSCT
factors move the EFFECTIVE weight matrix enough to matter?

W = U diag(s) V^T  (the product the layer actually computes). Report
||W_A - W_B||_F / ||W_A||_F -- the relative Frobenius error of the thing the
network uses, not of a factorisation of it. Compare against the seed-8 control,
which is the scale of a genuinely different initialisation.
"""
import numpy as np, dmck
R = "/home/sehaxe/cache/determinism/runs"

def W(run, base):
    t = dmck.load(f"{R}/{run}/m.bin")
    u, v, s = t[f"{base}.Tsct.u"].astype(np.float64), \
              t[f"{base}.Tsct.v"].astype(np.float64), \
              t[f"{base}.Tsct.s"].astype(np.float64)
    return (u * s) @ v.T

for base in ("lm_head.inner", "loop_block.out_proj.inner",
             "loop_block.expert_ffns.0.down.inner"):
    print(f"\n{base}")
    ref = W("A", base); nr = np.linalg.norm(ref)
    for x, y, lab in (("A","B","seed 7 vs 7, 68 s"), ("A","C","seed 7 vs 7, 893 s"),
                      ("D","E","seed 7 vs 7, 69 s"), ("A","G","seed 7 vs 8 (CONTROL)")):
        try:
            d = np.linalg.norm(W(x, base) - W(y, base))
            print(f"  {x}/{y}  {lab:24s} ||dW||_F/||W||_F = {d/nr:.3e}"
                  f"   elementwise max|dW| = {np.abs(W(x,base)-W(y,base)).max():.3e}")
        except FileNotFoundError:
            pass
