"""One table, one instrument, for any set of run labels.
Reports the two numbers that matter and refuses to be summarised by slot count:
  - direct (non-TSCT) slots differing : the SEED's job. 0 == seed works.
  - ||dW||_F / ||W||_F on the effective TSCT product : the MATERIAL error.
"""
import sys, itertools, numpy as np, dmck
R = "/home/sehaxe/cache/determinism/runs"

def load(r): return dmck.load(f"{R}/{r}/m.bin")
def ep(r):
    try: return int(open(f"/home/sehaxe/cache/determinism/{r}.epoch").read())
    except OSError: return 0

def W(t, base):
    return (t[f"{base}.Tsct.u"].astype(np.float64) * t[f"{base}.Tsct.s"].astype(np.float64)) \
           @ t[f"{base}.Tsct.v"].astype(np.float64).T
BASES = ["lm_head.inner", "loop_block.out_proj.inner",
         "loop_block.expert_ffns.0.down.inner", "loop_block.expert_ffns.2.gate_up.inner"]

def row(a, b, T):
    direct = dtot = 0; num = den = 0.0; mx = 0.0
    for k in T[a]:
        x = T[a][k].astype(np.float64).ravel(); y = T[b][k].astype(np.float64).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        d = x[m] - y[m]
        if "Tsct" not in k:
            direct += int((d != 0).sum())
        num += float((d*d).sum()); den += float((x[m]**2).sum()); mx = max(mx, float(np.abs(d).max()))
    rf = max(np.linalg.norm(W(T[a], bs) - W(T[b], bs)) / np.linalg.norm(W(T[a], bs))
             for bs in BASES)
    gap = ep(b) - ep(a)
    print(f"{a}/{b:6s} {gap:6d}s  direct={direct:9d}/{dtot or 7951694:<9d} "
          f"model relFro={(num/den)**0.5:.3e}  max|d|={mx:.3e}  dW/W={rf:.3e}")

runs = sys.argv[1:]
T = {r: load(r) for r in runs}
print(f"{'pair':10s} {'gap':>8s}  {'direct slots differing':>26s}  "
      f"{'model relFro':>13s}  {'max|d|':>10s}  {'worst dW/W':>10s}")
print("-"*96)
for a, b in itertools.combinations(runs, 2):
    row(a, b, T)
