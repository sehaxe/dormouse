"""Compare two checkpoints at three granularities:
   whole FILE  (what the brief's 0.167% / 12.50% actually used: 18 398 028 slots)
   MODEL record only (9 197 390 slots -- the parameters)
   per-tensor, so we learn WHICH parameters move.
Clean-slot instrument: finite and |v| < 1e30 only, per dmck.py.
"""
import sys, numpy as np, dmck

R = "/home/sehaxe/cache/determinism/runs"

def whole(path):
    b = open(path, "rb").read()
    return np.frombuffer(b, dtype="<f4")

def report(name, a, b):
    fin = np.isfinite(a) & np.isfinite(b) & (np.abs(a) < 1e30) & (np.abs(b) < 1e30)
    d = a[fin] != b[fin]
    nd, cnt = int(d.sum()), int(fin.sum())
    rf = float(np.linalg.norm(a[fin].astype(np.float64) - b[fin].astype(np.float64)) /
               max(1e-30, np.linalg.norm(a[fin].astype(np.float64))))
    print(f"{name:26s} {nd:10d}/{cnt:9d} = {100*nd/cnt:8.4f}%   relFro={rf:.3e}")
    return nd, cnt

for x, y in (("A","B"), ("A","C"), ("B","C"), ("A","D"), ("D","E"), ("E","F"), ("G","H")):
    print(f"=== {x} vs {y} " + "="*50)
    report("WHOLE FILE", whole(f"{R}/{x}/m.bin"), whole(f"{R}/{y}/m.bin"))
    rows, tot = dmck.compare(dmck.load(f"{R}/{x}/m.bin"), dmck.load(f"{R}/{y}/m.bin"))
    print(f"{'MODEL RECORD only':26s} {tot['diff']:10d}/{tot['clean']:9d} = "
          f"{100*tot['diff']/tot['clean']:8.4f}%   nan={tot['nan']} big={tot['big']}")
    movers = [(k, n, dd) for k, n, m, dd, ca, cb in rows if dd]
    if movers:
        print(f"   {len(movers)} tensors differ:")
        for k, n, dd in movers[:20]:
            print(f"     {k:40s} {dd:9d}/{n:9d} = {100*dd/n:7.3f}%")
    else:
        print("   NO tensor differs (bit-identical model section)")
