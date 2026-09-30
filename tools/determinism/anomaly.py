"""Which tensors move at step 20, and by how much more than everything else?

Parent AGENTS.md names a 20-step anomaly (aux.jepa_pred.norm.beta,
loop_block.iter_embed, 2-4 orders more than everything else, absent at
0/50/100) as unexplained. Rank the non-TSCT tensors by relative movement at
each step count so the anomaly is characterised rather than merely noted.
"""
import numpy as np, dmck
R = "/home/sehaxe/cache/determinism/runs"

def ranked(a, b, top=8):
    ta, tb = dmck.load(f"{R}/{a}/m.bin"), dmck.load(f"{R}/{b}/m.bin")
    rows = []
    for k in ta:
        if "Tsct" in k: continue
        x = ta[k].astype(np.float64).ravel(); y = tb[k].astype(np.float64).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        d = x[m] - y[m]
        nd = int((d != 0).sum())
        if nd == 0: continue
        rel = np.linalg.norm(d) / max(1e-30, np.linalg.norm(x[m]))
        rows.append((rel, nd, x.size, float(np.abs(d).max()), k))
    rows.sort(reverse=True)
    print(f"  {a}/{b}: {len(rows)} non-TSCT tensors moved at all")
    print(f"    {'tensor':40s} {'relFro':>10s} {'differing':>10s} {'of':>9s} {'max|d|':>10s}")
    for rel, nd, n, mx, k in rows[:top]:
        print(f"    {k:40s} {rel:10.2e} {nd:10d} {n:9d} {mx:10.2e}")
    return rows

for n, a, b in ((5,"s5a","s5b"), (20,"s20a","s20b"), (50,"s50a","s50b"), (100,"s100a","s100b")):
    print(f"=== steps {n} ===")
    try: ranked(a, b)
    except FileNotFoundError as e: print("  missing", e.filename)
