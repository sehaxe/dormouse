"""Where does divergence ENTER the 64-iteration Householder loop?

qr_householder runs j = 0..63, and column j of Q is finished by iteration j.
If a reduction's summation order changes at iteration j*, every column < j* is
bit-identical across processes and every column >= j* carries the amplified
perturbation. So the FIRST differing column is a direct read-out of which
iteration first produced a different result -- and if it moves around between
process pairs, the cause is a per-run reduction choice, not a fixed seed bug.
"""
import itertools, sys, numpy as np, dmck
R = "/home/sehaxe/cache/determinism/runs"
RUNS = sys.argv[1:] or ["A","B","C","D","E","F"]
T = {r: dmck.load(f"{R}/{r}/m.bin") for r in RUNS}
names = [k for k in sorted(T[RUNS[0]]) if "Tsct" in k]
print(f"{'tensor':42s} " + " ".join(f"{a}/{b}".rjust(9) for a,b in itertools.combinations(RUNS,2)))
print("-"*(42+10*len(list(itertools.combinations(RUNS,2)))))
firsts = []
for nm in names:
    cells = []
    for a, b in itertools.combinations(RUNS, 2):
        x = T[a][nm].astype(np.float64); y = T[b][nm].astype(np.float64)
        d = (np.abs(x - y).max(axis=0) > 0)          # per-column "any slot differs"
        j = int(np.argmax(d)) if d.any() else -1
        cells.append(f"{j if j>=0 else 'IDENT':>9}")
        firsts.append(j)
    print(f"{nm:42s} " + " ".join(cells))
f = [j for j in firsts if j >= 0]
print(f"\nfirst diverging column: n={len(f)} min={min(f)} max={max(f)} "
      f"median={int(np.median(f))}  identical-tensors={len(firsts)-len(f)}")
