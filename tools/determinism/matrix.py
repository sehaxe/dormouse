#!/usr/bin/env python3
"""Full pairwise tensor-level matrix for the zero-step runs.

Protocol is in research/2026-09-30-cross-process-repro.md. Instrument: the
checkpoint's CBOR tensor map, so every comparison is over NAMED PARAMETER
TENSORS, and NaN / |v|>=1e30 slots are excluded from BOTH sides and reported
separately instead of being counted as differences.
"""
import sys, itertools, time
sys.path.insert(0, "/home/sehaxe/cache/determinism")
import dmck

D = "/home/sehaxe/cache/determinism"
RUNS = sys.argv[1:] or ["A", "B", "C", "D", "E", "F"]
T = {r: dmck.load(f"{D}/runs/{r}/m.bin") for r in RUNS}
EP = {r: int(open(f"{D}/{r}.epoch").read()) for r in RUNS}
order = sorted(RUNS, key=lambda r: EP[r])
print("run order by epoch:", " ".join(f"{r}@{EP[r]}" for r in order))
print()

# per-tensor first, then the aggregate, so a single noisy tensor is visible
print(f"{'pair':7s} {'gap_s':>6s} {'slots':>9s} {'clean':>9s} {'diff':>8s} {'pct':>9s} "
      f"{'nanA':>5s} {'nanB':>5s} {'bigA':>5s} {'bigB':>5s}  top tensor")
print("-" * 118)
res = []
for a, b in itertools.combinations(order, 2):
    rows, tot = dmck.compare(T[a], T[b])
    pct = 100.0 * tot["diff"] / tot["clean"]
    res.append((a, b, EP[b] - EP[a], tot["diff"], pct, tot))
    top = max(rows, key=lambda r: r[3] / max(r[1], 1))
    print(f"{a} vs {b:3s} {EP[b]-EP[a]:6d} {tot['n']:9d} {tot['clean']:9d} "
          f"{tot['diff']:8d} {pct:8.4f}% {'':5s} {'':5s} {tot['big']:5d} {'':5s}  "
          f"{top[0]} {100*top[3]/max(top[1],1):.1f}%")
print()
d = [r[3] for r in res]
p = [r[4] for r in res]
print(f"pairs={len(res)}  diff min={min(d)} max={max(d)}  pct min={min(p):.4f}% max={max(p):.4f}%")
print(f"median pct = {sorted(p)[len(p)//2]:.4f}%")
gaps = sorted((r[2], r[4]) for r in res)
print("\ngap vs pct (is 'minutes later' worse than 'seconds apart'?)")
for g, pp in gaps:
    print(f"  {g:6d}s  {pp:8.4f}%")
