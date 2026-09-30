#!/usr/bin/env python3
"""The metric that actually answers the question.

Counting differing SLOTS is the wrong statistic: an f32 tensor whose last bit
moved counts exactly as much as one that moved by 1.0. This reports, for every
pair, both the slot count and the MAGNITUDE distribution, and splits the model
into the two populations the measurement shows it actually has:

  TSCT factors  - initialised by a 64-iteration Householder QR (burn-spectral
                  lib.rs:456-469, 688-715), 4 reductions + 3 host syncs per
                  iteration, so the summation order is kernel-choice dependent
  direct draws  - initialised straight from the seeded RNG

If the seeded RNG is doing its job, the second population is BIT-EXACT across
processes and the first is not.
"""
import sys, itertools
sys.path.insert(0, "/home/sehaxe/cache/determinism")
import dmck, numpy as np

D = "/home/sehaxe/cache/determinism"
RUNS = sys.argv[1:] or ["A", "B", "C", "D", "E", "F"]
T = {r: dmck.load(f"{D}/runs/{r}/m.bin") for r in RUNS}
EP = {r: int(open(f"{D}/{r}.epoch").read()) for r in RUNS}
order = sorted(RUNS, key=lambda r: EP[r])
K = sorted(T[order[0]])


def m_of(a):
    a = a.astype(np.float32)
    return np.isfinite(a) & (np.abs(a) < 1e30)


print("=== NaN / huge-slot census per run (model section) ===")
for r in order:
    nan = big = tot = 0
    for k in K:
        c = dmck.classify(T[r][k]); nan += c["nan"]; big += c["big"]; tot += c["n"]
    print(f"  {r}: slots={tot}  NaN={nan}  |v|>=1e30={big}  ({100*big/tot:.4f}%)")

print("\n=== per-pair: slot count vs MAGNITUDE, split by population ===")
print(f"{'pair':8s} {'gap':>5s} | {'TSCT diff%':>10s} {'TSCT max|d|':>11s} | "
      f"{'direct diff slots':>17s} {'direct max|d|':>13s} | {'whole-model max|d|':>17s}")
print("-" * 104)
agg = []
for a, b in itertools.combinations(order, 2):
    tmax = dmax = wmax = 0.0
    tdiff = tslots = ddiff = 0
    for k in K:
        x = T[a][k].astype(np.float32).ravel(); y = T[b][k].astype(np.float32).ravel()
        m = m_of(x) & m_of(y)
        if not m.any():
            continue
        d = np.abs(x[m].astype(np.float64) - y[m].astype(np.float64))
        nd = int((d > 0).sum())
        wmax = max(wmax, float(d.max()))
        if "Tsct" in k:
            tmax = max(tmax, float(d.max())); tdiff += nd; tslots += m.sum()
        else:
            dmax = max(dmax, float(d.max())); ddiff += nd
    agg.append((tdiff / tslots, tmax, ddiff, dmax, wmax))
    print(f"{a} vs {b:2s} {EP[b]-EP[a]:5d} | {100*tdiff/tslots:9.3f}% {tmax:11.3e} | "
          f"{ddiff:17d} {dmax:13.3e} | {wmax:17.3e}")

print("\n=== VERDICT NUMBERS ===")
td = [x[0] for x in agg]
dd = [x[2] for x in agg]
w = [x[4] for x in agg]
print(f"TSCT differing-slot fraction : min {100*min(td):.2f}%  max {100*max(td):.2f}%  "
      f"median {100*sorted(td)[len(td)//2]:.2f}%")
print(f"non-TSCT differing slots     : min {min(dd)}  max {max(dd)}  "
      f"(out of {sum(T[order[0]][k].size for k in K if 'Tsct' not in k)} slots)")
print(f"whole-model max |delta|      : min {min(w):.3e}  max {max(w):.3e}")
print(f"f32 eps at 1.0 = {np.finfo(np.float32).eps:.3e}")
