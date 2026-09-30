"""Does the ~1e-6 init jitter AMPLIFY during training?

If it stays ~1e-6, same-seed cross-process runs are interchangeable for an A/B.
If it grows to ~1e-2 by 100 steps, every archived A/B carries that much
run-to-run noise and ADR-0002 needs the number quoted per step budget.
Uses the inherited harness's steps runs (each pair = same flags, same seed 7,
two processes).
"""
import numpy as np, dmck
R = "/home/sehaxe/cache/determinism/runs"

def stats(a, b):
    ta, tb = dmck.load(f"{R}/{a}/m.bin"), dmck.load(f"{R}/{b}/m.bin")
    nd = tot = direct = 0
    num = den = 0.0
    mx = 0.0
    for k in ta:
        x = ta[k].astype(np.float64).ravel(); y = tb[k].astype(np.float64).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        d = x[m] - y[m]
        nd += int((d != 0).sum()); tot += int(m.sum())
        num += float((d * d).sum()); den += float((x[m] ** 2).sum())
        mx = max(mx, float(np.abs(d).max()))
        if "Tsct" not in k:
            direct += int((d != 0).sum())
    return nd, tot, direct, (num / den) ** 0.5, mx

print(f"{'steps':>6s} {'pair':>10s} {'slot%':>8s} {'direct-differing':>18s} "
      f"{'relFro(model)':>14s} {'max|d|':>11s}")
print("-" * 74)
for n, (a, b) in ((0, ("S0a", "S0b")), (5, ("s5a", "s5b")), (20, ("s20a", "s20b")),
                  (50, ("s50a", "s50b")), (100, ("s100a", "s100b"))):
    try:
        nd, tot, direct, rf, mx = stats(a, b)
        print(f"{n:6d} {a + '/' + b:>10s} {100*nd/tot:7.3f}% {direct:18d} "
              f"{rf:14.3e} {mx:11.3e}")
    except FileNotFoundError as e:
        print(f"{n:6d} {a + '/' + b:>10s}  missing: {e.filename}")
