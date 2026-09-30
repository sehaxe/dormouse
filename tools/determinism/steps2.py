import numpy as np, dmck
R = "/home/sehaxe/cache/determinism/runs"
def pop(a, b):
    ta, tb = dmck.load(f"{R}/{a}/m.bin"), dmck.load(f"{R}/{b}/m.bin")
    out = {}
    for popname, keep in (("TSCT", lambda k: "Tsct" in k), ("direct", lambda k: "Tsct" not in k)):
        num = den = mx = 0.0; nd = tot = 0
        for k in ta:
            if not keep(k): continue
            x = ta[k].astype(np.float64).ravel(); y = tb[k].astype(np.float64).ravel()
            m = np.isfinite(x) & np.isfinite(y)
            d = x[m] - y[m]
            num += float((d*d).sum()); den += float((x[m]**2).sum())
            mx = max(mx, float(np.abs(d).max())); nd += int((d != 0).sum()); tot += int(m.sum())
        out[popname] = ((num/den)**0.5 if den else 0.0, mx, nd, tot)
    return out
print(f"{'steps':>5s} {'pair':>12s} | {'TSCT relFro':>12s} {'max|d|':>10s} {'slot%':>7s} | "
      f"{'direct relFro':>13s} {'max|d|':>10s} {'differing':>10s}")
print("-"*92)
for n, (a, b) in ((0,("S0a","S0b")),(5,("s5a","s5b")),(20,("s20a","s20b")),
                  (50,("s50a","s50b")),(100,("s100a","s100b"))):
    try:
        o = pop(a,b)
        t, d = o["TSCT"], o["direct"]
        print(f"{n:5d} {a+'/'+b:>12s} | {t[0]:12.3e} {t[1]:10.3e} {100*t[2]/t[3]:6.2f}% | "
              f"{d[0]:13.3e} {d[1]:10.3e} {d[2]:10d}")
    except FileNotFoundError as e:
        print(f"{n:5d} missing {e.filename}")
