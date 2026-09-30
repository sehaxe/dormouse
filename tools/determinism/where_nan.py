"""The brief says the checkpoint FILE carries 0-4 NaN slots and ~475-501 slots
at |v|>=1e30. dmck says the MODEL section has none. So where are they?"""
import numpy as np, dmck
R = "/home/sehaxe/cache/determinism/runs"
for r in ("A", "B", "C", "D", "G"):
    s = dmck.read_sections(f"{R}/{r}/m.bin")
    print(f"\nrun {r}: step={s['step']} bytes={s['size']} "
          f"model={len(s['model'])} optim={len(s['optim'])} teacher={len(s['teacher'])}")
    for sec in ("model", "optim", "teacher"):
        a = np.frombuffer(s[sec], dtype="<f4")
        nan = int(np.isnan(a).sum()); big = int((np.isfinite(a) & (np.abs(a) >= 1e30)).sum())
        inf = int(np.isinf(a).sum())
        if nan or big or inf or sec == "model":
            print(f"   {sec:8s} slots={a.size:9d} NaN={nan:5d} Inf={inf:5d} "
                  f"|v|>=1e30={big:5d}  max|v|={np.abs(a[np.isfinite(a)]).max():.3e}")
