import numpy as np, dmck
R = "/home/sehaxe/cache/determinism/runs"
for r in ("A", "B"):
    rec = dmck.read_sections(f"{R}/{r}/m.bin")["model"]
    tl = dmck.tensors(rec)
    flat = np.frombuffer(rec, dtype="<f4")
    covered = np.zeros(flat.size, bool)
    for _, _, s, e in tl:
        covered[s//4:e//4] = True
    big = np.isfinite(flat) & (np.abs(flat) >= 1e30)
    print(f"run {r}: flat slots={flat.size}  named={covered.sum()}  "
          f"padding={int((~covered).sum())}")
    print(f"   |v|>=1e30 total={int(big.sum())}  inside a named tensor="
          f"{int((big & covered).sum())}  in PADDING={int((big & ~covered).sum())}")
    print(f"   gap index ranges holding them: "
          f"{sorted(set((np.nonzero(big)[0]//256).tolist()))[:12]} (bucket=256 slots)")
