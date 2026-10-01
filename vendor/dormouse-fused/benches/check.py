#!/usr/bin/env python3
"""Perf-regression check: compare a fresh bench JSON against committed baselines.

Usage: check.py <new.json> <baselines.json> <tolerance>
Exits non-zero when any fused kernel is slower than its baseline by more than
`tolerance` (fraction, e.g. 0.20 = 20%). Always exits 0 for NEW keys (a first
measurement is better than a failing PR) and prints the diff table.
"""
import json
import sys

new_path, base_path, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
new = json.load(open(new_path))
base = json.load(open(base_path))

worst = 0.0
fails = []
rows = []
for key in sorted(new):
    v, b = new[key], base.get(key)
    if b is None:
        rows.append(f"  {key:24s} {v:8.3f} ms   (new, no baseline)")
        continue
    ratio = v / b
    delta = ratio - 1.0
    # tiny kernels (<0.1 ms) are dominated by launch/sync jitter with a
    # run-to-run spread of roughly +-50%, so relax the tolerance for them:
    # a fixed 0.02 ms floor means a 0.02 ms kernel must double before failing
    eff_tol = max(tol, 0.02 / b)
    worst = max(worst, delta)
    flag = "REGRESSION" if delta > eff_tol else ("improved" if delta < -0.02 else "ok")
    rows.append(f"  {key:24s} {b:8.3f} -> {v:8.3f} ms  {delta:+.1%}  {flag}")
    if delta > eff_tol:
        fails.append((key, v, b))

print("dormouse-fused perf report (tolerance %+.0f%%):" % (tol * 100))
print("\n".join(rows))
if fails:
    print("\nFAIL: regressions detected:")
    for key, v, b in fails:
        print(f"  {key}: {v:.3f} ms vs baseline {b:.3f} ms")
    sys.exit(1)
print("\nPASS: no regressions beyond tolerance.")
