#!/usr/bin/env python3
"""Summarize DM_LAUNCH_ATLAS=1 run logs into the launch-atlas TSV.

Reads the ATLAS / ATLAS-WARM lines one run produced (warm window =
`finish()`'s last third), prints per-run per-stage means and the
differential (arm-minus-base) table the atlas report quotes.

Usage: atlas_summarize.py LOG [LOG ...]   (base run first)
"""
import sys
from collections import defaultdict

US_PER_LAUNCH = 24.0


def parse(path):
    stages = defaultdict(lambda: [0, 0.0])  # stage -> [launch_sum, ms_sum]
    warm_from = None
    for line in open(path, errors="replace"):
        f = line.split("\t")
        if len(f) == 5 and f[0] == "ATLAS-WARM":
            stages[f[1]] = [float(f[2]), float(f[3])]
    return warm_from, dict(stages)


def main():
    logs = sys.argv[1:]
    runs = {}
    order = []
    for p in logs:
        name = p.rsplit("/", 1)[-1].replace("atlas_", "").replace(".log", "")
        warm, stages = parse(p)
        if not stages:
            print(f"{name}: no ATLAS-WARM rows", file=sys.stderr)
            continue
        runs[name] = stages
        order.append(name)
    base = runs[order[0]]
    base_total = sum(v[0] for v in base.values())

    print("# per-run warm stage means (launches/step, ms/step actual, ms@24us)")
    for name in order:
        tot = sum(v[0] for v in runs[name].values())
        print(f"# run={name} total={tot:.1f}")
        for st, (lps, mps) in sorted(runs[name].items(), key=lambda kv: -kv[1][0]):
            print(f"{name}\t{st}\t{lps:.1f}\t{mps:.1f}\t{lps * US_PER_LAUNCH / 1000:.1f}")

    print("\n# arm deltas vs base (launches/step by stage; negative = removed by the toggle)")
    for name in order[1:]:
        tot = sum(runs[name].get(s, [0, 0])[0] - base.get(s, [0, 0])[0] for s in set(base) | set(runs[name]))
        print(f"# arm={name} total_delta={tot:+.1f} ({tot / base_total * 100:+.1f}% of base)")
        for st in sorted(base, key=lambda s: -(base[s][0] - runs[name].get(s, [0, 0])[0])):
            d = base[st][0] - runs[name].get(st, [0, 0])[0]
            if abs(d) > 0.5:
                print(f"{name}\t{st}\t{-d:+.1f}\t{-d * US_PER_LAUNCH / 1000:+.1f}")


if __name__ == "__main__":
    main()
