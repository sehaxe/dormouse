#!/usr/bin/env python3
"""Static half of the dormouse-fused quality matrix: LOC, test LOC, markers, doc lies.

Test LOC is brace-matched, not "files that mention #[test]" (that over-counts
badly: a 100-line module with 3 one-line tests scores 97% test LOC).
Counted as test code:
  * every line of tests/*.rs (integration tests are pure test code)
  * every line inside a `#[cfg(test)] mod ... { }` block in src
  * every line inside a `#[test]`-attributed fn outside such a module
  * every line inside a `#[bench]`-less criterion `fn` under tests/
"""
import re
import subprocess
import sys
from pathlib import Path

WS = Path("/home/sehaxe/dormouse/vendor/dormouse-fused")
CONCURRENT = {"dormouse-kda", "dormouse-gdn2", "dormouse-engram", "dormouse-mor", "dormouse-spectral"}
MARKERS = re.compile(r"\bunimplemented!|\btodo!|\bFIXME\b|\bunreachable!|\bTODO\b")


def block_end(lines, start):
    """Index one past the closing brace of the block opened at/after `start`."""
    depth = 0
    seen = False
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if lines[i].count("{"):
            seen = True
        if seen and depth <= 0:
            return i + 1
    return len(lines)


def test_loc(paths):
    total = 0
    for p in paths:
        lines = p.read_text(errors="replace").splitlines()
        if p.parent.name == "tests" and p.suffix == ".rs":
            total += len(lines)
            continue
        if p.suffix != ".rs" or "tests" not in p.name:
            pass
        covered = [False] * len(lines)
        for i, ln in enumerate(lines):
            if re.search(r"#\[cfg\(test\)\]", ln):
                for j in range(i, block_end(lines, i)):
                    covered[j] = True
        for i, ln in enumerate(lines):
            if re.search(r"#\[(test|tokio::test)\]", ln) and not covered[i]:
                for j in range(i, block_end(lines, i)):
                    covered[j] = True
        total += sum(covered)
    return total


def main():
    members = subprocess.run(
        ["bash", "-c",
         "sed -n '/^members/,/^]/p' %s/Cargo.toml | grep -oE '\"[^\"]+\"' | tr -d '\"'" % WS],
        capture_output=True, text=True).stdout.split()
    print(f"{'crate':<20}{'loc':>7}{'testloc':>9}{'pct':>6}{'marks':>7}  flags")
    rows = []
    for m in members:
        d = WS / m
        if not d.is_dir():
            continue
        rs = [p for p in d.rglob("*.rs") if "target" not in p.parts]
        loc = sum(len(p.read_text(errors="replace").splitlines()) for p in rs)
        tl = test_loc(rs)
        marks = sum(len(MARKERS.findall(p.read_text(errors="replace"))) for p in rs)
        pct = (tl * 100 // loc) if loc else 0
        flags = []
        if m.split("/")[-1] in CONCURRENT:
            flags.append("concurrent")
        if any("examples" in p.parts for p in rs):
            flags.append("examples")
        if any("benches" in p.parts for p in rs):
            flags.append("benches")
        has_cuda = bool(re.search(r"^cuda\s*=", (d / "Cargo.toml").read_text(), re.M))
        # doc lie: fused/cubecl kernel talk in src but no cuda feature
        doclie = ""
        if not has_cuda:
            txt = "\n".join(p.read_text(errors="replace") for p in rs if p.suffix == ".rs")
            hits = len(re.findall(r"cubecl|cubin|__global__|blockIdx|Fused|fused kernel|CUDA kernel", txt))
            if hits:
                doclie = f"DOCLIE({hits})"
        if not has_cuda:
            flags.append("no-cuda")
        print(f"{m.split('/')[-1]:<20}{loc:>7}{tl:>9}{pct:>5}%{marks:>7}  "
              f"{doclie} {' '.join(flags)}")
        rows.append((m, loc, tl, marks, doclie))
    tot = sum(r[1] for r in rows)
    tt = sum(r[2] for r in rows)
    print(f"\nTOTAL loc={tot} testloc={tt} ratio={tt/max(tot,1)*100:.1f}% "
          f"code-only-ratio={tt/max(tot-tt,1)*100:.1f}%")


main()
