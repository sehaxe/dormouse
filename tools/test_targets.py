#!/usr/bin/env python3
"""A feature-gated test file with no `required-features` runs ZERO tests and says `ok`.

Rule (burn-muon-plus, 243a003): a file-level `#![cfg(feature = "cuda")]` AND a
`[[test]] required-features = [...]` are BOTH needed. Either one alone leaves a
target that builds to an empty binary and prints `running 0 tests ... ok` - a
green line over a test that does not exist, indistinguishable from a pass.

This is the guard for that class across `vendor/burn-fused`: for every
`tests/*.rs` and `examples/*.rs` that compiles to zero tests on the default
feature set, the crate's Cargo.toml must declare the target and its
`required-features` must cover every non-default feature the file's own `cfg`
attributes name. Known-limitation: cfg evaluation understands
`all`/`any`/`not`/`feature = "x"`/`test`; anything else in a `cfg` is treated as
true (a file is only flagged when it is provably empty, so the check errs
toward silence, not toward a false alarm).

Run: python3 tools/test_targets.py     (exit 1 on findings)
"""
import re
import sys
from pathlib import Path

CRATES = Path(__file__).resolve().parents[1] / "vendor/burn-fused/crates"
FEATURE = re.compile(r'feature\s*=\s*"([^"]+)"')  # a cfg predicate
QUOTED = re.compile(r'"([^"]+)"')  # a bare name in a [...] list
DECL = re.compile(r"^([A-Za-z0-9_-]+)\s*=", re.M)  # a feature declaration
FEATURES_BLOCK = re.compile(r"^\[features\]$", re.M)


def cfg_args(text):
    """`cfg(<inner>)` -> `<inner>`, with paren-balanced so `all(a, b)` survives."""
    i = text.find("(")
    if i < 0:
        return text
    depth, j = 0, i
    while j < len(text):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
        j += 1
    return text[i + 1 :]


def holds(expr, defaults):
    """Evaluate a cfg predicate under `defaults`. Unknown syntax -> True."""
    expr = expr.strip()
    for op, ev in (("all", all), ("any", any)):
        if expr.startswith(op + "(") and expr.endswith(")"):
            return ev(holds(a, defaults) for a in split_args(cfg_args(expr)))
    if expr.startswith("not(") and expr.endswith(")"):
        return not holds(cfg_args(expr), defaults)
    for feat in FEATURE.findall(expr):
        if feat not in defaults:
            return False
    return True


def split_args(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [a for a in out if a.strip()]


def gated_features(lines):
    """Non-default features named by any `cfg` attribute in the file."""
    found = set()
    for m in re.finditer(r"^#!?\[cfg\((.*)\)\]\s*$", "\n".join(lines), re.M):
        found |= set(FEATURE.findall(cfg_args(m.group(1))))
    return found


def reachable(lines, defaults, marker):
    """How many `marker` items survive the default feature set (0 = a green no-op)."""
    live = 0
    for i, line in enumerate(lines):
        if not re.match(marker, line.strip()):
            continue
        j = i
        while j > 0 and lines[j - 1].lstrip().startswith("#["):
            j -= 1
        k = i
        while k + 1 < len(lines) and lines[k + 1].lstrip().startswith("#["):
            k += 1
        cfgs = [a[a.index("cfg(") :] for a in lines[j : k + 1] if "#[cfg(" in a]
        if all(holds(cfg_args(c), defaults) for c in cfgs):
            live += 1
    return live


def targets(toml, kind):
    """name -> set(required-features) for every [[kind]] block."""
    out = {}
    for m in re.finditer(rf"^\[\[{kind}\]\]$(.*?)(?=^\[|\Z)", toml, re.M | re.S):
        body = m.group(1)
        name = re.search(r'^name\s*=\s*"([^"]+)"', body, re.M)
        if name:
            req = re.search(r"^required-features\s*=\s*\[(.*?)\]", body, re.M | re.S)
            out[name.group(1)] = set(QUOTED.findall(req.group(1))) if req else set()
    return out


def main():
    findings = []
    for crate in sorted(p for p in CRATES.iterdir() if p.is_dir()):
        manifest = (crate / "Cargo.toml").read_text()
        block = manifest[FEATURES_BLOCK.search(manifest).end() :].lstrip("\n").split("\n[", 1)[0]
        defaults = set(QUOTED.findall(re.search(r"^default\s*=.*$", block, re.M).group(0)))
        declared = set(DECL.findall(re.sub(r"^default\s*=.*$", "", block, flags=re.M)))
        for kind, folder, marker in (
            ("test", "tests", r"^#\[test\]$"),
            ("example", "examples", r"^fn main\b"),
        ):
            if not (crate / folder).is_dir():
                continue
            for f in sorted((crate / folder).glob("*.rs")):
                lines = f.read_text().splitlines()
                file_cfg = next((l for l in lines if l.startswith("#![cfg(")), None)
                live = 0 if file_cfg and not holds(cfg_args(file_cfg[6:]), defaults) else reachable(lines, defaults, marker)
                if live:
                    continue
                need = gated_features(lines) - defaults
                if need - declared:
                    findings.append(
                        f"{crate.name}: {kind} {f.name} cfg names undeclared "
                        f"feature(s) {sorted(need - declared)}"
                    )
                    continue
                got = targets(manifest, kind).get(f.stem)
                if got is None:
                    findings.append(
                        f"{crate.name}/{folder}/{f.name}: nothing runs on the "
                        f"default cell, no [[{kind}]] target"
                    )
                elif not need <= got:
                    findings.append(
                        f"{crate.name}/{folder}/{f.name}: required-features "
                        f"{sorted(got)} misses {sorted(need - got)} the file's cfg needs"
                    )
                elif got & defaults:
                    findings.append(
                        f"{crate.name}/{folder}/{f.name}: required-features "
                        f"{sorted(got & defaults)} name a DEFAULT feature - the "
                        f"target could never run"
                    )
    for line in findings:
        print(f"FAIL {line}")
    print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
