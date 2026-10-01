#!/usr/bin/env python3
"""Every `pub` item of ours with no caller outside its own crate. A ranked REPORT, never a deletion list.

WHY THIS EXISTS. The owner asked for a large cut. A cut needs evidence, and the
evidence for a dead public item is a repo-wide name search, not a reading. This
is that search, made one command so the answer can be re-run after any commit and
so a second reader does not have to trust the table in the report.

SCOPE. Two crate families, both ours:
  crates/*                        the 6 workspace members
  vendor/dormouse-fused/crates/*  the technology library (its own workspace)
`vendor/cubecl-fix` and `vendor/cubek-fix` are UPSTREAM FORKS and are out of
scope as DEFINITIONS; their files are still read as potential callers, because a
fork of ours that calls into us is a live caller. `vendor/dormouse-fused/bench`,
`/benches`, `/tools` and our `tools/` are callers too.

METHOD, one pass per file. `^pub ` at column 0 is an item (this is the top-level
API only: inherent and trait `impl` bodies are indented, and their methods are
counted through the TYPE or TRAIT name that carries them - see NOISE below). The
item's name is then looked for in every .rs file in the repo, split three ways
because the three answers are three different verdicts:
  other files, in CODE            a real caller
  the defining file, in CODE      used only inside its own file -> `pub` is
                                  over-visible; DEMOTE, do not delete
  any file, in a COMMENT          documented, not called -> dead-but-documented
The crate's own `tests/` are counted separately and never as callers: a test is
not a caller, it is the thing that would have caught a removal.

OUTPUT, TSV on stdout: crate, item, kind, file:line, calls, self, test, doc, lines.
Columns are stable so the report can be regenerated and diffed.

NOISE, named because each one is a way this report can be wrong:
  (a) a `pub use` re-export IS a use - the facade chain counts, and a crate that
      only re-exports is `library-api`, not dead;
  (b) a trait's methods are reached through the trait name, not their own, so a
      trait with 0 direct mentions of each method is normal;
  (c) an item under `#[cfg(feature = ...)]` is CONDITIONAL, and a 0 here says
      "no caller in the default feature set", not "dead". Flagged, not counted;
  (d) a name shared by several items in the repo (`new`, `forward`, `loss`) makes
      the count an UPPER bound shared by all of them - every such item is marked
      needs-eyes and none is called dead.

One more approximation, measured and safe: the self-test split slices the
defining file's code at the ORIGINAL line number of its first `#[cfg(test)]`,
but comments are already stripped from that slice, so the cut lands a few lines
late. The error counts a couple of test-region uses as live self-uses - it can
flip a DEAD item to `demote`, never the reverse, which is the direction a
deletion list must err.

Run: python3 tools/dead_pub_audit.py [--tsv out.tsv] [--top N]
Exit 0 always: this is a report, not a gate. A gate would have to be right about
`trait methods` and `#[cfg]`, and neither is decidable by regex.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Our code. Definitions come from here; vendor/cubecl-fix and vendor/cubek-fix
# are forks of upstream and define nothing of ours.
DEF_ROOTS = [ROOT / "crates", ROOT / "vendor/dormouse-fused/crates"]
# Every .rs file anywhere in the repo is a POTENTIAL CALLER, forks included.
CALLER_SKIP_DIRS = {".git", "target", "node_modules", ".bulba"}

ITEM = re.compile(r"^pub\s+(?:async\s+)?(?:unsafe\s+)?(fn|struct|enum|trait|const)\s+([A-Za-z_][A-Za-z0-9_]*)")
REEXPORT = re.compile(r"^pub\s+use\b")
# A name is mentioned, not part of a longer identifier.
IDENT_OK = re.compile(r"[A-Za-z0-9_]")


def rust_files():
    for p in sorted(ROOT.rglob("*.rs")):
        if any(part in CALLER_SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        yield p


def crate_of(path):
    """(crate_name, crate_root) for a definition file, or (None, None)."""
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if parts[:1] == ("crates",):
        return parts[1], path.parents[len(parts) - 1 - parts.index("src")] if "src" in parts else path.parent
    if parts[:2] == ("vendor", "dormouse-fused") and parts[2] == "crates":
        return parts[3], ROOT / "vendor/dormouse-fused/crates" / parts[3]
    return None, None


def item_extent(lines, start):
    """Lines the item occupies: brace-balanced from its `pub` line.

    A `pub` line with no `{` on it (`pub const X: usize = 4;`, a one-line
    `pub fn f() {}`) is exactly 1 line - counting to the next top-level `pub`
    instead would swallow whatever private item follows it. Braces inside
    strings and char literals are not parsed, so a `}` in a literal ends the
    count early; that UNDER-counts, which is the safe direction for a
    deletion estimate."""
    i, open_at = start, None
    while i < len(lines):
        if "{" in lines[i]:
            open_at = i
            break
        if ";" in lines[i] and "fn " not in lines[i]:
            break  # bodyless: `pub const X: usize = 4;`
        i += 1
    if open_at is None:
        return max(1, i - start + 1)
    depth = 0
    for j in range(open_at, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if depth <= 0:
            return max(1, j - start + 1)
    return max(1, len(lines) - start)


def test_mod_line(lines):
    """1-based line of the file's first `#[cfg(test)]`, or None. A self-use
    BELOW it is a test, not a live internal caller."""
    for i, ln in enumerate(lines):
        if ln.strip() == "#[cfg(test)]":
            return i + 1
    return None


def cfg_gated(lines, start, lookback=60):
    """True if a `#[cfg(...)]` covers this item: on the item itself (its own
    contiguous attribute block) or on an ancestor `mod` in the walk upward.

    Known-limitation: the walk stops at the first blank line, so a `cfg` on an
    outer `mod` separated from the item by a blank line reads as ungated. That
    errs toward reporting the item as LIVE, which is the safe direction for a
    deletion list."""
    j = start - 1
    while j >= 0 and start - j <= lookback:
        s = lines[j].strip()
        if s.startswith("#[cfg("):
            return True
        if s.startswith("mod ") or s.startswith("pub mod ") or s.startswith("pub(crate) mod "):
            return False  # reached the module declaration ungated
        if s == "" or s.startswith("//") or s.startswith("#!["):
            break
        j -= 1
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tsv", help="write the TSV here as well as to stdout")
    ap.add_argument("--top", type=int, default=0, help="print the N largest by lines first")
    args = ap.parse_args()

    # 1. definitions.
    items = []
    for base in DEF_ROOTS:
        for src in sorted(base.glob("*/src/**/*.rs")):
            crate, crate_root = crate_of(src)
            if crate is None:
                continue
            lines = src.read_text(errors="replace").splitlines()
            for i, ln in enumerate(lines):
                m = ITEM.match(ln)
                if not m:
                    continue
                kind, name = m.group(1), m.group(2)
                items.append({
                    "crate": crate, "kind": kind, "name": name,
                    "file": src.relative_to(ROOT), "line": i + 1,
                    "lines": item_extent(lines, i),
                    "cfg": cfg_gated(lines, i),
                    "testline": test_mod_line(lines),
                })
    if not items:
        print("no definitions found - wrong roots?", file=sys.stderr)
        return 0

    # 2. one pass per file: which names are mentioned where, in code vs in a
    #    comment. A comment-only mention is documentation of a name nothing
    #    calls, which is a different verdict from a real call.
    names = sorted({it["name"] for it in items})
    pat = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9_])")
    hits = {}  # file -> (code Counter, doc Counter, code-below-cfg-test Counter)
    for f in rust_files():
        text = f.read_text(errors="replace")
        code_lines, doc_lines = [], []
        for ln in text.splitlines():
            stripped = ln.lstrip()
            (doc_lines if (stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*")) else code_lines).append(ln)
        cc = Counter(m.group(1) for m in pat.finditer("\n".join(code_lines)))
        dc = Counter(m.group(1) for m in pat.finditer("\n".join(doc_lines)))
        rel = f.relative_to(ROOT)
        if cc or dc:
            hits[rel] = (cc, dc, code_lines)

    # 3. name multiplicity: a shared name can never be called dead.
    per_name = Counter(it["name"] for it in items)

    rows = []
    for it in items:
        calls = self_uses = self_tests = tests = doc = 0
        callers = []
        for rel, (cc, dc, code_lines) in hits.items():
            in_def = rel == it["file"]
            in_tests = str(rel).startswith(f"{crate_root_rel(it['crate'])}/tests/")
            if in_def:
                # -1: the definition line itself matched the name.
                n = max(0, cc[it["name"]] - 1)
                # a self-use below `#[cfg(test)]` is a test use, not a live one
                cut = it["testline"] or len(code_lines) + 1
                in_test_region = Counter(
                    m.group(1) for m in pat.finditer("\n".join(code_lines[cut - 1:]))
                )[it["name"]]
                self_uses += max(0, n - in_test_region)
                self_tests += in_test_region
            elif in_tests:
                tests += cc[it["name"]]
            else:
                if cc[it["name"]]:
                    callers.append(f"{rel}:{cc[it['name']]}")
                calls += cc[it["name"]]
            doc += dc[it["name"]]
        it.update(calls=calls, self=self_uses, selftest=self_tests, tests=tests,
                  doc=doc, callers=callers, shared=per_name[it["name"]])
        rows.append(it)

    rows.sort(key=lambda r: (-r["lines"], -r["calls"], r["crate"], r["name"]))

    def verdict(r):
        if r["shared"] > 1:
            return "needs-eyes"          # a shared name cannot be called dead
        if r["calls"] > 0:
            return "live"
        if r["self"] > 0 or r["selftest"] > 0 or r["tests"] > 0:
            return "demote"              # `pub` is over-visible; the body stays
        if r["cfg"]:
            return "cfg-only"            # no caller in the DEFAULT feature set
        return "DEAD" if r["doc"] == 0 else "dead-documented"

    hdr = ["crate", "item", "kind", "file:line", "calls", "self", "selftest", "test", "doc",
           "lines", "cfg", "shared", "verdict"]
    out = ["\t".join(hdr)]
    for r in rows:
        r["verdict"] = verdict(r)
        out.append("\t".join([
            r["crate"], r["name"], r["kind"], f"{r['file']}:{r['line']}",
            str(r["calls"]), str(r["self"]), str(r["selftest"]), str(r["tests"]), str(r["doc"]),
            str(r["lines"]), "yes" if r["cfg"] else "no", str(r["shared"]), r["verdict"],
        ]))
    text = "\n".join(out) + "\n"
    if args.tsv:
        Path(args.tsv).write_text(text)

    print(text, end="", file=sys.stderr if args.tsv else sys.stdout)
    def count(v):
        sel = [r for r in rows if r["verdict"] == v]
        return f"{len(sel):4d} items {sum(r['lines'] for r in sel):6d} lines"
    print("\n".join([
        "",
        f"items scanned        : {len(rows)}  ({sum(r['lines'] for r in rows)} lines under them)",
        f"  DEAD              : {count('DEAD')}",
        f"  dead-documented   : {count('dead-documented')}",
        f"  demote            : {count('demote')}   (a self-use or test reaches it;"
        " `pub` is over-visible, the body is not dead)",
        f"  cfg-only          : {count('cfg-only')}",
        f"  needs-eyes (dup)  : {count('needs-eyes')}",
        f"  live              : {count('live')}",
    ]), file=sys.stderr)
    return 0


_ROOT_CACHE = {}


def crate_root_rel(crate):
    if crate in _ROOT_CACHE:
        return _ROOT_CACHE[crate]
    for base in DEF_ROOTS:
        if (base / crate).is_dir():
            _ROOT_CACHE[crate] = (base / crate).relative_to(ROOT)
            return _ROOT_CACHE[crate]
    _ROOT_CACHE[crate] = crate
    return crate


if __name__ == "__main__":
    sys.exit(main())