#!/usr/bin/env python3
"""Enforce ADR-0020 / AGENTS.md 1.4: a fidelity claim must name its evidence.

The rule this mechanises, verbatim from docs/adr/0020-oracle-discipline.md:

    A crate may only use the phrase "bit-for-bit" in its README or doc comment if
    it names the external source its reference came from -- <org>/<repo>, <path>,
    and the commit or the fixture that carries it. Absent that, the strongest
    permitted phrasing is "matches our transcription of <arXiv> to <tolerance>".

ADR-0020 wrote that grep out and nobody built it. This is that grep, plus the
one thing a grep cannot do: it consults docs/ORACLE-TIERS.tsv, which records
what each file is ACTUALLY compared against. A file whose declared tier is not
(a) and which says "bit-for-bit" anyway is a false claim, and this exits 1.

Five rules, all mechanical, no judgement calls:

  R1 COVERAGE    every tests/ and examples/ file in the audited crates has a
                 registry row, AND every src//tools//README file that makes a
                 fidelity claim has one. A new test cannot join the tree without
                 declaring what it is compared against.
  R2 VOCABULARY  a tier-!=a file may not contain a positive "bit-for-bit" /
                 "bit-exact" claim. A disclaimer ("is NOT bit-for-bit") is
                 allowed; a claim is not.
  R3 PROVENANCE  tier (a) requires a github.com URL in the file; (b)/(c) require
                 an arXiv id or a named fixture+generator. Tier (d) is exempt:
                 "no external reference" is the DEFAULT for a self-consistency
                 check, so demanding the phrase from all 12 arm-vs-arm files
                 would be 12 edits carrying no information.
  R4 STALE       a registry row naming a file that no longer exists is a defect:
                 the audit table is a lie about the tree.
  R5 FILENAME    a FILENAME is the loudest claim in the repo -- it is what
                 `cargo test --test bit_exact` prints, and it is what gets
                 copied into commit messages. A file called bit_exact.rs /
                 bitforbit.rs at tier != (a) is a violation in its own right.

Exit 0 = clean, or waived-only. Exit 1 = a real violation. Rows with a non-empty
WAIVER are reported as DEBT and do not fail the gate; that is the only escape
hatch and it puts the debt in a diff, on purpose.

Usage:  tools/oracle_gate.py [--verbose] [--scope burn-kda,burn-gdn2]
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "docs", "ORACLE-TIERS.tsv")
CRATES = os.path.join(ROOT, "vendor", "burn-fused", "crates")
DEFAULT_SCOPE = ("burn-kda", "burn-gdn2")

# Requires a hyphen or a space, so a BARE `bitforbit` (the burn-kda example
# binary's name) is not itself read as a claim. A filename mention is not a
# claim either -- `see tests/bit_exact.rs` is a pointer, not an assertion.
CLAIM_RE = re.compile(r"bit[\s\-]for[\s\-]bit|bit[\s\-]exact", re.IGNORECASE)
FILENAME_CLAIM_RE = re.compile(r"^bit[\s\-_]?for[\s\-_]?bit|^bit[\s\-_]exact", re.I)

# A claim is negated if a negation token sits within this many characters before
# the match, on the same line. Deliberately tight: "worst deviation 0e0, asserted
# below, not printed" is a CLAIM with an unrelated "not" later in the sentence,
# and a loose window would wave it through.
NEG_WINDOW = 32
NEG_RE = re.compile(r"\b(?:not|no|never|neither|despite|without)\b", re.IGNORECASE)

GITHUB_RE = re.compile(r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
ARXIV_RE = re.compile(r"arxiv\.org/abs/\d{4}\.\d{4,5}|arxiv:\d{4}\.\d{4,5}")
FIXTURE_RE = re.compile(r"ref_data\.bin|gen_reference")


def die(msg):
    sys.stderr.write("oracle_gate: %s\n" % msg)
    raise SystemExit(2)


def read_registry(path):
    rows, header = [], None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if header is None:
                header = line.split("\t")
                continue
            cells = line.split("\t")
            cells += [""] * (len(header) - len(cells))
            rows.append(dict(zip(header, cells)))
    if header is None:
        die("registry %s is empty" % path)
    for need in ("file", "tier", "target", "waiver"):
        if need not in header:
            die("registry %s has no %r column" % (path, need))
    return rows


def positive_claims(text):
    """[(line_no, line)] for every NON-negated fidelity claim in `text`."""
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        for m in CLAIM_RE.finditer(line):
            if not NEG_RE.search(line[max(0, m.start() - NEG_WINDOW) : m.start()]):
                hits.append((n, line.strip()))
                break  # one report per line is enough
    return hits


def files_in_scope(scope):
    """Every auditable file, repo-relative, deterministic order."""
    out = []
    for crate in scope:
        base = os.path.join(CRATES, crate)
        for sub in ("src", "tests", "examples", "tools"):
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            for dirpath, _dirs, names in os.walk(d):
                for name in sorted(names):
                    if name.endswith((".rs", ".py")):
                        out.append(os.path.relpath(os.path.join(dirpath, name), ROOT))
        for extra in ("README.md",):
            p = os.path.join(base, extra)
            if os.path.exists(p):
                out.append(os.path.relpath(p, ROOT))
    return sorted(out)


def main():
    verbose = "--verbose" in sys.argv
    scope = DEFAULT_SCOPE
    for i, a in enumerate(sys.argv):
        if a == "--scope" and i + 1 < len(sys.argv):
            scope = tuple(x for x in sys.argv[i + 1].split(",") if x)

    rows = read_registry(REGISTRY)
    by_file = {r["file"]: r for r in rows}
    failures, debt, claims_by_file = [], [], {}

    # ---- R4 STALE -------------------------------------------------------
    for f in sorted(by_file):
        if not os.path.exists(os.path.join(ROOT, f)):
            failures.append(("R4 stale registry row", f, "the file is gone; the table lies"))

    # ---- pass 1: claims, over EVERY auditable file ----------------------
    auditable = files_in_scope(scope)
    for rel in auditable:
        with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as fh:
            claims_by_file[rel] = positive_claims(fh.read())

    # ---- R1 COVERAGE ----------------------------------------------------
    for rel in auditable:
        parts = rel.split("/")
        is_test = "tests" in parts or "examples" in parts
        if not (is_test or claims_by_file[rel]):
            continue
        if rel not in by_file:
            why = (
                "no ORACLE-TIERS.tsv row: a new test must declare what it is "
                "compared against"
                if is_test
                else "makes a fidelity claim but has no ORACLE-TIERS.tsv row"
            )
            failures.append(("R1 unregistered file", rel, why))

    # ---- R2 / R3 / R5 ---------------------------------------------------
    for rel in auditable:
        row = by_file.get(rel)
        if row is None:
            continue
        tier, claims = row["tier"].strip(), claims_by_file[rel]
        waived = bool(row["waiver"].strip())
        with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as fh:
            text = fh.read()

        if claims and tier not in ("a", "x") and not waived:
            for n, line in claims[:3]:
                failures.append(
                    (
                        "R2 fidelity claim in a tier-(%s) file" % tier,
                        "%s:%d" % (rel, n),
                        "expected value came from: %s. ADR-0020: only tier (a) may "
                        "say bit-for-bit. Rewrite it, or record a waiver." % row["target"],
                    )
                )
        elif claims and waived:
            debt.append((rel, claims, row["waiver"]))

        base = os.path.basename(rel)
        if FILENAME_CLAIM_RE.match(base) and tier not in ("a", "x") and not waived:
            failures.append(
                (
                    "R5 fidelity claim in the FILENAME",
                    rel,
                    "the name is what `cargo test --test %s` prints and what gets "
                    "quoted; tier (%s) cannot support it" % (base, tier),
                )
            )

        if tier == "a" and not GITHUB_RE.search(text) and not waived:
            failures.append(("R3 tier (a) with no github.com URL", rel, "AUTHORS needs org/repo + path"))
        if tier in ("b", "c") and not (ARXIV_RE.search(text) or FIXTURE_RE.search(text)) and not waived:
            failures.append(
                (
                    "R3 tier (%s) with no arXiv id and no named fixture" % tier,
                    rel,
                    "name the paper or the fixture+generator the reference came from",
                )
            )

    for f, claims, waiver in debt:
        sys.stderr.write("DEBT  %s: %d claim(s) waived -- %s\n" % (f, len(claims), waiver))
        if verbose:
            for ln, line in claims:
                sys.stderr.write("        %s:%d: %s\n" % (f, ln, line[:90]))
    for rule, where, why in failures:
        sys.stderr.write("FAIL  %-40s %s\n        %s\n" % (rule, where, why))

    sys.stderr.write(
        "\noracle_gate: %d registered, %d scanned, %d violation(s), %d waived\n"
        % (len(rows), len(claims_by_file), len(failures), len(debt))
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
