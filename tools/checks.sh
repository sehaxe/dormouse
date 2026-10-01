#!/usr/bin/env bash
# THE CHECKS AGGREGATOR. One command that runs every gate this repo already
# has, so "green" stops meaning "whichever gate I happened to remember".
#
#     tools/checks.sh [--fast] [--only NAME]
#
# WHY THIS EXISTS (2026-10-01). The repo-excellence review named the honest
# gap: the gates live behind three doors (cargo, npm, python) and no single
# command runs everything, so a commit could pass `cargo test` and still ship
# a dead doc reference or a false fidelity claim. Burn's practice is one door
# (make/xtask); this file is that door. Every check below is an EXISTING gate,
# invoked verbatim - nothing new is enforced here, the new thing is only that
# the gates cannot be individually forgotten. A gate that crashes is still
# reported, as FAIL with the crash tail: an aggregator that hides a broken
# gate behind a polite SKIP would be ADR-0019's silent fallback in a new coat.
#
# WHAT IT RUNS (name -> the gate, and the one-line why):
#   lib-tests     cargo test --lib over the three workspace crates, through
#                 tools/build_lock.sh (`run checks --`): the lock is acquired
#                 HERE, per cargo command, per build_lock's own usage - this
#                 script does not wrap itself in the lock.
#   doc-warnings  RUSTFLAGS="-D warnings" cargo doc --no-deps over the three
#                 doc-carrying crates. CI's rustdoc gate is currently weaker
#                 than this local one (a separate lane fixes CI); this is the
#                 local truth. Also under build_lock: it compiles, and the
#                 lock's whole reason to exist is unsynchronised cargo work.
#   doc-refs      python3 tools/check_doc_refs.py - every repo path a document
#                 names must exist (the 2026-10-01 docs consolidation gate).
#   dead-pub      python3 tools/dead_pub_audit.py - the dead-`pub` audit. No
#                 --summary flag exists (checked 2026-10-01; the flags are
#                 --tsv/--top), so this is the plain run: the full TSV goes to
#                 the per-check log and its counts block is printed here. It
#                 always exits 0 BY DESIGN ("a ranked REPORT, never a deletion
#                 list"), so PASS means "ran", not "no dead pub".
#   oracle        python3 tools/oracle_gate.py - ADR-0020: a fidelity claim
#                 must name its evidence, against docs/protocols/ORACLE-TIERS.tsv.
#   docs-site     (cd docs-site && npm run check) - the docs-site manifest +
#                 mermaid gates. SKIP, not FAIL, when node_modules is absent:
#                 "the gate cannot run here" is a different fact from "the
#                 gate is red", and the summary must say which one happened.
#
# NOT AGGREGATED, ON PURPOSE:
#   tools/lib_gate.sh - its own header: 1401 s cold, a second cargo workspace
#   (686 packages). Not a fast gate under any reading; run it when you touch
#   vendor/dormouse-fused. tools/test_targets.py is likewise vendor-scoped.
#   Anything `--features cuda` - GPU runs are a human decision (AGENTS.md 2.4),
#   not a default gate.
#
# FLAGS. --fast: only the checks that carry no heavy cargo build beyond the
# lib test itself (lib-tests, doc-refs, dead-pub, oracle); the rest are listed
# as SKIP (--fast), because an honest summary shows what did NOT run, not a
# shorter list. --only NAME: exactly one check, by the names above (unknown
# name exits 2 with the list); --only wins over --fast. No `set -e`: one red
# gate must not hide the state of the others - the point of the aggregator is
# the whole picture, and the exit code carries the verdict.
#
# OUTPUT. One line per check: `[checks] <name> PASS|FAIL|SKIP <time> <cmd>`,
# the last 30 log lines under any FAIL, then a count line. Exit 1 on any
# FAIL, 0 otherwise. bash + the gates' own interpreters; installs nothing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FAST=0 ONLY=
while [ $# -gt 0 ]; do
    case $1 in
        --fast) FAST=1 ;;
        --only) [ $# -ge 2 ] || { echo "checks: --only needs a name" >&2; exit 2; }
                ONLY=$2; shift ;;
        *) echo "checks: unknown flag: $1 (flags: --fast, --only NAME)" >&2; exit 2 ;;
    esac
    shift
done

ALL="lib-tests doc-warnings doc-refs dead-pub oracle docs-site"
if [ -n "$ONLY" ]; then
    case " $ALL " in
        *" $ONLY "*) SELECTED=$ONLY ;;
        *) echo "checks: no such check '$ONLY'. checks are: $ALL" >&2; exit 2 ;;
    esac
elif [ "$FAST" = 1 ]; then
    SELECTED="lib-tests doc-refs dead-pub oracle"
else
    SELECTED=$ALL
fi

LOG="$(mktemp -d "${TMPDIR:-/tmp}/dormouse-checks.XXXXXX")"
trap 'rm -rf "$LOG"' EXIT

N_PASS=0 N_FAIL=0 N_SKIP=0 FAILED=""
T0=$(date +%s.%N)

run() { # run NAME CMD... : time it, log it, one verdict line; FAIL prints the tail
    local name=$1; shift
    local rc dt
    printf '[checks] %-13s running  %s\n' "$name" "$*"
    "$@" >"$LOG/$name.log" 2>&1
    rc=$?
    dt=$(awk -v a="$CUR_T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1fs", b-a}')
    if [ "$rc" -eq 0 ]; then
        N_PASS=$((N_PASS+1))
        printf '[checks] %-13s PASS   %8s  %s\n' "$name" "$dt" "$*"
        return 0
    fi
    N_FAIL=$((N_FAIL+1)); FAILED="$FAILED $name"
    printf '[checks] %-13s FAIL   %8s  %s  (exit %d), tail:\n' "$name" "$dt" "$*" "$rc"
    tail -n 30 "$LOG/$name.log" | sed 's/^/    /'
}

skip() { # skip NAME REASON
    local dt
    dt=$(awk -v a="$CUR_T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1fs", b-a}')
    N_SKIP=$((N_SKIP+1))
    printf '[checks] %-13s SKIP   %8s  %s\n' "$1" "$dt" "$2"
}

for name in $SELECTED; do
    CUR_T0=$(date +%s.%N)
    case $name in
        lib-tests)
            run "$name" tools/build_lock.sh run checks -- \
                cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib
            ;;
        doc-warnings)
            run "$name" tools/build_lock.sh run doc-warnings -- \
                env RUSTFLAGS="-D warnings" \
                cargo doc --no-deps -p dormouse-core -p dormouse-data -p dormouse-cli
            ;;
        doc-refs)
            run "$name" python3 tools/check_doc_refs.py
            ;;
        dead-pub)
            run "$name" python3 tools/dead_pub_audit.py
            # the counts block, not the TSV: the TSV is in the log, the verdict-sized
            # summary is what a green run should still show
            grep -h -A6 '^items scanned' "$LOG/$name.log" 2>/dev/null | sed 's/^/    /'
            ;;
        oracle)
            run "$name" python3 tools/oracle_gate.py
            ;;
        docs-site)
            if [ -d docs-site/node_modules ]; then
                run "$name" bash -c 'cd docs-site && npm run check'
            else
                skip "$name" "no docs-site/node_modules - npm install there first"
            fi
            ;;
    esac
done

printf '[checks] ------------------------------------------------------\n'
printf '[checks] %d PASS, %d FAIL, %d SKIP in %s\n' \
    "$N_PASS" "$N_FAIL" "$N_SKIP" \
    "$(awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1fs", b-a}')"
if [ "$N_FAIL" -gt 0 ]; then
    printf '[checks] FAILED:%s\n' "$FAILED"
    exit 1
fi
exit 0
