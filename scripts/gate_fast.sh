#!/usr/bin/env bash
# The fast gate. Everything here runs in under a minute on a laptop and in CI.
# A gate that cannot fail is decoration — every step below exits non-zero on a
# real defect, and each one has caught a real one in this repo's history.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
step() { printf '\n=== %s\n' "$1"; }

step "ruff check (lint)"
( cd port && .venv/bin/ruff check src scripts tests ) || fail=1

step "ruff format (style; FORMAT_SCOPE=--changed checks only this branch's files)"
bash scripts/gate_format.sh "${FORMAT_SCOPE:-}" || fail=1

step "doc references (a path in a comment must exist)"
python3 tools/check_doc_refs.py || fail=1

step "dead knobs (a preset must not set a field nothing reads)"
if [ -f port/scripts/dead_knob_audit.py ]; then
  # rc: 0 = nothing to report, 1 = DEAD/SHADOWED fields exist (documented, not a
  # gate), 2 = a preset SETS a field nothing reads (the muon_lr_mult class) — that
  # is the only failure this instrument should ever block on.
  ( cd port && PYTHONPATH=src .venv/bin/python scripts/dead_knob_audit.py --preset-glob 'configs/**/*.toml' >/dev/null 2>&1 ) || rc=$?
  rc=${rc:-0}
  if [ "$rc" -eq 2 ]; then echo "PRESET-USES-DEAD"; fail=1; else echo "dead knobs: rc=$rc (no preset sets a dead field)"; fi
  unset rc
else
  echo "SKIP: dead_knob_audit.py not present"
fi

step "claims ledger (a number must be re-derivable or marked unverifiable)"
if [ -f port/scripts/claims_ledger.py ]; then
  ( cd port && PYTHONPATH=src .venv/bin/python scripts/claims_ledger.py --check ) || fail=1
else
  echo "SKIP: claims_ledger.py not present"
fi

step "rust: cargo check"
if command -v cargo >/dev/null; then
  cargo check --workspace --quiet || fail=1
else
  echo "SKIP: no cargo"
fi

if [ "$fail" -ne 0 ]; then
  printf '\nGATE FAILED\n'
  exit 1
fi
printf '\nGATE PASSED\n'
