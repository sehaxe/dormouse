#!/usr/bin/env bash
# The reproducible engine benchmark. Docs: docs/guides/benchmarking.md.
#
#   bash scripts/bench_engine.sh --quick          # 3 shapes x 50 steps
#   bash scripts/bench_engine.sh --full           # 7 shapes x 200 steps
#   bash scripts/bench_engine.sh --quick --dry-run  # commands + bills, CPU only
#
# It launches the trainer exactly as scripts/recipe_ladder.sh does; the whole
# implementation (flags, parsing, FLOP ledger, JSONL + markdown) is
# scripts/bench_engine.py. PY=<interpreter> overrides the venv.
set -euo pipefail

HERE=${BASH_SOURCE[0]%/*}
if [ "$HERE" = "${BASH_SOURCE[0]}" ]; then HERE=.; fi
ROOT=$(cd "$HERE/.." && pwd)
cd "$ROOT"

PY=${PY:-$ROOT/.venv/bin/python}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$ROOT/scripts/bench_engine.py" "$@"
