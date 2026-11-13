#!/usr/bin/env bash
# One-shot lane split: port/ -> python/, and the Rust lane stops being the
# default thing CI builds first.
#
# Why a script: 2 145 path references across 727 files. A hand edit misses some,
# and a missed path is a broken doc reference (the repo gates on that) or a
# script that cds nowhere. This runs once, on main, when no agent is mid-flight.
#
# It is idempotent: if python/ exists and port/ does not, it exits 0.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d python ] && [ ! -d port ]; then
  echo "already split"
  exit 0
fi
[ -d port ] || { echo "no port/ — nothing to do"; exit 0; }

echo "== git mv port python"
git mv port python

echo "== rewrite live references (code, CI, guides, configs, scripts)"
# Historical records under docs/records/reviews and docs/records/research describe the tree as it
# was; rewriting them would falsify the record. They keep `port/` and
# tools/check_doc_refs.py resolves the alias.
files=$(git ls-files | grep -vE '^(docs/records/reviews|docs/records/research|docs/records/archive)/' | grep -E '\.(py|md|yml|yaml|sh|toml|rs|json)$' || true)
for f in $files; do
  if grep -q 'port/' "$f" 2>/dev/null; then
    perl -0pi -e 's{(?<![\w./-])port/}{python/}g' "$f"
  fi
done

echo "== rewrite the local (ignored) helper scripts too"
for f in scripts/*.sh; do
  [ -f "$f" ] && grep -q 'port/' "$f" && perl -0pi -e 's{(?<![\w./-])port/}{python/}g' "$f" || true
done

echo "== done. verify:"
echo "   bash scripts/gate_fast.sh"
echo "   python3 tools/check_doc_refs.py"
