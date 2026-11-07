#!/usr/bin/env bash
# Format gate: the whole tree must be ruff-format clean. While the legacy sweep
# has not landed yet, `--changed` checks only the files a branch touches, so new
# code cannot add to the debt and CI stays meaningful instead of decorative.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--changed" ]; then
  base=${2:-origin/main}
  # paths are repo-relative; ruff runs from port/ only for its config discovery,
  # so strip the prefix rather than cd and re-join it.
  files=$(git diff --name-only --diff-filter=ACMR "$base"...HEAD -- 'port/src/**/*.py' 'port/scripts/**/*.py' 'port/tests/**/*.py' 2>/dev/null | sed 's|^port/||' || true)
  if [ -z "$files" ]; then
    echo "format: no changed python files"
    exit 0
  fi
  echo "format: checking $(echo "$files" | wc -l) changed files"
  # shellcheck disable=SC2086
  ( cd port && .venv/bin/ruff format --check $files )
else
  ( cd port && .venv/bin/ruff format --check src scripts tests )
fi
