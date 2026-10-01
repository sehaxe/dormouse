#!/usr/bin/env bash
# The facade's feature matrix: the combinations a user actually picks, each one
# built, and the two common ones also tested. CPU only - the tests run on
# NdArray, so this needs no GPU and is safe to run next to a training run.
#
#   tools/test-feature-matrix.sh
#
# Two gates in one script, because they rot the same way:
#   1. `gen_facade.py --check` - the re-export list and the feature flags are
#      generated from the member manifests; a member added, renamed or given a
#      new feature without regenerating fails here. (It failed for months
#      silently: 13 crates declared `cuda`, the hand-written list named 12.)
#   2. every feature combination must BUILD. The feature set is cargo's, so
#      this has to be a sequence of cargo invocations, not a `#[test]`.
#      `--all-features` included: a feature nothing can select is a feature
#      that rots without a build ever exercising it.
# The test suite runs under the default and the full training combination; that
# is where `tests/facade.rs` resolves all 28 re-exports, and the two cfg'd
# backend re-exports (`burn_cuda`, `burn_autodiff`) need exactly those two.
set -uo pipefail
cd "$(dirname "$0")/.."

python3 tools/gen_facade.py --check || exit 1

COMBOS=(
  "--no-default-features"
  ""
  "--no-default-features --features std"
  "--no-default-features --features std,autodiff"
  "--no-default-features --features std,cuda"
  "--no-default-features --features std,cuda,autodiff"
  "--all-features"
)
# shellcheck disable=SC2124
TESTED=(
  ""
  "--no-default-features --features std,cuda,autodiff"
)

tested() {
  local combo=$1 t
  for t in "${TESTED[@]}"; do
    [[ $t == "$combo" ]] && return 0
  done
  return 1
}

fail=0
for combo in "${COMBOS[@]}"; do
  label="dormouse-fused ${combo:-<default>}"
  if tested "$combo"; then
    # shellcheck disable=SC2086
    cmd=(cargo test -p dormouse-fused --lib --test facade $combo)
  else
    # shellcheck disable=SC2086
    cmd=(cargo check -p dormouse-fused $combo)
  fi
  if out=$("${cmd[@]}" 2>&1); then
    echo "PASS  $label"
  else
    echo "FAIL  $label"
    echo "$out" | grep -E "^(error|warning: unused|test result)" | head -10
    fail=1
  fi
done
exit $fail
