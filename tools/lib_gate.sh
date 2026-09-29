#!/usr/bin/env bash
# THE CPU gate for the fused library (vendor/burn-fused). One command:
#
#     tools/lib_gate.sh [worktree-root]     # default: the repo this script is in
#
# WHY IT IS A SCRIPT AND NOT A SECOND `-p` ON tools/wt.sh's TEST_CMD.
# `vendor/burn-fused` is a SEPARATE cargo workspace: the root Cargo.toml
# `exclude`s it, because its crates inherit `workspace = true` from their own
# root and would otherwise resolve against ours. `cargo -p` cannot reach across
# an exclude - from the root, `cargo test -p burn-gdn2` is a guaranteed "package
# not found". So the crossing is not a flag, it is a `cd` plus a second cargo
# invocation, and that is what this file is: the one place the two workspaces
# are joined. It is CALLED from wt.sh's `test` branch rather than being a wt.sh
# subcommand, because a gate behind a subcommand is the defect this exists to
# end - the library had a CI workflow nobody had ever watched go red.
#
# WHAT IT RUNS. Exactly one cargo command, and it is verbatim the `test` step of
# job `ndarray-all-crates` in `.github/workflows/fused-library.yml`, so the local
# gate and the CI gate are the same gate rather than two that can drift.
# Measured here, cold, 2026-09-29, worktree `wt/libgate` off 46abc68: 686
# packages resolved, 253 tests run, 3 red, 1401 s. The `autodiff` targets
# (`autodiff_chunk`, `autodiff_nested_balanced`, `ops_batched_autodiff`) are
# in that run WITHOUT any feature flag - see the hazard noted in the commit.
#
# WHAT IT DOES NOT RUN, ON PURPOSE. No `cuda` feature, so no GPU is touched and
# none of the 18 `required-features = ["cuda", ...]` targets is even built -
# cargo SKIPS them, it does not pass them. tools/test_targets.py is the guard
# that proves each of those targets is declared at all; vendor/burn-fused/tools/
# gpu-gate.sh is the GPU half, and it refuses to pass without a visible device.
# There is deliberately no second `--features autodiff` cell: measured, it
# re-ran all 10 of burn-gdn2/burn-kda's binaries a second time for 781 s and
# added zero tests.
#
# RED IS THE POINT. Exits non-zero on any red, and does not filter, ignore,
# skip or soften a single test. It was written by the lane whose deliverable
# was to FIND the red ones.
set -uo pipefail
ROOT=${1:-$(cd "$(dirname "$0")/.." && pwd)}
FORK="$ROOT/vendor/burn-fused"
[ -d "$FORK" ] || { echo "lib_gate: no $FORK - run me from a dormouse checkout" >&2; exit 1; }

# One heavy thing at a time (AGENTS.md 1.5). wt.sh has already done this when it
# calls us; a bare invocation has not.
if pgrep -x cargo >/dev/null; then
    echo "lib_gate: another cargo is running. Waiting for it (one heavy thing at a time)..."
    while pgrep -x cargo >/dev/null; do sleep 20; done
fi

cd "$FORK" || exit 1
echo "── the fused library, every crate, default features, ndarray ──"
cargo test --workspace \
    --exclude burn-fused-benches --exclude cpu-probe --exclude launch-probe \
    --no-fail-fast
rc=$?

echo
if [ "$rc" -eq 0 ]; then
    echo "PASS  lib_gate: the fused library's CPU cell is green."
else
    echo "FAIL  lib_gate: the fused library's CPU cell is red (see above). Naming the"
    echo "      red tests IS the output of this gate; do not filter them."
fi
exit $rc
