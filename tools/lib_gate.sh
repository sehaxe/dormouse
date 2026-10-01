#!/usr/bin/env bash
# THE CPU gate for the fused library (vendor/dormouse-fused). One command:
#
#     tools/lib_gate.sh [worktree-root]     # default: the repo this script is in
#
# WHY IT IS A SCRIPT AND NOT A SECOND `-p` ON tools/wt.sh's TEST_CMD.
# `vendor/dormouse-fused` is a SEPARATE cargo workspace: the root Cargo.toml
# `exclude`s it, because its crates inherit `workspace = true` from their own
# root and would otherwise resolve against ours. `cargo -p` cannot reach across
# an exclude - from the root, `cargo test -p dormouse-gdn2` is a guaranteed "package
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
# that proves each of those targets is declared at all; vendor/dormouse-fused/tools/
# gpu-gate.sh is the GPU half, and it refuses to pass without a visible device.
# There is deliberately no second `--features autodiff` cell: measured, it
# re-ran all 10 of dormouse-gdn2/dormouse-kda's binaries a second time for 781 s and
# added zero tests.
#
# RED IS THE POINT. Exits non-zero on any red, and does not filter, ignore,
# skip or soften a single test. It was written by the lane whose deliverable
# was to FIND the red ones.
set -uo pipefail
ROOT=${1:-$(cd "$(dirname "$0")/.." && pwd)}
FORK="$ROOT/vendor/dormouse-fused"
[ -d "$FORK" ] || { echo "lib_gate: no $FORK - run me from a dormouse checkout" >&2; exit 1; }

# One heavy thing at a time (AGENTS.md 1.5). wt.sh has already done this when it
# calls us; a bare invocation has not.
if pgrep -x cargo >/dev/null; then
    echo "lib_gate: another cargo is running. Waiting for it (one heavy thing at a time)..."
    while pgrep -x cargo >/dev/null; do sleep 20; done
fi

cd "$FORK" || exit 1

# The f64 fixtures must be the output of the generator that ships beside them,
# and this is the only place that can be checked. `ca45600` fixed a head-major
# `g` layout bug in `crates/dormouse-gdn2/tools/gen_reference_f64.py` and
# regenerated `ref_f64.bin` and `ref_f64_faults.bin` but NOT
# `ref_f64_broad.bin`; the 1000-case sweep was left holding the bug, and both
# `oracle_breadth` and `oracle_chunk` read it as a kernel defect for a day
# (976/1000 cases, worst 8.940234e-01). No cargo test can see this, because the
# committed bytes ARE what the test compares against - so it runs first, and it
# needs only numpy, not a build.
if command -v python3 >/dev/null 2>&1; then
    echo "── dormouse-gdn2 f64 fixtures vs their generator ──"
    # `fixture_rc`, not `rc`: the cargo cell below assigns `rc=$?` and would
    # overwrite a failure recorded here.
    fixture_rc=0
    (cd crates/dormouse-gdn2 && python3 tools/check_f64_fixtures.py) || fixture_rc=1
    echo
else
    echo "── dormouse-gdn2 f64 fixtures: SKIPPED, no python3. The oracle layer is"
    echo "   UNVERIFIED in this run; the binary-tests cell below still compares"
    echo "   against whatever bytes are committed."
    echo
    fixture_rc=0
fi

echo "── the fused library, every crate, default features, ndarray ──"
cargo test --workspace \
    --exclude dormouse-fused-benches --exclude cpu-probe --exclude launch-probe \
    --no-fail-fast
rc=$?

# `binary-tests` is NOT a default of dormouse-gdn2, and that is deliberate: the
# facade classifies it in NOT_PUBLIC as a fork-CI marker no crate reads, so a
# default would make every consumer of the facade pay for a flag it never uses
# (and cargo refuses a facade `default` naming a feature it cannot forward).
# But it gates two of the crate's most valuable tests - `oracle_breadth`'s
# 1000-case sweep against the f64 oracle and `oracle_chunk`'s 5-chunk-size
# sweep over the same cases - and a gate that does not pass it is a gate that
# cannot see them. So ask for it explicitly. If this cell ever goes to 0 tests
# for dormouse-gdn2, THIS is the line that stopped asking, not a green suite.
echo
echo "── dormouse-gdn2 fixture-backed tests (binary-tests) ──"
cargo test -p dormouse-gdn2 --features binary-tests \
    --test oracle_breadth --test oracle_chunk --no-fail-fast
rc2=$?
[ "$rc" -eq 0 ] || rc=$rc2
[ "$rc2" -eq 0 ] || rc=$rc2

echo
[ "$fixture_rc" -eq 0 ] || rc=1
if [ "$rc" -eq 0 ]; then
    echo "PASS  lib_gate: the fused library's CPU cell is green."
else
    echo "FAIL  lib_gate: the fused library's CPU cell is red (see above). Naming the"
    echo "      red tests IS the output of this gate; do not filter them."
fi
exit $rc
