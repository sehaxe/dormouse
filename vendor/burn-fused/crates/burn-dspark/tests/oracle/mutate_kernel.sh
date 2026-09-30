#!/usr/bin/env bash
# PROVE THE TWO NEW ORACLES CAN FAIL.
#
# A test that has only ever been green is not evidence that it can fail. The
# two tier-(a) rows this lane added are RED on purpose, naming two real defects
# in burn-dspark's confidence term and one in the e2m1 tie rule. That is the
# "can fail" demonstration already: they are failing, on shipping code, with the
# cause in the message.
#
# This script covers the OTHER direction, which a red test cannot: it mutates
# the GREEN assertions' subjects and shows each goes red, so a future edit that
# silently weakens the comparison (a tolerance widened, a term dropped from the
# check, a case removed from the fixture) is caught rather than inherited.
#
# It mutates SOURCE files, which is the point -- the mutants have to be the real
# implementation -- so it snapshots first and restores on any exit path:
#
#   * ORIG is a copy taken at start-up, not a `git checkout`, so this works in a
#     dirty worktree and cannot discard someone else's edit;
#   * every `perturb` restores before patching, so a failure mid-script cannot
#     leave mutant N applied to mutant N+1's anchor;
#   * `trap ... EXIT` restores on Ctrl-C, a panic, or a normal exit;
#   * the final `cmp` prints whether the tree is back.
#
# Usage:  bash tests/oracle/mutate_kernel.sh
# Expected: baseline as described, every mutant >= 1 red, tree restored.
set -u

here="$(dirname "$(readlink -f "$0")")"   # .../burn-dspark/tests/oracle
crate="$here/../.."                        # .../burn-dspark
libws="$crate/../.."                       # .../burn-fused  (the library workspace)
repo="$libws/../.."                        # .../dormouse    (owns act_quant.rs)
cd "$crate" || exit 1
here="$(cd "$here" && pwd)"; crate="$(cd "$crate" && pwd)"
libws="$(cd "$libws" && pwd)"; repo="$(cd "$repo" && pwd)"
[ -f "$libws/Cargo.toml" ] || { echo "no library workspace at $libws" >&2; exit 1; }
[ -f "$repo/Cargo.toml" ] || { echo "no repo root at $repo" >&2; exit 1; }

DSPARK_SRC="$crate/src/lib.rs"
E2M1_SRC="$repo/crates/dormouse-core/src/act_quant.rs"
for f in "$DSPARK_SRC" "$E2M1_SRC"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

ORIG_D="$DSPARK_SRC.orig.$$"
ORIG_E="$E2M1_SRC.orig.$$"
cleanup() {
  [ -f "$ORIG_D" ] && cp "$ORIG_D" "$DSPARK_SRC" && rm -f "$ORIG_D"
  [ -f "$ORIG_E" ] && cp "$ORIG_E" "$E2M1_SRC" && rm -f "$ORIG_E"
}
trap cleanup EXIT INT TERM
cp "$DSPARK_SRC" "$ORIG_D" || exit 1
cp "$E2M1_SRC" "$ORIG_E" || exit 1
cmp -s "$DSPARK_SRC" "$ORIG_D" || { echo "dspark snapshot mismatch" >&2; exit 1; }
cmp -s "$E2M1_SRC" "$ORIG_E" || { echo "e2m1 snapshot mismatch" >&2; exit 1; }

patch() { # $1 = file, $2 = text to find (must occur once), $3 = replacement
  python3 - "$1" "$2" "$3" <<'PY' || { echo "    PATCH FAILED (anchor not unique)" >&2; return 1; }
import sys
p, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(a) != 1:
    sys.exit("anchor occurs %d times: %r" % (s.count(a), a))
open(p, "w").write(s.replace(a, b))
PY
}

run_dspark() {
  (cd "$libws" && cargo test -p burn-dspark --features training \
      --test dspark_loss_oracle 2>&1) \
    | grep -E '^test [a-z_]+ \.\.\.|^test result:' | sed 's/^/    /'
}
run_e2m1() {
  (cd "$repo" && cargo test -p dormouse-core --test e2m1_oracle 2>&1) \
    | grep -E '^test [a-z_]+ \.\.\.|^test result:' | sed 's/^/    /'
}

perturb() { # $1 = label, $2 = file, $3 = anchor, $4 = replacement
  cp "$ORIG_D" "$DSPARK_SRC"; cp "$ORIG_E" "$E2M1_SRC"
  echo "=== MUTANT: $1"
  patch "$2" "$3" "$4" || return 1
  diff "$(basename "$2").orig" "$2" 2>/dev/null | grep '^[<>]' | sed 's/^/    /' \
    || diff "$ORIG_D" "$2" 2>/dev/null | grep '^[<>]' | sed 's/^/    /'
  case "$2" in
    *act_quant.rs) run_e2m1 ;;
    *)             run_dspark ;;
  esac
  echo
}

echo "== 0. BASELINE (unperturbed) =="
echo "-- burn-dspark: 2 red ON PURPOSE (the confidence term), 6 green"
run_dspark
echo "-- dormouse-core e2m1: 1 red ON PURPOSE (the tie rule), 3 green"
run_e2m1
echo

# ── burn-dspark mutants ─────────────────────────────────────────────────────
# M1: drop the position decay. `position_weights` returns ones instead of
# exp(-k/gamma). The fixture's block7_exact carries a DECAYED denominator
# (3.73521233 against 7.0 undecayed), so the CE and L1 terms must both move.
perturb "M1 dspark: position decay dropped" "$DSPARK_SRC" \
  '.map(|k| (-(k as f64) / gamma).exp() as f32)' \
  '.map(|_k| 1.0f32)'

# M2: drop the mask from the weighted mean. Without the mask every padded
# position contributes, which the partial-mask cases (main, tiny_mask) see.
perturb "M2 dspark: mask dropped from the mean" "$DSPARK_SRC" \
  'let wm = w.clone() * mask.clone();' \
  'let wm = w.clone(); let _ = &mask;'

# M3: the accept-rate target loses its 0.5 factor, turning total-variation
# distance into L1. Small numerically, but the fixture's `aligned_identical`
# ceiling case pins accept_rate == 1 exactly and a doubled term breaks it.
perturb "M3 dspark: accept-rate 0.5 factor doubled" "$DSPARK_SRC" \
  'tv.mul_scalar(0.5).neg().add_scalar(1.0).clamp(0.0, 1.0)' \
  'tv.neg().add_scalar(1.0).clamp(0.0, 1.0)'

# ── e2m1 mutants ────────────────────────────────────────────────────────────
# M4: the grid loses 0.75 AGAIN. This is the 9b343d3 defect, re-introduced, and
# the grid test must catch it against torchao's table.
perturb "M4 e2m1: 0.75 back in the grid" "$E2M1_SRC" \
  'pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];' \
  'pub const E2M1: [f32; 8] = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0];'

# M5: the block scale maps onto 1 instead of the format's max -- the OTHER half
# of the 9b343d3 defect, and the one that made every level above 1 dead code.
perturb "M5 e2m1: block scale maps onto 1" "$E2M1_SRC" \
  'ActFormat::Fp4 => 6.0,' \
  'ActFormat::Fp4 => 1.0,'

# M6: ties broken DOWN instead of up. The reference is ties-to-EVEN, so this
# must NOT turn the tie test green -- it should move the disagreement to the
# other three interior ties and still be red. A mutant that "fixes" a red test
# is the most valuable line in this script: it proves the test is pinned to the
# reference's rule and not merely to "not ties-up".
perturb "M6 e2m1: ties broken DOWN (still != the reference)" "$E2M1_SRC" \
  'mask_fill(a.clone().greater_equal_scalar(lo), *level)' \
  'mask_fill(a.clone().greater_scalar(lo), *level)'

echo "== RESTORED =="
cp "$ORIG_D" "$DSPARK_SRC"
cp "$ORIG_E" "$E2M1_SRC"
if cmp -s "$DSPARK_SRC" "$ORIG_D" && cmp -s "$E2M1_SRC" "$ORIG_E"; then
  echo "    both source files are byte-identical to the pre-run snapshot"
else
  echo "    !! a source file DOES NOT MATCH the snapshot" >&2
fi
echo "-- burn-dspark back to 2 red / 6 green:"
run_dspark
echo "-- e2m1 back to 1 red / 3 green:"
run_e2m1
