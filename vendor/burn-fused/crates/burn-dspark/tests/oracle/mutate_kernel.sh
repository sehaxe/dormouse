#!/usr/bin/env bash
# PROVE THE ORACLES CAN FAIL.
#
# A test that has only ever been green is not evidence that it can fail. Both
# tier-(a) rows this lane added WERE red on purpose, naming two real defects in
# burn-dspark's confidence term and one in the e2m1 tie rule -- and all three
# have since been fixed, so the baseline is now 10 + 4 green.
#
# That is the reason this script exists at all, and it is worth being explicit
# about the direction it runs in. A red test proves it can fail, but only while
# it is red; once the defect is fixed the red is gone and so is the evidence.
# The mutants are what carry it forward: each one edits the REAL
# implementation and shows the corresponding gate going red, so a future edit
# that silently weakens the comparison -- a tolerance widened, a term dropped
# from the check, a case removed from the fixture, a `.detach()` deleted -- is
# caught rather than inherited.
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
# A DEAD ANCHOR IS A FAILURE OF THIS SCRIPT, and `patch()` is what makes that
# true: it exits non-zero when the anchor is not unique, `perturb` returns, and
# the mutant runs no test. That is load-bearing and it failed silently once --
# M6's anchor was removed by `dc5d667`, so the single most valuable mutant in
# the file ran no test and printed nothing about it. `patch()` is now also
# asserted to have been reached: the mutant list is swept for "PATCH FAILED"
# at the end, which turns "the script ran" into a different claim from "every
# mutant in the script ran".
#
# Usage:  bash tests/oracle/mutate_kernel.sh
# Expected: 10 + 4 green at baseline, every mutant >= 1 red, no
# "PATCH FAILED" anywhere, tree byte-identical to the snapshot at the end.
#
# CARGO: run under the build lock like everything else heavy --
#   tools/build_lock.sh run mut -- bash vendor/burn-fused/crates/burn-dspark/tests/oracle/mutate_kernel.sh
# It compiles and runs two test targets seven times.
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

# Every "PATCH FAILED" is recorded, and the sweep fails at the end if any
# occurred. Without this, a mutant whose anchor has gone stale looks exactly
# like a mutant that passed.
PATCH_FAILURES=0

patch() { # $1 = file, $2 = text to find (must occur once), $3 = replacement
  python3 - "$1" "$2" "$3" <<'PY' || { echo "    PATCH FAILED (anchor not unique)" >&2; PATCH_FAILURES=$((PATCH_FAILURES+1)); return 1; }
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

RED_SEEN=0
GREEN_MUTANT=""

perturb() { # $1 = label, $2 = file, $3 = anchor, $4 = replacement
  cp "$ORIG_D" "$DSPARK_SRC"; cp "$ORIG_E" "$E2M1_SRC"
  echo "=== MUTANT: $1"
  # A failed anchor is NOT a passing mutant. Restore, record, and say so here
  # rather than only in the trailer, so a reader scrolling the sweep sees it in
  # the mutant's own slot.
  if ! patch "$2" "$3" "$4"; then
    echo "    !! THIS MUTANT DID NOT RUN -- its anchor no longer matches the"
    echo "       source. It is dead, not green. Re-anchor it."
    cp "$ORIG_D" "$DSPARK_SRC"; cp "$ORIG_E" "$E2M1_SRC"
    return 1
  fi
  # Diff the mutated file against ITS OWN snapshot. The two constants
  # deliberately point at different crates, so diffing "$2" against "$ORIG_D"
  # unconditionally printed two unrelated files at each other -- 60 lines of
  # noise that buried the one line the mutant changed.
  case "$2" in
    *act_quant.rs) diff "$ORIG_E" "$2" ;;
    *)             diff "$ORIG_D" "$2" ;;
  esac | grep '^[<>]' | sed 's/^/    /'
  local out
  case "$2" in
    *act_quant.rs) out="$(run_e2m1)" ;;
    *)             out="$(run_dspark)" ;;
  esac
  echo "$out"
  # The mutation's whole job is to turn a green assertion red. If the gate
  # stayed green, this mutant has stopped testing anything -- and an unkillable
  # mutant means the gate underneath it is not measuring what it claims.
  if echo "$out" | grep -qE '^test result: FAILED| [0-9]+ failed'; then
    RED_SEEN=$((RED_SEEN+1))
  else
    GREEN_MUTANT="$1"
    echo "    !! THIS MUTANT LEFT THE GATE GREEN. Either the mutation is" >&2
    echo "       equivalent (say so here deliberately) or the assertion it is" >&2
    echo "       meant to kill is not measuring what its name claims." >&2
  fi
  echo
}

echo "== 0. BASELINE (unperturbed) =="
echo "-- burn-dspark: 10 green. These used to be 2 red ON PURPOSE; both are"
echo "   fixed (the confidence term's numerical space and the Option head, in"
echo "   8c3bd2a) and a third gradient-only defect since (dc5d667's sibling in"
echo "   src/lib.rs: an un-detached target, and the BCE kink at x == 0). The"
echo "   demonstration that they CAN fail is this script, not a red baseline."
run_dspark
echo "-- dormouse-core e2m1: 4 green. The tie rule was 1 red ON PURPOSE until"
echo "   dc5d667 fixed it."
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

# M6: the tie rule inverted. THE MOST VALUABLE LINE IN THIS SCRIPT, per the
# author's note at :162: a mutant that "fixes" a red test proves the test is
# pinned to the reference's RULE and not merely to "not ties-up".
#
# RE-ANCHORED 2026-09-30. This used to patch
#   'mask_fill(a.clone().greater_equal_scalar(lo), *level)'
# which `dc5d667` REPLACED: the ties-to-even-code rule is an `if i % 2 == 0`
# parity predicate, so that string occurred ZERO times afterwards. The script's
# `patch()` hard-fails on a non-unique anchor, so M6 printed
# "PATCH FAILED (anchor not unique)" and ran no test at all - the strongest
# anti-vacuity evidence in the file silently stopped running, and nothing said
# so. A dead anchor in a mutation sweep is the same defect class as a gate
# nothing runs: the artifact looks like coverage.
#
# The anchor is now the parity predicate itself, and the mutation is the
# INVERSION of it - every tie goes the other way. Inverting the predicate makes
# odd codes claim on a tie and even codes leave it, which is ties-to-ODD-code:
# still wrong at exactly the four ties the reference resolves to even, so the
# test must stay RED and move to a DIFFERENT set of complaints (the 3 that were
# agreeing plus the 4 that were not). If this mutant ever turns the test GREEN,
# the test is pinned to "not ties-up" rather than to the reference.
perturb "M6 e2m1: tie parity INVERTED (ties-to-odd-code, still != reference)" "$E2M1_SRC" \
  'let claimed = if i % 2 == 0 {' \
  'let claimed = if i % 2 == 1 {'

echo "== RESTORED =="
cp "$ORIG_D" "$DSPARK_SRC"
cp "$ORIG_E" "$E2M1_SRC"
if cmp -s "$DSPARK_SRC" "$ORIG_D" && cmp -s "$E2M1_SRC" "$ORIG_E"; then
  echo "    both source files are byte-identical to the pre-run snapshot"
else
  echo "    !! a source file DOES NOT MATCH the snapshot" >&2
fi
echo "-- burn-dspark back to 10 green:"
run_dspark
echo "-- e2m1 back to 4 green:"
run_e2m1

# The assertions that make "the sweep ran" mean "every mutant in the sweep ran
# AND each one killed something". M6's stale anchor cost this file its best
# mutant without a word; these are the lines that would have said so.
if [ "$PATCH_FAILURES" -ne 0 ]; then
  echo "    !! $PATCH_FAILURES mutant(s) did not run: their anchor no longer" \
       "matches the source. An anchor that has gone stale is a DEAD MUTANT," \
       "not a passing one -- re-anchor or delete it." >&2
  exit 1
fi
if [ -n "$GREEN_MUTANT" ]; then
  echo "    !! '$GREEN_MUTANT' left its gate GREEN." >&2
  exit 1
fi
echo "-- sweep verdict: $RED_SEEN mutants, every one red, 0 patch failures," \
     "0 dead anchors, tree byte-identical"
