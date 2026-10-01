#!/usr/bin/env bash
# PROVE THE AUDIT'S GATES CAN FAIL. Every gate landed by the 2026-10-01
# formula audit is perturbed here, reverted, and each perturbation is expected
# to turn a named test RED. Run from the crate root:
#
#   bash tests/oracle/falsify.sh
#
# Nothing here is a gate; it is the evidence that the gates ARE gates. A gate
# nobody has seen go red is a decoration (AGENTS.md 1.1: a fused/accelerated arm
# must be able to show it ran; the same applies to a claim). Every run prints
# the real cargo output. A perturbation that does NOT go red is printed as
# NOT-DETECTED and the script exits 1, so "the suite is green" can never be the
# only thing anyone reads.
#
# The restore is checked by SHA256, not by exit code: a `cp` back that silently
# no-ops is the failure mode this script exists to catch.
#
# Each perturbation asserts the EXACT number of sites it expects to edit, so a
# refactor that adds a second spelling of a constant cannot silently reduce a
# perturbation to a no-op. That check earned its keep twice. First: the NS
# quintic was spelled THREE times as a `let (a, b, c) = (...)` literal (the
# scalar arm, the batched arm, and the test-only
# `polar_orthogonalize_host_read` that pins the sync-free rewrite), so a
# perturbation that moved only the two production spellings would have perturbed
# one arm of a two-arm identity and blamed the other. Second: the first run of
# this script found the same blindness one level up - the basin gate tested a
# private COPY of the quintic, so NO perturbation of the production
# coefficients could reach it. Both are now one spelling and one reader:
# `NS_A`/`NS_B`/`NS_C`, and `the_quintic_basin_is_sqrt_7_over_3` reads them.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)   # .../burn-spectral/tests/oracle
cd "$HERE/../../../.."                # -> vendor/burn-fused
[ -d crates/burn-spectral ] || { echo "falsify: landed in $PWD, wanted vendor/burn-fused"; exit 2; }
PKG=crates/burn-spectral
SRC=$PKG/src/lib.rs
BACKUP=$(mktemp -d)
cp "$SRC" "$BACKUP/lib.rs"
ORIG_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)

restore() { cp "$BACKUP/lib.rs" "$SRC"; }
trap restore EXIT

FAILED_TO_DETECT=0

# expect_red <label> <expected-magnitude> <test-substring>
expect_red() {
  local label=$1 want=$2 test=$3 out rc
  out=$(cargo test -p burn-spectral --lib -- "$test" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    echo "NOT-DETECTED  $label — the gate stayed GREEN with this perturbation in"
    FAILED_TO_DETECT=1
    return
  fi
  if ! grep -qE "panicked at|assertion .* failed" <<<"$out"; then
    echo "DETECTED-WRONG-REASON  $label — red, but not from an assertion"
    tail -20 <<<"$out"
    FAILED_TO_DETECT=1
    return
  fi
  echo "DETECTED  $label"
  echo "          expected: $want"
  grep -E "panicked at|assertion .* failed|CLASS B|per-entry|sigma estimate|^  left|^ right" <<<"$out" | head -8
  echo
}

perturb_ns_c() {  # perturb_ns_c <delta-expression>
python3 - "$SRC" "$1" <<'PYEOF'
import sys
p, d = sys.argv[1], sys.argv[2]
s = open(p).read()
old = "const NS_C: f32 = 3.0 / 8.0;"
n = s.count(old)
# ONE: `NS_C` is a single definition used by the scalar arm, the batched arm
# and the test-only host-read fixture, so a perturbation cannot half-apply.
assert n == 1, f"expected 1 occurrence (the NS_C definition), found {n}"
open(p, "w").write(s.replace(old, f"const NS_C: f32 = 3.0 / 8.0 + {d};"))
print(f"  perturbed {n} occurrence with {d}")
PYEOF
}

echo "=== A. the NS quintic's last coefficient: 3/8 -> 3/8 + delta ==="
echo "    One site: the production constant. The FIRST version of this script"
echo "    perturbed three `let (a, b, c) = (...)` sites and the basin gate stayed"
echo "    GREEN, because that gate evaluated its OWN literal copy of the quintic."
echo "    A gate for a constant that tests a private copy is a gate for the copy;"
echo "    `p` now reads NS_A/NS_B/NS_C, so this perturbation is inside the gate."
perturb_ns_c 1e-3
expect_red "the quintic's c coefficient, +1e-3 (the basin gate)" \
  "p(1) and p'(1) are no longer 1 and 0, so the fixed point is neither at 1 nor \
   superattracting: the three properties this gate exists to pin all move" \
  "the_quintic_basin_is_sqrt_7_over_3"
restore

# A SECOND mutation of the same constant, at a magnitude the SECOND gate can
# see. The manifold gate's threshold is per-entry 1e-3, and a coefficient error
# d moves the fixed point by ~d, so the per-entry Gram error moves by ~d/4: at
# 1e-3 that gate is correctly GREEN, and reporting it as NOT-DETECTED would be
# reporting our own mutation size, not a defect in the gate. 1e-2 puts the
# fixed point ~1e-2 off the manifold, ~4x over its own threshold. Both facts are
# recorded rather than papered over: "the gate did not move at this size" and
# "the gate cannot move at any size" are different sentences, and only the
# second is a defect.
perturb_ns_c 1e-2
expect_red "the quintic's c coefficient, +1e-2 (the manifold gate)" \
  "retraction_holds_the_manifold_at_rank_64, because the map is no longer a \
   retraction: the fixed point sits ~1e-2 off the manifold, so the factor comes \
   back to somewhere else" \
  "retraction_holds_the_manifold_at_rank_64"
restore

echo "=== B. the prescale factor: 1.05 -> 1.00 in the SYNC-FREE arm only ==="
echo "    EXPECTED TO BE SEEN. Two corrections to the first version of this"
echo "    script, both found by running it:"
echo "      (i)  it predicted GREEN ('1.05 is only a nudge'), and that is a"
echo "           property of the ALGORITHM, not of this gate - the gate pins"
echo "           the sync-free path against a host-read FIXTURE, and a nudge"
echo "           is still a constant;"
echo "     (ii)  it perturbed BOTH arms, which cannot fail by construction: an"
echo "           identity checked against a fixture that moved with it is still"
echo "           an identity. ONE arm is the mutation this gate exists to catch."
python3 - "$SRC" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "sigma.mul_scalar(1.05)"
n = s.count(old)
# TWO: the scalar arm and the batched arm. The host-read fixture at
# `m = m.div_scalar(sigma * 1.05)` is DELIBERATELY left alone - that is what
# makes the identity check bite.
assert n == 2, f"expected 2 occurrences (scalar + batched, host-read fixture excluded), found {n}"
s = s.replace(old, "sigma.mul_scalar(1.0)")
open(p, "w").write(s)
print(f"  perturbed {n} occurrences, host-read fixture untouched")
PY
expect_red "the 1.05 prescale factor" \
  "sync_free_retraction_is_bit_identical_to_the_host_read_one, which compares \
   the sync-free path against the host-read fixture ELEMENTWISE. What is pinned \
   here is the BIT IDENTITY of two implementations, NOT the accuracy of the \
   factor - and it is a real check, because only one of the two arms moved" \
  "sync_free_retraction_is_bit_identical_to_the_host_read_one"
restore

echo "=== C. the class-A gate rewritten to assert the REFUTED claim ==="
echo "    The audit says the estimate is a LOWER bound, so the true prescaled"
echo "    value is unbounded above and the input is NOT pinned under 1. This"
echo "    perturbation asserts the opposite. If the gate cannot fail on the"
echo "    thing it exists to deny, it is not a gate."
python3 - "$SRC" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "                prescaled > 1.0,"
new = "                prescaled < 1.0,"
n = s.count(old)
assert n == 1, f"expected 1 occurrence, found {n}"
open(p, "w").write(s.replace(old, new))
print("  the gate now asserts the refuted claim (prescaled input pinned under 1.0)")
PY
expect_red "the refuted claim, asserted on purpose" \
  "the gate fires with a message naming the prescaled NS input, i.e. it CAN \
   fail and it fails for the reason it exists" \
  "the_sigma_estimate_is_a_lower_bound_and_the_1_05_factor_is_not_what_saves_it"
restore

echo "=== D. the class-B gates are RED, not merely #[ignore]d ==="
echo "    An #[ignore]d test that would PASS is a decoration; these must fail"
echo "    on their OWN assertion, with the magnitude they name."
for t in retraction_error_grows_with_spectral_spread sigma_max_estimate_diverges; do
  out=$(cargo test -p burn-spectral --lib -- --ignored --nocapture "$t" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    echo "NOT-DETECTED  $t — an #[ignore]d test that PASSES is a decoration"
    FAILED_TO_DETECT=1
  elif ! grep -q "the trainer's one-way max_ortho latch\|the start vector G\*1 cannot see" <<<"$out"; then
    echo "DETECTED-WRONG-REASON  $t — red, but not on the assertion it names"
    grep -E "panicked at" <<<"$out" | head -4
    FAILED_TO_DETECT=1
  else
    echo "DETECTED  $t"
    grep -E "CLASS B|per-entry|sigma estimate|panicked at" <<<"$out" | head -5
    echo
  fi
done
restore

echo "=== E. everything restored BY BYTES: the suite must be GREEN again ==="
NOW_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
if [ "$NOW_SHA" != "$ORIG_SHA" ]; then
  echo "RESTORE LEAKED — sha256 before $ORIG_SHA, after $NOW_SHA"
  FAILED_TO_DETECT=1
elif cargo test -p burn-spectral --lib >/dev/null 2>&1; then
  echo "GREEN  the restore is byte-identical (sha256 $NOW_SHA) and the suite passes"
else
  echo "NOT GREEN after restore, and the sha MATCHED: nothing leaked, so the suite"
  echo "  was ALREADY red when the script started. Section E cannot tell those two apart"
  echo "  by itself; the sha line above is what does."
  cargo test -p burn-spectral --lib 2>&1 | grep -E "panicked|failures:" | head -10
  FAILED_TO_DETECT=1
fi

echo
if [ $FAILED_TO_DETECT -eq 0 ]; then
  echo "PASS  every gate in this script was seen to fail, each on its own"
  echo "      assertion, and the restore is byte-identical."
else
  echo "FAIL  at least one gate could not be made to fail. The audit's gates are"
  echo "      not load-bearing and the claims they carry are not evidence."
fi
exit $FAILED_TO_DETECT
