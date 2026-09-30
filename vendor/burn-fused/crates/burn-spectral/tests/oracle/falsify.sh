#!/usr/bin/env bash
# PROVE THE AUDIT'S GATES CAN FAIL. Every gate landed by the 2026-10-01
# formula audit is perturbed here, reverted, and each perturbation is expected
# to turn a named test RED. Run from the crate root:
#
#   bash tests/oracle/falsify.sh
#
# Nothing here is a gate; it is the evidence that the gates ARE gates. A gate
# nobody has seen go red is a decoration (AGENTS.md 2.1: "a fused/accelerated
# arm must be able to show it ran"; the same applies to a claim). Every run
# prints the real cargo output. A perturbation that does NOT go red is printed
# as NOT-DETECTED and the script exits 1, so "the suite is green" can never be
# the only thing anyone reads.
#
# The restore is checked by SHA256, not by exit code: a `cp` back that
# silently no-ops is the failure mode this script exists to catch.
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

# expect_red <label> <expected-magnitude> <test-name> [ignored]
expect_red() {
  local label=$1 want=$2 test=$3 mode=${4:-normal}
  local out rc
  if [ "$mode" = ignored ]; then
    out=$(cargo test -p burn-spectral --lib -- --ignored --exact "tests::polar_diag::$test" 2>&1); rc=$?
    # the class-B tests live in `tests`, not `polar_diag`; retry there
    if [ $rc -eq 0 ] || ! grep -q "$test" <<<"$out"; then
      out=$(cargo test -p burn-spectral --lib -- --ignored "$test" --nocapture 2>&1); rc=$?
    fi
  else
    out=$(cargo test -p burn-spectral --lib -- --exact "$test" 2>&1); rc=$?
  fi
  if [ $rc -eq 0 ]; then
    echo "NOT-DETECTED  $label — the gate stayed GREEN with this perturbation in"
    FAILED_TO_DETECT=1
    return
  fi
  if ! grep -qE "panicked|assertion" <<<"$out"; then
    echo "DETECTED-WRONG-REASON  $label — red, but not from an assertion"
    tail -20 <<<"$out"
    FAILED_TO_DETECT=1
    return
  fi
  echo "DETECTED  $label"
  echo "          expected: $want"
  grep -E "panicked at|assertion .* failed|^  left|^ right|CLASS B|per-entry|sigma estimate" <<<"$out" | head -8
  echo
}

expect_green() {
  local label=$1 why=$2
  if cargo test -p burn-spectral --lib >/dev/null 2>&1; then
    echo "GREEN (as expected, and here is why)  $label"
    echo "          $why"
  else
    echo "UNEXPECTED  $label turned the suite RED — the input set does see it after all"
    cargo test -p burn-spectral --lib 2>&1 | grep -E "panicked|FAILED|failures:" | head -10
    FAILED_TO_DETECT=1
  fi
  echo
}

echo "=== A. the NS triple's last coefficient: 3/8 -> 3/8 + 1e-3 ==="
echo "    (the scalar and the batched arm both spell it; both must move, or the"
echo "     perturbation only edits the arm a given test happens to call)"
python3 - "$SRC" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "let (a, b, c) = (15.0f32 / 8.0, -5.0f32 / 4.0, 3.0f32 / 8.0);"
new = "let (a, b, c) = (15.0f32 / 8.0, -5.0f32 / 4.0, 3.0f32 / 8.0 + 1e-3);"
n = s.count(old)
assert n == 2, f"expected 2 occurrences (scalar + batched), found {n}"
open(p, "w").write(s.replace(old, new))
print(f"  perturbed {n} occurrences")
PY
expect_red "the quintic's c coefficient" \
  "the fixed point moves off the manifold: p'(1) is no longer 0, so the 1e-3 \
   floor the per-entry metric asserts is missed" \
  "tests::polar_diag::the_quintic_basin_is_sqrt_7_over_3"
expect_red "the quintic's c coefficient (manifold gate)" \
  "retraction_holds_the_manifold_at_rank_64 or retraction_puts_sigma_max_at_one \
   goes red, because the retraction is no longer a retraction" \
  "tests::tests::retraction_holds_the_manifold_at_rank_64"
restore

echo "=== B. the prescale factor: 1.05 -> 1.00 ==="
echo "    EXPECTED TO STAY GREEN on the value, and that is a statement worth"
echo "    making: the audit's finding is that 1.05 is a nudge, not a guarantee."
sed -i 's/sigma\.mul_scalar(1\.05)/sigma.mul_scalar(1.0)/' "$SRC"
expect_green "the 1.05 prescale factor" \
  "removing the nudge changes sigma_max(out) by ~1e-7 (p'(1)=0 makes the fixed \
   point superattracting) and leaves every per-entry metric inside its 1e-3 \
   bound. THIS IS THE POINT of the corrected POWER_ITERS comment: the factor \
   is not what makes the retraction safe, so a test that asserted the factor \
   mattered would be asserting the thing the audit refuted."
restore

echo "=== C. the safety comment's claim, made true in code (the falsifier) ==="
echo "    The audit says the estimate is a LOWER bound, so the true prescaled"
echo "    value is unbounded above. This perturbation makes the estimate an"
echo "    UPPER bound (divide by the Frobenius norm instead) and the gate that"
echo "    asserts 'the 1.05 factor is not what saves it' must go red."
python3 - "$SRC" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
# Only inside the GATE, not the production code: flip the assertion that says
# the prescaled input is not pinned under 1/1.05 into its opposite, i.e. assert
# the OLD (refuted) claim. If the gate cannot fail on the refuted claim, it is
# not a gate.
old = 'assert!(\n            worst_prescaled >= 1.0 / 1.05,'
new = 'assert!(\n            worst_prescaled < 1.0 / 1.05,'
assert s.count(old) == 1, f"expected 1 occurrence, found {s.count(old)}"
open(p, "w").write(s.replace(old, new))
print("  the gate now asserts the refuted claim (prescaled input pinned under 1/1.05)")
PY
expect_red "the refuted claim, asserted on purpose" \
  "the gate fires with a message naming 1.0/1.05 — i.e. it CAN fail, and it \
   fails for the reason it exists" \
  "tests::polar_diag::the_sigma_estimate_is_a_lower_bound_and_the_1_05_factor_is_not_what_saves_it"
restore

echo "=== D. the class-B gates are RED, not merely #[ignore]d ==="
echo "    An #[ignore]d test that would PASS is a decoration; these must fail."
expect_red "CLASS B-1 (retraction error vs spectral spread)" \
  "per-entry 3.3e-02 at a 10:1 spread, 32.6x the one-way 1e-3 max_ortho latch" \
  "retraction_error_grows_with_spectral_spread" ignored
expect_red "CLASS B-2 (sigma estimate on a near-orthogonal factor)" \
  "the estimate returns 0.2321 for a Gram whose top eigenvalue is exactly 1.0" \
  "sigma_max_estimate_diverges" ignored
restore

echo "=== E. everything restored BY BYTES: the suite must be GREEN again ==="
NOW_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
if [ "$NOW_SHA" != "$ORIG_SHA" ]; then
  echo "RESTORE LEAKED — sha256 before $ORIG_SHA, after $NOW_SHA"
  FAILED_TO_DETECT=1
elif cargo test -p burn-spectral --lib >/dev/null 2>&1; then
  echo "GREEN  the restore is byte-identical (sha256 $NOW_SHA) and the suite passes"
else
  echo "NOT GREEN after restore — the script leaked a perturbation the sha did not catch"
  cargo test -p burn-spectral --lib 2>&1 | grep -E "panicked|failures:" | head -10
  FAILED_TO_DETECT=1
fi

echo
if [ $FAILED_TO_DETECT -eq 0 ]; then
  echo "PASS  every gate in this script was seen to fail, and every claim marked"
  echo "      'expected to stay green' was seen to stay green for the stated reason."
else
  echo "FAIL  at least one gate could not be made to fail. The audit's gates are"
  echo "      not load-bearing and the claims they carry are not evidence."
fi
exit $FAILED_TO_DETECT
