#!/usr/bin/env bash
# PROVE THE ORACLE CAN FAIL. Four perturbations, each reverted, each expected to
# turn a test RED with a stated magnitude. Run from the crate root.
#
#   bash tests/oracle/falsify.sh
#
# Nothing here is a gate; it is the evidence that the gates in
# tests/muon_oracle.rs are load-bearing. A gate nobody has seen go red is a
# decoration. Every run prints the real compiler/test output; a perturbation
# that does NOT go red is printed as NOT-DETECTED and the script exits 1, so
# "the test suite is green" can never be the only thing anyone reads.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)   # .../burn-muon-plus/tests/oracle
cd "$HERE/../../../.."                # -> vendor/burn-fused
[ -d crates/burn-muon-plus ] || { echo "falsify: landed in $PWD, wanted vendor/burn-fused"; exit 2; }
PKG=crates/burn-muon-plus
SRC=$PKG/src/lib.rs
BACKUP=$(mktemp -d)
cp "$SRC" "$BACKUP/lib.rs"
cp "$PKG/tests/oracle/upstream/utils/optim/muon_plus.py" "$BACKUP/muon_plus.py"

restore() {
  cp "$BACKUP/lib.rs" "$SRC"
  cp "$BACKUP/muon_plus.py" "$PKG/tests/oracle/upstream/utils/optim/muon_plus.py"
}
trap restore EXIT

FAILED_TO_DETECT=0
# expect_red <label> <expect-substring> <grep-pattern-over-cargo-output>
expect_red() {
  local label=$1 want=$2 pat=$3 out rc
  out=$(cargo test -p burn-muon-plus --test muon_oracle 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    echo "NOT-DETECTED  $label — the suite stayed GREEN with this perturbation in"
    FAILED_TO_DETECT=1
    return
  fi
  if ! grep -qE "$pat" <<<"$out"; then
    echo "DETECTED-WRONG-REASON  $label — red, but not for the expected reason"
    grep -E "^test |panicked|assertion|left|right|max\|" <<<"$out" | head -20
    FAILED_TO_DETECT=1
    return
  fi
  echo "DETECTED  $label"
  echo "          expected: $want"
  grep -E "panicked at|assertion .* failed|^  left|^ right|max\|ours" <<<"$out" | head -8
  echo
}

# expect_green <label> <why-this-input-cannot-see-it>
# A perturbation the fixture is KNOWN not to see, with the reason. Recorded, not
# hidden: a limit nobody wrote down is indistinguishable from a passing gate.
expect_green() {
  local label=$1 why=$2
  if cargo test -p burn-muon-plus --test muon_oracle >/dev/null 2>&1; then
    echo "GREEN (as expected, and here is why)  $label"
    echo "          $why"
  else
    echo "UNEXPECTED  $label turned the suite RED — the input set does see it after all"
    cargo test -p burn-muon-plus --test muon_oracle 2>&1 | grep -E "panicked|max\|ours" | head -4
    FAILED_TO_DETECT=1
  fi
  echo
}

echo "=== A. one digit wrong in NS_COEFFS (2.0315 -> 2.0316) ==="
sed -i 's/NS_COEFFS: (f32, f32, f32) = (3.4445, -4.775, 2.0315)/NS_COEFFS: (f32, f32, f32) = (3.4445, -4.775, 2.0316)/' "$SRC"
grep -q "2.0316" "$SRC" || { echo "SETUP FAILED: the sed did not apply"; exit 1; }
expect_red "NS_COEFFS last digit" \
  "the f32 nearest 2.0315 vs the f32 nearest 2.0316" \
  "ns_coeffs_are_the_authors_literal"
restore

echo "=== B. the composition order swapped: ColRow becomes row-then-col ==="
# `normalize` has the ColRow body twice, once under #[cfg(feature = \"cuda\")]
# and once under #[cfg(not(feature = \"cuda\"))]. Both must move, or the
# perturbation only edits the arm a CPU test never runs - which is how a
# real fix would ship broken for the other backend. Assert on the count so a
# future refactor that adds a third arm cannot silently skip this.
python3 - "$SRC" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "Self::norm_row(Self::norm_col(xm))"
n = s.count(old)
assert n == 2, f"expected 2 occurrences (cuda + non-cuda arms), found {n}"
open(p, "w").write(s.replace(old, "Self::norm_col(Self::norm_row(xm))"))
print(f"  swapped {n} occurrences")
PY
expect_red "ColRow composition order" \
  "max|ours - authors| on the col_row cases = 2.54e-1, i.e. the order signal" \
  "normalize_matches_the_authors_implementation"
restore

echo "=== C. the norm_col axis transposed: sum_dim(D-2) -> sum_dim(D-1) ==="
python3 - "$SRC" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = """        let col_norms = x
            .clone()
            .mul(x.clone())
            .sum_dim(D - 2)"""
new = old.replace("sum_dim(D - 2)", "sum_dim(D - 1)")
assert s.count(old) == 1, f"expected 1 occurrence, found {s.count(old)}"
open(p, "w").write(s.replace(old, new))
print("  norm_col now reduces over the row axis")
PY
expect_red "norm_col axis transposed" \
  "an O(1) difference: the authors unit-normalize columns, we would unit-normalize rows" \
  "normalize_matches_the_authors_implementation"
restore

echo "=== C2. the eps floor moved three decades (clamp_min(1e-7) -> 1e-4) ==="
echo "    EXPECTED TO STAY GREEN, and this is a statement about the FIXTURE, not a"
echo "    pass. The floor only binds when an axis norm falls between 1e-7 and 1e-4."
sed -i 's/\.clamp_min(1e-7);/.clamp_min(1e-4);/g' "$SRC"
expect_green "the eps floor value" \
  "every fixture axis norm is >= 2.0e-3 (grid values are k/498-1, k in 0..996), so the \
floor never binds and any floor in that range gives the identical output. It is also \
out of the authors' reach: their rule is sqrt(v^2+1e-7) with the epsilon INSIDE the \
root, ours is max(v,1e-7) outside it. Where the floor binds the two are different \
functions — at v=1e-6 they differ by 5e4 relative — so no single bar can be both \
tight enough to catch a transposed axis (C) and loose enough to admit tiny-norm \
inputs. Inputs with a tiny axis norm are therefore EXCLUDED, which is a stated \
limitation and not a silent hole."
restore

echo "=== D. the pinned upstream file is edited by one byte ==="
sed -i 's/3\.4445, -4\.7750, 2\.0315/3.4445, -4.7750, 2.0316/' \
  "$PKG/tests/oracle/upstream/utils/optim/muon_plus.py"
PY=${PY:-/tmp/muon-oracle-venv/bin/python}
[ -x "$PY" ] || { echo "falsify: no interpreter at $PY - see tests/oracle/PROVENANCE.md"; exit 2; }
out=$("$PY" "$PKG/tests/oracle/gen_oracle.py" 2>&1); rc=$?
if [ $rc -eq 0 ]; then
  echo "NOT-DETECTED  the pin check — gen_oracle.py regenerated the fixture from an edited pin"
  FAILED_TO_DETECT=1
else
  echo "DETECTED  the pin check"
  echo "          expected: gen_oracle.py exits non-zero and says PIN ROTED"
  grep -E "PIN ROTED|expected|got" <<<"$out" | head -4
fi
echo
restore

echo "=== E. everything restored: the suite must be GREEN again ==="
if cargo test -p burn-muon-plus --test muon_oracle >/dev/null 2>&1; then
  echo "GREEN  the restore is clean"
else
  echo "NOT GREEN after restore — the script leaked a perturbation"
  FAILED_TO_DETECT=1
fi

exit $FAILED_TO_DETECT
