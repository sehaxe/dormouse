#!/usr/bin/env bash
# PROVE THE ORACLE CAN FAIL.
#
# A test that has only ever been green is not evidence that it can fail, and
# docs/ORACLE.md is about tests that agree with the code under test. This
# perturbs the KERNEL (src/lib.rs), one wrong formula at a time, runs the suite,
# and records which test goes red and by how much. Then it restores the kernel
# and checks the md5.
#
# It mutates src/lib.rs, which is a source file. That is the point -- the
# mutants have to be the real implementation -- so it is careful about putting
# it back:
#
#   * ORIG is a copy taken at start-up, not a `git checkout`, so this works in
#     a dirty worktree and cannot discard someone else's edit;
#   * every `perturb` restores before patching, so a failure mid-script cannot
#     leave mutant N applied to mutant N+1's anchor;
#   * `trap ... EXIT` restores on Ctrl-C, a panic, or a normal exit;
#   * the final `md5sum` comparison prints whether the tree is back.
#
# Usage:  bash tests/oracle/mutate_kernel.sh
# Expected: baseline 4/4 green, every mutant >= 1 red, restored 4/4 green.
set -u

here="$(dirname "$(readlink -f "$0")")"   # .../burn-rmsnorm/tests/oracle
crate="$here/../.."                        # .../burn-rmsnorm
ws="$crate/../.."                          # .../burn-fused  (the workspace root)
cd "$crate" || exit 1
# Normalise, so a symlinked path cannot silently point cargo at the OUTER
# dormouse workspace (which `exclude`s vendor/burn-fused, and answers
# "not a member of the workspace" for every invocation).
here="$(cd "$here" && pwd)" || exit 1
crate="$(cd "$crate" && pwd)" || exit 1
ws="$(cd "$ws" && pwd)" || exit 1
[ -f "$ws/Cargo.toml" ] || { echo "no workspace root at $ws" >&2; exit 1; }
grep -q '^\[workspace\]' "$ws/Cargo.toml" || { echo "$ws is not a cargo workspace" >&2; exit 1; }
ORIG="$crate/src/lib.rs.orig.$$"

cleanup() { [ -f "$ORIG" ] && cp "$ORIG" src/lib.rs && rm -f "$ORIG"; }
trap cleanup EXIT INT TERM
cp src/lib.rs "$ORIG" || { echo "cannot snapshot src/lib.rs" >&2; exit 1; }
# The snapshot must be byte-identical to what we just wrote, or the md5 check
# at the end would compare against a file this script wrote rather than the
# tree's.
cmp -s src/lib.rs "$ORIG" || { echo "snapshot mismatch" >&2; exit 1; }

patch() { # $1 = text to find (must occur exactly once), $2 = replacement
  python3 - "$1" "$2" <<'PY' || { echo "    PATCH FAILED (anchor not unique)" >&2; return 1; }
import sys
p = "src/lib.rs"
s = open(p).read()
a, b = sys.argv[1], sys.argv[2]
if s.count(a) != 1:
    sys.exit("anchor occurs %d times: %r" % (s.count(a), a))
open(p, "w").write(s.replace(a, b))
PY
}

run() { # the per-test verdict, plus any assertion message worth reading
  (cd "$ws" && cargo test -p burn-rmsnorm --test rmsnorm_oracle 2>&1) \
    | grep -E '^test [a-z_]+ \.\.\.|^test result:|differs from the reference|only separate|gain ranges|same tensor at eps|^error(\[|:)|not being decided|is not the formula|not reaching the implementation' \
    | sed 's/^/    /'
}

perturb() { # $1 = label, $2 = anchor, $3 = replacement
  cp "$ORIG" src/lib.rs
  echo "=== MUTANT: $1"
  patch "$2" "$3" || return 1
  diff "$ORIG" src/lib.rs | grep '^[<>]' | sed 's/^/    /'
  run
  echo
}

echo "== 0. BASELINE (unperturbed) =="
run
echo

# M1: eps OUTSIDE the sqrt: x / (sqrt(mean(x^2)) + eps). The classic misreading,
# and the one the fixture's small-magnitude cases exist to decide.
perturb "M1 eps outside the sqrt" \
  '            .mean_dim(2)
            .add_scalar(self.eps)
            .sqrt();' \
  '            .mean_dim(2)
            .sqrt()
            .add_scalar(self.eps);'

# M2: eps dropped. Invisible on any ordinary-magnitude activation.
perturb "M2 eps dropped" \
  '            .mean_dim(2)
            .add_scalar(self.eps)
            .sqrt();' \
  '            .mean_dim(2)
            .sqrt();'

# M3: gain replaced by its mean -- the "the broadcast collapsed to a scalar" bug.
# A bare `into_scalar()` is NOT this mutant: that is a rank error, not a
# numerical one, and it would fail to compile rather than fail a comparison.
perturb "M3 gain collapsed to its mean" \
  '        (x / rms) * self.weight.val().reshape([1, 1, d])' \
  '        (x / rms) * self.weight.val().mean().reshape([1, 1, 1])'

# M4: reduction over the TIME axis (dim 1) instead of the feature axis.
perturb "M4 reduced over dim 1 (time)" \
  '            .mean_dim(2)' \
  '            .mean_dim(1)'

# M5: eps hardcoded to torch's default, ignoring the parameter this struct was
# constructed with. Caught only by the two tests that run the kernel twice.
perturb "M5 eps hardcoded to 1.1920929e-7" \
  '            .add_scalar(self.eps)' \
  '            .add_scalar(1.1920929e-7_f32)'

echo "== RESTORED =="
cp "$ORIG" src/lib.rs
if cmp -s src/lib.rs "$ORIG"; then
  echo "    src/lib.rs is byte-identical to the pre-run snapshot"
else
  echo "    !! src/lib.rs DOES NOT MATCH the snapshot" >&2
fi
run
