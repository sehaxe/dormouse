#!/usr/bin/env bash
# PROVE THE CUDA-KERNEL ORACLE CAN FAIL.
#
# The sibling `mutate_kernel.sh` perturbs `src/lib.rs` — the TENSOR path — five
# times and shows which test goes red. `src/fused.rs` had no such evidence at
# all: the kernel shipped, it is `#[cfg(feature = "cuda")]`, and until
# 2026-09-30 nothing had ever compared its output to anything, on any device.
#
# So this is the same experiment on the other file, on the GPU: five wrong
# kernels, each a CUDA twin of M1..M5 in the sibling, and each must turn at
# least one test in `tests/rmsnorm_kernel_cuda.rs` red. The four tests in that
# file own four different questions, and the split matters when reading the
# output below:
#
#   the_fused_kernel_matches_both_upstreams    the kernel's OUTPUT vs 2 upstreams
#   eps_reaches_the_fused_kernel               the kernel's OWN PARAMETERS
#   the_fused_kernel_declines_on_an_autodiff…  the seam (asked vs ran)
#   …_on_the_trainers_eval_snapshot           ditto, `valid()` shape
#
# It mutates `src/fused.rs`, a source file. Same discipline as the sibling: the
# mutants have to be the real implementation, so it snapshots rather than
# `git checkout`s (works in a dirty worktree, cannot eat someone else's edit),
# restores before every patch, traps on every exit path, and md5-compares at the
# end.
#
# Run:  bash tests/oracle/mutate_fused_kernel.sh
# Needs a free GPU and one build lock held by the caller (each mutant is a
# rebuild). Expected: baseline 4/4 green, every mutant >= 1 red, restored 4/4.
set -u

here="$(dirname "$(readlink -f "$0")")"   # .../dormouse-rmsnorm/tests/oracle
crate="$here/../.."                        # .../dormouse-rmsnorm
ws="$crate/../.."                          # .../dormouse-fused  (the workspace root)
cd "$crate" || exit 1
# Normalise, so a symlinked path cannot silently point cargo at the OUTER
# dormouse workspace (which `exclude`s vendor/dormouse-fused).
here="$(cd "$here" && pwd)" || exit 1
crate="$(cd "$crate" && pwd)" || exit 1
ws="$(cd "$ws" && pwd)" || exit 1
[ -f "$ws/Cargo.toml" ] || { echo "no workspace root at $ws" >&2; exit 1; }
grep -q '^\[workspace\]' "$ws/Cargo.toml" || { echo "$ws is not a cargo workspace" >&2; exit 1; }
ORIG="$crate/src/fused.rs.orig.$$"

cleanup() { [ -f "$ORIG" ] && cp "$ORIG" src/fused.rs && rm -f "$ORIG"; }
trap cleanup EXIT INT TERM
cp src/fused.rs "$ORIG" || { echo "cannot snapshot src/fused.rs" >&2; exit 1; }
cmp -s src/fused.rs "$ORIG" || { echo "snapshot mismatch" >&2; exit 1; }
md5_before=$(md5sum src/fused.rs | cut -d' ' -f1)

patch() { # $1 = text to find (must occur exactly once), $2 = replacement
  python3 - "$1" "$2" <<'PY' || { echo "    PATCH FAILED (anchor not unique)" >&2; return 1; }
import sys
p = "src/fused.rs"
s = open(p).read()
a, b = sys.argv[1], sys.argv[2]
if s.count(a) != 1:
    sys.exit("anchor occurs %d times: %r" % (s.count(a), a))
open(p, "w").write(s.replace(a, b))
PY
}

run() { # the per-test verdict, plus an assertion message worth reading
  (cd "$ws" && cargo test -p dormouse-rmsnorm --test rmsnorm_kernel_cuda --features cuda 2>&1) \
    | grep -E '^test [a-z_]+ \.\.\.|^test result:|differs from the reference|moved the FUSED|is not reaching|did not run|is not the implementation|^error(\[|:)' \
    | sed 's/^/    /'
}

perturb() { # $1 = label, $2 = anchor, $3 = replacement
  cp "$ORIG" src/fused.rs
  echo "=== MUTANT: $1"
  patch "$2" "$3" || return 1
  diff "$ORIG" src/fused.rs | grep '^[<>]' | sed 's/^/    /'
  run
  echo
}

echo "== 0. BASELINE (unperturbed) =="
run
echo

# F1: eps OUTSIDE the sqrt — x / (sqrt(mean(x^2)) + eps). The classic
# misreading, and the twin of the sibling's M1.
perturb "F1 eps outside the sqrt" \
  '        partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();' \
  '        partial[0] = partial[0].sqrt() / F::cast_from(d as f32).sqrt() + F::cast_from(eps);'

# F2: eps dropped. Invisible on any ordinary-magnitude activation; the fixture's
# small-magnitude cases exist to catch it.
perturb "F2 eps dropped" \
  '        partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();' \
  '        partial[0] = (partial[0] / F::cast_from(d as f32)).sqrt();'

# F3: the per-feature gain collapsed to its first element — "the broadcast
# became a scalar", on a kernel that indexes `w[i]` by hand.
perturb "F3 gain collapsed to w[0]" \
  '        out[base + i] = x[base + i] * inv * w[i];' \
  '        out[base + i] = x[base + i] * inv * w[0];'

# F4: the reduction's `/d` dropped — a sum presented as a mean. This is the
# kernel's twin of the sibling's M4 (wrong reduction), and on `d = 8` it moves
# the output by sqrt(8) - 1 = 1.83.
perturb "F4 the mean dropped" \
  '        partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();' \
  '        partial[0] = (partial[0] + F::cast_from(eps)).sqrt();'

# F5: eps hardcoded to torch's default, ignoring the argument the kernel was
# called with. Caught only by a test that runs the KERNEL twice at two epses.
perturb "F5 eps hardcoded to 1.1920929e-7" \
  '        partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();' \
  '        partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(1.1920929e-7_f32)).sqrt();'

echo "== RESTORED =="
cp "$ORIG" src/fused.rs
if cmp -s src/fused.rs "$ORIG"; then
  md5_after=$(md5sum src/fused.rs | cut -d' ' -f1)
  if [ "$md5_before" = "$md5_after" ]; then
    echo "    src/fused.rs is byte-identical to the pre-run snapshot (md5 $md5_after)"
  else
    echo "    !! md5 changed: $md5_before -> $md5_after" >&2
  fi
else
  echo "    !! src/fused.rs DOES NOT MATCH the snapshot" >&2
fi
run
