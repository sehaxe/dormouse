#!/usr/bin/env bash
# PROVE THE burn-kda ORACLE CAN FAIL, IN BOTH DIRECTIONS.
#
# A test that has only ever been green is not evidence that it can fail. Two of
# the five green arms in `tests/kda_oracle.rs` are only green because our code
# currently agrees with FLA's executed code; this script is what stops that
# claim from being inherited unexamined.
#
# It covers BOTH directions, and the second one is the more valuable:
#
#   A. MUTANTS THAT BREAK A GREEN. Perturb our source and show the corresponding
#      green arm goes red. This is the ordinary "can it fail" demonstration.
#
#   B. MUTANTS THAT "FIX" A RED. The two red tests are red because our code
#      disagrees with FLA. A mutant that moves our code TOWARD FLA's formula
#      turns a red green. That is the strongest statement available: it proves
#      the red is pinned to the reference's actual rule and not merely to "not
#      what we happen to have written". The e2m1 lane's M6 is the precedent.
#      Two of these are shipped here, since they are the fixes the owner may
#      choose:
#        - moving `exp(A)` outside the softplus in DecayFn::Softplus;
#        - passing `head_k_dim**-0.5` instead of 1.0 as the chunk `scale`.
#      Run the full oracle after each to see the red go green AND check that no
#      OTHER test changed state.
#
# It mutates SOURCE files, so it snapshots first and restores on any exit path:
#   * ORIG is a copy taken at start-up, not a `git checkout`, so this works in a
#     dirty worktree and cannot discard someone else's edit;
#   * every `perturb` restores before patching, so a mid-script failure cannot
#     leave mutant N applied to mutant A's anchor;
#   * `trap ... EXIT` restores on Ctrl-C, a panic, or a normal exit;
#   * the final `cmp` reports whether the tree is back.
#
# Usage:  bash tests/oracle/falsify.sh
# Expected: baseline 5 green / 2 red; every A-mutant >= 1 red; every B-mutant
#           turns its target red green; tree restored.
set -u

here="$(dirname "$(readlink -f "$0")")"   # .../burn-kda/tests/oracle
crate="$here/../.."                        # .../burn-kda
libws="$crate/../.."                       # .../burn-fused (its own workspace)
cd "$crate" || exit 1
here="$(cd "$here" && pwd)"; crate="$(cd "$crate" && pwd)"; libws="$(cd "$libws" && pwd)"
[ -f "$libws/Cargo.toml" ] || { echo "no library workspace at $libws" >&2; exit 1; }

LIB="$crate/src/lib.rs"
FUSED="$crate/src/fused.rs"
for f in "$LIB" "$FUSED"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

ORIG_L="$LIB.orig.$$"
ORIG_F="$FUSED.orig.$$"
cleanup() {
  [ -f "$ORIG_L" ] && cp "$ORIG_L" "$LIB" && rm -f "$ORIG_L"
  [ -f "$ORIG_F" ] && cp "$ORIG_F" "$FUSED" && rm -f "$ORIG_F"
}
trap cleanup EXIT INT TERM
cp "$LIB" "$ORIG_L" || exit 1
cp "$FUSED" "$ORIG_F" || exit 1

patch() { # $1 = file, $2 = text to find (must occur ONCE), $3 = replacement
  python3 - "$1" "$2" "$3" <<'PY' || { echo "    PATCH FAILED (anchor not unique)" >&2; return 1; }
import sys
p, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(a) != 1:
    sys.exit("anchor occurs %d times: %r" % (s.count(a), a))
open(p, "w").write(s.replace(a, b))
PY
}

run() { # $1 = label ; runs the oracle and summarises pass/fail per test
  local out
  out="$(cd "$libws" && cargo test -p burn-kda --test kda_oracle 2>&1)"
  echo "$out" | grep -E '^test [a-z0-9_]+ \.\.\.' | sed 's/^/    /'
  echo "$out" | grep -E '^test result:' | sed 's/^/    /'
  # name the reds, so a mutant that "fixed" one is visible rather than inferred
  echo "$out" | sed -n '/^failures:$/,/^test result/p' \
    | grep -E '^    [a-z0-9_]+$' | sed 's/^/    RED: /'
}

run_rope() { # the RoPE arm's own gate (tests/kda_rope.rs)
  local out
  # `--nocapture`: the rope tests print their measured margin per case, and a
  # gate whose numbers are swallowed by the harness is a gate nobody reads.
  out="$(cd "$libws" && cargo test -p burn-kda --test kda_rope -- --nocapture 2>&1)"
  echo "$out" | grep -E '^test [a-z0-9_]+ \.\.\.' | sed 's/^/    /'
  echo "$out" | grep -E '^test result:' | sed 's/^/    /'
  echo "$out" | sed -n '/^failures:$/,/^test result/p' \
    | grep -E '^    [a-z0-9_]+$' | sed 's/^/    RED: /'
  # the measured margins the rope tests print, indented with the rest
  echo "$out" | grep -E 'worst normalised' | grep -v 'disagrees' | sed 's/^/    /'
}

perturb() { # $1 = label, $2 = file, $3 = anchor, $4 = replacement
  cp "$ORIG_L" "$LIB"; cp "$ORIG_F" "$FUSED"
  echo "=== MUTANT: $1"
  patch "$2" "$3" "$4" || { run; return 1; }
  diff "$2" "$ORIG_L" 2>/dev/null | grep '^[<>]' | sed 's/^/    /' \
    || diff "$2" "$ORIG_F" 2>/dev/null | grep '^[<>]' | sed 's/^/    /'
  run
  echo
}

perturb_rope() { # as `perturb`, but the gate it must break is tests/kda_rope.rs
  cp "$ORIG_L" "$LIB"; cp "$ORIG_F" "$FUSED"
  echo "=== MUTANT: $1"
  patch "$LIB" "$2" "$3" || { run_rope; return 1; }
  diff "$LIB" "$ORIG_L" | grep '^[<>]' | sed 's/^/    /'
  run_rope
  echo
}

echo "== 0. BASELINE (unperturbed) =="
echo "-- burn-kda oracle: 5 green, 2 red ON PURPOSE (the softplus placement and the read scale)"
run
echo "-- the RoPE arm's own gate (tests/kda_rope.rs), expected all green"
run_rope
echo

# ── A. mutants that must break a GREEN ──────────────────────────────────────
# A1: the K3 decay loses its g_min factor. The RUNNING form, and the green that
#     pins it. If this does not go red, the whole tier-(a) row is decoration.
perturb "A1 K3 decay: g_min factor dropped" "$LIB" \
  'DecayFn::Sigmoid => activation::sigmoid(scaled).mul_scalar(self.g_min as f32),' \
  'DecayFn::Sigmoid => activation::sigmoid(scaled).mul_scalar(-1.0),'

# A2: the b_alpha bias is not added to the low-rank logit. The green must see
#     it on the two cases whose bias is non-zero.
perturb "A2 K3 decay: b_alpha not added" "$LIB" \
  'let z = z.add(self.b_alpha.val().clone().reshape([1, 1, hd2]));' \
  'let z = z;'

# A3: the a_log clamp's UPPER bound is pulled inside the fixture's own A range.
#     `A_plus1_wide_z` uses A = +1, so a cap of 0.5 must change that case.
#
#     THE FIRST VERSION OF THIS MUTANT WIDENED THE CLAMP, and it changed
#     NOTHING. That is a coverage limit of the tier-(a) row rather than a broken
#     mutant, and it is recorded here instead of hidden: upstream has no clamp,
#     so a fixture case with A outside [-10, 20] would make our (deliberate,
#     documented) clamped answer differ from FLA's unclamped one and turn the
#     GREEN red. A fixture is not the place to adjudicate a choice we made on
#     purpose. The consequence is a real and stated limit: **no A in this
#     fixture is outside the clamp, so a mutant that WIDENS the clamp is
#     invisible to every arm of this oracle.**
perturb "A3 K3 decay: a_log clamp upper bound 20 -> 0.5" "$LIB" \
  '.clamp(-10.0, 20.0);' \
  '.clamp(-10.0, 0.5);'

# A4: the erase term loses its beta. Eq 1's `(I - beta k k^T)`, and the green
#     that compares it to FLA's executed scan.
perturb "A4 Eq 1: beta dropped from the erase term" "$LIB" \
  'let delta = v.clone().sub(v_hat).mul_scalar(beta);' \
  'let delta = v.clone().sub(v_hat);'

# A5: the decay is applied to the VALUE axis instead of the key axis. This is
#     the KDA-vs-GDN role confusion, and it is the one substitution a shape-only
#     check cannot see. It must break the recurrent green AND the chunked green.
perturb "A5 Eq 1: decay moved to the value axis" "$LIB" \
  'let state = state.mul(decay.clone().reshape([1, h, dk, 1]));' \
  'let state = state.mul(decay.clone().reshape([1, h, 1, dv]));'

# ── A6. WHY THERE IS NO "TURN THE SCALE RED GREEN" MUTANT ─────────────────
# The obvious mutant here -- change the `1.0` at src/lib.rs:646/:653 to
# K**-0.5 -- provably does NOT turn `chunked_wy_applies_no_read_scale` green,
# and the reason is worth stating rather than hiding behind a mutant that looks
# like it worked:
#
#   the scale reds drive `chunk_wy_forward` with an EXPLICIT scale, on purpose,
#   so that they compare the MECHANISM against FLA and not the wiring. The
#   evidence that the mechanism is right and only the argument is wrong is the
#   other test, `chunked_wy_honours_the_read_scale_when_asked`, which is GREEN:
#   asked for K**-0.5, `chunk_wy_forward` reproduces FLA's own `oK` row.
#
# So the fix is at the call site and the fixture already contains the row that
# call site must produce. There is nothing a source mutant can demonstrate that
# the green does not already demonstrate, and a mutant that appeared to "fix" the
# red would be a mutant measuring the test rather than the code.
#
# The one thing that IS worth a mutant, because it is our code and not the
# reference's, is the decay axis -- A5 above -- and it is there.

# ── A-rope. mutants that must break the RoPE gate ─────────────────────────
# The RoPE arm is a CROSS-FAMILY TRANSPLANT: FLA's official KDA layer has no
# rope at all (fla/layers/kda.py @ 9f38d249, zero `rotary`/`rope` hits), so
# there is no upstream layer to be unfaithful to. What there IS is FLA's own
# `rotary_embedding_ref`, and that is what the gate is pinned to. These three
# mutants are the three ways a hand-written RoPE is wrong in practice.

# A6: THE FREQUENCY LOSES ITS FACTOR OF 2. `inv_freq = base^(-2i/D)`
#     (fla/modules/rotary.py:410-414) pairs dim `i` with dim `i + D/2`, and the
#     `2` is what makes the pairing. Dropping it turns a rotation into a
#     different rotation that still preserves the norm -- which is why
#     `rope_commutes_with_l2norm` STAYS GREEN here, and why the commutation
#     test cannot be the gate. Only the FLA comparison sees it.
perturb_rope "A6 rope: frequency loses its factor of 2" \
  'let a = p as f32 * (ROPE_THETA as f32).powf(-(2.0 * i as f32) / head_dim as f32);' \
  'let a = p as f32 * (ROPE_THETA as f32).powf(-(1.0 * i as f32) / head_dim as f32);'

# A7: THE ROTATION IS APPLIED TO q AND NOT k. The pure-rotation test asks
#     about `q` and `k` as separate fixture rows, so both of those stay green;
#     what must go red is the module identity, because k comes back unrotated.
#     This is the mutant that justifies `module_projects_both_q_and_k_rotated`
#     existing at all: a gate built only on `apply_rope` cannot see a wiring
#     that drops one of the two.
perturb_rope "A7 rope: applied to q only" \
  '(apply_rope(q_act, h, hk), apply_rope(k_act, h, hk))' \
  '(apply_rope(q_act, h, hk), k_act)'

# A8: A WRONG BASE. 10000.0 is upstream's default in two independent places
#     (fla/models/hybrid.py:103 and the `RotaryEmbedding` signature), and a
#     drifted constant is exactly the kind of silent change no formula review
#     catches: the rotation is still orthogonal, still norm-preserving, still
#     self-consistent, and wrong.
perturb_rope "A8 rope: theta 10000 -> 1000" \
  'pub const ROPE_THETA: f64 = 10000.0;' \
  'pub const ROPE_THETA: f64 = 1000.0;'

# ── B. mutants that "fix" a RED ─────────────────────────────────────────────
# B1: the softplus form, corrected to FLA's. This one CAN be turned green,
# because the softplus red drives `KdaDecay::forward`, i.e. our code. Expect
# `kimi_linear_softplus_decay_matches_fla_reference` to go GREEN and nothing
# else to move. If it stays red, the diagnosis in S3.1 is wrong about the
# formula even though it is right that the two differ.
#
# The patch spans three lines because `a` AND `z_h` are both MOVED by
# `z_h.mul(a.exp())`, so the corrected arm must clone both to read them. Three
# earlier versions did not compile or did not match, and each failure is worth
# the line it cost:
#   1. `mul_scalar(exp(A)*...)` -- `mul_scalar` on a rank-4 `Tensor` wants an
#      `ElementConversion` bound a generic `B: Backend` does not supply;
#   2. clone `a` but not `z_h` -- E0382, `z_h` moved;
#   3. broadcast `mul(a)` instead of `mul(a.exp())` -- **compiles, and is wrong**:
#      `a` is the clamped `A_h` itself (`lib.rs:259-264`), not `exp(A_h)`, so
#      with `A = -3` the sign flips and every case reads positive where FLA
#      reads negative. A mutant that compiles and is silently wrong is the worst
#      kind, and it is the same shape as the bug this whole lane is about: a
#      decision that looks fine and is not the reference's.
perturb "B1 softplus: exp(A) moved OUTSIDE (expect the softplus red to GO GREEN)" "$LIB" \
  'let scaled = z_h.mul(a.exp());
        let g = match self.decay_fn {
            // Kimi Linear: g = -exp(A_h) * Softplus(z), alpha in (0, 1)
            DecayFn::Softplus => activation::softplus(scaled, 1.0).neg(),' \
  'let scaled = z_h.clone().mul(a.clone().exp());
        let g = match self.decay_fn {
            // Kimi Linear: g = -exp(A_h) * Softplus(z), alpha in (0, 1)
            DecayFn::Softplus => activation::softplus(z_h, 1.0).neg().mul(a.exp()),'

# B2: G_MIN changed from K3'"'"'s -5. Both gate greens must react, and the module
#     doc's published 0.0771 stops being the init'"'"'s alpha. The generator
#     '"'"'s vacuity guard 3 is the other half of this: it re-derives 0.0771 from
#     FLA'"'"'s executed reference and refuses to write the fixture if it moves.
perturb "B2 g_min: -5 (K3) -> -2" "$LIB" \
  'pub const G_MIN: f64 = -5.0;' \
  'pub const G_MIN: f64 = -2.0;'

echo "== RESTORED =="
cp "$ORIG_L" "$LIB"; cp "$ORIG_F" "$FUSED"
if cmp -s "$LIB" "$ORIG_L" && cmp -s "$FUSED" "$ORIG_F"; then
  echo "    both source files are byte-identical to the pre-run snapshot"
else
  echo "    !! a source file DOES NOT MATCH the snapshot" >&2
fi
echo "-- burn-kda oracle back to 5 green / 2 red:"
run
echo "-- the RoPE gate back to all green:"
run_rope
