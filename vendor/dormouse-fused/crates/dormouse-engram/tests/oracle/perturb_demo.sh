#!/usr/bin/env bash
# Demonstrate that each oracle assertion can FAIL: perturb the fixture, run the
# one test that owns the claim, show it red, restore.
#
#   tests/oracle/perturb_demo.sh
#
# A test that cannot go red is not a test. This is the evidence for that claim
# (ADR-0020), and it is why the tolerances in engram_oracle.rs are named
# constants checked against the fixture's own discriminating power.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../../../.." || exit 1   # tests/oracle -> dormouse-engram -> crates -> dormouse-fused
FIX=crates/dormouse-engram/tests/fixtures/engram_oracle.txt
[ -f "$FIX" ] || { echo "cannot find $FIX from $PWD" >&2; exit 1; }
BAK=$(mktemp)
cp "$FIX" "$BAK"
trap 'cp "$BAK" "$FIX"; rm -f "$BAK"' EXIT

# Perturb one value of a key: first value, or value $3 (1-based).
perturb() {
  /home/sehaxe/oracle-venv/bin/python - "$FIX" "$1" "$2" "${3:-1}" <<'PY'
import sys
path, key, delta, which = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
out, hit = [], False
for line in open(path):
    if not hit and not line.startswith('#') and ':' in line:
        k, body = line.split(':', 1)
        if k.strip() == key:
            vals = body.split()
            i = which - 1
            assert i < len(vals), f'{key} has only {len(vals)} values'
            vals[i] = repr(float(vals[i]) + delta)
            line = f'{k}: ' + ' '.join(vals) + '\n'
            hit = True
    out.append(line)
assert hit, f'key {key} not found'
open(path, 'w').writelines(out)
PY
}

run() {
  local name=$1 test=$2
  local out
  out=$(cargo test -q -p dormouse-engram --test engram_oracle -- --exact "$test" 2>&1)
  if echo "$out" | grep -q 'test result: FAILED'; then
    echo "  RED   $name  ($test)"
    echo "$out" | grep -E 'panicked at|gate row|assertion|left:|right:' | head -3 | sed 's/^/         /'
  else
    echo "  GREEN $name  ($test)  <-- THE ASSERTION IS NOT LOAD-BEARING"
  fi
  cp "$BAK" "$FIX"
}

echo "Perturbation demo. Each line perturbs the fixture, runs the test that owns"
echo "the claim, and expects RED."
echo

# 1. one prime off by 2
perturb ladder.0.primes 2 1
run "prime ladder, one prime off by 2" prime_ladder_matches_the_reference_exactly

# 2. one gate `add` value off by 1e-4 (the column the GREEN test pins)
perturb gate.out.gate_add_eps1e5 1e-4 6
run "gate, reference's add value off by 1e-4" the_gate_matches_the_reference_up_to_the_add_divergence

# 3. one gate `clamp_min` value off by 1e-4
perturb gate.out.gate_eps1e5 1e-4 6
run "gate, reference's clamp_min value off by 1e-4" gate_is_on_the_clamp_min_side

# 4. the reference's own f32 sensitivity understated 100x
perturb gate.row.f32_error 100.0 6
run "gate, a band row's f32 sensitivity understated 100x" the_claim

# 5. module output off by 1e-4
perturb module.out_eps1e5 1e-4
run "module forward, one output off by 1e-4" forward_embeds_matches_the_reference

# 6. one conv output off by 1e-3
perturb conv.y_hc1_reversed 1e-3
run "conv, one output off by 1e-3" depthwise_conv_is_the_time_reverse_of_the_reference_kernel

# 7. the OTHER direction of the conv claim: make the reference's un-reversed
#    column agree with us, so "the tap order is now the reference's" must fire.
#    (Perturbing conv.w_hc1 would not: it is the shared INPUT to both sides, so
#    a comparison of two implementations on one input cannot see it change.)
perturb conv.y_hc1 3.0
run "conv, reference's un-reversed column moved onto ours" depthwise_conv_is_the_time_reverse_of_the_reference_kernel

echo
echo "Restored. Baseline:"
cargo test -q -p dormouse-engram --test engram_oracle 2>&1 | grep 'test result'
echo "(1 red = the clamp_min/add bug at lib.rs:234, reported not fixed)"
