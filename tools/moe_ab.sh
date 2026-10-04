#!/usr/bin/env bash
# The sparse-routing arm (MoE top-k) against its own removal, per
# docs/protocols/AB-PROTOCOL.md: 3 seeds per arm, 2000 steps, batch 8 x seq 512,
# pure CE, fp32, --eval-batches 10 (the window below, printed by this script and
# printed by every eval line).
#
# WHY THIS RUNS AT ALL, when a wave-3 verdict already said "tie": that verdict
# was read on a stack whose fidelity fix landed AFTER it, so the arms were
# compared through an instrument that was wrong at the time (§1.2: a tie
# deletes the mechanism, but only a tie measured correctly deletes anything).
# This is the re-run on the fixed stack. It is the FIRST honest A/B this arm
# has had.
#
# THE CONTROLS ARE CHECKED, NOT ASSUMED. Three of them are specific to this
# arm, and each one is the shape in which the run looks perfectly healthy:
#
#  1. `moe=` must be PRESENT on the routed arm's log lines. The utilization
#     field (`moe=[...] H=... dead=...`, `moe::util_stats`) only prints when
#     the routed branch ran. Its ABSENCE on an arm whose config says
#     `moe_topk = 1` is the `probe::JEPA` defect: an arm that cannot show it
#     ran, whose loss curve is unaffected.
#  2. `H` must not be pinned at either end. `dead = 0` and `H ~ 1.0` means the
#     router did not specialize; `H ~ 0` means it collapsed onto one expert.
#     Either is a real RESULT about the arm, so this prints the trace and warns
#     rather than refusing - but a verdict read without it is not a verdict.
#  3. The eval line must show `engram` rows AND a non-zero KDA backward, the
#     two §3.2 retractions. Unchanged from mor_ab.sh; every control on record
#     predates those fixes and is not a baseline.
#
# COST IS MEASURED BEFORE ANYTHING LONG RUNS. §3.3 is explicit that the price of
# one A/B arm is currently UNKNOWN: AB-PROTOCOL's 53 min came from ~1.6 s/step
# on a run whose attention backward never executed, and the 25.8 s/step figure
# that would replace it has no committed log. So `preflight` prices 100 steps at
# THIS shape and refuses if six arms do not fit the box's wall-clock budget.
# There is no hardcoded per-step number anywhere in this file.
#
# ONE GPU PROCESS AT A TIME (§1.5 - the rule that cost official_v5): every
# launch waits for `nvidia-smi` to report no compute apps, confirmed twice with
# a gap. It will NOT start while another lane's run is on the card.
set -u
cd /home/sehaxe/dormouse

LOGDIR=${LOGDIR:-/home/sehaxe/logs/moe_ab}
CKPT=${CKPT:-/home/sehaxe/moe_ab_ckpt}
DATA=${DATA:-/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_sharded}
EVAL=${EVAL:-/home/sehaxe/eval_2m_v2}
STEPS=${STEPS:-2000}
BATCH=${BATCH:-8}
SEQ=${SEQ:-512}
# The eval window is EVAL_BATCHES x BATCH x SEQ BYTES (AGENTS.md §2.6). At
# batch 8 x seq 512 x 10 batches that is 40 960 B. Quoting "10 batches" without
# the window is not a number; every eval line prints `over <n> B` and
# `assert_window` checks this script's arithmetic against the log.
EVAL_BATCHES=${EVAL_BATCHES:-10}
WINDOW=$((EVAL_BATCHES * BATCH * SEQ))
# Wall-clock budget for ONE 2k-step arm, in seconds. Six arms must fit.
BUDGET=${BUDGET:-21600}
mkdir -p "$LOGDIR" "$CKPT"

echo "=== MoE A/B: $STEPS steps, batch $BATCH x seq $SEQ, eval window $WINDOW B"
echo "=== budget ${BUDGET}s per arm x 6 arms"

gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

# A free card, confirmed TWICE with a gap. The desktop's own GL context sits at a
# few hundred MiB, and a run that starts while the previous process is still
# tearing down dies on a 1 GB allocation.
wait_for_gpu() {
  for _ in $(seq 1 720); do
    if [ "$(gpu_mib)" -lt 800 ]; then
      sleep 30
      [ "$(gpu_mib)" -lt 800 ] && return 0
    fi
    sleep 30
  done
  return 1
}

launch() { # name preset steps seed eval_every [extra...]
  local name=$1 preset=$2 steps=$3 seed=$4 eval_every=$5; shift 5
  wait_for_gpu || { echo "=== GPU never freed for $name"; return 1; }
  echo "=== $name preset=$preset steps=$steps seed=$seed start $(date +%H:%M)"
  # `moe_topk` is passed as `--set` so the CONTROL is the shipped preset with a
  # field flipped, not a separate file: the two arms then differ in exactly one
  # config line, which is what makes the BPB difference attributable. The
  # routed arm's `moe_topk` equals `n_experts` (top-1 on every expert) because
  # a top-k above 1 at these widths is a different arm, not a rung.
  systemd-run --user --scope -q -p MemoryMax=40G \
    ./target/release/train \
      --data "$DATA" --eval "$EVAL" --eval-every "$eval_every" --eval-batches "$EVAL_BATCHES" \
      --preset "$preset" --steps "$steps" --seed "$seed" \
      --batch "$BATCH" --seq-len "$SEQ" \
      --jepa-weight 0 --dspark-weight 0 --quant fp32 --timers \
      --ckpt-name "moe_ab_$name" --ckpt-dir "$CKPT" \
      --log "$LOGDIR/$name.log" "$@" >/dev/null 2>&1
}

# `moe_topk` on the routed arm, from the preset's own `n_experts`.
topk_of() { # preset
  grep -E '^n_experts' "configs/$1.toml" | head -1 | tr -dc '0-9'
}

# Every control that decides whether this run is a measurement.
clean_control() { # log label expect_routed
  local log=$1 label=$2 expect_routed=$3 ok=0
  [ -f "$log" ] || { echo "=== $label: no log at $log"; return 1; }

  # --- 1. the routing arm actually ran (or deliberately did not) -------------
  # `grep -c` PRINTS 0 and EXITS 1 on no match, so `|| echo 0` appends a second
  # 0 and the test below gets "0\n0". The count is always printed; only the
  # exit code is conditional.
  local routed; routed=$(grep -c 'moe=\[' "$log" 2>/dev/null); routed=${routed:-0}
  local step; step=$(grep -oE 'moe=\[[^]]*\] H=[0-9.]+ dead=[0-9]+' "$log" | tail -1)
  echo "=== $label utilization (last): ${step:-ABSENT}"
  if [ "$expect_routed" = yes ]; then
    if [ "$routed" -eq 0 ]; then
      echo "=== DIRTY [$label]: moe_topk > 0 but NO utilization field on any step line - the \
routed branch never ran, so this arm is the dense control wearing the arm's name"
      return 1
    fi
    ok=1
  else
    if [ "$routed" -ne 0 ]; then
      echo "=== DIRTY [$label]: the CONTROL printed a utilization field - the arm is not off"
      return 1
    fi
    ok=1
  fi

  # --- 2. the router neither collapsed nor refused to move (WARN, not refuse) -
  # These are RESULTS, so the trace is printed and the run continues: a collapse
  # is the answer to "does the router learn to specialize", not a broken
  # instrument. What is not allowed is to read the BPB without it.
  if [ -n "$step" ]; then
    local h dead
    h=$(printf '%s' "$step" | sed -E 's/.*H=([0-9.]+).*/\1/')
    dead=$(printf '%s' "$step" | sed -E 's/.*dead=([0-9]+).*/\1/')
    if [ "$dead" -gt 0 ]; then
      echo "=== NOTE [$label]: $dead of the experts got ZERO tokens (H=$h). A dead expert's \
parameters still train through the dense blend, so this is a capacity result, not a crash."
    fi
    awk -v h="$h" 'BEGIN{exit !(h>0.98)}' \
      && echo "=== NOTE [$label]: H=$h is ~uniform. The router spread its load perfectly; if it \
is still 1.0 at step $STEPS the arm specialized in NO dimension and that is the finding."
    awk -v h="$h" 'BEGIN{exit !(h<0.05)}' \
      && echo "=== NOTE [$label]: H=$h is ~collapsed. One expert took everything."
  fi

  # --- 3. the eval line is a real measurement (the two §3.2 retractions) ----
  local ev; ev=$(grep EVAL "$log" | tail -1)
  [ -n "$ev" ] || { echo "=== DIRTY [$label]: no EVAL line at all"; return 1; }
  echo "=== $ev"
  case "$ev" in
    *engram=0/*) echo "=== DIRTY [$label]: the eval ran a memory-disabled forward (engram=0)"; return 1;;
    *engram=*)  ;;
    *)          echo "=== DIRTY [$label]: the eval line carries no engram= field"; return 1;;
  esac
  # The attention backward, read from `bwd=` and NOT from `fused kda=F/B`.
  # `mor_ab.sh` checks `fused kda=.../0`, which was right when the fused backward
  # was the only backward there was (§3.2's retraction). It is now the WRONG
  # field: `kda_seam_counts` orders the tuple (asked, fused_fwd, fused_bwd,
  # declined, ops_path, custom_node_bwd), so `fused kda=4/0` reads
  # "4 fused forwards, 0 fused backwards" - and per §3.3 the fused forward
  # launches and is DISCARDED while the real backward is `bwd=`. Checking the
  # fused pair would refuse EVERY arm, including a perfectly clean one. Caught
  # by the synthetic-log gate in this lane's commit message.
  # Anchored on the LEADING SPACE, not on `bwd=` alone: the same line carries
  # `node_bwd=0` (the custom-node backward), and a bare `bwd=[0-9]*` matches
  # inside it, so `tail -1` returns the wrong counter and reads a clean arm as
  # one with no attention backward. Second bug the synthetic gate caught here.
  local kb; kb=$(grep -oE ' bwd=[0-9]+' "$log" | tail -1 | tr -dc '0-9')
  local ff; ff=$(grep -o 'fused kda=[0-9]*/[0-9]*' "$log" | tail -1)
  echo "=== $label attention counter: ${kb:-absent} backwards (${ff:-no fused counter})"
  case "${kb:-}" in
    "")  echo "=== DIRTY [$label]: no bwd= counter in the log"; return 1;;
    0)   echo "=== DIRTY [$label]: the attention arm ran 0 backwards"; return 1;;
  esac

  # --- 4. the window is the one this script budgeted for (§2.6) --------------
  assert_window "$log" "$label" || return 1

  # --- 5. no NaN survived the firewall --------------------------------------
  local nan; nan=$(grep -c 'NaN' "$log" 2>/dev/null); nan=${nan:-0}
  echo "=== $label NaN lines: $nan"
  [ "$nan" -eq 0 ] || { echo "=== DIRTY [$label]: $nan NaN lines"; return 1; }
  return $(( 1 - ok ))
}

# The byte count on the eval line IS the authority on the window (§2.6); this
# compares this script's arithmetic against it, because a run at the wrong
# window is not comparable to anything and the flag that changes it (`--batch`)
# is a throughput knob no reader would expect to move the scored bytes.
assert_window() { # log label
  local log=$1 label=$2
  local scored; scored=$(grep -o 'over [0-9]* B' "$log" | tail -1 | tr -dc '0-9')
  [ -n "$scored" ] || { echo "=== DIRTY [$label]: no `over <n> B` on the eval line"; return 1; }
  if [ "$scored" -ne "$WINDOW" ]; then
    echo "=== DIRTY [$label]: the eval scored $scored B, this script budgeted $WINDOW B. A BPB \
is only comparable within one window."
    return 1
  fi
  echo "=== $label window OK: $scored B"
  return 0
}

# 100 steps at the A/B's own shape: the protocol's rung-1 smoke (NaN, speed,
# early slope) and the only thing cheap enough to run before committing hours.
preflight() { # label preset expect_routed
  local label=$1 preset=$2 expect=$3
  local t0 t1
  t0=$(date +%s)
  [ "$expect" = yes ] && launch "pre_$label" "$preset" 100 1 25 "--set" "moe_topk=$(topk_of "$preset")" \
    || launch "pre_$label" "$preset" 100 1 25 "--set" "moe_topk=0"
  t1=$(date +%s)
  # Setup (CUDA context, autotune, data indexing) is inside this, so it is an
  # UPPER bound on the per-step cost - the right direction for a gate.
  local per_step=$(( (t1 - t0) / 100 ))
  local projected=$(( per_step * STEPS ))
  echo "=== preflight $label: ~${per_step} s/step wall (incl. setup) -> ~$(( projected / 3600 )) h \
for $STEPS steps (window $WINDOW B)"
  grep -E "^(step|.*EVAL)" "$LOGDIR/pre_$label.log" | tail -3
  if [ "$per_step" -le 0 ] || [ "$projected" -gt "$BUDGET" ]; then
    echo "=== REFUSING the A/B: ~$(( projected / 3600 )) h per arm x 6 arms is over the \
${BUDGET}s budget. Re-cost with a larger BUDGET or fewer seeds, and record which."
    return 1
  fi
  clean_control "$LOGDIR/pre_$label.log" "pre_$label" "$expect" || {
    echo "=== REFUSING the A/B: $label itself is not a clean arm (see above)"; return 1; }
  return 0
}

run() { # label preset seed expect_routed
  local label=$1 preset=$2 seed=$3 expect=$4
  if [ -f "$LOGDIR/$label.done" ]; then echo "skip $label (done)"; return 0; fi
  [ "$expect" = yes ] && launch "$label" "$preset" "$STEPS" "$seed" 250 "--set" "moe_topk=$(topk_of "$preset")" \
    || launch "$label" "$preset" "$STEPS" "$seed" 250 "--set" "moe_topk=0"
  local rc=$?
  if [ $rc -ne 0 ] || ! grep -q "step $STEPS" "$LOGDIR/$label.log" 2>/dev/null; then
    echo "=== $label FAILED rc=$rc (see $LOGDIR/$label.log)"; return 1
  fi
  clean_control "$LOGDIR/$label.log" "$label" "$expect" || {
    echo "=== $label did NOT pass the clean-control checks - not a usable arm"; return 1; }
  touch "$LOGDIR/$label.done"
  echo "=== $label done $(date +%H:%M): $(grep EVAL "$LOGDIR/$label.log" | tail -1)"
}

# §1.2: the verdict is a comparison against the SPREAD of the control's own
# seeds, not against one control number. `jq` is not assumed; this is awk.
verdict() {
  local w moe
  w=$(grep EVAL "$LOGDIR"/ctrl_s*.log | grep -o 'bpb=[0-9.]*' | cut -d= -f2)
  moe=$(grep EVAL "$LOGDIR"/routed_s*.log | grep -o 'bpb=[0-9.]*' | cut -d= -f2)
  echo "=== control BPBs: $(echo $w)"
  echo "=== routed  BPBs: $(echo $moe)"
  [ -z "$w" ] && { echo "=== no control BPBs on record"; return 1; }
  [ -z "$moe" ] && { echo "=== no routed BPBs on record"; return 1; }
  # `verdict_awk` rather than inlined here, so the arithmetic `selftest`
  # exercises is the arithmetic that decides. Two copies of a §1.2 verdict is
  # one too many.
  echo "$w $moe" | verdict_awk
}

# `selftest` is defined BEFORE the dispatch so it can reuse `clean_control`
# directly, with fixtures written fresh on every run (a stale fixture is how the
# first version of this gate reported a false failure).
selftest() {
  local L; L=$(mktemp -d)
  trap 'rm -rf "$L"' RETURN
  cat > "$L/good_routed.log" <<'LOG'
step     0 ce=5.500 bpb=8.000 best=8.000 lr=1.00e-04 moe=[0.250,0.250,0.250,0.250] H=1.000 dead=0 retr_arm=batched:0/factor:14
step  1000 ce=4.900 bpb=7.100 best=7.100 lr=1.00e-04 moe=[0.480,0.520,0.000,0.000] H=0.510 dead=2 retr_arm=batched:0/factor:14
step  2000 EVAL ce=4.231 bpb=6.104 over 40960 B (fixed window) fused kda=4/0 asked=88 bwd=4 declined=260 ops=88 node_bwd=0 norm=0/109 muon_skipped=0/0 engram=88/88 bf=0 fb=0/0
LOG
  # The dense control: same run with the utilization field removed, which is
  # exactly what it looks like when `moe_topk = 0` (the field only prints when
  # the routed branch ran).
  sed 's/ moe=\[[^]]*\] H=[0-9.]* dead=[0-9]*//' "$L/good_routed.log" > "$L/good_dense.log"
  sed 's/engram=88\/88/engram=0\/88/'       "$L/good_routed.log" > "$L/dirty_engram.log"
  sed 's/bwd=4 /bwd=0 /'                    "$L/good_routed.log" > "$L/dirty_kda.log"
  sed 's/over 40960 B/over 20480 B/'        "$L/good_routed.log" > "$L/dirty_window.log"
  sed 's/ moe=\[[^]]*\] H=[0-9.]* dead=[0-9]*//' "$L/good_routed.log" > "$L/no_moe.log"
  cp "$L/good_routed.log" "$L/nan.log"; echo 'step 1 ce=NaN bpb=NaN' >> "$L/nan.log"

  local pass=0 fail=0
  local name want log label routed got
  check() { # name expect_pass log label expect_routed
    name=$1; want=$2; log=$3; label=$4; routed=$5
    if clean_control "$log" "$label" "$routed" >/dev/null 2>&1; then got=pass; else got=fail; fi
    if [ "$got" = "$want" ]; then
      pass=$((pass+1)); echo "  ok   $name"
    else
      fail=$((fail+1)); echo "  FAIL $name: expected $want, got $got"
      clean_control "$log" "$label" "$routed" 2>&1 | sed 's/^/       /' | tail -3
    fi
  }
  echo "=== the checks must ACCEPT a clean run of either arm"
  check "routed arm, clean" pass "$L/good_routed.log" t yes
  check "dense  arm, clean" pass "$L/good_dense.log"  t no
  echo "=== and REJECT each defect they claim to catch"
  check "the eval ran a memory-disabled forward" fail "$L/dirty_engram.log" t yes
  check "the attention arm ran 0 backwards"       fail "$L/dirty_kda.log"     t yes
  check "the eval window is not the budgeted one" fail "$L/dirty_window.log"  t yes
  check "routed arm, no utilization field"        fail "$L/no_moe.log"        t yes
  check "dense arm printed a utilization field"   fail "$L/good_routed.log"   t no
  check "a NaN line"                              fail "$L/nan.log"           t yes
  echo "=== $pass passed, $fail failed"

  # The VERDICT half, because a protocol that cannot say "tie" will be read as
  # a win. §1.2: a tie deletes the mechanism, so the interesting case is the one
  # where the delta is INSIDE the control's own seed spread, and that case has to
  # be demonstrably reached rather than assumed. Two synthetic triples:
  #   5.10 5.30 5.05 | 4.70 4.72 4.68  -> delta -0.45 beats spread 0.09 -> KEEP
  #   5.10 5.30 5.05 | 5.12 4.95 5.18  -> delta +0.00 inside spread 0.16 -> TIE
  local got_want got_got
  got_got=$(printf '5.10 5.30 5.05 4.70 4.72 4.68' | verdict_awk)
  case "$got_got" in *KEEP*) got_want=KEEP ;; *) got_want=other ;; esac
  local t_want t_got
  t_got=$(printf '5.10 5.30 5.05 5.12 4.95 5.18' | verdict_awk)
  case "$t_got" in *TIE*) t_want=TIE ;; *) t_want=other ;; esac
  if [ "$got_want" = KEEP ] && [ "$t_want" = TIE ]; then
    pass=$((pass+1)); echo "  ok   verdict: a delta outside the spread is a KEEP"
    pass=$((pass+1)); echo "  ok   verdict: a delta inside it is a TIE (which deletes)"
  else
    fail=$((fail+1)); echo "  FAIL verdict: expected KEEP and TIE, got $got_want / $t_want"
  fi
  echo "=== $pass passed, $fail failed"
  [ "$fail" -eq 0 ]
}

# The verdict arithmetic on stdin: six numbers, three control then three routed.
# Split out from `verdict` so `selftest` can drive it with a synthetic triple -
# a verdict function that has never been run is the one that decides whether a
# mechanism survives §1.2.
verdict_awk() {
  awk '
    { for (i = 1; i <= NF; i++) v[++n] = $i }
    END {
      ns = int(n / 2)
      for (i = 1; i <= ns; i++) { c[i] = v[i]; m[i] = v[ns + i] }
      for (i = 1; i <= ns; i++) { sc += c[i]; sm += m[i] }
      sc /= ns; sm /= ns
      for (i = 1; i <= ns; i++) { d = c[i] - sc; vc += d * d; d = m[i] - sm; vm += d * d }
      vc = sqrt(vc / (ns - 1)); vm = sqrt(vm / (ns - 1))
      dm = sm - sc
      printf "control %.4f sd %.4f | routed %.4f sd %.4f | delta %+.4f\n", sc, vc, sm, vm, dm
      pooled = sqrt((vc * vc + vm * vm) / 2)
      if (pooled > 0 && (dm < -pooled || dm > pooled)) {
        printf "VERDICT KEEP: routed %s by %.4f, beating the control spread %.4f\n", \
               (dm < 0 ? "wins" : "loses"), (dm < 0 ? -dm : dm), pooled
      } else {
        printf "VERDICT TIE: |delta| %.4f is inside the spread %.4f, and a tie DELETES the\n\
               mechanism (docs/protocols/AB-PROTOCOL.md)\n", (dm < 0 ? -dm : dm), pooled
      }
    }'
}

case "${1:-all}" in
  preflight)
    preflight ctrl small no && preflight routed moe-cap-4 yes ;;
  all)
    preflight ctrl   small    no  || exit 1
    preflight routed moe-cap-4 yes || exit 1
    for i in 1 2 3; do
      run "ctrl_s$i"   small    "$i" no  || break
      run "routed_s$i" moe-cap-4 "$i" yes || break
    done
    verdict $(ls "$LOGDIR"/ctrl_s*.log) $(ls "$LOGDIR"/routed_s*.log)
    echo "ALL DONE $(date +%H:%M)" ;;
  verdict)
    verdict $(ls "$LOGDIR"/ctrl_s*.log) $(ls "$LOGDIR"/routed_s*.log) ;;
  selftest)
    # THE CHECKS, EXERCISED. This lane had no GPU, so `clean_control` and
    # `assert_window` were written and never run - and a gate that cannot fail is
    # the defect class this repo keeps finding (AGENTS.md §3.2: the eval read a
    # memory-disabled forward, the attention arm ran 3126 forwards and 0
    # backwards, and every loss curve looked fine). A protocol whose checks have
    # never executed is exactly that shape.
    #
    # Eight cases against SYNTHETIC logs: the two clean arms must be ACCEPTED,
    # and each defect the checks claim to catch must be REJECTED. Three real
    # bugs were found this way the day it was written, all of which would have
    # refused or mis-read a good run:
    #   * the attention counter read `fused kda=F/B`, whose second field is the
    #     FUSED backward - which §3.3 says never ran - so it read every clean arm
    #     as having no backward at all. The real counter is `bwd=`.
    #   * `grep -o 'bwd=[0-9]*'` also matches inside `node_bwd=0`, so `tail -1`
    #     returned the custom-node counter. Anchored on the leading space.
    #   * `grep -c ... || echo 0` printed "0\n0" (grep -c prints 0 AND exits 1)
    #     and the integer test choked on it.
    selftest ;;
  *) echo "usage: $0 [preflight|all|verdict|selftest]"; exit 2 ;;
esac