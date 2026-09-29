#!/usr/bin/env bash
# MoR vs fixed-depth control, per docs/AB-PROTOCOL.md: 3 seeds per arm, 2000
# steps, batch 20 x seq 512, pure CE, fp32, --eval-every 250 --eval-depths.
#
# NOT RUN AS-IS on 2026-09-29, for three measured reasons, all checked by
# this script rather than assumed:
#
#  1. COST IS UNKNOWN. docs/AB-PROTOCOL.md prices a 2k-step arm at 53 min from
#     a ~1.6 s/step run whose attention backward never executed (AGENTS.md
#     §3.2/§3.3). The working tensor-op KDA backward measured 25.8 s/step at
#     batch 8. `preflight` below measures the real number at THIS shape before
#     anything long starts, and refuses if the projection is over budget.
#  2. THE SHARED top-k GATE IS RED on sm_120: `cargo test -p backend-parity
#     --features cuda --test topk_gather_parity` fails "masked: 2 row(s) picked
#     the wrong k" (2026-09-29). MoR's wiring does NOT ride that primitive -
#     mor.rs uses burn-mor's argsort-based `topk_indices` - but no A/B verdict
#     should be read until the gate is green.
#  3. THE CONTROL MUST BE CLEAN. Every control on record predates 7adda92
#     (eval measured a memory-disabled forward) and 8fa5d4c (attention arm had
#     no gradient), so they are not baselines. `clean_control` refuses to call
#     a run valid unless the eval line shows the memory arm ran AND the KDA
#     backward ran.
#
# ONE GPU PROCESS AT A TIME (the rule that cost official_v5 on 2026-09-27):
# every launch waits until `nvidia-smi` reports no compute apps.
set -u
cd /home/sehaxe/dormouse

LOGDIR=/home/sehaxe/logs/mor_ab
CKPT=/home/sehaxe/mor_ab_ckpt
DATA=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_sharded
EVAL=/home/sehaxe/eval_2m_v2
STEPS=2000
BATCH=20
SEQ=512
EVAL_BATCHES=20
# The window is EVAL_BATCHES x BATCH x SEQ bytes (AGENTS.md §2.6) - 204 800 B
# here, NOT the "100 KB" the protocol used to quote.
WINDOW=$((EVAL_BATCHES * BATCH * SEQ))
# Wall-clock budget for ONE 2k-step arm, in seconds. 6 arms must fit the box.
BUDGET=${BUDGET:-14400}
mkdir -p "$LOGDIR" "$CKPT"

gpu_busy() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -ne 0 ]; }

wait_for_gpu() {
  for _ in $(seq 1 720); do
    gpu_busy || return 0
    sleep 30
  done
  return 1
}

launch() { # name preset steps seed extra...
  local name=$1 preset=$2 steps=$3 seed=$4; shift 4
  wait_for_gpu || { echo "GPU never freed for $name"; return 1; }
  echo "=== $name preset=$preset steps=$steps seed=$seed start $(date +%H:%M)"
  systemd-run --user --scope -q -p MemoryMax=40G \
    ./target/release/train \
      --data "$DATA" --eval "$EVAL" --eval-every 250 --eval-batches "$EVAL_BATCHES" \
      --eval-depths --preset "$preset" --steps "$steps" --seed "$seed" \
      --batch "$BATCH" --seq-len "$SEQ" \
      --jepa-weight 0 --dspark-weight 0 --quant fp32 --timers \
      --ckpt-name "mor_ab_$name" --ckpt-dir "$CKPT" \
      --log "$LOGDIR/$name.log" "$@" >/dev/null 2>&1
}

# A 20-step run at the A/B's own shape, purely to price it. The protocol's
# rung 1 (smoke: NaN, speed, early slope) is the same run.
preflight() {
  local name="$1" preset="$2"
  # 60 steps, not 20: --timers prints every 50 steps and the wall clock below
  # needs a compile-out, a warmup and a steady state to mean anything.
  local t0 t1
  t0=$(date +%s)
  launch "pre_$name" "$preset" 60 1 || return 1
  t1=$(date +%s)
  # Setup (CUDA context, autotune, data indexing) is in there, so this is an
  # UPPER bound on the per-step cost, which is the right direction for a gate.
  local per_step=$(( (t1 - t0) / 60 ))
  local projected=$(( per_step * STEPS ))
  echo "=== preflight $name: ~${per_step} s/step wall (incl. setup) -> ~$(( projected / 3600 )) h for $STEPS steps (window $WINDOW B)"
  echo "=== preflight NaN lines: $(grep -c NaN "$LOGDIR/pre_$name.log")"
  grep -E "^(step|.*EVAL)" "$LOGDIR/pre_$name.log" | tail -3
  if [ "$per_step" -le 0 ] || [ "$projected" -gt "$BUDGET" ]; then
    echo "=== REFUSING the A/B: ~$(( projected / 3600 )) h per arm x 6 arms is over the ${BUDGET}s budget"
    return 1
  fi
  clean_control "$LOGDIR/pre_$name.log" || {
    echo "=== REFUSING the A/B: the control itself is not clean (see above)"; return 1; }
  return 0
}

# The two clean-control preconditions from AGENTS.md §3.2/§3.4, checked in the
# log rather than trusted: the memory arm must have run in the EVAL, and the
# attention arm must have a backward.
clean_control() {
  local log=$1 ok=0
  local ev; ev=$(grep EVAL "$log" | tail -1)
  [ -n "$ev" ] || { echo "=== no EVAL line"; return 1; }
  echo "=== $ev"
  case "$ev" in
    *engram=0/*) echo "=== DIRTY: eval ran a memory-disabled forward (engram=0)"; return 1;;
    *engram=*)  ok=1;;
    *)          echo "=== UNKNOWN: eval line carries no engram= field"; return 1;;
  esac
  local kb; kb=$(grep -o 'fused kda=[0-9]*/[0-9]*' "$log" | tail -1)
  echo "=== kda counter: ${kb:-absent}"
  case "${kb:-}" in
    *"/0") echo "=== DIRTY: attention arm ran 0 backwards"; return 1;;
    "")    echo "=== UNKNOWN: no fused kda= counter in the log"; return 1;;
  esac
  return $(( 1 - ok ))
}

run() { # name preset seed
  local name=$1 preset=$2 seed=$3
  if [ -f "$LOGDIR/$name.done" ]; then echo "skip $name (done)"; return 0; fi
  launch "$name" "$preset" "$STEPS" "$seed" || return 1
  local rc=$?
  if [ $rc -ne 0 ] || ! grep -q "step $STEPS" "$LOGDIR/$name.log" 2>/dev/null; then
    echo "=== $name FAILED rc=$rc (see $LOGDIR/$name.log)"; return 1
  fi
  if ! clean_control "$LOGDIR/$name.log"; then
    echo "=== $name did NOT pass the clean-control checks - not a usable arm"
    return 1
  fi
  touch "$LOGDIR/$name.done"
  echo "=== $name done $(date +%H:%M): $(grep EVAL "$LOGDIR/$name.log" | tail -1)"
  echo "=== depth curve: $(grep -m1 -o 'depths=.*' "$LOGDIR/$name.log" || echo none)"
}

case "${1:-all}" in
  preflight)
    preflight ctrl small && preflight mor mor ;;
  all)
    preflight ctrl small || exit 1
    preflight mor   mor   || exit 1
    for i in 1 2 3; do
      run "ctrl_s$i" small "$i" || break
      run "mor_s$i"  mor   "$i" || break
    done
    echo "ALL DONE $(date +%H:%M)" ;;
  *) echo "usage: $0 [preflight|all]"; exit 2 ;;
esac
