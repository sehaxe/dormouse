#!/bin/bash
# Engram A/B: control (arm off) vs treatment (the SHIPPED default arm), 3 seeds
# per arm, 2000 steps, batch 20 x seq 512, pure CE, fp32, at the program's
# operating depth (--max-iter 2, docs/AB-PROTOCOL.md 2026-09-27). The decision
# rule is in that file: the treatment wins only if its mean held-out BPB beats
# the control's mean by more than the control's own spread across its 3 seeds.
#
# The treatment is the preset's own budget (engram_rows = 25_000/order, 3
# orders x 32 dim = 2.4M memory params = 24% of the model), NOT the 500K/order
# of the iso-parameter curve: at this backbone 500K is 48M params = 86% of the
# model, i.e. the monopoly shape this work exists to remove. The 500K point at
# this scale is the next rung of the ladder and is worth one seed, not six.
#
# ONE GPU PROCESS AT A TIME (doctrine, 2026-09-27): two concurrent GPU
# processes corrupted the cubecl pool and killed a production run. The gate
# below refuses to start while any other process holds >512 MB of VRAM or
# while another dormouse trainer is alive. Telegram's idle 47 MB context is
# not a trainer and does not block.
set -u

DATA=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_sharded
EVAL=/home/sehaxe/eval_2m_v2
CKDIR=/home/sehaxe/engram_ab
LOGS=/home/sehaxe/logs
BIN=./target/release/train
STEPS=${STEPS:-2000}
DEPTH=${DEPTH:-2}
mkdir -p "$CKDIR" "$LOGS"

gpu_busy() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
    | awk -F', *' '$2+0 > 512 {print $1}' | grep -q . && return 0
  ps -eo pid,args | grep -E 'dormouse-train-image|target/release/train' \
    | grep -v grep | grep -v engram_ab | grep -q . && return 0
  return 1
}

wait_for_gpu() {
  local waited=0
  while gpu_busy; do
    [ "$waited" -eq 0 ] && echo "[$(date +%H:%M:%S)] GPU busy, waiting (one process at a time)"
    sleep 60
    waited=$((waited + 60))
    if [ "$waited" -ge 28800 ]; then
      echo "[$(date +%H:%M:%S)] GPU still busy after 8h, giving up"; exit 2
    fi
  done
  [ "$waited" -gt 0 ] && echo "[$(date +%H:%M:%S)] GPU free after ${waited}s"
}

COMMON="--data $DATA --eval $EVAL --preset small --batch 20 --seq-len 512 \
  --max-iter $DEPTH --steps $STEPS --lr 3e-4 --quant fp32 \
  --jepa-weight 0 --dspark-weight 0 \
  --eval-every 250 --eval-batches 10 --eval-depths \
  --log-every 50 --ckpt-every 0 --timers --ckpt-dir $CKDIR"

run() { # arm seed extra...
  local arm=$1 seed=$2; shift 2
  local name="engram_${arm}_s${seed}"
  local log="$LOGS/${name}.log"
  echo "[$(date +%H:%M:%S)] START $name  $*"
  wait_for_gpu
  # shellcheck disable=SC2086
  $BIN $COMMON --ckpt-name "$name" --log "$log" "$@" >/dev/null 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   $name rc=$rc  $(grep -E "^step +$STEPS " "$log" | tail -1)"
  sleep 20
}

# ---------------------------------------------------------------- 0. cost probe
# What a newcomer's FIRST run costs on the default preset: batch 10 x s512,
# 30 steps, with and without the arm. This is the number the owner asked for
# (step time and VRAM of the default), and it is cheap.
probe() { # name extra...
  local name=$1; shift
  local log="$LOGS/engram_probe_${name}.log"
  echo "[$(date +%H:%M:%S)] PROBE $name  $*"
  wait_for_gpu
  # shellcheck disable=SC2086
  $BIN --data "$DATA" --preset small --batch 10 --seq-len 512 --max-iter "$DEPTH" \
    --steps 30 --log-every 10 --ckpt-every 0 --timers --memlog \
    --jepa-weight 0 --dspark-weight 0 --quant fp32 \
    --ckpt-dir "$CKDIR" --ckpt-name "probe_${name}" --log "$log" "$@" \
    >/dev/null 2>&1 &
  local pid=$!
  local peak=0
  while kill -0 $pid 2>/dev/null; do
    local used
    used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
      | sort -rn | head -1)
    [ -n "$used" ] && [ "$used" -gt "$peak" ] && peak=$used
    sleep 3
  done
  wait $pid
  echo "[$(date +%H:%M:%S)] PROBE $name peak_vram=${peak}MiB step=$(grep -o 'gpu_step=[0-9]*ms' "$log" | tail -2 | tr '\n' ' ')"
  sleep 15
}

probe arm   --engram-ram --engram-slots 25000
probe noarm --no-engram

# ------------------------------------------------------------------- 1. the A/B
# Seed 1 of both arms FIRST: a 1-vs-1 signal is worth having even if the 6-run
# chain does not finish, and it is the cheapest way to catch a mis-specified
# arm (e.g. a floor that is not actually on the graph).
run control 1 --no-engram
run treat   1 --engram-ram --engram-slots 25000 --host-adam-every 1
run control 2 --no-engram
run treat   2 --engram-ram --engram-slots 25000 --host-adam-every 1
run control 3 --no-engram
run treat   3 --engram-ram --engram-slots 25000 --host-adam-every 1

echo
echo "=== held-out BPB at step $STEPS (the decision table) ==="
for arm in control treat; do
  for seed in 1 2 3; do
    log="$LOGS/engram_${arm}_s${seed}.log"
    [ -f "$log" ] || continue
    line=$(grep -E "^step +$STEPS: .*EVAL" "$log" | tail -1)
    [ -n "$line" ] && echo "$arm s$seed: $line"
  done
done
echo done
