#!/usr/bin/env bash
# MoR vs fixed-depth control, per docs/AB-PROTOCOL.md: 3 seeds per arm, 2000
# steps, batch 20 x seq 512, pure CE, fp32, fixed 100 KB held-out window with
# --eval-every 250 --eval-depths.
#
# ONE GPU PROCESS AT A TIME (the rule that cost official_v5 on 2026-09-27):
# every launch waits until `nvidia-smi` reports no compute apps, so this can
# be left running next to other agents' work without colliding.
set -u
cd /home/sehaxe/dormouse

LOGDIR=/home/sehaxe/logs/mor_ab
CKPT=/home/sehaxe/mor_ab_ckpt
DATA=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_sharded
EVAL=/home/sehaxe/eval_2m_v2
STEPS=2000
mkdir -p "$LOGDIR" "$CKPT"

wait_for_gpu() {
  # Telegram's desktop GL context is not a compute app on this driver; any
  # other name holding VRAM is a real process and we wait.
  for _ in $(seq 1 720); do
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "$busy" -eq 0 ]; then return 0; fi
    sleep 30
  done
  return 1
}

run() { # name preset
  local name=$1 preset=$2
  if [ -f "$LOGDIR/$name.done" ]; then echo "skip $name (done)"; return 0; fi
  wait_for_gpu || { echo "GPU never freed for $name"; return 1; }
  # "Seed" here is burn's per-process weight init (no --seed flag exists, and
  # the data stream is deterministically seeded), so the three runs per arm
  # differ exactly the way the A/B protocol means by a seed.
  echo "=== $name (preset=$preset) start $(date +%H:%M)"
  systemd-run --user --scope -q -p MemoryMax=40G \
    ./target/release/train \
      --data "$DATA" --eval "$EVAL" --eval-every 250 --eval-batches 20 --eval-depths \
      --preset "$preset" --steps "$STEPS" --batch 20 --seq-len 512 \
      --jepa-weight 0 --dspark-weight 0 --quant fp32 \
      --ckpt-name "mor_ab_$name" --ckpt-dir "$CKPT" \
      --log "$LOGDIR/$name.log" >/dev/null 2>&1
  local rc=$?
  # A run must finish its last eval; the final held-out BPB is what the
  # decision rule reads.
  if [ $rc -ne 0 ] || ! grep -q "step $STEPS" "$LOGDIR/$name.log" 2>/dev/null; then
    echo "=== $name FAILED rc=$rc (see $LOGDIR/$name.log)"; return 1
  fi
  touch "$LOGDIR/$name.done"
  echo "=== $name done $(date +%H:%M): $(grep "EVAL" "$LOGDIR/$name.log" | tail -1)"
}

for i in 1 2 3; do
  run "ctrl_s$i" small || break
  run "mor_s$i"  mor   || break
done
echo "ALL DONE $(date +%H:%M)"
