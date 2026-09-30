#!/usr/bin/env bash
# first_run.sh — the first training of the full architecture, as one command.
#
# Preconditions from AGENTS.md §2.4/§2.5/§2.6, all checked LOUD (§1.1: a
# precondition that fails must name its escape, not fall through):
#   - no other train process, GPU quiet (§1.5: ONE heavy thing at a time)
#   - RAM avail >= 25 GB (production wants 40; smoke is fine at 25)
#   - corpus + eval tail readable (the drive must be mounted, §2.6)
#   - release binary exists
# Then: systemd-run MemoryMax scope (§2.4 rule 2) and the §3.8 launch line,
# with `--steps` as the only knob:  `./tools/first_run.sh 500`   (smoke)
#                                   `./tools/first_run.sh 2000`  (control)
#
# The smoke run ALSO prices the step honestly: quote only steps >= 50 —
# step 0 is an 18-23x autotune artifact (AGENTS.md §3.1, retracted row).
# And it is where DM_GDN2_BWD_TRACE=1 answers "does the attention arm train
# again?" (§3.3): the ENTERED line must appear. Export it before calling.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:?usage: first_run.sh <steps> (500=smoke, 2000=control)}"

if pgrep -x train >/dev/null 2>&1 || pgrep -f "target/release/train" >/dev/null 2>&1; then
  echo "LOUD: a train process is already running — one heavy thing at a time (§1.5). Escape: wait or pgrep -ax train."; exit 1
fi
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
if [ "${AVAIL:-0}" -lt 25 ]; then
  echo "LOUD: RAM avail ${AVAIL}G < 25 GB (§2.4). Escape: close the desktop or wait."; exit 1
fi
CORPUS_DIR=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real
EVAL=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_eval/eval_tail.bin
if [ ! -r "$CORPUS_DIR/corpus.bin" ]; then
  echo "LOUD: corpus not readable: $CORPUS_DIR/corpus.bin — the drive must be mounted (§2.6)."; exit 1
fi
if [ ! -r "$EVAL" ]; then
  echo "LOUD: eval tail not readable: $EVAL. Pre-carve numbers are not comparable — do not substitute (§2.6)."; exit 1
fi
if [ ! -x target/release/train ]; then
  echo "LOUD: no target/release/train — build first: tools/build_lock.sh run build -- cargo build --release -p dormouse-cli --features dormouse-train/cuda"; exit 1
fi

LOG=/home/sehaxe/logs/first_run_${STEPS}_$(date +%m%d_%H%M).log
CKPT=first_run_${STEPS}
mkdir -p /home/sehaxe/logs
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo '?')
echo "preconditions OK: gpu=${UTIL}% ram=${AVAIL}G steps=${STEPS} ckpt=${CKPT}"
echo "log: $LOG   (eval line prints its own byte window — quote it, §2.6)"

exec systemd-run --user --scope -p MemoryMax=40G \
  ./target/release/train \
    --data "$CORPUS_DIR" \
    --eval "$EVAL" \
    --eval-every 500 \
    --preset small \
    --batch 8 \
    --seq-len 512 \
    --no-engram \
    --steps "$STEPS" \
    --ckpt-name "$CKPT" \
    --guard \
    --detach \
    --timers \
    --log "$LOG"
