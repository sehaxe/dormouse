#!/usr/bin/env bash
# The ONE gateway for GPU work when several agents share the card.
#
# AGENTS 1.5 says one heavy thing at a time; with a swarm, "one" has to be
# enforced, not hoped for. This wrapper does three things:
#
#   1. `flock` — concurrent callers queue instead of colliding (a collision once
#      wrote garbage into a checkpoint);
#   2. waits for the card to go IDLE (`> CARD_IDLE_MIB` refuses — a stray pytest
#      context from another lane would corrupt a throughput number);
#   3. `timeout` — a hung benchmark cannot block the queue forever.
#
# It does NOT decide what to run. Every caller passes the real command:
#
#   bash scripts/gpu_lock.sh .venv/bin/python -m dormouse.cli train ... --set steps=40
#   GPU_TIMEOUT=1200 GPU_IDLE_TIMEOUT=600 bash scripts/gpu_lock.sh \
#       .venv/bin/python scripts/bench_engine.py --quick
#
# A timeout exit code is 124; a busy-card refusal is 75. Both mean "nothing was
# measured" — never quote a number from them.
set -euo pipefail
cd "$(dirname "$0")/.."

IDLE_MIB=${GPU_IDLE_MIB:-2000}
TIMEOUT=${GPU_TIMEOUT:-900}
IDLE_WAIT=${GPU_IDLE_TIMEOUT:-300}
LOCK=${GPU_LOCK_FILE:-/tmp/dormouse-gpu.lock}

if [ "$#" -eq 0 ]; then
  echo "loud: gpu_lock.sh needs a command to run" >&2
  exit 2
fi

exec 9>"$LOCK"
if ! flock -w 1800 9; then
  echo "loud: another GPU job held the lock for 30 min; nothing was measured" >&2
  exit 75
fi

waited=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -le "$IDLE_MIB" ] && break
  if [ "$waited" -ge "$IDLE_WAIT" ]; then
    echo "loud: card held ${used} MiB > ${IDLE_MIB} for ${IDLE_WAIT}s (another lane is running); nothing was measured" >&2
    exit 75
  fi
  sleep 10
  waited=$((waited + 10))
done

echo "gpu_lock: card idle at ${used} MiB; running: $*" >&2
timeout "$TIMEOUT" "$@"
