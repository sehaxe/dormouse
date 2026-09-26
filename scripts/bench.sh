#!/usr/bin/env bash
# Canonical dormouse benchmark: appends one metrics line to benches/history.tsv.
#
# Usage:
#   scripts/bench.sh canary   # small, aux off, 2M slots, 30 steps (~2 GB RSS, ~2 min)
#   scripts/bench.sh flagship # small, aux on, 48M slots, batch 10 s512, 50 steps (~20 GB RSS)
#
# Preflight: one train process max, RAM headroom per mode. The metrics line is
# the regression gate for docs/PLAN.md's performance budget.
set -eu
cd "$(dirname "$0")/.."
MODE="${1:-canary}"
DIR=benches
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
COMMIT=$(git rev-parse --short HEAD)
mkdir -p "$DIR" /tmp/opencode/bench

AVAIL=$(free -g | awk '/^Mem/{print $7}')
[ -z "$(pgrep -ax train)" ] || { echo "bench: a train process is already running"; exit 1; }

DATA=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_filtered_v2
[ -d "$DATA" ] || { echo "bench: corpus v2 drive not mounted"; exit 1; }

case "$MODE" in
  canary)
    [ "$AVAIL" -ge 6 ] || { echo "bench: need >=6G available, have ${AVAIL}G"; exit 1; }
    ARGS=(--preset small --batch 10 --seq-len 512 --jepa-weight 0 --dspark-weight 0
          --engram-ram --engram-slots 2000000 --host-adam-every 0
          --steps 30 --log-every 10 --timers)
    BYTES_PER_STEP=$((10*512))
    ;;
  canary-bf16)
    [ "$AVAIL" -ge 6 ] || { echo "bench: need >=6G available, have ${AVAIL}G"; exit 1; }
    ARGS=(--preset small --batch 10 --seq-len 512 --jepa-weight 0 --dspark-weight 0 --bf16
          --engram-ram --engram-slots 2000000 --host-adam-every 0
          --steps 30 --log-every 10 --timers)
    BYTES_PER_STEP=$((10*512))
    ;;
  flagship)
    [ "$AVAIL" -ge 25 ] || { echo "bench: need >=25G available, have ${AVAIL}G"; exit 1; }
    ARGS=(--preset small --batch 10 --seq-len 512
          --engram-ram --engram-slots 48000000 --host-adam-every 1
          --steps 50 --log-every 10 --timers --memlog)
    BYTES_PER_STEP=$((10*512))
    ;;
  *) echo "usage: bench.sh [canary|canary-bf16|flagship]"; exit 1;;
esac

BIN=./target/release/train
[ -x "$BIN" ] || { echo "bench: build first (cargo build-train)"; exit 1; }

LOG=/tmp/opencode/bench/${MODE}_$$.log
"$BIN" --data "$DATA" "${ARGS[@]}" \
  --ckpt-dir /tmp/opencode/bench --ckpt-name bench_$MODE --log "$LOG" > /dev/null 2>&1

# steady-state step time: last timer line (sync-anchored)
TIMER=$(grep -a "^timer" "$LOG" | tail -1)
TOTAL_MS=$(echo "$TIMER" | grep -oE "total=[0-9]+ms" | grep -oE "[0-9]+")
[ -n "${TOTAL_MS:-}" ] || { echo "bench: no timer line, see $LOG"; exit 1; }

KBPS=$(python3 -c "print(f'{$BYTES_PER_STEP/($TOTAL_MS/1000)/1024:.2f}')")
MEM=$(grep -a "^step" "$LOG" | tail -1 | grep -oE "res=[0-9.]+MB" | head -1)
CE=$(grep -a "^done" "$LOG" | grep -oE "ce=[0-9.]+" | head -1)
LINE="$TS	$COMMIT	$MODE	${TOTAL_MS}ms	${KBPS}KB/s	${MEM:-res=n/a}	$CE"
echo "$LINE" | tee -a "$DIR/history.tsv"
echo "bench: appended to $DIR/history.tsv (compare with the previous line before merging)"
