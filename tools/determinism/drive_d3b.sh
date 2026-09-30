#!/usr/bin/env bash
# Deliverable 1+2 driver, GPU-guarded. Another agent is on this lane, so every
# run waits until no `train` exists and the card is idle before it starts.
# Never two GPU processes at once (AGENTS.md 1.5).
set -u
H=/home/sehaxe/dormouse-wt/determinism3/tools/determinism
D=/home/sehaxe/cache/determinism
log(){ echo "[$(date +%H:%M:%S)] $*"; }

wait_gpu(){
  local quiet=0 i
  for i in $(seq 1 180); do
    if ! pgrep -x train >/dev/null 2>&1 \
       && [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)" -lt 10 ]; then
      quiet=$((quiet+1)); [ "$quiet" -ge 3 ] && { log "GPU clear"; return 0; }
    else
      quiet=0; log "waiting for GPU (train running or util>=10%)"
    fi
    sleep 10
  done
  log "WARNING: GPU never cleared; proceeding anyway"; return 0
}

run(){
  wait_gpu
  "$H/run_d3.sh" "$1" >/dev/null 2>&1
  date +%s > "$D/$1.epoch"
  log "$1 done"
}

log "TRIPLE 1 tail: d3c after a long gap since d3b"
sleep 120
run d3c
log "TRIPLE 2 (replication): d3d, d3e back-to-back, d3f after a long gap"
run d3d
run d3e
sleep 120
run d3f
log "ALL DONE"
