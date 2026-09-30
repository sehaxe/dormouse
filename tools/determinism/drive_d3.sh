#!/usr/bin/env bash
# Deliverable 1+2 driver. Triple 1 = back-to-back pair + long-gap run.
# Triple 2 = repeat of the same shape, to satisfy "run it at least twice".
# Every run is step 0 (init determinism), seed 7.
set -u
H=/home/sehaxe/dormouse-wt/determinism3/tools/determinism
D=/home/sehaxe/cache/determinism
log(){ echo "[$(date +%H:%M:%S)] $*"; }
log "GPU preflight"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
pgrep -ax train || log "no train running"

log "TRIPLE 1: d3a, d3b back-to-back, then d3c after a 9-minute idle gap"
for L in d3a d3b; do
  "$H/run_d3.sh" $L >/dev/null 2>&1
  date +%s > "$D/$L.epoch"
  log "$L done"
done
log "idling 540 s (machine-state arm: does wall time move the number?)"
sleep 540
"$H/run_d3.sh" d3c >/dev/null 2>&1; date +%s > "$D/d3c.epoch"; log "d3c done"

log "TRIPLE 2 (replication): d3d, d3e back-to-back, then d3f after a 9-minute gap"
for L in d3d d3e; do
  "$H/run_d3.sh" $L >/dev/null 2>&1
  date +%s > "$D/$L.epoch"
  log "$L done"
done
log "idling 540 s"
sleep 540
"$H/run_d3.sh" d3f >/dev/null 2>&1; date +%s > "$D/d3f.epoch"; log "d3f done"
log "ALL DONE"
