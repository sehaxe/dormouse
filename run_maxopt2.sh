#!/usr/bin/env bash
# maxopt2 lane GPU runner. ONE CUDA process at a time; every cell waits for
# the card to be quiet (memory < 1 GB, no `train` process) before it starts.
# Usage: run_maxopt2.sh <cell> [extra train args]
# Recipe = the gbench2_control snapshot (small, b8, s512, max_iter=4,
# Fp8 factors, Muon+ ColRow ns=8, JEPA 0.05, dspark off, no engram).
set -u
BIN="$(cd "$(dirname "$0")" && pwd)/target/release/train"
DATA=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real
EVAL=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_eval
LOGS=/home/sehaxe/logs
mkdir -p "$LOGS"

wait_gpu() {
    for i in $(seq 1 180); do
        local mem
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "${mem:-99999}" -lt 1000 ] && ! pgrep -x train >/dev/null; then
            echo "gpu quiet after ${i}x20s (mem=${mem}MiB)"
            return 0
        fi
        sleep 20
    done
    echo "gpu never quiet" >&2
    exit 1
}

cell="${1:-}"
shift || true
extra=("$@")

case "$cell" in
reab) # retract-every A/B quality arm: $1 = cadence, $2 = seed
    wait_gpu
    "$BIN" --data "$DATA" --eval "$EVAL" --preset small --batch 8 --seq-len 512 \
        --no-engram --steps 2000 --eval-every 500 --eval-batches 20 \
        --retract-every "${extra[0]}" --seed "${extra[1]}" \
        --ckpt-name "maxopt2_re${extra[0]}_s${extra[1]}" --timers \
        --autotune full --log "$LOGS/maxopt2_re${extra[0]}_s${extra[1]}.log"
    ;;
jpre) # JEPA teacher targets precompute: $1 = steps
    wait_gpu
    "$BIN" --data "$DATA" --preset small --batch 8 --seq-len 512 \
        --no-engram --retract-every 4 --seed 1 --jepa-precompute "${extra[0]}" \
        --jepa-targets /home/sehaxe/maxopt2_jepa${extra[0]}.bin \
        --log "$LOGS/maxopt2_jpre${extra[0]}.log"
    ;;
jarm) # offline-JEPA quality arm: $1 = seed
    wait_gpu
    "$BIN" --data "$DATA" --eval "$EVAL" --preset small --batch 8 --seq-len 512 \
        --no-engram --steps 2000 --eval-every 500 --eval-batches 20 \
        --retract-every 4 --seed "${extra[0]}" \
        --jepa-targets /home/sehaxe/maxopt2_jepa2000.bin \
        --ckpt-name "maxopt2_jarm_s${extra[0]}" --timers --autotune full \
        --log "$LOGS/maxopt2_jarm_s${extra[0]}.log"
    ;;
atlas) # launch atlas replica for lever 3: $1 = extra flag (""|--jepa-targets ...)
    wait_gpu
    DM_LAUNCH_ATLAS=1 "$BIN" --data "$DATA" --preset small --batch 8 --seq-len 512 \
        --no-engram --steps 60 --retract-every 4 --seed 1 \
        --ckpt-name maxopt2_atlas --timers \
        --log "$LOGS/maxopt2_atlas.log" "${extra[@]}"
    ;;
atlas_jepa) # same with teacher precomputed
    wait_gpu
    DM_LAUNCH_ATLAS=1 "$BIN" --data "$DATA" --preset small --batch 8 --seq-len 512 \
        --no-engram --steps 60 --retract-every 4 --seed 1 \
        --jepa-targets /home/sehaxe/maxopt2_jepa2000.bin \
        --ckpt-name maxopt2_atlas_jepa --timers \
        --log "$LOGS/maxopt2_atlas_jepa.log"
    ;;
gpu) # warm ms/step: $1 = steps, $2 = label, rest = extra flags
    n="${extra[0]}"; label="${extra[1]}"; extra2=("${extra[@]:2}")
    wait_gpu
    "$BIN" --data "$DATA" --preset small --batch 8 --seq-len 512 \
        --no-engram --steps "$n" --retract-every 4 --seed 1 \
        --ckpt-name "maxopt2_${label}" --timers --log "$LOGS/maxopt2_${label}.log" \
        "${extra2[@]}"
    ;;
*) echo "unknown cell: $cell" >&2; exit 1 ;;
esac
