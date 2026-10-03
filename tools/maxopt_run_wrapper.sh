#!/usr/bin/env bash
# maxopt lane GPU cell runner. ONE CUDA process at a time; every cell waits
# for the card to be quiet (memory < 1 GB, no `train` process) before it
# starts. Usage: run_maxopt.sh <cell> [<args passed to train after flags>...]
#
# Cells are defined below; provenance of the recipe = copies of the
# gbench2_control.config.toml snapshot (main, 2026-10-02).
set -u
BIN="$(cd "$(dirname "$0")" && pwd)/target/release/train"
DATA=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real
EVAL=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_eval
LOGS=/home/sehaxe/logs
mkdir -p "$LOGS"

wait_gpu() {
    for i in $(seq 1 120); do
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
atlas)    # launch atlas replica on this tree, control shape
    wait_gpu
    DM_LAUNCH_ATLAS=1 "$BIN" --data "$DATA" --preset small --batch 8 --seq-len 512 \
        --no-engram --steps 30 --retract-every 4 --seed 1 --ckpt-name maxopt_atlas2 --log-every 10 \
        --timers --log "$LOGS/maxopt_atlas.log"
    ;;
gctrl)    # ms/step control replica: no evals, like gbench2_control
    wait_gpu
    "$BIN" --data "$DATA" --preset small --batch 8 --seq-len 512 \
        --no-engram --steps 500 --retract-every 4 --seed 1 --ckpt-name maxopt_gctrl \
        --timers --log "$LOGS/maxopt_gctrl.log"
    ;;
rab)      # retract-every A/B quality arm: $1 = cadence, $2 = seed
    wait_gpu
    "$BIN" --data "$DATA" --eval "$EVAL" --preset small --batch 8 --seq-len 512 \
        --no-engram --steps 2000 --eval-every 500 --eval-batches 20 --retract-every "$1" \
        --seed "$2" --ckpt-name "maxopt_re${1}_s${2}" --timers \
        --log "$LOGS/maxopt_re${1}_s${2}.log"
    ;;
jpre)     # JEPA teacher-target precompute pass: $1 = n_steps
    wait_gpu
    "$BIN" --data "$DATA" --preset small --batch 8 --seq-len 512 \
        --no-engram --retract-every 4 --seed 1 --jepa-precompute "$1" \
        --jepa-targets /home/sehaxe/maxopt_jepa"${1}".bin \
        --log "$LOGS/maxopt_jpre${1}.log"
    ;;
jarm)     # offline-JEPA quality arm: $1 = seed
    wait_gpu
    DM_LAUNCH_ATLAS=1 "$BIN" --data "$DATA" --eval "$EVAL" --preset small --batch 8 --seq-len 512 \
        --no-engram --steps 2000 --eval-every 500 --eval-batches 20 --retract-every 4 \
        --seed "$1" --jepa-targets /home/sehaxe/maxopt_jepa2000.bin \
        --ckpt-name "maxopt_jpre_s${1}" --timers --autotune full \
        --log "$LOGS/maxopt_jarm_s${1}.log"
    ;;
*)
    echo "unknown cell: $cell" >&2
    exit 1
    ;;
esac
