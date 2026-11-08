#!/usr/bin/env bash
# The budget run: the dense 95M on a Chinchilla-scale byte budget.
#
# WHY: the ladder's control reached 1.4906 BPB after 120 MB = 6.3% of the
# 20-bytes-per-parameter budget for 95M (1.9 GB). The curve was still falling
# steeply (1.77 @30 MB -> 1.49 @120 MB), so this run spends the card where the
# arithmetic says the model has not been fed yet. ~9.4 h at the measured
# 56 kB/s (7.8 h if `torch_compile` wins its A/B and is enabled).
#
# Every number from it is a NEW number on the refetched corpus.
#
#   bash scripts/train_budget.sh                       # 1.9 GB, b8, lr 3e-4
#   BYTES=2000000000 BATCH=8 ACCUM=4 LR=4e-4 COMPILE=1 bash scripts/train_budget.sh
#
# ACCUM grows the optimizer's batch without growing peak memory (the card fits
# b8; a real b32 does not). COMPILE=1 wraps the model in torch.compile (the A/B
# is `scripts/ab_compile.sh`; check it before believing the 1.2x).
set -euo pipefail
cd "$(dirname "$0")/.."

BYTES=${BYTES:-1900000000}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-1}
LR=${LR:-3e-4}
COMPILE=${COMPILE:-0}
# EMA horizon = 1/(1-decay) steps. The ladder's 0.999 is a 1 000-step horizon —
# 0.4 % of this run, too short to buy anything; 0.9999 is ~10 000 steps (~4 %).
EMA=${EMA:-0.9999}
TAG=${TAG:-budget}
OUT=${OUT:-/home/sehaxe/dormouse/checkpoints/$TAG}
MIX=${MIX:-configs/mix_refetch.toml}
EVAL=${EVAL:-../data_local/best/fwedu_eval.bin}
DATA=${DATA:-../data_local/best/fwedu_train.bin}

STEPS=$(( BYTES / (BATCH * ACCUM * 1024) ))
EVAL_EVERY=$(( STEPS / 20 ))
[ "$EVAL_EVERY" -lt 1 ] && EVAL_EVERY=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

ARGS=(
  --set "steps=$STEPS" --set "batch=$BATCH" --set "grad_accum=$ACCUM" --set "lr=$LR"
  --set seq_len=1024 --set seed=1 --set opt=mix --set muon_lr_mult=12
  --set "mix_plan=$MIX" --set doc_masking=false
  --set "data=$DATA" --set "eval_data=$EVAL"
  --set eval=true --set "eval_every=$EVAL_EVERY" --set eval_bytes=196608
  --set log_every="$EVAL_EVERY" --set ckpt_every=1000
  --set ns_dtype=bf16 --set muon_state_dtype=bf16
  # EMA is scored by the eval (loop.py wraps `_eval_line` in `_shadow_params`),
  # so this is a free change to the REPORTED number, not just the weights.
  --set "ema_decay=$EMA"
  --set "ckpt_name=$TAG" --set "out_dir=$OUT"
  --json-log "/home/sehaxe/logs/${TAG}.jsonl"
)
[ "$COMPILE" = 1 ] && ARGS+=(--set torch_compile=true)

echo "=== $TAG steps=$STEPS bytes=$(( STEPS * BATCH * ACCUM * 1024 )) b=$BATCH a=$ACCUM lr=$LR compile=$COMPILE ema=$EMA $(date +%H:%M:%S)"
exec .venv/bin/dormouse train configs/flagship3_dense.toml "${ARGS[@]}"
