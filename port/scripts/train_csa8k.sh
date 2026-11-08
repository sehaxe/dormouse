#!/usr/bin/env bash
# Train the 1M-capable artifact: KDA + CSA + HCA, 8K window, 102.7M params.
#
# WHY this exists: the 1M claim needs a model whose CSA ring was TRAINED. The
# only CSA checkpoint on record (`longctx1`, 102.7M, 8K) died with the corpus on
# 2026-11-08, and the 95M dense lane cannot express CSA at all
# (`docs/research/2026-11-08-long-context-eval.md` §0). This is the re-run of
# longctx1's recipe on the refetched corpus; it is the artifact gate A of the
# S-NIAH protocol (`port/scripts/niah.py`).
#
# COST, measured on the previous run: 16 000 steps at 8 192 x batch 1 x accum 4
# = 524 MB took 7 h 38 m on this card (~37 kB/s). `STEPS` cuts that linearly:
# 8 000 steps ≈ 3 h 50 m, 4 000 ≈ 1 h 55 m.
#
#   STEPS=8000 bash scripts/train_csa8k.sh
#   STEPS=16000 OUT=/home/sehaxe/dormouse/checkpoints/csa8k2 bash scripts/train_csa8k.sh
#
# The data is the 2026-11-08 refetch (a mix plan of six sources, or one flat
# file). A number from this run is a NEW number: the old mix4 window is gone.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-8000}
OUT=${OUT:-/home/sehaxe/dormouse/checkpoints/csa8k}
DATA=${DATA:-/home/sehaxe/dormouse/data_local/best/fwedu_train.bin}
EVAL=${EVAL:-/home/sehaxe/dormouse/data_local/best/fwedu_eval.bin}
MIX=${MIX:-}     # e.g. configs/mix_refetch.toml; empty = the flat DATA file
NAME=${NAME:-csa8k}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

ARGS=(
  --set "steps=$STEPS" --set batch=1 --set grad_accum=4
  --set seq_len=8192 --set max_seq_len=8192
  --set lr=1e-3 --set warmup_tokens=16384000 --set decay_frac=0.1
  --set use_msa=false --set use_csa=true --set csa_chunk=8 --set csa_coarse_chunk=256
  --set use_mhc=true --set use_attnres=true --set share_u=true --set use_engram=true
  --set engram_ram=false --set attn_latent_rank=256
  --set moe_kernel=triton --set moe_capacity=0.0
  # NOT the legacy `sign` controller: at 8K/b1 it wound its bias to [-95, +57]
  # over 8 000 steps and collapsed the routing (9 of 512 experts live on the
  # 2026-11-08 csa8k artifact). `prop` is the bounded arm the 2026-11-04
  # collapse reviews measured — |bias| <= moe_lb_clamp, and the bank recovers.
  --set moe_lb_mode=prop --set moe_lb_gamma=0.001 --set moe_lb_clamp=2.0
  # doc_masking needs a `<bin>.docs.json` sidecar; the refetch has none, so the
  # flat stream is the honest arm (say so in any number it produces).
  --set doc_masking=false
  --set "data=$DATA" --set "eval_data=$EVAL"
  --set eval=true --set eval_every=500 --set eval_batches=4
  --set log_every=100 --set ckpt_every=1000
  --set "ckpt_name=$NAME" --set "out_dir=$OUT"
)
[ -n "$MIX" ] && ARGS+=(--set "mix_plan=$MIX")

exec .venv/bin/dormouse train medium "${ARGS[@]}"
