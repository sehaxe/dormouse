#!/usr/bin/env bash
# Context-length A/B at MATCHED BYTES: is 8 KB better than 1 KB on our recipe?
#
# Why: the project has never compared contexts at fixed data. `longctx1` was
# trained at seq_len=8192 (102.7M, CSA) and read 1.6882 at 523 MB; `flagship3`
# trained at 1024 (95M, dense) and reads 1.5185 at 246 MB — different models,
# different data, different recipes, so neither number is a context verdict.
#
# What it costs (arithmetic, d=832, 12 layers, per BYTE):
#   projections+FFN  72·d²                    = 49.8 MFLOP/B
#   attention        12 · 2 · d · t (QKᵀ+AV)  = 20.4 MFLOP/B at t=1024,
#                                               163.6 MFLOP/B at t=8192
#   => total 70.2 vs 213.4 MFLOP/B: 8 KB is 3.04x the FLOPs per byte, so the
#      same wall-clock buys 0.33x the bytes.
#
# The break-even gate, from our OWN fitted curve (L = 0.990 + 2.282·D^-0.2895,
# RMSE 0.002 over 9 points, docs/records/reviews/2026-11-09-architecture-matrix.md):
#   losing 2/3 of the data costs 0.082 BPB at the 2 GB point (1.241 -> 1.323).
#   So 8 KB must beat 1 KB by MORE than ~0.08 BPB to pay for its compute.
#   Anything less and 1 KB + more data wins.
#
# Protocol: same config except seq_len; batch sized to a similar tokens/step so
# the optimizer sees comparable batches; the eval is the SAME 8 KB stream for
# every arm (a 1 KB model is honestly penalised: it cannot see more).
#
#   bash scripts/seqlen_ab.sh            # 1k, 4k, 8k at 120 MB
#   BYTES=60000000 ARMS="1k 8k" bash scripts/seqlen_ab.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python

OUT=${OUT:-/home/sehaxe/logs/seqlen}
mkdir -p "$OUT"
BYTES=${BYTES:-120000000}
LR=${LR:-3e-4}
SEED=${SEED:-1}
ARMS=${ARMS:-1k 4k 8k}
MIX=${MIX:-configs/mix_refetch.toml}
DATA=${DATA:-../data_local/best/fwedu_train.bin}
EVAL=${EVAL:-../data_local/best/fwedu_eval.bin}

arm() { # tag seq batch
  local tag=$1 seq=$2 batch=$3
  if [ "$tag" != control ]; then
    case " $ARMS " in *" $tag "*) ;; *) return 0 ;; esac
  fi
  local steps=$(( BYTES / (batch * seq) ))
  local eval_every=$(( steps / 4 ))
  echo "=== $tag seq=$seq batch=$batch steps=$steps bytes=$BYTES $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train configs/flagship3_dense.toml \
    --data "$DATA" --eval "$EVAL" \
    --set steps=$steps --set batch=$batch --set seq_len=$seq --set max_seq_len=$seq \
    --set seed=$SEED --set lr=$LR --set opt=mix --set muon_lr_mult=12 \
    --set mix_plan="$MIX" --set doc_masking=false \
    --set eval=true --set eval_every=$eval_every --set log_every=$eval_every \
    --set ckpt_every=1000000 --set out_dir=/tmp/opencode/seq_$tag \
    --set ns_dtype=bf16 --set muon_state_dtype=bf16 --set torch_compile=true \
    --json-log "$OUT/$tag.jsonl" > "$OUT/$tag.log" 2>&1
  [ $? -eq 0 ] || echo "=== $tag REFUSED/FAILED (see $OUT/$tag.log)"
}

# tokens/step held near 24 576 (the live b24 x 1024) so the optimizer sees
# comparable batches; memory is the constraint at 8192 (attention is flash).
arm 1k 1024 24
arm 4k 4096 8
arm 8k 8192 4
