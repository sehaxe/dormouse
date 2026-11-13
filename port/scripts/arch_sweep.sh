#!/usr/bin/env bash
# The architecture matrix, at a budget where a 0.05 BPB delta is visible.
#
# Why it exists: the live recipe (`configs/flagship3_dense.toml`) is a stock
# 12-layer transformer — `max_iter=1`, no MoE, no TSCT, no mHC/AttnRes, no loop.
# The 2026-11-06 "dense beats the loop by 0.126-0.37" verdict is NOT a block
# comparison: the loop arm ran with the broken Muon LR (0.7-17% of the AdamW
# half's first-order step, worth 0.248 BPB elsewhere) and no loop run ever used
# the live multiplier (docs/reviews/2026-11-08-loop-ab-fairness.md). Every arm
# here runs `muon_lr_mult=12`, the live value.
#
# The forms (validated with `dormouse info`, refusals in config.py):
#   transformer — `n_layers` unique blocks, one pass (the control)
#   kda         — the LoopBlock: `loop_blocks` unique bodies × `max_iter` passes,
#                 `n_layers=0` (the loop's depth is `loop_blocks`)
#   mHC/AttnRes are refused on the transformer form and live on the kda form.
#
#   ARMS="loop4x4 loop4x4_mhc loop8x8" bash scripts/arch_sweep.sh
#   BYTES=60000000 ARMS="tsct moe32" bash scripts/arch_sweep.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python

OUT=${OUT:-/home/sehaxe/logs/arch}
mkdir -p "$OUT"
BYTES=${BYTES:-120000000}
BATCH=${BATCH:-24}
LR=${LR:-3e-4}
MUON=${MUON:-12}
SEED=${SEED:-1}
TAG=${RUN_TAG:-}
MIX=${MIX:-configs/mix_refetch.toml}
DATA=${DATA:-../data_local/best/fwedu_train.bin}
# The DENOMINATOR of the sweep. Default is the historical control; a later arm
# that won its A/B (e.g. `arch_gated_swiglu.toml`) becomes the base for the next
# question, so the comparison is always against the current best on record.
BASE=${BASE:-configs/flagship3_dense.toml}
ARMS=${ARMS:-loop4x4 loop4x4_mhc loop8x8 tsct moe32}
STEPS=$(( BYTES / (BATCH * 1024) ))
EVAL_EVERY=$(( STEPS / 4 ))

arm() { # tag extra-set-args...
  local tag=$1; shift
  # `control` is always the denominator; never skip it.
  if [ "$tag" != control ]; then
    case " $ARMS " in *" $tag "*) ;; *) return 0 ;; esac
  fi
  echo "=== $tag steps=$STEPS batch=$BATCH bytes=$BYTES seed=$SEED $* $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train "$BASE" \
    --data "$DATA" --eval ../data_local/best/fwedu_eval.bin \
    --set steps=$STEPS --set batch=$BATCH --set seq_len=1024 --set seed=$SEED \
    --set lr=$LR --set opt=mix --set muon_lr_mult=$MUON \
    --set mix_plan="$MIX" --set doc_masking=false \
    --set eval=true --set eval_every=$EVAL_EVERY --set log_every=$EVAL_EVERY \
    --set ckpt_every=1000000 --set out_dir=/tmp/opencode/arch_$tag \
    --set ns_dtype=bf16 --set muon_state_dtype=bf16 \
    --set torch_compile=true \
    --json-log "$OUT/${tag}${TAG}.jsonl" "$@" > "$OUT/$tag.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || echo "=== $tag REFUSED/FAILED rc=$rc (see $OUT/$tag.log)"
}

LOOPB=(--set block_form=kda --set n_layers=0 --set use_kda=true)

arm control
# `use_kda=false` on the kda form is the ZERO attention branch (no attention at
# all), so every loop arm below must name its attention arm explicitly.
arm loop4x4_full   "${LOOPB[@]}" --set loop_blocks=4 --set max_iter=4 --set attn_kind=full
arm loop4x4_kda    "${LOOPB[@]}" --set loop_blocks=4 --set max_iter=4 --set attn_kind=kda
arm loop8x8_full   "${LOOPB[@]}" --set loop_blocks=8 --set max_iter=8 --set attn_kind=full
arm loop4x4_mhc    "${LOOPB[@]}" --set loop_blocks=4 --set max_iter=4 --set attn_kind=full --set use_mhc=true --set use_attnres=true
# The only loop arm that can see token ORDER with exact attention: `full` with
# `attn_pos=none` measured 3.66 BPB at 30 MB vs 2.199 for `kda` (2026-11-09).
arm loop4x4_rope   "${LOOPB[@]}" --set loop_blocks=4 --set max_iter=4 --set attn_kind=full --set attn_pos=rope
arm loop8x8_rope   "${LOOPB[@]}" --set loop_blocks=8 --set max_iter=8 --set attn_kind=full --set attn_pos=rope
# Equal-PARAMETER loop: the 18.3M arms are compared against a 95M control, which
# is the honest cost axis for wall-clock but not for capacity. `d_model=1792`
# (n_heads*head_dim = 28*64) puts 4 blocks x 4 passes at ~95M.
arm loop4x4_wide   "${LOOPB[@]}" --set loop_blocks=4 --set max_iter=4 --set attn_kind=full --set attn_pos=rope --set d_model=1792 --set n_heads=28 --set head_dim=64
arm tsct         --set use_tsct=true
arm moe32        --set n_experts=32 --set moe_topk=2 --set moe_capacity=2.0 --set ffn_form=bank
# Gated FFN (2026-11-10). Param-NEUTRAL against the control's `d_ffn=3072`:
# `3·d·2048 == 2·d·3072` exactly, so the only variable is the FFN's form.
# `d_ffn_gelu` is what the parity check compares against; without it the guard
# falls back to `4·d` and refuses 2048 (correctly — that would be a smaller
# model, not a different FFN).
arm gated_silu   --set ffn_gated=true --set expert_act=silu --set d_ffn=2048 --set d_ffn_gelu=3072
arm gated_situ   --set ffn_gated=true --set expert_act=situ --set d_ffn=2048 --set d_ffn_gelu=3072
# Full AttnRes (N=L) on the gated base: `BASE=configs/arch_gated_swiglu.toml
# ARMS="attnres"`. At L=12 the paper's block ablation says N=L is full and N=8
# already saturates; we run the full variant because it costs 0.02% of the step.
arm attnres      --set use_attnres=true

$PY - "$OUT" "$TAG" <<'PY'
import json, sys
from pathlib import Path
out, tag = Path(sys.argv[1]), sys.argv[2]
rows = {}
for p in sorted(out.glob(f"*{tag}.jsonl")):
    ev = [json.loads(l) for l in p.read_text().splitlines()]
    ev = [r for r in ev if "bpb" in r]
    if ev:
        rows[p.stem] = (ev[-1]["bpb"], ev[-1]["step"])
base = rows.get(f"control{tag}", (None, 0))[0]
print(f"\n{'arm':>14s} {'bpb':>7s} {'vs ctl':>8s}")
for name, (bpb, step) in sorted(rows.items(), key=lambda kv: kv[1][0]):
    d = f"{bpb - base:+8.4f}" if base else ""
    print(f"{name:>14s} {bpb:7.4f} {d}")
PY
