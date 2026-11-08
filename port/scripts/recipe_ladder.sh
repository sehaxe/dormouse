#!/usr/bin/env bash
# The recipe ladder: five cheap knobs from Ripple-65B's v1.0→v1.6 (16x compute
# efficiency at 4B), each its own arm at 95M on byte units, plus an all-on arm.
# Source: docs/reviews/2026-11-08-ripple-65b.md.
#
# Fixed budget per arm: BYTES (default 120 MB) at b=8, s=1024, lr=3e-4,
# muon_lr_mult=12, mix_v7 with doc_masking=false — the same data treatment as
# E1 v2, so an arm's BPB is comparable to the E1 control (bpb_fin at off=0).
#
# Read the result as a RANKING, not as a law: one seed per arm. Any arm that
# moves BPB by more than ~0.02 earns two more seeds; anything under the noise
# floor is dropped.
#
#   control        all knobs off
#   vr             value_residual=true          (ResFormer, 2410.17897)
#   gate           attn_gate=true               (sigmoid attention, 2505.06708)
#   clip           ffn_clip=7.0                 (GPT-OSS, 2505.09388)
#   nope           nope_layers=2                (global NoPE proxy, 2501.18795)
#   cooldown       min_lr_frac=0.0              (land on zero, not 0.1×peak)
#   all            every knob on together
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/ladder
mkdir -p "$OUT"
BYTES=${BYTES:-120000000}
# The corpus was deleted 2026-11-08 (docs/reviews/2026-11-08-merge-incident.md).
# MIX/EVAL let the ladder run on the refetched sources; a run on those is a NEW
# number and the caller must tag it (RUN_TAG).
MIX=${MIX:-configs/mix_v7.toml}
EVAL=${EVAL:-../data_local/best/fwedu_eval.bin}
DATA=${DATA:-../data_local/best/fwedu_train.bin}
TAG=${RUN_TAG:-}
# FAST=1 runs every arm on the SAFE Muon rung (see e1_bytes_vs_patch.sh):
# ns_dtype=bf16 + state bf16, ns_steps stays 5.
FAST=${FAST:-0}
FOPT=()
if [ "$FAST" = 1 ]; then
  FOPT=(--set ns_dtype=bf16 --set muon_state_dtype=bf16)
fi
STEPS=$(( BYTES / (8 * 1024) ))
EVAL_EVERY=$(( STEPS / 4 ))

run() { # tag extra-set-args...
  local tag=$1; shift
  echo "=== $tag steps=$STEPS $* $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train configs/flagship3_dense.toml \
    --data "$DATA" --eval "$EVAL" \
    --set steps=$STEPS --set batch=8 --set seq_len=1024 --set seed=1 \
    --set lr=3e-4 --set opt=mix --set muon_lr_mult=12 \
    --set mix_plan="$MIX" --set doc_masking=false \
    --set eval=true --set eval_every=$EVAL_EVERY --set log_every=$EVAL_EVERY \
    --set ckpt_every=1000000 --set out_dir=/tmp/opencode/ladder_$tag \
    --json-log "$OUT/${tag}${TAG}.jsonl" ${FOPT[@]+"${FOPT[@]}"} "$@" > "$OUT/$tag.log" 2>&1
}

run control
run vr     --set value_residual=true
run gate   --set attn_gate=true
run clip   --set ffn_clip=7.0
run nope   --set nope_layers=2
run cooldown --set min_lr_frac=0.0
run all    --set value_residual=true --set attn_gate=true --set ffn_clip=7.0 \
           --set nope_layers=2 --set min_lr_frac=0.0

echo "=== ranking $(date +%H:%M:%S)"
$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path

rows = []
for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    ev = [json.loads(l) for l in p.read_text().splitlines() if '"eval"' in l]
    if ev:
        rows.append((p.stem, ev[-1]["bpb_fin"], ev[-1]["bytes_seen"] / 1e6))
base = next((b for t, b, _ in rows if t.startswith("control")), None)
for tag, bpb, mb in sorted(rows, key=lambda r: r[1]):
    delta = "" if base is None else f"  delta {bpb - base:+.4f}"
    print(f"{tag:12s} bpb_fin={bpb:.4f} at {mb:.0f} MB{delta}")
PY
