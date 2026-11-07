#!/usr/bin/env bash
# Which part of the fast Muon rung broke the loss? — 2000 steps per arm.
#
# E1 v3 (fast rung: ns_dtype=bf16, ns_steps=3, muon_state_dtype=bf16) measured
# bpb 2.12 at 125 MB against 1.63 at the same bytes on the fp32 rung. That is
# 0.49 BPB — not a precision cost, a broken update. This ladder isolates the
# three rungs so the fast lane can be rebuilt from the ones that are safe.
#
#   fp32_5   control: ns fp32, 5 steps, state fp32   (every number on record)
#   bf16_5   ns bf16, 5 steps, state fp32
#   fp32_3   ns fp32, 3 steps, state fp32
#   bf16_3   ns bf16, 3 steps, state fp32
#   state    ns fp32, 5 steps, state bf16
#
# 2000 steps = 16.4 MB, ~5 min an arm on the fp32 rung. Pre-registered rule:
# the rung that ships is the fastest one whose bpb at 16.4 MB is within 0.02 of
# the control's.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/muonq
mkdir -p "$OUT"
STEPS=${STEPS:-2000}

run() { # tag extra...
  local tag=$1; shift
  echo "=== $tag $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train configs/flagship3_dense.toml \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=$STEPS --set batch=8 --set seq_len=1024 --set seed=1 \
    --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
    --set mix_plan=configs/mix_v7.toml --set doc_masking=false \
    --set eval=true --set eval_every=1000 --set log_every=1000 \
    --set ckpt_every=1000000 --set out_dir=/tmp/opencode/muonq_$tag \
    --json-log "$OUT/$tag.jsonl" "$@" > "$OUT/$tag.log" 2>&1
}

run fp32_5
run bf16_5  --set ns_dtype=bf16
run fp32_3  --set ns_steps=3
run bf16_3  --set ns_dtype=bf16 --set ns_steps=3
run state   --set muon_state_dtype=bf16

$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
rows = []
for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    ev = [json.loads(l) for l in p.read_text().splitlines() if '"eval"' in l]
    if ev:
        rows.append((ev[-1]["bpb_fin"], p.stem, ev[-1]["bytes_seen"] / 1e6))
for bpb, tag, mb in sorted(rows):
    print(f"{tag:8s} bpb={bpb:.4f} at {mb:.1f} MB")
PY
