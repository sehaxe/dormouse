#!/usr/bin/env bash
# Throughput ladder — pure SPEED, zero quality risk.
#
# Our measured MFU is 13.9% (6.35 TFLOP/s of a 45.7 TFLOP/s bf16 peak) and a
# warm step leaves the card idle most of the time. Ripple-65B reports 40% MFU on
# a 65B dense model with "simple sharding, no custom kernels" — so the gap is
# not the card, it is the step. Every factor recovered here multiplies the
# patch/recipe/MoD gains instead of competing with them.
#
# Each arm runs STEPS steps (default 40, the first 10 are compile warmup) and we
# take the steady-state bytes/s off the JSONL log line, not the wall clock.
#
#   bytes  b=8   main arms: batch 8/16/32, compile off/on, opt_graph off/on
#   patch  b=32  the same at patch-8 (the E1 arm's own best batch)
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/thru
mkdir -p "$OUT"
STEPS=${STEPS:-40}

arm() { # tag config batch extra...
  local tag=$1 cfg=$2 batch=$3; shift 3
  $PY -u -m dormouse.cli train "$cfg" \
    --data ../data_local/best/fwedu.bin --set steps=$STEPS --set batch=$batch \
    --set seq_len=1024 --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
    --set eval=false --set mix_plan=configs/mix_v7.toml --set doc_masking=false \
    --set log_every=10 --set ckpt_every=1000000 --set out_dir=/tmp/opencode/thru_$tag \
    --json-log "$OUT/$tag.jsonl" "$@" > "$OUT/$tag.log" 2>&1
  echo "=== $tag done $(date +%H:%M:%S)"
}

D=configs/flagship3_dense.toml
P=configs/arch_dense_byteflow8.toml
arm bytes_b8        $D  8
arm bytes_b16       $D 16
arm bytes_b32       $D 32
arm bytes_b8_comp   $D  8 --set torch_compile=true
arm bytes_b16_comp  $D 16 --set torch_compile=true
arm bytes_b8_graph  $D  8 --set opt_graph=true
arm patch_b32       $P 32
arm patch_b64       $P 64
arm patch_b32_comp  $P 32 --set torch_compile=true

$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
print(f"{'arm':16s} {'B/s':>10s} {'B/step':>8s} {'us/byte':>8s}")
rows = []
for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    logs = [json.loads(l) for l in p.read_text().splitlines() if '"log"' in l]
    if not logs:
        print(f"{p.stem:16s} {'(no log line)':>10s}")
        continue
    r = logs[-1]
    rows.append((r["tokens_per_s"], p.stem, r["bytes_seen"] / (r["step"] + 1)))
for rate, tag, bps in sorted(rows, reverse=True):
    print(f"{tag:16s} {rate:10.0f} {bps:8.0f} {1e6/rate:8.2f}")
PY
