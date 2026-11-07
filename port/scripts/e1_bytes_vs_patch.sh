#!/usr/bin/env bash
# E1 v2 — quality per byte, bytes vs patch-8, at MATCHED BYTES and matched data
# treatment. v1 never ran: its patch arm was refused by the doc_masking gate
# (`doc_masking=true` + `use_byteflow=true` is loud on purpose — a patch spans a
# document cut and the loss mask addresses bytes), and its bytes arm was killed
# at 25% by another GPU job. This version sets doc_masking=false on BOTH arms so
# the comparison stays matched.
#
# Measured step time on this card while TRAINING (not forward-only):
#   bytes  b=8  s=1024  8192 B/step   0.307 s/step   37.4 us/byte   26.7 kB/s
#   patch  b=32 s=1024 32768 B/step   0.264 s/step    8.1 us/byte  124.3 kB/s
#   -> 4.6x fewer seconds per byte for the patch arm (the 7.4x figure was
#      forward-only; backward and the optimizer dilute it).
#
# Pre-registered rule (unchanged from v1):
#   * patch wins if BPB_patch <= BPB_bytes + 0.02 at matched bytes;
#   * patch loses if BPB_patch > BPB_bytes + 0.05;
#   * between 0.02 and 0.05 -> "pays a small quality tax", decide by wall clock
#     (4.6x more bytes per hour beats a 0.03 BPB tax).
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=${OUT:-/home/sehaxe/logs/e1v2}
mkdir -p "$OUT"

BYTES=${BYTES:-250000000}   # 0.25 GB per arm per seed
# FAST=1 runs BOTH arms on the Muon precision rung (bf16 NS, 3 steps, bf16
# state). It must be the SAME on both arms or the A/B compares two optimizers:
# the rung is a training-speed change, and the patch question is about the unit
# of computation.
FAST=${FAST:-0}
FOPT=()
if [ "$FAST" = 1 ]; then
  FOPT=(--set ns_dtype=bf16 --set ns_steps=3 --set muon_state_dtype=bf16)
fi

run() { # tag config batch bytes_per_step seed
  local tag=$1 cfg=$2 batch=$3 bps=$4 seed=$5 steps eval_every
  steps=$(( BYTES / bps ))
  eval_every=$(( steps / 4 ))
  echo "=== $tag seed=$seed steps=$steps batch=$batch fast=$FAST $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train "$cfg" \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=$steps --set batch=$batch --set seq_len=1024 --set seed=$seed \
    --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
    --set eval=true --set eval_every=$eval_every \
    --set mix_plan=configs/mix_v7.toml --set doc_masking=false --set log_every=$eval_every \
    --set ckpt_every=1000000 --set ckpt_name="e1_${tag}_s${seed}" \
    --set out_dir=/tmp/opencode/e1v2_${tag}_s${seed} ${FOPT[@]+"${FOPT[@]}"} \
    --json-log "$OUT/${tag}_s${seed}.jsonl" > "$OUT/${tag}_s${seed}.log" 2>&1
  grep -oE "eval step=[0-9]+ off=[0-9]+ bpb_fin=[0-9.]+" "$OUT/${tag}_s${seed}.log" | tail -4
}

# SEEDS=1 runs the first seed only: the pre-registered rule is a 0.02/0.05
# decision band, so a single seed settles it whenever the gap is outside that
# band; the second seed is only worth its hours in the ambiguous middle.
SEEDS=${SEEDS:-2}
for seed in $(seq 1 "$SEEDS"); do
  run bytes configs/flagship3_dense.toml     8  $((8*1024))  $seed
  run patch configs/arch_dense_byteflow8.toml 32 $((32*1024)) $seed
done

echo "=== summary $(date +%H:%M:%S)"
$PY - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path

def last_eval(p):
    rows = [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    ev = [r for r in rows if r.get("event") == "eval"]
    return (ev[-1]["bpb_fin"], ev[-1]["bytes_seen"]) if ev else (None, None)

out = Path(sys.argv[1])
for tag in ("bytes", "patch"):
    vals = []
    for seed in range(1, int(__import__("os").environ.get("SEEDS", "2")) + 1):
        p = out / f"{tag}_s{seed}.jsonl"
        if p.exists():
            bpb, seen = last_eval(p)
            if bpb is not None:
                vals.append(bpb)
                print(f"{tag} s{seed}: bpb_fin={bpb:.4f} at {seen/1e9:.3f} GB")
    if vals:
        print(f"{tag}: mean={statistics.mean(vals):.4f} n={len(vals)}")
PY
