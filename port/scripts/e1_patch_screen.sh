#!/usr/bin/env bash
# E1 — the decisive screen: equal GPU-hours, bytes vs patch units.
#
# The question the owner asked: does a "smarter unit" (8 bytes per position)
# buy QUALITY per unit of wall clock, or only speed?
#
# Design (pre-registered before the run):
#   arm A  configs/flagship3_dense.toml      1 byte/position, sees B_A bytes
#   arm B  configs/arch_dense_byteflow8.toml 8 bytes/position, sees ~B_A*5.5 bytes
#   both run the SAME wall-clock per seed (see hours below), 3 seeds.
#
# At the measured 48-60 kB/s the byte arm covers ~1 GB in 4.6-5.8 h; the patch
# arm covers 5.2-7.6 GB in the same time. That is the whole point: same cost,
# 5-7x more text. If BPB does not improve, patching is a speed lever only.
#
# Pre-registered rule:
#   * B wins only if mean(BPB_B) < mean(BPB_A) - max(2*spread, 0.02);
#   * if B is within +-0.02 it is a TIE on quality and a WIN on bytes/second,
#     which still means we adopt it for the 1B run (more bytes seen per hour);
#   * if B is worse by >0.05 BPB it is a LOSS and the 1B run goes on byte units.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/e1_patch
mkdir -p "$OUT"

HOURS=${HOURS:-5}          # per seed, both arms
STEPS_A=${STEPS_A:-10000}  # ~1 GB at b8x1024x3 (24 576 B/step)
STEPS_B=${STEPS_B:-10000}  # same step count, 8x the bytes per step

run() { # tag config steps seed
  local tag=$1 cfg=$2 steps=$3 seed=$4
  echo "=== $tag seed=$seed $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train "$cfg" \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=$steps --set batch=8 --set seed=$seed \
    --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
    --set eval=true --set eval_every=$((steps/2)) \
    --set mix_plan=configs/mix_v7_docmask.toml --set doc_masking=true --set log_every=1000 \
    --set ckpt_every=1000000 --set ckpt_name="e1_${tag}_s${seed}" \
    --set out_dir=/tmp/opencode/e1_${tag}_s${seed} \
    --json-log "$OUT/${tag}_s${seed}.jsonl" > "$OUT/${tag}_s${seed}.log" 2>&1
  grep -oE "eval step=[0-9]+ off=[0-9]+ bpb_fin=[0-9.]+" "$OUT/${tag}_s${seed}.log" | tail -2
  grep -oE "[0-9]+ B/s|[0-9.]+ GiB" "$OUT/${tag}_s${seed}.log" | tail -2
}

for seed in 1 2 3; do
  run bytes configs/flagship3_dense.toml "$STEPS_A" $seed
  run patch configs/arch_dense_byteflow8.toml "$STEPS_B" $seed
done

echo "=== summary $(date +%H:%M:%S)"
$PY - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
by = {}
bytes_seen = {}
for f in sorted(d.glob("*.jsonl")):
    arm = f.stem.split("_")[0]
    ev = [json.loads(l) for l in open(f) if l.strip()]
    pts = [r for r in ev if r.get("event") == "eval" and "bpb_fin" in r]
    if pts:
        by.setdefault(arm, []).append(pts[-1]["bpb_fin"])
        bytes_seen[arm] = pts[-1]["step"] * 8 * 1024 * 3
for k, v in by.items():
    print(f"{k:6s} n={len(v)} mean={statistics.mean(v):.4f} spread={max(v)-min(v):.4f} "
          f"bytes~{bytes_seen.get(k, 0)/1e9:.2f} GB")
if "bytes" in by and "patch" in by:
    a, b = statistics.mean(by["bytes"]), statistics.mean(by["patch"])
    print(f"delta (patch - bytes) = {b - a:+.4f}")
PY
