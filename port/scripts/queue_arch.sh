#!/usr/bin/env bash
# The architecture queue — the owner's hypothesis, tested with a WORKING optimizer.
#
# Every earlier architecture verdict in this repo was measured with a crippled
# Muon (15-55x too small steps on 99.5% of params), so "dense wins at 95M" is
# currently an unverified claim in BOTH directions. This queue re-measures the
# three forms at matched total parameters (~95M), matched bytes, matched steps:
#
#   dense   configs/flagship3_dense.toml   95,035,200 params, 127 MFLOP/byte
#   kda     configs/arch_hybrid_kda.toml   95,875,404 params, linear attention, dense FFN
#   moe8    configs/arch_moe8.toml         94,072,908 params, 8 experts top-2
#
# Pre-registered decision rule (written before the runs):
#   * a form wins only if its mean is below the dense control's mean by more
#     than 2x its own spread AND by at least 0.02 BPB;
#   * ties go to the dense form (fewer moving parts, and it is the one every
#     other measurement is comparable to);
#   * a form that wins on BPB but costs >1.3x wall-clock per byte is flagged,
#     not adopted — the owner's criterion is quality per unit time, not BPB alone.
#
# 2 seeds x 3000 steps (73.7 MB) per arm, doc_masking on, honest metric
# (bpb_fin, 196 608 B window at offset 0).
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/arch_queue
MULT=${MULT:-12}
mkdir -p "$OUT"

run() { # tag config seed
  local tag=$1 cfg=$2 seed=$3
  local t0 t1
  echo "=== $tag seed=$seed $(date +%H:%M:%S)"
  t0=$(date +%s)
  $PY -u -m dormouse.cli train "$cfg" \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=3000 --set batch=8 --set seq_len=1024 --set grad_accum=3 \
    --set opt=mix --set lr=3e-4 --set muon_lr_mult=$MULT --set seed=$seed \
    --set eval=true --set eval_every=1500 \
    --set mix_plan=configs/mix_v7_docmask.toml --set doc_masking=true --set log_every=1500 \
    --set ckpt_every=9000 --set ckpt_name="aq_${tag}_s${seed}" \
    --set out_dir=/tmp/opencode/aq_${tag}_s${seed} \
    --json-log "$OUT/${tag}_s${seed}.jsonl" > "$OUT/${tag}_s${seed}.log" 2>&1
  t1=$(date +%s)
  local bpb
  bpb=$(grep -oE "bpb_fin=[0-9.]+" "$OUT/${tag}_s${seed}.log" | tail -1)
  echo "    $bpb  wall=$((t1-t0))s"
  # A mechanism that did not run is not measured: check its counters.
  grep -oE "csa_forward=[0-9]+|moe_expert_calls=[0-9]+|depth=[0-9]+|transformer_block=[0-9]+|kda=[a-z]+" "$OUT/${tag}_s${seed}.log" | tail -4 | tr '\n' ' '
  echo
}

for seed in 1 2; do
  run dense  configs/flagship3_dense.toml $seed
  run kda    configs/arch_hybrid_kda.toml $seed
  run moe8   configs/arch_moe8.toml $seed
done

echo "=== summary $(date +%H:%M:%S)"
$PY - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
by = {}
for f in sorted(d.glob("*.jsonl")):
    arm = f.stem.split("_")[0]
    ev = [json.loads(l) for l in open(f) if l.strip()]
    pts = [r for r in ev if r.get("event") == "eval" and "bpb_fin" in r]
    if pts:
        by.setdefault(arm, []).append(pts[-1]["bpb_fin"])
for k, v in sorted(by.items(), key=lambda kv: statistics.mean(kv[1])):
    print(f"{k:6s} mean={statistics.mean(v):.4f} spread={max(v)-min(v):.4f} n={len(v)}")
if "dense" in by:
    ref = statistics.mean(by["dense"])
    for k, v in by.items():
        if k == "dense":
            continue
        m = statistics.mean(v)
        win = (ref - m) > max(2 * (max(v) - min(v)), 0.02)
        print(f"vs dense: {k} {m-ref:+.4f} -> {'WINS' if win else 'no win'}")
PY
