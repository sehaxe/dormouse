#!/usr/bin/env bash
# Round 1 of the "perfect it before the long run" sweep.
#
# What we know: Muon at mult=12 gives 1.748 (spread 0.007) against AdamW's 1.895
# at 2 000 steps; mult=32 is worse (1.994); mult=1 is much worse. The optimum is
# somewhere between 8 and 16, and the curve near an optimum is shallow — so this
# round is 3000 steps (73.7 MB), two seeds, on the same data and eval window.
#
# Everything else is held fixed on purpose: doc_masking on, mix_v7_docmask,
# lr=3e-4, b8x1024x3. Changing two things at once is how this project lost weeks.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/sweep_mult
mkdir -p "$OUT"

run() { # mult seed
  local mult=$1 seed=$2
  local tag="m${mult}_s${seed}"
  echo "=== $tag $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train flagship3_dense \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=3000 --set batch=8 --set seq_len=1024 --set grad_accum=3 \
    --set opt=mix --set lr=3e-4 --set muon_lr_mult=$mult --set seed=$seed \
    --set eval=true --set eval_every=1500 \
    --set mix_plan=configs/mix_v7_docmask.toml --set doc_masking=true --set log_every=1500 \
    --set ckpt_every=9000 --set ckpt_name="sw_${tag}" \
    --set out_dir=/tmp/opencode/sw_${tag} \
    --json-log "$OUT/${tag}.jsonl" > "$OUT/${tag}.log" 2>&1
  grep -oE "eval step=[0-9]+ off=[0-9]+ bpb_fin=[0-9.]+" "$OUT/${tag}.log" | tail -1
}

for seed in 1 2; do
  for mult in 8 12 16; do
    run $mult $seed
  done
done

echo "=== summary $(date +%H:%M:%S)"
$PY - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
by = {}
for f in sorted(d.glob("*.jsonl")):
    mult = f.stem.split("_")[0]
    ev = [json.loads(l) for l in open(f) if l.strip()]
    pts = [r for r in ev if r.get("event") == "eval" and "bpb_fin" in r]
    if pts:
        by.setdefault(mult, []).append(pts[-1]["bpb_fin"])
for k, v in sorted(by.items(), key=lambda kv: statistics.mean(kv[1])):
    print(f"{k:5s} mean={statistics.mean(v):.4f} spread={max(v)-min(v):.4f} n={len(v)}")
PY
