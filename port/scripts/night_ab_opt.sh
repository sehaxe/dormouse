#!/usr/bin/env bash
# Night A/B: MixOptimizer (Muon+AdamW) vs plain AdamW on the dense form.
# Why: bench_compile measured opt.step at 43% of the step, and AdamW-only runs
# 0.314 s/step against Mix's 0.462 (two-point timing, same shape). Muon has to
# earn its 47% — this A/B is the only measurement that can say so.
#
# 2 arms x 2 seeds x 3000 steps (73.7 MB each), eval every 1000, honest metric.
# Then: whichever wins is used for the long flagship3 run.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/night_opt
mkdir -p "$OUT"

run() { # tag opt steps seed
  local tag=$1 opt=$2 steps=$3 seed=$4
  echo "=== $tag (opt=$opt seed=$seed) $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train flagship3_dense \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=$steps --set batch=8 --set seq_len=1024 --set grad_accum=3 \
    --set opt=$opt --set seed=$seed \
    --set eval=true --set eval_every=1000 \
    --set mix_plan=configs/mix_v7.toml --set log_every=500 \
    --set ckpt_every=3000 --set ckpt_name="ab_${tag}_s${seed}" \
    --set out_dir=/tmp/opencode/ab_${tag}_s${seed} \
    --json-log "$OUT/${tag}_s${seed}.jsonl" > "$OUT/${tag}_s${seed}.log" 2>&1
  grep -oE "eval step=[0-9]+ off=[0-9]+ bpb_fin=[0-9.]+" "$OUT/${tag}_s${seed}.log" | tail -2
}

for seed in 1 2; do
  run mix mix 3000 $seed
  run adam adamw 3000 $seed
done

echo "=== summary $(date +%H:%M:%S)"
$PY - "$OUT" <<'PY'
import json, sys, glob, statistics
from pathlib import Path
d = Path(sys.argv[1])
rows = []
for f in sorted(d.glob("*.jsonl")):
    ev = [json.loads(l) for l in open(f) if l.strip()]
    pts = [r for r in ev if r.get("event") == "eval" and "bpb_fin" in r]
    if not pts:
        continue
    last = pts[-1]
    rows.append((f.stem, last["bpb_fin"], last.get("bpb_mean"), last["step"]))
for r in rows:
    print(f"{r[0]:12s} fin={r[1]:.4f} mean={r[2]} step={r[3]}")
by = {}
for name, fin, _, _ in rows:
    arm = name.split("_")[0]
    by.setdefault(arm, []).append(fin)
for arm, v in by.items():
    print(f"{arm}: mean={statistics.mean(v):.4f} spread={max(v)-min(v):.4f} n={len(v)}")
if "mix" in by and "adam" in by:
    print(f"delta (mix - adam) = {statistics.mean(by['mix']) - statistics.mean(by['adam']):+.4f}")
PY

# Next: the long run with the winner's optimizer is launched by hand after
# reading the summary — a wrong optimizer for 8 hours is worse than a late start.
