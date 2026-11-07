#!/usr/bin/env bash
# Overnight chain, stage 2.
#
# Stage A (already running): Mix vs AdamW. AdamW is ahead by 0.23 BPB at 2 000
# steps — that is 5x the noise floor, so Muon's 47% step cost is buying a worse
# model at this scale. The chain waits for that A/B to finish.
#
# Stage B: learning-rate sweep for AdamW. The 2026-11-06 control used 6e-4 with
# cosine; this trainer defaults to 3e-4. Two seeds per LR, 2 000 steps each, so a
# 0.043 BPB difference is resolvable.
#
# Stage C: the long run with the winner of B, then the capability harness.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
LOG=/home/sehaxe/logs
OUT=$LOG/night_opt

wait_for() { # pid file-name label
  local pid=$1 label=$2
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    echo "[chain] waiting for $label pid $pid $(date +%H:%M:%S)"
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  fi
  echo "[chain] $label finished $(date +%H:%M:%S)"
}

ab_pid=$(pgrep -f "night_ab_opt.sh" | head -1 || true)
wait_for "${ab_pid:-}" "A/B mix-vs-adam"

echo "[chain] ---- A/B summary ----"
tail -12 "$LOG/night_ab_opt.log"

run_lr() { # lr seed steps
  local lr=$1 seed=$2 steps=$3 tag
  tag="lr${lr}_s${seed}"
  echo "[chain] $tag $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train flagship3_dense \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=$steps --set batch=8 --set seq_len=1024 --set grad_accum=3 \
    --set opt=adamw --set lr=$lr --set seed=$seed \
    --set eval=true --set eval_every=1000 \
    --set mix_plan=configs/mix_v7.toml --set log_every=1000 \
    --set ckpt_every=5000 --set ckpt_name="ab_${tag}" \
    --set out_dir=/tmp/opencode/ab_${tag} \
    --json-log "$OUT/${tag}.jsonl" > "$OUT/${tag}.log" 2>&1
  grep -oE "eval step=[0-9]+ off=[0-9]+ bpb_fin=[0-9.]+" "$OUT/${tag}.log" | tail -1
}

for seed in 1 2; do
  run_lr 3e-4 $seed 2000
  run_lr 6e-4 $seed 2000
done

echo "[chain] ---- LR summary ----"
$PY - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
by = {}
for f in sorted(d.glob("lr*.jsonl")):
    lr = f.stem.split("_")[0][2:]
    ev = [json.loads(l) for l in open(f) if l.strip()]
    pts = [r for r in ev if r.get("event") == "eval" and "bpb_fin" in r]
    if pts:
        by.setdefault(lr, []).append(pts[-1]["bpb_fin"])
for lr, v in sorted(by.items()):
    print(f"lr={lr}: mean={statistics.mean(v):.4f} spread={max(v)-min(v):.4f} n={len(v)}")
PY

best=$($PY - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
by = {}
for f in sorted(d.glob("lr*.jsonl")):
    lr = f.stem.split("_")[0][2:]
    ev = [json.loads(l) for l in open(f) if l.strip()]
    pts = [r for r in ev if r.get("event") == "eval" and "bpb_fin" in r]
    if pts:
        by.setdefault(lr, []).append(pts[-1]["bpb_fin"])
print(min(by, key=lambda k: statistics.mean(by[k])) if by else "3e-4")
PY
)
echo "[chain] long-run lr = $best"

$PY -u -m dormouse.cli train flagship3_dense \
  --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
  --set steps=${LONG_STEPS:-40000} --set batch=8 --set seq_len=1024 --set grad_accum=3 \
  --set opt=adamw --set lr=$best --set seed=1 \
  --set eval=true --set eval_every=2000 \
  --set mix_plan=configs/mix_v7_docmask.toml --set doc_masking=true --set log_every=500 \
  --set ckpt_every=5000 --set ckpt_name=flagship3 \
  --set out_dir=/tmp/opencode/flagship3 \
  --json-log "$LOG/flagship3.jsonl" > "$LOG/flagship3.log" 2>&1
echo "[chain] flagship3 done $(date +%H:%M:%S)"
grep -oE "eval step=[0-9]+ off=[0-9]+ bpb_fin=[0-9.]+" "$LOG/flagship3.log" | tail -3

$PY -u scripts/eval_agent.py --checkpoint /tmp/opencode/flagship3/flagship3.pt \
  --tasks 20 --json-out "$LOG/flagship3_capability.json" > "$LOG/flagship3_capability.log" 2>&1
echo "[chain] capability done $(date +%H:%M:%S)"
tail -6 "$LOG/flagship3_capability.log"
