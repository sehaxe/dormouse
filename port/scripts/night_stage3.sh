#!/usr/bin/env bash
# Stage 3 — the Muon question, asked properly.
#
# History: the first A/B said AdamW beats Mix by 0.2478 BPB. The muon-audit then
# found that the schedule wrote AdamW's LR into the Muon groups on every step,
# so `muon_lr_mult` was a dead knob and Muon always ran at AdamW's LR. A
# reference-parity test puts the update within 2.1x of Keller Jordan's Muon, so
# the rule is right and the LR is simply not Muon's (the original uses ~0.02).
#
# Arms (2000 steps = 49.2 MB, 2 seeds each, eval every 1000):
#   A  adamw,      lr 3e-4                      (the current winner)
#   B  mix,        mult 12                      (the CPU control's best)
#   C  mix,        mult 32                      (sqrt(832)~29 parity ballpark)
# Winner runs the long flagship3; a tie (>-0.043) stays with AdamW, which is 47%
# cheaper per step.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
LOG=/home/sehaxe/logs
OUT=$LOG/night_opt3
mkdir -p "$OUT"

run() { # tag opt lr mult seed
  local tag=$1 opt=$2 lr=$3 mult=$4 seed=$5
  echo "=== $tag seed=$seed $(date +%H:%M:%S)"
  local extra=()
  [ "$opt" = "mix" ] && extra=(--set muon_lr_mult=$mult)
  $PY -u -m dormouse.cli train flagship3_dense \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=2000 --set batch=8 --set seq_len=1024 --set grad_accum=3 \
    --set opt=$opt --set lr=$lr --set seed=$seed "${extra[@]}" \
    --set eval=true --set eval_every=1000 \
    --set mix_plan=configs/mix_v7.toml --set log_every=1000 \
    --set ckpt_every=5000 --set ckpt_name="ab3_${tag}_s${seed}" \
    --set out_dir=/tmp/opencode/ab3_${tag}_s${seed} \
    --json-log "$OUT/${tag}_s${seed}.jsonl" > "$OUT/${tag}_s${seed}.log" 2>&1
  grep -oE "eval step=[0-9]+ off=[0-9]+ bpb_fin=[0-9.]+" "$OUT/${tag}_s${seed}.log" | tail -1
}

for seed in 1 2; do
  run adam adamw 3e-4 1 $seed
  run m12  mix   3e-4 12 $seed
  run m32  mix   3e-4 32 $seed
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
for arm, v in sorted(by.items()):
    print(f"{arm:5s} mean={statistics.mean(v):.4f} spread={max(v)-min(v):.4f} n={len(v)}")
PY

best=$($PY - "$OUT" <<'PY'
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
if not by:
    print("adam 1")
else:
    m = {k: statistics.mean(v) for k, v in by.items()}
    win = min(m, key=m.get)
    print(f"{win} {1 if win == 'adam' else (12 if win == 'm12' else 32)}")
PY
)
set -- $best
arm=$1; mult=$2
echo "[chain] winner arm=$arm mult=$mult"
opt=adamw; extra=()
if [ "$arm" != "adam" ]; then opt=mix; extra=(--set muon_lr_mult=$mult); fi

$PY -u -m dormouse.cli train flagship3_dense \
  --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
  --set steps=${LONG_STEPS:-40000} --set batch=8 --set seq_len=1024 --set grad_accum=3 \
  --set opt=$opt --set lr=3e-4 --set seed=1 "${extra[@]}" \
  --set eval=true --set eval_every=2000 \
  --set mix_plan=configs/mix_v7_docmask.toml --set doc_masking=true --set log_every=500 \
  --set ckpt_every=5000 --set ckpt_name=flagship3 \
  --set out_dir=/tmp/opencode/flagship3 \
  --json-log "$LOG/flagship3.jsonl" > "$LOG/flagship3.log" 2>&1
echo "[chain] flagship3 done $(date +%H:%M:%S)"
grep -oE "eval step=[0-9]+ off=[0-9]+ bpb_fin=[0-9.]+" "$LOG/flagship3.log" | tail -3
