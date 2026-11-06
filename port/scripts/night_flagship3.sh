#!/usr/bin/env bash
# Overnight chain: wait for the Mix-vs-AdamW A/B, pick the winner, run flagship3
# long enough to move the curve, then measure capability on the final checkpoint.
#
# Rules baked in:
#   * one heavy process on the card at a time (this script waits, it does not
#     share the GPU);
#   * the optimizer is chosen by the A/B's own numbers, not by preference;
#   * a tie (|delta| < the 0.043 BPB noise floor) goes to AdamW, because it is
#     47% cheaper per step and a tie on quality at equal bytes means the cheaper
#     step buys more bytes in the same wall clock.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
LOG=/home/sehaxe/logs
OUT=/home/sehaxe/logs/night_opt
LONG_STEPS=${LONG_STEPS:-40000}

ab_pid=$(pgrep -f "night_ab_opt.sh" | head -1 || true)
if [ -n "${ab_pid:-}" ]; then
  echo "[chain] waiting for A/B pid $ab_pid $(date +%H:%M:%S)"
  while kill -0 "$ab_pid" 2>/dev/null; do sleep 60; done
fi
echo "[chain] A/B finished $(date +%H:%M:%S)"
tail -12 "$LOG/night_ab_opt.log"

pick=$($PY - "$OUT" <<'PY'
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
    print("mix")
    raise SystemExit
m = {k: statistics.mean(v) for k, v in by.items()}
delta = m.get("mix", 9) - m.get("adam", 9)
print("adam" if abs(delta) < 0.043 else ("mix" if delta < 0 else "adam"))
PY
)
echo "[chain] optimizer for the long run: $pick"

$PY -u -m dormouse.cli train flagship3_dense \
  --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
  --set steps=$LONG_STEPS --set batch=8 --set seq_len=1024 --set grad_accum=3 \
  --set opt=$pick --set seed=1 \
  --set eval=true --set eval_every=2000 \
  --set mix_plan=configs/mix_v7.toml --set log_every=500 \
  --set ckpt_every=5000 --set ckpt_name=flagship3 \
  --set out_dir=/tmp/opencode/flagship3 \
  --json-log "$LOG/flagship3.jsonl" > "$LOG/flagship3.log" 2>&1
echo "[chain] flagship3 done $(date +%H:%M:%S)"
grep -oE "eval step=[0-9]+ off=[0-9]+ bpb_fin=[0-9.]+" "$LOG/flagship3.log" | tail -3

# Capability: the same harness that measured 0% format on the old flagship.
# It loads the checkpoint through serve, so it also proves the dense form loads
# outside the trainer.
$PY -u scripts/eval_agent.py --checkpoint /tmp/opencode/flagship3/flagship3.pt \
  --tasks 20 --json-out "$LOG/flagship3_capability.json" > "$LOG/flagship3_capability.log" 2>&1
echo "[chain] capability done $(date +%H:%M:%S)"
tail -6 "$LOG/flagship3_capability.log"
