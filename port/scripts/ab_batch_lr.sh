#!/usr/bin/env bash
# Batch x LR at a FIXED byte budget: which (effective batch, lr) is best at 30 MB?
#
# WHY these six: the ladder ran one shape (8 KB/step, lr 3e-4) for every arm and
# never swept either. At 30 MB the ranking is a SCREENING result — an LR that wins
# at 30 MB does not have to win at 2 GB — so the sweep picks the top two and the
# long run verifies them. Effective batch grows by `grad_accum`, not by `batch`:
# peak memory stays at b8 (the card is 16 GB and b32/s1024 would not fit), while
# the optimizer sees 8/32 KB per step.
#
#   bash scripts/ab_batch_lr.sh                 # all six, ~1 h
#   ONLY="8 1 4e-4" bash scripts/ab_batch_lr.sh # one arm
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=${OUT:-/home/sehaxe/logs/sweep_blr}
BYTES=${BYTES:-30000000}
MIX=${MIX:-configs/mix_refetch.toml}
EVAL=${EVAL:-../data_local/best/fwedu_eval.bin}
DATA=${DATA:-../data_local/best/fwedu_train.bin}
ONLY=${ONLY:-}
mkdir -p "$OUT"

# (batch, grad_accum, lr): 8 KB/step keeps the ladder's shape; 32 KB/step is the
# smallest batch that a 2 GB run would plausibly want.
ARMS=(
  "8 1 2e-4" "8 1 4e-4" "8 1 8e-4"
  "8 4 2e-4" "8 4 4e-4" "8 4 8e-4"
)

for arm in "${ARMS[@]}"; do
  read -r b a lr <<< "$arm"
  tag="b${b}a${a}_lr${lr}"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ]; then continue; fi
  steps=$(( BYTES / (b * a * 1024) ))
  ev=$(( steps / 2 ))
  echo "=== $tag steps=$steps bytes=$(( steps * b * a * 1024 )) $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train configs/flagship3_dense.toml \
    --data "$DATA" --eval "$EVAL" \
    --set "batch=$b" --set "grad_accum=$a" --set "lr=$lr" \
    --set steps="$steps" --set seq_len=1024 --set seed=1 \
    --set opt=mix --set muon_lr_mult=12 \
    --set mix_plan="$MIX" --set doc_masking=false \
    --set eval=true --set eval_every="$ev" --set eval_bytes=196608 \
    --set log_every="$ev" --set ckpt_every=1000000 \
    --set out_dir="/tmp/opencode/sweep_$tag" \
    --set ns_dtype=bf16 --set muon_state_dtype=bf16 \
    --json-log "$OUT/$tag.jsonl" > "$OUT/$tag.log" 2>&1
done

echo "=== ranking $(date +%H:%M:%S)"
$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
rows = []
for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    ev = [json.loads(l) for l in p.read_text().splitlines() if '"eval"' in l]
    lg = [json.loads(l) for l in p.read_text().splitlines() if '"log"' in l]
    if ev:
        kbs = lg[-1].get("tokens_per_s", 0) / 1000 if lg else 0
        rows.append((p.stem, ev[-1]["bpb_fin"], ev[-1]["bytes_seen"] / 1e6, kbs))
base = min((r[1] for r in rows), default=None)
for tag, bpb, mb, kbs in sorted(rows, key=lambda r: r[1]):
    print(f"{tag:16s} bpb={bpb:.4f} at {mb:.0f} MB  {kbs:5.1f} kB/s  vs-best {bpb-base:+.4f}")
PY
