#!/usr/bin/env bash
# Does `torch.compile` pay at the LIVE shape (dense 95M, b8 x 1024)?
#
# The 2026-11-05 MFU memo measured 35.7 % -> 46.6 % MFU on a 523M spike at
# b16 x 1024 (1.30x) and 273.5 vs 357.3 ms/step at the b1 shape (1.20x). Neither
# is this shape. The live recipe has never run compiled, and compile also spends
# a one-off warmup (inductor cache persists in ~/.cache/torch, so it is once per
# machine, not per run).
#
# The comparison is the LAST logged `tokens_per_s` — after the warmup — not the
# mean, which a 1-2 min compile would dominate. 600 steps at ~0.15 s is ~90 s of
# steady state per arm.
#
#   bash scripts/ab_compile.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=${OUT:-/home/sehaxe/logs/ab_compile}
STEPS=${STEPS:-600}
mkdir -p "$OUT"

run() { # tag extra-args...
  local tag=$1; shift
  echo "=== $tag steps=$STEPS $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train configs/flagship3_dense.toml \
    --data ../data_local/best/fwedu_train.bin \
    --eval ../data_local/best/fwedu_eval.bin \
    --set steps="$STEPS" --set batch=8 --set seq_len=1024 --set seed=1 \
    --set lr=3e-4 --set opt=mix --set muon_lr_mult=12 \
    --set mix_plan=configs/mix_refetch.toml --set doc_masking=false \
    --set eval=false --set log_every=100 --set ckpt_every=1000000 \
    --set out_dir="/tmp/opencode/compile_$tag" \
    --set ns_dtype=bf16 --set muon_state_dtype=bf16 \
    --json-log "$OUT/$tag.jsonl" "$@" > "$OUT/$tag.log" 2>&1
}

run eager
run compiled --set torch_compile=true

echo "=== steady-state (last logged line) ==="
$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    lg = [json.loads(l) for l in p.read_text().splitlines() if '"log"' in l]
    if not lg:
        print(f"{p.stem:10s} no log line"); continue
    last = lg[-1]
    print(f"{p.stem:10s} step {last['step']:4d}  {last['tokens_per_s']/1000:5.1f} kB/s  "
          f"{last['bytes_seen']/1e6:5.1f} MB")
PY
