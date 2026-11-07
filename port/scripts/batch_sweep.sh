#!/usr/bin/env bash
# Batch sweep on the fast Muon rung — where the MODEL step still loses.
#
# After the NS fix, the step splits as 104 ms model (fwd+bwd) + 32 ms optimizer at
# b=8, s=1024. The model part runs at 54 TFLOP/s, i.e. ~62% of the card's bf16
# peak, and the remaining loss at this shape is the small batch: 8192 positions
# per step is a thin GEMM for 832-wide layers.
#
# This measures whether a larger batch converts that into bytes/s. It is a pure
# throughput sweep: the loss curve is a separate question (a bigger batch changes
# the gradient noise and the LR that goes with it).
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/batch
mkdir -p "$OUT"
STEPS=${STEPS:-40}

arm() { # batch extra...
  local b=$1; shift
  local tag="b$b"
  $PY -u -m dormouse.cli train configs/flagship3_dense.toml \
    --data ../data_local/best/fwedu.bin --set steps=$STEPS --set batch=$b \
    --set seq_len=1024 --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
    --set eval=false --set mix_plan=configs/mix_v7.toml --set doc_masking=false \
    --set ns_dtype=bf16 --set ns_steps=3 --set muon_state_dtype=bf16 \
    --set log_every=10 --set ckpt_every=1000000 --set out_dir=/tmp/opencode/batch_$tag \
    --json-log "$OUT/$tag.jsonl" "$@" > "$OUT/$tag.log" 2>&1
  echo "=== batch $b done $(date +%H:%M:%S)"
}

for b in 8 16 32 48 64; do arm $b; done

$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
print(f"{'batch':>6s} {'kB/s':>8s} {'steps/s':>8s} {'ms/step':>8s} {'vs b8':>7s}")
rows = []
for p in sorted(Path(sys.argv[1]).glob("*.jsonl"), key=lambda q: int(q.stem[1:])):
    logs = [json.loads(l) for l in p.read_text().splitlines() if '"log"' in l]
    if not logs:
        print(f"{p.stem:>6s} (no log — likely OOM)"); continue
    r = logs[-1]
    bps = r["bytes_seen"] / (r["step"] + 1)
    rows.append((r["tokens_per_s"], p.stem, bps))
base = next((rate for rate, tag, _ in rows if tag == "b8"), None)
for rate, tag, bps in rows:
    steps_s = rate / bps
    rel = f"{rate/base:6.2f}x" if base else ""
    print(f"{tag[1:]:>6s} {rate/1e3:8.1f} {steps_s:8.2f} {1e3/steps_s:8.0f} {rel:>7s}")
PY
