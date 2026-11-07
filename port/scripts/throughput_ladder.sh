#!/usr/bin/env bash
# Throughput ladder — pure SPEED, zero quality risk in the measurement itself.
#
# The measured split at b=8, s=1024 on `flagship3_dense` (median of the JSONL
# logs, 2026-11-08):
#   opt=adamw, grad_accum=3   78.3 kB/s  -> 0.104 s per micro-batch, optimizer ~0
#   opt=mix,   grad_accum=3   50.2 kB/s  -> 0.49  s per step of 3 micro-batches
# so the Muon pass costs ~0.22 s per step while fwd+bwd of one micro-batch
# costs ~0.09 s: the OPTIMIZER is 2.4x the model's own compute.
#
# Muon's NS chain at these shapes is 12 matrices of [832, 3072] in fp32:
#   gram 4.25 + squares 1.15 + px 4.25 GFLOP = 9.65 GFLOP/iter/matrix
#   5 iterations x 12 matrices = 579 GFLOP per step, on the CUDA-core fp32 lane
# That is the whole 0.22 s. The rungs that move it are all ALREADY IMPLEMENTED:
#   ns_dtype=bf16            tensor cores instead of CUDA cores
#   muon_state_dtype=bf16    half the momentum traffic (18% measured before)
#   ns_steps=3               -40% of the arithmetic (published as non-monotone)
#   torch_compile / opt_graph  launch overhead, not arithmetic
#
# This ladder measures the seconds. The BPB cost of each rung is a separate A/B
# (`scripts/muon_precision_ab.sh`) because a precision rung is a math change.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/thru
mkdir -p "$OUT"
STEPS=${STEPS:-40}
D=configs/flagship3_dense.toml

arm() { # tag extra-set-args...
  local tag=$1; shift
  $PY -u -m dormouse.cli train $D \
    --data ../data_local/best/fwedu.bin \
    --set steps=$STEPS --set batch=8 --set seq_len=1024 --set opt=mix \
    --set muon_lr_mult=12 --set lr=3e-4 --set eval=false \
    --set mix_plan=configs/mix_v7.toml --set doc_masking=false \
    --set log_every=10 --set ckpt_every=1000000 --set out_dir=/tmp/opencode/thru_$tag \
    --json-log "$OUT/$tag.jsonl" "$@" > "$OUT/$tag.log" 2>&1
  echo "=== $tag done $(date +%H:%M:%S)"
}

# --- the optimizer's precision rungs (the 2.4x) ----------------------------
arm muon_fp32        # control: ns fp32, state fp32, 5 steps
arm ns_bf16          --set ns_dtype=bf16
arm ns3              --set ns_steps=3
arm ns_bf16_3        --set ns_dtype=bf16 --set ns_steps=3
arm state_bf16       --set muon_state_dtype=bf16
arm muon_fast        --set ns_dtype=bf16 --set ns_steps=3 --set muon_state_dtype=bf16

# --- the launch rungs -------------------------------------------------------
arm compile          --set torch_compile=true
arm graph            --set opt_graph=true
arm muon_fast_graph  --set ns_dtype=bf16 --set ns_steps=3 --set muon_state_dtype=bf16 \
                     --set opt_graph=true

$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
print(f"{'arm':18s} {'kB/s':>8s} {'steps/s':>8s} {'ms/step':>8s}")
for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    logs = [json.loads(l) for l in p.read_text().splitlines() if '"log"' in l]
    if not logs:
        print(f"{p.stem:18s} (no log line)"); continue
    r = logs[-1]
    bps = r["bytes_seen"] / (r["step"] + 1)          # 8192 at b8 s1024 ga1
    rate = r["tokens_per_s"]
    print(f"{p.stem:18s} {rate/1e3:8.1f} {rate/bps:8.2f} {1e3/(rate/bps):8.0f}")
PY
