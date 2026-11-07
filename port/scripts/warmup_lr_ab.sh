#!/usr/bin/env bash
# Why is a higher LR worse? Every run on record has `warmup_tokens=0`.
#
# The one LR sweep that exists (`logs/night_opt/`) says 6e-4 is 0.27 BPB WORSE
# than 3e-4 at the same 49 MB (2.1457 vs 1.8822), which is backwards from every
# published recipe. All of those arms started at full LR on step 0 — there is no
# warmup anywhere in this repo, and neither QK-norm, z-loss nor a logit softcap
# is on (the dense config leaves all three at 0.0). A high LR with no warmup is
# the textbook way to land in a worse basin, so the sweep as run cannot separate
# "6e-4 is bad" from "no warmup is bad".
#
# This ladder separates them at a fixed budget (2000 steps = 16.4 MB, ~5 min an
# arm on the fast Muon rung). Pre-registered reading:
#   * if warmup fixes 6e-4 (arm 3 <= arm 1), the LR ceiling was an artefact and
#     the next run should spend a real budget on 6e-4..1e-3 with warmup;
#   * if 6e-4 stays worse with warmup, 3e-4 is the real optimum at 95M and the
#     higher-LR story dies here.
#
# `warmup_tokens` is a TOKEN count: at b=8, s=1024 one step is 8192 bytes, so a
# 2% warmup of a 2000-step run is 40 steps = 327680 bytes.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/warmup
mkdir -p "$OUT"
STEPS=${STEPS:-2000}

run() { # tag extra...
  local tag=$1; shift
  echo "=== $tag $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train configs/flagship3_dense.toml \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=$STEPS --set batch=8 --set seq_len=1024 --set seed=1 \
    --set opt=mix --set muon_lr_mult=12 \
    --set mix_plan=configs/mix_v7.toml --set doc_masking=false \
    --set ns_dtype=bf16 --set muon_state_dtype=bf16 \
    --set eval=true --set eval_every=1000 --set log_every=1000 \
    --set ckpt_every=1000000 --set out_dir=/tmp/opencode/warmup_$tag \
    --json-log "$OUT/$tag.jsonl" "$@" > "$OUT/$tag.log" 2>&1
}

W=$(( STEPS * 8192 * 2 / 100 ))   # 2% of the run, in tokens

run lr3_no_warm   --set lr=3e-4
run lr3_warm      --set lr=3e-4 --set warmup_tokens=$W
run lr6_no_warm   --set lr=6e-4
run lr6_warm      --set lr=6e-4 --set warmup_tokens=$W
run lr1e-3_warm   --set lr=1e-3 --set warmup_tokens=$W
run lr1e-3_stable --set lr=1e-3 --set warmup_tokens=$W \
                  --set z_loss=1e-4 --set logit_softcap=15.0

$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
rows = []
for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    ev = [json.loads(l) for l in p.read_text().splitlines() if '"eval"' in l]
    if ev:
        rows.append((ev[-1]["bpb_fin"], p.stem, ev[-1]["bytes_seen"] / 1e6))
for bpb, tag, mb in sorted(rows):
    print(f"{tag:16s} bpb={bpb:.4f} at {mb:.1f} MB")
PY
