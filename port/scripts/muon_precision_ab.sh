#!/usr/bin/env bash
# The BPB cost of the Muon precision rungs — the A/B the speed ladder cannot do.
#
# `scripts/throughput_ladder.sh` measures seconds. This measures what the seconds
# buy/lose: the same bytes seen, two arms, two seeds, the SAME eval window as
# every number on record (`mix4/web_eval.bin`, off=0, 196 608 B scored).
#
#   control  ns_dtype=fp32, muon_state_dtype=fp32, ns_steps=5   (every number on record)
#   fast     ns_dtype=bf16, muon_state_dtype=bf16, ns_steps=3   (tensor cores, -40% NS math)
#
# Pre-registered rule: the fast arm ships ONLY if its mean BPB is within +0.02 of
# the control. NS is a direction, not a number, so a small precision cost is
# plausible; a large one means the 3-step chain is not converging at our shapes
# and ns_steps goes back to 5 (a separate, cheaper decision than the dtype).
#
# bf16 NS is reference practice (nanogpt casts `X = G.bfloat16()`), and the
# 2026-10-16 cost note prices the chain's precision rungs at -100..-200 ms; the
# 3-step arm's quality literature is NON-MONOTONE (Jordan-4 3.2809 < Jordan-5
# 3.2822, 2606.00371 §3.2), so a tie is the prediction, not a hope.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/muonab
mkdir -p "$OUT"
BYTES=${BYTES:-120000000}
STEPS=$(( BYTES / 8192 ))
EVAL_EVERY=$(( STEPS / 4 ))

run() { # tag seed extra...
  local tag=$1 seed=$2; shift 2
  echo "=== $tag seed=$seed steps=$STEPS $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train configs/flagship3_dense.toml \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=$STEPS --set batch=8 --set seq_len=1024 --set seed=$seed \
    --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
    --set mix_plan=configs/mix_v7.toml --set doc_masking=false \
    --set eval=true --set eval_every=$EVAL_EVERY --set log_every=$EVAL_EVERY \
    --set ckpt_every=1000000 --set out_dir=/tmp/opencode/muonab_${tag}_s$seed \
    --json-log "$OUT/${tag}_s$seed.jsonl" "$@" > "$OUT/${tag}_s$seed.log" 2>&1
}

for seed in 1 2; do
  run control $seed
  run fast    $seed --set ns_dtype=bf16 --set muon_state_dtype=bf16 --set ns_steps=3
done

$PY - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path
out = Path(sys.argv[1])
for tag in ("control", "fast"):
    vals, rates = [], []
    for seed in (1, 2):
        p = out / f"{tag}_s{seed}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        ev = [r for r in rows if r.get("event") == "eval"]
        lg = [r for r in rows if r.get("event") == "log"]
        if ev:
            vals.append(ev[-1]["bpb_fin"])
        if lg:
            rates.append(statistics.median([r["tokens_per_s"] for r in lg]))
        print(f"{tag} s{seed}: bpb_fin={ev[-1]['bpb_fin']:.4f} at {ev[-1]['bytes_seen']/1e6:.0f} MB"
              if ev else f"{tag} s{seed}: no eval")
    if vals:
        m = statistics.mean(vals)
        r = statistics.mean(rates) / 1e3 if rates else float('nan')
        print(f"{tag}: mean BPB {m:.4f}  n={len(vals)}  median {r:.1f} kB/s")
PY
