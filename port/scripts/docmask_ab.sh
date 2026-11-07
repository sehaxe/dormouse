#!/usr/bin/env bash
# How much is document masking worth? — the confound behind E1's 0.49 BPB gap.
#
# E1 v1 (doc_masking=true, mix_v7_docmask, fp32 NS) measured bpb 1.732 at 62.5 MB.
# E1 v3 (doc_masking=false, mix_v7, bf16 NS) measured 2.191 at the same step.
# Two variables moved at once, so neither number can be attributed. The rung is
# isolated by `scripts/muon_quality_ab.sh`; THIS ladder isolates the masking,
# on one rung and one budget.
#
#   masked    mix_v7_docmask + doc_masking=true   (the record runs' setting)
#   unmasked  mix_v7         + doc_masking=false  (the only setting the patch
#                                                  arm accepts today)
#
# The two plans differ by the three sources whose boundaries cannot be recovered
# (wiki_en 4%, reasoning 2%, rudialog 1.5% — 3.2% of the corpus), renormalized;
# that is the closest fair comparison the builder supports.
#
# Why it matters beyond E1: if masking is worth >0.05 BPB, then a patch unit
# that cannot mask is structurally handicapped, and shipping it would require
# document-aware patch boundaries in the codec — a real item, not a config flag.
set -u
cd /home/sehaxe/dormouse/port
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
OUT=/home/sehaxe/logs/docmask
mkdir -p "$OUT"
STEPS=${STEPS:-2000}

run() { # tag plan docmask
  local tag=$1 plan=$2 mask=$3
  echo "=== $tag $plan doc_masking=$mask $(date +%H:%M:%S)"
  $PY -u -m dormouse.cli train configs/flagship3_dense.toml \
    --data ../data_local/best/fwedu.bin --eval ../data_local/mix4/web_eval.bin \
    --set steps=$STEPS --set batch=8 --set seq_len=1024 --set seed=1 \
    --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
    --set mix_plan="$plan" --set doc_masking="$mask" \
    --set eval=true --set eval_every=1000 --set log_every=1000 \
    --set ckpt_every=1000000 --set out_dir=/tmp/opencode/docmask_$tag \
    --json-log "$OUT/$tag.jsonl" > "$OUT/$tag.log" 2>&1
}

run masked   configs/mix_v7_docmask.toml true
run unmasked configs/mix_v7.toml         false

$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    ev = [json.loads(l) for l in p.read_text().splitlines() if '"eval"' in l]
    if ev:
        print(f"{p.stem:9s} " + "  ".join(f"{r['bytes_seen']/1e6:.0f}MB:{r['bpb_fin']:.4f}" for r in ev))
PY
