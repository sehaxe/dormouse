#!/usr/bin/env bash
# The night of 2026-11-13: the long TSCT run.
#
# The one question the rank ladder left open. At 120 MB, rank 256 cost +5.2%
# BPB against a dense FFN (1.7268 vs 1.6406 on the 275M shape) for 1.62x speed.
# The control here is the live 95M recipe at the full 2.5 GB corpus: 1.1871 at
# 107,421 steps, b24, s1024, compile, Muon 12, measured 2026-11-12.
#
# Does a bigger data budget make that price smaller (the model can afford to
# spend capacity on fewer, better directions) or larger (rank is a ceiling and
# more data pushes into it)? One arm, same steps, same batch, same recipe.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=/tmp/opencode/run2gb_tsct_r256
LOG=/home/sehaxe/logs/run_tsct_r256.log
JSON=/home/sehaxe/logs/arch/run_tsct_r256_s1.jsonl

mkdir -p "$OUT" "$(dirname "$JSON")"
{
  echo "### TSCT r256, full corpus, 107421 steps $(date +%H:%M:%S)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  GPU_TIMEOUT=43200 bash scripts/gpu_lock.sh \
    .venv/bin/python -u -m dormouse.cli train configs/arch_tsct95_r256.toml \
      --data ../data_local/best/fwedu_train.bin \
      --eval ../data_local/best/fwedu_eval.bin \
      --set steps=107421 --set batch=24 --set seq_len=1024 --set seed=1 \
      --set lr=3e-4 --set opt=mix --set muon_lr_mult=12 \
      --set mix_plan=configs/mix_refetch.toml --set doc_masking=false \
      --set eval=true --set eval_bytes=196608 --set eval_every=13000 \
      --set log_every=13000 --set ckpt_every=10000 \
      --set out_dir="$OUT" \
      --set ns_dtype=bf16 --set muon_state_dtype=bf16 --set torch_compile=true \
      --json-log "$JSON"
  echo "### TSCT run rc=$? $(date +%H:%M:%S)"
} > "$LOG" 2>&1
