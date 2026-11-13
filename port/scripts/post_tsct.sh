#!/usr/bin/env bash
# After the long TSCT run: the long-context smoke, then SFT on that checkpoint.
#
# Why this order: the smoke needs the card idle for a few minutes and answers a
# question already promised (ms/step and peak GiB at 16k/32k on the segmented
# path). The SFT then turns the pretrained checkpoint into something that
# answers, which is the point of the whole project.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=/home/sehaxe/logs/post_tsct.log
CKPT=/tmp/opencode/run2gb_tsct_r256/port.pt
SFT_OUT=/tmp/opencode/sft_tsct_r256

{
  echo "### waiting for the TSCT run to finish $(date +%H:%M:%S)"
  while pgrep -f "dormouse.cli trai[n]" > /dev/null; do sleep 60; done
  echo "### card free $(date +%H:%M:%S)"

  if [ ! -f "$CKPT" ]; then
    echo "### no checkpoint at $CKPT — stopping"
    exit 1
  fi

  echo "### long-context smoke (16k / 32k) $(date +%H:%M:%S)"
  GPU_TIMEOUT=2400 bash scripts/gpu_lock.sh \
    .venv/bin/python -u scripts/smoke_long.py 2>&1 | tail -40

  echo "### SFT on the TSCT checkpoint $(date +%H:%M:%S)"
  rm -rf "$SFT_OUT"; mkdir -p "$SFT_OUT"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_TIMEOUT=7200 \
  bash scripts/gpu_lock.sh \
    .venv/bin/python -u -m dormouse.cli sft configs/arch_tsct95_r256.toml \
      --build-sft --data "$SFT_OUT/corpus.bin" \
      --chat-data data_local/chat/ru_sft_mix.jsonl \
      --chat-data ../data_local/chat/local-voice.qa \
      --init-from "$CKPT" \
      --ckpt-name sft_tsct --out-dir "$SFT_OUT" \
      --steps 4000 --batch 8 --lr 1e-4 --holdout 300 2>&1 | tail -25

  echo "### done $(date +%H:%M:%S)"
} > "$LOG" 2>&1
