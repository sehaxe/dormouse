#!/usr/bin/env bash
# The b1 launch: the first real 1.08B run (port). Every knob here is a
# measured decision, named with its evidence; change one, say why.
#
#   bash port/scripts/run_b1.sh [steps] [batch] [seq_len]
#
# Defaults: 20000 steps b8 s768 (~2.3B bytes) — the first-run budget, chosen
# so WSD's stable phase can be cut short without waste.
#
# Evidence for the flags (all in .bulba/memory.md + docs/reviews/):
# - --tf32 true: +16% at b1 (1399->1201 ms/step, profile 2026-10-16).
# - engram_ram: 402M tables -> host RAM; VRAM frees ~1.5 GB (fits bigger forms).
# - torch_compile: spike's 1.20x at the b1 shape; first warmup is minutes,
#   cached in ~/.cache/torch afterwards.
# - moe_lb_coef 0.01 + cap 2.0: LFB measured (drops 17.6k->12.4k at cap 64);
#   cap 2.0 = the zero-drop setting from the capacity sweep (2026-10-16).
# - factor_qat stays "none": the compression tournament closed the arm for
#   TRAINING at 9.2M (+0.040 BPB annealed); QAT belongs to the post-base
#   finetune, not the base run.
# - MemoryMax=40G: AGENTS 1.5 — every launch goes under the cgroup.
set -euo pipefail
STEPS="${1:-20000}"
BATCH="${2:-8}"
SEQ="${3:-768}"
NAME="b1base"

cd "$(dirname "$0")/.."
systemd-run --user --scope -p MemoryMax=40G -- \
  uv run --extra gpu dormouse train b1 \
  --data /home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin \
  --eval /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_eval/eval_tail.bin \
  --device cuda \
  --set steps="$STEPS" \
  --set batch="$BATCH" \
  --set seq_len="$SEQ" \
  --set log_every=50 \
  --set eval=true \
  --set eval_every=500 \
  --set eval_batches=8 \
  --set lr=3e-4 \
  --set tf32=true \
  --set torch_compile=true \
  --set engram_ram=true \
  --set out_dir=checkpoints/b1base \
  --set ckpt_name="$NAME" \
  --json-log "/home/sehaxe/logs/${NAME}.jsonl" \
  2>&1 | tee "/home/sehaxe/logs/${NAME}.log"
