#!/usr/bin/env bash
# The b1 launch: the first real 1.08B-scale run (port). Every knob here is a
# measured decision, named with its evidence; change one, say why.
#
#   bash port/scripts/run_b1.sh [steps] [batch] [seq_len]
#
# Defaults: 50000 steps b8 s512 = 204.8 MB of bytes, ~8-9 h eager/compiled at
# ~6-7k tok/s — the first-light budget: enough for a real curve at 1B scale,
# cheap enough to stop. WSD's stable phase makes a later extension a legal
# resume (progress keys are exempt), so the cheap launch is the right one.
#
# Evidence for the flags (all in .bulba/memory.md + docs/reviews/):
# - FORM b8 s512: the VRAM ladder (2026-10-17, ~/logs/b1ladder_*.log) measured
#   b8 s768 OOM (256 MB alloc, ~15.2 GiB in use), b4 s768 OK, b8 s512 OK — and
#   b8 s512 carries MORE bytes per step (4096 vs 3072) than b4 s768, so it is
#   both the fit and the throughput choice. dropless (moe_capacity=0 in the
#   preset) is why the earlier "77% dropped" recipe is gone.
# - expandable_segments: warm2 (compile, b8 s768) survived 2 compiled steps WITH
#   it; the same form without it died on a 128 MB alloc (b1drop.log).
# - --tf32 true: +16% at b1 (1399->1201 ms/step, profile 2026-10-16).
# - engram_ram: 402M tables -> host RAM; VRAM frees ~1.5 GB (fits the form).
# - torch_compile: warm2 completed at b8 s768 with the cache persisting in
#   ~/.cache/torch; at b8 s512 the FIRST steps of this run pay a fresh compile
#   (~20-40 min, once) — watch the log; if it dies in compile, relaunch with
#   --set torch_compile=false (eager is the proven fallback).
# - no --optim_8bit: bnb8bit measured +180 ms/step for crumbs of state at this
#   shape (a VRAM lever, not speed); the ladder forms fit without it.
# - factor_qat stays "none": the compression tournament closed the arm for
#   TRAINING at 9.2M (+0.040 BPB annealed); QAT belongs to the post-base
#   finetune, not the base run.
# - MemoryMax=40G: AGENTS 1.5 — every launch goes under the cgroup.
set -euo pipefail
STEPS="${1:-50000}"
BATCH="${2:-8}"
SEQ="${3:-512}"
NAME="b1first"

cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
systemd-run --user --scope -p MemoryMax=40G -- \
  uv run --extra gpu dormouse train b1 \
  --data /home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin \
  --eval /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_eval/eval_tail.bin \
  --device cuda \
  --set steps="$STEPS" \
  --set batch="$BATCH" \
  --set seq_len="$SEQ" \
  --set log_every=100 \
  --set eval=true \
  --set eval_every=1000 \
  --set eval_batches=8 \
  --set lr=3e-4 \
  --set tf32=true \
  --set torch_compile=true \
  --set engram_ram=true \
  --set ckpt_every=5000 \
  --set out_dir=checkpoints/b1run \
  --set ckpt_name="$NAME" \
  --json-log "/home/sehaxe/logs/${NAME}.jsonl" \
  2>&1 | tee "/home/sehaxe/logs/${NAME}.log"
