#!/usr/bin/env bash
# TST (bag of bytes) versus the same recipe one byte per position, at equal WALL TIME.
#
# The paper's claim is equal-loss time, not equal-step quality, so the two arms
# must be compared on the clock: the TST arm reads 4x the bytes per step and is
# allowed to take longer per step to pay for it. Each arm gets the same 3000 s of
# training; what we compare is the eval BPB reached inside that budget.
#
# Both arms: 95M gated-SwiGLU recipe, b24 s1024 (so a TST step reads 98 304 bytes),
# one seed, eval every 500 steps on the same held-out window.
#
# Single-source on purpose: `tst_bag` refuses a mix_plan, because MixtureStream's
# resume replay assumes a fixed byte count per step and a bag changes it. The
# comparison does not need the mixture — both arms read the same file.
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=../data_local/pre/fwedu.bin
EVAL=../data_local/pre/fwedu_eval.bin
LOG=/home/sehaxe/logs/tst_ab.log

arm() { # arm <label> <extra set args...>
  local label=$1; shift
  echo "### $label $(date +%H:%M:%S)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_TIMEOUT=7200 bash scripts/gpu_lock.sh \
    .venv/bin/python -u -m dormouse.cli train configs/arch_gated_swiglu.toml \
      --data "$DATA" --eval "$EVAL" \
      --set steps=6000 --set batch=24 --set seq_len=1024 --set seed=11 \
      --set lr=3e-4 --set opt=mix --set muon_lr_mult=12 \
      --set doc_masking=false \
      --set eval=true --set eval_bytes=196608 --set eval_every=500 \
      --set log_every=250 --set use_engram=false \
      --set out_dir="/tmp/opencode/tst_$label" \
      --set ns_dtype=bf16 --set muon_state_dtype=bf16 --set torch_compile=false \
      "$@" 2>&1 | grep -E "^step=|eval step=|done steps|OutOfMemory|loud:" 
  echo "### $label rc=$? $(date +%H:%M:%S)"
}

{
  arm ctl
  arm bag4 --set tst_bag=4 --set tst_frac=0.3
  echo "### TST AB done $(date +%H:%M:%S)"
} > "$LOG" 2>&1
