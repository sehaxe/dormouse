#!/bin/bash
# Restart real15 training with proper env (libopenblas fix).
export LD_LIBRARY_PATH=/home/sehaxe/dormouse/target/release/build/openblas-src-805ad3aba34d46b8/out/OpenBLAS-0.3.32
export CUBECL_AUTOTUNE_LEVEL=minimal
export DM_MAX_ITER=4
export BF16=1
export DM_QUANT=fp32
export DM_ACT_QUANT=4
export DM_ACT_GROUP=128
export DM_ENGRAM_RAM=1
export DM_ENGRAM_SLOTS=4000000
export DM_TIMERS=1
cd /home/sehaxe/dormouse
while true; do
  ./target/release/train --data /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/mix/math --preset small --ckpt-name real15 --ckpt-dir /home/sehaxe/dormouse/checkpoints --steps 30000 --seq-len 512 --batch 12 --log-every 100 --lr 0.0003 --eval /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/mix/web --eval-every 500 --ckpt-every 1000 >> /tmp/opencode/real15.log 2>&1
  RC=$?
  echo "=== train exited rc=$RC $(date) ===" >> /tmp/opencode/dmon.log
  [ $RC -eq 0 ] && break
  sleep 30
done
