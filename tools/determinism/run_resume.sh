#!/usr/bin/env bash
set -u
D=/home/sehaxe/cache/determinism
cd /home/sehaxe/dormouse
exec systemd-run --user --scope -q -p MemoryMax=40G \
  /home/sehaxe/dormouse/target/release/train \
  --data "$D/corpus" --preset small --steps 0 --seed 7 \
  --batch 2 --seq-len 128 --no-kda --no-engram \
  --ckpt-dir "$D/runs/A" --ckpt-name m --log-every 1
