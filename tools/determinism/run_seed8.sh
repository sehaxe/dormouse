#!/usr/bin/env bash
# One zero-step trainer run into its own ckpt dir. $1 = run label.
set -u
L=$1
D=/home/sehaxe/cache/determinism
R=$D/runs/$L
rm -rf "$R"; mkdir -p "$R"
cd /home/sehaxe/dormouse
exec systemd-run --user --scope -q -p MemoryMax=40G \
  /home/sehaxe/dormouse/target/release/train \
  --data "$D/corpus" --preset small --steps 0 --seed 8 \
  --batch 2 --seq-len 128 --no-kda --no-engram \
  --ckpt-dir "$R" --ckpt-name m --log-every 1
