#!/usr/bin/env bash
set -u
L=$1; N=$2
D=/home/sehaxe/cache/determinism
R=$D/runs/$L; rm -rf "$R"; mkdir -p "$R"
cd /home/sehaxe/dormouse
exec systemd-run --user --scope -q -p MemoryMax=40G \
  /home/sehaxe/dormouse/target/release/train \
  --data "$D/corpus" --preset small --steps "$N" --seed 7 \
  --batch 2 --seq-len 128 --no-kda --no-engram \
  --ckpt-dir "$R" --ckpt-name m --log-every "$N"
