#!/bin/bash
# dormouse guard: restart on device death, resume from ckpt (loss <= 60s)
cd /home/sehaxe/dormouse
while true; do
  ./target/release/train "$@" >> /tmp/dormouse_guard.log 2>&1
  RC=$?
  echo "=== train exited rc=$RC $(date) ===" >> /tmp/dormouse_guard.log
  [ $RC -eq 0 ] && break
  sleep 30
done