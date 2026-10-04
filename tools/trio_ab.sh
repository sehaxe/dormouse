#!/usr/bin/env bash
# THE CONTROLLED TRIO: plain byte Transformer vs byteflow vs dormouse.
# One recipe, equal BYTES, 3 seeds per arm, 2k steps, an 81 920 B eval window.
#
# WHAT THIS IS FOR. The research expert's standing objection to every
# architecture comparison in this repo's archive is that it is a comparison of
# TRAINING RESULTS with too many confounds at once (docs/reviews/
# trio-prep-2026-10-04.md §1). This script removes the confounds it can and
# refuses to run the ones it cannot:
#
#   * ONE optimizer (AdamW), ONE lr, ONE batch x seq, ONE step count, so all
#     three arms consume the SAME bytes: `STEPS * BATCH * SEQ` per arm, printed
#     and asserted here, not assumed. Bytes/step is the unit of the comparison
#     (AGENTS.md §2.6: the eval window is `eval_batches * batch * seq_len`, so
#     a batch-size difference is a different measurement).
#   * PURE CE everywhere: `--jepa-weight 0 --dspark-weight 0` and the presets
#     carry every other aux weight at 0, so no arm pays for a second forward.
#   * fp32 (`--quant fp32`): bf16 has no tensor-core path on this backend
#     (AGENTS.md §2.1), so a bf16 arm would be slower for no reason.
#   * NO ENGRAM, in every arm: the memory arm is the one confound the preset
#     layer cannot switch off for `small` alone, and `--no-engram` is a flag on
#     every arm, so all three run without it. The cost is stated in the log:
#     the tables + key projections + value projection + `mem_dense` are still
#     PARAMETERS in a run that never reads them (plain9m_control.rs measures
#     it), so the "params" column of the summary table below is an UPPER bound
#     on what trains, and the summary prints it as such.
#
# WHAT IT REFUSES TO HIDE. Each arm's own log is checked for the fields that
# decide whether the arm RAN, because a run whose arm silently did not run
# looks perfectly healthy (AGENTS.md §1.1):
#
#   * `plain`   - the dense control's attention branch, on the control's eval
#                 lines. `plain=0` means the eval scored a network whose
#                 attention never entered the body.
#   * `engram`  - must read `0/0` on every arm: the memory arm is off by
#                 construction, so a non-zero first field means the flag did not
#                 land.
#   * `patcher=` - the byteflow arm's own arm line. The byteflow loop has NO
#                 engram/kda seam counters at all, so for that arm the arm line
#                 IS the check.
#   * THE BYTE COUNT - every eval line must print the same window, and it must
#                 equal this script's arithmetic. An arm whose eval scored a
#                 different number of bytes is not comparable at any BPB.
#
# THE BYTEFLOW ARM'S TWO KNOWN GAPS, PRINTED NOT PAPERED OVER (§4 of the
# review): its train CE is the POST-firewall value (a NaN step prints
# `ce=0.000` there and `ce=nan` in the dormouse loop), and it keeps only the
# LAST checkpoint - there is no `<name>.best` artifact, so "best held-out" for
# that arm means "best EVAL LINE", not "best weights on disk". The summary
# therefore reads the eval lines and never a checkpoint file.
#
# COST IS PRICED BEFORE ANYTHING LONG RUNS (§3.3: the price of one A/B arm is
# currently UNKNOWN in this repo - AB-PROTOCOL's 53 min came from a run whose
# attention backward never executed). `preflight` times 20 steps per arm and
# refuses if nine arms do not fit the budget. No per-step constant is
# hardcoded anywhere in this file.
#
# ONE GPU PROCESS AT A TIME (§1.5 - the rule that cost official_v5): every
# launch waits for `nvidia-smi` to report no compute apps, confirmed twice with
# a gap. It will NOT start while the byteflow bar test owns the card.
set -u
cd /home/sehaxe/dormouse

LOGDIR=${LOGDIR:-/home/sehaxe/logs/trio}
CKPT=${CKPT:-/home/sehaxe/trio_ckpt}
DATA=${DATA:-/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_sharded}
EVAL=${EVAL:-/home/sehaxe/eval_2m_v2}
STEPS=${STEPS:-2000}
BATCH=${BATCH:-8}
SEQ=${SEQ:-512}
# 20 x 8 x 512 = 81 920 B. The window is a FUNCTION of the batch size
# (AGENTS.md §2.6), so it is computed here, printed here and checked against
# every eval line - quoting "20 batches" without the bytes is not a number.
EVAL_BATCHES=${EVAL_BATCHES:-20}
WINDOW=$((EVAL_BATCHES * BATCH * SEQ))
# Bytes per step, and bytes per run. THE comparison is fixed on these, not on
# steps: all three arms take the same `batch * seq_len` through the same
# `ByteStream` seam (measured, docs/reviews/byteflow-ab-2026-10-02.md §1), so
# equal steps is equal bytes - and this script prints the product so the claim
# is checkable rather than inherited.
BYTES_PER_STEP=$((BATCH * SEQ))
BYTES_PER_RUN=$((STEPS * BYTES_PER_STEP))
# Wall-clock budget for the WHOLE trio in seconds (nine arms). Refused if
# preflight says it does not fit.
BUDGET=${BUDGET:-86400}
SEEDS=${SEEDS:-"1 2 3"}
# Preflight steps per arm (not per run: one shape, three arms).
PREFLIGHT_STEPS=${PREFLIGHT_STEPS:-20}
mkdir -p "$LOGDIR" "$CKPT"

echo "=== CONTROLLED TRIO: plain9m vs byteflow vs small"
echo "=== one recipe: --opt adamw (one optimizer for all three arms), batch $BATCH x seq $SEQ, $STEPS steps, pure CE, fp32"
echo "=== bytes/step $BYTES_PER_STEP, bytes/run $BYTES_PER_RUN (EQUAL for all three arms by construction)"
echo "=== eval window $WINDOW B ($EVAL_BATCHES batches x $BATCH x $SEQ), seeds: $SEEDS"

die() { echo "=== REFUSING: $*" >&2; exit 1; }

[ -x ./target/release/train ] || die "./target/release/train is missing - cargo build --release -p dormouse-cli first"
[ -d "$DATA" ] || die "DATA=$DATA is not mounted (AGENTS.md §2.6: the corpus drive must be mounted)"
[ -e "$EVAL" ] || die "EVAL=$EVAL does not exist - a held-out number without its file is not a number"
# The arms' configs are the experiment; a missing one would silently change
# which network a column of the summary describes.
for p in plain9m byteflow small; do
  [ -f "configs/$p.toml" ] || die "configs/$p.toml is missing"
done

gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

# A free card, confirmed TWICE with a gap. The desktop's own GL context sits at
# a few hundred MiB, and a run that starts while the previous process is still
# tearing down dies on a 1 GB allocation.
wait_for_gpu() {
  for _ in $(seq 1 720); do
    if [ "$(gpu_mib)" -lt 800 ]; then
      sleep 30
      [ "$(gpu_mib)" -lt 800 ] && return 0
    fi
    sleep 30
  done
  return 1
}

# ONE flag set for all three arms. Everything that could differ between arms
# lives here, so a reader can see there is nothing else.
common_flags() {
  printf '%s\n' \
    --data "$DATA" --eval "$EVAL" \
    --eval-every 200 --eval-batches "$EVAL_BATCHES" \
    --steps "$STEPS" \
    --batch "$BATCH" --seq-len "$SEQ" \
    --jepa-weight 0 --dspark-weight 0 \
    --opt adamw --quant fp32 --no-engram \
    --timers
}

launch() { # arm preset seed
  local arm=$1 preset=$2 seed=$3
  wait_for_gpu || die "the GPU never freed for $arm s$seed (another lane owns the card?)"
  echo "=== $arm s$seed: preset=$preset start $(date +%H:%M)"
  # shellcheck disable=SC2046  # a flag list, deliberately word-split
  systemd-run --user --scope -q -p MemoryMax=40G \
    ./target/release/train \
      $(common_flags) \
      --preset "$preset" --seed "$seed" \
      --ckpt-name "trio_${arm}_s${seed}" --ckpt-dir "$CKPT" \
      --log "$LOGDIR/${arm}_s${seed}.log" >/dev/null 2>&1
}

# --- the honesty gates, per arm ---------------------------------------------

# Every eval line's byte count, deduped. One value = every eval scored the same
# window; more than one = the run drifted and no two of its numbers are
# comparable.
windows_in() { grep -oE 'over [0-9]+ B' "$1" 2>/dev/null | awk '{print $2}' | sort -u; }

check_arm() { # log arm
  local log=$1 arm=$2 ok=0
  [ -f "$log" ] || { echo "=== $arm: NO LOG at $log"; return 1; }

  # 1. The eval window. This arm's own arithmetic, not the script's.
  local seen; seen=$(windows_in "$log" | tr '\n' ' ')
  if [ -z "$seen" ]; then
    echo "=== DIRTY [$arm]: no eval line printed a byte count - nothing to compare"
    return 1
  fi
  local n; n=$(echo "$seen" | wc -w)
  if [ "$n" -ne 1 ] || [ "$seen" != "$WINDOW " ]; then
    echo "=== DIRTY [$arm]: eval windows {seen} != the script's single $WINDOW B - this arm \
scored a different amount of text than the others (AGENTS.md §2.6)"
    return 1
  fi
  echo "=== $arm: eval window $WINDOW B on every eval line"

  # 2. The memory arm is off by construction on all three arms. `engram=0/0` is
  #    the reading; a non-zero row count is the §3.2 retraction's shape (an eval
  #    that scored a memory-enabled network while the config says otherwise).
  local eg; eg=$(grep -oE 'engram=[0-9]+/[0-9]+' "$log" 2>/dev/null | sort -u | tr '\n' ' ')
  case "$arm" in
    byteflow)
      # The byteflow loop prints no seam counters at all; its arm line is the
      # check, so assert the arm line is present instead of inventing a field.
      grep -q 'patcher=' "$log" || {
        echo "=== DIRTY [$arm]: no 'patcher=' arm line - the log does not say which patcher ran"; return 1; }
      ;;
    *)
      if [ -n "$eg" ] && [ "$eg" != "engram=0/0 " ]; then
        echo "=== DIRTY [$arm]: memory arm reported {eg} with --no-engram on every arm"
        return 1
      fi
      ;;
  esac

  # 3. The control's own arm: `plain=` must be non-zero on its eval lines. The
  #    two other arms print `plain=0`, which is CORRECT there (the arm is off)
  #    and is why this check is per-arm rather than global.
  local plain; plain=$(grep -oE 'plain=[0-9]+' "$log" 2>/dev/null | tail -1 | tr -dc '0-9')
  plain=${plain:-0}
  if [ "$arm" = plain9m ] && [ "$plain" -eq 0 ]; then
    echo "=== DIRTY [$arm]: plain=0 on every eval line - the control's dense attention never \
entered the body, so this column is a network without attention"
    return 1
  fi
  echo "=== $arm: plain=$plain, memory arm off as required"

  # 4. Steps actually ran, and the run's last log line is not a refusal.
  local last; last=$(grep -cE '^step +[0-9]+ ' "$log")
  if [ "$last" -lt 2 ]; then
    echo "=== DIRTY [$arm]: $last step lines - the run did not train"
    return 1
  fi
  grep -qiE 'panic|refus|non-finite|device error' "$log" && {
    echo "=== DIRTY [$arm]: the log carries a failure line:"; grep -iE 'panic|refus|non-finite|device error' "$log" | tail -3; return 1; }
  return 0
}

# --- preflight: price the run before spending the wall clock ----------------

preflight() {
  echo "=== preflight: $PREFLIGHT_STEPS steps per arm, to price the trio"
  local per_arm_total=0 a
  for a in plain9m byteflow small; do
    local t0 t1 ms
    t0=$(date +%s)
    # shellcheck disable=SC2046
    systemd-run --user --scope -q -p MemoryMax=40G \
      ./target/release/train \
        $(common_flags) --preset "$a" --seed 99 \
        --steps "$PREFLIGHT_STEPS" --eval-every 0 --ckpt-every 1000000 \
        --ckpt-name "trio_pre_${a}" --ckpt-dir "$CKPT" \
        --log "$LOGDIR/preflight_${a}.log" >/dev/null 2>&1
    t1=$(date +%s)
    # Step 0 pays the cold cubecl autotune and is NOT a measurement of a step
    # (AGENTS.md §3.1: a step-time reading with no step index is not a
    # measurement). So price the run by its ms/step field, not by wall clock.
    ms=$(grep -oE 'ms/step=[0-9]+' "$LOGDIR/preflight_${a}.log" | tail -1 | tr -dc '0-9')
    if [ -z "$ms" ] || [ "$ms" -eq 0 ]; then
      echo "=== preflight $a: no ms/step field - refusing to price the trio from wall clock alone"
      return 1
    fi
    local est=$(( (ms * STEPS + 999) / 1000 ))
    per_arm_total=$((per_arm_total + est))
    echo "=== preflight $a: ${ms} ms/step (step $((PREFLIGHT_STEPS - 1))+) -> ${est}s for $STEPS steps"
  done
  local total=$((per_arm_total * 3))   # three seeds each
  echo "=== preflight total: ${total}s for 9 arms, budget ${BUDGET}s"
  if [ "$total" -gt "$BUDGET" ]; then
    echo "=== REFUSING: the trio prices at ${total}s against a ${BUDGET}s budget. Re-cost with a \
smaller STEPS, a bigger budget, or fewer seeds - do NOT start it and find out."
    return 1
  fi
  return 0
}

# --- summary: read the LOGS, never a checkpoint file ------------------------
#
# The byteflow loop keeps only the last checkpoint (no `.best` artifact, no
# `.bpb` sidecar), so "the best model" means something different per arm and a
# summary that read files would silently compare a best-of-curve against a
# last-step. Every column below therefore comes from a printed line.

# Best held-out BPB over an arm's EVAL lines only. The train step lines print a
# `bpb=` too (the train CE in bits/byte), and reading those would compare train
# loss across arms - which is how a run that OVERFITS (train 2.569 against a
# held-out 5.969, the suspicion the research expert raised) reads as a winner.
# The two arms spell the line differently, so both are matched:
#   dormouse: "step  1200 EVAL ce=... bpb=..."
#   byteflow: "eval step 1200 bpb=... ce=..."
best_eval_bpb() { # log
  grep -E 'EVAL|^eval step' "$1" 2>/dev/null \
    | grep -oE 'bpb=[0-9.]+' | cut -d= -f2 | sort -g | head -1
}

# The parameter count the run printed in its own header (an UPPER bound: with
# --no-engram the memory subtree is still counted, and it never trains - see the
# review and crates/dormouse-core/tests/plain9m_control.rs).
params_of() { # log
  grep -oE 'params=[0-9_]+' "$1" 2>/dev/null | head -1 | cut -d= -f2
}

run_arm() { # arm preset
  local arm=$1 preset=$2 s
  for s in $SEEDS; do
    launch "$arm" "$preset" "$s" || die "$arm s$s did not complete"
    check_arm "$LOGDIR/${arm}_s${s}.log" "$arm" \
      || die "$arm s$s failed its honesty gate - the trio is not a measurement until it passes"
  done
}

case "${1:-all}" in
preflight) preflight ;;
all)
  preflight || die "preflight refused"
  run_arm plain9m plain9m
  run_arm byteflow byteflow
  run_arm small small
  echo
  echo "=== TRIO SUMMARY (best held-out BPB per arm over its eval lines; every arm scored $WINDOW B)"
  printf '%-10s' arm; for s in $SEEDS; do printf ' s%-10s' "$s"; done; printf ' %-14s %s\n' params train-bpb
  for arm in plain9m byteflow small; do
    printf '%-10s' "$arm"
    for s in $SEEDS; do printf ' %-11s' "$(best_eval_bpb "$LOGDIR/${arm}_s${s}.log")"; done
    printf ' %-14s %s\n' "$(params_of "$LOGDIR/${arm}_s${SEEDS%% *}.log")" \
      "$(grep -E '^step +[0-9]+ ' "$LOGDIR/${arm}_s${SEEDS%% *}.log" | tail -1 | grep -oE 'bpb=[0-9.]+' | cut -d= -f2)"
  done
  echo "=== bytes: $BYTES_PER_STEP/step, $BYTES_PER_RUN/run, identical for all three arms"
  echo "=== logs: $LOGDIR"
  ;;
*) echo "usage: $0 [all|preflight]"; exit 2 ;;
esac
