#!/usr/bin/env bash
# Baseline quality matrix probe for vendor/dormouse-fused (READ-ONLY).
# Usage: tools/fused_matrix.sh <phase> [crate ...]
#   phases: check | test | cuda-check | cuda-test | examples | static
# Results are TSV on stdout: crate<TAB>status<TAB>detail
# Every cargo invocation is serial (one lock on the target dir) and timeboxed.
set -uo pipefail

WS=/home/sehaxe/dormouse/vendor/dormouse-fused
cd "$WS" || exit 1

PHASE="${1:?phase required}"
shift || true

if [ "$#" -gt 0 ]; then
  CRATES=("$@")
else
  mapfile -t CRATES < <(
    sed -n '/^members/,/^]/p' Cargo.toml | grep -oE '"[^"]+"' | tr -d '"' | grep -v '^benches' | grep -v '^dormouse-fused$'
  )
fi

# dormouse-kda, dormouse-gdn2, dormouse-engram are being edited by other agents right now.
CONCURRENT=(dormouse-kda dormouse-gdn2 dormouse-engram dormouse-mor dormouse-spectral)

has_cuda_feature() {
  grep -qE '^cuda\s*=' "$1/Cargo.toml" 2>/dev/null
}

mkdir -p /tmp/opencode/fm/logs

err3() { # first 3 error-ish lines, single-line-ified, stderr+stdout
  grep -E '(^error|error\[E[0-9]+\]|^warning: unused|panicked at|FAILED|test result:)' "$1" \
    | head -3 | tr '\t\n' ' | ' | sed 's/  */ /g'
}

# summed "N passed" over every test binary in the run + ignored/failed counts
testcount() {
  awk '/test result:/ {
      for (i = 1; i <= NF; i++) {
        if ($i == "passed;" || $i == "passed,") p += $(i - 1)
        if ($i == "failed;") f += $(i - 1)
        if ($i == "ignored;") ig += $(i - 1)
      }
      n++
    } END { printf "bins=%d passed=%d failed=%d ignored=%d", n, p, f, ig }' "$1"
}

for c in "${CRATES[@]}"; do
  dir="$WS/crates/$c"
  [ -d "$dir" ] || dir="$WS/$c"
  [ -d "$dir" ] || continue
  log=$(mktemp)
  flag=""
  case "$PHASE" in
    check)       cmd=(cargo check -p "$c" -q) ;;
    test)        cmd=(cargo test  -p "$c" -q -- --test-threads=4) ;;
    cuda-check)  has_cuda_feature "$dir" || { echo -e "$c\tN-A\tno cuda feature"; continue; }
                 cmd=(cargo check -p "$c" --features cuda -q) ;;
    cuda-test)   has_cuda_feature "$dir" || { echo -e "$c\tN-A\tno cuda feature"; continue; }
                 cmd=(cargo test  -p "$c" --features cuda -q -- --test-threads=4) ;;
    examples)    has_cuda_feature "$dir" || { echo -e "$c\tN-A\tno cuda feature"; continue; }
                 cmd=(cargo check -p "$c" --examples --features cuda -q) ;;
    *) echo "bad phase" >&2; exit 2 ;;
  esac

  start=$SECONDS
  timeout 1800 "${cmd[@]}" >"$log" 2>&1
  rc=$?
  dur=$((SECONDS - start))
  cp -f "$log" "/tmp/opencode/fm/logs/${PHASE}-${c}.log"
  if [ $rc -eq 124 ]; then
    st="TIMEOUT"
  elif [ $rc -eq 0 ]; then
    st="PASS"
  else
    st="FAIL"
  fi
  case " ${CONCURRENT[*]} " in *" $c "*) flag=" [under-concurrent-edit]" ;; esac
  printf '%s\t%s\t%s\t%s(%ss)%s\n' "$c" "$st" "$(err3 "$log")" \
    "$(testcount "$log") " "$dur" "$flag"
  rm -f "$log"
done
