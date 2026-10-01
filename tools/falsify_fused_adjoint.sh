#!/usr/bin/env bash
# Falsify the two-chunk fused-adjoint gate. Per the rule in .bulba/memory.md
# 2026-09-30: a green gate nobody has seen RED is not a gate. `34c5631` shipped
# verified by nothing for exactly this reason.
#
# WHAT IT DOES. Re-introduces `c305ec5`'s defect — the DOUBLE EXP at
# `tools/gen_bwd_f64.py:258`, `np.exp(g_last - G)` where `g_last` was already
# `exp(G_last)` — regenerates ONLY the two-chunk fixture into a temp path, and
# rebuilds the test against it. The oracle is then a DIFFERENT FUNCTION, so the
# two-chunk arm must go red while the one-chunk arm stays green (the double exp
# is invisible at T == chunk, which is the fingerprint of the class: `859f350`
# records the same tell for a head-major layout bug).
#
#   --regen   copy the generator with the double exp back in, and print the diff
#   --run     run the test against the corrupted fixture
#   --restore put the committed fixture back and re-run (must be green)
#
# The fixture is a COMMITTED BINARY and is restored from git, never edited in
# place, so a crash here cannot leave the tree with a wrong oracle.
set -uo pipefail
WT=/home/sehaxe/dormouse-wt/kda-adjoint
GEN=$WT/vendor/dormouse-fused/crates/dormouse-gdn2/tools/gen_bwd_f64.py
FIX=$WT/vendor/dormouse-fused/crates/dormouse-gdn2/tests/ref_bwd_f64_carry.bin
TGT=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/kdagf-target
LOCK=/home/sehaxe/dormouse/tools/build_lock.sh

cd "$WT/vendor/dormouse-fused" || exit 1
sha_before=$(sha256sum "$FIX" | cut -d' ' -f1)
echo "committed carry fixture sha256: $sha_before"

case "${1:-}" in
--regen)
    # `c305ec5`'s bug, in the generator's CURRENT shape. The correct line is
    #   np.exp(g_last_log - G)
    # with `g_last_log = G[:, :, :, c-1:c, :]` — the RAW log cumsum, one exp.
    # The bug exponentiated the cumsum a second time by using `g_last`, which
    # the generator still defines on the line above (`g_last = E[...]`, already
    # exp'd) and no longer uses. So the injection is that one identifier.
    grep -n "g_last_log\|g_last =\|decay_last =" "$GEN" | head
    cp "$GEN" "$GEN.orig"
    python3 - "$GEN" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
good = 'np.exp(G - g_last_log) if fault == "decay-sign" else np.exp(g_last_log - G)'
if good not in s:
    sys.exit("the generator's decay_last expression moved - re-read it before injecting")
bad = 'np.exp(G - g_last_log) if fault == "decay-sign" else np.exp(g_last - G)'
s = s.replace(good, bad, 1)
open(p, "w").write(s)
print("--- injected: this is c305ec5's double exp, one identifier ---")
PY
    echo "--- regenerating ONLY the carry fixture into a temp file ---"
    # The generator writes its default outputs on RELATIVE paths
    # (`tests/ref_bwd_f64.bin`), so it has to run from the crate root or it
    # dies on a missing directory. Both defaults are redirected into /tmp so a
    # partial run cannot touch a committed fixture.
    ( cd "$WT/vendor/dormouse-fused/crates/dormouse-gdn2" && \
      python3 tools/gen_bwd_f64.py \
          --out /tmp/opencode/one_corrupt.bin \
          --faults-out /tmp/opencode/faults_corrupt.bin \
          --carry-out /tmp/opencode/carry_corrupt.bin ) 2>&1 | tail -5
    ls -l /tmp/opencode/carry_corrupt.bin
    ;;
--run)
    test -f /tmp/opencode/carry_corrupt.bin || { echo "no corrupt fixture; run --regen first"; exit 1; }
    cp /tmp/opencode/carry_corrupt.bin "$FIX"
    echo "=== EXPECTED: the TWO CHUNKS arm is red, the ONE CHUNK arm is green ==="
    env CARGO_TARGET_DIR=$TGT $LOCK run falsify -- \
        cargo test -p dormouse-gdn2 --features cuda,autodiff \
        --test fused_adjoint_f64 -- --nocapture 2>&1 | tail -40
    echo "GATE EXIT: ${PIPESTATUS[0]}"
    ;;
--restore)
    # Restore the GENERATOR from git, not from the .orig copy. An earlier
    # version of this script did `cp .orig; rm .orig` and then found nothing to
    # copy from, leaving the tree with the double exp still injected and the
    # .orig already deleted — i.e. the restore silently did nothing while
    # printing nothing wrong. git is the authority here and needs no backup file.
    rm -f "$GEN.orig"
    cd "$WT" && git checkout -- vendor/dormouse-fused/crates/dormouse-gdn2/tools/gen_bwd_f64.py \
                            vendor/dormouse-fused/crates/dormouse-gdn2/tests/ref_bwd_f64_carry.bin
    if git diff --quiet -- vendor/dormouse-fused/crates/dormouse-gdn2/tools/gen_bwd_f64.py; then
        echo "generator restored from git (clean)"
    else
        echo "RESTORE FAILED: the generator still differs from HEAD"; exit 1
    fi
    sha_after=$(sha256sum "$FIX" | cut -d' ' -f1)
    echo "carry fixture sha256: $sha_before -> $sha_after"
    [ "$sha_before" = "$sha_after" ] && echo "RESTORED byte-identical" || { echo "RESTORE FAILED"; exit 1; }
    if [ "${2:-}" = "--run" ]; then
        echo "=== EXPECTED: both arms green again ==="
        cd "$WT/vendor/dormouse-fused"
        env CARGO_TARGET_DIR=$TGT $LOCK run falsify-restore -- \
            cargo test -p dormouse-gdn2 --features cuda,autodiff \
            --test fused_adjoint_f64 -- --nocapture 2>&1 | tail -30
    fi
    ;;
*)
    echo "usage: $0 --regen | --run | --restore"; exit 1
    ;;
esac
