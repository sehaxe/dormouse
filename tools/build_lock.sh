#!/usr/bin/env bash
# THE BUILD LOCK. Serialises HEAVY cargo work; nothing else.
#
# WHY THIS EXISTS. On 2026-09-29 five agents were launched at once and every one
# needed a cold `vendor/dormouse-fused` build: 686 packages, ~1400 s, 41 GB of
# target dir. Result: 4 cargos, 10 rustc at 4.5-5.5 GB each, available RAM 11 GB
# against a 25 GB threshold, /proc/pressure/memory some at 8.32 %, swap 22 GB,
# /home at 95 %. That is a pre-freeze, and AGENTS.md 2.4 records three
# completed ones that look exactly like this. The lesson was written down as
# "run fewer agents", which was the WRONG LESSON: the agents were not the
# problem, the unsynchronised BUILDS were. Reading, reasoning and writing do not
# touch the machine.
#
# SO: many agents, one build. Acquire the lock, build, release. Waiters queue.
# A plain `pgrep -x cargo` preflight is not enough - it races, and an agent that
# starts "just as" another finishes is exactly how 4 cargos happened.
#
# USAGE
#   tools/build_lock.sh run <label> -- <cmd...>   # THE FORM TO USE: hold, run, release
#   tools/build_lock.sh hold [label]   # only from a shell that then STAYS alive
#   tools/build_lock.sh release        # release
#   tools/build_lock.sh status         # who holds it, or "free"
#   tools/build_lock.sh run <label> -- <command...>   # hold, run, release
#
# Stale locks: a lock whose PID is gone is broken and is taken over after
# STALE_SECONDS, because a killed agent must not wedge the queue forever.
set -uo pipefail

LOCKDIR="${TMPDIR:-/tmp}/dormouse-buildlock"
mkdir -p "$LOCKDIR" 2>/dev/null || LOCKDIR="$HOME/.cache/dormouse-buildlock"
mkdir -p "$LOCKDIR"
STALE_SECONDS=${BUILD_LOCK_STALE_SECONDS:-5400}
# Verified: hold+status shows STALE for a bare invocation, because the PID
# recorded is this script's and it exits at once. That is the reason `run`
# exists and the reason the note in the usage is there.

_alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

_acquire() {
    local label=${1:-unnamed} waited=0
    while true; do
        # mkdir is atomic; whoever creates the directory owns the lock.
        if mkdir "$LOCKDIR/hold" 2>/dev/null; then
            echo $$ > "$LOCKDIR/hold/pid"
            echo "$label" > "$LOCKDIR/hold/label"
            echo "build_lock: ACQUIRED by $$ ($label)"
            return 0
        fi
        local p l age
        p=$(cat "$LOCKDIR/hold/pid" 2>/dev/null || echo "")
        l=$(cat "$LOCKDIR/hold/label" 2>/dev/null || echo "?")
        if ! _alive "$p"; then
            local mtime now
            mtime=$(stat -c %Y "$LOCKDIR/hold" 2>/dev/null || echo 0)
            now=$(date +%s)
            age=$(( now - mtime ))
            if [ "$age" -gt "$STALE_SECONDS" ]; then
                echo "build_lock: taking over a STALE lock from dead pid $p ($l), age ${age}s" >&2
                rm -rf "$LOCKDIR/hold"
                continue
            fi
        fi
        waited=$(( waited + 15 ))
        echo "build_lock: waiting for $l (pid $p), ${waited}s elapsed"
        sleep 15
    done
}

_release() {
    rm -rf "$LOCKDIR/hold"
    echo "build_lock: RELEASED"
}

case "${1:-status}" in
    hold)    shift; _acquire "${1:-unnamed}" ;;
    release) _release ;;
    status)
        if [ -d "$LOCKDIR/hold" ]; then
            p=$(cat "$LOCKDIR/hold/pid" 2>/dev/null || echo "?")
            l=$(cat "$LOCKDIR/hold/label" 2>/dev/null || echo "?")
            if _alive "$p"; then echo "HELD by $p ($l)"
            else echo "STALE - held by dead pid $p ($l); next acquire will take it over"; fi
        else
            echo "free"
        fi ;;
    run)
        label=${2:-unnamed}; shift 2
        [ "${1:-}" = "--" ] && shift
        _acquire "$label"
        trap '_release' EXIT INT TERM
        "$@"; rc=$?
        _release; trap - EXIT
        exit $rc ;;
    *)
        echo "usage: $0 {run <label> -- cmd... | hold [label] | release | status}" >&2
        echo "NOTE: 'hold' only works from a shell that stays alive - the lock records" >&2
        echo "      this script's PID, and a bare invocation exits immediately, which" >&2
        echo "      makes its own lock look stale. Use 'run' for one command." >&2
        exit 2 ;;
esac
