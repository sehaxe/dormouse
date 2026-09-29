#!/usr/bin/env bash
# Worktree discipline, per docs/adr/0022-worktree-discipline.md.
#
# THE RULE: one task per worktree, and the worktree is removed when its task
# lands. Worktrees live OUTSIDE the repo root so nothing here can dirty the
# shared tree, and off /tmp (a 32 GB tmpfs that dies on reboot).
#
#   tools/wt.sh new <task>    create a worktree for one task, off HEAD
#   tools/wt.sh test <task>   the real test command, inside that worktree
#   tools/wt.sh rm <task>     land or discard the task, then drop the worktree
#   tools/wt.sh list          what exists
#
# `test` runs TWO gates: TEST_CMD below (our three crates, `--lib` only) and
# tools/lib_gate.sh (vendor/burn-fused, a separate cargo workspace, every crate
# on ndarray). Either red fails the worktree. lib_gate.sh also runs standalone.
#
# Builds are SERIAL. One heavy thing at a time (AGENTS.md doctrine 4): a cold
# worktree target dir compiles the whole burn+cubecl stack, and a mold link
# spikes tens of GB, so `test` refuses to start while another cargo is running
# rather than racing it.
set -u
REPO=/home/sehaxe/dormouse
WT_ROOT=${WT_ROOT:-/home/sehaxe/dormouse-wt}
# The repo's real check. `--lib` only: bare `cargo test` also builds examples,
# and the tree has unbuildable ones.
TEST_CMD="cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib"

die() { echo "wt: $*" >&2; exit 1; }
slug() { echo "$1" | tr 'A-Z ' 'a-z-'; }

# Uncommitted work in the shared tree is NOT in a new worktree. A worktree is
# a checkout of a COMMIT; if the thing you need exists only as a dirty file,
# this script will hand you a tree without it and you will debug the wrong
# build. Say so out loud rather than let it surprise you.
warn_dirty() {
    local n
    n=$(git -C "$REPO" status --porcelain | wc -l)
    [ "$n" -gt 0 ] || return 0
    echo "wt: NOTE the shared tree has $n dirty paths. They are NOT in this worktree."
    local missing
    missing=$(git -C "$REPO" status --porcelain | awk '{print $2}' \
        | grep -E '^(vendor/|Cargo\.(toml|lock)$)' || true)
    if [ -n "$missing" ]; then
        echo "wt: WARNING uncommitted manifest/vendor changes are the ones that change the build:"
        echo "$missing" | sed 's/^/wt:   /'
        echo "wt: This worktree builds from HEAD, which may be a DIFFERENT dependency graph."
        echo "wt: Land (or commit) those first if the task needs them."
    fi
}

case "${1:-}" in
new)
    [ $# -eq 2 ] || die "usage: wt.sh new <task>"
    s=$(slug "$2"); [ -n "$s" ] || die "empty task name"
    path="$WT_ROOT/$s"
    [ -e "$path" ] && die "$path already exists"
    mkdir -p "$WT_ROOT"
    git -C "$REPO" worktree add -b "wt/$s" "$path" HEAD || die "worktree add failed"
    warn_dirty
    echo "wt: $path  (branch wt/$s, off $(git -C "$REPO" rev-parse --short HEAD))"
    echo "wt: work here:  cd $path"
    echo "wt: test here:  tools/wt.sh test $s"
    ;;
test)
    [ $# -eq 2 ] || die "usage: wt.sh test <task>"
    path="$WT_ROOT/$(slug "$2")"
    [ -d "$path" ] || die "no worktree at $path"
    if pgrep -x cargo >/dev/null; then
        echo "wt: another cargo is running. Waiting for it (one heavy thing at a time)..."
        while pgrep -x cargo >/dev/null; do sleep 20; done
    fi
    avail=$(free -g | awk '/^Mem:/{print $7}')
    if [ "${avail:-0}" -lt 25 ]; then
        die "only ${avail} GB RAM available; a cold build needs 25 (AGENTS.md doctrine 4)"
    fi
    echo "wt: $TEST_CMD   (in $path)"
    cd "$path" || die "cd failed"
    s=$(date +%s); $TEST_CMD -j 4; rc=$?
    echo "wt: $(( $(date +%s) - s ))s, exit $rc"
    # The 21 crates of vendor/burn-fused are a SEPARATE cargo workspace (the
    # root Cargo.toml excludes it), so `-p` cannot reach them: the crossing is
    # a second cargo run from inside the fork. Without this branch the whole
    # technology library - including the 31 integration targets the `--lib`
    # above never builds - was ungated locally.
    tools/lib_gate.sh "$path" || rc=$?
    echo "wt: lib_gate exit $rc"
    [ "$rc" -eq 0 ] || die "gate red: see above"
    ;;
rm)
    [ $# -eq 2 ] || die "usage: wt.sh rm <task>"
    s=$(slug "$2"); path="$WT_ROOT/$s"
    [ -d "$path" ] || die "no worktree at $path"
    n=$(git -C "$path" status --porcelain | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "wt: $path has $n uncommitted paths:"
        git -C "$path" status --porcelain | sed 's/^/wt:   /'
        echo "wt: land or stash them yourself, then re-run. Not touching them."
        die "refusing to drop a worktree with unmerged work"
    fi
    git -C "$REPO" worktree remove "$path" || die "worktree remove failed"
    git -C "$REPO" branch -D "wt/$s" >/dev/null 2>&1
    echo "wt: dropped $path and branch wt/$s"
    ;;
list)
    git -C "$REPO" worktree list
    ;;
*)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
