#!/usr/bin/env bash
# ADR-0017: rename vendor/dormouse-fused -> dormouse-fused, and burn-* crates ->
# dormouse-*. A pure rename: no behavior change. Run it on a QUIET tree - it
# rewrites the git path of every file in the fork, so any agent with uncommitted
# work inside it will have its paths invalidated.
#
# Gate: the script refuses to run if `git status --porcelain` is non-empty.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run on a dirty tree: agents have uncommitted work" >&2
  git status --short >&2
  exit 1
fi
if pgrep -x train >/dev/null; then
  echo "refusing to run while a training run is live" >&2
  exit 1
fi

OLD=vendor/dormouse-fused
NEW=dormouse-fused
[[ -d $OLD ]] || { echo "no $OLD (already migrated?)" >&2; exit 1; }

echo "== 1. move the fork out of vendor/"
git mv "$OLD" "$NEW"

echo "== 2. rename the crate directories and their package/lib names"
cd "$NEW/crates"
for d in burn-*; do
  [[ -d $d ]] || continue
  nd=${d//burn-/dormouse-}
  git mv "$d" "$nd"
  # package name and lib name
  sed -i "s/^name = \"$d\"/name = \"$nd\"/" "$nd/Cargo.toml"
  sed -i "s/^name = \"$d\"$/name = \"$nd\"/" "$nd/Cargo.toml"
  # the lib target name, if declared explicitly
  sed -i "s/^\[lib\]/[lib]/" "$nd/Cargo.toml"
  python3 - "$nd/Cargo.toml" "$d" "$nd" <<'PY'
import re, sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
# [lib] name = "burn_foo"  ->  "dormouse_foo"
s = re.sub(r'(?m)^(\[lib\]\n(?:.*\n)*?\s*name\s*=\s*)"%s"' % re.escape(old.replace('-', '_')),
           lambda m: m.group(1) + '"%s"' % new.replace('-', '_'), s)
open(path, 'w').write(s)
PY
  # in-crate references to itself (doc comments, crate:: paths, #[doc])
  grep -rl --include='*.rs' --include='*.toml' -E "(^|[^a-z-])${d}([^a-z-]|$)|${d//-/_}::" . 2>/dev/null | while read -r f; do
    sed -i "s/\b${d}\b/${nd}/g; s/\b${d//-/_}\b/${nd//-/_}/g" "$f"
  done
done
# the meta-crate
if [[ -d burn-fused ]]; then git mv burn-fused dormouse-fused; fi

echo "== 3. rewrite imports and path deps across the workspace"
cd "$ROOT"
FILES=$(git ls-files 'crates/**' '*.toml' 'docs/**' 'research/**' '.bulba/**' 2>/dev/null || true)
# shellcheck disable=SC2086
for f in $FILES; do
  [[ -f $f ]] || continue
  grep -q 'burn-[a-z]' "$f" || continue
  sed -i 's/\bburn-\([a-z0-9]\)/dormouse-\1/g' "$f"
  # path deps lose the vendor/ prefix
  sed -i 's#\(\.\./\)\+vendor/dormouse-fused/#\1dormouse-fused/#g; s#vendor/dormouse-fused/#dormouse-fused/#g' "$f"
done

echo "== 4. root Cargo.toml: the exclude list keeps the fork out of our workspace"
grep -n 'exclude' -A 12 Cargo.toml || true

echo "== 5. build gate"
cargo check -p dormouse-core -p dormouse-train --features dormouse-train/cuda
cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib

echo "== 6. commit"
git add -A
git commit -q -m "refactor!: vendor/dormouse-fused -> dormouse-fused, burn-* crates -> dormouse-*

Pure rename, no behavior change (ADR-0017). The fork is ours - every one of
its 26 crates is our own code - so living under vendor/ and wearing the burn-*
namespace was a lie that made a researcher read it as a third-party dependency
instead of our technology library. The model crate keeps the model, the loop and
the config; every mechanism and kernel belongs to a library crate with its own
tests, its own arXiv reference and its own A/B."

echo "done. next: the mechanism moves listed in ADR-0017, one crate per commit."
