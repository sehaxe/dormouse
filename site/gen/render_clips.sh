#!/usr/bin/env bash
# Render the six loops in BOTH themes and drop them into site/media.
# CPU-only (cairo); safe to run while the GPU trains.
#
#   Bytes.webm / Bytes_dark.webm            (+ .jpg posters)
#
# The page swaps `src` on theme change; see the `data-dark` attribute in
# site/index.html and the theme handler in site/index.html.
set -euo pipefail
cd "$(dirname "$0")/../.."

MANIM=${MANIM:-/home/sehaxe/DeepTutor/.venv/bin/manim}
QUAL=${QUAL:--qm}          # 720p30; -ql for a fast preview
SCENES=${SCENES:-"Bytes Ternary Spectral Muon Curve Surprise"}
OUT=/tmp/manim-clips
mkdir -p "$OUT" site/media
shopt -s globstar nullglob

render() {                 # render <theme> <scene>
  local theme=$1 scene=$2
  CLIP_THEME=$theme "$MANIM" "$QUAL" --format=mp4 --media_dir "$OUT/$theme" \
    --output_file "$scene" site/gen/manim_scenes.py "$scene" >/dev/null 2>&1
}

for scene in $SCENES; do
  for theme in paper dark; do
    echo "### $scene ($theme)"
    render "$theme" "$scene"
    src=$(ls "$OUT/$theme"/videos/manim_scenes/*/"$scene".mp4 | head -1)
    if [ ! -f "$src" ]; then echo "!! $scene/$theme produced no mp4"; exit 1; fi
    suffix=""; [ "$theme" = dark ] && suffix="_dark"
    ffmpeg -y -loglevel error -i "$src" -c:v libvpx-vp9 -b:v 0 -crf 34 -an \
      -row-mt 1 -deadline good -cpu-used 2 "site/media/${scene}${suffix}.webm"
    ffmpeg -y -loglevel error -i "$src" -vf "select=eq(n\,10)" -vframes 1 -q:v 3 \
      "site/media/${scene}${suffix}.jpg"
    printf "   %-22s %5s KB\n" "${scene}${suffix}.webm" "$(( $(stat -c%s "site/media/${scene}${suffix}.webm") / 1024 ))"
  done
done
