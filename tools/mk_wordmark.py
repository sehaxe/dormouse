#!/usr/bin/env python3
"""The block-pixel wordmark — the brand identity, as the design system defines it.

opencode's DESIGN.md is explicit that the identity is not a drawn logo: "the
brand name rendered as a 5-row block of monospaced character cells, used in the
primary nav and as the centerpiece of the hero TUI mockup", and its icon rule is
"small monospaced character glyphs, not bitmaps or SVG". This script is the one
source of that wordmark, so the landing page, the banner, the social card and the
favicon cannot drift apart.

    python3 tools/mk_wordmark.py ascii  dormouse     # the 5 rows of block cells
    python3 tools/mk_wordmark.py svg    dormouse 8   # SVG rects, cell = 8 px

The SVG path emits RECTS, not <text>: a block character's advance width is a
property of whatever monospace font the renderer happens to have, so a text-based
wordmark would reflow differently in the browser, in the OG rasteriser and in a
viewer. Rects are exact everywhere.
"""

from __future__ import annotations

import sys

#: 5-row glyphs. '#' is a block cell, '.' is empty. Widths differ (m is 5, the
#: rest are 4) because a 4-wide m is unreadable at this height.
GLYPHS: dict[str, list[str]] = {
    "d": ["###.", "#..#", "#..#", "#..#", "###."],
    "o": [".##.", "#..#", "#..#", "#..#", ".##."],
    "r": ["#.##", "##..", "#...", "#...", "#..."],
    "m": ["##.##", "#.#.#", "#.#.#", "#.#.#", "#.#.#"],
    "u": ["#..#", "#..#", "#..#", "#..#", ".###"],
    "s": [".###", "#...", ".##.", "...#", "###."],
    "e": [".##.", "#..#", "####", "#...", ".###"],
    "n": ["#..#", "##.#", "#.##", "#..#", "#..#"],
    "a": [".###", "....", ".###", "#..#", ".###"],
    "c": [".###", "#...", "#...", "#...", ".###"],
    "h": ["#...", "###.", "#..#", "#..#", "#..#"],
    "i": ["#", "#", "#", "#", "#"],
    "l": ["#", "#", "#", "#", "#"],
    "t": ["#..", "###", "#..", "#..", ".##"],
    "p": ["###.", "#..#", "###.", "#...", "#..."],
    "g": [".###", "#..#", ".###", "...#", ".##."],
    "b": ["#...", "###.", "#..#", "#..#", "###."],
    "f": [".##", "#..", "###", "#..", "#.."],
    "w": ["#...#", "#...#", "#.#.#", "#.#.#", ".#.#."],
    "x": ["#..#", ".##.", ".##.", "#..#", "#..#"],
    "y": ["#..#", "#..#", ".###", "...#", ".##."],
    "v": ["#..#", "#..#", "#..#", ".##.", ".##."],
    "k": ["#..#", "#.#.", "##..", "#.#.", "#..#"],
    "j": ["..#", "..#", "..#", "#.#", ".#."],
    "z": ["####", "..#.", ".#..", "#...", "####"],
    "0": [".##.", "#..#", "#..#", "#..#", ".##."],
    "1": [".#", "##", ".#", ".#", ".#"],
    "2": [".##.", "#..#", "...#", "..#.", "####"],
    "3": ["###.", "...#", ".##.", "...#", "###."],
    "4": ["#..#", "#..#", "####", "...#", "...#"],
    "5": ["####", "#...", "###.", "...#", "###."],
    "6": [".###", "#...", "###.", "#..#", "###."],
    "7": ["####", "...#", "..#.", "..#.", "..#."],
    "8": ["###.", "#..#", "###.", "#..#", "###."],
    "9": ["###.", "#..#", ".###", "...#", "###."],
    "-": ["....", "....", "####", "....", "...."],
    ".": ["..", "..", "..", "..", "##"],
}

GAP = 1  # empty columns between glyphs


def rows(text: str) -> list[str]:
    """The wordmark as 5 strings of block cells (U+2588) and spaces."""
    out = [""] * 5
    for i, ch in enumerate(text.lower()):
        g = GLYPHS.get(ch)
        if g is None:
            raise SystemExit(f"mk_wordmark: no glyph for {ch!r} (add it to GLYPHS)")
        for r in range(5):
            out[r] += g[r].replace("#", "\u2588").replace(".", " ")
            if i < len(text) - 1:
                out[r] += " " * GAP
    return out


def svg(text: str, cell: int) -> str:
    """The same wordmark as SVG rects — deterministic across every renderer."""
    grid = [g for ch in text.lower() for g in (GLYPHS[ch],)]
    width_cells = sum(len(g[0]) for g in grid) + GAP * (len(grid) - 1)
    rects: list[str] = []
    x0 = 0
    for g in grid:
        w = len(g[0])
        for r in range(5):
            run_start: int | None = None
            for c in range(w + 1):  # +1 flushes a run at the row end
                filled = c < w and g[r][c] == "#"
                if filled and run_start is None:
                    run_start = c
                elif not filled and run_start is not None:
                    rects.append(
                        f'<rect x="{(x0 + run_start) * cell}" y="{r * cell}" '
                        f'width="{(c - run_start) * cell}" height="{cell}"/>'
                    )
                    run_start = None
        x0 += w + GAP
    return (
        f'<g fill="currentColor" width="{width_cells * cell}" height="{5 * cell}">\n  '
        + "\n  ".join(rects)
        + "\n</g>"
    )


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] not in ("ascii", "svg"):
        print(__doc__)
        return 2
    what, text = sys.argv[1], sys.argv[2]
    if what == "ascii":
        print("\n".join(rows(text)))
    else:
        cell = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        print(svg(text, cell))
    return 0


if __name__ == "__main__":
    sys.exit(main())
