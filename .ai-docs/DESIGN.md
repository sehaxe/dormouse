# dormouse design language

One page, one job: prove the numbers are real. The design serves that, so it is
deliberately plain — no gradient, no glass, no illustration, no web fonts.

## Palette

| token | value | use |
|---|---|---|
| `--bg` | `#0b0c0e` | page |
| `--panel` | `#121418` | cards, table stripes |
| `--line` | `#23262c` | hairlines, table rules |
| `--text` | `#e8e6e1` | body (warm off-white, not pure white) |
| `--dim` | `#8a8f98` | labels, secondary |
| `--accent` | `#d9a05b` | one accent: numbers that matter, links, the sparkline |
| `--good` | `#7fb069` | a win (rare — a win must be visible) |
| `--bad` | `#c46a5a` | a retraction, a loss |

Two rules: the accent appears at most once per card, and colour is never the
only signal (every coloured number also carries a word).

## Type

- **Display / body:** system UI stack (`ui-sans-serif, system-ui, -apple-system,
  "Segoe UI", Roboto, Helvetica, Arial`), weights 400 and 600 only.
- **Numbers:** `ui-monospace, "SF Mono", Menlo, Consolas, monospace`, with
  `font-variant-numeric: tabular-nums` so columns line up.
- Scale: 64 / 34 / 20 / 16 / 13 px, line-height 1.15 for display, 1.55 for body.
- Measure: 68ch max for prose; tables may run to 1080px.

## Layout

- Single column, `max-width: 1080px`, 24px gutters, 96px section rhythm.
- Sections separated by a hairline, never by a box.
- Cards only for the three hero stats; everything else is a table, because a
  table is the honest shape for a number with a unit and a source.
- Mobile: one column, stats stack, tables scroll horizontally rather than wrap.

## Content rules

- Every number carries its unit and its source (a run id or a doc).
- A claim that was retracted is shown as retracted, in `--bad`, next to the
  claim it replaced. Deleting it would be the one dishonest move available.
- No adjective does the work of a number. "Fast" is banned; "1.89× at +0.006
  BPB" is the sentence.
