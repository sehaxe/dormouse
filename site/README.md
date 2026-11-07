# The landing page

`index.html` is generated, never edited. Its numbers come from the run logs, so
a number that is not in a log is not on the page.

```
cd site/gen
cargo run --release --offline -- \
  --logs /home/sehaxe/logs \
  --quality-logs /home/sehaxe/logs/muonq \
  --surprise ../assets/surprise.json \
  --script ../site.js \
  --template ../template.html \
  --out ../index.html
```

`template.html` is the markup, `site.js` is the behaviour (ladder, curve,
portrait), `video.py` renders the three motion cards from the logs, and
`assets/surprise.json` is the hero image's data — 1024 held-out bytes with their
per-byte surprise, written by `port/scripts/surprise_map.py`.

What it reads:

| input | what it supplies |
|---|---|
| `--logs` (recursive `*.jsonl`) | every run: source, optimizer, batch/seq, median step rate, held-out BPB |
| `--quality-logs` | the same rungs at a fixed budget, so the ladder's BPB column is a join on a rung name, not a typed number |

The rung table marks a rung **rejected** when its BPB delta is outside the
repo's ±0.02 noise floor, and the headline speedup is computed from the SAFE
rungs only. That is deliberate: the fastest row (2.08×) loses 0.26 BPB and must
never be the number in the hero.

No CI gate on freshness: the logs live outside the repository, so a check would
fail on every machine that has not trained. Regenerate after a run and commit
the HTML.

Design tokens and content rules: `.ai-docs/DESIGN.md`.
