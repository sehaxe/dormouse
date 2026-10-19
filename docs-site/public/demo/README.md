# The demo's model artifact is GENERATED — and committed anyway, on purpose

`engine.js` (this directory) reads the `.dormouse` artifact that
`dormouse export` writes, and `docs-site/src/pages/demo.astro` fetches it at
`/demo/model.dormouse`.

The `small` preset's artifact is **16.3 MB** (15.51 MiB, measured 2026-10-19 on
a fresh `small` init) with `--engram-dtype fp16` — 22.6 MB with the default
fp32 table. Both are over the 8 MB rule for a committed blob, and the blob is
**committed anyway** (2026-10-19): the live page has to work without a build
step, and a demo that is the model beats a recording of the model. A fork
without the blob still degrades with a loud message naming the command below,
which is the behaviour the page was designed around. Generate it from a trained
checkpoint:

```sh
uv run dormouse export small --ckpt-name <name> --out-dir checkpoints \
  --engram-dtype fp16 --out docs-site/public/demo/model.dormouse --verify
```

`--verify` reads the file back and prints the max |Δlogits| between the source
model and the rebuilt one; the browser engine is checked against the Python
decode logits by `tests/test_demo_engine.py` (max |Δlogit| < 1e-4, measured
1.058e-6 on `small`).

The same file is what `dormouse decode --export …` and
`dormouse serve --export …` load, so the demo, the CLI and distribution read
one format rather than three copies of the weights.
