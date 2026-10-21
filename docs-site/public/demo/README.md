# The demo's model artifact is GENERATED — and committed anyway, on purpose

`engine.js` (this directory) reads the `.dormouse` artifact that
`dormouse export` writes, and `docs-site/src/pages/demo.astro` fetches it at
`/demo/model.dormouse`; `?engram=0` fetches `/demo/model-engram-off.dormouse`
instead (see "the optional memory arm" below).

Two artifacts are committed because the live page has to work without a build
step, and a demo that is the model beats a recording of the model:

| file | what it is | on disk |
|---|---|---|
| `model.dormouse` | the trained `small` polygon, Engram at fp16 | 16,265,176 B (15.51 MiB) |
| `model-engram-off.dormouse` | `medium` geometry, top-2 dropless, **no Engram table** | 37,261,384 B (35.53 MiB) |

Both are over the 8 MB rule and are committed anyway: the page fetches them and
they are regenerable by the commands below. A fork without a blob still
degrades with a loud message naming the command. The full medium artifact
(138,058,200 B) stays generated — GitHub's per-file limit is 100 MB — and is
the measurement the size table is about.

```sh
# the trained small polygon (the page's default)
uv run dormouse export small --ckpt-name <name> --out-dir checkpoints \
  --engram-dtype fp16 --out docs-site/public/demo/model.dormouse --verify

# medium geometry, dropless top-2, memory arm removed (`?engram=0`)
uv run dormouse export medium --ckpt-name <name> --out-dir checkpoints \
  --set use_msa=false,moe_capacity=0 --engram-dtype fp16 --no-engram \
  --out docs-site/public/demo/model-engram-off.dormouse
```

`--verify` reads the file back and prints the max |Δlogits| between the source
model and the rebuilt one; the browser engine is checked against the Python
decode logits by `tests/test_demo_engine.py` (max |Δlogit| < 1e-4, measured
1.058e-6 on `small`, 1.013e-6 on the full medium export, 2.384e-7 on the routed
tiny fixture). The same file is what `dormouse decode --export …` and
`dormouse serve --export …` load, so the demo, the CLI and distribution read
one format rather than three copies of the weights.

## The routed (top-k) arm

`engine.js` implements both MoE dispatch arms:

- `moe_topk = 0` — the dense blend (`small`): softmax over the whole bank,
  every expert computed.
- `moe_topk = k` with `moe_capacity = 0` — the DROPLESS top-k: the gate is
  `softmax(logits over all E)` gathered at the top-k and renormalised over
  those k, every routed token reaches its expert, the shared expert is always
  added, one shared `down`. This is `experts.py`'s `forward` + `_naive_dispatch`
  at `capacity = 0`, one to one.

Anything else is refused by field in the header's `engine_refusal` (MSA,
AttnRes, LOOM, ByteFlow, DSpark, a non-`silu` expert activation, a dense
projection, and a CAPACITY-dropping top-k, whose drop set is not a per-byte
function).

## Packed factors, unpacks nothing

The artifact already writes a ternary factor as 2-bit codes + one exact scale
(`model/export.py::pack_ternary`, four codes per byte, scale in the header).
The engine keeps exactly that: `readTensors` returns a `Uint8Array` view into
the file, and `matvec` decodes each code to `{0, ±scale}` as it multiplies.
There is no load-time expansion and therefore no float copy to hold.

The same rule for the Engram table: an fp16 table stays a `Uint16Array` view,
and `engramRead` converts only the 32 values (per order) that a step reads.

`W.stats` reports the boundary the tests assert:

| field | full medium | engram off | a widened copy would be |
|---|---:|---:|---:|
| `factorBytes` (codes + scales) | 19,859,480 B | 19,859,480 B | 317,652,480 B |
| `engramBytes` | 100,663,296 B | 0 B | 201,326,592 B |
| `tensorBytes` (all tensors held) | 155,292,700 B | 54,363,164 B | 553,748,996 B |

Measured 2026-10-22 on this box, Node 26, CPU. The full artifact's tensor heap
is **155.3 MB instead of 553.7 MB**, a saving of **398,456,296 B** (297,793,000
B from the factors, 100,663,296 B from the table). The old engine also held the
file's tensors and its own expanded copies at once, so RSS fell from **1032 MB
to 233 MB** on the same measurement script.

| artifact | on disk | load | decode (Node, fp32) | RSS |
|---|---|---|---|---|
| `small` (dense, engram fp16) | 16.3 MB (15.5 MiB) | 0.07 s | 107 ms/byte | 102 MB |
| medium, Engram fp16 | 138.1 MB (131.7 MiB) | 0.23 s | 131 ms/byte (7.7 B/s) | 233 MB |
| medium, Engram off | 37.3 MB (35.5 MiB) | 0.15 s | 124 ms/byte (8.1 B/s) | 134 MB |

The packed matvec is ~1.7x slower than the old widened one (13.4 B/s before),
which is the stated cost of not holding the copy: the decode adds a shift, an
and, and a branch per element. The table read is free at this scale.

## The optional memory arm (`?engram=0`)

`--no-engram` is a config change, not a second container format: the exporter
omits every `loop.engram.*` record and writes `use_engram=false`, exactly the
same header field a model trained without a memory module has. The engine then
takes the reference's `engram is None` branch (the whole memory mixture,
including `mem_dense`, is an exact zero) and says **engram off** in the status
line and in `W.stats.useEngram`. The parity gate keeps Engram ON; the switch
exists because the table is 100.7 MB of the 138.1 MB download and the live page
has to work on a phone. The page must not present the ablation as the full
model, and it does not.

The committed `model-engram-off.dormouse` is a **seeded** `medium` export:
no trained `medium` checkpoint exists on this box, so it demonstrates the
routed geometry and the memory accounting, not language quality. Regenerate it
from a trained checkpoint with the command above; the header carries `step`,
`seed` and `commit` when there is provenance to carry.

Two things the container still lacks for a *trained* routed model:

1. **The LFB selection bias** (`loop.moe.lfb_bias`) is a non-persistent buffer,
   so it is not in `state_dict()` and the artifact routes on unbiased logits.
   The smallest additive change is one dense record beside `loop.moe.lfb_counts`
   (the engine already reads it if present and reports its absence).
2. **A batched forward.** The engine is one byte per step, one state per
   iteration slot; the packed matvec is scalar JS. A worker plus SIMD would be
   the next speed lever, not a size one.
