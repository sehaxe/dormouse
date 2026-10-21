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
function). `tests/test_demo_engine.py` gates the routed arm on a 4-expert
top-2 fixture: max |Δlogit| **2.384e-7** against `decode._step_logits`
(tolerance 1e-4). The same gate on a **medium-shaped** artifact — d 1024,
256 experts, top-2, 50.3M Engram rows, `--engram-dtype fp16` — measures
**1.013e-6**:

| artifact | on disk | load | decode (Node, fp32) | RSS |
|---|---|---|---|---|
| `small` (dense, engram fp16) | 16.3 MB (15.5 MiB) | 0.13 s | 52 ms/byte | 128 MB |
| medium-shaped (top-2, engram fp16) | 138.1 MB (131.7 MiB) | 1.9 s | 75 ms/byte | 1032 MB |

Measured 2026-10-22 on this box, Node 26, CPU. The medium artifact is **not**
committed: it is generated from a checkpoint with the same command and
`--set use_msa=false,moe_capacity=0`.

Two things the container still lacks for a *trained* routed model:

1. **The LFB selection bias** (`loop.moe.lfb_bias`) is a non-persistent buffer,
   so it is not in `state_dict()` and the artifact routes on unbiased logits.
   The smallest additive change is one dense record beside `loop.moe.lfb_counts`
   (the engine already reads it if present and reports its absence).
2. **Packing.** The engine unpacks every ternary factor to fp32 at load: 79.4M
   factors × 4 B = 318 MB, plus the 50.3M-row Engram widened from fp16 to fp32
   (201 MB). Keeping the 2-bit codes and decoding per row inside the matvec
   (and reading the table's 32-wide row as fp16) would put the medium artifact's
   heap at ~155 MB instead of ~1 GB — ~20× less — at the cost of a slower
   matvec. That is the change to make if the page ever ships a routed model.
