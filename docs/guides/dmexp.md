# The model file: `.dmexp`

**A training checkpoint is a run, not a model.** `<ckpt-name>.bin` holds a model
section, an optimizer section and an EMA teacher, and it sits next to a `.ngram`
sidecar that reaches 34 GB at the 48M-slot budget. Measured on the run below:
**36.8 MB of actual weights inside a 104 MB training container** — and an
outsider's only download was that container plus whatever sidecar the run
happened to leave.

`dormouse export` produces the other thing: one self-describing file with the
optimizer dropped, the `.ngram` sidecar never opened, and the fp32 masters
narrowed to 2 or 4 bytes each.

Spec, evidence and gate:
[`docs/adr/0023-inference-export.md`](../adr/0023-inference-export.md) ·
`crates/dormouse-train/src/export.rs` ·
`cargo test -p dormouse-train --test export_roundtrip` (and again with
`--no-default-features --features cuda`).

## Using it

```sh
# convert. The config comes from <ckpt-name>.config.toml — the run's own
# snapshot — NOT from --preset. A preset is a guess about a shape; the snapshot
# is the shape the weights were trained under.
./target/release/export run --ckpt-dir checkpoints --ckpt-name latest --dtype bf16

# identify + verify, without loading a model
./target/release/export info checkpoints/latest.bf16.dmexp

./target/release/generate --export checkpoints/latest.bf16.dmexp --prompt "once" --steps 64
./target/release/serve   --export checkpoints/latest.bf16.dmexp --port 8000
```

`--out` defaults to `<ckpt-dir>/<ckpt-name>.<dtype>.dmexp`.

**`generate` and `serve` take `--export` and nothing else.** Pointed at a
training checkpoint they print what the file is and the command that converts it,
then exit — they never train-shaped-load the optimizer section and never go
looking for the sidecar. That refusal is ADR-0011: a wrong-but-plausible answer
is the cardinal sin, and a 34 GB read to obtain 30 MB of weights is that sin
wearing a plausible face.

## Layout

```text
[0..8)    magic "DMEXPRT" + format version byte
[8..12)   header length, u32 LE
[12..20)  payload length in bytes, u64 LE
[20..24)  CRC-32 (IEEE) of the payload, u32 LE
[24..24+H)  header, UTF-8 TOML
[24+H..)    payload: the tensors, in header order, contiguous
```

Implemented at `crates/dormouse-train/src/export.rs:280-286` (`encode`) and
`:305-338` (`decode`); `FIXED` is 24. There is no per-tensor padding and no
index block, because the index **is** the header — so the size claim is
arithmetic, and the round-trip test asserts exactly `header + 2 bytes × params`.

**Why the header is TOML and not a binary index**: the repository's own config
is TOML, a human can read the header of a file they cannot load, and the
alternative is a second format to keep in sync for a parser nobody has written.

## What the header carries

`Header` (`crates/dormouse-train/src/export.rs:134-158`): the format version
byte, the weight dtype, `source` (the checkpoint it was made from) and `step`,
`num_params` and `num_tensors`, `flushed` (weights that fell below the format's
smallest normal and became zero), `max_abs` and `min_nonzero_abs`, the full
`DormouseConfig` as TOML, and one entry per tensor with its module path and dims.

**The model shape travels with the weights** — a shape a reader has to be told
separately is a shape they can be told wrong. `overflowed` is per tensor and is
always 0 in a file that exists: `encode` refuses to write one, so a reader can
trust the number instead of suspecting it.

`export info` prints all of it and checks the CRC. That is the answer to
"someone handed me one file on a machine that has never seen this repository".

## `--dtype`: the trade-off, measured

Measured on a 20-step `small` run (**9,197,390 params in 54 tensors**, 6.05M
compute + 3.15M Engram memory rows) against the same checkpoint's fp32 logits
on 16 windows of 256 B from `real_eval_v2/eval_tail.bin`. Produced by
`cargo run --release -p dormouse-train --example export_divergence` (CPU backend,
so it can never contend with a training run for the card). Numbers as recorded
in ADR-0023, 2026-09-28.

| format | file bytes | vs 104 MB container | max abs weight | min non-zero weight | flushed to 0 | max abs logit delta | mean abs logit delta | top-1 agree | greedy 64 |
|---|---|---|---|---|---|---|---|---|---|
| `f32` | 36,795,131 | 2.8× | 5.1545 | 1.1023e-9 | 0 | **0.000e+0** | 0.000e+0 | 100.0% | identical |
| `f16` | 18,400,340 | 5.7× | 5.1562 | 5.9605e-8 | **8** | 1.247e-1 | 3.501e-3 | 100.0% | **DIFFERS** |
| `bf16` | 18,400,343 | 5.7× | 5.1562 | 1.1059e-9 | **0** | 1.025e-1 | 9.680e-3 | 75.0% | identical |

A 2-byte format halves the download: **18.4 MB against 36.8 MB for fp32, 5.7×
against the 104 MB training container.** f32 is bit-exact (delta exactly 0), so
the entire question is the narrowing and nothing else.

**Read the last two columns honestly.** This model's logits barely leave 0
(|max| 1.11 over a 256-way vocabulary) because 20 steps is not training, so the
argmax is a near-tie and top-1 agreement is a worst case, not a prediction. The
two metrics disagree about which format is safer: f16 wins top-1 and loses the
greedy continuation, bf16 the reverse. **Neither 2-byte format preserves greedy
decoding at this scale, and the data does not say which is better.** Settling it
needs a converged checkpoint's softmax, which this repo did not have when the
measurement was taken (`checkpoints/` is empty).

What *is* settled is the exponent range, and it is the durable half of the
measurement: the model's smallest non-zero weight is **1.1e-9**, four orders of
magnitude below f16's smallest normal (6.1e-5), so **f16 had to zero 8
weights**. bf16 reproduced 1.1059e-9 exactly, because bf16's exponent is f32's.
The weight-magnitude row is a property of the initialization and the optimizer,
not of how long the run went — which is why it survives as a conclusion when
the logits columns do not.

**Ship `bf16` (the default).** f16 is genuinely the *more accurate* format — 10
mantissa bits against 7, and 2.8× lower mean logit delta — so this costs real
precision. It buys a failure mode that stops being data-dependent: an f16
weight above 65504 becomes `inf`, and the only thing standing between that and a
broken model is a guard the exporter applies. bf16 needs no guard. **f16 remains
the better file when the measured range fits** — `export info` prints `max_abs`
and `min_nonzero_abs`, and comparing them against 65504 and 6.1e-5 IS the
decision; it ships so a stranger can make it without rebuilding this crate.

## This is a storage format, not a compute-precision change

Loading a bf16 export gives an **fp32** model. bf16 matmul does not exist on
this CUDA backend (ADR-0016 — the LLVM dialect has no bf16 type) and f16 matmul
works but falls back off the tensor cores without saying so. The narrowing and
widening are integer bit manipulation on the host
(`crates/dormouse-train/src/export.rs:85`, `:93`) — nothing in the export or the
load asks the backend for a narrow float tensor. The `u16`-bit-pattern storage
primitive the export uses is the one pinned against f64 in
`vendor/dormouse-fused/crates/burn-gdn2/tests/lowp_bf16_cuda.rs`. A weight file does
not need a matmul; that is what makes `bf16` safe to ship even though bf16
compute is not.

## What an export cannot carry

The **host table** of an `--engram-ram` run. That is training state: no 18 MB
file holds 34 GB of it, and pretending otherwise is the ADR-0011 failure.
`export_ckpt` never opens the sidecar. The **in-model** Engram tables do travel
(3.15M of the 9.2M params in the measured run).

The refusal is **on the shape, not the flag**: on that path the config seam
squeezes `engram_rows` to 1 per order
(`crates/dormouse-train/src/cfg.rs`), and `refuse_unservable_memory`
(`crates/dormouse-train/src/decode.rs:37-56`) rejects any config whose
`engram_rows` rounds up to one row per order — every key would read the same row
and the memory arm would be a constant that was never trained. Both binaries call
it **before a single byte is sampled**
(`crates/dormouse-cli/src/bin/generate.rs:28`, `.../serve.rs:203`). Gate:
`cargo test -p dormouse-train --test decode_seam`.

It also does not verify the model is any good. This is a format conversion with
a checksum, not an evaluation.