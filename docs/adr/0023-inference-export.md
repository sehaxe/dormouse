# ADR-0023: the inference export, and which weight format it ships

Status: landed 2026-09-28, `dormouse-train/src/export.rs` + the `export` bin.
The format is a public interface; this is its specification and its evidence.

## The gap

`dormouse`'s only serialization path was the training checkpoint: a container
holding a model section, an optimizer section and an EMA teacher, written next
to a `.ngram` sidecar that reaches 34 GB at the 48M-slot budget. That is a
*run*, not a model. But `generate` and `serve` - the only two ways anyone
outside this box can use a dormouse - read that run, optimizer state and all.

Measured on the run below: **36.8 MB of actual weights inside a 104 MB
training container**, and an outsider's only download is that container plus
whatever sidecar sits next to it. A competent person who wanted to run this
model had to accept 34 GB or nothing. There was no export path at all.

## The format: `.dmexp`

```text
[0..8)    magic "DMEXPRT" + format version byte
[8..12)   header length, u32 LE
[12..20)  payload length in bytes, u64 LE
[20..24)  CRC-32 (IEEE) of the payload, u32 LE
[24..24+H)  header, UTF-8 TOML
[24+H..)    payload: the tensors, in header order, contiguous
```

The header carries the weight dtype, the full `DormouseConfig` as TOML (so the
model SHAPE travels with the weights - a shape the reader has to guess is a
shape they can get wrong), every tensor's module path and dims, the source
checkpoint and step, the parameter count, the weight magnitude range, and the
count of values that fell below the format's smallest normal.

`dormouse export info <file>` prints all of it and verifies the checksum
without loading a model. That is the answer to "someone handed me one file on a
machine that has never seen this repository".

**Why the header is TOML and not a binary index**: the repository's own config
is TOML, a human can read the header of a file they cannot load, and the
alternative is a second format to keep in sync for a parser nobody has
written.

**Why the payload is a flat concatenation in header order**: the file is
`header + 2 bytes x params`, exactly, so the size claim is arithmetic that the
round-trip test asserts. There is no per-tensor padding and no index block,
because the index is the header.

## The narrowing happens on the host

`f32 -> f16/bf16` and back are integer bit manipulation. Nothing in the export
or the load asks the backend for a narrow float tensor. That is deliberate, and
ADR-0016 is why:

- **bf16 matmul does not exist on this backend.** The LLVM dialect has no bf16
  type, so `restrict_to_llvm_backend` deletes it from the advertised element
  types and strips the bf16 tensor-core families; even reading a bf16 buffer
  back as f32 dies at kernel-compile time. What works is bf16 *storage* as
  `u16` bit patterns, pinned against f64 in
  `vendor/burn-fused/crates/burn-gdn2/tests/lowp_bf16_cuda.rs`. The export uses
  exactly that primitive and nothing more.
- **f16 matmul works but is silently slow** - the f16 tensor-core candidate
  dies at compile time and the autotuner falls back without a word.

So the export is a STORAGE format and the model that comes out of it is fp32,
widened back from the stored bits. That is what makes `bf16` safe to ship even
though bf16 compute is not: a weight file does not need a matmul.

## The measurements

Produced by `cargo run --release -p dormouse-train --example export_divergence`
(CPU backend, so it can never contend with a training run for the card).

**The run.** 20 steps at the `small` preset, pure CE, batch 2 x seq 256, on the
real corpus, then `save_ckpt`. **208 s to produce a minimal checkpoint**
including process start and the final save - 5 s/step on the CPU backend, and
the 115 s before step 0 is `DormouseModel::new` on the CPU. 9,197,390 params in
54 tensors (6.05M compute + 3.15M Engram memory rows).

One honest caveat, because it changes how to read the table: **20 steps is not
a trained model.** Loss went 5.5475 -> 5.4241 and the logits are nearly
uniform (|max| 1.11 over a 256-way vocabulary), so the argmax is a near-tie on
most positions and the top-1 column below is a WORST case, not a prediction for
a converged model. The weight-magnitude row is the durable half of the
measurement - it is a property of the initialization and the optimizer, not of
how long the run went.

**The divergence**, against the same checkpoint's fp32 logits on 16 windows of
256 B evenly spaced through `real_eval_v2/eval_tail.bin`:

| format | file bytes | vs 104 MB container | max abs weight | min non-zero abs weight | flushed to 0 | max abs logit delta | mean abs logit delta | top-1 agree | greedy 64 |
|---|---|---|---|---|---|---|---|---|---|
| f32 | 36,795,131 | 2.8x | 5.1545 | 1.1023e-9 | 0 | **0.000e+0** | 0.000e+0 | 100.0% | identical |
| f16 | 18,400,340 | 5.7x | 5.1562 | 5.9605e-8 | **8** | 1.247e-1 | 3.501e-3 | 100.0% | **DIFFERS** |
| bf16 | 18,400,343 | 5.7x | 5.1562 | 1.1059e-9 | **0** | 1.025e-1 | 9.680e-3 | 75.0% | identical |

Read those numbers honestly:

- **f32 is bit-exact.** Max delta exactly 0, not "small". The container itself
  is lossless, so the whole question is the narrowing and nothing else.
- **Both 2-byte formats move the logits by ~10% of their own scale** (0.10-0.12
  against |max| 1.11). That is a relative error of the format, not a
  catastrophe, and it is the expected size: 8 mantissa bits (bf16) is 2^-8
  relative per weight, accumulated through the loop.
- **The top-1 column is a near-tie artifact and should not be read as bf16
  being worse.** f16 100% / bf16 75% on a 20-step model whose logits barely
  leave 0 - one flipped near-tie and a window's argmax moves. Meanwhile the
  greedy column says the OPPOSITE thing: bf16's 64-step greedy continuation is
  byte-identical to fp32's and f16's is not. Both rows are one sample of a
  coin-flip. **The honest conclusion: at this model scale, neither 2-byte
  format preserves greedy decoding, and the two metrics disagree about which
  does.** A trained model with a peaked softmax would resolve this, and that
  measurement has not been taken - it needs a real checkpoint, which this repo
  does not currently have (the `checkpoints/` directory is empty).
- **f16 flushed 8 weights to zero and bf16 flushed 0.** That is the whole
  exponent argument, measured: the model's smallest non-zero weight is
  **1.1e-9**, four orders of magnitude below f16's smallest normal (6.1e-5), so
  f16 has to zero it. bf16 reproduced 1.1059e-9 exactly, because bf16's exponent
  is f32's.

## The format decision

`--dtype f32 | f16 | bf16`. All three ship. **The default and the
recommendation is `bf16`.**

The f16 objection, stated as the objection it deserves: f16 is *more accurate*
of the two - 10 mantissa bits against 7 - and the measurement above agrees
(3.501e-3 mean delta against bf16's 9.680e-3, a 2.8x advantage). Shipping bf16
therefore costs real precision. What it buys is that **the failure mode stops
being data-dependent and becomes impossible**: a weight above 65504 becomes
`inf` in f16, and the exporter REFUSES to write such a file and names bf16 as
the format that works - but that guard is a guard I wrote, and a guard is not a
proof. bf16 needs no guard, because bf16 cannot represent a weight this model
produces that f32 cannot.

For THIS model the risk is theoretical in both directions: max |w| 5.16 against
f16's 65504 is four orders of magnitude of headroom, and the 8 flushed values
out of 9.2M are 0.00009% of the parameters. So the honest recommendation is
narrower than "bf16 is right":

- **`bf16` as the default** - impossible failure mode, and the measured
  precision cost (mean delta 9.7e-3 on a 1.11-magnitude logit) is invisible next
  to the 8-orders-of-magnitude spread between the model's weights.
- **f16 is shipped and is the better file for a model whose measured range
  fits** - `export info` prints `max_abs` and `min_nonzero_abs`; comparing them
  against 65504 and 6.1e-5 IS the decision, and a stranger should not have to
  rebuild this crate to make it.
- **f32 when the download is not the constraint** - bit-exact, and it is the
  reference every other format is measured against.

## What this does NOT do

- **It is not a compute-precision change.** Loading a bf16 export gives an fp32
  model. Nothing here makes bf16 matmul work (§2.1 of AGENTS.md).
- **It does not carry the n-gram sidecar.** An export of a `--engram-ram` run
  carries the in-model Engram tables (the arm's actual weights, 3.15M of the
  9.2M params here) and not the 34 GB host table. The host table is training
  state: no 15 MB file holds 34 GB of it, and pretending otherwise is the
  ADR-0011 failure. `export_ckpt` never opens the sidecar, and the header's
  `source` field names the checkpoint the file came from.
- **It does not verify the model is any good.** It is a format conversion with
  a checksum, not an evaluation.

## The refusal (ADR-0011, the cardinal sin)

`generate` and `serve` read an export and nothing else. Pointed at a training
checkpoint they print what the file is and the command that converts it, and
exit. They never train-shaped-load the optimizer section and never go looking
for the sidecar. There is no `--ckpt-name` on either binary any more - one load
path, and it is the one a stranger can use.

`export_ckpt` takes the model config from `<ckpt-name>.config.toml`, the
snapshot the run wrote next to its weights, NOT from a `--preset`. A preset is
a guess about a shape; the snapshot is the shape the weights were trained
under. `--preset` is accepted only when the snapshot is absent, and then only
because the caller has to name it explicitly.

## The gate

`crates/dormouse-train/tests/export_roundtrip.rs`, one test, and every claim
the ADR above makes is asserted in it rather than described here:

- f32 export reproduces the source logits BIT-EXACTLY (0, not "close"), so any
  drift in the container is the container's, not the narrowing's;
- f16 and bf16 land inside the measured tolerance;
- the file is the header plus exactly 2 bytes per parameter;
- the header carries the config, and every tensor comes back with its name,
  dims and element count;
- a flipped payload bit is refused by checksum, not loaded;
- a training checkpoint is refused with the command that converts it, and the
  conversion that command names actually works;
- an f16 export of a model with a weight above 65504 is refused, and bf16
  accepts the same model.

```sh
cargo test -p dormouse-train --test export_roundtrip
cargo test -p dormouse-train --no-default-features --features cuda --test export_roundtrip
```

Both run the same body: the narrowing is host arithmetic over parameters read
back once, so it is backend-agnostic by construction. Running both anyway is
the point - a gate that only ever runs on one backend proves one backend. And
running the CUDA half is what found the two defects this feature would otherwise
have shipped with, both of them the same shape: **the model loads, and then
lies.**

1. **A header whose config disagreed with its own payload loaded silently.**
   Editing `d_model` in the header (and fixing the CRC, so that the *load* was
   what had to catch it) installed the payload's shapes into a model built at
   the header's width: the load succeeded, the forward returned garbage, and
   nothing said so - because `Param::from_data` takes the payload's dims, so no
   downstream check can ever see it. The loader now compares every tensor's
   shape against the model it is installing into and refuses the file, and the
   test builds exactly that tampered file (CRC included, so the shape check is
   what fires).
2. **The test hardcoded `Device::ndarray()`, so its CUDA run compared two
   BACKENDS.** The f32 export measured 7.5e-8 away from fp32 on CUDA, which
   read like a container defect and was in fact ndarray-vs-CUDA arithmetic.
   `dormouse_train::device()` is now `pub`, and the gate, the exporter and the
   measurement all build on it, so there is one device factory and a second
   backend cannot creep in under a test. With that fixed, **f32 is bit-exact on
   both backends** and the f16/bf16 deltas on CUDA are 4.6e-2 / 4.9e-2 against
   the same 0.5 tolerance.

## Open

- **The greedy-decoding question is unanswered.** §"The measurements" says why:
  it needs a trained checkpoint, and the honest way to settle it is a
  top-1-agreement and greedy-identity sweep over a converged model at both
  formats. Add it to the A/B queue; it is one example-binary run.
- **No independent reader exists.** The format is specified here and in the
  README, and the only implementation is the Rust one. A second implementation
  (a 40-line Python script that parses the header and reads one tensor) would be
  the proof that the format is a format and not a serialization with a
  description. Not written; it is the follow-up if a second consumer ever
  appears.
