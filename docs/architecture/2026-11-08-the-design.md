# The architecture I would bet on

**Date:** 2026-11-08. This is a design document, not a result: every component is
tagged **[proven]** (measured here or published at our scale), **[hypothesis]**
(published elsewhere, untested at 95M), or **[bet]** (no close evidence).

The goal it serves, stated by the owner: *the best architecture in the world —
cheapest VRAM, cheapest compute, and the smartest.* Those are three different
axes and no single component serves all three; the design below is the combination
that serves each axis with the cheapest available mechanism.

## The design in one table

| # | component | what it buys | axis | status | evidence |
|---|---|---|---|---|---|
| 1 | **patch unit, 8 bytes/position** | 8× fewer positions per byte, ~3.8× less VRAM | compute + memory | **[proven speed]** 7.4× at b=32, **[hypothesis]** quality | `docs/research/2026-11-07-patch-compute.md`; bench in this repo |
| 2 | **14 unique layers, ONE pass** | no wasted compute | compute | **[proven]** φ=0.46: one recurrence = 0.46 block | 2604.21106v4 |
| 3 | **KDA linear attention** | O(1) state, ~0.4 B/token | memory | **[hypothesis]** +0.078 BPB at 60M | Olmo Hybrid 2604.03444v4 Table 5 |
| 4 | **CSA fine (8-byte chunks, top-k)** + **HCA coarse (256-byte)** | exact recall up to 1M at 34 MB | memory + context | **[proven mechanism]**, **[unmeasured quality]** | `model/csa.py`; probe driver |
| 5 | **MoD at capacity 0.5** | 0.63× FLOPs | compute | **[hypothesis]** 60M-3B | 2404.02258v1 §5.1 |
| 6 | **Muon shape-fair + bf16 state** | correct step on every shape; −181 MiB | compute + memory | **[proven]** RMS(dW)=lr; +18% speed | `docs/reviews/2026-11-07-muon-audit.md` |
| 7 | **QK-norm + z-loss + logit softcap** | allows LR 1e-3 without divergence | stability | **[proven here]** 1e-3 beats 3e-4 by 0.22 | our LR ladder |
| 8 | **doc_masking, 96.8% of corpus** | no cross-document bleed | quality | **[proven]** boundaries verified byte-exact | `docs/reviews/2026-11-07-data-pipeline.md` |
| 9 | **segment carry, 8 KB segments** | memory constant in context, 1M trainable | memory | **[in progress]** contract being implemented | `docs/reviews/2026-11-08-segment-csa.md` |
| 10 | **fine-grained MoE, low active fraction** | capacity per FLOP | compute | **[bet]** closest measurement 13.7× above us | 2402.07871v1 |

## The arithmetic of the combination

At 95M parameters, per byte of text:

| configuration | MFLOP/byte | vs today |
|---|---:|---:|
| today (1 byte/pos, 3 iterations) | 1710 | 1.0× |
| + one pass, 14 unique layers | 570 | 3.0× |
| + patch unit (÷8 positions) | 71 | 24× |
| + MoD 0.63 | 45 | **38×** |
| + MoE at 40% active (bet) | 18 | 95× |

Memory at 1M context: state 34 MB, activations constant with segments
(2.3 GiB per 8 KB segment, 0.8 GiB with checkpointing), optimizer 4× smaller with
bf16/int8 Muon momentum.

## What makes it different from a frontier model

Not size. Three properties that no open model at ≤1B has together:

1. **A patch-level compute unit with a causal codec** — the frontier all uses
   tokenizers; we use bytes and 8-byte patches, so Russian, code and numbers cost
   nothing extra.
2. **1M context at 34 MB of state** — V4.1 stores 890 B/token of KV, i.e. 890 MB
   at 1M; we store 34 MB and keep exact top-k recall over 8-byte chunks.
3. **Adaptive compute per token (MoD)** — the model spends FLOPs where the token
   is hard, and the frontier does not do this at all.

If 1 and 3 hold, the same quality costs **~38× fewer FLOPs per byte**. That is the
claim worth publishing, and it is measurable on one card.

## What must be true for this to work, in order

1. **E1: patches do not cost quality at equal bytes.** Running. If they cost
   >0.05 BPB, component 1 moves from "unit" to "speed-only" and the 38× collapses
   to ~6× (MoD + one pass).
2. **MoD A/B at 95M** — if 0.63× FLOPs costs >0.03 BPB, drop it.
3. **Segment carry parity** — a 2-segment pass must equal one continuous pass;
   without it there is no 1M training.
4. **KDA vs dense at 95M** — the Olmo result is at 60M on tokens, not bytes; ours
   may differ.

## What "smartest" actually means here

Not buildable by architecture at 1B. The capability comes from:
**post-training (SFT on 10-50k curated examples, RLVR on verifiable tasks, tool
calls) + retrieval.** That is where "smart per dollar" is won at our scale, and it
costs hours, not months. The architecture's job is to make that base as strong as
possible per FLOP.

## The honest ceiling

A 1B model trained on 22-88 GB of bytes will sit at GPT-2-small-class loss and
below frontier models on knowledge. What it *can* win: efficiency per FLOP, 1M
context at constant memory, byte-level multilingual behaviour, and cost. Those are
the four claims this design is built to make.
