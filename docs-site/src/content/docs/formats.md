---
title: Formats
description: bf16, 4-bit activations, ternary weights — and the export path that ships them.
---

The project trains in the format it ships in. There is no "train float, quantize
later" step: the forward pass is quantized from the first step, and a schedule
anneals the projection from full precision to hard ternary.

<figure class="clip">
  <video src="/dormouse/docs/media/Ternary.webm" poster="/dormouse/docs/media/Ternary.jpg" autoplay muted loop playsinline></video>
  <figcaption>Weights converge onto three levels during training, not after it; the file follows the levels.</figcaption>
</figure>

## The ladder

| format | bits per weight | what it is for |
|---|---:|---|
| bf16 | 16 | the optimization master; the reference |
| int4 / e2m1 | 4 | activations, and the storage of saved tensors |
| ternary | 2 | the weights that ship |

## Why the master stays full precision

A three-valued number cannot absorb a gradient. The optimizer holds one bf16
master per weight and accumulates into it; the network always *computes* in
ternary through a straight-through estimator, and the anneal pulls the master
onto the grid over the first two thousand steps.

This is what BitNet calls quantized-aware training, and the difference from
post-hoc quantization is measurable: a model that never saw the format during
training has to absorb the whole quantization error at the end.

## The export

`export_qat(model, path, mode="ternary")` walks the model, packs every weight to
2-bit codes plus one fp16 scale per tensor, and writes a container that reloads
into the same module. The round trip is checked, not trusted: the recovered
model differs from the trained one by 0.15% of the quantizer's own error, which
is the difference between "the file is the network" and "the file approximates
the network".

Two things the export refuses to do silently:

- report `mode=ternary` while packing nothing (that bug produced a file the same
  size as bf16 and called it quantized);
- pack a weight whose rank or shape has no codec.

## What the format is worth

The packing ratio is ~8× against bf16 — that is arithmetic, not a claim. The
quality cost is a measurement: a ternary run against its bf16 control, same
budget, same window, three seeds. The dated record lives in
[measurements](/dormouse/docs/measurements/) and in
`docs/reviews/2026-11-12-ternary-artifact.md`.

Whether the trade is worth taking depends on what the saved bytes buy — more
parameters, more bytes per token of context, or a smaller download. That is a
product decision, and the tables exist so it can be made with numbers.

---

## References

- **BitNet b1.58 (ternary weights)** — Ma et al. 2024, *The Era of 1-bit LLMs*, [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- **BitNet v2 (4-bit activations, Hadamard rotation)** — Wang et al. 2025, [arXiv:2504.18415](https://arxiv.org/abs/2504.18415)
- **Straight-through estimator** — Bengio et al. 2013, [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
- **Truncated SVD / Stiefel retraction** — Kohlberger, *Spectral Compact Training*, [arXiv:2604.00733](https://arxiv.org/abs/2604.00733)
