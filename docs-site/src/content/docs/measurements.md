---
title: Measurements
description: How a number is produced here, what the current record is, and where the raw evidence lives.
---

## The rule

A result is always three things: **a loss, a window, a budget**.

> Held-out bits per byte on a fixed window, after one pass over the corpus, on a
> single RTX 5070 Ti — compared against the same model with the mechanism
> removed.

The absolute height moves whenever the architecture changes, and it will keep
moving. The comparison is what survives.

- **Window:** `data_local/best/fwedu_eval.bin`, 196 608 bytes, `off=0`. It
  changed on 2026-11-08 when the corpus was refetched; anything from before that
  date is a different measurement.
- **Noise:** ±0.011 BPB per seed at the current budget. A delta below that is
  not a result, whatever the table says.
- **Seeds:** three per arm, paired deltas. A single-seed delta is a hypothesis.

## The current record

Dated, and deliberately not framed as the model's identity — the next
architecture will move it.

| run | held-out BPB | corpus | time | log |
|---|---:|---:|---:|---|
| 95M gated FFN, ternary (QAT) | 1.2220 | 2.64 GB | 9.5 h | `logs/arch/run2640_ternary.jsonl` |
| 95M gated FFN, bf16 control | **1.1871** | 2.64 GB | 8.4 h | `logs/arch/run2640_s1.jsonl` |
| 95M dense GELU, bf16 | 1.1940 | 2.00 GB | 6.6 h | `logs/arch/run2gb_s1.jsonl` |
| xz on the same window | 2.31 | — | — | reference compressor |

The difference between the first two rows is the price of the format at this
scale: **+2.94% BPB** for ~8× fewer bytes.

## What was measured and dropped

| mechanism | measured | verdict |
|---|---|---|
| MoE, 32 experts | +0.004 BPB, needs a bigger corpus to pay | deferred |
| low-rank at r=64 | 1.95× faster, **+28% loss** | refused |
| attention factorisation | 2.57×, **+27%** | refused |
| low-rank Muon (Orth-Dion) | 1.25×, **+33%** | refused |
| fewer Newton–Schulz steps | 1.08×, **+10%** | refused |
| gated FFN | 0.00% params, **−1.2%** | shipped |

## Where the raw evidence is

| what | where |
|---|---|
| the architecture matrix, with every arm and seed | `docs/reviews/2026-11-09-architecture-matrix.md` |
| the ternary artifact, its cost and its two corrections | `docs/reviews/2026-11-12-ternary-artifact.md` |
| the gated-FFN A/B (three seeds, paired) | `docs/reviews/2026-11-10-gated-ffn-ab.md` |
| every other review, by date | `docs/reviews/` — 145 files, one per lane |
| the literature notes behind each choice | `docs/research/` |
| the protocol a number must satisfy | `docs/protocols/ab-protocol.md` |

---

## References

- **Bits per byte as a byte-level metric** — MegaByte, [arXiv:2305.07185](https://arxiv.org/abs/2305.07185); BLT, [arXiv:2412.09871](https://arxiv.org/abs/2412.09871)
- **Scaling-law form used for projections** — Hoffmann et al., *Chinchilla*, [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- **Muon** — [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) · **Polar Express** — [arXiv:2505.16932](https://arxiv.org/abs/2505.16932)
- **BitNet b1.58** — [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) · **SCT** — [arXiv:2604.00733](https://arxiv.org/abs/2604.00733)
