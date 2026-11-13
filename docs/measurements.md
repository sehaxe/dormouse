# Measurements

The dated record. A number here is always three things: **a loss, a window, a
budget** — and it is a snapshot, not the model's identity.

The live, maintained view is the docs site: <https://sehaxe.github.io/dormouse/docs/measurements/>.

| run | held-out BPB | corpus | time | log |
|---|---:|---:|---:|---|
| 95M gated FFN, ternary (QAT from step 0) | 1.2220 | 2.64 GB | 9.5 h | `logs/arch/run2640_ternary.jsonl` |
| 95M gated FFN, bf16 control | 1.1871 | 2.64 GB | 8.4 h | `logs/arch/run2640_s1.jsonl` |
| 95M dense GELU, bf16 | 1.1940 | 2.00 GB | 6.6 h | `logs/arch/run2gb_s1.jsonl` |
| xz, same window | 2.31 | — | — | reference compressor |

Window: `data_local/best/fwedu_eval.bin`, 196 608 bytes, `off=0`. It changed on
2026-11-08 when the corpus was refetched; anything before that date is a
different measurement. Seed noise is ±0.011 BPB.

Every arm that decided the recipe is in `docs/records/reviews/` — start with
`2026-11-09-architecture-matrix.md` (all arms, all seeds) and
`2026-11-12-ternary-artifact.md` (the format's cost, and two corrections to
earlier claims).

## References

- MegaByte — [arXiv:2305.07185](https://arxiv.org/abs/2305.07185) · BLT — [arXiv:2412.09871](https://arxiv.org/abs/2412.09871)
- Chinchilla — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Muon — [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) · Polar Express — [arXiv:2505.16932](https://arxiv.org/abs/2505.16932)
- BitNet b1.58 — [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) · SCT — [arXiv:2604.00733](https://arxiv.org/abs/2604.00733)
