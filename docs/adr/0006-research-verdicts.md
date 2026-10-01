# Architecture verdicts from the 2026-09-21 survey

Evidence base: `docs/research/2026-09-21-per-gb-sota.md`. Keep: weight-shared LoopBlock, Engram (input side only), Muon+ with update-RMS matching, JEPA at 0.05, DSpark, ~3:1 linear:full blend, bf16 storage with fp32 heads. Knife via A/B: PonderNet halt head (measured ~0 at controlled readout, arXiv:2608.22347), KoLeo (no LM-scale evidence), fp8-forward-by-default (precision laws, arXiv:2411.04330), batch-size warmup. Pending one A/B each: MSA at short context (arXiv:2502.11089), TSCT vs plain small experts (arXiv:2409.02060), energy head (arXiv:2507.02092). Never: output-side lookup tables (arXiv:2501.16975), expert-choice routing, sub-4-bit training.

This ADR exists so future reviews do not re-suggest the knife list without new evidence.
