<div align="center">

<img src="site/media/readme-banner.png" alt="dormouse" width="1200">

# 🐹 dormouse

**A byte-level language model trained from scratch on one consumer GPU.**
95M parameters · no tokenizer · **1.19 bits per byte** · 8.4 hours on one 16 GB card.

[![CI](https://github.com/sehaxe/dormouse/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sehaxe/dormouse/actions/workflows/ci.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org)
[![CUDA](https://img.shields.io/badge/CUDA-sm_120-green.svg)](https://developer.nvidia.com/cuda-gpus)

**[Landing page →](https://sehaxe.github.io/dormouse/)** · **[Docs →](https://sehaxe.github.io/dormouse/docs/)**

</div>

---

## What this is

A byte-level model reads raw bytes — no tokenizer, no vocabulary, no language
assumption. The same weights handle English, Russian, Python and binary.

dormouse takes that idea and pushes on one axis: **how much model fits in one
GPU-hour and one file**. The recipe is a dense pre-LN transformer with a gated
FFN, trained with Muon on a single RTX 5070 Ti, and shipped in a ladder of
formats — bf16, 4-bit activations, ternary weights.

## Results

Held-out bits-per-byte on a fixed English window (`best/fwedu_eval.bin`,
196 608 bytes). A BPB is only comparable within its own window.

| run | BPB | corpus | time | note |
|---|---:|---:|---:|---|
| **95M gated FFN, bf16** | **1.1871** | 2.64 GB | 8.4 h | current best |
| 95M gated FFN, ternary weights | 1.2220 | 2.64 GB | 9.5 h | **82.4 MB file**, +2.9% |
| 95M dense GELU, bf16 | 1.1940 | 2.00 GB | 6.6 h | previous best |
| xz (same window) | 2.31 | — | — | classic compressor |
| uniform bytes | 8.00 | — | — | by definition |

1.19 bits per byte is **6.7× compression**, and 1.1 bits/byte better than xz on
the same text. Seed noise is ±0.011 BPB, so anything smaller is not a result.

## Speed

- **85 kB/s** training throughput at batch 24 on one 5070 Ti — 49.7 TFLOP/s,
  within 1% of the card's practical ceiling.
- The full recipe (batch 24 + `torch.compile`) is **1.43×** the naive one.
- Ternary weights export to **82.4 MB** (2.31× smaller than bf16), and the
  packed format is what the inference path will consume.

## Quick start

```bash
cd port
uv sync

# train the live recipe on the bundled slice
PYTHONPATH=src .venv/bin/python -m dormouse.cli train configs/arch_gated_swiglu.toml \
  --data ../data_local/best/fwedu_train.bin \
  --eval ../data_local/best/fwedu_eval.bin \
  --set mix_plan=configs/mix_refetch.toml \
  --set steps=4880 --set batch=24 --set seq_len=1024 \
  --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
  --set ns_dtype=bf16 --set muon_state_dtype=bf16

# gates before a commit
cd .. && bash scripts/gate_fast.sh
cd port && PYTHONPATH=src .venv/bin/python -m pytest -q
```

Data is not committed. `docs/guides/data-provenance.md` records every source,
revision and licence, and `port/scripts/refetch_pretrain.py` rebuilds the slice.

## The recipe

| piece | choice | why |
|---|---|---|
| block | dense pre-LN transformer, 12 layers, `d_model=832` | beat the loop/KDA arms at equal bytes |
| FFN | gated (SwiGLU, `d_ffn=2048`) | −0.0145 BPB at **0.00%** extra parameters |
| optimizer | Muon (5-step NS) for matrices, AdamW for the rest | 1.92× the step at equal quality |
| precision | bf16 masters, bf16 NS, fp32 accumulate | the smallest format that still learns |
| formats | bf16 · 4-bit activations · ternary weights | one ladder: train → export → deploy |

## Roadmap

1. **2B stored / ~180M active** — low-rank experts + ternary → **255 MB** file.
2. **12 000 tok/s** locally: 45 MB read per token instead of 190 MB.
3. **PyTorch-code specialist** — 22 GB curated corpus, execution-verified traces.

## Repo map

| path | what |
|---|---|
| `port/` | the live Python lane: config, model, trainer, post-training, scripts, tests |
| `crates/` | frozen Rust/burn lane (formula source; not the live path) |
| `docs/` | research, reviews, architecture notes, ADRs, guides |
| `site/` | landing page and the pitch deck |
| `configs/` | flat TOML presets, one per experiment |

## Honest notes

- The window changed on 2026-11-08 when the corpus was refetched; numbers before
  and after that date are not comparable and each table entry names its window.
- The 12 000 tok/s figure is a bandwidth calculation for the planned 2B model,
  not a measured decode of today's 95M.
- Ternary costs +2.9% BPB at this scale; it is a deployment format, not a free
  win, and the export path is what makes it useful.

## License

MIT — see [`LICENSE`](LICENSE). Model weights and corpus slices are separate:
the data carries the licences listed in
[`docs/guides/data-provenance.md`](docs/guides/data-provenance.md).
