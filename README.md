<div align="center">

<img src="site/media/readme-banner.png" alt="dormouse" width="1200">

# 🐹 dormouse

**Train a real model on your own computer.**

The goal is a cheap but capable chatbot: a model you can download, run locally,
and afford to train yourself in days instead of months. What we maximise is
**intelligence density** — ability per byte of file, per FLOP, per hour of
training — so the project is the research that buys it: every technique that
makes a model cheaper and smarter, measured on one consumer GPU, and thrown out
when it does not pay.

Byte-level input (256 symbols, no tokenizer), trained in the format it ships in,
with one rule for every mechanism: *it must beat its own removal on held-out
bits per byte, at the same budget.*

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

The project fixes the budget first and then asks what fits inside it: **one card, one file size**. Every architectural change
is measured the same way — held-out bits per byte, against the same model with
that change removed. If it does not move the loss, it does not ship, however
elegant it looks.

That is why the recipe is small: eleven mechanisms were measured and dropped,
and the numbers that decided it are in the repository, not in a blog post.

## How a result is stated

A number here is always three things: **a loss, a window, and a budget**.

> *Held-out bits per byte on a fixed English window, after one pass over 2.6 GB
> on a single RTX 5070 Ti, compared against the same model with the mechanism
> removed.*

We report the **comparison**, not a trophy: `−0.0145 BPB at 0.00% extra
parameters` is a result; "1.19 bits" is a snapshot that the next architecture
will move. Where a raw figure appears in `docs/measurements.md`, it is dated and
labelled as the record for that date, not as the model's identity.

## The method, in one table

| lever | measured against | verdict |
|---|---|---|
| gated FFN (SwiGLU, `d_ffn = 2/3` of the GELU width) | the GELU block at equal parameters | **−0.0145 BPB, 0.00% params** → shipped |
| batch 24 + static `torch.compile` | the naive loop | **1.43× the step**, loss unchanged |
| Muon (5-step Newton–Schulz, bf16) | AdamW on the same shapes | **1.92×** the step at equal quality |
| MoE, 32 experts | dense at the same active FLOPs | +0.004 BPB → deferred until the corpus is bigger |
| low-rank at r=64 | the dense layer at the same budget | 1.95× faster, **+28% loss** → refused |
| attention factorisation | the dense attention | 2.57×, **+27%** → refused |

The pattern is the project's whole argument: at a fixed budget, speed comes from
writing the same function cheaper, never from dropping work.

## Quick start

```bash
cd port
uv sync

PYTHONPATH=src .venv/bin/python -m dormouse.cli train configs/arch_gated_swiglu.toml \
  --data ../data_local/best/fwedu_train.bin \
  --eval ../data_local/best/fwedu_eval.bin \
  --set mix_plan=configs/mix_refetch.toml \
  --set steps=4880 --set batch=24 --set seq_len=1024 \
  --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
  --set ns_dtype=bf16 --set muon_state_dtype=bf16

cd .. && bash scripts/gate_fast.sh
cd port && PYTHONPATH=src .venv/bin/python -m pytest -q -m "not slow and not pins"
```

Data is not committed. `docs/guides/data-provenance.md` records every source,
revision and licence; `port/scripts/refetch_pretrain.py` rebuilds the slice.

## What is in the tree

| path | what |
|---|---|
| `port/` | the live Python lane: config, model, trainer, post-training, tests |
| `docs/` | handbook, measurements, decisions, research, archive |
| `site/` | landing page, the pitch deck, and the Manim clips |

The Rust/burn tree and the vendored fused-kernel library are not on this
branch: they live on **`rust-lane`** (split 2026-11-13, history intact), so this
tree builds only Python.

## Where to read next

- **[`docs/HANDBOOK.md`](docs/HANDBOOK.md)** — one document, the whole project.
- **[Docs site](https://sehaxe.github.io/dormouse/docs/)** — the same, rendered.
- **[`docs/measurements.md`](docs/measurements.md)** — the dated record of what
  was measured, including what was rejected.
- **`AGENTS.md`** — the rules an agent (or a person) must follow when changing
  this repo.

## License

MIT — see [`LICENSE`](LICENSE). Model weights and corpus slices are separate:
the data carries the licences listed in
[`docs/guides/data-provenance.md`](docs/guides/data-provenance.md).
