<p align="center">
  <img src="assets/banner.svg" alt="dormouse — a byte-level Mixture-of-Experts language model trained end to end on one consumer GPU" width="1200">
</p>

# 🐹 dormouse

[![ci](https://github.com/sehaxe/dormouse/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/dormouse/actions/workflows/ci.yml)

**English** · [Русский](README.ru.md)

A byte-level Mixture-of-Experts language model, trained end to end on **one consumer GPU** (RTX 5060 Ti, 16 GB, $429). No tokenizer, no cluster, no distillation teacher — the whole recipe runs on the card, and every claim in this repository carries its measurement.

The Rust/burn original this grew out of lives on the [`rust-legacy`](../../tree/rust-legacy) branch. `main` is the Python project.

**Contents** — [What is this](#what-is-this) · [Results](#results) · [Quick start](#quick-start) · [Architecture](#architecture) · [Documentation](#documentation) · [License](#license)

## What is this

- **Byte-level**: vocabulary of 256 bytes. No BPE, no tokenizer tax on Russian, code, or anything else the tokenizer did not see in training.
- **Mixture-of-Experts with dropless routing**: 1024 experts, top-k 2, no capacity drops — every token keeps every expert it routed to.
- **KDA linear attention** (Kimi-style chunked gated delta rule, via FLA): O(n) context, a fixed 256 KB recurrent state at *any* context length — no KV cache to blow up.
- **TSCT low-rank experts**: every expert is `U·diag(s)·Vᵀ` (rank 64), which is what lets a 1.07B-parameter model fit, train, and ship on 16 GB.
- **Engram memory**: 402M hash-table "parameters" that live in host RAM, outside the compute graph.
- **A weight-shared loop**: three passes through the same block per token (the measured optimum; four was a tie, two was worse).
- **The full pipeline in one CLI**: `train → bench → sft → rl → serve`, with a knowledge base rendered from the repo's own docs.

The presets: `small` (9.5M, the A/B polygon), `medium` (138M, the current workhorse), `b1` (1.07B total / 57M active, the flagship).

## Results

Measured, dated, reproducible — the numbers below are the state of 2026-10-19.

| what | number |
|---|---|
| **medium pretrain** (`mix1`, web+wiki+ru+math mixture) | **BPB 1.936** held-out (196,608 B window) after 1.47 GB seen |
| speed, medium (b24 × s1024) | **40.9k tokens/s**, ~12 GiB peak, one RTX 5060 Ti |
| speed, b1 (b8 × s512) | ~8k tokens/s eager, 13 GiB |
| first downstream numbers (zero-shot, n=1000, random = 25%) | HellaSwag **30.5%**, ARC-Easy **27.2%**, synth-math 24.0% — at 0.37B tokens seen |
| test suite | **861 tests**, ruff clean |

The HellaSwag row is the one worth reading twice: it is ahead of GPT-2-small (31.4%) on **24× fewer training tokens**, and ahead of Pythia-160M (30.3%) on ~800× fewer.

Every mechanism in the tree was admitted by a 3-seed A/B on a fixed protocol, and the losers are recorded, not deleted quietly: the ColRow fix (−0.043 BPB, the largest single win), depth-3 (−0.007), dropless routing, Triton Newton–Schulz (52 dispatches → 1), optimizer CUDA-graph (−11.7 ms/step), bf16 Muon momentum (−1.23 GiB), Engram in host RAM (bit-identical). Rejected by measurement: NorMuon (+0.072), schedule-free Muon (+0.043), SIGReg (+0.148), LOOM (tie), MSA at short context (tie), ternary weights from scratch (+0.04), KDA recompute flag (tie).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                # CPU environment; tests run here
uv sync --extra gpu    # the measured stack: torch 2.13.0+cu129 + FLA (sm_120)

uv run pytest                       # the suite (861 tests; CUDA gates self-skip)
uv run ruff check .                 # lint

# train a model from a byte corpus
uv run dormouse train medium --data corpus.bin --eval held.bin \
  --device cuda --steps 60000 --batch 24 --set seq_len=1024 \
  --set engram_ram=true --set moe_capacity=0 --set moe_kernel=triton \
  --ckpt-name mix1 --out-dir checkpoints/mix1

uv run dormouse bench --ckpt-name mix1 --out-dir checkpoints/mix1 \
  --tasks arc_easy,hellaswag --limit 1000     # downstream, byte-scored

uv run dormouse sft medium --data sft.bin --build-sft \
  --chat-data pairs.jsonl --init-from checkpoints/mix1/mix1.pt \
  --ckpt-name sft11 --out-dir checkpoints/sft

uv run dormouse serve --out-dir checkpoints/sft --ckpt-name sft11 --port 3783
```

`dormouse --help` lists every command; `dormouse info <preset>` prints a parameter budget table. Every knob is a typed flag or a `--set key=value` that refuses to load when the key is unknown — a config that lies is a bug, and the schema is strict on purpose.

## Architecture

```
src/dormouse/
├── config.py        strict schema; unknown keys and unported arms refuse loudly
├── instrument.py    the LOUD / COUNTED / SILENT degradation layer
├── model/
│   ├── model.py     DormouseModel: embed → loop → norm → head
│   ├── loopblock.py the weight-shared loop; adaptive depth; MSA gate
│   ├── kda.py       KDA attention (FLA chunked kernel; SDPA fallback, counted)
│   ├── experts.py   dropless MoE: router, LFB balancing, padded/Triton dispatch
│   ├── tsct.py      U·diag(s)·Vᵀ expert factors + the storage ladder (fp8→ternary)
│   ├── engram.py    hash memory, in-VRAM or host-RAM (bit-identical)
│   ├── byteflow.py  the byte patcher (local→global→decoder)
│   └── sparse/      MSA: block-sparse attention + the fused Triton gather
├── train/           loop, WSD schedule, Muon+AdamW, clip, SFT, mixture streaming
├── post/            verifiers (code/math), rollouts, CRR advantage, the RL stepper
├── bench/           downstream eval: ARC-Easy, HellaSwag, synth math (byte-scored)
├── selfplay/        the zero-data self-play core (UTM + gradient-alignment reward)
└── cli.py           train · bench · sft · rl · serve · decode · plot · info
```

Three things are custom because nothing in the ecosystem does them: the TSCT quantized experts, the Engram host tables, and the ByteFlow patcher. Everything else — attention, optimizer, quantization formats, fused kernels — is the best available dependency, and the places where that choice was made against a homegrown alternative are recorded in `docs/`.

## Documentation

The knowledge base is built from this repository's own documents and deployed with the site:

- **The docs site** (`docs-site/`): a Starlight site over `docs/`, `research/` and the working notes; `npm run build` in `docs-site/` to render it locally.
- **`docs/decisions/`**: the dated plans and ADRs — the 1B readiness plan is the map.
- **`docs/research/`**: the literature work, every claim carrying its arXiv id.
- **`docs/reviews/`**: the audits, including the ones that found our own bugs.
- **`AGENTS.md`**: the rulebook this repository is actually run by.

## License

MIT — see [LICENSE](LICENSE).
