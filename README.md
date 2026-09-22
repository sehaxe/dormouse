# dormouse

Byte-level language-model trainer in Rust, built on burn 0.22.0-pre.3 with the
cubecl CUDA backend. Everything — embeddings, weights, activations — is stored
in bf16 with fp32 accumulation where numerics require it. The design point is
training the largest useful model that fits on one workstation: 16 GB of VRAM
plus 64 GB of host RAM, with host RAM used as a latency-tolerant extension of
VRAM rather than a limit on parameter count.

## Model

- **PonderNet adaptive compute.** A single shared `LoopBlock` runs a variable
  number of iterations per sequence; a halt head learns when to stop.
- **Mixed attention.** Each iteration blends two attention arms with a learned
  gate: KDA gated-delta (linear attention, burn-kda) and MSA top-k sparse
  attention (burn-msa).
- **Engram n-gram memory.** FNV-hashed n-gram embedding tables (burn-engram).
  Tables can live in host RAM at millions of rows and train with CPU Adam;
  only the current batch's rows (~600 KB) are copied to the GPU each step.
- **Spectral experts.** Feed-forward experts are low-rank TSCT linear layers
  (burn-spectral); factors carry a configurable quantization format and are
  kept orthonormal by per-step polar retraction.
- **Auxiliary heads** (on by default, weights configurable): JEPA masked
  latent prediction against an EMA teacher with KoLeo regularization, and a
  DSpark draft head for next-K-token prediction.
- **Optimizer.** Muon+ (burn-muon-plus) on 2D weight matrices, AdamW on
  embeddings, heads and 1D parameters; Adan is available as an alternative
  fallback group.

## Repo layout

- `crates/dormouse-core` — the model: config presets (`nano`, `small`,
  `swift50`, `base`, `one_b`), LoopBlock, attention, Engram, experts, aux
  heads, fused CUDA kernels.
- `crates/dormouse-data` — streaming byte-level data pipeline: ring buffer,
  seeded file shuffle, deterministic train/eval split; text, parquet, images,
  binaries.
- `crates/dormouse-train` — training loop: Muon+ mixed optimizer, burnpack
  checkpoints and resume, host-RAM offload, stability stress protocol.
- `crates/dormouse-cli` — binaries: `train`, `generate`, `serve` (HTTP
  inference server).

## Build & run

The workspace patches cubecl and the burn-* crates to local forks; see
`[patch.crates-io]` in the root `Cargo.toml`. Without those directories
nothing builds.

```sh
cargo build-train   # release CUDA binary, no OpenBLAS
./target/release/train --data <dir> --preset small --ckpt-name latest --ckpt-dir checkpoints
```

Every knob is a typed CLI flag: `./target/release/train --help`. Notable ones:
`--bf16`, `--quant` (TSCT factor format), `--opt`, `--engram-ram`,
`--host-adam-every`, `--eval`/`--eval-every`. `generate` and `serve` load a
checkpoint from the same `--ckpt-dir`/`--ckpt-name` pair.

`--guard` re-execs the process on NaN or panic with a fresh CUDA context and
resumes from the last checkpoint, so long runs survive transient instability
without an external restart loop.

## Hardware target

16 GB VRAM GPU (developed on an RTX 5060 Ti, sm_120) and 64 GB host RAM. Rough
budget: a 1B-parameter model is ~2 GB of bf16 weights plus ~2 GB per
batch-16 s1024 step; n-gram memory scales beyond VRAM via `--engram-ram`.

## Docs

- `AGENTS.md` — agent-oriented repo state: measured numbers, hardware quirks,
  full knob list. Read it before changing training code.
- `bf16_KERNEL_PLAN.md` and `POST_TRAINING.md` — in Russian: bf16 kernel
  rules, and the post-training plan (SFT, RLVR, distillation, self-evolve).
- `research/` — research notes.
