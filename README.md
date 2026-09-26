# dormouse

Byte-level language-model trainer in Rust, built on burn 0.22.0-pre.4 with the
cubecl CUDA backend. The design point is training the largest useful model that
fits on one workstation: 16 GB of VRAM plus 64 GB of host RAM, with host RAM as
a latency-tolerant extension of VRAM rather than a limit on parameter count.
Precision is per-component, not blanket: fp32 where correctness requires it
(GEMM accumulate, logits, gates, TSCT masters), fp8/bf16/int8 where a verified
A/B says the bytes are wasted.

North star: a model that can code and code its own updates — patches enter the
repo only through the verification harness (tests + bench + A/B). Every speed
and size win below buys that loop: faster steps are more RLVR rollouts per
wall-clock hour.

## Model

- **PonderNet adaptive compute.** A single shared `LoopBlock` runs up to
  `max_iter` iterations per sequence (default 4 — measured 2.13x faster than 8
  with better stability); a halt head learns when to stop.
- **Attention.** KDA gated-delta (linear attention, burn-kda) as the primary
  arm. MSA top-k sparse attention (burn-msa) is **disabled on the pre.4
  stack** — its indexer feeds garbage indices and gathers read out of bounds
  (ADR-0012); it returns only with a rewritten kernel and a won A/B.
- **Engram n-gram memory.** FNV-hashed n-gram embedding tables. Tables live in
  host RAM at millions of rows; only the current batch's rows (~600 KB) touch
  the GPU each step.
- **Spectral experts.** Feed-forward experts are low-rank TSCT linear layers
  (burn-spectral); masters stay fp32, the forward reads fp8 on sm_120, and
  per-step polar retraction keeps factors orthonormal with a one-way fp32
  fallback if drift exceeds threshold.
- **Auxiliary heads** (configurable weights): JEPA masked latent prediction
  against an EMA teacher with KoLeo, and a DSpark draft head for next-K
  prediction. Both face a pending vs-pure-CE A/B — A/B or death applies.
- **Optimizer.** Muon+ (burn-muon-plus) on 2D weight matrices with routing
  validation, AdamW on embeddings/heads/1D, Adan as the alternative fallback.

## Repo layout

- `crates/dormouse-core` — the model: config seam (flat TOML presets in
  `configs/`, schema defaults mirror `small`), LoopBlock, attention, Engram,
  experts, aux heads, the fused CUDA kernel experiment.
- `crates/dormouse-data` — streaming byte pipeline: ring buffer, seeded
  shuffle, loud failures on bad corpora, plus `bin/filter.rs` (DCLM-style
  filtering with a two-region mode that structurally excludes the eval tail
  from training data).
- `crates/dormouse-train` — training loop: Muon+ mixed optimizer, burnpack
  checkpoints, resume with a config-drift check, host-RAM offload, stress
  protocol.
- `crates/dormouse-cli` — `train`, `generate`, `serve`.
- `vendor/` — three patched cubecl crates (runtime, server, cuda) carrying the
  slot-preserving memory-pool fix; mapped through `[patch.crates-io]`.

## Doctrine

1. **A/B or death** (ADR-0002): every mechanism beats its own removal on
   held-out BPB at fixed steps, or it is deleted. The same audit applies to
   code: every line is held up by a recorded verdict
   (`docs/audit-2026-09-25.md`).
2. **Loud failures** (ADR-0011): data that does not exist stops the run — it is
   never synthesized; shape mismatches assert; NASA P10 Rule 5 sets the
   assertion density; `--guard` is the recovery action.
3. **The bench gate is alive**: `scripts/bench.sh canary|flagship` appends to
   `benches/history.tsv`; a regression against the previous row blocks the
   merge. Verdicts land as TSV rows, not prose.
4. **One heavy thing at a time**: ram-guard service, MemoryMax cgroup caps,
   slot caps, persistent logs in `~/logs`. The desktop survived three freezes
   to learn this.

## Data

The training corpus is DCLM-filtered (46.2 GB raw → ~20 GB kept). Eval is a
carved tail that is excluded from the filtered corpus at filter time
(two-region mode, straddler documents dropped — ADR-0010): held-out numbers
measure generalization, not memorization. Cadence evals run on a 2 MiB slice
every 500 steps; the full tail is reserved for milestones.

## Build & run

Requires CUDA (developed on an RTX 5060 Ti, sm_120) and the vendored forks in
`vendor/` (nothing builds without them).

```sh
cargo build-train   # release CUDA binary, no OpenBLAS
./target/release/train --data <dir> --preset small --ckpt-name latest \
  --ckpt-dir checkpoints --eval <held-out-dir> --eval-every 500
./scripts/bench.sh canary
```

Every knob is a typed CLI flag: `./target/release/train --help`. Notable:
`--no-kda` / `--no-msa` / `--no-engram`, `--quant` (TSCT factor format),
`--act-quant`, `--opt`, `--engram-ram --engram-slots N`, `--host-adam-every`,
`--guard`, `--log`, `--detach`. `generate` and `serve` load checkpoints from
the same `--ckpt-dir`/`--ckpt-name` pair.

## Status (2026-09-26)

On `main`: burn/cubecl 0.22.0-pre.4 (the memory-pool stale-page fix lives in
`vendor/cubecl-fix` as slot-preserving pools), presets `use_msa = false`, the
config seam with snapshot+drift check, loud-failure data pipeline, corpus v2.
Open fronts, in order: flip the fusion backend and A/B it against the canary
baseline; the fused-kernel rewrite rungs (ADR-0009 — direct adjoints seeded
from saved buffers, hand matmuls replaced by autotuned ones, kill switch at
parity); the official 100k-step baseline on corpus v2; the knife A/Bs (aux
heads, GR, rank). The minimal-design path is written down in
`docs/design-minimal.md` (~-60% LOC with zero functionality loss, sequenced at
verdict moments).

## Docs

- `AGENTS.md` — agent-facing repo state: measured numbers, GPU quirks, knobs.
  Read before touching training code.
- `docs/PLAN.md` — the program plan, phase ladder, verdicts, north star.
- `docs/adr/0001..0012` — every recorded decision (config seam, A/B doctrine,
  fused verdicts, context ladder, research verdicts, scaling, reasoning,
  fused rungs, corpus v2, loud failures, MSA).
- `docs/audit-2026-09-25.md` — the codebase verdict audit (kill list, P10
  violations, missing A/Bs).
- `docs/design-minimal.md` — the minimal-architecture target and its math.
- `docs/fused-verification-2026-09-23.md`,
  `research/2026-09-25-fused-rewrite-plan.md` — the fused story: verification,
  the measured loss, the rewrite plan.
- `research/` — research notes (per-GB SOTA, fast-training kernels, NASA/burn
  practices).
- `bf16_KERNEL_PLAN.md`, `POST_TRAINING.md` — in Russian: bf16 kernel rules,
  and the post-training plan (SFT, RLVR, distillation, self-evolve).
