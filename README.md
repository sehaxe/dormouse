<div align="center">

# 🐹 dormouse

**A byte-level language-model trainer that fits on one workstation — published
with the measurements that justify it and the ones that refute it.**

[![CI](https://github.com/sehaxe/dormouse/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/dormouse/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-sm_120-green.svg)](https://developer.nvidia.com/cuda-gpus)
[![burn](https://img.shields.io/badge/burn-0.22.0--pre.4-red.svg)](https://github.com/burn-rs/burn)

Rust · burn + cubecl · one RTX 5060 Ti (16 GB, sm_120) · 5 140 lines in
`dormouse-core`, 14 437 across the six workspace crates, 28 crates of vendored
fused kernels

</div>

---

## STATUS

Read this section first. Everything below it is context; this is the part that
tells you whether the project is worth your time. Four of its own claims were
retracted on 2026-09-27, so nothing here is presented as verified unless the
verdict and the artifact are both named.

**Machine for every number below** unless stated otherwise: 1 × RTX 5060 Ti
16 GB (sm_120, consumer Blackwell, 170 SM, 1792 GB/s), 62 GB RAM, Linux,
rustc 1.98.1, burn 0.22.0-pre.4 + vendored cubecl.

### (a) What is measured

| claim | number | when / shape / how |
|---|---|---|
| Fused gated-delta (KDA) kernel, forward | **1.212 ms** vs `torch.compile` 54.5 ms = **45×**; vs PyTorch eager-best 160.1 ms = 132× | 2026-09-27. b=10, t=512, h=12, K=V=64, chunk=16, fp32, 10 runs × 200 iters, real `a_log=-3` decay. torch 2.11.0+cu128. [`research/2026-09-27-pytorch-baseline.md`](research/2026-09-27-pytorch-baseline.md) §1 |
| …same op, forward+backward | **25.9 ms** vs PyTorch eager-best 290.3 ms = **11.2×**; 138× fewer allocations, 2.5× less VRAM, 32 % of this card's memory roofline | same doc, fused as one autodiff node (the bare-kernel 45× is not the fwd+bwd number — do not quote it as one) |
| f16 GEMM, our path vs cuBLAS | **7.4× behind** (0.389 ms / 41.4 TFLOP/s cuBLAS vs 2.880 ms / 5.6 TFLOP/s ours) | 2026-09-27, `[5120,768]×[768,2048]`, zero-copy on cubecl's own stream, `max rel err 2.7e-4`. `crates/cublas-poc`; log `~/logs/cublas_poc_2026-09-27.log` |
| fp32 GEMM, our path vs cuBLAS | **no ratio claimed** — 4.8 / 47.1 / 47.7 / 51.7 / 71.1 ms for one fixed matmul in the *same binary* (15× spread, power-capped) | 2026-09-27. At the median we are 4.4× behind; at the fastest sample 2.3× ahead. The measurement does not resolve it, so neither number is published as fact |
| Training step, batch 10 × seq 512 | **1.58–1.84 s/step** over steps 1050–1500 of the best run (1.86–1.94 s over steps 50–100 of the same run). ~465 ms of it is fixed per-step launch cost, measured as `t = 465 ms + 0.067 ms/token` | `--timers`, `~/logs/official_v5e.log`, 2026-09-27 |
| **Best held-out BPB ever recorded** | **6.351** at step 1500, on a fixed 100 KB window (`--eval-batches 20`) | 2026-09-27. A 7 526 223-param model over a 19 GB sharded corpus, `~/logs/official_v5e.log`. **That parameter count is the pre-2026-09-27-pricing `small`; the shipped `small` is 9.20 M, so the best number on record was not produced by a preset in `configs/` today.** Anchors on comparable windows: uniform 8.000, unigram 5.17, **5-gram+backoff 2.572**. See (d) |
| Reference-fidelity suite | 1000 cases, fused vs an independent transcription, 5e-4 absolute, fixture regenerable byte-identically in CI | `vendor/burn-fused/crates/burn-gdn2/tests/bit_exact.rs`. Real work; **wrong provenance** — see (b) |
| Test inventory | 315 unit + 62 integration `#[test]` functions across 28 crates | counted from source 2026-09-27. Presence is not correctness — see (b) |

> **The raw training logs are not in this repository** (`~/logs/`, gitignored
> and outside the tree). Every number in this table is a transcription of a log
> that a stranger cannot read, a fixture that is committed, or a file:line in the
> tree. That is a real gap in the evidence, and it is the first thing to fix.

### (b) What is implemented but UNVERIFIED

- **0 of 28 library crates check any numeric output against an *authors' own*
  source code.** Six crates assert in their docs that they match an
  implementation; none of the six has a test wired to one. Full per-crate table
  with provenance and tolerances: [`research/2026-09-27-oracle-audit.md`](research/2026-09-27-oracle-audit.md).
  The honest sentence: *the library verifies fused kernels against our own
  tensor fallback, and transcriptions of papers against our own expectations of
  those papers.* That catches real drift. It is not "bit-for-bit against the
  reference", and no crate may say it is unless it names the external source.
- **28 tests cannot fail, report PASS having asserted nothing, or are
  unreachable by any CI job that has ever executed** (2 where the reference is a
  character-for-character copy of the code under test, 2 more where the expected
  value is a hand-copy that moves with the bug, 9 that silently skip or panic
  before asserting, 1 permanently unrunnable, 14 CUDA-executing on a runner that
  does not exist). Same audit, §3.
- **No mechanism in the model has an A/B verdict.** The protocol requires
  3 seeds per arm at 2000 steps ([`docs/AB-PROTOCOL.md`](docs/AB-PROTOCOL.md));
  the queue is written down and unrun. The Engram arm — 2.4 M memory parameters,
  24 % of `small`, on by default in every preset — has never been compared to
  `--no-engram`. Its capacity default cites a *third-party* curve (arXiv
  2601.16531, single-author preprint) measured on a 125 M backbone, 16× ours.
- **The fused KDA kernels are proven to launch from a benchmark and NOT proven
  to launch from a training step.** The dispatch gate was
  `TypeId::of::<B>() == TypeId::of::<Autodiff<CudaBare>>()`, which cannot match
  the trainer's `Autodiff<Cuda, BalancedCheckpointing>` — so every training step
  to date ran the tensor-ops path, where the same op costs 3628 ms fwd+bwd
  instead of 25.9 ms. A rewritten gate exists; no rebuilt training step has been
  shown entering it.
- **No fused multi-head attention exists in this stack at all.** dormouse's only
  attention arm is the gated-delta recurrence; `burn-cubecl` 0.22.0-pre.4 ships
  no fused attention kernel. Where PyTorch has one (mem-efficient, 0.656 ms at
  b=10 h=12 t=512) a burn-composite would be 5.980 ms — 9× slower.
- **Preset costs are asserted by a test, not by this file.** The parameter
  column in [Presets](#presets) is what `cargo test -p dormouse-core --test
  preset_exec` prints; that test is real and CI runs it, but it was not re-run
  for this README.

### (c) What is BROKEN or KNOWN-WRONG

- **bf16 matmul cannot work on this backend, and no flag will make it.**
  `restrict_to_llvm_backend` (`vendor/cubecl-fix/cubecl-cuda/src/runtime.rs:420-513`)
  deletes bf16 from the advertised element types because the pliron LLVM dialect
  this backend lowers through has `builtin.fp16`, `fp32`, `fp64` and nothing in
  between — a bf16 kernel compiles to something that quietly computes zeros. So
  `--bf16` stores bf16 and computes fp32 through cast copies, and **every bf16
  run on this box is slower than fp32**. The one primitive that works is bf16
  *storage* as `u16` bit patterns.
- **f16 matmul is correct but silently slow**: the tensor-core candidate dies at
  compile time (`dyn SizedType` unimplemented for pliron's `FP16Type`) and the
  autotuner falls back to a non-accelerated routine without a word. The fix is
  ~10 lines in `cubecl-ir` and belongs upstream; until it lands, cuBLAS's
  41–47 TFLOP/s is not reachable *through burn* — only through the POC.
- **`load_ckpt` used to wipe the quant format.** The format, bf16 compute and the
  one-way fp32 fallback live in non-param fields, so loading a checkpoint
  rebuilt the model from `new()` and dropped them: a resumed quantized run
  trained the fp32 forward while its own log said `quant format: Fp8`, and the
  fallback latch was lost with it. Fixed 2026-09-27, latch restored from a
  checkpoint flag ([ADR-0021](docs/adr/0021-resume-is-the-same-run.md)). **Any
  quantized result produced by a run that was ever resumed before that date is
  an fp32 result.**
- **Four claims this project made were retracted on 2026-09-27**, and they are
  listed here so nobody re-derives them from an older file: a `bit-for-bit`
  claim whose test had never run and whose reference fixtures were excluded by
  the crate's own `.gitignore`; a `bool → float` "backend bug" that is not a bug
  (the cast is correct on both backends — the miscount was `clone()` aliasing
  the device buffer under `mask_fill`, plus a firewall that did not skip the
  step); an `--act-quant fp4` "verification" whose attention path silently ran at
  8 bits (`ActFormat::attn()` maps `Fp4 → Int(8)`); and a "the drift check is
  inverted" claim whose check was never inverted.
- **PonderNet is gone.** The project's previous headline — adaptive compute
  that decides per sequence how deep to think — was measured and deleted
  ([ADR-0013](docs/adr/0013-fixed-depth.md)): λ-collapse made the reported loss
  fake (`p_dist → [9e-4, 4e-6, 1e-8, 2e-11]`, held-out frozen at exactly
  uniform), and the KL direction made collapse free. The loop now runs a **fixed**
  `max_iter` with an unweighted mean CE. Depth varies only under `--rand-depth`
  (training) and `--eval-depths` (measurement).
- **The most recent production run does not finish.** `official_v5` reached
  held-out BPB 6.351 at step 1500 and then went NaN at step 2000 on every one of
  its resume attempts (`~/logs/official_v5g.log`); the guard's restart chain is
  capped at 3. `checkpoints/` in this tree is empty.
- **The `engram:` startup line under-reports RAM by 2×**: it prints
  `total_rows × row_bytes` (table only) where the trainer also holds a full
  momentum buffer (`crates/dormouse-train/src/lib.rs:882-886`).
- **NaN on deep overfit is open.** It appears in any config on this box,
  including fp32 + AdamW, once the model overfits hard; `--no-kda` stays clean
  through deep overfit, so the KDA recurrence is the prime suspect. Never
  triggered on a real loss profile so far, never explained either.

### (d) Has any trained checkpoint ever beaten a 5-gram byte counter on held-out data?

**No. Not one, and not close.**

| model | held-out BPB | vs the counter |
|---|---|---|
| uniform byte | 8.000 | — |
| unigram counter | 5.17 | — |
| **dormouse, best run, step 1500 (2026-09-27)** | **6.351** | **+1.18 worse than the unigram counter** |
| 5-gram + backoff, ~24 lines of counting | **2.572** | **+3.78 worse** |

The 5-gram line is not a strawman: on the corpus the model is trained on, a
4-gram beats it by ~4.9 BPB, and on the text domains a 5-gram sits at 2.25–2.66
while the *agentic* slice collapses to 0.938 at 0.0 % unseen 5-gram contexts —
i.e. 21 GB of the available data is memorizable at order 5 and teaches format,
not language ([`research/2026-09-27-domain-anchors.md`](research/2026-09-27-domain-anchors.md)).

This is the number that matters, and it is the reason the project's own
milestone ladder reads "7.5 M beats a 24-line counter" as rung 1
([`research/2026-09-27-scale-ladder.md`](research/2026-09-27-scale-ladder.md)).
**Any capability claim about this model is premature until that number moves.**

### (e) What blocks a 1 B-parameter core model, in numbers

1 B parameters at Chinchilla's 20 tokens/param is **20 B tokens**. At the measured
7.5 M operating point (`1.58–1.84 s/step`, 5120 tokens/step, ~465 ms of that
fixed launch cost that does not grow with N) the same card does 0.5 B at
13.0 k tok/s and 1.5 B at 3.9 k tok/s in the only published benchmark of this
exact GPU (LLMQ, arXiv 2512.15306, 78–85 % MFU). Our own arithmetic, from
measured throughput (`research/2026-09-27-scale-ladder.md`):

| | 7.5 M (now) | 1 B |
|---|---|---|
| step time, fp32 | 1.6 s | ~16 s |
| step time, working f16 GEMM | 1.6 s | ~2.5 s |
| wall clock to 20 B tokens | **9 h** | **30–90 days**; ~11 days with both fixes |

Four blockers, all measured or code-verified:

1. **No tensor-core GEMM.** bf16 cannot work (see (c)); f16 is 7.4× behind
   cuBLAS. The zero-copy cuBLAS POC reaches 41.4 TFLOP/s on cubecl's own stream
   — it is a proof, not an integration, and `crates/cublas-poc` is not wired into
   the trainer.
2. **No optimizer-state offload for the backbone.** fp32 AdamW is 8 B/param =
   8 GB at 1 B, plus 4 GB of fp32 masters, on a 16 GB card that must also hold
   activations. LLMQ's recipe for this exact card is bf16 states with stochastic
   rounding, double-buffered host offload (it measured zero-copy as *bad* on
   5060 Ti/4090), and all allocations at startup. The host-offload machinery
   exists — but only for the n-gram tables.
3. **~465 ms of fixed per-step launch cost** dominates any model below ~80 M
   parameters, and burn's optimizers are functional, so a captured CUDA graph
   cannot be replayed without in-place parameter updates.
4. **The evidence base does not exist yet.** Per (b), no mechanism has an A/B
   verdict and no checkpoint beats a counter. Scaling a recipe that has not been
   shown to work at 7.5 M is optimizing the wrong term.

---

## Architecture

Four steps, in `crates/dormouse-core/src/model.rs:15-18`. The loop is run
`max_iter` times per sequence with **shared weights**; there is no per-depth
stack of layers.

```
 bytes 0..255
     │
     ▼
 ┌───────────┐   Embedding  [256, d_model]          model.rs:52
 └─────┬─────┘   the only place a byte becomes a vector
       │
       ▼
 ┌──────────────────────────────────────────────────┐
 │ LoopBlock  × max_iter   (shared weights)         │  loop_block.rs
 │                                                    │
 │   h_ctx ──RMSNorm──┐                              │
 │      controller ────┼─► w_attn  w_mem  w_ffn      │  one Linear,
 │                    └─► blend (softmax over       │  3 gates +
 │                          n_experts)               │  1 expert blend
 │      │                                             │
 │      ├─► KDA gated-delta  ──× w_attn ──┐         │  the recurrence
 │      │   (burn-kda, fused CUDA)         │         │  carried across
 │      ├─► Engram  ─┐                     │         │  iterations
 │      │   memory·min(w_mem,0.5)           │         │
 │      │   +  (1−min(·))·dense ──× w_mem ─┤         │  never more
 │      ├─► Σₑ blendₑ · FFNₑ (TSCT) ─× w_ffn         │  than half
 │      │                                     │      │  the branch
 │      └── ReZero scale (or GR write) ──────┘      │
 │                    │                              │
 │              per-iteration CE, uniform mean ──────┼──► rec
 └────────────────────┬─────────────────────────────┘
                      ▼
 ┌───────────┐   RMSNorm  fp32 even under --bf16    model.rs:220
 └─────┬─────┘   logits in fp32 or the NaN budget is gone
       ▼
 ┌───────────┐   lm_head (LinearLike, TSCT-capable)  model.rs:225
 └─────┬─────┘
       ▼
   next byte
```

One sentence per mechanism, and why it is there:

- **KDA gated-delta attention** (`shared_attn`) — the only attention arm. A
  recurrent delta rule with a learned per-channel decay, so the state is
  `[batch, heads, 64, 64]` and is *carried* across loop iterations instead of
  recomputed. The sparse-MSA arm was disabled by an out-of-bounds top-k gather on
  burn 0.22.0-pre.4 ([ADR-0012](docs/adr/0012-msa-broken-on-pre4.md)) and then
  cut along with its crate ([ADR-0014](docs/adr/0014-msa-cut.md)).
- **Engram hashed n-gram memory** (`engram`) — FNV-hashed rows of orders 2/3/4
  living in host RAM; deterministic addressing means the GPU only ever touches
  the batch's own rows (~600 KB/step). Mixed with a dense projection of the
  same hidden state under a *hard* floor `λ = min(w_mem, 0.5)`, because a
  memory branch that can carry more than half the signal starves the backbone of
  gradient — the failure that made nearly every run in this project's history
  get rescued by `--no-engram` instead of by a config.
- **TSCT low-rank expert FFNs** (`expert_ffns`) — fp32 masters, quantized
  forward, a real polar retraction every step to hold the factors
  orthonormal, and a one-way fp32 latch if they drift. Muon+ ColRow on the 2D
  maps, AdamW on embeddings/heads/1D, the n-gram table on Adam with no weight
  decay.
- **ReZero residual** (`residual_scale`, or `gr`) — one learned scalar per
  iteration on the sum of the three arms. The Gated Residual variant
  (`use_gr`, off by default for checkpoint compatibility) widens the residual to
  four branches with a low-rank sigmoid read.
- **Fixed depth + unweighted CE** — see (c). `--rand-depth` trains one model to
  be correct at every depth 1..`max_iter`; `--eval-depths` prints held-out BPB
  at each depth for free, which is how you find out whether the depth you pay
  for is being used.
- **Aux objectives, on by default** — JEPA masked-latent prediction against an
  EMA teacher (weight 0.05) + KoLeo, and a DSpark draft head for next-K
  prediction (weight 0.1, K = 4). Both have never been A/B'd against pure CE;
  `--jepa-weight 0 --dspark-weight 0` is arm 1 of the queue.

---

## Presets

Flat TOML in [`configs/`](configs/) — presets are data, not code, and the schema
defaults *are* `small`. Geometry below is read from the TOML; the memory-row
count is derived from the code (`engram_tables` rounds `engram_rows` up to a
power of two because the in-model read masks the hash — asserted at
`loop_block.rs:654`); the compute column is what
`cargo test -p dormouse-core --test preset_exec` prints on the instantiated
model, and the wide rows are behind `-- --ignored` because a full-width CPU
build costs minutes.

| preset | d_model | experts | max_iter | memory rows | total (compute) | what it is for |
|---|---|---|---|---|---|---|
| `nano` | 512 | 3 | 4 | 3.15 M | 7.19 M (4.05 M) | smoke tests |
| `nano-fused` | 512 | 3 | 4 | 3.15 M | 7.19 M (4.05 M) | `nano` with KDA+Engram+aux off, single-node path |
| `small` | 768 | 3 | 4 | 3.15 M | 9.20 M (6.05 M) | **the flagship on 16 GB** |
| `mor` | 768 | 3 | 4 | 3.15 M | 9.20 M (6.05 M) | `small` + Mixture-of-Recursions A/B arm |
| `base` | 1024 | 3 | 8 | 3.15 M | 13.07 M (9.92 M) | 24 GB-class work; OOMs on 16 GB at batch 6 |
| `swift50` | 1024 | 8 | 8 | 3.15 M | `-- --ignored` | many-expert experiments |
| `one_b` | 2048 | 4 | 12 | 3.15 M | `-- --ignored` | the 1 B dream |
| `p150` | 4096 | 4 (64 heads) | 12 | 3.15 M | ~160 M compute (159 720 195, per the header probe named in the TOML) | the 150 M-program flagship |

Two things that surprise everyone, both visible in that test's printed split:

- **Every preset pays the same 3.15 M memory rows** (3 tables × 32 768 rows × 32
  dims) — 24 % of `base` and 44 % of `nano`. The budget is a ratio, so the narrow
  presets pay it hardest.
- **`use_engram = false` does not shrink the model.** `nano-fused` ships the arm
  off and still allocates the full 3.15 M rows, because the tables are built
  unconditionally (`loop_block.rs:204`) and the flag is only a runtime switch.
  `--engram-ram` is what actually moves them out of the model: the config seam
  squeezes the in-model table to one row per order, because on that path the
  rows arrive pre-gathered from the host and the in-VRAM table is never read
  (`crates/dormouse-train/src/cfg.rs:50`).

---

## Quick start

**Prerequisites, all of them real:**

- **An NVIDIA GPU.** Developed and measured on sm_120 (RTX 5060 Ti, 16 GB).
  Without one, `cargo test --lib` still runs on the CPU backend.
- **CUDA toolkit + `mold`.** `.cargo/config.toml` links through mold; install it
  (`apt install mold`) or edit the `[target.x86_64-unknown-linux-gnu]` rustflags
  out. `CUDARC_CUDA_VERSION=12050` is pinned in the same file — a fresh target
  directory dies in cudarc's build script on CUDA 13.4 without it.
- **Rust stable** (developed on 1.98.1).
- **A byte corpus.** `--data` takes a *directory*; every readable file under it
  is streamed. A missing path or an empty directory is a hard error, by design
  (a silent filler once trained a model on constant bytes for 500 steps).
- **No OpenBLAS is needed** for the workspace crates — the CPU backend is
  burn-flex, whose matmuls go through the `gemm` crate. The GPU binary is built
  with `--no-default-features` so it never links BLAS at all. The
  `vendor/burn-fused` workspace is different: it enables burn-ndarray's
  `blas-openblas`, which **builds OpenBLAS from source** (a C/Fortran toolchain
  and several minutes). If one of its test binaries cannot find
  `libopenblas.so`, that is the reason:
  `export LD_LIBRARY_PATH=$PWD/vendor/burn-fused/target/release/build/openblas-src-*/out`.
- **RAM.** `--engram-ram` allocates `[slots, slots, slots]` tables at 32 dims
  (`crates/dormouse-train/src/lib.rs:869`), so the flag value is **per order**:
  8 M slots = 24 M rows = **6.1 GB**, 48 M slots = 144 M rows = **36.9 GB ≈ 37 GB**
  (f32 table + f32 momentum, 256 B/row × 2). A resumed run also reads the
  sidecar, so budget for the resumed figure and do not start one with a desktop
  running. ⚠ `AGENTS.md` §2.4 quotes ~25 GB for a resumed 48M-slot run. That
  derivation treats the flag as a *total* row count; the code triples it. Derive
  it yourself: `slots × 3 orders × 32 dims × 4 B × 2 buffers`.

```sh
git clone https://github.com/sehaxe/dormouse && cd dormouse

# build ALL THREE binaries. `cargo build-train` builds --bin train only, so
# `generate` and `serve` do not exist afterwards - a real trap, hit while
# writing this section.
cargo build --release -p dormouse-cli --bins --no-default-features --features cuda

# tests (CPU; no GPU needed)
cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib
cargo test -p dormouse-core --test preset_exec      # every preset: config, cost, and what EXECUTED

# a 2-step smoke run. Any preset works; `nano` is the cheapest.
mkdir -p /tmp/smoke && head -c 4000000 /path/to/corpus.bin > /tmp/smoke/corpus.bin
./target/release/train --data /tmp/smoke --preset nano \
  --steps 2 --seq-len 128 --batch 1 --ckpt-dir /tmp/smoke --ckpt-name smoke \
  --eval /tmp/smoke --eval-every 2 --eval-batches 2

# a real run (all knobs are typed flags: --help)
./target/release/train --data <corpus-dir> --preset small \
  --ckpt-name latest --ckpt-dir checkpoints \
  --eval <held-out-dir> --eval-every 500 \
  --engram-ram --engram-slots 8000000 --guard --detach --log /path/to/log

# inference from the same checkpoint pair
./target/release/generate --ckpt-name latest --ckpt-dir checkpoints --prompt "once" --steps 64
./target/release/serve   --ckpt-name latest --ckpt-dir checkpoints --port 8000
```

Long runs survive: `--guard` re-execs on NaN/panic with a fresh CUDA context and
resumes from the last checkpoint (capped at 3 restarts — a persistent failure
needs a human); `--detach` daemonizes in-process with `--log <file>`. Resuming
reuses `--ckpt-name`, and a resume whose resolved config differs from the
`<ckpt_name>.config.toml` snapshot is a hard error, except for `steps`,
`log_every` and `ckpt_every` — extending a run is a legal resume.

### Every flag on `train`

Transcribed from the real binary's `--help`, grouped. There are no other
runtime knobs except `DM_QUANT_DEBUG=1` (debug-only) — the autotuner level is
exposed as `--autotune` rather than left as an env var.

| group | flags |
|---|---|
| data / config | `--data` (required, a directory) · `--eval` · `--preset` (default `small`) · `--config` · `--set k=v` (repeatable) |
| schedule | `--steps` · `--seq-len` · `--batch` · `--log-every` · `--ckpt-every` · `--ckpt-name` · `--ckpt-dir` (default `checkpoints`) · `--eval-every` |
| optimization | `--lr` · `--wd` · `--grad-clip` · `--opt mix\|mix-adan\|adan\|adamw\|muon` · `--factors-fallback` · `--seed` |
| model arms | `--bf16[=true\|false]` · `--act-quant int4\|int8\|fp4` · `--act-group` · `--max-iter` · `--no-kda` · `--no-engram` · `--quant fp32\|bf16\|fp16\|fp8\|fp4` · `--retract-every` · `--retract-iters` · `--rand-depth` |
| aux objectives | `--jepa-weight` · `--jepa-k` is **not** a flag (K is `--dspark-k`) · `--dspark-weight` · `--dspark-k` · `--jepa-targets` · `--jepa-precompute` |
| memory offload | `--engram-ram` · `--engram-slots` (default 1 000 000) · `--host-adam-every` (default 1) |
| measurement | `--eval-batches` (default 20 = 100 KB) · `--eval-depths` · `--timers` · `--memlog` · `--quant-check` · `--autotune` |
| stability | `--stress` · `--stress-lr` · `--stress-every` |
| process | `--log` · `--detach` · `--guard` |

There is deliberately **no** `--no-gr` or `--no-mor`: those are config, not A/B
flags (`--set use_gr=false`, `--set use_mor=true`). `--rand-depth` and
`use_mor` are mutually exclusive and the config seam refuses the pair loudly.
**`--gen-max-iter` does not exist**, even though one flag's help string
advertises it (`crates/dormouse-cli/src/bin/train.rs:114`).

`generate` takes `--ckpt-dir --ckpt-name --prompt --steps --preset --config
--set --temp`; `serve` takes `--ckpt-dir --ckpt-name --preset --config --set
--port` (default 8000).

**What does not work, so you do not discover it the hard way:**

- `scripts/bench.sh` hardcodes this machine's corpus path
  (`/mnt/e43497ab-.../aria_data/pretrain/real_filtered_v2`) and exits 1 if the
  drive is not mounted. It is a local tool, not a portable gate.
- `vendor/burn-fused` is **not a git repository of its own** — it is a vendored
  copy inside this one, and the root `Cargo.toml` `exclude`s it (its crates are
  separate cargo workspace roots; without the exclude their `workspace = true`
  inheritance resolves against our root and fails). Consequence: the per-crate
  `.github/workflows/ci.yml` files it inherited (two survive in the tree after a
  2026-09-27 cleanup) have **never executed and cannot gate anything**. The real
  gate is
  [`.github/workflows/fused-library.yml`](.github/workflows/fused-library.yml),
  which runs cargo from *inside* the fork. Run it locally with
  `cd vendor/burn-fused && cargo test --workspace --exclude burn-fused-benches
  --exclude cpu-probe --exclude launch-probe`. Three of its files are currently
  untracked, including `burn-fused/tests/gpu_production_shape.rs`, so a fresh
  clone does not have them.
- There is **no GPU job in CI** and no registered runner — deliberately, because
  this workstation is simultaneously the trainer, the benchmark box and the
  agent workstation. The GPU check is a command you run on the GPU box:
  `vendor/burn-fused/tools/gpu-gate.sh`.

---

## Data pipeline

Streams bytes (text, parquet, images, binaries — a JPEG is just bytes to
predict) through a 64 MB ring refilled in 8 MB chunks, seeded Fisher-Yates
shuffled, split deterministically. A directory that yields no files, or a corpus
smaller than one batch, stops the run loudly; data is never synthesized.
`dormouse-data` also ships three binaries: `filter` (a DCLM-style corpus filter
with a two-region mode so the eval tail is excluded from training data at filter
time — straddler documents dropped, dedup shared, ADR-0010), `shard`, and
`anchors` (the n-gram counter that produces the bars in (d) — that is the tool
that keeps this project honest).

## Performance

The whole regression history is two rows. `benches/history.tsv`:

| date | commit | mode | step | throughput | final CE |
|---|---|---|---|---|---|
| 2026-09-25 | bdddbaf | canary (2 M slots, aux off) | 4081 ms | 1.23 KB/s | 5.141 |
| 2026-09-26 | b00ae06 | canary | 150548 ms | 0.03 KB/s | 5.074 |

`scripts/bench.sh canary` appends one. **It is a convention, not a gate**: no CI
job reads the file, and a regression does not block anything automatically — the
comparison is a human step. Reported here rather than implied.

## Working rules (CONTRIBUTING)

These are the rules this project actually runs on. They are not aspirations; each
one exists because breaking it cost a run, a week, or a claim.

1. **Every degradation carries one of three marks: LOUD, COUNTED, SILENT — and
   SILENT is a defect.** LOUD = a hard error naming the cause *and* the escape.
   COUNTED = the fallback happens and a counter or log line says so, so a reader
   of the training log sees it without a profiler. SILENT = it just happened.
   The reason this class is expensive here: a silent fallback usually computes
   the *right answer*, so the run is correct and a year slower. Missing data
   stops the run and is never synthesized (one run trained 500 steps on constant
   bytes); shape mismatches assert; `--guard` is the recovery action, not a
   substitute for the error. An arm that can be entered-but-skipped needs a
   counter — that is how the fused KDA dispatch stayed dead for a year behind a
   correct-looking run.
2. **A/B or death.** A mechanism either beats its own removal on held-out BPB at
   a fixed step budget with 3 seeds per arm, or it is deleted. PonderNet was
   deleted this way. Anything inside the control's own seed spread is "no
   difference", and no difference means gone.
3. **Zero host syncs outside a read.** The device is synchronized only on steps
   that already read something back (log cadence, host-Adam cadence, timers).
   A sync added for convenience is a step-time regression wearing a disguise.
   Corollary, learned the hard way: **never build a numeric indicator from a bool
   tensor on device — count on the host.** `mask_fill` writes in place through a
   buffer that `clone()` shares, and a counting indicator built that way
   reported 50 false non-finite losses per 50-step window.
4. **A bit-for-bit claim must name the external source.** "Verified against the
   reference" is not a claim; "verified against `NVlabs/GatedDeltaNet-2`,
   `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py`, tolerance 5e-4, harness at
   `tests/bit_exact.rs`" is. Where no reference code exists, the crate's doc
   comment says *that* instead of implying verification it does not have. An
   absolute tolerance wider than the effect it measures is not a gate.
5. **A verification claim must say which part of a run it covers.** "fp4
   quantization verified, 100 steps, 0 NaN" is false if the attention path ran
   at 8 bits. Name the configuration, the steps, the shape, the machine, and
   the part of the pipeline the test actually exercised. A claim that does not
   say what it covered is a claim about nothing.
6. **One heavy thing at a time.** One GPU process. One build. This workstation
   has frozen three times, and two legal processes colliding corrupted the
   memory pool and wrote garbage weights into a checkpoint.
7. **Never fix unrelated things in the same diff.** Follow-ups get reported,
   not smuggled.

## Docs

| doc | what |
|---|---|
| [`AGENTS.md`](AGENTS.md) | agent-facing repo state: measured numbers, CUDA quirks, every knob |
| [`docs/glossary.md`](docs/glossary.md) | the vocabulary ("fused", "arm", "sidecar", "iteration", "Engram") and the live list of places a document and the code disagree |
| [`docs/PLAN.md`](docs/PLAN.md) | program plan, phase ladder, verdicts, north star |
| [`docs/AB-PROTOCOL.md`](docs/AB-PROTOCOL.md) | the measurement instrument and the A/B queue (unrun) |
| [`docs/adr/`](docs/adr/) | ADR-0001..0022 — every recorded decision, including the retracted ones |
| [`docs/audit-2026-09-25.md`](docs/audit-2026-09-25.md) | codebase verdict audit: kill list, missing A/Bs |
| [`docs/design-minimal.md`](docs/design-minimal.md) | the minimal-architecture target |
| [`docs/mixture-arms.md`](docs/mixture-arms.md) | three priced training arms |
| [`research/2026-09-27-oracle-audit.md`](research/2026-09-27-oracle-audit.md) | what the library's verification claims are actually worth (read this before trusting any of them) |
| [`research/2026-09-27-pytorch-baseline.md`](research/2026-09-27-pytorch-baseline.md) | every number in (a), with the two ways it can mislead you |
| [`research/2026-09-27-scale-ladder.md`](research/2026-09-27-scale-ladder.md) | the 1B arithmetic in (e) |
| [`research/2026-09-27-domain-anchors.md`](research/2026-09-27-domain-anchors.md) | per-domain n-gram bars |
| [`POST_TRAINING.md`](POST_TRAINING.md), [`bf16_KERNEL_PLAN.md`](bf16_KERNEL_PLAN.md) | post-training loop, precision plans (in Russian) |

## License

**AGPL-3.0-only** — see [`LICENSE`](LICENSE) for the full text, and
`Cargo.toml`'s `license = "AGPL-3.0"` (which every crate manifest inherits).
Copyright (c) 2026 sehaxe.

Why AGPL and not MIT: the project is a derived work carrying patched forks of
[burn](https://github.com/tracel-ai/burn) and
[cubecl](https://github.com/cubecl/cubecl), both MIT, and it is built around the
premise that a model "codes its own updates, merged only through the
verification harness" — network-mediated modification. AGPL is the license that
covers that case, it is compatible with linking the MIT/Apache upstream
components, and it was the deliberate choice already recorded in every crate
manifest.

Third-party components vendored in this tree, with their own licenses and
notices, which must be preserved:

| path | what | upstream license |
|---|---|---|
| `vendor/burn-fused/` | 28-crate fused-kernel library, a fork of burn | MIT |
| `vendor/cubecl-fix/` | patched `cubecl-runtime`, `cubecl-server`, `cubecl-cuda` | MIT |
| `vendor/cubek-fix/` | patched `cubek-reduce` (top-k index fix, ADR-0015) | MIT |

> **Known defect, reported not fixed:** `vendor/burn-fused/LICENSE:3` and
> `vendor/cubecl-fix/LICENSE:3` both read `Copyright (c) 2026 sehaxe`, which
> drops the upstream copyright notices that MIT §a requires be retained
> (`Copyright (c) 2023 The Burn Project` / `The Cubecl Project`). Those files are
> not this task's to edit; whoever owns `vendor/` should restore the upstream
> lines and keep the fork's own additions attributed separately.
