# dormouse-spectral — Ternary Spectral Compact Training (TSCT)

TSCT reparameterizes a linear layer as a truncated-SVD product `W = U·diag(s)·Vᵀ` where the
orthonormal factors `U`, `V` are full-precision **masters** but the forward always runs their
**ternary projections** `{-1, 0, +1}·scale` (BitNet b1.58 absmean straight-through estimator,
[2504.12285](https://arxiv.org/abs/2504.12285)), so a `[m, n]` layer costs `k·(m+n)` ternary
values + `k` scales instead of `m·n`
dense values. On top of that, `SpectralMoE` gives **rank-1 ternary experts** (expert `i` is
`u_i ⊗ v_i`) behind a product-key router (cluster scorer → per-cluster expert scorer) with
token-choice top-k or Expert-Choice routing ([2202.09368](https://arxiv.org/abs/2202.09368)), plus a fully fused CUDA training
kernel and a 2-bit packed inference form with no FP32 matmuls.

Built on burn 0.22 / cubecl 0.11. Every number below is measured on the dev rig
(RTX 5060 Ti 16 GB, CUDA release build), and every loss that lost is reported.

## Measured results

### d=64 mini-GPT (T1.5, 2026-08-13) — 3 seeds, mean ± std

4-layer char model, MHA + RoPE, 600 steps, LR=1e-3, POLAR=1, 32 tokens/step,
SEED 42/43/44, true holdout (wiki book `00495` → `00496`).

| kind | val loss (mean±std) | FLOPs/token (vs dense) | ms/step | VRAM reserved |
|---|---|---|---|---|
| dense | 3.3544 ± 0.021 | 212,992 (1.00x) | 0.02 | 33 MB |
| **tsctmoe** | **3.2398 ± 0.066** | 170,496 (**1.25x**) | 0.05 | 717 MB |
| tsctmoe_wide | 3.2387 ± 0.066 | 235,520 (**0.90x — more than dense**) | 0.05 | 2.7 GB |
| tsct2 | 3.3576 ± 0.069 | 87,040 (2.4x) | 0.02 | 2.7 GB |
| sct8 | 3.3555 ± 0.014 | 102,400 (2.1x) | 0.02 | 4.7 GB |

Read honestly: `tsctmoe`/`tsctmoe_wide` beat dense on val by ≈0.115 (~1.7σ) at iso-ish FLOPs —
a hint at the Intelligence-gate, **not** proof (600 steps, d=64, high variance). `tsctmoe_wide`
is a loser on efficiency: it uses *more* FLOPs than dense (0.90x) for the same val. `tsct2` and
`SCT` are parity. FLOPs/token include the shared attention (4·D² + 2·NH·HD·seq per layer);
accounting rules: routers are MACs, not params; EC is batch-aware.

### d=256 GPT (T4.2 first pass, 2026-08-14) — **1 seed, indicative only**

8,000 steps, DIM=256, SEQ=128, BATCH=8 (1024 tok/step), LR=3e-4, POLAR=1, SEED=42,
wiki `00495` → `00496` holdout. Single-seed numbers are not conclusive; a 3-seed repeat is
scheduled.

| kind | val loss (final) | FLOPs/token (vs dense) | ms/step | VRAM reserved |
|---|---|---|---|---|
| dense | 2.6713 | 3,407,872 (1.00x) | 0.01 | 550 MB |
| tsctmoe | 2.7202 (**loser**) | 1,583,616 (**2.15x**) | 0.04 | 4,275 MB |
| **tsct2** | **2.6489 (winner)** | 1,331,200 (**2.56x**) | 0.02 | 4,275 MB |

`tsct2` beats dense on val at 2.56x fewer FLOPs; `tsctmoe` — the d=64 winner — **lost here**
(2.7202 vs 2.6713). There is no consistent winner yet. All three models generate real words and
grammar ("the X of the Y") locked to the training book's topic.

## Speed: fused training kernel (T2.1, 2026-08-14)

Measured @ B=16384, m=512, k=32, n=4096 (the harness "production" regime), CUDA:

| path | fwd | fwd+bwd |
|---|---|---|
| **fused kernel** | **1.35 ms** | **3.56 ms** |
| dense (burn matmul) | 9.6 ms | 54.3 ms |
| old tensor path (burn ops) | 39 ms | — |

The fused kernel is ~7x faster than dense fwd and **15x faster than dense fwd+bwd**; it replaced
a 23.5 ms tensor-op step that was the #1 blocker for the "2 days" dream. Honest caveat: the
step-level win does not carry 1:1 to an end-to-end model — attention is not spectralized, and
Amdahl puts the realistic end-to-end ceiling at **1.6–3.3x** at d=64 without sparse attention.
At small d the spectral tensor path is launch-overhead-bound (PROBE @16384, d=64: dense
layer_fwd 0.06 ms vs tsctmoe 268 / tsct2 130 / wide 44 ms); the fused kernel fixes that for
`SpectralLinear`, the MoE tensor path is still slow (T2.2).

## Memory

- **2-bit inference packs**: ternary values pack 4-per-byte (`pack_ternary`, 2 bits/value),
  so weight traffic is **16x below dense FP32** (0.25 B vs 4 B per value). `to_inference()`
  drops the FP32 masters: 18.4 KB for a 512×4096 layer at rank 8 vs 8.4 MB dense. The
  inference forward is add/sub-only — no FP32 weight matmuls.
- **Expert-Choice kills the activation blow-up**: the token-choice forward materializes
  `[B, k·r, in]` and `[B, k·r, out]` — ~1 GB/layer at B=16384, k·r=32, in=512, the root cause
  of the 24-layer OOM. EC (per-expert top-token picks within the cluster, never materialized)
  trains every expert every step and caps activation memory: measured 3.9 GB reserved for
  tsctmoe vs 14.3 GB (tsct2/wide, allocator fragmentation) and >12 GB→OOM (SCT) at
  B=16384/step.
- Masters stay FP32; MoE has ~4.6x *more* weights than dense by design (capacity win) — the
  memory win is inference and activations, not weights.

## Quickstart

```rust
use dormouse_spectral::{SpectralLinear, SpectralMoE, polar_orthogonalize};

// plain ternary-SVD linear: rank k (defaults: alpha=1 pure STE, fused kernels on)
let mut layer = SpectralLinear::new(512, 4096, 32, &device);
layer.set_stochastic(true);     // unbiased stochastic ternary rounding
layer.set_per_column(true);     // per-column ternary scale (keeps weak ranks alive)
layer.set_fused(false);         // opt out of the fused CUDA kernel
let y = layer.forward(x);       // [B, 512] -> [B, 4096]

// product-key MoE: 128 clusters x 8 rank-4 experts, top-2 per token
let mut moe = SpectralMoE::new(512, 4096, 128, 8, 2, 4, &device);
moe.set_expert_choice(true);    // per-expert top-token picks, no [B, k*r, in] materialization
let y = moe.forward(x);

// keep the masters orthonormal (Newton-Schulz polar, on device; CPU-QR-free)
layer.retract(3);

// freeze to the 2-bit inference form (panics if trained stochastic)
let inf = layer.to_inference();
let y = inf.forward(x);         // packed add-only matmul
```

Fused kernels engage automatically for plain mode (no stochastic/per_column, alpha=1, CUDA);
`TSCT_FUSED=0` or `set_fused(false)` falls back to the tensor path.

### Harness (examples/tsct_diag.rs)

Mini-GPT with real attention, seeds, and true holdout. Env knobs: `KINDS`, `STEPS`, `LR`,
`DEVICE=cuda`, `OPT=muon`, `POLAR=1`, `ACT`, `SEED`, `DIM`, `SEQ`, `BATCH`, `VAL_N`,
`VAL_EVERY`, `TRAIN_FILE`, `VAL_FILE`, `GEN` / `GEN_TEMP` / `GEN_TOKENS` / `GEN_GREEDY`,
`PROBE`. Canonical 3-seed run (the T1.5 protocol):

```sh
cargo run --release --features cuda -p dormouse-spectral --example tsct_diag \
  --env KINDS=dense,tsctmoe,tsctmoe_wide,tsct2,sct8 \
  --env STEPS=600 LR=1e-3 DEVICE=cuda POLAR=1 SEED=42 \
  --env TRAIN_FILE=wiki/books_00495.txt VAL_FILE=wiki/books_00496.txt
```

## Honest limits

- **Attention is not spectralized** — it is dense MHA + RoPE in the harness and dominates
  FLOPs at small d; end-to-end win is bounded by Amdahl (1.6–3.3x at d=64) until sparse
  attention lands.
- **Fused kernels cover plain `SpectralLinear` only** (no stochastic/per_column/alpha<1) and
  measured tile shapes (512×4096, k=32, B=16384); other shapes fall back to the tensor path.
  The MoE tensor path is still ~1000x slower than dense at small d (launch overhead) — fused
  MoE-EC is T2.2.
- **Regime B (16384 tokens/step) diverged at LR=1e-3** (val 5.5–5.9 for every kind) and SCT
  OOMs a 16 GB card there; the re-run needs LR=1e-4 (T1.5b, pending).
- **1-seed results are indicative, not conclusive** (d=256 table); the d=64 winner
  (tsctmoe) lost at d=256 — expect reshuffling at larger scale until 3-seed repeats land.
- **The 27B-in-2-days dream is a research target, not a claim**: it requires sparse
  attention plus a fitted scaling law (loss-vs-FLOPs curves at ≥3 sizes, T4.2–T4.3) before
  anything is extrapolated.

## Development

```sh
cargo test -p dormouse-spectral --lib                 # STE, polar, MoE, EC gradient checks
cargo test -p dormouse-spectral --example tsct_diag   # harness: shapes, seeds, holdout
cargo check -p dormouse-spectral --features cuda --example tsct_diag
cargo clippy -p dormouse-spectral
```

Roadmap: T2.2 fused MoE-EC, T1.5b Regime B re-run, T4.2 3-seed d=256 + d=512,
T5.2 `bench/baselines.json` re-baselined on the 5060 Ti.
