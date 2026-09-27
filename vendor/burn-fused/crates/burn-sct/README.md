# burn-sct - Spectral Compact Training

[![CI](https://github.com/sehaxe/burn-sct/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-sct/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-sct)](https://crates.io/crates/burn-sct)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Drop-in `nn.Linear` replacement for [Burn](https://burn.dev). Weights stored as
**permanent truncated SVD**: `W = U * diag(s) * V^T`. The dense matrix is never
materialized. After each optimizer step, U and V are retracted to the Stiefel
manifold via QR decomposition.

> Based on [Spectral Compact Training](https://arxiv.org/abs/2604.00733) (Kohlberger, 2026).
> Up to **199× memory reduction** per MLP layer at rank 32 — computed from
> model dims in the table below (dense vs SCT storage incl. Adam states).

## Quick start

```rust
use burn_sct::{SctConfig, SctLinear};

let device = Default::default();
let cfg = SctConfig::new(512, 2048, 64);  // in=512, out=2048, rank=64
let mut layer = SctLinear::<NdArray>::new(&cfg, &device);

// Forward pass - three small matmuls, no dense matrix
let y = layer.forward(x);  // [batch, 512] → [batch, 2048]

// After optimizer.step(), maintain orthonormality
layer.retract();
```

## How it works

```
Dense:   y = x @ W                          [m×n matrix, O(b·m·n) FLOPs]
SCT:     y = (x @ U) * s @ V^T              [three small matmuls, O(b·k·(m+n)) FLOPs]
```

Where `U ∈ ℝ^{m×k}`, `V ∈ ℝ^{n×k}` have orthonormal columns, `s ∈ ℝ^k`.

## QR retraction

`retract()` projects U/V back onto the Stiefel manifold (paper Eq 5):

```
Q, R = QR(M);  M ← Q * sign(diag(R))
```

The QR is the Householder decomposition adapted from
[burn-rs/burn](https://github.com/burn-rs/burn) —
`crates/burn-tensor/src/tensor/linalg/qr.rs` (main branch, by the burn-rs
maintainers, MIT). It is reduced to O(m·k²) for SCT's tall-skinny
factors (m ≫ k): the reflection vectors are stored in the R pass and Q is
built back-to-front (LAPACK `orgqr` scheme), so no `m×m` intermediate is
ever materialized. The `sign(diag(R))` correction matches the paper's
`safe_qr` (PyTorch `torch.linalg.qr` + sign flip).

## Memory savings (Adam, rank 32)

| Model | Dense MLP | SCT MLP | Compression |
|-------|-----------|---------|-------------|
| SmolLM2-135M | 14.2 MB | 1.1 MB | 13× |
| SmolLM2-1.7B | 268.4 MB | 5.2 MB | 51× |
| LLaMA-7B | 721.4 MB | 7.7 MB | 93× |
| LLaMA-70B | 3,758 MB | 18.9 MB | **199×** |

## Reference comparison — NOT YET RUNNING

`tests/cmp_reference.rs` (behind the non-default `binary-tests` feature) is
written to compare forward, retraction and `from_dense` against the official
PyTorch reference ([EctoSpace/SCT](https://github.com/EctoSpace/SCT)).

**It has never run.** The reference tensors it loads (`tests/ref_data/*.bin`)
are not in the tree, and the generator that produces them
(`tests/gen_reference.py`) is not either — this crate's `.gitignore`
inherits `*.py` and `*.bin` from a standalone-repo template and therefore
excludes exactly the two things the comparison needs. Any accuracy number
quoted for this crate before the generator and the fixture are committed is
unmeasured.

```
cargo test --release --features binary-tests --test cmp_reference -- --nocapture
```

Configs: tiny 64×128/k8, small 256×512/k16, med 512×1024/k32, large 1024×2048/k64.
When it does run, the harness compares rank-k reconstructions (sign-invariant),
never raw singular vectors (unique only up to sign), and the `from_dense` SVD is
one-sided (Hestenes) Jacobi — exact to f32 rounding, equivalent to
`torch.linalg.svd`.

## License

MIT

## Performance

- **Forward** is memory-optimal: `y = (x@U)·s @ Vᵀ` — peak footprint is just
  the input + the two GEMM outputs (no extra `[in, k]` intermediate).
- **QR retract** runs on CUDA via custom `sct_qr_r`/`sct_qr_q` kernels (no host
  round-trip, verified 2e-7); the CPU fallback uses AVX2/FMA SIMD dot3 on
  x86_64.
