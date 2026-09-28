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

## Reference comparison — UNIMPLEMENTED

**There is no comparison against the official PyTorch reference
([EctoSpace/SCT](https://github.com/EctoSpace/SCT)).** A harness existed
(`tests/cmp_reference.rs`, behind a non-default `binary-tests` feature) and was
deleted on 2026-09-27: the reference tensors it loaded (`tests/ref_data/*.bin`)
and the generator that produced them (`tests/gen_reference.py`) were both
absent from the tree, and this crate's `.gitignore` inherits `*.py` and `*.bin`
from a standalone-repo template — so it could not run by three independent
routes and reported a green suite having asserted nothing. A test that pretends
to be an oracle is worse than no test.

**Consequence: every accuracy number for this crate is UNMEASURED against the
authors' code.** What the suite does check is listed below.

To restore the comparison: write `tests/gen_reference.py` from
`spectral_compact_training/spectral_layer.py`, commit the four fixtures it
emits, and re-add the harness. The comparison should be rank-k
reconstructions (sign-invariant), never raw singular vectors (unique only up to
sign); the `from_dense` SVD is one-sided (Hestenes) Jacobi — exact to f32
rounding, equivalent to `torch.linalg.svd`. Planned configs: tiny 64×128/k8,
small 256×512/k16, med 512×1024/k32, large 1024×2048/k64.

### What IS tested

Invariants only, all on the ndarray backend unless noted:

| Check | Test |
|---|---|
| `U^T U = I`, `V^T V = I` after `retract()` | `retract_restores_ortho` |
| retraction error does not grow | `ortho_error_decreases` |
| init is orthonormal; `sign(diag(R))` correction applied | `orthonormal_init`, `sign_correction` |
| one-sided Jacobi SVD reproduces `A` (three shapes, incl. non-diagonal) | `svd_cpu_roundtrip_*` |
| `from_dense` rank-k reconstruction | `from_dense_roundtrip` |
| shapes, param count, rank auto-clamp, compression ratio | `forward_shape`, `param_count`, `rank_auto_clamped`, `compression_ratio` |
| fused CUDA QR vs the CPU Householder path (~1e-6) | `tests/cuda_retract.rs` — needs a GPU |

None of these is the paper's reference. They are invariants.

## License

MIT

## Performance

- **Forward** is memory-optimal: `y = (x@U)·s @ Vᵀ` — peak footprint is just
  the input + the two GEMM outputs (no extra `[in, k]` intermediate).
- **QR retract** runs on CUDA via custom `sct_qr_r`/`sct_qr_q` kernels (no host
  round-trip, verified 2e-7); the CPU fallback uses AVX2/FMA SIMD dot3 on
  x86_64.
