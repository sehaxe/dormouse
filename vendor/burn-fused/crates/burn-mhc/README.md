# burn-mhc - Manifold-Constrained Hyper-Connections for Burn

[![CI](https://github.com/sehaxe/burn-mhc/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-mhc/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-mhc)](https://crates.io/crates/burn-mhc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Manifold-Constrained Hyper-Connections (mHC) for Burn — full implementation of
[arXiv:2512.24880](https://arxiv.org/abs/2512.24880) (DeepSeek, 2025):

- **Eq 3**: n-stream residual `x_{l+1} = H_res x_l + (H_post)^T F(H_pre x_l)`
- **Eq 7**: first-order hyper-network — input-dependent mappings from
  `RMSNorm(vec(x))` linear projections + static biases, gating factors α init 0.01
- **Eq 8**: `H_pre = sigmoid`, `H_post = 2·sigmoid` (non-negativity prevents
  signal cancellation), `H_res = Sinkhorn-Knopp(...)`
- **Eq 9**: Sinkhorn-Knopp entropic projection (t_max = 20) onto the Birkhoff
  polytope — `H_res` is doubly stochastic (row/col sums = 1, spectral norm ≤ 1,
  compositional closure), restoring the identity-mapping property

At init the block reduces to the standard residual `h + Σ branches`
(`H_pre ≈ 1`, `H_post ≈ 1`, `H_res ≈ I`).

## Install

```bash
cargo add burn-mhc
```

## Quick start

```rust
use burn_mhc::MhcBlock;

let mhc = MhcBlock::new(4, d_model, &device); // n=4 streams, D = n·C
let out = mhc.forward(h, &[ffn_out]);         // h, ffn_out: [B, T, D]
```

## API

| Export | What |
|--------|------|
| `MhcBlock` | Full mHC block: hyper-net + Birkhoff-projected residual mixing |
| `sinkhorn_knopp` | Entropic projection onto the doubly stochastic manifold (Eq 9) |
| `SINKHORN_ITERS` | t_max = 20 (paper App. A.1) |
| `ALPHA_INIT` | Gating factor init 0.01 (paper App. A.1) |


## Performance (RTX 3090, CUDA, burn 0.22)

`SINKHORN_ITERS=20` Sinkhorn-Knopp is fused into one launch per (batch, time)
matrix (alternating row/column normalizations, `sync_cube()` between phases).

| Op | Config | Tensor path | Fused | Speedup |
|----|--------|-------------|-------|---------|
| forward | [8, 2048, 16] | 32.4 ms | **1.65 µs** | **19,600×** |
| forward | [4, 1024, 64] | 104.7 ms | **2.55 µs** | **41,000×** |
| backward | [8, 2048, 16] | 67.1 ms | **17.7 ms** | **4×** |

Verified == tensor path and doubly stochastic (<1e-2); the fused backward
reverses the 2·iters normalizations with per-step sums kept in shared memory.

## Training

The fused Sinkhorn runs as a single tracked node under `Autodiff<Cuda>`; the
backward is a fused kernel that recomputes the forward trajectory's per-step
row/col sums in shared memory and reverses the 2·iters normalizations
(`d_m_pre = d/s − m_pre·Σd/s²`). Verified fused backward == tensor-path
backward (<1e-3). Training and inference both use the fused kernels.

## Inference

Forward-only builds use the bare CUDA fused Sinkhorn directly (2 µs per
(b, t) matrix — the tensor path was 32-105 ms).

## License

MIT. See [LICENSE](LICENSE).

