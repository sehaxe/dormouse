# burn-bitnet - BitNet Quantization for Burn

> CI: this badge's workflow was deleted (`4963c3a`) — the gate is `../../.github/workflows/fused-library.yml`, and it has no GPU job (for CUDA: `tools/gpu-gate.sh`).
[![Crates.io](https://img.shields.io/crates/v/burn-bitnet)](https://crates.io/crates/burn-bitnet)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

**BitNet quantization family** for the [Burn](https://burn.dev) deep learning
framework. 1.58-bit ternary weight quantization + BitNet v2 Hadamard-based
activation quantization. Straight-through estimator (STE) gradients throughout.

> Papers: [BitNet b1.58](https://arxiv.org/abs/2402.17764) (Ma et al., 2024),
> [BitNet v2](https://arxiv.org/abs/2504.18415) (Wang et al., 2025).

## Install

```bash
cargo add burn-bitnet
```

## Quick start

```rust
use burn_bitnet::{weight_quant_ternary, quantize_4bit, quantize_8bit};

// Quantize weights to ternary {-scale, 0, +scale}
let w_q = weight_quant_ternary(weight);

// Quantize activations to 4-bit with Hadamard
let x_q = quantize_4bit(activation);

// 8-bit with Hadamard
let x_q = quantize_8bit(activation);
```

## API

| Function | Reference | What | Input |
|----------|-----------|------|-------|
| `weight_quant_ternary` | b1.58 (2024) | `W → sign(W−μ)·scale` | `[M, N]` |
| `activation_quant_8bit` | b1.58 (2024) | absmax per-token → int8 | `[B, T, D]` |
| `bitnet_v2_quantize` | v2 (2025) | Hadamard + absmax/absmean | `[B, T, D]` |
| `fast_walsh_hadamard` | v2 (2025) | FWHT O(n log n) | `[M, N]` |
| `quantize_4bit` | v2 (2025) | `bitnet_v2_quantize(x, 4)` | `[B, T, D]` |
| `quantize_8bit` | v2 (2025) | `bitnet_v2_quantize(x, 8)` | `[B, T, D]` |

## How it works

```
Ternary weight quantization (b1.58):
    scale = mean(|W|)
    W_q = sign(W − mean(W)) · scale

Tensor shapes: [in_features, out_features]

BitNet v2 quantization:
    x → FWHT → quantize → FWHT → output

8-bit: absmax per-token → int8 range [-128, 127]
4-bit: absmean per-token → int4 range [-8, 7]
```
    
All functions use the straight-through estimator (STE): forward pass uses
quantized values, backward gradient flows through the unquantized input.


## Performance (RTX 3090, CUDA, burn 0.22)

| Op | Tensor path | Fused | Speedup |
|----|-------------|-------|---------|
| FWT [256, 512] | 8.87 ms | **0.79 ms** | **11×** |
| quant8 [256, 512] | ~11 ms (scale+passes) | **1.38 ms** | ~8× |

The FWT runs the log-p butterfly rounds in shared memory (one cube per row,
comptime round loop, `sync_cube()` between rounds); the quant kernel is one
elementwise launch with the per-row scale precomputed. The FWT is in-place and
always operates on an internal copy (the caller's tensor is never aliased).

## Training

The fused kernels register as single tracked nodes under `Autodiff<Cuda>`, so
the same CUDA kernels run during training:

- `fast_walsh_hadamard`: exact backward = the same transform (the normalized
  Hadamard matrix is symmetric and orthogonal).
- 8/4-bit quantize: straight-through backward (identity gradient; the lib
  applies the `x + (q - x.detach())` pass-through at the outer level).

Verified fused backward == tensor-path backward (<1e-3).

## Inference

Forward-only builds use the bare CUDA fused FWT + quant kernels directly (no
graph overhead). The fused path works for row widths up to the 48 KB shared
ceiling (p <= 12000, power of two); other shapes fall back to the tensor path.

## License

MIT. See [LICENSE](LICENSE).


