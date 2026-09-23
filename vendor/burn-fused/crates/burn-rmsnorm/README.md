# burn-rmsnorm - RMS Normalization for Burn

[![CI](https://github.com/sehaxe/burn-rmsnorm/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-rmsnorm/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-rmsnorm)](https://crates.io/crates/burn-rmsnorm)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

**Root mean square layer normalization** for the [Burn](https://burn.dev)
deep learning framework. A simpler, faster alternative to LayerNorm - used
in every major LLM (LLaMA, Mistral, Qwen, DeepSeek, Gemma).

> Paper: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
> (Zhang & Sennrich, 2019).

## Install

```bash
cargo add burn-rmsnorm
```

## Quick start

```rust
use burn_rmsnorm::RMSNorm;

let norm = RMSNorm::new(512, 1e-5, &device);
let y = norm.forward(x);  // [B, T, 512] -> [B, T, 512]
```

## How it works

```
RMS(x) = sqrt(mean(x^2) + eps)
y = x / RMS(x) * weight
```

Unlike LayerNorm, RMSNorm does not re-center (no mean subtraction), only re-scales.

## License

MIT. See [LICENSE](LICENSE).
