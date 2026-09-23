# burn-ttt - Test-Time Training for Burn

[![CI](https://github.com/sehaxe/burn-ttt/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-ttt/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-ttt)](https://crates.io/crates/burn-ttt)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Test-time training loss for long-context LLMs. Learn from context at inference time
via next-token prediction, compressing the context into model weights. Constant
latency regardless of context length.

> Paper: [TTT-E2E](https://arxiv.org/abs/2512.23675) (Tandon et al., 2025).
> Paper's result (3B model): 2.7x faster than full attention at 128K context;
> this crate implements the training loss, not the inference path.

## Install

```bash
cargo add burn-ttt
```

## Quick start

```rust
use burn_ttt::ttt_loss;

// Test-time training loss on sliding window context
let loss = ttt_loss(predictions, targets, mask);
// Use loss with Burn autograd for weight updates
```

## API

| Export | What |
|--------|------|
| `ttt_loss(p, t, m)` | Masked MSE loss for TTT on context window |

## License

MIT. See [LICENSE](LICENSE).
