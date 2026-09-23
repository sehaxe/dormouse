# burn-mtp - Multi-Token Prediction for Burn

[![CI](https://github.com/sehaxe/burn-mtp/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-mtp/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-mtp)](https://crates.io/crates/burn-mtp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

**Multi-Token Prediction** for the [Burn](https://burn.dev) framework.
Predict the next `k` positions from each hidden state with exponential
decay weighting - improves model quality and sample efficiency.

> Paper: [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) (Gloeckle et al., 2024).

## Install

```bash
cargo add burn-mtp
```

## Quick start

```rust
use burn_mtp::MtpHeads;

let mtp = MtpHeads::new(512, 3, &device);  // 3 heads, predicting next 3 tokens
let loss = mtp.loss(hidden_states, targets);
```

## How it works

```
h[t] → head_0 → predict target[t+1]  (weight: 0.5^1 = 0.5)
h[t] → head_1 → predict target[t+2]  (weight: 0.5^2 = 0.25)
h[t] → head_2 → predict target[t+3]  (weight: 0.5^3 = 0.125)
```

Each head is an independent `Linear(D, D)` projecting hidden states to
next-token predictions. Earlier positions are weighted more heavily.

## License

MIT. See [LICENSE](LICENSE).
