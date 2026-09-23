# burn-dspark - DSpark Speculative Decoding for Burn

[![CI](https://github.com/sehaxe/burn-dspark/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-dspark/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-dspark)](https://crates.io/crates/burn-dspark)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Semi-autoregressive speculative decoding building blocks from [DSpark](https://arxiv.org/abs/2607.05147)
(DeepSeek AI, 2026). Replaces MTP for both training and inference.

> 60-85% faster generation vs MTP-1 in production DeepSeek-V4 serving.

## Install

```bash
cargo add burn-dspark                              # inference only
cargo add burn-dspark --features training          # + training losses
```

## Quick start

```rust
use burn_dspark::{VanillaMarkov, AcceptRatePredictor};

// Markov head: adds sequential dependency to parallel draft logits
let markov = VanillaMarkov::new(32000, 128, &device);
let conditioned = markov.apply(draft_logits, prev_token_ids);

// Confidence scheduling: estimate per-position acceptance probability
let predictor = AcceptRatePredictor::new(512, &device);
let p_accept = predictor.prob(hidden_states);
```

## API

| Export | Feature | What |
|--------|---------|------|
| `VanillaMarkov` | always | Sequential logit bias from previous token |
| `AcceptRatePredictor` | always | P(accept) per draft position |
| `dspark_loss` | `training` | Masked MSE loss for draft training |
| `accept_rate_loss` | `training` | MSE loss for confidence predictor |

## License

MIT. See [LICENSE](LICENSE).
