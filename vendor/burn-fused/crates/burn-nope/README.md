# burn-nope - No Positional Encoding for Burn

[![CI](https://github.com/sehaxe/burn-nope/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-nope/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-nope)](https://crates.io/crates/burn-nope)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Pure content-based scaled dot-product attention without positional embeddings
(Kimi K3). No RoPE rotation - relies on causal masking and content structure.

> Paper: [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
> (Moonshot, 2026). NoPE in KDA layers.

## Install

```bash
cargo add burn-nope
```

## Quick start

```rust
use burn_nope::nope_attention;

let out = nope_attention(q, k, v, true);  // causal, no RoPE
let out = nope_attention(q, k, v, false); // bidirectional
```

## License

MIT. See [LICENSE](LICENSE).

## Performance

`nope_attention` uses burn's fused `activation::softmax` (one kernel) instead
of the manual 5-pass max/sub/exp/sum/div chain that the naive implementation
needs — fewer launches and no redundant `[B,H,T,T]` intermediate passes.
