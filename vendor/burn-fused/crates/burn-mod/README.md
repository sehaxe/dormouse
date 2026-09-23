# burn-mod

[![CI](https://github.com/sehaxe/burn-mod/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-mod/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-mod)](https://crates.io/crates/burn-mod)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

[Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258) (Raposo et al., Google DeepMind 2024) for Burn.

A per-block router emits a scalar weight per token; the `k` highest-weighted tokens (expert-choice routing) run through the block (self-attention + MLP), the rest pass through a residual connection. The block output is scaled by the router weight, putting the router on the gradient path (paper eq. 1).

- `ModRouter` — linear `d -> 1` token router
- `select_topk` — expert-choice top-k selection (masked-argmax, GPU-safe)
- `route_block` — eq. 1 routing step around any block closure
- `bce_aux_loss` — binary cross-entropy aux loss for causal autoregressive sampling (paper §3.5)
- `ModPredictor` — optional stop-gradient MLP predictor for sampling (paper §3.5)

## License

MIT. See [LICENSE](LICENSE).
