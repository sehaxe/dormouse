# burn-mor

Mixture-of-Recursions (MoR) — per-token adaptive recursion depth for [Burn](https://burn.dev) 0.22.

Implements the expert-choice routing primitives from
*"Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"*
(Bae et al., 2025), [arXiv:2507.10524](https://arxiv.org/abs/2507.10524).

MoR reuses a single shared stack of layers across recursion steps, while a
lightweight router dynamically assigns **different recursion depths to
individual tokens**:

- low-importance tokens are dropped at deeper recursion levels (they pass
  through the residual only),
- quadratic attention (and the FFN) is computed **only among tokens still
  active** at a given depth,
- an auxiliary load-balancing loss keeps the router from collapsing onto a
  fixed token subset.

This crate provides the pure-tensor routing pieces; the recursion block itself
is your own (parameter-shared) layer stack.

## Usage

```rust
use burn::tensor::Tensor;
use burn_mor::{MoRConfig, MoRRouter, gather_active, load_balancing_loss, scatter_active, select_active};

fn mor_step<B: burn::tensor::backend::Backend>(
    block: impl Fn(Tensor<B, 3>) -> Tensor<B, 3>,
    router: &MoRRouter<B>,
    h: Tensor<B, 3>,
    config: MoRConfig,
) -> (Tensor<B, 3>, Tensor<B, 1>) {
    let device = h.device();
    let scores = router.scores(h.clone());
    let (active, _inactive) = select_active(scores.clone(), config.keep_frac, &device);
    let n = active.dims()[1];

    // Heavy computation (attention + FFN) runs only on active tokens.
    let out_active = block(gather_active(h.clone(), active.clone()));

    // Dropped tokens get zero block output -> residual-only pass-through.
    let out = scatter_active(Tensor::zeros_like(&h), active.clone(), out_active, &device);
    let h_next = h.add(out);

    let aux = load_balancing_loss(scores, active, n).mul_scalar(config.aux_weight);
    (h_next, aux)
}
```

At each recursion step `select_active` keeps `round(keep_frac * T)` of the
still-active tokens (hierarchical filtering per the paper); deeper recursion
levels therefore operate on progressively smaller active sets.

## API

| Item | Description |
|---|---|
| `MoRConfig` | `keep_frac`, `block_size`, `aux_weight`, `gradient_detach` (defaults: 0.5, 128, 0.01, true). |
| `MoRRouter` | Linear `d → 1` (no bias) importance scorer, `scores(&h) -> [B, T, 1]`. |
| `select_active` | Top-k active / inactive token split (no `gather_nd`, CUDA-safe). |
| `gather_active` | `[B, T, D]` + `[B, K]` → dense `[B, K, D]` active batch. |
| `scatter_active` | Place `[B, K, D]` block output back into `[B, T, D]` (zeros elsewhere). |
| `load_balancing_loss` | Auxiliary loss = mean squared deviation of per-token active usage from `n/T`, differentiable through the router scores. |

All functions are pure tensor ops (no host branching, no `into_data`), generic
over `B: Backend`, and compose with `Autodiff`.

## KV caching

Per the paper's KV-sharing variant (default ON): cache KV pairs once for all
tokens at the first recursion and reuse them at deeper levels; only the query
set shrinks. `block_size` is the granularity for that cache integration.

## Tests

```bash
cargo test --release   # assert-based self-checks on the ndarray backend
cargo clippy -- -D warnings
```
