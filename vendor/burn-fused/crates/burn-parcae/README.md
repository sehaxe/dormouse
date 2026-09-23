# burn-parcae — stable looping via spectral retention

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Implementation of the **Parcae** stability mechanism from
[Parcae: Scaling Laws For Stable Looped Language Models](https://arxiv.org/abs/2604.12946)
(Prairie, Novack, Berg-Kirkpatrick, Fu).

Looped (parameter-shared, iterative) transformers apply the same block `T`
times to the residual stream; without spectral control the activations
explode or spike. Parcae fixes this by recasting looping as a nonlinear
time-variant dynamical system:

```text
h_{t+1} = Ā·h_t + B̄·e + R̄(h_t, e)
```

with `Ā` state retention, `B̄` input injection and `R̄` the transformer
nonlinearities. Instability comes from large spectral norms of the injection
parameters. Parcae constrains `Ā` by discretizing a continuous-time
negative-diagonal parameterization:

```text
A  := diag(-exp(a))            a ∈ R^d, learnable, per-channel
Ā  := exp(Δ·A)                 Δ ∈ R^d_{>0}, learnable, per-channel step size
B̄  := Δ·B                      B unconstrained (input is LayerNorm'd)
```

`A` has strictly negative diagonal and `Δ > 0`, so `Δ·A` has strictly
negative entries and `exp(Δ·A)` has all eigenvalues in `(0, 1)` — **guaranteed
contraction, bounded residual dynamics for ANY loop count `T`, by
construction** (no clipping, no post-hoc normalization).

## What is implemented

- `SpectralRetentionConfig` — `full_b: bool` (paper's full `d×d` injection
  matrix vs the parameter-efficient diagonal variant, default diagonal).
- `SpectralRetention<B>` — `a` (log-rate), `delta` (step size, ones init) and
  `b` (injection) parameters:
  - `retention()` → `exp(-Δ·exp(a))` as `[d]`, all entries in `(0, 1)`.
  - `inject(e)` → `B̄·e = (Δ·B)·e`.
  - `forward(h, e)` → `Ā⊙h + B̄·e`.
- Free fn `retention_matrix(a, delta)` — the discretized retention diagonal.
- Pure tensor ops, zero host branching in forward (`exp`, `neg`, `mul`,
  `add` only).

## Usage

```rust,ignore
use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor};
use burn_parcae::SpectralRetention;

type Backend = NdArray<f32>;

let retention = SpectralRetention::<Backend>::new(64, &Default::default());
let h = Tensor::<Backend, 3>::random([2, 32, 64], Distribution::Default, &Default::default());
let e = Tensor::<Backend, 3>::random([2, 32, 64], Distribution::Default, &Default::default());

for _ in 0..T {
    // h stays bounded for ANY T (contraction by construction).
    let h = retention.forward(h, e.clone());
}
```

Defaults: `a = 0`, `Δ = 1` → `Ā = e^-1 ≈ 0.37`, a mild contraction. The
gradients flow through `exp` freely — nothing is clipped.

## Tests

```bash
cargo test --release
cargo clippy -- -D warnings
cargo fmt
```
