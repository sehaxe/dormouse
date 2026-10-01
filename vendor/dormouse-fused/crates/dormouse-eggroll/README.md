# dormouse-eggroll

> **Not in the dormouse build.** No crate under `crates/dormouse-{core,data,train,cli}/`
> depends on this one; the only incoming edge is the `dormouse-fused` facade. Kept as a
> **reference port** for the post-training phase named in `POST_TRAINING.md`
> ("EGGROLL — exploration for controllers"), which does not exist yet, so no A/B can
> be run against it before then. Fate table: `docs/architecture/library-crate-fate.md`.

EGGROLL — **low-rank evolutionary strategies** for Burn, per
[arXiv:2511.16652](https://arxiv.org/abs/2511.16652) *"Evolution Strategies
at the Hyperscale"*.

ES (evolution strategy) primitives to optimize **non-differentiable**
decisions in a language model (e.g. top-k token routing): instead of
backprop, perturb weights with rank-1 Gaussian noise, evaluate the model,
and update weights by the score-function gradient estimate.

## Mechanism (rank-1, as used in all LLM experiments)

1. For a weight matrix `M ∈ R^{m×n}`: sample `A ∈ R^{m×r}`, `B ∈ R^{n×r}`,
   entries i.i.d. `N(0,1)`.
2. Perturbation `E = (1/√r)·A·Bᵀ` — the `1/√r` keeps `Var(E)` bounded for
   any `r`. **Rank 1 is used throughout**: the paper finds negligible
   performance decrease vs full-rank noise.
3. Perturbed weights `W = M + σE`; LLM fine-tuning uses `σ = 0.001`,
   `α = 0.001`.
4. **Antithetic pairs**: evaluate `(M+σE, M−σE)`; fitness → ternary
   `sign(s⁺ − s⁻) ∈ {−1, 0, +1}`.
5. Update `M ← M + (α/√r)·Σᵢ Eᵢ·fᵢ` — without materializing E:
   `Σᵢ fᵢ·Aᵢ·Bᵢᵀ = A·diag(f)·Bᵀ` (batched matmul over the folded `(N·r)`
   axis). The `1/N` population average is the caller's job — `α` already
   includes it.
6. Noise is regenerated from seeds (store the seed, regenerate A/B
   deterministically). Burn's `Tensor::random` (0.21 through 0.22-pre) has
   no seed parameter yet — seeds are accepted but currently unused.
7. Theory: converges to Gaussian ES as `r, d` grow (`O(r⁻¹)` rate).

## Usage

```rust
use burn::tensor::{backend::Backend, Tensor};
use dormouse_eggroll::{
    sample_a, sample_b, perturb, update, update_batched, antithetic_sign,
    EggrollConfig,
};

fn es_step<B: Backend>(w: &Tensor<B, 2>, device: &B::Device) -> Tensor<B, 2> {
    let cfg = EggrollConfig::new(); // sigma=0.001, alpha=0.001, r=1, N=256

    // Per-member loop (N antithetic pairs)
    let mut m = w.clone();
    for i in 0..cfg.population {
        let a = sample_a::<B>(m.dims()[0], cfg.rank, i as u64, device);
        let b = sample_b::<B>(m.dims()[1], cfg.rank, i as u64, device);
        let w_plus = perturb(&m, &a, &b, cfg.sigma);
        let w_minus = perturb(&m, &a, &b, -cfg.sigma);
        let f = antithetic_sign(fitness(&w_plus), fitness(&w_minus)); // host-side
        m = update(&m, &a, &b, f, cfg.alpha, cfg.rank);
    }
    m
}

// Or batched, GPU-friendly:
//   let a = Tensor::random([N, m, r], Distribution::Normal(0.0, 1.0), device);
//   let b = Tensor::random([N, n, r], Distribution::Normal(0.0, 1.0), device);
//   let f = fitness_tensor; // [N]
//   let m = update_batched(&m, &a, &b, &f, cfg.alpha, cfg.rank);
```

## API

| Function | Signature | What |
|---|---|---|
| `EggrollConfig` | `{sigma, alpha, rank, population}` | paper defaults |
| `sample_a` | `(m, r, seed, device) -> [m,r]` | Gaussian A |
| `sample_b` | `(n, r, seed, device) -> [n,r]` | Gaussian B |
| `perturb` | `(M, A, B, σ) -> [m,n]` | `M + (σ/√r)·A·Bᵀ` |
| `antithetic_sign` | `(s⁺, s⁻) -> f32` | `sign(s⁺−s⁻) ∈ {−1,0,1}` |
| `update` | `(M, A, B, f, α, r) -> [m,n]` | single-member update |
| `update_batched` | `(M, A[N], B[N], f[N], α, r) -> [m,n]` | batched update |
| `eggroll_mutate` | `(W, r, σ, device) -> [m,n]` | convenience mutate |

All tensor code is pure `B: Backend` ops — no `into_data()` host branching
except the inherently host-side ES fitness `antithetic_sign`.

## Tests

```
cargo test --release
cargo clippy -- -D warnings
cargo fmt
```

## License

MIT. Research crate; paper mechanism per arXiv:2511.16652.
