# burn-diffusionblocks - Block-wise Diffusion Training for Burn

[![Crates.io](https://img.shields.io/crates/v/burn-diffusionblocks)](https://crates.io/crates/burn-diffusionblocks)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Block-wise training of residual networks as diffusion denoisers, from
[DiffusionBlocks](https://arxiv.org/abs/2506.14202) (Sakana AI, ICLR 2026):
train each block of a deep residual network **independently** on its own
noise range, cutting training memory from full BPTT cost to **one block's
activations (memory /B)**.

## The method

A residual network is reinterpreted as a VE (variance-exploding) diffusion
model. Corrupt the clean target with Gaussian noise,

```
z_σ = y + σ·ε,   ε ~ N(0, I),
```

and train an **x0-prediction** denoiser `D_θ(z_σ, σ)` with the weighted L2
(score-matching) loss

```
L(θ) = E[w(σ)·||D_θ(y + σ·ε, σ) − y||²],
```

where `log σ ~ N(p_mean, p_std²)` is the EDM log-normal schedule
(defaults `p_mean = -1.2`, `p_std = 1.2`, `σ ∈ [0.002, 80]`, "unless
otherwise specified", paper Appendix E preamble) and `w` is the EDM
weighting (Appendix C.2, verbatim: "with σ_data = 0.5 for all experiments"):

```
w(σ) = (σ² + σ_data²)/(σ·σ_data)²
```

A network of `L` layers is partitioned into `B` blocks; block `b` is trained
independently on the noise range `[σ_b, σ_{b-1}]`. The ranges come from the
**equi-probability partition** of the log-normal: every block sees the same
probability mass of noise,

```
q_b = q_min + (b/B)·(q_max − q_min)
q_min/max = Φ((ln σ_min/max − p_mean)/p_std)
σ_b = exp(p_mean + p_std·Φ⁻¹(q_b))
```

Block `b`'s loss depends on `z_σ = y + σ·ε`, a fixed leaf with no gradient
path through earlier blocks, so `.backward()` stops inside block `b`: peak
training memory is one block's activations instead of the whole unrolled
network (memory /B vs BPTT).

At inference the blocks compose via Euler steps of the probability-flow ODE
`dz/dσ = (z − D(z, σ))/σ`, walking from high noise (`z = x` at `σ_max`) down
to `σ_min`:

```
z_b = z_{b-1} + (Δσ_b/σ_{b-1})·(z_{b-1} − D_θ(x, z_{b-1}))
```

## Install

```bash
cargo add burn-diffusionblocks
```

## Usage

Two usage modes, mirroring the paper:

**Recurrent-depth networks (no partitioning)** — the whole looped network is
one denoiser (paper Appendix E.5: "Unlike other architectures, recurrent-depth
models do not require block partitioning"). Sample `σ` from the full
log-normal, corrupt once, run a single forward through the loop:

```rust
use burn_diffusionblocks::{NoiseSchedule, add_noise, denoising_loss};

let schedule = NoiseSchedule::default();
let mut rng = fastrand::Rng::with_seed(42);

let sigma = schedule.sample_sigma_full(&mut rng);   // σ ~ full log-normal
let noisy = add_noise(hidden_clean, sigma, &device);
let loss = denoising_loss(
    loop_net.forward(noisy, sigma),   // one forward through the loop
    hidden_clean,
    sigma,
    schedule.weight(sigma),
);
// grads = loss.backward();  // BPTT through the loop
```

**Partitioned architectures (ViT/DiT-style)** — split the network into `B`
blocks, each trained on its own restricted noise range:

```rust
use burn_diffusionblocks::{BlockPartition, NoiseSchedule, add_noise, blockwise_step};

let schedule = NoiseSchedule::default();
let partition = BlockPartition::new(schedule, 8);

// block b of B, independently:
let sigma = partition.sample_sigma(3, &mut rng);     // restricted to block 3
let noisy = add_noise(clean, sigma, &device);
let loss = blockwise_step(|z| block.forward(z, sigma), clean, noisy, schedule.weight(sigma));
```

Composition at inference walks blocks from high to low noise via
[`NoiseSchedule::euler_step`].

## API

| Export | What |
|--------|------|
| `NoiseSchedule` | EDM log-normal VE schedule: `p_mean`, `p_std`, `σ_min/max`, `σ_data`; `weight(σ)` (EDM w), `sample_sigma(b, B, rng)` (restricted), `sample_sigma_full(rng)`, `partition(b, B)`, `euler_step(z, D, σ_prev, σ_next)` |
| `BlockPartition` | Precomputed equi-probability partition: `range_for(b)`, `sample_sigma(b, rng)`, `sigma_edges()` |
| `add_noise` | `z_σ = y + σ·ε` (leaf — cuts the cross-block gradient graph) |
| `denoising_loss` | `w·mean(||pred − y||²)` for one sample |
| `blockwise_step` | Run a block's denoiser and return the weighted denoising loss to `.backward()` |

## Honest limits

- **Preconditioning is deferred to EDM.** The paper's `Loss` (Eq 1) is the
  plain weighted L2 on the raw x0 prediction; this crate implements exactly
  that. EDM-style preconditioning of `D_θ` (the `c_skip`/`c_in`/`c_out`
  parameterization) is not applied.
- **The noise-conditioning architecture is the user's choice.** FiLM
  scale/shift or an extra input channel for `σ` is not mandated by the paper
  and not provided; the crate only passes `σ` through to your forward.
- **Recurrent-depth = single denoiser, no partitioning.** The crate ships
  the schedule + objective; the model partitioning itself is the user's job
  (`BlockPartition` only precomputes the noise ranges).
- **Measured claim:** none yet — this is a fresh crate. Core schedule and
  objective are implemented and tested (equal-mass partition, inverse-CDF
  sampling, EDM weight, Euler-step exactness, gradient isolation per block);
  full model-partitioning machinery and a training-run comparison are next.

## Tests

```bash
cargo test -p burn-diffusionblocks
cargo clippy -p burn-diffusionblocks --all-targets -- -D warnings
```

## License

MIT. See [LICENSE](LICENSE).
