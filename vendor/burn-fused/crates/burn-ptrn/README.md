# burn-ptrn

> **Not in the dormouse build.** No crate under `crates/dormouse-{core,data,train,cli}/`
> depends on this one; the only incoming edge is the `burn-fused` facade. Kept as a
> **reference port**: test-time scaling is aimed at exactly this project's
> parameter-shared loop, but the mechanism as written scores rollouts with a
> **learned Q-head that ADR-0013 deleted with PonderNet** — `AGENTS.md:645` records
> that the selection rule must be re-specified before this crate can be used at all.
> Fate table: `docs/library-crate-fate.md`.

[![CI](https://github.com/sehaxe/burn-ptrn/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-ptrn/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-ptrn)](https://crates.io/crates/burn-ptrn)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

Probabilistic Tiny Recursive Model (PTRM) — test-time scaling for recursive
(looped, parameter-shared) models via recurrent Gaussian noise + best-Q@K
selection, for [Burn](https://burn.dev).

Reference: **PTRM: Probabilistic Tiny Recursive Models for Test-Time Scaling**
(arXiv [2605.19943](https://arxiv.org/abs/2605.19943)).

## Recipe (no retraining of loop weights)

1. **Recurrent Gaussian noise**: inject `ε ~ N(0, σ²I)` into the latent at every
   recursion step (σ ≈ 0.2–1.0 by task). Noise-only — no gradient term;
   it lets ~8% of rollouts escape bad basins.
2. **Best-Q@K selection**: run K parallel rollouts (width scaling is stronger
   and more practical than depth scaling), score each with a learned Q-head,
   take the argmax trajectory.
3. **Joint Q-head training**: `L_step = CE(f_O(y), y_true) + BCE(q̂, 1[ŷ = y_true])`
   trains the value head so Q separates correct/incorrect rollouts.

All primitives are pure tensor ops (no host branching) — drop into any burn
training loop or recursive inference loop.

## Usage

```rust
use burn_ndarray::NdArray;
use burn_ptrn::{PtrnConfig, QHead, add_recurrent_noise, best_of_k, correctness_target};

let config = PtrnConfig::new(); // σ=0.5, K=16, τ=1.0
let device = NdArrayDevice::default();
let q = QHead::<NdArray>::new(64, &device);

// recursion step, repeated:
z = add_recurrent_noise(z, config.noise_sigma, &device);
z = loop_cell.forward(z);

// after K rollouts (K = config.num_rollouts):
let (best_traj, best_idx) = best_of_k(q_logits, rollouts); // [B,T,D], [B]

// training:
let flag = correctness_target(pred_ids, truth_ids);          // [B,T] Int
let loss = q.q_loss(q.logit(h), flag, Some(mask));           // masked BCE
```

## API

| Item | Signature |
| --- | --- |
| `PtrnConfig` | `{ noise_sigma: f64 = 0.5, num_rollouts: usize = 16, tau: f64 = 1.0 }`, `new()` |
| `QHead` | `Linear(d, 1)`; `logit(h) -> [B,T,1]`, `prob(h) -> [B,T,1]`, `q_loss(logits, target, mask) -> [B,1]` |
| `add_recurrent_noise` | `z [B,T,D] + σ·N(0,I)` |
| `best_of_k` | `q_logits [K,B,T,1], rollouts [K,B,T,D] -> ([B,T,D], [B] Int)` |
| `correctness_target` | `pred [B,T] Int, truth [B,T] Int -> [B,T] Int` (1 = equal) |

## Tests

```bash
cargo test --release
```

## License

MIT — research crate for the [aria](https://github.com/sehaxe/aria) project.
