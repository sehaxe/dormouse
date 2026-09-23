//! # burn-diffusionblocks
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Block-wise training of residual networks as diffusion denoisers, from
//! [DiffusionBlocks](https://arxiv.org/abs/2506.14202) (Sakana AI,
//! ICLR 2026).
//!
//! ## The method
//!
//! A residual network is reinterpreted as a VE (variance-exploding)
//! diffusion model. Corrupt the target state with Gaussian noise,
//!
//! ```text
//! z_σ = y + σ·ε,   ε ~ N(0, I),
//! ```
//!
//! and train an x0-prediction denoiser `D_θ(z_σ, σ)` — it maps noisy input
//! to the clean target directly — with the weighted L2 loss
//!
//! ```text
//! L(θ) = E[w(σ)·||D_θ(y + σ·ε, σ) − y||²],
//! ```
//!
//! where `log σ ~ N(p_mean, p_std²)` follows the EDM log-normal schedule
//! (defaults `p_mean = -1.2`, `p_std = 1.2`, `σ_min = 0.002`, `σ_max = 80`;
//! "unless otherwise specified", Appendix E preamble) and `w` is the EDM
//! weighting `w(σ) = (σ² + σ_data²)/(σ·σ_data)²`, `σ_data = 0.5`
//! (Appendix C.2). A network of `L` layers is partitioned into `B` blocks;
//! block `b` is trained *independently* on the noise range
//! `[σ_b, σ_{b-1}]`:
//!
//! ```text
//! L_b(θ_b) = E_{σ~p_noise^(b), ε}[w(σ)·Loss(f̄_{θ_b|σ}(x, y + σ·ε), y)],
//! ```
//!
//! with `p_noise^(b)` the log-normal restricted and renormalized to the
//! block's range. The ranges come from the equi-probability partition
//! `q_b = q_min + (b/B)·(q_max − q_min)`, `q_min/max = Φ((ln σ_min/max −
//! p_mean)/p_std)`, `σ_b = exp(p_mean + p_std·Φ⁻¹(q_b))`: every block sees
//! the same probability mass of noise. At inference the blocks compose via
//! an Euler step of the probability-flow ODE `dz/dσ = (z − D(z, σ))/σ`,
//! walking from `σ_max` (the input) down to `σ_min`:
//!
//! ```text
//! z_b = z_{b-1} + (Δσ_b/σ_{b-1})·(z_{b-1} − D_θ(x, z_{b-1}))
//! ```
//!
//! ## The 3-step conversion (partitioned architectures)
//!
//! The conversion below applies to architectures trained with block
//! partitioning; recurrent-depth models skip partitioning entirely (see
//! below) and only need the schedule + objective.
//!
//! 1. **Partition**: [`NoiseSchedule::partition`] / [`BlockPartition`] splits
//!    `[σ_min, σ_max]` into `B` equal-mass noise intervals.
//! 2. **Noise ranges**: [`NoiseSchedule::sample_sigma`] draws a block's
//!    restricted log-normal `σ`; [`add_noise`] builds the corrupted target
//!    `y + σ·ε`.
//! 3. **Noise conditioning**: the block's forward must take `σ` (e.g. a
//!    FiLM scale/shift or an extra input channel) so it can learn
//!    `f̄_{θ_b|σ}`; [`blockwise_step`] runs it and returns the weighted
//!    denoising loss to `.backward()`.
//!
//! ## Memory /B
//!
//! Block `b`'s loss depends on `y + σ·ε`, a fixed leaf with no gradient
//! path through earlier blocks. Backward therefore stops inside block `b`:
//! no cross-block activations are kept, so peak training memory is one
//! block's activations instead of the whole unrolled network — a `B×`
//! reduction vs BPTT.
//!
//! ## Recurrent-depth usage sketch
//!
//! The simplest beneficiary is a looped ("recurrent-depth") network: the
//! same residual block applied `T` times. Unlike the partitioned
//! architectures, recurrent-depth models do NOT use block partitioning —
//! per Appendix E.5 (verbatim): "Unlike other architectures, recurrent-depth
//! models do not require block partitioning since the entire network is
//! applied recurrently. Instead, we train the full network as a denoiser by
//! sampling different noise levels σ at each training step." The WHOLE
//! looped network is ONE denoiser: each step samples σ from the FULL
//! log-normal `p_σ` (no per-block restriction), corrupts the clean hidden
//! state once (`z_σ = y + σε`), runs a SINGLE forward pass through the loop,
//! and optimizes the weighted L2 against the clean target (Appendix B: "we
//! train the network as a denoiser D_θ(z_σ, x, σ) by sampling σ ∼ p_σ and
//! performing a single forward pass to map noisy input to clean output").
//! No per-iteration blocks, no per-iteration targets, no BPTT-based
//! partitioning — the schedule is used only to sample σ, never to partition:
//!
//! ```text
//! sigma = schedule.sample_sigma_full(&mut rng)  # σ ~ full log-normal p_σ
//! noisy = add_noise(hidden_clean, sigma, &device)
//! loss  = denoising_loss(                       # ONE forward through the loop
//!             loop_net.forward(noisy, sigma),   #   (the denoiser)
//!             hidden_clean, sigma,
//!             schedule.weight(sigma))
//! grads = loss.backward()                       # BPTT through the loop
//! ```
//!
//! [`BlockPartition`] stays for the partitioned architectures (ViT/DiT-style
//! networks split into `B` blocks, each trained independently on its own
//! restricted noise range, composed by the Euler step at inference). For
//! recurrent-depth, use the schedule + objective WITHOUT partitioning.
//!
//! Noise conditioning (e.g. FiLM scale/shift or an extra input channel) is
//! an implementation choice here: the paper does not mandate a specific
//! σ-conditioning for recurrent-depth models, so the docs stay honest about
//! that. If the network is σ-conditioned, pass `sigma` alongside the noisy
//! input as above; a plain `denoiser(z)` without conditioning also fits
//! [`denoising_loss`].
//!
//! ## Example
//!
//! ```
//! use burn::nn::LinearConfig;
//! use burn::tensor::Tensor;
//! use burn_diffusionblocks::{
//!     BlockPartition, NoiseSchedule, add_noise, blockwise_step, denoising_loss,
//! };
//!
//! let device = burn::tensor::Device::ndarray();
//! let schedule = NoiseSchedule::default();
//! let partition = BlockPartition::new(schedule, 8);
//! let mut rng = fastrand::Rng::with_seed(42);
//!
//! // One independent block-training step (block 3 of 8):
//! let block = LinearConfig::new(16, 16).init(&device);
//! let clean = Tensor::<2>::random(
//!     [4, 16],
//!     burn::tensor::Distribution::Default,
//!     &device,
//! );
//!
//! let sigma = partition.sample_sigma(3, &mut rng);
//! let noisy = add_noise(clean.clone(), sigma, &device);
//! let loss = blockwise_step(|z| block.forward(z), clean, noisy, schedule.weight(sigma));
//! let loss_value: f32 = loss.into_scalar();
//! assert!(loss_value.is_finite());
//!
//! // Composition at inference: walk high -> low noise. The denoiser here is
//! // a stand-in; in a real model each block carries its own σ-conditioning.
//! let x = Tensor::<2>::random([4, 16], burn::tensor::Distribution::Default, &device);
//! let mut z = x.clone();
//! for b in (0..partition.n_blocks()).rev() {
//!     let (sigma_high, sigma_low) = partition.range_for(b);
//!     let denoised = z.clone(); // D_θ(x, z) of block b
//!     z = NoiseSchedule::euler_step(
//!         z,
//!         denoised,
//!         sigma_high as f32,
//!         sigma_low as f32,
//!     );
//! }
//! let _ = denoising_loss(z, x, 0.0, 1.0);
//! ```

pub mod blocks;
pub mod noise;
pub mod objective;

pub use blocks::BlockPartition;
pub use noise::NoiseSchedule;
pub use objective::{add_noise, blockwise_step, denoising_loss};

mod normcdf;
