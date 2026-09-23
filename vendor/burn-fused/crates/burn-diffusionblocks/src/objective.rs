//! Score-matching objective and per-block training helpers.

use burn::tensor::{Device, Distribution, Tensor};

/// Corrupt a clean target with VE noise: `z_σ = y + σ·ε`, `ε ~ N(0, I)`.
///
/// Drawn on `device` with the backend RNG. The result is a leaf (no gradient
/// path to `y`), which is what makes block-wise training cut the
/// cross-block graph.
pub fn add_noise<const D: usize>(y: Tensor<D>, sigma: f32, device: &Device) -> Tensor<D> {
    let noise = Tensor::<D>::random(y.shape(), Distribution::Normal(0.0, 1.0), device);
    y.add(noise.mul_scalar(sigma))
}

/// Denoising (score-matching) loss for one sample:
/// `L = w · mean(||pred − target||²)`, mean over all elements.
///
/// The denoiser is an **x0-prediction** model: `pred` is the model's direct
/// estimate of the clean target `y` (not the noise `ε`), so the L2 is taken
/// against `target = y` itself. `sigma` is informational: the weight is
/// expected to be `schedule.weight(sigma)`; the loss itself depends only on
/// `w`.
pub fn denoising_loss(pred: Tensor<2>, target: Tensor<2>, sigma: f32, w: f32) -> Tensor<1> {
    let _ = sigma;
    pred.sub(target).powf_scalar(2.0).mean().mul_scalar(w)
}

/// One independent block-training step.
///
/// Runs the denoiser `denoiser` (the block's forward with noise-level
/// conditioning, `D_θ(z_σ, σ)`) on the noisy input and returns
/// `w·mean(||D_θ(z_σ, σ) − y||²)` — [`denoising_loss`] against the clean
/// target. The denoiser is an x0-prediction model: it outputs its estimate
/// of the clean `y` directly, and the L2 is taken against `y`. Because
/// `noisy` (e.g. from [`add_noise`]) carries no gradient path to anything
/// but `denoiser`'s own parameters, calling `.backward()` on the result
/// flows gradients ONLY into the active block: no other block's parameters
/// or activations are on the graph, so peak memory is one block's worth
/// (memory /B vs full BPTT).
pub fn blockwise_step(
    denoiser: impl Fn(Tensor<2>) -> Tensor<2>,
    clean: Tensor<2>,
    noisy: Tensor<2>,
    w: f32,
) -> Tensor<1> {
    denoising_loss(denoiser(noisy), clean, 0.0, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::LinearConfig;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn noise_roundtrip() {
        let device = dev();
        let schedule = crate::NoiseSchedule::default();
        let y = Tensor::<2>::random([8, 16], Distribution::Default, &device);
        let sigma = 0.5f32;
        let w = schedule.weight(sigma);
        let z = add_noise(y.clone(), sigma, &device);

        // noisy input is finite, and the identity denoiser's loss matches
        // the closed form w·mean((z − y)²) exactly
        let max_z: f32 = z.clone().abs().max().into_scalar();
        assert!(max_z.is_finite(), "noisy input must be finite, got {max_z}");
        let loss_identity = denoising_loss(z.clone(), y.clone(), sigma, w);
        let loss_ref = z.sub(y.clone()).powf_scalar(2.0).mean().mul_scalar(w);
        let loss_identity: f32 = loss_identity.into_scalar();
        let loss_ref: f32 = loss_ref.into_scalar();
        assert!(loss_identity.is_finite() && loss_identity > 0.0);
        assert!(
            (loss_identity - loss_ref).abs() < 1e-5,
            "loss {loss_identity} != closed form {loss_ref}"
        );

        // a prediction closer to y scores lower: the perfect denoiser
        // (pred = y) hits zero
        let loss_perfect: f32 = denoising_loss(y.clone(), y.clone(), sigma, w).into_scalar();
        assert!(loss_perfect.abs() < 1e-6, "perfect denoiser must score ~0");
        assert!(loss_perfect < loss_identity, "loss must decrease towards y");
    }

    #[test]
    fn blockwise_step_backward() {
        let device = Device::ndarray().autodiff();
        let schedule = crate::NoiseSchedule::default();
        let block = LinearConfig::new(16, 16).with_bias(true).init(&device);
        let y = Tensor::<2>::random([4, 16], Distribution::Default, &device);
        let sigma = schedule.sample_sigma(2, 8, &mut fastrand::Rng::with_seed(7));
        let w = schedule.weight(sigma);
        let noisy = add_noise(y.clone(), sigma, &device);

        let loss = blockwise_step(|z| block.forward(z), y, noisy, w);
        let grads = loss.backward();

        // gradients land on the active block's params, finite and nonzero
        let gw: Vec<f32> = block
            .weight
            .grad(&grads)
            .unwrap()
            .into_data()
            .try_to_vec()
            .unwrap();
        let gb: Vec<f32> = block
            .bias
            .as_ref()
            .unwrap()
            .grad(&grads)
            .unwrap()
            .into_data()
            .try_to_vec()
            .unwrap();
        assert!(
            gw.iter().any(|g| g.is_finite() && g.abs() > 1e-8),
            "weight grads dead"
        );
        assert!(
            gb.iter().any(|g| g.is_finite() && g.abs() > 1e-8),
            "bias grads dead"
        );
    }
}
