//! Token sampling helpers (DeepSpec `utils/sampling.py`).
use burn::tensor::{Int, Tensor};

/// Sample from logits: argmax when `temperature < 1e-5`, otherwise
/// softmax(logits / t) categorical sampling (DeepSpec `sample_tokens`).
///
/// `logits`: `[B, L, V]`, returns ids `[B, L, Int]`.
pub fn sample_tokens(logits: Tensor<3>, temperature: f32) -> Tensor<2, Int> {
    let [b, l, v] = logits.dims();
    if temperature < 1e-5 {
        return logits.argmax(2).reshape([b, l]);
    }
    let probs = burn::tensor::activation::softmax(logits.div_scalar(temperature), 2);
    let flat = probs.reshape([b * l, v]);
    let sampled: Tensor<2, Int> = flat.categorical(1).reshape([b, l]);
    sampled
}

/// Residual sampling for speculative verification: sample from the
/// normalized excess `clamp(p_target - p_draft, 0)` (DeepSpec
/// `sample_residual`). Falls back to the target distribution where the
/// residual mass is ~0.
pub fn sample_residual(target_probs: Tensor<3>, draft_probs: Tensor<3>) -> Tensor<2, Int> {
    let [b, l, v] = target_probs.dims();
    let residual = target_probs.clone().sub(draft_probs).clamp_min(0.0);
    let mass = residual.clone().sum_dim(2); // [B, L, 1] (0.22 sum_dim keeps the dim)
    let safe = mass.clone().lower_equal_elem(1e-8); // [B, L, 1]
    let residual = residual.mask_where(safe.clone().expand([b, l, v]), target_probs);
    let ones = Tensor::<3>::ones([b, l, 1], &mass.device());
    let mass = mass.mask_where(safe, ones);
    let normalized = residual.div(mass);
    let flat = normalized.reshape([b * l, v]);
    flat.categorical(1).reshape([b, l])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dev() -> burn::tensor::Device {
        burn::tensor::Device::ndarray()
    }

    #[test]
    fn argmax_when_cold() {
        // deterministic: argmax position 2
        let logits = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(
                vec![0.1f32, 0.2, 10.0, 0.0, 0.5, 0.3, 0.9, 0.1],
                [1, 2, 4],
            ),
            &dev(),
        );
        let ids = sample_tokens(logits, 0.0);
        let v: Vec<i64> = ids.into_data().try_to_vec().unwrap();
        assert_eq!(v, vec![2, 2]);
    }

    #[test]
    fn temperature_sampling_in_range() {
        let logits = Tensor::<3>::zeros([2, 8, 32], &dev());
        let ids = sample_tokens(logits, 1.0);
        let v: Vec<i64> = ids.into_data().try_to_vec().unwrap();
        assert!(v.iter().all(|&x| (0..32).contains(&x)));
    }

    #[test]
    fn residual_samples_where_draft_low() {
        // draft puts all mass on token 0; target on token 1 -> residual picks 1
        let mut data = vec![0.0f32; 2 * 4];
        data[0] = 1.0; // draft row 0: token 0
        data[5] = 1.0; // draft row 1: token 1
        let draft = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(data.clone(), [1, 2, 4]),
            &dev(),
        );
        let mut tdata = vec![0.0f32; 2 * 4];
        tdata[1] = 1.0; // target row 0: token 1
        tdata[4] = 1.0; // target row 1: token 0
        let target =
            Tensor::<3>::from_data(burn::tensor::TensorData::new(tdata, [1, 2, 4]), &dev());
        let ids = sample_residual(target, draft);
        let v: Vec<i64> = ids.into_data().try_to_vec().unwrap();
        assert_eq!(v, vec![1, 0]);
    }
}
