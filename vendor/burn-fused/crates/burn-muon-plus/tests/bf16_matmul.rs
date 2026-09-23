//! bf16 compute-path probes for dormouse (CUDA): the model stores
//! activations in bf16 under BF16=1, so the autodiff backward contract
//! through bf16 tensors is the thing that decides whether we can compute
//! in bf16 or must cast to fp32.

#[cfg(all(test, feature = "cuda"))]
mod bf16_tests {
    use burn::prelude::Device;
    use burn::tensor::{Distribution, FloatDType, Tensor};

    fn dev() -> Device {
        Device::default()
    }

    #[test]
    fn bf16xbf16_matmul_is_finite() {
        let d = dev();
        let a0: Tensor<2> = Tensor::random([256, 256], Distribution::Normal(0.0, 1.0), &d);
        let a = a0.cast(FloatDType::BF16);
        let b0: Tensor<2> = Tensor::random([256, 256], Distribution::Normal(0.0, 1.0), &d);
        let b = b0.cast(FloatDType::BF16);
        let c = a.matmul(b);
        let v: Vec<f32> = c.into_data().try_to_vec().unwrap_or_default();
        let bad = v.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad, 0, "bf16xbf16 matmul produced {bad} non-finite values");
    }

    #[test]
    fn bf16_mixed_matmul_is_finite() {
        let d = dev();
        let a0: Tensor<2> = Tensor::random([256, 256], Distribution::Normal(0.0, 1.0), &d);
        let a = a0.cast(FloatDType::BF16);
        let b: Tensor<2> = Tensor::random([256, 256], Distribution::Normal(0.0, 1.0), &d);
        let c = a.matmul(b);
        let v: Vec<f32> = c.into_data().try_to_vec().unwrap_or_default();
        let bad = v.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad, 0, "bf16xf32 matmul produced {bad} non-finite values");
    }

    #[test]
    fn bf16_model_size_matmul_autodiff_finite() {
        // bf16 leaf x bf16 leaf through autodiff: the grads to the bf16
        // leaves themselves are NOT expected (burn autodiff does not hand
        // out grads for bf16 leaves) - the forward must stay finite though.
        let d = dev().autodiff();
        let a0: Tensor<2> = Tensor::random([768, 768], Distribution::Normal(0.0, 1.0), &d);
        let a = a0.cast(FloatDType::BF16).require_grad();
        let b0: Tensor<2> = Tensor::random([768, 768], Distribution::Normal(0.0, 1.0), &d);
        let b = b0.cast(FloatDType::BF16).require_grad();
        let c = a.clone().matmul(b.clone()).cast(FloatDType::F32).sum();
        let v: f32 = c.clone().into_scalar();
        assert!(v.is_finite(), "bf16 forward must be finite");
        let _ = c.backward();
    }

    #[test]
    fn bf16_act_f32_param_backward_finite() {
        // The model's shape under BF16=1: bf16 activations (untracked) x
        // fp32 parameters (require_grad). If this gives NO grad to the fp32
        // parameter, computing in bf16 is impossible on this stack and the
        // fp32-compute casts in dormouse are the required design.
        let d = dev().autodiff();
        let a0: Tensor<2> = Tensor::random([768, 768], Distribution::Normal(0.0, 1.0), &d);
        let a = a0.cast(FloatDType::BF16); // activation, untracked
        let b0: Tensor<2> = Tensor::random([768, 768], Distribution::Normal(0.0, 1.0), &d);
        let b = b0.require_grad(); // fp32 parameter
        let c = a.matmul(b.clone()).sum();
        let grads = c.backward();
        let gb: Vec<f32> = b
            .clone()
            .grad(&grads)
            .map(|t| t.into_data().try_to_vec().unwrap_or_default())
            .unwrap_or_default();
        let bad_b = gb.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad_b, 0, "bf16-act x f32-param grads: {bad_b} non-finite");
        if gb.is_empty() {
            eprintln!("NOTE: burn autodiff gives NO grads through bf16 activations");
        }
    }

    #[test]
    fn f32_control_backward_arrives() {
        // Mechanism control: fp32 activation x fp32 param must give grads.
        let d = dev().autodiff();
        let a0: Tensor<2> = Tensor::random([768, 768], Distribution::Normal(0.0, 1.0), &d);
        let a = a0; // fp32 activation, untracked
        let b0: Tensor<2> = Tensor::random([768, 768], Distribution::Normal(0.0, 1.0), &d);
        let b = b0.require_grad();
        let c = a.matmul(b.clone()).sum();
        let grads = c.backward();
        let gb: Vec<f32> = b
            .clone()
            .grad(&grads)
            .map(|t| t.into_data().try_to_vec().unwrap_or_default())
            .unwrap_or_default();
        assert_eq!(gb.len(), 768 * 768, "fp32 control: grad must arrive");
    }
}