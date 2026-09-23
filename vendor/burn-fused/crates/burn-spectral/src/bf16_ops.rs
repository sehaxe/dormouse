//! True bf16 compute for the spectral factors: forward runs the matmul on
//! bf16 (tensor cores), the autodiff graph and the backward stay in fp32.
//!
//! This is the honest mixed-precision design: the plain "bf16 storage, fp32
//! compute" cast scheme only saved memory, never speed. Here the dominant
//! FLOPs (the factorized FFN matmuls) actually execute in bf16, while the
//! backward differentiates the fp32 factors - exact gradients, Moonshot-style
//! training numerics.

use burn::backend::{AutodiffBackend, Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, FloatDType, Tensor};
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::NoCheckpointing;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_autodiff::{Autodiff, NodeId};

#[derive(Debug)]
struct Bf16Matmul;

impl<B: Backend> Backward<B, 2> for Bf16Matmul
where
    DispatchTensor: DispatchKindConversion<B>,
{
    type State = [Option<NodeId>; 2];

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let [id_a, id_w] = ops.state;
        let d_out = Tensor::<2>::from_primitive::<B>(grads.consume::<B>(&ops.node));
        let node = |i: usize| ops.parents[i].as_ref().expect("bf16 matmul input tracked");
        if let Some(idw) = id_w {
            let w = Tensor::<2>::from_primitive::<B>(checkpointer.retrieve_node_output(idw));
            // dA = dOut @ W^T
            let da = d_out.clone().matmul(w.transpose());
            grads.register::<B>(node(0).id, da.try_into_primitive::<B>().unwrap());
        }
        if let Some(ida) = id_a {
            let a = Tensor::<2>::from_primitive::<B>(checkpointer.retrieve_node_output(ida));
            // dW = A^T @ dOut
            let dw = a.transpose().matmul(d_out);
            grads.register::<B>(node(1).id, dw.try_into_primitive::<B>().unwrap());
        }
    }
}

/// `y = A @ W` computed on bf16 with an fp32 autodiff graph and exact fp32
/// backward gradients. Inputs must be fp32 tensors on `Autodiff<Inner>`.
pub fn bf16_matmul<Inner: Backend>(a: Tensor<2>, w: Tensor<2>) -> Tensor<2>
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
{
    let a = if matches!(a.clone().dtype(), burn::tensor::DType::F32) {
        a
    } else {
        a.cast(FloatDType::F32)
    };
    let w = if matches!(w.clone().dtype(), burn::tensor::DType::F32) {
        w
    } else {
        w.cast(FloatDType::F32)
    };
    let a_ad = a.clone().try_into_primitive::<Autodiff<Inner>>().unwrap();
    let w_ad = w.clone().try_into_primitive::<Autodiff<Inner>>().unwrap();
    // Forward in bf16 on the inner backend (tensor cores), fp32 output.
    let a_t = Tensor::<2>::from_primitive::<Inner>(a_ad.primitive.clone());
    let w_t = Tensor::<2>::from_primitive::<Inner>(w_ad.primitive.clone());
    let out = a_t
        .cast(FloatDType::BF16)
        .matmul(w_t.cast(FloatDType::BF16))
        .cast(FloatDType::F32);
    let out_p = out.try_into_primitive::<Inner>().unwrap();

    let nodes = [a_ad.node.clone(), w_ad.node.clone()];
    let prep = Bf16Matmul.prepare::<NoCheckpointing>(nodes);
    match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut prep) => {
            let ids = [Some(prep.checkpoint(&a_ad)), Some(prep.checkpoint(&w_ad))];
            let out = prep.finish(ids, out_p);
            Tensor::from_primitive::<Autodiff<Inner>>(out)
        }
        OpsKind::UnTracked(prep) => Tensor::from_primitive::<Autodiff<Inner>>(prep.finish(out_p)),
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type AD = burn::backend::autodiff::Autodiff<
        burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>,
    >;
    type Bare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;

    #[test]
    fn bf16_matmul_matches_fp32_with_grads() {
        let d = burn::tensor::Device::default().autodiff();
        let a: Tensor<2> = Tensor::random([256, 512], Distribution::Normal(0.0, 1.0), &d).require_grad();
        let w: Tensor<2> = Tensor::random([512, 256], Distribution::Normal(0.0, 1.0), &d).require_grad();
        let y = bf16_matmul::<Bare>(a.clone(), w.clone());
        // fp32 reference on the same backend (cast path is identity here).
        let y_ref = a.clone().matmul(w.clone());
        let diff: f32 = (y.clone() - y_ref.clone()).abs().mean().into_scalar();
        let scale: f32 = y_ref.abs().mean().into_scalar();
        let rel = diff / scale.max(1e-6);
        assert!(rel < 0.01, "bf16 forward too far from fp32: rel {rel:.4}");
        let grads = y.sum().backward();
        let ga: Vec<f32> = a.clone().grad(&grads)
            .map(|t| t.into_data().try_to_vec().unwrap_or_default())
            .unwrap_or_default();
        let gw: Vec<f32> = w.clone().grad(&grads)
            .map(|t| t.into_data().try_to_vec().unwrap_or_default())
            .unwrap_or_default();
        assert_eq!(ga.len(), 256 * 512, "grad a must arrive");
        assert_eq!(gw.len(), 512 * 256, "grad w must arrive");
        assert!(
            ga.iter().chain(gw.iter()).all(|x| x.is_finite()),
            "bf16 backward grads must be finite"
        );
    }

    #[test]
    fn bf16_matmul_ffn_sizes_finite() {
        // Real FFN shapes: [768, 2048] x [2048, 768], two chained matmuls
        // with an elementwise scale in between (the spectral y=(x@U)*s@Vt).
        let d = burn::tensor::Device::default().autodiff();
        let x: Tensor<2> = Tensor::random([512, 768], Distribution::Normal(0.0, 1.0), &d)
            .require_grad();
        let u: Tensor<2> = Tensor::random([768, 64], Distribution::Normal(0.0, 1.0), &d)
            .require_grad();
        let v: Tensor<2> = Tensor::random([2048, 64], Distribution::Normal(0.0, 1.0), &d)
            .require_grad();
        let s: Tensor<1> = Tensor::ones([64], &d);
        let mid = bf16_matmul::<Bare>(x.clone(), u.clone());
        let y = bf16_matmul::<Bare>(mid.mul(s.unsqueeze_dim::<2>(0)), v.clone().transpose());
        let loss = y.sum();
        let v0: f32 = loss.clone().into_scalar();
        assert!(v0.is_finite(), "bf16 FFN forward must be finite");
        let grads = loss.backward();
        for (name, t, n) in [
            ("x", x.clone(), 512 * 768),
            ("u", u.clone(), 768 * 64),
            ("v", v.clone(), 2048 * 64),
        ] {
            let g: Vec<f32> = t
                .clone()
                .grad(&grads)
                .map(|g| g.into_data().try_to_vec().unwrap_or_default())
                .unwrap_or_default();
            assert_eq!(g.len(), n, "grad {name} must arrive");
            assert!(
                g.iter().all(|x| x.is_finite()),
                "grad {name} must be finite"
            );
        }
    }
}