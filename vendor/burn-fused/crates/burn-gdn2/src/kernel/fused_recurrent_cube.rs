//! Fused per-token recurrence kernel (single kernel launch per decode step).
//!
//! Replaces ~8 tensor ops per token with one cubecl launch. Only used on the
//! CUDA backend (feature `cuda`); all other backends fall back to the tensor
//! ops in `super::fused_recurrent`.

use cubecl::prelude::*;

/// One cube per (batch, head); one lane per value channel.
///
/// The recurrence S ← S·diag(exp(g)) + k ⊗ (w⊙v − (b⊙k)ᵀS) is executed in two
/// passes over the key dimension so each value channel only needs one register
/// for the erase term (b⊙k)ᵀS instead of shared memory.
#[cube(launch_unchecked)]
fn gdn2_step_kernel<F: Float>(
    q: &[F],
    k: &[F],
    v: &[F],
    g: &[F],
    b: &[F],
    w: &[F],
    state_in: &[F],
    state_out: &mut [F],
    out: &mut [F],
    k_dim: u32,
    v_dim: u32,
    scale: f32,
) {
    let n = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let kd = k_dim as usize;
    let vd = v_dim as usize;
    let k_base = n * kd;
    let v_base = n * vd;
    let s_base = n * kd * vd;

    // Pass 1: accumulate the erase term (b⊙k)ᵀ S.
    let mut erased = F::new(0.0_f32);
    for kk in 0..kd {
        let s = state_in[s_base + kk * vd + lane];
        let eg = F::exp(g[k_base + kk]);
        erased += s * eg * b[k_base + kk] * k[k_base + kk];
    }
    // v_new = w ⊙ v - erased
    let v_new = w[v_base + lane] * v[v_base + lane] - erased;

    // Pass 2: S ← S + k ⊗ v_new (write into the new state), then o = qᵀ S * scale.
    let mut acc = F::new(0.0_f32);
    for kk in 0..kd {
        let s = state_in[s_base + kk * vd + lane];
        let eg = F::exp(g[k_base + kk]);
        let s2 = s * eg + k[k_base + kk] * v_new;
        state_out[s_base + kk * vd + lane] = s2;
        acc += s2 * q[k_base + kk];
    }
    out[v_base + lane] = acc * F::cast_from(scale);
}

#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;
    use burn::backend::{Backend, DispatchKindConversion};
    use burn::tensor::{DispatchTensor, Tensor};
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::CubeBackend;
    use std::any::{Any, TypeId};

    /// The bare (non-fusion) CUDA backend the fused kernel targets.
    ///
    /// The default `burn_cuda::Cuda` backend is `Fusion<CubeBackend<...>>`, and
    /// the fusion layer has no public way to hand out its underlying
    /// `CubeTensor` without a host round-trip, so fused dispatch only activates
    /// on this type (and on `burn_cuda::Cuda` built without the `fusion`
    /// feature, which aliases to it). Everything else falls back to tensor ops.
    pub type CudaBare = CubeBackend<cubecl::cuda::CudaRuntime>;

    fn is_cuda<B: Backend>() -> bool {
        TypeId::of::<B>() == TypeId::of::<CudaBare>()
    }

    /// Owned copy of the underlying `CubeTensor` of `t`.
    ///
    /// Returns `None` when `B` is not a CUDA `CubeBackend` (e.g. the
    /// fusion-wrapped `burn_cuda::Cuda`, or a non-CUDA backend) — the caller
    /// then falls back to the tensor-ops path. Dims of size 1 are ignored, so
    /// permuted single-token views still qualify.
    fn cube_of<B: Backend>(t: &Tensor<4>) -> Option<CubeTensor<cubecl::cuda::CudaRuntime>>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        if !is_cuda::<B>() {
            return None;
        }
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let cube = (&prim as &dyn Any).downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
        let shape = cube.meta.shape().dims::<4>();
        let strides = cube.meta.strides().to_vec();
        let mut expected = 1usize;
        for i in (0..4).rev() {
            if shape[i] > 1 && strides[i] != expected {
                return None; // non-contiguous buffer, use the tensor path
            }
            expected *= shape[i];
        }
        Some(cube.clone())
    }

    /// Run one fused decode step on CUDA. Returns `None` when the backend is
    /// not the bare CUDA `CubeBackend` or the dimensions exceed the kernel
    /// limits, in which case the caller falls back to the tensor-ops path.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_step<B: Backend>(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        g: Tensor<4>,
        b: Tensor<4>,
        w: Tensor<4>,
        state: Tensor<4>,
        scale: f64,
    ) -> Option<(Tensor<4>, Tensor<4>)>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [batch, hv, _, _] = q.shape().dims::<4>();
        let k_dim = q.shape().dims::<4>()[3];
        let v_dim = v.shape().dims::<4>()[3];
        let heads = batch * hv;
        if k_dim > 1024 || v_dim > 1024 || heads > 65535 {
            return None; // kernel limit, fall back
        }

        let device = state.device();
        let q = cube_of::<B>(&q)?;
        let k = cube_of::<B>(&k)?;
        let v = cube_of::<B>(&v)?;
        let g = cube_of::<B>(&g)?;
        let b = cube_of::<B>(&b)?;
        let w = cube_of::<B>(&w)?;
        let state = cube_of::<B>(&state)?;

        let client = state.client.clone();
        let n_heads = heads * v_dim;
        let n_keys = heads * k_dim;
        let n_state = heads * k_dim * v_dim;

        let out = Tensor::<4>::zeros([batch, hv, 1, v_dim], &device);
        let new_state = Tensor::<4>::zeros([batch, hv, k_dim, v_dim], &device);
        let out_cube = cube_of::<B>(&out).expect("backend mismatch");
        let new_state_cube = cube_of::<B>(&new_state).expect("backend mismatch");

        let cube_dim = CubeDim::new_3d(v_dim as u32, 1, 1);
        let cube_count = CubeCount::Static(heads as u32, 1, 1);
        unsafe {
            gdn2_step_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(q.handle, n_keys),
                BufferArg::from_raw_parts(k.handle, n_keys),
                BufferArg::from_raw_parts(v.handle, n_heads),
                BufferArg::from_raw_parts(g.handle, n_keys),
                BufferArg::from_raw_parts(b.handle, n_keys),
                BufferArg::from_raw_parts(w.handle, n_heads),
                BufferArg::from_raw_parts(state.handle, n_state),
                BufferArg::from_raw_parts(new_state_cube.handle, n_state),
                BufferArg::from_raw_parts(out_cube.handle, n_heads),
                k_dim as u32,
                v_dim as u32,
                scale as f32,
            );
        }

        Some((out, new_state))
    }
}
