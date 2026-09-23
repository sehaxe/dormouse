//! Autodiff registration for the fused sparse attention (MSA).
//!
//! The fused forward runs as a single tracked node on `Autodiff<Cuda>`; the
//! backward recomputes the masked attention on the tensor path (all burn ops)
//! and applies the standard attention-backward formulas over the selected
//! blocks. `block_indices` are indices, so no gradient flows to them; the
//! block-attention output `ba` is returned untracked (secondary KL signal).

use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, Int, Tensor};
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::NoCheckpointing;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_autodiff::Autodiff;

use crate::sparse_attn_batched_gqa;

#[derive(Debug, Clone)]
struct SparseAttnState {
    block_indices: Tensor<4, Int>,
    scale: f64,
    block_size: usize,
    n_heads_kv: usize,
    n_heads_q: usize,
    causal: bool,
}

#[derive(Debug)]
struct SparseAttnOp;

impl<B: Backend> Backward<B, 3> for SparseAttnOp
where
    DispatchTensor: DispatchKindConversion<B>,
{
    type State = SparseAttnState;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let node = |i: usize| ops.parents[i].as_ref().expect("msa input checkpointed");
        let q = Tensor::<4>::from_primitive::<B>(checkpointer.retrieve_node_output(node(0).id));
        let k = Tensor::<4>::from_primitive::<B>(checkpointer.retrieve_node_output(node(1).id));
        let v = Tensor::<4>::from_primitive::<B>(checkpointer.retrieve_node_output(node(2).id));
        let st = &ops.state;
        let d_out = Tensor::from_primitive::<B>(grads.consume::<B>(&ops.node));

        #[cfg(feature = "cuda")]
        {
            type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
            if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CudaBare>() {
                if let Some((dq, dk, dv)) = crate::sparse_kernel::msa_backward_cuda(
                    &q,
                    &k,
                    &v,
                    &st.block_indices,
                    st.scale,
                    st.block_size,
                    st.n_heads_kv,
                    st.n_heads_q,
                    st.causal,
                    &d_out,
                ) {
                    for (i, d) in [dq, dk, dv].into_iter().enumerate() {
                        if let Some(node) = ops.parents[i].clone() {
                            grads.register::<B>(node.id, d.try_into_primitive::<B>().unwrap());
                        }
                    }
                    return;
                }
            }
        }

        let (dq, dk, dv) = sparse_attn_backward_tensor(
            &q,
            &k,
            &v,
            &st.block_indices,
            st.scale,
            st.block_size,
            st.n_heads_kv,
            st.n_heads_q,
            st.causal,
            &d_out,
        );

        for (i, d) in [dq, dk, dv].into_iter().enumerate() {
            if let Some(node) = ops.parents[i].clone() {
                grads.register::<B>(node.id, d.try_into_primitive::<B>().unwrap());
            }
        }
    }
}

/// Masked-attention backward over the selected blocks (tensor path).
///
/// Blockwise like the forward (KV-outer order, online softmax recompute):
/// pass 1 recovers the running max/sum, pass 2 recomputes each block's
/// scores and applies the softmax-backward formulas. dk/dv are scattered
/// into the clamped token slots (Add), so peak memory stays linear in
/// seq_kv — no [B, Hkv, qpkv, S, S] matrix at any point.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn_backward_tensor(
    q: &Tensor<4>,
    k: &Tensor<4>,
    v: &Tensor<4>,
    block_indices: &Tensor<4, Int>,
    scale: f64,
    block_size: usize,
    n_heads_kv: usize,
    n_heads_q: usize,
    causal: bool,
    d_out: &Tensor<4>,
) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    let batch = q.dims()[0];
    let seq_q = q.dims()[2];
    let d_head = q.dims()[3];
    let seq_kv = k.dims()[2];
    let qpkv = n_heads_q / n_heads_kv;
    let topk = block_indices.dims()[3];
    let device = q.device();

    let q_s = q
        .clone()
        .reshape::<5, _>([batch, n_heads_kv, qpkv, seq_q, d_head])
        .swap_dims(2, 3); // [B,Hkv,S,qpkv,d]
    let d_out_5 = d_out
        .clone()
        .reshape::<5, _>([batch, n_heads_kv, qpkv, seq_q, d_head]);
    let offsets = Tensor::<1, Int>::arange(0..block_size as i64, &device)
        .reshape::<4, _>([1, 1, 1, block_size]);
    let q_pos =
        Tensor::<1, Int>::arange(0..seq_q as i64, &device).reshape::<4, _>([1, 1, seq_q, 1]);

    // Pass 1: recompute the online softmax running max and sum.
    let mut m = Tensor::<5>::full([batch, n_heads_kv, qpkv, seq_q, 1], -3.0e38_f32, &device);
    let mut l = Tensor::<5>::zeros([batch, n_heads_kv, qpkv, seq_q, 1], &device);
    for blk in 0..topk {
        let (scores_b, _, _, _) = crate::attention::block_scores(
            &q_s,
            k,
            v,
            block_indices,
            &offsets,
            &q_pos,
            blk,
            block_size,
            seq_kv,
            scale,
            causal,
        );
        let m_new = m.clone().max_pair(scores_b.clone().max_dim(4)); // [B,Hkv,qpkv,S,1]
        let e_sum = scores_b.sub(m_new.clone()).exp().sum_dim(4); // [B,Hkv,qpkv,S,1]
        let alpha = m.sub(m_new.clone()).exp(); // [B,Hkv,qpkv,S,1]
        m = m_new;
        l = l.mul(alpha).add(e_sum);
    }

    // Pass 2: the softmax-backward center term is the GLOBAL sum over every
    // attended token (all blocks): wd_sum = sum_b attn_b * d_attn_b.
    let mut wd_sum = Tensor::<5>::zeros([batch, n_heads_kv, qpkv, seq_q, 1], &device);
    for blk in 0..topk {
        let (scores_b, _, v_b, _) = crate::attention::block_scores(
            &q_s,
            k,
            v,
            block_indices,
            &offsets,
            &q_pos,
            blk,
            block_size,
            seq_kv,
            scale,
            causal,
        );
        let attn_b = scores_b.sub(m.clone()).exp().div(l.clone()); // [B,Hkv,qpkv,S,bs]
        let d_attn_b = d_out_5
            .clone()
            .swap_dims(2, 3)
            .matmul(v_b.swap_dims(3, 4))
            .swap_dims(2, 3); // [B,Hkv,qpkv,S,bs]
        wd_sum = wd_sum.add(attn_b.mul(d_attn_b).sum_dim(4)); // [B,Hkv,qpkv,S,1]
    }

    // Pass 3: per-block gradients; dk/dv scatter into the clamped slots.
    let mut dq = Tensor::<5>::zeros([batch, n_heads_kv, qpkv, seq_q, d_head], &device);
    let mut dk = Tensor::<4>::zeros([batch, n_heads_kv, seq_kv, d_head], &device);
    let mut dv = Tensor::<4>::zeros([batch, n_heads_kv, seq_kv, d_head], &device);
    for blk in 0..topk {
        let (scores_b, k_b, v_b, idx) = crate::attention::block_scores(
            &q_s,
            k,
            v,
            block_indices,
            &offsets,
            &q_pos,
            blk,
            block_size,
            seq_kv,
            scale,
            causal,
        );
        let attn_b = scores_b.sub(m.clone()).exp().div(l.clone()); // [B,Hkv,qpkv,S,bs]

        let d_attn_b = d_out_5
            .clone()
            .swap_dims(2, 3)
            .matmul(v_b.swap_dims(3, 4))
            .swap_dims(2, 3); // [B,Hkv,qpkv,S,bs]
        let d_scores_b = attn_b.clone().mul(d_attn_b.sub(wd_sum.clone())); // softmax backward

        dq = dq.add(
            d_scores_b
                .clone()
                .swap_dims(2, 3)
                .matmul(k_b)
                .swap_dims(2, 3)
                .div_scalar(scale),
        );

        let dk_b = d_scores_b
            .swap_dims(2, 3)
            .swap_dims(3, 4)
            .matmul(q_s.clone()) // [B,Hkv,S,bs,d] (contracts qpkv, not S)
            .div_scalar(scale);
        let dv_b = attn_b
            .swap_dims(2, 3)
            .swap_dims(3, 4)
            .matmul(d_out_5.clone().swap_dims(2, 3)); // [B,Hkv,S,bs,d]
                                                      // Flat scatter over (bs, d), mirroring the forward's flat gather:
                                                      // token position p maps to p*d + {0..d-1} inside [B,Hkv,Skv*d].
        let d_arange = Tensor::<1, Int>::arange(0..d_head as i64, &device)
            .reshape::<5, _>([1, 1, 1, 1, d_head]);
        let idx_flat = idx
            .unsqueeze_dim::<5>(4)
            .mul_scalar(d_head as i64)
            .add(d_arange)
            .reshape::<4, _>([batch, n_heads_kv, seq_q, block_size * d_head])
            .reshape::<3, _>([batch, n_heads_kv, seq_q * block_size * d_head]); // [B,Hkv,S*bs*d]
        dk = dk
            .reshape::<3, _>([batch, n_heads_kv, seq_kv * d_head])
            .scatter(
                2,
                idx_flat.clone(),
                dk_b.reshape::<4, _>([batch, n_heads_kv, seq_q, block_size * d_head])
                    .reshape::<3, _>([batch, n_heads_kv, seq_q * block_size * d_head]),
                burn::tensor::IndexingUpdateOp::Add,
            )
            .reshape::<4, _>([batch, n_heads_kv, seq_kv, d_head]);
        dv = dv
            .reshape::<3, _>([batch, n_heads_kv, seq_kv * d_head])
            .scatter(
                2,
                idx_flat,
                dv_b.reshape::<4, _>([batch, n_heads_kv, seq_q, block_size * d_head])
                    .reshape::<3, _>([batch, n_heads_kv, seq_q * block_size * d_head]),
                burn::tensor::IndexingUpdateOp::Add,
            )
            .reshape::<4, _>([batch, n_heads_kv, seq_kv, d_head]);
    }

    (
        dq.reshape::<4, _>([batch, n_heads_q, seq_q, d_head]),
        dk,
        dv,
    )
}

/// Fused sparse attention with exact backward on `Autodiff<Inner>`.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn_autodiff<Inner: Backend>(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    block_indices: Tensor<4, Int>,
    scale: f64,
    block_size: usize,
    n_heads_kv: usize,
    n_heads_q: usize,
    causal: bool,
) -> Option<(Tensor<4>, Tensor<4>)>
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
{
    let qa = q.try_into_primitive::<Autodiff<Inner>>().ok()?;
    let ka = k.try_into_primitive::<Autodiff<Inner>>().ok()?;
    let va = v.try_into_primitive::<Autodiff<Inner>>().ok()?;
    let q_t = Tensor::from_primitive::<Inner>(qa.primitive.clone());
    let k_t = Tensor::from_primitive::<Inner>(ka.primitive.clone());
    let v_t = Tensor::from_primitive::<Inner>(va.primitive.clone());
    let bi_t: Tensor<4, Int> = {
        let prim = block_indices.clone().try_into_primitive::<Inner>().ok()?;
        let prim: <Inner as burn::backend::BackendTypes>::IntTensorPrimitive = prim;
        Tensor::from_primitive::<Inner>(prim)
    };
    // Backward runs on bare Inner, so store bi_t (bare handle), not the
    // caller's block_indices (Autodiff<Inner> Dispatch tensor).
    let bi_state = bi_t.clone();

    let (out_t, ba_t) = {
        #[cfg(feature = "cuda")]
        {
            type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
            if std::any::TypeId::of::<Inner>() == std::any::TypeId::of::<CudaBare>() {
                if let Some((o, ba)) = crate::sparse_kernel::sparse_attn_cuda::<Inner>(
                    q_t.clone(),
                    k_t.clone(),
                    v_t.clone(),
                    bi_t.clone(),
                    scale,
                    block_size,
                    n_heads_kv,
                    n_heads_q,
                    causal,
                ) {
                    (o, ba)
                } else {
                    sparse_attn_batched_gqa(
                        q_t, k_t, v_t, bi_t, scale, block_size, n_heads_kv, n_heads_q, causal,
                    )
                }
            } else {
                sparse_attn_batched_gqa(
                    q_t, k_t, v_t, bi_t, scale, block_size, n_heads_kv, n_heads_q, causal,
                )
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            sparse_attn_batched_gqa(
                q_t, k_t, v_t, bi_t, scale, block_size, n_heads_kv, n_heads_q, causal,
            )
        }
    };

    let out_prim = out_t.try_into_primitive::<Inner>().unwrap();
    let ba_prim = ba_t.try_into_primitive::<Inner>().unwrap();
    let nodes = [qa.node.clone(), ka.node.clone(), va.node.clone()];
    let prep = SparseAttnOp.prepare::<NoCheckpointing>(nodes);
    let state = SparseAttnState {
        block_indices: bi_state,
        scale,
        block_size,
        n_heads_kv,
        n_heads_q,
        causal,
    };
    let out_adt = match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut prep) => {
            let _ids = [
                Some(prep.checkpoint(&qa)),
                Some(prep.checkpoint(&ka)),
                Some(prep.checkpoint(&va)),
            ];
            prep.finish(state, out_prim)
        }
        OpsKind::UnTracked(prep) => prep.finish(out_prim),
    };
    // ba is a secondary (KL) signal: returned untracked
    let ba_adt = <Autodiff<Inner> as burn::backend::AutodiffBackend>::from_inner(ba_prim);
    Some((
        Tensor::from_primitive::<Autodiff<Inner>>(out_adt),
        Tensor::from_primitive::<Autodiff<Inner>>(ba_adt),
    ))
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod ad_tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Int, Tensor};

    type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    fn raw_sparse_attn(
        q: &Tensor<4>,
        k: &Tensor<4>,
        v: &Tensor<4>,
        bi: &Tensor<4, Int>,
        scale: f64,
        bs: usize,
        hkv: usize,
        hq: usize,
    ) -> Tensor<4> {
        sparse_attn_batched_gqa(
            q.clone(),
            k.clone(),
            v.clone(),
            bi.clone(),
            scale,
            bs,
            hkv,
            hq,
            false,
        )
        .0
    }

    #[test]
    fn sparse_attn_fused_backward_matches_burn_autodiff() {
        let dev = Device::default();
        let adev = Device::default().autodiff();
        let (b, hq, hkv, s, d, bs, topk) =
            (1usize, 4usize, 2usize, 16usize, 8usize, 4usize, 2usize);
        let scale = (d as f64).sqrt();
        let q = Tensor::<4>::random([b, hq, s, d], Distribution::Normal(0.0, 1.0), &dev);
        let k = Tensor::<4>::random([b, hkv, s, d], Distribution::Normal(0.0, 1.0), &dev);
        let v = Tensor::<4>::random([b, hkv, s, d], Distribution::Normal(0.0, 1.0), &dev);
        let nblocks = (s / bs) as i64;
        let bi = Tensor::<4, Int>::from_data(
            burn::tensor::TensorData::new(
                (0..(b * hkv * s * topk) as i64)
                    .map(|i| {
                        (i * topk as i64 + i / (s as i64 * topk as i64)) % (nblocks * topk as i64)
                            / topk as i64
                    })
                    .collect(),
                [b, hkv, s, topk],
            ),
            &dev,
        );
        // fused op grads
        let hf = (
            Tensor::<4>::from_data(q.clone().into_data(), &adev).require_grad(),
            Tensor::<4>::from_data(k.clone().into_data(), &adev).require_grad(),
            Tensor::<4>::from_data(v.clone().into_data(), &adev).require_grad(),
        );
        let (outf, _ba) = sparse_attn_autodiff::<CudaBare>(
            hf.0.clone(),
            hf.1.clone(),
            hf.2.clone(),
            bi.clone(),
            scale,
            bs,
            hkv,
            hq,
            false,
        )
        .unwrap();
        let fwd_diff = {
            let of = Tensor::<4>::from_data(outf.clone().into_data(), &dev);
            let or_ = raw_sparse_attn(&q, &k, &v, &bi, scale, bs, hkv, hq);
            let or = Tensor::<4>::from_data(or_.into_data(), &dev);
            let ov = to_host(of);
            let rv = to_host(or);
            maxdiff(&ov, &rv)
        };
        assert!(fwd_diff < 1e-2, "fused forward vs raw {fwd_diff}");
        let grads_f = outf.powf_scalar(2.0).sum().backward();
        // burn's raw autodiff of the tensor path
        let hr = (
            Tensor::<4>::from_data(q.clone().into_data(), &adev).require_grad(),
            Tensor::<4>::from_data(k.clone().into_data(), &adev).require_grad(),
            Tensor::<4>::from_data(v.clone().into_data(), &adev).require_grad(),
        );
        let outr = raw_sparse_attn(&hr.0, &hr.1, &hr.2, &bi, scale, bs, hkv, hq);
        let grads_r = outr.powf_scalar(2.0).sum().backward();
        let aq = to_host(hf.0.grad(&grads_f).unwrap());
        let bq = to_host(hr.0.grad(&grads_r).unwrap());
        let md = maxdiff(&aq, &bq);
        let _ = (&aq, &bq);
        assert!(md < 5e-3, "dq vs burn raw {md}");
        let ak = to_host(hf.1.grad(&grads_f).unwrap());
        let bk = to_host(hr.1.grad(&grads_r).unwrap());
        let md = maxdiff(&ak, &bk);
        assert!(md < 5e-3, "dk vs burn raw {md}");
        let av = to_host(hf.2.grad(&grads_f).unwrap());
        let bv = to_host(hr.2.grad(&grads_r).unwrap());
        let md = maxdiff(&av, &bv);
        assert!(md < 5e-3, "dv vs burn raw {md}");
    }

    #[test]
    fn sparse_attn_fused_backward_matches_tensor() {
        let dev = Device::default().autodiff();
        let (b, hq, hkv, s, d, bs, topk) =
            (2usize, 8usize, 2usize, 64usize, 16usize, 8usize, 4usize);
        let scale = (d as f64).sqrt();
        let q = Tensor::<4>::random([b, hq, s, d], Distribution::Normal(0.0, 1.0), &dev);
        let k = Tensor::<4>::random([b, hkv, s, d], Distribution::Normal(0.0, 1.0), &dev);
        let v = Tensor::<4>::random([b, hkv, s, d], Distribution::Normal(0.0, 1.0), &dev);
        let nblocks = (s / bs) as i64;
        let bi = Tensor::<4, Int>::from_data(
            burn::tensor::TensorData::new(
                (0..(b * hkv * s * topk) as i64)
                    .map(|i| {
                        (i * topk as i64 + i / (s as i64 * topk as i64)) % (nblocks * topk as i64)
                            / topk as i64
                    })
                    .collect(),
                [b, hkv, s, topk],
            ),
            &dev,
        );

        // tensor reference grads (computed once)
        let (qt, kt, vt) = (
            q.clone().require_grad(),
            k.clone().require_grad(),
            v.clone().require_grad(),
        );
        let (outt, _ba) = sparse_attn_batched_gqa(
            qt.clone(),
            kt.clone(),
            vt.clone(),
            bi.clone(),
            scale,
            bs,
            hkv,
            hq,
            false,
        );
        let loss_t = outt.powf_scalar(2.0).sum();
        let grads_t = loss_t.backward();
        let (dqt, dkt, dvt) = (
            qt.grad(&grads_t).unwrap(),
            kt.grad(&grads_t).unwrap(),
            vt.grad(&grads_t).unwrap(),
        );

        // 30 fresh fused forward+backward runs: stresses the dk/dv atomics
        for run in 0..30 {
            let (qf, kf, vf) = (
                q.clone().require_grad(),
                k.clone().require_grad(),
                v.clone().require_grad(),
            );
            let (outf, _ba) = sparse_attn_autodiff::<CudaBare>(
                qf.clone(),
                kf.clone(),
                vf.clone(),
                bi.clone(),
                scale,
                bs,
                hkv,
                hq,
                false,
            )
            .unwrap();
            let loss_f = outf.powf_scalar(2.0).sum();
            let grads_f = loss_f.backward();
            let md = maxdiff(&to_host(qf.grad(&grads_f).unwrap()), &to_host(dqt.clone()));
            assert!(md < 2e-2, "run {run} dq maxdiff {md}");
            let md = maxdiff(&to_host(kf.grad(&grads_f).unwrap()), &to_host(dkt.clone()));
            assert!(md < 2e-2, "run {run} dk maxdiff {md}");
            let md = maxdiff(&to_host(vf.grad(&grads_f).unwrap()), &to_host(dvt.clone()));
            assert!(md < 2e-2, "run {run} dv maxdiff {md}");
        }
    }
}
