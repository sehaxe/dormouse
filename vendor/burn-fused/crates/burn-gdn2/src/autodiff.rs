//! Custom autodiff operation for the chunked WY forward.
//!
//! The whole [`chunk_wy_forward`] is executed
//! as a single tracked node on the inner backend (no per-op autodiff graph
//! overhead), and the backward pass computes *exact* gradients through the
//! WY representation:
//!
//! - the token loop of the WY solve `(I+L)·W = rhs_k`, `(I+L)·U = rhs_v` is
//!   differentiated at the matrix level via the triangular adjoint
//!   `d_rhs = (I+L)^-T · d_*`, `d_akk = tril_strict(-d_rhs·W^T - d_rhs·U^T)`,
//! - all elementwise gates are differentiated symbolically.
//!
//! The backward recomputes the forward intermediates (deterministic on a given
//! backend), so the op state only carries the input checkpoints — EXCEPT on
//! the fused CUDA path, where the state instead carries the buffers the fused
//! forward exported plus the adjoint that consumes them (see [`FusedAdjoint`]),
//! and the tensor adjoint below is then not run at all.
//!
//! ponytail: `new_state` is returned as an untracked leaf — gradients through
//! the *output* state (manual BPTT chaining) are not supported; the *input*
//! state gradient is exact. Add a two-output step if chaining is ever needed.

use burn::backend::{AutodiffBackend, Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, Tensor};
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::{CheckpointStrategy, NoCheckpointing};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_autodiff::{Autodiff, NodeId};

use crate::forward::{chunk_masks, chunk_wy_forward, chunk_wy_forward_impl};

const N_PARENTS: usize = 7;

/// The fused adjoint as a NO-CAPTURE closure, built where `Inner` is known.
///
/// `Backward::backward` is generic over its backend `B` and cannot name `Inner`
/// (burn 0.22 deleted `B::InnerBackend`), so the three steps — strip the
/// caller's tensors to bare, run the kernels on bare, rebuild the gradients on
/// the caller's own strategy — can only be done at the call site. The closure
/// carries no captures on purpose: the state must be `Send` and `Clone`, and
/// `Arc<dyn Fn>` is only `Send` for a `Sync` closure, which the exported
/// buffers are not required to be. They travel in the state instead.
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct FusedAdjoint(
    std::sync::Arc<
        dyn Fn(
                &crate::kernel::chunk_adjoint_cube::cuda::FusedBackwardInputs,
                &[Option<Tensor<4>>; 4],
                Tensor<4>,
                f64,
                usize,
            ) -> Result<[Tensor<4>; 7], &'static str>
            + Send
            + Sync,
    >,
);

#[cfg(feature = "cuda")]
impl core::fmt::Debug for FusedAdjoint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("FusedAdjoint")
    }
}

/// Build the fused adjoint for a caller on `Autodiff<Inner, S>`.
///
/// `fused_chunk_backward` gates on `is_cuda::<B>()`, which is the SAME dead
/// `TypeId` test that kept this adjoint out of every training step: the op's
/// `backward` always passes an autodiff backend, so the gate was always false
/// and the tensor adjoint ran. Here `B` IS the inner backend, so the gate asks
/// the right question, and the inputs are genuinely bare by the time it does.
#[cfg(feature = "cuda")]
fn fused_adjoint<Inner: Backend, S: CheckpointStrategy>() -> FusedAdjoint
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>> + DispatchKindConversion<Inner>,
{
    use crate::cuda_dispatch::{rebuild, strip};
    use crate::kernel::chunk_adjoint_cube::cuda::fused_chunk_backward;
    FusedAdjoint(std::sync::Arc::new(
        |fwd: &crate::kernel::chunk_adjoint_cube::cuda::FusedBackwardInputs,
         xs: &[Option<Tensor<4>>; 4],
         d_out: Tensor<4>,
         scale: f64,
         chunk_size: usize| {
            // k, v, b, w. NOT the input state: the fused adjoint takes its
            // per-chunk states from the forward's export and starts its BPTT
            // chain from a zero state adjoint, which is this op's contract —
            // the output state is an untracked leaf (see the module header).
            let [Some(k), Some(v), Some(b), Some(w)] = xs else {
                return Err("an op parent was not a tracked node");
            };
            // Bare, so `try_into_primitive` inside the kernels sees what it
            // expects: a hardware tensor, not an autodiff handle. Named
            // separately because a refusal here and a refusal inside the
            // kernels are different bugs, and a message that says neither is
            // the message that costs the next person an afternoon.
            let bare = |t: &Tensor<4>, name: &'static str| {
                strip::<Inner, S, 4>(t).ok_or(name)
            };
            let k = bare(k, "strip(k)")?;
            let v = bare(v, "strip(v)")?;
            let b = bare(b, "strip(b)")?;
            let w = bare(w, "strip(w)")?;
            let d_out = bare(&d_out, "strip(d_out)")?;
            let out = fused_chunk_backward::<Inner>(fwd, &k, &v, &b, &w, &d_out, scale, chunk_size)
                .ok_or(
                    "fused_chunk_backward refused (its own is_cuda gate, or a contiguity \
                     materialization that did not come back row-major)",
                )?;
            Ok([
                out.d_q,
                out.d_k,
                out.d_v,
                out.d_g,
                out.d_b,
                out.d_w,
                out.d_s,
            ]
            .map(|t| rebuild::<Inner, S, 4>(&t)))
        },
    ))
}

/// Backward data saved by the fused forward kernels (CUDA only): the buffers
/// the forward exported, and the adjoint to run on them.
#[cfg(feature = "cuda")]
type FusedState = Option<(
    crate::kernel::chunk_adjoint_cube::cuda::FusedBackwardInputs,
    FusedAdjoint,
)>;
#[cfg(not(feature = "cuda"))]
type FusedState = ();

#[derive(Debug)]
struct ChunkWy;

impl<B: Backend> Backward<B, N_PARENTS> for ChunkWy
where
    DispatchTensor: DispatchKindConversion<B>,
{
    type State = (
        [Option<NodeId>; N_PARENTS],
        f64,
        usize,
        Option<crate::forward::ChunkWyScratch>,
        crate::autodiff::FusedState,
    );

    fn backward(
        self,
        ops: Ops<Self::State, N_PARENTS>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
        let (ids, scale, chunk_size, scratch, fused_data) = ops.state;
        // q and g are not checkpointed: the scratch (always saved) carries
        // everything their values would contribute (qE, E), so the adjoint
        // never touches q or g. Keeping them out saves 2 full-sequence
        // tensors per training step.
        let k = ids[1]
            .map(|id| Tensor::from_primitive::<B>(checkpointer.retrieve_node_output(id)))
            .expect("k is checkpointed");
        let v = ids[2]
            .map(|id| Tensor::from_primitive::<B>(checkpointer.retrieve_node_output(id)))
            .expect("v is checkpointed");
        let b = ids[4]
            .map(|id| Tensor::from_primitive::<B>(checkpointer.retrieve_node_output(id)))
            .expect("b is checkpointed");
        let w = ids[5]
            .map(|id| Tensor::from_primitive::<B>(checkpointer.retrieve_node_output(id)))
            .expect("w is checkpointed");
        let s_in = ids[6]
            .map(|id| Tensor::from_primitive::<B>(checkpointer.retrieve_node_output(id)))
            .expect("state is checkpointed");

        let d_out = Tensor::from_primitive::<B>(grads.consume::<B>(&ops.node));

        #[cfg(feature = "cuda")]
        if let Some((fwd, adjoint)) = fused_data {
            // LOUD, not a fallback: the fused forward ran, so it did NOT keep
            // the chunk scratch, and the tensor adjoint below panics without
            // it. Dropping the gradients instead would train on a lie, so a
            // refusal here names the cause instead of hiding it.
            let d_inputs = (adjoint.0)(
                &fwd,
                &[
                    Some(k.clone()),
                    Some(v.clone()),
                    Some(b.clone()),
                    Some(w.clone()),
                ],
                d_out.clone(),
                scale,
                chunk_size,
            )
            .unwrap_or_else(|why| {
                panic!(
                    "the fused chunk forward ran, so its adjoint must, and it refused at \
                     {why}. The tensor adjoint cannot take over because the fused forward \
                     kept no chunk scratch, so there is no fallback that is not a lie."
                )
            });
            for (i, grad) in d_inputs.into_iter().enumerate() {
                if let Some(node) = ops.parents[i].clone() {
                    grads.register::<B>(node.id, grad.try_into_primitive::<B>().unwrap());
                }
            }
            return;
        }

        let [batch, heads, time, k_dim] = k.shape().dims::<4>();
        let v_dim = v.shape().dims::<4>()[3];
        let device = k.device();
        let scratch = scratch.expect("tensor-path backward needs the chunk scratch");
        let n_chunks = scratch.chunks.len();

        // --- forward pass over the chunks: derive the values the minimal
        // scratch does not store (kG, rhs, W, U, v_new) and replay the state
        // trajectory. Chunk i's adjoint uses the state *before* chunk i, and
        // the state output of chunk i feeds chunk i+1 (BPTT), so the state
        // adjoint accumulates backwards over chunks.
        // The scratch chunks are padded to a 16-token tile multiple (K3
        // scheme); the checkpoint slices must be padded identically.
        let pad_to = |t: Tensor<4>, c_pad: usize, d: usize| -> Tensor<4> {
            let cc = t.shape().dims::<4>()[2];
            if cc == c_pad {
                t
            } else {
                Tensor::cat(
                    vec![t, Tensor::zeros([batch, heads, c_pad - cc, d], &device)],
                    2,
                )
            }
        };
        let mut s_traj: Vec<Tensor<4>> = Vec::with_capacity(n_chunks + 1);
        s_traj.push(s_in.clone());
        for (ci, sc) in scratch.chunks.iter().enumerate() {
            let c_pad = sc.g_exp.shape().dims::<4>()[2];
            let start = ci * chunk_size;
            let c_real = (start + chunk_size).min(time) - start;
            let g_exp = sc.g_exp.clone();
            let g_last = g_exp
                .clone()
                .slice([0..batch, 0..heads, c_real - 1..c_real]);
            let k_c = pad_to(
                k.clone().slice([0..batch, 0..heads, start..start + c_real]),
                c_pad,
                k_dim,
            );
            let v_c = pad_to(
                v.clone().slice([0..batch, 0..heads, start..start + c_real]),
                c_pad,
                v_dim,
            );
            let b_c = pad_to(
                b.clone().slice([0..batch, 0..heads, start..start + c_real]),
                c_pad,
                k_dim,
            );
            let w_c = pad_to(
                w.clone().slice([0..batch, 0..heads, start..start + c_real]),
                c_pad,
                v_dim,
            );
            let rhs_k = b_c.clone() * k_c.clone() * g_exp.clone();
            let rhs_v = w_c.clone() * v_c.clone();
            let w_wy = sc.m_inv.clone().matmul(rhs_k);
            let u_wy = sc.m_inv.clone().matmul(rhs_v);
            let s_before = s_traj.last().unwrap().clone();
            let v_new = u_wy - w_wy.matmul(s_before.clone());
            let decay = g_last.clone() / g_exp;
            s_traj.push(
                s_before * g_last.swap_dims(2, 3) + (k_c * decay).swap_dims(2, 3).matmul(v_new),
            );
        }

        let mut d_s = Tensor::zeros_like(&s_in);
        let mut d_state_acc = Tensor::zeros_like(&s_in);
        let (causal_full, strict_full) = chunk_masks(chunk_size, &device);
        let scale_causal_full = causal_full.clone() * scale;
        // Per-chunk gradient parts, concatenated once at the end: incremental
        // slice_assign accumulation would copy the full-length buffer on every
        // chunk (refcount bump from the read clone), i.e. ~12 GB of copies per
        // training step at d=2048, T=4096.
        let mut d_q_parts: Vec<Tensor<4>> = Vec::with_capacity(n_chunks);
        let mut d_k_parts: Vec<Tensor<4>> = Vec::with_capacity(n_chunks);
        let mut d_v_parts: Vec<Tensor<4>> = Vec::with_capacity(n_chunks);
        let mut d_g_parts: Vec<Tensor<4>> = Vec::with_capacity(n_chunks);
        let mut d_b_parts: Vec<Tensor<4>> = Vec::with_capacity(n_chunks);
        let mut d_w_parts: Vec<Tensor<4>> = Vec::with_capacity(n_chunks);

        for (ri, sc) in scratch.chunks.iter().rev().enumerate() {
            let ci = n_chunks - 1 - ri;
            let c_pad = sc.g_exp.shape().dims::<4>()[2];
            let start = ci * chunk_size;
            let c_real = (start + chunk_size).min(time) - start;
            let range = |l: usize, d: usize| [0..batch, 0..heads, l..l + d];

            let (scale_causal, strict) = if c_pad == chunk_size {
                (scale_causal_full.clone(), strict_full.clone())
            } else {
                let (cau, str) = chunk_masks(c_pad, &device);
                (cau * scale, str)
            };
            let g_exp = sc.g_exp.clone();
            let q_gated = sc.q_gated.clone();
            let k_c = pad_to(k.clone().slice(range(start, c_real)), c_pad, k_dim);
            let v_c = pad_to(v.clone().slice(range(start, c_real)), c_pad, v_dim);
            let b_c = pad_to(b.clone().slice(range(start, c_real)), c_pad, k_dim);
            let w_c = pad_to(w.clone().slice(range(start, c_real)), c_pad, v_dim);
            let d_out_c = pad_to(d_out.clone().slice(range(start, c_real)), c_pad, v_dim);
            let s_before = s_traj[ci].clone();

            // Re-derive the per-chunk values (identical ops to the forward
            // pass, so bit-identical results).
            let k_g = k_c.clone() / g_exp.clone();
            let rhs_k = b_c.clone() * k_c.clone() * g_exp.clone();
            let rhs_v = w_c.clone() * v_c.clone();
            let w_wy = sc.m_inv.clone().matmul(rhs_k.clone());
            let u_wy = sc.m_inv.clone().matmul(rhs_v.clone());
            let v_new = u_wy.clone() - w_wy.clone().matmul(s_before.clone());

            // out = aqk·v_new + (q·E)·S_in·scale
            let d_aqk = d_out_c.clone().matmul(v_new.clone().swap_dims(2, 3));
            let d_qk = d_aqk * scale_causal;
            let d_v_new = sc.aqk.clone().swap_dims(2, 3).matmul(d_out_c.clone());

            // BPTT through the state output S_out = S·E_last^T + (k·decay)^T·v_new:
            // d_K̂ = v_new·d_state_acc^T, d_v_new += K̂·d_state_acc.
            let g_last = g_exp
                .clone()
                .slice([0..batch, 0..heads, c_real - 1..c_real]);
            let decay = g_last.clone() / g_exp.clone();
            let d_k_hat = v_new.clone().matmul(d_state_acc.clone().swap_dims(2, 3));
            let d_k_bptt = d_k_hat.clone() * decay.clone();
            let d_decay = d_k_hat * k_c.clone();
            let khat = decay.clone() * k_c.clone();
            let d_v_new = d_v_new + khat.matmul(d_state_acc.clone());

            // v_new = U - W·S_in  →  d_S_in = -W^T·d_v_new
            let mut d_s_c = -(w_wy.clone().swap_dims(2, 3).matmul(d_v_new.clone()));
            // inter = (q·E)·S_in·scale
            d_s_c = d_s_c
                + q_gated
                    .clone()
                    .swap_dims(2, 3)
                    .matmul(d_out_c.clone())
                    .mul_scalar(scale);
            // d_S_in += E_last ⊙ d_state_acc  (the state-input part of BPTT)
            d_s_c = d_s_c + g_last.clone().swap_dims(2, 3) * d_state_acc.clone();

            // decay = E_last / E: d_E_last = Σ_t d_decay/E + Σ_v d_state_acc·S,
            // d_E[t] = -d_decay·decay/E, and E_last = E[c-1].
            let d_e_last = (d_decay.clone() / g_exp.clone()).sum_dim(2)
                + d_state_acc
                    .clone()
                    .mul(s_before.clone())
                    .sum_dim(3)
                    .swap_dims(2, 3);
            let d_e_decay = -(d_decay * decay / g_exp.clone());

            // q·E: from inter (scale·d_out·S_in^T) and aqk (d_qk·kG)
            let d_qe = d_out_c
                .clone()
                .matmul(s_before.clone().swap_dims(2, 3))
                .mul_scalar(scale)
                + d_qk.clone().matmul(k_g.clone());
            let d_k_g = d_qk.swap_dims(2, 3).matmul(q_gated.clone());

            // WY solves: d_W = -d_v_new·S_in^T, d_U = d_v_new,
            // d_rhs = M⁻ᵀ·d_* (M⁻¹ from the scratch), d_akk =
            // strict ⊙ (-d_rhs_k W^T - d_rhs_v U^T)
            let d_w_wy = -(d_v_new.clone().matmul(s_before.clone().swap_dims(2, 3)));
            let d_rhs_k = sc.m_inv.clone().swap_dims(2, 3).matmul(d_w_wy);
            let d_rhs_v = sc.m_inv.clone().swap_dims(2, 3).matmul(d_v_new);
            let d_akk = (-d_rhs_k.clone().matmul(w_wy.clone().swap_dims(2, 3))
                - d_rhs_v.clone().matmul(u_wy.clone().swap_dims(2, 3)))
                * strict;

            let d_k_g = d_k_g + d_akk.clone().swap_dims(2, 3).matmul(rhs_k.clone());
            let d_bk_e = d_akk.matmul(k_g.clone());

            // rhs_k = b·k·E  →  bk = rhs_k / E, d_bk = (d_rhs_k + d_bkE) ⊙ E
            let bk = rhs_k.clone() / g_exp.clone();
            let d_bk = (d_rhs_k.clone() + d_bk_e.clone()) * g_exp.clone();
            let d_b_c = d_bk.clone() * k_c.clone();
            let d_k_bk = d_bk * b_c.clone();
            let d_e_rhsk = d_rhs_k * bk.clone();

            // rhs_v = w·v
            let d_w_c = d_rhs_v.clone() * v_c.clone();
            let d_v_c = d_rhs_v * w_c;

            // kG = k / E
            let d_k_kg = d_k_g.clone() / g_exp.clone();
            let d_e_kg = -(d_k_g * k_c.clone() / g_exp.clone().powf_scalar(2.0));

            // q·E, bk·E = b·k·E (d_E from the qE path uses qE/E instead of q)
            let d_q_c = d_qe.clone() * g_exp.clone();
            let d_e_qe = d_qe * q_gated / g_exp.clone();
            let d_e_bke = d_bk_e * bk;

            // E = exp(G), G = cumsum(g): d_G = d_E·E, d_g = reverse cumsum
            let mut d_e = d_e_rhsk + d_e_kg + d_e_qe + d_e_bke + d_e_decay;
            // E_last = E[c-1] receives the accumulated row adjoint
            let cur_last = d_e.clone().slice([0..batch, 0..heads, c_real - 1..c_real]);
            d_e = d_e.slice_assign(
                [0..batch, 0..heads, c_real - 1..c_real, 0..k_dim],
                cur_last + d_e_last,
            );
            let d_g_c = (d_e * g_exp.clone()).flip([2]).cumsum(2).flip([2]);

            // the padded rows carry zero gradient; slice back to the real
            // chunk length before concatenating
            d_q_parts.push(d_q_c.slice([0..batch, 0..heads, 0..c_real]));
            d_k_parts.push((d_k_bk + d_k_kg + d_k_bptt).slice([0..batch, 0..heads, 0..c_real]));
            d_v_parts.push(d_v_c.slice([0..batch, 0..heads, 0..c_real]));
            d_g_parts.push(d_g_c.slice([0..batch, 0..heads, 0..c_real]));
            d_b_parts.push(d_b_c.slice([0..batch, 0..heads, 0..c_real]));
            d_w_parts.push(d_w_c.slice([0..batch, 0..heads, 0..c_real]));

            // chunk 0 (the last processed, ci == 0) is the *input* state; its
            // state-input adjoint is the gradient of `s_in`. The adjoints of
            // later chunks are intermediate-state gradients that only chain
            // backwards through `d_state_acc`.
            d_state_acc = d_s_c;
            if ci == 0 {
                d_s = d_state_acc.clone();
            }
        }

        // The chunk loop ran in reverse; concatenate back into forward order.
        let d_q = Tensor::cat(d_q_parts.into_iter().rev().collect(), 2);
        let d_k = Tensor::cat(d_k_parts.into_iter().rev().collect(), 2);
        let d_v = Tensor::cat(d_v_parts.into_iter().rev().collect(), 2);
        let d_g = Tensor::cat(d_g_parts.into_iter().rev().collect(), 2);
        let d_b = Tensor::cat(d_b_parts.into_iter().rev().collect(), 2);
        let d_w = Tensor::cat(d_w_parts.into_iter().rev().collect(), 2);

        let d_inputs = [d_q, d_k, d_v, d_g, d_b, d_w, d_s];
        for (i, grad) in d_inputs.into_iter().enumerate() {
            if let Some(node) = ops.parents[i].clone() {
                grads.register::<B>(node.id, grad.try_into_primitive::<B>().unwrap());
            }
        }
    }
}

/// Fused chunked WY forward with exact backward, for autodiff backends.
///
/// Takes tensors dispatched on `Autodiff<Inner>` (default checkpoint
/// strategy). Returns `None` when the dispatch does not match, in which case
/// the caller falls back to the plain tensor-ops path. For an explicit
/// strategy see [`chunk_wy_forward_autodiff_s`].
#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward_autodiff<Inner: Backend>(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w: Tensor<4>,
    state: Tensor<4>,
    scale: f64,
    chunk_size: usize,
) -> Option<(Tensor<4>, Tensor<4>)>
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
{
    chunk_wy_forward_autodiff_s::<Inner, NoCheckpointing>(q, k, v, g, b, w, state, scale, chunk_size)
}

/// [`chunk_wy_forward_autodiff`] for an EXPLICIT checkpointing strategy.
///
/// The strategy is a parameter, not a hardcoded `NoCheckpointing`, because the
/// primitive type (`AutodiffTensor<Inner>`) does not carry it — only burn's
/// runtime dispatch context does — so an op that names the default strategy
/// cannot accept the tensors of a `BalancedCheckpointing` graph at all (every
/// conversion returns `None`). dormouse trains on Balanced, so the strategy
/// is exactly what made this op unreachable in production.
#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward_autodiff_s<Inner: Backend, S: CheckpointStrategy>(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w: Tensor<4>,
    state: Tensor<4>,
    scale: f64,
    chunk_size: usize,
) -> Option<(Tensor<4>, Tensor<4>)>
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>> + DispatchKindConversion<Inner>,
{
    use crate::cuda_dispatch::{autodiff_node, bare_from_node};
    let [q, k, v, g, b, w, state] =
        [q, k, v, g, b, w, state].map(|t| autodiff_node::<Inner, S, 4>(&t));
    let (q, k, v, g, b, w, state) = match (q, k, v, g, b, w, state) {
        (Some(q), Some(k), Some(v), Some(g), Some(b), Some(w), Some(state)) => {
            (q, k, v, g, b, w, state)
        }
        _ => return None,
    };
    // The kernels see the BARE backend, where `is_cuda::<Inner>()` is correct.
    let (q_t, k_t, v_t, g_t, b_t, w_t, s_t) = (
        bare_from_node::<Inner, S, 4>(&q),
        bare_from_node::<Inner, S, 4>(&k),
        bare_from_node::<Inner, S, 4>(&v),
        bare_from_node::<Inner, S, 4>(&g),
        bare_from_node::<Inner, S, 4>(&b),
        bare_from_node::<Inner, S, 4>(&w),
        bare_from_node::<Inner, S, 4>(&state),
    );

    // Forward on the inner backend. On the bare CUDA backend the two fused
    // chunk kernels run instead of the tensor path (2 launches per chunk
    // instead of ~150; verified in tests/fused_chunk_verify.rs). The tensor
    // path returns its scratch so the backward can skip the recompute; the
    // fused path does not expose intermediates, so its backward recomputes.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    let (out_t, new_state_t, scratch, fused_data) = {
        #[cfg(feature = "cuda")]
        {
            use crate::kernel::chunk_cube::cuda::{fused_chunk_forward_scratch, is_cuda};
            if is_cuda::<Inner>() {
                if let Some((o, ns, io)) = fused_chunk_forward_scratch::<Inner>(
                    q_t.clone(),
                    k_t.clone(),
                    v_t.clone(),
                    g_t.clone(),
                    b_t.clone(),
                    w_t.clone(),
                    s_t.clone(),
                    scale,
                    chunk_size,
                ) {
                    // Keep the flat exports for the fused backward and
                    // rebuild the minimal scratch from them (the backward
                    // never re-runs the forward). The kernel stores aqk
                    // transposed ([s][r] = score(q_r,k_s)) and qgt as [k][c];
                    // E is recovered from kgd = k·glast/E.
                    use crate::kernel::chunk_adjoint_cube::cuda::FusedBackwardInputs;
                    let fused_inputs = FusedBackwardInputs {
                        m_inv: io.m_inv.clone(),
                        aqk: io.aqk.clone(),
                        qgt: io.qgt.clone(),
                        glast: io.glast.clone(),
                        v_new: io.v_new.clone(),
                        states: io.states.clone(),
                        w: io.w.clone(),
                        u: io.u.clone(),
                        gexp: io.gexp.clone(),
                    };
                    (
                        o,
                        ns,
                        None, // fused backward uses the exported buffers, not the scratch
                        Some((fused_inputs, fused_adjoint::<Inner, S>())),
                    )
                } else {
                    let (o, ns, sc) = chunk_wy_forward_impl(
                        q_t, k_t, v_t, g_t, b_t, w_t, s_t, scale, chunk_size, None,
                    );
                    (o, ns, Some(sc), None)
                }
            } else {
                let (o, ns, sc) = chunk_wy_forward_impl(
                    q_t, k_t, v_t, g_t, b_t, w_t, s_t, scale, chunk_size, None,
                );
                (o, ns, Some(sc), None)
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let (o, ns, sc) =
                chunk_wy_forward_impl(q_t, k_t, v_t, g_t, b_t, w_t, s_t, scale, chunk_size, None);
            (o, ns, Some(sc), ())
        }
    };
    let out_prim = out_t.try_into_primitive::<Inner>().unwrap();
    let new_state_prim = new_state_t.try_into_primitive::<Inner>().unwrap();

    let nodes = [
        q.node(),
        k.node(),
        v.node(),
        g.node(),
        b.node(),
        w.node(),
        state.node(),
    ];
    let prep = ChunkWy.prepare::<S>(nodes);

    let (out_adt, new_state_adt) = match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut prep) => {
            let ids = [
                None, // q: not checkpointed (the scratch carries qE/E)
                Some(prep.checkpoint(&k)),
                Some(prep.checkpoint(&v)),
                None, // g: not checkpointed (only E is needed)
                Some(prep.checkpoint(&b)),
                Some(prep.checkpoint(&w)),
                Some(prep.checkpoint(&state)),
            ];
            let out = prep.finish((ids, scale, chunk_size, scratch, fused_data), out_prim);
            (
                out,
                <Autodiff<Inner, S> as AutodiffBackend>::from_inner(new_state_prim),
            )
        }
        OpsKind::UnTracked(prep) => {
            let out = prep.finish(out_prim);
            (
                out,
                <Autodiff<Inner, S> as AutodiffBackend>::from_inner(new_state_prim),
            )
        }
    };

    Some((
        Tensor::from_primitive::<Autodiff<Inner, S>>(out_adt),
        Tensor::from_primitive::<Autodiff<Inner, S>>(new_state_adt),
    ))
}

/// [`chunk_wy_forward_autodiff`] with a plain-tensor fallback, coercible to a
/// chunk function pointer for the module training path.
#[allow(clippy::too_many_arguments)]
pub fn chunk_autodiff_or_plain<Inner: Backend>(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w: Tensor<4>,
    state: Tensor<4>,
    scale: f64,
    chunk_size: usize,
) -> (Tensor<4>, Tensor<4>)
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
{
    chunk_wy_forward_autodiff::<Inner>(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        b.clone(),
        w.clone(),
        state.clone(),
        scale,
        chunk_size,
    )
    .unwrap_or_else(|| chunk_wy_forward(q, k, v, g, b, w, state, scale, chunk_size))
}
