//! The seam every fused CUDA kernel in this library has to cross to be
//! callable from a tensor that came out of an autodiff graph.
//!
//! Three separate things go wrong without it, and all three fail SILENTLY —
//! the tensor-ops fallback computes the same function, so a right answer and
//! a fallback look identical:
//!
//! 1. **The backend gate.** `TypeId::of::<B>() == TypeId::of::<CudaBare>()`
//!    is false for every autodiff backend whose checkpointing strategy is not
//!    the default one — dormouse trains on
//!    `Autodiff<CudaBare, BalancedCheckpointing>` — so the gate never opened
//!    in production. [`backend_matches`] is the sanctioned replacement:
//!    `Backend::name` is forwarded by every wrapper (`autodiff<...>`,
//!    `dispatch<...>`, `fusion<...>`, `cubecl<...>`) down to the cubecl
//!    runtime's own `"cuda"`, so it sees through any wrapper and any
//!    strategy, with no `TypeId` and no `AutodiffBackend` bound (burn pre.4
//!    has no generic autodiff route — that deletion is where the `TypeId`
//!    gates came from).
//! 2. **The context gate.** burn's dispatch layer REFUSES to hand a bare
//!    primitive to a tensor whose autodiff context is enabled
//!    (`"Expected concrete Cuda backend with disabled autodiff context"`), so
//!    every `try_into_primitive::<CudaBare>()` on a training tensor returned
//!    `Err`. [`strip`] is the legal way through it: lift to the autodiff
//!    PRIMITIVE, take `.primitive()` (whose context is disabled), and rebuild
//!    a bare tensor.
//! 3. **The strategy.** `Autodiff<Inner, S>`'s float primitive is
//!    `AutodiffTensor<Inner>` — `S` does not appear in the type — so one
//!    primitive type serves every strategy, and the node must be rebuilt with
//!    the strategy the CALLER is on, not a hardcoded `NoCheckpointing`.
//!
//! [`chunk_dispatch`] is all three, wired for the chunked WY op, and it
//! REPORTS which arm ran: a `None` cannot tell "computed" from "fell back".

use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, Tensor};
use burn_autodiff::checkpoint::strategy::{BalancedCheckpointing, CheckpointStrategy, NoCheckpointing};
use burn::backend::{AutodiffBackend, BackendTypes};
use burn_autodiff::Autodiff;

/// Which arm ran. A bare `Option`/`bool` cannot distinguish "computed" from
/// "silently fell back", which is the failure mode this module exists to
/// remove.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// [`backend_matches`] says this is not a CUDA backend (NdArray, wgpu, …).
    NotCuda,
    /// The kernel's own limits: dtype, chunk size, stride/contiguity, dims.
    KernelLimits,
    /// An autodiff graph over CUDA, but the node could not be rebuilt on the
    /// caller's backend (a checkpointing strategy this build does not know).
    UnknownStrategy,
}

/// The result of asking for the fused path: either it ran, or here is why not.
#[derive(Debug)]
pub enum Fused<T> {
    Fused(T),
    Fallback(Fallback),
}

impl<T> Fused<T> {
    /// The value, if the fused path ran.
    pub fn into_option(self) -> Option<T> {
        match self {
            Fused::Fused(v) => Some(v),
            Fused::Fallback(_) => None,
        }
    }

    /// `true` when the fused kernels ran.
    pub fn is_fused(&self) -> bool {
        matches!(self, Fused::Fused(_))
    }

    /// Why the fused path did not run, if it did not.
    pub fn reason(&self) -> Option<Fallback> {
        match self {
            Fused::Fused(_) => None,
            Fused::Fallback(r) => Some(*r),
        }
    }
}

/// `true` when `B`'s tensors live on a CUDA device — the sanctioned test,
/// forwarded by every burn wrapper down to the cubecl runtime's own `"cuda"`.
/// No `TypeId`, no `AutodiffBackend` bound, no enumerated strategies: a future
/// wrapper cannot fall off the fused path without this saying so.
pub fn backend_matches<B: Backend>() -> bool {
    B::name(&B::Device::default()).contains("cuda")
}

/// The autodiff primitive of `Autodiff<Inner, S>` — `burn_autodiff` keeps its
/// `AutodiffTensor` type crate-private, so the only public spelling is the
/// associated type.
pub type AdNode<Inner, S> = <Autodiff<Inner, S> as BackendTypes>::FloatTensorPrimitive;

/// The autodiff PRIMITIVE, whatever the strategy: `None` when the tensor is
/// not on `Autodiff<Inner, S>`. burn's dispatch layer refuses to hand a bare
/// primitive to a tensor whose autodiff context is enabled, so this — not
/// `try_into_primitive::<Inner>` — is the only legal way in, and it is the
/// half of the bug no `TypeId` could have caught.
pub fn autodiff_node<Inner: Backend, S: CheckpointStrategy, const D: usize>(
    t: &Tensor<D>,
) -> Option<AdNode<Inner, S>>
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>>,
{
    t.clone().try_into_primitive::<Autodiff<Inner, S>>().ok()
}

/// The bare tensor a kernel can take, from the autodiff primitive: the same
/// bytes with a DISABLED autodiff context.
pub fn bare_from_node<Inner: Backend, S: CheckpointStrategy, const D: usize>(
    node: &AdNode<Inner, S>,
) -> Tensor<D>
where
    DispatchTensor: DispatchKindConversion<Inner>,
{
    Tensor::from_primitive::<Inner>(node.primitive().clone())
}

/// The autodiff layer off: a tensor on `Autodiff<Inner, S>` becomes a bare
/// `Tensor<Inner>`, whatever the strategy. `None` when it is not on one.
pub fn strip<Inner: Backend, S: CheckpointStrategy, const D: usize>(
    t: &Tensor<D>,
) -> Option<Tensor<D>>
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>> + DispatchKindConversion<Inner>,
{
    Some(bare_from_node::<Inner, S, D>(&autodiff_node::<Inner, S, D>(t)?))
}

/// The bare tensor back into an autodiff leaf on `Autodiff<Inner, S>` — the
/// strategy the caller asked for, not a hardcoded `NoCheckpointing`. An op
/// that needs a real backward builds its own node instead; see
/// [`crate::autodiff::chunk_wy_forward_autodiff_s`].
pub fn rebuild<Inner: Backend, S: CheckpointStrategy, const D: usize>(t: &Tensor<D>) -> Tensor<D>
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>> + DispatchKindConversion<Inner>,
{
    let inner = t
        .clone()
        .try_into_primitive::<Inner>()
        .expect("strip produced a bare tensor");
    Tensor::from_primitive::<Autodiff<Inner, S>>(
        <Autodiff<Inner, S> as AutodiffBackend>::from_inner(inner),
    )
}

#[cfg(feature = "cuda")]
mod fused {
    use super::*;
    use crate::kernel::chunk_cube::cuda::CudaBare;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    static FUSED_FWD: AtomicU64 = AtomicU64::new(0);
    static FUSED_BWD: AtomicU64 = AtomicU64::new(0);
    static FUSED_DECLINED: AtomicU64 = AtomicU64::new(0);

    /// Called by the fused kernels when they actually launch (never when a
    /// caller merely *considered* the fused path and fell back).
    #[inline]
    pub fn note_fused_forward() {
        FUSED_FWD.fetch_add(1, Relaxed);
    }

    /// Called by the fused adjoint kernel when it actually launches.
    #[inline]
    pub fn note_fused_backward() {
        FUSED_BWD.fetch_add(1, Relaxed);
    }

    /// Called when the op's inputs carry no node id, so `prepare` returns
    /// `OpsKind::UnTracked` and the op's output is a LEAF: nothing downstream
    /// can send a gradient back through it. That is not a missing speedup,
    /// it is a silent gradient loss - this arm's parameters would never
    /// train while every other op in the model trains normally. Traced so the
    /// distinction is a measurement rather than a reading of the code.
    /// Called when the fused path is DECLINED - either because the op's
    /// inputs carry no node id (so a fused output would be a leaf and the arm
    /// would train nothing) or because `DM_FUSED_KDA=0`. COUNTED, not silent:
    /// a fused arm that quietly stops running is how 30 forwards and 0
    /// gradients read as healthy for a day.
    #[inline]
    pub fn note_fused_declined() {
        FUSED_DECLINED.fetch_add(1, Relaxed);
    }

    /// Fused forward declines since [`reset_fused_calls`].
    pub fn fused_declined() -> u64 {
        FUSED_DECLINED.load(Relaxed)
    }

    /// Forces the tensor path even where the fused path could deliver
    /// gradients. The kill switch ADR-0011 asks for, so an A/B arm can be run
    /// from the same binary instead of a rebuild.
    pub fn fused_forced_off() -> bool {
        std::env::var_os("DM_FUSED_KDA").as_deref() == Some(std::ffi::OsStr::new("0"))
    }

    #[inline]
    pub fn note_untracked() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            if std::env::var_os("DM_GDN2_BWD_TRACE").is_some() {
                eprintln!(
                    "[gdn2] ops kind = UnTracked -> the fused output is a LEAF, \
                     no gradient can reach this op's inputs"
                );
            }
        });
    }

    /// Called on ENTRY to the fused op's autodiff node, before any branch.
    ///
    /// `fused_calls().1 == 0` cannot tell two very different worlds apart:
    /// the node never ran (so this path contributed no gradient at all) vs
    /// the node ran and took the tensor branch (correct gradients, no fused
    /// acceleration). The first is a silent gradient loss; the second is
    /// only a missing speedup. Different bugs, so they need different
    /// evidence. Trace with `DM_GDN2_BWD_TRACE=1`.
    #[inline]
    pub fn note_backward_node() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            if std::env::var_os("DM_GDN2_BWD_TRACE").is_some() {
                eprintln!("[gdn2] ChunkWy::backward ENTERED");
            }
        });
    }

    /// Called when the node runs but does NOT use the fused adjoint, i.e. it
    /// replays the chunk trajectory on tensor ops. Gradients are then
    /// correct and the fused forward's saved state went unused.
    #[inline]
    pub fn note_tensor_branch() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            if std::env::var_os("DM_GDN2_BWD_TRACE").is_some() {
                eprintln!("[gdn2] ChunkWy::backward took the TENSOR branch");
            }
        });
    }

    /// `(forward, backward)` fused kernel launches since [`reset_fused_calls`].
    /// The assertion that a fused path was TAKEN — the one thing numbers
    /// cannot tell you.
    pub fn fused_calls() -> (u64, u64) {
        (FUSED_FWD.load(Relaxed), FUSED_BWD.load(Relaxed))
    }

    /// Zero the launch counters.
    pub fn reset_fused_calls() {
        FUSED_FWD.store(0, Relaxed);
        FUSED_BWD.store(0, Relaxed);
    }

    /// The dispatch layer can hand out bare `CudaBare` tensors and take
    /// autodiff tensors back, for every checkpointing strategy. Implemented
    /// for `DispatchTensor` itself so a caller that only knows
    /// `DispatchTensor: DispatchKindConversion<B>` — the bound every burn op
    /// carries — can reach the fused path without naming a strategy.
    pub trait FusedCudaAutodiff:
        DispatchKindConversion<CudaBare>
        + DispatchKindConversion<Autodiff<CudaBare, NoCheckpointing>>
        + DispatchKindConversion<Autodiff<CudaBare, BalancedCheckpointing>>
    {
    }

    impl FusedCudaAutodiff for DispatchTensor {}

    /// The fused chunked WY chunk op from ANY autodiff backend over the bare
    /// CUDA backend, whatever its checkpointing strategy — and it says which
    /// arm ran.
    ///
    /// The whole recurrence becomes ONE autodiff node over the fused kernels;
    /// the kernels themselves see the BARE backend, where
    /// `is_cuda::<Inner>()` is correct and needs no strategy-agnostic
    /// gymnastics.
    #[allow(clippy::too_many_arguments)]
    pub fn chunk_dispatch<B: Backend>(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        g: Tensor<4>,
        b: Tensor<4>,
        w: Tensor<4>,
        state: Tensor<4>,
        scale: f64,
        chunk_size: usize,
    ) -> Fused<(Tensor<4>, Tensor<4>)>
    where
        DispatchTensor: FusedCudaAutodiff + DispatchKindConversion<B>,
    {
        if !backend_matches::<B>() {
            return Fused::Fallback(Fallback::NotCuda);
        }
        macro_rules! probe {
            ($S:ty) => {
                if let Some((o, s)) = crate::autodiff::chunk_wy_forward_autodiff_s::<CudaBare, $S>(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    g.clone(),
                    b.clone(),
                    w.clone(),
                    state.clone(),
                    scale,
                    chunk_size,
                ) {
                    // The op only ever builds nodes on `Autodiff<CudaBare, S>`,
                    // so the result comes back as `B` exactly when that IS `B`.
                    if let (Ok(o), Ok(s)) = (o.try_into_primitive::<B>(), s.try_into_primitive::<B>()) {
                        return Fused::Fused((
                            Tensor::from_primitive::<B>(o),
                            Tensor::from_primitive::<B>(s),
                        ));
                    }
                }
            };
        }
        probe!(NoCheckpointing);
        probe!(BalancedCheckpointing);

        // THE OPS PATH - and the reason this function returns a value at all
        // on the trainer's backend.
        //
        // `chunk_wy_forward_autodiff_s` runs the forward on the BARE backend
        // and wraps the result in ONE hand-rolled autodiff node whose backward
        // replays the chunk trajectory itself. So gradients exist only if
        // that node is TRACKED - and its parents' node refs decide that, and
        // under `BalancedCheckpointing` the op's inputs are checkpoint
        // leaves, so `prepare` returns `UnTracked` and the output is a LEAF.
        // Measured 2026-09-28 on a real run: `fused kda=30/0`, 30 fused
        // forwards, 0 adjoint launches, 0 gradient into the attention arm.
        // Every KDA number this project had ever produced came from an arm
        // frozen at initialisation.
        //
        // `chunk_wy_forward_impl` is written with ordinary burn tensor ops
        // (`matmul`, `mul`, ...) and is generic over the backend, so handing
        // it the INCOMING tensors lets burn build the graph itself. No custom
        // node, no leaf, every step of the recurrence differentiable. Slower
        // than the fused kernels and correct - which is the trade the project
        // is making deliberately until a fused path is A/B-proven against
        // this one rather than against a run that trained nothing.
        let (o, s, _scratch) = crate::forward::chunk_wy_forward_impl(
            q, k, v, g, b, w, state, scale, chunk_size, None,
        );
        note_fused_declined();
        Fused::Fused((o, s))
    }
}

#[cfg(feature = "cuda")]
pub use fused::*;
