//! The two conversions the autodiff node needs, from
//! `burn-gdn2/src/cuda_dispatch.rs` (the proven pattern; kept local — a shared
//! helper crate for 40 lines is not a win).
//!
//! `try_into_primitive::<Inner>` — the bare demotion — REFUSES an autodiff
//! tensor (burn-dispatch `src/tensor.rs:481-487`); lifting to the autodiff
//! primitive and taking `.primitive()` is the only legal way in.

use burn::backend::{Backend, BackendTypes, DispatchKindConversion};
use burn::tensor::{DispatchTensor, Tensor};
use burn_autodiff::checkpoint::strategy::CheckpointStrategy;
use burn_autodiff::Autodiff;

/// The autodiff primitive of `Autodiff<Inner, S>` — `burn_autodiff` keeps its
/// `AutodiffTensor` type crate-private, so the only public spelling is the
/// associated type.
pub type AdNode<Inner, S> = <Autodiff<Inner, S> as BackendTypes>::FloatTensorPrimitive;

/// The autodiff PRIMITIVE, whatever the strategy: `None` when the tensor is
/// not on `Autodiff<Inner, S>`.
pub fn autodiff_node<Inner: Backend, S: CheckpointStrategy, const D: usize>(
    t: &Tensor<D>,
) -> Option<AdNode<Inner, S>>
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>>,
{
    t.clone().try_into_primitive::<Autodiff<Inner, S>>().ok()
}

/// The bare tensor a kernel can take, from an autodiff primitive: the same
/// bytes with a DISABLED autodiff context.
pub fn bare_from_node<Inner: Backend, S: CheckpointStrategy, const D: usize>(
    node: &AdNode<Inner, S>,
) -> Tensor<D>
where
    DispatchTensor: DispatchKindConversion<Inner>,
{
    Tensor::from_primitive::<Inner>(node.primitive().clone())
}
