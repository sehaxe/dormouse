#[cfg(feature = "cuda")]
pub mod chunk_adjoint_cube;
#[cfg(feature = "cuda")]
pub mod chunk_cube;
pub mod fused_recurrent;
#[cfg(feature = "cuda")]
pub mod fused_recurrent_cube;

/// `t`'s buffer as a row-major `CubeTensor`, materializing a strided view
/// ONCE. Every fused kernel here reads raw `CubeTensor` handles, so this is
/// the one place that decision is made for all of them.
///
/// The trainer hands this seam PERMUTED views — `KdaModule::project` builds
/// `[B,T,H,K].permute([0,2,1,3])` (burn-kda `lib.rs:409`) — and a permute is
/// metadata only, so the non-singleton dims do not nest in row-major order.
/// The materializer must therefore NOT be an elementwise op: on burn 0.22
/// `mul_scalar` allocates its output dense in the *memory* order of its
/// operand and then permutes the metadata back (burn-cubecl
/// `kernel/memory_order.rs:38`), so the "copy" comes back with the very
/// strides it was asked to remove. The recursive version of this function
/// that shipped with the seam therefore recursed on its own output and
/// overflowed the stack at every shape. A flatten cannot be expressed through
/// non-nesting strides, so burn materializes it in logical value order
/// (`ReshapeAction::Recompute`, burn-cubecl `ops/base.rs:309`) instead: one
/// copy, then contiguous. The result is VERIFIED, not assumed — a pitched
/// allocation still reports `None`, which is counted, and the caller falls
/// back to the tensor path.
#[cfg(feature = "cuda")]
pub(crate) fn contiguous_cube_of<B: burn::backend::Backend, const D: usize>(
    t: &burn::tensor::Tensor<D>,
) -> Option<burn_cubecl::tensor::CubeTensor>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    use burn::tensor::Tensor;
    use burn_cubecl::tensor::CubeTensor;
    use std::any::{Any, TypeId};

    if TypeId::of::<B>() != TypeId::of::<burn_cubecl::CubeBackend>() {
        return None;
    }
    let read = |t: &Tensor<D>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let cube = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
        Some(cube.clone())
    };
    let cube = read(t)?;
    if row_major(&cube.meta.shape().dims::<D>(), &cube.meta.strides().to_vec()) {
        return Some(cube);
    }
    let shape = t.dims();
    let flat: usize = shape.iter().product();
    let materialized = read(&t.clone().reshape([flat]).reshape::<D, _>(shape))?;
    if !row_major(
        &materialized.meta.shape().dims::<D>(),
        &materialized.meta.strides().to_vec(),
    ) {
        // A pitched (padded) allocation: the kernels index linearly, so this
        // is a real fallback and not a silent one.
        crate::alloc_trace::note_contiguous_fallback();
        return None;
    }
    crate::alloc_trace::note_contiguous_copy();
    Some(materialized)
}

/// Row-major (C-order) strides, ignoring size-1 dims: a `[1,H,T,K]` view of a
/// contiguous buffer is still linearly readable.
#[cfg(feature = "cuda")]
fn row_major(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected = 1usize;
    for i in (0..shape.len()).rev() {
        if shape[i] == 1 {
            continue;
        }
        if strides[i] != expected {
            return false;
        }
        expected *= shape[i];
    }
    true
}
