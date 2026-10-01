//! Deterministic top-k indices: one full descending sort along `dim`,
//! sliced to the first `k` index columns.

use burn::tensor::{Int, Tensor};

/// Indices of the `k` largest elements along `dim` of a 2D tensor, in
/// descending value order.
///
/// Implemented as a single [`Tensor::argsort_descending`] + `narrow` instead
/// of rounds of native `argtopk`: cubecl 0.11.0-pre.1 had a documented
/// garbage-ArgTopK defect, and pre.2 still shows uninitialized reads /
/// partial writes under compute-sanitizer initcheck (plus wild indices
/// causing illegal-address crashes), so the reduce op is avoided entirely.
///
/// Ordering semantics differ from the old argtopk-rounds impl on ties:
/// rounds returned tie order from argtopk internals across masked rounds;
/// the sort returns an unstable-sort-defined order (backend-specific). The
/// selected *set* for tied values may therefore differ between impls —
/// callers must not depend on tie order among equal scores.
///
/// `k = 0` returns an empty `[B, 0]` tensor; `k > nn` is clamped down to
/// `nn`.
///
/// The returned indices carry the DEVICE's `Int` dtype
/// (`DeviceSettings::int_dtype`), which is what `argsort_descending` already
/// produces and what every other `Int` producer here — `arange`, `empty`,
/// `expand` — produces. That is not cosmetic: a `Tensor<D, Int>` is only
/// interopable with the `gather` / `scatter` / `equal` of the same device, so
/// a hard-coded dtype here silently misreads every consumer's index buffer.
pub fn topk_indices(x: Tensor<2>, k: usize, dim: usize) -> Tensor<2, Int> {
    let device = x.device();
    // work along dim 1; transpose for dim 0
    let (work0, transposed) = if dim == 0 {
        (x.transpose(), true)
    } else {
        (x, false)
    };
    let [b, nn] = work0.dims();
    let k = k.min(nn);
    if k == 0 {
        let empty = Tensor::<2, Int>::empty([b, 0], &device);
        return if transposed { empty.transpose() } else { empty };
    }
    let idx = work0.argsort_descending(1).narrow(1, 0, k); // [B, k], device Int dtype
    if transposed {
        idx.transpose()
    } else {
        idx
    }
}
