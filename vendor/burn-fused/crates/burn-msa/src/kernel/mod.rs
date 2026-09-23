// Fused cubecl kernels (require cubecl >= 0.11).
#[cfg(any(feature = "cubecl", feature = "cuda"))]
pub mod topk_select;
