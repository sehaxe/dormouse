//! # dormouse-bitnet - BitNet quantization family for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! | Function | Reference | What |
//! |----------|-----------|------|
//! | `weight_quant_ternary` | b1.58 / v2 Eq.1 (2402.17764, 2504.18415) | W̃ = α·RoundClip(W/(α+ε), −1, 1), absmean |
//! | `weight_quant_2bit` | BitNet-style 2-bit | W -> {-2,-1,0,1,2}*row-absmean, STE |
//! | `weight_quant_ternary_nm` | Sparse-BitNet (2603.05168) | N:M sparse ternary, Dual-STE |
//! | `activation_quant_8bit` | b1.58 (2024) | absmax -> 8-bit |
//! | `bitnet_v2_quantize` | v2 (2504.18415) | Hadamard + absmax/absmean |
//! | `fast_walsh_hadamard` | v2 (2025) | FWHT O(n log n) |
//!
//! Module map: [`fwt`] — the Walsh–Hadamard rotation (tensor butterfly +
//! naive reference); [`quant::weights`] / [`quant::activations`] — the
//! quantizers; [`sparse`] — Sparse-BitNet N:M mask, Dual-STE and
//! [`crate::sparse::SparseBitLinear`]; [`fwt_cuda`] / fused kernels behind
//! the `cuda` feature; [`precision`] — matmul precision selector.

pub mod fwt;
#[cfg(any(feature = "cuda", feature = "autodiff"))]
pub mod fwt_cuda;
pub mod precision;
pub mod quant;
pub mod sparse;

pub use fwt::{fast_walsh_hadamard, fast_walsh_hadamard_tensor, fwt_reference_naive};
pub use quant::activations::{
    activation_quant_8bit, bitnet_v2_quantize, quantize_4bit, quantize_8bit, quantize_tensor,
};
pub use quant::weights::{
    weight_quant_2bit, weight_quant_b158, weight_quant_sign_centered, weight_quant_ternary,
    weight_quant_ternary_nm,
};
