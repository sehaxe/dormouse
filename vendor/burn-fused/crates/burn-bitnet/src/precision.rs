//! Runtime kernel-precision selector (fp32 / bf16 / fp8) for the quantized
//! activation matmuls, opt-in via the `KERNEL_PRECISION` env var.
//!
//! The default is [`Precision::Fp32`], which keeps the pre-selector behavior
//! bit-identical. `bf16` is opt-in: on this crate's target (RTX 5060 Ti) bf16
//! matmuls are faster only for large matmuls (~1.6-1.9x) and SLOWER for small
//! rank-64 ones (~2.65x), so it must not be the default.
//!
//! burn 0.22 has no FP8 support (no `FloatKind::F8` / `DType::F8`), so
//! [`Precision::Fp8`] is a documented stub that falls back to fp32 with a
//! one-time warning. A real fp8 dtype (custom FloatKind/DType + kernel support
//! across backends) is a separate task, deliberately not invented here.

use std::sync::atomic::{AtomicBool, Ordering};

/// Kernel precision for the activation matmul in quantized linear layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Full fp32 math. Default; bit-identical to the pre-selector path.
    Fp32,
    /// bf16 matmul: cast activations (and weights) to bf16, matmul in bf16,
    /// cast the output back to fp32.
    Bf16,
    /// Unsupported stub: burn has no FP8 dtype. Falls back to [`Precision::Fp32`]
    /// with a one-time warning.
    Fp8,
}

static FP8_WARNED: AtomicBool = AtomicBool::new(false);

/// Read the runtime kernel precision from `KERNEL_PRECISION`
/// (`fp32` | `bf16` | `fp8`; absent or unknown -> [`Precision::Fp32`]).
pub fn kernel_precision() -> Precision {
    precision_from_env(std::env::var("KERNEL_PRECISION").ok().as_deref())
}

/// Parse a `KERNEL_PRECISION` value; absent/unknown -> fp32.
fn precision_from_env(var: Option<&str>) -> Precision {
    match var {
        Some("bf16") => Precision::Bf16,
        Some("fp8") => Precision::Fp8,
        _ => Precision::Fp32,
    }
}

/// One-time warning that fp8 is unsupported (called when the stub is hit).
pub fn warn_fp8_unavailable() {
    if !FP8_WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "burn-bitnet: KERNEL_PRECISION=fp8 unsupported by burn 0.22 (no FP8 dtype); \
             falling back to fp32. A real FP8 dtype is a separate task."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kernel_precision_env() {
        assert_eq!(precision_from_env(None), Precision::Fp32);
        assert_eq!(precision_from_env(Some("fp32")), Precision::Fp32);
        assert_eq!(precision_from_env(Some("bf16")), Precision::Bf16);
        assert_eq!(precision_from_env(Some("fp8")), Precision::Fp8);
        assert_eq!(precision_from_env(Some("garbage")), Precision::Fp32);
        assert_eq!(precision_from_env(Some("BF16")), Precision::Fp32);
    }
}
