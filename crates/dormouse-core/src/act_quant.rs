//! BitNet a4.8-style activation quantization (b1.58 2B4T, 2025):
//! 4-bit activations + 1.58-bit weights trained from scratch with STE.
//!
//! The weight side already exists (SpectralLinear's ternary/2-bit STE
//! quantizers in burn-bitnet). This module adds the activation side with
//! straight-through estimators, applied before the matmuls. The quantizers
//! are f32 tensor ops in the autodiff graph (round has no gradient, so STE
//! is x + (xq - x).detach()), which sidesteps the stack's bf16-backward
//! limitation entirely.
//!
//! Formats:
//! - [`ActFormat::Fp4`]: **e2m1**, the OCP MX FP4 grid: 1 sign, 2 exponent
//!   (bias 1), 1 mantissa, so the magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6 -
//!   16 codes, 4 bits. There is no 0.75; the only subnormal is 0.5.
//! - [`ActFormat::Int`]: symmetric integer (int4 levels 7, int8 127).
//!
//! The scale maps a block's max onto the FORMAT's max ([`ActFormat::max_value`]),
//! not onto 1. Normalizing to [-1, 1] and calling that "fp4" is the bug this
//! comment used to hide: only {0, 0.5, 0.75, 1} were ever reachable, so the
//! format carried 3 bits of code space and none of 1.5..6 - and 1.0 only at
//! the exact block max.
//!
//! Scales are per-token (group = 0) or per-group of `group` columns; the
//! a4.8 recipe uses group scales for the FFN activations.

use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, Tensor};

/// Activation quantization format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActFormat {
    /// e2m1 fp4 - the OCP MX FP4 grid, [`E2M1`]. Four bits, and now four
    /// bits' worth of levels are actually reachable (they were not: see the
    /// module doc).
    Fp4,
    /// Symmetric integer, `bits` levels 2^(bits-1)-1.
    Int(u32),
}

impl ActFormat {
    /// Higher-precision variant for the sensitive attention path (the a4.8
    /// recipe keeps attention and the early/late layers at 8 bits).
    pub fn attn(self) -> Self {
        match self {
            ActFormat::Fp4 => ActFormat::Int(8),
            ActFormat::Int(b) => ActFormat::Int(b.max(8)),
        }
    }

    /// Largest magnitude the format represents, and therefore what a block's
    /// max must be scaled ONTO: 6 for e2m1, 2^(bits-1)-1 for the integers.
    /// The scale is `block_max / max_value`; using `block_max` alone is what
    /// left fp4 with 3 of its 8 magnitudes unreachable.
    pub fn max_value(self) -> f32 {
        match self {
            ActFormat::Fp4 => 6.0,
            ActFormat::Int(bits) => ((1i64 << (bits - 1)) - 1) as f32,
        }
    }
}

/// Quantize `x [B, D]` with a straight-through estimator.
/// `group = 0` -> one scale per token (row); `group > 0` -> one scale per
/// `group` columns (the a4.8 FFN recipe).
pub fn quant_act<B: Backend>(x: Tensor<2>, fmt: ActFormat, group: usize) -> Tensor<2>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let [b, d] = x.dims();
    let g = if group == 0 { d } else { group.min(d) };
    // The block max, in the format's OWN units: dividing by `block_max`
    // normalized to [-1, 1], which for e2m1 (max 6) made every level above 1
    // unreachable and cost the format half its code space.
    let fmax = fmt.max_value();
    let scale = if g == d {
        // Per-token scale: [b, 1] broadcast over d.
        x.clone().abs().max_dim(1)
    } else {
        // Per-group scale: [b, d/g, 1] broadcast inside each group.
        x.clone()
            .abs()
            .reshape([b, d / g, g])
            .max_dim(2)
            .reshape([b, d / g, 1])
            .repeat(&[1, 1, g])
            .reshape([b, d])
    };
    let scale = scale.clamp_min(1e-8).div_scalar(fmax);
    let norm = x.clone().div(scale.clone()); // in [-fmax, fmax]
                                             // `q` comes back in the FORMAT'S OWN UNITS (an e2m1 level, or an integer
                                             // level count) and is dequantized by the same `scale`, so the int path is
                                             // bit-identical to the old `round(norm*l)/l * block_max` spelling: the
                                             // format change is confined to Fp4.
    let q = match fmt {
        ActFormat::Fp4 => fp4_round::<B>(norm),
        ActFormat::Int(bits) => {
            let l = ((1i64 << (bits - 1)) - 1) as f32;
            debug_assert_eq!(l, fmax, "max_value and the int level must agree");
            norm.round().clamp(-l, l)
        }
    };
    let xq = q.mul(scale);
    // STE: forward uses the quantized value, backward flows through x.
    x.clone().add(xq.sub(x).detach())
}

/// The e2m1 magnitude grid, ascending. The OCP MX FP4 format: 1 sign bit,
/// 2 exponent bits (bias 1), 1 mantissa bit. Positive values are
/// 0, 0.5, 1, 1.5, 2, 3, 4, 6; 0.5 is the subnormal and 0.75 IS NOT IN THE
/// FORMAT. The old `0.625` half-step emitted 0.75, which is not e2m1.
pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Round to the nearest e2m1 value, **ties to the EVEN CODE** — the format is
/// round-half-to-even, the same rule IEEE 754 and torchao's MX-FP4 conversion
/// follow. `x` must already be in the format's range (the caller scales to
/// +/-6); anything above 6 saturates, anything below 0.25 is zero.
///
/// "Even" means the parity of the code, and because [`E2M1`] holds one level
/// per code, that is the parity of the index. The ties therefore do NOT all
/// go the same way, which is why no single `>` or `>=` expresses the rule and
/// why this comment used to be wrong: it said "ties away from zero", which is
/// ties-UP, and disagrees with the reference at 4 of the 7 interior ties (see
/// the ladder comment below for the table). Measured against
/// `pytorch/ao@3972ed01`'s `to_mx` by `crates/dormouse-core/tests/
/// e2m1_oracle.rs`, which reads the raw NIBBLES rather than decoded values.
///
/// The previous implementation derived the level from a log2 exponent plus a
/// mantissa step, which is where 0.75 came from, and the previous caller
/// normalized to [-1, 1] so the levels above 1 were dead code. This is a
/// direct search over [`E2M1`]: the grid is eight numbers, and a bucket test
/// per level is both shorter to read and impossible to get subtly wrong the
/// way a re-derived mantissa rule is.
fn fp4_round<B: Backend>(x: Tensor<2>) -> Tensor<2>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let dims = x.dims();
    let device = x.device();
    let a = x.clone().abs();
    let sign = x.div(a.clone().clamp_min(1e-12)).clamp(-1.0, 1.0);
    // Ascending ladder of thresholds, each one overwriting the last: 0.5
    // claims [0.25, ...), then 1.0 claims [0.75, ...) on top of it, and so on
    // to 6.0 at [5.0, ...), which saturates. Below 0.25 no threshold fires and
    // the value stays zero. A two-sided mask would be the same thing spelled
    // with a Bool AND; the thresholds are already monotonic.
    //
    // TIES GO TO THE EVEN CODE, and that is a fix rather than a detail.
    // The format is round-half-to-EVEN (IEEE 754's rule, which the hardware
    // MX-FP4 conversion follows), so at a midpoint the value is the one whose
    // CODE is even - and "even" alternates along the ladder, so no single
    // `>` or `>=` expresses it. The old code used `>=` everywhere, i.e.
    // ties-UP, which disagrees with the reference at 4 of the 7 interior ties:
    //   0.25 -> 0.0   (ref)  vs 0.5   (ours)   i=1 odd  -> take the LOWER
    //   0.75 -> 1.0   (ref)  vs 1.0   (agree)   i=2 even -> take the UPPER
    //   1.25 -> 1.0   (ref)  vs 1.5   (ours)   i=3 odd
    //   1.75 -> 2.0   (ref)  vs 2.0   (agree)   i=4 even
    //   2.5  -> 2.0   (ref)  vs 3.0   (ours)   i=5 odd
    //   3.5  -> 4.0   (ref)  vs 4.0   (agree)  i=6 even
    //   5.0  -> 4.0   (ref)  vs 6.0   (ours)  i=7 odd  - a 50% error, at the
    //                        top of the range where the block scale puts the
    //                        most-used values.
    // Measured against torchao's real MX-FP4 quantiser (`pytorch/ao@3972ed01`)
    // by `crates/dormouse-core/tests/e2m1_oracle.rs`.
    let mut val = Tensor::zeros(dims, &device);
    for (i, level) in E2M1.iter().enumerate().skip(1) {
        let lo = 0.5 * (E2M1[i - 1] + level);
        // i is the index of the UPPER level, i.e. the code we would claim.
        // Even code -> claim it on a tie (`>=`); odd -> leave it (`>`).
        let claimed = if i % 2 == 0 {
            a.clone().greater_equal_scalar(lo)
        } else {
            a.clone().greater_scalar(lo)
        };
        val = val.mask_fill(claimed, *level);
    }
    sign.mul(val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::Distribution;

    #[test]
    fn quant_act_bounds_and_ste() {
        let dev = burn::tensor::Device::flex();
        let x: Tensor<2> = Tensor::random([16, 64], Distribution::Normal(0.0, 1.0), &dev);
        let q = quant_act::<Flex>(x.clone(), ActFormat::Int(4), 0);
        let v: Vec<f32> = q.into_data().try_to_vec().unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "quantized acts must be finite"
        );
        let orig: Vec<f32> = x.into_data().try_to_vec().unwrap();
        let mut max_d = 0.0f32;
        for (a, b) in orig.iter().zip(v.iter()) {
            max_d = max_d.max((a - b).abs());
        }
        assert!(max_d > 1e-4, "quantization must actually quantize");
    }

    /// Every positive e2m1 magnitude must round to itself. The list is
    /// [`E2M1`] minus the sign, NOT a list written to match the
    /// implementation: 0.75 was on it and is not in the format.
    #[test]
    fn every_e2m1_magnitude_round_trips() {
        let dev = burn::tensor::Device::flex();
        let vals: Vec<f32> = E2M1.iter().skip(1).copied().collect();
        let x: Tensor<2> = Tensor::from_data(
            burn::tensor::TensorData::new(vals.clone(), [1, vals.len()]),
            &dev,
        );
        let out: Vec<f32> = fp4_round::<Flex>(x).into_data().try_to_vec().unwrap();
        for (a, b) in vals.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-4, "e2m1 round-trip: {a} -> {b}");
        }
        // And the negative mirror, which is where a lost sign shows up.
        let x: Tensor<2> = Tensor::from_data(
            burn::tensor::TensorData::new(
                vals.iter().map(|v| -v).collect::<Vec<_>>(),
                [1, vals.len()],
            ),
            &dev,
        );
        let out: Vec<f32> = fp4_round::<Flex>(x).into_data().try_to_vec().unwrap();
        for (a, b) in vals.iter().zip(out.iter()) {
            assert!((a + b).abs() < 1e-4, "e2m1 round-trip: -{a} -> {b}");
        }
    }

    /// A golden of the FORMAT, not of the code: for a grid of inputs, the
    /// output must be the nearest member of [`E2M1`] and nothing else. On an
    /// exact midpoint the answer is the member whose CODE is even, which
    /// alternates along the ladder and is therefore *not* always the coarser
    /// one — 0.75 and 3.5 go UP, the other five interior ties go DOWN.
    /// Written against the format's definition - 1
    /// sign, 2 exponent (bias 1), 1 mantissa, so the magnitudes are exactly
    /// 0, .5, 1, 1.5, 2, 3, 4, 6 - and it fails for any other grid, which is
    /// what the old test could not do.
    #[test]
    fn fp4_output_is_always_the_nearest_e2m1_level() {
        let dev = burn::tensor::Device::flex();
        // Midpoints are where a rounding rule differs; sample around each.
        let mut grid: Vec<f32> = (0..=120).map(|i| i as f32 * 0.1 - 1.0).collect();
        for w in E2M1.windows(2) {
            let mid = 0.5 * (w[0] + w[1]);
            for eps in [-0.02f32, -0.005, 0.0, 0.005, 0.02] {
                grid.push(mid + eps);
                grid.push(-(mid + eps));
            }
        }
        let n = grid.len();
        let want: Vec<f32> = grid
            .iter()
            .map(|v| {
                let a = v.abs();
                // Nearest level, TIES TO THE EVEN CODE - the format's own rule
                // (IEEE round-half-to-even, which the hardware MX-FP4
                // conversion follows), NOT "ties to the lower". The two differ
                // at 4 of the 7 interior ties and this test previously encoded
                // the wrong one, so it agreed with a `>=` ladder that disagrees
                // with torchao's quantiser at 5.0 by 50 %. Verified end to end
                // against the real thing in tests/e2m1_oracle.rs.
                let mut best = 0.0f32;
                for (i, l) in E2M1.iter().enumerate().skip(1) {
                    let claim = a >= 0.5 * (best + l);
                    if claim && (i % 2 == 0 || a > 0.5 * (best + l)) {
                        best = *l;
                    }
                }
                best.copysign(*v)
            })
            .collect();
        let got: Vec<f32> = fp4_round::<Flex>(Tensor::from_data(
            burn::tensor::TensorData::new(grid.clone(), [1, n]),
            &dev,
        ))
        .into_data()
        .try_to_vec()
        .unwrap();
        for (i, (a, b)) in grid.iter().zip(got.iter()).enumerate() {
            assert!(
                (want[i] - b).abs() < 1e-4,
                "e2m1: {a} -> {b}, nearest of E2M1 is {}",
                want[i]
            );
            assert!(
                E2M1.iter().any(|l| (l - b.abs()).abs() < 1e-6),
                "e2m1 produced {b} for {a}, which is not a level of the format"
            );
        }
    }

    /// A ramp over [0, 1] in one block, so the block max is 1 and the scale is
    /// `1 / max_value`. `quant_act` returns DEQUANTIZED values, so level `L`
    /// comes back as `L * scale`.
    fn dequantized_ramp(fmt: ActFormat, n: usize) -> Vec<f32> {
        let dev = burn::tensor::Device::flex();
        let ramp: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let q = quant_act::<Flex>(
            Tensor::from_data(burn::tensor::TensorData::new(ramp, [1, n]), &dev),
            fmt,
            0,
        );
        q.into_data().try_to_vec().unwrap()
    }

    /// The scale must reach the whole format. `--act-quant fp4` is a 4-bit
    /// claim; with the old `scale = block_max` the normalized values stopped at
    /// 1, so {1.5, 2, 3, 4, 6} were unreachable and the "e2m1" grid carried
    /// three levels. This is the test that would have caught it.
    #[test]
    fn fp4_reaches_every_level_through_the_public_quantizer() {
        let out = dequantized_ramp(ActFormat::Fp4, 4096);
        let s = 1.0 / ActFormat::Fp4.max_value();
        let missing: Vec<f32> = E2M1
            .iter()
            .copied()
            .filter(|l| !out.iter().any(|v| (v - l * s).abs() < 1e-4))
            .collect();
        assert!(
            missing.is_empty(),
            "fp4 through quant_act never emitted {missing:?} of the e2m1 grid (scale {s}) - the block scale is not reaching the format max"
        );
    }

    /// The int path's scale change (`block_max / (2^(b-1)-1)` instead of
    /// `block_max`) must be a no-op on the OUTPUT: the old code multiplied the
    /// normalized value by the level count and divided back, so both
    /// spellings produce the same grid. If this ever fails, the refactor
    /// changed a format instead of renaming it.
    #[test]
    fn int4_levels_are_the_whole_symmetric_range() {
        // Block max is 1, so level k dequantizes to k/7 either way - the
        // scale refactor must not move the int grid by a factor of 7.
        let out = dequantized_ramp(ActFormat::Int(4), 4096);
        let want: Vec<f32> = (0..=7).map(|k| k as f32 / 7.0).collect();
        let missing: Vec<f32> = want
            .iter()
            .copied()
            .filter(|v| !out.iter().any(|o| (o - v).abs() < 1e-5))
            .collect();
        assert!(missing.is_empty(), "int4 never emitted {missing:?}");
    }
}
