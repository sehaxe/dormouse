//! # dormouse-situ - SiTU-GLU Activation for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Sigmoid-Tanh Unit gated linear - bounded GLU variant from Kimi K3.
//! Applies `softcap(x, beta) = beta * tanh(x / beta)` to gate and up
//! branches before gating, preventing activation explosion.
//!
//! | Paper | What |
//! |-------|------|
//! | [Kimi K3](https://arxiv.org/abs/2607.24653) (Moonshot, 2026) | SiTU-GLU bounded GLU with tanh soft-caps |
//!
//! Benchmarked in Kimi K3 (2.8T MoE, 104B active): better stability
//! than SwiGLU for deep models and low-bit quantization (MXFP4).
use burn::tensor::{activation, Tensor};

/// The soft-cap threshold K3 runs on its **gate** branch: `4.0`.
///
/// arXiv:2607.24653v2 §2.3.2 ("we set the soft-cap hyperparameters to
/// β1 = 4 for the gate branch"), and the released
/// `moonshotai/Kimi-K3@main config.json` → `text_config.activation_situ_beta
/// = 4.0`.
///
/// This is a CONSTANT, and it is not the same number as this crate's
/// `situ_glu` default: `SituAndMul.__init__` itself defaults to `beta = 1.0`
/// (`modeling_kimi_linear.py:70`) and K3 overrides it in the config. A caller
/// that takes the default gets a cap 4× too tight, silently — see the note on
/// [`K3_UP_BETA`].
pub const K3_GATE_BETA: f64 = 4.0;
/// The soft-cap threshold K3 runs on its **up** branch: `25.0`.
///
/// arXiv:2607.24653v2 §2.3.2 (β2 = 25), and
/// `text_config.activation_situ_linear_beta = 25.0` in the same released
/// `config.json`.
///
/// β is FIXED in the paper — not learned, not swept — and it is not
/// scale-free: Eq (12) caps an absolute magnitude, so β=25 against a `Wu` that
/// does not keep `Wu x` O(1) caps the branch to a constant. K3's input to the
/// up-projection is RMSNormed, which is what makes its β portable at all.
pub const K3_UP_BETA: f64 = 25.0;

// `autodiff` alone also compiles this module, deliberately: the fused ADJOINT
// and its strategy seam live here, and a gate that can only be compile-checked
// with a GPU is how a `NoCheckpointing`-only entry survived review. The
// kernels inside stay `#[cfg(feature = "cuda")]`. `pub` because the seam
// counters are the arm's only externally visible evidence (ADR-0019).
#[cfg(any(feature = "cuda", feature = "autodiff"))]
pub mod fused_situ;

/// Soft-cap: `beta * tanh(x / beta)`
///
/// Bounded near ±beta, linear near 0. From Kimi K3.
pub fn softcap(x: Tensor<2>, beta: f64) -> Tensor<2> {
    x.div_scalar(beta).tanh().mul_scalar(beta)
}

/// SiTU-GLU: bounded gated linear unit.
///
/// ```text
/// gate, up = split(gate_up)
/// gate_cap = softcap(gate, beta_gate)
/// up_cap   = softcap(up, beta_up)
/// gate_act = gate_cap * sigmoid(gate)   // Swish factor on the RAW pre-activation
/// return gate_act * up_cap
/// ```
///
/// Per Kimi K3 Eq (12): `beta1*tanh(Wg x/beta1) ⊙ sigmoid(Wg x) ⊙ beta2*tanh(Wu x/beta2)`.
/// The sigmoid must act on the uncapped gate so the negative tail vanishes
/// (Swish-like), which is the paper's stated design goal.
///
/// `gate_up`: `[N, 2*hidden]` - concatenated gate and up projections
/// `hidden`: size of each half
/// `beta_gate`, `beta_up`: soft-cap thresholds (default: 1.0)
pub fn situ_glu(gate_up: Tensor<2>, hidden: usize, beta_gate: f64, beta_up: f64) -> Tensor<2> {
    let [n, _d2] = gate_up.dims();

    #[cfg(all(feature = "autodiff", feature = "cuda"))]
    {
        // Probe BOTH checkpointing strategies. `Autodiff<Inner>`'s second type
        // parameter defaults to `NoCheckpointing` and the downcast compares the
        // whole backend type, so probing only that one silently sent a
        // `BalancedCheckpointing` caller (dormouse's backend) to the ~7-pass
        // tensor path: the right answer, slowly, with nothing counting it.
        use burn_autodiff::checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing};
        type CudaBare = burn_cubecl::CubeBackend;
        if let Some(out) = crate::fused_situ::situ_glu_autodiff_s::<CudaBare, NoCheckpointing>(
            gate_up.clone(),
            hidden,
            beta_gate,
            beta_up,
        ) {
            return out;
        }
        if let Some(out) = crate::fused_situ::situ_glu_autodiff_s::<
            CudaBare,
            BalancedCheckpointing,
        >(gate_up.clone(), hidden, beta_gate, beta_up)
        {
            return out;
        }
    }
    #[cfg(feature = "cuda")]
    if let Some(out) = crate::fused_situ::situ_glu_cuda(&gate_up, hidden, beta_gate, beta_up) {
        return out;
    }

    let gate = gate_up.clone().slice([0..n, 0..hidden]);
    let up = gate_up.slice([0..n, hidden..2 * hidden]);

    let gate_cap = softcap(gate.clone(), beta_gate);
    let up_cap = softcap(up, beta_up);
    gate_cap.mul(activation::sigmoid(gate)).mul(up_cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Distribution};
    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn softcap_bounded() {
        let vals = vec![100.0f32, -100.0f32, 0.5f32, -0.5f32];
        let x = Tensor::<1>::from_floats(vals.as_slice(), &dev()).reshape([4, 1]);
        let capped = softcap(x, 1.0);
        let vals: Vec<f32> = capped
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(vals[0].abs() < 1.1, "tanh(100) should be ~1");
        assert!(vals[2] > 0.0, "near zero should be linear");
    }
    #[test]
    fn situ_shape() {
        let gu = Tensor::<2>::random([16, 128], Distribution::Default, &dev());
        assert_eq!(situ_glu(gu, 64, 1.0, 1.0).dims(), [16, 64]);
    }
    #[test]
    fn situ_finite() {
        let gu = Tensor::<2>::random([32, 256], Distribution::Default, &dev());
        let out = situ_glu(gu, 128, 1.0, 1.0);
        let vals: Vec<f32> = out
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(vals.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn situ_negative_gate_tail_vanishes() {
        // Paper Eq (12): sigmoid acts on the RAW gate, so a large negative
        // gate pre-activation makes the output -> 0 (Swish tail), instead of
        // sigmoid(capped) which leaves a ~ -0.27 residual at beta=1.
        let gu = Tensor::<1>::from_floats(vec![-50.0f32, 50.0, 50.0, 50.0].as_slice(), &dev())
            .reshape([2, 2]);
        let out = situ_glu(gu, 1, 1.0, 1.0);
        let vals: Vec<f32> = out
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(
            vals[0].abs() < 1e-3,
            "gate=-50 should vanish (Swish tail), got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 1.0).abs() < 1e-3,
            "gate=+50 should saturate near +1, got {}",
            vals[1]
        );
    }

    // ---------------------------------------------------------------------
    // THE FORM GATE. Everything above is blind where this arm's claim lives:
    // `situ_shape` pins dims, `situ_finite` pins finiteness, `softcap_bounded`
    // cannot tell `beta*tanh(x/beta)` from a bare `tanh`, and
    // `situ_negative_gate_tail_vanishes` pins the raw-gate sigmoid - which
    // SwiGLU ALSO has, so all four pass on a crate rewritten to plain SwiGLU.
    // A perturbation that leaves every existing gate green proves nothing, so
    // the two things this adds are (1) the reference's own numbers and (2) an
    // assertion that the fixture is FAR from the forms it is not - the gate
    // states its own separating power instead of trusting that it has any.
    // ---------------------------------------------------------------------

    /// The K3 form fixture, at K3's β. H = 1, so a row IS `(gate, up)` and
    /// every golden is the arm's whole contribution at that point.
    ///
    /// Columns: `(gate, up, golden, min_rel_from_swiglu, min_rel_from_silu)`.
    /// The separation columns are the SYMMETRIC relative distance
    /// `|a-b| / max(|a|,|b|)` - the same measure `tools/gen_ref.py` prints, so
    /// the thresholds below and the numbers in the findings file cannot drift
    /// apart. `0.0` means THIS ROW MAKES NO CLAIM about that form, not "they
    /// agree": row 7 (gate=100) is the honest case, where SiTU gives 99.93 and
    /// `silu(100)` gives 100.0, a distance of 6.7e-4, and a row that asserted
    /// otherwise would be asserting something false.
    ///
    /// Every claimed row measures at least 0.42, so 0.3 is a separation claim
    /// with margin rather than a golden that could drift: these are exact
    /// functions, and the distances are factors of 2 to 100, not ulps.
    ///
    /// The goldens are f32, the reference's op order, from
    /// `tools/gen_ref.py` - see that file's header for the external file, its
    /// sha256, and why these are a transcription and not a run of Moonshot's
    /// code. 1e-6 is the f32 ulp band, not a loose bound.
    const K3_FIXTURE: [(f32, f32, f32, f32, f32); 10] = [
        (0.000, 0.000, 0.0, 0.0, 0.0),
        (1.000, 1.000, 0.715_817_8, 0.0, 0.0), // near the origin: see Eq (18) below
        (-1.000, 1.000, -0.263_334_6, 0.0, 0.0),
        (-0.500, -0.250, 0.046_946_75, 0.0, 0.3),
        (4.000, 25.000, 56.959_32, 0.3, 0.3),
        (20.000, 100.000, 99.923_86, 0.3, 0.3),
        (100.000, 100.000, 99.932_93, 0.3, 0.0),
        (-20.000, 100.000, -2.059_584e-7, 0.3, 0.3),
        (-100.000, 50.000, -0.0, 0.0, 0.0), // both forms: the row is dead
        (0.125, 7.500, 0.483_430_18, 0.0, 0.3),
    ];

    fn f32s(t: Tensor<2>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Symmetric relative distance, so a 1e-7-scale row is not scored against
    /// its own magnitude.
    fn rel(a: f32, b: f32) -> f32 {
        (a - b).abs() / a.abs().max(b.abs()).max(1e-30)
    }

    /// A `[rows, 2]` gate_up from flat `(gate, up)` pairs. `from_floats` infers
    /// the rank from the slice length, so the pairs are concatenated into a 1-D
    /// tensor and reshaped.
    fn gate_up(pairs: &[(f32, f32)]) -> Tensor<2> {
        let flat: Vec<f32> = pairs.iter().flat_map(|(g, u)| [*g, *u]).collect();
        Tensor::<1>::from_floats(flat.as_slice(), &dev()).reshape([pairs.len(), 2])
    }

    /// Eq (12) at the released K3 β, against the reference's own numbers, AND
    /// far from the two forms it is not.
    #[test]
    fn situ_matches_the_released_reference_and_separates_from_what_it_is_not() {
        let pairs: Vec<(f32, f32)> = K3_FIXTURE.iter().map(|r| (r.0, r.1)).collect();
        let out = f32s(situ_glu(gate_up(&pairs), 1, K3_GATE_BETA, K3_UP_BETA));
        assert_eq!(out.len(), K3_FIXTURE.len(), "one output per fixture row");

        for (i, &(gate, up, golden, min_swi, min_silu)) in K3_FIXTURE.iter().enumerate() {
            let got = out[i];
            let err = rel(got, golden);
            assert!(
                err < 1e-6,
                "row {i} (gate={gate}, up={up}): got {got}, the reference says {golden} \
                 (rel {err:.3e}) - Eq (12) is no longer what runs"
            );

            // The two forms this arm is NOT, on the same row.
            let swiglu = gate * sigmoid(gate) * up;
            let silu = gate * sigmoid(gate);
            let d_swi = rel(got, swiglu);
            let d_silu = rel(got, silu);
            assert!(
                d_swi >= min_swi,
                "row {i} (gate={gate}, up={up}): SiTU {got} is only {d_swi:.3e} from SwiGLU \
                 {swiglu}, below the {min_swi} this row claims - the fixture has stopped \
                 separating the cap from no cap at all"
            );
            assert!(
                d_silu >= min_silu,
                "row {i} (gate={gate}, up={up}): SiTU {got} is only {d_silu:.3e} from silu \
                 {silu}, below the {min_silu} this row claims"
            );
        }
    }

    /// Eq (18)/App. B: the scaled tanh is `z + O(z^3)` near the origin, so
    /// SiTU is a small PERTURBATION of SwiGLU there and a bound away from it.
    /// Without this the row-1 golden could be satisfied by any activation that
    /// is roughly SwiGLU-shaped near zero, cap or no cap.
    #[test]
    fn situ_is_a_small_perturbation_of_swiglu_near_the_origin() {
        let got = f32s(situ_glu(gate_up(&[(1.0, 1.0)]), 1, K3_GATE_BETA, K3_UP_BETA))[0];
        let swiglu = sigmoid(1.0);
        let d = rel(got, swiglu);
        assert!(
            d < 0.05,
            "at gate=1 SiTU is {d:.3e} from SwiGLU, not the <5% Eq (18) predicts \
             (got {got}, swiglu {swiglu})"
        );
    }

    /// The bound is the mechanism, and it is a property of β: Eq (19),
    /// `‖SiTU-GLU(x)‖∞ <= beta1*beta2`. `softcap_bounded` cannot see the β
    /// scale, so this pins that the OUTPUT is bounded by β and that the
    /// product bound is reachable - a cap that saturated somewhere else would
    /// satisfy `softcap_bounded` and this at once.
    #[test]
    fn the_output_bound_is_beta1_times_beta2() {
        let rows = [(1e4f32, 1e4f32), (-1e4, 1e4), (1e4, -1e4), (0.0, 1e4)];
        let bound = (K3_GATE_BETA * K3_UP_BETA) as f32;
        for (i, got) in f32s(situ_glu(gate_up(&rows), 1, K3_GATE_BETA, K3_UP_BETA))
            .into_iter()
            .enumerate()
        {
            assert!(
                got.abs() <= bound * (1.0 + 1e-6),
                "row {i} gave {got}, past the Eq (19) bound |f| <= {bound}"
            );
        }
        // Reachable, not merely respected: tanh(25) and tanh(4) are both 1 to
        // within f32, so a saturated row sits ON the bound.
        let at_bound = f32s(situ_glu(gate_up(&rows[..1]), 1, K3_GATE_BETA, K3_UP_BETA))[0];
        assert!(
            rel(at_bound, bound) < 1e-4,
            "a saturated row gave {at_bound}, {bound} away from the bound - the caps are \
             not both saturating, so the bound is not what it claims"
        );

        // And β is what sets it: the same row at β = 1/1 is bounded by 1, which
        // is the crate's own default and NOT what K3 runs.
        let tight = f32s(situ_glu(gate_up(&rows[..1]), 1, 1.0, 1.0))[0];
        assert!(
            tight.abs() <= 1.0 + 1e-6 && tight.abs() > 0.9,
            "beta=1/1 gave {tight}: the bound must be the PRODUCT of the two betas"
        );
    }
}
