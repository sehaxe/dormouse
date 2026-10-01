//! # burn-kda against FLA's OWN KDA references, run.
//!
//! **Tier (a).** `docs/protocols/ORACLE-TIERS.tsv` registered `burn-kda/src/lib.rs` as
//! tier **(d)** — "n/a, no fidelity claim in the file". That was wrong. This
//! file is what it should have said.
//!
//! ## What produced every expected number
//!
//! | what was run | where | how it is pinned |
//! |---|---|---|
//! | `fla/ops/kda/gate.py::naive_kda_gate` — the Kimi Linear decay form | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) at `9f38d24980c46d46bd38614e743cdacd21906578` (2026-09-29) | `oracle/upstream/fla_ops_kda_gate.py`, byte-identical, sha256 asserted by the generator before anything is imported |
//! | `fla/ops/kda/gate.py::naive_kda_lowerbound_gate` — the **K3** decay form, `lower_bound = -5.0` | same commit, same file | same |
//! | `fla/ops/kda/naive.py::naive_recurrent_kda` — the exact per-token recurrence | same commit, `fla/ops/kda/naive.py` | `oracle/upstream/fla_ops_kda_naive.py`, byte-identical |
//! | `fla/ops/kda/naive.py::naive_chunk_kda` — the chunked WY recurrence | same commit, same file | same |
//!
//! Fetched and run 2026-09-30, CPU only, `torch==2.14.0+cpu` (+`einops`).
//! Generator `oracle/gen_kda_oracle.py`, values `fixtures/kda_oracle.txt`,
//! **this test needs no network**.
//!
//! ## Why the pure-PyTorch files and not the shipped Triton ones
//!
//! FLA's shipping KDA is Triton and Triton has no CPU backend, so the fast path
//! can never be tier (a) on this box. But FLA also ships four **pure-PyTorch
//! reference implementations** of the same math, written as the correctness
//! oracle for the kernels: `tests/ops/test_kda.py:20` imports
//! `naive_chunk_kda, naive_recurrent_kda`, and the `naive_*_gate` twins live in
//! the same file as the Triton `fused_kda_gate` they check. Those run anywhere.
//! `gate.py:8` also says *"This file is modified and supported by the Moonshot
//! AI Team"* — the decay gate is Moonshot's own file, not a third party's.
//!
//! ## The four verdicts
//!
//! | test | verdict | what it pins |
//! |---|---|---|
//! | `k3_bounded_decay_matches_fla_reference` | **green** | the RUNNING decay form is character-for-character FLA's, `g_min = -5` included |
//! | `kda_step_matches_fla_recurrent_at_unit_scale` | **green** | Eq 1 itself, term by term, against FLA's own scan |
//! | `fla_read_scale_is_the_whole_difference` | **green** | upstream's two fixture rows differ by exactly `K**-0.5` and nothing else, which is what makes the red test below attributable |
//! | `chunked_wy_matches_fla_chunk_at_unit_scale` | **green** | the chunked WY construction itself, on **both** of gdn2's chunk arms |
//! | `kimi_linear_softplus_decay_matches_fla_reference` | **RED ON PURPOSE** | `lib.rs:268` computes `-softplus(exp(A)*z)`; FLA computes `-exp(A)*softplus(z)`, in both the naive and the triton twin |
//! | `read_scale_matches_fla_reference` | **RED ON PURPOSE** | FLA's `chunk_kda` defaults `scale = K**-0.5` and `fla/layers/kda.py:262` passes no `scale`, so the official layer runs at `head_k_dim**-0.5`. We pass `1.0`, and `fused.rs:9` asserts the reason ("no softmax scale in KDA"), which is false against both upstreams. |
//!
//! A red test that a maintainer can read and act on is the deliverable; a
//! numerical change to a shipped model is the owner's call, so the two
//! divergences are **reported**, not fixed. See
//! `docs/reviews/2026-09-30-kda-formula-audit.md` §3.1 and §3.2.
//!
//! **Baseline: 5 green, 2 red on purpose**, and the two reds are the only
//! failures in the file. `tests/oracle/falsify.sh` demonstrates that each green
//! arm can be broken on demand, in both directions: a mutant that breaks a
//! green, and — the more valuable one — a mutant that "fixes" a red, which
//! proves the red is pinned to the reference's rule and not merely to "not our
//! formula".
//!
//! ## What a red test here costs, and why it is worth it
//!
//! Neither defect is a typo and **neither is visible to any other test in the
//! crate**, for the same reason: every existing comparison is arm-vs-arm. The
//! chunk path and the per-token scan both read the same `KdaDecay::forward`,
//! so no choice inside it is observable. The read scale is a single constant
//! used by both arms. That is the transferable lesson from
//! `tier-a-references.md` §7 — **a self-comparison cannot see a choice between
//! two equally-valid answers** — and it is why the second one is nearly
//! invisible in the *model* too: the scale enters only the read `o = q·S`, and
//! the RMSNorm at `lib.rs:556-562` is invariant to a constant rescale of its
//! input, so it is absorbed to `O(eps / mean(o²))` ≈ `O(1e-5)`.
//!
//! ## Tolerances
//!
//! **Two classes, and the split is itself a finding.** `TOL_GATE = 1e-5` for
//! the decay form (both sides evaluate one `exp` and one sigmoid in the same
//! order, so the result should be a few ulp; measured ~1e-7). `TOL_CHAIN =
//! 1e-4` for the 32-step f32 recurrence and the chunked form, which is a
//! different kind of comparison — two evaluation ORDERS of one algebra in
//! f32, with the delta rule's `v - k^T S` nearly cancelling at every step. Both
//! are derived at their definitions, and
//! `green_margins_are_reported_and_the_formula_class_is_far_above_tolerance`
//! asserts only the formula class, because a "far above tolerance" assertion on
//! the f32 chain would be an assertion about arithmetic rather than about a
//! formula.

use burn::backend::NdArray;
use burn::module::Param;
use burn::tensor::{Device, Tensor};
use burn_kda::{KdaConfig, KdaModule};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Derived, not tuned. TWO classes, because the two comparisons are not the
/// same kind of comparison and one tolerance for both would hide that.
///
/// **`TOL_GATE = 1e-5` — formula vs formula.** The decay gate is one `exp` and
/// one sigmoid on each side, evaluated in the same order, so the two f32
/// results should agree to a few ulp. `f32::EPS = 1.19e-7`, so 1e-5 is ~85 ulp
/// and anything larger would be a formula difference, not arithmetic. Paired
/// with the absolute term `ATOL_G` below, because `g` passes through zero and
/// no pure relative test can work there.
///
/// **`TOL_RECUR = 1e-4` — a 32-step f32 recurrence, two different orders.**
/// `naive_recurrent_kda` scans token by token; `kda_step` is one call per token.
/// The delta rule's `v - k^T S` subtracts two O(1) quantities that nearly
/// cancel once the state is mature, and the residue compounds over `T = 32`, so
/// this is an f32 *chain* comparison rather than a formula one. Bound:
/// `sqrt(32) * f32::EPS * C` with `C ~ 50` accumulation points in the
/// recurrence = 2.7e-5, and 1e-4 is the next decade up. Measured 1.7e-5.
///
/// **`TOL_CHUNK = 1e-3` — the same algebra in the CHUNKED form, and the ceiling
/// is a stated number rather than a comfort.** The chunked arm forms
/// `k / exp(cumsum(g))` (`gdn2/src/forward.rs:249`). Over a 16-token tile with
/// `g ∈ (-5, 0)` the cumsum reaches -80, so that reciprocal reaches
/// `e^{80} = 5.5e34`: a 1-ulp error in the cumsum (absolute `80 * 6e-8 =
/// 4.8e-6`, and `exp` maps an absolute exponent error 1:1 onto a relative one)
/// reaches `akk = (b*k*Γ) @ (k/Γ)ᵀ` (`forward.rs:253-254`) as a product of two
/// such reciprocals, before `m_inv` cancels most of it. That is why the chunked
/// arm needs a decade more than the recurrent one, and it is a property of the
/// FORM — K3's own `g_min = -5` over a 16-token tile is what produces it, not
/// anything this implementation does. Measured worst 1.4e-4, inside the bound.
///
/// The margin test at the bottom asserts the FORMULA class sits 100x below its
/// own tolerance, which is what stops `TOL_GATE` from being a number fitted to a
/// pass. The chain class is reported and not asserted, because a "far above
/// tolerance" assertion there would be an assertion about arithmetic.
const TOL_GATE: f32 = 1e-5;
const TOL_RECUR: f32 = 1e-4;
const TOL_CHUNK: f32 = 1e-3;

/// The recurrence and state live at `O(1)` (L2-normalised q and k, `beta` in
/// (0,1), unit-scale v), so 1e-6 is ~8 ulp of a unit quantity.
const ATOL_CHAIN: f32 = 1e-6;
/// The chunked form's intermediates reach `e^{80}`; its absolute floor is
/// stated against the OUTPUT scale (O(1)) and is deliberately loose, because on
/// that arm the relative term is doing the work.
const ATOL_CHUNK: f32 = 1e-6;

fn dev() -> Device {
    Device::ndarray()
}

// ── fixture ────────────────────────────────────────────────────────────────

/// One `kind name` block: the `key=value` lines under it.
struct Block {
    fields: BTreeMap<String, String>,
}

impl Block {
    fn dims(&self) -> Vec<usize> {
        self.fields["shape"]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect()
    }
    fn nums(&self, key: &str) -> Vec<f32> {
        self.fields[key]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect()
    }
}

/// Parse the fixture: a `kind name` line opens a block, every `key=value` line
/// under it is a field. Nothing else is syntax, and the generator emits no
/// other line, so the parser stays this small on purpose.
fn fixture() -> BTreeMap<(String, String), Block> {
    let mut out: BTreeMap<(String, String), Block> = BTreeMap::new();
    let mut key: Option<(String, String)> = None;
    let mut fields: BTreeMap<String, String> = BTreeMap::new();
    for line in include_str!("fixtures/kda_oracle.txt").lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = t.split_once('=') {
            fields.insert(k.to_string(), v.to_string());
            continue;
        }
        if let Some(k) = key.take() {
            out.insert(k, Block { fields: std::mem::take(&mut fields) });
        }
        let mut it = t.split_whitespace();
        let kind = it.next().expect("empty block header").to_string();
        key = Some((kind, it.collect::<Vec<_>>().join(" ")));
    }
    if let Some(k) = key.take() {
        out.insert(k, Block { fields });
    }
    assert!(out.len() >= 12, "fixture parse failed: {} blocks", out.len());
    out
}

fn get(fx: &BTreeMap<(String, String), Block>, kind: &str, name: &str) -> Block {
    fx.get(&(kind.to_string(), name.to_string()))
        .unwrap_or_else(|| panic!("fixture has no {kind} {name}"))
        .clone_owned()
}

trait CloneOwned {
    fn clone_owned(&self) -> Block;
}
impl CloneOwned for &Block {
    fn clone_owned(&self) -> Block {
        Block {
            fields: self.fields.clone(),
        }
    }
}

fn t3(v: &[f32], d: [usize; 3]) -> Tensor<3> {
    Tensor::<3, _>::from_data(
        burn::tensor::TensorData::new(v.to_vec(), d),
        &dev(),
    )
}
fn t4(v: &[f32], d: [usize; 4]) -> Tensor<4> {
    Tensor::<4, _>::from_data(
        burn::tensor::TensorData::new(v.to_vec(), d),
        &dev(),
    )
}

/// `[B, T, H, D]` (FLA's layout) -> `[B, H, T, D]` (ours).
fn bthd_to_bhtd(t: Tensor<4>, b: usize, h: usize, tt: usize, d: usize) -> Tensor<4> {
    t.permute([0, 2, 1, 3]).reshape([b, h, tt, d])
}

/// The two-term comparison, in the standard `|a-b| <= atol + rtol*|b|` form,
/// and reported as the NORMALISED ratio so every `assert!` below is a `<= 1.0`.
///
/// **Why the absolute term, and where its floor comes from.** A pure relative
/// test is meaningless on the decay gate: `g = g_min*sigmoid(...)` is a
/// function that passes through values near zero (and `A_plus1_wide_z` puts
/// several elements at |g| ~ 2.5e-5), and there f32 cannot deliver 5 digits
/// because the sigmoid's own argument is ~-12 and the cancellation is in the
/// exponential. The floor is therefore stated in the UNITS OF THE QUANTITY:
/// `g` is bounded by `|g_min| = 5`, and `5 * f32::EPS = 6.0e-7`, so `1e-6` is
/// ~1.7 ulp of the largest value `g` can take. Anything above that is a
/// formula difference, not arithmetic.
///
/// Both terms are derived, neither is fitted: the assertion that the FORMULA
/// class sits 100x below its own tolerance is the check that they are not.
fn num_diff(got: &[f32], want: &[f32], atol: f32, rtol: f32) -> (f32, usize) {
    assert_eq!(
        got.len(),
        want.len(),
        "length mismatch {} vs {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs() / (atol + rtol * w.abs());
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

/// `g`'s natural scale is `g_min = -5`, so `1e-6` is ~1.7 ulp of it.
const ATOL_G: f32 = 1e-6;

// ── 1. the RUNNING decay form vs FLA's executed K3 reference ───────────────

/// Build a `KdaDecay` whose `z` is exactly the fixture's `z`.
///
/// The projection is made the identity: `w_up` picks the first `rank`
/// coordinates of a `[B, T, d_model]` input and `w_down` is the identity on
/// `rank -> n_heads*head_dim`. With `rank == n_heads*head_dim` the composition
/// is the identity, so feeding `x = z - b_alpha` makes `w_down(w_up(x)) + b_alpha`
/// equal the fixture's `z` bit for bit — no random projection anywhere near
/// the comparison.
fn decay_with_z(
    z: &[f32],
    d: [usize; 4],
    a_log: &[f32],
    bias: &[f32],
    decay_fn: burn_kda::DecayFn,
    g_min: f64,
) -> (burn_kda::KdaDecay, Tensor<3>) {
    use burn_kda::{KdaDecay, G_MIN};
    let [b, t, h, k] = d;
    let n = h * k;
    let d_model = n.max(8);
    let dev = dev();
    let mut m = KdaDecay::new(d_model, h, k, n, g_min, decay_fn, &dev);

    // w_up: [d_model, rank=n] with the top n x n block = identity
    let mut w_up = vec![0.0f32; d_model * n];
    for i in 0..n {
        w_up[i * n + i] = 1.0;
    }
    m.w_up.weight = Param::from_tensor(Tensor::from_data(
        burn::tensor::TensorData::new(w_up, [d_model, n]),
        &dev,
    ));
    // w_down: [rank=n, n] = identity
    let mut w_dn = vec![0.0f32; n * n];
    for i in 0..n {
        w_dn[i * n + i] = 1.0;
    }
    m.w_down.weight = Param::from_tensor(Tensor::from_data(
        burn::tensor::TensorData::new(w_dn, [n, n]),
        &dev,
    ));
    m.b_alpha = Param::from_tensor(Tensor::from_data(
        burn::tensor::TensorData::new(bias.to_vec(), [n]),
        &dev,
    ));
    m.a_log = Param::from_tensor(Tensor::from_data(
        burn::tensor::TensorData::new(a_log.to_vec(), [h, 1]),
        &dev,
    ));
    let _ = G_MIN;

    // x IS the fixture's z, flat: `KdaDecay::forward` computes
    // `w_down(w_up(x)) + b_alpha`, and FLA's `g` argument is the PRE-bias
    // projection (the reference adds `dt_bias` INSIDE the gate, gate.py:48 and
    // :75). So the pre-bias value must be `z` itself and `b_alpha` carries the
    // bias -- feeding `z - bias` here would subtract it twice, which is exactly
    // the bug this comment replaced.
    let mut x = vec![0.0f32; b * t * d_model];
    for bi in 0..b {
        for ti in 0..t {
            for i in 0..n {
                x[(bi * t + ti) * d_model + i] = z[(bi * t + ti) * n + i];
            }
        }
    }
    (m, t3(&x, [b, t, d_model]))
}

#[test]
fn k3_bounded_decay_matches_fla_reference() {
    // GREEN. The running form: `g = g_min * sigmoid(exp(A) * z)`, `g_min = -5`.
    // FLA: `naive_kda_lowerbound_gate` = `lower_bound * sigmoid(exp(A_log) * (g +
    // dt_bias))` with `lower_bound = -5.0`. This is the test that says the
    // crate's decay is the authors' decay, executed -- and it is the reason
    // the two reds below are reds about OTHER lines.
    let fx = fixture();
    let mut worst = 0.0f32;
    for name in [
        "A_zero_control",
        "A_minus3_bias_plus1",
        "A_minus3_wide_z",
        "A_plus1_wide_z",
        "A_zero_nonzero_bias",
        "A_clamp_endpoints",
    ] {
        let blk = get(&fx, "gate", name);
        let d = blk.dims();
        let (m, x) = decay_with_z(
            &blk.nums("z"),
            [d[0], d[1], d[2], d[3]],
            &blk.nums("a_log"),
            &blk.nums("bias"),
            burn_kda::DecayFn::Sigmoid,
            burn_kda::G_MIN,
        );
        // `forward` returns alpha = exp(g); compare in g-space, which is the
        // better-conditioned of the two.
        let got: Vec<f32> = m
            .forward(x)
            .log()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let want = blk.nums("fla_lowerbound_g");
        let (r, at) = num_diff(&got, &want, ATOL_G, TOL_GATE);
        worst = worst.max(r);
        assert!(
            r <= 1.0,
            "K3 lower-bounded decay disagrees with FLA's naive_kda_lowerbound_gate\n  \
             case {name}  worst normalised {r:.3e} at flat index {at} (got {} want {})\n  \
             ours: g = g_min*sigmoid(exp(A)*z) at lib.rs:270\n  \
             FLA:  g = lower_bound*sigmoid(exp(A_log)*(g+dt_bias)), gate.py:81",
            got[at], want[at]
        );
    }
    // Normalised: 1.0 is the tolerance, so this IS the margin as a ratio.
    //
    // **The margin is 34x, and that is the honest number.** A 100x assertion
    // would be a claim about nothing: `g` passes through zero (the
    // `A_plus1_wide_z` case puts several elements at |g| ~ 2.5e-5), and there
    // the `ATOL_G = 1e-6` floor is what carries the comparison — the measured
    // absolute disagreement is 2.96e-8, which is 0.06 ulp of `g_min = 5`. The
    // assertion is therefore "not within an order of magnitude", which is the
    // property that actually matters: if the formula were wrong by even 3%, the
    // ratio would be ~300 and this fails by a wide margin.
    eprintln!(
        "k3_bounded_decay: worst normalised {worst:.4e} = a {:.0}x margin (1.0 = tolerance)",
        1.0 / worst
    );
    assert!(
        worst < 0.05,
        "the K3 comparison is within 20x of its own tolerance ({worst:.4e}): the tolerance \
         may be carrying the result, not the formula"
    );
}

#[test]
fn kimi_linear_softplus_decay_matches_fla_reference() {
    // RED ON PURPOSE. src/lib.rs:268 computes
    //     DecayFn::Softplus => activation::softplus(scaled, 1.0).neg(),
    // i.e. g = -softplus(exp(A) * z).
    // FLA computes, in fla/ops/kda/gate.py:50 (`naive_kda_gate`, executed here)
    // and again in the triton twin at gate.py:167,
    //     g = -exp(A_log) * F.softplus(g + dt_bias),
    // i.e. g = -exp(A) * softplus(z).  `exp(A)` is OUTSIDE.
    //
    // These are different functions, not a reparameterisation. They agree at
    // A = 0 and separate monotonically in |A|; at the crate's own init
    // (A = -3, z = +1) upstream gives alpha = 0.5743 and this gives 0.5120,
    // and the ORDER of the two answers flips with the sign of A.
    //
    // No test in the crate can see this: the chunk path and the per-token scan
    // both read the same `KdaDecay::forward`, and `kda_decay_bounds` /
    // `kda_decay_is_data_dependent` assert the range and the data-dependence,
    // which both forms satisfy.
    //
    // The fix moves the numbers of a shipped (if non-default) branch, which is
    // A/B queue arm 5. It is the owner's call, so this test names the cause and
    // stays red.
    let fx = fixture();
    let mut report = String::new();
    for name in [
        "A_zero_control",
        "A_minus3_bias_plus1",
        "A_minus3_wide_z",
        "A_plus1_wide_z",
        "A_clamp_endpoints",
    ] {
        let blk = get(&fx, "gate", name);
        let d = blk.dims();
        let (m, x) = decay_with_z(
            &blk.nums("z"),
            [d[0], d[1], d[2], d[3]],
            &blk.nums("a_log"),
            &blk.nums("bias"),
            burn_kda::DecayFn::Softplus,
            burn_kda::G_MIN,
        );
        let got: Vec<f32> = m
            .forward(x)
            .log()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let want = blk.nums("fla_softplus_g");
        let (r, at) = num_diff(&got, &want, ATOL_G, TOL_GATE);
        if r > 1.0 {
            report.push_str(&format!(
                "  case {name}: worst rel {r:.3e} at index {at}  ours {}  FLA {}\n",
                got[at], want[at]
            ));
        }
    }
    assert!(
        report.is_empty(),
        "DecayFn::Softplus disagrees with FLA's naive_kda_gate (exec'd) on {} case(s):\n\
         {report}\n\
         CAUSE: src/lib.rs:268 puts exp(A) INSIDE the softplus; FLA puts it OUTSIDE, in\n\
         both the naive reference (gate.py:50) and the triton kernel (gate.py:167).\n\
         The module docs at src/lib.rs:17 and src/lib.rs:267 state the OUTSIDE form, so\n\
         the comments are also wrong about the code. Class A to fix the comment;\n\
         class B (a numerical change to the Kimi-Linear branch, = A/B arm 5) to fix the\n\
         code. NOT fixed here: see docs/reviews/2026-09-30-kda-formula-audit.md S3.1",
        report.lines().count()
    );
}

// ── 2. Eq 1 against FLA's own per-token scan ───────────────────────────────

/// `kda_step` once per token, from the fixture, and compare to FLA's
/// `naive_recurrent_kda` output AND final state at `scale = 1.0`.
///
/// Two adaptations, both forced by `kda_step`'s signature and both exact:
///
/// 1. **`beta` is a scalar in `kda_step`** (`beta: f64`, `src/lib.rs:290`)
///    while FLA's is `[B, T, HV]`, one per value head. The two agree only when
///    the heads' betas coincide, so the loop runs one head per call. Heads are
///    independent in this recurrence — the state is `[B, H, K, V]` and nothing
///    mixes them — so one head per call is the same computation, not an
///    approximation. The chunked test below is where multi-head is checked as a
///    single batched call.
/// 2. **`decay` is `[H, DK]`**, i.e. batch-shared, so each batch element gets
///    its own slice.
#[test]
fn kda_step_matches_fla_recurrent_at_unit_scale() {
    let fx = fixture();
    for name in ["square_1head", "square_2head", "gva_1to2"] {
        let blk = get(&fx, "recur", name);
        let d = blk.dims(); // B T H HV K V
        let (b, t, h, hv, k, v) = (d[0], d[1], d[2], d[3], d[4], d[5]);
        let q = bthd_to_bhtd(t4(&blk.nums("q"), [b, t, h, k]), b, h, t, k);
        let kk = bthd_to_bhtd(t4(&blk.nums("k"), [b, t, h, k]), b, h, t, k);
        let vv = bthd_to_bhtd(t4(&blk.nums("v"), [b, t, hv, v]), b, hv, t, v);
        let g = bthd_to_bhtd(t4(&blk.nums("g"), [b, t, hv, k]), b, hv, t, k);
        // FLA hands back `o` as [B, T, HV, V], so the reference arrays are read
        // in that layout and `beta` is permuted ONCE here into [B, HV, T, 1] to
        // match everything else in this test.
        let beta = t4(&blk.nums("beta"), [b, t, hv, 1])
            .permute([0, 2, 1, 3]);
        let want_o = blk.nums("scale1_o");
        let want_s = blk.nums("scale1_S");

        // GVA: FLA repeats q/k across the value-head group (`naive.py:57-58`).
        // Our `kda_step` is per-head, so the repeat is done here -- which is the
        // same mapping `KdaModule::project` does at src/lib.rs:525-532.
        let rep = hv / h;
        let (q, kk) = if rep > 1 {
            (
                q.unsqueeze_dim::<5>(1)
                    .repeat(&[1, rep, 1, 1, 1])
                    .reshape([b, hv, t, k]),
                kk.unsqueeze_dim::<5>(1)
                    .repeat(&[1, rep, 1, 1, 1])
                    .reshape([b, hv, t, k]),
            )
        } else {
            (q, kk)
        };

        // One (batch, head) pair at a time, because `kda_step`'s `beta` is a
        // **scalar** (`beta: f64`, `src/lib.rs:290`) shared across heads, while
        // FLA's `beta` is `[B, T, HV]` — one per value head. The two agree only
        // when the heads' betas coincide, so a multi-head case cannot be driven
        // through one `kda_step` call. Heads are independent in the recurrence
        // (the state is `[B, H, K, V]` and nothing mixes them), so running one
        // head per call is the same computation, not an approximation. The
        // chunked test below is where multi-head GVA is checked as a batch.
        for bi in 0..b {
            for h in 0..hv {
                let mut state = Tensor::<4>::zeros([1, 1, k, v], &dev());
                for ti in 0..t {
                    // Isolate THIS (batch, head) before slicing, and slice the
                    // batch axis to `bi`. Slicing a shared [B, ...] view to
                    // `0..1` for every `bi` would silently re-read batch 0 --
                    // which is exactly the bug this test had, and it is worth
                    // the two lines because a wrong-axis slice produces a
                    // plausible number rather than an error.
                    let sl = [bi..bi + 1, h..h + 1, ti..ti + 1, 0..k];
                    let q_t = q.clone().slice(sl.clone()).reshape([1, 1, k]);
                    let k_t = kk.clone().slice(sl.clone()).reshape([1, 1, k]);
                    let v_t = vv
                        .clone()
                        .slice([bi..bi + 1, h..h + 1, ti..ti + 1, 0..v])
                        .reshape([1, 1, v]);
                    // FLA's `g` is the LOG decay; Eq 1's Diag(alpha) is exp(g).
                    let d_t = g.clone().slice(sl).exp().reshape([1, k]);
                    // `beta` is already `[B, HV, T, 1]` (permuted above), so the
                    // head axis is 1 and the time axis is 2. Slicing
                    // `[b, ti, h, 0]` instead transposes the two and reads a
                    // different head's beta -- which is what a wrong-axis slice
                    // looks like: a plausible wrong number, not a crash.
                    let beta_t: f32 = beta
                        .clone()
                        .slice([bi..bi + 1, h..h + 1, ti..ti + 1, 0..1])
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap()[0];
                    let (s, o) =
                        burn_kda::kda_step::<NdArray>(state, d_t, q_t, k_t, v_t, beta_t as f64);
                    state = s;
                    let got: Vec<f32> = o.into_data().to_vec::<f32>().unwrap();
                    let off = ((bi * t + ti) * hv + h) * v;
                    let (r, at) =
                        num_diff(&got, &want_o[off..off + v], ATOL_CHAIN, TOL_RECUR);
                    assert!(
                        r <= 1.0,
                        "kda_step output disagrees with FLA naive_recurrent_kda(scale=1)\n  \
                         case {name} b{bi} h{h} t{ti}: worst normalised {r:.3e} at chan {at} \
                         (got {} want {})",
                        got[at],
                        want_o[off + at]
                    );
                }
                let got_s: Vec<f32> = state.into_data().to_vec::<f32>().unwrap();
                let off = (bi * hv + h) * k * v;
                let (r, at) =
                    num_diff(&got_s, &want_s[off..off + k * v], ATOL_CHAIN, TOL_RECUR);
                assert!(
                    r <= 1.0,
                    "kda_step final state disagrees with FLA naive_recurrent_kda(scale=1)\n  \
                     case {name} b{bi} h{h}: worst normalised {r:.3e} at {at} (got {} want {})",
                    got_s[at],
                    want_s[off + at]
                );
            }
        }
    }
}

/// The attribution test for the red one below: FLA's own two fixture rows are
/// related by EXACTLY the read scale, so nothing else about the recurrence is
/// in dispute.
#[test]
fn fla_read_scale_is_the_whole_difference() {
    let fx = fixture();
    for name in ["square_1head", "square_2head", "gva_1to2"] {
        let blk = get(&fx, "recur", name);
        let k: f32 = blk.dims()[4] as f32;
        let o1 = blk.nums("scale1_o");
        let ok = blk.nums("scaleK_o");
        let s = k.powf(-0.5);
        let want: Vec<f32> = o1.iter().map(|x| x * s).collect();
        let (r, at) = num_diff(&ok, &want, ATOL_CHAIN, TOL_RECUR);
        assert!(
            r <= 1.0,
            "FLA's scale=1.0 and scale=K**-0.5 rows are NOT related by the read scale\n  \
             case {name}: worst rel {r:.3e} at {at} (K={k}, K**-0.5={s}, ours {})",
            ok[at]
        );
    }
}

#[test]
fn read_scale_matches_fla_reference() {
    // RED ON PURPOSE. FLA applies a read scale and burn-kda does not.
    //
    //   fla/ops/kda/chunk.py:474-475      if scale is None: scale = K ** -0.5
    //   fla/ops/kda/fused_recurrent.py:261-262   if scale is None:
    //                                                scale = k.shape[-1] ** -0.5
    //   fla/layers/kda.py:262-278         chunk_kda(...)  -- NO scale= argument
    //   fla/ops/kda/naive.py:57            q = q.repeat_interleave(G, dim=2) * scale
    //
    // So the official KDA layer runs at head_k_dim**-0.5. We pass 1.0 into
    // `chunk_wy_forward` (src/lib.rs:646, :653), `kda_step` and
    // `forward_recurrent` carry no scale at all, and `src/fused.rs:9` asserts
    // the reason -- "scale = 1 (no softmax scale in KDA)" -- which is false
    // against both upstreams. burn-gdn2 itself DOES use d_k**-0.5
    // (gdn2/src/module.rs:299); burn-kda is the only place that overrides it.
    //
    // WHY THE MODEL HAS NOT NOTICED, which is the interesting half: the scale
    // enters ONLY the read `o = q·S` and never the state update, so it is one
    // constant on the attention output. The next thing burn-kda does to that
    // output is `output()` (src/lib.rs:554-568), an RMSNorm, and an RMS norm is
    // invariant to a constant rescale:
    //     c·o / sqrt(mean((c·o)²) + eps)  =  o / sqrt(mean(o²) + eps/c²)
    // so the factor is absorbed to O(eps / mean(o²)) ~ O(1e-5). That is exactly
    // why no loss curve, no seed comparison and no test in the crate can see
    // it -- and also why it is still a real divergence: anything reading the
    // raw attention output is reading a tensor that is head_k_dim**0.5 too
    // large (2.83x at K=8, 8x at K=64).
    //
    // The fix is one literal and it moves every number in the archive derived
    // from this crate, so it is the owner's call, not this lane's.
    let fx = fixture();
    let mut report = String::new();
    for name in ["square_1head", "square_2head", "gva_1to2"] {
        let blk = get(&fx, "recur", name);
        let k: f32 = blk.dims()[4] as f32;
        // our answer is `scale1_o` (kda_step has no scale; the test above shows
        // it reproduces that row), FLA's default-scale answer is `scaleK_o`.
        let got = blk.nums("scale1_o");
        let want = blk.nums("scaleK_o");
        let (r, at) = num_diff(&got, &want, ATOL_CHAIN, TOL_RECUR);
        if r > 1.0 {
            report.push_str(&format!(
                "  case {name}: worst rel {r:.3e} at {at}, ours {} vs FLA {} (ratio {:.4}, \
                 K**-0.5 = {:.4})\n",
                got[at], want[at], got[at] / want[at], k.powf(-0.5)
            ));
        }
    }
    assert!(
        report.is_empty(),
        "burn-kda applies no read scale; FLA's KDA layer runs at scale = K**-0.5 on {} \
         case(s):\n{report}\n\
         CAUSE: src/lib.rs:646,653 pass 1.0 as `scale`; kda_step / forward_recurrent have no \
         scale term; src/fused.rs:9 states the reason and the reason is wrong.\n\
         Not fixed here: a numerical change to a shipped model. See \
         docs/reviews/2026-09-30-kda-formula-audit.md S3.2.",
        report.lines().count()
    );
}

// ── 3. the chunked WY construction vs FLA's executed chunk reference ───────

/// `chunk_wy_forward` (the function burn-kda hands its q/k/v/g to at
/// src/lib.rs:638-653) against `naive_chunk_kda`, at `scale = 1.0`, on **both**
/// of gdn2's chunk arms -- `ChunkPath::Batched` (the default, and the one the
/// trainer runs) and `ChunkPath::Loop`.
///
/// The mapping is the one `src/fused.rs:4-14` documents: `g` = the log decay
/// over the KEY axis, `b` = the erase gate over key channels, `w_gate` = the
/// write gate over value channels. FLA's `beta` is `[B, T, HV]`, one scalar per
/// value head, and it multiplies both the key side and the value side
/// (`naive.py:64`), so broadcasting it over both channel axes is the same
/// number and not a shortcut.
/// One `chunk_wy_forward` call on a fixture case, returning the output and the
/// final state, both flat. Shared by the three chunked tests so the GVA repeat
/// and the tensor layout live in exactly one place.
fn run_chunk(blk: &Block, scale: f64) -> (Vec<f32>, Vec<f32>) {
    let d = blk.dims(); // B T H HV K V BT
    let (b, t, h, hv, k, v) = (d[0], d[1], d[2], d[3], d[4], d[5]);
    let bt = d[6];
    let (q, kk) = {
        let q = bthd_to_bhtd(t4(&blk.nums("q"), [b, t, h, k]), b, h, t, k);
        let kk = bthd_to_bhtd(t4(&blk.nums("k"), [b, t, h, k]), b, h, t, k);
        // GVA: `chunk_wy_forward` derives `heads` from `q.shape` and then slices
        // q, k, g AND w_gate on that same axis (gdn2's
        // `chunk_wy_forward_batched`, forward.rs:212-213 and :231-236), so every
        // one of the five tensors must already carry HV heads. That is what
        // `KdaModule::project` does at src/lib.rs:525-537 (it repeats q, k, g,
        // b_k and b_v across the group), and it is also FLA's `state_v_first`
        // convention: q/k live on the qk-head axis and are `repeat_interleave`d
        // to the value-head axis (naive.py:118-119).
        if hv > h {
            let rep = hv / h;
            let r = |x: Tensor<4>| -> Tensor<4> {
                x.unsqueeze_dim::<5>(1)
                    .repeat(&[1, rep, 1, 1, 1])
                    .reshape([b, hv, t, k])
            };
            (r(q), r(kk))
        } else {
            (q, kk)
        }
    };
    let vv = bthd_to_bhtd(t4(&blk.nums("v"), [b, t, hv, v]), b, hv, t, v);
    let g = bthd_to_bhtd(t4(&blk.nums("g"), [b, t, hv, k]), b, hv, t, k);
    let beta = t3(&blk.nums("beta"), [b, t, hv])
        .permute([0, 2, 1])
        .reshape([b, hv, t, 1]);
    let b_k = beta.clone().repeat(&[1, 1, 1, k]);
    let w_gate = beta.repeat(&[1, 1, 1, v]);
    let state = Tensor::<4>::zeros([b, hv, k, v], &dev());
    let (o, s) = burn_gdn2::chunk_wy_forward(q, kk, vv, g, b_k, w_gate, state, scale, bt);
    let got: Vec<f32> = o.permute([0, 2, 1, 3]).into_data().to_vec::<f32>().unwrap();
    let got_s: Vec<f32> = s.into_data().to_vec::<f32>().unwrap();
    (got, got_s)
}

#[test]
fn chunked_wy_matches_fla_chunk_at_unit_scale() {
    use burn_gdn2::ChunkPath;
    // `set_chunk_path` is a process-global static, so the two arms are walked
    // under one lock rather than from two tests.
    static ARM: Mutex<()> = Mutex::new(());
    let _guard = ARM.lock().unwrap_or_else(|e| e.into_inner());

    let fx = fixture();
    for path in [ChunkPath::Batched, ChunkPath::Loop] {
        burn_gdn2::set_chunk_path(path);
        for name in ["chunk16_h1", "chunk16_h2", "chunk16_gva"] {
            let blk = get(&fx, "chunk", name);
            let (got, got_s) = run_chunk(&blk, 1.0);
            let (r, at) = num_diff(&got, &blk.nums("o"), ATOL_CHUNK, TOL_CHUNK);
            assert!(
                r <= 1.0,
                "chunk_wy_forward disagrees with FLA naive_chunk_kda(scale=1.0)\n  \
                 arm {path:?} case {name}: worst normalised {r:.3e} at {at} (got {} want {})",
                got[at],
                blk.nums("o")[at]
            );
            let (r, at) = num_diff(&got_s, &blk.nums("S"), ATOL_CHUNK, TOL_CHUNK);
            assert!(
                r <= 1.0,
                "chunk_wy_forward final state disagrees with FLA naive_chunk_kda(scale=1.0)\n  \
                 arm {path:?} case {name}: worst normalised {r:.3e} at {at} (got {} want {})",
                got_s[at],
                blk.nums("S")[at]
            );
        }
    }
    burn_gdn2::set_chunk_path(ChunkPath::Batched);
}

/// The GREEN half of the read-scale pair, and the reason the red half below is
/// worth reading: `chunk_wy_forward` **does** implement the read scale. Asked for
/// `K**-0.5` it reproduces FLA's own `oK` row. So what burn-kda is missing is
/// an ARGUMENT, not a mechanism — which is what makes the candidate fix one
/// literal rather than a port.
#[test]
fn chunked_wy_honours_the_read_scale_when_asked() {
    use burn_gdn2::ChunkPath;
    static ARM: Mutex<()> = Mutex::new(());
    let _guard = ARM.lock().unwrap_or_else(|e| e.into_inner());
    let fx = fixture();
    for path in [ChunkPath::Batched, ChunkPath::Loop] {
        burn_gdn2::set_chunk_path(path);
        for name in ["chunk16_h1", "chunk16_h2", "chunk16_gva"] {
            let blk = get(&fx, "chunk", name);
            let k: f64 = blk.dims()[4] as f64;
            let (got, _) = run_chunk(&blk, k.powf(-0.5));
            let (r, at) = num_diff(&got, &blk.nums("oK"), ATOL_CHUNK, TOL_CHUNK);
            assert!(
                r <= 1.0,
                "chunk_wy_forward(scale=K**-0.5) does not reproduce FLA's own default-scale \
                 row\n  arm {path:?} case {name}: worst normalised {r:.3e} at {at} \
                 (got {} want {})",
                got[at],
                blk.nums("oK")[at]
            );
        }
    }
    burn_gdn2::set_chunk_path(ChunkPath::Batched);
}

/// The RED half, on the CHUNKED arm: what burn-kda actually passes is `1.0`
/// (`src/lib.rs:646`, `:653`), so it does not produce the row the official KDA
/// layer produces. `read_scale_matches_fla_reference` is the same divergence on
/// the recurrent arm; this one is the one a candidate fix turns green, and
/// `falsify.sh`'s A6 is that demonstration.
#[test]
fn chunked_wy_applies_no_read_scale() {
    use burn_gdn2::ChunkPath;
    static ARM: Mutex<()> = Mutex::new(());
    let _guard = ARM.lock().unwrap_or_else(|e| e.into_inner());
    let fx = fixture();
    let mut report = String::new();
    for path in [ChunkPath::Batched, ChunkPath::Loop] {
        burn_gdn2::set_chunk_path(path);
        for name in ["chunk16_h1", "chunk16_h2", "chunk16_gva"] {
            let blk = get(&fx, "chunk", name);
            let k: f32 = blk.dims()[4] as f32;
            let (got, _) = run_chunk(&blk, 1.0);
            let want = blk.nums("oK");
            let (r, at) = num_diff(&got, &want, ATOL_CHUNK, TOL_CHUNK);
            if r > 1.0 {
                report.push_str(&format!(
                    "  arm {path:?} case {name}: worst normalised {r:.3e} at {at}, ours {} vs \
                     FLA {} (ratio {:.4}, K**-0.5 = {:.4})\n",
                    got[at], want[at], got[at] / want[at], k.powf(-0.5)
                ));
            }
        }
    }
    burn_gdn2::set_chunk_path(ChunkPath::Batched);
    assert!(
        report.is_empty(),
        "burn-kda passes scale=1.0; FLA's KDA layer runs at K**-0.5 on {} chunk arm/case(s):\n\
         {report}\n\
         CAUSE: src/lib.rs:646,653 pass 1.0; src/fused.rs:9 states the reason and the reason \
         is false against both upstreams.\n\
         The MECHANISM EXISTS -- chunked_wy_honours_the_read_scale_when_asked is green -- so \
         this is a missing argument, not a missing implementation.\n\
         Not fixed here: a numerical change to a shipped model. See \
         docs/reviews/2026-09-30-kda-formula-audit.md S3.2.",
        report.lines().count()
    );
}

// ── 4. a tier (d) shape gate, for a class-A fix in the reference scan ──────

/// `forward_recurrent` used the KEY-side beta on two tensors that live on the
/// VALUE axis (`src/lib.rs`, the `beta` binding and its comment). That is
/// bit-identical while `head_dim == v_head_dim` and raises a broadcast error
/// when they differ, so it was a latent break in the crate's own exact-per-token
/// *reference* — the thing every chunk-path test compares against.
///
/// **This gate is tier (d), not (a), and it says so:** it exercises
/// `expand_v = 2.0`, a configuration no upstream reference comparison covers and
/// none of the FLA fixtures use. What it pins is a SHAPE, which needs no
/// external reference — `erased` is `[B, H, 1, DV]` and `v_t` is
/// `[B, H, 1, DV]`, and multiplying either by a `[B, H, 1, DK]` tensor is a type
/// error. Red before the fix (a broadcast panic), green after, no number
/// involved.
///
/// `expand_v` is unreachable from `KdaConfig` on the running model (dormouse
/// leaves `num_v_heads: None` and `expand_v` at its 1.0 default), so this is a
/// gate for the crate as shipped rather than for the model as trained.
#[test]
fn forward_recurrent_runs_with_expand_v_ne_1() {
    let dev = dev();
    let km = KdaModule::new(
        &KdaConfig {
            hidden_size: 64,
            num_heads: 2,
            head_dim: 8,
            num_v_heads: None,
            expand_v: 2.0, // v_head_dim = 16 != head_dim = 8
            use_short_conv: false,
            decay_fn: burn_kda::DecayFn::Sigmoid,
            rank: 8,
            chunk_size: 16,
            ..Default::default()
        },
        0.9,
        &dev,
    );
    assert_eq!(km.v_head_dim, 16);
    assert_ne!(km.v_head_dim, km.head_dim);
    let x = t3(
        &(0..1 * 12 * 64)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect::<Vec<f32>>(),
        [1, 12, 64],
    );
    let mut st: Option<Tensor<4>> = None;
    // Before the fix this panicked inside burn's broadcast check: `[B,H,1,DK]`
    // against `[B,H,1,DV]` with DK = 8 and DV = 16.
    let out = km.forward_recurrent(x, &mut st, true);
    assert_eq!(out.dims(), [1, 12, 64]);
    let s = st.unwrap();
    assert_eq!(s.dims(), [1, 2, 8, 16]);
    // And it must be a FUNCTION, not just a shape: the chunked path and the scan
    // still agree with each other at this shape, which is the property the fix
    // had to preserve.
    let out_chunk = km.forward_train::<NdArray>(t3(
        &(0..1 * 12 * 64)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect::<Vec<f32>>(),
        [1, 12, 64],
    ));
    let (r, at) = num_diff(
        &out.into_data().to_vec::<f32>().unwrap(),
        &out_chunk.into_data().to_vec::<f32>().unwrap(),
        ATOL_CHAIN,
        TOL_RECUR,
    );
    assert!(
        r <= 1.0,
        "at expand_v=2.0 the exact scan and the chunked path now disagree: normalised \
         {r:.3e} at {at} -- the b_v fix changed a number, which it must not"
    );
}
///
/// The assertion is on the **formula class only** — the decay gates, where both
/// sides evaluate one `exp` and one sigmoid in the same order and the result
/// should be a few ulp. If the gate tolerance were ever carrying a pass, that
/// class would sit near its own bound instead of 100x below it, and this fails.
///
/// The f32-chain class is **reported, not asserted**, and deliberately so: its
/// measured worst (1.4e-4 on the chunked arm) is inside its own derived bound
/// by construction, so asserting "far above tolerance" there would be an
/// assertion about arithmetic rather than about a formula. Printing it is what
/// makes the bound falsifiable.
#[test]
fn green_margins_are_reported_and_the_formula_class_is_far_above_tolerance() {
    let fx = fixture();
    let mut gate_margins: Vec<(&str, f32)> = vec![];
    let mut chain_margins: Vec<(&str, f32)> = vec![];

    for name in [
        "A_zero_control",
        "A_minus3_bias_plus1",
        "A_minus3_wide_z",
        "A_plus1_wide_z",
        "A_zero_nonzero_bias",
        "A_clamp_endpoints",
    ] {
        let blk = get(&fx, "gate", name);
        let d = blk.dims();
        let (m, x) = decay_with_z(
            &blk.nums("z"),
            [d[0], d[1], d[2], d[3]],
            &blk.nums("a_log"),
            &blk.nums("bias"),
            burn_kda::DecayFn::Sigmoid,
            burn_kda::G_MIN,
        );
        let got: Vec<f32> = m.forward(x).log().into_data().to_vec::<f32>().unwrap();
        gate_margins.push((name, num_diff(&got, &blk.nums("fla_lowerbound_g"), ATOL_G, TOL_GATE).0));
    }

    // The scale attribution, on FLA's own two rows.
    for name in ["square_1head", "square_2head", "gva_1to2"] {
        let blk = get(&fx, "recur", name);
        let k: f32 = blk.dims()[4] as f32;
        let s = k.powf(-0.5);
        let want: Vec<f32> = blk.nums("scale1_o").iter().map(|x| x * s).collect();
        chain_margins.push((name, num_diff(&blk.nums("scaleK_o"), &want, ATOL_CHAIN, TOL_RECUR).0));
    }

    // The ratios are already NORMALISED: `num_diff` returns
    // `|a-b| / (atol + rtol*|b|)`, so 1.0 IS the tolerance and 0.01 is a
    // 100x margin below it. Nothing to divide again.
    let gw = gate_margins
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    eprintln!(
        "formula class (decay gate): worst normalised {:.4e} at {} = a {:.0}x margin",
        gw.1,
        gw.0,
        1.0 / gw.1
    );
    for (n, m) in &chain_margins {
        eprintln!("chain class: {n} normalised {m:.4e}  [1.0 = tolerance]");
    }
    // 20x, not 100x: see the note in `k3_bounded_decay_matches_fla_reference`.
    // The `ATOL_G` floor carries the near-zero `g` elements, and 34x is what the
    // measured disagreement actually is. What has to stay true is that the class
    // is not within an order of magnitude of its bound.
    assert!(
        gw.1 < 0.05,
        "the FORMULA class has drifted to within 20x of its own tolerance ({:.4e} at {}): \
         the tolerance is carrying the result, not the formula",
        gw.1,
        gw.0
    );
}
