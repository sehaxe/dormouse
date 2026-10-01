//! Assert-based self-checks (burn-ndarray + autodiff).
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
use burn::lr_scheduler::module_lr_scheduler::ModuleLearningRate;
use burn::nn::LinearConfig;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::Device;
use burn::tensor::{Distribution, Tensor};
use burn_muon_plus::{MuonPlus, MuonPlusConfig, NormDir};

fn dev() -> Device {
    Device::ndarray().autodiff()
}

/// Train a Linear(8,4) to map x -> y. Muon+ must drive loss down.
#[test]
fn muon_plus_converges_2d_quadratic() {
    let device = dev();
    // square Linear(4,4) without bias: same shape family as the hybrid test
    // below, which converges reliably on random data; the old Linear(8,4)
    // variant diverged on some random seeds (pre-existing flakiness)
    let mut linear = LinearConfig::new(4, 4).with_bias(false).init(&device);
    let x = Tensor::<2>::random([16, 4], Distribution::Default, &device);
    let y = Tensor::<2>::random([16, 4], Distribution::Default, &device);

    let config = MuonPlusConfig::new()
        .with_norm_dir(Some(NormDir::ColRow))
        .with_weight_decay(0.0);
    let mut optimizer = config.init();

    let mut loss_val = f32::INFINITY;
    let mut first = f32::INFINITY;
    for _ in 0..400 {
        let out = linear.forward(x.clone());
        let loss = (out - y.clone()).powf_scalar(2.0).mean();
        loss_val = loss.clone().into_scalar::<f32>();
        if first.is_infinite() {
            first = loss_val;
        }
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        linear = optimizer.step(ModuleLearningRate::from(0.03f64), linear, grads);
    }
    assert!(
        loss_val < first * 0.5,
        "Muon+ should cut ||Wx-y||² by 5x in 200 steps: {first} -> {loss_val}"
    );
}

/// Linear with bias: 2D weight gets Muon+, 1D bias gets AdamW. Must converge.
#[test]
fn muon_plus_hybrid_bias_adamw_converges() {
    let device = dev();
    let mut linear = LinearConfig::new(4, 4).with_bias(true).init(&device);
    let x = Tensor::<2>::random([16, 4], Distribution::Default, &device);
    let y = Tensor::<2>::random([16, 4], Distribution::Default, &device);

    let mut optimizer = MuonPlusConfig::new()
        .with_norm_dir(Some(NormDir::ColRow))
        .with_weight_decay(0.0)
        .init();

    let mut loss_val = f32::INFINITY;
    let mut first = f32::INFINITY;
    for _ in 0..400 {
        let out = linear.forward(x.clone());
        let loss = (out - y.clone()).powf_scalar(2.0).mean();
        loss_val = loss.clone().into_scalar::<f32>();
        if first.is_infinite() {
            first = loss_val;
        }
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        linear = optimizer.step(ModuleLearningRate::from(0.03f64), linear, grads);
    }
    assert!(
        loss_val < first * 0.5,
        "hybrid should cut loss by 2x in 200 steps: {first} -> {loss_val}"
    );
}

/// Newton-Schulz output must be near-orthogonal (X^T X ≈ I for wide m×n).
#[test]
fn orthogonalize_near_orthogonal() {
    let device = dev();
    let g = Tensor::<2>::random([16, 8], Distribution::Default, &device);
    let muon: MuonPlus = MuonPlusConfig::new().build();
    let q = muon.orthogonalize(g);
    let diff = q.clone().transpose().matmul(q.clone()) - Tensor::<2>::eye(8, &device);
    let err = diff.powf_scalar(2.0).sum().sqrt().into_scalar::<f32>();
    assert!(
        err < 2.0,
        "X^T X should be roughly orthogonal, got error {err}"
    );
}

/// Tall matrices (rows > cols) must preserve shape through orthogonalization.
#[test]
fn orthogonalize_preserves_shape_tall() {
    let device = dev();
    let g = Tensor::<2>::random([32, 8], Distribution::Default, &device);
    let muon: MuonPlus = MuonPlusConfig::new().build();
    let q = muon.orthogonalize(g);
    assert_eq!(q.dims(), [32, 8]);
}

/// ColRow normalization: after col-then-row, rows have unit L2 norm
/// (the last applied direction wins).
#[test]
fn normalize_colrow_rows_unit() {
    let device = dev();
    let x = Tensor::<2>::random([16, 8], Distribution::Default, &device);
    let muon: MuonPlus = MuonPlusConfig::new()
        .with_norm_dir(Some(NormDir::ColRow))
        .build();

    let normalized = muon.normalize(x);
    let m = normalized.dims()[0];

    // Row norms: sum over cols (dim 1).
    let row_norms = normalized.powf_scalar(2.0).sum_dim(1).sqrt();
    let row_vals: Vec<f32> = row_norms.into_data().try_to_vec().unwrap();
    assert_eq!(row_vals.len(), m);
    for v in row_vals {
        assert!((v - 1.0).abs() < 1e-3, "row norm should be 1, got {v}");
    }
}

/// Col-only normalization: columns unit.
#[test]
fn normalize_col_only_units_columns() {
    let device = dev();
    let x = Tensor::<2>::random([16, 8], Distribution::Default, &device);
    let muon: MuonPlus = MuonPlusConfig::new()
        .with_norm_dir(Some(NormDir::Col))
        .build();
    let normalized = muon.normalize(x);

    let col_norms = normalized.clone().powf_scalar(2.0).sum_dim(0).sqrt();
    for v in col_norms.into_data().try_to_vec::<f32>().unwrap() {
        assert!((v - 1.0).abs() < 1e-3, "column norm should be 1, got {v}");
    }
    let row_norms = normalized.powf_scalar(2.0).sum_dim(1).sqrt();
    for v in row_norms.into_data().try_to_vec::<f32>().unwrap() {
        assert!(
            (v - 1.0).abs() > 1e-3,
            "rows should NOT be unit after col-only, got {v}"
        );
    }
}

// ── Conformance to 2602.21545 §2.3, Eqs. (3)-(8) ────────────────────────────
//
// The references below are a LITERAL scalar transcription of the paper's
// equations, with no tensor library in them, so the test cannot inherit a bug
// from the implementation's own axis conventions or broadcasting. The
// implementations use a `clamp_min` floor where App. C's `norm_dir` puts
// `+eps` inside the root; for a column of norm `v` the two divide by
// `max(v, 1e-7)` and `sqrt(v² + 1e-8)`, which agree to ~1e-9 relative at the
// scale a Newton-Schulz output has (its singular values are driven to ~1).
// The tolerance below is four orders of magnitude above that and four below
// the operator's own scale, so it pins the ORDER and the AXIS and not the
// epsilon.

const M: usize = 16;
const N: usize = 8;
/// App. C's `eps` default.
const EPS: f32 = 1e-8;

/// Deterministic, so a failure names the same numbers every run.
fn grid() -> Tensor<2> {
    let vals: Vec<f32> = (0..M * N)
        .map(|i| ((i.wrapping_mul(2654435761) % 997) as f32 / 498.0) - 1.0)
        .collect();
    Tensor::<2>::from_data(burn::tensor::TensorData::new(vals, [M, N]), &dev())
}

fn flat(t: &Tensor<2>) -> Vec<f32> {
    t.clone().into_data().as_slice::<f32>().unwrap().to_vec()
}

/// Eq. (3)-(4): `X·D_col⁻¹`, `D_col := diag(sqrt(Σ_i x_ij²))`, eps inside.
fn paper_norm_col(x: &[f32]) -> Vec<f32> {
    let mut out = x.to_vec();
    for j in 0..N {
        let s: f32 = (0..M).map(|i| x[i * N + j] * x[i * N + j]).sum();
        let d = (s + EPS).sqrt();
        for i in 0..M {
            out[i * N + j] /= d;
        }
    }
    out
}

/// Eq. (5)-(6): `D_row⁻¹·X`, `D_row := diag(sqrt(Σ_j x_ij²))`, eps inside.
fn paper_norm_row(x: &[f32]) -> Vec<f32> {
    let mut out = x.to_vec();
    for i in 0..M {
        let s: f32 = (0..N).map(|j| x[i * N + j] * x[i * N + j]).sum();
        let d = (s + EPS).sqrt();
        for j in 0..N {
            out[i * N + j] /= d;
        }
    }
    out
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Every direction, against the equation it claims to implement — order
/// included. `ColRow` and `RowCol` are different functions and the paper
/// names which is which (Eq. (7) col-then-row, Eq. (8) row-then-col), so a
/// swap is a silent loss of a different update, not a cosmetic reorder.
#[test]
fn normalization_matches_the_paper_equations() {
    let x = grid();
    let raw = flat(&x);
    let col = paper_norm_col(&raw);
    let row = paper_norm_row(&raw);
    let eq7_col_then_row = paper_norm_row(&col);
    let eq8_row_then_col = paper_norm_col(&row);
    let norm = |d| MuonPlusConfig::new().with_norm_dir(Some(d)).build();

    for (dir, want, what) in [
        (NormDir::Col, &col, "Eq. (3)-(4) Norm_(col)"),
        (NormDir::Row, &row, "Eq. (5)-(6) Norm_(row)"),
        (NormDir::ColRow, &eq7_col_then_row, "Eq. (7) Norm_(col_row)"),
        (NormDir::RowCol, &eq8_row_then_col, "Eq. (8) Norm_(row_col)"),
    ] {
        let got = flat(&norm(dir).normalize(x.clone()));
        let d = maxdiff(&got, want);
        assert!(d < 1e-5, "{what}: max|ours - paper| = {d:e}");
    }

    // Without this, the four assertions above would also pass if `ColRow` and
    // `RowCol` were the same function and the input happened to make the two
    // orders agree. They do not, and that is what pins the composition ORDER.
    let cr = flat(&norm(NormDir::ColRow).normalize(x.clone()));
    let rc = flat(&norm(NormDir::RowCol).normalize(x.clone()));
    let d = maxdiff(&cr, &rc);
    assert!(
        d > 1e-3,
        "ColRow and RowCol differ by only {d:e} on this input, so the \
         composition order is not actually pinned by the checks above"
    );
    assert!(
        maxdiff(&cr, &eq8_row_then_col) > 1e-3,
        "ColRow matched the paper's ROW-then-COL: Eq. (7) and Eq. (8) are swapped"
    );
}

/// `norm_dir: None` is plain Muon — the BASELINE the paper measures Muon+
/// against, not Muon+. Eq. (4) has no such branch; the paper's function
/// default is the narrower `Norm_(col)` (App. C, `d="col"`). This pins the
/// crate default's meaning so the config doc and the code cannot drift.
#[test]
fn norm_dir_none_is_plain_muon() {
    let x = grid();
    let muon = MuonPlusConfig::new().build();
    assert_eq!(
        MuonPlusConfig::new().norm_dir,
        None,
        "the crate default must stay 'no normalization', i.e. plain Muon"
    );
    let got = flat(&muon.normalize(x.clone()));
    assert_eq!(
        got,
        flat(&x),
        "norm_dir = None must return the update unchanged (plain Muon)"
    );
    // ...and it must actually differ from Muon+, or "unchanged" is a claim
    // about an operator that does nothing anyway.
    let plus = flat(
        &MuonPlusConfig::new()
            .with_norm_dir(Some(NormDir::ColRow))
            .build()
            .normalize(x),
    );
    assert!(
        maxdiff(&got, &plus) > 1e-3,
        "None and ColRow are the same map"
    );
}

/// Weight decay must shrink weights toward zero on zero-gradient steps.
#[test]
fn weight_decay_shrinks() {
    let device = dev();
    let w = Tensor::<2>::ones([4, 4], &device);
    let muon: MuonPlus = MuonPlusConfig::new().with_weight_decay(0.5).build();
    let zero = Tensor::<2>::zeros([4, 4], &device);
    let (updated, _state) = muon.step(0.01, w.clone(), zero, None);
    let data = updated.into_data();
    let v = data.as_slice::<f32>().unwrap();
    assert!(
        v.iter().all(|x| *x < 1.0),
        "decoupled weight decay should shrink weights: {v:?}"
    );
}
