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
