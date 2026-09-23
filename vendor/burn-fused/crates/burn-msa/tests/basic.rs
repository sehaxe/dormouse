// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
use burn::module::Module;
use burn::tensor::{Device, Int, Tensor, TensorData};
use burn_msa::{
    IndexBranch, KlAlignmentLoss, MsaCache, MsaConfig, MsaModule, SparseAttention, TopKSelector,
};
use burn_ndarray::NdArray;
type B = NdArray;
fn dvc() -> Device {
    Device::ndarray()
}

fn cfg_small() -> MsaConfig {
    MsaConfig {
        d_model: 32,
        n_heads_q: 4,
        n_heads_kv: 2,
        d_head: 8,
        d_idx: 8,
        block_size: 4,
        topk: 2,
        force_local_block: false,
        causal: false,
        use_kl_loss: false,
        warmup_steps: 0,
        kl_coeff: 0.0,
        gradient_detach: true,
        use_rope: false,
        rope_base: 10000.0,
        rope_max_seq_len: 1024,
    }
}

#[test]
fn config_validates() {
    assert!(cfg_small().validate().is_ok());
}

#[test]
fn config_rejects_bad_dims() {
    let mut c = cfg_small();
    c.d_model = 0;
    assert!(c.validate().is_err());
    let mut c = cfg_small();
    c.n_heads_q = 3;
    assert!(c.validate().is_err());
}

#[test]
fn config_defaults() {
    let d = MsaConfig::default();
    assert_eq!(d.d_model, 1152);
    assert_eq!(d.n_heads_kv, 1);
    assert_eq!(d.topk, 16);
}

#[test]
fn index_branch_shapes() {
    let ib = IndexBranch::new(&cfg_small(), &dvc());
    let x = Tensor::<3>::random(
        [2, 8, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let (q_idx, k_idx) = ib.forward(x.clone(), x, 2, 8);
    assert_eq!(q_idx.dims(), [2, 2, 8, 8]);
    assert_eq!(k_idx.dims(), [2, 1, 8, 8]);
}

#[test]
fn index_branch_block_scores() {
    let ib = IndexBranch::new(&cfg_small(), &dvc());
    let q = Tensor::<3>::random(
        [1, 4, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let kv = Tensor::<3>::random(
        [1, 10, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let (q_idx, k_idx) = ib.forward(q, kv, 2, 8);
    let s = ib.compute_block_scores(q_idx, k_idx, 4, 3.0f64.sqrt(), false);
    assert_eq!(s.dims(), [1, 2, 4, 3]);
}

#[test]
fn topk_shapes() {
    let sel = TopKSelector::new(3, 4, false);
    let s = Tensor::<4>::random(
        [1, 2, 4, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    assert_eq!(sel.select::<B>(s).dims(), [1, 2, 4, 3]);

    let s = Tensor::<4>::random(
        [1, 2, 2, 5],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    assert_eq!(sel.select::<B>(s).dims(), [1, 2, 2, 3]);
}

#[test]
fn topk_force_local() {
    let sel = TopKSelector::new(3, 4, true);
    let s = Tensor::<4>::random(
        [1, 1, 8, 4],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let idx = sel.select::<B>(s);
    let td = idx.into_data();
    let vals: Vec<i64> = td
        .bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for i in 0..8 {
        let local = (i / 4) as i64;
        assert!(
            vals[i * 3..i * 3 + 3].contains(&local),
            "q_pos={} missing local block {}",
            i,
            local
        );
    }
}

#[test]
fn topk_single_row() {
    let sel = TopKSelector::new(2, 4, true);
    let s = Tensor::<4>::random(
        [1, 1, 1, 4],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let _ = sel.select::<B>(s);
}

#[test]
fn dense_forward() {
    let attn = SparseAttention::new(&cfg_small(), &dvc());
    let x = Tensor::<3>::random(
        [1, 4, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let (q, k, v) = {
        let qp = attn.q_proj.forward(x.clone());
        let kp = attn.k_proj.forward(x.clone());
        let vp = attn.v_proj.forward(x);
        (qp, kp, vp)
    };
    assert_eq!(attn.forward_dense::<B>(q, k, v).dims(), [1, 4, 32]);
}

#[test]
fn sparse_forward() {
    let attn = SparseAttention::new(&cfg_small(), &dvc());
    let q = Tensor::<3>::random(
        [1, 4, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let kv = Tensor::<3>::random(
        [1, 8, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let (qp, kp, vp) = {
        let qp = attn.q_proj.forward(q);
        let kp = attn.k_proj.forward(kv.clone());
        let vp = attn.v_proj.forward(kv);
        (qp, kp, vp)
    };
    let idx = Tensor::<4, Int>::from_data(
        TensorData::new(
            vec![0i64, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 2, 4, 2],
        ),
        &dvc(),
    );
    assert_eq!(attn.forward_sparse::<B>(qp, kp, vp, idx).dims(), [1, 4, 32]);
}

#[test]
fn module_self_attn() {
    let m = MsaModule::new(&cfg_small(), &dvc());
    let x = Tensor::<3>::random(
        [1, 4, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let result = m.forward::<B>(x);
    assert_eq!(result.output.dims(), [1, 4, 32]);
    assert!(result.kl_loss.is_none());
}

#[test]
fn module_cross_attn() {
    let m = MsaModule::new(&cfg_small(), &dvc());
    let q = Tensor::<3>::random(
        [1, 4, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let kv = Tensor::<3>::random(
        [1, 8, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let result = m.forward_cross::<B>(q, kv);
    assert_eq!(result.output.dims(), [1, 4, 32]);
}

#[test]
fn module_dense_forward() {
    let m = MsaModule::new(&cfg_small(), &dvc());
    let x = Tensor::<3>::random(
        [1, 4, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    assert_eq!(m.forward_dense::<B>(x.clone(), x).dims(), [1, 4, 32]);
}

#[test]
fn no_nan_sparse() {
    let m = MsaModule::new(&cfg_small(), &dvc());
    let x = Tensor::<3>::random(
        [1, 6, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let td = m.forward::<B>(x).output.into_data();
    for chunk in td.bytes.chunks_exact(4) {
        assert!(f32::from_le_bytes(chunk.try_into().unwrap()).is_finite());
    }
}

#[test]
fn no_nan_dense() {
    let m = MsaModule::new(&cfg_small(), &dvc());
    let x = Tensor::<3>::random(
        [1, 6, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let td = m.forward_dense::<B>(x.clone(), x).into_data();
    for chunk in td.bytes.chunks_exact(4) {
        assert!(f32::from_le_bytes(chunk.try_into().unwrap()).is_finite());
    }
}

#[test]
fn kl_loss() {
    let loss = KlAlignmentLoss;
    let bs = Tensor::<4>::random(
        [1, 2, 4, 4],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let aw = Tensor::<4>::random(
        [1, 2, 4, 2],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &dvc(),
    );
    let si = Tensor::<4, Int>::from_data(
        TensorData::new(
            vec![0i64, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 2, 4, 2],
        ),
        &dvc(),
    );
    assert_eq!(loss.compute::<B>(bs, aw, si).dims(), [1]);
}

#[test]
fn cache_ops() {
    let cfg = MsaConfig {
        d_model: 32,
        n_heads_q: 2,
        n_heads_kv: 2,
        d_head: 8,
        d_idx: 8,
        block_size: 4,
        topk: 1,
        causal: true,
        use_kl_loss: false,
        ..MsaConfig::default()
    };
    let m = MsaModule::new(&cfg, &dvc());
    let mut cache = MsaCache::new(cfg.clone());
    cache.reset(1);
    let x = Tensor::<3>::ones([1, 3, 32], &dvc());
    let out = cache.step(&m, x, 0);
    assert_eq!(out.dims(), [1, 3, 32]);
    assert_eq!(cache.stream_len(0), 3);
    cache.rollback(&[0]);
    assert_eq!(cache.stream_len(0), 0);
}

#[test]
fn edge_single_token() {
    let m = MsaModule::new(&cfg_small(), &dvc());
    let x = Tensor::<3>::random(
        [1, 1, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let td = m.forward::<B>(x).output.into_data();
    assert!(td.bytes.iter().any(|&b| b != 0));
}

#[test]
fn edge_kv_longer() {
    let m = MsaModule::new(&cfg_small(), &dvc());
    let q = Tensor::<3>::random(
        [1, 2, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let kv = Tensor::<3>::random(
        [1, 16, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    assert_eq!(m.forward_cross::<B>(q, kv).output.dims(), [1, 2, 32]);
}

#[test]
fn edge_q_longer() {
    let m = MsaModule::new(&cfg_small(), &dvc());
    let q = Tensor::<3>::random(
        [1, 16, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let kv = Tensor::<3>::random(
        [1, 2, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    assert_eq!(m.forward_cross::<B>(q, kv).output.dims(), [1, 16, 32]);
}

#[test]
fn edge_multi_batch() {
    let c = cfg_small();
    let m = MsaModule::new(&c, &dvc());
    let q = Tensor::<3>::random(
        [3, 8, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    let kv = Tensor::<3>::random(
        [3, 12, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dvc(),
    );
    assert_eq!(m.forward_cross::<B>(q, kv).output.dims(), [3, 8, 32]);
}

#[test]
fn serialization_roundtrip() {
    let c = cfg_small();
    let m1 = MsaModule::new(&c, &dvc());
    let rec = m1.into_record();
    let _ = MsaModule::new(&c, &dvc()).load_record(rec);
}
