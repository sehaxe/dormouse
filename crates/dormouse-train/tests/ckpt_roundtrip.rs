//! Checkpoint roundtrip through dormouse-train's public save/load API on the
//! CPU backend: `save_ckpt` (burnpack `.bin` container) must reload via
//! `load_ckpt` into a fresh model and optimizer and reproduce identical
//! logits on the same input.
//!
//! `save_ckpt`/`load_ckpt` are backend-agnostic (burnpack bytes + the
//! crate's own device factory), so the contract is testable without CUDA.

use burn::tensor::{Device, Int, Tensor, TensorData};
use dormouse_core::{fnv_hash, DormouseConfig, DormouseModel};
use dormouse_train::{build_optim, load_ckpt, save_ckpt, Backend, TrainCfg};

#[allow(deprecated)] // Device::ndarray is deprecated upstream; the repo still targets it
fn device() -> Device {
    Device::ndarray().autodiff()
}

/// Nano's shape at a debug-build-friendly width (same reasoning as the train
/// crate's own roundtrip test: full-nano init alone is ~a minute on the CPU
/// backend). The seam contract - burnpack bytes out, identical logits in -
/// is width-independent.
fn nano_cfg() -> DormouseConfig {
    dormouse_core::config::load_config(concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs/nano.toml"))
        .expect("configs/nano.toml loads")
}

fn mini_nano() -> DormouseConfig {
    DormouseConfig {
        d_model: 128,
        n_heads: 4,
        head_dim: 32,
        d_ffn: 256,
        rank: 32,
        msa_topk: 4,
        ..nano_cfg()
    }
}

fn hashed_ids(bytes: &[u8], b: usize, t: usize, dev: &Device) -> Tensor<3, Int> {
    let mut v = Vec::with_capacity(b * t * 3);
    for p in 0..t {
        let e = p + 1;
        v.push((fnv_hash(&bytes[e.saturating_sub(3)..e]) % 4096) as i64);
        v.push((fnv_hash(&bytes[e.saturating_sub(5)..e]) % 4096) as i64);
        v.push((fnv_hash(&bytes[e.saturating_sub(8)..e]) % 4096) as i64);
    }
    Tensor::from_data(TensorData::new(v, [b, t, 3]), dev)
}

/// Mini-nano model + fresh Muon+ optimizer: save_ckpt -> load_ckpt into
/// fresh instances -> identical logits, restored step, no NaNs.
#[test]
fn ckpt_roundtrip_identical_logits() {
    let t0 = std::time::Instant::now();
    let dir = std::env::temp_dir().join("dm-ckpt-roundtrip-seam");
    let _ = std::fs::remove_dir_all(&dir);

    let cfg = mini_nano();
    let dev = device();
    let model = DormouseModel::new(&cfg, &dev);
    let optim_cfg = TrainCfg {
        steps: 3,
        ..Default::default()
    };
    let optim = build_optim(&optim_cfg);

    save_ckpt(&dir, "seam", &model, &optim, 7, 5.5).expect("save_ckpt");
    assert!(dir.join("seam.bin").exists(), "container missing");
    let sidecar = std::fs::read_to_string(dir.join("seam.txt")).expect("sidecar");
    assert!(sidecar.contains("step 7"), "sidecar: {sidecar}");

    let mut model2 = DormouseModel::new(&cfg, &dev);
    let mut optim2 = build_optim(&optim_cfg);
    let step = load_ckpt(&dir, "seam", &cfg, &mut model2, &mut optim2).expect("load_ckpt");
    assert_eq!(step, 7);

    // Same input through both: logits must match to fp32 round-trip noise.
    let (b, s) = (1, 128);
    let bytes: Vec<u8> = (0..b * s)
        .map(|i: usize| (i.wrapping_mul(37) % 256) as u8)
        .collect();
    let v: Vec<i64> = bytes.iter().map(|&x| x as i64).collect();
    let x = Tensor::from_data(TensorData::new(v, [b, s]), &dev);
    let h = hashed_ids(&bytes, b, s, &dev);

    let (l1, _rec1, _k1, _aux1) =
        model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, None, None);
    let (l2, _rec2, _k2, _aux2) =
        model2.forward_with_hidden::<Backend>(x, Some(h), None, None, None);

    let v1: Vec<f32> = l1
        .clone()
        .into_data()
        .try_to_vec()
        .expect("logits readable");
    assert!(v1.iter().all(|x| x.is_finite()), "logits non-finite");
    let d = (l1 - l2).abs().max().into_scalar::<f32>();
    println!("ckpt_roundtrip: max |dlogit| = {d:.3e} ({} ms)", t0.elapsed().as_millis());
    assert!(
        d < 1e-5,
        "reloaded model diverges: max |dlogit| = {d:.3e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
