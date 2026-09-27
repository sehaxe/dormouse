//! Split-brain probe (diagnostic, GPU, ignored by default): does the
//! checkpoint contain the trained function? One real batch, two paths:
//! (a) training-path loss via forward_with_hidden + loss
//! (b) inference-path forward + manual next-byte CE + logits health
//! (a) ≈ 0.1 with (b) ≈ 5.5 (uniform) => the inference forward is a
//! different (broken) function than the training math.

#[test]
#[ignore = "GPU diagnostic: cargo test -p dormouse-train --features cuda --lib eval_split_brain -- --ignored --nocapture"]
fn eval_split_brain() {
    use burn::prelude::*;
    use crate::{build_model, build_optim, device, init_pools, load_ckpt, Backend, TrainCfg};

    let ckpt_dir = std::path::PathBuf::from(
        std::env::var("PROBE_CKPT").unwrap_or_else(|_| "/home/sehaxe/core_probe".into()),
    );
    let data = std::env::var("PROBE_DATA")
        .unwrap_or_else(|_| "/home/sehaxe/data_1m/corpus.bin".into());

    let mut cfg = dormouse_core::config::load_config("small").expect("small preset");
    cfg.use_engram = false; // core_probe ran --no-engram
    cfg.ponder_prior = 0.4;

    let device = device();
    init_pools(&device);
    let train_cfg = TrainCfg::default();
    let (mut model, _qfmt) = build_model(&cfg, &train_cfg, &device);
    let mut optim = build_optim(&train_cfg);
    let step = load_ckpt(&ckpt_dir, "core_probe", &cfg, &mut model, &mut optim)
        .expect("core_probe checkpoint must load");
    println!("loaded core_probe at step {step}");

    // one real batch: input = bytes[0..bt], targets = bytes[1..=bt]
    let raw = std::fs::read(&data).expect("read probe data");
    let (b, t) = (10usize, 512usize);
    let bt = b * t;
    assert!(raw.len() >= bt + 1, "data too small");
    let inp: Vec<u64> = raw[..bt].iter().map(|&x| x as u64).collect();
    let tgt: Vec<u64> = raw[1..=bt].iter().map(|&x| x as u64).collect();
    let input_ids =
        Tensor::<2, Int>::from_data(burn::tensor::TensorData::new(inp, [b, t]), &device);
    let targets =
        Tensor::<2, Int>::from_data(burn::tensor::TensorData::new(tgt, [b, t]), &device);

    // (a) training-path loss
    let (_logits_tr, rec, p_dist, _kda, _aux) = model.forward_with_hidden::<Backend>(
        input_ids.clone(),
        None,
        None,
        Some(targets.clone()),
        None,
    );
    let loss_tr: f32 = model.loss::<Backend>(rec.clone(), p_dist.clone()).into_scalar();
    println!("(a) training-path loss          = {loss_tr:.4}");
    let pdm = p_dist.clone().mean_dim(0); // [1, N] mean over batch
    let vals = pdm.into_data().to_vec::<f32>().expect("p_dist f32");
    println!("(a) p_dist mean/iter = {vals:?}  (~0 => lambda collapse)");

    // (b) inference-path logits + manual next-byte CE
    let logits = model.forward::<Backend>(input_ids, None);
    let lf = logits.cast(burn::tensor::FloatDType::F32);
    let [_b2, _t2, v] = lf.shape().dims();
    println!("(b) forward logits shape        = [{_b2},{_t2},{v}]");
    let flat = lf.clone().reshape([bt, v]);
    let lg = burn::tensor::activation::log_softmax(flat.clone(), 1);
    let tg2 = targets.reshape([bt, 1]);
    let picked = lg.gather(1, tg2).reshape([bt]);
    let ce_inf = -picked.clone().mean().into_scalar::<f32>();
    println!("(b) inference-path next-byte CE = {ce_inf:.4}");
    let first = lf.slice([0..1, 0..1]).reshape([v]);
    let mu: f32 = first.clone().mean().into_scalar();
    let sd: f32 = (first.clone() - mu)
        .powi_scalar(2)
        .mean()
        .into_scalar::<f32>()
        .sqrt();
    println!(
        "(b) logits[0,0,:]: mean={mu:.4} std={sd:.4}  (std~0 => constant logits => uniform CE)"
    );
    assert!(loss_tr.is_finite() && ce_inf.is_finite(), "NaN in probe");
}
