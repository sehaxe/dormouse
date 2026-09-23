// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
use burn_msa::{MsaCache, MsaConfig, MsaModule};
use burn_ndarray::NdArray;

type B = NdArray;

fn main() {
    let device = Default::default();
    let cfg = MsaConfig::new(128, 4, 2, 32, 16);
    println!(
        "d_model={} n_heads_q={} n_heads_kv={} topk={} block_size={}",
        cfg.d_model, cfg.n_heads_q, cfg.n_heads_kv, cfg.topk, cfg.block_size
    );
    cfg.validate().expect("invalid config");

    let module = MsaModule::new(&cfg, &device);

    let x = burn::tensor::Tensor::<3>::random(
        [2, 16, 128],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let result = module.forward::<B>(x);
    println!("self-attn output: {:?}", result.output.dims());
    assert_eq!(result.output.dims(), [2, 16, 128]);

    let q = burn::tensor::Tensor::<3>::random(
        [2, 8, 128],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let kv = burn::tensor::Tensor::<3>::random(
        [2, 32, 128],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let result = module.forward_cross::<B>(q, kv);
    println!("cross-attn output: {:?}", result.output.dims());
    assert_eq!(result.output.dims(), [2, 8, 128]);

    let mut cache = MsaCache::new(cfg.clone());
    cache.reset(1);
    let seg = burn::tensor::Tensor::<3>::ones([1, 3, 128], &device);
    let out = cache.step(&module, seg, 0);
    println!("cache step output: {:?}", out.dims());
    assert_eq!(out.dims(), [1, 3, 128]);
    assert_eq!(cache.stream_len(0), 3);

    println!("All checks passed");
}
