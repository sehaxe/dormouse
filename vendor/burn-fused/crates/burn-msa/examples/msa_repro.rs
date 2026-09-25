//! Staged MSA forward repro: run each stage, then read back — the stage
//! whose readback fails is where the context got poisoned.
use burn::backend::Autodiff as Ad;
use burn::prelude::*;

type CudaBare = burn_cubecl::CubeBackend;
type B = Ad<CudaBare, burn_autodiff::checkpoint::strategy::BalancedCheckpointing>;

fn probe(name: &str, t: &Tensor<3>) {
    let s: f32 = t.clone().sum().into_scalar();
    println!("stage {name}: OK sum={s}");
}

fn main() {
    let device = Default::default();
    // small preset: d_model 768, seq 512, batch 10; msa: topk 8, block 32
    let mut cfg = burn_msa::config::MsaConfig::default();
    cfg.d_model = 768;
    cfg.n_heads_q = 12;
    cfg.n_heads_kv = 3;
    cfg.d_head = 64;
    cfg.d_idx = 64;
    cfg.block_size = 32;
    cfg.topk = 8;
    cfg.use_rope = false;
    cfg.gradient_detach = false;
    let m = burn_msa::module::MsaModule::new(&cfg, &device);

    let base: Tensor<2> = Tensor::from_floats([[1.0f32, 2.0]], &device);
    let base: Tensor<2> = Tensor::from_floats([[1.0f32, 2.0]], &device);
    let x: Tensor<3> = base.clone()
        .repeat_dim(0, 10 * 512 * 768 / 2)
        .reshape([10, 512, 768]);
    // two iterations with different hidden states, like the ponder loop
    let x2: Tensor<3> = (base.clone() + 3.0)
        .repeat_dim(0, 10 * 512 * 768 / 2)
        .reshape([10, 512, 768]);
    let keep: Vec<Vec<burn::Tensor<3>>> = if std::env::var("KEEP").is_ok() {
        let mut all = Vec::new();
        for (pass, x) in [x, x2].into_iter().enumerate() {
            all.push(run_pass_keep(&m, &cfg, x, pass));
        }
        println!("keepers alive: {} tensors", all.iter().map(|v| v.len()).sum::<usize>());
        all
    } else {
        for (pass, x) in [x, x2].into_iter().enumerate() {
            run_pass(&m, &cfg, x, pass);
        }
        Vec::new()
    };
    let _ = keep;
    println!("MSA-REPRO-ALL-OK");
}

fn run_pass_keep(
    m: &burn_msa::module::MsaModule,
    cfg: &burn_msa::config::MsaConfig,
    x: Tensor<3>,
    pass: usize,
) -> Vec<burn::Tensor<3>> {
    let mut keep = Vec::new();
    let out = run_pass_inner(m, cfg, x, pass, Some(&mut keep));
    keep.push(out);
    keep
}

fn run_pass(
    m: &burn_msa::module::MsaModule,
    cfg: &burn_msa::config::MsaConfig,
    x: Tensor<3>,
    pass: usize,
) {
    run_pass_inner(m, cfg, x, pass, None);
}

fn run_pass_inner(
    m: &burn_msa::module::MsaModule,
    cfg: &burn_msa::config::MsaConfig,
    x: Tensor<3>,
    pass: usize,
    mut keep: Option<&mut Vec<burn::Tensor<3>>>,
) -> burn::Tensor<3> {
    let device = x.device();
    let stage = std::env::var("P1_STAGE").unwrap_or_default();
    let skip_scores = pass == 1 && stage == "idx";
    let skip_select = pass == 1 && (stage == "idx" || stage == "scores");
    let skip_sparse = pass == 1 && (stage == "idx" || stage == "scores" || stage == "select" || stage == "rope");

    let (q_idx, k_idx) = m.index_branch.forward(x.clone(), x.clone(), cfg.n_heads_kv, cfg.d_idx);
    if let Some(k) = keep.as_mut() { k.push(q_idx.clone().reshape([10, 3 * 512, 64])); k.push(k_idx.clone().reshape([10, 512, 64])); }
    println!("index_branch.forward done");
    let scale = (cfg.d_idx as f64).sqrt();
    let scores = if skip_scores {
        println!("pass {pass}: block_scores SKIPPED");
        Tensor::zeros([10, 3, 512, 16], &device)
    } else {
        let s = m
            .index_branch
            .compute_block_scores(q_idx.clone(), k_idx.clone(), cfg.block_size, scale, cfg.causal);
        probe("block_scores", &s.clone().reshape([10, 3 * 512, 16]));
        if let Some(k) = keep.as_mut() { k.push(s.clone().reshape([10, 3 * 512, 16])); }
        s
    };

    let idx = if skip_select {
        println!("pass {pass}: select SKIPPED");
        let z: Tensor<4, Int> = Tensor::zeros([10, 3, 512, 8], &device);
        z
    } else {
        let selector = burn_msa::topk::TopKSelector::new(cfg.topk, cfg.block_size, cfg.force_local_block);
        let i = selector.select::<B>(scores.clone());
        let s: f32 = i.clone().float().sum().into_scalar();
        println!("select: OK sum={s}");
        i
    };

    let q = m.attention.q_proj.forward(x.clone());
    let k = m.attention.k_proj.forward(x.clone());
    let v = m.attention.v_proj.forward(x.clone());
    println!("qkv projections done");
    let (q, k) = match &m.rope {
        Some(rope) => {
            let q = burn_rope::apply_rope_3d::<B>(q, rope.cos.clone(), rope.sin.clone(), cfg.n_heads_q);
            let k = burn_rope::apply_rope_3d::<B>(k, rope.cos.clone(), rope.sin.clone(), cfg.n_heads_kv);
            println!("rope done");
            if stage == "rope" && pass == 1 {
                let qs: f32 = q.clone().sum().into_scalar();
                let ks: f32 = k.clone().sum().into_scalar();
                println!("P1 rope probe: q={qs} k={ks}");
            }
            (q, k)
        }
        None => (q, k),
    };
    let (out, block_attn) = if skip_sparse {
        println!("pass {pass}: sparse SKIPPED");
        (Tensor::zeros([10, 512, 768], &device), Tensor::zeros([10, 3, 512, 16], &device))
    } else {
        let r = m
            .attention
            .forward_sparse_with_weights::<B>(q, k, v, idx.clone());
        println!("pass {pass}: sparse done");
        r
    };
    if let Some(k) = keep.as_mut() { k.push(out.clone()); }
    if !skip_sparse { probe("sparse_attn out", &out); }
    let s: f32 = block_attn.clone().sum().into_scalar();
    println!("pass {pass}: block_attn: OK sum={s}");
    if std::env::var("NO_KL").is_err() && !skip_select && !skip_sparse {
        let kl = burn_msa::loss::KlAlignmentLoss.compute::<B>(scores, block_attn, idx);
        let s: f32 = kl.sum().into_scalar();
        println!("pass {pass}: kl: OK sum={s}");
    }
    out
}
