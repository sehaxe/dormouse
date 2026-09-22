use dormouse_core::config::DormouseConfig;
use dormouse_core::model::DormouseModel;
use dormouse_core::param::{LinearLike, LinearLikeInner};
use dormouse_core::fused::{ponder_loop_step, PonderInputs, Fac};
use burn::tensor::{Device, Tensor, TensorData};
use burn::module::{Module, Param};
use std::time::Instant;

fn fac_of(ll: &LinearLike) -> Fac {
    match &ll.inner {
        LinearLikeInner::Tsct(l) => Fac { u: l.u.val(), s: l.s.val(), v: l.v.val() },
        _ => panic!("tsct"),
    }
}

fn main() {
    let dev = Device::cuda(0).autodiff();
    let cfg = DormouseConfig { max_iter: 8, use_kda: false, use_msa: false, use_engram: false, ..DormouseConfig::default() };
    let mut model = DormouseModel::new(&cfg, &dev);
    model.loop_block.residual_scale = Param::from_tensor(Tensor::<1>::from_data(TensorData::new(vec![0.7f32], [1]), &dev));
    let b=2; let t=512; let d=cfg.d_model;
    let xh = TensorData::new((0..b*t*d).map(|i| ((i%97) as f32 -48.0)/48.0).collect::<Vec<f32>>(), [b,t,d]);
    let x: Tensor<3> = Tensor::from_data(xh.clone(), &dev).require_grad();
    let tgth = TensorData::new((0..b*t).map(|i| ((i*31+7)%256) as i64).collect::<Vec<i64>>(), [b*t,1]);
    let tgt = Tensor::from_data(tgth.clone(), &dev);
    let hashed = Tensor::from_data(TensorData::new((0..b*t*3).map(|i| (i as i64*17+5)%4096).collect::<Vec<i64>>(), [b,t,3]), &dev);
    let lb_bytes = model.loop_block.clone().into_record().into_bytes().unwrap().to_vec();
    let inputs = PonderInputs {
        x: x.clone(),
        targets: tgt.clone(),
        controller_w: model.loop_block.controller.weight.val(),
        norm_g: model.loop_block.norm.weight.val(),
        final_norm_g: model.norm.weight.val(),
        iter_embed: model.loop_block.iter_embed.val(),
        residual_scale: model.loop_block.residual_scale.val(),
        halt_w: model.loop_block.halt_head.weight.val(),
        experts: model.loop_block.expert_ffns.iter().map(|e| [fac_of(&e.gate_up), fac_of(&e.down)]).collect(),
        out_proj: fac_of(&model.loop_block.out_proj),
        lm_head: fac_of(&model.lm_head),
        norm_eps: cfg.norm_eps,
        ponder_prior: model.ponder_prior,
        hashed_ids: Some(hashed.clone()),
        loop_block_bytes: Some(lb_bytes.clone()),
        cfg: Some(cfg.clone()),
    };
    // warmup
    let (logits, rec, pd, kl) = ponder_loop_step(inputs.clone());
    let loss = rec.clone() + kl.clone().mul_scalar(model.ponder_beta) + {
        let bt=b*t; let v=cfg.vocab;
        let lg = logits.clone().reshape([bt, v]);
        burn::tensor::activation::log_softmax(lg, 1).gather(1, tgt.clone()).neg().sum().div_scalar(bt as f32)
    };
    let _ = loss.backward();
    // timed fused
    let start = Instant::now();
    for _ in 0..3 {
        let (logits, rec, pd, kl) = ponder_loop_step(inputs.clone());
        let loss = rec.clone() + kl.clone().mul_scalar(model.ponder_beta) + {
            let bt=b*t; let v=cfg.vocab;
            let lg = logits.clone().reshape([bt, v]);
            burn::tensor::activation::log_softmax(lg, 1).gather(1, tgt.clone()).neg().sum().div_scalar(bt as f32)
        };
        let _ = loss.backward();
    }
    let elapsed = start.elapsed();
    println!("fused 3 steps avg {:.2?}", elapsed/3);
    // ref
    let mut ref_model = DormouseModel::new(&cfg, &dev);
    ref_model.loop_block.residual_scale = Param::from_tensor(Tensor::<1>::from_data(TensorData::new(vec![0.7f32], [1]), &dev));
    ref_model = ref_model.load_record(model.clone().into_record());
    let start2 = Instant::now();
    for _ in 0..3 {
        let x_r: Tensor<3> = Tensor::from_data(xh.clone(), &dev).require_grad();
        let tgt_r = Tensor::from_data(tgth.clone(), &dev);
        let hashed_r = Tensor::from_data(TensorData::new((0..b*t*3).map(|i| (i as i64*17+5)%4096).collect::<Vec<i64>>(), [b,t,3]), &dev);
        let (oa, rec, pd, _) = ref_model.loop_block.forward_full_state::<burn::backend::Autodiff<burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>>>(x_r.clone(), Some(hashed_r), None, None, Some(tgt_r.clone().reshape([b*t,1])), &ref_model.lm_head);
        let h = ref_model.norm.forward(oa.clone());
        let logits = ref_model.lm_head.forward::<burn::backend::Autodiff<burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>>>(h.reshape([b*t,d])).reshape([b,t,cfg.vocab]);
        let loss = ref_model.loss::<burn::backend::Autodiff<burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>>>(rec.clone(), pd.clone()) + {
            let lg = logits.clone().reshape([b*t, cfg.vocab]);
            burn::tensor::activation::log_softmax(lg, 1).gather(1, tgt_r.clone()).neg().sum().div_scalar((b*t) as f32)
        };
        let _ = loss.backward();
    }
    let elapsed2 = start2.elapsed();
    println!("ref 3 steps avg {:.2?}", elapsed2/3);
    println!("speedup: {:.2}x", elapsed2.as_secs_f64() / elapsed.as_secs_f64());
}
