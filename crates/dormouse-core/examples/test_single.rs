use burn::tensor::{Tensor, TensorData, Device};
use dormouse_core::config::DormouseConfig;
use dormouse_core::model::DormouseModel;
use dormouse_core::fused::{ponder_loop_step, PonderInputs, Fac};
use dormouse_core::param::{LinearLike, LinearLikeInner};
use burn::module::{Module, Param};
fn fac(ll: &LinearLike) -> Fac { match &ll.inner { LinearLikeInner::Tsct(l) => Fac{ u:l.u.val(), s:l.s.val(), v:l.v.val()}, _=>panic!() } }
fn main(){
 let dev = Device::cuda(0).autodiff();
 let cfg = DormouseConfig { max_iter: 1, use_kda:false, use_msa:false, use_engram:false, ..DormouseConfig::default() };
 let mut model = DormouseModel::new(&cfg, &dev);
 model.loop_block.residual_scale = Param::from_tensor(Tensor::<1>::from_data(TensorData::new(vec![0.7f32],[1]), &dev));
 let b=2; let t=16; let d=cfg.d_model; let bt=b*t;
 let start = std::time::Instant::now();
 for i in 0..3 {
   let xh = TensorData::new((0..b*t*d).map(|i| ((i%97) as f32 -48.)/48.).collect::<Vec<f32>>(), [b,t,d]);
   let x = Tensor::<3>::from_data(xh, &dev).require_grad();
   let tgt = Tensor::<2, burn::tensor::Int>::from_data(TensorData::new((0..bt).map(|i| (i*31%256) as i64).collect::<Vec<i64>>(), [bt,1]), &dev);
   let inputs = PonderInputs { x: x.clone(), targets: tgt.clone(), controller_w: model.loop_block.controller.weight.val(), norm_g: model.loop_block.norm.weight.val(), final_norm_g: model.norm.weight.val(), iter_embed: model.loop_block.iter_embed.val(), residual_scale: model.loop_block.residual_scale.val(), halt_w: model.loop_block.halt_head.weight.val(), experts: model.loop_block.expert_ffns.iter().map(|e| [fac(&e.gate_up), fac(&e.down)]).collect(), out_proj: fac(&model.loop_block.out_proj), lm_head: fac(&model.lm_head), norm_eps: cfg.norm_eps, ponder_prior: model.ponder_prior, hashed_ids: None, loop_block_bytes: None, cfg: None };
   let (logits, rec, pd, kl) = ponder_loop_step(inputs);
   let loss = rec.clone() + kl.clone().mul_scalar(model.ponder_beta) + { let lg=logits.clone().reshape([bt, cfg.vocab]); burn::tensor::activation::log_softmax(lg,1).gather(1, tgt.clone()).neg().sum().div_scalar(bt as f32) };
   let grads = loss.backward();
    let xg = x.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap();
    println!("iter {} done xg {} time {:?}", i, xg[0], start.elapsed());
 }
}
