use burn::tensor::{Tensor, TensorData, Device, Int};
use burn::module::{Module, Param};
use dormouse_core::config::DormouseConfig;
use dormouse_core::model::DormouseModel;
use dormouse_core::fused::{ponder_loop_step, PonderInputs, Fac};
use dormouse_core::param::{LinearLike, LinearLikeInner};
fn fac(ll: &LinearLike) -> Fac { match &ll.inner { LinearLikeInner::Tsct(l) => Fac{ u:l.u.val(), s:l.s.val(), v:l.v.val()}, _=>panic!() } }
fn main(){
 let dev: Device = Device::cuda(0).autodiff();
 let cfg = DormouseConfig { max_iter: 4, use_kda:true, use_msa:true, use_engram:true, ..DormouseConfig::default() };
 let mut model = DormouseModel::new(&cfg, &dev);
 model.loop_block.residual_scale = Param::from_tensor(Tensor::<1>::from_data(TensorData::new(vec![0.7f32],[1]), &dev));
 let b=2; let t=16; let d=cfg.d_model; let bt=b*t;
 let xh = TensorData::new((0..b*t*d).map(|i| ((i%97) as f32 -48.)/48.).collect::<Vec<f32>>(), [b,t,d]);
 let x = Tensor::<3>::from_data(xh.clone(), &dev).require_grad();
 let tgt = Tensor::<2, Int>::from_data(TensorData::new((0..bt).map(|i| (i*31%256) as i64).collect::<Vec<i64>>(), [bt,1]), &dev);
 let hashed = Tensor::<3, Int>::from_data(TensorData::new((0..b*t*3).map(|i| ((i*17+5)%4096) as i64).collect::<Vec<i64>>(), [b,t,3]), &dev);
 let lb_bytes = model.loop_block.clone().into_record().into_bytes().unwrap().to_vec();
 let inputs = PonderInputs { x: x.clone(), targets: tgt.clone(), controller_w: model.loop_block.controller.weight.val(), norm_g: model.loop_block.norm.weight.val(), final_norm_g: model.norm.weight.val(), iter_embed: model.loop_block.iter_embed.val(), residual_scale: model.loop_block.residual_scale.val(), halt_w: model.loop_block.halt_head.weight.val(), experts: model.loop_block.expert_ffns.iter().map(|e| [fac(&e.gate_up), fac(&e.down)]).collect(), out_proj: fac(&model.loop_block.out_proj), lm_head: fac(&model.lm_head), norm_eps: cfg.norm_eps, ponder_prior: model.ponder_prior, hashed_ids: Some(hashed), loop_block_bytes: Some(lb_bytes), cfg: Some(cfg.clone()) };
 println!("start fused");
 let (logits, rec, pd, kl) = ponder_loop_step(inputs);
 println!("fused forward done rec {:?}", rec.clone().into_data());
 let loss = rec.clone() + kl.clone().mul_scalar(model.ponder_beta) + { let lg=logits.clone().reshape([bt, cfg.vocab]); burn::tensor::activation::log_softmax(lg,1).gather(1, tgt.clone()).neg().sum().div_scalar(bt as f32) };
 println!("loss {:?}", loss.clone().into_data());
 let grads = loss.backward();
 println!("backward done");
 let xg = x.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap();
 println!("xg len {} first {}", xg.len(), xg[0]);
}
