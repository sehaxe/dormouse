//! CUDA gradcheck for the fused ops (M0+M1). Builds the `small` model, runs
//! the fused path vs the current burn autodiff path on identical inputs, and
//! asserts forward loss/outputs and every weight/input grad match within
//! rel < 1e-4 (fp32). Needs the `cuda` feature - NdArray can't run the op.

use super::*;
use crate::fused::backward::BWD_DUMP;
use crate::config::DormouseConfig;
use crate::model::DormouseModel;
use crate::param::{LinearLike, LinearLikeInner};
use burn::module::Param;
use burn::tensor::{Distribution, TensorData};

/// The erased grads container returned by `Tensor::backward()`.
type BridgedGrads = burn::tensor::Gradients;

fn rel_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mut max_rel = 0f32;
    let mut max_abs = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(x.is_finite() && y.is_finite(), "non-finite value {x} vs {y}");
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / y.abs().max(1e-2));
    }
    (max_rel, max_abs)
}

fn tsct(ll: &LinearLike) -> &burn_spectral::SpectralLinear {
    match &ll.inner {
        LinearLikeInner::Tsct(l) => l,
        _ => panic!("fused path requires the TSCT arm"),
    }
}

fn fac_of(ll: &LinearLike) -> Fac {
    let l = tsct(ll);
    Fac {
        u: l.u.val(),
        s: l.s.val(),
        v: l.v.val(),
    }
}

fn tsct_mut(ll: &mut LinearLike) -> &mut burn_spectral::SpectralLinear {
    match &mut ll.inner {
        LinearLikeInner::Tsct(l) => l,
        _ => panic!("fused path requires the TSCT arm"),
    }
}

fn grad3(t: &Tensor<3>, g: &BridgedGrads) -> Option<Vec<f32>> {
    t.clone().grad(g).map(|x| x.into_data().try_to_vec().unwrap_or_default())
}

fn grad2(t: &Tensor<2>, g: &BridgedGrads) -> Option<Vec<f32>> {
    t.clone().grad(g).map(|x| x.into_data().try_to_vec().unwrap_or_default())
}

fn grad1(t: &Tensor<1>, g: &BridgedGrads) -> Option<Vec<f32>> {
    t.clone().grad(g).map(|x| x.into_data().try_to_vec().unwrap_or_default())
}

fn scalar(t: Tensor<1>) -> f32 {
    t.try_into_scalar().unwrap()
}

fn small_model(dev: &Device) -> (DormouseConfig, DormouseModel) {
    let cfg = DormouseConfig {
        use_kda: false,
        use_msa: false,
        use_engram: false,
        ..DormouseConfig::small()
    };
    let mut model = DormouseModel::new(&cfg, dev);
    model.loop_block.max_iter = 1;
    // ReZero starts at 0, which would zero the whole FFN-arm gradient; give
    // the gradcheck a live residual path.
    model.loop_block.residual_scale =
        Param::from_tensor(Tensor::<1>::from_data(TensorData::new(vec![0.7f32], [1]), dev));
    deflake_ternary_edges(&mut model, dev);
    (cfg, model)
}

/// Push every TSCT factor element away from the 0.7·mean dead-zone edge.
/// The fused kernel and the burn reference compute mean(|w|) in different
/// summation orders (Δ ~ 1e-7 relative); an element within Δ of the
/// threshold ternarizes differently on each side - a discrete jump that
/// would swamp the 1e-4 gradcheck (~1 expected flip per run otherwise).
fn deflake_ternary_edges(model: &mut DormouseModel, dev: &Device) {
    fn deflake(t: Tensor<2>, dev: &Device) -> Tensor<2> {
        let dims = t.dims();
        let mut v: Vec<f32> = t.into_data().try_to_vec().unwrap();
        let mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
        let thr = 0.7 * mean;
        let eps = 1e-3 * mean;
        for x in v.iter_mut() {
            let d = x.abs() - thr;
            if d.abs() < eps {
                let s = if *x < 0.0 { -1.0f32 } else { 1.0f32 };
                let abs = if d >= 0.0 { x.abs() + eps } else { (x.abs() - eps).max(0.0) };
                *x = s * abs;
            }
        }
        let [r, c] = dims;
        Tensor::from_data(TensorData::new(v, [r, c]), dev)
    }
    fn deflake_ll(ll: &mut LinearLike, dev: &Device) {
        if let LinearLikeInner::Tsct(l) = &mut ll.inner {
            let u = deflake(l.u.val(), dev);
            let v = deflake(l.v.val(), dev);
            l.u = Param::from_tensor(u);
            l.v = Param::from_tensor(v);
        }
    }
    for e in &mut model.loop_block.expert_ffns {
        deflake_ll(&mut e.gate_up, dev);
        deflake_ll(&mut e.down, dev);
    }
    deflake_ll(&mut model.loop_block.out_proj, dev);
    deflake_ll(&mut model.lm_head, dev);
}

fn collect(g: &BridgedGrads, model: &DormouseModel, x: &Tensor<3>) -> Vec<(String, Vec<f32>)> {
    let lb = &model.loop_block;
    let mut v: Vec<(String, Vec<f32>)> = Vec::new();
    v.push(("x".into(), grad3(x, g).expect("grad x")));
    v.push(("controller".into(), grad2(&lb.controller.weight.val(), g).expect("controller")));
    // NOTE: `lb.norm.weight` is deliberately absent - burn-rmsnorm builds it
    // with `Param::initialized(id, Tensor::ones(..))`, which never marks the
    // leaf as require_grad, so NO path (burn or fused) ever produces its grad
    // (upstream quirk: the gamma stays frozen at 1.0 in training today).
    v.push(("iter_embed".into(), grad2(&lb.iter_embed.val(), g).expect("iter_embed")));
    v.push(("residual_scale".into(), grad1(&lb.residual_scale.val(), g).expect("residual_scale")));
    v.push(("halt_w".into(), grad2(&lb.halt_head.weight.val(), g).expect("halt_w")));
    for (ei, e) in lb.expert_ffns.iter().enumerate() {
        let gu = tsct(&e.gate_up);
        let dn = tsct(&e.down);
        v.push((format!("gu{ei}.u"), grad2(&gu.u.val(), g).expect("gu.u")));
        v.push((format!("gu{ei}.s"), grad1(&gu.s.val(), g).expect("gu.s")));
        v.push((format!("gu{ei}.v"), grad2(&gu.v.val(), g).expect("gu.v")));
        v.push((format!("dn{ei}.u"), grad2(&dn.u.val(), g).expect("dn.u")));
        v.push((format!("dn{ei}.s"), grad1(&dn.s.val(), g).expect("dn.s")));
        v.push((format!("dn{ei}.v"), grad2(&dn.v.val(), g).expect("dn.v")));
    }
    let op = tsct(&lb.out_proj);
    v.push(("op.u".into(), grad2(&op.u.val(), g).expect("op.u")));
    v.push(("op.s".into(), grad1(&op.s.val(), g).expect("op.s")));
    v.push(("op.v".into(), grad2(&op.v.val(), g).expect("op.v")));
    let lm = tsct(&model.lm_head);
    v.push(("lm.u".into(), grad2(&lm.u.val(), g).expect("lm.u")));
    v.push(("lm.s".into(), grad1(&lm.s.val(), g).expect("lm.s")));
    v.push(("lm.v".into(), grad2(&lm.v.val(), g).expect("lm.v")));
    v
}

/// M0: the single-matmul custom op is exact against a host fp32 reference
/// (same serial-k order as the kernel), and its grads flow through the
/// normal Gradients map (the map GradientsParams::from_grads feeds to
/// optim.step), matching burn's backward.
#[test]
fn fused_matmul_matches_burn() {
    let dev = burn::tensor::Device::default().autodiff();
    let a: Tensor<2> = Tensor::random([64, 96], Distribution::Normal(0.0, 1.0), &dev).require_grad();
    let w: Tensor<2> = Tensor::random([96, 48], Distribution::Normal(0.0, 1.0), &dev).require_grad();
    let av: Vec<f32> = a.clone().into_data().try_to_vec().unwrap();
    let wv: Vec<f32> = w.clone().into_data().try_to_vec().unwrap();

    // ---- fused path
    let y = fused_matmul(a.clone(), w.clone());
    let yv: Vec<f32> = y.clone().into_data().try_to_vec().unwrap();
    // host fp64 reference: the kernel must be exact fp32, so the max abs
    // error against f64 stays in fp32-accumulation noise (< 1e-3 for values
    // of magnitude ~40); layout/launch corruption shows up at O(1)+ instead.
    let (m, k, n) = (64usize, 96usize, 48usize);
    let host: Vec<f64> = (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            (0..k).map(|j| av[r * k + j] as f64 * wv[j * n + c] as f64).sum()
        })
        .collect();
    let max_abs = yv
        .iter()
        .zip(&host)
        .map(|(x, h)| (x - *h as f32).abs())
        .fold(0.0f32, f32::max);
    println!("M0 fwd vs host f64: max_abs={max_abs:.2e}");
    assert!(max_abs < 1e-3, "fused matmul forward vs host f64 max_abs {max_abs:.2e}");

    let grads_f = y.sum().backward();
    let ga_f = grad2(&a, &grads_f).expect("dA must arrive");
    let gw_f = grad2(&w, &grads_f).expect("dW must arrive");
    assert_eq!(ga_f.len(), m * k, "dA must arrive in full");
    assert_eq!(gw_f.len(), k * n, "dW must arrive in full");
    assert!(ga_f.iter().chain(gw_f.iter()).all(|x| x.is_finite()));

    // ---- exact analytic reference (dY = ones: dA rows all equal
    // rowsum(W), dW rows all equal colsum(A))
    let ana_a: Vec<f64> = (0..m * k)
        .map(|i| {
            let c = i % k;
            (0..n).map(|j| wv[c * n + j] as f64).sum()
        })
        .collect();
    let ana_w: Vec<f64> = (0..k * n)
        .map(|i| {
            let r = i / n;
            (0..m).map(|row| av[row * k + r] as f64).sum()
        })
        .collect();
    let (rel_a, _) = rel_stats(&ga_f, &ana_a.iter().map(|x| *x as f32).collect::<Vec<_>>());
    let (rel_w, _) = rel_stats(&gw_f, &ana_w.iter().map(|x| *x as f32).collect::<Vec<_>>());

    // ---- burn reference (informational: its fp32 matmul autotunes to tf32
    // tiles - burn-vs-f64 above is ~1e-2 - so it cannot certify rel < 1e-4
    // against an exact kernel; the gap below is the reference's own noise)
    let y_ref = a.clone().matmul(w.clone());
    let yrv: Vec<f32> = y_ref.clone().into_data().try_to_vec().unwrap();
    let burn_abs = yrv
        .iter()
        .zip(&host)
        .map(|(x, h)| (x - *h as f32).abs())
        .fold(0.0f32, f32::max);
    let grads_r = y_ref.sum().backward();
    let ga_r = grad2(&a, &grads_r).expect("ref dA");
    let gw_r = grad2(&w, &grads_r).expect("ref dW");
    let (rel_ab, _) = rel_stats(&ga_f, &ga_r);
    let (rel_wb, _) = rel_stats(&gw_f, &gw_r);
    println!("M0 fwd vs host f64: max_abs={max_abs:.2e} | burn-vs-host: max_abs={burn_abs:.2e}");
    println!(
        "M0 dA rel(analytic)={rel_a:.2e} rel(vs burn)={rel_ab:.2e} | dW rel(analytic)={rel_w:.2e} rel(vs burn)={rel_wb:.2e}"
    );
    assert!(max_abs < 1e-3, "fused matmul forward vs host f64 max_abs {max_abs:.2e}");
    assert!(rel_a < 1e-4, "dA rel {rel_a:.2e}");
    assert!(rel_w < 1e-4, "dW rel {rel_w:.2e}");
}

/// Buffer-level bisect: run the fused backward with DM_FUSED_BWD_DEBUG=1 on
/// a tiny model, then compare every dumped intermediate against a
/// hand-written f64 forward+backward of the same computation. Also
/// cross-checks the f64 grads against the NdArray burn reference so a
/// mismatch localizes to the fused kernel, not to the test math.
#[test]
fn fused_bwd_buffers_vs_f64() {
    std::env::set_var("DM_FUSED_BWD_DEBUG", "1");
    std::env::set_var("DM_FUSED_DEBUG", "1");
    let dev = burn::tensor::Device::default().autodiff();
    let cfg = DormouseConfig {
        d_model: 64,
        n_heads: 4,
        head_dim: 16,
        d_ffn: 128,
        rank: 16,
        max_iter: 1,
        use_kda: false,
        use_msa: false,
        use_engram: false,
        ..DormouseConfig::small()
    };
    let mut model = DormouseModel::new(&cfg, &dev);
    model.loop_block.max_iter = 1;
    model.loop_block.residual_scale =
        Param::from_tensor(Tensor::<1>::from_data(TensorData::new(vec![0.7f32], [1]), &dev));
    deflake_ternary_edges(&mut model, &dev);
    let (b, t) = (1usize, 8usize);
    let d = cfg.d_model;
    let f = cfg.d_ffn;
    let r = cfg.rank;
    let nexp = model.loop_block.n_experts;
    let pad = model.loop_block.controller.weight.dims()[1];
    let v = cfg.vocab;
    let bt = b * t;
    let xh = TensorData::new(
        (0..b * t * d).map(|i| ((i % 61) as f32 - 30.0) / 30.0).collect::<Vec<f32>>(),
        [b, t, d],
    );
    let x: Tensor<3> = Tensor::from_data(xh.clone(), &dev).require_grad();
    let tgth = TensorData::new(
        (0..b * t).map(|i| ((i * 7 + 3) % 256) as i64).collect::<Vec<i64>>(),
        [b * t, 1],
    );
    let tgt_t: Tensor<2, Int> = Tensor::from_data(tgth.clone(), &dev);

    let inputs = PonderInputs {
        x: x.clone(),
        targets: tgt_t,
        controller_w: model.loop_block.controller.weight.val(),
        norm_g: model.loop_block.norm.weight.val(),
        iter_embed: model.loop_block.iter_embed.val(),
        residual_scale: model.loop_block.residual_scale.val(),
        halt_w: model.loop_block.halt_head.weight.val(),
        experts: model
            .loop_block
            .expert_ffns
            .iter()
            .map(|e| [fac_of(&e.gate_up), fac_of(&e.down)])
            .collect(),
        out_proj: fac_of(&model.loop_block.out_proj),
        lm_head: fac_of(&model.lm_head),
        norm_eps: cfg.norm_eps,
    };
    let (rec_f, pd_f, _oa_f) = ponder_loop_step(inputs);
    let loss_f = model.loss::<CAd>(rec_f.clone(), pd_f.clone());
    let _grads_f = loss_f.backward();
    BWD_DUMP.with(|dr| {
        let dump = dr.borrow();
        let get = |name: &str| -> Vec<f64> {
            dump.iter()
                .rev()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| v.iter().map(|&x| x as f64).collect())
                .unwrap_or_else(|| panic!("no dump for {name}"))
        };
        let fwd_dump: Vec<(&'static str, Vec<f32>)> =
            crate::fused::FWD_DUMP.with(|d| d.borrow().clone());
        let fwd = |name: &str| -> Vec<f64> {
            fwd_dump
                .iter()
                .rev()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| v.iter().map(|&x| x as f64).collect())
                .unwrap_or_else(|| panic!("no fwd dump for {name}"))
        };

        // ---- f64 forward intermediates
        let xf: Vec<f64> = xh.clone().to_vec::<f32>().unwrap().iter().map(|&v| v as f64).collect();
        let ie = host2(&model.loop_block.iter_embed.val());
        let g = host1(&model.loop_block.norm.weight.val());
        let wc = host2(&model.loop_block.controller.weight.val());
        let wh = host2(&model.loop_block.halt_head.weight.val());
        let rs = host1(&model.loop_block.residual_scale.val())[0];
        let eps = cfg.norm_eps as f64;
        let mut facs: Vec<[Vec<f64>; 3]> = Vec::new();
        for e in &model.loop_block.expert_ffns {
            for ll in [&e.gate_up, &e.down] {
                let l = tsct(ll);
                facs.push([host2(&l.u.val()), host1(&l.s.val()), host2(&l.v.val())]);
            }
        }
        let lop = tsct(&model.loop_block.out_proj);
        facs.push([host2(&lop.u.val()), host1(&lop.s.val()), host2(&lop.v.val())]);
        let llm = tsct(&model.lm_head);
        facs.push([host2(&llm.u.val()), host1(&llm.s.val()), host2(&llm.v.val())]);
        let tgt: Vec<usize> = tgth.to_vec::<i64>().unwrap().iter().map(|&x| x as usize).collect();

        let tern = |w: &[f64]| -> Vec<f64> {
            let mu = w.iter().map(|x| x.abs()).sum::<f64>() / w.len() as f64;
            w.iter().map(|&x| if x.abs() > 0.7 * mu { x.signum() * mu } else { 0.0 }).collect()
        };
        let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());

        // forward buffers
        let mut h_ctx = vec![0f64; bt * d];
        let mut normed = vec![0f64; bt * d];
        let mut inv = vec![0f64; bt];
        let mut raw = vec![0f64; bt * pad];
        let mut w_ffn = vec![0f64; bt];
        let mut blend = vec![0f64; bt * nexp];
        let mut ffn = vec![0f64; bt * d];
        let mut y = vec![0f64; bt * d];
        let mut h = vec![0f64; bt * d];
        let mut z_o = vec![0f64; bt * r];
        let mut step_out = vec![0f64; bt * d];
        let mut z_l = vec![0f64; bt * r];
        let mut logits = vec![0f64; bt * v];
        for m in 0..bt {
            for j in 0..d {
                h_ctx[m * d + j] = xf[m * d + j] + ie[j];
            }
            let msq: f64 = (0..d).map(|j| h_ctx[m * d + j] * h_ctx[m * d + j]).sum::<f64>() / d as f64;
            inv[m] = 1.0 / (msq + eps).sqrt();
            for j in 0..d {
                normed[m * d + j] = h_ctx[m * d + j] * inv[m] * g[j];
            }
            for c in 0..pad {
                let mut acc = 0f64;
                for i in 0..d {
                    acc += h_ctx[m * d + i] * wc[i * pad + c];
                }
                for i in 0..d {
                    acc += xf[m * d + i] * wc[(d + i) * pad + c];
                }
                raw[m * pad + c] = acc;
            }
            w_ffn[m] = sigmoid(raw[m * pad + 2]);
            let mx = (0..nexp).map(|k| raw[m * pad + 3 + k]).fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = (0..nexp).map(|k| (raw[m * pad + 3 + k] - mx).exp()).sum();
            for k in 0..nexp {
                blend[m * nexp + k] = (raw[m * pad + 3 + k] - mx).exp() / sum;
            }
            // expert FFN chain: Z = normed@U_t (unscaled), A = (Z·s)@V_tᵀ,
            // mid = silu(A), Zd = mid@U2_t, out = (Zd·s2)@V2_tᵀ
            for e in 0..nexp {
                let [u, s, vv] = &facs[2 * e];
                let (ut, vt) = (tern(u), tern(vv));
                let zr: Vec<f64> = (0..r)
                    .map(|k| (0..d).map(|i| normed[m * d + i] * ut[i * r + k]).sum::<f64>())
                    .collect();
                let mut mid = vec![0f64; f];
                for j in 0..f {
                    let a: f64 = (0..r).map(|k| zr[k] * s[k] * vt[j * r + k]).sum();
                    mid[j] = a * sigmoid(a);
                }
                let [u2, s2, v2] = &facs[2 * e + 1];
                let (u2t, v2t) = (tern(u2), tern(v2));
                let mut zd = vec![0f64; r];
                for k in 0..r {
                    zd[k] = (0..f).map(|j| mid[j] * u2t[j * r + k]).sum::<f64>();
                }
                for c in 0..d {
                    let o: f64 = (0..r).map(|k| zd[k] * s2[k] * v2t[c * r + k]).sum();
                    ffn[m * d + c] += o * blend[m * nexp + e];
                }
            }
            for c in 0..d {
                y[m * d + c] = ffn[m * d + c] * w_ffn[m];
                h[m * d + c] = h_ctx[m * d + c] + y[m * d + c] * rs;
            }
            let [uo, so, vo] = &facs[2 * nexp];
            let (uot, vot) = (tern(uo), tern(vo));
            for k in 0..r {
                z_o[m * r + k] = (0..d).map(|i| h[m * d + i] * uot[i * r + k]).sum::<f64>();
            }
            for c in 0..d {
                step_out[m * d + c] = (0..r).map(|k| z_o[m * r + k] * so[k] * vot[c * r + k]).sum();
            }
            let [ul, sl, vl] = &facs[2 * nexp + 1];
            let (ult, vlt) = (tern(ul), tern(vl));
            for k in 0..r {
                z_l[m * r + k] = (0..d).map(|i| step_out[m * d + i] * ult[i * r + k]).sum::<f64>();
            }
            for c in 0..v {
                logits[m * v + c] = (0..r).map(|k| z_l[m * r + k] * sl[k] * vlt[c * r + k]).sum();
            }
        }
        let mut lam = vec![0f64; b];
        let mut halt_in = vec![0f64; b * d];
        for bi in 0..b {
            for j in 0..d {
                halt_in[bi * d + j] = (0..t).map(|ti| h_ctx[(bi * t + ti) * d + j]).sum::<f64>() / t as f64;
            }
            let pre: f64 = (0..d).map(|j| halt_in[bi * d + j] * wh[j]).sum();
            lam[bi] = sigmoid(pre);
        }
        let mut ce = vec![0f64; bt];
        let mut ceb = vec![0f64; b];
        for m in 0..bt {
            let mx = (0..v).map(|c| logits[m * v + c]).fold(f64::NEG_INFINITY, f64::max);
            let lse = mx + (0..v).map(|c| (logits[m * v + c] - mx).exp()).sum::<f64>().ln();
            ce[m] = -(logits[m * v + tgt[m]] - lse);
            ceb[m / t] += ce[m];
        }
        let rec: f64 = (0..b).map(|bi| lam[bi] * ceb[bi]).sum::<f64>() / (bt) as f64;
        for (name, truth) in [
            ("h_ctx", h_ctx.clone()), ("normed", normed.clone()), ("raw", raw.clone()),
            ("ffn", ffn.clone()), ("h", h.clone()), ("z_o", z_o.clone()),
            ("step_out", step_out.clone()), ("z_l", z_l.clone()),
        ] {
            let fv = fwd(name);
            let (rel, abs) = rel_stats(&fv.iter().map(|&x| x as f32).collect::<Vec<_>>(), &truth.iter().map(|&x| x as f32).collect::<Vec<_>>());
            let bad: Vec<(usize, f64, f64)> = fv
                .iter()
                .zip(truth.iter())
                .enumerate()
                .filter(|(_, (a, b))| (*a - *b).abs() > 1e-5)
                .map(|(i, (a, b))| (i, *a, *b))
                .take(4)
                .collect();
            println!("  fwd {name:>9}: rel={rel:.3e} abs={abs:.3e} bad={bad:?}");
        }
        println!("  fused loss val={:.6} rec val={:.6}", scalar(loss_f.clone()), scalar(rec_f.clone()));
        // KL(p||Geom(lambda_p)) at N=1: prior = [1], KL = lam·ln lam, mean over b·N
        let beta = cfg.ponder_beta as f64;
        let kl_mean: f64 = (0..b).map(|bi| lam[bi] * lam[bi].ln()).sum::<f64>() / (b * 1) as f64;
        let loss = rec + beta * kl_mean;
        println!("[f64] loss={loss:.6} rec={rec:.6}");

        // ---- f64 backward (drec=1, dp = beta*(ln lam + 1)/(b*N), dout_acc=0)
        let dp: Vec<f64> = (0..b).map(|bi| beta * (lam[bi].ln() + 1.0) / (b * 1) as f64).collect();
        let drec = 1f64;
        let mut dlogits = vec![0f64; bt * v];
        for m in 0..bt {
            let mx = (0..v).map(|c| logits[m * v + c]).fold(f64::NEG_INFINITY, f64::max);
            let lse = mx + (0..v).map(|c| (logits[m * v + c] - mx).exp()).sum::<f64>().ln();
            for c in 0..v {
                let sm = (logits[m * v + c] - lse).exp();
                let oh = if c == tgt[m] { 1.0 } else { 0.0 };
                dlogits[m * v + c] = (sm - oh) * drec * lam[m / t] / bt as f64;
            }
        }
        let [ul_, sl_, vl_] = &facs[2 * nexp + 1];
        let vlt = tern(vl_);
        let mut dm_l = vec![0f64; bt * r];
        for m in 0..bt {
            for k in 0..r {
                dm_l[m * r + k] = (0..v).map(|c| dlogits[m * v + c] * vlt[c * r + k]).sum();
            }
        }
        let sl: Vec<f64> = sl_.clone();
        let dz_l: Vec<f64> = (0..bt * r).map(|i| dm_l[i] * sl[i % r]).collect();
        let dsl: Vec<f64> = (0..r)
            .map(|k| (0..bt).map(|m| dm_l[m * r + k] * z_l[m * r + k]).sum())
            .collect();
        let dvl: Vec<f64> = (0..v * r)
            .map(|i| {
                let (c, k) = (i / r, i % r);
                (0..bt).map(|m| dlogits[m * v + c] * z_l[m * r + k]).sum::<f64>() * sl[k]
            })
            .collect();
        let _dul: Vec<f64> = (0..d * r)
            .map(|i| {
                let (c, k) = (i / r, i % r);
                (0..bt).map(|m| step_out[m * d + c] * dz_l[m * r + k]).sum()
            })
            .collect();
        let ult = tern(ul_);
        let mut dstep = vec![0f64; bt * d];
        for m in 0..bt {
            for i in 0..d {
                dstep[m * d + i] = (0..r).map(|k| dz_l[m * r + k] * ult[i * r + k]).sum();
            }
        }
        // out_acc path contributes 0 (loss does not use out_acc).
        let [uo_, so_, vo_] = &facs[2 * nexp];
        let vot = tern(vo_);
        let mut dmo = vec![0f64; bt * r];
        for m in 0..bt {
            for k in 0..r {
                dmo[m * r + k] = (0..d).map(|c| dstep[m * d + c] * vot[c * r + k]).sum();
            }
        }
        let so: Vec<f64> = so_.clone();
        let dz_o: Vec<f64> = (0..bt * r).map(|i| dmo[i] * so[i % r]).collect();
        let dso: Vec<f64> = (0..r)
            .map(|k| (0..bt).map(|m| dmo[m * r + k] * z_o[m * r + k]).sum())
            .collect();
        let _dvo: Vec<f64> = (0..d * r)
            .map(|i| {
                let (c, k) = (i / r, i % r);
                (0..bt).map(|m| dstep[m * d + c] * z_o[m * r + k]).sum::<f64>() * so[k]
            })
            .collect();
        let duo: Vec<f64> = (0..d * r)
            .map(|i| {
                let (c, k) = (i / r, i % r);
                (0..bt).map(|m| h[m * d + c] * dz_o[m * r + k]).sum()
            })
            .collect();
        let uot = tern(uo_);
        let mut dh_flat = vec![0f64; bt * d];
        for m in 0..bt {
            for i in 0..d {
                dh_flat[m * d + i] = (0..r).map(|k| dz_o[m * r + k] * uot[i * r + k]).sum();
            }
        }
        let mut dy = vec![0f64; bt * d];
        let mut dh_ctx = vec![0f64; bt * d];
        let mut drs = 0f64;
        for m in 0..bt {
            for c in 0..d {
                let i = m * d + c;
                dy[i] = dh_flat[i] * rs;
                dh_ctx[i] += dh_flat[i];
                drs += dh_flat[i] * y[i];
            }
        }
        let mut dffn = vec![0f64; bt * d];
        let mut draw = vec![0f64; bt * pad];
        for m in 0..bt {
            for c in 0..d {
                dffn[m * d + c] = dy[m * d + c] * w_ffn[m];
            }
            draw[m * pad + 2] = (0..d).map(|c| dy[m * d + c] * ffn[m * d + c]).sum::<f64>()
                * w_ffn[m]
                * (1.0 - w_ffn[m]);
        }
        // per-expert backward
        let mut dblend = vec![0f64; bt * nexp];
        let mut da2 = vec![0f64; bt * f];
        let mut dnormed = vec![0f64; bt * d];
        let mut grads_expert: Vec<Vec<f64>> = Vec::new();
        for e in 0..nexp {
            let dout_e: Vec<f64> = (0..bt * d)
                .map(|i| {
                    let m = i / d;
                    dffn[i] * blend[m * nexp + e]
                })
                .collect();
            // recompute forward intermediates for this expert (mid, Zd)
            let [u_, s_, v_] = &facs[2 * e + 1]; // down
            let v2t = tern(v_);
            let s2: Vec<f64> = s_.clone();
            let u2t = tern(u_);
            let mut dm_d = vec![0f64; bt * r];
            for m in 0..bt {
                for k in 0..r {
                    dm_d[m * r + k] = (0..d).map(|c| dout_e[m * d + c] * v2t[c * r + k]).sum();
                }
            }
            let dz_d: Vec<f64> = (0..bt * r).map(|i| dm_d[i] * s2[i % r]).collect();
            let mut mid = vec![0f64; bt * f];
            let mut zd_all = vec![0f64; bt * r];
            let mut sil = vec![0f64; bt * f];
            for m in 0..bt {
                let [ug, sg, vg] = &facs[2 * e];
                let (ut, vt) = (tern(ug), tern(vg));
                for j in 0..f {
                    let zr: Vec<f64> = (0..r)
                        .map(|k| (0..d).map(|i| normed[m * d + i] * ut[i * r + k]).sum::<f64>() * sg[k])
                        .collect();
                    let a: f64 = (0..r).map(|k| zr[k] * vt[j * r + k]).sum();
                    mid[m * f + j] = a;
                    sil[m * f + j] = sigmoid(a);
                }
                for k in 0..r {
                    zd_all[m * r + k] = (0..f)
                        .map(|j| mid[m * f + j] * sigmoid(mid[m * f + j]) * u2t[j * r + k])
                        .sum::<f64>()
                        * s2[k];
                }
            }
            // dblend uses out_e (recomputed here)
            for m in 0..bt {
                for c in 0..d {
                    let o: f64 = (0..r).map(|k| zd_all[m * r + k] * s2[k] * v2t[c * r + k]).sum();
                    dblend[m * nexp + e] += dffn[m * d + c] * o;
                }
            }
            let ds_d2: Vec<f64> = (0..r)
                .map(|k| (0..bt).map(|m| dm_d[m * r + k] * zd_all[m * r + k]).sum())
                .collect();
            let dv_d: Vec<f64> = (0..d * r)
                .map(|i| {
                    let (c, k) = (i / r, i % r);
                    (0..bt).map(|m| dout_e[m * d + c] * zd_all[m * r + k]).sum::<f64>() * s2[k]
                })
                .collect();
            let du_d: Vec<f64> = (0..f * r)
                .map(|i| {
                    let (j, k) = (i / r, i % r);
                    (0..bt).map(|m| mid[m * f + j] * dz_d[m * r + k]).sum()
                })
                .collect();
            grads_expert.push(du_d);
            grads_expert.push(ds_d2);
            grads_expert.push(dv_d);
            // dsil = dz_d @ U2t^T ; da = dsil * sig * (1 + a(1-sig))
            for m in 0..bt {
                for j in 0..f {
                    let dsil: f64 = (0..r).map(|k| dz_d[m * r + k] * u2t[j * r + k]).sum();
                    let a = mid[m * f + j];
                    let sg = sil[m * f + j];
                    da2[m * f + j] = dsil * sg * (1.0 + a * (1.0 - sg));
                }
            }
            // gate_up backward: dz_e = (da @ Vt) * s
            let [ug_, sg_, vg_] = &facs[2 * e];
            let vt = tern(vg_);
            let sg: Vec<f64> = sg_.clone();
            let mut dm_e = vec![0f64; bt * r];
            for m in 0..bt {
                for k in 0..r {
                    dm_e[m * r + k] = (0..f).map(|j| da2[m * f + j] * vt[j * r + k]).sum();
                }
            }
            let dz_e: Vec<f64> = (0..bt * r).map(|i| dm_e[i] * sg[i % r]).collect();
            let ut = tern(ug_);
            let mut z_all = vec![0f64; bt * r];
            for m in 0..bt {
                for k in 0..r {
                    z_all[m * r + k] = (0..d).map(|i| normed[m * d + i] * ut[i * r + k]).sum::<f64>() * sg[k];
                }
            }
            let ds_e2: Vec<f64> = (0..r)
                .map(|k| (0..bt).map(|m| dm_e[m * r + k] * z_all[m * r + k]).sum())
                .collect();
            let dv_e: Vec<f64> = (0..f * r)
                .map(|i| {
                    let (j, k) = (i / r, i % r);
                    (0..bt).map(|m| da2[m * f + j] * z_all[m * r + k]).sum::<f64>() * sg[k]
                })
                .collect();
            let du_e: Vec<f64> = (0..d * r)
                .map(|i| {
                    let (c, k) = (i / r, i % r);
                    (0..bt).map(|m| normed[m * d + c] * dz_e[m * r + k]).sum()
                })
                .collect();
            grads_expert.push(du_e);
            grads_expert.push(ds_e2);
            grads_expert.push(dv_e);
            for m in 0..bt {
                for i in 0..d {
                    dnormed[m * d + i] += (0..r).map(|k| dz_e[m * r + k] * ut[i * r + k]).sum::<f64>();
                }
            }
        }
        // softmax bwd
        for m in 0..bt {
            let dot: f64 = (0..nexp).map(|k| dblend[m * nexp + k] * blend[m * nexp + k]).sum();
            for k in 0..nexp {
                draw[m * pad + 3 + k] = blend[m * nexp + k] * (dblend[m * nexp + k] - dot);
            }
        }
        // controller
        let mut dwc = vec![0f64; 2 * d * pad];
        let mut dctrl = vec![0f64; bt * 2 * d];
        for m in 0..bt {
            for c in 0..pad {
                for i in 0..d {
                    dwc[i * pad + c] += h_ctx[m * d + i] * draw[m * pad + c];
                    dwc[(d + i) * pad + c] += xf[m * d + i] * draw[m * pad + c];
                }
                let dr = draw[m * pad + c];
                for i in 0..2 * d {
                    dctrl[m * 2 * d + i] += dr * wc[i * pad + c];
                }
            }
        }
        // halt
        let dosum = vec![0f64; b]; // out_acc unused by the loss
        let mut dhaltpre = vec![0f64; b];
        for bi in 0..b {
            let dlam = dp[bi] + drec * ceb[bi] / bt as f64 + dosum[bi];
            dhaltpre[bi] = dlam * lam[bi] * (1.0 - lam[bi]);
        }
        let dwh: Vec<f64> = (0..d).map(|j| (0..b).map(|bi| dhaltpre[bi] * halt_in[bi * d + j]).sum()).collect();
        let mut dhalt_in = vec![0f64; b * d];
        for bi in 0..b {
            for j in 0..d {
                dhalt_in[bi * d + j] = dhaltpre[bi] * wh[j];
            }
        }
        for bi in 0..b {
            for ti in 0..t {
                for j in 0..d {
                    dh_ctx[(bi * t + ti) * d + j] += dhalt_in[bi * d + j] / t as f64;
                }
            }
        }
        // rmsnorm bwd
        let dg: Vec<f64> = (0..d)
            .map(|j| (0..bt).map(|m| dnormed[m * d + j] * h_ctx[m * d + j] * inv[m]).sum())
            .collect();
        for m in 0..bt {
            let s_tot: f64 = (0..d)
                .map(|j| {
                    dnormed[m * d + j] * g[j] * (h_ctx[m * d + j] * inv[m])
                })
                .sum::<f64>()
                / d as f64;
            for j in 0..d {
                let rr = h_ctx[m * d + j] * inv[m];
                dh_ctx[m * d + j] += inv[m] * (dnormed[m * d + j] * g[j] - rr * s_tot);
            }
        }
        // cat split
        let mut dx = dnormed.clone();
        for m in 0..bt {
            for j in 0..d {
                dh_ctx[m * d + j] += dctrl[m * 2 * d + j];
                dx[m * d + j] += dctrl[m * 2 * d + d + j];
            }
        }
        let die: Vec<f64> = (0..d).map(|j| (0..bt).map(|m| dh_ctx[m * d + j]).sum()).collect();
        for i in 0..bt * d {
            dx[i] += dh_ctx[i];
        }

        // ---- compare fused dumps vs f64
        let cmp = |name: &str, truth: &[f64]| {
            let got = get(name);
            let (rel, abs) = rel_stats(&got.iter().map(|&x| x as f32).collect::<Vec<_>>(), &truth.iter().map(|&x| x as f32).collect::<Vec<_>>());
            println!("  {name:>10}: rel={rel:.3e} abs={abs:.3e} (n={})", truth.len());
        };
        cmp("dlogits", &dlogits);
        cmp("dm_l", &dm_l);
        cmp("dz_l", &dz_l);
        cmp("dsl", &dsl);
        cmp("dstep", &dstep);
        cmp("dzo", &dz_o);
        cmp("dh_flat", &dh_flat);
        cmp("dy", &dy);
        cmp("dffn", &dffn);
        cmp("dctrl", &dctrl);
        cmp("draw", &draw);
        cmp("dhaltpre", &dhaltpre);
        cmp("dwh", &dwh);
        cmp("dosum", &dosum);
        cmp("dg", &dg);
        cmp("die", &die);
        cmp("dh_ctx", &dh_ctx);
        cmp("dx", &dx);

        // ---- cross-check f64 grads against the NdArray burn reference
        type NdAd = burn::backend::Autodiff<burn::backend::NdArray>;
        let cpu_dev = burn::tensor::Device::ndarray().autodiff();
        let mut ref_model = DormouseModel::new(&cfg, &cpu_dev);
        ref_model.loop_block.max_iter = 1;
        ref_model.loop_block.residual_scale =
            Param::from_tensor(Tensor::<1>::from_data(TensorData::new(vec![0.7f32], [1]), &cpu_dev));
        copy_weights(&mut ref_model, &model, &cpu_dev);
        let x_r: Tensor<3> = Tensor::from_data(xh.clone(), &cpu_dev).require_grad();
        let tgt_r: Tensor<2, Int> = Tensor::from_data(tgth.clone(), &cpu_dev);
        let (oa_r, rec_r, pd_r, _k) = ref_model.loop_block.forward_full_state::<NdAd>(
            x_r.clone(), None, None, None, Some(tgt_r), &ref_model.lm_head,
        );
        let loss_r = ref_model.loss::<NdAd>(rec_r.clone(), pd_r.clone());
        let grads_r = loss_r.backward();
        let lb = &ref_model.loop_block;
        let ref2 = |t: &Tensor<2>| -> Vec<f64> {
            t.clone().grad(&grads_r).expect("ref grad").into_data().to_vec::<f32>().unwrap()
                .iter().map(|&x| x as f64).collect()
        };
        let ref1 = |t: &Tensor<1>| -> Vec<f64> {
            t.clone().grad(&grads_r).expect("ref grad").into_data().to_vec::<f32>().unwrap()
                .iter().map(|&x| x as f64).collect()
        };
        let cross = |name: &str, truth: &[f64], reff: &[f64]| {
            let (rel, _) = rel_stats(&truth.iter().map(|&x| x as f32).collect::<Vec<_>>(), &reff.iter().map(|&x| x as f32).collect::<Vec<_>>());
            println!("  [xcheck {name:>8}] f64-vs-burn rel={rel:.3e}");
            let pk: Vec<usize> = (0..truth.len())
                .filter(|&i| truth[i] != 0.0 || reff[i] != 0.0)
                .take(3)
                .collect();
            for i in pk {
                println!("    [{name}] i={i} f64={:+.6e} burn={:+.6e}", truth[i], reff[i]);
            }
        };
        let lmr = tsct(&ref_model.lm_head);
        cross("lm.s", &dsl, &ref1(&lmr.s.val()));
        cross("lm.v", &dvl, &ref2(&lmr.v.val()));
        let opr = tsct(&lb.out_proj);
        cross("op.s", &dso, &ref1(&opr.s.val()));
        cross("op.u", &duo, &ref2(&opr.u.val()));
        cross("halt_w", &dwh, &ref2(&lb.halt_head.weight.val()));
        cross("rs", &[drs], &ref1(&lb.residual_scale.val()));
        cross("iter_embed", &die, &ref2(&lb.iter_embed.val()));
        cross("x", &dx, &ref2_test(&x_r, &grads_r));
        let gu0 = tsct(&lb.expert_ffns[0].gate_up);
        cross("gu0.s", &grads_expert[4], &ref1(&gu0.s.val()));

    });
}

fn ref2_test(x_r: &Tensor<3>, grads_r: &BridgedGrads) -> Vec<f64> {
    x_r.clone()
        .grad(grads_r)
        .expect("x grad")
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&x| x as f64)
        .collect()
}

fn host1(t: &Tensor<1>) -> Vec<f64> {
    t.clone().into_data().to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect()
}

fn host2(t: &Tensor<2>) -> Vec<f64> {
    t.clone().into_data().to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect()
}

/// Copy every fused-op weight from `from` (CUDA) into `m` (CPU reference),
/// verbatim via host TensorData.
fn copy_weights(m: &mut DormouseModel, from: &DormouseModel, dev: &Device) {
    fn w2(t: Tensor<2>, dev: &Device) -> Param<Tensor<2>> {
        Param::from_tensor(Tensor::from_data(t.into_data(), dev))
    }
    fn w1(t: Tensor<1>, dev: &Device) -> Param<Tensor<1>> {
        Param::from_tensor(Tensor::from_data(t.into_data(), dev))
    }
    let lb = &mut m.loop_block;
    let src = &from.loop_block;
    lb.controller.weight = w2(src.controller.weight.val(), dev);
    lb.halt_head.weight = w2(src.halt_head.weight.val(), dev);
    lb.iter_embed = w2(src.iter_embed.val(), dev);
    lb.residual_scale = w1(src.residual_scale.val(), dev);
    for (dst, s) in lb.expert_ffns.iter_mut().zip(&src.expert_ffns) {
        copy_ll(&mut dst.gate_up, &s.gate_up, dev);
        copy_ll(&mut dst.down, &s.down, dev);
    }
    copy_ll(&mut lb.out_proj, &src.out_proj, dev);
    copy_ll(&mut m.lm_head, &from.lm_head, dev);
}
fn copy_ll(dst: &mut LinearLike, src: &LinearLike, dev: &Device) {
    fn w2(t: Tensor<2>, dev: &Device) -> Param<Tensor<2>> {
        Param::from_tensor(Tensor::from_data(t.into_data(), dev))
    }
    fn w1(t: Tensor<1>, dev: &Device) -> Param<Tensor<1>> {
        Param::from_tensor(Tensor::from_data(t.into_data(), dev))
    }
    let (LinearLikeInner::Tsct(d), LinearLikeInner::Tsct(s)) = (&mut dst.inner, &src.inner) else {
        panic!("fused path requires the TSCT arm");
    };
    d.u = w2(s.u.val(), dev);
    d.s = w1(s.s.val(), dev);
    d.v = w2(s.v.val(), dev);
}

/// Full single-iteration forward in f64 on the host (ground truth for the
/// gradcheck): returns out_acc = step_out · lam.
#[allow(clippy::too_many_arguments)]
fn f64_forward(
    xv: &[f32],
    ie0: &[f32],
    g: &[f32],
    wc: &[f32],
    wht: &[f32],
    rs: f32,
    eps: f32,
    facs: &[[Vec<f64>; 3]], // 6 expert linears, then out_proj, then lm (unused)
    nexp: usize,
    b: usize,
    t: usize,
    d: usize,
    f: usize,
    r: usize,
    pad: usize,
) -> (Vec<f64>, Vec<f64>) {
    let tern = |w: &[f64]| -> Vec<f64> {
        let mu = w.iter().map(|x| x.abs()).sum::<f64>() / w.len() as f64;
        w.iter().map(|&x| if x.abs() > 0.7 * mu { x.signum() * mu } else { 0.0 }).collect()
    };
    let bt = b * t;
    let mut oacc = vec![0f64; bt * d];
    let mut hall = vec![0f64; bt * d];
    let mut lam = vec![0f64; b];
    let mut halt_acc = vec![0f64; b * d];
    for row in 0..bt {
        let hc: Vec<f64> = (0..d).map(|j| xv[row * d + j] as f64 + ie0[j] as f64).collect();
        let msq = hc.iter().map(|x| x * x).sum::<f64>() / d as f64;
        let inv = 1.0 / (msq + eps as f64).sqrt();
        let normed: Vec<f64> = hc.iter().zip(g).map(|(x, g)| x * inv * *g as f64).collect();
        // raw = [h_ctx | x] @ Wc
        let mut raw = vec![0f64; pad];
        for c in 0..pad {
            let mut acc = 0f64;
            for i in 0..d {
                acc += hc[i] * wc[i * pad + c] as f64;
            }
            for i in 0..d {
                acc += xv[row * d + i] as f64 * wc[(d + i) * pad + c] as f64;
            }
            raw[c] = acc;
        }
        if row == 0 {
            println!("[f64] raw0: {raw:?}");
            println!("[f64] wc[0..8]: {:?}", &wc[..8.min(wc.len())]);
        }
        let w_ffn = 1.0 / (1.0 + (-raw[2]).exp());
        let mut blend = vec![0f64; nexp];
        let mx = raw[3..3 + nexp].iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let sum: f64 = raw[3..3 + nexp].iter().map(|x| (x - mx).exp()).sum();
        for (e, be) in blend.iter_mut().enumerate() {
            *be = (raw[3 + e] - mx).exp() / sum;
        }
        let mut ffn = vec![0f64; d];
        for e in 0..nexp {
            let [u, s, v] = &facs[2 * e];
            let [u2, s2, v2] = &facs[2 * e + 1];
            let (ut, vt) = (tern(u), tern(v));
            let (ut2, vt2) = (tern(u2), tern(v2));
            let mut z = vec![0f64; r];
            let mut mid = vec![0f64; f];
            for (mi, midv) in mid.iter_mut().enumerate() {
                for (kk, zv) in z.iter_mut().enumerate() {
                    *zv = (0..d).map(|i| normed[i] * ut[i * r + kk]).sum::<f64>() * s[kk];
                }
                let a: f64 = (0..r).map(|kk| z[kk] * vt[mi * r + kk]).sum();
                *midv = a / (1.0 + (-a).exp());
            }
            let mut zd = vec![0f64; r];
            for (kk, zv) in zd.iter_mut().enumerate() {
                *zv = (0..f).map(|i| mid[i] * ut2[i * r + kk]).sum::<f64>() * s2[kk];
            }
            for (cj, ffnv) in ffn.iter_mut().enumerate() {
                *ffnv += blend[e]
                    * (0..r)
                        .map(|kk| zd[kk] * s2[kk] * vt2[cj * r + kk])
                        .sum::<f64>();
            }
        }
        let hc_or: Vec<f64> = hc
            .iter()
            .enumerate()
            .map(|(j, h)| h + ffn[j] * w_ffn * rs as f64)
            .collect();
        if row == 0 {
            println!(
                "[f64] ffn0: {:?} w_ffn={w_ffn:.6} blend0={:?}",
                &ffn[..4],
                &blend
            );
            println!("[f64] h0: {:?}", &hc_or[..4]);
        }
        hall[row * d..row * d + d].copy_from_slice(&hc_or);
        // out_proj chain on h
        let [uo, so, vo] = &facs[2 * nexp];
        let (uot, vot) = (tern(uo), tern(vo));
        for (cj, ov) in oacc[row * d..row * d + d].iter_mut().enumerate() {
            let zo: Vec<f64> = (0..r)
                .map(|kk| (0..d).map(|i| hc_or[i] * uot[i * r + kk]).sum::<f64>() * so[kk])
                .collect();
            *ov = (0..r).map(|kk| zo[kk] * vot[cj * r + kk]).sum::<f64>();
        }
        let bi = row / t;
        for j in 0..d {
            // halt_in = mean_t h_ctx (BEFORE the residual add)
            halt_acc[bi * d + j] += hc[j] / t as f64;
        }
    }
    for bi in 0..b {
        let pre: f64 = (0..d).map(|j| halt_acc[bi * d + j] * wht[j] as f64).sum();
        lam[bi] = 1.0 / (1.0 + (-pre).exp());
    }
    for (i, o) in oacc.iter_mut().enumerate() {
        *o *= lam[i / (t * d)];
    }
    // row-0 intermediates for debugging: y (post w_ffn gate) and h
    (oacc, hall)
}

/// M1: the whole single-iteration loop body under one autodiff node -
/// forward loss + every weight/input grad vs the burn path, rel < 1e-4.
///
/// The burn reference runs on NdArray with the SAME weights/inputs: the CUDA
/// fp32 matmul autotunes to tf32 tiles (M0: burn-vs-f64 max_abs ~1e-2), so a
/// CUDA reference could not certify rel < 1e-4 even for an exact kernel.
#[test]
fn fused_gradcheck_single_iteration() {
    let dev = burn::tensor::Device::default().autodiff();
    let (cfg, model) = small_model(&dev);
    let (b, t) = (2usize, 16usize);
    let d = cfg.d_model;
    let xh = TensorData::new(
        (0..b * t * d)
            .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
            .collect::<Vec<f32>>(),
        [b, t, d],
    );
    let x: Tensor<3> = Tensor::from_data(xh.clone(), &dev).require_grad();
    let tgt: Vec<i64> = (0..b * t).map(|i| (i * 31 + 7) as i64 % 256).collect();
    let tgth = TensorData::new(tgt.clone(), [b * t, 1]);
    let tgt_t: Tensor<2, Int> = Tensor::from_data(tgth.clone(), &dev);

    // ---- fused path (CUDA, one autodiff node)
    let inputs = PonderInputs {
        x: x.clone(),
        targets: tgt_t,
        controller_w: model.loop_block.controller.weight.val(),
        norm_g: model.loop_block.norm.weight.val(),
        iter_embed: model.loop_block.iter_embed.val(),
        residual_scale: model.loop_block.residual_scale.val(),
        halt_w: model.loop_block.halt_head.weight.val(),
        experts: model
            .loop_block
            .expert_ffns
            .iter()
            .map(|e| [fac_of(&e.gate_up), fac_of(&e.down)])
            .collect(),
        out_proj: fac_of(&model.loop_block.out_proj),
        lm_head: fac_of(&model.lm_head),
        norm_eps: cfg.norm_eps,
    };
    let (rec_f, pd_f, oa_f) = ponder_loop_step(inputs);
    let loss_f = model.loss::<CAd>(rec_f.clone(), pd_f.clone());
    let loss_f_val: f32 = scalar(loss_f.clone());
    let grads_f = loss_f.backward();
    let fus_grads = collect(&grads_f, &model, &x);

    // ---- exact fp32 burn reference (NdArray, same weights/inputs)
    type NdAd = burn::backend::Autodiff<burn::backend::NdArray>;
    let cpu_dev = burn::tensor::Device::ndarray().autodiff();
    let mut ref_model = DormouseModel::new(&cfg, &cpu_dev);
    ref_model.loop_block.max_iter = 1;
    ref_model.loop_block.residual_scale =
        Param::from_tensor(Tensor::<1>::from_data(TensorData::new(vec![0.7f32], [1]), &cpu_dev));
    copy_weights(&mut ref_model, &model, &cpu_dev);
    let x_r: Tensor<3> = Tensor::from_data(xh.clone(), &cpu_dev).require_grad();
    let tgt_r: Tensor<2, Int> = Tensor::from_data(tgth.clone(), &cpu_dev);
    let (oa_r, rec_r, pd_r, _kda) = ref_model.loop_block.forward_full_state::<NdAd>(
        x_r.clone(),
        None,
        None,
        None,
        Some(tgt_r),
        &ref_model.lm_head,
    );
    let loss_r = ref_model.loss::<NdAd>(rec_r.clone(), pd_r.clone());
    let loss_r_val: f32 = scalar(loss_r.clone());
    let grads_r = loss_r.backward();
    let ref_grads = collect(&grads_r, &ref_model, &x_r);

    // ---- forward equality
    let (loss_rel, _) = rel_stats(&[loss_f_val], &[loss_r_val]);
    let (rec_rel, _) = rel_stats(&[scalar(rec_f.clone())], &[scalar(rec_r.clone())]);
    let pdv_f: Vec<f32> = pd_f.into_data().try_to_vec().unwrap();
    let pdv_r: Vec<f32> = pd_r.clone().into_data().try_to_vec().unwrap();
    let (pd_rel, _) = rel_stats(&pdv_f, &pdv_r);
    let oav_f: Vec<f32> = oa_f.into_data().try_to_vec().unwrap();
    let oav_r: Vec<f32> = oa_r.into_data().try_to_vec().unwrap();
    let (oa_rel, oa_abs) = rel_stats(&oav_f, &oav_r);
    // ground truth: full forward in f64 on the host, compared against BOTH
    // sides to attribute any residual difference.
    {
        let g8 = |t: Tensor<2>| -> Vec<f64> {
            t.into_data().to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect()
        };
        let g1 = |t: Tensor<1>| -> Vec<f64> {
            t.into_data().to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect()
        };
        let mut facs: Vec<[Vec<f64>; 3]> = Vec::new();
        for e in &model.loop_block.expert_ffns {
            for ll in [&e.gate_up, &e.down] {
                let l = tsct(ll);
                facs.push([g8(l.u.val()), g1(l.s.val()), g8(l.v.val())]);
            }
        }
        let opf = tsct(&model.loop_block.out_proj);
        facs.push([g8(opf.u.val()), g1(opf.s.val()), g8(opf.v.val())]);
        let (truth, _hall) = f64_forward(
            &xh.clone().to_vec().unwrap(),
            &model
                .loop_block
                .iter_embed
                .val()
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            &model.loop_block.norm.weight.val().into_data().to_vec::<f32>().unwrap(),
            &model.loop_block.controller.weight.val().into_data().to_vec::<f32>().unwrap(),
            &model.loop_block.halt_head.weight.val().into_data().to_vec::<f32>().unwrap(),
            scalar(model.loop_block.residual_scale.val()),
            cfg.norm_eps,
            &facs,
            model.loop_block.n_experts,
            b,
            t,
            d,
            cfg.d_ffn,
            cfg.rank,
            model.loop_block.controller.weight.dims()[1],
        );
        let mut e_f = 0f64;
        let mut e_r = 0f64;
        for i in 0..b * t * d {
            e_f = e_f.max((oav_f[i] as f64 - truth[i]).abs());
            e_r = e_r.max((oav_r[i] as f64 - truth[i]).abs());
        }
        println!("  out_acc vs f64 truth: fused max_abs={e_f:.3e} | cpu-ref max_abs={e_r:.3e}");
        // localize: which row (b,t) carries the fused error, and does h match?
        let (worst_i, _) = oav_f
            .iter()
            .zip(&truth)
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let da = (*a.0 as f64 - *a.1).abs();
                let db = (*b.0 as f64 - *b.1).abs();
                da.total_cmp(&db)
            })
            .unwrap();
        println!(
            "  worst fused out_acc i={worst_i} (b={},t={},d={})",
            worst_i / (t * d),
            (worst_i / d) % t,
            worst_i % d
        );
    }
    let mut offs: Vec<(usize, f32)> = oav_f
        .iter()
        .zip(&oav_r)
        .map(|(a, b)| (a - b).abs())
        .enumerate()
        .collect();
    offs.sort_by(|x, y| y.1.total_cmp(&x.1));
    for &(i, dd) in offs.iter().take(6) {
        println!(
            "  out_acc diff i={i} (b={},t={},d={}) fused={} ref={} |d|={dd:.3e}",
            i / (t * d),
            (i / d) % t,
            i % d,
            oav_f[i],
            oav_r[i]
        );
    }
    println!("M1 fwd loss {:.5}/{:.5} rel={loss_rel:.2e} rec rel={rec_rel:.2e} p_dist rel={pd_rel:.2e} out_acc rel={oa_rel:.2e} abs={oa_abs:.2e}", loss_f_val, loss_r_val);
    assert!(loss_rel < 1e-4, "loss rel {loss_rel:.2e}");
    assert!(rec_rel < 1e-4, "rec rel {rec_rel:.2e}");
    assert!(pd_rel < 1e-4, "p_dist rel {pd_rel:.2e}");
    assert!(oa_rel < 1e-4, "out_acc rel {oa_rel:.2e}");

    // ---- gradient equality, path by path
    let mut worst = (String::new(), 0f32);
    for ((name, gr), (_, gf)) in ref_grads.iter().zip(fus_grads.iter()) {
        let (rel, abs) = rel_stats(gf, gr);
        println!("  {name:>12}: rel={rel:.2e} abs={abs:.2e} (n={})", gr.len());
        if rel > worst.1 {
            worst = (name.clone(), rel);
        }
    }
    println!("worst: {} rel={:.2e}", worst.0, worst.1);
    assert!(worst.1 < 1e-4, "gradcheck failed on {}: rel {:.2e}", worst.0, worst.1);
}
