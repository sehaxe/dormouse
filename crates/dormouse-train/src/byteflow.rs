//! The ByteFlow arm (arXiv 2603.03583, the owner's staged plan Этап 1):
//! ByteFlow Net — local SWA+Canon encoder → coding-rate Top-K chunker →
//! global transformer → multi-linear upsampling → symmetric decoder —
//! trained for byte-level CE and scored in BPB, behind `use_byteflow`.
//!
//! WHAT THIS IS. The paper's own chunker, whole: marginal coding rate
//! `ΔR_t = R(h_1:t) − R(h_1:t−1)` (eq. 11/12) over the LOCAL ENCODER's hidden
//! states, Top-K borders with BOS forced, chronological — not a fixed stride
//! and not entropy (that comparison is the paper's Table 3). The patcher's
//! oracle lives with the patcher
//! (`vendor/dormouse-fused/crates/burn-byteflow/tests/dynamic_patcher_oracle.rs`).
//!
//! WHAT THIS IS NOT. Not a dormouse arm. ByteFlowNet has no LoopBlock, no KDA,
//! no Engram, no aux heads, no TSCT retraction, so every dormouse-only knob is
//! refused loudly here or in `config::validate` (ADR-0011) rather than
//! silently ignored, and the optimizer is plain AdamW — the mix router walks
//! `DormouseModel`'s parameter tree and does not apply. The step-0 arm line
//! says exactly what ran.
//!
//! The dormouse train loop itself is untouched (the NaN firewall and the
//! graph seam especially): this module is dispatched from `train_loop` right
//! after the device seed, so the seed-before-any-parameter rule, the config
//! snapshot and the drift check are shared code, and the data no-leak rule is
//! `ByteStream::train_and_eval`'s, as in the dormouse path.
//!
//! # Checkpoint container
//!
//! Same shape as the trainer's (`[magic 8B][step][model_len][optim_len]`
//! header + burnpack records, atomic tmp+rename, the previous save rotated
//! aside by hard link), with its OWN magic `DMBF` so a byteflow checkpoint
//! can never be parsed as a dormouse one: the loader refuses a foreign
//! container by magic loudly instead of starting over a file a human may
//! believe is this run's state.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::OptimizerRecord;
use burn::store::ModuleRecord;
use burn::tensor::{activation::log_softmax, Bool, Bytes, Int, Tensor, TensorData};
use burn_byteflow::{ByteFlowConfig, ByteFlowNet, RateMode};

use crate::cfg::RunCfg;
use crate::{device, init_pools, mask_nonfinite, Optim, TrainCfg};
use dormouse_core::DormouseConfig;

const BF_MAGIC: [u8; 8] = *b"DMBF\x00\x01\x00\x00";
const BF_HEADER: usize = 32;

/// The resolved model config → the crate's config. Fields not listed carry
/// the crate's defaults, which is where the value the paper does not state
/// lives (`rope_base`, marked `TODO(бумага)` on both sides).
fn net_config(c: &DormouseConfig) -> ByteFlowConfig {
    ByteFlowConfig {
        d_local: c.byteflow_d_local,
        d_global: c.byteflow_d_global,
        k_tokens: c.byteflow_k_tokens,
        e_layers: c.byteflow_e_layers,
        g_layers: c.byteflow_g_layers,
        n_heads_local: c.byteflow_heads_local,
        n_heads_global: c.byteflow_heads_global,
        w_local: c.byteflow_w_local,
        d_ff_local: c.byteflow_d_ff_local,
        d_ff_global: c.byteflow_d_ff_global,
        bins: c.byteflow_bins,
        eps2: c.byteflow_eps2,
        rate_mode: if c.byteflow_logdet {
            RateMode::LogDet
        } else {
            RateMode::L2
        },
        max_bytes: c.byteflow_max_bytes,
        ..ByteFlowConfig::default()
    }
}

/// The train-side refusals: the arms `config::validate` cannot see. Each
/// names the escape. `--quant` (the TSCT factor format) is deliberately a
/// COUNTED line in the arm banner, not a refusal — a preset carrying it is
/// not a different experiment, the field simply has no reader on this path.
pub fn check(cfg: &TrainCfg) -> Result<(), String> {
    let refuse = |what: &str, why: &str| Err(format!("byteflow refuses {what}: {why}"));
    if cfg.bf16.unwrap_or(false) {
        return refuse(
            "--bf16",
            "a cast-copy step on this backend, and ByteFlowNet has no bf16 handling, so the \
             flag would silently do nothing. Run the byteflow arm fp32 (the paper's recipe).",
        );
    }
    if cfg.graph_capture {
        return refuse(
            "--graph-capture",
            "v1 of the arm does not implement capture; the dormouse loop's graph seam is \
             untouched by this module and off here",
        );
    }
    if cfg.jepa_targets.is_some() {
        return refuse("--jepa-targets", "ByteFlowNet carries no JEPA teacher");
    }
    if cfg.engram_ram {
        return refuse("--engram-ram", "ByteFlowNet carries no Engram arm");
    }
    if cfg.rand_depth {
        return refuse(
            "--rand-depth",
            "ByteFlowNet's depth is layer counts (e_layers + g_layers), fixed by the config",
        );
    }
    if cfg.eval_depths {
        return refuse(
            "--eval-depths",
            "ByteFlowNet has one depth; the depth curve is a dormouse-loop instrument",
        );
    }
    if cfg.stress {
        return refuse(
            "--stress",
            "the stress monitor reads the dormouse optimizer's parameter groups",
        );
    }
    match cfg.opt.as_str() {
        "adamw" | "mix" => Ok(()),        other => Err(format!(
            "byteflow refuses --opt {other}: the arm runs plain AdamW (the mix router walks \
             DormouseModel's parameter tree). Leave --opt at its default; the step-0 arm line \
             names AdamW."
        )),
    }
}

/// Train ByteFlowNet on the same data seam, cadences and checkpoint
/// discipline as the dormouse loop. The device must already be seeded
/// (`device.seed` ran at the dispatch site, before any parameter existed —
/// the 2026-09-28 rule).
pub fn train_loop(
    run: RunCfg,
    data: PathBuf,
    ckpt_dir: Option<PathBuf>,
    eval_data: Option<PathBuf>,
) -> Result<(), String> {
    check(&run.train)?;
    let dir = ckpt_dir.unwrap_or_else(|| PathBuf::from("checkpoints"));
    let _ = std::fs::create_dir_all(&dir);
    let dorm = run.model;
    let cfg = run.train;
    let device = device();
    init_pools(&device);

    let net_cfg = net_config(&dorm);
    let mut net = ByteFlowNet::init(net_cfg, &device);
    // THE ARM LINE (ADR-0011): what ran, and the config fields this path does
    // not read.
    let patcher = if dorm.byteflow_logdet { "logdet" } else { "l2" };
    println!(
        "byteflow arm: params={} patcher={patcher} k={} bins={} eps2={} opt=AdamW(wd={:.2e}) \
         aux=none host-tables=none quant=unused",
        net.num_params(),
        dorm.byteflow_k_tokens,
        dorm.byteflow_bins,
        dorm.byteflow_eps2,
        cfg.wd,
    );

    let mut optim: Optim = burn::optim::AdamWConfig::new()
        .with_weight_decay(cfg.wd as f32)
        .init()
        .into();
    let mut step = load_ckpt(&dir, &cfg.ckpt_name, &dorm, &mut net, &mut optim);
    if step > 0 {
        println!("resumed {} at step {step}", cfg.ckpt_name);
    }

    // Same data seam and no-leak rule as the dormouse loop.
    let (mut stream, mut eval_stream) = dormouse_data::ByteStream::train_and_eval(
        cfg.seq_len,
        cfg.batch,
        &data,
        eval_data.as_deref(),
    );
    // The warmup batch is the dormouse loop's pre-loop consumption; a byteflow
    // run mirrors it so a shared preset's step index means the same batch.
    let pre_batches = cfg.warmup as u64;
    if step > 0 {
        let skip = (step + pre_batches) * (cfg.batch * cfg.seq_len) as u64;
        stream.skip_bytes(skip);
        println!("stream resume: skipped {skip} bytes (step {step} + {pre_batches} pre-loop batch(es))");
    } else if cfg.warmup {
        let _ = stream.next_batch();
    }

    let mut best = f32::INFINITY;
    let mut best_eval_bpb = f32::INFINITY;
    let mut best_eval_step: u64 = 0;
    let mut ce = f32::NAN;
    let t_run = std::time::Instant::now();
    while step < cfg.steps as u64 {
        let t_iter = std::time::Instant::now();
        let (bytes, _hashes) = stream.next_batch();
        let (x, y) = batch_tensors(&bytes, cfg.seq_len, cfg.batch, &device);

        let logits = net.forward(x);
        let [b, t, v] = logits.dims();
        // The honest CE: the mean over positions and batch of the target
        // log-prob, gathered (no one-hot [b*t,v] temporary).
        let flat = logits.reshape([b * t, v]);
        let ce_terms = log_softmax(flat, 1)
            .gather(1, y.reshape([b * t, 1]))
            .neg()
            .reshape([b, t]);
        let loss = ce_terms
            .sum_dim(1)
            .div_scalar(t as f32)
            .sum_dim(0)
            .reshape([1])
            .div_scalar(b as f32);
        let loss = mask_nonfinite(loss);
        // The RAW loss read once per step (this loop is CPU-first and the
        // smoke is 1 step; the dormouse loop's cadence trick is its launch
        // budget, not a semantic rule).
        ce = loss.clone().try_into_scalar().unwrap_or(f32::NAN);
        let mut raw_grads = loss.backward();
        sanitize_grads(&mut raw_grads, &net)?;
        let grads = burn::optim::GradientsParams::from_grads(raw_grads, &net);
        let lr = crate::wsd_factor(step, cfg.steps as u64, cfg.lr);
        net = optim.step(lr, net, grads);
        step += 1;
        if step % cfg.log_every as u64 == 0 || step == 1 {
            let ms = t_iter.elapsed().as_secs_f64() * 1000.0;
            if ce.is_finite() {
                best = best.min(ce);
            }
            println!(
                "step {step:6} ce={:.3} bpb={:.3} best={:.3} lr={lr:.2e} ms/step={ms:.0}",
                ce,
                crate::bpb(ce),
                best,
            );
        }
        if cfg.eval_every > 0 && step % cfg.eval_every as u64 == 0 {
            let Some(ev) = eval_stream.as_mut() else {
                continue;
            };
            // Rewind FIRST: every eval scores the SAME bytes (2026-09-27).
            ev.rewind();
            let eval_net = net.valid();
            let mut ce_sum = 0.0f32;
            let mut n: u32 = 0;
            for _ in 0..cfg.eval_batches.max(1) {
                let (eb, _) = ev.next_batch();
                let (ex, ey) = batch_tensors(&eb, cfg.seq_len, cfg.batch, &device);
                let el = eval_net.forward(ex);
                let [eb_, et, ev_] = el.dims();
                let ece: f32 = log_softmax(el.reshape([eb_ * et, ev_]), 1)
                    .gather(1, ey.reshape([eb_ * et, 1]))
                    .neg()
                    .mean()
                    .try_into_scalar()
                    .unwrap_or(f32::NAN);
                ce_sum += ece;
                n += 1;
            }
            let ece = ce_sum / n.max(1) as f32;
            if !ece.is_finite() {
                return Err(format!(
                    "step {step}: held-out eval is {ece:.3} (non-finite) - the model is broken, not the data"
                ));
            }
            let ebpb = crate::bpb(ece);
            let scored = (n as u64) * (cfg.batch * cfg.seq_len) as u64;
            let is_best = ebpb < best_eval_bpb;
            if is_best {
                best_eval_bpb = ebpb;
                best_eval_step = step;
            }
            println!(
                "eval step {step} bpb={ebpb:.3} ce={ece:.3} over {scored} B patcher={patcher}{}",
                if is_best { " BEST" } else { "" }
            );
        }
        if step.is_multiple_of(cfg.ckpt_every.max(1) as u64) {
            save_ckpt(&dir, &cfg.ckpt_name, &net, &optim, step, ce).map_err(|e| format!("byteflow ckpt: {e}"))?;
        }
    }
    save_ckpt(&dir, &cfg.ckpt_name, &net, &optim, step, ce).map_err(|e| format!("byteflow ckpt: {e}"))?;
    let mins = t_run.elapsed().as_secs_f64() / 60.0;
    if best_eval_bpb.is_finite() {
        println!(
            "done steps={step} best held-out bpb={best_eval_bpb:.3} at step {best_eval_step} ({mins:.1} min)"
        );
    } else {
        println!("done steps={step} best train ce={best:.3} (no held-out eval ran; {mins:.1} min)");
    }
    Ok(())
}

/// Zero the non-finite GRADIENTS of the ByteFlowNet on device (`loss` masking
/// is not enough — a NaN activation back-propagates through a masked zero
/// seed): the same pattern the dormouse loop's GradSanitizer runs, over the
/// byteflow module tree. A visited parameter with NO gradient is treated as a
/// wiring defect, loudly — a missing gradient is not a masked step.
fn sanitize_grads(grads: &mut burn::tensor::Gradients, net: &ByteFlowNet) -> Result<(), String> {
    struct San<'g> {
        grads: &'g mut burn::tensor::Gradients,
        visited: usize,
        bad: Vec<String>,
        path: Vec<String>,
    }
    impl ModuleVisitor for San<'_> {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.path.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.visited += 1;
            let name = self.path.join(".");
            match param.grad(self.grads) {
                Some(g) => {
                    let nonfinite: Tensor<D, Bool> = g.clone().is_finite().bool_not();
                    param.grad_replace(self.grads, g.mask_fill(nonfinite, 0.0));
                }
                None => self.bad.push(format!("{name}:missing")),
            }
        }
    }
    let mut san = San {
        grads,
        visited: 0,
        bad: Vec::new(),
        path: Vec::new(),
    };
    net.visit(&mut san);
    if !san.bad.is_empty() {
        return Err(format!(
            "byteflow: {} of {} parameter gradients missing after backward (a wiring defect): {:?}",
            san.bad.len(),
            san.visited,
            &san.bad[..san.bad.len().min(8)]
        ));
    }
    Ok(())
}

/// One batch → (input bytes `x`, next-byte labels `y`) int tensors — the same
/// shift the dormouse loop builds (`targets[q]` is the byte at `q + 1`, the
/// last entry the wraparound).
fn batch_tensors(
    bytes: &[u8],
    seq_len: usize,
    batch: usize,
    dev: &burn::tensor::Device,
) -> (Tensor<2, Int>, Tensor<2, Int>) {
    assert_eq!(bytes.len(), seq_len * batch, "batch length");
    let x: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
    let shifted: Vec<i64> = bytes
        .iter()
        .skip(1)
        .chain(std::iter::once(&bytes[0]))
        .map(|&b| b as i64)
        .collect();
    (
        Tensor::from_data(TensorData::new(x, [batch, seq_len]), dev),
        Tensor::from_data(TensorData::new(shifted, [batch, seq_len]), dev),
    )
}

/// Save a byteflow checkpoint: `[BF_MAGIC][step][model_len][optim_len]` +
/// the two burnpack records, atomic (tmp + rename), previous save rotated
/// aside by hard link — the trainer's own save pattern, one container apart.
fn save_ckpt(
    dir: &Path,
    name: &str,
    net: &ByteFlowNet,
    optim: &Optim,
    step: u64,
    ce: f32,
) -> std::io::Result<()> {
    let live = dir.join(format!("{name}.bin"));
    if live.exists() {
        let prev = dir.join(format!("{name}.prev.bin"));
        let _ = std::fs::remove_file(&prev);
        let _ = std::fs::hard_link(&live, &prev);
    }
    let eio = |e: String| std::io::Error::other(e);
    let model_bytes = net
        .clone()
        .into_record()
        .into_bytes()
        .map_err(|e| eio(e.to_string()))?;
    let optim_bytes = optim
        .to_record()
        .into_bytes()
        .map_err(|e| eio(e.to_string()))?;
    let tmp = dir.join(format!("{name}.bin.tmp.{}", std::process::id()));
    {
        let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&tmp)?);
        w.write_all(&BF_MAGIC)?;
        w.write_all(&step.to_le_bytes())?;
        w.write_all(&(model_bytes.len() as u64).to_le_bytes())?;
        w.write_all(&(optim_bytes.len() as u64).to_le_bytes())?;
        w.write_all(&model_bytes)?;
        w.write_all(&optim_bytes)?;
        w.flush()?;
    }
    std::fs::rename(&tmp, dir.join(format!("{name}.bin")))?;
    std::fs::write(
        dir.join(format!("{name}.txt")),
        format!("step {step} ce {ce:.3}\n"),
    )?;
    Ok(())
}

/// Load a byteflow checkpoint; returns the step it stopped at (0 = fresh).
/// A file that exists and is NOT a byteflow container is a loud panic naming
/// the magic, never a silent start-over; the inner records unparsable are the
/// same refusal.
fn load_ckpt(
    dir: &Path,
    name: &str,
    dorm: &DormouseConfig,
    net: &mut ByteFlowNet,
    optim: &mut Optim,
) -> u64 {
    for cand in [format!("{name}.bin"), format!("{name}.prev.bin")] {
        let path = dir.join(&cand);
        let Ok(raw) = std::fs::read(&path) else {
            continue;
        };
        if raw.len() < BF_HEADER || raw[..8] != BF_MAGIC {
            let head = String::from_utf8_lossy(&raw[..raw.len().min(8)]).to_string();
            panic!(
                "checkpoint {} exists and is not a byteflow container (magic {head:?}): refusing \
                 to start over a file that may carry this run's state; point --ckpt-name elsewhere",
                path.display()
            );
        }
        let u8b = |s: &[u8]| u64::from_le_bytes(s.try_into().expect("8-byte window"));
        let (step, mlen, olen) = (
            u8b(&raw[8..16]),
            u8b(&raw[16..24]) as usize,
            u8b(&raw[24..32]) as usize,
        );
        assert!(
            BF_HEADER + mlen + olen <= raw.len(),
            "checkpoint {} is truncated ({} < {})",
            path.display(),
            raw.len(),
            BF_HEADER + mlen + olen
        );
        let mb = Bytes::from_bytes_vec(raw[BF_HEADER..BF_HEADER + mlen].to_vec());
        let ob = Bytes::from_bytes_vec(raw[BF_HEADER + mlen..BF_HEADER + mlen + olen].to_vec());
        let mrec = ModuleRecord::from_bytes(mb).expect("byteflow model record unparsable");
        let orec = OptimizerRecord::from_bytes(ob).expect("byteflow optim record unparsable");
        let fresh = ByteFlowNet::init(net_config(dorm), &device());
        *net = fresh.load_record(mrec);
        *optim = optim.clone().load_record(orec);
        return step;
    }
    0
}
