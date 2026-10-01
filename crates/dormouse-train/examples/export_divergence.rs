//! Measure the export formats against the fp32 model on HELD-OUT text.
//!
//! This is the measurement ADR-0023's format decision rests on, and it is a
//! program rather than a paragraph so the numbers can be re-taken when the
//! model, the corpus or the format changes. It is NOT a gate: the gate is
//! `tests/export_roundtrip.rs`. This produces the constants that gate asserts,
//! and it trains its own short run because the export's whole subject is a
//! checkpoint nobody has.
//!
//! Two phases, one process, no GPU needed (the narrowing is host arithmetic, so
//! the measurement is backend-agnostic; this runs on the CPU backend so it can
//! never contend with a training run for the card):
//!
//! 1. TRAIN. `--steps` optimizer steps on the real corpus at `--preset`, with
//!    AdamW, then `save_ckpt` + the config snapshot `export_ckpt` reads. This is
//!    also the answer to "how long does a minimal checkpoint take to produce".
//! 2. MEASURE. Export at f32 / f16 / bf16, then compare each against the fp32
//!    checkpoint's logits on windows of the HELD-OUT eval file: max and mean
//!    |delta logit|, top-1 byte agreement (what decides generated text), and a
//!    greedy 64-step continuation compared byte for byte with fp32's.
//!
//! ```sh
//! cargo run --release -p dormouse-train --example export_divergence -- \
//!   --steps 20 --eval /mnt/.../real_eval_v2/eval_tail.bin --windows 16
//! ```

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::tensor::{Int, Tensor, TensorData};

use dormouse_core::{DormouseConfig, DormouseModel};
use dormouse_train::decode;
use dormouse_train::export::{self, DType};

#[derive(clap::Parser, Debug)]
struct Args {
    #[arg(long)]
    train: Option<PathBuf>,
    #[arg(long)]
    eval: PathBuf,
    #[arg(long, default_value = "nano")]
    preset: String,
    #[arg(long, default_value = "20")]
    steps: usize,
    #[arg(long, default_value = "8")]
    batch: usize,
    #[arg(long, default_value = "512")]
    seq_len: usize,
    #[arg(long, default_value = "1e-4")]
    lr: f64,
    /// Skip phase 1 and measure an existing `--ckpt-dir/--ckpt-name`.
    #[arg(long)]
    ckpt_dir: Option<PathBuf>,
    #[arg(long, default_value = "measured")]
    ckpt_name: String,
    #[arg(long, default_value = "16")]
    windows: usize,
}

fn main() {
    let a = <Args as clap::Parser>::parse();
    let dir = a
        .ckpt_dir
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("dm-export-divergence"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("ckpt dir");

    let ckpt = dir.join(format!("{}.bin", a.ckpt_name));
    let cfg: DormouseConfig = if a.train.is_some() {
        train_n_steps(&a, &dir, &ckpt)
    } else {
        let snap = dir.join(format!("{}.config.toml", a.ckpt_name));
        let text = std::fs::read_to_string(&snap).expect("the run wrote a config snapshot");
        dormouse_train::RunCfg::from_snapshot(&text)
            .expect("the snapshot parses")
            .model
    };

    let bytes = std::fs::read(&ckpt).expect("the checkpoint exists");
    let fp32 = dormouse_train::load_model_weights(&dir, &a.ckpt_name, cfg.clone())
        .expect("the checkpoint loads")
        .no_grad();
    let n_params: u64 = export::tensors_of(&fp32)
        .iter()
        .map(|(_, _, f)| f.len() as u64)
        .sum();

    // Held-out windows, evenly spaced through the eval file so the sample is
    // the whole tail and not its first kilobyte.
    let eval = std::fs::read(&a.eval).expect("the eval file exists");
    let n = a.windows.min(eval.len() / a.seq_len);
    let stride = (eval.len() - a.seq_len) / n.max(1);
    let windows: Vec<&[u8]> = (0..n)
        .map(|i| &eval[i * stride..i * stride + a.seq_len])
        .collect();

    let ref_logits: Vec<Vec<f32>> = windows
        .iter()
        .map(|w| decode::next_byte_logits::<dormouse_train::Backend>(&fp32, w))
        .collect();
    assert!(
        ref_logits.iter().flatten().all(|x| x.is_finite()),
        "the fp32 reference must be finite or every number below is noise"
    );
    let vocab = ref_logits[0].len();
    let absmax = ref_logits[0].iter().fold(0.0f32, |a, b| a.max(b.abs()));

    println!(
        "checkpoint   {} ({} B container)",
        ckpt.display(),
        bytes.len()
    );
    println!(
        "preset       {}  params {n_params}  tensors {}",
        a.preset,
        export::tensors_of(&fp32).len()
    );
    println!(
        "held-out     {} windows x {} B from {}",
        windows.len(),
        a.seq_len,
        a.eval.display()
    );
    println!("logits       |max| {absmax:.3} (vocab {vocab})");
    println!();
    println!(
        "{:>5} {:>12} {:>8} {:>13} {:>13} {:>7} {:>9}",
        "dtype", "bytes", "vs ckpt", "max |dlogit|", "mean |dlogit|", "top-1", "greedy"
    );

    for dtype in [DType::F32, DType::F16, DType::Bf16] {
        let out = dir.join(format!(
            "{}.{}.dmexp",
            a.ckpt_name,
            format!("{dtype:?}").to_lowercase()
        ));
        let r = export::export_ckpt(&ckpt, Some(&a.preset), &[], dtype, &out)
            .unwrap_or_else(|e| panic!("{dtype:?}: {e}"));
        let (m, _, _) = export::read(&out).unwrap_or_else(|e| panic!("{dtype:?}: {e}"));

        let (mut max_d, mut sum_d, mut count, mut top1) = (0.0f32, 0.0f64, 0u64, 0u64);
        for (w, r) in windows.iter().zip(&ref_logits) {
            let l = decode::next_byte_logits::<dormouse_train::Backend>(&m, w);
            for (x, y) in l.iter().zip(r) {
                max_d = max_d.max((x - y).abs());
                sum_d += (x - y).abs() as f64;
                count += 1;
            }
            if argmax(&l) == argmax(r) {
                top1 += 1;
            }
        }
        // The end-to-end question: does the format change the TEXT? Greedy,
        // because a sampled comparison confounds the format with the RNG.
        let same = greedy(&fp32, windows[0], 64) == greedy(&m, windows[0], 64);
        println!(
            "{dtype:>5?} {:>12} {:>7.1}x {:>13.3e} {:>13.3e} {:>6.1}% {:>9}",
            r.file_bytes,
            bytes.len() as f64 / r.file_bytes as f64,
            max_d,
            sum_d / count as f64,
            100.0 * top1 as f64 / windows.len() as f64,
            if same { "same" } else { "DIFFERS" },
        );
        println!(
            "      max|w| {:.4e}  min|w|>0 {:.4e}  flushed {}  (f16 range: max 65504, min normal 6.10e-5)",
            r.max_abs, r.min_nonzero_abs, r.flushed
        );
    }
    println!();
    println!("files in {}", dir.display());
}

/// Phase 1: a short real run. This is a real checkpoint, produced by the real
/// trainer, at a real step count - the point is a model whose weights have
/// moved off their init distribution, not a converged one.
fn train_n_steps(a: &Args, dir: &PathBuf, ckpt: &Path) -> DormouseConfig {
    use dormouse_train::{build_optim, save_ckpt, RunCfg, TrainCfg};
    let t0 = std::time::Instant::now();
    let cfg: DormouseConfig = {
        let mut c = dormouse_core::config::load_config(&a.preset).expect("the preset loads");
        // Pure CE: the aux heads cost a second forward and change nothing about
        // the export, and one of them needs the teacher path this measurement
        // does not want in the way.
        c.jepa_weight = 0.0;
        c.dspark_weight = 0.0;
        c
    };
    let train = TrainCfg {
        steps: a.steps,
        batch: a.batch,
        seq_len: a.seq_len,
        lr: a.lr,
        opt: "adamw".into(),
        ..Default::default()
    };
    let corpus = std::fs::read(a.train.as_ref().expect("--train is required to train"))
        .expect("the corpus exists");
    let dev = dormouse_train::device();
    println!(
        "training     {} batch {} seq {} steps {} on {} B of corpus",
        a.preset,
        a.batch,
        a.seq_len,
        a.steps,
        corpus.len()
    );
    let mut model = DormouseModel::new(&cfg, &dev);
    let mut optim = build_optim(&model, &train);
    let chunk = a.batch * a.seq_len;
    for step in 0..a.steps {
        let off = (step * chunk) % corpus.len().saturating_sub(chunk + 1);
        let bytes = &corpus[off..off + chunk];
        let (x, h, y) = batch(bytes, &dev);
        let (_logits, rec, _kda, _aux) =
            model.forward_with_hidden::<dormouse_train::Backend>(x, Some(h), None, Some(y), None);
        let loss = model.loss::<dormouse_train::Backend>(rec);
        let l: f32 = loss
            .clone()
            .into_data()
            .try_to_vec()
            .expect("loss readback")[0];
        let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &model);
        // The polar retraction the trainer does every step: the TSCT U/V
        // masters drift without it and the forward degenerates. (A 20-step run
        // at lr 2e-2 without it produced a NaN checkpoint at step 10 - that
        // number is why this line is here.) Clipping is not needed: it is
        // applied inside `optim.step` by the policy, not here.
        model.retract_tsct(train.retract_iters);
        model = optim.step(a.lr, model, grads);
        assert!(
            l.is_finite(),
            "step {step}: loss is {l} - these weights are not exportable"
        );
        println!(
            "  step {step:>3} loss {l:.4}  ({:.1}s)",
            t0.elapsed().as_secs_f32()
        );
    }
    save_ckpt(
        dir,
        &ckpt.file_stem().unwrap().to_string_lossy(),
        &model,
        &optim,
        None,
        false,
        a.steps as u64,
        0.0,
    )
    .expect("the checkpoint saves");
    let name = ckpt.file_stem().unwrap().to_string_lossy().to_string();
    let run = RunCfg {
        source: a.preset.clone(),
        model: cfg.clone(),
        train,
    };
    std::fs::write(dir.join(format!("{name}.config.toml")), run.snapshot_toml())
        .expect("the snapshot writes");
    println!(
        "checkpoint   {} written in {:.1}s",
        ckpt.display(),
        t0.elapsed().as_secs_f32()
    );
    cfg
}

/// One batch: bytes in, the shifted-label batch, and the FNV n-gram keys the
/// Engram arm indexes with.
///
/// The keys come from `dormouse_data::raw_keys` — the SAME derivation the
/// trainer and the decode seam use. This used to be a third hand-rolled FNV
/// over 3/5/8-grams reduced mod 4096, which is a different key space from the
/// one the model was trained on: a 20-step model trained through it, and a
/// measured "export divergence" computed against a decode path that read no
/// rows at all. One derivation, one key space.
fn batch(
    bytes: &[u8],
    dev: &burn::tensor::Device,
) -> (Tensor<2, Int>, Tensor<3, Int>, Tensor<2, Int>) {
    let (b, t) = (1usize, bytes.len());
    let ids: Vec<i64> = bytes.iter().map(|&x| x as i64).collect();
    let y: Vec<i64> = bytes
        .iter()
        .skip(1)
        .map(|&x| x as i64)
        .chain(std::iter::once(bytes[0] as i64))
        .collect();
    (
        Tensor::from_data(TensorData::new(ids, [b, t]), dev),
        Tensor::from_data(
            TensorData::new(dormouse_data::raw_keys(bytes), [b, t, 3]),
            dev,
        ),
        Tensor::from_data(TensorData::new(y, [b, t]), dev),
    )
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &x)| {
            if x > acc.1 {
                (i, x)
            } else {
                acc
            }
        })
        .0
}

/// Greedy (argmax) continuation - no temperature, no RNG, so two models can be
/// compared on the format alone.
fn greedy(m: &DormouseModel, prompt: &[u8], steps: usize) -> Vec<u8> {
    let mut b = prompt.to_vec();
    for _ in 0..steps {
        let l = decode::next_byte_logits::<dormouse_train::Backend>(m, &b);
        b.push(argmax(&l) as u8);
        if b.len() > m.max_seq_len() {
            break;
        }
    }
    b
}
