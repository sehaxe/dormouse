//! dormouse generate - inference from an exported model file.
//!
//! It reads an INFERENCE EXPORT (`.dmexp`), never a training checkpoint. That
//! is a deliberate refusal, not a missing feature: a checkpoint is a run, with
//! the optimizer section and a `.ngram` sidecar that reaches 34 GB, and a
//! loader that accepts one here would be a loader that quietly reads 34 GB to
//! produce 30 MB of weights. `dormouse export` turns a checkpoint into this
//! file, and it names itself when handed the wrong one.
use clap::Parser;
use rand::Rng;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, help = "inference export written by `dormouse export` (<name>.dmexp)")] export: PathBuf,
    #[arg(long, default_value = "hello")] prompt: String,
    #[arg(long, default_value = "32")] steps: usize,
    #[arg(long, default_value = "0.8")] temp: f32,
}

fn main() {
    let a = Args::parse();
    let (model, _cfg, h) = dormouse_train::export::read(&a.export)
        .unwrap_or_else(|e| { eprintln!("generate: {e}"); std::process::exit(1); });
    println!(
        "loaded {} ({:?} weights, {} params, step {})",
        a.export.display(),
        h.dtype,
        h.num_params,
        h.step
    );
    let mut bytes = a.prompt.as_bytes().to_vec();
    println!("prompt: {}", a.prompt);
    let mut rng = rand::thread_rng();
    for _ in 0..a.steps {
        let logits = model.forward_bytes::<dormouse_train::Backend>(&bytes);
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|x| ((x - max)/a.temp).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut r = rng.gen_range(0.0..sum);
        let mut next = 0u8;
        for (i, e) in exp.iter().enumerate() {
            r -= e;
            if r <= 0.0 { next = i as u8; break; }
        }
        if next < 32 || next > 126 { next = 32 + (next % 95); }
        bytes.push(next);
        if bytes.len() > model.max_seq_len() { break; }
    }
    println!("gen: {}", String::from_utf8_lossy(&bytes));
}
