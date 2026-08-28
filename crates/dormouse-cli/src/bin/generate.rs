//! dormouse generate - inference from a named ckpt
use clap::Parser;
use rand::Rng;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "checkpoints")] ckpt_dir: PathBuf,
    #[arg(long, default_value = "latest", help = "checkpoint file name (<name>.bin)")] ckpt_name: String,
    #[arg(long, default_value = "hello")] prompt: String,
    #[arg(long, default_value = "32")] steps: usize,
    #[arg(long, default_value = "small")] preset: String,
    #[arg(long, default_value = "0.8")] temp: f32,
}

fn main() {
    let a = Args::parse();
    let cfg = match a.preset.as_str() {
        "base" => dormouse_core::DormouseConfig::base(),
        "one_b" => dormouse_core::DormouseConfig::one_b(),
        _ => dormouse_core::DormouseConfig::small(),
    };
    let model = dormouse_train::load_model_weights(&a.ckpt_dir, &a.ckpt_name, cfg)
        .unwrap_or_else(|| { eprintln!("ckpt not found: {}/{}.bin", a.ckpt_dir.display(), a.ckpt_name); std::process::exit(1); });
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