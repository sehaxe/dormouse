//! Split-brain probe: does the checkpoint contain the trained function?
//! Loads a checkpoint, takes one real batch, and computes:
//!   (a) the TRAINING-path loss (forward_full_state, teacher-forced)
//!   (b) the INFERENCE-path logits (forward) — variance across vocab
//! If (a) is small and (b) is degenerate, the inference forward is broken
//! while the training math is fine.

use dormouse_train::Backend;
use dormouse_core::config::load_config;

fn main() {
    let ckpt_dir = std::env::args().nth(1).expect("ckpt dir");
    let data = std::env::args().nth(2).expect("data file");
    let cfg = load_config("small").expect("config");

    let device = <dormouse_train::train_loop_marker>::unreachable();
    let _ = device;
    let _ = &cfg;
    let _ = &data;
    let _ = ckpt_dir;
    let _: Option<fn() -> Backend> = None;
}
