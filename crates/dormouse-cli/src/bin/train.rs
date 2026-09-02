//! dormouse train - one command, CUDA by default
use clap::Parser;
#[derive(Parser, Debug)]
struct Args {
    #[arg(long, alias="data-dir")] data: String,
    #[arg(long, default_value="small")] preset: String,
    #[arg(long, default_value="100000")] steps: usize,
    #[arg(long, default_value="512")] seq_len: usize,
    #[arg(long, default_value="3")] batch: usize,
    #[arg(long, default_value="100")] log_every: usize,
    #[arg(long, default_value="1000", help="checkpoint every N steps (0 = off, final model still saved)")] ckpt_every: usize,
    #[arg(long, default_value="0.0001")] lr: f64,
    #[arg(long, default_value="0.01")] wd: f64,
    #[arg(long, default_value="1.0")] grad_clip: f64,
    #[arg(long, default_value="latest", help="checkpoint file name (<name>.bin)")] ckpt_name: String,
    #[arg(long, default_value="checkpoints")] ckpt_dir: String,
    #[arg(long, help="held-out eval data dir (separate from --data)")] eval: Option<String>,
    #[arg(long, default_value="0", help="eval every N steps (0 = off)")] eval_every: usize,
}
fn main() {
    let a = Args::parse();
    let cfg = dormouse_train::TrainCfg {
        steps: a.steps, ckpt_every: a.ckpt_every,
        seq_len: a.seq_len, batch: a.batch, lr: a.lr, wd: a.wd, grad_clip: a.grad_clip,
        log_every: a.log_every,
        ckpt_name: a.ckpt_name, eval_every: a.eval_every,
    };
    dormouse_train::train_loop(
        cfg,
        std::path::PathBuf::from(a.data),
        a.preset,
        Some(std::path::PathBuf::from(a.ckpt_dir)),
        a.eval.map(std::path::PathBuf::from),
    );
}