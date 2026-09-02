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
    #[arg(long, default_value="false", help="on NaN loss / panic: wait 30s, re-exec this process (fresh CUDA context) and resume from the last checkpoint")] guard: bool,
}

fn run(a: Args) -> Result<(), String> {
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
    )
}

fn main() {
    let a = Args::parse();
    let guard = a.guard;
    // Panics (e.g. CUDA OOM deep in cubecl) must reach the guard too, so the
    // run is wrapped in catch_unwind; a re-exec'd process rebuilds the CUDA
    // context and memory pools - neither is safe to reuse after a device
    // error, which is why the guard re-launches instead of looping in-process.
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || run(a)))
        .unwrap_or_else(|p| {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into());
            Err(format!("panic: {msg}"))
        });
    match res {
        Ok(()) => {}
        Err(e) => {
            eprintln!("train failed: {e}");
            if guard {
                eprintln!("guard: restarting in 30s (resume from last checkpoint)");
                std::thread::sleep(std::time::Duration::from_secs(30));
                use std::os::unix::process::CommandExt;
                let exe = std::env::current_exe().expect("current_exe");
                let err = std::process::Command::new(exe)
                    .args(std::env::args_os().skip(1))
                    .exec();
                eprintln!("guard: re-exec failed: {err}");
            }
            std::process::exit(1);
        }
    }
}
