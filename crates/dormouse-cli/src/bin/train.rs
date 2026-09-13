//! dormouse train - one command, CUDA by default. Every runtime knob is a
//! typed flag; there is no env-var surface.
use clap::Parser;
use dormouse_core::ActQuant;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Data directory (streamed byte corpus).
    #[arg(long, alias = "data-dir")]
    data: String,
    /// Held-out eval directory (optional).
    #[arg(long)]
    eval: Option<String>,
    /// Model preset: nano | small | base | one_b.
    #[arg(long, default_value = "small")]
    preset: String,

    #[arg(long, default_value = "100000")]
    steps: usize,
    #[arg(long, default_value = "512")]
    seq_len: usize,
    #[arg(long, default_value = "3")]
    batch: usize,
    #[arg(long, default_value = "100")]
    log_every: usize,
    /// Checkpoint every N steps (0 = off; the final model is always saved).
    #[arg(long, default_value = "1000")]
    ckpt_every: usize,
    #[arg(long, default_value = "0.0001")]
    lr: f64,
    #[arg(long, default_value = "0.01")]
    wd: f64,
    #[arg(long, default_value = "1.0")]
    grad_clip: f64,
    /// Checkpoint file name (<name>.bin in --ckpt-dir; resume reuses it).
    #[arg(long, default_value = "latest")]
    ckpt_name: String,
    #[arg(long, default_value = "checkpoints")]
    ckpt_dir: String,
    /// Eval every N steps (0 = off).
    #[arg(long, default_value = "0")]
    eval_every: usize,

    // --- optimizer ---
    /// Optimizer: mix | mix-adan | adamw | adan | muon.
    #[arg(long, default_value = "mix")]
    opt: String,
    /// A/B: drop expert TSCT factors from the Muon+ group to the fallback.
    #[arg(long, default_value = "false")]
    factors_fallback: bool,

    // --- model knobs (default = preset value) ---
    /// bf16 storage mode: activations bf16, compute fp32 (tensor cores).
    #[arg(long, default_value = "false")]
    bf16: bool,
    /// Activation quantization: int4 | int8 | fp4 (BitNet a4.8 style).
    #[arg(long, allow_hyphen_values = true)]
    act_quant: Option<String>,
    /// Activation-quant scale group size (0 = per-token).
    #[arg(long)]
    act_group: Option<usize>,
    /// PonderNet loop depth (default: preset).
    #[arg(long)]
    max_iter: Option<usize>,
    /// Disable an arm for A/B.
    #[arg(long, default_value = "false")]
    no_kda: bool,
    #[arg(long, default_value = "false")]
    no_msa: bool,
    #[arg(long, default_value = "false")]
    no_engram: bool,

    // --- auxiliary objectives (helpers on top of CE) ---
    /// JEPA aux weight (EMA-teacher masked latent prediction). 0 = off;
    /// default: preset (0.05).
    #[arg(long)]
    jepa_weight: Option<f32>,
    /// DSpark aux weight (DeepSeek draft head, instead of MTP). 0 = off;
    /// default: preset (0.1).
    #[arg(long)]
    dspark_weight: Option<f32>,
    /// DSpark draft depth K (default: preset, 4).
    #[arg(long)]
    dspark_k: Option<usize>,
    /// Offline JEPA targets: precomputed teacher-latent sidecar (from
    /// --jepa-precompute). Set = no per-step EMA teacher forward, no EMA
    /// advance.
    #[arg(long)]
    jepa_targets: Option<PathBuf>,
    /// Precompute N steps of JEPA teacher targets into --jepa-targets and
    /// exit (one forward per batch, current weights, no training).
    #[arg(long)]
    jepa_precompute: Option<usize>,

    // --- quantization / maintenance ---
    /// Force factor-quant format: fp32 | bf16 | fp16 | fp8 | fp4.
    #[arg(long)]
    quant: Option<String>,
    /// TSCT retraction cadence / Newton-Schulz iterations.
    #[arg(long, default_value = "1")]
    retract_every: usize,
    #[arg(long, default_value = "3")]
    retract_iters: usize,

    // --- stability protocol (report §3.3) ---
    #[arg(long, default_value = "false")]
    stress: bool,
    /// Constant-LR multiplier for the stress protocol (2x / 4x).
    #[arg(long, default_value = "1.0")]
    stress_lr: f64,
    #[arg(long, default_value = "50")]
    stress_every: usize,

    // --- memory ---
    /// Host-RAM n-gram tables (CPU Adam, prefetched rows).
    #[arg(long, default_value = "false")]
    engram_ram: bool,
    #[arg(long, default_value = "1000000")]
    engram_slots: usize,
    /// Host-table Adam cadence in steps (default 1 = every step; 0 = off).
    #[arg(long, default_value = "1")]
    host_adam_every: usize,

    // --- diagnostics ---
    /// Per-step GPU/CPU time split every 50 steps (forces a sync).
    #[arg(long, default_value = "false")]
    timers: bool,
    /// cubecl pool stats at log cadence (forces a sync).
    #[arg(long, default_value = "false")]
    memlog: bool,
    /// One-off quant-fidelity probe on the first step.
    #[arg(long, default_value = "false")]
    quant_check: bool,
    /// cubecl autotune level (minimal | medium | full); passed to the
    /// runtime, which reads it from the process environment.
    #[arg(long)]
    autotune: Option<String>,

    // --- process ---
    /// Append stdout+stderr to this file.
    #[arg(long)]
    log: Option<String>,
    /// Daemonize: ignore SIGHUP, fork to background.
    #[arg(long, default_value = "false")]
    detach: bool,
    /// On NaN loss / panic: wait 30s, re-exec (fresh CUDA context) and
    /// resume from the last checkpoint.
    #[arg(long, default_value = "false")]
    guard: bool,
}

fn parse_act_quant(v: &str) -> Result<ActQuant, String> {
    match v {
        "fp4" => Ok(ActQuant::Fp4),
        "4" => Ok(ActQuant::Int(4)),
        "8" => Ok(ActQuant::Int(8)),
        other => Err(format!("--act-quant: expected 4 | 8 | fp4, got {other:?}")),
    }
}

fn run(a: Args) -> Result<(), String> {
    let cfg = dormouse_train::TrainCfg {
        steps: a.steps,
        ckpt_every: a.ckpt_every,
        seq_len: a.seq_len,
        batch: a.batch,
        lr: a.lr,
        wd: a.wd,
        grad_clip: a.grad_clip,
        log_every: a.log_every,
        ckpt_name: a.ckpt_name,
        eval_every: a.eval_every,
        opt: a.opt,
        quant: a.quant,
        factors_fallback: a.factors_fallback,
        retract_every: a.retract_every,
        retract_iters: a.retract_iters,
        stress: a.stress,
        stress_lr: a.stress_lr,
        stress_every: a.stress_every,
        engram_ram: a.engram_ram,
        engram_slots: a.engram_slots,
        host_adam_every: a.host_adam_every,
        warmup: true,
        quant_check: a.quant_check,
        timers: a.timers,
        memlog: a.memlog,
        bf16: a.bf16,
        act_quant: a.act_quant.as_deref().map(parse_act_quant).transpose()?,
        act_group: a.act_group,
        max_iter: a.max_iter,
        no_kda: a.no_kda,
        no_msa: a.no_msa,
        no_engram: a.no_engram,
        jepa_weight: a.jepa_weight,
        dspark_weight: a.dspark_weight,
        dspark_k: a.dspark_k,
        jepa_targets: a.jepa_targets,
        qk_heads: None,
    };
    if let Some(n) = a.jepa_precompute {
        let Some(out) = &cfg.jepa_targets else {
            return Err("--jepa-precompute requires --jepa-targets <output file>".into());
        };
        return dormouse_train::precompute_jepa_targets(
            &cfg,
            std::path::Path::new(&a.data),
            &a.preset,
            n,
            out,
        );
    }
    dormouse_train::train_loop(
        cfg,
        PathBuf::from(a.data),
        a.preset,
        Some(PathBuf::from(a.ckpt_dir)),
        a.eval.map(PathBuf::from),
    )
}

fn main() {
    let a = Args::parse();

    // Daemonize first: SIGHUP ignored + background fork, so the training
    // process outlives its terminal.
    if a.detach {
        #[cfg(unix)]
        unsafe {
            libc::signal(libc::SIGHUP, libc::SIG_IGN);
            if libc::fork() > 0 {
                std::process::exit(0);
            }
            libc::setsid();
        }
    }

    // Log redirection: both streams into one append-only file. Must happen
    // before the guard re-exec too - dup2'd fds survive exec.
    if let Some(path) = &a.log {
        #[cfg(unix)]
        match std::fs::OpenOptions::new().create(true).append(true).open(path) {
            Ok(f) => unsafe {
                use std::os::unix::io::AsRawFd;
                libc::dup2(f.as_raw_fd(), 1);
                libc::dup2(f.as_raw_fd(), 2);
            },
            Err(e) => eprintln!("--log {path}: {e}"),
        }
    }

    // The one remaining environment knob: cubecl's autotuner is configured
    // via env by the runtime itself, before any device is initialized.
    if let Some(level) = &a.autotune {
        std::env::set_var("CUBECL_AUTOTUNE_LEVEL", level);
    }

    let guard = a.guard;
    // Resume state must exist for a restart to make sense: a fresh run that
    // dies before its first checkpoint (bad config, OOM at startup) would
    // otherwise re-exec into the same crash forever.
    let resumable = std::path::Path::new(&a.ckpt_dir)
        .join(format!("{}.bin", a.ckpt_name))
        .is_file();
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
            if guard && resumable {
                eprintln!("guard: restarting in 30s (resume from last checkpoint)");
                std::thread::sleep(std::time::Duration::from_secs(30));
                use std::os::unix::process::CommandExt;
                let exe = std::env::current_exe().expect("current_exe");
                let err = std::process::Command::new(exe)
                    .args(std::env::args_os().skip(1))
                    .exec();
                eprintln!("guard: re-exec failed: {err}");
            } else if guard {
                eprintln!("guard: no resumable checkpoint, giving up");
            }
            std::process::exit(1);
        }
    }
}
