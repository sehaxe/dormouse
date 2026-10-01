//! dormouse train - one command, CUDA by default. Every runtime knob is a
//! typed flag; there is no env-var surface. Config-valued flags are
//! Option-only (ADR-0005): the defaults live in the schema types, the flags
//! are layer overrides, and `resolve` merges preset -> --set -> flags.
use clap::Parser;
use dormouse_core::ActQuant;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Parser, Debug)]
struct Args {
    /// Data directory (streamed byte corpus).
    #[arg(long, alias = "data-dir")]
    data: String,
    /// Held-out eval directory (optional).
    #[arg(long)]
    eval: Option<String>,
    /// Model preset: nano | small | swift50 | base | one_b (name or path to .toml).
    #[arg(long, default_value = "small")]
    preset: String,
    /// Config file path (alias for --preset when a path is given).
    #[arg(long)]
    config: Option<String>,
    /// Override config key=value (e.g. --set max_iter=12 --set d_model=512). Can be repeated.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    set: Vec<String>,

    #[arg(long)]
    steps: Option<usize>,
    #[arg(long)]
    seq_len: Option<usize>,
    #[arg(long)]
    batch: Option<usize>,
    #[arg(long)]
    log_every: Option<usize>,
    /// Checkpoint every N steps (0 = off; the final model is always saved).
    #[arg(long)]
    ckpt_every: Option<usize>,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(long)]
    wd: Option<f64>,
    #[arg(long)]
    grad_clip: Option<f64>,
    /// Checkpoint file name (`<name>.bin` in --ckpt-dir; resume reuses it).
    #[arg(long)]
    ckpt_name: Option<String>,
    #[arg(long, default_value = "checkpoints")]
    ckpt_dir: String,
    /// Eval every N steps (0 = off).
    #[arg(long)]
    eval_every: Option<usize>,

    // --- optimizer ---
    /// Optimizer: mix | mix-adan | adamw | adan | muon.
    #[arg(long)]
    opt: Option<String>,
    /// A/B: drop expert TSCT factors from the Muon+ group to the fallback.
    #[arg(long)]
    factors_fallback: bool,

    // --- model knobs (absent = preset value) ---
    /// bf16 storage mode: activations bf16, compute fp32 (tensor cores).
    /// Bare --bf16 = true, --bf16 false disables, absent = preset value.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    bf16: Option<bool>,
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
    #[arg(long)]
    no_kda: bool,
    #[arg(long)]
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
    /// Seed for the model initialisation AND the JEPA span mask. This text said
    /// "the mask, the only stochastic input to a step" until 2026-09-29; that
    /// stopped being true in 4b42b6d, which added `device.seed(cfg.seed)`
    /// before any parameter is created — before it every parameter drew from
    /// process entropy, so two runs of one config were two different models.
    /// Both are pure functions of (seed, step), so an A/B of two arms replays
    /// the same init and the same masks, and a resume continues the interrupted
    /// run's sequence (ADR-0021). Part of the config snapshot: changing it
    /// mid-run is drift. What is and is not verified about this:
    /// `model_seam::two_models_one_seed_are_bit_identical`.
    #[arg(long)]
    seed: Option<u64>,

    /// Random-depth arm: sample the loop depth T in 1..=max_iter per step
    /// (deterministic from the step index) so the model is trained to be
    /// correct at EVERY depth. Adaptive depth without a learned halting head,
    /// so there is nothing to collapse (ADR-0013 rank 2). The depth at
    /// inference stays max_iter; --gen-max-iter picks it lower.
    #[arg(long)]
    rand_depth: bool,
    /// Batches averaged per held-out eval (20 = 100 KB). The eval window is
    /// fixed and rewound every time, so numbers are comparable across runs.
    #[arg(long)]
    eval_batches: Option<usize>,
    /// Also print the held-out BPB at depths 1..=max_iter at every eval (no
    /// extra training): measures depth-robustness and early-exit headroom.
    #[arg(long)]
    eval_depths: bool,

    // --- quantization / maintenance ---
    /// Force factor-quant format: fp32 | bf16 | fp16 | fp8 | fp4.
    #[arg(long)]
    quant: Option<String>,
    /// TSCT retraction cadence / Newton-Schulz iterations.
    #[arg(long)]
    retract_every: Option<usize>,
    #[arg(long)]
    retract_iters: Option<usize>,
    /// Retract the TSCT masters grouped by shape (one sync-free batched
    /// Newton-Schulz per group) instead of one per factor. Same numbers,
    /// fewer launches and no per-factor host syncs. Off = the per-factor
    /// path every run in the archive used.
    #[arg(long)]
    retract_batched: bool,

    // --- stability protocol (report §3.3) ---
    #[arg(long)]
    stress: bool,
    /// Constant-LR multiplier for the stress protocol (2x / 4x).
    #[arg(long)]
    stress_lr: Option<f64>,
    #[arg(long)]
    stress_every: Option<usize>,

    // --- memory ---
    /// Host-RAM n-gram tables (CPU Adam, prefetched rows).
    #[arg(long)]
    engram_ram: bool,
    #[arg(long)]
    engram_slots: Option<usize>,
    /// Host-table Adam cadence in steps (default 1 = every step; 0 = off).
    #[arg(long)]
    host_adam_every: Option<usize>,

    // --- diagnostics ---
    /// Per-step GPU/CPU time split every 50 steps (forces a sync).
    #[arg(long)]
    timers: bool,
    /// cubecl pool stats at log cadence (forces a sync).
    #[arg(long)]
    memlog: bool,
    /// One-off quant-fidelity probe on the first step.
    #[arg(long)]
    quant_check: bool,
    /// Capture the forward+backward window into a CUDA graph and replay it
    /// (one dispatch per step instead of ~14k launches). Off by default; the
    /// steps that read anything back run ungraphed, and the run reports
    /// captures / replays / refusals. Refused with --rand-depth, --engram-ram
    /// and --jepa-targets.
    #[arg(long)]
    graph_capture: bool,
    /// cubecl autotune level (minimal | medium | full); passed to the
    /// runtime, which reads it from the process environment.
    #[arg(long)]
    autotune: Option<String>,

    // --- process ---
    /// Append stdout+stderr to this file.
    #[arg(long)]
    log: Option<String>,
    /// Daemonize: ignore SIGHUP, fork to background.
    #[arg(long)]
    detach: bool,
    /// On NaN loss / panic: wait 30s, re-exec (fresh CUDA context) and
    /// resume from the last checkpoint.
    #[arg(long)]
    guard: bool,
}

/// Build the resolved run from the flags: defaults -> preset -> --set ->
/// typed flags -> validate (the one seam, `dormouse_train::resolve`). The
/// act-quant string parses through the schema's single `FromStr`.
fn build_run(a: &Args) -> Result<dormouse_train::RunCfg, String> {
    let preset_name = a.config.as_deref().unwrap_or(&a.preset);
    let mut train = dormouse_train::TrainCfg::default();
    // Config-valued flags are layer overrides: absent = schema default.
    train.steps = a.steps.unwrap_or(train.steps);
    train.ckpt_every = a.ckpt_every.unwrap_or(train.ckpt_every);
    train.log_every = a.log_every.unwrap_or(train.log_every);
    train.seq_len = a.seq_len.unwrap_or(train.seq_len);
    train.batch = a.batch.unwrap_or(train.batch);
    train.lr = a.lr.unwrap_or(train.lr);
    train.wd = a.wd.unwrap_or(train.wd);
    train.grad_clip = a.grad_clip.unwrap_or(train.grad_clip);
    train.ckpt_name = a.ckpt_name.clone().unwrap_or(train.ckpt_name);
    train.eval_every = a.eval_every.unwrap_or(train.eval_every);
    train.opt = a.opt.clone().unwrap_or(train.opt);
    train.quant = a.quant.clone().or(train.quant);
    train.factors_fallback |= a.factors_fallback;
    train.bf16 = a.bf16;
    train.act_quant = match &a.act_quant {
        Some(s) => Some(ActQuant::from_str(s).map_err(|e| format!("--act-quant: {e}"))?),
        None => train.act_quant,
    };
    train.act_group = a.act_group.or(train.act_group);
    train.max_iter = a.max_iter.or(train.max_iter);
    train.no_kda |= a.no_kda;
    train.no_engram |= a.no_engram;
    train.jepa_weight = a.jepa_weight.or(train.jepa_weight);
    train.dspark_weight = a.dspark_weight.or(train.dspark_weight);
    train.dspark_k = a.dspark_k.or(train.dspark_k);
    train.jepa_targets = a.jepa_targets.clone().or(train.jepa_targets);
    train.seed = a.seed.unwrap_or(train.seed);
    train.rand_depth |= a.rand_depth;
    train.eval_batches = a.eval_batches.unwrap_or(train.eval_batches);
    train.eval_depths |= a.eval_depths;
    train.retract_every = a.retract_every.unwrap_or(train.retract_every);
    train.retract_iters = a.retract_iters.unwrap_or(train.retract_iters);
    train.retract_batched = a.retract_batched || train.retract_batched;
    train.stress |= a.stress;
    train.stress_lr = a.stress_lr.unwrap_or(train.stress_lr);
    train.stress_every = a.stress_every.unwrap_or(train.stress_every);
    train.engram_ram |= a.engram_ram;
    train.engram_slots = a.engram_slots.unwrap_or(train.engram_slots);
    train.host_adam_every = a.host_adam_every.unwrap_or(train.host_adam_every);
    train.quant_check |= a.quant_check;
    train.timers |= a.timers;
    train.memlog |= a.memlog;
    train.graph_capture |= a.graph_capture;
    // warmup keeps its schema default (true) - no flag on purpose.
    dormouse_train::resolve(preset_name, &a.set, train)
}

fn execute(a: &Args, run: dormouse_train::RunCfg) -> Result<(), String> {
    if let Some(n) = a.jepa_precompute {
        let Some(out) = run.train.jepa_targets.clone() else {
            return Err("--jepa-precompute requires --jepa-targets <output file>".into());
        };
        return dormouse_train::precompute_jepa_targets(
            &run,
            std::path::Path::new(&a.data),
            n,
            &out,
        );
    }
    dormouse_train::train_loop(
        run,
        PathBuf::from(&a.data),
        Some(PathBuf::from(&a.ckpt_dir)),
        a.eval.clone().map(PathBuf::from),
    )
}

fn main() {
    let a = Args::parse();

    // Pin a stable executable image before anything else, but only when the
    // guard is in play. `cargo build-train` REPLACES target/release/train, so a
    // running process's /proc/self/exe resolves to a deleted inode and the
    // guard's re-exec fails with ENOENT - exactly when the recovery is needed
    // (measured 2026-09-27: official_v5f died, guard could not restart it, and
    // the crash loop went unnoticed). A sibling image file that cargo does not
    // know about survives rebuilds, and the re-exec points at it.
    if a.guard {
        const IMAGE_ENV: &str = "DORMOUSE_GUARD_EXE";
        if std::env::var(IMAGE_ENV).is_err() {
            let exe = std::env::current_exe().expect("current_exe");
            // A SUBDIRECTORY, and the file must stay named `train`: the
            // machine's ram-guard kills the heaviest process whose comm is
            // exactly "train" (~/bin/ram-guard.sh), so an image named
            // anything else would silently disable the memory guard for every
            // --guard run. cargo does not own this directory.
            let dir = exe.with_file_name("guard-image");
            let image = dir.join("train");
            // pid-suffixed tmp: two concurrent launches sharing a fixed
            // `train.tmp` can publish a truncated image through the rename.
            let tmp = dir.join(format!("train.tmp.{}", std::process::id()));
            // Best-effort: a read-only target/ must fall back to current_exe,
            // not panic before the config is even validated.
            let pinned = std::fs::create_dir_all(&dir).is_ok()
                && std::fs::copy(&exe, &tmp).is_ok()
                && std::fs::rename(&tmp, &image).is_ok();
            if !pinned {
                eprintln!("guard: could not pin an executable image, using the live path");
            }
            use std::os::unix::process::CommandExt;
            let err = std::process::Command::new(&image)
                .args(std::env::args_os().skip(1))
                .env(IMAGE_ENV, &image)
                .exec();
            eprintln!("guard image exec failed: {err}");
            std::process::exit(1);
        }
    }

    // Resolve the config BEFORE detach/guard wrapping: a bad preset, --set
    // key or flag value must fail in the foreground where the user can see
    // it, not in a detached log file or a guard re-exec loop.
    let run = match build_run(&a) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("train failed: {e}");
            std::process::exit(1);
        }
    };

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
        .join(format!("{}.bin", run.train.ckpt_name))
        .is_file();
    // Panics (e.g. CUDA OOM deep in cubecl) must reach the guard too, so the
    // run is wrapped in catch_unwind; a re-exec'd process rebuilds the CUDA
    // context and memory pools - neither is safe to reuse after a device
    // error, which is why the guard re-launches instead of looping in-process.
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || execute(&a, run)))
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
            let restarts: u32 = std::env::var("DORMOUSE_GUARD_RESTARTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            // An in-loop Err is now reachable for a PERMANENT condition (a
            // non-finite held-out eval = a broken allocator), and the guard
            // re-exec'd every 30 s forever - which also manufactures the
            // "two GPU processes at once" that corrupted the pool in the first
            // place (review 2026-09-27). Cap the chain: after 3 restarts the
            // operator has to look.
            if guard && resumable && restarts < 3 {
                // The guard re-execs with the ORIGINAL argv, so the new
                // process re-runs resolve() on identical inputs. resolve is
                // deterministic in argv (no env or randomness feeds the
                // config), which is exactly what lets the on-disk snapshot
                // drift check pass across restarts.
                eprintln!(
                    "guard: restarting in 30s (resume from last checkpoint, restart {} of 3)",
                    restarts + 1
                );
                std::thread::sleep(std::time::Duration::from_secs(30));
                use std::os::unix::process::CommandExt;
                // The pinned image if we have one, else our own path (which
                // works as long as nothing rebuilt the binary under us).
                let exe = std::env::var("DORMOUSE_GUARD_EXE")
                    .unwrap_or_else(|_| std::env::current_exe().expect("current_exe").display().to_string());
                let err = std::process::Command::new(exe)
                    .args(std::env::args_os().skip(1))
                    .env("DORMOUSE_GUARD_RESTARTS", (restarts + 1).to_string())
                    .exec();
                eprintln!("guard: re-exec failed: {err}");
            } else if guard {
                eprintln!(
                    "guard: giving up (resumable={resumable}, restarts={restarts}) - a persistent failure needs a human"
                );
            }
            std::process::exit(1);
        }
    }
}
