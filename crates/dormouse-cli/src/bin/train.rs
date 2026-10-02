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
    /// Model preset: nano | small | swift50 | base | one_b | mor | nano-fused | p150
    /// (a name in configs/ or a path to a .toml; all eight files load).
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
    /// Train ByteFlow Net (arXiv 2603.03583) instead of the DormouseModel:
    /// the coding-rate patcher arm. Hyperparams via --set byteflow_*.
    #[arg(long)]
    byteflow: bool,

    // --- auxiliary objectives (helpers on top of CE) ---
    /// JEPA aux weight (EMA-teacher masked latent prediction). 0 = off;
    /// default: preset (0.05).
    #[arg(long)]
    jepa_weight: Option<f32>,
    /// DSpark aux weight (DeepSeek draft head, instead of MTP). 0 = off;
    /// default: preset (0.0 — it ships off in every preset and in the schema;
    /// turn it on with `--set dspark_weight=0.1`). No DSpark measurement
    /// exists: the window-shift fix of 2026-09-29 voided every earlier one.
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
    /// inference stays max_iter; there is no flag to lower it.
    #[arg(long)]
    rand_depth: bool,
    /// Batches averaged per held-out eval. The scored window is
    /// `n × batch × seq_len` BYTES — 20 × 10 × 512 = 102 400, 20 × 2 × 512 =
    /// 20 480 — NOT a fixed size: it is rewound before every eval, so two
    /// evals of one run agree, but two runs at different batch sizes scored
    /// different amounts of text. The byte count on the eval line is the
    /// authority; quote it with the number (AGENTS.md 2.6).
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
    /// Host-RAM n-gram tables (CPU Nesterov+Sinkhorn (the flag name is historical), prefetched rows).
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
    /// Capture ONE stage - the EMA teacher's forward (the JEPA latents) - into
    /// a CUDA graph and replay it every step (one dispatch instead of ~4.4k
    /// launches). Off by default; the stage is bit-exact, so the objective is
    /// unchanged. Requires JEPA on and --no-engram; refused with
    /// --graph-capture, --engram-ram and --jepa-targets.
    #[arg(long)]
    graph_stage: bool,
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
    train.byteflow |= a.byteflow;
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
    train.graph_stage |= a.graph_stage;
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

    // Resolve the config BEFORE detaching: a bad preset, --set
    // key or flag value must fail in the foreground where the user can see
    // it, not in a detached log file.
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

    // Log redirection: both streams into one append-only file.
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

    // No auto-restart: a failure exits non-zero and the operator decides.
    // The --guard re-exec wrapper (restart from a pinned image, capped at 3)
    // was removed 2026-10-02 - its detached parent exited 0, so orchestrators
    // read a dead run as a finished one and stacked 15 runs on one GPU. The
    // NaN firewall inside train_loop is unconditional and needs no wrapper.
    if let Err(e) = execute(&a, run) {
        eprintln!("train failed: {e}");
        std::process::exit(1);
    }
}
