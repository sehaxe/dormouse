//! dormouse-train - CUDA training loop: Autodiff backend, Muon+ mixed
//! optimizer (see `optim`), burnpack checkpoints with custom name, resume,
//! opencode harness.
mod cfg;
mod jepa_targets;
mod offload;
mod optim;
mod stress;

use std::io::Write as _;
use std::path::{Path, PathBuf};

use burn::{
    backend::Backend as BurnBackend,
    module::Module,
    optim::{GradientsParams, OptimizerRecord},
    store::ModuleRecord,
    tensor::{Bytes, Device, Int, Tensor, TensorData},
};

use dormouse_core::{ActQuant, DormouseConfig, DormouseModel};

pub use cfg::{resolve, RunCfg};
pub use optim::{
    build_optim, is_engram_table_param, is_muon_param, validate_routing, GroupCounts, MUON_NS_STEPS,
};
pub use stress::{grad_norm, StressMonitor};

/// bits per byte from mean cross-entropy (nats).
pub fn bpb(ce: f32) -> f32 { ce / std::f32::consts::LN_2 }

#[cfg(feature = "cuda")]
pub type Backend = burn::backend::autodiff::Autodiff<
    burn_cuda::Cuda,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
#[cfg(all(feature = "cpu", not(feature = "cuda")))]
pub type Backend = burn::backend::autodiff::Autodiff<
    burn_ndarray::NdArray,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
#[cfg(not(any(feature = "cpu", feature = "cuda")))]
compile_error!("dormouse-train: enable a backend feature (cpu or cuda)");

pub type Optim = burn::optim::ModuleOptimizer;

/// The train-layer config. Defaults live ONLY in [`TrainCfg::default`]; the
/// serde impls fill missing fields from it (ADR-0005), so snapshots and
/// programmatic construction share one written copy.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TrainCfg {
    pub steps: usize,
    pub ckpt_every: usize,
    pub log_every: usize,
    pub seq_len: usize,
    pub batch: usize,
    pub lr: f64,
    pub wd: f64,
    pub grad_clip: f64,
    /// checkpoint file name (default "latest"), saved as `<name>.bin`
    pub ckpt_name: String,
    /// eval cadence in steps (0 = off). When >0 and an eval data dir is given,
    /// a held-out cross-entropy is reported every `eval_every` steps.
    pub eval_every: usize,
    /// Optimizer mode: "mix" | "mix-adan" | "adamw" | "adan" | "muon".
    pub opt: String,
    /// Forced quantization format ("fp32"|"bf16"|"fp16"|"fp8"|"fp4");
    /// None = auto (sm_120 -> Fp8, bf16 mode -> Bf16, else Fp32).
    pub quant: Option<String>,
    /// Drop expert TSCT factors from the Muon+ group to the fallback (A/B).
    pub factors_fallback: bool,
    /// TSCT U/V retraction cadence / Newton-Schulz iterations.
    pub retract_every: usize,
    pub retract_iters: usize,
    /// Stress protocol (report §3.3): constant LR at `stress_lr`x, spike
    /// counter + p99.9 grad norm logged every `stress_every` steps.
    pub stress: bool,
    pub stress_lr: f64,
    pub stress_every: usize,
    /// RAM-offload Engram tables (host memory, CPU Adam) with slot count.
    pub engram_ram: bool,
    pub engram_slots: usize,
    /// Host-table Adam cadence in steps (default 1 = every step, the
    /// report's rule; 0 = off). Each update syncs the row grads D2H, so
    /// values >1 trade table freshness for fewer syncs.
    pub host_adam_every: usize,
    /// Two fwd+bwd warmup steps before the loop (raises the pool high-water
    /// early); keep on for long runs.
    pub warmup: bool,
    /// One-off quant-fidelity probe on the first step (debug).
    pub quant_check: bool,
    /// Print per-step GPU/CPU time split every 50 steps (forces a sync).
    pub timers: bool,
    /// Print cubecl pool stats at log cadence (forces a sync).
    pub memlog: bool,
    /// bf16 storage override: Some forces the model's bf16 mode, None keeps
    /// the preset value (the `--set bf16=` layer sits below this).
    pub bf16: Option<bool>,
    /// Model-config overrides; `None` keeps the preset value.
    pub act_quant: Option<ActQuant>,
    pub act_group: Option<usize>,
    pub max_iter: Option<usize>,
    /// Disable a model arm for A/B (KDA / MSA / Engram).
    pub no_kda: bool,
    pub no_msa: bool,
    pub no_engram: bool,
    /// Auxiliary-loss weight overrides; None keeps the preset value.
    pub jepa_weight: Option<f32>,
    pub dspark_weight: Option<f32>,
    pub dspark_k: Option<usize>,
    /// Head-wise Muon for the attention Q/K projections: `Some(n_heads)`
    /// enables the per-head groups (derived from the model config in
    /// `resolve`, ADR-0005).
    pub qk_heads: Option<usize>,
    /// Offline JEPA targets (precomputed EMA-teacher latents per training
    /// chunk). When set, the hot loop runs NO second teacher forward and no
    /// EMA advance; targets are looked up per batch by chunk hash.
    pub jepa_targets: Option<PathBuf>,
}

impl Default for TrainCfg {
    /// The single written copy of the train-layer defaults; the serde
    /// deserialization (container-level `#[serde(default)]`) fills missing
    /// fields from this.
    fn default() -> Self {
        Self {
            steps: 100000, ckpt_every: 1000, log_every: 100,
            seq_len: 512, batch: 3, lr: 1e-4, wd: 0.01, grad_clip: 1.0,
            ckpt_name: "latest".into(), eval_every: 0,
            opt: "mix".into(), quant: None, factors_fallback: false,
            retract_every: 1, retract_iters: 3,
            stress: false, stress_lr: 1.0, stress_every: 50,
            engram_ram: false, engram_slots: 1_000_000, host_adam_every: 1,
            warmup: true, quant_check: false, timers: false, memlog: false,
            bf16: None, act_quant: None, act_group: None, max_iter: None,
            no_kda: false, no_msa: false, no_engram: false,
            jepa_weight: None, dspark_weight: None, dspark_k: None,
            qk_heads: None, jepa_targets: None,
        }
    }
}

fn device() -> Device {
    #[cfg(feature = "cuda")]
    { Device::cuda(0).autodiff() }
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    { Device::ndarray().autodiff() }
}

/// Pool introspection (debug): bytes reserved/used/live allocs on the CUDA client.
#[cfg(feature = "cuda")]
pub fn pool_stats(device: &Device) -> String {
    use burn_dispatch::DispatchDevice;
    use cubecl_cuda::CudaRuntime;
    use cubecl_runtime::client::ComputeClient;
    fn unwrap(d: &DispatchDevice) -> &burn_cuda::CudaDevice {
        match d {
            DispatchDevice::Cuda(dev) => dev,
            DispatchDevice::Autodiff(a) => match &**a {
                DispatchDevice::Cuda(dev) => dev,
                other => panic!("expected CUDA device, got {other:?}"),
            },
            other => panic!("expected CUDA device, got {other:?}"),
        }
    }
    let client = ComputeClient::<CudaRuntime>::load(unwrap(device.as_dispatch()));
    match client.memory_usage() {
        Ok(u) => format!("res={:.1}MB used={:.1}MB allocs={}", u.bytes_reserved as f64 / 1e6, u.bytes_in_use as f64 / 1e6, u.number_allocs),
        Err(_) => "mem-err".into(),
    }
}
#[cfg(not(feature = "cuda"))]
pub fn pool_stats(_device: &Device) -> String { "cpu".into() }

/// Return pooled VRAM pages to the driver. cubecl's pool is high-water: it
/// reserves pages for every allocation size ever seen and never frees them
/// on its own, so a long run eventually OOMs despite a low live footprint
/// (aria hit the same wall; periodic cleanup is the cheap fix).
#[cfg(feature = "cuda")]
pub fn memory_cleanup(device: &Device) {
    use burn_dispatch::DispatchDevice;
    use cubecl_cuda::CudaRuntime;
    use cubecl_runtime::client::ComputeClient;
    fn unwrap(d: &DispatchDevice) -> &burn_cuda::CudaDevice {
        match d {
            DispatchDevice::Cuda(dev) => dev,
            DispatchDevice::Autodiff(a) => match &**a {
                DispatchDevice::Cuda(dev) => dev,
                other => panic!("expected CUDA device, got {other:?}"),
            },
            other => panic!("expected CUDA device, got {other:?}"),
        }
    }
    let client = ComputeClient::<CudaRuntime>::load(unwrap(device.as_dispatch()));
    client.memory_cleanup();
}
#[cfg(not(feature = "cuda"))]
pub fn memory_cleanup(_device: &Device) {}

/// Must run BEFORE the first allocation on the device (model init).
/// ExclusivePages: one page per allocation, so `memory_cleanup` returns
/// freed pages to the driver; SubSlices keeps size buckets forever and a
/// graph-heavy autodiff loop (new shapes every step) OOMs at the high-water
/// mark.
#[cfg(feature = "cuda")]
pub fn init_pools(device: &Device) {
    use burn_dispatch::DispatchDevice;
    use cubecl_cuda::CudaRuntime;
    use cubecl_runtime::{
        client::ComputeClient,
        config::memory::{MemoryPoolsConfig, MemoryPoolsPreset},
    };
    fn unwrap(d: &DispatchDevice) -> &burn_cuda::CudaDevice {
        match d {
            DispatchDevice::Cuda(dev) => dev,
            DispatchDevice::Autodiff(a) => match &**a {
                DispatchDevice::Cuda(dev) => dev,
                other => panic!("expected CUDA device, got {other:?}"),
            },
            other => panic!("expected CUDA device, got {other:?}"),
        }
    }
    let client = ComputeClient::<CudaRuntime>::load(unwrap(device.as_dispatch()));
    let _ = client.install_memory_pools(&MemoryPoolsConfig::Preset(MemoryPoolsPreset::ExclusivePages));
}
#[cfg(not(feature = "cuda"))]
pub fn init_pools(_device: &Device) {}

/// WSD schedule (warmup 2% -> const -> 1.5-power decay over final 20%),
/// same shape as aria's schedule.rs.
pub fn wsd_factor(step: u64, total: u64, base_lr: f64) -> f64 {
    let total = total.max(1) as f64;
    let s = step as f64;
    let warmup = (total * 0.02).max(20.0);
    let decay_start = total * 0.8;
    if s < warmup {
        base_lr * (s / warmup)
    } else if s < decay_start {
        base_lr
    } else {
        let frac = ((s - decay_start) / (total - decay_start).max(1.0)).clamp(0.0, 1.0);
        base_lr * (1.0 - frac).powf(1.5)
    }
}

/// Pick the quantization format for the current device:
/// sm_120 (Blackwell) -> Fp8 by default (Fp4 via --quant fp4),
/// sm_86 (Ampere, 3090) -> Bf16/Fp16, everything else -> Fp32.
/// --quant fp32|bf16|fp16|fp8|fp4 forces a format (emulation anywhere).
#[cfg(feature = "cuda")]
pub fn quant_format(device: &Device, forced: Option<&str>, bf16: bool) -> burn_spectral::QuantFormat {
    use burn_spectral::QuantFormat;
    if let Some(v) = forced {
        return match v {
            "fp32" => QuantFormat::Fp32,
            "bf16" => QuantFormat::Bf16,
            "fp16" => QuantFormat::Fp16,
            "fp8" => QuantFormat::Fp8,
            "fp4" => QuantFormat::Fp4,
            _ => QuantFormat::Fp32,
        };
    }
    let sm = sm_of(device);
    // BF16 mode runs bf16 activations: the factors must match (bf16 quant),
    // not the fp32/fp8 split the plain mode picks per SM. A Fp8-quantized
    // factor under bf16 activations is an inconsistent mix.
    if bf16 {
        QuantFormat::Bf16
    } else if sm >= 120 {
        QuantFormat::Fp8
    } else {
        QuantFormat::Fp32
    }
}
#[cfg(not(feature = "cuda"))]
pub fn quant_format(_device: &Device, _forced: Option<&str>, _bf16: bool) -> burn_spectral::QuantFormat {
    burn_spectral::QuantFormat::Fp32
}

#[cfg(feature = "cuda")]
fn sm_of(_device: &Device) -> u32 {
    use cudarc::driver::sys::{
        cuDeviceGetAttribute, CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    };
    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    unsafe {
        cuDeviceGetAttribute(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
        cuDeviceGetAttribute(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
    }
    (major * 10 + minor) as u32
}

fn bytes_to_tensors<B: BurnBackend>(
    bytes: &[u8], hashes: &[i64], seq_len: usize, batch: usize, device: &Device,
) -> (Tensor<2, Int>, Tensor<3, Int>) {
    let ids: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
    let x: Tensor<2, Int> = Tensor::from_data(TensorData::new(ids, [batch, seq_len]), device);
    let h3: Vec<i64> = hashes.to_vec();
    let h: Tensor<3, Int> = Tensor::from_data(TensorData::new(h3, [batch, seq_len, 3]), device);
    (x, h)
}

/// ckpt container: [step u64][model_len u64][optim_len u64][model burnpack][optim burnpack]
/// Saved as `<dir>/<name>.bin`, atomically (tmp + rename). The parts stream
/// to the file through a BufWriter instead of concatenating a third full
/// copy in RAM (model+optim already exist as byte vectors; burnpack cannot
/// stream its records, so ~2x (model+optim) RAM is the floor without
/// upstream changes).
pub fn save_ckpt(dir: &Path, name: &str, model: &DormouseModel, optim: &Optim, step: u64, ce: f32) -> std::io::Result<()> {
    let model_bytes = model.clone().into_record().into_bytes().map_err(|e| std::io::Error::other(e.to_string()))?;
    let optim_bytes = optim.to_record().into_bytes().map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!("{name}.bin.tmp.{}", std::process::id()));
    let f = std::fs::File::create(&tmp)?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    w.write_all(&step.to_le_bytes())?;
    w.write_all(&(model_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&(optim_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&model_bytes)?;
    w.write_all(&optim_bytes)?;
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, dir.join(format!("{name}.bin")))?;
    std::fs::write(dir.join(format!("{name}.txt")), format!("step {step} ce {ce:.3}\n"))?;
    Ok(())
}

pub fn load_ckpt(dir: &Path, name: &str, cfg: &DormouseConfig, model: &mut DormouseModel, optim: &mut Optim) -> Option<u64> {
    let raw = std::fs::read(dir.join(format!("{name}.bin"))).ok()?;
    if raw.len() < 24 { return None; }
    let step = u64::from_le_bytes(raw[0..8].try_into().ok()?);
    let mlen = u64::from_le_bytes(raw[8..16].try_into().ok()?) as usize;
    let olen = u64::from_le_bytes(raw[16..24].try_into().ok()?) as usize;
    if 24 + mlen + olen > raw.len() { return None; }
    let mb = Bytes::from_bytes_vec(raw[24..24 + mlen].to_vec());
    let ob = Bytes::from_bytes_vec(raw[24 + mlen..24 + mlen + olen].to_vec());
    let mrec = ModuleRecord::from_bytes(mb).ok()?;
    let orec = OptimizerRecord::from_bytes(ob).ok()?;
    let device = device();
    *model = DormouseModel::new(cfg, &device).load_record(mrec);
    *optim = optim.clone().load_record(orec);
    Some(step)
}

/// Build the model with the run's factor-quant / bf16 compute settings.
fn build_model(
    dorm_cfg: &DormouseConfig,
    cfg: &TrainCfg,
    device: &Device,
) -> (DormouseModel, burn_spectral::QuantFormat) {
    let mut model = DormouseModel::new(dorm_cfg, device);
    let qfmt = quant_format(device, cfg.quant.as_deref(), dorm_cfg.bf16);
    if qfmt != burn_spectral::QuantFormat::Fp32 {
        model.loop_block.set_quant_all(qfmt);
        println!("quant format: {qfmt:?} ({} bits)", qfmt.bits());
    }
    // True bf16 compute: matmuls run on bf16 (tensor cores) through the
    // custom autodiff op; the graph and backward stay fp32.
    if dorm_cfg.bf16 {
        model.set_bf16_compute(true);
        println!("bf16 compute: tensor-core matmuls, fp32 graph");
    }
    (model, qfmt)
}

/// The EMA teacher exists only on the online JEPA path: with offline
/// targets (`--jepa-targets`) the hot loop must not build, advance, or pay
/// VRAM for it.
fn ema_teacher_for(
    cfg: &TrainCfg,
    dorm_cfg: &DormouseConfig,
    model: &DormouseModel,
) -> Option<DormouseModel> {
    if cfg.jepa_targets.is_some() {
        return None;
    }
    let aux_on = dorm_cfg.jepa_weight > 0.0 || dorm_cfg.dspark_weight > 0.0;
    aux_on.then(|| dormouse_core::aux::ema_update(model.clone(), model, 0.0))
}

/// Precompute `n_steps` of offline JEPA targets: walk the training stream
/// deterministically, run ONE forward per batch with the current weights
/// (no optimizer, no EMA advance), and stream `[chunk hash -> latent]`
/// records to `out`. Consume with `--jepa-targets <out>`; the per-step
/// lookup then costs one file read instead of a second full forward.
pub fn precompute_jepa_targets(
    run: &RunCfg,
    data: &Path,
    n_steps: usize,
    out: &Path,
) -> Result<(), String> {
    let dorm_cfg = &run.model;
    let cfg = &run.train;
    let device = device();
    init_pools(&device);
    let (model, qfmt) = build_model(dorm_cfg, cfg, &device);
    let mut stream = dormouse_data::ByteStream::new(cfg.seq_len, cfg.batch, data);
    // Mirror the train loop's pre-loop stream consumption (the warmup
    // forward and the quant-check probe each eat one batch) so record i
    // corresponds to training step i on a fresh run.
    if cfg.warmup {
        let _ = stream.next_batch();
    }
    if cfg.quant_check && qfmt != burn_spectral::QuantFormat::Fp32 {
        let _ = stream.next_batch();
    }
    let mut w = jepa_targets::JepaTargetWriter::create(out).map_err(|e| e.to_string())?;
    println!(
        "jepa precompute: {} steps -> {} (preset {}, batch {} s {})",
        n_steps,
        out.display(),
        run.source,
        cfg.batch,
        cfg.seq_len
    );
    for i in 0..n_steps {
        let (bytes, hashes) = stream.next_batch();
        let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
        let latent = model.forward_latent::<Backend>(x, Some(h), None).detach();
        let [b, t, d] = latent.dims();
        let vals: Vec<f32> = latent.into_data().try_to_vec().map_err(|e| e.to_string())?;
        w.push(&bytes, &vals, b, t, d).map_err(|e| e.to_string())?;
        if (i + 1) % 100 == 0 {
            println!("jepa precompute: {}/{}", i + 1, n_steps);
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    println!("jepa precompute: done ({n_steps} records)");
    Ok(())
}

/// Run pretraining. Returns `Err` on a fatal device condition (NaN loss /
/// unreadable loss scalar) with the last good checkpoint already on disk, so
/// the caller may resume by re-running with the same `ckpt_name`. On success
/// the final checkpoint is saved and `Ok(())` returned.
///
/// Before touching the device, the resolved [`RunCfg`] is snapshotted to
/// `<ckpt_dir>/<ckpt_name>.config.toml` (atomic tmp+rename, like save_ckpt).
/// A resume with a DIFFERENT resolved config is a hard error (drift check,
/// ADR-0005): the stored snapshot is diffed key-by-key against the fresh
/// resolve and mismatches abort the run before any GPU work.
pub fn train_loop(
    run: RunCfg,
    data: PathBuf,
    ckpt_dir: Option<PathBuf>,
    eval_data: Option<PathBuf>,
) -> Result<(), String> {
    let dir = ckpt_dir.unwrap_or_else(|| PathBuf::from("checkpoints"));
    let _ = std::fs::create_dir_all(&dir);
    // Config snapshot + drift check (ADR-0005), before init_pools so a
    // mismatched resume never reaches the GPU.
    let snap_path = dir.join(format!("{}.config.toml", run.train.ckpt_name));
    if snap_path.is_file() {
        let stored = std::fs::read_to_string(&snap_path)
            .map_err(|e| format!("config snapshot {}: {e}", snap_path.display()))?;
        let old = RunCfg::from_snapshot(&stored)?;
        let keys = run.diff_keys(&old);
        if !keys.is_empty() {
            return Err(format!(
                "config drift: {} key(s) differ from the stored snapshot {}:\n  {}",
                keys.len(),
                snap_path.display(),
                keys.join(", ")
            ));
        }
    } else {
        let tmp = dir.join(format!("{}.config.toml.tmp.{}", run.train.ckpt_name, std::process::id()));
        std::fs::write(&tmp, run.snapshot_toml())
            .map_err(|e| format!("config snapshot write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &snap_path).map_err(|e| format!("config snapshot {}: {e}", snap_path.display()))?;
    }
    let RunCfg { source: preset, model: dorm_cfg, train: cfg } = run;
    let device = device();
    init_pools(&device);
    let (mut model, qfmt) = build_model(&dorm_cfg, &cfg, &device);
    let mut optim = build_optim(&cfg);
    // Fail fast if the routing policy no longer matches the model (stale
    // marker after a module rename would silently degrade to AdamW).
    let gc = validate_routing(&model, cfg.factors_fallback, cfg.qk_heads)
        .unwrap_or_else(|e| panic!("optimizer routing check failed: {e}"));
    let opt_name = match cfg.opt.as_str() {
        "adan" => "Adan (all params)".to_string(),
        "adamw" => "AdamW (all params)".to_string(),
        "muon" => "Muon+ ColRow (all params, 1D -> Muon+'s AdamW)".to_string(),
        "mix-adan" => format!("Muon+ ColRow ns={MUON_NS_STEPS} + head-wise Muon q/k + Adam wd0 (tables) + Adan (rest)"),
        _ => format!("Muon+ ColRow ns={MUON_NS_STEPS} + head-wise Muon q/k + Adam wd0 (tables) + AdamW (rest)"),
    };
    let factors = if cfg.factors_fallback { " (expert TSCT factors on fallback)" } else { "" };
    println!("optimizer: {opt_name}{factors} [muon={} qk={} tables={} rest={}]", gc.muon, gc.qk, gc.tables, gc.rest);
    let mut step = load_ckpt(&dir, &cfg.ckpt_name, &dorm_cfg, &mut model, &mut optim).unwrap_or(0);
    if step > 0 { println!("resumed {} from {} step {step}", cfg.ckpt_name, dir.display()); }
    // EMA teacher for the JEPA aux (momentum 0.0 at init = exact copy with
    // fresh grad-free params); advanced after every optimizer step below.
    // Offline mode (--jepa-targets) skips the teacher entirely: no second
    // forward, no EMA advance, none of its VRAM.
    let mut teacher = ema_teacher_for(&cfg, &dorm_cfg, &model);
    let mut jepa_tgts = match &cfg.jepa_targets {
        Some(p) => Some(jepa_targets::JepaTargets::open(p, &device)?),
        None => None,
    };
    let mut stream = dormouse_data::ByteStream::new(cfg.seq_len, cfg.batch, &data);
    let mut eval_stream =
        eval_data.as_ref().map(|p| dormouse_data::ByteStream::new(cfg.seq_len, cfg.batch, p));
    let mut best = f32::INFINITY;
    let mut stress = cfg.stress.then(|| StressMonitor::new(cfg.stress_lr, cfg.stress_every));
    // RAM-offload n-gram tables (report §2.3): --engram-ram keeps the
    // tables in host memory (millions of slots in the 64 GB RAM), trains
    // them with CPU Adam, and copies only the batch's rows to the GPU.
    let mut host: Option<offload::HostNgram> = if cfg.engram_ram {
        let slots = cfg.engram_slots;
        let ng_path = dir.join(format!("{}.ngram", cfg.ckpt_name));
        let h = offload::HostNgram::new([slots, slots, slots], 32, 0x1234_5678);
        let h = match std::fs::read(&ng_path).ok() {
            Some(b) => match offload::HostNgram::from_bytes(&b, [slots, slots, slots], 32) {
                Some(h) => h,
                // v1 checkpoints carried Adam m+v state; the layout magic
                // rejects them so they are never misread as momentum.
                None => {
                    eprintln!(
                        "ngram ckpt {}: unknown layout (v1 Adam state?) - starting fresh tables",
                        ng_path.display()
                    );
                    h
                }
            },
            None => h,
        };
        println!(
            "engram: {} rows in RAM ({} MB), CPU nesterov momentum + Sinkhorn",
            h.total_rows(),
            h.total_rows() * h.row_bytes() / (1 << 20)
        );
        Some(h)
    } else {
        None
    };
    let n_params = model.num_params();
    println!(
        "dormouse pretrain {preset} params={n_params} steps={} data={:?} lr={} backend=cuda(autodiff)",
        cfg.steps, data, cfg.lr
    );
    // #7 warmup: 2 fwd+bwd at full depth raise the pool high-water before the
    // loop (then cleanup returns the pages - later steps reuse cached blocks).
    if cfg.warmup && step == 0 {
        let (bytes, hashes) = stream.next_batch();
        let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [cfg.batch, cfg.seq_len]), &device);
        for _ in 0..2 {
        let (_logits, rec, pd, _k, _aux) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, Some(y.clone()), None);
        let loss = model.loss::<Backend>(rec, pd);
            let _g = loss.backward();
        }
        memory_cleanup(&device);
        println!("warmup done");
    }
    // quant fidelity check: quantized model vs fp32 reference on one batch.
    // --quant-check prints max/mean logit deviation + loss delta so Fp8/Fp4
    // viability is measured, not assumed.
    if cfg.quant_check && qfmt != burn_spectral::QuantFormat::Fp32 && step == 0 {
        let (bytes, hashes) = stream.next_batch();
        let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [cfg.batch, cfg.seq_len]), &device);
        let (lq, rec_q, pd_q, _k, _aq) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, Some(y.clone()), None);
        let loss_q: f32 = model.loss::<Backend>(rec_q, pd_q).try_into_scalar().unwrap_or(f32::NAN);
        let mut ref_model = model.clone();
        ref_model.loop_block.set_quant_all(burn_spectral::QuantFormat::Fp32);
        let (lr, rec_r, pd_r, _k, _ar) = ref_model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y), None);
        let loss_r: f32 = ref_model.loss::<Backend>(rec_r, pd_r).try_into_scalar().unwrap_or(f32::NAN);
        let vq: Vec<f32> = lq.into_data().try_to_vec().unwrap_or_default();
        let vr: Vec<f32> = lr.into_data().try_to_vec().unwrap_or_default();
        println!("check sums: q={:.4} ref={:.4} lens={}/{}", vq.iter().sum::<f32>(), vr.iter().sum::<f32>(), vq.len(), vr.len());
        let n = vq.len().min(vr.len());
        let mut max_d = 0.0f32;
        let mut sum_d = 0.0f32;
        for i in 0..n {
            let d = (vq[i] - vr[i]).abs();
            if d > max_d { max_d = d; }
            sum_d += d;
        }
        let mean_d = if n > 0 { sum_d / n as f32 } else { f32::NAN };
        let rel = if n > 0 {
            let base: f32 = vr.iter().take(n).map(|x| x.abs()).sum::<f32>() / n as f32;
            100.0 * mean_d / base.max(1e-9)
        } else { f32::NAN };
        println!(
            "quant check {qfmt:?}: logits max|d|={max_d:.4} mean|d|={mean_d:.5} rel={rel:.2}% loss q={loss_q:.3} ref={loss_r:.3} d={:.4}",
            (loss_q - loss_r).abs()
        );
        // factor quantization error on U (is the format actually quantizing?)
        if let dormouse_core::param::LinearLikeInner::Tsct(l) =
            &model.loop_block.expert_ffns[0].gate_up.inner
        {
            let u = l.u.val();
            let q = burn_bitnet::quantize_tensor::<Backend>(u.clone(), qfmt.bits());
            let v: Vec<f32> = (u - q).into_data().try_to_vec().unwrap_or_default();
            let max_q = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
            let mean_q = v.iter().sum::<f32>() / v.len().max(1) as f32;
            println!("factor quant err (U): max={max_q:.5} mean={mean_q:.6}");
        }
    }
    // #5 prefetch: tensors for the NEXT step are prepared while the current
    // step's GPU work is still in flight.
    let (mut pbytes, mut phashes) = stream.next_batch();
    let mut         pshift: Vec<i64> = pbytes.iter().skip(1).chain(std::iter::once(&pbytes[0])).map(|&b| b as i64).collect();
    let mut ce = f32::NAN;
    // One-way fp32 fallback state. With `--quant fp32` the forward is already
    // exact — the monitor would have nothing to guard, so skip it entirely
    // (it costs 30+ device syncs per check).
    let mut ortho_fp32 = cfg.quant.as_deref() == Some("fp32");
    // M6: DM_FUSED=1 single-node path (default 0 preserves behavior). The
    // fused path is compiled only with the cuda feature; the cpu build never
    // takes it. Flagship-compatible since the arms-adjoint work: host rows
    // train through the op's rows parent, the aux heads run on the burn path
    // from the op's exposed latents. Still excluded: bf16 (kernels are f32),
    // act-quant (STE not in the kernels), GR. The gate is hoisted out of the
    // loop (its inputs are constants) so the startup log line provably
    // matches what every step does - a silent DM_FUSED=1 fallback is the
    // failure mode this run-length is too short to expose any other way.
    #[cfg(feature = "cuda")]
    let use_fused = dormouse_core::fused::fused_enabled()
        && !dorm_cfg.bf16
        && dorm_cfg.act_quant.is_none()
        && !dorm_cfg.use_gr;
    #[cfg(feature = "cuda")]
    {
        let reason = if !dormouse_core::fused::fused_enabled() {
            "DM_FUSED=0"
        } else if dorm_cfg.bf16 {
            "bf16 compute (fused kernels are f32)"
        } else if dorm_cfg.act_quant.is_some() {
            "act-quant (STE not in the fused kernels)"
        } else if dorm_cfg.use_gr {
            "Gated Residual"
        } else {
            ""
        };
        if reason.is_empty() {
            println!("forward arm: FUSED ponder_loop_step (host-rows engram OK, aux heads burn-side)");
        } else {
            println!("forward arm: burn (fused gated off: {reason})");
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _use_fused = false;
    while step < cfg.steps as u64 {
        let t_iter = std::time::Instant::now();
        let (bytes, hashes) = (std::mem::replace(&mut pbytes, Vec::new()), std::mem::replace(&mut phashes, Vec::new()));
        let t_io = std::time::Instant::now();
        let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
        let data_ms = t_io.elapsed().as_secs_f64() * 1000.0;
        // targets = next byte (shifted by one position)
        let shifted = std::mem::take(&mut pshift);
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [cfg.batch, cfg.seq_len]), &device);
        // prepare next batch right away (host->device copy is async). The
        // shift targets MUST be built from the prefetched batch: building
        // them from the consumed one trains every step after the first on
        // the previous batch's labels.
        let (nb, nh) = match &host {
            Some(h) => stream.next_batch_with_tables(h.slots),
            None => stream.next_batch(),
        };
        pshift = nb.iter().skip(1).chain(std::iter::once(&nb[0])).map(|&b| b as i64).collect();
        pbytes = nb;
        phashes = nh;

        // RAM-offload path: gather this batch's n-gram rows on the host,
        // copy them to the GPU as an autodiff leaf, and train the tables
        // with CPU Adam from the row gradients (report §2.3). The copy is
        // ~600 KB per step; the impact on the step time is negligible
        // (~1 ms CPU against ~10 s GPU).
        let (rows_param, host_rows, uniq_rows) = match &host {
            Some(h) => {
                let (p, embed, uniq) = offload::rows_for_batch::<Backend>(
                    h,
                    &hashes,
                    cfg.batch,
                    cfg.seq_len,
                    &device,
                    true,
                );
                (p, Some(embed), Some(uniq))
            }
            None => (None, None, None),
        };

        let t_fwd = std::time::Instant::now();
        // Offline JEPA: the frozen target for THIS chunk must exist in the
        // sidecar; a miss is a hard error (stale sidecar), never a silent
        // aux drop.
        let jepa_target = match jepa_tgts.as_mut() {
            Some(t) => Some(t.get(&bytes)?),
            None => None,
        };
        // M6: DM_FUSED=1 single-node path (default 0 preserves behavior).
        // The fused path is compiled only with the cuda feature; the cpu
        // build never takes it. Flagship-compatible since the arms-adjoint
        // work: host rows train through the op's rows parent, the aux heads
        // run on the burn path from the op's exposed latents. Still excluded:
        // bf16 (kernels are f32), act-quant (STE not in the kernels), GR.
        #[cfg(feature = "cuda")]
        let (_logits, rec_ce, p_dist, _kda, aux) = if use_fused {
            // Build PonderInputs from the live model (embedding + loop_block)
            let x_emb = {
                let e = model.embedding.forward(x.clone());
                if dorm_cfg.bf16 {
                    e.cast(burn::tensor::FloatDType::BF16)
                } else {
                    e
                }
            };
            // Fac helper (same as fused/tests.rs)
            let fac_of = |ll: &dormouse_core::LinearLike| {
                let l = match &ll.inner {
                    dormouse_core::param::LinearLikeInner::Tsct(l) => l,
                    _ => panic!("fused requires TSCT"),
                };
                dormouse_core::fused::Fac {
                    u: l.u.val(),
                    s: l.s.val(),
                    v: l.v.val(),
                }
            };
            let lb_bytes = {
                let rec = model.loop_block.clone().into_record();
                rec.into_bytes().unwrap().to_vec()
            };
            let mut experts = Vec::new();
            for e in &model.loop_block.expert_ffns {
                experts.push([fac_of(&e.gate_up), fac_of(&e.down)]);
            }
            // JEPA target: offline sidecar when present, else the live EMA
            // teacher's latent over the same inputs (the same second forward
            // forward_with_hidden would run). Both stop-grad on the burn path.
            let teacher_latent = match jepa_target {
                Some(tg) => Some(tg),
                None => teacher
                    .as_ref()
                    .map(|t| t.forward_latent::<Backend>(y.clone(), None, None)),
            };
            let inputs = dormouse_core::fused::PonderInputs {
                x: x_emb,
                targets: y.clone().reshape([cfg.batch * cfg.seq_len, 1]),
                controller_w: model.loop_block.controller.weight.val(),
                norm_g: model.loop_block.norm.weight.val(),
                final_norm_g: model.norm.weight.val(),
                iter_embed: model.loop_block.iter_embed.val(),
                residual_scale: model.loop_block.residual_scale.val(),
                halt_w: model.loop_block.halt_head.weight.val(),
                experts,
                out_proj: fac_of(&model.loop_block.out_proj),
                lm_head: fac_of(&model.lm_head),
                norm_eps: dorm_cfg.norm_eps,
                ponder_prior: model.ponder_prior,
                // RAM-offload rows drive the Engram; hashed_ids only without
                // them (a dead H2D copy otherwise), same as the burn path.
                hashed_ids: if host_rows.is_some() { None } else { Some(h.clone()) },
                host_rows,
                arm_leaves: Some(dormouse_core::fused::ArmLeavesPair {
                    attn: dormouse_core::fused::ArmLeaves::capture(&model.loop_block.shared_attn),
                    engram: dormouse_core::fused::ArmLeaves::capture(&model.loop_block.engram),
                }),
                loop_block_bytes: Some(lb_bytes),
                cfg: Some(dorm_cfg.clone()),
            };
            let out = dormouse_core::fused::ponder_loop_step(inputs);
            // Aux heads stay OUTSIDE the op: they consume the exposed latents
            // (out_acc for JEPA/KoLeo, the final-norm output for DSpark) on
            // the burn graph, so their own params train normally.
            let aux = model.aux_loss::<Backend>(
                &out.out_acc,
                teacher_latent,
                Some(y.clone()),
                &out.h,
                &out.logits,
            );
            let kda_dummy = Tensor::<4>::zeros([1, 1, 1, 1], &out.logits.device());
            (out.logits, out.rec, out.p_dist, kda_dummy, aux)
        } else if let Some(tg) = jepa_target {
            model.forward_with_jepa_targets::<Backend>(
                x,
                // RAM-offload path drives the Engram from host_rows; uploading
                // hashed_ids too would be a dead per-step H2D copy.
                if host_rows.is_some() { None } else { Some(h) },
                host_rows,
                Some(y),
                Some(tg),
            )
        } else {
            model.forward_with_hidden::<Backend>(
                x,
                if host_rows.is_some() { None } else { Some(h) },
                host_rows,
                Some(y),
                teacher.as_ref(),
            )
        };
        #[cfg(not(feature = "cuda"))]
        let (_logits, rec_ce, p_dist, _kda, aux) = if let Some(tg) = jepa_target {
            model.forward_with_jepa_targets::<Backend>(
                x,
                if host_rows.is_some() { None } else { Some(h) },
                host_rows,
                Some(y),
                Some(tg),
            )
        } else {
            model.forward_with_hidden::<Backend>(
                x,
                if host_rows.is_some() { None } else { Some(h) },
                host_rows,
                Some(y),
                teacher.as_ref(),
            )
        };
        let mut loss = model.loss::<Backend>(rec_ce, p_dist);
        // Host tables train at their own cadence (the report's rule: Adam on
        // the RAM tables every step), not piggybacked on the log cadence -
        // at log_every=100 they got 1/100 of their updates.
        let host_adam_step = cfg.host_adam_every > 0
            && step % cfg.host_adam_every as u64 == 0
            && rows_param.is_some();
        // Aux is read only on log steps; skip the clone (an extra autodiff
        // node) elsewhere.
        let aux_log = if step % cfg.log_every as u64 == 0 {
            aux.clone()
        } else {
            None
        };
        if let Some(a) = aux {
            loss = loss + a;
        }
        // The device syncs only on steps that read something back (loss
        // scalar at log cadence, host-table grads at the host-Adam cadence,
        // timers) so forward/backward/step of adjacent steps overlap on the
        // GPU. The scalar read rides along free inside the grads D2H.
        let loss_log = if step % cfg.log_every as u64 == 0
            || host_adam_step
            || (cfg.timers && step % 50 == 0)
        {
            Some(loss.clone())
        } else {
            None
        };
        let t_bwd = std::time::Instant::now();
        let fwd_ms = t_fwd.elapsed().as_secs_f64() * 1000.0;
        let raw_grads = loss.backward();
        let lr = match stress.as_ref() {
            // Constant LR at a multiple of the optimum (report §3.3).
            Some(s) => s.lr(cfg.lr),
            None => wsd_factor(step, cfg.steps as u64, cfg.lr),
        };
        if let Some(loss_log) = &loss_log {
            let ce_now: f32 = match loss_log.clone().try_into_scalar() {
                Ok(v) => v,
                Err(_) => {
                    return Err(format!("step {step}: device error (loss is not a scalar)"));
                }
            };
            if !ce_now.is_finite() {
                return Err(format!("step {step}: NaN loss (last good ckpt kept)"));
            }
            ce = ce_now;
            if ce < best { best = ce; }
            if host_adam_step {
                if let (Some(h), Some(p), Some(uniq)) =
                    (host.as_mut(), &rows_param, uniq_rows.as_ref())
                {
                    if let Some(g) = p.grad(&raw_grads) {
                        let g_vec: Vec<f32> = g.into_data().try_to_vec().unwrap_or_default();
                        if g_vec.len() == uniq.len() * h.dim {
                            h.momentum_update(uniq, &g_vec, lr as f32);
                        }
                    }
                }
            }
            if step % cfg.log_every as u64 == 0 {
                if let Some(s) = stress.as_mut() {
                    s.observe(ce, grad_norm(&model, &raw_grads));
                }
            }
        }
        let bwd_ms = t_bwd.elapsed().as_secs_f64() * 1000.0;
        let t_opt = std::time::Instant::now();
        let grads = GradientsParams::from_grads(raw_grads, &model);
        let model_new = optim.step(lr, model, grads);
        model = model_new;
        let opt_ms = t_opt.elapsed().as_secs_f64() * 1000.0;
        let t_retr = std::time::Instant::now();
        // TSCT ortho maintenance (bf16_KERNEL_PLAN): retract the U/V masters
        // every step so the quantized forward stays faithful; monitor the
        // drift at cadence and fall back to fp32 factors when it exceeds the
        // plan's 1e-3 threshold. --retract-every / --retract-iters override.
        if step % cfg.retract_every.max(1) as u64 == 0 {
            model.retract_tsct(cfg.retract_iters);
        }
        let retr_ms = t_retr.elapsed().as_secs_f64() * 1000.0;
        let t_ema = std::time::Instant::now();
        if let Some(t) = teacher.take() {
            teacher = Some(dormouse_core::aux::ema_update(t, &model, dormouse_core::aux::TEACHER_MOMENTUM));
        }
        let ema_ms = t_ema.elapsed().as_secs_f64() * 1000.0;
        // max_ortho reads every TSCT factor (30+ device syncs) - cadence,
        // not per-50-steps: each check drains the pipeline. The metric is
        // per-entry (F-norm/k) so the plan's 1e-3 threshold sits above the
        // retract's convergence floor and below real drift; before the
        // normalization (2026-09-04) the fallback fired at step 0 on every
        // fresh run, silently disabling the factor-quant forward.
        if step % 500 == 0 && !ortho_fp32 {
            let ortho = model.max_ortho();
            if ortho > 1e-3 {
                println!("max_ortho {ortho:.2e} > 1e-3 - fallback fp32 factors");
                model.set_quant_all(burn_spectral::QuantFormat::Fp32);
                ortho_fp32 = true;
            }
        }

        if cfg.timers && step % 50 == 0 {
            // Force a device sync so the elapsed wall time equals the true GPU
            // step time (forward+backward+optim+retract). data_ms is the CPU
            // side (read + bytes_to_tensors). Their difference is GPU compute.
            if let Some(ll) = &loss_log {
                let _: f32 = ll.clone().try_into_scalar().unwrap_or(0.0);
            }
            let total_ms = t_iter.elapsed().as_secs_f64() * 1000.0;
            println!(
                "timer step {step}: total={total_ms:.0}ms data={data_ms:.1}ms fwd={fwd_ms:.0}ms \
                 bwd={bwd_ms:.0}ms (incl. loss sync + host-adam D2H) opt={opt_ms:.0}ms \
                 retr={retr_ms:.1}ms ema={ema_ms:.1}ms gpu_step={:.0}ms",
                total_ms - data_ms
            );
        }
        if step % cfg.log_every as u64 == 0 {
            let bpb = bpb(ce);
            // pool_stats syncs the device; only with --memlog
            let mem = if cfg.memlog { pool_stats(&device) } else { String::new() };
            let aux_note = match aux_log.clone().map(|a| a.try_into_scalar::<f32>().ok()) {
                Some(Some(v)) => format!(" aux={v:.4}"),
                _ => String::new(),
            };
            println!("step {step:6} ce={ce:.3} bpb={bpb:.3} best={best:.3} lr={lr:.2e}{aux_note} {mem}");
            if let Some(s) = stress.as_ref() {
                if let Some(line) = s.report(step) {
                    println!("  {line}");
                }
            }
        }
        if cfg.eval_every > 0 {
            if let Some(ev) = eval_stream.as_mut() {
                if step % cfg.eval_every as u64 == 0 && step > 0 {
                    let (eb, eh) = match &host {
                        Some(h) => ev.next_batch_with_tables(h.slots),
                        None => ev.next_batch(),
                    };
                    let (ex, _eh_t) =
                        bytes_to_tensors::<Backend>(&eb, &eh, cfg.seq_len, cfg.batch, &device);
                    let eshift: Vec<i64> = eb
                        .iter()
                        .skip(1)
                        .chain(std::iter::once(&eb[0]))
                        .map(|&b| b as i64)
                        .collect();
                    let ey: Tensor<2, Int> =
                        Tensor::from_data(TensorData::new(eshift, [cfg.batch, cfg.seq_len]), &device);
                    let eval_rows = match &host {
                        Some(h) => {
                            let (_, embed, _) = offload::rows_for_batch::<Backend>(
                                h,
                                &eh,
                                cfg.batch,
                                cfg.seq_len,
                                &device,
                                false,
                            );
                            Some(embed)
                        }
                        None => None,
                    };
                    let (elogits, ..) =
                        model.forward_with_hidden::<Backend>(ex, None, eval_rows, None, None);
                    let v = model.vocab_size;
                    let eflat = elogits.reshape([cfg.batch * cfg.seq_len, v]);
                    // Gather the target log-prob: no one-hot [b*t,v] fp32
                    // tensor (extra H2D + traffic) per eval.
                    let etgt = ey.reshape([cfg.batch * cfg.seq_len, 1]);
                    let ece: f32 = burn::tensor::activation::log_softmax(eflat, 1)
                        .gather(1, etgt)
                        .neg()
                        .mean()
                        .try_into_scalar()
                        .unwrap_or(f32::NAN);
                    let ebpb = bpb(ece);
                    println!("step {step:6} EVAL ce={ece:.3} bpb={ebpb:.3}");
                }
            }
        }
        if cfg.ckpt_every > 0 && step % cfg.ckpt_every as u64 == 0 {
            let _ = save_ckpt(&dir, &cfg.ckpt_name, &model, &optim, step, ce);
            if let Some(h) = &host {
                let path = dir.join(format!("{}.ngram", cfg.ckpt_name));
                let tmp = dir.join(format!("{}.ngram.tmp.{}", cfg.ckpt_name, std::process::id()));
                if let Ok(f) = std::fs::File::create(&tmp) {
                    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
                    let _ = h.write_to(&mut w);
                    let _ = std::io::Write::flush(&mut w);
                    let _ = std::fs::rename(&tmp, &path);
                }
            }
            println!("ckpt {}.bin saved step {step}", cfg.ckpt_name);
        }
        // Returning pool pages forces the next steps to re-acquire them from
        // the driver; at a near-full high-water this stalls the GPU. 500
        // steps keeps the OOM guard while amortizing the reacquisition.
        if step % 500 == 0 {
            memory_cleanup(&device);
        }
        step += 1;
    }
    // Final ckpt records the LAST ce (best is tracked in the logs; the .txt
    // sidecar should describe this checkpoint's actual loss).
    let _ = save_ckpt(&dir, &cfg.ckpt_name, &model, &optim, step, ce);
    println!("done steps={step} best ce={best:.3}");
    Ok(())
}

/// Load weights only (inference) from a named ckpt container.
pub fn load_model_weights(dir: &Path, name: &str, cfg: DormouseConfig) -> Option<DormouseModel> {
    let raw = std::fs::read(dir.join(format!("{name}.bin"))).ok()?;
    if raw.len() < 24 { return None; }
    let mlen = u64::from_le_bytes(raw[8..16].try_into().ok()?) as usize;
    if 24 + mlen > raw.len() { return None; }
    let mb = Bytes::from_bytes_vec(raw[24..24 + mlen].to_vec());
    let mrec = ModuleRecord::from_bytes(mb).ok()?;
    let device = device();
    let model = DormouseModel::new(&cfg, &device);
    Some(model.load_record(mrec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::validate_routing_with;
    use dormouse_core::param::{LinearLike, LinearLikeInner};
    use dormouse_data::fnv;

    /// Retract must pull drifted TSCT masters back to orthonormal: corrupt
    /// U by scaling, verify ortho_error collapses below the plan's 1e-3
    /// threshold (the quantized forward depends on it).
    #[test]
    fn tsct_retract_restores_ortho() {
        let dev = device();
        let mut ll = LinearLike::new(64, 64, 16, &dev);
        if let LinearLikeInner::Tsct(l) = &mut ll.inner {
            let u = l.u.val().mul_scalar(3.0).detach();
            l.u = burn::module::Param::from_tensor(u.into());
        } else {
            panic!("LinearLike must be TSCT on ndarray");
        }
        let before = ll.max_ortho();
        ll.retract(3);
        let after = ll.max_ortho();
        assert!(
            after < before / 10.0 && after < 1e-3,
            "retract must restore orthonormality: {before:.2e} -> {after:.2e}"
        );
    }

    /// The streamed ckpt writer (save_ckpt) must produce a container that
    /// load_ckpt restores byte-faithfully: same step, same ce sidecar, and
    /// an identical forward after reload. Guards the 2026-09-04 rewrite
    /// (concat-buffer -> BufWriter streaming).
    #[test]
    fn ckpt_save_load_roundtrip() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let optim_cfg = TrainCfg { steps: 5, seq_len: 64, batch: 2, ..Default::default() };
        let optim = crate::optim::build_optim_mode(&optim_cfg, "mix");
        let dir = std::env::temp_dir().join("dm-ckpt-roundtrip-test");
        let _ = std::fs::remove_dir_all(&dir);
        save_ckpt(&dir, "rt", &model, &optim, 42, 3.25).expect("save");
        let txt = std::fs::read_to_string(dir.join("rt.txt")).expect("sidecar");
        assert!(txt.contains("step 42 ce 3.250"), "sidecar: {txt}");
        let mut model2 = DormouseModel::new(&cfg, &device());
        let mut optim2 = crate::optim::build_optim_mode(&optim_cfg, "mix");
        let step = load_ckpt(&dir, "rt", &cfg, &mut model2, &mut optim2).expect("load");
        assert_eq!(step, 42);
        // identical forward on the same batch
        let bytes: Vec<u8> = (0..128u32).map(|i| (i.wrapping_mul(37) % 256) as u8).collect();
        let mut hashes = Vec::with_capacity(128 * 3);
        for p in 0..128usize {
            let e = p + 1;
            hashes.push((fnv(&bytes[e.saturating_sub(3)..e]) % 4096) as i64);
            hashes.push((fnv(&bytes[e.saturating_sub(5)..e]) % 4096) as i64);
            hashes.push((fnv(&bytes[e.saturating_sub(8)..e]) % 4096) as i64);
        }
        let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, 64, 2, &device());
        let (l1, ..) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, None, None);
        let (l2, ..) = model2.forward_with_hidden::<Backend>(x, Some(h), None, None, None);
        let d = (l1 - l2).abs().max().into_scalar::<f32>();
        assert!(d < 1e-5, "reloaded model diverges: max |dlogit| = {d:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The report's §3.1 routing, asserted against burn's actual param paths
    /// (field names joined by "."; Vec entries are indices, e.g.
    /// `expert_ffns.0.gate_up.inner.u`). A typo in a marker silently leaves
    /// the param on AdamW, so every boundary is pinned here.
    #[test]
    fn muon_routing_matches_report() {
        // Muon+ group: small low-rank factors only (NS on [d,d] projections
        // is ~40 s/step on this box in fp32; see optim.rs doc).
        assert!(is_muon_param("loop_block.expert_ffns.0.gate_up.inner.u"));
        assert!(is_muon_param("loop_block.expert_ffns.2.down.inner.v"));
        assert!(is_muon_param("loop_block.engram.key_projs.0.weight"));
        assert!(is_muon_param("loop_block.out_proj.inner.u"));
        assert!(is_muon_param("loop_block.out_proj.inner.v"));
        // 1D TSCT scale stays on AdamW.
        assert!(!is_muon_param("loop_block.expert_ffns.0.gate_up.inner.s"));
        // [d,d] projections and dense linears stay on the fallback while the
        // fp32 NS cost is prohibitive.
        assert!(!is_muon_param("loop_block.shared_attn.gdn2.q_proj.weight"));
        assert!(!is_muon_param("loop_block.shared_attn.msa.attention.out_proj.weight"));
        assert!(!is_muon_param("loop_block.engram.value_proj.weight"));
        // Attention Q/K go to the head-wise Muon groups instead (not the
        // plain Muon+ marker list).
        assert!(crate::optim::is_qk_param("loop_block.shared_attn.gdn2.q_proj.weight"));
        assert!(crate::optim::is_qk_param("loop_block.shared_attn.gdn2.k_proj.weight"));
        assert!(crate::optim::is_qk_param("loop_block.shared_attn.msa.attention.q_proj.weight"));
        assert!(crate::optim::is_qk_param("loop_block.shared_attn.msa.attention.k_proj.weight"));
        assert!(!crate::optim::is_qk_param("loop_block.shared_attn.gdn2.v_proj.weight"));
        assert!(!crate::optim::is_qk_param("loop_block.shared_attn.msa.attention.out_proj.weight"));
        // The MQA indexer stays on the fallback (tiny, ambiguous heads).
        assert!(!crate::optim::is_qk_param("loop_block.shared_attn.msa.index_branch.q_proj.weight"));
        // Per-head scalar producers (decay/β gates): AdamW.
        assert!(!is_muon_param("loop_block.shared_attn.gdn2.decay.w_up.weight"));
        assert!(!is_muon_param("loop_block.shared_attn.gdn2.beta_proj.weight"));
        // Routers/scorers: AdamW.
        assert!(!is_muon_param("loop_block.controller.weight"));
        assert!(!is_muon_param("loop_block.halt_head.weight"));
        assert!(!is_muon_param("loop_block.shared_attn.router.inner.u"));
        assert!(!is_muon_param("loop_block.shared_attn.msa.index_branch.q_proj.weight"));
        // Embeddings, elongated readouts, output head: AdamW.
        assert!(!is_muon_param("embedding.weight"));
        assert!(!is_muon_param("lm_head.inner.v"));
        // n-gram tables: plain Adam, wd disabled; projections are Muon+.
        assert!(is_engram_table_param("loop_block.engram.memory.embedding.weight"));
        assert!(!is_engram_table_param("loop_block.engram.key_projs.0.weight"));
    }

    /// NdArray-friendly mini config: model init and steps take seconds on CPU
    /// (the small preset takes ~a minute just to build SpectralLinear).
    fn test_cfg() -> DormouseConfig {
        DormouseConfig {
            d_model: 128,
            n_heads: 2,
            head_dim: 32,
            d_ffn: 256,
            max_iter: 4,
            rank: 16,
            msa_block: 32,
            ..DormouseConfig::default()
        }
    }

    /// Live model: the policy must hold on the real module tree. The expected
    /// Muon+ count is derived from the config topology, not a literal:
    /// 3 KDA qkv + 4 sparse-core projections + n_experts x (gate_up u,v +
    /// down u,v) + 1 Engram key_proj + 1 value_proj. With head-wise Q/K
    /// routing on, the four q/k matrices move into the qk group.
    #[test]
    fn routing_validates_on_live_model() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let expected_muon = 4 * cfg.n_experts + 3;
        let c = validate_routing(&model, false, Some(cfg.n_heads)).expect("policy must hold on the live model");
        assert_eq!(c.muon, expected_muon, "Muon+ group must match the topology");
        assert_eq!(c.qk, 4, "gdn2 q/k + msa q/k must be head-wise Muon");
        assert_eq!(c.tables, 1, "n-gram tables group");
        assert!(c.rest > 0);
        // Without head-wise routing the q/k params fall back to rest.
        let c = validate_routing(&model, false, None).expect("policy must hold");
        assert_eq!(c.qk, 0);
    }

    /// The validator must fire when routing breaks: a marker that matches
/// nothing (stale after a rename) and a marker that routes a 1D param into
/// the Muon+ group. Silent fallback is the failure mode this check exists
/// to prevent.
    #[test]
    fn routing_validator_detects_stale_markers() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        // Stale Muon+ marker: matches no param.
        let err = validate_routing_with(&model, &["no.such.module"], "engram.memory", None)
            .expect_err("dead Muon+ markers must fail validation");
        assert!(err.contains("matches no param"), "unexpected error: {err}");
        // Stale table marker.
        let err = validate_routing_with(&model, crate::optim::MUON_PATH_MARKERS, "no.such.table", None)
            .expect_err("dead table marker must fail validation");
        assert!(err.contains("matches no param"), "unexpected error: {err}");
        // A marker that hits a 1D param (RMSNorm gain) must trip the rank
        // invariant: Muon+ only makes sense on matrices. (The Q/K group
        // cannot hit this: its fixed paths are always 2D Linears; the
        // non-2D guard there is defense-in-depth.)
        let err = validate_routing_with(&model, &["norm.weight"], "engram.memory", None)
            .expect_err("1D param in the Muon+ group must fail validation");
        assert!(err.contains("1D param routed to Muon+"), "unexpected error: {err}");
    }

    /// End-to-end on NdArray: the mixed optimizer must drive the PonderNet
    /// loss down on synthetic bytes (routing, Muon+ step and checkpoint-free
    /// resume all exercise the real path). Uses a shrunken config: NdArray
    /// autodiff on the `small` preset takes minutes per step.
    #[test]
    fn mixed_optim_converges() {
        let cfg = test_cfg();
        let mut model = DormouseModel::new(&cfg, &device());
        let optim_cfg = TrainCfg { steps: 40, seq_len: 64, batch: 2, lr: 1e-3, wd: 0.01, grad_clip: 1.0, ..Default::default() };
        let mut optim = crate::optim::build_optim_mode(&optim_cfg, "mix");

        let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;
        let next_u8 = |s: &mut u64| {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let x = (*s ^ (*s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            ((x ^ (x >> 27)) >> 33) as u8
        };
        let mut losses: Vec<f32> = Vec::with_capacity(40);
        for step_i in 0..40usize {
            let bytes: Vec<u8> = (0..128).map(|_| next_u8(&mut rng_state)).collect();
            // FNV 3/5/8-gram hashes mod 4096, mirroring ByteStream::hashes.
            let mut hashes = Vec::with_capacity(128 * 3);
            for p in 0..128usize {
                let e = p + 1;
                hashes.push((fnv(&bytes[e.saturating_sub(3)..e]) % 4096) as i64);
                hashes.push((fnv(&bytes[e.saturating_sub(5)..e]) % 4096) as i64);
                hashes.push((fnv(&bytes[e.saturating_sub(8)..e]) % 4096) as i64);
            }
            let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, 64, 2, &device());
            let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
            let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
            let (_logits, rec, pd, _k, _aux) = model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y), None);
            let loss = model.loss::<Backend>(rec, pd);
            let v: f32 = loss.clone().try_into_scalar().unwrap_or(f32::NAN);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            let lr = wsd_factor(step_i as u64, 40, 1e-3);
            model = optim.step(lr, model, grads);
            losses.push(v);
        }
        let head: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let tail: f32 = losses[35..].iter().sum::<f32>() / 5.0;
        assert!(losses.iter().all(|l| l.is_finite()), "loss must stay finite: {losses:?}");
        assert!(tail < head, "loss must decrease: head={head:.3} tail={tail:.3} {losses:?}");
    }

    /// Gated Residual must train too: the same synthetic-byte loop with
    /// use_gr=true must decrease loss (read/write gradients flow through
    /// the branches). Shorter than the plain run: the GR path is ~2x heavier.
    #[test]
    fn gr_converges() {
        let mut cfg = test_cfg();
        cfg.use_gr = true;
        let mut model = DormouseModel::new(&cfg, &device());
        let optim_cfg = TrainCfg { steps: 25, seq_len: 64, batch: 2, lr: 1e-3, wd: 0.01, grad_clip: 1.0, ..Default::default() };
        let mut optim = crate::optim::build_optim_mode(&optim_cfg, "adamw");
        let mut rng_state: u64 = 0xDEAD_BEEF;
        let next_u8 = |s: &mut u64| {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let x = (*s ^ (*s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            ((x ^ (x >> 27)) >> 33) as u8
        };
        let mut losses: Vec<f32> = Vec::with_capacity(25);
        for _ in 0..25usize {
            let bytes: Vec<u8> = (0..128).map(|_| next_u8(&mut rng_state)).collect();
            let mut hashes = Vec::with_capacity(128 * 3);
            for p in 0..128usize {
                let e = p + 1;
                hashes.push((fnv(&bytes[e.saturating_sub(3)..e]) % 4096) as i64);
                hashes.push((fnv(&bytes[e.saturating_sub(5)..e]) % 4096) as i64);
                hashes.push((fnv(&bytes[e.saturating_sub(8)..e]) % 4096) as i64);
            }
            let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, 64, 2, &device());
            let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
            let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
            let (_logits, rec, pd, _k, _aux) = model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y), None);
            let loss = model.loss::<Backend>(rec, pd);
            let v: f32 = loss.clone().try_into_scalar().unwrap_or(f32::NAN);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(1e-3, model, grads);
            losses.push(v);
        }
        let head: f32 = losses[..3].iter().sum::<f32>() / 3.0;
        let tail: f32 = losses[22..].iter().sum::<f32>() / 3.0;
        assert!(losses.iter().all(|l| l.is_finite()), "GR loss must stay finite: {losses:?}");
        assert!(tail < head, "GR must learn: head={head:.3} tail={tail:.3} {losses:?}");
    }

    /// The auxiliary objectives (JEPA + DSpark) must train end-to-end on the
    /// live model: aux loss finite, teacher EMA advances away from the
    /// student, and a full optim step keeps everything finite.
    #[test]
    fn aux_losses_and_ema_teacher() {
        let cfg = test_cfg();
        let mut model = DormouseModel::new(&cfg, &device());
        let optim_cfg = TrainCfg { steps: 3, seq_len: 64, batch: 2, lr: 1e-3, wd: 0.01, grad_clip: 1.0, ..Default::default() };
        let mut optim = crate::optim::build_optim_mode(&optim_cfg, "adamw");
        assert!(cfg.jepa_weight > 0.0 && cfg.dspark_weight > 0.0, "presets must ship aux ON");
        let mut teacher: Option<dormouse_core::DormouseModel> =
            Some(dormouse_core::aux::ema_update(model.clone(), &model, 0.0));

        let mut rng_state: u64 = 0x0A0B_0E5A_0001;
        let next_u8 = |s: &mut u64| {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let x = (*s ^ (*s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            ((x ^ (x >> 27)) >> 33) as u8
        };
        for _ in 0..3 {
            let bytes: Vec<u8> = (0..128).map(|_| next_u8(&mut rng_state)).collect();
            let hashes: Vec<i64> = (0..384).map(|p| (p as i64) % 4096).collect();
            let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, 64, 2, &device());
            let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
            let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
            let (_logits, rec, pd, _k, aux) = model.forward_with_hidden::<Backend>(
                x, Some(h), None, Some(y), teacher.as_ref(),
            );
            let a = aux.expect("aux must be Some with preset weights on");
            let a_val: f32 = a.clone().try_into_scalar().expect("aux scalar");
            assert!(a_val.is_finite(), "aux loss must be finite, got {a_val}");
            let grads = GradientsParams::from_grads((model.loss::<Backend>(rec, pd) + a).backward(), &model);
            let model_new = optim.step(1e-3, model, grads);
            model = model_new;
            teacher = Some(dormouse_core::aux::ema_update(teacher.unwrap(), &model, dormouse_core::aux::TEACHER_MOMENTUM));
        }
        // EMA must have moved the teacher away from the exact initial copy.
        let t_before: Vec<f32> = teacher.clone().unwrap().aux.jepa_pred.proj.weight.val().into_data().try_to_vec().unwrap();
        let s_now: Vec<f32> = model.aux.jepa_pred.proj.weight.val().into_data().try_to_vec().unwrap();
        assert!(t_before != s_now, "EMA teacher must differ from the trained student");
        let _ = optim_cfg;
    }

    /// Offline JEPA path: (1) the precomputed target equals what the online
    /// EMA teacher would compute (exact copy at teacher step 0), (2) the
    /// aux loss consumes the target (different target -> different aux), so
    /// the teacher forward is genuinely replaced, not silently skipped.
    #[test]
    fn offline_jepa_consumes_precomputed_targets() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let teacher = dormouse_core::aux::ema_update(model.clone(), &model, 0.0);
        let mut rng_state: u64 = 0x0A0B_0E5A_0002;
        let next_u8 = |s: &mut u64| {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let x = (*s ^ (*s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            ((x ^ (x >> 27)) >> 33) as u8
        };
        let bytes: Vec<u8> = (0..128).map(|_| next_u8(&mut rng_state)).collect();
        let hashes: Vec<i64> = (0..384).map(|p| (p as i64) % 4096).collect();
        let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, 64, 2, &device());
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
        // (1) The target the offline pass would store == the online teacher's
        // latent. Relative tolerance: under parallel test load OpenBLAS
        // reduction order differs between the two calls (~1e-5 rel).
        let target = model.forward_latent::<Backend>(x.clone(), Some(h.clone()), None).detach();
        let online = teacher.forward_latent::<Backend>(x.clone(), Some(h.clone()), None).detach();
        let dv: f32 = target.clone().sub(online.clone()).abs().max().into_scalar();
        let scale: f32 = online.abs().max().into_scalar();
        assert!(
            dv < 1e-3 * scale.max(1.0),
            "precomputed target must equal the online teacher latent: {dv:.2e} (scale {scale:.2e})"
        );
        // (2) The aux consumes the target tensor it is handed.
        let (_, _, _, _, aux_a) = model.forward_with_jepa_targets::<Backend>(
            x.clone(), Some(h.clone()), None, Some(y.clone()), Some(target.clone()),
        );
        let scaled = target.mul_scalar(3.0);
        let (_, _, _, _, aux_b) = model.forward_with_jepa_targets::<Backend>(
            x, Some(h), None, Some(y), Some(scaled),
        );
        let a: f32 = aux_a.expect("offline aux must be Some with JEPA on").try_into_scalar().unwrap();
        let b: f32 = aux_b.expect("offline aux must be Some").try_into_scalar().unwrap();
        assert!(a.is_finite() && b.is_finite(), "offline aux must be finite: {a} {b}");
        assert!(
            (a - b).abs() > 1e-4 * a.abs().max(1.0),
            "offline aux must depend on the supplied target: {a} vs {b}"
        );
    }

    /// With --jepa-targets the EMA teacher must never exist: no build, no
    /// per-step EMA advance, no second-forward VRAM.
    #[test]
    fn offline_jepa_skips_ema_teacher() {
        let dorm_cfg = test_cfg(); // aux weights on (preset values)
        let model = DormouseModel::new(&dorm_cfg, &device());
        let offline = TrainCfg { jepa_targets: Some(PathBuf::from("x.bin")), ..Default::default() };
        assert!(
            ema_teacher_for(&offline, &dorm_cfg, &model).is_none(),
            "offline mode must not build an EMA teacher"
        );
        let online = TrainCfg::default();
        assert!(
            ema_teacher_for(&online, &dorm_cfg, &model).is_some(),
            "online mode (aux on by default) must build the EMA teacher"
        );
    }

    /// The JEPA target sidecar: values must roundtrip by chunk hash, and a
    /// chunk that was never precomputed must be a loud error (a silently
    /// dropped aux would corrupt the run).
    #[test]
    fn jepa_sidecar_roundtrip_and_guard() {
        let dev = device();
        let dir = std::env::temp_dir().join("dm-jepa-targets-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("targets.bin");
        let mut w = jepa_targets::JepaTargetWriter::create(&path).unwrap();
        let a: Vec<u8> = (0..64u32).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..64u32).map(|i| (i * 7 % 256) as u8).collect();
        let la: Vec<f32> = (0..2 * 8 * 16).map(|i| i as f32 * 0.5).collect();
        let lb: Vec<f32> = (0..2 * 8 * 16).map(|i| -i as f32 * 0.25).collect();
        w.push(&a, &la, 2, 8, 16).unwrap();
        w.push(&b, &lb, 2, 8, 16).unwrap();
        w.flush().unwrap();
        let mut t = jepa_targets::JepaTargets::open(&path, &dev).unwrap();
        assert_eq!(t.len(), 2);
        let ta = t.get(&a).unwrap();
        assert_eq!(ta.dims(), [2, 8, 16]);
        let va: Vec<f32> = ta.into_data().try_to_vec().unwrap();
        assert_eq!(va, la, "roundtrip must preserve the latent values");
        let tb = t.get(&b).unwrap();
        let vb: Vec<f32> = tb.into_data().try_to_vec().unwrap();
        assert_eq!(vb, lb);
        let unknown: Vec<u8> = vec![9u8; 64];
        assert!(t.get(&unknown).is_err(), "unknown chunk must error loudly");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every OPT mode must build and take a step without NaN (AdamW, Adan,
    /// Muon+, mix, mix-adan - each with its own base optimizer and group
    /// routing). One step per mode keeps the suite fast on NdArray.
    #[test]
    fn every_opt_mode_steps() {
        let cfg = test_cfg();
        let optim_cfg = TrainCfg { steps: 1, seq_len: 64, batch: 2, lr: 1e-3, wd: 0.01, grad_clip: 1.0, ..Default::default() };
        for mode in ["adamw", "adan", "muon", "mix", "mix-adan"] {
            let mut model = DormouseModel::new(&cfg, &device());
            let mut optim = crate::optim::build_optim_mode(&optim_cfg, mode);
            let bytes: Vec<u8> = (0..128).map(|i| (i * 7) as u8).collect();
            let hashes: Vec<i64> = (0..384).map(|p| (p as i64) % 4096).collect();
            let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, 64, 2, &device());
            let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
            let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
            let (_logits, rec, pd, _k, _aux) = model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y), None);
            let loss = model.loss::<Backend>(rec, pd);
            let v: f32 = loss.clone().try_into_scalar().unwrap_or(f32::NAN);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(1e-3, model, grads);
            assert!(v.is_finite(), "mode {mode}: loss must be finite, got {v}");
        }
        // Head-wise Q/K routing (mix + qk_heads) must step clean too.
        let optim_cfg = TrainCfg {
            steps: 1, seq_len: 64, batch: 2, lr: 1e-3, wd: 0.01, grad_clip: 1.0,
            qk_heads: Some(2),
            ..Default::default()
        };
        let mut model = DormouseModel::new(&cfg, &device());
        let mut optim = crate::optim::build_optim_mode(&optim_cfg, "mix");
        let bytes: Vec<u8> = (0..128).map(|i| (i * 7) as u8).collect();
        let hashes: Vec<i64> = (0..384).map(|p| (p as i64) % 4096).collect();
        let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, 64, 2, &device());
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
        let (_logits, rec, pd, _k, _aux) = model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y), None);
        let grads = GradientsParams::from_grads(model.loss::<Backend>(rec, pd).backward(), &model);
        model = optim.step(1e-3, model, grads);
        let q = model.loop_block.shared_attn.gdn2.q_proj.weight.val();
        assert!(q.clone().into_data().try_to_vec().unwrap().iter().all(|x: &f32| x.is_finite()), "head-wise muon step must keep q_proj finite");
    }

    /// Head-wise Muon must equal plain Muon+ applied to each head slice
    /// independently: same momentum, one NS preconditioner + ColRow norm
    /// per [head_dim, d] block, blocks concatenated back. wd=0 and a zero
    /// weight isolate the update: `update = -updated / lr`.
    #[test]
    fn headwise_muon_matches_per_slice_muon() {
        use burn::optim::Optimizer as _;
        let tcfg = TrainCfg { wd: 0.0, ..Default::default() };
        let opt = crate::optim::HeadWiseMuon::new(&tcfg, 2);
        let dev = device();
        let (rows, cols, dh) = (64usize, 128usize, 32usize);
        let vals: Vec<f32> = (0..rows * cols)
            .map(|i| ((i.wrapping_mul(2654435761) % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let grad = Tensor::<2>::from_data(TensorData::new(vals, [rows, cols]), &dev);
        let w = Tensor::<2>::zeros([rows, cols], &dev);
        let lr = 1e-3;
        let (updated, state) = opt.step(lr, w, grad.clone(), None);
        assert!(state.unwrap().mu_momentum.is_some(), "momentum state must be kept");
        let update = updated.mul_scalar(-(1.0 / lr as f32));
        let muon = burn_muon_plus::MuonPlusConfig::new()
            .with_norm_dir(crate::optim::MUON_NORM_DIR)
            .with_ns_steps(crate::optim::MUON_NS_STEPS)
            .build();
        for h in 0..2 {
            let slice = grad
                .clone()
                .slice([h * dh..(h + 1) * dh, 0..cols])
                .mul_scalar(0.05); // first-step momentum: (1 - mu) * g, mu=0.95
            let expect = muon.normalize(muon.orthogonalize(slice));
            let d: f32 = update
                .clone()
                .slice([h * dh..(h + 1) * dh, 0..cols])
                .sub(expect)
                .abs()
                .max()
                .into_scalar();
            assert!(d < 1e-4, "head {h} update diverges from per-slice Muon: {d:.2e}");
        }
    }

    /// --factors-fallback drops the expert TSCT factors from the Muon+ group:
    /// counts must shift from muon to rest, and validation must stay green
    /// with the effective marker list.
    #[test]
    fn factors_fallback_routing() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let full = crate::optim::MUON_PATH_MARKERS;
        let dropped: Vec<&str> = full.iter().copied().filter(|m| *m != "expert_ffns.").collect();
        let with_factors = validate_routing_with(&model, full, "engram.memory", None).unwrap();
        let without = validate_routing_with(&model, &dropped, "engram.memory", None).unwrap();
        assert_eq!(
            without.muon + 4 * cfg.n_experts,
            with_factors.muon,
            "factors must move from the Muon+ group to the fallback"
        );
        assert_eq!(without.rest - with_factors.rest, 4 * cfg.n_experts);
    }
}