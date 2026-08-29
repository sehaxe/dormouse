//! dormouse-train - CUDA training loop: Autodiff backend, Muon+ mixed
//! optimizer (see `optim`), burnpack checkpoints with custom name, resume,
//! opencode harness.
pub mod harness;
mod offload;
mod optim;
mod stress;

use std::path::{Path, PathBuf};

use burn::{
    backend::Backend as BurnBackend,
    module::Module,
    optim::{GradientsParams, OptimizerRecord},
    store::ModuleRecord,
    tensor::{Bytes, Device, FloatDType, Int, Tensor, TensorData},
};

use dormouse_core::{DormouseConfig, DormouseModel};

pub use optim::{
    build_optim, is_engram_table_param, is_muon_param, validate_routing, GroupCounts, MUON_NS_STEPS,
};
pub use stress::{grad_norm, StressMonitor};

#[cfg(feature = "cuda")]
pub type Backend = burn::backend::autodiff::Autodiff<
    burn_cuda::Cuda,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
#[cfg(not(feature = "cuda"))]
pub type Backend = burn::backend::autodiff::Autodiff<
    burn_ndarray::NdArray,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;

pub type Optim = burn::optim::ModuleOptimizer;

pub struct TrainCfg {
    pub steps: usize,
    pub ckpt_every: usize,
    pub ckpt_secs: u64,
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
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            steps: 100000, ckpt_every: 1000, ckpt_secs: 600, log_every: 100,
            seq_len: 512, batch: 3, lr: 1e-4, wd: 0.01, grad_clip: 1.0,
            ckpt_name: "latest".into(), eval_every: 0,
        }
    }
}

fn device() -> Device {
    #[cfg(feature = "cuda")]
    { Device::cuda(0).autodiff() }
    #[cfg(not(feature = "cuda"))]
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
/// sm_120 (Blackwell) -> Fp8 by default (Fp4 via DM_QUANT=fp4),
/// sm_86 (Ampere, 3090) -> Bf16/Fp16, everything else -> Fp32.
/// DM_QUANT=fp32|bf16|fp16|fp8|fp4 forces a format (emulation anywhere).
#[cfg(feature = "cuda")]
pub fn quant_format(device: &Device) -> burn_spectral::QuantFormat {
    use burn_spectral::QuantFormat;
    if let Ok(v) = std::env::var("DM_QUANT") {
        return match v.as_str() {
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
    if dormouse_core::param::bf16_on() {
        QuantFormat::Bf16
    } else if sm >= 120 {
        QuantFormat::Fp8
    } else {
        QuantFormat::Fp32
    }
}
#[cfg(not(feature = "cuda"))]
pub fn quant_format(_device: &Device) -> burn_spectral::QuantFormat {
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
/// Saved as `<dir>/<name>.bin`, atomically (tmp + rename).
pub fn save_ckpt(dir: &Path, name: &str, model: &DormouseModel, optim: &Optim, step: u64, ce: f32) -> std::io::Result<()> {
    let model_bytes = model.clone().into_record().into_bytes().map_err(|e| std::io::Error::other(e.to_string()))?;
    let optim_bytes = optim.to_record().into_bytes().map_err(|e| std::io::Error::other(e.to_string()))?;
    let mut buf = Vec::with_capacity(24 + model_bytes.len() + optim_bytes.len());
    buf.extend_from_slice(&step.to_le_bytes());
    buf.extend_from_slice(&(model_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(optim_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&model_bytes);
    buf.extend_from_slice(&optim_bytes);
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!("{name}.bin.tmp.{}", std::process::id()));
    std::fs::write(&tmp, &buf)?;
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

pub fn train_loop(
    cfg: TrainCfg,
    data: PathBuf,
    preset: String,
    ckpt_dir: Option<PathBuf>,
    eval_data: Option<PathBuf>,
) {
    let dir = ckpt_dir.unwrap_or_else(|| PathBuf::from("checkpoints"));
    let _ = std::fs::create_dir_all(&dir);
    let dorm_cfg: DormouseConfig = match preset.as_str() {
        "base" => DormouseConfig::base(),
        "one_b" => DormouseConfig::one_b(),
        _ => DormouseConfig::small(),
    };
    let device = device();
    init_pools(&device);
    let mut model = DormouseModel::new(&dorm_cfg, &device);
    let qfmt = quant_format(&device);
    if qfmt != burn_spectral::QuantFormat::Fp32 {
        model.loop_block.set_quant_all(qfmt);
        println!("quant format: {qfmt:?} ({} bits)", qfmt.bits());
    }
    let mut optim = build_optim(&cfg);
    // Fail fast if the routing policy no longer matches the model (stale
    // marker after a module rename would silently degrade to AdamW).
    let gc = validate_routing(&model)
        .unwrap_or_else(|e| panic!("optimizer routing check failed: {e}"));
    let opt_name = match std::env::var("OPT").unwrap_or_default().as_str() {
        "adan" => "Adan (all params)".to_string(),
        "adamw" => "AdamW (all params)".to_string(),
        "muon" => "Muon+ ColRow (all params, 1D -> Muon+'s AdamW)".to_string(),
        "mix-adan" => format!("Muon+ ColRow ns={MUON_NS_STEPS} + Adam wd0 (tables) + Adan (rest)"),
        _ => format!("Muon+ ColRow ns={MUON_NS_STEPS} + Adam wd0 (tables) + AdamW (rest)"),
    };
    let factors = if std::env::var("DM_FACTORS_FALLBACK").map(|v| v != "0").unwrap_or(false) {
        " (expert TSCT factors on fallback)"
    } else {
        ""
    };
    println!("optimizer: {opt_name}{factors} [muon={} tables={} rest={}]", gc.muon, gc.tables, gc.rest);
    let mut step = load_ckpt(&dir, &cfg.ckpt_name, &dorm_cfg, &mut model, &mut optim).unwrap_or(0);
    if step > 0 { println!("resumed {} from {} step {step}", cfg.ckpt_name, dir.display()); }
    let mut t_step_start = std::time::Instant::now();
    let _ = &mut t_step_start;

    let mut stream = dormouse_data::ByteStream::new(cfg.seq_len, cfg.batch, &data);
    let mut eval_stream =
        eval_data.as_ref().map(|p| dormouse_data::ByteStream::new(cfg.seq_len, cfg.batch, p));
    let mut best = f32::INFINITY;
    let mut stress = StressMonitor::from_env();
    // RAM-offload n-gram tables (report §2.3): DM_ENGRAM_RAM=1 keeps the
    // tables in host memory (millions of slots in the 64 GB RAM), trains
    // them with CPU Adam, and copies only the batch's rows to the GPU.
    let mut host: Option<offload::HostNgram> = if std::env::var("DM_ENGRAM_RAM").map(|v| v != "0").unwrap_or(false) {
        let slots = std::env::var("DM_ENGRAM_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1_000_000);
        let ng_path = dir.join(format!("{}.ngram", cfg.ckpt_name));
        let h = offload::HostNgram::new([slots, slots, slots], 32, 0x1234_5678);
        let h = std::fs::read(&ng_path)
            .ok()
            .and_then(|b| offload::HostNgram::from_bytes(&b, [slots, slots, slots], 32))
            .unwrap_or(h);
        println!(
            "engram: {} rows in RAM ({} MB), CPU Adam",
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
    if std::env::var("DM_NO_WARMUP").is_err() && step == 0 {
        let (bytes, hashes) = stream.next_batch();
        let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [cfg.batch, cfg.seq_len]), &device);
        for _ in 0..2 {
        let (_logits, rec, pd, _k) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, Some(y.clone()));
        let loss = model.loss::<Backend>(rec, pd);
            let _g = loss.backward();
        }
        memory_cleanup(&device);
        println!("warmup done");
    }
    // quant fidelity check: quantized model vs fp32 reference on one batch.
    // DM_QUANT_CHECK=1 prints max/mean logit deviation + loss delta so Fp8/Fp4
    // viability is measured, not assumed.
    if std::env::var("DM_QUANT_CHECK").is_ok() && qfmt != burn_spectral::QuantFormat::Fp32 && step == 0 {
        let (bytes, hashes) = stream.next_batch();
        let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [cfg.batch, cfg.seq_len]), &device);
        let (lq, rec_q, pd_q, _k) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, Some(y.clone()));
        let loss_q: f32 = model.loss::<Backend>(rec_q, pd_q).try_into_scalar().unwrap_or(f32::NAN);
        let mut ref_model = model.clone();
        ref_model.loop_block.set_quant_all(burn_spectral::QuantFormat::Fp32);
        let (lr, rec_r, pd_r, _k) = ref_model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y));
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
    let mut pshift: Vec<i64> = pbytes.iter().skip(1).chain(std::iter::once(&pbytes[0])).map(|&b| b as i64).collect();
    while step < cfg.steps as u64 {
        t_step_start = std::time::Instant::now();
        let (bytes, hashes) = (std::mem::replace(&mut pbytes, Vec::new()), std::mem::replace(&mut phashes, Vec::new()));
        let (x, h) = bytes_to_tensors::<Backend>(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
        // targets = next byte (shifted by one position)
        let shifted = std::mem::take(&mut pshift);
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [cfg.batch, cfg.seq_len]), &device);
        // prepare next batch right away (host->device copy is async)
        let (nb, nh) = match &host {
            Some(h) => stream.next_batch_with_tables(h.slots),
            None => stream.next_batch(),
        };
        pbytes = nb;
        phashes = nh;
        pshift = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();

        // RAM-offload path: gather this batch's n-gram rows on the host,
        // copy them to the GPU as an autodiff leaf, and train the tables
        // with CPU Adam from the row gradients (report §2.3). The copy is
        // ~600 KB per step; the impact on the step time is negligible.
        let mut rows_param: Option<burn::module::Param<Tensor<2>>> = None;
        let host_rows = match &host {
            Some(h) => {
                let (uniq, pos) = h.unique_rows(&hashes);
                let mut rows = Vec::new();
                h.gather(&uniq, &mut rows);
                let rows_t: Tensor<2> = Tensor::from_data(
                    TensorData::new(rows, [uniq.len(), 32]),
                    &device,
                )
                .require_grad();
                rows_param = Some(burn::module::Param::from_tensor(rows_t.into()));
                let pos_t: Tensor<1, Int> = Tensor::from_data(
                    TensorData::new(pos, [cfg.batch * cfg.seq_len * 3]),
                    &device,
                );
                let idx2 = pos_t.unsqueeze_dim::<2>(1).repeat(&[1, 32]);
                Some(
                    rows_param
                        .as_ref()
                        .unwrap()
                        .val()
                        .gather(0, idx2)
                        .reshape([cfg.batch, cfg.seq_len, 96]),
                )
            }
            None => None,
        };

        let (_logits, rec_ce, p_dist, _kda) =
            model.forward_with_hidden::<Backend>(x, None, host_rows, Some(y));
        let loss = model.loss::<Backend>(rec_ce, p_dist);
        let t_fwd = std::time::Instant::now();
        let ce: f32 = match loss.clone().try_into_scalar() {
            Ok(v) => v,
            Err(_) => {
                // device thread died (cubecl "Memory location was never
                // initialized") - context is poisoned, nothing to save
                println!("step {step} device error - exit");
                std::process::exit(1);
            }
        };
        let t_sync = std::time::Instant::now();
        if !ce.is_finite() {
            println!("step {step} NaN loss - skip");
            step += 1;
            continue;
        }
        if ce < best { best = ce; }
        let grads = loss.backward();
        let t_bwd = std::time::Instant::now();
        let lr = match stress.as_ref() {
            // Constant LR at a multiple of the optimum (report §3.3).
            Some(s) => s.lr(cfg.lr),
            None => wsd_factor(step, cfg.steps as u64, cfg.lr),
        };
        // CPU Adam for the RAM tables from this batch's row gradients.
        if let (Some(h), Some(p)) = (host.as_mut(), &rows_param) {
            if let Some(g) = p.grad(&grads) {
                let (uniq, _) = h.unique_rows(&hashes);
                let g_vec: Vec<f32> = g.into_data().try_to_vec().unwrap_or_default();
                if g_vec.len() == uniq.len() * 32 {
                    h.adam_update(&uniq, &g_vec, lr as f32);
                }
            }
        }
        if let Some(s) = stress.as_mut() {
            // Pre-clip grad norm (p99.9 in the report's protocol). Syncs the
            // device: only under DM_STRESS.
            s.observe(ce, grad_norm(&model, &grads));
        }
        let grads = GradientsParams::from_grads(grads, &model);
        let model_new = optim.step(lr, model, grads);
        model = model_new;
        // TSCT ortho maintenance (bf16_KERNEL_PLAN): retract the U/V masters
        // every step so the quantized forward stays faithful; monitor the
        // drift at cadence and fall back to fp32 factors when it exceeds the
        // plan's 1e-3 threshold. DM_RETRACT_EVERY / DM_RETRACT_ITERS override.
        let retract_every: usize = std::env::var("DM_RETRACT_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        if step % retract_every as u64 == 0 {
            let iters: usize = std::env::var("DM_RETRACT_ITERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3);
            model.retract_tsct(iters);
        }
        if step % 50 == 0 {
            let ortho = model.max_ortho();
            if ortho > 1e-3 {
                println!("max_ortho {ortho:.2e} > 1e-3 - fallback fp32 factors");
                model.set_quant_all(burn_spectral::QuantFormat::Fp32);
            }
        }
        let t_step = std::time::Instant::now();
        if std::env::var("DM_TIMERS").is_ok() && step % 50 == 0 {
            println!(
                "timers step {step}: fwd={:.0}ms sync={:.0}ms bwd={:.0}ms step={:.0}ms",
                t_fwd.duration_since(t_step_start).as_secs_f64() * 1000.0,
                t_sync.duration_since(t_fwd).as_secs_f64() * 1000.0,
                t_bwd.duration_since(t_sync).as_secs_f64() * 1000.0,
                t_step.duration_since(t_bwd).as_secs_f64() * 1000.0,
            );
        }

        if step % cfg.log_every as u64 == 0 {
            let bpb = dormouse_bench::bpb(ce);
            // pool_stats syncs the device; only with DM_MEMLOG=1
            let mem = if std::env::var("DM_MEMLOG").is_ok() { pool_stats(&device) } else { String::new() };
            println!("step {step:6} ce={ce:.3} bpb={bpb:.3} best={best:.3} lr={lr:.2e} {mem}");
            if let Some(s) = stress.as_ref() {
                if let Some(line) = s.report(step) {
                    println!("  {line}");
                }
            }
        }
        if cfg.eval_every > 0 {
            if let Some(ev) = eval_stream.as_mut() {
                if step % cfg.eval_every as u64 == 0 && step > 0 {
                    let (eb, eh) = ev.next_batch();
                    let (ex, eh_t) =
                        bytes_to_tensors::<Backend>(&eb, &eh, cfg.seq_len, cfg.batch, &device);
                    let eshift: Vec<i64> = eb
                        .iter()
                        .skip(1)
                        .chain(std::iter::once(&eb[0]))
                        .map(|&b| b as i64)
                        .collect();
                    let ey: Tensor<2, Int> =
                        Tensor::from_data(TensorData::new(eshift, [cfg.batch, cfg.seq_len]), &device);
                    let (elogits, ..) = model.forward_with_hidden::<Backend>(ex, Some(eh_t), None, None);
                    let v = model.vocab_size;
                    let eflat = elogits.reshape([cfg.batch * cfg.seq_len, v]);
                    let etgt = ey
                        .reshape([cfg.batch * cfg.seq_len])
                        .one_hot::<2>(v)
                        .cast(FloatDType::F32);
                    let ece: f32 = burn::tensor::loss::cross_entropy_with_logits(eflat, etgt)
                        .mean()
                        .try_into_scalar()
                        .unwrap_or(f32::NAN);
                    let ebpb = dormouse_bench::bpb(ece);
                    println!("step {step:6} EVAL ce={ece:.3} bpb={ebpb:.3}");
                }
            }
        }
        if cfg.ckpt_every > 0 && step % cfg.ckpt_every as u64 == 0 {
            let _ = save_ckpt(&dir, &cfg.ckpt_name, &model, &optim, step, ce);
            if let Some(h) = &host {
                let _ = std::fs::write(dir.join(format!("{}.ngram", cfg.ckpt_name)), h.to_bytes());
            }
            println!("ckpt {}.bin saved step {step}", cfg.ckpt_name);
        }
        if step > 0 && step % 2000 == 0 { let _ = harness::run(&format!("step {step} ce {ce:.3}")); }
        if step % 50 == 0 {
            memory_cleanup(&device);
        }
        step += 1;
    }
    let _ = save_ckpt(&dir, &cfg.ckpt_name, &model, &optim, step, best);
    println!("done steps={step} best ce={best:.3}");
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
    use crate::optim::{build_optim_mode, validate_routing_with};
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
            ..DormouseConfig::small()
        }
    }

    /// Live model: the policy must hold on the real module tree. The expected
    /// Muon+ count is derived from the config topology, not a literal:
    /// 3 KDA qkv + 4 sparse-core projections + n_experts x (gate_up u,v +
    /// down u,v) + 1 Engram key_proj + 1 value_proj.
    #[test]
    fn routing_validates_on_live_model() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let expected_muon = 4 * cfg.n_experts + 3;
        let c = validate_routing(&model).expect("policy must hold on the live model");
        assert_eq!(c.muon, expected_muon, "Muon+ group must match the topology");
        assert_eq!(c.tables, 1, "n-gram tables group");
        assert!(c.rest > 0);
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
        let err = validate_routing_with(&model, &["no.such.module"], "engram.memory")
            .expect_err("dead Muon+ markers must fail validation");
        assert!(err.contains("matches no param"), "unexpected error: {err}");
        // Stale table marker.
        let err = validate_routing_with(&model, crate::optim::MUON_PATH_MARKERS, "no.such.table")
            .expect_err("dead table marker must fail validation");
        assert!(err.contains("matches no param"), "unexpected error: {err}");
        // A marker that hits a 1D param (RMSNorm gain) must trip the rank
        // invariant: Muon+ only makes sense on matrices.
        let err = validate_routing_with(&model, &["norm.weight"], "engram.memory")
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
            let (_logits, rec, pd, _k) = model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y));
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
            let (_logits, rec, pd, _k) = model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y));
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
            let (_logits, rec, pd, _k) = model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y));
            let loss = model.loss::<Backend>(rec, pd);
            let v: f32 = loss.clone().try_into_scalar().unwrap_or(f32::NAN);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(1e-3, model, grads);
            assert!(v.is_finite(), "mode {mode}: loss must be finite, got {v}");
        }
    }

    /// DM_FACTORS_FALLBACK drops the expert TSCT factors from the Muon+ group:
    /// counts must shift from muon to rest, and validation must stay green
    /// with the effective marker list.
    #[test]
    fn factors_fallback_routing() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let full = crate::optim::MUON_PATH_MARKERS;
        let dropped: Vec<&str> = full.iter().copied().filter(|m| *m != "expert_ffns.").collect();
        let with_factors = validate_routing_with(&model, full, "engram.memory").unwrap();
        let without = validate_routing_with(&model, &dropped, "engram.memory").unwrap();
        assert_eq!(
            without.muon + 4 * cfg.n_experts,
            with_factors.muon,
            "factors must move from the Muon+ group to the fallback"
        );
        assert_eq!(without.rest - with_factors.rest, 4 * cfg.n_experts);
    }
}