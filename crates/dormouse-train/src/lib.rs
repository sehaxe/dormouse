//! dormouse-train - CUDA training loop: Autodiff backend, burn-optim AdamW
//! (grad clip in-optimizer), burnpack checkpoints with custom name, resume,
//! opencode harness.
pub mod harness;

use std::path::{Path, PathBuf};

use burn::{
    backend::Backend as BurnBackend,
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::{AdamWConfig, GradientsParams, OptimizerRecord},
    store::ModuleRecord,
    tensor::{Bytes, Device, Int, Tensor, TensorData},
};

use dormouse_core::{DormouseConfig, DormouseModel};

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
}

impl Default for TrainCfg {
    fn default() -> Self {
        Self {
            steps: 100000, ckpt_every: 1000, ckpt_secs: 600, log_every: 100,
            seq_len: 512, batch: 3, lr: 1e-4, wd: 0.01, grad_clip: 1.0,
            ckpt_name: "latest".into(),
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
    if sm >= 120 { QuantFormat::Fp8 } else { QuantFormat::Fp32 }
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

pub fn train_loop(cfg: TrainCfg, data: PathBuf, preset: String, ckpt_dir: Option<PathBuf>) {
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
    let mut optim = AdamWConfig::new()
        .with_weight_decay(cfg.wd as f32)
        .with_grad_clipping(
            (cfg.grad_clip > 0.0).then_some(GradientClippingConfig::Norm(cfg.grad_clip as f32)),
        )
        .init();
    let mut step = load_ckpt(&dir, &cfg.ckpt_name, &dorm_cfg, &mut model, &mut optim).unwrap_or(0);
    if step > 0 { println!("resumed {} from {} step {step}", cfg.ckpt_name, dir.display()); }
    let mut t_step_start = std::time::Instant::now();
    let _ = &mut t_step_start;

    let mut stream = dormouse_data::ByteStream::new(cfg.seq_len, cfg.batch, &data);
    let mut best = f32::INFINITY;
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
            let (logits, ponder) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()));
            let loss = model.loss::<Backend>(logits, y.clone(), ponder);
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
        let (lq, pq) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()));
        let loss_q: f32 = model.loss::<Backend>(lq.clone(), y.clone(), pq).try_into_scalar().unwrap_or(f32::NAN);
        let mut ref_model = model.clone();
        ref_model.loop_block.set_quant_all(burn_spectral::QuantFormat::Fp32);
        let (lr, pr) = ref_model.forward_with_hidden::<Backend>(x, Some(h));
        let loss_r: f32 = ref_model.loss::<Backend>(lr.clone(), y, pr).try_into_scalar().unwrap_or(f32::NAN);
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
        let (nb, nh) = stream.next_batch();
        pbytes = nb;
        phashes = nh;
        pshift = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();

        let (logits, ponder) = model.forward_with_hidden::<Backend>(x, Some(h));
        let loss = model.loss::<Backend>(logits, y, ponder);
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
        let grads = GradientsParams::from_grads(grads, &model);
        let lr = wsd_factor(step, cfg.steps as u64, cfg.lr);
        let model_new = optim.step(lr, model, grads);
        model = model_new;
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
        }
        if cfg.ckpt_every > 0 && step % cfg.ckpt_every as u64 == 0 {
            let _ = save_ckpt(&dir, &cfg.ckpt_name, &model, &optim, step, ce);
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