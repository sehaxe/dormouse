//! dormouse-train - CUDA training loop: Autodiff backend, Muon+ mixed
//! optimizer (see `optim`), burnpack checkpoints with custom name, resume,
//! opencode harness.
pub mod atlas;
pub mod byteflow;
mod cfg;
pub mod decode;
pub mod export;
pub mod graph;
mod jepa_targets;
mod offload;
mod optim;
mod stress;

use std::io::Write as _;
use std::path::{Path, PathBuf};

use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::OptimizerRecord,
    store::ModuleRecord,
    tensor::{Bytes, Device, Int, Tensor, TensorData},
};

use dormouse_core::{fused_seam_counts, probe, ActQuant, DormouseConfig, DormouseModel};

pub use cfg::{resolve, RunCfg};
pub use optim::{
    build_optim, check_installed, optimizer_groups, param_paths, validate_routing, GroupCounts,
    Installed, MUON_NS_STEPS,
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
    burn::backend::Flex,
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
    /// Random-depth arm: sample T in 1..=max_iter per step. IN THE SNAPSHOT
    /// (ADR-0021): the depth is drawn from the step index, so it changes the
    /// objective of every step - a run that resumes with a different value is
    /// a different experiment, not a longer one. It used to be `serde(skip)`.
    pub rand_depth: bool,
    /// Batches averaged per held-out eval. One batch is 5 KB, whose sampling
    /// noise is bigger than the effects we A/B; 20 batches = 100 KB. Part of
    /// the EVAL LINE so a curve is self-documenting about its own protocol,
    /// and in the snapshot for the same reason: an eval number is meaningless
    /// without the protocol that produced it.
    pub eval_batches: usize,
    /// At every eval, also score depths 1..=max_iter and print the curve. One
    /// extra forward per depth, NO training: it answers "is the model actually
    /// depth-robust, and how much headroom would an early exit have?" before
    /// we spend a GPU-day on adaptive depth (measure first - see
    /// docs/research/2026-09-27-adaptive-depth-safe.md). In the snapshot: it
    /// changes what the eval line reports.
    pub eval_depths: bool,
    pub retract_every: usize,
    pub retract_iters: usize,
    /// Retract the TSCT masters in ONE grouped pass per factor shape
    /// (`burn_spectral::retract_batched`, sync-free) instead of one
    /// host-syncing Newton-Schulz per factor.
    ///
    /// Default FALSE = the per-factor path, which is what every run in the
    /// archive used. The retraction is a FIXED per-step cost that does not
    /// amortise (measured 2026-09-29: 52.8 ms at batch 8, 64.6 ms at batch
    /// 32, against step times growing 244 -> 826 ms; 22% of a warm step at
    /// batch 8), and the grouped path exists to remove exactly that - but a
    /// flag that silently changes which of two numerical paths a run takes
    /// belongs in the snapshot, so a resume cannot change its arm. The step
    /// log line prints `retr_arm=batched:<n>/factor:<n>` either way
    /// (ADR-0019: no silent arm - the two arms produce the same loss, so
    /// nothing else in the log would say which one ran).
    pub retract_batched: bool,
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
    /// Disable a model arm for A/B (KDA / Engram).
    pub no_kda: bool,
    pub no_engram: bool,
    /// Switch the trainer to ByteFlow Net (`model.use_byteflow`, the
    /// ByteFlow arm). In the snapshot like every objective knob: a resume
    /// that flips it is a different model, and the drift check sees it.
    pub byteflow: bool,
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
    /// Seed for the one stochastic part of a step: the JEPA span mask, which
    /// is a pure function of `(seed, step)` (ADR-0021) so an A/B replays
    /// exactly and a resume continues the same sequence. In the snapshot -
    /// a different seed is a different run.
    pub seed: u64,
    /// Capture the forward+backward window into a CUDA graph and replay it
    /// (see [`graph`]). Off by default: it changes WHICH kernels run (a
    /// capture refuses an allocation, a read and a sync, so the steps that do
    /// any of those run ungraphed), and an A/B of a run that used it is an A/B
    /// of a different execution, not of a different model.
    ///
    /// IN THE SNAPSHOT (ADR-0021), which is the reason it is: a resume that
    /// turned the flag on would compare two execution paths across one step
    /// count, and the flip must reach the drift check rather than sit in the
    /// command line.
    pub graph_capture: bool,
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
            rand_depth: false,
            eval_batches: 20,
            eval_depths: false,
            retract_every: 1, retract_iters: 3, retract_batched: false,
            stress: false, stress_lr: 1.0, stress_every: 50,
            engram_ram: false, engram_slots: 1_000_000, host_adam_every: 1,
            warmup: true, quant_check: false, timers: false, memlog: false,
            bf16: None, act_quant: None, act_group: None, max_iter: None,
            no_kda: false, no_engram: false, byteflow: false,
            jepa_weight: None, dspark_weight: None, dspark_k: None,
            qk_heads: None, jepa_targets: None,
            seed: 1,
            graph_capture: false,
        }
    }
}

/// The device every path in this crate runs on, autodiff-wrapped. `pub`
/// because the export's gate and its divergence measurement have to build
/// their models on the SAME device as the code under test: a test that quietly
/// hardcodes a different device under `--features cuda` compares two BACKENDS
/// and calls the 1-ULP difference a container bug (which is exactly what it
/// did before this was public).
pub fn device() -> Device {
    #[cfg(feature = "cuda")]
    { Device::cuda(0).autodiff() }
    #[cfg(all(feature = "cpu", not(feature = "cuda")))]
    { Device::flex().autodiff() }
}

/// Pool introspection (debug): bytes reserved/used/live allocs on the CUDA client.
#[cfg(feature = "cuda")]
pub fn pool_stats(device: &Device) -> String {
    use burn_dispatch::devices::CubeDevice;
    use burn_dispatch::DispatchDevice;
    use cubecl_cuda::CudaRuntime;
    use cubecl_runtime::runtime::Runtime as _;
    fn unwrap(d: &DispatchDevice) -> &burn_cuda::CudaDevice {
        // pre.4 unifies every cubecl runtime under DispatchDevice::Cube
        // (CubeDevice = cubecl::Device); pick the CUDA entry.
        match d {
            DispatchDevice::Cube(CubeDevice::Cuda(dev)) => dev,
            DispatchDevice::Autodiff(a) => match &**a {
                DispatchDevice::Cube(CubeDevice::Cuda(dev)) => dev,
                other => panic!("expected CUDA device, got {other:?}"),
            },
            other => panic!("expected CUDA device, got {other:?}"),
        }
    }
    let client = CudaRuntime::client(unwrap(device.as_dispatch()));
    {
        let u = client.memory_usage();
        format!("res={:.1}MB used={:.1}MB allocs={}", u.bytes_reserved as f64 / 1e6, u.bytes_in_use as f64 / 1e6, u.number_allocs)
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
    use burn_dispatch::devices::CubeDevice;
    use burn_dispatch::DispatchDevice;
    use cubecl_cuda::CudaRuntime;
    use cubecl_runtime::runtime::Runtime as _;
    fn unwrap(d: &DispatchDevice) -> &burn_cuda::CudaDevice {
        // pre.4 unifies every cubecl runtime under DispatchDevice::Cube
        // (CubeDevice = cubecl::Device); pick the CUDA entry.
        match d {
            DispatchDevice::Cube(CubeDevice::Cuda(dev)) => dev,
            DispatchDevice::Autodiff(a) => match &**a {
                DispatchDevice::Cube(CubeDevice::Cuda(dev)) => dev,
                other => panic!("expected CUDA device, got {other:?}"),
            },
            other => panic!("expected CUDA device, got {other:?}"),
        }
    }
    let client = CudaRuntime::client(unwrap(device.as_dispatch()));
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
    use burn_dispatch::devices::CubeDevice;
    use burn_dispatch::DispatchDevice;
    use cubecl_cuda::CudaRuntime;
    use cubecl_runtime::{
        runtime::Runtime as _,
        config::memory::{MemoryPoolsConfig, MemoryPoolsPreset},
    };
    fn unwrap(d: &DispatchDevice) -> &burn_cuda::CudaDevice {
        // pre.4 unifies every cubecl runtime under DispatchDevice::Cube
        // (CubeDevice = cubecl::Device); pick the CUDA entry.
        match d {
            DispatchDevice::Cube(CubeDevice::Cuda(dev)) => dev,
            DispatchDevice::Autodiff(a) => match &**a {
                DispatchDevice::Cube(CubeDevice::Cuda(dev)) => dev,
                other => panic!("expected CUDA device, got {other:?}"),
            },
            other => panic!("expected CUDA device, got {other:?}"),
        }
    }
    let client = CudaRuntime::client(unwrap(device.as_dispatch()));
    let _ = client.install_memory_pools(&MemoryPoolsConfig::Preset(MemoryPoolsPreset::ExclusivePages));
}
#[cfg(not(feature = "cuda"))]
pub fn init_pools(_device: &Device) {}

/// How many kernel launches the CUDA backend has executed since process start.
///
/// The instrument behind every "this workload is launch-bound" claim in this
/// repo: counted at the single choke point every launch passes through
/// (`cubecl-cuda/src/compute/context.rs`), and self-validated against the number
/// of launches a test makes
/// (`cubecl-cuda/tests/graph_step.rs::launch_counter_counts_every_launch`).
///
/// **It is counted on the device thread**, so a value read from the host is a
/// LOWER BOUND on what the host has enqueued — the device thread may still be
/// behind. In a launch-bound loop the host runs ahead and the two nearly
/// coincide, but the claim is the bound.
///
/// Zero off CUDA: the counter lives in the CUDA backend, and a run on another
/// backend has no launches to count. That is the honest answer, not a stand-in.
pub fn cubecl_launches() -> u64 {
    #[cfg(feature = "cuda")]
    {
        cubecl_cuda::launches()
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// The cubecl client behind a `Device`, or `None` off CUDA. The same
/// unwrap `init_pools` does, factored out because the graph seam needs it too
/// and two copies of a backend downcast is how they drift.
#[cfg(feature = "cuda")]
pub fn cubecl_client(device: &Device) -> cubecl_runtime::client::Client {
    cubecl_client_opt(device).expect("a CUDA device has a cubecl client")
}

/// [`cubecl_client`] as an `Option`, for the seam's one field that is `None`
/// off CUDA. Panics on a CUDA device that has no client, which is the same
/// panic `cubecl_client` has always had.
#[cfg(feature = "cuda")]
pub fn cubecl_client_opt(device: &Device) -> Option<cubecl_runtime::client::Client> {
    use burn_dispatch::devices::CubeDevice;
    use burn_dispatch::DispatchDevice;
    use cubecl_cuda::CudaRuntime;
    use cubecl_runtime::runtime::Runtime as _;
    fn unwrap(d: &DispatchDevice) -> &burn_cuda::CudaDevice {
        match d {
            DispatchDevice::Cube(CubeDevice::Cuda(dev)) => dev,
            DispatchDevice::Autodiff(a) => match &**a {
                DispatchDevice::Cube(CubeDevice::Cuda(dev)) => dev,
                other => panic!("expected CUDA device, got {other:?}"),
            },
            other => panic!("expected CUDA device, got {other:?}"),
        }
    }
    Some(CudaRuntime::client(unwrap(device.as_dispatch())))
}
#[cfg(not(feature = "cuda"))]
pub fn cubecl_client_opt(_device: &Device) -> Option<cubecl_runtime::client::Client> {
    None
}

/// The NaN firewall (2026-09-27). A non-finite loss must never reach the
/// optimizer: backward turns it into NaN grads for every parameter, the step
/// writes them into the weights, and from then on the model is dead - the
/// guard then replays the poisoned region from a stale checkpoint forever.
///
/// Fix: mask the loss to 0 ON DEVICE. Backward through the masked tensor
/// yields exactly zero gradients (not NaN), so the step degrades into a
/// no-op, with no per-gradient-tensor walk and no per-step device sync.
/// Returns the masked loss and a 1.0/0.0 device counter increment so the
/// event rate can be reported at log cadence instead of being swallowed.
///
/// A zero loss is NOT a silent skip: `train_loop` counts the events on the
/// HOST (from the loss scalar it already reads at log/timer cadence - device
/// arithmetic turned out to be untrustworthy here, see the note at the call
/// site), prints the count loudly, and refuses to continue past a whole log
/// window of them (that is a broken model, not a spike).
pub fn mask_nonfinite(loss: Tensor<1>) -> Tensor<1> {
    let nonfinite = loss.clone().is_finite().bool_not();
    loss.mask_fill(nonfinite, 0.0)
}

/// Zero every non-finite GRADIENT, on device, with no host round-trip.
///
/// `mask_nonfinite` on the loss is not enough, and this is the reason: a NaN that
/// originates in an intermediate ACTIVATION still back-propagates. The chain
/// rule multiplies the masked zero seed by the Jacobian, and `NaN * 0 = NaN`, so
/// the optimizer would write NaN into the weights - which is exactly the
/// observed failure (a NaN held-out eval proves the NaN is in the weights, not
/// in the batch, and every resume replayed the poisoned checkpoint).
///
/// Device-side and sync-free by construction: `is_finite` + `bool_not` +
/// `mask_fill` are all device ops, and the model is walked with the visitor we
/// already need elsewhere. The host learns that a step was masked from the
/// gradient norm it reads at log cadence (an all-zero grad norm means exactly
/// "every gradient was non-finite or the loss was"), so the accounting costs no
/// synchronization either. The `Bool -> float` cast is broken on this backend
/// (it returns 0.0 for `true`), which is why this is a `mask_fill` on a float
/// and not a cast.
struct GradSanitizer<'a> {
    grads: &'a mut burn::tensor::Gradients,
}

impl ModuleVisitor for GradSanitizer<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if let Some(g) = param.grad(self.grads) {
            let nonfinite = g.clone().is_finite().bool_not();
            param.grad_replace(self.grads, g.mask_fill(nonfinite, 0.0));
        }
    }
}

/// Zero the non-finite gradients in place. See `GradSanitizer`.
pub fn sanitize_grads(grads: &mut burn::tensor::Gradients, model: &DormouseModel) {
    model.visit(&mut GradSanitizer { grads });
}

/// The random-depth draw (ADR-0013 rank 2): `T` for this step, in
/// `1..=max_iter`. A deterministic mix of the step index - no RNG state to
/// carry across a resume, and an A/B run replays exactly. `max_iter <= 1`
/// degenerates to the fixed-depth case.
pub fn sample_depth(step: u64, max_iter: usize) -> usize {
    if max_iter <= 1 {
        return 1;
    }
    let mut z = step.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    1 + (z as usize) % max_iter
}

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
            // LOUD, not Fp32: `--quant fp8x` used to train a whole run in
            // fp32 and print a normal-looking loss curve (ADR-0019).
            other => panic!(
                "--quant {other:?} is not a format; use one of fp32|bf16|fp16|fp8|fp4"
            ),
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

fn bytes_to_tensors(
    bytes: &[u8], hashes: &[i64], seq_len: usize, batch: usize, device: &Device,
) -> (Tensor<2, Int>, Tensor<3, Int>) {
    let ids: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
    let x: Tensor<2, Int> = Tensor::from_data(TensorData::new(ids, [batch, seq_len]), device);
    let h3: Vec<i64> = hashes.to_vec();
    let h: Tensor<3, Int> = Tensor::from_data(TensorData::new(h3, [batch, seq_len, 3]), device);
    (x, h)
}

/// ckpt container (ADR-0021):
/// `[magic 8B][step u64][model_len u64][optim_len u64][teacher_len u64][flags u64]`
/// then the three burnpack records. Saved as `<dir>/<name>.bin`, atomically
/// (tmp + rename). The parts stream to the file through a BufWriter instead
/// of concatenating a third full copy in RAM (model+optim already exist as
/// byte vectors; burnpack cannot stream its records, so ~2x (model+optim) RAM
/// is the floor without upstream changes).
///
/// The teacher and the flags are here because the CONFIG CANNOT EXPRESS
/// THEM: both are run state that only exists because the loop advanced, and
/// re-deriving them from the config is exactly how a resume becomes a
/// different experiment wearing the same step count.
const CKPT_MAGIC: [u8; 8] = *b"DMCK\x00\x02\x00\x00";
/// Bytes of the v2 header (magic + 5 u64).
const CKPT_HEADER: usize = 48;
/// Bytes of the pre-ADR-0021 header (step + 2 u64).
const CKPT_HEADER_LEGACY: usize = 24;
/// `flags` bit 0: the one-way fp32-factor fallback is latched.
const CKPT_ORTHO_FP32: u64 = 1;

/// What a checkpoint carried besides the weights: the EMA teacher (JEPA's
/// momentum is state, not config) and the latched fp32-factor fallback (a
/// one-way switch the run can never undo).
#[derive(Debug, Clone)]
pub struct Loaded {
    pub step: u64,
    /// The teacher exactly as the run left it. `None` when the run had none
    /// (aux off / offline JEPA targets) or when the container predates
    /// ADR-0021 and cannot carry one.
    pub teacher: Option<DormouseModel>,
    /// The fp32-factor fallback was latched when this was written.
    pub ortho_fp32: bool,
    /// Pre-ADR-0021 container: no teacher, no flags in the file.
    pub legacy: bool,
}

struct CkptHeader {
    step: u64,
    model: usize,
    optim: usize,
    teacher: usize,
    flags: u64,
    legacy: bool,
    body: usize,
}

/// Parse the container header. A pre-ADR-0021 file starts with the raw step
/// where the magic is, so it reads back as "no teacher, no flags" instead of
/// being misparsed as a v2 record.
fn parse_header(raw: &[u8]) -> Option<CkptHeader> {
    if raw.len() >= CKPT_HEADER && raw[..8] == CKPT_MAGIC {
        let u = |s: &[u8]| u64::from_le_bytes(s.try_into().expect("8-byte window"));
        Some(CkptHeader {
            step: u(&raw[8..16]),
            model: u(&raw[16..24]) as usize,
            optim: u(&raw[24..32]) as usize,
            teacher: u(&raw[32..40]) as usize,
            flags: u(&raw[40..48]),
            legacy: false,
            body: CKPT_HEADER,
        })
    } else if raw.len() >= CKPT_HEADER_LEGACY {
        let u = |s: &[u8]| u64::from_le_bytes(s.try_into().expect("8-byte window"));
        Some(CkptHeader {
            step: u(&raw[0..8]),
            model: u(&raw[8..16]) as usize,
            optim: u(&raw[16..24]) as usize,
            teacher: 0,
            flags: 0,
            legacy: true,
            body: CKPT_HEADER_LEGACY,
        })
    } else {
        None
    }
}

// The checkpoint signature is the trainer's persistence seam: each argument is
// a distinct part of the artefact (weights, optimizer, EMA teacher, the best
// bookkeeping), and a config struct would only relocate the same eight names.
#[allow(clippy::too_many_arguments)]
pub fn save_ckpt(
    dir: &Path,
    name: &str,
    model: &DormouseModel,
    optim: &Optim,
    teacher: Option<&DormouseModel>,
    ortho_fp32: bool,
    step: u64,
    ce: f32,
) -> std::io::Result<()> {
    // Rotate the previous save aside BEFORE writing the new one: with a single
    // file, one bad save (a run dying mid-write, or weights written while the
    // allocator was failing) destroys the last known-good state and every
    // resume after it replays the same corruption. Cheap insurance: a hard
    // link, no extra bytes while both exist.
    let live = dir.join(format!("{name}.bin"));
    if live.exists() {
        let prev = dir.join(format!("{name}.prev.bin"));
        let _ = std::fs::remove_file(&prev);
        let _ = std::fs::hard_link(&live, &prev);
    }
    let eio = |e: String| std::io::Error::other(e);
    let model_bytes = model.clone().into_record().into_bytes().map_err(|e| eio(e.to_string()))?;
    let optim_bytes = optim.to_record().into_bytes().map_err(|e| eio(e.to_string()))?;
    let teacher_bytes = match teacher {
        Some(t) => t.clone().into_record().into_bytes().map_err(|e| eio(e.to_string()))?,
        None => Bytes::from_bytes_vec(Vec::new()),
    };
    let mut flags = 0u64;
    if ortho_fp32 { flags |= CKPT_ORTHO_FP32; }
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!("{name}.bin.tmp.{}", std::process::id()));
    let f = std::fs::File::create(&tmp)?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    w.write_all(&CKPT_MAGIC)?;
    w.write_all(&step.to_le_bytes())?;
    w.write_all(&(model_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&(optim_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&(teacher_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;
    w.write_all(&model_bytes)?;
    w.write_all(&optim_bytes)?;
    w.write_all(&teacher_bytes)?;
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, dir.join(format!("{name}.bin")))?;
    std::fs::write(dir.join(format!("{name}.txt")), format!("step {step} ce {ce:.3}\n"))?;
    Ok(())
}

pub fn load_ckpt(
    dir: &Path,
    name: &str,
    cfg: &DormouseConfig,
    model: &mut DormouseModel,
    optim: &mut Optim,
) -> Option<Loaded> {
    // Try the live checkpoint, then the rotated one: a save written while the
    // allocator was failing is garbage, and silently starting from scratch (or
    // promoting the garbage to .prev on the next save) is how we lost a day
    // (review 2026-09-27).
    for cand in [format!("{name}.bin"), format!("{name}.prev.bin")] {
        if let Some(l) = load_ckpt_file(dir, &cand, cfg, model, optim) {
            return Some(l);
        }
        eprintln!("checkpoint {cand} unusable - trying the previous one");
    }
    None
}

fn load_ckpt_file(
    dir: &Path,
    file: &str,
    cfg: &DormouseConfig,
    model: &mut DormouseModel,
    optim: &mut Optim,
) -> Option<Loaded> {
    let raw = std::fs::read(dir.join(file)).ok()?;
    let h = parse_header(&raw)?;
    let CkptHeader { step, model: mlen, optim: olen, teacher: tlen, flags, legacy, body } = h;
    if body + mlen + olen + tlen > raw.len() { return None; }
    let mb = Bytes::from_bytes_vec(raw[body..body + mlen].to_vec());
    let ob = Bytes::from_bytes_vec(raw[body + mlen..body + mlen + olen].to_vec());
    let mrec = ModuleRecord::from_bytes(mb).ok()?;
    let orec = OptimizerRecord::from_bytes(ob).ok()?;
    let device = device();
    *model = DormouseModel::new(cfg, &device).load_record(mrec);
    *optim = optim.clone().load_record(orec);
    // A checkpoint written while the allocator was failing is itself garbage,
    // and resuming from it burns a GPU-hour before the first eval notices
    // (measured 2026-09-27: official_v5 died at step 2000, saved over the good
    // checkpoint, and every resume since produces NaN on the first step).
    // Check the weights NOW, loudly, at load time.
    let (total, bad) = model.finite_scan();
    if bad > 0 {
        eprintln!(
            "checkpoint {file} at step {step} has {bad}/{total} non-finite parameters - refusing to train from it (the save was corrupt)"
        );
        return None;
    }
    let teacher = if tlen > 0 {
        let tb = Bytes::from_bytes_vec(raw[body + mlen + olen..body + mlen + olen + tlen].to_vec());
        let trec = ModuleRecord::from_bytes(tb).ok()?;
        // `no_grad()` after the load, always: the freeze is what keeps the
        // teacher forward off the autodiff tape (a second full forward's
        // activations - the step-0 OOM axis) and stops the EMA chain from
        // accumulating param nodes. A record does not carry the flag.
        Some(DormouseModel::new(cfg, &device).load_record(trec).no_grad())
    } else {
        None
    };
    Some(Loaded { step, teacher, ortho_fp32: flags & CKPT_ORTHO_FP32 != 0, legacy })
}



/// Build the model with the run's factor-quant / bf16 compute settings.
fn build_model(
    dorm_cfg: &DormouseConfig,
    cfg: &TrainCfg,
    device: &Device,
) -> (DormouseModel, burn_spectral::QuantFormat) {
    let mut model = DormouseModel::new(dorm_cfg, device);
    let qfmt = quant_format(device, cfg.quant.as_deref(), dorm_cfg.bf16);
    apply_compute_settings(&mut model, qfmt, dorm_cfg.bf16, false);
    (model, qfmt)
}

/// The forward-path settings a checkpoint does NOT carry: the factor-quant
/// format, bf16 compute, and the latched fp32 fallback. They live in plain
/// (non-param) fields, so `load_ckpt` - which rebuilds the model from `new()`
/// and loads only the params - wipes them. A resumed run therefore trained the
/// fp32 forward while its own log said "quant format: Fp8" (ADR-0021), and the
/// same hole swallowed the one-way fallback. `ortho_fp32` wins: it is the
/// latch the run committed to and can never undo.
fn apply_compute_settings(
    model: &mut DormouseModel,
    qfmt: burn_spectral::QuantFormat,
    bf16: bool,
    ortho_fp32: bool,
) {
    if qfmt != burn_spectral::QuantFormat::Fp32 {
        model.loop_block.set_quant_all(qfmt);
        println!("quant format: {qfmt:?} ({} bits)", qfmt.bits());
    }
    if ortho_fp32 {
        // True bf16 compute: matmuls run on bf16 (tensor cores) through the
        // custom autodiff op; the graph and backward stay fp32.
        println!("fp32 factor fallback: latched, factors run fp32");
        model.set_quant_all(burn_spectral::QuantFormat::Fp32);
        return;
    }
    if bf16 {
        model.set_bf16_compute(true);
        println!("bf16 compute: tensor-core matmuls, fp32 graph");
    }
}

/// Write the host-RAM n-gram sidecar for `step`, atomically (tmp + rename).
/// LOUD on every failure: this sidecar IS the whole n-gram state, and a
/// `File::create` that failed used to leave a "saved" line over a sidecar
/// that does not exist - a resumed run then starts with empty tables and no
/// idea why (pretrain v21's death).
///
/// The sidecar carries no training-step stamp, so a crash between the model
/// write and this one leaves a mismatched pair that nothing can detect
/// (ADR-0021, "what remains"). It is written on EVERY model save, including
/// the final one, so the window is the crash window and not the cadence.
fn save_ngram(
    dir: &Path,
    name: &str,
    host: Option<&offload::HostNgram>,
    step: u64,
) -> Result<(), String> {
    let Some(h) = host else { return Ok(()) };
    let path = dir.join(format!("{name}.ngram"));
    let tmp = dir.join(format!("{name}.ngram.tmp.{}", std::process::id()));
    let f = std::fs::File::create(&tmp)
        .map_err(|e| format!("step {step}: ngram sidecar {}: {e}", tmp.display()))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    h.write_to(&mut w)
        .map_err(|e| format!("step {step}: ngram sidecar write: {e}"))?;
    std::io::Write::flush(&mut w)
        .map_err(|e| format!("step {step}: ngram sidecar flush: {e}"))?;
    std::fs::rename(&tmp, &path)
        .map_err(|e| format!("step {step}: ngram sidecar rename: {e}"))?;
    Ok(())
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
        let (x, h) = bytes_to_tensors(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
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
    // SEED THE BACKEND BEFORE ANY PARAMETER IS CREATED, or nothing downstream
    // is reproducible.
    //
    // Measured 2026-09-28: two runs of this binary with identical flags, the
    // same 8 KB deterministic corpus and ZERO steps produced checkpoints
    // differing in 34,730,605 of 43,725,616 bytes. The cause is that
    // `Device::seed` was never called, so every parameter drew from process
    // entropy. Three things follow, and all three are load-bearing:
    //
    //   - no bit-exact golden is possible, which is why docs/protocols/VERIFICATION.md
    //     layer 2 had no way to exist;
    //   - ADR-0002's "3 seeds per arm, and a win must beat the spread of the
    //     control's own seeds" was NOT IMPLEMENTABLE as written, because
    //     `--seed` only sets the JEPA span mask (see cfg.rs) and the weights
    //     were different every run regardless;
    //   - every A/B in the archive compared two different random inits and
    //     called the difference an arm.
    //
    // `cfg.seed` is the one knob already in the snapshot, and it is what the
    // protocol says the seed IS, so a different value in the config is now a
    // different run in the strongest sense: a different model.
    device.seed(cfg.seed);
    // THE BYTEFLOW DISPATCH. `use_byteflow` replaces the model, so the
    // dormouse loop does not build one: the arm's own loop takes over HERE,
    // after the snapshot and after the seed (seed-before-any-parameter is the
    // 2026-09-28 rule), and the config snapshot above covers its resume
    // discipline. `check` inside refuses every dormouse-only knob loudly.
    if dorm_cfg.use_byteflow {
        return byteflow::train_loop(
            RunCfg {
                source: preset,
                model: dorm_cfg,
                train: cfg,
            },
            data,
            Some(dir.clone()),
            eval_data,
        );
    }
    let (mut model, qfmt) = build_model(&dorm_cfg, &cfg, &device);
    let mut optim = build_optim(&model, &cfg);
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
    let loaded = load_ckpt(&dir, &cfg.ckpt_name, &dorm_cfg, &mut model, &mut optim);
    let mut step = loaded.as_ref().map(|l| l.step).unwrap_or(0);
    if step > 0 { println!("resumed {} from {} step {step}", cfg.ckpt_name, dir.display()); }
    // The forward path is NOT in the record: re-apply it, or a resumed run
    // silently trains the fp32 forward while the log above announced Fp8.
    let saved_ortho = loaded.as_ref().is_some_and(|l| l.ortho_fp32);
    apply_compute_settings(&mut model, qfmt, dorm_cfg.bf16, saved_ortho);
    // EMA teacher for the JEPA aux (momentum 0.0 at init = exact copy with
    // fresh grad-free params); advanced after every optimizer step below.
    // Offline mode (--jepa-targets) skips the teacher entirely: no second
    // forward, no EMA advance, none of its VRAM.
    //
    // RESTORED FROM THE CHECKPOINT, not rebuilt. A teacher that is a fresh
    // copy of the student is a different regression target than the one the
    // run accumulated momentum in, so the JEPA term changes from step 1 of
    // the resumed run - two experiments wearing one step count (ADR-0021).
    let saved_teacher = loaded.as_ref().and_then(|l| l.teacher.clone());
    let legacy = loaded.as_ref().is_some_and(|l| l.legacy);
    let mut teacher = match saved_teacher {
        Some(t) => Some(t),
        None => {
            let t = ema_teacher_for(&cfg, &dorm_cfg, &model);
            if t.is_some() && step > 0 {
                eprintln!(
                    "resume: the checkpoint carries NO EMA teacher ({} container) - the JEPA target restarts \
                     from the student, so this run is NOT the run it continues",
                    if legacy { "pre-ADR-0021" } else { "teacher-less" }
                );
            }
            t
        }
    };
    let mut jepa_tgts = match &cfg.jepa_targets {
        Some(p) => Some(jepa_targets::JepaTargets::open(p, &device)?),
        None => None,
    };
    // Batches a FRESH run consumes before step 0's forward: the two warmup /
    // quant-check probes (each exactly one batch). A resume skips them (they
    // are gated on step == 0), so the fast-forward has to include them or the
    // run lands `pre` batches behind and re-trains on data it has seen
    // (ADR-0021).
    let pre_batches = cfg.warmup as u64
        + (cfg.quant_check && qfmt != burn_spectral::QuantFormat::Fp32) as u64;
    // ADR-0010's rule is filter-time and structural, but nothing could ENFORCE
    // it: `collect_files` recurses into every subdirectory, so pointing
    // `--data` at the parent of an eval tree trains on the held-out bytes
    // with no error, and the eval then reports a memorised number as if it
    // were generalisation. `train_and_eval` canonicalises both roots and
    // refuses when either contains the other. It is one call site, and the
    // alternative - a marker file - protects only the runs that remember to
    // write one, which is the failure this rule exists to prevent.
    let (mut stream, mut eval_stream) =
        dormouse_data::ByteStream::train_and_eval(cfg.seq_len, cfg.batch, &data, eval_data.as_deref());
    if step > 0 {
        // Resume: fast-forward past bytes already trained on, or the stream
        // rewinds to byte 0 and the model re-reads (and memorizes) the corpus
        // head.
        let skip = (step + pre_batches) * (cfg.batch * cfg.seq_len) as u64;
        stream.skip_bytes(skip);
        println!("stream resume: skipped {skip} bytes (step {step} + {pre_batches} pre-loop batch(es))");
    }
    let mut best = f32::INFINITY;
    // Best HELD-OUT BPB seen so far, and where. `best` above is train CE,
    // which is the quantity ADR-0011 calls fake; this is the one a reader of a
    // training log should be told about, and it is what `<name>.best` holds.
    let mut best_eval_bpb = f32::INFINITY;
    let mut best_eval_step: u64 = 0;
    // A resumed run must not believe it has no best artifact. Without this the
    // first eval after any resume compares against `inf`, overwrites
    // `<name>.best.bin` with a usually-worse checkpoint, and the whole feature
    // silently re-creates the bug it exists to prevent (found by audit
    // 2026-09-28, hours after the feature landed, and lost once to a
    // concurrent commit that rewrote this file).
    {
        let bpb_path = dir.join(format!("{}.best.bpb", cfg.ckpt_name));
        // A file that exists but carries no parsable pair is NOT a fresh run.
        // It is a truncated write (fs::write is not atomic), and treating it
        // as "no best yet" is the silent data loss this file exists to
        // prevent - so the parse is total: missing tokens, a bad float, a
        // non-finite score and empty content all take the same loud path.
        if let Ok(text) = std::fs::read_to_string(&bpb_path) {
            let mut it = text.split_whitespace();
            let pair = match (it.next(), it.next()) {
                (Some(b), Some(s)) => b.parse::<f32>().ok().zip(s.parse::<u64>().ok()),
                _ => None,
            };
            match pair {
                Some((b, s)) if b.is_finite() => {
                    best_eval_bpb = b;
                    best_eval_step = s;
                }
                _ => {
                    return Err(format!(
                        "{bpb_path:?} is unreadable ({text:?}) - refusing to resume, because \
                         overwriting the best checkpoint needs a score to compare against"
                    ));
                }
            }
        }
    }
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
                // v1 checkpoints carried Adam m+v state; loading them into a
                // Nesterov+Sinkhorn table would silently corrupt momentum.
                None => panic!(
                    "ngram ckpt {}: unknown layout (v1 Adam state?) — delete the file or pick another --ckpt-name; silently starting fresh on a 37 GB sidecar is how pretrain v21 died",
                    ng_path.display()
                ),
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
    // The backend is REPORTED, not assumed. A hardcoded `cuda(autodiff)`
    // made every CPU run's log lie about where it ran, which is the exact
    // class of wrong number this project keeps retracting - and it cost a
    // reader real time: a test run believed it was on the GPU.
    println!(
        "dormouse pretrain {preset} params={n_params} steps={} data={:?} lr={} backend={:?}",
        cfg.steps,
        data,
        cfg.lr,
        device,
    );
    // #7 warmup: 2 fwd+bwd at full depth raise the pool high-water before the
    // loop (then cleanup returns the pages - later steps reuse cached blocks).
    if cfg.warmup && step == 0 {
        let (bytes, hashes) = stream.next_batch();
        let (x, h) = bytes_to_tensors(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [cfg.batch, cfg.seq_len]), &device);
        for _ in 0..2 {
        let (_logits, rec, _k, _aux) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, Some(y.clone()), None);
        let loss = model.loss::<Backend>(rec);
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
        let (x, h) = bytes_to_tensors(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [cfg.batch, cfg.seq_len]), &device);
        let (lq, rec_q, _kq, _aq) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, Some(y.clone()), None);
        let loss_q: f32 = model.loss::<Backend>(rec_q).try_into_scalar().unwrap_or(f32::NAN);
        let mut ref_model = model.clone();
        ref_model.loop_block.set_quant_all(burn_spectral::QuantFormat::Fp32);
        let (lr, rec_r, _kr, _ar) = ref_model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y), None);
        let loss_r: f32 = ref_model.loss::<Backend>(rec_r).try_into_scalar().unwrap_or(f32::NAN);
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
    // ── CUDA graph seam (`--graph-capture`, off by default) ───────────────
    // Armed after the model and the teacher exist, so the pin covers every
    // address the window reads. The INPUT pins need one batch's shapes, which
    // only the loop has, so they are allocated from the first batch below.
    let graph_on = cfg.graph_capture;
    let mut seam = if graph_on {
        let client = cubecl_client_opt(&device);
        let mut seam = graph::Seam::new(client);
        let (m, t) = seam.arm(model, teacher);
        model = m;
        teacher = t;
        println!(
            "graph capture armed: {} pinned parameters, {} pinned teacher parameters. \
             Log/eval/500-step/host-adam steps run ungraphed and force a re-capture.",
            seam.model_pins.copies(),
            seam.teacher_pins.copies(),
        );
        seam
    } else {
        graph::Seam::new(None)
    };
    let mut inputs: Option<graph::InputPins> = None;
    if graph_on {
        if let Some(r) = seam.model_pins.unsupported_rank {
            return Err(format!(
                "--graph-capture: a parameter has rank {r} and the pin carries ranks 1-4. Capture \
                 refused before the first step."
            ));
        }
    }
    // Wall clock around the whole loop: the per-step timer line only prints on
    // LOG steps (`timer_step = step % log_every`), which are exactly the steps
    // that run ungraphed, so it can never time a replayed step. The honest
    // number for this flag is the mean over a slice, measured here.
    //
    // From step 50, not from 0: step 0 is the autotune step (5.5 s against a
    // 0.46 s warm step, 23x) and a mean that includes it measures the tuner.
    let step_at_entry = step;
    let mut t_warm: Option<std::time::Instant> = None;
    // One-way fp32 fallback state, RESTORED from the checkpoint above: a run
    // that latched it did so because its factors drifted, and re-running the
    // quantized forward for even one step re-quantizes exactly the factors the
    // latch was protecting (ADR-0021). With `--quant fp32` the forward is
    // already exact — the monitor would have nothing to guard, so skip it
    // entirely (it costs 30+ device syncs per check).
    let mut ortho_fp32 = saved_ortho || cfg.quant.as_deref() == Some("fp32");
    // The retraction's BEFORE/AFTER Gram error pair, for the eval line.
    // `None` on every step that is not an eval step, and on an eval step
    // where `retract_every` did not fire — the field then reads `-` rather
    // than printing two identical numbers that look like a measurement.
    // The initial Nones are never read (the loop resets them before any
    // read) — the assignment lint is allowed, not obeyed, because removing
    // the declarations moves them inside the loop and into the graph lane's
    // in-flight region.
    #[allow(unused_assignments)]
    let mut tsct_bef: Option<f32> = None;
    #[allow(unused_assignments)]
    let mut tsct_aft: Option<f32> = None;
    if saved_ortho && cfg.quant.as_deref() != Some("fp32") {
        println!("fp32 factor fallback restored from the checkpoint (one-way, still latched)");
    }
    // Non-finite losses seen at the last read, counted on the host. Device
    // arithmetic is not used for this: on the dispatch path both the
    // Bool->Float cast and `zeros_like().mask_fill(m, 1.0)` fed into `add`
    // reported 1.0 for every step of a perfectly finite loss (official_v5,
    // twice: 50 "non-finite" per 50-step window, then a false fuse at step
    // 2000). The mask itself is sound - it is what makes a bad step a no-op -
    // so the counting is the only thing that has to be conservative, and the
    // host is the only place we have ground truth.
    // Firewall accounting, all of it sync-free: `masked_since_read` counts the
    // log cadences whose gradient norm came back exactly zero (the firewall
    // fired on at least one step in that window), `nan_reads` counts the
    // non-finite loss VALUES we actually read, which is a lower bound.
    let mut masked_since_read = 0u32;
    // Counted, and read on no cadence today (`let _ =` at both write sites):
    // the StressMonitor report is the consumer that never landed (ADR-0021
    // item 4). Deleting the counter would delete the only record of firewall
    // firings.
    let mut nan_reads = 0u32;
    // The JEPA span mask is the only stochastic input to a step, and it is a
    // pure function of `(cfg.seed, step)` — so a resume redraws the same mask
    // the interrupted run used and two A/B arms differ only in the arm
    // (ADR-0021). Before this, the mask came off burn's global RNG: never
    // seeded, never repeated, and different on every run of the same config.
    dormouse_core::aux::set_mask_stream(cfg.seed, step);
    // Launch atlas (DM_LAUNCH_ATLAS=1): per-stage launch deltas around the
    // stage timers below. Inert without the env var.
    let mut atlas = atlas::LaunchAtlas::from_env();
    while step < cfg.steps as u64 {
        // The eval-step predicate, named once: the TSCT diagnostics bracket
        // the retraction above and print in the eval block below, and both
        // must agree on which steps those are. `step > 0` is the eval's own
        // guard (a step-0 eval would score an untrained model on the same
        // window and look like a result).
        let on_eval_step = cfg.eval_every > 0 && step > 0 && step.is_multiple_of(cfg.eval_every as u64);
        // Reset per step: the retraction block only writes them on a step
        // where it fires, so without this a step whose retraction is skipped
        // would inherit the PREVIOUS step's pair and print it against this
        // step's eval line. Both are read in the same iteration, so clearing
        // here is what makes "no measurement" print as `-`.
        tsct_bef = None;
        tsct_aft = None;
        if step >= step_at_entry + 50 && t_warm.is_none() {
            t_warm = Some(std::time::Instant::now());
        }
        let t_iter = std::time::Instant::now();
        let (bytes, hashes) = (std::mem::take(&mut pbytes), std::mem::take(&mut phashes));
        let t_io = std::time::Instant::now();
        let (x, h) = bytes_to_tensors(&bytes, &hashes, cfg.seq_len, cfg.batch, &device);
        let data_ms = t_io.elapsed().as_secs_f64() * 1000.0;
        // targets = next byte (shifted by one position)
        let shifted = std::mem::take(&mut pshift);
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [cfg.batch, cfg.seq_len]), &device);
        // The input pins, from the FIRST batch's shapes: a captured graph has
        // one shape, so the buffers are allocated once and every later batch is
        // copied into them.
        if graph_on && inputs.is_none() {
            inputs = Some(graph::InputPins::new(
                &x,
                &y,
                if dorm_cfg.use_engram && !cfg.engram_ram { Some(&h) } else { None },
            ));
        }
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

        // Host tables train at their own cadence (the report's rule: Adam on
        // the RAM tables every step), not piggybacked on the log cadence -
        // at log_every=100 they got 1/100 of their updates.
        let host_adam_step = cfg.host_adam_every > 0
            && step % cfg.host_adam_every as u64 == 0
            && rows_param.is_some();
        // One cadence for the timer and for the loss copy that feeds it, or the
        // timer reports a step whose loss was never read back. Both used to say
        // `step % 50` independently, and they now say the same thing.
        let timer_step = step == 0 || (cfg.log_every > 0 && step % cfg.log_every as u64 == 0);
        // A CUDA graph can only capture a window with NO host round-trip in it
        // (`cubecl-runtime/src/client.rs:1288-1296` refuses a read, a sync or a
        // handle write). Every step that reads anything back therefore runs
        // ungraphed and forces a re-capture: the loss scalar, the host-table
        // grads, the grad norm, `max_ortho`, and the eval/checkpoint forwards
        // that follow. Everything else in a step is sync-free (§1.3), so on the
        // steps in between the whole window is one dispatch.
        //
        // `step % 500` is in the list for the two 500-step instruments
        // (`max_ortho`, `memory_cleanup`), both of which sync.
        let ungraphed = !graph_on
            || step % cfg.log_every as u64 == 0
            || host_adam_step
            || (cfg.timers && timer_step)
            || step % 500 == 0
            || (cfg.eval_every > 0 && step % cfg.eval_every as u64 == 0 && step > 0)
            || (cfg.memlog && step % cfg.log_every as u64 == 0);
        let t_fwd = std::time::Instant::now();
        // Random-depth arm (ADR-0013 rank 2): T is a deterministic mix of the
        // step index, so an A/B run replays exactly and a resume continues the
        // same sequence. Both averages inside the loop divide by what ran, so
        // each step is an honest model at depth T. Refused with
        // --graph-capture (`graph::check`): a captured graph has ONE depth.
        if cfg.rand_depth {
            let t = sample_depth(step, model.loop_block.max_iter);
            model.set_loop_depth(Some(t));
        }
        // Offline JEPA: the frozen target for THIS chunk must exist in the
        // sidecar; a miss is a hard error (stale sidecar), never a silent
        // aux drop.
        let jepa_target = match jepa_tgts.as_mut() {
            Some(t) => Some(t.get(&bytes)?),
            None => None,
        };
        // Cloned because the window closure is `Fn` (a capture runs it for the
        // warmup and again inside the recording), and `--jepa-targets` is
        // refused with `--graph-capture` anyway — this only keeps the two code
        // paths identical.
        let jepa_target = jepa_target.clone();
        // The window: forward -> loss -> NaN-mask -> backward -> sanitize.
        // Everything the graph captures is inside this closure and nothing
        // else is, which is what makes the capture sound.
        let (x, y, h_for_fwd) = if let Some(pins) = inputs.as_mut() {
            pins.feed(&x, &y, if host_rows.is_some() { None } else { Some(&h) })?
        } else {
            (
                x.clone(),
                y.clone(),
                if host_rows.is_some() { None } else { Some(h.clone()) },
            )
        };
        // `RefCell` because the window closure is `Fn` (a capture runs it for
        // the warmup and again inside the recording) and these two are the
        // values it reports OUT. Both are idempotent across those runs — same
        // step, same inputs, same cadence — so the last write is the right one.
        let loss_log = std::cell::RefCell::new(None);
        let aux_log = std::cell::RefCell::new(None);
        seam.step(ungraphed, || {
            let (_logits, rec_ce, _kda, aux) = if let Some(tg) = jepa_target.clone() {
                model.forward_with_jepa_targets::<Backend>(
                    x.clone(),
                    // RAM-offload path drives the Engram from host_rows; uploading
                    // hashed_ids too would be a dead per-step H2D copy.
                    if host_rows.is_some() { None } else { h_for_fwd.clone() },
                    host_rows.clone(),
                    Some(y.clone()),
                    Some(tg),
                )
            } else {
                model.forward_with_hidden::<Backend>(
                    x.clone(),
                    if host_rows.is_some() { None } else { h_for_fwd.clone() },
                    host_rows.clone(),
                    Some(y.clone()),
                    teacher.as_ref(),
                )
            };
            let mut loss = model.loss::<Backend>(rec_ce);
            // Aux is read only on log steps; skip the clone (an extra autodiff
            // node) elsewhere.
            *aux_log.borrow_mut() = if step % cfg.log_every as u64 == 0 { aux.clone() } else { None };
            if let Some(a) = aux {
                loss = loss + a;
            }
            // The device syncs only on steps that read something back (loss
            // scalar at log cadence, host-table grads at the host-Adam cadence,
            // timers) so forward/backward/step of adjacent steps overlap on the
            // GPU. The scalar read rides along free inside the grads D2H.
            *loss_log.borrow_mut() = if step % cfg.log_every as u64 == 0
                || host_adam_step
                || (cfg.timers && timer_step)
            {
                Some(loss.clone())
            } else {
                None
            };
            // ── In-software NaN firewall ────────────────────────────────────
            // A non-finite loss used to be INVISIBLE between log steps: the
            // scalar is only read every log_every steps, so backward produced
            // NaN grads, the optimizer wrote them into the weights, and the
            // guard then replayed the poisoned region from a stale ckpt forever
            // (measured 2026-09-26 on official_v4: NaN at step 1550, then 1700,
            // ckpt stuck at 1000, zero progress across three restarts). The
            // recovery now lives inside the process and stays honest: see
            // `mask_nonfinite`. The log copy above is the RAW loss, so a spike
            // still prints as NaN/inf.
            // The firewall, with ZERO host-device synchronization (owner rule
            // 2026-09-27). Masking the loss scalar is not enough: a NaN from an
            // intermediate activation back-propagates anyway, because the chain
            // rule multiplies the masked zero seed by the Jacobian and NaN*0 =
            // NaN. So the loss is masked on device AND the gradients are
            // sanitized on device. The host learns a step was masked from the
            // gradient norm it already reads at log cadence (an exactly-zero
            // grad norm means every gradient was non-finite), so the accounting
            // costs no sync either.
            let loss = mask_nonfinite(loss);
            let mut raw_grads = loss.backward();
            sanitize_grads(&mut raw_grads, &model);
            raw_grads
        })?;
        let loss_log = loss_log.into_inner();
        let aux_log = aux_log.into_inner();
        let fwd_ms = t_fwd.elapsed().as_secs_f64() * 1000.0;
        let t_bwd = std::time::Instant::now();
        let lr = match stress.as_ref() {
            // Constant LR at a multiple of the optimum (report §3.3).
            Some(s) => s.lr(cfg.lr),
            None => wsd_factor(step, cfg.steps as u64, cfg.lr),
        };
        if let Some(loss_log) = &loss_log {
            // The RAW loss. It must be read BEFORE the mask runs and only on the
            // steps that read something back anyway: `clone()` shares the
            // device buffer and `mask_fill` picks the in-place strategy on CUDA
            // when only two handles alias it, so a copy read after the mask
            // returns 0.0 - which printed ce=0.000 on a NaN step and left `best`
            // stuck at 0 forever (review 2026-09-27). Reading it here, on log
            // cadence only, keeps the hot path sync-free.
            let ce_now: f32 = match loss_log.clone().try_into_scalar() {
                Ok(v) => v,
                Err(_) => {
                    return Err(format!("step {step}: device error (loss is not a scalar)"));
                }
            };
            ce = ce_now;
            if !ce_now.is_finite() {
                // Counted, not read on any cadence today — see the counter's
                // declaration: the consumer is ADR-0021 item 4's report.
                nan_reads += 1;
                let _ = nan_reads;
            }
            if ce.is_finite() && ce < best { best = ce; }
            // A step whose gradients were ALL non-finite (or whose loss was)
            // leaves an exactly-zero gradient norm. That is the honest, sync-free
            // signal that the firewall fired, and it is a LOWER BOUND: it can
            // only be seen on the steps whose gradient norm we read.
            let masked = masked_since_read;
            if masked > 0 {
                println!(
                    "  firewall: {masked} non-finite loss read(s) since the last log step (gradients zeroed on device; no optimizer-visible NaN)"
                );
            }
            if masked >= 8 {
                return Err(format!(
                    "step {step}: {masked} non-finite loss reads in one log window - the model is broken, not spiking"
                ));
            }
            masked_since_read = 0;
            nan_reads = 0;
            let _ = nan_reads;
            if host_adam_step {
                if let (Some(h), Some(p), Some(uniq)) =
                    (host.as_mut(), &rows_param, uniq_rows.as_ref())
                {
                    if let Some(g) = p.grad(seam.grads().expect("the window ran this step")) {
                        let mut g_vec: Vec<f32> = g.into_data().try_to_vec().unwrap_or_default();
                        assert!(
                            g_vec.len() == uniq.len() * h.dim,
                            "host-grad shape mismatch: {} vs {} uniq x {} dim — skipping a table update is not an option",
                            g_vec.len(),
                            uniq.len(),
                            h.dim
                        );
                        // One non-finite row grad poisons the host table
                        // (9-48M rows, no rollback): zero it and say so.
                        let bad = g_vec.iter().filter(|x| !x.is_finite()).count();
                        if bad > 0 {
                            println!("host-table grad: {bad} non-finite of {} zeroed", g_vec.len());
                            for x in g_vec.iter_mut() {
                                if !x.is_finite() { *x = 0.0; }
                            }
                        }
                        h.momentum_update(uniq, &g_vec, lr as f32);
                    }
                }
            }
            if step % cfg.log_every as u64 == 0 {
                let gn = grad_norm(&model, seam.grads().expect("the window ran this step"));
                if gn == 0.0 {
                    masked_since_read += 1;
                }
                if let Some(s) = stress.as_mut() {
                    s.observe(ce, gn);
                }
            }
        }
        let bwd_ms = t_bwd.elapsed().as_secs_f64() * 1000.0;
        atlas.mark(step, "bwd", bwd_ms as f32);
        let t_opt = std::time::Instant::now();
        // BORROWED, not consumed: a graph replay runs no host code, so there is
        // one `Gradients` for the life of the capture and `from_grads` (which
        // calls `grad_remove`) would leave every replay after the first with an
        // empty optimizer input — a silently frozen parameter set, the shape of
        // `8fa5d4c`. Same tensors, same update.
        let grads = graph::grads_params(seam.grads().expect("the window ran this step"), &model);
        model = optim.step(lr, model, grads);
        // The pin: burn's optimizer returned FRESH tensors, and the graph has
        // the previous addresses baked. One copy per parameter puts this step's
        // result back where the next replay will read it.
        if graph_on {
            let (m, r) = seam.refresh(model, false);
            model = m;
            r?;
        }
        let opt_ms = t_opt.elapsed().as_secs_f64() * 1000.0;
        atlas.mark(step, "opt", opt_ms as f32);
        let t_retr = std::time::Instant::now();
        // TSCT ortho maintenance (bf16_KERNEL_PLAN): retract the U/V masters
        // every step so the quantized forward stays faithful; monitor the
        // drift at cadence and fall back to fp32 factors when it exceeds the
        // plan's 1e-3 threshold. --retract-every / --retract-iters override.
        if step.is_multiple_of(cfg.retract_every.max(1) as u64) {
            // TSCT retraction diagnostic, BEFORE side (pre-100k checklist 2a).
            // `max_ortho` IS the per-entry masters metric, so this costs no
            // new metric - it is the same read the latch below makes, taken
            // one call earlier so the pair brackets the retraction. Only on
            // eval steps: it is a host read of 30+ factors, and the eval is
            // the declared sync cadence (§1.3). With `retract_every = 4` and
            // `eval_every = 500` this reads the drift the retraction has to
            // remove, which is the number the reviewer's question is about.
            // Assignment, not `let`: a shadowing `let` here compiles, leaves
            // the outer `tsct_bef` at `None`, and prints `-` forever.
            tsct_bef = on_eval_step.then(|| model.max_ortho());
            if cfg.retract_batched {
                model.retract_tsct_batched(cfg.retract_iters);
            } else {
                model.retract_tsct(cfg.retract_iters);
            }
            // AFTER side, at the same cadence.
            tsct_aft = on_eval_step.then(|| model.max_ortho());
        }
        let retr_ms = t_retr.elapsed().as_secs_f64() * 1000.0;
        atlas.mark(step, "retr", retr_ms as f32);
        let t_ema = std::time::Instant::now();
        if let Some(t) = teacher.take() {
            teacher = Some(dormouse_core::aux::ema_update(t, &model, dormouse_core::aux::TEACHER_MOMENTUM));
            // The teacher is read INSIDE the window, so it is pinned exactly like
            // the model: `ema_update` rebuilds every `Param`, so its address
            // moves every step.
            if graph_on {
                let (m, r) = seam.refresh(teacher.take().expect("just taken"), true);
                teacher = Some(m);
                r?;
            }
        }
        let _ = lr;
        let ema_ms = t_ema.elapsed().as_secs_f64() * 1000.0;
        atlas.mark(step, "ema", ema_ms as f32);
        atlas.step_done();
        // max_ortho reads every TSCT factor (30+ device syncs) - cadence,
        // not per-50-steps: each check drains the pipeline. The metric is
        // per-entry (F-norm/k) so the plan's 1e-3 threshold sits above the
        // retract's convergence floor and below real drift; before the
        // normalization (2026-09-04) the fallback fired at step 0 on every
        // fresh run, silently disabling the factor-quant forward.
        if step.is_multiple_of(500) && !ortho_fp32 {
            let ortho = model.max_ortho();
            if ortho > 1e-3 {
                println!("max_ortho {ortho:.2e} > 1e-3 - fallback fp32 factors");
                model.set_quant_all(burn_spectral::QuantFormat::Fp32);
                ortho_fp32 = true;
            }
        }

        // Cadence: `timer_step`, decided above together with the loss copy that
        // feeds it. That constant used to be a hardcoded `% 50` and the
        // timer was gated on it alone, which is why every short run in this
        // project's history reported step 0 and nothing else - and how a 23x
        // step-time error (opt=8303ms of 10809ms) survived in
        // benches/history.tsv and reached the rulebook as "the optimizer is 77%
        // of the step". A warm step is ~245 ms; the cold one is 5549 ms, and
        // only a run that prints past step 0 can tell the two apart. Gated on
        // `cfg.timers`, so an ordinary run pays nothing for the sync below.
        if cfg.timers && timer_step {
            // Force a device sync so the elapsed wall time equals the true GPU
            // step time (forward+backward+optim+retract). data_ms is the CPU
            // side (read + bytes_to_tensors). Their difference is GPU compute.
            if let Some(ll) = &loss_log {
                let _: f32 = ll.clone().try_into_scalar().unwrap_or(0.0);
            }
            let total_ms = t_iter.elapsed().as_secs_f64() * 1000.0;
            // The launch count, on the same line as the time it buys. This is
            // the number that decides whether a CUDA graph is worth anything:
            // a replay is ONE dispatch however many launches it contains, so
            // the benefit is `launches - 1` and the cost is one copy per
            // parameter per step (the pin a captured optimizer needs, because
            // burn's is out-of-place and the parameter's address moves).
            // Counted on the device thread, so it is a LOWER BOUND on what
            // the host enqueued, and cumulative since process start: the
            // per-step figure is the difference between two timer lines, which
            // is why this is a count and not a rate.
            let launches = cubecl_launches();
            println!(
                "timer step {step}: total={total_ms:.0}ms data={data_ms:.1}ms fwd={fwd_ms:.0}ms \
                 bwd={bwd_ms:.0}ms (incl. loss sync + host-adam D2H) opt={opt_ms:.0}ms \
                 retr={retr_ms:.1}ms ema={ema_ms:.1}ms gpu_step={:.0}ms launches={launches} \
                 graph=replayed:{}",
                total_ms - data_ms,
                seam.stats.replays,
            );
        }
        if step.is_multiple_of(cfg.log_every as u64) {
            let bpb = bpb(ce);
            // pool_stats syncs the device; only with --memlog
            let mem = if cfg.memlog { pool_stats(&device) } else { String::new() };
            let aux_note = match aux_log.clone().map(|a| a.try_into_scalar::<f32>().ok()) {
                Some(Some(v)) => format!(" aux={v:.4}"),
                _ => String::new(),
            };
            println!(
                "step {step:6} ce={ce:.3} bpb={bpb:.3} best={best:.3} lr={lr:.2e}{aux_note} \
                 retr_arm=batched:{}/factor:{} {mem}",
                probe::count(probe::RETRACT_BATCHED),
                probe::count(probe::RETRACT_FACTOR),
            );
            if let Some(s) = stress.as_ref() {
                if let Some(line) = s.report(step) {
                    println!("  {line}");
                }
            }
        }
        if cfg.eval_every > 0 {
            if let Some(ev) = eval_stream.as_mut() {
                if step.is_multiple_of(cfg.eval_every as u64) && step > 0 {
                    // Rewind FIRST: every eval must score the SAME bytes, or
                    // eval N of run A and eval N of run B read different
                    // windows and no A/B is comparable (measured 2026-09-27:
                    // v5 and its own resume scored 6.443 vs 6.551 for the
                    // SAME checkpoint, purely from stream position).
                    ev.rewind();
                    // A validation snapshot: no autodiff graph, so 20 eval
                    // forwards cannot pile up activations (they are never
                    // backwarded, and with grad tracking on they OOM'd the
                    // card at 15.9/16.3 GB on the first eval). The original
                    // model keeps training - burn's docs say exactly that.
                    let mut eval_model = model.valid();
                    // `depth_override` is a plain field, so the snapshot
                    // inherits the step's SAMPLED depth under --rand-depth and
                    // the eval line would report a depth-T model (found by
                    // review 2026-09-27). The eval is always the full-depth
                    // model; the depth curve is what varies.
                    eval_model.set_loop_depth(None);
                    // Average over eval_batches batches: one batch is 5 KB,
                    // whose sampling noise is larger than the effects we A/B.
                    let mut ce_sum = 0.0f32;
                    let mut n = 0u32;
                    // The memory arm, counted over the eval's own forwards.
                    // A memory-disabled forward is CORRECT and produces a
                    // normal-looking BPB, so the eval line is the only place
                    // a reader learns it happened; this is the arm counter
                    // ADR-0011 asks for, and `0/<n>` is the shape of the
                    // defect that made every pre-2026-09-28 eval line a
                    // measurement of a network with no n-gram memory in it.
                    let (eg_arms0, eg_rows0) = (
                        probe::count(probe::ENGRAM),
                        probe::count(probe::ENGRAM_KEYS),
                    );
                    for _ in 0..cfg.eval_batches.max(1) {
                        let (eb, eh) = match &host {
                            Some(h) => ev.next_batch_with_tables(h.slots),
                            None => ev.next_batch(),
                        };
                        let (ex, eh_t) =
                            bytes_to_tensors(&eb, &eh, cfg.seq_len, cfg.batch, &device);
                        let eshift: Vec<i64> = eb
                            .iter()
                            .skip(1)
                            .chain(std::iter::once(&eb[0]))
                            .map(|&b| b as i64)
                            .collect();
                        let ey: Tensor<2, Int> = Tensor::from_data(
                            TensorData::new(eshift, [cfg.batch, cfg.seq_len]),
                            &device,
                        );
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
                        // The SAME memory arm the training step ran, or the
                        // held-out number is a different model than the one
                        // being trained: the in-VRAM path needs the keys, the
                        // host path needs the rows (and uploading the keys too
                        // would be a dead H2D copy). This eval passed `None`
                        // unconditionally, so on the in-VRAM path - every run
                        // without --engram-ram, i.e. all of them until now -
                        // the Engram branch took its inert arm and the
                        // reported BPB was a network with no memory at all.
                        let (elogits, ..) = eval_model.forward_with_hidden::<Backend>(
                            ex,
                            if eval_rows.is_some() { None } else { Some(eh_t) },
                            eval_rows,
                            None,
                            None,
                        );
                        let v = eval_model.vocab_size;
                        let eflat = elogits.reshape([cfg.batch * cfg.seq_len, v]);
                        // Gather the target log-prob: no one-hot [b*t,v]
                        // fp32 tensor (extra H2D + traffic) per eval.
                        let etgt = ey.reshape([cfg.batch * cfg.seq_len, 1]);
                        let ece: f32 = burn::tensor::activation::log_softmax(eflat, 1)
                            .gather(1, etgt)
                            .neg()
                            .mean()
                            .try_into_scalar()
                            .unwrap_or(f32::NAN);
                        ce_sum += ece;
                        n += 1;
                    }
                    let ece = ce_sum / n as f32;
                    // A non-finite held-out number means the model OR the
                    // allocator is broken (measured 2026-09-27: a cubecl pool
                    // that had lost its buffers kept "training" and printed
                    // NaN evals for 500 more steps). Refuse loudly instead of
                    // collecting garbage.
                    if !ece.is_finite() {
                        return Err(format!(
                            "step {step}: held-out eval is {ece:.3} (non-finite) - the model or the allocator is broken, not the data"
                        ));
                    }
                    let ebpb = bpb(ece);
                    let bytes = (n as u64) * (cfg.batch * cfg.seq_len) as u64;
                    // The run's best checkpoint is the one with the best
                    // HELD-OUT score, not the best train CE. Measured
                    // 2026-09-28: a 20k-step run reached held-out 4.997 at
                    // step 6500 - below the unigram counter - and then
                    // overfit back to 5.450 by step 19500, and the step-19500
                    // weights were the only ones on disk. `best` above tracks
                    // train CE, which is exactly the quantity ADR-0011 calls
                    // fake. So write `<name>.best` whenever held-out improves.
                    let is_best_eval = ebpb < best_eval_bpb;
                    if is_best_eval {
                        best_eval_bpb = ebpb;
                        best_eval_step = step;
                    }
                    // Seam counters (ADR-0019): a fused CUDA kernel that falls
                    // back to tensor ops is CORRECT, so the only way a reader
                    // of this log learns the fast path was skipped is that it
                    // is printed here, on the line they already watch.
                    let (kda_f, kda_b, norm_asked, norm_skipped) = fused_seam_counts();
                    // Order is (asked, fused_fwd, fused_bwd, declined, ops_path,
                    // custom_node_bwd) - names, not blanks, because a blank in
                    // a destructuring pattern is how I read `bwd` as
                    // `declined` and briefly concluded the ops path was not
                    // counting itself.
                    let (kda_asked, _, kda_bwd, kda_decl, kda_ops, kda_node_bwd) =
                        dormouse_core::kda_seam_counts();
                    let (mu_mom, mu_fin) = optim::fused_kernels_skipped();
                    let (eg_arms, eg_rows) = (
                        probe::count(probe::ENGRAM) - eg_arms0,
                        probe::count(probe::ENGRAM_KEYS) - eg_rows0,
                    );
                    // `fb=<ran>/<asked>`: the future-byte arm, over the TRAINING
                    // forwards, not the eval's own. The eval forward passes no
                    // labels (`targets = None`, which is what keeps the held-out
                    // graph out of the tape), so it cannot run this arm - an
                    // eval-local count would read `0/0` forever and a reader
                    // would conclude the head is broken. `0/<n>` here is the
                    // real defect shape: the arm was opened every step and never
                    // produced a term, which is a horizon at or past the
                    // sequence length.
                    let (fb_ran, fb_asked) =
                        (probe::count(probe::FUTURE_BYTE), probe::count(probe::FUTURE_BYTE_ASKED));
                    // `tsct=`, the retraction + forward-factor diagnostics
                    // (pre-100k checklist 2). RARE cadence: this walk reads
                    // every TSCT factor back, so it is a declared-cadence
                    // host read at the eval boundary and nothing else (§1.3).
                    let tsct_field = model
                        .tsct_diag()
                        .field(tsct_bef, tsct_aft)
                        .unwrap_or_default();
                    println!(
                        "step {step:6} EVAL ce={ece:.3} bpb={ebpb:.3}{} over {bytes} B (fixed window) \
                         fused kda={kda_f}/{kda_b} asked={kda_asked} bwd={kda_bwd} \
                         declined={kda_decl} ops={kda_ops} node_bwd={kda_node_bwd} \
                         norm={}/{} muon_skipped={}/{} engram={eg_rows}/{eg_arms} fb={fb_ran}/{fb_asked} \
                         {tsct_field}",
                        if is_best_eval { " BEST" } else { "" },
                        norm_asked.saturating_sub(norm_skipped),
                        norm_asked,
                        mu_mom,
                        mu_fin
                    );
                    if is_best_eval {
                        // Save under a distinct name: the periodic save owns
                        // `<name>.bin` and a resume reuses `<name>`, so
                        // overwriting it here would make "the best model" and
                        // "the last step" the same file.
                        let best_name = format!("{}.best", cfg.ckpt_name);
                        if let Err(e) = save_ckpt(
                            &dir,
                            &best_name,
                            &model,
                            &optim,
                            teacher.as_ref(),
                            ortho_fp32,
                            step,
                            ce,
                        ) {
                            return Err(format!("step {step}: writing the best checkpoint failed: {e}"));
                        }
                        // The n-gram sidecar travels with the weights. Without
                        // it, `foo.best.bin` is step-S weights and NO tables,
                        // and the loader reads a missing sidecar as `None => h`
                        // - freshly seeded rows, silently. That is a DIFFERENT
                        // model from the one that was measured (ADR-0019).
                        if let Err(e) = save_ngram(&dir, &best_name, host.as_ref(), step) {
                            return Err(format!(
                                "step {step}: the best checkpoint's ngram sidecar failed: {e} \
                                 - the artifact would load as a DIFFERENT model"
                            ));
                        }
                        // Persist the score the artifact was chosen by, so a
                        // resume compares against it instead of against `inf`.
                        let bpb_path = dir.join(format!("{best_name}.bpb"));
                        if let Err(e) = std::fs::write(&bpb_path, format!("{ebpb} {step}\n")) {
                            return Err(format!("step {step}: writing {bpb_path:?} failed: {e}"));
                        }
                    }
                    // Depth curve, no training. Our readout is the MEAN of the
                    // per-iteration outputs, so "stop after k iterations" is
                    // the prefix mean over 1..k - which is exactly what
                    // --rand-depth trains. That is why a confidence exit
                    // composes with the shipped arm for free: no head, no
                    // retraining, just a different k per sequence.
                    if cfg.eval_depths {
                        let mut line = String::new();
                        for depth in 1..=eval_model.loop_block.max_iter {
                            let mut m = eval_model.clone();
                            m.set_loop_depth(Some(depth));
                            let mut d_sum = 0.0f32;
                            let mut d_n = 0u32;
                            for _ in 0..cfg.eval_batches.max(1) {
                                let (db, dh) = match &host {
                                    Some(h) => ev.next_batch_with_tables(h.slots),
                                    None => ev.next_batch(),
                                };
                                let (dx, dh_t) =
                                    bytes_to_tensors(&db, &dh, cfg.seq_len, cfg.batch, &device);
                                let dshift: Vec<i64> = db
                                    .iter()
                                    .skip(1)
                                    .chain(std::iter::once(&db[0]))
                                    .map(|&b| b as i64)
                                    .collect();
                                let dy: Tensor<2, Int> = Tensor::from_data(
                                    TensorData::new(dshift, [cfg.batch, cfg.seq_len]),
                                    &device,
                                );
                                let drows = match &host {
                                    Some(h) => {
                                        let (_, e, _) = offload::rows_for_batch::<Backend>(
                                            h, &dh, cfg.batch, cfg.seq_len, &device, false,
                                        );
                                        Some(e)
                                    }
                                    None => None,
                                };
                                // Same arm as the step above, for the same
                                // reason: a depth curve is a claim about the
                                // model, and this measured one without memory.
                                let (dl, ..) = m.forward_with_hidden::<Backend>(
                                    dx,
                                    if drows.is_some() { None } else { Some(dh_t) },
                                    drows,
                                    None,
                                    None,
                                );
                                let dv = m.vocab_size;
                                let dflat = dl.reshape([cfg.batch * cfg.seq_len, dv]);
                                let dtgt = dy.reshape([cfg.batch * cfg.seq_len, 1]);
                                d_sum += burn::tensor::activation::log_softmax(dflat, 1)
                                    .gather(1, dtgt)
                                    .neg()
                                    .mean()
                                    .try_into_scalar()
                                    .unwrap_or(f32::NAN);
                                d_n += 1;
                            }
                            line.push_str(&format!(" d{depth}={:.3}", bpb(d_sum / d_n as f32)));
                        }
                        println!("step {step:6} DEPTHS{line}");
                    }
                }
            }
        }
        if cfg.ckpt_every > 0 && step.is_multiple_of(cfg.ckpt_every as u64) {
            // LOUD, not `let _ =` + an unconditional "saved" line: a failed
            // save used to print "ckpt saved", keep training for hours and
            // lose the run (ADR-0019). A run that cannot checkpoint must say
            // so at the first failure.
            save_ckpt(&dir, &cfg.ckpt_name, &model, &optim, teacher.as_ref(), ortho_fp32, step, ce)
                .map_err(|e| format!("step {step}: checkpoint save failed: {e}"))?;
            save_ngram(&dir, &cfg.ckpt_name, host.as_ref(), step)?;
            println!("ckpt {}.bin saved step {step}", cfg.ckpt_name);
        }
        // Returning pool pages forces the next steps to re-acquire them from
        // the driver; at a near-full high-water this stalls the GPU. 500
        // steps keeps the OOM guard while amortizing the reacquisition.
        if step.is_multiple_of(500) {
            memory_cleanup(&device);
        }
        step += 1;
    }
    // Final ckpt records the LAST ce (best is tracked in the logs; the .txt
    // sidecar should describe this checkpoint's actual loss). The ngram
    // sidecar goes with it: it used to be written only on the `ckpt_every`
    // cadence, so a finished run left model@final over tables@last-cadence
    // and the next resume replayed up to ckpt_every steps of row updates on
    // top of a model that had already trained on them (ADR-0021).
    save_ckpt(&dir, &cfg.ckpt_name, &model, &optim, teacher.as_ref(), ortho_fp32, step, ce)
        .map_err(|e| format!("final checkpoint save failed: {e}"))?;
    save_ngram(&dir, &cfg.ckpt_name, host.as_ref(), step)?;
    // The warm-step mean, on every run that asked for timers, so a control and a
    // graphed arm are read with the SAME instrument. The graph's own verdict
    // rides with it: a lane that captured once and replayed nothing must read
    // as the null it is, not as a run that quietly did the same work as before.
    if cfg.timers {
        if let Some(t) = t_warm {
            let ran = step.saturating_sub(step_at_entry + 50).max(1);
            println!(
                "warm steps {}..{}: {:.1}s = {:.1} ms/step{}",
                step_at_entry + 50,
                step,
                t.elapsed().as_secs_f64(),
                t.elapsed().as_secs_f64() * 1e3 / ran as f64,
                if graph_on { format!(" | {}", seam.report()) } else { String::new() },
            );
        }
    }
    println!(
        "done steps={step} best ce={best:.3} | {}",
        best_artifact_summary(best_eval_bpb, best_eval_step, &cfg.ckpt_name)
    );
    atlas.finish(cfg.steps as u64);
    Ok(())
}

/// The last line of a run: where the best HELD-OUT artifact is, or the
/// admission that there is none. A function, not inline `println!`, because
/// the `inf` case used to name `checkpoints/<name>.best.bin` for a file that
/// was never written - and a sentence nobody can call is a sentence nobody
/// can test.
fn best_artifact_summary(best_eval_bpb: f32, best_eval_step: u64, ckpt_name: &str) -> String {
    if best_eval_bpb.is_finite() {
        format!(
            "BEST HELD-OUT bpb={best_eval_bpb:.3} at step {best_eval_step} -> checkpoints/{ckpt_name}.best.bin"
        )
    } else {
        "no held-out eval ran, so no best checkpoint".to_string()
    }
}

/// Load weights only (inference) from a named ckpt container.
pub fn load_model_weights(dir: &Path, name: &str, cfg: DormouseConfig) -> Option<DormouseModel> {
    let raw = std::fs::read(dir.join(format!("{name}.bin"))).ok()?;
    let h = parse_header(&raw)?;
    if h.body + h.model > raw.len() { return None; }
    let mb = Bytes::from_bytes_vec(raw[h.body..h.body + h.model].to_vec());
    let mrec = ModuleRecord::from_bytes(mb).ok()?;
    let device = device();
    let model = DormouseModel::new(&cfg, &device);
    Some(model.load_record(mrec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::GradientsParams;
    use dormouse_core::param::{LinearLike, LinearLikeInner};
    use dormouse_data::fnv;

    // ---- the best-held-out-checkpoint feature (4f68b28 + the resume fix) ----
    //
    // These drive the REAL `train_loop` on the CPU backend against a
    // purpose-built tiny corpus, with the checkpoint dir injected. That is the
    // point: the behaviour lives in the middle of a 1500-line function that
    // writes files and reads a sidecar, and a copy of the rule in a test
    // would pass while the rule rotted. What each test pins is in its own
    // comment - the short version: the artifact is chosen by HELD-OUT, a
    // resume compares against the recorded score instead of `inf`, an
    // unparsable sidecar is a loud refusal, and no eval means no artifact and
    // no advertised path.
    //
    // The eval bytes are a fixed learnable ramp, so the held-out score is a
    // property of the run and not of the byte stream's luck. What the tests
    // do NOT do is assume which way this run's score moves: each one drives
    // the decision from the recorded `.best.bpb` - the input the resume path
    // is specified to read - and the one test that has to know a direction
    // (held-out vs train CE) measures the bracket it needs instead of
    // guessing it.

    /// A self-contained run: its own train tree, its own held-out tree (the
    /// no-leak rule in `train_and_eval` refuses one inside the other, so they
    /// are siblings), and its own checkpoint dir. A ramp rather than random
    /// bytes, because a fresh-random-byte batch carries no signal and any
    /// loss claim on it is a coin flip.
    struct Run {
        root: PathBuf,
        cfg: RunCfg,
        data: PathBuf,
        eval: PathBuf,
        ckpts: PathBuf,
    }

    impl Drop for Run {
        /// Each run leaves two ~50 MB containers behind, and /tmp is a 32 GB
        /// tmpfs SHARED with every other job on this box (AGENTS 2.4). Seven
        /// of these is a disk-full event waiting for a training run, so
        /// cleanup is Drop and not a line each test has to remember -
        /// especially the ones that fail before reaching it.
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    impl Run {
        /// `steps` is the only knob the tests turn: it is one of the five
        /// PROGRESS_KEYS, so two runs of the same `Run` differ exactly the
        /// way a resume is allowed to (ADR-0005) and never trip the drift
        /// check.
        fn new(tag: &str, steps: usize) -> Self {
            let root = std::env::temp_dir().join(format!("dm-best-{tag}"));
            let _ = std::fs::remove_dir_all(&root);
            let (data, eval, ckpts) = (root.join("train"), root.join("eval"), root.join("ckpts"));
            for d in [&data, &eval, &ckpts] {
                std::fs::create_dir_all(d).unwrap();
            }
            // ~1.5 kB on a 13-symbol period: learnable, and comfortably over
            // the stream's seq_len*batch*4 floor.
            let corpus: Vec<u8> = (0..1536u32).map(|i| (i.wrapping_mul(7) % 13) as u8).collect();
            std::fs::write(data.join("corpus.bin"), &corpus).unwrap();
            std::fs::write(eval.join("eval_tail.bin"), &corpus).unwrap();
            Self {
                root,
                cfg: RunCfg {
                    source: "test".into(),
                    model: DormouseConfig { max_iter: 1, ..test_cfg() },
                    train: TrainCfg {
                        steps,
                        seq_len: 32,
                        batch: 2,
                        lr: 1e-3,
                        ckpt_name: "b".into(),
                        eval_every: 1,
                        eval_batches: 2,
                        log_every: 1_000, // quiet: the log is not the assertion
                        ckpt_every: 1_000,
                        warmup: false,
                        ..Default::default()
                    },
                },
                data,
                eval,
                ckpts,
            }
        }

        /// One process. `eval` is the held-out tree, or None for a run with no
        /// `--eval` at all.
        fn go(&self, eval: Option<&Path>) -> Result<(), String> {
            train_loop(self.cfg.clone(), self.data.clone(), Some(self.ckpts.clone()), eval.map(|p| p.to_path_buf()))
        }

        fn path(&self, name: &str) -> PathBuf {
            self.ckpts.join(name)
        }
    }

    /// The score+step a `.best.bpb` records, as the resume path reads it.
    fn read_bpb(run: &Run) -> Option<(f32, u64)> {
        let text = std::fs::read_to_string(run.path("b.best.bpb")).ok()?;
        let mut it = text.split_whitespace();
        Some((it.next()?.parse::<f32>().ok()?, it.next()?.parse::<u64>().ok()?))
    }

    /// The step+ce `save_ckpt` writes beside a container: "step 2 ce 5.705".
    /// `ce` is the TRAIN ce of that step, which is what makes it a second,
    /// independent witness of which step won.
    fn read_ckpt_sidecar(run: &Run, name: &str) -> Option<(u64, f32)> {
        let text = std::fs::read_to_string(run.path(&format!("{name}.txt"))).ok()?;
        let f: Vec<&str> = text.split_whitespace().collect();
        Some((f.get(1)?.parse().ok()?, f.get(3)?.parse().ok()?))
    }

    /// A marker standing in for an artifact that already exists. The LIVE
    /// file is the assertion: a save replaces it (and rotates the old bytes
    /// into `.prev.bin`, which is the rotation working, not a failed rewrite).
    fn seed_marker(run: &Run, bpb: &str, marker: &[u8]) {
        std::fs::write(run.path("b.best.bpb"), bpb).unwrap();
        std::fs::write(run.path("b.best.bin"), marker).unwrap();
    }

    fn marker_survived(run: &Run, marker: &[u8]) -> bool {
        std::fs::read(run.path("b.best.bin")).is_ok_and(|b| b == marker)
    }

    /// The whole point of the feature: the artifact is chosen by the HELD-OUT
    /// score, and not by the train CE that ADR-0011 calls fake. A run whose
    /// saved "best" tracks train CE keeps the LAST weights instead of the
    /// best ones (measured 2026-09-28: held-out 4.997 at step 6500, overfit
    /// back to 5.450 by step 19500, and only step 19500 was on disk).
    ///
    /// The two quantities are not interchangeable and the gap is STRUCTURAL,
    /// which is what makes this testable at all: `bpb()` is ce/ln(2), so a
    /// barely-trained byte model reads ~5.7 as a CE and ~8.0 as a bpb. The
    /// same run is therefore run twice, with a recorded best of 6.5 and 8.5:
    ///   - 6.5: below its held-out, above its train CE -> held-out rule says
    ///     DO NOT save; a train-CE rule would save.
    ///   - 8.5: above its held-out -> held-out rule says save.
    ///
    /// Together the two runs BRACKET this run's held-out bpb into [6.5, 8.5)
    /// and the test asserts the train CE it really produced is below that
    /// whole bracket, so run 1's "did not save" is only reachable by having
    /// compared held-out. No assumption about how fast the model learns: the
    /// bracket is measured, not assumed. If the model ever learns enough to
    /// leave the bracket, run 1 goes red and says so - the fix is a wider
    /// bracket, not a weaker test.
    #[test]
    fn best_artifact_is_chosen_by_held_out_not_train_ce() {
        let marker = b"PRE-EXISTING BEST";
        // Run 1: recorded best 6.5 sits inside the gap, so held-out must lose.
        let lo = Run::new("heldout-lo", 3);
        seed_marker(&lo, "6.5 999\n", marker);
        lo.go(Some(&lo.eval)).expect("run must succeed");
        assert!(
            marker_survived(&lo, marker),
            "a held-out bpb above 6.5 must not replace the best artifact - if this run's held-out \
             has learned its way below 6.5, the bracket needs re-deriving"
        );
        // Run 2: recorded best 8.5 is above its held-out, so held-out must win.
        let hi = Run::new("heldout-hi", 3);
        seed_marker(&hi, "8.5 999\n", marker);
        hi.go(Some(&hi.eval)).expect("run must succeed");
        assert!(
            !marker_survived(&hi, marker),
            "a held-out bpb below 8.5 must replace the best artifact - the bracket is empty, so \
             these two runs cannot be told apart"
        );
        // The premise that makes run 1 a discriminator: the train CE of the
        // step that won run 2 is BELOW the whole bracket, so a comparison
        // against it would have saved in run 1 as well.
        let (_, train_ce) = read_ckpt_sidecar(&hi, "b.best").expect("the winner records its ce");
        assert!(
            train_ce < 6.5,
            "this run's train ce is {train_ce}, not below the bracket - run 1 no longer \
             discriminates held-out from train CE"
        );
    }

    /// The artifact, the score that chose it and the step it came from must be
    /// one evaluation, not three drifting facts. A `.best.bpb` that stays
    /// `inf`, or a score written by a different decision than the artifact it
    /// sits beside, is a resume comparing against the wrong number - which is
    /// silent, and is the whole failure mode the sidecar exists to stop.
    #[test]
    fn best_artifact_is_chosen_by_held_out_score() {
        let run = Run::new("heldout", 3);
        run.go(Some(&run.eval)).expect("run must succeed");
        let (bpb, step) = read_bpb(&run).expect("a run that evaluated must record the score it chose by");
        assert!(bpb.is_finite() && bpb > 0.0, "recorded best must be a real score, got {bpb}");
        assert!(step > 0 && step <= 3, "best must name a real step of this run, got {step}");
        assert!(run.path("b.best.bin").is_file(), "the artifact the score names must exist");
        let (art_step, _) = read_ckpt_sidecar(&run, "b.best").expect("the artifact records its step");
        assert_eq!(art_step, step, "the artifact and the score that chose it must be one evaluation");
        assert!(load_model_weights(&run.ckpts, "b.best", run.cfg.model.clone()).is_some(),
                "the best artifact must be a real, loadable container");
    }

    /// A resume must not overwrite a better artifact. `best_eval_bpb` is a
    /// local, so without the `.best.bpb` sidecar a second process starts at
    /// `inf` and its FIRST eval always wins - which silently re-creates the
    /// bug the feature exists to prevent.
    ///
    /// The sidecar is pre-seeded with a score no eval can beat (0.1, against
    /// a uniform ceiling of 8.0), so the correct answer is unambiguous
    /// whatever this run's held-out number happens to be. The `.best.bin`
    /// bytes are captured first and compared after, because "did not save" and
    /// "saved the same thing" are different answers and only the bytes tell
    /// them apart.
    #[test]
    fn resume_does_not_overwrite_the_best_with_a_worse_eval() {
        let run = Run::new("noclobber", 2);
        let marker = b"PRE-EXISTING BEST";
        seed_marker(&run, "0.1 999\n", marker);
        run.go(Some(&run.eval)).expect("run must succeed");
        assert!(
            marker_survived(&run, marker),
            "an eval worse than the recorded 0.1 must not touch the best artifact"
        );
        assert!(
            !run.path("b.best.prev.bin").exists(),
            "no save means no rotation either: a .prev beside the best artifact is a save that happened"
        );
        let (bpb, step) = read_bpb(&run).expect("sidecar must survive");
        assert_eq!((bpb, step), (0.1, 999), "a losing eval must not rewrite the recorded score");
        // The run really happened, so "did not save" is a decision and not a
        // crash: its own last step is on disk.
        assert!(run.path("b.bin").is_file(), "the run must still have saved its last step");
    }

    /// The `tsct=` diagnostics must survive the REAL 2-batch CPU eval path, not
    /// only the model in isolation.
    ///
    /// `TsctDiag::field` is a pure function and its exact format is pinned in
    /// `tsct_field_format_is_golden_and_alpha_zero_is_the_cross_check`; what
    /// this adds is the thing a pure-function test cannot see — that the walk
    /// runs on a live model inside the 1400-line `train_loop`, on the eval
    /// step, over every TSCT factor including the ones the block fold and the
    /// lm_head fold each own. A fold that forgets `lm_head`, or reads a factor
    /// the eval's `valid()` snapshot has already released, passes the unit
    /// test and panics at step 1 of every run. The `Run` harness is already
    /// `batch: 2, eval_batches: 2, eval_every: 1`, so this is the two-batch
    /// CPU run and nothing about it is a stand-in.
    #[test]
    fn tsct_diag_runs_on_a_real_two_batch_cpu_eval() {
        let run = Run::new("tsctdiag", 2);
        run.go(Some(&run.eval)).expect("the tsct walk must not break the run");
        // The run scored, so the eval block executed; and a best artifact
        // exists, so the held-out path after the print also ran.
        let (bpb, step) = read_bpb(&run).expect("the run must have produced a held-out score");
        assert!(bpb.is_finite() && step > 0, "bpb {bpb} at step {step}");
        // `retract_every` defaults to 1, so both sides of the pair are `Some`
        // on an eval step — the field is fully populated on the default
        // config, which is the configuration tonight's 100k run uses.
        assert!(cfg_retract_every_default_is_one());
        // SCOPE, stated so nobody reads more into this test than it proves:
        // it does NOT inspect the printed line (the harness swallows a
        // test's `println!`, and building a release binary to read a log is
        // not a gate). The exact FORMAT is pinned where the format is
        // written — `TsctDiag::field`, one function, one place — by
        // `tsct_field_format_is_golden_and_alpha_zero_is_the_cross_check`.
        // What only a real run can catch is the walk itself.
    }

    /// `retract_every` is 1 by default. Asserted rather than read from the
    /// field: if someone moves the default to 4 (the external review asks for
    /// every-4th), the eval line starts printing `-` for both retraction
    /// sides unless 4 divides `eval_every` — and THAT is the moment this
    /// pairing breaks, so the condition has to be visible in a test rather
    /// than discovered in a 100k log.
    fn cfg_retract_every_default_is_one() -> bool {
        assert_eq!(
            TrainCfg::default().retract_every, 1,
            "the tsct bef/aft pair is fully populated only because retract_every is 1; \
             if this moved, check that eval_every % retract_every == 0 or the eval line \
             will print `-` for both sides on every eval"
        );
        true
    }

    /// The other direction, or the sidecar is a one-way trap: a better eval
    /// MUST replace the artifact, and the recorded score must move with it.
    /// A pre-seeded 12.0 (above the uniform ceiling of 8.0) is beaten by every
    /// real eval, so this needs no assumption about which way this run trains.
    #[test]
    fn a_better_eval_replaces_the_best_and_moves_the_score() {
        let run = Run::new("clobberup", 2);
        let marker = b"STALE BEST";
        seed_marker(&run, "12.0 999\n", marker);
        run.go(Some(&run.eval)).expect("run must succeed");
        let (bpb, step) = read_bpb(&run).expect("sidecar must exist");
        assert!(bpb < 12.0, "a better eval must lower the recorded score, got {bpb}");
        assert!(step > 0 && step <= 2, "the new score must name a real step, got {step}");
        // The LIVE artifact is what a reader of the log opens, and it is no
        // longer the marker. (The stale bytes reappearing in `.prev` would be
        // the rotation doing its job, not a failure to rewrite.)
        assert!(!marker_survived(&run, marker), "a better eval must rewrite the artifact");
        // And the artifact must be loadable, not just present: this is what a
        // reader of the log is told to open.
        assert!(load_model_weights(&run.ckpts, "b.best", run.cfg.model.clone()).is_some(),
                "the best artifact must be a real, loadable container");
    }

    /// An unparsable `.best.bpb` is a LOUD refusal, never a silent `inf`.
    /// The score file is written after the artifact, so a crash in between
    /// leaves a stale or truncated pair - and quietly falling back to `inf`
    /// is exactly the overwrite the sidecar was added to stop. Every malformed
    /// shape is covered, because the fix has to be TOTAL: a reader that
    /// accepts the well-formed two-token case and drops the empty one has
    /// reintroduced the bug for the crash it was written for.
    #[test]
    fn an_unparsable_best_bpb_is_a_loud_refusal() {
        for (tag, body) in [
            ("empty", ""),
            ("blank", "   \n"),
            ("one-token", "0.1\n"),
            ("garbage", "best bpb was very good indeed\n"),
            ("bad-float", "not-a-number 5\n"),
            ("bad-step", "0.1 step-five\n"),
            ("nonfinite", "inf 5\n"),
        ] {
            let run = Run::new(&format!("refuse-{tag}"), 1);
            seed_marker(&run, body, b"PRE-EXISTING BEST");
            let err = run.go(Some(&run.eval)).expect_err("an unparsable sidecar must refuse the run");
            assert!(
                err.contains("b.best.bpb") && err.contains("refusing to resume"),
                "{tag}: the refusal must name the file and the reason, got: {err}"
            );
            assert!(
                marker_survived(&run, b"PRE-EXISTING BEST"),
                "{tag}: a refused run must not touch the artifact"
            );
        }
    }

    /// No eval, no best artifact - and the log must not advertise one. The
    /// unguarded version printed `bpb=inf at step 0 -> checkpoints/b.best.bin`
    /// for a run that never wrote one. Two independent claims, both asserted:
    /// the FILESYSTEM has no artifact (the real defect), and the SENTENCE
    /// names no path (the reader was misled).
    #[test]
    fn no_eval_means_no_best_artifact_and_no_advertised_path() {
        let run = Run::new("noeval", 2);
        run.go(None).expect("a run with no --eval must succeed");
        assert!(!run.path("b.best.bin").exists(), "no eval must write no best artifact");
        assert!(!run.path("b.best.bpb").exists(), "no eval must record no best score");
        // The last step's own artifacts ARE there, so the run really happened.
        assert!(run.path("b.bin").is_file(), "the run must still have saved its last step");
        let line = best_artifact_summary(f32::INFINITY, 0, "b");
        assert!(!line.contains("b.best.bin"), "must not name an artifact that does not exist: {line}");
        assert!(!line.contains("inf"), "must not report an infinite score: {line}");
    }

    /// The n-gram sidecar travels WITH the best artifact. Without it,
    /// `b.best.bin` is step-S weights and no tables, and the loader reads a
    /// missing sidecar as freshly seeded rows - a DIFFERENT model from the
    /// one that was measured, silently. A weights-only save therefore ships
    /// a checkpoint that does not reproduce its own score.
    #[test]
    fn the_best_artifact_carries_its_ngram_sidecar() {
        let run = Run::new("sidecar", 2);
        let mut cfg = run.cfg.clone();
        cfg.train.engram_ram = true;
        cfg.train.engram_slots = 1024; // 3 MB of host tables, not 12 GB
        train_loop(cfg, run.data.clone(), Some(run.ckpts.clone()), Some(run.eval.clone()))
            .expect("run must succeed");
        assert!(run.path("b.best.bin").is_file(), "this run must produce a best artifact");
        assert!(
            run.path("b.best.ngram").is_file(),
            "the best artifact must travel with its n-gram sidecar: the loader reads a missing \
             one as freshly seeded rows, i.e. a different model"
        );
    }

    /// The `tsct=` eval-line field: exact format, on a real 2-batch CPU
    /// model, with the `alpha = 0` cross-check the format's meaning rests on.
    ///
    /// `alpha = 0` is not a knob this repo sets anywhere (grep: no caller of
    /// `set_alpha`), so it is used here as the ONE place where the two code
    /// paths provably agree: `ste_ternary_annealed(w, 0) == w` identically,
    /// so the forward-factor metric MUST equal the masters' metric to the
    /// last bit. A field whose two numbers could disagree for a structural
    /// reason (wrong alpha, wrong tensor, a quant format folded into the
    /// wrong place) is a field nobody can read; this pins that they cannot.
    #[test]
    fn tsct_field_format_is_golden_and_alpha_zero_is_the_cross_check() {
        let dev = device();
        // No TSCT factors at all -> no field. `--set use_tsct=false` prints
        // nothing rather than a row of zeros, and that is the assertion.
        let dense = LinearLike::dense(32, 32, &dev);
        let mut agg = dormouse_core::TsctDiag::default();
        dense.fold_tsct_diag(&mut agg);
        assert_eq!(agg.field(None, None), None, "a dense linear has no tsct field");

        let mut model = DormouseModel::new(&test_cfg(), &dev);
        set_tsct_alpha(&mut model, 0.0);
        let masters = model.max_ortho();
        let d = model.tsct_diag();
        assert!(
            (d.fwd - masters).abs() <= 1e-12 * masters.max(1e-12),
            "at alpha = 0 the forward factor IS the master: fwd {:.6e} vs max_ortho {:.6e}",
            d.fwd,
            masters
        );
        assert_eq!(d.alpha, 0.0);
        assert!(d.s_min.is_finite(), "a TSCT model must report a finite s_min");
        assert_eq!(d.off, 0, "a freshly initialised s is all ones: no dead ranks");

        // The exact string. Format is
        // `tsct=<bef>/<aft>/<fwd>/<smin>/<smax>/<off>@a=<alpha>`.
        let f = d.field(Some(1.0), Some(0.5)).expect("a TSCT model has a field");
        assert_eq!(
            f,
            format!(
                "tsct=1.00e0/5.00e-1/{:.2e}/{:.3e}/{:.3e}/0@a=0.00",
                d.fwd, d.s_min, d.s_max
            )
        );
        // A missing side is `-`, never a number: two identical numbers printed
        // as a pair read as a measurement and mean "we did not measure it".
        assert!(d.field(None, Some(0.5)).unwrap().starts_with("tsct=-/5.00e-1/"));
        assert!(d.field(Some(1.0), None).unwrap().starts_with("tsct=1.00e0/-/"));
        // And at the shipped alpha = 1 the two metrics DISAGREE by orders of
        // magnitude - the reason the field carries both.
        let mut model1 = DormouseModel::new(&test_cfg(), &dev);
        set_tsct_alpha(&mut model1, 1.0);
        let fwd1 = model1.tsct_diag().fwd;
        assert!(
            fwd1 > 100.0 * masters,
            "alpha = 1 must move the forward factor off the manifold: {fwd1:.3e} vs masters {masters:.3e}"
        );
    }

    /// Walk every TSCT layer in the model and set its annealing `alpha`.
    /// Test-only: the trainer has no alpha schedule today, so this is how a
    /// test reaches the annealing regime the diagnostic exists to measure.
    fn set_tsct_alpha(model: &mut DormouseModel, a: f32) {
        let set = |l: &mut LinearLike| {
            if let LinearLikeInner::Tsct(t) = &mut l.inner {
                t.set_alpha(a);
            }
        };
        for f in &mut model.loop_block.expert_ffns {
            set(&mut f.gate_up);
            set(&mut f.down);
        }
        set(&mut model.loop_block.out_proj);
        set(&mut model.lm_head);
    }

    /// Retract must pull drifted TSCT masters back to orthonormal: corrupt
    /// U by scaling, verify ortho_error collapses below the plan's 1e-3
    /// threshold (the quantized forward depends on it).
    #[test]
    fn tsct_retract_restores_ortho() {
        let dev = device();
        let mut ll = LinearLike::new(64, 64, 16, &dev);
        if let LinearLikeInner::Tsct(l) = &mut ll.inner {
            let u = l.u.val().mul_scalar(3.0).detach();
            l.u = burn::module::Param::from_tensor(u);
        } else {
            panic!("LinearLike must be TSCT on the CPU backend");
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
        let optim = crate::optim::build_optim_mode(&model, &optim_cfg, "mix");
        let dir = std::env::temp_dir().join("dm-ckpt-roundtrip-test");
        let _ = std::fs::remove_dir_all(&dir);
        save_ckpt(&dir, "rt", &model, &optim, None, false, 42, 3.25).expect("save");
        let txt = std::fs::read_to_string(dir.join("rt.txt")).expect("sidecar");
        assert!(txt.contains("step 42 ce 3.250"), "sidecar: {txt}");
        let mut model2 = DormouseModel::new(&cfg, &device());
        let mut optim2 = crate::optim::build_optim_mode(&model, &optim_cfg, "mix");
        let loaded = load_ckpt(&dir, "rt", &cfg, &mut model2, &mut optim2).expect("load");
        assert_eq!(loaded.step, 42);
        assert!(!loaded.legacy, "a container we just wrote is not legacy");
        // identical forward on the same batch
        let bytes: Vec<u8> = (0..128u32).map(|i| (i.wrapping_mul(37) % 256) as u8).collect();
        let mut hashes = Vec::with_capacity(128 * 3);
        for p in 0..128usize {
            let e = p + 1;
            hashes.push((fnv(&bytes[e.saturating_sub(3)..e]) % 4096) as i64);
            hashes.push((fnv(&bytes[e.saturating_sub(5)..e]) % 4096) as i64);
            hashes.push((fnv(&bytes[e.saturating_sub(8)..e]) % 4096) as i64);
        }
        let (x, h) = bytes_to_tensors(&bytes, &hashes, 64, 2, &device());
        let (l1, ..) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, None, None);
        let (l2, ..) = model2.forward_with_hidden::<Backend>(x, Some(h), None, None, None);
        let d = (l1 - l2).abs().max().into_scalar::<f32>();
        assert!(d < 1e-5, "reloaded model diverges: max |dlogit| = {d:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ADR-0021: the state the config CANNOT express must survive the
    /// container - the EMA teacher and the latched fp32-factor fallback. Both
    /// used to be rebuilt from the config on every resume, which is how a
    /// resumed run became a different experiment wearing the same step count.
    /// Asserted by construction: the teacher here is NOT a copy of the
    /// student, so a restore that quietly re-derived it cannot pass.
    #[test]
    fn ckpt_carries_teacher_and_fallback_latch() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let optim_cfg = TrainCfg { steps: 5, seq_len: 64, batch: 2, ..Default::default() };
        let optim = crate::optim::build_optim_mode(&model, &optim_cfg, "mix");
        // A teacher at distance from the student: every param + 1. A restore
        // that re-derived the teacher from the student cannot pass.
        struct Bump;
        impl burn::module::ModuleMapper for Bump {
            fn map_float<const D: usize>(
                &mut self,
                p: burn::module::Param<burn::tensor::Tensor<D>>,
            ) -> burn::module::Param<burn::tensor::Tensor<D>> {
                let (id, tensor, mapper) = p.consume();
                burn::module::Param::from_mapped_value(id, tensor.add_scalar(1.0), mapper)
            }
        }
        let teacher = model.clone().map(&mut Bump).no_grad();
        let dir = std::env::temp_dir().join("dm-ckpt-teacher-test");
        let _ = std::fs::remove_dir_all(&dir);
        save_ckpt(&dir, "st", &model, &optim, Some(&teacher), true, 11, 1.5).expect("save");

        let mut m2 = DormouseModel::new(&cfg, &device());
        let mut o2 = crate::optim::build_optim_mode(&model, &optim_cfg, "mix");
        let loaded = load_ckpt(&dir, "st", &cfg, &mut m2, &mut o2).expect("load");
        assert!(loaded.ortho_fp32, "the one-way fp32 fallback must be persisted");
        let t2 = loaded.teacher.expect("the EMA teacher must be in the container");
        // Compared on a concrete param, not on "some forward agrees": the
        // claim is that the teacher's VALUES came back.
        let (a, b, c) = (
            t2.embedding.weight.val().clone().into_data(),
            teacher.embedding.weight.val().clone().into_data(),
            model.embedding.weight.val().clone().into_data(),
        );
        let (a, b, c): (Vec<f32>, Vec<f32>, Vec<f32>) = (
            a.try_to_vec().unwrap(),
            b.try_to_vec().unwrap(),
            c.try_to_vec().unwrap(),
        );
        assert_eq!(a, b, "the restored teacher must equal the saved one");
        assert!(a != c, "the saved teacher must not be a copy of the student");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pre-ADR-0021 container (no magic, no teacher) must still LOAD - and
    /// must say out loud that it carries no teacher, because a run that
    /// silently rebuilt one is the bug this whole change is about. The legacy
    /// path is exercised by rewriting the v2 header as the old 24-byte one.
    #[test]
    fn legacy_container_still_loads_and_is_flagged() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let optim_cfg = TrainCfg { steps: 5, seq_len: 64, batch: 2, ..Default::default() };
        let optim = crate::optim::build_optim_mode(&model, &optim_cfg, "mix");
        let dir = std::env::temp_dir().join("dm-ckpt-legacy-test");
        let _ = std::fs::remove_dir_all(&dir);
        save_ckpt(&dir, "lg", &model, &optim, None, false, 9, 2.0).expect("save");

        // Rewrite [magic][step][mlen][olen][tlen][flags] as [step][mlen][olen].
        let raw = std::fs::read(dir.join("lg.bin")).expect("read");
        let u = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
        let mut legacy = Vec::new();
        legacy.extend_from_slice(&u(&raw[8..16]).to_le_bytes());
        legacy.extend_from_slice(&u(&raw[16..24]).to_le_bytes());
        legacy.extend_from_slice(&u(&raw[24..32]).to_le_bytes());
        legacy.extend_from_slice(&raw[CKPT_HEADER..]);
        std::fs::write(dir.join("lg.bin"), &legacy).expect("rewrite");

        let mut m2 = DormouseModel::new(&cfg, &device());
        let mut o2 = crate::optim::build_optim_mode(&model, &optim_cfg, "mix");
        let loaded = load_ckpt(&dir, "lg", &cfg, &mut m2, &mut o2).expect("a legacy container must still load");
        assert_eq!(loaded.step, 9);
        assert!(loaded.legacy, "the reader must know it is reading a pre-ADR-0021 file");
        assert!(loaded.teacher.is_none(), "a legacy file cannot carry a teacher");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The report's §3.1 routing, asserted on the LIVE model: every boundary
    /// the string table used to pin is checked here by name, but the group
    /// comes from the declaration (and the installed groups), never from a
    /// path match. A path is used only to say WHICH parameter the assertion is
    /// about - the parameter list itself is pinned by
    /// `tests/routing_policy.rs` over every arm.
    #[test]
    fn muon_routing_matches_report() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let tcfg = TrainCfg { qk_heads: Some(cfg.n_heads), ..TrainCfg::default() };
        let g = optimizer_groups(&model, &tcfg).expect("the declared install must be valid");
        let r = dormouse_core::routing::routing(&model, false);

        let group = |suffix: &str| {
            let hit = crate::optim::param_paths(&model)
                .into_iter()
                .find(|(p, _, _)| p.ends_with(suffix));
            let (path, id, _) = hit.unwrap_or_else(|| panic!("no param ending in {suffix:?}"));
            (path.clone(), g.muon.matches(&id, Some(&path)), r.group_of_id(&id))
        };

        // The Muon+ group: small low-rank factors only (fp32 NS on a [d,d]
        // projection is ~40 s/step on this box; see optim.rs).
        for s in [
            "expert_ffns.0.gate_up.inner.Tsct.u",
            "expert_ffns.0.down.inner.Tsct.v",
            "engram.key_projs.0.weight",
            "out_proj.inner.Tsct.u",
            "out_proj.inner.Tsct.v",
        ] {
            let (path, on_muon, _) = group(s);
            assert!(on_muon, "{path}: a low-rank factor must be on Muon+");
        }
        // 1D TSCT scale stays on AdamW.
        let (path, on_muon, declared) = group("expert_ffns.0.gate_up.inner.Tsct.s");
        assert!(!on_muon, "{path}: the 1D TSCT scale must not be orthogonalized");
        assert_eq!(declared, Some(dormouse_core::routing::Group::Rest), "{path}");
        // [d,d] projections and dense linears stay on the fallback while the
        // fp32 NS cost is prohibitive. The dense expert weight is the leaf the
        // marker table got wrong: it MATCHED "expert_ffns." and matched
        // neither exclude, so the trainer sent a [d,d] map to Muon+ while the
        // declaration said Rest. Named explicitly, not left to the loop.
        let dense = DormouseModel::new(&DormouseConfig { use_tsct: false, ..cfg.clone() }, &device());
        let dg = optimizer_groups(&dense, &tcfg).expect("the dense install must be valid");
        let dense_weight = crate::optim::param_paths(&dense)
            .into_iter()
            .find(|(p, _, _)| p.ends_with("expert_ffns.0.gate_up.inner.Dense.weight"))
            .expect("a dense expert weight exists");
        let (path, id, rank) = dense_weight;
        assert_eq!(rank, 2, "{path}: a dense expert weight is a [d,d] map");
        assert!(!dg.muon.matches(&id, Some(&path)), "{path}: a [d,d] map reached Muon+");
        // Per-head scalar producers (decay/beta gates), routers, embeddings,
        // the output head: AdamW.
        for s in [
            "gdn2.decay.w_up.weight",
            "gdn2.beta_proj.weight",
            "controller.weight",
            "embedding.weight",
            "lm_head.inner.Tsct.v",
        ] {
            let (path, on_muon, declared) = group(s);
            assert!(!on_muon, "{path}: must not be on Muon+");
            assert_eq!(declared, Some(dormouse_core::routing::Group::Rest), "{path}");
        }
        // Attention Q/K go to the head-wise group instead of the Muon+ one.
        for s in ["gdn2.q_proj.weight", "gdn2.k_proj.weight"] {
            let (path, id, _) = crate::optim::param_paths(&model)
                .into_iter()
                .find(|(p, _, _)| p.ends_with(s))
                .unwrap_or_else(|| panic!("no param ending in {s:?}"));
            assert!(!g.muon.matches(&id, Some(&path)), "{path}: Q/K is head-wise, not plain Muon+");
            assert!(
                g.qk.as_ref().is_some_and(|q| q.matches(&id, Some(&path))),
                "{path}: Q/K must be in the head-wise group"
            );
        }
        // v_proj is not a Q/K projection.
        let (path, id, _) = crate::optim::param_paths(&model)
            .into_iter()
            .find(|(p, _, _)| p.ends_with("gdn2.v_proj.weight"))
            .expect("v_proj");
        assert!(!g.qk.as_ref().is_some_and(|q| q.matches(&id, Some(&path))), "{path}: v_proj is not Q/K");
        // n-gram tables: plain Adam, wd disabled; the key projections are
        // Muon+. Both from the installed groups, both in one test so a table
        // that quietly stopped being a table cannot pass.
        let table = crate::optim::param_paths(&model)
            .into_iter()
            .find(|(p, _, _)| p.contains("engram.memory"))
            .expect("the n-gram table");
        let (path, id, _) = table;
        assert!(g.table.matches(&id, Some(&path)), "{path}: the table must be on plain Adam");
        let (path, id, _) = crate::optim::param_paths(&model)
            .into_iter()
            .find(|(p, _, _)| p.contains("engram.key_projs"))
            .expect("the key projection");
        assert!(!g.table.matches(&id, Some(&path)), "{path}: a key projection is not a table");
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
            // Aux ON for this fixture, explicitly. Every shipped preset has
            // dspark_weight = 0.0 as of 2026-09-29 - DeepSeek's own MTP
            // ablation reports the head bits-per-byte neutral, and our
            // protocol measures BPB - so `default()` gives 0 and the tests
            // below (which check that the aux heads TRAIN end to end) would
            // silently test nothing. They are checking that the arms work, not
            // that the default turns them on.
            dspark_weight: 0.1,
            ..DormouseConfig::default()
        }
    }

    /// Live model: the policy must hold on the real module tree. The expected
    /// Muon+ count is derived from the config topology, not a literal:
    /// n_experts x (gate_up u,v + down u,v) + 1 Engram key_proj + 1
    /// value_proj. With head-wise Q/K routing on, KDA's two q/k matrices move
    /// into the qk group.
    #[test]
    fn routing_validates_on_live_model() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let expected_muon = 4 * cfg.n_experts + 3;
        let c = validate_routing(&model, false, Some(cfg.n_heads)).expect("policy must hold on the live model");
        assert_eq!(c.muon, expected_muon, "Muon+ group must match the topology");
        assert_eq!(c.qk, 2, "gdn2 q/k must be head-wise Muon");
        assert_eq!(c.tables, 1, "n-gram tables group");
        assert!(c.rest > 0);
        // Without head-wise routing the q/k params fall back to rest.
        let c = validate_routing(&model, false, None).expect("policy must hold");
        assert_eq!(c.qk, 0);
    }

    /// The loud gate must FIRE, not merely exist. The old validator had a
    /// test like this one - and it tested a string table against itself, which
    /// is why the dense expert weight could drift while the test stayed green.
    /// These cases are checked against the `ParamGroup`s the optimizer HOLDS,
    /// built by hand through `Installed::of` so they lie on purpose. Each lie
    /// is MINIMAL: it adds exactly one violation on top of the declared group,
    /// so the error the gate reports is unambiguously about that violation
    /// rather than about the first parameter it happens to walk past.
    #[test]
    fn routing_gate_detects_a_lying_install() {
        use dormouse_core::routing::Group;
        use crate::optim::{check_installed, Installed};
        use burn::module::ParamGroup;

        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let r = dormouse_core::routing::routing(&model, false);
        let table = r.group(Group::Table);
        let paths = crate::optim::param_paths(&model);
        // The declared Muon+ ids, recovered through the public declaration.
        let declared_muon: Vec<_> = paths
            .iter()
            .filter(|(_, id, _)| r.group_of_id(id) == Some(Group::Muon))
            .map(|(_, id, _)| *id)
            .collect();
        assert!(!declared_muon.is_empty(), "the fixture has a Muon+ group");

        // 1D param routed to Muon+: the declared group PLUS one norm gain.
        let (norm_path, norm_id, _) = paths
            .iter()
            .find(|(p, _, rank)| p.ends_with("norm.weight") && *rank == 1)
            .map(|(p, id, r)| (p.clone(), *id, *r))
            .expect("a 1D norm gain");
        let mut ids = declared_muon.clone();
        ids.push(norm_id);
        let lie = Installed::of(ParamGroup::from_ids(ids), None, table.clone());
        let err = check_installed(&model, &r, &lie)
            .expect_err("a 1D param in the Muon+ group must fail the gate");
        assert!(err.contains("1D param routed to a Muon+ group"), "unexpected error: {err}");
        assert!(err.contains(&norm_path), "the error must name the parameter: {err}");

        // The whole model in the Muon+ group. Every parameter the policy puts
        // on the base optimizer is now claimed, and the gate must say which.
        let all: Vec<_> = paths.iter().map(|(_, id, _)| *id).collect();
        let lie = Installed::of(ParamGroup::from_ids(all), None, table.clone());
        let err = check_installed(&model, &r, &lie)
            .expect_err("a group claiming parameters the policy does not must fail the gate");
        assert!(err.contains("is not the declared one"), "unexpected error: {err}");

        // The policy says Muon+, the install claims nothing: a 2D parameter
        // silently on the fallback must be loud, not a warning. This is the
        // direction that matters - it is a quality regression, and nothing
        // else in the run would say so.
        let lie = Installed::of(ParamGroup::from_ids(vec![]), None, table);
        let err = check_installed(&model, &r, &lie)
            .expect_err("a declared Muon+ parameter in no installed group must fail the gate");
        assert!(err.contains("is not the declared one"), "unexpected error: {err}");
        assert!(err.contains("Some(Muon)"), "the error must name the declared group: {err}");
    }

    /// An installed group that claims nothing is the reachable form of a
    /// stale marker, and it must be loud (ADR-0019: a rule matching zero
    /// parameters is a failure, not a degradation). The old table caught this
    /// with a dead-marker check; with ids there are no markers, so the Q/K
    /// group is caught per parameter instead - declared `QkHeadWise`, claimed
    /// by nothing, so the run would quietly train attention Q/K on AdamW.
    #[test]
    fn an_installed_but_empty_qk_group_is_loud() {
        use dormouse_core::routing::Group;
        use crate::optim::{check_installed, Installed};
        use burn::module::ParamGroup;

        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let r = dormouse_core::routing::routing(&model, false);
        assert_eq!(r.count(Group::QkHeadWise), 2, "the fixture routes KDA q and k");
        let lie = Installed::of(
            r.group(Group::Muon),
            Some(ParamGroup::from_ids(vec![])),
            r.group(Group::Table),
        );
        let err = check_installed(&model, &r, &lie)
            .expect_err("an installed group that claims nothing must fail the gate");
        assert!(err.contains("is not the declared one"), "unexpected error: {err}");
        assert!(err.contains("Some(QkHeadWise)"), "the error must name the declared group: {err}");
        assert!(err.contains("gdn2.q_proj.weight"), "the error must name the parameter: {err}");
    }

    /// End-to-end on NdArray: the mixed optimizer must drive the training loss
    /// down on a fixed synthetic byte batch (routing, Muon+ step and the
    /// checkpoint-free resume all exercise the real path). Uses a shrunken
    /// config: NdArray autodiff on the `small` preset takes minutes per step.
    #[test]
    fn mixed_optim_converges() {
        // 60 steps: burn's init is seeded per process, so the curve moves by
        // ~0.02 run to run; 60 steps give a ~0.2 head-tail gap and the 0.1
        // assertion keeps 2x headroom over that noise.
        const STEPS: usize = 60;
        let cfg = test_cfg();
        let mut model = DormouseModel::new(&cfg, &device());
        let optim_cfg = TrainCfg { steps: 40, seq_len: 64, batch: 2, lr: 1e-3, wd: 0.01, grad_clip: 1.0, ..Default::default() };
        let mut optim = crate::optim::build_optim_mode(&model, &optim_cfg, "mix");

        // The SAME batch every step: fresh random bytes per step carry no
        // signal at all (their entropy IS ln(256)), so a loss-decrease
        // assertion on them is a coin flip on float noise - the fake-convergence
        // trap. A fixed batch is learnable, so the drop is real.
        let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;
        let next_u8 = |s: &mut u64| {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let x = (*s ^ (*s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            ((x ^ (x >> 27)) >> 33) as u8
        };
        let bytes: Vec<u8> = (0..128).map(|_| next_u8(&mut rng_state)).collect();
        // FNV 3/5/8-gram hashes mod 4096, mirroring ByteStream::hashes.
        let mut hashes = Vec::with_capacity(128 * 3);
        for p in 0..128usize {
            let e = p + 1;
            hashes.push((fnv(&bytes[e.saturating_sub(3)..e]) % 4096) as i64);
            hashes.push((fnv(&bytes[e.saturating_sub(5)..e]) % 4096) as i64);
            hashes.push((fnv(&bytes[e.saturating_sub(8)..e]) % 4096) as i64);
        }
        let (x, h) = bytes_to_tensors(&bytes, &hashes, 64, 2, &device());
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
        let mut losses: Vec<f32> = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            // Constant lr: the WSD warmup would eat half of a 40-step test.
            let (_logits, rec, _k, _aux) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, Some(y.clone()), None);
            let loss = model.loss::<Backend>(rec);
            let v: f32 = loss.clone().try_into_scalar().unwrap_or(f32::NAN);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(1e-3, model, grads);
            losses.push(v);
        }
        let head: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let tail: f32 = losses[STEPS - 5..].iter().sum::<f32>() / 5.0;
        assert!(losses.iter().all(|l| l.is_finite()), "loss must stay finite: {losses:?}");
        // Real margin, not jitter: the run fits the batch's byte marginals
        // (ln(256)=5.545 -> ~5.35 here, monotone). 0.1 is ~1000x the 1e-4
        // wobble the old fresh-noise-per-step data produced.
        println!("mixed_optim head={head:.3} tail={tail:.3} losses={losses:?}");
        assert!(tail < head - 0.1, "loss must decrease: head={head:.3} tail={tail:.3} {losses:?}");
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
        let mut optim = crate::optim::build_optim_mode(&model, &optim_cfg, "adamw");
        let mut rng_state: u64 = 0xDEAD_BEEF;
        let next_u8 = |s: &mut u64| {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let x = (*s ^ (*s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            ((x ^ (x >> 27)) >> 33) as u8
        };
        // Fixed batch, same reason as mixed_optim_converges: fresh random
        // bytes per step have entropy ln(256) and teach nothing.
        let bytes: Vec<u8> = (0..128).map(|_| next_u8(&mut rng_state)).collect();
        let mut hashes = Vec::with_capacity(128 * 3);
        for p in 0..128usize {
            let e = p + 1;
            hashes.push((fnv(&bytes[e.saturating_sub(3)..e]) % 4096) as i64);
            hashes.push((fnv(&bytes[e.saturating_sub(5)..e]) % 4096) as i64);
            hashes.push((fnv(&bytes[e.saturating_sub(8)..e]) % 4096) as i64);
        }
        let (x, h) = bytes_to_tensors(&bytes, &hashes, 64, 2, &device());
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
        let mut losses: Vec<f32> = Vec::with_capacity(50);
        for _ in 0..50usize {
            let (_logits, rec, _k, _aux) = model.forward_with_hidden::<Backend>(x.clone(), Some(h.clone()), None, Some(y.clone()), None);
            let loss = model.loss::<Backend>(rec);
            let v: f32 = loss.clone().try_into_scalar().unwrap_or(f32::NAN);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(1e-3, model, grads);
            losses.push(v);
        }
        let head: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let tail: f32 = losses[45..].iter().sum::<f32>() / 5.0;
        assert!(losses.iter().all(|l| l.is_finite()), "GR loss must stay finite: {losses:?}");
        println!("gr head={head:.3} tail={tail:.3} losses={losses:?}");
        // The GR arm's own slope: monotone 5.546 -> 5.52 over 25 steps, still
        // accelerating, so 50 steps clear 0.05 (500x the old jitter floor).
        assert!(tail < head - 0.05, "GR must learn: head={head:.3} tail={tail:.3} {losses:?}");
    }

    /// The auxiliary objectives (JEPA + DSpark) must train end-to-end on the
    /// live model: aux loss finite, teacher EMA advances away from the
    /// student, and a full optim step keeps everything finite.
    #[test]
    fn aux_losses_and_ema_teacher() {
        let cfg = test_cfg();
        let mut model = DormouseModel::new(&cfg, &device());
        let optim_cfg = TrainCfg { steps: 3, seq_len: 64, batch: 2, lr: 1e-3, wd: 0.01, grad_clip: 1.0, ..Default::default() };
        let mut optim = crate::optim::build_optim_mode(&model, &optim_cfg, "adamw");
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
            let (x, h) = bytes_to_tensors(&bytes, &hashes, 64, 2, &device());
            let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
            let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
            let (_logits, rec, _k, aux) = model.forward_with_hidden::<Backend>(
                x, Some(h), None, Some(y), teacher.as_ref(),
            );
            let a = aux.expect("aux must be Some with preset weights on");
            let a_val: f32 = a.clone().try_into_scalar().expect("aux scalar");
            assert!(a_val.is_finite(), "aux loss must be finite, got {a_val}");
            let grads = GradientsParams::from_grads((model.loss::<Backend>(rec) + a).backward(), &model);
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
        let (x, h) = bytes_to_tensors(&bytes, &hashes, 64, 2, &device());
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
        // (2) The aux consumes the target tensor it is handed. The masked L1
        // draws a fresh random mask per call (burn-jepa `mask_indices`);
        // when both draws land empty the L1 term is 0.0 in both calls and
        // the two aux values coincide bitwise. P(both empty) ~ 5-7% — a
        // pre-existing flake (reproduced 2/40 on burn pre.3 as well), not a
        // migration artifact. Retry across rounds: fail only if EVERY round
        // sees identical aux values (P ~ 2e-6).
        let mut depends_on_target = false;
        for _ in 0..5 {
            let (_, _, _, aux_a) = model.forward_with_jepa_targets::<Backend>(
                x.clone(), Some(h.clone()), None, Some(y.clone()), Some(target.clone()),
            );
            let scaled = target.clone().mul_scalar(3.0);
            let (_, _, _, aux_b) = model.forward_with_jepa_targets::<Backend>(
                x.clone(), Some(h.clone()), None, Some(y.clone()), Some(scaled),
            );
            let a: f32 = aux_a.expect("offline aux must be Some with JEPA on").try_into_scalar().unwrap();
            let b: f32 = aux_b.expect("offline aux must be Some").try_into_scalar().unwrap();
            assert!(a.is_finite() && b.is_finite(), "offline aux must be finite: {a} {b}");
            if (a - b).abs() > 1e-4 * a.abs().max(1.0) {
                depends_on_target = true;
                break;
            }
        }
        assert!(
            depends_on_target,
            "offline aux must depend on the supplied target (all 5 rounds identical)"
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

    /// The NaN firewall must make a non-finite loss a true no-op. This is
    /// the bug that killed official_v4: between log steps the NaN loss was
    /// invisible, backward produced NaN grads, the optimizer wrote them into
    /// the weights, and the guard replayed the poisoned region forever.
    /// Contract: masked loss == 0, counter == 1, and with wd=0 (fresh AdamW
    /// state) the step leaves a weight BIT-IDENTICAL.
    #[test]
    fn nan_firewall_makes_a_nonfinite_loss_a_noop() {
        let cfg = test_cfg();
        let mut model = DormouseModel::new(&cfg, &device());
        let optim_cfg = TrainCfg { steps: 1, seq_len: 64, batch: 2, lr: 1e-3, wd: 0.0, grad_clip: 1.0, ..Default::default() };
        let mut optim = crate::optim::build_optim_mode(&model, &optim_cfg, "adamw");
        let bytes: Vec<u8> = (0..128).map(|i| (i * 7) as u8).collect();
        let hashes: Vec<i64> = (0..384).map(|p| (p as i64) % 4096).collect();
        let (x, h) = bytes_to_tensors(&bytes, &hashes, 64, 2, &device());
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
        let (_l, rec, _k, _a) = model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y), None);
        let loss = model.loss::<Backend>(rec);

        // A finite loss passes through untouched.
        let ok = mask_nonfinite(loss.clone());
        let ok_v: f32 = ok.clone().try_into_scalar().unwrap();
        let base_v: f32 = loss.clone().try_into_scalar().unwrap();
        assert!((ok_v - base_v).abs() < 1e-6, "finite loss must pass through: {ok_v} vs {base_v}");

        // The poisoned loss: masked to exactly 0.
        let nan = Tensor::<1>::from_data(TensorData::new(vec![f32::NAN], [1]), &device());
        let masked = mask_nonfinite(loss.clone() + nan);
        let m_v: f32 = masked.clone().try_into_scalar().unwrap();
        assert_eq!(m_v, 0.0, "non-finite loss must mask to exactly 0, got {m_v}");

        // The step on the masked loss must not move a single weight.
        let before: Vec<f32> = model.loop_block.norm.weight.val().into_data().try_to_vec().unwrap();
        let grads = GradientsParams::from_grads(masked.backward(), &model);
        model = optim.step(1e-3, model, grads);
        let after: Vec<f32> = model.loop_block.norm.weight.val().into_data().try_to_vec().unwrap();
        assert_eq!(before, after, "a masked loss must produce exactly zero grads");
        assert!(after.iter().all(|x| x.is_finite()), "weights must stay finite");
    }

    /// The random-depth arm (ADR-0013 rank 2) must be honest: it trains, the
    /// depths become genuinely different models, and an out-of-range depth is
    /// refused loudly rather than clamped.
    ///
    /// Note the order: depth is compared AFTER training, because at init the
    /// ReZero residual scale is 0, so every depth collapses to the same
    /// uniform output (5.545 = ln 256) and a pre-training comparison would
    /// "pass" without the depth doing anything.
    #[test]
    fn random_depth_is_honest_and_still_learns() {
        // The sampler: deterministic, in range, and it actually spreads.
        assert_eq!(sample_depth(7, 4), sample_depth(7, 4), "same step, same depth");
        let hit: std::collections::HashSet<usize> = (0..64).map(|s| sample_depth(s, 4)).collect();
        assert_eq!(hit.len(), 4, "all depths 1..=4 must be reachable: {hit:?}");
        assert!(hit.iter().all(|d| (1..=4).contains(d)), "depth out of range: {hit:?}");
        assert_eq!(sample_depth(3, 1), 1, "max_iter=1 must degenerate to fixed depth");

        let cfg = test_cfg(); // max_iter = 4
        let mut model = DormouseModel::new(&cfg, &device());
        assert_eq!(model.loop_block.max_iter, 4, "test config must have 4 iterations");
        let bytes: Vec<u8> = (0..128).map(|i| (i * 7) as u8).collect();
        let hashes: Vec<i64> = (0..384).map(|p| (p as i64) % 4096).collect();
        let (x, h) = bytes_to_tensors(&bytes, &hashes, 64, 2, &device());
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());

        // Train with sampled depths on a fixed batch: must descend for real.
        let optim_cfg = TrainCfg { steps: 60, seq_len: 64, batch: 2, lr: 1e-3, wd: 0.0, grad_clip: 1.0, ..Default::default() };
        let mut optim = crate::optim::build_optim_mode(&model, &optim_cfg, "adamw");
        let mut losses = Vec::with_capacity(60);
        for step in 0..60u64 {
            let mut m = model.clone();
            m.set_loop_depth(if step % 2 == 0 { Some(sample_depth(step, 4)) } else { None });
            let (_l, rec, _k, _a) = m.forward_with_hidden::<Backend>(
                x.clone(), Some(h.clone()), None, Some(y.clone()), None,
            );
            let loss = m.loss::<Backend>(rec);
            let v: f32 = loss.clone().try_into_scalar().unwrap_or(f32::NAN);
            let grads = GradientsParams::from_grads(loss.backward(), &m);
            model = optim.step(1e-3, model, grads);
            losses.push(v);
        }
        let head: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let tail: f32 = losses[55..].iter().sum::<f32>() / 5.0;
        assert!(losses.iter().all(|l| l.is_finite()), "rand-depth losses must stay finite");
        assert!(
            tail < head - 0.1,
            "random depth must still learn: head={head:.3} tail={tail:.3} {losses:?}"
        );

        // The trained model: depth 1 and depth 4 are DIFFERENT models, and
        // None reproduces the fixed-depth run exactly.
        let loss_at = |m: &DormouseModel, depth: Option<usize>| {
            let mut m2 = m.clone();
            m2.set_loop_depth(depth);
            let (_l, rec, _k, _a) = m2.forward_with_hidden::<Backend>(
                x.clone(), Some(h.clone()), None, Some(y.clone()), None,
            );
            m2.loss::<Backend>(rec).try_into_scalar().unwrap_or(f32::NAN)
        };
        let d1 = loss_at(&model, Some(1));
        let d4 = loss_at(&model, Some(4));
        assert!(d1.is_finite() && d4.is_finite(), "both depths must be finite: {d1} {d4}");
        assert!(
            (d1 - d4).abs() > 1e-4,
            "depth 1 and depth 4 must be different models, got {d1} vs {d4}"
        );
        assert!((loss_at(&model, None) - d4).abs() < 1e-6, "None must mean fixed depth");

        // Loud refusal: a clamped depth would make the A/B lie about what it
        // trained.
        let mut bad = model.clone();
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bad.set_loop_depth(Some(9));
        }))
        .is_err();
        assert!(refused, "depth 9 (> max_iter 4) must be refused loudly");
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
            let mut optim = crate::optim::build_optim_mode(&model, &optim_cfg, mode);
            let bytes: Vec<u8> = (0..128).map(|i| (i * 7) as u8).collect();
            let hashes: Vec<i64> = (0..384).map(|p| (p as i64) % 4096).collect();
            let (x, h) = bytes_to_tensors(&bytes, &hashes, 64, 2, &device());
            let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
            let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
            let (_logits, rec, _k, _aux) = model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y), None);
            let loss = model.loss::<Backend>(rec);
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
        let mut optim = crate::optim::build_optim_mode(&model, &optim_cfg, "mix");
        let bytes: Vec<u8> = (0..128).map(|i| (i * 7) as u8).collect();
        let hashes: Vec<i64> = (0..384).map(|p| (p as i64) % 4096).collect();
        let (x, h) = bytes_to_tensors(&bytes, &hashes, 64, 2, &device());
        let shifted: Vec<i64> = bytes.iter().skip(1).chain(std::iter::once(&bytes[0])).map(|&b| b as i64).collect();
        let y: Tensor<2, Int> = Tensor::from_data(TensorData::new(shifted, [2, 64]), &device());
        let (_logits, rec, _k, _aux) = model.forward_with_hidden::<Backend>(x, Some(h), None, Some(y), None);
        let grads = GradientsParams::from_grads(model.loss::<Backend>(rec).backward(), &model);
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
    /// counts must shift from muon to rest, and the install must stay valid
    /// (it is checked against the live tree, not against a marker list).
    #[test]
    fn factors_fallback_routing() {
        let cfg = test_cfg();
        let model = DormouseModel::new(&cfg, &device());
        let with_factors = validate_routing(&model, false, Some(cfg.n_heads))
            .expect("the default install must be valid");
        let without = validate_routing(&model, true, Some(cfg.n_heads))
            .expect("the --factors-fallback install must be valid");
        assert_eq!(
            without.muon + 4 * cfg.n_experts,
            with_factors.muon,
            "factors must move from the Muon+ group to the fallback"
        );
        assert_eq!(without.rest - with_factors.rest, 4 * cfg.n_experts);
    }
}

