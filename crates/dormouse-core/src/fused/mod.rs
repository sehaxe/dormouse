//! fused - the PonderNet loop step as ONE custom autodiff op with
//! hand-written cubecl kernels (plan M0+M1, fp32). See `backward.rs` for the
//! gradient math and `kernels.rs` for the device code.
//!
//! M0: [`fused_matmul`] - a `Backward` op skeleton wrapping a single fp32
//! matmul: extract the cube handle from the autodiff tensor, launch on the
//! ComputeClient, `grads.register` both inputs. Proves the burn plumbing:
//! `loss.backward()` sees our grads exactly like a built-in op, so
//! `GradientsParams::from_grads` / `optim.step` downstream are unchanged.
//!
//! M1: [`ponder_loop_step`] - with max_iter=1 and the attention/engram arms
//! off, the whole iteration body (controller gate + expert TSCT
//! gate_up->silu->down + ReZero residual + halt head + per-step CE) runs as
//! raw kernels under ONE autodiff node. The CPU only issues launches; no
//! per-op graph nodes are built.
//!
//! Deviations from the plan, deliberate:
//! - matmuls are plain fp32 FMA kernels, not cmma: fp32 cmma on NVIDIA is
//!   TF32-lossy or FFMA-emulated (no speed win), while the acceptance
//!   criterion is exactness vs the burn reference (rel < 1e-4). bf16 cmma is
//!   the M5 fast path, per the plan itself.
//! - act-quant STE is not in the kernel yet (M1 runs act_quant=None; STE is
//!   grad passthrough, added with the quant kernel in M2).
//!
//! Kernel conventions mirror burn-spectral's `moe_fused.rs`: flat 1D dense
//! buffers, `#[cube(launch_unchecked)]`, `grads.register::<Inner>(node.id,
//! grad)`. No 4D autodiff tensor is ever created (sm_120 4D-slice crash).

#![allow(clippy::too_many_arguments)]

pub mod backward;
pub mod kernels;
#[cfg(all(test, feature = "cuda"))]
mod tests;

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::autodiff::Autodiff;
use burn::backend::{DispatchKindConversion, DispatchTensor};
use burn::tensor::{Device, Int, Tensor};
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use backward::PonderState;
use crate::param::{LinearLike, LinearLikeInner};
use crate::loop_block::LoopBlock;
use crate::config::DormouseConfig;
use burn::module::Module;

pub(crate) type Cuda = cubecl::cuda::CudaRuntime;
pub(crate) type CB = burn_cubecl::CubeBackend<Cuda>;
pub(crate) type CAd = Autodiff<CB>;
pub(crate) type AdPrim = <CAd as burn::backend::BackendTypes>::FloatTensorPrimitive;
/// Bare cube tensor of the CUDA runtime (buffer handle + client).
pub(crate) type CubeTensor = burn_cubecl::tensor::CubeTensor<Cuda>;

const UNITS: CubeDim = CubeDim::new_3d(32, 1, 1);
const EW: CubeDim = CubeDim::new_3d(256, 1, 1);

#[cfg(test)]
thread_local! {
    /// Forward-buffer dumps for the bisect test (DM_FUSED_DEBUG=1).
    pub static FWD_DUMP: std::cell::RefCell<Vec<(&'static str, Vec<f32>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// `DM_FUSED=1` enables the fused path at the M6 wiring point (default off:
/// today's burn path is untouched).
pub fn fused_enabled() -> bool {
    std::env::var("DM_FUSED").map(|v| v == "1").unwrap_or(false)
}

/// `DM_FUSED_BF16=1` halves the workspace (M5): per-iteration buffers are
/// stored bf16, compute stays fp32. Off by default until the cast kernels
/// are profiled.
pub fn fused_bf16_enabled() -> bool {
    std::env::var("DM_FUSED_BF16").map(|v| v == "1").unwrap_or(false)
}

// ---------------------------------------------------------------------------
// handle plumbing (mirror of burn-spectral moe_fused.rs)
// ---------------------------------------------------------------------------

pub(crate) fn dense<const D: usize, K>(t: Tensor<D, K>) -> Tensor<D, K>
where
    K: burn::tensor::kind::Basic,
{
    let dims = t.dims();
    let n = dims.iter().product::<usize>();
    t.reshape::<1, _>([n]).reshape::<D, _>(dims)
}

pub(crate) fn cube_of2(t: &Tensor<2>) -> Option<CubeTensor> {
    t.clone().try_into_primitive::<CB>().ok()
}

pub(crate) fn cube_of1(t: &Tensor<1>) -> Option<CubeTensor> {
    t.clone().try_into_primitive::<CB>().ok()
}

pub(crate) fn cube_int2(t: &Tensor<2, Int>) -> Option<CubeTensor> {
    t.clone().try_into_primitive::<CB>().ok()
}

pub(crate) fn zeros1(dev: &Device, n: usize) -> (Tensor<1>, CubeTensor) {
    let t = Tensor::<1>::zeros([n], dev);
    let c = cube_of1(&t).expect("fused op requires CUDA tensors");
    (t, c)
}

/// Zero-fill on OUR stream: burn-side `Tensor::zeros` dispatch may land on a
/// different stream than the raw `launch_unchecked` kernels, and in backward
/// there is no fence between allocation and first use (the forward fences
/// with a `sync` before its kernels). A raw fill is FIFO-ordered with the
/// other raw launches, so `+=` kernels can never race the init.
pub(crate) fn zeros_raw(dev: &Device, client: &ComputeClient<Cuda>, n: usize) -> CubeTensor {
    let t = Tensor::<1>::empty([n], dev);
    let c = cube_of1(&t).expect("fused op requires CUDA tensors");
    unsafe {
        kernels::fill_kernel::launch_unchecked::<f32, Cuda>(
            client,
            ew_cubes(n),
            EW,
            BufferArg::from_raw_parts(c.handle.clone(), n),
            0.0,
            n as u32,
        );
    }
    c
}

pub(crate) fn empty1(dev: &Device, n: usize) -> (Tensor<1>, CubeTensor) {
    let t = Tensor::<1>::empty([n], dev);
    let c = cube_of1(&t).expect("fused op requires CUDA tensors");
    (t, c)
}

/// An autodiff tensor split into the bare cube primitive + graph pieces.
/// (The `NodeRef` type is private in burn-autodiff; always reach it through
/// `at.node.clone()` so it stays inferred.)
pub(crate) struct Ad {
    pub t: Tensor<2>,
    pub at: AdPrim,
    pub prim: CubeTensor,
}

pub(crate) fn ad2(t: Tensor<2>) -> Ad {
    let t = dense(t);
    let at = t
        .clone()
        .try_into_primitive::<CAd>()
        .expect("fused op requires Autodiff<CudaBackend> tensors");
    Ad {
        prim: at.primitive.clone(),
        at,
        t,
    }
}

pub(crate) fn ad1(t: Tensor<1>) -> (Tensor<1>, AdPrim, CubeTensor) {
    let t = dense(t);
    let at = t
        .clone()
        .try_into_primitive::<CAd>()
        .expect("fused op requires Autodiff<CudaBackend> tensors");
    (t, at.clone(), at.primitive.clone())
}

/// The (bare) allocation device of a bare cube primitive.
pub(crate) fn bare_device(prim: &CubeTensor) -> Device {
    Tensor::<2>::from_primitive::<CB>(prim.clone()).device()
}

fn ew_cubes(n: usize) -> CubeCount {
    CubeCount::Static(n.div_ceil(256) as u32, 1, 1)
}

/// Block until every task enqueued on this client completes. Raw
/// `launch_unchecked` calls run on the caller's stream, which is distinct
/// from streams other burn dispatch tasks may land on - without this barrier
/// our kernels race the tensor-graph kernels that produce/consume the same
/// buffers (observed: matmul results racing the `zeros` init of their own
/// output buffer). Same escape hatch as burn-spectral's moe_fused.
pub(crate) fn sync(client: &ComputeClient<Cuda>) {
    futures_lite::future::block_on(client.sync()).expect("cuda sync failed");
}

/// The single matmul launch used by every mm in forward+backward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_mm(
    client: &ComputeClient<Cuda>,
    a: &CubeTensor,
    b: &CubeTensor,
    s: &CubeTensor,
    bm: &CubeTensor,
    out: &CubeTensor,
    m: usize,
    k: usize,
    n: usize,
    am: usize,
    ak: usize,
    bk: usize,
    bn: usize,
    al: usize,
    bl: usize,
    ta: bool,
    tb: bool,
    tern_b: bool,
    scale_k: bool,
    accum: bool,
) {
    unsafe {
        kernels::mm_kernel::launch_unchecked::<f32, Cuda>(
            client,
            ew_cubes(m * n),
            EW,
            AddressType::U32,
            BufferArg::from_raw_parts(a.handle.clone(), al),
            BufferArg::from_raw_parts(b.handle.clone(), bl),
            BufferArg::from_raw_parts(s.handle.clone(), k.max(1)),
            BufferArg::from_raw_parts(bm.handle.clone(), 1),
            BufferArg::from_raw_parts(out.handle.clone(), m * n),
            m as u32,
            k as u32,
            n as u32,
            am as u32,
            ak as u32,
            bk as u32,
            bn as u32,
            ta,
            tb,
            tern_b,
            scale_k,
            accum,
        );
    }
}

fn launch_absmean(w: &CubeTensor, out: &CubeTensor, n: usize) {
    let client = w.client.clone();
    unsafe {
        kernels::absmean_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(1, 1, 1),
            UNITS,
            BufferArg::from_raw_parts(w.handle.clone(), n),
            BufferArg::from_raw_parts(out.handle.clone(), 1),
            n as u32,
        );
    }
}

// ---------------------------------------------------------------------------
// M0: one fp32 matmul as a single custom autodiff op
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MmState {
    a: Tensor<2>,
    w: Tensor<2>,
    m: usize,
    k: usize,
    n: usize,
}

#[derive(Debug)]
struct PonderLoopMatmul;

impl Backward<CB, 2> for PonderLoopMatmul {
    type State = MmState;

    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let st = ops.state;
        let dy = Tensor::<2>::from_primitive::<CB>(grads.consume::<CB>(&ops.node));
        let dyc = cube_of2(&dense(dy)).expect("cuda grad");
        let ac = cube_of2(&st.a).expect("cuda tensor");
        let wc = cube_of2(&st.w).expect("cuda tensor");
        let client = dyc.client.clone();
        // fence: the grad was produced by burn-side kernels; ours follow
        sync(&client);
        let dev = bare_device(&dyc);

        // dA = dY @ W^T: [m,n]·[n,k] -> [m,k]. B'[j,c] = W[c,j] lives at
        // flat[c·n + j] (W is [k,n] row-major): tb=true with bn = n.
        let (da_keep, dac) = empty1(&dev, st.m * st.k);
        launch_mm(&client, &dyc, &wc, &dyc, &dyc, &dac, st.m, st.n, st.k, st.n, 1, st.n, 1, st.m * st.n, st.k * st.n, false, true, false, false, false);
        // dW = A^T @ dY: [k,m]^T·[m,n] -> [k,n]
        let (dw_keep, dwc) = empty1(&dev, st.k * st.n);
        launch_mm(&client, &ac, &dyc, &dyc, &dyc, &dwc, st.k, st.m, st.n, st.k, 1, st.n, 1, st.m * st.k, st.m * st.n, true, false, false, false, false);
        sync(&client);
        // registered grads carry the parent's rank (shape-checked downstream)
        let dac2 = Tensor::<1>::from_primitive::<CB>(dac)
            .reshape::<2, _>([st.m, st.k])
            .try_into_primitive::<CB>()
            .expect("grad reshape");
        let dwc2 = Tensor::<1>::from_primitive::<CB>(dwc)
            .reshape::<2, _>([st.k, st.n])
            .try_into_primitive::<CB>()
            .expect("grad reshape");
        if let Some(node) = ops.parents[0].clone() {
            grads.register::<CB>(node.id, dac2);
        }
        if let Some(node) = ops.parents[1].clone() {
            grads.register::<CB>(node.id, dwc2);
        }
        let _ = (da_keep, dw_keep);
    }
}

/// M0 entry: `y = a @ w` on one custom-op node (fp32, CUDA only).
pub fn fused_matmul(a: Tensor<2>, w: Tensor<2>) -> Tensor<2>
where
    DispatchTensor: DispatchKindConversion<CAd> + DispatchKindConversion<CB>,
{
    let aa = a
        .clone()
        .try_into_primitive::<CAd>()
        .expect("fused_matmul requires Autodiff<CudaBackend> tensors");
    let wa = w
        .clone()
        .try_into_primitive::<CAd>()
        .expect("fused_matmul requires Autodiff<CudaBackend> tensors");
    let a_t = Tensor::<2>::from_primitive::<CB>(aa.primitive.clone());
    let [m, k] = a_t.dims();
    let w_t = Tensor::<2>::from_primitive::<CB>(wa.primitive.clone());
    let [k2, n] = w_t.dims();
    assert_eq!(k, k2, "matmul shape mismatch");
    let dev = bare_device(&aa.primitive);

    let a_d = dense(a_t);
    let w_d = dense(w_t);
    let ac = cube_of2(&a_d).expect("cuda tensor");
    let wc = cube_of2(&w_d).expect("cuda tensor");
    let client = ac.client.clone();
    // unused mm operands must be real allocations at least as long as the
    // declared binding length (never declare more than you allocate)
    let (_dm_t, dummy) = empty1(&dev, 1);
    let (_ds_t, dums) = empty1(&dev, k.max(1));
    // rank-2 metadata matters: burn wraps the primitive as-is, so the op
    // output must carry [m,n] shape/strides (a raw 1D buffer would break
    // later rank-based ops on the node's grad). Allocating 1D and reshaping
    // keeps the buffer DENSE: a direct 2D alloc gets cubecl's pitched row
    // layout (rows padded to the alignment), which the flat kernel writes
    // wrong (row 0 lands flat, everything after is garbage).
    let out_keep = Tensor::<1>::zeros([m * n], &dev).reshape::<2, _>([m, n]);
    let outc = cube_of2(&out_keep).expect("cuda tensor");
    sync(&client);
    launch_mm(&client, &ac, &wc, &dums, &dummy, &outc, m, k, n, k, 1, n, 1, m * k, k * n, false, false, false, false, false);
    sync(&client);
    let nodes = [aa.node.clone(), wa.node.clone()];
    let prep = PonderLoopMatmul.prepare::<NoCheckpointing>(nodes);
    match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut p) => {
            let _ = [p.checkpoint(&aa), p.checkpoint(&wa)];
            let out = p.finish(
                MmState {
                    a: a_d,
                    w: w_d,
                    m,
                    k,
                    n,
                },
                outc,
            );
            Tensor::from_primitive::<CAd>(out)
        }
        OpsKind::UnTracked(p) => Tensor::from_primitive::<CAd>(p.finish(outc)),
    }
}

// ---------------------------------------------------------------------------
// M1: the fused single-iteration Ponder step
// ---------------------------------------------------------------------------

/// Refuse to run the fused op on any TSCT factor that is not the PLAIN
/// fp32-master global-absmean ternary STE the kernels hardcode (`mm_kernel`
/// tern_b: global mean(|w|) scale, 0.7 dead zone, STE through fp32
/// masters). burn-spectral's SpectralLinear has other factor modes (2-bit,
/// N:M, stochastic, per-column, asym, annealing alpha<1, Fp8/Fp4/Bf16/Fp16
/// factor quant) whose forward computes a DIFFERENT function - the op only
/// sees detached `Fac` triples and cannot tell, so call this at extraction
/// time (where the triples are pulled off the model) and fail loudly.
pub fn assert_fusable(ll: &LinearLike, what: &str) {
    let LinearLikeInner::Tsct(l) = &ll.inner else {
        panic!("{what}: fused path requires the TSCT arm");
    };
    let bad = if l.two_bit {
        "two_bit (5-level) factor quant".to_string()
    } else if l.nm_on() {
        format!("N:M sparsity ({}/{})", l.nm_n, l.nm_m)
    } else if l.stochastic {
        "stochastic ternary rounding".to_string()
    } else if l.per_column {
        "per-column ternary scaling".to_string()
    } else if l.asym {
        "asym mode (V left unquantized)".to_string()
    } else if l.alpha != 1.0 {
        format!("ternary annealing alpha={}", l.alpha)
    } else if l.quant != burn_spectral::QuantFormat::Fp32 {
        format!("factor-quant format {:?}", l.quant)
    } else {
        return;
    };
    panic!("{what}: fused op only supports plain fp32-master ternary STE, found {bad}");
}

/// One TSCT factor triple (u master, scale, v master) as tracked tensors.
#[derive(Clone, Debug)]
pub struct Fac {
    pub u: Tensor<2>,
    pub s: Tensor<1>,
    pub v: Tensor<2>,
}

/// Everything the fused N-iteration ponder loop needs, as tracked tensors.
#[derive(Clone, Debug)]
pub struct PonderInputs {
    /// loop input (embedding output) [b, t, d]
    pub x: Tensor<3>,
    /// target byte indices [b·t, 1]
    pub targets: Tensor<2, Int>,
    pub controller_w: Tensor<2>,
    /// loop_block.norm gamma (per-iteration pre-norm)
    pub norm_g: Tensor<1>,
    /// model.norm gamma (final readout over out_acc)
    pub final_norm_g: Tensor<1>,
    /// [max_iter, d]
    pub iter_embed: Tensor<2>,
    pub residual_scale: Tensor<1>,
    pub halt_w: Tensor<2>,
    /// [gate_up, down] factor triples per expert
    pub experts: Vec<[Fac; 2]>,
    pub out_proj: Fac,
    pub lm_head: Fac,
    pub norm_eps: f32,
    /// PonderNet KL prior λ_p (model.rs ponder_kl)
    pub ponder_prior: f32,
    /// Optional arms (KDA/MSA/Engram) — when Some, the fused loop includes
    /// the shared attention (KDA+MSA+router blend) and Engram (hashed_ids path)
    /// inside the single node. `cfg` carries the geometry for the copy.
    pub hashed_ids: Option<Tensor<3, Int>>,
    pub loop_block_bytes: Option<Vec<u8>>,
    pub cfg: Option<crate::config::DormouseConfig>,
}

#[derive(Clone, Debug)]
pub(crate) struct FacC {
    u: CubeTensor,
    v: CubeTensor,
    s: CubeTensor,
    mu: CubeTensor,
    mv: CubeTensor,
}

/// Extract one TSCT factor: allocate the absmean outputs + the graph pieces.
/// NO raw launches here: the u/v masters are burn-side buffers that
/// optim.step/retract rewrite every step on the burn stream - the absmean
/// reads must wait for the fence in `ponder_loop_step` (see it for the
/// invariant). The bridged twins keep every buffer alive until backward
/// completes.
fn fac_cuda(
    f: &Fac,
    keep2: &mut Vec<Tensor<2>>,
    keep1: &mut Vec<Tensor<1>>,
) -> (FacC, [AdPrim; 3]) {
    let u = ad2(f.u.clone());
    let v = ad2(f.v.clone());
    let (s_t, s_at, s_prim) = ad1(f.s.clone());
    let dev = bare_device(&u.prim);
    let (mu_t, mu_c) = empty1(&dev, 1);
    let (mv_t, mv_c) = empty1(&dev, 1);
    keep2.push(u.t);
    keep2.push(v.t);
    keep1.push(s_t);
    keep1.push(mu_t);
    keep1.push(mv_t);
    (
        FacC {
            u: u.prim,
            v: v.prim,
            s: s_prim,
            mu: mu_c,
            mv: mv_c,
        },
        [u.at, s_at, v.at],
    )
}

/// absmean of both masters of one factor (raw launch; FIFO-ordered on the
/// caller's stream, so it must be enqueued after the fence and before the
/// ternary mms that consume `mu`/`mv`).
fn launch_fac_means(fc: &FacC) {
    let n_u = fc.u.meta.shape().dims::<2>().iter().product::<usize>();
    let n_v = fc.v.meta.shape().dims::<2>().iter().product::<usize>();
    launch_absmean(&fc.u, &fc.mu, n_u);
    launch_absmean(&fc.v, &fc.mv, n_v);
}

/// Parent order (fixed; matches the registration sequence in `backward`):
/// 0 x | 1 controller_w | 2 norm_g | 3 iter_embed | 4 residual_scale |
/// 5 halt_w | per expert e: 6+6e gate_up u/s/v, 9+6e down u/s/v |
/// 24 out_proj u/s/v | 27 lm_head u/s/v | 30 final_norm_g.
#[derive(Debug)]
struct PonderLoop;

impl Backward<CB, 31> for PonderLoop {
    type State = PonderState;

    fn backward(self, ops: Ops<Self::State, 31>, grads: &mut Gradients, cp: &mut Checkpointer) {
        backward::ponder_backward(ops, grads, cp)
    }
}

/// Saved forward buffers of one loop iteration (backward reads them back).
/// All flat 1D, one allocation set per iteration: raw `BufferArg`s carry no
/// offset field, so the N-iteration workspace is the N-fold of these flat
/// buffers - never a 4D tensor.
#[derive(Clone, Debug)]
pub(crate) struct IterBufs {
    pub h_ctx: CubeTensor,
    pub normed: CubeTensor,
    pub inv: CubeTensor,
    pub ctrl_in: CubeTensor,
    pub raw: CubeTensor,
    pub w_ffn: CubeTensor,
    pub blend: CubeTensor,
    pub ffn: CubeTensor,
    pub y: CubeTensor,
    pub h: CubeTensor,
    pub z_o: CubeTensor,
    pub step_out: CubeTensor,
    pub z_l: CubeTensor,
    pub logits: CubeTensor,
    pub halt_in: CubeTensor,
    pub ce: CubeTensor,
    pub ceb: CubeTensor,
    /// halting state: λ_n, p_n, not-halted factor entering this iteration
    pub lam: CubeTensor,
    pub p: CubeTensor,
    pub nh: CubeTensor,
    /// per-expert [z_e, a_e, mid, z_d, out_e]
    pub ws: Vec<[CubeTensor; 5]>,
}

/// The fused N-iteration Ponder loop + PonderNet loss parts + final readout
/// (M3). Returns `(logits [b,t,v], rec [1], p_dist [b,N], kl [1])` - the
/// `forward_with_hidden` triple plus the in-op PonderNet KL scalar. Per-step
/// CE, the halting recurrence and the final norm + lm_head readout (fp32,
/// bf16-logits NaN rule) all run under ONE autodiff node; the per-iteration
/// attention/engram arms stay off (M4 adds them).
pub fn ponder_loop_step(inp: PonderInputs) -> (Tensor<3>, Tensor<1>, Tensor<2>, Tensor<1>)
where
    DispatchTensor: DispatchKindConversion<CAd> + DispatchKindConversion<CB>,
{
    // M4: if arms are requested, delegate to the arms-aware path which
    // includes KDA/MSA/Engram inside the single node (still 1 node/step).
    if inp.loop_block_bytes.is_some() || inp.hashed_ids.is_some() || inp.cfg.is_some() {
        // Check if any arm would actually run (use_* true)
        let wants_arms = inp
            .cfg
            .as_ref()
            .map(|c| c.use_kda || c.use_msa || c.use_engram)
            .unwrap_or(false)
            || inp.loop_block_bytes.is_some();
        if wants_arms {
            return ponder_loop_step_arms(inp);
        }
    }
    let nexp = inp.experts.len();
    // The op's parent list is fixed at compile time (Backward<CB, 31>):
    // 6 dense weights + 6 per expert + 3 out_proj + 3 lm_head + 1 final
    // norm gamma = 13 + 6·nexp. Only n_experts = 3 is compiled
    // (`small`/`base`; one_b runs 4).
    assert_eq!(nexp, 3, "fused op is compiled for n_experts=3 (31 parents = 13 + 6·n_experts); got n_experts={nexp} - lift PonderLoop's arity before running one_b");
    let [b, t, d] = inp.x.dims();
    let bt = b * t;
    // burn-nn Linear uses the Col layout: the weight is [in, out] = [2d, pad]
    // (x @ W, no transpose in the reference forward).
    let [two_d, pad] = dense(inp.controller_w.clone()).dims();
    assert_eq!(two_d, 2 * d, "controller weight must be [2d, pad]");
    let n_iter = inp.iter_embed.dims()[0];
    // Truncated-geometric KL prior, renormalized - host math copied exactly
    // from model.rs ponder_kl (the log is taken inside the kernel).
    let mut prior: Vec<f32> = Vec::with_capacity(n_iter);
    let mut mass = 1.0f32;
    let mut total = 0.0f32;
    for _ in 0..n_iter {
        let prob = inp.ponder_prior * mass;
        prior.push(prob);
        total += prob;
        mass *= 1.0 - inp.ponder_prior;
    }
    let inv = 1.0 / total;
    let prior_v: Vec<f32> = prior.iter().map(|x| x * inv).collect();
    let x2 = inp.x.clone().reshape([bt, d]);
    let xa = x2
        .clone()
        .try_into_primitive::<CAd>()
        .expect("fused op requires Autodiff<CudaBackend> tensors");
    let x_prim = xa.primitive.clone();
    let x_t = Tensor::<2>::from_primitive::<CB>(x_prim.clone());
    let dev = x_t.device();
    // bare-backend tensor: cube_of1 refuses autodiff-wrapped primitives
    let prior_t = Tensor::<1>::from_data(
        burn::tensor::TensorData::new(prior_v.clone(), [n_iter]),
        &dev,
    );
    let prior_c = cube_of1(&dense(prior_t)).expect("cuda prior");
    let tgt_c = cube_int2(&inp.targets).expect("cuda targets");

    let mut keep2: Vec<Tensor<2>> = Vec::new();
    let mut keep1: Vec<Tensor<1>> = Vec::new();
    let mut ats: Vec<AdPrim> = Vec::with_capacity(31);
    let mut nodes = Vec::with_capacity(31); // [NodeRef; 31], inferred

    // ---- parents 1..6 (dense weights + params)
    let wc = ad2(inp.controller_w.clone());
    let (g_t, g_at, g_prim) = ad1(inp.norm_g.clone());
    let (gf_t, gf_at, gf_prim) = ad1(inp.final_norm_g.clone());
    let ie = ad2(inp.iter_embed.clone());
    let (rs_t, rs_at, rs_prim) = ad1(inp.residual_scale.clone());
    let wh = ad2(inp.halt_w.clone());
    keep2.push(wc.t);
    keep2.push(ie.t);
    keep2.push(wh.t);
    keep1.push(g_t);
    keep1.push(gf_t);
    keep1.push(rs_t);
    nodes.push(xa.node.clone());
    ats.push(xa.clone());
    nodes.push(wc.at.node.clone());
    ats.push(wc.at);
    nodes.push(g_at.node.clone());
    ats.push(g_at);
    nodes.push(ie.at.node.clone());
    ats.push(ie.at);
    nodes.push(rs_at.node.clone());
    ats.push(rs_at);
    nodes.push(wh.at.node.clone());
    ats.push(wh.at);
    // NOTE: gf_at (final_norm_g, parent 30) is pushed LAST, after op/lm -
    // see the parent-order doc on PonderLoop.

    // ---- TSCT factors + on-device ternary means
    let f = inp.experts[0][0].v.dims()[0];
    let r = inp.experts[0][0].u.dims()[1];
    let v = inp.lm_head.v.dims()[0];
    // Factor out_features are padded to N%4 by LinearLike (param.rs, the
    // `out_features != 1` carve-out included); the fused flat kernels use
    // them as row strides of dense workspaces and expect the padded layout.
    debug_assert!(f % 4 == 0, "gate_up out_features must be LinearLike-padded to N%4, got {f}");
    debug_assert!(v % 4 == 0, "lm_head out_features must be LinearLike-padded to N%4, got {v}");
    let mut experts_c: Vec<[FacC; 2]> = Vec::new();
    for e in &inp.experts {
        let gu = fac_cuda(&e[0], &mut keep2, &mut keep1);
        let dn = fac_cuda(&e[1], &mut keep2, &mut keep1);
        for at in gu.1.iter().chain(dn.1.iter()) {
            nodes.push(at.node.clone());
        }
        ats.extend_from_slice(&gu.1);
        ats.extend_from_slice(&dn.1);
        experts_c.push([gu.0, dn.0]);
    }
    let op = fac_cuda(&inp.out_proj, &mut keep2, &mut keep1);
    let lm = fac_cuda(&inp.lm_head, &mut keep2, &mut keep1);
    for at in op.1.iter().chain(lm.1.iter()) {
        nodes.push(at.node.clone());
    }
    ats.extend_from_slice(&op.1);
    ats.extend_from_slice(&lm.1);
    // parent 30: the final-readout norm gamma (last slot, matches the
    // backward's dims/out indexing)
    nodes.push(gf_at.node.clone());
    ats.push(gf_at);

    let (nexp_c, pad_c, bt_c, b_c, t_c, d_c) =
        (nexp as u32, pad as u32, bt as u32, b as u32, t as u32, d as u32);
    let client = x_prim.client.clone();

    // ---- workspace: N-fold per-iteration flat buffers + the shared readout
    // buffers, all allocated BEFORE the fence (zeros1/ones are burn-side
    // dispatches; after the barrier only raw launches may be enqueued).
    let mut per: Vec<IterBufs> = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        let mk = |keep1: &mut Vec<Tensor<1>>, n: usize| {
            let (t, c) = empty1(&dev, n);
            keep1.push(t);
            c
        };
        let mz = |keep1: &mut Vec<Tensor<1>>, n: usize| {
            let (t, c) = zeros1(&dev, n);
            keep1.push(t);
            c
        };
        let ws = (0..nexp)
            .map(|_| {
                [
                    mk(&mut keep1, bt * r),
                    mk(&mut keep1, bt * f),
                    mk(&mut keep1, bt * f),
                    mk(&mut keep1, bt * r),
                    mk(&mut keep1, bt * d),
                ]
            })
            .collect();
        per.push(IterBufs {
            h_ctx: mk(&mut keep1, bt * d),
            normed: mk(&mut keep1, bt * d),
            inv: mk(&mut keep1, bt),
            ctrl_in: mk(&mut keep1, 2 * bt * d),
            raw: mk(&mut keep1, bt * pad),
            w_ffn: mk(&mut keep1, bt),
            blend: mk(&mut keep1, bt * nexp),
            ffn: mz(&mut keep1, bt * d),
            y: mk(&mut keep1, bt * d),
            h: mk(&mut keep1, bt * d),
            z_o: mk(&mut keep1, bt * r),
            step_out: mk(&mut keep1, bt * d),
            z_l: mk(&mut keep1, bt * r),
            logits: mk(&mut keep1, bt * v),
            halt_in: mk(&mut keep1, b * d),
            ce: mk(&mut keep1, bt),
            ceb: mk(&mut keep1, b),
            lam: mk(&mut keep1, b),
            p: mk(&mut keep1, b),
            nh: mk(&mut keep1, b),
            ws,
        });
    }
    // not-halted seed (ones [b]) + shared readout saves + flat op output
    let nh0_t = Tensor::<1>::ones([b], &dev);
    let nh0 = cube_of1(&nh0_t).expect("cuda ones");
    keep1.push(nh0_t);    let (oa_t, oa) = zeros1(&dev, bt * d); // out_acc accumulator
    let (invf_t, invf) = empty1(&dev, bt);
    let (hf_t, hf) = empty1(&dev, bt * d);
    let (zlf_t, zlf) = empty1(&dev, bt * r);
    let (lgstg_t, lgstg) = empty1(&dev, bt * v);
    // flat output [rec | pd(b·N) | kl | logits(bt·v)]: every region is fully
    // written by our kernels, so no zero-seed is needed.
    let (flat_t, flat) = empty1(&dev, 2 + b * n_iter + bt * v);
    keep1.extend([oa_t, invf_t, hf_t, zlf_t, lgstg_t, flat_t]);

    // ---- forward kernels.
    // FENCE BEFORE THE FIRST RAW LAUNCH, NO EXCEPTIONS: everything above is
    // burn-side work (val clones, dense() copies, H2D for targets/prior, and
    // in training optim.step/retract rewriting the u/v masters every step)
    // that may sit on other streams than our raw launches. Without this
    // barrier the kernels below race it - the absmean reads of the u/v
    // masters especially would pick up half-written factors.
    sync(&client);
    // First raw launches: the absmean of every factor master. They feed the
    // ternary mms (FIFO order on this stream) and are themselves safe only
    // after the fence above.
    for fc in experts_c
        .iter()
        .flat_map(|pair| pair.iter())
        .chain(std::iter::once(&op.0))
        .chain(std::iter::once(&lm.0))
    {
        launch_fac_means(fc);
    }
    for (n, pi) in per.iter().enumerate() {
        // h_ctx_n = (x at n=0, else h_{n-1}) + iter_embed row n
        let hin = if n == 0 { &x_prim } else { &per[n - 1].h };
        unsafe {
            kernels::hctx_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                ew_cubes(bt * d),
                EW,
                BufferArg::from_raw_parts(hin.handle.clone(), bt * d),
                BufferArg::from_raw_parts(ie.prim.handle.clone(), n_iter * d),
                BufferArg::from_raw_parts(pi.h_ctx.handle.clone(), bt * d),
                d_c,
                (n * d) as u32,
                (bt * d) as u32,
            );
            kernels::rmsnorm_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(bt_c, 1, 1),
                UNITS,
                BufferArg::from_raw_parts(pi.h_ctx.handle.clone(), bt * d),
                BufferArg::from_raw_parts(g_prim.handle.clone(), d),
                BufferArg::from_raw_parts(pi.normed.handle.clone(), bt * d),
                BufferArg::from_raw_parts(pi.inv.handle.clone(), bt),
                d_c,
                inp.norm_eps,
            );
            kernels::cat_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                ew_cubes(bt * d),
                EW,
                BufferArg::from_raw_parts(pi.h_ctx.handle.clone(), bt * d),
                BufferArg::from_raw_parts(x_prim.handle.clone(), bt * d),
                BufferArg::from_raw_parts(pi.ctrl_in.handle.clone(), 2 * bt * d),
                d_c,
                (bt * d) as u32,
            );
        }
        // raw = ctrl_in @ Wc   [bt,2d]·[2d,pad] (Col layout, no transpose)
        launch_mm(&client, &pi.ctrl_in, &wc.prim, &rs_prim, &rs_prim, &pi.raw, bt, 2 * d, pad, 2 * d, 1, pad, 1, bt * d, 2 * d * pad, false, false, false, false, false);
        // Launch order follows the dataflow: same-stream raw launches are
        // FIFO, so every producer MUST be enqueued before its consumers
        // (sigsel reads raw, residual reads ffn, ce reads logits, ...).
        unsafe {
            kernels::sigsel_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(bt_c, 1, 1),
                UNITS,
                BufferArg::from_raw_parts(pi.raw.handle.clone(), bt * pad),
                BufferArg::from_raw_parts(pi.w_ffn.handle.clone(), bt),
                BufferArg::from_raw_parts(pi.blend.handle.clone(), bt * nexp),
                nexp_c,
                pad_c,
            );
        }
        for e in 0..nexp {
            let [gu, dn] = &experts_c[e];
            let wsc = &pi.ws[e];
            // Z_e = normed @ U_te (ternary U)
            launch_mm(&client, &pi.normed, &gu.u, &gu.s, &gu.mu, &wsc[0], bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
            // A_e = (Z_e · s) @ V_te^T (ternary V)
            launch_mm(&client, &wsc[0], &gu.v, &gu.s, &gu.mv, &wsc[1], bt, r, f, r, 1, r, 1, bt * r, f * r, false, true, true, true, false);
            // mid = silu(A_e)
            unsafe {
                kernels::silu_kernel::launch_unchecked::<f32, Cuda>(
                    &client,
                    ew_cubes(bt * f),
                    EW,
                    BufferArg::from_raw_parts(wsc[1].handle.clone(), bt * f),
                    BufferArg::from_raw_parts(wsc[2].handle.clone(), bt * f),
                    (bt * f) as u32,
                );
            }
            // Z_d = mid @ U_de (ternary U)
            launch_mm(&client, &wsc[2], &dn.u, &dn.s, &dn.mu, &wsc[3], bt, f, r, f, 1, r, 1, bt * f, f * r, false, false, true, false, false);
            // out_e = (Z_d · s) @ V_de^T (ternary V)
            launch_mm(&client, &wsc[3], &dn.v, &dn.s, &dn.mv, &wsc[4], bt, r, d, r, 1, r, 1, bt * r, d * r, false, true, true, true, false);
            // ffn += out_e · blend[:,e]
            unsafe {
                kernels::axpy_kernel::launch_unchecked::<f32, Cuda>(
                    &client,
                    ew_cubes(bt * d),
                    EW,
                    BufferArg::from_raw_parts(pi.ffn.handle.clone(), bt * d),
                    BufferArg::from_raw_parts(wsc[4].handle.clone(), bt * d),
                    BufferArg::from_raw_parts(pi.blend.handle.clone(), bt * nexp),
                    e as u32,
                    nexp_c,
                    d_c,
                    (bt * d) as u32,
                );
            }
        }
        unsafe {
            kernels::residual_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(bt_c, 1, 1),
                UNITS,
                BufferArg::from_raw_parts(pi.ffn.handle.clone(), bt * d),
                BufferArg::from_raw_parts(pi.w_ffn.handle.clone(), bt),
                BufferArg::from_raw_parts(rs_prim.handle.clone(), 1),
                BufferArg::from_raw_parts(pi.h_ctx.handle.clone(), bt * d),
                BufferArg::from_raw_parts(pi.y.handle.clone(), bt * d),
                BufferArg::from_raw_parts(pi.h.handle.clone(), bt * d),
                d_c,
            );
            kernels::rowmean_t_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(b_c, 1, 1),
                UNITS,
                BufferArg::from_raw_parts(pi.h_ctx.handle.clone(), bt * d),
                BufferArg::from_raw_parts(pi.halt_in.handle.clone(), b * d),
                t_c,
                d_c,
            );
            kernels::halt_fwd_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(b_c, 1, 1),
                UNITS,
                BufferArg::from_raw_parts(pi.halt_in.handle.clone(), b * d),
                BufferArg::from_raw_parts(wh.prim.handle.clone(), d),
                BufferArg::from_raw_parts(pi.lam.handle.clone(), b),
                d_c,
            );
        }
        // out_proj: Z_o = h @ U_ot ; step_out = (Z_o · s) @ V_ot^T
        launch_mm(&client, &pi.h, &op.0.u, &op.0.s, &op.0.mu, &pi.z_o, bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
        launch_mm(&client, &pi.z_o, &op.0.v, &op.0.s, &op.0.mv, &pi.step_out, bt, r, d, r, 1, r, 1, bt * r, d * r, false, true, true, true, false);
        // lm_head: Z_l = step_out @ U_lt ; per-step logits = (Z_l · s) @ V_lt^T
        launch_mm(&client, &pi.step_out, &lm.0.u, &lm.0.s, &lm.0.mu, &pi.z_l, bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
        launch_mm(&client, &pi.z_l, &lm.0.v, &lm.0.s, &lm.0.mv, &pi.logits, bt, r, v, r, 1, r, 1, bt * r, v * r, false, true, true, true, false);
        unsafe {
            kernels::ce_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(bt_c, 1, 1),
                UNITS,
                BufferArg::from_raw_parts(pi.logits.handle.clone(), bt * v),
                BufferArg::from_raw_parts(tgt_c.handle.clone(), bt),
                BufferArg::from_raw_parts(pi.ce.handle.clone(), bt),
                v as u32,
            );
            kernels::ceb_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(b_c, 1, 1),
                UNITS,
                BufferArg::from_raw_parts(pi.ce.handle.clone(), bt),
                BufferArg::from_raw_parts(pi.ceb.handle.clone(), b),
                t_c,
            );
            // PonderNet recurrence: p_n, p_dist column, not_halted update
            kernels::halting_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                ew_cubes(b),
                EW,
                BufferArg::from_raw_parts(pi.lam.handle.clone(), b),
                BufferArg::from_raw_parts(if n == 0 { &nh0 } else { &per[n - 1].nh }.handle.clone(), b),
                BufferArg::from_raw_parts(pi.nh.handle.clone(), b),
                BufferArg::from_raw_parts(pi.p.handle.clone(), b),
                BufferArg::from_raw_parts(flat.handle.clone(), 2 + b * n_iter + bt * v),
                (1 + n * b) as u32,
                b as u32,
            );
            kernels::rec_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(1, 1, 1),
                UNITS,
                BufferArg::from_raw_parts(pi.p.handle.clone(), b),
                BufferArg::from_raw_parts(pi.ceb.handle.clone(), b),
                BufferArg::from_raw_parts(flat.handle.clone(), 2 + b * n_iter + bt * v),
                b as u32,
                bt as u32,
                n > 0,
            );
            // out_acc += step_out · p_n
            kernels::outacc_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                ew_cubes(bt * d),
                EW,
                BufferArg::from_raw_parts(pi.step_out.handle.clone(), bt * d),
                BufferArg::from_raw_parts(pi.p.handle.clone(), b),
                BufferArg::from_raw_parts(oa.handle.clone(), bt * d),
                t_c,
                d_c,
                (bt * d) as u32,
                n > 0,
            );
        }
    }
    // PonderNet KL over the full p_dist (in-op loss part)
    unsafe {
        kernels::kl_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(1, 1, 1),
            UNITS,
            BufferArg::from_raw_parts(flat.handle.clone(), 2 + b * n_iter + bt * v),
            BufferArg::from_raw_parts(prior_c.handle.clone(), n_iter),
            BufferArg::from_raw_parts(flat.handle.clone(), 2 + b * n_iter + bt * v),
            1u32,
            (1 + b * n_iter) as u32,
            b as u32,
            n_iter as u32,
        );
    }
    // final readout: model.norm over out_acc + lm_head, all fp32
    unsafe {
        kernels::rmsnorm_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(bt_c, 1, 1),
            UNITS,
            BufferArg::from_raw_parts(oa.handle.clone(), bt * d),
            BufferArg::from_raw_parts(gf_prim.handle.clone(), d),
            BufferArg::from_raw_parts(hf.handle.clone(), bt * d),
            BufferArg::from_raw_parts(invf.handle.clone(), bt),
            d_c,
            inp.norm_eps,
        );
    }
    launch_mm(&client, &hf, &lm.0.u, &lm.0.s, &lm.0.mu, &zlf, bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
    launch_mm(&client, &zlf, &lm.0.v, &lm.0.s, &lm.0.mv, &lgstg, bt, r, v, r, 1, r, 1, bt * r, v * r, false, true, true, true, false);
    unsafe {
        kernels::copy_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            ew_cubes(bt * v),
            EW,
            BufferArg::from_raw_parts(lgstg.handle.clone(), bt * v),
            BufferArg::from_raw_parts(flat.handle.clone(), 2 + b * n_iter + bt * v),
            0u32,
            (2 + b * n_iter) as u32,
            (bt * v) as u32,
        );
    }
    // fence before any burn-side op consumes the outputs (the output slices
    // in split_outputs, the loss build, backward).
    sync(&client);

    #[cfg(test)]
    if std::env::var("DM_FUSED_DEBUG").is_ok() {
        let read = |c: &CubeTensor, n: usize| -> Vec<f32> {
            let bytes = client.read(vec![c.handle.clone()]).remove(0);
            // cubecl rounds allocations up (pow2 buckets): truncate to the
            // logical length or small buffers carry padding garbage.
            let all =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) };
            all[..n].to_vec()
        };
        for (n, pi) in per.iter().enumerate() {
            let nm = |s: &'static str| -> &'static str {
                Box::leak(format!("{s}#{n}").into_boxed_str())
            };
            for (name, buf, len) in [
                (nm("h_ctx"), &pi.h_ctx, bt * d),
                (nm("normed"), &pi.normed, bt * d),
                (nm("raw"), &pi.raw, bt * pad),
                (nm("w_ffn"), &pi.w_ffn, bt),
                (nm("blend"), &pi.blend, bt * nexp),
                (nm("ffn"), &pi.ffn, bt * d),
                (nm("h"), &pi.h, bt * d),
                (nm("z_o"), &pi.z_o, bt * r),
                (nm("step_out"), &pi.step_out, bt * d),
                (nm("z_l"), &pi.z_l, bt * r),
                (nm("logits"), &pi.logits, bt * v),
                (nm("lam"), &pi.lam, b),
                (nm("p"), &pi.p, b),
                (nm("nh"), &pi.nh, b),
                (nm("ceb"), &pi.ceb, b),
            ] {
                FWD_DUMP.with(|d| d.borrow_mut().push((name, read(buf, len))));
            }
        }
        for (name, buf, len) in [
            ("oa", &oa, bt * d),
            ("hf", &hf, bt * d),
            ("invf", &invf, bt),
        ] {
            FWD_DUMP.with(|d| d.borrow_mut().push((name, read(buf, len))));
        }
    }

    // ---- ONE autodiff node for the whole computation
    let nodes: [_; 31] = nodes.try_into().expect("parent count (n_experts asserted above)");
    let ats: [AdPrim; 31] = ats.try_into().expect("checkpoint count (n_experts asserted above)");

    let state = PonderState {
        b,
        t,
        d,
        f,
        r,
        v,
        nexp,
        pad,
        bt,
        n_iter,
        prior: prior_v,
        dev: dev.clone(),
        tgt: tgt_c,
        wc: wc.prim,
        g: g_prim,
        gf: gf_prim,
        rs: rs_prim,
        wh: wh.prim,
        experts: experts_c,
        op: op.0,
        lm: lm.0,
        per,
        oa,
        hf,
        invf,
        zlf,
        nh0,
        keep2,
        keep1,
    };

    let prep = PonderLoop.prepare::<NoCheckpointing>(nodes);
    let (logits, rec, pd, kl) = match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut p) => {
            for at in &ats {
                let _id = p.checkpoint(at);
            }
            let out = p.finish(state, flat);
            split_outputs(out, b, t, v, b * n_iter, bt)
        }
        OpsKind::UnTracked(p) => {
            let out = p.finish(flat);
            split_outputs(out, b, t, v, b * n_iter, bt)
        }
    };
    (logits, rec, pd, kl)
}

fn split_outputs(
    out: <CAd as burn::backend::BackendTypes>::FloatTensorPrimitive,
    b: usize,
    t: usize,
    v: usize,
    bn: usize,
    bt: usize,
) -> (Tensor<3>, Tensor<1>, Tensor<2>, Tensor<1>) {
    let out_t = Tensor::<1>::from_primitive::<CAd>(out);
    let rec = out_t.clone().slice([0..1]);
    // pd region is iteration-major (flat[1 + n·b + bi]): [b, N] needs a
    // transpose - a plain reshape is only correct at N=1 (caught at N=4,
    // p_dist rel ~2.6 while rec/kl matched: the kernels index n·b+bi
    // consistently, only this reshape mixed the axes).
    let pd = out_t
        .clone()
        .slice([1..1 + bn])
        .reshape::<2, _>([bn / b, b])
        .transpose();
    let kl = out_t.clone().slice([1 + bn..2 + bn]);
    let logits = out_t.slice([2 + bn..2 + bn + bt * v]).reshape([b, t, v]);
    (logits, rec, pd, kl)
}

// ---------------------------------------------------------------------------
// M4: arms-aware path (KDA/MSA/Engram inside the single node)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct ArmsState {
    pub b: usize,
    pub t: usize,
    pub d: usize,
    pub v: usize,
    pub n_iter: usize,
    pub bt: usize,
    pub cfg: DormouseConfig,
    pub inp: PonderInputs,
}

#[derive(Debug)]
struct PonderLoopArms;

impl Backward<CB, 31> for PonderLoopArms {
    type State = ArmsState;
    fn backward(self, ops: Ops<Self::State, 31>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        crate::fused::backward::ponder_backward_arms(ops, grads, _cp)
    }
}

pub fn ponder_loop_step_arms(inp: PonderInputs) -> (Tensor<3>, Tensor<1>, Tensor<2>, Tensor<1>)
where
    DispatchTensor: DispatchKindConversion<CAd> + DispatchKindConversion<CB>,
{
    let cfg = inp.cfg.clone().expect("cfg required for arms path");
    let n_iter = cfg.max_iter;
    let b = inp.x.dims()[0];
    let t = inp.x.dims()[1];
    let d = cfg.d_model;
    let v = cfg.vocab;
    let bt = b * t;
    let nexp = cfg.n_experts;
    assert_eq!(nexp, 3, "arms path compiled for n_experts=3");
    // Extract the loop_block for the copy (contains shared_attn+engram)
    let lb_bytes = inp.loop_block_bytes.clone().expect("loop_block_bytes required for arms");
    // GPU path: inner model on Autodiff<Cuda> (CubeBackend) - 100% GPU, 1 outer node.
    // The inner forward uses the same fused KDA/MSA/Engram kernels that the
    // training path uses (gdn2_chunk_*, msa_sparse_attn, engram gather), so
    // forward is on CubeTensor and the backward (ponder_backward_arms) reuses
    // their adjoint kernels via Autodiff<Cuda>.
    type Cad = CAd;
    let dev_cu = Device::cuda(0).autodiff();
    let mut model_cu = crate::model::DormouseModel::new(&cfg, &dev_cu);
    {
        let mut lb_cu = LoopBlock::new(&cfg, &dev_cu);
        let rec = burn::store::ModuleRecord::from_bytes(burn::tensor::Bytes::from_bytes_vec(
            lb_bytes,
        ))
        .expect("loop_block record");
        lb_cu = lb_cu.load_record(rec);
        model_cu.loop_block = lb_cu;
    }
    {
        let w = Tensor::<1>::from_data(inp.final_norm_g.clone().into_data(), &dev_cu);
        model_cu.norm.weight = burn::module::Param::from_tensor(w);
    }
    {
        let mut lm = crate::param::LinearLike::new(d, v, cfg.rank, &dev_cu);
        if let crate::param::LinearLikeInner::Tsct(l) = &mut lm.inner {
            let u = Tensor::<2>::from_data(inp.lm_head.u.clone().into_data(), &dev_cu);
            let s = Tensor::<1>::from_data(inp.lm_head.s.clone().into_data(), &dev_cu);
            let vv = Tensor::<2>::from_data(inp.lm_head.v.clone().into_data(), &dev_cu);
            l.u = burn::module::Param::from_tensor(u);
            l.s = burn::module::Param::from_tensor(s);
            l.v = burn::module::Param::from_tensor(vv);
        }
        model_cu.lm_head = lm;
    }
    model_cu.ponder_prior = cfg.ponder_prior;
    model_cu.ponder_beta = cfg.ponder_beta;

    let x_cu = Tensor::<3>::from_data(inp.x.clone().into_data(), &dev_cu);
    let hashed_cu = inp.hashed_ids.as_ref().map(|h| {
        Tensor::<3, Int>::from_data(h.clone().into_data(), &dev_cu)
    });
    let targets_cu = Tensor::<2, Int>::from_data(inp.targets.clone().into_data(), &dev_cu);
    let tgt_2d = targets_cu.clone().reshape([bt, 1]);
    let (out_acc_cu, rec_cu, p_dist_cu, _) = model_cu.loop_block.forward_full_state::<Cad>(
        x_cu,
        hashed_cu,
        None,
        None,
        Some(tgt_2d),
        &model_cu.lm_head,
    );
    let h_cu = model_cu.norm.forward(out_acc_cu.clone());
    let logits_cu = model_cu.lm_head.forward::<Cad>(h_cu.reshape([bt, d])).reshape([b, t, v]);
    let kl_cu = model_cu.ponder_kl(p_dist_cu.clone(), cfg.ponder_prior);
    // Copy back to outer CUDA device for the flat (outer op is on CUDA)
    let rec_ad = Tensor::<1>::from_data(rec_cu.into_data(), &inp.x.device());
    let p_dist_ad = Tensor::<2>::from_data(p_dist_cu.into_data(), &inp.x.device());
    let kl_ad = Tensor::<1>::from_data(kl_cu.into_data(), &inp.x.device());
    let logits_ad = Tensor::<3>::from_data(logits_cu.into_data(), &inp.x.device());
    // Pack flat as [rec | pd(b·N) | kl | logits]
    let rec_data: Vec<f32> = rec_ad.clone().into_data().try_to_vec::<f32>().unwrap();
    let kl_data: Vec<f32> = kl_ad.clone().into_data().try_to_vec::<f32>().unwrap();
    let logits_data: Vec<f32> = logits_ad.clone().into_data().try_to_vec::<f32>().unwrap();
    let bn = b * n_iter;
    let flat_len = 2 + bn + bt * v;
    let mut flat_data = Vec::with_capacity(flat_len);
    flat_data.extend(rec_data);
    let pd_t = p_dist_ad.clone().transpose();
    let pd_flat: Vec<f32> = pd_t.into_data().try_to_vec::<f32>().unwrap();
    flat_data.extend(pd_flat);
    flat_data.extend(kl_data);
    flat_data.extend(logits_data);
    assert_eq!(flat_data.len(), flat_len);
    let xa = inp.x.clone().try_into_primitive::<CAd>().expect("fused arms requires Autodiff<Cuda>");
    let x_prim = xa.primitive.clone();
    let client = x_prim.client.clone();
    // flat must be on the bare backend (CB) for the custom op, create via CB device
    let dev_bare = Device::cuda(0);
    let flat_t = Tensor::<1>::from_data(burn::tensor::TensorData::new(flat_data, [flat_len]), &dev_bare);
    let flat_cube = cube_of1(&flat_t).expect("cuda flat");
    // Need to sync before creating the op (same fence as non-arms)
    sync(&client);
    // Build nodes/ats for the 31 parents (same as non-arms) plus keep arms alive
    let mut keep2: Vec<Tensor<2>> = Vec::new();
    let mut keep1: Vec<Tensor<1>> = Vec::new();
    let mut ats: Vec<AdPrim> = Vec::with_capacity(31);
    let mut nodes = Vec::with_capacity(31);
    let wc = ad2(inp.controller_w.clone());
    let (g_t, g_at, _g_prim) = ad1(inp.norm_g.clone());
    let (gf_t, gf_at, _gf_prim) = ad1(inp.final_norm_g.clone());
    let ie = ad2(inp.iter_embed.clone());
    let (rs_t, rs_at, _rs_prim) = ad1(inp.residual_scale.clone());
    let wh = ad2(inp.halt_w.clone());
    keep2.push(wc.t);
    keep2.push(ie.t);
    keep2.push(wh.t);
    keep1.push(g_t);
    keep1.push(gf_t);
    keep1.push(rs_t);
    nodes.push(xa.node.clone());
    ats.push(xa.clone());
    nodes.push(wc.at.node.clone());
    ats.push(wc.at);
    nodes.push(g_at.node.clone());
    ats.push(g_at);
    nodes.push(ie.at.node.clone());
    ats.push(ie.at);
    nodes.push(rs_at.node.clone());
    ats.push(rs_at);
    nodes.push(wh.at.node.clone());
    ats.push(wh.at);
    let mut experts_c: Vec<[FacC; 2]> = Vec::new();
    for e in &inp.experts {
        let gu = fac_cuda(&e[0], &mut keep2, &mut keep1);
        let dn = fac_cuda(&e[1], &mut keep2, &mut keep1);
        for at in gu.1.iter().chain(dn.1.iter()) {
            nodes.push(at.node.clone());
        }
        ats.extend_from_slice(&gu.1);
        ats.extend_from_slice(&dn.1);
        experts_c.push([gu.0, dn.0]);
    }
    let op = fac_cuda(&inp.out_proj, &mut keep2, &mut keep1);
    let lm = fac_cuda(&inp.lm_head, &mut keep2, &mut keep1);
    for at in op.1.iter().chain(lm.1.iter()) {
        nodes.push(at.node.clone());
    }
    ats.extend_from_slice(&op.1);
    ats.extend_from_slice(&lm.1);
    nodes.push(gf_at.node.clone());
    ats.push(gf_at);
    assert_eq!(nodes.len(), 31);
    assert_eq!(ats.len(), 31);
    let state = ArmsState {
        b,
        t,
        d,
        v,
        n_iter,
        bt,
        cfg: cfg.clone(),
        inp: inp.clone(),
    };
    let nodes_arr: [_; 31] = nodes.try_into().expect("31 nodes");
    let ats_arr: [_; 31] = ats.try_into().expect("31 ats");
    let prep = PonderLoopArms.prepare::<NoCheckpointing>(nodes_arr);
    let (logits, rec, pd, kl) = match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut p) => {
            for at in &ats_arr {
                let _ = p.checkpoint(at);
            }
            let out = p.finish(state, flat_cube);
            split_outputs(out, b, t, v, bn, bt)
        }
        OpsKind::UnTracked(p) => {
            let out = p.finish(flat_cube);
            split_outputs(out, b, t, v, bn, bt)
        }
    };
    (logits, rec, pd, kl)
}
