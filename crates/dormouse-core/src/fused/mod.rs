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
use burn::tensor::{Device, Int, Tensor, TensorData};
use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use backward::PonderState;
use crate::param::{LinearLike, LinearLikeInner};
use crate::loop_block::LoopBlock;
use crate::config::DormouseConfig;
use burn::module::{Module, ModuleVisitor, Param};

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
/// For large matmuls (bt >= 512, m*n >= 32k) the naive per-element kernel
/// is memory-latency bound; the tiled kernel uses shared memory and ~2-3x
/// faster while staying bit-identical (same serial k order, same ternary
/// handling). Small mats stay on the naive path (lower launch overhead).
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
    // Threshold: large bt (training s512) or large output tile.
    let use_tiled = m * n >= 32768 && k >= 16;
    if use_tiled {
        const TILE: usize = 16;
        let gx = m.div_ceil(TILE);
        let gy = n.div_ceil(TILE);
        unsafe {
            kernels::mm_tiled_kernel::launch_unchecked::<f32, Cuda>(
                client,
                CubeCount::Static(gx as u32, gy as u32, 1),
                CubeDim::new_3d(256, 1, 1),
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
    } else {
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

/// Autodiff node ids of every float param of one module subtree, in burn's
/// module traversal order. Captured from the LIVE model at op-input build
/// time; the arms inner-graph backward walks the identically-structured AD
/// clone it rebuilds from `loop_block_bytes`, pulls each param's grad from
/// the inner map positionally, and registers it onto these ids so the outer
/// map (and thereby the optimizer) sees exact arm gradients.
#[derive(Clone, Debug, Default)]
pub struct ArmLeaves {
    pub ids: Vec<burn::backend::autodiff::NodeId>,
}

impl ArmLeaves {
    /// Capture the param node ids under `m` (shared_attn, engram).
    pub fn capture<M: Module>(m: &M) -> Self {
        struct Cap {
            out: Vec<burn::backend::autodiff::NodeId>,
        }
        impl ModuleVisitor for Cap {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                let at = param
                    .val()
                    .try_into_primitive::<CAd>()
                    .expect("fused arm capture requires CUDA autodiff tensors");
                if std::env::var("DM_FUSED_DEBUG").is_ok() {
                    eprintln!("arm cap: id={:?} dims={:?}", at.node.id, param.val().dims());
                }
                self.out.push(at.node.id);
            }
        }
        let mut cap = Cap { out: Vec::new() };
        m.visit(&mut cap);
        Self { ids: cap.out }
    }
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
    /// RAM-offload Engram rows, already gathered+expanded `[b, t, 3·dim]`
    /// (the autodiff leaf the trainer's host tables gathered through). When
    /// Some it takes priority over `hashed_ids`, exactly like
    /// `forward_with_hidden`; its gradient is registered back onto this
    /// tensor's node so the burn-side gather chain trains `rows_param`.
    pub host_rows: Option<Tensor<3>>,
    /// Captured live arm weights (shared_attn + engram subtrees) for the
    /// inner-graph arms backward; None skips arm-gradient computation (the
    /// op still runs its raw arms forward).
    pub arm_leaves: Option<ArmLeavesPair>,
    pub loop_block_bytes: Option<Vec<u8>>,
    pub cfg: Option<crate::config::DormouseConfig>,
}

/// The two arm subtrees whose weights the arms forward reads outside the
/// parent list: `shared_attn` (KDA + MSA + router) and `engram`.
#[derive(Clone, Debug, Default)]
pub struct ArmLeavesPair {
    pub attn: ArmLeaves,
    pub engram: ArmLeaves,
}

/// The op's outputs: the `forward_with_hidden` triple plus the in-op PonderNet
/// KL scalar and the latents the burn-path aux heads consume (`out_acc` for
/// JEPA/KoLeo, `h` — the final-norm output — for DSpark).
pub struct PonderOutputs {
    pub logits: Tensor<3>,
    pub rec: Tensor<1>,
    pub p_dist: Tensor<2>,
    pub kl: Tensor<1>,
    pub out_acc: Tensor<3>,
    pub h: Tensor<3>,
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
/// 24 out_proj u/s/v | 27 lm_head u/s/v | 30 final_norm_g (nexp=3).
/// For nexp=8 the arity is 61: 6+48+3+3+1.
#[derive(Debug)]
struct PonderLoop;

impl Backward<CB, 61> for PonderLoop {
    type State = PonderState;

    fn backward(self, ops: Ops<Self::State, 61>, grads: &mut Gradients, cp: &mut Checkpointer) {
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
pub fn ponder_loop_step(inp: PonderInputs) -> PonderOutputs
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
    // Generic arity 61 covers nexp up to 8 (swift50): 13+6*8=61. For nexp=3 the
    // extra slots are padded with None/dummy and ignored in backward.
    assert!(nexp == 3 || nexp == 8, "fused op supports n_experts 3 or 8, got {nexp}");
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
    let mut ats: Vec<AdPrim> = Vec::with_capacity(61);
    let mut nodes = Vec::with_capacity(61);
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
    // NOTE: gf_at (final_norm_g, parent 60 for nexp=8, 30 for nexp=3) is pushed LAST

    // ---- TSCT factors + on-device ternary means
    let f = inp.experts[0][0].v.dims()[0];
    let r = inp.experts[0][0].u.dims()[1];
    let v = inp.lm_head.v.dims()[0];
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
    nodes.push(gf_at.node.clone());
    ats.push(gf_at);
    // pad to 61 for generic Backward<61> (nexp=3 needs 30 dummies) - use real nodes
    let dummy_node = nodes[0].clone();
    let dummy_at2 = ats[0].clone();
    while nodes.len() < 61 {
        nodes.push(dummy_node.clone());
    }
    while ats.len() < 61 {
        ats.push(dummy_at2.clone());
    }

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
    // flat output [rec | pd(b·N) | kl | logits(bt·v) | oa(bt·d) | hf(bt·d)]:
    // every region is fully written by our kernels, so no zero-seed is needed.
    // oa/hf expose the latents the burn-path aux heads consume (JEPA/KoLeo on
    // out_acc, DSpark on the final-norm output).
    let off_oa = 2 + b * n_iter + bt * v;
    let (flat_t, flat) = empty1(&dev, off_oa + 2 * bt * d);
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
        // latents for the burn-path aux heads: out_acc + final-norm output
        kernels::copy_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            ew_cubes(bt * d),
            EW,
            BufferArg::from_raw_parts(oa.handle.clone(), bt * d),
            BufferArg::from_raw_parts(flat.handle.clone(), off_oa),
            0u32,
            off_oa as u32,
            (bt * d) as u32,
        );
        kernels::copy_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            ew_cubes(bt * d),
            EW,
            BufferArg::from_raw_parts(hf.handle.clone(), bt * d),
            BufferArg::from_raw_parts(flat.handle.clone(), off_oa + bt * d),
            0u32,
            (off_oa + bt * d) as u32,
            (bt * d) as u32,
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

    // ---- ONE autodiff node for the whole computation (61 covers up to 8 experts)
    let nodes: [_; 61] = nodes.try_into().expect("61 parents");
    let ats: [AdPrim; 61] = ats.try_into().expect("61 checkpoints");

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
    match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut p) => {
            for at in &ats {
                let _id = p.checkpoint(at);
            }
            let out = p.finish(state, flat);
            split_outputs(out, b, t, v, b * n_iter, bt, d)
        }
        OpsKind::UnTracked(p) => {
            let out = p.finish(flat);
            split_outputs(out, b, t, v, b * n_iter, bt, d)
        }
    }
}

fn split_outputs(
    out: <CAd as burn::backend::BackendTypes>::FloatTensorPrimitive,
    b: usize,
    t: usize,
    v: usize,
    bn: usize,
    bt: usize,
    d: usize,
) -> PonderOutputs {
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
    let off_oa = 2 + bn + bt * v;
    let logits = out_t
        .clone()
        .slice([2 + bn..2 + bn + bt * v])
        .reshape([b, t, v]);
    let out_acc = out_t
        .clone()
        .slice([off_oa..off_oa + bt * d])
        .reshape([b, t, d]);
    let h = out_t.slice([off_oa + bt * d..off_oa + 2 * bt * d]).reshape([b, t, d]);
    PonderOutputs {
        logits,
        rec,
        p_dist: pd,
        kl,
        out_acc,
        h,
    }
}

// ---------------------------------------------------------------------------
// M4: arms-aware path (KDA/MSA/Engram inside the single node)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct ArmsDirectState {
    pub base: PonderState,
    pub cfg: DormouseConfig,
    pub w_attn: Vec<CubeTensor>,
    pub w_mem: Vec<CubeTensor>,
    pub kda_out: Vec<CubeTensor>,
    pub msa_out: Vec<CubeTensor>,
    pub engram_out: Vec<CubeTensor>,
    pub attn: Vec<CubeTensor>,
    pub gate: Vec<CubeTensor>,
    pub hashed: Option<CubeTensor>,
    /// RAM-offload rows ([bt, 96] bare primitive) when the engram arm ran in
    /// host-rows mode; its grad comes from the inner graph (parent 61).
    pub rows: Option<CubeTensor>,
    /// Captured live arm weights for the inner-graph backward.
    pub arm: Option<ArmLeavesPair>,
    pub lb_bytes: Vec<u8>,
    pub x: CubeTensor,
}

#[derive(Debug)]
struct PonderLoopArms;

impl Backward<CB, 62> for PonderLoopArms {
    type State = ArmsDirectState;
    fn backward(self, ops: Ops<Self::State, 62>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        crate::fused::backward::ponder_backward_arms_direct(ops, grads, _cp)
    }
}



pub fn ponder_loop_step_arms(inp: PonderInputs) -> PonderOutputs
where
    DispatchTensor: DispatchKindConversion<CAd> + DispatchKindConversion<CB>,
{
    // DIRECT path: no inner Autodiff graph, 1 outer node, fence before first raw launch,
    // 1D workspaces, CubeTensor handles for KDA (gdn2_chunk), MSA (sparse), Engram (gather).
    let cfg = inp.cfg.clone().expect("cfg required for arms path");
    let n_iter = cfg.max_iter;
    let b = inp.x.dims()[0];
    let t = inp.x.dims()[1];
    let d = cfg.d_model;
    let v = cfg.vocab;
    let bt = b * t;
    let nexp = cfg.n_experts;
    assert!(nexp == 3 || nexp == 8, "arms path supports n_experts 3 or 8, got {nexp}");
    let lb_bytes = inp.loop_block_bytes.clone().expect("loop_block_bytes required for arms");
    let dev_bare = Device::cuda(0);
    let mut lb_bare = LoopBlock::new(&cfg, &dev_bare);
    {
        let rec = burn::store::ModuleRecord::from_bytes(burn::tensor::Bytes::from_bytes_vec(
            lb_bytes.clone(),
        ))
        .expect("loop_block record");
        lb_bare = lb_bare.load_record(rec);
    }
    let [two_d, pad] = dense(inp.controller_w.clone()).dims();
    assert_eq!(two_d, 2 * d, "controller weight must be [2d, pad]");
    // prior
    let mut prior: Vec<f32> = Vec::with_capacity(n_iter);
    let mut mass = 1.0f32;
    let mut total = 0.0f32;
    for _ in 0..n_iter {
        let prob = cfg.ponder_prior * mass;
        prior.push(prob);
        total += prob;
        mass *= 1.0 - cfg.ponder_prior;
    }
    let inv = 1.0 / total;
    let prior_v: Vec<f32> = prior.iter().map(|x| x * inv).collect();
    let x2 = inp.x.clone().reshape([bt, d]);
    let xa = x2.clone().try_into_primitive::<CAd>().expect("fused arms requires Autodiff<Cuda>");
    let x_prim = xa.primitive.clone();
    let x_t = Tensor::<2>::from_primitive::<CB>(x_prim.clone());
    let dev = x_t.device();
    let prior_t = Tensor::<1>::from_data(burn::tensor::TensorData::new(prior_v.clone(), [n_iter]), &dev);
    let prior_c = cube_of1(&dense(prior_t)).expect("cuda prior");
    let tgt_c = cube_int2(&inp.targets).expect("cuda targets");
    let mut keep2: Vec<Tensor<2>> = Vec::new();
    let mut keep1: Vec<Tensor<1>> = Vec::new();
    let mut ats: Vec<AdPrim> = Vec::with_capacity(61);
    let mut nodes = Vec::with_capacity(61);
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
    let f = inp.experts[0][0].v.dims()[0];
    let r = inp.experts[0][0].u.dims()[1];
    debug_assert!(f % 4 == 0);
    debug_assert!(v % 4 == 0);
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
    // parent 61: the RAM-offload rows leaf ([bt, 96], dense-forced). Its grad
    // is produced by the arms inner graph and registered here so the burn-side
    // gather chain (rows_param -> embed) trains the host tables.
    const ROW_DIM: usize = 96; // 3 tables x 32 (offload::HostNgram dim)
    let rows_ad = inp
        .host_rows
        .as_ref()
        .map(|r| ad2(r.clone().reshape([bt, ROW_DIM])));
    if let Some(ra) = &rows_ad {
        nodes.push(ra.at.node.clone());
        ats.push(ra.at.clone());
        keep2.push(ra.t.clone());
    }
    // pad to 62 for generic Backward<62> (nexp=3 needs 30 dummies)
    let dummy_node = nodes[0].clone();
    let dummy_at = ats[0].clone();
    while nodes.len() < 62 {
        nodes.push(dummy_node.clone());
    }
    while ats.len() < 62 {
        ats.push(dummy_at.clone());
    }
    let (nexp_c, pad_c, bt_c, b_c, t_c, d_c) = (nexp as u32, pad as u32, bt as u32, b as u32, t as u32, d as u32);
    let client = x_prim.client.clone();
    // RAM-offload rows: the bare primitive of the [bt, 96] embed, consumed by
    // the raw engram forward below (the burn gather chain stays outside).
    let rows_cube: Option<CubeTensor> = rows_ad.as_ref().map(|ra| ra.prim.clone());
    // hashed cube for engram (small, host copy is fine)
    let hashed_cube: Option<CubeTensor> = inp.hashed_ids.as_ref().map(|h| {
        let dev = Device::cuda(0);
        let h_bare = Tensor::<3, Int>::from_data(h.clone().into_data(), &dev);
        let dims = h_bare.dims();
        let n = dims.iter().product::<usize>();
        let flat = h_bare.reshape::<1, _>([n]).reshape::<3, _>(dims);
        flat.try_into_primitive::<CB>().expect("hashed cube")
    });
    // per-iteration base bufs + arms bufs (all 1D, flat)
    let mut per: Vec<IterBufs> = Vec::with_capacity(n_iter);
    let mut w_attn_vec: Vec<CubeTensor> = Vec::with_capacity(n_iter);
    let mut w_mem_vec: Vec<CubeTensor> = Vec::with_capacity(n_iter);
    let mut kda_vec: Vec<CubeTensor> = Vec::with_capacity(n_iter);
    let mut msa_vec: Vec<CubeTensor> = Vec::with_capacity(n_iter);
    let mut engram_vec: Vec<CubeTensor> = Vec::with_capacity(n_iter);
    let mut attn_vec: Vec<CubeTensor> = Vec::with_capacity(n_iter);
    let mut gate_vec: Vec<CubeTensor> = Vec::with_capacity(n_iter);
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
        let ws = (0..nexp).map(|_| [mk(&mut keep1, bt * r), mk(&mut keep1, bt * f), mk(&mut keep1, bt * f), mk(&mut keep1, bt * r), mk(&mut keep1, bt * d)]).collect();
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
        let (t1, c1) = empty1(&dev, bt);
        keep1.push(t1);
        w_attn_vec.push(c1);
        let (t2, c2) = empty1(&dev, bt);
        keep1.push(t2);
        w_mem_vec.push(c2);
        let (t3, c3) = empty1(&dev, bt * d);
        keep1.push(t3);
        kda_vec.push(c3);
        let (t4, c4) = empty1(&dev, bt * d);
        keep1.push(t4);
        msa_vec.push(c4);
        let (t5, c5) = empty1(&dev, bt * d);
        keep1.push(t5);
        engram_vec.push(c5);
        let (t6, c6) = empty1(&dev, bt * d);
        keep1.push(t6);
        attn_vec.push(c6);
        let (t7, c7) = empty1(&dev, bt);
        keep1.push(t7);
        gate_vec.push(c7);
    }
    let nh0_t = Tensor::<1>::ones([b], &dev);
    let nh0 = cube_of1(&nh0_t).expect("cuda ones");
    keep1.push(nh0_t);
    let (oa_t, oa) = zeros1(&dev, bt * d);
    let (invf_t, invf) = empty1(&dev, bt);
    let (hf_t, hf) = empty1(&dev, bt * d);
    let (zlf_t, zlf) = empty1(&dev, bt * r);
    let (lgstg_t, lgstg) = empty1(&dev, bt * v);
    let off_oa = 2 + b * n_iter + bt * v;
    let (flat_t, flat) = empty1(&dev, off_oa + 2 * bt * d);
    keep1.extend([oa_t, invf_t, hf_t, zlf_t, lgstg_t, flat_t]);
    // FENCE BEFORE FIRST RAW LAUNCH
    sync(&client);
    for fc in experts_c.iter().flat_map(|p| p.iter()).chain(std::iter::once(&op.0)).chain(std::iter::once(&lm.0)) {
        crate::fused::kernels::arms::launch_gdn2_chunk_dummy(&client);
        let n_u = fc.u.meta.shape().dims::<2>().iter().product::<usize>();
        let n_v = fc.v.meta.shape().dims::<2>().iter().product::<usize>();
        unsafe {
            crate::fused::kernels::absmean_kernel::launch_unchecked::<f32, Cuda>(&fc.u.client, CubeCount::Static(1,1,1), UNITS, BufferArg::from_raw_parts(fc.u.handle.clone(), n_u), BufferArg::from_raw_parts(fc.mu.handle.clone(), 1), n_u as u32);
            crate::fused::kernels::absmean_kernel::launch_unchecked::<f32, Cuda>(&fc.v.client, CubeCount::Static(1,1,1), UNITS, BufferArg::from_raw_parts(fc.v.handle.clone(), n_v), BufferArg::from_raw_parts(fc.mv.handle.clone(), 1), n_v as u32);
        }
    }
    for (n, pi) in per.iter().enumerate() {
        let hin = if n == 0 { &x_prim } else { &per[n-1].h };
        unsafe {
            kernels::hctx_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(hin.handle.clone(), bt*d), BufferArg::from_raw_parts(ie.prim.handle.clone(), n_iter*d), BufferArg::from_raw_parts(pi.h_ctx.handle.clone(), bt*d), d_c, (n*d) as u32, (bt*d) as u32);
            kernels::rmsnorm_kernel::launch_unchecked::<f32, Cuda>(&client, CubeCount::Static(bt_c,1,1), UNITS, BufferArg::from_raw_parts(pi.h_ctx.handle.clone(), bt*d), BufferArg::from_raw_parts(g_prim.handle.clone(), d), BufferArg::from_raw_parts(pi.normed.handle.clone(), bt*d), BufferArg::from_raw_parts(pi.inv.handle.clone(), bt), d_c, inp.norm_eps);
            kernels::cat_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(pi.h_ctx.handle.clone(), bt*d), BufferArg::from_raw_parts(x_prim.handle.clone(), bt*d), BufferArg::from_raw_parts(pi.ctrl_in.handle.clone(), 2*bt*d), d_c, (bt*d) as u32);
        }
        launch_mm(&client, &pi.ctrl_in, &wc.prim, &rs_prim, &rs_prim, &pi.raw, bt, 2*d, pad, 2*d, 1, pad, 1, bt*d, 2*d*pad, false, false, false, false, false);
        unsafe {
            kernels::sigsel_arms_kernel::launch_unchecked::<f32, Cuda>(&client, CubeCount::Static(bt_c,1,1), UNITS, BufferArg::from_raw_parts(pi.raw.handle.clone(), bt*pad), BufferArg::from_raw_parts(w_attn_vec[n].handle.clone(), bt), BufferArg::from_raw_parts(w_mem_vec[n].handle.clone(), bt), BufferArg::from_raw_parts(pi.w_ffn.handle.clone(), bt), BufferArg::from_raw_parts(pi.blend.handle.clone(), bt*nexp), nexp_c, pad_c);
        }
        // KDA / MSA / Engram via TRUE direct Cube kernels (no inner Autodiff).
        // Only one fence before the first raw launch (above); all launches are
        // on the same ComputeClient stream, so no extra syncs are needed here.
        // KDA: gdn2_chunk_intra + gdn2_chunk_inter via kda_forward_cube
        if cfg.use_kda {
            let kda_cube = crate::fused::kernels::arms::kda_forward_cube(&lb_bare.shared_attn.gdn2, &pi.normed, b, t, d);
            unsafe { kernels::copy_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(kda_cube.handle.clone(), bt*d), BufferArg::from_raw_parts(kda_vec[n].handle.clone(), bt*d), 0, 0, (bt*d) as u32); }
        } else {
            unsafe { kernels::fill_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(kda_vec[n].handle.clone(), bt*d), 0.0, (bt*d) as u32); }
        }
        // MSA: msa_sparse_attn_kernel via msa_forward_cube
        if cfg.use_msa && t > 1 && t >= cfg.msa_block {
            let msa_cube = crate::fused::kernels::arms::msa_forward_cube(&lb_bare.shared_attn.msa, &pi.normed, b, t, d);
            unsafe { kernels::copy_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(msa_cube.handle.clone(), bt*d), BufferArg::from_raw_parts(msa_vec[n].handle.clone(), bt*d), 0, 0, (bt*d) as u32); }
        } else {
            unsafe { kernels::fill_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(msa_vec[n].handle.clone(), bt*d), 0.0, (bt*d) as u32); }
        }
        // router gate: direct via launch_mm + sigmoid (Cube)
        {
            let gate_cube = crate::fused::kernels::arms::router_gate_cube(&lb_bare.shared_attn.router, &pi.normed, b, t, d);
            unsafe { kernels::copy_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt), EW, BufferArg::from_raw_parts(gate_cube.handle.clone(), bt), BufferArg::from_raw_parts(gate_vec[n].handle.clone(), bt), 0, 0, bt as u32); }
        }
        // attn blended + scaled: gdn2_chunk + msa_sparse + router blend
        unsafe { kernels::attn_blend_scale_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(kda_vec[n].handle.clone(), bt*d), BufferArg::from_raw_parts(msa_vec[n].handle.clone(), bt*d), BufferArg::from_raw_parts(gate_vec[n].handle.clone(), bt), BufferArg::from_raw_parts(w_attn_vec[n].handle.clone(), bt), BufferArg::from_raw_parts(attn_vec[n].handle.clone(), bt*d), d_c, (bt*d) as u32); }
        // Engram: engram_gather_kernel via engram_forward_cube. Host-rows mode
        // (RAM offload) takes priority, mirroring forward_with_hidden.
        if cfg.use_engram {
            if let Some(ref rc) = rows_cube {
                let eng_cube = crate::fused::kernels::arms::engram_forward_embeds_cube(&lb_bare.engram, rc, &pi.h_ctx, b, t, d);
                unsafe { kernels::copy_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(eng_cube.handle.clone(), bt*d), BufferArg::from_raw_parts(engram_vec[n].handle.clone(), bt*d), 0, 0, (bt*d) as u32); }
                unsafe { kernels::engram_scale_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(engram_vec[n].handle.clone(), bt*d), BufferArg::from_raw_parts(w_mem_vec[n].handle.clone(), bt), BufferArg::from_raw_parts(engram_vec[n].handle.clone(), bt*d), d_c, (bt*d) as u32); }
            } else if let Some(ref hc) = hashed_cube {
                let eng_cube = crate::fused::kernels::arms::engram_forward_cube(&lb_bare.engram, hc, &pi.h_ctx, b, t, d, 3);
                unsafe { kernels::copy_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(eng_cube.handle.clone(), bt*d), BufferArg::from_raw_parts(engram_vec[n].handle.clone(), bt*d), 0, 0, (bt*d) as u32); }
                unsafe { kernels::engram_scale_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(engram_vec[n].handle.clone(), bt*d), BufferArg::from_raw_parts(w_mem_vec[n].handle.clone(), bt), BufferArg::from_raw_parts(engram_vec[n].handle.clone(), bt*d), d_c, (bt*d) as u32); }
            } else {
                unsafe { kernels::fill_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(engram_vec[n].handle.clone(), bt*d), 0.0, (bt*d) as u32); }
            }
        } else {
            unsafe { kernels::fill_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(engram_vec[n].handle.clone(), bt*d), 0.0, (bt*d) as u32); }
        }
        // experts
        for e in 0..nexp {
            let [gu, dn] = &experts_c[e];
            let wsc = &pi.ws[e];
            launch_mm(&client, &pi.normed, &gu.u, &gu.s, &gu.mu, &wsc[0], bt, d, r, d, 1, r, 1, bt*d, d*r, false, false, true, false, false);
            launch_mm(&client, &wsc[0], &gu.v, &gu.s, &gu.mv, &wsc[1], bt, r, f, r, 1, r, 1, bt*r, f*r, false, true, true, true, false);
            unsafe { kernels::silu_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*f), EW, BufferArg::from_raw_parts(wsc[1].handle.clone(), bt*f), BufferArg::from_raw_parts(wsc[2].handle.clone(), bt*f), (bt*f) as u32); }
            launch_mm(&client, &wsc[2], &dn.u, &dn.s, &dn.mu, &wsc[3], bt, f, r, f, 1, r, 1, bt*f, f*r, false, false, true, false, false);
            launch_mm(&client, &wsc[3], &dn.v, &dn.s, &dn.mv, &wsc[4], bt, r, d, r, 1, r, 1, bt*r, d*r, false, true, true, true, false);
            unsafe { kernels::axpy_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(pi.ffn.handle.clone(), bt*d), BufferArg::from_raw_parts(wsc[4].handle.clone(), bt*d), BufferArg::from_raw_parts(pi.blend.handle.clone(), bt*nexp), e as u32, nexp_c, d_c, (bt*d) as u32); }
        }
        // y = attn + engram + ffn*w_ffn
        let ffn_scaled = {
            let (t, c) = empty1(&dev, bt*d);
            keep1.push(t);
            c
        };
        unsafe { kernels::engram_scale_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(pi.ffn.handle.clone(), bt*d), BufferArg::from_raw_parts(pi.w_ffn.handle.clone(), bt), BufferArg::from_raw_parts(ffn_scaled.handle.clone(), bt*d), d_c, (bt*d) as u32); }
        unsafe { kernels::add3_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(attn_vec[n].handle.clone(), bt*d), BufferArg::from_raw_parts(engram_vec[n].handle.clone(), bt*d), BufferArg::from_raw_parts(ffn_scaled.handle.clone(), bt*d), BufferArg::from_raw_parts(pi.y.handle.clone(), bt*d), (bt*d) as u32); }
        unsafe { kernels::h_from_y_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(pi.h_ctx.handle.clone(), bt*d), BufferArg::from_raw_parts(pi.y.handle.clone(), bt*d), BufferArg::from_raw_parts(rs_prim.handle.clone(), 1), BufferArg::from_raw_parts(pi.h.handle.clone(), bt*d), (bt*d) as u32); }
        unsafe {
            kernels::rowmean_t_kernel::launch_unchecked::<f32, Cuda>(&client, CubeCount::Static(b_c,1,1), UNITS, BufferArg::from_raw_parts(pi.h_ctx.handle.clone(), bt*d), BufferArg::from_raw_parts(pi.halt_in.handle.clone(), b*d), t_c, d_c);
            kernels::halt_fwd_kernel::launch_unchecked::<f32, Cuda>(&client, CubeCount::Static(b_c,1,1), UNITS, BufferArg::from_raw_parts(pi.halt_in.handle.clone(), b*d), BufferArg::from_raw_parts(wh.prim.handle.clone(), d), BufferArg::from_raw_parts(pi.lam.handle.clone(), b), d_c);
        }
        launch_mm(&client, &pi.h, &op.0.u, &op.0.s, &op.0.mu, &pi.z_o, bt, d, r, d, 1, r, 1, bt*d, d*r, false, false, true, false, false);
        launch_mm(&client, &pi.z_o, &op.0.v, &op.0.s, &op.0.mv, &pi.step_out, bt, r, d, r, 1, r, 1, bt*r, d*r, false, true, true, true, false);
        launch_mm(&client, &pi.step_out, &lm.0.u, &lm.0.s, &lm.0.mu, &pi.z_l, bt, d, r, d, 1, r, 1, bt*d, d*r, false, false, true, false, false);
        launch_mm(&client, &pi.z_l, &lm.0.v, &lm.0.s, &lm.0.mv, &pi.logits, bt, r, v, r, 1, r, 1, bt*r, v*r, false, true, true, true, false);
        unsafe {
            kernels::ce_kernel::launch_unchecked::<f32, Cuda>(&client, CubeCount::Static(bt_c,1,1), UNITS, BufferArg::from_raw_parts(pi.logits.handle.clone(), bt*v), BufferArg::from_raw_parts(tgt_c.handle.clone(), bt), BufferArg::from_raw_parts(pi.ce.handle.clone(), bt), v as u32);
            kernels::ceb_kernel::launch_unchecked::<f32, Cuda>(&client, CubeCount::Static(b_c,1,1), UNITS, BufferArg::from_raw_parts(pi.ce.handle.clone(), bt), BufferArg::from_raw_parts(pi.ceb.handle.clone(), b), t_c);
            kernels::halting_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(b), EW, BufferArg::from_raw_parts(pi.lam.handle.clone(), b), BufferArg::from_raw_parts(if n==0{&nh0}else{&per[n-1].nh}.handle.clone(), b), BufferArg::from_raw_parts(pi.nh.handle.clone(), b), BufferArg::from_raw_parts(pi.p.handle.clone(), b), BufferArg::from_raw_parts(flat.handle.clone(), 2+b*n_iter+bt*v), (1+n*b) as u32, b as u32);
            kernels::rec_kernel::launch_unchecked::<f32, Cuda>(&client, CubeCount::Static(1,1,1), UNITS, BufferArg::from_raw_parts(pi.p.handle.clone(), b), BufferArg::from_raw_parts(pi.ceb.handle.clone(), b), BufferArg::from_raw_parts(flat.handle.clone(), 2+b*n_iter+bt*v), b as u32, bt as u32, n>0);
            kernels::outacc_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(pi.step_out.handle.clone(), bt*d), BufferArg::from_raw_parts(pi.p.handle.clone(), b), BufferArg::from_raw_parts(oa.handle.clone(), bt*d), t_c, d_c, (bt*d) as u32, n>0);
        }
    }
    unsafe { kernels::kl_kernel::launch_unchecked::<f32, Cuda>(&client, CubeCount::Static(1,1,1), UNITS, BufferArg::from_raw_parts(flat.handle.clone(), 2+b*n_iter+bt*v), BufferArg::from_raw_parts(prior_c.handle.clone(), n_iter), BufferArg::from_raw_parts(flat.handle.clone(), 2+b*n_iter+bt*v), 1, (1+b*n_iter) as u32, b as u32, n_iter as u32); }
    unsafe { kernels::rmsnorm_kernel::launch_unchecked::<f32, Cuda>(&client, CubeCount::Static(bt_c,1,1), UNITS, BufferArg::from_raw_parts(oa.handle.clone(), bt*d), BufferArg::from_raw_parts(gf_prim.handle.clone(), d), BufferArg::from_raw_parts(hf.handle.clone(), bt*d), BufferArg::from_raw_parts(invf.handle.clone(), bt), d_c, inp.norm_eps); }
    launch_mm(&client, &hf, &lm.0.u, &lm.0.s, &lm.0.mu, &zlf, bt, d, r, d, 1, r, 1, bt*d, d*r, false, false, true, false, false);
    launch_mm(&client, &zlf, &lm.0.v, &lm.0.s, &lm.0.mv, &lgstg, bt, r, v, r, 1, r, 1, bt*r, v*r, false, true, true, true, false);
    unsafe { kernels::copy_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*v), EW, BufferArg::from_raw_parts(lgstg.handle.clone(), bt*v), BufferArg::from_raw_parts(flat.handle.clone(), 2+b*n_iter+bt*v), 0, (2+b*n_iter) as u32, (bt*v) as u32); }
    unsafe {
        // latents for the burn-path aux heads (JEPA on out_acc, DSpark on hf)
        kernels::copy_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(oa.handle.clone(), bt*d), BufferArg::from_raw_parts(flat.handle.clone(), off_oa), 0, off_oa as u32, (bt*d) as u32);
        kernels::copy_kernel::launch_unchecked::<f32, Cuda>(&client, ew_cubes(bt*d), EW, BufferArg::from_raw_parts(hf.handle.clone(), bt*d), BufferArg::from_raw_parts(flat.handle.clone(), off_oa + bt*d), 0, (off_oa + bt*d) as u32, (bt*d) as u32);
    }
    // No extra sync here: the single fence was at the start, and the final
    // sync is after the last kernel before the autodiff node is created.
    // The 1 outer node will be created below; its backward will run the
    // exact adjoint via gdn2_chunk and msa_sparse kernels (no inner Autodiff).
    sync(&client);
    let nodes_arr: [_; 62] = nodes.try_into().expect("62 parents for arms (3 or 8 experts)");
    let ats_arr: [_; 62] = ats.try_into().expect("62 checkpoints");
    let base = PonderState { b, t, d, f, r, v, nexp, pad, bt, n_iter, prior: prior_v, dev: dev.clone(), tgt: tgt_c, wc: wc.prim, g: g_prim, gf: gf_prim, rs: rs_prim, wh: wh.prim, experts: experts_c, op: op.0, lm: lm.0, per, oa, hf, invf, zlf, nh0, keep2, keep1 };
    let state = ArmsDirectState { base, cfg: cfg.clone(), w_attn: w_attn_vec, w_mem: w_mem_vec, kda_out: kda_vec, msa_out: msa_vec, engram_out: engram_vec, attn: attn_vec, gate: gate_vec, hashed: hashed_cube, rows: rows_cube, arm: inp.arm_leaves.clone(), lb_bytes: lb_bytes.clone(), x: x_prim.clone() };
    let prep = PonderLoopArms.prepare::<NoCheckpointing>(nodes_arr);
    match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut p) => {
            for at in &ats_arr { let _id = p.checkpoint(at); }
            let out = p.finish(state, flat);
            split_outputs(out, b, t, v, b*n_iter, bt, d)
        }
        OpsKind::UnTracked(p) => { let out = p.finish(flat); split_outputs(out, b, t, v, b*n_iter, bt, d) }
    }
}
