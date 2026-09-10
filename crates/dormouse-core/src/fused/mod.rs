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

/// Everything the fused single-iteration step needs, as tracked tensors.
pub struct PonderInputs {
    /// loop input (embedding output) [b, t, d]
    pub x: Tensor<3>,
    /// target byte indices [b·t, 1]
    pub targets: Tensor<2, Int>,
    pub controller_w: Tensor<2>,
    pub norm_g: Tensor<1>,
    pub iter_embed: Tensor<2>,
    pub residual_scale: Tensor<1>,
    pub halt_w: Tensor<2>,
    /// [gate_up, down] factor triples per expert
    pub experts: Vec<[Fac; 2]>,
    pub out_proj: Fac,
    pub lm_head: Fac,
    pub norm_eps: f32,
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
/// 24 out_proj u/s/v | 27 lm_head u/s/v.
#[derive(Debug)]
struct PonderLoop;

impl Backward<CB, 30> for PonderLoop {
    type State = PonderState;

    fn backward(self, ops: Ops<Self::State, 30>, grads: &mut Gradients, cp: &mut Checkpointer) {
        backward::ponder_backward(ops, grads, cp)
    }
}

/// The fused single-iteration Ponder step (M1). Returns
/// `(rec [1], p_dist [b,1], out_acc [b,t,d])` - the same triple
/// `LoopBlock::forward_full_state` produces at max_iter=1 with the
/// attention/engram arms off - computed under ONE autodiff node.
pub fn ponder_loop_step(inp: PonderInputs) -> (Tensor<1>, Tensor<2>, Tensor<3>)
where
    DispatchTensor: DispatchKindConversion<CAd> + DispatchKindConversion<CB>,
{
    let nexp = inp.experts.len();
    // The op's parent list is fixed at compile time (Backward<CB, 30>):
    // 6 dense weights + 6 per expert + 3 out_proj + 3 lm_head = 12 + 6·nexp.
    // Only n_experts = 3 is compiled (`small`/`base`; one_b runs 4).
    assert_eq!(nexp, 3, "fused op is compiled for n_experts=3 (30 parents = 12 + 6·n_experts); got n_experts={nexp} - lift PonderLoop's arity before running one_b");
    let [b, t, d] = inp.x.dims();
    let bt = b * t;
    // burn-nn Linear uses the Col layout: the weight is [in, out] = [2d, pad]
    // (x @ W, no transpose in the reference forward).
    let [two_d, pad] = dense(inp.controller_w.clone()).dims();
    assert_eq!(two_d, 2 * d, "controller weight must be [2d, pad]");
    let max_iter = inp.iter_embed.dims()[0];
    assert_eq!(max_iter, 1, "M1 is single-iteration; M3 lifts this");
    let x2 = inp.x.clone().reshape([bt, d]);
    let xa = x2
        .clone()
        .try_into_primitive::<CAd>()
        .expect("fused op requires Autodiff<CudaBackend> tensors");
    let x_prim = xa.primitive.clone();
    let x_t = Tensor::<2>::from_primitive::<CB>(x_prim.clone());
    let dev = x_t.device();
    let tgt_c = cube_int2(&inp.targets).expect("cuda targets");

    let mut keep2: Vec<Tensor<2>> = Vec::new();
    let mut keep1: Vec<Tensor<1>> = Vec::new();
    let mut ats: Vec<AdPrim> = Vec::with_capacity(30);
    let mut nodes = Vec::with_capacity(30); // [NodeRef; 30], inferred

    // ---- parents 1..6 (dense weights + params)
    let wc = ad2(inp.controller_w.clone());
    let (g_t, g_at, g_prim) = ad1(inp.norm_g.clone());
    let ie = ad2(inp.iter_embed.clone());
    let (rs_t, rs_at, rs_prim) = ad1(inp.residual_scale.clone());
    let wh = ad2(inp.halt_w.clone());
    keep2.push(wc.t);
    keep2.push(ie.t);
    keep2.push(wh.t);
    keep1.push(g_t);
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

    let (nexp_c, pad_c, bt_c, b_c, t_c, d_c) =
        (nexp as u32, pad as u32, bt as u32, b as u32, t as u32, d as u32);
    let client = x_prim.client.clone();

    // ---- workspace (bridged twins moved into the state keep buffers alive)
    let (h_ctx, h_ctxc) = empty1(&dev, bt * d);
    let (normed, normedc) = empty1(&dev, bt * d);
    let (inv, invc) = empty1(&dev, bt);
    let (ctrl_in, ctrl_inc) = empty1(&dev, 2 * bt * d);
    let (raw, rawc) = empty1(&dev, bt * pad);
    let (w_ffn, w_ffnc) = empty1(&dev, bt);
    let (blend, blendc) = empty1(&dev, bt * nexp);
    let (ffn, ffnc) = zeros1(&dev, bt * d);
    let (y, yc) = empty1(&dev, bt * d);
    let (h, hc) = empty1(&dev, bt * d);
    let (z_o, z_oc) = empty1(&dev, bt * r);
    let (step_out, step_outc) = empty1(&dev, bt * d);
    let (z_l, z_lc) = empty1(&dev, bt * r);
    let (logits, logitsc) = empty1(&dev, bt * v);
    let (halt_in, halt_inc) = empty1(&dev, b * d);
    let (ce, cec) = empty1(&dev, bt);
    let (ceb, cebc) = empty1(&dev, b);
    let (_flat_t, flat) = zeros1(&dev, 1 + b + bt * d);

    let mut ws: Vec<[CubeTensor; 5]> = Vec::new(); // z_e, a_e, mid, z_d, out_e
    for _ in 0..nexp {
        let (z_e_t, z_e) = empty1(&dev, bt * r);
        let (a_e_t, a_e) = empty1(&dev, bt * f);
        let (mid_t, mid) = empty1(&dev, bt * f);
        let (zd_t, z_d) = empty1(&dev, bt * r);
        let (oe_t, out_e) = empty1(&dev, bt * d);
        keep1.extend([z_e_t, a_e_t, mid_t, zd_t, oe_t]);
        ws.push([z_e, a_e, mid, z_d, out_e]);
    }

    // ---- forward kernels.
    // FENCE BEFORE THE FIRST RAW LAUNCH, NO EXCEPTIONS: everything above is
    // burn-side work (val clones, dense() copies, H2D for targets, and in
    // training optim.step/retract rewriting the u/v masters every step) that
    // may sit on other streams than our raw launches. Without this barrier
    // the kernels below race it - the absmean reads of the u/v masters
    // especially would pick up half-written factors.
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
    unsafe {
        kernels::hctx_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            ew_cubes(bt * d),
            EW,
            BufferArg::from_raw_parts(x_prim.handle.clone(), bt * d),
            BufferArg::from_raw_parts(ie.prim.handle.clone(), max_iter * d),
            BufferArg::from_raw_parts(h_ctxc.handle.clone(), bt * d),
            d_c,
            (bt * d) as u32,
        );
        kernels::rmsnorm_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(bt_c, 1, 1),
            UNITS,
            BufferArg::from_raw_parts(h_ctxc.handle.clone(), bt * d),
            BufferArg::from_raw_parts(g_prim.handle.clone(), d),
            BufferArg::from_raw_parts(normedc.handle.clone(), bt * d),
            BufferArg::from_raw_parts(invc.handle.clone(), bt),
            d_c,
            inp.norm_eps,
        );
        kernels::cat_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            ew_cubes(bt * d),
            EW,
            BufferArg::from_raw_parts(h_ctxc.handle.clone(), bt * d),
            BufferArg::from_raw_parts(x_prim.handle.clone(), bt * d),
            BufferArg::from_raw_parts(ctrl_inc.handle.clone(), 2 * bt * d),
            d_c,
            (bt * d) as u32,
        );
    }
    // raw = ctrl_in @ Wc   [bt,2d]·[2d,pad] (Col layout, no transpose)
    launch_mm(&client, &ctrl_inc, &wc.prim, &rs_prim, &rs_prim, &rawc, bt, 2 * d, pad, 2 * d, 1, pad, 1, bt * d, 2 * d * pad, false, false, false, false, false);
    // Launch order follows the dataflow: same-stream raw launches are FIFO,
    // so every producer MUST be enqueued before its consumers (sigsel reads
    // raw, residual reads ffn, ce reads logits, ...).
    unsafe {
        kernels::sigsel_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(bt_c, 1, 1),
            UNITS,
            BufferArg::from_raw_parts(rawc.handle.clone(), bt * pad),
            BufferArg::from_raw_parts(w_ffnc.handle.clone(), bt),
            BufferArg::from_raw_parts(blendc.handle.clone(), bt * nexp),
            nexp_c,
            pad_c,
        );
    }
    for e in 0..nexp {
        let [gu, dn] = &experts_c[e];
        let wsc = &ws[e];
        // Z_e = normed @ U_te (ternary U)
        launch_mm(&client, &normedc, &gu.u, &gu.s, &gu.mu, &wsc[0], bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
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
                BufferArg::from_raw_parts(ffnc.handle.clone(), bt * d),
                BufferArg::from_raw_parts(wsc[4].handle.clone(), bt * d),
                BufferArg::from_raw_parts(blendc.handle.clone(), bt * nexp),
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
            BufferArg::from_raw_parts(ffnc.handle.clone(), bt * d),
            BufferArg::from_raw_parts(w_ffnc.handle.clone(), bt),
            BufferArg::from_raw_parts(rs_prim.handle.clone(), 1),
            BufferArg::from_raw_parts(h_ctxc.handle.clone(), bt * d),
            BufferArg::from_raw_parts(yc.handle.clone(), bt * d),
            BufferArg::from_raw_parts(hc.handle.clone(), bt * d),
            d_c,
        );
        kernels::rowmean_t_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(b_c, 1, 1),
            UNITS,
            BufferArg::from_raw_parts(h_ctxc.handle.clone(), bt * d),
            BufferArg::from_raw_parts(halt_inc.handle.clone(), b * d),
            t_c,
            d_c,
        );
        kernels::halt_fwd_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(b_c, 1, 1),
            UNITS,
            BufferArg::from_raw_parts(halt_inc.handle.clone(), b * d),
            BufferArg::from_raw_parts(wh.prim.handle.clone(), d),
            BufferArg::from_raw_parts(flat.handle.clone(), 1 + b),
            d_c,
        );
    }
    // out_proj: Z_o = h @ U_ot ; step_out = (Z_o · s) @ V_ot^T
    launch_mm(&client, &hc, &op.0.u, &op.0.s, &op.0.mu, &z_oc, bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
    launch_mm(&client, &z_oc, &op.0.v, &op.0.s, &op.0.mv, &step_outc, bt, r, d, r, 1, r, 1, bt * r, d * r, false, true, true, true, false);
    // lm_head: Z_l = step_out @ U_lt ; logits = (Z_l · s) @ V_lt^T
    launch_mm(&client, &step_outc, &lm.0.u, &lm.0.s, &lm.0.mu, &z_lc, bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
    launch_mm(&client, &z_lc, &lm.0.v, &lm.0.s, &lm.0.mv, &logitsc, bt, r, v, r, 1, r, 1, bt * r, v * r, false, true, true, true, false);
    unsafe {
        kernels::ce_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(bt_c, 1, 1),
            UNITS,
            BufferArg::from_raw_parts(logitsc.handle.clone(), bt * v),
            BufferArg::from_raw_parts(tgt_c.handle.clone(), bt),
            BufferArg::from_raw_parts(cec.handle.clone(), bt),
            v as u32,
        );
        kernels::ceb_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(b_c, 1, 1),
            UNITS,
            BufferArg::from_raw_parts(cec.handle.clone(), bt),
            BufferArg::from_raw_parts(cebc.handle.clone(), b),
            t_c,
        );
        kernels::rec_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(1, 1, 1),
            UNITS,
            BufferArg::from_raw_parts(cebc.handle.clone(), b),
            BufferArg::from_raw_parts(flat.handle.clone(), 1 + b),
            1u32,
            b_c,
            bt_c,
        );
        kernels::outacc_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            ew_cubes(bt * d),
            EW,
            BufferArg::from_raw_parts(step_outc.handle.clone(), bt * d),
            BufferArg::from_raw_parts(flat.handle.clone(), 1 + b + bt * d),
            1u32,
            (1 + b) as u32,
            t_c,
            d_c,
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
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
                .to_vec()
        };
        for (name, buf, n) in [
            ("h_ctx", &h_ctxc, bt * d),
            ("normed", &normedc, bt * d),
            ("raw", &rawc, bt * pad),
            ("w_ffn", &w_ffnc, bt),
            ("blend", &blendc, bt * nexp),
            ("ffn", &ffnc, bt * d),
            ("h", &hc, bt * d),
            ("z_o", &z_oc, bt * r),
            ("step_out", &step_outc, bt * d),
            ("z_l", &z_lc, bt * r),
        ] {
            FWD_DUMP.with(|d| d.borrow_mut().push((name, read(buf, n))));
        }
    }

    // ---- ONE autodiff node for the whole computation
    let nodes: [_; 30] = nodes.try_into().expect("parent count (n_experts asserted above)");
    let ats: [AdPrim; 30] = ats.try_into().expect("checkpoint count (n_experts asserted above)");

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
        max_iter,
        eps: inp.norm_eps,
        dev: dev.clone(),
        x: x_prim,
        tgt: tgt_c,
        wc: wc.prim,
        g: g_prim,
        ie: ie.prim,
        rs: rs_prim,
        wh: wh.prim,
        experts: experts_c,
        op: op.0,
        lm: lm.0,
        h_ctx,
        normed,
        inv,
        ctrl_in,
        raw,
        w_ffn,
        blend,
        ffn,
        y,
        h,
        z_o,
        step_out,
        z_l,
        logits,
        halt_in,
        ce,
        ceb,
        ws,
        flat_out: flat.clone(),
        keep2,
        keep1,
    };

    let prep = PonderLoop.prepare::<NoCheckpointing>(nodes);
    let (rec, pd, oa) = match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut p) => {
            for at in &ats {
                let _id = p.checkpoint(at);
            }
            let out = p.finish(state, flat);
            split_outputs(out, b, t, d, bt)
        }
        OpsKind::UnTracked(p) => {
            let out = p.finish(flat);
            split_outputs(out, b, t, d, bt)
        }
    };
    (rec, pd, oa)
}

fn split_outputs(
    out: <CAd as burn::backend::BackendTypes>::FloatTensorPrimitive,
    b: usize,
    t: usize,
    d: usize,
    bt: usize,
) -> (Tensor<1>, Tensor<2>, Tensor<3>) {
    let out_t = Tensor::<1>::from_primitive::<CAd>(out);
    let rec = out_t.clone().slice([0..1]);
    let pd = out_t.clone().slice([1..1 + b]).reshape([b, 1]);
    let oa = out_t.slice([1 + b..1 + b + bt * d]).reshape([b, t, d]);
    (rec, pd, oa)
}
