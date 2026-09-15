//! `ponder_backward` - the hand-written backward for the fused N-iteration
//! Ponder loop. Implements the plan's "Backward math checklist":
//!
//! - dense matmul `Y = X·W`: `dW = Xᵀ·dY`, `dX = dY·Wᵀ`
//! - TSCT `Y = ((X·U)·s)·Vᵀ` with STE-ternary factors: grads flow through the
//!   ternary `U_t`/`V_t` (recomputed on the fly from master + saved mean) and
//!   land on the masters unchanged:
//!   `dM = dY·V_t`, `dV = (dYᵀ·(Z·s))`, `ds = Σ_m(dM⊙Z)`, `dZ = dM·s`,
//!   `dU = Xᵀ·dZ`, `dX = dZ·U_tᵀ` (the `·s` on dV is folded as a post column
//!   scale, since `(dYᵀZ)·s = dYᵀ(Z·s)` columnwise)
//! - RMSNorm: `dX = invσ·(dY·g − r·mean(dY·g·r))`, `dg = Σ_m dY·g·r`
//! - sigmoid `dX = dY·out·(1−out)`; softmax `dRaw = blend⊙(dBlend − Σ dBlend·blend)`
//! - silu `dz = dY·(σ+z·σ·(1−σ))`; ReZero `dY = dH·rs`, `drs = Σ dH⊙y`
//! - final readout: `dOut_acc` = RMSNorm backward of the upstream final-logits
//!   grad; the lm_head weight grad sums the per-step CE contributions and the
//!   final readout (both accumulate into the same TSCT grad buffers).
//! - PonderNet recurrence (per iteration n, see `lam_bwd_kernel`):
//!   `dp̃ = dp_dist + dkl·(ln p − ln prior_n + 1)/(b·N) + dRec·CE + dot`,
//!   `dλ_n = (dp̃ − g_{n+1})·nh_n`, `g_n = g_{n+1}·(1−λ_n) + dp̃·λ_n`.
//!
//! The upstream grad arrives on the single op node as one flat buffer with the
//! output layout `[rec | p_dist (b·N) | kl | final logits (b·t·v)]`. Weight
//! grads accumulate across iterations (weights are shared); iteration
//! temporaries stay alive until after the final sync (same-stream FIFO makes
//! reuse safe, and the N-fold footprint at training shapes is an M5/M6
//! workspace concern, not a correctness one).

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::Ops;
use burn::tensor::{Tensor, TensorData};
use cubecl::prelude::*;

use super::{
    cube_of1, dense, empty1, ew_cubes, launch_mm, zeros_raw, ArmsState, CubeTensor, FacC, IterBufs,
    CB, Cuda, UNITS, EW,
};

/// Saved forward state for the fused Ponder loop (see `mod.rs`).
#[derive(Clone, Debug)]
pub(crate) struct PonderState {
    pub b: usize,
    pub t: usize,
    pub d: usize,
    pub f: usize,
    pub r: usize,
    pub v: usize,
    pub nexp: usize,
    pub pad: usize,
    pub bt: usize,
    pub n_iter: usize,
    /// host copy of the renormalized KL prior (per-iteration log_prior scalars)
    pub prior: Vec<f32>,
    pub dev: burn::tensor::Device,
    // masters
    pub tgt: CubeTensor,
    pub wc: CubeTensor,
    pub g: CubeTensor,
    pub gf: CubeTensor,
    pub rs: CubeTensor,
    pub wh: CubeTensor,
    pub experts: Vec<[FacC; 2]>,
    pub op: FacC,
    pub lm: FacC,
    // per-iteration saved forward workspace
    pub per: Vec<IterBufs>,
    // final readout saves (out_acc, its norm output + inv, the lm z-factor)
    pub oa: CubeTensor,
    pub hf: CubeTensor,
    pub invf: CubeTensor,
    pub zlf: CubeTensor,
    /// not-halted seed (ones [b]): lam_bwd needs the nh ENTERING iteration n,
    /// which for n=0 is this, not per[0].nh (that one is post-update).
    pub nh0: CubeTensor,
    // keep-alive twins: never read, dropping them early would free buffers
    // the enqueued-but-unfinished raw kernels still write/read.
    #[allow(dead_code)]
    pub keep2: Vec<Tensor<2>>,
    #[allow(dead_code)]
    pub keep1: Vec<Tensor<1>>,
}

macro_rules! buf {
    ($h:expr, $len:expr) => {
        BufferArg::from_raw_parts($h.handle.clone(), $len)
    };
}

#[cfg(test)]
thread_local! {
    /// Buffer-level backward dumps (DM_FUSED_BWD_DEBUG=1): (name, values).
    /// Consumed by the gradcheck bisect test; empty unless enabled.
    pub static BWD_DUMP: std::cell::RefCell<Vec<(&'static str, Vec<f32>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Buffer dump for the gradcheck bisect test; compiles to nothing outside
/// `cfg(test)` (the sync-ing read would perturb the measured pipeline).
#[cfg(test)]
fn dump(client: &ComputeClient<Cuda>, name: &'static str, c: &CubeTensor, n: usize) {
    if std::env::var("DM_FUSED_BWD_DEBUG").is_ok() {
        let bytes = client.read(vec![c.handle.clone()]).remove(0);
        // cubecl rounds allocations up (pow2 buckets): truncate to the
        // logical length or small buffers carry padding garbage.
        let all =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) };
        let v = all[..n.min(all.len())].to_vec();
        BWD_DUMP.with(|d| d.borrow_mut().push((name, v)));
    }
}

#[cfg(not(test))]
fn dump(_client: &ComputeClient<Cuda>, _name: &'static str, _c: &CubeTensor, _n: usize) {}

/// Run the backward kernels and register every weight/input gradient.
pub(super) fn ponder_backward(
    ops: Ops<PonderState, 31>,
    grads: &mut Gradients,
    _cp: &mut Checkpointer,
) {
    let st = ops.state;
    let gfull = Tensor::<1>::from_primitive::<CB>(grads.consume::<CB>(&ops.node));
    let gc = cube_of1(&dense(gfull)).expect("cuda grad");
    let client = gc.client.clone();
    // The upstream grad was produced by burn's backward kernels on their
    // streams; raw launches below run on the caller's stream - fence before
    // (and after, for the burn-side consumers of the registered grads).
    super::sync(&client);
    let dev = st.dev.clone();
    let (b, t, d, f, r, v, nexp, pad, bt, n_iter) =
        (st.b, st.t, st.d, st.f, st.r, st.v, st.nexp, st.pad, st.bt, st.n_iter);
    let bn = b * n_iter;
    // flat output layout [rec | pd(b·N) | kl | logits(bt·v)]
    let (pd_off, kl_off, lg_off) = (1usize, 1 + bn, 2 + bn);
    let one = st.per[0].ce.clone(); // dummy mm operand (never read: s/bm slots)

    let mut keep: Vec<Tensor<1>> = Vec::new();
    // Parent count = 13 + 6·nexp (n_experts checked at extraction in
    // `ponder_loop_step`); size everything off the live parent list so
    // lifting the arity touches one place.
    let mut out: Vec<Option<CubeTensor>> = vec![None; ops.parents.len()];
    let mut alloc = |n: usize| {
        let (t, c) = empty1(&dev, n);
        keep.push(t);
        c
    };
    // Weight-grad accumulation order: the FIRST reverse iteration (n = N-1)
    // writes the accumulators, later ones += (empty buffers hold garbage).
    let accum = |n: usize| n != n_iter - 1;

    // ---- final readout: dLogits_final = the logits region of the upstream
    // grad (the outside CE loss's backward already produced softmax−onehot).
    let dlf = alloc(bt * v);
    unsafe {
        super::kernels::copy_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            ew_cubes(bt * v),
            EW,
            buf!(gc, lg_off + bt * v),
            buf!(dlf, bt * v),
            lg_off as u32,
            0u32,
            (bt * v) as u32,
        );
    }
    dump(&client, "dlf", &dlf, bt * v);
    // lm_head TSCT backward (accumulates with the per-step contributions)
    let dm_lf = alloc(bt * r);
    launch_mm(&client, &dlf, &st.lm.v, &st.lm.s, &st.lm.mv, &dm_lf, bt, v, r, v, 1, r, 1, bt * v, v * r, false, false, true, false, false);
    let dz_lf = alloc(bt * r);
    unsafe {
        super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
            &client, ew_cubes(bt * r), EW,
            buf!(dm_lf, bt * r), buf!(st.lm.s, r), buf!(dz_lf, bt * r),
            r as u32, (bt * r) as u32,
        );
    }
    let dul = alloc(d * r);
    launch_mm(&client, &st.hf, &dz_lf, &one, &one, &dul, d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, false);
    let dvl_raw = alloc(v * r);
    launch_mm(&client, &dlf, &st.zlf, &one, &one, &dvl_raw, v, bt, r, v, 1, r, 1, bt * v, bt * r, true, false, false, false, false);
    let dsl = alloc(r);
    unsafe {
        super::kernels::sum_ds_kernel::launch_unchecked::<f32, Cuda>(
            &client, CubeCount::Static(1, 1, 1), UNITS,
            buf!(dm_lf, bt * r), buf!(st.zlf, bt * r), buf!(dsl, r),
            bt as u32, r as u32, false,
        );
    }
    let dpre = alloc(bt * d);
    // bm MUST be U's own absmean: tern_b=true reads bm[0] as the scale.
    launch_mm(&client, &dz_lf, &st.lm.u, &one, &st.lm.mu, &dpre, bt, r, d, r, 1, r, 1, bt * r, d * r, false, true, true, false, false);
    dump(&client, "dpre", &dpre, bt * d);
    // final RMSNorm backward: dOut_acc (the readout input is out_acc)
    let dgf = alloc(d);
    unsafe {
        super::kernels::dg_kernel::launch_unchecked::<f32, Cuda>(
            &client, CubeCount::Static(1, 1, 1), UNITS,
            buf!(dpre, bt * d), buf!(st.oa, bt * d), buf!(st.invf, bt),
            buf!(dgf, d),
            bt as u32, d as u32, false,
        );
    }
    let dout_acc = zeros_raw(&dev, &client, bt * d);
    unsafe {
        super::kernels::rms_bwd_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(bt as u32, 1, 1),
            UNITS,
            buf!(dpre, bt * d),
            buf!(st.gf, d),
            buf!(st.oa, bt * d),
            buf!(st.invf, bt),
            buf!(dout_acc, bt * d),
            d as u32,
        );
    }
    dump(&client, "dout_acc", &dout_acc, bt * d);
    out[30] = Some(dgf);

    // ---- reverse loop N-1..0. Weight-grad accumulators are seeded by the
    // first reverse iteration, then accumulated. The loop input x receives
    // grads from TWO places per iteration: the controller's [h_ctx | x] cat
    // (x-half of dctrl, every iteration) and the h_ctx_0 = x + ie_0 identity
    // (iteration 0's dh_ctx) - the h-chain carries only the dh_ctx half.
    let dxg = zeros_raw(&dev, &client, bt * d);
    let mut g_next = zeros_raw(&dev, &client, b);
    let mut dx_carry: Option<CubeTensor> = None;
    // lam_bwd_kernel covers exactly one 32-wide cube (UNITS; CubeCount 1×1×1):
    // a larger b would silently drop the tail.
    debug_assert!(b <= 32, "lam_bwd_kernel b={b} exceeds the single-cube cap (32)");
    // lm_head grad tails finished after the loop (dV accumulates raw, the
    // column scale commutes with the sum over iterations)
    let dvl = alloc(v * r);
    for n in (0..n_iter).rev() {
        let pi = &st.per[n];
        let acc = accum(n);

        // dot_n[b] = Σ_{t,d} dOut_acc·step_out_n (the out_acc halting path)
        let dot = zeros_raw(&dev, &client, b);
        unsafe {
            super::kernels::dlam_outacc_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(b as u32, 1, 1),
                UNITS,
                buf!(dout_acc, bt * d),
                buf!(pi.step_out, bt * d),
                buf!(dot, b),
                0u32,
                t as u32,
                d as u32,
            );
        }
        dump(&client, Box::leak(format!("dot#{n}").into_boxed_str()), &dot, b);

        // per-step CE: dLogits_n (drec from gc[0], p_n as the row weight)
        let dlogits = alloc(bt * v);
        unsafe {
            super::kernels::dlogits_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(bt as u32, 1, 1),
                UNITS,
                buf!(pi.logits, bt * v),
                buf!(st.tgt, bt),
                buf!(pi.p, b),
                buf!(gc, lg_off + bt * v),
                buf!(dlogits, bt * v),
                0u32,
                t as u32,
                v as u32,
                bt as u32,
            );
        }
        dump(&client, Box::leak(format!("dlogits#{n}").into_boxed_str()), &dlogits, bt * v);

        // lm_head TSCT backward (per-step contribution)
        let dm_l = alloc(bt * r);
        launch_mm(&client, &dlogits, &st.lm.v, &st.lm.s, &st.lm.mv, &dm_l, bt, v, r, v, 1, r, 1, bt * v, v * r, false, false, true, false, false);
        let dz_l = alloc(bt * r);
        unsafe {
            super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(bt * r), EW,
                buf!(dm_l, bt * r), buf!(st.lm.s, r), buf!(dz_l, bt * r),
                r as u32, (bt * r) as u32,
            );
        }
        // lm-head accumulators are pre-seeded by the final-readout
        // contributions, so these always ACCUMULATE (never overwrite).
        launch_mm(&client, &pi.step_out, &dz_l, &one, &one, &dul, d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, true);
        launch_mm(&client, &dlogits, &pi.z_l, &one, &one, &dvl_raw, v, bt, r, v, 1, r, 1, bt * v, bt * r, true, false, false, false, true);
        unsafe {
            super::kernels::sum_ds_kernel::launch_unchecked::<f32, Cuda>(
                &client, CubeCount::Static(1, 1, 1), UNITS,
                buf!(dm_l, bt * r), buf!(pi.z_l, bt * r), buf!(dsl, r),
                bt as u32, r as u32, true,
            );
        }
        // dStep = lm_bwd(dLogits) + dOut_acc·p_n (the out_acc readout path)
        let dstep = alloc(bt * d);
        launch_mm(&client, &dz_l, &st.lm.u, &one, &st.lm.mu, &dstep, bt, r, d, r, 1, r, 1, bt * r, d * r, false, true, true, false, false);
        unsafe {
            super::kernels::dso_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(bt * d), EW,
                buf!(dout_acc, bt * d),
                buf!(pi.p, b),
                0u32,
                buf!(dstep, bt * d),
                0u32,
                t as u32, d as u32, (bt * d) as u32,
            );
        }
        dump(&client, Box::leak(format!("dstep#{n}").into_boxed_str()), &dstep, bt * d);

        // ---- out_proj TSCT backward (dstep -> dh_flat)
        let dmo = alloc(bt * r);
        launch_mm(&client, &dstep, &st.op.v, &st.op.s, &st.op.mv, &dmo, bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
        let dz_o = alloc(bt * r);
        unsafe {
            super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(bt * r), EW,
                buf!(dmo, bt * r), buf!(st.op.s, r), buf!(dz_o, bt * r),
                r as u32, (bt * r) as u32,
            );
        }
        if !accum(n) {
            out[24] = Some(alloc(d * r));
            out[26] = Some(alloc(d * r));
        }
        launch_mm(&client, &pi.h, &dz_o, &one, &one, out[24].as_ref().unwrap(), d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, acc);
        launch_mm(&client, &dstep, &pi.z_o, &one, &one, out[26].as_ref().unwrap(), d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, acc);
        if out[25].is_none() {
            out[25] = Some(alloc(r));
        }
        unsafe {
            super::kernels::sum_ds_kernel::launch_unchecked::<f32, Cuda>(
                &client, CubeCount::Static(1, 1, 1), UNITS,
                buf!(dmo, bt * r), buf!(pi.z_o, bt * r),
                buf!(out[25].as_ref().unwrap(), r),
                bt as u32, r as u32, acc,
            );
        }
        let dh_flat = alloc(bt * d);
        launch_mm(&client, &dz_o, &st.op.u, &one, &st.op.mu, &dh_flat, bt, r, d, r, 1, r, 1, bt * r, d * r, false, true, true, false, false);
        // the next-lower iteration consumes this iteration's input grad
        if let Some(carry) = dx_carry.take() {
            unsafe {
                super::kernels::add_kernel::launch_unchecked::<f32, Cuda>(
                    &client, ew_cubes(bt * d), EW,
                    buf!(dh_flat, bt * d),
                    buf!(carry, bt * d),
                    (bt * d) as u32,
                );
            }
        }
        dump(&client, Box::leak(format!("dh_flat#{n}").into_boxed_str()), &dh_flat, bt * d);

        // ---- residual split: dy, dh_ctx += dh_flat, drs_part, dwffn_part
        let dy = alloc(bt * d);
        let dh_ctx = zeros_raw(&dev, &client, bt * d);
        let drs_part = alloc(bt);
        let dwffn_part = alloc(bt);
        unsafe {
            super::kernels::residual_bwd_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(bt as u32, 1, 1),
                UNITS,
                buf!(dh_flat, bt * d),
                buf!(st.rs, 1),
                buf!(pi.y, bt * d),
                buf!(pi.ffn, bt * d),
                buf!(dy, bt * d),
                buf!(dh_ctx, bt * d),
                buf!(drs_part, bt),
                buf!(dwffn_part, bt),
                d as u32,
            );
        }
        if out[4].is_none() {
            out[4] = Some(alloc(1));
        }
        unsafe {
            super::kernels::sum_part_kernel::launch_unchecked::<f32, Cuda>(
                &client, CubeCount::Static(1, 1, 1), UNITS,
                buf!(drs_part, bt),
                buf!(out[4].as_ref().unwrap(), 1),
                bt as u32, acc,
            );
        }

        // ---- ffn gate row (w_ffn jacobian) -> dRaw[:,2], dFfn
        let dffn = alloc(bt * d);
        let draw = zeros_raw(&dev, &client, bt * pad);
        unsafe {
            super::kernels::ffnrow_bwd_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(bt as u32, 1, 1),
                UNITS,
                buf!(dy, bt * d),
                buf!(pi.w_ffn, bt),
                buf!(dwffn_part, bt),
                buf!(dffn, bt * d),
                buf!(draw, bt * pad),
                pad as u32,
                d as u32,
            );
        }

        // ---- per-expert backward (du/dv accumulate across iterations)
        let dblend = alloc(bt * nexp);
        let dout_e = alloc(bt * d);
        let da2 = alloc(bt * f);
        // dx = grad of normed_n (fresh per iteration; rms_bwd consumes it)
        let dx = zeros_raw(&dev, &client, bt * d);
        for e in 0..nexp {
            let [gu, dn] = &st.experts[e];
            let wsc = &pi.ws[e];
            unsafe {
                super::kernels::dblend_kernel::launch_unchecked::<f32, Cuda>(
                    &client,
                    CubeCount::Static(bt as u32, 1, 1),
                    UNITS,
                    buf!(dffn, bt * d),
                    buf!(wsc[4], bt * d),
                    buf!(pi.blend, bt * nexp),
                    buf!(dout_e, bt * d),
                    buf!(dblend, bt * nexp),
                    e as u32,
                    nexp as u32,
                    d as u32,
                );
            }
            // down TSCT: dOut_e -> dsil
            let dm_d = alloc(bt * r);
            launch_mm(&client, &dout_e, &dn.v, &dn.s, &dn.mv, &dm_d, bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
            let dz_d = alloc(bt * r);
            unsafe {
                super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                    &client, ew_cubes(bt * r), EW,
                    buf!(dm_d, bt * r), buf!(dn.s, r), buf!(dz_d, bt * r),
                    r as u32, (bt * r) as u32,
                );
            }
            if out[9 + 6 * e].is_none() {
                out[9 + 6 * e] = Some(alloc(f * r));
                out[11 + 6 * e] = Some(alloc(d * r));
            }
            launch_mm(&client, &wsc[2], &dz_d, &one, &one, out[9 + 6 * e].as_ref().unwrap(), f, bt, r, f, 1, r, 1, bt * f, bt * r, true, false, false, false, acc);
            launch_mm(&client, &dout_e, &wsc[3], &one, &one, out[11 + 6 * e].as_ref().unwrap(), d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, acc);
            if out[10 + 6 * e].is_none() {
                out[10 + 6 * e] = Some(alloc(r));
            }
            unsafe {
                super::kernels::sum_ds_kernel::launch_unchecked::<f32, Cuda>(
                    &client, CubeCount::Static(1, 1, 1), UNITS,
                    buf!(dm_d, bt * r), buf!(wsc[3], bt * r),
                    buf!(out[10 + 6 * e].as_ref().unwrap(), r),
                    bt as u32, r as u32, acc,
                );
            }
            let dsil = alloc(bt * f);
            launch_mm(&client, &dz_d, &dn.u, &one, &dn.mu, &dsil, bt, r, f, r, 1, r, 1, bt * r, f * r, false, true, true, false, false);
            unsafe {
                super::kernels::silu_bwd_kernel::launch_unchecked::<f32, Cuda>(
                    &client, ew_cubes(bt * f), EW,
                    buf!(wsc[1], bt * f), buf!(dsil, bt * f), buf!(da2, bt * f),
                    (bt * f) as u32,
                );
            }

            // gate_up TSCT: da2 -> dx (accumulate)
            let dm_e = alloc(bt * r);
            launch_mm(&client, &da2, &gu.v, &gu.s, &gu.mv, &dm_e, bt, f, r, f, 1, r, 1, bt * f, f * r, false, false, true, false, false);
            let dz_e = alloc(bt * r);
            unsafe {
                super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                    &client, ew_cubes(bt * r), EW,
                    buf!(dm_e, bt * r), buf!(gu.s, r), buf!(dz_e, bt * r),
                    r as u32, (bt * r) as u32,
                );
            }
            if out[6 + 6 * e].is_none() {
                out[6 + 6 * e] = Some(alloc(d * r));
                out[8 + 6 * e] = Some(alloc(f * r));
            }
            launch_mm(&client, &pi.normed, &dz_e, &one, &one, out[6 + 6 * e].as_ref().unwrap(), d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, acc);
            launch_mm(&client, &da2, &wsc[0], &one, &one, out[8 + 6 * e].as_ref().unwrap(), f, bt, r, f, 1, r, 1, bt * f, bt * r, true, false, false, false, acc);
            if out[7 + 6 * e].is_none() {
                out[7 + 6 * e] = Some(alloc(r));
            }
            unsafe {
                super::kernels::sum_ds_kernel::launch_unchecked::<f32, Cuda>(
                    &client, CubeCount::Static(1, 1, 1), UNITS,
                    buf!(dm_e, bt * r), buf!(wsc[0], bt * r),
                    buf!(out[7 + 6 * e].as_ref().unwrap(), r),
                    bt as u32, r as u32, acc,
                );
            }
            // bm MUST be U's own absmean (see the dpre note above)
            launch_mm(&client, &dz_e, &gu.u, &one, &gu.mu, &dx, bt, r, d, r, 1, r, 1, bt * r, d * r, false, true, true, false, true);
        }
        dump(&client, Box::leak(format!("dx#{n}").into_boxed_str()), &dx, bt * d);

        // ---- controller: softmax jacobian -> dRaw; dWc, dCtrl
        // Wc is stored [2d, pad] (burn Linear Col layout): dWc = ctrl_inᵀ·draw,
        // dctrl = draw·Wcᵀ.
        unsafe {
            super::kernels::softmax_bwd_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(bt as u32, 1, 1),
                UNITS,
                buf!(pi.blend, bt * nexp),
                buf!(dblend, bt * nexp),
                buf!(draw, bt * pad),
                nexp as u32,
                pad as u32,
            );
        }
        if out[1].is_none() {
            out[1] = Some(alloc(2 * d * pad));
        }
        launch_mm(&client, &pi.ctrl_in, &draw, &one, &one, out[1].as_ref().unwrap(), 2 * d, bt, pad, 2 * d, 1, pad, 1, bt * 2 * d, bt * pad, true, false, false, false, acc);
        let dctrl = alloc(bt * 2 * d);
        launch_mm(&client, &draw, &st.wc, &one, &one, &dctrl, bt, pad, 2 * d, pad, 1, pad, 1, bt * pad, 2 * d * pad, false, true, false, false, false);
        dump(&client, Box::leak(format!("dctrl#{n}").into_boxed_str()), &dctrl, bt * 2 * d);

        // ---- halting recurrence: dLam -> dHaltpre -> dWh, dHalt_in
        let dhaltpre = alloc(b);
        let g_cur = alloc(b);
        unsafe {
            super::kernels::lam_bwd_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(1, 1, 1),
                UNITS,
                buf!(gc, lg_off + bt * v),
                buf!(pi.p, b),
                // ENTERING nh: per[n].nh is post-update (nh·(1−lam_n));
                // the recurrence and the dhaltpre factor need Π_{j<n}.
                buf!(if n == 0 { &st.nh0 } else { &st.per[n - 1].nh }, b),
                buf!(pi.lam, b),
                buf!(g_next, b),
                buf!(pi.ceb, b),
                buf!(dot, b),
                buf!(dhaltpre, b),
                buf!(g_cur, b),
                (pd_off + n * b) as u32,
                kl_off as u32,
                st.prior[n].ln(),
                b as u32,
                bt as u32,
                bn as u32,
            );
        }
        g_next = g_cur;
        dump(&client, Box::leak(format!("dhaltpre#{n}").into_boxed_str()), &dhaltpre, b);
        if out[5].is_none() {
            out[5] = Some(alloc(d));
        }
        launch_mm(&client, &dhaltpre, &pi.halt_in, &one, &one, out[5].as_ref().unwrap(), 1, b, d, 1, 1, d, 1, b, b * d, true, false, false, false, acc);
        let dhalt_in = alloc(b * d);
        launch_mm(&client, &dhaltpre, &st.wh, &one, &one, &dhalt_in, b, 1, d, 1, 1, d, 1, b, d, false, false, false, false, false);
        unsafe {
            super::kernels::haltmean_bwd_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(b as u32, 1, 1),
                UNITS,
                buf!(dhalt_in, b * d),
                buf!(dh_ctx, bt * d),
                t as u32,
                d as u32,
            );
        }

        // ---- RMSNorm backward (dx currently holds grad of normed_n)
        if out[2].is_none() {
            out[2] = Some(alloc(d));
        }
        unsafe {
            super::kernels::dg_kernel::launch_unchecked::<f32, Cuda>(
                &client, CubeCount::Static(1, 1, 1), UNITS,
                buf!(dx, bt * d), buf!(pi.h_ctx, bt * d), buf!(pi.inv, bt),
                buf!(out[2].as_ref().unwrap(), d),
                bt as u32, d as u32, acc,
            );
            super::kernels::rms_bwd_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(bt as u32, 1, 1),
                UNITS,
                buf!(dx, bt * d),
                buf!(st.g, d),
                buf!(pi.h_ctx, bt * d),
                buf!(pi.inv, bt),
                buf!(dh_ctx, bt * d),
                d as u32,
            );
        }

        // ---- controller split (after the RMSNorm backward: the x-half must
        // not leak into the d(normed) buffer rms_bwd consumed above). The
        // x-half accumulates straight into the x grad; the h_ctx half stays
        // in dh_ctx and becomes the carry into iteration n-1 (h_ctx_n =
        // h_{n-1} + ie_n is an identity wrt h_{n-1}).
        unsafe {
            super::kernels::cat_bwd_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(bt * d), EW,
                buf!(dctrl, bt * 2 * d),
                buf!(dh_ctx, bt * d),
                buf!(dxg, bt * d),
                d as u32,
                (bt * d) as u32,
            );
        }
        if n == 0 {
            // h_ctx_0 = x + ie_0: the loop input also inherits dh_ctx_0
            unsafe {
                super::kernels::add_kernel::launch_unchecked::<f32, Cuda>(
                    &client, ew_cubes(bt * d), EW,
                    buf!(dxg, bt * d),
                    buf!(dh_ctx, bt * d),
                    (bt * d) as u32,
                );
            }
            out[0] = Some(dxg.clone());
        } else {
            // handle clone (Arc): the buffer itself feeds ie_grad + dumps below
            dx_carry = Some(dh_ctx.clone());
        }

        // ---- iter_embed row n
        if out[3].is_none() {
            out[3] = Some(alloc(n_iter * d));
        }
        unsafe {
            super::kernels::ie_grad_kernel::launch_unchecked::<f32, Cuda>(
                &client, CubeCount::Static(1, 1, 1), UNITS,
                buf!(dh_ctx, bt * d),
                buf!(out[3].as_ref().unwrap(), n_iter * d),
                (n * d) as u32,
                bt as u32,
                d as u32,
            );
        }
        dump(&client, Box::leak(format!("dh_ctx#{n}").into_boxed_str()), &dh_ctx, bt * d);
    }
    // finish the lm_head dV column scale (raw accumulator -> master grad)
    unsafe {
        super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
            &client, ew_cubes(v * r), EW,
            buf!(dvl_raw, v * r), buf!(st.lm.s, r), buf!(dvl, v * r),
            r as u32, (v * r) as u32,
        );
    }
    // out_proj dV column scale (raw accumulator -> master grad; dV = (dYᵀ·Z)·diag(s),
    // the ·s folds out of the iteration sum like the lm_head dV above)
    {
        let dv_o = alloc(d * r);
        unsafe {
            super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(d * r), EW,
                buf!(out[26].as_ref().unwrap(), d * r), buf!(st.op.s, r), buf!(dv_o, d * r),
                r as u32, (d * r) as u32,
            );
        }
        out[26] = Some(dv_o);
    }
    // expert dV column scales (raw accumulators -> master grads; like the
    // lm_head dV, the ·s folds out of the iteration sum)
    for e in 0..nexp {
        let [gu, dn] = &st.experts[e];
        let dv_e = alloc(f * r);
        unsafe {
            super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(f * r), EW,
                buf!(out[8 + 6 * e].as_ref().unwrap(), f * r), buf!(gu.s, r), buf!(dv_e, f * r),
                r as u32, (f * r) as u32,
            );
        }
        out[8 + 6 * e] = Some(dv_e);
        let dv_d = alloc(d * r);
        unsafe {
            super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(d * r), EW,
                buf!(out[11 + 6 * e].as_ref().unwrap(), d * r), buf!(dn.s, r), buf!(dv_d, d * r),
                r as u32, (d * r) as u32,
            );
        }
        out[11 + 6 * e] = Some(dv_d);
    }
    // expert weight-grad dumps for the f64 bisects (post-scale, registered values)
    #[cfg(test)]
    for e in 0..nexp {
        if std::env::var("DM_FUSED_BWD_DEBUG").is_ok() {
            for (tag, buf, len) in [
                (format!("du_d{e}"), out[9 + 6 * e].as_ref().unwrap(), f * r),
                (format!("ds_d{e}"), out[10 + 6 * e].as_ref().unwrap(), r),
                (format!("dv_d{e}"), out[11 + 6 * e].as_ref().unwrap(), d * r),
                (format!("du_e{e}"), out[6 + 6 * e].as_ref().unwrap(), d * r),
                (format!("ds_e{e}"), out[7 + 6 * e].as_ref().unwrap(), r),
                (format!("dv_e{e}"), out[8 + 6 * e].as_ref().unwrap(), f * r),
            ] {
                dump(&client, Box::leak(tag.into_boxed_str()), buf, len);
            }
        }
    }
    out[27] = Some(dul);
    out[28] = Some(dsl);
    out[29] = Some(dvl);
    // op/lm weight-grad dumps for the f64 bisects (post-scale)
    #[cfg(test)]
    if std::env::var("DM_FUSED_BWD_DEBUG").is_ok() {
        for (tag, buf, len) in [
            (format!("du_op"), out[24].as_ref().unwrap(), d * r),
            (format!("ds_op"), out[25].as_ref().unwrap(), r),
            (format!("dv_op"), out[26].as_ref().unwrap(), d * r),
            (format!("du_lm"), out[27].as_ref().unwrap(), d * r),
            (format!("ds_lm"), out[28].as_ref().unwrap(), r),
            (format!("dv_lm"), out[29].as_ref().unwrap(), v * r),
        ] {
            dump(&client, Box::leak(tag.into_boxed_str()), buf, len);
        }
    }
    dump(&client, "dg", out[2].as_ref().unwrap(), d);
    dump(&client, "die", out[3].as_ref().unwrap(), n_iter * d);
    dump(&client, "dwh", out[5].as_ref().unwrap(), d);
    dump(&client, "drs", out[4].as_ref().unwrap(), 1);
    dump(&client, "dwc", out[1].as_ref().unwrap(), 2 * d * pad);
    dump(&client, "dgf", out[30].as_ref().unwrap(), d);
    dump(&client, "dxg", out[0].as_ref().unwrap(), bt * d);

    // ---- register every parent grad.
    // Registered primitives must carry the parent's rank/shapes: burn-optim
    // consumes them through shape-checked tensor ops against the params. The
    // grads were computed into flat dense buffers, so the wrap is a free
    // stride relabel (1D -> 2D on a dense buffer never copies).
    let dims: Vec<Option<[usize; 2]>> = {
        let mut dm: Vec<Option<[usize; 2]>> = vec![None; ops.parents.len()];
        dm[0] = Some([bt, d]);
        dm[1] = Some([2 * d, pad]); // burn Linear Col layout
        dm[2] = None; // norm g: [d]
        dm[3] = Some([n_iter, d]);
        dm[4] = None; // residual scale: [1]
        dm[5] = Some([d, 1]); // halt head weight, Col layout
        for e in 0..nexp {
            dm[6 + 6 * e] = Some([d, r]);
            dm[7 + 6 * e] = None; // s: [r]
            dm[8 + 6 * e] = Some([f, r]);
            dm[9 + 6 * e] = Some([f, r]);
            dm[10 + 6 * e] = None;
            dm[11 + 6 * e] = Some([d, r]);
        }
        let (p_out, p_lm) = (6 + 6 * nexp, 9 + 6 * nexp); // out_proj, lm_head u/s/v
        dm[p_out] = Some([d, r]);
        dm[p_out + 2] = Some([d, r]);
        dm[p_lm] = Some([d, r]);
        dm[p_lm + 2] = Some([v, r]);
        dm[30] = None; // final norm g: [d]
        dm
    };
    // Capture every grad to host memory here (the buffers were just verified
    // by the f64 bisects) and register fresh device tensors rebuilt from the
    // captured bytes. Reading a registered reshape back through the grads map
    // has been observed to hand back foreign bytes for the expert slots
    // (gu.u: 49152/49152 mismatched vs this same buffer read moments earlier,
    // measured 2026-09-13), so no GPU reads are deferred past this point.
    let mut snap: Vec<Option<(Vec<f32>, Option<[usize; 2]>)>> = vec![None; out.len()];
    for (i, g) in out.iter().enumerate() {
        if let Some(gt) = g {
            let n = match dims[i] {
                Some([r0, r1]) => r0 * r1,
                None => match i {
                    4 => 1,
                    30 | 2 => d,
                    _ => r,
                },
            };
            let bytes = client.read(vec![gt.handle.clone()]).remove(0);
            assert!(bytes.len() >= n * 4, "grad readback short: {} < {}", bytes.len(), n * 4);
            snap[i] = Some((
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n) }.to_vec(),
                dims[i],
            ));
        }
    }
    for (i, snap) in snap.into_iter().enumerate() {
        if let (Some(node), Some((mut vals, dims_i))) = (&ops.parents[i], snap) {
            let gt = match dims_i {
                Some([r0, r1]) => {
                    let n = vals.len();
                    Tensor::<2>::from_data(
                        TensorData::new(std::mem::take(&mut vals), [r0, r1]),
                        &dev,
                    )
                    .try_into_primitive::<CB>()
                    .expect("grad rebuild")
                }
                None => {
                    let n = vals.len();
                    Tensor::<1>::from_data(
                        TensorData::new(std::mem::take(&mut vals), [n]),
                        &dev,
                    )
                    .try_into_primitive::<CB>()
                    .expect("grad wrap")
                }
            };
            grads.register::<CB>(node.id, gt);
        }
    }
    super::sync(&client);
}

/// M4 arms path: run a reference `DormouseModel` forward on Autodiff<CB> with
/// the same `x`/`hashed_ids`/`targets` and the same weights (taken from the
/// stored `PonderInputs`), then weight the four outputs by the upstream flat
/// grad and backward. This reuses the exact burn-kda/burn-msa adjoint kernels
/// (via Autodiff<Cuda>) for the attention arms, so the grads are correct
/// without hand-written KDA/MSA kernels. The cost is one extra forward+backward
/// on the tiny test shapes (b=2 t=16) and one per step in training — still a
/// single outer node, so CPU graph construction stays at 1 node/step.
pub(super) fn ponder_backward_arms(
    ops: Ops<ArmsState, 31>,
    grads: &mut Gradients,
    _cp: &mut Checkpointer,
) {
    use burn::backend::DispatchKindConversion;
    use burn::module::Module;
    use burn::tensor::{Device, Int, Tensor, TensorData};

    let st = ops.state;
    let gfull = Tensor::<1>::from_primitive::<CB>(grads.consume::<CB>(&ops.node));
    // gfull is on CB bare (flat [rec|pd|kl|logits]); read it to host
    let g_data: Vec<f32> = gfull.clone().into_data().try_to_vec::<f32>().unwrap();
    let bn = st.b * st.n_iter;
    let bt = st.bt;
    let v = st.v;
    assert!(g_data.len() == 2 + bn + bt * v, "gfull len mismatch");
    let g_rec = g_data[0];
    let g_pd = g_data[1..1 + bn].to_vec();
    let g_kl = g_data[1 + bn];
    let g_logits = g_data[2 + bn..].to_vec();

    // GPU path: inner model on Autodiff<Cuda> (CubeBackend) - 100% GPU.
    // Reuses the hand-written KDA/MSA adjoints (gdn2_chunk_intra/inter,
    // msa_backward) via Autodiff<Cuda> - no NdArray fallback.
    type CuAd = crate::fused::CAd;
    let dev_cu = Device::cuda(0).autodiff();
    let cfg = st.cfg.clone();
    let mut model_cu = crate::model::DormouseModel::new(&cfg, &dev_cu);
    // Copy the 31 parents' weights from the outer CUDA tensors to inner Cuda
    // (via host data, numerical values only — NodeIds differ, but we will
    // use the numerical grads to register for the outer NodeIds)
    let x_cu = Tensor::<3>::from_data(st.inp.x.clone().into_data(), &dev_cu).require_grad();
    let controller_w_cu =
        Tensor::<2>::from_data(st.inp.controller_w.clone().into_data(), &dev_cu).require_grad();
    let norm_g_cu = Tensor::<1>::from_data(st.inp.norm_g.clone().into_data(), &dev_cu).require_grad();
    let iter_embed_cu =
        Tensor::<2>::from_data(st.inp.iter_embed.clone().into_data(), &dev_cu).require_grad();
    let residual_scale_cu =
        Tensor::<1>::from_data(st.inp.residual_scale.clone().into_data(), &dev_cu).require_grad();
    let halt_w_cu = Tensor::<2>::from_data(st.inp.halt_w.clone().into_data(), &dev_cu).require_grad();
    let mut experts_cu: Vec<[crate::fused::Fac; 2]> = Vec::new();
    for e in &st.inp.experts {
        let gu = crate::fused::Fac {
            u: Tensor::<2>::from_data(e[0].u.clone().into_data(), &dev_cu).require_grad(),
            s: Tensor::<1>::from_data(e[0].s.clone().into_data(), &dev_cu).require_grad(),
            v: Tensor::<2>::from_data(e[0].v.clone().into_data(), &dev_cu).require_grad(),
        };
        let dn = crate::fused::Fac {
            u: Tensor::<2>::from_data(e[1].u.clone().into_data(), &dev_cu).require_grad(),
            s: Tensor::<1>::from_data(e[1].s.clone().into_data(), &dev_cu).require_grad(),
            v: Tensor::<2>::from_data(e[1].v.clone().into_data(), &dev_cu).require_grad(),
        };
        experts_cu.push([gu, dn]);
    }
    let out_proj_cu = crate::fused::Fac {
        u: Tensor::<2>::from_data(st.inp.out_proj.u.clone().into_data(), &dev_cu).require_grad(),
        s: Tensor::<1>::from_data(st.inp.out_proj.s.clone().into_data(), &dev_cu).require_grad(),
        v: Tensor::<2>::from_data(st.inp.out_proj.v.clone().into_data(), &dev_cu).require_grad(),
    };
    let lm_head_cu = crate::fused::Fac {
        u: Tensor::<2>::from_data(st.inp.lm_head.u.clone().into_data(), &dev_cu).require_grad(),
        s: Tensor::<1>::from_data(st.inp.lm_head.s.clone().into_data(), &dev_cu).require_grad(),
        v: Tensor::<2>::from_data(st.inp.lm_head.v.clone().into_data(), &dev_cu).require_grad(),
    };
    let final_norm_g_cu =
        Tensor::<1>::from_data(st.inp.final_norm_g.clone().into_data(), &dev_cu).require_grad();
    // Build model_cu's loop_block from experts_cu etc
    {
        let lb = &mut model_cu.loop_block;
        lb.controller.weight = burn::module::Param::from_tensor(controller_w_cu.clone());
        lb.norm.weight = burn::module::Param::from_tensor(norm_g_cu.clone());
        lb.iter_embed = burn::module::Param::from_tensor(iter_embed_cu.clone());
        lb.residual_scale = burn::module::Param::from_tensor(residual_scale_cu.clone());
        lb.halt_head.weight = burn::module::Param::from_tensor(halt_w_cu.clone());
        for (i, e) in experts_cu.iter().enumerate() {
            let dst = &mut lb.expert_ffns[i];
            if let crate::param::LinearLikeInner::Tsct(l) = &mut dst.gate_up.inner {
                l.u = burn::module::Param::from_tensor(e[0].u.clone());
                l.s = burn::module::Param::from_tensor(e[0].s.clone());
                l.v = burn::module::Param::from_tensor(e[0].v.clone());
            }
            if let crate::param::LinearLikeInner::Tsct(l) = &mut dst.down.inner {
                l.u = burn::module::Param::from_tensor(e[1].u.clone());
                l.s = burn::module::Param::from_tensor(e[1].s.clone());
                l.v = burn::module::Param::from_tensor(e[1].v.clone());
            }
        }
        if let crate::param::LinearLikeInner::Tsct(l) = &mut lb.out_proj.inner {
            l.u = burn::module::Param::from_tensor(out_proj_cu.u.clone());
            l.s = burn::module::Param::from_tensor(out_proj_cu.s.clone());
            l.v = burn::module::Param::from_tensor(out_proj_cu.v.clone());
        }
        model_cu.norm.weight = burn::module::Param::from_tensor(final_norm_g_cu.clone());
        if let crate::param::LinearLikeInner::Tsct(l) = &mut model_cu.lm_head.inner {
            l.u = burn::module::Param::from_tensor(lm_head_cu.u.clone());
            l.s = burn::module::Param::from_tensor(lm_head_cu.s.clone());
            l.v = burn::module::Param::from_tensor(lm_head_cu.v.clone());
        }
        if let Some(lb_bytes) = st.inp.loop_block_bytes.clone() {
            let mut lb2 = crate::loop_block::LoopBlock::new(&cfg, &dev_cu);
            let rec = burn::store::ModuleRecord::from_bytes(burn::tensor::Bytes::from_bytes_vec(
                lb_bytes,
            ))
            .expect("loop_block record");
            lb2 = lb2.load_record(rec);
            // Keep shared_attn/engram as constants (not tracked)
            lb.shared_attn = lb2.shared_attn;
            lb.engram = lb2.engram;
            lb.use_kda = lb2.use_kda;
            lb.use_msa = lb2.use_msa;
            lb.use_engram = lb2.use_engram;
        }
        model_cu.ponder_prior = st.inp.ponder_prior;
        model_cu.ponder_beta = cfg.ponder_beta;
    }
    let hashed_cu = st.inp.hashed_ids.as_ref().map(|h| {
        Tensor::<3, Int>::from_data(h.clone().into_data(), &dev_cu)
    });
    let targets_cu = Tensor::<2, Int>::from_data(st.inp.targets.clone().into_data(), &dev_cu);
    let tgt_2d = targets_cu.clone().reshape([bt, 1]);
    let (out_acc_cu, rec_cu, p_dist_cu, _) = model_cu.loop_block.forward_full_state::<CuAd>(
        x_cu.clone(),
        hashed_cu.clone(),
        None,
        None,
        Some(tgt_2d),
        &model_cu.lm_head,
    );
    let h_cu = model_cu.norm.forward(out_acc_cu.clone());
    let logits_cu = model_cu
        .lm_head
        .forward::<CuAd>(h_cu.reshape([bt, cfg.d_model]))
        .reshape([st.b, st.t, v]);
    let kl_cu = {
        let n = p_dist_cu.dims()[1];
        let b_ = p_dist_cu.dims()[0];
        let dev = p_dist_cu.device();
        let mut prior = Vec::with_capacity(n);
        let mut mass = 1.0f32;
        let mut total = 0.0f32;
        for _ in 0..n {
            let prob = cfg.ponder_prior * mass;
            prior.push(prob);
            total += prob;
            mass *= 1.0 - cfg.ponder_prior;
        }
        let inv = 1.0 / total;
        let prior_v: Vec<f32> = prior.iter().map(|x| x * inv).collect();
        let prior_t = Tensor::<1>::from_data(TensorData::new(prior_v, [n]), &dev).log();
        let log_p = p_dist_cu.clone().log();
        let term = p_dist_cu.clone() * (log_p - prior_t.unsqueeze_dim::<2>(0));
        term.sum_dim(1).sum_dim(0).reshape([1]).div_scalar((b_ * n) as f32)
    };

    // g_* are on Cuda, create loss on Cuda as well
    let dev_cu_clone = dev_cu.clone();
    let g_pd_t = Tensor::<2>::from_data(TensorData::new(g_pd, [st.b, st.n_iter]), &dev_cu_clone);
    let g_logits_t = Tensor::<3>::from_data(
        TensorData::new(g_logits, [st.b, st.t, v]),
        &dev_cu_clone,
    );
    let loss = rec_cu.clone().mul_scalar(g_rec)
        + (p_dist_cu.clone() * g_pd_t).sum()
        + kl_cu.clone().mul_scalar(g_kl)
        + (logits_cu.clone() * g_logits_t).sum();

    let grads_cu = loss.backward();

    // Map the 31 parents' grads from the inner Cuda graph to the outer CUDA grads
    // (numerical values only, NodeIds differ, so we copy via host Vec)
    let mut out: Vec<Option<CubeTensor>> = vec![None; 31];
    // Helper to get grad Vec on Cuda and convert to CubeTensor on CUDA
    let to_cube = |v: Vec<f32>, dims: Option<[usize; 2]>, dev: &Device| -> CubeTensor {
        match dims {
            Some([r0, r1]) => {
                let t = Tensor::<2>::from_data(TensorData::new(v, [r0, r1]), dev);
                t.try_into_primitive::<CB>().expect("cuda grad")
            }
            None => {
                let len = v.len();
                let t = Tensor::<1>::from_data(TensorData::new(v, [len]), dev);
                t.try_into_primitive::<CB>().expect("cuda grad")
            }
        }
    };
    let dev_cuda = Device::cuda(0);
    // 0: x
    if let Some(g) = x_cu.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[0] = Some(to_cube(v, Some([bt, st.d]), &dev_cuda));
    }
    if let Some(g) = controller_w_cu.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[1] = Some(to_cube(v, Some([2 * st.d, st.inp.controller_w.dims()[1]]), &dev_cuda));
    }
    if let Some(g) = norm_g_cu.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[2] = Some(to_cube(v, None, &dev_cuda));
    }
    if let Some(g) = iter_embed_cu.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[3] = Some(to_cube(v, Some([st.n_iter, st.d]), &dev_cuda));
    }
    if let Some(g) = residual_scale_cu.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[4] = Some(to_cube(v, None, &dev_cuda));
    }
    if let Some(g) = halt_w_cu.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[5] = Some(to_cube(v, Some([st.d, 1]), &dev_cuda));
    }
    for (ei, e) in experts_cu.iter().enumerate() {
        if let Some(g) = e[0].u.grad(&grads_cu) {
            let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
            out[6 + 6 * ei] = Some(to_cube(v, Some([st.d, e[0].u.dims()[1]]), &dev_cuda));
        }
        if let Some(g) = e[0].s.grad(&grads_cu) {
            let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
            out[7 + 6 * ei] = Some(to_cube(v, None, &dev_cuda));
        }
        if let Some(g) = e[0].v.grad(&grads_cu) {
            let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
            out[8 + 6 * ei] = Some(to_cube(
                v,
                Some([e[0].v.dims()[0], e[0].u.dims()[1]]),
                &dev_cuda,
            ));
        }
        if let Some(g) = e[1].u.grad(&grads_cu) {
            let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
            out[9 + 6 * ei] = Some(to_cube(
                v,
                Some([e[1].u.dims()[0], e[1].u.dims()[1]]),
                &dev_cuda,
            ));
        }
        if let Some(g) = e[1].s.grad(&grads_cu) {
            let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
            out[10 + 6 * ei] = Some(to_cube(v, None, &dev_cuda));
        }
        if let Some(g) = e[1].v.grad(&grads_cu) {
            let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
            out[11 + 6 * ei] = Some(to_cube(v, Some([st.d, e[1].u.dims()[1]]), &dev_cuda));
        }
    }
    if let Some(g) = out_proj_cu.u.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[24] = Some(to_cube(v, Some([st.d, out_proj_cu.u.dims()[1]]), &dev_cuda));
    }
    if let Some(g) = out_proj_cu.s.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[25] = Some(to_cube(v, None, &dev_cuda));
    }
    if let Some(g) = out_proj_cu.v.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[26] = Some(to_cube(v, Some([st.d, out_proj_cu.u.dims()[1]]), &dev_cuda));
    }
    if let Some(g) = lm_head_cu.u.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[27] = Some(to_cube(v, Some([st.d, lm_head_cu.u.dims()[1]]), &dev_cuda));
    }
    if let Some(g) = lm_head_cu.s.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[28] = Some(to_cube(v, None, &dev_cuda));
    }
    if let Some(g) = lm_head_cu.v.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[29] = Some(to_cube(v, Some([st.v, lm_head_cu.u.dims()[1]]), &dev_cuda));
    }
    if let Some(g) = final_norm_g_cu.grad(&grads_cu) {
        let v: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
        out[30] = Some(to_cube(v, None, &dev_cuda));
    }

    // Register (same snapshot logic as ponder_backward)
    let dims: Vec<Option<[usize; 2]>> = {
        let mut dm: Vec<Option<[usize; 2]>> = vec![None; 31];
        dm[0] = Some([bt, st.d]);
        dm[1] = Some([2 * st.d, st.inp.controller_w.dims()[1]]);
        dm[2] = None;
        dm[3] = Some([st.n_iter, st.d]);
        dm[4] = None;
        dm[5] = Some([st.d, 1]);
        for e in 0..3 {
            dm[6 + 6 * e] = Some([st.d, st.inp.experts[e][0].u.dims()[1]]);
            dm[7 + 6 * e] = None;
            dm[8 + 6 * e] = Some([st.inp.experts[e][0].v.dims()[0], st.inp.experts[e][0].u.dims()[1]]);
            dm[9 + 6 * e] = Some([st.inp.experts[e][1].u.dims()[0], st.inp.experts[e][1].u.dims()[1]]);
            dm[10 + 6 * e] = None;
            dm[11 + 6 * e] = Some([st.d, st.inp.experts[e][1].u.dims()[1]]);
        }
        dm[24] = Some([st.d, st.inp.out_proj.u.dims()[1]]);
        dm[26] = Some([st.d, st.inp.out_proj.u.dims()[1]]);
        dm[27] = Some([st.d, st.inp.lm_head.u.dims()[1]]);
        dm[29] = Some([st.v, st.inp.lm_head.u.dims()[1]]);
        dm[30] = None;
        dm
    };
    let dev = Device::cuda(0);
    // Use the same snapshot pattern as the non-arms path: read each grad to
    // host, then rebuild fresh device tensors for registration (avoids the
    // foreign-bytes bug observed with direct register of reshaped grads).
    let client = {
        let dummy = Tensor::<1>::from_data(TensorData::new(vec![0.0f32], [1]), &Device::cuda(0));
        dummy.try_into_primitive::<CB>().unwrap().client.clone()
    };
    let mut snap: Vec<Option<(Vec<f32>, Option<[usize; 2]>)>> = vec![None; 31];
    for (i, g) in out.iter().enumerate() {
        if let Some(gt) = g {
            let n = match dims[i] {
                Some([r0, r1]) => r0 * r1,
                None => match i {
                    4 => 1,
                    30 | 2 => st.d,
                    _ => st.inp.experts[0][0].u.dims()[1],
                },
            };
            let bytes = gt.client.read(vec![gt.handle.clone()]).remove(0);
            assert!(bytes.len() >= n * 4);
            snap[i] = Some((
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n) }.to_vec(),
                dims[i],
            ));
        }
    }
    for (i, s) in snap.into_iter().enumerate() {
        if let (Some(node), Some((mut vals, dims_i))) = (&ops.parents[i], s) {
            let gt = match dims_i {
                Some([r0, r1]) => {
                    let data = std::mem::take(&mut vals);
                    Tensor::<2>::from_data(TensorData::new(data, [r0, r1]), &dev)
                        .try_into_primitive::<CB>()
                        .expect("grad rebuild")
                }
                None => {
                    let data = std::mem::take(&mut vals);
                    let len = data.len();
                    Tensor::<1>::from_data(TensorData::new(data, [len]), &dev)
                        .try_into_primitive::<CB>()
                        .expect("grad wrap")
                }
            };
            grads.register::<CB>(node.id, gt);
        }
    }
    super::sync(&client);
}
