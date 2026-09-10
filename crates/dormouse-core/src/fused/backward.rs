//! `ponder_backward` - the hand-written backward for the fused single-iteration
//! Ponder step. Implements the plan's "Backward math checklist":
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
//! - PonderNet at N=1: `dLam = dP + dRec·CE/(b·t)`; CE:
//!   `dLogits = −(softmax − onehot(tgt))·dRec·lam/(b·t)`
//!
//! The upstream grad arrives on the single op node as one flat buffer with the
//! output layout `[rec | lam (b) | out_acc (b·t·d)]`.

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::Ops;
use burn::tensor::Tensor;
use cubecl::prelude::*;

use super::{
    cube_of1, dense, empty1, ew_cubes, launch_mm, zeros_raw, CubeTensor, FacC, CB, Cuda, UNITS,
    EW,
};

/// Saved forward state for the fused Ponder step (see `mod.rs`).
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
    pub max_iter: usize,
    pub eps: f32,
    pub dev: burn::tensor::Device,
    // masters
    pub x: CubeTensor,
    pub tgt: CubeTensor,
    pub wc: CubeTensor,
    pub g: CubeTensor,
    pub ie: CubeTensor,
    pub rs: CubeTensor,
    pub wh: CubeTensor,
    pub experts: Vec<[FacC; 2]>,
    pub op: FacC,
    pub lm: FacC,
    // forward workspace (bridged twins keep buffers alive)
    pub h_ctx: Tensor<1>,
    pub normed: Tensor<1>,
    pub inv: Tensor<1>,
    pub ctrl_in: Tensor<1>,
    pub raw: Tensor<1>,
    pub w_ffn: Tensor<1>,
    pub blend: Tensor<1>,
    pub ffn: Tensor<1>,
    pub y: Tensor<1>,
    pub h: Tensor<1>,
    pub z_o: Tensor<1>,
    pub step_out: Tensor<1>,
    pub z_l: Tensor<1>,
    pub logits: Tensor<1>,
    pub halt_in: Tensor<1>,
    pub ce: Tensor<1>,
    pub ceb: Tensor<1>,
    pub ws: Vec<[CubeTensor; 5]>,
    pub flat_out: CubeTensor,
    pub keep2: Vec<Tensor<2>>,
    pub keep1: Vec<Tensor<1>>,
}

macro_rules! buf {
    ($h:expr, $len:expr) => {
        BufferArg::from_raw_parts($h.handle.clone(), $len)
    };
}

thread_local! {
    /// Buffer-level backward dumps (DM_FUSED_BWD_DEBUG=1): (name, values).
    /// Consumed by the gradcheck bisect test; empty unless enabled.
    pub static BWD_DUMP: std::cell::RefCell<Vec<(&'static str, Vec<f32>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn dump(client: &ComputeClient<Cuda>, name: &'static str, c: &CubeTensor, _n: usize) {
    if std::env::var("DM_FUSED_BWD_DEBUG").is_ok() {
        let bytes = client.read(vec![c.handle.clone()]).remove(0);
        let v =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
                .to_vec();
        BWD_DUMP.with(|d| d.borrow_mut().push((name, v)));
    }
}

/// Run the backward kernels and register every weight/input gradient.
pub(super) fn ponder_backward(
    ops: Ops<PonderState, 30>,
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
    let (b, t, d, f, r, v, nexp, pad, bt) = (st.b, st.t, st.d, st.f, st.r, st.v, st.nexp, st.pad, st.bt);
    let (lam_off, out_off) = (1usize, 1 + b);
    let c1 = |x: &Tensor<1>| cube_of1(x).expect("cuda tensor");
    let one = c1(&st.ce); // dummy mm operand (any valid buffer)

    let h_ctx = c1(&st.h_ctx);
    let normed = c1(&st.normed);
    let inv = c1(&st.inv);
    let ctrl_in = c1(&st.ctrl_in);
    let w_ffn = c1(&st.w_ffn);
    let blend = c1(&st.blend);
    let ffn = c1(&st.ffn);
    let y = c1(&st.y);
    let h = c1(&st.h);
    let z_o = c1(&st.z_o);
    let step_out = c1(&st.step_out);
    let z_l = c1(&st.z_l);
    let logits = c1(&st.logits);
    let halt_in = c1(&st.halt_in);
    let ceb = c1(&st.ceb);

    let mut keep: Vec<Tensor<1>> = Vec::new();
    let mut out: Vec<Option<CubeTensor>> = vec![None; 30];
    let alloc = |n: usize| empty1(&dev, n);

    // ---- per-step CE: dLogits
    let (dl_k, dlogits) = alloc(bt * v);
    unsafe {
        super::kernels::dlogits_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(bt as u32, 1, 1),
            UNITS,
            buf!(logits, bt * v),
            buf!(st.tgt, bt),
            buf!(st.flat_out, 1 + b + bt * d),
            buf!(gc, 1 + b + bt * d),
            buf!(dlogits, bt * v),
            lam_off as u32,
            t as u32,
            v as u32,
            bt as u32,
        );
    }

    // ---- lm_head TSCT backward
    dump(&client, "dlogits", &dlogits, bt * v);
    let (dm_k, dm_l) = alloc(bt * r);
    launch_mm(&client, &dlogits, &st.lm.v, &st.lm.s, &st.lm.mv, &dm_l, bt, v, r, v, 1, r, 1, bt * v, v * r, false, false, true, false, false);
    dump(&client, "dm_l", &dm_l, bt * r);
    let (dzl_k, dz_l) = alloc(bt * r);
    unsafe {
        super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
            &client, ew_cubes(bt * r), EW,
            buf!(dm_l, bt * r), buf!(st.lm.s, r), buf!(dz_l, bt * r),
            r as u32, (bt * r) as u32,
        );
    }
    let (dul_k, dul) = alloc(d * r);
    launch_mm(&client, &step_out, &dz_l, &one, &one, &dul, d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, false);
    let (dvlr_k, dvl_raw) = alloc(v * r);
    launch_mm(&client, &dlogits, &z_l, &one, &one, &dvl_raw, v, bt, r, v, 1, r, 1, bt * v, bt * r, true, false, false, false, false);
    let (dvl_k, dvl) = alloc(v * r);
    unsafe {
        super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
            &client, ew_cubes(v * r), EW,
            buf!(dvl_raw, v * r), buf!(st.lm.s, r), buf!(dvl, v * r),
            r as u32, (v * r) as u32,
        );
    }
    let (dsl_k, dsl) = alloc(r);
    unsafe {
        super::kernels::sum_ds_kernel::launch_unchecked::<f32, Cuda>(
            &client, CubeCount::Static(1, 1, 1), UNITS,
            buf!(dm_l, bt * r), buf!(z_l, bt * r), buf!(dsl, r),
            bt as u32, r as u32,
        );
    }
    let (ds_k, dstep) = alloc(bt * d);
    // bm MUST be U's own absmean: tern_b=true reads bm[0] as the scale (the
    // `one` dummy here ternarized U with a garbage scale - measured as the
    // dstep corruption in the buffer bisect).
    launch_mm(&client, &dz_l, &st.lm.u, &one, &st.lm.mu, &dstep, bt, r, d, r, 1, r, 1, bt * r, d * r, false, true, true, false, false);
    dump(&client, "dz_l", &dz_l, bt * r);
    dump(&client, "dsl", &dsl, r);
    unsafe {
        super::kernels::dso_kernel::launch_unchecked::<f32, Cuda>(
            &client, ew_cubes(bt * d), EW,
            buf!(gc, 1 + b + bt * d),
            buf!(st.flat_out, 1 + b + bt * d),
            out_off as u32,
            buf!(dstep, bt * d),
            lam_off as u32,
            t as u32, d as u32, (bt * d) as u32,
        );
    }
    out[27] = Some(dul);
    out[28] = Some(dsl);
    out[29] = Some(dvl);
    keep.extend([dl_k, dm_k, dzl_k, dul_k, dvlr_k, dvl_k, dsl_k, ds_k]);

    // ---- out_proj TSCT backward (dstep -> dh_flat)
    let (dmo_k, dmo) = alloc(bt * r);
    launch_mm(&client, &dstep, &st.op.v, &st.op.s, &st.op.mv, &dmo, bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
    let (dzo_k, dz_o) = alloc(bt * r);
    unsafe {
        super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
            &client, ew_cubes(bt * r), EW,
            buf!(dmo, bt * r), buf!(st.op.s, r), buf!(dz_o, bt * r),
            r as u32, (bt * r) as u32,
        );
    }
    let (duo_k, duo) = alloc(d * r);
    launch_mm(&client, &h, &dz_o, &one, &one, &duo, d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, false);
    let (dvor_k, dvo_raw) = alloc(d * r);
    launch_mm(&client, &dstep, &z_o, &one, &one, &dvo_raw, d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, false);
    let (dvo_k, dvo) = alloc(d * r);
    unsafe {
        super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
            &client, ew_cubes(d * r), EW,
            buf!(dvo_raw, d * r), buf!(st.op.s, r), buf!(dvo, d * r),
            r as u32, (d * r) as u32,
        );
    }
    let (dso_k, dso) = alloc(r);
    unsafe {
        super::kernels::sum_ds_kernel::launch_unchecked::<f32, Cuda>(
            &client, CubeCount::Static(1, 1, 1), UNITS,
            buf!(dmo, bt * r), buf!(z_o, bt * r), buf!(dso, r),
            bt as u32, r as u32,
        );
    }
    let (dhf_k, dh_flat) = alloc(bt * d);
    launch_mm(&client, &dz_o, &st.op.u, &one, &st.op.mu, &dh_flat, bt, r, d, r, 1, r, 1, bt * r, d * r, false, true, true, false, false);
    out[24] = Some(duo);
    out[25] = Some(dso);
    out[26] = Some(dvo);
    dump(&client, "dzo", &dz_o, bt * r);
    dump(&client, "dstep", &dstep, bt * d);
    keep.extend([dmo_k, dzo_k, duo_k, dvor_k, dvo_k, dso_k, dhf_k]);
    dump(&client, "dh_flat", &dh_flat, bt * d);

    // ---- residual split: dy, dh_ctx += dh_flat, drs_part, dwffn_part
    let (dy_k, dy) = alloc(bt * d);
    let dh_ctx = zeros_raw(&dev, &client, bt * d);
    let (drsp_k, drs_part) = alloc(bt);
    let (dwfp_k, dwffn_part) = alloc(bt);
    unsafe {
        super::kernels::residual_bwd_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(bt as u32, 1, 1),
            UNITS,
            buf!(dh_flat, bt * d),
            buf!(st.rs, 1),
            buf!(y, bt * d),
            buf!(ffn, bt * d),
            buf!(dy, bt * d),
            buf!(dh_ctx, bt * d),
            buf!(drs_part, bt),
            buf!(dwffn_part, bt),
            d as u32,
        );
    }
    keep.extend([dy_k, drsp_k, dwfp_k]);

    // ---- ffn gate row (w_ffn jacobian) -> dRaw[:,2], dFfn
    let (dffn_k, dffn) = alloc(bt * d);
    let draw = zeros_raw(&dev, &client, bt * pad);
    unsafe {
        super::kernels::ffnrow_bwd_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(bt as u32, 1, 1),
            UNITS,
            buf!(dy, bt * d),
            buf!(w_ffn, bt),
            buf!(dwffn_part, bt),
            buf!(dffn, bt * d),
            buf!(draw, bt * pad),
            pad as u32,
            d as u32,
        );
    }
    keep.extend([dffn_k]);
    dump(&client, "dy", &dy, bt * d);
    dump(&client, "dffn", &dffn, bt * d);

    // ---- per-expert backward
    let (_dbl_k, dblend) = alloc(bt * nexp);
    let (_doe_k, dout_e) = alloc(bt * d);
    let (_da2_k, da2) = alloc(bt * f);
    let dx = zeros_raw(&dev, &client, bt * d);
    for e in 0..nexp {
        let [gu, dn] = &st.experts[e];
        let wsc = &st.ws[e];
        unsafe {
            super::kernels::dblend_kernel::launch_unchecked::<f32, Cuda>(
                &client,
                CubeCount::Static(bt as u32, 1, 1),
                UNITS,
                buf!(dffn, bt * d),
                buf!(wsc[4], bt * d),
                buf!(blend, bt * nexp),
                buf!(dout_e, bt * d),
                buf!(dblend, bt * nexp),
                e as u32,
                nexp as u32,
                d as u32,
            );
        }
        // down TSCT: dOut_e -> dsil
        let (dmd_k, dm_d) = alloc(bt * r);
        launch_mm(&client, &dout_e, &dn.v, &dn.s, &dn.mv, &dm_d, bt, d, r, d, 1, r, 1, bt * d, d * r, false, false, true, false, false);
        let (dzd_k, dz_d) = alloc(bt * r);
        unsafe {
            super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(bt * r), EW,
                buf!(dm_d, bt * r), buf!(dn.s, r), buf!(dz_d, bt * r),
                r as u32, (bt * r) as u32,
            );
        }
        let (dud_k, du_d) = alloc(f * r);
        launch_mm(&client, &wsc[2], &dz_d, &one, &one, &du_d, f, bt, r, f, 1, r, 1, bt * f, bt * r, true, false, false, false, false);
        let (dvd_k, dv_d_raw) = alloc(d * r);
        launch_mm(&client, &dout_e, &wsc[3], &one, &one, &dv_d_raw, d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, false);
        let (dvdsc_k, dv_d) = alloc(d * r);
        unsafe {
            super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(d * r), EW,
                buf!(dv_d_raw, d * r), buf!(dn.s, r), buf!(dv_d, d * r),
                r as u32, (d * r) as u32,
            );
        }
        let (dsd_k, ds_d) = alloc(r);
        unsafe {
            super::kernels::sum_ds_kernel::launch_unchecked::<f32, Cuda>(
                &client, CubeCount::Static(1, 1, 1), UNITS,
                buf!(dm_d, bt * r), buf!(wsc[3], bt * r), buf!(ds_d, r),
                bt as u32, r as u32,
            );
        }
        let (dsil_k, dsil) = alloc(bt * f);
        launch_mm(&client, &dz_d, &dn.u, &one, &dn.mu, &dsil, bt, r, f, r, 1, r, 1, bt * r, f * r, false, true, true, false, false);
        unsafe {
            super::kernels::silu_bwd_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(bt * f), EW,
                buf!(wsc[1], bt * f), buf!(dsil, bt * f), buf!(da2, bt * f),
                (bt * f) as u32,
            );
        }
        out[9 + 6 * e] = Some(du_d);
        out[10 + 6 * e] = Some(ds_d);
        out[11 + 6 * e] = Some(dv_d);
        keep.extend([dmd_k, dzd_k, dud_k, dvd_k, dvdsc_k, dsd_k, dsil_k.clone()]);

        // gate_up TSCT: da2 -> dx (accumulate)
        let (dme_k, dm_e) = alloc(bt * r);
        launch_mm(&client, &da2, &gu.v, &gu.s, &gu.mv, &dm_e, bt, f, r, f, 1, r, 1, bt * f, f * r, false, false, true, false, false);
        let (dze_k, dz_e) = alloc(bt * r);
        unsafe {
            super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(bt * r), EW,
                buf!(dm_e, bt * r), buf!(gu.s, r), buf!(dz_e, bt * r),
                r as u32, (bt * r) as u32,
            );
        }
        let (due_k, du_e) = alloc(d * r);
        launch_mm(&client, &normed, &dz_e, &one, &one, &du_e, d, bt, r, d, 1, r, 1, bt * d, bt * r, true, false, false, false, false);
        let (dve_k, dv_e_raw) = alloc(f * r);
        launch_mm(&client, &da2, &wsc[0], &one, &one, &dv_e_raw, f, bt, r, f, 1, r, 1, bt * f, bt * r, true, false, false, false, false);
        let (dvesc_k, dv_e) = alloc(f * r);
        unsafe {
            super::kernels::col_scale_kernel::launch_unchecked::<f32, Cuda>(
                &client, ew_cubes(f * r), EW,
                buf!(dv_e_raw, f * r), buf!(gu.s, r), buf!(dv_e, f * r),
                r as u32, (f * r) as u32,
            );
        }
        let (dse_k, ds_e) = alloc(r);
        unsafe {
            super::kernels::sum_ds_kernel::launch_unchecked::<f32, Cuda>(
                &client, CubeCount::Static(1, 1, 1), UNITS,
                buf!(dm_e, bt * r), buf!(wsc[0], bt * r), buf!(ds_e, r),
                bt as u32, r as u32,
            );
        }
        launch_mm(&client, &dz_e, &gu.u, &one, &one, &dx, bt, r, d, r, 1, r, 1, bt * r, d * r, false, true, true, false, true);
        out[6 + 6 * e] = Some(du_e);
        out[7 + 6 * e] = Some(ds_e);
        out[8 + 6 * e] = Some(dv_e);
        keep.extend([dme_k, dze_k, due_k, dve_k, dvesc_k, dse_k]);
    }
    drop(da2);
    drop(dout_e);
    let (drs_k, drs) = alloc(1);
    unsafe {
        super::kernels::sum_part_kernel::launch_unchecked::<f32, Cuda>(
            &client, CubeCount::Static(1, 1, 1), UNITS,
            buf!(drs_part, bt),
            buf!(drs, 1),
            bt as u32,
        );
    }
    out[4] = Some(drs);
    keep.push(drs_k);

    // ---- controller: softmax jacobian -> dRaw; dWc, dCtrl
    // Wc is stored [2d, pad] (burn Linear Col layout): dWc = ctrl_inᵀ·draw,
    // dctrl = draw·Wcᵀ.
    unsafe {
        super::kernels::softmax_bwd_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(bt as u32, 1, 1),
            UNITS,
            buf!(blend, bt * nexp),
            buf!(dblend, bt * nexp),
            buf!(draw, bt * pad),
            nexp as u32,
            pad as u32,
        );
    }
    let (dwc_k, dwc) = alloc(2 * d * pad);
    launch_mm(&client, &ctrl_in, &draw, &one, &one, &dwc, 2 * d, bt, pad, 2 * d, 1, pad, 1, bt * 2 * d, bt * pad, true, false, false, false, false);
    let (dc_k, dctrl) = alloc(bt * 2 * d);
    launch_mm(&client, &draw, &st.wc, &one, &one, &dctrl, bt, pad, 2 * d, pad, 1, pad, 1, bt * pad, 2 * d * pad, false, true, false, false, false);
    // NOTE: cat_bwd (which splits dctrl into the h_ctx and x halves) runs
    // AFTER the RMSNorm backward below - the x-half must not leak into the
    // d(normed) buffer that rms_bwd consumes.
    out[1] = Some(dwc);
    keep.extend([dwc_k, dc_k]);
    dump(&client, "dctrl", &dctrl, bt * 2 * d);
    dump(&client, "draw", &draw, bt * pad);

    // ---- halt head: dLam -> dHaltpre -> dWh, dHalt_in
    let dosum = zeros_raw(&dev, &client, b);
    unsafe {
        super::kernels::dlam_outacc_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(b as u32, 1, 1),
            UNITS,
            buf!(gc, 1 + b + bt * d),
            buf!(step_out, bt * d),
            buf!(dosum, b),
            out_off as u32,
            t as u32,
            d as u32,
        );
    }
    let (dhp_k, dhaltpre) = alloc(b);
    unsafe {
        super::kernels::lam_bwd_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(b as u32, 1, 1),
            UNITS,
            buf!(st.flat_out, 1 + b),
            buf!(gc, 1 + b),
            buf!(ceb, b),
            buf!(dosum, b),
            buf!(dhaltpre, b),
            lam_off as u32,
            bt as u32,
        );
    }
    let (dwh_k, dwh) = alloc(d);
    launch_mm(&client, &dhaltpre, &halt_in, &one, &one, &dwh, 1, b, d, 1, 1, d, 1, b, b * d, true, false, false, false, false);
    let (dhi_k, dhalt_in) = alloc(b * d);
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
    dump(&client, "dhaltpre", &dhaltpre, b);
    dump(&client, "dwh", &dwh, d);
    dump(&client, "dosum", &dosum, b);
    out[5] = Some(dwh);
    keep.extend([dhp_k, dwh_k, dhi_k]);

    // ---- RMSNorm backward (dX buffer currently holds grad of `normed`)
    let (_dg_k, dg) = alloc(d);
    unsafe {
        super::kernels::dg_kernel::launch_unchecked::<f32, Cuda>(
            &client, CubeCount::Static(1, 1, 1), UNITS,
            buf!(dx, bt * d), buf!(h_ctx, bt * d), buf!(inv, bt),
            buf!(dg, d),
            bt as u32, d as u32,
        );
        super::kernels::rms_bwd_kernel::launch_unchecked::<f32, Cuda>(
            &client,
            CubeCount::Static(bt as u32, 1, 1),
            UNITS,
            buf!(dx, bt * d),
            buf!(st.g, d),
            buf!(h_ctx, bt * d),
            buf!(inv, bt),
            buf!(dh_ctx, bt * d),
            d as u32,
        );
    }
    dump(&client, "dg", &dg, d);
    out[2] = Some(dg);

    // ---- controller split (after the RMSNorm backward: see the note above)
    unsafe {
        super::kernels::cat_bwd_kernel::launch_unchecked::<f32, Cuda>(
            &client, ew_cubes(bt * d), EW,
            buf!(dctrl, bt * 2 * d),
            buf!(dh_ctx, bt * d),
            buf!(dx, bt * d),
            d as u32,
            (bt * d) as u32,
        );
    }

    // ---- iter_embed + input passthrough
    let die = zeros_raw(&dev, &client, st.max_iter * d);
    unsafe {
        super::kernels::ie_grad_kernel::launch_unchecked::<f32, Cuda>(
            &client, CubeCount::Static(1, 1, 1), UNITS,
            buf!(dh_ctx, bt * d),
            buf!(die, st.max_iter * d),
            bt as u32,
            d as u32,
        );
        super::kernels::add_kernel::launch_unchecked::<f32, Cuda>(
            &client, ew_cubes(bt * d), EW,
            buf!(dx, bt * d),
            buf!(dh_ctx, bt * d),
            (bt * d) as u32,
        );
    }
    dump(&client, "die", &die, st.max_iter * d);
    dump(&client, "dh_ctx", &dh_ctx, bt * d);
    dump(&client, "dx", &dx, bt * d);
    out[0] = Some(dx);
    out[3] = Some(die);

    // ---- register every parent grad.
    // Registered primitives must carry the parent's rank/shapes: burn-optim
    // consumes them through shape-checked tensor ops against the params. The
    // grads were computed into flat dense buffers, so the wrap is a free
    // stride relabel (1D -> 2D on a dense buffer never copies).
    let dims: [Option<[usize; 2]>; 30] = {
        let mut dm: [Option<[usize; 2]>; 30] = std::array::from_fn(|_| None);
        dm[0] = Some([bt, d]);
        dm[1] = Some([2 * d, pad]); // burn Linear Col layout
        dm[2] = None; // norm g: [d]
        dm[3] = Some([st.max_iter, d]);
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
        dm[24] = Some([d, r]);
        dm[25] = None;
        dm[26] = Some([d, r]);
        dm[27] = Some([d, r]);
        dm[28] = None;
        dm[29] = Some([v, r]);
        dm
    };
    for (i, g) in out.into_iter().enumerate() {
        if let (Some(node), Some(gt)) = (&ops.parents[i], g) {
            let gt = match dims[i] {
                Some([r0, r1]) => Tensor::<1>::from_primitive::<CB>(gt)
                    .reshape::<2, _>([r0, r1])
                    .try_into_primitive::<CB>()
                    .expect("grad reshape"),
                None => gt,
            };
            grads.register::<CB>(node.id, gt);
        }
    }
    super::sync(&client);
}
