//! Hand-written cubecl kernels for the fused Ponder step (fp32, flat dense
//! buffers, `launch_unchecked`). All matmuls funnel through [`mm_kernel`];
//! the rest are elementwise/row/reduction kernels sized for the loop-body
//! shapes. Conventions mirror burn-spectral `moe_fused.rs`.

use cubecl::prelude::*;

/// Generic fp32 matmul: out[m,n] (= or +=) sum_k A'(m,k)·B'(k,n), where B' is
/// the absmean-ternary of `b` (0.7 dead zone, burn-spectral `ternarize`)
/// under `tern_b`, scaled per-k by `s` under `scale_k`.
///
/// Storage conventions (`ta`/`tb` select between a matrix and its transpose):
/// - `ta=false`: A' row-major [m,k] -> (am=k, ak=1);
///   `ta=true`: A' = Xᵀ for X row-major [k,m] -> (am=m, ak=1), i.e. am is
///   X's row length (the kk stride) and ak=1.
/// - `tb=false`: B' row-major [k,n] -> (bk=n, bn=1);
///   `tb=true`: B' = Yᵀ for Y row-major [n,k] -> (bk=k, bn=1), i.e. bk is
///   Y's row length (the kk stride) and bn=1.
/// One thread per output cell, serial k accumulation, so the per-element FMA
/// order is fixed.
/// ponytail: no tiling, no cmma - exact fp32 and dead simple; tile it when a
/// profile demands (M6), the launch convention stays identical.
#[cube(launch_unchecked, address_type = "dynamic")]
pub fn mm_kernel<F: Float>(
    a: &[F],
    b: &[F],
    s: &[F],
    bm: &[F],
    out: &mut [F],
    m: u32,
    k: u32,
    n: u32,
    am: u32,
    ak: u32,
    bk: u32,
    bn: u32,
    #[comptime] ta: bool,
    #[comptime] tb: bool,
    #[comptime] tern_b: bool,
    #[comptime] scale_k: bool,
    #[comptime] accum: bool,
) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let (am, ak, bk, bn) = (am as usize, ak as usize, bk as usize, bn as usize);
    if i < m * n {
        let row = i / n;
        let c = i % n;
        let mut acc = F::new(0.0_f32);
        let mut kk = 0usize;
        while kk < k {
            let av = if ta {
                a[kk * am + row * ak]
            } else {
                a[row * am + kk * ak]
            };
            let bv_raw = if tb {
                b[c * bk + kk * bn]
            } else {
                b[kk * bk + c * bn]
            };
            let bv = if tern_b {
                let keep = if bv_raw.abs() > bm[0] * F::new(0.7_f32) {
                    F::new(1.0_f32)
                } else {
                    F::new(0.0_f32)
                };
                let sg = if bv_raw > F::new(0.0_f32) {
                    F::new(1.0_f32)
                } else if bv_raw < F::new(0.0_f32) {
                    F::new(-1.0_f32)
                } else {
                    F::new(0.0_f32)
                };
                sg * bm[0] * keep
            } else {
                bv_raw
            };
            let sv = if scale_k { s[kk] } else { F::new(1.0_f32) };
            acc += av * bv * sv;
            kk += 1usize;
        }
        if accum {
            out[i] += acc;
        } else {
            out[i] = acc;
        }
    }
}

/// out[i] = val for i < n (elementwise fill; launch-probe helper).
#[cube(launch_unchecked)]
pub fn fill_kernel<F: Float>(out: &mut [F], val: f32, n: u32) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        out[i] = F::cast_from(val);
    }
}

/// out[0] = mean(|w|) over n (one cube).
#[cube(launch_unchecked)]
pub fn absmean_kernel<F: Float>(w: &[F], out: &mut [F], n: u32) {
    let unit = UNIT_POS_X as usize;
    let mut shared = Shared::new_slice(32usize);
    let mut acc = F::new(0.0_f32);
    let mut i = unit;
    while i < n as usize {
        acc += w[i].abs();
        i += 32usize;
    }
    shared[unit] = acc;
    sync_cube();
    if unit == 0usize {
        let mut s = F::new(0.0_f32);
        #[unroll]
        for u in 0..32 {
            s += shared[u];
        }
        out[0] = s / F::cast_from(n as f32);
    }
}

/// normed = h_ctx·inv_rms·g; inv[row] = 1/sqrt(mean(h_ctx²)+eps).
#[cube(launch_unchecked)]
pub fn rmsnorm_kernel<F: Float>(x: &[F], g: &[F], out: &mut [F], inv: &mut [F], d: u32, eps: f32) {
    let row = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let d = d as usize;
    let mut shared = Shared::new_slice(32usize);
    let mut acc = F::new(0.0_f32);
    let mut j = unit;
    while j < d {
        acc += x[row * d + j] * x[row * d + j];
        j += 32usize;
    }
    shared[unit] = acc;
    sync_cube();
    if unit == 0usize {
        let mut s = F::new(0.0_f32);
        #[unroll]
        for u in 0..32 {
            s += shared[u];
        }
        shared[0] = (s / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();
    }
    sync_cube();
    let inv_rms = F::new(1.0_f32) / shared[0];
    if unit == 0usize {
        inv[row] = inv_rms;
    }
    let mut j = unit;
    while j < d {
        out[row * d + j] = x[row * d + j] * inv_rms * g[j];
        j += 32usize;
    }
}

/// h_ctx = x + iter_embed[0] (row 0, broadcast over rows).
#[cube(launch_unchecked)]
pub fn hctx_kernel<F: Float>(x: &[F], ie: &[F], out: &mut [F], d: u32, n: u32) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        out[i] = x[i] + ie[i % d as usize];
    }
}

/// ctrl_in = [a | b] along columns, row-interleaved: row r of the [n/d, 2d]
/// output holds a[r·d..] then b[r·d..].
#[cube(launch_unchecked)]
pub fn cat_kernel<F: Float>(a: &[F], b: &[F], out: &mut [F], d: u32, n: u32) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        let d = d as usize;
        let row = i / d;
        let j = i % d;
        out[row * 2 * d + j] = a[i];
        out[row * 2 * d + d + j] = b[i];
    }
}

/// w_ffn = sigmoid(raw[:,2]); blend = softmax(raw[:,3..3+nexp]).
#[cube(launch_unchecked)]
pub fn sigsel_kernel<F: Float>(raw: &[F], w_ffn: &mut [F], blend: &mut [F], nexp: u32, pad: u32) {
    let row = CUBE_POS_X as usize;
    if UNIT_POS_X == 0u32 {
        let base = row * pad as usize;
        let mut mx = raw[base + 3usize];
        let mut e = 1usize;
        while e < nexp as usize {
            let v = raw[base + 3usize + e];
            if v > mx {
                mx = v;
            }
            e += 1usize;
        }
        let mut sum = F::new(0.0_f32);
        e = 0usize;
        while e < nexp as usize {
            sum += (raw[base + 3usize + e] - mx).exp();
            e += 1usize;
        }
        w_ffn[row] = F::new(1.0_f32) / (F::new(1.0_f32) + (-raw[base + 2usize]).exp());
        e = 0usize;
        while e < nexp as usize {
            blend[row * nexp as usize + e] = (raw[base + 3usize + e] - mx).exp() / sum;
            e += 1usize;
        }
    }
}

/// ffn[i] += out_e[i] · blend[row(i)·nexp + e].
#[cube(launch_unchecked)]
pub fn axpy_kernel<F: Float>(
    ffn: &mut [F],
    out_e: &[F],
    blend: &[F],
    e: u32,
    nexp: u32,
    d: u32,
    n: u32,
) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        ffn[i] += out_e[i] * blend[(i / d as usize) * nexp as usize + e as usize];
    }
}

/// y = ffn·w_ffn; h = h_ctx + y·rs (ReZero residual; arms off => y is the
/// whole block body).
#[cube(launch_unchecked)]
pub fn residual_kernel<F: Float>(
    ffn: &[F],
    w_ffn: &[F],
    rs: &[F],
    h_ctx: &[F],
    y: &mut [F],
    h: &mut [F],
    d: u32,
) {
    let row = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let d = d as usize;
    let mut j = unit;
    while j < d {
        let i = row * d + j;
        let yv = ffn[i] * w_ffn[row];
        y[i] = yv;
        h[i] = h_ctx[i] + yv * rs[0];
        j += 32usize;
    }
}

/// halt_in[b·d + j] = mean_t h_ctx[(b·t+ti)·d + j].
#[cube(launch_unchecked)]
pub fn rowmean_t_kernel<F: Float>(h_ctx: &[F], out: &mut [F], t: u32, d: u32) {
    let b = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let (t, d) = (t as usize, d as usize);
    let mut j = unit;
    while j < d {
        let mut acc = F::new(0.0_f32);
        let mut ti = 0usize;
        while ti < t {
            acc += h_ctx[(b * t + ti) * d + j];
            ti += 1usize;
        }
        out[b * d + j] = acc / F::cast_from(t as f32);
        j += 32usize;
    }
}

/// flat[1+b] = sigmoid(halt_in[b] · wh) (lam; p_dist at N=1).
#[cube(launch_unchecked)]
pub fn halt_fwd_kernel<F: Float>(halt_in: &[F], wh: &[F], flat: &mut [F], d: u32) {
    let b = CUBE_POS_X as usize;
    let d = d as usize;
    let mut acc = F::new(0.0_f32);
    let mut j = 0usize;
    while j < d {
        acc += halt_in[b * d + j] * wh[j];
        j += 1usize;
    }
    flat[1usize + b] = F::new(1.0_f32) / (F::new(1.0_f32) + (-acc).exp());
}

/// flat[out+i] = step_out[i] · lam[(i/d)/t].
#[cube(launch_unchecked)]
pub fn outacc_kernel<F: Float>(
    step_out: &[F],
    flat: &mut [F],
    lam_off: u32,
    out_off: u32,
    t: u32,
    d: u32,
    n: u32,
) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        let bidx = i / d as usize / t as usize;
        flat[out_off as usize + i] = step_out[i] * flat[lam_off as usize + bidx];
    }
}

/// ce[row] = -log_softmax(logits[row])[tgt[row]].
#[cube(launch_unchecked)]
pub fn ce_kernel<F: Float>(logits: &[F], tgt: &[i32], ce: &mut [F], v: u32) {
    let row = CUBE_POS_X as usize;
    let v = v as usize;
    let base = row * v;
    let mut mx = logits[base];
    let mut j = 1usize;
    while j < v {
        if logits[base + j] > mx {
            mx = logits[base + j];
        }
        j += 1usize;
    }
    let mut sum = F::new(0.0_f32);
    j = 0usize;
    while j < v {
        sum += (logits[base + j] - mx).exp();
        j += 1usize;
    }
    let t = tgt[row] as usize;
    ce[row] = F::new(0.0_f32) - (logits[base + t] - mx - sum.ln());
}

/// ceb[b] = sum_t ce[b·t+ti].
#[cube(launch_unchecked)]
pub fn ceb_kernel<F: Float>(ce: &[F], ceb: &mut [F], t: u32) {
    let b = CUBE_POS_X as usize;
    let t = t as usize;
    let mut acc = F::new(0.0_f32);
    let mut ti = 0usize;
    while ti < t {
        acc += ce[b * t + ti];
        ti += 1usize;
    }
    ceb[b] = acc;
}

/// flat[0] = sum_b lam_b · ceb_b / (b·t).
#[cube(launch_unchecked)]
pub fn rec_kernel<F: Float>(ceb: &[F], flat: &mut [F], lam_off: u32, b: u32, bt: u32) {
    let mut acc = F::new(0.0_f32);
    let mut i = 0usize;
    while i < b as usize {
        acc += flat[lam_off as usize + i] * ceb[i];
        i += 1usize;
    }
    flat[0] = acc / F::cast_from(bt as f32);
}

// ---------------- backward kernels ----------------

/// dLogits = (softmax - onehot(tgt)) · dRec·lam_b/(b·t).
///
/// Sign note: CE = -log p already carries the minus, so d(CE)/dlogit =
/// (softmax - onehot) POSITIVE - the grad of rec = Σ lam·CE/(bt) wrt the
/// logits has no extra negation (the `-` here once sign-flipped the whole
/// backward chain; FD-vs-burn caught it).
#[cube(launch_unchecked)]
pub fn dlogits_kernel<F: Float>(
    logits: &[F],
    tgt: &[i32],
    flat_fwd: &[F],
    flat_g: &[F],
    dlogits: &mut [F],
    lam_off: u32,
    t: u32,
    v: u32,
    bt: u32,
) {
    let row = CUBE_POS_X as usize;
    let v = v as usize;
    let base = row * v;
    let mut mx = logits[base];
    let mut j = 1usize;
    while j < v {
        if logits[base + j] > mx {
            mx = logits[base + j];
        }
        j += 1usize;
    }
    let mut sum = F::new(0.0_f32);
    j = 0usize;
    while j < v {
        sum += (logits[base + j] - mx).exp();
        j += 1usize;
    }
    let drec = flat_g[0] / F::cast_from(bt as f32);
    // lam comes from the saved FORWARD flat output (flat_g holds gradients)
    let lam = flat_fwd[lam_off as usize + row / t as usize];
    let tgt = tgt[row] as usize;
    j = 0usize;
    while j < v {
        let sm = (logits[base + j] - mx).exp() / sum;
        let onehot = if j == tgt {
            F::new(1.0_f32)
        } else {
            F::new(0.0_f32)
        };
        dlogits[base + j] = (sm - onehot) * drec * lam;
        j += 1usize;
    }
}

/// dStep[i] += dOut_acc[i] · lam[(i/d)/t] (out_acc path of the readout);
/// lam from the saved forward flat output.
#[cube(launch_unchecked)]
pub fn dso_kernel<F: Float>(
    flat_g: &[F],
    flat_fwd: &[F],
    out_off: u32,
    dstep: &mut [F],
    lam_off: u32,
    t: u32,
    d: u32,
    n: u32,
) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        let bidx = i / d as usize / t as usize;
        dstep[i] += flat_g[out_off as usize + i] * flat_fwd[lam_off as usize + bidx];
    }
}

/// out[i] = x[i] · s[i % cols].
#[cube(launch_unchecked)]
pub fn col_scale_kernel<F: Float>(x: &[F], s: &[F], out: &mut [F], cols: u32, n: u32) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        out[i] = x[i] * s[i % cols as usize];
    }
}

/// ds[j] = sum_m dM[m,j]·Z[m,j] (one cube, unit-strided columns).
#[cube(launch_unchecked)]
pub fn sum_ds_kernel<F: Float>(dm: &[F], z: &[F], ds: &mut [F], rows: u32, r: u32) {
    let unit = UNIT_POS_X as usize;
    let (rows, r) = (rows as usize, r as usize);
    let mut j = unit;
    while j < r {
        let mut acc = F::new(0.0_f32);
        let mut m = 0usize;
        while m < rows {
            acc += dm[m * r + j] * z[m * r + j];
            m += 1usize;
        }
        ds[j] = acc;
        j += 32usize;
    }
}

/// ReZero/gate split: dy = dh_flat·rs; dh_ctx += dh_flat;
/// drs_part[row] = sum_d dh_flat·y; dwffn_part[row] = sum_d dy·ffn.
#[cube(launch_unchecked)]
pub fn residual_bwd_kernel<F: Float>(
    dh_flat: &[F],
    rs: &[F],
    y: &[F],
    ffn: &[F],
    dy: &mut [F],
    dh_ctx: &mut [F],
    drs_part: &mut [F],
    dwffn_part: &mut [F],
    d: u32,
) {
    let row = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let d = d as usize;
    let mut j = unit;
    while j < d {
        let i = row * d + j;
        dh_ctx[i] += dh_flat[i];
        dy[i] = dh_flat[i] * rs[0];
        j += 32usize;
    }
    sync_cube();
    if unit == 0usize {
        let mut drs = F::new(0.0_f32);
        let mut dwf = F::new(0.0_f32);
        let mut j = 0usize;
        while j < d {
            let i = row * d + j;
            drs += dh_flat[i] * y[i];
            dwf += dy[i] * ffn[i];
            j += 1usize;
        }
        drs_part[row] = drs;
        dwffn_part[row] = dwf;
    }
}

/// dRaw[:,2] = dwffn·w·(1-w); dFfn = dy·w_ffn.
#[cube(launch_unchecked)]
pub fn ffnrow_bwd_kernel<F: Float>(
    dy: &[F],
    w_ffn: &[F],
    dwffn_part: &[F],
    dffn: &mut [F],
    draw: &mut [F],
    pad: u32,
    d: u32,
) {
    let row = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let (pad, d) = (pad as usize, d as usize);
    let w = w_ffn[row];
    let mut j = unit;
    while j < d {
        dffn[row * d + j] = dy[row * d + j] * w;
        j += 32usize;
    }
    sync_cube();
    if unit == 0usize {
        draw[row * pad + 2usize] = dwffn_part[row] * w * (F::new(1.0_f32) - w);
    }
}

/// dOut_e = dFfn·blend[:,e]; dBlend[:,e] = sum_d dFfn·out_e.
#[cube(launch_unchecked)]
pub fn dblend_kernel<F: Float>(
    dffn: &[F],
    out_e: &[F],
    blend: &[F],
    dout_e: &mut [F],
    dblend: &mut [F],
    e: u32,
    nexp: u32,
    d: u32,
) {
    let row = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let (nexp, d) = (nexp as usize, d as usize);
    let mut j = unit;
    while j < d {
        let i = row * d + j;
        dout_e[i] = dffn[i] * blend[row * nexp + e as usize];
        j += 32usize;
    }
    sync_cube();
    if unit == 0usize {
        let mut acc = F::new(0.0_f32);
        let mut j = 0usize;
        while j < d {
            acc += dffn[row * d + j] * out_e[row * d + j];
            j += 1usize;
        }
        dblend[row * nexp + e as usize] = acc;
    }
}

/// dRaw[:,3+k] = blend_k·(dBlend_k - dot), dot = sum_j dBlend_j·blend_j.
#[cube(launch_unchecked)]
pub fn softmax_bwd_kernel<F: Float>(
    blend: &[F],
    dblend: &[F],
    draw: &mut [F],
    nexp: u32,
    pad: u32,
) {
    let row = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let (nexp, pad) = (nexp as usize, pad as usize);
    let mut shared = Shared::new_slice(2usize);
    if unit == 0usize {
        let mut dot = F::new(0.0_f32);
        let mut k = 0usize;
        while k < nexp {
            dot += dblend[row * nexp + k] * blend[row * nexp + k];
            k += 1usize;
        }
        shared[0] = dot;
    }
    sync_cube();
    let dot = shared[0];
    let mut k = unit;
    while k < nexp {
        draw[row * pad + 3usize + k] = blend[row * nexp + k] * (dblend[row * nexp + k] - dot);
        k += 32usize;
    }
}

/// silu backward: dA = dsil · σ(a)·(1 + a·(1-σ(a))).
#[cube(launch_unchecked)]
pub fn silu_bwd_kernel<F: Float>(a: &[F], dsil: &[F], da: &mut [F], n: u32) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        let av = a[i];
        let s = F::new(1.0_f32) / (F::new(1.0_f32) + (-av).exp());
        da[i] = dsil[i] * s * (F::new(1.0_f32) + av * (F::new(1.0_f32) - s));
    }
}

/// dH_ctx += dCtrl[:, :half]; dX += dCtrl[:, half:] (row-interleaved layout,
/// inverse of [`cat_kernel`]).
#[cube(launch_unchecked)]
pub fn cat_bwd_kernel<F: Float>(dctrl: &[F], dh_ctx: &mut [F], dx: &mut [F], d: u32, n: u32) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        let d = d as usize;
        let row = i / d;
        let j = i % d;
        dh_ctx[i] += dctrl[row * 2 * d + j];
        dx[i] += dctrl[row * 2 * d + d + j];
    }
}

/// dx += dh.
#[cube(launch_unchecked)]
pub fn add_kernel<F: Float>(dx: &mut [F], dh: &[F], n: u32) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        dx[i] += dh[i];
    }
}

/// dIterEmbed[0,j] = sum_m dH_ctx[m,j] (higher rows stay zero from init).
#[cube(launch_unchecked)]
pub fn ie_grad_kernel<F: Float>(dh_ctx: &[F], die: &mut [F], rows: u32, d: u32) {
    let unit = UNIT_POS_X as usize;
    let d = d as usize;
    let mut j = unit;
    while j < d {
        let mut acc = F::new(0.0_f32);
        let mut m = 0usize;
        while m < rows as usize {
            acc += dh_ctx[m * d + j];
            m += 1usize;
        }
        die[j] = acc;
        j += 32usize;
    }
}

/// dLam = dP + dRec·ceb/(b·t) + dOutAccSum; dHaltpre[b] = dLam·lam·(1-lam).
/// lam comes from the saved forward flat output (flat_g holds gradients).
#[cube(launch_unchecked)]
pub fn lam_bwd_kernel<F: Float>(
    flat_fwd: &[F],
    flat_g: &[F],
    ceb: &[F],
    dsum: &[F],
    dhaltpre: &mut [F],
    lam_off: u32,
    bt: u32,
) {
    let b = CUBE_POS_X as usize;
    let drec = flat_g[0];
    let dp = flat_g[lam_off as usize + b];
    let lam = flat_fwd[lam_off as usize + b];
    let dlam = dp + drec * ceb[b] / F::cast_from(bt as f32) + dsum[b];
    dhaltpre[b] = dlam * lam * (F::new(1.0_f32) - lam);
}

/// dH_ctx[(b·t+ti)·d + j] += dHalt_in[b·d + j]/t.
#[cube(launch_unchecked)]
pub fn haltmean_bwd_kernel<F: Float>(dhalt_in: &[F], dh_ctx: &mut [F], t: u32, d: u32) {
    let b = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let (t, d) = (t as usize, d as usize);
    let mut j = unit;
    while j < d {
        let g = dhalt_in[b * d + j] / F::cast_from(t as f32);
        let mut ti = 0usize;
        while ti < t {
            dh_ctx[(b * t + ti) * d + j] += g;
            ti += 1usize;
        }
        j += 32usize;
    }
}

/// dg[j] = sum_m dXn[m,j]·r[m,j], r = h_ctx·inv (RMSNorm weight grad; the
/// output is r·g, so the weight jacobian carries r, not r·g).
#[cube(launch_unchecked)]
pub fn dg_kernel<F: Float>(dxn: &[F], h_ctx: &[F], inv: &[F], dg: &mut [F], rows: u32, d: u32) {
    let unit = UNIT_POS_X as usize;
    let d = d as usize;
    let mut j = unit;
    while j < d {
        let mut acc = F::new(0.0_f32);
        let mut m = 0usize;
        while m < rows as usize {
            acc += dxn[m * d + j] * h_ctx[m * d + j] * inv[m];
            m += 1usize;
        }
        dg[j] = acc;
        j += 32usize;
    }
}

/// dLam[b] += sum_{t,d} dOut_acc·step_out (the out_acc = step_out·p path of
/// the halting distribution; N=1 so the p·π term is identity). `g_out` is the
/// FULL flat upstream grad; `out_off` selects its out_acc region (without the
/// offset this sums the rec/lam grad region against step_out - garbage).
#[cube(launch_unchecked)]
pub fn dlam_outacc_kernel<F: Float>(
    g_out: &[F],
    step_out: &[F],
    dlam: &mut [F],
    out_off: u32,
    t: u32,
    d: u32,
) {
    let b = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let (t, d) = (t as usize, d as usize);
    let total = t * d;
    let mut acc = F::new(0.0_f32);
    let mut i = unit;
    while i < total {
        let idx = b * total + i;
        acc += g_out[out_off as usize + idx] * step_out[idx];
        i += 32usize;
    }
    let mut shared = Shared::new_slice(32usize);
    shared[unit] = acc;
    sync_cube();
    if unit == 0usize {
        let mut s = F::new(0.0_f32);
        #[unroll]
        for u in 0..32 {
            s += shared[u];
        }
        dlam[b] += s;
    }
}

/// RMSNorm backward: dH_ctx += inv·(dXn·g - r·S/d), S = sum_j dXn·g·r.
#[cube(launch_unchecked)]
pub fn rms_bwd_kernel<F: Float>(
    dxn: &[F],
    g: &[F],
    h_ctx: &[F],
    inv: &[F],
    dh_ctx: &mut [F],
    d: u32,
) {
    let row = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let d = d as usize;
    let mut shared = Shared::new_slice(32usize);
    let mut acc = F::new(0.0_f32);
    let mut j = unit;
    while j < d {
        let i = row * d + j;
        acc += dxn[i] * g[j] * h_ctx[i] * inv[row];
        j += 32usize;
    }
    shared[unit] = acc;
    sync_cube();
    if unit == 0usize {
        let mut s = F::new(0.0_f32);
        #[unroll]
        for u in 0..32 {
            s += shared[u];
        }
        shared[31] = s / F::cast_from(d as f32);
    }
    sync_cube();
    let s_tot = shared[31];
    let mut j = unit;
    while j < d {
        let i = row * d + j;
        let r = h_ctx[i] * inv[row];
        dh_ctx[i] += inv[row] * (dxn[i] * g[j] - r * s_tot);
        j += 32usize;
    }
}

/// out[0] = sum parts (one cube).
#[cube(launch_unchecked)]
pub fn sum_part_kernel<F: Float>(parts: &[F], out: &mut [F], n: u32) {
    let unit = UNIT_POS_X as usize;
    let mut acc = F::new(0.0_f32);
    let mut i = unit;
    while i < n as usize {
        acc += parts[i];
        i += 32usize;
    }
    let mut shared = Shared::new_slice(32usize);
    shared[unit] = acc;
    sync_cube();
    if unit == 0usize {
        let mut s = F::new(0.0_f32);
        #[unroll]
        for u in 0..32 {
            s += shared[u];
        }
        out[0] = s;
    }
}

/// silu elementwise.
#[cube(launch_unchecked)]
pub fn silu_kernel<F: Float>(a: &[F], out: &mut [F], n: u32) {
    let i = CUBE_POS_X as usize * 256usize + UNIT_POS_X as usize;
    if i < n as usize {
        let av = a[i];
        out[i] = av / (F::new(1.0_f32) + (-av).exp());
    }
}
