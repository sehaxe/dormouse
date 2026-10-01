#!/usr/bin/env python3
"""f64 CPU reference for Gated DeltaNet-2. Generates `tests/ref_f64.bin`.

WHAT THIS IS, AND WHAT IT IS NOT.

This is an **f64 transcription of the paper's equations**, written out one
token at a time, plus a committed fixture of its outputs. It exists because
every other test in this crate compares our code to our code
(`vendor/dormouse-fused/TEST-AUDIT.md`, `docs/protocols/ORACLE.md` §2): a bug above the
fused/ops branch point moves both arms together and the difference is exactly
zero. This layer is the first one whose expected value does not come from a
second implementation of the same code.

**It is tier (b), not tier (a).** It is a transcription, so a shared
*misreading* of the paper survives it. It is not the authors' own bytes; those
need `NVlabs/GatedDeltaNet-2`'s Triton kernel actually run, which is
`docs/protocols/ORACLE.md` §8 candidate (2) and was not attempted here. What it buys is
the removal of one confound: with f64 at the top of the stack, a discrepancy is
no longer ambiguous between "our maths" and "f32 conditioning".

PROVENANCE, line by line. Every non-obvious line cites where it comes from:

  * the recurrence, Eq. 9/10:      arXiv:2605.22791 §3.1
        S̄_t = D_t S_{t-1},  r_t = S̄_t^T e_t,  S_t = S̄_t + k_t (z_t - r_t)^T
        e_t = b_t ⊙ k_t (§8),  z_t = w_t ⊙ v_t (§8),  α_t = exp(g_t) (§12)
  * o_t = S_t^T q_t:              §3.1, and App. C.5 for the decode kernel
  * `scale = 1/sqrt(K)`:          NOT in the paper. From the authors' kernel,
        `lit_gpt/gdn2_ops/chunk_gdn2.py`:
            if scale is None: scale = k.shape[-1] ** -0.5
        and it multiplies the WHOLE readout, not one term:
        `fla/ops/gla/chunk.py::chunk_gla_fwd_kernel_o` does `b_o *= scale` on
        the inter-chunk (state) term, while the intra-chunk term arrives
        through `Aqk`, which was already built with `* scale`.
  * g = -exp(A) ⊙ softplus(W_f x + dt), A per head broadcast over d_k:
        paper Eq. 12 + App. C.1; `lit_gpt/gdn2.py` builds it on the FLAT
        `key_dim` (`A_log.float().exp().repeat_interleave(head_k_dim)`) and
        then GVA-repeats it, which is the same thing as building it per head.
  * L2 norm of q and k per head, `x / sqrt(sum(x^2) + 1e-6)`:
        paper App. D.2; the exact form (add eps INSIDE the sqrt) is
        `fla/modules/l2norm.py::l2norm_fwd_kernel`:
            b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
  * short conv, depthwise, W=4, **ZERO left padding**, last tap is the
        current token:
        `fla/modules/conv/triton/kernels.py::causal_conv1d_fwd_kernel`, the
        `not USE_INITIAL_STATE` branch:
            for i_w in tl.static_range(-W + 1, 1):
                o_x = o_t + i_w
                b_yi = tl.load(..., mask=((o_x >= 0) & (o_x < T))[...,], other=0.0)
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
        i.e. out[t] = sum_i w[i] * x[t - W + 1 + i], and x[s<0] is ZERO, not a
        copy of x[0]. Cross-checked three ways: `ShortConvolution` is an
        `nn.Conv1d(padding=kernel_size-1)` with the default
        `padding_mode='zeros'`; fla's own test reference
        (`tests/modules/test_conv.py::causal_conv1d_ref_torch`) is
        `F.conv1d(..., padding=width-1)`; and that test builds its cache from
        an explicit `torch.zeros(B, D, 1)`.
  * output: `FusedRMSNormSwishGate` = `(x / sqrt(mean(x^2) + eps)) * w * silu(g)`,
        w initialised to ones, NO bias.
        THE GATE IS A CONFIGURATION, NOT A TRANSCRIPTION, and this layer is
        where a reader previously got it wrong in both directions, so the whole
        dispatch is spelled out. Fetched 2026-09-29:
          fla-org/flash-linear-attention @ 9f38d249 (main)
            fla/modules/fused_norm_gate.py:101-104, `layer_norm_gated_fwd_kernel`
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    b_y = b_y * b_g * tl.sigmoid(b_g)
                elif ACTIVATION == "sigmoid":
                    b_y = b_y * tl.sigmoid(b_g)
            ONE kernel, TWO branches. The line quoted above is the `swish` one.
            :1074 `class FusedRMSNormSwishGate(FusedRMSNormGated)` does not pass
            `activation` to super().__init__, so it takes the class default at
            :997, `activation: str = "swish"` => silu.
            :997 `FusedRMSNormGated.__init__` default; `register_parameter(
            "bias", None)`, so the norm has no bias.
          NVlabs/GatedDeltaNet-2 @ a5552fe3 (main)
            lit_gpt/gdn2.py:212  self.o_norm = FusedRMSNormSwishGate(...)  => silu
          fla/layers/gdn2.py:197 (same repo, DIFFERENT FILE)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid")
            => sigmoid.
        So the two credible upstreams DISAGREE, and they disagree at the LAYER
        (`fla/layers/gdn2.py`), not inside the kernel. `src/module.rs` and
        `docs/papers/output-gate-silu-vs-sigmoid.md` carry the finding; it is
        an unresolved technology A/B arm and is deliberately NOT settled here.
        What is settled is that the choice is not silent: `output-gate-sigmoid`
        is one of the committed wrong formulas, so the distance between the two
        branches on these weights and this input is data in the tree. A claim
        that this layer "transcribes the FLA file" and therefore picks sigmoid
        is wrong in a way worth naming: the citation above is to
        `fla/modules/`, and the file that selects sigmoid is `fla/layers/`.
  * GVA: q, k, g, b repeated across value-head groups; v and w already live on
        the value-head axis. Paper §3.5 and App. C.1.
  * `allow_neg_eigval` scales ONLY b by 2, never w. Paper §3.1 and App. C.1.

THE SWEEP FOR THE CLASS THE OUTPUT GATE BELONGS TO. A citation to file A and a
kernel that lives in file B is a blind spot if A and B can disagree. Four lines
here cite one file and are implemented against another, so all four were checked
against BOTH upstreams on 2026-09-29 at a5552fe3 (NVlabs) and 9f38d249 (fla).
Three cannot diverge; one does, and it is the output gate.

  short conv   CANNOT DIVERGE. `lit_gpt/gdn2.py:36` is
               `from fla.modules import FusedRMSNormSwishGate,
               ShortConvolution` - NVlabs imports FLA's class, so "our source" and
               "their source" are the same object.
  L2 norm      CANNOT DIVERGE. `lit_gpt/gdn2_ops/chunk_gdn2.py:64` is
               `from fla.modules.l2norm import l2norm_fwd, l2norm_bwd`, called
               at :2060-2061. And the one place NVlabs does NOT import it, the
               recurrent kernel's own `USE_QK_L2NORM_IN_KERNEL` branch
               (`fused_recurrent_gdn2.py:198-200`), hardcodes the same
               `1 / sqrt(sum(x*x) + 1e-6)` this file uses.
  scale        CANNOT DIVERGE. The reference claims 1/sqrt(K) multiplies the
               WHOLE readout, not one term. `fla/ops/gla/chunk.py` carries it on
               both: `:188`/`:267` scale the intra-chunk `b_A`, and `:427`
               `b_o *= scale` scales the inter-chunk readout. NVlabs's chunk
               kernel is the same shape (`:196`/`:334` on `b_Aqk`). The arm this
               fixture actually exercises is the recurrent one, and
               `fused_recurrent_gdn2.py:201` does `b_q = b_q * scale` with the
               default `scale = k.shape[-1] ** -0.5` at `:322` - scaling the
               readout query, which is the same thing.
  output gate  DIVERGES, and is the finding. One kernel, two `ACTIVATION`
               branches; the two LAYERS select them differently.

So the output gate is the only line in this transcription where "what NVlabs
does" and "what fla does" are different functions, and it is the one the review
found by diffing the two upstreams. Anyone extending this file should re-run the
sweep rather than assume the next citation is safe.

WHY f32 WEIGHTS IN THE FIXTURE. The weights and the inputs are stored f32, so
both arms of the comparison start from bit-identical values and the ONLY thing
the bar is measuring is arithmetic. If the weights were stored f64, burn would
round them to f32 on load and the test would be measuring that too - a real
effect, but one that muddies the question the test exists to answer.

WHY THE BAR IS 1e-3 RELATIVE. A semantic error is O(1) relative; f32
reassociation over 8 chunk boundaries is O(1e-6). A bar at 1e-3 sits three
orders of magnitude above the noise and three below the smallest semantic
error. `tests/ref_f64.rs` carries a negative control that MEASURES both sides
of that gap every run, so the claim cannot rot.

Usage:
    python3 tools/gen_reference_f64.py               # write tests/ref_f64.bin
    python3 tools/gen_reference_f64.py --broad       # write ref_f64_broad.bin
    python3 tools/gen_reference_f64.py --self-test   # print the margin table
    python3 tools/gen_reference_f64.py --fault conv-padding --out /tmp/x.bin
                                                       # a wrong formula
"""

import argparse
import math
import struct
import sys

import numpy as np

# --- configuration, matching the crate's existing test matrix -----------------
D, H, HK, HV = 64, 4, 16, 4
EXPAND_V = 1.5
USE_SHORT_CONV = True
ALLOW_NEG_EIGVAL = False
NORM_EPS = 1e-5
L2_EPS = 1e-6
CONV_W = 4
SEED = 20260929

KD = H * HK              # 64, key_dim  (flattened)
VH = int(HK * EXPAND_V)  # 24, head_v_dim
VD = HV * VH             # 96, value_dim

# T=1 is a canary: with one token there is no state carry and no cross-token
# conv tap, so several candidate layouts coincide. The rest straddle every
# chunk size the crate tests (4, 8, 16, 32, 64).
SEQ_LENS = [1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 21, 32, 33, 37, 64, 65, 70]

# THE BREADTH SWEEP, and why it is a SEPARATE file rather than a bigger
# `SEQ_LENS`. `tests/ref_f64.bin` is hand-picked (one length per interesting
# boundary) and its wrong-formula companion `tests/ref_f64_faults.bin` is
# committed against `FAULT_CASE = len(SEQ_LENS) - 1`, i.e. against the LAST
# entry, chosen because the longest length is the only one where a semantic
# error and f32 noise are both fully present. Widening the list would silently
# move that anchor to a short case and cost the margin table its worst case. So
# breadth is a second fixture in the SAME binary format, read by the same
# loader, and the matrix above keeps its meaning.
#
# The sweep is the old f32 fixture's T formula verbatim - `gen_reference.rs:482`,
# `let seq_len = (1usize << (i % 6)) + (i % 7);` - so the breadth these two tests
# cover is preserved EXACTLY and the only thing that changes is which arm the
# expected value comes from. It spans T in 1..=38 (the `gen_reference.rs:482`
# comment says "2..70"; the formula's real max is 32 + 6 = 38, which is the
# figure `tests/bit_exact.rs` used and the one measured here), so chunk sizes
# 4/8/16/32 each get straddled many times and 64 does not - as before. The
# 18-case matrix above is what covers chunk 64 (T = 64, 65, 70) and the long
# state carry, and `tests/ref_f64.rs` runs BOTH arms over it.
BROAD_CASES = 1000


def broad_seq_len(i):
    return (1 << (i % 6)) + (i % 7)

MASK64 = (1 << 64) - 1


class Rng:
    """splitmix64 + Box-Muller. stdlib-only and provably stable, so the fixture
    is byte-reproducible forever without pinning a numpy or torch version."""

    def __init__(self, seed):
        self.s = seed & MASK64

    def _next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def _u01(self, n):
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = (self._next() >> 11) * (2.0 ** -53)
        return out

    def uniform(self, lo, hi, shape):
        return lo + (hi - lo) * self._u01(int(np.prod(shape))).reshape(shape)

    def normal(self, shape):
        n = int(np.prod(shape))
        u = self._u01(2 * n).reshape(n, 2)
        u1 = np.clip(u[:, 0], 1e-300, 1.0)
        r = np.sqrt(-2.0 * np.log(u1))
        return (r * np.cos(2.0 * math.pi * u[:, 1])).reshape(shape)


def xavier(rng, out_f, in_f, gain):
    """torch's `nn.init.xavier_uniform_(w, gain)` on a [out, in] matrix."""
    a = gain * math.sqrt(6.0 / (in_f + out_f))
    return rng.uniform(-a, a, (out_f, in_f))


def softplus(z):
    """`F.softplus(z, beta=1)`. `logaddexp(0, z)` is the stable form; above the
    torch threshold of 20 it differs from `z` by < 2e-9 absolute, which is far
    below the f32 the authors themselves use for this quantity."""
    return np.logaddexp(0.0, z)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def short_conv(x, w, pad="zeros"):
    """Causal depthwise conv, kernel CONV_W, then SiLU.

    x: [T, C], w: [C, CONV_W]. Returns [T, C].

    out[t] = sum_i w[:, i] * x[t - CONV_W + 1 + i], with x[s] = 0 for s < 0
    (ZERO padding - see the module docstring for the three sources). Tap
    CONV_W-1 multiplies the CURRENT token, because `F.conv1d` is a
    cross-correlation and the causal left pad is CONV_W-1 wide.
    """
    t, c = x.shape
    assert w.shape == (c, CONV_W), w.shape
    if pad == "replicate":
        xp = np.concatenate([np.repeat(x[:1], CONV_W - 1, axis=0), x], axis=0)
    else:
        xp = np.concatenate([np.zeros((CONV_W - 1, c)), x], axis=0)
    out = np.zeros((t, c))
    for i in range(CONV_W):
        out += xp[i : i + t] * w[:, i]
    return silu(out)


def gdn2_forward(x, P, fault=None):
    """The whole layer, f64 throughout. x: [T, D] -> [T, D]."""
    t = x.shape[0]
    g_pre = P["f_proj_1"] @ (P["f_proj_0"] @ x.T)          # [KD, T]
    # Eq. 12 / App. C.1. A is per head, broadcast over that head's d_k.
    decay = np.repeat(np.exp(P["A_log"]), HK)
    g = -decay[:, None] * softplus(g_pre + P["dt_bias"][:, None])

    if fault == "transposed-proj":
        # W instead of W^T, applied to the projections where that is still
        # shape-legal - q, k and b are all [64, 64] in this config. That is the
        # whole danger: a transposed projection is only a SILENT wrong answer on
        # a square matrix, and the key side usually is.
        qp, kp = x @ P["q_proj"], x @ P["k_proj"]
        bp = x @ P["b_proj"]
        vp, wp = (P["v_proj"] @ x.T).T, (P["w_proj"] @ x.T).T
    else:
        qp, kp, vp = (P["q_proj"] @ x.T).T, (P["k_proj"] @ x.T).T, (P["v_proj"] @ x.T).T
        bp, wp = (P["b_proj"] @ x.T).T, (P["w_proj"] @ x.T).T

    if USE_SHORT_CONV:
        pad = "replicate" if fault == "conv-padding" else "zeros"
        q = short_conv(qp, P["q_conv_w"], pad)
        k = short_conv(kp, P["k_conv_w"], pad)
        v = short_conv(vp, P["v_conv_w"], pad)
    else:
        q, k, v = silu(qp), silu(kp), silu(vp)

    b = sigmoid(bp)                                       # [T, KD]
    wg = sigmoid(wp)                                      # [T, VD]

    if fault == "decay-sign":                              # wrong sign on g
        g = -g

    # -> per head, [H, T, *]
    #
    # ALL SIX use the same idiom, and all six are [T, flat] token-major buffers
    # being split as (T, H, HK) and then transposed to [H, T, HK]. `g` used to be
    # `g.T.reshape(H, t, HK)` instead - a head-major reshape of the same
    # token-major buffer. That is correct for no t and wrong for every t > 1;
    # it agrees with the correct form ONLY at t == 1, where the two indexings
    # coincide on every element. Arithmetic, H=4 HK=16 t=2, input g[kd, i]:
    #     out[h, 0, 0] = g[(16*(h*t+0)+0) % 64, (h*t+0)//4] = g[32*h, h*t//4]
    # so head 1 read g[32, ...] where it must read g[16, ...]. Verified
    # numerically: the two forms are bit-identical at t=1 (max|d| = 0.0) and
    # differ by 1.037e-02 at t=2, rising to 2.1e-01 at t=70.
    #
    # This is the `ff7cd57` defect again, in a file written after that fix:
    # `tools/gen_reference.rs` read token-major scan inputs with head-major
    # offsets, and exactly the 24 single-token cases there coincided. The
    # consequence for that test was a false RED; here it was a false FIXTURE,
    # and the cost was that `tests/ref_f64.rs` was red on every case from the
    # second one onwards for a reason that had nothing to do with the kernel.
    # See the STATUS block in tests/ref_f64.rs.
    q = q.reshape(t, H, HK).transpose(1, 0, 2)
    k = k.reshape(t, H, HK).transpose(1, 0, 2)
    g = g.T.reshape(t, H, HK).transpose(1, 0, 2)
    b = b.reshape(t, H, HK).transpose(1, 0, 2)
    v = v.reshape(t, HV, VH).transpose(1, 0, 2)
    wg = wg.reshape(t, HV, VH).transpose(1, 0, 2)

    # GVA: key-side repeated across value-head groups; v and w already there.
    if HV > H:
        rep = HV // H
        q = np.repeat(q, rep, axis=0)
        k = np.repeat(k, rep, axis=0)
        g = np.repeat(g, rep, axis=0)
        b = np.repeat(b, rep, axis=0)

    # App. D.2. eps goes INSIDE the sqrt (fla/modules/l2norm.py).
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + L2_EPS)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + L2_EPS)

    if ALLOW_NEG_EIGVAL:
        b = b * 2.0

    scale = HK ** -0.5
    if fault == "no-scale":
        scale = 1.0

    # ---- Eq. 9, one token at a time, f64 --------------------------------
    S = np.zeros((HV, HK, VH))
    outs = np.empty((HV, t, VH))
    for i in range(t):
        Sb = S * np.exp(g[:, i])[:, :, None]          # S̄ = Diag(α_t) S, Eq. 9
        e = b[:, i] * k[:, i] if fault != "no-erase-gate" else k[:, i]
        r = np.einsum("hkv,hk->hv", Sb, e)            # r = S̄^T e
        z = wg[:, i] * v[:, i]                        # z = w ⊙ v
        if fault == "no-write-gate":
            z = v[:, i]
        S = Sb + k[:, i][:, :, None] * (z - r)[:, None, :]   # S̄ + k (z-r)^T
        # The readout is S^T q, from the state AFTER the write. Reading S̄^T q
        # instead is the `gr.rs` bug class (docs/protocols/ORACLE.md B2).
        src = Sb if fault == "read-before-write" else S
        outs[:, i] = np.einsum("hkv,hk->hv", src, q[:, i])
    outs = outs.transpose(1, 0, 2) * scale             # the WHOLE readout

    # ---- FusedRMSNormSwishGate -----------------------------------------
    # g_proj is a bare Sequential (Linear, Linear-with-bias) with NO activation
    # between the two - lit_gpt/gdn2.py __init__ and its `self.g_proj(x)` call.
    gate = (P["g_proj_1"] @ (P["g_proj_0"] @ x.T) + P["g_proj_1_b"][:, None]).T
    gate = gate.reshape(t, HV, VH)
    rms = np.sqrt((outs * outs).mean(-1, keepdims=True) + NORM_EPS)
    # The `sigmoid` branch is the OTHER credible upstream's layer-level choice
    # (`fla/layers/gdn2.py:197`, @ 9f38d249) and is an unresolved A/B arm - see
    # the docstring. It lives here, in the FAULT list, so the distance between
    # the two branches is a committed number on these weights and this input
    # rather than a sentence in a docstring. `silu` is the branch we implement
    # and the branch `lit_gpt/gdn2.py:212` (@ a5552fe3) selects.
    act = sigmoid(gate) if fault == "output-gate-sigmoid" else silu(gate)
    outs = outs / rms * P["o_norm_w"] * act
    return (P["o_proj"] @ outs.reshape(t, VD).T).T


def make_params(rng):
    """The authors' own init: xavier_uniform gain 2**-2.5, zero biases,
    A_log = log(U(1,16)), dt_bias from the dt parameterisation, o_norm = ones.
    Paper App. D.5. Layer weights are stored in burn's [out, in] layout."""
    gain = 2.0 ** -2.5
    P = {}
    for name, out_f, in_f in [
        ("q_proj", KD, D), ("k_proj", KD, D), ("v_proj", VD, D),
        ("f_proj_0", VH, D), ("f_proj_1", KD, VH),
        ("b_proj", KD, D), ("w_proj", VD, D),
        ("g_proj_0", VH, D), ("g_proj_1", VD, VH),
        ("o_proj", D, VD),
    ]:
        P[name] = xavier(rng, out_f, in_f, gain)
    # g_proj's bias is the ONE place this fixture departs from the authors'
    # init, and only for conditioning. `lit_gpt/gdn2.py` zeroes every bias, which
    # puts the output gate at silu(0) = 0 exactly: a freshly-initialised layer's
    # output is then ~7e-4 where its pre-gate value is O(1), i.e. the output is a
    # near-null quantity and a RELATIVE bar through it is meaningless (it reads
    # 1e+0 for two implementations that differ by f32 noise). Moving the gate
    # bias off silu's zero crossing fixes the conditioning. It changes no
    # formula, no projection and no gate - only the scale the output is measured
    # against. Measured: max|out| goes from 7.5e-04 to O(1).
    P["g_proj_1_b"] = rng.uniform(0.5, 1.5, VD)

    P["A_log"] = np.log(rng.uniform(1.0, 16.0, (H,)))
    dt = np.clip(
        np.exp(rng.uniform(0, 1, (KD,)) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)),
        1e-4, None,
    )
    P["dt_bias"] = dt + np.log(-np.expm1(-dt))
    P["o_norm_w"] = np.ones(VH)
    if USE_SHORT_CONV:
        for name, c in [("q_conv_w", KD), ("k_conv_w", KD), ("v_conv_w", VD)]:
            P[name] = rng.uniform(-0.5, 0.5, (c, CONV_W))
    return P


# --- fixture format ----------------------------------------------------------
# All little-endian. Weights and inputs f32 (so both arms start from identical
# bits); the OUTPUTS f64, which is the entire point of this layer.
#
# Linear weights are stored in burn's on-disk `Linear` layout,
# `[d_input, d_output]` — the same convention as the sibling `ref_data.bin`, so
# a `LinearConfig::new(shape[0], shape[1])` reproduces the tensor exactly. The
# 1-D parameters and the depthwise conv weights keep their natural order.
#   "GDN2F64\0" | u32 d,h,hk,hv | f64 expand_v | u8 use_sc, allow_neg
#   | u8 n_tensors | { u32 namelen, name, u32 ndim, u32 numel, i32[ndim], f32[numel] }
#   | u32 n_cases  | { u32 t, f32[d*t] x, f64[d*t] y }
TENSOR_ORDER = [
    "q_proj", "k_proj", "v_proj", "f_proj_0", "f_proj_1", "b_proj", "w_proj",
    "g_proj_0", "g_proj_1", "g_proj_1_b", "A_log", "dt_bias", "o_norm_w", "o_proj",
    "q_conv_w", "k_conv_w", "v_conv_w",
]
# The subset that is a burn `Linear`, and therefore stored [d_input, d_output].
LINEARS = {
    "q_proj", "k_proj", "v_proj", "f_proj_0", "f_proj_1", "b_proj", "w_proj",
    "g_proj_0", "g_proj_1", "o_proj",
}


def make_inputs(seq_lens=SEQ_LENS):
    """The fixture's inputs, in fixture order, from ONE rng stream consumed once.

    Everything that needs case i's input must go through this. It used to be
    re-drawn per consumer, and `write_fault_fixture` drew only the case it
    wanted as the stream's FIRST draw - so the "wrong formula" outputs were
    computed on a different input than the "right formula" ones, and every fault
    read O(1) for the wrong reason. The fault fixture now carries a hash of the
    input it used and `tests/ref_f64.rs` checks it against the main fixture, so
    that class of mistake fails loudly instead of looking like a huge margin.

    `seq_lens` is a parameter so the breadth sweep draws its own stream
    positionally the same way; the default keeps the 18-case matrix.
    """
    rng = Rng(SEED)
    return [rng.normal((t, D)).astype(np.float32) for t in seq_lens]


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & MASK64
    return h


def write_fixture(path, P, fault=None, seq_lens=SEQ_LENS):
    xs = make_inputs(seq_lens)
    cases = [(t, x, gdn2_forward(x.astype(np.float64), P, fault))
             for (t, x) in zip(seq_lens, xs)]
    with open(path, "wb") as f:
        f.write(b"GDN2F64\0")
        f.write(struct.pack("<4I", D, H, HK, HV))
        f.write(struct.pack("<d", EXPAND_V))
        f.write(bytes([int(USE_SHORT_CONV), int(ALLOW_NEG_EIGVAL), len(TENSOR_ORDER)]))
        for name in TENSOR_ORDER:
            a = np.ascontiguousarray(P[name], dtype=np.float32)
            if name in LINEARS:
                a = np.ascontiguousarray(a.T)
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(struct.pack("<2I", a.ndim, a.size))
            f.write(struct.pack(f"<{a.ndim}i", *a.shape))
            f.write(a.astype("<f4").tobytes())
        f.write(struct.pack("<I", len(cases)))
        for t, x32, y in cases:
            f.write(struct.pack("<I", t))
            f.write(np.ascontiguousarray(x32, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(y, dtype="<f8").tobytes())
    return cases


FAULTS = ["decay-sign", "transposed-proj", "no-write-gate", "no-erase-gate",
          "read-before-write", "conv-padding", "no-scale", "output-gate-sigmoid"]

# The case whose wrong-formula outputs get committed. The longest one: it is
# the only length where a semantic error and f32 noise are both fully present
# (a long state carry plus every cross-token conv tap).
FAULT_CASE = len(SEQ_LENS) - 1


def write_fault_fixture(path, P):
    """The other side of the margin, committed.

    `ref_f64.bin` says what the layer should produce. This says how far off a
    WRONG formula lands, on the same weights and the SAME input. The Rust test
    asserts our f32 output is within the bar of the first and outside the bar
    of every one of these, which is what makes the bar falsifiable rather than
    a self-consistency check - and it keeps that true without a second
    implementation living in the tree.

    The input hash is not decoration. The first version of this file re-drew
    the RNG for this case instead of consuming the stream up to it, so these
    outputs were computed on a DIFFERENT input and every fault read O(1)
    for a reason that had nothing to do with the formulas. The test now
    recomputes this hash from `ref_f64.bin` and refuses to compare if it
    differs."""
    t = SEQ_LENS[FAULT_CASE]
    x = make_inputs()[FAULT_CASE]
    h = fnv1a64(np.ascontiguousarray(x, dtype="<f4").tobytes())
    with open(path, "wb") as f:
        f.write(b"GDN2FLT\0")
        f.write(struct.pack("<3IQ", FAULT_CASE, len(FAULTS), t, h))
        for name in FAULTS:
            nb = name.encode()
            y = gdn2_forward(x.astype(np.float64), P, name)
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(np.ascontiguousarray(y, dtype="<f8").tobytes())
    print(f"wrote {path}: {len(FAULTS)} wrong formulas on case {FAULT_CASE} (T={t}, "
          f"input fnv1a64 = {h:#018x})")


def stages(x, P):
    """Every intermediate of the layer, in f64, for one case.

    Returned in the same order `examples/ref_f64_stages.rs` dumps them, so
    `--diff-stages` is a positional diff. The point is to name the FIRST stage
    that disagrees: that is the defect, and everything after it is downstream
    noise."""
    t = x.shape[0]
    out = {}
    qp = (P["q_proj"] @ x.T).T
    kp = (P["k_proj"] @ x.T).T
    vp = (P["v_proj"] @ x.T).T
    out["raw_q_proj"] = qp
    out["raw_k_proj"] = kp
    out["raw_v_proj"] = vp
    out["f0"] = (P["f_proj_0"] @ x.T).T
    out["f1"] = (P["f_proj_1"] @ (P["f_proj_0"] @ x.T)).T
    out["gp0"] = (P["g_proj_0"] @ x.T).T
    out["gp1"] = (P["g_proj_1"] @ (P["g_proj_0"] @ x.T) + P["g_proj_1_b"][:, None]).T
    out["a_exp"] = np.repeat(np.exp(P["A_log"]), HK)
    out["dt_bias"] = P["dt_bias"]
    out["A_log"] = P["A_log"]
    out["q_conv"] = short_conv(qp, P["q_conv_w"])
    out["k_conv"] = short_conv(kp, P["k_conv_w"])
    out["v_conv"] = short_conv(vp, P["v_conv_w"])
    q, k, v = out["q_conv"], out["k_conv"], out["v_conv"]
    b = sigmoid((P["b_proj"] @ x.T).T)
    wg = sigmoid((P["w_proj"] @ x.T).T)
    g_pre = P["f_proj_1"] @ (P["f_proj_0"] @ x.T)
    g = -np.repeat(np.exp(P["A_log"]), HK)[:, None] * softplus(g_pre + P["dt_bias"][:, None])
    out["g_recomputed"] = g.T          # [T,KD] == g_unpermuted

    # The 4-D per-head tensors, in the SAME [H, T, HK] / [HV, T, VH] layout the
    # crate hands to the recurrence — so the Rust side only has to
    # `.contiguous()` them and read them back. These are the stages the
    # localisation could not see before: `diff_stages` used to compare only
    # 2-D and 1-D rows, on the grounds that a permuted 4-D view reads back
    # unstably on ndarray. Forcing a copy on the Rust side removes the
    # ambiguity, and without these rows a per-head layout defect or a wrong GVA
    # repeat is invisible to the whole diagnostic.
    def to_heads(a, n, d):
        """[T, n*d] -> [n, T, d], the crate's `to_4d`.

        The transposed result's flat order is h*(T*d) + t*d + d, which IS the
        crate's [B, n, T, d] flat order, so `diff_stages` can compare it
        positionally. (Contrast `gate4d`, which is token-major on the Rust side
        and must stay token-major here.)"""
        return a.reshape(t, n, d).transpose(1, 0, 2)

    # Per head, the way `to_4d` does it. `out["q_conv"]` and friends are [T, KD]
    # token-major, so they go through `to_heads` before the per-channel
    # operations; doing the L2 over the flat [T, KD] axis normalises across
    # HEADS, which is a different quantity and reads as a garbage-scale row
    # rather than a finding.
    q4, k4 = to_heads(out["q_conv"], H, HK), to_heads(out["k_conv"], H, HK)
    v4 = to_heads(out["v_conv"], HV, VH)  # the VALUE head is VH = HK*expand_v
    g4 = g.T.reshape(t, H, HK).transpose(1, 0, 2)
    b4 = sigmoid((P["b_proj"] @ x.T).T).reshape(t, H, HK).transpose(1, 0, 2)
    w4 = sigmoid((P["w_proj"] @ x.T).T).reshape(t, HV, VH).transpose(1, 0, 2)
    if HV > H:  # the GVA repeat; an identity in this crate's own config (H == HV)
        rep = HV // H
        for a in (q4, k4, g4, b4):
            a[:] = np.repeat(a, rep, axis=0)
    # NO `q4d_norm` ROW, and the reason is worth keeping: it looks like the
    # obvious thing to add (dump `l2_normalize_4d`'s divisor separately) and it
    # is a trap. `project` returns q and k ALREADY normalised, so a norm taken
    # from `projected.q` is ~1.0 by construction while the f64 reference's
    # pre-L2 norm is ~1e-4 - the row then reads O(1) and names a defect that is
    # not there, which is exactly how two mutually inconsistent numbers were
    # produced during this localisation. Measuring the divisor needs the
    # PRE-L2 tensor, which `project` does not expose; `q4d` agreeing to ~2e-07
    # is the evidence that the normalisation is right, because it is the
    # quotient.
    # post-L2, which is where `project` leaves them (l2_normalize_4d, 1e-6)
    q4 = q4 / np.sqrt((q4 * q4).sum(-1, keepdims=True) + L2_EPS)
    k4 = k4 / np.sqrt((k4 * k4).sum(-1, keepdims=True) + L2_EPS)
    out["q4d"], out["k4d"], out["g4d"] = q4, k4, g4
    out["b4d"], out["v4d"], out["w4d"] = b4, v4, w4
    # Token-major, NOT [HV, T, VH]. `diff_stages` compares positionally after
    # `got.reshape(ref.shape)`, and the crate's tensors are [B, T, HV, VH], whose
    # flat order is t*(HV*VH) + h*VH + v. A transposed [HV, T, VH] reference has
    # the flat order h*(t*VH) + t*VH + v - the same values in a different order,
    # so the row reads O(1) and names a defect that is not there. Measured:
    # that mistake put `gate4d` at 7.5e-01 at T=70 while the 2-D `gp1` row, same
    # tensor, sat at 1.6e-07. The per-head rows below are [H, T, D] for the same
    # reason: `to_heads` is applied to a token-major [T, n*d] array, so its flat
    # order already matches.
    out["gate4d"] = (
        (P["g_proj_1"] @ (P["g_proj_0"] @ x.T) + P["g_proj_1_b"][:, None])
        .T
        .reshape(t, HV, VH)
    )
    return out


# The stages `examples/ref_f64_stages.rs` dumps SQUARED, because it has to push
# a permuted 4-D view through an elementwise op to get a dense readback out of
# burn 0.22 (no `contiguous()`), and squaring is the one op that cannot be
# optimised away. `diff_stages` squares the reference to match.
SQUARED_STAGES = {"q4d", "k4d", "g4d", "b4d", "v4d", "w4d", "gate4d"}


def diff_stages(path, P):
    """Read the crate's dumped stages and report max |ours - f64| per stage."""
    ours = {}
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        ours[parts[0]] = np.array([float(v) for v in parts[1:]], dtype=np.float64)
    xs = [x.astype(np.float64) for x in make_inputs()]
    print(f"{'stage':<14} {'max|ours-ref|':>14} {'max|ref|':>12} {'rel':>10}")
    for key, got in ours.items():
        ci = int(key[1 : key.index("/")])
        stage = key[key.index("/") + 1 :]
        ref = stages(xs[ci], P)[stage]
        # The 4-D per-head stages are dumped SQUARED (the example pushes the
        # permuted view through `powf_scalar(2.0)` because burn 0.22 has no
        # `contiguous()`, and an elementwise op is the only thing guaranteed to
        # write a dense buffer in logical order), so square the reference to
        # match. The row is self-verifying: a readback that permutes the data
        # gives an O(1) diff here rather than a silent pass.
        if stage in SQUARED_STAGES:
            ref = ref * ref
        d = np.abs(got.reshape(ref.shape) - ref)
        scale = max(np.abs(ref).max(), 1e-30)
        print(f"{stage:<14} {d.max():>14.3e} {scale:>12.3e} {d.max()/scale:>10.3e}")


def self_test(P):
    """The margin. Prints max |wrong - right| / max |right| for each fault, so
    the bar's two sides are measured rather than asserted in prose."""
    rng = Rng(SEED + 1)
    xs = [rng.normal((t, D)) for t in SEQ_LENS]
    ref = np.concatenate([gdn2_forward(x, P) for x in xs])
    denom = np.abs(ref).max()
    rows = [("f32 reassociation (est.)", None)]
    for fault in FAULTS:
        got = np.concatenate([gdn2_forward(x, P, fault) for x in xs])
        rel = np.abs(got - ref).max() / denom
        rows.append((fault, rel))
    print(f"reference output scale: max|out| = {denom:.6g}")
    print(f"{'perturbation':<32} {'max rel dev':>12}")
    for name, rel in rows:
        print(f"{name:<32} {'(measured in Rust)':>12}" if rel is None else f"{name:<32} {rel:>12.3e}")
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output path (default tests/ref_f64.bin, or "
                         "tests/ref_f64_broad.bin with --broad)")
    ap.add_argument("--broad", action="store_true",
                    help="write the 1000-case breadth sweep instead of the "
                         "18-case length matrix")
    ap.add_argument("--fault", default=None,
                    choices=["decay-sign", "transposed-proj", "no-write-gate",
                             "no-erase-gate", "read-before-write", "conv-padding",
                             "no-scale", "output-gate-sigmoid"])
    ap.add_argument("--faults-out", default="tests/ref_f64_faults.bin",
                    help="where to write the wrong-formula fixture")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--diff-stages", metavar="FILE",
                    help="diff the crate's dumped project() stages against f64")
    args = ap.parse_args()

    P = make_params(Rng(SEED))
    if args.diff_stages:
        diff_stages(args.diff_stages, P)
        return
    if args.self_test:
        self_test(P)
        return
    if args.broad:
        out = args.out or "tests/ref_f64_broad.bin"
        lens = [broad_seq_len(i) for i in range(BROAD_CASES)]
        cases = write_fixture(out, P, args.fault, lens)
        assert args.fault is None, (
            "--broad never writes the wrong-formula fixture: it is committed "
            "against the 18-case matrix's FAULT_CASE, not this sweep")
        print(f"wrote {out}: {len(cases)} cases, d={D} h={H} hk={HK} hv={HV} "
              f"expand_v={EXPAND_V}, T in {min(lens)}..{max(lens)}")
        return
    out = args.out or "tests/ref_f64.bin"
    cases = write_fixture(out, P, args.fault)
    tag = f" WITH FAULT '{args.fault}'" if args.fault else ""
    print(f"wrote {out}{tag}: {len(cases)} cases, d={D} h={H} hk={HK} hv={HV} "
          f"expand_v={EXPAND_V}, T in {SEQ_LENS[0]}..{SEQ_LENS[-1]}")
    if args.fault is None:
        write_fault_fixture(args.faults_out, P)


if __name__ == "__main__":
    main()
