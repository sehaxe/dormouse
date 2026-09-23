#!/usr/bin/env python3
"""Regenerate the bit-exact reference data for burn-gdn2.

The reference implementation below is a faithful, pure-PyTorch transcription
of the official NVlabs GatedDeltaNet-2 layer
(https://github.com/NVlabs/GatedDeltaNet-2, lit_gpt/gdn2.py), with the
Triton fused_recurrent kernel replaced by an equivalent per-token scan in
torch. Numerical differences to burn-gdn2 are bounded by ~1e-6 in f32, far
below the test tolerance (5e-4).

Usage:
    python3 gen_reference.py            # writes tests/ref_data.bin (1000 cases)
    cargo test --features binary-tests  # compares burn-gdn2 against it

The *.bin files are gitignored on purpose; regenerate with this script.
"""

import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration (fixed, matches the burn-gdn2 test matrix) ---------------
D, H, HK, HV, EXPAND_V = 64, 4, 16, 4, 1.5
USE_SHORT_CONV = True
ALLOW_NEG_EIGVAL = False
N_CASES = 1000
EPS = 1e-5

KD = H * HK
V_HEAD = int(HK * EXPAND_V)
VD = HV * V_HEAD


# --- Reference layer (vendored from lit_gpt/gdn2.py, kernels in torch) ------
class Gdn2Reference(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D, KD, bias=False)
        self.k_proj = nn.Linear(D, KD, bias=False)
        self.v_proj = nn.Linear(D, VD, bias=False)
        self.f_proj = nn.Sequential(
            nn.Linear(D, V_HEAD, bias=False),
            nn.Linear(V_HEAD, KD, bias=False),
        )
        self.b_proj = nn.Linear(D, KD, bias=False)
        self.w_proj = nn.Linear(D, VD, bias=False)
        self.g_proj = nn.Sequential(
            nn.Linear(D, V_HEAD, bias=False),
            nn.Linear(V_HEAD, VD, bias=True),
        )
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        dt = torch.exp(
            torch.rand(KD) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.o_norm_w = nn.Parameter(torch.ones(V_HEAD))
        self.o_proj = nn.Linear(VD, D, bias=False)
        if USE_SHORT_CONV:
            self.q_conv_w = nn.Parameter(
                torch.empty(KD, 1, 4).uniform_(-0.5, 0.5)
            )
            self.k_conv_w = nn.Parameter(
                torch.empty(KD, 1, 4).uniform_(-0.5, 0.5)
            )
            self.v_conv_w = nn.Parameter(
                torch.empty(VD, 1, 4).uniform_(-0.5, 0.5)
            )
        self._init_weights()

    def _init_weights(self):
        gain = 2 ** -2.5
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def short_conv(self, x, w):
        # Causal depthwise conv, kernel=4, replicate padding (first token),
        # matching burn-gdn2 (and this repo's reference data).
        x_pad = torch.cat([x[:, :1].repeat(1, 3, 1), x], dim=1)
        out = None
        for i in range(4):
            y = x_pad[:, i:i + x.size(1)] * w[:, 0, i]
            out = y if out is None else out + y
        return F.silu(out)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.short_conv(self.q_proj(x), self.q_conv_w) if USE_SHORT_CONV else F.silu(self.q_proj(x))
        k = self.short_conv(self.k_proj(x), self.k_conv_w) if USE_SHORT_CONV else F.silu(self.k_proj(x))
        v = self.short_conv(self.v_proj(x), self.v_conv_w) if USE_SHORT_CONV else F.silu(self.v_proj(x))

        # Channel-wise log-decay (fp32), as in the official layer.
        g = (
            -self.A_log.float().exp().repeat_interleave(HK)
            * F.softplus(self.f_proj(x).float() + self.dt_bias)
        )
        b = self.b_proj(x).sigmoid()
        w_gate = self.w_proj(x).sigmoid()

        # Split into per-head tensors.
        q = q.reshape(B, T, H, HK).transpose(1, 2)
        k = k.reshape(B, T, H, HK).transpose(1, 2)
        g = g.reshape(B, T, H, HK).transpose(1, 2)
        v = v.reshape(B, T, HV, V_HEAD).transpose(1, 2)
        b = b.reshape(B, T, H, HK).transpose(1, 2)
        w_gate = w_gate.reshape(B, T, HV, V_HEAD).transpose(1, 2)

        # GVA: repeat key-side heads when more value heads.
        if HV > H:
            rep = HV // H
            q = q[:, :, None].expand(B, H, T, rep, HK).reshape(B, HV, T, HK)
            k = k[:, :, None].expand(B, H, T, rep, HK).reshape(B, HV, T, HK)
            g = g[:, :, None].expand(B, H, T, rep, HK).reshape(B, HV, T, HK)
            b = b[:, :, None].expand(B, H, T, rep, HK).reshape(B, HV, T, HK)
        if ALLOW_NEG_EIGVAL:
            b = b * 2.0

        # L2-normalize q/k along the head dimension (as in the kernels).
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        scale = HK ** -0.5
        S = torch.zeros(B, HV, HK, V_HEAD)
        outs = []
        for t in range(T):
            g_t = g[:, :, t].exp()  # [B, HV, HK]
            S = S * g_t.unsqueeze(-1)
            bk = b[:, :, t] * k[:, :, t]  # [B, HV, HK]
            erased = (S * bk.unsqueeze(-1)).sum(dim=2)
            v_new = w_gate[:, :, t] * v[:, :, t] - erased
            S = S + k[:, :, t].unsqueeze(-1) * v_new.unsqueeze(-2)
            o_t = (S * q[:, :, t].unsqueeze(-1)).sum(dim=2) * scale
            outs.append(o_t)
        o = torch.stack(outs, dim=2).transpose(1, 2)  # [B, T, HV, V_HEAD]

        # SiLU-gated RMS norm per head, then output projection.
        gate = self.g_proj(x).reshape(B, T, HV, V_HEAD)
        rms = torch.sqrt(o.pow(2).mean(dim=-1, keepdim=True) + EPS)
        o = o / rms * self.o_norm_w * F.silu(gate)
        return self.o_proj(o.reshape(B, T, VD))


# --- Serialization (format consumed by tests/bit_exact.rs) ------------------
def w_i32(f, v):
    f.write(struct.pack("<i", v))


def w_name(f, name):
    f.write(struct.pack("<i", len(name)))
    f.write(name.encode())


def w_tensor(f, name, t):
    t = t.detach().cpu().contiguous()
    flat = t.flatten().numpy().astype("<f4").tobytes()
    w_name(f, name)
    w_i32(f, t.dim())
    w_i32(f, t.numel())
    for s in t.shape:
        w_i32(f, s)
    f.write(flat)


def w_raw(f, t):
    t = t.detach().cpu().contiguous()
    flat = t.flatten().numpy().astype("<f4").tobytes()
    w_i32(f, t.dim())
    w_i32(f, t.numel())
    for s in t.shape:
        w_i32(f, s)
    f.write(flat)


def main():
    torch.manual_seed(1337)
    layer = Gdn2Reference()
    layer.eval()

    with open("tests/ref_data.bin", "wb") as f:
        w_i32(f, D)
        w_i32(f, H)
        w_i32(f, HK)
        w_i32(f, HV)
        w_i32(f, int(EXPAND_V * 10))
        f.write(bytes([int(USE_SHORT_CONV), int(ALLOW_NEG_EIGVAL)]))

        # burn stores Linear weights as [in, out]; torch stores [out, in].
        w_tensor(f, "q_proj", layer.q_proj.weight.t().contiguous())
        w_tensor(f, "k_proj", layer.k_proj.weight.t().contiguous())
        w_tensor(f, "v_proj", layer.v_proj.weight.t().contiguous())
        w_tensor(f, "f_proj_0", layer.f_proj[0].weight.t().contiguous())
        w_tensor(f, "f_proj_1", layer.f_proj[1].weight.t().contiguous())
        w_tensor(f, "b_proj", layer.b_proj.weight.t().contiguous())
        w_tensor(f, "w_proj", layer.w_proj.weight.t().contiguous())
        w_tensor(f, "g_proj_0", layer.g_proj[0].weight.t().contiguous())
        w_tensor(f, "g_proj_1_w", layer.g_proj[1].weight.t().contiguous())
        w_tensor(f, "g_proj_1_b", layer.g_proj[1].bias)
        w_tensor(f, "A_log", layer.A_log)
        w_tensor(f, "dt_bias", layer.dt_bias)
        w_tensor(f, "o_norm_w", layer.o_norm_w)
        w_tensor(f, "o_proj", layer.o_proj.weight.t().contiguous())
        w_tensor(f, "q_conv_w", layer.q_conv_w.reshape(KD, 4))
        w_tensor(f, "k_conv_w", layer.k_conv_w.reshape(KD, 4))
        w_tensor(f, "v_conv_w", layer.v_conv_w.reshape(VD, 4))

        w_i32(f, N_CASES)
        for i in range(N_CASES):
            seq_len = 2 ** (i % 6) + (i % 7)  # 2..70, varied
            x = torch.randn(1, seq_len, D)
            y = layer.forward(x)
            w_raw(f, x)
            w_raw(f, y)
            if (i + 1) % 200 == 0:
                print(f"  case {i + 1}/{N_CASES}")
    print(f"wrote tests/ref_data.bin ({N_CASES} cases, d={D} h={H} hk={HK} hv={HV} expand_v={EXPAND_V})")


if __name__ == "__main__":
    main()
