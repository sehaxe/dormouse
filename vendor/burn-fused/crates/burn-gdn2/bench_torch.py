#!/usr/bin/env python3
"""Compare burn-gdn2 (CUDA) against equivalent pure-PyTorch GDN-2 layers.

Two torch references, same GPU, same configs:
  - chunked WY forward  (mirrors burn's `chunk_wy_forward` math, cuBLAS)
  - chunked WY training (forward + autograd backward)

Run burn side:
  cargo test --release --features "cuda,autodiff" -p burn-gdn2 --test bench_train_cuda -- --ignored --nocapture
"""
import time

import torch
import torch.nn.functional as F


class Gdn2Torch(torch.nn.Module):
    def __init__(self, d, h, hk, ev=1.0, kv=None):
        super().__init__()
        hv = kv or h
        kd, vh, vd = h * hk, int(hk * ev), (kv or h) * int(hk * ev)
        self.d, self.h, self.hk, self.hv, self.vh = d, h, hk, hv, vh
        self.kd, self.vd = kd, vd
        self.q_w = torch.nn.Parameter(torch.randn(kd, d) * 0.05)
        self.k_w = torch.nn.Parameter(torch.randn(kd, d) * 0.05)
        self.v_w = torch.nn.Parameter(torch.randn(vd, d) * 0.05)
        self.f_w0 = torch.nn.Parameter(torch.randn(vh, d) * 0.05)
        self.f_w1 = torch.nn.Parameter(torch.randn(kd, vh) * 0.05)
        self.b_w = torch.nn.Parameter(torch.randn(kd, d) * 0.05)
        self.w_w = torch.nn.Parameter(torch.randn(vd, d) * 0.05)
        self.g_w0 = torch.nn.Parameter(torch.randn(vh, d) * 0.05)
        self.g_w1 = torch.nn.Parameter(torch.randn(vd, vh) * 0.05)
        self.g_b1 = torch.nn.Parameter(torch.zeros(vd))
        self.A_log = torch.nn.Parameter(torch.log(torch.rand(h) * 15 + 1))
        dt = torch.exp(torch.rand(kd) * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001))) + torch.log(torch.tensor(0.001))).clamp(min=1e-4)
        self.dt_bias = torch.nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.o_norm_w = torch.nn.Parameter(torch.ones(vh))
        self.o_w = torch.nn.Parameter(torch.randn(d, vd) * 0.05)
        self.q_cw = torch.nn.Parameter(torch.randn(kd, 4) * 0.5)
        self.k_cw = torch.nn.Parameter(torch.randn(kd, 4) * 0.5)
        self.v_cw = torch.nn.Parameter(torch.randn(vd, 4) * 0.5)

    def short_conv(self, x, w):
        x_pad = torch.cat([x[:, :1].repeat(1, 3, 1), x], dim=1)
        out = None
        for i in range(4):
            y = x_pad[:, i:i + x.size(1)] * w[:, i]
            out = y if out is None else out + y
        return F.silu(out)

    def project(self, x):
        q = F.normalize(self.short_conv(x @ self.q_w.t(), self.q_cw), dim=-1)
        k = F.normalize(self.short_conv(x @ self.k_w.t(), self.k_cw), dim=-1)
        v = self.short_conv(x @ self.v_w.t(), self.v_cw)
        g = -self.A_log.exp().repeat_interleave(self.hk) * F.softplus(x @ self.f_w0.t() @ self.f_w1.t() + self.dt_bias)
        b = (x @ self.b_w.t()).sigmoid()
        w_gate = (x @ self.w_w.t()).sigmoid()
        B, T, _ = x.shape
        H, HK, HV, VH = self.h, self.hk, self.hv, self.vh
        r = lambda t: t.reshape(B, T, H, HK).transpose(1, 2)
        q = r(q); k = r(k); v = r(v); g = r(g); b = r(b)
        w_gate = w_gate.reshape(B, T, HV, VH).transpose(1, 2)
        return q, k, v, g, b, w_gate

    def out_norm(self, o, x, B, T):
        gate = (x @ self.g_w0.t() @ self.g_w1.t() + self.g_b1).reshape(B, T, self.hv, self.vh)
        rms = torch.sqrt(o.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        o = o / rms * self.o_norm_w * F.silu(gate)
        return o.reshape(B, T, self.vd) @ self.o_w.t()

    def forward_fused(self, x):
        """Token-by-token scan (slow reference)."""
        B, T, _ = x.shape
        q, k, v, g, b, w_gate = self.project(x)
        H, HK, HV, VH = self.h, self.hk, self.hv, self.vh
        scale = HK ** -0.5
        S = torch.zeros(B, HV, HK, VH, device=x.device)
        outs = []
        for t in range(T):
            S = S * g[:, :, t].exp().unsqueeze(-1)
            bk = b[:, :, t] * k[:, :, t]
            erased = (S * bk.unsqueeze(-1)).sum(dim=2)
            v_new = w_gate[:, :, t] * v[:, :, t] - erased
            S = S + k[:, :, t].unsqueeze(-1) * v_new.unsqueeze(-2)
            outs.append((S * q[:, :, t].unsqueeze(-1)).sum(dim=2) * scale)
        o = torch.stack(outs, dim=2).transpose(1, 2)
        return self.out_norm(o, x, B, T)

    def forward_chunked(self, x, cs=64):
        """Chunked WY forward, mirroring burn's chunk_wy_forward."""
        B, T, _ = x.shape
        q, k, v, g, b, w_gate = self.project(x)
        H, HK, HV, VH = self.h, self.hk, self.hv, self.vh
        scale = HK ** -0.5
        tril = torch.tril(torch.ones(cs, cs, device=x.device))
        strict = torch.tril(torch.ones(cs, cs, device=x.device), -1)
        S = torch.zeros(B, HV, HK, VH, device=x.device)
        outs = []
        for t0 in range(0, T, cs):
            c = min(cs, T - t0)
            q_c = q[:, :, t0:t0 + c]; k_c = k[:, :, t0:t0 + c]
            v_c = v[:, :, t0:t0 + c]; g_c = g[:, :, t0:t0 + c]
            b_c = b[:, :, t0:t0 + c]; w_c = w_gate[:, :, t0:t0 + c]
            E = torch.cumsum(g_c, dim=2).exp()
            kG = k_c / E
            aqk = ((q_c * E) @ kG.transpose(-1, -2)) * scale * tril[:c, :c]
            bk = b_c * k_c
            akk = ((bk * E) @ kG.transpose(-1, -2)) * strict[:c, :c]
            rhs_k = bk * E
            rhs_v = w_c * v_c
            Ws = [rhs_k[:, :, 0]]
            Us = [rhs_v[:, :, 0]]
            for i in range(1, c):
                Ws.append(rhs_k[:, :, i] - torch.einsum('bhi,bhid->bhd', akk[:, :, i, :i], torch.stack(Ws, dim=2)))
                Us.append(rhs_v[:, :, i] - torch.einsum('bhi,bhid->bhd', akk[:, :, i, :i], torch.stack(Us, dim=2)))
            W = torch.stack(Ws, dim=2)
            U = torch.stack(Us, dim=2)
            v_new = U - W @ S
            outs.append(aqk @ v_new + (q_c * E) @ S * scale)
            E_last = E[:, :, c - 1:c]
            S = S * E_last.transpose(-1, -2) + (k_c * (E_last / E)).transpose(-1, -2) @ v_new
        o = torch.cat(outs, dim=2).transpose(1, 2)
        return self.out_norm(o, x, B, T)


def bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    dev = "cuda"
    torch.manual_seed(0)
    for d, h, hk, T, iters in [
        (256, 4, 64, 256, 10),
        (512, 8, 64, 1024, 10),
        (1024, 8, 128, 2048, 5),
        (2048, 16, 128, 4096, 3),
    ]:
        m = Gdn2Torch(d, h, hk).to(dev).train()
        x = torch.randn(1, T, d, device=dev)
        print(f"--- d={d} h={h} hk={hk} T={T} ---")
        # forward, no grad
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            t = bench(lambda: m.forward_chunked(x), iters)
        peak_fwd = torch.cuda.max_memory_allocated() / 1e6
        print(f"{'torch chunked fwd':<28} {t*1e3:>9.1f} ms  {T/t:>12,.0f} tok/s  peak={peak_fwd:.0f} MB")
        with torch.no_grad():
            t = bench(lambda: m.forward_fused(x), max(1, iters // 2))
        print(f"{'torch fused fwd (scan)':<28} {t*1e3:>9.1f} ms  {T/t:>12,.0f} tok/s")
        # training: forward + backward
        def train():
            loss = m.forward_chunked(x).pow(2).mean()
            loss.backward()
        torch.cuda.reset_peak_memory_stats()
        t = bench(train, max(1, iters // 2))
        peak_tr = torch.cuda.max_memory_allocated() / 1e6
        print(f"{'torch chunked train':<28} {t*1e3:>9.1f} ms  {T/t:>12,.0f} tok/s  peak={peak_tr:.0f} MB")


if __name__ == "__main__":
    main()
