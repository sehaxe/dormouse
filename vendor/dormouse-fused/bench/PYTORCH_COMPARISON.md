# dormouse-fused vs PyTorch: head-to-head on RTX 3090

> **Status 2026-10-01.** Measured 2026-08-09; the numbers below are unchanged
> and are not re-measured by any CI job in this copy. The crate list changed
> 2026-09-28 (28 → 20). **The `Sparse attn top-k` row is void** — it measures
> `burn-msa`, deleted 2026-09-27 (ADR-0014); it is kept because deleting a
> measurement is not the same as deleting the fact that it was taken, and
> because a reader who finds the number in `bench/baselines.json`
> (`msa_sparse.fused_ms`) deserves to learn here that the crate is gone. Two
> more caveats from the library README: several benches read the clock with no
> device flush inside the timed loop, and no GPU in the last two years is a
> 3090.

Measured 2026-08-09 on the same RTX 3090, same shapes as `benches/src/main.rs`.
Both sides use min-of-runs with CUDA sync. PyTorch 2.6.0+cu124.

| op | dormouse-fused (ms) | PyTorch reference (ms) | dormouse-fused vs torch |
|----|----------------:|-----------------------:|--------------------:|
| RoPE (4×2048×32×128) | 0.33 | 1.46 (HF transformers style) | **4.4× faster** |
| FWT (256×512) | 0.015 | 0.031 (Hadamard matmul) | **2.1× faster** |
| Sinkhorn (8×2048×16×16, 20 it) | 2.78 | 10.53 (logsumexp) | **3.8× faster** |
| Situ-GLU (2048×5120, Kimi-K3) | 0.15 | 1.03 | **7.1× faster** |
| Muon+ colrow norm (2048×5120) | 0.78 | 0.68 (F.normalize pair) | 0.87× (torch 1.2×) |
| Sparse attn top-k (1×8×64×16) | 0.013 | 0.28 (gather) | **21× faster** |
| AttnRes depth attend (24×1×2048×4096) | 3.10 | 5.59 (stack+softmax) | **1.8× faster** |
| GDN2 chunked prefill (1×4×128×64) | 0.13 | 15.0 (per-token loop, NVlabs style) | **113× faster** |
| KDA chunked prefill (1×8×128×64) | 0.14 | 19.5 (per-token loop, FlashKDA style) | **139× faster** |
| SCT from_dense (512×256, 15 sweeps) | 619 | 9.13 (torch.linalg.svd / cuSOLVER) | 0.015× (torch 68×) |

## Interpretation

- **8 of 10 ops beat the PyTorch reference**, from 1.8× (AttnRes) to 139×
  (KDA prefill). The big wins are the fused prefill kernels (GDN2/KDA: the
  per-token tensor loop is the reference everyone starts with) and the tiny
  fused ops (MSA, Situ, RoPE, FWT).
- **Muon+ norm loses 1.2×**: torch's `F.normalize` pair is a leaner pipeline
  (mean/std/normalize all use the same fused kernels). Our fused col+row
  norm is within striking distance; the gap is the two-launch + shared
  reduction overhead.
- **SCT from_dense loses 68×**: torch delegates to cuSOLVER (vendor-tuned
  SVD). The hand-rolled Jacobi SVD is launch-dispatch bound on cubecl
  0.11-pre (~0.15 ms per round kernel, 255 rounds × 15 sweeps). It is a
  one-time weight decomposition, not a per-forward op, but it is the honest
  gap: vendor libraries win at dense linear algebra.

## References (popular implementations + research sources)

| op | code (benchmarked reference) | research source |
|----|------------------------------|-----------------|
| RoPE | [HF transformers rotary_embedding](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py), [lucidrains/rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch) | [RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864), [YaRN (Peng et al., 2023)](https://arxiv.org/abs/2309.00071) |
| FWT | [Dao-AILab/fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) (CUDA, wheel build failed on this box) | [The Era of 1-bit LLMs / BitNet b1.58](https://arxiv.org/abs/2402.17764), [BitNet v2](https://arxiv.org/abs/2504.18415) |
| Sinkhorn | [lucidrains/sinkhorn-router-pytorch](https://github.com/lucidrains/sinkhorn-router-pytorch), Megatron DMoE | [Manifold-Constrained Hyper-Connections (DeepSeek, 2025)](https://arxiv.org/abs/2512.24880), [Sinkhorn Distances (Cuturi, 2013)](https://arxiv.org/abs/1306.0895) |
| Situ-GLU | [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3) | [Kimi K3](https://arxiv.org/abs/2607.24653) |
| Muon | [KellerJordan/muon](https://github.com/KellerJordan/muon) | [Muon+: One Additional Normalization Step (UCSB, 2026)](https://arxiv.org/abs/2602.21545), [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) |
| Sparse attn | [mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) | [MiniMax Sparse Attention (Lai et al., 2026)](https://arxiv.org/abs/2606.13392), [BiFormer bi-level routing (CVPR 2023)](https://arxiv.org/abs/2303.08810) |
| AttnRes | naive stack+softmax (arXiv reference impl) | [Attention Residuals (Moonshot/Kimi, 2026)](https://arxiv.org/abs/2603.15031) |
| GDN2 | [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet), [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) | [Gated Delta Networks (ICLR 2025)](https://arxiv.org/abs/2412.06464), [GDN-2: Decoupling Erase and Write](https://arxiv.org/abs/2605.22791) |
| KDA | [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA), [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | [Kimi Linear](https://arxiv.org/abs/2510.26692), [Kimi K3](https://arxiv.org/abs/2607.24653) |
| SCT | [EctoSpace/SCT (official PyTorch reference)](https://github.com/EctoSpace/SCT), [torch.linalg.svd](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html) | [Spectral Compact Training (Kohlberger, 2026)](https://arxiv.org/abs/2604.00733) |

The crate READMEs cite the same sources; this table adds the code links
used as the PyTorch side of each benchmark.

## Reproduce

```
/home/sehaxe/bench_torch.py   # PyTorch side (min-of-runs, same shapes)
BURN_DEVICE=cuda cargo run -p dormouse-fused-benches --release
```

## New fused kernels (ported crates, 2026-08-09)

| op | fused (ms) | native ops (ms) | gain |
|----|-----------:|----------------:|-----:|
| RMSNorm [1,2048,4096] | 0.147 | 0.43+ (rms part alone) | ~3-4× |
| SwiGLU gate [1,2048,8192] | 0.156 | ~0.5 (4 passes) | ~3× |

Both are single-launch elementwise kernels with a shared-memory reduction
(RMSNorm) or fused silu·mul (SwiGLU), hitting ~600+ GB/s on the 3090.

## SVD status (LAPACK host path)

The ops-Jacobi experiment was dropped in favor of the LAPACK-style
Golub-Kahan + dbdsqr (the user's direction): bidiagonalization is now
parallel (6 worker threads, barrier protocol, raw shared access through a
Send+Sync method guard — current rustc considers scoped-spawn closures with
raw-pointer temporaries non-Send). from_dense: 56 -> 49 ms. The residual
~8x gap vs torch.linalg.svd is the vendor cuSOLVER on GPU vs scalar Rust on
CPU (Ryzen 5500 measures ~0.7-7 GFLOP/s single-thread depending on warmup);
the 0-sync ops-SVD alternative was ~1 s (launch-bound), so the host LAPACK
path stays.
