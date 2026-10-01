# Research & reference verification

> **Status 2026-10-01.** Last updated 2026-08-09. The crate list changed
> 2026-09-28: **20 crates, not 28** — and **six rows below verify crates that no
> longer exist** (`burn-msa` deleted 2026-09-27 per ADR-0014; `burn-mtp`,
> `burn-fastblt`, `burn-antihall`, `burn-nope`, `burn-mod`, `burn-ttt` deleted
> 2026-09-28 as unreachable, fate table `docs/library-crate-fate.md` in the
> dormouse repo). A row here is a claim about the code as it stood on
> 2026-08-09; the verification was real when made and it says nothing about the
> 14 surviving crates since. For the comparisons that *are* maintained, the
> per-comparison oracle tier — authors' code vs transcription vs nothing — is
> `docs/ORACLE-TIERS.tsv` in the dormouse repository, with the prose in
> `docs/ORACLE.md`. That is the document to read before calling anything here
> verified; this table has no tier column and cannot tell a (b) from a (d).

Every crate is verified against (a) the research paper it implements and
(b) the original / most popular reference code, for correctness, memory and
performance. Status per crate, last updated 2026-08-09.

## Verified

| crate | paper | reference code checked | result |
|-------|-------|------------------------|--------|
| burn-rope | [RoFormer 2104.09864](https://arxiv.org/abs/2104.09864), [YaRN 2309.00071](https://arxiv.org/abs/2309.00071) | HF transformers `rotary_embedding` (einsum+cat) | formula identical; fused kernel FD-checked 12/12; 4.4x vs the HF reference on 3090 |
| burn-situ | [Kimi K3 2607.24653](https://arxiv.org/abs/2607.24653) | MoonshotAI/Kimi-K3 formula | `beta*tanh(g/beta)*sigmoid(g) * bu*tanh(u/bu)` matches the report; fused = tensor (FD-checked); 7.1x vs torch |
| burn-muon-plus | [Muon+ 2602.21545](https://arxiv.org/abs/2602.21545) | [KellerJordan/muon](https://github.com/KellerJordan/muon) muon.py | NS coeffs (3.4445, -4.775, 2.0315) exact; factored quintic; Frobenius norm; tall->transpose; 5 steps; post-polar ColRow norm. **Fixed**: lr scale was sqrt(m/n), reference uses max(1, m/n)^0.5 (shrank lr on wide matrices) |
| burn-msa | [MiniMax Sparse Attention 2606.13392](https://arxiv.org/abs/2606.13392) | MiniMax `sparse_fmha_plan` | top-k KV blocks per query, kv_block_num in {4,8,16,32} matches our topk range; block score = **max** over the block (MiniMax: "per-(Hq, kv_block, q) max scores") — our `compute_block_scores` uses block-max |
| burn-sct | [Spectral Compact Training 2604.00733](https://arxiv.org/abs/2604.00733) | [EctoSpace/SCT](https://github.com/EctoSpace/SCT) `spectral_layer.py` | from_dense = truncated SVD with U=Vh[:k]^T, V=U_full[:,:k], s=S[:k] (same orientation); retract = QR (Stiefel); forward y=(x@U)*s@V^T exact paper order; f32-tolerance tests vs the reference pass |
| burn-gdn2 | [GDN ICLR'25 2412.06464](https://arxiv.org/abs/2412.06464), [GDN-2 2605.22791](https://arxiv.org/abs/2605.22791) | NVlabs/GatedDeltaNet-2 | recurrence S <- S*exp(g) + k (x) (w.v - (b.k)^T S), o = q^T S matches the equations; chunked fused kernel FD-checked |
| burn-kda | [Kimi Linear 2510.26692](https://arxiv.org/abs/2510.26692), [Kimi K3 2607.24653](https://arxiv.org/abs/2607.24653) | MoonshotAI/FlashKDA | delta-rule step S <- S*decay + k (x) beta(v - S^T k), o = q^T S matches; chunk <= 16 constraint matches FlashKDA's f32 limit |
| burn-bitnet | [BitNet b1.58 2402.17764](https://arxiv.org/abs/2402.17764), [BitNet v2 2504.18415](https://arxiv.org/abs/2504.18415) | ternary {-1,0,1} weight quantization; FWT fused kernel | formulas match the papers; FWT self-adjoint (same kernel for backward) |
| burn-mhc | [mHC 2512.24880](https://arxiv.org/abs/2512.24880) | log-domain Sinkhorn references (lucidrains sinkhorn-router) | forward = alternating row/col normalization; backward **fixed earlier**: the weighted sum used m_pre*sum(d) instead of sum(d*m_pre); FD-checked 9/9 |
| burn-attnres | [Attention Residuals 2603.15031](https://arxiv.org/abs/2603.15031) | arXiv reference pseudocode | depth-attend: normalized history h/||h||, scores q.h^T/sqrt(d), softmax over layers, weighted sum; manual math == fused kernel == tensor path |

## Method

- Math: equations from the paper compared line by line against the crate code.
- Code: the original/most popular implementation cloned and inspected
  (EctoSpace/SCT, MiniMax-AI/MSA, KellerJordan/muon, NVlabs/GatedDeltaNet-2,
  MoonshotAI repos).
- Correctness: per-crate tests (fused == tensor, FD gradient checks,
  f32-tolerance vs the reference).
- Performance: head-to-head on the same RTX 3090, min-of-runs, same shapes —
  see `PYTORCH_COMPARISON.md`.
- Memory: fused kernels allocate only the output (no intermediates); the
  tensor-path references allocate per-op temporaries. The chunked designs
  (msa CHUNK=8, attnres G=8, gdn2/kda chunk <= 16) cap the peak tensors.

## Ported 0.21 -> 0.22 (2026-08-09)

All 17 previously-0.21 crates are now on burn 0.22 in the workspace
(ambient tensors, `Param::initialized`, ndarray-backed tests): burn-antihall,
burn-dspark, burn-eggroll, burn-engram, burn-es, burn-fastblt, burn-jepa,
burn-mod, burn-mor, burn-mtp, burn-nope, burn-parcae, burn-ptrn,
burn-rmsnorm, burn-swiglu, burn-ttt (+ burn-sct already migrated).

Fix found during the port: burn-mor's `topk_indices` used cubecl `argtopk`,
which returns garbage indices on 0.11-pre (the same defect found in the msa
investigation); replaced with k masked-argmax passes (correct on every
backend). Research sources of the ported crates: DSpark 2607.05147, EGGROLL
2511.16652, FastBLT 2605.08044 / BLT 2412.09871, MTP 2404.19737, MoR
2507.10524, H-Neurons 2512.01797, RMSNorm 1910.07467, SwiGLU 2002.05202,
TTT-E2E 2512.23675; their tests (fused/tensor equivalences, reference-value
checks like the BLT patcher's exact patch layout) pass on 0.22.

## Memory/performance audit of the ported crates (2026-08-09)

All 17 ported crates are ops-only (no custom kernels): their speed tracks the
burn backend, and the audit found no hot-path allocation issues.

| crate | audit result |
|-------|--------------|
| burn-es | OpenAI ES utility function (1703.03864): u_i = max(0, ln(N/2+1) − ln(N+1−rank)), normalized u/sum − 1/N — exact; BitNet-style ternary (sign·absmean, 0.7 threshold) |
| burn-eggroll | rank-1 ES perturbations never materialize E (batched (f·A)ᵀ·B) — matches the paper's ~100x cheaper storage claim |
| burn-jepa | LeJEPA = mean² + ||cov−I||² (standard); KoLeo = −mean(log nn_dist) with a documented soft-min surrogate (τ→0 recovers DINOv2 exactly), strided subsample to ≤256 (O(n²) cap) |
| burn-parcae | retention Ā = exp(−(|δ|+1e-8)·exp(a)) — contractive in [0,1) by construction, B̄ = Δ·B diag scaling; full-B variant matches diag (test) |
| burn-ptrn | Q-head BCE loss, recurrent noise, best-of-k gather — all light ops |
| burn-engram | multi-hash embedding + depthwise conv + per-head gated fusion; per-head `embeds.clone()` is small relative to the weights — left as is |
| burn-mtp | shared unembedding f_u, linear-probing heads (paper appendix), per-depth λ (V3) — verified by ce_equals_manual_gather |
| burn-mor | expert-choice routing via masked-argmax topk (see the topk fix), scatter Add-only semantics |
| burn-dspark | Markov head (embedding→bias) + accept-rate predictor — standard modules |
| burn-fastblt | BLT patcher semantics verified byte-exact (patch layout test: [16,14,16,16,2]); bltd_loss and self_spec are light |
| burn-antihall | per-neuron sigmoid gates, three adaptation levels — light |
| burn-nope | content-based attention with pure-GPU causal mask (triu_mask) |
| burn-rmsnorm/swiglu/ttt | single-pass elementwise / standard modules |
