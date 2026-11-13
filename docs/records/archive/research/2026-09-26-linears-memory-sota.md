# Linear Attention & Memory Layers SOTA 2025–2026 — replacements for dormouse

Date: 2026-09-27. Method: 8 research rounds, primary sources only (arXiv abstracts opened, GitHub repos opened: README, license, file trees). No numbers below are invented; every claim carries its source. "Port cost" entries are engineering estimates, marked as such.

Context: dormouse = 7.5M-param byte-level LM, one looped block (max_iter=4), arms = KDA gated-delta-rule (own Rust port of FlashKDA math: chunked WY, per-channel decay) + hashed n-gram Engram (retired: monopolized loss; host-RAM tables + CPU Adam sidecar). Goal: simpler/faster/equal-or-smarter replacements with ready code.

---

## 1. Linear attention SOTA (RQ1)

### 1.1 The FLA library is the reference kernel source

Source: https://github.com/fla-org/flash-linear-attention (opened 2026-09-27). **MIT license**, 5.8k stars, verified on NVIDIA/AMD/Intel, H100 CI badge, `fla-hub` pretrained checkpoints, `flame` (torchtitan) training framework.

Full model zoo shipped with ready Triton kernels (from the repo's Models table; "ops" = fused kernels, "layers/models" = trainable layer):

| Year | Model | Kernel/impl path | Notes for dormouse |
|---|---|---|---|
| 2022–23 | ABC, RetNet, HGRN | `fla/layers/*` | older, superseded |
| 2024 | GLA | `fla/layers/gla.py` | simple gating, weak recall |
| 2024 | Based | `fla/layers/based.py` | Taylor-approx softmax attn |
| 2024 | DeltaNet | `fla/layers/delta_net.py` | the delta-rule origin (arXiv:2406.06484) |
| 2024 | HGRN2, RWKV6, LightNet, YOCO, Mamba2, MLA, Rebased | layers/models | state-expansion / SSM family |
| 2024 | **GSA** | `fla/models/gsa` | two-layer GLA linked via softmax; strong recall; small state (arXiv:2409.07146, NeurIPS 2024) |
| 2025 | Samba, Rodimus, DeltaProduct, PaTH, Comba, MesaNet, Log-Linear, DeltaFormer | layers/ops | niche variants |
| 2025 | **Gated DeltaNet (GDN)** | `fla/ops/gated_delta_rule` | scalar decay + scalar β; adopted by Qwen3-Next (FLA README News 2025-09, links the Qwen blog) |
| 2025 | RWKV7 | `fla/ops/rwkv7` | dynamic state evolution (arXiv:2503.14456) |
| 2025 | NSA (native sparse attn), MoBA (block sparse attn) | `fla/ops/nsa`, `fla/layers/moba.py` | sparse softmax attention, not linear |
| 2025 | **KDA (Kimi Delta Attention)** | `fla/ops/kda` — 10 files: `chunk.py, chunk_fwd.py, chunk_bwd.py, chunk_intra.py, chunk_intra_token_parallel.py, fused_recurrent.py, gate.py, naive.py, wy_fast.py, backends/` | **what dormouse already ported**; `naive.py` is a PyTorch reference for bit-for-bit checks |
| 2025 | **MoM** | `fla/layers/mom.py` | Mixture-of-Memories — see §3 (arXiv:2502.13685) |
| 2025 | FoX (forgetting softmax attn) | `fla/ops/forgetting_attn` | softmax + forget gate |
| 2026 | **Mamba3** | `fla/models/mamba3` (arXiv:2603.15569) | SSM |
| 2026 | **Raven** | `fla/models/raven` (goombalab/raven, MIT) | sparse-routed slot memory on GSA kernels — see §3 |
| 2026 | **GDN-2** | `fla/ops/gdn2` (arXiv:2605.22791) | current delta-rule frontier — see §1.2 |
| 2026 | **PGDN / PKDA** | `fla/ops/precond_gated_delta_rule`, `fla/ops/precond_kda` (arXiv:2604.21100) | curvature-aware ATK preconditioner on the recurrence; added 2026-06, no large-scale validation yet |
| 2026 | Wall, Parallax, CAT, AttnRes, YOCO | ops/models | attention-structure variants |

Kernel speed evidence (FLA README benchmark table, GB200, opened): `chunk_gdn` fwd 1.265 ms vs `flash_attn` 3.753 ms at B1/T8192/H96/D128; fwdbwd 4.738 ms vs 15.371 ms — the delta-rule kernels are several× faster than softmax attention at long T, and GDN ≈ RetNet speed with much better quality.

### 1.2 Quality frontier: GDN → KDA → GDN-2

**Kimi Linear (Kimi Team, arXiv:2510.26692, opened)** — KDA = Gated DeltaNet extended with *fine-grained (per-channel) gating* of the decay; a specialized Diagonal-Plus-Low-Rank transition keeps the chunkwise algorithm hardware-efficient. Hybrid 3:1 KDA:MLA (3B active / 48B total, 5.7T tokens) **outperforms full MLA under identical recipe**, −75% KV cache, up to 6× decode throughput at 1M ctx. Kernels open-sourced in FLA; checkpoints + vLLM support public. Repo: https://github.com/MoonshotAI/Kimi-Linear (opened: **MIT license**, 1.6k stars).

**Gated DeltaNet-2 (Hatamizadeh, Choi, Kautz / NVIDIA, arXiv:2605.22791, opened)** — the current best-performing gated delta-rule variant. Update rule (from the repo README, opened):

```
S_t = (I − k_t (b_t ⊙ k_t)ᵀ) D_t S_{t−1}  +  k_t (w_t ⊙ v_t)ᵀ
```

with **channel-wise erase gate b_t ∈ [0,1]^d_k** and **channel-wise write gate w_t ∈ [0,1]^d_v** on top of KDA's channel-wise decay D_t. Strict generalization: collapses to KDA when both gates collapse to one scalar, to GDN when decay also collapses. Head-to-head at 1.3B / 100B FineWeb-Edu tokens, matched recurrent state (README tables, opened):

| Model | Wiki ppl ↓ | Avg acc ↑ (recurrent) | Avg acc ↑ (hybrid+SWA) | MK-NIAH-1 @4K (recurrent) |
|---|---|---|---|---|
| Mamba-2 | 16.79 | 51.82 | 50.86 | — |
| GDN | 16.40 | 52.07 | 52.25 | 27.8 |
| KDA | 16.81 | 52.28 | 52.68 | 28.0 |
| Mamba-3 (MIMO) | 16.45 | 52.39 | 52.72 | 18.0 |
| **GDN-2** | **15.90 / 15.62** | **53.11** | **53.97** | **37.8 / 48.0** |

GDN-2's ablation: **the erase gate b_t accounts for most of the gain**; advantage largest on interference-heavy multi-key retrieval (RULER). Code: https://github.com/NVlabs/GatedDeltaNet-2 (opened: **NVIDIA Source Code License-NC**, non-commercial; lit-gpt trainer, full 1.3B recipe: AdamW 4e-4, wd 0.1, clip 1.0, cosine, 0.5M-token batch, 4K seq, 16 heads d=128). The *kernels* are also in FLA (`fla/ops/gdn2`) under FLA's **MIT** license, and GDN-2's chunkwise WY algorithm "with channel-wise decay absorbed into asymmetric erase factors" plus gate-aware backward is described in the paper abstract.

**Also 2026, in FLA**: PGDN/PKDA (arXiv:2604.21100) — preconditioned delta rule; Context Parallel for KDA/GDN (2026-03); FlashQLA backend for GDN (2026-07). Mamba3 (arXiv:2603.15569). None of these has Kimi-Linear-scale validation yet.

**Qwen3-Next's choice**: Gated DeltaNet — evidenced by FLA README News 2025-09: "GDN has been integrated into Qwen3-Next" with link to the official Qwen blog (opened via FLA README).

### 1.3 Simplest math in the zoo

- **GLA** (arXiv:2312.06635): `S_t = α_t S_{t−1} + k_t v_tᵀ` with a single logsigmoid gate — simplest usable, weakest recall.
- **GSA** (arXiv:2409.07146 abstract, opened): "two-layer GLA linked via softmax, context-aware memory reading and adaptive forgetting" — compact state, strong in-context recall, good T2R; math is elementary (no WY/delta erase step).
- **KDA/GDN/GDN-2**: delta-rule family — requires chunked WY for parallel training, i.e., the most complex math in the zoo, but dormouse already owns that port.
- Raven (§3) is arguably the simplest *memory-bearing* design: a slot matrix with top-k sigmoid routing and exponential decay, implemented by reusing FLA's existing GSA kernels (repo README, opened).

### 1.4 Rust port cost (estimates, from the FLA file trees opened)

- **GDN-2 from dormouse's existing KDA port**: the recurrence differs by two elementwise gate vectors (b_t, w_t) entering the erase/write factors of the WY recursion — same chunk structure, same backward skeleton. Estimate: **300–800 lines delta, days, not weeks**; verify bit-for-bit against `fla/ops/gdn2/naive.py` (same reference pattern as kda's). Risk: gate-aware backward subtleties — mitigated by the naive.py reference.
- **GSA**: new recurrence (softmax-normalized slot read over a small state), no delta erase; estimate **~1–1.5k lines**, comparable to the KDA port effort; reference = `fla/models/gsa` + GSA kernels.
- **PGDN/PKDA**: adds a preconditioner solve per chunk — more machinery, freshest paper, skip.
- Everything else (RWKV7, Mamba3, Log-Linear, …) = a fresh port with no quality win over the delta-rule family at this scale, per GDN-2's matched table.

---

## 2. Memory layers SOTA (RQ2)

### 2.1 Memory Layers at Scale (Meta, arXiv:2412.09764 — verified via arXiv API, opened)

Trainable key-value lookup (product-key memory, PK+) that adds parameters without FLOPs; memory-augmented models beat dense models with >2× compute budget and beat MoE at matched params+compute; gains strongest on factual tasks; demonstrated to 128B memory params / 1T tokens; "fully parallelizable memory layer implementation" released. Code: https://github.com/facebookresearch/memory (opened) — **CC-BY-NC** (non-commercial; fine for dormouse research, blocks commercial use), PyTorch on Meta-Lingua, core in `lingua/product_key/{memory.py, colwise_embeddingbag.py, xformer_embeddingbag.py}`. Note: the repo's optimizer runs the memory tables through standard backprop + Adam *on the accelerator* — there is no host-RAM optimizer sidecar in the reference design at dormouse's scale; EmbeddingBag-style sparse lookup is the whole mechanism.

### 2.2 Fast-weight Product Key Memory (Sakana AI, arXiv:2601.00671, opened)

FwPKM = product-key memory whose activated slots receive **chunk-level gradient descent on a local memory-rewrite objective inside the layer** (TTT-style), at both train and test time. Acts as an *episodic* memory complementing the *semantic* memory of the dense modules; significant perplexity reductions on long-context; NIAH generalizes 4K-train → 128K-test. Code: https://github.com/SakanaAI/fast-weight-product-key-memory (located via GitHub search, opened: Python, updated 2026-03) + independent lucidrains port. Key simplification for dormouse: **the in-forward slot rewrite replaces the host-side Adam sidecar entirely** — no m/v state, no D2H grad sync, the "optimizer" is a few lines inside the layer.

### 2.3 δ-mem (arXiv:2605.12357, opened)

Compact online delta-rule state (as small as 8×8) attached to a **frozen full-attention backbone**; its readout produces low-rank corrections to attention during generation. Gains (1.10×–1.31× relative) are in assistant/agent memory benchmarks. Not a pretraining architecture and not a value-path memory — **not applicable** to dormouse's from-scratch 7.5M model; cited here to close the question.

### 2.4 Memory Attention (arXiv:2609.28399, opened — the MA paper)

Values are formed by combining a **layer-specific token-indexed memory table** with contextual keys: `v = memory[token] ⊕ f(keys)` — the memory supplies token-specific content, keys restore context dependence. At inference, normalization folds into the tables so value construction = **lookup + add**; token-indexed retrieval enables **CPU offload with prefetching**. Improved LM + downstream averages at matched token budgets with extra memory params. Single author; **no code link in the abstract** (checked the arXiv page). This is the direct published analogue of dormouse's "MA-synthesis" idea, but for softmax attention, and the memory is *token-indexed* (a per-token table), not *n-gram-content-hashed* like the Engram.

### 2.5 Titans (Google, arXiv:2501.00663 — verified via arXiv API, opened)

Neural long-term memory trained by surprise-based gradient updates at test time, combined with attention as short-term memory; >2M-context NIAH claims. **No official code from the authors** (paper-only; the many GitHub repos are unofficial re-implementations). For a 7.5M model the memory-as-MLP-with-Momentum machinery is heavier than every alternative below — **not recommended**.

### 2.6 Ranking by implementation simplicity for dormouse (7.5M params, 16 GB GPU + 64 GB RAM)

| # | Design | Why simple/fitting | Ready code | Caveat |
|---|---|---|---|---|
| 1 | **MA-style token-indexed value memory** (2609.28399) | It is an embedding table + add — the simplest possible memory; no router, no hash, no separate optimizer; folds into the existing LinearLike value path; CPU-offload story already matches dormouse's `offload.rs` | none (paper only) — but the mechanism is a plain `Embedding`; a burn port is trivial (est. <300 lines) | token-indexed ≠ content-addressed; value depends on re-reading §2.4 |
| 2 | **Product-key memory, Meta PK+** (2412.09764) | Learned soft retrieval (top-k over two subkey products) — no FNV hash collisions problem; trains in-graph, no sidecar; strongest scaling evidence of any memory layer | facebookresearch/memory (**CC-BY-NC**), core ≈ 3 files | non-commercial license on reference impl; PyTorch→burn port est. 500–1000 lines |
| 3 | **FwPKM** (2601.00671) | Same retrieval as #2 plus in-layer slot rewrite → *deletes* the host Adam sidecar dormouse currently maintains | SakanaAI/fast-weight-product-key-memory (license unverified) + lucidrains port | newest of the three; chunk-rewrite numerics need care |
| 4 | **MoM** (2502.13685) | Memory *states* (not param tables) inside the linear recurrence; router = standard MoE top-k; reuses KDA/GDN kernels | OpenSparseLLMs/MoM + Linear-MoE; `fla/layers/mom.py` | N memory states = N× recurrence state VRAM |
| 5 | **Titans** (2501.00663) | — | no official code | heaviest machinery |

---

## 3. Linear attention + memory in the value path (RQ3)

No opened paper combines *linear attention* with a *hashed/token memory in the value path* in exactly dormouse's MA-synthesis form. Closest, in order:

1. **MoM — "Linear Sequence Modeling with Mixture-of-Memories"** (arXiv:2502.13685 v4, opened): linear attention with **multiple memory states + a router** directing tokens to top-N states; "surpasses existing linear sequence modeling on recall-intensive tasks, comparable to Transformers". This is memory **inside** the linear-attention recurrence (state memory, not parameter memory). Code: OpenSparseLLMs/MoM + Linear-MoE; shipped in FLA (`fla/layers/mom.py`). A direct value-path analog: route to memories *before* the delta-rule write.
2. **Raven — "High-Recall Sequence Modeling with Sparse Memory Routing"** (goombalab/raven, opened; also FLA `fla/models/raven`; **MIT**): a slot matrix `H ∈ R^(slots×d_v)` per head; per-token **top-k learned router** selects slots; `H = H·decay + (1−decay)·k⊗v`; read `o = q·H`; decay from Mamba2/GLA; **reuses FLA's GSA chunked + fused-recurrent kernels** (no new kernels needed). ~400M-param models: best/second-best on SWDE/FDA/SQuAD and NIAH-1/2/3 vs linear peers. This is the *learned-router cousin of the Engram*: sparse slot writes with decay, but soft/learned addressing instead of FNV content hashing, and interference-free untouched slots — the published evidence that sparse-slot memory fixes the recall weakness of single-state linear models.
3. **Memory Attention** (2609.28399, opened): token-indexed memory **is** the value projection — but on softmax attention, not linear. The MA recipe transfers to a linear-attention value path mechanically (lookup + add before the delta write), but no opened paper has done it: this is dormouse's open combination, now with two ready-code halves (KDA from FLA + memory patterns from MA/PKM/Raven).
4. **FwPKM** (2601.00671, opened): episodic memory layer explicitly framed as *complementing* the semantic memory of standard modules — the composition argument, without attention-internals surgery.
5. **DeltaFormer** (arXiv:2505.19488, in FLA): frames the transformer block itself as associative memory updated by the delta rule — conceptual support for memory/attention unification, no value-path memory.

---

## 4. FINAL — Replacements for dormouse (ranked)

| # | Component | Candidate | Why better / simpler / faster | Ready code (license) | Rust-port cost (est.) | Risk |
|---|---|---|---|---|---|---|
| 1 | **Attention arm: keep KDA, add GDN-2's decoupled gates** | GDN-2 (arXiv:2605.22791) | Best recurrent mixer at matched 1.3B/100B (53.11 vs KDA 52.28 avg acc; MK-NIAH 37.8 vs 28.0); strict superset of dormouse's KDA — two extra channel-wise gate vectors, same WY chunk skeleton | FLA `fla/ops/gdn2` (**MIT**, has `naive.py`-style reference pattern); NVlabs/GatedDeltaNet-2 (**NC**) for the training recipe | 300–800 lines on top of the existing KDA port; days | Gate-aware backward is the newest math (May 2026); verify against FLA reference before trusting; NC license on NVIDIA repo → port from FLA's MIT kernels only |
| 2 | **Memory arm: replace hashed n-gram Engram with product-key retrieval** | Meta PK+ (2412.09764) and/or FwPKM (2601.00671) | Learned soft top-k addressing replaces FNV hash + host-Adam sidecar; FwPKM's in-slot rewrite *deletes* the sidecar entirely (no m/v, no D2H sync); PK+ has the strongest scaling evidence (128B mem params) | facebookresearch/memory (**CC-BY-NC**); SakanaAI/fast-weight-product-key-memory (+ lucidrains port); license check needed on Sakana repo | PK core ≈ 500–1000 lines in burn (product-score top-k + gather-scatter + optional slot SGD) | CC-BY-NC reference; retrieval quality at 256-vocab byte granularity is unproven anywhere — needs dormouse's own A/B |
| 3 | **Memory arm alternative: Raven-style routed slots inside the loop block** | Raven (goombalab/raven, 2026) | Engram's "sparse slot writes" without the hash: learned top-k router + decay; published recall wins at 400M; slots untouched when unrouted → no loss monopoly mechanism via interference, and it *is* part of the recurrent state (no separate optimizer ever) | goombalab/raven + `fla/models/raven` (**MIT**); reuses FLA GSA kernels | Port = GSA kernels (~1–1.5k lines) + a tiny router; or pair with the existing KDA port and keep only the routing rule (~300 lines) | Youngest paper (2026, MDPI); 400M-scale evidence only |
| 4 | **Value-path MA-synthesis** | Memory Attention recipe (2609.28399) applied to dormouse's linear block | The exact "memory in the value path" idea, now published for softmax attention with CPU-prefetch offload matching `offload.rs` | none — but mechanism is an embedding lookup (est. <300 lines) | trivial code, novel composition | No ready code, no linear-attention precedent — research risk, do after #2/#3 |
| 5 | Not recommended | GSA/GLA/RWKV7 as KDA *replacement*; Mamba3; PGDN/PKDA; Titans; δ-mem | GSA/GLA trade recall for simplicity dormouse doesn't need (KDA port already exists); Mamba3/PGDN unproven at scale; Titans has no official code and heaviest machinery; δ-mem is a frozen-backbone bolt-on | — | — | — |

### Explicit answers

**Should our KDA be replaced by anything FLA ships?** — Replace it with **nothing currently shipped**, *extend* it to **GDN-2** (`fla/ops/gdn2`, MIT): it is the strict generalization of KDA, wins the only opened matched-comparison table (1.3B/100B: avg acc 53.11/53.97 vs KDA's 52.28/52.68, recurrent/hybrid; the erase gate carries most of the gain), and the port is a small delta on the WY machinery dormouse already owns. GDN alone is a simplification but a quality regression; GSA/GLA/RWKV7/Mamba3 don't beat the delta-rule family at matched state in any opened source. PGDN/PKDA are the watchlist.

**Should our Engram arm be replaced by product-key memory or MA-style lookup?** — Yes to both directions, in this order: (a) **product-key memory** (Meta PK+ / FwPKM) replaces the FNV-hash + host-Adam design: learned addressing, in-graph training, and with FwPKM's in-layer slot rewrite the host optimizer sidecar disappears — directly answering the Engram's complexity and its loss-monopoly failure mode (soft gated read instead of an uncontrolled shortcut; MoM/Raven show memory that stays coupled to the recurrent read). (b) **MA-style token-indexed lookup** is the cheapest possible value-path memory (embedding + add, CPU-prefetchable) but has no ready code and no linear-attention precedent — implement it as the low-risk control arm against (a). The literal MA-synthesis (MA lookup inside the KDA value path) remains unpublished; Raven (MIT, on FLA GSA kernels) is the closest shipped artifact and the third option if a router-based design is preferred over parameter tables.

---

## Source ledger (all opened 2026-09-27)

1. https://github.com/fla-org/flash-linear-attention — README (model zoo, news incl. Qwen3-Next GDN integration, GB200 benchmark tables), MIT license, `fla/ops/kda` file tree.
2. https://arxiv.org/abs/2510.26692 — Kimi Linear / KDA abstract.
3. https://github.com/MoonshotAI/Kimi-Linear — README (3:1 KDA:MLA, 5.7T tokens, checkpoints), MIT license.
4. https://arxiv.org/abs/2605.22791 — GDN-2 abstract.
5. https://github.com/NVlabs/GatedDeltaNet-2 — README (update rule, 1.3B/100B result tables, ablation, recipe), NVIDIA-NC license.
6. https://arxiv.org/abs/2502.13685 — MoM abstract (code: OpenSparseLLMs/MoM, Linear-MoE).
7. https://arxiv.org/abs/2412.09764 — Memory Layers at Scale (via arXiv API).
8. https://github.com/facebookresearch/memory — README (product_key file tree, Lingua base), CC-BY-NC license.
9. https://arxiv.org/abs/2601.00671 — FwPKM abstract.
10. GitHub search results — SakanaAI/fast-weight-product-key-memory, lucidrains/fast-weight-product-key-memory.
11. https://arxiv.org/abs/2605.12357 — δ-mem abstract.
12. https://arxiv.org/abs/2609.28399 — Memory Attention abstract (no code link on page).
13. https://arxiv.org/abs/2501.00663 — Titans (via arXiv API; no code claim).
14. https://arxiv.org/abs/2409.07146 — GSA abstract (via arXiv API).
15. https://github.com/goombalab/raven — README (mechanism, GSA-kernel reuse, 340M configs), MIT license.
