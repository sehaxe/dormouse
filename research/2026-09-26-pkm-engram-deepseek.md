# PKM vs hashed-Engram, and what DeepSeek actually ships

**Date:** 2026-09-26 · **Method:** primary sources only (arXiv HTML/full text, official repos read in full), 10 rounds. Every load-bearing claim is cited; anything not confirmed is marked **NOT VERIFIED**.

**Correction up front:** the PKM paper is arXiv **1907.05242** ("Large Memory Layers with Product Keys", Lample et al., NeurIPS 2019) — the task brief's "1907.05642" does not resolve to it. The other two IDs are correct.

---

## 1. Product-Key Memory mechanics

### 1.1 The sub-key trick (why millions of slots without hashing)

Source: [arXiv 1907.05242](https://arxiv.org/abs/1907.05242) (read via arXiv HTML v2 + official `facebookresearch/XLM` PKM notebook).

- The memory has three parts: a **query network**, two sets of **sub-keys** C, C′ (each √N sub-keys of dimension d_q/2), and a **value table** of N = |C|·|C'| slots. The full key set (the Cartesian product C × C′) is **never materialized**.
- Selection: split the query into halves q₁, q₂; take top-k sub-keys of each set; the true top-k product keys are **guaranteed** to lie in the k×k combinations; refine by scoring those k² candidates. Complexity O((√N + k²)·d_q) — ~1000× cheaper than exhaustive at N = 1024².
- Consequence: **slot addressing is content-based and learned**, but key storage is only √N·d_q params — keys are ~free vs the values, which hold "the bulk of the parameters" and scale quadratically in sub-keys.
- Contrast with hashing: a hash fixes the slot before the model exists and never learns. Product keys let the *backbone's own gradients* decide which slots answer which contexts. That is the single structural difference that matters for us (§2).

### 1.2 Soft top-k reads

Source: same paper + XLM `PKM-layer.ipynb` (official reference implementation, read in full).

- Read = weighted **sum over k slots**: scores of the top-k product keys → `softmax(scores)` (temperature 1, no learned temperature) → `EmbeddingBag(per_sample_weights=scores)`. Reference hyperparameters: **k_dim=256, heads=4, knn=32** (top-32 per head, 128 slots touched per token), n_keys=512 → **512² = 262,144 slots**.
- **Keys are learned** (`nn.Parameter`, uniform init from a seeded RNG; Meta's repo likewise stresses keys are "continually trained" — this is exactly why they couldn't use an external ANN index and adopted product keys). Query normalization is a first-class stabilizer: the paper's config applies **BatchNorm to queries**, "improves the performance and memory usage" (XLM notebook warning text); the popular PyTorch port (lucidrains/product-key-memory) uses LayerNorm and notes "batchnorm would break causality".
- **Multi-head**: each head has its own query net and sub-keys, values shared; heads' selections barely overlap → "increase key usage and generally improves performance".
- **Sparse updates**: "All the memory parameters are trainable, yet only k memory slots are updated for each input." The notebook explicitly recommends a **separate optimizer for values: Adam, constant lr 1e-3**, independent of the rest of the network.
- Integration: PKM **replaces one FFN** in typically 1–2 layers, added through the residual: `x ← x + PKM(x)`.

### 1.3 Scale proof: Meta "Memory Layers at Scale"

Source: [arXiv 2412.09764](https://arxiv.org/abs/2412.09764) (v1/v2 HTML + `facebookresearch/memory` repo).

- Same trainable product-quantized keys; **2²⁰ ≈ 1M values** default, scaled to **64M keys = 128B memory params** on a 1.3B base model, 1T tokens. A 1.3B Memory+ model approaches Llama2-7B (2× tokens, 10× FLOPs) on factual QA; beats parameter-matched MoE and PEER.
- Engineering facts that transfer: values sharded by embedding dim; custom EmbeddingBag kernels (3 TB/s fwd); backward via atomic-free `reverse_indices`; **"memory layers are naturally memory-intensive, mostly due to the large number of trainable parameters and associated optimizer states"** — optimizer state on the table is a first-class cost (our host-RAM experience mirrors this exactly).
- Ablations: more memory keeps helping factual QA; value dim = model dim beats trading dim for more slots; bigger key dim helps but adds dense params.

### 1.4 FwPKM: the in-slot gradient rewrite

Source: [arXiv 2601.00671](https://arxiv.org/abs/2601.00671) (Zhao & Jones, Sakana AI; ICML 2026 workshop; repo `SakanaAI/fast-weight-product-key-memory`).

- Standard PKM is slow-weight (semantic memory). FwPKM keeps retrieval identical but **rewrites activated slots at train *and* inference time** by chunk-level gradient descent (TTT-style): V is updated on a **gated reconstruction loss L_mem**, and K¹,K² on an **auxiliary addressing loss L_addr whose explicit job is to prevent slot collapse**.
- Config: 512² slots; PKM reads top-128 (4 heads × 32), FwPKM reads **top-8** (1 head × 8). A learned **scalar gate** (slow weights) controls the memory's contribution; "effective iterative memorization correlates with high gating values".
- Results: trained on 4K sequences, generalizes NIAH to 128K; rereading (2-iter) lifts retrieval <10% → >70%. Role separation: PKM = semantic, FwPKM = episodic.

### 1.5 Documented PKM failure modes — is there a "memory monopolizes the loss" mode?

**No PKM paper documents our exact pathology** (memory row directly explains the target so the backbone's gradient vanishes). **NOT VERIFIED as a known PKM failure mode** — I found no such claim in any of the four papers above. What *is* documented:

1. **Softmax over-centralization at large slot counts** — Shen et al., [arXiv 2302.06461](https://arxiv.org/abs/2302.06461): softmax "provides over-centralized distribution… only a few elements are highlighted"; with many slots most values are ignored; **LayerNorm on the memory output** makes softmax ≈ ReLU, and non-competitive gates (ReLU) beat softmax for large memories.
2. **Key-usage imbalance** — Lample et al. measure "memory usage" explicitly; product keys, query BN, and multi-head all exist to raise it.
3. **Slot collapse in fast weights** — FwPKM's L_addr exists precisely because online rewrites collapse addressing.
4. **The honest small-scale result** — Csordás, Irie, Schmidhuber, [arXiv 2310.10837](https://arxiv.org/abs/2310.10837) (EMNLP 2023 Findings, `robertcsordas/moe`): with every stabilizer applied (non-competitive ReLU gates, proper init), **parameter-matched PKM still loses to the dense baseline**: WikiText-103 perplexity 13.96 (PKM softmax) → 12.77 (PKM ReLU) vs **11.81 dense**. Their words: "even the best PKM models underperform the dense baselines, indicating the fundamental limitation of PKMs."

**How papers prevent memory domination, concretely:** query normalization (BN/LN), softmax-or-ReLU over only the k read scores (the read is a *mixture*, bounded by construction), a *separate constant-lr optimizer for values* (Lample's 1e-3 Adam), memory in 1–2 layers replacing one FFN with residual add, and — in Engram's case (§3) — capacity allocation laws that keep the backbone as the dominant compute.

---

## 2. Our Engram's failure, mapped mechanism-by-mechanism

What dormouse had (read from `vendor/burn-fused/crates/burn-engram/src/lib.rs`, `crates/dormouse-train/src/offload.rs`): FNV-hashed byte 3/5/8-grams → direct row in 24M-row host-RAM tables → per-head sigmoid gate (RMSNorm key·query / √d, compressed: σ(√(|s|+ε)·sign s)) → value_proj → added to the residual; rows trained by an external CPU Nesterov+Sinkhorn optimizer from D2H'd gradients (~15K row-touches/step).

Observed failure (`.bulba/memory.md` 2026-09-26): train CE → ~0.17, held-out stuck ~5.5. The memory explained the targets itself; the core starved.

**Which PKM/Engram mechanism would have prevented it — precise, not vibes:**

1. **Unlearned addressing (the root cause).** Our hash assigns slots before training; nothing about a slot is chosen by content. A byte 8-gram is nearly unique in 46 GB, so its row *is* the empirical next-byte distribution for that exact context — SGD on the CE loss makes the row a counter. Once fit, the read returns the answer, CE ≈ 0, and **∂L/∂h → 0 kills every gradient into the backbone**. In PKM, addressing is a learned projection of the hidden state: while any loss remains, the query network (a backbone param) keeps receiving gradient, so starvation is structurally impossible. In DeepSeek Engram, same: addressing is a fixed hash **but** the read is 16 rows (orders {2,3,4} × 8 heads) mixed by a learned `value_proj` — no single row ever has to carry the whole answer.
2. **Row specificity / missing "tokenizer compression".** DeepSeek's #1 modernization is compressing 128K BPE IDs into ~26K normalized tokens before hashing; their n-grams run over *tokens* (each 3-gram spans ~10–30 bytes) and cap at **n=3 (Engram-27B)** or **{2,3,4} (V4.1-Flash)**. Byte-level 3/5/8 with n=8 has ~5–8× the specificity per order; rare-order rows see a handful of samples and fit noise. This matches our own finding (`research/2026-09-26-byte-lm-recipes.md`: hash n-grams as input features, n=3–4 first, diminishing returns past 300–500K rows) — the published configs independently corroborate it.
3. **Capacity ratio.** DeepSeek's allocation law (U-shaped): optimum ≈ **20–25% of the sparse budget** to Engram (val loss 1.7248 → 1.7109 at ρ≈80/20 in the 10B regime); production: 196B Engram vs 552B backbone (~35%), and the module sits at **2 of 40 layers**, added to the residual. Ours: 24M rows × 96 dim ≈ 2.3B params vs a **7.5M backbone** — the "memory" was 99.7% of the model. No gating scheme rescues that ratio; the backbone had almost nothing to starve *for*.
4. **What was *not* the problem.** The gate is faithful to the official demo (verified line-by-line, §3.2) and bounded ∈ (0,1); even gate ≈ 1 isn't domination — informational monopoly with CE→0 is. And the external row optimizer is *also* faithful: V4.1-Flash really does use momentum+Sinkhorn on Engram tables (§3.3). The design copy was accurate; the *configuration* (n=8, byte-level, 24M rows on 7.5M backbone, and judging by train CE) was off-recipe.
5. **kNN-LM's one-line diagnosis** ([arXiv 1911.00172](https://arxiv.org/abs/1911.00172)): "although the Transformer is expressive enough to memorize all training examples, learning to do so does not result in context representations that generalize. In contrast, kNN-LM memorizes training data **while improving generalization**" — because the memory is *interpolated at λ=0.25*, never allowed to be the whole answer, and the LM stays the primary path.

---

## 3. What DeepSeek actually ships

### 3.1 The Engram paper is real, and it is DeepSeek's

[arXiv 2601.07372](https://arxiv.org/abs/2601.07372), "Conditional Memory via Scalable Lookup", DeepSeek-AI + PKU, published 2026-01-12 (ACL 2026 version calls the module **Deep Sparse Embedding, DSE** — same thing). Repo: `github.com/deepseek-ai/Engram` (4.7k stars, Apache-2.0, contains `engram_paper.pdf` + `engram_demo_v1.py`). **Not** EMC²/NVIDIA — that attribution in the task brief is wrong.

- Mechanism: **suffix token n-grams → deterministic multi-head hash → O(1) row lookup → contextualized sigmoid gating by the hidden state → optional short conv → added to the residual**. Modernizations over classic n-gram embeddings: *tokenizer compression, multi-head hashing (distinct primes per head), contextualized gating, multi-branch integration*.
- Results at 27B (iso-param, iso-FLOPs vs MoE-27B, 262B tokens): MMLU +3.4, CMMLU +4.0, **BBH +5.0**, ARC +3.7, HumanEval +3.0, MATH +2.4; NIAH 84.2 → 97.0. Mechanistically: memory absorbs early-layer static recall, freeing effective depth + attention.
- Allocation law: U-shaped; pure MoE is *suboptimal*; ~20–25% of sparse params to Engram wins. Engram-27B: 5.7B embedding params, **max n-gram 3**, 8 heads, dim 1280, modules at **layers 2 and 15**, **Adam lr 5×, no weight decay**, conv zero-init. 100B-param table offload to host RAM on vLLM: **<3% throughput overhead**.
- Scale behavior: "scaling the table increases stored parameters but does not increase the compute" — retrieval cost constant in table size; deterministic addressing enables prefetch.

### 3.2 "DeepSeek 4.1 uses Engram" — **VERIFIED TRUE**

DeepSeek-V4.1-Flash ([arXiv 2609.19969](https://arxiv.org/abs/2609.19969), Sept 2026; weights + `inference/model.py` on HF `deepseek-ai/DeepSeek-V4.1-Flash`; API news 2026-09-10):

- **§2.4.2**: "We augment DeepSeek-V4.1-Flash with Engram… We follow the original Engram design—tokenizer compression, multi-head hashing, context-aware gating, and multi-branch integration—with two modifications. First, we omit the short causal convolution… Second, we optimize the Engram embedding with momentum-based update followed by Sinkhorn balancing."
- **196B Engram parameters** (vs 552B backbone), two modules at **layers 1 and 14** (zero-indexed), n-gram orders **{2,3,4}**, **8 hash heads**, total embedding dim **2048 per order**, **each head indexes a ~16M-entry table, sizes = distinct primes**, tables + key/value projections in **FP8**, inference prefetch from host RAM via background RDMA.
- No DeepSeek model before V4.1 shipped it: V3/V3.1/V3.2 are MLA/DSA transformers; V4 (arXiv 2606.19348) adds CSA/HCA, mHC, Muon, a static hash-MoE bootstrap — **no Engram**. V4.1-Flash is the first production integration. The owner's claim is correct, with the nuance that "works great" rests on the 27B paper + V4.1's benchmark wins, and the production config dropped the short conv.

**Bit-for-bit anchor we already ported correctly:** the official `engram_demo_v1.py` gate is

```python
gate = (normed_key * normed_query).sum(-1) / sqrt(D)      # RMSNorm on both sides
gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
gate = gate.sigmoid()
```

— identical to `compute_gate()` in `vendor/burn-fused/crates/burn-engram/src/lib.rs` (which even documents why the plain sigmoid diverges). The demo's `MultiHeadEmbedding` (offset-addressed single table over prime-sized per-head tables) and XOR-multiply-mod-prime multi-hash also match our `hasher.rs` approach. Our port is faithful; this file is the canonical reference for any re-verification.

### 3.3 The row optimizer: dormouse's Nesterov+Sinkhorn is real — with one deviation

V4.1-Flash **§2.5, Algorithm 1** ("Sinkhorn-Balanced Updates for Engram / Embedding / Prediction Head"): Adam's m+v on 196B sparse params was too expensive, so tables/token-embedding/head use **momentum → Nesterov look-ahead → mask near-zero rows (ρᵢ ≤ τ·ρ̄) → alternate row/column L2 normalization (K odd steps) → Δ = √n·U⁽ᴷ⁾ → lr correction γ = 0.18** to match Adam's update magnitude; Nesterov momentum, **no weight decay**. "Like Muon, this approach requires only a momentum buffer while empirically outperforming Adam." Eq. 7: it equalizes row-wise and column-wise **RMS** of the update ("one row corresponds to one token index or n-gram identity"). §3.1.3: tables row-partitioned, batch-level prefetch (indices depend only on tokens), FP8 fetch, row/column scaling vectors kept across iterations.

Our `offload.rs` header cites "DeepSeek V4.1-Flash §2.5" — **verified accurate** (one momentum buffer, Nesterov, no WD). **One deviation: our `sinkhorn_l1` alternates L1 norms; the paper's Algorithm 1 alternates L2 (row-RMS equalization) and adds the near-zero-row mask + γ=0.18 correction.** If we ever re-enable host tables, aligning those three details is a ~10-line change and makes the optimizer claim exact.

### 3.4 DeepSeek 4.1's encoder-decoder — **VERIFIED TRUE**

- It is real, and it is called **CED (Causal Encoder-Decoder)**, §2.2, inspired by YOCO: 40 layers = **20-layer causal encoder + 20-layer decoder**. For global attention, the decoder's KV entries are **projected directly from the last encoder hidden state** H_{L/2} with per-layer weights (`C_l = H_{L/2} W_l^KV`) instead of each layer's own hidden state; SWA stays layer-wise everywhere. Result: prefill cost ≈ halved (O(NL) → O(NL/2)), **8B active params/token at prefill vs 16B at decode**; total 552B backbone, 1M ctx, FP4 KV cache (890 bytes/token, ~¼ of V4-Flash), SWA Bounded Replay (~⅛ persistent cache). Released 2026-09-10; paper arXiv 2609.19969 (17 Sep 2026); beats V4-Flash and (per DeepSeek) routes over V4-Pro.
- **Roles**: encoder = one-pass deep computation of a *reusable global KV*; decoder = cheap per-token generation that reads that KV. Asymmetric activation is the whole point — input-heavy agent workloads pay the small side.
- **Relevance to a 7.5M byte-level model:** mostly negative, honestly. CED buys prefill FLOPs at 552B scale; dormouse trains at batch-10 s512 where prefill/decode asymmetry is irrelevant, and a fixed-depth single stack is already our default (architecture v2). The *transferable* ideas are the cheap ones: YOCO-style KV sharing if we ever do long-context, head-wise Muon (§2.5), mHC-with-Sinkhorn as a PonderNet-free residual mixer (our `gr.rs` already prototypes gated residuals), and DSpark (already our aux head). **"Build an encoder-decoder" as a goal: not recommended at this scale.**

---

## 4. For dormouse — verdict

**Recommendation: keep the hash, keep the verified optimizer, fix the configuration — option (B) with DeepSeek's published discipline; do not build PKM now; do not build the hybrid.**

The train-CE collapse was an information-routing pathology specific to *unlearned addressing at insane capacity ratio*, not a verdict on lookup memory: DeepSeek ships essentially our module (verified gate formula, verified row optimizer) at 27–752B scale with (a) n ≤ 4 over compressed tokens, (b) 20–35% of params, (c) 2 of 40 layers, (d) eval-judged training. Our byte-level setup can't do (a) via tokenization (vocab 256 is already minimal — a byte-pair "poor-man's compression" would be our own invention, unpublished), so we compensate with small n and small tables.

**Option B — keep hash, Engram-faithful config, bounded contribution.** ~30–80 LOC (mostly config + logging; the module code already exists and is demo-faithful).
- n-grams ≤ 4 (drop the 8-gram arm), total rows ≤ ~1M (Engram-27B ratio: 4.45M rows / 262B tokens → ~800K rows at our 46B bytes; matches our own 300–500K-row diminishing-returns finding), memory at 1–2 layers, Adam-or-Nesterov rows with no weight decay, L2 not L1 Sinkhorn + γ≈0.18, and **judge only by eval BPB** — train CE is meaningless with any memorizer attached. The existing sigmoid gate already bounds the memory's share per head; if we want the kNN-LM guarantee verbatim, fix λ=0.25-style scaling on the memory branch (~5 LOC).
- **Verifiable against:** `github.com/deepseek-ai/Engram` `engram_demo_v1.py` (gate, hashing, offsets — already matching line-for-line), Engram paper §4 config table, V4.1-Flash §2.4.2/§2.5 + HF `inference/model.py` (config fields `engram_max_ngram_size`, `engram_num_embeddings`, `engram_n_heads`…).
- **Collapse-resistance argument:** the backbone always keeps ≥ the non-memory share of the probability mass and its gradients; rows are coarse enough (n≤4, high per-row support) that no row singly answers; capacity ratio restored to ~5–10% so the core is the model.

**Option A — full PKM (learned product keys, soft top-k reads).** ~300–500 LOC (query proj, two sub-key params, topk → k² combine → softmax → weighted gather-sum, separate value optimizer; tests).
- **Blocked in practice:** the core op is top-k → gather — exactly the primitive that broke on pre.4 and got MSA disabled (ADR-0012: "pre.4 topk feeds garbage indices → gather OOB"). Revisit only after that's fixed.
- **Verifiable against:** `facebookresearch/XLM` `PKM-layer.ipynb` (reference implementation, ~150 lines) and `facebookresearch/memory`.
- **Collapse-resistance:** learned addressing keeps backbone gradients alive by construction; reads are softmax mixtures; documented stabilizers exist (query LN, values-only Adam 1e-3, output LN / non-competitive gates per 2302.06461/2310.10837).
- **But the honest counter-evidence:** at parameter-matched *small* scale, PKM loses to a dense FFN (WT-103: 12.77 vs 11.81, Csordás et al.). At 7.5M params, the lazy reading is that PKM adds machinery to lose to "make the FFN bigger". Skip until scale justifies a memory layer (>100M params, or factual-QA-style eval targets).

**Option C — hybrid (product keys over n-gram hashes).** No published design combines them; product keys exist precisely to *replace* hash addressing with learned addressing, so hashing beneath them forfeits their only benefit while keeping their cost. **NOT VERIFIED as an existing technique — would be pure invention. Skip.**

---

## Source register (all opened and read)

| Claim cluster | Source |
|---|---|
| PKM mechanics, defaults, sparse updates, multi-head, usage | [1907.05242](https://arxiv.org/abs/1907.05242) HTML v2 + `facebookresearch/XLM` `PKM-layer.ipynb` |
| PKM at 128B params, optimizer-state cost, kernels | [2412.09764](https://arxiv.org/abs/2412.09764) v1/v2 + `facebookresearch/memory` README |
| FwPKM, L_addr anti-slot-collapse, top-8, gates | [2601.00671](https://arxiv.org/abs/2601.00671) v1/v2 + `SakanaAI/fast-weight-product-key-memory` |
| Softmax over-centralization, LN fix, ReLU gates | [2302.06461](https://arxiv.org/abs/2302.06461) |
| PKM < dense at small scale (12.77 vs 11.81) | [2310.10837](https://arxiv.org/abs/2310.10837) + ACL Findings PDF |
| kNN-LM λ=0.25, memorization-vs-generalization quote | [1911.00172](https://arxiv.org/abs/1911.00172) v2 |
| Engram paper, U-law, 27B config, <3% offload | [2601.07372](https://arxiv.org/abs/2601.07372) + ACL 2026 long.226 PDF |
| Official gate/hash/embedding code | `github.com/deepseek-ai/Engram` `engram_demo_v1.py` (read in full) |
| V4 lineage (no Engram before V4.1), CSA/HCA/mHC/Muon | [2606.19348](https://arxiv.org/abs/2606.19348) + deepseek.com V4 preview + HF DeepseekV4Config |
| V4.1-Flash: CED, 196B Engram config, Algorithm 1, §3.1.3 | [2609.19969](https://arxiv.org/abs/2609.19969) full HTML + HF README + `inference/model.py` + API news 260910 |

**NOT VERIFIED (stated plainly):** any PKM-era documentation of a "memory monopolizes the loss" failure mode (§1.5); any published product-keys-over-hash hybrid (§4C); the task brief's PKM arXiv ID 1907.05642 and the "EMC²/NVIDIA" Engram attribution (both wrong as stated).
