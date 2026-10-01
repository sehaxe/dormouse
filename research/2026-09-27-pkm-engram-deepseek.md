# Product-key memory vs our hashed Engram, and what DeepSeek actually ships

**Date:** 2026-09-27 · **Method:** primary sources only, 12 rounds. Every arXiv PDF was downloaded and read with `pdftotext -layout`; every repo file was fetched raw and read in full. Load-bearing claims carry a section/line/equation anchor. Anything unconfirmed is marked **NOT VERIFIED**.

**Relationship to the 2026-09-26 note** (`research/2026-09-26-pkm-engram-deepseek.md`): that file's conclusions on PKM mechanics and on the DeepSeek Engram mechanism **re-check out**. This round adds four things it did not have:

1. **A published, measured instance of our exact pathology** in Engram-style memory — arXiv 2601.16531, which finds the gate is **anti-correlated with per-token loss**. This is the single most important new source.
2. **A published saturation curve for hashed n-gram slot count** (same paper): 300K → 500K helps, 500K → 800K *hurts*.
3. **The real shipped DeepSeek-V4.1-Flash code** (`config.json` + `inference/engram.py` + `inference/model.py` on HF) as a bit-for-bit anchor, including the production gate kernel.
4. **Our own code read line-by-line**, which corrects two details in the task brief and produces a sharper capacity arithmetic (§2.1).

Corrections to the brief, stated plainly:
- The brief's arXiv ID for the PKM paper is wrong; it is **arXiv 1907.05242** ("Large Memory Layers with Product Keys", Lample, Sablayrolles, Ranzato, Denoyer, Jégou, NeurIPS 2019).
- "the 2026 paper *Engram: conditional memory via scalable lookup* … (EMC²/NVIDIA?)" — the real title is **"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"**, arXiv **2601.07372**, **DeepSeek-AI + Peking University**. There is **no** EMC² paper and **no** NVIDIA-authored Engram paper; "NVIDIA" appears in 2601.07372 only as the benchmark hardware (H800). The ACL 2026 camera-ready (aclanthology 2026.acl-long.226) renames the module **Deep Sparse Embedding (DSE)** — same design, earlier name.
- "DeepSeek 4.1 has an encoder and a decoder" is **TRUE**, but the mechanism is **YOCO-style KV provenance sharing**, not a seq2seq cross-attention encoder-decoder (§4).

---

## 1. Product-key memory: the mechanics, exactly

### 1.1 The sub-key product structure — why millions of slots with no hashing

Source: 1907.05242 §3.1, §3.2, read in full.

The memory has three parts: a **query network** `q: x ↦ q(x) ∈ R^{d_q}` (typically a linear or MLP reducing to `d_q = 512`), two **sub-key codebooks** `C, C'` of `√N` vectors each of dimension `d_q/2`, and a **value table** of `N = |C|·|C'|` slots. The full key set

```
K = {(c, c') | c ∈ C, c' ∈ C'}          |K| = |C| × |C'|
```

is **never materialized**. The query is split into halves; the top-k sub-keys of each codebook are taken; the true top-k product keys are **guaranteed** to lie in the `k × k` combinations; those `k²` candidates are then rescored. Complexity `O((√N + k²)·d_q)` — the paper reports ~10³ fewer operations than exhaustive search at `|K| = 1024²`.

**Memory per parameter (the ratio the brief asks about).** The values hold the bulk of the parameters and "scale quadratically with the number of sub-keys" (1907.05242 §1). Key storage is `2·√N·(d_q/2) = √N·d_q` parameters. At the reference config (`n_keys=512` → `N = 262144` slots, `k_dim=256`): 262144 value rows vs `512·256 = 131072` key params — **half a key param per slot**. So product keys give you a huge address space for a *linear* key budget, and the keys are **trainable** (`--mem_keys_learn`, default `True`; `nn.Parameter`, not a buffer — `facebookresearch/XLM xlm/model/memory/memory.py:566`).

### 1.2 The read: soft top-k, k = 32, no learned temperature

The reference equations (1907.05242 §3.1, eqs. 1–3):

```
I = TopK_k( q(x)ᵀ k_i )                  # k nearest product keys
w = Softmax( q(x)ᵀ k_i )_{i∈I}           # renormalized over the top-k ONLY
m(x) = Σ_{i∈I} w_i · v_i
```

- **Softmax is over the k selected scores only** — the read is a *bounded convex mixture of k rows*, never a single row. This is the structural reason a PKM read cannot become a lookup table for one key.
- **No learned temperature, no temperature at all** in the paper. The XLM reference exposes `mem_score_softmax` (default on), plus optional `mem_score_subtract` (min/mean/median) and `mem_score_normalize` (L1) variants — all off by default.
- **Reference hyperparameters** (XLM `memory.py` `register_args`, lines 234–290): `mem_implementation=pq_fast`, `mem_k_dim=256`, `mem_heads=4`, `mem_knn=32`, `mem_n_keys=512` → `n_indices = 512² = 262144` slots, `mem_v_dim=-1` (= model dim), `mem_sparse=False`, `mem_query_batchnorm=False`.
- Paper's own conclusion (§4.5, "Number of heads / k-NN"): configurations with equal `h × k` — `(1,64) (2,32) (4,16) (8,8)` — all land at ~70% memory usage and ~20.5 ppl; "**using 4 heads and 32 k-NN strikes a good trade-off**". **128 slots touched per token.**
- **Separate optimizer for the values**: `--mem_values_optimizer` default `"adam,lr=0.001"` (XLM `memory.py:240`) while keys + query net train at the backbone LR (2.5e-4). Paper §4.3: "Since the memory values are learned with sparse updates, we found it beneficial to learn them with a higher Adam learning rate of 10⁻³." The value table is an `nn.EmbeddingBag(..., mode='sum', sparse=mem_sparse)` (`memory.py:79`).
- **Integration**: replaces the FFN of `N ∈ {0,1,2}` layers, `x ← x + m(x)` (§3.3, Fig. 3). Paper's placement ablation on a 6-layer model (§4.5, Table 3): layer 1 → 21.5 ppl, **layer 4 → 20.1, layer 5 → 19.8**, layer 6 → 20.3. "The best position to insert the memory is at an intermediate layer."

### 1.3 Meta at scale: what 128B memory params actually needed

Source: 2412.09764 v2, read in full.

- Same formulation, **shared value pool across all memory layers** (§3.1.3): "multiple memory layers increase performance significantly over having a single layer with the same total parameter count, up to a certain number of layers (in our case, 3). **Beyond this point, replacing further FFN layers degrades performance**, showing sparse and dense layers are both needed and likely complementary."
- **Why product keys and not an ANN index** (§3.1.1) — the crispest one-sentence statement of the whole PKM design rationale: "*it's a challenge to incorporate [fast approximate vector similarity] when the keys are being continually trained and need to be re-indexed.*" A hash has no index to re-index; that is precisely the property PKM gives up and pays √N·d_q params to buy back.
- **Read** (eq. 1): `I = SelectTopkIndices(Kq)`, `s = Softmax(K_I q)`, `y = s V_I`.
- **Memory+ block** (eq. 2): `output = (y ⊙ silu(xᵀ W₁))ᵀ W₂` — input-dependent gating. Ablation (Table 3, right): **swilu consistently helps; plain gating only sometimes;** adding random key-value pairs and a "softmax sink" fixed key gave gains inside the noise and were dropped for speed.
- **The documented instability** (§3.1.4), verbatim: "We find that **for large memory layers, training can become unstable, especially for small base models. We use qk-normalization when needed to alleviate this issue.**"
- **Optimizer state is a first-class cost** (§3.1.2): "memory layers are naturally memory-intensive, mostly due to the large number of trainable parameters and associated optimizer states." Kernels: 3 TB/s forward EmbeddingBag (vs <400 GB/s PyTorch), atomic-free `reverse_indices` backward, 6× end-to-end over PyTorch.
- **Sizing** (Table 4): value dim = model dim is optimal; trading dim for slots loses (1024 dim / 1M values 2.11 > 2048 dim / 512K values 2.14); key dim = half the model dim is best of those tried; 16M values × 2048 = 64B memory params on an 8B base, 138.7B total.
- **Where the gains are**: factual QA, decisively. Memory+ 138.7B beats parameter- and compute-matched MoE and PEER, and approaches Llama2-7B trained with 2× tokens and 10× FLOPs. Table 1: Memory+ (1T) 60.29 HellaSwag / 63.04 MMLU / 27.06 NQ vs dense (1T) 58.90 / 59.68 / 25.24.

### 1.4 FwPKM: the in-slot gradient rewrite, and the gate that does the real work

Source: 2601.00671 v2 (Zhao & Jones, Sakana AI), read in full; repo `SakanaAI/fast-weight-product-key-memory` exists (README + code, confirmed via the GitHub contents API).

- **Retrieval is unchanged from PKM**; only the memorization changes. Within a chunk of `C` tokens, the value matrix `V` is updated by chunk-level gradient descent on a local reconstruction loss, and the key matrices `K₁, K₂` by an auxiliary **addressing loss**.
- **Gated reconstruction loss** (eq. 13): `L_mem = Σ_t (1/C)·(g_t/2)·‖v_t − v̂_t‖²`. Note the `g_t` factor — **the loss that trains the memory is itself gated**, so when the model stops trusting the memory the pressure stops too.
- **Gated residual output** (eq. 12) — this is the mechanism the brief is really asking about:

```
o_t = g_t · v̂_t  +  (1 − g_t) · v_t          g_t = σ(Linear(RMSNorm(h_t))) ∈ (0,1)
```

  A **convex combination** of the memory read and a dense value path from the same hidden state. Compare to our Engram, which computes `y = attn·w_attn + engram·w_mem + ffn·w_ffn` — a *sum* of three branches with no convexity anywhere.
- **Anti-slot-collapse objective** (eqs. 16–18): marginal slot-usage distributions `p̄₁ = (1/C)Σ_t s'_{t,1}`, `p̄₂ = (1/C)Σ_t s'_{t,2}` over the chunk, and `L_addr = −H(p̄₁) − H(p̄₂)`. Maximizing the entropy of the *marginal* key usage (not per-query uniformity) updates only the keys. Stated motivation: "Sparse memories can suffer from **memory slot collapsing**, where only a small fraction of slots are used."
- **Practical stabilizers** (§3.5): lookahead value targets (pair `q_t` with `v_{t+1}`), **inverse-distance-weighting** scoring (`−log(ε + ‖q − K_i‖²₂)` instead of dot product, so keys become clustering centroids), z-scored value targets.
- **Read width**: PKM reads Top-128 (4 heads × 32); FwPKM reads **Top-8 (1 head × 8)**.
- **The documented failure is the OPPOSITE of ours** (§4.2, Finding 2): "these models **learn to ignore FwPKM**, with gating weights clustering near zero." Their fix is to *starve* the long-range path (sliding-window attention with p=0.9 at train time) so the fast memory has to be used. **This is important: a learned scalar gate in this literature is empirically a two-way knife, and the measured failure direction is usually gate→0, not gate→1.**

### 1.5 Does PKM suffer "memory monopolizes the loss"? — and the honest small-scale result

**No PKM paper documents our exact pathology. NOT VERIFIED as a known PKM failure mode.** What *is* documented:

1. **Over-centralized softmax at large slot counts.** Shen et al., 2302.06461 ("A Study on ReLU and Softmax in Transformer"): softmax+LayerNorm makes the read "equivalent" to ReLU, and "**ReLU outperforms Softmax on both FFN and key-value memory when the number of value slots is large**" — the softmax "carries too little information" and becomes "over-centralized in a small number of slots, thus insufficient to utilize the context information of other slots."
2. **Key-usage collapse is measured, not hypothetical.** Lample §4.2 defines *memory usage* (fraction of accessed values) and KL-to-uniform, and reports query **BatchNorm** lifting usage from **25.8% → 80.3%** at 1M slots with ppl 19.8 → 18.0. Table 4: flat keys get 10–20% usage; product keys get 97–100%.
3. **Slot collapse in fast weights** is what `L_addr` exists for (§1.4).
4. **The strongest counter-evidence for PKM at our scale.** Csordás, Irie, Schmidhuber, 2310.10837, Table 2, **parameter-matched** (they enlarge the dense baseline's `d_ff` to match):

   | variant | nonlinearity | WT-S | WT-B | Enwik8 |
   |---|---|---|---|---|
   | Dense baseline | ReLU | **11.81** | **9.46** | **1.08** |
   | PKM | Softmax | 13.96 | 11.10 | 1.16 |
   | PKM | ReLU | 12.77 | 9.98 | 1.11 |

   Their text: "**even the best PKM models underperform the dense baselines, indicating the fundamental limitation of PKMs.**" Note the caveats: they place PKM in *every* MLP block (a stress test, not the 1-in-2 config either Lample or Meta use), on a 47M backbone.
5. **kNN-LM's memorization experiment** (1911.00172, ICLR 2020) is the closest published statement of our failure mode, arrived at from the other direction: "**although the Transformer is expressive enough to memorize all training examples, learning to do so does not result in context representations that generalize. In contrast, kNN-LM memorizes training data while improving generalization.**" Measured: interpolating a *memorizing* LM with the base LM buys 0.1 ppl, versus 1.9 from kNN-LM.

**What papers actually do to bound the memory's contribution — the concrete list:** convex/softmax-mixture reads over k rows (§1.2); query BatchNorm or query normalization for usage coverage (Lample) / qk-normalization for small-base instability (Meta) / L2-normalized query halves (XLM `mem_normalize_query`); **ReLU instead of softmax** for large slot counts (2302.06461, 2310.10837); a **separate, higher-LR, no-weight-decay optimizer for the value table** (Lample 1e-3 Adam; DeepSeek Adam ×5 wd=0); an **entropy bonus on marginal key usage** (FwPKM `L_addr`); **convex gating of both the read and its loss** (FwPKM eqs. 12–13; kNN-LM eq. 3); **memory in 1–3 layers, replacing ≤3 FFNs** (Lample, Meta — more hurts); **qk-norm, not a bigger LR** (Meta); and **an explicit capacity budget** (§2.3).

---

## 2. Our Engram, read from the code, mapped mechanism by mechanism

### 2.1 What dormouse actually had (read from source, not from the brief)

| Property | Value in our code | Anchor |
|---|---|---|
| Hash | **FNV-1a 64**, not the reference prime-hash | `crates/dormouse-data/src/lib.rs:20-27, 315-330` |
| Orders | **3, 5, 8 over BYTES** (vocab 256) | `dormouse-data/src/lib.rs:318-328` |
| Heads | **1 per order** (3 rows/token) | `dormouse-data/src/lib.rs:329` |
| Address | `row = table_base(t) + (fnv(ngram) % slots[t])` | `dormouse-train/src/offload.rs:94-101` |
| Host table | `HostNgram::new([S,S,S], 32, seed)` — 3 tables × `--engram-slots` × dim **32** | `dormouse-train/src/lib.rs:603-605` |
| In-GPU table | `[4096,4096,4096]`, dim 32 | `dormouse-core/src/loop_block.rs:120` |
| Gate | `σ(sign(s)·√(|s|+1e-6))`, `s = ⟨RMSNorm(W_K e), RMSNorm(h)⟩/√d`, per hc-copy | `vendor/burn-fused/crates/burn-engram/src/lib.rs:214-233` |
| Integration | `y = attn·w_attn + engram·w_mem + ffn·w_ffn` — a **sum**, plus optional zero-init depthwise short conv | `dormouse-core/src/loop_block.rs:323`; `burn-engram/src/lib.rs:159-164` |
| Row optimizer | external host Nesterov + Sinkhorn | `dormouse-train/src/offload.rs` |

Two things this corrects or sharpens versus the brief:

- **The module port is faithful; the hasher is not.** `burn-engram`'s gate, branch structure, shared value proj, and zero-init conv match `engram_demo_v1.py` line-for-line (and the *production* V4.1 kernel, §3.3). But `burn-engram/src/hasher.rs` — which *is* a faithful port of the reference's odd-multiplier XOR-mod-prime polynomial hash — is **not on the training path**; the training path uses the FNV hash in the data crate. Same addressing family (deterministic, collision-noisy, ~50% collision rate at n=3 with 8M rows), different constant.
- **There already is a per-token-unaware global λ.** `w_mem` is the controller's per-iteration memory weight. It is a *scalar per loop iteration*, shared by all tokens — it can scale the memory branch globally but cannot discriminate per token. That is exactly the granularity kNN-LM's tuned λ operates at, and it is already wired. The kNN-LM interpolation guarantee is therefore a **one-line change**, not a new mechanism.

### 2.2 The capacity arithmetic (checkable; the interpretation is mine)

With `--engram-slots 8000000` (the interactive 8M-slot setting), 3 tables × 8M rows × dim 32:

- **768M memory params vs a 7.5M backbone → the memory was 99.0% of the model.** Row touches: 3/token × batch 10 × seq 512 = **15,360/step** (matches the observed figure in `.bulba/memory.md`).
- Per-order collision structure over a 46.2 GB corpus:

  | order | key space 256ⁿ | distinct n-grams in corpus | n-grams averaged per row | random-key collision prob |
  |---|---|---|---|---|
  | n=3 | 1.68e7 | ~4.6e10 (space exhausted 2750×) | **~2.1 by space; huge by frequency** | 52.3% |
  | n=5 | 1.10e12 | ~4.6e10 | ~5,775 | ~100% |
  | n=8 | 1.85e19 | ~4.6e10 | ~5,775 | ~100% |

  Reading: **only the 3-gram arm has per-key support.** n=5 and n=8 rows are each a ~5,775-way average of contexts — a high-dimensional smoothed feature, carrying no more information than the average of its members and burning 512M of the 768M params. The 3-gram arm, by contrast, is an ordinary byte-level n-gram cache with well-estimated conditional distributions on the frequent keys. **So the monopoly was a 3-gram phenomenon, and the 8-gram arm was 2/3 dead weight.** This matches the published ordering independently: DeepSeek caps at n=3 (Engram-27B) or n∈{2,3,4} (V4.1) and measured that spending budget on 4-grams is "slightly suboptimal… because it dilutes capacity from the more frequent 2/3-gram patterns" (2601.07372 §6.2).

### 2.3 Which PKM mechanism would have prevented the collapse — one mechanism per failure

| What went wrong | The precise PKM-family mechanism that prevents it | Source |
|---|---|---|
| **The gate cannot tell a trustworthy row from a collided one.** Our gate is `σ(⟨RMSNorm(h), RMSNorm(W_K e)⟩/√d)` — it sees the *embedding*, never the *support count* or the *collision rate*. It learns "frequent n-gram ⇒ open", and that preference then **fixates**. | **Give the read a bounded convex mixture over k slots instead of one row** (softmax over top-k, 1907.05242 eq. 2), or **convex-gate the read against a dense path** (FwPKM eq. 12: `o = g·v̂ + (1−g)·v`). A mixture of 128 rows cannot be a single-row lookup table, and the `(1−g)·v` term guarantees a dense floor under the memory. | §1.2, §1.4 |
| **The backbone's gradient dies because the memory explains the targets.** With CE→0 from the row itself, `∂L/∂h → 0`. | **Learned addressing.** A dot-product gate in PKM is a *backbone parameter*: while any loss remains, `W_K` and the query net keep receiving gradient, so the backbone can never be structurally starved. A hash has no such parameter — nothing in the hash is trainable, so the memory branch is a pure sink for the loss. This is the **root-cause** difference, and it is the same sentence Meta uses to justify product keys over an ANN index ("keys are being continually trained"). | 2412.09764 §3.1.1; §1.1 |
| **The gate is miscalibrated and fixates** (measured in the closest published replica: 2601.16531 Table 7 — α 0.2–0.4 ⇒ loss 3.90, α 0.8–1.0 ⇒ **loss 5.28**; ~70% of the high-α bucket is high-frequency). | **Entropy on marginal key usage** — `L_addr = −H(p̄₁) − H(p̄₂)` over the chunk — which forces the *addressing distribution* to stay broad instead of letting trust concentrate. And the negative-entropy signal is on the keys, so it keeps gradient flowing into the addressing path. | 2601.00671 §3.4 |
| **Softmax/concentration of the read at 24M slots** (the Lample BN result: usage 25.8% without query BN at 1M slots). | **Query normalization** — Lample's BatchNorm on the query net (25.8% → 80.3% usage), Meta's qk-normalization for small-base instability, XLM's L2-normalized query halves. Our gate RMSNorms *both* sides, so the scalar is bounded, but nothing bounds *which* row it trusts. | §1.2, §1.3 |
| **Row optimizer is decoupled from the graph**, so the rows take unbounded steps on a sink loss. | **A separate, higher-LR, no-weight-decay optimizer *inside* the graph** (Lample: Adam 1e-3 for values only; DeepSeek: Adam ×5, wd 0, and V4.1's momentum+Nesterov+Sinkhorn with γ=0.18). Ours already matches the *form*; what it lacks is the upstream **capacity budget** that keeps the ratio sane. | §1.2, §3.2 |
| **99% of the parameters were memory.** | **The allocation budget.** DeepSeek's law: 20–25% of the *sparse* budget to Engram (val loss 1.7248 → 1.7109 at ρ≈80/20 in the 10B regime); production V4.1 = 196B Engram vs 552B backbone. LongCat-Flash-Lite: "allocating over 30B parameters to embeddings" on a 68.5B model works, with the guideline **embed ≤ 50% of total**, N=3–5, K≥2. Our 99% is outside every published operating point. | §3.1; 2601.21204 |
| **n=8 diluted the budget** and bought nothing. | **n ≤ 4 over compressed tokens.** 2601.07372 §6.2 measures 4-grams as *slightly suboptimal* under a fixed budget. | §3.1 |

**What was NOT the problem**, so we do not rebuild it: the gate formula, the branch structure, the shared value projection, the zero-init conv, and the row optimizer are all faithful to the official reference (§3.3). The collapse was **configuration + unlearned addressing**, not a bad port.

### 2.4 The one paper that measures this exact thing

**arXiv 2601.16531, "A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics"** (Tao Lin, single-author preprint, 23 Jan 2026 — treat as weak evidence, but it is on-point and at our scale: 125M GPT-2 backbone, 128M Engram params, 500K slots/order, orders [2,3], 2 heads, dim 64, FineWeb-Edu 100M tokens, iso-parameter).

Three findings we should internalize:

1. **The gate is anti-correlated with loss** (Table 7, Hash-500K): α bucket 0.2–0.4 → avg loss **3.90** (lowest); α bucket 0.8–1.0 → avg loss **5.28** (highest). "The model assigns higher α to positions with higher loss… completely opposite to the gating design intent." ~70% of the high-α bucket is high-frequency keys. This *is* the monopoly mechanism, measured: the gate opens where the memory is about to be wrong.
2. **Hot→cold loss flip + preference fixation.** Early (iter 1k–2k) hot positions have lower loss and the gate prefers them; from iter ~3000 cold positions win but **α_hot stays above α_cold for the rest of training**. The gate's preference "crystallized". The whole α level declines 0.7–0.8 → 0.6–0.7 (the model gets more cautious in aggregate but never re-ranks).
3. **More slots is not better** (Table 3): Hash-300K 4.4825 · **Hash-500K 4.4809** · **Hash-800K 4.4961**. 500K is the optimum; **800K is worse than 500K by 0.015, ~2σ.** And collision-free indexing (MPHF hot tier) does **not** help (Nine-100/400K 4.4799 vs Hash-500K 4.4809, Δ=0.001 ≪ σ=0.008–0.012) and costs ~11% throughput (1910 → 1690 tok/s). Their conclusion: "**collisions act as implicit regularization**… the dominant limitation may lie in **gating credit assignment** rather than index accuracy."

This is the published **saturation curve** the brief was reaching for. It is not exactly 300–500K rows in the abstract — it is 300K/500K/800K **slots per (order, head)** with a 125M backbone and 100M tokens, and the optimum is 500K with a measured penalty at 800K. Our 8M slots/order sit **16× past** the measured optimum in that setup (and our whole table, 24M rows, is 48× past the brief's 300–500K figure). The conclusion is the same and is now *measured* rather than inferred.

---

## 3. What DeepSeek actually ships

### 3.1 The lineage, checked one model at a time

| Model | Memory mechanism | Verified how |
|---|---|---|
| V2, **V3** (2412.19437) | MLA + DeepSeekMoE. **Zero** occurrences of "engram" in the full text. | downloaded 2412.19437v1, `grep -c -i engram` = **0** |
| **V3.1** | none. Official model card: "The model structure of DeepSeek-V3.1 is the same as DeepSeek-V3." | HF `deepseek-ai/DeepSeek-V3.1` card |
| **V4** (2606.19348) | **Zero** occurrences of "engram". Has **hash routing for the first 3 MoE layers** (Roller et al. 2021, "Hash Layers for Large Sparse Models") — a *static expert router*, not a lookup memory. Also CSA + HCA, mHC, Muon. | downloaded 2606.19348v1, `grep -c -i engram` = **0**; §"Hash routing" at lines 347, 1379, 1399 |
| **V4.1-Flash** (2609.19969, 17 Sep 2026) | **First DeepSeek model to ship Engram.** §2.4.2 + §3.1.3. Plus CED, CSA2, DSpark. | full text + shipped code, below |

Note the hash-routing point is worth internalizing separately: DeepSeek *does* use a fixed hash in V4, but to route tokens to experts. Routing has a **balance constraint built in** — every token must go somewhere, and load is monitored and rebalanced every step — so it cannot suffer our pathology. A lookup table has no such constraint. Same primitive, opposite risk profile.

### 3.2 The Engram paper (2601.07372), verified details

Mechanism, in the order the paper describes it:

1. **Tokenizer compression** (§2.2): a surjective map `P: V → V'` collapsing raw token ids by normalized textual equivalence (NFKC, lowercase, …). "a 23% reduction in the effective vocabulary size for a 128k tokenizer." The shipped config says `engram_compressed_vocab_size: 99092` against `vocab_size: 129280` — **23.4%**, matching.
2. **Multi-head hashing** (§2.2, eq. 1): `K` distinct hash heads per n-gram order, "implemented as a lightweight multiplicative-XOR hash", each head indexing a table of prime size `M_{n,k}`. "Following Tito Svenstrup et al. (2017)."
3. **Context-aware gating** (§2.3, eq. 4): `α_t = σ( RMSNorm(h_t)ᵀ RMSNorm(k_t) / √d )`, with `k_t = W_K e_t`, `v_t = W_V e_t`. "if the retrieved memory `e_t` contradicts the current context `h_t`, the gate `α_t` tends toward zero, effectively suppressing the noise."
4. **Short depthwise causal conv** (eq. 5): `Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ`, kernel 4, dilation = max n-gram order.
5. **Multi-branch integration** (§2.4, eq. 6): **one shared value table and one shared `W_V`; `M` distinct `W_K`**, one gate per branch, mHC with `M = 4`.
6. **Deterministic addressing ⇒ host offload** (§2.5): 100B-param table in host DRAM, peak throughput penalty **2.8%** on an 8B backbone.

**Configurations, measured:**

- **Engram-27B** (Table 5): layers **[2, 15]**, n-grams **[2,3]**, `d_mem` 1280, 8 heads, per-head table 2,262,400 rows → 5.7B params; 262B tokens; Muon backbone, **Adam ×5 on embeddings, weight decay 0.0**; conv **zero-init**. Total 26.7B, 3.8B active. MoE-27B 1.634 → Engram-27B **1.622** → Engram-40B **1.610**.
- **Engram-40B** is the saturation datapoint: 3.2× the memory (18.5B vs 5.7B) buys **Δ0.012** val loss.
- **Placement**: layer sweep says **layer 2 is the single best injection point** (1.770 vs MoE 1.808); splitting the same budget over layers 2 and 6 is better still (1.768). "one round of attention is already sufficient to provide a meaningfully contextualized `h_t` for gating, while still being early enough to replace the backbone's bottom-layer local aggregation."
- **Ablation, in order of importance**: multi-branch fusion > context-aware gating > tokenizer compression. Removing 4-grams is mildly harmful ("dilutes capacity from the more frequent 2/3-gram patterns"). Removing the short conv is "marginally" harmful.
- **The allocation law is U-shaped and the memory end of it fails too**: at ρ→0% (memory-dominated), "the model loses conditional computation capacity, hurting tasks that require dynamic, context-dependent reasoning; **memory cannot replace computation in this regime**." Optimum ρ ≈ 75–80% (20–25% of the sparse budget to Engram), stable across two compute regimes.
- **Gate behaviour** (§6.5): α spikes on completed multi-token entities and formulaic phrases ("Alexander the Great", "By the way", "四大发明"), in both English and Chinese. "effectively relieving the Transformer backbone from memorizing these static associations."

### 3.3 DeepSeek-V4.1-Flash: the shipped code is the anchor

**Paper §2.4.2**, verbatim: "We augment DeepSeek-V4.1-Flash with Engram (Cheng et al., 2026c)… We follow the original Engram design—tokenizer compression, multi-head hashing, context-aware gating, and multi-branch integration—with **two modifications. First, we omit the short causal convolution** because its performance gains do not justify the added complexity in our inference stack. **Second, we optimize the Engram embedding with momentum-based update followed by Sinkhorn balancing.**"

"196B Engram parameters evenly across two modules. Each module uses N-gram orders {2,3,4}, with 8 hash heads and a total embedding dimension of 2048 per order. Each head indexes a table of approximately 16M entries, with table sizes chosen to be distinct primes. Both the embedding tables and the key/value projections use FP8 precision. The modules are placed at **layers 1 and 14 (zero-indexed)**… deterministic addressing enables embeddings to be prefetched from host memory via background RDMA transfers."

**Verified against the actual shipped files** on `huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash`:

`config.json` → `engram_layer_ids: [1, 14]`, `engram_num_embeddings: [384006168, 384016682]`, `engram_max_ngram_size: 4` (⇒ orders {2,3,4}), `engram_vocab_size: 16000000` (prime search starts below 16M), `engram_n_heads: 8`, `engram_head_dim: 256`, `engram_compressed_vocab_size: 99092`, `hc_mult: 4`, `num_hidden_layers: 40`, `hidden_size: 5120`, `n_routed_experts: 384`. Arithmetic check: 2 layers × 3 orders × 8 heads × 16M rows × 256 dim = **196.6B** ✓ matches the paper's 196B.

`inference/engram.py` (184 lines) — the hash, verbatim structure:
- `find_next_prime(start, seen)`; per layer, per (n-gram order, head) a **distinct prime, never reused**; ranges are disjoint bucket ranges in one offset table.
- `compute_hash_multipliers`: one odd multiplier per (layer, lookback) from `np.random.default_rng(10007 * layer_id)`, bounded so `token_id * multiplier` cannot overflow int64.
- `NgramHashState.forward`: ids → compressed table → per-position rolling XOR of `token * multiplier` over the `max_ngram_size` lookbacks, each reducing `% primes[:, i-1]`, then `+ offsets`. Look-back stops at sequence start **and at any `DEAD` token** (an image span), so an n-gram never spans one. A persistent `cache` tensor carries prefill/decode state. Comment: "every hash multiplier derives from the compressed vocab size, so a mismatch there would silently rehash the whole table."

`inference/model.py` (1309 lines), class `Engram` — **the production gate kernel**:
```python
kv = self.wkv(self.embed(hash_ids).flatten(-2))
key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)   # key: per-hc-copy, value: SHARED
rstd = rsqrt(mean(h²)+eps) * rsqrt(mean(key²)+eps)
dot  = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))  # "matching the training kernel"
return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2))
```
- The **signed-sqrt gate is the production kernel**, not a demo artifact. `q_weight`/`k_weight` are learnable per-(hc, dim) scale vectors used only as an elementwise product.
- `ParallelEngramEmbedding`: table stored as `float8_e4m3fn` + per-block `scale`, **dequantized on lookup**, masked rows zeroed, `all_reduce` across the row-shard group.
- `token_mask` zeroes the gate on positions that should pass through untouched.

**Optimizer, §2.5 Algorithm 1 (verified line by line):** Nesterov momentum → mask rows with `ρ_i ≤ τ·ρ̄` → alternate row-L2 (odd k) / col-L2 (even k) normalization for odd `K` → `Δ = √n·U^(K)` (converts unit row ℓ2 to unit row RMS) → `η̃ = γη` with **γ = 0.18** ("close to the factor 0.2 used in Moonlight") to match Adam's update magnitude. One momentum buffer, **no weight decay**. Rationale given: "applying Adam to the newly introduced Engram parameters substantially increases the optimizer-state memory footprint." §3.1.3: Sinkhorn row/column scaling vectors are carried across iterations to avoid rewriting the full normalized matrix; row normalization + partial column statistics fused into one kernel; tables stay in GPU memory during RL rollouts.

### 3.4 Second independent capacity data point

**arXiv 2601.21204, "Scaling Embeddings Outperforms Scaling Experts in Language Models"** (Meituan, 29 Jan 2026 → v2 11 Feb 2026): N-gram Embedding scaling beats expert scaling past a saturation threshold. **LongCat-Flash-Lite, 68.5B total / ~3B active, with over 30B parameters in embeddings**, surpasses parameter-equivalent MoE baselines. Operating points found: **N ∈ {3,4,5}, K ≥ 2**; embedding budget **≤ ~50% of total params** (beyond that, MoE overtakes it); more gains from width than depth. Also documents a collision-rate spike for 2-grams at table sizes near integer multiples of the base vocab — pick sizes off the multiples. This is a second, independent confirmation of both the capacity ceiling and the "n=3-5" ordering, from a group with no Engram affiliation.

---

## 4. The DeepSeek 4.1 encoder-decoder

**It exists, and it is called CED — Causal Encoder-Decoder, §2.2 of 2609.19969.** But the owner's framing needs one correction: it is **not** a seq2seq encoder-decoder with cross-attention. Both halves are causal; the change is **where the KV comes from.**

Verbatim mechanics:
- "For global attention, CED treats the bottom `L/2` layers of the Transformer as the **causal encoder**. For the upper half layers (i.e., the decoder, `l > L/2`), the KV entries are not derived from their respective hidden states `H_l`. Instead, they are **projected directly from the hidden state of the `(L/2)`-th layer**, `H_{L/2}`, using layer-dependent projection weights": `C^l = H_{L/2} W_l^KV`, `Z^l = H_{L/2} W_l^Z`.
- SWA stays **layer-wise everywhere** (local KVs from each layer's own `H_l`), which "effectively increases the computational depth of local KV generation" but requires an **SWA replay** process.
- "CED reduces the prefill complexity from `O(NL)` to `O(NL/2 + n_win·L/2) ≈ O(NL/2)`, effectively halving the overall computation." Decoder SWA Bounded Replay prefills only the last `n_win` tokens to kill the replay cost.
- Model: 40 layers = 20 encoder + 20 decoder, 552B backbone, **16B active/token at decode but 8B at prefill**, 1M context, CSA2 cross-layer KV reuse, FP4 global KV at 890 B/token (¼ of V4-Flash), SWA Bounded Replay persistent KV at ~⅛, 45T tokens. Shipped `config.json` has `kv_source_layer_ids: [2,8,14,20]`.

**Why it exists**: agentic workloads are input-heavy; "frequent tool calls generate extensive prefill requests… To alleviate this prefill bottleneck."

**Relevance to a 7.5M–1B byte-level trainer: negative as an architecture, and that is worth saying plainly.** CED buys prefill FLOPs at 552B scale where prefill dominates. dormouse trains at batch-10 s512 where prefill and decode are the same forward pass, so there is nothing to halve. Our default is already fixed-depth (architecture v2, `official_v4`). The transferable ideas are the cheap ones already on our list: head-wise Muon for q/k (§2.5, validated by GLM 5 and Kimi-K3), mHC-with-Sinkhorn as a PonderNet-free residual mixer (our `gr.rs` already prototypes gated residuals), and DSpark. **"Build an encoder-decoder" is not a goal at this scale.**

---

## 5. For dormouse — verdict and three ranked options

**The one-line verdict:** the collapse was not "hash memory is bad" — it was **a 3-gram n-gram cache holding 99% of the parameters with a gate that cannot see whether a row is trustworthy, judged on train CE**. The two published fixes for exactly that are (a) bound the memory's share with a convex mixture, and (b) give the read an entropy/balance signal so trust cannot concentrate. Both are cheap. Full PKM is not warranted at 7.5M: the strongest small-scale result in the literature (2310.10837) has parameter-matched PKM *losing* to a dense FFN 12.77 vs 11.81.

### Option B — keep the hash, add kNN-LM-style convex interpolation + the capacity budget · **RANK 1**

- **What changes:** (1) replace the *sum* `y = attn + engram + ffn` with a convex mix on the memory branch — `engram_branch = λ·(gate·W_V e) + (1−λ)·(dense value path)`, i.e. FwPKM eq. 12. We already have `w_mem` (a per-iteration scalar) doing the first half of this; we need the *dense* `(1−λ)` term and the λ to be a tunable constant swept on eval BPB, not on train CE. (2) Drop the n=8 arm (512M of the 768M params are a 5,775-way average). (3) Cut the table to ~500K slots/order — the measured optimum. (4) Enforce the budget: memory ≤ 10–20% of total params.
- **LOC:** **~40–70.** The convex mix and the n=8 drop are edits inside the existing `forward_embeds` (`burn-engram/src/lib.rs:182-210`); the dense value path is one `Linear(d, d)` reused from the backbone; λ sweep and the slot/order knobs are config. Everything else — hash, gate, row optimizer — stays.
- **Bit-for-bit verifiable against:** `SakanaAI/fast-weight-product-key-memory` eq. 12 (the convex form), `facebookresearch/XLM xlm/model/memory/memory.py:196-203` (`F.embedding_bag(..., per_sample_weights=scores)` — a weighted-sum read, same primitive), `1911.00172` eq. 3 for the λ-in-probability-space formulation, and `2601.16531` Table 3 for the 300K/500K/800K slot curve.
- **Collapse resistance:** the `(1−λ)` term is a hard floor — the memory can never be more than λ of the branch, so CE cannot be driven to 0 by the table alone and `∂L/∂h` cannot vanish. λ=0.25 (kNN-LM's tuned optimum on WT-103) is the starting point. The capacity cut removes the ratio pathology. The n=8 drop removes the dilution. This is the only option whose core guarantee is a *hard bound* rather than a learned one.

### Option A — full PKM (learned product keys, soft top-k read) · **RANK 3**

- **What changes:** query MLP with BatchNorm → split into halves → two `Linear` + `topk(k)` over sub-keys → `k×k` outer-sum of scores → second `topk(k)` + `gather` → softmax over the k → weighted `EmbeddingBag` sum; keys are `nn.Parameter`; values get their own Adam at 1e-3.
- **LOC:** **~250–400 in burn** (the PyTorch reference is ~90 lines: `XLM memory.py:640-687` is the whole index computation, `138-216` the forward, `525-566` key creation). Plus: a **separate value optimizer group** in `optim.rs`, and a test that the top-k→gather path doesn't reproduce the pre.4 gather-OOB bug that got MSA disabled (ADR-0012). That last item is the real blocker, not the LOC.
- **Bit-for-bit verifiable against:** `facebookresearch/XLM` `xlm/model/memory/` (4 files, `pq_fast` class) and `facebookresearch/memory` `lingua/product_key/` + `apps/main/configs/pkplus_373m_1024k.yaml`. Defaults to reproduce: `mem_k_dim=256, mem_heads=4, mem_knn=32, mem_n_keys=512, mem_values_optimizer="adam,lr=0.001", mem_query_batchnorm=True` (paper-recommended; XLM's own default is False), 262144 slots, replace 1–2 FFNs at layers 4–5 of 6.
- **Collapse resistance:** the strongest of the three — learned addressing keeps backbone gradients alive by construction, the read is a bounded 128-row mixture, and query BN + separate-value-LR + qk-norm are documented. But the *measured* small-scale verdict is that this buys nothing over a bigger FFN (2310.10837), and Lample's own placement ablation says more than 2 memory layers hurts.

### Option C — hybrid: product keys over n-gram hashes · **RANK 2 as insurance, but do not build it as a *design***

- **What it would be:** keep the FNV n-gram tables as the *values*, but replace the fixed addressing with learned product keys over a low-dimensional projection of the n-gram. This is the one option with **no published precedent** — product keys exist precisely to *replace* hash addressing, so putting a hash underneath keeps the cost and forfeits the benefit. **NOT VERIFIED as an existing technique; it would be our own invention.**
- **LOC:** **~200–300**, and the expensive part is the FwPKM `L_addr` term: a second gradient path through `topk` (entropy of the marginal key usage), which is a genuinely new backward for our stack.
- **The one thing it buys that A and B do not:** it is the only option where the memory *cannot* memorize a specific context, because no key ever resolves to a unique address. If we ever want a big table on a 7.5M backbone and distrust our own gate, this is the honest way to get it.
- **Verifiable against:** nothing bit-for-bit. Verifiable only as "the composition of two published mechanisms." That is a real cost, and it is why it ranks below B despite B's λ being a learned-free heuristic.

### Ranking, with the reasoning that produced it

1. **B** — smallest diff, the only hard bound, the anchor repos are the ones we already ported from, and both the capacity ceiling (2601.07372, 2601.21204) and the slot-count saturation (2601.16531) are *measured*, not inferred.
2. **C** — right instinct (no memorizable addresses), wrong provenance (no reference implementation to diff against). Park it.
3. **A** — best-understood mechanism, and the reason to skip it is empirical: at 47M–125M backbones, parameter-matched PKM loses to a dense FFN, and our top-k→gather primitive is currently the known-broken one on this burn version.

**Do not do:** rebuild the gate (it is faithful to production, §3.3), rebuild the row optimizer (it is faithful to V4.1 §2.5; the only deviation is L1 vs L2 Sinkhorn alternation and the missing γ=0.18 lr correction, ~10 lines if we ever re-enable), or re-litigate "DeepSeek uses Engram" (it does, from V4.1-Flash onward, with a shipped reference implementation we can diff against).

---

## Source register (all opened; PDFs read with `pdftotext -layout`, code fetched raw)

| Claim cluster | Source |
|---|---|
| PKM equations, defaults, BN/usage, placement, heads×k, separate value LR, flat-vs-product | arXiv **1907.05242** v2, full text §1, §3.1–3.3, §4.2–4.5 (Tables 2–4) |
| PKM reference implementation + exact CLI defaults, `EmbeddingBag`, `_get_indices` | `facebookresearch/XLM` `xlm/model/memory/memory.py` (687 L), `query.py`, `utils.py`, fetched raw; `register_args` lines 234–290 |
| Why product keys (keys continually trained), shared memory ≤3 layers, swilu, qk-norm instability, value-dim/key-dim ablations, 128B params | arXiv **2412.09764** v2, full text §3.1.1–3.1.4, §4, Tables 2–4; `facebookresearch/memory` README |
| Gated residual `o=g·v̂+(1−g)·v`, gated loss, `L_addr` negative entropy, IDW scoring, top-8 read, gate-ignores-FwPKM | arXiv **2601.00671** v2, full text §3.2–3.5, §4.2, §5.2, Figs 3/8; repo `SakanaAI/fast-weight-product-key-memory` (contents API) |
| ReLU > softmax at large slot counts; softmax+LN ≡ ReLU | arXiv **2302.06461** v1, abstract + §1 |
| PKM < parameter-matched dense at small scale (11.81 / 13.96 / 12.77) | arXiv **2310.10837** v1, §6.2 + Table 2 |
| kNN-LM eq. 3, λ=0.25 optimal, memorizing-LM experiment, generalization quote | arXiv **1911.00172** v2, §3, §4.2, §5 (Fig. 5) |
| Engram mechanism (eqs. 1–6), allocation law, Engram-27B/40B, Table 5, layer sweep, component ablation, gate visualization, 2.8% offload | arXiv **2601.07372** v2, full text §2.1–2.5, §3.1–3.2, §4.1, §6.2, §6.5, Table 5; ACL camera-ready `2026.acl-long.226` (module renamed DSE) |
| Engram official demo: signed-sqrt gate, prime hash, offsets, ShortConv, mHC branches | `github.com/deepseek-ai/Engram` `engram_demo_v1.py`, 422 lines, read in full |
| V4.1 Engram config, CED §2.2, Algorithm 1, §3.1.3, 45T tokens | arXiv **2609.19969** v1, full text §2.2, §2.4.2, §2.5, §3.1.3 |
| V4.1 **shipped code**: production gate kernel, FP8 table, prime hash, dead-token handling, config numbers | HF `deepseek-ai/DeepSeek-V4.1-Flash` `config.json`, `inference/model.py` (1309 L), `inference/engram.py` (184 L) — all fetched raw and read |
| V3 has no Engram | arXiv **2412.19437** v1 downloaded; `grep -c -i engram` = 0 |
| V3.1 structure == V3 | HF `deepseek-ai/DeepSeek-V3.1` model card |
| V4 has no Engram; has static hash routing for first 3 MoE layers | arXiv **2606.19348** v1 downloaded; `grep -c -i engram` = 0; hash-routing §at lines 347/1379/1399 |
| Gate anti-correlation with loss, preference fixation, hot→cold flip, 300K/500K/800K slot curve, collisions=regularization | arXiv **2601.16531** v1, full text, Tables 1–3, 7, §5.3–5.4, §6.3, §7–8 |
| Second capacity-ratio data point: >30B embeddings on 68.5B, ≤50% budget, N=3–5, K≥2 | arXiv **2601.21204** v2, abstract + reported findings |
| dormouse's own Engram: FNV 3/5/8-gram, table sizes, gate, residual form, host optimizer | read-only inspection of `crates/dormouse-data/src/lib.rs`, `crates/dormouse-train/src/{lib,offload}.rs`, `crates/dormouse-core/src/loop_block.rs`, `vendor/burn-fused/crates/burn-engram/src/{lib,hasher}.rs` |

### NOT VERIFIED (stated plainly)

1. **No PKM paper documents a "memory monopolizes the training loss" failure mode.** The nearest documented relatives are Meta's "training can become unstable, especially for small base models" (no mechanism given) and FwPKM's slot-collapse `L_addr` (a different pathology). The measured gate/loss anti-correlation that matches our symptom is from **2601.16531, a single-author preprint** — real, on-point, and weak evidence.
2. **A published product-keys-over-hash hybrid does not exist.** Option C is our own composition; there is no reference implementation to diff against.
3. **The task brief's "EMC²/NVIDIA" attribution for the Engram paper is wrong.** No such paper found. The paper is DeepSeek-AI + PKU, arXiv 2601.07372. NVIDIA appears only as benchmark hardware.
4. **The task brief's PKM arXiv ID (1907.05642) does not resolve to the PKM paper.** The correct ID is 1907.05242.
5. **"300–500K rows is where returns saturate" is not a citable published number in the abstract.** What is citable and adjacent: 2601.16531 measures Hash-300K 4.4825 / **Hash-500K 4.4809** / **Hash-800K 4.4961** at a 125M backbone with 128M memory params — optimum at 500K, measurably worse at 800K. Different setup, same shape. The brief's figure should be attributed as a dormouse measurement, not a paper's.
6. **We have no published byte-level n-gram BPB table for mixed text+math.** Every Engram number above is token-level (DeepSeek-V3 tokenizer, 26K–99K compressed ids). Our n=3-over-bytes row-support arithmetic in §2.2 is mine, derived from the corpus size and the slot count — the arithmetic is checkable, the interpretation is not a citation.
7. **The row-optimizer deviation is not re-verified in this round's source read.** Our `sinkhorn_l1` alternates L1 norms where V4.1 Algorithm 1 alternates L2 (`‖U_i,:^(k−1)‖₂`) and adds the `ρ_i ≤ τ·ρ̄` mask plus `γ = 0.18`; that comparison was made in the 09-26 note against the same §2.5 text and the §2.5 text I re-read this round is consistent. The claim about our own `offload.rs` internals was not re-audited line-by-line this round.
