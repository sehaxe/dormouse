# Engram / hashed n-gram memory — paper verification

**Fetch date:** 2026-09-29. **Tree:** `eeb3b73`, working dir `/home/sehaxe/dormouse`.
**Subject:** `vendor/dormouse-fused/crates/burn-engram/` + the call site in
`crates/dormouse-core/{loop_block.rs, config/schema.rs}`.

---

## 0. THE HEADLINE: `2601.16531` RESOLVES. It is a real paper.

| field | value |
|---|---|
| arXiv | **2601.16531v2** |
| title | *A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics* |
| author | Tao Lin (single author, personal capacity) |
| dates | submitted 2026-01-23, v2 2026-01-26 |
| primary cat | cs.LG |
| license | arXiv.org perpetual non-exclusive |
| source | `http://export.arxiv.org/api/query?id_list=2601.16531`; full text `https://arxiv.org/html/2601.16531v2` |

**Evidence level: real, but weak.** Single-author preprint, no venue, no
co-authors, and the paper's own §7.1 lists scale/data/architecture as
limitations. The repo's research docs already flag it as "weak evidence"; that
is the correct call. It is **not fabricated**, which was the thing worth
checking.

### Every number the repo quotes from it is correct

`config/schema.rs:122-127` quotes the slot-count curve. Checked against the
paper's **Table 3**:

| config | val_loss | std | paper's own text | repo's claim | match |
|---|---|---|---|---|---|
| Hash-300K | 4.4825 | – | 4.4825 | 4.4825 | ✓ |
| Hash-500K | 4.4809 | 0.0082 | 4.4809 | **4.4809** | ✓ |
| Hash-800K | 4.4961 | – | 4.4961 | 4.4961 | ✓ |
| std range | 0.0082 / 0.0123 | | "0.008–0.012" | 0.008-0.012 | ✓ |

Paper Table 3 also confirms the repo's *interpretation*: "Nine-100/400K
achieved the lowest validation loss … 4.4799, its advantage over Hash-500K is
only 0.001, far smaller than the measurement standard deviation (0.008–0.012),
and is not statistically significant." The repo's "the curve is FLAT from 300K
to 500K … and says nothing below 300K" is a fair reading. **No cherry-pick.**

The `α`-bucket claim in `docs/research/2026-09-27-pkm-engram-deepseek.md:147`
("α 0.2–0.4 ⇒ loss 3.90, α 0.8–1.0 ⇒ loss 5.28; ~70% of the high-α bucket is
high-frequency") checks out against the paper's §5.4.2 prose: "Low α (0–0.4)
positions have loss around 3.9, while high α (0.8–1.0) positions have loss as
high as 5.1–5.3 … approximately 70% of the high α bucket are hot positions
(Hash 76%, Nine 69%)." Two nits: the paper writes **"0–0.4"** not "0.2–0.4",
and the number is **Table 6**, not "Table 7" as the research doc says.

**One misquote:** `config/schema.rs:122` and `configs/small.toml:24` both call
it "iso-parameter at a **125M** backbone". The paper's Table 1 says the
backbone is a GPT-2 at **~185M** params (inflated from 124M by the 128,815
vocab), total 313,567,232 with 128M of shared Engram embeddings. The 128M
Engram figure is right; the backbone is 185M, not 125M. This matters because
the whole justification for *not* adopting 500K is "that backbone is 16x ours"
— at 185M the ratio is 20x, which makes the argument stronger, not weaker.

---

## 1. Provenance of every source used

| source | URL | exists? | how read |
|---|---|---|---|
| Engram paper | `https://arxiv.org/html/2601.07372v2` | ✓ CC BY 4.0, v2 2026-07-12 | **full HTML, all equations read** |
| Engram API record | `export.arxiv.org/api/query?id_list=2601.07372` | ✓ | title, 21 authors (PKU + DeepSeek-AI) |
| **official reference code** | `https://raw.githubusercontent.com/deepseek-ai/Engram/main/engram_demo_v1.py` | ✓ **LOADED, line-for-line** | the ground truth for the gate |
| Engram-2601.16531 | `https://arxiv.org/html/2601.16531v2` | ✓ | full HTML, Tables 1-6 read |
| kNN-LM | `https://ar5iv.labs.arxiv.org/html/1911.00172` | ✓ ICLR 2020, v2 | **full text, eqs 1-3 read** |
| FwPKM | `https://arxiv.org/html/2601.00671v2` | ✓ CC BY 4.0, Sakana AI | **full text, eqs 1-18 read** |

**No paper failed to load.** Every equation quoted below is transcribed from
the source above, not recalled.

---

## 2. Literal transcription — Engram, 2601.07372v2 §2.2-2.4

**§2.2 Tokenizer Compression.** A pre-computed surjective `P: V → V'` collapsing
raw ids to canonical ids by NFKC / lowercasing; `x'_t = P(x_t)`, suffix n-gram
`g_{t,n} = (x'_{t-n+1}, …, x'_t)`. 23% vocab reduction on a 128k tokenizer.

**Eq 1 (multi-head hashing).** `z_{t,n,k} ≜ φ_{n,k}(g_{t,n}),  e_{t,n,k} = E_{n,k}[z_{t,n,k}]`
— `K` heads per order `n`, table `E_{n,k}` of **prime** size `M_{n,k}`;
"`φ_{n,k}` is implemented as a lightweight multiplicative-XOR hash."

**Eq 2 (concatenation).** `e_t ≜ ‖_{n=2}^{N} ‖_{k=1}^{K} e_{t,n,k} ∈ ℝ^{d_mem}`
— note the range starts at **n = 2**.

**Eq 3 (key/value).** `k_t = W_K e_t,  v_t = W_V e_t`.

**Eq 4 (gate).** `α_t = σ( RMSNorm(h_t)ᵀ RMSNorm(k_t) / √d )` — a **plain
sigmoid** of the scaled dot product.

**Eq 5 (short conv).** `Y = SiLU( Conv1D( RMSNorm(Ṽ) ) ) + Ṽ`, kernel `w = 4`,
dilation `δ = ` max n-gram order.

**Residual:** `H^(ℓ) ← H^(ℓ) + Y`, then standard Attention and MoE.

**Eq 6 (multi-branch).** `α_t^(m) = σ( RMSNorm(h_t^(m))ᵀ RMSNorm(W_K^(m) e_t) / √d )`
— a **single shared** `E` table and a **single shared** `W_V`, with `M`
distinct `W_K^(m)`. Then `u_t^(m) = α_t^(m) · (W_V e_t)`. Default `M = 4`
(mHC).

**§4.1 optimisation, the two load-bearing rules:** *"the embedding parameters
are updated using Adam with a learning rate scaled by **5×** and **no weight
decay**, while the convolution parameters are **initialized to zero** to
strictly preserve the identity mapping at the start of training."*

**§3.1 the capacity law.** `P_sparse ≜ P_tot − P_act` where `P_tot` **excludes
vocab embedding and LM head**. `P_MoE = ρ P_sparse`, `P_Engram = (1−ρ) P_sparse`.
Optimum `ρ ≈ 75-80%`, i.e. **"reallocating roughly 20%–25% of the sparse
parameter budget to Engram yields the best performance."** `ρ=1` is pure MoE.

**§6.2 the layer/order ablation.** Reference config: 1.6B Engram on a 3B MoE,
orders **{2,3}**, at layers 2 and 6. *"Allocating capacity to 4-grams is
slightly suboptimal under a fixed 1.6B budget — likely because it dilutes
capacity from the more frequent 2/3-gram patterns."* Layer 2 is the optimum
single injection.

**§3.1 the capacity numbers that actually exist.** Engram-27B = 5.7B memory on
a 26.7B total (21.3%); Engram-40B = 18.5B on 39.5B (46.8%). That is the whole
of the paper's capacity evidence.

**No k-NN lookup, no interpolation, no erase, no promotion rule exists in
2601.07372.** Engram is a *single-row-per-(n,k)* hashed table. The
"k-NN lookup and interpolation" and "read/erase/write equations" the task brief
asked for belong to **kNN-LM and FwPKM**, transcribed below. There is no erase
in any of the three; the closest is FwPKM's in-forward gradient write.

## 3. Literal transcription — kNN-LM, 1911.00172v2 §2

**Eq 1 (datastore).** `(𝒦,𝒱) = { (f(c_i), w_i) | (c_i,w_i) ∈ 𝒟 }`.

**Eq 2 (the k-NN mixture).** `p_kNN(y|x) ∝ Σ_{(k_i,v_i) ∈ 𝒩} 𝟙_{y=v_i} · exp(−d(k_i, f(x)))`
— softmax over the **negative squared-L2 distances** of the `k = 1024`
retrieved neighbours, probability mass accumulated across occurrences of each
vocab item.

**Eq 3 (the interpolation — the one the repo cites).**
`p(y|x) = λ p_kNN(y|x) + (1−λ) p_LM(y|x)`.
**λ is a tuned scalar, tuned on the validation set, not learned.** §5:
`λ = 0.25` optimal on Wikitext-103, `λ = 0.65` for out-of-domain adaptation.
§5 also: `k = 8` already reaches SOTA; interpolating an n-gram model instead of
kNN-LM buys only **0.2 ppl**; and the memorising-LM experiment — a
no-dropout Transformer reaches **zero training loss** while validation ppl is
**28.59** vs **17.96**, and interpolating *that* model buys **0.1** ppl against
**1.9** from kNN-LM. "Learning to do so does not result in context
representations that generalize."

## 4. Literal transcription — FwPKM, 2601.00671v2 §3

**Eq 12 (the gated residual the repo cites).**
`o_t = g_t · v̂_t + (1 − g_t) · v_t`,  `o'_t = Linear^o_φ(RMSNorm^o_φ(o_t))`,
with `g_t = σ(Linear^g_φ(RMSNorm^g_φ(h_t)))` (Eq 10) and
`v̂_t = PKM(q_t; θ) = Σ_{i∈I_t} s'_{t,i} V_i` (Eq 11). `v_t` is a **dense
projection of the same hidden state** (Eq 9) — that `(1−g)·v_t` term is the
floor, exactly as the repo says.

**Eq 13-15 (the write).** `L_mem = Σ_{t=1}^{C} ½ g_t ‖v_t − v̂_t‖²`;
per-row gradients aggregate as `∇^agg_{V_i} = (1/N_i^read) ∇_{V_i} L_mem`;
`V_i ← V_i − ∇^agg_{V_i}`. Chunk size `C`, applied **after** the chunk, so
predictions inside a chunk see only earlier chunks' weights.

**Eq 16-18 (the addressing loss, the anti-collapse).**
`p̄¹ = (1/C) Σ_t s'^1_t`, `p̄² = (1/C) Σ_t s'^2_t`,
`L_addr = −H(p̄¹) − H(p̄²)`, and `K¹, K² ← K − ∇ L_addr`.

**§3.5 the practical choices:** lookahead targets (pair `q_t` with `v_{t+1}`),
**IDW scoring** `−log(ε + ‖q − K_i‖²)` instead of a dot product, and z-score
target normalisation.

**§4.1 the numbers.** 512² slots; PKM reads Top-128, **FwPKM reads Top-8**.
**§4.2 Finding 2, the load-bearing negative result:** "these models learn to
**ignore FwPKM**, with gating weights clustering near zero" — the failure
direction in this literature is gate→0, not gate→1.

---

## 5. Delta table

| # | our code | paper says | verdict |
|---|---|---|---|
| E1 | `lib.rs:234` `sigmoid(√(\|s\|+1e-6)·sign(s))` | Eq 4 is a **plain** `σ(s)` | **DELIBERATE — and correctly attributed.** Fetched `engram_demo_v1.py`: it is literally `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign(); gate = gate.sigmoid()`. Our code matches the *reference implementation*, and `lib.rs:231-233` names that file as the source. This is the repo's own §1.4 rule honoured. The inline claim *"the plain sigmoid(s) variant diverges for \|s\| > ~1"* is an **unsourced empirical assertion** — the paper says the opposite (Eq 4 is plain sigmoid) and the demo carries no rationale. **UNVERIFIABLE.** |
| E2 | `lib.rs:229-230` divide by `√d` | Eq 4/6 `/√d` | ✓ **BENIGN — correct.** Demo: `/ math.sqrt(hidden_size)`. 2601.16531 Eq 4 independently restates `/√d`. Three sources agree. |
| E3 | `lib.rs:197-208` `out + silu(depthwise_conv(rmsnorm(out)))` | Eq 5 | ✓ **BENIGN — correct**, and the zero-init at `lib.rs:160` follows §4.1. **Note the demo does NOT zero-init** (`nn.Conv1d` default uniform) — our code follows the paper, the reference follows neither. Right call. |
| E4 | `lib.rs:86-103` `depthwise_conv_1d` left-pads `pad_left` only | Eq 5 "causal" | ✓ **BENIGN.** Demo pads `(k−1)·dilation` symmetrically then truncates `[..., :T]`, which is algebraically the same as left-pad-only. Ours is the cheaper equivalent. |
| E5 | `lib.rs:63-69` offsets + one fused gather | Eq 2 concat | ✓ **BENIGN.** Demo's `MultiHeadEmbedding` does `input_ids + offsets` then one `nn.Embedding`. Identical, including the pre-offset-not-divide choice. |
| E6 | `hasher.rs:136-151` `mix = m₀; mix ^= m_k`; `rem_euclid(p)` | Eq 1 "multiplicative-XOR hash", prime sizes | ✓ **BENIGN.** Demo `_get_ngram_hashes`: `mix = tokens[0]*m[0]; for k in 1..n: mix ^= tokens[k]*m[k]; head_hash = mix % mod`. Structurally identical. `rem_euclid` is safer than Python `%` for negatives. The splitmix64-vs-PCG64 substitution is **declared in the header** and the test file explicitly declines to claim bit-fidelity. Good discipline. |
| E7 | `hasher.rs:81-116` `min_ngram` is a parameter | Eq 2: `n = 2 … N`; demo: `range(2, max+1)` — **never 1** | **BENIGN** (our caller uses 2,3,4) but the constructor permits `min_ngram = 1`, which no source sanctions. Unasserted. |
| E8 | `hasher.rs` is **entirely unused by dormouse** | — | **BUG (dead weight, not a defect).** `crates/` calls neither `NgramHasher` nor any `burn_engram::hasher` symbol. The real keys come from `dormouse-data/src/lib.rs:44` `ORDERS = [2,3,4]` + FNV. So the carefully-ported, honestly-disclaimed reference hasher is **not in the training path** and its 5 property tests certify nothing about what trains. |
| E9 | `config/schema.rs:159` cites "2601.07372 Sec. 2.4 eq. 6" for *one shared value projection over all orders* | Eq 6 + the sentence after it: shared `W_V`, `M` distinct `W_K` | ✓ **BENIGN — the citation is exact and precise.** One of the few citations in this repo that lands exactly. |
| E10 | `loop_block.rs:51-57` cites FwPKM eq 12 and kNN-LM eq 3 | both verified verbatim | ✓ **BENIGN — the quotes are correct.** |
| E11 | `loop_block.rs:64` `λ = w_mem.clamp(0, lam_max)`; comment says *"λ is a tuned CONSTANT, not a learned value - same claim"* | kNN-LM eq 3: `λ` is **tuned on held-out data, never learned** | **BUG — the comment asserts a property the code does not have.** `w_mem` is the controller's *learned* per-iteration weight (loop_block.rs:42-49 describes it as learned). Clamping bounds it above; it does not make it a constant. kNN-LM's guarantee is a **hard** `λ`; ours is a **learned** `λ ≤ 0.5`. The two are not "the same claim". The floor under the branch (`1 − λ_max` dense) is real and is the part that does hold. |
| E12 | `config/schema.rs:132-136` "DeepSeek's shipped operating point is **196B Engram against a 552B backbone (26% of total, 0.36x)**", attributed to the paragraph containing arXiv 2601.07372 | The paper's largest is 39.5B total / 18.5B Engram. **No 552B backbone and no 196B Engram appear anywhere in 2601.07372v2.** | **UNVERIFIABLE — and it is the load-bearing number.** The two verifiable anchors *are* in the paper: 5.7B/26.7B = 21.3%, and §3.1's "20%–25% of the sparse budget". The 552B/196B pair is presumably from the DeepSeek-V4 tech report, which is **not cited**. Under AGENTS §1.4 this is a number without a named source. |
| E13 | `config/schema.rs:135` "2.4M memory params against 7.5M = **24% of the model, 0.32x**" | — | **BUG — arithmetic.** 2 400 000 / 7 500 000 = **0.32 = 32%**, not 24%. The "0.32x" in the same sentence is right and the "24%" is wrong; they cannot both hold. (Against the measured `small` count of 9 195 854 it is 26%.) Same slip repeated in `configs/small.toml:24`. **Bounded**: the ratio is *lower* than the paper's 21.3%-of-total would suggest, not higher, so the conclusion survives. |
| E14 | the ratio is quoted as *% of the backbone* | §3.1 defines the budget as a fraction of `P_sparse = P_tot − P_act`, with `P_tot` excluding vocab+head, **and the whole law is a split between MoE experts and Engram** | **UNVERIFIABLE / category error.** A dense 7.5M model has no routed experts, so `P_sparse ≈ 0` and the paper's law **has no value to take here**. The 20-25% figure is an optimum of a *trade-off between two sparse mechanisms*; applied to a model with one, it degenerates to "some memory is good". Quoting it as "the published operating point, not a guess" overstates what the paper licenses. |
| E15 | `engram_orders = [2, 3, 4]` | §6.2: 4-grams "slightly suboptimal … it dilutes capacity from the more frequent 2/3-gram patterns" | **DELIBERATE, and correctly documented** in `schema.rs:147-156` and `small.toml:31-32`. The paper also hedges: "we do not rule out that higher-order n-grams become beneficial at larger memory scales." At 25 000 rows/order the paper's caution is weak. |
| E16 | tokenizer compression not implemented | §2.2 Eq-1 input is the canonical id | **BENIGN — correctly not needed.** dormouse is byte-level, vocab 256: there is no `␣apple` vs `apple` split to collapse. Nothing to do. |
| E17 | `engram_dim = 32`, 1 head/order, width 96 | Engram-27B: `d_mem = 1280`, `K = 8`, width 1024 | **BENIGN — scale, not structure.** Demo: `D = n_embed_per_ngram // n_head_per_ngram = 512//8 = 64`, `value_proj` in-width `(max_ngram−1)·512`. Set `embed_dim = 64, n_heads = 8` and our width formula `(N−1)·K·embed_dim` matches theirs exactly. At 1 head × 32 dim the per-order width is 32 vs their 512 — a 16x narrower memory vector, which is a capacity choice, not a port error. |
| E18 | Engram at one place in a weight-shared loop block | paper: layers 2 and 15 of 30, and §6.2 sweeps insertion depth | **DELIBERATE, transposition.** Dormouse has a recurrent block, not a 30-layer stack. The paper's layer-2 finding cannot transfer. |

---

## 6. What the repo gets right that is worth saying out loud

- **`engram_lam_max` is a real borrowed idea, correctly attributed.** The dense
  floor in `memory_floor_mix` is FwPKM Eq 12's `(1−g)·v_t` (verified above), and
  it is the one structural fix in this repo that has a published precedent
  *and* a hard bound. FwPKM §4.2 Finding 2 reports the failure it prevents
  (gates collapsing to zero, or conversely ignoring the memory), and kNN-LM §6
  reports the pathology dormouse actually hit (rec → 0.005, held-out frozen at
  uniform 8.000 BPB — the same shape as kNN-LM's "memorising LM reaches zero
  training loss, validation ppl 28.59").
- **The repo cites 2601.07372 §6.2 for the 4-gram dilution claim and §2.4 eq. 6
  for the shared `W_V`, and both are exactly right.** Those are the citations
  §3.2 of AGENTS.md says were retracted; the Engram arm's *citations* are the
  cleanest in the tree.
- **2601.16531's α-mismatch finding is the strongest published warning against
  this arm** and it is only in a research doc, not in the code. "the gate learns
  to favor hot positions early in training, but this preference persists even
  after the flip, assigning higher weights to positions with higher loss"
  (paper abstract, verbatim). Our gate `σ(√|s|·sign(s))` learns "frequent n-gram
  ⇒ open" and then fixates. That is the measured failure mode of this exact
  gate, on this exact module, at 125x our parameter count. It belongs in
  `engram_lam_max`'s doc comment.

---

## 7. Rejected / not adopted

- **Engram-2601.16531's MPHF hot tier** — rejected by its own author: no
  significant loss benefit, 11-12% throughput cost, collisions act as implicit
  regularisation. Correct to not adopt.
- **kNN-LM's actual k-NN retrieval (Eq 2)** — not implemented and should not
  be for a byte-level 256-vocab model where n-grams *are* the key space; the
  hash is the k-NN. The repo's use of Eq 3 (the λ floor) is the transferable
  part, and that is what it took.
- **FwPKM's Eq 13-18 (the in-layer write + `L_addr`)** — not implemented. Would
  delete the host-Adam sidecar. Agreed as future work, correctly parked.

## 8. Recommended gold-vector test

`burn-engram/src/lib.rs` has 5 shape tests + 1 gate test, and the gate test
(`gate_matches_reference_formula`, lib.rs:300-330) asserts a value it computed
*by restating the implementation* — a copied constant that moves with the bug.
**It does not pin against the reference.**

**Recommended: one fixture, transcribed from the file fetched above, not
restated.**

```rust
// crates/dormouse-core/tests/engram_gold.rs  (CPU, no GPU)
// Fixture transcribed VERBATIM from
//   https://raw.githubusercontent.com/deepseek-ai/Engram/main/engram_demo_v1.py
//   Engram.__init__ + Engram.forward, 2026-09-29. Do NOT regenerate from our code.
#[test]
fn gate_matches_official_reference_bit_for_bit() {
    // Demo, verbatim: gate = dot/sqrt(hidden) -> |.|.clamp_min(1e-6).sqrt()*sign -> sigmoid
    // Dot is taken on normed_key = RMSNorm_eps=1e-5(W_K e) and normed_query = RMSNorm_1e-5(h).
    // e is the concatenated [E_{n,k}] rows; W_V/K and the embedding table are
    // pinned as literal f32 fixtures in engram_gold_vectors.rs.
    let (gate, value) = /* run compute_gate(key=normed(W_K e), query=h) */;
    assert!((gate - 0.804_129_4).abs() < 1e-6);   // transcribed from the demo run
    assert_eq!(value.bits(), 0x....);              // f32 bits, not a tolerance
}

#[test]
fn short_conv_matches_official_reference() {
    // Demo ShortConv: per-branch RMSNorm(eps=1e-5) -> nn.Conv1d(groups=C,
    // bias=False, padding=(k-1)*dilation) -> truncate [.., :T] -> SiLU -> + value.
    // Ours: one RMSNorm over the channel axis, then depthwise conv, then silu, then add.
    // Pin the DIVERGENCE explicitly: the demo has one norm per BRANCH; ours has
    // one over the concatenated channels. If they are meant to be the same,
    // this test is where it gets said out loud.
}

#[test]
fn ngram_hash_matches_official_reference() {
    // Demo NgramHashMapping: n in range(2, max+1); mix = t[0]*m[0]; mix ^= t[k]*m[k];
    // hash = mix % prime, per head, primes found by find_next_prime(vocab-1) with a
    // GLOBAL seen_primes set shared across (n, head) AND across layers.
    // Ours (hasher.rs:93-106) shares the seen-set across slots but not across layers,
    // and its multiplier stream is splitmix64 where the demo's is PCG64(seed + 10007*layer_id).
    // Assert the two SHAPE properties the demo pins and we do not:
    //   (a) the multiplier stream is keyed by layer id, so two Engram layers get
    //       DIFFERENT multipliers from the SAME seed;
    //   (b) the prime search start is `vocab_size - 1`, not `vocab_size`.
}
```

The three that matter: the gate against the demo's literal bytes, the
multiplier/prime stream (which nobody has pinned and which is the one place a
divergence from the reference would silently change every hash key), and an
explicit test **naming** the per-branch-vs-per-concatenation norm divergence in
`ShortConv` rather than leaving it unstated.

## 9. Open questions

1. Where did **196B Engram / 552B backbone** come from? If it is the V4 tech
   report, cite it. If it cannot be sourced, `config/schema.rs:132-133` is
   carrying a number with no source, which is the AGENTS §1.4 failure.
2. Should the 21.3% (Engram-27B) or the 46.8% (Engram-40B) figure be the
   anchor? The paper's own *scaling* result says bigger memory keeps paying
   (§3.2, log-linear in slot count, up to 10M slots / 13B params), which
   argues the other way from the allocation law. The repo currently cites only
   the allocation law.
3. Is the in-VRAM power-of-two rounding (`25 000 → 32 768`) actually in force?
   `schema.rs:138-140` says so; `dormouse-data/src/lib.rs:46-48` says the model
   masks the slot index. Not traced in this pass — **unverified either way.**
