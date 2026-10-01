# Attention Residuals (arXiv:2603.15031) vs `vendor/burn-fused/crates/burn-attnres/`

**Scope.** Paper-fidelity audit of one crate. No code was built or run (no GPU on this box, and the
box is busy). Every "what we do" claim below is read out of the source at the line numbers given; every
"what the paper says" claim is quoted or transcribed from the PDF listed in §1. Nothing here is a
measurement.

**Verdict up front.** The paper's *only* executable specification is Fig. 2 (22 lines of PyTorch).
Our crate implements a **related but different function**: the `1/√d` logit scale is not in the paper,
our "RMSNorm" is not RMSNorm, and the Block streaming path violates Eq. 6 in three separate ways —
most seriously by folding the token embedding `b_0` into block 1's representation and by returning a
*plain unnormalized sum* where the paper returns a softmax over two sources. The fused CUDA
kernels are internally consistent with the tensor path (so the existing parity tests are green) and
are therefore *incapable* of catching any of this: every reference they compare against is our own
scaled formula.

---

## 1. Provenance

| item | value |
|---|---|
| paper | **Attention Residuals**, Kimi Team (Moonshot AI) |
| arXiv id | **2603.15031** (`cs.CL`), v1 submitted **16 Mar 2026**, 09:32:21 UTC, 622 KB |
| abstract page | https://arxiv.org/abs/2603.15031 |
| PDF (the copy I read) | https://arxiv.org/pdf/2603.15031v1 — downloaded to `/tmp/opencode/attnres/attnres.pdf`, 1 065 095 bytes, PDF 1.7, 9 pages, extracted with `pdftotext -layout` |
| DOI | https://doi.org/10.48550/arXiv.2603.15031 |
| license | CC BY-NC-ND 4.0 |
| authors' repo | https://github.com/MoonshotAI/Attention-Residuals (default branch `master`, HEAD `85e22310`, 4 commits, 3.5k stars, 205 forks) |
| date fetched | **2026-09-29** (paper, repo, third-party ports) |
| crate audited | `vendor/burn-fused/crates/burn-attnres/` — `src/lib.rs` (517 lines), `src/fused_attnres.rs` (1821 lines), `README.md`, `benches/attnres.rs` |

### Does original source code exist? **No.**

Verified, not assumed. The GitHub API tree for `master` (`/repos/MoonshotAI/Attention-Residuals/git/trees/master?recursive=1`)
returns exactly seven entries:

```
blob Attention_Residuals.pdf   952700
blob README.md                   5997
blob assets/logo.png            234859
blob assets/overview.png        324332
blob assets/scaling_law.png     122277
blob assets/training_dynamics.png 186168
```

* **one branch** (`master`), **no tags**, **no releases**, **no PyPI package**
  (`pypi.org/pypi/attnres/json` → 404, `pypi.org/pypi/attention-residuals/json` → 404),
  **12 open issues, 0 pull requests**.
* The arXiv LaTeXML HTML build is **broken** (https://arxiv.org/html/2603.15031v1 returns a stub page
  with no body). I used the PDF. If a tool hands you that URL as "the paper", it has nothing.
* The authors' repo PDF (952 700 B) is **not byte-identical** to the arXiv PDF (1 065 095 B); the
  arXiv one is the citable artifact and the one quoted here.

**Therefore the paper is being used as the specification, and there is no executable oracle.** The
closest thing to a reference implementation is the **22-line PyTorch pseudocode in Fig. 2 of the PDF,
which is duplicated verbatim in the authors' repo README** (https://github.com/MoonshotAI/Attention-Residuals#pytorch-style-pseudocode).
That is the only thing `burn-attnres` was ported from, and it is 22 lines of Python with no types,
no shapes beyond comments, and no numerical values.

### Third-party reimplementations (NOT oracles — independent transcriptions of the same 22 lines)

| repo | what it is | why it is not a reference |
|---|---|---|
| https://github.com/nktkt/attention-residuals | PyTorch, 1 commit, 5 stars, `attention_residuals/attn_res.py` | Read it. `nn.Linear(d,1,bias=False)` + `nn.RMSNorm`, `logits = torch.einsum("d, n b t d -> n b t", self.proj.weight.squeeze(0), K)` — **no `1/√d`**, matching the paper. **But it initialises `proj.weight` with `normal_(std=0.02)`, directly contradicting §5 ("all pseudo-query vectors must be initialized to zero")** — a clean demonstration of exactly the failure mode this project exists to catch. |
| https://github.com/kyegomez/attn_res | single-file PyTorch, 41 stars, 2026-03-16 | Not audited in detail. |
| https://github.com/wdlctc/open-attention-residuals | 83 stars, 2026-03-26 | Not audited in detail. |
| https://github.com/pytorch/pytorch/issues/177537 | open RFC `nn.BlockAttentionResidual` by a PyTorch core dev, 2026-03-16 | Not a merge. Useful only as a **second independent reading**: its `_attn_residual_op` is `logits = torch.einsum("d, n b t d -> n b t", w_l, K)` on `nn.RMSNorm(K)` — again **no scale** — and it explicitly documents `b_0` as "always the token embedding" and the partial as `b_n^i`. |

**These are corroboration, not verification.** Three parties transcribing the same pseudocode without
`1/√d` is evidence; it is not a source. The source is Eq. 2, Table 5 footnote 2, and Fig. 2 line 11,
and all three are unambiguous (§2.1).

---

## 2. The paper's specification, transcribed

### 2.1 Fig. 2 — literal transcription (PDF p. 5, lines 1–22 of the code box; identical in the repo README)

```python
 1 def block_attn_res(blocks: list[Tensor], partial_block: Tensor, proj: Linear, norm: RMSNorm) -> Tensor:
 2     """
 3     Inter-block attention: attend over block reps + partial sum.
 4     blocks:
 5         N tensors of shape [B, T, D]: completed block representations for each previous block
 6     partial_block:
 7         [B, T, D]:    intra-block partial sum (b_n^i)
 8     """
 9     V = torch.stack(blocks + [partial_block]) # [N+1, B, T, D]
10     K = norm(V)
11     logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
12     h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
13     return h
14
15 def forward(self, blocks: list[Tensor], hidden_states: Tensor) -> tuple[list[Tensor], Tensor]:
16     partial_block = hidden_states
17     # apply block attnres before attn
18     # blocks already include token embedding
19     h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
20
21     # if reaches block boundary, start new block
22     # block_size counts ATTN + MLP; each transformer layer has 2
23     if self.layer_number % (self.block_size // 2) == 0:
24         blocks.append(partial_block)
25         partial_block = None
26
27     # self-attention layer
28     attn_out = self.attn(self.attn_norm(h))
29     partial_block = partial_block + attn_out if partial_block is not None else attn_out
30
31     # apply block attnres before MLP
32     h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
33
34     # MLP layer
35     mlp_out = self.mlp(self.mlp_norm(h))
36     partial_block = partial_block + mlp_out
37
38     return blocks, partial_block
```

`proj` is a `Linear` used **bias-free** (`proj.weight.squeeze()` must yield `[d]`, so
`weight ∈ R^{1×d}`). `norm` is an `RMSNorm`. `block_size` counts **sublayers** (attention + MLP),
so a transformer layer contributes two.

### 2.2 The equations the pseudocode is a rendering of

Notation (§2): `h_l ∈ R^d` is the hidden state **entering** layer `l`; **the token embedding is
`h_1`**; `f_l` is the transformation applied by layer `l`; **each self-attention or MLP is an
individual "layer"**, so `L` = 2 × (transformer blocks).

```
Eq. 1   h_l = α_{0→l}·h_1 + Σ_{i=1}^{l-1} α_{i→l}·f_i(h_i),      Σ_{i=0}^{l-1} α_{i→l} = 1

        ϕ(q, k) = exp( qᵀ · RMSNorm(k) )                        ← §3.1, no scale factor
Eq. 2   α_{i→l} = ϕ(q_l, k_i) / Σ_{j=0}^{l-1} ϕ(q_l, k_j)

Eq. 3   q_l = w_l ;   k_i = v_i =  h_1  if i=0,  f_i(h_i)  if 1≤i≤l-1
Eq. 4   h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i

Eq. 5   b_n = Σ_{j∈B_n} f_j(h_j)                                 ← sum of LAYER OUTPUTS only
        b_n^i = partial sum over the first i layers in B_n ;  b_n = b_n^S

Eq. 6   V = [b_0, b_1, …, b_{n-1}]ᵀ                   if i = 1  (first layer of block n)
        V = [b_0, b_1, …, b_{n-1}, b_n^{i-1}]ᵀ        if i ≥ 2  (subsequent layers)
        with  b_0 = h_1  (the token embedding), §3.2: "The input of the very first layer of the
        network is the token embeddings, i.e. b_0 = h_1."
```

Table 5, footnote 2 (the compact statement of the whole mechanism):
`ϕ(q,k) = exp qᵀ RMSNorm(k)`; `k_i = v_i`; `v_0 = h_1`, `v_i≥1 = f_i(h_i)`;
**"softmax jointly normalized over all sources."**
Table 5, footnote 3 (Block): "Same ϕ and normalization as Full; `v_i = b_i`, `v_n^j = b_n^j`."

§5, the only hyperparameter statement: **"AttnRes introduces only one RMSNorm and one pseudo-query
vector `w_l ∈ R^d` per layer … Crucially, all pseudo-query vectors must be initialized to zero. This
ensures that the initial attention weights `α_{i→l}` are uniform across source layers, which reduces
AttnRes to an equal-weight average at the start of training."**

§3.2, the degenerate cases: **"`N = L` recovers Full AttnRes, while `N = 1` reduces to standard
residual connections with the embedding isolated as `b_0`."** And: "When `L` is not divisible by
`N`, the final partial sum is taken as the last block's representation."

### 2.3 Algorithm 1 — literal transcription (PDF p. 7)

```
Algorithm 1: Two-phase computation for block n
Input: Pseudo queries {w_l}_{l∈B_n}, block representations {b_0, …, b_{n-1}}
 /* Phase 1: Parallel inter-block attention */
 1  Q ← [w_l]_{l∈B_n}                                     // [S, d]
 2  K, V ← [b_0 ; … ; b_{n-1}]                            // [n, d]
 3  {o_l^(1), m_l^(1), l_l^(1)}_{l∈B_n} ← ATTN_WITH_STATS(Q, K, V)      // Return LSE
 /* Phase 2: Sequential intra-block attention + Online softmax merge */
 5  i ← 0
 6  for l ∈ B_n do
 7      if i = 0 then
 8          h_l ← o_l^(1) / l_l^(1)                      // Inter-block only
 9      else
10          o_l^(2), m_l^(2), l_l^(2) ← ATTN_WITH_STATS(w_l, b_n^i, b_n^i)     // Intra-block
11          m_l ← max(m_l^(1), m_l^(2))
12          h_l ← (e^{m_l^(1)-m_l} o_l^(1) + e^{m_l^(2)-m_l} o_l^(2))
               / (e^{m_l^(1)-m_l} l_l^(1) + e^{m_l^(2)-m_l} l_l^(2))   // Online softmax merge
13      i ← i + 1
14      b_n^i ← b_n^{i-1} + f_l(h_l)                       // Update partial sum; b_n^0 := 0
15  return {h_l}_{l∈B_n}
```

**Two ambiguities in Algorithm 1, and which reading the text supports:**

* **A1 — line 10 says `b_n^i`, Eq. 6 says `b_n^{i-1}`.** The text supports **`b_n^{i-1}`**. Line 14
  is the *update* for this layer and happens *after* line 10, so at line 10 the accumulator still
  holds the sum of the *previous* layers' outputs; and `b_n^0 := 0` (line 14) says the accumulator
  is zero before the first layer. Using `b_n^i` would make `h_l` depend on `f_l(h_l)` — circular.
  Eq. 6 is explicit ("the subsequent layers **additionally** attend to the partial sum
  `b_n^{i-1}`") and matches. **Our `two_phase_attend` uses the pre-update partial, i.e. the correct
  reading.**
* **A2 — line 3 says "Return LSE" but line 12's formula needs raw sums.** Line 12 is
  `h_l = (e^{m1-m}·o1 + e^{m2-m}·o2) / (e^{m1-m}·l1 + e^{m2-m}·l2)`, which is the standard
  online-softmax merge and requires `o` = *unnormalised* `Σ_j e^{s_j-m}·v_j` and `l` = *unnormalised*
  `Σ_j e^{s_j-m}`. Our reading (line 12 wins; "LSE" is loose prose) is the one that makes the merge
  correct.

### 2.4 The `b_0` / first-layer-of-block question — RESOLVED

The brief says the pseudocode and Eq. 6 disagree. **They do not**; the apparent disagreement is an
artefact of reading the first `block_attn_res` call's `partial_block` argument as belonging to the
*current* block.

Trace Fig. 2 in order, transformer layer `n`, sublayer 1, `blocks = [b_0,…,b_{n-2}]` and
`partial_block = b_{n-1}` (the previous block's completed sum, carried across calls):

* line 19 computes `V = stack(blocks + [partial_block]) = [b_0, …, b_{n-2}, b_{n-1}]` — exactly
  Eq. 6's `i = 1` row, **n elements, no current-block partial**.
* **then** line 23–25 fires the boundary check, pushing `b_{n-1}` into `blocks` and setting
  `partial_block = None`. `b_n^0 := 0`, matching Alg. 1 line 14.
* line 29 sets `partial_block = attn_out` (the `None` branch — no embedding is added).
* line 32 computes `V = [b_0, …, b_{n-1}, b_1^{n}]` = `[b_0,…,b_{n-1}, b_n^1]` — exactly Eq. 6's
  `i ≥ 2` row.

The **very first** transformer layer is the only special case: `blocks` is empty and
`partial_block = hidden_states = h_1 = b_0`, so `b_0` arrives through the *partial* argument
(line 9 `stack([] + [b_0])` → `V = [b_0]`, which is Eq. 6 with `n=1, i=1`), and the boundary check
then appends it to `blocks` — which is what the line-18 comment `"# blocks already include token
embedding"` refers to. From the second layer on that comment is literally true.

**The text is unambiguous and agrees with Eq. 6 on all three points:**

* Eq. 6, `i = 1` row: sources are `[b_0,…,b_{n-1}]` — **no current-block partial**.
* §3.2: "In each block, the first layer receives the previous block representations and the token
  embeddings, and the subsequent layers **additionally** attend to the partial sum `b_n^{i-1}`."
* Eq. 5 + Alg. 1 line 14: a block representation is `b_n = Σ_{j∈B_n} f_j(h_j)` — **sums layer outputs
  only, never the embedding** — with `b_n^0 := 0` and `b_0 = h_1` defined *separately* (§3.2).

**So: `b_0` is a first-class, permanently-attended source in its own right, and is never summed into
any `b_n`.** Fig. 8 confirms it empirically: *"The embedding `h_1` retains non-trivial weight
throughout, especially in pre-attention layers"* — a separate row/column in the heatmap, not folded
into block 0's cell. **Our streaming `BlockAttnRes` violates this** (D5 below).

---

## 3. Delta table

`file:line` are in `vendor/burn-fused/crates/burn-attnres/`.
**BUG** = our output is a different function from the paper's on ordinary inputs.
**BENIGN** = different spelling, same function.
**PAPER-AMBIGUITY** = the paper contradicts itself; we picked the reading the text supports.
**CONTRACT** = the API permits a silent misuse the paper forbids.
**MATCH** = checked, correct — listed because "we got this right" is also a finding.

| # | `file:line` | paper says | we do | class |
|---|---|---|---|---|
| **D1** | `lib.rs:109`, `:155`; `lib.rs:180`; `lib.rs:410`, `:428`; `fused_attnres.rs:111`, `:164`; `:313`, `:383`, `:390`, `:438`, `:440`, `:521`; `:554`, `:603`; `:689`, `:711`; `:756`, `:765`; `:1142`, `:1165`; backward-tensor `:1348`, `:1356`, `:1372`, `:1376` | `logits = einsum(q, RMSNorm(k))`. **No `1/√d` anywhere.** Eq. 2 defines `ϕ(q,k) = exp qᵀ RMSNorm(k)`; Table 5 fn 2 repeats it; Fig. 2 line 11 is a bare einsum; Alg. 1 line 3 is `ATTN_WITH_STATS(Q,K,V)`. I grepped the whole 1439-line extraction for `sqrt`/`scale`/`temperature` — the only hits are "scaled residual paths" (a cited *different* method, ref [54]), "attention temperature rescaling" (about MLA/NoPE context extension, §5.2), and `d_model`/dimension prose. **There is no scaling factor in this paper.** | `let scale = (d as f64).powf(-0.5);` then `scores … .mul_scalar(scale)`, and the same `scale` threaded into all three kernels and both backward paths. | **BUG** — the known one, confirmed. *Nuance that matters:* because `w_l` is a free learnable vector, `softmax(d^{-1/2}·w_lᵀk)` is reachable by the paper's function class with `w_l` rescaled by `√d`, so this does **not** change what the model can express at convergence — it changes the **optimisation trajectory and the effective softmax temperature**, and at the mandated `w_l = 0` init both give uniform α so init is unaffected. It is still a silent, undocumented divergence from the only spec, and it confounds any A/B against a paper-faithful baseline. |
| **D2** | `lib.rs:150`–`:152`, `:171`–`:177`, `:420`–`:425`, `:443`–`:449`; `fused_attnres.rs:164`, `:383`, `:390`, `:432`, `:603`, `:852`, `:898`, `:1351`, `:1495`, `:1651` | `RMSNorm` (paper cites Zhang & Sennrich, ref [66]): `x / √((1/d)Σx² + ε)`. The `1/d` is what makes it an **R**MS norm. | `x / √(Σx² + 1e-5)` — no `1/d`, ε inside the square root. This is an L2 normalisation (times `√d` relative to RMSNorm) with a different ε placement. | **BUG** — it is not the named function. |
| **D3** | (consequence of D1+D2) | logit = `w·h / √(Σh²/d + ε)` | logit = `w·h / (√d · √(Σh² + 1e-5))` → for `‖h‖²` not tiny, **our logit ≈ the paper's logit / d** | **BUG (observable)** — a `d`-fold temperature compression. For `d = 4096` that is a very flat softmax; the paper's `w_l` must grow ~`√d` more than ours to reach the same sharpness. Stated separately from D1/D2 because this is the number a reader would want. |
| **D4** | `lib.rs:450` vs `lib.rs:428` | Alg. 1 line 12 merges Phase-1 (inter-block) and Phase-2 (intra-block) statistics with **one shared scale** — Alg. 1 never introduces a scale in either phase. | Phase 1 inter-block logits get `.mul_scalar(scale)` (`:428`); **Phase 2's intra-block logit at `:450` gets no scale at all** — `let s2 = (q_i * p_norm).sum_dim(1);` and `s2` is fed straight into the merge at `:466`. | **BUG**, and **independent of D1** — it exists even if you decide to keep the `1/√d`. The two source *groups* being merged are compared at temperatures differing by `√d`, so the inter-block/intra-block relative weighting produced by the merge is wrong. This one is invisible to the existing tests (`two_phase_merge_matches_full_attention` only exercises `i = 0`, which bypasses the merge entirely at `:459`). |
| **D5** | `lib.rs:270`–`:304` (`BlockAttnRes::step`), `:80`–`:94` (`forward`) | Eq. 5 `b_n = Σ_{j∈B_n} f_j(h_j)` (layer **outputs** only); Eq. 6 `b_0 = h_1` is a **separate, permanently attended source**; Alg. 1 line 14 `b_n^0 := 0`; Fig. 8 shows persistent separate weight on the embedding. | `st.partial` is seeded with the caller's first `h` (`:272`–`:276`), and the first block's incorporated representation is therefore `b_0 + Σf` — the embedding is **summed into block 1 and thereafter is not a source of its own**. | **BUG.** Concretely, with `S = 2` and the embedding `b_0` as the first `h`: `step 1` → `out = b_0` ✓; `step 2` → `incorporate(b_0 + f_1)`, so the persistent state holds **one** source where the paper holds **two** (`b_0` and `b_1^1 = f_1`). Every subsequent block-boundary softmax therefore ranks `N−1` sources, not `N`, and the one source the paper's own analysis singles out is gone. |
| **D6** | `lib.rs:282`–`:286` | Eq. 6 `i = 1`: `V = [b_0,…,b_{n-1}]ᵀ`, softmax over all of them. §3.2: "In each block, the first layer receives the previous block representations and the token embeddings." | `let mut out = if st.started \|\| st.partial_count >= 2 { st.attended() } else { h.clone() };` — on the first sublayer of every block, `out` is whatever the online state already holds. | **BUG.** `st.attended()` is `acc / sum_exp` over the stored sources, so at a block boundary it returns the **previous block's representation unchanged** — with one source in the state (D5) that softmax is the identity, so `step` degenerates to a **standard residual connection at every block boundary**. The mechanism is only active for the *second and later* sublayers of a block, where the partial is merged in. |
| **D7** | `lib.rs:292`–`:294` | Eq. 6 `i ≥ 2` for `n = 1`: `V = [b_0, b_1^1]ᵀ`, `h = softmax(α)·V`. | `} else { out = st.partial.clone(); }` — on the second sublayer of the **first** block (`!st.started`), `out` is the raw accumulated sum `b_0 + f_1`. | **BUG.** The function returns an **unnormalised two-term sum** where the paper returns a softmax-weighted mixture. No normalisation, no keys, no `w_l` involved. It is the one case where the crate returns something that is not attention at all, and the case that is *least* likely to be caught because `n = 1` is the trivial-looking start of the network. |
| **D8** | `lib.rs:288`–`:302` | Alg. 1 keeps the attended-source set and the output computation separate; a source enters the set **once**. | At `partial_count == block_size` the code first calls `st.merge_source(&p, &s_p)` (`:291`, *output* computation) which mutates the persistent `acc`/`max_score`/`sum_exp` in place, and then calls `st.incorporate(&query, &completed)` (`:299`) on the **same tensor with the same score**, which merges it a **second time** into the same persistent state. | **BUG — state double-count.** After any block boundary the persistent state carries the just-completed block with weight ~2 and `sum_exp` inflated by 1, so **the step after a boundary is computed from a wrong source set**. *Read from source, not executed* (no GPU permitted here); §5's test G below is the one-line gate. |
| **D9** | `lib.rs:297`–`:302` | §3.2: "When `L` is not divisible by `N`, **the final partial sum is taken as the last block's representation**." | The tail block's partial is merged for the current step's output but is never `incorporate`d, so it never becomes a source. | **BUG (minor)** — the last `L mod S` sublayers of a network are never represented as a block. |
| **D10** | `lib.rs:26`–`:51`, `:103`–`:161` | Eq. 3 / Table 5 fn 2: `v_0 = h_1`, the token embedding is source 0. | `depth_attend(history, query)` treats `history[0]` as an arbitrary tensor. No doc line, no assert, no test says element 0 must be the embedding. `full_attnres_module` (`:343`) passes three `random` tensors. | **CONTRACT** — a caller that passes only layer outputs silently drops `v_0`. Given D5/D6/D7, the streaming path already drops it; the full path drops it only if the caller does. The one source the paper's Fig. 8 singles out as retaining persistent weight is the one with no enforcement. |
| **D11** | `lib.rs:439`–`:476` | Alg. 1 line 10 prints `b_n^i`; Eq. 6 says `b_n^{i-1}`; line 14 updates *after*. | `partial` at `:450` is the pre-update accumulator, i.e. `b_n^{i-1}`. | **PAPER-AMBIGUITY → we chose right** (matches Eq. 6 and is the only non-circular reading). |
| **D12** | `lib.rs:430`–`:435`, `:466`–`:471` | Alg. 1 line 3 "Return LSE" vs line 12's merge formula. | `m1 = max(scores)`, `l1 = Σ e^{s−m1}` (raw), `o1 = Σ e^{s−m1}·v` (raw) — the reading line 12 requires. | **PAPER-AMBIGUITY → we chose right** ("LSE" treated as loose prose). |
| **D13** | `lib.rs:34`–`:36`, `:58`–`:63` | One `w_l ∈ R^d` **per layer** (per sublayer); §2 counts attention and MLP as separate layers, so a transformer layer has **two** (`attn_res_proj`, `mlp_res_proj`, Fig. 2 lines 19 and 32). §5.2's model: 6 sublayers per block, 9 blocks + embedding = 10 sources. | Each module instance owns exactly one `query: Param<Tensor<1>>`, with no layer index. | **BENIGN / API shape** — one instance per sublayer reproduces the paper, but nothing in the crate states the per-layer requirement, and `BlockAttnRes::new(d, block_size)` takes a single vector for a whole block, which is *not* the paper's parameterisation. Footgun, not a formula error. |
| **D14** | `lib.rs:39`–`:45`, `:66`–`:71` | Fig. 2 line 11 `proj.weight.squeeze()`; §5 "**all** pseudo-query vectors **must** be initialized to zero. This ensures that the initial attention weights α are uniform … and prevents training volatility". | `Initializer::Zeros.init([d_model], device)` in both constructors. | **MATCH** — and the only place a third-party port (`nktkt`) gets it wrong. No gate prevents a caller from building a randomly-initialised query, but the constructors are right. |
| **D15** | `lib.rs:150`–`:159`; `:426`–`:435`; Fig. 2 lines 10–12 | `K = norm(V)`; `h = einsum(α, V)` — keys normalised, **values raw**. | Scores from `h_norm`; aggregation over the un-normalised `h_stack`/`kv`. | **MATCH.** |
| **D16** | `lib.rs:156`; `fused_attnres.rs:216`–`:217`, `:388`–`:394` | Table 5 fn 2: "softmax **jointly** normalized over all sources." | `activation::softmax(scores, 0)` over the stacked source axis; the online merge folds groups into one denominator. | **MATCH.** |
| **D17** | `lib.rs:238`–`:246`; `fused_attnres.rs:231`–`:233`, `:637`–`:651` | Alg. 1 line 11–12: `m' = max(m1,m2)`, `acc' = acc·e^{m−m'} + e^{m2−m'}·o2`, `sum' = …`, `h = acc'/sum'`. | Identical algebra, in both the tensor and the fused kernel. | **MATCH** (modulo D4's missing scale on the intra-block leg, and the `clamp_min(1e-12)` guards, which are numerical hygiene, not spec). |
| **D18** | `lib.rs:104`–`:107`; `:83`–`:85` | `softmax` over a single source returns that source. | `if n == 1 { return history[0].clone(); }` | **MATCH** (short-circuit, mathematically identical). |
| **D19** | `lib.rs:103`–`:107` | — | `history[0].dims()` is reached for `n == 0` → index panic, not a loud error with a cause and an escape (ADR-0011). | **not a paper delta**, listed for completeness; 1 line, CPU-reachable. |

**Counts.** 19 rows. **9 BUG** (D1, D2, D3, D4, D5, D6, D7, D8, D9 — of which D3 is a consequence of
D1+D2 and D4 is independent). 1 CONTRACT (D10). 3 BENIGN/ambiguity (D11, D12, D13). 1 out-of-spec
(D19). **5 MATCH** (D14–D18). Clustering: **D1/D2/D3 are one deviation expressed in six places**;
**D5/D6/D7/D8 are one deviation (the streaming Block path is not Eq. 6) expressed in four places**;
**D4 is its own bug**; **D9 is minor**.

### 3.1 The existing tests cannot see any of this — by construction

Every reference in `fused_attnres.rs`'s test module is a re-transcription of **our own** formula:

* `ref_depth_attend` (`:847`–`:861`) has `let scale = (d as f64).powf(-0.5);` at `:850` and applies
  it at `:855`; `depth_attend_fused_matches_tensor` (`:864`) compares the fused kernel against it.
* `source_score_fused_matches_tensor` (`:887`) vs a reference with `powf(-0.5)` at `:892`.
* `merge_state_writeback_matches_host_reference` (`:1016`) is a **host** reference — good practice, but
  it re-derives the same online-softmax algebra, and its own doc comment (`:1012`–`:1015`) already
  admits it *"does NOT reliably catch the missing-barrier race … that race is timing-dependent, and
  this test passes with the barrier deleted (measured 2026-09-29)"*. That admission is correct and is
  the right level of honesty.
* `depth_attend_grad_matches_finite_difference` (`:1720`) differentiates **our** function, so it is
  green with or without the scale. Tolerance is `rel < 1e-1` — loose.
* `streaming_matches_full_recompute` (`lib.rs:362`) compares `BlockAttnRes::step` against
  `BlockAttnRes::forward`, which calls **the same `step`** (`:91`) on a fresh state. It is a
  self-consistency check; it cannot see D5–D8, all of which are properties of the shared `step`.

This is precisely the situation ADR-0020 calls kind (d). The crate README already withdraws the word
"verified" from the backward check for exactly this reason — the same reasoning extends to every
other comparison in the file.

### 3.2 The data race: current state — **FIXED, and the doc comments now agree with the code**

The reviewer's note ("the file's own doc comment claimed the opposite of what the code did") is
**stale as of this checkout**.

| site | barrier | write | doc comment |
|---|---|---|---|
| `merge_kernel` | `sync_cube();` at `fused_attnres.rs:658` | `if tid == 0 { max_s[bt] = …; sum_exp[bt] = …; }` at `:659`–`:662` | `:654`–`:657` — *"read by every thread above … and written by `tid == 0` below. Without this barrier a slower warp can reach the read after tid 0's write landed, silently computing rescale = 1."* **Matches the code.** |
| `attnres_chunk_kernel` | `sync_cube();` at `fused_attnres.rs:271` | `if tid == 0 { max_s[bt] = …; sum_e[bt] = …; }` at `:272`–`:275` | `:268`–`:270` — *"Same barrier requirement as `merge_kernel` … only `tid == 0` writes them below."* **Matches the code.** |
| `attnres_scores_kernel` | `sync_cube();` at `:150` between reading `red[0]` and writing `red[tid]`; `sync_cube();` at `:141`, `:147` | — | — |
| `depth_attend_backward_kernel` | `sync_cube();` at `:359`, `:377`, `:419` after each `…[li] = red[0]` | — | — |

**Both barriers are present, both comments describe them, and the residual-state read→write hazard is
genuinely closed in the current tree.** The `#[ignore]`d `source_score_noncontiguous_matches_tensor`
(`:986`–`:999`) and the honest scope note on `merge_state_writeback_matches_host_reference` are both
appropriately labelled as claims-to-be-falsified rather than green tests.

### 3.3 Severity, in context

`docs/architecture/library-crate-fate.md:62` records `burn-attnres` as fate class **b / REFERENCE**, and
`crates/burn-attnres/README.md:3`–`:8` states it is **not in the dormouse build** (no incoming edges
from `dormouse-{core,data,train,cli}`; the residual-stream A/B named at `PLAN-minimal-core.md` §M2
has never been run). So **none of the nine bugs has ever executed inside a training run**, and no
retraction of a dormouse number follows from this audit. The cost of leaving them is that the crate
is 2268 lines of "reference port" that is not a port.

---

## 4. Everything the paper defines that we do not implement **at all**

This is the list I was asked not to skip. Ordered by how much it would change an A/B.

**Architecture / mechanism (not present anywhere in the crate):**

1. **The Fig. 2 `forward` loop.** Two AttnRes points per transformer layer — `attn_res_proj`/`attn_res_norm`
   before attention, `mlp_res_proj`/`mlp_res_norm` before the MLP — plus the boundary bookkeeping and
   the `partial_block = None` reset. There is no sublayer loop anywhere in `burn-attnres`; the crate is
   5 free functions and 2 structs. This is the *paper's* model, and none of it is here.
2. **`b_0 = h_1` as a first-class, permanently-attended source** (Eq. 6, §3.2, Table 5 fn 2, Fig. 8).
   Handled only if a caller remembers to pass it, unenforced (D10), and actively destroyed by the
   streaming path (D5).
3. **The full source set `[b_0 … b_{n-1}, b_n^{i-1}]` for the Block variant.** `two_phase_attend`
   implements this shape; `BlockAttnRes::step` does not (D5–D7). Two functions in one crate implement
   two different block mechanisms.
4. **`ATTN_WITH_STATS` as a named primitive** returning `(o, m, l)`. Implicit in `two_phase_attend`; no
   seam, so D4 (the asymmetric scale) is invisible to review.

**Ablation arms the paper defines, none of which exist as options (§5.3, Table 4):**

5. **Input-dependent query** — project `q_l` from the current hidden state instead of learning it.
   Loss **1.731 vs 1.737**: the paper's *best* variant, and the one it declines only because it costs a
   `d×d` projection per layer. Our `query: Param<Tensor<1>>` cannot express it; no flag, no constructor.
6. **Input-independent learned scalar mixing** (1.749) — query and key removed, learnable scalars instead.
7. **Sigmoid kernel instead of softmax** (1.741) — the paper attributes softmax's value to
   "competitive normalization, which forces sharper selection among sources".
8. **Multihead depth aggregation, `H = 16`** (1.752) — per-channel-group depth attention. Our score is
   a single scalar per (source, token); there is no head dimension anywhere in the crate. The paper's
   finding is that this *hurts* (1.752 vs 1.746), so its absence is defensible — but it is a deliberate
   omission that is not recorded anywhere.
9. **SWA, `W = 1 + 8`** (1.764) — sliding window over the `W` most recent layer outputs plus the embedding.
10. **`w/o RMSNorm`** (1.743 Full / 1.750 Block) — the ablation that justifies D2's norm existing at all.
    No flag to turn it off, so the claim "RMSNorm matters" is untestable with this crate.

**Systems contributions of §4, none implemented:**

11. **Cross-stage caching for pipeline parallelism** — Eq. 7 (`Comm_naive = C(C−1)/2 · N_p d`),
    Eq. 8 (`Comm_cached = P(P−1)/2 · N_p d + (V−1)P²N_p d`), Fig. 3. Explicitly a headline
    contribution ("Infrastructure for scale") and the thing that makes Block AttnRes drop-in.
12. **Sequence-sharded prefilling** (§4.2) — shard the `N·T·d` block-rep store along sequence across
    `P` TP devices so Phase 1 runs on local shards; merge into the TP all-reduce path (reduce-scatter
    → local merge → all-gather, fusible with RMSNorm); chunked prefill. Paper's numbers: 15 GB → 1.9 GB
    per device at 128K/8 blocks; <0.3 GB with 16K chunks.
13. **The two-phase schedule for *Full* AttnRes** (Appendix B, Eqs. 11–17): per-layer I/O
    `(S+N−2)d` read + `2d` write, total `(S+N)d`, versus `O(Ld)` naive. `two_phase_attend` implements
    the **Block** variant's Algorithm 1 only. Full AttnRes gets one monolithic pass
    (`depth_attend`, `CHUNK_G = 8` chunking) — which is the naive schedule with a memory bound bolted on.

**Parameterisation / structural claims with no code or test behind them:**

14. **`S = L/N` and the `L mod N ≠ 0` tail rule** (§3.2) — `BlockAttnRes` takes `block_size` directly,
    asserts nothing, and drops the tail (D9).
15. **The two degenerate endpoints**: `N = L` ≡ Full AttnRes and `N = 1` ≡ "standard residual
    connections with the embedding isolated as `b_0`" (§3.2); Fig. 6 sweeps `S = 1…32` and shows
    `S = 1` reproduces Full AttnRes at 1.737. **Neither endpoint is pinned by any test.** The `N = 1`
    identity is the cheapest possible sanity check on the whole crate and it does not exist.
16. **The `N ≈ 8` operating point** (§3.2, §5.3: "*we fix the number of blocks to ≈ 8 for
    infrastructure efficiency*"; §5.2's 48B model uses 6 sublayers/block → 9 blocks + embedding = 10
    sources). Nothing records that `block_size` should be `L/8`, and `CHUNK_G = 8` in
    `fused_attnres.rs:23` is a *different* 8 (a CUDA chunk width, commented "*paper's N ≈ 8*" — the
    comment conflates the memory-chunk constant with the architectural block count, which are unrelated).

**Analysis surfaces the paper defines, with no corresponding API:**

17. **Readable `α`.** `depth_attend` returns only the aggregated output. There is no way to obtain the
    weight vector, so **Fig. 8's weight heatmaps, Fig. 5's per-block output/gradient magnitudes, and
    the depth-attention-sink observation (§6.2 "Practicality") are all unreproducible from this crate.**
    `two_phase_attend` computes `l1`/`m1` internally and discards them.
18. **The structured-matrix view** (§6.2, Fig. 9): depth-mixing matrix `M`, and the rank claim that
    Block AttnRes's effective rank lies between `N` and `N+S`. An analysis, not code — but it implies a
    cheap check (rank of `M` recovered from `α`) that would catch D5/D6 immediately, and no such check exists.

---

## 5. Recommended gold-vector test

Design goal: **fail if any of D1–D9 is a bug**, and fail *loudly* rather than by tolerance drift.
No GPU needed for A–D, G, H; E and F need CUDA.

### The reference

Hand-write it once, in **f64**, with no burn and no `1/√d` and a **true RMSNorm** — a direct
transcription of §2.2. Keep it in the test file as literal numbers, not as a call into the crate:

```rust
// Transcribed from arXiv:2603.15031 Eq. 2-6 + Fig. 2 line 11. NO 1/sqrt(d).
// RMSNorm per Zhang & Sennrich [66]: x / sqrt(mean(x^2) + eps).
fn ref_attnres(sources: &[Vec<f64>], d: usize, w: &[f64], eps: f64) -> Vec<f64> {
    let n = sources.len();
    let logits: Vec<f64> = sources.iter().map(|v| {
        let ms: f64 = v.iter().map(|x| x * x).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        w.iter().zip(v).map(|(a, b)| a * b * inv).sum()   // q . RMSNorm(k)
    }).collect();
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z: Vec<f64> = logits.iter().map(|s| (s - m).exp()).collect();
    let sum: f64 = z.iter().sum();
    (0..d).map(|j| sources.iter().enumerate()
        .map(|(i, v)| z[i] / sum * v[j]).sum()).collect()
}
```

**Write the *expected* outputs as literals**, not as calls to `ref_attnres`. The function above is the
transcription; the literals are the oracle. A transcription bug then cannot hide behind itself.

### Test A — the single-vector scale/norm probe. **Catches D1, D2, D3. No tolerance argument needed.**

```
B = T = 1, L = 2, d = 4
h_0 = [1, 0, 0, 0]
h_1 = [0, 1, 0, 0]
w   = [2, 0, 0, 0]
```

Three conventions give three exactly-separable answers on the first output component:

| implementation | score_0 | α_0 | out[0] |
|---|---|---|---|
| **paper** (`q·RMSNorm(k)`, no scale) | 4.000002 | 0.9820138 | **0.9820138** |
| our norm, our scale removed | 1.999990 | 0.8807971 | 0.8807971 |
| our norm, our scale kept (**what we ship**) | 0.999995 | 0.7310586 | 0.7310586 |

```rust
assert!((out[0] - 0.982_013_8).abs() < 1e-5, "out[0] = {got}");
assert!((out[1] - 0.017_986_2).abs() < 1e-5, "out[1] = {got}");
```

Separation is **≥ 0.10** — four orders of magnitude above a `1e-5` tolerance. The test is
self-certifying: anyone who changes the scale or the norm convention sees which of the three numbers
they now produce. **This test is red today and should stay red until the deviation is either removed
or written down as an ADR.**

### Test B — the uniform-average init invariant. **Catches D10, and any key/value swap. Scale-blind by construction.**

§5: zero-initialised `w_l` ⇒ uniform `α` ⇒ "an equal-weight average at the start of training".
With `L = 5` random `[B,T,D]` histories and `q = 0`, `depth_attend` must equal the arithmetic mean of
the history to `1e-6`:

```rust
let mean = history.iter().fold(zeros, |a, h| a + h) / 5.0;
assert!(maxdiff(out, mean) < 1e-6);
```

This is the one test that pins "**V is un-normalised**": an implementation that mistakenly weighted by
`h_norm` returns the mean of *normalised* states and fails by O(1). Cheap, and it is the paper's own
stated starting point, so it doubles as a check that `Initializer::Zeros` is still in place.

### Test C — the Eq. 6 source-set walk. **Catches D5, D6, D7. This is the headline test.**

`d = 4`, `w = [2, 0, 0, 0]`, `S = 2`, and a `b_0` that is **not** a sum of any `f` so that folding it
in is arithmetically visible:

```
b_0 = [3, 0, 0, 0]   f_1 = [0,1,0,0]   f_2 = [0,0,1,0]   f_3 = [0,0,0,1]   f_4 = [0,1,1,1]
```

Feed `BlockAttnRes::step` these five in order and assert the **exact source set** the paper prescribes
at every step, by comparing against `ref_attnres` applied to that set:

| step | Eq. 6 sources | expected out (d=4, w=[2,0,0,0]) | ours today |
|---|---|---|---|
| 1 | `[b_0]` | `[3, 0, 0, 0]` | `[3,0,0,0]` ✓ |
| 2 | `[b_0, f_1]` | `[2.9460414, 0.0179862, 0, 0]` | **`[3,1,0,0]`** ✗ (D7) |
| 3 | `[b_0, b_1]`, `b_1 = f_1+f_2` | `[2.9460414, 0.0179862, 0.0179862, 0]` | **`[3,1,0,0]`** ✗ (D5+D6) |
| 4 | `[b_0, b_1, f_3]` | see literal table in the test file | wrong set ✗ |
| 5 | `[b_0, b_1, b_2]`, `b_2 = f_3+f_4` | " | wrong set ✗ (also D8) |

Tolerance `1e-5` absolute. Steps 2 and 3 are ~`1e-1` and ~`1.4e-1` away today, so the margin is
unambiguous. Name the violated Eq. in the assert message so a red run points at the equation, not the
line.

### Test D — two-phase scale consistency. **Catches D4. Needs no GPU.**

`two_phase_attend` with `S = 2`, `N = 2`, `d = 8`, one fixed `partial` and a fixed `blocks` matrix.
Assert Phase 2's `s2` and Phase 1's `scores` are computed on the **same** scale by comparing the merged
output to an f64 transcription of Algorithm 1 with both legs unscaled. A `1e-3` mismatch on the merged
weights separates the two conventions at `d = 8` (the scale is `0.354`). Also assert `l2 == 1.0` and
`m2 == s2` explicitly — those are the single-key identities from Alg. 1 line 10, and today they hold only
by accident of the missing scale.

### Test E — re-point the existing fused/tensor parity tests at the paper reference

`ref_depth_attend` (`fused_attnres.rs:847`) is the reason D1 is invisible. Replace its
`powf(-0.5)` at `:850` and its `mul_scalar(scale)` at `:855` with the paper formula, and
`depth_attend_fused_matches_tensor` (`:864`) immediately becomes a real check on both the tensor path
and the kernel. Same for `:892`/`:903` in `source_score_fused_matches_tensor` and the in-file
`raw_depth_attend` at `:1645`/`:1654`. Tolerance `1e-4` as today. **Until this is done, the crate has no
CUDA test that can fail on a paper divergence.**

### Test F — re-point the gradient test at the paper reference, and tighten it

`depth_attend_grad_matches_finite_difference` (`:1720`) differentiates *our* function, so it is
scale-blind; its `rel < 1e-1` is also loose for an `f32` analytic gradient against an `f64`
finite difference. Re-point at the paper reference: `rel < 1e-2` against central differences with
`eps = 1e-3`, and `rel < 1e-3` against the `f64` analytic Jacobian. The current `rel < 1e-1` would not
notice a missing `1/√d` factor in the gradient alone.

### Test G — the online state must not double-count. **Catches D8. Cheapest real gate in the set.**

No paper reference needed; the invariant is arithmetic. After each `BlockAttnRes::step`, with a known
`blocks` list, `st.sum_exp` (a `[B,T,1]` tensor) must equal the number of **distinct** sources in the
state, because the paper's `ℓ` is `Σ_j e^{s_j − m}`:

```rust
// S = 2, b = t = 1: after step 2 (the boundary) the state holds 2 sources, not 3.
assert!((sum_exp_at(0, 0) - 2.0).abs() < 1e-6, "state holds {got} weights, expected 2");
```

Today it reads `3.0`. This is one line, needs no external reference, and is the direct assertion that
"the output computation and the state update are the same fields" (D8) is wrong. Pair it with
`maxdiff(st.acc, expected_acc_over({b_0, f_1}))`.

### Test H — the two degenerate endpoints, unpinned until now

* `BlockAttnRes` with `S = 1` must equal `depth_attend` over the same sources (§3.2, `N = L`; Fig. 6,
  `S = 1` → 1.737 = Full AttnRes).
* `BlockAttnRes` with `N = 1` (one block) must equal **standard residual accumulation with the
  embedding isolated as `b_0`** (§3.2) — i.e. the output is the plain sum, and the `b_0` source is
  never merged into a block. This is the smallest possible statement of the whole mechanism, it needs
  no softmax reference, and **it is red today** (D5: our first block is `b_0 + Σf`).
* Trailing partial: `L = 5, S = 2` — the last block's partial must become a source (§3.2, "the final
  partial sum is taken as the last block's representation"). Catches D9.

### Tolerances, in one place

| comparison | tolerance | why |
|---|---|---|
| f32 kernel/tensor vs f64 paper reference | `1e-5` abs on values `O(1)`; `1e-4` on raw scores | f32 has ~7 digits; the code already asserts `1e-4` and passes on self-consistency |
| `q = 0` uniform-average invariant | `1e-6` | pure mean of f32 values, no transcendentals |
| two-phase merge weights | `1e-3` | one `exp` and two f32 reductions |
| analytic gradient vs f64 Jacobian | `1e-3` rel | — |
| analytic gradient vs central difference, `eps = 1e-3` | `1e-2` rel | f32 truncation floor; the current `1e-1` is too loose to gate anything |
| `sum_exp` source-count invariant | `1e-6` | exact integer count in f32 |

**One rule for all of them:** the expected values are **literals transcribed from the paper**, or
`1/L`/integer counts. No expected value may be produced by calling another function in the same crate.
That is the single change that turns this suite from kind (d) into something that can fail on a
transcription error.

---

## VERIFIED (sourced)

* The paper is **arXiv:2603.15031**, Kimi Team, v1 2026-03-16; PDF read in full (`pdftotext -layout`,
  1439 lines, 9 pages). Fetched 2026-09-29.
* **No original source code exists.** `MoonshotAI/Attention-Residuals` `master` = 7 blobs
  (1 PDF, 1 README, 4 PNGs), 1 branch, 0 tags, 0 releases, 0 PyPI. Verified via the GitHub trees API.
* Fig. 2 is 22 lines of PyTorch, reproduced in §2.1, identical in the paper and the authors' README.
* **`logits` has no `1/√d` in the paper.** Eq. 2 (`ϕ(q,k) = exp qᵀ RMSNorm(k)`), Table 5 fn 2, Fig. 2
  line 11, Alg. 1 line 3. A full-text grep for `sqrt`/`scale`/`temperature` returns no scaling factor
  anywhere. Corroborated by two independent transcriptions (`nktkt`, PyTorch RFC #177537).
* `RMSNorm` is the `1/d`-mean version (paper cites Zhang & Sennrich [66]); ours is a `√d`-larger L2
  normalise. Combined with the scale, our logit ≈ the paper's `/d`.
* **`b_0` / first-layer-of-block: the pseudocode and Eq. 6 agree**; the text supports "first layer of a
  block attends to `[b_0…b_{n-1}]` only, with no current-block partial, and `b_0 = h_1` is a separate
  source never summed into any `b_n`" (§2.4, three independent textual supports).
* Algorithm 1 is internally inconsistent about `b_n^i` vs `b_n^{i-1}` (line 10 vs Eq. 6); Eq. 6 is
  correct and is what we implemented.
* The `sync_cube()` race in `merge_kernel` / `attnres_chunk_kernel` is **fixed** in the current tree
  (`fused_attnres.rs:658`, `:271`) and **both doc comments now describe what the code does** — the
  reviewer's note is stale.
* Five formulas match the paper exactly: keys normalised / values raw (D15), joint softmax over all
  sources (D16), the Alg. 1 line-12 online merge algebra (D17), zero-init of `w_l` (D14), the `n == 1`
  short-circuit (D18).
* `burn-attnres` is **not in the dormouse build** (`crates/burn-attnres/README.md:3`–`:8`,
  `docs/architecture/library-crate-fate.md:62`, class **b / REFERENCE**), so none of the nine bugs has ever run in
  training and **no dormouse number is retracted by this audit**.

## SPECULATION (my inference)

* **D8 (double-incorporate) is read from source, not executed.** I traced
  `lib.rs:288`–`:302` and `merge_source`'s in-place mutation of `acc`/`max_score`/`sum_exp`, and the
  arithmetic is unambiguous: at a block boundary the same tensor is merged twice into the same state.
  But I could not run the code, and `streaming_matches_full_recompute` (`lib.rs:362`) *should* already
  be red for the same reason — I do not know its current status. Test G is designed to settle it in
  one line on a free GPU.
* **Whether the `1/√d` was intentional.** Nothing in the crate, its README, its git history, or
  `docs/` says where it came from; the doc comment at `lib.rs:28`–`:32` documents it as
  `query · norm(h_i) / sqrt(d)`, i.e. it is documented as intended, but never justified against the
  paper. I cannot tell a deliberate deviation from a habit borrowed from softmax attention. It must be
  either removed or written down — and the fact that the doc comment presents it as the definition of
  the mechanism, with no paper citation, is the kind of thing ADR-0020 exists for.
* **Severity ranking assumes a future A/B on the residual stream** (`PLAN-minimal-core.md` §M2). If the
  crate is deleted per `docs/architecture/library-crate-fate.md`, D1–D9 cost nothing and §4's list is moot. If it is
  kept as the AttnRes arm, D5–D8 mean the arm measures a mechanism that is **not AttnRes** — it is a
  plain residual at block boundaries with a partial-sum mix inside blocks — so an A/B run today would
  be charging a wrong mechanism's numbers to the paper.
* **The `kyegomez` and `wdlctc` ports were not read.** I list them as existing, not as agreeing. The
  three I did check (`nktkt`, the PyTorch RFC, and the paper itself) are enough for the `1/√d` verdict;
  a full cross-port audit is not needed and I did not do one.

## Open questions for the next agent

1. Is the `1/√d` a deliberate decision? If yes, it needs an ADR that states the temperature argument;
   if no, delete it and re-point `ref_depth_attend` (`fused_attnres.rs:847`) at the same commit, or the
   deletion is invisible to CI.
2. Does `BlockAttnRes::step` have any caller that depends on the D5/D6/D7 semantics? It has none in
   this repo, but the crate is published (`crates.io/crates/burn-attnres`, per its README badge) and the
   git history was not walked for external users.
3. `two_phase_attend` is the only function that tracks Eq. 6 + Alg. 1 reasonably closely, and it is a
   free function with a `[S,d] × [N,d]` signature that does not compose with the crate's `[B,T,D]`
   streaming API. Which of the two is meant to survive?
4. Should the crate expose `α`? Without it, none of Fig. 5 / Fig. 8 / §6.2 is reproducible, and the
   rank-of-`M` check would have caught D5 and D6 for free.
