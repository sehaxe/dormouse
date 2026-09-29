# Kimi K3 (arXiv 2607.24653) — §2.1.1 and the delta-network / state formulation

Transcription only. Every quote is verbatim from the source, with a location.
Nothing in this file is inferred, paraphrased, or reconstructed.

---

## 1. Provenance

| field | value |
|---|---|
| arXiv id | `2607.24653` — **resolves** |
| title | *Kimi K3: Open Frontier Intelligence* |
| latest version | **v2**, published 2026-07-27, updated 2026-08-07 (`arxiv:comment` = "K3 tech report", primary category `cs.CL`) |
| id verification | `https://export.arxiv.org/api/query?id_list=2607.24653` → `totalResults 1`, entry `http://arxiv.org/abs/2607.24653v2` (fetched 2026-09-29) |
| primary URL read | `https://arxiv.org/html/2607.24653v2` — HTTP 200, 1 520 019 bytes |
| secondary URL read | `https://arxiv.org/html/2607.24653v1` — HTTP 200, 1 366 070 bytes |
| cross-check | `https://arxiv.org/pdf/2607.24653v2` → `pdftotext -layout`, 3094 lines |
| how read | HTML → text: stripped `script/style/svg`, promoted each `<math alttext="…">` to `$…$`, `</td>`→TAB, block ends→newline, collapsed whitespace. Line numbers below are **extracted-text line numbers in `k3_v2.txt`**, 5604 lines total. v1 extracted to 5578 lines. |
| fetch date | 2026-09-29 |
| scope of this file | §2.1.1 "Kimi Delta Attention", extracted lines **447–551** (v2) |

**Citation-numbering caveat — read this before citing a reference number.**
The HTML build and the PDF disagree on *every* in-text citation number, and the
HTML's own rendered bibliography is offset from its in-text markers (`id="bib.bib13"`
renders as "[15]"). The PDF's bibliography is internally self-consistent and is
what the authors compiled. **This file quotes the PDF's numbers in prose and the
HTML's numbers in the `loc` line, and resolves no citation number to a work.**
Verified PDF bibliography entries: `[64]` = *Kimi Team et al., Kimi Linear…* (pdf line 2453);
`[24]` = Dao & Gu, *Transformers are SSMs* (pdf line 2370); `[140]` = Yang/Kautz/Hatamizadeh, *Gated Delta Networks* (pdf line 2615);
`[142]` = *Gated Linear Attention Transformers* (pdf line 2624); `[143]` = *Parallelizing Linear Transformers with the Delta Rule* (pdf line 2626);
`[106]` = Schlag et al., *Linear Transformers Are Secretly Fast Weight Programmers* (pdf line 2539).

---

## 2. State / decay formulation, transcribed

Section: **§2.1.1 Kimi Delta Attention** (body begins v2 line 447).
Subsection headings, as printed: *Chunkwise parallel form* (line 480), *Lower-bounded decay* (line 508), *Full-rank gate* (line 544).

### 2.1 Symbols the paper defines (v2 line 451)

> For clarity, we first describe a single attention head, with query and key vectors $\bm{q}_{t},\bm{k}_{t}\in\mathbb{R}^{d_{k}}$ , value vector $\bm{v}_{t}\in\mathbb{R}^{d_{v}}$ , and recurrent state $\mathbf{S}_{t}\in\mathbb{R}^{d_{k}\times d_{v}}$ .

Also v2 line 450:

> Consider a sequence of hidden states $\bm{x}_{t}\in\mathbb{R}^{d}$ , where $t$ indexes the token position and $d$ is the model hidden dimension.

v2 line 449 (PDF renders the same sentence as `[106, 140]` and `[64]`):

> KDA extends the delta-rule recurrence [ 105 , 139 ] with a channel-wise forget gate [ 57 ] .

### 2.2 Eq. 1 — the state update

v2 line 452 introduces it:

> KDA applies channel-wise decay before the delta-rule update:

v2 line 454, equation number `(1)` on v2 line 456:

$$\mathbf{S}_{t}=\left(\mathbf{I}-\beta_{t}\bm{k}_{t}\bm{k}_{t}^{\top}\right)\operatorname{Diag}(\bm{\alpha}_{t})\mathbf{S}_{t-1}+\beta_{t}\bm{k}_{t}\bm{v}_{t}^{\top},\qquad\tilde{\bm{o}}_{t}=\mathbf{S}_{t}^{\top}\bm{q}_{t}.$$

PDF cross-check (pdf line ~247, the same equation, same label `(1)`):
`St = (I − βt kt kt⊤) Diag(αt )St−1 + βt kt vt⊤ ,   õt = S⊤t qt.`

Definitions of the two scalars, v2 line 458 (the only place they are defined):

> Here, $\bm{\alpha}_{t}\in(0,1)^{d_{k}}$ is the channel-wise one-step retention factor, and $\beta_{t}\in(0,1)$ controls the delta-rule write strength.

### 2.3 Eq. 2 — per-head projections, incl. the short convolution

v2 line 460:

> Following Kimi Linear [ 57 ] , KDA parameterizes the per-head quantities as

v2 lines 462–472, equation number `(2)` on v2 line 474:

$$\bm{q}_{t}^{h},\bm{k}_{t}^{h}=\operatorname{L_{2}Norm}\!\left(\operatorname{Swish}\!\left(\operatorname{ShortConv}\!\left(\mathbf{W}_{q/k}^{h}\bm{x}_{t}\right)\right)\right)\in\mathbb{R}^{d_{k}}$$
$$\bm{v}_{t}^{h}=\operatorname{Swish}\!\left(\operatorname{ShortConv}\!\left(\mathbf{W}_{v}^{h}\bm{x}_{t}\right)\right)\in\mathbb{R}^{d_{v}}$$
$$\beta_{t}^{h}=\operatorname{Sigmoid}\!\left(\mathbf{W}_{\beta}^{h}\bm{x}_{t}\right)\in(0,1),$$
$$\bm{z}_{t}^{h}=\mathbf{W}_{\alpha}^{\uparrow}\mathbf{W}_{\alpha}^{\downarrow}\bm{x}_{t}+\bm{b}_{\alpha}^{h}\in\mathbb{R}^{d_{k}}.$$

Verbatim descriptive text, v2 lines 476–478:

> The query, key, and value projections apply $\operatorname{ShortConv}$ followed by $\operatorname{Swish}$ [ 139 ] , and the query and key are further normalized with $\operatorname{L_{2}Norm}$ [ 141 ] .
> The low-rank projection and head-specific bias $\bm{b}_{\alpha}^{h}\in\mathbb{R}^{d_{k}}$ produce a fine-grained decay logit $\bm{z}_{t}^{h}$ for each key channel.
> The lower-bounded mapping from $\bm{z}_{t}^{h}$ to $\bm{\alpha}_{t}^{h}$ is introduced after the chunkwise formulation below.

**Short convolution: NOT DEFINED.** `ShortConv` appears in the paper only as an
operator symbol in Eq. 2 (v2 lines 463, 466) and in the sentence at v2 line 476.
The paper gives **no** kernel size, **no** depth, **no** causal-vs-full statement,
**no** padding convention and **no** normalisation. Checked: every occurrence of
`conv`/`ShortConv` in the full 5604-line v2 extraction (grep, case-insensitive, all
hits reviewed) plus all appendices A–F (v2 lines 4527, 5335, 5368, 5470, 5504, 5525).
The only other mention is an unrelated pass-by in §5.4.2 (v2 line 1527): "…a single
fused kernel covering short convolution, input normalization, gating, the KDA
recurrence, and output normalization." It defines nothing.

### 2.4 Eq. 3 — cumulative decay

v2 lines 485–487, equation number `(3)` on v2 line 489:

> For positions $1\leq i\leq j\leq C$ , define the channel-wise cumulative decay

$$\bm{\gamma}_{[t]}^{i\rightarrow j}:=\prod_{r=i}^{j}\bm{\alpha}_{[t]}^{r},\qquad\bm{\gamma}_{[t]}^{r}:=\bm{\gamma}_{[t]}^{1\rightarrow r}.$$

v2 line 491:

> As in Kimi Linear, $\bm{\Gamma}_{[t]}^{1\rightarrow C}\in\mathbb{R}^{C\times d_{k}}$ stacks $\bm{\gamma}_{[t]}^{1},\ldots,\bm{\gamma}_{[t]}^{C}$ row-wise.

Chunk index context, v2 lines 483–484:

> For a chunk size $C$ , $\mathbf{X}_{[t]}$ stacks the token vectors in the $t$ -th chunk for $\mathbf{X}\in\{\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O},\mathbf{U},\mathbf{W}\}$ .
> The matrix $\mathbf{S}_{[t]}\in\mathbb{R}^{d_{k}\times d_{v}}$ denotes the recurrent state entering chunk $t$ .

### 2.5 Eq. 4 — chunkwise parallel form

v2 line 492:

> The UT transform produces $\mathbf{U}_{[t]}$ and $\mathbf{W}_{[t]}$ , from which we define the pseudo-value term $\widetilde{\mathbf{V}}_{[t]}:=\mathbf{U}_{[t]}-\mathbf{W}_{[t]}\mathbf{S}_{[t]}$ .

v2 line 493:

> Given the incoming state $\mathbf{S}_{[t]}$ , all outputs in chunk $t$ are computed in parallel as

v2 lines 495–499, equation number `(4)` on v2 line 501:

$$\mathbf{A}_{[t]}=\operatorname{Tril}\!\left[(\mathbf{Q}_{[t]}\odot\bm{\Gamma}_{[t]}^{1\rightarrow C})(\mathbf{K}_{[t]}/\bm{\Gamma}_{[t]}^{1\rightarrow C})^{\top}\right],$$
$$\mathbf{O}_{[t]}=\underbrace{(\bm{\Gamma}_{[t]}^{1\rightarrow C}\odot\mathbf{Q}_{[t]})\mathbf{S}_{[t]}}_{\text{inter-chunk}}+\underbrace{\mathbf{A}_{[t]}\widetilde{\mathbf{V}}_{[t]}}_{\text{intra-chunk}}.$$

v2 lines 503–506 (mask semantics, verbatim):

> For a matrix $\mathbf{M}$ , $\operatorname{Tril}(\mathbf{M})$ sets all strictly upper-triangular entries to zero and retains the lower-triangular entries, including the diagonal.
> This mask enforces causal interactions within the chunk, and the diagonal is retained because each output reads the state after the current-token update.
> The first term in $\mathbf{O}_{[t]}$ carries information from preceding chunks, whereas the second term accounts for interactions within the current chunk.
> We refer readers to Kimi Linear [ 57 ] for the UT transform and the full derivation of the chunkwise form.

**The UT transform itself is NOT derived here** — the paper explicitly defers it to
Kimi Linear (v2 line 506).

### 2.6 Eq. 5 — decay parameterisation (this is the K3 change)

Preamble, v2 lines 510–514 (verbatim, abridged only by `…`):

> Eq. 4 rescales the keys in each chunk by the reciprocal cumulative decay $1/\bm{\Gamma}_{[t]}^{1\rightarrow C}$ .
> Because $\bm{\Gamma}_{[t]}^{1\rightarrow C}$ is a product of retention factors in $(0,1)$ , this reciprocal can grow without bound and overflow in finite precision [ 140 , 57 ] .
> Kimi Linear controls this numerical range by computing relative decay in log space and dividing each chunk into secondary $16$ -token tiles [ 140 , 57 ] .
> …

v2 lines 516–518:

> Kimi K3 addresses this bottleneck by changing the mapping from the decay logits $\bm{z}_{t}^{h}$ to the per-step log-decay $\bm{g}_{t}^{h}$ .
> Following GDN and Mamba-2, Kimi Linear uses the negative-Softplus mapping $\bm{g}_{t}^{h}=-e^{A_{h}}\operatorname{Softplus}(\bm{z}_{t}^{h})\in(-\infty,0)^{d_{k}}$ [ 139 , 24 , 57 ] .
> Kimi K3 instead uses a scaled sigmoid to bound the log-decay from below:

v2 lines 520–524, equation number `(5)` on v2 line 526:

$$\bm{g}_{t}^{h}=g_{\min}\operatorname{Sigmoid}\!\left(e^{A_{h}}\bm{z}_{t}^{h}\right)\in(g_{\min},0)^{d_{k}},\qquad\bm{\alpha}_{t}^{h}=\exp(\bm{g}_{t}^{h})\in\left(e^{g_{\min}},1\right)^{d_{k}}$$

v2 line 528 — **the definition of `A_h`**:

> where $A_{h}$ is a learnable per-head log-scale and $g_{\min}=-5$ is fixed.

Consequences, v2 lines 530–533 (verbatim):

> With $g_{\min}=-5$ , every retention factor satisfies $\alpha_{t,j}^{h}>e^{-5}\approx 6.7\times 10^{-3}$ , and the cumulative log-decay over a $16$ -token tile lies in $(-80,0)$ .
> The corresponding reciprocal rescaling factor is therefore smaller than $e^{80}$ and remains within the BF16 dynamic range.
> This finite range allows both diagonal and off-diagonal tiles to use dense Tensor Core matrix multiplications, eliminating the position-pair diagonal path.
> This parameterization is closely related to the lower-bounded recurrence gates in prior work [ 97 , 26, 91 ] .

Figure 3 caption, v2 lines 540–542:

> Figure 3: Lower-bounded decay and its effect on chunkwise KDA computation.
> (a) Kimi Linear uses an unbounded negative-Softplus mapping, whereas Kimi K3 bounds the log-decay with a scaled sigmoid; the curves show $A=0$ and $g_{\min}=-5$ .
> (b) Kimi Linear evaluates each diagonal tile with an explicit position-pair computation, while the bounded range in Kimi K3 allows all causal tiles to use dense Tensor Core matrix multiplications.

### 2.7 Eq. 6 — full-rank output gate

v2 lines 546–547:

> Finally, Kimi K3 changes KDA’s output gate from the low-rank parameterization used by Kimi Linear [ 57 ] to an input-dependent full-rank projection.
> After applying head-wise RMSNorm [ 147 ] to the recurrent output, KDA applies data-dependent output gating [ 99 ] :

v2 line 549, equation number `(6)` on v2 line 551:

$$\bm{y}_{t}=\mathbf{W}_{o}\!\left[\operatorname{Sigmoid}\!\left(\mathbf{W}_{g}\bm{x}_{t}\right)\odot\operatorname{RMSNorm}(\tilde{\bm{o}}_{t})\right].$$

### 2.8 Sibling location — the KDA:MLA ratio

v2 line 442 (§2.1, not §2.1.1):

> Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a $3{:}1$ mixing ratio.

---

## 3. Every initialisation the paper states in §2.1.1

There is **exactly one** initialisation statement in §2.1.1.

**The `A_h` quote — VERIFIED, present, verbatim.** v2 line **529**:

> We initialize $A_{h}=0$ , and each bias $\bm{b}_{\alpha}^{h}$ is initialized following [ 57 , 24, 139 ] .

* PDF cross-check, v2 pdf line **342** (pdftotext flattens the subscript):
  `where Ah is a learnable per-head log-scale and gmin = −5 is fixed. We initialize Ah = 0, and each bias bhα is initialized`
* **v1 agrees, verbatim, at v1 line 529** (identical sentence, only the citation numbers differ: v1 renders `[56, 24, 137]`).
* Secondary corroboration inside the paper: the Fig. 3 caption says "the curves show $A=0$ and $g_{\min}=-5$" (v2 line 541).

Related fixed (non-learned) constant, v2 line 528: `$g_{\min}=-5$ is fixed`. The paper
does not state whether `g_min` is a hyperparameter that was swept or simply chosen.

**Other initialisations elsewhere in the paper, outside §2.1.1** (for completeness;
none of them touch KDA):

* v2 line 1108 (§4.1.3, MTP feature projection): `initialized as $[\,\bm{0}\;\;\bm{0}\;\;\bm{I}\,]$` — the projector $\bm{W}_{\mathrm{E3}}$.
* v2 line 5431 (Appendix C, Quantile Balancing): `Initialize $\bm{\beta}=\bm{0}_{1\times n}$ ;` — the MoE load-balancing bias.
* v2 lines 759, 761, 763 (§2.4): vision-tower initialisation is discussed (MoonViT-V2 **from scratch** vs SigLIP-initialised MoonViT-3D), but no tensor-value initialisation is given.

---

## 4. What the paper does NOT say — the ceiling

Each item below was checked against the full 5604-line v2 extraction and the 3094-line
PDF text, including all six appendices (A Contributions, B SiTU-GLU, C Quantile
Balancing, D Histogram Estimation, E MoonEP bound, F Chat Template).

1. **`ShortConv` is never defined.** No kernel width, no depth, no causal flag, no
   padding, no normalisation. Checked: all `conv` occurrences in the document; the
   only other hit is a §5.4.2 pass-by that adds no definition. If you need the short
   conv, this paper is not the source — Kimi Linear (arXiv 2510.26692) is where §2.1.1
   sends you for the UT transform, and the same gap almost certainly exists there.
2. **The chunk size `C` is never given a value.** `C` is introduced at v2 line 483
   and used throughout §2.1.1; the only tile figure in the paper is the `16`-token
   secondary tile (v2 lines 512, 530), which is a *sub-tile* of a chunk, not the
   chunk size. No value for `C` appears in §2.1.1, §3.3, §5.1, or Table 1.
3. **`b_alpha^h` is not initialised to a stated value** — the paper defers entirely to
   three citations (v2 line 529). If you need a number, this paper does not have one.
4. **`A_h`'s shape/rank is not specified beyond "per-head"** (v2 line 528). Whether
   `A_h` is a scalar per head or a per-head vector is not disambiguated; the scalar
   reading is the natural one from `e^{A_h}` multiplying a `d_k`-vector, but the paper
   does not say so.
5. **The rank of the low-rank decay projection `W_alpha^down` / `W_alpha^up` is not
   given.** Eq. 2 (v2 line 472) names the factors and the arrows; no rank, no
   expansion factor. Table 1 gives `Latent MoE Dimension 3584 (0.5×)`, which is the MoE,
   not the KDA decay projection — do not conflate them.
6. **No `d_k` / `d_v` values for the KDA heads.** Table 1 (v2 lines 812–960) gives
   `Hidden Dimension 7,168` and `Attention Heads 96` for the model, but never splits
   these into per-head KDA `d_k`/`d_v`, and never states the number of KDA heads.
   `Attention-Layer Composition 69 KDA + 24 MLA` is a layer count, not a head count.
7. **The UT transform `U_[t]`, `W_[t]` is not defined** — explicitly deferred to
   Kimi Linear (v2 line 506). Any implementation must source it elsewhere.
8. **No ablation isolates `A_h` init, `g_min = -5`, or the tile size `16`.** §2.1.1
   states the range argument and shows Fig. 3 as an illustration; there is no measured
   comparison of any of these three choices anywhere in the paper.
9. **The HTML build's citation numbers are unreliable** (see Provenance). Do not quote
   an in-text `[57]`/`[139]` from the HTML and expect it to name Kimi Linear / Swish.

---

*Transcribed 2026-09-29 from `https://arxiv.org/html/2607.24653v2`, cross-checked
against `https://arxiv.org/pdf/2607.24653v2` and `https://arxiv.org/html/2607.24653v1`.
No GPU, no cargo, no build. No other file in the repository was read or modified.*
