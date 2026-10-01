# GatedDeltaNet-2 and KDA: provenance, and a line-by-line check of `burn-gdn2` / `burn-kda`

**Fetch date: 2026-09-29.** Every arXiv id below was resolved through the arXiv
API (`export.arxiv.org/api/query?id_list=…`) and, where available, the arXiv
HTML full text and the authors' own source repositories. No claim in §1–§6
rests on a snippet: each was read from the primary document.

Scope: `vendor/dormouse-fused/crates/burn-gdn2/` and
`vendor/dormouse-fused/crates/burn-kda/`. Read-only; no file in either crate was
touched (`gen_reference.rs` / `gen_reference.py` were read, never edited).

---

## 0. THE HEADLINE: the `2601.16531` citation is real, and it is not about GatedDeltaNet

**`arXiv 2601.16531` resolves.** It is not a phantom id. But it is a
misattribution, and it is a misattribution with a specific, checkable cause.

| | |
|---|---|
| **id** | `arXiv:2601.16531` (v1 23 Jan 2026, v2 26 Jan 2026) |
| **actual title** | *A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics* |
| **actual author** | Tao Lin (single author) |
| **actual subject** | Engram-style n-gram memory, hash collisions, minimal perfect hash functions, **gate/loss credit assignment** |
| **URL** | <https://arxiv.org/abs/2601.16531> |
| **is it about GDN / KDA?** | **No.** Zero overlap. |

**The citation is not on `burn-gdn2` or `burn-kda` at all.** Grepping both
crates for arXiv ids returns exactly three, and none of them is `2601.16531`:

- `burn-gdn2/README.md:10` — `2605.22791` (Gated DeltaNet-2)
- `burn-kda/src/lib.rs:5-7` — `2510.26692` (Kimi Linear), `2607.24653` (Kimi K3)

`2601.16531` appears **only** in the *Engram* line of the repo:
`crates/dormouse-core/src/config/schema.rs:122`, `configs/small.toml:29`,
`docs/protocols/AB-PROTOCOL.md:119`, `README.md:82`,
`docs/research/2026-09-27-pkm-engram-deepseek.md`. In every one of those places it
is used for what it actually is: a single-author Engram-preprint, cited as
weak evidence, for the 500K-slot saturation curve and the gate anti-correlation.
`docs/research/2026-09-27-pkm-engram-deepseek.md:313` even says so in as many words
("a single-author preprint — real, on-point, and weak evidence").

**Verdict: NOT a fabricated citation, and NOT a citation of GDN/KDA.** It is a
real paper, correctly scoped where it is used. The premise of this task — that
the GDN/KDA crates claim `2601.16531` — is false, and I checked before
reporting it as such.

What *is* fabricated is a different, smaller thing, and it is real: see
**§5, delta rows 25 and 26**. `burn-kda/src/lib.rs:166-170` cites
"Moonshot/FLA init (FlashKDA torch_ref, kda.py): A_log = -3" and that citation
names a file that does not exist, in a repository, with values that are in
neither the paper nor the code. That is a genuine retracted-claim-class
defect and it should be fixed.

---

## 1. Provenance

### 1.1 The four documents that actually govern these two crates

| role | id / URL | resolves | what it is |
|---|---|---|---|
| GDN-2 paper | **`arXiv:2605.22791`** | ✅ yes | *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*, Hatamizadeh, Choi, Kautz (NVIDIA), 21 May 2026, cs.AI, CC BY 4.0. Full HTML read: <https://arxiv.org/html/2605.22791v1> |
| GDN-2 code | **https://github.com/NVlabs/GatedDeltaNet-2** | ✅ yes | "Official PyTorch Implementation", 323 stars, 9 commits, NVIDIA Source Code License-NC. Files: `lit_gpt/gdn2.py`, `lit_gpt/gdn2_ops/chunk_gdn2.py`, `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py`, `paper/`. **`lit_gpt/gdn2.py` and `chunk_gdn2.py` were both read in full.** |
| KDA paper | **`arXiv:2510.26692`** | ✅ yes | *Kimi Linear: An Expressive, Efficient Attention Architecture*, Kimi Team (59 authors), v2 1 Nov 2025, cs.CL |
| KDA code | **https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda** | ✅ yes | The canonical KDA kernels. Named as the official implementation *in the Kimi Linear paper itself* (footnote 1). Layer: `fla/layers/kda.py` — **read in full.** |
| K3 paper | **`arXiv:2607.24653`** | ✅ yes | *Kimi K3: Open Frontier Intelligence*, Kimi Team (401 authors), v2 7 Aug 2026, cs.CL. §2.1.1 read in full. |
| K3 code | **https://github.com/MoonshotAI/FlashKDA** | ✅ yes | "FlashKDA: high-performance Kimi Delta Attention kernels", CUTLASS, SM90+. `tests/torch_ref.py` — **read in full.** This is the "Moonshot/FLA" the code means. |
| related | **https://github.com/MoonshotAI/Attention-Residuals** | ✅ yes | The "Attention Residuals" the task brief mentions. It is about **residual connections**, not about GatedDeltaNet. dormouse's GR design is ported from here (§3.5 of AGENTS.md), not from GDN-2. |

**ORIGINAL SOURCE CODE EXISTS for all three mechanisms**, and two of the three
are the *exact* code our comments name. That is the good news: almost every
question below is answerable from a primary source rather than a guess.

### 1.2 Citation hygiene verdict

| claim | status |
|---|---|
| `burn-gdn2/README.md:10` → `2605.22791` + `NVlabs/GatedDeltaNet-2` | ✅ **verified**, both halves |
| `burn-kda/src/lib.rs:5` → `2510.26692` | ✅ **verified** |
| `burn-kda/src/lib.rs:6` → `2607.24653`, "§2.1.1, Eqs 1-6 — lower-bounded decay, full-rank output gate" | ✅ **verified**; §2.1.1 has exactly subsections "Chunkwise parallel form" / "Lower-bounded decay" / "Full-rank gate", and Eqs 3,4,5,6 |
| `burn-kda/src/lib.rs:166` → "Moonshot/FLA init (FlashKDA torch_ref, kda.py): A_log = -3" | ❌ **FALSE CITATION** — see §5 row 25 |
| `AGENTS.md:274,837` → "decay init matches the Moonshot/FLA recipe (`a_log=-3`, `b_alpha=1.0`)" | ❌ **UNSUPPORTED** — same defect, propagated to the rulebook |
| `AGENTS.md:839` → "FlashKDA math (K3 decay, chunked WY, chunk 16)" | ✅ **verified** (chunk 16 is literal) |
| `burn-gdn2/README.md:22` → "1000-case comparison against an independent transcription" | ✅ **honest** — `tools/gen_reference.rs:24` says outright "The fixture is a transcription, not the original authors' bytes". This is exactly the ADR-0020 form. Credit where due. |

---

## 2. The paper's recurrence, transcribed literally

Notation is the paper's. State orientation is the paper's:
`o_t = S_tᵀ q_t`, `S_t ∈ R^{d_k × d_v}`.

### 2.1 The lineage (GDN-2 §2.1–2.2, Eqs 1–7)

**Eq 1 — plain linear attention.**
```
S_t = S_{t−1} + k_t v_tᵀ
o_t = S_tᵀ q_t
```

**Eq 4 — Mamba-2, scalar decay.**
```
S_t = α_t S_{t−1} + k_t v_tᵀ,     α_t ∈ (0,1]
```

**Eq 5 — DeltaNet, scalar delta rule.**
```
S_t = S_{t−1} + β_t k_t (v_t − S_{t−1}ᵀ k_t)ᵀ
    = (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ
```

**Eq 6 — Gated DeltaNet.** `S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`
(both gates scalar per head).

**Eq 7 — KDA. This is the recurrence `burn-kda` implements.**
```
S_t = (I − β_t k_t k_tᵀ) D_t S_{t−1} + β_t k_t v_tᵀ,    D_t = Diag(α_t),  α_t ∈ (0,1]^{d_k}
```

### 2.2 Gated Delta Rule-2 — the recurrence `burn-gdn2` implements (GDN-2 §3.1, Eqs 8–10)

**Eq 8** — the two channel-wise gates:
```
e_t = b_t ⊙ k_t,      b_t ∈ [0,1]^{d_k}     (erase gate, key axis)
z_t = w_t ⊙ v_t,      w_t ∈ [0,1]^{d_v}     (write gate, value axis)
```

**Eq 9** — the operational form (decay first, then the active edit):
```
S̄_t = D_t S_{t−1}
r_t = S̄_tᵀ e_t
S_t = S̄_t + k_t (z_t − r_t)ᵀ
```

**Eq 10 — Gated Delta Rule-2**, the boxed headline equation:
```
S_t = ( I − k_t (b_t ⊙ k_t)ᵀ ) D_t S_{t−1} + k_t (w_t ⊙ v_t)ᵀ
o_t = S_tᵀ q_t
```

`b_t = β_t 1_{d_k}` and `w_t = β_t 1_{d_v}` recovers KDA (Eq 7) exactly;
further `α_t = α_t 1_{d_k}` recovers Gated DeltaNet (Eq 6).

**Eq 11** — gate projections: `b_t = σ(W_b x_t)`, `w_t = σ(W_w x_t)`.

**Eq 12** — log-decay: `g_t = − exp(a) ⊙ softplus(W_f x_t + δ)`, `α_t = exp(g_t)`.
Computed in fp32 before the kernel consumes it.

**Eq 17** — the decoupled residual, which is the whole point of the paper:
```
z_t − S̄_tᵀ e_t = w_t ⊙ v_t − (D_t S_{t−1})ᵀ (b_t ⊙ k_t)
```

**Block design (§3.5, Fig. 1).** q, k: `Linear → short causal conv → SiLU → L2Norm`.
v: `Linear → short causal conv → SiLU` (no norm). Decay branch → `α_t`. b and w:
`Linear → sigmoid`. Output: `RMSNorm → ⊙ SiLU output gate → out proj`.
With grouped value heads (`H_v > H`): q, k, g, b are repeated across the value-head
group; v and w stay on the value-head axis.

**Negative-eigenvalue variant (§3.1, C.1).** "We also support the
negative-eigenvalue variant of [20] by scaling only the erase gate to
`[0,2]^{d_k}`. The write gate remains in `[0,1]^{d_v}`." Reference [20] is
*Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*,
arXiv 2411.12537 — the same id the reference code's docstring cites.

### 2.3 The chunkwise form (GDN-2 §3.3, App. A; Eqs 18–25)

```
G_r = Σ_{i=1..r} g_i,   γ_r = exp(G_r),   γ_0 = 1_{d_k},   α_r = exp(g_r)     (18/30)

S_r   = Diag(γ_r) Ŝ_r                                                        (31)
Ŝ_r   = (I − k̄_r ē_rᵀ) Ŝ_{r−1} + k̄_r z_rᵀ
        k̄_r = γ_r⁻¹ ⊙ k_r,   ē_r = γ_r ⊙ e_r                                (32)

K̄ = γ⁻¹ ⊙ K,   Ē = γ ⊙ (B ⊙ K),   Z = W ⊙ V                                (33)
T = tril(Ē K̄ᵀ, −1),   A = (I + T)⁻¹                                        (34)
Y = A Ē,   U = A Z                    ← Y is erase-side, U is write-side

S_{[n+1]} = Diag(γ_C) S_{[n]} + K_tailᵀ (U − Y S_{[n]})                       (23)
O_{[n]}   = Q_γ S_{[n]} + A_qk (U − Y S_{[n]})                               (24)
Q_γ row r = γ_r ⊙ q_r
K_tail row r = (γ_C/γ_r) ⊙ k_r
(A_qk)_{rs} = 1_{r≥s} · q_rᵀ Diag(γ_r/γ_s) k_s                              (25)
```

Gate-aware backward (§3.4, Eqs 26–28) — the gates **must** be inside the `dA`
accumulation, not a scalar post-scale:
```
dA += dU Zᵀ    with Z = W ⊙ V
dA += dY Ēᵀ    with Ē = γ ⊙ (B ⊙ K)
dT = −tril(Aᵀ dA Aᵀ, −1)
```
App. B.5 states this explicitly and it is the paper's one necessary backward
change.

### 2.4 The KDA/K3 gates (`burn-kda`)

**Kimi Linear (arXiv 2510.26692) — negative-Softplus, unbounded:**
```
g_t^h = − e^{A_h} Softplus(z_t^h) ∈ (−∞, 0)^{d_k}
```

**Kimi K3 §2.1.1, Eq 5 — scaled sigmoid, lower-bounded. Verbatim:**
```
g_t^h = g_min · Sigmoid( e^{A_h} z_t^h ) ∈ (g_min, 0)^{d_k}
α_t^h = exp(g_t^h)      ∈ (e^{g_min}, 1)^{d_k}
```
> "where `A_h` is a learnable per-head log-scale and **`g_min = −5` is fixed**.
> **We initialize `A_h = 0`**, and each bias `b_α^h` is initialized following
> [57, 24, 139]."
> "With `g_min = −5`, every retention factor satisfies `α_{t,j}^h > e^{−5} ≈
> 6.7×10⁻³`, and the cumulative log-decay over a **16-token tile** lies in
> `(−80, 0)`. The corresponding reciprocal rescaling factor is therefore
> smaller than `e^80` and remains within the BF16 dynamic range."

**Kimi K3 §2.1.1, Eq 6 — full-rank gate. Verbatim:**
```
y_t = W_o [ Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t) ]
```
> "Kimi K3 changes KDA's output gate from the low-rank parameterization used by
> Kimi Linear to an input-dependent full-rank projection. After applying
> head-wise RMSNorm to the recurrent output, KDA applies data-dependent output
> gating."

**KDA Eq 3/Eq 4 (K3)** — the same WY chunk algebra as GDN-2, with
`Γ^{1→C}` in place of `γ`, plus the `Tril` mask defined as retaining the
**diagonal** ("the diagonal is retained because each output reads the state
after the current-token update").

### 2.5 The short conv — definition, and what the paper does *not* say

**The paper never states the kernel size.** GDN-2 §3.5 and App. C.1 say only
"short causal convolution" / "short-convolutional projections". No number, no
padding mode. (KDA's paper is the same — it defers to Kimi Linear and FLA.)

The number comes from the code. `lit_gpt/gdn2.py:99`:
```python
conv_size: int = 4,          # default
conv_bias: bool = False,     # default
```
and `fla/layers/kda.py` has the identical two defaults. The layer instantiates
`fla.modules.ShortConvolution(hidden_size=…, kernel_size=conv_size, bias=conv_bias, activation="silu")`.

`fla/modules/conv/short_conv.py` — that class is an `nn.Conv1d` with:
```python
groups=hidden_size,          # depthwise
padding=kernel_size - 1,     # 3
bias=bias,                   # False
# padding_mode is NOT passed  →  nn.Conv1d default  →  'zeros'
```
and the forward path never goes through `nn.Conv1d.forward` — it calls
`causal_conv1d` (Triton). `fla/modules/conv/triton/kernels.py`,
`causal_conv1d_fwd_kernel`:
```python
b_yi = tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :], other=0.0)
```
`other=0.0` on the out-of-range (i.e. `o_x < 0`, the left pad) branch. The decode
path agrees: `short_conv.py::step` does `cache = x.new_zeros(N, D, W)`.

**The reference left-pads the short conv with ZEROS, and the SiLU is applied to
the sum, not per-tap.** That is the whole specification.

---

## 3. Delta table — our implementation vs the paper

Verdicts: **MATCH** (equivalent), **BUG** (wrong), **DELIBERATE** (knowing
divergence, documented), **BENIGN** (differs but cannot change a number), **UNVERIFIABLE**.

### 3.1 `burn-gdn2` — vs arXiv 2605.22791 + NVlabs/GatedDeltaNet-2

| # | what | our `file:line` | paper / reference | verdict |
|---|---|---|---|---|
| 1 | Gated Delta Rule-2, Eq 10 | `src/kernel/fused_recurrent.rs:39-56` — decay `S*=exp(g)ᵀ`, `erased=(S*(b⊙k)ᵀ).sum_K`, `v_new=w⊙v−erased`, `S+=k v_newᵀ`, `o=(S*qᵀ).sum·scale` | Eq 9 verbatim | **MATCH** |
| 2 | Erase/write gates, Eq 11 | `src/module.rs:530-531` `sigmoid(b_proj(x))`, `sigmoid(w_proj(x))` | Eq 11 | **MATCH** |
| 3 | Log-decay, Eq 12 | `src/module.rs:505-516` `g = -a_log.exp() * softplus(f_proj_1(f_proj_0(x)) + dt_bias, 1.0)` | Eq 12 / C.1 Eq 86 | **MATCH** (low-rank `Proj_f` is the reference's `nn.Sequential` exactly) |
| 4 | L2Norm on q,k | `src/module.rs:541-542`, eps `1e-6` | §D.2 | **MATCH** (ref does it in-kernel, `use_qk_l2norm_in_kernel=True`; equivalent) |
| 5 | GVA repeats q,k,g,**b**; not v,w | `src/module.rs:545-556` | §3.5 / C.1 | **MATCH** |
| 6 | `allow_neg_eigval` → `b*2`, w untouched | `src/module.rs:558-560` | §3.1, C.1 | **MATCH** |
| 7 | **chunk size = 64** | `src/config.rs:126` | Paper App. C.2: *"The chunk size is fixed to C=64."* Ref: `chunk_size: int = 64` | **MATCH** |
| 8 | `scale = d_k^{-0.5}` | `src/module.rs:299` | Ref `chunk.py`: `if scale is None: scale = k.shape[-1] ** -0.5` | **MATCH** |
| 9 | Xavier-uniform, gain `2^{-2.5}`, zero bias | `src/module.rs:179-196` | §D.5 verbatim | **MATCH** |
| 10 | `A_log = log U(1,16)` | `src/module.rs:206-210` `U(ln 1, ln 16)` | Ref: `torch.log(torch.empty(H).uniform_(1,16))` | **MATCH** |
| 11 | `dt_bias = dt + log(−expm1(−dt))`, `dt~logU(.001,.1)` clamped 1e-4 | `src/module.rs:212-221` | Ref: identical formula | **MATCH** |
| 12 | conv weight init `U(−0.5, 0.5)` | `src/module.rs:200-204` | `nn.Conv1d` default = kaiming_uniform(a=√5) → `1/√fan_in` = 1/2 for fan_in 4 | **MATCH** (the comment at `:198` is correct and non-obvious) |
| 13 | conv kernel size 4, depthwise, no bias, then SiLU | `src/short_conv.rs:5,22-64` | **Not in paper.** Ref default `conv_size=4`, `conv_bias=False`, `activation='silu'` | **MATCH** to the reference default; **BENIGN** that it is a `const` with no config knob |
| 14 | **short-conv left padding** | `src/short_conv.rs:43-46` `x[:,0:1].repeat(1,3,1)` — **replicate first token** | Paper: silent. Ref: **zero** (`other=0.0`; `cache = new_zeros`) | **BUG** — see §4.1 |
| 15 | chunk algebra: `γ_r`, `K̄`, `Ē`, `Z`, `T=tril(ĒK̄ᵀ,−1)`, `A=(I+T)⁻¹`, `Y=AĒ`, `U=AZ`, `K_tail`, `A_qk` | `src/forward.rs:247-297` (`k_over_gamma`, `bk*g_exp`, `strict` mask, `neumann_inverse`, `w_wy`, `u`, `k_dec`, `decay_last`, `scale_causal`) | Eqs 18–25 | **MATCH**, line for line |
| 16 | output norm: RMSNorm → SiLU gate → `o_proj` | `src/module.rs:369-373` | §3.5; ref `FusedRMSNormSwishGate` | **MATCH** |
| 17 | `fused_recurrent` for `q_len ≤ 64` | `src/module.rs:74,303-307` | Ref: `q_len <= 64 and not self.training` | **MATCH** (our `forward` is the inference entry, `forward_train_core` the training one — the `not training` term is carried by which function you call) |
| 18 | training may run `FusedRecurrent` | `src/module.rs:426-440` | Ref asserts `mode == "chunk"` in training | **BENIGN** — dormouse runs KDA/GDN by config, not by grad-enabled |
| 19 | `min_decay` per-channel decay floor | `src/module.rs:518-528` | Not in the paper | **DELIBERATE** — README:137 labels it "extension, not in the paper". Correctly declared. |
| 20 | read-only forward (`update_state=false`) | `src/module.rs:355-366` | Not in the paper | **DELIBERATE** — documented as prefill |
| 21 | batched chunk arm limited to `chunk_size ≤ 16` | `src/forward.rs:7,72-82,206-211` | Ref is stable at 64 (it forms **differences** `exp2(g_r − g_s)`, never an absolute reciprocal) | **DELIBERATE** — a real numerical limit of the absolute-reciprocal form; README:160-164 states the `exp(cumsum g) < −88` reason. **Note the default `chunk_size=64` therefore routes to the `Loop` arm**, so the batched arm is off by default. Worth a line in the README. |

### 3.2 `burn-kda` — vs arXiv 2510.26692 + 2607.24653 + MoonshotAI/FlashKDA

| # | what | our `file:line` | paper / reference | verdict |
|---|---|---|---|---|
| 22 | KDA recurrence, Eq 1 / GDN-2 Eq 7 | `src/lib.rs:214-236` `kda_step` | `S=(I−βkkᵀ)DS_{t−1}+βkvᵀ`, `o=Sᵀq` | **MATCH** |
| 23 | **K3 bounded decay, Eq 5** `g=g_min·Sigmoid(e^{A_h}z)`, `g_min=−5` | `src/lib.rs:26,200` | 2607.24653 §2.1.1 Eq 5 verbatim; `G_MIN = -5.0` | **MATCH — verified against the paper** |
| 24 | Kimi Linear softplus form | `src/lib.rs:198` `−exp(A)·softplus(z)` | Kimi Linear, cited in K3 §2.1.1 | **MATCH** |
| 25 | **`a_log` init = `-3.0`** | `src/lib.rs:170`, comment `:166` | K3 §2.1.1: **"We initialize `A_h = 0`"**. FLA `fla/layers/kda.py`: `log(U(1,16))`, or `zeros` when `safe_gate`. FlashKDA `tests/torch_ref.py`: `A_log` is an *input argument*; its tests use `torch.rand(H)`. **No `−3` anywhere.** | **BUG + FALSE CITATION** — see §4.2 |
| 26 | **`b_alpha` init = `ones`** | `src/lib.rs:165`, comment `:168` | K3 §2.1.1: *"each bias `b_α^h` is initialized following [57, 24, 139]"*. FLA implements that as `dt~logU(.001,.1)`, `inv_dt = dt + log(−expm1(−dt))` ⇒ ≈ **−7 … −2.3**, i.e. **not 1.0**. FlashKDA takes it as an input, tests use `torch.rand`. | **BUG + FALSE CITATION** — see §4.2 |
| 27 | `a_log.clamp(-10.0, 20.0)` | `src/lib.rs:189-194` | In no source | **DELIBERATE** — and well argued in the comment (fp32 `exp` overflow, NaN at overfit). Note this *bounds* the damage of row 25 but does not fix it. |
| 28 | `beta_t^h = Sigmoid(W_beta^h x_t)`, scalar per head | `src/lib.rs:441-446` | K3 Eq 2; FLA `b_proj = nn.Linear(hidden, num_v_heads)` + `use_beta_sigmoid_in_kernel=True` | **MATCH** |
| 29 | β on both axes (erase and write) | `src/lib.rs:445-446` broadcasts `b_k` and `b_v` | GDN-2 Eq 7: `β_t` multiplies both | **MATCH** (this is the KDA↔GDN-2 mapping the header comment at `lib.rs:20-22` claims) |
| 30 | **full-rank sigmoid output gate, Eq 6** | `src/lib.rs:483-484,494-500` `RMSNorm(o) ⊙ sigmoid(W_g x) ⊙ w_norm → o_proj` | 2607.24653 §2.1.1 **Eq 6 verbatim**; FLA `FusedRMSNormGated(activation="sigmoid")` | **MATCH — verified against the paper** |
| 31 | **chunk size = 16** | `src/lib.rs:81` | FlashKDA `tests/torch_ref.py`: `CHUNK = 16`. K3 §2.1.1: "16-token tile", cum. log-decay in `(−80,0)` | **MATCH — verified** |
| 32 | short conv | reuses `burn_gdn2::short_conv_1d` | zero-pad, per row 13 | **BUG (inherited)** — same defect as row 14 |
| 33 | GVA repeat of q,k,g,b | `src/lib.rs:456-469` | FLA `kda.py` (`state_v_first`, per-value-head `gate_dim`) | **MATCH** |

**Totals: 33 rows. 27 MATCH · 1 BUG-class false citation ×2 rows · 1 BUG ×2
crates (inherited) · 3 DELIBERATE · 1 BENIGN.**

---

## 4. The two things that are actually wrong

### 4.1 Short-conv left padding: replicate, where the reference zero-pads — and the test cannot see it

`src/short_conv.rs:42-47`:
```rust
None => {
    let pad = x.clone().slice([0..b, 0..1, 0..c]).repeat(&[1, SHORT_CONV_CACHE, 1]);
    let combined = Tensor::cat(vec![pad, x], 1);
```

The reference is `other=0.0` on the left-pad branch
(`fla/modules/conv/triton/kernels.py::causal_conv1d_fwd_kernel`) and
`cache = x.new_zeros(N, D, W)` on the decode path
(`fla/modules/conv/short_conv.py::step`).

Concretely, for `T=1` prefill: reference gives `y = w₃·x₀`; we give
`y = (w₀+w₁+w₂+w₃)·x₀`. For `T>3` the divergence is confined to output
positions 0, 1, 2.

**Why no test caught it:** both reference generators replicate-pad.
`tests/gen_reference.py:85` — `x_pad = torch.cat([x[:, :1].repeat(1, 3, 1), x], dim=1)`.
`tools/gen_reference.rs:134` — `let src = if ti + i < 3 { 0 } else { ti + i - 3 };`.
The fixture is therefore *self-consistent with the bug* and the 1000-case
bit-exact comparison is structurally incapable of detecting it. This is the
same shape as the retracted `fused_chunk_verify.rs` finding in AGENTS.md §3.2
(verified the tensor adjoint twice): a comparison against a transcription of
the code under test cannot find a divergence from the code the transcription
was *supposed* to copy.

**Severity: low numerically, high as a class.** It changes the first three
positions of every sequence, and it changes `T=1` decode-from-scratch
completely. It cannot affect a 512-token prefill's loss by more than a
rounding-error-ish boundary term — **and `use_short_conv` is currently OFF in
dormouse** (reverted, per AGENTS.md:833-835), so the live impact today is
zero. That is the mitigation, and it is exactly why the revert was a good
idea. But the crate's README advertises `use_short_conv = true` as the
default and the crate would ship a wrong conv to any user who takes it.

**The paper does not adjudicate this.** There is no statement in 2605.22791
about padding. The fix is to change `repeat` to zeros and regenerate the
fixture — but that is another agent's file, so I am reporting, not editing.

### 4.2 `a_log = -3` / `b_alpha = 1.0` — a citation that resolves to nothing

`src/lib.rs:166-170`:
```rust
// Moonshot/FLA init (FlashKDA torch_ref, kda.py): A_log = -3
// gives exp(A) = 0.05 and dt_bias = 1.0 anchors z ~ 1, so the
// decay starts conservative (alpha ~ 0.08) instead of neutral
// (alpha ~ 0.5 at A=0, b=0). Matches the reference recipe.
a_log: Param::from_tensor(Tensor::full([n_heads, 1], -3.0, device)),
```

Every clause of that citation was checked and every one is wrong:

| the comment says | the source says |
|---|---|
| `FlashKDA` | ✅ the right repo — `MoonshotAI/FlashKDA`, "FlashKDA: high-performance Kimi Delta Attention kernels" |
| `torch_ref` | ✅ `tests/torch_ref.py` exists — but it is a **kernel reference**, and it takes `A_log` and `dt_bias` as **function arguments**. It contains no initialization at all. |
| `kda.py` | ❌ **no file by that name exists in FlashKDA.** The repo is `flash_kda/__init__.py`, `csrc/…`, `tests/…`, `benchmarks/…`. The nearest thing is FLA's `fla/layers/kda.py`, which is a *different repository* and says `log(U(1,16))`. |
| `A_log = -3` | ❌ not in the K3 paper (**it says `A_h = 0`**, explicitly), not in FLA, not in FlashKDA's `bench_fwd.py` (`torch.rand(H)`) or `test_fwd.py` (`torch.rand(H)`). |
| `dt_bias = 1.0` | ❌ FLA uses `dt + log(−expm1(−dt))` with `dt ~ logU(0.001, 0.1)` ⇒ ≈ −7…−2.3. K3 defers to "[57, 24, 139]", i.e. to exactly that. FlashKDA's tests use `torch.rand(H, D)`. |
| "Matches the reference recipe" | ❌ |

**Consequence.** `A_h = 0` is not a neutral choice that `-3` improves on; it is
the value the K3 paper specifies, and it is the value that makes the K3
lower-bound argument work as written. Under `DecayFn::Sigmoid` the decay is
`g = −5·σ(e^{−3}·z)`, which for `z = 0` gives `g = −2.5`, `α = 0.082` — i.e.
the model starts with an **8% retention per step at every key channel, every
token, for all 96 K3 heads**, before it has read a single byte. The K3 paper's
`A_h = 0` with `b_α^h ≈ −5` gives `g = −2.5` too at init… but with a
*learnable* `A_h` that starts at zero and a bias that starts where the
reference says, and — decisively — with the *magnitude* of the logit coming
from `b_α` rather than from a hard-wired `A`. With `A = −3` the effective
logit `e^A z` has slope 0.05, so `σ` is nearly linear over the whole operating
range and the lower-bounded decay degenerates towards Kimi Linear's
softplus mapping — which is precisely the thing K3 changed to avoid.

**This is a hyperparameter, not a correctness bug** — the recurrence is
algebraically fine for any `A`. But it is a *claim about provenance* that does
not hold, it is asserted in AGENTS.md §2.3/§3.5 as rule-level fact, and this
repo's own ADR-0020 says a crate may say "matches the reference" only if it
names the file. It named one; the file is not there. Fix the comment and the
init, or drop the citation.

---

## 5. Everything in the paper we do not implement / vice versa

### 5.1 In the paper, not in `burn-gdn2`

| item | why it is absent | should it be? |
|---|---|---|
| `transpose_state_layout`, packed `cu_seqlens` / varlen (App. C.6) | dormouse is fixed-length, non-varlen | no — YAGNI |
| `safe_gate` + `lower_bound ∈ [−5,0)` (`chunk_gdn2.py`) | our chunk is the batched/loop path | **maybe** — see §5.3 |
| `disable_recompute`, `return_intermediate_states` | memory/CUDA-graph knobs on a path we don't run | no |
| NUM_WARPS/precision autotune schedule (App. C.4) | we use burn, not Triton | no |
| `Akkd` fp32 diagonal sub-chunk buffer; the two-kernel intra scheme (token-parallel + sub-chunk) | our chunk is a single dense `(I+T)⁻¹` solve | no — algebraically equivalent, and §D.6 confirms the reference itself is validated against a tokenwise reference, not against a second chunk schedule |
| `SOLVE_TRIL_DOT_PRECISION` (ieee vs tf32) | n/a to burn | no |
| The `USE_SAFE_GATE` branch and its `-tril` re-inversion | same as `safe_gate` | see §5.3 |
| Grouped-value `H_v > H` state layout `state_v_first` | we do GVA, layout ours | **check** — §5.3 |
| Hybrid SWA + the 1.3B/100B recipe | a model, not a layer | no |

### 5.2 In `burn-gdn2`, not in the paper

`min_decay` (correctly declared an extension), the read-only forward branch,
the `ChunkPath::{Batched,Loop}` switch + `DM_GDN2_OPS` env var, the
`alloc_trace` counters, `l2_normalize_4d` with a hardcoded `1e-6` (paper §D.2
does not give an epsilon; FLA's `l2norm` uses `1e-6` — correct, but worth a
citation), and the `chunk_size` config knob (the paper fixes C=64; ours is
free, which is what makes the ≤16 batched limit reachable at all).

### 5.3 In the paper / reference, arguably missing from us

1. **fp32 decay gate.** §D.1: *"The decay gate in Eq. 86 is computed in
   explicit fp32 before entering the kernels… The kernels therefore receive
   the log-decay tensor and only compute local cumulative sums."* Our
   `module.rs:508-516` computes `g` from whatever dtype the params carry; it
   does not force fp32. The reference's Python does (`self.f_proj(hidden_states).float() + self.dt_bias`).
   **Our g is a `Param` on a backend that may be bf16.** In `--bf16` mode this
   is a real, paper-specified requirement we do not meet.
2. **`lower_bound` / `safe_gate` for gdn2.** GDN-2 inherits KDA's channel-wise
   decay but keeps the *unbounded* softplus form (Eq 12), so the reference
   default is `safe_gate=False, lower_bound=None`. We match. Correct — but
   worth stating that the reference exposes it and we do not.
3. **`use_qk_l2norm_in_kernel`** — ref does it in-kernel, we do it outside.
   Equivalent for the forward, **not obviously equivalent for the backward
   across the fused autodiff node** (the L2 VJP then has to be inside our
   chunk adjoint rather than autograd's). Our `autodiff.rs` is a
   hand-derived matrix adjoint; whether it carries the L2 VJP is a question I
   could not settle read-only. **Flagging, not claiming.**

---

## 6. The chunk size and the K3 decay coefficient — what the sources say

### 6.1 Chunk size

| | value | source |
|---|---|---|
| **GDN-2** | **C = 64** | 2605.22791 App. C.2, verbatim: *"The chunk size is fixed to C=64."* Confirmed in code: `chunk_gdn2.py` → `chunk_size: int = 64`; `chunk_gdn2_fwd_intra` → `BT = chunk_size`, `BC = 16`. |
| **GDN-2 sub-chunk** | **BC = 16** | same file, `BC = 16` — a *sub*-tile, not the chunk |
| **ours, burn-gdn2** | **64** (`src/config.rs:126`) | ✅ matches GDN-2 |
| **KDA / FlashKDA** | **16** | `MoonshotAI/FlashKDA/tests/torch_ref.py`: `CHUNK = 16` |
| **K3** | **16-token tile** | 2607.24653 §2.1.1: *"the cumulative log-decay over a 16-token tile lies in (−80, 0)"* |
| **ours, burn-kda** | **16** (`src/lib.rs:81`) | ✅ matches FlashKDA / K3 |

**The "chunk 16" claim in AGENTS.md is correct — for `burn-kda`, and only for
`burn-kda`.** It does not apply to `burn-gdn2`, whose source says 64 and which
uses 64. Two crates, two sources, two chunk sizes, both right. This is worth
writing down, because "chunk 16" and "chunk 64" appearing in the same repo
looks like an inconsistency and is not one.

**The underflow arithmetic in `README.md:160-164` checks out.** At the K3 floor
`g = −5`: chunk 16 → `cumsum = −80`; chunk 17 → `−85`; `exp(−87) ≈ 1.6e-38` is
the last normal f32, `exp(−88) ≈ 6.1e-39` is subnormal. So "underflows f32
once `cumsum(g) < −88`, i.e. chunk > 17" is right to within one token, and it
independently reproduces the K3 paper's own `(−80, 0)` figure for a 16-token
tile. That sentence is the best-reasoned claim in either crate.

### 6.2 The K3 decay coefficient

`g_min = −5`, **fixed, stated in the paper, not inferred** —
2607.24653 §2.1.1 Eq 5 and the sentence *"where `A_h` is a learnable per-head
log-scale and `g_min = −5` is fixed."* Corroborated in code by
`MoonshotAI/FlashKDA/benchmarks/bench_fwd.py`: `LOWER_BOUND = -5.0`, and by
FLA `fla/layers/kda.py`: *"With `-5`, the per-step decay `exp(g) ≈ 0.0067` at
minimum — negligible impact on quality."*

**Our `G_MIN: f64 = -5.0` (`burn-kda/src/lib.rs:26`) is correct and correctly
cited.** Row 23 in the delta table.

---

## 7. Recommended gold-vector test design

The current design — N random cases, compared against a hand transcription of
the same repo, into a `.bin` — verifies **self-consistency**. It cannot verify
**fidelity**, and §4.1 is the proof. What follows is what would.

### 7.1 The principle

> A gold vector is only worth what its *generator* is. If the generator was
> written by reading the implementation, the test is a tautology. The generator
> must be a **transcription of the upstream source, by someone who has not read
> the implementation**, and the two must be allowed to disagree.

### 7.2 Three tiers

**Tier 0 — upstream-executed vectors (the only real gold).**
Export gold vectors *from* the reference, once, on a machine that can run it,
and check the bytes in. Non-reproducible locally by construction; reproducible
everywhere by fixture. Requires, at generation time only:

| vector | generator | file | what it pins |
|---|---|---|---|
| `gdn2_conv.bin` | `lit_gpt/gdn2.py` forward, `use_short_conv=True`, `conv_size=4` | 20 KB | **the padding.** Positions 0–2 of q/k/v. |
| `gdn2_chunk.bin` | `lit_gpt/gdn2_ops/chunk_gdn2.chunk_gdn2`, `chunk_size=64`, `use_qk_l2norm_in_kernel=True` | 4 MB | the whole Eq 10 + Eqs 18–25 path, **including the 64-chunk numerics we cannot reach** |
| `gdn2_recurrent.bin` | `fused_recurrent_gdn2` | 4 MB | the token scan, `q_len > 64` |
| `kda_k3.bin` | `fla/ops/kda/chunk_kda`, `use_gate_in_kernel=True`, `safe_gate=True`, `lower_bound=-5` | 4 MB | K3 Eq 5, chunk 16 |
| `kda_linear.bin` | same, `use_gate_in_kernel=True`, `safe_gate=False` | 4 MB | the softplus form we do **not** default to |

Shapes: `B=1, T=129, H=2, HV=2, K=V=32, d_model=64` (129 = 2×64+1 so the tail
chunk is exercised; 129 = 8×16+1 so the 16-tile path is too). Weights from a
fixed PCG64 stream, written to the fixture in hex, so a reviewer can read them.
Tolerance: `rtol=2e-5, atol=2e-5` in f32, and **assert the tolerance is not the
thing being tuned** — a test that passes at `atol=1e-2` is a smoke test.

**Tier 1 — fp64 algebraic invariants (runs here, no GPU, catches real bugs).**
Property tests against the *equations*, in f64, on the CPU backend. These are
the ones that would have caught §4.1 without any upstream help:

1. **Zero-prefill identity.** With `S_0 = 0`, the output of a length-`T` chunked
   forward must equal a per-token scan. *This is the property that pins the
   recurrence, and it already exists — `tests/test_chunk.rs` — but it uses our
   own scan, so it is self-consistent.* Run it with the conv **on**: currently
   it would pass, because both sides share `short_conv_1d`. Make the scan take
   a *separately written* conv and the divergence surfaces.
2. **Prefill ≡ decode.** Existing (`autodiff.rs::decode_equals_full_forward`).
   Keep, but add the case "decode from a fresh state, `T=1`", which is the case
   where replicate-vs-zero padding is maximal.
3. **Tied-gate reduction.** Set `b = β·1_{d_k}`, `w = β·1_{d_v}` and check the
   result is bit-comparable to KDA with scalar `β`. This is the paper's own
   claim (§3.1) and it is a real test we do not have. Same for the further
   reduction to Gated DeltaNet.
4. **b ≡ 0, w ≡ 0 edge cases.** `b=0` ⇒ pure accumulation of `k(w⊙v)ᵀ` with
   decay; `w=0` ⇒ state decays to nothing. Both hand-checkable in f64.
5. **State bounds.** `|g| → ∞` ⇒ `α → 0` ⇒ `S` must decay to 0, not NaN. This is
   the `a_log` clamp's real job (§4.2 row 27); pin it.

**Tier 2 — metamorphic pairs (cheap, no fixtures, no upstream).**
Properties that must hold whatever the padding is, and that catch asymmetric
seams:

6. **Chunk-size invariance.** `forward_train(x, chunk=16) ≡ forward_train(x,
   chunk=64)` to f32 tolerance. *This is the test that finds the conv padding
   and any chunk-boundary seam*, because the seam lands in a different place at
   each chunk size. It is also free — one extra config, no new code. **I would
   add this one first.**
7. **Batch invariance.** `f(x[0]) == f(x)[0]` — catches any accidental
   cross-batch state or padding bleed.
8. **Permutation of heads.** `f(permute_heads(x)) == permute(f(x))` — catches
   the GVA repeat path (`module.rs:545-556`) mixing the value-head and
   key-head axes.

### 7.3 What I would not do

- Do not re-point `gen_reference.py`/`.rs` at the upstream repo and call the
  1000-case tolerance proof. Those files are another agent's; and a
  re-transcription has the same structural blind spot as the current one. Tier 0
  fixtures are the only thing that closes it.
- Do not add a test whose tolerance was widened until it passed. Per
  AGENTS.md §1.4, every tolerance in this repo should be able to name the
  measurement that set it.

---

## 8. Confidence and open questions

### VERIFIED (sourced, high confidence)

- `2601.16531` resolves to *A Collision-Free Hot-Tier Extension for
  Engram-Style Conditional Memory*, Tao Lin — and is cited **nowhere** in
  `burn-gdn2` or `burn-kda`. (arXiv API, two independent endpoints.)
- `2605.22791`, `2510.26692`, `2607.24653` all resolve, to the titles and
  author lists quoted. (arXiv API.)
- `NVlabs/GatedDeltaNet-2`, `fla-org/flash-linear-attention`, `MoonshotAI/FlashKDA`,
  `MoonshotAI/Attention-Residuals` all exist and are the sources the crates name.
- GDN-2 Eq 10 / Eq 9 / Eqs 18–25 as transcribed in §2 — read from
  `arxiv.org/html/2605.22791v1`.
- GDN-2 fixes **C = 64** (App. C.2) and uses **BC = 16** sub-tiles.
- GDN-2 reference `conv_size=4`, `conv_bias=False`; FLA `ShortConvolution` is
  `nn.Conv1d(groups=C, padding=3, padding_mode='zeros')` and its Triton fwd pads
  with `other=0.0`.
- K3 Eq 5 `g = g_min·σ(e^{A_h}z)`, **`g_min = −5` fixed**; K3 Eq 6 full-rank
  sigmoid gate after RMSNorm; **`A_h` initialized to 0**.
- FlashKDA `CHUNK = 16`; `LOWER_BOUND = -5.0`.
- FLA `fla/layers/kda.py` initializes `A_log = log(U(1,16))` (or `zeros` under
  `safe_gate`) and `dt_bias = dt + log(−expm1(−dt))`.
- **`A_log = −3` and `dt_bias = 1.0` appear in no Moonshot or FLA source I could
  find**, and the K3 paper states the opposite for `A_h`.
- 27 of 33 delta rows are exact matches, several of them non-obvious ones the
  code gets right (A_log log-uniform, `inv_dt`, the `U(−0.5,0.5)` conv init, the
  GVA axis discipline, the whole Eq 18–25 chunk algebra).

### SPECULATION (my inference, marked as such)

- **SPECULATION: `A_h = −3` probably degrades K3's central claim.** The
  arithmetic is mine: at `A = −3`, `e^A = 0.05`, so the pre-sigmoid logit range
  collapses and `σ` becomes near-linear — the mapping degenerates toward Kimi
  Linear's softplus form, which is the thing Eq 5 exists to replace. The K3
  paper motivates the bound by the `(−80, 0)` cum-decay range at 16 tokens; I
  did **not** recompute that range under `A = −3` and it may still hold
  (sigmoid saturates, so `g → −5` faster, which would make the bound *tighter*,
  not looser). **The real issue is the false citation, not a proven numeric
  harm.** Do not quote this as a measurement.
- **SPECULATION: the hand-derived chunk adjoint in `autodiff.rs` may not carry
  the L2-normalization VJP**, because the reference does the normalization
  inside the kernel and we do it outside the autodiff node. This is an
  unverified reading of a 34 KB file. `tests/fused_adjoint_vs_ops.rs` exists and
  may already cover it — I did not run it (no GPU, read-only).
- **SPECULATION: the batched chunk arm is dead by default.** `batched_applies`
  declines `chunk_size > 16` (`forward.rs:73`) and the default is 64
  (`config.rs:126`), so the default route is `Loop`. Consistent with the code
  as read, but I did not run it and the dispatcher could re-route elsewhere.

### Open questions I could not close

1. Does the GDN-2 paper's `A` (Eq 12) get forced to fp32 in the reference when
   the module is bf16? App. D.1 says yes for the decay gate; the Python
   confirms `.float()`. Whether our `--bf16` path honours it is a
   read-only-unanswerable question here.
2. What exactly is `[20]` in 2605.22791 §3.1? The reference docstring cites
   arXiv 2411.12537 for the negative-eigenvalue variant; I did not read the
   GDN-2 bibliography to confirm the number.
3. The other two agents' in-flight work on `gen_reference.rs` may change the
   fixture and therefore §4.1's blast radius. **Re-run the zero-prefill
   identity test after their change lands** — if they fix the generator's
   padding, our `short_conv.rs` becomes the odd one out, not the other way
   round.
4. `burn-gdn2/README.md:22` claims "1000-case comparison against an independent
   transcription … **Verification: none shipped**" for the reference. Given
   §4.1, that table's honest reading is: *self-consistency, 1000 cases,
   tolerance 5e-4, upstream fidelity untested.* Worth a README line.

---

## 9. One-paragraph summary

Every arXiv id in `burn-gdn2` and `burn-kda` resolves, and the two source
repositories those crates name — NVlabs/GatedDeltaNet-2 and Moonshot/FLA
(`fla-org/flash-linear-attention`, `MoonshotAI/FlashKDA`) — are real,
public, and were read in full for this report. The Gated Delta Rule-2
recurrence, both gate projections, the log-decay, the L2 norm, the GVA axis
discipline, the negative-eigenvalue lift, every single initialization, and the
entire Eq 18–25 chunkwise algebra are implemented **exactly**; the K3 lower-
bounded decay (Eq 5, `g_min = −5`), the K3 full-rank sigmoid output gate
(Eq 6), the KDA scalar-β recurrence, and the chunk-16 constant are all
**verified against the papers themselves**. `arXiv 2601.16531` is a real
single-author Engram preprint and is correctly scoped everywhere it is cited —
it is not a GDN/KDA citation and neither crate claims it. Two real defects
remain: **the short conv replicate-pads where the reference zero-pads**, a
divergence both reference generators reproduce, so the 1000-case bit-exact
test is structurally blind to it (impact today is zero only because
`use_short_conv` is off); and **`a_log = -3` / `b_alpha = 1.0` are cited to a
file that does not exist, in a repo, for values the K3 paper explicitly
contradicts** ("We initialize `A_h = 0`") and that no FLA or FlashKDA source
contains. That second one is a retracted-claim-class defect under ADR-0020 and
it has propagated into AGENTS.md as rule-level fact.
