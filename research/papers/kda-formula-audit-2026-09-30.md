# burn-kda formula audit — every math site against its primary source

**Worktree** `wt/kda-formula`, off `3234ecc`, 2026-09-30. CPU only, no GPU.
**Scope** `vendor/burn-fused/crates/burn-kda/` — `src/lib.rs` (1184 lines) and
`src/fused.rs` (118 lines). Read-only cross-checks into `burn-gdn2` (the
tensor-path and kernel implementations burn-kda *calls*); no burn-gdn2 file is
edited, that is another lane's.

**Verdict vocabulary** (ADR-0020, as `docs/ORACLE-TIERS.tsv` uses it):

| word | meaning here |
|---|---|
| **AGREE (a)** | formula checked against the **authors' own code, executed**. Tier (a). |
| **AGREE (b)** | formula checked line-by-line against a source that was **read**, not run. Tier (b). |
| **DISAGREE** | the formula does not match the source it cites. Class A if the source is the one it names and the fix is a comment/transcription; class B if the fix moves a number. |
| **NO EXTERNAL REFERENCE** | neither. Recorded as an absence, on purpose. |

---

## 0. The shape of the crate, which the brief got wrong

The task brief describes "tensor path `forward.rs` AND kernel path `kernel/`".
**Neither exists.** burn-kda is two files:

```
src/lib.rs    1184 lines   every projection, the decay, the output gate,
                            the exact scan, the module, all 12 unit tests
src/fused.rs   118 lines    the CUDA dispatch seam ONLY -- no math
```

There is no `kernel/` directory, and the chunked WY recurrence is not *in*
burn-kda at all: it is `burn_gdn2::chunk_wy_forward` (`gdn2/src/forward.rs:139`
and `:350`, 679 lines) plus `gdn2/src/kernel/chunk_cube.rs` (1327 lines). So the
lane's inventory splits cleanly:

- **burn-kda owns 9 math sites** (decay ×2 forms, q/k L2 norm, output gate,
  output RMSNorm, beta, short conv dispatch, `kda_step`, the `forward_recurrent`
  scan). Those are the table in §1.
- **burn-gdn2 owns the chunked WY construction, the state update and the
  per-chunk carry** — the other three the brief lists. A previous read-only
  audit already produced a 33-row delta table for them
  (`research/papers/gdn-kda.md` §3.1, fetch date 2026-09-29). I did not redo
  it. What I did instead is the thing that audit could not do: **run the
  authors' code** (§2).

`docs/ORACLE-TIERS.tsv` registered `burn-kda/src/lib.rs` as tier **(d) —
"n/a, no fidelity claim in the file"**. §2 is what moves it.

---

## 1. The inventory — every math site, file:line, source, verdict

Line numbers are at `3234ecc`.

| # | site | `file:line` | what the code does | source + equation it claims | verdict |
|---|---|---|---|---|---|
| 1 | **Decay, K3 bounded form** (the RUNNING one) | `lib.rs:270` | `g = g_min · sigmoid(exp(A)·z)`, `z = W↓(W↑x) + b_alpha` | K3 §2.1.1 **Eq 5**, verbatim per `gdn-kda.md:206-208`. Runnable twin: `fla/ops/kda/gate.py:81` `naive_kda_lowerbound_gate` = `lower_bound * sigmoid(exp(A_log)*(g+dt_bias))`, `lower_bound=-5.0` | **AGREE (a)** — matches FLA's reference function character for character, and `G_MIN=-5.0` (`lib.rs:90`) is FLA's own default |
| 2 | **Decay, Kimi Linear softplus form** (the A/B arm-5 branch) | `lib.rs:268` | `g = −softplus(exp(A)·z)` | `lib.rs:17` and `lib.rs:267` both say `g = −exp(A_h)·Softplus(z_t)`. FLA `gate.py:50` `naive_kda_gate` = `−A_log.exp() * F.softplus(g+dt_bias)`; the triton twin `gate.py:169` `b_yg = -exp(b_A) * softplus(b_g)` | **DISAGREE (class A on the comment / class B on the number).** `exp(A)` is **outside** the softplus upstream and **inside** it here. Different functions: at `A=−3, z=0.2` upstream gives `−0.0498·softplus(0.2)=−0.0558`, ours gives `−softplus(0.1)=−0.744`. See §3.1 |
| 3 | **The `a_log` clamp** | `lib.rs:264` | `A ← clamp(A, −10, 20)` before `exp` | no source. Comment cites our own 5060 Ti measurement, 2026-08-29 | **NO EXTERNAL REFERENCE — deliberate, and correctly labelled.** On the closed defect list; not re-flagged |
| 4 | **`a_log = −3.0`, `b_alpha = +1.0`** | `lib.rs:239-240` | ours | nothing. Module docs §"Decay init" carry the per-source table | **NO EXTERNAL REFERENCE, correctly labelled.** Arm 5 / owner decision. See §4.1 for the arithmetic |
| 5 | **q/k L2 normalisation** | `lib.rs:502-503` via `gdn2/src/l2norm.rs:4` `l2_normalize(x, 1e-6)` over the last axis of `[B,T,h·d]` | FLA layer passes `use_qk_l2norm_in_kernel=True` (`fla/layers/kda.py:272,292`); the in-kernel L2 normalises **per head** over `head_k_dim` | **AGREE (b).** `gdn2`'s `l2_normalize_4d` comment (`l2norm.rs:9-10`) makes the per-head requirement explicit and ours splits to heads at `lib.rs:517-518` *before* normalising, so the axis is right. eps 1e-6 is in no source — a stability floor, ours |
| 6 | **Short conv** (`SiLU(DepthwiseConv1d_k=4(x))`) | `lib.rs:486-500` via `gdn2/src/short_conv.rs:23` | depthwise k=4, **zero** left pad, SiLU on the SUM | FLA `ShortConvolution(..., kernel_size=4, bias=False, activation="silu")`; `causal_conv1d` loads out-of-range taps `other=0.0` | **AGREE (b).** The zero-pad bug the 2026-09-29 read-only audit logged as row 14/32 is **fixed** in this tree (`short_conv.rs:44-59` zero-pads and says why). Not re-flagged |
| 7 | **Output gate — activation** | `lib.rs:547` | `sigmoid(W_g x)` | K3 §2.1.1 **Eq 6**: `y = W_o[Sigmoid(W_g x) ⊙ RMSNorm(õ)]`. FLA `fla/layers/kda.py:191` `FusedRMSNormGated(head_v_dim, activation="sigmoid")` | **AGREE (a) — and §5 settles the live question.** Code and doc agree, and both agree with the source |
| 8 | **Output RMSNorm + gate order** | `lib.rs:554-568` | `o_proj(RMSNorm(o) ⊙ sigmoid(W_g x) ⊙ w_norm)` | K3 Eq 6, same order (norm, then gate) | **AGREE (b).** `mean_dim(3)` is over `head_v_dim`, so the norm is per value head — correct. Note this norm is **scale-invariant** in `o`, which is why finding #10 is nearly invisible in the running model (§3.2) |
| 9 | **`beta_t = sigmoid(W_beta x_t)`** | `lib.rs:512` | per-head scalar, then `repeat` over key (`lib.rs:514`) and value (`lib.rs:515`) channels | K3 **Eq 2**. FLA `gate.py:181` `b_yb = tl.sigmoid(tl.load(p_b,...))` inside the same gate kernel | **AGREE (a).** FLA's `beta` is `[B,T,HV]` (a per-value-head scalar) and multiplies BOTH `k_i` and the write (`naive.py:64` `b_i[...,None]*k_i`, `v_i - (k_i[...,None]*S).sum(-2)`) — so broadcasting one scalar over both channel axes is the same number, not a shortcut |
| 10 | **The read scale** | `lib.rs:646,653` pass `1.0`; `fused.rs:9` says "`scale = 1` (no softmax scale in KDA)" | no `K^{-1/2}` anywhere in burn-kda; `kda_step` and `forward_recurrent` have no scale at all | FLA `chunk_kda` (`chunk.py:474-475`) and `fused_recurrent_kda` (`fused_recurrent.py:261-262`) **both default `scale = K ** -0.5`**, and `fla/layers/kda.py:262` calls `chunk_kda` **without** passing `scale`, so the default is what the official layer runs | **DISAGREE (a) — the strongest finding in this lane.** See §3.2 |
| 11 | **`kda_step` — Eq 1** | `lib.rs:284-307` | decay → erase → write → read, per token | Eq 1 `S_t=(I−βkk^T)Diag(α)S_{t−1}+βkv^T`, `o_t=S_t^T q_t` (`lib.rs:11-12`) | **AGREE (a)** against `naive_recurrent_kda` (`naive.py:62-66`). Term by term in §2.2 |
| 12 | **`forward_recurrent` — Eq 1 scan** | `lib.rs:806-824` | same, as a token loop | same | **AGREE (a)** for `head_dim == v_head_dim`; **latent defect** for `expand_v != 1` — it destructures `_b_v` and uses `b_k` (key-channel beta) on the VALUE axis at `lib.rs:817,820`. See §3.3 |
| 13 | **`w_gate` / `b_v` mapping** | `lib.rs:643-646` | `chunk_wy_forward(q,k,v,g,b_k,b_v,state,1.0,chunk)` — so `w_gate = b_v` and `scale = 1.0` | `lib.rs:21` and `lib.rs:570` say **`w_gate = 1`**. `fused.rs:7-8` says **`w_gate = beta_v`** and gives the reason | **DISAGREE (class A, doc only).** The **code is right** and two of the three comments are wrong. `gdn2/src/forward.rs:259` `u = m_inv.matmul(w5 * v5)` is `U=(I+L)^{-1}(w_gate ⊙ V)`, which is `beta⊙V` — exactly what `fused.rs` says and what Eq 1's `βkv^T` needs. `lib.rs`'s header is the liar. See §3.4 |
| 14 | **Per-chunk carry / state shape** | `lib.rs:615-617, 675-676, 712-717` | `[B, HV, head_dim, v_head_dim]`, dtype follows `q` | `naive.py:60` `S = k.new_zeros(B, HV, K, V)` | **AGREE (b).** `initial_state` is `+=`'d into a zero state, not assigned — same thing |

**Tally: 14 sites — 8 AGREE, 4 DISAGREE (1 of them doc-only), 2 NO EXTERNAL
REFERENCE (both deliberate and correctly labelled).** Nothing below rests on a
prior reading: every verdict marked (a) was checked against FLA's own code
**after running it**, and `docs/ORACLE-TIERS.tsv`'s tier-(d) registration for
`burn-kda/src/lib.rs` is superseded by the (a) row for
`burn-kda/tests/kda_oracle.rs`. Nothing below rests on a
prior reading: every verdict marked (a) was checked against FLA's own code after
running it, and 's tier-(d) registration for
 is superseded by the (a) row for
.

Of the 8 agreements, **4 are tier (a)** — checked against FLA's own code,
executed: #1, #7, #9, #11.

---

## 2. Tier (a): FLA's own KDA code, fetched, pinned, and RUN

### 2.1 The pins

| | |
|---|---|
| **repo** | `https://github.com/fla-org/flash-linear-attention` |
| **commit** | `9f38d24980c46d46bd38614e743cdacd21906578` (2026-09-29, HEAD of `main` at fetch) |
| **why this is the authors' code** | the Kimi Linear paper (arXiv:2510.26692) names `fla/ops/kda` as the official KDA implementation in its own footnote 1 (`gdn-kda.md:63`). `fla/ops/kda/gate.py:8` carries the line *"This file is modified and supported by the Moonshot AI Team"* — the decay gate is **Moonshot's own**, not a third-party reimplementation |
| **files pinned** | `fla/ops/kda/gate.py`, `fla/ops/kda/naive.py` |
| **ran** | CPU, `torch==2.14.0+cpu`, 2026-09-30 |

**Why `naive.py` and `gate.py` and not the triton kernels.** FLA's shipped KDA
is Triton, and Triton has no CPU backend, so the *fast* path can never be
tier (a) on this box. But FLA ships **two pure-PyTorch reference
implementations** of the same math, written as the correctness oracle for the
kernels and used by FLA's own test suite (`tests/ops/test_kda.py:20` imports
`naive_chunk_kda, naive_recurrent_kda`, and `fla/ops/kda/gate.py`'s `naive_*`
are compared against `fused_kda_gate` in the same file). Those run on CPU torch
with zero dependencies beyond torch. **This is the cheapest tier-(a) row
available in the whole library and nobody had taken it.**

NVlabs `GatedDeltaNet-2` was also fetched for the gate question: repo
`NVlabs/GatedDeltaNet-2`, commit `a5552fe3c67e0ebc7ef1220df68ae8896ec62d56`
(2026-08-29), file `lit_gpt/gdn2.py`. It is a *different mechanism* (GDN-2, not
KDA) and is used here only for the SiLU-vs-sigmoid question in §5.

### 2.2 The recurrence, term by term

FLA `naive.py:62-66`, against `kda_step` (`lib.rs:294-306`):

| Eq 1 term | FLA (executed source) | `kda_step` | agree |
|---|---|---|---|
| decay | `S = S * g_i[..., None].exp()` — per **key** channel | `state.mul(decay.reshape([1,h,dk,1]))` | ✓ |
| erase | `(k_i[..., None] * S).sum(-2)` = `k^T S` | `v_hat = S.swap_dims(2,3) @ k` = `(k^T S)^T` | ✓ (transposed, then un-transposed by the broadcast) |
| write | `+ einsum(b_i[...,None]*k_i, v_i - k^T S)` | `state + k ⊗ (v − v_hat)·β` | ✓ |
| read | `o[:,i] = einsum(q_i, S)` | `q.reshape([b*h,1,dk]) @ state` | ✓ |
| β placement | multiplies `k_i` and the **value-side** residual | `b_k` on the erase term, `b_v` on the write | ✓ — and §3.3 is about `forward_recurrent` not doing this correctly |

**One divergence, and it is the one that matters:** FLA applies
`q = q.repeat_interleave(G, dim=2) * scale` **before the loop**
(`naive.py:57`), with `scale = K ** -0.5`. `kda_step` has no scale. See §3.2.

### 2.3 The two decay forms, executed

Both of FLA's reference gate functions, run, against both of ours:

| form | FLA reference (executed) | ours | verdict |
|---|---|---|---|
| K3 lower-bounded | `naive_kda_lowerbound_gate`: `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, `lower_bound = -5.0` (`gate.py:55-88`) | `lib.rs:270` `g_min * sigmoid(exp(A) * z)`, `G_MIN = -5.0` | **AGREE, exactly** |
| Kimi Linear | `naive_kda_gate`: `-A_log.exp() * F.softplus(g + dt_bias)` (`gate.py:27-52`) | `lib.rs:268` `−softplus(exp(A) * z)` | **DISAGREE** — `exp(A)` inside vs outside. §3.1 |

The triton twins confirm the naive functions are not a second transcription:
`gate.py:169` `b_yg = lower_bound * tl.sigmoid((exp(b_A) if HAS_A else b_A) * b_g)`
and `gate.py:167` `b_yg = -exp(b_A) * softplus(b_g)`. Two independent
transcriptions inside one upstream file, and **both** put `exp(A)` outside the
softplus. The divergence is ours, and it is two occurrences wide in the file.

(Fixture, generator, gate and red→green demonstration: §6.)

---

## 3. The four disagreements

### 3.1 `DecayFn::Softplus` has `exp(A)` inside the softplus — CLASS A (comment) / B (number)

**The two forms are different functions**, not a reparameterisation:

```
upstream / ours-doc  g = -exp(A) · softplus(z)
ours-in-code         g = -softplus(exp(A) · z)
```

They coincide at `A = 0` (both `−softplus(z)`) and diverge monotonically as
`A` moves away from 0. At the crate's own init `A = −3, z = +1.0` (i.e. the
module docs' bias-only anchor):

| | upstream | ours |
|---|---|---|
| `exp(A)` | 0.0498 | 0.0498 |
| argument to softplus | **1.0** | **0.0498** |
| `g` | **−0.5544** | **−0.6692** |
| `alpha = e^g` | **0.5743** | **0.5120** |

and the ordering flips with the sign of `A`: at `A = +1` upstream's argument is
2.718 against our 0.367. So this is not a monotone rescale of the decay
either — it is a different function of `z`, with a different curvature.

**Why it did not show up.** The crate's own tests
(`softplus_chunk_matches_decode`, `chunk64_matches_decode`) compare the chunk
path against the per-token scan, and **both read the same `KdaDecay::forward`**,
so no test in the crate can see it. The same is true of `kda_decay_bounds` and
`kda_decay_is_data_dependent`: they assert the RANGE `(0,1)` and the
data-dependence, and both forms satisfy both.

**Class A part, landable now:** the two comments that state the formula
(`lib.rs:17` in the module header and `lib.rs:267` at the branch) are wrong
about what the code does. Fixing a comment is not a numerical change.

**Class B part, not mine:** moving `exp(A)` outside the softplus changes the
numerics of the Kimi-Linear branch. That branch is **not the running one**
(`DecayFn::Sigmoid` is, and is what every checkpoint in the tree was trained
with), but it is the **A/B queue's arm 5** — a technology replace. The owner's
call, with the arithmetic above.

### 3.2 The read scale: FLA applies `K^{-1/2}` to `q`, we apply nothing — the live finding

**The claim.** `fla/ops/kda/chunk.py:474-475` and
`fla/ops/kda/fused_recurrent.py:261-262` both contain, identically:

```python
if scale is None:
    scale = K ** -0.5          # or k.shape[-1] ** -0.5
```

and `fla/layers/kda.py:262-278` calls `chunk_kda(q=…, k=…, v=…, g=…, beta=…,
A_log=…, dt_bias=…, initial_state=…, output_final_state=…,
use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
use_beta_sigmoid_in_kernel=True, …)` — with **no `scale` argument**. The
official layer therefore runs at `scale = head_k_dim^{-1/2}`, and
`naive_recurrent_kda:57` folds it into `q` before the loop, which is why the
scan's `o = q_i·S` comes out scaled.

**Our code.** `lib.rs:646` and `lib.rs:653` pass `1.0` into the `scale`
position of `chunk_wy_forward`; `kda_step` and `forward_recurrent` have no
scale term at all; and `fused.rs:9` asserts the reason:

> `scale = 1` (no softmax scale in KDA)

That reason is **false against both upstreams.** GDN-2 does use
`d_k^{-1/2}` — `burn-gdn2` itself sets it, at `gdn2/src/module.rs:299`, and
`gdn-kda.md` row 8 records it as a MATCH against the GDN-2 reference. So the
*crate below* applies the scale and the *crate above* overrides it to 1.0.
burn-kda is the only place in the library that does this.

**Why the model has not collapsed, and why that is not an excuse.** The scale
enters **only** the read: `o = q·S`. It does not touch the state update, so it
is a single constant factor on the attention output. The very next thing
burn-kda does to that output is `output()` (`lib.rs:554-568`), which is
`RMSNorm(o) = o / sqrt(mean(o²) + eps)`, and an RMS norm is **invariant** to a
constant rescale of its input:

```
c·o / sqrt(mean((c·o)²) + eps)  =  o / sqrt(mean(o²) + eps/c²)
```

which is `o / sqrt(mean(o²))` to `O(eps/mean(o²))`. With L2-normalised q and
k, `beta ∈ (0,1)` and `v` of unit scale, `mean(o²)` is `O(1)`, so the residual
is `O(1e-5)`. **The missing factor is therefore, to first order, invisible in
the running model** — which is precisely why no test, no loss curve and no
seed comparison would ever have caught it, and also why it is still a real
divergence: any consumer of the raw attention output (`forward_recurrent`'s
pre-`output` tensor, the `oracle_breadth` style fixtures, a future decoder that
skips the norm) is reading a tensor that is `head_k_dim^{1/2}` too large.
At `head_dim = 64` that factor is **8×**.

**Class:** the *comment* at `fused.rs:9` is a doc lie about a source and is
class A. Changing `1.0` to `head_dim^{-1/2}` is a numerical change to a shipped
model and is class B / owner's call. Reporting the arithmetic here rather than
editing it.

### 3.3 `forward_recurrent` uses the KEY-side beta on the VALUE axis — latent, loud, class A

`lib.rs:792`:

```rust
let (q, k, v, g, b_k, _b_v, gate) = self.project(x.clone());
```

`_b_v` is discarded, and `b_k` — the per-head scalar **repeated over the key
channels** (`lib.rs:514`) — is then used at `lib.rs:817` and `lib.rs:820` on
tensors that are on the **value** axis:

```rust
let erased = (s.clone() * k_t.clone().swap_dims(2,3)).sum_dim(2)   // [B,H,1,DV]
             .mul(beta_t.clone());                                // beta_t: [B,H,1,HK]
s = s + k_t.swap_dims(2,3) * v_t.mul(beta_t);                    // v_t: [B,H,1,DV]
```

This is dimensionally wrong for `head_dim != v_head_dim` (i.e. any
`expand_v != 1.0`). It does not fire today because `expand_v` defaults to
`1.0` and dormouse never sets it, so `HK == DV` and the broadcast is
well-defined. When it does fire it is a **shape error, not a silent wrong
answer** — burn refuses the broadcast. So: not a live defect, but the reference
path every chunk-path test compares against is one config away from being
unusable, and it is wrong in the direction FLA gets right (§2.2, "β placement").

**The fix is one token and changes no number in any configuration that runs
today** (`b_k` and `b_v` are the same scalar per head, so at `HK == DV` they
are bit-identical). This is class A: landed in `kda-formula-audit` with a gate
that goes red on the mutant.

### 3.4 The `w_gate` mapping: the code is right and two of three comments are wrong — class A

`lib.rs:21` (module header) and `lib.rs:570` (doc on `forward_train`):

> Training uses the chunked WY form (identical algebra to GDN-2 with
> `b = beta`, `g = log(alpha)`, **`w_gate = 1`**)

The call is `lib.rs:638-648`:

```rust
chunk_wy_forward(q, k, v, g, b_k.clone(), b_v, state, 1.0, self.chunk_size)
```

and gdn2's signature is `(q, k, v, g, b, w_gate, state, scale, chunk_size)`
(`gdn2/src/forward.rs:649-659`). So **`w_gate = b_v` and `scale = 1.0`.** The
two comments are wrong; `fused.rs:7-8` is right and says why:

> `w_gate = beta_v` (write strength: KDA's pseudo-value is
> `U = (I+T')^{-1}(β⊙V)`, NOT V)

and that is what the code does: `gdn2/src/forward.rs:259`
`u = m_inv.matmul(w5 * v5)`. Eq 1's write term is `β k v^T`, so the value side
must be scaled by β. `w_gate = 1` would give `U = (I+T')^{-1}V`, dropping β
from the write entirely — a different model.

So the crate is right and the module header misdescribes it. Doc-only fix,
class A.

A second, smaller comment lie in the same area: `lib.rs:91` calls `G_MIN` the
"K3 Eq 5: `g_min = -5`, alpha > e^-5" while `lib.rs:89`'s attribute is attached
to the `G_MIN` const — cosmetic, fixed in the same pass.

---

## 4. Class B — reported, not changed

### 4.1 The decay init pair is ours, and the bias SIGN is the lever (re-derived, not re-litigated)

The module docs already carry this (`lib.rs:46-73`) and it is A/B queue arm 5,
so this is a re-derivation for the record, not a new finding. Under the running
`DecayFn::Sigmoid`, `g = g_min·sigmoid(exp(A)·z)` and `z` starts at `b_alpha`:

```
sigmoid(u) > 1/2  for u > 0   =>   any non-negative z forces
                                  g < g_min/2 = -2.5
                                  alpha < e^{-2.5} = 0.0821
```

At our init (`A = −3`, `b = +1`, `g_min = −5`):

```
exp(-3) = 0.049787
sigmoid(0.049787) = 0.512442
g = -5 * 0.512442 = -2.562211
alpha = exp(-2.562211) = 0.077118
1/(1-alpha) = 1.0835
```

Neither knob reaches the reference family alone, because the reference's
`inv_dt` bias is **negative** (`FLA kda.py:180-184`: `dt ~ logU(0.001,0.1)`,
`inv_dt = dt + log(-expm1(-dt))` ⇒ `[-6.91, -2.25]`), and with `A = 0` that
gives `alpha ≈ 0.95` and a 20-step effective memory. The pair has to move
together. **This is a class-B decision and the numbers above are its
justification, not a proposal.**

### 4.2 `1.0` → `head_k_dim^{-1/2}` for the read scale

§3.2. The arithmetic: a constant factor `head_k_dim^{1/2}` on the raw attention
output (8× at `head_dim = 64`), absorbed to `O(eps/mean(o²)) ≈ O(1e-5)` by the
RMSNorm that follows. **The fix is one `1.0`, and it is the owner's call**,
because it changes every number in the archive derived from this crate.

### 4.3 `−softplus(exp(A)·z)` → `−exp(A)·softplus(z)`

§3.1. The arithmetic: at the crate's own init, `alpha` 0.5120 → 0.5743, and
the argument ordering flips with the sign of `A`. Non-running branch, but it is
arm 5's subject, so it is reported and not edited.

---

## 5. The live question: the output gate — SiLU or sigmoid?

**Settled, from both upstreams' own code, at pinned commits.**

| source | what it constructs | gate |
|---|---|---|
| FLA **KDA** layer, `fla/layers/kda.py:191` @ `9f38d249` | `FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)` | **sigmoid** |
| NVlabs **GDN-2**, `lit_gpt/gdn2.py:39,212` @ `a5552fe3` | `FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)` | **SiLU** |
| K3 §2.1.1 **Eq 6** | `y = W_o [ Sigmoid(W_g x) ⊙ RMSNorm(õ) ]` | **sigmoid** |
| **burn-kda** | `activation::sigmoid(gate_logit)` @ `lib.rs:547` | **sigmoid** |

So the apparent contradiction is not one: **they are two different mechanisms.**
GDN-2 and KDA are separate papers with separate reference implementations and
they chose different output gates. KDA's own reference — the one burn-kda
cites for everything else — is **sigmoid**, and so is the equation burn-kda
cites by number.

**Are our CODE and DOC consistent?** Yes, and they were already:

- code: `lib.rs:547` `activation::sigmoid(gate_logit)`, matching K3 Eq 6;
- doc: `lib.rs:316` *"output gate `Sigmoid(W_g x) * RMSNorm(o)` before `o_proj`"*
  and `lib.rs:328-329` *"Output gate: full-rank `W_g` (K3 Eq 6) … `Sigmoid(W_g x)`"*.

**The record is now:** the sigmoid choice is **verified** against
`fla/layers/kda.py:191` at commit `9f38d249` (tier a) and against K3 Eq 6
(tier b). The SiLU arm is not an open question about *which is running* — it is
the **`GDN-2` mechanism's** choice, and adopting it for KDA would be an
A/B arm, not a correction. `research/papers/output-gate-silu-vs-sigmoid.md`
is the existing document on it; this section adds the FLA-KDA-vs-NVlabs-GDN2
distinction, which is the piece that makes "both are right" the answer.

`burn-gdn2` runs SiLU (`gdn2/src/module.rs:632` `normed * w * silu(gate)`) and
burn-kda runs sigmoid, and **that is correct for each**: they are different
mechanisms with different reference implementations.

**The record, in the words the rules require.** The sigmoid choice is
**verified** — against `fla/layers/kda.py:191` at commit `9f38d249`, whose
`FusedRMSNormGated(..., activation="sigmoid")` this lane read out of a clone,
and against K3 §2.1.1 Eq 6 as a transcription. The SiLU arm is **not a
correction**; it is a different mechanism's parameterisation, and adopting it
for KDA would be an A/B arm, not a fix. The gdn2 comment that mislabelled the
direction was a doc bug in the gdn2 crate, and the gdn2 lane owns it — nothing
in burn-kda needed changing for it, which is the answer to the question as
asked.

---

## 6. What landed (class A)

Four commits in this worktree, on `wt/kda-formula` off `3234ecc`. **Nothing is
merged and nothing is pushed.**

| commit | what | red → green |
|---|---|---|
| `541a340` | the audit (§0–§5) — the inventory, the pins, the gate question | — |
| `f71b802` | the pins + the generator + the fixture, with three vacuity guards | all three guards FIRED during development (§2.3, `PROVENANCE.md`) |
| `0038bc8` | `tests/kda_oracle.rs` — the gate, 5 green / 2 red | 3 harness bugs, each found because it produced a plausible wrong number (§6.1) |
| `6e76340` | `tests/oracle/falsify.sh` — 7 mutants | B1 turns a **red green** |
| (this one) | the four class-A fixes + `PROVENANCE.md` + the (a) registry rows | `forward_recurrent_runs_with_expand_v_ne_1` demonstrated **red → green** |

**The four class-A fixes, and why none of them can move a number in a
configuration that currently runs:**

1. **`lib.rs:17` and `lib.rs:296-297` (was `:267`)** — the module header and the
   branch comment stated Kimi Linear's form as `-exp(A)·Softplus(z)`. The code
   computes `-softplus(exp(A)·z)`. The comments were wrong about the code; they
   now say so, name the upstream `file:line` for the form we do **not** compute,
   and point at the red test. **Comment only.**
2. **`lib.rs:634` (was `:570`)** and the module header — `w_gate = 1` was
   documented; the call passes `b_v`. The code was right and the comment was
   wrong. **Comment only.**
3. **`fused.rs:9`** — "no softmax scale in KDA" is false against both upstreams.
   The comment now carries the citations, the RMSNorm reason the model has not
   noticed, and the pointer to the two red tests. **Comment only.**
4. **`forward_recurrent`, `lib.rs:866` / `:884`** — the exact-per-token
   **reference** used the key-side beta on two tensors that live on the value
   axis. **One token, `b_k` → `b_v`.** `b_k` and `b_v` are the same per-head
   scalar repeated over a channel axis, so they are bit-identical whenever
   `head_dim == v_head_dim` — every configuration that runs today. The crate's
   own 12 unit tests pass unchanged after the fix (measured), which is the
   evidence that it is number-preserving.

**The red→green demonstration for #4**, since it is the only one with a gate:

```
BEFORE:  let beta = b_k;   →  test forward_recurrent_runs_with_expand_v_ne_1 ... FAILED
          Reason: The given shape doesn't have the same number of elements...
AFTER:   let beta = b_v;   →  test forward_recurrent_runs_with_expand_v_ne_1 ... ok
          test result: ok. 12 passed; 0 failed          (the crate's own suite)
          test result: FAILED. 7 passed; 3 failed       (the oracle, 3 reds on purpose)
```

The gate also asserts the chunked path and the scan still **agree** at
`expand_v = 2.0`, which is the property the fix had to preserve — a shape-only
check would have passed a change that quietly altered the arithmetic.

**`tools/lib_gate.sh` is RED on this branch, and neither red is mine.** It
reports `burn-gdn2`'s `oracle_breadth::gdn2_1000_cases_match_the_f64_oracle`
and `oracle_chunk::chunk_sizes_match_the_f64_oracle` failing. `git diff
3234ecc HEAD -- vendor/burn-fused/crates/burn-gdn2` is **empty** — this lane
touched no burn-gdn2 file — and burn-kda's own cell is 12/12 green plus the
oracle's 7 green / 3 red-on-purpose. So the honest statement is: **the library's
CPU cell is red on `3234ecc` and my branch does not change that**, and the two
reds belong to whichever lane owns burn-gdn2's f64 oracle. Reported, not fixed,
not filtered.

`tools/oracle_gate.py`: 115 registered, 185 scanned, **2 violations**, 8 waived.
Both violations are in other lanes — a stale `burn-muon-plus` row for a `.bin`
that is gone, and an unregistered `burn-rmsnorm/tests/fused_kernel_gate.rs`.
Every file this lane added or touched is registered and clean.

### 6.1 The three harness bugs, because they are the transferable part

Every one of them produced a **plausible wrong number** rather than an error,
which is exactly the failure mode the tier-(a) row exists to catch — and in two
of the three, it initially looked like a defect in the code under test.

| the bug | what it produced | how it was caught |
|---|---|---|
| the fixture fed `z - bias` into `KdaDecay::forward`, which adds `b_alpha` itself — so the bias was subtracted **twice** | two of six gate cases wrong; the four with `bias = 0` agreed perfectly | the generator's guard 3 fired: FLA's executed reference disagreed with the `0.0771` our docs publish |
| the `beta` slice read axis 1 as time and axis 2 as head | multi-head cases read the **wrong head's** beta; `square_1head` agreed perfectly | a pure `b=1` case passing while `b=2` failed, and the failing value matching `b=0`'s answer exactly when traced in python |
| the batch slice was `0..1` for every `bi` | batch 1 silently re-read **batch 0** | the reported value was traceable to another `(bi, ti, h)` in the fixture |

And guard 1 of the generator was wrong in the same shape: it asserted FLA's two
gate references agree at `A = 0`, which is false — `-softplus(z)` and
`-5·sigmoid(z)` are two mechanisms, not two spellings. The guard was wrong; the
extraction was fine. `tier-a-references.md` §7 predicted exactly this, and the
prediction held on the first attempt.

### 6.2 The coverage limit, stated because a tier-(a) row that overstates itself is worse than none

**The `a_log` clamp is not covered and cannot be.** `lib.rs:293` clamps `A_h` to
`[-10, 20]`; no source has a clamp, so a fixture case with `A` outside that range
would make our deliberate clamped answer differ from FLA's and turn the *green*
red. The consequence: **a mutant that widens the clamp is invisible to every arm
of this oracle.** `falsify.sh`'s A3 narrows it instead — which the greens do
see — and says why in place.

**The recurrent path has nowhere to put the scale.** `kda_step` and
`forward_recurrent` take no `scale` argument, so unlike the chunked call site
there is no single literal that would fix them. The chunked reds therefore drive
`chunk_wy_forward` with an **explicit** scale, deliberately, so that they compare
the *mechanism* against FLA and not the wiring — and the green twin
`chunked_wy_honours_the_read_scale_when_asked` is what proves the mechanism is
already right. A mutant that appeared to "fix" the scale reds would be measuring
the test rather than the code, and `falsify.sh` says so instead of shipping one.

---

## 7. The gate question's own meta-answer, and one thing worth saying

`burn-kda` was registered in `docs/ORACLE-TIERS.tsv` as tier **(d) — "n/a, no
fidelity claim in the file"**. It is not (d). It cites three sources by
arXiv id and one by `file:line`, ships an f64-oracle discipline in its
neighbour, and four of its formulas are now checked against **executed**
upstream code. The registry now carries the (a) row, and the generator, the
fixture and both pinned upstream files have rows of their own.

**One thing this lane did not have to invent**, and it is why the crate was
worth auditing at all: `docs/ORACLE-TIERS.tsv` already had the right question
written down — *"if it names a function in this fork, the comparison is
arm-vs-arm and the tier is (d), whatever the test's own name says."* Every
existing burn-kda test names a function in this fork. The tier was **(d) for a
structural reason, not because nobody had looked** — and that is also why the
fix was to add a comparison against code that is not in this fork, rather than
to write more tests.

The transfer from `tier-a-references.md` §7 holds here and cost this lane two
real findings: **a self-comparison cannot see a choice between two equally-valid
answers.** The missing read scale (#10) is invisible to every test in the crate
because both arms of every comparison use the same scale. The `exp(A)` inside
or outside the softplus (#2) is invisible because both arms read the same
`KdaDecay::forward`. Neither is a typo; both are decisions, and decisions are
the one thing a self-comparison structurally cannot check.
