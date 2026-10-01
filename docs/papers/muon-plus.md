# Muon+ — arXiv:2602.21545 — verification, literal transcription, delta vs `burn-muon-plus`

**Fetch date: 2026-09-29.** Read-only pass over `vendor/burn-fused/crates/burn-muon-plus/`
and `crates/dormouse-train/src/optim.rs`. No GPU, no build, no test.

## TL;DR

1. **The citation is REAL.** `arXiv:2602.21545` resolves, HTTP 200, correct title, real
   authors (Zhang, Zhao, Liu, Wang, Su, Tan, Zhang — UC Santa Barbara), submitted
   2026-02-25, **three versions, the current one v3 from 2026-05-14**. Full LaTeX-rendered
   HTML is available. **This is not a synthetic id.**
2. **Two real BUGs found**, one of them severe (a dead code path in `orthogonalize` that
   makes the claimed speedup unreachable — see D1).
3. **A provenance error**: the `37%` speedup number our README quotes exists only in **v3**,
   not in the v1 most code comments were written against. And `burn-spectral`'s polar
   retraction cites 2602.21545 §1 for a formula that is **not in that paper** (D5).
4. Our `ns_steps=8` vs the paper's **5** is a **deliberate, documented** deviation, not a
   bug. The "optimizer is 19% of a step" figure is orthogonal to the paper and unaffected.

---

## 1. Provenance

| item | value | checked how |
|---|---|---|
| arXiv id | `2602.21545` | `curl https://arxiv.org/abs/2602.21545` → **HTTP 200** |
| Title (v3, current) | *Muon+: Towards More Effective Muon via One Additional Normalization Step for LLM Pre-training* | `citation_title` meta |
| Title (v1/v2) | *Muon+: Towards Better Muon via One Additional Normalization Step* | HTML `<title>`, v1 and v2 |
| Authors | Ruijie Zhang, Yequan Zhao, Ziyue Liu, Zhengyang Wang, Yupeng Su, Liyan Tan, Zheng Zhang (UCSB) | `citation_author` |
| Versions | v1 2026-02-25, v2 2026-02-26, **v3 2026-05-14** | submission-history block |
| Full text | **available** — `https://arxiv.org/html/2602.21545v1`, `v2`, `v3` all HTTP 200 | fetched, 39.8K / 39.8K / 71.9K chars extracted |
| Reference code | **https://github.com/K1seki221/MuonPlus** — **HTTP 200, repo exists** | fetched |
| Verdict | **REAL PAPER, original source exists and was read in full** | — |

**Lineage — all real, all fetched:**

| ancestor | what it is | url | status |
|---|---|---|---|
| **Muon** — Jordan, Jin, Boza, You, Cesista, Newhouse, Bernstein (2024) | the optimizer itself: SGD-momentum + NS polar on the momentum | `kellerjordan.github.io/posts/muon/` (cited by the paper) | real |
| **muon.py** | Keller Jordan's actual implementation | `raw.githubusercontent.com/KellerJordan/muon/master/muon.py` | **fetched, 13205 bytes, read** |
| **"Muon is Scalable for LLM Training"** `2502.16982` | the scale validation | `arxiv.org/abs/2502.16982` | **HTTP 200** |
| **Polar Express** `2505.16932` (Amsel, Persson, Musco, Gower) | better `Ortho(·)` approximation | cited by paper §5.3 | real (per paper's own bib) |
| **NorMuon** (Li et al. 2025) | neuron-wise adaptive scaling, a *different* post-ortho normalization | paper App. C.1 | real |
| **Mano** `2601.23000` (Gu & Xie) | manifold-inspired updates | paper App. C.2 | per paper's bib |
| **Bernstein (2025)** "Deriving muon" | the `√(m/n)` pre-factor | paper §2 | real |

> Note: the paper's `muon_plus_step` pseudocode cites `Ortho(M) # newton-schulz` and says
> §3.1 "For the polar operator, we adopt the same configuration as in Jordan et al. (2024)".
> The paper **never prints the NS coefficients itself** — Jordan's `muon.py` is the
> authority, and it is `(3.4445, -4.7750, 2.0315)`. Our `NS_COEFFS` matches that exactly. ✅

---

## 2. Literal transcription — the paper's formulas

From the **v3** PDF→HTML (identical in v1/v2; verified by diff, only the abstract and
§3.4 numbering changed).

**Muon baseline, Eq. (1):**

```
M_t = μ·M_{t-1} + (1-μ)·G_t
O_t = Ortho(M_t)
W_t = W_{t-1} − η·√(m/n)·O_t
```

**Muon+, Eq. (4):**

```
M_t = μ·M_{t-1} + (1-μ)·G_t
O_t = Norm_(d)( Ortho(M_t) )
W_t = W_{t-1} − η·√(m/n)·O_t
```

`Ortho(·)` = the semi-orthogonal matrix closest to the input under Frobenius norm
(Higham 2008). If `M = UΣVᵀ` then `Ortho(M) := UVᵀ`. The `√(m/n)` pre-factor was
suggested by Bernstein (2025).

**Normalization operators, §2.3.** For `X = [x_ij] ∈ ℝ^{m×n}`, `ε > 0`:

```
(3)  Norm_(col)(X) := X · D_col^{-1}
(4)  D_col := diag( √(Σ_{i=1..m} x_{i1}²), …, √(Σ_{i=1..m} x_{in}²) )

(5)  Norm_(row)(X) := D_row^{-1} · X
(6)  D_row := diag( √(Σ_{j=1..n} x_{1j}²), …, √(Σ_{j=1..m} x_{mj}²) )

(7)  Norm_(col_row)(X) := Norm_(row)( Norm_(col)(X) )
(8)  Norm_(row_col)(X) := Norm_(col)( Norm_(row)(X) )
```

**Algorithm 1, verbatim pseudocode (App. C, identical in v1 and v3):**

```python
def muon_plus_step(W, M_prev, G, mu, lr, d="col", eps=1e-8):
    # momentum
    M = mu * M_prev + (1.0 - mu) * G
    # orthogonalize
    U = Ortho(M)   # newton-schulz
    # normalize
    O = norm_dir(U, d=d, eps=eps)
    # update
    m, n = W.shape[-2], W.shape[-1]
    W = W - lr * (m / n) ** 0.5 * O
    return W, M

def norm_dir(X, d="col", eps=1e-8):
    if d == "col":
        denom = (X.square().sum(dim=-2, keepdim=True) + eps).sqrt()
        return X / denom
    if d == "row":
        denom = (X.square().sum(dim=-1, keepdim=True) + eps).sqrt()
        return X / denom
    if d == "col_row":
        return norm_dir(norm_dir(X, "col", eps), "row", eps)
    if d == "row_col":
        return norm_dir(norm_dir(X, "row", eps), "col", eps)
```

Note the **`+ eps` inside the sqrt**, and `d="col"` as the function default.

### `ns_steps` — the paper's answer

> "Note that, all the experiments in this paper use **5 iterations** in `Ortho(·)` to
> approximate `UVᵀ`." (§3, v1 and v3, verbatim)

Every table in the paper — 60M→7B, compute-optimal and T2P≈200 overtraining, all three
polar methods — uses **5**. There is no 8 anywhere.

### Parameter-group routing — the paper's answer

> "Following the setup in Amsel et al. (2025), we apply Muon+ (or Muon) to **all
> parameters except embeddings, unembeddings, normalization layers, and positional
> encodings**, which are optimized using **AdamW**." (§3.1, v1 and v3)

That is the whole routing policy. The Muon+ group is "everything that is a weight
matrix"; embeddings / unembed / norms / positional → AdamW.

### Results (v3 Tables 1, 2, 5)

| model | params | tokens | Muon | Muon+ |
|---|---|---|---|---|
| GPT-Small | 124M | 3.0B | 29.66 | **27.64** (−2.02) |
| GPT-Base | 362M | 7.2B | 21.70 | **19.98** (−1.72) |
| LLaMA-60M | 58M | 1.1B | 25.75 | **25.25** (−0.50) |
| LLaMA-130M | 134M | 2.2B | 19.06 | **18.65** (−0.41) |
| LLaMA-350M | 368M | 6.4B | 14.02 | **13.41** (−0.61) |

Normalization-direction ablation (v1 Table 5 / v3 Table 6), LLaMA-350M:
`none 14.11 · col 13.73 · row 13.46 · col_row 13.41 · row_col 13.44`.
**Bi-directional wins; the two orders are within noise of each other; `row` consistently
beats `col`.** v3 states it more strongly than v1: "applying bi-directional normalization
consistently outperforms single-directional normalization."

**The 37.1% number (v3 only):** "Muon+ … speeds up the pre-training up to **37.1%**,
while requiring zero additional optimizer states" (§1) and §3.5 Table 5 "Speed-up ↑ 37.1%".
This is *wall-clock time to reach the same target loss*, not per-step time — the paper
explicitly says "Muon+ has nearly the same per-step runtime and memory cost as Muon".

---

## 3. Delta table

`file:line` against `vendor/burn-fused/crates/burn-muon-plus/src/` and
`crates/dormouse-train/src/optim.rs`. Verdicts: **BUG** / **DELIBERATE** / **BENIGN** /
**UNVERIFIABLE**.

| # | file:line | Paper / reference says | Our code does | Verdict |
|---|---|---|---|---|
| **D1** | `lib.rs:146` `if nc * 4 < nr` | (not in paper — our own optimization) | **Never true.** `nr`/`nc` are read from `x` *after* the transpose at `:130-134`, so `x` is always the wide side: `nc ≥ nr` always. `nc*4 < nr` is unsatisfiable. The entire factored branch `:147-170` and its fused kernel call `ns_combine_cuda` (`:154`) are **dead code**. | **BUG** (severe) |
| D2 | `lib.rs:142-145` comment | — | Comment claims factored is "measured 3.6x on [8192,512] via two `[c,c]@[c,r]` matmuls instead of `[c,c]@[c,c] + [c,c]@[c,r]`". FLOP count: factored = 3·(`2·nr²·nc`); direct = `2·nr²·nc + nr³`. At `nr=512, nc=8192` factored is **1.48× MORE** flops. The claim is backwards even as arithmetic. | **BUG** (stale/unreachable claim; the README already retracts the measurement as unmeasured) |
| D3 | `lib.rs:303-308` | Paper Eq. (4) + Alg. 1 line 10: `lr * (m/n)**0.5` | `lr * (m/n).max(1.0).sqrt()` — i.e. `max(1, m/n)^0.5` | **DELIBERATE** — matches Jordan's `muon.py` (`update *= max(1, m/n)**0.5`), documented at `:303-305` and in `bench/RESEARCH_VERIFICATION.md:13`. It is a **deviation from the Muon+ paper**, correctly sourced to Jordan instead. |
| D4 | `optim.rs:82` `MUON_NS_STEPS = 8` | Paper: **5**, everywhere, explicitly (§3) | 8 | **DELIBERATE** — comment at `:80-81` names the real source (the Qwen3.8-Flash-Next report §3.1, not Muon+). Correctly attributed. |
| D5 | `burn-spectral/src/lib.rs:164-166` doc comment | Muon+ §1 defines `Ortho(·)` abstractly and **prints no NS coefficients** | Comment says "Newton-Schulz polar iteration (Muon+ 2602.21545 §1)" — but the code below uses **cubic** coefficients `(15/8, −5/4, 3/8)` (`:205`, `:256`), which are **not** Muon+'s quintic `(3.4445, −4.775, 2.0315)` | **BUG** (mis-citation; see `tsct.md` §3) |
| D6 | `lib.rs:61` `NS_COEFFS = (3.4445, -4.775, 2.0315)` | Jordan `muon.py`: `a, b, c = (3.4445, -4.7750, 2.0315)` | exact match | **BENIGN** ✅ |
| D7 | `lib.rs:215-234` `norm_col`/`norm_row` | Alg. 1: `denom = (X.square().sum(dim) + eps).sqrt()`, `eps = 1e-8` | `.sqrt().clamp_min(1e-7)` | **BENIGN** — different epsilon placement (add-then-sqrt vs clamp-after-sqrt). Only differs when a norm is below ~1e-4, where the result is ~0 either way. `optim.rs:86` calls ColRow, so this is the live path. |
| D8 | `lib.rs:197-211` `ColRow`/`RowCol` | Eq. (7)/(8): `col_row := row(col(X))`, `row_col := col(row(X))` | `ColRow => norm_row(norm_col(x))`; `RowCol => norm_col(norm_row(x))` | **BENIGN** ✅ exact match, including composition order. |
| D9 | `lib.rs:274-297` momentum | Eq. (4): `M = μ·M_{t-1} + (1−μ)·G` | identical, no Nesterov | **BENIGN** ✅ (note: Jordan's `muon_update` defaults to `nesterov=True` and lerps `grad` toward `momentum`; the **paper drops Nesterov** and so do we — we follow the paper.) |
| D10 | `lib.rs:271` `if D == 2` | Paper routing: Muon+ on "all parameters except embeddings, unembeddings, normalization layers, positional encodings" | **Rank-based**: every 2-D tensor gets Muon+. On its own this would send the **2-D embedding** to Muon+, which the paper forbids. | **BENIGN** — in the live trainer this never happens: `routing.rs:96` routes `(Head, …)` and `rest_of(gdn2)` to `Group::Rest` and the embedding to the base optimizer; `burn-muon-plus` is a library whose `D==2` branch is only reached for params the trainer put in `Group::Muon`. `check_installed` (a 1-D param in Muon+ is a loud error) backs it. |
| D11 | `optim.rs:90-101` `group_of` | Paper: Muon+ on weight matrices, AdamW on embeddings/head/norms/positional | Policy: Muon+ on **only the small TSCT factors** + head-wise Q/K; AdamW on embeddings, head, routers, gates, convs, dense `[m,n]` | **DELIBERATE** — a *narrower* group than the paper's, justified by measured fp32 NS cost on `[d,d]` (`~40 s/step`, `routing.rs:76-80`). Deviation is documented and the reason is a machine fact, not a paper claim. |
| D12 | `lib.rs:192` `normalize` returns `x` unchanged when `norm_dir = None` | Paper always normalizes | gated off by config | **BENIGN** — `MuonPlusConfig::norm_dir` defaults to `None` (`:87`), so the crate's *default* is plain Muon, and `optim.rs:86` is what turns ColRow on. A/B-able by construction, which is what ADR-0002 wants. |
| D13 | `lib.rs:262-269` `g_active` mask | Not in the paper | A zero grad ⇒ zero update, decided on device | **DELIBERATE** — project rule ADR-0018 r2; long comment at `:249-261`. Correctly built with `mask_fill` on a float tensor, not a bool→float indicator. |
| D14 | `README.md` "up to 37% pre-training speedup" | **37.1% is v3-only** (2026-05-14). v1/v2 have no percentage at all. | quoted without a version | **BUG** (minor, provenance) — the number is real but the citation must name v3. Note also it is *time-to-target-loss*, not per-step. |
| D15 | `fused_kernels.rs` all 4 kernels | not in paper | README **already retracts** the 96×/84× and 3.6× numbers (no device flush in the timed loops, `README.md` "Performance") | **BENIGN** — self-corrected in-tree. `fused_match_tensor` (`:344`) *does* check ns_combine/momentum/finalize against the tensor path at <1e-5, which the README's "withdraw" paragraph understates. |
| D16 | `optim.rs:100-146` `HeadWiseMuon` | not in the paper (it is the Qwen report's rule) | per-head NS on Q/K | **DELIBERATE**, correctly attributed to the Qwen report, not to Muon+. |
| D17 | `lib.rs:113-121` `optimizer_groups` comment at `optim.rs:1-6` | — | says the policy "lives in `dormouse_core::routing`" | **BENIGN** ✅ — and it is not "lives there while the markers run": `optimizer_groups` **calls** `dormouse_core::routing::routing`, and the path-marker copy is deleted (`831e3a0`, 2026-09-28). The one policy is the id-based one. The old "tests-only per `routing.rs`" half of this row was wrong and was corrected 2026-10-01 (`docs/reviews/dedup-optimizer-2026-10-01.md`). |

### The `batched` vs `loop` 4.1–4.8× measurement

Not checked against the paper — the paper contains no batching discussion and Jordan's
`muon.py` has no batched path. `burn-spectral/src/lib.rs:225-290` (`retract_batched` /
`polar_orthogonalize_batched`) is the sync-free batched formulation, and it is
**identical math to the scalar path by construction** (same coefficients, same loop) —
only the σ_max estimation and the norms are batched. A 4.1–4.8× ratio on ops with
identical gradients is therefore consistent with the reference and is not evidence of a
divergence. **The 7 host syncs it removes are a real win** (the scalar path has 7
`into_scalar` calls per factor); the `§2.2`/`§3.1` "launch-bound" story is the same
phenomenon measured at step level. **Verdict: BENIGN, consistent.**

### The "optimizer is ~19% of a warm 246 ms step"

Not comparable to any paper number — 9.2M params, depth 2, batch 8, one A100-class
consumer GPU, vs the paper's 60M–7B on H100/A100. The paper's own claim is that
**per-step** cost is nearly unchanged; ours is that at 9.2M params the optimizer is a
large *share* of a step, which is a scale artifact (the NS matmuls are `[d,64]`
factors, not `[d,d]`, and everything else in the step is small too). **No conflict.
Verdict: BENIGN, orthogonal.**

---

## 4. What is UNVERIFIABLE from the paper

- **Whether `ns_steps=8` is better than 5.** The paper only ever ran 5. Our 8 is
  attributed to a different document. **No evidence in 2602.21545 supports 8**, and
  none contradicts it. A/B never run.
- **Whether `ColRow` specifically is right for us.** The paper's own best direction
  *flips by model* (`col_row` for 130M/350M, `row_col` for 60M/1B, plain `row` for
  GPT-Base, and **no normalization at all** for GPT-Large at 774M — Table 9, v1). Our
  choice of `ColRow` is one of two statistically-tied options in the paper, not *the*
  paper's best. The `lib.rs:55` doc comment "paper's best single combination" is
  **overstated** (v3 says bi-directional > single-directional, and the two orders are
  "nearly identical" — Table 11 caption). **Flagged, not fixed here.**
- **Head-wise Q/K NS, the whole Qwen routing, the Engram-table-Adam rule.** Not in this
  paper at all.
- **Our 4.1–4.8× batched-vs-loop and the 96×/84× fused-kernel numbers** — retracted
  in-tree; the paper cannot adjudicate them.

---

## 5. Severity ranking (subject 1)

1. **D1 — dead factored branch (`lib.rs:146`).** Not a correctness bug: the direct form
   at `:172-178` is a correct, complete Newton-Schulz quintic and is what actually runs.
   It is a *performance-claim* bug (the fused `ns_combine_cuda` at `:154` is unreachable,
   so that kernel is dead in production despite being tested at `fused_kernels.rs:344`)
   plus an unsatisfiable condition that reads like a guard. **Fix is one line** — either
   delete the branch or invert to `nc * 4 > nr` and re-measure with a device flush.
2. **D5 — `burn-spectral` cites Muon+ §1 for a cubic it does not have there.** Cross-file,
   handled in `tsct.md`.
3. **D2 / D14 — stale measurement claim, version-less citation.** Both cheap to fix.
4. **D3 / D4 / D11 — deliberate, documented deviations.** Correct as they stand; listed so
   nobody re-derives them as bugs.
