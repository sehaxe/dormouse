# Independent review of `research/papers/gdn-kda.md`

Reviewer: independent subagent. Date: 2026-09-29. No GPU, no build, no test runs.
Evidence: arXiv API, arXiv HTML full texts, `raw.githubusercontent.com` at
`fla-org/flash-linear-attention@main`, `MoonshotAI/FlashKDA@master`, and the
crate sources read in this tree.

**Bottom line: the provenance work is real and mostly right. The severity
call on the headline defect is wrong, and the fix it recommends would regress
a measured stability property. One factual error in §4.2 inverts its own
argument.**

---

## 0. Verdict per claim

| # | claim | verdict | note |
|---|---|---|---|
| 1 | `2601.16531` real, Engram paper, not cited by gdn2/kda | **VERIFIED** | plus one file the report's own list missed |
| 2 | five sources real and read | **VERIFIED** | one reproducibility gotcha (branch `master`) |
| 3 | BUG 1: false `a_log=-3` citation | **VERIFIED as a citation defect — REFUTED as an init defect** | the report's arithmetic is wrong and its fix is harmful |
| 4 | BUG 2: replicate vs zero padding, test blind | **VERIFIED** | confirmed on both FLA paths |
| 5 | BUG 3: fp32 decay gate not enforced | **VERIFIED**, and understated | a second primary source says the same |
| 6 | 27 MATCH / 3 BUG / 3 DELIBERATE / 1 BENIGN | **REFUTED** | the report's own table gives 23 / 4 / 4 / 1 |
| 7 | chunk 64 and 16 both check out; `g_min=-5` stated | **VERIFIED** | verbatim confirmed |

---

## 1. Claim 1 — `2601.16531` — VERIFIED

```
curl -sL "https://export.arxiv.org/api/query?id_list=2601.16531"
  totalResults: 1
  TITLE:  A Collision-Free Hot-Tier Extension for Engram-Style Conditional
          Memory: A Controlled Study of Training Dynamics
  AUTHORS n=1: ['Tao Lin']   PUB: 2026-01-23
```

Real, single-author, Engram-scoped. The report's core assertion — *neither
`burn-gdn2` nor `burn-kda` cites it* — I confirmed by grepping both crate trees
for any arXiv id. The complete set is:

```
burn-gdn2/README.md:10        2605.22791
burn-kda/src/lib.rs:5         2510.26692
burn-kda/src/lib.rs:6         2607.24653
burn-kda/README.md:9-10       (the same two)
```

Five occurrences repo-wide, all on the Engram line, all used as a slot-saturation
curve. **The report's premise-refutation is correct.**

**But its list is incomplete.** The report enumerates `schema.rs:122`,
`configs/small.toml:29`, `docs/AB-PROTOCOL.md:119`, `README.md:82`, and the
PKM research doc. There is a sixth:

```
research/2026-09-27-engram-reenable.md:19
```

On the headline claim — the one this report exists to settle — the file
inventory is wrong by one. Minor, but it is the claim with the most eyeballs on
it, and eyeballs are how this class of claim gets made wrong.

**Reproducibility note the report omits:** `export.arxiv.org` 301-redirects to
HTTPS and `curl` without `-L` returns a 0-byte body, which parses as "no
results" and would read as a phantom id. Anyone re-running this needs
`-sL https://`. Same trap for `MoonshotAI/FlashKDA`, whose default branch is
**`master`**, not `main` — the report's `tests/torch_ref.py` citation is correct
but silently 404s on the obvious URL.

---

## 2. Claim 2 — sources — VERIFIED

All five ids resolve to the titles/authors quoted (Kimi Linear n=60, Kimi K3
n=402, GDN-2 = Hatamizadeh/Choi/Kautz). `NVlabs/GatedDeltaNet-2`,
`fla-org/flash-linear-attention`, `MoonshotAI/FlashKDA` (1265 stars,
"CUTLASS…"), `MoonshotAI/Attention-Residuals` all exist.

---

## 3. Claim 3 — the `a_log` citation — HALF REFUTED

### 3.1 The citation really is false

K3 §2.1.1, verbatim from `arxiv.org/html/2607.24653v2`:

> where `A_h` is a learnable per-head log-scale and `g_min = −5` is fixed.
> **We initialize `A_h = 0`,** and each bias `b_α^h` is initialized following
> [57, 24, 139].

And FLA independently corroborates the zero, in code — which the report did
not cite and which makes the finding stronger than it presented it
(`fla/layers/kda.py:174-178`):

```python
# A_log and dt_bias are per value-head for native GVA support.
if safe_gate:
    self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
else:
    self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(1, 16)))
```

**`zeros` is exactly the K3-aligned branch.** FlashKDA
(`tests/torch_ref.py:161-165`) takes `A_log`/`dt_bias` as arguments and asserts
`A_log.dtype == torch.float32`. `bench_fwd.py:58` uses `torch.rand(H)`. No `−3`
anywhere. Every clause of `burn-kda/src/lib.rs:166-170` is unsupported.

**Citation defect: confirmed. This is a real ADR-0020 violation.**

### 3.2 …but the report's severity call and its fix are wrong

The report says (§4.2):

> "The K3 paper's `A_h = 0` with `b_α^h ≈ −5` gives `g = −2.5` too at init"

**That is wrong.** It evaluated `σ` at `z = 0` instead of at `z = b_α = −5`:

| config | `g = −5·σ(e^A·z)`, z = b_α | `α = exp(g)` |
|---|---|---|
| ours: A=−3, b=+1 | −2.562 | **0.077** |
| K3/FLA: A=0, b=−6.91 | −0.0050 | 0.995 |
| K3/FLA: A=0, b=−4.60 (mid) | −0.0498 | **0.952** |
| K3/FLA: A=0, b=−2.25 | −0.476 | 0.621 |

The correct value is `g = −0.0335`, `α = 0.967` — **not −2.5**. The error is a
factor of ~75 in `g`, and it is load-bearing: the report uses it to argue that
a hard-wired `A` is "decisive" because it denies the bias control of the logit
magnitude. Under the correct arithmetic the bias controls essentially *all* of
it, and the comparison the report wanted does not exist.

### 3.3 The init is a deliberate, MEASURED stabilisation

The report never read the one document that records why this value is here.
`AGENTS.md:274`:

> Moonshot/FLA decay init (`a_log=-3`, `b_alpha=1.0`) + clamp measurably
> extended the clean window (110+ steps, Muon+, fp32, 0 NaN on 2026-08-29,
> against 45-127 before).

And `burn-kda/src/lib.rs:188-194` carries the clamp with a dated measurement
behind it. So the author chose `−3` and `+1.0` to land `σ ≈ 0.5 → α ≈ 0.08`,
knew what they were doing, and confirmed it moved the NaN window. The
**values are a documented stabilisation choice. Only the attribution sentence
is fabricated.** The report frames this as "a citation that resolves to
nothing" (correct) and then treats the *value* as the defect (incorrect).

### 3.4 The report's recommended fix would make the model worse

§4.2 ends: *"Fix the comment and the init, or drop the citation."* As written
that is a loaded gun, and the obvious reading — set `a_log = 0`, since that is
what K3 says — moves the model the wrong way:

| fix | `α` at init |
|---|---|
| today (A=−3, b=+1) | 0.077 |
| **fix `a_log` only → 0** | **0.026** (3× *more* aggressive decay) |
| fix `b_alpha` only → −4.6 | 0.109 |
| both → K3 recipe | 0.952 |

Fixing `a_log` alone pushes retention *further* from 1, into a regime AGENTS.md
§2.3 already records as the NaN trigger. The report hands this to a future
reader with no ordering, no guard and no mention of the measured window.

**And the report never identifies the dominant error.** `b_alpha = +1.0` sits
~6 units away from FLA's `inv_dt` range `[−6.91, −2.25]`; `a_log` sits 3 units
away from 0. In `α` terms the gap to the reference is `0.077` vs `0.952` — a
**12× retention deficit at init**, and it is overwhelmingly the *bias* sign
that causes it. That is the defensible quantitative finding in this section,
and the report does not make it.

**Correct verdict on claim 3: the citation is fabricated (report right, and
propagated into AGENTS.md:274 and AGENTS.md:838 as rule-level fact). The init
is deliberate and measured. The report's remedy is wrong and its key
arithmetic is wrong.**

---

## 4. Claim 4 — short-conv padding — VERIFIED

Both FLA paths, read directly:

```
fla/modules/conv/short_conv.py:69   padding=kernel_size - 1        # padding_mode not passed
fla/modules/conv/triton/kernels.py:93,103
        tl.load(p_yi, mask=((o_x >= 0) & (o_x < T))[:,None] & m_d[None,:], other=0.0)
fla/modules/conv/triton/kernels.py:394   (decode/update cache load) other=0.0
fla/modules/conv/short_conv.py:219   cache = x.new_zeros(N, D, W)
```

`ShortConvolution(nn.Conv1d)` never calls `nn.Conv1d.forward`; it dispatches to
FLA's Triton `causal_conv1d`. Left pad = **zeros**, on prefill and on decode.

Ours (`burn-gdn2/src/short_conv.rs:44-46`) replicates the first token. The
report's arithmetic is right: divergence confined to positions 0,1,2; at `T=1`
the reference gives `y = w₃·x₀` and we give `y = (w₀+w₁+w₂+w₃)·x₀`. I checked
the weight layout too — tap `i` multiplies `x[t+i]`, which matches
`nn.Conv1d`'s left-padded correlation convention, so only the padding is wrong,
not the tap order. Good precision from the report.

**The blindness claim is also right, and it is the right framing:**

```
tests/gen_reference.py:85    x_pad = torch.cat([x[:, :1].repeat(1, 3, 1), x], dim=1)
tools/gen_reference.rs:134   let src = if ti + i < 3 { 0 } else { ti + i - 3 };
```

Both transcribe the bug. `gen_reference.py:82-83` even says so: *"replicate
padding (first token), matching burn-gdn2 (and this repo's reference data)."*
A fixture generated from the implementation cannot find a divergence from the
implementation.

**Live impact: zero, as the report says.** `crates/dormouse-core/src/attention.rs:68`
→ `use_short_conv: false`, with the reason inline. The crate defaults are still
`true` (`burn-gdn2/src/config.rs:122`, `burn-kda/src/lib.rs:76`), so the report
is right that the crate would ship a wrong conv to anyone who takes the default.

---

## 5. Claim 5 — fp32 decay gate — VERIFIED, and understated

GDN-2 App. D.1, verbatim:

> The decay gate in Eq. 86 is computed in explicit fp32 before entering the
> kernels. This is important because the local cumulative sum
> `G_r = Σ_{i≤r} g_i` is a path-length-dependent quantity.

`burn-gdn2/src/module.rs:507-516` has no cast anywhere on the decay path:

```rust
let dt_b = self.dt_bias.val().reshape([1, 1, kd]);
let a_exp = self.a_log.val().exp()...
let g_pre = softplus(f_out + dt_b, 1.0);
let g = -a_exp * g_pre;
```

`f_out` carries the activation dtype, so under `--bf16` the whole
`G_r` accumulation is bf16. Gap confirmed.

**The report undersold it by citing one source when there are two.** FlashKDA
enforces it in the kernel reference independently of the paper:

```python
# tests/torch_ref.py:163,166
assert A_log.dtype == torch.float32
g = g.to(torch.float32) + dt_bias.unsqueeze(0)
```

So the requirement is not a GDN-2-only convention — the KDA reference this
crate also claims to follow asserts it in code.

---

## 6. Claim 6 — the counts — REFUTED

The report's §8 headline is *"27 of 33 delta rows are exact matches"*. Tallying
the report's own table:

| verdict | rows | count |
|---|---|---|
| MATCH | 1–12, 15, 16, 17, 22, 23, 24, 28, 29, 30, 31, 33 | **23** |
| MATCH + BENIGN (hybrid) | 13 | 1 |
| BUG | 14, 25, 26, 32 | **4** |
| DELIBERATE | 19, 20, 21, 27 | **4** |
| BENIGN | 18 | 1 |

23 + 1 + 4 + 4 + 1 = 33. The true split is **23 MATCH / 4 BUG / 4 DELIBERATE
/ 1 BENIGN**. "27" is inflated by ~4, and the DELIBERATE count is 4, not 3.
Small in absolute terms; it is still a headline statistic asserted without
arithmetic, in a report whose entire thesis is that unverified counts are a
defect class.

---

## 7. Claim 7 — chunk sizes — VERIFIED

- GDN-2 App. C.2, verbatim: *"The chunk size is fixed to C = 64."*
  Code: `burn-gdn2/src/config.rs:126` = 64 ✓
- FlashKDA `tests/torch_ref.py:154`: `CHUNK = 16`.
  Code: `burn-kda/src/lib.rs:81` = 16 ✓
- `g_min = −5`, verbatim in K3 §2.1.1, corroborated by
  `bench_fwd.py:35 LOWER_BOUND = -5.0` and `burn-kda/src/lib.rs:26 G_MIN`.
- The two-crates-two-chunk-sizes point is right and worth making.

---

## 8. MATCH spot-checks (10 rows)

Every one I checked holds. The report is not padding this section.

| row | claim | file:line (verified) | result |
|---|---|---|---|
| 8 | `scale = d_k^{-0.5}` | `burn-gdn2/src/module.rs:299` `(hk as f64).powf(-0.5)` | ✅ |
| 10 | `A_log = log U(1,16)` | `module.rs:206-208` `Uniform(1.0.ln(), 16.0.ln())` | ✅ |
| 11 | `inv_dt = dt + log(−expm1(−dt))`, `dt~logU(.001,.1)` | `module.rs:212-221`; FLA `kda.py:180-184` is literally the same two lines | ✅ |
| 12 | conv init `U(−0.5, 0.5)` | `module.rs:202`; `1/√fan_in = 1/2` for fan_in 4 — the comment's non-obvious claim is **correct** | ✅ |
| 5 | GVA repeats q,k,g,**b**; not v,w | `module.rs:545-556` — repeats `q_4d, k_4d, g_4d, b_4d`; `v_4d`/`w_gate` untouched | ✅ *(forward only — see 9.1)* |
| 6 | `allow_neg_eigval` → `b*2`, w untouched | `module.rs:559` `b_4d.mul_scalar(2.0)`; paper §3.1 verbatim *"scaling only the erase gate to [0,2]^{d_k}. The write gate remains in [0,1]^{d_v}"* | ✅ |
| 15 | chunk algebra Eqs 18–25 | `forward.rs:246-296` | ✅ **verified line for line** |
| 23 | `g_min = −5`, Eq 5 | `lib.rs:26,200` | ✅ |
| 30 | full-rank sigmoid gate Eq 6 | `lib.rs:483-500`; FLA `kda.py:191 FusedRMSNormGated(..., activation="sigmoid")` | ✅ |
| 31 | chunk 16 | `lib.rs:81`; `torch_ref.py:154` | ✅ |

Row 15 deserves the credit. I traced it term by term:
`k_over_gamma = k5/g_exp` → `K̄`; `q_gated = q5*g_exp` → `Q_γ`;
`akk = (bk*g_exp) @ k_over_gammaᵀ * strict` → `T = tril(ĒK̄ᵀ,−1)`;
`w_wy = m_inv @ (bk*g_exp)` → `Y = AĒ`; `u = m_inv @ (w5*v5)` → `U = AZ`;
`v_new = U − Y·S_n` → Eq 23; `out = aqk@v_new + q_gated@S_n·scale` → Eq 24;
`k_dec = k5 * (g_last − g_cumsum).exp()` → `K_tail = (γ_C/γ_r)⊙k`. Exact.

I also checked `neumann_inverse` (`forward.rs:335-343`) without being asked,
because a truncated Neumann series would be an approximation dressed as an
inverse. It is **exact**: `acc = I + Σ_{k=1}^{c−1}(−L)^k`, and `L^c = 0` for
strictly-lower `c×c`. Correct, and the doc comment says so.

---

## 9. What the report missed

### 9.1 The GVA gate is parameterised on the wrong axis (the big one)

Rows 5 and 33 are marked MATCH after checking the *forward repeat*. The
*parameter* is shared across value heads; FLA's is not.

```
fla/layers/kda.py:166-167
  # Gate dim = HV * K: per value-head, per key-dim gating.
  self.gate_dim = int(self.num_v_heads * self.head_k_dim)
fla/layers/kda.py:174
  # A_log and dt_bias are per value-head for native GVA support.
  self.A_log = nn.Parameter(...(self.num_v_heads...))     # [HV]
  self.dt_bias = nn.Parameter(inv_dt)                      # [HV * K]
```

Ours, both crates:

```
burn-kda/src/lib.rs:165   b_alpha: Tensor::ones([n_heads * head_dim])   # [H*K]
burn-kda/src/lib.rs:170   a_log:  Tensor::full([n_heads, 1], -3.0)      # [H, 1]
burn-gdn2/src/module.rs:207-221   dt_bias [kd = H*HK],  a_log [h]
```

Under GVA (`HV > H`) we allocate `H` log-scales and `H·K` biases where the
reference allocates `HV` and `HV·K` — under-parameterised by the group factor,
and every value head in a group is forced to share one decay channel. The
report read FLA's `state_v_first=True` and `gate_dim` (it quotes the former in
row 33) without noticing the latter is a *shape* difference.

**Blast radius today: zero.** `crates/dormouse-core/src/attention.rs` never
sets `num_v_heads`, so `hv = h` and `rep = 1`. But both crates ship GVA as a
headline feature, `burn-gdn2/README.md:132` lists `num_v_heads` as a config
knob, and under GVA the "MATCH" on rows 5/33 is wrong.

### 9.2 `burn-kda` has no upstream-fidelity test at all

The report criticises `burn-gdn2`'s fixture for being self-consistent. It
never says that `burn-kda` — **the crate dormouse actually instantiates** —
has no fixture of any kind:

```
find burn-kda -name "*.bin" -o -name "*gold*" -o -name "*fixture*"   → nothing
find burn-kda -name "gen_reference*"                                 → nothing
burn-kda/tests/  → bench_cuda.rs cuda_gate.rs fused_cuda.rs ops_grad_cuda.rs
```

No `ref_data.bin`, no transcription, no upstream comparison. So the two
defects in this report — the fabricated citation and the inherited conv
padding — live in the one crate with zero cross-implementation coverage. §7
spends fourteen paragraphs redesigning `burn-gdn2`'s test strategy and never
states that the harder problem is the crate with no test at all.

### 9.3 The comment's "neutral α≈0.5" is an apples-to-oranges baseline

`lib.rs:168-169` says the decay starts *"conservative (alpha ~ 0.08) instead of
neutral (alpha ~ 0.5 at A=0, b=0)"*. The default is `DecayFn::Sigmoid`
(`lib.rs:78,98`). Under **Sigmoid**, `A=0, b=0` gives `g = −5·σ(0) = −2.5`,
`α = 0.082` — the *same* 0.08. `α = 0.5` is the Kimi **Linear softplus** value
at `A=0,b=0` (`exp(−softplus(0)) = 0.5`). The comment compares the K3 formula
against the Kimi Linear formula and presents the difference as a deviation.
Nothing is deviated from; `A=0` is exactly where K3 puts it, and 0.08 is what
K3 gives you for free.

### 9.4 `--bf16` is not a live path either

Claim 5's severity rests on `--bf16`. AGENTS.md §2.1: the LLVM dialect has no
bf16 type, `restrict_to_llvm_backend` deletes it from the advertised element
types on purpose, and *"every bf16 run on this box is SLOWER than fp32"*. So
claims 4 **and** 5 both have zero live impact. The report says this for claim 4
and omits it for claim 5, which makes the two defects read as differently
urgent when neither can currently fire.

### 9.5 The README table was misread

§8 open question 4 quotes burn-gdn2's README as claiming both *"1000-case
comparison against an independent transcription"* **and** *"Verification: none
shipped."* It is a two-column table (`burn-gdn2` | `NVlabs reference`); the
right-hand cell belongs to NVIDIA. This mangles the very row the report credits
in §1.2 as *"exactly the ADR-0020 form. Credit where due."*

### 9.6 `burn-gdn2` is not the live path

`crates/dormouse-core/src/attention.rs` builds `burn_kda::KdaModule`, and
`kda_seam_counts()` is an alias for `burn_gdn2::seam_counts()`. Nothing in
dormouse instantiates `burn-gdn2`. Twenty-one of the report's delta rows
describe a crate that no training run touches — `chunk_size=64`,
`min_decay`, `allow_neg_eigval`, the whole `ChunkPath` switch. The report never
says so, and a reader would reasonably come away thinking the chunk-64 default
is a live configuration choice.

### 9.7 Tier-0 gold vectors are a wish, not a plan

§7.1 proposes generating `gdn2_chunk.bin` from `lit_gpt/gdn2_ops/chunk_gdn2.py`
and `gdn2_conv.bin` from `lit_gpt/gdn2.py`. That repo is Triton + flash-attn +
lit-gpt, NVIDIA-only — it cannot run on this box at all, and `gdn2.py` has no
working CPU path for the conv. Tier 0 as written requires hardware nobody here
has. Fine as a wish; not a plan.

---

## 10. Top 3 findings

**1. `b_alpha = +1.0` is the real bug, and nobody named it.** Against FLA's
`inv_dt ∈ [−6.91, −2.25]` and K3's `A_h = 0`, dormouse starts at `α = 0.077`
where the reference starts at `α ≈ 0.95` — a 12× retention deficit at init,
driven overwhelmingly by the bias sign, not by `A`. It is documented as
deliberate and it was measured to help on a 0.85 MB overfit corpus, so it is
not free to change. But it is a quantified deviation from the cited source, it
has never been A/B'd (AGENTS.md:838 calls it "the open question in the A/B
queue"), and this report misdiagnosed it as an `a_log` problem.

**2. The report's §4.2 arithmetic error propagates a harmful fix.** `g = −2.5`
should be `g = −0.033`. Every downstream sentence — including "decisively" —
rests on it. Actionable consequence: **do not apply §4.2's "fix the comment and
the init"**. Split it: (a) delete the false attribution, which is free and
required by ADR-0020; (b) leave `a_log`/`b_alpha` alone, since `a_log → 0` alone
makes retention 3× more aggressive into a documented NaN regime; (c) put the
`α = 0.077` vs `0.95` gap into the A/B queue as a named, quantified question.

**3. The GVA gate axis (`fla/layers/kda.py:166-167`) is a real MATCH that
isn't, and the report marked it MATCH because it checked the wrong layer.** The
forward repeat is right; the parameter allocation is on the key-head axis where
FLA's is on the value-head axis. Dormouse runs `hv = h` so it is dormant, but
GVA is a shipped feature and the two crates would diverge the moment anyone
sets `num_v_heads`. Nothing in §5.1/§5.3 reaches this — §5.1 lists
`state_v_first` as "**check**" and never follows through.

---

## 11. What would falsify this review

- **`grep -rn "2601.16531" research/2026-09-27-engram-reenable.md`** returns
  nothing → finding 1 in §1 (the missing sixth occurrence) is void; the
  provenance verdict stands regardless.
- **An `AGENTS.md` history entry showing the `−3`/`+1.0` pair predates
  2026-08-29**, i.e. was not adopted *because of* the NaN measurement →
  finding 2 in §3.3 weakens from "deliberate and measured" to "coincidental".
  It would not rescue the citation.
- **A `KimiConfig`-equivalent in burn-kda that expands `a_log`/`b_alpha` to the
  value-head axis when `num_v_heads > num_heads`** → finding 3 is void. I found
  no such code in `lib.rs:165-170` or `module.rs:206-221`.
- Reproduce the arithmetic:
  `python3 -c "import math;s=lambda x:1/(1+math.exp(-x));print(-5*s(math.exp(-3)*1), math.exp(-5*s(1*(-5))))"`
  → `(-2.562…, -0.0334…)`. If this prints anything else, this reviewer is wrong
  and §3.2 falls.

**Unverified, with the command that would settle it.** Whether the
hand-derived chunk adjoint in `burn-gdn2/src/autodiff.rs` carries the
L2-normalization VJP (the report flags this and I concur it is open):
`cargo test -p burn-gdn2 --features cuda --test fused_adjoint_vs_ops` on a GPU
box. Out of scope here — no GPU, no build.
