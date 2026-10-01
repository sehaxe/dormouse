# gdn-kda — two independent reviews, one lane

Merged 2026-10-01 from the `gdn-kda-review-a.md` and `gdn-kda-review-b.md` halves, both at
`5cfdbda`, which is where a reader finds each one whole. The halves were written independently
and neither read the other (A: Independent review of `docs/papers/gdn-kda.md`; B: Review B — the CODE, not the paper),
which is why both verdicts are kept: where they agree the finding is settled,
where they disagree the disagreement is the finding. Nothing was reworded.

## Reviewer A — Independent review of `docs/papers/gdn-kda.md`

Reviewer: independent subagent. Date: 2026-09-29. No GPU, no build, no test runs.
Evidence: arXiv API, arXiv HTML full texts, `raw.githubusercontent.com` at
`fla-org/flash-linear-attention@main`, `MoonshotAI/FlashKDA@master`, and the
crate sources read in this tree.

**Bottom line: the provenance work is real and mostly right. The severity
call on the headline defect is wrong, and the fix it recommends would regress
a measured stability property. One factual error in §4.2 inverts its own
argument.**

---

### 0. Verdict per claim

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

### 1. Claim 1 — `2601.16531` — VERIFIED

```
curl -sL "https://export.arxiv.org/api/query?id_list=2601.16531"
  totalResults: 1
  TITLE:  A Collision-Free Hot-Tier Extension for Engram-Style Conditional
          Memory: A Controlled Study of Training Dynamics
  AUTHORS n=1: ['Tao Lin']   PUB: 2026-01-23
```

Real, single-author, Engram-scoped. The report's core assertion — *neither
`dormouse-gdn2` nor `dormouse-kda` cites it* — I confirmed by grepping both crate trees
for any arXiv id. The complete set is:

```
dormouse-gdn2/README.md:10        2605.22791
dormouse-kda/src/lib.rs:5         2510.26692
dormouse-kda/src/lib.rs:6         2607.24653
dormouse-kda/README.md:9-10       (the same two)
```

Five occurrences repo-wide, all on the Engram line, all used as a slot-saturation
curve. **The report's premise-refutation is correct.**

**But its list is incomplete.** The report enumerates `schema.rs:122`,
`configs/small.toml:29`, `docs/protocols/AB-PROTOCOL.md:119`, `README.md:82`, and the
PKM research doc. There is a sixth:

```
docs/research/2026-09-27-engram-reenable.md:19
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

### 2. Claim 2 — sources — VERIFIED

All five ids resolve to the titles/authors quoted (Kimi Linear n=60, Kimi K3
n=402, GDN-2 = Hatamizadeh/Choi/Kautz). `NVlabs/GatedDeltaNet-2`,
`fla-org/flash-linear-attention`, `MoonshotAI/FlashKDA` (1265 stars,
"CUTLASS…"), `MoonshotAI/Attention-Residuals` all exist.

---

### 3. Claim 3 — the `a_log` citation — HALF REFUTED

#### 3.1 The citation really is false

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
anywhere. Every clause of `dormouse-kda/src/lib.rs:166-170` is unsupported.

**Citation defect: confirmed. This is a real ADR-0020 violation.**

#### 3.2 …but the report's severity call and its fix are wrong

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

#### 3.3 The init is a deliberate, MEASURED stabilisation

The report never read the one document that records why this value is here.
`AGENTS.md:274`:

> Moonshot/FLA decay init (`a_log=-3`, `b_alpha=1.0`) + clamp measurably
> extended the clean window (110+ steps, Muon+, fp32, 0 NaN on 2026-08-29,
> against 45-127 before).

And `dormouse-kda/src/lib.rs:188-194` carries the clamp with a dated measurement
behind it. So the author chose `−3` and `+1.0` to land `σ ≈ 0.5 → α ≈ 0.08`,
knew what they were doing, and confirmed it moved the NaN window. The
**values are a documented stabilisation choice. Only the attribution sentence
is fabricated.** The report frames this as "a citation that resolves to
nothing" (correct) and then treats the *value* as the defect (incorrect).

#### 3.4 The report's recommended fix would make the model worse

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

### 4. Claim 4 — short-conv padding — VERIFIED

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

Ours (`dormouse-gdn2/src/short_conv.rs:44-46`) replicates the first token. The
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
padding (first token), matching dormouse-gdn2 (and this repo's reference data)."*
A fixture generated from the implementation cannot find a divergence from the
implementation.

**Live impact: zero, as the report says.** `crates/dormouse-core/src/attention.rs:68`
→ `use_short_conv: false`, with the reason inline. The crate defaults are still
`true` (`dormouse-gdn2/src/config.rs:122`, `dormouse-kda/src/lib.rs:76`), so the report
is right that the crate would ship a wrong conv to anyone who takes the default.

---

### 5. Claim 5 — fp32 decay gate — VERIFIED, and understated

GDN-2 App. D.1, verbatim:

> The decay gate in Eq. 86 is computed in explicit fp32 before entering the
> kernels. This is important because the local cumulative sum
> `G_r = Σ_{i≤r} g_i` is a path-length-dependent quantity.

`dormouse-gdn2/src/module.rs:507-516` has no cast anywhere on the decay path:

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

### 6. Claim 6 — the counts — REFUTED

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

### 7. Claim 7 — chunk sizes — VERIFIED

- GDN-2 App. C.2, verbatim: *"The chunk size is fixed to C = 64."*
  Code: `dormouse-gdn2/src/config.rs:126` = 64 ✓
- FlashKDA `tests/torch_ref.py:154`: `CHUNK = 16`.
  Code: `dormouse-kda/src/lib.rs:81` = 16 ✓
- `g_min = −5`, verbatim in K3 §2.1.1, corroborated by
  `bench_fwd.py:35 LOWER_BOUND = -5.0` and `dormouse-kda/src/lib.rs:26 G_MIN`.
- The two-crates-two-chunk-sizes point is right and worth making.

---

### 8. MATCH spot-checks (10 rows)

Every one I checked holds. The report is not padding this section.

| row | claim | file:line (verified) | result |
|---|---|---|---|
| 8 | `scale = d_k^{-0.5}` | `dormouse-gdn2/src/module.rs:299` `(hk as f64).powf(-0.5)` | ✅ |
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

### 9. What the report missed

#### 9.1 The GVA gate is parameterised on the wrong axis (the big one)

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
dormouse-kda/src/lib.rs:165   b_alpha: Tensor::ones([n_heads * head_dim])   # [H*K]
dormouse-kda/src/lib.rs:170   a_log:  Tensor::full([n_heads, 1], -3.0)      # [H, 1]
dormouse-gdn2/src/module.rs:207-221   dt_bias [kd = H*HK],  a_log [h]
```

Under GVA (`HV > H`) we allocate `H` log-scales and `H·K` biases where the
reference allocates `HV` and `HV·K` — under-parameterised by the group factor,
and every value head in a group is forced to share one decay channel. The
report read FLA's `state_v_first=True` and `gate_dim` (it quotes the former in
row 33) without noticing the latter is a *shape* difference.

**Blast radius today: zero.** `crates/dormouse-core/src/attention.rs` never
sets `num_v_heads`, so `hv = h` and `rep = 1`. But both crates ship GVA as a
headline feature, `dormouse-gdn2/README.md:132` lists `num_v_heads` as a config
knob, and under GVA the "MATCH" on rows 5/33 is wrong.

#### 9.2 `dormouse-kda` has no upstream-fidelity test at all

The report criticises `dormouse-gdn2`'s fixture for being self-consistent. It
never says that `dormouse-kda` — **the crate dormouse actually instantiates** —
has no fixture of any kind:

```
find dormouse-kda -name "*.bin" -o -name "*gold*" -o -name "*fixture*"   → nothing
find dormouse-kda -name "gen_reference*"                                 → nothing
dormouse-kda/tests/  → bench_cuda.rs cuda_gate.rs fused_cuda.rs ops_grad_cuda.rs
```

No `ref_data.bin`, no transcription, no upstream comparison. So the two
defects in this report — the fabricated citation and the inherited conv
padding — live in the one crate with zero cross-implementation coverage. §7
spends fourteen paragraphs redesigning `dormouse-gdn2`'s test strategy and never
states that the harder problem is the crate with no test at all.

#### 9.3 The comment's "neutral α≈0.5" is an apples-to-oranges baseline

`lib.rs:168-169` says the decay starts *"conservative (alpha ~ 0.08) instead of
neutral (alpha ~ 0.5 at A=0, b=0)"*. The default is `DecayFn::Sigmoid`
(`lib.rs:78,98`). Under **Sigmoid**, `A=0, b=0` gives `g = −5·σ(0) = −2.5`,
`α = 0.082` — the *same* 0.08. `α = 0.5` is the Kimi **Linear softplus** value
at `A=0,b=0` (`exp(−softplus(0)) = 0.5`). The comment compares the K3 formula
against the Kimi Linear formula and presents the difference as a deviation.
Nothing is deviated from; `A=0` is exactly where K3 puts it, and 0.08 is what
K3 gives you for free.

#### 9.4 `--bf16` is not a live path either

Claim 5's severity rests on `--bf16`. AGENTS.md §2.1: the LLVM dialect has no
bf16 type, `restrict_to_llvm_backend` deletes it from the advertised element
types on purpose, and *"every bf16 run on this box is SLOWER than fp32"*. So
claims 4 **and** 5 both have zero live impact. The report says this for claim 4
and omits it for claim 5, which makes the two defects read as differently
urgent when neither can currently fire.

#### 9.5 The README table was misread

§8 open question 4 quotes dormouse-gdn2's README as claiming both *"1000-case
comparison against an independent transcription"* **and** *"Verification: none
shipped."* It is a two-column table (`dormouse-gdn2` | `NVlabs reference`); the
right-hand cell belongs to NVIDIA. This mangles the very row the report credits
in §1.2 as *"exactly the ADR-0020 form. Credit where due."*

#### 9.6 `dormouse-gdn2` is not the live path

`crates/dormouse-core/src/attention.rs` builds `dormouse_kda::KdaModule`, and
`kda_seam_counts()` is an alias for `dormouse_gdn2::seam_counts()`. Nothing in
dormouse instantiates `dormouse-gdn2`. Twenty-one of the report's delta rows
describe a crate that no training run touches — `chunk_size=64`,
`min_decay`, `allow_neg_eigval`, the whole `ChunkPath` switch. The report never
says so, and a reader would reasonably come away thinking the chunk-64 default
is a live configuration choice.

#### 9.7 Tier-0 gold vectors are a wish, not a plan

§7.1 proposes generating `gdn2_chunk.bin` from `lit_gpt/gdn2_ops/chunk_gdn2.py`
and `gdn2_conv.bin` from `lit_gpt/gdn2.py`. That repo is Triton + flash-attn +
lit-gpt, NVIDIA-only — it cannot run on this box at all, and `gdn2.py` has no
working CPU path for the conv. Tier 0 as written requires hardware nobody here
has. Fine as a wish; not a plan.

---

### 10. Top 3 findings

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

### 11. What would falsify this review

- **`grep -rn "2601.16531" docs/research/2026-09-27-engram-reenable.md`** returns
  nothing → finding 1 in §1 (the missing sixth occurrence) is void; the
  provenance verdict stands regardless.
- **An `AGENTS.md` history entry showing the `−3`/`+1.0` pair predates
  2026-08-29**, i.e. was not adopted *because of* the NaN measurement →
  finding 2 in §3.3 weakens from "deliberate and measured" to "coincidental".
  It would not rescue the citation.
- **A `KimiConfig`-equivalent in dormouse-kda that expands `a_log`/`b_alpha` to the
  value-head axis when `num_v_heads > num_heads`** → finding 3 is void. I found
  no such code in `lib.rs:165-170` or `module.rs:206-221`.
- Reproduce the arithmetic:
  `python3 -c "import math;s=lambda x:1/(1+math.exp(-x));print(-5*s(math.exp(-3)*1), math.exp(-5*s(1*(-5))))"`
  → `(-2.562…, -0.0334…)`. If this prints anything else, this reviewer is wrong
  and §3.2 falls.

**Unverified, with the command that would settle it.** Whether the
hand-derived chunk adjoint in `dormouse-gdn2/src/autodiff.rs` carries the
L2-normalization VJP (the report flags this and I concur it is open):
`cargo test -p dormouse-gdn2 --features cuda --test fused_adjoint_vs_ops` on a GPU
box. Out of scope here — no GPU, no build.

## Reviewer B — Review B — the CODE, not the paper

Second independent review of `docs/papers/gdn-kda.md`. Angle: the engineering
conclusions. I assume every citation in the report is correct and do not re-check the
papers. I assume nothing in the report. Read-only: no `cargo build`, no `cargo test`, no
GPU. Everything below is derived from source in this tree plus arithmetic I computed by
hand against the **committed bytes** of `tests/ref_data.bin`.

Verdict up front, before the detail: **the report's headline defect (the short-conv
padding) is real, and its diagnosis and its remedy are both wrong.** The remedy is a test
that *cannot fail on this bug*, and the diagnosis rests on a claim about the state of
`bit_exact` that the file itself contradicts in plain prose 15 lines from the top. A
second defect the report does not mention at all — a layout bug that makes the fixture
wrong for 976 of 1000 cases — is what actually gates everything.

---

### 1. Does chunk size change the padding behaviour? **No.** The recommended test is inert.

This is the most checkable claim in the report (§7.2 item 6) and it is the one I was asked
to run first. It fails.

The report says, at `gdn-kda.md:554-558`:

> **6. Chunk-size invariance.** `forward_train(x, chunk=16) ≡ forward_train(x,
>    chunk=64)` to f32 tolerance. *This is the test that finds the conv padding and any
>    chunk-boundary seam*, because the seam lands in a different place at each chunk size.
>    It is also free — one extra config, no new code. **I would add this one first.**

**Chunk size cannot move the conv's left pad, because the conv runs before the chunking
and is not a function of `chunk_size` at all.**

`short_conv_1d` is called in exactly two places, both on the full `[B, T, C]` projection,
before any reshaping into chunks and before `chunk_size` is read:

- `vendor/dormouse-fused/crates/dormouse-gdn2/src/module.rs:492-503` — inside `project()`. The
  conv runs at `:494-499`; `project()` returns at `:566`; the chunk path is not invoked
  until `forward_train_core` calls it at `module.rs:442-452`, i.e. *after* `project()`
  has already returned at `:417`.
- `vendor/dormouse-fused/crates/dormouse-kda/src/lib.rs:416-430` — same shape: conv on the full
  `[B, T, C]` tensor at `:416-430`, `to_4d` only at `:447-449`.

`short_conv_1d` takes no `chunk_size` argument (`short_conv.rs:22-26`). Its left pad is
built once, at `short_conv.rs:43-47`, from `x[:, 0:1]` of the whole sequence. The padding
lives at sequence positions −3…−1 relative to the sequence, and a chunk boundary at
offset 16 or 64 does not relocate it. For `f=16` and `f=64` the conv output is
**bit-identical**. Therefore `forward_train(x, 16)` and `forward_train(x, 64)` receive
byte-for-byte the same buggy `q/k/v` and the test passes with the bug present, in every
configuration, on every backend.

The report's justification — "the seam lands in a different place at each chunk size" —
is true of the *chunk-boundary* seam in the Eq 18–25 algebra and false of the conv. Those
are two different seams and the report fuses them.

#### It is worse than inert: at `f=16` vs `f=64` it is not a chunk-size test at all

`forward_train_core` reaches `chunk_wy_dispatch` (`module.rs:37-68`), which has three
routes, and the route is selected by something other than `chunk_size` alone:

| route | chosen when | file:line |
|---|---|---|
| `fused_chunk_forward` (fused CUDA kernel) | `feature = "cuda"` **and** `is_cuda::<B>()` and the fused op returns `Some` | `module.rs:51-66` |
| `chunk_wy_forward` → Batched arm | `chunk_size <= TILE(16)` | `forward.rs:151-155`, `forward.rs:72-82` |
| `chunk_wy_forward` → Loop arm | `chunk_size > 16`, or `m_invs` supplied | `forward.rs:151-155` |

So `f=16` vs `f=64` is, on CPU, a **Batched-arm-vs-Loop-arm** comparison, and on CUDA it
compares the fused kernel against the tensor-op path — with the route flipping again at
`f=16`. A failure is therefore ambiguous between "the two arms disagree", "the batched
arm is wrong", and "the fused kernel is wrong", and the report's text does not
acknowledge that ambiguity or that a second failure mode is even reachable. "Free — one
extra config, no new code" is true and irrelevant: the test is not the test.

#### What *would* catch it, and it is one line

`short_conv_1d` has **no direct unit test anywhere** — `grep -rn "short_conv_1d" tests/`
in `dormouse-gdn2` returns nothing. The function with the BUG has zero direct coverage. The
gate is a direct f64 test on the function with a hand-computed expectation:

- `T=1`, `cache=None`: reference `y = w₃·x₀`; ours `y = (w₀+w₁+w₂+w₃)·x₀`.
- `T=1`, `cache=Some(zeros[1,3,C])`: ours `y = w₃·x₀` — the reference's answer.

I traced both by hand through `short_conv.rs:30-64` and they are as stated. This is ~15
lines of CPU-only test, it needs no fixture, no upstream, no GPU, and it fails on the
first line. The report's Tier 0 (export gold vectors from the upstream repo, which needs
torch + Triton + an NVIDIA box) is a much larger project for the same coverage.

#### The sharper statement of the bug, which the report does not make

`Gdn2State::zeros` (`module.rs:104-119`) initialises `conv_q`/`conv_k`/`conv_v` to
**zeros** — `Tensor::zeros([batch, SHORT_CONV_CACHE, kd], device)` at `:115-117`. So the
crate contains **two different answers to the same question**:

| how you say "a fresh state" | conv branch | `T=1` position-0 output |
|---|---|---|
| `state = None` (`short_conv.rs:42-54`) | replicate-pad | `(w₀+w₁+w₂+w₃)·x₀` |
| `state = Some(Gdn2State::zeros(..))` | zero-cache | `w₃·x₀` |

`module.rs:100` documents `Gdn2State::zeros` as "used as a fresh prefill/decoding state".
The two branches disagree at `T=1` and at positions 0–2 for every `T`. That is derivable
from this repository alone, needs no upstream reference, and it is a **self-inconsistency**
rather than a **divergence from a paper** — a strictly stronger claim than the report's,
and one that is immediately testable by pointing the existing decode test at the other
constructor (see §5.2).

---

### 2. Do the two defects mask each other? **Asymmetrically: fixing the layout bug unmasks the padding bug; fixing the padding bug alone changes nothing observable.**

The brief's framing is right that there are two defects and I confirm the layout bug and
its blast radius, which I derived independently.

#### The layout bug, in `tools/gen_reference.rs`

`gen_reference.rs:313` defines the accessor as **head-major**:

```rust
let at = |a: &[f32], h: usize, ti: usize, i: usize, n: usize| a[h * t * n + ti * n + i];
```

Six tensors are read through it. Four of them are correct, because `expand()`
(`gen_reference.rs:292-306`) converts token-major → head-major first: `q`, `k`, `g`, `b`
(`:307-310`). **Two are not**:

- `v` — built at `gen_reference.rs:257-258` as `[T, VD]`, token-major. Never passed through
  `expand`. Read at `:340` as `at(&v, h, ti, vv, V_HEAD)`.
- `w_gate` — built at `gen_reference.rs:284-285` as `[T, VD]`, token-major. Never passed
  through `expand`. Read at `:340` as `at(&w_gate, h, ti, vv, V_HEAD)`.

Both are the **value axis and the write gate** — i.e. exactly the `z_t − S̄ᵀe_t` half of
Eq 10. The indices stay in bounds (`(HV−1)·T·V_HEAD + … = T·VD − 1`), so there is no
out-of-bounds signal; it is a silent permutation of the buffer.

**The Python generator does not have this bug.** `tests/gen_reference.py:110` and `:112`
transpose both:

```python
v = v.reshape(B, T, HV, V_HEAD).transpose(1, 2)
w_gate = w_gate.reshape(B, T, HV, V_HEAD).transpose(1, 2)
```

as it does for `q`, `k`, `g`, `b` at `:107-111`. So `gen_reference.rs`, which
`gen_reference.rs:5-11` calls "a line-for-line port of `tests/gen_reference.py`", is not
one. It diverges on exactly these two tensors.

**Which generator produced the committed fixture?** I reproduced the Rust generator's
splitmix64 + Box-Muller stream (`gen_reference.rs:58-91`) in Python and compared against
the bytes of `tests/ref_data.bin`:

| tensor | committed `.bin` | splitmix64 reproduction |
|---|---|---|
| `q_proj[0..5]` | `0.016344, -0.008746, -0.025791, -0.011164, 0.035219` | identical |
| `A_log` | `0.787259, 2.291152, 2.762833, 2.406840` | identical |
| `dt_bias[0..4]` | `-4.91878, -4.30008, -5.81127, -4.87279` | identical |

The committed 6.8 MB fixture is the **buggy** generator's output, not the Python's.
`bit_exact.rs:13-33` measures the consequence: `1000 cases: max_diff = 1.38e-2, failures =
976/1000`, diff *growing* with sequence length (2.9e-3 at T=3 → 1.2e-2 at T=37), with
**exactly 24 cases passing**. I confirmed 24 independently: `seq_len = (1<<(i%6)) + (i%7)`
is 1 exactly when `i ≡ 0 mod 42`, giving i ∈ {0, 42, …, 966} — 24 values. And T=1 is
precisely where head-major and token-major coincide (`h·1·V_HEAD + 0 + vv` =
`0·VD + h·V_HEAD + vv`), which is why exactly the single-token cases pass. The brief's
premise is confirmed and mechanised.

I also confirmed the range: **T ∈ 1..=38**, not the "2..70" in the comments at
`gen_reference.rs:450` and `gen_reference.py:214`. Both are wrong at both ends. So with
`chunk_size: 64` (`bit_exact.rs:183`) the fixture never exercises a second chunk, never a
ragged tail, and never more than one chunk. It does not test the chunk algebra at all.

#### The masking question, answered

- **At T=1**: the layout bug is a no-op, *and* the padding bug is a no-op **in the
  comparison** — the Rust generator's conv at `gen_reference.rs:134`
  (`if ti + i < 3 { 0 } else { ti + i - 3 }`) reproduces the replicate exactly. So the two
  defects cancel in the *fixture comparison* at T=1. 24 cases pass. Nothing about the
  padding is proven correct by those 24.
- **At T>1**: the layout bug fires and **dominates**. `max_diff` is set by a permutation
  of the value/write-gate tensors, which is an O(1) error, not a boundary-localised one.
  The padding bug's contribution (positions 0–2 only) is buried under it.
- **Fix A alone** (padding → zeros, in burn + both generators): the fixture stays wrong
  for T>1 for the layout reason. `976/1000` failures, `max_diff ≈ 1.38e-2` — **before and
  after, the test is red and the residual is unchanged.** The fix is invisible.
- **Fix B alone** (layout → correct indexing): the padding bug **unmasks**. The residual
  becomes confined to output positions 0, 1, 2 and *shrinks* with T — the opposite
  signature to the current "grows with T" (bit_exact.rs:17-18). The test is still red,
  but now with a clean, tiny, boundary-localised signal that a zero-pad fix then clears
  to noise.

**So: B first, then A, and A is not observable until B is fixed.** The brief's suspicion
is correct and the mechanism is specific: they mask asymmetrically because the layout bug
is a whole-buffer error and the padding bug is a 3-position boundary error.

Corollary the report misses: **fixing the layout bug means regenerating
`ref_data.bin`**, so the `git diff --exit-code` CI check named at `gen_reference.rs:34-36`
and `bit_exact.rs:44-45` will fire once, legitimately. That is a one-time 6.8 MB diff,
not a regression, and whoever lands the layout fix must expect it.

---

### 3. Is `a_log = -3` actionable now, and is it the same class of risk as `use_short_conv`?

**No, they are not the same class, and the report's arithmetic for the `a_log` case is
wrong in the direction that reverses its own conclusion.**

#### 3.1 The report's §4.2 arithmetic is wrong

`gdn-kda.md:389-390` says:

> The K3 paper's `A_h = 0` with `b_α^h ≈ −5` gives `g = −2.5` too at init

`g = g_min·σ(e^{A}·z) = −5·σ(0·(−5))` = `−5·σ(0)` = `−2.5` only if `z = 0`, i.e.
`b_α = 0`. With `b_α ≈ −5` the correct value is `−5·σ(−5) = −5·0.00669 = −0.0335`.
The report computed the `b=0` case and attributed it to `b≈−5`. Concretely, with
`g_min = −5` as the report itself quotes at `:212-215`:

| init | `g` | `α = e^g` | effective context `1/(1−α)` | `α^16` (one K3 tile) |
|---|---|---|---|---|
| **ours**: `a_log=−3`, `b_alpha=+1` | −2.562 | **0.0771** | **1.08 tokens** | 1.6e−18 |
| K3 + FLA: `a_log=0`, `b_alpha=−6.908` (`dt=0.001`) | −0.005 | 0.9950 | 200.8 tokens | 9.2e−1 |
| K3 + FLA: `a_log=0`, `b_alpha=−2.352` (`dt=0.1`) | −0.435 | 0.6476 | 2.8 tokens | 9.6e−4 |
| *(what the report computed: `b=0`)* | −2.5 | 0.0821 | — | — |

`b_alpha` range confirmed from `gen_reference.py:55-59` / `gen_reference.rs:193-201`
(`dt ~ exp(U(ln0.001, ln0.1))`, `inv_dt = dt + log(−expm1(−dt))` → −6.908 … −2.352);
`dormouse-kda` uses `Tensor::ones` at `src/lib.rs:165`.

**The consequence is not "they coincide by accident".** Ours starts with 7.7% per-token
retention; the K3/FLA init starts with 65–99%. The recurrent state underflows to
numerically zero within a couple of tokens under our init, and survives a 16-token tile at
9.2e−1 … 9.6e−4 under theirs. That is a **one-to-two orders of magnitude** difference in
the starting receptive field, and it is arithmetic, not speculation.

#### 3.2 The report's second §4.2 conclusion does not follow either

`gdn-kda.md:393-396` says the near-linear sigmoid means the mapping "degenerates towards
Kimi Linear's softplus mapping". It does not. Kimi Linear is
`g = −e^A·softplus(z) = −0.05·softplus(z)`, which at `z=1` gives `g = −0.034`; ours at
`z=1` gives `g = −2.562`. The two are not converging on anything. What `A=−3` actually
does is **pin the sigmoid at its midpoint** and shrink the reachable logit range 20×:
`g(−1)=−2.438, g(0)=−2.500, g(1)=−2.562, g(10)=−3.110, g(20)=−3.651`. Reaching either
saturation edge needs `|z| ≈ 20`, which a bias initialised at 1.0 and a low-rank
projection at `std = 0.02` (`lib.rs:152-154`) will not produce quickly. The practical
statement is: **the K3 decay sits frozen near `g = −2.5`, the middle of its own legal
range, at initialisation.** The paper's `(−80, 0)` 16-tile argument is computed for a
regime the code never starts in.

#### 3.3 It is live in dormouse, right now

`crates/dormouse-core/src/attention.rs:60-71` builds `dormouse_kda::KdaConfig` with
`..Default::default()`, and `KdaConfig::default()` (`dormouse-kda/src/lib.rs:68-85`) sets
`decay_fn: DecayFn::Sigmoid` and `g_min: G_MIN`. `KdaDecay::new` then sets
`a_log = −3.0` (`lib.rs:170`) and `b_alpha = ones` (`lib.rs:165`). **`AdaptiveAttention`
holds a `KdaModule` (`attention.rs:55`) and that is what `loop_block.rs:384-386` calls.**
So every `use_kda = true` run in this repo's history has run the K3 Eq 5 decay with our
init, not the paper's.

And note the reach: this is a **forward-pass** property, not a learning property. Even
though AGENTS.md §3.2 records that the attention arm took no gradient, `a_log` and
`b_alpha` sit upstream of the state update, so the arm's *output values* in every
`use_kda=true` run were computed with a 7.7%-retention decay. The AGENTS.md retraction
covers the arm not learning; it does not cover the arm having contributed a
numerically-dead-at-init readout to the residual stream.

#### 3.4 The three honest options

1. **Follow K3** (`a_log = 0`, `b_alpha = dt + log(−expm1(−dt))`, `dt ~ logU(0.001,0.1)`).
   This is the only option with a citation that resolves, it makes the `(−80,0)` argument
   applicable, and it is the smallest diff. **But it is an A/B, not a bugfix**: it changes
   the initialisation of a live parameter in the model every `use_kda=true` reading was
   produced with, and AGENTS.md §3.4 already requires the control to be re-baselined
   *before* any arm is judged. Shipping this as an unmeasured "fix" repeats the exact
   failure AGENTS.md §3.2 catalogues — a change to a network charged to a mechanism that
   was never separated from its control. It belongs in `docs/protocols/AB-PROTOCOL.md` as an arm
   with 3 seeds, not in a bugfix commit.
2. **Keep ours, delete the claim.** Fix the comment at `lib.rs:166-169` and `AGENTS.md:274,
   837-839` to say what is true — "chosen locally; *not* a Moonshot or FLA recipe; measured
   α(0) ≈ 0.077, effective starting context ≈ 1 token" — and file the init as an open A/B
   arm. This is the ADR-0020-shaped fix: the defect the report correctly identifies is
   **the citation**, not necessarily the value. The value may well be right for a
   9.2M-param model; there is simply no evidence either way, and the report is right that
   saying otherwise is a retracted-claim-class defect.
3. **Follow FlashKDA.** Not available: `tests/torch_ref.py` takes `A_log` as an argument
   and contains no initialisation, per the report's own §4.2 table (which I accept). There
   is no "FlashKDA init" to follow. Option 3 is empty.

**My recommendation: option 2 now, option 1 as the first A/B arm.** Option 2 is a comment
and a doc fix, carries no numerical risk, discharges the ADR-0020 debt the report
correctly identified, and does not smuggle an unmeasured hyperparameter change into a
tree whose A/B queue has not been re-baselined.

#### 3.5 Not the same risk class as `use_short_conv`

`use_short_conv` was reverted because fp32+AdamW NaN'd at step 60 (AGENTS.md:832-835,
`attention.rs:64-67`). That is a **numerical-instability** risk. `a_log = −3` is a
**bad-starting-point** risk, and the paths do not overlap:

- `a_log` is clamped to `[−10, 20]` (`lib.rs:189-194`), so `e^A ∈ [4.5e−5, 4.8e8]` and is
  always finite.
- Under `DecayFn::Sigmoid`, `g = g_min·σ(...) ∈ (−5, 0)` **by construction**, so
  `α = e^g ∈ (0.0067, 1)`. `exp` of a number in `(−5, 0)` can only underflow toward
  0.0067; there is no overflow path to NaN.
- A *small* `e^A` makes `∂L/∂A` **smaller** (`∂g/∂A = g_min·σ'(·)·e^A·z`), so the parameter
  moves more slowly. Slow is not unstable.

**They must not be batched into one "revert and revisit" item.** And there is a real
coupling in the other direction, which the report never draws and which is worth a line in
the A/B doc: the padding bug is a *candidate* explanation for the reverted NaN, and if it
is, then the padding fix is a **precondition** for re-enabling the conv, not an
independent cleanup. The arithmetic, at the `U(−0.5, 0.5)` conv init
(`module.rs:200-204`, `gen_reference.rs:204-208`): position 0's gain is `|w₀+w₁+w₂+w₃|`
under replicate and `|w₃|` under zero-pad. Over 200k draws the ratio has median 1.81,
p90 8.2, p99 80; **30.3% of channels exceed 3×** and 16.9% exceed 5×. With `VD = 96` that
is ~26–32 value channels at positions 0–2 whose activation is 3× the reference's *at
initialisation* — and `v` is the one projection that is **not** L2-normalised
(`module.rs:540-542` normalises q and k only; `dormouse-kda/src/lib.rs:432-433` likewise), so
it reaches the state unattenuated. **This is a hypothesis, explicitly not a claim** — the
revert was never bisected and the weights had moved by step 60. It is, however, free to
test and it would be a shame to re-enable the conv without having checked it.

---

### 4. Ordering: what must be fixed first for the others to be observable

The report presents 3 BUGs and 27 MATCH as a flat table. It is not flat. Ranked by
dependency and by whether the defect can be observed at all today:

| order | item | why here |
|---|---|---|
| **1** | **the layout bug in `tools/gen_reference.rs`** (`v`, `w_gate` read head-major at `:340`) | nothing else in the fixture is observable until this lands. It owns 976/1000 of the residual and its 24 passing cases are the *only* cases where the padding cancels. Must land first, **and must come with a `ref_data.bin` regeneration**. |
| **2** | **re-enable the fixture tests** | All three fixture tests are behind `#[cfg(feature = "binary-tests")]` (`bit_exact.rs:150`, `test_chunk.rs:19`, `test_chunk.rs:213`) and `binary-tests` is **not** in `default = ["std"]`** (`Cargo.toml`). `cargo test -p dormouse-gdn2` runs **zero** of them. This is why nothing caught the padding: not blindness, **absence**. |
| **3** | **ungate `test_chunk_matches_fused_with_real_decay`** (`test_chunk.rs:212-262`) | It uses **no fixture** — it is a pure chunk-vs-fused-recurrent comparison on random tensors, asserting `d_out < 1e-4` and `d_state < 1e-4`. The single most valuable recurrence gate in the crate, costing milliseconds on CPU, disabled by a flag whose only purpose is to gate the two fixture tests. **This one needs no step 1 and no step 2** — it is a pure availability fix and could be #0. |
| **4** | **the short-conv padding** (`short_conv.rs:43-47`, `gen_reference.py:85`, `gen_reference.rs:134`) | real, but **not observable until 1 and 2 land**. Fixing it before 1 changes no test result. |
| **5** | **the `None` vs `Some(Gdn2State::zeros)` conv-cache inconsistency** (`module.rs:104-119` vs `short_conv.rs:42`) | same root cause as 4, and it is the version of 4 that is provable from this repo alone with no upstream reference. Should be fixed *in the same commit* as 4 — they are the same line of reasoning. |
| **6** | **the `a_log`/`b_alpha` citation** (`dormouse-kda/src/lib.rs:166-169`, `AGENTS.md:274, 837-839`) | **independent of 1–5**. Comment-only fix; zero numerical risk; do it whenever. The *value* change is an A/B arm, not this row. |
| **7** | **fp32 decay gate under `--bf16`** (report §5.3 item 1) | latent: only bites under `--bf16`, and it is unanswerable read-only. Real, but the fix belongs in a dtype audit of the `γ = exp(cumsum g)` chain, not in this batch. |
| **8** | **`tests/autodiff.rs:39` dead loop** (§5.2 below) | one-line fix, no dependency, but low urgency on its own — it becomes valuable the moment `use_short_conv` is re-enabled, because it is currently the crate's *only* conv test. |

**Latent-only, can wait:** row 7 (`--bf16` only) and the short-conv family (rows 4/5,
`use_short_conv = false` in `attention.rs:68`, so live impact is genuinely zero today).
**The report's "impact today is zero" for the padding is correct** — but its stated reason
(replicate-vs-zero is numerically small) is the wrong reason, and the stronger reason,
that both crates' **published defaults** are `use_short_conv: true`
(`config.rs:122`, `dormouse-kda/src/lib.rs:76`) and ship the wrong conv to every external
user, is not stated at all.

---

### 5. My top 3 findings about the code

Independent of the report's conclusions.

#### 5.1 `tools/gen_reference.rs:313` + `:340` — the fixture is wrong for 976/1000 cases, and it is the write-gate half of Eq 10

Covered in §2. Restated as a finding because it is larger than the report's whole BUG
count and the report does not have it: `v` and `w_gate` are token-major `[T, VD]` and are
read with head-major offsets. `q`, `k`, `g`, `b` are correctly converted by `expand()`.
`gen_reference.py:110,112` transposes both, so the two generators are not ports of each
other despite `gen_reference.rs:5-11` saying they are. The committed `ref_data.bin` is the
buggy one (verified by RNG reproduction against the file's bytes). The tests are disabled
rather than red, which is why this survived.

**The one-line tell nobody wrote down:** `gen_reference.rs:22` says the file is
"regenerated with `tests/gen_reference.py`" — no wait, `README.md:22` says that. And it
is false: the Python and the Rust use different RNGs entirely (`torch.manual_seed(1337)`
vs splitmix64 seed 1337), so regenerating with the Python as the README instructs
rewrites all 6.8 MB and would fire the `git diff --exit-code` CI check
(`bit_exact.rs:44-45`) on a spurious diff. The README names the one generator that is
*not* the one that made the fixture, and it names the one that is *not* the one with the
bug. That is a small defect with an outsized blast radius: it sends the next person to
regenerate with the wrong tool.

#### 5.2 `tests/autodiff.rs:39` — a two-iteration loop that runs the same configuration twice

```rust
fn cfg(hidden, heads, head_dim, mode) -> Gdn2Config {          // :13-22
    Gdn2Config { hidden_size: hidden, num_heads: heads, head_dim,
                 use_short_conv: true,                         // :18  <-- hardcoded
                 mode, ..Default::default() }
}

for use_sc in [true, false] {                                   // :39
    let c = cfg(hidden, heads, hk, Gdn2Mode::FusedRecurrent);  // :40  <-- ignores use_sc
    ...
    assert!(diff < 1e-6, "decode != full forward (use_short_conv={use_sc}): ...");  // :64-67
}
```

`use_sc` is read at **exactly one place**, the format string on line 66. It never reaches
the config. The second iteration is a byte-identical rerun of the first, and if it ever
failed the message would say `use_short_conv=false` — a test that lies about which arm it
is testing, in the one test that exists for the short conv. The `use_short_conv = false`
path of `decode_equals_full_forward` has never run.

This is the same class the report is hunting elsewhere (`bit_exact.rs` "verified the tensor
adjoint twice", AGENTS.md §3.2): a test whose second case is a copy of its first. It
costs one line — `cfg(..)` taking `use_sc` as a parameter, or
`Gdn2Config { use_short_conv: use_sc, ..cfg(..) }`.

**And it is the cheapest available gate for the padding bug**, once pointed the right way.
As written it is blind, for a *different* reason than the fixture: the prefill takes
`state = None` (`:49-50`) and the decode loop also starts from `state = None` (`:54`),
so **both sides take the replicate branch** and agree. Change the decode side to start
from `Some(Gdn2State::zeros(&device, batch, hv, hk, v_head, kd, vd))` — the constructor
that already exists at `module.rs:104-119` and whose conv caches are **zeros** — and the
existing test starts comparing zero-pad against replicate-pad. It will fail, and it will
localise the failure to positions 0–2. No fixture, no upstream, no new file.

#### 5.3 The report's §4.1 premise is false, and the honest reason is worse

`gdn-kda.md:341-347`:

> The fixture is therefore *self-consistent with the bug* and the 1000-case bit-exact
> comparison is structurally incapable of detecting it.

The first half is right. The second half is wrong, and `tests/bit_exact.rs:13-33` says so
in the file's own words, fifteen lines from the top, in a `STATUS:` block:

> **STATUS: RED, MEASURED 2026-09-27 AGAINST THIS FIXTURE, AND `binary-tests` IS
> THEREFORE NOT IN THE CRATE'S `default` FEATURES.**
> `1000 cases: max_diff = 1.38e-2, failures = 976/1000` (EPSILON = 5e-4)

The comparison is not "structurally incapable" — it is **switched off**. There is no
running test to be blind. And it is not self-consistent with the implementation; it
disagrees with it on 976 cases, for a reason that has nothing to do with the padding. So:

- The report's causal story ("a self-consistent fixture hides the bug") is wrong.
- The real story ("no fixture test runs at all, and the one real comparison is red for an
  unrelated reason") is worse, because it means the crate has **no** verified path to
  upstream fidelity, and has not had one since at least 2026-09-27.
- The report's Tier 0 recommendation (export gold vectors from the upstream repo) is the
  right destination, but the report does not mention that a prerequisite — re-enabling the
  harness — is a smaller and unblocked piece of work sitting in front of it. The report
  also does not mention the free, fixture-free recurrence gate
  (`test_chunk.rs:212-262`) that is disabled and needs no upstream anything.

§7.2 Tier 1 item 1 is also wrong about what exists: "it already exists —
`tests/test_chunk.rs`". There is no zero-prefill-identity test. `test_chunk.rs:20` is a
**second consumer of the same broken fixture** (also `binary-tests`-gated, also red) and
`test_chunk.rs:214` is chunk-vs-fused. Neither is the property the report means. The
report cites a test file and does not name a test in it that does what it says.

---

### 6. What is still unverified after the report's recommendations are followed in full

Taking §7 at face value — all three tiers implemented, all three "actually wrong" items
fixed:

1. **Whether the attention arm trains at all.** AGENTS.md §3.3 says it is unverified
   (`DM_GDN2_BWD_TRACE=1` `ENTERED` on `ChunkWy::backward` has never been seen) and
   §3.2 retracts every `use_kda=true` reading. The report does not touch this. **Every
   recommendation in it is about the forward and the fixture; none of them makes the
   arm's gradient exist.** Until it does, a corrected decay init has nothing to learn
   from and the A/B that §4.2 needs cannot be run.
2. **Whether the tier-0 fixtures can be produced at all.** `lit_gpt/gdn2.py` needs torch,
   Triton and an NVIDIA box. The report calls Tier 0 "the only real gold" and never costs
   it. If nobody on this box can run it, §7 is unimplemented and §4.1's fix ships on the
   strength of a 20-line CPU test. **The honest gate is the direct `short_conv_1d` unit
   test, and it should not wait on Tier 0.**
3. **The fp32 decay gate under `--bf16`.** Report open question 1, and it is the one that
   matters most, because **K3's central numerical claim is a bf16 claim**: the `(−80, 0)`
   cum-decay over a 16-token tile and the `e^80 < bf16 range` reciprocal argument
   (`gdn-kda.md:212-215`). Both crates compute `g` in the params' dtype
   (`dormouse-gdn2/src/module.rs:508-516`; `dormouse-kda/src/lib.rs:180-201`), so under the
   trainer's `--bf16` the whole `γ = exp(cumsum g)` chain — and therefore the
   `g_min = −5` lower bound that is otherwise a verified match — is outside the regime the
   paper reasoned about. Following the report in full leaves this exactly where it was,
   flagged and unanswered. It deserves an experiment, not a footnote.
4. **Whether the padding is what NaN'd `use_short_conv`.** Never bisected. §3.5's
   arithmetic makes it a live candidate (~30% of value channels at 3× the reference tap at
   init, `v` unnormalised). If it is, then re-enabling the conv without fixing the
   padding re-runs the experiment that already failed once. Unverified, cheap to check.
5. **The `min_decay` / `max_ortho` interaction with a changed decay init.** `module.rs:
   518-528` composes a floor on top of `g`; a changed `a_log` changes what that floor is
   clamping. The report's row 27 says the clamp "bounds the damage of row 25 but does not
   fix it" and moves on. Nobody has looked at what the two do together.
6. **Seed determinism.** AGENTS.md §3.7: after `4b42b6d` a repeated run still differs in
   409 043 values (~4% of the model). "3 seeds per arm" (§1.2) is *nearly* implementable.
   None of the report's A/Bs — including the `a_log` one it recommends — are
   reproducible until that reaches zero. The report proposes A/B work without noting
   that the instrument it depends on is not yet sound.

---

### Appendix — what I actually checked, and how

Read-only, no build, no GPU.

- **Traced by hand, both branches:** `short_conv.rs:22-64`, `module.rs:104-119`,
  `module.rs:291-292`, `module.rs:346-353`, `module.rs:469-583`,
  `dormouse-kda/src/lib.rs:412-449`. Established: conv is upstream of chunking in both
  crates; `None` and `Some(Gdn2State::zeros)` give different `T=1` answers.
- **Counted in the environment:** `cargo`-independent greps for `short_conv_1d` in
  `tests/` (0 hits), for `#[cfg(feature = "binary-tests")]` (3 sites), and
  `grep -n "default = " Cargo.toml` (`default = ["std"]`). No crate was built.
- **Computed:** the 24 single-token cases and `T ∈ 1..=38` from
  `seq_len = (1<<(i%6)) + (i%7)`; the `α`/`g` table in §3.1; the conv-init gain
  distribution (200k draws) in §3.5.
- **Verified against the committed bytes:** reproduced `gen_reference.rs`'s splitmix64 +
  Box-Muller stream in Python and matched `q_proj`, `A_log` and `dt_bias` against
  `tests/ref_data.bin` — establishing that the buggy Rust generator, not the Python,
  produced the shipped fixture. Read-only read of a 6.8 MB file; nothing was written.
- **Accepted from the report without re-checking** (per the brief): all arXiv ids, the
  upstream `other=0.0` / `cache = new_zeros` claims, K3 Eq 5/6 text, FLA's
  `dt_bias` formula, FlashKDA's `CHUNK=16` / `LOWER_BOUND=-5`. Where I use them (§3.1) I
  use them as inputs to arithmetic, and I flag where the report's use of them is wrong.

**What I did not check:** I did not run `bit_exact`, so the `976/1000` figure is the
file's own recorded measurement, not one I re-measured — though the 24-case count, the
T-range and the T=1 layout-coincidence that explain it are all mine and all check out.
I did not read `autodiff.rs` (34 KB) for the L2-VJP question the report flags as
speculation; I have no opinion on it.

---

**Bottom line for the engineering reader.** The padding bug is real and the report found
it. The report's cheapest recommendation cannot detect it, its stated reason the bug
survived is false, the arithmetic in its second finding is wrong in the direction that
understates the problem, and it misses a larger bug in the same file that is currently
switched off rather than passing. Fix `gen_reference.rs` first, re-enable the harness
second, pad third — and change no hyperparameter until the control is re-baselined.

Status: complete
Integrity: clean
Contract: aligned
