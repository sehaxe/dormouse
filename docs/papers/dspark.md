# DSpark — paper verification

**Fetch date:** 2026-09-29. **Tree:** `eeb3b73`, working dir `/home/sehaxe/dormouse`.
**Subject:** `vendor/burn-fused/crates/burn-dspark/` + the call sites in
`crates/dormouse-core/src/{aux.rs, model.rs}`.

---

## 0. Provenance — the citation is real, and it is DeepSeek's own

| arXiv | title | resolves? | v / date | authors |
|---|---|---|---|---|
| **2607.05147** | *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation* | **✓ v1, 2026-07-06, cs.AI** | DeepSeek-AI + Peking University, 31 authors, first author **Xin Cheng** (also first author of the Engram paper) |

Code repo named by the paper: `https://github.com/deepseek-ai/DeepSpec`
("an algorithm-driven training repository for speculative decoding").
`burn-dspark/src/lib.rs:5-9` cites the paper, names DeepSpec, and **says in the
crate header that it is NOT matched against DeepSpec**. That self-declaration is
the single most useful sentence in the file and it is honest.

**Not fabricated. And the loss port is faithful** — see D3-D8 below. The real
problems are elsewhere, and they are real.

---

## 1. The K-step formulation, transcribed

**§3.1 Sequential stage — Eq 4 (the block factorization).**
`P(X | x₀) = Π_{k=1}^{γ} p_k(x_k | x₀, x_{<k})`, with
`p_k(v | x₀, x_{<k}) = exp(U_k(v) + B_k(x₀, x_{<k}, v)) / Σ_{u∈V} exp(U_k(u) + B_k(x₀, x_{<k}, u))`
— `x₀` is the **anchor** (the target model's last token from the previous round),
`U_k` are the **parallel backbone's** base logits at draft position `k`, and
`B_k` is a transition bias added in logit space. Note the architecture: the
expensive part is fully parallel; only `B` is sequential, and
`T_sequential ≪ T_parallel` is a hard design requirement.

**Eq 5 — Markov head (the default).** `B(x_{k−1}, ·) = W₁[x_{k−1}] W₂ ∈ ℝ^V`,
`W₁ ∈ ℝ^{V×r}`, `W₂ ∈ ℝ^{r×V}`, **`r = 256` by default**. `W₁` is an embedding
lookup, `W₂` a logit projection.

**Eq 6 — RNN head.** `z_k = [s_{k−1}; W₁[x_{k−1}]; h_k] ∈ ℝ^{2r+d}`, then
`s_k = σ(W_g z_k) ⊙ s_{k−1} + (1 − σ(W_g z_k)) ⊙ tanh(W_c z_k)`, and
`B_k(x_{<k}, ·) = W₂ᵀ tanh(W_o z_k)`, where `W_g, W_c, W_o ∈ ℝ^{r×(2r+d)}` are
**jointly parameterized by a single linear projection split into three**.
`s₀ = 0`.

**Eq 7 — confidence head.** `c_k = σ( wᵀ [h_k ; W₁[x_{k−1}]] ) ∈ (0,1)`, the
*conditional* probability that draft position `k` survives verification.

**Eq 8 — the soft acceptance label.**
`c_k* = 1 − ½ ‖p_k^d − p_k^t‖₁`.

**§3.2.1 Post-hoc Calibration — Sequential Temperature Scaling.** Because
`c_i` are conditional, the prefix survival probability factorizes as
`Π_{i≤k} c_i`; STS calibrates that product left-to-right by a **1-D grid search
on ECE at each position, holding the already-calibrated prefix fixed**.

**§3.3 Training objective.** *"we **randomly sample** multiple anchor positions
from each target sequence to form γ-token blocks"*; the target is frozen
throughout.

**Eq 9.** `L_ce = − Σ_{k=1}^{γ} w_k log p_k^d(x_k*)`
**Eq 10.** `L_tv =  Σ_{k=1}^{γ} w_k ‖p_k^d − p_k^t‖₁`
**Eq 11.** `L_conf = − Σ_{k=1}^{γ} w_k [ c_k* log c_k + (1−c_k*) log(1−c_k) ]`
**Eq 12.** `L = α_ce L_ce + α_tv L_tv + α_conf L_conf`, with
**`α_ce = 0.1, α_tv = 0.9, α_conf = 1.0`** (defaults).
**Position weights.** `w_k = exp(−(k−1)/γ)`, attributed to DFlash (Chen et al.
2026), "which emphasizes earlier block positions that contribute more to the
expected acceptance length under prefix-based verification."

**§4.1 the operating point.** Qwen3-{4B, 8B, 14B} + Gemma4-12B targets;
**block size γ = 7**; 5 drafter layers; *"Unless otherwise stated, DSpark denotes
the **Markov-head** variant; we study the RNN-head variant in §4.3.2."*

**γ is overloaded in this paper and the overload is load-bearing:** it is the
block size (§2.1 "proposes γ candidate tokens"), the position-decay denominator
(Eq 9-11), and the head count. There is one number.

---

## 2. **THE OFF-BY-ONE IS STILL THERE.**

AGENTS.md §3.3: *"DSpark's window is one position shifted (`model.rs:197-203`):
the draft head's step s is fed x[p+s+1] and trained to emit x[p+s+2]. Fixing it
changes every DSpark number, so it needs its own A/B."*

**Confirmed open, at exactly the cited lines.**

`crates/dormouse-core/src/model.rs:197-204`:

```rust
// DSpark's window tokens only (the JEPA teacher's input is
// `input_ids` above, see `forward_with_hidden`). KNOWN, NOT FIXED:
// these are the LABEL sequence, so the draft head's step s is fed
// x[p+s+1] and trained to emit x[p+s+2] - a one-position shift
// against the sequence the model actually consumed. ...
let ids_raw = targets.clone();
```

### Tracing it exactly (`aux.rs:263-282`)

With `anchors p = i·stride`, `s ∈ 0..k`, and `ids = targets` (so
**`y[i] = x[i+1]`**):

| line | gathered | resolves to |
|---|---|---|
| `aux.rs:277` `token_cols[s] = ids[p+s]` | `y[p+s]` | **`x[p+s+1]`** ← the prev-token feature |
| `aux.rs:280` `id_cols[s] = ids[p+s+1]` | `y[p+s+1]` | **`x[p+s+2]`** ← the supervision target |
| `aux.rs:278` `hidden_cols[s] = hidden[p+s]` | — | encodes `x[0..=p+s]` |
| `aux.rs:279` `base_cols[s] = logits[p+s]` | — | the model's prediction of **`x[p+s+1]`** |
| `aux.rs:281` `target_cols[s] = frozen[p+s+1]` | — | the target distribution for **`x[p+s+2]`** |

So the head's Markov/recurrent feature `W₁[x_{k−1}]` is a function of **the very
token its own base logits already predict**, and it is then asked to move the
prediction one step further, to a token that no bias on that feature can
address. The RNN state compounds it: `s_k` accumulates a chain of
`x[p+s+1]`-derived embeddings and supervises against `x[p+s+2]`.

### The fix is one line, and the code comment understates the problem

```rust
let ids_raw = input_ids.clone();   // was: targets.clone()
```

With `input_ids` every quantity lines up against paper Eq 4:

| | prev token | base logits at `p+s` predict | CE target | TV target dist |
|---|---|---|---|---|
| **with `input_ids`** | `x[p+s]` ✓ (Eq 5/6: `x_{k−1}`) | `x[p+s+1]` | `x[p+s+1]` ✓ (Eq 9) | for `x[p+s+1]` ✓ (Eq 10) |
| with `targets` (today) | `x[p+s+1]` ✗ | `x[p+s+1]` | `x[p+s+2]` ✗ | for `x[p+s+2]` ✗ |

**The window arithmetic in `dspark_aux_loss` is correct.** `pos`, `pos+1`, the
gather order, the stacking — all of it is right. The *only* wrong thing is which
tensor is passed in. That is a better situation than the AGENTS.md note
implies, and it is worth saying so: this is a one-token fix, not a redesign.

**Verdict: BUG, open, correctly and honestly documented.** The comment's
diagnosis ("these are the LABEL sequence") is right; it just stops one sentence
short of "which is the only wrong thing".

**Not done, correctly:** the comment declines to fix it here ("not a drive-by in
a JEPA bugfix"), and the A/B is the right gate. But note the interaction with
D2 and D9 below: three independent changes to this objective are now stacked,
and any A/B that fixes one will not isolate it from the others.

---

## 3. **`dspark_stride` — what it actually does**

**Found.** It is the **anchor spacing**: the number of sequence positions
between consecutive draft windows.

- `config/schema.rs:73` — `fn d_dspark_stride() -> usize { 16 }` (default)
- `aux.rs:253` — `let n = if t > k + 1 { (t - k - 1) / stride.max(1) } else { 0 };`
  → number of anchors that fit
- `aux.rs:257-258` — `arange(0, n).mul_scalar(stride)` → anchors at
  **p = 0, 16, 32, …**
- `aux.rs:313` — `mask` is all-ones; `stride` does not enter the loss at all
- `config/validation.rs:22-25` — refuses `stride == 0` ("places every draft
  anchor at position 0"), with a test at line 92

At the `small` recipe (t = 512, k = 4, stride = 16): **31 anchors**, 4 steps
each, 124 supervised positions per sequence.

**So, explicitly, because the paper overloads γ and the field name invites the
error:**

| symbol | what it is | where |
|---|---|---|
| `dspark_k` | **block size** = the paper's γ in Eq 4/9-11 | `schema.rs:110`, default 4 |
| `DSPARK_GAMMA` | **decay denominator** only, a hardcoded `const` | `aux.rs:27`, 4.0 — **not a config field** |
| `dspark_stride` | **anchor spacing** — not in the paper at all | `schema.rs:111`, default 16 |

`dspark_stride` is the one the paper does *not* have, because the paper
**randomly samples** anchor positions (§3.3). We stride them deterministically,
which is right for this repo — a random anchor draw is a reproducibility
hazard of exactly the class ADR-0021 exists to kill (see the same reasoning at
`aux.rs:50-58` for the JEPA mask). So the *mechanism* is a deliberate and
justified substitution.

**Verdict: BUG (documentation).** It is functionally correct, validated, and
tested. It is documented **nowhere**: `schema.rs:111` carries no doc comment at
all, while its neighbours `dspark_weight`/`dspark_k` (`schema.rs:109-110`) have
none either. AGENTS.md §3.3's "a config field with NO documented meaning
anywhere" is accurate. Given that the same paper uses γ for three things and
the repo has a second hardcoded γ three fields away, this is a collision waiting
to happen. A one-line doc comment — "anchor spacing: consecutive draft windows
start every `stride` positions; not the paper's γ, which is `dspark_k`" —
dissolves it.

---

## 4. Delta table

| # | our code | paper says | verdict |
|---|---|---|---|
| D1 | `model.rs:204` `ids_raw = targets.clone()` | Eq 4/5/6/7 all condition on **`x_{k−1}`, the previous token of the same block** | **BUG — OPEN.** §2 above. One-line fix. Correctly documented as known. |
| D2 | `aux.rs:27` `DSPARK_GAMMA = 4.0`; `schema.rs:110` `dspark_k = 4` | paper has **one** γ: block size *and* decay denominator, **γ = 7** | **BUG — the objective's shape is wrong.** We decoupled two numbers the paper ties together, and picked a decay 1.75× steeper. `w = [1, .78, .61, .47]` vs the paper's `w = [1, .87, .75, .65, .57, .49, .43]` at γ=7. `DSPARK_GAMMA` is a hardcoded `const`, not a config field, so it cannot be swept without touching source. |
| D3 | `lib.rs:137-142` `position_weights` = `exp(−k/γ)`, k 0-based | `w_k = exp(−(k−1)/γ)`, k 1-based | ✓ **BENIGN — correct**, these are the same function. The 0-based/1-based conversion is stated in the doc comment. |
| D4 | `lib.rs:216` `0.1*L_ce + 0.9*L_tv + 1.0*L_conf` | Eq 12, `α_ce=0.1, α_tv=0.9, α_conf=1.0` | ✓ **BENIGN — exact.** |
| D5 | `lib.rs:186-191` CE on `log_softmax(draft_logits)` gathered at the target | Eq 9 | ✓ **BENIGN — exact**, and note it gathers rather than materialising a one-hot (§3.8 of AGENTS.md). |
| D6 | `lib.rs:194-197` `tv = ‖softmax(draft) − softmax(target)‖₁` | Eq 10, on the **probabilities** | ✓ **BENIGN — exact, and it is the easy one to get wrong.** The crate header names "L1-on-logits instead of L1-on-probs" as a failure its tests cannot detect; the code does it correctly. |
| D7 | `lib.rs:151-156` `c* = clamp(1 − ½‖p_d − p_t‖₁, 0, 1)` | Eq 8 | ✓ **BENIGN — exact.** `accept_rate_target_bounds` (`lib.rs:334-351`) checks both endpoints and derives `1/V` for one-hot-vs-uniform by hand. Good test. |
| D8 | `markov.rs:185-199` `z = [s; W₁[x_{k−1}]; h_k]`, joint `Linear(2r+d → 3r)` split into `[gate; candidate; output]`, `s_k = σ(g)⊙s_{k−1} + (1−σ(g))⊙tanh(c)`, `B = W₂ tanh(output)`, `s₀ = 0` | Eq 6, verbatim | ✓ **BENIGN — a faithful port**, including the `W_g, W_c, W_o` joint-projection split. |
| D9 | `aux.rs:45` `AcceptRatePredictor::new(d_model, …)`; `aux.rs:311` `.prob(hidden_win, **None**)` | **Eq 7: `c_k = σ(wᵀ[h_k ; W₁[x_{k−1}]])`** | **BUG — the confidence head is not the paper's confidence head.** `AcceptRatePredictor::with_markov` (`lib.rs:83-93`) implements Eq 7 exactly and **has zero callers in `crates/`**; the wired path passes `None` and gets the hidden-only variant. The paper is explicit that the Markov feature is what ties the head to non-anticipation (§3.2.2, and Appendix A's selection-bias counterexample is *about* this feature). Losing it makes `c_k` a function of `h_k` alone, which is not measurable to a standard, because `h_k` does not know which candidate was drafted. This is exactly the "a wrong loss term" class the crate header says its 7 tests cannot catch. |
| D10 | `aux.rs:253,257` anchors at `i·stride` | §3.3 "**randomly sample** multiple anchor positions" | **DELIBERATE, correct, and undocumented.** Deterministic striding is right for ADR-0021 reproducibility (the same reasoning the JEPA mask stream uses at `aux.rs:50-58`). See §3 above. |
| D11 | `lib.rs:184` `den = Σ w_k·mask`; each term divided by it | paper **sums** over k | **BENIGN — a constant rescale.** All three terms share the divisor, so the α ratio is preserved; only the overall magnitude moves, and `dspark_weight = 0.1` absorbs it. |
| D12 | `aux.rs:320` `DSPARK_GAMMA` passed at the call site | — | **BENIGN** — it *is* threaded as a parameter, so the coupling is honest. It is just not a config field (D2). |
| D13 | `lib.rs:240-274` `sts_calibrate` | §3.2.1 STS: left-to-right 1-D grid search on the cumulative product, prefix held fixed | **BENIGN-in-form, DEAD.** The structure matches the paper. But its ECE is `mean |cum − target|` (`lib.rs:261`), not a binned ECE, and it has **zero callers in `crates/`**. Dormouse has no serving scheduler, so this is correctly unwired — but it is ~35 lines of unexercised host code in a crate whose header apologises for having unverified tests. |
| D14 | `markov.rs:83-102` `sample_block_tokens` autoregressive within block | §3.1 "the sequential block samples left to right according to `p_k(·|x₀,x_{<k})`" | ✓ **BENIGN — correct**, and it threads `prev_ids` exactly as the paper's factorization requires. Unused by the trainer (which is teacher-forced, per `aux.rs:261-262`), correct for training. |
| D15 | `aux.rs:259` `frozen = logits.detach()` | §3.3 "the **target** model is frozen throughout training" | ✓ **BENIGN — correct**, and deliberately so: `aux.rs:234-236` says hidden carries grad into the backbone while logits enter detached. The draft and target are the same network here, so "frozen target" means frozen *logits*, not frozen parameters — a defensible reading, stated in the comment. |
| D16 | `aux.rs:308` the RNN head trains jointly with the backbone | §3.3 "updating only the backbone drafter, sequential block, and confidence head" | ✓ **BENIGN — matches.** |

**Summary: the LOSS is a faithful port (D3-D8 exact). The WIRING has two real
bugs (D1 off-by-one, D9 missing Markov feature) and one shape error (D2
decoupled γ), and the one field with no documentation is `dspark_stride`.**

---

## 5. What this costs us, in one paragraph

The single most valuable thing in this crate is its honesty: `lib.rs:6-9` says
out loud that the tests are hand-derived and cannot detect a wrong loss term, a
wrong `gamma`, or L1-on-logits. **That warning is accurate, and D2 and D9 are
both inside the class it names.** The loss port is exact; the two places the
paper puts a *feature* — the Markov embedding in the RNN head (D8, present and
correct) and the same embedding in the confidence head (D9, **absent**) — are
where the implementation drifted. The DSpark paper's central mechanism is
"inject the previous sampled token"; we wired it into the drafter and dropped it
from the confidence head, which is the half that decides how much of the block
gets verified. And the D1 shift is still open, on top of D2, on top of D9. **Any
A/B of the DSpark head right now would move three things at once and attribute
the result to none of them.**

## 6. Rejected / not adopted

- **The Hardware-Aware Prefix Scheduler (Algorithm 1)** — correctly absent.
  Dormouse is a trainer, not a serving engine; there is no `SPS(B)` capacity
  curve to profile. `sts_calibrate` (D13) is its one trainable-free prerequisite
  and is also unused.
- **DFlash / Eagle3 comparison (Chen et al. 2026, Li et al. 2026b)** — the
  paper's `w_k` is explicitly attributed to DFlash, and we use the formula
  without the citation. Minor: worth naming in the doc comment, since §3.3 is
  where the attribution lives.
- **Parallel drafter backbone** — we reuse the backbone itself as `U_k` rather
  than a 5-layer DFlash. That is the point of the exercise (an aux head, not a
  second model) and costs nothing in fidelity to the sequential stage, which is
  where all of Eq 4-11 lives.

## 7. Recommended gold-vector test

`burn-dspark` has 7 tests: 5 shape, 1 decay-monotonicity, 1 bounds. **None
checks a value against a reference.** The one that gets closest
(`accept_rate_target_bounds`, `lib.rs:334-351`) hand-derives `1/V` and gets it
right. Add:

```rust
// crates/dormouse-core/tests/dspark_gold.rs

/// 1. THE OFF-BY-ONE GATE. Build a 1-row sequence where x[i] = i, so
///    targets y[i] = i+1 and every position is identifiable. Then assert, for
///    anchor p and step s, that the tensor fed to the RNN head at (p, s) is
///    x[p+s] and the CE target is x[p+s+1]  -- i.e. paper Eq 4/5/6's `x_{k-1}`
///    and Eq 9's `x_k*`. Today this FAILS by one. It should be the test that
///    fails before the fix and passes after, and it must not be a tolerance
///    test: with x[i] = i every value is exact and an off-by-one is
///    unmistakable.
/// 2. Eq 6 against a hand-rolled host reference: three positions, fixed
///    weights, s_0 = 0, step through gate/candidate/output. Pins the
///    `W_g, W_c, W_o` joint-split and the s_0 = 0 init.
/// 3. Eq 7 wire check. Assert that the confidence head's Linear input width is
///    `d_model + markov_rank` on the training path, not `d_model`. A width
///    assertion, not a value: it is what catches D9 regressing, and it is
///    zero-tolerance.
```

(1) is the one to write first: it is the cheapest possible gate for the open
defect, it fails loudly on the current tree, and it is the test the D1 fix
should have to go green.

## 8. Open questions

1. **Is γ = 7 or γ = 4 the right operating point here?** The paper's γ=7 was
   chosen for a 112M-param QwenNext-style drafter serving Qwen3-4B. Dormouse is
   a 9.2M byte-LM with `d_model=768` and `dspark_k=4`. D2 says the two knobs
   should move together; nothing here says they should.
2. **Does the D1 shift, once fixed, need its own A/B, or does it fold into the
   "do the aux heads earn their share of the step" arm in the A/B queue?** It
   changes the objective's meaning, so it probably needs its own — but three
   stacked changes make that expensive. Recommend fixing D1 + D9 together behind
   two separate tests, then running one A/B of the whole corrected head against
   pure CE, and treating D1/D9 as a single "the head was wired wrong" change.
3. **Is `RNNHead` the right choice when the paper's default is the Markov head?**
   §4.1: "Unless otherwise stated, DSpark denotes the **Markov-head** variant."
   `aux.rs:36` picks `RNNHead`. That is a deliberate upgrade (more capacity from
   the same `W₁`), and D8 shows it is a faithful port — but it is a deviation
   from the paper's *default*, and the paper's §4.3.2 compares the two. Not
   documented as a choice anywhere in the tree.
