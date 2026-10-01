# Review of `engram.md` / `jepa.md` / `dspark.md`

**Reviewer:** independent, 2026-09-29. **Tree reviewed:** `eeb3b73`, working dir
`/home/sehaxe/dormouse` (the same tree the three reports name).
**What I did:** fetched every paper myself (arXiv API + full HTML), fetched both
DeepSeek reference repositories myself, and read every `file:line` the reports
cite. No GPU, no build, no test run. Everything below is a command or a
quotation, not a recollection.

**Headline:** 6 of the 9 claims are confirmed with exact line numbers. **Two are
confirmed as facts but overturned as verdicts** (E11/D2), **one is split** —
J5 is confirmed and strengthened, J4 is wrong and inverted. The three reports
are honest about what they did not read, which is to their credit; but the two
places they marked ⚠️ are exactly the two places they got wrong, and the one
place they claimed to have a true external oracle they used on 6 of ~30 delta
rows and never mentioned that a **second** oracle, of the same kind, exists for
DSpark and was not opened.

---

## 1. Verdict per claim

| # | claim | verdict |
|---|---|---|
| 1 | `engram_demo_v1.py` exists and is what is claimed | **CONFIRMED, byte-exact** |
| 2 | 2601.16531 real; every quoted number correct | **CONFIRMED, all five** |
| 3 | 196B/552B absent from 2601.07372v2; 24% is really 32% | **CONFIRMED** (plus a third error the report missed) |
| 4 | JEPA teacher defect fixed and gated | **CONFIRMED, non-vacuous** |
| 5 | DSpark off-by-one still there at `model.rs:197-204` | **CONFIRMED, exact** |
| 6 | `dspark_stride` = anchor spacing; paper samples randomly | **CONFIRMED** |
| 7 | D9: `None` passed, `with_markov` uncalled, Eq 7 quoted right | **fact CONFIRMED / verdict OVERTURNED** |
| 8 | E11: learned `w_mem` clamped, comment says "constant" | **fact CONFIRMED / severity OVERTURNED** |
| 9 | J4 misattribution + J5 raw un-normalised regression | **J5 CONFIRMED and strengthened / J4 OVERTURNED and inverted** |

Two claims the task brief did not list are also overturned below: **D2**
(γ decoupling) and **J3** (momentum ramp).

### Claim 1 — the reference implementation: CONFIRMED

```
$ curl -sSL -o engram_demo_v1.py \
    https://raw.githubusercontent.com/deepseek-ai/Engram/main/engram_demo_v1.py
HTTP 200 | 15017 bytes
```

423 lines. `github.com/deepseek-ai/Engram` also ships `Engram_paper.pdf`,
`README.md`, `drawio/Engram.drawio` and five figures. Every fragment the report
quotes matches byte-for-byte:

| report's claim | demo |
|---|---|
| `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign(); gate = gate.sigmoid()` | `engram_demo_v1.py:372-373` — exact |
| `/ math.sqrt(hidden_size)` | `:371` — exact |
| "the demo does NOT zero-init" | `:138-146` `nn.Conv1d(...)`, default init — **correct** |
| pads `(k−1)·dilation` then truncates `[..., :T]` | `:144`, `:173` — exact |
| `input_ids + offsets` then one `nn.Embedding` | `:311-322` — exact |
| `mix = t[0]*m[0]; mix ^= t[k]*m[k]; hash = mix % prime` | `:285-287`, `:293` — exact |
| primes from `vocab_size − 1`, `seen_primes` global across (n,head) **and layers** | `:236-255` — exact |
| `D = n_embed_per_ngram // n_head_per_ngram = 512//8 = 64` | `:342` — exact |

So: the oracle is real, it is what the report says it is, and the report read
it. That is the strongest thing in the three documents.

### Claim 2 — 2601.16531: CONFIRMED, every number

arXiv API: *A Collision-Free Hot-Tier Extension for Engram-Style Conditional
Memory*, Tao Lin, submitted 2026-01-23, v2 2026-01-26. Metadata as reported.

Table 3, verbatim:

```
Config        val_loss  std     hot_cold_delta  Throughput  Params
Hash-500K     4.4809    0.0082  +0.07           ~1910       313,567,232
Nine-100/400K 4.4799    0.0123  +0.10           ~1693       313,567,232
Hash-300K     4.4825    –       +0.08           ~1917       313,567,232
Hash-800K     4.4961    –       +0.08           ~1917       313,567,232
```

§5.1 prose, verbatim: *"its advantage over Hash-500K is only 0.001, far smaller
than the measurement standard deviation (0.008–0.012), and is not statistically
significant."* ✓
§5.4.2, verbatim: *"approximately 70% of the high α bucket are hot positions
(Hash 76%, Nine 69%)"*, *"Low α (0–0.4) positions have loss around 3.9, while
high α (0.8–1.0) positions have loss as high as 5.1–5.3"*, and it is **Table 6**
(0.2–0.4 → 3.90, 0.8–1.0 → 5.28, hot proportion 76.2%). ✓ All three of the
report's nits are right.

**And the report's misquote catch is right, and matters more than it says.**
`schema.rs:122` says "125M backbone". The string `125M` / `125 M` appears
**zero times** in 2601.16531v2. Table 1 says *"GPT-2 architecture (~185M params†,
12 layers, 768 dim)"* and the footnote gives
*"Total = GPT-2 backbone (~185M) + shared Engram embeddings (128M)"*.
The report says the correction "makes the argument stronger, not weaker" — it
does more than that: the schema's own ratio `125/7.5 = 16x` becomes `185/7.5 =
24.7x`, so the repo understates its own over-allocation by 1.5×.

### Claim 3 — 196B/552B absent; the arithmetic: CONFIRMED

I fetched `https://arxiv.org/html/2601.07372v2` (HTTP 200, 477 956 B) and
stripped it to 112 543 characters.

- `"552"` → **0 occurrences.**
- `"196"` → **1 occurrence**: *"Princess of Wales (1961-1997), the first wife of
  Prince Charles"* — a benchmark example in a table, not a parameter count.
- Largest models: Dense-4B (4.1 B), MoE-27B (26.7 B), Engram-27B (26.7 B, 5.7 B
  Engram, ρ=74.3 %), Engram-40B (39.5 B, **18.5 B** Engram).

The report's strong claim — "the largest model there is 39.5B/18.5B" — is
**exactly right**. Table 5 also confirms `Engram Layer [2,15]`, `d_mem 1280`,
`Num Head 8`, `N-gram [2,3]`, and `Engram Vocab Size 2262400 / 7239680`.

`schema.rs:132-133` carries the 196B/552B pair with **no citation on the
sentence at all**, and the *nearest* citation in that doc comment is
`schema.rs:122`, which is **2601.16531**, not 2601.07372 — 2601.07372 appears
only at `schema.rs:159`, twenty-six lines later, in an unrelated doc comment.
The report's E12 says the number is "attributed to the paragraph containing
arXiv 2601.07372"; that attribution is itself wrong, which makes the finding
*worse*: a reader skimming the comment would more plausibly mis-belong the
number to 2601.16531, which is a 313 M-param GPT-2 study that also has no
552 B model. (It also has no 196 B — the strongest numbers in it are 313 M
total.)

Arithmetic: `2 400 000 / 7 500 000 = 0.320 = 32 %`, not 24 %. The same
sentence's `0.32x` is right. **CONFIRMED** — two numbers in one sentence cannot
both hold, and `schema.rs:135` holds both.

**A third error in the same comment, which the report missed.**
`schema.rs:138-140` continues: *"25_000 -> 32_768 rows, 3.1M params, **29% of
the model**."* With `engram_dim = 32`, `engram_orders = [2,3,4]`
(`schema.rs:75-76`), the real figure is `32768 × 3 × 32 = 3 145 728`, which is
**34.2 %** of the measured `small` count (9 195 854) or **41.9 %** of the
comment's own 7.5 M denominator. 29 % is wrong under every reading. So the
comment carries three mutually inconsistent percentages (24 %, 32 %, 29 %) for
the same quantity, and the one the report did *not* check is the one that
describes the code that actually runs.

### Claim 4 — JEPA teacher fixed: CONFIRMED

`model.rs:150-156`, verbatim:

```rust
let teacher_latent = teacher.zip(targets.as_ref()).map(|(t, _)| {
    t.forward_latent::<B>(
        input_ids.clone(),      // the student's INPUT
        hashed_ids.clone(),     // the student's KEYS
        host_rows.clone().map(|r| r.detach()),
    )
});
```

`targets` is used **only** as the `Option` inside `zip`. The label tensor is
never read. Exact line numbers as reported.

The gate is real and it is good. `tests/jepa_teacher_seam.rs`:
`teacher_target_is_the_student_latent` at **line 109** (report: 109 ✓), with
the 10×-sensitivity vacuity guard at **lines 171-176** (report: 171 ✓) and the
latent-separation guard at **lines 177-181** (report: 177 ✓); mask pinned to
`frac = 1.0` at line 114. `teacher_engram_arm_runs` at **line 196** (report:
196 ✓) is an **integer** `assert_eq!` on `probe::ENGRAM_KEYS` with no tolerance.

I checked the obvious hole — *would this gate actually catch the regression?*
Yes. `correct` is `forward_with_jepa_targets(x, h, None, y, Some(forward_latent(x,h,None)))`.
Regressing `input_ids.clone()` → `targets.clone()` makes `got` the aux of a
teacher fed `y`, which the file's own comments measure at ~1500× tol, so
`|got − correct| ≤ tol` fails. Regressing `hashed_ids` → `None` zeroes the
teacher's `ENGRAM_KEYS` contribution and the integer assert fails. **This is
the one item in the three reports that is unambiguously done, and the report is
right to say so.**

One thing the report does not say, and a reader should: the *regression-
specific* quantities `wrong` (labels) and `no_keys` (inert engram) are
**printed, not asserted** — `jepa_teacher_seam.rs:148-157` says so in as many
words ("a flaky guard, not a pin"). The test still catches both regressions
(above), but only because the *reference* side is pinned, not because the
failure cases are.

### Claim 5 — DSpark off-by-one open: CONFIRMED

`model.rs:204` — `let ids_raw = targets.clone();` — exact, and the
`KNOWN, NOT FIXED` comment at 197-203 is accurate.

The report's trace table is exact. `aux.rs:277` `token_cols.push(ids.gather(1, g2))`
→ `ids[p+s]`; `aux.rs:280` `id_cols.push(ids.gather(1, nxt))` → `ids[p+s+1]`;
`aux.rs:278` hidden; `aux.rs:279` base logits; `aux.rs:281` target dist. All at
the cited lines.

And the "one-line fix" claim is right, for a reason the report does not give:
the *window arithmetic* is correct. `pos = anchors + s`, then `nxt = pos + 1`,
and the only wrong input is which tensor `ids` names. With `ids = input_ids`,
every row of the report's Eq-4 alignment table lines up. Confirmed.

### Claim 6 — `dspark_stride` = anchor spacing: CONFIRMED

`aux.rs:253` `let n = if t > k + 1 { (t - k - 1) / stride.max(1) } else { 0 };`
✓ exact line. `aux.rs:257-258` `arange(0..n).mul_scalar(stride)` → anchors at
`p = 0, 16, 32, …` ✓. `aux.rs:313` `mask = ones`, stride never enters the loss
✓.

Arithmetic: t=512, k=4, stride=16 → `n = 507/16 = 31` anchors ✓, and
`(n−1)·stride = 480 ≤ t−k−2 = 506`, so no out-of-range gather. The report's
"31 anchors, 124 supervised positions" is right.

Paper side, verbatim from 2607.05147v1 §3.3: *"we randomly sample multiple
anchor positions from each target sequence to form γ-token blocks as training
data."* ✓ The report's verdict — deliberate determinism substitution,
correct, documented nowhere — is right, and the doc gap at `schema.rs:109-111`
is real (no doc comment on any of the three fields).

### Claim 7 — D9: fact confirmed, verdict overturned

The facts are all exact:

- `aux.rs:45` `conf: AcceptRatePredictor::new(d_model, device)` ✓
- `aux.rs:311` `.prob(hidden_win.reshape([b * n, k, d]), None)` ✓
- `grep -rn with_markov crates/` → **empty** ✓
- Paper Eq 7 verbatim: `c_k = σ(w^⊤[h_k ; W₁[x_{k−1}]])` ✓

**Three reasons the verdict is wrong.**

**(a) The stated harm inverts the paper.** The report writes: *"Losing it makes
`c_k` a function of `h_k` alone, which is not measurable to a standard,
because `h_k` does not know which candidate was drafted."* The paper says the
opposite about *why the feature exists*: *"Because our confidence head relies on
the Markov feature of the previously sampled token, computing the next survival
probability a_{r,k+1} explicitly requires the instantiated candidate x_{r,k}. A
retrospective global search would thus inadvertently leak x_{r,k} into the
admission decision."* The feature is there to **enforce** non-anticipation by
making the dependence explicit. Removing it yields a head that depends on
*strictly less* information, which is strictly safer for the property the paper
cares about. And `h_k` is a function of the whole prefix, so `c* = 1 − ½‖p_d −
p_t‖₁` remains learnable from `h_k` alone — less well-conditioned, not
unmeasurable.

**(b) The report missed an oracle that makes this a config-default bug.** I
fetched `deepseek-ai/DeepSpec`. `config/dspark/dspark_qwen3_4b.py`, verbatim:

```python
exp_name = "dspark_block7_qwen3_4b"
block_size=7,  num_draft_layers=5,  num_anchors=512,  markov_rank=256,
## confidence head
confidence_head_alpha=1.0,
confidence_head_with_markov=True,
```

The reference ships the Markov-conditioned head **on by a boolean**. Our fix is
`AcceptRatePredictor::with_markov(d_model, rank, device)` — and `rank` is
already a parameter of `AuxHeads::new(d_model, vocab, rank, device)`
(`aux.rs:41`). The panic guards for the other direction already exist
(`burn-dspark/src/lib.rs:103-115`). This is a constructor argument, not an
architecture gap, and the report's severity language does not survive the
fetch.

**(c) The report's own gold test is better than its verdict.** §7 test 3 — assert
the confidence head's `Linear` input width is `d_model + markov_rank` — is
exactly right, and is what a config-default bug deserves.

### Claim 8 — E11: fact confirmed, severity overturned

Exact: `loop_block.rs:64` is `let lam = w_mem.clamp(0.0, lam_max);` ✓ and
`loop_block.rs:53` does read *"(lambda is a tuned CONSTANT, not a learned value
- same claim…)"* ✓. `w_mem` is learned: `loop_block.rs:377`
`sigmoid(self.controller.forward(ctrl_in).slice(…))`, and `controller` is a
trainable `Linear` (`loop_block.rs:85,193`). `engram_lam_max` default 0.5
(`schema.rs:77`). So kNN-LM's hard tuned λ and our learned σ(·) clamped at 0.5
are genuinely different objects. **That much is right.**

But the report calls it "the comment asserts a property the code does not have"
and rates it the load-bearing Engram bug. Reading the whole comment
(`loop_block.rs:36-57`), the operative claim is unambiguous and *does* hold:

> *"The point is that the guarantee is STRUCTURAL: whatever the controller
> learns, the memory's coefficient in this branch cannot exceed `lam_max`, so
> the backbone's share is never below `1 − lam_max`."*

And it is gated: `loop_block.rs:629` `memory_floor_caps_the_controller` drives
the controller to saturation and asserts the mix sits **at** the cap, then
asserts it follows `w_mem` below the cap. The kNN-LM sentence is a comparative
aside about the *source*, and the sloppiness is the word "same claim" — a tuned
constant and a clamped learned value are not the same claim. That is a
**one-word wording fix in a doc comment**, not a bug, and it is already the
best-documented structural guarantee in the Engram arm. Rating it alongside
E13 (a wrong percentage) is a severity error.

### Claim 9 — split: J5 confirmed and strengthened, J4 wrong and inverted

**J4 is overturned.** The phrase is verbatim, but it means the opposite of what
the report used it for.

2212.07525v2 abstract, verbatim: *"**We do not encode masked tokens**, use a
fast convolutional decoder and amortize the effort to build teacher
representations."*

The paper's body disambiguates it twice, and both times it is about **encoder
compute**, not about the loss:

> *"the output of the student encoder is then merged with fixed representations
> for the masked portions and fed to a decoder network"* (§2.2)
>
> *"Since we **only encode unmasked time-steps**, we use a simple strategy to
> assimilate the number of unmasked time-steps…"* (§2.2)

And data2vec 2.0 **does** mask, with an explicit hyperparameter. Appendix Table 8:

```
B (block width)   3    3    3
R (mask ratio)  0.80 0.75 0.75
A (mask adjust) 0.07 0.10 0.10
```

So the report's J4 cell — *"data2vec 2.0 has **no input masking at all**, and
**computes its loss over all positions**"* — is **flatly contradicted by Table
8**. data2vec 2.0's objective is a subset-sampled regression at R = 0.75 with
block masking. Our `mask_frac = 0.15`, `span = 8` is the same family and the
same order of magnitude, and the paper's own ablation (§4.3, Table 7) is
*"block masking… performs less well than **inverse block masking** (our
standard)"* — which is what `mask.rs`'s Bernoulli-start-dilated contiguous spans
are.

Worse, in our code the mask **never touches `input_ids`**:
`aux.rs:175-179` builds `mask2` from the host-drawn flags and passes it only to
`jepa_l1_loss`. The student sees every byte. So the real delta is the
**opposite** of the one reported: the paper hides the masked positions from the
student encoder, and we do not. The report found a real difference and named
the wrong end of it.

**J5 is confirmed, and the report's own ⚠️ caveat is the right one.** §4.2,
verbatim: *"Training targets are based on averaging the top **K** FFN blocks of
the teacher. Before averaging, activations are normalized using **instance
normalization** (Ulyanov et al., 2016)."* Table 8 gives the recipe as
`Target normalization: IN → AVG → LN` (vision) / `IN → AVG` (speech), on top of
`K (layers to average) 10 / 18 / 32`.

Our code: `losses.rs:11-20` is plain L1 with no normalisation, and
`grep -rn "normal\|Normal" vendor/burn-fused/crates/burn-jepa/src/ crates/dormouse-core/src/aux.rs`
returns **nothing**. Zero normalisation steps on the path, where the paper has
two or three. **Confirmed, and the largest JEPA delta is real.**

But **two thirds of the report's "source says" cell are data2vec 1.0, not 2.0**:

- *"smooth-L1 (Huber, β=2.0)"* — `"smooth"` appears **0 times** and `"Huber"`
  **0 times** in the paper body. Not there.
- *"summed with per-layer weights"* — data2vec 2.0 **averages the top K blocks**
  uniformly; per-layer weights are data2vec **1.0**'s design.

Both were flagged ⚠️ *"not read from the paper body"*, and both are wrong. This
is the repo's own AGENTS §1.4 failure mode reproduced inside the report: a
number or a formulation presented as a source's when it is recall.

---

## 2. The oracle question (claim 1) in depth

**Was the reference used?** Yes — on E1, E2, E3, E4, E5, E6, E17 and in the §8
test plan. Where it was used it was used well: E3's *"the demo does NOT
zero-init … our code follows the paper, the reference follows neither"* is
exactly the kind of three-way judgement the other two reports do not attempt.

**Where it was not used**, despite being available and relevant:

| delta | oracle available | report's verdict | would the oracle have changed it? |
|---|---|---|---|
| **E1** gate | demo `:372` | DELIBERATE, matches reference | no — but the report stops one step short of the real conclusion: **the official reference contradicts the official paper's Eq 4.** Eq 4 in 2601.07372v2 is verbatim `α_t = σ(RMSNorm(h_t)ᵀ RMSNorm(k_t)/√d)` — a *plain* sigmoid, and the paper attributes the RMSNorm to "gradient stability (Dehghani et al., 2023)". The demo inserts `abs().clamp_min(1e-6).sqrt()·sign`. So there is **no oracle on this point**, and "our code matches the reference implementation" is not the exoneration the row reads like. |
| **E5/E6/E7** | demo `:241,246,285` | BENIGN | no |
| **E17** | demo `:342,350` | BENIGN | no — but the row is internally contradictory, see §4.6 |
| **E11** | n/a | BUG | the oracle is kNN-LM; already fetched |
| **live path** | demo `:236-255` | **not in the table** | **yes — see §4.2** |
| **norm gains** | demo `:148-151,355-356` | **not in the table** | **yes — see §4.1** |

**The bigger miss: a second oracle, of exactly the same kind, exists for DSpark
and was never opened.** 2607.05147v1 names the repository. `burn-dspark`'s own
doc comments name three functions inside it — `DeepSpec loss.py`,
`DeepSpec _compute_accept_rate_3d`, `DeepSpec compute_dspark_loss`. The
repository is public and complete:

```
deepspec/modeling/dspark/loss.py              11 314 B   ← cited by name
deepspec/modeling/dspark/markov_head.py       10 869 B   ← Eq 5/6
deepspec/modeling/dspark/common.py             9 516 B
deepspec/eval/dspark/confidence_head.py       21 077 B   ← Eq 7 + STS (D13)
config/dspark/dspark_qwen3_4b.py              1 514 B   ← every hyperparameter
```

The Engram report's provenance table has a row for its oracle. The DSpark
report's has none, and its D2 verdict is decided against the paper while the
repo's config would have decided it outright. **`dspark.md` §0 even quotes the
paper's sentence naming the repo, and then does not go there.**

This overturns D2 outright:

> **D2 is OVERTURNED.** The report says: *"BUG — the objective's shape is
> wrong. We decoupled two numbers the paper ties together, and picked a decay
> 1.75× steeper. `w = [1, .78, .61, .47]` vs the paper's `w = [1, .87, .75, .65,
> .57, .49, .43]` at γ=7."*

`config/dspark/dspark_qwen3_4b.py`: `block_size=7`, **`loss_decay_gamma=4.0`**,
and the run is literally named `dspark_block7_qwen3_4b`. `loss.py:33-35`
applies `exp(-k/loss_decay_gamma)` over `block_size` positions.

So the reference implementation **decouples the two γ's exactly as we do**, and
**our `DSPARK_GAMMA = 4.0` is the reference's value verbatim**. The reference's
own decay weights at block 7 are `exp(−k/4) = [1, .779, .607, .472, .368,
.287, .223]` — *steeper* than the paper's, and ours is the same function
truncated to our block size. The "1.75× steeper" comparison is against a paper
setting the reference implementation does not use, and the two weight vectors
are over different lengths, so they are not comparable at all. D2's only
surviving element is the one the report treats as a parenthetical: `gamma` is a
hardcoded `const` (`aux.rs:27`), not a config field. That is a real nit with
the right answer for the wrong reason.

Also overturned by reading the paper body (30 seconds):

> **J3 is OVERTURNED.** The report says data2vec 2.0 *"uses β = 0.999 (base) /
> 0.9998 (large) / 0.9999 (huge), on a ramp over the first ~40% of training"*,
> flagged ⚠️ *"not read from the paper body; this is the standard published
> recipe."* Those are **data2vec 1.0's** values. Table 8 (vision):
> `τ₀ (EMA start) 0.9998 / 0.9998 / 0.9998`, `τₑ (EMA end) 0.99999 / 1.0 / 1.0`,
> `τₙ (EMA anneal steps) 100,000 / 500,000 / 300,000`. The report's own arithmetic
> survives (`0.999^2000 = 0.135`, so ~86 % of the way to the student by step
> 2000) but its premise is wrong twice over.

---

## 3. Delta spot-check — 13 checked, 5 overturned

| delta | report's verdict | mine | basis |
|---|---|---|---|
| E2 `√d` | BENIGN | **agree** | demo `:371`, Eq 4, Eq 6 — three sources agree |
| E3 short conv + zero-init | BENIGN | **agree** | Eq 5 verbatim; demo does *not* zero-init |
| E6 hasher | BENIGN | **agree** | `hasher.rs:142-148` vs demo `:285-293` |
| E7 `min_ngram` permits 1 | BENIGN, unasserted | **agree** | `hasher.rs:89` `assert!(min_ngram >= 1)` |
| E8 hasher unused | BUG (dead weight) | **agree, and it is worse** | `grep -rn "NgramHasher\|burn_engram::hasher" crates/` → empty |
| E9 `Sec. 2.4 eq. 6` citation | BENIGN, exact | **agree** | §2.4 is "Integration with Multi-branch Architecture"; Eq 6 is there; shared `W_V`, `M` distinct `W_K` verbatim |
| E11 | BUG | **overturned → wording** | §1 above |
| E15 4-gram orders | DELIBERATE | **agree** | §6.2: *"slightly suboptimal under a fixed 1.6B budget — likely because it dilutes capacity from the more frequent 2/3-gram patterns — though we do not rule [out]"* |
| E16 tokenizer compression | BENIGN | **agree** | §2.2: *"23% reduction in the effective vocabulary size for a 128k tokenizer"* |
| D1 | BUG open | **agree** | §1 above |
| D2 | BUG | **overturned** | §2 above |
| D9 | BUG | **overturned** | §2 above |
| J8 KoLeo on `[b,d]` | BUG, "statistically vacuous" | **partly agree, mislabelled** | `aux.rs:178` `student_latent.mean_dim(1).reshape([b,d])` — exact. At b=10 it is 10 points on a 768-sphere at weight 0.005. That is an **uncalibrated term**, not a defect, and the report's own §6 test 3 says the right thing to do is "make the test say so in its own failure message rather than letting it ride". A row whose remedy is "let it ride and document it" should not be filed as a BUG next to E13. |
| J4 | BUG misattribution | **overturned, inverted** | §1 above |
| J5 | BUG | **confirmed** | §1 above |

---

## 4. What the report missed

### 4.1 Every RMSNorm in the Engram path lost its learnable gain — the largest port delta in the crate, absent from the table

The demo's norms are `torch.nn.RMSNorm` with `elementwise_affine=True` by
default, i.e. a **learnable** `Parameter(torch.ones(dim))`:

```python
self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)])   # :148-151
self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(hc_mult)]) # :355
self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(hc_mult)]) # :356
```

Ours has none. `burn-engram/src/lib.rs:214-236` `compute_gate` is a bare
`key / sqrt(mean(key²) + 1e-5)` with no parameter; `lib.rs:199-205` is
`out / sqrt(out.powf(2).mean_dim(3) + 1e-5)` — also bare. At `hc_mult = 4,
hidden = 1024` that is **2 × 4 × 1024 = 8 192 learnable parameters removed from
the gate, plus `hc_mult × hidden = 4 096` from the short conv**, on the module
whose entire mechanism is a learned gate. The report's E1/E2 say only that the
gate "matches the reference implementation" and that the norm layout is
"BENIGN"; both are true of the *arithmetic* and false of the *parameterisation*.

### 4.2 The live addressing path is a power-of-two mask, not a prime table — and no row in the table notices

Paper Eq 1 requires tables of **prime** size `M_{n,k}`. The demo implements it:
`find_next_prime(vocab_size − 1, seen_primes)` (`:236-255`), `mix % mod` (`:293`).

What actually runs:

```
crates/dormouse-core/src/loop_block.rs:32-33   let pow2 = rows.next_power_of_two();
                                               (vec![pow2; n_tables], (pow2 - 1) as i32)
crates/dormouse-core/src/loop_block.rs:432     .forward(hashed.clone()
                                                   .bitwise_and_scalar(self.engram_slot_mask), eg_in)
crates/dormouse-data/src/lib.rs:70             out.push(((fnv(&bytes[s..e]) as u32) & 0x7fff_ffff) as i64);
```

So the training path is **FNV-1a → truncate to 31 bits → AND with `2^k − 1`**.
Prime sizing is gone, and only the low `log2(rows)` bits of the FNV hash are
ever used — FNV's avalanche is weakest in exactly those bits, which is a
different collision profile from the one the paper measured (and the one
2601.16531's whole hot/cold collision study is about).

This is E8 followed one step further. E8 correctly establishes that
`hasher.rs` is unwired, then concludes *"dead weight, not a defect"*. But
`hasher.rs` is the **only** thing in the tree that honours Eq 1's prime sizing,
`rem_euclid`, and the per-layer multiplier stream — and nothing else does any of
the three. The report's §9 Q3 even notices the power-of-two question and leaves
it *"unverified either way"*; it is settled by two `grep`s. And the report's own
§8 test 3(b) — *"assert the prime search start is `vocab_size - 1`, not
`vocab_size`"* — is already satisfied by `hasher.rs:94`
(`vocab_size.saturating_sub(1)`), i.e. half of a recommended test is already
true and the other half is about a file nothing calls.

### 4.3 Two more oracle divergences the report could have had for free

- The demo ships `layer_ids = [1, 15]` (`:45`) while the paper's Table 5 ships
  `Engram Layer [2,15]`. The official demo is off by one on the first injection
  layer relative to the official paper. That is another reason the demo is a
  weak oracle on *configuration* questions, and the report says nothing about
  it.
- The demo's `engram_hidden_size = (max_ngram−1)·n_embed_per_ngram = 1024`
  (`:350`) and `d_mem = 1280` in the paper's Table 5 are different numbers for
  what Eq 2 calls the same quantity.

### 4.4 The report's §8 headline gold-vector constant is wrong, and its test would fail

`engram.md` §8 test 1, described as *"the one that matters"*:

```rust
assert!((gate - 0.804_129_4).abs() < 1e-6);   // transcribed from the demo run
```

Replicating `compute_gate` exactly for the fixture the report describes
(`key = query = 2·ones([1,1,4])`, `d = 4`), in the same order the code performs
it:

```
kr = sqrt(mean(key²) + 1e-5)      = 2.0000024999984376
dot = sum(k·q)/sqrt(4)             = 1.9999950000125
g   = sign(dot)·sqrt(|dot| + 1e-6) = 1.414212148163245
sigmoid(g)                         = 0.8044294600197353
```

The proposed constant is **0.804 129 4**. The difference is **3.0 × 10⁻⁴**,
against a proposed tolerance of **1 × 10⁻⁶**. **The test as specified fails on
the current, correct implementation.** It also cannot have been "transcribed
from the demo run": the demo needs torch, `sympy`, and a
`deepseek-ai/DeepSeek-V3` tokenizer download, and no run is recorded.

Worse, 0.804 is what the **existing** `gate_matches_reference_formula`
(`lib.rs:300-330`) already pins via `expected = 1/(1+exp(-sqrt(2.000001)))`. So
the report's flagship recommendation replaces a restatement of the formula with
a mistyped restatement of the same value. The report's own diagnosis — *"it
asserts a value it computed by restating the implementation … it does not pin
against the reference"* — is correct; its replacement does not pin against the
reference either. Settling this requires actually running the demo under torch
with the tokenizer available, or admitting there is no oracle for the gate.

### 4.5 The report's §8 test 2 asks to pin a divergence that does not exist

`engram.md` §8 test 2: *"Pin the DIVERGENCE explicitly: the demo has one norm
per BRANCH; ours has one over the concatenated channels."*

Ours is per-branch. `lib.rs:199-205` computes `out.powf_scalar(2.0).mean_dim(3)`
on a `[B, L, HC, D]` tensor — axis 3 is `D`, so the reduction is *within* each
`(b, l, hc)` row. That is exactly `RMSNorm(hidden_size)` applied per branch,
which is what the demo does (`:165-170`, norm each `x[:, :, i, :]`, then
`torch.cat`). **There is no divergence.** The report's other gloss is also
wrong: it calls the norm `RMSNorm_eps=1e-5(W_K e)`, glossing over the fact that
`nn.RMSNorm` is affine and ours is not (§4.1). Written as specified, this test
would either fail or have to be inverted into an equality assertion — and the
report has told the reader which way to expect it to go.

### 4.6 E17's cell is internally contradictory

> *"Engram-27B: `d_mem = 1280`, `K = 8`, width 1024"*

The report's own §2 transcription of Eq 2 — verbatim from the paper — is
`e_t ≜ ‖_{n=2}^{N} ‖_{k=1}^{K} e_{t,n,k}`, **`e_t ∈ R^{d_mem}`**. The width of
`e_t` *is* `d_mem`, so Engram-27B's is 1280. 1024 is the **demo's**
`(3−1)·512`. The row presents a paper number and a demo number in one cell as
though both were Engram-27B's, in the row whose job is to certify that the
width formula `(N−1)·K·embed_dim` matches. The formula does match; the numbers
in the cell do not go with each other.

### 4.7 Smaller things

- `schema.rs:159` cites "Sec. 2.4 eq. 6" — verified exact (§2.4 *Integration
  with Multi-branch Architecture*, Eq 6 there). One of the few clean citations;
  the report is right and I confirm it.
- `loop_block.rs:51-57`'s FwPKM eq 12 and kNN-LM eq 3 quotes — I did not
  re-verify against 2601.00671v2 / 1911.00172v2 in this pass; the arXiv records
  resolve and the titles match the report's ("*Fast-weight Product Key Memory*",
  Tianyu Zhao). **UNVERIFIED**: `curl -sSL https://arxiv.org/html/2601.00671v2 | sed 's/<[^>]*>/ /g' | grep -o 'o_t = g_t[^.]*'`
- The report's `engram.md` §1 asserts *"No paper failed to load. Every equation
  quoted below is transcribed from the source above, not recalled."* That is
  true of `engram.md` and `dspark.md`. It is **not** true of `jepa.md`, whose
  §0 states *"I fetched only the **abstracts** of 2212.07525, 2511.08544 and
  2304.07193"*. The set-level impression of a fully-sourced pass is misleading,
  and both of `jepa.md`'s recalled-from-memory items (J3's momentum, J5's
  smooth-L1 + per-layer weights) are wrong. Fetching the two HTML bodies takes
  about thirty seconds and fixes all three.

---

## 5. My own top-3 findings

### 1. `jepa.md`'s central JEPA verdict is inverted (J4), and its largest JEPA
finding rests on two data2vec **1.0** facts presented as 2.0's (J5, J3)

The report's headline JEPA claim is that the crate "is marketing data2vec 2.0's
name over a loss data2vec 2.0 does not have." data2vec 2.0's abstract says "We
do not encode masked tokens" and the report read that as *the loss is
unmasked*. The paper's own §2.2 says the phrase means the **encoder** skips the
masked positions, and its Table 8 ships `R (mask ratio) 0.75` with block masking
and an explicit ablation for *"inverse block masking (our standard)"*. data2vec
2.0's objective is a masked, subset-sampled, block-structured regression — the
same family as `mask.rs`.

The delta that *is* real runs the other way and is one the report never names:
`aux.rs:175-179` builds the mask from host-drawn flags and passes it **only** to
`jepa_l1_loss`. `input_ids` is never masked. The paper hides the loss positions
from the student encoder; we show them to it. That makes the task strictly
easier than the cited parent and is a different defect from the one reported.

Underneath both, `jepa.md` flagged J3 and J5 with ⚠️ *"not read from the paper
body"*, and both are wrong in the same way: **data2vec 1.0's recipe, labelled
2.0's.** J3's `0.999 / 0.9998 / 0.9999` is 1.0's; 2.0's Table 8 is
`τ₀ = 0.9998 → τₑ = 0.99999 / 1.0` over 100k–500k steps. J5's *"smooth-L1
(Huber β=2.0)"* and *"summed with per-layer weights"* are both 1.0's; the string
"smooth" occurs zero times in the 2.0 body and 2.0 **averages the top-K FFN
blocks** uniformly. J5's *actual* finding — no instance normalisation anywhere,
against the paper's `IN → AVG → LN` — is correct, and is the largest real JEPA
delta. It survived the report by accident, carried on a wrong premise.

### 2. The DSpark oracle was never opened, and it inverts D2 and deflates D9

2607.05147v1 names `github.com/deepseek-ai/DeepSpec`.
`burn-dspark/src/lib.rs:135,144,158` cites three functions *in it by name*.
`dspark.md` §0 quotes the sentence naming the repo and does not go.

`config/dspark/dspark_qwen3_4b.py`:

```python
exp_name = "dspark_block7_qwen3_4b"
block_size = 7
loss_decay_gamma = 4.0        # ← the paper's γ is 7
confidence_head_with_markov = True
```

- **D2 dies.** The reference decouples block size from the decay denominator
  *exactly as we do*, and our `DSPARK_GAMMA = 4.0` (`aux.rs:27`) is the
  reference's value. The report's "we decoupled two numbers the paper ties
  together / a decay 1.75× steeper" describes the reference implementation, not
  our code. Only the surviving nit — that `gamma` is a `const` and not a config
  field — is real.
- **D9 deflates.** The reference exposes the paper's Eq 7 head as a boolean that
  defaults on. Our fix is `AcceptRatePredictor::with_markov(d_model, rank, device)`
  with `rank` already threaded into `AuxHeads::new`; the reverse-direction
  panics already exist at `burn-dspark/src/lib.rs:103-115`. It is a
  constructor argument, not a missing mechanism.
- **D9's stated harm is backwards** regardless: the paper's non-anticipation
  argument runs *through* the Markov feature, so removing it is strictly safer
  for the property the paper cares about, and `c*` remains a function of `h_k`.
- **D13 partially survives**: `deepspec/eval/dspark/confidence_head.py` (21 kB)
  is the real STS implementation and is the right thing to compare
  `sts_calibrate` against. It is still uncalled here, which is correct for a
  trainer.

### 3. `engram.md` missed the two deltas that the oracle it already fetched would
have exposed, and shipped a gold-vector test that fails

The report is the best of the three — it found the oracle, read it line for
line, and made three-way judgements the others do not attempt. But it spent the
oracle on the seven rows where it was already right and left the two rows where
it was wrong unasked:

- **Learnable RMSNorm gains.** Demo `:148-151,355-356` are affine
  `nn.RMSNorm`; `lib.rs:199-236` are bare normalisations with no parameter.
  12 288 learnable parameters absent from the gate module, unmentioned.
- **Prime sizing.** Eq 1 requires prime `M_{n,k}`; the demo implements
  `find_next_prime`; the live path is `loop_block.rs:32-33` + `:432`
  `bitwise_and_scalar(engram_slot_mask)` over `dormouse-data/src/lib.rs:70`'s
  31-bit FNV. E8 notices the hasher is unwired and calls it dead weight; the
  follow-through — that nothing else does primes either, so the live module is
  power-of-two addressed against a paper whose collision behaviour is the entire
  subject of the 2601.16531 study the schema cites — is absent, and the report
  leaves the power-of-two question explicitly "unverified" when two greps settle
  it.

And its §8 remedy is worse than the disease it diagnoses: the constant
`0.804_129_4` is **3 × 10⁻⁴** away from the true `0.8044294600` under a
`1e-6` tolerance, so the "one that matters" test fails on correct code and
cannot have come from a run; and test 2 asks to pin a per-branch-vs-concatenated
norm divergence that does not exist in this codebase. The diagnosis —
*"it restates the implementation"* — is right. The prescription restates it
with a typo and calls it a transcription.

---

## 6. Everything the report got right, so it is on the record

Not much is missing from this list, and it is worth stating plainly because the
criticisms above are all about the same failure mode — recall where a citation
was owed.

- The Engram oracle exists, was fetched, and every fragment quoted from it is
  byte-exact (E1–E6, §8).
- E1's flag that `lib.rs:233`'s *"the plain sigmoid(s) variant diverges for
  |s| > ~1"* is an unsourced empirical assertion the paper contradicts —
  correct, and Eq 4 is verbatim plain σ.
- E2, E3, E4, E5, E6, E7, E9, E15, E16 are all correctly judged against
  sources I independently verified.
- E12/E13: 196B/552B is unsourced and 24 % should be 32 % — both true.
- E13's *"Bounded: the ratio is lower than the paper's 21.3 %-of-total would
  suggest, not higher, so the conclusion survives"* — a correct severity
  judgement, and the reason it is right is that the report actually did the
  ratio in both directions.
- Every number in §0 of `engram.md` against 2601.16531 is correct, including
  the Table-6-vs-Table-7 correction and the 185 M-vs-125 M catch, which is the
  best single finding in the three documents and which the repo's own schema
  needs.
- The JEPA teacher fix is correctly diagnosed as fixed *and* correctly
  credited with a gate that cannot be satisfied by a coincidence — I checked
  that the gate would actually fail on the regression and it does.
- The DSpark off-by-one trace is exact to the line, and the observation that
  the window arithmetic is correct and only the input tensor is wrong is a
  better diagnosis than AGENTS.md §3.3's.
- D3–D8 and D11, D14, D15, D16 are correct, including the point that Eq 10 is
  L1-on-probabilities not L1-on-logits, which is the easy thing to get wrong.
- The dead-code claims are all verified: `sts_calibrate`, `lejepa_loss`,
  `with_markov`, `sample_block_tokens` have zero callers under `crates/`.
- `dspark.md` §5's closing judgement — three stacked changes (D1, D2, D9) mean
  any A/B now moves three things at once — is the right operational conclusion
  even though D2 turns out to be a non-issue, which leaves two.

---

## 7. Settling commands

```bash
# claim 1 — the oracle
curl -sSL https://raw.githubusercontent.com/deepseek-ai/Engram/main/engram_demo_v1.py -o /tmp/engram.py
sed -n '371,373p;148,151p;355,356p;45p' /tmp/engram.py

# the second oracle, which dspark.md never opened
curl -sSL https://raw.githubusercontent.com/deepseek-ai/DeepSpec/main/config/dspark/dspark_qwen3_4b.py
curl -sSL https://raw.githubusercontent.com/deepseek-ai/DeepSpec/main/deepspec/modeling/dspark/loss.py | sed -n '25,40p'

# claim 3 — 196B/552B absent from 2601.07372v2
curl -sSL https://arxiv.org/html/2601.07372v2 \
  | sed 's/<[^>]*>/ /g' | grep -oE '.{60}(552|196|39\.5|18\.5).{60}'

# claim 9 — the data2vec 2.0 mask ratio that overturns J4
curl -sSL https://arxiv.org/html/2212.07525v2 | sed 's/<[^>]*>/ /g' \
  | grep -oE '.{80}(mask ratio|do not encode masked|instance normalization).{120}'

# §4.2 — the live addressing path
grep -n 'bitwise_and_scalar\|next_power_of_two' crates/dormouse-core/src/loop_block.rs
grep -n 'fnv' crates/dormouse-data/src/lib.rs | head -3
grep -rn 'NgramHasher\|burn_engram::hasher' crates/ --include=*.rs   # empty = dead

# §4.1 — no learnable gain in the gate
sed -n '199,236p' vendor/burn-fused/crates/burn-engram/src/lib.rs

# §4.4 — the report's proposed constant
python3 -c "import math;k=2.0000024999984376;d=1.9999950000125;\
print(1/(1+math.exp(-math.copysign(math.sqrt(abs(d)+1e-6),d))), 'vs 0.8041294')"

# arithmetic in schema.rs
python3 -c "print(2.4e6/7.5e6, 32768*3*32/7.5e6, 32768*3*32/9195854)"
```

**Confidence:** high on claims 1–6 and 8 and on every `file:line` above (each
one was opened). High on the J4/J5/J3 overturns (paper body read in full). High
on the D2/D9 overturns (reference config read). High on §4.4's arithmetic
(closed-form replication of `compute_gate`).

**Not settled here:** whether `engram_demo_v1.py` is runnable end-to-end without
a `deepseek-ai/DeepSeek-V3` tokenizer download, which is what §4.4's premise
depends on. Settling command:
`pip install torch sympy transformers && python3 /tmp/engram.py` — a download
and a network fetch, out of scope for this pass.
