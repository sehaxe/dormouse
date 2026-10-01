# Review B — the official Engram oracle, and what the reports do with it

**Reviewer angle:** the consequence of the reports' load-bearing structural claim
(`deepseek-ai/Engram/engram_demo_v1.py` exists and is authoritative). Not citations,
not line numbers — what the project *gains* from an oracle it has not consulted, and
what a gate built on it would have to say.

**Files reviewed:** `research/papers/{engram,jepa,dspark}.md`, tree `eeb3b73`, 2026-09-29.

**What I did:** re-fetched the reference myself, read it in full, read the call sites
and the tests, and did the arithmetic. **I ran no `cargo` command, no test, no GPU** —
every claim below is static reading plus arithmetic, and I say so where that limits me.
I did not read the other reviewer's file.

**One correction to the brief up front.** It says "the report's 9 BUGs". `engram.md`
carries **three** verdicts labelled BUG (E8, E11, E13) and three labelled UNVERIFIABLE
(E1's inline claim, E12, E14). I apply the owner's rule to all six, because the
UNVERIFIABLE three are the ones the rule bites hardest on. The count discrepancy is
itself worth a line: the report's own severity vocabulary is not stable enough to be
counted by a reader, and §1.4 exists precisely because severity words here have to be
traceable to a named file.

---

## 1. The oracle: what is the project actually gaining from it today?

**Nothing numerical. One comment, and one comment that is wrong.**

### 1.1 It is not wired to anything

```
$ grep -rn "engram_demo_v1" --include=*.rs .
vendor/burn-fused/crates/burn-engram/src/lib.rs:231:   // Official reference (deepseek-ai/Engram, engram_demo_v1.py Engram.forward):
vendor/burn-fused/crates/burn-engram/src/hasher.rs:1:    //! CPU n-gram hashing for Engram (deepseek-ai/Engram, arxiv 2601.07372).
vendor/burn-fused/crates/burn-engram/src/hasher.rs:4:    //! `engram_demo_v1.py`: for every n-gram order in `min_ngram..=max_ngram`
vendor/burn-fused/crates/burn-engram/src/hasher.rs:236:  /// Deliberately absent: a bit-exact comparison with `engram_demo_v1.py`.
```

Five hits, all comments. Zero tests. There is no `crates/dormouse-core/tests/engram_gold.rs`
(the file `engram.md` §8 recommends), no `dspark_gold.rs`, no `jepa_gold.rs`; the three
vendored crates have no `tests/` directory at all, only inline `#[cfg(test)] mod tests`.

And the arm is not marginal. `use_engram = true` in **seven of eight** presets
(`base mor nano one_b p150 small swift50`; only `nano-fused` is false), and the shipped
table is **3 145 728 parameters — 34.2 % of the `small` model** (`3 × 32_768 × 32`;
`engram_tables` rounds 25 000 up, `loop_block.rs:24-34`, asserted at `loop_block.rs:735`;
`README.md:521` says "3.15M of the 9.2M params"). So: the largest single mechanism in
the flagship preset, on in every production recipe, with an official reference
implementation downloaded by two research passes and compared to **zero** times.

### 1.2 The one test that claims to be the gate is a restatement

`burn-engram/src/lib.rs:300-330`, `gate_matches_reference_formula`:

```rust
let expected = 1.0 / (1.0 + (-(2.0f32 + 1e-6).sqrt()).exp());
assert!((v[0] - expected).abs() < 1e-3, ...);
```

`expected` is the implementation's own formula, retyped. It moves with the bug — the
report says this and it is right. But the report undersells what the test *is*: it is a
**two-formula discriminator**, and a decent one. I checked the discrimination margin:

```
s=1: signed-sqrt 0.731059  plain 0.731059  diff 0.000000
s=2: signed-sqrt 0.804430  plain 0.880797  diff 0.076367   <- the test's point
s=3.36 (max):                     diff 0.104308
s=32:                             diff 0.003481
s=64:                             diff 0.000335
```

The test sits at 73 % of the maximum available separation. It would catch a regression
to the plain sigmoid. What it cannot catch is a transcription error *inside* the
compressed form — and that is exactly where the divergences from the reference are.

### 1.3 Three ways our gate is not the reference's gate

I fetched the reference (`HTTP 200`, 15 017 bytes, 422 lines,
`sha256 9d082070654df217e21bbca9926a4267bdf2cce7777aa6739747c24de30d2044`) and read
`Engram.__init__` / `Engram.forward`. The report never looks at the reference's *module
inventory*, only at its gate line. Three divergences, in descending order of importance:

| # | reference | ours | where | consequence |
|---|---|---|---|---|
| **A** | `self.norm1/norm2 = ModuleList([nn.RMSNorm(hidden_size) …])` (`engram_demo_v1.py:355-356`) — `elementwise_affine=True` by default, so each has a **learnable per-channel gain**, applied to key and query before the dot (`:368, :370`) | none. `burn-engram` never constructs an `RMSNorm`; `lib.rs:215-228` hand-rolls `x / sqrt(mean(x²)+1e-5)` | `lib.rs:214-236` | the gate is a **strictly smaller function class**. `σ(sign·√(⟨γ_k⊙k̂, γ_h⊙ĥ⟩/√d))` ≠ `σ(sign·√(⟨k̂, ĥ⟩/√d))` for any learned γ. Not absorbable: a diagonal reweighting is not reachable by an inner product of unit-RMS vectors. |
| **B** | `value_proj = nn.Linear(mem, hidden)` and `key_projs = [nn.Linear(mem, hidden)]` — **bias=True** by default (`:351-354`) | `.with_bias(false)` on both | `lib.rs:136-148` | `value = gate ⊙ (W_V e + b_V)` vs `gate ⊙ W_V e`. `b_V` is constant across positions but `gate` is not, so it contributes a position-varying vector. Not absorbable. |
| **C** | `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()` (`:372`) — a **clamp** | `dot.abs().add_scalar(1e-6).sqrt()` — an **add** | `lib.rs:234` | numerically ~0 in the normal regime, but the code comment at `lib.rs:231-232` transcribes the file it cites as `sqrt(|s| + 1e-6)`, i.e. it restates *our* code and cites the reference for it. That is the §1.4 failure in its smallest possible form. |

**A and B are real structural deviations that nobody in the tree has named** — not the
report, not `docs/`, not a code comment. Both are defensible (the paper's Eq 3/4 has
neither a bias nor a gain, and following the paper is the right instinct at 9.2 M
params). Neither is *declared*, which is the whole of the owner's rule. `burn-engram`'s
parameter inventory is missing 2·hc·d + hc·d + d parameters per Engram layer relative
to the reference — 3 072 at d=768, negligible in count, load-bearing in function class.

### 1.4 The report's own E1 verdict is false as written

> E1 … "Fetched `engram_demo_v1.py`: it is literally `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`.
> **Our code matches the *reference implementation***"

It does not, three ways (A, B, C). Scoped to the single gate line it is *nearly* right,
and even there the code comment that names the file misquotes it. The report had the
file open and reported a fidelity claim it had not checked.

### 1.5 The one inline claim the oracle settles outright, and nobody checked

`lib.rs:233`: *"the plain sigmoid(s) variant diverges for |s| > ~1"*.

The report's verdict is "UNVERIFIABLE — unsourced". It is worse than unsourced; it is
**false as stated**, and the reference's own formula settles it (numbers above):
at |s| = 1 the two are *identical* (`√1 = 1`), and past |s| ≈ 30 they **reconverge**
(diff < 0.004 at s = 32, < 0.0004 at s = 64, both → 1.0 as s → ∞). It is not a
divergence, it is a **bounded peak of width ~1 decade and height 0.104**.

What the compression actually does is pull the pre-activation toward 1 in log-space: for
|s| > 1 it is *less* saturated than plain σ (more gradient survives); for |s| < 1 it is
*more* open. So it is a **compressor toward the middle, not a divergence preventer** —
which is the right instinct, because the failure it defends against is the gate
saturating to 0 or 1, i.e. exactly the FwPKM "gates cluster near zero" finding the repo
already cites at `loop_block.rs:51-57`. One sentence fixes a claim that is currently
wrong in the file that names the authoritative source.

### 1.6 What the project *does* have: a structural oracle, and only that

`probe.rs` is genuinely good. `ENGRAM` (branch entered) and `ENGRAM_KEYS` (a row was
actually read) are incremented at `loop_block.rs:402` and `:443`, asserted in
`preset_exec.rs:188,192,376`, `jepa_teacher_seam.rs:208-224` (integer, exactly
`2 × max_iter`), and carried onto the eval line at `train/src/lib.rs:1412-1413` as
`engram=<rows>/<arms>`. That is the §1.1 "a fused arm must be able to show it ran"
obligation discharged properly.

The shape of the gap is therefore exactly one sentence: **the tree can prove the memory
branch ran and read a row, and cannot prove that the row, the gate applied to it, or the
mix it produced is numerically the reference's.** A counter answers "did it run"; nothing
answers "did it compute the right thing". That is precisely the class the `burn-dspark`
header (`lib.rs:6-9`) and the `burn-engram` hasher header (`hasher.rs:236`) already
admit to in prose and then do nothing about.

### 1.7 The oracle is *partial*, and that is a finding, not a caveat

`engram_demo_v1.py:38-58` is the entire config:

```python
engram_vocab_size = [129280*5, 129280*5];  max_ngram_size = 3
n_embed_per_ngram = 512; n_head_per_ngram = 8
layer_ids = [1, 15];  kernel_size = 4;  pad_id = 2;  seed = 0
backbone = hidden_size 1024, hc_mult 4, vocab_size 129280, num_layers 30
```

No parameter count, no model size, no 552B/196B anywhere, and `layer_ids = [1, 15]`
where the paper's §6.2 reads "layers 2 and 6" per the report's transcription. It is a
**shape demo**. So:

- The oracle **can** arbitrate: the gate, the norms, the biases, the conv, the
  embedding offsets, the multiplier/prime stream, the width formula.
- The oracle **cannot** arbitrate: E12's "196B Engram / 552B backbone" and E14's
  allocation-ratio law. Those need the paper's tables, and the report already checked
  those (correctly).

Any move that says "the reference validates the 24 %/32 % ratio" is a category error.
This distinction is the practical output of the oracle question and neither report
draws it.

### 1.8 Two of the three recommended gold tests pin dead code

```
$ grep -rn with_short_conv --include=*.rs .
burn-engram/src/lib.rs:155:  pub fn with_short_conv(...)
burn-engram/src/lib.rs:283:  ... .with_short_conv(4, 3, &dev())        <- its own shape test

$ grep -rn NgramHasher --include=*.rs .        # 6 hits, all in hasher.rs's own tests
```

`EngramModule::new(&tables, cfg.engram_dim, d, 1, device)` at `loop_block.rs:204` never
calls `with_short_conv`, so `depthwise_conv_1d`, paper Eq 5, the zero-init, the
per-branch-vs-per-concatenation norm argument — **none of it is in the training path**.
`NgramHasher` has zero callers outside `hasher.rs`. The report finds this (E8) and files
it as "dead weight, not a defect", then recommends `short_conv_matches_official_reference`
and `ngram_hash_matches_official_reference` as gold tests. Both would certify code that
no run executes. Worse, the "per-branch norm vs per-concatenation norm" divergence the
second test is meant to *name* **is not a divergence at all**: `nn.RMSNorm(D)` normalises
over the last axis, and so does our `mean_dim(3)`, so at `hc_mult = 1` the two are the
same function on the same numbers.

And the addressing itself can never be bit-for-bit: the training path's keys are
FNV-1a low-31-bits masked by `engram_slot_mask` (`dormouse-data/src/lib.rs:65-74`,
`loop_block.rs:432`) against the reference's odd-multiplier XOR-mod-prime
(`engram_demo_v1.py:285-293`). So **no fidelity claim about this arm can extend past the
gate**, and the report never scopes its claims that way.

---

## 2. The gate I would specify

Three tests, not one. The organising principle: **split the claim by reproducibility.**
Bit-exact where the arithmetic is exactly reproducible (integers, shapes, widths);
relative tolerance where it is not (floats across libm/torch/burn). A single
`assert_eq!(value.bits(), 0x....)` — which is what `engram.md` §8 proposes — is
unachievable across three different float implementations and would be a test that only
passes on the machine that generated it.

**Name the artifact immutably.** The report cites a `main`-branch raw URL. `main` moves.
A §1.4-compliant claim names the file *and* its digest. Use:

```
https://raw.githubusercontent.com/deepseek-ai/Engram/main/engram_demo_v1.py
sha256 9d082070654df217e21bbca9926a4267bdf2cce7777aa6739747c24de30d2044   (422 lines, 15017 bytes, fetched 2026-09-29)
```

**Where it goes:** `vendor/burn-fused/crates/burn-engram/tests/gate_oracle.rs` (an
integration test, not a unit test — the point is that it does not have `compute_gate`
in scope to restate it), plus the fixture as a `const` array in the same file so the
test and the data cannot drift apart. CPU, `Device::ndarray()`, no GPU.

### Test 1 — `gate_matches_reference_on_the_regime_where_they_differ`

**Compares:** `burn_engram::compute_gate(key, query, d)` against an **f64 host
re-implementation of `engram_demo_v1.py:371-373` transcribed verbatim** — `clamp_min`,
not `add` — fed *the same literal f32 fixtures* for `key` and `query`.

**Shapes:** `[1, 1, d]` and `[1, 512, d]`, `d ∈ {512, 768}` (the two shipped
`d_model`s: `nano` 512, `small` 768).

**Input distribution — this is the load-bearing part, and it is the part the report's
proposal gets wrong.** The fixture is a *sweep over |s|* chosen from the measured
divergence table, not a random draw:

```
|s| ∈ {0, 1e-9, 1e-6, 1e-4, 0.01, 0.5, 1.0, 2.0, 3.36, 5.0, 20.0, 100.0}  and their negatives
```

Two entries earn their place for reasons a random draw would miss:

- **|s| = 1e-6 is the ONLY point where divergence C exceeds 1e-5.** Measured:
  `add` gives 0.50035355, `clamp_min` gives 0.50025000, **d = 1.04e-4**. At the |s| = 2
  the existing test uses, d = 5.6e-8, which is *below one f32 ulp at 0.8* (5.96e-8) and
  rounds to the identical f32 — the divergence is **provably invisible** there. A gold
  test built only on well-scaled inputs is blind to the exact defect the oracle was
  fetched to find.
- **|s| = 3.36 is the maximum** of |σ(√s) − σ(s)| = 0.1043, and |s| = 1 is where the two
  are *identical*. So the discriminating assertions are
  `assert!((gate(s=2) − plain_σ(2)).abs() > 0.07)` and
  `assert!((gate(s=1) − plain_σ(1)).abs() < 1e-6)` — the second pins that s = 1 is the
  fixed point, which is the actual content of the compression claim.
- **|s| = 0 exactly** pins `sign(0) == 0` on **every backend** (0.5 exactly, 1e-6
  tolerance). A backend whose `sign(0)` returns +1 moves the gate at a zero dot product
  by 2.5e-4. It is the only *backend-dependent* input to the gate and nothing in the
  tree touches it.

**Tolerance, and why that number:** the reference is fp32 torch on CPU; ours is fp32
ndarray. The gate chain is ≈ 9 float ops (rms, div, mul, sum, div, abs, add-or-clamp,
sqrt, mul, sigmoid). Worst-case relative drift over n rounding steps is n·2⁻²⁴; at
n = 9 that is **5.4e-7**. So **`rel ≤ 1e-6` (≈ 8 ulp) is the justified gate**, with the
add-vs-clamp difference (1.04e-4 at |s| = 1e-6) sitting 100× above it. I would state
the claim as *"agrees with `engram_demo_v1.py:371-373` to 1e-6 relative"*, **not**
"bit-for-bit", and say so in the test name, because bit-for-bit is not available here
and claiming it would be the exact lie §1.4 exists to prevent.

**What makes it fail:** a change to the `/√d` scale; to the RMS eps; to `mean_dim`
vs `sum/d`; a flip of the sign; a `clamp_min`→`add` regression (caught only at
|s| ≤ 1e-6); a backend whose `sign(0) ≠ 0`; a `powf_scalar`/`sqrt` change on cubecl
(this is the one that would be backend-specific, and it is exactly why the test wants a
second copy run on `--features cuda`).

**The report's proposed constant is wrong, and that is the meta-point.**
`engram.md` §8 proposes `assert!((gate - 0.804_129_4).abs() < 1e-6)`. The gate at
s = 2 — the only s the tree has a fixture for — is **0.80442974**. The proposed literal
is off by **3.0e-4, i.e. 300× its own stated tolerance**, so the recommended test fails
on the tree it is meant to certify. A gold-vector test whose constant ships with **no
fixture data** is not a gold-vector test; it is a number in a document. The fixture
vectors are the artifact. Ship them or ship nothing.

### Test 2 — `our_gate_path_declares_its_three_deviations_from_the_reference`

A **width/shape** test, zero tolerance, no floats — and the most valuable of the three,
because it is the only one that stops a *silent* fidelity drift:

```
assert!(value_proj.weight.dims() == [mem_dim, d]);   // and NO bias term exists in the record
assert!(key_projs[0].weight.dims() == [mem_dim, d]);
assert_eq!(param_count(EngramModule), expected_without_affine_terms);
```

with the test's own doc comment carrying the three sentences: the reference has
`nn.RMSNorm` gains (`:355-356`), we follow paper Eq 4 and have none; the reference's
`nn.Linear`s have biases (`:351-354`), we follow paper Eq 3 and have none; the
reference zero-inits nothing and we follow §4.1. **If a future edit adds the affine
terms, this test goes red and forces the author to decide whether the doc claim of
reference fidelity is still true.** It is a change-detector, not a value check, and it is
the direct instrument for the owner's rule.

### Test 3 — `the_addressing_is_declared_divergent_and_bounded`

Not a fidelity test — a **scope-and-safety** test, because fidelity is impossible here:

- (a) `raw_keys` output `& engram_slot_mask < table_size[0]` for the extreme keys `0` and
  `2^31 − 1` (guards the one op on the training path that indexes a table);
- (b) a `#[test] fn the_reference_hasher_is_not_the_training_path()` that asserts
  `NgramHasher` has no caller — or, better, that it is gone. **The cheapest honest
  outcome of the whole oracle exercise is deletion**: `hasher.rs` is a port that
  declares itself non-bit-for-bit (`hasher.rs:12-13`), is called by nothing, and whose
  5 property tests certify nothing about training. An oracle you ship but never run is
  the same defect class as a silent fallback — it makes the tree *look* checked.

### What I would NOT build

The report's `short_conv_matches_official_reference` and
`ngram_hash_matches_official_reference`. Both pin code with no production caller
(§1.8), and the first's headline "divergence" is not a divergence. If the conv is ever
wired, `engram_lam_max`'s doc comment is where the gate-vs-conv interaction belongs —
and note the conv is *inert* today in a second sense: `conv_weight` is `None` unless
`with_short_conv` is called, so Eq 5 is not merely unused, it is unconstructed.

---

## 3. The owner's rule, applied

> Rule as I read it: a deviation from a paper is allowed **only** if verified
> bit-for-bit against a reference implementation.

The rule is *stronger* than the reports treat it, and applying it literally sorts every
finding into one of three bins. Note that the rule is about **deviation from a paper**,
and only Engram has an official implementation in hand (DeepSpec is named by the DSpark
paper; I did not fetch it and the brief does not ask me to).

### Bin 1 — "fix the code" (nothing to do with the oracle)

| # | finding | why the oracle is irrelevant |
|---|---|---|
| **E8** `hasher.rs` unused | the reference cannot arbitrate whether *we* call it. The arbiter is `grep`. And the port declares itself non-bit-for-bit, so the rule gives it no cover at all: a deviation that is neither bit-for-bit nor on the training path has no justification under any reading. **Delete or feature-gate.** Highest value-per-byte item in the whole review. |
| **E11** λ comment says "a tuned CONSTANT, not a learned value" (`loop_block.rs:53-54`) | kNN-LM has no implementation in this project; the rule cannot license the deviation, so the comment must simply be corrected. `w_mem` is a sigmoid of a controller output (`loop_block.rs:377`) and `lam = w_mem.clamp(0, lam_max)` — learned and bounded. **The tree already says the opposite correctly 100 lines away** (`schema.rs:162-166`: "a floor, not a learned value"), so this is a §1.7 one-word-one-meaning violation *inside the same mechanism*, and it is a two-line fix. Nothing in the reports connects the two comments. |
| **`lib.rs:233`** "diverges for \|s\| > ~1" | settled by arithmetic on the reference's own formula (§1.5): identical at \|s\|=1, reconvergent past ~30, peak 0.104 at 3.36. Not "unverifiable" — **wrong**, and the oracle was in hand. |
| **`lib.rs:231-232`** cites the reference while transcribing our own `add` | the rule is a §1.4 claim; the claim is wrong about the file it names. One word: `add_scalar` → `clamp_min`, in code *and* comment. |

### Bin 2 — "the paper and the reference disagree, so find out which the reference does"

**A and B from §1.3, plus the zero-init.** Four places where 2601.07372 and
`engram_demo_v1.py` disagree and our code silently picked one:

| site | paper | reference | ours | status |
|---|---|---|---|---|
| gate | Eq 4: plain `σ(s)` | `σ(√\|s\|·sign s)` | reference | **declared** at `lib.rs:231`. Good — this is the one the rule was written for and it is honoured. |
| conv init | §4.1: **zero-init** | `nn.Conv1d` default (kaiming) | paper | declared, and correct to prefer the paper. **But the code is dead** (§1.8), so the declaration is about nothing. |
| norm gain | Eq 4: none | `nn.RMSNorm` affine | paper | **undeclared** |
| proj bias | Eq 3: none | `nn.Linear` bias | paper | **undeclared** |

The last two are the finding. Under the rule they are *permitted* — the paper is a
named, citable source and we follow it — but the rule's whole purpose is that the
choice be **visible**. Nobody wrote them down, and `lib.rs:231`'s comment, by naming the
reference in that function, implies the whole gate path is the reference's. Test 2 above
is the instrument that makes them visible.

### Bin 3 — "the reference cannot settle this" (and the reports should say so)

**E12** ("196B Engram / 552B backbone, 0.36x") and **E14** (the allocation law quoted as
a fraction of *total* rather than of `P_sparse`). The demo's config carries no model
size (§1.7) and `layer_ids = [1, 15]` disagrees with the paper's layer choice — it is a
shape demo. So these two need the paper's Tables 1/3, which the report already checked
and got right for the numbers that exist. The correct action is the one the report
reaches by a different route: **delete the unsourced 552B/196B pair and keep only
5.7B/26.7B = 21.3 % and the §3.1 "20–25 % of the sparse budget" sentence.** Nothing
here is a code fix; it is a citation deletion, and it is cheap.

**E13** does not belong in any bin — see §5. The report's own arithmetic is wrong.

### Count, for the record

3 BUG + 3 UNVERIFIABLE in `engram.md`; under the owner's rule: **2 are pure code fixes**
(E8, E11), **1 is a wrong claim the oracle settles** (the `|s| > ~1` line, plus the
`add`/`clamp` misquote beside it), **2 are undeclared paper-vs-reference deviations**
(gains, biases), **2 cannot be arbitrated by any reference** (E12, E14), **1 is
misdiagnosed** (E13). That is six items of work, of which four are comment-shaped and
two are code-shaped, and the two code-shaped ones (E8 delete, `clamp_min`) are both
smaller than the paragraph explaining them.

---

## 4. DSpark: which one-liner first

### 4.1 Both are one-liners only if you count the wrong thing

**D1 (the off-by-one)** is one line of *substitution* — `model.rs:204`,
`let ids_raw = targets.clone();` → `input_ids.clone()` — **and it silently removes a
guard.** `aux_loss` opens with `let ids = ids?;` (`model.rs:259`). Today `ids_raw` is
derived from `targets`, so **`targets.is_some()` is the de-facto switch for the whole
aux block**, and three call paths depend on that:

| call site | `targets` | today | after a naive one-line fix |
|---|---|---|---|
| `train/src/lib.rs:1164` (the step) | `Some(y)` | aux runs | unchanged |
| `train/src/lib.rs:1455` (held-out eval, on `model.valid()`) | `None` | aux skipped | **full DSpark objective computed and discarded, per eval batch** |
| `train/src/lib.rs:1608` (`--eval-depths` loop) | `None` | aux skipped | same |
| `train/src/lib.rs:2074-2075` (reload-divergence check) | `None` | aux skipped | same |
| `model.rs:81` `forward` (generate/serve) | `None` | aux skipped | early-returns (`t ≤ k+1` ⇒ `n = 0` ⇒ zeros), so harmless |

So the honest diff is **two lines**: substitute the tensor *and* keep the
`targets.is_some()` guard explicit (`let ids = ids.filter(|_| targets.is_some())` or an
early return in `forward_with_latent`). Neither `dspark.md` §2 nor AGENTS.md §3.3 names
this. It costs no training number — the step always passes `Some(y)` — but it puts a
discarded training loss inside every held-out eval, which is exactly the "a fallback
whose reader cannot tell which arm ran" shape ADR-0019 is about.

**D9 (the missing Markov feature in the confidence head) is not a one-liner.** It is
three sites plus a checkpoint break:

1. `markov.rs:205-239` — `apply_block_logits` computes `prev_emb` per step
   (`:216-224`) and **throws it away**; it must also return the stacked `[B,N,L,r]`.
2. `aux.rs:45` — `AcceptRatePredictor::new(d_model, …)` → `with_markov(d_model, rank, …)`
   (`rank` is already in scope at `aux.rs:41`).
3. `aux.rs:307-312` — pass the returned embeddings instead of `None`.

And step 2 **changes a parameter's shape** (`proj.weight` `[d_model,1]` →
`[d_model+rank,1]`). The aux head is serialized with the model
(`aux.rs:31-32`: "Serialized with the model"), so **every existing checkpoint with a
trained or freshly-initialised `conf` head becomes unresumable** — a loud failure at
load, but a real one, and it is the kind of thing that has to be decided before the
change, not after.

### 4.2 The ordering

**D1 first, on three grounds.**

1. **It is the one that changes the objective's meaning.** D1 makes the head learn a
   *different function* of its inputs (today it is conditioned on the token its own base
   logits already predict, and asked to go one step further); D9 gives an existing
   function one more input feature. D1 is a semantic break, D9 is a capacity addition,
   and §1.2's A/B-or-death needs the semantic break isolated.
2. **It is shape-preserving and checkpoint-safe.** D1 touches no parameter, so the
   `official_v5*` lineage still resumes. Doing D9 first would strand that lineage and
   couple the shape break to the A/B.
3. **Its gate is free and zero-tolerance.** D1 has an exact test: build a 1-row
   sequence with `x[i] = i`, so every position is identifiable, and assert the tensor fed
   to the head at `(p, s)` is `x[p+s]` while the CE target is `x[p+s+1]`. Integers — no
   tolerance, no fixture, and it is **red on the current tree by exactly one**, which is
   the only property that makes a gate trustworthy. D9's gate is a width assertion,
   which is also free, but it is a *shape* check on a struct that has no shape bug until
   somebody fixes it.

**One correction to the report's mechanism, because it will mislead the fixer.**
`dspark.md` §2 says the head "is then asked to move the prediction one step further, to
a token **that no bias on that feature can address**." That is false: `W₂` is `r × V`
with r = 64 and V = 256, so the bias can address every token in the vocabulary. The
actual defect is sharper and worse: with `ids = targets`, the Markov feature at step
`s+1` is the **CE target of step `s`**. So the head is trained as an autoregressive
model *over the very block it is being scored on*, one step of leakage at a time — while
`U_k` (the base logits it corrects) points one position earlier than the target it is
being matched against. That is the thing to write in the comment when the one-liner
lands; the report's sentence would send a fixer looking for a capacity problem that does
not exist.

**Then D9, and gate it by a width assertion** (`d + markov_rank`, not `d`) exactly as
`dspark.md` §7(3) says — that proposal is right, and the constraint the report does not
state is that the fix must **reuse `VanillaMarkov`'s existing `W₁`**, not add a second
one. Two `W₁`s would be a new mechanism with its own init story, which is the thing the
paper does not do and the thing `§1.4` would then require naming.

### 4.3 Does fixing D1 change every DSpark number ever recorded? — measured, not assumed

`~/logs/` is the archive; the `aux=` field appears on a step line only when the aux block
ran and its scalar read back (`train/src/lib.rs:1364-1372`).

| log | `aux=` lines | held-out BPB on record | DSpark was live |
|---|---|---|---|
| `train_nokda.log` | **0** of 239 steps | **4.997** @ 6500 | **no** |
| `train_engram25k.log` | **0** of 239 | 6.453 @ 2000 (already retracted) | no |
| `train_kda_full.log` | **0** of 11 | — | no |
| `official_v5e.log` | **4** | **6.351** @ 1500 | **yes** (`aux=0.1199` at step 1500) |

So the answer is **no, and the split is clean**: the project's single cleanest
held-out number — the 4.997, the only one in the archive free of both §3.2 defects —
was produced by a run with **no aux at all**, and the off-by-one cannot touch it. The
6.351 was produced with the aux live and **is** affected; it is already disqualified for
`use_kda = true`, and this adds a second reason. Same answer for the 27 other logs
carrying both `aux=` and `bpb=`: every one of them is aux-on and therefore
off-by-one-affected. **Nothing in the archive needs retracting because of D1; one
headline number in §3.1 needs one more reason on its list.**

(The 6.351's own log also shows the run dying at step 2000 — "50 non-finite losses in
one log window" — and `--guard` restarting it into the same failure. Worth knowing
before anyone builds an A/B on this lineage.)

---

## 5. The 24 % / 32 % in `config/schema.rs:135` — the report is wrong, and the real answer is worse

### 5.1 Confirm the arithmetic error: **there isn't one**

`config/schema.rs:135`:

> 2.4M memory params against 7.5M = **24% of the model, 0.32x** — the published
> operating point, not a guess.

The report's E13: *"2 400 000 / 7 500 000 = 0.32 = 32%, not 24%. The '0.32x' in the
same sentence is right and the '24%' is wrong; they cannot both hold."*

**They can both hold, because there are two denominators, and the repository already
defines which is which.** `loop_block.rs:719-720`:

```rust
const BACKBONE_SMALL: usize = 7_500_000;
let share = mem as f64 / (mem + BACKBONE_SMALL) as f64;
assert!((0.20..0.30).contains(&share), ...);   // 2.4 / 9.9 = 0.2424  ✓
```

- **"24 % of the model"** = memory ÷ (memory + backbone) = 2.4/9.9 = **24.24 %** — of
  the **total**.
- **"0.32x"** = memory ÷ backbone = 2.4/7.5 = **0.32** — of the **backbone**.

Both are exactly right, they are the two conventional denominators, and the repo's own
test computes the first one. `engram.md`'s "they cannot both hold" is the classic
percent-of-what ambiguity, and the repo had already disambiguated it in code. **E13's
verdict is a misdiagnosis, and the remedy it implies ("write 32 %") would introduce the
error it claims to remove** — it would make the sentence say "32 % of the model" where
24.24 % is the share of the model.

### 5.2 What *is* wrong is worse, and it is already on the project's own list

Two independent stalenesses, both invisible in the sentence and both load-bearing:

1. **The denominator is retracted.** `7_500_000` is the `small` count AGENTS.md §3.2
   retracts; the measured figure is 9 195 854. `AGENTS.md:717-718` already lists this as
   an open glossary disagreement: *"the `small` param count the memory budget is argued
   from (`loop_block.rs:706`, `schema.rs:130`)"*. The report found the line and did not
   find the note.
2. **The numerator is not the number that ships.** The same test asserts
   `tables == [32_768; 3]` (`loop_block.rs:735`) — i.e. it *knows* the rows round up —
   and then computes the share from the nominal `2_400_000` anyway. The shipped memory
   is **3 145 728** parameters.

### 5.3 Propagation: **into a real memory budget, in seven presets, as the justification for the number**

The question asked was comment-vs-preset. Precise answer:

- The *ratio* as a literal appears in **no** preset. `configs/*.toml` carry
  `engram_rows = 25_000` and nothing else about capacity.
- But the ratio is **the stated reason that number is 25 000**:
  `configs/small.toml:31-32` — *"the curve is flat 300K-500K and silent below, so **the
  ratio anchors the default** and the 500K point stays a ladder rung."* Read that with
  `schema.rs:130-136`: the measured curve says nothing below 300 K, so the **ratio is
  the only thing that picks 25 000**. And `engram_rows = 25_000` is live in `base mor
  nano one_b p150 small swift50`. So the ratio is comment-shaped and the *conclusion it
  justifies is a real 3.15 M-parameter memory budget in every production recipe.*

**And here is the part the report got exactly backwards.** `engram.md` E13 closes:
*"Bounded: the ratio is lower than the paper's 21.3 %-of-total would suggest, not
higher, so the conclusion survives."* On the numbers that actually ship:

| preset | memory | total | share |
|---|---|---|---|
| `nano` | 3 145 728 | ~7.19 M | **~43.8 %** |
| `small` / `mor` (flagship) | 3 145 728 | 9 195 854 | **34.2 %** |
| `base` | 3 145 728 | ~13.07 M | ~24.1 % |
| `p150` | 3 145 728 | ~163 M | ~1.9 % |

The arm is **above** the paper's 21.3 % and above its own claimed 20–25 % band on the
flagship and on the smoke preset, not below. The containment argument is inverted.

**The consequence nobody has stated, and it is the reason this is not a comment fix:**
the only assertion in the tree carrying the 20–25 % claim is
`loop_block.rs:721-725`, and it is computed from inputs the model does not use. Make it
describe the model that ships — `3_145_728` and `9_195_854` — and it **fails**:
0.3421 ∉ (0.20, 0.30). The band is not enforced against the shipped configuration; it is
enforced against a configuration that stopped existing. The only real gate,
`preset_exec.rs:479-490`, measures the actual share on the instantiated model and
allows **50 %** — which is how a 43.8 % `nano` passes a test whose stated purpose is
"no preset is a lookup table with a model attached".

**So: urgent, but not because of the arithmetic.** Three sub-items, in order of value:

1. **Decide the real band and say which presets are inside it.** 34 % on the flagship is
   a *decision*, not an error — the arm may well be right at a third of the model on a
   9.2 M byte-LM. But it is currently documented as "DeepSeek's own operating point",
   which at 34 % is false, and that mislabelling is what a future reader will quote.
2. **Make the assertion describe the model.** `mem` → `3 × tables[0] × dim`; drop
   `BACKBONE_SMALL` for the measured count. It will go red, which is the point: the
   red is the finding.
3. **Delete `24%` and `0.32x` from `schema.rs:135` and `small.toml:26` and replace both
   with the two denominators spelled out**, e.g. *"3.15 M memory rows (32 768/order after
   the power-of-two round) against a 9.20 M model = 34 % of the total"*. One sentence,
   no arithmetic left to misread, and the retraction becomes unnecessary because the
   number is current.

---

## 6. My top three

1. **The oracle is not wired, and the one fidelity claim made about it is false in three
   ways.** `grep engram_demo_v1` over `*.rs` returns five comment lines and no test.
   The arm is 34 % of the flagship model and on in seven of eight presets. Against the
   reference: our gate has **no learnable RMSNorm gains** (it has none; the reference has
   `hc_mult` of them, `engram_demo_v1.py:355-356`, `:368-370`) and our key/value
   projections have **no biases** (`lib.rs:136-148`; the reference's `nn.Linear`s do,
   `:351-354`). Both are defensible — the paper has neither — and both are undeclared,
   which under the owner's rule is the whole failure. `engram.md` E1's "our code matches
   the reference implementation" was written with the file open and is not true.
   `lib.rs:231-232` compounds it by citing the reference while transcribing our own
   `add_scalar(1e-6)` for the reference's `clamp_min(1e-6)`.

2. **The 20–25 % memory claim is not enforced against the model that ships, and
   enforcing it fails.** `loop_block.rs:719-725` divides nominal 2 400 000 by a
   **retracted** 7 500 000; the shipped table is 3 145 728 against a measured 9 195 854
   — **34.2 % of `small`, ~43.8 % of `nano`**, above the band the comment claims to sit
   in, not below it as the report asserts. Fix the inputs and the assertion goes red;
   the only live gate (`preset_exec.rs:479-490`) allows 50 %. The report called a
   correct two-denominator sentence an arithmetic bug, and its containment argument is
   inverted. `AGENTS.md:717` already had the denominator on the open list; nobody
   followed it to the *consequence*.

3. **The DSpark off-by-one is one line of substitution and one lost guard, and its gate
   is the only zero-tolerance red-to-green test in the whole set.** `model.rs:204` →
   `input_ids.clone()` also removes the `let ids = ids?` short-circuit at `model.rs:259`,
   which is currently the de-facto `targets.is_some()` switch — so the held-out eval
   (`lib.rs:1455`), the `--eval-depths` loop (`:1608`) and the reload check (`:2074`)
   would start computing and discarding a full DSpark objective. Do D1 first with the
   guard made explicit; D9 is three sites and **breaks every existing checkpoint's
   `conf` head shape**, so it goes second. And the report's mechanism sentence — "a
   token that no bias on that feature can address" — is wrong: `W₂` is r×V and addresses
   all 256. The real defect is that the Markov feature at step `s+1` *is* the CE target
   of step `s`.

**Runners-up, in one line each.** `burn-engram`'s `hasher.rs` and `with_short_conv`
have no production callers, and two of the report's three recommended gold tests would
certify them; the `per-branch`-vs-`per-concatenation` norm "divergence" they are meant
to name is not a divergence (`nn.RMSNorm(D)` and `mean_dim(3)` are the same function).
The report's proposed fixture constant `0.804_129_4` is off by 3.0e-4 from the true
0.80442974, so its recommended test fails on the tree it certifies — a gold-vector test
with no shipped fixture is a number in a document. And a tolerance-based gate built on
well-scaled inputs is *provably blind* to the one divergence the oracle was fetched for:
`add` vs `clamp_min` is 5.6e-8 at |s| = 2 (below one f32 ulp — same bits) and
**1.04e-4** at |s| = 1e-6, so the fixture must reach |s| ≤ 1e-6 or it tests nothing.
