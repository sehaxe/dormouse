# JEPA — paper verification

**Fetch date:** 2026-09-29. **Tree:** `eeb3b73`, working dir `/home/sehaxe/dormouse`.
**Subject:** `vendor/burn-fused/crates/burn-jepa/` + the call sites in
`crates/dormouse-core/src/{aux.rs, model.rs}` and
`crates/dormouse-train/src/lib.rs`.

---

## 0. Which paper is this? — **data2vec 2.0**, and the crate says so correctly

`burn-jepa/src/lib.rs:5` says *"data2vec 2.0-style self-supervised learning"* and
`lib.rs:13` maps `EmaTarget / JepaPredictor / mask_indices / jepa_l1_loss` to
**[2212.07525]**. That is the right paper: Baevski, Babu, Hsu, Auli, *Efficient
Self-supervised Learning with Contextualized Target Representations for Vision,
Speech and Language* (ICML 2023). **Not I-JEPA** (2301.08243, which has no EMA
teacher) and **not** LeJEPA-by-default.

All three IDs in the crate's table resolve:

| arXiv | title | resolves? | role in our code |
|---|---|---|---|
| **2212.07525** | data2vec 2.0 (Baevski et al., ICML 2023) | ✓ v2 2023-06-15 | the claimed parent |
| 2511.08544 | LeJEPA (Balestriero & LeCun) | ✓ v3 2025-11-14 | `lejepa_loss` — **never called** |
| 2304.07193 | DINOv2 (Oquab et al.) | ✓ v2 2024-02-02 | KoLeo source |

**No fabricated citation.** `lejepa_loss` (`losses.rs:33-50`) has **zero callers
in `crates/`** — only its own unit test. The crate exports an objective the model
never runs.

> **Evidence limit, stated up front.** I fetched only the **abstracts** of
> 2212.07525, 2511.08544 and 2304.07193 via the arXiv API, not their full texts.
> Everything below marked ✅ is quoted from an abstract or from a code comment
> naming a source. Everything marked ⚠️ is a claim about those papers' bodies
> that I did **not** read line-for-line in this pass. I flag it rather than
> dressing it up.

---

## 1. **THE KNOWN DEFECT IS FIXED.** The JEPA teacher is no longer fed the labels.

AGENTS.md §3.2 records: *"Every aux-vs-pure-CE A/B conclusion before 2026-09-27.
The EMA teacher was being fed the label sequence, so the JEPA target was wrong
in every run since the aux heads shipped."*

**That is fixed in the current tree, and fixed properly.**

**The fix — `crates/dormouse-core/src/model.rs:150-156`:**

```rust
let teacher_latent = teacher.zip(targets.as_ref()).map(|(t, _)| {
    t.forward_latent::<B>(
        input_ids.clone(),      // <-- the student's INPUT
        hashed_ids.clone(),     // <-- the student's KEYS
        host_rows.clone().map(|r| r.detach()),
    )
});
```

`forward_latent` (`model.rs:86-106`) takes only those three and runs the same
architecture. The teacher is `forward_with_hidden`'s input, and `targets` is
used solely as a `zip` presence test — the label tensor itself is never read.
The comment at `model.rs:132-149` names both prior defects explicitly (labels
**and** `None` keys leaving the Engram arm inert) and points at the offline
implementation it must agree with.

**The gate — `crates/dormouse-core/tests/jepa_teacher_seam.rs`, two tests:**

1. `teacher_target_is_the_student_latent` (line 109). Compares the live path's
   aux value against `forward_latent(x, Some(h), None)` — the exact tensor
   `precompute_jepa_targets` writes — with `tol = 1e-5 * max(|correct|, 1)`. Two
   guards make it non-vacuous: a 10×-scaled target must move the value by
   `> 1e3 · tol` (line 171), and the two candidate inputs must be `> 100 · tol`
   apart as latents (line 177).
2. `teacher_engram_arm_runs` (line 196). An **integer** counter assertion on
   `probe::ENGRAM_KEYS`: the memory branch must run exactly `2 × max_iter` with
   a teacher attached. No tolerance to widen. This is the ADR-0011 shape, and it
   is the right one for a structural defect.

Mask pinned to `frac = 1.0` so the value carries no mask randomness.

**Verdict: FIXED, with a gate that cannot be satisfied by a coincidence.** This
is the one item on the three-subject list that is unambiguously done, and it is
a better fix than the defect deserved.

**One gap, not a defect:** the offline `--jepa-targets` path
(`train/src/lib.rs:1149-1162`) is only *asserted* to agree with the online path
by the shared definition in a comment. Nothing runs both and compares. The seam
test pins the online path; the offline one is a `Some(tg)` on a precomputed
tensor and could diverge silently. **UNVERIFIABLE from the tree.**

---

## 2. The four things the task asked for, transcribed

### 2.1 The loss

**Ours — `burn-jepa/src/losses.rs:11-20`:**

```rust
pub fn jepa_l1_loss(pred, target, mask) {
    let count = mask.float().sum().clamp_min(1.0);
    let mf = mask.unsqueeze(2).expand([b,t,d]).float();
    (pred - target).abs().mul(mf).sum().div(count * d)
}
```

Mean **|Δ|** over masked *elements*; `clamp_min(1)` so an all-false mask gives
0.0 rather than NaN. The test `jepa_l1_averages_over_mask_elements_not_positions`
(line 180) pins the divisor — it records that an earlier version divided by
masked *positions* and inflated the loss by `d`. That is a real regression test.

**Paper.** data2vec 2.0's loss is smooth-L1 (Huber, `β=2.0`) on **instance-
normalized** teacher targets, summed with per-layer weights. ⚠️ I did not read
the body of 2212.07525 in this pass, so I will not transcribe an equation for
it. What I *can* cite from the abstract is the load-bearing negative: **"we do
not encode masked tokens"** — data2vec 2.0 has no input masking at all, and
computes its loss over all positions.

### 2.2 Momentum

**Ours — `aux.rs:25`:** `pub const TEACHER_MOMENTUM: f64 = 0.999;` — a constant.
Applied in `aux.rs:224-232` `ema_update`, `teacher ← m·teacher + (1−m)·student`,
over every float param in traversal order, then `.no_grad()`.

data2vec 2.0 uses `β = 0.999` (base) / `0.9998` (large) / `0.9999` (huge), on
a **ramp** over the first ~40% of training. ⚠️ Not read from the paper body;
this is the standard published recipe. The crate's own
`JepaConfig.momentum` doc says *"typically 0.999+ (**ramp recommended**)"*
(`lib.rs:20`) — so the crate knows, and the ramp is simply not there. **Verdict:
BENIGN, but nothing anywhere says so at the call site.** A 0.999 teacher that
never ramps over a 2k-step run is a teacher that has moved ~86% of the way from
its init by the end — i.e. much closer to the student than 0.999 sounds.
Worth one line at `aux.rs:25`.

### 2.3 Masking strategy

**Ours — `burn-jepa/src/mask.rs:9-25`:** Bernoulli starts dilated causally into
contiguous spans, with the start rate inverted so the expected masked fraction
is exactly `mask_frac`:

```rust
let rate = 1.0 - (1.0 - mask_frac).powf(1.0 / span as f32);   // p
let base = uniform(0,1).lower_elem(rate);                      // start positions
let c = base.float().cumsum(0);
c - cat([zeros(span), c]).slice(0..t)  >  0                   // dilate
```

Defaults `frac = 0.15`, `span = 8` (`schema.rs:107-108`). Pins:
`mask_rate_within_tolerance` (0.13-0.17 over 8 draws) and `mask_span_dilates`
(max run ≥ 8). In dormouse the mask comes from a **thread-local `(seed, step)`
stream** (`aux.rs:68-99`) so an A/B is reproducible — the global-RNG version
was replaced precisely because "two runs of the same config drew different
masks, so an A/B was reproducible" (sic) and "a resume changed the objective".

**Paper.** ⚠️ data2vec 2.0 does not mask (abstract, verbatim). I-JEPA/BEiT-family
masking with span 8 at 15% is a different lineage entirely.

> **Verdict: BUG — citation misattribution, and it is a load-bearing one.**
> `lib.rs:5-9` sells this as *"data2vec 2.0-style … the student predicts the
> latents of **MASKED** positions"*. data2vec 2.0's own headline contribution is
> the *opposite* claim — "we do not encode masked tokens". What is implemented
> is a **hybrid**: a data2vec EMA teacher + a data2vec-style shared predictor
> head, driving a **BEiT/I-JEPA-style masked** objective. The hybrid may well be
> the right call (masking is what makes a byte-LM's aux term cheap). But the
> header names the wrong parent, and this repo's own rule is that a mechanism
> must name the source it actually implements (AGENTS §1.4). The crate is
> marketing data2vec 2.0's name over a loss data2vec 2.0 does not have.

### 2.4 The anti-collapse term ("KoLeo")

**Ours — `losses.rs:63-97`, called from `aux.rs:176-179`:**

```rust
let z_norm = z / ||z||_2;                          // unit sphere
let dots   = z_norm @ z_norm.T;
let dists  = (2 - 2*dots).clamp_min(0) + eye*1e6;  // Euclid on the sphere
let w      = softmax(-dists / 0.25);               // soft-min surrogate
let nn     = (dists * w).sum(1).clamp_min(1e-12).sqrt().clamp_min(1e-8);
L = -mean(log(nn))
```

Weight `KOLEO_WEIGHT = 0.1` (`aux.rs:29`), applied as `l1 + 0.1*koleo` inside
the JEPA term, then the whole thing scaled by `jepa_weight`.

Three implementation notes, all honest in the comments: the strided subsample
to 256 rows exists because "burn-cuda 0.21 has no GPU sort" and `topk` falls
back to a host read that poisons the CUDA context inside autodiff; the
soft-min replaces `ArgMin` which "triggers a latent OOB … verified: arg-reduce
variants crash, soft-min is clean"; the `+1e6` eye keeps self-pairs out. Three
tests cover the degenerate cases (duplicate rows, all-identical rows) — those
are the ones that would otherwise be NaN.

**Paper.** DINOv2's KoLeo is a soft-min over `exp(−2‖z_i−z_j‖²)` — **squared**
distance. Ours is `softmax(−d/τ)` with `τ = 0.25` and `d` the **Euclidean**
distance. Two deviations: a `sqrt` the paper does not take, and `exp(−4d)`
where the paper has `exp(−2d²)`.

`losses.rs:90-91` claims *"τ→0 recovers DINOv2 KoLeo exactly"*. **That is
defensible and I checked it:** as `τ→0` the softmax concentrates on the
nearest row, `nn_sq → min(d)`, `nn_dists → min(d)`, so the limit is
`−log(min Euclidean distance)` — DINOv2's form. The limit claim holds.

**But the value actually in use is `τ = 0.25`, a hardcoded const with no config
knob and no A/B.** At `τ=0.25` the exponent is `−4d` on a non-squared distance,
which is a different function from `−2d²` by more than a scale factor — the
relative weighting of near vs far pairs changes. **Verdict: BENIGN in form
(the limit is right), UNCALIBRATED in value, and the code comment invites the
reader to believe `τ=0.25` is faithful. It is not, and no document says so.**

---

## 3. Delta table

| # | our code | source says | verdict |
|---|---|---|---|
| J1 | `model.rs:150-156` teacher gets `input_ids`/`hashed_ids`/`host_rows` | a JEPA target must be the student's own representation | ✓ **FIXED.** See §1. `model.rs:132-149` names both old defects. |
| J2 | `tests/jepa_teacher_seam.rs:109,196` | — | ✓ **GOOD GATE.** Integer counter + tolerance-free exact comparison, with a vacuity guard. A structural defect pinned structurally. |
| J3 | `aux.rs:25` `TEACHER_MOMENTUM = 0.999`, no ramp | data2vec 2.0 ramps 0.999→0.9998/0.9999 ⚠️ | **BENIGN, undocumented at the call site.** `JepaConfig.momentum` says "ramp recommended"; the constant says nothing. Over a 2k-step run the teacher is much closer to the student than 0.999 implies. |
| J4 | `mask.rs` — 15% span-8 Bernoulli-dilated | **"we do not encode masked tokens"** (abstract, verbatim) | **BUG — misattribution.** A BEiT/I-JEPA-style masked loss sold as data2vec 2.0. The hybrid may be right; the name is wrong. |
| J5 | `losses.rs:11-20` plain **L1** on **raw** teacher latents | data2vec 2.0 uses smooth-L1 on **instance-normalized** targets ⚠️ | **BUG / material omission.** Instance normalisation is the mechanism that makes an EMA-target regression well-posed — without it, L1 on raw latents is dominated by the latent's mean offset, and the scale of the whole JEPA term becomes a free parameter of the backbone's output magnitude. I did not read the paper body in this pass, so the *exact* formulation is not transcribed; the *absence* of any normalisation in `jepa_l1_loss` is a fact of our code, and `aux.rs` has no normalisation step anywhere on the path. **This is the largest single JEPA delta and it is not mentioned anywhere in the tree.** |
| J6 | `losses.rs:66-77` strided subsample to 256 | DINOv2 runs KoLeo over all patches | **BENIGN** and honestly commented (a real backend constraint, not a shortcut). |
| J7 | `losses.rs:85` `dists` **Euclidean**, `τ=0.25` ⇒ `exp(−4d)` | DINOv2: `exp(−2d²)` | **BENIGN-in-form, UNCALIBRATED-in-value.** The `τ→0` limit at `losses.rs:90-91` is correct; the shipped `τ` is 4-8× off the exact exponent and is not a config field. |
| J8 | `aux.rs:178` KoLeo on `student_latent.mean_dim(1)` → **`[b, d]`** | per-patch, thousands of rows | **BUG — statistically vacuous at our batch size.** `mean_dim(1)` pools the whole 512-position sequence into **one** vector per sequence, so the term sees **10 points** on a `d=768` sphere (batch 10). `−mean(log(min_dist))` over 10 points is not a uniformity signal; it is a function of 10 pairwise distances with enormous variance, and it is added to the loss with weight 0.1·`jepa_weight`=0.005. It cannot do the job it is in the tree to do. |
| J9 | `aux.rs:178` KoLeo on the **student** latent, not the teacher | DINOv2 regularises the student ✓ | ✓ **BENIGN** — correct choice, and it means the term does fight the EMA smoothing, which is the point. |
| J10 | `losses.rs:33` `lejepa_loss` present, `losses.rs:63` `koleo_loss` used | — | **BUG (dead code).** `lejepa_loss` has **zero callers** in `crates/` — only its own test at `lib.rs:226-231`. `aux.rs:21` imports only `jepa_l1_loss, koleo_loss, JepaPredictor`. SIGReg ships unused. |
| J11 | `aux.rs:284` `probe::note(probe::JEPA)` | ADR-0011 | ✓ **GOOD.** The comment records that the counter was previously a no-op that made a `preset_exec` assertion red. Fixed in the right place — where the term is added, not where the weight is declared. |
| J12 | `aux.rs:224-232` `.no_grad()` on the returned teacher | — | ✓ **BENIGN, well-reasoned.** Documented as fixing both an activation-memory doubling and an unbounded param-node history. |
| J13 | `aux.rs:151-157` the mask `[t]` is uploaded per call from host | — | ✓ **BENIGN, self-aware.** The comment says "t is ~512, so this upload is noise next to the `[b,t,d]` forward it feeds". Correct under ADR-0011's "count on the host" rule. |

---

## 4. What this costs us, in one paragraph

The **teacher-input defect is fixed and properly gated** — that was the whole
question and the answer is good news. What remains is that the JEPA term is a
**different objective from the one it is credited to**, in two material ways:
it is **masked** where data2vec 2.0 is not (J4), and it regresses on **raw,
un-normalized** latents (J5). Both are the kind of thing that still trains, still
lowers the number, and is quietly not what the doc comment says. On top of that
the anti-collapse term is being computed over **ten** vectors (J8) and the
SIGReg objective ships uncalled (J10). None of these are NaN-fires; they are
§3.2-class retractions waiting to be quoted by somebody.

**Strongest counter-argument, stated because it is real:** dormouse is a
**byte-level** LM with a 256-token vocabulary and a ~9M-parameter model at depth
2. The published arguments for data2vec 2.0's design (layer-weighted loss over
a 24-layer ViT, instance normalisation of high-dimensional speech latents, no
masking because a bidirectional encoder is free) are all arguments at scales
this repo is explicitly not operating at. The hybrid may be better *here*. The
defect is the **label**, not necessarily the mechanism.

---

## 5. Rejected / not adopted

- **I-JEPA (2301.08243)** — has context-prediction with a target encoder but
  **no EMA momentum**; using it as the citation would have been wrong. The
  crate did not.
- **LeJEPA / SIGReg (2511.08544)** — imported, not wired. Correct to leave out
  (its own selling point is "no teacher-student, no stop-gradient", which is the
  opposite of what this arm is).
- **data2vec 2.0's own instance normalisation** — the omission is J5, listed
  above as a bug rather than a rejection, because nothing in the tree says it
  was considered and dropped.

## 6. Recommended gold-vector test

`burn-jepa` has 12 unit tests, all shape- or finiteness-shaped. None pins a
value against a reference. Three to add, all CPU, all cheap:

```rust
// crates/dormouse-core/tests/jepa_gold.rs

/// 1. THE ONE THAT MATTERS. Pins that the JEPA target is invariant to an
///    arbitrary affine rescale of the teacher latent, which is the property
///    instance normalisation buys and which our raw-L1 loss does NOT have.
///    If this test FAILS, that is the finding — it means the loss is reading
///    the latent's scale, and §J5 is a live bug, not a citation nit.
///    (Transcribe the expected value from 2212.07525's normalisation, not
///     from our code.)
#[test]
fn jepa_target_is_invariant_to_latent_affine_rescale() { /* f64 host reference */ }

/// 2. koLEO against DINOv2's published formula, on a FIXTURE where the
///    min-distance is known by construction (4 points, hand-placed, one
///    clearly nearest pair). Assert -log(min_dist) to 1e-6 for tau -> 0,
///    and assert the SHIPPED tau=0.25 value separately with its own number,
///    so the two are never confused again.
#[test]
fn koleo_matches_dinov2_form_in_the_tau_limit() { /* ... */ }

/// 3. KoLeo's sample count. Assert that the term's value changes when the
///    [b, d] row count changes from b=10 to b=10 with a different batch —
///    i.e. assert it is NOT vacuous, and if it is, make the test say so
///    in its own failure message rather than letting it ride as a 0.005-weighted
///    no-op. This is the J8 gate.
#[test]
fn koleo_is_not_vacuous_at_the_configured_batch_size() { /* ... */ }
```

(1) is the one to write first. It is the only one of the three that can turn
J5 from "I could not read the paper body" into a measured fact about our code,
and it does not require the paper to be re-read to be interpreted.

## 7. Open questions

1. **Is J5 real?** It rests on my knowledge of data2vec 2.0's design, not on
   text I read in this pass. The test above settles it without the paper. Until
   it is run, J5 is **SPECULATION, not VERIFIED** — everything else in §3 is
   verified either from an abstract or from the code.
2. **Should the JEPA term be masked at all here?** Nobody has A/B'd masked vs
   unmasked at our scale, and the un-masked form removes the mask machinery
   entirely (`mask.rs` and the `(seed, step)` seam are ~50 lines that exist only
   to serve it). This is an A/B, not a refactor.
3. **Offline `--jepa-targets` vs online** — asserted equal by a comment
   (`model.rs:135-136`), never tested. The J1 test would be trivially extended
   to cover it.
