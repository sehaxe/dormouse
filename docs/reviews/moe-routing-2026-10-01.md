# MoE routing over the loop's expert FFNs — the 4-expert / top-1 arm

**Date:** 2026-10-01. **Lane:** sparse expert routing in `dormouse-core`.
**Worktree:** `wt/moe-routing` off `8f97411`. **Base for every `file:line`:**
`8f97411` unless stated otherwise.
**Mid-lane correction applied:** 4 experts + top-1 (not 8–16), anchors from
Sparse-Layers (arXiv:2605.09165v2 p.6), load-balance coefficient swept rather
than copied, A/B row = dense shared FFN vs 4× top-1 at equal ACTIVE FFN FLOPs.

**Provenance bar (§1.4).** Nothing here is "verified" against a reference. The
Sparse-Layers numbers are quoted from a paper we do not have code for; our
figures are measurements on this box at this commit, CPU backend, recorded with
the interpreter that produced them. No external reference implementation of this
arm exists.

---

## 0. What is being decided

The loop already has `n_experts` TSCT FFNs blended by a **dense softmax** over
the controller's expert columns (`loop_block.rs:432`, `:516-523`): every expert
runs for every token, every pass. This arm replaces the dense blend with a
**sparse top-k selection** and asks whether specialization beats one shared FFN
at equal active FFN compute.

The corrected design, and why each choice is what it is:

| choice | value | why (and what it costs) |
|---|---|---|
| experts | **4** | external review: 8-expert top-2 evidence is at **168M+ active params**. At 9.2M total, 4× the expert count may leave too few tokens per expert for sparse routing to be stable. See §4 for the measurement of exactly that worry. |
| k | **1** first | top-2 is the SECOND A/B configuration, at the same active FFN compute, not the first. With 4 experts, top-2 activates half the expert bank per token. |
| router input | **pass number included** | the router's decision is per (position, pass). The existing controller already receives it — §3.1 — so this costs zero new parameters. |
| load-balance coeff | **swept** | no established value exists at ≤100M; large-MoE values were explicitly rejected as the default. See §5. |

---

## 1. Zero-default identity

`moe_topk = 0` is the default and means **the existing dense softmax blend,
unchanged**. The new config fields default to `0` / `0.0`, the routing arm
builds **no new parameters** (it reuses the controller's existing expert
columns), and no field on the module tree changes shape. Therefore an off-arm
model has the same parameter set — and therefore the same checkpoint — as a
build from before this file existed, and queue row 1 (pure CE) needs no
re-baseline.

Gate: `off_is_the_dense_blend_bit_for_bit` — with `moe_topk = 0` the returned
expert blend is the plain softmax of the controller's expert columns, compared
against an independent host-side softmax of the same controller output.

---

## 2. Where the routing primitive comes from (ladder rung 2: reuse)

`dormouse_mor::topk_indices` (`vendor/dormouse-fused/crates/dormouse-mor/src/topk.rs`) is
the crate's committed top-k: one `argsort_descending` + `narrow`, deliberately
avoiding `argtopk` because cubecl 0.11.0-pre.2 has a documented garbage-index
defect. `mor::route` (`crates/dormouse-core/src/mor.rs:59-84`) already turns
those indices into a 0/1 membership mask with the sanctioned `mask_fill`-on-a-
float idiom (no bool→float cast, no gather, no host round-trip).

Both are reused verbatim. Nothing new is written for the selection.

---

## 3. The router already sees the pass number

This is the one design item that costs **zero** parameters, and it is worth
stating because it was the cheapest rung of the ladder:

- `loop_block.rs:371-376` slices `iter_embed[row]` — one learned row per loop
  iteration slot.
- `loop_block.rs:381-390` (`add_iter`) **adds** that row to `h` producing
  `h_ctx`.
- `loop_block.rs:420` builds the controller input as
  `cat([h_ctx, h0])`, and `:428` is the projection whose expert columns are the
  blend.

So the controller — and therefore the expert blend — is a function of
`(hidden state, pass index)` already. Routing granularity is per
(position × pass) by construction, and no separate pass-embedding is needed.
Gate: `router_input_carries_the_pass_index` — the same hidden state routed under
two different iteration embeddings can produce different expert selections.

---

## 4. STEP 1 — the diagnostic, and its numeric anchors

**The anchor.** Sparse-Layers (arXiv:2605.09165v2, p.6) reports for its
looped-MoE that between two loop passes **25–53 % of tokens get fully DISJOINT
expert pairs, while identical pairs are only 4–14 %**. So in a working looped
MoE, routing is *mostly* unstable across passes: the same token reaches for a
different expert on the next pass most of the time.

**The question for our fixed mixture.** The brief asks what our CURRENT dense
blend does against those anchors, and states the expectation: it should show NO
iteration-divergence, because a dense blend has no selection to diverge.

That expectation is right, and it is worth being precise about WHY, because the
naive version of this measurement is a category error:

> On the dense path every expert runs for every token on every pass. The
> "selected expert set" of a token is the **whole bank** in both passes, so
> disjoint% is **identically 0** and identical% is **identically 100** — by
> arithmetic, not by measurement. Any non-zero reading on the dense path would
> mean the mask, not the model, was wrong.

So the fixed mixture is the **control endpoint**, not a second data point, and
the anchors are only meaningful **on a routed arm**. This file therefore
reports both k=1 and k=2 on the routed arm against the anchors, and states the
dense endpoint as the definitional limit.

**Measurement instrument.** ONE instrument, in
`crates/dormouse-core/src/mixture_probe.rs`, reads the weight vector the FFN
branch actually multiplied each expert's output by - captured per EXECUTED
iteration, disarmed by default, so the production path pays one thread-local
check and nothing else. Per capture it reports:

- `cross_iter_cosine` - mean over positions of the cosine between iteration
  `i`'s weight vector and iteration 0's.
- `cross_iter_top1_agree` - fraction of positions whose largest-weight expert
  is the same on both passes.
- `cross_iter_identical` / `cross_iter_disjoint` - **the paper's pair**, taken
  over each row's SUPPORT (its strictly positive entries).
- `mean_support` - mean support size: the instrument's own honesty check.
- `load`, `effective_experts` - the load distribution and its concentration.
- `top1_disagreement(a, b)` - the same statistic across two different inputs,
  which is the "does ANY input change the selection" half.

**Why `mean_support` is the load-bearing field.** The argument above - that the
dense path's identical%/disjoint% are 100/0 *by arithmetic* - holds only while
every expert's softmax share is strictly positive. Were any of them to
underflow to 0.0, the support would shrink, the row's "selected set" would stop
being the whole bank, and the identical/disjoint numbers would be measuring an
underflow instead of the routing. `mean_support == n_experts` on the dense
path is the assertion that keeps the arithmetic argument honest, and
`moe_step1_gate.rs::the_dense_mixture_has_full_support_so_its_set_is_the_whole_bank`
holds it.

**RESULT: see §6 — measured on this box, this commit.**

---

## 5. The load-balance coefficient — swept, not copied

The review's finding is that **no established coefficient exists at ≤100M**, so
copying a large-MoE value (Switch's 0.01, GShard's 0.1) would be importing a
number from a regime where it was tuned against a token population we do not
have. The coefficient is therefore swept on a **hostile toy batch** and the
result reported, not chosen by citation.

**Form** (Switch/GShard, the standard one): `aux = E · Σ_e f_e · P_e` where
`f_e` is the fraction of tokens whose top-k includes expert `e` and `P_e` the
mean gate for `e`. This is computable with device ops only — `f_e` is
`mask[:, e].mean()`, `P_e` is `gates[:, e].mean()` — so it adds no host-device
synchronization (§1.3). It is returned as a tensor and added to the loss; it is
never read on the host in the hot path.

**The two properties this form has, which the gates pin:**

1. it is **minimized at exactly 1** (uniform routing): `E · Σ (1/E)(1/E) = 1`.
2. a **collapsed** router (every token → expert 0) scores exactly `E`.

So the term's whole range is `[1, E]` = `[1, 4]` here. That range is what makes
the coefficient sweep meaningful, and it is why the coefficient question is
really "what fraction of the task gradient should the balancer be allowed to
apply" rather than "what number does the literature use".

**Hostile toy batch**: one deliberately built so the router wants to collapse
(overwhelmingly skewed logits), because a balanced batch cannot distinguish a
coefficient that balances from one that is too weak to matter.

**RESULT: see §6.**

---

## 6. RESULTS

Two measurements, both taken on this box at this commit, CPU (burn-flex)
backend, reproducible by running the named gate. **Nothing here is verified
against an external reference and no external reference exists.**

### 6.1 STEP 1 — the routing divergence of the EXISTING mixture

`cargo test -p dormouse-core --test moe_step1_gate -- --nocapture`, 4 experts,
depth 4, 2x32 = 64 positions, 8 model draws per arm. The two arms are separate
constructions (the seeded stream is not rewindable — AGENTS.md 3.7), so each
number is a mean over draws, and the gate's assertion is a margin seed noise
cannot produce.

| metric | DENSE (control) | ROUTED k=1 |
|---|---:|---:|
| cross-pass top-1 agreement | **0.9766** | 0.9590 |
| cross-pass weight cosine | **0.9999** | 0.9590 |
| identical% (paper's anchor) | 1.0000 | 0.9590 |
| disjoint% (paper's anchor) | 0.0000 | 0.0410 |
| mean support (experts/token) | 4.0000 | 1.0000 |
| load share per expert | 0.265 / 0.237 / 0.255 / 0.243 | 0.156 / 0.387 / 0.293 / 0.164 |
| **top-1 flips when the INPUT changes** | **0.7285** | 0.7207 |

Chance rate for a 4-way top-1 is 0.2500.

**These are means over 8 draws, and the run-to-run spread is part of the
number, not noise to be hidden.** Four consecutive runs of the gate gave
cross-pass top-1 agreement 0.9766 / 0.9805 / 0.9766 / 0.9766 (dense) and
0.9590 / 0.9531 / 0.9609 / 0.9648 (routed), with disjoint% 0.0000 (dense,
every time) and 0.0410 / 0.0469 / 0.0391 / 0.0352 (routed). Quote the dense
agreement as **~0.977-0.981** and the routed as **~0.953-0.965**: the ORDERING
is stable across every run (dense agreement is always the higher of the two,
and dense disjoint% is always exactly 0), while the individual digits are a
property of the draws. A single-run digit quoted to four places would be a
number this fixture cannot support.

**THE HEADLINE, and it is two numbers, not one.** The existing mixture is
**input-dependent** (0.7285 flips against a 0.25 chance rate: the expert
weights genuinely move when the text moves) and **pass-INDEPENDENT** (0.9766
cross-pass agreement — the same token reaches for the same expert on the next
pass 98% of the time). The loop's weight-tied FFN is therefore running, on
almost every token, **the same expert on every pass**. That is precisely the
regime arXiv 2605.09165 identifies as the expressiveness bottleneck of a
looped model, and it is the number that justifies the lane: a mixture that
averages four experts but picks the same one every pass has the parameter
count of one expert and none of the specialization.

**Why `disjoint%` is 0.0000 on the dense path is arithmetic, not measurement**,
and the instrument checks that it still is: every expert's softmax share is
strictly positive, so a token's "selected set" is the whole bank on both
passes, so identical% = 1 and disjoint% = 0. `mean_support = 4.0000` (asserted
to be `n_experts`) is what keeps that argument honest — if any share had
underflowed, the support would shrink and the pair would be measuring an
underflow instead of the routing.

**The anchors do not transfer, and this is the second finding.** The paper
reports 0.04–0.14 identical and 0.25–0.53 disjoint at k=2 of E=8. Our routed
k=1 reads 0.959 identical / 0.041 disjoint — an order of magnitude the wrong
way. Stated rather than tuned to; a different model at a different scale is not
expected to reproduce it, and the gap is a reason to distrust the transplant,
not a knob.

**AND THE FINDING THAT CHANGES THE ARM'S DESIGN.** Top-1 routing moved the
*support* from 4 experts per token to 1 — and moved **which** expert wins by
almost nothing (0.9766 → 0.9590). The reason is structural, not a
hyperparameter: **softmax is monotone, so the top-1 of the logits is the argmax
of the full softmax.** A top-k mask over the same logits therefore cannot
change the winner at k=1; it only zeroes the losers. "Does routing diverge
across passes?" is consequently a property of the **controller's logits**, and
the 0.9766 above is the measurement of it — the router adds nothing to that
number. What the mask buys at k=1 is a sparser mixture at identical scale, not
divergence.

Two consequences, both of which change what the lane should do next, and
neither of which is "tune k":

1. **Divergence has to be bought where the paper buys it** — by the
   selection *feeding back into the computation*. Masking cannot do it, because
   the argmax is unchanged. A real dispatch (compute only the selected experts,
   so the unselected ones contribute no state) changes the pass-to-pass
   trajectory, which changes the next pass's logits. That is the gather/scatter
   the `ponytail:` note in `moe.rs` defers, and this measurement is the reason
   it is not optional.
2. **The arm as landed is still worth its A/B**, but for the reason measured
   here and not for the one in the abstract: does a token that consults ONE
   expert per pass, chosen by that pass's state, beat one that averages four?
   That is a real question with the dense mixture as the control.

### 6.1b Gradient flow and tie behaviour — the gates that were MISSING

Two of the brief's gates had no test, and both are now in
`crates/dormouse-core/tests/moe_grad_seam.rs`.

**Gradient flow** (the `8fa5d4c` class, and the reason it is worth a file: the
routing arm adds NO parameters, so "the router trains" has to be checked on the
controller's EXISTING expert columns, next to a control that must also be live).
Measured on the ndarray backend, `k = n_experts` so every expert is selected on
every row — L2 norms of the parameter gradients:

| quantity | L2 |
|---|---:|
| controller arm-gate columns `[0..3]` (the control) | 3.366e-02 |
| controller **router** columns `[3..3+E]` | 5.198e-02 |
| per-expert `gate_up` weight | 0.037 / 0.039 / 0.040 / 0.036 |

At `k = 1` all four experts still received gradient on the fixture draw
(32 positions over 4 experts covers the whole bank), so the k-dependent claim is
the converse — an expert that wins no position is masked to exactly zero, by
construction (`d/d out_e = gate_e = 0`), not by defect. Asserting an exact set
would be asserting the RNG.

**Tie behaviour, defined.** The selection is `dormouse_mor::topk_indices` — one
`argsort_descending` + `narrow` — and that primitive's own doc says the selected
SET among equal values may differ between impls and backends. So the promise is
deliberately narrow and is exactly what is tested: **the COUNT is always exactly
`k`, never `k±1`, including on a row where all `E` scores are equal**; WHICH
tied expert wins is unspecified and nothing asserts it. A `k±1` leak would be
silent and total — a row selecting 0 experts gets a zero FFN output and one
selecting all `E` is the dense blend, so both are the control wearing the arm's
label.

### 6.2 The load-balancing coefficient — swept, and the sweep says NO

`cargo test -p dormouse-core --lib moe -- --nocapture`. Hostile batch: 64 rows,
4 experts, top-1, 40 steps, task gradient deliberately unopposed (every row
wants expert 0). The control is the same simulation with no balancer.

| quantity | value |
|---|---:|
| `\|\|grad L_LB\|\|` at coef 1 | 0.039622 |
| `\|\|grad task\|\|` | 8.717798 |
| ratio per unit coef | 0.004545 |
| **coef_crit (balancer matches the task)** | **220.02** |

| coef | load spread | share on expert 0 |
|---|---:|---:|
| off | 1.0000 | 1.0000 |
| 0.01 (Switch's) | 1.0000 | 1.0000 |
| 0.1 (GShard's) | 1.0000 | 1.0000 |
| 2 x coef_crit | 0.1719 | 0.3594 |

**Switch's 0.01 and GShard's 0.1 do nothing at all here**: the router ends
100% on one expert, exactly as it does with no balancer, and the term's
gradient is ~220x too small to matter. The cause is structural and worth
stating because it generalises: Switch's `P_i` is a **mean over tokens**, so
the balancer's gradient per token shrinks as the batch grows, while the task
gradient per token does not. The coefficient that matches the two therefore
**scales with the token count** — roughly linearly, from 220 at 64 tokens to
~1.4e4 at the trainer's ~4096-token batch — and at that size the term's VALUE
(`coef x L_LB`, with `L_LB` in `[1, E]`) would be ~1e4 against a CE of ~5.5,
i.e. it would replace the objective.

**So no coefficient is shipped, and `moe_lb_coef` stays 0.0 with the arm off.**
A default here would be a number copied from a regime we are not in, and the
honest form of the setting is that the Switch load-balancing loss **does not
transfer to our scale in its standard form**. Fixing the scaling properly — e.g.
making the term's normalization token-count-invariant so a constant coefficient
transfers — is the named follow-up, and until it lands the arm is off by default
and its A/B row has to carry a swept coefficient.

**What `validate` does about it, and why it is not a refusal.** It refuses
`moe_lb_coef > 0` with `moe_topk = 0` — a term with no selection to balance is
the `aux_fb_weight`-without-a-head shape, and this repo has already been bitten
by one (`probe::JEPA` was bumped nowhere, so the field read like an objective
while contributing nothing). It does **NOT** require a non-zero coefficient
with the arm on, and that is deliberate:

1. routing with no balancer is the arm's **own removal** under the A/B rule
   (AGENTS.md §1.2) — an arm that cannot be run without its balancer cannot be
   measured against one;
2. a refusal would force the operator to pass *some* coefficient, and the sweep
   above measures that every published value is ~3 orders of magnitude too weak
   here — manufacturing exactly the defect the review named;
3. the collapse risk is already **visible without a refusal**: `probe::MOE_ROUTE`
   counts selections and `probe::MOE_LB` counts balancer terms added, so the
   eval line's `moe=<lb>/<sel>` reads `0/<n>` for a routed run with no balancer.
   That is the repo's own COUNTED mark — the right instrument for "a degradation
   happened and a reader can tell", not a refusal of a legal run.

### 6.3 THE FORMULATION DEFECT the sweep found — the most important line here

The first implementation passed `lb_aux` the **renormalized gates**, which is the
obvious thing to pass and is **exactly wrong at k=1**. A top-1 token's
renormalized gate is exactly `1.0` (a softmax over one surviving term), so `P_e`
is piecewise *constant* in the logits and the term's gradient is **identically
zero wherever the selection does not change** — measured `‖grad L_LB‖ = 0`.

A term with zero gradient is decorative. It trains for thousands of steps,
contributes nothing, and the loss curve stays perfectly healthy: the ADR-0019
shape, verbatim. Switch's `P_i` is the router probability **before** top-k for
exactly this reason; `topk_blend` now returns that `probs` tensor separately and
`lb_aux` takes it. **The sweep's first assertion is now `‖grad L_LB‖ > 0`**,
and that assertion is the one that would have caught it — the sweep's *value*
did not, the sweep's *gradient check* did.

### 6.4 Run-to-run spread on §6.1, stated because it is not small

§6.1's numbers are means over 8 model draws, and a second run of the same gate
on the same commit gave cross-pass top-1 agreement 0.9707 (dense) / 0.9727
(routed) and disjoint 0.0273, against 0.9766 / 0.9590 and 0.0410 above. So the
seed spread on `disjoint%` at 64 positions is roughly 0.027–0.041. The
conclusions survive it comfortably — the routed `disjoint%` is an order of
magnitude below the paper's 0.25–0.53 band either way, and the dense endpoint
is arithmetic — but a 3–4 % statistic measured on 64 positions with 8 draws
should not be quoted to three digits.

### 6.5 Two agents were in this worktree

`wt/moe-routing` had **two** agents on this lane at once (this one and a second
dispatch of the same brief), interleaving writes into the same files
minute-by-minute — `probe.rs`, `config/override.rs`, `moe.rs`,
`tests/moe_step1_gate.rs`. AGENTS.md §1.6 is explicit that a shared checkout
with N writers makes `cargo test` a lottery and `file:line` citations
meaningless, and it did: one `cargo test --lib` run failed a test that passes in
isolation and in every run since, on a tree that was still moving. The tree was
quiesced before the numbers above were taken and all of them are from the stable
tree.

Where the two designs disagreed — whether `validate` should *require* a balancer
— the disagreement was resolved in favour of the measurement (§6.2), which is
the rule AGENTS.md §3.2 would want applied to this file rather than to someone
else's code. Two assertions written by the second agent were also wrong and are
corrected here: a cross-arm top-1 equality that cannot hold (§6.1's monotone
argument — it is a same-model fact, and the gate that tried to assert it could
only have been asserting seed spread), and `lb_aux == E` exactly for a collapsed
router, which holds only in the one-hot limit
(`lb_aux_floor_is_exactly_one_and_the_ceiling_is_e_in_the_limit`).

### 6.6 Two measurement bugs and one fixture bug in the sweep

Recorded because a sweep that cannot fail is worse than no sweep. All three were
found by an assertion firing, not by reading the code:

- **`lo` seeded with `f32::NEG_INFINITY`**, so `f32::min` returned `-inf` for any
  non-empty set, the spread was `hi - (-inf) = inf`, and the whole sweep
  compared `inf` to `inf`. Every coefficient "passed".
- **The per-expert token count ignored its expert index**, so all four experts
  received the same share, the spread was identically zero, and the "collapse"
  reading was an artifact of the counter. It now chunks by row and reads column
  `j`.
- **The fixture's rows were all identical, from a precedence accident.**
  Written `if j == 0 { 3.0 } else { 0.0 } + jitter(r, j)`, Rust binds `+ jitter`
  to the **else branch only**, so every row got the same expert-0 logit and the
  batch was 64 identical copies. The selection is a per-row argmax, so 64
  identical rows can only ever all pick the same expert and **no coefficient
  could spread that load** — the sweep was measuring a batch with no tokens to
  distribute. It surfaced as "the balancer's gradient is zero", which was true
  of the batch and not of the term. The parenthesised version plus a bounded
  jitter is the fixture now: expert 0 leads by 0.30, less than the row-to-row
  spread, so most rows want expert 0 and some do not.

### 6.7 The gates, and what each would take to fool it

| gate | what it would take to fool it |
|---|---|
| `off_is_the_dense_mixture_and_allocates_nothing` | three ways: `num_params` **equal** at `moe_topk` 0 and 1, `MOE_ROUTE == 0`, and the blend has FULL SUPPORT with unit row sums. Stated limit: it does **not** prove bit-identity against a pre-change build — it proves the arm is off, allocates nothing, and produces the dense signature. |
| `routed_support_is_exactly_one_and_the_counter_follows` | support exactly 1, the live gate exactly 1.0, and the counter equal to the executed depth. |
| `the_router_receives_the_pass_index` | a mutation, not a value: at depth 1 the only pass signal in play is `iter_embed[0]`, with weights and input fixed. If the mixture does not move when that row moves, the router provably ignores which pass it is on. This gate is what justifies **not** adding a pass-embedding to the router input. |
| `the_balancer_reaches_the_loss_only_with_a_selection` | the term is `None` at coefficient 0 and a finite **positive** addition above it, bounded by `coef · E`. |
| `topk_selects_exactly_the_host_top_k` | a host-side argsort on a **no-tie** fixture; the code is never compared to itself. |
| `top1_gate_is_exactly_one_on_the_winner_and_zero_elsewhere` | pins the renormalization that makes the A/B row a comparison of specialization rather than of output scale. |
| `lb_aux_floor_is_exactly_one_and_the_ceiling_is_e_in_the_limit` | the floor is **attained**; the ceiling is only **approached** — a merely-confident router still leaks probability to the experts it rejected, so `collapsed == E` exactly is false at finite confidence. An earlier version asserted it and would have passed for the wrong reason. |
| `lb_aux_rises_as_the_router_concentrates` | no cliff: a term with one would make the coefficient a coin flip rather than a dial. |
| `topk_refuses_a_set_that_cannot_fill` | `k ∈ {0, 5, 9}` all panic; `k ∈ {1, 4}` do not. |
| `moe_topk_that_cannot_fill_is_refused_and_the_defaults_are_off` | defaults are `(0, 0.0)`; `topk > bank` refused **naming the DENSE blend it would silently run**; `lb > 0` with routing off refused; the four legal positions accepted. |
| `the_dense_mixture_has_full_support_so_its_set_is_the_whole_bank` / `anchor_pair_reads_zero_and_one_at_both_ends` | the anchor statistics can be read at both 0 % and 100 %, so a reported 2.7–4.1 % is a measurement and not a floor artifact. |

## 7. What is NOT claimed

- **No A/B number exists for this arm.** The row is registered, not run: the GPU
  was occupied by a live run for the whole of this lane (§1.5, one heavy thing
  at a time), and §3.3 of AGENTS.md already records that the per-arm cost is
  unknown.
- **The arm is NOT a speedup as implemented.** Every expert is still computed and
  then masked, so the executed FLOPs are 4× the control. What the arm buys at
  this stage is *specialization at equal active parameters*, and the A/B row is
  written to say so. Harvesting the FLOP saving needs a gather/scatter dispatch,
  which is a separate step and is not built here.
- The Sparse-Layers anchors are from a different model at a different scale.
  They are a **comparison target**, not a like-for-like reproduction.