# AttnRes gate falsification — the gate can go red, and did

**Date:** 2026-09-30, branch `wt/attnres-model`. The lane brief required a gate
that goes red on a perturbation, with the magnitude recorded, then green on
restore. This is that record. Every number below is the output of the command
shown; none is a description of what the command would do.

The point of this file is that **three of these gates were green before this
lane and could not have failed** — the audit in
`research/papers/attnres.md` §3.1 established that every reference in the
fused path re-derived our own scaled formula, so "kernel matches tensor" was a
consistency check between two transcriptions of the same wrong thing. Tonight's
lesson, applied: a gate that cannot fail is not evidence.

---

## Gate 1 — the score convention (the lane's open question)

`burn-attnres/src/lib.rs::tests::paper_form_has_no_temperature_and_this_is_pinned`

Fixture: `B=T=1`, `L=2`, `d=4`, `h_0 = e_0`, `h_1 = e_1`, `w = 2·e_0`.
Expected `out[0]`: **0.9820138** (paper), 0.7310586 (the old `SqrtD` form).

| state | result |
|---|---|
| as written | **green** — `out[0] = 0.9820138`, and the `SqrtD` leg reads 0.7310586 |
| **perturb**: the model's `AttnRes::new` → `AttnRes::with_form(d, dev, ScoreForm::SqrtD)` | **RED**, `Eq. 2 says out[0] = 0.9820138, got 0.731058` — a gap of **0.251**, 25 000× the 1e-5 tolerance |
| restore | **green** again, same two numbers |

That gap is the `1/√d` question answered as a number rather than an opinion:
the two conventions differ by 0.25 on a `[1,1,4]` tensor, and no tolerance
argument is anywhere near it.

## Gate 2 — the model readout must move

`loop_block.rs::tests::attnres_moves_the_readout_at_every_depth`, and the
`preset_exec.rs` `use_attnres` block (which asserts the counter is
`max_iter` **and** `max |Δlogit| > 1e-6`).

| state | result |
|---|---|
| as written | **green** at depths 1, 2, 3 |
| **perturb**: `res.push(y.clone())` → the AttnRes branch computes `h = h_ctx.clone()` (i.e. the arm runs, counts, and its result is thrown away) | **RED**, `AttnRes changed no logits` — max `|Δlogit|` = **0.0** against a 1e-6 floor |

This is the `9b343d3` GR defect reproduced on purpose: a counter that fires
while the output does not move is the exact shape of "the mechanism is dead
and the loss curve looks fine". The count assertion alone would have stayed
**green** under this perturbation — only the logit assertion catches it.

## Gate 3 — the aggregation is a real softmax mixture, not an average

`burn-attnres/src/lib.rs::tests::zero_query_is_an_equal_weight_average_at_init`
plus the recovered-mixture check below.

At `w_l = 0` every score is 0, so **both** conventions give a uniform
average — this gate is scale-blind by construction and says so. It pins the
thing that is *not* scale-blind: that the **values** are aggregated
un-normalised. An implementation that weighted by `RMSNorm(h)` would return
the mean of normalised states and fail by O(1).

| state | result |
|---|---|
| as written | **green**, max deviation from the mean `< 1e-6` for both forms |
| **perturb**: the aggregation weights applied to `h_norm` instead of `h_stack` | **RED**, off by **0.4** on unit-scale states — a full `O(1)` miss, not a rounding one |

## Gate 4 — the counter cannot be satisfied by an inert arm

`loop_block.rs::tests::attnres_counts_every_iteration_it_aggregates`

| state | result |
|---|---|
| as written | **green**: 4 at depth 4, **0** on the ReZero path, **2** under `set_depth(Some(2))` |
| **perturb**: `note(ATTNRES)` moved to the top of the loop body, outside the `use_attnres` branch | **RED** at the `0` assertion — the ReZero path would have reported 4 aggregations it never performed |

## Gate 5 — the arms are mutually exclusive

`loop_block.rs::tests::attnres_and_gr_are_refused_together`

`use_attnres = true` with `use_gr = true` is refused in `config::validate`
**and** re-asserted at the branch (`assert!`), because a hand-built
`LoopBlock` bypasses the config path. Both halves named in the error message,
so a reader knows which two fields to choose between.

---

## What was NOT falsified, and stays unproven

- **The CUDA fused path under the trainer's backend.** `burn-attnres` with
  `--features cuda` compiles and its parity tests run in both score forms
  (see the test tallies in the report), but the run was done with a training
  job on the card, per §1.5. `fused_attnres` is one of the tree's working
  kernels and the form is a runtime scalar in all three of them, so the
  dispatch is unchanged in shape — but **"the fused path is reached under
  `Autodiff<Cuda, BalancedCheckpointing>` with the paper's form" is a claim
  this lane did not measure on a quiet card.** The `seam_counts` readout is
  what settles it in one line on the next free GPU.
- **`BlockAttnRes`.** Untouched, still wrong against Eq. 6, deliberately not
  wired. See the integration doc §5 for the four fixes and the gate each
  needs.
