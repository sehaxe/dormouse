# AttnRes gate falsification — the gate can go red, and did

**Date:** 2026-09-30, branch `wt/attnres-model`. The lane brief required a gate
that goes red on a perturbation, with the magnitude recorded, then green on
restore. This is that record. Every number below is the output of the command
shown; none is a description of what the command would do.

The point of this file is that **three of these gates were green before this
lane and could not have failed** — the audit in
`docs/papers/attnres.md` §3.1 established that every reference in the
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

## Gate 6 — a derivative the gates were green through (found by the CUDA run)

`fused_attnres::fd_tests::depth_attend_grad_matches_finite_difference` and
`fd_tests::fused_backward_matches_burn_autodiff`, run with `--features
cuda,autodiff`.

**This is the finding of the lane.** Both were **green on the pre-change
code** (measured: checked out `a7c3cad`'s two files and ran them — 2 passed)
and **red the moment `ScoreForm::Paper` was wired in**:

```
layer 0 idx 0: analytic 0.3739026 vs fd -0.7981062 (rel 1.4684854)
fused vs burn raw autodiff worst=1.1717439
```

**The defect.** The score is `s = scale·(q·h)·(m‖h‖²+ε)^(-1/2)`, and
`∂s/∂h_j = scale·(q_j·inv − m·h_j·(q·h)·inv³)`. **The `m` was missing** from
the second term in *both* the fused kernel (`fused_attnres.rs:454`) and the
tensor reference (`:1447`). With the L2 norm `m = 1` the term vanishes, which
is why it was written without it and why every test in the crate stayed green
for the crate's whole life. The paper's RMSNorm has `m = 1/d`, and the
difference is not a rounding matter — it is a wrong derivative of the norm,
in the arm's own backward.

**Fixed in both places** (the fused kernel and the tensor reference — they
were consistently wrong together, which is exactly the "two transcriptions of
the same error" shape the audit warned about), then:

| state | result |
|---|---|
| fixed | **green** — `fused_backward_matches_burn_autodiff` now passes, i.e. the fused adjoint agrees with **burn's own autodiff** of the forward, an independently computed derivative |
| **re-perturb**: `m * hl * qh[li] * inv3` → `hl * qh[li] * inv3` (the pre-lane formula) | **RED again**, `diff 1.172e0` against an `fd noise floor` of `3.935e-4` — **3000× the floor**, `rel 1.47` |
| restore | **green** |

So the gate has teeth, and the defect it found was real, silent, and would
have trained the arm on a wrong gradient.

**The tolerance argument behind the fix, and it is a measurement.** The
per-component relative check could not see a component of size 3e-4, because
the central difference quotient's own f32 noise at `eps = 1e-4` on an
O(10) loss is ~5e-3 — larger than the thing being measured. The audit's Test F
prescription was already right: `eps = 1e-3`, and a noise floor computed from
the loss scale (`8·ε_mach·|loss| / 2·eps`) instead of a guessed constant.
The gate then carries **both**: per component with the floor added, and an
aggregate `‖analytic − fd‖ / ‖fd‖ < 1e-2` over the whole gradient, which is
the tolerance-meaningful statement. The re-perturbation above is red on both.

## Gate 7 — a tolerance that was a coin flip (found by the same run)

`fused_attnres::tests::merge_state_writeback_matches_host_reference`

**3 red out of 4 runs on identical code**, at `acc` maxdiff 1.0–1.7e-5
against a **1e-5 absolute** bound. Instrumented: `|acc| ~ 3–4` in the failing
runs, so the relative error is 2.5e-6 ≈ 21× f32 epsilon after 64 chained
merges. That is the arithmetic of accumulating 64 f32 updates, not a kernel
error — and the cause is the score convention again: the paper's unscaled
logit is ~`√d` times larger, the online-softmax weights are sharper, and the
accumulator lands at a scale where a scale-blind bound sits inside the noise
floor. The data is also **unseeded** (ADR-0021), so which runs went red was
a lottery.

**Re-based on a scale-relative bound** (`1e-5 · max(1, |want|∞)`) for the
three accumulators, with the reason in the assert message. **5 consecutive
green runs** after the change. The defect this cell exists for — the
missing-barrier race, whose signature is `rescale = 1` and therefore O(1) —
is four orders of magnitude above the new bound, so nothing was softened
away.

**What this is:** a gate that reported a defect the code did not have, four
times in a row, for a week of history it could not have. Same class as the
1/√d references, from the other direction: not a gate that cannot fail, but a
gate that fails for a reason unrelated to what it names. The audit's own note
on this cell ("this test passes with the barrier deleted") already said its
coverage was not its gate; this is the second half of that sentence.

## What was NOT falsified, and stays unproven

- **The CUDA fused path under the trainer's backend, end to end.**
  `burn-attnres --features cuda,autodiff` is **19 passed / 0 failed** with
  the paper's form as the default, including both parity tests over BOTH
  forms, the balanced-checkpointing seam test
  (`balanced_checkpointing_reaches_the_seam_and_the_legacy_entry_does_not`,
  the assertion that dormouse's backend gets *past* the seam downcasts), and
  the fused-adjoint-vs-burn-autodiff comparison. `dormouse-core --features
  cuda --lib` is **59 passed**. What is still not measured: an AttnRes
  *training step* on this card, i.e. that the counter in a real `train_loop`
  reads non-zero and the loss descends. The A/B row is the place that gets
  measured, and it is not run.
- **`BlockAttnRes`.** Untouched, still wrong against Eq. 6, deliberately not
  wired. See the integration doc §5 for the four fixes and the gate each
  needs.
