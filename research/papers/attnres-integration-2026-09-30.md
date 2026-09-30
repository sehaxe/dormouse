# AttnRes in the model — formula verification and integration record

**Lane:** wire `burn-attnres` (arXiv:2603.15031) into `dormouse-core`'s loop.
**Date:** 2026-09-30. **Branch:** `wt/attnres-model`, off `a7c3cad`.
**Verdict up front:** the mechanism is wired behind `use_attnres` (off by
default), the score convention is settled to the **paper's** form, and the
honest status of every claim below is **tier-(b) transcription** — no author
code exists, so nothing here is "verified" in the ADR-0014 sense. The Block
variant is **not** wired: it is still wrong against Eq. 6, and wiring it would
have put a known-bad function in the model.

---

## 1. Provenance, and what "no external reference" means here

| item | value |
|---|---|
| paper | *Attention Residuals*, Kimi Team (Moonshot AI), arXiv:2603.15031 (`cs.CL`), v1 2026-03-16 |
| PDF read | `Attention_Residuals.pdf` from `MoonshotAI/Attention-Residuals` @ `85e2231`, **md5 `f8351f26bce4c33b2880dee3d101f4b0`**, 952 700 B, read with `pdftotext -layout` into 1436 lines |
| author code | **none.** The repository is 6 files: this PDF, `README.md`, 4 PNGs. Measured 2026-09-30 via the GitHub trees API; re-confirmed by the README copy pinned in the crate at `tests/oracle/upstream/attnres_README.md` |
| the only executable spec | Fig. 2, 22 lines of PyTorch, identical in the paper and the README |
| our crate before this lane | `vendor/burn-fused/crates/burn-attnres/` — `src/lib.rs` 517, `src/fused_attnres.rs` 1821 |
| the prior audit | `research/papers/attnres.md` (2026-09-29) — 19 deltas, 9 of them BUG |

**Therefore: tier-(b), transcription.** Every "the paper says" below quotes
the PDF named above. No line of this document may be read as verification
against an implementation, because there is none. The word "verified" is not
used about the mechanism anywhere in this lane's code.

---

## 2. The mechanism, equation by equation, mapped to `file:line`

Left column: the paper. Right: the code that runs it, after this lane.

| # | paper | our code | status |
|---|---|---|---|
| **Eq. 1** | `h_l = α_{0→l}·h_1 + Σ_{i=1}^{l-1} α_{i→l}·f_i(h_i)`, `Σ α = 1` | `loop_block.rs` AttnRes branch: sources are `[h0, y_0 … y_iter]`, output is their softmax mixture | **transcribed**; placement is ours, see §4 |
| **Eq. 2** | `α_{i→l} = ϕ(q_l,k_i) / Σ_j ϕ(q_l,k_j)`, `ϕ(q,k) = exp qᵀ RMSNorm(k)` | `burn-attnres/src/lib.rs` `depth_attend_form` (score + `softmax(scores, 0)` over the source axis) | **transcribed, and the score convention is now the paper's** — it was not, see §3 |
| **Eq. 3** | `q_l = w_l`; `k_i = v_i = h_1` if `i=0`, `f_i(h_i)` if `1≤i≤l−1` | `slot_query` (per-slot `w_l`), `res = vec![h0]` then one push per iteration | **transcribed** |
| **Eq. 4** | `h_l = Σ_{i=0}^{l-1} α_{i→l}·v_i` | `depth_attend_form` weighted sum over the source axis | **transcribed** |
| **§5** | "one RMSNorm and one pseudo-query vector `w_l ∈ R^d` per layer" | `attnres: Option<Vec<AttnRes>>`, one `AttnRes` per iteration slot, `query: Param<Tensor<1>>` of length `d_model` | **transcribed**; the paper's *learned* RMSNorm gain is not implemented — see §5 |
| **§5** | "all pseudo-query vectors must be initialized to zero" | `AttnRes::new` uses `Initializer::Zeros`; `loop_block.rs` test `attnres_at_init_is_a_uniform_average_of_its_sources` | **transcribed, gated** |
| **Table 5 fn 2** | "softmax **jointly** normalized over all sources" | one `softmax` over the stacked source axis | **match** |
| **Fig. 2 L10/L12** | `K = norm(V)`, `h = einsum(α, V)` — keys normalised, **values raw** | scores from the normalised stack, aggregation over the un-normalised one | **match** |
| **Eq. 5/6, Alg. 1** | Block AttnRes: `b_n = Σ_{j∈B_n} f_j(h_j)`, `b_0 = h_1` a separate permanent source | `BlockAttnRes` — **still wrong** (D5–D9 of the prior audit) | **NOT WIRED**, deliberately; §5 |

### 2.1 What the model arm does *not* claim

- **The Block variant.** `BlockAttnRes` remains in the crate, still carrying
  the four defects the 2026-09-29 audit read out of it: the embedding folded
  into block 1 (D5), the first sublayer of a block returning the previous
  block unchanged (D6), the un-normalised sum on the second sublayer of the
  first block (D7), and the double-merge at a boundary (D8). Those are
  properties of `BlockAttnRes::step` and are **unchanged by this lane**. The
  model does not call it. Wiring it would have meant shipping a function
  known to differ from Eq. 6 as a "faithful port" — which is how the
  `9b343d3` GR defect happened.
- **Input-dependent query** (§5.3's best variant, 1.731 vs 1.737): not
  implemented. `AttnRes.query` is a free vector, not a projection of the
  hidden state. The paper declines it on cost grounds; we decline it because
  it is not in the crate at all.
- **Multihead depth aggregation** (§5.3, `H = 16`): not implemented. The paper
  reports it **hurts** (1.752 vs 1.746), so its absence is a deliberate
  omission and not a gap.

---

## 3. The `1/sqrt(d)` question — settled by measurement, not by taste

**The question.** Our crate computed `q·h / sqrt(Σh² + 1e-5) * d^-0.5`. The
paper's Eq. 2 has no temperature, and its RMSNorm has a `1/d` inside the
root. Two independent deviations.

**What the PDF says** (three independent places, and a grep that finds
nothing else):

- §3.1: *"we adopt `ϕ(q, k) = exp qᵀ RMSNorm(k)` [66] with normalization,
  yielding softmax attention over depth"*.
- Table 5, footnote 2: *"`ϕ(q, k) = exp qᵀ RMSNorm(k)`; `k_i = v_i`;
  `v_0 = h_1`, `v_i≥1 = f_i(h_i)`. softmax jointly normalized over all
  sources."*
- Fig. 2 line 11: `logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)`
  — a bare einsum over `K = norm(V)`.
- Reference [66] is *Biao Zhang and Rico Sennrich, "Root mean square layer
  normalization", NeurIPS 32 (2019)* — so the norm is the **mean** form
  `x / sqrt(mean(x²) + ε)`; the `1/d` is what makes it an RMS norm rather than
  an L2 one.
- `grep -in "sqrt|scale|temperature"` over all 1436 extracted lines returns no
  scaling factor anywhere in the mechanism. The only hit is *"attention
  temperature rescaling"* (§5.2), about MLA/NoPE context extension — a
  different method entirely.

**The measurement.** Three conventions, one fixture chosen so they cannot be
confused: `B=T=1`, `L=2`, `d=4`, `h_0 = e_0`, `h_1 = e_1`, `w = 2·e_0`.

| convention | score_0 | α_0 | out[0] |
|---|---|---|---|
| **paper** — `q·RMSNorm(k)`, no temperature | 4.000002 | 0.9820138 | **0.9820138** |
| our original — scale kept, L2 norm | 0.999995 | 0.7310586 | 0.7310586 |
| scale removed, L2 norm kept | 1.999990 | 0.8807971 | 0.8807971 |

`4.000002` rather than `4.0` is the `ε = 1e-5` inside the root:
`1/sqrt(1/4 + 1e-5) · 2 = 4.000002`. The three answers are separated by
**0.10 and 0.15** — four orders of magnitude above a `1e-5` tolerance, so the
assertion has no tolerance argument in it. Pinned as literals in
`burn-attnres/src/lib.rs::tests::paper_form_has_no_temperature_and_this_is_pinned`;
it is the crate's own gate and it is **red** if either convention changes.

**The decision: `ScoreForm::Paper` is the default, and the model arm uses
it.** Both halves move together, because they are the same defect: with a
free learnable `w_l`, a `d`-fold temperature is absorbable by rescaling `w_l`,
so the deviation does not change what the model can *express* — it changes
the *optimisation trajectory* and the effective softmax temperature, and at
the mandated `w_l = 0` init both give uniform α, so init is unaffected. In
exchange for a faithful port we get a **d-fold** difference in how fast α
sharpens during training, and the paper's own ablation table is a loss table,
not a temperature table, so nothing in it transfers. The asymmetry decides it:
a wrong-by-`d` temperature is a confound in every future A/B; the cost of the
paper's form is one scalar in a kernel.

**Both routes are expressible, and both are tested.**
`ScoreForm::{Paper, SqrtD}` is a runtime value threaded from the module
through `depth_attend_form` → all three CUDA dispatches → all three kernels
(added as `norm_m: f32` beside the existing `scale: f32`) → and the autodiff
node's state, so the **backward** differentiates the same form the forward
computed. `depth_attend_fused_matches_tensor` and
`source_score_fused_matches_tensor` now loop over **both** forms against a
reference that takes the form as an argument, so a kernel that ignored
`norm_m` or `scale` is red rather than consistently wrong.

**Also fixed here, because the form is one decision:** `two_phase_attend`
applied the scale to Phase 1 only (prior audit D4), comparing the inter-block
and intra-block source groups at temperatures differing by `√d`. Both legs
carry the form now. The existing `two_phase_merge_matches_full_attention`
could not see it: it exercises `i = 0`, which bypasses the merge.

**What making the form paper-faithful FOUND, and this is the important part.**
The CUDA run of this lane turned two previously-green tests red, and the cause
was a defect that had been in the crate since it was written:

> The score is `s = scale·(q·h)·(m‖h‖²+ε)^(-1/2)`, so
> `∂s/∂h_j = scale·(q_j·inv − m·h_j·(q·h)·inv³)`. **The `m` was missing** —
> in the fused kernel (`fused_attnres.rs:454`) *and* in the tensor reference
> (`:1447`). With the L2 norm `m = 1` the term vanishes, so the formula was
> correct for the form the crate shipped and wrong for the paper's, and
> nothing could see it. `ScoreForm::Paper` (`m = 1/d`) made
> `depth_attend_grad_matches_finite_difference` red at **rel 1.47** and
> `fused_backward_matches_burn_autodiff` red at **worst 1.17**.

Both are fixed; the fused adjoint now agrees with **burn's own autodiff** of
the forward, which is an independently computed derivative. Re-perturbing the
`m` back out reproduces the red at **diff 1.17 against a 3.9e-4 noise floor**,
so the gate is not soft. **Consequence for any A/B:** an AttnRes arm built on
this crate before this commit would have trained the residual stream on a
wrong gradient for every source except the one with `‖h‖²` small. No run in
`~/logs/` used the arm, so nothing is retracted.

A second finding from the same run, in the other direction: the crate's
`merge_state_writeback_matches_host_reference` used a **1e-5 absolute**
tolerance on 64 chained f32 merges, and went red 3 times out of 4 on
identical code once the paper's sharper softmax moved `|acc|` to ~3-4 (2.5e-6
relative ≈ 21× f32 epsilon). Re-based on a scale-relative bound, 5 green runs
in a row. A gate that reports a defect the code does not have is the mirror
image of one that cannot fail, and it was the second of the two in one hour.

---

## 4. Where AttnRes sits in the loop, and why there

`loop_block.rs`, one `if` arm in the chain that already chooses between ReZero
and GR:

```rust
let y = attn + engram_a + ffn;            // f_iter(h_iter): the block body
if use_attnres {
    res.push(y.clone());                  // v_i = f_i(h_i)
    h = depth_attend(&res, slot_query);   // Eq. 1/3/4, w_l of this slot
} else if use_gr { … } else { /* ReZero */ }
```

**Placement decision, and the two it was chosen over.** The paper computes
`h_l` — the state *entering* layer `l` — from `v_0 … v_{l−1}`, so the
current layer's own output is not in `h_l`. Taken literally in a loop whose
per-iteration readout *is* the CE, that means iteration 0's readout is the
embedding alone and the block body reaches the loss only at iteration ≥ 1: at
`max_iter = 1` the body is computed and discarded. That is the **depth-(iters−1)
model** defect, and this repo already paid for it once in Gated Residual
(`9b343d3`: the readout was taken from the state before the write).

So AttnRes is applied **after** the body, with the body's own output as a
source: the state leaving iteration `n` is the mixture over
`{h_0, y_0 … y_n}`. Three consequences, all intended:

- every iteration's readout contains that iteration's body, so the arm is
  comparable to ReZero **at every depth** including 1;
- the token embedding stays a **permanent, first-class source** — it is never
  summed into a later group, which is the D5 defect the Block path still has;
- at init (`w_l = 0`) iteration 0's state is `(h_0 + y_0)/2` and iteration 1's
  is `(h_0 + y_0 + y_1)/3` — §5's "equal-weight average at the start of
  training", which is a **magnitude change** relative to ReZero's `h + y·s`.
  The A/B therefore measures "softmax mixture with a `1/(n+1)` init scale"
  against "additive with a learned scalar", not a pure weights swap. Stated
  here because it would otherwise be a confound discovered after the run.

**The one pseudo-query per slot is a choice.** The paper's `w_l` is per layer;
our loop is weight-shared, so iteration slot ≠ layer. One query per slot
(`[max_iter][d_model]` parameters) is the faithful reading of "per layer" at
unrolled depth `max_iter`, and it is what `Vec<AttnRes>` holds. A single
shared query would be a smaller model and a different mechanism; it is not
offered, and `docs/AB-PROTOCOL.md` does not ask for it.

**No 4D tensor is materialised** (AGENTS §2.2: dynamic slicing of a 4D
autodiff tensor is `CUDA_ERROR_ILLEGAL_ADDRESS` on sm_120). The history is a
`Vec<Tensor<3>>` of at most `max_iter + 1` entries — 3 at the operating depth
of 2. `depth_attend_form`'s own internal `Tensor::cat` into `[L,b,t,d]` is the
crate's pre-existing tensor fallback and predates this lane; **on CUDA the
fused path is taken and the stack is never built** (the dispatch probe at
`lib.rs` tries the bare and both checkpointing strategies first). Flagged as a
follow-up, not fixed here: the tensor fallback on a CUDA device would build
the very tensor §2.2 forbids, and the fix belongs to the crate's own
dispatch, not to this integration.

---

## 5. The Block variant is not wired, and what it would take

Wiring `BlockAttnRes` needs four fixes, each ~5-15 lines in
`burn-attnres/src/lib.rs`, plus a gate for each (prior audit §5 tests C, G, H):

| delta | what | a gate that catches it |
|---|---|---|
| D5 | `b_0 = h_1` must be its own permanently-attended source, never summed into block 1 | test C: feed `b_0` that is not a sum of any `f`, assert the exact source set per step |
| D6 | first sublayer of a block attends `[b_0 … b_{n−1}]`, not the stored state | test C, step 3 |
| D7 | the second sublayer of the **first** block returns a softmax mixture, not `b_0 + f_1` | test C, step 2 |
| D8 | a completed block is merged into the state **once** | test G: `sum_exp` must equal the number of distinct sources |

Until those are green, the arm in the queue is Full-mode AttnRes only. At our
depth (2-4) Full *is* Block: the paper's own §3.2 says `N = L` recovers Full
and Fig. 6 reports `S = 1` matching Full at 1.737, so the Block variant buys
memory at a depth where the memory is four `[b,t,d]` tensors. **Recommendation:
do not wire it.** The follow-up worth doing is the fix-and-gate set, not the
wiring.

---

## 6. Seam counter (ADR-0019)

`probe::ATTNRES` (index 10; `N_ARMS` 12 → 13), noted at the aggregation
branch. `probe::NAMES` carries `"attnres"`.

- `preset_exec.rs` `fields_no_preset_ships_still_take_effect` asserts the
  count equals `max_iter` on a `small`-shaped model **and** that the logits
  move — a counter that fires while the output does not is the GR defect.
- `loop_block.rs::attnres_counts_every_iteration_it_aggregates` pins
  `max_iter` at depth 4, `0` on the ReZero path, and `2` under
  `set_depth(Some(2))` — random depth must count **what ran**, not what the
  config permits.
- **The eval line does not yet print it.** The trainer's eval line
  (`dormouse-train/src/lib.rs:1547`) carries `engram=<rows>/<arms>` and
  `norm=<ran>/<asked>`; adding `attnres=` there is a one-line change and is
  listed as a follow-up rather than done, because the train crate is a
  different lane's file this run and a counter in a log nobody reads is the
  weaker half of the ADR-0019 rule. **The counter exists and is gated; the log
  field is owed.**

---

## 7. Claim ledger

**Sourced (external file named).**
- The paper is arXiv:2603.15031; PDF md5 `f8351f26bce4c33b2880dee3d101f4b0`;
  read 2026-09-30. §3.1, Table 5 fn 2, Fig. 2, Eq. 1-6, §5 quotes above.
- No author code: repository tree at `85e2231` is 6 files, 0 source. Re-measured
  2026-09-30.
- Reference [66] is Zhang & Sennrich, NeurIPS 32 (2019) — the `1/d`-mean
  definition of RMSNorm.
- The three-convention numbers in §3 are arithmetic on the fixture above,
  computed by the gate in `lib.rs`; they are **ours**, reproducible by running
  the test, and they are not a measurement of anything the authors published.

**Measured on this box, 2026-09-30, `wt/attnres-model`.**
- `burn-attnres --features cuda,autodiff`: **19 passed / 0 failed** (3
  `#[ignore]`d benchmarks), with `ScoreForm::Paper` as the default. Includes
  both fused/tensor parity tests over **both** forms, the balanced-
  checkpointing seam test, and the fused-adjoint-vs-burn-autodiff comparison.
- `burn-attnres` ndarray: **10 passed / 0 failed**.
- `dormouse-core -p dormouse-data -p dormouse-train --lib`: **59 / 15 / 50
  passed, 0 failed**. `dormouse-core --features cuda --lib`: **59 passed**.
- The derivative defect above, red on the pre-change code's successors and
  green after the fix, with the magnitudes in the falsification record.

**Transcription (our reading, no external implementation).** Every "the paper
says" in §2. The score convention. The placement.

**Not claimed.** That AttnRes helps *this* model: no A/B has been run, the
queue row is registered, and the paper's GPQA/HumanEval deltas are theirs at
48B, not transferable to a 9M-parameter byte LM.
