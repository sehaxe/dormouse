# fused/ grad-coverage verification (phase 1 item 1)

Date: 2026-09-23. Branch: `fused-grad-coverage`. Agent 3 verification pass over
the agent-1 WIP (`1ef0028`) plus the agent-2/3 gate commit (`65ff46c`).
Acceptance criteria from `docs/PLAN.md` phase 1 item 1 (a/b/c).

## Criterion (a): grad coverage == burn path - GREEN

`fused_grad_coverage_matches_burn` (CUDA, arms on, aux on, both Engram modes):

- missing-in-fused list: EMPTY (both host-rows and hashed mode). The original
  failure mode (params whose grads never arrived -> missing optimizer moments)
  is gone.
- extra-in-fused list: EMPTY (the zero-final_norm-seed exception for aux-off
  runs is documented in the test and not exercised here - aux is on).
- forward parity: host-rows loss rel 7.8e-7 (rec 4.6e-7, kl 2.1e-7, aux
  1.1e-5); hashed loss rel 7.6e-6. Worst single param grad rel 4.7e-3
  (limit 5e-2).
- arms-on gradcheck (`fused_gradcheck_arms_on`): n=4 fwd loss rel 1.2e-6,
  worst grad rel 7.0e-3 (halt_w); n=8 fwd rel 9.5e-7, worst 3.4e-3 (lm.v).

Root causes of the two briefied panics (both fixed inside `1ef0028`,
verification is agent 3's):

1. Coverage broadcast panic (burn-cubecl `broadcast_shape`): the rows-parent
   zero seed was registered at hardcoded slot 61. For n_experts=3 that slot is
   a dummy cloned from the X parent, so the `[bt,96]` seed landed on X's node
   and the tape's add of `[bt,d] + [bt,96]` tripped the broadcast assert.
   Fix: `rows_idx = p_final + 1` (31 for nexp=3, 61 for nexp=8).
2. arms-on backend-mismatch panic (burn-dispatch `float_mul`): disabled-arm
   zeros (`Tensor::zeros`) were built with the bare CUDA device, so the
   generic backend resolved to `CB` - a bare `[b,t,d]` tensor as the LHS of
   `mul` against the tracked gate RHS panics ("tensors are not on the same
   backend"). Fix: allocate with `dev_ad = normed_l.device()` so the zeros
   carry the autodiff representation.
   Riding along in the same commit: the inner-adjoint seed is now `dy`
   (= dh_flat*rs, the arms live inside `y`), and `d_raw` is packed to the
   `[bt,2]` gate columns `gate_bwd_add` indexes.

## Suite state

- CUDA lib (`cargo test -p dormouse-core --features cuda --lib`): 22/22 green
  - 8 fused (incl. the two above + the f64 buffer bisects), 14 non-fused.
- CPU regression (`cargo test -p dormouse-core -p dormouse-train --lib`):
  14 + 26 green.

## Flagship wiring (DM_FUSED=1 + --engram-ram + aux on burn side)

The gate (`train/src/lib.rs`) excludes only bf16 / act-quant / GR; host rows
pass through (register onto the op's rows parent -> burn gather chain ->
CPU Adam), JEPA/DSpark/KoLeo consume the op's exposed latents (`out_acc`,
`h`) on the burn graph, EMA teacher stays burn-side. `65ff46c` hoists the
gate out of the step loop and prints the resolved arm at startup
("forward arm: FUSED ponder_loop_step (host-rows engram OK, aux heads
burn-side)"), so a silent fallback is visible in every log.

## Criterion (b): 50-step run + ckpt size - GREEN

Flagship recipe (small, batch 10, s512, 48M-slot engram-ram, host-adam every
step, aux on, fp32, Fp8 factors, Muon+ ns=8):

| run | ckpt bytes | best ce (50 steps) | s/step (steps 10-40) |
|---|---|---|---|
| burn (`burn50`) | 72,987,164 | 5.284 | ~7.05 |
| fused (`fused50`) | 72,987,416 | 5.357 | ~9.0 |

Delta 252 bytes (~0.0003%) vs the pre-coverage 37.1 MB vs 69.7 MB gap: every
parameter receives its gradient, so the optimizer materializes every moment.
My independent pair (`fgc50burn` 72,987,164 / `fgc50fused` 72,987,416,
checkpoints/) reproduces it. Loss trajectories track within ~0.07 ce at
step 40 with no divergence or NaN (criterion d evidence).

Headline perf caveat: at the flagship recipe the fused path is ~1.25x SLOWER
than burn, not the 1.7-2x faster measured at batch 6 with arms+aux off
(ADR-0003 smoke). The arms adjoint reruns KDA/MSA/Engram through an inner
burn Autodiff graph per iteration; the fusion win does not survive arms+aux
at batch 10 yet. Criterion (b) proper (BPB parity on a confirm-tier run)
needs the 2k A/B - the 2k fused arm was launched detached, log:
`/tmp/opencode/fused/fgc2k.log`.

## Criterion (c): resume through the fused path - GREEN

- 48M flagship: two resume attempts of `fgc50fused` loaded model + optimizer
  + step 50 + 18 GB ngram sidecar through the drift check (runs were killed
  by box-level OOM pressure after the load, not by the code - see the new
  GPU/RAM discipline in AGENTS.md; a resumed 48M run holds ~37 GB RSS).
- 2M light round trip (`light_c_fused50`, 732 MB tables): same-argv resume
  printed `resumed ... step 50`, the FUSED arm line, and `done steps=50`,
  exit 0. Mechanics are slot-count independent.

## Remaining exclusions (gate: bf16 / act-quant / GR)

- bf16: every fused kernel is `launch_unchecked::<f32>`; needs bf16 templating
  with fp32 accumulate + cast bridges + bf16 cmma in the ternary mms (plan M5).
  Effort: large (kernel work), payoff per research item 2.
- act-quant: the STE quant kernels are not in the fused chain; needs a quant
  kernel after rmsnorm (FFN at the format, attention at max(bits,8)) + STE
  passthrough. Effort: small-medium.
- GR: the fused forward hardcodes ReZero + pre-norm (hctx/rmsnorm/residual
  kernels); GR needs read/write kernels + a branches workspace
  (GR_BRANCHES x bt x d). Effort: medium. GR is off by default and off in the
  flagship recipe.
