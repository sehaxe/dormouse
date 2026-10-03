# rmsnorm-live — the fused RMSNorm kernel enters the autodiff path (2026-10-03)

**Lane:** `rmsnorm-live` (new worktree `wt/rmsnorm-live` off `7c442ce`).
**Decision:** arm **(a)** — the kernel gets a real backward via an autodiff
node, not a LOUD refusal. Arm (b) was rejected because a bare forward on a
no-grad path would still leave the training forward on the tensor path and
`norm=0/N` forever; the node is what makes the fused kernel reachable where
the loss is.

## What landed (one commit, `42f3cdb`, cherry-picked `ec438ec` + merge fixes)

- `src/ops.rs` — `rmsnorm_node_autodiff_s<Inner, S>` (the `burn-gdn2`
  `chunk_wy_forward_autodiff_s` pattern): lifts the caller's tensors to their
  autodiff primitives, strips to bare for the launch, registers a 2-parent
  node with an analytic backward (`dx = gy_w·inv − x·inv³/d·Σ gy·w·x`,
  `dw = Σ_rows gy·x·inv`), tensor ops on the bare backend. A bare result
  wrapped into the graph would be a LEAF and train nothing — the 8fa5d4c
  defect — so relaxing the dispatch guard was NOT taken.
- `src/fused.rs` — `rmsnorm_launch` split out of `rmsnorm_cuda`; kill switch
  `DM_RMSNORM_FUSED=0` (counted via ASKED+SKIPPED); `arm_counts() =
  (asked, skipped, node_ran)`; `NODE_RAN` counter.
- `src/lib.rs` — one ask per module forward, node arm first, bare arm second,
  tensor path last; every decline counted (ADR-0019).
- `src/cuda_dispatch.rs` — the two conversions, kept local (burn-gdn2 pattern).
- Gates: `tests/autodiff_node_cuda.rs` (node vs tensor path gradients on the
  trainer backend `Autodiff<Cuda, BalancedCheckpointing>` — dx 1.3e-7,
  dw 1.1e-7 scale-relative, bar 1e-6; counter gate on `arm_counts()`; kill
  switch) and `tests/fused_kernel_gate.rs` (bare-CUDA arm TAKEN, answer vs a
  scalar f64 definition, `takes == ROWS.len()`).
- The two stale gates in `tests/rmsnorm_kernel_cuda.rs` inverted: they
  asserted the kernel DECLINED on an autodiff tensor / on the trainer's
  `valid()` snapshot (the doc string encoded `norm=0/N`). They now assert the
  node arm RAN (ASKED +1, SKIPPED +0) and the output still matches both
  upstream fixtures. Had they been left, the node landing would have gone red
  for a reason nobody would have noticed.
- Counter race fixed: `the_kill_switch_declines_both_arms_and_still_counts`
  read the process-global `arm_counts()` WITHOUT the SEAM mutex the two
  sibling tests hold. All three now serialize.

## Gates, honestly

- `cargo test -p burn-rmsnorm` (ndarray, vendor workspace): **green**, 7/7.
- `cargo check -p burn-rmsnorm --features cuda` / `--features wgpu`: green.
- `cargo test -p burn-rmsnorm --features cuda --no-run` (all 6 targets incl.
  `autodiff_node_cuda`) and `--features wgpu --no-run`: green.
- `python3 tools/check_doc_refs.py`: 0 named paths missing.
- **CUDA gate runs: NOT done.** `night2e` owns the card (8.4-9.1 GiB, PID
  2511393), and the brief's protocol is conservative — no second GPU process,
  no `nvidia-smi <1GB` window. The gates are committed, compiled, and ready:
  `cargo test -p burn-rmsnorm --features cuda --test autodiff_node_cuda
  --test fused_kernel_gate --test rmsnorm_kernel_cuda`, plus
  `DM_RMSNORM_FUSED=0 ... --test autodiff_node_cuda` for the kill-switch arm.
  The verdict that the fused kernel trains (gradients non-zero, matches the
  tensor path) is still a claim pending a green run, as on 2026-10-02.

## Not fixed here (follow-ups)

- The node's backward is ~10 tensor-op launches; a fused adjoint kernel is
  the follow-up, gated by a step-time A/B, not by this file.
- `arm_counts()`'s `node_ran` is NOT yet printed on the trainer's eval line
  (the eval prints `fused::calls()` = asked/skipped, which now reads
  asked>skipped — i.e. the node arm shows up as a "run" — but `node_ran`
  itself is dormouse-train-side wiring).
- The stale "norm=0/N" doc comment in `attention.rs:23` now understates: the
  arm can run. One-line doc follow-up.
