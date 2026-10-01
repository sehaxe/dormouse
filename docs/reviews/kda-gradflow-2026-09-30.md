# KDA attention arm: per-group gradient proof on CUDA — work log

Task: close `AGENTS.md` §3.3's oldest open debt — prove on CUDA that the KDA
attention arm receives gradients. `8fa5d4c` fixed the gate; its own commit
message says the gradient-flowing check was NOT done, and it still was not.

Lane: worktree `wt/kda-gradflow`, off `a071ecd`.
Deliverable: `vendor/dormouse-fused/crates/dormouse-kda/tests/kda_param_grads_cuda.rs`.

## The seam, as read (2026-09-30)

`crates/dormouse-core/src/loop_block.rs:397` →
`KdaModule::forward_train_state::<B>(normed_attn, kda_s.take())` →
`self.project(x)` → `dormouse_kda::fused::cuda::kda_fused_chunk_reported::<B>` →
(not bare cuda) → `dormouse_gdn2::chunk_dispatch::<B>`.

`chunk_dispatch` (`cuda_dispatch.rs:365-441`) on the trainer's backend
`Autodiff<CudaBare, BalancedCheckpointing>`:

1. `note_dispatch_asked()`;
2. `probe!(NoCheckpointing)` — the op returns `Autodiff<CudaBare,
   NoCheckpointing>` tensors, which do **not** convert to `B`, so the
   `try_into_primitive::<B>()` pair fails and the arm is skipped;
3. `probe!(BalancedCheckpointing)` — the op builds ONE autodiff node on the
   caller's own strategy. Reached only if
   `chunk_wy_forward_autodiff_s`'s `any_tracked` guard passes, i.e. at least
   one of q/k/v/g/b/w/state is a tracked node;
4. otherwise `chunk_wy_forward_impl` on the INCOMING tensors, so burn builds
   the graph itself (`ops_path`), plus `fused_declined`.

`chunk_wy_forward_autodiff_s` (`autodiff.rs:511-722`) therefore declines to a
LEAF-free op in two cases: `!any_tracked`, or `DM_FUSED_KDA=0`. Both now fall
through to step 4, which is a real gradient. That is the fix.

The counters that say which arm ran — `dormouse_gdn2::seam_counts()` returns
`(asked, fused_fwd, fused_bwd, declined, ops_path, custom_node_bwd)`. The
decisive one for "did a backward run inside the op" is `custom_node_bwd`
(index 5), incremented by `note_backward_node()` on entry to `ChunkWy::backward`
— which is also the hook that `DM_GDN2_BWD_TRACE=1` prints `ENTERED` from.

## Why the test is in dormouse-kda, not dormouse-gdn2

The brief asked for `dormouse-gdn2/tests/`. The arm the trainer runs is
`dormouse_kda::KdaModule` (`crates/dormouse-core/src/attention.rs:55`) and its
parameter names are the ones the brief lists — `a_log` / `b_alpha` are
`KdaDecay`'s. dormouse-gdn2's own `GatedDeltaNet2` has `a_log` + `dt_bias` and no
`b_alpha`; it is not what the trainer builds, and dormouse-gdn2 cannot depend on
dormouse-kda without a cycle. So the test lives in dormouse-kda, which can reach both
the module and the seam counters it already depends on.

## What already existed, and what it left open

`dormouse-kda/tests/ops_grad_cuda.rs` (the strong gate) differentiates exactly one
parameter, `q_proj.weight`, against central differences at `h = 1e-2` with
`REL_BAR = 5e-2`, and asserts `ops_path > 0 || fused_bwd > 0`. That is a real
test of one tensor. It leaves open the class the defect actually lives in: a
backward that reaches the first projection and stops. 8fa5d4c gave the WHOLE
arm no gradient, so a one-tensor gate would have gone red — but the per-group
structure is what makes the failure legible and what the brief demands.

## Design of the new test

- Same config, same seed, same `min_decay = 0.9`, same `H`, same `REL_BAR` as
  `ops_grad_cuda.rs`, so the two files are comparable line for line.
- `[2, 256, 64]` = 512 tokens, 16 chunks at `chunk_size = 16` — every chunk
  boundary crossed. The defect is gradient flow, which does not depend on the
  token count, and FD cost is linear in forwards.
- Reference = central differences of the SAME forward on the SAME weights, with
  the loss accumulated **in f64 on the host**. `into_scalar::<f32>()` sums
  ~10⁴ terms in f32 and is then differenced against itself; that cancellation
  is the instrument's whole noise budget.
  **Not a true f64 forward** — the forward is f32 on CUDA, exactly as the arm
  runs. An f64 module would need a transcription of the whole KDA block
  (`KdaModule` is not generic over its float element; `NdArray` is f32-only in
  burn 0.22 — `burn-ndarray-0.22.0-pre.4/src/backend.rs:55`, `pub struct
  NdArray;` with `DType::F32` in `NdArrayDevice::defaults`), and the tree's
  existing f64 layer is host-side fixture data, not a backend. So the
  resolvable floor is derived and printed instead:
  `fd_abs_floor(L) = 32·eps_f32·|L|/H`, and an FD below it is refused as
  unresolvable rather than compared.
- Per group: `grad()` is `Some`, all entries finite, `amax > 0`, and
  `min(rel(h), rel(h/10)) < REL_BAR` at the analytic argmax + 2 spread
  coordinates. Taking the min of the two step sizes is deliberate: a
  truncation-dominated residual falls with h, a round-off-dominated one rises,
  and the min refuses a value that only looks good at one step.
- Groups enumerated EXPLICITLY (11 on the trainer's shape, 14 with the short
  conv on), and `every_group_is_present_for_the_config` compares the list
  against a spelled-out `want` — so a parameter added to `KdaModule` and
  forgotten is a loud failure, not a silent hole.
- Both dispatch arms: default (fused) and `DM_FUSED_KDA=0` (ops). The env var
  is read on every dispatch, so both tests take one file-wide mutex.
  `every_group_gets_a_matching_gradient_on_the_fused_arm` asserts
  `custom_node_bwd > 0` — if the fused arm ever declines by default again, the
  test goes red instead of quietly re-testing the ops arm under a fused label,
  which is the mistake 8fa5d4c made.

## Log

(appended as the runs land)

## Run 1 — first instrument (per-group central differences, f64 loss sum)

Ran with the shared target dir. Result: **the q/k/v groups pass at rel 1e-5..1e-3,
and `decay.w_up.weight` fails the instrument's own resolution floor**:
`|L| = 9.898e1`, floor `32·eps_f32·|L|/H = 3.78e-2`, and the FD at that
coordinate was `3.31e-4`.

This is a real property of the module, not a bug in the check: the decay path is
TWO stacked 0.02-std projections (`KdaDecay::w_up` → `w_down`), so its gradient
is ~1e-4 while the projections' are ~1e0-1e1. A central difference resolves a
derivative of size `m` against a loss of size `|L|` to about `eps32·|L|/(2h·m)`,
and at `m/|L| ≈ 3e-6` in f32 that is 3.6% — one digit. So the FD is not a usable
instrument for four of the eleven groups, and no step size fixes it: making `h`
big enough to beat the round-off pushes the perturbation out of the linear
regime of a 0.02-scale parameter.

Consequence for the design: a SECOND, exact reference. The same module, the same
weight bytes and the same input bytes on CPU `NdArray`, its own autodiff graph.
Exact rather than truncated, so it covers every magnitude, and a different device
and a different chunk implementation, so it also cross-checks the fused CUDA
forward against the tensor-ops one. It cannot catch a wrong formula (both run
`chunk_wy_forward_impl`), which is why the FD stays for the groups it can
resolve. A group whose FD is unresolvable is held to a **five-times tighter** bar
against the CPU reference (5e-3 vs 5e-2), not waved through.

The FD step is now derived per group, `h* = (3·eps32·|L|/|g|max)^(1/3)` clamped
to 5% of the parameter's RMS, and each group prints which of the two bounds
bound. For the groups where the old fixed `H = 1e-2` was already right the rule
reproduces it to within ~1.3x, which is the check that the rule is the right one.

## FINDING 1 — on the trainer's backend the fused op DECLINES; the ops path carries the gradient

Run 1's seam line, with `DM_FUSED_KDA` UNSET, on `AdBal` with a real
`KdaModule` at the trainer's parameter shape:

```
seam after backward: asked=1 fused_fwd=0 fused_bwd=0 declined=3 ops_path=1 custom_node_bwd=0
arm: OPS path (burn's own graph)
```

So: **the KDA attention arm DOES receive gradients on the trainer's backend** —
`8fa5d4c`'s fix works — but it receives them through burn's own graph, and the
fused CUDA chunk kernel **never launches in training**. That is what the
`fused kda=0/0` field in §3.1's logs has been reading, and it is a different
statement from the one that reading invites. `chunk_wy_forward_autodiff_s`'s
`any_tracked` guard (`autodiff.rs:541`) declines under `BalancedCheckpointing`
because the projections' outputs are checkpoint leaves, and the decline is what
makes the ops path carry the gradient — a correct fix that trades the fused
kernel for correctness.

`dormouse-kda/tests/cuda_gate.rs` already gates the decline; run 1 confirms it on a
real module rather than a synthetic fixture, which is what was missing.

**Consequence for this lane's tests:** the "two arms" are two checkpointing
STRATEGIES, not two env-var settings. `Balanced` is the trainer's and takes the
ops path; `NoCheckpointing` leaves every intermediate a real graph node, so the
fused op is reachable there and `custom_node_bwd` moves. The `DM_FUSED_KDA=0`
env-var arm was dropped: on the trainer's backend it lands on the SAME ops path
(verified in run 1: `declined=3 ops_path=1`, identical to the default), so it was
a second label on one program.

## The falsification patch (prepared, applied after GREEN)

`dormouse-gdn2/src/cuda_dispatch.rs:449`, one line:

```diff
-        Fused::Fused((o, s))
+        Fused::Fused((o.detach(), s.detach()))
```

This is the 8fa5d4c defect in its modern form: a correct value returned as a
LEAF, so nothing downstream can send a gradient back. It severs the graph at the
ops path, i.e. on the arm the trainer actually runs, which is the point — a
falsification on a path production does not take proves nothing about it.

Expected (and recorded, not assumed): 10 of 11 groups get `grad() == None` and
`o_gate.weight` gets an all-zero gradient tensor, because the gate is the one
parameter applied AFTER the severed op and so still receives a chain. Both are
caught by the `non_zero`/`finite` requirement, with the group names in the
message. `sha256` of the two patched files is recorded before and after.

## FINDING 2 — the fused adjoint kernel's gradients are WRONG (never previously compared)

This is the second half of §3.2's standing warning ("the fused adjoint kernels
have therefore never been numerically compared to anything"). On
`NoCheckpointing`, where the fused op is reached (`fused_fwd=1 fused_bwd=1
custom_node_bwd=1 ops_path=0`), the fused arm's gradient does **not** match
central differences of the fused arm's own forward:

| group | fd rel (fused arm) | fd rel (ops arm) | cpu_rel (fused) | cpu_rel (ops) |
|---|---|---|---|---|
| `q_proj.weight` | 1.2e-3 | 5.2e-4 | 4.0e-7 | 0 |
| `k_proj.weight` | 4.7e-2 | 4.1e-5 | 1.8e-2 | 0 |
| `v_proj.weight` | 6.2e-2 | 6.0e-5 | 1.3e-2 | 0 |
| `decay.b_alpha` | 1.5e-1 | 3.5e-4 | 1.7e-1 | 0 |
| `decay.a_log` | 4.0e-2 | 3.7e-4 | 4.0e-2 | 0 |
| `beta_proj.weight` | 2.5e-1 | 8.3e-4 | 7.9e-2 | 0 |
| `o_norm_w` | 1.9e-6 | 2.0e-6 | 2.2e-7 | 0 |
| `o_proj.weight` | 7.6e-7 | 1.6e-6 | 1.5e-7 | 0 |
| `o_gate.weight` | 2.4e-4 | 6.8e-4 | 2.8e-7 | 0 |

The signature is diagnostic, not just a magnitude: the three parameters applied
AFTER the recurrence (`o_norm_w`, `o_proj`, `o_gate`) agree to 1e-6..1e-7, and
every parameter that reaches the op's INPUTS is off by 2% to 25%. A noisy or
non-deterministic forward would move the downstream three too (it did not: the
two FD step sizes agree to 3 digits on the fused arm). So the fused forward's
OUTPUT is right and its adjoint's INPUT gradients are wrong, and the error is
largest on `beta_proj.weight` (the `b`/`w` write-strength input) and
`decay.b_alpha` (the `g` decay input) — i.e. the two gate inputs.

**This does not affect training today**, because Finding 1 says the trainer's
backend never reaches the fused op. It does mean the fused path is not a
candidate for A/B until its adjoint is fixed, and that `8fa5d4c`'s fix should
not be read as "the fused path works".

Filed as a follow-up with `file:line` for the dormouse-gdn2 owner (§1.6: not this
lane's file). The test asserts FLOW on that arm and PRINTS the disagreement on
every run, so the fused arm can never be reported as numerically verified.

## FINDING 3 — a fused forward launches on the trainer's backend and its result is discarded

`seam: asked=1 fused_fwd=1 fused_bwd=0 declined=2 ops_path=1 custom_node_bwd=0`
on `AdBal` with `DM_FUSED_KDA` unset. Read straight: the dispatch was asked
once, ONE fused chunk kernel launched, the op declined twice, the ops path
carried the step, and the custom node's backward never ran. A fused forward
whose output nothing consumes is a fused path that runs and is thrown away —
the exact shape of a silent fallback (ADR-0019's third mark), and worth the
dormouse-gdn2 owner's attention. What this file asserts is unaffected either way:
whichever arm runs, every group gets a correct gradient.

## Result (green) — the trainer's backend, `AdBal`, trainer parameter shape

`seam: asked=1 fused_fwd=1 fused_bwd=0 declined=2 ops_path=1 custom_node_bwd=0`,
`loss = 9.898036775359e1`, and the CPU `NdArray` reference's loss is the SAME
number to 12 significant digits (relative 0.000e0) — the two backends agree
bit-for-bit on the ops path, which is why every `cpu_rel` below is exactly 0.
That is a measurement, not an assumption, and the fused arm's loss differs by
5.0e-9, so the two backends are not trivially the same computation.

| group | non-zero | finite | amax | cpu_rel (maxdiff) | fd rel |
|---|---|---|---|---|---|
| `q_proj.weight` | yes | yes | 7.175e0 | 0 (0) | 5.2e-4 |
| `k_proj.weight` | yes | yes | 8.375e0 | 0 (0) | 4.1e-5 |
| `v_proj.weight` | yes | yes | 1.687e1 | 0 (0) | 6.0e-5 |
| `decay.w_up.weight` | yes | yes | 3.412e-4 | 0 (0) | UNRESOLVED |
| `decay.w_down.weight` | yes | yes | 4.470e-4 | 0 (0) | UNRESOLVED |
| `decay.b_alpha` | yes | yes | 1.390e-2 | 0 (0) | 3.5e-4 |
| `decay.a_log` | yes | yes | 7.436e-2 | 0 (0) | 3.7e-4 |
| `beta_proj.weight` | yes | yes | 1.404e0 | 0 (0) | 8.3e-4 |
| `o_norm_w` | yes | yes | 1.699e1 | 0 (0) | 2.0e-6 |
| `o_proj.weight` | yes | yes | 2.474e1 | 0 (0) | 1.6e-6 |
| `o_gate.weight` | yes | yes | 1.380e0 | 0 (0) | 6.8e-4 |

Short conv ON (a config the trainer has never run), same backend — and the three
conv weights get gradients for the **first time on record**:

| group | non-zero | finite | amax | cpu_rel | fd rel |
|---|---|---|---|---|---|
| `q_conv_w` | yes | yes | 6.735e0 | 0 | 7.8e-5 |
| `k_conv_w` | yes | yes | 3.137e0 | 0 | 1.7e-4 |
| `v_conv_w` | yes | yes | 7.069e0 | 0 | 1.3e-5 |

(14 groups, all the rest as the table above at a different `|L|` = 2.97.)

## Two instrument bugs this file made, both of which produced a plausible lie

Worth recording, because both read as findings and neither was one:

1. **Per-entry relative** `|a−b|/|b|` over a gradient tensor. A gradient spans
   three orders of magnitude within one group, so an entry where the reference
   is 1e-7 scores 1e3 while meaning nothing. Every group read 1e1..1e4 on a
   comparison agreeing to 1e-6. Fixed to `max|a−b| / max|b|` — the measure every
   other fused test in this library uses. The absolute maxdiff and the
   reference's own amax are printed next to it, so a ratio of 0 cannot be
   confused with a comparison that never ran.
2. **The CPU reference drew its own input.** `run` called `input()` again,
   which advanced the device RNG, so the reference saw a different sequence than
   the gradient was taken on. Every group read `cpu_rel` ~ 1.0, which is
   indistinguishable from a real disagreement. Fixed by threading one draw to
   both sides, and — because the class will recur — by a forward-agreement
   assertion (`LOSS_BAR`, 1e-3) that says the two sides computed the same
   FUNCTION before any gradient is compared, plus a device assertion that the
   reference is not on CUDA, because a `Device::default()` that changed its mind
   would make the reference measure itself.

Both are the same defect the project keeps meeting: an instrument that cannot
distinguish agreement from noise is a green light, not a check.

## Correction to Finding 2: the fused arm is also NON-REPRODUCIBLE run to run

Two runs of the same binary, same seed, same fixture (the green run above and
the one before it) disagree on the fused arm's *amax*:

| group | amax run A | amax run B | spread |
|---|---|---|---|
| `decay.b_alpha` | 1.735e-2 | 1.425e-2 | **18%** |
| `decay.w_up.weight` | 4.148e-4 | 3.868e-4 | **7%** |
| `decay.w_down.weight` | 6.185e-4 | 4.539e-4 | **27%** |
| `k_proj.weight` | 8.412e0 | 8.400e0 | 0.14% |
| `q_proj.weight` | 7.175e0 | 7.175e0 | 0 |
| `v_proj.weight` | 1.686e1 | 1.686e1 | 0 |
| `o_norm_w` | 1.699e1 | 1.699e1 | 0 |
| `o_proj.weight` | 2.474e1 | 2.474e1 | 0 |
| `o_gate.weight` | 1.380e0 | 1.380e0 | 0 |

The OPS arm is bit-reproducible across the same two runs and bit-identical to
the CPU `NdArray` reference (`maxdiff` exactly 0 on all 11 groups, twice). So
the non-determinism is the fused adjoint kernel's, and it is concentrated in the
decay path — the same groups whose central-difference disagreement is largest.

This WEAKENS the reading of Finding 2 and the correction is the point: a
non-deterministic quantity cannot be pinned against a central difference, so
"the fused adjoint's `b_alpha` gradient is 26% wrong" is not established — "the
fused adjoint's decay-path gradients vary by 7-27% between identical runs, and
disagree with a central difference of its own forward by 1.5e-1 to 2.6e-1 at
EITHER run" is. The structural evidence survives the weakening and is the
stronger half of the claim anyway: the three parameters applied AFTER the
recurrence agree to 1e-7 with an independent CPU autodiff, run after run, while
every parameter reaching the op's INPUTS does not. A noisy forward would move
the downstream three too.

What it costs, and what it means for the project: `dormouse-gdn2`'s fused adjoint
cannot be A/B'd, and no fused number on it can be quoted, until both the
non-determinism and the offset are explained. This is the same finding §3.2
recorded as "the fused adjoint kernels have never been numerically compared to
anything" — the comparison now exists, and the answer is not good.

Falsification, the trace, and the commit: below.

## THE FALSIFICATION — red, with the group named, then restored byte-identical

The break is one line, on the arm the trainer runs
(`dormouse-gdn2/src/cuda_dispatch.rs:449`, the ops path):

```diff
         note_fused_declined();
         note_ops_path();
-        Fused::Fused((o, s))
+        Fused::Fused((o.detach(), s.detach()))
```

A correct VALUE returned as a LEAF — the 8fa5d4c defect in its modern form.
Run of the same binary, same seed, same fixture; the test run's own exit was
`RUN_EXIT=101`:

| group | GREEN | FALSIFIED | why |
|---|---|---|---|
| `q_proj.weight` | amax 7.175e0 | **amax 0.0, no gradient tensor** | upstream of the break |
| `k_proj.weight` | 8.375e0 | **0.0** | upstream |
| `v_proj.weight` | 1.687e1 | **0.0** | upstream |
| `decay.w_up.weight` | 3.412e-4 | **0.0** | upstream |
| `decay.w_down.weight` | 4.470e-4 | **0.0** | upstream |
| `decay.b_alpha` | 1.390e-2 | **0.0** | upstream |
| `decay.a_log` | 7.436e-2 | **0.0** | upstream |
| `beta_proj.weight` | 1.404e0 | **0.0** | upstream |
| `q/k/v_conv_w` | 6.7e0 / 3.1e0 / 7.1e0 | **0.0** (conv-on run) | upstream |
| `o_norm_w` | 1.699e1 | **1.699e1, cpu_rel 0** | APPLIED AFTER the break |
| `o_proj.weight` | 2.474e1 | **2.474e1, cpu_rel 0** | APPLIED AFTER the break |
| `o_gate.weight` | 1.380e0 | **1.380e0, cpu_rel 0** | APPLIED AFTER the break |

The two failure messages, verbatim:

```
the KDA attention arm has NO USABLE GRADIENT for 8 of 11 parameter groups:
[q_proj.weight, k_proj.weight, v_proj.weight, decay.w_up.weight,
 decay.w_down.weight, decay.b_alpha, decay.a_log, beta_proj.weight].
The arm is frozen — the 8fa5d4c signature (AGENTS.md 3.2). A group that gets
no gradient while its neighbours do is that defect, not a numerical accident.

the KDA attention arm has NO USABLE GRADIENT for 11 of 14 parameter groups:
[... , q_conv_w, k_conv_w, v_conv_w]. The arm is frozen — ...
```

**The three groups that stayed green are exactly the three applied after the
severed op.** That is the whole argument for the per-group design, made by the
experiment rather than by assertion: a single "does the arm get a gradient"
check would have reported three healthy groups and passed this run. A gate that
cannot name what is missing is a gate that cannot fail.

The fused-arm test correctly stayed green — the break is on the ops path and
that test asserts `ops_path == 0`. A falsification on a path production does
not take would have proved nothing about production; this one is on the path
production takes.

Restored: `sha256(cuda_dispatch.rs) = e28acfb9…401d` before the patch and after
the revert, `git diff --stat` empty for that file, and the suite green again on
the restored tree.
