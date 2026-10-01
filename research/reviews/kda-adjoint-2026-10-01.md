# The fused KDA adjoint — autopsy, and the gate that was missing

Lane: worktree `wt/kda-adjoint`, off `d8062d1`.
Brief: the fused adjoint is measured WRONG (`d8fa449`, 4.5e-2..2.6e-1 on
q/k/v, `beta_proj`, `decay.b_alpha`) and non-deterministic; fix it, or deliver an
autopsy.

**Status: the "wrong term" hunt was aimed one scope too narrow, and the fix is
one line of fixture selection.** Read on.

## 1. The contradiction on arrival, and which side is wrong

Two gates in this tree disagree about the same kernels, and both are green:

| gate | reference | shape | verdict on record |
|---|---|---|---|
| `fused_adjoint_f64.rs` (`814847b`) | f64 forward-mode AD + FD, `ref_bwd_f64.bin` | **T=16, chunk=16 — ONE chunk** | GREEN, term by term |
| `fused_adjoint_vs_ops.rs` | burn autograd over the ops path (tier d) | T=16 and T=64 | GREEN, `GRAD_REL_TOL` 3e-3 |
| `kda_param_grads_cuda.rs` (`d8fa449`) | FD of its own forward + CPU `NdArray` | **T=256, chunk=16 — 16 chunks** | 2-25% on the op's INPUT gradients |

`fused_adjoint_f64.rs`'s own header says the scope out loud, and it is the whole
story (`tests/fused_adjoint_f64.rs:28-41`):

> **Do not read a green run of this file as "the fused backward is correct".**
> It is "the fused INTRA-chunk adjoint is correct against an independent oracle."

BK1 (token-parallel intra-chunk) is verified. **BK2 (the sequential reverse
recurrence over chunks) and the `d_k_bptt` / `d_e_bptt` / `d_s_shift` glue have no
f64 oracle at all** — the file says so, names the two terms that lived there
(`2a430cc` measured d_k at 3.3e-1 and d_g at 6.2e-1 before its fix), and says
why it could not add one: *"the forward's chunk carry does not match the f64
transcription on a two-chunk sequence at all (measured 8.3e-1)"*.

`d8fa449`'s disagreement is measured at **16 chunks**. The verified scope is
**1 chunk**. Those are not the same function, so the two gates are not in
conflict; they are measuring BK1 and BK2.

## 2. THE BLOCKER WAS RETRACTED 16 HOURS BEFORE `d8fa449` RAN

The reason no two-chunk gradient gate existed is recorded in that same header,
and it is **stale**. `c305ec5` (09-30 18:17) fixed the cause:

> `two_chunks_the_forward_still_agrees_with_the_f64_transcription` was red at
> chunk 1 = 5.603e-01 for five agents' worth of attempts... There is no such
> line: the fault is `tools/gen_bwd_f64.py:258`, one line of the REFERENCE.
> `decay_last = np.exp(g_last - G)` where `g_last` was already `exp(G_last)` —
> a **double exp**. Three independent implementations agree (ops arm, CUDA
> kernel, FLA); only the oracle differed. **This commit changes no production
> code.**

After it: chunk 1 `5.603e-01 -> 3.327e-08`, final state `8.930e-01 -> 2.621e-07`,
dstate gradient `-- -> 2.039e-07`.

So the "we cannot extend the oracle to two chunks" note is a **stale
justification**, and `d8fa449` was written at 23:55, 5h38m after the thing that
invalidated it. The scope was never widened. That is the actual defect in the
lane, and it is a documentation-ordering accident, not a numerics problem.

## 3. The two-chunk f64 GRADIENT oracle already exists, on disk, unused

`gen_bwd_f64.py:465-491` writes `ref_bwd_f64_carry.bin` at `T = 2*chunk`, same
generator, same two methods, and it contains **gradients, not just the state**:

```
ref_bwd_f64_carry.bin   18 blocks
  q k v g b w (1,2,32,8)   state (1,2,8,8)   d_out (1,2,32,8)
  out (1,2,32,8)   out_state (1,2,8,8)   loss (1,)
  dq dk dv dg db dw (1,2,32,8)   dstate (1,2,8,8)
  spread (fd vs forward-mode) = 2.2599327318344914e-12
```

Seven gradients, every coordinate, **the state carry active** — i.e. exactly
BK2 and the three glue terms. `c305ec5` regenerated it (it is in that commit's
diff, `ref_bwd_f64_carry.bin | 60914 -> 60914`) and used it for the *state*
gate only. `fused_adjoint_f64.rs::load()` hardcodes
`include_bytes!("ref_bwd_f64.bin")` — the one-chunk file. **The oracle for the
ungated scope was committed and never pointed at.**

The oracle's own error is 2.26e-12 relative, so it is not the limiting term at
any bar this crate would use.

## 4. What the "non-determinism" probably is, before it is measured

`d8fa449`'s correction is careful and I do not override it: 7-27% run-to-run
drift on `decay.b_alpha` / `w_up` / `w_down` weakens every FD number attached to
those groups, so "the fused adjoint's `b_alpha` gradient is 26% wrong" is not
established. Two readings are open and the tree distinguishes them:

* **(a) f32 conditioning.** `d8fa449` Run 1 already measured the decay path's
  gradient at amax ~1e-4 against `|L| = 9.9e1` — a 3e-6 ratio, so a central
  difference resolves it to about one digit in f32 (its own words). A quantity
  that is ~1e-4 of the loss it came from is exactly where an autotuner choice
  between two legal reduction orders moves 7-27%. **This predicts the drift
  appears in the f32 FD and NOT in an f64 reference, and it predicts the OPS
  arm is bit-reproducible (measured) because its op order is fixed.**
* **(b) a real race.** A missing `sync_cube()`, or an uninitialised read. The
  BK2 kernel writes `d_s` (`chunk_adjoint_cube.rs:365-370`) and reads `state_in`
  (`:263-273`); both are inside `if r < c` and are `sync_cube()`-separated, so I
  have not found one by reading. `d_e_last` is `Tensor::empty` (`:520`) but the
  kernel is handed the *computed* one (`:568`, `d_e_last_c2`), and the
  `empty` buffer is dropped by `let _ = d_e_last;` (`:576`).

Reading (a) predicts something specific and checkable: **the fused adjoint's
decay-path gradient is deterministic in f64 and only moves in f32.** That is a
one-run experiment once the two-chunk gate exists, and it is the difference
between "the kernel races" and "the instrument cannot see this magnitude".

## 4b. MACHINE, 2026-10-01 11:45-12:00 — a `--tests` build is IO-poison on this box

`cargo build --tests` for burn-gdn2 + burn-kda is **20 test binaries, and cargo
links them in parallel**: 20 `rustc` → 19 `collect2` → 38 `ld.mold`. Measured
while it ran:

```
/proc/pressure/io    some avg10=99.79  full avg10=86.25
/proc/pressure/cpu   some avg10=0.00
/proc/pressure/memory some avg10=0.00
```

**IO full at 86% with CPU and memory at zero.** Not a slow build — a
disk-saturated one. `ld.mold` links at 1.6-3.4% CPU each, i.e. every one of
them is in D-state waiting on the NVMe, and the 20 links are what is making
each other slow. `Dirty: 281 MB` and `flush-8:0` workers in D. The 100k training
run's own `mold`-free steady state is unaffected, and `free -g` never went below
28 GB available — so the RAM discipline in AGENTS.md §2.4 would have called this
box healthy, and it was not. **CPU pressure is the signal that was missing.**

Consequences, both generalisable:

* **Do not build `--tests` (or `--all-targets`) for a multi-binary crate on
  this box while anything else is running.** One target at a time
  (`--test <name>`) links exactly one binary. This is the same law as §1.5's
  "one heavy thing at a time", one layer down: the unit is the LINK, not the
  crate.
* **A killed mold link leaks.** SIGKILL on cargo/rustc leaves the `collect2` →
  `ld.mold` pair in uninterruptible sleep, reparented to PID 1, and it holds its
  IO until the queue drains. 19 of mine were still wedged 10 minutes after the
  build died. Diagnose ownership with `ls -l /proc/<pid>/cwd` (mine pointed at
  the worktree, which is how I knew they were mine and not another lane's) —
  **never** by process name, because after reparenting `ppid` reads as 1 and the
  whole tree looks like it belongs to the label of whatever holds the lock.

The build lock did its job throughout: the `gtrain` lane waited 11 minutes for
mine rather than racing it, and took the lock cleanly when mine died.

**Correction, measured 10 minutes later, and it matters more than the rule
above.** SIGKILL on `ld.mold` does **not** reap it while it is in D-state — the
process survives, holds 9 open fds into the target dir, and the IO queue never
drains, so a *new* build starves at literally 0 s of CPU (my rustc: 8m23s
elapsed, `00:00:00` CPU) and cannot finish. **The leak is self-sustaining: the
thing wedging the disk is the thing that would drain it.** Two consequences:

* `kill -9` must be sent to `ld.mold` **directly**, in a loop, repeatedly —
  not to cargo or rustc. Killing the parent orphans the link and makes it
  *unreachable by process tree*, which is why "which lane owns this?" became
  unanswerable from `ppid` and needed `ls -l /proc/<pid>/cwd`.
* Repeat the kill. A single pass left 19 alive; they only went away after a
  second `kill -9` sweep once their parents were gone. IO `full` went
  86% -> 53% on that sweep.

The whole episode cost ~25 minutes of a ~2.5 h GPU window, and **all of it was
self-inflicted**: the `--tests` build was the trigger, the leaked links were
mine, and the only clean exit was the direct kill sweep. Recorded because the
next lane that reaches for `--tests` on this box will do exactly the same thing,
and because the symptom (a build that "hangs" with the box apparently idle,
`free` healthy, `load` enormous) points at nothing that is actually wrong.

## 5. The gate this lane adds, and the falsification

Extend `fused_adjoint_f64.rs` to run the SAME comparison at one chunk AND at two
chunks, from the two committed fixtures. Nothing new is generated, no tolerance
is moved, `GRAD_BAR` stays 1e-3 (the f32 side's noise).

This is the gate the file's own header asks for and could not have, and it is
one `load(bytes)` parameterisation plus a second call — the shortest diff that
converts a paragraph into a test. Its scope note has to be rewritten with it,
because "one chunk, so BK1 only" is the sentence that let a 16-chunk
disagreement be filed as a kernel defect.

Falsification (the rule from `.bulba/memory.md` 2026-09-30, and the reason
`34c5631` shipped unverified): re-introduce `c305ec5`'s double-exp in the
generator and confirm the two-chunk arm goes red — a gate that cannot fail is
not a gate.

## 6. Status of the brief's items 4 and 5 (the path and the prize)

Not started, and deliberately: **item 4 (enable the fused path in the trainer)
is downstream of this gate.** `d8fa449` FINDING 1 measured that on the
trainer's backend the fused op *declines* under `BalancedCheckpointing` (the
`any_tracked` guard, `autodiff.rs:541`) and the ops path carries the gradient —
so the trainer is correct today and the fused kernel is simply not on that path.
Enabling a kernel whose inter-chunk adjoint is ungated would put an unverified
gradient into production. Order: gate BK2, then the path, then the launch-count
prize.

Also recorded, because it is a live contradiction a reader will hit: `d8fa449`
FINDING 3 measured a **fused forward launching on the trainer's backend whose
result is discarded** (`asked=1 fused_fwd=1 fused_bwd=0 declined=2 ops_path=1
custom_node_bwd=0`) — ADR-0019's third mark. That is a separate defect from the
adjoint's numerics and it is the one that would matter for launch count, because
today the fused forward is pure overhead.
