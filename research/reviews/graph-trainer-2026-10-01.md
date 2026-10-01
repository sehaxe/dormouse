# The graph seam in the training loop — v1: fwd+bwd captured, the tail not

Lane: owner's directive "0 GPU sync, fix the 400 ms". Status at the time of
writing: **the seam is wired and CPU-green (52/52), the CUDA gates and every
number are NOT yet run** — a 100k production run owns the card until ~14:40
(`first_run_100000_1001_0202.log`, step 92.5k of 100k at 13:41). So this file
records what was built and why, and marks every claim that waits on a
measurement instead of making it.

Read first: `research/reviews/cuda-graph-2026-09-30.md` (the handover) and
`vendor/cubecl-fix/cubecl-cuda/tests/graph_step.rs` (the mechanism, 5/5 on this
card). This file records what changed in the trainer and why the window is
smaller than the handover's.

---

## 1. What I did NOT do, and why

**I did not capture the whole step.** The handover's §3 argues that
"494 of 500 ms is capturable as ONE graph" because the loss-scalar read is
conditional (log cadence ∥ host-Adam cadence ∥ timers). That argument is
correct about syncs and silent about a second problem:

| tensor | read by the window? | allocated inside the window? | so its address... |
|---|---|---|---|
| params | yes | **no** — `optim.step` is what produces the next one | moves every step (burn is out-of-place) |
| optimizer moments (`moment_1/2`, `mu_momentum`) | yes, by `optim.step` | yes, if `optim.step` is inside | retained ⇒ stable |
| the EMA teacher | yes (the forward runs it) | only if `ema` is inside | moves every step |
| `x`, `y`, `hashed_ids` | yes | no — built from host bytes every step | moves every step |

The middle row is the interesting one and it decides the design: **a capture
window RETAINS every slice it allocated**
(`PersistentPool::retain_touched`, `cubecl-server/.../persistent_pool.rs:97`),
so anything the window allocates is address-stable for free — and the moments
are exactly that, *if* the optimizer is inside. But the moments are reachable
only through `ModuleOptimizer`'s private `param_context: HashMap<ParamId,
OptimizationContext>`; `burn-optim` exposes no accessor, and `to_record()` /
`into_bytes()` read to the host (a sync). So pinning them from outside would
mean reimplementing the optimizer — the one thing this lane must not do
(ADR-0009's hand-fused rewrite was slower; the discipline here is *replay
burn's own launches, do not rewrite them*).

**So v1 captures `fwd → loss → mask → bwd → sanitize`, which is ~85% of the
launches**, and pays the pin for the three rows that move. The tail
(`opt 46 + retr 21 + ema 2.5 = ~70 ms` of a ~460 ms step, `first_run_100000`)
stays ungraphed and stays correct by construction. If `L` confirms the
inference (≈85% of launches ⇒ ≈300-400 ms of host launch time), the tail is
~15% of the win and the version that would capture it is a rewrite of the
optimizer, which is a separate decision with its own cost.

## 2. The pin, and what it costs

Three pins, one primitive (`copy_into_cuda`, one launch, no allocation):

| pinned | price | why |
|---|---|---|
| every model parameter | 1 copy each per step (54 at `small`) | `optim.step` returns fresh tensors |
| every teacher parameter | 1 copy each per step | `ema_update` rebuilds every `Param` |
| `x`, `y`, `hashed_ids` | 1 copy each per step | built from host bytes every step |

54 + 54 + 3 ≈ **111 launches/step ≈ 3.8 ms of host time** at the measured
34.4 µs/launch, against a ~460 ms step. The count is printed (`pin=… launches`)
rather than asserted in a comment, and it is `P` in the handover's
`win iff P < L − 1`.

**Why a copy kernel and not a burn op.** `slice_assign` returns whatever the
backend allocated; `reshape` is a metadata op whose buffer behaviour is a
backend detail. Both would make the "master" move, i.e. an unpinned graph with
no error — the `graph_step.rs` "got -3, correct -6" failure. The kernel lives in
`burn-muon-plus::fused_kernels` next to the `cube_of` downcast that already
exists (`momentum_cuda` uses it on the same dispatch backend — which is why
`muon_skipped=0/0` on the trainer's eval line). That placement is off-mission
for that crate and is called out rather than hidden.

**Why the gradients need no pin.** They are allocated inside the window, so the
capture retains them; the graph rewrites exactly the buffers the retained
`Gradients` object points at. What they DO need is for the optimizer to stop
taking them out: `GradientsParams::from_module` calls `Tensor::grad_remove`,
which removes. One `Gradients` for the life of the capture would then be empty
from the second replay on — a silently frozen parameter set. `graph::grads_params`
uses `Tensor::grad` (borrow, not remove) instead.

## 3. The capture window, and what forces it to end

A capture window refuses stream reads, syncs and handle writes
(`cubecl-runtime/src/client.rs:1288-1296`). The trainer's host round-trips are
therefore all OUTSIDE the window, and each one forces the step to run ungraphed
and the graph to be re-captured (`destroy → prepare → capture`, never capture
over a live graph — both refusals are loud, `graph_step.rs`):

| what | cadence | in the window? |
|---|---|---|
| loss scalar / grad norm / stress | log cadence, host-adam cadence, timers | no |
| `max_ortho`, `memory_cleanup` | every 500 | no |
| eval + checkpoint forwards | `eval_every` / `ckpt_every` | no |
| `sanitize_grads`, `mask_nonfinite` | every step | **yes**, on device (§1.3) |

So on a run with `--log-every 100` and `--eval-every 5000`, ~99% of steps are
one dispatch. A refused capture is COUNTED with its reason printed once and the
run continues ungraphed (`seam.report()` prints captures/replays/refusals at the
end) — a capture that failed quietly would be the `8fa5d4c` shape.

### 3.1 The capture protocol, found by reading the pool (and it is not obvious)

The first version of this seam went `graph_prepare → start_capture → body`, and
**every capture in every run would have been refused.** Reading
`memory_manage.rs` says why:

- `capture_begin` forces `mode = Persistent`, so every allocation inside the
  window goes to the **persistent** pool (exact-sized slices) — not the
  `ExclusivePages` pool `init_pools` installs for the rest of the run.
- The persistent pool is EMPTY at that moment, because the trainer never opens a
  persistent window outside a capture. So the recorded pass allocates everything:
  every allocation is a memory node, and `stop_capture` rejects the graph
  (`cubecl-cuda/src/compute/capture.rs:110-120`) — with a message that says "the
  window grew the pool", which is true and caused by the seam, not by the pool.
- The protocol that avoids it is in `graph_prepare`'s own doc and in
  `graph_step.rs`: **prepare, then warm up, then capture**. `graph_prepare`
  opens a PRIMING window in which `capture_touch` RETAINS every slice the pass
  touches, so one unrecorded pass grows the pool to that pass's full working set
  instead of its transient peak; `start_capture` ends priming
  (`Window::begin` → `capture_priming_end`) and releases those slices as free, and
  the recorded pass reuses them.

So the seam runs the window `CAPTURE_WARMUP = 1` time(s) unrecorded between
prepare and capture, which makes the window closure `Fn` rather than `FnOnce`.
That is the whole reason the closure is `Fn`, and the reason `loss_log` /
`aux_log` come back through `RefCell`s.

### 3.2 A refusal is not free, so there is a valve

Each attempt grows the persistent pool by a working set (priming retains what it
touches). A run that is refused on every step would grow that pool until it
OOMs, which is worse than being slow. After `MAX_REFUSALS = 20` the seam stops
attempting, says `DISABLED` in the report, and continues ungraphed — correct,
because an ungraphed step trains exactly what a replayed one would.

## 4. Refusals, not fallbacks (ADR-0011)

`graph::check` runs in `resolve`, before any GPU work, and refuses:

- off CUDA — a flag that cannot do what it says is not a flag;
- `--rand-depth` — the captured graph has ONE loop depth;
- `--engram-ram` — `host_rows` is a per-step variable-row upload that v1 does
  not pin, and with `--host-adam-every 1` every step would be a host round-trip
  anyway (no step would be replayable);
- `--jepa-targets` — a per-chunk target upload that v1 does not pin.

## 5. Gates (all computed, never hand-written)

The handover records two false greens by the previous agent (an oracle that
happened to equal the stale-pointer output, and a units bug). So:

- `graph::stats_line_names_the_three_numbers_and_the_refusal` — green. The
  report names every number, including a null.
- `graph::a_master_keeps_its_rank_and_refuses_one_it_cannot_carry` — green. The
  rank-erased master round-trips at its own rank and refuses another (no
  reshape anywhere, because a reshaped master that moved is an unpinned graph).
- `cargo test -p dormouse-train --lib` — **52/52 green**, so the seam's
  integration into `train_loop` did not break the 50 tests that were already
  there (including the config-snapshot gate, which is what forced the new field
  into the snapshot — see §8).
- `burn-muon-plus::copy_into_writes_the_existing_buffer_and_leaves_the_source_alone`
  — written, **not run** (CUDA).
- `tests/graph_seam_cuda.rs` — four gates, **built, not run**:
  1. `the_stale_pointer_trap_is_reproduced_without_the_pin` — the NEGATIVE. The
     same run without the pin must DISAGREE with fresh launches. It asserts a
     signature (`> 1e-3` relative), never a magnitude, because a specific wrong
     number is a hand-written oracle — the mistake this lane already made once.
     If it ever goes red, the finding is "the pin is unnecessary on this
     backend", which is worth knowing.
  2. `a_pinned_replay_agrees_with_fresh_launches_to_f32_noise` — the
     differential, against the software run of the same steps.
  3. `replays_are_bit_identical_to_the_first_replay` — N replays, `== 0.0` on
     the worst relative difference.
  4. `a_replay_launches_no_kernels` — the mechanism through the trainer's path.

  Every oracle is computed by running the other arm. The window in the test is
  the same shape as the trainer's and re-derives its own gradients; it is not
  the trainer's 1400-line loop, which is not callable from a test (the same
  wall §3.3 of AGENTS names for the eval call site).

## 8. Two things the repo's own gates caught

- `cfg::tests::snapshot_carries_every_train_field` went red on the new field,
  which is that test doing its job: a new `TrainCfg` field must be asked about.
  Answer: **in the snapshot** (ADR-0021) — a resume that turned the flag on
  would compare two execution paths across one step count.
- `cube_of` only downcast float tensors, so the Int pin (`x`, `y`, the hashed
  keys are `Tensor<_, Int>`) did not compile: `try_into_primitive` is bounded on
  `K: BackendPrimitive<B>`, so one generic over the kind cannot serve both.
  Two four-line functions instead of one that cannot typecheck.

## 6. Open, in the order the evidence says

1. **`L`** — launches per warm step, from the timer line's `launches=`
   difference. Decision rule (fixed before the number, §10 of the handover):
   `L > ~200` ⇒ the graph is the step; `L < ~200` ⇒ null, report and stop. The
   arithmetic already in the tree says `L` is ~14.5k (460 ms ÷ 34.4 µs), but
   that is an inference from a per-launch cost, not a count. **NOT TAKEN.**
2. **Does the window allocate?** §3.1 says it must not, and the refusal message
   names the cause, so this is answered by the first run rather than by an
   argument. **NOT TAKEN.**
3. **The four CUDA gates** (§5). **NOT TAKEN** — the test binary builds, the
   card is busy.
4. **Control vs graph over 500 steps.** `--timers` now prints a warm-step mean
   (steps ≥50, because step 0 is the autotune step at 23× a warm step, AGENTS
   §3.1) on **every** run, so both arms are read with the same instrument.
   **NOT TAKEN.**
5. **The autotuner** may sync inside the window; a refused capture says so.
6. **VRAM**: the retained slices are the step's working set, held for as long as
   the graph lives, and every refused attempt adds one (§3.2). At batch 8 that
   is a large fraction of 16 GB — the first thing to watch if a graphed run OOMs
   where an ungraphed one did not.
7. **A v2 that captures the tail** (`opt`, `retr`, `ema`, ~15% of the launches)
   needs the optimizer's moments pinned, which means reaching into
   `ModuleOptimizer`'s private state or reimplementing the optimizer. That is a
   separate decision with its own cost; §1 is why v1 stops here.

## 7. Files

- `crates/dormouse-train/src/graph.rs` — the seam (new).
- `crates/dormouse-train/src/lib.rs` — `TrainCfg::graph_capture`, the arm, the
  window closure, the pin refreshes, the timer/report lines.
- `crates/dormouse-train/src/cfg.rs` — `graph::check` in `resolve`.
- `crates/dormouse-cli/src/bin/train.rs` — `--graph-capture`.
- `vendor/burn-fused/crates/burn-muon-plus/src/fused_kernels.rs` —
  `copy_into_cuda`, `copy_into_i32_cuda`.