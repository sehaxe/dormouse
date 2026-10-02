# The graph seam in the training loop — v1: fwd+bwd captured, the tail not

Lane: owner's directive "0 GPU sync, fix the 400 ms". Status: **`L` is measured
and the pin's necessity is measured on the trainer's own path; the positive
gate and the 500-step A/B are not.** Numbers below carry their command and the
commit they were taken at.

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

### 5.1 What the gates said on their first run, and what each refusal taught

The CUDA gates ran three times on 2026-10-01 and the history is the useful
part:

| run | negative | positive | what it found |
|---|---|---|---|
| 1 | **green** — unpinned diverges by **1.98e0** relative at `aux.dspark.joint_proj.weight`; 2.98e0 at `loop_block.mor_router.proj.bias`; 1.015e1 on a third run | red: `could not pin x` | **The trap is real on the trainer's own path**, not just on the toy. The pin is mandatory, and the negative is reproducible (three runs, three different parameters first to break, all ≈ 1-10 relative). |
| 2 | green (same shape) | red, now with burn's own words: `BackendMismatch("Expected concrete Cube backend with disabled autodiff context, got Enabled(Disabled)")` | **A raw handle is unreachable from an autodiff tensor.** Int inputs have no `detach()` (float-only), so the plain-device experiment. |
| 3 | green | red on the FLOAT pin, same message | The same wall for parameters: `optim.step`'s output is `Enabled` too. The answer is `as_constant` (§7): `DispatchTensor.autodiff` is a pub field. |
| 4 | green (3.285e0 at `loop_block.mor_router.proj.bias`) | red on the float pin, **after** `as_constant` compiled | One level deeper than the context: `DispatchTensorKind::Autodiff` is a `Box<DispatchTensorKind>`, so a tracked float sits two levels above the concrete `Cube` variant. The Int half of the same test file went green on the previous run, which is what said the two sides differ. Fix written (`as_constant_float` unwraps the box), **not yet compiled**. |

Every run so far changed WHICH parameter the unpinned graph broke first
(`aux.dspark.joint_proj.weight`, `loop_block.mor_router.proj.bias` twice,
`aux.jepa_pred.proj.weight`), and the magnitude was always 1-10 RELATIVE. That
variety is itself the finding: a stale-pointer graph does not fail in one place,
it fails wherever the optimizer's free list happened to hand the old buffer to
something else. A reader looking for "the" symptom will not find it.

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

1. ~~**`L`**~~ — **TAKEN, 2026-10-01 15:00, the 100k run's card.** `--preset
   small --batch 8 --seq-len 512 --no-engram --steps 300 --log-every 100
   --timers`, release, this binary. `L = (launches@200 − launches@100)/100 =
   (4 410 168 − 2 266 805)/100` = **21 434 launches per warm step** (the
   handover's `@300 − @100` form is unavailable: 300 is not a log step for
   `--log-every 100`, so the timer never prints it). The decision rule, fixed
   before the number: `> ~200` ⇒ the graph is the step. **It is 107× the
   threshold**, and the handover's own inference (~14.5k from 460 ms ÷ 34.4 µs)
   was in the right place and 47% low.
2. **Does the window allocate?** §3.1 says it must not, and the refusal message
   names the cause, so this is answered by the first graphed run. **NOT TAKEN.**
3. **The positive gates** (§5). **NOT GREEN** — see §5.1. The last fix
   (`as_constant_float` also unwrapping `DispatchTensorKind::Autodiff`, which is
   a `Box<DispatchTensorKind>`) is written and committed but **not compiled and
   not run**: the build lock was held by four other lanes for the last hour of
   this session. One command settles it:
   ```bash
   CARGO_TARGET_DIR=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/dt-graph \
     tools/build_lock.sh run graph -- cargo test -p dormouse-train \
     --features cuda --test graph_seam_cuda -- --test-threads=1 --nocapture
   ```
4. **Control vs graph over 500 steps.** `--timers` now prints a warm-step mean
   (steps ≥50, because step 0 is the autotune step at 23× a warm step, AGENTS
   §3.1) on **every** run, so both arms are read with the same instrument, and
   both binaries are built. **NOT TAKEN** — blocked on gate 3.
5. **The autotuner** may sync inside the window; a refused capture says so.
6. **VRAM**: the retained slices are the step's working set, held for as long as
   the graph lives, and every refused attempt adds one (§3.2). At batch 8 that
   is a large fraction of 16 GB — the first thing to watch if a graphed run OOMs
   where an ungraphed one did not.
7. **A v2 that captures the tail** (`opt`, `retr`, `ema`, ~15% of the launches)
   needs the optimizer's moments pinned, which means reaching into
   `ModuleOptimizer`'s private state or reimplementing the optimizer. That is a
   separate decision with its own cost; §1 is why v1 stops here.

## 6.1 What `L = 21 434` buys, in arithmetic

A replay is one dispatch (~42 µs of host time, measured) instead of 21 434
launches. The host is spending 517 ms / 21 434 = **24.1 µs per launch** on this
run — lower than the 34.4 µs a bare `::launch` costs, because a real op does more
work per launch than the anchor kernel does. Device time is ~60 ms (13.3%
utilisation, §3.1). So the arithmetic says a captured window turns a ~517 ms
step into roughly (60 ms of device work + one dispatch + the pin's ~111 copies),
i.e. **a 4-8× step-time win is what the mechanism promises**, and the promise is
arithmetic, not measurement: no A/B has been run. The pin's price against `L` is
`111 / 21 434` = **0.5%** of the launches it removes.

## 7. Files

- `crates/dormouse-train/src/graph.rs` — the seam (new). `as_constant_float` /
  `as_constant_int` are the pin's reason to exist in one sentence: burn's
  `Tensor::try_into_primitive` refuses any tensor whose autodiff context is
  `Enabled`, and a captured graph needs the raw device address, so the seam
  marks the tensor `Disabled` — which is what burn itself means by that word —
  for the length of one copy kernel. Data, device and allocation are untouched;
  the result never goes back into a burn op.
- `crates/dormouse-train/src/lib.rs` — `TrainCfg::graph_capture`, the arm, the
  window closure, the pin refreshes, the timer/report lines.
- `crates/dormouse-train/src/cfg.rs` — `graph::check` in `resolve`.
- `crates/dormouse-cli/src/bin/train.rs` — `--graph-capture`.
- `vendor/burn-fused/crates/burn-muon-plus/src/fused_kernels.rs` —
  `copy_into_cuda`, `copy_into_i32_cuda`.
---

## Continuation — 2026-10-02, the gates finally ran (morning operator)

The settling command of §6.3 ran at 03:55 on the free card (build 5m45 in
`/mnt/…/dt-graph`, test 75 s, `c2e580b`). **Result: 1 of 4 green. No green
light for `--graph-capture`; the branch does NOT merge.**

| gate | result |
|---|---|
| `a_replay_launches_no_kernels` | **green** — a replay moves the launch counter by 6 (the pin's per-step copies + input feeds), everything else replays |
| `a_pinned_replay_agrees_with_fresh_launches_to_f32_noise` | **red** — 8.917e-1 relative at `loop_block.iter_embed[3,64]` after 7 graphed steps (`captured 1 replayed 6 refused 0, pin=384 launches`) |
| `replays_train_and_are_reproducible` | **red, and this is the loud one** — 6 replays leave the parameters **bit-identical to the capture step** (0.000e0 movement, "(none)"): the replayed window trains nothing |
| `the_stale_pointer_trap_is_reproduced_without_the_pin` | **red, wrong reason** — the unpinned arm no longer diverges by a clean signature; it dies with `CUDA_ERROR_ILLEGAL_ADDRESS` at the first read ("The bytes were never written (failure #271231)"). The trap is real but the gate cannot always measure it as a number |

**Reading (hypothesis, clearly marked):** the two red positives are one
defect. `pin=384` over 7 steps ≈ 55 address refreshes per step against
38 gradient tensors + params + inputs — and the reproducibility gate says
weight updates from the outside `optim.step` never enter the captured
window at all: the replay recomputes with the capture step's buffers
while burn's out-of-place optimizer allocates fresh ones each step. That
is §1's own wall, now measured: **a window that excludes the optimizer
cannot train a model whose optimizer is out-of-place.** The fix is
either the optimizer inside the window (the moment-pinning rewrite §7
defers) or feeding the optimizer from the retained grad buffers — both
owner-lane decisions with real cost, neither a morning fix.

Also recorded: the negative gate's instability (a clean 1-10 relative
signature on 2026-10-01, an illegal-address crash today on identical
logic) means *measuring* the trap is itself nondeterministic — a stale
pointer lands wherever the pool's free list hands the buffer, and
sometimes the answer is a dead context, not a wrong number. Any future
version of that gate should assert on the crash OR the divergence, not
insist on the divergence.

Decision recorded per the brief: gates not green → `wt/graph-trainer`
stays unmerged, `--graph-capture` ships in nothing, this file is the
blocker report.
