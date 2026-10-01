# CUDA graph capture of the training step — instrument, capture, measure

Lane: owner's top priority (maximum speed from the card). The card idles 87% of
a warm step, so the work is launch-bound and a graph replay is the only lever
that attacks that directly. Everything below is measured or read out of the
tree; nothing is asserted from a comment.

Status: **the mechanism is done and measured; the size of the win is not.**
§8 has 5/5 green on the card, a launch counter, a proof that a captured step is
silently wrong without a pin, and the per-launch host cost that explains the idle
card. §9 has the one number that decides whether this lane is a 10x win or a null,
and the command that takes it.

---

## 1. The instrument we already had, free: the control run's own log

The re-baselined CONTROL is running right now (`first_run_10000_0930_2138.log`,
launched 21:38, `--preset small --batch 8 --seq-len 512 --no-engram --timers`).
Its timer lines are a step-indexed stage split of the shape this lane cares
about, on a card with nothing else on it:

```
timer step 6300: total=497ms data=0.1ms fwd=199ms bwd=215ms
                 (incl. loss sync + host-adam D2H) opt=51ms retr=25.7ms ema=3.2ms
```

| stage | ms | share |
|---|---|---|
| data (H2D) | 0.1 | 0.0% |
| fwd | 199 | 40% |
| bwd | 215 | 43% |
| opt | 51 | 10% |
| retr | 25.7 | 5% |
| ema | 3.2 | 0.6% |

`step 0: total=17086ms` on the same run — the 34× autotune artifact, quoted here
so no one reads step 0 as a step.

**So the window worth capturing is 494 of the 500 ms**, not the 76 ms tail
(`opt`+`retr`) that the abandoned attempt in `wt/graph` chose. See §3.

## 2. What the eval line says actually ran (and it is not the whole model)

Same run, step 500:

```
fused kda=2012/0 asked=4096 bwd=0 declined=10276 ops=4096 node_bwd=0
norm=0/4619 muon_skipped=0/0 engram=0/0
```

- `bwd=0` — **the attention arm still has no backward** (the `8fa5d4c` defect,
  §3.3). 215 ms of `bwd` is a backward that is not doing the whole job, so
  "capture the backward" prices a stage that is not yet the real one.
- `norm=0/4619` — the fused RMSNorm kernel ran zero times out of 4619 asks
  (§3.3, ungated).
- `engram=0/0` — correct for `--no-engram`; not a defect here.

This matters for the lane in one specific way: **a captured step freezes the
current numerics**, defects included. Any A/B of "graph vs no graph" must be a
correctness A/B (replay == fresh launch), not a quality A/B, and the quality
claim stays blocked on the attention backward.

## 3. The capture window is bigger than "one stage" — the read is conditional

The documented v1 boundary (`client.rs:1288-1296`: a capture window refuses a
read, a sync or a handle write) is a boundary on the **loss scalar read**, and
that read is **conditional**, not per-step. From `train/src/lib.rs:1111-1343`:

| what | cadence | host round-trip? |
|---|---|---|
| `loss_log` scalar read (`:1256`) | `step % log_every == 0` \|\| host-Adam step \|\| timers | yes, on those steps only |
| host-table grad D2H (`:1289`) | `--host-adam-every` | yes, on those steps |
| `grad_norm` (`:1311`) | `step % log_every == 0` | yes, on log steps |
| `stress.observe` (`:1316`) | log steps | yes |
| `max_ortho` (`:1350`) | `step % 500 == 0` | yes, every 500 |
| `memory_cleanup` (`:1682`) | every 500 | yes |
| `sanitize_grads`, `mask_nonfinite` | every step | **no** — on device (§1.3) |

So on a non-log, non-host-Adam step there is **no host round-trip anywhere
between the start of the forward and the end of the EMA**. The honest v1 window
is therefore the whole GPU step, with log steps (and every 500th) running
ungraphed. That is 4× the window the abandoned attempt aimed at.

**Correction to a comment I would otherwise have repeated:** `probe.rs` describes
`RETRACT_FACTOR` as "one host-syncing `polar_orthogonalize` per factor", which
would have put the retraction outside the window. That is **stale**:
`burn-spectral/src/lib.rs:208-233` is sync-free since 2026-09-29 ("112 host reads
per training step ... now takes none"). A doc-vs-code disagreement in the
direction that would have cost us 26 ms of window (§1.7: report it, don't
invent a third answer).

## 4. The blocker nobody had written down: burn's optimizer is out-of-place

A captured graph bakes **raw device pointers**. So the question is not "is the
stage launch-bound", it is: **are the addresses it reads still the addresses
the tensors live at, on the next step?**

`burn-optim-0.22.0-pre.4/src/optim/adam.rs:88`:

```rust
(tensor - delta, Some(state))
```

Out-of-place. The new parameter is a fresh allocation and the old one is freed
*after* the allocation (Rust evaluates the RHS first, then the assignment). So
every step:

```
step k:   param at S_k.  alloc → S_{k+1} (S_k still alive, cannot reuse it).
          free S_k.
step k+1: param at S_{k+1}. alloc → takes S_k from the free list. free S_{k+1}.
```

**The parameter's address alternates with period 2.** A graph captured at step
`k` reads `S_k` and writes `S_{k+1}`; replayed at step `k+1` it reads the buffer
the parameter *used to live in* and writes the buffer it lives in *now*. The
update is computed from two-step-old weights and lands on the right address.
That is silent, it is exactly the `8fa5d4c` failure shape (a healthy-looking
loss curve, a frozen/garbage arm), and it is why the previous attempt's design
was never measured.

Three ways out, in increasing order of how much they assume:

1. **Pin the parameters.** Keep a persistent master tensor per parameter; after
   `optim.step`, `master.copy_from(new)` and re-point the model's `Param` at the
   master. Costs **one extra device copy per parameter per step** (~54 groups
   per the optimizer banner; the tensor count is the number that matters) and
   buys address-stable parameters forever. Requires a burn API that copies into
   existing storage — `copy_from`/`*_assign` — to be checked.
2. **Two graphs, alternated by step parity.** If the cycle really is 2, capture
   one graph in each phase and alternate. Zero extra launches. This is a **bet
   on the pool's free-list order being LIFO**, i.e. on an implementation detail
   of the memory pool. It is testable, and if the test is green over 20 steps it
   is paid — but it is a bet, and the doc must say so.
3. **Nothing.** If the address does not repeat with any period, no stage that
   touches parameters is capturable, the whole lane is a null, and the honest
   deliverable is the launch counts that say the stages were not launch-bound
   after all.

**This is why instrumentation comes before capture**: option 1 costs P launches
per step and only pays if the stage has more than P launches in it, and P is
known from the module tree while L (launches per stage) is not.

## 5. The launch counter (the instrument)

Every kernel launch on this backend passes through **one** function:
`cubecl-cuda/src/compute/context.rs:512` `Context::execute_task` →
`cudarc::driver::result::launch_kernel` at `:535`. A single
`AtomicU64::fetch_add(1, Relaxed)` there is the whole instrument: one line at
the choke point, one getter, and every burn/cubecl/matmul/kernel launch on the
card is counted, including the ones inside autotuning and the fused kernels.

Counted on the **server** thread, so a host-side sample at a stage boundary is
a *lower bound* (it counts what the device thread has executed, not what the
host has enqueued). In a launch-bound loop the host runs ahead and the two
nearly coincide; the number is reported as a lower bound, not as a count.

## 6. The design — and what the probe (§8) then did to it

**Measured outcome, stated up front: escape 1 is dead, escape 3 is the design.**
§8.1 shows the parameter's address moves between steps, so a graph that is not
pinned reads a stale input and trains on two-step-old weights, silently. The
design below is therefore §6 **with the pin row mandatory**, not optional.

Two facts make the window the **whole step** and make the graph one dispatch:

- **The path is sync-free.** Every `into_scalar` / `into_data` /
  `try_into_scalar` in `train/src/optim.rs`, `burn-muon-plus/src/fused_kernels.rs`
  and `core/src/loop_block.rs` is inside a `mod tests` (optim.rs:551,
  fused_kernels.rs:208, loop_block.rs:616). The production
  fwd → bwd → opt → retr → ema path reads nothing back.
- **A replay is one dispatch**, whatever it contains — asserted, not assumed, in
  `graph_step.rs::a_pinned_parameter_is_address_stable_and_costs_one_copy`
  (the launch counter must not move across N replays).

So the step becomes:

```
per step:   H2D write x, y, hashed_ids  (outside the window — the only legal feed)
            replay(graph)               ← ONE dispatch: fwd+bwd+opt+retr+ema
per 100th:  log step, runs ungraphed, re-captures
```

with two pins that must be built for it to be correct:

| what | why it moves | the pin |
|---|---|---|
| every parameter | burn's optimizer is out-of-place (`adam.rs:88`), and the pool does **not** hand the block back — MEASURED (§8.1) | keep a master tensor per parameter, `master.inplace(\|_\| new)` after the step, re-point the model's `Param` at the master. Proven exact over 8 steps, one extra launch each (§8.2) |
| `x`, `y`, hashed ids | built fresh per step from host bytes | allocate once, `x.inplace(\|_\| Tensor::from_data(..))` per step (a H2D write into the same buffer, no extra launch) |

**`Tensor::inplace` is a soft contract, and the doc must say so**:
`burn-tensor-0.22.0-pre.4/src/tensor/api/base.rs:118-122` — *"This won't
necessarily reuse the same tensor data/buffer, but it should if there is no other
reference pointing to the same tensor."* A graph holds a raw pointer, not a burn
reference, so the refcount is 1 and the buffer should be reused. "Should" is not
"does", which is why the differential gate has to be the arbiter and not the
mechanism. The same clause is why the pin is a per-parameter `copy_into`-shaped
kernel in the probe rather than a burn call: the probe's pin is a launch the
test itself issues, so the price is counted, not inferred.

**The price, which decides everything**: the pin costs one extra device copy per
parameter per step. The optimizer banner says 54 groups
(`muon=15 qk=2 tables=1 rest=36`); the number that matters is the *tensor* count.
A stage is only worth capturing if it holds more launches than the pin costs, so
the launch count decides whether this lane is a win or a null — which is why the
counter is step one and not an afterthought.

## 7. Known risk, and why its failure is acceptable

**The autotuner.** cubecl benchmarks candidates with syncs, and a sync inside a
capture window is refused. A capture at a step where every kernel is already in
the tune cache should take the hit path (launch, no sync), but that is an
assumption about `cubecl-autotune`'s warm path, not something this repo's code
can prove — it is a crates.io crate, not vendored here.

The failure mode is the acceptable one: the server **refuses the capture** and
names what it refused, so the flag reports a loud error naming the stage rather
than a silently-wrong graph. It is a risk to the *speedup*, not to *correctness*.

The same applies to anything else the window contains: the first-iteration
compile, a pool growth the persistent pool cannot serve (rejected at
`stop_capture` by design, `capture.rs:110-120`), a handle write. All loud.

## 8. Results — 4/4 green, and one of them changes the design

`cargo test -p cubecl-cuda --test graph_step` on this box, 2026-10-01, commit
`wt/cuda-graph`. The card was free; one process at a time.

### 8.1 The address moves. A single graph is silently wrong.

```
ONE GRAPH, 3 STEPS: got -3, correct -6 -> the parameter address MOVED, silently: a pin is mandatory
```

Three steps, deltas 1/2/3, one graph replayed each step. The correct answer is
-6. The graph produced **-3**, which is exactly `0 - delta_of_the_last_step`:
the graph read the buffer the parameter lived in **at capture time** — frozen at
0 — and wrote the same output buffer every replay. So every replay recomputed
from a stale input.

**No error anywhere.** Not a NaN, not a shape check, not a counter. The run
would have trained, printed a plausible loss, and quietly learned from weights
two steps old. This is `8fa5d4c`'s shape exactly, which is the strongest
possible argument for the differential gate rather than a hand-written
assertion.

It also settles §4: escape 1 (no pin) is dead, and the reason is burn's
out-of-place optimizer plus a pool that does not hand the same block back.

### 8.2 Pinning is exact, and a replay is one dispatch

`a_pinned_parameter_is_address_stable_and_costs_one_copy`: a master buffer the
graph always reads, with the fresh update copied back into it. Over 8 steps the
pinned value equals the software run **exactly** (f32 equality, not a tolerance),
and the launch counter moves by **0** across 8 replays — the property that makes
the whole idea worth anything: *however many launches the graph contains, it
costs one dispatch.*

The price is one extra launch per parameter per step, counted rather than
asserted (`pin_price == 1`).

### 8.3 Re-capturing is a contract with two loud refusals

A trainer seam has to re-capture after every ungraphed step (a log step runs
outside the window). Two refusals stand in the way and **both name their
cause**, which is what makes the seam safe to write:

1. `begin_capture` after `end_capture` without a fresh `graph_prepare`:
   `begin_capture: call graph_prepare before starting a capture`
   (`cubecl-server/src/stream/capture.rs:437-453` returns the stream to
   `NoCapture`).
2. A second capture **while the first graph is alive** is rejected at
   `stop_capture`: `capture recorded 1 memory node(s): an allocation inside the
   capture window makes the graph un-relaunchable` — the first graph *pins* the
   slices its window allocated, so the persistent pool cannot serve the second
   one (`cubecl-cuda/src/compute/capture.rs:110-120`).

**The rule for the seam: destroy, prepare, capture — never capture over a live
graph.** `graph_destroy` releases the retained handles
(`compute/server.rs:303-321`), and after that the second capture is accepted
and replays correctly. All three assertions in
`recapturing_needs_the_old_graph_destroyed_and_a_fresh_prepare` are green.

### 8.4 The instrument is exact

`launch_counter_counts_every_launch`: 10 launches move the counter by exactly 10,
after a drain. One `fetch_add` at `context.rs:535`, the single place every
kernel launch on this backend passes through.

### 8.5 A false green, and what it cost

The first version of §8.1 **hand-wrote** its oracle: `-3.0` as "correct" for
three steps of deltas 1,2,3. The correct value is -6. The wrong oracle happened
to equal what a stale-pointer replay produces, so the test printed
`got -3, correct -3.0 -> the address is STABLE` and **passed, saying the exact
opposite of the truth**. An agent reading that line would have shipped a graph
that trains on stale weights.

The oracle is now computed by running the same steps without a graph. This is
the same defect class as the two the attnres lane found on 2026-09-30 (a gate
that cannot fail, and a gate that fails for an unrelated reason): **a
hand-written expected value is a guess wearing a test's clothes.**

### 8.6 Not taken: the two-graph parity escape

Available, and deliberately declined. If the cycle is 2, capturing one graph per
phase and alternating costs nothing — but it is a bet on the pool's free-list
order being LIFO, with no mechanism behind it, and pinning costs one launch per
parameter per step and is *provably* right (bit-exact over 8 steps). Where a bet
and a proof cost the same order of magnitude, take the proof.

### 8.7 What a launch costs on this box — the reason the card is idle 87%

`a_replay_costs_one_dispatch_not_n_launches`, 40,000 trivial launches, one at a
time, versus the same 40,000 as 20 replays of one 2,000-launch graph:

```
40,000 launches, one at a time:
  enqueue 1374.3ms (34.4 us each, HOST) + drain 2.9ms (0.1 us each, DEVICE)
the same 40,000 as 20 replays of one 2000-launch graph:
  enqueue 0.84ms (41.9 us each, HOST) + drain 55.2ms (DEVICE)
=> 24.6x end to end
```

**34.4 µs of HOST time per launch, against 0.1 µs of DEVICE time to run it.** The
GPU is ~340× faster than the host can feed it. That is §3.1's "mean utilisation
13.3%, 79% of samples ≤5%" reduced to a mechanism: the step time is not compute,
it is the host walking the runtime one launch at a time. And a replay collapses
that walk into one dispatch worth ~42 µs.

This is the strongest thing the lane produced, and it is a *mechanism*
measurement — no trainer, no model, no assumptions beyond "a bare cubecl
`::launch` of an 8-element kernel is the cheapest path through the runtime".

**What it implies, and how firmly.** A 500 ms step at 34.4 µs of host time per
launch is ~14,500 launches — but that is an **inference from a per-launch cost,
not a count of launches**, and the two are not the same thing: a burn op does
more host work per launch than a bare `::launch` (autotune lookup, tensor
allocation, dispatch bookkeeping), so the real `L` could be lower with each
launch costing more. The direction of the error is known, the size is not. `L`
itself is one `cubecl_launches()` sample away (§9) and is still unmeasured.

The pin's price against that: the optimizer banner names 54 groups, so `P` is
order 100 launches ≈ 3.4 ms of host time against a ~500 ms step. If `L` is
anywhere near the inference, the pin is ~1% of the step and the graph is the
step. **If `L` comes back in the hundreds, this is a null and the effort goes
elsewhere** — that is the number to take next, and it is the only thing left.

## 9. What is still missing: `L`, the launch count of a real step

`L` decides the lane, because the design is now:

```
cost of the graph  = P copies   (P = parameter tensors, one per parameter per step)
benefit of it     = L - 1      (L = launches inside the captured window)
win iff P < L - 1
```

and **P is knowable from the module tree while L is not**. If `L` is in the
hundreds the pin eats the win and this lane is a null; if `L` is in the tens of
thousands it is the biggest speedup available on this box.

**State: instrumented, protocol written, number not taken.** The counter is wired
into the timer line (`cubecl_launches()`, §11) and the run that reads it is one
command. What blocked it, in order, and none of it is the mechanism:

| blocker | what happened |
|---|---|
| disk | `/home` is at 97% with 16-17 GB free; the vendored cubecl build that produced §8 is 6.2 GB, and a cold `dormouse-train` CUDA build does not fit. Worked around by pointing `CARGO_TARGET_DIR` at the 1.8 TB data mount (673 GB free, 2.5 GB used by the release build). |
| the wrong feature | the first release build came out on `Flex(Cpu)` — `dormouse-cli`'s default is `cpu`, not `cuda` — and printed `launches=0`, which is the honest answer off CUDA and would have read as "the instrument is broken". The repo already had the right invocation in `.cargo/config.toml` (`build-probe`). |
| the shared card | **a 100,000-step production run took the card at 02:02 and is at step 23,100 with ~9.6 h left** (`~/logs/first_run_100000_1001_0202.log`, 450 ms/step). §1.5's "one GPU process at a time" is not negotiable and this lane's measurement is 3 minutes, so it waits for a gap that is not coming tonight. |
| the build lock | six lanes queued behind one CPU test holding the lock for 36 min; the CUDA release build this lane needed took **100 minutes to reach the front** and 7 min to compile. The lock is doing its job (no build storm) at the cost of a lane whose whole point is the card. |

**`L` is unaffected by GPU contention** — it is a count of launches and the launch
structure does not depend on what else is on the card. So the next card window,
however long it is, is only 3 minutes of work.

### 9.1 The cheapest possible way to finish this lane

The counter is already in the trainer and already on the timer line, and it costs
one relaxed atomic add per launch — so **the production binary can carry it at no
cost**, and the next 100k run's timer line then reports `L` for free:

```bash
# the CUDA release build this lane made is still on the data mount
# (CARGO_TARGET_DIR=/mnt/e43497ab-.../dt-graph), so there is no rebuild:
D=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain
/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/dt-graph/release/train \
  --data $D/real --eval $D/real_eval --eval-every 100000 \
  --preset small --batch 8 --seq-len 512 --no-engram \
  --steps 300 --log-every 100 --ckpt-name lcprobe- --timers --seed 1 2>&1 | grep timer
# L = (launches@300 - launches@100) / 200
```

Read it as: **if `L` > ~200 the pin is noise and the graph is the step; if `L`
comes back under ~200 the pin eats the win and this lane is a null.** Nothing
else in the repo has to change for that number to exist.

The honest summary of the lane so far: the mechanism is **proven to work and
proven to be unsafe without a pin**; the size of the win is **unmeasured**, and
it is one `cubecl_cuda::launches()` sample away on a card that is free.

## 10. The measurement protocol (fixed BEFORE the number, so it cannot be argued with later)

Same shape as the control, on a free card, one process at a time, **release**:

```
--preset small --batch 8 --seq-len 512 --no-engram --timers --log-every 100 --steps 300
```

- **step 0 is excluded from every number.** It is 17 s against a 0.5 s warm step
  and its launch count carries the autotuner's own candidate benchmarking
  (`CUBECL_AUTOTUNE_LEVEL` moves it), so `L` is read as
  `(launches@300 − launches@100) / 200` — the count between two warm steps.
- `L` is cumulative-since-start, so the per-step figure is a **difference of two
  timer lines**, not a rate. That is why the timer prints the raw count.
- the pin's price `P` is one launch per parameter tensor, from the same run.

## 11. Reproduction

```bash
# the mechanism, the instrument and the escapes (needs the card)
cd vendor/cubecl-fix
/home/sehaxe/dormouse/tools/build_lock.sh run graph -- cargo test -p cubecl-cuda --test graph_step -- --test-threads=1 --nocapture
/home/sehaxe/dormouse/tools/build_lock.sh run graph -- cargo test -p cubecl-cuda --test graph -- --test-threads=1

# L, the number that decides the lane (release + the cuda feature, quiet card)
cargo build-probe          # debug: launches are right, timings are not
D=/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain
./target/release/train --data $D/real --eval $D/real_eval --eval-every 100000 \
  --preset small --batch 8 --seq-len 512 --no-engram \
  --steps 300 --log-every 100 --ckpt-name lcprobe- --timers --seed 1 2>&1 | grep timer
# L = (launches@300 - launches@100) / 200.  Step 0 is excluded: its count
# carries the autotuner's own candidate benchmarking.
```

Five probe tests, each owning one thing: `launch_counter_counts_every_launch`
(the instrument against itself), `one_graph_every_step_is_reported_not_asserted`
(the divergence, reported with a computed oracle),
`recapturing_needs_the_old_graph_destroyed_and_a_fresh_prepare` (the lifecycle
contract, both refusals),
`a_pinned_parameter_is_address_stable_and_costs_one_copy` (the escape that works,
and its price), and `a_replay_costs_one_dispatch_not_n_launches` (the host/device
cost split). 5 passed, 0 failed, on this card, 2026-10-01.

`graph.rs` is the regression: 5 passed, 0 failed, unchanged by the counter.
