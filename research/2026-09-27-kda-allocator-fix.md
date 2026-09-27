# KDA at the production shape: the allocator is NOT the cost

Date: 2026-09-27. Subagent deliverable. Companion to (and correction of)
`research/2026-09-27-kda-sota-ceiling.md`.

Shape: `b=10 t=512 d=768 h=12 K=V=64 chunk=16 fp32`, the dormouse `small` preset, on
`Autodiff<Cuda, BalancedCheckpointing>` — verbatim the backend in
`crates/dormouse-train/src/lib.rs:33`, which is what every KDA call in
`loop_block.rs:373` runs on (full sequence, no routed subset).
Binary: `crates/dormouse-train/examples/kda_alloc_probe.rs`, built in the
trainer's own dependency graph so it links the **patched** `vendor/cubecl-fix`
cubecl. Every boundary is a real `Client::sync()` barrier.

---

## 0. TL;DR

1. **§2.5's allocator hypothesis is falsified on this box.** Warm fwd+bwd is
   **470-635 ms per KDA call, of which the GPU is 1-20 ms: 99% host time.**
2. **Pool misses cost nothing measurable.** `client.memory_cleanup()` returned
   **887 MB** to the driver; the next call re-reserved **+887.2 MB** and ran in
   **561 ms** — indistinguishable from the 519 ms and 470 ms warm calls. The
   "≈1 ms per pool-miss `cudaMalloc` → ~1000 ms/step" arithmetic does not hold
   here (the driver's own caching absorbs the misses).
3. **What the pool does cost is VRAM, not time:** +918 MB reserved per
   forward-only call, never reused (`34 → 4284` live allocations over 5 calls,
   reserved `914 → 1833 MB`). That is the long-run-OOM mechanism AGENTS.md
   warns about, and it is real — but it is not the 1445 ms.
4. **The trainer was never running the fused kernels.** burn-kda's dispatch was
   `TypeId::of::<B>() == TypeId::of::<Autodiff<CudaBare>>()`, i.e. against
   *NoCheckpointing*, which `Autodiff<Cuda, BalancedCheckpointing>` fails by
   construction. Every production step ran the ~150-ops-per-chunk **tensor**
   path. (Independently confirmed in the fix's own docstring, now in
   `burn-gdn2/src/cuda_dispatch.rs`: "which is what kept the fused path out of
   every training step".)
5. **Engaging the fused node is worth ~1.57x on the KDA call** (505 → 321 ms,
   warm, b=10). That is the fix now in the tree; it is a real win and it is
   *not* the 100x the "1300x gap" framing implies.

---

## 1. The falsification: cold vs warm vs after-cleanup

`kda_alloc_probe`, one fwd+bwd per row, `Client::sync()` at every boundary.
`call 1` is the first call into an empty pool; `call 4` runs after every cached
page was handed back to the driver.

| call | fwd host | fwd GPU | fwd | bwd host | bwd GPU | bwd | pair |
|---|---|---|---|---|---|---|---|
| 1 cold pool | 58 459.6 | 348.6 | 58 808.2 | 94 079.1 | 111.3 | 94 190.4 | 153 000 |
| 2 warm | 222.2 | 1.0 | 223.2 | 401.4 | 10.7 | 412.1 | 635.3 |
| 3 warm | 143.4 | 1.2 | 144.6 | 363.7 | 11.1 | 374.8 | 519.4 |
| **4 after cleanup (887 MB re-missed)** | 140.3 | 1.9 | 142.2 | 398.8 | 20.2 | 419.0 | **561.2** |
| 5 warm again | 119.7 | 1.1 | 120.7 | 338.6 | 12.0 | 350.6 | 470.6 |

*(ms. The cold row is dominated by kernel compilation and autotuning, not
allocation: its GPU column is 348 ms, its host column 152 s.)*

**Read:** rows 3, 4 and 5 are the same call. Row 4 is the test the hypothesis
predicts a difference on, and there is none. Whatever the 470-635 ms is, it is
not `cudaMalloc`. And the `fwd GPU`/`bwd GPU` columns put **1.1-20.2 ms** of the
**470-635 ms** on the device: the GPU is idle 95-99% of the time, so the
"fused kernel is 1300x off its own roofline" question is downstream of a host
problem and cannot be the whole story.

## 2. The allocator accounting

Same rows, `client.memory_usage()` at the same three points.

| call | allocs before → after fwd → after bwd | in use (MB) | reserved (MB) | Δ reserved |
|---|---|---|---|---|
| 1 cold | 3 → 869 → 44 | 15.0 → 301.8 → 68.3 | 15.6 → 509.7 → 898.9 | +883.3 |
| 2 warm | 30 → 883 → 50 | 26.7 → 301.8 → 68.3 | 898.9 → 914.4 | +15.6 |
| 3 warm | 36 → 886 → 50 | 26.7 → 301.8 → 68.3 | 914.4 (flat) | +0.0 |
| 4 after cleanup | 12 → 871 → 49 | 26.7 → 301.8 → 68.3 | 27.2 → 914.4 | **+887.2** |
| 5 warm | 35 → 885 → 49 | 26.7 → 301.8 → 68.3 | 914.4 (flat) | +0.0 |

* `memory_cleanup()`: reserved `914.4 → 27.2 MB`, 24 allocations dropped.
* 5 warm forward-only calls: `147.7 ms/call`, allocs `34 → 4284` (**+4250**),
  reserved `914.4 → 1832.8 MB` (**+918.3**).

**Per call: ~840-856 fresh allocations and 301.8 MB live at the forward's peak,
894 MB reserved.** So:

* The doc's **248 MB** of scratch is right in kind and close in size (301.8 MB
  in use, 914 MB reserved including the pool's retained pages).
* The doc's **"17 fresh tensors"** is wrong: it is ~850 allocation *slices* per
  forward. Seventeen is the count of the kernel's own export buffers; the
  tensor path's slicing/repeat/permute chain is the rest.
* The pool **never reuses** its pages across calls (+918 MB per forward-only
  call, 5 calls deep). That is the AGENTS.md high-water behaviour, and it is
  the one thing here that bites: it is why `train_loop` must call
  `memory_cleanup()`, and it is a long-run OOM, not a per-step tax.

## 3. The path A/B — what the fix is worth

The `TypeId` gate means the trainer ran the tensor chunk path. The same call
behind one fused autodiff node, warm, `b=10 t=512`, measured in
`vendor/burn-fused/crates/burn-kda/examples/kda_step_probe.rs`:

| path | fwd | bwd | pair |
|---|---|---|---|
| `Autodiff<Cuda, Balanced>` → tensor chunk path (**what the trainer ran**) | 159-273 | 345-1126 | ~505 |
| `Autodiff<Cuda, NoCheckpointing>` → fused kernels, one node | 110-115 | 210-244 | **~321** |

**1.57x**, and per-chunk scaling (32 chunks vs 4 at fixed batch) localises it:
tensor **10.7 ms/chunk**, fused **6.3 ms/chunk** — a per-chunk host-dispatch
cost in both, ~60% lower on the fused path. The fused forward is also the
*stable* one (110-115 ms across 5 runs vs 159-273 for the tensor path).

Two caveats, stated because they matter:

* This A/B is from the `vendor/burn-fused` build, which links **unpatched
  registry cubecl** (that workspace has no `[patch.crates-io]`). The ratio is
  the transferable number; the absolutes are not. The §1/§2 tables are from
  the dormouse build (patched) because the allocator is precisely what differs.
* The end-to-end step-time before/after was **not** measured: the trainer
  rebuild is 5-12 min in a workspace three other agents are editing, and the GPU
  never had a clean window (another agent's bench loop had it continuously).
  Command, ready to run:
  `./target/release/train --data <sharded dir> --preset small --batch 10 --seq-len 512 --no-engram --quant fp32 --timers --log-every 100 --steps 60 --ckpt-name kdafix --ckpt-dir /tmp/opencode/kdabench`
  and read `timer step 50` before vs after.

## 4. Mechanism

burn pre.4 deleted the generic autodiff route, so every fused kernel in this
library gated on `TypeId` — and an autodiff wrapper is a *different type per
checkpointing strategy*. The trainer's `Autodiff<Cuda, BalancedCheckpointing>`
therefore never matched `Autodiff<CudaBare>`, the chunk loop fell through to
`chunk_wy_forward`'s tensor path (~150 host-dispatched ops per 16-token chunk
plus their backward), and nothing failed loudly: the tensor path computes the
same function, so a right answer and a fallback look identical. That is now
fixed in `burn-gdn2/src/cuda_dispatch.rs` (`backend_matches` via
`Backend::name`, a legal `strip`/`rebuild` across the dispatch layer's context
gate, and a node rebuilt with the caller's strategy) and wired through
`burn-kda/src/fused.rs:108`.

The remaining ~321 ms per call is still ~95% host. It is the *op count*, not
the kernel: the fused node itself is ~2 launches plus ~30 tensor ops, and the
rest is `project()`'s slicing/permute/`repeat` chain and the autodiff walk.

## 5. Corrections to `2026-09-27-kda-sota-ceiling.md`

That document is now the reference; these claims in it are false and must not
be relied on.

| claim | status |
|---|---|
| §0/§2.4/§3(a): "the fused kernels **are engaged** (3 launches fwd, 2 bwd) … their floor is 80-100 µs, we measure 120 ms, a ~1300x gap" | **False for the trainer's backend.** The `TypeId` gate made them unreachable; production ran the tensor path. The 1300x was never a kernel gap. |
| §2.5: "`fused_chunk_forward_scratch` allocates 17 fresh tensors totalling 248 MB … at ~1 GB per step" | **Bytes right, count wrong.** ~850 allocation slices per forward; 301.8 MB in use / 914 MB reserved per call. |
| §2.5: "At 1 ms per pool-miss `cudaMalloc`, 1 GB of misses per step is ~1000 ms. That is the 1445 ms." | **Falsified.** 887 MB of deliberate re-misses (after `memory_cleanup()`) cost ~0 ms; warm steady state is identical with and without them. |
| §2.5: the hypothesis "explains 4x tokens costs 1.47x time, the ~465 ms fixed cost, and the 0.23 ms-vs-1445 ms contradiction" | **Not needed.** A single measurement explains all three: the cost is host-side op dispatch, which is per-op and per-chunk, not per-byte. |
| §2.4 roofline framing (fraction of HBM roofline) | **Kept, but demoted.** With 99% of the time on the host, the binding constraint is dispatch, and the honest comparison is against a per-op host cost. The kernel is still worth ~1.57x once it is actually reached. |
| §4's `max_iter` lever | **Unchanged and still the biggest single one** (3 of 4 full-sequence KDA passes per step are removable). Nothing in my data argues against it. |

## 6. Ranked next options

1. **The dispatch fix (landed in `burn-gdn2` + `burn-kda`)** — worth the 1.57x
   above; measure the step end-to-end before believing it.
2. **CUDA-graph the step** (AGENTS.md's own "capture the step in a CUDA graph",
   §3 of the design playbook). With 99% host dispatch and ~4.8k ops per chunk
   pass, this is the structural fix for what remains; more kernel work is not.
3. **`max_iter` 4 → fewer full-sequence KDA passes.** Unchanged, still the
   largest single lever, still cheap.
4. **Drop the 63 MB per-chunk state export.** Correct for VRAM, which §2 shows
   is where the pool actually hurts; worth ~0 for time on this box.
5. **Pool reuse** (the +918 MB/call growth). Correct for long runs; ~0 for step
   time. `memory_cleanup()` cadence is the current mitigation.

## 7. Method notes, for whoever re-runs this

* **Run the harness inside the dormouse workspace.** In `vendor/burn-fused`,
  one elementwise `mul` on a 24 576-element tensor measures **295 µs**; on the
  patched build the same class of op is far cheaper. Both builds' absolute
  numbers are unusable for op-count conclusions if you mix them, and the probe
  docstrings now say so.
* **`Instant` around a cubecl op measures the host.** The old probe's
  "0.23 ms fused forward" was launch time. `into_data()` is not a reliable
  barrier either; `Client::sync()` is. Both facts are why the tables above
  split host from GPU instead of reporting one number.
* One process on the GPU. A second process's 200-900 MB probe was enough to
  make this one OOM mid-ramp on a card with 14.6 GB free.
