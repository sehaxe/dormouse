# fused/ rewrite plan: how burn writes fast CUDA, what we do wrong, and the A/B-or-death gate

Date: 2026-09-25. Inputs: burn @ /tmp/opencode/burn (shallow main), cubek 0.3.0-pre.3 +
cubecl 0.11.0-pre.3 from the cargo registry (what we compile against; cubecl-runtime/cubecl-cuda
from `vendor/cubecl-fix`), our `crates/dormouse-core/src/fused/` (mod 1629, backward 1768,
kernels 1553, tests 2487 = 7.4k LOC), `docs/archive/fused-verification-2026-09-23.md`,
`docs/archive/research/2026-09-23-fused-flagship50.md`. Flagship geometry for all numbers: b=10 t=512
(bt=5120), d=768, r=64, f=2048, v=256, nexp=3, N=8 iterations (configs/small.toml).

Measured baseline (docs/archive/research/2026-09-23-fused-flagship50.md:48-58): fused 9.21 s/step vs
burn 7.05 s/step — 1.3x SLOWER, both arms paying the identical burn-side aux stack.

---

## §1 burn's performance playbook

### 1.1 Nothing launches without autotune; nothing is hardcoded

Every kernel family goes through a `LocalTuner` keyed on a problem key (shapes, strides,
dtypes, device). Matmul: `matmul_autotune` builds a `TunableSet` of ~25 candidates in
priority groups — burn/crates/burn-cubecl/src/kernel/matmul/tune/base.rs:92-96 (static
`local_tuner!`), :102-211 (TuneGroups: accelerated/tma/gemv/unit with priority fns),
:610-625 (key from shapes+strides+dtypes). Reduce tunes routine × vectorization
(kernel/reduce/tune.rs:80-120). `CUBECL_AUTOTUNE_LEVEL` maps to
`AutotuneLevel {Minimal, Balanced, Extensive, Full}`
(cubecl-runtime-0.11.0-pre.3/src/config/autotune.rs:158-175) which only scales the
*benchmarking* of candidates: `BenchConfig` min 3 / max 10 samples, adaptive round-robin
with 1.5× elimination, 2-sample short-circuit written to the persistent cache
(autotune.rs:59-91). Launch configs come from device properties, never constants:
`CubeDim::new(&lhs.client, working_units)` + `calculate_cube_count_elemwise`
(burn-cubecl/src/kernel/binary_float.rs:83-86). Why it wins on sm_120: the tuner picks
tensor-core/TMA kernels per problem shape; a hand-picked config is right for one shape
and wrong for every other shape a run touches.

### 1.2 Comptime specialization: Line/vectorization and dtype promotion

Elementwise kernels are generic over a comptime line size `N: Size` and operate on
`Vector<C, N>` (binary_float.rs:15-34); the launch site picks
`min(max_vector_size(lhs), max_vector_size(rhs))` (binary_float.rs:62-67) so loads are
128-bit. Matmul promotes register dtypes to the accelerator's taste: "the selection paths
promote them with `adjust_dtypes` when the tile matmul needs an accelerator (f32 to tf32,
flex32 to f16)" (tune/base.rs:42-48) — the f32 problem gets TF32 tensor cores instead of
FFMA emulation. Candidates that can't run are demoted, never compiled to fail
(tune/base.rs:557-570).

### 1.3 Matmul anatomy (cubek): the thing our hand-rolled mm competes against

cubek-matmul 0.3.0-pre.3 is a multi-level routine family: global load (sync/async-cyclic/
async-strided/TMA loaders with barriers and swizzle validation) → shared-memory stage
(double/specialized/ordered buffering) → tile CMMA/MMA instruction. The double-buffered
plan alone instantiates `AsyncFullCyclicLoading` etc. with `ASYNC_COPY_WIDTH` vector
copies (cubek-matmul .../multi_level/components/global/read/strategy/async_full_cyclic.rs:1-50;
batch/double_buffering.rs:1-80). Plane ops (warp-level) appear as first-class partitioning:
`OrderedDoubleMma { partition_k: Some(2), row_count: Some(4), rows_per_plane: Some(2) }`
(tune/base.rs:408-418). Why it wins on sm_120: TMA + double-buffered smem keeps the
tensor cores fed; our 16×16 tile with two `sync_cube()` per k-step cannot.

### 1.4 Memory: pools, persistent allocations, graph capture

cubecl-runtime exposes `memory_persistent_allocation` — allocations made inside the
closure are routed `MemoryAllocationMode::Persistent` on the stream, immune to pool
cleanup (cubecl-runtime-0.11.0-pre.3/src/client.rs:483-512); burn-cubecl surfaces it as
`Backend::memory_persistent_allocations` (burn-cubecl/src/backend.rs:230-240) and the
fusion thread pins its whole stream via `allocation_mode` (burn-cubecl/src/fusion.rs:95-106).
Pre.3 already has CUDA graph capture: `graph_prepare` (persistent pool + recording),
`start_capture`, replay with inputs refreshed in-place via `client.write`
(client.rs:494-520, :958-975). Why it matters: AGENTS.md records our step loop as
launch-bound — replaying a captured step graph is the endgame for launch-bound steps.

### 1.5 The fusion backend: what it actually does

burn-fusion traces ops into a stream; fusers compete for segments. The cubecl elementwise
fuser fuses chains into ONE kernel with `FuseSettings { broadcast: true, inplace: true,
vectorization: Activated, ref_layout: Any, choose_output_layout: true }` capped by
`props.hardware.max_bindings` (burn-cubecl-fusion/src/optim/elemwise/fuser.rs:27-48).
In-place aliasing inside kernels (`can_mut_broadcast` + `as_linear_view_alias`,
binary_float.rs:89-101) removes output allocations. There is also a matmul fuser, reduce
fusers, and an NHWC-relayout fuser (burn-cubecl-fusion/src/optim/{matmul,reduce,nhwc_relayout}).
Custom/user fusions plug in via the `OptimizationProvider` registry
(burn-cubecl/src/fusion/registry.rs:30-54: "Register a provider with `register` **at the
start of the program**").

### 1.6 Extension crates and the fusion flag (the #5673 story)

Backend-extension ops (our vendored burn-kda/msa/engram/rmsnorm/spectral/dspark/jepa
under vendor/dormouse-fused/crates/) cross the dispatch layer through
`#[backend_extension]`-generated code (burn-backend-extension/src/lib.rs:65-86). To ride
the fusion stream an extension registers a metadata callback describing outputs from
input `TensorSpec`s — `FusionValueAdapter` (burn-fusion/src/custom.rs:6-110); execution
is deferred, opaque, and validate-on-publish. The pre.3→pre.4 delta that broke extension
crates: pre.4's `DispatchKindConversion` gained a `DispatchAutodiffContext` check —
`try_into_backend` now errors unless the tensor's autodiff context is explicitly
`Disabled` (diff of burn-dispatch-0.22.0-pre.3/src/tensor.rs:495-545 vs
pre.4/src/tensor.rs:464-515). Generated extension glue must be regenerated against that
contract — exactly what the pre.4 merge (debugger agent, #5673) fixes. Sync discipline
underpinning all of it: burn syncs only when a value is read back (logging, optimizer
CPU steps); kernels are enqueued and the pipeline stays full.

---

## §2 Our fused/ violations, ranked by expected step-time impact

1. **`arms_inner_adjoint` re-runs the arms forward + builds an inner AD graph PER
   ITERATION in backward** — backward.rs:152-306 (the function), called per reverse
   iteration at :1413-1425; each call does a full arms forward (router+sigmoid+KDA
   `forward_train_state`+MSA+Engram, :213-250), `loss_n.backward()` (:251), and TWO
   device syncs around it (:1408, :1426). N=8 ⇒ 8 extra arms forwards, 8 AD graph
   builds, 16 pipeline drains per step. This is the diagnosed flagship killer
   (docs/archive/fused-verification-2026-09-23.md:78-86).
2. **Grad registration is a D2H→host→H2D round trip of every weight grad, every step**
   — backward.rs:984-1036 (`ponder_backward`) and :1725-1766 (`..._arms_direct`): ~26
   grads `client.read()` (blocking) then `Tensor::from_data` re-upload, total ~2×30 MB
   of PCIe traffic + ~52 blocking transactions + serialization per step. It exists to
   dodge a buffer-aliasing bug ("foreign bytes", backward.rs:984-989) — a workaround
   promoted into the hot path.
3. **Two full `sync()` fences per step minimum, plus per-iteration drains** —
   mod.rs:850/1115 (plain), :1503/1615 (arms); plus #1's 16. Each drain empties the
   launch pipeline; at ~45 launches/iteration the CPU can never run ahead.
4. **Per-step host serialization of a full LoopBlock** — mod.rs:1305-1314
   (`ModuleRecord::from_bytes` + `load_record` every forward) and backward.rs:1197-1218
   (again per backward, with a `require_grad` mapper). Serializes, deserializes, and
   re-allocates the module's params on device twice per step.
5. **fp32 FFMA matmuls, no tensor cores, no autotune** — `mm_kernel`: one thread per
   output cell, serial k, ternary computed per element (kernels.rs:23-89);
   `mm_tiled_kernel`: BLOCK=16, 2 `sync_cube()` per k-tile, no vector load, no async
   copy (kernels.rs:91-195); heuristic switch at mod.rs:189-268. ~46 mm launches per
   iteration forward (experts 4×3 + controller + out_proj + lm×2 + router 2) ⇒ ~370
   forward mms/step at N=8, all on the slow path. Burn's own path hits cubek tensor
   cores for the same shapes (§1.3).
6. **Zero vectorization anywhere** — every kernel indexes scalars `&[F]`; no `Line`/`Vector`
   args in kernels.rs (contrast burn-cubecl binary_float.rs:15-34). rmsnorm/absmean/l2norm
   use 32-thread single-cube loops with `#[unroll] for u in 0..32`
   (kernels.rs:207-227, 230-262) — a warp, not a block, on a GPU with 128-lane schedulers.
7. **Hardcoded launch configs** — `UNITS = CubeDim::new_3d(32,1,1)`, `EW = (256,1,1)`
   (mod.rs:59-60); `CubeCount::Static(...,256)` everywhere; `lam_bwd_kernel` silently
   caps b≤32 with only a debug_assert (backward.rs:455-457); ternarize branches inside
   the k-loop re-read `bm[0]` per element (kernels.rs:62-78, 156-172).
8. **Cargo-cult launch in the hot path** — `launch_gdn2_chunk_dummy` is called once per
   factor (8×/step) in the fence loop (mod.rs:1505); it allocates a tensor and
   string-compiles kernel names (kernels.rs:1539-1552). Pure checker-compliance junk.
9. **Fresh allocations for ~200 buffers per step** — the `mk`/`mz` closures build new
   `Tensor::empty` per iteration per step (mod.rs:780-841, 1426-1501), again in backward
   (alloc closures at backward.rs:339-343, 1092-1096), all kept alive via `keep1/keep2`
   vecs. Burn sizes workspaces once and reuses (§1.4).
10. **Redundant copies + recomputed means in the arms forward** — every arm result is
    computed in its own temp then `copy_kernel`'d into the flat buffer (mod.rs:1530,
    1537, 1544, 1553-1558); `router_gate_cube` re-extracts factors and relaunches
    absmean every call (kernels.rs:1486-1507) though `fac_cuda` already computed them.
11. **The "direct" adjoints are unwired and unusable as written** — zero callers
    (grep over crates/, 2026-09-25). Each rebuilds the module from serialized bytes,
    reads inputs to host (`into_data`), re-uploads, runs an inner-AD forward+backward:
    kernels.rs:1317-1360 (kda), :1376-1404 (msa), :1441-1472 (engram). They move the
    round-trip problem around; they don't seed a real adjoint.
12. **`ponder_backward` and `ponder_backward_arms_direct` are 90% duplicates**
    (backward.rs:310-1038 vs 1047-1768) — every fix must be applied twice; they have
    already drifted (grad-snapshot bug handling differs).
13. **f32-only, `launch_unchecked::<f32>` hardcoded at every call site** — blocks the
    bf16 storage path (AGENTS.md kernel plan) and the Fp8-forward A/B inside fused;
    docs/archive/fused-verification-2026-09-23.md:111-113 lists it as plan M5.
14. **Fusion-flag blocker** — enabling burn's `fusion` feature fails to compile on
    pre.3 because the vendored extension crates' `#[backend_extension]` glue predates
    the `DispatchAutodiffContext` contract (§1.6). Unlocked by the pre.4 merge; our op
    also sits OUTSIDE any fusion stream, so once fusion is on, every tensor flowing
    into/out of the op still terminates fusion traces at its boundaries.

---

## §3 The rewrite plan

### 3.1 Keep / rewrite / delete per file

**kernels.rs**
- KEEP (rewrite bodies): rmsnorm, absmean, rowmean_t, halt_fwd, sigsel(_arms), silu(bwd),
  col_scale, sum_ds, halting, rec, kl, ce/ceb, dlogits, residual(_bwd), lam_bwd —
  the loop-specific math with no burn equivalent. Add `Line<f32,4>` vector args +
  device-derived CubeDim; replace 32-thread reduction cubes with plane reductions
  (`cubek::reduce` PlaneStrategy pattern, kernel/reduce/tune.rs:86-90).
- DELETE: `mm_kernel` + `mm_tiled_kernel` + the `launch_mm` heuristic (mod.rs:189-268).
  Ternarize factors ONCE per step into materialized buffers (absmean + a new ternarize
  kernel — both trivial elementwise over ≤ d·r elements), then launch matmuls through
  burn's own `matmul_autotune` (kernel/matmul/tune/base.rs:75) with f32 in / f32 out;
  the tuner's `adjust_dtypes` picks TF32-CMMA where legal. Exactness guard: the f64
  bisect tests already pin tolerances (rel 5e-2 grad, 7.8e-7 fwd); if TF32 breaks them,
  restrict the tunable set to non-accelerated candidates — still faster than our 16×16
  tile via double buffering.
- DELETE: `launch_gdn2_chunk_dummy` and the arms::* byte-serialization helpers
  (kernels.rs:1317-1472, 1539-1552) — replaced by §3.2.
- NEW: `ternarize_kernel` (masters + mean → ±mean·1{·>0.7·mean}), reused by fwd+bwd.

**mod.rs**
- KEEP: the single-op plumbing — `Backward<CB, 61/62>` node, parent ordering, `PonderInputs/
  Outputs`, the flat-output layout, `split_outputs`. This one-node-per-step design is the
  part burn cannot do natively (it skips 8× graph builds) and is worth keeping.
- REWRITE: allocation — build the whole workspace ONCE per shape-key in a
  `static WORKSPACES: OnceLock<Mutex<HashMap<Key, Workspace>>>` (Key = the geometric
  tuple) or inside `client.memory_persistent_allocation` (§1.4); per-step code becomes
  buffer lookups. Removes ~200 allocs + the keep-alive vecs.
- DELETE: per-step `ModuleRecord::from_bytes` of LoopBlock (mod.rs:1305-1314) — hoist to
  first call (workspace init) and store the bare module in the workspace; the weights it
  exposes are re-read live from the model each step only where the op actually needs them.
- REWRITE: the fence policy — one `sync` before the first raw launch (unavoidable:
  optim/retract wrote masters on burn streams) and one after the last; nothing else.
- DELETE: `launch_gdn2_chunk_dummy` call (mod.rs:1505).

**backward.rs**
- REWRITE: merge the two functions into one `ponder_backward(cfg_kind)`; the arms path
  adds ~40 lines, not 700.
- REWRITE: the arms adjoint → direct adjoint kernels (§3.2); delete `arms_inner_adjoint`,
  the 16 syncs, and the inner graphs.
- DELETE: the grad-snapshot round trip (backward.rs:984-1036, 1725-1766) — register the
  device buffers directly (`grads.register::<CB>(node.id, flat_relabel)`); the "foreign
  bytes" bug it dodges is the dummy-parent aliasing at mod.rs:764-771 (dummy nodes cloned
  from the x parent) — root-fix by registering only real parents, or verify it was a
  pre-fix artifact. If a readback is ever needed (debug), gate it on `DM_FUSED_DEBUG`.

### 3.2 Direct-adjoint wiring (seed from saved forward buffers)

The arms forward already saves per iteration into the workspace: `normed` (mod.rs:1447),
`h_ctx` (:1446), gate columns in `raw` (:1451), and the arm outputs `kda_vec/msa_vec/
engram_vec/attn_vec/gate_vec` (mod.rs:1475-1489). Missing: **KDA's recurrent state `S`** —
`forward_train_state` returns `(out, S)` (vendor/dormouse-fused/crates/burn-kda/src/lib.rs:509,
530) and the raw forward drops it (`forward_train`, mod.rs:1529). Add an `S` buffer per
iteration (b×heads×state_dim, small vs bt×d), filled by switching the raw KDA call to
`forward_train_state` + the same copy pattern.

Backward per reverse iteration then launches, all raw, no AD, no syncs:
1. `residual_bwd_kernel` → `dy` (unchanged, backward.rs:1330-1344).
2. KDA adjoint: burn-kda's own `gdn2_chunk_intra_adjoint` + `gdn2_chunk_inter_adjoint`
   seeded with `dy` and saved `S` — the exact kernels the module's AD backward runs; add
   thin raw-launch wrappers in burn-kda (vendored, we own it) mirroring its AD call
   signature (dY, x, S, weights → dX + weight-grad partials into accumulators).
3. MSA adjoint: `msa_backward_kernel` via the same wrapper trick on burn-msa.
4. Engram: host-rows mode is a gather + gate — scatter-add `dy` through the row indices
   (new small kernel; the rows grad lands on parent `rows_idx`, backward.rs:1077-1088
   unchanged) and `d(h_ctx)` = `dy · w_mem ⊙ gate'`.
5. Gate jacobian (`d_raw[:,0:2]`) and blend: two elementwise kernels (the math is already
   written in `arms_inner_adjoint`'s contract, backward.rs:122-129).
6. Add `dX_kda + dX_msa` into `dx`, `d(h_ctx)` into `dh_ctx` — the same add sites that
   today consume `adj.d_normed/.d_hctx` (backward.rs:1428-1440).

Cost: ~10-14 launches per iteration replace a full arms forward + inner graph + 2 syncs.
The arm-weight grads accumulate into the same `out[]` slots — `ArmLeaves` node capture
(mod.rs:453-479) survives unchanged.

### 3.3 Fusion-flag integration (on pre.4)

1. Debugger agent merges pre.4 (unblocks `DispatchKindConversion` + `DispatchAutodiffContext`;
   §1.6, memory note 2026-09-25: upstream pre.4 is green on sm_120, our fork port is the
   problem).
2. Enable `burn-cuda/fusion` for dormouse-train. Immediately the burn arm's elementwise
   chains (aux heads, JEPA/DSpark/KoLeo, optimizer elementwise, loss build) fuse via
   `ElementWiseFuser` with inplace+vectorization (§1.5) — **the bar rises**: the A/B
   baseline is now fusion-on burn, not today's 7.05.
3. Our op stays outside the fusion stream (it's a raw-launch op under one AD node). That
   is acceptable while DM_FUSED is measured; IF it survives §4, register it as an
   `OptimizationProvider` (registry.rs:30-54) so it participates in stream execution.
   Do not build this before the §4 gate says it deserves to exist.
4. bf16: all kernels templated `<F: Float>` already take F params — the blocker is the
   hardcoded `::<f32>` at every launch site and the fp32-only KDA/MSA kernels (AGENTS.md).
   Template the launch helper on storage dtype with fp32 casts before Linears (the bf16
   compute rule); do this AFTER the §4 gate, not before.

### 3.4 Expected step-time model (falsifiable)

Fused today 9.21 vs burn 7.05 → Δ=2.16 s/step. Attribution:
- 8 arms re-forwards + 8 inner graph builds + 16 drains (§2.1): the arms forward inside
  the burn path costs O(0.5-1 s) (KDA is the dominant arm); paying it 8× in backward
  plus graph-build CPU time plausibly accounts for ~1-1.5 s of the delta.
- Grad readback round trip (§2.2): ~30 blocking D2H + 30 H2D ≈ 100-300 ms.
- Naive mms (§2.5): the mm work itself is a fraction of step time (weights are small vs
  activations), but the 16×16-tile kernel at bt=5120,k=768 is ~5-10× off cubek's
  tensor-core kernels; worth ~0.3-0.8 s of GPU time, partially hidden by enqueue.
- Syncs/dummies/copies/allocs (§2.3,8,9,10): ~0.1-0.3 s combined.
Post-rewrite prediction: fused op ≈ burn's own loop cost minus graph-build overhead plus
adjoint kernels ≈ **5.5-6.5 s/step** at the flagship — i.e. fused must win by SKIPPING
burn's per-op graph construction (~1000 nodes/step) and aux-head fusion gains, not by
kernel heroics. If the measured number is not below the fusion-on burn arm, §4 applies.

---

## §4 Decision tree: A/B or death

The 50-step flagship A/B (small, batch 10, s512, 48M engram-ram, host-adam every step,
aux on, sequential arms, quiet box, `free -g` gate) is the only judge. Same commit for
both arms; the burn arm runs with the fusion feature ON once pre.4 lands.

- **Gate 0 (pre-work sanity)**: after the pre.4 merge, re-run the A/B once with DM_FUSED=0
  vs 1 unchanged code. If burn+fusion has already dropped far enough that fused's
  structural advantage (one node vs ~1000-node graph) cannot plausibly close, skip to
  delete.
- **Rung 1 — adjoint wiring + sync/readback removal (§3.2, §3.1 backward/mod edits; ~2-3
  days)**: must show ≥ 1.0 s/step improvement at flagship. If fused still ≥ burn →
  **delete**.
- **Rung 2 — matmul via cubek + persistent workspace + vectorized row kernels (§3.1;
  ~3-5 days)**: if fused still ≥ burn → **delete**.
- **Rung 3 — bf16 storage + graph capture**: only if Rung 2 lands within 10% ABOVE burn —
  one bounded attempt (CUDA graph replay of the step is the one lever left for a
  launch-bound step, §1.4), else **delete**.
- **Delete means delete**: all of crates/dormouse-core/src/fused/ (7.4k LOC incl.
  tests), the DM_FUSED gate in train/src/lib.rs, and the flag plumbing; keep
  docs/archive/fused-verification-2026-09-23.md + docs/archive/research/2026-09-23-fused-flagship50.md as
  the record. The one thing worth salvaging into burn-kda/msa upstream: the raw
  adjoint-launch wrappers from Rung 1 (they're useful without our op).

Rationale: burn's autodiff path is a moving target (fusion backend, cubek kernels,
pre.4 graph capture) maintained by upstream; our fused/ is 7.4k LOC maintained by us
and currently loses. Every engineering hour spent inside fused/ must buy a win that
survives the next burn upgrade; two rungs without one is the signal.

---

## §5 References

Burn (shallow clone /tmp/opencode/burn, main):
- Matmul autotune + tune groups + TF32 promotion: crates/burn-cubecl/src/kernel/matmul/tune/base.rs:42-68,92-211,408-443,557-570,610-625
- Elementwise vectorization + in-place aliasing + device-derived launch: crates/burn-cubecl/src/kernel/binary_float.rs:15-34,62-67,83-101
- Reduce autotune (routine × vectorization, plane strategy): crates/burn-cubecl/src/kernel/reduce/tune.rs:28-120
- Memory hooks: crates/burn-cubecl/src/backend.rs:230-290; persistent alloc: cubecl-runtime-0.11.0-pre.3/src/client.rs:483-512; graph capture: :958-975
- Fusion: crates/burn-cubecl-fusion/src/optim/elemwise/fuser.rs:27-48; registry: crates/burn-cubecl/src/fusion/registry.rs:30-54; stream glue: crates/burn-cubecl/src/fusion.rs:18-108
- Custom fusion ops for extension crates: crates/burn-fusion/src/custom.rs:6-110; backend_extension: crates/burn-backend-extension/src/lib.rs:65-86
- pre.3→pre.4 DispatchKindConversion delta: burn-dispatch-0.22.0-pre.3/src/tensor.rs:495-545 vs -pre.4/src/tensor.rs:464-515
- cubek matmul internals: registry cubek-matmul-0.3.0-pre.3/src/multi_level/{routines/batch/double_buffering.rs:1-80, components/global/read/strategy/async_full_cyclic.rs:1-50}
- Autotune levels: cubecl-runtime-0.11.0-pre.3/src/config/autotune.rs:36-91,158-175

Dormouse:
- fused/ sources: crates/dormouse-core/src/fused/{mod.rs,backward.rs,kernels.rs} (line refs inline, §2/§3)
- Flagship A/B: docs/archive/research/2026-09-23-fused-flagship50.md:48-58; grad-coverage + structural diagnosis: docs/archive/fused-verification-2026-09-23.md:57-116
- KDA state API: vendor/dormouse-fused/crates/burn-kda/src/lib.rs:503-530
- GPU constraints: AGENTS.md (4D-slice ban, bf16 fp32-cast rule, KDA f32-only, launch-bound step, pool doctrine)
- Preset: configs/small.toml
