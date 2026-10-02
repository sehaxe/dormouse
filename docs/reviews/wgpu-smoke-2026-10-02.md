# wgpu cross-vendor smoke — our cubecl kernels on AMD/Intel-class runtimes

**Date:** 2026-10-02 · **Lane:** goal.md «компилируются/бегут ли наши cubecl-ядра
на wgpu (AMD/Intel)?» — the question had never been asked, let alone answered.
**Verdict: YES for the kernel dialect we use — measured, not assumed. One of our
fused kernels (RMSNorm) compiles through cubecl-wgpu's WGSL pipeline and runs
**bit-correct** on this box under software Vulkan (lavapipe). The landed gate is
`vendor/dormouse-fused/crates/burn-rmsnorm/tests/rmsnorm_kernel_wgpu.rs`
(`cargo test -p burn-rmsnorm --features wgpu`).**

## What was measured (this box, 2026-10-02, CPU-adapter Vulkan only)

The GPU was left alone (owner's daytime discipline). All runs target
`WgpuDeviceKind::Cpu` — **lavapipe**, Mesa's software Vulkan driver, which is
exactly the point: correctness is vendor-independent, speed is not claimed.

**Recovery provenance.** This lane recovered a dead session's worktree
(`wt/wgpu`): its report claimed the stages below PASS, but the gate file it
carried had **never compiled** (three type errors, found at the first real
build), and its committed form raced its own process-global seam counters
between two `#[test]`s. Both are fixed here; the committed evidence is the gate
test's green run on 2026-10-02 (stages 1a/1b/2 are the dead session's scratch
probes — sources not preserved, mechanism story only — while stage 3 is the
committed gate, which is the instrument that survives). A recovered "PASS"
without a green run is a hypothesis; this section is written so the next reader
does not have to trust one.

| stage | what | result |
|---|---|---|
| 1a | raw `cubecl-wgpu` runtime, trivial `#[cube]` kernel (`add_one`, 16 lanes) — *scratch probe* | PASS |
| 1b | the **exact reduction shape of `rmsnorm_kernel`**: `Shared::new_slice` sized from `#[comptime]`, strided load, comptime-unrolled binary-tree reduction, `sync_cube()` per step — *scratch probe* | PASS, 8.0 exact |
| 2 | burn tensor ops (`matmul`) on the wgpu runtime through the dispatch layer (`Device::wgpu(DeviceKind::Cpu)`) — *scratch probe* | PASS, `[[10,13],[22,29]]` exact |
| 3 | **our `rmsnorm_kernel` launched through the verbatim seam** (`try_into_primitive::<CubeBackend>` → `CubeTensor` downcast → `launch_unchecked`), three `d ∈ {8, 16, 32}` fixtures, committed gate `tests/rmsnorm_kernel_wgpu.rs` — **re-run green on 2026-10-02 in this worktree** | PASS, relative error < 1e-5 vs scalar f64 reference |

Stage 3 means: the function the crate already ships — `rmsnorm_cuda::<CubeBackend>`
— accepts a wgpu-device tensor and runs correctly with only a feature-gate
diff. The name says CUDA; the code never did.

## Why it works (the mechanism, not the luck)

1. **`burn-cubecl`'s `CubeTensor` erases the runtime** (`tensor/base.rs:20`,
   burn-cubecl 0.22.0-pre.4: `client: Client`, `handle: Handle` — no runtime
   type parameter). The backend type is ONE (`burn_cubecl::CubeBackend`); the
   DEVICE picks the runtime: `Device::cuda(0)` → PTX via pliron/LLVM,
   `Device::wgpu(..)` → WGSL via cubecl-wgpu → SPIR-V via naga. Every
   `#[cfg(feature = "cuda")]` kernel in our library that launches through a
   `CubeTensor`'s client is runtime-agnostic by construction; only the feature
   gate and the CUDA-specific test fixtures are CUDA's.
2. **cubecl-wgpu 0.11.0-pre.4 is the same generation we pin** — it is already
   in our dependency graph (root `Cargo.lock` resolves it via `burn-core`'s
   unconditional `burn-wgpu` dependency; until today no code path enabled it).
   It carries its own WGSL compiler (`compiler/wgsl/`), a Vulkan/SPIR-V
   backend, a Metal/MSL one, and upstream runs cubecl's own `testgen_all!`
   suite against it (`cubecl-wgpu/src/lib.rs`, the crate's test module).
3. **Our kernel dialect survives the WGSL lowering** — stage 1b is the proof:
   shared memory sized from `#[comptime]`, comptime-unrolled
   reduction loops with per-step barriers, strided reads — all of it lowers and
   answers exactly. (The crate's own `THREADS` comment records that keeping
   `#[comptime]` was twice mis-credited as the CUDA-side fix — `34c5631`, then
   `9ac0377`, the real fix was the loop header; on WGSL the comptime shape
   simply lowers and answers, which is all stage 1b claims.)

## What landed

- `burn-rmsnorm` gains a `wgpu` feature (`dep:burn-cubecl`, `dep:cubecl`,
  `burn/wgpu`, `burn-tensor/wgpu`, `burn/extension` — the cuda feature minus
  autodiff, which the wgpu smoke does not exercise), the fourteen
  `#[cfg(feature = "cuda")]` gates over the runtime-agnostic code widen to
  `#[cfg(any(feature = "cuda", feature = "wgpu"))]`, and
  `tests/rmsnorm_kernel_wgpu.rs` pins: the arm is **TAKEN** (seam counters,
  `ASKED` +1 per call / `SKIPPED` +0), correct to 1e-5 relative against the
  scalar f64 oracle, and `RMSNorm::forward` on a wgpu device routes through the
  fused arm. The file is **ONE test function in two phases** — the counters are
  process-global and cargo runs `#[test]`s in parallel threads, so two readers
  of them in one binary race; the recovered two-test form failed exactly that
  way on its first real run.
  No CPU Vulkan ICD → the adapter request fails **loudly**; the test never
  skips. Default builds (`std`) compile none of it.
- Every claim above is from a run on 2026-10-02 on this box; the gate test is
  the committed instrument, not this document.

## What else was found on the way (reported, not fixed here)

- **`tools/lib_gate.sh` masks a red first cell.** In this lane's run, the
  workspace cell died at link stage (`mold: Disk full?` + `rustc-LLVM ERROR: No
  space left on device` on eight untouched test binaries) and the script then
  overwrote that exit code with the binary-tests cell's 0
  (`lib_gate.sh`, the `[ "$rc" -eq 0 ] || rc=$rc2` chain) and printed `PASS`.
  A disk-full red printed as green is exactly the false-green class
  `tools/test_targets.py` exists to end. Fix is two lines (`rc=$((rc || rc2))`
  per cell); it belongs to the lib_gate owner, not this diff.
- **The CUDA-side twin gate has the same latent counter race.**
  `rmsnorm_kernel_cuda.rs` has four `#[test]`s, three of which read `seam()`
  (`:256, :373, :415`) — parallel threads reading process-global counters. It
  has not fired because the CUDA fixtures are slower than the race window;
  the fix is the same single-sequence shape as the wgpu gate. Owner: the
  `wt/rmsnorm-kernel` lane.
- **The mesa/llvm-libs skew on this box is a partial-upgrade artifact**
  (mesa 26.2.3 wants LLVM 23.1, system `llvm-libs` is 22.1.8) — lavapipe's
  `libvulkan_lvp.so` needs `LD_LIBRARY_PATH=/tmp/opencode/llvm231/usr/lib`
  until the next coherent `pacman -Syu`. Details at the end of this file.

## What this does NOT prove (the honest half)

- **One kernel of ~28 crates.** The other fused kernels (KDA chunked WY, its
  adjoint, the Engram row updates, Sinkhorn, AttnRes merge) share the *dialect*
  but not yet a *gate*. The KDA fused adjoint is wrong and non-deterministic on
  CUDA itself (§3.3, `kda-gradflow-2026-09-30.md`) — there is nothing to port
  correctly until that is fixed.
- **The CUDA-only defects need re-asking on wgpu, each with its own gate:**
  the `d < 4` trailing-cubes launch defect (`d2_isolate.rs`) is "inside
  cubecl's launch path" and was measured on CUDA — wgpu's launch path is
  different code, so the `MIN_FUSED_D` guard stays (it covers the class, and
  the wgpu gate runs at d ∈ {8, 16, 32}); the bf16-as-u16 storage trick is
  pinned against CUDA (`lowp_bf16_cuda.rs`) and untested on WGSL (WGSL has no
  bf16 type either — plausible, unproven); f16 has no tensor cores behind it on
  wgpu at all.
- **The trainer does not build for wgpu.** `dormouse-train/cuda` is the only
  backend feature; nothing here changes that. No `--quant fp8/fp4` factor path
  has a wgpu gate. A wgpu training run is a separate task, not a flag.
- **Speed is unmeasured and unmeasurable here.** lavapipe is a CPU
  rasterizer; "оптимизировано под все карты" in the performance sense needs an
  actual AMD/Intel card. What is proven is *runs correctly*, which is the
  precondition the goal question asked about.

## Labour to full cross-vendor

| step | size | note |
|---|---|---|
| `wgpu` feature + gate per library crate (the burn-rmsnorm pattern, mechanical) | ~1-2 days incl. builds for 27 crates | the `CubeBackend` launch path carries; each kernel wants its own correctness gate |
| kernels with CUDA-specific numerics (KDA adjoint, bf16 storage, factor-quant) | blocked on their CUDA-side fixes | A/B or death applies per kernel; a tie deletes |
| `dormouse-train --features wgpu` + a CPU smoke train run | ~2-4 days | NaN firewall / stress protocol are burn-op level, likely portable; the CUDA-specific step plumbing is the unknown |
| performance on real AMD/Intel | hardware | this box cannot answer it; lavapipe numbers would be fiction |

## This box's Vulkan story (so the next person doesn't re-derive it)

- `vulkan-swrast` (lavapipe) installed via pacman — the only persistent system
  change. **A llvm-libs upgrade to 23.1 was made and immediately ROLLED BACK
  from the pacman cache: system `rustc` links `libLLVM.so.22.1` and stopped
  working.** The box has a mesa 26.2.3 (needs LLVM 23.1) / llvm-libs 22.1.8
  (needs 22.1) skew — a partial-upgrade artifact that predates this lane.
- Working around it without touching the system: LLVM 23.1 extracted to
  `/tmp/opencode/llvm231/usr/lib` (private dir), and every run that must load
  `libvulkan_lvp.so` goes with
  `LD_LIBRARY_PATH=/tmp/opencode/llvm231/usr/lib`. The sonames differ
  (`.so.22.1` vs `.so.23.1`), so rustc and lavapipe coexist. When the box gets
  a coherent `pacman -Syu`, the hack dies with it.
- Selection: `Device::wgpu(DeviceKind::Cpu)` (burn) or
  `WgpuDevice::from(WgpuDeviceKind::Cpu)` (raw cubecl) — the CPU kind is
  documented as "a software rasterizer such as lavapipe"
  (`cubecl-runtime/src/device/wgpu.rs:59`).

## Next step

Extend the same gate to `burn-attnres`'s fused merge (same dialect, kernel
already runs on every training step on CUDA), then decide per crate whether a
`wgpu` feature is worth its lockfile weight — the burn-wgpu/wgpu/naga stack is
a large new dep tree for the fused workspace. Training-side cross-vendor is a
separate plan, gated on the KDA adjoint fix.
