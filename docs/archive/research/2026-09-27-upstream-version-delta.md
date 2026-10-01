# Upstream version delta: are we on the right burn, and is vendoring cheapest?

**Date:** 2026-09-27 · **Scope:** CPU + reading only, no GPU runs. **Method:** `git clone --filter=blob:none`
of `tracel-ai/burn` and `tracel-ai/cubecl`, `git show <ref>:<path>` at every release tag and at
`origin/main`, plus the pristine `~/.cargo/registry` copies of the crates we already build against.
Every verdict below is settled by a cited file:line, a commit, or an md5/blob identity comparison.
Anything not settled that way is marked **NOT VERIFIED**.

## Reference points (all re-confirmed today)

| Thing | Value | How |
| --- | --- | --- |
| `burn-tensor` newest on crates.io | **0.22.0-pre.4**, 2026-09-22 | crates.io API |
| `cubecl-runtime` / `cubecl-core` newest | **0.11.0-pre.4**, 2026-09-22 | crates.io API |
| Last stable burn line | **0.21.0**, 2026-05-07 | crates.io API |
| What we build against | `burn-* 0.22.0-pre.4`, `cubecl-* 0.11.0-pre.4` | `Cargo.lock:1065,1455,2082,2299,2327` |
| burn commits since `v0.22.0-pre.4` | **45** (`474f46b7` = main) | `git rev-list --count` |
| cubecl commits since `v0.11.0-pre.4` | **12** (`a1bb768c` = main) | `git rev-list --count` |
| `pliron` / `pliron-llvm` newest | **0.18.0**, 2026-09-16 — *same version pre.4 already uses* | crates.io API |

So the pre-release window we are on is 3 days old and the delta to `main` is 57 commits total. That
is a small, readable delta — which is why this document can be definitive rather than directional.

---

## 1. The three bugs, per bug, per version

### 1.1 `Bool -> float` cast returns 0.0 for `true` — **STILL PRESENT in every version**

The code is `burn-cubecl/src/kernel/cast/bool_cast.rs`, kernel `bool_cast_kernel`:

```rust
output.write(ABSOLUTE_POS,
    Vector::cast_from(input.read(ABSOLUTE_POS) & Vector::one()));
```

| Ref | md5 of the file | the masking expression |
| --- | --- | --- |
| `v0.21.0` | `087fac11…` | `output[ABSOLUTE_POS] = Vector::cast_from(input[ABSOLUTE_POS] & Vector::one())` (`:24`) |
| `v0.22.0-pre.1` | `bb494c10…` | present (`:29`) |
| `v0.22.0-pre.2` | `bb494c10…` | present |
| `v0.22.0-pre.3` | `539c7222…` | present |
| `v0.22.0-pre.4` | `2b46c17f…` | present (`:28`) |
| `origin/main` | **`2b46c17f…`** | present (`:28`) |

`git log v0.22.0-pre.4..origin/main -- crates/burn-cubecl/src/kernel/cast/bool_cast.rs` is **empty**:
not one commit has touched the file since the tag. The masking expression is byte-identical from
0.21.0 through main. `bool_cast` is still the only bool→numeric path — `crates/burn-cubecl/src/kernel/cast/mod.rs`
is just `mod base; mod bool_cast; pub use base::*; pub use bool_cast::*;`, and main added nothing
else there.

**Verdict: PRESENT upstream, everywhere, including main.** Switching versions cannot fix it.
Patching is the only route, and the patch is one expression. Note the shape of the failure is the
same family as bug 1.2's upstream comment ("quietly computes zeros") — this stack lowers
vector bitwise ops to zeros, and both bugs are that.

**Cost of the patch, when you write it:** it does not have to be a kernel change. The repo already
carries a working caller-side workaround (`.bulba/memory.md`, 2026-09-27: use
`zeros_like().mask_fill(mask, 1.0)` and never build a numeric indicator from `bool_tensor.float()`).
A caller-side `mask_fill` guard is ~1 line and touches no vendored crate at all. Compare: patching
the vendored kernel is ~5 lines in a crate that must then be re-vendored on every burn bump.

### 1.2 bf16 matmul — **STILL PRESENT upstream, and upstream calls it unfixed in a comment**

The reference in our own docs (`docs/architecture/PLAN.md:256-257`, `crates/dormouse-core/examples/gemm_probe.rs:50`)
"burn-cubecl ops/tensor.rs:150" is exact. In `burn-cubecl 0.22.0-pre.4`, `src/ops/tensor.rs:148-151`:

```rust
fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
    let dtype = lhs.dtype;
    matmul(lhs, rhs, None, MatmulStrategy::default(), dtype).unwrap()   // <- line 150
}
```

`origin/main` has the same function at the same line 148 with the same `.unwrap()` at 150 — the file
is unchanged. So **the panic site is identical in pre.4 and main**.

The cause is upstream-documented, in the CUDA runtime's feature advertisement:

* `v0.11.0-pre.4`, `crates/cubecl-cuda/src/runtime.rs:420` — `fn restrict_to_llvm_backend(...)`, and
  at `:488-492`:
  > `bf16` has no type in the LLVM dialect this backend lowers through -- pliron has `builtin.fp16`,
  > `fp32` and `fp64` and nothing between -- so a `bf16` kernel compiles to something that quietly
  > computes zeros. Until it is either given a type or carried as an `i16` the way the minifloats
  > are, it must not be offered.

  followed by `let bf16 = ElemType::Float(FloatKind::BF16); props.features.types.elem.remove(&bf16);`
* `origin/main`, `crates/cubecl-llvm/src/shared/lowered_features.rs:121-126` — the **same comment,
  same code**, relocated into the LLVM crate by commit `0fb304de` "Perf/llvm gpu parity (#1687)".

So upstream's *response* to bf16 was to stop advertising it, not to fix it, and the comment says so
in as many words ("Until it is either given a type …"). Two consequences:

1. On the LLVM/CUDA path bf16 is refused at feature-advertisement time on **both** pre.4 and main.
   The matmul cannot select a strategy, `matmul()` returns `Err(MatmulSetupError)`, and
   `ops/tensor.rs:150`'s `.unwrap()` panics. That is a **loud** failure, and the comment's
   "quietly computes zeros" applies to a bf16 *kernel* compiled anyway, not to this path.
2. Our own note (`.bulba/memory.md`, 2026-09-26: "llvm strips bf16 mma features (by design — no bf16
   registers in the dialect)") is **confirmed correct against the source** — that design is present
   in pre.4, before the commit that reorganised it.

**Verdict: PRESENT on main; the refusal is deliberate and the root cause is stated as unfixed.**
This is the one bug of the three where **patching is the wrong tool**. You cannot add a bf16 tensor
to a pliron/LLVM dialect from a vendored burn or cubecl crate:

* the dialect type would have to live in `pliron` / `pliron-llvm` (both published at 0.18.0 — no newer
  release exists, verified on crates.io today), **and**
* `pliron-llvm` would then have to depend on `cubecl-ir` to carry cubecl's `SizedType`/`AlignedType`
  interfaces — a cycle, since `cubecl-ir` depends on `pliron`.

The three real routes, cheapest first:

| Route | Cost | Note |
| --- | --- | --- |
| (a) cuBLAS `gemm_ex` for the big GEMMs, bypass cubecl's IR | 1-3 d | `crates/cublas-poc` already exists; measured on this card at **43.7 TFLOP/s** f16/f32-accum vs cubecl f32 3.5-7.6 (`.bulba/memory.md` 2026-09-27). This is `docs/architecture/PLAN.md:261-263` option (iii) and it is the only route that buys speed, not just correctness. |
| (b) `CudaBackend::Cpp` for bf16 | ~0 d to try | NVRTC lowers real `__nv_bfloat16`. Our note says this path is broken/slow (150 s/step) — treat as already measured-and-rejected. |
| (c) add a bf16 type to the pliron LLVM dialect | 1-2 w | Vendoring `pliron-llvm` (a crates.io dep) to add a type. Highest risk, buys a kernel we must maintain. |

### 1.3 f16 matmul, `Expected type builtin.fp16 to implement dyn SizedType` — **STILL PRESENT, and structurally unpatchable at our layer**

The error is real and the mechanism is fully traceable:

* `crates/cubecl-llvm/src/shared/to_llvm/ty.rs:55` on main (`:46` on pre.4):
  `impl_cube_to_llvm_type!(Float16Type, self, ctx => FP16Type::get(ctx));`
  — cube's `cube.f16` is converted to pliron-llvm's **builtin** `FP16Type`.
* `crates/cubecl-ir/src/interfaces/mod.rs:396` and `:402` — `TypedExt::size()` / `size_bits()` both do
  `let sized = try_cast_ty!(ty, ctx, dyn SizedType);`, which is where the panic text comes from.
  **These two lines are byte-identical on pre.4 and main.**
* Every `SizedType` impl in the whole tree, enumerated on main:
  `cubecl-ir/src/interfaces/mod.rs:225` (macro), `types/mod.rs:63,127,208` (`VectorType`,
  `AtomicType`, `ArrayType`), `types/scalar.rs:37,88,118` (`IntegerType`, `IndexType`, and the
  `float_type!` macro). The `float_type!` macro *does* generate `impl SizedType` for
  `Float16Type` (`types/scalar.rs:182`) — but that is `cube.f16`, not `builtin.fp16`. **No impl
  exists for the LLVM builtin type**, and the `float_type!` table is byte-identical pre.4 → main
  (verified by grepping every invocation in both refs).
* `crates/cubecl-llvm/src/nvptx/matrix.rs` — the NVPTX matrix lowering, the only place a
  fragment-shaped f16 asks for its width (`nvptx/matrix.rs:501-502`,
  `in_ty.elem_ty.size_bits(ctx)`) — is **the same git blob** (`900c5467…`) on pre.4 and on main.
  `git diff v0.11.0-pre.4 origin/main -- crates/cubecl-llvm/src/nvptx/matrix.rs` is 0 lines.

**Verdict: PRESENT on main; the offending code is literally the same bytes.**

And it is **not fixable by vendoring**, which is the important part:

* `SizedType` is defined in `cubecl-ir`; `FP16Type` is defined in `pliron-llvm` (imported at
  `crates/cubecl-llvm/src/prelude.rs:44`). `cubecl-llvm` depends on both, so a local
  `impl SizedType for FP16Type` is an **orphan-rule violation**.
* `cubecl-ir` cannot name `FP16Type` at all — `crates/cubecl-ir/Cargo.toml` depends on `pliron` and
  optional `pliron-spirv`, never `pliron-llvm`.

So the only layers that *can* fix it are `pliron-llvm` (needs a cubecl dependency → cycle) or the
NVPTX lowering itself (stop asking a converted type for its size). Both are upstream work. There is
no free upgrade either: `pliron` newest is 0.18.0 and that is what pre.4 already pins.

**This is the bug that decides the verdict.** It is the only one of the three where neither a
version bump nor a vendored patch can help. Route (a) — cuBLAS — bypasses cubecl's IR entirely and
sidesteps it, which is why (a) is the recommendation and not just a fallback.

### 1.4 Summary table

| Bug | 0.21.0 | 0.22.0-pre.1…pre.4 | `main` | Patchable by us? |
| --- | --- | --- | --- | --- |
| `bool -> float` → 0.0 | present | present | present (file untouched since tag) | **Yes** — 1-line caller-side `mask_fill` guard, or ~5 lines in a vendored kernel |
| bf16 matmul | n/a (present in spirit) | present (refused by design, `runtime.rs:488-492`) | present (refused, relocated to `lowered_features.rs:121-126`) | **No** — needs a pliron dialect type. Use cuBLAS. |
| f16 `builtin.fp16` `SizedType` | n/a | present (`ty.rs:46` + `interfaces/mod.rs:396,402`) | present (identical bytes, `nvptx/matrix.rs` same blob) | **No** — orphan rule. Use cuBLAS. |

---

## 2. The `TypeId` gate: how widespread, and is there a sanctioned check?

### 2.1 How widespread — ours, and upstream's

**Ours:** 19 `TypeId::of` sites in **11 files across 9 vendored fork crates**
(`burn-attnres`, `burn-bitnet`, `burn-gdn2` ×3, `burn-kda`, `burn-mhc`, `burn-rope`, `burn-sct`,
`burn-situ`, `burn-spectral`). **10 of the 19** compare the *generic* `B` against `CudaBare` — those
are the ones that die under `Autodiff`; the other 9 compare an already-unwrapped `Inner`/`CB` and are
fine. `crates/burn-kda/src/fused.rs:35` already carries the hand-patched second arm
`TypeId::of::<B>() == TypeId::of::<Autodiff<CudaBare>>()` — i.e. the whack-a-mole has already started
and the next wrapper type (`Dispatch`, `Fusion`) re-breaks it.

**Upstream:** 16 files use `TypeId::of` in cubecl and 22 sites in burn, but **none of them is a
backend-kind gate**. They are enum/registry lookups (`cubecl-core/src/frontend/cmma.rs:1133,1186`
compare two *element* types inside the kernel IR; `burn-backend/src/backend/distributed/api.rs:29,38,46`
is a global client registry keyed by the full `B`). Upstream does not gate kernels on
"am I the bare CUDA backend" anywhere. **The pattern is ours, not inherited.**

### 2.2 The sanctioned check — yes, one exists, and it is one line

**There is no backend-kind constant, and there never was.** Verified:

* `Backend` (`crates/burn-backend/src/backend/base.rs`) has associated types `Device`,
  `FloatTensorPrimitive`, `IntTensorPrimitive`, `BoolTensorPrimitive`,
  `QuantizedTensorPrimitive`, `GraphPrimitive` — **no `InnerBackend`**.
* `AutodiffBackend::InnerBackend` exists at `base.rs:370`, but only on the `AutodiffBackend` trait,
  so it is unreachable for a generic `B: Backend`. The premise in the brief is confirmed.
* `git grep -c "B: AutodiffBackend"` in `crates/burn-tensor/src/tensor/api/`: **2 hits on
  `v0.21.0`, 0 hits on `v0.22.0-pre.4`.** Upstream deliberately removed that bound — the official
  guide, `burn-book/src/migrating-to-0.22.md` §"Autodiff is runtime state": *"Removing
  `B: AutodiffBackend` moves precondition checks to runtime."* **This is the exact version boundary
  that created our problem: the sanctioned route existed in 0.21 and was deleted in 0.22.**
* `BackendKind` does not exist in burn. The only near-miss is
  `BackendKindArms` in `crates/burn-backend-extension/src/routing.rs:103`, which is `pub(crate)`
  macro plumbing for `DispatchTensorKind` — and `DispatchTensorKind`
  (`crates/burn-dispatch/src/tensor.rs:257`) is the typed answer *only for `Dispatch` tensors*.
  Upstream's own doc on that variant says the right thing: `DispatchTensorKind::Cube` is
  *"A tensor on the cubecl backend — **its device says which runtime**."*

**The sanctioned hook that survives every wrapper is `Backend::name`:**

```rust
// crates/burn-backend/src/backend/base.rs:177
fn name(device: &Self::Device) -> String;
```

It survives because **every wrapper forwards**:

| Layer | file:line | value |
| --- | --- | --- |
| `Autodiff<B, C>` | `crates/burn-autodiff/src/backend.rs:50` | `format!("autodiff<{}>", B::name(device))` |
| `Dispatch` | `crates/burn-dispatch/src/backend.rs:200` | `format!("dispatch<{inner}>")` |
| `Fusion<B>` | `crates/burn-fusion/src/backend.rs:46` | `format!("fusion<{}>", B::name(device))` |
| `CubeBackend` | `crates/burn-cubecl/src/backend.rs:99` | `format!("cubecl<{}>", client.name())` |
| `Client::name()` | `crates/cubecl-runtime/src/client.rs:200` | `self.utilities.name` |
| the runtime name | `crates/cubecl-runtime/src/server/base.rs:117-119` | doc: *"`cuda`, `wgpu<spirv>`"* |

So `<B as Backend>::name(&device)` on `Autodiff<CudaBare, BalancedCheckpointing>` yields
`"autodiff<cubecl<cuda>>"`, and `.contains("cuda")` is correct. Verified present on **pre.4 as
well** (`client.rs:204-206` and `server/base.rs:118-120` are the same in pre.4), so it works on the
version we build today.

**Recommendation: one helper, `<B as Backend>::name(&device).contains("cuda")`, replacing all 10
generic `B` sites (and deleting the hand-patched arm at `burn-kda/src/fused.rs:35`).** Cost ~1 h,
zero vendoring, no version bump, and it keeps working when someone swaps in `Dispatch`/`Fusion`.
The 9 `Inner`-vs-`CudaBare` sites are already correct and should be left alone.

Cost table for this item, which is the *cheapest* of everything in this document:

| Option | Cost |
| --- | --- |
| `Backend::name()` helper, 10 sites | **~1 h, 0 vendor churn** ← do this |
| extend to `DispatchTensorKind` matching | 2-4 d, and requires switching to the Dispatch backend (our own 2026-09-26 note: "48 unsatisfied `DispatchKindConversion`" when the vendored crates downcast a bare `CubeBackend`) |
| stay on `TypeId` and add an arm per wrapper | unbounded; it already failed once |

---

## 3. burn-flex: what it is, and what it costs

### 3.1 What it is

`crates/burn-flex` — a from-scratch CPU backend, `no_std`/`std`/WASM, SIMD via `macerator`, matmul
via the `gemm` crate with native f16, optional `rayon`. It is **first-class, not an experiment**:
`crates/burn/Cargo.toml:131` has `flex = ["burn-core/flex", ...]`, `burn-dispatch` has
`DispatchTensorKind::Flex` and `devices::FlexDevice` and a `DispatchGraph::Flex` variant, and
`burn-backend-tests/tests/tensor_f16.rs` is gated on `feature = "flex"` (commit `6b154950`
"burn-flex: enable f16 tests …"). Upstream's own summary: *"burn-flex is a from-scratch replacement
for burn-ndarray … It implements all required Backend traits … and passes the same test suite."*
Two post-pre.4 commits improve it (`1dafe9f4` ConvTranspose perf, `d918eff5`/`ebad4f9d` layer_norm
variance and shift-width fixes).

### 3.2 The migration is already sanctioned and already available

**`burn-flex` exists in `v0.21.0`, `v0.22.0-pre.1`, `v0.22.0-pre.4` and `main` — identical
`ACKNOWLEDGMENTS.md` blob `1d0915d2…` at all four refs.** So **migrating costs no version bump at
all.** That is the single most decision-relevant fact in this section, and it is what makes option
(d) cheap rather than expensive.

**And the debt is real and acknowledged by the compiler, not just by our doc note.**
`crates/burn-ndarray/src/lib.rs:3-8` on both pre.4 and main:

```rust
#![deprecated(
    since = "0.22.0",
    note = "burn-ndarray is deprecated and will be removed in a future release.
            Use burn-flex for pure-Rust CPU execution (std, no_std, WebAssembly), or one of
            the CubeCL backends (burn-cuda, burn-rocm, burn-wgpu, burn-cpu) for GPU acceleration."
)]
```

That attribute is **absent on `v0.21.0`** — the deprecation is new in 0.22, i.e. it is a cost we
already carry today. 40 of our vendored crates carry the matching doc note.

### 3.3 The actual cost, counted

Upstream's own migration table (`crates/burn-flex/COMPARISON.md:609-623`) is the API-break spec:

| Change | Details |
| --- | --- |
| Type parameter | `NdArray<f32>` becomes `Flex` |
| Device | `NdArrayDevice::Cpu` becomes `FlexDevice` |
| Feature flags | `multi-threads` becomes `rayon` |
| BLAS features | **No equivalent** (gemm handles matmul) |
| Autodiff | `Autodiff<Flex>` (same pattern) |
| f16/bf16 | Works out of the box (**new capability**) |
| Tests | Same `burn-backend-tests` suite passes |
| Lost | `export_tests` reference-implementation feature |

Counted in **our** tree today:

| Surface | Count | Notes |
| --- | --- | --- |
| Workspace crates referencing ndarray | **7 files** | `dormouse-core/Cargo.toml`, `dormouse-core/src/{aux,act_quant}.rs`, `dormouse-core/tests/{model_seam,backend_parity}.rs`, `dormouse-train/{Cargo.toml,src/lib.rs}` |
| Vendored fork | **76 files, 149 occurrences** | but almost all is `[dev-dependencies] burn-ndarray` + a `type B = ...` test alias |
| `NdArrayDevice` in `.rs` | **1 occurrence** | the device API barely appears → the one true API rename is nearly free |
| `blas-openblas` feature refs | **31** | all in `Cargo.toml`; deleting them is mechanical |
| Real code edits | **~15-25 sites** | the type alias, the `DispatchKindConversion<Autodiff<NdArray>>` bounds in `aux.rs:60,77,153`, 2 `Cargo.toml` dep lines, the `cpu` feature in `dormouse-train/Cargo.toml:9` |

Two things make this cheaper than it looks:

1. **`blas-openblas` goes away entirely.** Our AGENTS.md already records "The GPU binary carries no
   OpenBLAS: the CPU ndarray backend is the optional `cpu` feature". Dropping BLAS removes a native
   build dependency from the CPU test path — 31 feature references deleted, not ported.
2. **It fixes a test gap we already hit.** `crates/dormouse-core/tests/backend_parity.rs:14-15`
   documents that burn-ndarray declares `DType::F16 | DType::BF16` an *empty usage set*
   (`burn-ndarray/src/backend.rs:97`) and refuses them — so **CPU tests structurally cannot catch
   bf16/f16 bugs.** flex supports f16/bf16 natively. Given that bugs 1.2 and 1.3 are both
   half-precision, this is not cosmetic: it widens what the CPU suite can gate.

**Does the vendored `[patch.crates-io]` strategy survive? Yes, completely — and this is worth being
explicit about.** `[patch.crates-io]` is version-agnostic: swapping `burn-ndarray` (an ordinary
crates.io dep) for `burn-flex` (another ordinary crates.io dep) touches only `Cargo.toml` lines and
cannot interact with the five `path = "vendor/…"` patches. Nothing about the root-`exclude` list
changes. If the fork goes standalone, the patch set travels unchanged.

**Cost estimate: 0.5-1 day.** It is a type-alias rename plus a `Cargo.toml` sweep, and it is
independent of every other decision in this document — it can be done on pre.4, today, without
touching a vendored crate.

---

## 4. Verdict and the four options

Days assume one engineer, and "GPU smoke" means one training run to first log line.

### Option (a) Stay on 0.22.0-pre.4, keep patching what we must — **RECOMMENDED**

* **Cost:** 0.5 d for bug 1 (caller-side `mask_fill` guard); 1-3 d for bugs 1.2/1.3 via the cuBLAS
  route already scaffolded in `crates/cublas-poc`; 1 h for the `name()` helper. **Total ≈ 2-4 d.**
* **Fixes:** everything reachable. Bugs 1.2/1.3 are not "fixed" — they are *routed around*, which is
  the only available outcome (§1.2, §1.3), and the route is also the fast one (43.7 TFLOP/s).
* **Risks:** low. Nothing moves. The two remaining landmines (bf16, f16) are *loud* failures, not
  silent ones, so they cannot quietly corrupt a run.
* **Carries:** a real fork-drift bill — 5 vendored crates to re-vendor on every burn bump, and 57
  upstream commits currently unreviewed.

### Option (b) Move to 0.21.0 stable — **REJECT. This is the expensive one, not the cheap one.**

* **Cost:** **5-10 d, and it is the only option that can cost more than everything else combined.**
* **Why it is expensive:** 0.21 predates the whole 0.22 line. `burn-book/src/migrating-to-0.22.md` is
  311 lines of migration guidance — "Types and devices", "Autodiff is runtime state", "Moving tensors
  and switching module state", "Migrating checkpoints", "Custom integrations" → *Modules, Optimizers
  and schedulers, Custom metrics, Renderers and event processors, Distributed training, Storage
  adapters and checkpointers, Backend extensions* — and we touch most of those surfaces: our own
  optimizer policy (`train/src/optim.rs`), the burnpack checkpoint path, the `--aux` heads, the
  backend extensions. Plus: `burn-flex` exists on 0.21 but the whole `Dispatch` layer we build on
  (`burn_dispatch::{DispatchDevice, devices::CubeDevice}`, `crates/dormouse-train/src/lib.rs:174-175`)
  and the 4D-autodiff `fused` rework are 0.22-era.
* **What it would fix: NOTHING.** Bug 1.1 is present in `v0.21.0` with the identical masking
  expression (`bool_cast.rs:24`); the deprecation of ndarray would go away, but that is the only
  gain, and it is a cosmetic one bought at 5-10 d.
* **Risks:** high. A large silent-semantics change set (autodiff moved from type-level to
  runtime state) on a training stack that is already three weeks into an optimisation campaign.
* **Kill it on the evidence:** a version with a *known, documented, machine-level* bf16 hole in its
  CUDA backend is the worst candidate to retreat to.

### Option (c) Re-vendor a newer upstream commit than pre.4 — **WORTH A SPIKE, NOT A MOVE. This is the expensive one for the bugs.**

* **Cost:** 1-2 d to re-vendor `burn-cubecl`/`cubecl-*` at `main`, **plus** re-reviewing 57 commits,
  **plus** absorbing 2 breaking changes: `e2014871` "make TensorData fields private" and `af8b3306`
  "remove `into_tiled` from the public API". **Both look free for us** — `git grep "TensorData {"` and
  `into_tiled` return **zero hits** across `crates/` and `vendor/burn-fused/`. Estimate 2-3 d total,
  and it grows linearly with every week we wait.
* **What it would fix: none of the three bugs.** All three are present on main (§1.4). The four
  remaining cubecl commits are unrelated to them.
* **What it *would* buy, and this is real:** `c8517b38` "fix(llvm): emit the PTX version the CUDA
  driver loads (#1684)" wires `cuDriverGetVersion` → `PtxVersion::for_driver(...)`, replacing
  pre.4's `None` (= "LLVM's default, the oldest the architecture accepts", per the pre.4 doc comment
  at `nvptx/codegen.rs:277`). On a consumer Blackwell part that is exactly our failure class — the
  `.bulba/memory.md` 2026-09-25 note about the `cuEventCreate-700` failure and the pinned
  `CUDARC_CUDA_VERSION=12050` is the same neighbourhood. Also `81b5a3b3` "fix(optim): compute gradient
  norm clipping in F32 for half precision (#5843)", which speaks directly to our NaN firewall.
* **Risks:** `0fb304de` also *narrows* what the CUDA backend advertises beyond bf16 — it strips TMA,
  clusters, async copy, and `OpaqueType::TensorMap`/`Barrier` (moved to
  `cubecl-llvm/src/shared/lowered_features.rs:103-141`). Per that file's own module doc,
  *"an advertisement that cannot be honoured is a launch that fails rather than one that falls
  back."* So the upgrade can convert a silent-wrong into a hard-fail in kernels we have not audited.
  We do not use TMA or clusters today, which bounds the blast radius — but that is a fact to verify,
  not assume.
* **Verdict:** do the 1-2 d spike, on a branch, gated on one `small`-preset training run. Adopt only
  if it is green. **It is not the cheapest path to the three bugs, because it fixes none of them.**

### Option (d) Migrate to burn-flex now — **DO IT, but it is orthogonal. Cheapest for zero bugs.**

* **Cost:** **0.5-1 d** (§3.3). ~15-25 code sites, 31 feature references deleted, 1 device rename.
* **What it would fix: none of the three bugs.** It removes a deprecation we already carry, deletes
  the OpenBLAS native dep from the CPU path, and — the only real win — **widens the CPU test backend
  from fp32-only to f16/bf16**, which is the only lever we have to gate half-precision bugs like
  1.2 and 1.3 without a GPU. Per `backend_parity.rs:14-15` and `burn-ndarray/src/backend.rs:97`, that
  coverage gap is why bugs 1.2/1.3 were only found by hand.
* **Risks:** low. No version bump required; the `[patch.crates-io]` strategy is untouched; upstream
  ships and actively improves the crate (3 post-pre.4 commits). Residual: we lose `export_tests`
  and the BLAS knobs — neither of which we use.
* **Why it is cheap when the brief assumed it would be expensive:** because burn-flex is already on
  crates.io at our exact version. The migration was never a version decision.

### The two answers, separated as asked

| | Cheapest | Why the others lose |
| --- | --- | --- |
| **The three bugs** | **(a) stay on pre.4**, bug 1 via a 1-line caller-side guard, bugs 1.2/1.3 via **cuBLAS** (`crates/cublas-poc`) | (b) 0.21 fixes nothing and costs 5-10 d. (c) main has all three bugs byte-identical. (d) touches none of them. Bugs 1.2/1.3 are **structurally unpatchable at our layer** (orphan rule / dialect type), so routing around them is not a fallback, it is the only option. |
| **The TypeId gate** | **`<B as Backend>::name(&device).contains("cuda")`**, ~1 h, 10 sites, delete the hand-patched arm at `burn-kda/src/fused.rs:35` | (b) is the *only* option that fixes it properly — 0.21 still has `B: AutodiffBackend` (2 hits vs 0), so `InnerBackend` is reachable — but 5-10 d to buy a 1 h fix. Do not walk back a whole version for this. (c)/(d) are irrelevant. |

### Do these, in this order

1. **~1 h** — `Backend::name()` helper; replace the 10 generic `B` gates. This is the fused-KDA path
   you lost today, and it is the cheapest item in the document.
2. **~1 h** — bug 1: ban `bool_tensor.float()` at the call site (`mask_fill` guard). Prefer this over
   patching the vendored kernel: it removes the reason to vendor `burn-cubecl` for this bug at all.
3. **0.5-1 d** — burn-flex. Independent of everything above; do it whenever.
4. **1-2 d spike** — re-vendor at `main` for the PTX-version fix only. Adopt on a green
   `small`-preset run. Do **not** expect it to touch bugs 1.1/1.2/1.3.
5. **1-3 d** — cuBLAS GEMM for the 1B path. The only route that makes tensor cores reachable at all
   on this stack, and the only one that is a speed win rather than a correctness patch.
6. **Never** — 0.21.0. It fixes nothing we have and costs more than items 1-5 combined.

**Net effect on the vendor-and-patch strategy: it is right, and it is cheaper than it looks — but
only for bug 1.** Bugs 1.2 and 1.3 were never vendor-and-patch problems; they are a missing type in
someone else's compiler dialect, and the honest answer is to stop asking cubecl for that computation
and call cuBLAS. The two items that would shrink the vendor surface fastest are the 1-hour
`name()` helper (which removes a whole class of "our fault" bugs, not a vendored patch) and the
`mask_fill` guard (which removes one vendored crate from the patch set).

---

## NOT VERIFIED

Written down rather than reasoned around, per the brief.

1. **The exact error text/panic of the bf16 matmul failure on pre.4.** The *mechanism* is verified by
   code read (`restrict_to_llvm_backend` removes bf16 → no strategy → `.unwrap()` at
   `burn-cubecl/src/ops/tensor.rs:150`), but no GPU run was made, so "fails its own two tests" is
   carried over from `docs/architecture/PLAN.md:256-257`, not reproduced here.
2. **Whether the f16 `SizedType` panic is reached specifically through `nvptx/matrix.rs:501-502`.** That
   is the only f16-width query found in the matrix lowering, and the type mapping plus the missing
   impl are both confirmed, but the exact call path was not executed.
3. **Whether main's narrowed feature set (TMA/clusters/async-copy removed by `0fb304de`) breaks any
   kernel we actually run.** We believe not — we use none of them — but that is a code-reading
   belief, not a measured one. This is the main risk of option (c) and is exactly what the 1-2 d
   spike is for.
4. **The comparative speed of burn-flex vs burn-ndarray on our test suite.** `COMPARISON.md` claims
   1.1-9.7× compute and up to 166,000× on zero-copy structural wins; not measured by us. Irrelevant
   to the decision (the CPU backend is for tests), but do not quote those numbers as ours.
5. **Whether `cubek` at main's pinned rev changes any of the three verdicts.** The matmul launch
   delegates to `cubek::matmul::launch::launch_ref` (`burn-cubecl/src/kernel/matmul/base.rs`), and
   burn bumped cubek 4 times post-pre.4 (`c240f001`, `86781f08`, `fafc9d7a`, `474f46b7`). We did not
   read the cubek repo, so a cubek-side bf16/f16 matmul path could in principle exist. The bf16
   element-type refusal upstream of it in `cubecl-ir`/`cubecl-llvm` is what makes this unlikely, but
   it is not verified.

## Method notes

* `git clone --filter=blob:none --no-checkout` of both repos; every claim is a `git show <ref>:<path>`
  at a release tag or at `origin/main`, or a local `~/.cargo/registry` read of the exact crate
  version in `Cargo.lock`. No blog posts, no release-note prose, no memory.
* The repo was not modified. `vendor/burn-cubecl/` was observed to be byte-identical to the registry
  copy of `burn-cubecl 0.22.0-pre.4` at the time of reading (only `Cargo.lock` differs), i.e. the
  concurrent vendoring agent had not yet applied its patch — so the local fork's fix content is not
  reflected in §1.1, which assesses **upstream** only.
* The `[patch.crates-io]` + root-`exclude` mechanism itself is sound for all of this: it is
  version-agnostic and independent of which burn version the patched crates claim to be, which is
  what makes options (a) and (c) composable and makes (d) free of side effects.
