# The three CUDA-backend precision bugs: two are not bugs, one is upstream, and the gate is the deliverable

Status: gate landed and green. **Bug 1 does not reproduce - the documented cause
was wrong. Bug 3 is a silent performance loss, not a wrong answer. Bug 2 is a
real upstream gap and is the only one that cannot be worked around.**
Measured 2026-09-27 on burn 0.22.0-pre.4 / cubecl 0.11.0-pre.4, sm_120 (RTX 5060 Ti).

The gate: `crates/backend-parity/tests/backend_parity.rs`

```
cargo test -p backend-parity --features cuda --test backend_parity
test result: ok. 8 passed; 0 failed; 2 ignored
```

Always with `--features cuda`. A default run proves nothing about this bug
class, and two of these three bugs were *never* CPU-detectable.

## The fact that frames bugs 2 and 3

`cubecl-cuda` compiles every kernel through one of two backends, and the choice
is a cargo feature, not a runtime property
(`vendor/cubecl-fix/cubecl-cuda/src/compiler.rs:16-32`):

```rust
pub enum CudaBackend { Cpp, Llvm }
impl Default for CudaBackend {
    fn default() -> Self {
        if cfg!(feature = "cpp") { CudaBackend::Cpp } else { CudaBackend::Llvm }
    }
}
```

`cpp` is an **empty feature that is off by default** and that nothing in our
graph enables. So this box runs the **pliron -> LLVM -> NVPTX** backend, not
NVRTC. Every "the CUDA backend cannot do X" below is a fact about the LLVM
dialect, and the single most useful file to read on this stack is
`restrict_to_llvm_backend` (`.../runtime.rs:420-513`), which advertises only
what that dialect can lower and says why in comments.

## Bug 1 - `bool_tensor.float()`: NOT REPRODUCIBLE. The cast is correct.

The claim in `AGENTS.md` and `.bulba/memory.md:15` is that `bool.float()`
returns 0.0 for `true` on the CUDA backend. **It does not.** Four tests, one
assertion body, green:

| test | body | result |
|---|---|---|
| `bool_to_float_ndarray` | raw CPU backend | ok |
| `bool_to_float_ndarray_autodiff` | CPU dispatch path | ok |
| `bool_to_float_cuda` | raw CUDA backend | ok |
| `bool_to_float_cuda_autodiff` | CUDA dispatch path (what the trainer runs) | ok |

The body is adversarial on purpose: all-false and all-true and mixed at
n = 1/3/4/8/15/16/17/64/1000 (the sizes that straddle cubecl's vector-width
choices), a mask produced by a *comparison* rather than by `from_bool`, its
`bool_not`, a 2-D mask, and the sibling `bool.int()` which shares the kernel
(`burn-cubecl/src/kernel/cast/bool_cast.rs`). Exact equality, not a tolerance:
`true` is 1.0 and `false` is 0.0 on all four.

The autodiff variants matter and were added after the first pass: on this stack
the backend is **not** a type parameter (`Tensor`/`Device` are backend-agnostic
and the device picks the backend at runtime), and the trainer's tensors are the
*dispatch* kind, which reaches a different entry point
(`Dispatch::bool_into_float`) than a raw one. A raw-only gate would not have
tested the path that broke.

### What actually broke the run

The two candidates that the cast was blamed for, both now also gated, and both
**pass** on both backends:

- `device_side_bool_indicator` - the `zeros_like(mask).mask_fill(mask, 1.0)`
  counter and its `sum()`, the expression the NaN firewall was built on.
- `clone_is_a_snapshot_before_mask_fill` - the aliasing hazard: on CUDA
  `clone()` shares the device buffer and `mask_fill` writes in place, so a
  "raw" copy taken before a mask can read the masked value. This is what made a
  run log `ce=0.000` with `best` stuck at 0 (`.bulba/memory.md:20`), and it was
  one of the three structural bugs fixed on 2026-09-27.

So the most likely history is: the counter was wrong because of the
clone/mask_fill aliasing plus a firewall that did not skip the step, both since
fixed, and the cast got the blame by association. **The actionable consequence
is not a patch, it is the gate**: from now on "the cast is broken" is a claim
with a command attached, not a paragraph in a memory file. Nobody should re-add
a device-side numeric indicator without running it through this gate first.

One honest caveat: this exonerates the cast *on this stack, as of
cubecl 0.11.0-pre.4 with the vendored cubecl-cuda*. It does not prove the
`& Vector::one()` mask in `bool_cast.rs:26-29` is well-formed - it is the only
cast kernel there that is not a bare `Vector::cast_from`, and the broadcast
helper it leans on (`cubecl-core/src/frontend/element/cast.rs:62-71`) asserts
its argument is a scalar while the kernel launches `vector_size` as a *runtime*
argument. It lowers correctly today. If a future cubecl breaks it, the file is
named here and the vendoring recipe is three lines (copy the crate to
`vendor/burn-cubecl/`, add the path to the root `exclude`, add
`burn-cubecl = { path = ... }` to `[patch.crates-io]`) - the same shape as
`vendor/cubecl-fix`. That wiring was built for this ADR and then **removed**,
because owning an unmodified vendored copy of a fast-moving pre-release crate
pins it and forces a manual bump on every upstream release. Do not leave a
vendored crate that carries no patch.

## Bug 2 - bf16 matmul: a real upstream gap, no workaround

Not a burn bug, and not fixable in our tree: **the LLVM dialect has no bf16
type.** The failure is not even in the matmul - the f32 readback of the bf16
buffer dies first:

```
compiling 'cast_element_i_bf16_o_f32_n_1_72947875' for sm_120:
Compilation error: invalid input program.
Type cube.bf16 does not have a conversion to LLVM type implemented
```

Three independent in-tree confirmations:

1. `restrict_to_llvm_backend` deletes bf16 from the advertised element types on
   purpose (`vendor/cubecl-fix/cubecl-cuda/src/runtime.rs:488-498`): *"the
   dialect this backend lowers through has no type for them, so there is
   nothing to put in a register"*.
2. `cubecl-llvm` has no bf16 lowering rule (bf16 exists only in the amdgpu
   matrix path).
3. The bf16 tensor-core families are stripped from the advertised
   `matmul.cmma` / `matmul.mma` configs (same function, lines 435-454), so no
   selector can pick one even if it existed.

So dormouse-spectral's own bf16 test failing at `burn-cubecl ops/tensor.rs:150`
(`float_matmul`) is a backend correctly saying "I have no bf16". **A silent
fp32 fallback dressed up as bf16 is the one thing not to do here**: it makes
`--bf16` a lie with a 2-3 digit error bar and no error message, which is
exactly the shape of the `--act-quant fp4` claim that turned out to verify the
wrong path (ADR-0019).

**What a real fix needs** (upstream, cubecl-ir + cubecl-llvm + cubecl-cuda):
carry bf16 the way the minifloats are carried - an `i16` value plus a
truncate/extend pair around the arithmetic - then re-advertise the bf16 cmma/mma
families. `mma.sync.aligned.m16n8k16.bf16` exists for sm_80+, so the hardware
is there; only the type is missing.

**What works today**: bf16 *storage* as `u16` bit patterns - a bf16 is the top
16 bits of its f32, so `f32::reinterpret(u32 << 16)` in and round-to-nearest-even
truncation out, which is integer ops plus a bitcast and no bf16 value ever enters
the dialect. Pinned against f64 on the host in
`vendor/dormouse-fused/crates/dormouse-gdn2/tests/lowp_bf16_cuda.rs` (not ours; that
agent's probe). That is the primitive any bf16 compute path here must be built
from.

The two gate tests for it are `#[ignore]`d **by design**, the same convention as
dormouse-spectral's own probe: they are a report, not a requirement, and a
permanently red test is not a gate. They flip to green the day a bf16 type
lands:
`cargo test -p backend-parity --features cuda --test backend_parity -- --ignored`

## Bug 3 - f16 matmul: the ANSWER is right, the tensor-core path is not there

This is the one that was misdiagnosed as "f16 matmul fails outright". It does
not. The gate passes: f16 matches the fp32 result of the same product to 1e-2,
both a 1xK@Kx1 dot and a 64x96x48 GEMM.

What is broken is one *candidate* kernel. The f16 tensor-core routine dies at
compile time with exactly the reported error

```
cubecl-ir/src/interfaces/mod.rs:402: Expected type builtin.fp16  to implement dyn SizedType
```

and **the autotuner catches it and falls back to a non-accelerated routine**, so
the result is right and the cost is silent. Anyone reading only the panic, or
only a log line, would call this a hard failure; it is a performance cliff.

f16 is exactly one interface short of working, which is what makes it worth
fixing upstream. Everything around it is already there:

- `cubecl-ir` declares f16 as a first-class sized scalar:
  `float_type!("cube.f16", Float16Type, F16, 2)` (`src/types/scalar.rs:182`).
- The nvptx matrix lowering has an f16 register form:
  `mma_type -> Some(("f16", RegisterForm::Packed(2)))`
  (`cubecl-llvm/src/nvptx/matrix.rs:551-553`).
- `restrict_to_llvm_backend` **keeps** the f16 cmma and mma families
  (`runtime.rs:435-454`), and says so: the narrowing is to "the element types
  its lowering has register shapes for - `f16` operands throughout".

The gap is that the LLVM-level type for f16 is pliron's `builtin.fp16`
(`pliron-0.17.0/src/builtin/types.rs:161`), and it implements
`FloatTypeInterface` but **not the `SizedType` interface the size queries ask
for**:

- `cubecl_ir::interfaces::TypedExt::size` does
  `try_cast_ty!(ty, ctx, dyn SizedType).unwrap()` - the failing line, 402.
- `cubecl-opt`'s shared-memory sizing calls exactly that:
  `MemoryResource::size = self.value_ty.size(ctx)` (`cubecl-opt/src/lib.rs:36-40`),
  reached from `SmemAllocation::end` (`analyses/liveness.rs:40-42`) which sizes
  the single shared-memory block every staged matmul tile lands in.
- `cubecl-llvm` queries the same interface the other way
  (`src/shared/shared_memory.rs:36-45`).

An f16 matmul is the one matmul that stages f16 tiles, so f16 is the one
element type that walks into that query. Hence the advertisement and the
lowering disagree.

**The patch** is small and belongs in **cubecl-ir**, not ours: the orphan rule
is satisfied because the *trait* is local, so the impl sits right next to the
ones its own `float_type!` macro emits.

```rust
// cubecl-ir/src/types/scalar.rs, after the float_type! block
use pliron::builtin::types::FP16Type;

#[type_interface_impl]
impl SizedType for FP16Type {
    fn size(&self, _ctx: &Context) -> usize { 2 }
    fn size_bits(&self, _ctx: &Context) -> usize { 16 }
}

#[type_interface_impl]
impl AlignedType for FP16Type {
    fn align(&self, ctx: &Context) -> usize { self.size(ctx) }
}
```

Ten lines, then the f16 tensor cores come up - which is the difference between
the measured 43.7 TFLOP/s of cuBLAS f16-in/fp32-accumulate
(`.bulba/memory.md:17`) and what a burn matmul can reach. Two caveats to carry
with the patch: the f64 and bf16 paths are in the same boat and would need the
same audit, and the autotuner's "keep unsupported accelerated candidates at
PRIORITY_MIN so the plan is never empty" behaviour
(`burn-cubecl/src/kernel/matmul/tune/base.rs:551-561`) is what turned a
compiler panic into a silent cliff. Both are worth an upstream word.

Until it lands, f16 through burn is correct and slow; the fast route is cuBLAS
directly, which is what `crates/cublas-poc` (another agent's) is for.

## Why this class of bug stayed invisible

- `burn-ndarray` **refuses** f16 and bf16 outright:
  `DType::F16 | DType::BF16 => DTypeUsageSet::empty()`
  (`burn-ndarray-0.22.0-pre.4/src/backend.rs:97`). A CPU-only suite cannot even
  express the assertion. (`burn-flex` does support both dtypes, so this gate
  could widen to true two-backend parity for bugs 2 and 3; it is not used here
  because enabling burn's `flex` feature re-fingerprints the entire burn stack
  for every crate in the workspace. Do it as a standalone change.)
- `bool.float()` is the rare cast that compiles on both backends, so it is the
  one that can silently disagree - and the disagreement is data-dependent, not
  a compile error.

## Follow-ups

- Fold these assertions into `crates/dormouse-core/tests/model_seam.rs` once
  that crate is out of its refactor, so the gate runs with the suite people
  already run. It lives in its own crate for now, which is also why it kept
  running while `dormouse-core` was mid-refactor and uncompilable.
- Land the ten-line `SizedType` patch for pliron's `FP16Type` upstream.
- Every *other* device-side numeric indicator in the trainer should be read on
  the host, and the ones that cannot be should be listed in AGENTS.md's
  GPU/CUDA quirks section, which now says which of these are fixed.
