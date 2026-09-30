# Is `burn-rmsnorm`'s fused CUDA kernel reachable, and is it correct?

**Lane:** rmsnorm-kernel · **worktree:** `wt/rmsnorm-kernel` off `c3314e9` ·
**date:** 2026-09-30 · **box:** RTX 5060 Ti, one GPU, shared with a step-time
lane (it was idle for every measurement here; both readings below name a GPU
that was at 2 % and 428 MB when they were taken).

The question was AGENTS.md §3.3's: *"The fused RMSNorm kernel never engages on
the trainer's backend — the eval line shows `norm=0/N` because an autodiff
tensor cannot be handed a bare kernel."* The claim is **correct**, and the
mechanism is **not** what the sentence implies. Two things were wrong with the
state of the lane, and both are now measured rather than argued:

1. **"Never engages" was scoped to a shape nobody had ever run.** The kernel
   engages on any *bare* (non-autodiff) cubecl device, and the crate's own
   `BURN_DEVICE=cuda` route is exactly that. It had never been run because
   nothing enabled the `cuda` feature for a test of this crate.
2. **The kernel did not compile.** It had never produced a number, on any
   device, in the life of the project — and when it was finally launched, the
   lowered module failed LLVM verification before a byte was written. This is
   the finding that mattered: a "dead" kernel is cheap, a kernel that is dead
   *and broken* is one `?` away from a training run.

---

## 1. Reachability: the trace

`RMSNorm::forward` (`src/lib.rs:41-66`) asks the fused path on every call under
`feature = "cuda"` and falls back when `fused::rmsnorm_cuda` returns `None`:

```rust
#[cfg(feature = "cuda")]
{
    let [b, t, _] = x.dims();
    if let Some(out) = crate::fused::rmsnorm_cuda::<burn_cubecl::CubeBackend>(
        x.clone().reshape([b * t, d]), self.weight.val().clone(), self.eps,
    ) { return out.reshape([b, t, d]); }
    crate::fused::SKIPPED.fetch_add(1, Relaxed);
}
```

The decline happens one line into `rmsnorm_cuda`, at
`x.try_into_primitive::<burn_cubecl::CubeBackend>()` (`src/fused.rs:110` →
`:123`). In burn 0.22.0-pre.4 a `Tensor` is **not** generic over a backend
(`burn-tensor .../tensor/api/base.rs:76`: `pub struct Tensor<const D: usize, K =
Float>`); routing is a runtime dispatch, and this call is a runtime *demotion*:

* `DispatchTensor: DispatchKindConversion<CubeBackend>` — satisfied, because
  `burn_cuda::Cuda` is a re-export of `burn_cubecl::Cube`
  (burn-cuda-0.22.0-pre.4 `src/lib.rs:10`: `pub type Cuda = burn_cubecl::Cube;`),
  so `Device::cuda(0)` and the kernel's `B` are the *same* type, not two
  runtimes. That re-export is the reason a bare CUDA device is not a second
  backend and is therefore eligible at all.
* `DispatchKindConversion::try_into_backend` — this is the wall.
  burn-dispatch-0.22.0-pre.4 `src/tensor.rs:481-487`, verbatim:

  ```rust
  if tensor.autodiff != DispatchAutodiffContext::Disabled {
      return Err(format!("Expected concrete {} backend with disabled \
                          autodiff context, got {:?}", stringify!($backend), …));
  }
  ```

**So the deciding input is the device's autodiff context, not the tensor's
identity and not the backend type.** Two shapes, both real:

| shape | asked | ran | who has it |
|---|---|---|---|
| bare `Device::cuda(0)` | +1 | **+1** | the crate's own `BURN_DEVICE=cuda` test route; any gradient-free forward |
| `Device::cuda(0).autodiff()` | +1 | **+0** | every training forward, and the trainer's `model.valid()` eval |

### The `norm=0/N` claim, reproduced

Not re-derived — read off the trainer's own logs, on the GPU, on real runs.
`~/logs/train_nokda.log:13` reads `norm=0/1569` at step 500 and
`norm=0/3129` at step 1000 (`:19`): 1560 asks per 500 steps, **zero** runs. The
printed field is `norm={asked−skipped}/{asked}`
(`crates/dormouse-train/src/lib.rs:1543-1550`), so the first `0` is the count of
*runs* and the second the count of *asks*. 22 distinct `norm=0/N` readings
exist across five logs, and none has a non-zero first field.

One correction to the claim's wording, which is worth a docs fix by the owner:
the sentence implies the trainer's *autodiff* device is the reason. It is, but
that is not the same as "never engages" — and the eval snapshot is worth
naming, because it is the one shape where the trainer came closest:
`model.valid()` strips the autodiff context from the params
(`burn-core .../module/param/tensor.rs:248`, `val().without_autodiff()`), so
the *weights* in the eval are plain. The eval still declines, because the
*input* to the norm is the hidden state and the counters say so. That is now a
test rather than a hope: `the_fused_kernel_declines_on_the_trainers_eval_snapshot`.

## 2. The kernel did not compile

First launch, 2026-09-30, on a bare CUDA tensor:

```
the lowered module does not verify: Compilation error: verification failed.
Expected operand type llvm.ptr, but found builtin.integer
```

which surfaces at the next read as `The bytes were never written … A
compilation error happened during launch`. It is a **LOWERING** failure, not a
runtime one, and AGENTS.md §2 says what that means on this box: the CUDA
backend is pliron → LLVM → NVPTX, and `restrict_to_llvm_backend` advertises
only what that dialect can lower.

Cause: **`Shared::new_slice` sized by a runtime value.**

```rust
let threads = 256usize;                      // runtime
let mut partial = Shared::<[F]>::new_slice(threads);   // …used as an allocation size
```

Every kernel in this tree that runs sizes its shared memory from a
`#[comptime]` value — `burn-attnres/src/fused_attnres.rs:127` (`#[comptime]
threads`), `burn-gdn2/src/kernel/chunk_adjoint_cube.rs:61` (`c * c` from
`#[comptime] chunk_c`) — and this one did not. The fix is the same pattern, 4
lines, and it also removed the possibility of `CubeDim` and the reduction
width disagreeing:

```rust
pub const THREADS: u32 = 256;              // one source of truth
fn rmsnorm_kernel<F: Float>(
    …, eps: f32, #[comptime] d: u32, #[comptime] threads: u32,
) { let threads = threads as usize; … }
rmsnorm_kernel::launch_unchecked::<f32>(…, CubeDim::new_3d(THREADS, 1, 1), …, d as u32, THREADS);
```

## 3. Correctness, against the fixture

(measured numbers below)

## 4. What this does NOT establish

* **No backward, and that is the routing constraint.** `rmsnorm_cuda` returns
  a fresh `Tensor::<2>::empty` and the launch writes raw handles into it, so the
  result carries no graph. If the kernel were ever reached from an autodiff
  input, the arm would return a leaf, receive **no gradient**, and every loss
  curve would still look healthy — the `burn_gdn2` defect, verbatim
  (`docs/ORACLE.md`, and AGENTS.md §3.2's second retraction). The dispatch
  refusal above is the only thing preventing it. **Wiring this into a training
  forward means writing the adjoint first**, not relaxing a `?`. The routing
  decision is the owner's; this lane did not touch it.
* **The runtime is not the trainer's.** The CUDA test compiles in the
  `vendor/burn-fused` workspace, where the repo root's `[patch.crates-io]`
  (which maps five vendored cubecl crates) does not apply, so the kernel ran on
  the **registry `cubecl 0.11.0-pre.4`** while the trainer runs the vendored
  fork. The arithmetic under test is our own source and does not depend on the
  runtime version; the claim is bounded by that and by nothing else. The
  existing CPU/tensor oracle has the same boundary in a weaker form.
* bf16/f16 are declined by design (`fused.rs:128`): the kernel is hard-coded to
  f32 lanes, and a bf16 buffer read as f32 would be garbage. Correct refusal,
  untested, and out of scope here.
* arXiv:1910.07467's authors ship no code, so this is tier (a) against *the two
  public reference implementations of the mechanism*, not against the paper.
  A shared misreading of Zhang & Sennrich survives it (stated in
  `docs/ORACLE.md` §3, and inherited unchanged from the sibling file).

## 5. Follow-ups (reported, not done)

1. `src/fused.rs:31-34` — `#[cfg(not(feature = "cuda")))] pub fn calls()`
   returns a hard-coded `(0, 0)`, which is **indistinguishable** from "the
   kernel was asked zero times". The doc comment claims the reader can tell the
   two apart; the printed field cannot. A `compiled: bool` in the tuple, or
   `asked = u64::MAX`, would restore the distinction. Not touched: it changes a
   public signature a caller in another crate holds.
2. AGENTS.md §3.3's sentence should say *why* (the autodiff context, and the
   fact that the kernel DOES run on a bare device and did not compile until
   2026-09-30), not only that the eval line reads `norm=0/N`. Same for
   `docs/ORACLE-TIERS.tsv`'s rmsnorm row, which is updated in this commit for
   the test file but whose prose still carries the old scope.
3. The kernel is **correct but unclaimed**: nothing in the trainer uses it, and
   after this commit nothing will until an adjoint exists. If the owner decides
   the speedup is not worth an adjoint, `src/fused.rs` is ~150 lines of
   verified-but-unused code and the honest move is to say so in its header
   rather than leave "fused" implying a live path.

## 6. Reproduce

```bash
# the kernel path, all 12 fixture cases, on a GPU
cargo test -p burn-rmsnorm --test rmsnorm_kernel_cuda --features cuda -- --nocapture
# the crate's own BURN_DEVICE=cuda route (the shape the kernel runs on)
BURN_DEVICE=cuda cargo test -p burn-rmsnorm --features cuda --lib
# the red demo: five wrong fused kernels, restored byte-identical
bash tests/oracle/mutate_fused_kernel.sh
```
