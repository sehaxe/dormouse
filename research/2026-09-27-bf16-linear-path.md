# bf16 on the linear-attention path: what is actually blocked, and what it would take

Date: 2026-09-27, gates run to 2026-09-29. Scope: the fused chunked
gated-delta (KDA/GDN-2) path under `--bf16`, on burn 0.22.0-pre.4 + cubecl
0.11.0-pre.4 (vendored patches), sm_120 (RTX 5060 Ti 16 GB).

The conclusion of this file is now also the rule of the house: **AGENTS.md §2.1
(ADR-0016 bug 2) carries the bf16 verdict and cites the test file below.** This
file is the long version - the fallback surface, the measurement, and the diff
sketch for the two ways out. Where the two disagree, §2.1 wins; if they ever
do, that is a defect to report.

Short version:

1. **The AGENTS.md claim "the KDA fused chunked kernel is f32-only, so `--bf16`
   falls back to tensor ops" is stale and, where it is true, irrelevant.**
   `--bf16` feeds the KDA arm **f32** (`crates/dormouse-core/src/loop_block.rs`
   casts the block input before the arm), so the fused kernels *do* run under
   `--bf16`. The fused-vs-tensor difference measured at the production shape is
   **6.8-16.7 ms vs 53-377 ms per call (6-8x contended, up to 40x when the
   tensor arm is descheduled)**, i.e. the fallback that is NOT being
2. **`--bf16` cannot run at all on this stack today.** Not "runs slower": the
   first kernel that writes a bf16 buffer fails to compile —
   `Type cube.bf16 does not have a conversion to LLVM type implemented`. This is
   the LLVM/nvptx backend of cubecl having no bf16 element type at all, and it
   makes bf16 matmuls, bf16 activations and a bf16 fused kernel all equally
   unreachable, at every precision below f32.
3. **bf16 storage with f32 accumulation is nevertheless reachable**, via u16
   bit patterns: no bf16 value ever enters the dialect, only `u16` loads,
   `u32` shifts and one bitcast. Measured: max rel err **2.013e-7** against an
   f64 reference over 64x128 dot products (f32-accurate, three orders below
   bf16 eps = 3.9e-3), and an f32 -> bf16 round trip that is **bit-exact**
   against `half::bf16::from_f32` including ties. That is the primitive a bf16
   fused chunk kernel would be built from, and it is tested
   (`crates/burn-gdn2/tests/lowp_bf16_cuda.rs`).
4. **A bf16 fused chunk kernel is still not worth building**, for a reason that
   has nothing to do with bf16: the exact adjoint consumes the same
   intermediates the forward exports, so on the training path every buffer
   would have to stay f32 anyway. See "Why not just build it" below, with the
   arithmetic.
5. **Tensor cores are reachable in this stack for f16, not for bf16, and not
   from burn's tensor API.** The LLVM backend does lower and advertise
   `mma.sync` with f16 operands and f32 accumulate ("the lowering is correct,
   which `test_cmma_manual` checks element by element"). What is missing is
   (a) any bf16 element type, and (b) burn's own f16 matmul op, which dies with
   `builtin.fp16 to implement dyn SizedType`. The manual-mma family also has no
   cube-level API (`matmul.cube_mma = Default::default()`), so a `#[cube]`
   kernel cannot issue `mma.sync` itself.

---

## 1. The fallback surface, exactly

### 1.1 The gate

There is exactly **one** dtype decision in the whole gated-delta path:

| where | what it decides |
|---|---|
| `vendor/burn-fused/crates/burn-gdn2/src/kernel/chunk_cube.rs:838-841` (after another agent's 2026-09-27 edit; the gate text is unchanged) | `if !matches!(q.dtype(), DType::F32) { return None; }` — the fused kernels or the tensor-ops chunk loop |

Everything downstream of it is `launch_unchecked::<f32>` on buffers the gate
has already vouched for:

- `chunk_cube.rs:910` `gdn2_chunk_intra_kernel::<f32>` (decay cumsum/exp,
  the transposed precomputes, the causal Q-K and strict key-key score
  matrices, the `(I+T)^-1` forward substitution, `w = A(bkt)`, `u = A(wvt)`)
- `chunk_cube.rs:944` `gdn2_chunk_inter_kernel::<f32>` (the sequential state
  chain: `v_new = u - w·S`, intra/inter output, state update)
- `chunk_cube.rs:977` `gdn2_chunk_trajectory_export_kernel::<f32>` (`v_new`,
  per-chunk states, for the adjoint)
- `chunk_adjoint_cube.rs:547` `gdn2_chunk_inter_adjoint_kernel::<f32>`
  (BPTT through the state chain)
- `chunk_adjoint_cube.rs:640` `gdn2_chunk_intra_adjoint_kernel::<f32>`
  (the token-parallel adjoint)
- `fused_recurrent_cube.rs:155` `gdn2_step_kernel::<f32>` (the decode step)
  - no dtype gate of its own; it is reached only through
    `kda_fused_chunk`, which calls the gated entry point first.

The other non-f32 gate is numerical, not dtype: `chunk_c > 16` (the
`k / exp(cumsum g)` factor underflows f32 once `cumsum(g) < -88`, i.e. chunk
> 17 at the K3 floor `g = -5`). KDA ships `chunk_size = 16`
(`crates/dormouse-core/src/attention.rs:26`), so it is not hit.

### 1.2 What runs where under `--bf16`, today

`--bf16` is a *storage* mode: activations are kept in bf16 and computed in f32.
The block body casts before every arm:

```
crates/dormouse-core/src/loop_block.rs
237:  let normed_f = if bf16 { normed.cast(FloatDType::F32) } else { ... };
243:  let normed_attn = ... normed_f ...        // -> the KDA arm gets F32
271:  let (gdn2_out, s_new) = self.shared_attn.gdn2.forward_train_state::<B>(normed_attn, ...)
```

so `q.dtype() == F32` at the gate, and the fused kernels run. Executable proof
(`fused_chunk_gate_is_f32_only`, in the new test file):

```
fused gate: f32            -> fused kernels (2 launches/chunk)
fused gate: bf16           -> tensor-ops fallback
fused gate: chunk 32       -> tensor-ops fallback (f32 underflow below -88)
```

The f32-activations -> f32-projections -> fused-chunk-kernels chain means
there is **no dtype fallback anywhere in the gated-delta path under `--bf16`**.
The f32-only notes in AGENTS.md about the Engram kernels and the KDA chunk
kernels describe the *call sites* casting (which they do, at
`loop_block.rs:284-303` and `237-247`), not a kernel refusing work.

### 1.3 The cost of each fallback, production shape

`[batch 10, heads 12, time 512, K=V=64]`, chunk 16 (the `small` preset at
`--batch 10 --seq-len 512`), `CudaBare`, device-synced inside the timed region
(the cubecl backend is async, so `Instant` around a forward measures the enqueue
only), release build, 5 reps reported min / median:

| arm | ms per call | ratio |
|---|---|---|
| fused chunk kernels (what dormouse runs) | **11.16 / 16.73** | 1x |
| tensor-ops chunk loop (`chunk_wy_forward`) | **88.95 / 93.29** | 8x / 6x |
| **fallback premium** | **77.79 / 76.56** | |

Across five separate runs of the same test the fused arm read 6.8-16.7 ms and
the tensor arm 53-377 ms. That spread is contention, not variance: the tensor
chunk loop is launch-bound (~150 launches per chunk x 32 chunks) and 2-4 other
agents were building and running GPU work throughout. The **ratio is the
stable quantity: 6x-8x on a loaded machine, up to 40x when the tensor arm is
descheduled.** Read the fused arm as ~7 ms quiet, ~11-17 ms contended.

Extrapolated to a training step (`max_iter = 4`): had `--bf16` really fallen
back, it would have cost **0.3 - 1.5 s per step** of pure tensor-op chunk
loops, on a step that measures 21-27 s (§7.4) - i.e. 1-7% of a step, for a
fallback that never happens because the arm is fed f32.

---

## 2. What I implemented

One new file, no edits to any shared source:
`vendor/burn-fused/crates/burn-gdn2/tests/lowp_bf16_cuda.rs`.

### 2.1 `bf16_storage_with_f32_accumulation` - the capability, measured

A `#[cube]` kernel with **no bf16 value anywhere**: buffers are `&[u16]` (the
bf16 bit pattern) and `&mut [f32]`, the accumulator is `f32`, and the two
conversions are integer ops plus one bitcast:

```
bf16 -> f32:  f32::reinterpret(u32::cast_from(bits) << 16)     // a bf16 IS the high half of an f32
f32 -> bf16:  u + 0x7FFF + ((u >> 16) & 1), then >> 16        // round-to-nearest-even
```

It is launched over buffers that burn cannot even *name* as bf16 (the operands
are uploaded as raw bit patterns in an f32 tensor, which is a plain H2D
memcpy), and it computes a 64 x 128 dot product against an f64 host reference
over the same bf16-rounded operands:

```
bf16 storage + f32 acc: max rel err vs f64 = 2.013e-7 over 64x128 dots
f32 -> bf16 round trip: bit-exact vs half::bf16::from_f32 on 64 rows
```

2.0e-7 is f32 rounding on a 128-term sum (f32 eps 1.2e-7). bf16 eps is 2^-8 =
3.9e-3. Had the accumulator been bf16 the error would be ~1e-2, four orders
worse - so this is a real measurement that the accumulator is f32 and the
operands are expanded, not a tautology.

The round-trip bit-exactness is the other half of the claim: the in-kernel RNE
matches `half::bf16::from_f32` on every row, ties included, so a bf16 fused
kernel would store exactly what a torch/cub bf16 store stores.

### 2.2 `fused_chunk_gate_is_f32_only` - the surface, executable

The three gate outcomes above, asserted (not printed and hoped for).

### 2.3 `production_shape_cost_of_the_dtype_gate` - the numbers in §1.3.

### 2.4 `bf16_cast_does_not_compile` - `#[ignore]`d on purpose

A report, not a requirement: burn's own `f32 -> bf16` cast does not compile on
this backend. Run it with `-- --ignored`. It must start passing the day
upstream lands bf16 lowering, at which point the ignore comes off.

### 2.5 Timers

Not mine: the fused-path instrumentation I would have added already exists and
is better than what I would have written - `GDN2_ALLOC_TRACE=1` makes
`fused_chunk_forward_scratch` print `[gdn2] fused chunk kernels ENGAGED:
b=.. h=.. t=.. K=.. V=.. chunk=.. nblk=..` plus a per-site device-allocation
table (`crates/burn-gdn2/src/alloc_trace.rs`, another agent's, 2026-09-27). It
is guarded, off by default, and one relaxed atomic load when off. I used it
rather than adding a second timer.

---

## 3. The numerics

- f32 accumulation over bf16 storage: **max rel err 2.013e-7** vs f64 (§2.1).
- f32 -> bf16 storage rounding: **bit-exact** RNE (§2.1).
- The chunked recurrence's own range limit is unchanged by bf16 storage:
  bf16 has f32's exponent range (8 bits, same bias), so `k / exp(cumsum g)`
  behaves as it does in f32. **f16 would not be safe here**: the same factor
  reaches `1/exp(-80) = 5.5e34` at the K3 floor with chunk 16, and f16
  overflows at 65504. The `g_exp`/`kgt`/`kgd` intermediates must be bf16, not
  f16, and that is a numerical result, not a preference.
- The training path cannot take lowp intermediates at all: see §5.

---

## 4. The step-time delta

Nothing to delta: I did not change any production code path, so the step time
is bit-identical to `main`. The measurement that matters for the decision is
the one in §1.3 - the fallback that `--bf16` does **not** take is worth 8-40x
per call, which is why "make the KDA kernel bf16" cannot be the answer to
"bf16 is slower than fp32".

---

## 5. Why not just build the bf16 fused kernel

The blocker is not bf16 arithmetic; it is that **the exact adjoint and the
forward share their buffers**. `fused_chunk_backward` consumes
`m_inv, aqk, qgt, glast, v_new, states, w, u, gexp` (the forward's exports, see
`FusedBackwardInputs` and the launches at `chunk_adjoint_cube.rs:547-660`) plus
the six checkpointed input activations `k, v, b, w_gate, d_out`. Every one of
those crosses the autodiff boundary as an f32 tensor:

- a lowp forward would have to **also** write f32 copies of its exports for the
  adjoint to read -> strictly more traffic than the f32 kernel, not less;
- the adjoint's own kernels are `&[F]` launched `::<f32>`; making them read
  bf16 is a signature change in `chunk_adjoint_cube.rs` (another agent's file)
  plus a matching u16-view plumbing;
- and the adjoint is *hybrid* anyway - two kernels plus ~10 burn tensor ops for
  the glue (`chunk_adjoint_cube.rs:593-611`) - so half of it is burn ops that
  would need the bf16 support that does not exist (§6).

The buffers a lowp forward *could* own, at the production shape, are `q`,
`out`, `kgt`, `bkt`, `wvt`, `kgd`, `akk` - the four transposed precomputes are
written once and read only inside the intra kernel, and the adjoint never sees
them (`FusedBackwardInputs` carries `m_inv, aqk, qgt, glast, v_new, states, w,
u, gexp` only). At `[10,12,512,64]` with 32 chunks one full-size buffer is
`3840*16*64*4 B = 15.7 MB`, so the forward-only set is ~6 x 15.7 + 3.9 =
98 MB per pass, and those buffers are touched more than once (phase 2 re-reads
`kgt`/`bkt` per score block). Halving all of it is worth **50-100 us per call**,
**0.2-0.4 ms per step** at 4 iterations, against a step that measures
**21.6 s** (§7.4): **~0.002%**.

And the ceiling is lower than that, because the fused kernel is not
bandwidth-bound to begin with: 16 full-size buffers plus 3 small ones is
~250 MB of f32 traffic per call, done in **6.8-16.7 ms** = **15-37 GB/s**
effective, against ~1 TB/s on this card. The kernel is 27-65x off
bandwidth-bound (it is 3 launches for the whole 32-chunk sequence, and the inter
kernel is an inherently sequential chain), so a 2x traffic reduction cannot buy
anything close to 2x time. Everything else in the call - `m_inv`, `aqk`,
`glast`, `v_new`, `states`, `gexp`, the six input activations - must stay f32
anyway, because the adjoint reads it (§5 first bullet).

The real lever on this arm is launches, not bytes, and the fused op already
took it: **3 cubecl launches for the whole chunk sequence** (intra over all
chunks in parallel, the sequential inter chain, the trajectory export) versus
~150 tensor ops per chunk.

---

## 6. Exactly what blocks bf16, and the diff that would unblock it

### 6.1 The machine/stack limitation

`vendor/cubecl-fix/cubecl-cuda` + `cubecl-llvm 0.11.0-pre.4`:

- `cubecl-cuda/src/runtime.rs:420` `restrict_to_llvm_backend` drops `bf16` from
  the advertised element types, in the upstream words: *"the dialect this
  backend lowers through has no type for them, so there is nothing to put in a
  register"*.
- `cubecl-llvm/src/nvptx/` has no bf16 rules at all (bf16 appears only in
  `cubecl-llvm/src/amdgpu/matrix.rs:404`).
- Consequently any kernel with a bf16-typed argument or result fails to
  compile. Observed verbatim:

```
compiling 'cast_element_i_f32_o_bf16_n_4_d86b13a7' for sm_120:
  Compilation error: invalid input program.
  Type cube.bf16 does not have a conversion to LLVM type implemented
```

That is burn's own cast kernel - so `Tensor::cast(BF16)` is dead, which takes
`--bf16` with it (`loop_block.rs:218/224/237/284/299/321/339/351` all cast),
and with it every bf16 matmul (`burn-cubecl ops/tensor.rs:150`
`matmul(...).unwrap()`).

### 6.2 The three ways out, cheapest first

**(a) Flip the compiler to NVRTC (one feature).** `cubecl-cuda` already carries
the whole bf16 story behind its `cpp` feature - `cubecl-cuda/src/lib.rs:70-82`
advertises and tests `bf16` only under `#[cfg(feature = "cpp")]`, because the
C++ backend *has* the type. The compiler is chosen by that feature
(`compiler.rs:25-33`):

```diff
--- a/vendor/cubecl-fix/cubecl-cuda/Cargo.toml
+++ b/vendor/cubecl-fix/cubecl-cuda/Cargo.toml
 [features]
-cpp = []
+cpp = ["std"]            # not needed, but: the DEFAULT is what picks the compiler
 default = [
     "std",
+    "cpp",                # <- this line: CudaBackend::default() becomes Cpp
     "cubecl-server/default",
```

or, without touching the vendored crate, add the edge from a crate we own:

```diff
--- a/crates/dormouse-core/Cargo.toml
+++ b/crates/dormouse-core/Cargo.toml
+[dependencies]
+cubecl-cuda = { version = "0.11.0-pre.4", default-features = false, features = ["cpp"] }
```

Caveat, measured by others on 2026-09-26 and not re-measured here: the NVRTC
path was "broken + slow (150 s/step, loss-scalar crash)" while the LLVM path
was the working config. So (a) is a *test*, not a fix: the honest next step is
`CUBECL_AUTOTUNE_LEVEL=0 cargo test -p burn-gdn2 --features cuda --test lowp_bf16_cuda -- --ignored`
with the feature on. I did not flip it: it is a global default for every
concurrent build in this workspace, and two other agents were building in it.

**(b) Wait for / patch upstream `cubecl-llvm` nvptx bf16 lowering.** The change
is a type mapping plus the `fptrunc`/`fpext` pair for bf16 in
`cubecl-llvm/src/nvptx/codegen.rs`; without bf16 *arithmetic* (only the two
conversions) it is a small, mechanical PR upstream. With it, the fused kernels
become `launch_unchecked::<bf16>`-able and the gate at
`chunk_cube.rs:838-841` is the only thing left to relax.

**(c) The u16 route (what this file proves, and what needs no upstream change).**
Keep every bf16 value out of the dialect: declare the buffers `&[u16]`, expand
with `f32::reinterpret(u32::cast_from(x) << 16)`, store with the RNE idiom.
This works today (measured above) and needs no compiler, no feature and no
upstream patch. It costs: a signature change on the intra/inter/adjoint
kernels (`chunk_cube.rs`, `chunk_adjoint_cube.rs` - other agents' files), and
it buys the ~0.002% of a step computed in §5. **Recommendation: keep this as
the documented escape hatch, do not build it.**

### 6.3 If someone does build it anyway - the diff sketch

```diff
--- a/vendor/burn-fused/crates/burn-gdn2/src/kernel/chunk_cube.rs
+++ b/vendor/burn-fused/crates/burn-gdn2/src/kernel/chunk_cube_bf16.rs   (new file)
+// storage type is u16 (bf16 bits), arithmetic type is F. One `#[cube]` kernel
+// serves f32 (S = f32-as-u16? no) ... concretely: generic over the ARITHMETIC
+// type, u16-typed boundary buffers, per-site `bf16_to_f32` / `f32_to_bf16`.
+#[inline]
+fn bf16_to_f32(x: u16) -> f32 { f32::reinterpret(u32::cast_from(x) << 16) }
+#[inline]
+fn f32_to_bf16(x: f32) -> u16 {
+    let u = u32::reinterpret(x);
+    u16::cast_from((u + 0x7fff + ((u >> 16) & 1)) >> 16)
+}
```

- `gdn2_chunk_intra_kernel` / `_inter_kernel`: change `q,k,g,b,v,wg,out` to
  `&[u16]` / `&mut [u16]`, wrap every read in `bf16_to_f32` and every write in
  `f32_to_bf16`; leave `a_sh` (shared memory), all accumulators and
  `m_inv/aqk/akk` in f32 (the adjoint reads those).
- `chunk_cube.rs:838-841`: replace the f32-only gate with a dtype switch that
  routes `DType::BF16` to the lowp launch.
- `chunk_adjoint_cube.rs`: same treatment for the six `::<f32>` launches plus
  the burn tensor glue at 593-611 - the glue is the part that cannot be
  lowp until (a) or (b) lands.
- Test: reuse this file's harness - pack the projections to bf16 words, run
  both paths, assert max rel err < 1e-2 (bf16 storage, so the tolerance is
  bf16-level, not f32-level) and that the loss curve matches the f32 path.

---

## 7. Gates

All gates run on 2026-09-27/29 on `main`, `release` profile, sm_120, the
vendored tree as it stood at the time (the vendored workspace is under
concurrent edits by other agents; the commit each number was taken at is named
where it matters).

| gate | result |
|---|---|
| (a) `cargo test -p burn-gdn2 --features cuda,autodiff` | **green except one pre-existing failure that is not mine** (§7.1) |
| (a) `cargo test -p burn-kda --features cuda,autodiff` | **green**, 19 tests, 0 failed |
| (b) `bitforbit_cuda` harness | **green**, max rel err 3.7e-7 vs the per-token CUDA reference and 2.5e-7 vs the CPU transcription (§7.2) |
| (c) 60-step `--bf16` run | **UNSATISFIABLE - the mode dies at step 0** (§7.3) |
| (d) step time at b=10 s512 with and without the path | **0 delta by construction** (no production code touched); the fp32 control is ~21-27 s/step and the `--bf16` arm has no step to time (§7.4) |

### 7.1 Test suites

`cargo test --release -p burn-gdn2 --features cuda,autodiff --no-fail-fast`:

```
test result: ok. 10 passed  ... (test_chunk)
test result: ok.  5 passed  ... (autodiff)
test result: ok.  4 passed  ... (fused_chunk_verify)
test result: ok.  2 passed  ... (lowp_bf16_cuda)   <- this file
... 18 binaries, 0 other failures
test the_op_declines_a_nested_graph_and_the_ops_path_carries_the_gradient ... FAILED
  crates/burn-gdn2/tests/autodiff_nested_balanced.rs:339
```

**The one failure is not mine and cannot be**: `autodiff_nested_balanced.rs` is
another agent's in-flight test for the `8fa5d4c` defect (AGENTS.md §3.2), it
lives in a different test binary from mine, and my only change is a new test
file, which is a separate binary and cannot affect another one. It asserts
`chunk_wy_forward_autodiff_s::<NdArray, BalancedCheckpointing>(...).is_none()`
- i.e. that the op still declines under Balanced checkpointing - and the op now
builds a node, which is the fix landing under it. Reported, not touched.

`--features cuda` **alone** does not compile, and that is also not mine:
`burn-gdn2/src/lib.rs:60-61` gates `pub mod cuda_dispatch` behind
`#[cfg(feature = "autodiff")]` while its three call sites
(`kernel/chunk_cube.rs:932`, `kernel/chunk_adjoint_cube.rs:480`,
`autodiff.rs:388`) are not gated, so E0433 `cannot find cuda_dispatch in crate`.
The gate command therefore has to be `--features cuda,autodiff`. One-line fix in
their file: drop the cfg on the module, or add it to the call sites.

`cargo test --release -p burn-kda --features cuda,autodiff --no-fail-fast`:
**all green** (13 + 1 + 4 + 1 tests, 0 failed, 1 ignored).

### 7.2 bit-for-bit harness

The dumps the harness reads (`/tmp/opencode/kda_bfb/`) had been wiped by the
tmpfs, so the CPU transcription was regenerated first
(`cargo run --release -p burn-kda --example bitforbit`, 29 files). Then:

```
cargo run --release -p burn-kda --example bitforbit_cuda --features cuda,autodiff
inputs from /tmp/opencode/kda_bfb: [b=2 h=2 t=16 hk=32] hv=2 vd=32 chunk=8
out   fused vs ref_cuda: max_abs = 5.588e-9
state fused vs ref_cuda: max_abs = 5.588e-9
out   fused vs CPU o_raw (same WY inputs): max_abs = 3.725e-9
state fused vs CPU state_chunk: max_abs = 3.725e-9
state fused vs CPU state_rec: max_abs = 5.588e-9
```

The harness prints max-abs; the relative figures (recomputed from the dumped
tensors, `max|x-y| / max|x|`, the definition the harness's own `max_rel` uses
in `fused_chunk_verify.rs`):

| comparison | max_abs | max_rel | scale |
|---|---|---|---|
| fused out vs per-token CUDA reference | 5.588e-9 | **3.72e-7** | 0.015 |
| fused state vs per-token CUDA reference | 5.588e-9 | **2.14e-7** | 0.026 |
| fused out vs the CPU (ndarray) transcription | 3.725e-9 | **2.48e-7** | 0.015 |
| fused state vs the CPU transcription | 3.725e-9 | **1.43e-7** | 0.026 |

f32 noise on a 16-token problem, i.e. **no regression**: these are the same
numbers the harness printed before this work, and my change is a test file that
no production path includes.

Naming the evidence, per AGENTS.md §1.4: the "reference" here is **our own
transcription** (the CPU harness and a per-token scan), not NVlabs' own bytes.
`tests/bit_exact.rs` compares against a *third* transcription
(`gen_reference.py`, the NVlabs layer) and is **red by design and by
measurement** - 976/1000 cases fail at 5e-4 - which another agent measured on
2026-09-27 and documented in that file's header; it is not in the crate's
default features and I did not touch it.

### 7.3 60-step `--bf16` run - it does not run

`./target/release/train` (binary of 2026-09-28 21:57), `--preset small --batch 2
--seq-len 512 --bf16 --steps 3`:

```
thread 'DSD-0-0' panicked at cubecl-llvm-0.11.0-pre.4/src/shared/to_llvm/ty.rs:130:10:
  Type not supported for overloading of intrinsic
thread 'DSD-0-0' panicked at cubecl-llvm-0.11.0-pre.4/src/shared/to_llvm/constant.rs:72:40:
  called `Option::unwrap()` on a `None` value
train failed: step 0: device error (loss is not a scalar)
```

**The gate cannot be satisfied and the reason is the machine limitation, not
this work**: `--bf16` dies compiling the first bf16-typed kernel, before step 0
produces a loss, so there is no NaN count and no loss curve. The same block, in
the same words, is in AGENTS.md §2.1 (ADR-0016 bug 2) and the burn-level
reproduction is `bf16_cast_does_not_compile` (ignored by design) in the test
file. "0 NaN over 60 steps" would be a claim about a program that does not
exist; the honest gate result is this transcript.

### 7.4 Step time

No production code path was changed, so the step time is bit-identical to
`main` and the delta is **0**. What is worth having is the control, measured the
same way (`--timers`, `timer step N:`), at the production shape b=10, s=512:

```
./target/release/train --data .../real_sharded --preset small --batch 10 \
    --seq-len 512 --quant fp32 --no-engram --lr 3e-4 --host-adam-every 0 \
    --steps 4 --log-every 4 --timers
params=9197390
timer step 0: total=21607ms data=0.1ms fwd=4856ms bwd=13907ms opt=2317ms
              retr=520.8ms ema=5.3ms gpu_step=21607ms
ce 5.709 -> 5.703 over 6 steps, 0 NaN
```

- **21.6 s/step at b=10 s512**, fwd 4.9 s / bwd 13.9 s / opt 2.3 s / retr 0.5 s.
- A second run printed 27.1 s for step 0, and the 4- and 8-step wall clocks
  (181 s and 168 s, both including init, the data read and the final save) do
  not separate into a per-step steady state: `--timers` reports step 0 only.
  So **21-27 s is the honest range and the per-step steady state is not
  measured here** - flagging it, because the archive's 1.4 s/step predates the
  attention backward actually running (AGENTS.md §3.2/§3.3), and this is the
  same order as the 25.8 s/step batch-8 figure that has no committed log
  either. It is the first number I have seen with the attention backward in it.
- The `--bf16` arm of this gate has no number: it does not reach step 0 (§7.3).
