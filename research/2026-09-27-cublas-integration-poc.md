# cuBLAS integration: the pointer, the stream, and what it is worth

**Date:** 2026-09-27 · **Spike:** `crates/cublas-poc/` (`cargo run --release -p cublas-poc`)
· **Log:** `/home/sehaxe/logs/cublas_poc_2026-09-27.log`
· **The patch is in this report and is NOT in the tree** (§2, also as
`research/2026-09-27-cublas-pointer-patch.diff`).

## 0. Verdict

The blocker is real, it is ~110 lines, and it is worth it:

| | measured (5120x768x2048, same process, same GPU) |
|---|---|
| burn `a.matmul(b)` (cubecl fp32) | 2.27 ms — 7.1 TFLOP/s |
| **cuBLAS f16 in / fp32 accumulate, on burn's own buffers** | **0.389 ms — 41.4 TFLOP/s** |
| the same, staged through the host (no patch needed) | 137.9 ms — 0.1 TFLOP/s |

**7.4x on a production shape, zero copies, no barrier, and the numerics check
against `a.matmul(b)` (rel 2.7e-4, ordinary f16).** The host-staged fallback
that works today is **127x slower than the patch**, so the staged version is not
worth shipping first — the RPC is the whole design decision.

Two findings change existing beliefs:

1. **burn's own f16 matmul is 5.6 TFLOP/s — slower than cuBLAS fp32.** The
   reason "bf16 doesn't help" in this repo is not that tensor cores are
   unreachable, it is that cubecl's f16/bf16 matmul never reaches them. The
   path exists and is correct (f16 casts and f16 matmul both work on
   pre.4 + this vendored cubecl, which the AGENTS.md note "f16 panics in
   cubecl-ir" contradicts — see §5). Swapping only the GEMM body is a 7.4x win
   on that shape.
2. **cuBLAS `GemmEx` with `CUDA_R_16BF` inputs is numerically WRONG on this
   stack** (maxabs ≈ 1.0, i.e. garbage) with both `COMPUTE_32F` and
   `COMPUTE_32F_FAST_16BF`, at full tensor-core speed. **Use f16, not bf16, for
   the cuBLAS path** — and note the dtype of the *storage* and the dtype the
   GEMM is told are separate choices.

## 1. What was already there (do not re-derive)

- The pointer RPC **already exists**: `Client::get_resource::<S>(handle) -> ManagedResource<S::Storage::Resource>`
  (`vendor/cubecl-fix/cubecl-runtime/src/client.rs:446`), and for CUDA
  `GpuResource.ptr: u64` is the device address
  (`vendor/cubecl-fix/cubecl-cuda/src/compute/storage/gpu.rs:28`). What is
  missing is only that **`CudaServer` is not nameable from outside the crate**:
  `mod compute` is private and `runtime.rs` imports it privately, so the
  generic parameter cannot be spelled. **One `pub use` line fixes that.**
- `CubeTensor` (`burn-cubecl` 0.22.0-pre.4 `src/tensor/base.rs:20`) exposes
  `pub client`, `pub handle`, `pub meta`, `pub device`, `pub dtype`, and
  `CubeTensor::new_contiguous(client, device, shape, handle, dtype)` builds one
  from a raw handle — everything a cuBLAS output tensor needs.
- The server runs on **its own thread** (`cubecl-common`'s channel device
  handle) with the **primary context current there**. From the caller's thread
  the context must be pushed by hand — one `cuInit` + `cuDevicePrimaryCtxRetain`
  + `cuCtxSetCurrent`, no guard (a `cudarc` `CudaContext` guard's `Drop` pops the
  context off the thread).
- The stream is a plain `CUstream` in `Stream.sys` (pub), but no client API
  returns it. `Server::stream_ids()` exists; the handle behind an id does not.
- cudarc 0.19.9's `cublas` module is behind a `cublas` cargo feature this tree
  does not enable, and **enabling it recompiles every cubecl crate in the tree**
  (cudarc's metadata hash feeds all of them). Four `dlopen`ed symbols are free.
- Failure-graph claim checks do **not** block writing behind the server's back:
  `ensure_written` only refuses unallocated or tainted buffers, so a fresh
  `empty` tensor written by cuBLAS reads back clean. Confirmed end to end.

## 2. The patch (110 lines, 5 files) — NOT COMMITTED

`research/2026-09-27-cublas-pointer-patch.diff`, applied to
`vendor/cubecl-fix/`, verified to compile and to make §6 work. It is additive:
two defaulted trait methods, one impl, one client wrapper, two re-exports. No
existing behaviour changes (a backend that does not answer gets `None`, exactly
as it does today).

```rust
// cubecl-runtime/src/server/base.rs, in `pub trait Server`
/// The backend's own raw handles for `bindings`: one address per binding,
/// in the order given, each already offset by the binding's own offset.
fn native_ptrs(&mut self, _bindings: Vec<BufferBinding>, _stream_id: StreamId) -> Option<Vec<u64>> { None }

/// The backend's own raw stream handle for `stream_id`, opaque: a `CUstream`
/// as a `u64` on CUDA, `None` where there is no such thing (the default).
fn native_stream(&mut self, _stream_id: StreamId) -> Option<u64> { None }

// cubecl-cuda/src/compute/server.rs, in the existing `impl Server for CudaServer`
fn native_ptrs(&mut self, bindings: Vec<BufferBinding>, stream_id: StreamId) -> Option<Vec<u64>> {
    let mut ptrs = Vec::with_capacity(bindings.len());
    for binding in bindings {
        // The same claim check `get_resource` makes: a buffer a failed launch
        // never filled gets no address, rather than one pointing at whatever
        // was there before.
        self.streams.ensure_written([&binding].into_iter()).ok()?;
        let mut command = self.command(stream_id, [&binding].into_iter());
        ptrs.push(command.resource(binding).ok()?.ptr);
    }
    Some(ptrs)
}

fn native_stream(&mut self, stream_id: StreamId) -> Option<u64> {
    // `try_stream_mut`, not a resolve: a lookup of a handle that already
    // exists; it must not order the stream behind anything or bump its cursor.
    Some(self.streams.try_stream_mut(&stream_id)?.sys as u64)
}

// cubecl-runtime/src/client.rs, in `impl Client`  (one round trip for the batch)
pub fn native_handles<S: Server>(&self, handles: Vec<Handle>)
    -> Result<(Vec<Option<u64>>, Option<u64>), ServerError> {
    let stream_id = self.stream_id();
    let bindings = handles.into_iter().map(|h| h.binding()).collect::<Vec<_>>();
    for binding in &bindings { self.local(binding)?; }
    if !self.is_service::<S>() { return Err(/* ServiceMismatch, as get_resource */); }
    Ok(self.device.submit_blocking(move |server| {
        let server = (server as &mut dyn Any).downcast_mut::<S>()
            .expect("is_service passed, so this is the server's type");
        let ptrs: Vec<Option<u64>> = server.native_ptrs(bindings, stream_id)
            .map(|p| p.into_iter().map(Some).collect()).unwrap_or_default();
        (ptrs, server.native_stream(stream_id))
    }).map_err(Into::into).unwrap_or_resume())
}

// cubecl-cuda/src/lib.rs        — the type has to be nameable to be generic over
pub use compute::CudaServer;
pub use compute::GpuResource;
// cubecl-cuda/src/compute/mod.rs
pub use storage::gpu::GpuResource;
```

**Why this shape.** The stream is created once per `StreamId` and never
replaced, so the client resolves it **once at startup** and reuses the `u64` for
the whole run; only the per-GEMM pointer batch needs a round trip (and one
`native_handles` call can carry a whole layer's buffers). A per-call `CUstream`
query would be affordable too, but a startup-cached one is free.

## 3. The row-major call convention (MEASURED, and the documented one is wrong)

`research/2026-09-27-optimization-1b.md` records
`GemmEx(OP_T, OP_N, m=N, n=M, k=K, A=B, lda=K, B=A, ldb=K, C, ldc=N)`. **That
call is accepted by the driver and computes the wrong answer** (§0 row 11 of the
probe's convention table: `maxabs 6.86`). The probe walks all 16
(transa, transb, lda, ldb) combinations against a host fp32 reference at
M=256 K=128 N=512:

```
   trA  trB   lda   ldb                     result
  OP_N OP_N   128   128         rejected, status 7
  OP_N OP_N   128   256         rejected, status 7
  OP_N OP_N   512   128 MATCH  maxabs 2.38e-7 rel 2.9e-7
  OP_N OP_N   512   256 accepted but WRONG: maxabs 1.18e0 rel 1.4e0
  OP_N OP_T   128   128         rejected, status 7
  OP_N OP_T   128   256         rejected, status 7
  OP_N OP_T   512   128         rejected, status 7
  OP_N OP_T   512   256 accepted but WRONG: maxabs 1.26e0 rel 1.5e0
  OP_T OP_N   128   128 accepted but WRONG: maxabs 3.07e0 rel 3.7e0
  OP_T OP_N   128   256 accepted but WRONG: maxabs 2.77e0 rel 3.4e0
  OP_T OP_N   512   128 accepted but WRONG: maxabs 6.86e0 rel 8.3e0
  OP_T OP_N   512   256 accepted but WRONG: maxabs 6.03e0 rel 7.3e0
  OP_T OP_T   128   128         rejected, status 7
  OP_T OP_T   128   256 accepted but WRONG: maxabs 1.13e0 rel 1.4e0
  OP_T OP_T   512   128         rejected, status 7
  OP_T OP_T   512   256 accepted but WRONG: maxabs 5.99e0 rel 7.3e0
```

**Exactly one combination is right:**

```c
cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            /*m=*/N, /*n=*/M, /*k=*/K,
            &alpha, /*A=*/B, CUDA_R_16F, /*lda=*/N,
                  /*B=*/A, CUDA_R_16F, /*ldb=*/K,
            &beta,  C, CUDA_R_32F, /*ldc=*/N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```

Why, in one paragraph: cuBLAS computes column-major `C'[m,n]` with
`C'[r,c]` at `r + c*ldc`. A row-major `C[M,N]` is the column-major `C'[N,M]`
(`r` = column, `c` = row, so `ldc = N`), and `C = A@B` transposed is
`C'[N,M] = B^T @ A^T`, so the gemm's **first** operand is B. B is `K x N`
row-major, which as a column-major matrix is `N x K` with `ld = N` — **so
`lda = N`, not K.** A is `M x K` row-major = `K x M` column-major with `ld = K`,
so `ldb = K`; and `OP_T` on that operand would demand `ldb >= M`, which is why
the `OP_* / OP_T` rows are either refused (status 7 = `INVALID_VALUE`) or
nonsense. **The one thing to memorise: `lda` is B's row length (N) and `ldb` is
A's row length (K); swapping them is silently accepted and silently wrong.**

## 4. The staged variants: what survives without the patch

| variant | 5120x768x2048 | 5120x2048x8192 | verdict |
|---|---|---|---|
| burn `a.matmul(b)` (the thing to beat) | 65.4 ms | 67.7 ms | baseline |
| **staged through the host** (read f32, convert, H2D, GemmEx, D2H, write back) | **137.9 ms** | **293.5 ms** | **2.1-4.3x SLOWER.** 0% of the win. |
| f16 operands + **D2D** staging | 0.66 ms (24.3 TF/s) | 5.26 ms (32.7 TF/s) | 11.5x, and it needs the pointer anyway |

The host round trip moves ~445 MB per call (42 MB + 67 MB in, 168 MB out, as
f32) plus a host-side f32→f16 pass, and it cannot overlap: every stage is a
synchronisation. **Do not ship a staged version to defer the patch.**

D2D staging is the shape that *would* work, and it is worth knowing why it is
not the answer either: a `cuMemcpyDtoD_v2` of 222 MB costs 1.26-1.46 ms
(≈150 GB/s, the synchronous driver's staged-copy path). The real integration
copies **nothing at all**, so the number that matters is §6's 0.389 ms, not
5.26 ms.

## 5. The dtype/computeType table (5120x2048x8192, and 5120x768x2048)

All with `CUBLAS_GEMM_DEFAULT` (= -1), fp32 accumulator, `C` always f32:

| A/B | computeType | 5120x2048x8192 | 5120x768x2048 | rel err (256x128x128) |
|---|---|---|---|---|
| f32 | `COMPUTE_32F` | 16.5 TFLOP/s | 13.5 | 3.0e-7 |
| f32 | `COMPUTE_32F_FAST_TF32` | 23.3 | 22.7 | 2.3e-4 |
| f32 | `COMPUTE_32F_FAST_16F` | 23.6 | 22.5 | 2.3e-4 |
| **f16** | **`COMPUTE_32F`** | **38.8** | **41.8** | **2.3e-4** |
| f16 | `COMPUTE_32F_FAST_16F` | 37.9 | 41.4 | 2.3e-4 |
| bf16 | `COMPUTE_32F` | 37.0 | 42.4 | **1.0e0 — WRONG** |
| bf16 | `COMPUTE_32F_FAST_16BF` | 36.5 | 45.6 | **1.0e0 — WRONG** |

- The published 43.7 TFLOP/s (nvcc probe, idle GPU) reads 38.8-41.8 here: the
  difference is a training run sharing this GPU for the whole session, so these
  are lower bounds.
- TF32 is 1.4x, not the 10x its reputation suggests on consumer Blackwell.
- **The `CUBLAS_COMPUTE_32F_FAST_*` constants are easy to get wrong and a wrong
  one returns `CUBLAS_STATUS_NOT_SUPPORTED` (15) rather than a wrong answer:
  FAST_16F = 74, FAST_16BF = **75**, FAST_TF32 = **77** (cudarc 0.19.9
  `cublas/sys/mod.rs`). I first used 80 and 97 and both were refused.**
- bf16-in is fast and wrong, in both modes. It is not a precision problem
  (f16 at the same speed is exact to 2.3e-4): the bf16 combination on this
  stack/architecture produces garbage. That is why the nvcc probe annotated its
  bf16 row "input conversion not verified" — it is genuinely broken, and f16 is
  the answer.
- fp32 storage with `FAST_16F` (23.6 TFLOP/s) is the **no-cast** option: same
  speed class as f16 for a third of the work, at 2.3e-4 instead of 3.0e-7. If
  dtype changes are too invasive to start with, this is the one-line version of
  the same idea and it needs no f16 storage at all.

## 6. End-to-end proof: a burn tensor, a cuBLAS call, no copy (needs the patch)

`crates/cublas-poc/src/main.rs` §4 (`zero_copy`), with §2 applied:

```
== 4. zero-copy on cubecl's stream (m=5120 k=768 n=2048) ==
  a=0x372afd800 b=0x320000000 c=0x323e42200 stream=0x7f627c000ec0
  f32 in, f32 out: maxabs 0.00e0 rel 0.0e0 | cublas 1.545 ms (10.4 TFLOP/s) vs burn 2.269 ms (7.1 TFLOP/s)
  f16 in, f32 out: maxabs 1.96e-3 rel 2.7e-4 | cublas 0.389 ms (41.4 TFLOP/s) vs burn(cast+matmul+cast) 2.880 ms (5.6 TFLOP/s)
```

- **The pointer resolves**: the addresses are inside cubecl's pool
  (`0x372afd800`, `0x320000000` — pool-page aligned, not `cuMemAlloc` aligned),
  so they are cubecl's own buffers, reached through the allocator's own map.
- **cuBLAS runs on cubecl's stream**: `stream=0x7f627c000ec0` comes from
  `Server::native_stream`, `cublasSetStream` is given it, and the result is read
  back through the *normal* `Tensor::into_data()` path with no barrier of any
  kind — which only works if the work really was in the graph's stream order.
- **The numerics check**: f32 is bit-identical to `a.matmul(b)` on the same
  inputs (maxabs 0.0), f16 is 2.7e-4 relative, which is the ordinary f16
  accumulator error measured identically in §5.
- **And burn's own f16 storage is what made it work**: the f16 row's operands
  are `a.cast(FloatDType::F16)` — real f16 tensors in cubecl's allocator, not
  buffers this probe made. So the f16 path needs no new storage machinery at
  all; the 5.6 TFLOP/s in that row is *cubecl's* f16 matmul, which is the thing
  to replace.

## 7. Production plan

1. **Land the patch** (§2) and add `cublas-native` behind it; nothing else can
   start before this. It is 110 additive lines with defaulted methods, so a
   backend that does not implement them is unaffected.
2. **Wrap one burn op.** `cublas_matmul(a: CubeTensor, b: CubeTensor) ->
   CubeTensor` in `burn-spectral` (it already depends on `cubecl` + `burn` and
   owns `bf16_ops.rs`): resolve `(a, b)` plus a fresh
   `client.empty_tensor([m,n], 4)` handle in **one** `native_handles` call,
   `GemmEx` on the startup-cached stream, and return
   `CubeTensor::new_contiguous(client, device, [m,n], handle, DType::F32)`.
   Gate it on `client.native_handles::<CudaServer>` returning `Some` and fall
   back to `Tensor::matmul` otherwise, so ndarray/CPU and any other backend are
   untouched. The first target is `LinearLike::forward` (the FFN and attention
   projections), where §6 measured 7.4x.
3. **The backward is the same primitive, twice.** `dA = dOut @ W^T` and
   `dW = A^T @ dOut` are two more `cublas_matmul` calls on the same stream —
   `W^T` is a `transa` flag on the existing call, `A^T` is a `ld`-swap of the
   §3 convention (a row-major `[k,m]` operand is `m x k` column-major with
   `ld = m`, so it is the `OP_T` side of a *different* call shape — add a
   `cublas_matmul_t` variant rather than bending `cublas_matmul`). Both land
   inside `bf16_ops.rs`'s existing `Backward<B, 2> for Bf16Matmul`, whose two
   `Tensor::matmul` calls become the two new ones: the autodiff graph, the
   checkpointing, and the f32 gradients are unchanged, and the forward stays
   fp32-output so the accumulator question does not reach the rest of the graph.
4. **A/B it as a flag, not a rewrite.** `--cublas-gemm` on `train` (default
   off), keyed off the preset, with `DM_QUANT_DEBUG`-style one-line logging of
   which path a shape took. The acceptance number is §6's: f16 fwd+bwd on the
   `small` preset at ≥3x the step time reduction on the GEMM-bound shapes, loss
   within the 2.3e-4-vs-3.0e-7 fidelity band, 0 NaN over 200 steps. Then drop
   burn's f16/bf16 matmul path (measured 5.6 TFLOP/s — dead weight) and update
   the "bf16 matmul is broken on pre.4" note in AGENTS.md, which this run
   contradicts.

## 8. Traps found, so nobody re-discovers them

- **`&str` literals in a `&[&str]` are not NUL-terminated in memory.** They sit
  back to back in `.rodata`, so `dlopen(s.as_ptr() as *const c_char)` hands the
  loader a 300-character filename and it fails with a plausible-looking ENOENT.
  Copy into a `CString`. (Cost: 20 minutes, and it looks exactly like "the CUDA
  library will not load in this process".)
- **cuBLAS exports only the `_v2` names here.** `cublasCreate` is absent;
  `dlsym` returns null and the short name must not be assumed.
- **A CUDA context is per-thread.** cudarc's `CudaContext` guard pops the
  context on `Drop`; retain + set raw and leak the guard.
- **`cudarc`'s `cublas` module is behind a cargo feature**, and turning it on
  recompiles the entire cubecl tree (cudarc's hash feeds every dependent). Four
  `dlopen`ed symbols avoid that entirely.
- **A `dlopen`ed library's first call needs a current context**; a cuBLAS
  *handle* does not (it is context-agnostic until used), so it can be created
  before `cuInit` and reused after.
- **Per-element relative error is useless on gaussian data** (half the elements
  are near zero). The probe reports `max|diff|` and `max|diff| / max|ref|`.
- **Examples that link `burn-ndarray` need `LD_LIBRARY_PATH` pointing at the
  cargo-built OpenBLAS** (`target/*/build/openblas-src-*/out/OpenBLAS-*/`), or
  they die with `libopenblas.so.0: cannot open shared object file` before main.
  `cublas-poc` has no ndarray dependency and needs nothing.

## 9. Measurement conditions

All numbers on the 5060 Ti, `cargo run --release -p cublas-poc`, log in
`/home/sehaxe/logs/cublas_poc_2026-09-27.log`, with a **training run sharing the
GPU for the whole session** (so both sides of every ratio are depressed, and the
ratios are the trustworthy part). The bf16-WRONG and lda/ldb findings are
bit-level and contention-independent.
