# Integrating burn-fused

20 fused-kernel and ops crates for [Burn](https://burn.dev) 0.22, one
dependency, one `burn` version, three feature flags. Not affiliated with the
official burn project; MIT.

This is the whole integration story. If something here is wrong for your setup,
the bug is in this file — open an issue rather than working around it.

## 1. Add it

```toml
[dependencies]
burn = { version = "0.22.0-pre.4", default-features = false, features = ["std", "cuda"] }
burn-fused = { git = "https://github.com/sehaxe/burn-fused", features = ["cuda"] }
```

Vendored/local checkout:

```toml
burn-fused = { path = "../burn-fused", features = ["cuda"] }
```

Nothing else. Every member crate is re-exported, so `burn-fused` is the only
`burn-*` name you need in your manifest:

```rust
use burn_fused::burn_kda::KdaModule;   // the crate, as a module
use burn_fused::burn_rope::apply_rope_4d;
```

### The version rule (read this one)

Every crate in the workspace pins the same `burn` **pre-release**,
`0.22.0-pre.4`. That is a single choice made in one place — the workspace
manifest — instead of 20 manifests you would each have to keep in step.

It also means your own `burn` must resolve to a version compatible with ours.
Cargo unifies semver-compatible versions, so `burn = "0.22"` is fine when
`0.22` is out; if your project is pinned to a *different* `0.22.0-pre.*`,
cargo keeps two copies of `burn-tensor` in the graph and every `Tensor` type
from us is a different type from yours. The symptom is a wall of
`expected Tensor<...>, found Tensor<...>` in your own code, with no mention of
versions. Check `cargo tree -d | grep burn` first if you see that.

## 2. Features

| feature | default | what it turns on |
|---------|---------|------------------|
| `std` | **yes** | the runtime features every member has. No fused kernels, no autodiff. |
| `cuda` | no | the fused kernels of the 13 members that have them, and the `burn_cuda` re-export. |
| `autodiff` | no | the fused *backward* ops of the 8 members that have them, and the `burn_autodiff` re-export. |
| `serde` | no | `burn-sct` serialization. |
| `training` | no | `burn-dspark`'s training-mode code (the draft head). |

`cuda` and `autodiff` are independent on purpose: inference wants the fused
forward without pulling the autodiff graph into your binary, training wants
both. `tools/test-feature-matrix.sh` builds and tests every combination.

The flag list is **generated** from the members' own manifests
(`tools/gen_facade.py`), so it cannot drift: today 13 members declare `cuda`
and all 13 are listed. The hand-maintained list we had before silently missed
`burn-mor`, and nothing noticed because nothing consumed the facade.

The re-exports are unconditional, too: a member is always reachable as
`burn_fused::burn_<name>`, whatever the flags. The flags control each member's
*runtime* features (which kernels get compiled), not whether you can name the
crate. That is deliberate — a re-export behind a `cfg` can rot without anyone
noticing, and `burn-fused/tests/facade.rs` resolves all 20 of them on every CI
run.

## 3. Which backend types work

| backend | mechanism path | fused kernels |
|---------|----------------|---------------|
| `burn::backend::NdArray` (CPU) | tensor ops | never (no GPU) |
| `burn::backend::Wgpu`, `Rocm`, `Cpu` (cubecl) | tensor ops | never — the kernels in this workspace are CUDA-only |
| `burn_fused::burn_cuda::Cuda` | tensor ops + fused | **yes** |
| `Autodiff<Cuda>` | tensor ops + fused fwd/bwd (8 members) | yes, where implemented |

### The fusion footgun (silent, and it costs you all the performance)

The fused kernels dispatch on an exact type match against the **bare**
cubecl backend, `burn_cubecl::CubeBackend`. In burn 0.22, the public name is
a conditional alias:

```text
burn_cuda::Cuda == burn_cubecl::Cube
  == CubeBackend                        when burn-cubecl/fusion is OFF
  == burn_fusion::Fusion<CubeBackend>   when it is ON
```

`burn-cuda`'s **default features include `fusion`**. So if your own manifest
says `burn-cuda = "0.22"` (defaults on) while ours says
`default-features = false` (as the facade does), cargo unions the two and
turns `fusion` on for the whole graph. Your backend type becomes
`Fusion<CubeBackend>`, the type match fails, and **every mechanism in this
workspace quietly runs its plain tensor path** — correct results, none of the
speed, no error anywhere.

Two ways to stay on the fast path:

1. use the backend the facade re-exports: `type B = burn_fused::burn_cuda::Cuda;`
   (its `burn-cuda` dependency has `default-features = false`, so it is the
   bare `CubeBackend`), and do not depend on `burn-cuda` yourself; or
2. if you must, write `burn-cuda = { version = "0.22.0-pre.4", default-features = false, features = ["std", "cuda"] }`.

How to tell which path ran: `GDN2_ALLOC_TRACE=1` makes `burn-gdn2` print
`[gdn2] fused chunk kernels ENGAGED: ...` whenever the chunk kernels run; a
silent tensor fallback prints nothing.

## 4. Hello, gated delta net

Three lines, and the whole point of the facade: one dependency, one import,
one version.

```rust
use burn::backend::Autodiff;
use burn::tensor::{Distribution, Tensor};
use burn_fused::burn_kda::KdaModule;

type B = Autodiff<burn_fused::burn_cuda::Cuda>;

let device = Default::default();
let layer = KdaModule::new(&Default::default(), 0.0, &device);  // 128-d, 4 heads
let x = Tensor::random([1, 16, 128], Distribution::Default, &device);
let y = layer.forward_train::<B>(x);
assert_eq!(y.dims(), [1, 16, 128]);
```

Drop `Autodiff<...>` and use `burn_fused::burn_cuda::Cuda` directly for
inference. The same code with `type B = NdArray` is the doctest in
`src/lib.rs` and the `hello_gated_delta_net` test in `tests/facade.rs`, so it
is compiled and run on CPU by CI and cannot rot out of the docs.

Entry points on a mixer, and which one to use when:

| call | use for |
|------|---------|
| `forward_train::<B>(x)` | a whole-sequence pass. Training, and the fused chunked kernel lives here. |
| `forward::<B>(x, &mut state, true)` | autoregressive decoding, carrying the state. Also fused. |
| `forward_recurrent(x, &mut state, true)` | exact per-token reference; the test oracle for the two above. |
| `forward::<B>(x, &mut None, false)` | **broken today — see §6.** |

## 5. Precision, per mechanism

**The rule: every fused kernel in this workspace is compiled for `f32` only.**
They all launch `launch_unchecked::<f32>` with a hand-computed byte length. If
you hand one a `f16`/`bf16` buffer it reads twice the bytes it should: garbage
output, or `CUDA_ERROR_ILLEGAL_ADDRESS`, depending on the kernel.

Only some members check. This is the table, and it is the reason the rule is
stated here rather than left to be discovered:

| member | fused | non-f32 input |
|--------|-------|---------------|
| `burn-gdn2`, and `burn-kda` through it | chunk fwd + adjoint bwd | **checks** `DType::F32` and falls back to the tensor path (`burn-gdn2/src/kernel/chunk_cube.rs:853`). Safe. |
| `burn-rmsnorm` | RMSNorm | **checks** both operands, falls back (`burn-rmsnorm/src/fused.rs:95`). Safe. |
| `burn-swiglu` | SwiGLU gate | **checks**, falls back (`burn-swiglu/src/fused.rs:56`). Safe. |
| `burn-spectral` | TSCT factors | **casts** f32 for the kernel and back, so bf16 works. |
| `burn-rope` | RoPE fwd + bwd | no check. Pass f32. |
| `burn-mhc` | Sinkhorn | no check. Pass f32. |
| `burn-situ` | SiTU-GLU fwd + bwd | no check, and `hidden % 8 == 0` (or `hidden == 4`) is required or it falls back. Pass f32. |
| `burn-attnres` | depth-attend fwd + bwd | no check. Pass f32. |
| `burn-muon-plus` | Newton–Schulz + momentum | operates on f32 parameters (masters are always f32). |
| `burn-sct` | QR / retraction | no check. Pass f32. |
| `burn-mor` | none of its own (`cuda` only enables burn's CUDA backend) | — |
| the other 17 | no fused kernels | dtype-agnostic; plain tensor ops. |

In short: **f32 in, f32 out, on any member with a fused kernel.** If you train
in bf16, either keep the fused members' inputs in f32 (they are small
projections, and a cast is cheap) or accept the tensor path. The
`burn-gdn2`/`burn-kda`/`burn-rmsnorm`/`burn-swiglu` members degrade on their
own; the rest are on you. `tests/facade.rs` checks the fallback is finite on
CPU, so the "safe" column is tested, not asserted.

Two more f32-only limits worth knowing, both documented in the code: the
chunked WY kernel refuses `chunk_size > 16` and head dims `> 256` (the
`K/exp(cumsum(g))` factor underflows f32), and it falls back to the K3 16-tile
tensor path there.

## 6. Known-broken in this workspace today

Both are in member crates, both are layout bugs in the *read-only prefill*
arm (`update_state == false` with no incoming state), and both panic for any
sequence length that differs from the head count:

- `burn-gdn2/src/module.rs:368` — the `update_state == false` branch already
  permutes to `[B, T, H, D]` at line 365 and line 368 permutes again, so
  `rms_norm_gate_per_head` gets `[B, H, T, D]` against a `[B, T, H, D]` gate.
- `burn-kda/src/lib.rs:708` — the same arm computes `q.matmul(state)` and
  permutes; it also ignores `k`, `v` and the gates, so it is not the
  recurrence at all.

Workaround: use `forward_train::<B>(x)` for a whole-sequence pass (that is
what the example in §4 does), or hand `forward` a state. Decode with
`update_state = true` and a carried state — that path is correct and fused.

## 7. What is where

```
crates/<name>/          one mechanism per crate, its paper citation, its tests
burn-fused/             this facade: generated re-exports + generated features
tools/gen_facade.py     regenerates both from the workspace member list
tools/test-feature-matrix.sh   the feature matrix, CPU only
```

Adding a crate: create it under `crates/`, add it to the workspace members,
run `tools/gen_facade.py`, commit all three. CI runs
`gen_facade.py --check` and the matrix, so a crate that is added but not
generated fails the build rather than being invisible.
