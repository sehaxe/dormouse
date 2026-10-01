# Precision on this backend: why bf16 is slower than fp32

> Updated 2026-10-01. Sources: `AGENTS.md` §2.1 (precision machine facts),
> §2.2 (GEMM ceilings, mixed-dtype rules), §2.3 (the `max_ortho` latch),
> §3.7 (`--quant`/`--act-quant`), §3.8, ADR-0016 (the three precision bugs and
> their retractions), `docs/architecture/bf16-plan.md` (the Russian plan —
> linked, not translated), `docs/guides/dmexp.md` (storage vs compute), and
> the vendored patch itself, `vendor/cubecl-fix/cubecl-ir/src/types/scalar.rs`.
> Every number names its source; where a document and the code disagree, both
> are shown with dates.

On this box, `--bf16` makes training **slower** than fp32. That is not a bug
in burn, not a bug in our code, and not fixable by trying harder — it is a
property of the CUDA backend this repo is built on. This page explains what
the backend actually is, what each precision option really does, where
precision saves and where it breaks, and the cautionary tale of `fp4`.

## The backend, in two sentences

The CUDA backend here is the **pliron → LLVM → NVPTX** one, not NVRTC: the
`cpp` feature is empty and off by default, and nothing in our graph enables it
(`vendor/cubecl-fix/cubecl-cuda/src/compiler.rs:16-32`). Then
`restrict_to_llvm_backend`
(`vendor/cubecl-fix/cubecl-cuda/src/runtime.rs:420-513`) advertises only what
that dialect can lower, with the reason in comments. So **"the CUDA backend
cannot do X" is almost always "the LLVM dialect has no type or rule for X"** —
and the file that says so is that one function (`AGENTS.md` §2, framing).

## bf16: no workaround, by construction

The LLVM dialect has **no bf16 type**. `restrict_to_llvm_backend` deletes it
from the advertised element types on purpose and strips the bf16 tensor-core
families — and even reading a bf16 buffer back as f32 dies at kernel-compile
time with `Type cube.bf16 does not have a conversion to LLVM type implemented`
(ADR-0016 bug 2, `AGENTS.md` §2.1). That is why burn-spectral's own bf16 test
fails where it does (`burn-cubecl ops/tensor.rs:150`).

Consequences, in order of how often they bite:

- **`--bf16` stores bf16 and computes fp32 through cast copies** — there is no
  tensor-core bf16 to compute with, so every bf16 run on this box is slower
  than fp32 (`AGENTS.md` §2.1).
- **Do not paper over it** with a silent fp32 fallback dressed as bf16 — that
  is precisely the ADR-0019 silent-fallback failure mode: a run that is
  correct and a year slower, with nothing in the log to say so.
- **The one bf16 primitive that works is storage as `u16` bit patterns** —
  integer ops plus bitcast, pinned against f64 in
  `vendor/dormouse-fused/crates/burn-gdn2/tests/lowp_bf16_cuda.rs`. A weight
  file does not need a matmul, which is what makes the `.dmexp` bf16 export
  safe to ship even though bf16 *compute* is not
  ([dmexp.md](dmexp.md); the narrowing/widening is integer bit manipulation on
  the host, `crates/dormouse-train/src/export.rs:85` and `:93`).

## f16: from silently slow to patched

f16 matmul is **numerically correct** — the answer matches fp32 to 1e-2 — but
it was silently slow: the f16 **tensor-core** candidate died at compile time
with `Expected type builtin.fp16 to implement dyn SizedType`, and the
autotuner fell back to a non-accelerated routine **without a word** (ADR-0016
bug 3). The cause was one interface short: pliron's `builtin.fp16` — the type
`cube.f16` lowers to — implemented `FloatTypeInterface` and not the `SizedType`
that shared-memory sizing queries
(`vendor/dormouse-fused/crates/cubecl-opt/src/lib.rs:36-40`).

**Status correction, code over document:** `AGENTS.md` §2.1 still says the fix
"is ~10 lines in cubecl-ir and belongs upstream; until it lands the 43.7
TFLOP/s cuBLAS f16 number is not reachable through burn". The fix has since
**landed in the vendored fork**: root `Cargo.toml` `[patch.crates-io]` maps
`cubecl-ir` ("pliron's f16 size interfaces, ADR-0016 bug 3"), and
`impl SizedType for FP16Type` — `size` 2, `align` 2, deliberately equal to
what `cube.f16` reports — is at
`vendor/cubecl-fix/cubecl-ir/src/types/scalar.rs:240-251`, with a two-assertion
gate test (`fp16_llvm_type_answers_the_size_query_like_the_cube_type`, same
file) pinning that the LLVM-level type answers the size query exactly like the
cube type. The f64 and bf16 equivalents are deliberately **not** touched: bf16
has no register form in the dialect at all, so a size for it would answer a
question nothing is asking (`Cargo.toml:115-117` comment). §2.1's sentence is
stale relative to §2.5 and the code; the patch table is the authority
(`AGENTS.md` §2.5).

## What the hardware could do — if the plumbing existed

Measured on this GPU, 2026-09-27 (`AGENTS.md` §2.2):

| path | TFLOP/s |
|---|---|
| cuBLAS f16-in / fp32-accumulate | **43.7** |
| cuBLAS fp32 | 13.2 |
| cubecl f32 (ours) | 3.5–7.6 |
| TF32 | 17.5 — pointless on consumer Blackwell |
| hand-rolled WMMA | 8.4 |

The gap between cuBLAS f16 (43.7) and our cubecl f32 (3.5–7.6) is the whole
precision opportunity on this card, and the two working routes to it are
**route big matmuls through cuBLAS; do not write kernels** — the integration
crux being that a burn tensor's buffer is a cubecl `Handle`, not a raw
pointer, and `cubecl-runtime` 0.11.0-pre.4 has no pointer-resolution API
(`AGENTS.md` §2.2; `crates/cublas-poc` is the direct-cuBLAS demonstration).

## fp8: the sm_120 default, and the latch that guards it

`--quant fp32|bf16|fp16|fp8|fp4` forces the TSCT **factor** format. Default
auto: bf16 mode → `Bf16`, else sm ≥ 120 → `Fp8`, else `Fp32` — the forward
path only, fp32 masters, so checkpoints are format-agnostic and `--quant fp32`
after a fallback run reproduces the old behaviour exactly (`AGENTS.md` §3.7).

The guard is `max_ortho`, monitored every 500 steps — and the metric itself
has a history worth knowing, because it once silently disabled the entire
fp8 forward path:

- The raw Frobenius `‖UᵀU−I‖_F` scales ~k and sits **above** the threshold
  from step 0; the NS-3 retraction's own convergence floor is ~4e-3 raw
  (~6e-5 per-entry at r=64). The old unnormalized check therefore fired the
  fp32 fallback **on every fresh run**, and no pre-2026-09-04 run ever used
  the factor-quant forward (`AGENTS.md` §2.3; the pre-fix story is retracted
  in §3.2).
- The fixed metric is **per-entry** (Frobenius ÷ rank). Fresh init reads
  ~1.4e-4; the threshold is **1e-3 per-entry** — genuine drift, not noise.
- Above it, **all** factors irreversibly switch to fp32, the checks stop, and
  the latch is persisted in the checkpoint — one-way by design
  (`AGENTS.md` §2.3).

Where the orthogonalization itself is expensive: fp32 Newton-Schulz on a
`[768,768]` projection costs ~40 s/step, which is why Muon+ stays off the
`[d,d]` matrices — the low-rank factors are orthogonalized in their factored
`[d,64]`/`[64,f]` form, ~1000× cheaper (`AGENTS.md` §2.3).

## The rules every forward follows (where precision breaks)

- **Mixed-dtype ops NaN on this stack** (bf16 activations × fp32 weights, bf16
  + f32 adds). The rule: **cast to fp32 before every Linear, compute in fp32,
  cast the residual writes back to the activation dtype** — verified 60
  steps, 0 NaN, loss 5.53 → 2.21 (2026-08-29) (`AGENTS.md` §2.2).
- **Logits stay fp32.** The final norm + `lm_head` run in fp32 even in BF16
  mode (bf16 logits NaN), and the loop-internal per-iteration CE logits
  likewise (`AGENTS.md` §2.2).
- **The Engram kernels are f32-only** (ILLEGAL_ADDRESS on bf16, measured
  2026-08-29), and so is the KDA fused chunked kernel — it falls back to
  tensor ops, so both need their inputs cast under `--bf16`
  (`AGENTS.md` §2.2).
- **KDA state dtype follows the inputs** — bf16 under `--bf16`, fp32
  otherwise; Moonshot's FlashKDA trains bf16 state (`AGENTS.md` §3.5, item 5).
- **bool→float is fine; the rule built on it is not negotiable.** The old
  claim that `bool_tensor.float()` returns 0.0 for `true` on cuda was
  **retracted** (ADR-0016, 2026-09-27 — the miscount was `clone()` sharing a
  device buffer while `mask_fill` wrote in place). The rule survives its
  retraction: never build a numeric indicator from a bool tensor on device;
  count on the host (`AGENTS.md` §2.1, §1.3). When a new op misbehaves on one
  backend only, the gate is
  `cargo test -p backend-parity --features cuda --test backend_parity` —
  it turns "the cast is broken" into a command.

## act-quant `fp4`: a cautionary tale in two retractions

The claim on record was *"fp4 + group 128, 100 steps, 0 NaN, convergence ==
fp32"*. It was retracted twice, for two independent reasons (`AGENTS.md`
§3.2, §3.7):

1. **It never ran 4-bit attention.** `ActFormat::attn()` upgrades the
   attention path unconditionally — `Fp4 → Int(8)`, `Int(b) → Int(b.max(8))`
   — so `--act-quant fp4` verified the FFN path only.
2. **It was not e2m1 at all.** Real e2m1 has no 0.75 — the grid is
   `0, .5, 1, 1.5, 2, 3, 4, 6` (8 magnitudes, 16 codes) — but the mantissa
   rule emitted 0.75 for `a ∈ [0.625, 1)`, and the caller scaled each block's
   max onto **1** instead of onto the format's max (6). Only
   `{0, 0.5, 0.75, 1}` of the eight magnitudes were ever reachable: a
   ~3-level quantizer wearing a 4-bit label, with 1.5…6 as dead code
   (fix `9b343d3`).

Post-fix: the grid is the real `E2M1` with `ActFormat::max_value = 6`, and
**every `--act-quant fp4` number before 2026-09-28 is invalidated**. The `4`
and `8` paths were untouched throughout and are pinned bit-identical
(`int4_levels_are_the_whole_symmetric_range`). The general lesson is the same
one §1.4 draws for "verified": a precision label is a claim about a
*measured* format, and a quantizer that computes the right-looking answer at
the wrong resolution is the numerical cousin of a silent fallback.

## Where precision saves

- **VRAM.** The budget rule of thumb: 1B model bf16 ≈ 2 GB weights, +2 GB per
  batch-16 s1024 step (`AGENTS.md` §3.8, from `docs/architecture/bf16-plan.md`
  §6). fp32 masters + an fp8 forward path buy the FLOP savings without
  touching checkpoint portability (above).
- **Files.** The `.dmexp` measurement (20-step `small`, 9 197 390 params):
  fp32 = 36 795 131 B (2.8× the 104 MB training container), bf16/f16 =
  ~18.4 MB (5.7×). f16 is the *more accurate* format (10 mantissa bits vs 7;
  2.8× lower mean logit delta) but had to **flush 8 weights to zero** — the
  model's smallest non-zero weight is 1.1e-9, four orders below f16's smallest
  normal (6.1e-5) — while bf16 reproduced 1.1059e-9 exactly, because bf16's
  exponent is f32's. bf16 ships as the export default because its failure
  mode (an f16 weight above 65504 → inf) stops being data-dependent; the
  decision procedure for f16 is printed by `export info`
  ([dmexp.md](dmexp.md), ADR-0023). **This is storage, not compute** — the
  file loads back as an fp32 model.
- **The optimizer, carefully.** The bf16 plan targets Muon moments bf16 +
  stochastic rounding and MuonQ 4-bit — both **blocked by the bf16-compute
  fact above** (`AGENTS.md` §3.5, item 3). The plan itself,
  [`docs/architecture/bf16-plan.md`](../architecture/bf16-plan.md) (Russian,
  cited as written), is the all-bf16 target state: everything stored and
  computed in bf16 with f32 accumulation and no fp32 masters — reachable only
  when the backend grows a bf16 type or the matmuls route around burn.

## The one-page decision table

| you want | do this | because |
|---|---|---|
| fastest training today | fp32 (the default), batch as large as fits | bf16 computes fp32 through cast copies; GEMMs are a small share of a launch-bound step (`AGENTS.md` §3.1) |
| smaller checkpoints / download | `export --dtype bf16` | storage is bit-manipulation; exponent range fits the weights ([dmexp.md](dmexp.md)) |
| cheaper TSCT forward | nothing — auto already picks Fp8 on sm_120 | §3.7; the `max_ortho` latch guards it (§2.3) |
| cheap FFN activations | `--act-quant 8` (or `4`, post-fix) | the pinned paths (`AGENTS.md` §3.7) |
| 4-bit attention | it does not exist | `ActFormat::attn()` floors attention at int8 (§3.7) |
| bf16 compute | not on this backend | the LLVM dialect has no bf16 type (§2.1) |
| f16 tensor cores | available again via the vendored `cubecl-ir` patch — but burn-side matmul wins are small on a launch-bound workload | `scalar.rs:240`; GEMMs are 2–5 % of a step (§2.2) |
