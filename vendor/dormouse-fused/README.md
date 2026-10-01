# dormouse-fused

Fused CUDA kernels and reference ports for [Burn](https://burn.dev) 0.22, one
workspace, one facade crate. MIT. Not affiliated with the official burn project.

**This directory is a vendored copy.** It is `exclude`d from the dormouse
workspace, it has no `.git` of its own and no `.github/` of its own, and 13 of
its crates are pulled into a training run by explicit `path =` dependencies
rather than by workspace membership. Consequences, all of them load-bearing:

- **the gate is `../../.github/workflows/fused-library.yml`** (root of dormouse),
  which runs cargo from *inside* this directory. There is no workflow here.
  `cargo test -p dormouse-gdn2` from the repo root fails with "package not found" —
  that is the `exclude` working, not a broken build.
- **there is deliberately no `cuda-tests` job.** `cuda-compiles` only does
  `--no-run`, which cannot fail on a CUDA-only bug. The GPU check is a command
  a contributor runs on the GPU box: [`tools/gpu-gate.sh`](tools/gpu-gate.sh).
- **the benchmark baselines are not re-checked by any CI job in this copy.**
  `bench/baselines.json` holds 10 entries, one of them (`msa_sparse`) for a crate
  deleted 2026-09-27 (ADR-0014). It is a record of a past measurement, not a
  regression gate. The README of an earlier revision claimed otherwise.

## Layout

```
crates/           20 crates: fused kernels + ops-only technologies
dormouse-fused/       the facade: re-exports + feature flags, GENERATED
bench/            committed baselines + two dated comparison writeups
benches/          perf harness (a GPU-runner tool, not a CI gate)
tools/            gen_facade.py, gpu-gate.sh, test-feature-matrix.sh
```

## The 20 crates

`cuda` = ships a fused kernel. `autodiff` = ships a fused backward. **In build**
= `dormouse-{core,train}` or `backend-parity` has a `path =` dependency on it.

| crate | fused | in build | what it is |
|---|:--:|:--:|---|
| `dormouse-attnres` | fwd+bwd | yes | Attention Residuals — depth-wise attention over layer outputs (Moonshot/Kimi) |
| `dormouse-bitnet` | fwd+bwd | yes | ternary, 8/4-bit absmax/absmean with a Fast Walsh-Hadamard rotation |
| `dormouse-dspark` | — | yes | speculative decoding (DeepSeek AI, arXiv 2607.05147) |
| `dormouse-eggroll` | — | no | EGGROLL — low-rank evolutionary strategies (arXiv:2511.16652) |
| `dormouse-engram` | — | yes | conditional memory — n-gram hash embeddings, multi-head gated fusion |
| `dormouse-es` | — | no | Evolution Strategies |
| `dormouse-gdn2` | fwd+bwd | yes | Gated DeltaNet 2 — channel-wise erase/write gates |
| `dormouse-jepa` | — | yes | data2vec 2.0-style EMA teacher + masked latent prediction |
| `dormouse-kda` | fwd+bwd | yes | Kimi Delta Attention — data-dependent write strength, channel-wise decay |
| `dormouse-mhc` | fwd+bwd | yes | Manifold-Constrained Hyper-Connections (DeepSeek) |
| `dormouse-mor` | — | yes | Mixture-of-Recursions routing (arXiv:2507.10524) |
| `dormouse-muon-plus` | fwd | yes | NS polar orthogonalization + one post-polar row/col normalization |
| `dormouse-parcae` | — | no | stable looping via spectral retention |
| `dormouse-ptrn` | — | no | Probabilistic Tiny Recursive Model — test-time scaling |
| `dormouse-rmsnorm` | fwd | yes | RMS normalization |
| `dormouse-rope` | fwd+bwd | no | RoPE with YaRN extrapolation |
| `dormouse-sct` | fwd+bwd | no | permanent truncated SVD with Stiefel QR retraction |
| `dormouse-situ` | fwd+bwd | yes | SiTU-GLU (Kimi K3) |
| `dormouse-spectral` | fwd | yes | ternary SVD weights with rank-1 ternary MoE routing |
| `dormouse-swiglu` | fwd | no | SiLU-gated linear unit |

There were 28 crates until 2026-09-28; eight unreachable ones were deleted. The
per-crate fate table — what each was for, how reachability was measured, why
each deletion landed — is `docs/architecture/library-crate-fate.md` in the dormouse repository.

## Use it

```toml
[dependencies]
burn = { version = "0.22.0-pre.4", default-features = false, features = ["std", "cuda"] }
dormouse-fused = { git = "https://github.com/sehaxe/burn-fused", features = ["cuda"] }
```

Every member is re-exported, so `dormouse-fused` is the only `burn-*` name you need:

```rust
use dormouse_fused::dormouse_kda::KdaModule;   // the crate, as a module
```

**Read [`dormouse-fused/INTEGRATION.md`](dormouse-fused/INTEGRATION.md) before you wire
it up.** It is the facade's crates.io readme (`readme =` in its `Cargo.toml`), so
it is the version a user sees on crates.io, and it is where the version rule, the
feature table, the `fusion` footgun that silently turns every fused path off, the
per-mechanism f32-only precision table and the known-broken decode arm live.

## Work on it

```bash
# one crate, CPU
cargo test -p dormouse-gdn2

# one crate, on the GPU — the check that is not a CI job
tools/gpu-gate.sh

# the generated facade files must match the member manifests
tools/gen_facade.py --check

# every feature combination a user can type
tools/test-feature-matrix.sh
```

All four run from inside this directory. Adding a crate: new crate under
`crates/`, add it to the workspace `members`, run `tools/gen_facade.py` and
commit the result, add a case to `benches/src/main.rs` if it has a fused path.
See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## What "verified" means here

Not every claim in this workspace is a measurement, and the difference is
recorded rather than asserted. Four documents, in increasing order of how much
they can be trusted:

| document | what it is |
|---|---|
| [`dormouse-fused/INTEGRATION.md`](dormouse-fused/INTEGRATION.md) | how to depend on this and what breaks. Current. |
| [`TEST-AUDIT.md`](TEST-AUDIT.md) | what the tests actually assert, per crate. Dated 2026-09-27, before the deletions. |
| `docs/protocols/ORACLE.md` + `docs/protocols/ORACLE-TIERS.tsv` (dormouse repo) | per-comparison tier for `dormouse-kda`/`dormouse-gdn2`: **(a)** compared against the authors' own code, **(b)** a transcription, **(c)** a transcription of a transcription, **(d)** nothing external exists, **(x)** not a correctness test. Currently 47 (a), 22 (b), 1 (c), 27 (d), 30 (x). `tools/oracle_gate.py` reads the tsv. |
| `bench/RESEARCH_VERIFICATION.md` | dated 2026-08-09, and it has rows for crates that no longer exist. |

Several benches read the clock with no device flush inside the timed loop, so
they time CPU enqueue rather than the kernel; those are retracted in place and
named — `crates/dormouse-mhc/README.md` and `crates/dormouse-gdn2/README.md` carry the
list. Quoting a speedup from this workspace means checking that its bench
flushes.

## Conventions

- Burn 0.22.0-pre.4, Rust stable (workspace MSRV 1.85; leaf crates may raise it
  locally — dormouse-gdn2 requires 1.95), edition 2021, MIT.
- Fused dispatch: `try_into_primitive` + downcast to the bare
  `CubeBackend<CudaRuntime>`; grads via burn-autodiff `Ops`/`Backward`/`Checkpointer`.
  Fused kernels hard-require f32 buffers and fall back to the tensor path for
  any other dtype — but only some of them check, which is why the precision table
  is a table.
- Feature matrix is uniform across kernel crates: `std`, `cuda`, `autodiff`
  (ops-only crates ship `std`). The facade's flag list is GENERATED from these
  manifests by `tools/gen_facade.py`; `gen_facade.py --check` fails if it drifts.
