# ADR-0020 — Oracle discipline: what "verified" is allowed to mean

Date: 2026-09-27. Status: accepted. Amends ADR-0018 rule 1.

## The rule, in one sentence

**A crate may only use the word "verified" — and may only use "bit-for-bit" at all —
if it names the external file the reference came from; where the reference is our own
transcription, the crate says "transcription", and where there is no reference it says
"no external reference exists".**

ADR-0018 rule 1 said the port is "verified bit-for-bit (or to a stated tolerance)
against the ORIGINAL implementation wherever the authors shipped code". Read as a
statement about the library, that is false today. Measured on `vendor/dormouse-fused/`
at `23f2b1c` (evidence: `docs/research/2026-09-27-oracle-audit.md`):

> **0 of 28 crates verify any numeric output against an authors' own source code.
> 0 of the 3 crates that use the phrase "bit-for-bit" have an oracle from the
> authors.** Ten crates have an available external oracle (the `REFERENCE` list in
> `docs/research/2026-09-27-adopt-vs-port.md`); **none of the ten has wired a test to it.**

The rule is not being weakened to match reality. Reality is being made to match the
rule, and until then the words in a README are wrong and are the words that get quoted.

## The taxonomy — four kinds of oracle, named

Every reference comparison in the library is one of these. Every crate's doc comment
and README must place itself in one, explicitly.

| | kind | what the comparison is worth | how it is labelled |
|---|---|---|---|
| **(a)** | **AUTHORS** | the reference is bytes, or a direct run, of the mechanism's authors' own shipped code | "verified against `<org/repo>` `<path>`" |
| **(b)** | **TRANSCRIPTION** | the reference is our hand transcription of a paper or a spec | "matches our transcription of Eq. N of <arXiv>" |
| **(c)** | **TRANSCRIPTION-OF-TRANSCRIPTION** | the reference is derived from a transcription, or a transcription of a transcription, of the authors' code | "matches `tests/gen_reference.py`, a transcription of `<org/repo>` `<path>`" |
| **(d)** | **NOTHING** | no reference comparison exists: shape, finiteness, invariant, or self-consistency only | "no external reference; invariants only" |

A transcription error is **symmetric**: we get the same wrong answer from both sides of
the comparison and the test goes green. That is the whole argument for preferring (a),
and it is why (b) and (c) may never be described with (a)'s vocabulary.

Two additions the audit forced:

- **A self-consistency check is kind (d), not (b).** "the chunked path matches the
  recurrent path" compares two of *our* formulations of the same recurrence. It is a
  real and useful test. It is not verification of the port. `burn-kda`'s entire test
  suite is this.
- **An invariant check is kind (d), not (b).** "row and column sums are 1" is a
  property of Sinkhorn-Knopp, not a comparison with DeepSeek's `ref_mhc.py`.
  `burn-mhc` is the library's best property check and has no external oracle at all.

**The one genuine exception, stated so it is not abused:** where the authors' artifact
*is* the pseudocode and nothing else exists, (b) is not a weaker oracle. Per
`docs/research/2026-09-27-adopt-vs-port.md` §2, Muon+ (2602.21545) ships its update rule as
a 17-line Algorithm 1 and the repo adds only a `Keller` polar-method variant. For
`burn-muon-plus` a transcription of the pseudocode *is* the reference. Nothing else in
the library may claim this.

## The test must be falsifiable — checklist

A test that compares against a reference passes only if all of these are true. A test
that fails any of them is kind (d) and must be relabelled, not counted.

1. **The reference is not the code under test.** The expected value must not be a
   transcription of the same expression the implementation evaluates. A test that
   restates `(x / g).round().clamp(-1, 1) * g` next to a function that computes
   `(x / g).round().clamp(-1, 1) * g` cannot fail, and moves with the bug when the
   bug moves.
2. **The branch under test is actually taken.** If the assertion is about a fused CUDA
   kernel, the test's device must be one the fused path dispatches on. A test in a
   `#[cfg(feature = "cuda")]` module whose `dev()` is `Device::ndarray()` asserts
   nothing about CUDA. See ADR-0012 / the `TypeId` gate.
3. **The tolerance is tight against a named wrong answer.** A test must be able to
   name the specific wrongness it catches and roughly how far that wrongness is from
   the expected value. Tolerance without that number is decoration. (The good pattern
   is in `burn-rope/src/lib.rs:112-165`: it states the inverted-ramp error is ~2.6 rad,
   compares in the **cos** domain because an f32 `cos` collapses to 1.0 below ~3e-4
   rad, and uses 1e-5.)
4. **It is a domain where the wrong answer is far from the tolerance.** In the cos
   domain that fails, and in any saturating nonlinearity, a wide tolerance is
   indistinguishable from no test. `burn-sct`'s advertised `from_dense` tolerance of
   1e-3 against a 5.9e-4 observed error is the shape of this failure.
5. **The comparison runs in CI, on the backend the bug needs.** A test behind a
   non-default feature, an `#[ignore]`, a gitignored fixture, a missing generator, or a
   silent `if env::var("BURN_DEVICE") != "cuda" { return; }` is not a gate. It must be
   in a job that a runner executes, and that runner must be the one with the GPU.
6. **The reference is regenerable and the fixture is committed.** A gitignored fixture
   is a claim. `tests/ref_data.bin` is committed and CI diffs it against its generator
   (`fused-library.yml`, `ref-data-reproducible`); that is the standard.
7. **The backend under test is the backend we ship on.** `Autodiff<Cuda,
   BalancedCheckpointing>` is what dormouse trains on. A CUDA test on `CudaBare`
   cannot catch a balanced-checkpointing bug.

## The mechanical gate

> A crate may only use the phrase **"bit-for-bit"** in its README or doc comment if it
> names the external source its reference came from — `<org>/<repo>`, `<path>`, and the
> commit or the fixture that carries it. Absent that, the strongest permitted phrasing
> is **"matches our transcription of <arXiv> to <tolerance>"**.

Mechanical form, so it can be grepped and does not depend on anyone's memory:

- A CI step greps every `crates/*/README.md` and every `src/lib.rs` for
  `bit.?for.?bit|bit.?exact` and fails if the same crate's tree does not contain an
  external `github.com/<org>/<repo>` reference or an explicit
  `no external reference` / `no reference implementation exists` disclaimer.
- The same grep applies to the words "official", "reference implementation", and
  "upstream" when they modify a claim of agreement.
- A test that names a comparison file must have that file on disk. `burn-sct` today
  names `tests/ref_data/*.bin` and `gen_reference.py`; neither exists.

## Crates that must change their wording today

Five false claims. The replacement is exact and ready to paste.

### 1. `burn-sct/README.md:68-84` — the whole "Bit-exactness vs the reference" section

**False.** `tests/cmp_reference.rs:49` is behind `#[cfg(feature = "binary-tests")]`;
`binary-tests = []` is not in `default = ["std"]` (`Cargo.toml:15`); `tests/ref_data/`
does not exist in the tree; `gen_reference.py`, named at `README.md:74`, does not
exist in the crate at all. The test has never run. The reported results at `:81`
("forward ~2e-7, retract ~1e-7, `from_dense` ≤ 5.9e-4") cannot have been produced by
any run in this repository.

Replace lines 68-84 with:

> ## Reference comparison — not currently runnable
>
> `tests/cmp_reference.rs` is a comparison harness against a PyTorch reference for
> [EctoSpace/SCT](https://github.com/EctoSpace/SCT). **It has never been executed here**:
> it is behind the non-default `binary-tests` feature, its fixture
> (`tests/ref_data/*.bin`) and its generator (`gen_reference.py`) are not in the tree.
> Until a generator lands and the fixture is committed, this crate is **kind (d) —
> no external reference**; its correctness evidence is the `retract` invariant tests in
> `src/lib.rs` and `tests/cuda_retract.rs`.
> The intended upgrade is to regenerate from `spectral_compact_training/spectral_layer.py`
> in that repo.

### 2. `burn-sct/src/lib.rs:321-322` — "verified bit-close by tests/cmp_reference.rs"

**False.** Cites a test that has never run.

Replace:

> /// (PyTorch linalg.qr + sign flip), matching the paper's `safe_qr`. **Not yet
> /// verified against the authors' code** — `tests/cmp_reference.rs` exists but cannot
> /// run (no fixture, no generator, non-default feature).

### 3. `burn-gdn2/README.md:22` — "| Verification | 1000-case bit-exact reference tests … |"

**False, and self-contradicted.** The crate's own `tests/bit_exact.rs:8-10` says
verbatim: *"despite this file's name this is NOT a bit-for-bit comparison: it is an
absolute-tolerance comparison of two independent implementations of the same math."*

Replace the table cell with:

> | Verification | 1000-case reference tests vs **our** PyTorch transcription of `lit_gpt/gdn2.py`, absolute tolerance 5e-4 — not bit-for-bit | none shipped |

### 4. `burn-gdn2/README.md:262` — "# 1000-case bit-exact vs the paper reference"

**False** on both halves: not bit-exact, and not "the paper reference" — it is a
transcription of the paper's *layer*, with the Triton kernel's recurrence replaced by
our own per-token scan.

Replace:

> cargo test -p burn-gdn2 --features binary-tests    # 1000 cases vs our transcription of NVlabs' layer, tol 5e-4

### 5. `burn-dspark/src/lib.rs:5-6` — "matched against the official DeepSpec implementation"

**False.** Seven tests exist (`src/lib.rs:284-380`, `src/markov.rs`, `src/sampling.rs`);
none compares against DeepSpec. Per checklist item 1 they are all kind (d): shape,
range, finiteness, `sts_reduces_ece`.

Replace:

> //! Building blocks from [DSpark](https://arxiv.org/abs/2607.05147) (DeepSeek AI, 2026).
> //! Transcribed from the paper's Eq. 5-12. **No parity test against
> //! [DeepSpec](https://github.com/deepseek-ai/DeepSpec) exists yet**, so the loss
> //! (`w_k = exp(-k/gamma)`, the 0.1/0.9/1.0 mixing, `L_tv` on softmax probabilities
> //! rather than logits) is unverified against the authors' code. The tests here are
> //! kind (d): shapes, ranges, finiteness.

Two further relabellings that are not falsehoods but read as claims:

- **`burn-engram/src/hasher.rs:3`** — "Port of `NgramHashMapping._get_ngram_hashes`
  from the official reference". Defensible as a provenance statement, and `:12-13`
  honestly says "Same structure, different constants". But the multiplier stream is
  splitmix64, **not** the reference's PCG64, so this port is *structurally incapable*
  of matching the reference bit-for-bit. The README/doc must say so in the headline,
  not in line 12. The test at `:219` cannot detect it: its expected values are the same
  expressions the implementation evaluates.
- **`burn-ttt/src/lib.rs:13`** — "Matched to the official implementation
  `test-time-training/e2e` (`ttt/model/loss.py`)". No test compares against that file;
  the four tests are zero-when-perfect and masked-mean checks. Relabel to
  "transcribed from `ttt/model/loss.py`; not yet compared against it".
  **RESOLVED BY DELETION 2026-09-28** (`docs/architecture/library-crate-fate.md`): `burn-ttt` was
  unreachable from `crates/dormouse-*` and had no other reason to exist, so it was
  removed — the claim needed no relabelling because it no longer exists. The finding
  stays here because the class is not retired: every remaining crate still has to
  place itself in the taxonomy.

## What this costs, and what it buys

The three highest-value fixes, by fidelity gained per line written:

1. **`burn-gdn2`** — the harness, the committed fixture, the regenerable generator and
   the CI diff all exist. Only the reference body changes: replace the per-token scan
   in `tests/gen_reference.py` with the recurrence from `NVlabs/GatedDeltaNet-2`
   `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py`. ~60 lines of Python. This is the crate
   that carries the headline, so it is also the one whose wording is most often quoted.
2. **`burn-sct`** — 2,738 LOC claiming a bit-exactness it never ran. Write
   `gen_reference.py` against `EctoSpace/SCT` `spectral_compact_training/spectral_layer.py`
   (confirmed live, 3,947 bytes), commit four fixtures, put `binary-tests` in the CI job
   that exists. ~150 lines. Or delete the claim per item 1 above and stop there.
3. **`burn-dspark`** — turns a false claim into a real gate. `deepseek-ai/DeepSpec`
   `deepspec/modeling/dspark/{loss.py, markov_head.py, common.py}`. ~150 lines of
   Python + one Rust test. The loss has five weighted terms; a `gamma` error or an
   L1-on-logits-vs-probs error is invisible to every test in the crate today.

## One gate that is not about wording

The largest hole is not linguistic. `.github/workflows/fused-library.yml`'s `cuda-tests`
job declares `runs-on: [self-hosted, gpu, linux]` and its own comment records that **no
runner is registered**, so it has never executed. Every CUDA-executing test in the
library has therefore never run in CI — 9 in `burn-attnres`, 5 in `burn-mhc`, 4 each in
`burn-rope` and `burn-kda`, 6 in `burn-situ`, and the fused paths of
`burn-gdn2`/`burn-spectral`/`burn-bitnet`. The `cuda-compiles` job does
`cargo test --no-run`, which cannot fail on a CUDA-only bug. Checklist item 5 is
therefore unsatisfiable for the whole library until a GPU runner exists, and no amount
of README editing fixes that.

`burn-bitnet` is the sharpest instance: `src/fwt_cuda.rs` holds exactly one test,
`bitnet_bench` at `:535`, which is `#[ignore]`d and is a benchmark. The fused CUDA path
in that crate has zero correctness coverage.
