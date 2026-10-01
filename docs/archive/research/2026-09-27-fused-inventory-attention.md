# Fused-library inventory: ATTENTION / RECURRENCE / RESIDUAL-STREAM crates

Date: 2026-09-27. Slice owner: inventory subagent for ADR-0017.
Scope: `vendor/burn-fused/crates/{burn-attnres, burn-gdn2, burn-kda, burn-mhc, burn-mod, burn-mor, burn-nope}`.
Nothing in the repo was modified; nothing was committed. Only GPU work: four short
test-suite runs (< 60 s each) after `nvidia-smi --query-compute-apps` showed no
other compute process. No training run was started.

**The one thing to read first.** Every fused CUDA path in this slice is gated on a
`TypeId` comparison against a *hardcoded* checkpointing strategy. Our trainer's
backend is

```rust
// crates/dormouse-train/src/lib.rs:33-36
pub type Backend = burn::backend::autodiff::Autodiff<
    burn_cuda::Cuda,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;
```

and `Autodiff<B, C = NoCheckpointing>` (`burn-autodiff-0.22.0-pre.4/src/backend.rs:25`)
carries the strategy as a *type parameter*. So `TypeId::of::<Autodiff<Cuda, Balanced>>() !=
TypeId::of::<Autodiff<Cuda>>()`. The gates:

| crate | gate | site |
|---|---|---|
| `burn-attnres` | `depth_attend_autodiff::<CudaBare, 64>` needs `try_into_primitive::<Autodiff<CubeBackend>>()` | `fused_attnres.rs:1213-1223`, called at `lib.rs:111` |
| `burn-attnres` | `cube_of` downcasts `try_into_primitive::<CubeBackend>()` | `fused_attnres.rs:84-89` |
| `burn-mhc` | `TypeId::of::<Inner>() == TypeId::of::<CudaBare>()` | `sinkhorn.rs:25-27`, `sinkhorn_cuda.rs:360` |
| `burn-kda` | `is_autodiff_cuda::<B>()` = `TypeId::of::<B>() == TypeId::of::<Autodiff<CudaBare>>()` | `fused.rs:34-36`, used at `lib.rs:565` |

Consequence: under the backend the trainer runs, `burn-kda` silently takes
`chunk_wy_forward` (tensor ops) and the fused chunk kernels never launch — the only
crate here that runs in production has its kernel disabled by a one-token type
difference. `burn-attnres` and `burn-mhc` are unaffected in practice because nothing
calls them at all. This agrees with, and extends, the existing
`docs/archive/research/2026-09-27-attenres.md` §1.2 (independently derived).

---

## 1. burn-attnres — BROKEN (fused streaming path)

**1. What it is.** Replaces PreNorm's fixed unit-weight accumulation of all layer
outputs with a softmax over the depth axis driven by one learned pseudo-query per
layer (zero-initialised, which makes the initial alpha exactly uniform); the
`Block` mode partitions layers into blocks, attends over block summaries via an
online-softmax merge, and cuts memory from O(L·d) to O(N·d).
arXiv **2603.15031** — "Attention Residuals" (Kimi Team). **VERIFIED** against
`arxiv.org/abs/2603.15031` (citation_title confirmed). Claimed headline
(Kimi Linear 48B/3B-activated, 1.4T tokens): GPQA +7.5, HumanEval +3.1, MMLU +1.1.
The abstract confirms the 48B/3B setup and says only "improves downstream
performance across all evaluated tasks" — the three numbers are paper-body, not in
the abstract.

**2. Size and shape.** 2060 src LOC (`lib.rs` 495 + `fused_attnres.rs` 1565) + 64 bench.
**Fused kernels: yes** — 6 `#[cube(launch_unchecked)]`, and every launch site is
hardcoded `::<f32>` (`fused_attnres.rs:495,507,...`), so a bf16 build would still
launch f32 kernels. `CHUNK_G = 8` chunking keeps peak memory at (G+2)·B·T·D instead
of (L+1)·B·T·D; the chunk stack is a `thread_local` single-slot cache keyed on
`(g,b,t,d)` (`fused_attnres.rs:43-82`). Public API: `AttnRes::new/forward`,
`BlockAttnRes::new/forward/step/init_state`, `depth_attend(&[Tensor<3>], Tensor<1>)`,
`two_phase_attend`, `BlockAttnState`. Features `std` (default) / `cuda` / `autodiff`;
needs `burn-cubecl` + `cubecl` for the fused path.

**3. State.** Builds today on ndarray; `cuda` + `autodiff` both compile clean.
Tests: 8 ndarray unit tests, all pass. With `--features cuda,autodiff`: **16 pass,
1 FAILS, 2 ignored**:

```
fused_attnres::tests::streaming_fused_matches_tensor_path ... FAILED
  panicked at crates/burn-attnres/src/fused_attnres.rs:951:
  step 3: maxdiff 0.8292272      (re-run: step 3: maxdiff 0.85232615)
```

Reproducible, same step and magnitude across runs. This is the crate's *own*
CUDA-vs-CPU equivalence test for the fused `BlockAttnRes` streaming step. It is
**not** one of the four documented sm_120 landmines (no Bool→float cast, no bf16
matmul, no f16, no top-k→gather in that kernel). The isolated components pass
(`source_score_fused_matches_tensor`, `merge_fused_matches_tensor`), so the bug is
in the *composition* (first real online-softmax merge over a state produced by
`incorporate`), not in either kernel alone. Verification harness: the strongest in
this slice — `fused_attnres::tests` / `ad_tests` / `fd_tests`, including
finite-difference gradient checks (`fused_backward_matches_burn_autodiff`,
`depth_attend_grad_matches_finite_difference`), all green. Tolerance-based (1e-4
relative), not bit-for-bit.

**4. Wiring.** **0 references** anywhere in `crates/dormouse-*`. Not in
`dormouse-core/Cargo.toml`, not in `configs/`, not in any `src/`. DEAD-but-implemented.

**5. Verdict: BROKEN** — `streaming_fused_matches_tensor_path`, `maxdiff 0.83` at
step 3, the fused `BlockAttnRes` streaming path only. The full-depth
`depth_attend` fused path and both fused backward kernels are verified green, so
"delete the crate" would be wrong; "trust the streaming path" would also be wrong.

**6. Overlap.** See the map at the end.

---

## 2. burn-gdn2 — WIRED (as burn-kda's engine; its own layer is unused)

**1. What it is.** Gated DeltaNet 2: a linear-complexity recurrent token mixer that
decouples the delta rule's *erase* gate (per key channel, `b`) from its *write* gate
(per value channel, `w`); training uses a chunked WY decomposition, decoding a fused
per-token state recurrence. `allow_neg_eigval` lifts the erase range to [0,2] to
allow negative eigenvalues in the state transition.
arXiv: **the crate cites none.** `grep -rn "arxiv\|arXiv\|2412" src/` returns nothing;
`config.rs:24-26` references the follow-up by title only
("Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"). The parent
paper is Gated DeltaNet (2412.06464, Yang et al.) but that ID appears nowhere in the
crate. **NOT VERIFIED — the crate carries no arXiv ID at all**, which violates
ADR-0017 §3's own rule that every library crate have one. Claimed headline: none in
the crate.

**2. Size and shape.** 4186 src LOC + 2910 test LOC. **Fused kernels: yes** —
`chunk_cube.rs` 1243, `chunk_adjoint_cube.rs` 700, `fused_recurrent_cube.rs` 176,
each with a tensor-ops twin. Public API: `GatedDeltaNet2::new/forward/forward_train/
forward_train_fused`, `chunk_wy_forward`, `chunk_autodiff_or_plain`,
`fused_recurrent_forward`, `l2_normalize`, `short_conv_1d`, `Gdn2Config`/`Gdn2Mode`.
Features `std` / `autodiff` / `cuda` / `binary-tests`.

**3. State. UNDER CONCURRENT EDIT — not built, per instruction.** The other agent
added `src/alloc_trace.rs`, `tests/alloc_probe.rs`, `tests/lowp_bf16_cuda.rs` today
17:49-17:58 and modified `forward.rs`, `chunk_cube.rs`, `chunk_adjoint_cube.rs`,
`lib.rs`. Tests: 10 test files. The only **bit-exact** reference harness in this
slice is `tests/bit_exact.rs` + `tests/gen_reference.py` behind `feature =
"binary-tests"`, reading `include_bytes!("ref_data.bin")` — **and `ref_data.bin` does
not exist in the repo** (`find . -name 'ref_data*'` → empty). So the feature does not
compile, and without the feature the file contributes 0 tests. That harness has
therefore never run here.

**4. Wiring.** **Yes, 3 call sites, all in one dev-only file**:
`crates/dormouse-core/Cargo.toml:31` declares it a **dev-dependency** (with a comment
saying so) and `crates/dormouse-core/examples/kda_backward_probe.rs:165,218` call
`burn_gdn2::chunk_wy_forward`. Zero `src/` call sites. **Trap for the reader:** the
model field is named `loop_block.shared_attn.gdn2` but its type is
`burn_kda::KdaModule` (`crates/dormouse-core/src/attention.rs:12`) — the name is a
historical artifact kept for checkpoint-prefix stability, not evidence of a gdn2
wiring. It is, however, a **load-bearing production dependency of `burn-kda`**, which
imports `chunk_wy_forward`, `l2_normalize`, `short_conv_1d`, `Gdn2Config`, `CudaBare`
and calls its fused kernels. Only the *tensor* half of that runs under our backend
(see the TypeId table).

**5. Verdict: WIRED** — as burn-kda's chunked-WY engine. (The `GatedDeltaNet2` module
layer, 620 LOC of `module.rs`, is the unused part.)

**6. Overlap.** See the map at the end.

---

## 3. burn-kda — WIRED (and its fused path is dead on arrival)

**1. What it is.** Kimi Delta Attention: the delta rule with a *data-dependent*
per-head write strength `β_t = σ(W_β x_t)` (K3 Eq 2) and a *channel-wise* decay
`α_t = exp(g_t)` from a low-rank logit (Eq 2), where `g_t = g_min·σ(e^{A_h} z_t)` is
lower-bounded at `g_min = -5` (K3 Eq 5) — Kimi Linear instead uses
`-e^{A_h}·softplus(z)`, unbounded. Training runs the chunkwise WY form.
arXiv **2510.26692** ("Kimi Linear") and **2607.24653** ("Kimi K3") — both
**VERIFIED** against `arxiv.org/abs/...`. Headline: KDA is what lets Kimi Linear run
48B/3B-activated at KDA cost; K3's contribution over it is the bounded decay and the
full-rank output gate.

**2. Size and shape.** 1208 src LOC (of which `fused.rs` is 73) + 759 test/example.
**Fused kernels: borrowed, not owned** — `fused.rs:69-71` calls
`burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward`, gated on
`chunk_size <= 16` because the reused kernel underflows f32 once `cumsum(g) < -88`.
Public API: `KdaModule::new/forward_train/forward_train_state/forward_train_fused/
forward/forward_recurrent`, `KdaConfig`, `kda_step`, `KdaDecay`, `DecayFn`, `GateMode`.
Features `std` / `autodiff` / `cuda` (each forwarded to `burn-gdn2`).

**3. State. UNDER CONCURRENT EDIT — not built, per instruction** (its
`examples/kda_step_probe.rs` was modified today and its dependency gdn2 is being
edited right now). Static reading of the state: 7 in-crate ndarray unit tests (decay
bounds, decay data-dependence, shapes, `chunked_decode_matches_recurrent`), plus
`tests/fused_cuda.rs` (fused == exact decode, `max_diff < 1e-3`),
`tests/bench_cuda.rs`, and four examples of which `bitforbit.rs` /
`bitforbit_cuda.rs` are the named bit-for-bit harness. **Every one of the fused
tests instantiates the BARE backend** (`type Bare = burn_kda::fused::cuda::CudaBare`),
so nothing verifies the path the trainer takes. And it does not take it:
`is_autodiff_cuda::<B>()` requires `TypeId::of::<B>() == TypeId::of::<Autodiff<CudaBare>>()`
and the trainer is `Autodiff<Cuda, BalancedCheckpointing>` → false → `chunk_wy_forward`
tensor ops. Our own `AGENTS.md` says "the KDA fused chunked kernel is f32-only
(falls back to tensor ops)"; the stronger, code-level truth is that it falls back for
a *type-identity* reason, before f32-ness is ever considered.

**4. Wiring.** **3 production call sites**, the only live wiring in this slice:
`crates/dormouse-core/src/attention.rs:8` (`use burn_kda::KdaModule`), `:17-28`
(`KdaConfig`), `:30` (`KdaModule::new`); `crates/dormouse-core/src/loop_block.rs:272`
(`forward_train_state`). `Cargo.toml:38-39` enables `burn-kda/cuda` +
`burn-kda/autodiff` — which is exactly the setup that does *not* select the kernel.
Param prefix `loop_block.shared_attn.gdn2.*` feeds the Muon+ routing policy
(`dormouse-train/src/optim.rs:127`).

**5. Verdict: WIRED** (3 call sites, the only attention arm).

**6. Overlap.** See the map at the end.

---

## 4. burn-mhc — IMPLEMENTED-UNUSED (best verified, zero used)

**1. What it is.** Manifold-Constrained Hyper-Connections: widens the residual stream
to `n` branches and mixes them with a first-order hyper-network producing per-token
`H_pre`/`H_post` (sigmoid and 2·sigmoid, non-negativity) and an `n×n` `H_res`
projected onto the Birkhoff polytope by Sinkhorn-Knopp, which restores the
identity-mapping property that plain Hyper-Connections destroy.
arXiv **2512.24880** — "mHC: Manifold-Constrained Hyper-Connections" (DeepSeek).
**VERIFIED**. Claimed headline (paper): effective at scale, tangible improvements and
superior scalability, identity mapping restored. The crate's own doc adds no numbers.

**2. Size and shape.** 1011 src LOC, of which 568 is the fused Sinkhorn. **Fused
kernel: yes, one** — a single `#[cube]` that runs all 20 iterations in one launch,
plus a hand-derived **exact** backward that recomputes only the per-step
normalization sums (`sinkhorn_cuda.rs:392-395`). Public API: `MhcBlock::new/
forward/forward_no_residual/hyper_mappings`, `sinkhorn_knopp`, `SINKHORN_ITERS`.
Features `std` / `cuda` / `autodiff`.

**3. State.** Builds ndarray and compiles `cuda`+`autodiff`. Tests:
ndarray 7/7 pass. `--features cuda`: 8 pass, 1 ignored, **0 fail**.
`--features cuda,autodiff`: **10 pass, 2 ignored, 0 fail**, including
`sinkhorn_fused_backward_matches_tensor` and `sinkhorn_grad_matches_finite_difference`.
**This is the only crate in the slice whose fused forward, fused exact backward and
finite-difference gradient check all pass on this sm_120 box today.** Tolerance 1e-3,
not bit-for-bit.
Reachability: dead for us (TypeId gate). One latent defect worth naming: the fused
path's *own* fallback inside `sinkhorn_autodiff` (`sinkhorn_cuda.rs:364-371`) is the
naive `m / m.sum` form, **not** the log-domain form whose doc comment
(`sinkhorn.rs:12-21`) explains that the naive form overflows f32 and NaNs at extreme
logits. The regression test `sinkhorn_large_logits_stay_finite` only exercises the
*non-autodiff* path, so the autodiff fallback's overflow behaviour is untested.

**4. Wiring.** **0 references.** DEAD-but-implemented.

**5. Verdict: IMPLEMENTED-UNUSED.**

**6. Overlap.** See the map at the end.

---

## 5. burn-mod — DUPLICATE-OF burn-mor (routing core; `ModPredictor` is unique)

**1. What it is.** Mixture-of-Depths: caps the tokens entering a block at
`k = round(C·T)` selected by expert-choice top-k on a learned scalar router weight;
routed tokens get `x + r·f(x~)` (eq. 1), the rest pass through untouched. Because
top-k is non-causal, the paper adds a BCE aux loss and a small second MLP predictor
(§3.5 "Sampling") so autoregressive sampling can route on `weight > 0.5` alone.
arXiv **2404.02258** — **VERIFIED**. Headline: matches baseline at equal FLOPs and
wall-clock while using a fraction of the forward FLOPs; up to 50% faster steps in
post-training sampling.

**2. Size and shape.** 411 src LOC. **Fused kernel: none — tensor ops only**, and the
crate has no `cuda` feature at all. Public API: `ModRouter::new/weights`,
`route_block(x, &router, capacity_frac, block, dev)`, `select_topk`, `gather_selected`,
`bce_aux_loss`, `ModPredictor`, `ModConfig`. Features `std` only.

**3. State.** Builds ndarray; 4/4 tests pass. Zero CUDA. **Backward is explicitly
unverified** — `predictor_learns_targets` says so in the test body: "Autodiff +
scatter of Int indices is broken on 0.22, so no backward here". `select_topk` is
rounds of `argtopk(take<=16)` + Add-scatter (`routing.rs:19-41`), i.e. it is built on
the exact primitive **ADR-0015 indicts** (upstream `cubek-reduce` 0.3.0-pre.4
`ArgTopK` emitting `u32::MAX` sentinel coordinates → `gather` →
`CUDA_ERROR_ILLEGAL_ADDRESS`). It dodges the documented trigger (it masks with
`-1e30`, not `-inf`, and clamps `k ≤ T-1`), so it is probably safe — but it is the
primitive `burn-mor` deliberately abandoned, and `burn-mod`'s own "ponytail" note
still points at `topk_with_indices`, which `burn-mor` rejected as unsafe under
autodiff. No reference harness.

**4. Wiring.** **0 references.** DEAD-but-implemented.

**5. Verdict: DUPLICATE-OF burn-mor** — the top-k-token-selection + gather/scatter
core is the same problem, and burn-mor has the committed, safe primitive and the
better conceptual fit. The one piece burn-mor does not have is `ModPredictor`, the
§3.5 sampling router.

**6. Overlap.** See the map at the end.

---

## 6. burn-mor — BROKEN (mid-edit; committed state was green)

**1. What it is.** Mixture-of-Recursions: one shared layer stack re-applied up to N
recursion steps while a linear `d→1` router gives each token its own depth; per step
the top-k of the still-active tokens is gathered into a dense active batch so the
quadratic attention and FFN run only on those, then scattered back (dropped tokens
get a zero block output and pass through the residual). Ships a load-balancing aux
loss to stop router collapse.
arXiv **2507.10524** — **VERIFIED**. Headline: new Pareto frontier from 135M to 1.7B —
lower validation perplexity and better few-shot accuracy at equal training FLOPs and
*smaller* model size, with higher throughput.

**2. Size and shape.** 727 src LOC + 71 example, plus the 11 KB `src/topk_gather.rs`
added today. **No kernel of its own** — a `cuda` feature was *added today* only to
gate the `topk_gather_repro` example, because the primitive is a cubecl-side
`cubek-reduce` repair, not a fused kernel. Public API: `MoRRouter::new/scores`,
`select_active` / `select_active_only`, `gather_active`, `scatter_active`,
`topk_indices`, `load_balancing_loss`, `MoRConfig`.
`topk_indices` is deliberately `argsort_descending` + `narrow`, not `argtopk`:
`topk.rs:9-13` records that cubecl's ArgTopK had a "documented garbage-ArgTopK defect"
and pre.2 still shows uninitialized reads and wild indices under compute-sanitizer.

**3. State. UNDER CONCURRENT EDIT, and the tree does not compile right now:**

```
error[E0599]: no method named `int` found for struct `Tensor<3, burn::tensor::Int>`
  --> crates/burn-mor/src/topk_gather.rs:50:26
   |
50 |     scores.argtopk(k, 2).int()
   = help: method `int` is available on `Tensor<3>` (use `.into()`)
```

×2 (also `:37`). `argtopk` already returns an Int tensor in 0.22.0-pre.4; the fix is
`.cast(IntDType::I64)`, which is what the committed `topk.rs:40` does. This landed
between my first grep and my first build. The **pre-edit** state is what I measured:
builds clean, **7/7 ndarray tests pass plus 1 doctest** (`crates/burn-mor/src/lib.rs`
usage sketch). No cuda verification yet; the replacement primitive needs the
`cubek-fix` vendor crate, which is still untracked (`?? vendor/cubek-fix/`).

**4. Wiring. Being wired RIGHT NOW, by another agent.** `crates/dormouse-core/Cargo.toml:21`
(dependency) and `:31` (dev-dependency), plus the new untracked
`crates/dormouse-core/src/mor.rs` with 2 call sites — `pub use burn_mor::MoRRouter`
and `burn_mor::topk_indices` — plus `configs/mor.toml` and `tools/mor_ab.sh`. Zero
committed history of use: the crate is unwired in `HEAD` and half-wired in the
working tree.

**5. Verdict: BROKEN** — `E0599: no method named int` ×2 in `src/topk_gather.rs`
(concurrent edit, not a design flaw). The committed version is IMPLEMENTED-UNUSED.

**6. Overlap.** See the map at the end.

---

## 7. burn-nope — IMPLEMENTED-UNUSED (102 lines, and the premise is contradicted)

**1. What it is.** Content-based Q·K^T attention with **no RoPE rotation** — the
same softmax attention, same causal mask, minus positional encoding; the doc claims
it "works best with recurrent architectures (KDA/GDN2) where temporal order is
captured through state updates".
arXiv **2607.24653** — "Kimi K3: Open Frontier Intelligence" — **VERIFIED** as the
cited source, but **the abstract makes no NoPE claim at all**, and our own
`AGENTS.md` (playbook item 5) records the opposite finding from our own runs: "keep
RoPE in the attention arm — NoPE → endless generation after post-training".

**2. Size and shape.** 102 src LOC. **No kernel — tensor ops only, no `cuda` feature
whatsoever**, two dependencies (`burn`, `burn-tensor`). The entire public API is one
function: `nope_attention(q, k, v, causal) -> Tensor<4>`.

**3. State.** Builds; 3/3 tests pass (2 shape, 1 semantic: causal row *t* equals
bidirectional attention restricted to the prefix). No reference harness, no perf
claim, no measured result of any kind.

**4. Wiring.** **0 references.** The only `nope` hits in the workspace are unrelated
string collisions (`ActQuant` parse tests, a cfg-validation test).

**5. Verdict: IMPLEMENTED-UNUSED.** 102 lines, one function, zero wiring, zero
measurements, and the enabling assumption is contradicted by our own documented
experiment. Deletable in one commit.

**6. Overlap.** None inside this slice. Nearest published relative: Kimi K3's own
attention design, from which the crate borrows only the arXiv ID.

---

## Overlap map

**Residual stream (4 mechanisms, 0 comparisons ever run).** `burn-attnres` (softmax
over depth, *drops* identity-mapping) and `burn-mhc` (Sinkhorn-projected
doubly-stochastic branch mixing, *keeps* it by construction) are direct published
competitors — same job, opposite choice on the one property HC/mHC argue matters
most, and the papers never meet. Against our own two: ReZero
(`loop_block.rs:209`, scalar, init 1.0) and Gated Residual (`gr.rs`, 117 LOC, behind
`use_gr`, never A/B'd) are the cheap end of the same axis. ADR-0017 already flagged
this; the sharp new datum is that mHC is the only *verified* mechanism of the four
and the residual stream is the one axis where a free, already-written A/B exists.

**Recurrence (not competitors — one is the other's engine).** `burn-kda` is GDN-2
under a relabelling: `fused.rs:3-14` states the mapping exactly (`g = log α`,
`b = β_k`, `w_gate = β_v`, `scale = 1`) and reuses gdn2's chunk kernels verbatim.
Published relatives: Kimi Linear 2510.26692 and GDN-2 (referenced by title only, no
arXiv ID in the crate). There is no redundancy to reclaim here — but there *is* an
unclaimed dependency: `burn-gdn2` is a production dependency with zero arXiv credit.

**Compute routing (the real duplicate).** `burn-mod` and `burn-mor` solve the same
problem — top-k token selection plus gather/scatter around a block — with different
primitives and different fates. `burn-mod::select_topk` is built on `argtopk`, the
primitive ADR-0015 traced to a `u32::MAX` sentinel index; `burn-mor::topk_indices`
is `argsort_descending` + `narrow`, committed precisely because of that defect, and
is the one actually being wired into the model today. Of the two, only MoR's
*recursive* framing fits a shared-weight `LoopBlock` (per-token depth over a shared
stack) — MoD is per-layer, not per-token, and would need restructuring to apply.
`burn-nope` overlaps nothing in the slice; it is the only one of the seven with no
published mechanism actually implemented (attention without RoPE is a
*configuration choice*, not a mechanism, and ours is already RoPE).

---

## Most surprising finding

The fused kernel in every crate that has one is unreachable under the backend our
trainer actually runs, and the reason is not numerical — it is a type-identity check
against a hardcoded checkpointing strategy. All four gates compare
`TypeId::of::<B>()` to `Autodiff<Cuda, NoCheckpointing>` (or downcast to a bare
`CubeBackend`), while `dormouse-train` uses `Autodiff<Cuda, BalancedCheckpointing>`.
So the one crate we actually run, `burn-kda`, silently executes its tensor-ops chunk
path, and its entire fused verification suite instantiates the bare backend that the
trainer never uses. Meanwhile the crate whose fused path *is* verified three ways
including finite differences, `burn-mhc`, is one of the 16 nobody wired — and the
crate that would be the natural A/B partner, `burn-attnres`, fails its own fused
streaming equivalence test today at `maxdiff 0.83`.

---

## What I did not do

- Did not build `burn-kda` or `burn-gdn2` (concurrent edit, as instructed).
- Did not run the gdn2 `binary-tests` harness: its `ref_data.bin` is absent, so the
  feature cannot compile. That is a finding, not a gap in my run.
- Did not verify gdn2's claimed arithmetic against a published reference: the crate
  cites no arXiv ID, so there is nothing to verify against.
- Did not modify anything, including the two-line `int()` → `cast(IntDType::I64)`
  compile fix in `burn-mor/src/topk_gather.rs` that another agent is mid-way
  through writing.
