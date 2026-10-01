# dormouse-fused inventory — slice: MEMORY / LEARNED-STATE + OBJECTIVES

Date 2026-09-27. Slice owner: subagent. Read-only on code; nothing built that is under
concurrent edit; no training run; no GPU work (another process held the GPU at 47 + 162 MiB
throughout, so all verification was CPU).

**Method.** Read `vendor/burn-fused/crates/*/src/**` (not the doc comments), grepped
`crates/dormouse-*` for wiring, ran `cargo test` on every crate in the slice except
`burn-engram`, checked the one fused-CUDA crate's feature/dispatch path by hand plus
`cargo check --features cuda,autodiff`, and verified **all 18 arXiv IDs in one call against
the arXiv API** (`export.arxiv.org/api/query?id_list=...`). Every ID resolved and every
title matched the crate's claim. No "NOT VERIFIED" entries in this slice.

**One structural fact that colours every row below:** 13 of the 26 workspace crates define
a `cuda` feature; **exactly one crate in this slice is a real fused CUDA kernel**
(`burn-situ`). Everything else in the memory/objective half is pure `Tensor` ops, and
`burn-engram` says so explicitly in its own source (`lib.rs:79`, "this crate has no cubecl
dependency or kernel infrastructure"). So for this slice the "fused or tensor ops" column
is answered: tensor ops, and that is not a defect — none of these are kernel candidates
except SiTU-GLU.

---

## 1. `burn-engram` — Conditional Memory via Scalable Lookup (Engram)

**What.** N-gram **hash embedding** tables (3 slots by default, one per n-gram order, each a
prime-sized `Embedding`) looked up in O(1), concatenated, then fused back into the residual
by a multi-head gate: `value_proj(e)` scaled per head by
`sigmoid(sqrt(|s|+1e-6)·sign(s))` where `s` is the RMS-normalised dot of a per-head
`key_proj(e)` against that head's slice of the hidden state, plus an optional
zero-initialised depthwise causal `ShortConv` (Eq 5).
arXiv **2601.07372** (DeepSeek, 2026) — **VERIFIED** via arXiv API, title *"Conditional
Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"*, v2.
Headline, verbatim from the abstract: Engram scaled to **27B params** beats a strictly
iso-param/iso-FLOP MoE; **MMLU +3.4, CMMLU +4.0, BBH +5.0, ARC-C +3.7, HumanEval +3.0,
MATH +2.4**, Multi-Query NIAH **84.2 → 97.0**. A **U-shaped scaling law** allocates tokens
between MoE compute and static memory.

**Size and shape.** **586 LOC**, 2 files (`lib.rs` 331 + `hasher.rs` 255). Tensor ops only —
no cubecl dependency, by design and documented. Public API: `EngramModule::{new,
with_short_conv, forward, forward_embeds}`, `MultiHashEmbedding::{new, forward}`,
`depthwise_conv_1d`, `compute_gate`, and the `hasher::NgramHasher::{new, table_sizes,
num_tables, hash_ids, hash_tensor}`. Features: `std` only — **no `cuda` feature**; it runs on
any backend through the generic `B: Backend` bound, so ndarray and cuda both work with zero
code change.

**State.** **Under concurrent edit by another agent right now** (they are mid-flight in
`crates/dormouse-core/src/loop_block.rs`, +346 lines, plus `config/schema.rs` and every
`configs/*.toml`). I did **not** build it. Pre-existing state: 10 inline `#[test]`s
(5 shape/gate, 4 hasher, 1 formula). One of them **is** a reference check worth naming:
`gate_matches_reference_formula` (`lib.rs:300`) hand-computes the gate from
`deepseek-ai/Engram engram_demo_v1.py`'s `sigmoid(sqrt(|s|+1e-6)·sign(s))` and asserts the
plain `sigmoid(s)` variant is *rejected* — that is a hand-transcribed parity assertion, not a
run of the reference. `hasher::tests::matches_reference_algorithm` does the same for the
XOR-mix. **No bit-for-bit harness against a running reference anywhere in the slice.**

**Wiring.** 3 references, **2 real call sites**: `loop_block.rs:304` (`forward`, in-VRAM
FNV path) and `loop_block.rs:289` (`forward_embeds`, the host-RAM offload path), plus the
`use burn_engram::EngramModule` at `loop_block.rs:9` and the Cargo path at
`dormouse-core/Cargo.toml:14`. Optimizer routing is keyed on the literal strings
`engram.memory` / `engram.key_projs` (`dormouse-train/src/optim.rs:75,81`) with three
assertions pinning the markers.

> **The finding that matters for the re-enable.** 255 of the crate's 586 LOC — 43% —
> is `NgramHasher`, and **it is dead**. The trainer never calls it. It has its own FNV-1a
> in `dormouse-data`: `hashes_raw` returns the low 32 bits of FNV-1a over 3/5/8-grams
> (`dormouse-data/src/lib.rs:407`), and the in-model read **masks** rather than divides —
> `hash & mask` with `mask` from `engram_tables` (rounding `engram_rows` up to a power of
> two, `loop_block.rs:24-34`). So the addressing actually in use is FNV-1a-truncated-to-32
> mod a power of two, **not** the paper's splitmix64 odd-multiplier XOR-mix reduced mod a
> distinct prime per (ngram, head) slot. That is the *entire* mechanism the prime-per-slot
> design exists to provide, and it is not what runs. The other agent is mid-way through
> parameterising it (`engram_rows`, `engram_orders = [2,3,4]`, `engram_lam_max = 0.5` are
> now schema fields; `engram_tables` rounds 500 000 → 524 288).

**Verdict: WIRED** — with `NgramHasher` a 255-LOC dead branch inside it (delete, or wire it;
right now the crate advertises a paper-faithful hasher the trainer silently does not use).

---

## 2. `burn-fastblt` — Byte Latent Transformer + FastBLT

**What.** Two unrelated halves in one crate. (a) BLT: entropy-based dynamic byte patching
(a byte opens a patch when its next-byte predictive entropy exceeds a threshold) and byte
n-gram **hash embeddings** — one `Embedding` per (hash function, group size) **summed** onto
the base byte embedding. (b) FastBLT: absorbing-discrete block diffusion loss
(`L = -(1/t)·Σ_{masked} log p`), greedy draft verification with a free bonus byte, and
confidence-/entropy-bounded block unmasking, all device-side with zero host sync.
arXiv **2412.09871** (BLT, Meta 2024) and **2605.08044** (FastBLT) — **BOTH VERIFIED**,
titles *"Byte Latent Transformer: Patches Scale Better Than Tokens"* and *"Fast Byte Latent
Transformer"*. Headline: BLT matches tokenised LMs at scale with better inference
efficiency (FLOP-controlled study to 8B/4T bytes); FastBLT claims **>50% lower estimated
memory-bandwidth cost** on generation, 2.7×-class parallel byte emission via BLT-D.

**Size and shape.** **513 LOC**, 2 files (`lib.rs` 352 + `hash.rs` 161). Tensor ops only,
no cubecl. API: `byte_patch`, `bltd_loss`, `verify_draft`, `unmask_confidence`,
`unmask_entropy_bounded`, `HashEmbeddings::{new, forward, forward_precomputed}`,
`byte_group_hash_ids`. Features: `std` only. `forward_precomputed` is the interesting one —
hash once in the dataloader, skip the device→host roundtrip per forward.

**State.** Builds, 8 tests pass (ndarray). `hash_matches_reference_formula` checks against
Meta's `bytelatent` polynomial rolling hash. Fidelity note in-source and **honest**: the
port weights the *newest* byte by `p⁰` where Meta's Horner fold weights the *oldest* by the
highest power — "no downstream behavior depends on matching Meta's table indices", so
pretrained hash tables would not port. Not a defect, a stated limit.

**Wiring.** **Zero.** 0 references in `crates/dormouse-*`.

**Verdict: IMPLEMENTED-UNUSED** — and its `HashEmbeddings` half is the direct ancestor of
`burn-engram`'s `MultiHashEmbedding` (see overlap map).

---

## 3. `burn-ptrn` — Probabilistic Tiny Recursive Model

**What.** Test-time compute scaling for a **looped, parameter-shared** recursion, with no
retraining: inject Gaussian noise into the latent at every recursion step, run K parallel
rollouts, score each with a learned scalar **Q-head**, take the best. Plus the joint training
objective `L = CE(f_O(y), y_true) + BCE(q̂, 1[ŷ = y_true])`.
arXiv **2605.19943** — **VERIFIED**, *"Probabilistic Tiny Recursive Model"*. Headline:
Sudoku-Extreme **87.4% → 98.75%**, Pencil Puzzle Bench **62.6% → 91.2%** (vs 55.1% for
frontier LLMs) at **7M params and <0.0001× the cost**; "without requiring retraining or
task-specific augmentations". Noise-only beat Q-guided Langevin in the paper's ablations.

**Size and shape.** **399 LOC**, 3 files. Tensor ops only. API: `QHead::{new, logit, prob,
q_loss}`, `add_recurrent_noise`, `best_of_k`, `best_of_k_sampled`, `correctness_target`,
`PtrnConfig` (σ=0.5, K=16, τ=1.0). Features: `std` only. Pure `Tensor`, no host branching in
the rollout path (the doc comment commits to that explicitly and the code honours it).

**State.** Builds, 12 tests pass. The strongest test set in the slice relative to size —
`best_of_k_sampled_tau_zero_is_argmax` asserts bitwise equality with the deterministic path,
and `best_of_k_sampled_stays_in_range` guards the index range.

**Wiring.** **Zero.**

**Verdict: POST-TRAINING-ONLY** (self-evolve / best-of-N stage, `POST_TRAINING.md` §C
"Task-time: K=8 noisy rollouts (PonderNet Q-head) + pick by Q"). Two caveats for whoever
picks it up: it is an **inference-time** recipe (it cannot be trained into the base model),
and its `QHead` has **nowhere to plug in** — `loop_block.rs:118` says the depth is set per
step and *"there is no learned halting here"*, and line 475 records that *"the PonderNet
variant lost its A/B"*. `POST_TRAINING.md`'s "PonderNet Q-head" refers to a component that
was cut, so this is a **new head**, not a reuse.

---

## 4. `burn-ttt` — Test-Time Training

**What.** **It is not an implementation of TTT.** The whole crate is **125 LOC in one
file** exporting three functions: `token_log_probs`, `ttt_ce_loss` (masked next-token CE,
mean over each row's valid length), `ttt_latent_loss` (masked MSE on the next latent). There
is no hidden-state-as-model, no inner-loop optimiser, no meta-learned initialisation, no
learned state of any kind.
arXiv **2512.23675** (TTT-E2E) and **2407.04620** (TTT) — **BOTH VERIFIED**, titles *"End-to-
End Test-Time Training for Long Context"* and *"Learning to (Learn at Test Time)"*.
Headline: 3B / 164B tokens scales with context like full attention where Mamba-2 and Gated
DeltaNet do not, at **2.7× faster than full attention at 128K**, constant latency.

**Size and shape.** **125 LOC**, 1 file, 3 functions. Tensor ops only. Features: `std` only.

**State.** Builds, 4 tests pass, and the tests are honest about what they cover — they check
`ce_loss_zero_when_perfect` and `ce_loss_masked_positions_ignored` against hand-computed
values. The doc comment claims it is "matched to the official implementation
`test-time-training/e2e` (`ttt/model/loss.py`)" — that claim is about the two loss functions
only, and it is not machine-verified.

**Wiring.** **Zero.**

**Verdict: IMPLEMENTED-UNUSED.** Flagging the sharper truth: this crate is *named* for a
memory mechanism and *contains* only the loss. It cannot be a rival to the Engram because it
has no state. If anyone cites "we have TTT" off this crate's name, that is wrong.

---

## 5. `burn-jepa` — JEPA / data2vec 2.0 + KoLeo

**What.** An EMA-teacher (`EmaTarget`, momentum, ramped) produces target latents under
`no_grad`; a `JepaPredictor` head predicts the latents of **masked** positions only, trained
with masked L1. Plus two anti-collapse regularisers: KoLeo (uniformity, soft-min surrogate
for the row-min) and LeJEPA SIGReg (isotropic Gaussian).
arXiv **2212.07525** (data2vec 2.0), **2511.08544** (LeJEPA), **2304.07193** (KoLeo, DINOv2)
— **ALL THREE VERIFIED**. Headline: LeJEPA reaches 79% ImageNet-1k linear eval with a frozen
ViT-H/14, single trade-off hyperparameter, linear time/memory, no stop-grad and no
teacher-student.

**Size and shape.** **443 LOC**, 5 files. Tensor ops only. API: `EmaTarget::{new, update,
val}`, `JepaPredictor::{new, forward}`, `mask_indices`, `jepa_l1_loss`, `koleo_loss`,
`lejepa_loss`, `JepaConfig`. Features: `std` only.

**State.** Builds, 14 tests pass. This is the only crate in the slice whose tests encode
**hard-won backend landmines as invariants**, and that is its real value:
- `koleo_subsamples` exists because *"burn-cuda 0.21 has no GPU sort, so `topk_with_indices`
  falls back to a host read that fails inside autodiff (and its panic poisons the CUDA
  context → illegal address)"* — hence stride sampling, not norm-topk.
- The soft-min replaces `ArgMin`/`ArgMax` reduce because *"cubek-reduce 0.2's ArgMin/ArgMax
  triggers a latent OOB on burn-cuda 0.21 inside autodiff, flaky, in-training only —
  verified: arg-reduce variants crash, soft-min is clean"*.
- `koleo_duplicate_rows_finite` / `koleo_all_same_rows_finite` pin the clamp-before-sqrt
  (rounding pushes `2-2·dot` negative → `sqrt(NaN)` → NaN grads).
- `jepa_l1_averages_over_mask_elements_not_positions` pins a real old bug: the old formula
  divided by masked *positions*, inflating the loss by `d`.

**Wiring.** 4 references, **3 symbols called**: `mask_indices` (`aux.rs:63`), `jepa_l1_loss`
(`aux.rs:82`), `koleo_loss` (`aux.rs:85`), plus the `JepaPredictor` module
(`aux.rs:34,42,53,71`). On by default at weight 0.05.

**Verdict: WIRED.** `lejepa_loss` is exported and unused — the crate is 1/3 unwired.

---

## 6. `burn-dspark` — DSpark speculative decoding

**What.** A semi-autoregressive **draft head** that corrects the frozen backbone's logits
into the next-K tokens. Three head variants (Vanilla low-rank Markov `B = W1[x]W2`; Gated;
`RNNHead` GRU-like over `[s; W1[x]; h]`), an `AcceptRatePredictor` confidence head
`sigmoid(wᵀ[h_k; W1[x_{k-1}]])`, sampling, and the full training objective
`L = 0.1·CE + 0.9·TV(L1 to frozen probs) + 1.0·BCE(c, 1 - 0.5·‖p_d - p_t‖₁)` with position
decay `w_k = exp(-k/γ)`, γ=4.0, plus Sequential Temperature Scaling.
arXiv **2607.05147** — **VERIFIED**, *"DSpark: Confidence-Scheduled Speculative Decoding
with Semi-Autoregressive Generation"*.

**Size and shape.** **865 LOC**, 3 files (`lib.rs` 408, `markov.rs` 372, `sampling.rs` 85) —
the largest crate in the slice. Tensor ops only, **no cubecl**. API: `RNNHead`, `GatedMarkovHead`,
`VanillaMarkov`, `AcceptRatePredictor::{new, with_markov, logit, prob}`, `dspark_loss`,
`position_weights`, `accept_rate_target`, `accept_rate_loss`, `sts_calibrate`, `sample_tokens`,
`sample_residual`, `greedy_draft`. Features: `std` + **`training`** — and `training` gates
`dspark_loss`, `accept_rate_loss`, `sts_calibrate` **and the `Int` import**, so without it the
loss is not even in the binary. `dormouse-core` enables it explicitly.

**State.** Builds, **16 tests pass** with `--features training` (14 without — the 2 gated
ones). Notable: `accept_rate_predictor_markov_conditioned` is a regression test for a real
bug (the conditioned variant sized its `Linear` for `input_dim` while `logit()` concatenates
`input_dim + rank`, making the documented path uncallable), and
`accept_rate_predictor_rejects_missing_markov_embeddings` asserts a panic on the mismatched
pair rather than letting a silent shape bug through.

**Wiring.** 4 references, **3 symbols called**: `RNNHead` (`aux.rs:35,43,144`),
`AcceptRatePredictor` (`aux.rs:36,44,145`), `dspark_loss` (`aux.rs:220`). On by default,
weight 0.1, K=4.

**Verdict: WIRED.**

---

## 7. `burn-mtp` — Multi-Token Prediction

**What.** `k` linear-probe heads (LayerNorm → Linear(D,D)) predicting tokens `t+i+1` from the
shared backbone hidden state, all sharing **one** unembedding matrix `f_u`; loss is the summed
per-head CE, equal weights (Gloeckle) or uniform per-depth λ (DeepSeek-V3).
arXiv **2404.19737** — **VERIFIED**, *"Better & Faster Large Language Models via
Multi-token Prediction"*.

**Size and shape.** **196 LOC**, 1 file. Tensor ops only. API: `MtpHeads::{new, with_lambda,
forward, loss}`. Features: `std` only.

**State.** Builds, 4 tests pass. `ce_equals_manual_gather` recomputes the loss from the raw
log-probs by hand — a genuine internal consistency check.

**Wiring.** **Zero.**

**Verdict: SUPERSEDED — replaced by `burn-dspark`.** `AGENTS.md` states it outright
("DSpark ... used instead of MTP"), and the code agrees: same job (k auxiliary future-token
heads on a shared backbone, extra CE on shifted targets), different plumbing. `dspark`'s
`RNNHead` adds the recurrent state across the draft block and the **TV distillation against
the frozen target logits** (`0.9` weight vs CE's `0.1`), which is the part that actually
teaches the draft head; `MtpHeads` is plain parallel heads on frozen hidden states. `MtpHeads`
is 196 LOC with no caller. Keep it only as the citation of record; the code is dead weight.

---

## 8. `burn-antihall` — anti-hallucination

**What.** `HallSuppressor`: per-FFN-neuron **sigmoid gates**, optionally conditioned on a
domain embedding and on the context, trained end to end — the *learned* variant of H-Neurons'
hard column scaling. `intervene(x, &[idx], scale)`: the paper's actual hard ablation, scaling
`down_proj` columns. `HallDetector`: a probe head producing a hallucination logit.
arXiv **2512.01797** — **VERIFIED**, *"H-Neurons: On the Existence, Impact, and Origin of
Hallucination-Associated Neurons in LLMs"*. Headline: **<0.1% of neurons** reliably predict
hallucination occurrence, generalise across scenarios, and are **causally** linked to
over-compliance — and they already exist in the *base* model. Supporting citations
2604.19765 / 2607.00158 / 2512.18623 / a Nature DOI are in the doc comment; I verified the
primary only.

**Size and shape.** **302 LOC**, 3 files. Tensor ops only. API: `HallSuppressor::{new,
with_domain_proj, with_context_proj, forward, forward_domain, forward_adaptive}`,
`HallDetector::{new, logit, prob}`, `intervene`. Features: `std` only.

**State.** Builds, 10 tests pass. `domain_forward_batched` is a regression test (the gate used
to reshape `[B,H]` to `[1,1,H]`, which panics for `B > 1`); `init_near_one` pins the gates
starting near-saturated so the suppressor is a no-op at init.

**Wiring.** **Zero.**

**Verdict: POST-TRAINING-ONLY** (post-training SFT / RLVR reliability stage). It is an
inference-time and/or fine-tune-time edit to a *trained* model — scaling specific FFN
columns of a base model you already have. It is not a base-model component. Note the paper's
own finding cuts against adding it to pretraining: the H-Neurons are already present in the
pretrained base, so there is nothing for a from-scratch run to learn early.

---

## 9. `burn-situ` — SiTU-GLU (the only fused CUDA kernel in this slice)

**What.** A bounded GLU: `beta·tanh(g/beta) ⊙ sigmoid(g) ⊙ beta₂·tanh(u/beta₂)` — two
soft-caps plus a Swish factor on the **raw** pre-activation (the uncapped sigmoid is the
paper's stated design goal, so the negative tail vanishes).
arXiv **2607.24653** (Kimi K3 tech report) — **VERIFIED**. Headline: 2.8T MoE / 104B active;
SiTU-GLU benchmarked there for better deep-model stability than SwiGLU and for MXFP4
low-bit quantisation. No numeric delta is claimed in the abstract.

**Size and shape.** **675 LOC**, 2 files. **Real fused CUDA:** `situ_glu_kernel` and
`situ_glu_backward_kernel` (`#[cube(launch_unchecked)]`), one cube per row
(`CUBE_POS_X`), comptime loop over hidden in 256-column slabs, `threads = 256`,
`f32` only. Plus a real `Backward` impl (`ad::SituGlu`, `N_PARENTS = 1`) with an **exact**
elementwise backward that recomputes gate/up from the checkpointed input. API: `situ_glu`,
`softcap`, plus the two fused entry points. Features: `std`, **`cuda`**, **`autodiff`**.

**State.** `cargo check -p burn-situ --features cuda,autodiff` → **clean** (3.48 s, artifacts
cached). ndarray: 4 tests pass; 6 more are `#[cfg(feature = "cuda")]` and did not run.
**Fused-path dispatch traced under our trainer's exact conditions** — it *would* be taken:
```rust
if std::any::TypeId::of::<Inner>() == std::any::TypeId::of::<burn_cubecl::CubeBackend>()
```
with `Inner = burn_cuda::Cuda`. I checked this rather than assuming it, because
`burn_cubecl::Cube` is `#[cfg(feature = "fusion")] Fusion<CubeBackend>` and a fusion wrap
would have made the `TypeId` compare **false** and silently dropped us to the tensor path.
It is safe here: `dormouse-*` pulls `burn-cuda` with `default-features = false,
features = ["std", "autotune"]`, `burn-cuda/fusion` is therefore off, and
`cargo tree -p dormouse-core` shows **zero** `burn-fusion` edges. The second guard
`hidden % 8 == 0 || hidden == 4` (cubecl's CUDA codegen corrupts coalesced stores on row
strides that are not 32-byte multiples) also passes for **every** preset:
`d_ffn` = 2048 / 2048 / 2048 / 2816 / 4096 / 5632 / 11264, all `% 8 == 0`. And
`situ_glu` wants a 2-D `Tensor`, which is what the loop already has (`[b*t, 2f]`).

**Wiring.** **Zero.** The loop uses plain `activation::silu` (`loop_block.rs:321`).

**Verdict: IMPLEMENTED-UNUSED.** The highest-value unadopted thing in the slice: a working,
autodiff-registered, shape-compatible fused activation sitting one import away. Caveat before
anyone wires it: the expert FFNs are **TSCT low-rank `LinearLike`**, not plain `Linear`, so
`gate_up`'s output layout is not a plain `[N, 2·hidden]` concat and the shape guards and
memory layout need re-checking against `param::LinearLike` first.

---

## 10. `burn-parcae` — stable looping via spectral retention

**What.** Recasts a loop as a nonlinear time-variant dynamical system
`h_{t+1} = Ā·h_t + B̄·e + R̄(h_t, e)` and constrains retention by discretising a
continuous-time negative-diagonal parameterisation: `A := diag(-exp(a))`,
`Ā := exp(Δ·A)` with `Δ` read as `|δ| + 1e-8` at use. So `Δ·A` has strictly negative entries,
`exp(Δ·A)` has **all eigenvalues in [0,1)** — contraction for any loop count `T`, for **any**
optimiser, by construction, with no clipping and no post-hoc normalisation. `B̄ = Δ·B`.
arXiv **2604.12946** — **VERIFIED**, *"Parcae: Scaling Laws For Stable Looped Language
Models"* (Prairie, Novack, Berg-Kirkpatrick, Fu).

**Size and shape.** **317 LOC**, 1 file. Tensor ops only. API: `SpectralRetention::{new,
from_config, forward, retention()}`, `retention_matrix`, `SpectralRetentionConfig{full_b}`.
Both the paper's full `d×d` `B` and a diagonal `B` (d params) are implemented. Features:
`std` only.

**State.** Builds, 8 tests pass. The "any optimiser / any `T`" claim is the **paper's** and
is documented as such in the crate; it is not measured here and I did not try to falsify it.
Note the `|δ| + 1e-8` trick is what makes it optimiser-agnostic: SGD can push a raw `delta`
param negative, and the abs at read time keeps the guarantee with gradient flow intact.

**Wiring.** **Zero.**

**Verdict: IMPLEMENTED-UNUSED.** Flagging it because it is the cheapest untried answer to the
open problem in `AGENTS.md`: the recurrent-depth NaN episodes ("the KDA recurrence goes NaN,
bisect showed `--no-kda` stays clean"). Parcae's guarantee is structural and would not depend
on tracing the deeper overflow path. It is a *rival mechanism* to our ReZero/GR residual, not
a duplicate of one.

---

## 11. `burn-byteflow` — ByteFlow Net

**What.** A complete competing tokenizer-free byte LM, not a mechanism: exact lossy coding-rate
chunking `R_ε(h) = ½ log det(I + (d/ε²)·H Hᵀ)` plus an Appendix-B L2 streaming approximation
and top-K boundary selection with a forced BOS, and a five-stage network (local encoder,
SWA global transformer with Canon layers, multi-linear upsampling with a large residual,
symmetric decoder). Depends on `burn-swiglu`.
arXiv **2603.03583** — **VERIFIED**, *"ByteFlow: Language Modeling through Adaptive Byte
Compression without a Tokenizer"* (Deng et al., ICLR 2026).

**Size and shape.** **1121 LOC**, 3 files — the largest crate in the slice. Tensor ops only,
**no cubecl**. API: `coding_rate_exact`, `marginal_gains_exact`, `marginal_gains_l2`,
`select_positions`, `RateMode`, `ByteFlowConfig`, `ByteFlowNet::{init, forward,
encode_chunks, decode_chunks}`, `CanonLayer`, `FlowAttention`, `FlowBlock`, `RopeTable`,
`VOCAB`. Features: `std` only.

**State.** Builds, 14 tests pass. No parity harness.

**Wiring.** **Zero.**

**Verdict: IMPLEMENTED-UNUSED.** The only crate in the slice that is a whole alternative
*model* rather than a swappable part — it is the reference design against which the
next-byte-per-token choice is a decision, not a component. It also re-implements
attention/RoPE/GLU that `burn-attnres` / `burn-rope` / `burn-swiglu` already provide, so
wiring it would mean three more duplicate kernels.

---

## 12. `burn-diffusionblocks` — block-wise training as diffusion

**What.** Reinterprets a residual net as a VE diffusion model: corrupt the target with
`z_σ = y + σ·ε`, train an x0-predictor with the EDM weighted L2, `log σ ~ N(-1.2, 1.2²)`,
`w(σ) = (σ² + σ_data²)/(σ·σ_data)²`. Networks are partitioned into blocks trained
**independently** on equal-mass noise ranges, so peak memory is `/B` vs BPTT; at inference the
blocks compose via an Euler step of the probability-flow ODE. Recurrent-depth models skip
partitioning entirely (per the paper's Appendix E.5).
arXiv **2506.14202** — **VERIFIED**, *"DiffusionBlocks: Block-wise Neural Network Training
via Diffusion Interpretation"* (Sakana AI, ICLR 2026). Headline: matches end-to-end
training across vision, diffusion, autoregressive, **recurrent-depth** and masked-diffusion
architectures, with memory reduced proportionally to the block count.

**Size and shape.** **793 LOC**, 5 files. Tensor ops only. API: `NoiseSchedule::{partition,
sample_sigma, sample_sigma_full, weight, euler_step}`, `BlockPartition`, `add_noise`,
`denoising_loss`, `blockwise_step`, `normal_cdf`, `inv_normal_cdf`. Features: `std` only.
Depends on `fastrand` + `libm` (the only crate in the slice with non-burn runtime deps).

**State.** Builds, 10 tests pass, including its own `normal_cdf` / `inv_normal_cdf` (the
schedule is only as good as the inverse-CDF, so testing it is not optional).

**Wiring.** **Zero.**

**Verdict: IMPLEMENTED-UNUSED.** Worth naming that the doc comment carries a dedicated
**"Recurrent-depth usage sketch"** section that is a line-for-line description of dormouse:
the whole looped network is ONE denoiser, sample σ from the full log-normal, corrupt the
clean state once, a single forward through the loop, no per-iteration blocks, no per-iteration
targets, no BPTT. This crate was written *at* our architecture and never adopted. It is an
alternative to the CE objective, not to the Engram.

---

## 13. `burn-eggroll` — low-rank ES

**What.** Rank-1 Gaussian perturbations `E = (1/√r)·A·Bᵀ`, antithetic evaluation of
`M ± σE`, and `M ← M + (α/√r)·Σᵢ Eᵢ·sign(s⁺ᵢ - s⁻ᵢ)` — never materialising `E` (the sum is
`(f·A)ᵀ·B` as one batched matmul). For non-differentiable decisions such as top-k routing.
arXiv **2511.16652** — **VERIFIED**, *"Evolution Strategies at the Hyperscale"*. Headline
from the crate: rank 1 is as effective as full-rank noise at **~100×** fewer random numbers
and storage.

**Size and shape.** **327 LOC**, 3 files. Tensor ops only. API: `EggrollConfig`,
`sample_a`, `sample_b`, `eggroll_mutate`, `perturb`, `antithetic_sign`, `update`,
`update_batched`. Features: `std` only.

**State.** Builds, 7 tests pass — and the good ones are *deterministic*:
`perturb_rank1_variance_bounded` uses A=1, B=1 (every entry exactly σ, std exactly 0) and
A=1, B=alternating (std exactly σ), so it cannot flake. `update_batched_equals_loop` pins
the batched path against the per-sample loop. Defect found while reading: **`sample_a` and
`sample_b` take a `_seed` parameter that is ignored** — so `burn-es`'s claim that burn-eggroll
has a "seed-regeneration API" is false, and EGGROLL rollouts here are not reproducible.

**Wiring.** **Zero.**

**Verdict: POST-TRAINING-ONLY** (exploration stage; `POST_TRAINING.md` §"EGGROLL —
exploration for controllers", "PPO exploits policy, EGGROLL explores routers"). It is a
black-box optimiser and cannot be in a backprop base model.

---

## 14. `burn-es` — Evolution Strategies (OpenAI)

**What.** The classical ES loop: Gaussian mutation, ternary/BitNet-b1.58 quantised mutation
with a 0.7·absmean dead zone, antithetic sampling (~2× variance reduction for free), rank
utilities, and the estimator `grad = (1/(N·σ))·Σᵢ uᵢ·εᵢ` with the paper-exact
`u_i = max(0, ln(N/2+1) - ln k_i)` centred to sum zero.
arXiv **1703.03864** — **VERIFIED**, *"Evolution Strategies as a Scalable Alternative to
Reinforcement Learning"* (Salimans et al.). Headline: scales to 1000+ parallel CPU workers
communicating only scalars; humanoid walking in 10 min, competitive Atari in 1 h.

**Size and shape.** **345 LOC**, 1 file. Tensor ops only. API: `gaussian_mutate`, `ternarize`,
`ternary_mutate`, `antithetic_pair`, `openai_rank_utilities`, `openai_rank_utilities_paper`,
`rank_fitness`, `es_gradient`, `eggroll_mutate`. Features: `std` only.

**State.** Builds, 9 tests pass.

**Wiring.** **Zero.**

**Verdict: POST-TRAINING-ONLY** (RLVR / ES stage). It also carries a self-declared duplicate:
its own `eggroll_mutate` uses the **unnormalised** `σ·A·Bᵀ` (entry std ≈ `σ·√rank`) where
burn-eggroll implements the paper's `(σ/√r)·A·Bᵀ` (entry std ≈ `σ`). The crate's own doc
comment says "prefer burn-eggroll for real ES loops; the function here stays for quick
experiments". **Delete it** — it is a wrong-scaled copy of a sibling crate with a test.

---

## 15. `burn-fused` — the meta-crate

**What.** 44 lines, 29 `pub use` re-exports, zero logic. Exists so one dependency pulls the
whole workspace behind feature flags: `std` fans out to all 29 members' `std`; `cuda` fans out
to the members that define one. Members are non-optional dependencies, so
`use burn_fused::burn_engram` always compiles and the feature matrix only controls what each
member enables. `default-members = ["burn-fused"]` in the root `Cargo.toml` keeps the default
build to just this crate.

**Does dormouse use it? No.** Zero references in any `crates/dormouse-*/Cargo.toml`;
`burn-fused` has **0 entries** in the root `Cargo.lock`. Every wired member is depended on by
explicit `path = "../../vendor/burn-fused/crates/..."`. The meta-crate is pure dead weight
*for this repo* — the integration test of a facade nobody calls.

**Does it build? No.**

```
error[E0308]: mismatched types
   --> crates/burn-muon-plus/src/lib.rs:298:22
    |  .mul(g_active.reshape([1, 1]));
    |  expected `D`, found `2`
error[E0308]: mismatched types
   --> crates/burn-muon-plus/src/lib.rs:364:38
error: could not compile `burn-muon-plus` (lib) due to 2 previous errors
```
A rank-generic `Tensor<D>` multiplied by a rank-2 / rank-1 `reshape` — a burn
0.22.0-pre.4 const-generic API break, **not** one of the four sm_120 landmines. Because
`burn-fused` depends on all 29 members unconditionally, one broken member takes the facade
down with it. `burn-muon-plus` is git-clean, so this is committed breakage, not a
concurrent edit (it is in another slice's territory — I am read-only and did not touch it).

**The feature matrix is also stale.** 13 crates define a `cuda` feature; the meta's `cuda`
list names 12. Missing: **`burn-mor/cuda`** — which is the crate that carries the repaired
top-k→gather primitive (ADR-0015), the one member whose CUDA path dormouse actually depends on.

**Verdict: BROKEN** (does not compile). Recommendation, if ADR-0017 keeps it as
`dormouse-fused`: (1) fix `burn-muon-plus` — 2 lines, it is a rank-generic reshape that must
preserve `D`; (2) add `burn-mor/cuda`; (3) decide whether it earns its keep at all. It is
the only crate in the library with no caller, and the only one whose build status is coupled
to all 28 others. A facade that nothing imports and that fails to build is the definition of
a crate to delete — but that is a call for the ADR, not for me, and it is cheap to keep once
(1) and (2) are done.

---

## Overlap map — the memory family

**Genuine rivals: none of them.** Only `burn-engram` holds a learned state keyed by context
in the sense that matters for the loss — a parameter table indexed by a context key. The
other three touch that phrase and nothing else.

**Same idea, different plumbing: `burn-engram` vs `burn-fastblt`.** Both are
"context n-gram → hash → learned table row → embedding". They differ in two axes.
*Addressing*: fastblt uses a polynomial rolling hash `Σ x[i]·prime^i` mod `max_hash` (BLT
2024, pre-Engram); engram uses splitmix64 odd multipliers XOR-mixed over the causal window
and reduced mod a **distinct prime per (ngram, head) slot** — the paper's design, where the
prime moduli are what make the addressing non-learned yet collision-structured. *Fusion*:
fastblt **sums** table embeddings into the byte embedding; engram **concatenates** them and
routes them through a per-head `sigmoid(√|s|·sign s)` gate plus a value projection. Engram
strictly supersedes fastblt on both axes and adds a mechanism (gated multi-head read) that
has no fastblt analogue. **Fastblt would not have served better.** Note also that dormouse
uses *neither* hasher: the trainer's FNV-1a-32 `& mask` is a third addressing scheme again
(`dormouse-data/src/lib.rs:407`, `loop_block.rs:24-34`).

**Complement, not rival: `burn-ptrn`.** No state at all — it perturbs the latent with noise
and *selects among* trajectories with a scalar Q-head at inference. It competes for
inference-time compute budget with the loop (PonderNet-style extra iterations), never for
parameters. Orthogonal to the Engram; the one place they could interact is the Q-head, and
there is no Q-head to reuse (the learned halting was cut).

**Neither rival nor complement: `burn-ttt`.** The concept *is* a rival — "compress the
context into weights by learning on it at test time" is the other way to get a learned state
out of context, and it is strictly more expressive than a hash table (it can represent
contexts no table was sized for) while being vastly more expensive and non-deterministic.
But **this crate implements only the two loss functions**; 125 LOC, no inner loop, no state.
It is a name without a mechanism. If we ever want real TTT, this crate is a 2-function
starting point, not an implementation.

**So: would one of these have served better than the Engram at 500K rows with a lambda
floor?** On the evidence in this slice, no. The Engram is the only one of the four that
actually stores anything, and `burn-fastblt`'s version of the same thing is strictly weaker.
The honest counter-argument the slice supports is *within* the Engram, not across crates: the
2.3x-SRAM "monopolised the loss" result was measured with addressing that is **neither** the
crate's paper-faithful prime-per-slot hasher (dead in `hasher.rs`) **nor** BLT's — it is
FNV-1a-truncated-to-32 masked to a power of two. If the diagnosis is "the table learned to
memorise training bytes through an under-powered random address", then the fix is a
better-conditioned address function, and the crate already ships one that was never wired.
That is a strictly cheaper experiment than re-enabling TTT, and it is the one this slice
recommends trying first.

---

## Cross-cutting notes

- **Zero bit-for-bit reference harnesses in this slice.** The closest are hand-transcribed
  formula assertions (`burn-engram`'s gate and hasher, `burn-fastblt`'s rolling hash,
  `burn-mtp`'s CE). The fork's `benches/` is a **perf**-regression harness
  (`check.py` vs committed baselines, ±20% tolerance), not a correctness parity harness, and
  its members cover the fused crates (`attnres`, `bitnet`, `gdn2`, `kda`, `mhc`, `mor`,
  `rmsnorm`, `rope`) — none of mine.
- **The most valuable thing in this slice is not a kernel, it is a comment library.**
  `burn-jepa/src/losses.rs` documents three reproducible backend landmines in the exact
  failure mode we keep re-discovering (no GPU sort → `topk_with_indices` host read poisons
  the CUDA context inside autodiff; `ArgMin`/`ArgMax` reduce → latent OOB, flaky,
  in-training only; clamp-before-sqrt or duplicate rows → `sqrt(NaN)` → NaN grads). That is
  worth preserving verbatim through the rename.
- **Every one of these 14 crates is `std`-only and CUDA-agnostic except `burn-situ`.** The
  "fused kernel ecosystem" branding in the meta-crate's description ("100+ technologies,
  faster than naive chains, less memory, any hardware") is accurate for the library as a
  whole and inaccurate for this half of it. `burn-engram` is the only crate here with a
  comment explaining that its own missing kernel is a deliberate deferral rather than an
  oversight, and that explanation is good enough that it should survive the rename.
- **`burn-mtp` and `burn-es::eggroll_mutate` are the two deletion candidates**: 196 LOC with
  no caller that a live sibling crate replaced, and a wrong-scaled copy of a sibling crate
  that has its own test.
