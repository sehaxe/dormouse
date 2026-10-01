# Is there a better way to do trainable low-rank factorization than TSCT?

**Fetch date: 2026-09-29.** Literature-only pass. No GPU, no cargo, no build, no test.
Code read read-only: `crates/dormouse-core/src/param.rs`,
`crates/dormouse-core/src/loop_block.rs`,
`vendor/dormouse-fused/crates/dormouse-spectral/src/lib.rs`.

**Scope note.** `docs/papers/tsct.md` (same tree) already establishes the lineage and
the code deltas. This file answers a different, narrower question: *as of 2026-09-29,
does the literature have a better way to do trainable low-rank factorization of a dense
layer, where the bar is removing the per-step retraction?* Where the two overlap on
`2604.00733` I repeat the citation rather than cross-reference, so this file stands alone.

---

## 1. Verdict

**No — nothing in the literature removes the per-step retraction from a trainable
low-rank weight factorization, and our cost is not an artefact of a bad implementation.**
The mechanism we invented is the mechanism the field converged on independently: a
three-factor `W = U·diag(s)·Vᵀ` with `U, V` orthonormal columns, re-projected onto the
Stiefel manifold every step. Two papers publish that exact structure — **StelLA**
(NeurIPS 2025 Spotlight, arXiv:2510.01938) for adapters and **SCT**
(arXiv:2604.00733) for permanent pretrained weights — and SCT independently measures the
same thing we do: retraction at **40-50% of total step time**, against our 22%. The
constraint *can* be enforced by construction (Cayley, matrix-exponential, Householder —
PyTorch ships all three), but every such parametrization spends the same work in the
**forward** instead of the optimizer step, and the two papers that actually publish the
cost comparison both chose the per-step retraction over it anyway.

**The honest headline is uncomfortable and worth stating plainly: we did not invent
this.** Our name implies novelty we do not have.

---

## 2. Provenance

All URLs fetched 2026-09-29. "read" = full text or full HTML body, not the abstract only.

| # | id / URL | title | venue | read |
|---|---|---|---|---|
| P1 | [arXiv:2510.01938](https://arxiv.org/abs/2510.01938) v2 (rev 2026-04-02) | StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold — Li, Sajadmanesh, Li, Lyu (Sony AI) | NeurIPS 2025 **Spotlight** | **full HTML incl. App. D, E** |
| P2 | [arXiv:2604.00733](https://arxiv.org/abs/2604.00733) v2 (rev 2026-04-05) | Spectral Compact Training: Pre-Training LLMs via Permanent Truncated SVD and Stiefel QR Retraction — Kohlberger (EctoSpace) | arXiv preprint, **8pp, patent pending** | **full HTML** |
| P3 | [arXiv:2602.12429](https://arxiv.org/abs/2602.12429) v2 (rev 2026-07-15) | Stabilizing Native Low-Rank LLM Pretraining (Spectron) — Janson, Oyallon, Belilovsky | **ICML 2026** | **full HTML** |
| P4 | [arXiv:2002.01113](https://arxiv.org/abs/2002.01113) | Efficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform — Li, Fuxin, Todorovic | ICLR 2020 | abstract + body via PDF extract |
| P5 | [arXiv:2004.08675](https://arxiv.org/abs/2004.08675) v3 | CWY Parametrization: A Solution for Parallelized Optimization of Orthogonal **and Stiefel** Matrices — Likhosherstov, Davis, Choromanski, Weller | AISTATS 2021 (PMLR 130) | abstract + body via PMLR PDF extract |
| P6 | [arXiv:2102.07432](https://arxiv.org/abs/2102.07432) v2 | Fast and accurate optimization on the orthogonal manifold **without retraction** (landing algorithm) — Ablin, Peyré | AISTATS 2022 (PMLR 151) | abstract + body via PMLR PDF extract |
| P7 | [arXiv:2303.16510](https://arxiv.org/abs/2303.16510) v2 | Infeasible Deterministic, Stochastic, and Variance-Reduction Algorithms for Optimization under Orthogonality Constraints — Ablin, Vary, Gao, Absil | JMLR 25 (2024) | abstract + intro via JMLR PDF extract |
| P8 | [arXiv:1612.00188](https://arxiv.org/abs/1612.00188) v5 | Efficient Orthogonal Parametrisation of RNNs Using Householder Reflections — Mhammedi, Hellicar, Rahman, Bailey | ICML 2017 (PMLR 70) | abstract + body |
| P9 | [arXiv:1901.08428](https://arxiv.org/pdf/1901.08428) | Cheap Orthogonal Constraints in Neural Networks: A Simple Parametrization of the Orthogonal and Unitary Group — Lezcano-Casado, Martínez-Rubio | ICML 2019 (PMLR 97) | abstract + body |
| P10 | [docs.pytorch.org/2.2 `parametrizations.orthogonal`](https://docs.pytorch.org/docs/2.2/generated/torch.nn.utils.parametrizations.orthogonal.html) | shipped library API: `matrix_exp` / `cayley` / `householder` | PyTorch 2.2 docs | **full page** |
| P11 | [arXiv:2404.02948](https://arxiv.org/abs/2404.02948) v4 | PiSSA — Meng, Wang, Zhang | NeurIPS 2024 | abstract + method |
| P12 | [arXiv:2406.01775](https://arxiv.org/pdf/2406.01775) | OLoRA: Orthonormal Low-Rank Adaptation — Büyükakyüz | arXiv preprint | abstract + algorithm |
| P13 | [arXiv:2508.17901](https://arxiv.org/pdf/2508.17901) | Riemannian Optimization for LoRA on the Stiefel Manifold — Park et al. | Findings EMNLP 2025 | abstract + body |
| P14 | [aclanthology 2025.findings-emnlp.1143](https://aclanthology.org/2025.findings-emnlp.1143/) | same as P13 (publisher version) | Findings EMNLP 2025 | metadata |
| P15 | [arXiv:2004.09031](https://arxiv.org/abs/2004.09031) v1 | Learning Low-rank DNNs via Singular Vector Orthogonality Regularization — Yang et al. | CVPRW 2020 | abstract |
| P16 | [arxiv.org/pdf/2403.11418](https://arxiv.org/pdf/2403.11418) *(via P1 ref [38])* | Riemannian metric on the low-rank quotient — cited by P1, **not independently fetched** | — | **UNRESOLVED** |
| P17 | NeurIPS 2024 [Group and Shuffle](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7f0f24deb34c21ee590d8cece365710b-Abstract-Conference.html) — Gorbunov et al. | structured orthogonal parametrization | NeurIPS 2024 | abstract only |
| P18 | NeurIPS 2024 [HRA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html) — Yuan, Liu, Xu | Householder Reflection Adaptation | NeurIPS 2024 Spotlight | abstract only |

**Rejected during the search, with reasons.** LoRA itself (Hu et al. 2021) — freezes the
dense matrix, which is the opposite of our setting (SCT P2 §2 makes the same
distinction). AdaLoRA, GeoLoRA, TriLoRA, MoSRA — three-factor LoRA with *soft*
orthogonality or none; P1 §2 records that none of them maintain orthonormality during
training. qGOFT (PMLR 235) and OFT — orthogonal *finetuning* with a frozen dense model.
BitNet b1.58 / a4.8 / v2 — ternary weights with a per-column scale and **no
low-rank factorization and no manifold constraint at all**; relevant only as the
alternative that removes the constraint entirely by dropping the factorization, which
changes the model. DLRT / TDLRT / ELRT / Zangrando et al. — low-rank *training* from
scratch by gradient flow, no per-step retraction, but they are ODE/continuous-time
integrators and none reports a wall-clock A/B against a retracted factorization. DCP,
Grassmann-net (arXiv:1611.05742), ReFT (CVPR 2025) — orthogonal parametrization for
*data* subspaces or reconstructions, not for the weight being trained.

---

## 3. Candidate table

**Per-step constraint enforcement** is the column that decides this question for us.
`YES` = a projection runs every optimizer step. `BY CONSTRUCTION` = the constraint holds
for *every* value of the unconstrained parameters, at the cost of work in the forward.

| # | method | what it changes vs TSCT | per-step enforcement | scale / evidence | cost on 1×16 GB fp32 |
|---|---|---|---|---|---|
| **P1** | **StelLA** | *Same structure*: `W + (α/r)USVᵀ`, `U∈St(r,m)`, `V∈St(r,n)`, `S∈R^{r×r}` | **YES** — polar retraction, Alg. 1 line 9 | LLaMA2-7B/3-8B, ViT-B/L, SD1.5/2.0; +1.3 to +2.33 pts over best baseline | **Measured, P1 App. D: batched polar retraction 1.6-5.8 ms** for 320 matrices (H100); end-to-end **"only 15% slower than vanilla LoRA"** (5.2 h vs 4.5 h) |
| **P2** | **SCT** | *Same structure*, permanent (no frozen dense), **QR** retraction not NS | **YES** | SmolLM2-135M/1.7B, 2000 steps, A100; 8pp preprint, patent pending | **Measured: retraction = 3.02 s of a 3.41 s step (89%, M4 Pro) and 2.58 s of 6.28 s (41%, Steam Deck) at 70B.** Names Cayley (P4) as a cheaper alternative |
| **P3** | **Spectron** | **Drops the Stiefel constraint entirely.** Orthogonalizes the *momentum* (Muon-style NS), then rescales by `ρ = η/(‖A‖₂+‖B‖₂+1)` | **NO — by construction of the update, not the weight** | **ICML 2026.** FineWeb pretraining 94M/297M/454M; beats naive AdamW *and* self-guided at every scale | **"sub-1%" overhead**, stated as 6·k_ns·n·m² + 2·m·n FLOPs; vs 25% for self-guided |
| **P4** | Cayley retraction | NS → Cayley for the retraction | **YES** | ICLR 2020, CNN + RNN | Wen & Yin's trick applies when `2p ≪ n`: invert a **2p×2p** matrix, not n×n. At k=64 that is a 128×128 inverse per factor |
| **P5** | CWY / **T-CWY** | Replace the retraction with a **parametrization** = product of Householder reflections in compact-WY form; **explicitly covers St(n,m)** | **BY CONSTRUCTION** | AISTATS 2021; NMT + video prediction; convergence proved | Claims 20× over sequential Householder, **1-3 orders of magnitude over matrix-exp and Cayley**; parallel complexity O(log LN) |
| **P6/P7** | **Landing** | **No retraction at all.** Potential-energy ODE attracted to the manifold | **NO — but not exactly feasible** | AISTATS 2022 (O(p)); **Stiefel extension in JMLR 2024 (P7)** | Matmuls only. **Caveat that decides it for us: the iterate is *not* on the manifold.** "Infeasible" is in the JMLR title |
| **P8/P9/P10** | Householder / matrix-exp / Cayley **parametrizations** | Constraint encoded in the parameterization; nothing to project | **BY CONSTRUCTION** | ICML 2017, ICML 2019; **shipped in PyTorch as one API** | P10: `householder` is the **default for non-square** weights precisely because "matrix_exp/cayley … are slower to compute for very thin or very wide matrices" |
| **P11/P12** | PiSSA / OLoRA | SVD/QR **initialization**, then free Euclidean training | **NO — init only** | NeurIPS 2024 (PiSSA: 184M-70B, 12 models) | Zero per-step cost. Rank-*stability* is not guaranteed and neither paper claims it |
| **P13/P14** | Stiefel-LoRA | Stiefel on LoRA's `B` only | **YES** | Findings EMNLP 2025 | Not separately costed |
| **P15** | SVD training | Orthogonality **regularization** on singular vectors, not projection | **NO — soft penalty** | CVPRW 2020 | ~zero compute; no guarantee |

### What the two cost measurements actually say

Both are external corroborations of our own number, and they bracket it:

- **P2 (SCT):** retraction is **40-50% of total step time** at 70B on consumer hardware.
  Its own §5 lists "QR retraction cost" as a named limitation and points at Cayley (P4).
- **P1 (StelLA) App. D:** retraction is "the dominant cost"; batched SVD cuts it
  **14.4-24.9×**; end to end it is **15% over LoRA**, and the authors call the overhead
  "small". P1 §6 Limitations.
- **Ours:** 52.8 ms of ~245 ms warm = **22%** (AGENTS.md §3.1, measured 2026-09-29).

We sit **between** the two. That is the expected place to be: P1's overhead is on a
frozen 7B model where the retraction is a smaller fraction of a huge step; P2's is on
consumer CPUs where nothing is fused. Our 22% is what the same operation costs when the
rest of the step is *also* launch-bound.

---

## 4. The best candidate for US specifically

**Nothing removes the retraction. The best available change is to stop paying for 112
host synchronisations — and that is not a literature finding, it is already in our tree.**

### 4.1 The measured cost is mostly an implementation artefact, and we already wrote the fix

Our `polar_orthogonalize` (`dormouse-spectral/src/lib.rs:168-217`) is called per factor from
`SpectralLinear::retract` (`:613-621`). It contains, per factor: 5 power iterations,
**each with one `into_scalar::<f32>()`** (`lib.rs:186`), plus two more at `:197` and
`:198` — **7 host syncs per factor**. 16 factors → **112 per step**. Every one of them is
a CPU-GPU round trip inside a workload that is launch-bound anyway (AGENTS.md §3.1: mean
GPU utilisation 13.3%).

`dormouse-spectral/src/lib.rs:226-292` already contains `polar_orthogonalize_batched` and
`retract_batched`, whose own doc comment states the fix in as many words: *"the norm is a
`[B,1,1]` tensor broadcast across the batch instead of an extracted scalar. … so there is
not a single host↔device sync (the scalar path syncs 7× per factor)."* They are tested
(`retract_batched_identity_with_per_factor_path`, `:1327`; `retract_batched_deterministic`,
`:1371`) and **not called from `retract_tsct`** (`loop_block.rs:170-176`, which loops
`f.gate_up.retract(iters)` per factor).

**Migration cost: call `retract_batched` from `retract_tsct`.** It groups factors by exact
shape so mixed shapes never pad each other, and it is the identical math (`:256-262` vs
`:205-211` — same `(a,b,c)`, same 1.05 sigma safety factor). P1 App. D independently
measures the same effect from the other direction: batching their retraction gave
**14.4-24.9×**. **SPECULATION: this should recover most of the 52.8 ms. I have not run
it, and another agent is editing `dormouse-spectral` right now, so this is a claim to
measure, not a claim to trust.** Note also that `--retract-every 1000` already gives
`retr=0.0` at 188 ms (AGENTS.md §3.1), which bounds the win: the total addressable is
52.8 ms, not the whole step.

### 4.2 If the constraint must go rather than get cheaper: Spectron (P3)

**This is the only candidate that removes the per-step retraction and keeps low-rank
training working — and it is the right paper to read before deciding TSCT's constraint is
load-bearing.** It is ICML 2026, it is *pretraining from scratch* (our case, not LoRA's
case), and it reports beating both naive AdamW and self-guided training at 94M/297M/454M
with sub-1% overhead.

**What it gives up, and this is the reason it is not a drop-in:** Spectron does not
maintain `UᵀU = I`. It orthogonalizes the momentum and bounds the *update's* spectral
norm. Our constraint exists for a different reason — `param.rs:174-175` states it as
*"Without periodic retract the factors drift and the quantized forward degrades"* — i.e.
our retraction exists to keep the **ternary/2-bit quantized forward** well-conditioned,
not to prevent spectral-norm blow-up. P3 fixes a failure mode we have not been shown to
have. **Adopting it is a bet that the quantized forward does not need exact
orthonormality, and BitNet b1.58 (ternary weights, per-column absmean scale, no
orthonormality constraint, no factorization) is weak evidence that the bet is winnable —
but BitNet is a different architecture and I am not transferring its result.**

**Cost: real.** It replaces AdamW on the factors with a Muon-style optimizer (we already
run Muon+, `dormouse-muon-plus`, so the machinery exists), and it invalidates every TSCT
number. It is an A/B, not a refactor.

### 4.3 If the constraint must stay and the parametrization must change

**Cayley (P4) is the only retraction swap with a published cost argument.** Wen & Yin's
formulation inverts a **2p×2p** matrix when `2p ≪ n` — at k=64, a 128×128 inverse per
factor, against our current 5-iteration power loop plus 7 syncs. P2 names Cayley
specifically as the cheaper alternative to its QR. **But note the trap:** P4 is a
*retraction* (still per-step, `YES`), while P10's `cayley` is a *parametrization*
(`BY CONSTRUCTION`, cost moved into the forward). P2's own §5 and P1's §5.5 both chose
the retraction — P1 measured polar vs exponential map at **86.72 vs 86.76** accuracy, i.e.
indistinguishable, and took polar for cost. **At our scale neither of the
by-construction routes is obviously cheaper than a batched NS retraction, and both are
strictly larger changes than §4.1.**

---

## 5. Is TSCT genuinely novel?

**No. Plainly, and on two independent counts.**

1. **The parameterization is not ours.** `LinearLike` factors every 2D weight as
   `U·diag(s)·Vᵀ` with `U ∈ ℝ^{in×k}`, `V ∈ ℝ^{out×k}` orthonormal columns and `s ∈ ℝ^k`
   (`param.rs:15-17`, `dormouse-spectral/src/lib.rs:343-348`, forward at `:338-341`,
   `:488`). That is a truncated SVD parameterization with the singular values split
   evenly across the two factors — the definition of the thing.
   - **SCT (P2)** publishes `W = U·diag(s)·Vᵀ`, U and V orthonormal, for **permanent
     pretrained weights**, with a per-step Stiefel retraction. This is TSCT with QR
     instead of Newton-Schulz. It is not in our model path, but it *is* in our tree:
     `param.rs:1` names `dormouse_sct::SpectralLinear` as the origin, and
     `dormouse-spectral/Cargo.toml:33` still declares `dormouse-sct` with zero source references.
   - **StelLA (P1)** publishes the same three factors with the same two Stiefel
     constraints and a per-step **polar** retraction — our exact retraction, in a NeurIPS
     2025 Spotlight.

2. **The retraction is not ours.** NS polar retraction onto the Stiefel manifold is
   standard Riemannian machinery (Absil–Mahony–Sepulchre 2008) and is what P1 uses.

**What may still be ours** (and per `docs/papers/tsct.md` is the real delta): the
**quantizer family on the factors** — ternary/2-bit/N:M/fp8/fp4 with STE through the
orthonormal masters — and the fused/batched GPU retraction. **I found no paper that
combines a low-bit quantized forward with a Stiefel-constrained factorization.** BitNet
b1.58 quantizes weights with a per-column scale but does not factorize; SCT and StelLA
factorize with a manifold constraint but keep fp32 factors. That combination is
**UNRESOLVED** — I did not search the low-bit-QA-training literature exhaustively, and a
negative here is weaker than the negatives above.

**Recommendation on the name.** "TSCT" implies novelty that does not exist and will mislead
the next reader the way a retracted citation would. Something like
`quantized Stiefel low-rank (qSLR)` — the Stiefel part cited to P1/P2, the quantization
part ours — would be defensible. This is a naming decision, not mine to make.

---

## 6. VERIFIED (sourced)

- TSCT's structure = the structure published by P1 (arXiv:2510.01938, NeurIPS 2025
  Spotlight, three-factor `USVᵀ`, U/V on Stiefel, polar retraction per step) and by P2
  (arXiv:2604.00733, permanent truncated SVD, Stiefel QR retraction per step). Both read
  in full.
- **P2 measures retraction at 40-50% of total step time** on consumer hardware at 70B
  scale, and names Cayley (P4) as a cheaper alternative (§5, "QR retraction cost").
- **P1 App. D measures the retraction as the dominant cost**, with batched SVD giving
  **14.4-24.9×** speedup, and **15% end-to-end over vanilla LoRA** (4.5 h → 5.2 h,
  LLaMA3-8B, H100).
- **P3 (arXiv:2602.12429, ICML 2026) removes the per-step retraction entirely** —
  orthogonalized momentum + spectral renormalization of the update, no Stiefel constraint
  on the factors — and reports **sub-1% overhead** vs 25% for self-guided training, with
  better perplexity than both baselines at 94M/297M/454M.
- **The constraint CAN be enforced by construction**: Householder/CWY (P5, and the
  ICML 2017 predecessor P8), matrix exponential and Cayley (P9, P10). **P5 explicitly
  covers the Stiefel manifold** (T-CWY, Thm. 3). PyTorch ships all three as one API
  (P10).
- P10 states `householder` is the **default for non-square** weights because
  `"matrix_exp"/"cayley" … are slower to compute for very thin or very wide matrices`.
- **P4's Wen & Yin trick** reduces the Cayley retraction to inverting a **2p×2p** matrix
  when `2p ≪ n`.
- **P1's polar-vs-exponential-map ablation is a tie**: 86.72 vs 86.76 average accuracy,
  polar chosen for cost (§5.5, Table 6).
- **Our code does 7 host syncs per factor**: 5 in the power-iteration loop
  (`dormouse-spectral/src/lib.rs:186`) + 2 more at `:197-198`; 16 factors → 112.
- **Our tree already contains the sync-free fix**:
  `polar_orthogonalize_batched` (`lib.rs:226`) and `retract_batched` (`:275`) are
  implemented, tested (`:1327`, `:1371`), carry the identical NS coefficients, and are
  **not** called from `retract_tsct` (`loop_block.rs:170`).
- `--retract-every 1000` gives `retr=0.0` and a 188 ms step, bounding the addressable
  cost at 52.8 ms (AGENTS.md §3.1, measured 2026-09-29).

## 7. SPECULATION (my inference — not evidence)

- **SPECULATION: routing `retract_tsct` through the existing `retract_batched` would
  recover most of the 52.8 ms.** Basis: the batched path is the same math with the
  syncs removed, and P1 measured a 14.4-24.9× batching speedup on the equivalent
  operation. **Not measured here. No GPU, no build. Another agent is editing
  `dormouse-spectral` concurrently.** This is a hypothesis with a one-command test
  (`--timers` at step 50/100/150, same run shape as AGENTS.md §3.1), not a result.
- **SPECULATION: the constraint is probably not load-bearing at the level the 22% cost
  implies** — because P3 removes it and still trains well, and BitNet trains ternary
  weights with no constraint at all. **Counter-argument, and it is the strong one:** our
  constraint's stated purpose (`param.rs:174-175`) is quantized-forward conditioning,
  which P3 never had to solve. If we drop the retraction and the factor quant degrades,
  the 52.8 ms was buying something after all. **This is exactly the TSCT-vs-dense A/B that
  `docs/protocols/AB-PROTOCOL.md` has never run — the literature cannot settle it for us.**
- **SPECULATION: a Cayley retraction (P4) would beat our current NS at k=64**, replacing
  5 power iterations + 7 syncs with a 128×128 inverse per factor. Basis: P2 names it as
  the cheaper alternative; the 2p ≪ n condition holds comfortably at our shapes. **No
  measurement, and a burn/cubecl `linalg` inverse on this backend is not obviously
  available.**
- **SPECULATION: no paper combines low-bit factor quantization with a Stiefel-constrained
  factorization** — i.e. the quantized-forward half of TSCT may be the novel part. Basis:
  I checked the low-bit-QA and manifold-optimization sides and found nothing at the
  intersection. **This negative is weak; the search was not exhaustive on the
  quantization side.**

## 8. Open questions this search could not close

1. **Does anything enforce orthonormality *for free* on the parameter itself, at
   non-square aspect ratios, with fp32 on a 16 GB consumer card?** P5/P8/P10 say yes in
   principle and give complexity bounds; none reports a wall-clock A/B against a batched
   NS retraction at our shapes (k=64, n=768-2048). **UNRESOLVED.**
2. **Does the landing algorithm (P6/P7) work for a Stiefel factor in an SGD loop at our
   scale?** P7 extends it to Stiefel but calls the method "infeasible" — the iterate is not
   on the manifold, which is precisely what a quantized forward needs. **UNRESOLVED, and
   I read only the abstract and intro of P7.**
3. **Is there low-bit-QA work that constrains factors to be orthogonal?** **UNRESOLVED.**
4. **What is a TSCT A/B actually worth?** The literature has answered the
   retraction-cost question (§3) and the constraint-value question for *unquantized*
   factors (P1: Euclidean three-factor scores 84.4 vs StelLA 86.7 on LLaMA3-8B
   commonsense — the constraint is worth ~2.3 points *there*). **Neither transfers to a
   quantized forward.** Our own A/B remains the only instrument that would settle it,
   and it has not been run.
