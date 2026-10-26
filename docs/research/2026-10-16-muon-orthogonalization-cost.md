# Cutting the orthogonalization cost at b1 shapes — what is correct, what is cheap, what is neither

**Date:** 2026-10-16. **Read-only:** one file; nothing committed, no GPU work run.
**Ask:** cheapest *correct* way to cut Muon's Newton–Schulz cost at our shapes
(~`[2048,64]` / `[64,5461]`, rank 64), verified against primary sources.
**Method:** arXiv MCP, 10 query angles, every id resolved with `get_abstract`,
full text read for `2606.00371`, `2605.16341`, `2509.11983`, `2505.16932`,
`2608.11612`, `2602.02500`, `2512.04632`; the AGENTS §3.7 coefficient citation
**checked inside `2602.21545v3` and confirmed** (App. D.1 = Jordan `(3.4445,−4.7750,2.0315)`,
D.2 = You schedule, D.3 = PolarExpress ending `(1.875,−1.25,0.375)`).
Code read: `Dao-AILab/gram-newton-schulz`, `dengzhanwang/Low-rank-Muon`,
`microsoft/dion`. Repo read: `src/dormouse/train/optim.py` (whole),
`src/dormouse/model/tsct.py:1-30`, `src/dormouse/config.py:271-297`,
`docs/reviews/2026-10-16-b1-profile.md`, `docs/protocols/AB-PROTOCOL.md:25-60,200-240,370-400`.
Marks: **[A]** paper's words · **[code]** authors' code · **[ours]** repo fact · **[S]** SPECULATION.

## TL;DR

1. **Try `Gram Newton–Schulz` first (one function, ~10 lines).** Same polynomial,
   iterated on the `[64,64]` Gram instead of on `X` — **mathematically
   equivalent**, so no quality A/B is needed, only an equivalence gate
   [A `2608.11612` §4; code README]. Big rectangular matmuls: **10 → 4** per shape-group.
2. **Then, separately, `NS 5 → 3`.** Published ablations: the cost is small and
   **non-monotone** (Jordan at 4 steps beat 5) [A `2606.00371` §3.2]. It is the
   only lever here that changes the math, so it is the one needing the 3-seed arm.
3. **Orth-Dion fixes a bug we do not have.** Its `√r` penalty is a property of
   **column normalization** (Dion), not of our port; and its rate needs a spectral
   gap at rank `r` [A Assumption B′], degenerate when `r = min(m,n) = 64` [ours].
4. **Low-rank Muon cannot apply**: it needs `r ≪ ϱ = min(m,n)`, and our `ϱ` **is** 64.
5. NS is the biggest block of the optimizer but not the whole of it: `clip_grads`
   107 ms and ColRow/decay/AdamW elementwise ≈300 ms are untouched here [ours, profile §3].

## 1. Where our cost actually sits (why the shapes decide everything)

`zeropower_via_ns` transposes when `rows > cols` (`optim.py:58-64`), so **both**
orientations execute NS in the `[64,64]` Gram space:

| shape group | runs as | MACs per NS iteration | 5 iterations | matmul kernels/group |
|---|---|---|---|---|
| `U [2048,64]` | `[64,2048]` | ≈16.8 M | ≈84 M | 15 |
| `V [64,5461]` | `[64,5461]` | ≈44.7 M | ≈224 M | 15 |

So this is a **tiny-Gram, launch-count** problem, not a FLOP problem: 5 iterations
× 3 kernels per group, inside 246 `mm` + 366 `bmm` ≈ **535 ms CUDA / 530 ms
`opt.step`** [ours, profile §3]. A lever that does not cut rectangular matmuls or
launches cannot pay.

## 2. Decision table (ranked by "cheap AND correct")

| # | technique | what changes in our `optim.py` | expected ms at our profile | quality risk | source |
|---|---|---|---|---|---|
| **1** | **Gram Newton–Schulz** (1 restart) | rewrite the NS loop body (`optim.py:66-77`, `:85-96`): iterate `R,Q` in `[64,64]`, apply `X ← Q X` once at the end; transpose + coefficients unchanged | **−180…−300 [S]** (6 of 10 big matmuls gone; NS block 535 ms, big matmuls ~90 % of it) | **none on the math** — same polynomial, regrouped; only fp32 round-off order moves | [A] `2608.11612` §4 + [code] `gram-newton-schulz` README |
| **2** | **NS 5 → 3** (Jordan triple) | expose `ns_steps` in config, pass at `loop.py:321`; `Muon` already takes it (`optim.py:96,136`) | **−214 arithmetic**, **−130** as the profile itself prices it | small, published, non-monotone: Jordan-4 3.2809 < Jordan-5 3.2822, PE-5 3.2798, exact SVD 3.2809 [A] | [A] `2606.00371` §3.2/§3.3; [code] nanogpt NS comment |
| **3** | **PolarExpress truncated to 3 steps** | same loop, 3 different `(a,b,c)` per iteration | same as #2 (same structure) | **lower than #2**: PE beats Jordan at every LR (3.340 vs 3.398, GPT-2-L 1 B tokens) [A] | [A] `2505.16932`; coeffs in `2602.21545v3` App. D.3 (verified) |
| **4** | **NS chain in bf16** | `x.to(torch.float32)` (`:60`) → masters fp32, polynomial bf16, cast back | **−100…−200 [S]** (this box 44.06 vs 20.03 TFLOP/s; NS rides the fp32 `magma` lane) | low–med: bf16 NS is reference practice | [code] nanogpt `X = G.bfloat16()`; [A] `2505.16932` finite-precision §; AGENTS §2.1 |
| **5** | **Turbo-Muon AOL preconditioning** (drops 1 iteration) | replace `x/(x.norm()+1e-7)` with a column scaling from the first Gram's row sums | **−107 arithmetic**, ~0 overhead [A] | **unverified at our shape**: derived on square full-rank matrices; needs near-column-orthogonality | [A] `2512.04632` §4.1 |
| **6** | **Orth-Dion** | new branch: power iteration + 2 QR + persistent error-feedback buffer | unpriced [S] (≈4–5 kernels vs 15; +1 full-size buffer) | **high**: FSDP/column-normalized design; gap assumption undefined at `r = min(m,n)` | [A] `2605.16341` Alg. 1, Prop. 3.1, Lemma 4.1 |
| **7** | **Low-rank Muon** | sketch + QR, NS on the projection | ≈0 at our shapes | n/a — `r<64` would discard in-factor directions | [A] `2509.11983` Alg. 1/Thm 1; [code] 2-commit repo, stock NS inside |
| **8** | IFNSO / CANS / HiMuon / MuonBP / AMO | various | none beats #1 on kernels | weaker or wrong-shaped evidence | `2602.02500`, `2506.10935`, `2606.27216`, `2510.16981`, `2605.17806` |

## 3. The four questions

**1 · Is NS-3 good enough at our shapes?** No published step-count ablation uses a
64-row matrix — they are all square-ish GPT-2 Small / LLaMA / MoE [A `2606.00371`
Table 1, Fig. 3]. Published: 3.2811 (cubic-5), 3.2798 (PE-5), 3.2809 (Jordan-4),
3.2822 (Jordan-5), 3.2809 (**exact FP32 SVD polar**) — a <0.003 spread with a
non-monotone curve; the authors' own conclusion is that *"training quality is not
governed monotonically by polar-decomposition accuracy"* [A]. Our box has the one
shape-correct reading: at a 10:1 spread, 3 iterations sit **1.17e-15 from the
manifold** but **14.5 % from `polar(X)`** [ours, AGENTS §3.7]. So 3 steps is safe
for *orthonormality*, approximate for the *direction*. **[S] the risk is small; it
is not zero and it is not measured at 64 rows.**

**2 · Orth-Dion.** Exact algorithm (`2605.16341` Alg. 1): `M = G + R`;
`U = QR(M·V_{t−1})`; `W = MᵀU`; `V̄ = QR(W)` *(Orth-Dion)* vs `ColNorm(W)`
*(Dion)*; update `D̂ = U V̄ᵀ`; feedback `R ← β(M − UUᵀM)`; warm start `V ← V̄`.
The `√r` gain is about *scale inside the chosen subspace* — `ColNorm` lets
`‖D̂‖_op → √r`, QR forces ν = 1 [A Prop. 3.1, Lemma 4.1] — while both pick **the
same subspace** [A Lemma 4.3]. Saves vs 5-NS: one power iteration + 2 QR ≈
`O(mnr) + O(nr²)`, ~4–5 kernels vs 15, at the cost of a persistent residual
buffer. **Does the geometry survive on `[d,r]`, `r ≪ d`?** The partial-isometry
claim is shape-free (Lemma 4.1), the *rate* claim is not: it needs a gap at rank
`r` [Assumption B′], and our `r = min(m,n)` makes `σ_{r+1} ≡ 0`. And we never
column-normalize, so the mismatch it fixes is not our bug. **Does not transfer.**

**3 · Low-rank Muon.** Computes `Q = QR(MG)` (Gaussian `G`) and returns
`M_O = Q·msgn(QᵀM)`, `msgn` via NS on the small `[r,n]` [A Alg. 1]; the error
bound scales with `‖M − [M]_{r*}‖_F` [A Thm 1] — justified only when the gradient
spectrum decays, i.e. on big weight matrices [A §1]. Our `[64,5461]` factor is
*already* the low-rank object; its 64 singular values are the whole story, so the
`r ≪ ϱ` regime is unenterable. [code] repo: 2 commits, 0 stars, `train_gpt.py`
still stock quintic. **Does not transfer.**

**4 · Anything 2026 newer?** Strongest is **Gram Newton–Schulz** (= row #1):
`2608.11612` §4 + `Dao-AILab/gram-newton-schulz`, CuteDSL symmetric GEMMs, "up to
2× faster", "mathematically equivalent → no training accuracy tradeoff" [code],
with the largest FLOP win for *"highly asymmetric shapes"* [A] — us. **Caveat:**
the shipped kernels need Hopper/B200-class parts and `nvidia-cutlass-dsl`; our card
is sm_120, so the *algorithm* (10 lines of torch) transfers and the *kernel
package* may not. Second: **Cost-free spectral estimation** (`2609.33047`) makes NS
adaptive from Gram moments already computed — no extra matmuls, lower error at
fixed step count [A]: the insurance policy for row #2. Third: 2026 papers that only
reschedule the NS budget (AMO, MuonBP) or change the norm/parametrization
(AuON, LiMuon, LionMuon, LoRA-Muon, Intrinsic Muon) do not beat #1 on
kernels-per-step at our shape.

## 4. Pre-registered plan

**Stage 0 — offline gate, no training (~10 min, one process).** On saved momentum
buffers per shape group, run NS-5, NS-3, Gram-NS-5; report `‖outᵀout − I‖_F/rank`,
cosine to NS-5, wall-clock. **Gate: Gram-NS must equal NS-5 to fp32 round-off
(rel ≤ 1e-5)** — failing means the port is wrong, not the paper. NS-3's deficit is
recorded, not gated.

**Stage 1 — the arm (the only part needing training).** One factor: `ns_steps`.
**A0** = Jordan-5 (current), **A1** = Jordan-3. 3 seeds each, 2000 steps, pure CE
(`--jepa-weight 0 --dspark-weight 0`), **b8 × s512, eval_batches 20 ⇒ 81 920 B on
every eval line**, nothing else differs. Gram-NS must **not** ride along here: it is
exact, so it belongs in *both* arms after Stage 0 passes — one factor at a time.

**Verdict rule (pre-registered).** Keep NS-3 iff (a) its mean held-out BPB is not
worse than A0's by more than **A0's own 3-seed spread**, **and** (b) the step time
shows the saving (≥50 warm steps, quiet card, same config). **Prediction [S]: a tie
at 2k** — published step-count deltas are <0.003 and our seed spread is larger; a
tie is the designed outcome, and the decision then rests on (b) plus Stage 0's
deficit, never on the endpoint alone.

**Void checks:** identical byte count on all 6 eval lines; the 3 seeds must differ
in init (confirm the port's `--seed` reaches init — the Rust lane needed `4b42b6d`);
NaN-skipped steps = 0; Muon path ran; **`--retract-iters` untouched** (the TSCT
retraction is *also* an NS-3, burn-spectral's PolarExpress triple — changing it in
the same arm confounds two NS changes [ours, AGENTS §3.7]).

## 5. Rejected, one line each

- Orth-Dion — right fix for column-normalized distributed Dion, wrong assumption at `r = min(m,n)`.
- Low-rank Muon — `r ≪ ϱ` unenterable at `ϱ = 64`; unmaintained repo.
- HiMuon `2606.27216` — tiling keeps intra-tile spectra and drops inter-tile ones; a 64-row polar *is* the inter-tile structure.
- IFNSO `2602.02500` — ~14 polynomial terms ⇒ no kernel-count win over Gram-NS; evidence is MNIST on an RTX 4050.
- Dion2 `2512.16928` / column-sampling in `2608.11612` §6 — subsampling a 64-row matrix discards real directions.
- MuonBP `2510.16981` — a cadence change for distributed gathers, not a cost cut at one GPU.
- `2601.19156`, `2610.03306`, `2608.04607`, `2609.39595` — finite-step NS convergence theory: they justify the rates, they do not price our shapes.
- `2604.01472`, `2608.14492`, `2606.12921`, `2509.24320`, `2509.14562`, `2609.35297` — change the objective or the parametrization, not the NS chain's cost.

## 6. Sources

Papers (ids resolved 2026-10-16): `2608.11612` · `2606.00371` · `2605.16341` ·
`2509.11983` · `2505.16932` · `2602.21545` · `2512.04632` · `2602.02500` ·
`2609.33047` · `2606.27216` · `2512.16928` · `2510.16981` · `2605.17806` ·
`2601.19156` · `2610.03306` · `2609.39595` · `2608.04607`.
Code: `Dao-AILab/gram-newton-schulz` · `dengzhanwang/Low-rank-Muon` ·
`microsoft/dion` · `modded-nanogpt` NS routine.
Repo: `src/dormouse/train/optim.py:16-22,56-96,136-184` ·
`src/dormouse/train/loop.py:321` · `src/dormouse/model/tsct.py:1-30` ·
`docs/reviews/2026-10-16-b1-profile.md:19-43,68-84` ·
`docs/protocols/AB-PROTOCOL.md:25-60,200-240,370-400` · `AGENTS.md` §2.1 table, §3.7 ·
`docs/research/2026-10-16-sf-muon.md`.
Owed: a `docs-site/tools/manifest.mjs` line for this file (AGENTS §1.8).

## 7. Open questions

1. What is the singular-value spread of our TSCT-factor momentum buffers? Every
   step-count decision inherits it, and nobody has measured it here.
2. Does fp32 Gram-NS stay stable at `[64,64]` on our spectra without a restart
   (README recommends one at 5 steps)? Stage 0 answers it.
3. Does `2609.33047`'s moment-based adaptivity recover the NS-3 deficit at zero
   extra matmuls? The follow-up arm if Stage 1 ties.
