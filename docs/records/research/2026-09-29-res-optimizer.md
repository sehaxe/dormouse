# Is Muon+ still the right optimizer? (fetched 2026-09-29)

Scope: one question only — has anything newer beaten **Muon+** (`arXiv 2602.21545`) **on Muon+'s own
terms**, i.e. in a head-to-head where Muon+ is the baseline. Nothing else is evaluated.

---

## VERDICT (first)

**Yes — keep Muon+, and treat it as the *floor*, not the ceiling.** As of 2026-09-29 **zero papers
compare against Muon+ as a baseline**; every paper that cites it either mentions it in related work
or is a survey. So there is no refutation of Muon+, but also **no independent replication of it**,
and the one live community instrument that would have caught a successor — the modded-nanogpt
*Optimization Benchmark* — has **never run Muon+ at all** (0 mentions in the leaderboard), so it
neither confirms nor denies it.

The uncomfortable part is second-hand and worth stating plainly: in that same benchmark, the
family Muon+ belongs to (post-polar row/column normalization: NorMuon, Muown, Contra-Muon,
Soft-Muon, Aurora) was submitted repeatedly, **scored worse than plain tuned Muon, and was
explicitly deleted from the record chain** to reach the current world record. The winning stack is
SOAP-Muon + a per-row update floor + weight decay + LR schedule. That is an argument that the
*specific* normalization step may be worth less in practice than its paper's 0.37–2.02 PPL deltas
suggest — and it is a different claim from "Muon+ has been beaten", which nobody has made.

Confidence: **high** that nothing has beaten Muon+ head-to-head (checked the full text of every
plausible candidate); **low** on whether Muon+ is actually *better* than plain Muon at our scale,
because it has never been A/B'd against Muon here and the closest live evidence is secondhand.

---

## Provenance

All fetched **2026-09-29** via the arXiv API (`https://export.arxiv.org/api/query`, `https://` form —
the bare `export.arxiv.org` host 301-redirects), arXiv abstract pages, arXiv HTML full text
(`https://arxiv.org/html/<id>`), the Semantic Scholar Graph API, and two GitHub raw files.

| source | identifier | version / date |
|---|---|---|
| Muon+ | `arXiv:2602.21545` | **v3, 2026-05-14** (v1 2026-02-25); Zhang, Zhao, Liu, Wang, Su, Tan, Zhang — UC Santa Barbara |
| Semantic Scholar citations of Muon+ | paperId `699a5966c588da002d5f99b28608d9b7db18494a` | 45 refs, **4 citations**, retrieved 2026-09-29 |
| modded-nanogpt | `github.com/KellerJordan/modded-nanogpt` | `master`, README + `records/track_3_optimization/README.md`, fetched 2026-09-29 |
| Awesome-Optimizers | `github.com/JiwenJ/Awesome-Optimizers` | `main` README.md, fetched 2026-09-29 |

**Muon+'s own evidence** (`2602.21545v3` §3): LLaMA 58M–7B and GPT 124M–6.6B, FineWeb, batch 512,
**H100/A100, bf16 mixed precision**, compute-optimal and overtraining (T2P ≈ 200). Validation PPL vs
Muon: GPT-Small −2.02, GPT-Base −1.72, GPT-Large −0.91; LLaMA-350M −0.61, LLaMA-1B −0.37,
GPT-Huge 6.6B −1.21. Wall-clock to Muon's target loss: +22.4% to **+37.1%**. It also states it beats
its own in-paper competitors **NorMuon and AdaMuon** (Table 9) and adds **no optimizer state**.

---

## The four papers that cite Muon+ — and what each actually does with it

Semantic Scholar reports exactly 4 citations. All four were read in full text.

| paper | how it uses Muon+ | head-to-head vs Muon+? |
|---|---|---|
| `arXiv:2609.21102` Spectral Deflation (math.OC/math.NA, 2026-09-17) | related-work sentence: "Muon+ adds row- or column-wise normalization to correct norm imbalance in the polar update" | **No** |
| `arXiv:2606.27216` Hierarchical Muon (2026-06-25) | related-work sentence classifying Muon+ among normalization variants | **No** — baseline is full-matrix Muon |
| `arXiv:2608.20818` Scaling Muon for Diffusion Transformers (2026-08-21) | "Periodic Row-wise Muon" cites it (0 literal `Muon+` hits in full text) | **No** — baselines are AdamW and vanilla Muon, on 256×H100 |
| `arXiv:2603.15914` The Agentic Researcher | survey/guide | **No** |

### Two name collisions that would have produced a false claim

Both surfaced in search and both are **not** Muon+:
- `arXiv:2605.07815` OrScale — "improves **Muon+Moonlight**" = Muon with Moonlight (Kimi) scaling.
- `arXiv:2609.34681` SOLAR — "**Muon+Cosine** 14.36 → 13.79" = plain Muon with a cosine schedule;
  its optimizers are AdamW, Muon, Prodigy, Schedule-Free AdamW (Appendix B.5).

Also checked and **not** about Muon+: `arXiv:2609.33194`, `2609.35297`, `2609.33047`, `2609.34915`,
`2609.33152` — 0 literal `Muon+` mentions in every full text.

---

## Candidate table

Only papers that claim to improve on the Muon family. **"Baseline" column is the whole answer.**

| # | paper (arXiv) | date | claim | hardware | scale | baseline compared | **vs Muon or vs Muon+?** |
|---|---|---|---|---|---|---|---|
| 1 | **MeqMuon** `2609.35701` | 2026-09-28 | row- *and* column-equilibrating norm chosen per matrix per step by CV; also drops AdamW 2nd moments for embed/head | RTX A6000, PyTorch 2.6 | LLaMA 60M–350M, SmolLM2 135M–360M, Qwen2 0.5B | AdamW, SCALE, Muon, **NorMuon** | **Muon + NorMuon. Does not cite Muon+ at all.** Margin over NorMuon 0.05–0.13 PPL on a ~15–19 PPL base (≈0.4–0.6%), single-run tables, 1 day old, unreplicated |
| 2 | **SAMuon** `2608.25990` | 2026-08-26 | 13.3–24.0% fewer tokens to same val loss; bulk/head spectral allocation | not stated in abstract (modded-nanogpt harness) | 124M–1B | tuned AdamW, Muon (**Scion** impl.) | **Muon** |
| 3 | **QSD** `2609.07597` (v2 2026-09-26) | 2026-09-07 | −0.0158 loss vs Muon at +6% step cost; 8.49% wall-clock; GPT-350M 14.3% / 7.4% | **4× RTX 5000 Ada**, bf16 | GPT-124M, GPT-350M | Muon, **NorMuon**, **Newton-Muon** | **Muon / NorMuon.** §5.3 verbatim: "we compare QSD with the other Muon variants, NorMuon and Newton-Muon". Muon+ appears only in §5.1 as a setup citation |
| 4 | **Aurora** `2606.27715` | 2026-06-26 | SOTA on modded-nanogpt **optimizers track**; +9.1 MMLU at 1.1B; dead-neuron fix | modded-nanogpt (8×H100 class) | 340M, 1.1B | Muon, **NorMuon**, Contra-Muon | **Muon / NorMuon.** 0 literal `Muon+` hits; its entire row-norm section is about **NorMuon** and argues post-hoc row-norm *moves geometry away from the polar factor* |
| 5 | **SOAP, Muon, and Beyond** `2607.20548` (NVIDIA) | 2026-07-13 | Muon & SOAP beat AdamW at 100M-token batch; per-step QR fixes SOAP spikes | multi-billion, Megatron-LM | multi-B params, 1–3T tokens | AdamW, Muon, SOAP, KL-SOAP | **Muon** |
| 6 | **Dion3** `2608.11612` (Microsoft) | 2026-08-12 | matches/beats Muon loss at **up to 6× faster optimizer step**; Gram-NS + CuteDSL + megabatching | distributed (sharded) | not stated in abstract | Muon, **Dion** | **Muon** |
| 7 | **Musec** `2609.11655` | 2026-09-10 | spectral **clipping** instead of flattening; stable where Muon variants diverge | not stated | 491M, 613M, 1.63B NanoGPT | Muon, MuonClip, other Muon variants | **Muon** (and it concedes best-tuned Muon variants still trail Adam/AdamW) |
| 8 | **Mousse** `2603.09697` | 2026-03-10 | ~12% fewer training steps vs Muon | not stated | 160M–800M | Muon | **Muon** |
| 9 | **FISMO** `2601.21750` | 2026-01-29 | Kronecker-Fisher trust region; O(1/√T) | not stated | image cls + LM | Adam-family, Muon | **Muon** (predates Muon+ v1) |
| 10 | **MuonEq** `2603.28254` (incl. Ruijie Zhang, a Muon+ author) | 2026-03-30 | pre-orthogonalization row/col equilibration beats Muon | not stated | LLaMA2 130M/350M/1B on C4 | Muon | **Muon** |
| 11 | **RMNP** `2603.20527` | 2026-03-20 | replaces NS5 with row ℓ2-norm: O(mn·min(m,n)) → **O(mn)**, "comparable" perf | not stated | LLM pretraining | Muon | **Muon** |
| 12 | **TrasMuon** `2602.13498` | 2026-02-13 | trust-region clipping of update energy; no-warmup stability | not stated | vision + LM | baselines (Muon-family) | **Muon** |
| 13 | **Nora** `2605.03769` | 2026-05-05 | O(mn), stabilizes weight norms + angular velocity | not stated | "preliminary" | Muon, RMNP | **Muon** |
| 14 | **Reassessing Muon for Matrix Factorization** `2607.13246` | 2026-07-14 | **negative result**: Muon does not consistently beat AdamW on low-rank matrix factorization; advantages are hyperparameter-sensitive | CPU | synthetic | tuned adaptive | n/a — but the most important skeptical result in the set |
| 15 | **Muon Sublates the Edge of Stability** `2609.34915` | 2026-09-28 | analysis of stochastic Muon (T1/T2) | not stated | 130M, 1B | — | analysis, not a competitor |

### Rejected as non-evidence

`2608.20818` (256×H100 DiT — different modality, unattainable hardware), `2608.11612` (Dion3's win is
a *systems* result on sharded training), `2609.09676` (Muon-C, conv kernels), `2609.02734` /
`2609.12123` / `2607.17620` (LoRA-only), `2609.06073` (federated), `2609.24678` (continual learning),
`2609.23055` (diffusion), `2606.30461` / `2608.03941` (SSM/Mamba), `2606.27216` (tiling, no quality
claim), `2608.16760` / `2608.28557` (surveys).

---

## The strongest evidence is not a paper

**modded-nanogpt Optimization Benchmark** (`records/track_3_optimization/README.md`, fetched
2026-09-29) — fixed architecture, fixed data, fixed batch size, target 3.28 val loss on FineWeb,
multi-seed with recorded p-values. This is the closest thing to a controlled optimizer shootout.

- **`Muon+` appears 0 times** in the leaderboard, the current-record description, or the rules.
- The current world record is **#46 at 2690 steps** vs the tuned Muon baseline **#36 at 3250 steps**
  — a **20.8% step reduction** (the README's own figure).
- #46's active techniques: **SOAP-Muon** on all hidden matrices (row/column gradient covariance
  preconditioning *before* orthogonalization, `precondition_frequency=1`) + **RowUpdateFloor** +
  radial brake / radius rescale + post-pin **Cautious Weight Decay** + **EMA-Nesterov** wrapper +
  **PowerCool** LR + **Tail-EMA** readout.
- NorMuon appears **10×** in the leaderboard (#8, #10, #31, #26 SinkSOAP…) and is the only
  row-normalization variant that ever held a record slot.
- **#44 explicitly removes the normalization family**: "remove neutral geometry modules including
  (Circuit,Contra)Muon and Aurora". #46's description states it does **not** use NorMuon-lite row/col
  variance preconditioning, Aurora, Contra-Muon or Soft-Muon.
- The **main** speedrun track's active technique list also names **NorMuon**, not Muon+, plus an
  unnamed **"ANVIL optimizer: twin-rail (fast + slow) momentum, re-derived orthogonalization maps,
  cautious decay gated on the slow rail"** — **UNRESOLVED**: no arXiv paper located for ANVIL.

Caveat, stated rather than hidden: these are leaderboard *entries read from a README*, not runs I
reproduced, and every record is a tuned stack on a ~2B-token nanoGPT budget.

---

## What each candidate would cost us to adopt

Our envelope, from `AGENTS.md` §2.1/§3.1: **one RTX 5060 Ti, 16 GB, fp32** (bf16 *matmul* is
unusable on this backend — the LLVM dialect has no bf16 type, so every bf16 run is slower than fp32),
and the optimizer is **43–47 ms of a ~245 ms warm step (~19%)**, launch-bound, on a card that is
**idle 87% of the time**. The binding constraint is therefore **extra kernel launches and extra
optimizer state**, not FLOPs.

- **Keep Muon+ (do nothing).** Zero cost. It is one normalization over an already-materialized
  update and adds no state. This is the status quo and the recommendation.
- **MeqMuon** — cheapest real change: a per-matrix CV comparison choosing row vs column norm, plus
  replacing AdamW on 2D embed/head with two-sided normalized momentum. It *drops* state (21.6% less
  on Qwen2-0.5B), which helps at 16 GB, and it needs no new kernels. But it also removes the
  `rs_scale` term so **LR transfer breaks** and our embed/head AdamW routing changes; and its margin
  over NorMuon is ~0.5% with no seeds reported. Not justified until it is replicated.
- **SAMuon** — needs a low-rank randomized SVD (or power iteration) to fit a *static* spectral prior.
  That is new kernels and new launches on a launch-bound step. **Rejected on cost**, independent of
  whether the claim is real.
- **QSD** — inversion-free K-FAC factor estimation + online Frank-Wolfe inner solves. Its own paper
  prices this at **+6% step time** on a 4-GPU node; for us that is several extra passes over every
  2D parameter per step. **Rejected on cost** at a 9.2M-param model on 16 GB.
- **Dion3** — the *right direction* (6× faster optimizer step, and opt is our 19%), but the result is
  a distributed-sharding result and the kernels are CuteDSL. The one transferable piece is
  **selecting a fraction of momentum rows to orthogonalize**; on its own that is an unverified guess.
- **SOAP-Muon** — the empirically strongest thing going (it is the record), but it is a **stack**,
  needs two extra per-matrix moment buffers, a trust gate and an alignment computation per matrix.
  On 16 GB with a 9.2M model the state is affordable; the **launch count is not**, and it was tuned
  on a fixed 2B-token nanoGPT budget whose LR schedule does not transfer to a 46 GB corpus run.
- **RMNP** — the one that would actually buy back our 43 ms: it replaces NS5 with a row ℓ2-norm,
  `O(mn·min(m,n)) → O(mn)`, i.e. essentially free. But it claims *comparable* quality, **not better**,
  and it is a large semantic change to a component we have already debugged. Its own argument
  (orthogonalization ≈ row-ℓ2 asymptotically) is exactly the premise Muon+ disputes.
- **Musec / Nora / Mousse / TrasMuon / MuonEq / Dion** — each adds state, a new kernel family, or a
  new hyperparameter for a **stability or theory** claim, none of which is a quality claim against
  Muon+. We have no measured stability problem attributable to the optimizer (§3.3's NaN episodes
  are attributed to overfit-corpus KDA recurrence and are explicitly an overfit artifact).
  **Not adoption candidates.**

---

## SPECULATION (my inference, not sourced)

1. Muon+ has had **4 citations in 7 months and zero head-to-head tests**, while its closest rival
   NorMuon (2025-10) has a live speedrun record and a 2026-06 NVIDIA-style "use this" citation. My
   reading is that the field absorbed the *idea* through NorMuon and did not adopt the *paper*. That
   is an inference from citation counts and leaderboard occupancy, not a measurement.
2. I expect Muon+'s delta to **shrink at our scale**. Its own Table shows the gain falling
   monotonically with size (GPT-Small −2.02 → GPT-Large −0.91; LLaMA-60M −0.50 → LLaMA-1B −0.37),
   and our model is ~9.2M params, below every scale in its compute-optimal tables. Extrapolating a
   shrinking trend below the range where it was measured is unsupported.
3. The gap between "post-polar row norm helps" (Muon+, consistent across 8 model sizes, 5 seeds on
   two) and "post-polar row norm is not in the winning record stack" (track 3) is most plausibly
   explained by **budget and schedule**: a 2B-token fixed-budget nanoGPT run is a different regime
   from a 46 GB corpus at extended token-to-parameter ratio. **Untested** — nothing in either source
   tests the other regime.

## Open questions I could not close

- Is there a post-Muon+ paper that compares to it and is **not indexed by Semantic Scholar** (which
  reported only 4 citations, almost certainly an undercount for a 7-month-old cs.LG paper)? I
  searched by `abs:`, `all:` and category but arXiv's API does **not** index full text, so a paper
  that mentions Muon+ only in an experiments table would be invisible to me. **This is the main
  residual risk of a "nothing found" result.**
- No arXiv paper found for **ANVIL**, which is in the modded-nanogpt main-track record.
- MeqMuon (1 day old) has no independent replication and no error bars.
