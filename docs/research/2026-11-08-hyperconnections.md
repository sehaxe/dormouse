# Hyperconnections at our scale: what they are, what our mHC/AttnRes already are, and whether the loop sign can flip

**Date:** 2026-11-08. **Scope:** read-only except this file. No code, config or `port/src` touched; nothing run on the GPU.
**Inputs:** the local PDF `docs/papers/iso-depth-2604.21106.pdf` (v3) read in full via `pdftotext -layout`; **arXiv:2604.21254v3 downloaded and read in full this pass** (the paper the 2026-11-06 note missed — see §1.2); arXiv:2409.19606v3 HTML (Tab. 1–3 verified directly); the repo's `port/src/dormouse/model/mhc.py`, `loopblock.py`, and `docs/reviews/2026-11-06-mechanism-liveness.md`, `docs/reviews/mhc-2026-09-30.md`, `docs/reviews/2026-11-05-kernel-profile.md`.
**Marks:** **[T]** table read · **[F]** figure/prose number, no table · **[A]** abstract only · **[ours]** repo document/measurement · **[S]** SPECULATION (my inference, nothing measured).

---

## TL;DR

1. **Hyperconnections are parallel residual lanes mixed across loop iterations — 32 scalars at K=2/r=4, 147 kFLOP/token, φ 0.453 → 0.646.** Not a depth mechanism, not a width trick: `ℓ(i+1) = M_i ℓ(i) + β_i ⊗ RecurBlock(α_i ⊙ ℓ(i))`, lanes initialised by broadcasting the prelude output, coda sums them [T, 2604.21106v3 Eq. 7 + App. G.1].
2. **The 2026-11-06 note's "no published fix reaches parity at ≤200M" is now FALSE.** arXiv:2604.21254v3 (Hyperloop Transformers) has a **135.7M looped model at 14.40 PPL vs a depth-matched 238.0M dense at 14.65**, same 12.5 B tokens, same 16 layer-applications, and the intervention is mHC at the loop boundary [T, Table 1]. That is the first ≤200M iso-FLOP + iso-token depth-recurrence **win** with a dense control, and it is worth −0.017 nats (1.7 % relative).
3. **But it is not our loop.** The winning shape keeps **8 unique layers** (prelude 2 + middle 4 ×3 + coda 2, 16 passes). The same paper's shape ablation is monotone against us: middle 4L **14.40**, 3L **14.62**, 2L **15.06** [T, Table 3]. We run a **1-layer** middle with **zero** prelude/coda and 3 passes.
4. **The arithmetic is a ceiling, not a slope.** With one unique layer and 3 passes, `N_eff = 3^φ`; even φ = 1 gives **3** layer-equivalents against a 12-layer control — **25 %**, and 17 % at φ = 0.65. Parity needs φ = **2.26**, and no measured φ exceeds 1. Hyperconnections cannot fix a missing block; they can only make a present block count for more.
5. **Our mHC is the same family, in its worst published parameterisation.** The repo's `MHCMix` is DeepSeek's constrained member (2512.24880): full-width lanes, 20-iteration Sinkhorn, state-dependent `H_res`. The Hyperloop ablation says the **diagonal** sigmoid `H_res` beats Sinkhorn by **0.19 PPL** (14.40 vs 14.59) and beats identity by 0.21 [T, 2604.21254v3 Table 6] — and our own in-place test says the Sinkhorn mixing is worth **+0.0001 BPB** while costing **23.2 ms = 15.4 %** of kernel time [ours, kernel-profile 2026-11-05; mechanism-liveness 2026-11-06 §2].
6. **AttnRes is a different thing, and it is the one that works.** It is a softmax readout over `[h0, y_0 … y_i]` that *replaces* residual accumulation, not lanes — worth **+0.238 BPB** in place at 100 M [ours]. It is not a subset of hyperconnections and does not add effective depth.
7. **Verdict: hyperconnections do not change the decision about *our* loop; they change the reason.** Do not spend the A/B on 1×3. If the loop lane is to be kept alive at all, the arm is the published shape (8 unique layers) and the cheapest decisive test is **74 MB × 3 seeds × 2 arms ≈ 2.2 h**, not 30 MB — 30 MB cannot resolve the effect (predicted 0.006–0.025 BPB vs a 0.035 pass rule).

---

## 0. The arithmetic the question starts from, corrected

The brief says: *"φ = 0.65 at r = 3 gives 2.04 layer-equivalents vs the dense control's 12, so the loop still has 17 % of the capacity."* The 2.04 is right; the denominator is comparing **3 layer-passes against 12**, i.e. the loop spending a quarter of the FLOPs. That is a valid statement about *our current config*, but it is not the φ law's comparison, and quoting it as "the capacity ceiling under φ" understates the equal-FLOP case by 2.4×:

| comparison | r | `N_eff = r^φ` (one unique layer) | vs 12-layer dense | what it means |
|---|---|---|---|---|
| **equal passes, 3 vs 12** (our config, ¼ the FLOPs) | 3 | 3^0.65 = **2.04** (φ=0.46: 1.66) | **17 %** (14 %) | what we run today |
| **equal FLOPs, 12 vs 12 passes** | 12 | 12^0.65 = **5.03** (φ=0.46: 3.14) | **42 %** (26 %) | the best a 1-layer loop can do at the control's compute |
| **passes needed for parity at equal width** | **46** | 46^0.65 = 12.0 | 100 % | 3.8× the control's FLOPs |
| `N_eff` for a *3-pass* block to equal 12 layers | 3 | needs φ = ln12/ln3 = **2.26** | — | never measured; φ>1 is unobserved [T, §3.3] |

Arithmetic is the paper's own units (non-embedding layers at equal width, `N_eff = N_once + r^φ N_rec`) [T, 2604.21106v3 Eq. 3]; applying it to a 1-layer block is **[S]** — the paper's smallest recurrent block is 2 layers. Nothing in this table depends on hyperconnections; it is what "one unique layer" costs under *any* φ.

The ceiling reading is the one that decides: **at 3 passes, φ would have to reach 2.26 for the loop to tie a 12-layer control.** At 12 passes, **46 passes are needed** — i.e. even a perfect tie at φ=0.65 costs 3.8× the dense FLOPs at equal width. Loops can only escape this by being *wider* per pass (which the compute-optimal reallocation does, and which is why the fitted frontier gap is only 0.03–0.12 nats rather than 3.8×) [T, §4.2–4.3].

---

## 1. The mechanism, from the two papers that measure it

### 1.1 Iso-depth case study (2604.21106v3 §5.2, App. G.1) — the φ = 0.65 arm

Replaces the looped model's **linear input injection** `u(t) = W_inject[e ∥ h(t)]` with K parallel lane states `ℓ(i) ∈ R^{K×d_model}` mixed at every recurrence:

```
u(i) = α_i · ℓ(i)              α_i ∈ R^K
s(i) = RecurBlock(u(i))
ℓ(i+1) = M_i ℓ(i) + β_i ⊗ s(i)  M_i ∈ R^{K×K}, β_i ∈ R^K
```

- **What is connected:** the *loop boundary only* — the lanes carry the recurrent state between iterations; nothing changes inside the block. The coda receives the **sum-pooled** lanes `Σ_k ℓ_k`.
- **Init:** cyclic, `α_i = e_{i mod K}`, `M_i = I`, `β_i = 1`, so the first forward is a plain looped model [T, App. G.1; the same trick as 2409.19606 Eq. 14].
- **Parameter cost:** `r(K² + 2K)` scalars = **32 at K=2, r=4** [T, App. G.1]. Not a hyper-network: per-iteration *static* scalars, not state-dependent matrices.
- **FLOP cost:** `6r(K²+2K)d_model` per token (fwd 2× + bwd 4×) = **147 kFLOP/token** at K=2, r=4, d=768 — ~32× cheaper than the `2d²`-per-iteration injection it replaces [T, App. G.1].
- **Measured:** φ **0.453 → 0.646**, R² 0.9900, on 83 runs over the four lower budgets 4.64e17–4.64e18 FLOPs [T, Table 3]. K=2 only, full BPTT.

Direct single-point read (Table 5, s=10, r=4, C=1e18): linear injection **2.793**, additive **2.797**, passthrough **7.400** (fails to train), **hyperconnections K=2 2.757** — −0.036 nats, with 973 M vs 955 M tokens (+2 % data from the saved injection FLOPs) [T, Table 5].

### 1.2 Hyperloop (2604.21254v3 §3) — the same idea at ≤200M, and it wins

Begin block → **middle block looped L times with mHC at each loop boundary** → end block. `Y^(0) ∈ R^{T×n×C}` is the begin output copied n times; per loop `l`:

```
y^(l+1) = H_res_l y^(l) + H_post_l F(H_pre_l y^(l)) + e_l
H_res = diag(σ(α_res·(W_res z) + b_res))     ← diagonal, not Sinkhorn
readout: average the n lanes → X_end
```

- **Three departures from mHC, all named by the authors:** diagonal sigmoid `H_res` instead of Sinkhorn over a dense matrix; a **loop position embedding** `e_l` added into the stream; hyperconnections **only at the loop level** (3 loops ⇒ 3 connections, vs mHC's one per sublayer) [T, §3].
- **Parameter cost:** **150–300 K extra** over the vanilla looped model — 135.7 M vs 135.5 M at the smallest scale [T, §4.2, Table 1].
- **Compute cost:** throughput 750 K tok/s vs 786 K for both the dense and the plain looped model — **−4.6 %**; the full mHC Transformer runs at **514 K (−35 %)** [T, Table 1, 8×H100].
- **Placement ablation:** HC every loop **14.40** < every layer (12 HCs) 14.45 < every 2–6 layers 14.50 < one HC every 12 layers 14.63 [T, Table 5]. Loop-level placement is the best of six.
- **`H_res` parameterisation ablation (the one that matters for us):** **diagonal 14.40**, Sinkhorn 14.59, identity 14.61 [T, Table 6]. The manifold constraint is *worse than not constraining at all* in this setting.
- **Stream count:** n = 2 **14.43**, 4 **14.40**, 6 14.38, 8 14.39, 10 14.35 — most of the gain by n = 2 [T, Table 4], matching 2409.19606 Tab. 1 (n=1 2.819 worse than baseline 2.811; n=2 2.802; n=4 2.781; n=8 2.778 at OLMo-1B/500 B tokens) [T, verified directly from the v3 HTML].

---

## 2. What the two papers measured, side by side

| | 2604.21106v3 (iso-depth, HC case study) | 2604.21254v3 (Hyperloop) |
|---|---|---|
| loops / shape | r ∈ {2,4,8}; 20 effective layers; prelude 2 + `16/r` recurrent + coda 2 | L = 3; 16 or 18 unrolled layers; prelude 2 + **4** recurrent + coda 2 (8 unique layers) |
| lanes | K = 2, static per-iteration scalars | n = 4, per-loop projections + loop embedding |
| `H_res` | dense, Sinkhorn-projected (mHC-style here: unconstrained HC per the citation) | **diagonal sigmoid** |
| params added | 32 scalars | 150–300 K (0.1–0.2 %) |
| FLOPs added | 6r(K²+2K)d ≈ 147 k/token, negligible | −4.6 % throughput vs dense in plain PyTorch |
| headline | **φ 0.453 → 0.646** | 135.7 M **14.40** vs 238.0 M dense **14.65**; 579.7 M **9.65** vs 990.5 M **10.19**; 990.8 M **8.49** vs 2018 M **8.60** |
| dense control also HC'd? | **no** — r=1 runs reused unchanged; the paper says an HC'd r=1 "would likely shift the calibration of φ downward" [T, §5.2] | **yes** — "mHC Transformer" 241.0 M **14.55** vs plain 238.0 M 14.65 |
| tokens | iso-FLOP budgets up to 2.15e19 (≈ 36 B tokens at 98 M) | 12.5 B (2.5× Chinchilla for the *dense* size) and 100 B |
| verdict at our scale | loops trail; r=2 "matches or beats r=1 at some budgets" [F, §5.2], never at all budgets | **win** at 12.5 B (−0.017 nats), **tie** at 100 B (12.19 vs 12.15) |

**The reconciliation is the token count, and it is measurable.** Isolate the HC effect in Hyperloop's own rows: dense 14.65 → dense+mHC **14.55** (−0.0069 nats); looped 14.85 → Hyperloop **14.40** (−0.0306 nats). So HC is worth **4.4× more at the loop boundary than between layers** — the paper's "loop-level beats layer-level" claim, quantified [T, Table 1, my subtraction]. And at 100 B tokens the same HC effect is −0.0299 nats (12.56 → 12.19) while the *loop's* own penalty grows from +0.0136 to +0.0331 nats, which is exactly why the win decays to a tie [T, Table 2].

**Inverting the φ law on the Hyperloop cell [S, arithmetic only]:** with `N_once = 4`, `N_rec = 4`, r = 3, parity with a 16-layer dense requires `4 + 3^φ·4 = 16` ⇒ **φ = 1.00**. The published Hyperloop result at 135 M/12.5 B is therefore consistent with an *effective* φ ≈ 1 for that recipe — far above the 0.646 that iso-depth measures at 20 layers, r ∈ {2,4,8} and 300+ tokens/param. Two published numbers, two different regimes; neither is wrong, and the difference is not the lanes (both use them) but the depth of the recurrent block and the token budget.

---

## 3. Our nearest arms: same idea, subset, or something else

| | repo `MHCMix` (`mhc.py`, `use_mhc`) | repo `AttnRes` (`loopblock.py`, `use_attnres`) | 2604.21106v3 HC | 2604.21254v3 Hyperloop |
|---|---|---|---|---|
| family | HC, **constrained** member (2512.24880) | different: depth-softmax **readout** (2603.15031) | HC (2409.19606) | HC, simplified (2512.24880 → diagonal) |
| what is parallel | n full-width lanes | nothing — one stream, attention over history | n lanes | n lanes |
| mixing matrix | `H_res` **dense, Sinkhorn, 20 iters**, state-dependent via φ_res | softmax weights `α_{i→l} = softmax(q_l·RMSNorm(k_i))` | dense, per-iteration static (paper) | **diagonal, sigmoid, state-dependent** |
| depth-connection | `H_post = 2σ(·)` per lane | the softmax mixture *replaces* `h + y·s` | `β_i` per lane | `H_post` per lane |
| read-out | sum of lanes ÷ n (composition) or raw sum (mHC-only) | the mixture itself | sum of lanes | **average** of lanes |
| params at d=1024 | **16 395** (0.02 % of 102.7 M) | 6 144 (0.01 %) | 32 scalars | ~200 K per 3 loops |
| placement | loop boundary | every iteration's write | loop boundary | loop boundary |
| measured at our scale | **+0.0001 BPB** in place (`H_res`→I); whole-flag arm invalid (topology change, +2.76 BPB, diverges) [ours] | **+0.238 BPB** in place (gain→0); whole-flag +0.36 [ours] | φ 0.65 at 30–400 M | −0.017 nats at 135 M |
| A/B run? | **no** — `ab_mhc.py` exists, 3 seeds × 2 arms × 2000 steps, never run [ours, battery 2026-10-21; AGENTS queue row 7b] | **no** — row 7 of `AB-PROTOCOL.md`, never run [ours] | — | — |

**Precise verdict on identity.** `MHCMix` is **the same idea in a superset parameterisation**: it can express Hyperloop's operator (`M_i = diag`, `β_i` fixed) by driving `α_res → 0` and freezing the Sinkhorn, but nothing in the repo selects that setting, and the three published ablations all point *away* from what we shipped:

1. **Sinkhorn is the worst of the three `H_res` choices at 135 M** (14.59 vs diagonal 14.40) [T, 2604.21254v3 Table 6] — and at our scale it is measurably inert (+0.0001 BPB) while consuming 23.2 ms of a 151 ms kernel budget [ours].
2. **`H_pre` is not wired** — 11 % of the base paper's effect, deliberately dropped because our body runs at full width [ours, `mhc-2026-09-30.md` §3.3; 2409.19606 Tab. 1: `H_res`-only −0.022, +`H_pre` −0.025, all three −0.027].
3. **One shared mixing block for all iterations**, where Hyperloop uses per-loop projections + a loop position embedding [ours, §3.2 of the same review; T, 2604.21254v3 §3].

`AttnRes` is **not hyperconnections and not a subset of it.** It has no lanes: it re-weights *historical iteration outputs* by a learned query and returns a convex combination instead of a residual sum. In HC's vocabulary it is the depth-connection half with softmax coefficients over time, and it cannot increase `N_rec` or `r`, so it moves the *readout* and not φ. It is nonetheless the single most valuable residual mechanism in the 102 M flagship by our own measurement (+0.238 BPB, 100× the mHC mixing) — which says the depth-mixing idea pays, but through the readout, not through the manifold.

**Nothing in the repo's A/B archive measures either arm at iso-FLOPs.** The only quantity we have is the forward-only in-place ablation on one checkpoint (`checkpoints/flagship2`, step 49 999) with the block's own trained weights; it is strong evidence that the Sinkhorn mixing is dead weight, and it is not a substitute for a from-scratch arm.

---

## 4. Does anything with hyperconnections reach parity at ≤200 M?

**Yes — and the config is specific.** From 2604.21254v3 Table 1 and Table 2, the 135 M class:

| model (d=1024, unrolled depth 16, 12.5 B tokens) | params | PPL BF16 | PPL INT4 | task acc | tok/s |
|---|---:|---:|---:|---:|---:|
| Transformer (dense 16L) | 238.0 M | 14.65 | 14.85 | 41.1 % | 786 K |
| **mHC Transformer** (dense + lanes between layers) | 241.0 M | **14.55** | 14.73 | 41.1 % | 514 K |
| Looped [2 → 4×(×3) → 2] | 135.5 M | 14.85 | 15.18 | 41.4 % | 786 K |
| **Hyperloop** [2 → 4×(×3) → 2] | **135.7 M** | **14.40** | **14.68** | **41.6 %** | 750 K |

| same models at 100 B tokens (overtrained) | PPL |
|---|---:|
| Transformer 238.0 M | 12.15 |
| mHC Transformer 241.0 M | 12.16 |
| Looped 135.5 M | 12.56 |
| Hyperloop 135.7 M | **12.19** |

Three readings, in decreasing order of how much they matter to us:

1. **At 52 tokens/param the loop wins outright** — 14.40 vs 14.65 vs its FLOP twin, and 14.40 vs 14.55 against a dense model that also has HC. Parameter count is 57 % of the dense. This is a genuine iso-depth/iso-token/iso-FLOP win with a dense control, at 135 M, and the intervention is hyperconnections at the loop boundary [T, Table 1].
2. **The win is fragile in one direction we can name:** it is gone by 400 tokens/param (12.19 vs 12.15) [T, Table 2]. Everything in the iso-depth grid lives at 300+ tokens/param, which is why that paper sees φ = 0.65 and a trailing frontier; everything in our own box lives at **2.6 tokens/param** (0.25 GB / 95 M).
3. **The shape matters more than the intervention.** Middle block 4L → **14.40**, 3L → 14.62, 2L → **15.06**, at fixed 16 unrolled layers [T, Table 3]. Monotone, no knee. Extrapolating one step below 2L is what we run: a **1-layer** middle with no prelude and no coda.

**Answer to "does any published config with hyperconnections reach parity at ≤200 M":** yes, `[2 → 4×3 → 2]` at 135.7 M, and only in that neighbourhood. **No published config reaches parity with a 1-layer recurrent block**, at any φ, because the ceiling argument in §0 does not involve φ's *value* — it involves `N_rec = 1`.

---

## 5. The arithmetic at 95 M, done on our shapes

Our current family: `max_iter=1`, 12 unique layers, d=832, **1.4621 BPB at 0.25 GB** [ours, `2026-11-08-efficiency-frontier.md` §2]. The loop family that lost: `max_iter=3, loop_blocks=1, prelude=0, coda=0`, 102.7 M total / 17.26 M MAC/token, **137.42 MFLOP/B measured** against a 13.5 M dense control at **135.36 MFLOP/B** — i.e. our loop-vs-dense comparison *was* iso-FLOP, at d=1024 vs d=304, and the loop lost by **0.126–0.27 BPB** [ours, `2026-11-06-control-transformer.md`].

| candidate at ~95 M, iso-FLOPs with the 12L×832 dense | unique layers | passes | `N_eff` at φ=0.65 | vs dense | published prior |
|---|---:|---:|---:|---:|---|
| our 1 × 3 (d≈1024, ¼ the FLOPs) | 1 | 3 | 2.04 | 17 % | — (measured: +0.13…0.37 BPB) [ours] |
| 1 × 12 (same FLOPs, same width) | 1 | 12 | 5.03 | 42 % | none published |
| **2 + 4×2 + 2** (iso-pass, 8 unique) | 8 | 12 | 4 + 2^0.65·4 = **10.3** | **86 %** | r=2 is the least-bad rung [T, iso-depth Table 3]; MoR cycle-2 is the one ≤200 M win without HC [T, 2507.10524 Table 10] |
| **2 + 4×3 + 2** (published winner, 16 passes) | 8 | 16 | 4 + 3^0.65·4 = **12.2** | 76 % of *its* 16L control | **wins** 14.40 vs 14.65 [T, 2604.21254 Table 1] |

The middle row is the one worth stating plainly: **the same FLOPs, spent as 12 passes of one layer, buy 42 % of the dense's effective depth under the best published φ; the same FLOPs spent as 12 passes over 8 unique layers buy 86 %.** The difference between those two rows is not φ and not hyperconnections — it is **seven more unique layers**. That is a config change (`prelude_layers`, `coda_layers`, `loop_blocks`), all fields already in `config.py`.

Why the 76 % row still wins while the 86 % row is untested [S]: the φ law is fit on iso-compute runs at 300+ tokens/param, and Hyperloop's own 100 B cell shows the loop penalty *growing* with tokens (0.014 → 0.033 nats) while the HC effect stays at 0.030. Extrapolating the other way — to our 2.6 tokens/param — the penalty should shrink, not grow. That is an extrapolation in a direction neither paper measured, and it is contradicted by our own 0.25 GB measurement *for a 1-layer block*. It is not a reason to run anything; it is a reason not to quote F.4-style extrapolations at our token budget.

---

## 6. The cheap test — and why 30 MB is the wrong size

**Is the answer already no?** For the arm we have: **yes**, and no test is needed (§0). For the published shape: **no**, there is a live prior, so the test below is specified — with the honest power calculation first, because it is what kills it.

| budget | 1σ per arm | 2-arm × 3-seed pass rule (2σ) | minutes/arm | total |
|---|---:|---:|---:|---:|
| **30 MB** (the brief) | 0.02–0.03 (interpolated; nearest measurement 0.051 at 36.9 MB) [ours, `2026-11-08-lr-schedule.md` §4.1] | **≥ 0.035** | 9.1 | 0.9 h |
| 74 MB | **0.011** measured [ours, same table] | ≥ 0.011–0.02 | 22 | **2.2 h** |
| 250 MB (our production budget) | 0.005–0.01 [S, 1/√D] | ≥ 0.01–0.02 | 74 | 7.4 h |

**Predicted effect at our scale: 0.006–0.025 BPB.** Lower bound = the absolute nats transfer (−0.017 nats/token ÷ ~4 B/token ÷ ln2); upper bound = the relative transfer (1.7 % of 1.46 BPB) — the same convention the 2026-11-06 note used, and the two differ by 4×, which is itself the honest state of the transfer [S]. **At 30 MB the pass rule (0.035) is above the top of the prediction. The test is VOID by construction** — it would return "tie" whether HC works or not, and §1.2 says a tie deletes the mechanism. That is the one outcome worse than not running it.

**If the owner overrules and wants the lane alive, this is the arm:**

| field | value | why |
|---|---|---|
| shape | `prelude_layers=2`, `loop_blocks=4`, `max_iter=2`, `coda_layers=2` | ISO-PASS with our 12-layer dense: 8 unique layers, 12 passes, ~1.0× FLOPs; closest published shape is Hyperloop's 4L middle |
| `use_mhc` | **true, `mhc_streams=2`** | most of the gain by n=2 [T, 2604.21254 Table 4]; K=2 is what the φ study ran |
| `H_res` | **diagonal or identity, NOT Sinkhorn** | 14.40 / 14.61 / 14.59 [T, Table 6]; ours is inert and costs 15.4 % of kernel time [ours] |
| `use_attnres` | **false** | it is worth +0.238 BPB on its own, and composing it would confound the one comparison this arm exists to make |
| control | the existing 12L×832/95 M dense, the **same bytes and the same 12 passes** | our eval window rule: one batch size across arms |
| budget | **74 MB, 3 seeds × 2 arms** — not 30 MB | 30 MB cannot resolve 0.025 |
| pass rule (pre-registered) | arm mean BPB **≥ 0.020 better** than control (2σ at 74 MB) **and** step time ≤ +5 % | below that it is a tie, and a tie deletes the arm |
| cost | 6 × 22 min ≈ **2.2 h** + eval | |
| why it is probably still a loss | predicted 0.006–0.025 vs a 0.020 bar; and the arm's whole product is parameter savings we do not need at 95 M on 16 GB | |

The cheaper, strictly-better experiment is not a loop arm at all: **replace `MHCMix`'s Sinkhorn with identity/diagonal and measure step time.** Our own number says the mixing is worth +0.0001 BPB and costs 23.2 ms of 151 ms (15.4 %) [ours]. That is a one-line arithmetic change in `mhc.py`, no A/B needed for the deletion (the mechanism is already measured as inert), and it buys ~15 % of the GPU back for whatever arm the owner does want to run.

---

## 7. VERDICT (Russian)

**Гиперконнекции не меняют решения про *нашу* петлю. Они меняют его обоснование.**

1. **Для 1 × 3 — нет, и это арифметика, а не мнение.** `N_eff = r^φ` при одном уникальном слое: φ=0.65, r=3 → **2.04** слое-эквивалента против 12; даже φ=1 даёт **3** (25 %). Для паритета нужно φ=**2.26**, а измеренного φ>1 не существует. Никакая схема связей не добавит отсутствующий блок. Опубликованный аблашен формы это подтверждает монотонно: средний блок 4L **14.40**, 3L 14.62, 2L **15.06** — у нас 1L и без prelude/coda [T, 2604.21254 Table 3].
2. **Литературное утверждение «до 200M ни одна правка не выходит на паритет» — ЛОЖНО с этого прохода.** 2604.21254v3: 135.7M петля **14.40** против 238.0M dense **14.65** (те же 12.5B токенов, те же 16 слое-проходов), и это mHC на границе петли. Против dense, у которого тоже есть HC — **14.40 против 14.55**, тоже победа [T, Table 1]. То есть знак переворачивается — но только у формы с **8 уникальными слоями**.
3. **Наш mHC — та же идея в худшей из опубликованных параметризаций.** Sinkhorn проигрывает диагонали **0.19 PPL** (14.59 vs 14.40) и почти не отличается от identity (14.61) [T, Table 6]; у нас он в изоляции стоит **+0.0001 BPB** и **23.2 мс = 15.4 %** ядерного времени [ours]. `H_pre` не подключён (11 % эффекта базовой статьи), матрица одна на все итерации, loop-embedding отдельный. Это не «у нас уже есть интервенция» — это «у нас есть её самая дорогая и самая слабая версия».
4. **AttnRes — не гиперконнекции.** Это softmax-чтение по истории, заменяющее накопление residual; оно не добавляет ни уникальных слоёв, ни проходов, и φ не двигает. Но по нашему же замеру оно стоит **+0.238 BPB** — в ~2400× больше, чем микширование mHC. Его надо оставить и перестать называть HC.
5. **A/B на петлю не тратить.** Для текущей формы — закрыто арифметикой. Для опубликованной формы — минимальный валидный тест **74 МБ × 3 сида × 2 плеча ≈ 2.2 ч**, проходной порог **≥0.020 BPB**; 30 МБ физически не разрешает эффект (прогноз 0.006–0.025 против порога 0.035), и такой тест вернёт «ничью» независимо от истины. А ничья удаляет механизм.
6. **Что сделать вместо этого, за час:** выключить Sinkhorn (identity/диагональ). Механизм уже измерен как инертный, значит удаление не требует A/B — оно возвращает ~15 % шага. Это единственное действие, которое сегодня окупается на этом плече.

---

## 8. VERIFIED (sourced)

1. Hyperconnection operator at the loop boundary, K=2 lanes, cyclic init, sum-pooled coda, `r(K²+2K)` = 32 scalars, `6r(K²+2K)d` FLOPs/token — [T] 2604.21106v3 §5.2 + App. G.1.
2. φ = 0.453 baseline → **0.646** with HC, R² 0.9900, 83 runs, four budgets 4.64e17–4.64e18 — [T] 2604.21106v3 Table 3. Full grid φ = 0.459, CI [0.41, 0.53], 116 runs — [T] Table 2.
3. Table 5, s=10, r=4, C=1e18: linear 2.793 (955 M tok), additive 2.797, passthrough 7.400, HC K=2 **2.757** (973 M tok) — [T] Table 5.
4. "r=2 matches or beats r=1 at some budgets"; φ=1 would need all four architectures on one frontier; an HC'd r=1 baseline would shift the calibration of φ downward — [F] 2604.21106v3 §5.2.
5. Hyperloop 135.7 M **14.40 / 14.68 INT4 / 41.6 % / 750 K tok/s** vs Transformer 238.0 M **14.65 / 14.85 / 41.1 % / 786 K**, mHC 241.0 M **14.55**, Looped 135.5 M **14.85**, same 12.5 B tokens, unrolled depth 16, d=1024 — [T] 2604.21254v3 Table 1.
6. At 100 B tokens: 12.15 / 12.16 / 12.56 / **12.19** — [T] 2604.21254v3 Table 2.
7. Shape ablation 4L → 14.40, 3L → 14.62, 2L → 15.06 at fixed 16 unrolled layers — [T] 2604.21254v3 Table 3.
8. `H_res`: Identity 14.61, Sinkhorn 14.59, **Diagonal 14.40** — [T] 2604.21254v3 Table 6. Streams: 2 → 14.43, 4 → 14.40, 10 → 14.35 — [T] Table 4. HC placement: every loop 14.40, every layer 14.45, every 12 layers 14.63 — [T] Table 5. Extra params 150–300 K — [T] §4.2.
9. Per-loop LoRA does not close the loop's gap: rank 0/4/8/16/32 → 14.85/14.85/14.81/14.80/14.77 vs dense 14.65 — [T] 2604.21254v3 Table 7.
10. Base HC paper, OLMo-1B/500 B tokens, V2 eval loss: baseline 2.811, DHC×1 2.819, DHC×2 2.802, DHC×4 2.781, DHC×8 2.778; `WC` trainability is worth 0.025 and `B` 0.002 — [T] 2409.19606v3 Tab. 1/3, read directly from the v3 HTML this pass.
11. Our `MHCMix`: Sinkhorn 20 iterations, α=0.01, b_pre=10, b_post=0, b_res=0, full-width lanes, `H_pre` applied by the caller — [ours] `port/src/dormouse/model/mhc.py:1-52,71-89`.
12. Flagship: d=1024, `max_iter=3`, `loop_blocks=1`, `prelude_layers=0`, `coda_layers=0`, `use_mhc=true`, `mhc_streams=2`, `use_attnres=true` — [ours] `port/configs/flagship_base.toml:3-50`.
13. In-place ablation at 102.7 M / step 49 999: mHC `H_res`→I **+0.0001 BPB** (INERT); AttnRes gain→0 **+0.238 BPB**; `loop.mhc` 16 395 params (0.02 %), `loop.attnres` 6 144 — [ours] `docs/reviews/2026-11-06-mechanism-liveness.md` §1–§2, §5.
14. mHC write kernel **23.2 ms of 151 ms kernel time (15.4 %)**, more than KDA+CSA combined (22 ms) — [ours] `docs/reviews/2026-11-05-kernel-profile.md` §"По регионам".
15. Neither arm has an A/B: `ab_mhc.py` is specified (3 seeds × 2 arms, 2000 steps, `medium`) and was never run; AttnRes is queue row 7, never run — [ours] `2026-10-21-battery.md:86`, `AB-PROTOCOL.md:225`, `mhc-2026-09-30.md` header.
16. Our loop-vs-dense is iso-FLOP: flagship 137.42 MFLOP/B vs control 135.36 (−1.5 %), and the gap is 0.126–0.27 BPB — [ours] `docs/reviews/2026-11-06-control-transformer.md` §1–§Verdict.
17. Eval noise floor: 1σ 0.051 at 36.9 MB, 0.019–0.022 at 49.2 MB, **0.011 at 73.7 MB**; 30 MB interpolated 0.02–0.03; 9.1 min/arm at 30 MB, 56 kB/s — [ours] `docs/research/2026-11-08-lr-schedule.md` §4.1–4.2.
18. Sharpest ≤200 M baseline: 12L × d=832, 95.0 M, 0.25 GB → **1.4621 BPB**, 1.42e17 FLOPs — [ours] `docs/research/2026-11-08-efficiency-frontier.md` §2.

---

## 9. SPECULATION (my inference, nothing measured)

- **[S1] φ ≈ 1 for the Hyperloop cell.** Inverting `N_once + r^φ N_rec` at `N_once = 4`, `N_rec = 4`, r = 3, parity with 16 layers needs φ = 1.00, and the measured cell beats parity by 0.017 nats. So *something* in that recipe (4 lanes, diagonal H_res, loop embedding, 8 unique layers, 52 tokens/param) is worth more than the 0.646 the iso-depth study measures. Which ingredient is unknown; the two papers do not share a grid.
- **[S2] The transfer to our regime is unmeasured in either direction.** Hyperloop's win decays 0.017 → 0.003 nats between 52 and 400 tokens/param. We train at **2.6 tokens/param**, byte-level, 20× below their shortest run and on a different tokenizer. The trend extrapolates to *bigger* wins, our own 0.25 GB measurement says otherwise for a 1-layer block, and no published number sits at our token count.
- **[S3] BPB conversion.** The 0.006–0.025 BPB band is the absolute-vs-relative transfer of −0.017 nats/subword-token. Neither convention is measured at byte level; the 4× spread is the honest uncertainty, and it straddles the 74 MB pass rule.
- **[S4] A 1-layer recurrent block is worse than the φ fit.** The fit averages over 2–8-layer blocks; a 1-layer block cannot host the depth-wise specialisation, so 3^0.65 = 2.04 is if anything optimistic. Direction unchanged.
- **[S5] AttnRes may raise φ slightly**, because it is the depth-connection half of HC with learned convex coefficients. Untested — no one has fit φ with AttnRes.
- **[S6] The 15 % step-time recovery from deleting the Sinkhorn** assumes `mhc_write`'s 23.2 ms is dominated by the 20 Sinkhorn iterations; the profile does not split the epilogue's norm/RMS work from the projection. The `log_softmax` measured variant (881 → 81 dispatches) supports it [ours, `mhc.py:71-82`], but the fused-write path is a separate kernel.

---

## 10. Open questions

1. **Does a 8-unique-layer loop with mHC beat our 12L×832 dense at 0.25 GB?** 2.2 h at 74 MB (power ~40 %), 7.4 h at 250 MB (power ~80 %). The only version of the loop lane with a published prior.
2. **What does deleting the Sinkhorn cost in quality and buy in time?** One flag, one in-place eval, one step-time reading. If BPB moves < 0.005, it is free.
3. **Does the φ law hold for a 1-layer block at all?** A 3-point sweep (1×3, 2×3, 4×3 at 20 M, 300 MB) settles the extrapolation in §0 — but the answer cannot change the decision, only the explanation.
4. **Would the Hyperloop result replicate at byte level and 2.6 tokens/param?** No published number, and it is the whole question for us.
5. **Does `H_pre` (the 11 % we dropped) matter more at 1–2 passes than at 16?** Unmeasured anywhere; our body would need a quarter-width variant to test it.

---

## Sources

1. Schwethelm, Rückert, Kaissis — *How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models*, arXiv:2604.21106v3 (2026-05-07). Local PDF `docs/papers/iso-depth-2604.21106.pdf`, sha256 in `docs/papers/provenance.tsv:29`. §3.2, §4.2–4.3, §5.2, App. G.1, Tables 2/3/5.
2. Zeitoun, Torroba-Hennigen, Kim — *Hyperloop Transformers*, arXiv:2604.21254v3 (2026-07-02). Downloaded and read in full this pass. §3, §4.1–4.3, Tables 1–7.
3. Zhu et al. (ByteDance Seed) — *Hyper-Connections*, arXiv:2409.19606v3 (2025-03-18). v3 HTML read directly: §2.1–2.3, Eq. 1–14, Table 1 (expansion rate), Table 3 (`WC`/`B` ablation), App. B (parameter count).
4. Xie et al. (DeepSeek-AI) — *mHC: Manifold-Constrained Hyper-Connections*, arXiv:2512.24880v2 — the paper `burn-mhc` / `mhc.py` transcribe; cited here through `docs/reviews/mhc-2026-09-30.md` §1–§2, not re-fetched.
5. Kimi Team — *Attention Residuals*, arXiv:2603.15031 — `AttnRes`, transcribed in `docs/research/2026-09-30-attnres-integration.md`.
6. Repo: `port/src/dormouse/model/mhc.py`, `port/src/dormouse/model/loopblock.py:1340-1390`, `port/configs/flagship_base.toml`, `port/scripts/ab_mhc.py`, `docs/reviews/mhc-2026-09-30.md`, `docs/reviews/2026-11-06-mechanism-liveness.md`, `docs/reviews/2026-11-05-kernel-profile.md`, `docs/reviews/2026-11-06-control-transformer.md`, `docs/research/2026-11-08-lr-schedule.md`, `docs/research/2026-11-08-efficiency-frontier.md`, `docs/research/2026-11-06-depth-recurrence.md`, `docs/protocols/AB-PROTOCOL.md:225`.
