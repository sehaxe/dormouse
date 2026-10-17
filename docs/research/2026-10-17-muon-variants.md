# The Muon-variant landscape, ranked for a host-dispatch-bound port

**Date:** 2026-10-17. **Read-only:** one file written; no GPU run, no code change,
nothing committed. `optim.py` / `config.py` / `loop.py` untouched.
**Ask:** rank the 2026 Muon-variant landscape against the port's own constraint —
the optimizer is 44-53% of the b1 step and the step is **host-dispatch-bound**
(GPU residue ≈ 0 ms, wall ≈ enqueue) [ours, `docs/reviews/2026-10-16-b1-profile.md`
§1, §3; `.bulba/memory.md:317`], so variants that ADD torch ops per step lose, and
variants that REDUCE launch count or change the update rule at near-zero overhead
are the only ones worth an A/B.
**Method:** arXiv MCP, 12 `get_abstract` resolutions (every brief ID resolved, none
paraphrased from a title), 14 section reads across 8 papers, three repos' code read
(`microsoft/dion` tree + `README` + `requirements_gns.txt` +
`dion/newton_schulz_triton.py`; `thib-s/flash-newton-schulz` README;
`zichongli5/NorMuon` metadata only). Measured on `meta`, not the card:
`configs/medium.toml` = **141 910 405** params, **90 925 952** in the Muon group,
**12** distinct shapes [ours, 2026-10-17].
**Marks:** **[A]** the paper's own words · **[code]** the authors' code · **[ours]**
repo fact, file named · **[S]** SPECULATION (my inference, no source).

## TL;DR

1. **Muon+ is already our optimizer.** `2602.21545`'s best arm is `Norm_(col_row)`
   — column-normalize then row-normalize the polar update — which IS
   `colrow_normalize`, the port's default since `9ee82c5` and its largest measured
   A/B win (−0.0428 BPB) [A §3.3, Table 6; ours `optim.py:322-330`]. One candidate
   needs no work, and every other "beats Muon" number below was measured against a
   control **we do not run**.
2. **The only variant that attacks our actual bottleneck is Turbo-Muon's Triton
   Newton–Schulz** (`2512.04632` + `thib-s/flash-newton-schulz`): AOL
   preconditioning fused into iteration 1, a fused third kernel, 4 iterations —
   ~24 fewer torch dispatches per NS chain per step [A Table 1; code]. It is
   **plain Triton**, so the Hopper/TMEM wall that blocks Dion3 does not apply [code].
3. **This does not resurrect Gram-NS**, measured 1.2-1.4× SLOWER here (1.37-1.50
   vs 1.09-1.15 ms) because it hands back more calls [ours `.bulba/memory.md:317`].
   Turbo-Muon's kernels are the other half of that idea: same polynomial, fewer
   launches. **The dispatch column kills most of the field for us** — Muon-NSR/
   Muon-VS (+~14 ops/matrix/step, +1× state), Mousse (Kronecker state m²+n² per
   matrix), Dion3's fractional update (+5-8 ops, no launch reduction), AdaMuon
   (2mn state, dominated by Muon-VS in the one paper that compared them).
4. **Top pick:** Turbo-Muon Triton as a new `ns_kind` arm, with `ns_steps 5 → 4`
   riding on it — the published 4-iteration result is AOL-conditioned, so it is one
   arm, not two [A Table 2a]. Second: NorMuon's EMA row statistics in place of our
   instantaneous row norm (~+5 ops/chain, +m scalars).
5. **The honest caveat on every quality number here:** they are measured against a
   Muon baseline without post-polar normalization, which we already run. The
   transferable deltas are unknown, and our control's own 3-seed spread is the only
   bar that counts (AGENTS §1.2).

## 1. Verification table

"Cost on our box" is the **per-step dispatch delta per NS chain** (one shape-group
chunk; `MUON_CHUNK_ELEMS = 1<<25`, `optim.py:106`). At `medium` there are **14 NS
chains/step** — 7 batched (5 shape groups, 2+2+1+1+1 chunks) and 7 per-param
singletons falling back to `_step_one` (`optim.py:571-581`) [ours, measured]. The
classic torch chain is ~53 dispatches (2 prescale + 3 orientation + 5×(3 matmul +
5 elementwise) + 2 exit + 4 ColRow + 2 write) — an **estimate**, method shown so a
reader can check it; the profile's 246 `mm` + 366 `bmm` at b1 is its measured
counterpart [ours, profile §3].

| variant | arXiv ID (how verified) | what it changes | memory delta | measured gain (scale) | cost on OUR box |
|---|---|---|---|---|---|
| **Turbo-Muon** | `2512.04632` v2 — `get_abstract` + §4.1/§4.3/§6 read; repo README read | AOL pre-conditioning (column scaling from row sums of \|Gram\|) fused into iteration 1, a fused `ns_line_3` kernel, **4 NS iterations instead of 5** | **+O(min(m,n))** — a scaling vector [A §4.1] | polar error 0.25→0.06 at 4096; runtime 130→46 ms, 2.83× [A Table 1]. 144M GPT/FineWeb: 4 iters = 5-iter loss (3.28±0.001) [A Table 2a]; nanoGPT speedrun 273.8→266.0 s (−2.8%), ImageNet −3.6% [A Table 2b] | **−24 dispatches/chain** (~53→~29): 12 kernel launches replace 40 ops, AOL replaces the Frobenius prescale, one iteration gone. **The one lever that cuts calls** |
| **Dion3** | `2608.11612` v1 — §6/§7/§8.1/§8.2 read; repo tree + `requirements_gns.txt` read | Gram-NS + CuteDSL symmetric GEMM + **fractional update** (top-`⌈fn⌉` rows by ℓ1 norm, error feedback on selected rows only) | none (v is `[n]`) [A Alg. 4] | 1.5× (Gram+symmetric kernels); +2× at f=1/2, +3.7× at f=1/4 ⇒ 6.5× total, GH200, compute-bound [A §8.2]. 1B/100B tok: f=1/8 lowest loss; 3-14B: beats NorMuon at every scale, −0.027 at 14B, +0.7 pp downstream [A §8.1] | **+5-8 dispatches/chain** (ℓ1 norm, topk, gather, scatter, mask decay) and **no launch reduction** — selection shrinks matmul *sizes*, not counts. Gram-NS already measured 1.2-1.4× slower here [ours]. Kernels: CuteDSL, "Hopper and Blackwell", "tensor memory (Blackwell)" [A §6] — **sm_120 has no TMEM** [ours, task context] |
| **NorMuon** | `2510.05491` v1 — §3.1/§3.3/§4.1.4 read; repo `zichongli5/NorMuon` verified via GitHub API ("Official Implementation for NorMuon paper", 95★, MIT, last push 2026-04-30) | row-wise **EMA** second moment `v∈R^m` of the post-polar update, row-normalize by `√v`, then an RMS-matching rescale | **m(n+1)** vs Muon's mn [A §3.3] — 5.4B: 21.19 vs 21.14 GB of state | +11.31% over Muon, +21.74% over Adam at 1.1B [A abstract]; 124M/350M modded-nanoGPT beats Muon [A Fig. 5]; step time 2.9% over AdamW vs Muon's 2.8% [A Table 2] | **+~5 dispatches/chain**: mean of O², EMA on `[m]`, sqrt/div. **Replaces** our instantaneous row norm (1 norm + 1 div), so the net is small. **Overlaps ColRow's row half** |
| **SF-NorMuon** | `2605.23061` v2 — `get_abstract`; §3.1/§3.2 already read in `2026-10-16-sf-muon.md` | schedule-free spectral: `c_{t+1}=η²/s`, weight decay at the fast iterate Z | none beyond NorMuon | 125M/772M, 1-8× Chinchilla: matches tuned AdamW, **trails cosine NorMuon by ~0.03 nats** [A abstract] | ±0 dispatches (a schedule change). Not testable at a fixed 2k horizon [ours, `2026-10-16-sf-muon.md`] |
| **Muon-NSR / Muon-VS** | `2601.14603` v2 — §4/§5.1/§5.3 read | **pre**-orthogonalization variance modulation: `Γ_t=βΓ+(1-β)β(M_{t-1}-G_t)²`; VS: `M̃/(√Γ̂+ε)`; NSR adds `γ` | **+mn** (a full-size variance buffer) [A Alg. 1] | Llama-1.2B: Muon-VS **1.33×** step-to-target over tuned Muon; beats NorMuon and AdaMuon at 130M/300M [A §5.1] | **+~14 dispatches/chain** (Γ update, 2 bias corrections, proxy, 5-op modulation). Same class of add that made Gram-NS lose 1.2-1.4× [ours] |
| **MuonMax-Momo** | `2510.09827` v1 — §1/§4 read | a novel **product norm** + a **stale** per-layer nuclear-norm estimate of the momentum; Momo = model-based momentum | "near-identical" [A §1] | "5% additional wall-clock time per iteration"; LR-multiplier sweep {0.03…100}, 3 seeds: share of LRs with loss < 3.65 = 25% (MuonAdam/Scion), **50% (MuonMax-Momo)**, 62.5% (MuonAdam-Momo) [A §1, §5] | **+~4-6 dispatches/chain** (nuclear norm ≈ SVD/reduction, staler). Its benefit is LR robustness — a tuning-budget property our fixed-LR 2k A/B does not measure |
| **Mousse** | `2603.09697` v2 — §3.2/§4.2/§5.3 read | Shampoo Kronecker whitening then polar: `ΔW = −L^{-1/4} msign(L^{-1/4}GR^{-1/4}) R^{-1/4}`, `Eigh` every T steps | **m²+n² per matrix** (L, R) [A Alg. 1]; single-sided halves it [A §5.3] | 160M-800M: −0.012 final val loss at 800M, **~12% fewer steps**, "training speed nearly identical to Muon" [A §4.2] | **+~6-8 dispatches/chain** + an `Eigh` every T steps. **Memory is the killer at our shapes** — see §3 |
| **Muon+** | `2602.21545` v3 — §3.3 + Table 6/7/8 read | one normalization after polar; best arm = `col_row` | **zero** ("adds no optimizer state") | LLaMA-60M/130M/350M: None 25.75/19.34/14.02 → Col-Row 25.25/18.65/13.41; LLaMA-350M Jordan: 14.02→13.41; GPT-Small even with exact SVD 29.17→27.90 [A Tables 6-8] | **already implemented** — `colrow_normalize`, default, measured −0.0428 BPB here [ours] |
| **AdaMuon** | `2507.11005` — `get_abstract` (note: **2025-07-15**, not 2026) | elementwise second moment on the **orthogonalized** direction + sign-transform before orthogonalization + RMS-aligned rescale | **+mn** | "surpass Adam by more than 40% training efficiency" — vs **Adam** [A abstract]; Muon-NSR's own paper reports beating AdaMuon at 130M/300M [A `2601.14603` §5.1] | +~10 dispatches/chain. Dominated in the only head-to-head read |
| **Lotus** | `2602.01233` — `get_abstract` | **NOT Muon**: GaLore-family randomized low-rank gradient projection with adaptive subspace switching | +low-rank buffers | −30% training time / −40% gradient+optimizer memory vs GaLore, which pays an SVD [A abstract] | **No fit.** Our Muon state is already 1mn (bf16 optional); our low-rank is **structural in the model** (rank-64 TSCT factors), not an optimizer compression. Nothing to compress |
| **Survey of Optimizers** | `2608.28557` — §1/§3 read | meta-review, four axes | — | "matrix-aware methods are a real advance, but there is **no context-free replacement for AdamW**. Rankings change with scale, data-to-parameter ratio, batch size, schedule, parameter partition, tuning budget, and whether the objective is tokens, FLOPs, wall-clock, or memory" [A abstract]. Three evidence levels (scaled / controlled / emerging); §8.4's three fair comparisons (fixed model / fixed memory / fixed wall-clock) [A] | Not a variant — it is the protocol warning that a 141.9M / 2k-step A/B cannot settle any of the cross-scale rankings above |

**ID resolutions, and how:** every brief ID resolved via `get_abstract` (no 429 this
session). Three were **not** in the brief and were found by search, not guessed:
Turbo-Muon = `2512.04632` (`ti:"Turbo-Muon"`), Mousse = `2603.09697`
(`all:"Mousse" AND all:"optimizer"`), AdaMuon = `2507.11005` (`all:"AdaMuon"`).
`2512.04632` is the id the earlier cost note cites for the same method. **Four brief
facts do not survive contact with the source:** Muon-NSR's headline is **1.33×**
(Muon-VS, Llama-1.2B), not 1.36× [A §5.1]; MuonMax's LR sweep is {0.03 … 100} ≈ 3.5
decades with the prose claim "several orders of magnitude", not a measured 5 [A §1,
§5]; Mousse's "~3% overhead" appears nowhere I read — the paper says "nearly
identical training speed to the highly efficient Muon optimizer" [A §4.2];
AdaMuon's id resolves to **2025-07-15**, not 2026. NorMuon's "mn+m" is `m(n+1)`
[A §3.3] — same thing, their notation.

## 2. Ranked shortlist FOR US

Protocol for every row: 3 seeds × 2 000 steps, `medium` (141.9M), b8 × s512, pure
CE (`--jepa-weight 0 --dspark-weight 0`), eval window 20 × 8 × 512 = **81 920 B on
every line**, verdict against the control's own 3-seed spread (AGENTS §1.2,
`docs/protocols/AB-PROTOCOL.md`), one batch size across all arms.

**#1 — Turbo-Muon Triton NS, `ns_kind="triton"`, with `ns_steps 4`.**
The only candidate whose mechanism is *fewer launches* — our measured failure mode.
Pre-registered prediction **[S]**: **BPB ties** (their 144M table: 4 AOL iters ≈
5-iter Muon, 3.28 vs 3.28) and the **warm step time falls**. Since a tie deletes a
mechanism (AGENTS §1.2) while this arm's claim is *speed at equal quality*, the
verdict must be the speed rule: keep iff (a) no seed's BPB is worse than the
control mean + the control's own spread, **and** (b) ≥ 50 warm steps on a quiet
card show a drop above the run-to-run noise floor. Void check: the Triton path must
be COUNTED, never a silent fallback to classic (ADR-0019).

**#2 — NorMuon-style EMA row statistics replacing our instantaneous row norm.**
+`m` scalars, ~+5 dispatches/chain, landing in the slot ColRow already owns.
Pre-registered prediction **[S]**: **tie or a small win** — the published +11.31%
is over Muon-without-post-polar-normalization and we already run the row half.
Keep iff its mean BPB beats the control mean by more than the control's spread.
Caveat first: NorMuon's RMS-matching rescale needs a scalar — off-device it is an
AGENTS §1.3 violation, so it must be computed on device.

**#3 — `ns_steps 5 → 4` alone (zero code: `config.py:432`, `loop.py:483`).**
Run it **as the control for #1**, not instead of it: the published 4-iteration
result is AOL-conditioned, so 4 iters *without* AOL is an unvalidated quality
change (Table 1 gives 4 iters polar error 0.12 vs 5-iter 0.25, and only the
AOL+4-iter row is validated end-to-end). Prediction **[S]**: **tie at 2k** (the
earlier cost note predicted the same for 5→3); keep only on step time. It costs
nothing, so it is the correct first arm.

**#4 — Muon-VS (pre-orthogonalization variance scaling).** Conditional last slot:
strongest evidence (1.33× step-to-target at Llama-1.2B, ahead of NorMuon and
AdaMuon at 130M/300M) but predicted to **lose on our box** — quality up, step time
**worse** (+~14 dispatches/chain, +mn state = +363 MB fp32 at `medium`, +2.46 GiB
at b1). Run it only if #1-#3 leave the BPB-vs-AdamW gap unresolved.

## 3. Explicit rejections

- **Muon+ `2602.21545` — already shipped.** Its `col_row` arm is our default
  (`colrow_normalize`, −0.0428 BPB measured, `optim.py:322-330`). The remaining
  delta is `row_col` vs `col_row` (LLaMA-130M 18.68 vs 18.65 [A Table 6]) and
  `√(m/n)` vs our `max(1, m/n)^0.5` — both sub-noise at our spread.
- **Dion3 `2608.11612` — wrong bottleneck + wrong silicon.** The fractional update
  shrinks matmul *sizes* and adds ops; our NS is launch-bound, so the size
  reduction buys nothing while the ops cost. Its 6.5× was measured on a GH200 in a
  compute-bound regime where "neither communication nor host time is exposed"
  [A §8.2] — the opposite of our box. Its kernels are CuteDSL symmetric GEMMs
  targeting "Hopper and Blackwell … tensor memory (Blackwell)" [A §6], the package
  pinning `nvidia-cutlass-dsl==4.5.2` via `gram-newton-schulz==0.1.6` +
  `quack-kernels==0.5.0` [code] — **sm_120 has no tcgen05/TMEM**. *[S] I found no
  arch guard in the code, so "will not run" is inference from the stated target and
  TMEM use, not a read error message.*
- **Gram-NS (already an arm here) stays off** — 1.2-1.4× slower on this card [ours,
  `.bulba/memory.md:317`]. Dion3's speedup comes from its kernels, not its algebra.
- **Mousse `2603.09697` — memory, at exactly our shapes.** `L∈R^{m×m}`,
  `R∈R^{n×n}` per matrix [A Alg. 1]. Our dominant groups are `[64, 2730]` × 256
  and `[1024, 64]` × 513 [ours], so the `R` factor of a `[64, 5461]` b1 factor is
  5461² ≈ 29.8M elements **per expert** (~30.5B across the b1 bank). The
  single-sided variant halves it [A §5.3], but the paper's own ablation says the
  **left** factor is better than R — and across our two orientations the better
  side is not consistently the small side. Reject as specified.
- **MuonMax-Momo `2510.09827` — the benefit is a tuning-budget property.** 5%
  extra wall-clock buys LR robustness across ~3.5 decades [A §1, §5]; our A/B fixes
  the LR and judges at 2k steps, so the measured quantity does not exist here.
  Insurance if the b1 LR transfer fails, not a step-time arm.
- **AdaMuon `2507.11005` — 2mn state, dominated.** Its headline is vs Adam; the
  one paper that ran it against the field reports Muon-NSR/Muon-VS beating it at
  130M and 300M [A `2601.14603` §5.1].
- **SF-NorMuon `2605.23061` — right idea, wrong instrument.** A fixed-horizon 2k
  A/B cannot test a schedule-free claim; it trails horizon-aware cosine NorMuon by
  ~0.03 nats [A abstract] and needs weight decay at the fast iterate, which our
  Muon group does not carry [ours, `2026-10-16-sf-muon.md` §3]. Already DERIVE'd.
- **Lotus `2602.01233` — a different family.** GaLore-style randomized low-rank
  *gradient* projection with adaptive subspace switching [A abstract]. Not Muon, and
  it compresses the optimizer's gradient state — ours is already 1mn, and our
  low-rank structure lives in the **model** (rank-64 TSCT factors, `tsct.py:1-15`)
  where it cannot be re-projected.
- **Per-head NS** (adjacent, not on the brief's list): it **changes the update
  magnitude**, not just the direction — the adjusted lr becomes
  `lr·√(head_dim/in_features)`, ~3.5× smaller at 12 heads [code, `dion` README]. A
  direction REPLACE with an LR retune, not a drop-in.

## 4. Implementation sketch per shortlist item

| # | files to touch | LOC | gates it needs |
|---|---|---|---|
| **#1 Turbo-Muon Triton** | new `src/dormouse/train/ns_triton.py` (3 kernels + AOL + `zeropower_via_ns_triton`, 3-D batched, `[N,64,K]` in — the same orientation `zeropower_via_ns` already transposes to); `optim.py` `NS_KINDS` (+1 name) and one branch in `zeropower_via_ns`; `config.py` `NS_KINDS` mirror. **`ns_steps` needs no code** | ~130 | (a) **not** bit-identity — a *tolerance* gate against classic on saved b1/medium buffers: `‖outᵀout − I‖_F/rank` and cosine to the classic polar, because AOL is a different starting point, not a regrouping (unlike the Gram-NS gate in `tests/test_muon_gram.py`); (b) **default-path bit-identity**: `ns_kind="classic"` must produce the byte-identical tensor it does today; (c) COUNTED skip when triton is absent or the tensor is not CUDA — never a silent fallback (ADR-0019); (d) the 4-iteration arm's orthogonality bound, since it is the arm that changes the math |
| **#2 NorMuon row stats** | `optim.py`: a `[N, m]` `v` buffer on `_MomGroup` (built in `_build_groups`, so `load_state_dict` keeps working), the post-NS step in `batched_step_from_model` **and** `_step_one` (one definition, two callers — the file's own rule); `config.py` a `muon_norm: colrow\|nor` arm + the mirror tuple | ~45 | (a) a hand-computed gold step on a `[2,3]` matrix, 2 steps, pinning `v` and `O` as literals (the `test_dspark_gold` idiom); (b) default-path bit-identity for `colrow`; (c) checkpoint round-trip of the new state key — the state dict is per-parameter (`optim.py:363-366`), so a new key must survive save→load and a *pre-existing* checkpoint must fill it |
| **#3 `ns_steps 4`** | none — `config.py:432`, `loop.py:483` | 0 | no gate; the A/B is the gate |
| **#4 Muon-VS** | `optim.py`: a full-size `Γ` buffer per shape group + the update in `_momentum_update_`'s neighbourhood; `config.py` an arm name | ~60 | same three gates as #2, plus a memory assertion (the buffer is `mn`, so it must ride the existing chunking or it OOMs a 16 GB card at b1) |

## 5. What composes with what

- **Turbo-Muon's kernels compose with everything.** Same polynomial, better start,
  fused launches — a *representation* change, orthogonal to any direction change:
  NorMuon, Muon-VS, MuonMax and SF all keep the NS call intact. That is why #1 is
  first — it does not compete for another item's slot.
- **NorMuon × ColRow overlap, and it is not additive.** ColRow = column norm then
  row norm, instantaneous; NorMuon = row norm only, EMA statistics, plus an
  RMS-matching rescale. Same slot: adopting NorMuon *replaces* our row half. Expect
  a small and possibly negative transferable delta (§2 #2).
- **SF-NorMuon × NorMuon × our missing weight decay.** SF-NorMuon needs weight
  decay at the fast iterate Z for long-horizon stability [A `2605.23061` abstract];
  the port's Muon group carries none [ours, `2026-10-16-sf-muon.md` §3]. Landing
  NorMuon without deciding that is half a change — do not bundle them.
- **Muon-NSR/Muon-VS × ColRow is a conflict, not a stack.** Muon-NSR modulates
  **before** NS, arguing that "a nonuniform element-wise gate applied after the
  exact polar map need not preserve orthogonality" [A `2601.14603` §5.3] — an
  argument against ColRow. ColRow is nonetheless the port's largest measured win
  (−0.0428 BPB), so theory does not outrank the measurement; but an arm doing both
  is two changes at once and must not be run as one.
- **Dion3's fractional update × NorMuon.** Dion3's NorMuon flavor is literally
  NorMuon's row statistics + a norm-preserving rescale + selection
  [A `2608.11612` Alg. 4]. If #2 lands, its update rule becomes a smaller
  additional change — still rejected on launch count.
- **#1 × `ns_dtype="bf16"`.** The Triton kernels are bf16-native [code], so the
  Triton arm *subsumes* the bf16 NS rung instead of composing with it — the two
  must not ride in one A/B (`optim.py:47-50` prices that rung as unmeasured).

## 6. Sources

Papers, all ids resolved 2026-10-17 by `get_abstract`: `2512.04632` (Turbo-Muon)
· `2608.11612` (Dion3) · `2510.05491` (NorMuon) · `2605.23061` (SF-NorMuon) ·
`2601.14603` (Muon-NSR/VS) · `2510.09827` (MuonMax-Momo) · `2603.09697` (Mousse)
· `2602.21545` (Muon+) · `2507.11005` (AdaMuon) · `2602.01233` (Lotus) ·
`2608.28557` (Survey).
Code read: `microsoft/dion` (tree, `README.md`, `requirements_gns.txt`,
`dion/newton_schulz_triton.py`, `dion/dion2_triton.py`) ·
`thib-s/flash-newton-schulz` (README) · `zichongli5/NorMuon` (repo metadata only —
**the code was not read**). Turbo-Muon's HAL preprint is
`hal.science/hal-05390446`.
Repo: `port/src/dormouse/train/optim.py:67,106,153-214,303-354,450-605` ·
`port/src/dormouse/config.py:432,458-504` · `port/src/dormouse/train/loop.py:483` ·
`port/src/dormouse/model/tsct.py:1-15` · `configs/medium.toml` ·
`docs/reviews/2026-10-16-b1-profile.md:9-43` ·
`docs/research/2026-10-16-muon-orthogonalization-cost.md` ·
`docs/research/2026-10-16-sf-muon.md` · `.bulba/memory.md:317` ·
`docs/protocols/AB-PROTOCOL.md`.
Owed: a `docs-site/tools/manifest.mjs` line for this file (AGENTS §1.8).

## 7. Open questions

1. **Does a Triton NS kernel actually reduce wall time on THIS card?** Every
   published number for it is Hopper/datacenter [A, code]. The mechanism (fewer
   launches) is right for a dispatch-bound step, but the claim is unmeasured here —
   the arm's first deliverable is a warm-step timing, not a BPB.
2. **What is the run-to-run step-time noise floor at `medium`?** #1's verdict rule
   depends on it and no document states it.
3. **Which orientation does NorMuon's row statistic live in for us?** `O` returns
   in the original orientation, so `m` is 2048 for the U factors and 64 for the V
   ones [ours]; the paper only reasons about `m ≤ n`.
4. **Does the Triton autotune cost matter?** `@triton.autotune` benchmarks its grid
   on first call ("may take several minutes" [code]) — a one-time cost the A/B
   budget must include.
