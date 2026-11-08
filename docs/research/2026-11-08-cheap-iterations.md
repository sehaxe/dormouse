# Cheap iterations: can a low-rank residual buy depth for a few percent of FLOPs?

**Date:** 2026-11-08. **Mode:** read-only research, CPU only; no GPU run, no config or model code changed. **Scope:** one file, `docs/research/2026-11-08-cheap-iterations.md`.
**Question (owner):** can an ITERATION be made much cheaper than a full block pass — so that "extra depth" costs a few percent of FLOPs instead of a full block — and would it buy quality?
**Proposal under test:** between (or instead of) full block passes, apply a low-rank residual to the stream, `h += B(A(h))`, rank `r=64`, optionally conditioned on the iteration index, shared across depth. Cost ≈ `2·d·r` MAC/byte vs the block's **567 MFLOP/byte**.
**Marks:** **[T]** table read · **[A]** abstract only · **[C]** computed here from stated numbers (formula shown) · **[ours]** measured in this repo · **[S]** SPECULATION, no direct measurement.

**The live model this is priced against.** `port/configs/flagship3_dense.toml`: 12 unique layers, `d_model=832`, `d_ffn=3072`, `n_heads=13`, `head_dim=64`, `block_form="transformer"`, `max_iter=1`, 95.0M params, byte vocab 256, 1 byte = 1 token. The config's own ledger: **block 567.41 MFLOP/byte**, attention core 122.68, head 1.28, **total 691.37** (fwd+bwd at b1×s1024) [ours, `flagship3_dense.toml:25-26`]. So a "block pass" in the brief's sense is the 12-layer body's 567.41; the full fwd+bwd bill including the attention core is 691.37. The live model is dense, not the KDA/MoE loop block; variants (b), (c) and (e) below presuppose a tied block and are priced for the loop arm, not for this control.

---

## TL;DR (RU)

1. **Арифметика правильная: итерация действительно почти бесплатна.** Rank-64 residual — 0.213 MFLOP/byte в forward-проходе и 0.639 в обучении на итерацию, это 0.038% / 0.113% от блока 567.41. При k=4: 1.8% блока в обучении для h=256 и 0.45% для r=64. Даже k=16 укладывается в 2–7%. Вопрос не в цене FLOPs.
2. **Но это не глубина в том смысле, который покупает качество.** Нет token mixing: модуль не может изменить, какие позиции на какие смотрят. Веса общие: k итераций — композиция одной и той же функции, а не k разных слоёв. И жёсткий потолок ранга: при общем `B` все k итераций остаются в `h0 + im(B)`, размерность ≤ r; k не добавляет измерений, а выбирает точку в том же r-мерном срезе. Это математическая форма «refine, not add capability».
3. **Ровно этот механизм при ≤200M в литературе не измерен.** Ближайшее — полный блок: φ=0.46 (каждая рекурренция стоит 0.46 уникального блока), а 4-я итерация полного блока на 1B даёт **0.006 CE** [ours]. Cheap fixes (per-iteration LoRA, iteration embeddings, residual scaling) возвращают ≤0.01–0.02 BPB и никогда не переворачивают знак. Наш gap 0.13–0.37 BPB — это не tying, и дешёвая итерация его не лечит.
4. **Варианты (b), (c), (e) — не дешёвые итерации.** Это способы дифференцировать полный проход: per-iteration LoRA, FiLM/гейтинг, residual scaling. Они не удешевляют шаг глубины; они пытаются сделать tied-проход похожим на untied. Только (a) и (d) действительно стоят 0.1–0.9% блока за итерацию.
5. **Единственный эксперимент, который стоит своих ~95 минут:** dense 12L контроль vs те же +k cheap thinking modules (h=256, k=4), в двух реализациях — shared B и per-iteration B — 3 arms × 3 seeds × 30 MB. Pre-registered pass: любой из двух вариантов ≥0.03 BPB лучше контроля; если оба — побеждает тот, что лучше другого на ≥0.01; иначе lane закрыт. Ожидание: tie, потому что даже полный 4-й проход стоит 0.006 CE, а дешёвый модуль слабее полного блока.
6. **Честный вердикт:** дешёвая итерация не может добавить способность — только уточнить то, что уже посчитали 12 слоёв. Единственный ненулевой сценарий — test-time compute (почти бесплатные итерации на инференсе), но CART и DCP говорят, что лишняя глубина сверх обученной деградирует. Покупать здесь нечего; можно сэкономить FLOPs, но не купить качество.

---

## 1. The arithmetic: it is cheap, and that is not the problem

### 1.1 The bill

Convention: 2 FLOP per MAC; training = 3× forward. The brief's `2·d·r` MAC is **0.106 MFLOP/byte** if one counts 1 FLOP/MAC, or **0.213** under the standard 2 FLOP/MAC; the brief's "~0.04%" is the forward-only share. The training share is 3× larger, which is the number that matters for an A/B.

| variant | params | fwd MFLOP/byte / iter | train MFLOP/byte / iter | % of block 567.41 (train) | % of total 691.37 (train) |
|---|---:|---:|---:|---:|---:|
| (a) shared low-rank, r=64 | 0.110M | 0.213 | 0.639 | 0.113% | 0.092% |
| (a/d) 2-layer MLP, h=256 | 0.431M | 0.852 | 2.556 | 0.450% | 0.370% |
| (b) LoRA, attention-only, r=64 | 5.11M / iter | +10.2 | +30.7 | +5.4% | +4.4% |
| (b) LoRA, all 12 maps, r=64 | 11.1M / iter | +22.2 | +66.6 | +11.7% | +9.6% |
| (c) FiLM, 2 sites × 12 layers | 0.040M / iter | 0.040 | 0.120 | 0.021% | 0.017% |
| (e) residual scale, 2/layer/iter | ~0.0001M | ~0 | ~0 | ~0 | ~0 |

**Reading.** (a) and (d) are the only rows that make an *iteration* cheap. (b) and (c) are deltas on top of a **full** 567 MFLOP/byte pass: they add 5–12% (LoRA) or 0.02% (FiLM) to a pass that still costs 567. (e) adds nothing but also computes nothing. The 0.04% in the brief is the forward-only version of (a); in training it is 0.11%, and at k=4 it is 0.45% (r=64) or 1.8% (h=256). All still "a few percent", so the cost claim survives [C].

### 1.2 The rank ceiling [C]

The proposal is `h_{i+1} = h_i + B σ(A h_i)`, with `A ∈ R^{r×d}`, `B ∈ R^{d×r}`. Every update `B σ(·)` lies in `im(B)`. By induction:

`h_k = h_0 + Σ_{i<k} B σ(A h_i) ∈ h_0 + im(B)`, `dim im(B) ≤ r`.

So the reachable set of k iterations is an affine subspace of dimension **≤ r**, independent of k. Extra iterations choose a different point in the same 64-dimensional slice; they do not add dimensions. If the head is linear, the logits change by a rank-r term. This is not an implementation detail; it is the definition of a shared low-rank residual. Two ways to escape it, both of which change the mechanism:

- **Per-iteration `B_i`** (not shared): the union is `h_0 + span(B_1, …, B_k)`, dimension ≤ `k·r` (up to `d`). Still cheap (k × 0.110M params for r=64, or 1.07M for h=256 at k=4), but it is no longer "shared across depth".
- **Nonlinearity after `B`**: `h_{i+1} = h_i + σ(B A h_i)`. The elementwise σ can leave `im(B)`, so the reachable set can be full-dimensional. But then the update is a pointwise nonlinear map with a rank-r pre-activation, not a low-rank linear residual. It is a different variant, and it still has no token mixing.

The owner's `h += B(A(h))` with no activation is the first case exactly; with an activation between A and B it is still the first case (the image is in `im(B)`); only an activation after B escapes. The report prices the standard adapter (σ between A and B) because that is what "low-rank residual adapter" means in the literature [C].

### 1.3 No token mixing [C/S]

A transformer block is attention (mixes positions) + FFN (per-position). The cheap module has no attention. Its output at position p is a function of `h[p]` only; it cannot retrieve a different context, compose a new attention pattern, or move information between positions. The 12 layers already did the token mixing; the cheap module can only post-process the result. This is the difference between **refinement** (recalibrate, denoise, sharpen a per-position decision) and **capability** (compute a function the 12 layers could not). The rank ceiling is the linear-algebra version of the same statement; the no-token-mixing argument is the architectural version. Either one alone is enough to put the mechanism below a full block in function class [C]; the inference that it therefore buys less quality is [S] but a strong one.

### 1.4 What "a full block's 4th iteration" is worth at our scale [ours]

The repo's own per-iteration CE (full KDA/MoE block, deep supervision):

| run | iter 1 | iter 2 | iter 3 | iter 4 | marginal 2/3/4 (CE) |
|---|---:|---:|---:|---:|---|
| 1B, step 26 000 (`b1first_resume2.log:906`) | 2.1503 | 1.8668 | 1.8163 | 1.8107 | 0.284 / 0.050 / **0.006** |
| medium, step 34 000 (`medium_long.log:355`) | 2.1536 | 1.8070 | 1.7478 | 1.7131 | 0.347 / 0.059 / **0.035** |

At 1 byte/token, 0.0056 nats/byte ≈ **0.008 BPB**. That is already at the bottom of the published tying-fix bar (0.01–0.02 BPB) and below the 30 MB noise (0.02–0.03). A cheap module is strictly weaker than the full block that produced this marginal (no attention, lower rank, shared), so its expected marginal is **< 0.008 BPB** [S]. This is the single strongest quantitative argument in the report: even the expensive 4th pass is worth less than the screen can resolve; the cheap version cannot be worth more.

Caveat: the 1B/medium numbers are the KDA/MoE loop block, not the 95M dense 12L. The dense model's 13th layer could be worth more or less; no 13L measurement exists. But the depth–width evidence says 12 layers at d=832 is near `D_crit ≈ 14` (extrapolated from 2601.20994), i.e. on the flat part of the depth curve [S]. The prior is small.

---

## 2. The design space: five variants

`Cheap?` = does it reduce the cost of an extra depth step? `No` rows are differentiators of a full pass, not cheap iterations.

| # | Variant | Cheap? | Params | fwd / train MFLOP/byte per iter | What it could plausibly add | Priority |
|---|---|---|---:|---|---|---|
| a | Shared low-rank residual after the stack, `h += B σ(Ah + e_i)`, r=64 | **yes** | 0.110M | 0.213 / 0.639 | per-position calibration/denoising; a small fixed-point refinement; a near-free test-time knob. No new context. | test only as the cost-optimised subset of (d) |
| b | Per-iteration LoRA on the block's projections, `W_i = W + B_i A_i` | **no** | 5.11M (attn) / 11.1M (all) per iter | +10.2 / +30.7 (attn); +22.2 / +66.6 (all) | differentiation of a tied block: each pass specialises; recovers part of the tying penalty | skip (literature null; not a cheap pass) |
| c | Iteration-index FiLM/gating, `γ_i ⊙ LN(x) + β_i` | **no** | 0.040M per iter | 0.040 / 0.120 | the shared block changes its effective function per pass; matters only when depth is extrapolated | skip at fixed training depth |
| d | 2-layer MLP "thinking" module, `h += W2 σ(W1 LN(h + e_i))`, h=256 | **yes** | 0.431M | 0.852 / 2.556 | the strongest cheap refiner: a per-position nonlinear iteration; could implement a small constraint solver/denoiser. Still no token mixing. | **test — the upper bound of the family** |
| e | Skip-connection scaling per iteration (DeepLoop) | **no** | ~0.0001M | ~0 / ~0 | a numerical fix for tied depth: keeps the unrolled variance bounded; lets existing depth train | skip unless the loop returns |

### (a) Shared low-rank residual after the stack

- Mechanism: `h_{i+1} = h_i + B σ(A h_i + e_i)`, `i = 1..k`, A/B shared, `e_i` a learned per-iteration embedding (`k·d` params).
- Params: `2dr + kd` = 0.110M (r=64, k=4) or 0.431M (h=256). FLOPs: 0.213/0.639 per iter (r=64); 0.852/2.556 (h=256).
- What it could add: per-position calibration/denoising, a small fixed-point refinement of the final state, a nearly free test-time compute knob. No new context, no new token mixing.
- Evidence: **`[S]` no direct measurement at ≤200M.** The closest are the full-block loop rows in §3 and the rank-ceiling argument in §1.2. The prior is below the bar: 0–0.005 BPB at r=64, 0–0.01 at h=256.
- Priority: the r=64 version is the owner's literal proposal; test it only after the h=256 upper bound. If h=256 fails, r=64 is dead a fortiori.

### (b) Per-iteration LoRA on the block's projections

- Mechanism: `W_i = W + B_i A_i` for the shared block's maps; each iteration gets its own low-rank delta. The block pass stays a full pass.
- Params: attention-only r=64: 4 maps × 12 layers × `2dr` = 5.11M per iteration (5.4% of 95M); all 12 maps: 11.1M per iteration (11.7%). At r=16: 1.28M / 2.78M.
- FLOPs: the pass is still 567; the adapter adds 5.4–11.7% in training. **It is not a cheap iteration.**
- What it could add: differentiation of a tied block — each pass specialises, recovering part of the tying penalty.
- Evidence: **`[T]` null at Qwen3-1.7B, T=3** ("training losses nearly match the shared model… within two points, neither ahead", 2610.00673v1 §4.4); **`[T]` recovery at Gemma 2B→1B uptrain**, rank-512 depth-wise LoRA 58.4 vs 58.6 full 2B, SVD init +6.5 points (2410.20672v3 §3.3/3.5). No from-scratch ≤200M measurement. The prior is the tying penalty itself: 0.01–0.02 BPB at best, and it does not flip the sign.
- Priority: skip. It is the literature's fix for a problem the repo has already measured as not the main gap (0.13–0.37 BPB vs 0.015–0.045 published tying penalty).

### (c) Iteration-index FiLM/gating

- Mechanism: per-iteration scale/shift or gate on the shared block's norms/activations: `y = γ_i ⊙ LN(x) + β_i`; γ, β per iteration.
- Params: `2 × 2d × 12` ≈ 0.040M per iteration (0.042%); FLOPs 0.12 train MFLOP/byte per iteration (0.02% of the block). It rides on a full pass.
- What it could add: the shared block can change its effective function per pass without new full weights; useful when depth is extrapolated beyond training.
- Evidence: **`[T]` Qwen3-0.6B at inference depth 56**: branch-gating 38.46 vs base loop 36.59, AdaLN 38.01; "much smaller or negative" at other depth allocations; gains at extrapolation, not training depth (2609.36636v1 §5.2). **`[T]` one scalar input re-injection at Qwen3-1.7B** turns steps 2/3 from +0.01/+0.02 nats worse to −0.02 better (2610.00673v1 §4.1). At fixed training depth (the loop arm runs fixed 3), the prior is ~0.
- Priority: skip as a quality arm. It is a conditioning knob for extrapolation, and the repo does not extrapolate at training time.

### (d) Small "thinking" module (2-layer MLP) iterated

- Mechanism: `h_{i+1} = h_i + W2 σ(W1 LN(h_i + e_i))`, W1: d×h, W2: h×d, shared across k; h=256 (or 512).
- Params: `2dh + 2d + kd` = 0.431M (h=256, k=4) or 0.857M (h=512). FLOPs: 0.852/2.556 per iter (h=256), k=4 = 1.8% of the block in training.
- What it could add: the strongest version of the cheap-iteration idea. A per-position nonlinear refiner with a wider bottleneck than (a); could implement a small iterative algorithm (a per-position constraint solver, a denoiser, a calibration map). Still no token mixing, still shared, still rank-h in the image (unless the nonlinearity is after W2).
- Evidence: **`[S]` no direct measurement at ≤200M.** The closest are the full-block loop rows and the negative test-time-depth results (CART, DCP). The prior is below the bar: 0–0.01 BPB.
- Priority: **the one to test**, because it is the upper bound of the family. If it fails, (a) and the r=64 version cannot succeed. If it passes, the next question is how low h can go.

### (e) Skip-connection scaling per iteration (DeepLoop)

- Mechanism: per-iteration residual scale `α_i`, `β_i` on the skip path; `h = α_i h + β_i f(h)`, with the DeepLoop exponents `α = (2N)^½`, `β = (8N)^−½` or learned.
- Params: 2 per layer per iteration (~0.0001M); FLOPs ~0 (elementwise, foldable).
- What it could add: a numerical fix for tied depth — keeps the unrolled network's variance bounded, so the tied passes train. It does not create depth; it lets the existing depth work.
- Evidence: **`[T]` GPT-2 small 124M, 50B tokens**: R=3 2.8077→2.7917, R=5 2.7910→2.7679, R=7 2.7700→2.7514, R=1 +0.0004 (2607.13491v2 Table 5.1). Within-loop, not iso-FLOP vs dense. The repo's depth-recurrence report already prices this at ~0.02 BPB and says it does not flip the sign.
- Priority: skip unless the loop returns. It is a fix, not a capability.

---

## 3. What the literature says

### 3.1 The measured rows

`Match` — what is held equal. `Full block` rows are the upper bound: they measure a *full* extra block, so the cheap module cannot exceed them.

| # | Paper (id, version, date) | Scale / setup | Mechanism | Measured Δ (metric) | Table / § | Match | Mark | What it says for the cheap iteration |
|---|---|---|---|---|---|---|---|---|
| 1 | Iso-Depth, 2604.21106v3/v4 (2026-05-07) | 20 eff. layers, r∈{1,2,4,8}, 30–400M, 116 runs | full-block loop | φ = 0.46 [0.41, 0.53]; looped frontier +0.03…0.12 nats worse; gap widens at low budgets | Table 2/3, §4.2 | iso-FLOP | [T] | each extra *full* block is worth 0.46 of a unique one; a cheaper, weaker block is below that |
| 2 | MoR, 2507.10524v3 (2025-07-14) | 135M/2e18; 360M/2e18 | loop + routing | 135M: 3.0922 → 3.1077 (**+0.0155**); 360M: 3.3785 → 3.1026 (**−0.2759**) | Table 9/15 | iso-FLOP | [T] | loops lose at our scale; the crossover is above us |
| 3 | SMELT, 2609.01343v3 (2026-09-01) | 200M, S≈85%, 96–504 experts | mid-stack loop + huge MoE | 1.9445 → 1.9257 nats (**−0.019**); DCLM Core 24.92→27.57 | Table 3.5 | iso-FLOP+params+KV | [T] | the only ≤200M positive loop, and it needs a huge MoE; no dense transfer |
| 4 | LOOM, 2610.01153v1 (2026-10-01) | 700M, 5 loops; ~100M native loop | loop + per-loop routers | 700M: 18.36→16.54 ppl; ~100M: 26.26→27.61 (**+1.35**) at 3× | Tables 2/3 | near-iso-FLOP (700M); none (~100M) | [T] | the win is at 700M with per-loop routers; at 100M the loop loses |
| 5 | Sparse Layers, 2605.09165v2 (2026-05-09) | 168M looped / 246M base, 1e18 | looped vs looped-MoE | Base 38.7 / Looped **37.4** / Looped-MoE **39.6** Core9 | Table 4 | iso-FLOP | [T] | looped loses; MoE recovers, not the loop |
| 6 | DeepLoop, 2607.13491v2 (2026-07-15) | GPT-2 small 124M, 50B tok | per-iteration residual scale | R=3 **−0.016**, R=5 **−0.023**, R=7 **−0.019** nats; R=1 +0.0004 | Table 5.1 | params, not iso-FLOP | [T] | a fix for tied depth, not a capability; ~0.02 BPB |
| 7 | Closing the Loop, 2610.00673v1 (2026-09-30) | Qwen3-1.7B, T=3 | per-step LoRA; one scalar re-injection | per-step LoRA **null** (within 2 points); scalar: steps 2/3 go from +0.01/+0.02 to **−0.02** nats | §4.1–4.4 | conversion, not from scratch | [T] | the cheap tying fixes are at most ~0.02 nats, and LoRA is a null |
| 8 | Relaxed Recursive, 2410.20672v3 (2024-10-28) | Gemma 2B→1B, 60B uptrain | rank-512 depth-wise LoRA | 58.4 vs 58.6 full 2B; SVD init +6.5 points | §3.3/3.5 | uptrain, params | [T] | recovery of an existing model, not a from-scratch win |
| 9 | What Makes Recurrence Effective, 2609.36636v1 (2026-09-29) | Qwen3-0.6B, inference depth 56 | iteration gating / AdaLN | BG **38.46** vs base 36.59; AdaLN 38.01; "much smaller or negative" at other depths | §5.2 | inference depth | [T] | iteration conditioning is an extrapolation tool; ~0 at training depth |
| 10 | CART, 2606.01495v2 (2026-06-03) | d=1024, R=2–10, ~1B tok, 3 seeds, RTX 3090 | variable-R loop | more iterations than trained **consistently degrade** HellaSwag/LAMBADA/PIQA | abstract | param parity | [A] | at our scale and hardware, extra depth beyond training is negative |
| 11 | Depth Control Protocol, 2609.19934v1 (2026-09-17) | 542.8M recurrent | inference-depth decomposition | naive truncation 1.4251 nats; **−2.8% "distinct iterations"** | abstract | diagnostic | [A] | distinct iterations contribute ≈nothing |
| 12 | Truncate or Not?, 2610.26030v2 (2026-10-20) | ~600 runs | full vs truncated BPTT | optimum **2 fully trained loops**; 3rd/4th regress | abstract | mixed | [A] | extra depth beyond 2 loops regresses |
| 13 | Shared Memory in Looped Transformers, 2610.02383v1 (2026-10-01) | 1B, R=5 | shared KV across recursions | 10.23 vs 11.35 PPL vs single-pass; **3→5 recursions 10.45→10.32** (0.13 PPL for +67% decode FLOPs) | Table 1 | params | [T] | even a full block's extra iterations are diminishing at 1B |
| 14 | Universal Transformers, 1807.03819v3 (2018-07-10) | ~65M, WMT14 | full block ×6 | UT **28.9** vs Transformer **28.0** BLEU | Table 7 | iso-depth | [T] | MT only, not LM; does not transfer to byte BPB |
| 15 | MoEUT, 2405.16039v2 (2024-05-25) | 44M–719M | MoE + loop | "slightly outperforms parameter-matched"; large margin at equal compute — figure only | abstract/§3 | params | [A/F] | the paper's abstract says no *previous* shared-layer transformer is competitive in parameter-count-dominated LM; MoEUT needs MoE to recover |

**What the table says as a whole.** At ≤200M with a dense control, the only positive loop is SMELT's −0.019 nats, and it is paid for by 96–504 experts per layer. Everything else is a loss at iso-FLOP (MoR 135M, iso-depth, Sparse Layers, LOOM ~100M). The cheap fixes are measured at 1.7B/2B/124M and are at most 0.02 nats/0.02 BPB. A shared rank-64 pointwise module is strictly weaker than every mechanism in this table: it has no attention, no per-iteration full weights, and a hard rank ceiling. Its prior cannot be above the table's best full-block result, and the full-block result at our scale is already at or below the 30 MB noise.

### 3.2 What is not evidence

- **Looped Transformers as Programmable Computers** (2301.13196): expressivity of a looped *full* transformer; no BPB at 200M. `[S]` — it says a loop can simulate algorithms, not that a rank-64 pointwise map buys language quality.
- **Deep Equilibrium Models** (1909.01377): implicit depth via a fixed point; no 200M BPB A/B. `[S]`.
- **Soft MoR** (2607.00774): vision, T=3 is 3× depth, not iso-FLOP; does not transfer. `[A]`.
- **Dual-Path** (2605.30202v2): "at fixed FLOPs a looped model has strictly less capacity" `[A]` — the φ<1 law in words; it does not price a cheap module, but it sets the direction.
- **Hyperloop** (2604.21254v3) and **AdaPonderLM** (2603.01914v2): parameter-efficiency and inference-cost wins, not BPB at iso-FLOP; not evidence for the cheap iteration.
- **The repo's own loop numbers** (1B/medium per-iteration CE) are for a full KDA/MoE block, not a cheap module. `[ours]` — they bound the *value of a full block*, not the value of the cheap one.

### 3.3 Per-variant verdict relative to the bar

The bar is the repo's own published-tying-fix range, **0.01–0.02 BPB** (2026-11-06-depth-recurrence TL;DR 4). The 30 MB noise is **0.02–0.03 BPB**.

| Variant | Closest evidence | Measured at ≤200M? | Expected ΔBPB at 95M | vs 0.01–0.02 bar | vs 0.02–0.03 noise | Verdict |
|---|---|---|---|---|---|---|
| (a) r=64 | none; rank ceiling [C] | **no** `[S]` | 0 … −0.005 | below | below | skip unless (d) passes |
| (a/d) h=256 | none; full-block rows bound it | **no** `[S]` | 0 … −0.010 | below | below | **test** |
| (b) per-iter LoRA | 2610.00673 §4.4 (null); 2410.20672 §3.5 (uptrain) | yes at 1.7B/2B, not ≤200M from scratch | −0.005 … −0.02 | at | below/at | skip (not a cheap pass) |
| (c) iteration FiLM | 2609.36636 §5.2; 2610.00673 §4.1 | yes at 0.6B/1.7B, not ≤200M | ~0 at training depth | below | below | skip |
| (e) residual scale | 2607.13491 Table 5.1 | yes at 124M | −0.006 … −0.02 (unit-dependent) | at | at/below | skip (a fix) |

**Above the bar?** None of the five variants is *above* the 0.01–0.02 BPB bar. (b) and (e) are *at* it; (a), (c) and (d) are below it. **None flips the sign of the loop gap.** That is the answer to deliverable 3.

---

## 4. The prior from this repo

### 4.1 The 2026-11-06 verdict

> "The published tying penalty at our scale is ~0.015–0.045 BPB… Making the tie 'cheaper' (per-iteration LoRA, norms, iteration embeddings) can recover at most ~0.01–0.02 BPB and never flips the sign." — `2026-11-06-depth-recurrence.md`, TL;DR 4.

The cheap iteration is not a tying fix, but it is a *strictly weaker* function class than the full block that the tying literature measures: no attention, lower rank, shared weights. So the same bar applies, and the cheap iteration is below it — except (b) and (e), which are *at* it and are not cheap passes anyway.

### 4.2 The 30 MB resolution

- **1σ ≈ 0.02–0.03 BPB at 30 MB** (interpolation, not a measured power analysis; `2026-11-08-lr-schedule.md` §4.1).
- Same-config run-to-run RMS at 30 MB: **0.0048 BPB** as reported (CUDA nondeterminism; the three pre-decay deltas are +0.0073/+0.0002/+0.0039; `2026-11-08-lr-schedule.md` §1.2).
- 3 seeds → SE ≈ **0.012–0.017**; a 0.03 delta is ~2σ. A 0.02 delta is ~1.2–1.7σ: suggestive, not a result.
- Therefore the expected effect of the best cheap variant (0–0.01 BPB) is **at or below the noise floor**. The experiment is a kill test, not a discovery test. A plan that claims a 0.01 effect at 30 MB is claiming 0.6–0.8σ at 3 seeds (0.3σ at 1 seed).

### 4.3 The gap arithmetic

- Published tying penalty at our scale: **0.015–0.045 BPB**.
- Repo's measured loop gap: **0.13–0.37 BPB** — 3–10× larger.
- Therefore the gap is the block (KDA/Engram/256-expert MoE) and the 1-layer effective depth, not the tie `[S]`, consistent with `2026-11-06-architecture-at-100m.md`. **A cheaper iteration cannot recover a gap that the tie did not create.**

---

## 5. The single most promising experiment

### 5.1 Why this one

The strongest version of the family is a 2-layer MLP at **h=256**, **k=4**, after the 12-layer dense stack, in **two implementations**: shared `W2` (the literal "shared across depth") and per-iteration `W2_i` (the escape hatch from the rank ceiling, still only 1.1% params). If the escape-hatch version cannot move BPB, the shared r=64 version cannot; if the escape hatch works but the shared version does not, the rank ceiling is the binding constraint and the owner's cost target is the wrong implementation. Testing both in one screen is the cheapest way to distinguish "the mechanism is worthless" from "the shared parameterisation is the problem".

The dense control is the best-case stream: 12 unique layers, no tying penalty, full attention. A cheap module has the best chance to add value there. If it fails there, it cannot rescue a loop, which is a weaker stream.

### 5.2 Exact config

- **Base:** `port/configs/flagship3_dense.toml` (12L, d=832, d_ffn=3072, 95.0M) + data `port/configs/mix_refetch.toml`; b8×s1024; byte vocab 256; bf16; `opt=mix`; `muon_lr_mult=12`; `warmup_tokens=0`; `decay_frac=0.1`; `ckpt_every` off; eval **196 608 B** at offset 0; **3 662 steps = 30 MB**.
- **Module:** `h_{i+1} = h_i + W2_i · SiLU(W1 · LN(h_i + e_i))`, W1 832×256, W2 256×832 (shared in C1; per-iteration `W2_i`, i=1..4, in C2), `e_i` 4×832. W1 Kaiming, W2 zero (LoRA convention: the module starts as an exact no-op; the first gradient moves W2, then W1), `e_i` zero. Insert in `_readout` before `self.norm` (`port/src/dormouse/model/model.py:1405`), so both stack forms share it.
- **Params:** 0.431M shared (+0.45%); 1.07M per-iteration (+1.13%). **FLOPs:** 10.2 train MFLOP/byte at k=4 (+1.80% of 567.41; +1.48% of 691.37), the same for both.
- **Arms:** C0 = k=0; C1 = shared `W2`, k=4; C2 = per-iteration `W2_i`, k=4. **3 seeds each = 9 runs.**
- **Cost:** 30 MB at the measured 49.8 kB/s = **10.0 min/run**; 9 runs = 90 min; + eval ~3 min; + module overhead 1–2 min → **≈95 min** (±10%). If the module adds >3% step time, the "few percent" premise fails on this card; log `gpu_step` and the update norm.
- **Diagnostics:** mean L2 norm of the module update / stream norm, per iteration; train CE and held-out BPB separately; `gpu_step`. The update-norm diagnostic distinguishes "no quality effect" from "the module never ran" (the ADR-0019 class).

### 5.3 Pre-registered pass rule

- **PASS (per-iteration):** `mean(C2) ≤ mean(C0) − 0.03` **and** `mean(C2) ≤ mean(C1) − 0.01` **and** the sign holds in ≥2/3 seeds. The rank ceiling was binding; the escape hatch works. Then a **120 MB confirm**, 3 seeds, C0 vs C2, threshold 0.02; cost ≈4 h.
- **PASS (shared):** `mean(C1) ≤ mean(C0) − 0.03` **and** `mean(C2) ≥ mean(C1) − 0.01`, same seed rule. The shared version is the winner and the rank ceiling is not binding; confirm C1 at 120 MB.
- **KILL:** both `C1` and `C2` are `> mean(C0) − 0.03`, **or** the update-norm ratio <1% at step ≥1000, **or** train CE improves while held-out BPB does not. Close the lane; no 1.5 GB run.
- **AMBIGUOUS:** one arm in `−0.03…−0.02` and the other below it: not a result at 30 MB; the 120 MB confirm is allowed only because the mechanism ordering is right, but the prior says it regresses to zero.

### 5.4 What result would kill the idea

1. **A tie at k=4.** The cheap iteration adds no measurable quality.
2. **In a follow-up dose arm, `k=4` not better than `k=1`.** The "iteration" part is dead; a single cheap adapter is just a wider per-position head. (Add a k=1 arm only if the first screen is ambiguous; it doubles the cost.)
3. **The update norm collapses to ~0.** The model learns a soft skip — the LatentLoop failure mode, where the learned writeback coefficients settled at 0.199/0.160, i.e. the model damps the update by ~5× (2610.19975 §3.1/Fig. 6). The module is a no-op.
4. **A win at h=256 that disappears at r=64.** The mechanism works only at a rank the owner did not ask for; the literal r=64 proposal is dead, and the follow-up is a cost/capability trade, not the original idea.

### 5.5 If the owner wants the loop version

Replace the **3rd full pass** of a `1 block × 3` loop with the same module (k=1, h=256): 3 seeds, 30 MB; pass = `BPB ≤ +0.02` vs the 3-pass loop **and** `gpu_step` −≥30%. Cost ≈60 min (3+3 runs). This is a throughput test, not a quality test, and it should not run before §5.2: the 3rd full pass is worth **0.05 CE = 0.073 BPB** at 1B [ours, `b1first_resume2.log:906`], so replacing it is a real quality bet; a cheap module that adds nothing loses up to 0.073 BPB. The prior says it will not recover most of it.

### 5.6 The cost of being wrong

95 min for the screen. If it passes, ~4 h for the 120 MB confirm. If that passes, **≈25–27 h per arm** for the 1.5 GB run (3 seeds; 1.5 GB at 49.8 kB/s = 8.4 h/seed). The screen is ~3% of the two-arm confirm cost; that is the right order.

---

## 6. Verdict (RU)

**Короткий ответ: нет, дешёвая итерация не покупает качество.** Она покупает *дешёвую итерацию*: 0.1–0.9% блока за проход. Но способность, которую можно купить за 0.1% FLOPs, по построению меньше способности полного блока, а полный блок на 4-й итерации уже стоит 0.006 CE (0.008 BPB) — ниже шума 30 MB. Три причины, в порядке силы:

1. **Потолок ранга.** При общем `B` все k итераций живут в `h0 + im(B)`, dim ≤ r. k не добавляет измерений; он выбирает точку в том же срезе. Это не баг реализации, это определение shared low-rank residual.
2. **Нет token mixing.** Модуль не может изменить, какие позиции на какие смотрят. 12 слоёв уже сделали mixing; модуль только постобрабатывает результат. Это refinement, не capability.
3. **Литература и наш собственный замер.** φ=0.46 для полного блока; 4-я итерация 0.006 CE; cheap fixes 0.01–0.02 BPB и никогда не переворачивают знак; наш gap 0.13–0.37 BPB — не tying.

**Что остаётся ненулевым:** test-time compute. Дешёвая итерация почти бесплатна на инференсе, и если модуль выучит сжимающее отображение (fixed point), больше итераций может улучшить калибровку или трудные токены. Но CART (d=1024, ~1B токенов, 3 seeds) говорит, что больше итераций, чем при обучении, стабильно ухудшает; DCP говорит, что «различные итерации» дают −2.8%. При ≤200M положительного измерения нет. Это ставка, а не план.

**Что делать вместо:** если цель — качество при 95M/1 GB, очередь — данные (1 GB), ширина (10L×912), ByteFlow; глубина уже у порога `D_crit ≈ 14`. Дешёвая итерация — способ дешевле сделать то, что и так почти не даёт качества. 95-минутный kill стоит провести, чтобы закрыть файл; 1.5 GB confirm — только если 30 MB пройдёт pre-registered rule.

---

## 7. VERIFIED / SPECULATION

### VERIFIED (sourced or measured)

| # | Claim | Source |
|---|---|---|
| V1 | Block 567.41, attention 122.68, head 1.28, total 691.37 MFLOP/byte (fwd+bwd, b1×s1024) | `[ours]` `port/configs/flagship3_dense.toml:25-26` |
| V2 | (a) r=64 = 0.213 fwd / 0.639 train MFLOP/byte per iter; (d) h=256 = 0.852 / 2.556; (b) attn-only = +10.2 / +30.7; (b) all maps = +22.2 / +66.6 | `[C]` §1.1, formulas `2dr`, `4dr`, `6·2dr`, `2dh`, `4dh`, `6·2dh` |
| V3 | Shared `B` ⇒ every k-iteration state is in `h0 + im(B)`, dim ≤ r | `[C]` §1.2, induction |
| V4 | 1B per-iteration CE 2.1503/1.8668/1.8163/1.8107 → marginal 0.284/0.050/0.006 | `[ours]` `b1first_resume2.log:906`, quoted in `2026-10-23-shared-memory-loops.md:158` |
| V5 | medium per-iteration CE 2.1536/1.8070/1.7478/1.7131 → marginal 0.347/0.059/0.035 | `[ours]` `medium_long.log:355`, quoted in `2026-10-23-shared-memory-loops.md:157` |
| V6 | φ=0.46 [0.41, 0.53], looped +0.03…0.12 nats, 116 runs, 30–400M | `[T]` 2604.21106v3 Table 2/3, §4.2 |
| V7 | MoR 135M +0.0155, 360M −0.2759 | `[T]` 2507.10524v3 Table 9/15 |
| V8 | SMELT 200M −0.019 nats with 96–504 experts | `[T]` 2609.01343v3 Table 3.5 |
| V9 | LOOM 700M 18.36→16.54; ~100M 26.26→27.61 | `[T]` 2610.01153v1 Tables 2/3 |
| V10 | Sparse Layers base 38.7 / looped 37.4 / looped-MoE 39.6 | `[T]` 2605.09165v2 Table 4 |
| V11 | DeepLoop −0.016/−0.023/−0.019 nats at R=3/5/7, 124M | `[T]` 2607.13491v2 Table 5.1 |
| V12 | Closing the Loop: per-step LoRA null; one scalar −0.02 nats | `[T]` 2610.00673v1 §4.1–4.4 |
| V13 | Relaxed Recursive: 58.4 vs 58.6; SVD +6.5 points | `[T]` 2410.20672v3 §3.3/3.5 |
| V14 | Iteration gating at inference depth 56: 38.46 vs 36.59; ~0 at training depth | `[T]` 2609.36636v1 §5.2 |
| V15 | 30 MB noise 1σ 0.02–0.03; same-config RMS 0.0048 | `[ours]` `2026-11-08-lr-schedule.md` §4.1, §1.2 |
| V16 | The tying-fix bar 0.01–0.02 BPB, never flips the sign | `[ours]` `2026-11-06-depth-recurrence.md` TL;DR 4 |
| V17 | LatentLoop learned writeback 0.199/0.160 (damps the update ~5×) | `[T]` 2610.19975 §3.1/Fig. 6 |

### SPECULATION (mine, no direct measurement)

- **S1** — Expected ΔBPB per variant: (a) r=64 0…−0.005, (a/d) h=256 0…−0.010, (b) −0.005…−0.02, (c) ~0, (e) −0.006…−0.02. These are priors from the function-class argument and the full-block rows, not measurements of the cheap module.
- **S2** — The 95M dense 13th layer is worth <0.008 BPB. The closest measurement is the 1B/medium loop block; no 13L dense measurement exists. `D_crit ≈ 14` is an extrapolation of 2601.20994's constant to W=832.
- **S3** — h=256 per-iteration `W2_i` is the upper bound of the family tested here. A wider h, an activation after `W2`, or a full attention in the cheap module would be a different mechanism; the experiment does not bound those.
- **S4** — The 0.03 BPB pass threshold is derived from the repo's noise interpolation, not a measured power analysis at 95M/30 MB. A 0.03 effect at 3 seeds is ~2σ; a 0.02 effect is ~1.2–1.7σ.
- **S5** — The "few percent" wall-clock premise: at 95M the MFU is 43.8% (`docs/reviews/2026-11-08-mfu-first-look.md`), so 1.8% FLOPs should be ~1% step time; if the module is launch-bound, the wall-clock cost is larger than the FLOPs share. The experiment must log `gpu_step`. The MFU doc's rate is 56.1 kB/s; the cost below uses the ladder's slower 49.8 kB/s, a conservative margin.

---

## Sources

**Papers (arXiv; version and date in the table above):** 2604.21106v3/v4 (iso-depth) · 2507.10524v3 (MoR) · 2609.01343v3 (SMELT) · 2610.01153v1 (LOOM) · 2605.09165v2 (Sparse Layers) · 2607.13491v2 (DeepLoop) · 2610.00673v1 (Closing the Loop) · 2410.20672v3 (Relaxed Recursive) · 2609.36636v1 (What Makes Recurrence Effective) · 2606.01495v2 (CART) · 2609.19934v1 (DCP) · 2610.26030v2 (Truncate or Not?) · 2610.02383v1 (Shared Memory in Looped Transformers) · 1807.03819v3 (Universal Transformers) · 2405.16039v2 (MoEUT) · 2605.30202v2 (Dual-Path) · 2604.21254v3 (Hyperloop) · 2603.01914v2 (AdaPonderLM) · 2601.20994v1 (Depth Delusion, `D_crit` extrapolation) · 2301.13196 (Looped Transformers as Programmable Computers, theory only) · 1909.01377 (Deep Equilibrium Models, theory only) · 2610.19975 (LatentLoop, failure mode).

**Repo documents:** `docs/research/2026-11-06-depth-recurrence.md` (TL;DR 4, §1.5, §4) · `docs/research/2026-11-06-architecture-at-100m.md` · `docs/research/2026-11-07-depth-and-jepa.md` (§3, `D_crit`) · `docs/research/2026-11-08-lr-schedule.md` §1.2/§4.1 · `docs/reviews/2026-11-08-mfu-first-look.md` · `docs/research/2026-10-23-shared-memory-loops.md` · `docs/research/2026-10-23-latentloop.md` · `docs/research/2026-11-08-efficiency-frontier.md`.

**Configs / code:** `port/configs/flagship3_dense.toml:25-26` (FLOP ledger) · `port/configs/mix_refetch.toml` (data) · `port/src/dormouse/model/model.py:1405` (`_readout`, the insertion point).

**Hardware/recipe:** 95.0M dense, b8×s1024, bf16, `opt=mix`, `muon_lr_mult=12`, 49.8 kB/s (ladder) / 56.1 kB/s (MFU doc), eval 196 608 B at offset 0, 30 MB = 3 662 steps, ~10 min/run, per-seed 1σ 0.02–0.03 BPB.

**Manifest line owed:** `docs-site/tools/manifest.mjs` is not updated for this document (the task restricted this pass to one file); `python3 tools/check_doc_refs.py` will flag it until that line lands.
