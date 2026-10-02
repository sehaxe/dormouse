# Fidelity audit — what the papers promise vs what we run (moe · mhc · attnres · fb)

**Date** 2026-10-02. **Lane** the owner's hypothesis after wave 3: four arms tied
(moe 6.3377 / mhc 6.3653 / attnres 6.3300 / fb 6.4200 against the aux-ON control
6.3433, spread 0.0730 — `docs/reviews/night-verdicts-2026-10-02.md`, `benches/history.tsv:144-147`),
and the deletions were suspended pending an implementation-quality audit
(`.bulba/goal.md` "РЕШЕНИЕ ВЛАДЕЛЬЦА (2026-10-02)"). This file is that audit:
per technology, the paper's own operating point next to ours, every parameter
classified, and a verdict per class. **Read-only**: no code changed, one
document added. Owner's numeric decisions are recommendations, not edits.

**Method.** Originals: `docs/papers/2603.15031-attention-residuals.pdf` (extracted
`pdftotext -layout`, 1 439 lines), `2607.24653-kimi-k3.pdf` (SiTU + Stable
LatentMoE), mHC fetched live from arxiv 2512.24880v2 HTML (2026-10-02, sha =
full-text fetch), HC 2409.19606v3 fetched, MoE refs Switch JMLR 2022 /
Mixtral 2401.04088 / ST-MoE 2202.08906 / Loss-Free 2408.15664 (all cited inside
2605.09165v2, also fetched live), MTP 2404.19737v1 (HTML, full text). Our side:
`grep` + line reads of `crates/dormouse-core/src/*`, the four library crates, and
`configs/small.toml`. Nothing was run; the GPU is not in this lane.

**Context every verdict must carry** (§1.2, ADR-0002): the step budget itself.
The A/Bs ran 2 000 steps ≈ 8.2 MB of bytes on a 9.2M model. The three controls we
have that fair-compare arms are 2k; the one 100k run exists (held-out 5.545,
`bench` 2026-10-01) but no arm has a 100k equivalent. Two of the four papers'
evidence is at ≥300M params and ≥40B tokens: K3 is 2.8T total/104B active (§1
Intro), mHC is 3B/9B/27B with 39-262B tokens (Tab. 5), AttnRes's scaling sweep is
194M-528M × 39-119B tokens (Tab. 2). Our A/B runs 9.2M × 8.2MB. That is
**30× or more below every published operating point on three of four arms**, and
the honest way to say that is *capacity* / *budget* mismatch, not "the mechanism
is empty". The scope classes below are the audit's way to separate what was
*implemented wrongly* (fixable today) from what was *never given a fair
operating point* (park, re-judge at a bigger gate).

---

## 0. Provenance table (per paper: version read, where transcribed)

| technology | paper id (our copy) | version | read from | our-side transcription | caveat read out loud in-tree |
|---|---|---|---|---|---|
| **mhc** | 2409.19606 (HC base) + 2512.24880 (mHC) | v3 / v2 | arxiv HTML v3 / v2 full text, 2026-10-02 | `docs/reviews/mhc-2026-09-30.md` §1-§2 (delta table, Eq. 3-9 map) | H_pre not wired; n=2 not 4 — both named before the A/B |
| **moe** | 2605.09165 (Sparse-Layers) + Switch JMLR 23(1):120 (Fedus, cited §2.2) + ST-MoE + Loss-Free | v2 full text fetched 2026-10-02 | arxiv HTML v2 | `docs/reviews/moe-routing-2026-10-01.md` §5-§6 | "no established coefficient at ≤100M sources" is the file's own conclusion, quoted with the paper |
| **attnres** | 2603.15031 | v1 | PDF in `docs/papers/`, extracted 2026-10-02 | `docs/papers/attnres.md` (19 deltas, 9 BUG) + `docs/research/2026-09-30-attnres-integration.md` (fix record) | 22-line Fig. 2 is the only executable spec; the repo has no authors' code |
| **fb** | 2404.19737 | v1 | arxiv HTML v1 full text, 2026-10-02 | `crates/dormouse-core/src/future_byte.rs` module docs + `docs/reviews/future-byte-2026-09-30.md` | "our adaptation" — head shapes, vocab, and byte-level claims are ours, not Meta's |

Every row names the exact §, equation, or table the number came from. Under the
repo method (AGENTS §1.4) a citation without the § is storytelling, so they are in
every row of the tables below.

---

## 1. moe — top-k routing over the shared expert FFNs

### 1.1 What the papers promise

**The claim-shaping paper is 2605.09165 (Sparse Layers are Critical to Scaling
Looped LMs, USC + Netflix, May 2026).** It is *not* generic-MoE advocacy: it is
specifically about **looped** models, i.e. our shape. Two claims carry it:

1. **"Looped-MoE scales better than the standard baseline while dense looped
   models do not"** (abstract; §5.1: α = 0.077 Looped-MoE vs 0.076 Base, at the
   10¹⁸-FLOP isoFLOP band from 190M to ~305M active / 711M stored). The
   mechanisms: **routing divergence between loops** — the same token on pass 2
   picks a different expert set from pass 1 — recovers expressiveness the dense
   weight-tied loop loses. §6.1 quantifies it: **25-53 % of tokens get fully
   disjoint expert pairs between loops, only 4-14 % fully overlap** (k=2 of E=8)
   (Fig. 5). **Loop 7 is the single exception** (37 % identical, 3 % disjoint,
   "stabilizing embeddings before vocabulary projection at the loop boundary")
   — that is, **one loop SHOULD be allowed to be pass-consistent**; the rest
   must diverge.
2. **Scaling-law significance at the top end**: Looped-MoE's compute-optimal
   downstream average (39.6 OLMES Core-9) *beats* Base's (38.7) at 216M unique
   params vs Base's 246M (§5.2, Tab. 4). The gap it closes is the looped
   model's own deficit (Looped dense = 37.4), **not** a win over a dense
   standard baseline at equal *unique* params.

3. **Load balancing is part of the recipe in every cited config**: 2605.09165
   §2.2 lists TWO aux losses — `L_LB = E·Σ f_i·p̄_i` (Switch/GShard) AND the
   router z-loss `L_RZ = (1/B)·Σ (log Σ exp h_i)²` (ST-MoE, Zoph et al.), under
   **k=2 of E=8, Mixtral-style** (`citation: §2.2 "we use k=2 active experts
   out of E=8 total, following Mixtral"`). Kimi K3 §2.3 (a different branch)
   goes further: **auxiliary-loss-FREE routing** with a quantile-based expert
   bias update (§2.3.3, Eq. 13-14), **with the bias still required** — "Because
   b is omitted from p_i,j, it regulates dispatch without altering the mixture
   weights or the gradient-based optimization of the router" (K3 §2.3.3,
   verbatim). Smaller-scale support: Switch (Fedus, JMLR 2022 §3.3) mandates
   the same lb term at all scales it ran (64-4096 experts).

### 1.2 What we run

| | our value | paper value | class | where |
|---|---|---|---|---|
| experts | **3** (top-1 of 3) | **8, top-2** (Mixtral recipe; K3: 896 experts top-16 out of switched pool) | **scale-mismatch + hyperparam-wrong** (combined) | `configs/small.toml:16` (`n_experts = 3`); the wave ran `--set moe_topk=1` on it (`docs/reviews/ab-wave-2026-10-01.md` §2.2); paper: 2605.09165 §2.2 |
| k | **1** | **2** (both the Sparse-Layers and Mixtral operating point) | **hyperparam-wrong with a parse** — at E=3, k=1 IS "sparse"; at E=8 the Moore-near paper say k=2; our E Forbes k clamped by our bank size, so this is a JOINT choice not a free param | `crates/dormouse-core/src/config/schema.rs:602` `d_moe_topk() = 0` (off); on-position was 1 |
| **load-balance weight** | **0.0 — the balancer is OFF** | Switch §3.3, ST-MoE §(§2.2), 2605.09165 §2.2: **the term is not optional**; K3: aux-loss-FREE but requires an *equivalent* bias | **hyperparam-wrong — the single biggest one** | `configs/small.toml:35-37` (`moe_lb_coef = 0.0`), schema.rs:605. The sweep that argued for 0.0 is `crates/dormouse-core/src/moe.rs:628` "no value transfers" — reexamined in §1.3 |
| **router z-loss** | **absent** | ST-MoE §(Eq. 12): penalizes large logits, named in the 2605.09165 recipe | **missing entirely** (not even a config field) | no `z_loss`, `router_z`, `RZ` hits anywhere in `crates/` or the fused vendor tree (grep 2026-10-02) |
| **balancer form** | `E·(Σ f_e)·(Σ p_e)` with e as the index; **pre-top-k `p`** (`probs` returned by `topk_blend`, `moe.rs:79,145`) | Switch Eq. 6: `aux = E·Σ f_i·P_i`, P_i **pre-top-k** — our form is a **match, and the "which P" pitfall is gated** | **match (+)** | `moe.rs:79,86-146`, with the k=1 piecewise-constant-P trap documented at `moe.rs:96-106` and its gradient checked (`moe.rs:158` `g_lb_norm > 0`) |
| **routing granularity** | per (position, pass) — the controller reads `[h_ctx, h0]` where `h_ctx` carries `iter_embed[row]`, so the top-k can differ from pass to pass **in the logits**; **but only at mask time** see below | per (position, pass) — the paper's §6.1 finding is that pass-to-pass divergence IS the mechanism | **match** | `loop_block.rs:789-810`, the router-input gate `the_router_receives_the_pass_index` |
| **divergence actually observed** | **pass-INDEPENDENT: cross-pass top-1 agreement 0.9766 (dense) / 0.9590 (routed); disjoint % = 0.0410 routed** | Sparse-Layers §6.1: disjoint 25-53 %, identical 4-14 % | **mechanism-not-active** — see §1.3, this is the headline | `docs/reviews/moe-routing-2026-10-01.md` §6.1 (means over 8 model draws on 64-position fixtures, spread recorded there in §6.4) |
| **selection feedback** | **the mask zeroes losers AFTER they are computed.** All 3 experts run per position per pass; only their weights change. Trajectory-through-selection is not realized | Sparse-Layers §6.1: **"different experts are activated on each pass"** — the expert SET changes the computation, which changes the next state, which changes the next router top-k | **implementation-bug (structural), fixable** | `moe.rs:22-31` says it in the module docs ("every expert is still computed and then masked, so executed FLOPs are unchanged"); `loop_block.rs:807-810` is the current call site; the ponytail note at `moe.rs:34-43` names the gap |
| **expert FFN implementation** | each `expert_ffns[e]` is a TSCT (low-rank spectral) FFN, rank 64 | SwiGLU dense FFN per expert (2605.09165 §2.2; Mixtral) | **deliberately ours** (the loop's shared tech), NOT a bug — but it changes what "an expert" specializes; Sparse-Layers's own experts are SwiGLU (§2.2: "we use top-k token-choice routing … following Mixtral"), ours are TSCT | `loop_block.rs:236-241` field doc; `configs/small.toml:8` (`rank = 64`) |
| **dense blend (the k=0 control)** | controller softmax over 3 experts, all active | not the paper's arm — ours is a deliberate control | **deliberately-ours** | `loop_block.rs:789-793` + the zero-default gate `off_is_the_dense_blend_bit_for_bit` |

### 1.3 What is structurally wrong vs the papers (the top-3 moe findings)

1. **The two "papers promise divergence, we measure none" rows is the whole
   moe story.** One commitment the implementations did not meet: at E=3/k=1,
   the *selection* cannot change the computation because all experts run
   anyway and only their weights are renormalized top-1-softmax (always 1.0
   for the winner). Sparse-Layers (2605.09165 §6.1) sets its 25-53 % divergence
   on routing **as a compute assignment**, not as a mask: the paper skips the
   un-selected experts' GEMMs. Our reading hits it from the module's own docs
   (`moe.rs:22-35`): "Every expert is still computed and then masked, so
   executed FLOPs are unchanged" — the paper's *mechanism of action* (the
   selected expert changes the residual stream, so the next pass sees a
   different state, so the next router top-k differs) **is not reachable with
   a post-hoc mask**. The `ponytail:` at `moe.rs:36-43` names the missing
   piece: gather/scatter dispatch. **Class: implementation-bug (structural,
   the whole arm inactivates the paper's central finding).** Not a knob.

2. **The Step-1 measurement (§6.1 of `moe-routing-2026-10-01.md`) is an
   indictment, not a label.** With `moe_topk=1` of E=3, `disjoint% = 0.041` on
   64-position fixtures over 8 draws. Sparse-Layers' own Looped-MoE: **25-53 %**
   disjoint at k=2. Both our dense and our top-1 arm sit at pass-to-pass
   top-1 agreement of 0.96-0.98 — the exact regime the paper names as what
   kills a looped model ("almost every token, the same expert on every pass").
   Even at `E=8, k=2`, our mask cannot make pass 2 different from pass 1
   because the argmax (sub-mask) is unchanged. **Class: follows from #1.**

3. **`moe_lb_coef = 0.0` is the arm's own removal per ADR-0019's own standard,
   AND still the configuration the A/B ran.** The 2026-10-01 sweep
   (`moe.rs:628`) refused to ship a coefficient on the argument that Switch's
   0.01 and GShard's 0.1 "do nothing at all here" — the measured ratio is
   ~220× off at 64 rows and would need ~1.4e4 at the trainer's 4 096-token
   batch (moe.rs's `coef_crit`). Legal per `docs/reviews/moe-routing-2026-10-01.md`
   §6.2, but **every published MoE recipe has a balancer**: Switch §3.3,
   ST-MoE (z-loss), 2605.09165 §2.2 (`L_LB` AND `L_RZ`), K3 (aux-loss-free
   with bias Eq. 14). Running top-1 with no balancer is not a paper
   configuration; it is our arm-minus-balancer control, and both were tied.
   **Class: hyperparam-wrong at A/B time, but NOT because the code is
   wrong — because 0.01/0.1 fail at our batch, and we did not re-derive the
   form (token-count-invariant normalization) before the A/B ran.** §1.4 has
   the fix-list.

### 1.3b One root the sweep named in the code, re-priced here

Switch's `P_i` is a **mean over tokens**, so the balancer's gradient per token
scales as `E/tokens` while the CE gradient per token does not. At 64 rows the
gap is 220×; at 4 096 tokens it is ~1.4e4×. Every published coefficient
assumes a token population we do not have — **this is a scale-mismatch inside
the hyperparam-wrong class**. The form is fixable at our scale by the
**normalization** (e.g. drop the outer `E·` sum, mean over tokens per expert,
normalize the resulting aux by `(1 - 1/E)`, or make `f` a *router-weighted*
token-count), but the number has to be re-derived from the ratio measurement,
not copied. `moe.rs:628` and `moe.rs:222-234` both say this today: the trained
object is "what fraction of task gradient the balancer applies"; a
token-count-invariant form is the owner's call before any re-run.

### 1.4 Verdict and fix-list

**Class: hyperparam-wrong (k, lb) × implementation-bug (no divergence axis)**
on the routing; **deliberately-ours** on TSCT experts, the dense control, and
param-neutrality at n_experts=3. The A/B verdict of 2026-10-02 ("tie") was
honest to what ran — but what ran was **not Sparse-Layers' arm**:
E=3 top-1 unbalanced, masked-not-dispatched.

**Fix-list (what to change before any re-A/B):**
1. **Top-2 with E≥4** (`moe_topk=2`, raise `n_experts` to 4-6 in the arm's
   preset, not in the global recipe) — this is the Sparse-Layers operating
   point at our scale (k=2 of E=8 is theirs; 8 experts at 9.2M params per
   §1.1's own review is the outer bound). Cost: +1 expert's TSCT weights ≈
   +2.1 % params over `small` (owner's gate, not a silent change).
2. **A token-count-invariant LB form** — re-derive the coefficient from the
   measured `‖grad L_LB‖ / ‖grad task‖` ratio (the quantity `moe.rs` already
   prints) for the 4 096-token batch, then re-run the collapse sweep. The
   **1/r** form (currently `lb_aux(probs, mask, n_experts)`) is NOT the only
   one in the papers; ST-MoE's z-loss (`L_RZ` above) is also missing
   wholesale and is a one-line `logsumexp` away from `lb_aux`'s shape.
3. **Real dispatch** (`moe.rs`'s own ponytail note): compute only the k
   selected experts per (token, pass). This makes pass-to-pass divergence
   *possible*; the measurement above (§1.2 "divergence actually observed")
   then measures a real thing instead of an underflow. Without this, no
   other fix in the list can produce a Sparse-Layers-like result.
4. **Router z-loss** as a separate additive term (or confirmed-absent with a
   reason in the sweep file): it exists in both Switch Lineage (ST-MoE) and
   2605.09165 and it costs nothing to add to `lb_aux`'s interface.

**Park-list:** the full Sparse-Layers recipe (μP-tuned LR, WSD schedule, 216M
unique params, 10B-token FineWeb, AI2 OLMES) is a ×22-35 budget mismatch; the
mechanism's value is *capacity on the same FLOPs*, which matches the owner's
scale-gate. **Re-visit at ×10 params** per `.bulba/goal.md`'s ×10 note, or
when the loop's FFN branch is the profiled hot spot (not today: it is ~5 % of
a warm step per `loop_block.rs:583-592`).

### 1.5 What the code gets right that the AIP should keep

The **pre-top-k `probs`** pitfall is caught by a gate (`moe.rs:100-106`,
"the sweep's first assertion is now `‖grad L_LB‖ > 0`") — a bug the switch
paper's own text forces and that killed the first version of `lb_aux`
(measured, `‖grad‖ = 0`). `topk_indices` reuse from `burn_mor` is the
documented cross-credit; the "renormalized gate is exactly 1.0 at k=1"
trap is named in the module docs before the sweep could mis-price it. This
is a good implementation of a bad (for our scale) design the owner picked.
The k=1 renormalization claim in `moe.rs:31-42` ("a top-1 token's gate is
exactly 1.0 — the same scale as the single shared FFN the control uses")
makes the A/B **a comparison of network shapes**, not a paper-fidelity test;
§1.1 says which test the paper actually runs.

---

## 2. mhc — Manifold-Constrained Hyper-Connections

### 2.1 What the papers promise

Two sources, both tier-fetched (2026-10-02):

1. **Hyper-Connections (2409.19606v3, ByteDance; the base mechanism)**: the
   residual stream is `n` parallel C-dim copies (`n=4` in all tables), Tb at
   1.5B and 7B; **HC beats the residual baseline consistently** (§4, Tables
   [the paper's Tab. 1-3]). The mechanism HC's own paper claims: a
   **learnable, input-dependent mixing matrix** `A_r` over the `n` stream
   copies at every layer, plus a wide read-in `A_m` and a per-layer write.
   Its own most important ablation (**Tab. 3, DHC×4**): freezing `WC` costs
   +0.021 V2 loss; freezing `B` costs +0.002 — the **width-connections are
   the load-bearing half**.
2. **mHC (2512.24880v2, DeepSeek)**: **expansion rate n = 4** on every model
   in Tab. 5 (3B/9B/27B/3B-1T), t_max = 20 Sinkhorn iterations, ε = 1e-20
   RMSNorm eps, gating factor init α = 0.01, layers = 12/18/30, and **4.14B
   active / 27B total** parameters. Its system claim (§4.3, opening): **"
   implementing mHC (with n=4) in large-scale models with a marginal training
   overhead of only 6.7%"** — with three fused TileLang kernels
   (§4.3.1 Eq. 14-19), a recompute schedule (§4.3.2 Eq. 20), and DualPipe
   overlap (§4.3.3). mHC's own ablation (§3 Tab. 1) is the same seed story:
   **H_res alone = -0.022 of the -0.027 total; H_pre adds only +0.003** (from
   the full-3-mapping arm). Its stability story (§3.1, Fig. 2/3): HC's
   unconstrained composite reaches Amax Gain ~3000 and mHC cuts it to ~1.6
   (the doubly-stochastic closure). Its quality claim vs baseline at 27B:
   **-0.021 final loss** (§5.2 Fig. 5(a)); BBH +2.1 %, DROP +2.3 % downstream.

### 2.2 What we run

| | our value | paper value | class | where |
|---|---|---|---|---|
| **streams n** | **2** | HC/mHC both run **4** (HC §4 Tables; mHC Tab. 5) | **hyperparam-wrong** (deliberate, named; re-priced below) | `configs/small.toml:57` absent → schema default `d_mhc_streams() = 2` (schema.rs:599-601); the A/B ran `mhc_streams = 2` (`docs/reviews/ab-wave-2026-10-01.md:432-434`) |
| **Placement: deep vs layer** | **one shared block at the loop boundary** (per-iteration write, not per-layer) | HC/mHC put it **at every layer** (the mechanism IS "input-dependent mixing per sublayer", Fig. 1 of 2512.24880; Tab. 3 ablation freezes per-layer `A_r`); the φ-claim paper's own setup is "across loops" but its model is a looped stack, not a single loop | **placement-transposed** — see §2.3 | `loop_block.rs:979-998` — one `MhcBlock` applied at every iteration's residual write (loop block = one iteration = one whole KDA+Engram+FFN body) |
| **H_res** | Sinkhorn-Knopp, 20 iters, log-domain | mHC Eq. 9: same t_max=20, same pipeline | **match** | `burn-mhc/src/sinkhorn.rs:6,47-70` |
| **H_post** | 2σ(·), per iteration | mHC Eq. 8: 2σ | **match** | `burn-mhc/src/block.rs:113-116` |
| **H_pre** | **NOT wired** (allocated, never read) | HC Eq. 3 / mHC Eq. 7-8: `H_pre x` feeds the block's input, `H_pre = σ(...)` | **deliberately-ours** (named as ~11 % of the paper's effect in our own docs) | `burn-mhc/src/block.rs:93-92` per map; `loop_block.rs` calls `MhcBlock::forward` which reads only `mappings_post_res` (block.rs:93-147); `mhc-2026-09-30.md` §3.3 names the 11 % |
| **streams construction** | **disjoint slices of D** — `h.reshape([b,t,n,C])` | **n replicated C-dim copies** — `x_{l,0}ᵀ … x_{l,n-1}ᵀ` (HC §3: "n·C dim residual stream"; mHC §3) | **deliberately-ours + meaning-changing**: theirs is a re-weighting of one vector; ours is a cross-stream exchange of n real sub-states. **NOT a bug** — but it means our `H_res` operator is a DIFFERENT function from theirs | `burn-mhc/src/block.rs:180-199` (slice + concat readout); `mhc-2026-09-30.md` §2.2 documents it as forced by our `d_model = D` body |
| **readout** | **concatenate** the n slices back to D | HC §2: **sum** the n streams; DeepSeek's 2409.19606 §2.3 scales output stds by `√n` to compensate | **deliberately-ours** (avoid `√n` retune) | same lines as above |
| **`α` init** | 0.01 (the paper's Tab. 5 value) | mHC α = 0.01 | **match** | `burn-mhc/src/block.rs:6` `ALPHA_INIT` |
| **bias init** | static `b_pre = b_res = 10`, `b_post = 0` | **mHC's paper states NO bias init** (the constraint's own docs say this); HC's Eq. 14 uses `B=1, A_m = e_{k mod n}, A_r = I` — a **rotation**, not uniform | **hyperparam-wrong (ours named)**: our `10` puts the off-diagonal at `e^{-10}`, so `H_res ≈ I` and gradient `α_res ≈ 7.3e-9` at init; training has to fight an exponential stiffness ~`e^{-10}` off-diagonal | `burn-mhc/src/block.rs:7-8,51-61`; measured in `mhc-2026-09-30.md` §5.3b — "the arm's whole parameter cost is 0.067 % of small"; `alpha_res` gradient 7.288e-9 |
| **dynamic half** | `Normal(0, 0.01)` init on `phi_*` | HC §2.3: `W_β, W_m, W_r` init **zero** ("the dynamic half is switched off at step 0") | **hyperparam-wrong (ours named)** | `burn-mhc/src/block.rs:51-61`; the delta is documented in `mhc-2026-09-30.md` §2.1 |
| **tanh** | absent (mHC Eq. 7's linear `α(x̃'φ) + b`) | HC Eq. 5 has tanh; **mHC's printed equations do not** (crate follows the printed Eq. 7, and the base paper's own Tab. 1 finds "w/o tanh" neutral-to-better) | **match (mHC branches)** — but this is the branch the φ-claim does NOT transfer from | `burn-mhc/src/lib.rs:17-26` header; 2409.19606 Tab. 1 |
| **stream width on our model** | n=2 → `D = 768` slices of 384 | mHC: n=4 at `d_model = 2560` (27B), `d_ffn = 1536`; the whole mechanism is designed to give a 30-layer deep 4.14B-active network a **non-trivial residual mixing topology** | **scale-mismatch on the placement axis** — see §2.3 | `configs/small.toml:4` `d_model = 768`; `loop_block.rs:509-511` builds `MhcBlock::new(2, 768, …)` |
| **depth the mechanism spans** | **T = 2-4 iterations**, all orthogonal to each other | **30 layers** with per-layer `H_res`, i.e. the composite `∏ H_res^i` spans 30 different matrices and the paper's stability claim is about exactly that composite | **scale-mismatch** | `configs/small.toml:5` `max_iter = 4`; `mhc-2026-09-30.md` §5.1b measures the composite at T=2-4 only |
| **step-time claim** | unmeasured (the A/B was wall-clocked on a loaded card; `573 ms/step` is recorded but marked "ran beside other lanes' CPU work") | K3/mHC: **+6.7 % step time at n=4** (`2512.24880v2 §4.3 Intro`) with 3 fused kernels + recompute | **unmeasured-vs-published, a fairness obligation not a bug** | `benches/history.tsv:144` carries the honesty note verbatim; mHC's own number has a kernel-fusion precondition (§4.3.1) our crate deliberately does not run (`mhc-2026-09-30.md` §6) |

### 2.3 What is wrong here (the honest verdict tree)

- **placement is NOT simply "transposed"** and calling it so would be a lie.
  In HC/mHC the mechanism sits **between every adjacent pair of
  sublayers** of a 12-30-layer stack. In our loop block it sits **once per
  loop iteration**, i.e. between 2-4 whole-body passes of a single weight
  set. Our own doc `mhc-2026-09-30.md` §3.2 solved this deliberately:
  "weight sharing is the loop's defining property; per-slot residual matrices
  would make the residual operator the only un-shared thing in the block" and
  the composite `H_res^T` (T = the depth) preserves the Birkhoff structure
  exactly. So our arm measures **mHC-at-the-loop-boundary**, not mHC.
  **Class: placement-transposed.**

- **n = 2 was justified by the φ evidence from a DIFFERENT paper** (2604.21106,
  "φ = 0.65 hyperconnections") and by the base paper's Tab. 1 trajectory
  (`n=1: worse than baseline`, 2.819 vs 2.811; `n=4` best 2.781, `n=8`
  2.778). Our own `mhc-2026-09-30.md` §1.3 documents it, so this is a
  **deliberate deviation** that the tie result does NOT falsify: **n=2 is the
  rung the base paper says is where most of the gain has NOT yet arrived**
  (2.802 of the 2.778-2.778 best = 0.030 total; n=4 buys 0.021 of the
  family's 0.033 over baseline). At n=2 the residual mixing eigenvalue is
  `-0.76` and the off-diagonal mass **oscillates** around the uniform
  fixed point across `T=2-4`, measured and pinned as the composite's own
  property (`mhc-2026-09-30.md` §5.1b). **The strongest honest framing: at
  n=2 with the EXACT-identity init `b_res = 10I`, the operator is ReZero-at-
  scale-1 with 280 tiny Sinkhorn launches per iteration and an `α_res`
  frozen to 7e-9.** That is a paper-configuration the papers never ran.
  **Class: hyperparam-wrong (n=2 AND the 10I init together, jointly).**

- **The composite ∏ H_res^i argument the mechanism uses across 30 layers
  has no story at T=2,4.** The paper's selling point (closure + bounded
  gain) is a property of propagation over MANY layers; with T=2 the
  composite is one matrix's square and the whole "this is why the manifold
  matters" argument has nowhere to bite. `mhc-2026-09-30.md` §3.2 point 1
  says the constraint claim survives **structurally** — correct — but the
  quality claim needs the depth regime where instability would otherwise
  appear. Our control never shows the instability mHC exists to cure (small
  d_model 768, depth 2-4, 2m tokens) so the constraint is solving a problem
  absent here. **Class: scale-mismatch (the depth half).**

- **There is NO fused-kernel path actually enabled** — our `burn-mhc/cuda`
  is compiled off (`mhc-2026-09-30.md` §6), the tensor path issues **280
  forward + 40-something backward tiny launches per call** (own doc §6),
  and on a 13 %-utilised card this arm pays the launch-bound tax, not the
  arithmetic one. The papers' 6.7 % number carries **3 fused kernels + a
  recompute schedule**; our arm carries 320 un-fused launches/time
  per step. The cost *cannot* be recovered by turning n up (launch count is
  n-invariant, §6.1) — only fused kernels collapse them. **Class:
  implementation-bug-adjacent (a paper with fused kernels is not the thing
  the A/B measured); the fix is written down in the report and gated on
  §4.1's 1e-7 floor defect, which is still open.**

- **The static-identity init plus `b_res = 10I` is more quietly bad than it
  looks.** Return to `mhc-2026-09-30.md` §5.3b's own number: "One unit on a
  single off-diagonal of `b_res` moves `H_res` by **2.1e-4**, because the
  diagonal sits at `e^10` … So an `H_res` that training has not yet moved is
  *exactly* the identity, and the arm's whole contribution in its first
  steps is `H_post ≈ 1` — i.e. a ReZero." That is a **side-effect of our
  own init choice**, not a property of mHC. DeepSeek's paper puts α = 0.01
  on a Sinkhorn that has to learn to move; we picked a bookkeeping-safe
  10I. **Class: hyperparam-wrong (ours), fix trivial and named — the init
  has to be moved off the exponential's flat top, e.g. `σ⁻¹(0.5) = 0`**

### 2.3 Verdict and fix-list

**Overall class: placement-transposed (a deliberate, documented restructuring
of the operator's host loop) × hyperparam-wrong (n=2, identity-init, zero-init
dynamic half) × scale-mismatch (T=2-4 vs 12-30 layers).** We measured the
tie against **an mHC-shaped function at n=2 with a 10I identity init and no
fused kernels**, inside a 9.2M-parameter 2-iteration weight-shared loop.
Nothing in the tie says "mHC does not help at its paper's shape".

**Fix-list (re-A/B blockers, in order of cost):**
1. **n = 4** (`mhc_streams=4`, +18 459 params ≈ +0.2 % of `small`). This is
   mHC's and HC's shipped operating point in Tab. 5, and our own doc
   already names it as the follow-up (`mhc-2026-09-30.md` §7.7).
2. **The identity-init out.** Drop the static `b_res` from 10 to something
   the Sinkhorn can actually leave within ≤4 steps, e.g. `b_res = 0` +
   `α_res` learning the diagonal — or directly the Dynamic Zero init (HC
   §2.2: zero-init `W_r`), so the dynamic half starts in the paper's
   configuration. Requires a **gate that watches `min(b_res)^i` leave its
   init** on the eval line (our own doc §7.9 already asks for this — "the
   arm's first N steps are a ReZero with a Sinkhorn in front of it").
3. **Fused Sinkhorn on (§4.1's 1e-7 floor + §4.2/§4.4's gate fixes)** so the
   arm's cost at n=4 is launch-bound-safe, matching the paper's own
   cost story, and step-time reads at idx ≥ 50 on a quiet card
   (`docs/protocols/AB-PROTOCOL.md` "What is NOT an A/B") — the 573 ms
   number in `history.tsv:144` is honest but it measured a DEGRADED launch,
   not the arm's price.
4. Placement: a **per-layer variant inside the loop** is a different
   experiment (the loop's readout `out_proj` per iteration makes it a real
   option on our architecture: one `MhcBlock` per `iter` slot, like AttnRes's
   `Vec<AttnRes>`, is +0.4-1.5k params for a single second `H_post`-owning
   block). Not required by this audit; **explicitly NOT recommended as the
   first fix.**

**Park-list:** the 2409.19606 DHC×8 / Amax-Gain-3000-regime claims do not
transfer to T=2-4 at 9.2M params. The honest entry is "**retire until the
loop's depth is ×10 and/or the model is ×10**" — which means AttnRes and mHC
end up on the SAME wait-list (both are depth mechanisms), not on a deletion
queue.

---

## 3. AttnRes — Attention Residuals (Full mode wired; Block not)

### 3.1 What the paper promises (the PDF, page-cited)

`2603.15031v1`, read via `pdftotext -layout` into 1 439 lines (§1 of
`docs/papers/attnres.md` has the byte count and md5).

**The promises, in the paper's own tables** (aliased page numbers from the
extracted text):
- **Tab. 2, `N=8` block variant**: at 436M active params / 87.9B tokens:
  baseline 1.766, **Block AttnRes 1.746, Full 1.737, mHC(-lite) 1.747**.
- **§5.2 scaling curves across 194M-528M / 39-119B tokens**: AttnRes is
  lower across the whole range (Fig. 4).
- **§5.3, Tab. 4 ablations** (16-layer model): Full 1.737; **input-
  dependent query 1.731 (their best variant, declined on cost)**; **w/
  sigmoid 1.741**; **w/o RMSNorm 1.743** (the norm is doing work);
  SWA W=1+8 1.764; Block S=4 1.746; multihead H=16 1.752 (their negative);
  DenseFormer 1.767 ≈ baseline.
- **The mechanism is softmax attention over depth**
  (Eq. 1-4, Fig. 2 pseudocode), with `ϕ(q,k) = exp(qᵀ RMSNorm(k))` — **no
  temperature, and RMSNorm with the mean form** (`kᵢ = vᵢ = h₁ if i=0` — the
  token embedding is永久 a source) (Table 5 fn 2).
- **"All pseudo-query vectors must be initialized to zero"** (§5, the only
  hyperparameter statement in §5.3), so init is a uniform average.
- **Block AttnRes** partitions the L layers into `N ≈ 8 blocks` of `S = L/N`
  sublayers (§3.2, Fig. 6 sweeps S; S=1 recovers Full at 1.737).

### 3.2 What we run

| | our value | paper value | class | where |
|---|---|---|---|---|
| mode | **Full** (`N = L`, every iteration is its own block) | paper's Tab. 2 headline is **Block N=8**; Full is the 1.737 ablation row | **deliberately-ours**, documented at `docs/research/2026-09-30-attnres-integration.md` §5 — "at our depth (2-4) Full *is* Block: the paper's own §3.2 says N=L recovers Full and Fig. 6 reports S=1 matching Full at 1.737" | `loop_block.rs:951-964` (`res.push` + `depth_attend(&res, slot_query)`) |
| score form | `ScoreForm::Paper`, **no temperature, mean-RMSNorm** | `exp qᵀ RMSNorm(k)`, joint softmax | **match** (formerly sqrt-d — **FIXED and gated**, `lib.rs::tests::paper_form_has_no_temperature_and_this_is_pinned` red on either convention) | `burn-attnres/src/lib.rs:84-115,194-206`; the integration doc §3 has the 3-convention table |
| sources | token embedding `b₀ = h₁` (permanently) + one entry per iteration `y_i` | Eq. 6/Table 5 fn 2: `b₀` separate and permanent; `v_i = f_i(h_i)` else | **match** (the D5 defect in the BLOCK path is not reachable from the model — the model took the Full path) | `loop_block.rs:659-664,963` |
| query | one zero-init learned `w_l` per **iteration slot** | one per **layer** (attn_res_proj before attention, mlp_res_proj before MLP) | **placement-transposed (accepted)**: our loop has ONE body per iteration, not two sublayers. The param count is `[max_iter, d]` = 4×768. The alternative (a shared single query) is NOT the paper's either | `loop_block.rs:503-505,707-718` |
| query init | `Zeros` | §5: "all pseudo-query vectors must be initialized to zero" | **match** | `AttnRes::new` at `burn-attnres/src/lib.rs:103-110` (`Initializer::Zeros`, with the paper's §5 quoted over the line); gated at `attnres_at_init_is_a_uniform_average_of_its_sources` |
| RMSNorm | mean-RMSNorm, **no learned gain** | Zhang & Sennrich ref [66], a learned-gain RMSNorm — the paper's Tab. 4 `w/o RMSNorm` row (1.743 vs 1.737 Full) says the norm is doing real work; ours has no gain knob at all | **hyperparam-wrong** (minor): the paper's `RMSNorm` includes a learnable affine, ours is a bare mean-norm | the norm is computed inline in `depth_attend_form` (`burn-attnres/src/lib.rs:194-230+`, `let m = form.norm_m(d)` at :206); `docs/research/2026-09-30-attnres-integration.md` §2 row "§5" names the missing gain |
| **query is FREE, not input-dependent** | free `w_l` | paper's **best variant** is the input-dependent one (1.731 vs 1.737); the free-query form is their default, and ours matches it | **deliberately-ours** — matches the paper's main arm, not the paper's best ablation | `burn-attnres/src/lib.rs:90,107`; `docs/papers/attnres.md` Table 4 row "w/ input-dependent query 1.731" |
| **d^{-1/2}** | NOT present (Score::Paper) | absent in the paper | **match** | `burn-attnres/src/lib.rs:184-230` |
| **the derivative was wrong** | `∂s/∂h = ...` **was missing its `m`** (fixed 2026-09-30, gated red at rel 1.47 on pre-fix) | the paper's derivative (math) finds it | **implementation-bug — FIXED in-tree, gated** | `docs/papers/attnres.md:251` (`D1`), fixed in the same lane (`docs/research/2026-09-30-attnres-integration.md` §3 "What making the form paper-faithful FOUND") |
| **sinks, Block variant** | not wired | S ≈ 8 blocks | **deliberately-ours** (Block is still wrong per the 2026-09-29 audit D5-D9 and is NOT wired; at T=2-4 the two are the same operator) | `docs/research/2026-09-30-attnres-integration.md` §5, §2.1 |
| **the `α` normalization is SOFTMAX over sources, includes `b̄ₙ^{i-1}` partial** | matches Fig. 2 | Fig. 2 lines 10-12 | **match** | `burn-attnres/src/lib.rs:194-230` |

### 3.3 What the A/B actually compared (and what needs to be true for a second one)

Wave-3 attnres 3-seed: **6.191 / 6.346 / 6.453, mean 6.3300, range 0.262**
(`docs/reviews/night-verdicts-2026-10-02.md`, `history.tsv:146`); the control's
own range is 0.0730. The paired same-seed deltas are **-0.196 / +0.032 /
+0.124** (mean -0.013). Seed s1's 6.191 is a **-0.196** against the control,
which is well outside the control's ±0.0730 band, and the night doc already
names it honestly: "s1's 6.191 was seed spread, not signal" — but **0.262 of
arbitrary seed spread in a 3-seed sample is itself 3.6× the control variance
that the bar was read from.** The arm's mechanism is not variance-reduced and
its largest same-seed reading goes BEYOND the bar. Either:
1. the arm is noisy in a way the control is not, or
2. the arm does something real at some seeds and nothing at others.

The wave's honest verdict was right per the §1.2 bar. **But the bar itself was
computed from three control seeds, and a mechanism whose seed spread is
3.6× the yardstick's is not measured by that yardstick.** The 0.0730 bar
needs a波-3 rung of its own before it can judge an arm (this is a
methodological finding, fix is in §5 below).

### 3.4 The paper-fidelity verdict on AttnRes

**Our implementation is the CLOSEST of the four to its source, by a wide
margin** — the score form, zero-init, source set, Free-vs-input-dependent
query, softmax schedule and unnormalized values are all a match, each with
its gate and (since 2026-09-30) with the formula's own derivative fixed. Two
things remain that the paper's own numbers say are material and that our
arm does not have:
- **no learned RMSNorm gain in the scorer's norm** (the "w/o RMSNorm" run
  1.743 stays 0.006 above Full; ours has no gain knob at all — a real
  ablation arm we do not offer).
- **input-dependent query (1.731, their best)** is a cost no at 9.2M params —
  but it is the paper's own upper bound on what the mechanism can buy, and it
  is exactly what our A/B is measuring against when we read 6.330. Their 1.737
  vs base 1.766 delta at 436M×88B is the mechanism's whole effect; **at
  9.2M × 8.2MB any effect that is real but small cannot clear a 0.0730 bar.**

**Class: mostly clean transcription + scale-mismatch on the quality budget.**
The model mechanism is; the scale attribution is not.

### 3.5 Fix-list
1. **Read s1's 6.191 at a bigger step budget** before treating the mean as a
   null. The verdict already says "the signal for re-run with a bigger step is
   queued" (`.bulba/goal.md` — "s1=6.191 is the number for a re-run with a
   bigger step (in the queue)"). This is a **time-money question, not an
   implementation one** and it is the single most likely place the owner's
   "maybe our implementation is wrong" hypothesis gets tested for attnres
   honestly.
2. **the 0.0730 yardstick was measured on a control family with `retract_every
   = 1` while the arms all ran 4** (one paired seed measured -0.058 for the
   wave's cadence). Run the two missing `retract_every=4` control seeds
   (§7.2 of the wave doc says this already and it is still open). This is
   **|^0.05-of-systematic on every wave-3 verdict simultaneously** —
   the single cheapest improvement to the wave's evidence.
3. **Learned RMSNorm gain** (`γ` on the normed K project), one scalar per
   slot — 4×768 params on `small`, one `mul` in the kernel. Closes the only
   real hyperparam-wrong row the arm has, and it is the paper's own ablation
   (`w/o RMSNorm` costs them 0.006). Cheap, cannot be silently wrong given
   the existing paper-form gate.
4. **Input-dependent query** — the paper's only reported better variant. At
   cost `+d_model per slot` (matmul `h → w_l` instead of a free param). The
   paper declines it on cost at 436M; at 9.2M the cost line is the one it
   lives on. **Register as an arm, not a fix** — it changes the mechanism
   class (from "one learned query per layer" to "depth attention with queries
   projected from the hidden state"), and §3.3's evidence gap should be
   closed first.

### 3.6 Park-list
- **Block AttnRes** (S ≈ 8): a memory mechanism; our depth makes it identical
  to Full. Blocks become materially different at S ≥ 4, which needs L ≥ 16 or
  a 4× depth growth — the ×10 gate.
- **Multihead depth aggregation (H=16)** — the paper's own negative result;
  we correctly ship it absent.
- **SWA window** — the paper's `W = 1 + 8` rung; a deployable memory story,
  gated on our depth.

---

## 4. fb — future-byte auxiliary head (MTP, arXiv 2404.19737)

### 4.1 What the paper promises

The 2404.19737v1 full text (fetched 2026-10-02). Two result classes matter:

1. **The effect is real at ≥1B-token scale and grows with model size**
   (§3.1 Fig. 3: 300M → 13B on code with 91B+ tokens; **the 7B models solve
   +12 % HumanEval / +17 % MBPP**; n=4 at 7B ≥ 200B tokens is best on
   HumanEval/MBPP). §3.3 byte-level 7B: **8-byte prediction solves 67 % more
   MBPP pass@1 than next-byte** on 313B bytes — byte-level lookahead
   specifically is one of the paper's STRONGEST results.
2. **BUT §4.1's induction result has a sharp scale line**: "2-token
   prediction loss leads to a vastly improved formation of induction
   capability for models of size 30M nonembedding parameters and below, with
   their advantage **disappearing for sizes of 100M nonembedding parameters
   and above**" (§4.1, and its Fig. 7 caption). At 30M-and-below MTP-BUILDS
   induction; at ≥100M the model learns induction regardless and extra n
   becomes a wash. Ours is 7.4M non-embedding — **inside the range where the
   paper says the effect is REAL**, which is worth saying out loud: the
   scale-mismatch argument that kills moe/mhc/attnres has a paper-named
   *counter-evidence* in fb's favour at our scale. And §3.7: "Multi-token
   training with 7B models doesn't improve performance on choice tasks …
   the 2-future-token model has the same performance as the baseline and
   the 4-future-token model regresses a bit" — i.e. **n=4 on natural
   language MULTI-CHOICE benchmarks at 7B is slightly NEGATIVE** (their own
   Fig. 5), while n=8 on BYTES is +67 %/+20 % at 7B (§3.3). The paper's own
   optimum for byte-level is **8 bytes**, not 4, and it is measured on a
   7B byte-level model TRAINED at code.
3. **The architecture is `n` independent heads at ONE shared trunk**:
   each head is a **transformer layer `f_{h_i}` whose output feeds the
   shared unembedding `f_u`** (§2: "`P_θ(x_{t+i}|x_{t:1}) =
   softmax(f_u(f_{h_i}(f_s(x_{t:1}))))`"). **Only the per-horizon layers
   `f_{h_i}` are separate; the unembedding is SHARED** — the paper's main
   architecture is NOT "n untied vocab heads". Our head is a single linear
   `d_model → vocab`, fully untied, no transformer layer in front. (The
   paper's Appendix B does list bare linear heads and replicated
   unembeddings as ALTERNATIVE architectures — "Linear heads", App. B —
   i.e. our shape is the paper's appendix arm, not its main one.)

### 4.2 What we run

| | our value | paper value | class | where |
|---|---|---|---|---|
| **head shape** | **ONE `LinearLike(768 → 256)` untied, no trunk layer between it and the readout** — the T-averaged loop readout feeds the head directly | **n independent transformer layers `f_{h_i}`, one per horizon, each feeding the SHARED unembedding `f_u`** (§2; Fig. 1) — concretely: the aux head has its OWN depth, it is not a bare linear | **implementation-shape, arguably the biggest delta** — the extra capacity + `f_u` sharing is where the paper gets its "no-train-time-overhead" free expression gain; our head trains `d_model→vocab` in ONE dense Linear | `future_byte.rs:78-79` ("One `LinearLike`, `d_model -> vocab`"); `model.rs:177` `aux.fb = (cfg.aux_fb_weight > 0.0).then(\|\| LinearLike::dense(d, v, device))` |
| **untied vs shared unembedding** | **fully untied** (the gradient never touches `lm_head`, gated) | **shared `f_u`, separate `f_{h_i}`** — the paper's main architecture shares the unembedding across all heads AND the main head; the untied-vocab-head shape is its App. B alternative | **hyperparam-wrong (a deliberate reading, but the module docs' "untied for the paper's reason" is not what the paper's §2 says)** — the real delta is the missing `f_{h_i}` layers | paper §2 + App. B ("Linear heads"); our `future_byte.rs:28-40` "untied for the paper's reason (separate heads per horizon)" |
| **n / horizon** | **k ∈ {2}** (one head, one horizon) | **n ∈ {4} best for tokens, n ∈ {8} at byte-level**; each head is per-horizon, ALL n heads train at once | **hyperparam-wrong** at our scale. The paper's byte-level shoulder is **8** (§3.3), and n=2 is the least-trained row of the whole table (the weakest `n` at every §3.1 scale except APPS) | `configs/small.toml:30` `aux_fb_weight = 0.0, aux_fb_horizon = 2`; the wave ran `--set aux_fb_weight=0.1` at horizon 2 |
| **number of heads A/B'd** | **one horizon head at a time** (k=2) | **all `n` at once** per position | **placement-transposed** (mixed with hyperparam) | `future_byte.rs` (single head, single horizon) |
| **weight** | **0.1** (in the arm) — the term is `0.1 · CE_future`, added to `jepa_weight 0.05 + koleo` | no aux weight — the aux is a **full second loss at `n` heads**, alongside the main CE; the trainer's balance is set BY the head count, and the loss is plain CE, not weighted | **hyperparam-wrong** — 0.1 at n=1 has no paper analogue | `model.rs:526` `f.mul_scalar(self.aux_fb_weight)`; K3's fresh MTP at scale uses a full-blown head-stack (K3 §2.5 tab, "Number of MTP Layers 1") |
| **vocab** | 256 (byte) | 1 of 129 280 (token); the byte-level ablation is separate | **scale-mismatch, NOT wrong** — 2404.19737 §3.3 is itself a byte-level test, ours is a byte-level test; the vocab/bin ratios differ by ~505× | `configs/small.toml:3` `vocab = 256` |
| **model scale** | 9.2M total | 7B byte-level; 300M+ token-level | **scale-mismatch**, BUT §4.1's 30M-and-below line is in fb's FAVOR at our scale, and it is the one line our `delta` is measured on (the induction-head formation claim, their setting 30M) | — |
| **data scale** | 8.2 MB over 2 000 steps (train set; ~2 000× tokens less than the smallest pass@1 experiment in the paper) | 91B-1T token passes; the smallest `loop` in §4 is 90 epochs on toy texts | **scale-mismatch**; §4.1's own setup trains on ~∞ epochs by toy-story standards, and multi-epoch is where the FB gains survive (§3.5) | `docs/reviews/ab-wave-2026-10-01.md` §0.3 recipe |
| **params the arm costs** | **196 864** (+2.1 %) = `256·768 + 256` (wave measured `params=9394318`, +196 864) | cost-fair by design in the paper (they shave trunk layers to keep params equal — §3 "we remove n−1 layers from the shared model trunk"); our +2.1 % params is a **compute match**, not a params match | **deliberately-ours** (we match COMPUTE not stored params — ours costs 2.1 % more params to give the aux arm its fair shot) | `future_byte.rs:85` (`FB_PARAMS_AT_SMALL = 256*768 + 256`); the wave ran the head on the SAME trunk as the control (no trunk shave). NOTE a §1.7-class defect found here: the doc comment above the constant says "`197 120 = 256*768 + 256`" — that arithmetic is wrong by 256 (the comment double-counts the bias row); the constant itself (196 864) is correct and matches the wave's `params=` header. One comment line to fix, not code |

### 4.3 What the wave actually measured

`aux_fb_weight 0.1, horizon 2, k=2` — and the arm scored **6.406 / 6.391 /
6.463, mean 6.4200** vs control 6.3433, **the widest loss of the wave**
(+0.150). Its train CE (best 2.998-3.040) is the worst of any completed
arm, and its own step-0 objective already costs **+0.58** of loss
(step-0 ce 6.141-6.150 vs control ~5.57 in the same family,
`docs/reviews/night-2026-10-02.md`).

**Nothing in the wave contradicts the paper. What the wave measured is: one
auxiliary term costs +0.15 held-out at 2k steps of overfit-window training,
with a fully-untied head at horizon 2, no transformer layer in the aux
branch, and a multi-CE objective where the aux is the LEAST aligned of the
three terms present (the other two are JEPA 0.05 and KoLeo).** The paper's
own multi-CE conditions favor larger aux budgets: they share `f_u` with the
main head and add full transformer layers per horizon; the biggest deltas in
their tables come from **code** benchmarks, at scales and data budgets we are
not in.

### 4.4 Verdict

**Class: hyperparam-wrong (n=2 instead of the byte-level n=8, weight shape
bare-linear instead of per-horizon transformer layer + shared unembedding) ×
scale-mismatch (8.2 MB vs 91B+ tokens).** The **one concretely fixable item**
is the head shape — and the fix is NOT free: a transformer-layer head at
horizon k means `k+1` full aux forwards or an MTP-style shallow-trunk design
(K3's "Number of MTP Layers 1" / MTP block), which doubles the already-2×
stepping cost the aux side already carries (JEPA teacher = full forward,
`schema.rs` doc line). That cost is not free at 9.2M params; the re-A/B
should at minimum re-run at **n=8** (`aux_fb_horizon=8`, byte-level optimum,
which the paper's own §3.3 named), with the head left bare-linear
(gated, honest), and NOT with the depth-2 head.

### 4.5 Fix-list
1. **Re-A/B at `aux_fb_horizon = 8`** — the paper's own byte-level shoulder;
   2404.19737 §3.3. Same 3-seed protocol, and note that horizon 8 at byte
   level is the paper's **most-transferable number to us** (a byte model
   predicting 8 future bytes matches their 7B byte-level result's own
   shape).
2. **Not-hidden-bug but hidden-choice: the aux objective is currently
   2404.19737-shaped ONLY in name; the paper does NOT prescripe `d_model→
   vocab` dense-no-trunk.** If the re-A/B at n=8 still ties, the honest next
   fix is a small `d_model → d_aux → vocab` two-layer head (a "f_{h_i}" with
   hidden = `d_model/4` = 192) rather than a bare linear, at +~150k params.
   That is also the shape the paper's memory-efficient implementation §2
   prefers.
3. **Open the JEPA-vs-fb combined test** (`aux_fb + jepa = 0.05 + 0.1` was
   never the same A/B as `fb-only`): queue row 1's winner is JEPA; queue
   row 1b's design said "against row 1, not the default recipe" — the wave
   ran fb ON TOP OF the JEPA-piece default, mixing two aux objectives. The
   paper's claim is a **plain-CE backbone**; 2404.19737 prescribes nothing
   about co-training a latent-Prediction aux. For a clean re-A/B, run
   `fb at n=8 with jepa_weight 0.0`, not `fb on top of jepa 0.05`.

### 4.6 Park-list
- **The transformer-layer aux head** — a real cost, gated on ×10.
- **n=41 / the >1 head family of the appendix** (n=6, n=16, all horizons
  simultaneously). The paper's own sweeps say 4-8 is the interesting range;
  costs `n` × readout.

---

## 5. The wave-level findings (cuts across all four hands)

### 5.1 The `retract_every` confound is still open and it is ±0.05

§7.2 of `docs/reviews/ab-wave-2026-10-01.md`: the 3-seed control family was
run at `retract_every=1`; the wave's four arms all ran at `--retract-every 4`
(the owner's working mode), and the one paired seed measured **-0.058** for
the cadence (6.387 @re1 vs **6.329** @re4 at seed 1, §0.3). **0.058 is 79 %
of the 0.0730 bar** — the bar is being read from a control family whose
single named measured difference vs the arm family is roughly the same size
as the bar itself. Two `retract_every=4` control seeds (2 and 3) close this
at ~50 min of GPU each; they are still owed as of `history.tsv:146`
("s1's 6.191 was seed spread" paired against a control mean that carries this
±0.05 systematic on every verdict in wave 3).

### 5.2 The bar itself is measured from 3 (not 4) numbers, one of which is
probably no longer the same run's family

The 0.0730 range comes from **6.387 / 6.314 / 6.329** — a 3-seed range. A
3-seed range is a **bound on one family's spread**, not on a null
hypothesis; the wave's own data (night-verdicts §"the table") shows
**attnres's range at 0.262 on the same window**, which makes 0.0730 a
single-family estimate of a quantity with cross-arm variation of ~3.6×. Two
improvements before wave 4:
1. **Fold the `retract_every=4` paired control seeds into the family** (§5.1)
   and **reprint the bar** with the cadence-matched population.
2. **Do not use a range as a p-threshold**: a range of 3 samples has
   expectation E(range) = μ ± 1.69σ for a normal-family (any stat-honest
   reader reads it as a spread estimate with large error bars), and the
   wave's own table shows a spread that varies 3.6× between arms. **The
   verdict of record (tie) is correct under the written rule; the written
   rule is much looser than the evidence needs** and the fix is a control
   population, not a new verdict call.

### 5.3 The `--set` seam and the log-line counters (the §1.1 debts)

`override.rs` now carries the arm keys (`use_situ`, `use_attnres`, `use_mhc`,
`mhc_streams`, `moe_topk`, `moe_lb_coef`, `aux_fb_weight` —
`crates/dormouse-core/src/config/override.rs:128-173`, coverage noted in the
module doc at `:16-17`) — the A/B wave-1 seam gap of 2026-10-01 is closed.
The four arms' eval-line counters (`attnres=`, `mhc=`, `situ=`,
`moe=`) are **still missing** — `crates/dormouse-core/src/probe.rs` counts them
(`probe::ATTNRES`, `MOE_ROUTE`, `MHC`, `SITU`) and the trainer's eval line at
`crates/dormouse-train/src/lib.rs:1652` prints only `fb=`, `engram=` and the
legacy seams. This is a §1.1 SILENT gap the wave report already listed as owed
(`ab-wave-2026-10-01.md` §7.1 point 3); **the gap is now load-bearing for a
re-A/B**: without `moe=`/`mhc=` on the line, a re-run that ties cannot show
the reader it engaged — that was exactly the check the `params=` header line
substituted for in wave 3.

---

## 6. Fix list and park list — the deliverables

### 6.1 Fix list (what to change before any re-A/B; the owner's decision on each)

| # | arm | change | paper's authority | cost class |
|---|---|---|---|---|
| F1 | moe | **k=2 of E=4-6**; real dispatch (`moe.rs` ponytail note) so routing changes the state | 2605.09165 §2.2 (k=2/8 Mixtral-form), §6.1 (divergence) | hyperparam + structural |
| F2 | moe | **token-count-invariant `L_LB`** (re-derive from the ratio `moe.rs` already prints) + optional router z-loss (ST-MoE Eq. 12) | ST-MoE; Switch §3.3 | hyperparam |
| F3 | mhc | **n = 4** (`mhc-streams 4`, +0.20 % params — 18 459 vs 6 155) | HC Tab. 1 (n=4 best); mHC Tab. 5 (n=4) | hyperparam |
| F4 | mhc | **leave the 10I identity init** — set `b_res` near `σ⁻¹(0.5)` or take HC's zero-init dynamic-half | HC §2.3 Eq. 14; mHC Tab. 5 α init | hyperparam |
| F5 | mhc | **fused Sinkhorn** after the §4.1 1e-7 floor and §4.2/§4.4's gate fixes land (`mhc-2026-09-30.md` §7.2-§7.4) | mHC §4.3.1 | implementation |
| F6 | attnres | **2 missing retract_every=4 control seeds** (a control-family fix, not an arm fix) | AB-PROTOCOL §2.6 | measurement |
| F7 | attnres | **learned RMSNorm gain γ** on the depth-attention's key norm | 2603.15031 §2.3 ref [66]; Tab. 4 w/o RMSNorm 1.743 | hyperparam |
| F8 | attnres | **s1 6.191 at a bigger step budget** (owner's own queue note) | — | budget |
| F9 | fb | **horizon 8** (the paper's byte-level optimum), same head shape | 2404.19737 §3.3 (byte 8, +67 % MBPP) | hyperparam |
| F10 | fb | **re-A/B with jepa_weight 0.0 on the fb arm** (the paper has no co-trained JEPA; the wave's fb measured fb+jepa+koleo) | 2404.19737 §3 (plain CE + n heads only) | hygiene |
| F11 | all | **print `moe=`/`mhc=`/`attnres=`/`situ=` on the eval line** (`crates/dormouse-train/src/lib.rs:1652` + `crates/dormouse-core/src/probe.rs` names exist) | ADR-0019 §1.1 | 1-line each |
| F12 | all | **extend the bar's population** (§5.2) before wave 4 | AB-PROTOCOL §2.6 | measurement |

### 6.2 Park list (×10-gated, named honestly, NOT deleted)

| # | arm | why it is parked | gate |
|---|---|---|---|
| P1 | moe | Sparse-Layers' own results are at 100-250M *unique* / 10B+ tokens with a μP-tuned LR transfer; ours is 9.2M × 8.2 MB. AND the FLOP-saving argument (what the mechanism is really FOR) is invisible until the gather/scatter dispatch lands, which is only worth doing once FFN is a measured hot spot (`loop_block.rs:583-591`: GEMMs are a small fraction of a step) | the FFN branch becomes measurable in `--timers`; or the ×10 gate |
| P2 | mhc | the composite-mapping stability story (`∏ H_res^i`, the paper's own §4.1) has nothing to stabilize at T=2-4 and `d=768`; the Amax-Gain-3000 collapse it exists to cure does not occur here | depth ×10, or the loop's residual becomes a measured instability site |
| P3 | attnres (Block) | Full ≡ Block at T=2-4 (Fig. 6 S=1); Block's Cross-stage caching / two-phase schedule (§4) target pipeline-parallel infra we do not run | N ≥ 8 blocks (= depth ×10) OR a pipeline-parallel training build |
| P4 | fb (transformer-head) | the paper's aux heads are `f_{h_i}` transformer layers, not bare linears; the trunk-saving shape is a real memory commitment for a 2 % param arm | aux head invests >2 % params, or the parent path lowers to ×10 |
| P5 | moe/expert TSCT-vs-SwiGLU | Sparse-Layers's experts are SwiGLU; ours is TSCT — a deliberately-ours deviation, not a bug, but it changes what "an expert" specializes over; a same-activation-function expert arm is one `ExpertFFN::new` flag away | the TSCT-vs-dense row (AB queue row 2, dense FFN) lands first |

### 6.3 What this audit does NOT claim
- **None of the four ties is a verdict on the mechanisms.** Two of the four
  (moe, fb) were run at configurations the papers explicitly do not
  recommend for our scale; one (mhc) at a rung the base paper itself
  measures as the second-worst point on its own ladder (n=2 of 1/2/4/8 with
  an exponential-stiffness identity init); one (attnres) as a FULL
  implementation whose formula the papers measure at a scale and budget that
  makes the size of their 1.746-1.737 deltas (0.009 - 0.030) UNRESOLVABLE at
  0.0730 seed noise.
- **Every "fixed" named above with a `file:line` is a RECOMMENDATION, not an
  applied change.** No config number moved in this lane; the choices are the
  owner's (§1.6 of the rulebook).

---

## 7. Commits, provenance, and the process trail

| claim | source (paper side) | source (our side) |
|---|---|---|
| moe: k=2/8, E=8, LB+z-loss, and the 25-53 % divergence stat | 2605.09165v2 §2.2, §6.1 (fetched 2026-10-02, HTML full text) | `crates/dormouse-core/src/moe.rs:22-43,86-146`; `configs/small.toml:16,35-37`; `docs/reviews/moe-routing-2026-10-01.md` §6.1-§6.2 |
| mhc: n=4, t=20, ε=1e-20, α=0.01, 6.7 % overhead, -0.021 loss at 27B, H_res -0.022 / H_pre +0.003 | 2512.24880v2 §3-§5, Tab. 1-5 (HTML full text, fetched 2026-10-02); 2409.19606v3 Tab. 1/3 | `vendor/dormouse-fused/crates/burn-mhc/src/{block,sinkhorn,sinkhorn_cuda}.rs`; `crates/dormouse-core/src/loop_block.rs:979-998`; `docs/reviews/mhc-2026-09-30.md` §1-§7 |
| attnres: Fig. 2, Eq. 1-6, Tab. 2/4/6, Tab. 5 fn 2/3, §5 zero-init | `docs/papers/2603.15031-attention-residuals.pdf` (extracted 2026-10-02, 1439 lines; sha in `docs/papers/provenance.tsv`) | `vendor/dormouse-fused/crates/burn-attnres/src/lib.rs`; `crates/dormouse-core/src/loop_block.rs:659-664,951-964,707-718`; `docs/research/2026-09-30-attnres-integration.md` |
| fb: n=8 byte-level, 30M-nonembedding cutoff, f_u-sharing, aux-loss ≠ weight shape | 2404.19737v1 §2, §3.1/3.3/4.1/5 (HTML full text, fetched 2026-10-02); abstract | `crates/dormouse-core/src/future_byte.rs`; `configs/small.toml:29-30`; `docs/reviews/future-byte-2026-09-30.md`; `docs/reviews/night-verdicts-2026-10-02.md` |
| bar / verdict numbers and the retract confound | — | `docs/reviews/night-verdicts-2026-10-02.md`; `docs/reviews/ab-wave-2026-10-01.md` §0.1/§0.3/§6.3/§7.2; `benches/history.tsv:138,141,144-147` |
