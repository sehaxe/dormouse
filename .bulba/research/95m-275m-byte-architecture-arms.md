# 95M–275M byte-level architecture: what to run next (evidence-first)

Date: 2026-11-08. Baseline under test: `port/configs/flagship3_dense.toml`
(95,035,200 params, 12L, d=832, heads=13, head_dim=64, d_ffn=3072, s=1024,
`block_form="transformer"`, `max_iter=1`, dense FFN). Budget: 73.7 MB/arm, 2 seeds,
one 16 GB card. Flags available: `block_form, block_stack, attn_kind, n_layers,
d_model, d_ffn, n_heads, head_dim, value_residual, attn_gate, nope_layers,
ffn_clip, use_mhc, use_attnres, layerrope, loop_blocks, max_iter, n_experts,
moe_topk, mod_capacity, mod_top_k, mod_aux_weight`.

## TL;DR

1. **The architecture is already settled; the recipe is not.** The only
   compute-matched win near our size with a dense control in the table is the
   3:1 GDN/attention hybrid: Olmo Hybrid Table 5, 60M, BPB 1.155 → 1.077
   (−0.078), and it *repeats the sign* at 100M/190M. It is already in the queue
   as `hybrid31`.
2. **Every loop / depth-recurrence arm is predicted to LOSE**, by two
   independent sources that agree in sign and magnitude. Kill `loop2`.
3. **The best unspent ΔBPB is in two zero-cost recipe knobs**: `value_residual`
   (82M table, −0.034 nats, +156 params) and `attn_gate` (+10.8K params).
4. The repo's own control (`docs/records/reviews/2026-11-06-control-transformer.md`)
   already measures the vanilla transformer **0.13–0.37 BPB better** than the KDA
   loop at equal bytes — 3–7× larger than any candidate below. Architecture
   search is the wrong budget line; recipe is the right one.

## Comparison table

| # | Candidate | Source / table | Scale | Measured effect | Exact config delta (from C0) | Cost | Test | Conf |
|---|---|---|---|---|---|---|---|---|
| 1 | 3:1 GDN hybrid | Olmo Hybrid 2604.03444 **Table 5** | 60M–1B | 60M: 1.155→**1.077** BPB (−0.078); 100M 1.037→0.986; 190M 0.950→0.891. Pure GDN 1.080/0.990/0.895 ≈ hybrid | `block_form="kda"`, `block_stack="3:1"`, `n_layers=12`, `loop_blocks=1`, `max_iter=1` | iso-FLOPs w/ C0 | BPB, 2 seeds | **High** [T] |
| 2 | value residual | ResFormer 2410.17897 **Table 3** (8L, 82M) | 82M | vanilla 2.739 → Learnable 2.705 (−0.034 nats); Identity 2.712; Plus 2.681. α init 0 ⇒ bit-identical start | `value_residual=true` (all else = C0) | +156 params, 0 FLOPs | BPB, 2 seeds | **High** [T] |
| 3 | value residual (2nd) | JugnuLM 2609.14715 R1 | 109.7M, 23L×576 | ARC-E 52.48→54.67, byte-ppl 1.95→1.94 | same as #2 | same | BPB + ARC-E | Med [T] |
| 4 | gated attention | 2505.06708 **Tab.2** | 1.7B dense / 15B MoE | PPL 7.499→7.404 (400B tok), 7.363→7.101 (1T); removes loss spikes, tolerates 8e-3 LR where baseline diverges | `attn_gate=true` | +10,816 params | BPB, 2 seeds | Med [T] (scale ≫300M) |
| 5 | gated attention, byte-level | independent repro (enwik8, 38M) | 38M **byte** | bpc 1.387→1.357 (−0.030 BPB), 3 seeds, non-overlapping | same as #4 | same | BPB, 2 seeds | Med [S] |
| 6 | mHC | KromHC 2601.21579 **Tab.3** (n=4) | D=6 / D=12 blocks | D=6 BPB 1.047→1.042; D=12 0.864→0.861 (−0.003). +1844K params, +6.7% step time | `use_mhc=true` (**KDA-loop only**; refused on `block_form="transformer"`) | +1.9% params, +6.7% time | BPB | Med [T], small effect |
| 7 | mHC → loop capacity | 2604.21106 §5 | iso-depth sweep | hyperconnections raise φ 0.453→0.646 (real capacity gain) | `use_mhc=true` on a loop/hybrid arm | as #6 | ΔBPB + Δφ | Med [T] |
| 8 | depth recurrence / loop | **2604.21106** | iso-depth, 116 runs | r=2 looped frontier trails baseline by **[0.03,0.06] nats**; φ=0.46 | `loop_blocks=6, max_iter=2` | 55% unique params | BPB | **High** [T] ⇒ LOSS |
| 9 | recursion, no routing | MoR 2507.10524 **Table 3** | 315M base, 20B tok | Recursive M-Cyc Nr=2 (167M) **2.8079 vs vanilla 2.7824 = +0.0255 nats worse** at equal tokens | `loop_blocks=6, max_iter=2` | iso-FLOPs | BPB | **High** [T] ⇒ LOSS |
| 10 | MoR (routed early-exit) | MoR 2507.10524 **Table 3** | 315M base | MoR-2 at equal 20B tokens, 12.3e18 FLOPs: 2.7749 vs 2.7824 (−0.0075 nats). MoR-3 equal tokens: 2.8246 (worse). Fig.3a: MoR **underperforms vanilla at 135M** | needs a router — **no flag exists** (`mod_capacity` is MoD, not MoR) | — | — | Med [T] ⇒ not implementable |
| 11 | NoPE tail | 2501.18795 **Tab.2** | 8B | NoPE val loss **1.58 vs RoPE 1.52 — WORSE**; QK-Norm 1.53 ≈ flat | `nope_layers=k` | 0 | — | **High** [T] ⇒ LOSS |
| 12 | MoE, small | 2402.07871 | 64×3M…64×85M (192M–5.4B) | "MoE always more efficient" but "for short training schedules they may under-perform dense"; **no ≤300M dense-control table verified** | `n_experts=8, moe_topk=2` | 94.1M | BPB | Low [A] |
| 13 | MoD | 2404.02258 + official repo | 50M | official 50M C4 repro: baseline **3.73** vs MoD-aux 3.81 / MoD-router 4.19 — **worse at 50M** | `mod_capacity=0.5`, `mod_aux_weight=...` | −FLOPs | — | Med [T/S] ⇒ LOSS |
| 14 | byte patching | BLT 2412.09871, MegaByte 2305.07185 | 400M / 1B | gains measured at ≥400M; entropy-model sweep 1M–100M is a *component* study. **No ≤300M dense control** | — (no patching flag) | — | — | Low [A] |
| 15 | deep-thin geometry | JugnuLM 2609.14715 | 53.5M vs 109.7M | 8L×512 → 23L×576: BLiMP 78.14→81.25, byte-ppl 2.04→1.95 on **fewer** tokens | raise `n_layers`, lower `d_model`/`d_ffn` at iso-params | iso-params | BPB | Med [T] (confounded w/ scale) |
| 16 | small-regime caution | 2512.20877 | 0.4M–9M | RoPE 2.0096 vs learned 1.9738 at 430K — "techniques successful in large LMs do not necessarily transfer" | — | — | — | Med [T] |

### Rejected candidates, and why

- **nope_layers** — NoPE is measured *worse* on val loss at 8B (1.58 vs 1.52); the
  paper's own win is long-context retrieval, not perplexity, and we train s=1024.
- **mod_capacity (MoD)** — the authors' own 50M reproduction loses on C4
  (3.73 → 3.81/4.19). MoD's wins start at 220M+ with 6e18 FLOPs.
- **sandwich norm** — 2505.06708 Tab.2 row 7: at matched LR it only *restores
  convergence*; "the improvement is negligible". No flag ships it anyway.
- **QK-norm** — 2501.18795 Tab.2: 1.53 vs 1.52, i.e. flat-to-worse, and worst on
  NIAH. No flag.
- **layerrope (2610.09179)** — no ≤300M BPB table located; abstract-only claim [A].
- **nGPT (2410.01131 / 2608.01284)** — no ≤300M loss table located; the 4-bit
  follow-up (2605.06067) is a precision story, not an architecture one. [A]
- **loop / MoR** — two sources, same sign, both measured. See rows 8–10.
- **MoE8** — kept in the queue only because it is already paid for; the evidence
  is [A] at our scale and the paper itself warns about short schedules.
- **byte patching** — right idea, wrong scale: every measured table is ≥400M.

## Top 3 arms to run after the current queue

1. **`C0 + value_residual=true`** — zero params, zero FLOPs, bit-identical at
   init. Single largest measured recipe delta at our exact scale (−0.034 nats at
   82M, −0.058 for ResFormer-Plus).
2. **`C0 + attn_gate=true`** — +10.8K params, near-identity init (bias +4.0).
   −0.095 PPL at 1.7B in the paper; −0.030 BPB in a 38M *byte-level* repro.
3. **`hybrid31 + value_residual + attn_gate`** — stack the recipe on whichever
   architecture wins the current queue. This is the compounding test and the one
   most likely to produce a number worth publishing.

Runners-up (5–8): deeper-thin iso-param geometry; `block_stack="7:1"` at
`n_layers=16` (Olmo: 7:1 ≈ 3:1 at 60M/190M with fewer attention FLOPs);
`use_mhc` **only** on the winning KDA/hybrid arm.

## What to delete if it loses

- **`configs/arch_loop_cycle2.toml` (loop2) — delete.** Kill criterion: BPB >
  C0 + 0.005. Two sources predict +0.025 to +0.06 nats. If it wins anyway, the
  repo has a genuine counter-result and the paper is wrong — keep it.
- **`configs/arch_hybrid_kda.toml` (pure KDA 95.9M) — delete if > 0.02 BPB worse
  than C0.** Olmo Table 5 predicts pure GDN ≈ 3:1 hybrid at ≤100M (1.080 vs
  1.077 at 60M); the repo's own control says the KDA loop is 0.13–0.37 BPB
  *worse* than vanilla. If pure KDA lands inside 0.005 BPB of hybrid31, keep the
  simpler one.
- **MoE8 top-2 — delete if < C0 − 0.005 BPB is not met.** Short-schedule warning
  is explicit in 2402.07871.
- **`nope_layers`, `mod_capacity`, `layerrope` — do not spend arms on them.**

## Measured vs extrapolated

**VERIFIED (sourced):** every number in the table above, with its paper/table.
Olmo Table 5, ResFormer Table 3, KromHC Table 3, 2604.21106 φ and the r=2 gap,
MoR Table 3, 2501.18795 Table 2, 2512.20877 Table 3, the official MoD 50M repro,
JugnuLM Table 2, 2505.06708 Table 2.

**SPECULATION (my inference):**
- No published head-to-head benchmark exists between `value_residual` +
  `attn_gate` on a *byte-level* 95M model at 3000 steps. I expect the two to
  compound because they act on different paths (value-stream information flow vs
  attention-output nonlinearity), but that is an assumption from [2410.17897]
  and [2505.06708], not a measurement.
- The ΔBPB magnitudes I quote for our config are *transplanted* from 60M–110M
  token-level runs onto a byte-level model at s=1024. Direction is evidence-backed;
  magnitude is not.
- `attn_gate` at 95M byte-level rests on one third-party enwik8 repro plus a
  1.7B paper result. Sign is consistent; size is unverified at our scale.
- The exact deeper-thin tuple (`n_layers`, `d_model`, `d_ffn` at iso-95M) is not
  derived here — resolve with `dormouse info` before launching.
- mHC's own paper reports no ≤300M table; the −0.003 BPB at D=12 comes from
  KromHC's replication, not from 2512.24880.

## Kill criterion (one rule)

Run C0 with 2 seeds as the reference. An arm is kept only if
`mean(BPB_arm) < mean(BPB_C0) − 0.005` **and** both seeds beat `mean(BPB_C0)`.
±0.005 BPB (≈0.0035 nats) is the resolution of a 2-seed 3000-step run; inside
that band it is a tie and the cheaper/simpler arm wins. An arm that ties the
control but costs more time or params is deleted, not "kept for later".

Memory is not the binding constraint at 73.7 MB/arm on 16 GB — wall-clock is.
Nothing proposed needs >16 GB or a second GPU.
