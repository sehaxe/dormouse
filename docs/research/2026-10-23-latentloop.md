# LatentLoop (arXiv:2610.19975v1) — design memo for THIS repo

**Read:** the full paper (24 pages: abstract, §1-6, Limitations, Appendices A-E, Tables 1-16,
Figures 1-8), from `https://arxiv.org/html/2610.19975v1` + the alphaXiv PDF (both fetched
2026-10-23; page refs below are the paper's printed page numbers). The released code
`github.com/lcqysl/LatentLoop` (9 model files, 9 configs, `pretrain.py`, `inference.py`; no
experiment variants — the README says internal run dirs and analysis tools are omitted).
Repo facts read from `src/dormouse/model/{loopblock,byteflow,model,kda}.py`,
`src/dormouse/{config,train/optim}.py`, `docs/reviews/{launch-atlas-2026-10-02,2026-10-21-battery,2026-10-23-architecture-verdict}.md`,
`docs/research/2026-10-21-adaptivity.md`, `docs/decisions/2026-10-15-1b-readiness-plan.md`,
`docs/protocols/AB-PROTOCOL.md`.

**Marking convention.** `[V]` = verified against the paper page or a file:line. `[S]` =
SPECULATION (my inference, no measurement behind it). `[U]` = a claim in the task brief I could
**not** find in the written record (flagged, not repeated as fact).

---

## TL;DR

1. **The paper's real finding is a placement result and it says our Plan (a) is the wrong
   placement**: compressing the *last* visit costs **+0.0067 NLL**, compressing the *first*
   gains **−0.0120**, measured against the correct control (untied attention, no compression,
   2.0319 — **not** the 2.0293 the brief compares to) `[V]` Table 13 p.22 + Table 5 p.10.
2. **But their last visit is their *only* valuable visit** (a 2-visit loop, +0.0133 from the
   second visit alone) `[V]` Table 3 p.9. Our last visit is our cheapest (0.01 of 0.35 CE at 1B,
   ~3 %) `[U→our brief]`. Their placement result therefore **does not transfer**, and the
   transfer argument runs the other way: the mechanism they measured is exactly the one that
   should be *safe* for us.
3. **The writeback coefficients are the load-bearing part, not the pooling** — a *single learned
   scalar* recovers 84 % of their writeback gain (2.0210 vs 2.0199, direct-writeback control
   2.0268) `[V]` Table 4 p.10. Their learned per-channel coefficients settle at **0.199 / 0.160**
   — the model wants to damp the pooled update by ~5× `[V]` Fig. 6 p.11. That is the evidence
   that a compressed visit degenerates gracefully into a *soft skip*, which is exactly what we
   want.
4. **Our block already has a learned per-position, per-branch write gate**
   (`loopblock.py:399-401`, sigmoid, one controller for all visits) `[V]`. A per-visit residual
   coefficient (their §3.1 diagnostic, the "cheapest possible port") is therefore **mostly
   redundant here** — it is worth adding as a *logged diagnostic*, not as an A/B arm composed
   against the gates.
5. **The measured prize is ~10 % of step time, not 18-25 %.** One iteration is 20.8 % of the
   medium step (119 of 573 ms, GPU 100 %) `[V]`; compression keeps the *kernel count* and shrinks
   the *rows*, so it buys `(1 − 1/k)` of that iteration only — 10.4 % at k=2, 15.6 % at k=4
   `[S, arithmetic]`. At `small` it buys ~nothing (launch-bound, 27 µs/launch, GPU 13 % `[V]`).

---

## 0. Evidence ledger

### VERIFIED (sourced)

| claim | source |
|---|---|
| learned residual coefficients (0.27, 0.87) @1.3B, (0.36, 1.09) @3B; α per visit, same α for attn and MoE branches, all layers of that visit, entry/exit unchanged, both start at 1 | p.4, Eq. 1 |
| fixed sweep {0.125,0.25,0.5,2}: reducing the **first** does not hurt (improves), reducing the **second** hurts; **(0.25,1) best at both sizes**, −0.008 / −0.003 NLL vs (1,1); (0.25,1) < (0.5,0.5) symmetric | p.4, Fig. 2(b) |
| with 3 and 4 visits the learned coefficients **increase with visit index** (first smallest, last largest); consecutive ratios ≈2.4× / 2.0× at 90 % progress; "we leave explaining this pattern to future work" | p.11 Fig. 5, p.17 Limitations |
| final forward: `Z=Enc(H)`, `Δ=F1(Z)−Z`, `H̃=H+Dec(Δ)`, `Y=F2(H̃)`; H is **preserved** at full resolution throughout | p.6 Eq. 2 |
| encoder = parameter-free mean of adjacent pairs, k=2; decoder = per-channel **and per-receiver-position** coefficient `w_s`, `Dec(Δ)_{kj+s}=w_s⊙Δ_j`, positions before k get zero; `w=1+δ`, δ zero-init; **2d params** at k=2 | p.6 Eqs. 3-4, p.19 |
| causal alignment: first receiver = last token of the source pair, second = the next token; token 1 receives nothing | p.6 |
| shared: MoE routers + expert weights. separate: attention blocks. reason: MoE is most of the params (untying would make it "close to a non-looped Transformer"); attention is cheap — one stored expert per layer (76→75 @3B) pays for it; KV already separate per visit | p.7, §3.4 |
| code matches: `attn_repeat` + separate norms built per middle layer, `self.mlp` shared, KV cache per layer *execution* | `LatentLoop/model/model.py:50-57, 93-94, 188-197` |
| compute reallocation to width: d 1296 vs 1184 (+9.5 %), experts 75 vs 92 (−18.5 %), head_dim 140 vs 118, FLOPs/step +0.5 % vs Base | Table 6/7 p.17 |
| strategy ablation @3B/40B: Base 2.0426, Loop 2.0293, residual-width 2.0361, expert-width 2.0364, active-experts 2.0349, **sequence length 2.0303**, LatentLoop 2.0199 | Table 3 p.9 |
| placement ablation @3B/40B: full/full 2.0293 · latent/full **2.0199** · full/latent **2.0386** · latent/latent **2.0467** (both-compressed gets width 1296→1440 and still loses) | Table 13 p.22 |
| untied attention *alone* is **worse**: 2.0293 → 2.0319 (shared MoE, no compression) | Table 5 p.10 |
| k sweep: k=2 2.0199, k=3 2.0257, k=4 2.0297; delaying both receivers by one token alone: 2.0269 (+0.0070) | Table 14 p.23 |
| learned writeback means 0.199 / 0.160; nearer receiver > farther; a +1-token delay drops them to 0.161 / 0.138 (−19 % / −14 %) | Fig. 6 p.11, Fig. 8 p.23 |
| the 19.0 % / 7.9 % are **fitted power-law frontier** numbers at NLL 1.988, not measured wall-clock ("predicted"; dashed segments are extrapolation) | abstract p.1, §4.2 p.8, Fig. 4 |
| measured wall-clock: LatentLoop uses ~9 % / 13 % **less GPU time than MoE Loop** at 3B / 7B, "similar" at 1.3B; 1.3B baselines ran on 8 GPUs vs 64 for LatentLoop (their own caveat) | p.21, Fig. 7 |

### VERIFIED (this repo)

| claim | source |
|---|---|
| loop = one `LoopBlock` applied `max_iter` times; `out_acc/ran` mean readout; per-iteration CE inside the loop | `loopblock.py:369, 487-488, 510` |
| one **shared** learned scalar `residual_scale` (shape `(1,)`, init 1.0) multiplies every visit's write | `loopblock.py:187, 485` |
| per-position, per-branch learned gates `sigmoid(w_attn/w_mem/w_ffn)` from ONE controller over `[ctx; h0]` | `loopblock.py:399-401, 461` |
| no RoPE anywhere in the loop block or KDA — position enters only through the causal recursion | `loopblock.py`/`kda.py` (no `rope`/`rotary`) |
| medium step 573 ms (fwd 194.5 / bwd 282.0 / clip 18.3 / opt 78.3), GPU 100 %; iteration ≈ 119 ms = **20.8 %** | `docs/research/2026-10-21-adaptivity.md:87-90` |
| `small`: 21 427 launches / 581.5 ms; max_iter 4→2 removes 7 778 launches (36.3 %) and 284.9 ms → iteration = **18.1 % launches, ~24.5 % wall** | `launch-atlas-2026-10-02.md:100`, `adaptivity.md:41-45` |
| ByteFlow channel: **−0.4848 BPB** medium vs same-geometry control (−0.4276 capacity-matched); **fp32-only, Engram-incompatible**; profile encode 2.3 % / **loop 95.9 %** / decode 1.8 % (fp32, naive MoE) | `2026-10-21-battery.md:44-48, 60`; `2026-10-23-architecture-verdict.md:37` |
| ByteFlow chunker = **global top-k over the window** (`select_positions`), and its streaming twin `RatePatcher.push` takes a **host** `list[float]` | `byteflow.py:537, 662, 730-751` |
| 9.2M per-iteration CE: iter1 ≈2.075-2.086, iters 2-3 ≈2.13-2.14, iter4 ≈2.12; inference best at depth 1-2 | `2026-10-15-1b-readiness-plan.md:478-479` |
| AdamW gets one group with `weight_decay=wd` (0.01 default) covering `residual_scale` | `optim.py:1125`, `config.py:735` |
| A/B rule: 3 seeds, fix batch/seq, treatment must beat the control's mean by more than the control's spread | `docs/protocols/AB-PROTOCOL.md` |

### NOT VERIFIED / SPECULATION

- `[U]` **"per-iteration CE on the live 1B run 2.15 / 1.86 / 1.81 / 1.80"** — I found no written
  record of this (searched `docs/**`, `~/logs`, `/tmp/opencode`). It contradicts the 9.2M record,
  which is **non-monotone** (iter 1 best). I use the brief's number but the contradiction is
  load-bearing, see §3(a).
- `[U]` **"attention is ~5 % of our step"** — no measured per-arm time split exists in the tree.
  By hand-count at medium (q/k/v/g = 4d², MoE = 2·3·d·d_ffn) attention is **~24 %** of per-token
  matmul MACs. Either number kills port (b) (see §3(b)).
- `[S]` all step-time savings arithmetic below (rows scale, launches do not).
- `[S]` the transfer of their placement result to a 4-visit loop.

---

## 1. What they do, exactly

### 1.1 The diagnostic: per-visit residual coefficients (§3.1, p.3-4)

Two learnable scalars, one per visit, multiplying every residual branch of the repeated middle
block (`h' = h + α_v f_ℓ(h)`, Eq. 1, p.4). Same α for attention and MoE, all layers in that
visit; entry/exit untouched; both init at 1. Learned jointly at 1.3B/20B and 3B/40B tokens
(same data/settings as their fixed-coefficient controls). Result: **(0.27, 0.87)** and
**(0.36, 1.09)** — the first visit wants ~⅓ of the write, the second ~1.

Then a from-scratch grid: hold one visit at 1, sweep the other over {0.125, 0.25, 0.5, 2}
(Fig. 2b, p.4). The cells the caption/text pin down at 3B (the rest of the heatmap is figure-only;
I read 2.037 at one extreme of the second-coefficient axis, so ±0.001 on any cell):

- **(0.25, 1) = 2.026 — the grid minimum at both sizes.**
- (0.5, 1) = (0.5, 0.5) = 2.027 → symmetric 1/N scaling is **worse** than the asymmetric best
  (their direct shot at Wang et al. 2026a, p.4).
- (0.125, 1) = 2.027 → over-damping the first visit gives back the gain (the optimum is interior).
- (1, 1) = 2.029 (the MoE Loop number, Table 3) → −0.003.
- any change to the **second** coefficient away from 1 is ≤ (1,1) or worse; 2× is worse.

So the whole effect is "the first visit should write at ~¼, the second at 1". Cost: zero extra
parameters, zero FLOPs.

With 3 and 4 visits (Fig. 5, p.11) the coefficients **rise** with visit index (ratios ≈2.4× /
2.0×), i.e. later visits write *harder*. They explicitly do not explain it (Limitations, p.17).

### 1.2 Final architecture (§3.2-3.4, p.4-7; Fig. 3 p.5)

```
entry block (L/4 layers, once)          → H  [T × d]   (full resolution, PRESERVED)
Enc: Z_j = (H_{2j-1} + H_{2j}) / 2      → Z  [T/2 × d] (parameter-free, k=2)
first middle-block pass at T/k          → Δ = F1(Z) − Z
Dec: Dec(Δ)_{kj+s} = w_s ⊙ Δ_j          → H̃ = H + Dec(Δ)   (w ∈ R^d per slot, 1+δ, zero-init)
second middle-block pass at full T      → Y = F2(H̃)
exit block (L/4 layers, once)
```

Causality of the writeback (p.6): receivers are the **last token of the source pair** and the
**next token**; the first token of the sequence gets nothing. At decode this is *just in time*,
not lagged: the pair `[2j, 2j+1]` completes when `2j+1` arrives, and position `2j+1` is the one
that predicts `2j+2` (Appendix B, "Causal incremental decoding", p.20 — an unfinished pair is
buffered, RoPE uses the pair's last-token position `2j−1` zero-based).

Encoder/decoder ablation (Table 4, p.10): mean+direct 2.0268; softmax+softmax (SpiralFormer-style)
2.0264; conv-encoder+direct 2.0219; **mean+SD 2.0199**; conv+SD 2.0209 (no gain from learning
both); SD with RMS-normalized coefficients 2.0228 (worse → the **overall scale** matters, not the
shape); **scalar gain only 2.0210**. That last row is the most important number in the paper for
us: **one learned scalar buys 0.0058 of the 0.0069 total writeback gain (84 %)**.

Learned writeback coefficients (Fig. 6, p.11): channel means **0.199** (nearer receiver) and
**0.160** (farther), i.e. the pooled increment is written at ~1/5 strength. Fig. 8 (p.23): a
+1-token delay drops them to 0.161/0.138, and the delayed model's `w0` (0.161) lands on the
original's `w1` (0.160) — their cleanest mechanistic result: **write strength tracks how recent
the information is**, not the slot index.

### 1.3 What is shared vs separate (§3.4, p.7)

- **Shared:** MoE routers and expert weights ("they contain most of the model's parameters";
  untying MoE "would leave most parameters unshared … close to a non-looped Transformer").
- **Separate:** both attention blocks of the middle stack, because the two visits see different
  sequence resolutions. Paid for by dropping **one** stored expert (76→75 @3B), keeping the
  active count at 8 so expert FLOPs are unchanged; KV is already per-visit, so no KV growth
  (p.7, Table 7 p.17).
- Code adds something the paper never mentions: **separate RMSNorms for the repeat visit**
  (`norm_1_repeat`, `norm_2_repeat`, `model.py:54-55, 190-191`). Divergence #2 below.

Ablations: untied attention alone is **worse** (2.0293 → 2.0319, Table 5); it only pays when the
first visit is compressed (2.0303 → 2.0268 with direct writeback; 2.0237 → 2.0199 with SD).
Untying MoE loses badly in both columns (2.0348 / 2.0340, Table 12 p.22 — the expert pool would
shrink 76→51 / 75→50 to pay for it). So: **the only reason to untie attention is a resolution
mismatch, and the price is one expert.**

### 1.4 Placement ablation — the number that decides our port (a)

Table 13 p.22, 3B/40B, every cell budget-matched to identical params/KV/FLOPs/steps:

| first visit | second visit | NLL |
|---|---|---|
| full | full (MoE Loop) | 2.0293 |
| **latent (k=2)** | **full** | **2.0199** |
| full | latent | 2.0386 |
| latent | latent | 2.0467 (width 1296→1440, still worst) |

**Isolation correction.** The three latent cells carry untied attention + SD; the full/full cell
is plain MoE Loop. The clean control is **full/full *with untied attention* = 2.0319** (Table 5).
Against it: compressing the **first** visit = **−0.0120**; compressing the **second** = **+0.0067**;
both = **+0.0148**. So a compressed visit **loses about half of a full visit's value** (0.0067 of
the 0.0133 that the extra visit is worth against MoE Base), and the placement sign is unambiguous
*in their architecture*.

### 1.5 Honest caveats they state — and the one they don't

Stated: the residual-coefficient mechanism is "incompletely understood", especially at 3-4 visits
(Limitations, p.17); the 1.3B wall-clock comparison mixes 8-GPU and 64-GPU pools (p.21); SpiralFormer
comparison uses quoted numbers because its implementation is not public (p.23); the learned
coefficient pattern with more visits is left to future work.

Not stated as a caveat, but true and important: **19.0 % / 7.9 % are fitted-frontier predictions at
NLL 1.988, not measured wall-clock** (abstract says "predicted"; Fig. 4 marks extrapolated segments
dashed). The measured wall-clock they do report is **−9 % / −13 % vs MoE Loop** at 3B/7B (Fig. 7,
p.21) — a much smaller number, and the one to compare to our step budget.

### 1.6 Paper vs code divergences

1. **No residual coefficients in the released code.** All residual adds are hard-coded
   `torch.add(..., alpha=1.0)` (`model/model.py:193, 196, 197`), and no α/`residual_coef` exists
   anywhere in the repo. §3.1 and Fig. 2 are **not reproducible from the release**; only the final
   architecture and the two baselines are.
2. **Separate RMSNorms** for the second visit (`model.py:54-55, 190-191`) — the paper says
   "independent attention blocks" and is silent on norms. The released LatentLoop therefore unties
   2 norms/layer beyond what §3.4 describes; their parameter accounting (Table 6) folds this in
   without naming it. Minor, but it means "untied attention" in the paper ≠ "untied attention" in
   the code.
3. Matches, checked: `w=1+δ` zero-init depthwise-conv decoder (`model/latent.py:7-22`; App. B p.19);
   receiver slots `(2j, 2j+1)` zero-based `[2j+1, 2j+2]` (`latent.py:25-58`, README); latent RoPE at
   the group's last token `2j−1` (`model.py:149-150`; App. B p.20); 8 query / 2 KV heads, 8 active,
   75 stored (`configs/3b/latentloop.json`, `model/config.py:14-19`); 15 endpoints (Table 16 p.24 =
   4+5+6); Table 16's 3B/40B row reproduces Tables 3/4/5 exactly (2.0426 / 2.0293 / 2.0199).
4. `Fig. 7` reading: MoE Loop 1.43 / 1.34 / 1.39 × MoE Base; LatentLoop 1.43 / 1.22 / 1.21 ×
   (1.3B / 3B / 7B) — consistent with the text's "9 % and 13 % less than MoE Loop at 3B/7B" and
   "similar at 1.3B".

---

## 2. Map onto this repo's measured facts

| paper | this repo | consequence |
|---|---|---|
| 2-visit loop, first visit cheap, second carries all the value (`+0.0133`) `[V]` Table 3 | **4-visit loop**; per-iter CE 2.15/1.86/1.81/1.80 at 1B → **iter 4 is worth 0.01 of 0.35** `[U]`; at 9.2M the curve is **non-monotone** (iter 1 best, iter 4 recovers) `[V]` plan:479 | their placement result is about compressing a **valuable** visit; ours would compress the **cheapest** one. Direction of transfer is favourable, strength unknown |
| encoder = mean over token positions; H preserved | stream is **bytes**; loop block has **no positional embedding at all** `[V]` | pooling needs **no RoPE surgery** here (unlike their `2j−1` remap). Cheaper than their port |
| writeback = 2d params/channel/position, learns ~0.2 | our block already has per-position sigmoid gates on attn/mem/ffn, one controller shared by all visits `[V]` `loopblock.py:399-401` | a per-visit scalar is a **coarser duplicate** of a knob we already own |
| one iteration ≈ their whole first visit (≈40 % of step in a 2-visit loop) | one iteration = **20.8 %** of a medium step (573 ms) `[V]`; 18.1 % launches / 24.5 % wall at small `[V]` | the prize is ~1/5 of the step, and only the *compute* half of it |
| MoE = most params, attention cheap | medium: KDA ≈ 4d² = 4.2 M MAC/token vs MoE topk-2 ≈ 16.8 M → attention ≈ 24 % of body MACs (with `out_proj`) `[S, hand count]` | untying attention duplicates ~4 M params/visit ≈ **+8.8 % params at 138M** for zero FLOP change |
| ByteFlow not in the paper | our learned chunker exists: −0.4848 BPB, **fp32-only, Engram-incompatible, 1.6× slower** `[V]`; loop = 95.9 % of its step `[V]` | see §4 — the chunker is **not** a drop-in and is **not** affordable at k=2-4 |

**The one structural mismatch that matters.** Their compressed visit is followed by a full visit
that can *repair* the pooled blur, and their exit block reads a single full-resolution readout
(no averaging). Ours averages `T` readouts (`out_acc/ran`, `loopblock.py:510`), so degrading the
last readout moves the mean by `1/T` of its deviation — the cost of a bad last visit is damped
twice (once by the writeback coefficient, once by the `1/T` average). `[S]` This is the strongest
structural argument that (a) is safer for us than for them.

---

## 3. Three candidate ports, ranked

### Rank 1 — (c) per-visit residual coefficients, as a **diagnostic**, not an A/B
*Cost: `T` params (3-4), ~6 LOC. Expected quality effect: ~0.*

Their diagnostic is a scalar per visit on the write. Our block already writes
`h = ctx + y * self.residual_scale` with **one** scalar for all visits (`loopblock.py:187, 485`),
and `y` already carries three learned per-position sigmoids per visit (`:399-401`), fed by a
controller that sees `iter_embed[i]` through `ctx`. So the model can already express "this visit
writes less", per position, and it has done so since the port landed. `[V]` code.

What is genuinely missing is the **measurement**: we do not log the per-visit gate means, so we
have never known whether our visits self-differentiate the way theirs do. **Port the diagnostic,
not the knob:** count `mean(w_attn), mean(w_mem), mean(w_ffn)` per iteration slot into the counter
line (0 params, ~6 LOC, no objective change). If the means are flat across visits, the paper's
diagnostic is falsified on our architecture and (c) is closed for good; if they diverge, we have
the per-visit scale for free and can skip the A/B.

If we do add a knob, parameterize it as `1 + δ` (δ zero-init), **not** a raw scale: our
`residual_scale` sits in the AdamW group with `weight_decay=0.01` (`optim.py:1125`,
`config.py:735`), so a raw scale is pulled toward 0 by `(1 − lr·wd)` every step — only ~0.6 % over
600 steps at lr 1e-3, i.e. a small bias today, but a confound at a higher `wd` or a longer horizon,
and exactly the artifact that would fake the paper's "later visits shrink" result. Their Appendix B
(p.19) gives this as the reason for `w = 1 + δ`.
`residual_scale * (1 + δ_i)` is **bit-identical** to today at δ=0 (multiplying by exactly 1.0).

**Pre-registered A/B (only if the diagnostic is inconclusive):** baseline vs per-visit δ,
3 seeds, 600 steps, medium — with the prediction **later visits get δ < 0** (opposite sign to their
coefficient *growth*), because our first visit is the valuable one. Decision rule: their
0.003-0.008 NLL at 3B is ~0.01 BPB; at 138M/600 steps the seed spread is of that order
(`spread 0.015-0.038` on the same protocol, `2026-10-23-architecture-verdict.md`) → **expect a
tie, and treat a tie as a close.**

### Rank 2 — (a) compress the LAST visit, k=2 and k=4
*Cost: ~90 production LOC + ~50 decode LOC + ~130 test LOC. Expected: −10.4 % step time (k=2) /
−15.6 % (k=4) for +0.003…+0.010 BPB. `[S]`*

**Why it is not refuted by their Table 13.** They compressed their *only* valuable visit and lost
~50 % of its value. We would compress a visit worth ~3 % of the loop's total improvement. If the
same 50 %-retention ratio holds, the cost is `0.50 × 0.01 ≈ 0.005 BPB`. `[S]` And our own 9.2M
record says the last visit's *own* readout is the **worst** of the four (2.12 vs 2.08 for iter 1)
`[V]` — in that regime compression may even help.

**Why the mechanism is sound even though bytes are not mergeable.** The full-resolution `H` is
preserved; only the *increment* is computed coarsely, and its per-channel/per-position coefficient
starts at 1 and is free to learn down. Their coefficients land at 0.199/0.160 `[V]` — i.e. the
model's own answer is "write the pooled thing at ⅕ strength at most positions". A compressed last
visit therefore degrades into a **learned soft skip**, not into a corrupted state. That is a
better-placed mechanism than our halt head (which measured **+0.026 with 0.079 spread**, i.e.
unstable, `2026-10-23-architecture-verdict.md:47`), because it is differentiable everywhere and
never branches.

**The measured saving is smaller than the brief assumes.** Compression shrinks **rows**, not
**launches**: same kernel count, smaller tiles. So it pays only in the compute-bound regime.
At medium (573 ms, GPU 100 %, one iteration = 119 ms = 20.8 %): k=2 saves `0.5 × 20.8 = 10.4 %`,
k=4 saves `0.75 × 20.8 = 15.6 %`. At `small` it saves ~0 % (launch-bound, 27 µs/launch, GPU 13 %).
`[S, arithmetic on V numbers]` At T=3 (the brief's "3 iterations") the same iteration is 119/454 =
26.2 % of the step → **12.3 % / 18.6 %**.

**Causality, stated exactly.** Zero-based pair `[2j, 2j+1]` may update positions **`2j+1` and
`2j+2` only** (never `2j`, whose prediction is byte `2j+1` — the pair's own second byte). `k=4`:
group `[4j..4j+3]` updates `4j+3 … 4j+6`. The mechanism is a gather `index_select` over a `[T/k,
k, d]` update tensor, exactly as `LatentLoop/model/latent.py:50-56` does it. Byte 0 gets no update.
No RoPE remap needed (our block has none) `[V]`.

**The A/B that decides it:** 3 arms × 3 seeds, 600 steps, medium, b8/s512, the live wave's data —
see §5 for the exact driver. Two-number acceptance: **ΔBPB ≤ +0.010** *and* **measured
`gpu_step` −≥8 %** (k=2) / **−≥13 %** (k=4). Report `periter=`/`fin=` on every eval so the
medium per-iteration curve finally enters the record.

### Rank 3 — (b) separate attention per visit — **reject**
*Cost: +12.15 M params at 138M (+8.8 %) for zero FLOP change. Expected: 0.*

Their own ablation says untied attention is a **patch for a resolution mismatch**, not a win:
without compression it *loses* (2.0293 → 2.0319), and the gain appears only in the compressed-first
column `[V]` Table 5. Our attention is KDA with an O(1) state and **no positional embedding** `[V]`
— there is no resolution to accommodate, because every visit already sees the same positions.

If (a) ships, the resolution mismatch *does* appear, and their result predicts untied attention
would recover part of the loss (2.0303 → 2.0268, i.e. **+0.0035** in the direct-writeback cell).
That is a follow-up arm for (a), not a standalone change. Note the sign of the trade: their
budget-matched untied arm pays with one expert (76→75), so the "gain" is partly the expert-count
change; ours would pay +8.8 % params for nothing we can measure.

**The A/B that decides it (if anyone insists):** baseline vs 4 separate KDA parameter sets, 3
seeds, 600 steps. Pre-register the expectation: **tie**, because KDA is O(1)-state and carries no
position signal that a per-visit copy could specialise.

---

## 4. What must NOT be copied

**1. The mean-pool encoder as a "merge".** Their encoder is parameter-free over **token**
positions, and it works because a token pair is a sub-word unit *and because H is preserved*. For
bytes, `(byte_{2j} + byte_{2j+1})/2` is not a merge of two units — it is a blur of two arbitrary
adjacent bytes. It is *admissible* here only because of the preservation property (nothing is
replaced) and the learned writeback (which can damp it). Do not describe it as compression of the
stream; it is compression of the **compute**, and the docstring must say so or a reader will
believe we are running the loop on a shorter byte stream.

**2. ByteFlow's chunker inside the loop — required? No. Affordable? No.** Three independent
reasons, all verifiable:

- **Not causal.** `select_positions` is a **global top-k over the whole window**
  (`byteflow.py:537`, called from `encode_chunks:662`). A boundary before position `t` can be
  displaced by a *future* position's gain, which changes `t`'s chunk latent, which changes `t`'s
  logits → the model sees bytes `> t` while predicting byte `t+1`. Inside a training loop that is
  a leak; the current channel hides it because the objective there is the decoder's byte CE over
  the same window and the selection is treated as "teacher-forced structure" (its own docstring,
  `byteflow.py:651-656`) — a logic we must not import into the loop.
- **Its causal twin cannot run in the step.** `RatePatcher.push(self, h: list[float])`
  (`byteflow.py:751`) reads the state to the **host** per position. A host read in the step is
  forbidden (ADR-0018, and it makes the step uncapturable — `cubecl` client `:1288`).
- **It costs more than it saves.** The front stage is a whole local stack over **T bytes**
  (`e_layers` encoder blocks + sliding-window attention). At k=2-4 on *one* visit we would save
  10-16 % of the step and pay an extra local encoder over every byte — at the measured channel
  shape the front+back stages are 4.1 % of a step *while the loop runs on K = T/4 patches*
  (`2026-10-21-battery.md:60`); at K = T/2 the encoder is a far bigger share and fp32-only
  (+60 % params, 1.6× slower). **Net loss at the k we care about.**

**Verdict for §4:** a learned chunker is **not required** (the mean pool is causally clean and
needs no new parameters) and **not affordable** at k=2-4. Revisit only if we ever want k=8+ on the
*whole* loop, which is a different architecture (their k=4 already costs +0.0098 NLL over k=2
`[V]` Table 14 — and that is with their cheap visit compressed, not ours).

**3. Their `2j−1` RoPE convention** — irrelevant here (no RoPE in the loop) but do not transcribe
it as if it were needed. The only positional fact our port needs is the receiver index.

**4. Their residual-coefficient *sign*.** Their coefficients **grow** with visit index; ours
should shrink if anything. Do not copy the direction as a prior. `[S]` for ours; `[V]` for theirs.

---

## 5. Implementation sketch — port (a), `compress_last_k`

### 5.1 Touched functions

| file | change | LOC |
|---|---|---|
| `model/loopblock.py::__init__` | `self.compress_last_k = int(getattr(cfg, "compress_last_k", 0))`; if >0 build `self.cw = nn.Parameter(torch.zeros(k, d))` (the `1+δ` writeback, per slot, per channel) — **built only when the flag is on**, the `loom`/`mhc` pattern (`:199-209`) | ~8 |
| `model/loopblock.py::_window` | at `i == ran-1` (the *actual* last, not `T-1`, so `depth<T` works) and `compress_last_k>0`, call `_compress_visit` instead of the inline body; keep `out_proj` at **full T** and keep the per-iteration CE at full T (see 5.2) | ~12 |
| `model/loopblock.py::_compress_visit` (new) | pool `h`→`z`; pool `h0`→`z0` (the controller's second input); run norm+controller+attn(+msa)+moe on `[b, t/k, d]`; `Δ = (z + e_i + y_z·residual_scale) − z`; write back with `index_select` over a `[b, t/k+1, k, d]` update tensor using `source=(p-1)//k`, `slot=(p-1)%k` (the `latent.py:50-56` gather, re-derived for our layout); return full-T `h` | ~35 |
| `config.py` | `compress_last_k: int = 0` + `validate`: `k in {0,2,4}`, `t % k == 0`, and the refusals of 5.3 | ~12 |
| `model/decode.py` + `loopblock.py::recurrent_step` | the pair/group buffer: retain the last `k−1` byte states, and a **separate KDA state for the compressed slot** (the loop already keeps one state per iteration slot, `loopblock.py:571-575`); emit the update to `kj+1 … kj+k-1` as the bytes arrive; **without this, train and decode diverge silently** — the one deliverable that must not be skipped | ~50 |
| `tests/test_compress_last.py` | (i) flag off = `torch.equal` vs today, (ii) causality: perturb byte `p`, every logit at `< p` must be bit-identical, (iii) decode parity: `recurrent_step` over the sequence equals the one-shot forward at k=2 and k=4, (iv) `t % k != 0` refused | ~130 |

Production LOC: **~90 model/config + ~50 decode**; tests **~130**.

### 5.2 Design decisions that are not optional

- **Keep the readout at full T.** `step_out = out_proj(h_full)` (the writeback already produced a
  full-T `h`). Otherwise the per-iteration CE at the compressed visit would be a CE over *pooled*
  states, i.e. a state asked to predict two different bytes — a corrupted loss term, not a
  cheaper one. Keeping `out_proj`+`lm_head` at T costs ~4.5 % of the iteration and keeps the
  objective untouched (the `_forward_segmented` argument, `loopblock.py:638-645`: the loss must
  not change shape under a compute flag).
- **`Δ` is relative to the pooled state, not the pooled `ctx`**: `Δ = h_z − z`, so `e_i`
  (iteration embedding) is inside the increment. Getting this wrong writes `e_i` into every visit.
- **The controller's `h0` input must be pooled too** (`ctrl = cat([ctx, h0])`, `:383`). Pooling
  only the first term silently changes the controller's semantics between visits.
- **The writeback is applied once, after the body, before `out_proj`** — matching their Eq. 2/4,
  not per-layer.
- **`ran` vs `T`**: the "last visit" is `i == ran-1`. With `depth=2` of `max_iter=4` the last
  visit is iteration 2; a `T-1` test would silently compress nothing (or the wrong visit).
- **`moe_capacity=2.0` confound**: the capacity-padded dispatch's drop set is a function of the
  token count (`loopblock.py:665-674`). Halving the rows changes the absolute per-expert capacity,
  so the arm differs in drop rate as well as in resolution. Acceptable (it is a model change), but
  it must be named in the report, and the medium preset's `moe_capacity=2.0` means it is not zero.

### 5.3 Refusals (each with its `Escape:` line, the `config.py:1800` house style)

| combination | why |
|---|---|
| `compress_last_k > 0` + `use_engram` | Engram hashes **byte ids** per position (`_memory`, `loopblock.py:584-611`); a pooled position has no byte id. The arm would be built, counted and trained by nothing — the exact defect class the counters exist for |
| + `segment_len > 0` | the pair grid would cross segment cuts and the KDA boundary state is per-position; the "memory decision, not a model change" contract (`loopblock.py:47-60`) would break |
| + `use_mhc` | `K=2` full-width lanes would each need pooling and the readout is the lane **sum** (`:477-482`); two unvalidated axes in one run |
| + `use_attnres` | the last visit's write becomes a mixture over sources at different positions (`:473-476`); the pooled increment has no single source |
| + `halt_mode` / `halt_weight > 0` | the halt head reads the last visit's readout and its target is the per-iteration CE's improvement; a compressed last visit changes both |
| + `use_byteflow` / `bf_net` | the loop already runs on K patch latents (`model.py:349`); compressing patch positions again is a different experiment. Same refusal shape as `config.py:1800` |
| `t % k != 0` or `t // k < 1` | the same divisor refusal `segment_len` already makes (`loopblock.py:255-262`) |

### 5.4 The 3-seed A/B driver

Arms: `base` / `k2` / `k4` × seeds 1-3 = **9 runs**. Everything except the flag copied from the
live wave, so the numbers are comparable to it.

```bash
# docs/protocols/AB-PROTOCOL.md: 3 seeds, ONE fixed window, quote the byte count.
# window = eval_batches(10) x batch(8) x seq_len(512) = 40 960 B
DATA=/home/sehaxe/dormouse/data_local/mix4/web.bin
EVAL=/home/sehaxe/dormouse/data_local/mix4/web_eval.bin
COMMON="--set steps=600 --set batch=8 --set seq_len=512 \
  --set data=$DATA --set eval_data=$EVAL --set eval=true --set eval_every=300 \
  --set eval_batches=10 --set eval_depths=true --set log_every=100 \
  --set ckpt_every=100000000 --set moe_kernel=triton --set ns_kind=triton --set ns_steps=5 \
  --set opt=mix --set muon_colrow_3d=colrow --set opt_graph=false \
  --set use_mhc=true --set share_u=true"     # <- the running wave's flagship arm; see note
for seed in 1 2 3; do
  .venv/bin/dormouse train medium $COMMON --set seed=$seed \
      --set ckpt_name=cl_base_s$seed --set out_dir=/tmp/opencode/cl_ab &   # wait for GPU
  .venv/bin/dormouse train medium $COMMON --set seed=$seed --set compress_last_k=2 \
      --set ckpt_name=cl_k2_s$seed   --set out_dir=/tmp/opencode/cl_ab &
  .venv/bin/dormouse train medium $COMMON --set seed=$seed --set compress_last_k=4 \
      --set ckpt_name=cl_k4_s$seed   --set out_dir=/tmp/opencode/cl_ab &
done
```

Rules for the driver, non-negotiable:
- **One GPU process at a time** (the `engram_ab.sh` gate, `scripts/engram_ab.sh:24-38`) — the
  compare must run the arms *serially*, not the three-way `&` above; that `&` is a placeholder for
  "queue three commands".
- **Do not change two axes.** If the `use_mhc`/`share_u` wave is still in flight, wait for its
  verdict and use its winner as the control; otherwise this A/B charges the compressor for the
  flagship arm's own spread.
- **Decision:** accept k=2 iff `mean(BPB) ≤ mean(base) + max(0.010, base spread)` **and**
  `mean(gpu_step)` is ≥8 % lower. Accept k=4 only if it beats k=2 on *both* numbers (their k-sweep
  says it will not: +0.0098 NLL for k=4 at equal everything else `[V]` Table 14).
- **Record:** `periter=[...]`, `fin=`, `bpb_fin=`, `depths=`, `gpu_step=` on every eval line — the
  medium per-iteration curve does not exist yet and is the single most decision-relevant number
  this wave can produce.

---

## 6. Verdict

**Is it worth a GPU wave? Yes — one, and it is a throughput experiment, not a quality bet.**
Port (a) at k=2 is the only candidate here with a real prize (~10 % of a compute-bound step,
~12 % at T=3), it is causally clean, it needs no new parameters beyond 2d writeback coefficients,
and its failure mode is the graceful one (the learned `1+δ` writeback damps the pooled increment
toward a soft skip — their own coefficients settle at 0.20/0.16). The paper's placement result
does **not** forbid it: they compressed the one visit that carried 100 % of their loop's value and
still kept ~50 % of it; we would compress a visit that carries ~3 % of ours. `[S]`

**Expected effect size at OUR scale (138M, T=3-4, 600 steps):** step time **−10 … −12 %** at k=2
(arithmetic on the measured 119 ms/iteration of a 573 ms step; 26.2 % at T=3), BPB **+0.003 …
+0.010**, most likely a **wash on the quality-vs-compute frontier** and a **win on wall-clock per
token**. Confidence in the *time* number: high (it is arithmetic on measured stage times).
Confidence in the *BPB* number: low — the 1B per-iteration curve that motivates it is unwritten
`[U]` and the 9.2M curve points the other way `[V]`. That asymmetry is exactly why the wave must
report `periter=`.

**Do not do:** (b) separate attention (pre-refuted by their own Table 5, +8.8 % params, zero FLOP
change, no positional mismatch to fix); a learned chunker inside the loop (non-causal in the batch
path, host-read in the streaming path, and more expensive than the visit it replaces at k≤4); the
mean-pool *framing* ("merging bytes") anywhere in the docs.

**Do, for free, now:** log the per-iteration gate means (`w_attn/w_mem/w_ffn`) — 0 params, ~6 LOC —
so the paper's §3.1 diagnostic is answered on our architecture before anyone spends a run on it.
If those means are flat across visits, port (c) is closed permanently and port (a) inherits the
argument.

**What would change the verdict:** a medium `periter` curve that is monotone like the 1B one
(then k=2 is close to free); or a `periter` curve like the 9.2M one (then compress harder, and
consider compressing *two* visits); or `gpu_step` flat under the flag (then the arm is launch-bound
and there is nothing to win — stop).

---

## Open questions

1. **Why is our 9.2M per-iteration curve non-monotone while the 1B one is monotone?** Nothing in
   the tree explains it. It decides whether the last visit is our cheapest or our worst.
2. **What is `residual_scale` actually learned to at 1B?** Never logged. One counter would tell us
   whether our single shared scalar is already doing the paper's per-visit job.
3. **Does a compressed visit still help *training* even when it hurts the readout?** Their Table 13
   cannot separate "worse readout" from "worse gradient path". Our `fin` vs `periter` split can.
4. **Is the 50 %-retention ratio a property of pooling or of the delay?** Their k=3/4 numbers
   conflate the two (they say so, p.23). Our port has no delay for the last visit (the receiver is
   the group's last byte), so it is the clean test of coarseness alone.
