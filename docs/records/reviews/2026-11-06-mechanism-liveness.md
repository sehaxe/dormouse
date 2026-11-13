# 2026-11-06 — mechanism liveness in the 105M flagship (`flagship2`, step 49 999)

**Scope.** Read-only liveness audit of every non-vanilla mechanism in
`port/configs/flagship_base.toml`, measured on the live checkpoint
`checkpoints/flagship2/flagship2.pt` (**step 49 999**; the run was at step
71 100 when this finished). I modified no file under `port/`, `crates/` or
`docs/` except this one; no commit, no push. Nothing was written to the live
run's directory.

**Tags.** `[M]` measured on this box, instrument named. `[D]` derived
arithmetic from measured numbers. `[L]` literature. `[U]` unmeasured — an
explicit absence, not a zero.

---

## 0. Instrument, and the trap that had to be cleared first

The eval instrument for every number below: the **head 8 192 bytes of
`data_local/mix4/web_eval.bin`**, scored as 8 windows of `b=2, t=512` by
`model(x, y, pure=True)` at the checkpoint's depth. Base arm: **BPB 1.6679**
(headline `(ΣCE_i + CE_final)/4`), **1.5782** `fin` (the mean readout alone).
The live run's own eval at step 64 000 scores 1.7423 BPB over its own 65 536-byte
window, so this window is comparable but not identical — every delta below is
*within* one fixed window, never across windows.

**The trap.** `dormouse eval --device cpu` on this checkpoint prints **6.666 BPB
(fin) / 9.110 headline** — a 4.2× error. On CPU, `KDAAttention.forward_state`
takes the COUNTED `_sdpa_path` (`kda.py:435-441`): causal SDPA with `g` as an
*output gate* and `beta` unused, i.e. **a different model**, because FLA's
`chunk_kda` is CUDA-only (`kda.py:16-18`). So no CPU measurement of this model
is valid as shipped, and any A/B that ran on CPU (including `tests/` gates that
are CPU-true) measured the fallback.

The CPU numbers here are produced with the exact per-token recurrence
substituted — `kda._recurrent_step` (the module's own decode step, the anchor
`tests/test_kda_gpu.py::_recurrent` pins to FLA at 1.5e-8) walked over the
window. **Validation**: same 8 192 bytes, same checkpoint, CPU-fp32 surrogate
1.1560 nats vs GPU-bf16-FLA 1.1561 nats; per-iteration CE identical to 3
decimals (`1.2939/1.1386/1.0976` both). Everything below is therefore fp32 CPU
at 2.4 GHz with a faithful KDA, not the shipped CPU path.

Two more loader facts, both verified rather than assumed: `lfb_bias` is a
non-persistent buffer that travels in the checkpoint's `moe` section and
`load_state_dict` never touches it — a raw `load_state_dict` scores a router
with a **zero** bias (delta ≤ 0.0002 BPB here, so no number below moves, but the
next reader should know); and `unpack_state_dict` is a no-op for this file
(`model_quant_rung = None`).

The live GPU job was never given more than ~0.1 GB of my own; the two short GPU
probes (`b=2, s=512`, no optimizer) dropped its throughput from 59.8 k to
27 k tok/s while they ran, so the rest was moved to CPU.

**The checkout moved under the audit.** Five `port/src` files carry a sibling's
uncommitted edits (`config.py`, `post/rl_train.py`, `train/{loop,optim,sft}.py`
— a new `TrainConfig.muon_lr_mult` and the optimizer-audit wiring). None of them
touches `model/`, and every number here comes from `model.forward` with hand-
built tensors, never from `train()`, so the numbers are unaffected — but a
reader reproducing this should re-run it on one tree, not across a live edit.

---

## 1. What the 102 678 943 parameters actually are

`[M]` module tree, `param_count` (`model.py:1193`), keyed by top-level group:

| group | params | share | touched by one byte |
|---|---:|---:|---|
| `loop.engram.tables` | 50 398 208 | 49.1 % | 3 rows × 32 = 96 weights |
| `loop.moe` | 48 867 584 | 47.6 % | 2 of 256 experts (`v` only: 349 696) + shared + `down` |
| `loop.attn` (KDA q/k/v/g/beta) | 1 327 104 | 1.29 % | all |
| `loop.mem_dense` | 1 049 600 | 1.02 % | all |
| `loop.csa` | 524 288 | 0.51 % | all |
| `embed` | 262 144 | 0.26 % | 1 row |
| `loop.out_proj` | 131 136 | 0.13 % | all |
| `lm_head` | 81 984 | 0.08 % | all |
| `loop.mhc` | 16 395 | 0.02 % | all |
| `loop.controller` | 6 144 | 0.01 % | all |
| `loop.attnres` | 6 144 | 0.01 % | all |
| `layerrope` (4 sites) | 4 112 | 0.00 % | all |
| `loop.iter_embed` | 3 072 | 0.00 % | all |
| `norm`, `csa_gate`, `residual_scale` | 1 028 | 0.00 % | all |

**Half the model is a hash table, half is a bank, 3.4 % is the network.** The
routed bank is 44 761 088 of the 48 867 584 `loop.moe` params, and each of its
256 experts owns only `v [64, 2730]` + two `[64]` scales — the `gate_u`/`up_u`
projections are shared across all 256 (`share_u = "plane"`, rank 64). So the
per-expert individual capacity is 174 848 weights, and the routed expert that a
byte actually touches is 350 k weights of the 102.7 M.

`[D]` Per-byte ledger from the repo's own `scripts/capacity_math.py`:
**14 503 040 MAC/byte** (it prices the MoE arms once although the loop runs them
3×; ×3-corrected ≈ 18.1 M). A 12-layer × 1024 transformer with `f = 2730` has
`12 × (4d² + 2df) = 117 M` params and `≈ 235 M` MAC/byte. **At equal parameter
count the vanilla shape has 13–16× the per-byte compute, and 12 distinct
attention+MLP pairs instead of one body run three times.**

---

## 2. The liveness table

`off-Δ BPB` = the same weights, same window, one thing changed in the forward
(in-place intervention; positive = the mechanism *helps*, negative = it hurts).
`flag arm` = the model rebuilt with the flag off, which on this architecture
usually changes the residual *topology* — those are reported separately in §3
and are not the verdict's evidence.

| mechanism | params | grad share `[M]` | off-Δ BPB | verdict |
|---|---:|---:|---:|---|
| **Engram memory read** (tables) | 50 398 208 | 5.8 % | **+2.306** | **LIVE (dominant)** |
| ├ order 4 (3-byte ctx, 41 % rows warm) | 16 777 536 | — | **+1.097** | LIVE |
| ├ order 3 (2-byte ctx, 13 % warm) | 16 777 536 | — | **+0.339** | LIVE |
| └ order 2 (1-byte ctx, 2.3 % warm) | 16 777 536 | — | **+0.112** | LIVE (weak) |
| **Attention arm** (KDA + CSA out of `y`) | 1 851 392 | 17.9 % (KDA) + 10.2 % (CSA) | **+0.633** | **LIVE** |
| **LayerRoPE depth** (per-iteration gains) | 4 112 | 2.8 % | **+0.444** ¹ | **LIVE** |
| **Loop iterations 2..3** | 0 (weight-tied) | — | **+0.289** (3 vs 1) | **LIVE** |
| **AttnRes** (the gate it actually is) | 6 144 | 3.4 % | **+0.238** ² | **LIVE as a gain** |
| **Whole FFN / MoE branch** | 48 867 584 | 9.8 % | **+0.101** | LIVE (thin) |
| **CSA fine** (top-32 of `t/8`) | 524 288 | 10.2 % | +0.048 | LIVE (thin) |
| **Routed expert bank** (256, top-2) | 44 761 088 | — | **+0.067** ⁴ | LIVE (thin: 43.6 % of the params) |
| **Routing selection** (top-2 vs all 256) | — | — | +0.008 ⁴ | redundant |
| **LFB routing controller** (±4, γ=0.001) | 256 (bias) | — | +0.0002 | **INERT** |
| **CSA coarse** (chunk 256) | 262 144 | — | **−0.0002** | **INERT** (dead weight) |
| **mHC `H_res` Sinkhorn mixing** | 4 104 | 0.002 % ³ | **+0.0001** | **INERT** |
| `residual_scale` | 1 | 0 (structurally dead) | — | **INERT by construction** |

Nothing crosses the −0.02 "actively harmful" line: **no mechanism is harmful at
this checkpoint.** The problem is not a broken mechanism; it is where the
capacity and the FLOPs sit.

Read the small numbers against the repo's own paired-eval resolution, **0.002–
0.005 BPB** (`docs/protocols/AB-PROTOCOL.md`, quoted in `AGENTS.md` §3.1):
+0.008 and ±0.0002 are **ties**, and §1.2's rule is "a tie deletes the
mechanism" — so CSA-coarse, the LFB bias controller and the `H_res` Sinkhorn
mixing are delete-candidates on this evidence, not keepers. +0.048 (CSA fine) is
10–20× the resolution and +0.067 (the bank) is 13×, so those two are real but
small; everything at ≥ +0.10 is far outside it.

¹ Measured by freezing `LayerRoPEPaperGains.for_iteration` to depth 0 (all three
iterations share iteration 0's gains). The `layerrope=false` flag arm is **not**
a valid ablation here (§3).
² Measured by setting every `attnres.{i}.gain` to 0, which makes `q_eff = 0`,
`scores = (0, 0)`, `w = (0.5, 0.5)`, i.e. the exact identity write `y`.
³ Grad share of the *whole* `loop.mhc` group is 92.3 %, but 3.476 of that 3.477
is one scalar, `alpha_post`; `alpha_res` is 7.8e-6 and `b_res` 5.4e-7. §5.
⁴ `_dispatch` returning zeros (shared expert and `down` kept) costs **+0.0668**
BPB; `moe_topk=0` — which is the **dense blend over all 256 experts**, not
"off" (`experts.py:969`, `:1549`) — costs **+0.0079**. So the bank is worth
0.067 and the *choice* of two experts over all 256 is worth 0.008: 254 of the
256 experts are redundant.

Gradient share is `‖g_group‖ / ‖g_all‖` on one real batch (`b=2, t=512`, loss
1.1320 nats, `‖g‖ = 3.768`); the shares do not partition (23.4 % `mem_dense` +
17.9 % KDA + … > 100 % because `‖Σg‖ ≠ Σ‖g‖`). Per-parameter RMS is the more
honest "is it learning" number: `mem_dense` 3.2e-2, `head` 1.3e-2, `controller`
1.0e-2, `kda` 5.0e-3, `attnres` 4.8e-3, `out_proj` 4.6e-3, `csa` 4.4e-3,
`layerrope` 2.2e-3, `moe` 4.2e-4, `engram` 5.2e-4, `embed` 1.1e-4.
The two 50 M-parameter blocks have the *smallest* per-weight gradients in the
model — 30–80× below `mem_dense`.

---

## 3. Flag rebuilds — reported, but they are topology changes

`[M]` Same checkpoint, every shape-compatible tensor loaded (dropped/missing
counted per arm), same window:

| arm | dropped keys | BPB | Δ |
|---|---|---:|---:|
| `use_csa=false` | 9 (`csa.*`, `csa_gate`) | 1.7163 | **+0.0484** |
| `use_attnres=false` | 6 (`attnres.*`) | 2.0276 | +0.3597 |
| `use_mhc=false` | 9 (`mhc.*`) | 4.4281 | +2.7602 |
| `use_engram=false` | 6 (`engram.*`, **`mem_dense.*`**) | 4.9398 | +3.2719 |
| `max_iter=2` | 8 (`iter_embed`, `csa_gate`, 3rd slot) | 1.8765 | +0.2086 |
| `layerrope=false` | 28 (`layerrope.*`), **missing `loop.norm.weight`** | 310.29 | invalid |
| `use_kda=false` | — | **crash** | — |
| `share_u=false`, `attn_latent_rank=0` | — | not ablatable | — |

Four of these cannot be read as "what is mechanism X worth":

* **`use_mhc=false`** falls to the AttnRes-only readout `h = depth_attend(h0,
  y0, …, y_i)` — a softmax **average** of the history instead of a residual
  accumulation. Per-iteration BPB `3.05 / 6.24 / 5.05`: the arm diverges, which
  says the *composition* is load-bearing, not that Sinkhorn mixing is. The
  in-place test (§2, +0.0001) is the measurement of the mixing itself.
* **`use_attnres=false`** switches the mHC arm to its unnormalised lane sum, a
  different readout scale. +0.36 conflates AttnRes with that.
* **`use_engram=false`** also deletes `mem_dense`, and replaces the branch with
  an exact zero (`loopblock.py:1219-1231`), so it removes the dense half of the
  memory branch too. The per-order in-place ablations in §2 are the clean
  version.
* **`layerrope=false`** replaces the block's norm with a plain `nn.RMSNorm`
  whose gain is a **fresh parameter** (`missing: loop.norm.weight`), while the
  trained gains live in `layerrope.sites.*.gamma`. 310 BPB is the fresh gain,
  not the mechanism. Measured in place instead: +0.444 BPB.
* **`use_kda=false` + CSA on crashes**: `UnboundLocalError: prev` —
  `prev = attn_state` is assigned inside `if blk.attn is not None:`
  (`loopblock.py:1115-1118`) but read unconditionally by the CSA block below
  (`loopblock.py:1140`). The documented "zero branch" arm is unreachable with
  CSA on.
* **`share_u=false` / `attn_latent_rank=0`** change parameter *shapes*
  (per-expert `u` instead of the shared one; `d → d` projections instead of
  `d → 256 → d`), so no checkpoint tensor can be transferred. `[U]` — a proper
  arm needs a from-scratch run, not an ablation. What can be said from the
  weights: the four KDA projections read a rank-256 latent, so the attention
  operates in 256 of 1 024 dims, and every expert's `gate`/`up` path is the
  *same* rank-64 projection regardless of which expert is routed.

---

## 4. Are KDA and CSA actually attending?

**KDA — yes, and the spectrum is learned, not initial.** `[M]` `exp(g)` (the
per-channel, per-byte retention of the delta state), 3 iterations × 512 bytes ×
16 heads:

| statistic | value | meaning |
|---|---|---|
| mean | 0.693 | dominated by the middle of the distribution |
| **median** | **0.990** | half the channels keep ≥ 100 bytes |
| p10 | 0.0016 | a tenth of the channels forget within one byte |
| p90 | 0.99998 | the top decile is effectively infinite memory |
| `beta` median | 0.9998 | the delta rule writes the full error (saturated rail) |

Bimodal by design: KDA is doing *per-channel multi-timescale* memory, not
copying. (At init `g = logsigmoid(0) = −0.693` for every channel, so 0.5; the
trained distribution is not that.) State RMS 1.4–2.9, output RMS 0.43 against
an input RMS 1.41 — a gated, damped write.

**CSA fine — yes, and it looks far back, not at the current byte.** `[M]` 64
chunks at `t=512` (`csa_chunk=8`), `csa_topk=32`:

| statistic | value |
|---|---|
| keys available / selected | 64 / 32 |
| entropy | 2.379 nats of `ln 32 = 3.466` (**68.6 %** of max) |
| mean max weight | 0.285 (not one-hot) |
| weight-weighted **age** | **133 B** (median 128, p90 240) |
| mass at age < 16 B | 9.1 % |
| mass at 16–128 B | 48.4 % |
| mass at > 128 B | 42.4 % |

A trained, broad, long-range retrieval over the past ~500 bytes. It is worth
+0.048 BPB on its own (its write is scaled by `sigmoid(csa_gate)`, 0.18–0.30
over the three iteration slots).

**CSA coarse — structurally degenerate at this sequence length.** `[M]` with
`csa_coarse_chunk=256`, `t=512` gives **2** coarse entries (`pos_coarse =
arange(t) // 256`), so the first 256 positions have *no* legal key (50 % of rows
are zero-weight by the `allowed` mask) and the second 256 attend exactly **one**
key (entropy 0.0, max weight 1.0). At the live `t=1024` there are 4 entries, at
most 3 usable. Turning
the level off changes BPB by **−0.0002**. The "coarse" level is a 256-byte box
filter with a learned projection, and it costs a 262 k-parameter branch plus a
`t/256`-key attention at every iteration for nothing measurable.
`csa_topk=32` also exceeds the whole coarse key count, so *selection* never
happens there.

---

## 5. mHC / Sinkhorn: the mixing is a no-op, the lanes are not

`[M]` `_post_res_tilde` + `pre_mix` on a real batch (2 streams, 20 Sinkhorn
iterations, `mhc_sinkhorn=log_softmax`):

| quantity | value |
|---|---|
| `H_pre` per lane (σ) | mean **0.990** overall (per-lane means 0.998 / 0.982 at iteration 0, 0.999 / 0.999 at iteration 2), p01 0.939 — saturated at 1 |
| `H_post` per lane (2σ) | mean 0.996, median 0.693, p10 0.24, p90 1.77 — real per-lane gate |
| `H_res` entries | mean 0.500, **median 0.025**, p10 0.0, p90 1.0 |
| mean `H_res` matrix | `[[0.634, 0.377], [0.366, 0.623]]` |
| lane cosine after the first write | 1.000 → **0.176** → 0.20 |
| lane RMS after the first write | 1.20 vs 0.66 |

Per token the doubly-stochastic matrix is **at a vertex of the Birkhoff
polytope** — a keep-or-swap, not a mixture — and because `H_pre ≈ 1` on both
lanes the body reads `L0 + L1`, i.e. the sum, at 2× the residual magnitude.
`H_res` is worth **+0.0001 BPB** when replaced by the identity, and its own
parameter has no gradient to speak of: `alpha_res` 7.8e-6, `b_res` 5.4e-7
against `alpha_post` **3.476** (92 % of the model's total gradient norm).

So the mHC block is, in practice, **two residual lanes with a per-lane scalar
write and a per-token swap**. That is a real (if degenerate) mechanism — the
lanes diverge to RMS 1.20/0.66 and that is worth something — but the
manifold-constrained *mixing* the paper is about does nothing at this scale, and
`alpha_post` (value −0.0007, while `phi_post` has grown to |57.7|) is an
ill-conditioned reparameterisation carrying essentially the entire gradient of
the block.

---

## 6. Engram: which half of the table is warm, and what it reads

`[M]` Row warmth from the checkpoint itself (a row "moved" if its norm deviates
from the `N(0, 0.02)` init by > 50 % — the threshold that filters AdamW's
decoupled-decay shrink, which alone would take every row to 0.78× init):

| order | rows | rows ever moved | `[M]` lookups landing on a moved row | distinct rows per 8 KB | ΔBPB when that order's table is zeroed |
|---|---:|---:|---:|---:|---:|
| 2 | 524 288 | **11 989 (2.3 %)** | 100.0 % | 602 | +0.112 |
| 3 | 524 288 | **70 501 (13.4 %)** | 99.6 % | 2 097 | +0.339 |
| 4 | 524 288 | **213 343 (40.7 %)** | 99.3 % | 3 823 | **+1.097** |

`[M]` Distinct byte n-grams per 100 MB of the live mixture's sources
(`scripts`-free numpy census, offset 300 MB into each file):

| source | order-2 | order-3 | order-4 |
|---|---:|---:|---:|
| fwedu | 14 287 | 137 687 | 649 509 |
| cosmopedia | 12 344 | 116 957 | 493 182 |
| code | 18 502 | 396 422 | 1 897 301 |
| ru | 11 601 | 105 794 | 406 129 |
| science | 14 882 | 181 065 | 859 692 |

So the three tables are three different objects:

* **order 2** — at most 65 536 distinct contexts exist and only ~12–18 k occur
  per source. A 524 288-row × 32-dim table (16.78 M params) is **97 % dead by
  construction**; a 65 536-row table (2.1 M) would hold every context that can
  ever occur. It is still worth +0.11 BPB, so the *feature* is fine — the
  allocation is 8× too wide.
* **order 3** — 87 % of rows cold after 1.6 GB of training. Not yet the
  bottleneck. Worth +0.34.
* **order 4** — **saturated in a single 100 MB sample** (649 k–1.9 M distinct
  contexts against 524 288 rows), i.e. every row is a hash bucket averaged over
  ~1.2–3.6+ contexts. It is the most valuable single order (+1.10), which says
  the memory's power is in the 3-byte context and that **collisions are the
  next lever**, not more rows per se.

The read is added, not hidden: the DeepSeek gate is `mean 0.507` (order means
0.407 / 0.499 / 0.614, std 0.17) and the read's RMS is **0.876** against the
attention branch's 0.433 — the memory is the largest single term in `y`. The
convex floor is what actually sets its strength: the controller wants
`w_mem ≈ 1.0` and the hard cap `engram_lam_max = 0.5`
(`engram.py:620-634`) clamps it on **every token and every iteration**, so the
branch is exactly `0.5·mem + 0.5·dense`. Forcing lam to 0 costs **+2.525**, lam
to 1 costs **+1.214**: the 50/50 split is where the trained network lives, not
a free choice.

---

## 7. Is the loop doing anything? Effective depth

`[M]` Depth curve (`evaluate(depths=True)` equivalent, same window):

| depth | headline BPB | `fin` BPB | per-iteration BPB |
|---|---:|---:|---|
| T = 1 | 1.9561 | 2.0455 | 1.8668 |
| T = 2 | 1.7379 | 1.7047 | 1.8668, 1.6422 |
| T = 3 | **1.6679** | **1.5782** | 1.8668, 1.6422, 1.5835 |

The loop is **live and diminishing**: iteration 2 buys 0.218 BPB on the
headline (0.225 on its own readout), iteration 3 buys 0.070 (0.059). Per-iteration
state: `‖h‖` (RMS) 0.747 → 0.889 → 0.931, and `cos(h_i, h_{i−1})` = 0.47 then
0.80 — the third pass is a refinement of the second, not new work.

**Effective depth in transformer terms: one body.** `prelude_layers = 0`,
`coda_layers = 0`, `loop_blocks = 1` → `embed → [norm · KDA+CSA · Engram read ·
MoE] × 3 → RMSNorm → head`. The three iterations are the *same* attention, the
*same* memory read and the *same* MoE bank. There is no unique second layer
anywhere. Against a 12-layer GPT-2-small-shaped model at the same parameter
count, the model has **1/12 of the distinct layers and 1/13–1/16 of the
per-byte MAC** (§1).

Effective depth is also reduced inside the loop: the controller closes the FFN
on the first iteration. `[M]` `w = σ(controller · [ctx; h0])` per iteration:

| iteration | `w_attn` | `w_mem` (→ `lam`) | `w_ffn` |
|---|---:|---:|---:|
| 0 | 0.9991 | 0.9999 (0.5) | **0.0035** |
| 1 | 0.9662 | 1.0000 (0.5) | 0.978 |
| 2 | 0.9588 | 1.0000 (0.5) | 0.991 |

The first pass therefore computes the whole routed bank and multiplies it by
0.0035. Three sigmoid gates sit on a rail (`w_attn → 1`, `w_mem → 1` clipped to
0.5, KDA's `beta → 1`), which is why the *controller* is where the block's
conditioning lives.

---

## 8. The single most likely reason we are 0.4 BPB behind

**Not a silent no-op — a shape problem with a misallocated bank.** Every
mechanism that carries signal is demonstrably alive (§2), and nothing is
actively harmful. But the model is **one unique layer deep**, unrolled three
times with tied weights, and about **45 % of the per-byte multiply-accumulates
in the loop body** (module-tree arithmetic, §1, including the shared `down`;
29 % by the repo ledger's own split) **goes to the FFN / MoE branch, which is
worth 0.101 BPB**. Of that, the routed 256-expert bank —
**44.8 M of the 102.7 M parameters (43.6 %, and 254 of its 256 experts are
redundant at top-2)** — is worth **0.067 BPB**, while **4 112** LayerRoPE gains
are worth 0.444. In the same ledger the memory does 2.31 BPB of work with a
gather and 50.4 M parameters that are 81 % cold. So the parameter budget is
spread over two half-dead tables and a redundant bank, while the depth that
would convert parameters into per-byte compute is absent.

A vanilla transformer at 102.7 M parameters spends its parameters on 12
distinct attention+MLP pairs and puts 13–16× more MAC behind every byte. At
1.6 GB of training data that difference is expected to show up as roughly the
gap we see; no bug is required to explain 0.4 BPB, and no single flag flip
recovers it. `[D]` The cheapest shapes the evidence points at, in order:

1. **Move the bank's parameters into depth.** `n_experts 256 → 8` frees
   ~43.7 M params ≈ 4.5 dense `d=1024, f=2730` layers' worth; the routed bank
   it removes costs 0.067 BPB, and a *second unique* layer cannot be worth less
   than the *repeated* second iteration, which is worth 0.225 (§7).
2. **`loop_blocks = 2..3`** (distinct bodies inside the loop) instead of one
   body × 3 — but it duplicates the engram and the bank per body at today's
   config (`body_only` keeps both), so it is only cheap *after* (1).
3. **`prelude_layers`/`coda_layers` ≥ 2** as the cheapest unique-depth probe:
   ~9.8 M params per layer, and it tests the hypothesis directly.
4. **Order-4 Engram rows are a hash bucket average already** (§6): more rows
   and/or a 2-way associative table is the memory's next lever; the order-2
   table should shrink to 65 536 rows and give 14.7 M params back for free.
5. **Delete the CSA coarse level** at these context lengths (INERT, −0.0002),
   and stop paying for the `mHC` Sinkhorn mixing at forward time if the
   two-lane behaviour is all that is wanted — but *fix the `alpha_post`
   conditioning first*, since it holds 92 % of the gradient norm.

**What to do about the 0.4 BPB comparison itself:** any arm scored on CPU in
this repo is scoring the SDPA fallback (§0), which is 4.2× off. The vanilla
control must run on the GPU with FLA, or it is not a control.

---

## 9. Defects found (all pre-existing, none introduced here)

1. **`use_kda=false` with CSA on crashes** — `UnboundLocalError: prev`,
   `loopblock.py:1115-1118` assigns it inside `if blk.attn is not None:`,
   `loopblock.py:1140` reads it unconditionally. The documented "zero branch"
   arm (`ModelConfig.use_kda`) is unreachable in the flagship shape.
2. **`dormouse eval --device cpu` prints a BPB for a different model** — 6.666
   vs 1.578 `fin` on the same checkpoint (§0). It is COUNTED as `kda_sdpa`
   (`kda.py:439`) but the printed number carries no warning, and every
   CPU-side A/B in the repo inherits the error.
3. **`layerrope=false` on a trained checkpoint is not an ablation** — the trained
   norm gain lives in the LayerRoPE sites, and the flag swaps in a fresh
   `nn.RMSNorm` gain (310 BPB, §3). Worth a refusal or a documented
   "copy the gain" escape, since a reader will try it.
4. **Order-2 Engram table is 8× wider than its key space** (§6): 16.78 M params
   address ≤ 65 536 possible contexts; 2.3 % of rows have ever been touched.

---

## 10. What I did not measure `[U]`

* **`share_u=false` and `attn_latent_rank=0`** — shape-changing, needs fresh
  runs (§3). Their *capacity* consequence is arithmetic, not measured.
* **The 200-step re-adaptation arms.** Started, **not completed**, and the
  reason is the box, not the design: four arms (`base`, `use_csa=false`,
  `use_engram=false`, `max_iter=1`) at `b=1 × s=512`, lr 2e-4, 200 steps each
  were still inside the optimizer after ~26 min wall (~20 CPU-min each, no
  output) and were killed rather than left to compete with the live run and two
  sibling processes (`AGENTS.md` §1.5 — one heavy thing at a time; the box was
  at load 15 with a sibling's `scripts/control_transformer.py` running). A
  60-step fallback pair was killed for the same reason. The design was weak
  anyway: every flag arm starts from a topology the checkpoint never trained
  (`use_engram=false` starts at 4.94 BPB), so a short recovery measures
  plasticity, not the mechanism. **The in-place §2 deltas are the evidence** —
  they answer "is it live" at fixed weights, without an optimizer in the loop.
  A real adaptation arm needs a from-scratch twin run per mechanism: 3 seeds ×
  2k steps per `docs/protocols/AB-PROTOCOL.md`, not a 200-step graft.
* **A trained-shape vanilla control, and the 0.53 GB accounting** — the sibling
  agents own those. My capacity arithmetic in §1/§8 uses the repo's own ledger
  and GPT-2-small's published shape, not a run.
* **Whether the eval bytes were in the memory's training data.** 99–100 % of
  the eval window's lookups hit moved rows, which is consistent with both
  "English n-grams are dense in any corpus" and "the eval file overlaps the
  training mixture". The +2.31 BPB the memory is worth *could* be partly
  memorisation of the eval window; that is a data-path question, not a
  mechanism one, and it would move the *absolute* number, not the ranking.

---

## Appendix — raw artefacts

`/tmp/opencode/live/` (ephemeral, this box):
`lab.py` (loader + fixed eval window + KDA substitute), `probe.py`,
`probe2.py` (gates, branch split, mHC, CSA age), `probe.json`, `probe2.json`,
`dyn.py`/`dyn2.py` (in-place interventions) with `dyn.json`/`dyn2.json`,
`evalarms.py` + `evalarms.json` (flag rebuilds), `grad.py` + `grad.json`,
`ngrams.py` + `ngrams.json`, `ablate.py` (the re-adaptation arms — its
`A_*.json`/`S_*.json` outputs are empty because the arms were killed before
completion, §10).
Every number in this document is reproducible from those files with
`cd port && .venv/bin/python <file>`.
