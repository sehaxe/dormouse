# The final architecture — design, and the one experiment that decides it

**Date:** 2026-10-23 · **Status:** design frozen; one A/B outstanding (the ByteFlow arm).
**Evidence:** `docs/reviews/2026-10-23-architecture-verdict.md` (every number below traces to a log there).

---

## 1. The shape

```
bytes → [ OPTIONAL: learned patcher ] → prelude? → LOOP BLOCK ×T → coda? → norm → byte head
```

The loop block is one parameter set applied T times (weight-shared), and it contains,
in this order:

```
x_l  = x + e_i                     iteration embedding (per-slot, learned)
ctx  = H_pre(x_l)                  mHC: K residual lanes mixed down to d
body = KDA(ctx) + Engram(ctx) + MoE(ctx)
a    = AttnRes(source = {0, body}) AttnRes: softmax mixture over write UPDATES
x'   = H_res(x_l) + H_postᵀ(a)     mHC: doubly-stochastic write-back across lanes
out  = Σ x' / K                    mHC: readout
```

Three mechanisms act on the residual stream and they were designed to compose:
`H_pre` mixes lanes, `AttnRes` picks which update to write, `H_post` deposits it,
`H_res` mixes again. Measured together: **−0.142 BPB** (2.508 vs 2.650, 3 seeds,
spread 0.009) — super-additive, because all three answer the same weakness: the
loop's later iterations were doing nothing.

### The expert bank (TSCT)

Per routed expert, a low-rank GLU with a **shared input basis**:

```
gate = (x·U_shared[g]) · s_g · V_g        up = (x·U_shared[u]) · s_u · V_u
ffn  = down(gate ⊙ SiLU(up))              (ONE shared down projection)
```

- `U_shared` is one `[d, rank]` pair for the whole bank instead of E pairs:
  **−23.5% of the model's parameters**, measured to cost **−0.013 BPB** (i.e. nothing).
- rank 64 stays. rank 32 is measured worse at fixed width (+0.027), and every
  compensation was tried and failed: width 2048 (+0.064), width 1152 + f4096
  (+0.104), rank 16 (+0.136), rank 8 (kernel crash), dict rank 2 (NaN).
- E=256, top-k 2, **dropless** (a capacity cap dropped 73% of assignments on b1
  and 18% here; dropless costs +0.007 at E=256, so the cap is kept only where it
  is harmless — see §5).

Measured against a dense FFN: **same BPB at equal parameters, 35× fewer MACs per
token**. That is the bank's whole claim, and it is a compute claim, not a quality one.

---

## 2. Component verdicts — what is in, what is out

### In (every one measured, 3 seeds)

| component | effect | why it earns its place |
|---|---|---|
| **mHC** (K=2 lanes) | −0.078 (600 steps), −0.028 (2000) | the largest structural win; costs +40% VRAM |
| **AttnRes** | −0.050 | decomposes the residual write; composes with mHC |
| **share_u** (shared input basis) | −0.013, **−23.5% params** | free parameters |
| **KDA** attention | O(1) state, 393 KB at any context | no KV cache; the 1M path |
| **Engram** (in VRAM) | tie on quality, 50M params of memory | free at medium; **not** on the host path (§4) |
| **TSCT + Muon/ColRow + Triton NS + opt_graph** | optimiser 44-53% → 13.7% of the step | the speed that pays for everything |
| **adaptive top-k** | tie on quality | a compute lever on the ragged arms |

### Out (measured, with the number)

| component | why out |
|---|---|
| rank < 64 | +0.027 … +0.136 BPB, or a crash |
| more experts (E=512) | +0.033 — capacity is not in the count |
| prelude/coda (1+1, 2+2) | tie with the recipe at +1.6M…+3.1M params |
| MTP / JEPA / energy / halt as pretraining losses | all ties at our scale |
| ternary in the *training* recipe | +0.005 net with mHC, but it only pays at export (§6) |
| dense_qat on attention | +0.198 (the broken arm that taught us attention stays ≥8-bit) |
| LOOM / SIGReg / NorMuon / SF-Muon / ns_steps 3 | losses or ties |
| ByteFlow in *this* shape | the one open question — §3 |

---

## 3. The open question: the patcher

ByteFlow (a learned 4-byte→1-patch channel) measured **−0.485 BPB** on medium —
11× bigger than any other mechanism in the project. It also:

- **refuses `use_mhc`** (the channel runs the loop over K patch latents; mHC's lane
  replication is written for the byte geometry) and **refuses `use_engram`**
  (the channel consumes the byte position the n-gram hash needs);
- ran in **fp32** in that A/B, at **1.64× the step time**.

So the decision is between two finished designs:

| | **A: byte-level (built, running)** | **B: patched** |
|---|---|---|
| recipe | mHC + AttnRes + share_u + Engram | ByteFlow + AttnRes + share_u |
| measured/expected BPB | **2.508** (measured) | 2.65 − 0.485 − 0.050 − 0.013 ≈ **2.10** (parts measured, stack unmeasured) |
| params | 105.0M | 113.5M |
| speed | **30.8k tok/s** | unknown after the bf16 port + fp32 Triton tiles |
| context | 1024 bytes | 512 patches = 2048 bytes |
| engram | yes | no |
| status | **training now** (234k steps, 5.77 GB) | needs one A/B wave |

**The gate (3 seeds, 600 steps, mix4/web, window 40,960 B):**

```
arm 1  baseline      = the running recipe             (expect 2.508)
arm 2  patched       = ByteFlow + AttnRes + share_u   (expect ≈2.10)
arm 3  patched−attnres = ByteFlow + share_u           (isolates attnres in the patched geometry)
arm 4  patched bf16  = arm 2 with mm_dtype=bf16       (does the fp32 tax still exist?)
```

Promotion rule: arm 2 (or 4) must beat arm 1 by more than 2× the control spread
**and** run at ≥ 0.6× arm 1's tokens/s. A win on BPB that costs half the throughput
is a wash on tokens-per-hour and is rejected by the owner's rule.

If B wins, the flagship is re-run on the patched recipe and Engram is dropped;
if B loses, A stands and ByteFlow stays a flag with a dated verdict.

---

## 4. Two traps this design is built to avoid

**1. Engram on the host.** `engram_ram=true` costs **0.83 BPB** at our budget
(3.476 vs 2.649, isolated across five runs). The host update is burn's verbatim
`b = 0.95·m + 0.05·g`, which converges to `g` only on rows touched repeatedly;
at 600 steps the table is almost entirely cold and the first touch moves at
**0.0975×** the gradient. Fixed with per-row bias correction (`b/(1−0.95^t)`),
which is now in the tree with nine gates — but the recipe still keeps the table
in VRAM at medium (201 MB), and only `b1` (402M rows) is allowed to use the host
path, with the correction on.

**2. A shared basis that is not shared.** `share_u` must quantise the *dictionary*
once per step (`2f55c1c`), not per expert: the naive path issues 512 quantiser
calls where 3 suffice. The counter `share_u_qat_forward` proves the fast path ran.

---

## 5. The config as it ships (byte-level, running today)

```toml
[model]
d_model = 1024                 # 1536+ costs more in attention (∝d²) than it buys
n_heads = 16
max_iter = 3                   # iter 4 is +0.013 BPB for +25% step time at 600 steps
max_seq_len = 1024
use_kda = true
use_engram = true              # VRAM, not host
engram_rows = 524288
n_experts = 256
moe_topk = 2
moe_capacity = 2.0             # a +0.007 cap at E=256; dropless is for b1
rank = 64
share_u = true
use_mhc = true
use_attnres = true
use_msa = false                # tie at 2k; costs 3.5× wall-clock on this form
use_byteflow = false

[train]
batch = 24
seq_len = 1024
lr = 1e-3
warmup_tokens = 50_000_000
decay_frac = 0.1
moe_kernel = "triton"
ns_kind = "triton"
opt = "mix"
muon_colrow_3d = "colrow"
opt_graph = true
```

---

## 6. Where the compressed form enters (and does not)

Ternary is **not** a training-recipe component: it frees no VRAM (fp32 masters stay)
and costs +0.005 BPB net (with mHC) to +0.25 (without). It pays exactly once, at
**export**: the bank goes 626M → 78M packed on b1, and the whole model 2.1 GB → 0.3 GB.
So the shipped artifact is produced by a **short QAT-finish on the trained masters**
(the ladder already measures the rungs), not by training in the rung.

QAT-finish on a converged model was measured to **destroy** it (BPB 38.58 at step
500 when converting `mix1`) — the recipe therefore trains in bf16 and the export rung
is applied to the *bank only*, where `factor_qat=ternary` measured +0.005 with mHC.
Dense layers stay bf16: quantising the router flips the top-k and costs +0.25.

---

## 7. What the final model is expected to be

| | value | basis |
|---|---|---|
| stored parameters | 105.0M | `dormouse info` |
| dense-equivalent function shapes | ~1.4B | bank ×18.3 |
| active per token | ~20M | top-2 of 256 experts |
| held-out BPB on mix4 | **1.65-1.80** | mix1 1.883 − 0.142 (recipe) − data domein + 38% more tokens |
| HellaSwag | 33-36% | 26-31% at 1.47 GB → ~5.77 GB |
| ARC-Easy | 32-36% | 29.5% at 1.47 GB |
| GPT-2 small on the same harness | **35.0 / 38.0** | measured today with `scripts/bench_hf.py` |
| shipped size (ternary bank) | ~0.3 GB | export rung |
| context | 1024 bytes trained; 32k via segments; 1M needs the selector | measured separately |

**Honest statement of the claim:** at equal parameters and equal data we beat a
dense transformer by **0.145 BPB**; against GPT-2 small we are behind on both BPB
and benchmarks, because it saw ~7× our data (40 GB vs 5.77) and 12 distinct layers.
The design's advantage is **compute per unit quality** (35× on the bank) and
**memory at inference** (constant state, no KV cache) — not raw quality at any price.
