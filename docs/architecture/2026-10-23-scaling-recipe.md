# The scaling recipe — every direction, with the arithmetic that sets it

**Date:** 2026-10-23 · **Status:** recipe; the sub-1B points are measured, everything above is fitted and marked.
**Fact base:** `docs/reviews/2026-10-23-architecture-verdict.md`, `docs/architecture/2026-10-23-final-architecture.md`.

---

## 1. What scales, and in what order

The recipe has four knobs. Our measurements rank them by how much quality each buys per unit compute:

| rank | knob | evidence | rule |
|---|---|---|---|
| **1** | **data (bytes)** | BPB slope ≈ 0.1 per 4× data at our scale; the mix4 jump from a math/code corpus to real prose moved the plateau ×5.5 | **data first**: never grow the model past the point where tokens/param < 20 |
| **2** | **width `d_model`** | capacity lives in width: rank↓ hurt, E↑ hurt, d↑ helped when the parameter budget was held | grow `d` while keeping rank/`d` and experts fixed |
| **3** | **expert count `E`** | E=512 at rank 32 was +0.033 BPB at 76% more parameters than E=256 | grow `E` only AFTER `d` and only when tokens/param ≥ 20 |
| **4** | **loop iterations** | CE per iteration: 2.15 / 1.86 / 1.81 / 1.80 → iteration 3 is worth ~0.01 | keep 3 until the data budget is ≥ 20 tok/param; never 5+ without a halt policy |

**Never scale**: expert `rank` below 64 (measured +0.027…+0.136), the byte vocabulary (256 is the point), or attention kind blindly (see §3).

---

## 2. The measured grid (all via `dormouse info`, so these are real parameter counts)

| `d` | rank | `E` | iters | params | expert MAC/token | dense MAC/token |
|---|---|---|---|---|---|---|
| 1024 | 64 | 256 | 3 | **108.5M** | 1.0M | 16.4M |
| 1536 | 64 | 256 | 3 | 121.2M | 1.4M | 37.0M |
| 2048 | 64 | 256 | 3 | 138.2M | 1.9M | 65.7M |
| 2048 | 128 | 256 | 4 | 183.6M | 3.8M | 82.5M |
| 2560 | 128 | 512 | 4 | 295.1M | 4.8M | 128.9M |
| 3072 | 128 | 512 | 4 | 320.8M | 5.8M | 185.6M |
| 3072 | 192 | 1024 | 4 | 681.4M | 8.6M | 185.6M |
| 4096 | 192 | 1024 | 4 | 746.2M | 11.5M | 329.9M |
| 4096 | 256 | 2048 | 4 | 1.65B | 15.4M | 329.9M |

**The one number that governs width**: `dense MAC/token ≈ 4·iters·d²` grows **quadratically** in `d`, while the expert bank grows **linearly** (`E·rank·d`). That is why width beyond ~1536 needs either fewer iterations or an accept-the-cost decision:

```
d=1024: dense 16.4M   vs expert 1.0M   (16x)
d=2048: dense 65.7M   vs expert 1.9M   (34x)   ← width is now the cost, not the bank
d=4096: dense 329.9M  vs expert 11.5M  (29x)
```

At 1024 the bank is a rounding error; at 3072 the *dense* side (KDA projections + `out_proj` + `down`) dominates the step. **The scaling consequence: width is the expensive direction, and experts+rank are the cheap one.** So a large model should be **wide-ish and expert-rich**, not enormously deep in `d`.

---

## 3. Attention, by context length

| context | mechanism | why |
|---|---|---|
| ≤ 2k | **exact causal attention** | costs `4·t·d = 4.2M MAC/token` at t=1024, the same order as the expert bank; gives exact recall |
| 2k-8k | **MSA** (block-sparse + learned selector) | the selector's overhead amortises; a tie on quality at 2k, a win on cost above it |
| ≥ 8k | **KDA + sliding window 2048** | exact attention at 8192 costs 34M MAC/token; KDA is constant-state and the window restores local exactness |
| decode | KDA state only | O(1) per byte, 0.52 MB/layer at d=1024, independent of context |

The A/B (`scripts/ab_attn.py`) settles the ≤2k row; the window arm is not built yet.

---

## 4. Data: the constraint that binds

Chinchilla-optimal parameters for a given corpus (20 tokens/param, 1 byte ≈ 1 token for a byte model):

| corpus | optimal params |
|---|---|
| **mix4 5.77 GB** (today) | **288M** |
| mix5 20 GB | 1.0B |
| 60 GB | 3.0B |
| 160 GB (all disk) | 8.0B |

Our flagship is **105M on 5.77 GB = 55 tok/param**, i.e. 2.7× past Chinchilla — deliberately, because inference cost matters more than training cost here (the overtraining rule: 20-200 tok/param is legal, 4× data repetition is not).

**So the honest trajectory**: to justify 300M we need 20 GB (a week of GPU); to justify 1B we need 60 GB (a month); 8B needs the whole disk and 400+ days on this card.

---

## 5. The recipe as a function of size

| size | `d` | rank | `E` | iters | attn | data needed | wall clock here |
|---|---|---|---|---|---|---|---|
| **105M (now)** | 1024 | 64 | 256 | 3 | KDA (full ≤2k pending A/B) | 5.8 GB | 2 days |
| **300M** | 2560 | 128 | 512 | 4 | KDA+window / MSA | 20 GB | 8 days |
| **750M** | 4096 | 192 | 1024 | 4 | KDA+window | 45 GB | 25 days |
| **1.6B** | 4096 | 256 | 2048 | 4 | KDA+window (MSA > 8k) | 90 GB | 70 days |
| frontier | — | — | — | — | — | 10+ TB | decades |

**The last row is why the answer to "scale to the frontier" is money, not architecture.** Our compute is 2.4×10¹⁷ FLOPs per run; a GPT-5-class run is 10²⁶ — a gap of 10⁹ that no recipe closes.

---

## 6. What must be re-tuned at each point (and what stays)

**Stays fixed** (measured scale-invariant in our range):
- rank 64 minimum and `share_u=true` (the shared basis is a parameterisation, not a hyperparameter);
- mHC + AttnRes (residual-stream logic, costs ~5% time and ~40% VRAM);
- byte vocabulary 256, dropless MoE at `E ≥ 512`;
- Muon + ColRow, Triton NS, `opt_graph` (optimiser is 13.7% of the step, not 50%).

**Must be re-tuned** (no cheap transfer law in this repo):
- **learning rate**: no muP. The recipe uses 1e-3 at 105M; a 300M run needs its own sweep (our only measured scaling: LR 3e-4 on medium, 1e-3 on the flagship, both by search, not by law);
- **warmup**: currently 50M tokens; keep it proportional to ~10% of the budget;
- **decay fraction**: 0.1 of the budget (WSD);
- **batch**: as large as VRAM allows — we are launch-bound below b16, and b24 s1024 is the measured sweet spot at 31k tok/s.

---

## 7. What this recipe does NOT claim

1. **That the efficiency holds at scale.** Every number here was measured at 105-142M. Low-rank experts are the most likely thing to degrade (the literature is mixed), and the width/depth trade could invert above ~1B. The first thing to check at 300M is whether rank-64 experts still match a dense FFN.
2. **That it beats the frontier.** It does not, and cannot: the FLOP gap is 10⁹.
3. **That it beats SmolLM2-135M.** It has 2.4 TB of data. Ours has 5.8 GB. At equal data we are ahead; at their data we have never been.
