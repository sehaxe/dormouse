# Engram re-enabled at a capacity budget, with a hard floor on the memory

**Date:** 2026-09-27 · **Status:** implementation landed, A/B queued
**Parent:** `research/2026-09-27-pkm-engram-deepseek.md` (the mechanism survey
this decision is built on) · **Rule:** `docs/AB-PROTOCOL.md`

## 1. The decision, in one line

The hashed n-gram memory arm was switched **off** on 2026-09-27 because at
8M rows/order it was 99% of the model and monopolized the loss. It comes back
**smaller** (500K rows/order, orders 2/3/4, 16x less memory) and
**structurally bounded** (a convex mixture with a hard floor at
`lam = min(w_mem, 0.5)`), because the pathology was the *configuration*, not
the mechanism: DeepSeek-V4.1-Flash ships the same thing (arXiv 2601.07372,
ACL camera-ready "Deep Sparse Embedding").

## 2. Capacity: the curve we chose from

The budget is a measurement, not a preference. arXiv 2601.16531 (125M
backbone, 128M Engram, iso-parameter, slots per order) measures the hashed
memory's saturation:

| slots/order | held-out loss |
|---|---|
| 300K | 4.4825 |
| **500K** | **4.4809** (optimum) |
| 800K | 4.4961 (**worse** than 500K, ~2 sigma) |

We ran 8M per order - **16x past the knee**, on the far side of a curve that
has already turned over. The shipped default is therefore 500_000 rows/order.
In-VRAM tables round **up** to a power of two (the slot index is masked, not
divided, on device): 500_000 -> 524_288, +4.9%, on the flat part of the curve.

The order list moved from 3/5/8 to **2/3/4** on the same evidence:

| order | key space (bytes) | n-grams averaged per row at 500K | verdict |
|---|---|---|---|
| 2 | 65,536 | <1 (the space is smaller than the table) | lossless bigram table |
| 3 | 1.7e7 | ~34 | the only arm that had per-key support at 8M |
| 4 | 4.3e9 | ~8,600 | largest order a 46 GB byte corpus can populate |
| 5 | 1.1e12 | ~2.2e6 | dead: a 5775-way average at 8M, worse here |
| 8 | 1.9e19 | - | dead |

2/3/4 is also DeepSeek's own shipped set over compressed tokens (V4.1-Flash
`n in {2,3,4}`; Engram-27B `[2,3]`), and 2601.07372 Sec. 6.2 measures 4-grams
as "slightly suboptimal" only in the sense of *diluting budget from the more
frequent 2/3-grams* - at 500K rows/order, which is exactly the trade we make.

Where the capacity lives, and the one seam:

- `DormouseConfig`: `engram_rows`, `engram_orders`, `engram_dim`,
  `engram_lam_max` - first-class, preset-settable, `--set`-able, validated.
  A preset with `use_engram = false` still carries the budget, so the arm can
  be switched back on without editing anything.
- **in-VRAM** path: the model owns the capacity. The data crate emits RAW FNV
  hashes (`hashes_raw`) and the model masks the slot index with its own
  `engram_slot_mask`. One copy of the row count, and every index is in range
  by construction. (`--engram-ram` squeezes the in-VRAM table to 1 row per
  order in `cfg::resolve`, because on that path the in-model table is never
  read - it was 201 MB of dead weights and 201 MB of dead bytes per
  checkpoint at the 500K budget.)
- **host-RAM** path: the row count comes from `--engram-slots` (the trainer
  builds `HostNgram` from the flag), the orders from `dormouse_data::ORDERS`.
  The config and the data crate cannot both own the order list until the
  trainer's plumbing can pass the config into `ByteStream`, so
  `cfg::model_and_data_agree_on_the_engram_orders` pins them together in the
  one test that sees both crates. **This is the seam to close when
  `dormouse-train/src/lib.rs` is free.**

## 3. The read: a convex mixture with a floor, not a row copy

Before, per loop iteration:

```text
y = attn*w_attn + (value_proj(e) * gate) * w_mem + ffn*w_ffn
```

`w_mem` is a sigmoid: unbounded up to 1.0, learned, and nothing in the
structure prevented it from settling there. A lookup table keyed on the exact
n-gram that is allowed to own the whole branch will explain the training
targets and starve the backbone - which is what was measured (rec -> 0.005,
out_acc -> 0, held-out frozen at exactly uniform 8.000 BPB).

Now:

```text
lam = min(w_mem, engram_lam_max)          # engram_lam_max = 0.5
y   = attn*w_attn + [lam*memory + (1-lam)*mem_dense(normed)] + ffn*w_ffn
```

`memory` is the module's existing gated read and its gate is **unchanged**:
`sigmoid(copysign(|s|.clamp_min(1e-6).sqrt(), s))` with
`s = <RMSNorm(W_K e), RMSNorm(h)>/sqrt(d)` - the shipped DeepSeek kernel, not
the textbook `sigmoid(s)`, which diverges for `|s| > ~1`. `mem_dense` is a
plain `d_model -> d_model` projection of the same hidden state the gate is
computed against.

The guarantee is **structural**: whatever the controller learns, the memory's
coefficient in that branch cannot exceed `engram_lam_max`, so the backbone's
share is never below `1 - engram_lam_max`. That is the whole point - a floor,
not a learned value. Copied from:

- **FwPKM eq. 12** (arXiv 2601.00671): `o_t = g_t*v_hat_t + (1-g_t)*v_t`,
  where `v_t` is a dense value path from the *same* hidden state. The dense
  path is what makes it a floor rather than a gate; the same paper gates the
  memory's *loss* too (eq. 13).
- **kNN-LM eq. 3** (arXiv 1911.00172): `p = lambda*p_knn + (1-lambda)*p_lm`
  with lambda a tuned **constant**, and the measured result is that kNN-LM
  memorizes the training set *while improving generalization*, where a
  memorizing LM interpolated at 0.1 buys 1.9 ppl -> 0.1. Same claim, and the
  reason the literature uses a constant.
- **XLM's PKM** `EmbeddingBag(per_sample_weights=True)`
  (`xlm/model/memory/memory.py:79`): the read is a bounded weighted average
  of rows, not one row.

`memory_floor_caps_the_controller` recovers the mixture coefficient from the
output (`a = (out - dense) / (mem - dense)`) with the controller saturated,
and asserts `a <= engram_lam_max`. A ratio, not a magic number: delete the
clamp and it fails.

**Cost, stated because it confounds the A/B:** `mem_dense` is
`d_model^2` = 590K params on `small` (a 7.9% increase over the 7.5M
backbone) that the control arm does not have. A treatment win would need a
param-matched control to attribute; a treatment loss is unaffected (the extra
params did not help). This is the one deliberate addition - FwPKM's dense
path is a projection, and borrowing an existing branch (the FFN) would have
double-counted it into the block sum.

## 4. The addressing path, honestly

The key projection (`loop_block.engram.key_projs.0.weight`, a
`3*engram_dim -> d_model` Linear over the looked-up rows, in the optimizer's
Muon+ group) **does receive gradient through the memory branch**, and
`mem_dense` does too - `engram_addressing_path_receives_gradient` asserts
both, so the addressing path cannot silently die again.

What it **cannot** do is learn the address. The slot index is a fixed FNV-1a
digest of the byte context: the key projection can only re-weight which of
three fixed tables to trust. Making the address learnable is a different
mechanism - product keys / learned sub-keys (1907.05242, 2412.09764, where
Meta's stated reason for paying sqrt(N)*d_q key params is that "the keys are
being continually trained and need to be re-indexed") - and that is a separate
decision, deliberately not taken here. The cheap route taken: the gradient
path exists, the floor bounds the damage, and the limitation is written down
next to the test that could detect a change.

## 5. The A/B

Protocol (`docs/AB-PROTOCOL.md`), unmodified: control = arm off, treatment =
arm on at 500K rows/order, **3 seeds per arm**, 2000 steps, batch 20 x seq
512, `--jepa-weight 0 --dspark-weight 0` (pure CE, so the arm is judged
alone), fp32, host-RAM tables with per-step CPU Nesterov+Sinkhorn, scored on
the fixed held-out window with `--eval-every 250 --eval-batches 10
--eval-depths`. The bar is the n-gram counter on the same window.

RESULTS: see `docs/AB-PROTOCOL.md` (the queue row is the verdict of record).
