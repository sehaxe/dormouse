# AttnRes / mHC cost memory, not FLOPs — measured 2026-11-10

Two arms landed today, both param-cheap and FLOP-cheap, both **unrunnable at the
batch the control uses**. This note exists because the first report of each said
"0.04% of the step" and that was true and misleading: the binding cost is VRAM.

## What fit

| batch | control (gated FFN) | AttnRes | mHC |
|---|---|---|---|
| b24 | ✅ 1.4892 @ 4880 steps | ❌ OOM — "Tried to allocate 858.00 MiB", 15.56 GiB card | not reached |
| b20 | ✅ 1.4990 @ 5856 steps | ❌ OOM | ❌ OOM |
| b16 | running | queued | queued |

## Where the memory goes

AttnRes (full N=L) builds a score tensor `[T, S, d]` per layer, `S = i+1` sources:

| S | buffer, bf16, b24 × 1024 |
|---:|---:|
| 2 | 78 MiB |
| 6 | 234 MiB |
| 12 | 468 MiB |
| 13 | **507 MiB** |

plus the retained per-layer history `[L, T, d]` (12 × 39 MiB = 468 MiB).

mHC (K=2) carries two full-width lanes per layer: 2 × 39 MiB × 12 layers =
**936 MiB**, plus a `[b,t,K,K]` fp32 mixing matrix (0.375 MiB/layer) and a
triple `[b,t,K,d]` transient for the convex read.

Total: **+2…3 GiB** on top of a control peak that already sits at ~14.9 GiB at
b24. Hence b24 → b16.

## The cost, stated as a number

b24 → b16 is **−33% bytes per step** for the same model. A BPB win from either
arm has to be read against that: at 120 MB the arms pay 1.5× the wall-clock, and
the acceptance bar (memory #52: 1.5–2 % quality per 10 % speed) demands
**+5…+7 % quality** for a 33 % throughput loss.

## The fix (not a workaround)

1. **Score one source at a time.** Replace the `[T, S, d]` score tensor with `S`
   separate `[T, d]` passes: the peak drops from 507 MiB to 39 MiB per layer and
   the arithmetic is identical (the softmax is over `S` scalars per token either
   way). This is the change that should return b24.
2. **Recompute the lane state instead of retaining it** for mHC, or store lanes
   in the bank dtype and rebuild on demand.
3. Only if 1–2 are not enough: activation checkpointing on the block, which we
   already measured elsewhere at −65% peak for +11% time.

Neither arm should be judged on a verdict measured at a batch the control never
uses without recording that batch penalty next to the BPB delta.
