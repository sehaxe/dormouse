# Overnight run — long context where CSA/HCA are load-bearing (2026-10-27)

**Result: BPB 1.7634 (headline) / 1.6882 (fin) on a 512 KiB held-out window**,
16 000 steps, 523 MB of text, 7.6 h on one card.

## The run

```
medium · 102 678 928 params · trained from scratch
seq_len = max_seq_len = 8192 B · batch 1 · grad_accum 4  → 32 768 B per optimizer step
CSA fine (chunk 8, kv 64, topk 32) + HCA coarse (chunk 256)
mHC (K=2) + AttnRes (composed) · share_u · attn_latent_rank 256
dropless MoE (moe_capacity=0) · Triton · Engram in VRAM · doc_masking
lr 1e-3 · warmup 16.4 MB tokens · decay_frac 0.1 · 16 000 steps
data mix4/web.bin (2.35 GB) · eval mix4/web_eval.bin
```

Step cost ≈1.73 s → 7.6 h for the whole run.

## The curve (32 KiB window, as logged)

| step | 500 | 1 000 | 2 000 | 6 000 | 9 000 | 12 500 | 15 000 | 16 000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BPB | 2.371 | 2.236 | 2.204 | 2.060 | 1.886 | 1.791 | 1.726 | **1.693** |

## The honest number, on a window 16× larger

`dormouse eval` pins the window to the checkpoint snapshot, so the 32 KiB logged
figure is noisy. `scripts/eval_window.py` re-scores the same checkpoint on 64
batches (512 KiB):

```
longctx1 · window 524288 B · headline 1.7634 BPB · fin 1.6882 BPB
```

Same bytes, same tool, for the anchors:

| model | BPB on this 512 KiB window | data seen |
|---|---:|---|
| zstd -19 | 2.9540 | — |
| **this run** | **1.7634** | **523 MB** |
| GPT-2 small (124M) | 1.0580 | ~40 GB |

So: **1.68× better compression than zstd**, and **0.71 BPB behind GPT-2 small**
while having seen **76× less text**.

## Depth and context are substitutes — measured

`periter` (CE per loop iteration, evaluated with the run's own next-byte
prediction) on the 512 KiB window:

```
2.0405 → 1.7200 → 1.6846 → 1.6838     (gain per iteration: 0.32, 0.035, 0.0008)
```

The 4th iteration buys **0.0008 BPB** — nothing. At 1 KiB context the same metric
gave 2.15 → 1.86 → 1.81 → 1.80 (0.29, 0.05, 0.01). Long context **eats the depth
payoff**: once the model attends exactly over 1 024 CSA entries, "think longer"
stops adding anything.

**Consequence for the recipe: at 8 KiB, `max_iter=3` — the 4th iteration is 25%
of the step for 0.0008 BPB.**

## `doc_masking` barely fired — and the reason is the sidecar

```
docmask_scored 523 973 088 B · docmask_masked 314 912 B (0.06%)
docmask_doc_ends 83 windows
```

`<data>.docs.json` is derived from the source files, and the web shards are
~6.3 MB blobs (first `starts`: 0, 6 295 756, 12 587 441 …). One 8 KiB window
almost never reaches a unit boundary, so there is almost nothing to mask. The
feature is correct and inert at this granularity; real document boundaries need
`build_mix4.py` to split the source blobs, not a config flag.

## A real bug this run caught

`_naive_dispatch` raised `TypeError: unsupported operand type(s) for &: 'NoneType'
and 'Tensor'` whenever `keep is None` — the dropless route, which is exactly the
arm `doc_masking` forces (`moe_capacity=0`). `_padded_dispatch` had been taught
the same signature months earlier; the CPU reference loop had not. Fixed with a
gate that compares `keep=None` against an all-ones mask bit for bit:
`test_the_naive_dispatch_keeps_every_row_when_keep_is_none`.

## What to run next

1. **`max_iter=3`** on the same shape — free 25%.
2. **A/B `use_csa=false`** at 8 KiB, 600 steps: the level that has to earn its
   keep. HCA at 8 KiB has only 32 entries — untested whether it adds anything.
3. **16 KiB** via the state-carrying prefill (a one-shot 16 KiB forward does not
   fit; the ring path does).
4. Long-context SFT later: this checkpoint is a *pretrain* base, not a chat model.
