# Overnight run — long context where CSA/HCA are load-bearing (2026-10-27)

## The run

```
medium · 102 678 928 params · from scratch
seq_len = max_seq_len = 8192 B · batch 1 · grad_accum 4  → 32 768 B per optimizer step
CSA fine (chunk 8, kv 64, topk 32) + HCA coarse (chunk 256)
mHC (K=2) + AttnRes (composed) · share_u (one basis per site) · attn_latent_rank 256
dropless MoE (moe_capacity=0) · Triton · Engram in VRAM · doc_masking
lr 1e-3 · warmup 16.4 MB tokens · decay_frac 0.1 · 16 000 steps
data  mix4/web.bin (2.35 GB) · eval mix4/web_eval.bin, 4 × 8192 B window
ckpt every 2 000 → checkpoints/longctx1/
```

Measured step cost **≈1.7 s** → **≈7.7 h** for the whole run, **524 MB** of text
read. Shape search (why 8192×1 rather than 4096×2 or 1024×8):

| shape | per-step | bytes/s |
|---|---:|---:|
| b8 × 1024 | 197 ms | **41 500** |
| b1 × 4096 | 278 ms | 14 700 |
| **b1 × 8192** | 441 ms | **18 600** |
| b2 × 8192 | — | OOM on 15.6 GiB |

Long context costs 2.2× the throughput of the short window. That is the price of
the arm, not a defect; 8192 is the longest context the card holds in one forward
(`segment_len>0` is refused with `use_csa`, and the reason is real: what a
segment cut should do to a half-full CSA chunk is undecided).

## What this run is for

CSA fine has 1 024 entries at 8 192 bytes (128 at 1 024) and HCA coarse has 32
(4 at 1 024). Below ~4 KB the coarse level has nothing to summarise and the fine
level has fewer than a thousand keys, so **a 1 KB run cannot show whether either
level pays**. This is the first training run whose shape makes the question
answerable.

The comparison point is not a previous run — no previous run used this corpus,
this context, or this residual stack — so the number to watch is the **curve**
(eval BPB every 500 steps), not a delta against `mix1`.

## `doc_masking` is wired but inert on this corpus — measured, not assumed

`docmask_masked: 0`, `docmask_padded: 0`, `docmask_doc_ends: 0` on every logged
step. The sidecar is not broken; its units are **source files**, and the web
shards are ~6.3 MB text blobs (first `starts`: 0, 6 295 756, 12 587 441 …).
One 8 192-byte window therefore never crosses a unit, so there is nothing to
mask.

Consequence: the run is **not** harmed (a window is inside one blob), but the
claimed benefit — no loss across a real document boundary — **does not apply at
document granularity**. Getting it would need the builder to split the source
blobs into documents (blank-line / heading heuristics) and emit those offsets,
which is a `build_mix4.py` change, not a config flag.

## Fixes landed with this run

* `_naive_dispatch` raised `TypeError: NoneType & Tensor` when `keep is None`
  (the dropless route) — the exact arm `doc_masking` forces, so the run died on
  any CPU-fallback forward. Fixed + gate
  (`test_the_naive_dispatch_keeps_every_row_when_keep_is_none`).
* `scripts/ab_width_clean.py` was the repo's last two ruff errors.
* `docs/guides/dataset.md` did not exist although the config and the builder
  cited it twice; now it does, with the three named source deviations and the
  measured ×1.00–1.07 near-dup multiplier.

## What to check when it finishes

1. eval BPB at 2 000 / 8 000 / 16 000 — is the curve still falling at the end?
2. `periter=` — does the 4th loop iteration still give ~0.01, or does long
   context/`doc_masking` change the depth payoff?
3. `csa_forward` / `csa_coarse_forward` counters are 1 per loop iteration per
   forward — confirm they are not silently zero (a zero would mean the branch
   never ran).
4. A/B against the same shape with `use_csa=false` on 600 steps, then a
   compressed-16k arm if the 8 KB curve is healthy.
