# MODE: AWAY

Owner asleep (2026-10-28 night). Full autonomy until he returns.

## State (2026-10-29 night, updated)

**Flagship is RUNNING and healthy** — do not restart it.

```
step 55 300 / 320 000  (17%)   loss 1.1551   26 799 B/s   1.36 GB of 7.86   0 nan_skipped
eval: 50k 1.8062 · 52k 1.7849 · 54k 1.7800      ETA ~67 h
checkpoints: dormouse/checkpoints/flagship/flagship.pt (979 MB, every 25k)
recipe: medium 102.7M · CSA fine+coarse · mHC · AttnRes · share_u · LayerRoPE(paper)
        latent 256 · E=256 rank 64 · iters 3 · b8 s1024 accum 3 · lr 5e-4 · mix_v7 · 8k
log: ~/logs/flagship.log · ~/logs/flagship.jsonl · dashboard http://127.0.0.1:3789 (tailnet)
```

Reference: mix1 finished at BPB 1.883 after 4.18 GB; we are at **1.7800 after 1.36 GB**.

## Next run (flagship2, NOT started — needs the card)
- `lr=1e-3` (measured best on the full recipe: 3.2373 ± 0.081 vs 3.3938 ± 0.026 at 5e-4,
  3 seeds, same battery). Everything else identical, so the two runs form a clean at-scale A/B.
- Missing measurement before trusting it: there is **no 5e-4 arm in the same ladder** — the
  comparison comes from the earlier layerrope2 battery (b8 s512). Re-run both LR at the
  flagship's exact shape (b8 s1024 × accum 3) with 2 seeds when the card is free.

## Open items (owner decisions)
1. **Cleanup**: 65 worktrees, 127 GB; 55 of them dirty (uncommitted agent leftovers).
   Deleting them loses work — needs an explicit OK. Only 4 were clean+merged (removed).
2. **Formatting**: 55 files would be reformatted by `ruff format` (pre-existing debt). Safe but
   a big diff that would collide with those dirty worktrees.
3. **GPU purchase**: cart holds 2 × MSI RTX3090 AERO 24G @ $616.91 = $1 233.82 (Superbuy).
   Verified: single-SKU lot (no variant trap), price = Chinese domestic teardown tier.
   Not verified: seller DSR (none shown), photos, mining history. Recommendation: buy ONE with
   detailed inspection + FurMark video, then the second. Alibaba/BY delivery works (checked).
   Migration work needed for sm_86: `moe_kernels.py` has sm_120's 101 376 B smem ceiling
   hardcoded; tile table must be re-measured on Ampere.
