# MODE: AWAY

Owner asleep (2026-10-28 night). Full autonomy until he returns.

## Goal
1. Fix the tile-estimator bug that voids every rank >= 128 arm (`_grouped_gemm_pair_kernel`
   needs 196 608 B while its own docstring table says 98 304 — the table is wrong by 2x and the
   tile search does not descend on that path).
2. Re-run the rank ladder (64 / 128 / 192 / 256, 3 seeds, lr=5e-4) and pick the rank by
   measurement, not by the SCT paper's prior.
3. Launch the flagship pre-train on the measured recipe and leave it running with a monitor.

## Recipe (measured winners only, all 3-seed A/B)
- CSA on (+0.0106), mHC on, AttnRes on (-0.050), share_u on (-0.013), attn_latent_rank 256
  (-0.021 and -36% dense MAC), max_iter 3, LayerRoPE paper form (widen the LR basin).
- lr 5e-4 — the only LR with a tight spread in two independent A/Bs (0.007 / 0.036).
  1e-3 is stable ONLY with LayerRoPE (3.2373 +- 0.081) and is kept for the next run, not this one:
  an unattended multi-day run must not gamble on a 600-step stability reading.
- 8k window, doc_masking on (needs the .docs.json sidecars that mix_v7's sources carry).

## Closures tonight (do not reopen without a new measurement)
- Softmax/CE collapse: does not fire here (fp32 loss, max margin 4.69 vs cliff 16.6; gradient
  sum violation 0). `ce_fixed` stays default-false.
- LayerRoPE v1 is superseded by the paper form (`v1` dropped); depth index = iteration, not position.
- ByteFlow: -0.43 BPB quality, -15% speed. Not in this run.

## Report on return
STATUS: DONE + what ran, what the numbers are, what is still open.

## Launched overnight (2026-10-28 22:40)
- `logs/night_chain.sh` (setsid): waits for the rank A/B (`logs/ab_rank_up/`, 4 arms x 3 seeds,
  lr 5e-4), then `logs/pick_rank.py` decides (an arm must beat control by more than 2x the larger
  spread; ties keep rank 64), then launches the flagship.
- Flagship: medium (141.9M, share_u -> ~105M active-mem), max_iter 3, CSA(chunk 8 + coarse 256),
  mHC + AttnRes, LayerRoPE paper form, attn_latent_rank 256, rank from the A/B, E=256,
  b24 s1024, lr 5e-4, warmup 150M bytes (1.9%), decay_frac 0.15, mix_v7 (22.2 GB),
  320k steps = 7.86 GB of tokens, ckpt_every 25k (~12 x 1.5 GB), eval_every 2k.
- doc_masking OFF: the sidecars (`docs.json`) are not built for mix_v7's sources; building them is
  the first task for the next run (scripts/build_doc_sidecar.py).
