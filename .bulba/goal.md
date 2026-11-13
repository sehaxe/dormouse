MODE: AWAY
STARTED: 2026-11-13 23:58
UNTIL: the user returns.

## What is running

`port/scripts/run_tsct_r256.sh` — the long TSCT run, launched 23:58, expected
5-7 h, log `/home/sehaxe/logs/run_tsct_r256.log`, jsonl in
`/home/sehaxe/logs/arch/run_tsct_r256_s1.jsonl`, checkpoints in
`/tmp/opencode/run2gb_tsct_r256`.

The arm: `configs/arch_tsct95_r256.toml` — the live 95M recipe with a factorised
FFN at rank 256 (60.2M params), full corpus, 107,421 steps, b24, s1024,
compile, Muon 12. Control: 1.1871 at the same budget (2026-11-12, gated FFN).

The question: at 120 MB rank 256 cost +5.2% BPB for 1.62x speed. Does the price
shrink with 20x more data, or grow because rank is a ceiling?

## Next in the queue (in order, no permission needed)

1. Read the arm's first eval at step 13,000 and compare with the control's
   13,000 point (1.4361). Threshold: if the gap is under ~+3% at 39k, the arm
   is a win and the report says so; above +7% it is a measured refusal.
2. After the arm: `scripts/smoke_long.py`-style probe at 16k/32k for the
   segmented path (ms/step, peak GiB) — the table promised to the owner.
3. Write `docs/records/reviews/2026-11-14-tsct-long.md` with the numbers,
   whatever they say.
4. Keep `docs/` clean; no new top-level files outside records/.

## Rules that apply tonight

- GPU work only through `scripts/gpu_lock.sh`; nothing else on the card.
- Every number names its window and budget.
- If the run dies: read the error, fix the cause, relaunch from the last
  checkpoint; never silently switch to a different arm.
