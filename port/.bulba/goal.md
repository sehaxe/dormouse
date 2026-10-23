# MODE: AWAY

Owner asleep 2026-10-22 ~00:00, expects autonomous work until return.
Goal: freeze the ternary release recipe, then start the real pretrain on it.

## Where things stood at handoff

- Ternary ladder running (`~/logs/ternary4.sh`, one process at a time):
  bfctl (bf16 control, Engram ON, 141.9M) / q64b (ternary, rank64 d1024) /
  q32w (ternary, rank32 d1920) / q16w (ternary, rank16 d2560) /
  q64a4 (q32w-form + int4 activations). 3 seeds each, 2000 steps,
  medium, b8 s512, mix_big, window 81 920 B.
- **First verdict (valid, same params, same flags):**
  bf16 control 3.0800 · ternary rank64 3.1367 (+0.057) ·
  **ternary rank32 d1920 3.0650 (−0.015 vs bf16)** → the rank→width
  reinvestment pays for the quantization and then some.

## What "done" means tonight

1. Score the ladder (3 seeds each arm), write the verdict into
   `docs/reviews/2026-10-22-ternary-ladder.md` with windows and the
   matched-params check.
2. Freeze the recipe: `configs/release_ternary.toml`.
3. Launch the real pretrain on it (`mix_v3` = mix_big + instructions
   2%→4%→7% decay, ternaried factors+dense, attention bf16, rank32/d1920,
   MSA on, attnres on, seq 2048 segmented or 1024 one-shot — whichever
   fits, measured first with `--fit`).
4. Do NOT claim a number without its config, window and commit.

## Hard rules for this session

- One GPU process at a time; the control runs in the same script as its arms.
- A comparison is valid only if `params` in the first log line match.
- Every fix gets a gate that fails without it.
- Commit per task, path-scoped; no push without the owner.
- Nothing gets deleted without a dated verdict or a written reason.
