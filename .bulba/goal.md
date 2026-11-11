# MODE: AWAY — night of 2026-11-11

Owner is asleep. Work autonomously until he returns, then close with a full
report (`.bulba/report-2026-11-11.md` + a summary in chat).

## The question this night must answer

"Highest intelligence density": minimum file bytes, minimum inference FLOPs,
minimum training data and FLOPs, maximum quality. The owner's own framing:
never learn what you cannot keep — no compressing a smart model afterwards.

## Night queue (strict order, card must never idle)

1. **fp4 fused kernel** (subagent `ses_e38b868b2ffe4eQwLI5eKxE9m4`, running).
   Measured basis: `_scaled_mm` fp4 is **3.16× faster** than bf16 at
   `[24576,832]x[832,2048]`; the eager `quantize_nvfp4` costs 10.67 ms and is
   the entire 5.4× slowdown of the `e2m1` arm. Fix = fused kernel, target
   ≤0.6 ms total. Gate: the blocked-scale swizzle, because a wrong swizzle
   "succeeds" with 46% error.
2. **TSCT rank-64 at 275M** — the biggest single lever (0.048/0.071/0.096/0.121
   = 5/8/11/14/17 h saved), 3 seeds vs the gated control, 120 MB. This is the
   experiment that decides whether the project's central bet (low-rank =
   quality) survives at a scale that matters.
3. **AttnRes memory fix** (score one source at a time: 507 MiB → 39 MiB peak per
   layer) so the arm can actually run at b24 — it OOM'd at b24, b20 AND b16.
4. **QAT-finish at 95M** if time remains: the 18.8 MB artifact.

## Rules for the night

- Every number comes from a command, not from a paper or a prior claim.
- Every A/B is 3 seeds; a delta inside 2× the seed spread is "not separated".
- Any arm that loses is recorded as a measured refusal with its number, in
  `docs/reviews/`, and stays as a flag — never deleted silently.
- `scripts/gpu_lock.sh` for every GPU job. Never bypass it, never kill another
  lane's job to take the card.
- Commit per task, own files only, no preset of a frozen control edited.
- If a run fails, read the reason and record it; do not retry blindly.

## Progress (updated 04:35)

1. fp4 kernel: done (`00c9626`) — quantize 13.57 → 0.32 ms (43x), bare GEMM 3.16x
   faster than bf16, but the end-to-end arm is a 6.7% loss (bytes/s is tokens/s).
2. TSCT wired into the live transformer block (`3e5c727`) — was a silent no-op.
3. **The first A/B was CONFOUNDED and killed**: the tsct arm's init was 13.3x
   larger than the block contract (review caught it). Fixed in `1043573`, gate
   added that fails on the old code.
4. Mechanism arm measured: tsct275 110.1M = 2.1067 vs dense275 273.4M = 1.6406
   (+0.466). That is a parameter-count difference, not a rank verdict.
5. Reinvestment arm (`arch_tsct275_wide`, 272.8M/544M == control) running at b4.
6. AttnRes training peak fixed (`0bfe05e`): the mixer's retained bytes
   3045.8 → 2.6 MiB per block; the earlier fix was forward-only.

## Known results to carry into the report

| arm | result |
|---|---|
| gated FFN (SwiGLU, param-neutral) | **−0.0145 BPB**, 3 seeds, t=−19 → live recipe |
| mHC in transformer | **+0.0853 BPB** at b16 (1.5714 vs 1.4861), +936 MiB — measured loss |
| AttnRes full N=L | unrunnable: OOM at b24 (858 MiB short), b20, b16 (48 MiB short) |
| MoE bank buffers | −468 MiB/layer ≈ −5.5 GiB at b24 (fixed), cap 2.0→1.25/dropless |
| fp4 GEMM | **3.16×** vs bf16 at live shape; 7.5× at 4096³ |
| 4-bit activation storage | 4.34 GiB saved of the activation sites (80%), max batch 24→28, but 5.4× slower because of the eager quantizer |
| qwen3.8-27B | 3.6→3.8 at IDENTICAL architecture: +8.2 SWE-Pro, +20.4 OSWorld — post-training, not architecture |
