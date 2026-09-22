# fused/ fast path: measure, then keep or delete

Status: accepted, smoke verdict 2026-09-21.

`crates/dormouse-core/src/fused/` is 6,568 LOC, 54% of core: a hand-written CUDA autodiff op for the whole ponder loop. It is off by default (`DM_FUSED=1`), excluded from the flagship recipe (host rows, JEPA targets, bf16, act quant, GR all bypass it), and its backward contains acknowledged approximations (gradcheck limit 2.0). Decision: one smoke measuring tokens/s fused vs unfused on a config where both paths run. At 1.5x or better we finish compatibility with the flagship recipe; below that we delete the directory and pursue performance with CUDA graphs and burn-fused kernels.

Verdict: kept. Smoke 2026-09-21, small preset, batch 6, s512, 100 steps, JEPA+DSpark off, Fp8 factors, Muon+: unfused ~8.4 s/step vs fused ~4.8 s/step (wall clock 15 min vs 9 min including shared init), 1.7-2.0x. Compatibility work must carry two caveats: the fused forward is not bit-identical (step-0 ce 5.537 vs 5.514), and the fused checkpoint is 37 MB vs 70 MB, so state completeness needs verification before resume-through-fused is trusted.

Considered options: delete now (loses real, gradcheck-tested perf work unmeasured) and keep unconditionally (risks months of maintenance for an unproven gain). Both rejected.
