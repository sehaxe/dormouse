# Fused rewrite: two rungs, then death

ADR-0003 measured fused and it lost (1.25x slower at flagship; the 1.7-2.0x
smoke number was pre-fix and is retracted). The 2026-09-25 analysis
(research/2026-09-25-fused-rewrite-plan.md) quantified three causes: the
`arms_inner_adjoint` re-run per loop iteration (~1-1.5 s of the 2.16 s deficit),
a full weight-grad D2H round trip every step (~30 blocking transactions, ~60 MB
PCIe), and hand-rolled fp32 FFMA matmuls against burn's autotuned tensor-core
path. The "direct adjoints" already in kernels.rs relocate the round trip rather
than remove it and stay unwired.

Decision: one rewrite window with two ranked rungs and a hard kill switch.
Rung 1 — seed KDA/MSA/Engram adjoint kernels from the per-iteration buffers the
forward already saves (only KDA's recurrent state `S` needs adding to the
workspace), delete the inner-adjoint re-run, one device sync per step, grad
registration on device buffers, junk-kernel launches removed. Rung 2 — delete
the hand matmuls, materialize + burn's `matmul_autotune`, persistent workspace.
Rung 1 must buy >= 1 s/step at flagship; Rung 2 must reach parity-or-better vs
the fusion-on burn path (pre.4). Either rung failing its number deletes all
~7.4k LOC of fused/ with this ADR as the pre-justification. CUDA graph capture
(available since pre.3) is the single bounded last resort and nothing more.

The fusion backend flag is part of the same window: it unlocks on pre.4
(#5673 extension-crate metadata callbacks) and is benched as its own arm — the
bar fused must beat is fusion-on burn, not yesterday's burn.
