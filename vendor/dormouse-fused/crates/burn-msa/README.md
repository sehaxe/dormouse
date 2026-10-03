# burn-msa — Qwen Sparse Attention (QSA), the re-entry

Compressed MQA block indexer + micro-block masked sparse softmax attention,
after the tech report's §QSA (Eq. 12-19) — the ADR-0014 re-entry conditions.

Three gates, all green on CPU (ndarray), the CUDA arm feature-gated:

1. **gate1** — sparse over ALL complete blocks == dense causal attention,
   bit-near (incl. the always-included incomplete tail). `tests/gate1_dense_contract.rs`
   (CPU) and `tests/gate1_cuda.rs` (`required-features = ["cuda"]`).
2. **gate2** — indexer picks in range and equal to the scores' own top-2;
   the final incomplete block's tokens are ALWAYS members of the window.
   `tests/gate2_indices.rs`.
3. **gate3** — stage (a) of the two-stage training: KL-distill the dense
   head-summed attention into the indexer; >=90% top-1 agreement after
   400-1000 AdamW steps (paper LR 1e-3) on synthetic data.
   `tests/gate3_distill.rs`.

The tensor path consumes the picks as a micro-block MASK (the paper's own
tensor stage); the ADR-0015 topk->gather primitive is NOT load-bearing here
(its remains "partial row" duplicate defect is neutralized: a repeated id is
a no-op on a mask). The fused kernel that gathers blocks outright is stage
(b)'s GPU work and must reproduce gate 1 bit-near.
