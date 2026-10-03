# MSA re-entry (wt/msa, 2026-10-02)

Lane wt/msa (Qwen Sparse Attention, tech report §QSA): gates 1-3 GREEN on CPU.

- gate1 (sparse KB=all ≡ dense, bit-near < 2e-5): ndarray
  `vendor/dormouse-fused/crates/burn-msa/tests/gate1_dense_contract.rs` 3/3,
  flex `crates/dormouse-core/tests/msa_gate1_flex.rs` 2/2.
- gate2 (indices in range, top-kb = scores' own top, tail always in mask):
  `tests/gate2_indices.rs` 2/2.
- gate3 (KL distill of max-pooled dense into indexer, ≥90% top-1 teacher
  agreement): `tests/gate3_distill.rs` 1/1 (68.8 s, 1000 AdamW @ 3e-3,
  synthetic).
- Core wiring: `use_msa` off-default, `--msa` path via config seam, probe
  counters `msa`/`msa_distill`, eval line prints them. `msa_kb >= B` refused
  loud in validation.

Open: fused gather kernel (stage b, GPU — card busy), indexer LR group in the
trainer (still `Group::Rest`), no A/B yet. Do not cite as quality evidence.
