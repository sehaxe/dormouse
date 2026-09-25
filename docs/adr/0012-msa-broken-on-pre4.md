# MSA is broken on the pre.4 stack: arm off, kernel rewrite deferred

On burn/cubecl 0.22.0-pre.4 the MSA arm produces out-of-bounds gathers in both
of its paths: the hand-written fused kernel (gather reads up to 11.5 GB past a
7.5 MB allocation; compute-sanitizer: gather_kernel_t_f32_i_i64 OOB) and the
tensor-op fallback (burn gather with the same garbage indices). The garbage is
upstream of gather — pre.4's changed topk/indexing semantics feed wild block
indices; pre.3's reshape materialization hid this by copying inputs. Minimal
repro: vendor/burn-fused/crates/burn-msa/examples/msa_repro.rs (pass 1 dies,
pass 0 clean).

Decision: MSA ships disabled on pre.4. The fused kernel is env-gated off
(DM_MSA_FUSED, sparse_kernel.rs), presets carry `use_msa = false`, and the
official baseline runs without the arm. This is doctrine-consistent, not a
retreat: MSA never won an A/B (ADR-0004 long gate pending), and every run until
then paid its gradient_detach leak-tax for free. Return conditions, all three:
(1) the indexer's indices verified in-range on pre.4, (2) the kernel's dense
contract re-established against pre.4 views with a clean two-forward
compute-sanitizer run, (3) the MSA-vs-dense A/B finally run. Until then the
attention arm is KDA + (Engram + experts + PonderNet) as usual.
