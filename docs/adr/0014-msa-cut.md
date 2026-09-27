# MSA is cut, not deferred (supersedes ADR-0012's "arm off, rewrite deferred")

ADR-0012 shipped the MSA arm disabled on the pre.4 stack: the indexer emits
garbage block indices and every gather goes out of bounds (cuEventCreate 700 /
IllegalAddress), in both the fused kernel and the tensor fallback. That left
the repo carrying a broken mechanism at full price - the crate, the config
keys, the CLI flag, the optimizer markers, the module, the gradient-free
exceptions in the seam test, the block-sparse FLOPs counters in the TSCT
diagnostic - with `use_msa = false` in every preset and no run that ever
executed it. Off is not the same as gone: a disabled arm still gets renamed,
still breaks the routing validator, and still has to be reasoned about.

Decision: delete it. `vendor/burn-fused/crates/burn-msa`, `use_msa` /
`msa_topk` / `msa_block` in the config schema and presets, `--no-msa`, the
`AdaptiveAttention::blend` router, and the QK_KV optimizer group go away. The
attention block is KDA only; the loop's `w_attn` controller weight still gates
it. The `AdaptiveAttention` wrapper struct stays, for one reason: it keeps the
checkpoint param prefix `loop_block.shared_attn.gdn2.*` stable, so every
existing checkpoint (the 48M-slot production runs included) still loads.

Why this is not a retreat: MSA never won an A/B (ADR-0004's long gate never
ran), and every run until then paid its gradient_detach leak-tax for a
mechanism that could not execute.

What replaces it for now

- **Retrieval / exact memory**: the Engram n-gram input features, per BLT's
  ablation (byte-level n-gram tables feeding the residual stream). That is
  the cheap half of "exact retrieval" and it already trains.
- **Long-context exact attention**: a *working* sparse-attention
  implementation, re-added from FLA / flash-attn rather than from this crate.
  The 2026 hybrid literature still wants it at scale (Kimi Linear, Qwen3-Next
  at 3:1 linear:full) - see the AGENTS.md design playbook, item 4. What is
  deleted here is a broken indexer on a broken kernel, not the idea.

Re-entry conditions (all three, same as ADR-0012's, plus the source)

1. A working block-sparse attention implementation exists: FLA's
   `sparse_attn`/top-k kernels or a flash-attn varlen path, ported in the
   vendored-crate style, with its indices verified in-range under
   compute-sanitizer on pre.4.
2. The block-indexer runs under autodiff with a two-forward
   compute-sanitizer run, and the dense contract is re-established against
   pre.4 views.
3. The MSA-vs-dense A/B finally runs and wins on BPB at a matched budget.
4. A third attention arm is wanted at all: at the current context lengths a
   1-arm KDA block plus a 48M-row Engram is the honest design, and adding an
   arm back means paying for the blend again (this ADR removed the
   degenerate one-arm version of it).
