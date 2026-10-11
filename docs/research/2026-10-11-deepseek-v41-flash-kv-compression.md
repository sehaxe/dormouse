# DeepSeek-V4.1-Flash: the RAM-offload playbook, confirmed at the frontier

**2026-10-11.** Source: arXiv:2609.19969 (DeepSeek-AI, submitted 2026-09-17),
abstract read through the arxiv MCP and the arXiv abs page. Weights are published
(`deepseek-ai/DeepSeek-V4.1-Flash` on HuggingFace); no code release is claimed in
the abstract. Numbers below are the authors'.

## What they did

A 552B-parameter multimodal MoE for 1M-token contexts, whose whole paper is the
deployment cost of long context, in the order the cost appears:

1. **A causal encoder-decoder split (CED)** that activates 8B parameters per
   token during prefill and 16B during decode - the architecture is shaped by the
   workload being input-heavy (agents read far more than they write).
2. **Cross-layer KV cache reuse in a compressed sparse attention (CSA2)** - later
   layers read earlier layers' compressed cache instead of carrying their own.
3. **FP4 KV caching.** Together, these put the global KV cache (always in HBM) at
   **890 bytes per token**, about a quarter of the previous model's.
4. **SWA Bounded Replay**, which puts the *persistent* cache - the part that must
   live on SSD or in host memory - at about **one eighth** of the previous
   model's, by replaying a bounded sliding window rather than everything.

Their claim is that the compressed model is better than the uncompressed
baseline, not merely cheaper.

## Why this is the playbook we are already on

§3.5's item 1 - the highest-ranked item in the Qwen3.8 playbook and the reason
dormouse has an `--engram-ram` path at all - is *host RAM as an extension of
VRAM, with prefetch, for the tables that grow without bound*. DeepSeek arrives at
the same place from the KV side, and then goes further on every axis we care
about:

| the currency | them | us |
|---|---|---|
| per-token memory in fast memory | 890 B/token (FP4 KV + cross-layer reuse) | the KDA state is O(1) per sequence, not per token |
| per-token memory that can spill | ~1/8 of the previous model, bounded replay | `--engram-ram` tables in host RAM, 12.3 GB at 48M rows |
| what grows without bound | the KV of a 1M context | the n-gram table only grows with `engram_rows`, i.e. never at runtime |

The two decisions that matter for us are the ones they had to make and we have
not: **the spilling tier has a bounded-replay policy** (their persistent cache is
1/8, not 1/4 - the extra factor comes from *replaying a window* instead of
keeping everything), and **the fast tier is compressed to a format (FP4) rather
than to a smaller dtype**.

## What is adoptable here, and what the measurement costs

- **A replay/bounded-resident window for the host tables.** Our Engram rows are
  hash-addressed and never evicted: a collision overwrites, and the *training*
  rows are the whole table. A bounded-resident policy would be about which rows
  stay hot in host RAM and which are re-fetched, which is a real design change
  with a real risk (a replayed row is a stale row). Cost: a lane, not a patch.
- **FP4 for the memory that matters.** We already ship Fp4 in both places it
  could go (`act_quant`, and the TSCT factor formats) and §2.1 records what
  happened when it was used: the sm_120 fp8 path degraded at overfit, so the
  honest reference for any FP4-KV analogue here is the lowp_bf16_cuda.rs one -
  bit patterns in storage, mathematical computation in fp32.
- **The 8B-prefill/16B-decode asymmetry is a note, not a plan.** Our workload is
  already input-heavy (one byte per step), but the reason to want a cheaper
  *read* path does not exist in the same way: we do not carry a per-token cache
  at all.

## What does not transfer

The whole KV-cache framing: dormouse's per-step state is the KDA recurrence's
`[b, h, k, v]` tensor, which is fixed-size in the sequence length, so there is
nothing to compress, spill, or replay. What transfers is the *direction* - the
frontier is spending parameters and precision asymmetrically to keep the growing
part of memory in host RAM, which is the bet §3.5 already placed.
