# arXivBangers scan — 2026-11-08

**Source:** <https://arxivb.org/> (a curated feed, not peer review; every claim below
is the paper's abstract, tagged **[A]**, unless it is our own measurement **[M]** or
arithmetic **[S]**). **Scope:** 43 cards on the feed (30 published, 10 contenders,
3 upcoming) screened by title/summary; 14 abstracts read in depth. The site's
editorial score is a ranking signal, not evidence.

**Why this scan exists:** the current live recipe is the dense **bytes** control
(`flagship3_dense`, 95M) after the patch arm lost +1.8 BPB at matched bytes. The
papers below are the ones that could change the **training** path to the 275M/1B
release, not the ones that are merely interesting.

## Act now (tied to the current queue)

| paper | arXiv | claim that matters here | action | priority |
|---|---|---|---|---|
| **Hybrid linear attention (3:1 Gated DeltaNet : full attention)** | 2604.03444 (Olmo Hybrid); 2510.26692 (Kimi Linear) | At **60M**, a 3:1 GDN/attention hybrid reaches **1.077 BPB vs a transformer's 1.155** (−0.078, Olmo Table 5); Kimi Linear repeats the sign at 653M active (PPL 5.65 vs 5.77). This is the one mechanism with a published compute-matched win near our size. | The repo already has the arms: `arch_hybrid_kda.toml` (95.9M, pure KDA) and `arch_hybrid31.toml` (109.9M, 9 KDA + 3 attention, iso-FLOPs with dense C0 at 692.7 vs 692.9 MFLOP/byte). `scripts/queue_arch.sh` runs dense/KDA/MoE8/hybrid31, 2 seeds × 73.7 MB, pre-registered rule (2× spread + 0.02 BPB). **Running now**; winners get 3 seeds. | **P1** |
| **Multi-Token Prediction Done Right (Soft-MTP)** | 2610.38331 | Standard MTP's CE gradient has a low-SNR residual that buries the useful KL self-distillation term; a straight-through KL-recovery trick removes it with no params and no extra logits; wins from 167M to 1.2B, including spec-decode speed | Our soft-MTP DROP (`2026-11-03-three-verdicts-and-a-leak.md:18`) was a different mechanism. Read the code, implement the KL-recovery in `model/mtp.py`, A/B at 95M/275M; gate BPB + spec acceptance. | **P1** |
| **Pretraining Data Scale Reverses the Muon-AdamW Fine-Tuning Gap** | 2611.03633 | Beyond a large pretraining budget, AdamW-pretrained models fine-tune better than Muon-pretrained ones; the crossover comes at smaller token/param ratios as models grow; explained by margin-to-spread | We already audited Muon's LR (0.24 BPB defect, `2026-11-07-muon-audit.md`). Before locking Muon for 275M/1B, run the same SFT on a short Muon vs AdamW pretrain pair. Do not switch on the abstract alone. | **P1** |
| **You don't need long context** | 2611.05306 | Frequent context compaction preserves long-horizon/long-input accuracy; later tokens cost up to 7× more; questions the need for long-context models | Before the 1M/CSA investment, run a task-level eval with a short window + compaction. If it holds, the 1M goal drops to 8–16K and the CSA/HCA work is re-scoped. | **P1 (decision)** |
| **Near-Zero Token Embeddings Silently Stall Training in Gated Sequence Models** | 2611.04449 | Untied gated/hybrid models can grow embedding rows with norm <1e-6; zero input makes SiLU/query readouts vanish, gradients explode, FP32 global-norm clipping overflows and silently drops ~20% of steps; GateFloor (`SiLU(z)+δ`) / GroupClip fix it | Cheap diagnostic on the old TSCT/KDA checkpoints: min row norm of the embedding table; if any row is near zero, add GateFloor to the KDA path. Byte-level vocab 256 makes it unlikely, but the gate path is exactly the vulnerable class. | **P2** |

## File for later (scale / long-context / serving; gated by the decision above)

| paper | arXiv | relevance |
|---|---|---|
| Dory1.0-4B: A Looped Normalized-GPT | 2611.03998 | 4B looped nGPT, elastic depth, last-loop KV, NVFP4, 6T tokens. **Not a 95M action:** our own 2026-11-06 measurements put loops behind dense at ≤200M (MoR 135M +0.0155 NLL; φ=0.46; 1 layer × 3 = 14% of a 12-layer control; `docs/research/2026-11-06-depth-recurrence.md` says do not spend the 3-seed A/B). The older 2.508-vs-2.653 loop win was confounded by the Muon LR bug (`docs/reviews/2026-11-06-control-transformer.md:14-27`). Revisit at 275M/1B; the last-loop KV trick is a serving idea. |
| Shallow Encoding Emits Deep Memories (SEED) | 2611.06084 | Shallow prefill via emitters + low-rank recurrent-state compression (sink + factors) for Gated DeltaNet/Kimi Delta/Mamba-2; 1.2–1.4× serving on 30B–300B hybrids. Applies if the 1B release keeps KDA/CSA. |
| Latent Sparse Attention | 2610.42147 | Hierarchical key clusters in a learned latent space; D-LSA distils a pretrained model, N-LSA trains natively; 22.7×/4.6× layer-level prefill/decode at 10M tokens. A candidate for the 1M path if it survives the compaction decision. |
| BellowsKV | 2610.35023 | One shared full-dimension latent per block + low-precision/eviction; 97–98% accuracy at 13–14× KV compression, training-free. Serving-only; relevant after the release exists. |
| ParaTTT | 2610.38255 | Parallel test-time-training layers; 340M/1.3B match or beat Gated DeltaNet/Mamba-2, 4.9× faster than TTT-MLP. A memory-layer alternative, not a drop-in for our KDA recipe. |
| End-to-End MXFP8 RL on Blackwell | 2610.34915 | MXFP8 pays above ~4.5K tokens/micro-batch; BF16 stays 2.5–10× more consistent; seven weight-sync conditions, five silent failures. Confirms our fp8-at-M≈4K negative; relevant only if we do RL with a separate sampler. |
| Nemotron 3.5 Super/Lightning | 2610.38095 | 120B/12B-active hybrid Mamba-Transformer MoE, NVFP4, native MTP, 4,856 tok/s/GPU on GB200. A reference for the 1B hybrid direction, not reproducible on one 16 GB card. |
| Test-Time Self-Play | 2611.07367 | Verifier-free test-time scaling, +35% absolute mean@8 on LiveCodeBench at no verifier. Relevant if the release gets an agentic post-training phase. |
| MobileSpec | 2611.04797 | Forward-only block-spectral updates; 1–5 MB specialists, INT4 bases, 8B on an iPhone. Not for our GPU pretraining; a possible on-device later. |
| The Behavioral Rate of Fine-Tuning | 2611.06598 | LoRA compressibility is a dataset–receiver property; aggressive adapter compression denoises corrupted fine-tunes. Relevant to QAT/adapters, not pretraining. |

## Screened, not for this project

Robotics/world models/video (2611.06905, 2611.03815, 2611.02497, 2611.06627,
2610.36729, 2610.33589, 2610.41216, 2610.37585, 2610.40747, 2611.03634,
2611.06956), evaluation/agent harnesses (2611.06910, 2611.07071, 2611.03233,
2611.05139, 2611.07372, 2611.06372, 2611.07920), and mechanistic/interpretability
notes (2611.05344, 2610.39302, 2610.41014) have no action for the 95M→275M→1B
training queue. `2610.37741` (0.1% pretraining poisoning → 92% backdoor success)
is a **data-security** note, not a quality lever: it argues for provenance checks
on the refetched corpus, which we already need for the incident record.

## What this does not change

- The E1 dense patch result stands (+1.806 BPB at matched bytes, 196,608 B window,
  same eval offset). Dory is a 4B loop data point; the 95M lane stays dense.
  Soft-MTP is an objective, not an architecture rescue. The E1 test was dense-only —
  the loop/hybrid needs its own bytes-vs-patch test before the input unit is frozen.
- No GPU work was run for this scan; the card is held by the optimizer-VRAM experiment.
- Claims here are abstracts, not reproductions. Any arm that lands needs its own
  config, window, commit and 3-seed gate.
