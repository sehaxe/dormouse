# MoE "to ideal" — the frontier stack, transcribed to our scale

Two sources, both read at the paper: **Kimi K3** (arXiv:2607.24653, §2.3 Stable
LatentMoE, §2.5 Per-Head Muon) and **DeepSeek-V4** (arXiv:2606.19348, §2.1
DeepSeekMoE adjustments, §2.2 mHC, §3.1 EP). Everything below is a leg one of
those two reports states explicitly, with what it fixes there and what it would
cost here.

## What we already have (measured, in the tree)

| leg | source | status |
|---|---|---|
| fine-grained routed experts + shared expert | DeepSeekMoE (V3→V4 unchanged) | `ffn_form="bank"`, `shared_rank` |
| SiTU-GLU experts | K3 §2.3.2 | `expert_act="situ"`, β1=4, β2=25, tests green |
| Quantile Balancing | K3 §2.3.3 | `moe_qb`, commit `3978c7b`, loads within ±5.2% |
| Per-Head Muon | K3 §2.5 | `muon_per_head`, commit `6826bf6`, spread 2.12% vs 223.2% |
| latent (shared) expert projection | K3 latent MoE | `moe_shared_compute`, d→2·rank once per token |
| fused expert kernel (Triton) | DSV4 §3.1 | `moe_kernel="triton"`, wide-weight path COUNTED |

## The legs we do NOT have

| # | leg | source & why there | cost here | gate |
|---|---|---|---|---|
| 1 | **RMSNorm before the up-projection** | K3 §2.3: without it the routed branch's accumulated norm feeds the up-projection and the activations grow across 93 layers | ~10 lines in the bank expert | loss parity with norm OFF at step 0 (it is a re-parameterisation), then divergence-rate at 3 seeds |
| 2 | **`sqrt(Softplus)` affinity** | DSV4 §2.1: V4 replaced sigmoid with sqrt(softplus) for the routing score | ~1 line + clamp; softplus ≥ 0 so sqrt halves the dynamic range | BPB at 120 MB, 3 seeds |
| 3 | **hash routing on the first block(s)** | DSV4 §2.1: V4 replaced the dense FFN of the first layers with hash-routed MoE — `token_id → expert` by a fixed hash, no router | ~50 lines: a deterministic id→expert map, no bias/no QB on those layers | removes the router entirely on those layers; gate = equal BPB, *faster* step |
| 4 | **sequence-level balance loss** (on top of the aux-free bias) | DSV4 §2.1: "a slight sequence-level balance loss to prevent extreme imbalance inside a single sequence" | ~30 lines; a per-sequence load term | load spread within a batch sequence, and BPB 3 seeds |
| 5 | **hybrid Newton–Schulz schedule** | DSV4 §2.2: 8 iterations at (3.4445, −4.7750, 2.0315) then 2 at (2, −1.5, 0.5) to park singular values exactly at 1 | a coefficient table in `ns_kernels` | orthogonalization error after 10 iterations vs the current `ns_steps` |
| 6 | **MTP head** | DSV4 §2.1 (unchanged from V3) | a second prediction head + loss; we already have `dspark` for the same "predict ahead" idea | do not build twice — decide dspark vs MTP first |

## The one number that decides whether ANY of this pays

Capacity is only worth paying for when it is fed:

| corpus | 299M total / 100M active |
|---|---|
| 2.64 GB (today) | 8.8 B/param — dead capacity, MoE measured *worse* per byte |
| 8 GB | 26.8 B/param — the leg becomes measurable |
| 20 GB | 66.9 B/param — capacity has room |

And the cost of padding, at the same achieved throughput (49.7 TFLOP/s measured
on the dense control):

| `moe_capacity` | rows run vs assigned | predicted step | vs dense b12 |
|---|---|---|---|
| 2.0 (what we measured) | 2.52× | 321 ms | 2.16× slower |
| 1.25 (with QB on) | 1.25× | 201 ms | 1.35× |
| dropless (batched) | 1.0× | 177 ms | 1.19× |

Rule (owner calibration, memory #52): 1.5–2 % relative quality per 10 % speed.
At 1.19× the arm must return **+3 % quality (≈0.04 BPB)**; at 1.35× it must
return **+6 % (≈0.07 BPB)**. That is the bar, and it is measured at 3 seeds.

## Order of landing (each behind a flag, off by default)

1. legs 1 + 2 + 4 — small, independent, one commit each, CPU gates only;
2. re-run the b24/b16 MoE bench (`scripts/bench_moe_package.py`) and record the
   peak that the buffer fixes bought (−468 MiB/layer, ≈−5.5 GiB at b24);
3. leg 3 (hash routing) — it changes the block layout, so it lands alone;
4. leg 5 — optimizer-only, gate against the current NS on the shared test;
5. leg 6 — only after dspark vs MTP is decided by a measurement, not by taste.

Legs 1–5 are cheap enough to land in a day; none of them is allowed to become
default-on before the A/B above, and none of them is worth a GPU-hour until the
corpus is ≥8 GB.
