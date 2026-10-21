# The test ledger — every measurement this lane has taken, with its verdict

**Date** 2026-10-15 → 2026-10-21 · **lane** Python port · **card** RTX 5060 Ti 16 GB
**Rule** AGENTS §1.2: 3 seeds per arm, a win must beat the control's own spread,
a tie deletes the mechanism (or keeps it only with a named, scheduled regime).
**A BPB is only comparable within one window** — `eval_batches × batch × seq_len`.

This file indexes the measurements so a reader can find one without grepping
`~/logs`. Each row names its evidence directory.

## 1. A/B verdicts — WINS

| mechanism | control | arm | Δ | window | evidence |
|---|---|---|---|---|---|
| **ColRow** (Muon normalization) | 3.0542 | **3.0113** | **−0.0428** (8.7× spread) | 81 920 B | `~/logs/ab_colrow` |
| **attnres** (attention residuals) | 3.0750 | **3.0118** | **−0.0632** (7.5× spread) | 81 920 B | `~/logs/battery` |
| **ByteFlow** (learned patches, small) | 3.0447 | **2.3581** | **−0.6865** (22× spread) | 40 960 B | `~/logs/ab_byteflow` |
| **ByteFlow** (medium, driver fixed) | 3.4185 | **2.9337** | **−0.4848** | 40 960 B | `~/logs/bf_medium3` |
| **ByteFlow** vs capacity-matched wide control | 3.3613 (d=1792, 137.2 M) | 2.9337 | **−0.4276** | 40 960 B | `~/logs/capmatch` |
| **rank 32** (speed trade, not quality) | 3.0750 @ rank 64 | 3.1024 (+0.0274) | **+23% tok/s, −27.7% params** | 81 920 B | `~/logs/battery` |
| **wide** = rank 32 spent on width (d=1920) | 3.0750 (91.5 M) | **2.999** (91.7 M) | **−0.076** (n=1, s2 on trajectory) | 81 920 B | `~/logs/wide` |

## 2. A/B verdicts — TIES (kept only with a scheduled regime)

| mechanism | control | arm | Δ | why it may stay |
|---|---|---|---|---|
| **MSA @2048** | 3.0744 | 3.0702 (spread 0.0045 vs 0.0084) | −0.0042 | long-context arm; halved spread |
| **dspark** (pretrain aux) | 3.0750 | 3.0733 | −0.0017 | spec decode draft head |
| **LOOM** (depth 4) | 3.0559 | 3.0521 | −0.0038, worse on `fin` | nothing scheduled → delete |
| **act_quant** int8/int4/fp4 | 2.2027 | 2.2022 / 2.2071 / 2.2042 | all inside spread | release rungs (QAT) |
| **KDA disable_recompute** | 621 ms | 621 ms | 0.2%, +960 MiB | default off, all record numbers stay |
| **ByteFlow K=64** (8 B/patch) | 2.9410 | 2.9431 | +0.0021 | halves loop positions for free |
| **ByteFlow SDPA rewrite** | — | — | parity **1.19e-7** | pure speed, no quality change |

## 3. A/B verdicts — LOSSES

| mechanism | control | arm | Δ | evidence |
|---|---|---|---|---|
| **NorMuon** | 2.1713 | 2.2435 | **+0.0722** | `~/logs/ab_muon_variants` |
| **SF-NorMuon** | 2.1713 | 2.2138 | +0.0426 | same |
| **SF-Muon** | 2.1713 | 2.2257 / **4.0573** / 2.2065 | +0.0448 as a **two-seed mean**; one seed diverged 8× | same — **re-run needed** |
| **SIGReg** as a standalone aux | — | — | **+0.1480** | `~/logs/ab_sigreg` |
| **rank 16** | 3.0750 | 3.2106 | +0.1356 | `~/logs/battery` |
| **rank 8** (triton) | — | crash (tile) | — | same |
| **rank 8 + topk 4** (naive) | 3.0750 | 3.9057 | +0.83 | `~/logs/tsct` |
| **E=512 at rank 32** | 3.0750 | 3.1075 | +0.0325 | `~/logs/e512` |
| **r16 + topk 4** (FLOP-matched composition) | 3.0750 | 3.2017 | +0.1267 | `~/logs/tsct` |
| **ByteFlow `e_layers` 6→3** | 2.9410 | 3.1956 | **+0.2546** | `~/logs/bf_opt` |
| **ns_steps 5→3** | — | — | +0.0058 | `~/logs/ab_ns` |
| **ternary at 9.2 M** | — | — | +0.04 BPB (1.3%), 0 speed, 0 VRAM | `~/logs/ab_ternary` |
| **torch.compile dynamic** | — | — | **0.20×** (5× slower; dynamo recompiles on `_banks`) | `~/logs` |

## 4. Mechanism measurements (not A/B)

| mechanism | measurement | verdict |
|---|---|---|
| **Triton NS** | Muon stage 68.1 → 18.6 ms (52 dispatches → 1) | in the recipe |
| **opt_graph** (CUDA graph over the optimizer step) | −12 ms/step (3239 → 2 launches), bit-identical pin | in the recipe (off in ternary arms until the `why` crash was fixed) |
| **bf16 Muon state** | −1.23 GiB VRAM, first step bit-exact | in the recipe |
| **fused MoE backward** | 3 kernels + 358 MB → 1 kernel −179 MB | in the recipe |
| **MoE seam bf16** | gather 12·R·d → 4·R·d = **−384 MiB/pass**, `max|Δlogit| = 0.0` | in the recipe |
| **SonicMoE gather** | forward **−35.3%** bytes, **whole pass +21.8%** (no adjoint → recompute) | off; needs the adjoint |
| **fp32 Triton tiles** | shared memory 196 608 B → **65 536 B** (ceiling 101 376) | makes the fp32 (ByteFlow) route legal |
| **ByteFlow bf16** | parity **1.43·bf16-eps**; chunker selection agreement 96.9-100% **with** the fp32 island, **0.00%** without | bf16 path kept, island mandatory |
| **ternary quantize-once** | 245.4 M → 122.7 M element-passes/step (2.0×), second forward 3.75× | landed |
| **ternary at 138 M** | 11.5k → **6.6k tok/s** (1.75× slower; was 2.4× before the cache) | price of the release form |
| **mHC paper-first init** | logits **bit-identical** at init; `H_res` doubly stochastic 1.19e-7 | port done, A/B not run |
| **rank-2 dictionary** | expert subspaces differ by **1.42** relative (plane mode: 0.0) | port done, A/B not run |
| **share_u arithmetic** | bank 79.2 M (r64) → **2.32 M** (r2) = 34× | port done, A/B not run |
| **energy head (EBT-lite)** | calibration **1.0000** vs **0.0059** random-init | port done, A/B + rerank not run |
| **halt head** | stub stop at iteration 1 → loop runs exactly 1 | port done, A/B not run |
| **adaptive topk** | padded vs naive `max_rel = 0.0` on mixed widths | port done, A/B not run |
| **MTP** | k=0 bit-identical; shift fixture; no label leak | port done, A/B not run |

## 5. Model runs (recorded, not A/B)

| run | steps | data | BPB | window | note |
|---|---|---|---|---|---|
| medium (fb) | 40 000 | 0.61 GB | 2.541 | 24 576 B | first honest medium |
| b1first | 50 000 | 0.20 GB | 2.557 | 32 768 B | 1.07 B |
| medbig2 | 50 000 | 1.23 GB | 2.374 | 196 608 B | **invalid** (bank cache froze) |
| mix1 | 60 000 | 1.47 GB | 1.936 | 196 608 B | web+wiki+ru+math+code |
| **mix1 → 170 000** | 170 000 | **4.18 GB** | **1.883** (fin 1.695) | 196 608 B | 30.2 B/param, 37.7% Chinchilla |
| mix_v2 (seq 2048, MSA+attnres) | 12 000 | 98 MB | 2.082 | 40 960 B | bench macro 0.2727 |
| sw1 (wide, E=256, d=768) | 35 000 | 0.57 GB | 2.209 | 131 072 B | width experiment |
| sft13 | 4 000 | 49 484 pairs | 1.296 | 32 768 B | holdout CE 0.898, `first=` 39.5% |

## 6. Downstream (zero-shot, n as noted, random = 25%)

| checkpoint | ARC-Easy | HellaSwag | synth-math | macro |
|---|---|---|---|---|
| mix1 @60k (n=1000) | 27.2 | **30.5** | 24.0 | 0.272 |
| mix1 @80k (n=1000) | 29.5 | 26.3 | 21.4 | 0.257 |
| mix1 @170k (n=1000) | 29.5 | 26.3 | 21.4 | 0.257 |
| mix_v2 (n=500) | 27.0 | **30.8** | 24.0 | 0.273 |

Reference points at the same task family: SmolLM2-135M HellaSwag 0.430 /
ARC 0.585 (600 B tokens); GPT-2 small HellaSwag 0.314, ARC 0.397 (~40 GB);
Pythia-160M 0.303 / 0.364 (300 B tokens). **Our protocol is byte-level
`acc_mean`, theirs is token-level `acc_norm` — the comparison is ordinal.**

## 7. What is measured on the CARD vs only on CPU

Everything in §4 was ported and gated on CPU; the following have **no GPU
number yet** because the card was held by the ladder: fp32 Triton tile timing,
MoE seam timing, SonicMoE gather timing, ByteFlow bf16 timing, MTP/mHC/halt/
topk/energy/rank-2 A/Bs, adaptive-depth Pareto, SF-Muon re-run.
