# 2026-11-06 — VRAM ledger: what this 16 GB card can train

**Task.** The definitive VRAM ledger for one 16 GB card (15.56 GiB usable, hard
ceiling 15 GiB), for four targets: 100M (today), 400M, 1B dense and 1B sparse
(~40M active). Every parameter count comes from the real `ModelConfig` tree; the
activation model is arithmetic anchored to the one measured `b8/s1024` peak
(**7.2 GiB**, `port/src/dormouse/model/act_mem.py:4`). The tool is
`scripts/vram_ledger.py`; the gates are `port/tests/test_vram_ledger.py`.
**No GPU command was run** — the card belongs to `m400b`. The driver
(`--measure`) is written and refuses a busy card; the owner runs it later.

The script reproduces every table below:

```
python scripts/vram_ledger.py
python scripts/vram_ledger.py --measure --target spa1b --batch 6 --seq 1024 --muon-dtype bf16
```

---

## 0. TL;DR

| target | params | active MAC/B | max b @ s1024 | max b @ s8192 | B/s @ 18 % MFU | verdict |
|---|---:|---:|---:|---:|---:|---|
| 100M flagship (today) | 102,678,943 | 17,255,552 | **24** | **3** | 79.5 k | fits today, b8 is comfortable (6.50 GiB) |
| 400M MoE | 401,880,095 | 36,110,336 | **11** | **1** | 38.0 k | fits at b8 (12.50 GiB) |
| 1B dense | 1,012,146,774 | 1,004,957,013 | **1** | **0** | 1.36 k | only b1, at 14.01 GiB; b2 OOMs |
| 1B sparse (~40M active) | 1,017,151,084 | 40,349,376 | **0** | **0** | 34.0 k | b1 alone is 15.24 GiB, over by 0.24; needs one lever |

**The largest model that trains here without levers is the 1B dense at batch 1.**
That is not a usable training point (1.36 kB/s = 117 MB/day). The sparse 1B is
0.24 GiB over at b1; the FREE `muon_state_dtype=bf16` alone (1.89 GiB) makes it
fit and reaches b6, and full checkpointing + bf16 state reaches b49. The single
highest-value lever is **full activation checkpointing**: **3.41 GiB** at the
100M, **11.80 GiB** at the 1B dense, **2.26 GiB** at the sparse 1B, at +33 %
compute and bit-identical values. On the sparse 1B the checkpoint cannot touch
the 2.54 GiB bank cache, so there the biggest single object is the bank and the
ternary option (2.31 GiB) beats the checkpoint by 0.05 GiB.

---

## 1. Hardware constants

| quantity | value | source |
|---|---:|---|
| card capacity | 15.56 GiB usable (16 GB decimal) | brief |
| hard ceiling used here | 15.0 GiB | brief (leaves 0.56 GiB) |
| bf16 tensor peak | 45.7 TFLOP/s | [M] `docs/research/2026-11-05-small-model-mfu.md` §0.3 (purchase report's measured 5060 Ti ceiling) |
| HBM bandwidth | ~1.9 TB/s | [M] same memo's machine class |
| DDR4 | 68 GB/s (4 channels, measured brief) | brief |
| PCIe 3.0 x8 | 6.5 GB/s | brief |
| CUDA context + allocator floor | 1.0 GiB | [M] `act_mem.py:4-5` |

The decision table uses **18 % MFU** as requested. The same memo's measured
live m400b is ~10 % (26.8 kB/s against this ledger's 38.0 kB/s at 18 %), so the
B/s column is an upper bound at a fixed MFU, not a forecast.

---

## 2. Method: where every number comes from

### 2.1 Parameters — the module tree, not a hand sum

`param_ledger(cfg)` instantiates `DormouseModel(cfg, device="meta")` and sums
`named_parameters()`. The optimizer split is the optimizer's own rule
(`optim.is_muon_param`): 2-D, both dims > 1, not a hashed-memory table and not
embeddings/head/controller/router → Muon; everything else → AdamW. This is the
same tree `dormouse info` prints, so the gate is exact:

```
flagship_base  102,678,943  == `dormouse info flagship_base`
m400b          199,275,551  == `dormouse info m400b`
```

### 2.2 Optimizer state — the codec, not a guess

| tensor | bytes | where |
|---|---:|---|
| parameters | `4 · P` | fp32 storage (`model.py`: mixed precision casts at arm boundaries; the parameters are fp32) |
| gradients | `4 · P` | one per parameter |
| Muon momentum | `2 · P_muon` (bf16) or `4 · P_muon` (fp32) | `optim.Muon` allocates ONE buffer (`momentum`) per parameter |
| AdamW `exp_avg` + `exp_avg_sq` | `8 · P_adamw` | `_foreach_adamw_step` creates both with `torch.zeros_like(p)` |
| fp32 master | **0** | there is no separate master; the parameter itself is fp32 |
| Muon NS workspace | `6 · min(P_muon, chunk) · 4` | `muon_chunk_elems`' docstring: ~6 fp32 temporaries over the chunk |

The NS workspace is an **optimizer-stage** peak, not a forward/backward one.
The ledger takes `train_peak = max(fwd/bwd peak, optimizer peak)`, which is why
a 1B model's `parameters + gradients + state + context` (≈10.3 GiB) is the
floor and the NS 1.6 GiB competes with the activations, never adds to them.

For the four targets (Muon fp32, the baseline):

| target | P | P_muon | P_adamw | params+grads | Muon | AdamW | NS (chunk) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100M | 102,678,943 | 51,686,276 | 50,992,667 | 0.765 GiB | 0.193 | 0.380 | 0.750 (1<<25) |
| 400M | 401,880,095 | 348,877,828 | 53,002,267 | 2.994 GiB | 1.300 | 0.395 | 1.500 (1<<26) |
| 1B dense | 1,012,146,774 | 1,006,639,872 | 5,506,902 | 7.542 GiB | 3.750 | 0.041 | 1.500 (1<<26) |
| 1B sparse | 1,017,151,084 | 1,015,301,668 | 1,849,416 | 7.578 GiB | 3.782 | 0.014 | 1.500 (1<<26) |

### 2.3 Activation memory — named components with measured multiplicities

The model is not a standard transformer, so the standard `34·b·s·d` formula
would be wrong. The ledger uses the autograd save census of *this* model:
`saved_tensors_hooks` on CPU at `b1/s128` and `b2/s256`, grouped by shape and
normalized per iteration. The body has 3 loop iterations; the census found the
save multiplicities below (each is the number of copies of `b·s·d` or the
corresponding shape kept for the backward):

| component | elements per iteration | dtype | multiplicity source |
|---|---:|---|---|
| residual stream | `4 · b·s·d` | fp32 | census: 4 `[b,s,d]` copies |
| norms / normalized reads | `4 · b·s·d` | fp32 | census |
| KDA boundary state | `8 · b·s·d` | fp32 | census: 8 `[b,h,s,hd]`, `h·hd = d` |
| attention projections q/k/v/g | `6 · b·s·d` | bf16 | census |
| widened projection reads | `5 · b·s·2d` | fp32 | census |
| CSA fine scores | `b·h·s·chunk` | fp32 | `csa_chunk = 8` |
| CSA selected-block gather | `b·s·topk·hd` | fp32 | `csa_topk = 32` |
| mHC lanes | `b·s·K·d` | fp32 | `mhc_streams = 2` |
| branch writes | `6 · b·s·d` | bf16 | census |
| widened branch writes | `b·s·2d` | bf16 | census |
| Engram read | `b·s·d` | bf16 | one read/iteration |
| Engram slots | `b·s·orders` | int64 | 3 hashed rows/token |
| expert gate/up latents | `2 · b·s·topk·r` | bf16 | 2 latents/assignment |
| expert write rows | `b·s·topk·f` | bf16 | one write/assignment |
| shared expert | `b·s·(2·shared_rank + shared_rank)` | bf16 | gate+up+s |
| shared down input | `b·s·f` | bf16 | one per token |
| expert bank cache | `E·(d·r + r·f)` | bf16 | **once**, not per iteration |
| logits | `b·s·V` | fp32 | final + one per iteration |
| CE masks/scalars | `b·s` | fp32 | per head term |

Per-window (outside the loop): embedding output, final norm/readout, the
`(max_iter+1)` logits. All per-iteration terms are multiplied by `max_iter`
(`3` for the 100M/400M, `1` for the 1B pair).

**The anchor.** At the measured recipe (`medium`, `max_iter=3`, `b8/s1024`,
`moe_capacity=0`, CSA+mHC+AttnRes+Engram+share_u — the `ab_actmem.py` common
set) the autograd saves sum to **4.042 GiB**, plus 0.115 GiB of module-owned
bank cache, 0.392 params, 0.392 grads, 0.202 Muon, 0.380 AdamW and 1.000
context = **6.524 GiB**, which is **−9.4 %** from the recorded **7.2 GiB**
(`act_mem.py:4`). The gap is consistent with the forward's un-saved transient
workspace (the largest single op temporary, absent from a `saved_tensors_hooks`
census) plus the census being one shape sample; it is inside the 10 % gate. The
full breakdown:

| reference component | GiB |
|---|---:|
| residual stream | 0.3750 |
| norms / normalized reads | 0.3750 |
| KDA boundary state | 0.7500 |
| attention projections | 0.2812 |
| widened projection reads | 0.9375 |
| CSA fine scores + gather | 0.1992 |
| mHC lanes | 0.1875 |
| branch writes + widened | 0.3750 |
| Engram read + slots | 0.0474 |
| expert latents/write/shared | 0.4130 |
| embedding + final readout + logits + CE | 0.1013 |
| **autograd-save total** | **4.0421** |
| expert bank cache (module-owned) | 0.1146 |
| parameters | 0.3923 |
| gradients | 0.3923 |
| Muon momentum (fp32) | 0.2023 |
| AdamW m+v | 0.3799 |
| CUDA context | 1.0000 |
| **train peak** | **6.5238** |

### 2.4 What "fits" means

`peak_bytes` sums parameters, gradients, optimizer state (on-card, after levers),
the bank cache, activations and the 1 GiB context for the forward/backward
branch, and `parameters + gradients + state + NS workspace + context` for the
optimizer branch; the train peak is the max. `max_batch` scans integer batches
until the peak crosses 15 GiB. The `1B sparse` baseline has no integer:
b1 = 15.235 GiB, over by 0.235 GiB.

---

## 3. Component ledgers at `b8 s1024`

The four rows are the same arithmetic as the script; `%` is of the fwd/bwd peak.

| component | 100M GiB | 400M GiB | 1B dense GiB | 1B sparse GiB |
|---|---:|---:|---:|---:|
| parameters (fp32) | 0.383 | 1.497 | 3.771 | 3.789 |
| gradients (fp32) | 0.383 | 1.497 | 3.771 | 3.789 |
| Muon momentum (fp32) | 0.193 | 1.300 | 3.750 | 3.782 |
| AdamW m+v | 0.380 | 0.395 | 0.041 | 0.014 |
| expert bank cache | 0.115 | 0.859 | 0 | 2.538 |
| activations | 4.042 | 5.947 | 13.408 | 2.581 |
| CUDA context | 1.000 | 1.000 | 1.000 | 1.000 |
| **fwd/bwd peak** | **6.495** (43 %) | **12.495** (83 %) | **25.740** (172 %) | **17.493** (117 %) |
| NS workspace (opt stage) | 0.750 | 1.500 | 1.500 | 1.500 |
| **optimizer-stage peak** | **3.087** | **7.189** | **13.832** | **13.874** |
| **train peak** | **6.495** | **12.495** | **25.740** | **17.493** |

The percentages in parentheses are `peak / 15 GiB`: the 100M has 8.5 GiB of
headroom at b8; the 400M has 2.5 GiB; both 1B models are already over at b8.
Note the 1B dense optimizer peak (13.83) is below its b8 activation peak
(25.74) but the *b1* train peak is 14.01 (the activation branch), so the model
does fit at batch 1 — the state is what makes b2 impossible (14.01 + 1.7).

### Max batch under the 15 GiB ceiling (no levers)

| target | s=1024 | s=8192 |
|---|---|---|
| 100M | **24** (14.579 GiB) | **3** (14.579 GiB) |
| 400M | **11** (14.725 GiB) | **1** (12.495 GiB) |
| 1B dense | **1** (14.008 GiB) | **0** — b1 is 25.740 GiB, over by 10.740 |
| 1B sparse | **0** — b1 is 15.235 GiB, over by 0.235 | **0** — b1 is 17.493 GiB, over by 2.493 |

At `s=8192` the `b·s` activation term grows 8×; the 100M still fits b3, the
400M only b1, and neither 1B. The 100M preset ships `max_seq_len=1024`; the
s=8192 rows assume the field is raised (the driver does that automatically),
and the parameter/activation arithmetic is unchanged by it.

---

## 4. Levers: GiB saved, cost, numerics

Each saving is a difference of two `peak_bytes` calls at `b8 s1024` — no
estimate. A negative/zero is printed as-is. Cost is a formula; the assumption is
named.

| lever | 100M | 400M | 1B dense | 1B sparse | speed cost | numerics |
|---|---:|---:|---:|---:|---|---|
| `iter_checkpoint` / `activation_recompute=full` | **3.407** | **5.095** | **11.798** | **2.261** | +1 forward/iteration: step = 3 fwd units → 4 = **+33.3 %** | exact recompute; CPU measures bit-identity, GPU gate 1e-6 relative |
| `activation_recompute=selective` (write half) | 1.082 | 1.509 | 3.790 | 0.698 | recomputes **81/87/99/94 %** of active MACs (the FFN is the write half) | same ops/values; 1e-6 gate |
| `act_ckpt` mxfp4 (4-bit saved tensors) | 3.357 | 4.937 | 11.081 | 2.137 | codec compute; CPU 2.5-3× the step, **GPU unmeasured** (`act_ckpt.py`); loss bit-identical, worst gradient move **2.4e-1 relative** | lossy backward only |
| expert bank ternary (2-bit) | 0.104 | 0.781 | 0 | **2.307** | same MACs/launches; 2-bit unpack + one fp32 scale/row | weights rounded to {-1,0,1}·scale: lossy; **not wired to the bank** (`factor_qat` is the TSCT seam) |
| `muon_state_dtype=bf16` | 0.096 | 0.650 | 1.875 | 1.891 | free (same launches, half the store bandwidth) | accumulation fp32, store rounds; 1e-2 relative gate |
| ZeRO-1, 2 cards (state only) | 0.286 | 0.847 | 1.896 | 1.898 | all-gather `P·4` over 6.5 GB/s = **0.063 / 0.247 / 0.623 / 0.626 s** per step (+61/+114/+10/+260 % of a step at 18 % MFU) | bit-identical (fp32 copies) |
| host offload of optimizer state | 0.572 | 1.695 | 3.791 | 3.796 | `2·state / 6.5 GB/s` = **0.189 / 0.560 / 1.252 / 1.254 s** (+183/+259/+21/+520 %); the 68 GB/s DDR4 side alone is 0.018/0.054/0.120/0.120 s | bit-identical (same values moved) |
| gradient accumulation | 0 | 0 | 0 | 0 | free VRAM: one micro-batch's activations; same total FLOPs | summation order changes, drift bounded by fp32 adds (not bit-identical) |
| `segment_len=512` | 0 | 0 | 0 | 0 | more segments = more launches (+2 % per extra, **assumption, unmeasured**) | exact; only the boundary state moves |

**What "int4" is here.** `ModelConfig.act_quant="int4"` is FAKE-quant: it
dequantizes in place and saves no bytes. The packing 4-bit path is
`act_ckpt`'s MXFP4 (`mxfp.bytes_per_elements_mxfp4` = 17 B per 32 elements,
0.53125 B/element), and that is the row above. It is not wired into the trainer;
the GPU cost is unmeasured.

**The host-offload plan (state only).** Keep the Muon momentum and the AdamW
`exp_avg`/`exp_avg_sq` in pinned host memory. After the backward, D2H the
optimizer state (or the shard), run the update on the host in the same fp32
arithmetic at 68 GB/s (0.018-0.120 s of host time for 0.57-3.80 GiB), then H2D
the updated parameters. The PCIe transfer is the wall, not the DDR4: `2·state /
6.5 GB/s` is 0.189 / 0.560 / 1.252 / 1.254 s per step for the four targets. It
is bit-identical because it moves the same fp32 values; it is only sensible
where the step is long enough (the 1B dense, +21 %) and wrong for the sparse 1B
(+520 %).

**Why selective is dominated.** It saves 0.7-3.8 GiB (less than full
checkpointing everywhere) while recomputing 81-99 % of the active MACs, because
the FFN/MoE *is* the write half and it is most of the arithmetic. There is no
target on this card where selective beats full checkpointing on the
memory-per-recompute trade.

**Why `segment_len` is zero here.** All four targets use KDA (linear attention,
no `s²` score matrix), so chunking the sequence cannot shrink an activation
that does not exist; the body's `b·s·d` saves are retained once per iteration
regardless of segment length. On the repo's `attn_kind="full"` arm the score
term is `b·h·s·s·bytes`; at `b8/s8192`, `n_heads=16`, bf16 that is **17.2 GiB**,
and `segment_len=1024` cuts it 8× (to 2.15 GiB). That is the only context where
this lever matters, and no target here uses it.

**Host offload is the wrong lever except at 1B dense.** The optimizer state is
3.8 GiB at 1B; moving it over x8 PCIe costs 1.25 s per step. That is 21 % of
the 1B dense step (6.0 s at 18 % MFU) but 520 % of the sparse step (0.241 s).
ZeRO-1 on two cards is the same shape: the all-gather is 0.62 s, 10 % of the
dense step and 260 % of the sparse step. The dense target is the only one where
the link is not the wall. At 400M the offload is 0.56 s on a 0.216 s step
(+259 %), so it is not a speed-neutral way to buy 1.7 GiB.

---

## 5. Decision table

`s=1024`, ceiling 15 GiB, MFU 18 % of 45.7 TFLOP/s. `maxb` is the largest
integer batch that fits; `GiB` is the train peak at that batch; `B/s` is
independent of batch (same FLOPs/byte).

```
target                 levers                         maxb     GiB       B/s  verdict
100M flagship_base     none                             24  14.579    79,453  ACHIEVABLE at b8
100M flagship_base     muon=bf16                        25  14.989    79,453  ACHIEVABLE at b8
100M flagship_base     ckpt=full                       176  15.000    79,453  ACHIEVABLE at b8
100M flagship_base     ckpt=full+muon=bf16             177  14.975    79,453  ACHIEVABLE at b8
100M flagship_base     ckpt=full+muon=bf16+act=mxfp4   385  14.991    79,453  ACHIEVABLE at b8
100M flagship_base     ckpt=full+muon=bf16+offload-state   184  14.997    79,453  ACHIEVABLE at b8
100M flagship_base     ckpt=full+muon=bf16+zero2       180  14.950    79,453  ACHIEVABLE at b8
100M flagship_base     ckpt=full+muon=bf16+bank=ternary  178  14.942    79,453  ACHIEVABLE at b8
400M MoE               none                             11  14.725    37,967  ACHIEVABLE at b8
400M MoE               muon=bf16                        12  14.819    37,967  ACHIEVABLE at b8
400M MoE               ckpt=full                        79  14.958    37,967  ACHIEVABLE at b8
400M MoE               ckpt=full+muon=bf16              85  14.947    37,967  ACHIEVABLE at b8
400M MoE               ckpt=full+muon=bf16+act=mxfp4   185  14.993    37,967  ACHIEVABLE at b8
400M MoE               ckpt=full+muon=bf16+offload-state   95  14.966    37,967  ACHIEVABLE at b8
400M MoE               ckpt=full+muon=bf16+zero2        90  14.956    37,967  ACHIEVABLE at b8
400M MoE               ckpt=full+muon=bf16+bank=ternary   92  14.911    37,967  ACHIEVABLE at b8
1B dense               none                              1  14.008     1,364  ACHIEVABLE only at b1
1B dense               muon=bf16                         2  13.809     1,364  ACHIEVABLE only at b2
1B dense               ckpt=full                        13  14.947     1,364  ACHIEVABLE at b8
1B dense               ckpt=full+muon=bf16              22  14.883     1,364  ACHIEVABLE at b8
1B dense               ckpt=full+muon=bf16+act=mxfp4    77  14.971     1,364  ACHIEVABLE at b8
1B dense               ckpt=full+muon=bf16+offload-state   32  14.979     1,364  ACHIEVABLE at b8
1B dense               ckpt=full+muon=bf16+zero2        27  14.931     1,364  ACHIEVABLE at b8
1B sparse (40M active) none                              0  15.235    33,976  IMPOSSIBLE: b1 = 15.24 GiB, over by 0.24
1B sparse (40M active) muon=bf16                         6  14.957    33,976  ACHIEVABLE only at b6
1B sparse (40M active) ckpt=full                         2  14.993    33,976  ACHIEVABLE only at b2
1B sparse (40M active) ckpt=full+muon=bf16              49  14.983    33,976  ACHIEVABLE at b8
1B sparse (40M active) ckpt=full+muon=bf16+act=mxfp4   171  14.996    33,976  ACHIEVABLE at b8
1B sparse (40M active) ckpt=full+muon=bf16+offload-state   96  14.960    33,976  ACHIEVABLE at b8
1B sparse (40M active) ckpt=full+muon=bf16+zero2        73  14.992    33,976  ACHIEVABLE at b8
1B sparse (40M active) ckpt=full+muon=bf16+bank=ternary  107  14.999    33,976  ACHIEVABLE at b8
```

**Reading it.** A row ending `ACHIEVABLE at b8` means b8 fits and the batch can
grow further (the number in `maxb` is the largest). `ACHIEVABLE only at bN`
means the lever changes the verdict but the model still cannot use b8. The one
`IMPOSSIBLE` baseline is the sparse 1B: at b1 its 3.79 params + 3.79 grads +
3.78 Muon (fp32) + 0.01 AdamW + 2.54 bank + 1.00 context + 0.32 activations =
**15.24 GiB**, 0.24 over. The FREE bf16 state (`muon=bf16`, −1.89) alone makes
b1 fit and reaches b6; checkpointing alone reaches b2 (bank + state remain);
checkpointing + bf16 state reaches b49; adding the ternary bank reaches b107.
`muon=bf16` is the cheapest enabler for the sparse model; checkpointing is the
bigger lever for the dense 1B (b1 → b13).

The 100M/400M max batches of 176/79 under checkpointing are real arithmetic but
not a recommendation: at 18 % MFU the step time at b176 is 2.27 s and the
optimizer state, not the activations, dominates. The useful point is that the
activation wall is gone.

---

## 6. Sparse-1B: bank bytes, active bytes, FLOPs/byte

**Configuration** (constructed in the ledger; not a tree preset): `d=2048`,
`f=5461`, 172 experts, `rank=1055`, top-2, `share_u="plane"`, one iteration.
Stored **1,017,151,084** params; **37,362,034** active params/token, **40.35 M
active MAC/B** → **242.1 MFLOP/B** for training (6×).

| quantity | bf16 | ternary (2-bit) |
|---|---:|---:|
| routed expert weights (raw) | 1.846 GiB | — |
| expanded bank cache `[E,d,r]+[E,r,f]` | **2.538 GiB** | **0.231 GiB** |
| per-token active bytes (whole active path) | **71.3 MiB** | **52.0 MiB** (only the 11.53 M routed weights shrink; attention/embeddings stay bf16) |
| bank storage vs bf16 | 1× | **10.99× smaller** (9.5× entropy bound + scales) |

The expanded bank is the resident object (`E·(d·r + r·f)` bf16): 172 ×
(2048·1055 + 1055·5461) = 1.363 G elements × 2 B = 2.538 GiB. The raw routed
weights are 991.3 M × 2 B = 1.846 GiB; the extra 0.69 GiB is the shared plane
materialized per expert. The ternary line is `ceil(991.3 M / 4) + 172·1055·4 B`
= 247.9 MB + 0.69 MB = **0.231 GiB** — the repo's own 2-bit rule
(`ternary.py`: 4 weights/byte, one fp32 scale per row).

**Does 242 MFLOP/B fit at 62 kB/s?** 62,000 B/s × 242.1 MFLOP/B = **15.01
TFLOP/s = 32.8 % of 45.7 TFLOP/s**, i.e. the task's 31 % MFU. Compute alone is
feasible (the card can do it). The question is the memory side:

* **Weights once per step (the real batched path):** 2.538 GiB read once per
  step. At 18 % MFU a b8/s1024 step is 0.241 s, so the bank sees **11.3 GB/s**
  — 0.6 % of the 1.9 TB/s HBM. It fits with three orders of magnitude to spare,
  because every expert sees ~95 tokens in the batch (8·1024·2/172) and the
  weights are reused.
* **Per-token streaming (batch 1, no reuse):** 71.3 MiB/token × 62 k tokens/s =
  **4.6 TB/s**, 2.4× the card; ternary bank 3.38 TB/s, still over. Arithmetic
  intensity is `6 FLOPs / 2 B = 3 FLOP/B` against a machine balance of
  **24.1 FLOP/B**, so a one-token-at-a-time MoE is memory-bound by ~8×. This is
  the number that kills an inference-style reading of the model; it is not how
  the trainer runs it.

At b8/s1024 the reuse raises the intensity to `3 × 95 = 286 FLOP/B`, ten times
the machine balance, so the sparse 1B is compute-bound at the batch a trainer
actually uses. At batch 1 it is memory-bound and would run at ~12 % of peak even
with a perfect kernel.

The verdict: **the FLOPs/byte fit at 62 kB/s (32.8 % MFU is reachable on
paper), and the bank fits in VRAM (2.54 GiB bf16); the sparse 1B is blocked by
optimizer state, not by the bank or the arithmetic.** The baseline b1 is
15.235 GiB (over by 0.235); `muon=bf16` alone (−1.891) fits b1 and reaches b6;
checkpointing (−2.261) reaches b2; checkpointing + bf16 state (−4.152) reaches
b49; ternary bank (−2.307) reaches b7 alone and b107 with checkpointing + bf16
state.

---

## 7. Measured vs computed

| number | status | source / method |
|---|---|---|
| 7.2 GiB `b8/s1024` peak | **[M]** measured | `port/src/dormouse/model/act_mem.py:4`; `port/scripts/ab_actmem.py:60` |
| parameter counts / groups | **[M]** measured | `DormouseModel(..., device="meta")` + `dormouse info`; gate |
| Muon/AdamW routing | **[M]** code | `optim.is_muon_param`, `optim.optimizer_bytes` |
| 45.7 TFLOP/s bf16 | **[M]** measured | purchase report via `docs/research/2026-11-05-small-model-mfu.md` §0.3 |
| 14.2 GiB m400b b8 peak | **[M]** brief via the memo | same memo §0.1; it is the run's nvidia-smi/reserved figure with the unfused-residual arm, not `max_memory_allocated` — the ledger's m400b config gives 9.71 GiB allocated, so this is not the like-for-like anchor |
| mxfp4 17 B/32 elements | **[M]** code | `mxfp.MX_BLOCK/MX_BYTES_PER_BLOCK` |
| mxfp4 gradient drift 2.4e-1 | **[M]** measured (CPU tiny) | `act_ckpt.py`'s error table |
| ternary 4 weights/byte | **[M]** code | `ternary.py`'s worked example |
| activation component multiplicities | **[M]** census (CPU) | `saved_tensors_hooks`, `b1/s128` + `b2/s256`; this is the calibrated part (total at −9.4 % of the anchor) |
| every GiB / max batch / B/s | **[E]** arithmetic | `scripts/vram_ledger.py`; each row prints its formula |
| GPU step time per lever | **unmeasured** | no GPU run; the driver is written |

The m400b cross-check is deliberately not a second gate: the brief's 14.2 GiB is a
reserved/`nvidia-smi` number on the unfused-residual arm; the ledger's allocated
model at that config is 9.71 GiB. The only like-for-like anchor in the tree is
`act_mem.py:4`'s 7.2 GiB, which is what the formula is gated to. A GPU
`--measure --target m400 --batch 8` row would settle the difference.

The one soft spot is the *split* of the activation memory between components:
the total is pinned to within 9.4 % of the measured peak, but two components
with the same total can trade GiB. The multipliers are the census integers, so
the split is checkable, not fitted. A GPU `--measure` run of the checkpoint arm
would tighten it; the off/iter peak comparison is exactly what `ab_actmem.py`
was written for and never ran.

---

## 8. The single highest-value lever

**Full activation checkpointing (`iter_checkpoint`/`activation_recompute=full`).**

| target | GiB saved at b8/s1024 | new max batch | price |
|---|---:|---:|---|
| 100M | 3.407 | b8 → b176 | +33 % compute (one extra forward per iteration) |
| 400M | 5.095 | b8 → b79 | +33 % |
| 1B dense | **11.798** | b1 → b13 | +33 % |
| 1B sparse | 2.261 | b1 impossible → b2 (b49 with bf16 state) | +33 % |

It is the only lever that is exact (bit-identical on CPU; the recompute is the
same ops on the same inputs), removes the activation wall at every target, and
composes with every other lever. It beats the lossy mxfp4 arm at three of the
four targets (3.407 vs 3.357 at 100M, 5.095 vs 4.937 at 400M, 11.798 vs 11.081
at 1B dense); on the sparse 1B mxfp4 is 2.137 vs 2.261, and the 2.54 GiB bank
cache is outside the autograd graph, so neither can touch it. The second lever,
only if gradient quality can be traded, is mxfp4; the third is
`muon_state_dtype=bf16` (1.89 GiB at 1B, free, and the sparse model's cheapest
enabler).

**The model that actually trains today is the 100M at b8 (6.50 GiB, 8.5 GiB of
headroom) or b24 at the ceiling; the 400M at b8 (12.50 GiB) is the realistic
next step; the 1B dense needs checkpointing to be a training point at all, and
the sparse 1B is 0.24 GiB over at b1 and needs bf16 state first.**

---

## 9. Gates and how to run the driver

`port/tests/test_vram_ledger.py` (8 tests, CPU, ~8.5 s):

1. `flagship_base` 102,678,943 and `m400b` 199,275,551 equal the instantiated
   tree AND the `dormouse info` banner.
2. the activation formula is within 10 % of 7.2 GiB and the citation
   `act_mem.py:4` still says it;
3. every lever that claims a saving equals the independent `peak_bytes`
   difference (a removed formula fails); grad_accum and segment_len are pinned
   at zero on the KDA arm;
4. `max_batch` is the largest fitting integer (b fits, b+1 does not);
5. the sparse bank bf16/ternary arithmetic and the 242 MFLOP/B target;
6. the decision table marks the impossible row with its over-by arithmetic;
7. the driver refuses a busy card before building a model (preflight
   monkeypatch), and `_lane` parses a busy `nvidia-smi` row correctly.

```
cd port && CUDA_VISIBLE_DEVICES="" /home/sehaxe/dormouse/port/.venv/bin/python -m pytest tests/test_vram_ledger.py -q
ruff check scripts/vram_ledger.py
```

On the free card, one real row (the driver is the only GPU path; it calls
`_lane.preflight` first and prints the SKIP banner if any compute process holds
the card):

```
python scripts/vram_ledger.py --measure --target dense1b --batch 1 --seq 1024
python scripts/vram_ledger.py --measure --target flagship --batch 8 --seq 1024 --recompute full
python scripts/vram_ledger.py --measure --target flagship --batch 8 --seq 1024 --recompute full --muon-dtype bf16
```

The driver measures `torch.cuda.max_memory_allocated` and the synchronized step
time, prints JSON (`peak_gib`, `step_s`, `bytes_per_s`, `mfu`), and needs no
other flag. It does not wire mxfp4 (the codec is not in the trainer), does not
shard across two cards, and does not move optimizer state to the host: those
three are arithmetic here, by design, until a lane implements them.
