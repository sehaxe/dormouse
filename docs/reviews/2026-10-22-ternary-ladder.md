# The ternary ladder (2026-10-21/22) — the release recipe

Protocol: `medium` preset, `b8 s512`, **2000 steps**, `mix_big`, Engram in host RAM,
`opt_graph`, `ns_kind=triton`, dropless `moe_kernel=triton`, `bf16=true`,
eval window **81 920 B** (`20 x 8 x 512`), 3 seeds per arm.

**The comparison rule this document exists for.** An earlier attempt compared
ternary arms at 141,910,405 params against a control at 91,512,197 (the battery
ran with `use_engram=false`) and produced a "cost" of +0.060 BPB that was
actually the difference between two models. The control now runs **in the same
script, with the same flags**: every row below is 141.9M-142.2M params and
Engram ON. A comparison is valid only when the `params` in the first log line
match.

## Results

| arm | quantisation | geometry | BPB (3 seeds) | mean | spread | vs control |
|---|---|---|---|---|---|---|
| **bfctl** | none (bf16) | rank 64, d 1024, 141.9M | 3.085 / 3.074 / 3.081 | **3.0800** | 0.011 | — |
| **q64b** | factors + all dense ternary, attention bf16 | rank 64, d 1024, 141.9M | 3.126 / 3.140 / 3.144 | **3.1367** | 0.018 | **+0.057** |
| **q32w** | same | **rank 32, d 1920**, 142.2M | 3.063 / 3.067 / 3.097 | **3.0757** | 0.034 | **−0.004** (tie) |
| q16w | same | rank 16, d 2560, 146.0M | 3.223 (1 seed, stopped) | — | — | +0.143 |
| q64a4 | q32w + `act_quant=int4` | rank 32, d 1920 | running | — | — | — |

## What it says

1. **Ternary costs +0.057 BPB (+1.9%)** when the geometry is held fixed. That is
   the honest price of a 2-bit linear stack, measured on a 2000-step budget —
   where quantisation is most expensive, because the model has no time to adapt.
2. **Moving the rank into width pays for it and then some**: rank 32 at d 1920
   is **−0.061** against the same ternary at rank 64/d 1024, landing at
   **3.0757 vs the bf16 control's 3.0800** — a tie inside both spreads, at the
   same parameter count, with an 8x smaller release artifact.
3. **Rank 16 is a cliff, not a slope**: 3.223 (+0.143). The bf16 ladder measured
   +0.136 at the same rank, so this is a property of the rank, not of the
   quantiser. Two independent curves agree.
4. **The AdamW arm was not the thing to protect**: `dense_qat` on the KDA
   attention projections cost **+0.198** (the first, wrong ternary run). Keeping
   attention at bf16 (`dense_qat_skip_attn=true`, commit `20e9a93`) is worth
   3.5x more than everything else in this table.

## The recipe this freezes

```
model:  medium-shaped, rank 32, d_model 1920, n_heads 30
train:  factor_qat=ternary, dense_qat=ternary (attention stays bf16),
        factor_qat_anneal_steps=500 (soft->hard ramp)
        dropless top-2 MoE, attnres on, MSA on, Engram in host RAM
speed:  12.0k tok/s vs 11.5k bf16 at b8 s512 (the stacked refresh,
        768 -> 3 quantiser calls per bank site, commit 4827c48)
release: 2-bit factors (19.9 MB for the medium bank), fp16 dense (17.3 MB),
         Engram fp16 (100.7 MB) or omitted (37.3 MB artifact)
```

## Not measured here

- **100k+ step behaviour**: the +0.057 is the worst case (short budget); the
  QAT-finish run is what tests whether it shrinks.
- **Downstream** (ARC/HellaSwag) in the ternary form: BPB only so far.
- **b1 scale**: the rank-32 saving there is 313M params (~2.5 GiB VRAM), the
  most valuable consequence, and it is untested.
- **`q64a4`**: in flight.
