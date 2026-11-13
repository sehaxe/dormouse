# Gated FFN in the live transformer block — 3-seed verdict (2026-11-10)

**Question.** The live `block_form="transformer"` block is a plain GELU FFN
(`d_ffn = 3072`, `2·d·d_ffn` weights). Does replacing it with a gated GLU at
**exactly equal parameters and FLOPs** buy quality?

**Arms** (`scripts/arch_sweep.sh`, 120 MB, 4 882 steps, b24, s1024, compile on,
`muon_lr_mult=12`, one GPU lock for the whole chain):

| arm | `d_ffn` | FFN weights/layer | activation |
|---|---:|---:|---|
| `control` | 3072 | 5 111 808 | GELU |
| `gated_silu` | **2048** | **5 111 808** | SwiGLU |
| `gated_situ` | **2048** | **5 111 808** | SiTU-GLU (K3 Eq. 12, β1=4, β2=25) |

`3·d·2048 = 2·d·3072 = 5 111 808` — the gated arms are **param- and FLOP-neutral**
by construction, not by tolerance (the config refuses any width more than 2 % off).

## Result — all nine runs at 4 880 steps

| arm | seed 1 | seed 2 | seed 3 | mean | Δ vs control | sd(Δ) | t (df=2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 1.5045 | 1.5122 | 1.5049 | **1.5072** | — | 0.0043 | — |
| `gated_silu` | 1.4890 | 1.4972 | 1.4919 | **1.4927** | **−0.0145** | 0.0013 | **−19.0** |
| `gated_situ` | 1.4885 | 1.4979 | 1.4942 | **1.4935** | −0.0137 | 0.0027 | −8.7 |

Per-seed deltas, all six negative:

```
gated_silu  −0.0155  −0.0150  −0.0130
gated_situ  −0.0160  −0.0143  −0.0107
```

Critical values at df=2: p<0.05 → 4.303, p<0.01 → 9.925. `gated_silu` clears
p<0.01; `gated_situ` clears p<0.05. The single-seed noise on record is
0.008–0.011 BPB, and the paired deltas are 5–12× smaller in spread than that.

## Verdict

**`gated_silu` wins: −0.0145 BPB (−0.96 % relative), at 0.00 % parameter and FLOP
cost.** The pre-registered bar was ≥0.01 BPB better on 3 seeds; it is met with a
3-seed mean of −0.0145 and a paired t of −19.

`gated_situ` is not separated from `gated_silu` in the direction the K3 paper
would suggest (situ is 0.0008 *worse*, well inside its own sd) — so SiTU stays a
flag, and its justification remains **stability at scale**, not BPB at 95M.

**Promotion:** `configs/arch_gated_swiglu.toml` becomes the live recipe for new
runs. `configs/flagship3_dense.toml` is NOT edited — it stays the frozen control
every arm on record is measured against.

## Boundaries (what this does not prove)

- One scale, one budget: 95M at 120 MB. The effect at 2 GB is unmeasured; the
  next long run should carry the gated FFN and report against the 1.1940 dense
  control as a single point, not as a new A/B.
- The win is at fixed parameters, so it cannot be traded for speed — it is pure
  quality, which is why it is worth taking on a free basis.
- `expert_act="memglu"` and the fused kernel were out of scope; `fused_glu`
  remains refused on this form.
