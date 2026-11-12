# The ternary artifact — what it costs and what it is worth (2026-11-12)

Two full-corpus runs (2.64 GB, 107 421 steps at b24×1024, one pass), same shape,
same seed, same recipe — the only difference is the weight format.

| run | BPB | vs bf16 | tok/s |
|---|---:|---:|---:|
| **bf16 (control)** | **1.1871** | — | 85 422 |
| ternary, FFN + lm_head (`skip_attn=true`) | 1.2220 | **+2.94 %** | 70 839 |
| ternary, all weights (`skip_attn=false`) | *running* | | |

The 1.1871 is a new record: the previous best was 1.1940 on 2.0 GB, so the extra
25 % of corpus bought −0.007 BPB. The calibration law would have predicted 1.22.

## The gap closes as training proceeds

| step | bf16 | ternary | Δ |
|---:|---:|---:|---:|
| 13 000 | 1.4361 | 1.5005 | +4.5 % |
| 26 000 | 1.3463 | 1.4036 | +4.3 % |
| 39 000 | 1.3100 | 1.3537 | +3.3 % |
| 104 000 | 1.1871 | 1.2220 | **+2.94 %** |

That shape is what the annealed STE (alpha 0→1 over 2 000 steps) is supposed to
produce: the model spends the early run in bf16-like precision and converges to
the quantized network. It is also why "hard ternary from step one" is not what
runs here.

## The artifact, and two corrections to my own claims

Export of the ternary checkpoint: **82,396,943 B (82.4 MB)** against 190 MB in
bf16 = **2.31×**, not the 10× I quoted.

- **Correction 1 — the run did not quantize attention.** `dense_qat_skip_attn`
  defaults to `true`, so `qkv` (49.84 MB) and `proj` (16.61 MB) stayed bf16 and
  are **99.2 % of the remaining bytes**. The export path was also broken until
  tonight: `_kind()` only packed TSCT factor keys, so a plain transformer
  exported byte-identical to bf16 while reporting `mode=ternary`. Fixed
  (`1d91e38`), with a refusal for the silent-no-op case.
- **Correction 2 — the codec is 2 bits + an fp16 scale, not 1.585 bits.** A fully
  quantized 95M model is **23.8 MB payload**, not 18.8 MB. The 1.585 figure is
  what a 5-trits-per-byte packing would give; we do not implement it.

Round-trip is exact enough to call the file the network: load→forward differs
from the reference by 0.2188 (rel 1.24e-2 of max logits), which is **0.154 %** of
the quantizer's own error (142.0).

## Is +2.94 % for 2.31× a good trade?

Against the owner's compression rule (1 % quality for hundreds of times the
size): **no, not on its own.** 2.31× is not hundreds.

Against the reinvestment rule it is worth exactly one experiment: the same file
budget could hold a **219M ternary** model (82.4 MB) where bf16 holds 95M. But at
2.64 GB of data that is 12 bytes/param — an even more undertrained model, not a
better one. The trade only resolves once the corpus is 8–20 GB.

So the artifact question is not "is 2.31× worth 3 %" — it is "does the ternary
format let us buy parameters we could not otherwise afford". That question needs
data, and data is the thing we do not have.
