# dormouse bf16 — all-bf16, no fp32 waste

Цель: всё хранение и вычисление в bf16, acc f32, master fp32 нет.

## 1. Хранение

- Веса: `LinearLike` `bf16` (U/V в TSCT тоже bf16, retract every 1 step, Newton-Schulz 3 iters)
- Если ortho >1e-3 — fallback fp32 только для U/V, остальное bf16
- Квантование: `e2m1` уже bf16, `int4`/`int8` bf16 scale

## 2. Вычисление

- Matmul: `bf16 × bf16 → f32` acc, затем cast bf16
- RMSNorm: bf16 in, f32 sum, eps 1e-3 (was 1e-5 for fp32)
- RoPE: cos/sin bf16 table
- Softmax: bf16 in, f32 exp, bf16 out
- KDA/MSA/Engram: bf16 paths, no f32→bf16 casts in loop (z bf16 in, no g→F32)

## 3. Оптимизатор

- AdamW/Muon moments bf16 + stochastic rounding
- Loss scaling 1024 for CE/JEPA/KoLeo
- MuonQ 4-bit later 7× saving

## 4. Кернелы (cubek path)

- `cubek-matmul`: tiled bf16 (M/N/K%vs tails already fixed), naive bf16
- `cubek-quant`: bf16 quant/dequant
- `burn-kda`, `burn-msa`, `burn-engram`: bf16 forward
- `burn-sct`: TSCT bf16 retract

## 5. Риски

- SCT bf16 ortho drift → retract every 1, monitor `max_ortho`
- Small GEMM rank64 bf16 slower → fuse + memoize TSCT ternary
- Layernorm bf16 eps → 1e-3, test finite

## 6. Профиль

- loop_probe bare-Cuda 0.4s vs autodiff 1.35s → 60% graph, bf16 cuts 1.8× on large FFN
- VRAM: 1B fp32 4GB → bf16 2GB, +2GB for batch 16 s1024
