# dormouse v2 — полная архитектура (финал, 2026-09-13)

Решения юзера: без учителя (самодостаточная), RU-приоритет, сначала框架 потом 100M, 1B отложен.

## Поток данных

```
UTF-8 bytes
  → ByteStream (256, next-byte)
  → Embedding (d)
  → LoopBlock ×T, T ~ random 1..8 на batch (PonderNet halt на сэмплированной глубине)
      каждый луп:
        controller (sigmoid w_attn/w_mem/w_ffn + softmax expert-blend)   [уже есть]
        attention: KDA (gated-delta) ⊕ MSA top-k sparse, blend           [уже есть]
        Engram: FNV n-gram 3/5/8, таблицы в host RAM, Sinkhorn-momentum  [есть, опт. W2]
        experts: top-1 hard-route + 1 shared expert (TSCT ternary)       [ЗАМЕНА soft-blend]
        ReZero residual (GR - флаг)
        halt head → lam_n
  → RMSNorm → lm_head (fp32 logits)
  → loss = Σ p_n·CE_n + β·KL(p‖prior) + aux:
        JEPA: ОФФЛАЙН таргеты (sidecar, --jepa-targets), без второго forward  [W2]
        DSpark next-K draft (по стадии после претрейна, замороженный backbone) [перенос V4.1]
        KoLeo
```

## Ключевые параметры (100M-ран)

| Компонент | Значение |
|---|---|
| core params | ~100M dense (d≈768, слоёв-лупов эквив. ~16-24, n_experts 8 top-1 + shared) |
| память | Engram 8M слотов × 3 порядка (2 GB RAM, Sinkhorn ×2) |
| контекст | s512 (потом 1024) |
| оптимизатор | Muon+ (head-wise Q/K) + AdamW rest + Sinkhorn-momentum таблицы |
| точность | fp32 compute; bf16-storage флаг; ternary-SCT факторы |
| луп | max 8, random depth |

## Отличия от текущего кода (что менять)

1. experts: soft-blend (все считаются) → **hard top-1 + shared expert** — реальная экономия FLOPs; общие базы + пер-экспертные тернарные ядра (STSB идея, без Hard-Concrete).
2. random-depth: семпл T на batch в train-лупе (host-side, дёшево), loss через p_n взвешивание уже корректен.
3. DSpark: совместный претрейн → отдельная стадия после (заморозка backbone).
4. fused M4-M6: KDA/MSA/Engram в один узел, wiring за DM_FUSED=1.

## Не делаем
ByteFlow-chunker (до 4K контекстов), per-token halt, иерархический роутер, cross-model KD, Hard-Concrete rank-гейты, per-group ternary scales (дешёвый апгрейд — потом, отдельным A/B).
