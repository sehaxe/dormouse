# AGENT CONTEXT — полный контекст проекта dormouse для супер-агента

**Этот файл — всё, что нужно агенту для продуктивной работы над проектом.**
Обновлён: 2026-10-04. Владец: sehaxe. Стек: Rust/burn/cubecl/CUDA. Карта: RTX 5060 Ti (16 GB).

---

## 0. СЕВЕРНАЯ ЗВЕЗДА

**MoE+TSCT = беспрецедентный оптимизационный инструмент; интеллект датацентра на ОДНОЙ карте.**
Рамка: не фронтир-GPT-4, а 1–10B overtrained (10–50B токенов) вертикального уровня; оптимум по цене инференса. Ёмкость 8B при вычислениях 1B (MoE) + дешёвые эксперты (TSCT). Беспрецедентность = измеримая: params/FLOP, BPB на равном бюджете, цена инференса/токен.

**Тезис «100× меньше»**: петля (LoopBlock, ~10×, ICLR 2024) × overtrained (10–30×) × пост-тренинг SFT→RLVR (10–100×) = ~100×. Калибровка: «1–10B overtrained + пост-тренинг = 100–1000B dense на КОНКРЕТНЫХ задачах» — ДА; «= фронтир вообще» — НЕТ.

---

## 1. ЧТО ТАКОЕ DORMOUSE

Byte-level LM (256 байт, без токенизатора). 9.2M параметров. Одна потребительская карта.

```
DormouseModel = Embedding(256) → LoopBlock × max_iter → RMSNorm → lm_head
LoopBlock (веса общие по итерациям):
  controller: sigmoid-гейты (w_attn, w_mem, w_ffn) + softmax-бленд экспертов
  → KDA (gated-delta attention, единственный attention-ARM, O(n))
  → Engram (FNV-hashed n-gram таблицы, ORDERS=[2,3,4])
  → n_experts × TSCT-экспертов (FFN-рука: W = U·diag(s)·Vᵀ, rank 64)
  → ReZero residual (или Gated Residual)
  → out_proj, усреднение по итерациям
```

**ByteFlow** (канон, интеллект-путь): патч-токены (coding-rate, R_ε = ½·logdet(I+(d/ε²)HHᵀ), границы = top-K по ΔR) вместо байтов. Сжимает байты 3–5×. `--byteflow` флаг, дефолт = true.

---

## 2. АРХИТЕКТУРА — ВСЕ РУКИ (что в main)

| рука | механизм | статус |
|---|---|---|
| **LoopBlock** | веса общие по итерациям (петлевой трансформер) | ✅ |
| **KDA** | gated-delta attention, O(n), fused backward жив (adjoint ≤6.3e-7, −8.3× после тюнинга клея) | ✅ |
| **Engram** | FNV-hashed n-gram память | ✅ |
| **TSCT** | spectral low-rank эксперты (rank 64, NS-ретракция, max_ortho latch 1e-3) | ✅ |
| **MoE** | роутер softmax p_i* БЕЗ ренорма (Switch §2.1), lb_aux | ✅ |
| **BitNet a4.8** | квант активаций (E2M1, int4/int8, STE, f32-граф) | ✅ |
| **Квант факторов** | fp8/fp4 (max_ortho latch), лестница fp8→fp4→int4→ternary | ✅ fp8 |
| **MSA** | sparse attention (MQA 4q/1k, avg-pool r=4, top-KB, хвост блока) | ✅ |
| **JEPA+KoLeo** | self-supervised aux (EMA-учитель), бьёт pure CE 3/3 сида | ✅ |
| **DSpark** | draft head aux | ✅ |
| **GR** | Gated Residual | ✅ |
| **AttnRes** | attention residual | ✅ |
| **mHC** | multi-head controller | ✅ |
| **Muon+** | оптимизатор (ColRow ns=8, head-wise q/k, Adam tables, AdamW rest) | ✅ |
| **ByteFlow** | coding-rate патчер (канон) | ✅ |
| **SFT-скелет** | ChatML-байты + loss-маска assistant-спанов + градиентный гейт | ✅ |
| **byteflow×KDA** | совместимость (патч-токены → LoopBlock) | ✅ |
| **Оптимизации** | graph-stage (−30ms), retract batched (−44ms), Q/K batching (−15ms), fused adjoint (−8.3× bwd) | ✅ |

---

## 3. ИЗМЕРЕННЫЕ РЕЗУЛЬТАТЫ (provenance: benches/history.tsv)

| рука | конфиг | шаги | байты | held-out BPB | ms/step |
|---|---|---|---|---|---|
| **byteflow 100k** | патч-токены, L2-патчер | 100k | 409 MB | **2.255** | 49 |
| byteflow_9m | то же | 2k | 8.2 MB | 3.458 | 49 |
| dormouse small | все руки, Muon+ | 80k | 327 MB | 5.969 | 245 |
| контроль small pureCE | без рук | 2k | 8.2 MB | 6.465 | 450 |

**Якоря (8 MB fit, eval_tail.bin)**: uniform 8.000 · unigram 5.009 · **5-gram+backoff 2.580** (10.3% unseen).
**Бар ВЗЯТ**: byteflow 2.255 < 2.580 (на 70k). Кривая падает до конца.

**dormouse ПЕРЕОБУЧАЕТСЯ**: train-bpb 2.569 vs held-out 5.969 (разрыв 3.4). train-bpb включает aux (CE+JEPA+DSpark). byteflow: train 1.865 vs held-out 2.477 (разрыв 0.6, здорово).

**Эффективность (банк)**: retract batched −44ms, graph-stage −30ms, Q/K batching −15ms, fused adjoint −8.3× bwd, пул −30%.

---

## 4. ТЕКУЩЕЕ СОСТОЯНИЕ (2026-10-04)

### Что идёт прямо .now
- **Тройка** (контролируемый эксперимент эксперта): plain9m vs byteflow vs small, один рецепт, равные байты, 3 сида, 2k шагов. Отвечает: архитектура, патчинг, или рецепт?

### Очередь (по приоритету)
1. **topk-dispatch** (настоящий sparse MoE: gather→expert→scatter, FLOPs = k/E) — ПРИОРИТЕТ №1, rate-limit
2. **Гигант 1B** (bf-giant, 30 ч, overtrained) — чат-порог
3. **1M-окно** (RoPE/max_bytes, рекуррентное локальное окно ~300-400 LOC)
4. **SFT-проводка** (--sft-file в train_loop, флаг — единственный разрыв)
5. **Лестница кванта** (fp4/int4/ternary на факторах, гейт = max_ortho latch)
6. **Depth A/B с φ** (Iso-Depth методика)
7. **Код-корпус** (mix/code = 0% кода, 0.000% скобок — нужно найти/построить)

### Решения владельца
- Все модальности (текст, картинки, аудио, ЭКГ, ДНК)
- 1M контекста, максимум оптимизации
- byteflow = каноническая база (временно), dormouse = ветка
- MoE+TSCT = сверх-технология (спек: docs/reviews/tsct-spec-2026-10-03.md)

---

## 5. КЛЮЧЕВЫЕ ФАЙЛЫ (file:line)

| файл | что |
|---|---|
| `crates/dormouse-core/src/loop_block.rs` | LoopBlock, контроллер, MoE-бленд (982-997: плотный цикл — topk-dispatch чинит) |
| `crates/dormouse-core/src/moe.rs` | роутер softmax p_i*, lb_aux, topk_blend, util_stats |
| `crates/dormouse-core/src/act_quant.rs` | BitNet a4.8 (E2M1, STE) |
| `crates/dormouse-core/src/aux.rs` | JEPA+KoLeo, DSpark |
| `crates/dormouse-train/src/lib.rs` | train_loop, eval (36: bpb=ce/ln2), NaN-файрвол |
| `crates/dormouse-train/src/byteflow.rs` | byteflow-тренер, validate (отказы) |
| `crates/dormouse-data/src/sft.rs` | SFT-формат + маска |
| `vendor/dormouse-fused/crates/burn-spectral/` | NS-ретракция, max_ortho |
| `vendor/dormouse-fused/crates/burn-msa/` | sparse attention |
| `vendor/dormouse-fused/crates/burn-byteflow/` | coding-rate патчер |
| `tools/trio_ab.sh` | контролируемая тройка |
| `docs/reviews/tsct-spec-2026-10-03.md` | полная спека TSCT |
| `benches/history.tsv` | ВСЕ числа с provenance |

---

## 6. ОТКРЫТЫЕ ВОПРОСЫ (research-эксперт, лист №3)

**Диагностика**: Неприводимая энтропия? Разрыв byteflow/dormouse — архитектура или рецепт? Потолок петли?
**Архитектура**: byte vs token, оптимальная для байтов, MoE на байтах, патчинг?
**Рецепт**: LR, batch×seq, оптимизатор, переобучение, порядок данных?
**Данные**: mix/code-не-код, качество корпуса, мультимодальность?
**Масштаб**: 1B overtrained → чат-порог? Потолок 9.2M?
**Пост-тренинг**: SFT, RLVR, агент для byte-level?
**Барьеры**: что между 2.5 и 1.0? Контроль «обычная 9M»?
**Эксперт**: ОДИН эксперимент на сутки = тройка (plain9m vs byteflow vs dormouse).

---

## 7. ДИСЦИПЛИНА (§1.2, ADR)

- **A/B or death**: каждый механизм бьёт своё удаление на held-out BPB, 3 сида, спред контроля. Tie = удаление.
- **LOUD/COUNTED** (ADR-0011/0019): ни одного silent fallback.
- **Zero host-device sync** (§1.3): счёт на хосте, не на bool-тензоре.
- **Одна тяжёлая вещь**: один CUDA-процесс, build lock, MemoryMax=40G.
- **doc-refs green** перед коммитом; НЕ мёржить main без гейтов.
- **history.tsv**: ТОЛЬКО АППЕНД (никогда sort/checkout --theirs).

---

## 8. СЕССИЯ / РЕЖИМЫ

- **AWAY**: `.bulba/goal.md` — ночная программа, задачи, вердикты.
- **План**: `.bulba/plan.md` — минимальное ядро (C0-C12, AWAITING_APPROVAL).
- **Память**: `.bulba/memory.md` — решения, уроки.
- **Конфиги**: `configs/*.toml` (small, base, byteflow, byteflow_9m, plain9m, moe-cap-*).
- **Пресеты**: `--preset <name>`; флаги: `--byteflow`, `--msa`, `--quant`, `--act-quant`, `--retract-every`, `--jepa-precompute`, `--sft-file`, `--graph-stage`, `--graph-capture`.

---

## 9. БЫСТРЫЙ СТАРТ АГЕНТА

1. Прочитать этот файл.
2. `.bulba/goal.md` — текущая программа и вердикты.
3. `benches/history.tsv` — все числа.
4. `docs/reviews/tsct-spec-2026-10-03.md` — спека сверх-технологии.
5. `research/questions-3-for-research-agent-2026-10-04.md` — вопросы эксперту.
6. `docs/reviews/bpbaudit-2026-10-04.md` — честность оценки.
7. Проверить `git log --oneline -10` и статус дорожек.
8. **Не запускать GPU без протокола**: `pgrep -x train` пуст И `nvidia-smi` <1GB.

**Текущая задача**: тройка идёт → дождаться → вердикт. Потом topk-dispatch (№1).
