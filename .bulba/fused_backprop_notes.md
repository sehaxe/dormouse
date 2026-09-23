# M4: явный backward для KDA (burn-kda/burn-gdn2) и MSA (burn-msa)

Цель: сделать backward обоих attention-arms ЯВНЫМ, чтобы `ponder_backward` (план M4) стал
механической работой. Все формулы переписаны из исходников `/home/sehaxe/burn-fused/` и
проверены численно: grad-дамп через burn autodiff (NdArray) + кросс-валидация точного
адъютанта `ChunkWy` против наивного per-op графа (тест в `/tmp/opencode/refdump/`,
вектора в §5). Код репо не менялся.

## TL;DR

1. **Главный факт, которого нет в plan.md**: у обеих рук УЖЕ ЕСТЬ рукописные cubecl-backward
   кернелы, и они УЖЕ работают в текущем тренировочном пути dormouse (`Autodiff<Cuda>`):
   - KDA: forward `gdn2_chunk_intra_kernel`/`gdn2_chunk_inter_kernel` +
     `gdn2_chunk_trajectory_export_kernel` (chunk_cube.rs), backward
     `gdn2_chunk_intra_adjoint_kernel` + `gdn2_chunk_inter_adjoint_kernel` + врапер
     `fused_chunk_backward` (burn-gdn2/src/kernel/chunk_adjoint_cube.rs:23, 221, 467).
     Весь чанк = ОДИН autodiff-узел (`ChunkWy`), backward через экспортированные буферы,
     forward в backward НЕ перезапускается (autodiff.rs:386-430).
   - MSA: forward `msa_sparse_attn_kernel`, backward `msa_backward_kernel` +
     `msa_backward_cuda` (burn-msa/src/sparse_kernel.rs:49, 161, 282) — тоже ОДИН узел
     (`SparseAttnOp`), backward на голом CUDA через atomics.
   M4 = не «вывести и портировать математику», а «вызвать эти кернелы из
   `ponder_backward` и зарегистрировать грады в правильные узлы». Тензорный adjoint
   (§2) — эталон той же математики для CPU-тестов.
2. **Разрыв градиента через KDA state между итерациями loop** — текущее поведение
   dormouse на CUDA: `forward_train_state` возвращает `new_state` как НЕОТРЕКОВАННЫЙ лист
   (burn-gdn2/src/autodiff.rs:16-18, 473: `from_inner(new_state_prim)`), а loop_block.rs:260
   кладёт его в `kda_s` без detach. Численно: градиент между итерациями loop через KDA
   state НЕ течёт вообще (только через выход слоя). Подтверждено численно (§5.3:
   попытка включить `S_out` в loss через adjoint-op не доходит до op). Fused-версия
   обязана воспроизвести этот разрыв, иначе grad-check vs burn разойдётся (k/v/g/b/w
   расходятся на O(1), замерено: maxabs до 1.0 на том же тесте).
3. **Index branch MSA в dormouse НЕ обучается**: `loop_block.rs:265` берёт только
   `.output`, KL-loss отбрасывается; единственный путь градиента к block_scores — KL
   (`loss.rs:13`), второй путь (top-k → Int) градиента не имеет по построению.
   Численно: backward-граф вообще не содержит узлов index branch
   (`INDEX_BRANCH grads absent (no node): q=false k=false`, §5.4). «top-k
   straight-through» отсутствует как класс: выборка жёсткая (Int), градиент к
   невыбранным блокам = 0, к выбранным = обычный softmax-backward внутри выбранных
   блоков. `gradient_detach=false` влияет только на KL-путь (не используется).
4. **plan.md врёт про MSA**: нет никакой average-pool агрегации ключей и нет ReLU.
   «Пулинг» = **max-pool** по блоку (`reshape [B,Hkv,S,c,bs].max_dim(4)`,
   index_branch.rs:128-132), скоринг = сырой `q_idx·k_idx/√d_idx` с каузальной
   маской −1e4 (module.rs:99, index_branch.rs:93-111), `NEG_INF_SAFE = -1e4`
   (burn-msa/src/lib.rs:3). Плейсхолдер plan.md «block-causal ReLU scoring» — стейл.
5. Чанк-размер KDA = 16 (`attention.rs:42`), state fp32 при BF16=0 (dtype следует
   входам, burn-kda/src/lib.rs:545-547). scale=1.0 (lib.rs:579/591). короткая свёртка
   ВЫКЛЮЧЕНА в dormouse (`use_short_conv: false`, attention.rs:41) → q/k/v =
   `silu(Linear(x))`, backward свёртки не нужен.

Обозначения: `dY` — upstream-градиент выхода Y; размеры dormouse `small`:
B·T=1536, d=768, H=12, HD=64 (head_dim), Hkv=3, d_idx=64, chunk C=16, n_chunks=T/C.

---

## 1. KDA: forward-контракт (что должно быть сохранено для backward)

Полный модуль `KdaModule::forward_train_state` (burn-kda/src/lib.rs:530-608) =
`project` → chunked WY → `output`. Инференс-шейпы (dormouse, H=12, HD=64, HV=H,
VD=HD, GVA-повторов нет при HV==H):

### 1.1 project (lib.rs:390-482)

```
x: [B,T,D]
q_raw = x·W_q, k_raw = x·W_k, v_raw = x·W_v        # W_*: [D, H·HD] runtime-лейаут [in,out]
q_act = silu(q_raw), k_act = silu(k_raw), v_act = silu(v_raw)   # use_short_conv=false, lib.rs:419/424/429
q = l2norm(q_act, eps=1e-6), k = l2norm(k_act, eps=1e-6)        # по последней оси HD, lib.rs:432-433, l2norm.rs:4-7
# decay (K3 Sigmoid, lib.rs:177-203):
z   = (x·W_up)·W_dn + b_alpha                     # W_up: [D,r], r=HD; W_dn: [r,H·HD]; b_alpha: [H·HD]
a   = clamp(a_log, -10, 20)                       # a_log: [H,1], кламп lib.rs:189-194 (NaN-фикс 2026-08-29)
g   = g_min · sigmoid(exp(a) · z_h)               # z_h: [B,T,H,HD], g_min = -5 (lib.rs:26, 300-304)
alpha = exp(g) ∈ (e^-5, 1)
log_decay g4 = log(alpha), permute → [B,H,T,HD]   # lib.rs:437
beta  = sigmoid(x·W_beta)                         # [B,T,H], W_beta: [D,H], lib.rs:442
b_k = beta.repeat(HD) → [B,H,T,HD]; b_v = beta.repeat(VD) → [B,H,T,VD]  # lib.rs:443-445
gate= sigmoid(x·W_g) → [B,H,T,VD]                 # FullRank, lib.rs:470-479
```

Backward-формулы этих поэлементных частей — тривиальные (sigmoid/silu/exp/кламп
— STE не нужен, кламп a_log: градиент 0 вне [-10,20], burn clamp так и считает).

l2norm: `y = x/√(Σx²+eps)` по HD →
`dx = (dy − y·Σ(dy·y)) / √(Σx²+eps)` (жакобиан деления на норму по последней оси).

### 1.2 chunked WY (burn-gdn2/src/forward.rs:57-349, путь C≤16 = TILE=16)

На чанк c (позиции [s, s+C)):
```
E     = exp(cumsum_t(g_c))                        # [B,H,C,HD], forward.rs:154-155
kG    = k_c / E                                   # forward.rs:156
qE    = q_c ⊙ E                                   # forward.rs:164
aqk   = (qE·kGᵀ) ⊙ causal ⊙ scale                 # scale=1.0, forward.rs:157-158
bk    = b_k ⊙ k_c                                 # forward.rs:159
akk   = ((bk ⊙ E)·kGᵀ) ⊙ strict                   # strict-lower маска, forward.rs:160-161
M     = I + tril_strict(akk);  m_inv = M⁻¹        # построчная инверсия forward.rs:300-312
rhs_k = bk ⊙ E;  rhs_v = w_gate ⊙ v_c             # forward.rs:162-163 (w_gate=1 в KDA!)
W_wy  = m_inv·rhs_k;  U = m_inv·rhs_v             # forward.rs:314-315
v_new = U − W_wy·S_in                             # forward.rs:318
out_c = aqk·v_new + qE·S_in·scale                 # forward.rs:319-321
S_out = S_in ⊙ E_lastᵀ + (k_c ⊙ decay)ᵀ·v_new     # decay = E_last/E (построчный), forward.rs:337-339
```
`ChunkScratch` (то, что backward НЕ перепосчитывает): `g_exp=E, q_gated=qE, aqk,
m_inv` (forward.rs:39-49, 328-334). Путь CUDA: fused-кернелы экспортируют
`FusedBackwardInputs {m_inv, aqk, qgt(=qEᵀ [k][r]), glast, v_new, states, w(=W_wyᵀ),
u(=Uᵀ), gexp}` (chunk_adjoint_cube.rs:431-445, autodiff.rs:408-417).

### 1.3 output (lib.rs:484-498)

```
rms   = √(mean_vd(attn_out²) + eps)               # eps = norm_eps (K3, 1e-5), по VD
normed= attn_out / rms
gated = normed ⊙ gate ⊙ o_norm_w                  # o_norm_w: [VD]
out   = gated.reshape([B,T,HV·VD])·W_o + 0        # W_o: [HV·VD, D]
```

---

## 2. KDA: явный backward (адъютант чанка)

Единственный источник истины: `ChunkWy::backward` (burn-gdn2/src/autodiff.rs:53-338).
Ниже — построчная транскрипция. По чанку (в обратном порядке чанков, autodiff.rs:184);
входы: d_out_c (паддинг до C нулями), d_state_acc (аккумул. адъютант S_out), scratch,
k/v/b/w — чекпоинты; q и g НЕ чекпоинтируются — их вклад входит через scratch
(autodiff.rs:61-64, ids[0]=None/ids[3]=None на 461-468):

```
# пере-вычисление (bit-identical forward, autodiff.rs:208-213):
kG = k/E; rhs_k = b⊙k⊙E; rhs_v = w⊙v; W_wy = m_inv·rhs_k; U = m_inv·rhs_v; v_new = U − W_wy·S_in
decay = E_last / E;  khat = decay ⊙ k

# out = aqk·v_new + qE·S_in·scale:                                     (autodiff.rs:216-218)
d_aqk   = d_out_c · v_newᵀ
d_qk    = d_aqk ⊙ (causal·scale)
d_v_new = aqkᵀ·d_out_c

# BPTT через S_out = S⊙E_lastᵀ + khatᵀ·v_new:                          (autodiff.rs:222-230)
d_k_hat  = v_new · d_state_accᵀ
d_k_bptt = d_k_hat ⊙ decay
d_decay  = d_k_hat ⊙ k
d_v_new += khat · d_state_acc

# v_new = U − W_wy·S_in; inter = qE·S_in·scale; S_in-часть BPTT:        (autodiff.rs:233-242)
d_s_c   = −W_wyᵀ·d_v_new + qEᵀ·d_out_c·scale + E_lastᵀ ⊙ d_state_acc

# decay = E_last/E (E_last — строка C−1):                              (autodiff.rs:244-252)
d_e_last  = Σ_t(d_decay/E) + Σ_vd(d_state_acc ⊙ S_in)        # адъютант строки E_last
d_e_decay = −d_decay ⊙ decay / E

# qE: из inter и aqk:                                                  (autodiff.rs:255-260)
d_qe = d_out_c·S_inᵀ·scale + d_qk·kG
d_kg = d_qkᵀ·qE                              # [C,HD]ᵀ-матmul по чанку

# WY-сOLVE через треугольный адъютант:                                 (autodiff.rs:262-270)
d_w_wy  = −d_v_new·S_inᵀ
d_rhs_k = m_invᵀ·d_w_wy                      # M⁻ᵀ-решение
d_rhs_v = m_invᵀ·d_v_new
d_akk   = (−d_rhs_k·W_wyᵀ − d_rhs_v·Uᵀ) ⊙ strict

# akk → bk/kG:                                                         (autodiff.rs:272-280)
d_kg  += d_akkᵀ·rhs_k
d_bk_e = d_akk·kG
bk     = rhs_k/E                             # = b⊙k
d_bk   = (d_rhs_k + d_bk_e) ⊙ E
d_b_c  = d_bk ⊙ k;  d_k_bk = d_bk ⊙ b
d_e_rhsk = d_rhs_k ⊙ bk

# rhs_v = w⊙v:                                                         (autodiff.rs:283-284)
d_w_c = d_rhs_v ⊙ v;  d_v_c = d_rhs_v ⊙ w

# kG = k/E:                                                            (autodiff.rs:287-288)
d_k_kg = d_kg / E
d_e_kg = −d_kg ⊙ k / E²

# qE = q⊙E:                                                            (autodiff.rs:291-293)
d_q_c  = d_qe ⊙ E
d_e_qe = d_qe ⊙ qE / E
d_e_bke = d_bk_e ⊙ bk

# E = exp(G), G = cumsum(g): d_G = d_E⊙E; d_g = reverse-cumsum:         (autodiff.rs:296-303)
d_e  = d_e_rhsk + d_e_kg + d_e_qe + d_e_bke + d_e_decay
d_e[last] += d_e_last                                  # slice_assign, autodiff.rs:298-302
d_g_c = revcumsum(d_e ⊙ E)
```
Сборка по чанкам (autodiff.rs:305-331): каждый d_*_c обрезается до c_real и
конкатенируется в обратном порядке; `d_state_acc ← d_s_c` (BPTT-цепочка);
**только для ci==0** `d_s = d_s_c` — это градиент ВХОДНОГО state (autodiff.rs:314-321).
Далее d_q..d_w идут в проектные линейные слои и в dX (обычные dW = Xᵀ·dY),
d_g — в decay-ветку (через alpha=exp(g): d_logit = d_g·g_min·σ'(...)·exp(a) и т.д.),
d_b — в beta_proj, d_s — Никуда на CUDA (лист, см. TL;DR п.2).

Градиент d_w_gate: burn-kda передаёт w_gate ≡ 1 (lib.rs:579, 591) — d_w не нужен, но
слот в сигнатуре есть.

---

## 3. MSA: forward и backward как реализовано

Конфиг dormouse (`attention.rs:45-52`): Hq=12, Hkv=3 (n_heads/4), d_head=64, **d_idx=64**
(жёстко `MsaConfig::new(d_model, n_heads, n_kv, head_dim, 64)`, attention.rs:46 — НЕ
дефолтные 32), block=32, topk=8, force_local_block=true, causal=true, use_rope=false,
use_kl_loss=true (результат отбрасывается), gradient_detach=false.

### 3.1 forward (module.rs:75-131)

```
# index branch (grad НЕ нужен — см. TL;DR п.3):
q_idx = (x·Wqi).reshape[B,S,Hkv,d_idx].permute → [B,Hkv,S,d_idx]     # index_branch.rs:35-43
k_idx = (kv·Wki).unsqueeze → [B,Hkv,Skv,d_idx]
scale_idx = √d_idx                                                   # module.rs:99
block_scores = max-пул: (q_idx·k_idxᵀ/√d_idx + каузальная маска −1e4)
               .reshape[B,Hkv,S,nblocks,bs].max_dim(4)               # index_branch.rs:93-132
# чанкинг по 8 блоков (index_branch.rs:67) — только память, не математика.
idx = topk(block_scores + 1e4·1[t/bs==block])  → Int [B,Hkv,S,topk]  # topk.rs:126-170
#   argtopk3 на CUDA = exp_free_topk_kernel (topk.rs:67-105); force_local — баас +1e4
#   своему блоку (topk.rs:153-163); k≥nblocks → все блоки (topk.rs:142-152).
# real attention:
q = x·Wq → [B,Hq,S,d]; k,v = kv·Wk/Wv → [B,Hkv,Skv,d]                # module.rs:107-109
out = Σ_blk online-softmax(Q K_blkᵀ/√d ⊙ causal(−1e4)) V_blk          # attention.rs:217-294
ba  = per-block attention sums (mean по qpkv) — untracked (autodiff.rs:362-363)
out3 = out_proj(out.reshape[B,S,Hq·d])
```

### 3.2 backward (autodiff.rs:106-270, тензорный эталон; кернел — sparse_kernel.rs:161)

Градиенты идут ТОЛЬКО в q/k/v (SparseAttnOp, N_PARENTS=3, autodiff.rs:32). out_proj,
index branch, top-k — вне узла (обычные burn-узлы / нет графа). Пусть `attended =
topk·bs`, `A_b = softmax(scores_b)` по выбранному объединению блоков
(online-softmax: один общий max/sum по ВСЕМ topk блокам — three-pass, autodiff.rs:138-186):

```
# pass 1: m, l — глобальные running max/sum по всем блокам             (autodiff.rs:139-160)
# pass 2: wd_sum = Σ_b Σ_tok A_b ⊙ dA_b  (центральный член softmax-bwd) (autodiff.rs:164-186)
# pass 3, на блок b:                                                   (autodiff.rs:192-263)
dA_b    = d_out·V_bᵀ
dS_b    = A_b ⊙ (dA_b − wd_sum)                # жакобиан softmax по объединению блоков
dq     += (dS_b·K_b)/√d                        # див по всем выбранным токенам
dk_b    = dS_bᵀ·Q/√d;  dv_b = A_bᵀ·d_out
dk/dv  += scatter_add(dk_b/dv_b, idx)          # clamp idx∈[0,Skv−1]: хвост-дубликаты СКЛАДЫВАЮТСЯ
```
Семантика градиентов:
- невыбранные блоки: градиент ровно 0 (нет терма) — hard selection, не STE;
- сам выбор (block_scores): градиента нет (Int);единственный путь — KL-loss
  (loss.rs:7-26: dlog_p через softmax от gathered scores), в dormouse отброшен;
- clamp хвоста: дубликаты последнего токена суммируются (scatter Add, autodiff.rs:243-262);
- каузальная маска: −1e4 аддитивно до softmax ⇒ маскированные колонки получают
  A≈0 и не дают градиента (в точной арифметике e^{-1e4−m} = 0 в f32).

Fused-кернел `msa_backward_kernel` (sparse_kernel.rs:161-254) считает то же самое
per (b·hkv cube, token, q-head-in-group): пересчитывает scores в shared, один max по
нединамическому пути, softmax, wd_sum, потом `dq += dsc·k·scale` (обычная запись),
`dk/dv fetch_add` — atomics. НЮАНС: маскированные записи получают score 0.0, а не −1e4
(sparse_kernel.rs:198-201), т.е. их вклад в знаменатель softmax = exp(−max), а не 0 —
маленькое систематическое расхождение tensor-path vs fused (у обоих тесты проходят
с допуском 1e-2/2e-2, autodiff.rs:465, 560). Для M4: внутри ponder_backward можно
пользоваться любым, но grad-check vs burn-тензорного пути брать с rel-допуском ≥1e-2
или через сам fused-kerнел как эталон.

### 3.3 router blend (attention.rs:74-89)

`out = gdn2·σ(r) + msa·(1−σ(r))`, r = router(x) [bt,1]:
`d_r = (d_gdn2_out·gdn2_out − d_msa_out·msa_out)·σ(r)(1−σ(r))` — поэлементно.

---

## 4. Минимальный набор cubecl-кернелов для M4

Существующие (переиспользовать как есть, launch-конвенция `launch_unchecked` +
`CubeTensor` downcast, как в chunk_adjoint_cube.rs:399-429 и sparse_kernel.rs:297-317):

| Кернел | Файл | Тип | Входы → Выходы |
|---|---|---|---|
| `gdn2_chunk_intra_kernel` | chunk_cube.rs:28 | per-chunk (см. шапку файла) | fwd чанка: aqk, akk, M⁻¹, W, U, v_new |
| `gdn2_chunk_inter_kernel` | chunk_cube.rs:319 | per-chunk | state-trajectory + out |
| `gdn2_chunk_trajectory_export_kernel` | chunk_cube.rs:642 | export | FusedBackwardInputs (9 буферов, §1.2) |
| `gdn2_chunk_intra_adjoint_kernel` (BK1) | chunk_adjoint_cube.rs:23 | token-parallel | d_out,v_new,qgt,gexp,m_inv,w,u,k,b,v,wg,s_before,d_v_new,d_k_bptt,d_e_bptt,d_e_last → d_q,d_k,d_b,d_g,d_v,d_w |
| `gdn2_chunk_inter_adjoint_kernel` (BK2) | chunk_adjoint_cube.rs:221 | sequential reverse | BPTT через state (d_v_new/d_k_bptt/d_e_bptt/d_e_last для BK1) |
| `fused_chunk_backward` (врапер) | chunk_adjoint_cube.rs:467 | glue | + d_s, d_e reverse-cumsum в tensor-glue |
| `msa_sparse_attn_kernel` | sparse_kernel.rs:49 | per (b,h,t) | out [B,Hq,S,d], ba [B,Hkv,S,topk] |
| `msa_backward_kernel` | sparse_kernel.rs:161 | per (b,h,t) | q,k,v,bi,dout → dq, dk(atomic), dv(atomic) |
| `exp_free_topk_kernel` | kernel/topk_select.rs | per-row | scores → topk idx |

Что ДОПИСАТЬ в dormouse (fused.rs) для M4:
1. `k_gather_rows`/scatter-glue: у BK1/BK2 лейаут `[nblk,·,·]` плоский (nblk = B·H·nt,
   chunk_adjoint_cube.rs:17-20) — нужен pack/unpack между burn-тензорами [B,H,T,·] и
   плоскими буферами (в burn-gdn2 это делает врапер; скопировать).
2. Проекционные dW: обычные `k_matmul` из плана M0-M3 (dW_q = q_act_inᵀ·d_q_raw и
   т.д.) + silu/l2norm/decay-ветки поэлементно — это не новые типы кернелов.
3. output-гейт: rms по VD (редукция), остальное поэлементно + `k_matmul` W_o.
4. MSA: дописывать НЕчего, кроме вызова `msa_backward_cuda` из своего Backward-импл
   (референс вызова — burn-msa/src/autodiff.rs:51-75). ba не трекается.
5. Порядок в ponder_backward (reverse loop): d(step_out)→out_proj; d(h)→branch-ветки;
   d_attn = w_attn·d_y + d_w_attn·attn (через контроллер), затем blend-джакобиан (§3.3),
   затем KDA (BK2→BK1 per chunk, обратный порядок чанков) и MSA, scatter в d_normed_attn.

---

## 5. Численные векторы (реальный прогон burn autodiff, /tmp/opencode/refdump)

Полный тест: `cargo test --release -- --nocapture` в /tmp/opencode/refdump (NdArray,
фиксированные веса, детерминировано). Исходник src/lib.rs; код ниже сокращён до
схемы. **Все градиенты ниже — реальный вывод, не ручной вывод.**

### 5.1 KDA-модуль целиком (hidden=8, H=2, HD=4, T=4, chunk=16, Sigmoid decay, g_min=−5,
FullRank gate, use_short_conv=false; x = [[−0.9,0.2,0.5,−0.3,0.8,−0.6,0.1,0.4],
[0.3,−0.7,0.9,0.2,−0.4,0.6,−0.8,0.5], [−0.2,0.4,−0.5,0.7,0.3,−0.9,0.6,0.1],
[0.6,−0.1,0.8,−0.4,0.2,0.5,−0.3,0.7]]; loss = out².sum(); веса запинены
детерминированным генератором ((i·37)%23)/50−0.22, b_alpha=0.1+0.02i, a_log=[−3,−2.5],
o_norm=[1,0.9,1.1,1.05]; см. src/lib.rs kda_module_reference_grads)

```
KDA out shape=[1, 4, 8];  state shape=[1, 2, 4, 4]
state vals=[0.005290378, -0.0008177067, -0.0015825499, -0.012468916, 0.001199178, 0.000481613,
 -0.0011843282, -0.001854735, 0.007123002, -0.0038564336, 0.0025716878, -0.023578323,
 0.0020852247, -6.292347e-6, -0.0013061407, -0.0038869726, -0.003942526, -0.010079359,
 -0.0010427961, -0.02177901, 0.0034856717, 0.009856628, 0.0015503897, 0.017468423,
 0.001231527, 0.0035107436, 0.00080831983, 0.0062850006, 0.0057233125, 0.014157224,
 0.0012754528, 0.03268009]
GRAD kda_x [1,4,8] = [6.9617887, 5.831655, -2.8834178, -2.430118, -2.0227132, -7.593444,
 -6.9595094, 6.6016045, 7.0502663, 4.556194, -3.272282, -1.6936412, -0.45257717, -8.015177,
 -7.179617, 6.7407527, 0.17975429, 0.05466534, -0.29904717, -0.19057724, 0.017841896,
 -0.109971866, -0.22739424, 0.40132317, -0.6071461, -0.5184954, 0.26526102, 0.17824204,
 0.2820666, 0.24351604, -0.37295932, -0.1603902]
GRAD kda_q_proj_w [8,8] = [0.06888623, -0.016434796, -0.044593066, 0.0583028, -0.16605584,
 0.16298552, 0.6858342, 0.6071769, ...]            # полный вектор в canonical_vectors.txt
GRAD kda_k_proj_w [8,8] = [-0.8672034, 0.08217005, -0.5608145, 0.5146952, 1.4662111,
 7.5213623, 12.972239, -16.610155, ...]
GRAD kda_v_proj_w [8,8] = [0.21624604, 0.23123018, -0.96197844, -0.053427946, ...]
GRAD kda_beta_proj_w [8,2] = [0.0009982181, 0.026500596, 0.0020287388, -0.05288392,
 -7.2603165e-5, 0.07778355, 0.014696758, 0.016491238, 0.0033277324, -0.024829194,
 -0.011484099, 0.044127196, 0.004613714, -0.061682276, 0.00867851, 0.04963853]
GRAD kda_decay_w_up [8,4]; kda_decay_w_down [4,8]  # ~1e-4, полный вектор в файле
GRAD kda_decay_b_alpha [8] = [-0.0011759596, -1.022713e-5, -0.0005567357, -0.00026554253,
 0.0015774973, -0.0032576907, 0.0026242665, 0.0001002611]
GRAD kda_decay_a_log [2,1] = [-0.00022318923, 0.0003782183]
GRAD kda_o_gate_w [8,8]; kda_o_norm_w [4] = [0.28280047, 0.6796751, 0.4841561, 0.9450027]
GRAD kda_o_proj_w [8,8] = [-0.44955966, -0.07145259, 0.051387236, 0.24948908, ...]
```

### 5.2 Точный адъютант ChunkWy vs per-op граф (листовые q,k,v,g<0,b,w,s;
loss = out².sum(); полный дамп d_q..d_s в canonical_vectors.txt):
```
ADJ_CHECK q:    maxabs=2.980e-8  maxrel=1.055e-7   refmaxabs=4.885e-1
ADJ_CHECK k:    maxabs=6.858e-5  maxrel=9.979e-4   refmaxabs=6.583e-1
ADJ_CHECK v:    maxabs=0.000e0   maxrel=0.000e0    refmaxabs=7.387e-1
ADJ_CHECK g:    maxabs=5.960e-8  maxrel=1.196e-4   refmaxabs=4.034e-1
ADJ_CHECK b:    maxabs=7.451e-9  maxrel=3.379e-7   refmaxabs=1.049e-1
ADJ_CHECK w:    maxabs=0.000e0   maxrel=0.000e0    refmaxabs=8.442e-1
ADJ_CHECK s_in: maxabs=0.000e0   maxrel=0.000e0    refmaxabs=3.315e-1
```
Формулы §2 подтверждены. k — худший (двойной проход через M⁻¹ и k/E·bk·E) — допуск
для grad-check KDA брать rel ≤ 1e-3 по k и abs ≤ 1e-4, остальное ≤1e-6 (совпадает с
собственным допуском burn-gdn2 tests/autodiff_chunk.rs:15-16 ATOL 1e-5 / REL 1e-3).
Замечание: листовые кернел-чекпоинты burn-а требуют leaf-тензолов (NoCheckpointing:
«Can't convert a non leaf tensor into a tracked tensor») — в fused-пути буферы и так
листовые, ограничение не кусается.

### 5.3 Разрыв градиента через выходной state (адъютант-op):
если в loss добавить `S_out².sum()`: per-op путь даёт другие d_k/d_v/d_g/d_b/d_w/d_s
(замер maxabs до 1.0/отн. до 19), adjoint-op — НЕТ (state — неотрекованный лист,
`from_inner`, burn-gdn2/src/autodiff.rs:473). Градиент d(state-input) при этом точен в
ОБОИХ путях (kda_state_in_grad дампится, [1,2,4,4], полный вектор в файле). M4-вывод:
в ponder_backward state-градиент между итерациями = 0 (совпадает с текущим CUDA-
поведением), d_s внутри одной итерации не возникает (state один на вызов).

### 5.4 MSA-модуль (d=8, Hq=2, Hkv=1, d_head=4, d_idx=4, block=2, topk=1, causal,
KL отброшен как в loop_block.rs:265; веса запинены ((i·41)%19)/50−0.18):
```
MSA out shape=[1, 4, 8]
GRAD msa_x [1,4,8] = [-0.08435, -0.06854248, 0.17863476, ...]
GRAD msa_attn_q_w [8,8]; msa_attn_k_w [8,4]; msa_attn_v_w [8,4]; msa_attn_out_w [8,8]
   # k_w имеет симметричную структуру ±: дубликат хвоста из clamp виден в граде
INDEX_BRANCH grads absent (no node): q=false k=false   # index branch вне графа
```
Полные числа: /tmp/opencode/refdump/canonical_vectors.txt (38 строк, воспроизводимо).

---

## 6. fp32-ограничения и приоритеты фьюзинга

1. **Обе руки f32-only сегодня**: MSA-кернелы читают буфер как f32 — bf16-вход =
   ILLEGAL_ADDRESS (loop_block.rs:262-264, замер 2026-08-29); KDA-fused — то же
   (f32-компут, state dtype = dtype входов, lib.rs:542-547). В ponder_forward перед
   KDA/MSA каст в f32 обязателен (сейчас так и есть: normed_attn из normed_f).
2. Самое узкое место autodiff-графа, которое фьюзится в первую очередь — НЕ
   KDA/MSA (они уже 2-3 узла), а проекции/silu/l2norm/decay вокруг них: ~десятки
   burn-опов на итерацию в `project`+`output` (lib.rs:390-498) плюс GVA-repeat'ы.
   Точная граница: оставлять `ChunkWy`/`SparseAttnOp`-подобные одиночные узлы,
   сворачивая всё между ними в k_matmul + поэлементные кернелы.
3. Клампы/маски в f32: a_log clamp [-10,20] (lib.rs:189-194), NEG_INF_SAFE=−1e4
   конечный (index_branch.rs:112-114: −inf даёт 0·inf=NaN в маске). В bf16-режиме
   −1e4 представим, но экспоненциалы считать в f32 (правило bf16_KERNEL_PLAN §2).
4. Atomics в msa_backward_kernel (dk/dv fetch_add) недетерминированы по порядку —
   grad-check сравнивать с допуском (burn-msa сам использует 2e-2 после 30 прогонов,
   autodiff.rs:538-565), либо суммировать детерминированно по qpkv.

## 7. Расхождения с plan.md (громко)

- **«top-k straight-through» в plan.md (M4-строка, Backward checklist)**: НЕ существует.
  Выбор жёсткий, Int, без градиента; градиент к невыбранным блокам = 0; «index branch
  trains» (AGENTS.md/plan) фактически ложь для текущего лосса dormouse — градиент
  отсутствует ВООБЩЕ (§5.4), т.к. KL отброшен в loop_block.rs:265.
- **«avg-pool keys r=4»**: нет пула ключей; max-пул скроров по блоку bs=32 (dormouse
  small), index_branch.rs:128-132. ReLU в скоринге нет.
- **«Портировать backward из burn-kda/burn-msa»**: backward уже написан и
  выполняется (fused-кернелы, §4). M4 — интеграционная работа, не математическая.
- **State через итерации loop**: план молчит; фактическая семантика — разрыв
  (§5.3). Если хочется настоящего BPTT через loop — это изменение поведения
  относительно burn-эталона, отдельный A/B, не M4.

## Источники

- burn-gdn2: src/forward.rs (WY forward + scratch), src/autodiff.rs (ChunkWy backward —
  §2 дословно), src/kernel/chunk_cube.rs, src/kernel/chunk_adjoint_cube.rs (fused bwd),
  src/l2norm.rs, src/short_conv.rs (не используется), tests/autodiff_chunk.rs (допуски).
- burn-kda: src/lib.rs (KdaModule/KdaDecay, forward_train_state, клампы, init).
- burn-msa: src/module.rs, src/attention.rs (sparse_attn_batched_gqa + block_scores),
  src/autodiff.rs (SparseAttnOp + sparse_attn_backward_tensor), src/sparse_kernel.rs
  (fused fwd/bwd), src/index_branch.rs, src/topk.rs, src/loss.rs, src/lib.rs (NEG_INF_SAFE).
- dormouse: crates/dormouse-core/src/attention.rs (конфиг рук), src/loop_block.rs
  (потребление), src/config.rs (small: d=768, H=12, HD=64, msa_block=32, topk=8).
- Численный прогон: /tmp/opencode/refdump (Cargo.toml + src/lib.rs + canonical_vectors.txt),
  burn 0.22.0-pre.3, NdArray + Autodiff, фиксированные веса. Код репо не тронут.
