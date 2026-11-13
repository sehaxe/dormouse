# 2026-11-08 — Data/eval integrity: are the reported numbers measurements of what they claim?

**Mode:** read-only audit. No code, config, checkpoint or corpus was modified. CPU only.
**Live lane:** `port/` (Python). **Training logs:** `/home/sehaxe/logs/`. **Corpora:** `data_local/`.
**Tags:** **[M]** measured here or in the named log · **[C]** computed from named numbers · **[T]** read from a cited paper/table · **[S]** speculation (no direct measurement).

**Method / receipts.** Eval windows come from the JSONL `eval` rows and the `eval step=... off=... over ...` lines. Leakage was measured by taking every 64-byte anchor of `web_eval.bin[0:196800]` (2,148 unique printable-ASCII anchors at 64-byte stride), scanning each source with `rg -a -b -o -F -f`, extending every hit left/right in both files, and unioning the eval intervals. A hit is a verbatim run ≥64 B; runs shorter than that are a lower-bound miss. Sidecars were checked by reading `starts` against the `.bin` with `mmap`. The claims ledger was re-run with `--check` into `/tmp`, not into the repo.

---

## 0. Verdict (RU)

**Арифметика честная, атрибуция — нет.** Все проверенные BPB пересчитываются из `nats/ln2` и совпадают с логами до 0.0005. Но как минимум два числа выдаются за held-out, хотя это не так, а один «фикс на −0.267 BPB» измерен на трёх одновременно изменённых переменных.

Пять главных находок, по убыванию цены:

1. **`1.4621` — это eval на `fwedu.bin`, то есть на тренировочном источнике, и окно лежит внутри прочитанного префикса.** Признак `off=100,000,000 + 196,563 B` воспроизводится только на `best/fwedu.bin` с `doc_masking=true` (45 замаскированных позиций). К 0.25 ГБ курсор fwedu стоит на 108.9–110.0 МБ, а окно — `[100.000, 100.197) МБ`. Это не «другое окно» — это увиденные байты. **[M]/[C]**
2. **Исторический `mix_v4/web.bin` → `web_eval.bin` leak: 9,322 B из 196,800 = 4.74 %, максимальный непрерывный run 5,948 B.** `web.bin` — тренировочный файл `flagship1`, который скорит 196,608 B; `doc_sha1_disjoint` в `.web.state.json` сравнивает 6.3 МБ шарды, а не документы. **[M]**
3. **`web_v7` leak: 2,735 B = 1.39 % контента, но реально прочитано 182 B (0.25 ГБ), 250 B (983 МБ), 499 B (4.2 ГБ).** Утверждение leakage-аудита «the model has not seen those bytes» верно только для 2k шагов (0 B). **[M]**
4. **«The two run lines differ only in steps» — ложь.** `f3short` — `mix_v7` (12 источников), `doc_masking=false`; `flagship3` — `mix_v7_docmask` (9 источников), `doc_masking=true`. Дельта `1.927 → 1.660` не является эффектом Muon-фикса. **[M]**
5. **Гейт claims-ledger не гейт.** `scripts/gate_fast.sh` в рабочем дереве переведён в report-only (`git diff`, незакоммичено); в CI `claims_ledger` не вызывается вообще; сам `--check` падает только на «нет ни одной строки evidence», а не на несуществующий путь. Регенерация: `FAIL — 373 new claim(s) name no evidence`, exit 1, 9 «mismatches» (7 из 9 — явный миспарс парсера, 2 — кандидаты в устаревшие числа). **[M]**

---

## 1. Leakage (Q1)

### 1.1 The direct hit — `1.4621` is evaluated on a training source

The doc says the point is `fwedu, off=100M, 196,563 B`
(`docs/records/research/2026-11-08-efficiency-frontier.md:68,157`). There is no artifact for it
(`:157`; zero `eval_offset=100000000` rows exist in any JSONL — see §2). But the two
distinguishing numbers are a fingerprint:

| test | result | source |
|---|---|---|
| `web_eval.bin` has no sidecar; a doc-masked eval on it cannot score 196,563 | 0 masked positions, flat | `web_eval.bin`, `data.py:1010-1027` |
| `best/fwedu.bin` + sidecar + `doc_masking=true`, `off=100,000,000`, 24×8×1024 | **scored 196,563, masked 45** | **[M]** ran `MaskedByteStream` over `fwedu.bin` |
| first fwedu document start after 100M | 100,014,698 | `fwedu.bin.docs.json` starts **[M]** |
| fwedu is a `[sources]` entry of both live plans | yes | `configs/mix_v7.toml:34`, `configs/mix_v7_docmask.toml:24` |
| 0.25 GB flat run, fwedu bytes drawn (seeds 1/2/3/20261027) | 109.47–110.55 MB | `per_source_bytes` / `replay_bytes` **[C]** |
| 0.25 GB doc-masked run, fwedu cursor after its 14,662 steps (seed 1; 14,641–14,789 across seeds) | **109.02 MB** (108.86–109.97 MB) | `DocPackedStream.skip_steps` **[M]** |
| eval window `[100.000, 100.1968) MB` is inside that prefix | **yes, every seed** | **[C]** |

**Вердикт: `1.4621` — не held-out число.** Если это действительно 0.25 ГБ прогон на
fwedu, окно eval лежит в первых 109–110 МБ fwedu, которые тот же прогон прочитал.
Даже без прогона: `fwedu.bin` — тренировочный источник, поэтому eval на нём не
является held-out по построению. Под текущим `train()` такой eval был бы отвергнут
(`eval_train_overlap` по inode: `data.py:979-998`, вызов `loop.py:1201-1209`);
`scripts/eval_window.py` этот гейт **не** вызывает (`:51-70`) — им можно спокойно
посчитать training-байты. Это и есть механизм, которым такое число могло появиться.

**Следствие для документа.** `efficiency-frontier.md:165` объясняет разрыв 0.057
«трудностью окна». Это не проверено и, вероятно, неверно: окно на тренировочных
байтах должно быть *легче*, а не труднее. Сравнивать `1.4621` с `1.519` нельзя ни
по файлу, ни по смещению, ни по held-out статусу. Разница в scored bytes —
`196,608 − 196,563 = 45 B`; разница в файле — `web_eval.bin` vs `fwedu.bin`; разница
в offset — 0 vs 100,000,000.

### 1.2 `web.bin` → `web_eval.bin`: the 5.9 KB duplicate

`web_eval.bin` was carved from source shard `web_04108.txt`; `.web.state.json` says
`train_units=373`, `eval_units=1`, `method=doc_sha1_disjoint`, `overlap=0`, `pass=true`.
That check is at **shard** granularity: it hashes the 6.29 MB source file, not the
articles inside it. Byte-level result on the actual eval window:

| metric | value | source |
|---|---:|---|
| eval content also present in `mix4/web.bin` | **9,322 B / 196,800 = 4.74 %** | **[M]** 15 union intervals |
| largest contiguous run | **5,948 B** | eval `[118,841,124,789)` = `web.bin [460,404,765,460,410,713)` |
| same, in the first 65,536 B | 1,988 B = 3.03 % | **[M]** |
| text of the largest run | Cotton Mather, *Wonders of the Invisible World* (public-domain book), shared by two shards | **[M]** |

`web.bin` is a training source of `configs/mix_v4.toml:69`; `flagship1` trained on it
(`checkpoints/flagship/flagship1.config.toml`: `data=mix4/web.bin`, `eval_data=mix4/web_eval.bin`,
`steps=234000`, batch 24, s1024; `grad_accum` absent = default 1) for
234,000 × 24 × 1024 = 5.75 GB of mixture, of which `web` is 2.349 GB — **0.999
epoch of the 2.35 GB file** (`per_source_bytes`; 3× that if the run used
`grad_accum=3`, which the snapshot does not say). Either way the cursor passes
`web.bin` offset 460,404,765, where the 5,948 B duplicate sits. It scores
`over 196608 B (8 x 24 x 1024)` (`flagship1.log:17`). So every scored
position of the 4.74 % was trained on, and `doc_sha1_disjoint` does not say otherwise.

**Вердикт: для `mix_v4`/`web.bin`-прогонов held-out окно заражено на 4.74 %,
включая 5.9 КБ непрерывного текста.** Это не текущая v7-линия: `flagship2`,
`dense103`, `final_dense` и `flagship.pt` обучались на `mix_v7` (12 источников,
`web.bin` там нет — `flagship2_stdout.log:4`, `dense103_stdout.log:4`,
`final_dense_stdout.log:4`), поэтому сравнение `2026-11-06-bpb-accounting.md`
(`flagship2` vs `flagship.pt`) этим leak'ом **не** заражено. Заражены
`mix_v4`-прогоны, в частности `flagship1.log` (105M, mix_v4, web.bin); его
собственные eval-числа не являются held-out на 4.74 %. (Док
`2026-11-05-architecture-review-dense.md:85,772` упоминает `flagship1.log`
только как другой баннер, не как источник BPB; число 1.734 там — из
`flagship_resume.log`.) Полный список прогонов на `mix_v4` не отделён от
`mix_v7` в логах: `flagship1.log:4` — `mix_v4`, `flagship2_stdout.log:4` —
`mix_v7`.

Отдельно: `2026-11-06-datapath-audit.md:117-127` объявляет **«Overlap: 0. [M] …
No leak»** по 1,000 случайным 64-байтовым окнам и двум 4 КБ блокам. Этот метод
не имеет мощности найти 5.9 КБ run на смещении 118,841 (вне его 65,599-байтового
диапазона) и 1,988 B разбросанных run'ов внутри него; точный скан находит
и то, и другое. «No leak» — **ложноотрицательный вывод выборки**, а не измерение.

### 1.3 The v7 mixture: content overlap vs bytes actually seen

Same method, the 9 sources of `mix_v7_docmask` (the 12-source `mix_v7` adds
wiki_en/reasoning/rudialog, all 0 B of hits in this window):

| source | union in eval window | max run | seen by 0.25 GB (flat) | seen by 983 MB (docmask, upper bound) | seen by 4.2 GB (flat, 65,536 window) |
|---|---:|---:|---:|---:|---:|
| fwedu | 2,190 B | 416 B | 182 B | 182 B | 283 B |
| finemath | 797 B | 351 B | 0 | 0 | 0 |
| science | 378 B | 378 B | 0 | 0 | 216 B |
| math | 101 B | 101 B | 0 | 0 | 0 |
| cosmo / ru / code / wiki_ru / instruct | 0 | 0 | 0 | 0 | 0 |
| **union, v7 sources** | **2,735 B = 1.39 %** | 416 B | **182 B = 0.093 %** | **250 B = 0.127 %** | **499 B = 0.761 %** of the 65,536 window |

The "seen" column is exact for flat streams (the `ByteStream` cursor is the
per-source draw) and an **upper bound** for doc-masked streams (`DocPackedStream`
consumes a variable number of source bytes per row, `data.py:600-608`; the flat
`batch×(seq_len+1)` count is the maximum). The source offsets of the fwedu overlap
are 78.7 MB (182 B), then 175/195/208 MB, 801 MB, 1.33/1.58 GB, and 2.4–11.6 GB;
that is why the seen fraction grows with the budget.

**Вердикт: контентное перекрытие v7 — 1.39 % (нижняя граница), но реально
прочитано 0.09 % на 0.25 ГБ, 0.13 % на 983 МБ и 0.76 % на 4.2 ГБ.** Для абсолютного
BPB это ≤0.06 BPB сверху (0.76 % × 8) и реалистично ~0.015; для A/B на одном окне
эффект общий и в дельту не входит. На 0.25 ГБ он пренебрежим, но **не ноль**.

### 1.4 What the prior leakage audit got right, and what it missed

`docs/records/reviews/2026-11-07-leakage.md:137-157` (F4) reports:

| claim | prior audit | this audit | verdict |
|---|---:|---:|---|
| total exact overlap, `[0,196800)` | 2,241 B = 1.14 % | **2,735 B = 1.39 %** | prior is a sampled lower bound (1,024 random 64 B windows); it undercounts by ~18 % |
| fwedu | 1,688 B, max 414 B | **2,190 B, max 416 B** | prior undercounts fwedu by 23 % |
| finemath / science / math | 730 / 378 / 101 B | 797 / 378 / 101 B | same shape |
| "at the current budget it is 0 … the model has not seen those bytes" | 0 | 0 at 2k steps; **182 B at 0.25 GB; 250 B at 40k; 499 B at 171k** | **contradicted as an absolute**; the conclusion is still small, not zero |
| method's scope | mix_v7 only | mix_v7 + `web.bin` | **`web.bin` was never measured**; it is the 4.74 % hit above |
| `2026-11-06-datapath-audit.md:117-127` "Overlap: 0 … No leak" | 1,000 random 64 B windows + two 4 KB blocks | exact 64 B anchors | **contradicted**: 1,988 B in the very `[0,65,599)` range it sampled, and 9,322 B in `[0,196,608)`; the sampling had no power for a 5.9 KB run at 118,841 |

F4's honest caveat ("verbatim ≥64 B, a lower bound") stands. The absolute sentence
"the model has not seen those bytes" does not: it is a budget-specific 0 that was
generalised to 40k steps, where the fwedu cursor is past 78.7 MB and 250 B are seen.

### 1.5 The no-leak gate is identity-only

`eval_train_overlap` maps the eval span and every train file by `(st_dev, st_ino)`
(`data.py:979-998`). It catches the same file under two paths. It does **not**
catch:

* a copy of the eval file with a different inode (the doc itself says this);
* a shared article across two different shards (`web.bin`/`web_eval.bin`);
* an eval on `fwedu.bin` at an offset inside the training range if the file is
  opened through `eval_window.py`, which never calls the gate.

**Вердикт: гейт закрывает класс «--eval указывает на --data», но не класс
«held-out карвирован из того же источника» и не класс «eval_window.py».**

---

## 2. The eval window (Q2)

### 2.1 Which runs used which window

All 1,675 JSONL eval rows in `/home/sehaxe/logs/`: **1,596 have no offset field**,
**79 have `eval_offset`**, and **all 79 are `0`**, all `eval_bytes=196608`, all
`scored=196608`, across 34 files. There is **no** logged row at `off=100000000`
(`grep -r "100000000"` over logs: zero; over docs: only the efficiency-frontier
doc). The task's premise "the ledger has numbers from off=0 and off=100000000"
is **false for artifacts**: the ledger has no `eval_offset` column and no
`100000000` string at all; the off=100M number lives only in
`docs/records/research/2026-11-08-efficiency-frontier.md`, written after the ledger.

| run family | eval file | offset | scored B | geometry | mixture / masking | evidence |
|---|---|---:|---:|---|---|---|
| `flagship3` (1.519/1.660) | `mix4/web_eval.bin` | 0 | 196,608 | 24×8×1024 | `mix_v7_docmask`, mask=true | `flagship3.log:2-3,15,31`; `flagship3.jsonl` |
| `f3short` (1.927) | `mix4/web_eval.bin` | 0 | 196,608 | 24×8×1024 | `mix_v7`, mask=false | `f3short.log:2,49`; `f3short.jsonl:48` |
| `sweep_mult` m8/m12/m16 | `mix4/web_eval.bin` | 0 | 196,608 | 24×8×1024 | `mix_v7_docmask`, mask=true | `sweep_mult/*.log`, `*.jsonl` |
| `docmask` A/B | `mix4/web_eval.bin` | 0 | 196,608 | 24×8×1024 | masked=v7_docmask/mask; unmasked=v7/flat | `docmask/*.jsonl` |
| E1 v4 bytes | `mix4/web_eval.bin` | 0 | 196,608 | 24×8×1024 | `mix_v7`, flat | `e1v4/bytes_s1.jsonl` |
| E1 v4 patch | `mix4/web_eval.bin` | 0 | 196,608 | 6×32×1024 | `mix_v7`, flat | `e1v4/patch_s1.jsonl` |
| owner `1.4621` | **`best/fwedu.bin`** | **100,000,000** | **196,563** | 24×8×1024 | doc-masked (45 starts) | **[M]** fingerprint; no artifact |
| `flagship1` (mix_v4) | `mix4/web_eval.bin` | 0 | 196,608 | 8×24×1024 | `mix_v4` incl. `web.bin` | `flagship1.log:17` |
| `flagship2`/`dense103` | `mix4/web_eval.bin` | 0 | 65,536 | 8×8×1024 | `mix_v7` | `flagship2.jsonl`, `dense103.jsonl` |

**The E1 geometry is not a different window.** At batch 8 (24×8×1025 = 196,800
source bytes) and batch 32 (6×32×1025 = 196,800), the flat reader skips one
successor byte per 1,025-byte row; the union of scored positions is the same
192 rows × 1,024 = 196,608 in both. `flagship1`'s 8×24×1024 is likewise 196,608
but the padding of the batch axis differs only in row count; the same flat
arithmetic applies. So the *scored positions* are comparable across batch sizes;
the eval **read span** is 196,800 B, not 196,608 (the `eval_bytes` label is the
prediction count, `data.py:1067-1082`).

### 2.2 Cross-run comparisons that mix windows

| doc claim | run A | run B | bytes differ | file/offset differ | verdict |
|---|---|---|---:|---|---|
| `efficiency-frontier.md:156-157,165` `1.519` vs `1.4621` | `flagship3` 1.51858 (196,608, web_eval, off=0) | owner 1.4621 (196,563, fwedu, off=100M) | **45 B** | **yes: file + offset + mask** | **invalid as a curve**; the doc flags the window but not that B is a training source |
| `efficiency-frontier.md:163` "the two run lines differ only in steps" | `f3short` 1.927 (mix_v7 12 src, flat) | `flagship3` 1.660 (mix_v7_docmask 9 src, mask) | 0 | same eval window | **contradicted**: mixture + masking + Muon fix all differ (`f3short.log:2` vs `flagship3.log:2-3`) |
| `architecture-frontier.md:19,125` "1.927 @98.3 MB → 1.519 @268 MB, 0.408 = 89 %" | `f3short` 1.927 (pre-fix) | `flagship3` 1.519 (post-fix) | 0 | same eval window | **arithmetic right, curve wrong**: 1.927 is a retracted-class point (pre-`0a7dfcd`); the corrected same-run pair is 1.660 → 1.519 = 0.141. Also V=268 MB is not the run: step 10000 is 245.76 MB |
| `the-16x.md:11` "1.519 (today's 0.25 GB point on `fwedu` off=0)" | — | `flagship3` 1.519 | — | **eval file is `web_eval.bin`, not fwedu** | **contradicted** |
| `docmask-ab.md:6-9` masked vs unmasked | 2.2941/2.0406 | 2.2801/2.0236 | 0 | same eval file/offset | **window matches**, but the arms differ in the mixture (9 vs 12 sources) *and* masking, 1 seed each |
| `control-transformer.md:16` 1.492 vs block 1.618/1.735 | control 65,536 | block 65,536 | 0 | same | window matches; the control number has no persisted log (§5) |

**Вердикт: в живых доках есть ровно одна кросс-оконная пара — `1.519` vs `1.4621`;
она отличается на 45 scored байт, но сильнее — файлом, offset'ом и held-out
статусом. Остальные сравнения живого лейна используют одно и то же окно
196,608 B @ off=0.** Старые доки (65,536/81,920/20,480) смешивают окна, но
AGENTS §2.6 и bpb-accounting это явно проговаривают.

---

## 3. The mixture, as executed (Q3)

`per_source_bytes` (`mixture.py:412-431`) was run for the actual plans and budgets.
`window = batch×(seq_len+1) = 8,200`; the stage weights are a probability vector and
each stage sums to 1.000000 (`mix_v7.toml:47-88`, `mix_v7_docmask.toml:30-79`; the
parser refuses otherwise, `mixture.py:257-263`). The 0.25 GB E1 geometry is
`steps=30,517, batch=8, seq_len=1024, grad_accum=1`; `micro_steps` is exactly
`steps×grad_accum` (`loop.py:1176-1190`).

| source | file size | bytes @0.25 GB (mix_v7, flat) | epochs | bytes @0.25 GB (mix_v7_docmask) | epochs | bytes @983 MB (docmask) | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| fwedu | 12.00 GB | 110,005,624 | 0.0092 | 120,340,607 | 0.0100 | 473,205,872 | 0.0394 |
| cosmo | 3.00 GB | 26,125,036 | 0.0087 | 28,506,397 | 0.0095 | 112,093,650 | 0.0374 |
| ru | 3.00 GB | 32,756,417 | 0.0109 | 35,800,025 | 0.0119 | 140,773,756 | 0.0469 |
| code | 690.0 MB | 24,373,106 | 0.0353 | 26,142,104 | 0.0379 | 102,797,728 | 0.1490 |
| math | 632.5 MB | 9,258,784 | 0.0146 | 9,931,796 | 0.0157 | 39,054,445 | 0.0617 |
| finemath | 1.50 GB | 7,807,384 | 0.0052 | 8,371,348 | 0.0056 | 32,918,449 | 0.0219 |
| science | 2.00 GB | 9,258,784 | 0.0046 | 9,931,796 | 0.0050 | 39,054,445 | 0.0195 |
| instruct | 120.5 MB | 5,084,754 | **0.0422** | 5,484,316 | **0.0455** | 21,566,071 | 0.1790 |
| wiki_ru | 230.0 MB | 5,240,038 | 0.0228 | 5,731,013 | 0.0249 | 22,535,583 | 0.0980 |
| reasoning | 176.6 MB | 6,846,475 | 0.0388 | — | — | — | — |
| rudialog | 119.6 MB | 2,897,823 | 0.0242 | — | — | — | — |
| wiki_en | 466.2 MB | 10,585,175 | 0.0227 | — | — | — | — |
| **sum** | | **250,239,400** | | **250,239,402** | | **983,999,999** | |

**No source exceeds 1.0 epochs at 0.25 GB; the maximum is instruct 0.0422 (flat) /
0.0455 (docmask).** Even the full 4.2 GB `flagship2` budget tops out at instruct
0.709 / code 0.594 / reasoning 0.652 epochs (`per_source_bytes` at 513,000 micro-steps).
The 4-epoch cap is never approached; the lane is data-starved, not repeating.

Two definitional corrections a reader needs:

* the "0.25 GB" run reads **250,239,400 source bytes**, not 250,000,000: the `+1`
  per row is the successor byte (`data.py:77-86`). The header's `bytes_seen` is
  optimizer-step arithmetic (`loop.py:1101`), not source bytes.
* under `doc_masking=true`, a row advances by the whole document length (or
  `seq_len+1` for a long doc), not by `batch×(seq_len+1)` (`data.py:600-608`).
  The `bytes_seen` counter therefore counts **positions**, including padding. The
  m8 run's own counters: 72,000 windows, `docmask_scored=65,600,031`,
  `docmask_masked=8,127,969` → **11.0 % of the "16.4 MB" is padding/masked**
  (`sweep_mult/m8_s1.log`, final `doc_masking:` line). The docmask A/B compares
  two arms at equal *positions*, not equal target bytes. `check_epoch_cap` uses
  the flat window as an upper bound, so it is conservative (correct direction),
  but the "equal bytes" language in the docs is not exact.

**Вердикт: ни один источник не превышает 1.0 эпохи на 0.25 ГБ; веса стадий
суммируются в 1.0; главная неточность — `bytes_seen` при `doc_masking` считает
позиции, а не исходные байты.**

---

## 4. Document sidecars (Q4)

### 4.1 The nine live sources: boundaries are correct

`build_doc_sidecar.py` verifies every replayed record byte-for-byte at the cursor
and refuses if `sum(record lengths) != file size` (`:145-192`). The five
`best/*` sidecars are not built by that script (no `builder`/`method` key), but
their counts match the build manifests exactly:

| sidecar | file size | sidecar size | starts | manifest/audit count | match |
|---|---:|---:|---:|---:|---|
| fwedu | 11,999,999,974 | 11,999,999,974 | 2,374,741 | 2,374,741 | yes |
| cosmo | 2,999,999,923 | 2,999,999,923 | 799,484 | 799,484 | yes |
| ru | 2,999,999,758 | 2,999,999,758 | 451,353 | 451,353 | yes |
| finemath | 1,499,999,641 | 1,499,999,641 | 279,234 | 279,234 | yes |
| science | 1,999,999,952 | 1,999,999,952 | 1,507,368 | 1,507,368 | yes |
| code | 690,009,645 | 690,009,645 | 138,871 | 138,871 | yes |
| math | 632,508,389 | 632,508,389 | 47,986 | 47,986 | yes |
| wiki_ru | 230,028,784 | 230,028,784 | 29,943 | 29,943 | yes |
| instruct | 120,500,114 | 120,500,114 | 446,511 | 446,511 | yes |

`docmask_audit.py` reports 9/12 REAL, 23.17 GB = 96.8 % (`docmask-v7.md:26-38`).
Across **all** starts of these nine files, **0** land on a UTF-8 continuation
byte; the sampled 20 are all preceded by `\n\n` (or `\x03` for instruct) and
followed by text/`\x01`/`#`:

| source | start | prev 16 B | next 16 B | utf-8 boundary | verdict |
|---|---:|---|---|---|---|
| fwedu | 6,316 | `b' IDisposables.\n\n'` | `b'CANUSWEST and CA'` | yes | doc start |
| fwedu | 2,397,496,554 | `b'ate referrals.\n\n'` | `b'Immigration has '` | yes | doc start |
| fwedu | 4,800,944,120 | `b'healthy aging.\n\n'` | `b'Fair use is hard'` | yes | doc start |
| fwedu | 7,204,298,640 | `b'ood to Have\xe2\x80\x99\n\n'` | `b'Traditional inte'` | yes | doc start |
| fwedu | 9,604,145,844 | `b' for disaster.\n\n'` | `b'These plants liv'` | yes | doc start |
| code | 5,522 | `b'Child(link);\n}\n\n'` | `b'import React fro'` | yes | doc start |
| code | 225,154,164 | `b"vex.module';\n\n\n\n"` | `b"if('serviceWorke"` | yes | doc start |
| code | 455,377,026 | `b"s]' % elapsed\n\n\n"` | `b'package contentf'` | yes | doc start |
| math | 9,558 | `b'smc.vnet.net>\n\n\n'` | `b'\nThis content fo'` | yes | doc start |
| math | 211,517,525 | `b'ing strategy.\n\n\n'` | `b'\n### Post: #5\n\n*'` | yes | doc start |
| math | 420,218,590 | `b'ed on page 2.\n\n\n'` | `b'\nThis session em'` | yes | doc start |
| instruct | 402 | `b'ist philosophy.\x03'` | `b'\x01\xd0\x9c\xd0\xbe...'` | yes | doc start |
| instruct | 40,150,672 | `b'considerations.\x03'` | `b'\x01\xd0\x9a\xd0\xb0...'` | yes | doc start |
| instruct | 80,333,328 | `b' (died in 1945)\x03'` | `b'\x01Implement a con'` | yes | doc start |
| cosmo | 4,328 | `b'es and beyond.\n\n'` | `b' Introduction:\nI'` | yes | doc start |
| cosmo | 1,500,966,925 | `b'clusive world.\n\n'` | `b' Welcome to our '` | yes | doc start |
| ru | 11,897 | `b'\xb7\xd0\xb0\xd0\xbf\xd0\xb8\xd1\x81\xd0\xb8 \xc2\xbb\n\n'` | `b'\xd0\x9c\xd1\x83\xd0\xbb...'` | yes | doc start |
| ru | 1,362,292,552 | `b'\xb8\xd0\xb2\xd0\xb0\xd0\xbd\xd0\xb8\xd0\xb5\xd0\xbc.\n\n'` | `b'\xd0\x9d\xd0\xb0\xd1\x88 \xd0\xb8...'` | yes | doc start |
| science | 2,433 | `b' well applied.\n\n'` | `b'Appraisal of the'` | yes | doc start |
| wiki_ru | 18,953 | `b'\xd1\x82\xd1\x80\xd0\xb0\xd1\x87\xd0\xb5\xd0\xbd\xd1\x8b\n\n'` | `b'# \xd0\xa0\xd0\xb0\xd1\x82...'` | yes | doc start |
| **web(bogus)** | 6,295,756 | `b'or another time.'` | `b'Amateur radio op'` | yes | shard seam, by luck a period |
| **web(bogus)** | 1,172,465,307 | `b'all so indicate.'` | `b'|Wednesday, 2 Ju'` | yes | **MID-TABLE** |

**Вердикт: девять используемых сайдкаров размечают документы правильно.** Счёт
совпадает с манифестами, все границы — UTF-8 boundary, префикс — `\n\n`/`\x03`,
суффикс — текст/`\x01`/`#`. Ошибки в них не найдено.

### 4.2 The latent wrong sidecar: `mix4/web.bin.docs.json`

`web.bin.docs.json` still exists as a live `.docs.json` with
`method="shard-head-tail"`, 373 starts for a 2,350,670,235 B file → mean 6.3 MB.
`docmask_audit.py` would call it BOGUS (`REAL_MAX_MEAN=20 KB`, `:35,118-127`), and
the second sampled seam is mid-table. But **`load_doc_starts` does not apply the
audit's mean/count rule**: it only checks version/size/ascending/bounds
(`data.py:457-514`). So a `doc_masking=true` run on `configs/mix_v4.toml` (whose
`[sources]` include `web = mix4/web.bin`, `:69`) would *accept* this sidecar and
mask at shard seams, not document boundaries. The docmask plan avoids it only
because `mix_v7_docmask` does not include `web`.

**Вердикт: для живого плана — правильно; `web.bin.docs.json` — латентная
неверная маска, которую загрузчик не отличит от настоящей.** Либо удалить/переименовать
его в `.bogus`, как сделали с wiki_en/reasoning/rudialog, либо перенести проверку
`docmask_audit` в `load_doc_starts`.

### 4.3 The eval sidecar policy

`web_eval.bin` has no sidecar, so every live eval prints
`eval_docmask=flat (no sidecar)` (`flagship3.log:10`) and scores the flat window.
The JSONL `eval_row` (`loop.py:1096-1110`) does **not** record the eval file path
or the `doc_boundary_policy`; only the text log does. A JSONL-only reader cannot
tell a masked eval from a flat one. For the current runs this is harmless (all
flat), but it is an instrumentation gap that the claims ledger inherits.

---

## 5. The claims ledger and its gate (Q5)

### 5.1 Regeneration

Command (read-only; output to `/tmp/opencode/leak/`):

```
cd port && PYTHONPATH=src .venv/bin/python scripts/claims_ledger.py \
  --docs ../docs --repo .. --today 2026-11-07 --check \
  --out /tmp/opencode/leak/ledger.md --csv /tmp/opencode/leak/ledger.csv
```

| metric | ledger doc (2026-11-07) | regenerated 2026-11-08 | verdict |
|---|---:|---:|---|
| claims | 1,627 | **1,875** | stale |
| derived | 12 | **32** | stale |
| unverifiable | 1,606 | **1,834** | stale |
| retracted | 9 | 9 | same |
| mismatches | **0** | **9** | **contradicted** |
| gate | FAIL — 140 new claims | **FAIL — 373 new claims name no evidence; exit 1** | **worse, not fixed** |

The 9 "mismatches" are mostly parser artefacts, not 9 wrong numbers: e.g.
`docs/records/reviews/2026-11-06-optimizer-audit.md:38-41` has "286 / 558 / 1 / 557" in a
sentence parsed as four `params` claims against 102,678,943, and
`efficiency-frontier.md:275` parses the Δ `−0.107 BPB` as an absolute value
(derived 1.51858). So the mismatch counter is not a trustworthy integrity signal;
the stale header "0" is still wrong.

### 5.2 The gate is disabled, and its rule is too weak

* `scripts/gate_fast.sh` in the working tree is **modified** (`git status: M`):
  the committed HEAD runs `claims_ledger.py --check` and fails; the working tree
  changes it to `--no-params ... | grep`, "report only", with `|| true`
  (`gate_fast.sh:33-45`). The comment's "140 claims dated 2026-11-06/07" is stale
  (now 373), and "the blocking version runs in CI" is **false**: no workflow in
  `.github/workflows/` references `claims_ledger` (`ci.yml` only has a comment
  containing the word "claims").
* Even the committed `--check` is weak: the gate list is
  `status == UNVERIFIABLE and not c.evidence` (`claims_ledger.py:1499-1506`).
  A claim that names a **nonexistent** path still has `evidence` non-empty and
  passes. The doc says this explicitly ("an UNVERIFIABLE claim that names an
  artifact the ledger cannot read passes").
* The verifier reads `fin`/`loss` and `step` from the JSONL but **never reads
  `eval_offset`/`eval_bytes`**, and the JSONL does not carry the eval file or
  policy. It cannot detect the `1.519` vs `1.4621` window/file mismatch even in
  principle.

**Вердикт: claims-ledger — список, не гейт. В рабочем дереве он отключён, в CI
его нет, а правило проверяет наличие строки evidence, а не существование
артефакта. 373 новых утверждения без evidence; 9 «mismatches» — в основном
миспарс.**

### 5.3 Ten live claims, checked against the artifact they name

| # | claim (file:line) | evidence it names | what the artifact says | verdict |
|---:|---|---|---|---|
| 1 | `efficiency-frontier.md:156` 1.519 @ 246 MB | `logs/flagship3.log:31` | 1.5185367, step 10000, off=0, 196,608 B | **verified [M]** |
| 2 | `efficiency-frontier.md:155` 1.660 @ 98 MB | `logs/flagship3.log:15` | 1.6598098, step 4000 | **verified [M]** |
| 3 | `efficiency-frontier.md:154` 1.927 @ 98 MB | `logs/f3short.log:49` | 1.9271916, step 4000; mtime 11-06 22:11 < `0a7dfcd` | **number verified; retracted class [M]** |
| 4 | `efficiency-frontier.md:157` 1.4621 @ 250 MB | **none** ("owner-supplied") | no JSONL with off=100M; the only fingerprint is `fwedu.bin` doc-masked, a training source inside the 0.25 GB prefix | **unverifiable; contradicted as held-out [M]/[C]** |
| 5 | `efficiency-frontier.md:163` "differ only in steps" | `f3short.log`, `flagship3.log` | f3short: `mix_v7` 12 src, no mask; flagship3: `mix_v7_docmask` 9 src, mask=true | **contradicted [M]** |
| 6 | `docmask-ab.md:8-9` 2.2941/2.2801 | no log named | `docmask/masked.jsonl` 2.2940966; `unmasked.jsonl` 2.2800678; 196,608, off=0 | **numbers verified [M]; ledger calls them UNVERIFIABLE; +0.017 is confounded (9 vs 12 sources + masking, 1 seed) [M]** |
| 7 | `docmask-ab.md:16` E1 v3 2.12 vs v1 1.63 @125 MB | no log named | `e1v3/killed/bytes_s1.jsonl` 2.1245728 at step 15258; `e1/bytes_s1.jsonl` 1.6338982 at 15258 | **verified [M]; n=1 each** |
| 8 | `the-16x.md:11` "1.519 … on fwedu off=0" | none | `flagship3.log` runs `mix_v7_docmask` and evals `mix4/web_eval.bin`; `eval_docmask=flat (no sidecar)` | **contradicted [M]** |
| 9 | `tsct-ceiling.md:280` night_stage3 adam 1.8947 / m12 1.7478 / m32 1.9942 (2 seeds, 2k) | `~/logs/night_stage3.log` | adam mean 1.89466 spread 0.03246; m12 1.74777 spread 0.00706; m32 1.99423 spread 0.04322 | **verified [M]** |
| 10 | `architecture-frontier.md:19,125` 1.927→1.519 = 0.408 = 89 % of literature; V=268 MB | `f3short`, `flagship3` numbers | 0.408 arithmetic right; 1.927 is pre-fix; actual V=245.76 MB; corrected same-run 1.660→1.519 = 0.141 | **arithmetic right, curve/verdict contradicted [C]** |
| 11 | `efficiency-frontier.md:159` "116M / 1.356 @500 MB — NOT FOUND" | none | `~/logs/ctrl896.log`: 116,109,056 params, 499,998,720 B, final `bpb=1.3671`; `+517.9 %` FLOPs/byte vs flagship | **contradicted: the artifact exists, 0.011 above the quoted 1.356, and is not compute-matched [M]** |
| 12 | `control-transformer.md:16` 1.492 @500 MB | `scripts/control_transformer.py` (no log) | the 13.5M control's `/tmp/c*.json` are gone; only the doc's table remains; the persisted control log is `ctrl896.log` (a different, 116M, non-matched run) | **unverifiable as recorded [M]** |

**Вердикт: из 12 проверенных живых утверждений 6 подтверждены числом (1, 2, 3,
6, 7, 9), 5 опровергнуты (4, 5, 8, 10, 11), 1 недоказуемо (12); 6 и 10 —
подтверждённая арифметика с неверной атрибуцией.** Ошибки — в том, *что именно*
измерено, а не в делении на `ln 2`.

---

## 6. Seed noise (Q6)

All arms below are `flagship3_dense`, `mix_v7*`, b8 × s1024, `opt=mix`, `lr=3e-4`,
same eval file/offset. "Spread" is max−min of the arm's own seeds (the repo's
protocol wording); pooled 1σ is `sqrt(mean(spread²/2))` over the arms at that
budget, which is the right unit for a two-seed arm.

| budget | arm | seeds | BPB values | spread | pooled 1σ |
|---|---:|---:|---|---:|---:|
| 36.864 MB (1500 steps) | m8 / m12 / m16 | 2 / 2 / 2 | 1.92582,1.96145 / 1.85420,1.88145 / 1.87100,1.98759 | 0.03563 / 0.02725 / **0.11659** | **0.051** |
| 49.152 MB (2000 steps, `night_opt3`) | AdamW / m12 / m32 | 2 / 2 / 2 | 1.91089,1.87843 / 1.75130,1.74424 / 2.01584,1.97262 | 0.03246 / 0.00706 / 0.04322 | 0.022 (3 arms) |
| 49.152 MB (`night_opt`) | mix / AdamW / lr3e-4 | 2 / 2 / 2 | 2.17838,2.18419 / 1.92429,1.94641 / 1.88222,1.90926 | 0.00582 / 0.02212 / 0.02705 | 0.019 (6 arms) |
| 73.728 MB (3000 steps) | m8 / m12 / m16 | 2 / 2 / 2 | 1.67638,1.69080 / 1.67563,1.68429 / 1.69106,1.71329 | 0.01441 / 0.00866 / 0.02223 | **0.011** |
| 73.728 MB | AdamW / mix (`night_opt`) | 2 / 2 | 1.76814,1.77320 / 2.01497,2.02201 | 0.00506 / 0.00704 | — |
| 16.384 MB | docmask masked vs unmasked | **1 / 1** | 2.0406 / 2.0236 | — | — |
| 0.25 GB | E1 bytes / patch | **1 / 1** | 1.49461 / 3.30069 | — | — |

**E1 has no multi-seed arms at all.** `e1/bytes_s2.jsonl` contains only a `run`
header; `e1v2/killed/` and `e1v3/killed/` contain only `bytes_s1`; `e1v4/` contains
`bytes_s1` and `patch_s1`. The only same-budget E1 comparison is 1 seed each. The
task's premise "the E1 arms have multiple seeds" is **contradicted by the
artifacts**. The E1 script default is `SEEDS=2`, but the v4 invocation ran 1; its
own comment justifies that only for a gap outside the 0.02/0.05 band
(`e1_bytes_vs_patch.sh:58-60`), and the patch arm's gap is 1.806 — far outside.

**A/B verdicts inside the seed noise:**

* `docmask-ab.md:11` "masking costs +0.017 BPB" — **inside** the 16 MB noise (the
  nearest measured 1σ is 0.051 at 37 MB; 16 MB is noisier), and the arms differ in
  the mixture (9 vs 12 sources) as well as masking. The doc says "within noise";
  the sentence "Masking costs +0.017" is still a causal attribution the design
  cannot support.
* m8 vs m12 at 73.7 MB: mean difference **0.0036**, 0.3σ — **inside**. `goal.md:22-23`
  correctly calls it a "plateau" and fixes 12; that is a choice, not a measured
  optimum. The m16 vs m12 difference 0.0222 is 1.95σ and exactly the m16 arm's own
  spread; **not a clean verdict** either.
* m12 vs AdamW at 49 MB: 0.147 = 6.6σ (pooled 0.022) — **outside**, but only 3 seeds
  per arm are the protocol; 2 are enough here.
* `the-16x.md` safe rung `+0.006` (bf16 NS) and `+0.008` (bf16 state) — **inside**
  16 MB noise; the doc says so.
* `the-16x.md` `ns_steps=3 +0.274` — **outside** (5.4σ at 37 MB); this one is real.
* The hybrid pre-registration `≥0.02 BPB` (`hybrid-stack.md:139`) is ~1.8σ at 73.7 MB;
  at 1 GB it should be smaller, but no 1 GB seed data exists yet. **Unverifiable
  until the run.**

**Вердикт: шум сидов — 1σ ≈ 0.011 BPB на 73.7 МБ, 0.019–0.022 на 49 МБ, 0.051 на
37 МБ; на 16 МБ он больше 0.05. Вердикты `docmask +0.017`, `m8≈m12` и
`muon_state/ns bf16 +0.006` лежат внутри; `ns_steps=3 +0.27`, `m12 vs AdamW +0.147`
и E1 `patch −1.8` — снаружи. У E1 нет второго сида, поэтому «seed spread E1» не
существует.**

---

## 7. Direct answers (RU)

1. **Leakage.** Да. Для `web_eval.bin[0:196608)`:
   * текущая линия (`mix_v7`/`mix_v7_docmask`) — **2,735 B = 1.39 %** контента,
     max run 416 B (fwedu/finemath/science/math); **реально прочитано 0 B на
     73.7 МБ, 182 B на 0.25 ГБ, 250 B на 983 МБ, 499 B на 4.2 ГБ**;
   * исторический `mix_v4/web.bin` (тренировочный файл `flagship1`) —
     **9,322 B = 4.74 %**, max непрерывный **5,948 B**; `doc_sha1_disjoint`
     сравнивает шарды, не документы; текущие v7-прогоны на web.bin не учатся;
   * **`1.4621` — eval на `fwedu.bin` off=100M, а окно лежит внутри прочитанных
     109–110 МБ того же 0.25 ГБ прогона.** Это не «другое окно», это training-байты.
2. **Eval window.** Все 79 логированных eval-строк — `off=0`, `196,608 B`; ни
   одной `off=100,000,000`. Окно одно и то же во всех сравнениях живого лейна.
   Единственная кросс-оконная пара — `1.519` vs `1.4621`: **45 scored байт** разницы,
   но разные файлы (`web_eval` vs `fwedu`), offset (0 vs 100M) и held-out статус.
   `the-16x.md:11` ещё и называет 1.519 «fwedu» — это ложь.
3. **Mixture.** На 0.25 ГБ максимум — `instruct` 0.0422 (flat) / 0.0455 (docmask)
   эпохи; **ни один источник >1.0**. Веса всех стадий = 1.000000. `bytes_seen`
   при `doc_masking` считает позиции с padding (11.0 % у m8), а не исходные байты.
4. **Sidecars.** 9 используемых — корректны (20/20 sampled boundaries, счёт =
   манифест, 0 mid-UTF8 на всех starts). `web.bin.docs.json` — латентно неверный
   (373 шард-шва, средний 6.3 МБ), загрузчик его примет.
5. **Claims ledger.** 12 проверенных живых утверждений: 6 подтверждены числом
   (1,2,3,6,7,9), 5 опровергнуты (4,5,8,10,11), 1 недоказуемо (12); у 6 и 10
   верна арифметика, неверна атрибуция. Сам ledger устарел (1,627→1,875 claims,
   0→9 mismatches), гейт отключён в рабочем дереве и отсутствует в CI; `--check`
   падает на 373 claims без evidence, но пропускает несуществующий путь.
6. **Seed noise.** 1σ = 0.011 BPB @73.7 МБ, 0.019–0.022 @49 МБ, 0.051 @37 МБ,
   >0.05 @16 МБ. Внутри шума: docmask +0.017, m8≈m12, bf16-optimizer +0.006/+0.008.
   Снаружи: ns_steps=3 +0.27, m12 vs AdamW +0.147, E1 patch −1.8. У E1 **нет**
   multi-seed артефактов — только seed 1.

---

## 8. What this audit does not cover

* **Edited near-duplicates.** The leak scan finds exact ≥64 B runs; a document with
  one changed sentence is invisible. The numbers are lower bounds, not "no overlap".
* **The fused KDA/GPU path.** CPU-only; the eval EWMA/readout was not re-run on a GPU.
  The BPB arithmetic was checked from the logs, not by an independent forward here.
* **`web.bin`'s 373-shard source list.** The state records the carve; the mount
  `/mnt/e434...` was not re-read to prove which shard `web_04108.txt` is. The
  5.9 KB exact run is sufficient proof of shared content regardless.
* **The 9 "mismatches"** are reported as the tool produced them; 8 are parser
  artefacts (params/Δ), so do not read them as 9 wrong measurements.
* **No artifact for the 1.4621 checkpoint/run** was found; the leak conclusion is
  from the eval fingerprint (off=100M + 196,563) and the training cursor, not from
  the owner's log.

---

**Итог одной строкой:** арифметика проекта — да; held-out статус — нет: один
headline-пункт посчитан на тренировочном `fwedu`, исторический web-лейн заражён
на 4.74 %, а собственный гейт утверждений отключён и не умеет отличать
существующий артефакт от строки в документе.
