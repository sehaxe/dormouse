# 2026-11-08 — deletion audit

**Read-only audit.** No file outside this report was modified. No GPU command was run
(a training run holds the card; `ps` shows `dormouse.cli train configs/arch_dense_byteflow8.toml`).
All line counts are `wc -l` over the working tree at `89cd4ed`; `[C]` = counted,
`[S]` = estimate, `[T]` = paper/derived constant.

**Scope.** `port/src` 48,179 lines, `port/tests` 58,167, `port/scripts` 43,125 (the
brief's 42,928 is 197 lines lower — a counting-method difference, not a different
tree), `crates/` 34,035 Rust lines, 304 Markdown files under `docs/` (the brief's
298 is an older snapshot; 4,249 in the whole tree including `node_modules/`).

---

## Verdict (по-русски, без смягчений)

Репозиторий **не болен мёртвыми функциями**. На уровне символов он почти чист:
`dead_knob_audit.py` даёт 252 поля / 2 `DEAD` / 0 `SHADOWED`, а AST-обход
`port/src` находит всего 14 определений без единого упоминания (из них 6 —
typer-команды, зарегистрированные декоратором). Проблема в трёх других местах:

1. **Целые подсистемы без точки входа.** `selfplay/` (1,420 строк) не импортируется
   ничем, кроме собственного теста; `model/act_ckpt.py` (418) — только тестом и
   bench-скриптом; `post/agent_rl.py` (100) — только тестом. Это не «код, который
   пока не подключили», это код, у которого **нет вызывающего**.
2. **Флаги, которые не включались ни разу.** По 215 Python-конфигам на записи
   (26 пресетов + 153 снапшота прогонов + 36 чекпоинтов): `halt_weight=0` везде,
   `energy_weight=0` везде, `prelude_layers=0` везде, `moe_gather="off"` везде,
   `route_fused=false`, `fused_kda_tail=false`, `moe_shared_compute=false`,
   `moe_offload=false`, `mod_capacity=1.0`, `nope_layers=0`, `attn_gate=false`,
   `ffn_clip=0.0`. Это ~4.4k строк кода и ~3k строк тестов, которые не может
   выполнить ни один конфиг на записи.
3. **Второй тренер на Rust.** `crates/dormouse-train` (10,415) + `dormouse-cli`
   (879) + `dormouse-data` (4,177) = **15,471 строка**, которые дублируют живой
   Python-путь. Rust `dormouse-core` (16,811) — это формульный источник, его
   трогать нельзя; train/cli/data — нет.

Плюс **долг по собственному правилу**: ADR-0002 говорит «tie удаляет механизм»,
а `halt`, `energy`, `prelude`, `JEPA`, `SIGReg` и `moe_gather` имеют
записанные tie/loss-вердикты и всё ещё лежат в дереве. Это ещё ~4.9k строк.

**Честный итог.** «Мёртвого» кода, который можно удалить без любого решения
владельца, — **~2.0k строк**. Ещё ~3.1k — дормантные подсистемы (self-play,
QAT-export). Остальное — либо политика («отклонено ≠ удалено»), либо Rust-дубль.
Если владелец хочет smallest honest diff: четыре коммита на 2,010 строк, ноль
изменений в поведении любого прогона. Если хочет настоящую чистку: ~17k строк
всего (из них 0.7k vendor, остальное Python), и отдельное решение по 15.5k Rust.

---

## Метод и что именно проверено

| проверка | команда | результат |
|---|---|---|
| поля конфига | `port/scripts/dead_knob_audit.py` (CPU) | 252 поля: 250 LIVE, **2 DEAD**, 0 SHADOWED |
| импортный граф | AST по `port/src`, `port/tests`, `port/scripts` | 4 модуля без production-импортёра + пакет `selfplay` |
| символы | AST + подсчёт вхождений по всем `.py` | 14 определений без ссылок; 6 — typer-команды |
| конфиги на записи | 215 Python-конфигов (26 пресетов + 153 лога + 36 чекпоинтов), `tomllib` | таблицы ниже |
| тесты | `pytest --collect-only -q` | 166 файлов, **1,865** `test_*` функций |
| Rust | `cargo metadata`, `cargo check --workspace` | 6 пакетов; **exit 0** (только warnings) |
| Rust-бинарники | `find target -name train/export/serve/generate` | **нет ни одного**; `target/release` собран 2026-09-27, содержит `cublas-poc` и контейнерный образ, но ни одного CLI-бинарника |
| живые тесты | `pytest tests/test_act_ckpt.py tests/test_export.py tests/test_invariants_history.py` | 39 passed, 1 skipped |
| живые тесты | `pytest tests/test_selfplay.py` | 45 passed |
| конфиг/knobs | `pytest tests/test_config.py tests/test_dead_knob_audit.py` | 35 passed |

---

## 1. Dead code: модули без production-вызывающего

Критерий: модуль не импортируется ни одним файлом `port/src` (кроме самого себя),
и его единственные «вызовы» — тест, bench или demo. Тест, который существует
только чтобы тестировать такой модуль, по брифу тоже мёртв.

| path | src | test | script | evidence | verdict |
|---|---:|---:|---:|---|---|
| `port/src/dormouse/selfplay/` (`__init__` 98, `loop` 390, `pool` 400, `reward` 305, `utm` 227) | **1,420** | 706 (`test_selfplay.py`) | 0 | нет `from .selfplay` ни в одном src-файле; только `tests/test_selfplay.py:36-39`; `PLAN.md:351` упоминает self-play как план, но точки входа нет | **DELETE** или добавить драйвер+гейт |
| `port/src/dormouse/model/act_ckpt.py` | 418 | 455 | 206 (`bench_act_ckpt.py`) | единственные импорты — `tests/test_act_ckpt.py:42`, `scripts/bench_act_ckpt.py:46`; `model/act_mem.py:39` ссылается только в докстринге | **DELETE** |
| `port/src/dormouse/post/agent_rl.py` | 100 | ~60 (4 теста в `test_agent.py:367-426`) | 0 | импорт только `tests/test_agent.py:40`; `post/__init__.py` его не импортирует намеренно | **DELETE** |
| `port/src/dormouse/model/export.py` | 542 | 315 | 113 (`qat_export_demo.py`) | импорт только `tests/test_export.py:23`, `scripts/qat_export_demo.py:33`; в `cli.py` команды `export` нет | **DORMANT**: удалить, если QAT-релиз не планируется; иначе оставить и дать CLI-команду |
| `port/src/dormouse/runlog.py` | 166 | 171 | 0 | импорт только `tests/test_invariants_history.py:21`, и тест **использует** reader для аудита лога, а не тестирует его | **KEEP**, лучше перенести в `tests/` |
| `port/src/dormouse/train/__init__.py` | 48 | 0 | 6 | фасад импортируют `scripts/smoke_gpu.py:35`, `ab_muon_variants.py:96`, `bench_opt.py:178`, `smoke_byteflow.py:39` и др. | **KEEP** (это не dead; удаление сломает скрипты) |
| `config.py:1463` `jepa: bool` | 1 | 0 | 0 | dead_knob: DEAD; только consistency-проверка `config.py:1637`; 2 пресета ставят `false` | **DELETE field + check** |
| `config.py:2150` `muon_sf_retract` | 1 | 0 | 0 | dead_knob: DEAD; только собственный отказ `config.py:5232`; вызовов `retract_tsct` нет | **DELETE field + refusal** |
| `vendor/.../burn-eggroll` | 327 | 0 | 0 | нет path-edge в `cargo tree`; `library-crate-fate.md:75` — «REFERENCE», не «WIRE» | **DELETE** |
| `vendor/.../burn-ptrn` | 399 | 0 | 0 | нет path-edge; `library-crate-fate.md:73` — «REFERENCE» | **DELETE** |
| `vendor/.../burn-parcae` | 317 | 0 | 0 | нет path-edge, но `library-crate-fate.md:76` — **WIRE** (записанная причина оставить) | **KEEP** до закрытия причины |
| `crates/cublas-poc` | 828 | 0 | 0 | одноразовый spike 2026-09-27; ссылки только в research-доке; в `target/release` есть бинарник | **ARCHIVE** (не delete: это запись измерения) |

**Итого Tier 1: 2,010 строк без любого решения владельца** (`act_ckpt` 1,079 +
`agent_rl` 160 + два поля 45 + vendor 726). С self-play и export — 5,106.

---

## 2. Код, который ADR-0002 уже убил, но он остался

ADR-0002 (`docs/adr/0002-ab-or-death.md:3`): «Every mechanism must beat its own
removal on BPB, otherwise it is deleted; a tie deletes it too». Вердикты записаны,
код — нет. Это не спекуляция: у каждого механизма есть строка с вердиктом.

| mechanism | модуль | test | scripts | вердикт (docs) | конфигов с ненулевым значением |
|---|---:|---:|---|---|---:|
| halt head | 145 (`model/halt.py`) | 312 | `ab_depth8.py` 450 | `2026-10-23-architecture-verdict.md:50`: +0.026, spread 0.079 — tie | **0/123** логов, 0/2 пресетов |
| energy head | 170 (`model/energy.py`) | 294 | `ab_energy.py` 429, `rerank.py` 164 | `architecture-verdict.md:49`: +0.0030 — tie; w=0.1 нестабилен | **0/123** |
| prelude/coda | 97 (`model/prelude.py`) + ~180 в `model.py` | 277 | `ab_prelude.py` 434 | `architecture-verdict.md:175,178`: tie, «prelude is not taken» | **0/123** |
| JEPA + SIGReg | 95 + 90 | 302 + 160 | `ab_jepa.py` 468, `ab_sigreg.py` 350 | `architecture-verdict.md:48` JEPA +0.0017 tie; `2026-10-21-test-ledger.md:42` SIGReg +0.148 (loss) | **0/123** (JEPA weight), SIGReg 0.05 в 3 логах — это и есть отрицательное плечо |
| MoE gather | ~250 в `experts.py`/`moe_sonic.py` | часть `test_moe_sonic.py` | `bench_gather.py` 272 | `2026-10-21-battery.md:84`: backward без adjoint, +21.8% medium / +51.3% b1 — loss | **0/123**, `moe_gather="off"` везде |

**Итого ~4.9k строк** (модули+тесты+скрипты+конфиг/модель-швы). Все флаги — в
дефолте во всех 215 Python-конфигах на записи, поэтому удаление не меняет
поведение ни одного прогона. Гейт: `dead_knob_audit` rc 0 + `tests/test_train.py`
(бит-идентичность двух шагов) + полный CPU-suite.

**Спорные, но с записанной причиной оставить** — не трогать без нового решения:
`MTP` (`architecture-verdict.md:47`: tie, «code + gates stay»), `adaptive topk`
(`:46`: tie, «kept»), `MSA @2k` (`:51`: tie, «kept for long context»).

---

## 3. Speculative abstractions: флаги, которые не включались ни разу

Все значения проверены по 215 Python-конфигам. `0/N` = сколько раз флаг был
включён / в скольких конфигах поле вообще присутствует.

| flag | default | on record | модуль/ветка | lines | verdict |
|---|---|---|---:|---:|---|
| `moe_topk_mode` | `"off"` | 0/128 | `model/adaptive_topk.py` 191 + `experts.py:277` | ~1,005 с тестами/бенчами | **не удалять**: в `architecture-verdict.md:46` записано «kept» (tie, экономит compute) |
| `moe_route_fused` | `false` | 0/5 | `model/route_fused.py` 352 + `experts.py:1049` | ~1,031 | **DELETE** (нет вердикта, нет прогона; `tsct-ceiling.md:235` — только предложение на будущее) |
| `fused_kda_tail` | `false` | 0/5 | `model/fused_kda.py` 685 + `kda.py:262`, `loopblock.py:528` | ~1,437 | **DELETE** (bench написан, **не запускался**; в `m400b` OFF) |
| `moe_shared_compute` | `false` | 0/5 | `experts.py:1039,1179` | ~550 | **DELETE** (паритет измерен, quality-плеча нет; 2026-11-05 memory) |
| `moe_offload` | `false` | 0/153 | `experts.py:1048` | ~200 | **DELETE** (нет ни одного прогона) |
| `tsct_packed` | `false` | 0 Python | `experts.py:1082,1222` | ~541 | **оставить как ветку**: 521→5 запусков измерены (`tsct-ceiling.md:63`), но банк не тренируется; вернуть вместе с банком |
| `mod_capacity`/`mod_top_k`/`mod_aux_weight` | 1.0 / 0 / 0 | поле не встречается | `transformer.py:84-133,204` | ~1,226 с `test_mod*`, `ab_mod`, `mod_ledger` | **DELETE** (нет ни одного замера в репо; `cheap-architecture.md` — только оценка) |
| `nope_layers` | 0 | поле не встречается | `transformer.py:158,255` | ~20 | **DELETE** |
| `attn_gate` | false | поле не встречается | `transformer.py:180` | ~15 | **DELETE** |
| `ffn_clip` | 0.0 | поле не встречается | `transformer.py:162,271` | ~15 | **DELETE** |
| `loom` | false | 1 чекпоинт true; 0/153 логов | `loopblock.py:482` | ~100 | **OWNER**: один прогон, вердикта нет |
| `full_graph` | false | 0/108 логов | `train/graph.py:560-1028` (`full_graph_refusal` + `FullGraph`) | ~2,100 с тестами | **KEEP**: 2.3% измерены на `final_byte`; на `m400b` — «NOT MEASURED», но это не мёртвый код |
| `ce_fixed` | false | 0/48 логов | `losses.py` 276 + тест 601 | ~1,100 | **KEEP**: страховка на cliff, prevalence 0 — это вывод, а не смерть механизма |
| `activation_recompute` / `activation_offload` / `iter_checkpoint` / `segment_len` | none / 0 / false / 0 | 0 / 0 / 0 / 0 | `train/loop.py`, `model/recompute.py` | — | **KEEP**: memory-рычаги, измерены в `actmem`/`smoke_long`; `iter_checkpoint` — дубль `activation_recompute="full"` (см. §4) |

**Итого Tier 3 без спорных: ~4.4k строк.** Это самый большой блок, который
требует решения владельца: удаление не меняет ни одного прогона, но убирает
механизмы, которые кто-то может захотеть вернуть. Правило «отклонено ≠ удалено»
(`2026-11-05-dense-beats-bank.md:39`) прямо противоречит ADR-0002; выбрать
должен владелец.

---

## 4. Duplicated logic

| # | pair | path:line | what differs | survivor / action |
|---|---|---|---|---|
| D1 | `--init-from` реализован трижды | `train/loop.py:1370-1409`, `train/sft.py:1153-1175`, `post/rl_train.py:445-462` | train предпочитает `model_ema` и вызывает `_with_moe(..., refuse=True)` + `init_controller`; SFT/RL читают только `foreign["model"]` и **не** применяют LFB-секцию. Это не косметика: SFT/RL из EMA-чекпоинта стартуют с сырых весов, а не с сервируемых | оставить версию `train/loop.py`; вынести `init_from_checkpoint()`; сначала починить расхождение (тест: init из файла с `model_ema` даёт разные веса в SFT/RL) |
| D2 | `LN2` / `bpb` | `train/loop.py:64`; `cli.py:36`; `scripts/eval_window.py:32`, `noise_floor.py:30`, `surprise_map.py:29`, `prune_by_perplexity.py:47`, `ce_cliff_report.py:112` (литерал `0.693147...`), `ab_depth8.py:197`, `sft.py:1416` | одна и та же константа в 8 местах; `cli.py` считает её через `torch.log(torch.tensor(2.0)).item()` | импортировать `bpb` из `train.loop`; −~15 строк, один источник |
| D3 | `iter_checkpoint` vs `activation_recompute` | `config.py:2424` vs `:2435`; `model/model.py:575-607` | два имени одного recompute-плеча; `model.py:659-666` уже отказывает при обоих | удалить `iter_checkpoint` (0/28 логов), оставить enum |
| D4 | `jepa` vs `jepa_weight` | `config.py:1463` vs `:1459`; проверка `:1637` | булев дубль условия `jepa_weight > 0`; поведение читает только weight | удалить `jepa` (dead_knob DEAD) |
| D5 | checkpoint loader | `train/loop.py:504` `load_model_half`; `serve.py:218` `load_checkpoint` | **не дубль**: `serve` — тонкая обёртка над тем же loader'ом; `load_ckpt` и CLI используют `load_model_state` | не трогать; это уже один seam |
| D6 | eval loops | `train/loop.py:981` `evaluate`; `post/rl_train.py:270` `evaluate_holdout`; `bench/runner.py:84` `run_task`; `train/sft.py:1391-1425` masked-CE + first-byte | четыре протокола: BPB, RL holdout accuracy, MC tasks, SFT masked CE; общий только `bpb`-перевод | не дубль; не объединять, кроме D2 |
| D7 | LR schedule | `train/lr.py:30` `wsd_lr` | один WSD, используется `loop.py:1776` и `sft.py:1227`; RL идёт своим lr | не дубль |
| D8 | Rust trainer vs Python trainer | `crates/dormouse-train` 10,415 + `dormouse-cli` 879 + `dormouse-data` 4,177 | тот же train loop, те же ckpt, тот же CLI; **ни один Rust-бинарник не собран**, `run_maxopt2.sh:8` зовёт отсутствующий `target/release/train` | см. §5; Python — живой, Rust — архив |
| D9 | A/B harness | 37 × `port/scripts/ab_*.py` (15,256 строк) | у каждого свой argparse/model/trainer/JSONL boilerplate; `_lane.py` — только guard, не общий драйвер | не удалять wholesale (это evidence path); при желании — один data-driven driver, но это refactor, не deletion |
| D10 | FFN / block / attention arms | `model/ffn.py` vs `experts.py`; `transformer.py` vs `loopblock.py`; `fullattn.py` vs `kda.py` | это **не** дубли: выбор по `ffn_form`/`block_form`/`attn_kind` | не трогать |
| D11 | CE paths | `losses.py:fixed_cross_entropy` + `fixed_soft_ce`; `F.cross_entropy` в `model.py`/`sft.py`/`rl.py` | это один выбор `ce_fixed` (OFF во всех 48 логах, где поле есть), а не две реализации одного правила: fixed-CE — autograd-функция с другим градиентом, не копия `F.cross_entropy` | не дубль; `ce_fixed` оставить как страховку |

---

## 5. Rust lane: статус, а не мнение

| вопрос | ответ | evidence |
|---|---|---|
| собирается? | **да**, `cargo check --workspace` exit 0; 6 packages | запущено 2026-11-08, warnings только `msa_stage.rs` missing docs |
| reachable из продукта? | **нет**. Нет PyO3, нет `cdylib`, нет `ffi/` | `grep -rn pyo3 crates vendor` → 0; `port/README.md:74` обещает `ffi/`, которого нет |
| есть ли живой бинарник? | **нет**. `target/release` содержит `cublas-poc` (2026-09-27), но `train`/`export`/`serve`/`generate` не собраны ни разу | `find target -name train -o -name export ...` → 0; `run_maxopt2.sh:8` указывает на отсутствующий `target/release/train` |
| дублирует Python? | **да**: train 10,415 + cli 879 + data 4,177 = **15,471** строк | `cargo metadata`: `dormouse-train` зависит от core/data; `dormouse-cli` — от train/core; Python `dormouse.cli` — живой |
| CI его держит? | **да**: `cargo test -p dormouse-core -p dormouse-data -p dormouse-train --lib`, `model_seam`, `cargo check -p dormouse-cli`; отдельный `fused-library.yml` гоняет 20 vendor-крейтов | `.github/workflows/ci.yml`, `fused-library.yml` |
| vendor: что не в сборке? | `burn-eggroll` 327, `burn-parcae` 317, `burn-ptrn` 399 — нет path-edge в `cargo tree`; но `fused-library.yml` собирает все 20 | `cargo tree --workspace`; `library-crate-fate.md:73-76` |
| ADR-0014 исполнен? | **нет**: `burn-msa` (1,144) существует и **входит** в `dormouse-core`; `use_msa`/`msa_topk`/`msa_block` живы в `config.py:1585-1587` и в 9 пресетах | `docs/adr/0014-msa-cut.md:13-15` vs `crates/dormouse-core/Cargo.toml:45` |
| `burn-byteflow`? | док говорит «DELETE 2026-09-28», код — **живая зависимость** core и train | `library-crate-fate.md:80` vs `crates/dormouse-core/Cargo.toml:50`, `crates/dormouse-train/Cargo.toml:33`; crate 2,189 строк |

**Вердикт.** Rust-лейн — это **не второй продукт, а второй тренер**. `dormouse-core`
(16,811) остаётся формульным источником; `dormouse-train`+`cli`+`data` (15,471) —
кандидат на архив/удаление, если владелец не планирует возвращаться к Rust-обучению.
`burn-msa` (1,144) — отдельный кандидат: ADR-0014 уже решил его судьбу, но решение
не исполнено; удаление требует снятия Rust MSA-плеча (~300 строк `msa_stage.rs` +
ветки в `loop_block.rs`). `burn-eggroll`+`burn-ptrn` (726) — удалить, `burn-parcae`
(317) — оставить до закрытия записанной причины WIRE.

---

## 6. Script sprawl: 125 файлов, 43,125 строк

| группа | файлов | строк | что это | судьба |
|---|---:|---:|---|---|
| (a) живой контур/гейты | 32 | 16,625 | `run_b1.sh`, `build_mix4.py`, `build_doc_sidecar.py`, `docmask_audit.py`, `eval_window.py`, `plan_restart.py`, `dashboard.py`, `route_hist.py`, `collect_ab.py`, `noise_floor.py`, `compare_stairs.py`, `dead_knob_audit.py`, `claims_ledger.py`, `step_audit.py`, `sync_audit.py`, `kernel_profile.py`, `dispatch_census.py`, `pass_census.py`, `precision_census.py`, `capacity_math.py`, `mod_ledger.py`, `bench_compile.py`, `bench_graph.py`, `probe_capture.py`, `probe_fp8.py`, `probe_1m.py`, `ab_longctx.py`, `eval_agent.py`, `build_sft_v2.py`, `build_rl_tasks.py`, `chat.py`, `_lane.py` | keep |
| (b) one-off research A/B | 37 | 15,256 | все `ab_*.py` | keep как evidence path; удалять только те, чей механизм удалён в §2/§3 |
| (b) benches/probes | 42 | ~9,066 | `bench_*.py` 35, `probe_*.py` 7 | большинство keep; 14 superseded — см. ниже |
| (b) data builders | 4 | 3,684 | `build_mix4.py` 2,028, `build_sft_v2.py` 575, `build_rl_tasks.py` 555, `build_doc_sidecar.py` 526 | keep (данные воспроизводимы) |
| (c) superseded | **14** | **2,726** | см. таблицу ниже | **DELETE** после проверки, что новый драйвер воспроизводит таблицу |

**14 superseded, с новым драйвером:**

| old | lines | successor | evidence |
|---|---:|---|---|
| `bench_moe.py` | 125 | `bench_moe_paths.py` (290, 2026-11-04: naive/triton/batched/bucketed, one shape) | docstrings |
| `bench_moe_dropless.py` | 224 | `bench_moe_paths.py` | " |
| `bench_moe_fp32.py` | 168 | `bench_moe_paths.py` | " |
| `bench_moe_kernel.py` | 29 | `bench_moe_paths.py` | " |
| `bench_moe_seam.py` | 193 | `bench_moe_paths.py` | " |
| `bench_decode.py` | 198 | `bench_decode_fast.py` (measured table on a real checkpoint) | docstrings |
| `bench_shape.py` | 208 | `bench_step.py` (generic step timing) | `bench_step.py` importers |
| `bench_full_graph.py` | 217 | `bench_graph.py` (2026-11-05, live m400b shape) | docstrings |
| `probe_ops.py` | 88 | `probe_capture.py` (the tool, `probe_ops` is the old op list) | `probe_capture.py` |
| `probe_vram.py` | 142 | `scripts/vram_ledger.py` (root) | docstrings |
| `profile_phase.py` | 175 | `kernel_profile.py` (per-kernel attribution) | docstrings |
| `smoke_gpu.py` | 180 | `smoke_byteflow.py` (the WIRING gate) | `smoke_byteflow.py` |
| `bench_opt.py` | 418 | `bench_opt_graph.py` + `bench_opt_launches.py` (two axes, not one) | docstrings |
| `bench_epilogue.py` | 361 | `bench_fused_residual.py` (parity/launch/ms on the same seam) | docstrings |

**Что не удалять wholesale.** `ab_*.py` — это не «мусор», это единственный
воспроизводимый путь к числам в `docs/records/reviews`. Их удаление сэкономит 15k строк,
но нарушит правило «a claim names its evidence». Удалять только те, чей механизм
уходит в §2/§3: `ab_depth8`, `ab_energy`, `rerank`, `ab_prelude`, `ab_jepa`,
`ab_sigreg`, `bench_gather`, `ab_mod`, `mod_ledger`, `bench_fused_kda`,
`bench_share_u_compute`, `ab_tsct2`, `ab_tsct_starvation`. Это ~4.7k строк,
уже включённых в оценки §2/§3.

---

## 7. Docs that contradict code (10 checked, 10 drifted)

Все выбранные доки — до 2026-11-01, кроме `port/README.md` (без даты, но содержимое
старше). Проверялся **один конкретный** claim на документ.

| # | doc:line | claim | code now | verdict |
|---|---|---|---|---|
| 1 | `docs/adr/0003-fused-measure-then-decide.md:5,7` | `crates/dormouse-core/src/fused/` = 6,568 LOC, «Verdict: kept», `DM_FUSED=1` | каталога нет; `grep DM_FUSED crates` → 0; `git log --diff-filter=D` показывает удаление в `5abdf3c` | **ЛОЖЬ**: ADR описывает несуществующий каталог |
| 2 | `docs/adr/0014-msa-cut.md:13-15` | `use_msa`/`msa_topk`/`msa_block` и `vendor/.../burn-msa` удалены | `config.py:1585-1587` живы; 9 пресетов их ставят; `vendor/.../burn-msa` = 1,144 строки; `crates/dormouse-core/Cargo.toml:45` | **НЕДОПИСАНО**: MSA вернули, ADR не обновлён |
| 3 | `port/README.md:40` | b1 = **1,066,354,113** params | `tests/test_param_count.py:26` = **1,066,358,209**; `dormouse info b1` печатает 1,066,358,209; `configs/b1.toml:110,127` = 1,066,358,209 | **ЛОЖЬ** на 4,096 params |
| 4 | `port/README.md:68` | «tests/ 117 tests» | 166 test-файлов, 1,865 `test_*` функций (`pytest --collect-only`) | **УСТАРЕЛО** (117 — это ни файлы, ни кейсы) |
| 5 | `port/README.md:74` | Rust — «as a PyO3 extension (`ffi/`)» | `ffi/` нет; `grep pyo3` → 0; Rust-бинарники не собраны | **ПЛАН ≠ КОД** |
| 6 | `docs/records/architecture/post-training.md:16` | «RLVR — PPO + value head (not GRPO)» | `post/rl.py:46`: «no value head»; `grep -rn '\bPPO\b' port/src` → 0; реализованы CRR/group advantages | **ЛОЖЬ**: план выдан за реализацию |
| 7 | `docs/records/decisions/2026-10-15-1b-readiness-plan.md:67` | b1 `max_iter 2` | `port/configs/b1.toml:14` `max_iter = 4` | **УСТАРЕЛО** |
| 8 | `docs/records/architecture/2026-10-23-final-architecture.md:3,46-48` | «design frozen»; E=256, top-k 2, dropless bank — финальный рецепт | `2026-11-05-dense-beats-bank.md:1-10` — dense бьёт банк; живой прогон `port/configs/arch_dense_byteflow8.toml:34,42,44` — `block_form="transformer"`, `n_experts=1`, `use_tsct=false` | **УСТАРЕЛО**, статус «frozen» не снят |
| 9 | `docs/records/architecture/2026-10-23-scaling-recipe.md:45` | «large model should be wide-ish and **expert-rich**» | тот же dense-вердикт 2026-11-05; банк не в рецепте | **ПРОТИВОРЕЧИТ** более позднему замеру |
| 10 | `docs/records/architecture/library-crate-fate.md:80` | `burn-byteflow` — «**DELETE** … 1,121 LOC» | crate = 2,189 строк; **является зависимостью** `dormouse-core/Cargo.toml:50` и `dormouse-train/Cargo.toml:33` | **ЛОЖЬ**: док говорит «удалён», код зависит |

Дополнительно, не вошло в десятку: `docs/records/reviews/2026-10-21-battery.md:87` говорит
halt «not run», а `2026-10-23-architecture-verdict.md:50` приводит замер halt;
`docs/records/reviews/2026-10-17-port-audit.md:22` говорит «12.8k lines of source» при
48,179 сейчас; `docs/adr/0023-inference-export.md:3` объявляет `export` bin
«landed», но в живом Python CLI команды `export` нет, а Rust `export` не собран.

---

## 8. Commit sequence (lowest risk first)

| # | commit | exact files | removed | risk | gate |
|---|---|---|---:|---|---|
| C1 | `docs: correct ten stale claims` | 10 docs above + `README.md:40,67,68,74`; не код | ~30 changed | none | `python3 tools/check_doc_refs.py`; claims_ledger |
| C2 | `chore(config): delete the two DEAD knobs` | `port/src/dormouse/config.py` (`jepa` 1463 + check 1637; `muon_sf_retract` 2150 + refusal 5232), `port/configs/flagship_base.toml`, `m400b.toml`, `tests/test_jepa.py`, `tests/test_optimizer_suite.py`, `tests/test_sf_muon_legal.py` | **45** [C] | lowest | `dead_knob_audit` rc 0; `pytest test_config.py test_optimizer_suite.py test_sf_muon_legal.py` |
| C3 | `chore(act_ckpt): delete the unwired saved-tensor experiment` | `model/act_ckpt.py`, `tests/test_act_ckpt.py`, `scripts/bench_act_ckpt.py`; docstrings `model/act_mem.py:39`, `model/recompute.py:21` | **1,079** [C] | low | `grep -rn act_ckpt port/src` empty; full CPU suite |
| C4 | `chore(agent_rl): delete the unwired trajectory-advantage` | `post/agent_rl.py`, `tests/test_agent.py:40,367-426`, docstrings `post/__init__.py:19`, `post/agent.py:27` | **160** [C] | low | `pytest tests/test_agent.py`; grep |
| C5 | `chore(selfplay): delete the package with no entry point` | `src/dormouse/selfplay/*` (5 files), `tests/test_selfplay.py` | **2,126** [C] | medium (plan mentions self-play) | `grep -rn selfplay port/src` only package; full suite; if lane is funded, add driver instead |
| C6 | `chore(export): delete the unwired QAT export` | `model/export.py`, `tests/test_export.py`, `scripts/qat_export_demo.py`, comment `configs/tiny.toml:3` | **970** [C] | medium (QAT relation) | grep; `dormouse info tiny`; full suite |
| C7 | `chore(vendor): delete eggroll + ptrn, keep parcae` | `vendor/dormouse-fused/Cargo.toml` members, `crates/burn-eggroll`, `crates/burn-ptrn` | **726** [C] | low | `cargo check --workspace`; `fused-library.yml`; `cargo metadata` no path edge |
| C8 | `chore(model): execute ADR-0002 on halt/energy/prelude/JEPA/SIGReg/gather` | `model/halt.py`+`test_halt.py`+`ab_depth8.py`; `model/energy.py`+`test_energy.py`+`ab_energy.py`+`rerank.py`; `model/prelude.py`+`test_prelude.py`+`ab_prelude.py`; `model/jepa.py`+`model/sigreg.py`+`test_jepa.py`+`test_sigreg.py`+`ab_jepa.py`+`ab_sigreg.py`; gather branch + `bench_gather.py`; config/model seams | **~4,900** [C]+[S] | medium | 215-config census: halt/energy/prelude/JEPA/gather all 0; SIGReg only the 3 negative-arm logs; `dead_knob_audit`; `tests/test_train.py` bit-identity; full suite |
| C9 | `chore(model): delete never-enabled speculative arms` | `model/route_fused.py`+`test_moe_route_fused.py`; `model/fused_kda.py`+`test_fused_kda.py`+`bench_fused_kda.py`; `transformer.py` mod/nope/attn_gate/ffn_clip + `test_mod*`+`ab_mod.py`+`mod_ledger.py`; `moe_shared_compute`+`bench_share_u_compute.py`; `moe_offload` | **~4,400** [C]+[S] | medium-high | same census; `dead_knob_audit`; full suite |
| C10 | `chore(scripts): delete 14 superseded drivers` | table in §6 | **2,726** [C] | low | `ruff check`; `python -m compileall`; grep no refs; docs still name logs |
| C11 | `chore(rust): archive the duplicate trainer` (owner decision) | `crates/dormouse-train`, `crates/dormouse-cli`, `crates/dormouse-data`, root `Cargo.toml`, `.github/workflows/ci.yml`; `run_maxopt2.sh`; docs refs | **15,471** [C] | high | `cargo check -p dormouse-core -p backend-parity`; `cargo test -p dormouse-core --lib`; Python untouched |
| C12 | `chore(rust): delete burn-msa per ADR-0014` (owner decision) | `vendor/.../burn-msa`, `crates/dormouse-core/src/msa_stage.rs`, `loop_block.rs` MSA branches, `Cargo.toml` edge | **~1,500** [C]+[S] | high | ADR-0014 gate; `cargo check`; no Python change |
| C13 | `chore(worktrees): prune stale checkouts` (not a code commit) | `git worktree list`: ~80 worktrees, ~4–5 GB (incl. `wt/publish` 665M) | 0 code | medium (some may hold unmerged work) | `git status` in each; `git worktree remove` only clean ones |

**Smallest honest diff:** C1+C2+C3+C4+C7 = **2,010 removed**, no behavior change,
no policy decision. Do this first.

**If the owner accepts ADR-0002 enforcement:** add C5+C6+C8+C10 = **~10,800**.
**If the owner also kills speculative arms:** add C9 = **~15,200**.
**Rust archive is a separate product decision**, not a cleanup: C11 alone is
15,471 lines and removes the Rust training path; C12 is another 1,500.

---

## 9. What not to delete, and why

| item | lines | why keep |
|---|---:|---|
| `crates/dormouse-core` | 16,811 | formula source for `model/*.py`; many docstrings cite `crates/dormouse-core/src/*.rs`; deleting it breaks the provenance claim |
| `vendor` wired crates (16) | ~44k | every `dormouse-core`/`train` path dependency; `burn-msa` is the only contradictory one |
| `port/tests` (except tests of deleted modules) | 58,167 | each is a gate; deleting a test without deleting its code is forbidden by `CONTRIBUTING.md` |
| `ab_*.py` with doc refs | 15,256 | the reproducible evidence path for `docs/records/reviews`; deleting breaks "a claim names its evidence" |
| `full_graph` | ~2,100 | 2.3% measured on `final_byte` (`2026-10-24` memory); not a tie |
| `ce_fixed` | ~1,100 | insurance for a cliff that has not fired; the absence of a firing is not a deletion verdict |
| `segment_len` / `activation_*` / `iter_checkpoint` | ~1,000 | memory levers measured in `smoke_long`/`actmem`; `iter_checkpoint` only the duplicate name goes |
| `runlog.py` | 337 | test helper for HI-1; move to `tests/`, don't delete |
| `train/__init__.py` | 48 | imported by 6 scripts; not a dead facade |
| `cublas-poc` | 828 | one-off spike, but it is a recorded measurement; archive, don't delete |

---

## 10. Open questions the owner must answer

1. **«Отклонено ≠ удалено» или ADR-0002?** If the former, §2/§3 are off the table
   and the honest deletion is 2,010 lines. If the latter, ~15k Python lines go.
2. **Self-play lane:** funded (then add a driver + a gate; 1,420 lines currently
   cannot run) or abandoned (delete 2,126 lines)?
3. **QAT export:** shipping (then add `dormouse export` to the CLI) or not
   (delete 970 lines)?
4. **Rust trainer:** reference (archive `dormouse-train`/`cli`/`data`, keep
   `dormouse-core`) or product (then Python is the duplicate — not proposed)?
5. **`burn-msa`:** ADR-0014 says delete, code says it is a live dependency of
   `dormouse-core`. Which document is authoritative?

---

## Appendix: commands run (read-only)

```sh
cd /home/sehaxe/dormouse/port
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python scripts/dead_knob_audit.py --json /tmp/knobs.json
.venv/bin/python -m pytest --collect-only -q          # 166 files
.venv/bin/python -m pytest tests/test_act_ckpt.py tests/test_export.py tests/test_invariants_history.py
.venv/bin/python -m pytest tests/test_selfplay.py     # 45 passed
.venv/bin/python -m pytest tests/test_config.py tests/test_dead_knob_audit.py  # 35 passed

cd /home/sehaxe/dormouse
cargo metadata --no-deps --format-version 1
cargo check --workspace --quiet                       # exit 0
cargo tree --workspace -e normal --prefix none        # 16 vendor crates reachable
find target -name train -o -name export -o -name serve -o -name generate  # no binaries
python3 tools/check_doc_refs.py
```

No GPU command, no training, no checkpoint write, no repository file changed except
this report.
