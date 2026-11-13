# 2026-11-08 — claims repair list

**Read-only audit.** No code, config, checkpoint or corpus was modified. No GPU
training or benchmark was run. The test run recorded below was repeated with
`CUDA_VISIBLE_DEVICES=""`; the one CUDA parity test self-skips when the card is busy.
The only file written is this report. Tags: **[M]** measured here or in the named log,
**[C]** computed from named numbers, **[S]** inference with no direct measurement.
`UNVERIFIABLE-NOW` means the artifact was deleted in the 2026-11-08 merge and the
named replacement is what would settle it.

**Scope.** `port/scripts/claims_ledger.py` on the working tree at `64bf076`; nine
derived mismatches; the ten claims listed in `docs/records/reviews/2026-11-08-deletion-audit.md` §7.
The brief says "ten documents"; it is **ten claims in eight files** — `port/README.md`
supplies three of them.

---

## 0. Verdict (RU)

**Девять «mismatch» — это не девять неверных чисел.** Семь строк — артефакты
парсера ledger'а; два документа действительно устарели, и в них неверны **пять**
значений: `token-skipping` (старый b1: total и `max_iter`) и `muon-variants`
(medium: total, Muon-группа, число форм). Оба числа были верны на дату документа —
это не выдумка, а устаревшее текущее состояние.

**Команда из брифа не воспроизводит девять.** С `--no-params` ledger печатает
`mismatch 1`; девять появляются только когда param-подпроцесс включён. Хуже:
`scripts/gate_fast.sh:40` вызывает ledger именно с `--no-params`, поэтому гейт
видит один **ложный** mismatch и не видит ни одного из двух реальных устаревших
чисел. Это дефект гейта, а не косметика.

**Десять claim'ов §7 — это 10 claim'ов в 8 файлах.** Восемь чинятся в теле
(README ×3, `post-training.md`, план 1B ×2, `final-architecture.md`,
`scaling-recipe.md`, `library-crate-fate.md`); два ADR — это датированные записи
решений, их тело не переписывается, ставится баннер `superseded` / `not executed`.
Ни один из десяти не предлагается переносить в `docs/records/archive/`: все восемь файлов
всё ещё цитируются как текущие, а правило проекта — «retracted claim stays published».

**Самое дорогое по последствиям.** ADR-0014 говорит, что MSA удалён, а `burn-msa`
— живая зависимость `dormouse-core`, и `~/logs/medium_long.log:11` печатает
`msa_blocks=588,816,384`. README называет тест, который «не даёт числу сгнить»,
и этот же тест показывает, что README врёт на 4,096. `post-training.md` выдаёт
PPO + value head за реализацию, тогда как в коде явно «no value head».

**Проверка UNVERIFIABLE-NOW.** Мерж 2026-11-08 удалил `data_local/` и 14 ГБ
чекпоинтов (`docs/records/reviews/2026-11-08-merge-incident.md`). Ни один из 19 claim'ов
ниже не зависит от удалённого: решающие артефакты — git-конфиги,
`port/tests/test_param_count.py`, `~/logs/flagship3.log`, `~/logs/medium_long.log`
и история коммитов. Единственное число, которое не воспроизводится текущим
деревом, — 6,568 LOC в ADR-0003; дерево `5abdf3c^` есть в git, но точный
code-line counter 2026-09-21 не сохранён. Это помечено `UNVERIFIABLE-NOW` в D5.

---

## 1. Reproduction: why the brief's command says one, not nine

| state | command | claims | derived | mismatches | risk list |
|---|---|---:|---:|---:|---|
| before this report (audited tree) | `claims_ledger.py --docs ../docs --repo .. --no-params` | 1,944 | 18 | **1** | `efficiency-frontier.md:275` only |
| before this report (audited tree) | same, **without** `--no-params` | 1,944 | 34 | **9** | six documents, nine rows |
| after this report | same, **without** `--no-params` | 1,964 | 36 | **10** | + this file's `−0.107` slope |
| after this report | `--no-params` | 1,964 | 18 | **1** | same false positive |

The nine claims analysed below are the **pre-report** nine; adding this report does not
change their status, it only adds a 10th row from this file's own quotation of the
slope (§4.4).

`--no-params` skips the one subprocess that rebuilds parameter counts
(`claims_ledger.py:658-660`, `:1479`), so every `params` claim becomes
`UNVERIFIABLE` and cannot mismatch. The nine are exactly the param claims.

The committed ledger is also stale: `docs/records/reviews/2026-11-07-claims-ledger.md:5`
says **1,627 claims / 0 mismatches** at commit `8229564`; the working tree has
1,944 claims in 123 documents. Do not cite the committed summary as current.

### 1.1 The parser defects behind seven of the nine

| # | parser behavior | code | what it false-positives |
|---|---|---|---|
| P1 | `config_for()` returns the **first** config named on the line | `claims_ledger.py:645-648` | `token-skipping:19` names b1 first, so the *small* claim is derived against b1 |
| P2 | a table column headed `params` is read as **total model params** | `claims_ledger.py:409-410` | `optimizer-audit:38-41` — the column is a count of named tensors |
| P3 | `INLINE_BPB` matches a signed number before `BPB` | `claims_ledger.py:319-340` | `efficiency-frontier:275` — `−0.107 BPB/doubling` is a slope, derived as an absolute |
| P4 | a params cell is matched against the first number even when the cell carries two | `claims_ledger.py:475-497` | `data-capacity:39` — port 9,476,993 and burn 9,197,390 in one cell |
| P5 | a scale range is treated as a point claim | `claims_ledger.py:319-340` | `depth-recurrence:4` — `20–200M params` is a range, not "flagship is 200M" |

This report will itself be scanned by the next ledger run: self-exclusion is
`if "claims-ledger" in path.name` (`claims_ledger.py:1468`), which does not match
`claims-repair`. That is a sixth parser/scope defect; see §5.

---

## 2. The nine MISMATCH claims (ranked)

Ranked by whether the document or the derived number is wrong, and by blast radius.
`Stated` is the document's value; `derived` is the ledger's recomputation.

| # | claim (doc:line) | stated | derived | artifact it names | wrong side | required action |
|---:|---|---:|---:|---|---|---|
| M1 | `muon-variants.md:16` "medium = 141 910 405 params / 90 925 952 Muon / 12 shapes" | 141,910,405 | 138,436,993 | `port/configs/medium.toml` | **document** (stale since `907b3b5`, 2026-11-05) | annotate old as the `use_msa=true` config; add current 138,436,993 |
| M2 | `token-skipping.md:19` "small = 9,476,993" | 9,476,993 | 1,066,358,209 | `port/configs/b1.toml`, `port/configs/small.toml` | **ledger** for small; **document** for b1 on the same line | keep small; fix b1 1,066,354,113 → 1,066,358,209 and `max_iter 2` → 4 |
| M3 | `optimizer-audit.md:38` "Muon: 286" | 286 | 102,678,943 | `port/configs/flagship_base.toml` | **ledger** | no number change; optional header "params (tensors)" |
| M4 | `optimizer-audit.md:39` "AdamW: 558" | 558 | 102,678,943 | same | **ledger** | no number change |
| M5 | `optimizer-audit.md:40` "Engram table: 1" | 1 | 102,678,943 | same | **ledger** | no number change |
| M6 | `optimizer-audit.md:41` "non-table AdamW: 557" | 557 | 102,678,943 | same | **ledger** | no number change |
| M7 | `data-capacity.md:39` "burn 9 197 390" | 9,197,390 | 9,476,993 | `port/configs/small.toml` | **ledger** (port vs burn are different counters) | no number change; optionally name `crates/dormouse-core/src/param.rs:139` |
| M8 | `depth-recurrence.md:4` "20–200M params" | 200M | 102,678,943 | `port/configs/flagship_base.toml` | **ledger** (range endpoint, not flagship's count) | no number change; optionally cite `m400b` 199,275,551 |
| M9 | `efficiency-frontier.md:275` "−0.107 BPB/doubling" | −0.107 | 1.51858 | `logs/flagship3.log` | **ledger** (slope parsed as absolute) | no number change; add the division that proves the slope |

### M1 — `docs/records/research/2026-10-17-muon-variants.md:16-17` — the one real stale document

> `configs/medium.toml` = **141 910 405** params, **90 925 952** in the Muon group,
> **12** distinct shapes [ours, 2026-10-17].

**What the code says now** (`port/configs/medium.toml:48`, `use_msa = false`):
138,436,993 total; 87,452,544 in the Muon group; 11 distinct shapes [M/C].
The old config at `907b3b5^` (`use_msa = true`) is exactly 141,910,405 / 90,925,952 / 12 [M/C].
`907b3b5` (2026-11-05) set `use_msa = false`; the MSA block accounted for the delta.

**Wrong side:** the document's present-tense claim, not the historical number. It was
true on 2026-10-17 (`port/checkpoints/medium/med1.config.toml:57` has `use_msa = true`;
`~/logs/medium_run.log` is that 141.9M run). It is false for today's `medium.toml`.

**Replacement (keep the old numbers named as superseded):**

```text
`configs/medium.toml` = **138,436,993** params, **87,452,544** in the Muon group,
**11** distinct shapes [ours, 2026-11-05 re-measure: `907b3b5` set `use_msa=false`.
The 2026-10-17 values **141,910,405 / 90,925,952 / 12** were the `use_msa=true`
config at `907b3b5^`; they are the historical MSA-on measurement, not today's file].
```

### M2 — `docs/records/research/2026-10-16-token-skipping.md:19` — ledger artifact plus a real b1 drift

> b1 = 1,066,354,113 total / **~57M active**, `max_iter 2` …
> Port `small` = **9,476,993** params, `max_iter 4` in the file …

**What the code says:** `small` = 9,476,993 [M] (`port/configs/small.toml`);
`b1` = 1,066,358,209 [M] (`port/configs/b1.toml:110,127`,
`port/tests/test_param_count.py:26`); `b1` `max_iter = 4` [M] (`port/configs/b1.toml:14`).
The 4,096 delta is exactly two iteration-embedding rows of 2,048 (`af8b0c9`, 2026-10-16).

**Wrong side:** the ledger is wrong about `small` (it picked the first config on the
line, b1, via `claims_ledger.py:645-648`); the document is wrong about b1's total and
depth. The doc was true before `af8b0c9`; it is stale now.

**Replacement:**

```text
* **Shape.** b1 = **1,066,358,209** total / **~57M active**, **`max_iter 4`**
  (train deeper than infer; `port/configs/b1.toml:14`), topk-2 over 1024 TSCT
  experts + Engram (`port/configs/b1.toml:110,127`;
  `port/tests/test_param_count.py:26`). The old **1,066,354,113 / `max_iter 2`**
  is the pre-`af8b0c9` (2026-10-16) count and is superseded. Port `small` =
  **9,476,993** params, `max_iter 4` in the file, **operating depth 2** on the
  burn side (`port/configs/small.toml`; `docs/protocols/AB-PROTOCOL.md:179`).
```

### M3–M6 — `docs/records/reviews/2026-11-06-optimizer-audit.md:38-41` — four ledger artifacts

The table header is `| group | params | elements | share | LR |`. In this table
**`params` means the count of named tensors**, not the model's parameter total.
The ledger reads the header via `_column_unit` (`claims_ledger.py:409-410`) and
rebuilds 102,678,943 from `flagship_base.toml`, then calls 286 / 558 / 1 / 557
mismatches. Measured on `port/configs/flagship_base.toml` with
`param_count` (`port/src/dormouse/model/model.py:1523`) and `is_muon_param`
(`port/src/dormouse/train/optim.py:1667`), meta device [M]:

| group | doc count | measured count | doc elements | measured elements |
|---|---:|---:|---:|---:|
| Muon | 286 | **286** | 51,686,276 | **51,686,276** |
| AdamW | 558 | **558** | 50,992,667 | **50,992,667** |
| Engram table (`loop.engram.tables`) | 1 | **1** | 50,331,648 | **50,331,648** |
| non-table AdamW | 557 | **557** | 661,019 | **661,019** |

**Wrong side:** derived. No number change is required. A header rename
`params` → `params (tensors)` would stop a reader (and the parser) from making the
same mistake; that is the only recommended edit.

### M7 — `docs/records/research/2026-10-18-data-capacity.md:39` — port and burn are different counters

> | `small` | 9 476 993 (port; burn 9 197 390) | …

The ledger takes `9 197 390` as a claim and derives the port's 9,476,993, then calls
it a mismatch. Both numbers are correct for their own stack: port `small` = 9,476,993
[M]; burn `small` = 9,197,390, corroborated by `crates/dormouse-core/src/param.rs:139`
and `docs/protocols/determinism.md:23,31` [C]. **Wrong side:** derived. No number
change. Optionally append the burn artifact to the cell, e.g.
`burn 9 197 390 (`crates/dormouse-core/src/param.rs:139`)`.

### M8 — `docs/records/research/2026-11-06-depth-recurrence.md:4` — a scale range, not a model count

> we train byte-level LMs at 20–200M params on one 16 GB card.

The line names `port/configs/flagship_base.toml` for the 3-loop shape; that config is
102,678,943 params [M]. The 200M is the top of the research envelope, and the repo has
`port/configs/m400b.toml` = 199,275,551 [M], so the top is real. **Wrong side:** derived.
No number change. If the author wants the envelope evidenced, cite
`port/configs/m400b.toml` (199,275,551) and note that the 20M lower bound is a rounded
envelope — the nearest in-tree points are `small` 9,476,993 and `mirror_control`
13,478,448 [C], not 20M.

### M9 — `docs/records/research/2026-11-08-efficiency-frontier.md:275` — a slope, not a BPB

> 1.660 at step 4000 (98.3 MB) → 1.519 at step 10000 (245.76 MB);
> slope **−0.107 BPB/doubling**.

`~/logs/flagship3.log:15` has `bpb_fin=1.660`, `:31` has `bpb_fin=1.519`; bytes are
8 × 1,024 × 3 × 4,000 = 98,304,000 and 245,760,000. The division is
(1.6598098 − 1.5185367) / log2(245.76 / 98.304) = **0.10687** ≈ 0.107 [C].
The ledger's `INLINE_BPB` reads `−0.107 BPB` as an absolute and compares it to
1.51858. **Wrong side:** derived. No number change; optionally make the division
explicit on the line so the next reader does not repeat the parser's mistake.

### Mismatch summary

| category | rows | documents | wrong numbers |
|---|---:|---:|---:|
| ledger artifacts (P1–P5) | 7 | 4 | 0 |
| document stale (historical value was correct) | 2 | 2 | 5 |
| **total** | **9** | **6** | **5** |

The five wrong values are: b1 total, b1 `max_iter`, medium total, medium Muon group,
medium shape count. All five are in two dated research notes, not in the code.

**Вердикт по девяти.** Реально устарели два документа (пять значений); семь строк —
ошибка парсера, а не кода. Ни одно число в коде менять не нужно. Оба устаревших
документа датированы, поэтому старые значения остаются опубликованными как
historical, а рядом появляется текущее число.

---

## 3. The ten drifted claims (8 files)

Ranked by blast radius. `Grammar` is exactly the audit's §7 list; the last column is
the proposed repair. Group A rewrites a current instruction/number; group B keeps the
dated record and adds a banner. Nothing is archived.

| # | doc:line | claim (short) | code now | wrong side | group | repair |
|---:|---|---|---|---|---|---|
| D1 | `docs/adr/0014-msa-cut.md:13-15` | MSA crate + `use_msa`/`msa_topk`/`msa_block` **deleted**; no run executed it | live dependency `crates/dormouse-core/Cargo.toml:45`; fields `config.py:1586-1588`; `medium_long.log:11` has `msa_blocks=588,816,384` | **document** (decision not executed) | B | banner `not executed`; keep body |
| D2 | `docs/records/architecture/post-training.md:16,31-34` | "RLVR — PPO + value head"; value head, GAE, PPO clip, token-level advantage | `post/rl.py:43-46` "no value head"; two arms CRR/group `:355-372`; `grep -rn '\bPPO\b' port/src` = 0 | **document** (plan presented as code) | A | rewrite heading + lines 31-34 |
| D3 | `port/README.md:40,70` | b1 = 1,066,354,113 | `test_param_count.py:26` + `b1.toml:110,127` = **1,066,358,209** | **document** (stale) | A | replace with new total; keep old as pre-`af8b0c9` |
| D4 | `docs/records/decisions/2026-10-15-1b-readiness-plan.md:67,81` | b1 `max_iter 2`; total 1,066,354,113 | `b1.toml:14` `max_iter 4`; `test_param_count.py:26` = 1,066,358,209 | **document** (stale) | A | replace both; the same file `:478` already says 4 |
| D5 | `docs/adr/0003-fused-measure-then-decide.md:3,5,7` | `fused/` = 6,568 LOC; `DM_FUSED=1`; "Verdict: kept"; 1.7–2.0× | directory absent; `grep DM_FUSED crates` = 0; deleted `5abdf3c` (7,432 lines); lost 1.25× (`ADR-0009:3`) | **document** (status obsolete) | B | banner `superseded`; keep body |
| D6 | `docs/records/architecture/2026-10-23-final-architecture.md:3,46-48` | "design frozen"; E=256 top-k-2 dropless bank is the final recipe | dense verdict `2026-11-05-dense-beats-bank.md:7-10`; live `arch_dense_byteflow8.toml:34,42-44` | **document** (superseded) | B | status banner + retraction note at 46-48 |
| D7 | `docs/records/architecture/2026-10-23-scaling-recipe.md:45` | "wide-ish and **expert-rich**" | same dense verdict; bank not in recipe | **document** (superseded) | B | keep arithmetic; add measured retraction |
| D8 | `docs/records/architecture/library-crate-fate.md:80,114` | `burn-byteflow` 1,121 lines, **DELETE**, deleted 2026-09-28 | 1,357 src + 832 t/b = 2,189; dependency `dormouse-core:50`, `dormouse-train:33`; re-added `8d61d74`, wired `48ef9ae` | **document** (snapshot stale) | B | add post-snapshot row; keep 1,121 as history |
| D9 | `port/README.md:68` | "tests/ 117 tests" | 167 files, 2,194 collected, 1,870 `def test_` | **document** (stale) | A | replace count |
| D10 | `port/README.md:74` | Rust "as a PyO3 extension (`ffi/`)" | no `port/ffi`, no `pyo3`, no Rust binaries | **document** (plan ≠ code) | A | replace with "not wired" |

### Group A — fix the body (current instructions/facts)

#### D2 — `docs/records/architecture/post-training.md:16,31-34`

**Current text:** `## RLVR — PPO + value head (not GRPO)` and the four bullets
`Value head V(s) bf16, GAE λ0.95, PPO clip 0.2, KL 0.01`,
`reward = verifier_pass − λ_tokens·T − λ_cost·tools`,
`Token-level advantage, not group-relative (GRPO is DPO, encourages verbosity)`,
`EBRM … кандидат после первого PPO-прохода`.

**Code:** `port/src/dormouse/post/rl.py:43-46` — "no importance-sampling clip … no
value head"; `:355-372` — two arms, `group` (GRPO's `(r−mean)/std`) and `crr`
(fork-replay, `post/train_steps.py:1`); `RLConfig.agg = "prompt-mean"` at `:113-115`.
`grep -rn '\bPPO\b' port/src` returns nothing [M]. The statement "GRPO is DPO" is also
false and is contradicted by the shipped `group` arm.

**Replacement:**

```text
## RLVR — CRR / group-relative advantages (no value head; the PPO row is a plan, not code)

- **Shipped:** two advantage arms, `group` (GRPO's `(r − group_mean) / group_std`,
  `post/rl.py:364`) and `crr` (fork-replay, `post/train_steps.py:1`), aggregated
  per prompt (`post/rl.py:374-375`). There is **no value head, no GAE, and no
  importance-sampling clip** (`post/rl.py:43-46`).
- reward = `R_test · S_sol · S_beh` minus the gated group-relative length term
  (`post/rl.py:1-42`); prompt-mean aggregation is the default (`post/rl.py:113-115`).
- **Not shipped:** the PPO value-head recipe (`V(s)`, GAE λ0.95, PPO clip 0.2,
  KL 0.01) and the "token-level advantage" claim; `grep -rn '\bPPO\b' port/src`
  returns nothing. EBRM remains a candidate after the first **RL** pass, not a
  PPO pass.
```

#### D3 — `port/README.md:40,70`

**Current:** "The b1 preset is **1,066,354,113 params (57M active)** — `tests/test_param_count.py`
recomputes it against the module tree, so no preset comment can rot." The same old number
is at line 70. The named test's `B1_PORT_TOTAL` is 1,066,358,209
(`port/tests/test_param_count.py:26`), and the config's own arithmetic ends at
1,066,358,209 (`port/configs/b1.toml:110,127`) [M]. `af8b0c9` (2026-10-16) moved b1
from `max_iter 2` to 4, adding two iteration-embedding rows (+4,096) [C].

**Replacement:**

```text
The b1 preset is **1,066,358,209 params** (the plan's ~57M active; the port's
touched set is ~37.9M/iteration — `port/configs/b1.toml:151-156`) —
`tests/test_param_count.py:26` recomputes it against the module tree, so no preset
comment can rot. The old **1,066,354,113** was the `max_iter 2` count; `af8b0c9`
(2026-10-16) moved b1 to `max_iter 4` (+4,096) and the test now pins the new total.
```

Line 70 replacement:

```text
configs/           small (9,476,993) · b1 (1,066,358,209)
```

#### D4 — `docs/records/decisions/2026-10-15-1b-readiness-plan.md:67,81`

**Current:** `max_iter 2` at line 67; total `1,066,354,113` at line 81. The same file
already records the correction at `:478` ("`configs/b1.toml` = **max_iter 4**"), so the
document contradicts itself. `port/configs/b1.toml:14` is `max_iter = 4`;
`port/tests/test_param_count.py:26` is 1,066,358,209 [M].

**Replacement:**

```text
d 2048 · H 32×64 · d_ffn 5461 · rank 64 · **max_iter 4** · E 1024 experts, topk 2 ·
… (the 2026-10-15 draft said `max_iter 2`; `af8b0c9` changed b1 to 4 on
2026-10-16 — the depth verdict is recorded in this file at `:478`)
```

```text
| **total** | **1,066,358,209 ≈ 1.07B (MEASURED, `tests/test_param_count.py:26`;
  `port/configs/b1.toml:110,127`)** |
```

**Same table, second-order drift (not one of the ten):** line 83 says "both iters" and
line 84 says `~0.40 GFLOP`. With `max_iter 4`, the file's own formula
`6 × active_params_per_pass × max_iter` and the config's touched set
(~37.9M/iteration, `port/configs/b1.toml:151-156`) give `6 × 37.9M × 4 = 0.91 GFLOP`,
not 0.40. The 0.40 does not reconcile even at 2 iters with 57M (0.68) [C]. Re-derive
lines 83-87 before using the wall-clock table; the artifact that settles it is the
model's FLOP counter (not present) plus the config's active-set comment.

#### D9 — `port/README.md:68`

**Current:** `tests/             117 tests, CPU-true by default, CUDA gates opt-in`.
Measured 2026-11-08 [M]: `pytest --collect-only` → **2,194 tests collected in 1.90s**;
167 files matching `port/tests/test_*.py`; 1,870 `def test_` functions. `117` is
neither files nor cases (the deletion audit said 166/1,865; both moved after it).

**Replacement:**

```text
tests/             167 test files · 2,194 collected tests (1,870 `test_*`
                   functions), CPU-true by default, CUDA gates opt-in
```

Artifact: `pytest --collect-only` printed `2194 tests collected in 1.90s`;
`find port/tests -name 'test_*.py' | wc -l` = 167;
`grep -rn 'def test_' port/tests/*.py | wc -l` = 1,870.

#### D10 — `port/README.md:74`

**Current:** "Rust stays where it is already the best code (data pipeline, Engram host
tables, ByteFlow patcher) — as a PyO3 extension (`ffi/`), wired when a profile says a
hot loop is CPU-bound." There is no `port/ffi/`; `grep -rn pyo3` over `Cargo.toml` and
`crates/*/Cargo.toml` returns nothing; `find target -name train -o -name export -o
-name serve -o -name generate` returns 0 [M]. The Rust tree is a parallel workspace,
not a Python extension.

**Replacement:**

```text
**Rust stays where it is already the best code** (data pipeline, Engram host tables,
ByteFlow patcher) — but it is **not wired into Python today**: there is no `ffi/`
directory and no PyO3 dependency (`find port -maxdepth 2 -name ffi` → none;
`grep -rn pyo3 Cargo.toml crates/*/Cargo.toml` → none), and the Rust trainer's
binaries are not built (`find target -name train -o -name export -o -name serve
-o -name generate` → 0). The burn crates are the formula source and a parallel
trainer, not a Python extension. Wire it when a profile says a hot loop is CPU-bound;
GPU-side Rust buys nothing here.
```

Artifact: `ls port/ffi` → absent; `grep -rn pyo3 Cargo.toml crates/*/Cargo.toml` →
empty; `find target -name train -o -name export -o -name serve -o -name generate`
→ 0.

### Group B — history: keep the record, add a dated banner (do not archive)

These five are dated decision/snapshot records. The repo's rule is that a retracted
claim stays published; rewriting the body would erase the record. Insert the banner
and leave the original sentence intact.

#### D1 — `docs/adr/0014-msa-cut.md:13-15` (highest risk: a live dependency called deleted)

**Current:** "Decision: delete it. `vendor/dormouse-fused/crates/burn-msa`, `use_msa` /
`msa_topk` / `msa_block` … go away" and "no run that ever executed it" (`:9-10`).
The arm was re-landed **after** this ADR: `24c3325` (2026-10-03) restored the crate,
`c366b3b` (2026-10-03) wired it into `loop_block.rs` [M]. Today:
`crates/dormouse-core/Cargo.toml:45` has `burn-msa`; `crates/dormouse-core/src/msa_stage.rs`
is 145 lines; `port/src/dormouse/config.py:1586-1588` has `use_msa`/`msa_topk`/`msa_block`;
7 `port/configs/` presets and 2 root `configs/` presets carry the keys; 26 checkpoint
snapshots carry them, 11 with `use_msa = true`; `~/logs/medium_long.log:11` prints
`msa_blocks=588,816,384`, `msa_triton=2000` — the arm executed. The crate is 718
`src` + 426 `tests` = 1,144 code lines (1,201 with `Cargo.toml`/`README`), not deleted
[M/C]. The audit's "9 presets / 1,144 lines / config.py:1585-1587" is close: 1,144 is
`src`+`tests` and 9 is 7 port + 2 root presets; only the line refs are off by one
(`1586-1588`). The counts below are the current ones.

**Banner (insert after line 1):**

```text
**Status: not executed — the arm was re-landed 2026-10-03.** `24c3325` restored the
crate and `c366b3b` wired the QSA arm into `loop_block.rs`. Today `burn-msa` is a
live dependency of `dormouse-core` (`crates/dormouse-core/Cargo.toml:45`);
`msa_stage.rs` is 145 lines; `use_msa`/`msa_topk`/`msa_block` are live
(`port/src/dormouse/config.py:1586-1588`); 7 `port/configs/` presets and 2 root
presets carry the keys; 26 checkpoint snapshots carry them and 11 have
`use_msa = true`; `~/logs/medium_long.log:11` prints `msa_blocks=588,816,384`.
The crate is 718 `src` + 426 `tests` = 1,144 code lines (1,201 with
`Cargo.toml`/`README`), not deleted. The decision below is kept as the record; the
code is authoritative until a new ADR cuts it again.
```

#### D5 — `docs/adr/0003-fused-measure-then-decide.md:3,5,7`

**Current:** `Status: accepted, smoke verdict 2026-09-21`; `crates/dormouse-core/src/fused/`
is 6,568 LOC and off by `DM_FUSED=1`; `Verdict: kept … 1.7-2.0x`. The directory does
not exist; `grep -rn DM_FUSED crates` returns 0; `5abdf3c` (2026-09-27) deleted the
four files (7,432 lines) [M]. ADR-0009 (`docs/adr/0009-fused-rewrite-rungs.md:3`) says
the fusion lost at flagship: 1.25× slower, and the 1.7–2.0× smoke is retracted;
`AGENTS.md:715-716` records 9.21 s/step vs burn 7.05 s/step [M]. The `6,568` line
count is **UNVERIFIABLE-NOW** by `wc -l`: the four files at `5abdf3c^` total 7,432
(4,945 without `tests.rs`); the artifact that would settle it is the 2026-09-21
checkout plus the same code-line counter used then (the files are unchanged from
2026-09-15 to deletion, so git history has the tree).

**Banner (insert after line 3; do not edit line 7):**

```text
Status: accepted 2026-09-21; **superseded 2026-09-27** — the flagship measurement
found fused slower: **1.25×** in `docs/adr/0009-fused-rewrite-rungs.md:3`, with
`AGENTS.md:715-716` recording 9.21 s/step vs burn 7.05 s/step (~1.3×). The module was
deleted in `5abdf3c` (7,432 lines across four files). The 1.7–2.0× smoke in the
Verdict below is **retracted**. The 6,568 LOC figure is UNVERIFIABLE-NOW (the four
files total 7,432 by `wc -l` at `5abdf3c^`). This ADR is kept as the decision record,
not current state.
```

#### D6 — `docs/records/architecture/2026-10-23-final-architecture.md:3,46-48`

**Current:** `Status: design frozen; one A/B outstanding (the ByteFlow arm)`; E=256,
top-k 2, dropless bank as the final recipe. The 2026-11-05 dense verdict measured
dense f=4096 at **2.3195 BPB** vs E=512 rank-64 MoE at **2.3655**, with 2.5× fewer
parameters (`docs/records/reviews/2026-11-05-dense-beats-bank.md:7-10`); the note says the
final run should be dense ~80-150M and the bank does not enter the recipe
(`:36-40`). The live config is dense: `block_form = "transformer"`, `n_experts = 1`,
`moe_topk = 0`, `use_tsct = false` (`port/configs/arch_dense_byteflow8.toml:34,42-44`)
[M/C].

**Replacement for line 3 and a correction after line 48 (keep the original 46-48):**

```text
**Date:** 2026-10-23 · **Status:** **superseded 2026-11-05** — the dense model beat
the expert bank at our data budget (`docs/records/reviews/2026-11-05-dense-beats-bank.md:7-10`:
2.3195 vs 2.3655 BPB at 2.5× fewer parameters). The live run is dense
(`port/configs/arch_dense_byteflow8.toml:34,42-44`). This file is the 2026-10-23
design record, not the current recipe.
```

```text
- **Retracted 2026-11-05 (quality):** the E=256/top-k-2/dropless bank is not in the
  live recipe. Dense f=4096 scored **2.3195 BPB** vs E=512 rank-64 MoE **2.3655**
  (2 seeds, 2,000 steps; `docs/records/reviews/2026-11-05-dense-beats-bank.md:7-10`); TSCT
  remains in the tree behind `use_tsct=false`
  (`port/configs/arch_dense_byteflow8.toml:44`), not as the final recipe. The row
  above is the 2026-10-23 design record.
```

#### D7 — `docs/records/architecture/2026-10-23-scaling-recipe.md:45`

**Current:** "So a large model should be **wide-ish and expert-rich**, not enormously
deep in `d`." The arithmetic before it (width is the expensive direction; experts+rank
are cheap per parameter) stands; the quality recommendation does not. The dense verdict
is 2.3195 vs 2.3655 BPB at 2.5× fewer parameters
(`docs/records/reviews/2026-11-05-dense-beats-bank.md:7-10`), and the bank re-enters only at
≥10× the current data (`:39-40`) [M/C].

**Replacement (keep the original sentence as the retracted claim):**

```text
At 1024 the bank is a rounding error; at 3072 the *dense* side (KDA projections +
`out_proj` + `down`) dominates the step. **The scaling consequence: width is the
expensive direction, and experts+rank are the cheap one per parameter.** The
2026-10-23 sentence "a large model should be wide-ish and expert-rich" is
**retracted 2026-11-05 (quality)**: dense f=4096 scored **2.3195 BPB** vs E=512
rank-64 MoE **2.3655** at 2.5× fewer parameters
(`docs/records/reviews/2026-11-05-dense-beats-bank.md:7-10`); the bank re-enters only at
≥10× the current data (`:39-40`). The parameter arithmetic above stands; the
recommendation is **wide-ish and dense**.
```

#### D8 — `docs/records/architecture/library-crate-fate.md:80,114`

**Current:** the table says line counts are at `422414c` (`:55`) and the row says
`| burn-byteflow | 1121 | 0 | – | **DELETE** | ... |` under `**deleted 2026-09-28**`.
The crate was re-added on 2026-10-02 (`8d61d74`), and the byteflow patch-token channel
was wired on 2026-10-03 (`48ef9ae`) [M]. Today `src` is 1,357 lines and `tests` 832
(2,189 total); it is a live dependency of `dormouse-core`
(`crates/dormouse-core/Cargo.toml:50`) and `dormouse-train`
(`crates/dormouse-train/Cargo.toml:33`), used at `crates/dormouse-core/src/bf.rs:59`
and `crates/dormouse-train/src/byteflow.rs:42` [M]. The old 1,121-line row is the
pre-deletion snapshot; it is history, not a description of today.

**Replacement row and header note (keep the 1,121 row as history if the table is
append-only; otherwise annotate it):**

```text
`src` is `wc -l` over `src/**/*.rs`; `t/b` is `tests/` + `benches/` + `examples/`.
Line counts are at `422414c` unless a row says "post-snapshot". Post-snapshot
changes: `burn-byteflow` was re-added 2026-10-02 (`8d61d74`) and wired 2026-10-03
(`48ef9ae`).
```

```text
| `burn-byteflow` | 1357 | 832 | **W** | **WIRED** (re-added 2026-10-02,
  `8d61d74`; the 2026-09-28 `DELETE` row is history) | live dependency of
  `dormouse-core` (`Cargo.toml:50`) and `dormouse-train` (`Cargo.toml:33`); used at
  `crates/dormouse-core/src/bf.rs:59`, `crates/dormouse-train/src/byteflow.rs:42`.
  The old **1,121 / 0 / DELETE** row was the pre-deletion snapshot. |
```

Line 114's rationale ("nothing in the plan says that, so by default it does not
happen") was true at `422414c`; the code later adopted the patcher **channel**
(`use_byteflow` + `use_kda`, `48ef9ae`), without replacing `dormouse-core`. Keep the
old rationale as history under the post-snapshot header note; do not reuse it as the
current decision.

### 3.1 Group summary

| group | files | claims | action |
|---|---:|---:|---|
| A — fix the body | 3 (`README.md`, `post-training.md`, plan) | 6 | replace the current number/text; keep the old value named |
| B — history, banner only | 5 (two ADRs, two design docs, one snapshot) | 5 | insert `superseded` / `not executed` / `post-snapshot`; do not rewrite |
| **total** | **8** | **11** (10 from §7 + plan line 81) | none archived |

**Вердикт по десяти.** Восемь claim'ов — текущие инструкции/факты, их тело надо
исправить. Два ADR — датированные записи решений: их тело остаётся, сверху
ставится `superseded`/`not executed`. Архивировать нечего: все восемь файлов
всё ещё цитируются как текущие; `docs/records/archive/` не трогаем.

---

## 4. Same defect elsewhere (not in the ten)

### 4.1 The 4,096-param b1 cluster

The old `1,066,354,113` is not only in the ten. Current non-archive occurrences:
`port/README.md:40,70`; `docs/records/research/2026-10-16-token-skipping.md:19`;
`docs/records/research/2026-10-15-tsct-super-kernel.md:17`;
`docs/records/research/2026-10-18-act4.md:252`;
`docs/records/research/2026-10-16-budget-savers.md:165`;
`docs/records/decisions/2026-10-15-1b-readiness-plan.md:81`; `.bulba/memory.md:307`.
All are wrong by 4,096 against `port/configs/b1.toml:110,127` and
`port/tests/test_param_count.py:26`; the fix is the same one-line replace plus
"old = pre-`af8b0c9`". Do them in one commit so the number cannot be half-fixed.
the 2026-10-19 model scorecard exists only in `wt/publish`, not in the live tree
(so it is not a repo path to resolve); if that worktree is ever published, it
already carries both numbers and needs the same note.

### 4.2 Fused "kept" outside ADR-0003

`docs/records/architecture/PLAN.md:442` says the fused smoke is `done, kept (1.7-2.0x)` and
`:456` carries the 1.7–2.0× claim. Both are retracted by ADR-0009:3 and
`AGENTS.md:715-716`, and the directory is gone (`5abdf3c`). Add the same
`superseded` note; do not delete the row (it is the record of the pre-deletion plan).

### 4.3 The committed claims ledger

`docs/records/reviews/2026-11-07-claims-ledger.md:5` (1,627 claims, 0 mismatches, commit
`8229564`) is a dated artifact. It is not current: the working tree has 1,944 claims
and 9 derived mismatches. Either regenerate it in the same commit as the doc fixes or
add a `superseded` header; do not cite its "0" as evidence that the ledger is clean.

### 4.4 The gate's own numbers

`scripts/gate_fast.sh:33-41` calls the ledger with `--no-params` and `|| true`; the
comment says "140 claims dated 2026-11-06/07 predate the rule", while the current run
reports 440 new claims with no evidence (or 373 in the 2026-11-08 data-eval audit's
run). The report-only step hides the two real stale numbers (both params) and prints
one false positive. This is a gate defect, not a doc defect.

Verified after the first draft of this report: a full run gave **1,964 claims / 36
derived / 10 mismatches**. The 20 new claims and the 10th mismatch come from this
file — e.g. the `−0.107` slope in §2 is re-parsed exactly as M9. The report is
evidence, not a new source of claims; widen the self-exclusion to `claims-*` (or add a
`<!-- ledger:ignore -->` marker) before the next gate run, otherwise the repair list
makes the counter worse while fixing the documents.

### 4.5 Audit numbers that need qualification

The audit's §7 is right about the *claims*. Its supporting numbers are mostly right,
but they need qualification before reuse:

| audit says | checked here | note |
|---|---|---|
| `config.py:1585-1587` | `config.py:1586-1588` | off by one |
| `burn-msa` = 1,144 lines; "9 presets" | 1,144 = 718 `src` + 426 `tests` (1,201 with manifest/README); 9 = 7 port + 2 root | both correct once defined |
| 166 test files, 1,865 tests | 167 files, 2,194 collected / 1,870 functions | the tree moved after the audit; use the current count |
| `burn-byteflow` = 2,189 lines | 2,189 = 1,357 `src` + 832 `tests`; the row's 1,121 is the pre-deletion `src` | 2,189 is correct for the current crate; the doc's 1,121 is the stale row |

---

## 5. Do this first

1. **ADR-0014 banner (D1).** A live dependency (`burn-msa`) is described as deleted,
   and the log proves it ran (`~/logs/medium_long.log:11`). Highest risk: a reader
   acting on the ADR deletes a crate `dormouse-core` links.
2. **`post-training.md` RLVR section (D2).** The heading and four bullets present PPO
   + value head as shipped; the code says "no value head" and implements CRR/group.
   Fix the heading and keep the PPO row as "not shipped".
3. **b1 4,096 sweep (D3, D4, §4.1).** One commit: README (2 lines), readiness plan
   (2 lines), token-skipping, tsct-super-kernel, act4, budget-savers, memory. Use
   `port/tests/test_param_count.py:26` as the authority; keep the old value named as
   pre-`af8b0c9`.
4. **`muon-variants.md:16-17` (M1).** Three stale values; the old config is exactly
   `907b3b5^` and the current `medium.toml` is 138,436,993 / 87,452,544 / 11.
5. **ADR-0003 banner (D5).** The directory is gone and the smoke is retracted; add the
   `superseded 2026-09-27` banner, do not rewrite the record.
6. **Dense supersession (D6, D7).** Add the 2026-11-05 retraction to
   `final-architecture.md` and `scaling-recipe.md`; do not archive them until a dense
   design doc exists (they are still the cited design of record).
7. **`library-crate-fate.md:80` (D8).** Add the post-snapshot row; `burn-byteflow` is
   wired, not deleted.
8. **README tests/PyO3 (D9, D10).** Replace the 117 count and the "PyO3 extension"
   sentence; both are current instructions and both are false.
9. **Fix the ledger before re-arming the gate (P1–P5 + §4.4).** The five parser
   classes above produce 7 false positives; `--no-params` hides the 2 real stale
   docs; the self-exclusion misses `claims-repair` and will scan this report. Re-arm
   only after the parser is fixed; otherwise the gate will fail on artifacts and pass
   on stale numbers.

---

## Appendix — commands run (read-only)

```sh
cd /home/sehaxe/dormouse/port
PYTHONPATH=src .venv/bin/python scripts/claims_ledger.py --docs ../docs --repo .. \
  --no-params --out /tmp/ledger.md
PYTHONPATH=src .venv/bin/python scripts/claims_ledger.py --docs ../docs --repo .. \
  --out /tmp/ledger_params.md          # mismatch 9

.venv/bin/python -m pytest tests/test_param_count.py -q       # 6 passed
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_post_rl.py -q  # 24 passed, 1 skipped
.venv/bin/python -m pytest tests/test_msa.py -q               # 7 passed
.venv/bin/python -m pytest --collect-only -q                  # 2194 tests collected

# parameter counts, meta device
PYTHONPATH=src .venv/bin/python - <<'PY'
from dormouse.config import load_run
from dormouse.model import DormouseModel, param_count
for p in ["configs/small.toml","configs/medium.toml","configs/flagship_base.toml",
          "configs/b1.toml","configs/m400b.toml"]:
    print(p, param_count(DormouseModel(load_run(p).model, device="meta")))
PY

# optimizer split
PYTHONPATH=src .venv/bin/python - <<'PY'
from dormouse.config import load_run
from dormouse.model import DormouseModel, param_count
from dormouse.train.optim import is_muon_param
m = DormouseModel(load_run("configs/flagship_base.toml").model, device="meta")
named = list(m.named_parameters())
mu = [(n,p) for n,p in named if is_muon_param(n,p)]
print(param_count(m), len(named), len(mu), sum(p.numel() for _,p in mu))
PY

git show --stat 5abdf3c
git show 907b3b5^:port/configs/medium.toml | grep use_msa
wc -l vendor/dormouse-fused/crates/burn-msa/src/*.rs
grep -rn "use_msa" port/configs port/checkpoints
grep -n "msa_blocks" ~/logs/medium_long.log
python3 tools/check_doc_refs.py        # 5 known-dead named paths, exit 0
```

No GPU training or benchmark was run. The only file written is this report.

Status: complete
Integrity: clean
Contract: aligned
