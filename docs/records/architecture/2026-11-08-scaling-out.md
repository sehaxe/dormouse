# Scaling out: the smallest honest path from one card to 2–8, and what blocks it

**Date:** 2026-11-08. **Status:** design, not a result. No code changed, no commit, no GPU
job (the card holds a training run). The CPU-side facts were read from the working tree and
from `port/.venv` (torch 2.13.0+cu129; `torch.distributed.is_available()` **and**
`is_gloo_available()` are both True) [M].
**Scope:** the PyTorch port (`port/`). The burn stack (`crates/`, `vendor/`) is measuring
history here, not the thing being scaled.
**Marks:** **[M]** measured in this repo · **[C]** arithmetic done here, formula printed ·
**[S]** speculation, no direct evidence. An untagged sentence is a design decision, not a fact.
**Line numbers:** `port/src/dormouse/train/loop.py` was being edited by another agent while
this document was written (2 150 → 2 402 lines, 2026-11-08 13:00). Every pin therefore names
the **function** and the line as read at 13:05; `optim.py`, `mixture.py`, `data.py`,
`experts.py`, `engram.py`, `graph.py` were unchanged at their mtimes and their numbers are
stable.

---

## TL;DR — вердикт по-русски

1. **Честный минимальный путь — не FSDP.** Реплицировать веса и градиенты, шардить только
   моментум и работу Ньютона–Шульца, и делить **батч, а не шаг**. Тогда чекпойнт, resume,
   eval, роутер и EMA не учатся слову «шард» вообще — а это 80 % риска проекта.
2. **Три вещи сломаются молча, и все три — это состояние, которое сегодня существует в
   одном экземпляре:** флаг NaN-файрвола (`loop.py::train`), интегратор LFB
   (`experts.py::_lfb_step`) и чекпойнт-писатель (`loop.py::save_ckpt`). Пока данные у
   всех рангов одинаковые, ранги не расходятся *вообще* — поэтому эти три бага сегодня
   невидимы и проявятся ровно в тот шаг, когда данные станут разными.
3. **Muon шардится по матрицам, а не внутри матрицы.** Наш NS уже считается в пространстве
   Грама `[64,64]` (`optim.py:273-301`), поэтому шардинг внутри матрицы — это лишний
   all-reduce Грама на каждую из 5 итераций; при `r = min(m,n) = 64` это не экономит ни
   одного запуска ядра. Литература (Dion / Gram-NS) даёт ту же структуру, но нам она нужна
   на уровне *чанка*, а чанк в коде уже есть (`optim.py:1451`).
4. **Порог окупаемости посчитан:** при 95M градиент — 0.38 ГБ на шаг, при b1 — 4.27 ГБ
   [C]. При измеренном шаге 95M = 0.307 с all-reduce это 6 % шага, при b1 = 1.2 с — 18 %
   (при 20 ГБ/с и `grad_accum=1`), и это **амортизируется `grad_accum`, а не железом**.
5. **b1 на 2–8 картах — это уже не strong scaling.** Пер-ранговый батч не может быть меньше
   1, значит глобальный батч растёт, значит нужен LR-арм (3 сида) *до* того, как кривая
   станет сравнимой с записью. 95M–300M на 8 картах — strong scaling с той же кривой.
6. **Что не строить:** FSDP2/ZeRO-3, шардинг внутри матрицы, ZeRO-2 для AdamW (9 МиБ на
   1B [C]), tensor/pipeline parallel, multi-node, `full_graph` под DDP, `engram_ram` под DDP.
7. **Дешевле одной карты:** `ns_dtype=bf16` даёт **1.89×** и уже измерен
   (`docs/records/architecture/2026-11-08-the-16x.md`). Вторая карта — это ~2× на бумаге и 2–4 недели
   инженерии; флаг — это 1.89× за один `--set`. Порядок работ должен это отражать.

---

## 1. What is actually being split

The measured step, so the reader knows which fraction a second card can touch:

| quantity | 95M (`flagship3_dense`) | b1 (1 066 358 209 params) | source |
|---|---:|---:|---|
| measured step, b8·s1024 | 0.307 s [M] | 1.399 s eager / **1.201 s TF32** [M] | `docs/records/research/2026-11-08-1b-on-one-card.md`; `docs/records/reviews/2026-10-16-b1-profile.md:11-19` |
| forward / backward | — | 278 / 404 ms [M] | `docs/records/reviews/2026-10-16-b1-profile.md:11-15` |
| **optimizer** | — | **530–850 ms = 44–53 %** [M] | same, `:15`; `docs/records/research/2026-10-17-muon-variants.md:6` |
| — of which Muon NS matmuls | — | ≈ 535 ms [M] | `docs/records/reviews/2026-10-16-b1-profile.md:40-41` |
| gradient bytes per optimizer step | 0.38 GB [C] | **4.27 GB** [C] | `N·4 B`, N = 1 066 358 209 (README) |
| all-reduce @ 20 GB/s, accum=1 | 19 ms = 6 % [C,S] | 0.214 s = 18 % [C,S] | bandwidth **[S]** — no second GPU here |
| all-reduce @ accum=8 | 0.8 % [C] | 2.3 % [C] | same bytes, 8× the compute per step |

Two facts decide the strategy and both are already in the repo:

* **The step is host-dispatch-bound, not GPU-bound** (`docs/records/research/2026-10-17-muon-variants.md:6`:
  "GPU residue ≈ 0 ms, wall ≈ enqueue"). A second *process* gets a second host thread and a
  second launch queue. That is why 2–8 processes can scale this workload closer to linearly
  than the 13.3 % GPU-utilisation figure (measured on the burn stack, AGENTS §3.1) would
  suggest.
* **The optimizer is the half that shards cleanly.** It is per-matrix arithmetic; two ranks
  doing disjoint matrices need no per-element communication at all. The forward/backward is
  the half that needs the *model* replicated (which it is).

---

## 2. Blockers, ranked

Severity: **S\*** = silently trains a different model; **S** = silently wastes or diverges on
cost; **C** = crash/refuse.

| # | blocker | where (path:line) | what happens under `world_size > 1` | smallest fix | phase | sev |
|---|---|---|---|---|---|---|
| 1 | NaN firewall flag is per-process | `train/loop.py::train` (`flag_dev`, read at 13:05: `1765`, `1882-1919`, `2166-2172`, `2191`) | rank A sees NaN, skips the write; rank B applies it → parameters diverge **permanently**, no mechanism re-syncs them | one `all_reduce(flag_dev, MIN)` before `opt_stage()`; `world=1` no-op | 0 | S\* |
| 2 | LFB controller state is per-rank | `model/experts.py:1421-1430` (buffers), `:1721` (call), `:1828-1874` (step), `:521-537` (the per-expert LR it feeds) | each rank integrates its own histogram → different `lfb_bias` → different router → the "averaged gradient" is an average of **different functions** | `all_reduce(counts)` + `/world` inside `_lfb_step` before the cumulative add; `world=1` no-op | 0 | S\* |
| 3 | no gradient all-reduce seam; grads are mutated in place | `train/loop.py:1894` (`grads = [...]`), `:1919` (`_gate_grads`), `:2185` (`opt_stage`) | each rank optimizes its own slice → N divergent models; nothing is averaged | one bucket all-reduce after the last micro-batch, outside `opt_stage` | 0–1 | S\* |
| 4 | data window is rank-blind | `train/mixture.py:593-605`, `train/data.py:405-408`; resume `train/loop.py:1687-1691` | every rank reads the **same** bytes: correct, and worth 0× the data | slice rows `[rank·b/W : (rank+1)·b/W]` of the same window; cursor and mixture untouched | 0 | S |
| 5 | checkpoint has one writer and no dist metadata | `train/loop.py::save_ckpt` (`574`, payload `614-631`), `.ngram` sibling, `.config.toml` | N processes rename over one path; last-writer-wins; no way to know which world size wrote a file | rank-0 write; add `dist: {world_size, backend, data_split}` to the **v2** payload that landed today | 0 | C |
| 6 | EMA shadow is a full second copy | `train/loop.py::_shadow_state` (`974`), `_shadow_params` (`1020`), `_ema_update` (`1044`) | exact under replicated params, but 4 GiB/rank at b1 [C]; `.data` swap is illegal on a sharded param | keep the shadow on rank 0 only (its params already ARE the global params) | 1 | S |
| 7 | `is_muon_param` reads the **live** shape | `train/optim.py:1667-1690` | under any flat-sharded param every tensor is 1-D → **everything silently routes to AdamW** | carry the routing decision from the unsharded spec; never infer it from a shard | 2+ | S\* |
| 8 | Muon state is whole-matrix stacks; `_foreach_` writers walk a local list | `train/optim.py:970-1070` (`_build_groups`), `:1005`, `:1451` (chunk), `:2143-2185` | the elementwise writers are **fine** on a shard; the aggregating ones are not: `clip_grads`/`_grad_vector_norm` (`:2333`, `:2413`) and `group_grad_norms` (`:2252`) would report a *local* norm as the global one | shard **by chunk** (`:1451` is already the unit); all-reduce one scalar for the clip norm | 2 | S\* |
| 9 | `full_graph` captures forward+backward | `train/graph.py:569-...` (`full_graph_refusal`), module note `:70`; `train/loop.py:1922-2015` | if the all-reduce is outside the captured body, a replay trains on **rank-local** gradients and looks healthy | refuse `full_graph` for `world>1`; `opt_graph` stays legal (it captures the local clip+step) | 0 | S\* |
| 10 | host Engram tables are per-process | `train/loop.py:1512-1519` (setup), `model/engram.py:532-543` (`host_step_all`), `:546` (`host_state`) | 12.3 GB of tables **per rank** at 48M slots [M, AGENTS §2.4.3 as the ledger states it] → 8 ranks = 98 GB on a 64 GB host [C]; and row updates are local, so each rank's memory is a different memory | refuse `engram_ram` for `world>1`; the real fix is a shard-by-hash table + indexed reduce, a project of its own | 0 | C |
| 11 | `tsct_expert_revive` copies weights outside the optimizer | `model/experts.py:1883-1938` | ranks pick different `(hot, cold)` from different counts → a parameter write no collective ever reconciles | refuse for `world>1` (off by default, so no run loses anything) | 0 | C |
| 12 | eval runs on every rank | `train/loop.py:2330-2348` | identical numbers (the eval window is pinned, `train/data.py:844`), N× the cost | rank 0 only; others skip the block | 0 | S |
| 13 | `moe_router_jitter` uses the global RNG | `model/experts.py:1606-1614` | independent noise per rank — a *valid* stochastic router, not a divergence bug; but it is unseeded, so it is already a resume defect (ADR-0021 class) | none for DDP; do not "fix" it by seeding all ranks identically | — | — |

**The logic that makes 1–3 mandatory and 4 optional.** With identical bytes on every rank and
one seed, N replicas stay bit-identical **with no collectives at all** — each computes the same
update. That is also why the three silent bugs are invisible today: they are only *decisions*,
and one rank makes every decision. The moment the data differs (item 4), every rank-local
decision becomes a divergence source, and the only decisions that matter are the ones that are
multiplied into a parameter write (the flag), the ones that change the forward (LFB bias), and
the ones that change which bytes the next step reads (the stream cursor).

**Not blockers, checked:** `_sample_depth` is a pure `(seed, step)` function
(`train/loop.py::_sample_depth`); `factor_qat_alpha` is a pure step function
(`train/lr.py::anneal_alpha`) — the `ternary_s3t` *mask* beside it is a process-local stream
seeded from `factor_qat_seed` (the trainer's own log line says a resume restarts it), but the
seed and the draw count are the same on every rank, so DDP does not make that caveat worse;
`torch.compile` rewriting names is already handled by the `_orig_mod.` strip in
`save_ckpt`/`unpack_state_dict`; the LFB bias *report* printing N times is noise, not a defect.

---

## 3. The three that corrupt silently

### 3.1 The NaN firewall's flag (blocker 1)

The step's decision is a 0-dim int32 tensor (`train/loop.py:1765`), ANDed by each micro-batch
(`:1908`) and multiplied into every update. It is **never** communicated, and the optimizer's
whole write path is scaled by it (`optim.py:1500-1501`, `:2179`, `:2164-2168`).

The failure is not "a bad gradient crosses the wire": if the global step is skipped, the
already-averaged NaN/zero gradient is multiplied by 0 and nothing moves. The failure is that
**the ranks disagree about whether to skip**. Rank A's `flag_dev` is 0, rank B's is 1 → B writes
the step, A does not → the parameters differ by one optimizer step forever, because nothing in
this codebase ever broadcasts parameters. That is a permanent fork that no later healthy step
repairs.

The fix is one collective of **4 bytes** and it is not on the hot path (once per optimizer
step):

```python
if dist.is_initialized():
    dist.all_reduce(flag_dev, op=dist.ReduceOp.MIN)   # device tensor, no host read
```

`MIN` over int32 is the AND. It stays a device tensor, so ADR-0018 rule 2 (no host read in a
step) survives, and `world=1`/`dist`-absent is a no-op — the bit-identity gate in §8 depends on
that.

### 3.2 The LFB integrator (blocker 2)

`lfb_bias` and `lfb_counts` are buffers of the MoE module
(`experts.py:1421-1430`), updated **inside the forward** (`:1721` → `:1828`) from the *local*
routing histogram, and `lfb_bias` is added to the router logits unconditionally. They are not
parameters, so no DDP/FSDP mechanism touches them — and, when `tsct_expert_lr > 0`, the same
counts become a **per-expert LR multiplier** on the Muon rows (`experts.py:521-537` →
`optim.py:2039-2043`). Two ranks therefore route differently, and on that arm they also step
their shared parameters with different per-row scales.

The minimal fix preserves the single-GPU trajectory exactly: all-reduce the per-forward `counts`
and divide by `world` before `lfb_counts.add_(...)`. All ranks then integrate the same global
histogram and move the same bias. One collective of `8·E` bytes per MoE forward (the histogram
is int64, `experts.py:311`; 8 KB at E=1024, `max_iter` times per micro-batch). It serialises the
stream, it does not read a scalar back to the host, and `world=1` is a no-op.

**A cheaper cadence exists and is a behaviour change, not a free fix:** move the bias move out of
the forward into `opt_stage` and all-reduce once per *optimizer* step. That halves the
collectives and matches the paper's per-step controller, but it changes a controller that has
never been A/B'd — so it is its own arm, not Phase 0.

### 3.3 The checkpoint writer (blocker 5)

`save_ckpt` (`loop.py:574`) writes weights, optimizer state, the EMA half and (on the host-table
path) the `.ngram` sidecar, atomically, to *one* path. N ranks doing that simultaneously is N
corruptions racing a rename. Rank 0 must write and everyone else must barrier.

The same function is the right place for the metadata the resume needs:

| key | why it must be in the file |
|---|---|
| `dist.world_size` | the bytes-per-step formula is `batch·seq·accum·world`; a resume at a different world size is legal **only** if the product is preserved (strong scaling), and a loader cannot check what it did not record |
| `dist.data_split` | `rows` (batch split, §5) vs `replicate`; a resume that changes it changes which bytes are seen |
| `dist.backend` | `nccl`/`gloo` — a CPU-written checkpoint resumed on CUDA is legal today; this says what produced it |

The v2 payload (`format_version`, `git_sha`, `corpus`) *landed in the working tree while this
document was written* (`loop.py:614-631`), so this is one dict entry, not a new section. And
`config.diff_keys` already exempts `batch`/`grad_accum` when their product is unchanged
(`config.py:2581-2585`); the same product gains `world_size`, which is a one-line edit plus the
test in §8.

---

## 4. Muon: a shard is not a matrix

### 4.1 What the code does today

* The routing rule is shape-based: 2-D and both dims > 1 goes to Muon (`optim.py:1667-1690`),
  with named exceptions (`embed`, `lm_head`, `controller`, `router`, `tables`).
* State is one **stacked momentum buffer per unique shape** (`optim.py:970-1070`), built once and
  views per parameter (`:1005`), so the step never stacks the group in and copies it back.
* The step is chunked by `n_chunk = max(1, min(n, chunk_elems // (d0·d1)))` (`optim.py:1451`) —
  whole matrices per chunk, so the chunk is already a legal shard unit.
* Newton–Schulz transposes tall matrices (`optim.py:273-275`, `:287-289`) and then runs the
  polynomial in the **Gram space of the small side**: `gram = x @ x.T` (`:278`), `poly @ x`
  (`:280`). The repo's own cost study says this explicitly for both shapes:
  `U [2048,64]` runs as `[64,2048]` and `V [64,5461]` as `[64,5461]`, "both orientations execute
  NS in the `[64,64]` Gram space" (`docs/records/research/2026-10-16-muon-orthogonalization-cost.md` §1).
* The tail is `colrow_normalize` — columns then rows (`optim.py:486-488`) — then
  `p.add_(update, alpha=-lr·scale)`.

### 4.2 What the literature and the repo notes say

| source | what it gives us | what it does not |
|---|---|---|
| **Dion / Gram-NS**, `2608.11612`, repo row: `docs/records/research/2026-09-29-res-optimizer.md:86` ("Gram-NS + CuteDSL + megabatching … distributed (sharded)") | the sharding structure: form the small Gram, all-reduce it, apply the polynomial to the **local block**; never gather the matrix | no number at our shapes; the "6× optimizer step" is their megabatched kernel, not our launch-bound one |
| **Gram-NS regrouping**, `2608.11612` §4, priced for us in `docs/records/research/2026-10-16-muon-orthogonalization-cost.md` §2 row 1 | iterating on the `[64,64]` Gram is *mathematically the same polynomial*, so a sharded implementation needs only an equivalence gate, not a quality A/B | it is a cost cut here, not a task requirement |
| **Orth-Dion**, `2605.16341`, rejected at `docs/records/research/2026-10-16-muon-orthogonalization-cost.md:26-27` | — | its `√r` gain is about **column normalisation** and its rate needs a spectral gap at rank `r`; our `r = min(m,n) = 64` makes `σ_{r+1} ≡ 0`. Take the *sharding shape*, not the normalisation |
| **SF-Muon note** (`docs/records/research/2026-10-16-sf-muon.md`) | nothing about sharding — grepped: zero hits for distributed/shard/ZeRO | it is about the LR schedule, not about where the matrix lives |
| **DDP-Muon notes** | there are none in this repo beyond the two Dion rows above | — |

The `[A]`-style tags in those repo rows are the notes' own; the two papers were **not re-read**
for this document, so a paper-level claim here is a citation of the note, not a re-verification
(ADR-0020).

The naive shard is the failure worth naming: take a flat parameter's byte shard (what FSDP2
does) and run NS on it, and you orthogonalise a **submatrix** — a different update, not a
smaller one. Combined with the transpose rule, half our shapes cannot even form a partial Gram
from such a shard, so the naive path is wrong in two independent ways.

One doc-vs-code disagreement worth flagging while the reader is here (AGENTS §1.7): the
research note's sentence "we never column-normalize" sits against `colrow_normalize`, which
divides by the **column** norm first (`optim.py:487`). The note is about Dion's
*pre-orthogonalisation* column scaling, but as written it reads as a claim about our tail. Fix
the sentence or the code comment when that note is next touched; do not build on it.

### 4.3 The rule for us

**Shard by matrix (or by existing chunk). Never inside one matrix.**

Why, with the numbers:

* A shard of `U [2048,64]` along rows is a column-block of the working `[64,2048]`; the Gram is a
  sum over exactly those columns → `all_reduce` of a **64×64 fp32 = 16 KB** tensor per NS
  iteration, 5 per matrix. That is 80 KB per matrix per step, against 4.27 GB of gradients at
  b1. It is affordable — but it buys nothing, because the NS kernel count is 5 iterations × 3
  matmuls **per rank regardless of how many columns each rank holds**. On a launch-bound step,
  fewer columns per rank is not fewer launches.
* The other orientation is worse, and it is the trap: `V [64,5461]` sharded along dim 0 gives
  each of 8 ranks 8 of the 64 Gram rows → **no rank can form a partial Gram**; the NS cannot run
  at all. A generic `Shard(0)` (FSDP2's default) picks exactly this direction for our wide
  factors. That is the concrete "FSDP2 cannot shard Muon" argument, and it is a property of the
  code's own transpose rule (`optim.py:287-289`), not of the framework.
* The ColRow tail is the other hidden collective: normalising columns is a reduction across rows,
  so a row-shard needs one all-reduce of the norm vector per matrix; a column-shard needs the
  same for rows. Cheap, but it exists and must be written down, or the sharded update is silently
  a different normalisation.
* A **dense** 1B makes intra-matrix sharding actively expensive: `[1792, 6656]` has a `1792²`
  Gram = 12.8 MB fp32, ×5 iterations × 56 matrices ≈ 3.6 GB/step of Gram traffic [C] — as large
  as the gradient. Another reason the answer is "shard by matrix".

Consequence: **Phase 2 is a ZeRO-1 with Muon's shape, not a ZeRO-3.** Momentum and the
orthogonalisation work move to the rank that owns the chunk; parameters and gradients stay
replicated, so nothing is ever gathered.

---

## 5. Data: is the seeded i.i.d. draw rank-safe?

**Yes — and that is why the cheap split is the batch, not the stream.**

The draw is a pure function of `(seed, step)`: `source_for_step` hashes `(seed, step)` and
compares against the stage's cumulative weights (`mixture.py:329-338`, `:374-387`), and the
stream's step is `steps × grad_accum` micro-batches (`loop.py:1419`). There is no RNG cursor to
replay, which is the property ADR-0021 bought. The **cursor** inside a source is sequential
(`data.py:405-408`), and it is shared by all ranks, so:

* If each rank runs the same global micro-step, it draws the **same source** and advances that
  source's cursor by the **same** window → the mixture, the epoch cap
  (`mixture.py:412-431`), the resume formula (`loop.py:1687-1691`) and the replay helpers
  (`mixture.py:434-474`) need **no** changes at all.
* Only the **rows** differ. `ByteStream.next_batch` returns `[batch, seq_len]`; rank `r` takes
  rows `[r·b/W : (r+1)·b/W]` (and the same slice of `loss_mask` on the doc-masking path). The
  cursor still advances by the full `batch·(seq_len+1)` on every rank, so the byte position is
  identical everywhere and the resume stays a multiplication.

That is a ~10-line change at the eager block's two `stream.next_batch*()` call sites
(`loop.py:2144-2154`; the `full_graph` path is refused under `world>1`, blocker 9), plus one
refusal: `batch % world_size != 0` is LOUD. The alternative
(splitting micro-steps across ranks) is *worse*, not just bigger: each rank would own a
different global micro-step, so `MixtureStream._step`, `source_for_step` and both replay helpers
would all need a `(rank, world)` filter — the exact "a number that is a pure function of
`(seed, step)`" property would become a pure function of four arguments, and a resume would have
to re-derive per-rank served bytes or land on the wrong byte. Do not do that.

**Why not split the file range instead?** A contiguous per-rank byte range breaks the mixture
(one stage's sources live in different files), breaks the epoch cap's accounting, and makes the
eval/train overlap check (`data.py:1034`) a per-rank question. Row-slicing the same window keeps
one global timeline.

**And the "per-rank stream seeding" item disappears.** The obvious DDP-ready change is a
`(seed, rank)` stream seed so the ranks read different data. This design needs **no** per-rank
seed: the seed stays one, the draw stays a function of the global step, and the ranks differ only
in *which rows of the drawn window* they read. That is one fewer persisted thing in the
checkpoint and one fewer way for a resume to land on bytes an unbroken run would not have read.

**What the batch split costs in semantics:** nothing measurable. A single-GPU run with
`grad_accum=k` already averages `k` micro-batch means (`loop.py:1884-1888` then the
`n_micro > 1` aggregate at `:2166-2172`); splitting the same window into `W` rank slices and
averaging them is the same operation on the same partition. The only difference is the order of
the fp32 additions, which is the tolerance in §8's gates — not a different objective.

---

## 6. Phased plan

Each phase names (a) what changes, (b) how it is tested **without multi-GPU hardware**, (c) the
risk, (d) the number that kills it.

### Phase 0 — the seam, `world_size = 1` only

| what | where |
|---|---|
| a `dist` seam: rank/world from `RANK`/`WORLD_SIZE`/`LOCAL_RANK`, backend from device (`nccl` cuda, `gloo` cpu), default world 1 | new module, wired at `train()`'s top (`loop.py:1307`) |
| gradient bucket all-reduce after the last micro-batch; the `None` semantics are decided below | between `loop.py:2165` and `:2185` |
| global NaN flag (`MIN`) | before `opt_stage()` (`loop.py:1808`) |
| LFB `counts` all-reduce + `/world` | `experts.py::_lfb_step` (`:1859`) |
| batch-row slicing + `batch % world` refusal | `loop.py:2144-2154` |
| rank-0 checkpoint / rank-0 eval | `loop.py:2330-2348`, `:2356-2361` |
| `dist` metadata in the v2 payload + the `world` factor in `diff_keys`' product | `loop.py:614-631`; `config.py:2581-2585` |
| LOUD refusals for `world>1`: `engram_ram`, `tsct_expert_revive`, `full_graph` | `config.validate_train` (`config.py:5048`) |

**The `grad is None` question — decide it once, in Phase 0.** A parameter with no gradient is
*skipped* by the Muon step (`optim.py:1392-1400`), so its momentum neither decays nor moves. An
all-reduce needs a fixed tensor list, so the bucket must do one of two things:

* (a) carry a presence mask (one `int8` per parameter, one extra collective) and restore `None`
  where the mask is globally zero — exact, preserves today's semantics, one more moving part;
* (b) zero-fill and accept that a **globally** unused expert's momentum now decays by `mu` per
  step (single-GPU freezes it). Two lines, and under a uniform router it touches ~0.03 % of
  experts per step (§7 item 5).

Take **(b)** and say it in the docstring. A silent change of optimizer semantics is the ADR-0019
class; a stated one with a measured rate is a design choice. Revisit (a) only if the measured
rate is not negligible on the live router.

**Tested without multi-GPU:** gloo, two processes, the tiny `conftest` model (d=64, E=4) —
gates G0.1–G0.6 in §8. All of it runs on this box today [M].

**Risk:** the seam changes a 2 400-line function that is being edited by another agent right
now. Mitigation: land it as a separate `train/dist.py` with `world=1` early-returns, so the
trainer's diff is call sites only; and G0.2 is a bit-identity gate that fails on any accidental
arithmetic change.

**Kill criterion:** if `world=1` is not bit-identical (G0.2), the seam is wrong — do not
continue.

### Phase 1 — `world > 1`, replicated state

| what | detail |
|---|---|
| gradient all-reduce | once per optimizer step; `no_sync`'s job is **structural** here (there is no hook to suppress), which is one fewer way to be wrong than DDP's `no_sync()` around micro-batches 0..k-2 |
| strong scaling | `rows_per_rank = batch // world`, global batch and LR unchanged; legal for `batch % world == 0`. The all-reduce **averages** (`sum/world`), it does not sum: the loss is already a mean over the rank's rows (`loop.py:1884-1888`), so the average over ranks is the mean over the global batch |
| weak scaling | `rows_per_rank = batch` (1B: b=1 is the fit shape, `docs/records/research/2026-11-08-1b-on-one-card.md` §TL;DR 1), global batch ×W; **requires an LR arm** — `muon_lr_mult` is the seam (`optim.py:1888`), and the linear rule for a *normalised* Muon update is not published; 3 seeds, `docs/protocols/AB-PROTOCOL.md` |
| rank-0 eval, rank-0 EMA | the shadow's source params are already the global params |

**Tested without multi-GPU:** the two-process gloo run at `world=2` with a **replicated** batch
must be **bit-identical** to `world=1` (every parameter, `torch.equal`) — gloo's all-reduce of
two identical fp32 values is an exact sum and the `/2` is exact, so this gate is bitwise, not a
tolerance. Then a `world=2` **split** batch must match a 1-process control that runs the same two
half-batches as two micro-batches to ≤1e-6 relative over 50 steps.

**Risk:** the all-reduce is the new global cost, and consumer cards have no NVLink — NCCL may be
host-mediated (peer-to-peer is disabled on GeForce SKUs) at ~6–10 GB/s rather than ~20 [S]. That
is why the first hardware measurement is `nvidia-smi topo -m` + one bandwidth test, before any
code.

**Kill criterion:** if the measured all-reduce makes a b1 step >1.4× longer at `world=2` with
`grad_accum=1`, raise `grad_accum` (free amortisation, §1) before writing Phase 2 — and do not
buy the third card until it does.

### Phase 2 — ZeRO-1 with Muon's shape

| what | detail |
|---|---|
| swap all-reduce → **reduce-scatter** | each rank keeps only the gradient rows it will use (its chunks); same bytes on the wire |
| shard the Muon groups by **chunk** | `optim.py:1451` is already the unit; a chunk is whole matrices, so the NS, the ColRow tail and the write-back are all local |
| one scalar all-reduce for the clip | `clip_grads`/`_grad_vector_norm` are global-norm (`optim.py:2333`, `:2413`); the per-rank partial norm is a scalar → `all_reduce` of one float |
| keep `AdamW` whole | it owns 1.12M of 1.03B params at b1 [C]; sharding 9 MiB is not a project |
| params, grads, checkpoint unchanged | **nothing is gathered**, so `save_ckpt` keeps writing a full state dict from rank 0 |

**Tested without multi-GPU:** two gloo processes, same global batch; every parameter
**`torch.equal`** to the single-process run — the shard is a set of whole matrices, the NS is
per-matrix, and the momentum update is elementwise, so this phase is exact, not approximate. The
gate is a bitwise one and it fails if a shard boundary ever cuts a matrix.

**Risk:** the momentum buffers currently exist as one stack per shape, and the checkpoint writes
them per parameter (`_MomGroup`'s docstring, `optim.py:654-690`). A sharded optimizer must
serialise "this rank owns chunks c ≡ r (mod W)" and rank 0 must stitch the union — or every rank
writes its own shard file and the loader concatenates. Prefer the union on rank 0: it keeps the
file format unchanged, which is the whole point of this phase.

**Kill criterion:** if the sharded step is not bit-identical, a chunk boundary cut a matrix —
fix the boundary, never the tolerance.

### Phase 3 — intra-matrix sharding: **do not build**

Kept as a phase only to record what would have to change, and the conditions under which the
answer flips:

* It becomes necessary when one **matrix's momentum** does not fit on one card. At our shapes the
  largest 2-D tensor is tens of MB [C]; the order where it matters is one matrix holding ~250M
  parameters (≈1 GB of fp32 momentum). We are 3 orders below that.
* It becomes *cheap* only if the NS stops being launch-bound (a fused kernel: `ns_kernels.py`,
  `docs/records/research/2026-10-17-muon-variants.md` #1) **and** the Gram all-reduce is fused into it.
* Until both hold, per-matrix sharding is a tax with no product.

---

## 7. What to measure first, in order

1. **Bytes on the wire (free, arithmetic).** 4.27 GB/step at b1 fp32, 2.13 GB bf16 [C]. This one
   number decides whether multi-GPU is a 6 % tax or a 40 % tax; it needs no hardware.
2. **`nvidia-smi topo -m` + one all-reduce bandwidth test on the two cards.** The only number
   that cannot be derived here, and the only one that can kill the plan. [S] until measured.
3. **The 2-process gloo bit-identity gate** (G0.2/G0.3 below). Runs today; proves the seam and
   the three collectives without a second GPU.
4. **The LFB divergence, measured.** 50 steps at `world=2` with different data and the counts
   all-reduce **off**: report `max|lfb_bias(rank0) − lfb_bias(rank1)|`. Today's answer is the
   number that sizes blocker 2; after the fix it must be exactly 0.
5. **The unused-expert rate** at the target batch: the fraction of the 1024 experts with zero
   assignment in one global window. It sizes the `grad is None` semantics: a zero-filled
   gradient makes a *globally* unused expert's momentum decay (`mom.mul_(mu)`,
   `optim.py:1434`) instead of freezing it. At 8192 assignments / 1024 experts, P(0) ≈ e⁻⁸ ≈
   0.03 % under a uniform router [C] — and the measured router is skewed, so the real rate is
   higher. This is a semantic change and must be stated, not discovered.
6. **The optimizer stage's shardability, measured as launch count.** Phase 2's win is
   `1/W` of the enqueues, not of the FLOPs; the b1 profile says the stage is 44–53 % and the NS
   is ≈535 ms of it [M]. Measure the per-chunk launch count once, so Phase 2's prediction is a
   number rather than "should scale".

---

## 8. The Phase 0 acceptance gate

The gates are written to be runnable on this box, and each one is written so that it **fails**
if the thing it names is wrong.

| gate | assertion | why it can fail |
|---|---|---|
| **G0.1** `uv run pytest -q` | the CPU suite stays green (2 194 tests at the README's count; the number moves with the live lane) | the seam touched `train()` and `_lfb_step` |
| **G0.2** `test_dist_identity.py::world_one_is_inert` | 20 steps of `train()` at `world_size=1` vs the same run with the seam bypassed: `torch.equal` on **every** parameter and the same loss bits | a collective that leaks into the arithmetic, a `dist` branch that reorders ops |
| **G0.3** `test_dist_identity.py::two_ranks_replicated_are_bit_identical` | two gloo processes, `data_split=replicate`: rank-0's checkpoint is `torch.equal` to G0.2's | a non-exact all-reduce (should be exact: sum of 2 identical, `/2`), a divergent LFB/EMA/flag |
| **G0.3b** `test_dist_identity.py::a_nan_on_one_rank_skips_both` | force `inf` on rank 1 for one step; both ranks' parameters are **unchanged** and `nan_skipped` is 1 on both | **this is blocker 1's gate**; without the `MIN` all-reduce it fails by exactly one optimizer step |
| **G0.3c** `test_dist_identity.py::lfb_bias_is_rank_identical` | 20 steps at `world=2` with different data per rank: `torch.equal(lfb_bias_r0, lfb_bias_r1)` | **blocker 2's gate**; today's code fails it |
| **G0.4** `test_dist_identity.py::split_batch_matches_the_single_process_partition` | `world=2`, `batch=2·b` split in half vs a 1-process run of the same two half-batches as two micro-batches: every param `allclose(atol=1e-6, rtol=1e-5)` after 50 steps, loss delta printed | a wrong row offset (both ranks read the same rows) makes the loss match the control's *first* half only and then diverge |
| **G0.5** `test_dist_resume.py` | a checkpoint written at `world=2` resumed at `world=1` with the token product preserved: the next step's loss matches within G0.4's tolerance; a resume at a *different* product is refused LOUD | the resume byte formula and `diff_keys`' new factor |
| **G0.6** `port/scripts/sync_audit.py` / `stepaudit` (`port/scripts/step_audit.py`) | the per-step host-read count is unchanged from HEAD; no `.item()` appears in any collective path | a collectives path that reads a scalar back to the host breaks ADR-0018 rule 2 |

**Acceptance:** G0.1–G0.6 green, and a **single-GPU CUDA** run at `world=1` bit-identical to
HEAD (CPU gates cannot catch a backend-specific class — AGENTS §2.5). Only then is Phase 1 a
config flag rather than a rewrite.

---

## 9. What NOT to build (and when the answer would flip)

| not | why (with the number) | flips when |
|---|---|---|
| **FSDP2 / ZeRO-3** | its default `Shard(0)` cuts our wide factors in the Gram direction (§4.3); `is_muon_param` sees 1-D shards and silently routes everything to AdamW (`optim.py:1667-1690`); `_shadow_params` assigns `.data` (`loop.py:1020-1042`), illegal on a sharded param; and params+grads are 8.6 GB at b1 [C], which fits | a single model whose weights+grads exceed the card **after** momentum is sharded, i.e. ≈ 2B params fp32 [C] |
| **ZeRO-2 for AdamW state** | the AdamW group is 1.12M of 1.03B params at b1 → **9 MiB** of state [C]. Sharding it saves nothing and adds a code path | a run with `engram_ram` and tables in-VRAM (402.7M AdamW params, ~3.2 GB of m+v, `docs/records/research/2026-11-08-1b-on-one-card.md`), and even then `engram_ram` is the better lever |
| **intra-matrix Muon sharding** | §4.3: no launch saving at `r=64`; the Gram traffic on a dense 1B is ~3.6 GB/step [C] | a single matrix's momentum ≳ 1 GB, or the NS stops being launch-bound |
| **tensor / sequence / pipeline parallel** | the model fits with replicated weights; the loop block is weight-shared, so pipeline boundaries would cut the same matrices that already shard by chunk | a model whose **single layer's** activations or GEMM do not fit on one card (≳10B dense) [S] |
| **multi-node** | one 16-GB card per rank and no NVLink already; the inter-node link is strictly worse. And the repo's own rule is one heavy thing at a time | not before single-node 8-GPU is saturated |
| **`full_graph` under DDP** | a captured body cannot contain the all-reduce; a capture without it silently trains on rank-local gradients (blocker 9) | a NCCl-aware capture with the `release()` ceremony the reference needs (`train/graph.py:70`) |
| **`engram_ram` under DDP** | 12.3 GB/rank at 48M slots [M] → 98 GB at 8 ranks [C]; and the row updates are rank-local, so the "memory" would be N different memories | the tables get a shard-by-hash layout + indexed reduce, which is a project, not a phase |
| **async SGD / stale gradients / gradient compression** | each is a different objective needing its own 3-seed arm, and the all-reduce is 6 % at 95M | after Phase 1's measured tax says the all-reduce is the bottleneck |
| **DDP's `find_unused_parameters`** | the unused set is the MoE's starved experts and it moves every step (`optim.py:1392-1400` documents `grad is None` as routine), so `static_graph=True` is unavailable and the per-step graph traversal lands on the critical path. The trainer already has the seam (`micro_step` returns `grads`), so a flat bucket is smaller *and* is what Phase 2 needs | if the hand-rolled seam ever needs autograd hooks for a reason this document does not foresee |

---

## 10. Risks

1. **The live file.** `train/loop.py` was being rewritten while this was written. A 2 400-line
   function is where a one-line all-reduce goes missing. Mitigation: a separate `train/dist.py`,
   call sites only, and G0.2 as the tripwire.
2. **The b1 recipe is single-GPU-shaped by construction.** `engram_ram` (48M rows, 12.3 GB) is
   the mechanism that makes 1B fit, and it is the one mechanism that cannot be replicated. The
   honest consequence: **the 1B host-table recipe stays single-GPU until the tables are
   sharded**, and multi-GPU at 1B is a no-tables run (which changes the model). Say this to the
   owner before the second card is ordered, not after.
3. **Weak scaling is a different experiment.** At 1B the per-rank batch is already 1, so any
   `world > 1` grows the global batch and needs an LR arm at 3 seeds. If that arm is skipped, the
   loss curves are not comparable with the record — the exact class the port's whole validation
   culture exists to prevent.
4. **A consumer-card all-reduce may be host-mediated.** No NVLink, P2P disabled on GeForce → [S]
   the effective bandwidth could be 6–10 GB/s, making the b1 tax 35–60 % of a step instead of
   18 %. This is the one risk that can flip the verdict from "design now" to "measure before
   designing", and it is why §7 item 2 is a hardware test and not a calculation.
5. **`world=1` bit-identity is necessary but not sufficient.** CPU gloo passing does not prove the
   CUDA path (AGENTS §2.5's backend-specific class). The CUDA `world=1` bit-identity run in §8
   is not optional.
6. **Phase 2's exactness claim rests on chunk boundaries being matrix boundaries.** If anyone
   later changes `chunk_elems` to a per-element split, the bitwise gate silently becomes a
   tolerance. The gate must assert the shard's shape (`[k, d0, d1]` with `k ≥ 1`), not just the
   values.
7. **This document does not measure anything on two GPUs.** Every percentage that depends on
   interconnect is tagged [S] on purpose. Anybody quoting the 18 % without that tag is quoting a
   guess.

---

## 11. Owed

* `docs-site/tools/manifest.mjs`: `docs/records/architecture/` is an explicit page list, not a glob, so
  this file is not on the site until one line is added (`{ src:
  'docs/records/architecture/2026-11-08-scaling-out.md', out: 'architecture/scaling-out' }`). Owed by the
  next docs commit; this lane was one file only.
* `python3 tools/check_doc_refs.py` was run: **0 named paths missing** (1 275 files scanned,
  3 428 references resolved). It is green with this file in the tree.
* Not run: `npm run check` in `docs-site` (needs the manifest line above) and any GPU test
  (the card is busy). Every claim in §1 that depends on interconnect bandwidth is tagged [S] for
  that reason.
