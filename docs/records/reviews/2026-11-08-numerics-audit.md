# Numerics audit — dtype, overflow, empty paths, aliasing, determinism, checkpoints, the byteflow codec

**2026-11-08 · subagent report · read-only.** Lane: `port/src/dormouse/` at
`0029993e` (`git log -1`), the dense byte-level transformer
(`block_form="transformer"`, 95 M), the KDA/CSA/HCA loop arms, `MixOptimizer` /
Muon, WSD, and `model/byteflow.py` + `model/convcodec.py`. **CPU only** (the GPU
was declared off-limits; no CUDA number below was measured). Every claim carries
a tag: `[M]` measured here, `[C]` computed, `[T]` quoted from a paper or from the
repo's own provenanced docstring, `[S]` speculation with the reason it is not
measured.

Reproduction scripts live in `/tmp/opencode/numerics/`:
`dtype_sites.py`, `probe_losses.py`, `probe_nan.py`, `probe_optim.py`,
`determinism_probe.py`, `ckpt_probe.py`, `codec_probe.py`,
`patch_sensitivity.py`, `ngram_absdisc.py`. The doc site's manifest line
(`docs-site/tools/manifest.mjs`) is **owed** — this file was written read-only.

---

## 0. Вердикты (кратко)

1. **Живая плотная ветка численно чистая.** Ни одного смешанного бинарного
   операнда за прямой проход; правило «fp32 residual, bf16 GEMM» держится.
   Единственные нарушения — в KDA/MoE-ветке, и оба всплывают вверх в fp32,
   то есть безвредны, но неявны.
2. **Найден слепой участок NaN-файрвола:** `fixed_cross_entropy` при
   `+inf` логите на ЦЕЛИ возвращает ровно `0.0` — идеальный лосс. `logit_softcap`
   делает то же для любого `inf` — и лосс, и градиент остаются КОНЕЧНЫМИ, то
   есть не срабатывает ни один файрвол. На дефолтном `log_softmax` оба дают NaN
   и ловятся. Ветки опциональные (0/48 логов), но это не «страховка», а дыра.
3. **LayerRoPE paper-форма имеет неограниченный `exp` от обучаемого скаляра**:
   `a_mag=95` → NaN лосса end-to-end. В живом dense-плейне форма v1/off; в
   снапшотах KDA-ветки — paper.
4. **Muon NS-цепочка не ортогонализует.** 5 шагов дают σ ∈ [0.68, 1.13],
   ошибка против истинного polar 0.20–0.23; 50 шагов — те же числа (2-цикл
   коэффициентов). `ns_steps` 3..8 почти не меняет программу. Арифметика —
   опубликованная (nanogpt/Jordan), дефект — в утверждении «polar factor» и в
   трактовке `ns_steps` как ручки сходимости.
5. **Дефолты конструктора Muon противоречат записанному решению.**
   `Muon(lr_mode="shape_fair")` / `MixOptimizer(muon_lr_mode="shape_fair")`
   против `TrainConfig.muon_lr_mode="uniform"` и докстринга «flipping the default
   before measuring would silently rescale every running experiment». 11 гейтов
   `test_muon_fast.py` красные на HEAD. Тренер не задет (передаёт `cfg`), но
   скрипты/бенчи, строящие `MixOptimizer` напрямую, меряют шаг в 8–55× больше.
6. **Детерминизм на CPU: два процесса — бит-в-бит.** Модель 51/51, оптимизатор
   50/50 тензоров, разные `PYTHONHASHSEED`. Не засеяно: S3T-поток (COUNTED,
   рестарт на resume), негативы ENERGY (глобальный RNG, не посчитан, рука
   выключена), CUDA-атомики `index_add_` в MoE (не проверяемо без GPU).
7. **Чекпойнт: save→load→save бит-в-бит; resume воспроизводит следующий шаг
   точно** (`A6` vs `B6`: модель 51/51, оптимизатор 0 отличий, лосс идентичен).
   Дропнутого состояния оптимизатора не найдено.
8. **Кодек не сломан — сломан интерфейс патча.** Encode→decode каузален (гейт
   проходит), обе струи живые: на 2000 шагах зануление skip стоит 1.12 BPB,
   зануление стека — 0.54 BPB. То есть **побайтовый skip несёт вдвое больше,
   чем весь стек в этом пробнике (1.06M параметров; 95M-арм не проверен —
   чекпойнта нет). Разрыв с byte-армом растёт с обучением (0.76 BPB на
   500 шагах → 1.20 на 2000; срезы разные), а сравнение на 250 MB вдобавок перекошено: у
   патч-арма в 4 раза меньше шагов оптимизатора при том же числе байт.
9. **Conv-кодек при `K<P` падает по индексу, и падение зависит от данных:**
   `_patches` выдаёт `P+1` колонку, глобальный top-k может выбрать фантомную
   `P`, и декодер выходит за границу `[b,P]`. На `K=P` (живой арм) фантомная
   колонка молча отбрасывается — корректно. Гейт на `K<P` не существует.
10. **Собственный стоимостный гейт кодека посчитан не на той ширине:** 2 %-потолок
   заявлен при `d_global=1024`, а пресет идёт на дефолтных 1536 → 525 312 MAC
   на патч = 2.18 %, потолок 482 000 нарушен.

Severity, most to least: **HIGH** = the Muon constructor defaults (red gates +
wrong benches); **HIGH** (dormant) = the two loss-stage blind spots (`ce_fixed` inf, softcap
inf — neither firewall fires); **MEDIUM** = LayerRoPE overflow, NS claim, codec cost gate, patch-interface
verdict, conv codec at `K<P`; **LOW** = NaN on a no-op resume, energy global RNG.

---

## 1. dtype promotion and silent casts

Method: patched `torch.Tensor.__add__/__sub__/__mul__/__truediv__/__pow__/
__matmul__` (and the reflected forms) for ONE forward and recorded every call
whose two floating operands had different dtypes. The harness is self-checked
(a synthetic `bf16 + fp32` is recorded). One thread, fp32 and bf16, four shapes.

| form | forward | mixed-dtype binary ops |
|---|---|---|
| dense transformer, `bf16=true` | `[2,64]`, 2 blocks, d=64 | **0** [M] |
| dense transformer, `bf16=false` | same | 0 [M] |
| `byte_patch` conv codec, `bf16=true` | `[2,64]`, patch 8 | **0** [M] |
| KDA loop (`block_form="kda"`, TSCT+MoE), `bf16=true` | `[2,64]`, 2 iters | **2 sites** [M] |

`dtype_sites.py` output for the KDA loop:

| count | op | dtypes | site | verdict |
|---|---|---|---|---|
| 4 | `*` | fp32 `[128,64]` × bf16 `[128,1]` | `model/loopblock.py:1130` `attn_out.reshape(b*t,d) * w_attn` | upcast to fp32 — the residual dtype; correct, but implicit |
| 2 | `+` | bf16 `[128,128]` + fp32 `[128,128]` | `model/experts.py:1736` `weighted + self.shared(...)` | upcast; the ROUTED sum was accumulated in bf16 before this line |

**1.1 `[M] LOW` the controller gate runs in bf16.** `loopblock.py:1104-1125`:
`raw = mm_mixed(ctrl, controller.weight, compute_dtype)` returns the compute
dtype, so `w_attn = torch.sigmoid(raw[:,0:1])` is **bf16** under `--bf16`, and it
multiplies an fp32 branch. Measured resolution: `sigmoid(bf16(0.25)) = 0.5625`
against `sigmoid(fp32(0.25)) = 0.56218` — a 6e-4 absolute (1e-3 relative)
quantization on every branch's write weight, for the whole run. Not a NaN path.
Blast radius: every KDA/MoE run with `bf16=true` — `flagship_base`, `m400b`, and the `final_dense` / `dense103` / `m400` snapshots. The two live 95 M presets (`flagship3_dense`, `arch_dense_byteflow8`) are the unique-block transformer and build no controller.
Smallest fix: `w_attn/w_mem/w_ffn = torch.sigmoid(raw[:, k:k+1].float())` — one
`.float()`, the same cast `depth_attend` already does explicitly for its softmax
(`loopblock.py:216-222`). Gate: extend `tests/test_step_audit.py`'s dtype census
(or the same patch used here) to assert no fp32×bf16 product in the loop.

**1.2 `[M] LOW` the routed expert sum accumulates in bf16.** `_dispatch` returns
the compute dtype (expert `mm_mixed` outputs, bf16); E outputs are summed by
`index_add_` in that dtype and only the shared branch promotes the sum
(`experts.py:1736`). At E=512 the sum of 512 bf16 terms carries ~9 bits. The live
dense lane has `n_experts=1, moe_topk=0` (no MoE), so no recorded number is
affected. Fix if the MoE lane returns: upcast the accumulator
(`out.float()` before `index_add_`), which costs one temporary per dispatch.

**1.3 `[M] MEDIUM` the Newton–Schulz chain is not the polar factor — it is a
2-cycle.** For a matrix with singular values σ, the Jordan chain is the scalar
map `f(σ) = 3.4445σ − 4.775σ³ + 2.0315σ⁵` (`optim.py:269-282` 2-D,
`:283-300` 3-D). `f(1) = 0.701`, `f` has an unstable fixed point at σ = 0.868 and
a stable **2-cycle 0.68 ⇄ 1.13**. Measured on `[64,64]` and `[256,64]`
Gaussians (`probe_optim.py` §5):

| steps | singular-value range | rel. error vs the true polar `U Vᵀ` |
|---|---:|---:|
| 5 | 0.68 – 1.13 | 0.234 |
| 10 | 0.68 – 1.13 | 0.202 |
| 50 | 0.68 – 1.13 | 0.200 |

More steps do not help: the composition is already on the cycle.
`colrow_normalize` afterwards fixes the *magnitude* exactly (RMS = 0.12500 =
`1/√64` for every `ns_steps ∈ 1..8`, and `RMS(dW)/lr = 1.0000` on `[64,32]`,
`[256,64]`, `[64,512]` — the `shape_fair` claim holds) but leaves the anisotropy:
post-colrow singular values 1.39–2.65, cond 1.65–1.82 for every `ns_steps`
3..8. Consequences: (a) the update is a ~1.7× anisotropic matrix, not an
orthogonal one; (b) the docstring "the polar factor's polynomial approximation"
(`optim.py:234`) is false as stated; (c) `--ns-steps 3` vs `5` is nearly the same
program (cond 1.65 vs 1.72), so a step-count A/B has almost no mechanism to
measure. This is the published Jordan/nanogpt chain, not a port transcription
error — the cycle is a property of the coefficients. The `bf16` chain is fine
*relative to* the fp32 chain (rel 2.9–3.4e-2 across shapes and seeds) and the
bf16 momentum rung drifts 3.6e-3 relative over 20 steps [M]; both are documented
opt-in rungs, off by default. Blast radius: every Muon step's direction.
Reproduction: `zeropower_via_ns(torch.randn(64,64), 5)` then `svdvals`. Smallest
fix: do not touch the arithmetic without an A/B — fix the docstring and add a
fixture gate pinning σ ∈ [0.68, 1.13] so an "orthogonal" claim cannot reappear.

**1.4 `[M] HIGH` (gates and benches; the trainer is safe) the `Muon`
constructor default contradicts `TrainConfig`.** `Muon(lr_mode="shape_fair")`
(`optim.py:892`) and `MixOptimizer(muon_lr_mode="shape_fair")` (`optim.py:1727`)
against `TrainConfig.muon_lr_mode="uniform"` (`config.py:2057`), whose docstring
says *"flipping the default before measuring would silently rescale every
running experiment"* and *"`uniform` also reproduces the pre-fix step
bit-for-bit"*. Measured cause (`_pair` from the test module, explicit kwarg):

| `lr_mode` | `old_batched_step` vs `batched_step_from_model` |
|---|---|
| `"uniform"` | params equal, momentum equal [M] |
| `"shape_fair"` | **params differ at step 0** [M] |

`tests/test_muon_fast.py` builds `Muon` with the default (`_pair`, `:145-150`)
and compares against a transcription of the pre-2026-10-16 step, so **11 tests
are red at HEAD**; every other gate I ran is green. The trainer is safe:
`loop.py:1278`, `sft.py:1115`, `post/rl_train.py:416` all forward
`cfg.muon_lr_mode`. The blast radius is every caller that builds `MixOptimizer`
directly and omits the kwarg — `scripts/bench_opt_graph.py:80`,
`scripts/bench_opt_launches.py:76,91`, `scripts/bench_shape.py:105`,
`scripts/compile_retry.py:121` — those measure a step `sqrt(cols)` larger
(8–55× on the real shapes) than the trainer's, so a bench-run optimizer A/B
measures a different arm. Smallest fix: `"uniform"` in both constructor
defaults (two words); the 11 tests go green. Gate: those 11 tests.

**Null result — the documented rule, restated exactly.** AGENTS §2.2 says "cast
to fp32 before every Linear, cast the residual back". The code does the opposite
and is right: the residual is **always fp32**, and `mm_mixed`
(`tsct.py:259-292`) casts activation *down* to the compute dtype so the GEMM runs
on the bf16 tensor-core path, with fp32 accumulation inside the GEMM. The
census found zero mixed adds in the dense lane because every write is explicitly
`.to(x.dtype)` (`transformer.py:266-274`) and every head/norm is explicitly
`.float()` (`model.py:1103-1108`, `model.py:1403-1406`). The *docstring* is what is
wrong, not the arithmetic; anyone auditing from the docstring would look for the
wrong pattern.

---

## 2. Overflow and NaN sources

| # | finding | evidence | blast radius |
|---|---|---|---|
| 2.1 | `fixed_cross_entropy` with `+inf` on the TARGET logit returns exactly `0.0` | `[M]` `losses.py:139-141`: `zt=+inf → u=-inf → where(u<U_MIN, 0)` | every `ce_fixed=true` run; 0/48 logs carry it (`docs/records/reviews/2026-11-08-deletion-audit.md:138`), but `post/rl_train.py:591` sets it |
| 2.2 | `logit_softcap` maps ANY `inf` logit to `±cap`, so the CE is finite | `[M]` `losses.py:92`; `cap(inf)=15.0`, CE over an inf logit = `7.15e-7`, gradient finite with `-0.0` on the inf coordinate — **no firewall fires** | any `logit_softcap>0` run; no preset sets it (all `0.0`), so dormant |
| 2.3 | LayerRoPE paper form: `exp()` of two unbounded learned scalars | `[M]` `layerrope.py:365-368`; `a_mag=100 → table inf`, `a_mag=95 → model loss NaN` | `layerrope_form="paper"`: the `final_dense`, `dense103`, `m400`, `m400b` snapshots; not the dense lane |
| 2.4 | z-loss | `[M]` `logsumexp^2` finite to 1e19, `inf` at 2e19 → the loss firewall sees it | none — correct |
| 2.5 | dead head (all logits `-inf`) | `[M]` CE = NaN on BOTH paths | none — caught |

**2.1 is the one to fix.** Reproduction:
```python
logits = torch.zeros(1,256); logits[0,7] = float("inf")
F.cross_entropy(logits, torch.tensor([7]))              # nan  -> firewall fires
fixed_cross_entropy(logits, torch.tensor([7]))          # 0.0  -> firewall passes
```
A forward that has exploded onto the target byte reads as a perfect prediction.
The *gradient* on that arm is `-(sum of the others)` = 0, so the gradient firewall
does not fire either; the corrupted weight simply stays. This is the
`8fa5d4c` shape (an arm that reports healthy while doing nothing) on the loss.
Smallest fix: one line after `losses.py:141`,
`loss = torch.where(torch.isfinite(u), loss, u)` — a non-finite `u` then reaches
`isfinite(loss)` in `train/loop.py:1635` and the step is skipped. Gate: add an
`inf`-target case to `tests/test_ce_fixed.py` asserting the returned loss is not
finite.

**2.2** is the same class one layer up: the cap is a *bounded transform of an
unbounded quantity*. Measured on a leaf with one `inf` logit: loss `7.15e-7`,
gradient finite, and the inf coordinate's gradient is exactly `-0.0` — so **no
firewall fires, neither the loss nor the gradient**; the inf logit is pinned at
`cap` and the model takes a silent zero step while printing a perfect loss. On an
eval (`torch.no_grad`, `loop.py:874-965`) there is no gradient check at all, so a
capped eval of a corrupted forward prints a finite BPB. (My first probe reported a
NaN gradient; that was my fixture's `w @ eye` producing `0·inf` on the *other*
logits, not the cap. The clean measurement is the one above.) Smallest fix:
`capped = cap*torch.tanh(logits/cap); return torch.where(torch.isfinite(logits),
capped, logits)`, so the inf survives to the firewall that does exist. Gate: one
test in `tests/test_ce_fixed.py`.

**2.3** reproduction: build `ModelConfig(layerrope=True, layerrope_form="paper",
block_form="kda", …)`, `m.layerrope.sites[k].a_mag.fill_(95)`, forward →
`loss=nan`, `table()` non-finite. `a_mag` gets there only through a corrupted
gradient step — `[S]` on reachability, `[M]` on the mechanism; the point is
there is no bound at all on a parameter that feeds `exp`. Smallest fix:
`r.clamp(max=80.0)` and `(a_rot + b_rot*log_l).clamp(max=80.0)` with a device
counter (`count_device("layerrope_clamped", …)`), since a clamp that fires
silently is the ADR-0019 defect. Gate: push `a_mag=200`, assert the table is
finite and the counter is non-zero.

`[M]` **What is fine:** `_recurrent_step`'s `exp(g)` with `g = logsigmoid(·)`
(`kda.py:185, 362`) is bounded in `(0,1]` by construction — checked, no clamp
needed. `CSA._select`'s `w / w.sum().clamp_min(finfo.tiny)` returns 0 on an
all-masked row instead of NaN (`csa.py:618-621`), again by construction. `RMSNorm`
/ `rsqrt(x+eps)` all carry eps (`loopblock.py:168`, `kda.py:180-181`,
`mhc.py:184`). No `torch.autocast` anywhere in `model/` or `train/` (grep: zero
hits), so the `autocast` interaction the brief asked about does not exist.

---

## 3. Division by zero, empty batch / empty mask

| path | behaviour | verdict |
|---|---|---|
| `masked_ce_mean` with a zero mask | `[M]` `0/0 = nan`, fp32/bf16/fp16 all nan (`loopblock.py:171-181`) | **guarded twice** — `DocPackedStream.next_batch_masked` refuses an all-zero window (`data.py:650-657`), the eval asserts `w > 0` (`loop.py:950`). Null on the live path |
| eval over zero scored bytes | `[M]` `assert weight_total > 0` (`loop.py:962`) | stops; message is a bare assert, not a `loud()`. Cosmetic |
| `grad_accum=0` | `[M]` refused: `validate_train` (`config.py:5043`) and `load_run` (`config.py:5492`) | correct |
| `batch=0`, `seq_len=0`, `steps=0` | `[M]` `validate_train` **accepts** them; `load_run` refuses (`config.py:5492`) | the public seam is guarded; a hand-built `TrainConfig` + `train()` would run a NaN step. Note, not a live bug |
| `eval_batches=0` | `[M]` loud in `eval_window_bytes` (`data.py:853`) | correct |
| mixture with ONE source | `[M]` `source_for_step` returns it for every `u ∈ [0,1)`; `stage_bounds`, `per_source_bytes` fine | correct |
| mixture weights not summing to 1 | `[M]` the draw falls through to `names[-1]` (`mixture.py:384`) | the plan loader refuses a stage whose weights do not sum to 1 (`mixture.py:257-260`, `WEIGHT_TOL`) |
| `select_positions(k > T)` | `[M]` returns `T` columns, not `k` (`byteflow.py:548-550`) | every caller asserts `1 <= k <= P` first (`convcodec.py:281`, `byteflow.py:666`); null |
| `effective_bins(B, T)` for any T | `[M]` always returns a divisor of T (1 divides) | correct |

**3.1 `[M] LOW` a resume with nothing left to do prints `done … loss=nan`.**
`last_loss_dev` is initialised to NaN (`loop.py:1519`) and read unconditionally
at the end (`loop.py:2122`); when `start == cfg.steps` the loop body never runs
and the report line is `done steps=4 loss=nan`. Reproduction (measured):
```bash
# run 4 steps with ckpt_every=4, then resume the same out_dir with --resume
.venv/bin/python -m dormouse.cli train tiny --device cpu --data … --steps 4 \
  --set ckpt_every=4 --set ckpt_name=r --set out_dir=/tmp/x --resume
# -> "resume: step=4 opt_steps=4" then "done steps=4 loss=nan"
```
Blast radius: an orchestrator reading the final loss of a no-op resume sees NaN
and exits 0 (the guard wrapper was removed 2026-10-02, so nothing else notices).
Smallest fix: at the top of `train`, `if start >= cfg.steps: loud("resume at
step N of N: nothing to run; extend --steps or pick another ckpt")`. One guard,
no default change.

---

## 4. In-place aliasing

I grepped all 175 in-place sites in `model/`, `train/loop.py`, `train/graph.py`,
`train/optim.py` and read every one on a returned/saved/accumulated tensor.
**No live aliasing bug found.** The sites that looked dangerous and are not:

| site | why it is safe |
|---|---|
| `loop.py:95` `g.nan_to_num_(0,0,0)` then `:99` `_foreach_mul_(group, gate)` | the ORDER is the fix: `nan_to_num_` removes `inf` before the multiply, so `inf·0 = nan` cannot happen. My first probe multiplied before sanitising and produced NaN *by construction*; the trainer's order is correct, measured |
| `optim.py:2402-2418` `clip_grads`: `coef = max_norm/(total+1e-6)`, `total=inf → coef=0 → 0·inf=nan` | reachable only if `clip_grads` is called before the firewall's `nan_to_num_`. In `opt_stage` (`loop.py:1562-1587`) it never is. A public helper, not a live path — but a caller that clips first gets an all-NaN group [M] |
| `experts.py:1859` `self.lfb_counts.add_(counts.float())` | `_lfb_step` returns early unless `self.training` (`:1851`), so an eval does not move the router — the `7adda92` class is absent here |
| `experts.py:2900/3060/3097/3582` `out.index_add_(0, owner, y)` | `out` is a fresh `zeros`; no other name holds it. (Determinism caveat in §5) |
| `engram.py:502` `acc.index_add_` | fresh `zeros`, host path |
| `experts.py:1931` `dst.copy_(src.detach()+noise*sigma)` | revive, `no_grad`, no other holder |
| `byteflow.py:484/766` `m.diagonal().add_(1.0)` | `m = c·gram`, a fresh tensor |
| `convcodec.py:306`, `byteflow.py:710` `bound.scatter_add_` | int64 counters; integer addition is order-exact |
| `loop.py:871` `_foreach_lerp_(shadow, param)` | shadow tensors are `p.detach().clone()` (`loop.py:820-825`), never the params |
| `transformer.py:297-300`, `mod_route` | `new_zeros(...).scatter_` — fresh buffers |
| `losses.py:146-152` `p.scatter_` | `p` is built two lines above; nothing else references it |

The history's `clone()` + `mask_fill` bug has **no analogue**: `mask_fill`/`masked_fill`
appear nowhere in `port/src/dormouse/` (grep: 2 hits, both in comments).
`_shadow_params` swaps `p.data` for the shadow and restores it in a `finally`
(`loop.py:834-850`); `save_ckpt` is called outside that `with` (`loop.py:2114-2123`),
so a checkpoint never captures the EMA in the raw slot. Checked, correct.

---

## 5. Determinism

**The test the brief asked for: two processes, same seed.** `determinism_probe.py`
and the real CLI, two runs per arm, different `PYTHONHASHSEED` (7919 vs 15838):

| arm | init | after N steps | optimizer state | losses |
|---|---|---|---|---|
| `tiny` preset (KDA + Engram + MoE + Muon + `kda=sdpa`), 4 steps, CLI | — | model **51/51** tensors equal, `moe` 1/1 | muon 27 entries, adamw 23 entries, **0** differing tensors [M] | every `step=` line identical, `moe_dropped_assignments` identical [M] |
| dense transformer, 5 steps, hand-rolled | 19/19 | 19/19 [M] | 19/19 [M] | identical to 12 dp [M] |
| `byte_patch` conv codec, 5 steps, hand-rolled | 37/37 | 37/37 [M] | 37/37 [M] | identical to 12 dp [M] |

So on CPU the same seed **is** the same run, including the Muon NS chain and the
KDA/Engram/MoE forward. What is seeded: `torch.manual_seed(train.seed)` before
the model build (`cli.py:178-194`), `shard_order` from `random.Random(seed)`
(`data.py:148-166`), the mixture draw from `(seed, step)` (`mixture.py:329-384`),
`_sample_depth` from `(seed, step)` (`loop.py:765-777`), the S3T stream from
`factor_qat_seed` (`tsct.py:316-330`), the TSCT widened plane
(`experts.py:506`), `share_u` (`experts.py:568`), revive noise from
`(seed, step)` (`experts.py:1919`). **Dropout does not exist** in the forward.

What would differ between two runs of the same config:

| source | measured? | consequence |
|---|---|---|
| S3T factor-QAT stream (`factor_qat="ternary_s3t"`) | `[M]` the run prints `ternary_s3t … seeded 0 — a resumed run restarts it, so its masks differ from an unbroken run's` (`loop.py:1259-1266`) | a resumed run is a different program; **counted, not silent** |
| ENERGY negatives: `torch.multinomial(..., generator=None)` (`energy.py:130`, no generator passed from `model.py:1288-1296`) | `[M]` mechanism; the arm is off (`energy_weight=0.0` in every preset and in the default) | a resumed `energy_weight>0` run draws different negatives at the same step. The exact class ADR-0021 exists for, and unlike S3T it is **not counted** |
| CUDA `index_add_` / `scatter_add_` in the MoE dispatch | `[S]` — no GPU allowed; `index_add_` is an atomic accumulation whose order varies, so the fp32 sum is not associative | a CUDA MoE run is likely not bitwise reproducible across processes. `scatter_add_` here is int64 (exact) and is fine |
| cuBLAS kernel selection / `mm_dtype` | `[S]` — torch picks a GEMM kernel per shape; the burn side measured a per-process autotune coin flip (docs/records/reviews/2026-10-09-022-autotune-corruption.md) | a cross-process CUDA bitwise claim needs the two-process test run on the card; the port has no such gate |

Smallest fix for the ENERGY hole: pass `generator=torch.Generator().manual_seed(
f(seed, step))` from `model.py`, exactly as `_sample_depth` is spelled. Gate: the
two-process harness with `energy_weight>0` and zero tolerance.

---

## 6. Checkpoint round-trip

Measured with the real `save_ckpt`/`load_ckpt`/`load_model_state` (tiny dense
model, 3 steps, `ema_decay>0`, then one more step from both a straight run and a
fresh model loaded from the step-3 file):

| check | result |
|---|---|
| step-3 weights after `load_state_dict` | **19/19 bitwise** [M] |
| EMA shadow after load | **19/19** [M] |
| resume reproduces the NEXT step's loss | `A = 5.776627540588379`, `B = 5.776627540588379`, **identical** [M] |
| weights after that step | 19/19 [M] |
| optimizer state after that step | 41 tensors, **0** differing [M] |
| save → load → save, file payload | `model` 19/19, `model_ema` 19/19, `opt` 0 differing [M] |
| the real trainer, 6 steps straight vs 4 + save/load + resume | model **51/51**, optimizer **0** differing, final loss identical [M] |

State inventory (what the file carries): `model` (fp32 masters, storage rung
packed), `model_ema` (only when `ema_decay>0`), `opt` = `{steps, muon, adamw}`
(Muon momentum views → one tensor per parameter, `x`/`z`/`v`/`sf_s` when their
arms are on; AdamW `exp_avg`/`exp_avg_sq`/`step`), `moe` (the LFB bias — a
non-persistent buffer), `step`, `model_quant_rung`, and the `<name>.ngram`
sibling for host tables. `_adamw_t` is rebuilt from one restored `step` tensor
(`optim.py:2290-2306`); `_MomGroup.mom` is rebuilt by stacking the loaded views
(`optim.py:1072-1078`).

State that is **not** in the file, and why it is acceptable:

| state | consequence |
|---|---|
| S3T mask-stream position | a resume restarts the stream; COUNTED at startup |
| `MixOptimizer.steps_batched` | a log counter; `opt.steps` is saved and is the one that matters |
| `window`/`total` counters, `nan_skipped` | telemetry, not the objective |
| the data-stream byte position | reconstructed from `step` (`loop.py:1444`, `skip_bytes`) and, for `doc_masking`, by replaying the mixture (`mixture.py:replay_steps`); the resume test above proves it for the flat stream |
| the ENERGY RNG | §5, dormant |

**No dropped optimizer state found.** The one thing to know: a file written by a
`muon_lr_mode="shape_fair"` caller and resumed by the default `uniform` (or vice
versa) is the same tensor set but a different step — `diff_keys` refuses the
config drift only if the snapshot records the field. It does
(`train.muon_lr_mode` is a snapshot key), so the refusal fires. Good.

---

## 7. The byteflow codec

### 7.1 The verdict (item 7 of the brief)

**Кодек не сломан. Сломан интерфейс патча.** The codec is causal end to end
(the repo's gate passes for both codecs, fp32 and bf16, at every `q`), gradients
flow through both exits, and the decoder is not a shortcut: the pre-fix
"patch sees its own bytes" form collapses to **0.89 BPB in 500 steps** on real
text (measured, §7.3), which is the leak the 2026-11-03 fix removes and the
current gate catches. What is wrong is the information the patch latent is
allowed to carry: **patch `i` is built from bytes `[is-8, is-1]` and predicts
bytes `[is, is+7]`; the current patch's own prefix never reaches the stack.**
At position `q = is+j` the model has the previous 8 bytes (in the latent), the
current byte (via the skip), and 2 previous logits (via the decoder's smoothing
conv) — but not bytes `is..q-1`. The per-byte skip then carries the load.

### 7.2 Causality — checked, and one gap in the gate

| check | result |
|---|---|
| `bytes[:,q:]` must not move `logits[:,:q]`, conv + attention, fp32 + bf16, every `q` | **green** [M] (`tests/test_codec_causality.py`, 4 parametrizations) |
| the conv codec at `K<P` | **same uncausal global top-k as the attention codec, and the gate never runs it** [M]: the K<P xfail builds only the attention codec (`tests/test_codec_causality.py:141-171`), while `select_positions` is a global top-k (`byteflow.py:553`) and both codecs call it (`convcodec.py:291`) |
| is a shipped config exposed? | **no** [M]: `arch_dense_byteflow8.toml:53` sets `byteflow_k_tokens = 128 = 1024/8 = P`; `arch_byteflow8.toml` and `arch_kda_byteflow8.toml` do not even load (`load_run` refuses both — dead presets) |
| the decoder reading `h` | `skip(h)` is per-position (`convcodec.py:312`), `smooth` is left-padded (`:313-315`); `h` is the *embedding* of the input bytes, so it carries no future byte |

Smallest fix: add `"conv"` to the K<P xfail's parametrization (one line) so the
finding is loud for both codecs; the chunker itself needs no change while every
live config is K=P.

**7.2b `[M] MEDIUM` the conv codec at `K<P` is an out-of-bounds crash, and it is
data-dependent.** `_patches` pads by `patch_bytes` and then runs a strided conv of
the same size, so it emits **`P+1`** patch features, not `P`
(`convcodec.py:255-256`; measured: `T=32, s=4 → feats [2, 9, 64]`). At `K=P` the
caller takes `arange(P)` and the phantom last column is silently dropped — the
live arm is correct. At `K<P` the selection runs over all `P+1`
(`convcodec.py:291`), so the top-k can pick patch index `P`, and
`decode_chunks` then scatters into a `[b, P]` index
(`convcodec.py:305-307`) → `RuntimeError: index 8 is out of bounds for dimension
1 with size 8`. Measured on the same 32-byte window: `K=8` ok, **`K=6` crashes**,
`K=4` ok, `K=2` ok — the failure depends on the data draw, which is worse than a
deterministic one. `convcodec.py:269` advertises `k < p` as supported, and the
whole rate-chunking saving in the MAC table (`:54-55`) is about `K<P`. No shipped
config is exposed (`arch_dense_byteflow8` is `K=P`; the two other conv presets do
not load), so no recorded number is wrong. Smallest fix: select over
`feats[:, :p]` (one slice, `encode_chunks`), so the selection domain matches the
decoder's. Gate: one `K<P` forward test on a fixture whose top-k takes the tail
(a seeded draw that reproduces the crash), plus the causality check at that K.

### 7.3 Where the patch arm's loss comes from — the zero-stack probe

Tiny but real: `d=128`, 4 blocks, `t=256`, `b=8`, patch 8, `d_global=128`,
`byteflow_codec="conv"`, AdamW 1e-3, `books.bin`, CPU. 2000 steps, 7.2 MB train
/ 0.8 MB held, **equal steps and equal batch for both arms**:

| arm (same data, same 2000 steps, same batch) | params | held-out BPB | CE |
|---|---:|---:|---:|
| byte transformer (the control) | 854 144 | **2.547** | 1.766 |
| patch-8 | 1 060 928 | **3.742** | 2.593 |
| patch-8, `bf.out_proj.weight` zeroed (the stack contributes nothing) | — | **4.285** | 2.970 |
| patch-8, `bf.net.skip.weight` zeroed (the per-byte stream contributes nothing) | — | **4.864** | 3.371 |
| patch-8, both zeroed | — | **6.810** | 4.721 |

Read: **the stack is worth 0.54 BPB; the per-byte skip is worth 1.12 BPB.** In
this 1.06 M probe the stack adds half of what the 64→256 skip linear (16 384
parameters) on the current byte adds. The 95 M arm was not measured (no
checkpoint, §7.6) — do not scale this number to it.
The same decomposition on the 500-step / 4 MB split reads patch 3.705 /
no-stack 5.391 / no-skip 4.683 / neither 7.602 — i.e. the stack's share was
**1.69 BPB** there and *fell* to 0.54 as the skip learned the unigram/bigram
statistics. Within each split the byte arm is ahead (2.940 vs 3.705 at 500;
2.547 vs 3.742 at 2000), and the gap widens with training — the signature of an
information ceiling, not of undertraining.

Counterfactuals, same 1000-step / 4 MB split (`patch_sensitivity.py`):

| variant | held-out BPB | delta vs 3.607 |
|---|---:|---:|
| baseline: patch 8, rank 64, width 64 | 3.607 | — |
| decoder rank 64 → 256 | 3.597 | **−0.010** (null: the rank-64 low-rank head is *not* the bottleneck) |
| patch 8 → 4 (twice the positions, half the discarded prefix) | 3.497 | −0.110 |
| codec byte width 64 → 128 | 3.550 | −0.058 |
| skip reads the codec's causal byte stack (5-byte context) instead of the raw embedding | 3.572 (500-step split) | −0.133 vs 3.705 |
| **the pre-fix patchify, patch `i` sees its own bytes** | **0.890** (500-step split) | −2.815 |

The leak line is the upper bound on what the discarded own-patch information is
worth: ~2.8 BPB at this scale. The causal fixes recover only ~0.1-0.13 of it,
which says the latent/decoder interface is not merely "one byte short" — it is
lossy in a way that smaller patches and a wider skip only partially repair.

Calibration on the same data (`ngram_absdisc.py`, 4 MB train, 200 KB held,
interpolated absolute discounting, D=0.75):

| order k | 1 | 3 | 5 | 7 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| H(byte₊₁ \| last k bytes), BPB | 3.62 | 2.51 | **2.06** | 2.12 | 2.28 | 2.35 |

The patch arm (3.74) is **1.5 BPB worse than a 9-byte counting model** on the
same split, and the byte arm (2.55) is at that counter's level while being
undertrained at 2 k steps. So the codec is not extracting what even a 9-gram of
its nominal context extracts; the per-byte path is doing the work and the stack's
latent is a weak extra.

### 7.4 The 250 MB comparison is confounded

`~/logs/e1v4/patch_s1.log` (patch 3.301 BPB, 7629 steps, batch 32) vs
`~/logs/e1v4/bytes_s1.log` (byte 1.495 BPB, 30517 steps, batch 8). Both scored
196 608 B at `off=0` and both consumed ~250 MB of tokens — but **not the same
number of optimizer steps**: 7629 vs 30517, a 4× difference from the batch size
alone. At this token budget (250 MB against 95 M params, ~2.6 tokens/param) step
count is not a free knob. Part of the 1.8 BPB gap is the batch/step trade, not
the codec. My probe holds the step count equal and still shows 1.20 BPB at
2 k steps, so the gap is real — but **the published 1.80 must not be quoted as
the codec's cost**; the equal-step number is the honest one, and it is not 1.80.

### 7.5 The codec's cost gate is computed at the wrong width

| `d_global` | `mac_per_patch()` | `CODEC_MAC_CEILING` | fraction of the 24.1 M loop | under the gate? |
|---|---:|---:|---:|---|
| 1024 (the gate, `tests/test_convcodec.py:266`) | 459 776 | 482 000 | 1.91 % | yes |
| **1536 (the shipped default, `config.py:1015`)** | **525 312** | 482 000 | **2.18 %** | **no** [M] |

The codec's docstring table (`convcodec.py:50-55`) and the gate use the *profile*
width 1024; `arch_dense_byteflow8` runs the default 1536, so the "2 % ceiling"
claim is false at the shipped width by 9 %. Blast radius: the cost claim only —
no number moves. Smallest fix: build the gate at 1536 too and set the constant to
`LOOP_MAC_PER_LATENT // 45` (2.2 %) with the measured 525 312 pinned, or drop
`byteflow_d_global` to 1024 in the preset. Either way `tests/test_convcodec.py`
is the gate.

### 7.6 The probe the 95 M checkpoint still owes

This repo has **no trained `byte_patch` checkpoint** (checked `checkpoints/`:
`final_dense`, `dense103`, `m400`, `m400b` are all KDA-form), so the requested
"zero the stack output on the real arm" cannot be run here. The exact probe, to
run against the step-7629 patch checkpoint on CPU:

```python
# load the patch checkpoint, then on the 196 608 B eval window:
for attr in ("bf.out_proj.weight",):            # g = 0 -> constant latent
    getattr(model, attr).requires_grad_(False)
    with torch.no_grad(): getattr(model, attr).zero_()
loss_no_stack = evaluate(model, eval_stream, batches=6, device="cpu")
```
Interpretation, fixed in advance: if `loss_no_stack` is within ~0.2 BPB of
3.301, the stack is not carrying the arm and the patch idea is dead; if it jumps
by ≥1 BPB, the stack carries the arm and the byte arm's 1.49 is a target the
codec can be held to. **Do not change any codec default before this number
exists.**

---

## 8. Null results — checked and fine

- **The dense lane has zero mixed-dtype binary ops** in fp32 and bf16; the
  residual never leaves fp32, the head/norm are explicitly fp32.
- **`masked_ce_mean` zero mask** is a NaN by construction but both live call
  sites refuse an all-zero mask (packer: `data.py:650`; eval: `loop.py:950`).
- **`grad_accum=0`, `eval_batches=0`, `batch=0`, `seq_len=0`** are all refused at
  the `load_run` seam (`config.py:5492`, `data.py:853`). `validate_train` alone
  is lax on three of them — noted, not reachable through the CLI.
- **One-source mixture** draws that source for every step; the epoch math is fine.
- **`clip_grads` with a non-finite total** yields NaN, but the trainer always
  sanitises first (`nan_to_num_`); a public-helper caveat only.
- **z-loss, dead head, `select_positions(k>T)`, `effective_bins`, `rsqrt+eps`,
  CSA all-masked row, KDA `exp(logsigmoid)`, `TiedHead` fp32, `_grad_vector_norm`
  with a mixed fp32/bf16 list (exact to the reference)** — all correct.
- **Checkpoint save→load→save and resume-next-step** — bitwise, three different
  models including the real trainer.
- **Two-process determinism on CPU** — bitwise, four models, different hash seeds.
- **Muon update magnitude** — `RMS(dW)/lr = 1.0000` on `[64,32]`, `[256,64]`,
  `[64,512]`; `shape_fair` does what its docstring says.
- **`ns_steps` does not silently rescale the step**: after `colrow_normalize`
  the RMS is exactly `1/√cols` for every `ns_steps ∈ 1..8` [M]. The parity of
  `ns_steps` changes only the anisotropy (cond 1.65-1.82), not the magnitude.
- **`test_muon_fast.py`'s 11 reds are the one red group**; every codec, resume,
  NaN-firewall, mixture, muon-reference and shape-fair test is green (314 lines
  of log, 11 failures, all in `test_muon_fast.py`).

---

## 9. What I could not check

- **Every CUDA number.** The card was declared off-limits. So: the atomics
  determinism claim, the cross-process bitwise claim on the trainer's backend,
  the softcap eval blind spot under bf16, and any wall-time statement are `[S]`.
- **The 95 M patch checkpoint's decomposition** — no such checkpoint exists.
- **A trained `ce_fixed` run** — 0/48 logs (`deletion-audit.md:138`); the inf blind spot is a
  mechanism-level finding, not a measured corruption in a live run.
- **`arch_dense_byteflow8`'s real BPB at equal steps** — CPU-only, 1 M params.

---

## 10. Fix list, ranked, each with its gate

| # | fix | file | gate that catches the regression |
|---|---|---|---|
| 1 | `Muon(lr_mode=…)` / `MixOptimizer(muon_lr_mode=…)` defaults → `"uniform"`, matching `TrainConfig` and the config docstring | `optim.py:892`, `optim.py:1727` | the 11 `test_muon_fast.py` tests (red today, green after) |
| 2 | `fixed_cross_entropy`: a non-finite `u` must reach the firewall | `losses.py:141` | new `test_ce_fixed` case with an `inf` target logit |
| 3 | `logit_softcap`: pass non-finite logits through uncapped | `losses.py:92` | new test: CE over an inf logit is not finite |
| 4 | `LayerRoPE` paper form: clamp the two exponents + a device counter | `layerrope.py:365-368` | push `a_mag=200`, assert the table is finite and the counter fires |
| 5 | `train()`: refuse a resume with `start >= steps` | `loop.py` entry | one CLI test: no-op resume is loud, not `loss=nan` |
| 6 | codec cost gate at `d_global=1536` (or the preset at 1024) | `tests/test_convcodec.py:266`, `convcodec.py:125` | the test itself, at both widths |
| 7 | conv codec K<P in the causality xfail | `tests/test_codec_causality.py:141` | one parametrize |
| 8 | ENERGY negatives from a `(seed, step)` generator | `model.py:1288`, `energy.py:130` | two-process equality with `energy_weight>0` |
| 8b | conv codec `K<P`: select over `feats[:, :p]`, not `P+1` | `convcodec.py:291` | a `K<P` forward test on a draw that picks the tail patch (today it raises `index P out of bounds`) |
| 9 | NS docstring: state the measured band (σ ∈ [0.68,1.13], 2-cycle), and stop calling it the polar factor; do **not** change the arithmetic without an A/B | `optim.py:234` | a fixture pinning the singular-value band |
| 10 | The patch-arm verdict: run the zero-stack probe (§7.6) on the 95 M checkpoint at equal steps **before** any codec/default change | — | the probe's own two numbers |

Items 1-3 are one-line changes. Items 4 and 8 add a counter/generator. Items 9-10
are documentation and a measurement, not code. **No default is proposed to move
without the gate named beside it.**
