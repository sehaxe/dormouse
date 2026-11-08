# 2026-11-08 — the fused/CUDA-graph cluster: what the seven reds actually are

**Status: 1 real code defect, 3 broken/never-run tests, 1 too-tight tolerance,
2 card gates that no CPU process can even fail.** The table in
`.bulba/plan.md:159-168` is wrong on two counts: it calls all seven
"pre-existing" and it says several of them pass alone. Only one of them is red
in a CUDA-hidden process, and that one is flaky, not order-dependent.

Tags: `[M]` reproduced on this box by the command named; every `[M]` command in
§9 is CPU-only and under two minutes. `[C]` proved from the code by a runnable
inspection (bytecode/signature/import), behaviour not executed. `[S]` card-only:
no CPU reproduction exists; the exact check that would settle it is named.

All commands run from `/home/sehaxe/dormouse/port`. `uv run --no-extra gpu`
because the CPU venv is the same one; `CUDA_VISIBLE_DEVICES=""` is what makes a
run CPU-only.

**Worktree state at review time.** Another writer is live in this checkout:
`port/src/dormouse/train/optim.py` carries an **uncommitted** fix at `:2071-2085`
(the §1 1a change) and `port/tests/test_optimizer_suite.py` carries its gate
(`test_the_torch_adamw_path_advances_the_bias_correction_counter`, passing on
CPU). Line numbers below are against `HEAD` (`64bf076`) unless marked
"working tree"; the uncommitted diff adds 13 lines after `optim.py:2066`, so
every later `optim.py` reference in this file shifts by +13 in the working
tree. Do **not** apply 1a twice — the uncommitted version already increments
`_adamw_t` once per non-fused step.

---

## 0. The environment contradiction first

The seven are not "the CPU suite at 7 red". Run exactly the seven named nodes
with CUDA hidden:

```bash
cd /home/sehaxe/dormouse/port
CUDA_VISIBLE_DEVICES="" uv run --no-extra gpu pytest -q -rs \
  "tests/test_full_graph.py::test_the_first_replay_is_the_eager_step" \
  "tests/test_fused_epilogue.py::test_the_kernel_is_fp32_roundoff[one-gain]" \
  "tests/test_fused_kda.py::test_kernel_launch_counter_drops" \
  "tests/test_fused_residual.py::test_gate_reddens_when_the_pre_bias_moves" \
  "tests/test_fused_residual.py::test_the_fused_path_issues_far_fewer_launches" \
  "tests/test_invariants_checkpoint.py::test_every_nonpersistent_buffer_is_carried_or_provable[kda+shareu]" \
  "tests/test_moe_combine_fused.py::test_the_fused_combine_counter_is_counted_by_the_kernel"
```

Result in this process: **6 skipped, 1 failed** — and the failure is
`test_invariants_checkpoint.py:160` (the flaky one, §6), not any of the kernels.
In another process the same command is **6 skipped, 1 passed** (§6 is
seed-dependent). `[M]`

So five of the seven (`test_full_graph`, both `test_fused_residual` card gates,
`test_fused_kda`, `test_moe_combine_fused`) **skip** without CUDA:

```
SKIPPED tests/test_full_graph.py:519                  no CUDA device
SKIPPED tests/test_fused_epilogue.py:466              needs the card
SKIPPED tests/test_fused_kda.py:453                   needs the card
SKIPPED tests/test_fused_residual.py:119              needs the card
SKIPPED tests/test_fused_residual.py:154              needs the card
SKIPPED tests/test_moe_combine_fused.py:410           needs the card: the combine asserts CUDA
```

`torch.cuda.is_available()` is **True** on this box, so a plain `pytest` run
executes them and uses the card. The "7 red CPU suite" therefore cannot have
come from a CUDA-hidden process; it came from a run with the card visible (or
from a different worktree). Anyone re-reading the list as "CPU-only" will chase
five failures that do not exist there.

| # | test | class | verdict | evidence |
|---|---|---|---|---|
| 1 | `tests/test_full_graph.py:520` | **real defect** (optimizer fallback) | the capture is fine; the *eager reference* changes AdamW implementation mid-run (`_adamw_t` desync); 1a applied uncommitted in the worktree | `[M]` CPU sim 4.51e-05 @ `embed.weight`; `[C]` `optim.py:2086,2130,2237` |
| 2 | `tests/test_fused_epilogue.py:468` | **stale tolerance** | 1e-6 on a gradient where this module's own rule is 5e-5; the kernel is unchanged since the test was written | `[M]` CPU mirror 4.6e-7…8.5e-7 rel; `[S]` the kernel |
| 3 | `tests/test_fused_kda.py:454` | **unverified card gate** | not run when written (`a28716a` says so); no mutable module state to leak; absolute count with no floor | `[C]` globals probe; `[S]` mixed-run claim |
| 4 | `tests/test_fused_residual.py:120` | **stale test** | patches `fr.pre`; `parity()` calls `epilogue_read`, never `pre` | `[M]` bytecode + 1e-4 nudge gives 6.09e-05 where tolerance is 1.32e-05 |
| 5 | `tests/test_fused_residual.py:155` | **unverified card gate** | no CPU evidence; the fused-residual file has no arithmetic change since the test landed | `[S]` |
| 6 | `tests/test_invariants_checkpoint.py:123` | **flaky fixture** | unseeded global RNG + the sign-LFB integrator returning exactly zero | `[M]` seeds 1/4/9 red, 0/2 green; it also fails file-alone |
| 7 | `tests/test_moe_combine_fused.py:402` | **stale test; first run 2026-11-08** | asserts `moe_combine_bf16` without asking for a bf16 store; the counter is host-side, not kernel-side | `[M]` CPU probe: `KeyError: 'moe_combine_bf16'`; `[C]` `moe_kernels.py:1604` |

---

## 1. `test_the_first_replay_is_the_eager_step` — real defect in the fallback, not in the graph

**Вердикт: это реальный дефект кода, но не в CUDA-графе. Граф пишет шаг
правильно; эталон, с которым тест сравнивает, в момент падения capture
переключается на другую реализацию AdamW и считает bias correction с t=1
вместо t=6. Половина 1a уже исправлена в рабочем дереве (uncommitted);
вторая половина — Muon LR в `_fall_back` — ещё нет.**

### What the test actually fails on

In a clean run the capture **succeeds**. This is `[S]` — it needs the card —
but it is the run whose output the plan's table is describing, and it fails at
the numeric assert, not at the log asserts. The two log assertions at
`tests/test_full_graph.py:560-561` pass; the failure is the one at `:568`:

```
[first replay] worst |delta| = 3.147e-05 @ embed.weight; loss 4.46668434 vs 4.46668434
AssertionError: the capture moved the arithmetic: (3.147125244140625e-05, 'embed.weight')
assert 3.147125244140625e-05 <= 1e-06
```

The plan's line ("the reference log never says 'capture failed'") describes a
*different* run — one in which the capture arm logged the `_fall_back` line
(`train/graph.py:543-545`). The reason could be headroom, a stale graph pool, or
something else; print it. That is a separate, card-side failure mode; it is not
what the clean run reports and it is not what this section fixes. `[S]` for
both the clean numeric failure and the mixed-run refusal.

Two facts pin the cause:

* the loss after six steps is identical to nine digits — the last forward is
  the same, so the divergence is in the **optimizer update**, not in the
  captured forward;
* the worst parameter is `embed.weight` — an **AdamW** parameter
  (`optim.is_muon_param` sends `embed` to AdamW, `optim.py:1671-1685`). Muon
  parameters go through the same code in both arms and do not show up.

`[S]` for the observed numbers; `[M]` for every mechanism below, which is
reproduced on CPU. The test's premise is not reachable on CPU even without its
`@CUDA` marker: `train()` loud-refuses `full_graph` on a non-CUDA model before
the first step (`train/loop.py:1691-1700`), so no CPU process can capture.

**Counter answer:** on the CPU path neither `full_graph_captured`
(`train/graph.py:418-435`) nor `full_graph_fallback` (`:536-538`) can be
emitted, because the arm is refused before a `FullGraph` object exists. The
test does not assert those counters — it asserts two log lines — but the log
line it demands ("capture failed") is written by `_fall_back`, so the same
premise (a capture was attempted) is what makes it reachable.

### The mechanism

`FullGraph.__init__` calls `_bind_device_lr` (`train/graph.py:151-157`): every
group's `lr` becomes a 0-dim device tensor and every AdamW group gets
`capturable=True`. `MixOptimizer._fast_adamw_ok` refuses the fused writer when
either is present (`train/optim.py:2086-2093`), so **the whole warmup runs
`_gated_adamw_step` → `self.adamw.step()`** (`optim.py:2237`), i.e. torch's
capturable AdamW, whose step counter is the per-parameter `state["step"]`.

`_adamw_t`, the host counter the fused writer uses for bias correction, is at
`HEAD` **only incremented in `_foreach_adamw_step`** (`optim.py:2130-2131`) and
**only synced on `load_state_dict`** (`optim.py:2296-2305`). During the warmup
it stays 0.

At step 6 the two runs diverge:

* **capture arm:** capture + replay of `opt_stage()`; torch AdamW at t=6.
* **eager reference:** `FullGraph.capture` is monkeypatched to raise, so
  `_fall_back` restores float `lr` and `capturable=False`
  (`graph.py:520-523`). `_fast_adamw_ok()` now returns True, so the step runs
  `_foreach_adamw_step`, which does `_adamw_t += 1` → **t=1**.

That is a real program difference, not "the fp32 ulp the recording costs". The
`_foreach` implementation itself is fine — a CPU probe shows it is
**bit-identical to `torch.optim.AdamW` for six steps** (`[M]`):

```
worst |MixOptimizer(foreach) - torch.optim.AdamW| = (0.0, '')
```

The 3.1e-05 matches the *desynced bias correction*, simulated on CPU by running
five `_gated_adamw_step` steps and then one fused step, with and without
advancing `_adamw_t` (`[M]`):

```
after 5 steps _adamw_t = 5  state step = 5.0
stale vs synced (the proposed one-line fix): 4.506111145019531e-05
scale 4.346385955810547
```

Same order, same parameter class, same signature as the live 3.147e-05. The
live number differs only because the live model/shape differ.

### Second latent defect found while reading the same fallback

`_fall_back` restores **every** group to `self._lr_value` (`graph.py:521-523`),
which is the WSD `lr` passed to `full.set_lr(lr)`. But the loop immediately
before `full.step()` calls `opt.set_lr(lr, muon_lr)` (`loop.py:1875-1884`),
which fills the Muon group with `muon_lr * muon_lr_mult`. The fallback step
therefore runs the Muon half at `lr` instead of `muon_lr * mult`. With the
default `muon_lr_mult=1` and `muon_sf=false` it is invisible; with
`muon_lr_mult != 1` it is one wrong step, silently. It is in the same function
and the same failure path, so it belongs in the same fix.

### Fix

1. **Already applied in the working tree** (`src/dormouse/train/optim.py:2071-2085`,
   uncommitted at review time, not by this review): after the non-fused
   `self.adamw.step()`, advance `_adamw_t` by one when at least one AdamW
   parameter has a gradient. The gate is
   `tests/test_optimizer_suite.py::test_the_torch_adamw_path_advances_the_bias_correction_counter`
   (passing on CPU). **Do not also add the increment inside
   `_gated_adamw_step`** — that would double-count. Verify the gate, then
   re-run `test_the_first_replay_is_the_eager_step` on the card.
2. `src/dormouse/train/graph.py:521-523` — restore the host float only for the
   AdamW groups (`self._capturable_restore` already names them); leave the
   Muon group's device tensor alone — `opt.set_lr` has just filled it with
   `muon_lr * mult`, and the batched Muon step explicitly supports a 0-dim
   tensor (`optim.py:1368-1371`). The next loop iteration's `full.set_lr`
   replaces it with a float anyway. This one is **not** in the working tree.

**Gate for 2:** a CPU wrapper test: `set_lr` + `opt.set_lr` + a forced
`_fall_back` → each group's LR equals what the eager step would read. The
`_adamw_t` gate is the one already added.

**Risk:** low. The change only fires when the implementation switches mid-run
(capture fallback). A run that never captures, or captures and never falls
back, is untouched; run `test_muon_launch`, `test_nan_firewall`,
`test_opt_region` after the change to confirm the goldens do not move. The
whole `tests/test_optimizer_suite.py` is green on CPU with the applied 1a
(36 passed, `CUDA_VISIBLE_DEVICES=""`). Do **not** loosen the 1e-6 bound in the
test before the card re-run: 3e-05 is real.
After the fix, if a residual ~1e-7 remains (torch capturable vs the fused
writer), that residual is the only tolerance decision left.

---

## 2. `test_the_kernel_is_fp32_roundoff[one-gain]` — the tolerance is wrong, the chain is not

**Вердикт: 1.37x за допуском — это допуск, а не цепь. Тест применяет 1e-6 к
градиенту, тогда как собственное правило этого же модуля для градиентов —
5e-5; ядро и его fp32-путь не менялись с коммита, который написал тест.**

`[S]` for the kernel number itself: the test is `@CUDA` and there is no CPU
reproduction. Do not confuse it with `test_read_parity_is_fp32_roundoff`
(`tests/test_fused_epilogue.py:147`), which is the CPU mirror of the same
formula and is **green** here (`CUDA_VISIBLE_DEVICES="" pytest
tests/test_fused_epilogue.py` passes). What is reproducible:

* `git diff acb013f HEAD -- src/dormouse/model/fused_residual.py` changes only
  formatting, the refusal strings and the `unsupported_reason(kernel=...)`
  argument. `_Pre.forward`, `_pre_kernel`, `_Pre.backward` and
  `_bwd_pre_kernel` are byte-for-byte the same. The eager reference on the
  fp32 path is a no-op (`mhc._x_norm`'s `.float()`/`.to(phi_pre.dtype)` are
  identity on fp32, `mhc.py:183-184`). `[C]`
* The CPU mirror of the same formula passes the same 1e-6 bound with room:
  `parity_read(b=2, t=8, separate=False)` → `d_b_pre` Δ=4.77e-07 on a
  reference of 1.047; `parity_read(b=3, t=8, separate=True)` → Δ=4.29e-06 on
  5.039 (rel 8.5e-07). `[M]`
* `_assert_read_parity` (`tests/test_fused_epilogue.py:132-145`) applies
  **one** `rtol=1e-6` to outputs and gradients. The module it is testing
  states the tree's rule ten lines below `parity`: `PARITY_RTOL = 1e-5` for
  forward, `PARITY_GRTOL = 5e-5` for gradients, with a degenerate-tensor rule
  (`fused_residual.py:1264-1300`). `fused_kda.GRTOL` is also 5e-5. `[C]`
* The same test, via
  `fr.assert_parity(fr.parity(..., epilogue=True))`
  (`tests/test_fused_epilogue.py:470`), **already** compares the same
  `d_b_pre` at 5e-5. The 1e-6 pass is redundant *and* 50x stricter than the
  rule the same file's other half uses. `[C]`

`d_b_pre` is a sum over `b·t` rows of `h_pre(1-h_pre)·dctx` — a
cancellation-sensitive reduction. An absolute error of 2.861e-06 on a
reference of 2.082 is ~11 fp32 epsilons; a real wiring error is O(1) (the
`H_pre` case in the test's own docstring: 8.1e-4) or O(1e-4) (the §4 probe:
6.1e-05). `[C]`

**What would overturn this verdict:** run the kernel and print every entry
(the CPU mirror prints them; the test only prints the worst). If `ctx` or
`normed` (the forwards) are also at ~1e-6 relative, the chain moved; if only
`d_b_pre` is over, it is reduction noise. Command:

```bash
cd /home/sehaxe/dormouse/port && uv run --extra gpu python - <<'PY'
from dormouse.model import fused_residual as fr
for sep in (False, True):
    rep = fr.parity_read(b=2, t=8, d=64, separate=sep, use_kernel=True)
    print("separate", sep)
    for k, v in sorted(rep.items()):
        print(f"  {k}: {v}")
PY
```

### Fix

`tests/test_fused_epilogue.py:132-145` — split the tolerance in
`_assert_read_parity`: keep `1e-6` for `ctx`/`normed`/`normed_mlp`, use
`5e-5` for every name starting with `d_` (cite `fused_residual.PARITY_GRTOL`).
The two mutation gates in the same file (`:158-186`) all perturb forward
tensors and keep biting, and the write chain's `assert_parity` still pins the
gradients at 5e-5.

**Gate:** the kernel test itself plus
`test_the_kernel_gate_reddens_when_the_gain_moves`
(`tests/test_fused_epilogue.py:550`). **Risk:** a genuine 1e-5-scale gradient
error in the read chain would now pass; that scale is still covered at 5e-5 by
the same test's `assert_parity(parity(..., epilogue=True))`. Do not raise the
forward bound — it is 4.6e-7 on the mirror and a forward defect is what the
epilogue is for.

---

## 3. `test_kernel_launch_counter_drops` — an unverified card gate, not a known kernel defect

**Вердикт: классифицировать как «реальный дефект» нельзя. Тест ни разу не
исполнялся на карте (это написано в коммите, который его добавил); на CPU он
skip; в модуле нет изменяемого состояния, которое могло бы «протечь».**

`a28716a` (the commit that added `fused_kda.py` and this test) says the GPU
gates and the bench "are written; a multi-day training run owns the card, so
they were NOT run." The plan's "passes alone, red in a mixed run" is therefore
the first execution(s), not a regression. `[C]`

There is no shared mutable state in the module. A CPU probe of module globals
found **zero** mutable globals in `fused_kda` and `fused_residual`; only
`moe_kernels` has shape caches (`TILES`, `TILE_SEARCH`, `_WGRAD_STAGES`), and
`fused_kda` does not use them:

```bash
cd /home/sehaxe/dormouse/port && uv run --no-extra gpu python - <<'PY'
import sys, types; sys.path.insert(0, "src")
import dormouse.model.fused_kda as fk, dormouse.model.fused_residual as fr, dormouse.model.moe_kernels as mk
for m in (fk, fr, mk):
    g = [(k, type(v).__name__) for k, v in vars(m).items()
         if isinstance(v, (dict, list, set)) and not k.startswith("__")]
    print(m.__name__, g)
PY
```

Output: `fused_kda []`, `fused_residual []`, `moe_kernels [TILES, _WGRAD_STAGES, _DTYPE_NAMES, TILE_SEARCH]`. `[M]`

The only process-global dependency the test actually has is
`torch.profiler`/kineto (initialised process-wide by
`test_fused_residual.py:146` and `test_fused_epilogue.py:482` in a mixed run)
and Triton's JIT cache. That is a hypothesis, not a finding: no CPU process can
run the test to distinguish it from a plain mistake in the test's premise
("one tail launch + one combine launch where the eager arms make ELEVEN" is an
unmeasured sentence).

Also: the test's bounds are brittle in ways that will look like order
dependence:

* `assert n_fused <= 3` is absolute with one launch of slack and **no floor on
  `n_fused`**. If the profiler yields 0, the assert passes (`0 <= 3`) and the
  test proves nothing; it does not print the counts' origin.
* `n_eager - n_fused >= 8` depends on the eager arm's launch count, and the
  assertion message prints only the two counts when it fails — the eager count
  is never printed on a pass, so a baseline move is invisible.

**Check that settles it (card only, `[S]`):**

```bash
cd /home/sehaxe/dormouse/port
uv run --extra gpu pytest -q -s tests/test_fused_kda.py::test_kernel_launch_counter_drops
uv run --extra gpu pytest -q -s \
  tests/test_fused_residual.py::test_the_fused_path_issues_far_fewer_launches \
  tests/test_fused_kda.py::test_kernel_launch_counter_drops
uv run --extra gpu pytest -q -s \
  tests/test_fused_kda.py::test_kernel_launch_counter_drops \
  tests/test_fused_residual.py::test_the_fused_path_issues_far_fewer_launches
```

The counts are printed by both tests. If the same test prints different
`fused`/`eager` numbers in the two invocations, the profiler state is the
variable; if it prints the same numbers and only the assertion differs, the
test's arithmetic is wrong. Do not change the kernel before that output exists.
**No CPU reproduction of the ordering exists** — both nodes skip without CUDA.
That is the honest limit of this report: the state named above is a
*process-global*, not a module counter, and only the card can show whether it
matters.

**Minimal test hardening (not the fix):** `assert n_fused > 0` (a zero count
must not pass), print both counts on the pass path too,
`torch.cuda.synchronize()` before opening the profiler, and assert the
deterministic `Counters` half (`drained["fused_kda_tail_launches"] == 2`)
*before* the profiler half so a failure names which half broke.

---

## 4. `test_gate_reddens_when_the_pre_bias_moves` — the gate is dead; it patches a function the path no longer calls

**Вердикт: устаревший тест. Он monkeypatch'ит `fr.pre`, а `fr.parity` с
коммита `acb013f` вызывает `epilogue_read`; `pre` в этом пути не вызывается
вообще. Поэтому 1e-4 возмущение не применяется, и Δctx — это обычный
roundoff 1.19e-07.**

Before `acb013f`, `parity`'s fused branch called `pre(mhc, xl, c)` — the test
was right. `git show acb013f^:...fused_residual.py` and `git show
acb013f:...fused_residual.py` show the switch. Today:

* `src/dormouse/model/fused_residual.py:1214-1215` — `parity`'s nested `run`
  calls `epilogue_read(...)` and `write(...)`.
* `tests/test_fused_residual.py:126-135` — patches `fr.pre` and calls
  `fr.parity(...)`.

CPU proof that `pre` is never looked up (bytecode of the nested `run`):

```bash
cd /home/sehaxe/dormouse/port && uv run --no-extra gpu python - <<'PY'
import sys, types; sys.path.insert(0, "src")
from dormouse.model import fused_residual as fr
run = [c for c in fr.parity.__code__.co_consts if isinstance(c, types.CodeType)]
print({c.co_name: sorted(set(c.co_names) & {"pre", "write", "epilogue_read"}) for c in run})
PY
```

Output: `{'run': ['epilogue_read', 'write'], '<genexpr>': []}` — no `pre`. `[M]`

The gate is not wrong in principle: a 1e-4 nudge of `b_pre` in the *fused read
arm* does redden `assert_parity` on the forward. CPU measurement of the same
formula (b=3, t=8, d=64, the `parity` init):

```
nudge 0.0001: dctx=6.092e-05 (|ctx| 1.324e+00, tol 1.324e-05) dnormed=3.183e-05 (tol 3.514e-05)
```

4.6x over the forward tolerance. `[M]` So once the test patches the function
`parity` actually calls, it will bite.

### Fix

`tests/test_fused_residual.py:126-135` — `real = fr.epilogue_read` and wrap it
with the signature `(mhc, xl, gain, gain_mlp, eps, counters)`; keep the same
nudge/restore. The docstring should say `epilogue_read`, not "the pre kernel".
`fr.pre` is still the right function for the *write-chain* launch test
(`tests/test_fused_residual.py:159-161` calls it directly), so leave that one.

**Gate:** the test itself; it is currently the only thing that would catch a
`b_pre` wired into the wrong tensor. **Risk:** none — the patch is scoped and
restores `b_pre` in `finally`.

---

## 5. `test_the_fused_path_issues_far_fewer_launches` — no CPU evidence, and no arithmetic change since it was written

**Вердикт: это card-gate; на CPU он skip, и никакого CPU-воспроизведения нет.
Файл ядра не менялся по арифметике с коммита, который добавил тест, поэтому
«регрессия launch count» ничем не подтверждена.**

`[S]`. The test is `@CUDA` (`tests/test_fused_residual.py:154`) and calls
`fr.pre`/`fr.write` directly, so it cannot run in a CPU process. `git diff
acb013f HEAD -- src/dormouse/model/fused_residual.py` shows no change to
`_pre_kernel`, `_write_kernel`, `_bwd_write_kernel` or `_Write` (only the
refusal string and formatting). `[C]`

The two plausible failure modes, neither verified:

* the profiler reports no `cudaLaunchKernel` events (then both counts are 0 and
  the print divides by zero — `scripts/bench_opt_graph.py:148-150` has an
  explicit note for exactly this build condition);
* a Triton launch was refused/fallen back for a shape the test does not print.

**Check that settles it:** run it with `-s` and print `n_eager`, `n_fused`, and
whether `fr.fused_available()` is True in that process. If `n_fused == 0`, it
is the profiler, not the chain. If `n_fused` is O(eager), the fused path fell
back and the reason must be printed. Do not touch the kernel before that.

Same hardening as §3: `assert n_fused > 0`, `max(n_fused, 1)` in the print,
and a `torch.cuda.synchronize()` before the profiled call. The 5x ratio itself
is the arm's stated claim (`fused_residual.py:52-60`: 729 → 29 launches on the
region shape); keep it.

---

## 6. `test_every_nonpersistent_buffer_is_carried_or_provable[kda+shareu]` — flaky fixture, not a checkpoint defect

**Вердикт: тест флаки, а не код. Он не сеет глобальный RNG, а sign-LFB
интегратор за 5 шагов может вернуться ровно в ноль; тогда падает
собственная проверка «фикстура действительно сдвинула bias», хотя сам гейт
чекпоинта выше прошёл.**

The failing assertion is `tests/test_invariants_checkpoint.py:158-162`:

```
fresh = DormouseModel(run.model)
assert not torch.equal(live, _lfb_modules(fresh)[mod].lfb_bias),
       "the fixture must actually move lfb_bias"
```

`live` is `[0,0,0,0]`. The checkpoint gate above it passed: the file carries
the bias (`assert torch.equal(payload["moe"][mod], buf)` at `:155`). The bias
is zero because the sign controller is an integrator:

```python
# src/dormouse/model/experts.py:1867-1869
self.lfb_bias.sub_(
    self.lfb_coef * torch.sign(self.lfb_counts - self.lfb_counts.mean())
)
```

Five steps at `coef=0.5` can cancel exactly. The fixture uses the *global*
RNG: `_run` (`:83-92`) and `DormouseModel(run.model)` (`:127`) never seed, and
`_train_steps` (`:102-109`) uses a fixed `arange` input. Parameter cases before
`[kda]` consume RNG and change its init; the process-start seed also matters.
So it is not strictly order-dependent — it also fails when the file runs
alone, depending on the process seed.

Measured with a fixed seed before pytest (`[M]`):

| seed | result |
|---:|---|
| 0, 2, 6, 8, 10, 11 | green |
| 1, 3, 5, 7 | `[kda]` red — the same `lfb_bias == 0` assert |
| 4, 9 | `[kda+shareu]` red — same assert |

```bash
cd /home/sehaxe/dormouse/port
# deterministic red:
uv run --no-extra gpu python -c "import torch,pytest; torch.manual_seed(1); raise SystemExit(pytest.main(['-q','tests/test_invariants_checkpoint.py::test_every_nonpersistent_buffer_is_carried_or_provable']))"
# deterministic green in the same process shape:
uv run --no-extra gpu python -c "import torch,pytest; torch.manual_seed(0); raise SystemExit(pytest.main(['-q','tests/test_invariants_checkpoint.py::test_every_nonpersistent_buffer_is_carried_or_provable']))"
```

A probe with the `_train_steps` fixture (2×64, `KDAFORM`, seed 1) shows the
counts are all live — 113/178/111/110 at step 0 — and the bias still returns to
`[0,0,0,0]` in the failing run: this is cancellation, not dead routing. `[M]`

### Fix

`tests/test_invariants_checkpoint.py:123` — before `DormouseModel(run.model)`,
`torch.manual_seed(0)` (the smallest deterministic fix; verified green for all
five parameter cases). If the intent is to prove a *nonzero* bias survives
rather than a lucky init, use `moe_lb_mode="prop"` in `_run` for this fixture:
the proportional arm's move is a continuous function of the histogram, so an
exact zero needs exactly uniform counts. Either way, the assertion message
"the fixture must actually move lfb_bias" is the right message; it is the
fixture that is wrong, not the checkpoint.

**Gate:** the parametrized test run twice in one process, plus seeds 0–5; the
test must be deterministic. **Risk:** seeding hides init-dependent controller
behaviour; that is acceptable here because this file tests checkpoint
completeness, and `test_lfb_winds_up_when_the_router_cannot_answer`
(`tests/test_train.py`) is the controller's own gate.

---

## 7. `test_the_fused_combine_counter_is_counted_by_the_kernel` — stale; first run was the 2026-11-08 suite, and it asks for a bf16 counter without a bf16 store

**Вердикт: тест неверен с рождения; до прогона 2026-11-08 он не исполнялся.
Он ждёт `moe_combine_bf16 == 1` при `dtype="fp32"` (дефолт), а этот счётчик
появляется только при bf16-сторе. Вдобавок счётчик не «kernel», а host-side.**

`tests/test_moe_combine_fused.py:402-416`:

```python
mk.combine(y, w, rowmap.reshape(-1), rowmap, 8, counters=c, fused=True)
assert c["moe_combine_fused"] == 1 and c["moe_combine_bf16"] == 1, c
```

`combine`'s signature is `dtype: str = "fp32"` (`moe_kernels.py:1692-1704`),
and `_Combine.forward` only counts the bf16 name when the store is bf16
(`moe_kernels.py:1604-1607`). The first comparison is true, the second
subscript raises `KeyError` — the test cannot fail with a clean assertion. Its
own docstring says "Written and skipped: ... the card is on a multi-day run"
(`:405-407`), and `b342390` confirms it was not run before the 2026-11-08
suite.

CPU reproduction without the card (the launch is replaced by a no-op; the
counter is host-side, so this exercises exactly the counter logic):

```bash
cd /home/sehaxe/dormouse/port && uv run --no-extra gpu python - <<'PY'
import sys, torch; sys.path.insert(0, "src")
from dormouse.instrument import Counters
from dormouse.model import moe_kernels as mk
mk._launch = lambda *a, **k: None
y = torch.randn(16, 32, dtype=torch.bfloat16); w = torch.rand(16)
rowmap = torch.arange(16, dtype=torch.int32).view(8, 2)
tiles = mk._tiles(None, y.dtype, y.shape[1], y.shape[1])
for dt in ("fp32", "bf16"):
    c = Counters()
    mk._Combine.apply(y, w, rowmap.reshape(-1), rowmap, 8, mk.BwdWorkspace(), tiles, c, True, dt)
    print(dt, dict(c))
PY
```

Output: `fp32 {'moe_combine_fused': 1}` then
`KeyError: 'moe_combine_bf16'`; `bf16 {'moe_combine_fused': 1, 'moe_combine_bf16': 1}`.
`[M]`

The docstring's "the kernel counts `moe_combine_fused` after it LAUNCHES" is
also false: the count is `counters.count(...)` in Python, after `_launch`
returns (`moe_kernels.py:1604`), not a device counter. The *fused* branch is
what is counted, which is the useful signal; the provenance sentence is wrong.
On the CPU path the counter is unreachable by construction: the model's routed
pass never selects `fused=True` without the Triton dispatch, and `mk.combine`
itself asserts `y.is_cuda` (`moe_kernels.py:1728`) before it can count. That is
why the probe stubs the launch — it exercises the counter the test names, not
the kernel the test claims to exercise.

### Fix

`tests/test_moe_combine_fused.py:402-416` — two calls, two assertions:

* default (`dtype="fp32"`): `c["moe_combine_fused"] == 1` and
  `"moe_combine_bf16" not in c`;
* `dtype="bf16"`: both counters are 1.

Rewrite the docstring as "the forward counts the arm after the launch returns".
Do not delete the test: the counter is a real log-reader signal, and the
`fused=True` branch is exactly what it should pin. **Gate:** the test itself;
it is CPU-provable with the probe above, so it can be turned into a CPU gate
instead of a `skip`-on-CPU card gate by stubbing `_launch` in the test (the
combine's CUDA assert is the only thing blocking it). **Risk:** none; the
counter logic is what is being pinned, not the kernel.

---

## 8. Fix plan

Ordered so every gate can be checked on CPU before the card is used.

| # | file:line | minimal change | gate that catches a regression | risk |
|---|---|---|---|---|
| 1a | `src/dormouse/train/optim.py:2071-2085` (working tree) | **already applied, uncommitted**: `_adamw_t += 1` after the non-fused `self.adamw.step()` when any AdamW param has a grad | `test_optimizer_suite.py::test_the_torch_adamw_path_advances_the_bias_correction_counter` (exists, green on CPU) | do not add a second increment inside `_gated_adamw_step`; run `test_muon_launch`, `test_nan_firewall`, `test_opt_region` |
| 1b | `src/dormouse/train/graph.py:521-523` | restore the host `lr` float for AdamW groups only; leave the Muon tensor (it holds `muon_lr*mult`) | CPU wrapper test: `set_lr` + `opt.set_lr` + forced `_fall_back` → each group's LR equals what the eager step would read | one eager step with a tensor LR in Muon is supported (`optim.py:1368-1371`); the next `full.set_lr` restores floats |
| 1c | `tests/test_full_graph.py:526-528, :568` | after 1a/1b, re-run; if a ~1e-7 residual remains, say in the docstring that the bound is capturable-vs-fused rounding and raise to 1e-5 | the test itself | do not raise before 1a/1b — 3e-05 is real |
| 2 | `tests/test_fused_epilogue.py:132-145` | outputs 1e-6, `d_*` 5e-5 (`PARITY_GRTOL`), with the citation | the two mutation gates; the kernel test's own `assert_parity` at 5e-5 | a 1e-5-scale read gradient error would pass at the read gate but is still caught at 5e-5 by `parity(epilogue=True)` |
| 4 | `tests/test_fused_residual.py:126-135` | patch `fr.epilogue_read` (signature `mhc, xl, gain, gain_mlp, eps, counters`), not `fr.pre` | the test itself; measured Δctx 6.09e-05 vs tol 1.32e-05 | none |
| 6 | `tests/test_invariants_checkpoint.py:127` | `torch.manual_seed(0)` before the model; or `moe_lb_mode="prop"` in `_run` | the parametrized test, run twice in one process + seeds 0–5 | seeding hides init-dependent controller behaviour; acceptable for a checkpoint-completeness file |
| 7 | `tests/test_moe_combine_fused.py:402-416` | assert `moe_combine_fused` for fp32; add a `dtype="bf16"` call for both; fix the docstring | the test itself (CPU-provable by stubbing `_launch`) | none |
| 3/5 | `tests/test_fused_kda.py:454`, `tests/test_fused_residual.py:155` | **no code change**; add `assert n_fused > 0`, `max(n_fused, 1)`, `torch.cuda.synchronize()`, and print both counts | the card run in both orders (§3, §5) | the guard does not fix an unknown cause; it makes the failure legible |

The plan's line for #1 must change after the fix: it is not "the reference log
never says capture failed", it is "the fallback must be the same optimizer
program". The other six one-line descriptions are not classifications; replace
the table with §0's, which says for each one whether the code or the test is
wrong.

---

## 9. Reproductions (all CPU, all under 2 minutes)

### 9.1 The environment (§0)

```bash
cd /home/sehaxe/dormouse/port
CUDA_VISIBLE_DEVICES="" uv run --no-extra gpu pytest -q -rs \
  tests/test_full_graph.py::test_the_first_replay_is_the_eager_step \
  "tests/test_fused_epilogue.py::test_the_kernel_is_fp32_roundoff[one-gain]" \
  tests/test_fused_kda.py::test_kernel_launch_counter_drops \
  tests/test_fused_residual.py::test_gate_reddens_when_the_pre_bias_moves \
  tests/test_fused_residual.py::test_the_fused_path_issues_far_fewer_launches \
  "tests/test_invariants_checkpoint.py::test_every_nonpersistent_buffer_is_carried_or_provable[kda+shareu]" \
  tests/test_moe_combine_fused.py::test_the_fused_combine_counter_is_counted_by_the_kernel
# -> 6 skipped, 1 red (test_invariants_checkpoint:160) or 6 skipped, 1 passed
```

### 9.2 #6 — the seed table

```bash
cd /home/sehaxe/dormouse/port
for s in 0 1 2 4 9; do
  printf "seed %s: " $s
  uv run --no-extra gpu python -c "import torch,pytest,sys; torch.manual_seed($s); sys.exit(pytest.main(['-q','tests/test_invariants_checkpoint.py::test_every_nonpersistent_buffer_is_carried_or_provable']))" 2>&1 | tail -1
done
# seed 1 -> [kda] red; seed 4/9 -> [kda+shareu] red; seed 0/2 -> green
```

### 9.3 #4 — `fr.pre` is not in the parity path

```bash
cd /home/sehaxe/dormouse/port && uv run --no-extra gpu python - <<'PY'
import sys, types; sys.path.insert(0, "src")
from dormouse.model import fused_residual as fr
run = [c for c in fr.parity.__code__.co_consts if isinstance(c, types.CodeType)]
print({c.co_name: sorted(set(c.co_names) & {"pre", "write", "epilogue_read"}) for c in run})
PY
# {'run': ['epilogue_read', 'write'], '<genexpr>': []}
```

### 9.4 #4 — the nudge that *would* redden

```bash
cd /home/sehaxe/dormouse/port && uv run --no-extra gpu python - <<'PY'
import sys, torch; sys.path.insert(0, "src")
import torch.nn.functional as F
from dormouse.model.mhc import MHCMix
def run(nudge):
    torch.manual_seed(0); m = MHCMix(64, 2, device="cpu")
    with torch.no_grad():
        m.phi_pre.normal_(0, .02); m.phi_post.normal_(0, .02); m.phi_res.normal_(0, .02)
        m.alpha_pre.fill_(.3); m.alpha_post.fill_(.2); m.alpha_res.fill_(1.)
        m.b_pre.normal_(0, .5); m.b_res.normal_(0, 1.5)
        m.b_pre.add_(nudge)
    base = (torch.randn(3, 8, 2, 64) * .5).requires_grad_(True)
    gain = (torch.randn(64) * .1 + 1.).requires_grad_(True)
    ctx = m.pre_mix(base); n = F.rms_norm(ctx, (64,), gain, 1e-3)
    (ctx * torch.randn(3, 8, 64)).sum().add_((n * torch.randn(3, 8, 64)).sum()).backward()
    return ctx.detach()
a, b = run(0.0), run(1e-4)
print(f"dctx={(a-b).abs().max():.3e}  tol={1e-5*a.abs().max():.3e}")
PY
# dctx=6.092e-05  tol=1.324e-05
```

### 9.5 #7 — the counter, without the card

```bash
cd /home/sehaxe/dormouse/port && uv run --no-extra gpu python - <<'PY'
import sys, torch; sys.path.insert(0, "src")
from dormouse.instrument import Counters
from dormouse.model import moe_kernels as mk
mk._launch = lambda *a, **k: None
y = torch.randn(16, 32, dtype=torch.bfloat16); w = torch.rand(16)
rowmap = torch.arange(16, dtype=torch.int32).view(8, 2)
tiles = mk._tiles(None, y.dtype, y.shape[1], y.shape[1])
for dt in ("fp32", "bf16"):
    c = Counters()
    mk._Combine.apply(y, w, rowmap.reshape(-1), rowmap, 8, mk.BwdWorkspace(), tiles, c, True, dt)
    print(dt, dict(c))
PY
# fp32 {'moe_combine_fused': 1} ; bf16 {'moe_combine_fused': 1, 'moe_combine_bf16': 1}
```

### 9.6 #1 — the `_adamw_t` fork

```bash
cd /home/sehaxe/dormouse/port && CUDA_VISIBLE_DEVICES="" uv run --no-extra gpu python - <<'PY'
import sys, copy, torch; sys.path.insert(0, "tests"); sys.path.insert(0, "src")
from conftest import tiny_cfg
from dormouse.model import DormouseModel
from dormouse.train.optim import MixOptimizer
torch.manual_seed(1); ids = torch.randint(0, 256, (2, 32)); tgt = (ids + 1) % 256
torch.manual_seed(0); m = DormouseModel(tiny_cfg(), device="cpu")
o = MixOptimizer(m, lr=3e-4, wd=0.01, opt="adamw")
for _ in range(5):
    o.zero_grad(set_to_none=True); _, loss, _ = m(ids, targets=tgt); loss.backward(); o.step()
sm, so = copy.deepcopy(m.state_dict()), copy.deepcopy(o.state_dict())
def variant(sync):
    torch.manual_seed(0); m = DormouseModel(tiny_cfg(), device="cpu")
    o = MixOptimizer(m, lr=3e-4, wd=0.01, opt="adamw")
    m.load_state_dict(sm); o.load_state_dict(copy.deepcopy(so))
    for _ in range(5):
        o.zero_grad(set_to_none=True); _, loss, _ = m(ids, targets=tgt); loss.backward()
        o._gated_adamw_step(torch.tensor(1.0))
        if sync: o._adamw_t += 1          # the proposed fix in _gated_adamw_step
    o.zero_grad(set_to_none=True); _, loss, _ = m(ids, targets=tgt); loss.backward(); o.step()
    return m.embed.weight.detach()
print("stale vs synced:", (variant(False) - variant(True)).abs().max().item())
PY
# stale vs synced: 4.506111145019531e-05
```

### 9.7 #2 — the CPU mirror of the read chain

```bash
cd /home/sehaxe/dormouse/port && uv run --no-extra gpu python - <<'PY'
import sys; sys.path.insert(0, "src")
from dormouse.model import fused_residual as fr
for b, sep in ((2, False), (3, False), (3, True)):
    rep = fr.parity_read(b=b, t=8, d=64, separate=sep, seed=0)
    d, m = rep["d_b_pre"]; print(f"b={b} sep={sep}: d_b_pre d={d:.3e} ref={m:.3e} rel={d/m:.2e}")
PY
# b=2 sep=False: d=4.768e-07 ref=1.047 rel=4.56e-07
# b=3 sep=False: d=4.768e-07 ref=8.840 rel=5.39e-08
# b=3 sep=True:  d=4.292e-06 ref=5.039 rel=8.52e-07
```
