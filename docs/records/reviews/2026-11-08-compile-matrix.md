# `torch_compile` at the live shape — 2026-11-08 (port lane, GPU)

**Question.** `torch_compile` is a config flag whose only implementation is
`torch.compile(model, dynamic=True)` (`port/src/dormouse/train/loop.py:1775`,
flag at `port/src/dormouse/config.py:1991`). The kernel audit recorded that
wrapper at **0.49×** (repo-root `docs/records/reviews/2026-11-08-kernel-audit.md` §C4) while the
code comments quote 1.20–1.31× from a b1-shape spike. What does the flag
actually buy at the recipe the trainer runs — and is a static graph better?

Line numbers are the worktree at 2026-11-08 ~17:00; `loop.py` is clean at
`HEAD = 3072c2a`, `config.py` carries one uncommitted line from another lane.

**Verdict.**

1. **`dynamic=True` as shipped does not reproduce 0.49× end to end.** 158.3 vs
   150.8 ms/step = **0.95×** (0.99× on the median of the last 5 windows): a
   wash-to-slight-loss. The 0.49× is a property of the `bench_compile.py`
   harness arm, not of the trainer's program (§4).
2. **A static graph is the win.** `dynamic=False` = 131.3 ms = **1.149×**;
   `dynamic=False, mode="max-autotune"` = 118.3 ms = **1.275×**, with 0.42 GiB
   *less* allocated peak.
3. **Warmup eats the win in a short run.** Extra-over-eager warmup is 28.6 s
   (static) and 78.1 s (autotune); at 19.5 / 32.5 ms saved per step that is
   **~1 470 / ~2 400 steps to break even**. At 150 steps both arms are slower
   wall-clock than eager; on the 232k-step budget run the warmup is ~0.3 % of
   the run. The flag should stay OFF by default and be turned on for long runs.
4. **Recommendation:** keep the flag, change its implementation to
   `torch.compile(model, dynamic=False, mode="max-autotune")` — minimal diff in
   §6, not applied. The 1.15× gate was set for (c): (c) lands *at* it (1.149,
   a tie) and (d) clears it (1.275), so the proposal is the (d) line.

---

## 1. Protocol

- Shape **b8 × s1024, grad_accum=1** (8 192 B/step; this is a byte-level model,
  1 byte = 1 token, so `tokens_per_s` is bytes/s). 150 optimizer steps, seed 1,
  `opt=mix`, `muon_lr_mult=12`, `lr=3e-4`, `mix_plan=configs/mix_refetch.toml`,
  `doc_masking=false`, `eval=false`, `log_every=10` (15 windows), checkpoints
  off. FAST rung: `ns_dtype=bf16`, `ns_steps=3`, `muon_state_dtype=bf16`.
- One fresh process per arm, each under `bash scripts/gpu_lock.sh` — the card
  was exclusively held for the arm's whole run. The box was shared with a
  swarm: 8–11 agents queued on the lock, CPU load 16–43 on 20 threads, so
  queueing was minutes-to-tens-of-minutes per arm, but no arm shared the GPU.
- Metric: the **last JSONL `tokens_per_s`** (the step-150 window) and
  `ms/step = 8192 × 1000 / tokens_per_s`; the median of the last 5 windows and
  their coefficient of variation are reported with it.
- Wall/warmup/VRAM come from a sidecar written by the driver (below):
  per-window `StepTimer.window_s` timestamps, process wall clock,
  `torch.cuda.max_memory_allocated/reserved`.

**How the static arms were produced (no repo file edited).** A driver
(`/tmp/opencode/compile_matrix/driver.py`) replaces `torch.compile` on the
`torch` module *before* importing `dormouse.cli`, then runs the normal
`dormouse.cli train` command line. `loop.py:1775` calls `torch.compile(...)` by
attribute lookup at call time, so the trainer's own call is what gets patched.
The sidecar records the kwargs the trainer actually passed:

| arm | kwargs seen by torch.compile |
|---|---|
| (b) | `{'dynamic': 'True'}` |
| (c) | `{'dynamic': 'False'}` |
| (d) | `{'dynamic': 'False', 'mode': 'max-autotune'}` |

`TORCHDYNAMO_DYNAMIC_SHAPES=0` was **not** used: an explicit `dynamic=True`
argument overrides that environment default, so it would not have produced a
static arm. The wrapper is applied after the resume load, unchanged.

Warmup is the first (10-step) window's wall time minus 10 × the arm's steady
median step, i.e. the extra time to the first useful step; it includes
inductor tracing + autotune but not model/data init (that is in `process_s`).
The inductor cache (`/tmp/torchinductor_sehaxe`, 2.2 GB) was **warm** for all
four arms — a cold cache pays more, and is not measured here.

## 2. Results

| # | arm | last `tokens_per_s` | ms/step | × vs (a) | med-5 ms/step | compile warmup | peak alloc GiB | peak reserved GiB | window CV |
|---|---|---|---|---|---|---|---|---|---|
| a | `torch_compile=false` (baseline) | 54 323.9 | **150.8** | 1.000 | 150.8 | 2.3 s | 4.40 | 4.90 | 2.4 % |
| b | `dynamic=True` (as shipped) | 51 742.2 | **158.3** | **0.952** | 152.7 | 49.3 s | 4.39 | 5.06 | 2.5 % |
| c | `dynamic=False` (static) | 62 400.8 | **131.3** | **1.149** | 131.2 | 30.9 s | 4.39 | 4.92 | 0.06 % |
| d | `dynamic=False, mode="max-autotune"` | 69 240.5 | **118.3** | **1.275** | 118.4 | 80.4 s | **3.98** | 5.13 | 0.05 % |

Process wall clock (includes imports + data plan + run): 33.6 / 82.7 / 57.0 /
105.0 s. Loss at step 150: 2.3111 / 2.3121 / 2.3162 / 2.3233 — the same
trajectory at bf16 round-off, i.e. the compiled graph is the same program.
This is a speed run, not a quality A/B (150 steps, no eval).

### Windows

Per-window ms/step, steps 10…150 (the first window is the warmup window):

```
eager     380.4 152.3 160.1 158.9 141.6 152.8 149.4 147.7 148.0 143.6 153.2 144.7 149.1 154.9 150.8
dynamic  5089.4 161.6 155.8 162.0 158.9 154.8 149.3 157.4 148.0 144.9 148.5 152.7 149.2 156.7 158.3
static   3219.3 131.8 131.0 130.9 131.1 131.1 130.7 130.8 131.3 130.9 131.1 131.1 131.2 131.2 131.3
autotune 8160.1 118.8 117.9 117.9 118.1 118.2 118.1 118.4 118.9 119.2 118.4 118.4 118.3 118.5 118.3
```

**The JSONL windows are stable in every arm, and the compiled arms are
markedly more stable than eager**: CV of the last five windows is 2.4 % (a),
2.5 % (b), 0.06 % (c), 0.05 % (d). Eager's 141.6–160.1 ms band is host-dispatch
jitter under a loaded CPU; the compiled graphs collapse it. The last window is
the registered metric and is not a cherry-pick: every steady window of (c) is
130.7–131.8 ms and every one of (d) is 117.9–119.2 ms, while eager's *best*
window is 141.6 ms.

## 3. What the numbers say

- (b) is a **loss of 5 %** on the last window and a wash on the median. Its
  49 s warmup is paid for nothing at 150 steps.
- (c) is **1.149×** — just under the 1.15× gate on the last window, but robust:
  it beats eager's *best* window by 7.9 % and the independent same-day eager
  baseline (§5) by 22 %.
- (d) is **1.275×** and the only arm with a lower memory peak (3.98 vs
  4.40 GiB allocated) — `max-autotune` includes CUDA graphs of the compiled
  region, which removes allocator churn as well as launch overhead.
- Break-even against eager (warmup extra ÷ ms saved): static 28.6 s /
  19.5 ms ≈ **1 470 steps**; autotune 78.1 s / 32.5 ms ≈ **2 400 steps**. A
  150-step benchmark cannot see the win; the 232k-step budget run will.

## 4. Why (b) is not the audit's 0.49×

The audit's 954.0 ms / 0.49× is the `trainer` arm of `scripts/bench_compile.py`
(`scripts/bench_compile.py:197`, `Arm("trainer", {"dynamic": True})`) — the same
wrapper params, but a different program: a bare fwd+bwd+`opt.step` loop over a
fixed batch, min-of-5 after `torch._dynamo.reset()`, with no data loader, no
telemetry and no drain of the window counters. That arm's own log line
(`~/logs/compile_f3.log`) reads `opt.step 358.0 ms` where the fused route in the
same process measured 172.6 ms — the whole arm ran at ~2× the cost of an
uncompiled stage it does not compile. The audit's C4 already discounted a
related arm for "jitter of an unrelated stage" (30.8 of 56.9 ms).

The trainer's own program is what the flag turns on, and it measures 0.95×.
Both numbers are real measurements; only one of them is the trainer. Gate the
flag on the trainer's number.

## 5. What could skew this

1. **Time drift.** (a) ran at 14:10, the compiled arms at 15:40–16:26. The
   independent same-day eager run on the identical shape/rung
   (`/tmp/opencode/accum/logs/speed_a1.jsonl`, 14:26–14:55, another lane)
   reads **160.6 ms/step** (median of its 6 steady windows, range 152.6–165.1;
   its first and eval-containing last windows excluded) — if anything the later
   eager baseline is slower than mine, so the compile ratios are *understated*.
2. **CPU contention.** The swarm's CPU load (16–43) inflates eager's dispatch
   jitter; the compiled arms are immune to it. On a quiet box eager would sit
   nearer its 141.6 ms best window, and the honest lower bounds become 1.08×
   (static) and 1.20× (autotune). Still wins, smaller.
3. **One run per arm**, 15 windows; no repeat of (b)/(c)/(d). The reported CVs
   bound within-run noise, not run-to-run variance.
4. **Warm inductor cache** for every arm; a cold cache would add to all three
   compile warmups (not to any steady-state number).
5. **`max-autotune` includes CUDA graphs**, which can change numerics slightly;
   the 150-step loss traces are within 0.012 of eager, and the repo already
   refuses `torch_compile` with `opt_graph`/`full_graph` (`config.py:5219`), so
   the two graph paths cannot be combined by accident.
6. **`tokens_per_s` here is bytes/s** (1 byte = 1 token); `ms/step` divides
   8 192 by it. Comparing against `throughput_ladder.sh` numbers is valid only
   at the same g=1 shape.

## 6. Change proposal (NOT applied)

`port/src/dormouse/train/loop.py`, lines 1768–1775:

```diff
--- a/port/src/dormouse/train/loop.py
+++ b/port/src/dormouse/train/loop.py
@@ -1768,8 +1768,10 @@
-        # torch.compile(dynamic=True) — the spike's measured winner at the b1
-        # shape (273.5 vs 357.3 ms/step, 1.31x; its dynamic=False arm read
-        # 298.0, 1.20x). Wrapped AFTER the resume
+        # torch.compile(dynamic=False, mode="max-autotune") — measured 118.3
+        # vs 150.8 ms/step (1.27x) at b8 x s1024 FAST rung, against 0.95x for
+        # dynamic=True (docs/records/reviews/2026-11-08-compile-matrix.md). Static is
+        # safe: one seq_len per run, and eval scores the same batch x seq_len
+        # shape. Wrapped AFTER the resume
         # load: an OptimizedModule's state_dict carries a `_orig_mod.` prefix,
         # so loading into the wrapped model would rename every key. The
         # inductor cache persists in ~/.cache/torch — the warmup is once per
         # machine, not per run.
-        model = torch.compile(model, dynamic=True)
+        model = torch.compile(model, dynamic=False, mode="max-autotune")
```

Notes for the implementer, not part of the diff:

- `config.py:1980-1991`'s comment is now doubly stale: it calls the b1 spike
  1.31× while `loop.py`'s comment says 1.20×, and it justifies `dynamic=True`
  with "eval/resume lengths vary" — `evaluate()` scores
  `batch × seq_len` windows (`train/loop.py:1307`), the same shape as training.
- A resume or eval at a *different* `seq_len` costs one recompile (one warmup);
  nothing in the checkpoint format requires `dynamic=True`.
- A run shorter than ~2 400 steps at this shape should leave the flag off: the
  warmup costs more than the steady state saves.

## 7. Artifacts

- `/tmp/opencode/compile_matrix/{eager,dynamic,static,autotune}.jsonl` — the
  150-step JSONL telemetry, one row per 10 steps.
- `/tmp/opencode/compile_matrix/{eager,dynamic,static,autotune}.json` — sidecar:
  kwargs, per-window wall times, process wall, peak VRAM.
- `/tmp/opencode/compile_matrix/logs/*.log` — trainer stdout (`timers=true`
  prints the per-window ms/step), plus the `gpu_lock` idle line that proves the
  arm started on an idle card.
- `/tmp/opencode/compile_matrix/{driver.py,run2.sh,analyze.py}` — the harness.
- Commands (per arm): `DORM_ARM=<arm> DORM_SIDE=... GPU_TIMEOUT=2400
  bash scripts/gpu_lock.sh .venv/bin/python -u <driver> train
  configs/flagship3_dense.toml --data ../data_local/best/fwedu_train.bin
  --eval ../data_local/best/fwedu_eval.bin --set steps=150 --set batch=8
  --set seq_len=1024 --set seed=1 --set lr=3e-4 --set opt=mix
  --set muon_lr_mult=12 --set mix_plan=configs/mix_refetch.toml
  --set doc_masking=false --set eval=false --set log_every=10 --set timers=true
  --set ckpt_every=1000000 --set ns_dtype=bf16 --set ns_steps=3
  --set muon_state_dtype=bf16 --set torch_compile=true` (not passed for arm (a)).
- Environment: RTX 5070 Ti 16 303 MiB, driver 615.78.08, torch 2.13.0+cu129,
  CUDA 12.9, Xeon E5-2640 v4 (20 threads), kernel 7.3.1, load 16–43.
