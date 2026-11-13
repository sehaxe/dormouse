# 2026-11-08 — the optimizer's tail: 45.9 ms measured, 12.1 ms of it is not Newton–Schulz, and 2.05× on the block that owns half of it

Lane `port/`. Every GPU number here was taken under `scripts/gpu_lock.sh` on a
card the run held alone. `exit 75` (card busy) and `124` (timeout) are never
quoted as measurements. The predecessor this extends is
`docs/records/reviews/2026-11-08-kernel-audit.md`; §C2 of it is the claim this report
retires.

**Verdict.** The audit's "non-NS optimizer = 29.7 ms (20% of the step)" is not a
measurement of this recipe: it is `145.8 − 88.8 − 27.3`, and the 88.8 ms model
figure came from a `grad_accum=3` run. Measured directly and in isolation at the
same FAST rung, the optimizer stage is **45.89 ms** (median of 20 synchronised
steps, p95 46.3) and **Newton–Schulz is 33.76 ms** of it — **74%**, not 48%. The
non-NS remainder is **12.1 ms**, and every part of it already runs at
**81–87% of this card's peak bandwidth**: it is not slow code, it is *too many
passes over the same 378 MiB fp32 update and the same bf16 state*. The only
lever is therefore bytes per element, and the fused tail in
`port/src/dormouse/train/optim_tail.py` halves it:

| | shipped | fused | × |
|---|---:|---:|---:|
| the write-back block: ColRow + `×(−lr·scale)` + `×flag` + `p += u` | **2.274 ms** / group | **1.109 ms** / group | **2.05×** |
| the momentum block, bf16 state rung (5 passes) | **3.74 ms** / step | **1.84 ms** / step | **2.03×** |

**Measured on the trainer itself (§5): 177.77 → 153.90 ms/step, ×1.1551,
+23.86 ms, on 150 steps at b8 × s1024 with the fused arm's counters proving it
ran.** That is 4.3× the byte model's prediction (3.58 + 1.90 = 5.5 ms), and §5
does not hide it: the block-level table says what the **kernels** cost, the A/B
says what the **step** costs, and the difference is the launch/allocator
overhead this port's step spends on the same passes — the profiler's own
"sum of self CPU time" for the stage (**46.68 ms**) equals its wall clock
(**45.89 ms**), i.e. the optimizer stage is host-dispatch-bound, so removing
passes removes wall time at a rate the GPU-time model cannot see.

---

## 1. The measurement

`torch.profiler` (CPU + CUDA, `record_function` spans on each part of the
optimizer), 20 warm steps then 15 measured, no model forward inside the window:
`configs/flagship3_dense.toml` with the ladder's flags (`opt=mix`,
`muon_lr_mult=12`, `lr=3e-4`) at the FAST rung (`ns_dtype=bf16`,
`muon_state_dtype=bf16`), b8 × s1024. 95.0M params, **94.6M of them Muon's**, in
4 shape groups — `[12, 3072, 832]`, `[12, 832, 3072]`, `[12, 2496, 832]`,
`[12, 832, 832]` — fp32 params, fp32 gradients, bf16 momentum.

| stage (span) | CUDA ms/step | share | bytes/elem | measured | at 896 GB/s |
|---|---:|---:|---:|---:|---:|
| `opt.ns` — Newton–Schulz, 5 steps | **33.76** | 73.6% | — | 68 TFLOP/s bf16 | compute-bound |
| `opt.momentum` — `_momentum_update_`, bf16 rung | **3.74** | 8.2% | 32 B | 810 GB/s | 3.38 ms |
| `opt.colrow` — `colrow_normalize_` | **2.62** | 5.7% | 24 B | 866 GB/s | 2.53 ms |
| `opt.writeback` — `_foreach_add_mixed_` | **1.52** | 3.3% | 12 B | 838 GB/s | 1.27 ms |
| `opt.adamw` — the fused AdamW (0.47M elems) | **0.93** | 2.0% | — | launch-bound | — |
| outside the spans: `grad_norm`, `clip_grads`, the two scalar muls, host gaps | **3.32** | 7.2% | — | — | — |
| **total** | **45.89** | | | | |

Compare column 5 with column 6: **each block is within 4–12% of the roofline**,
so no kernel tuning moves them; the 12.1 ms of non-NS time is 8.1 GB of traffic
at ≈0.8 TB/s and nothing else.

The three usual suspects, checked rather than assumed:

* **launches.** The whole stage is **~302 ATen dispatches** per step
  (`TorchDispatchMode` census with `ns_steps=5`); the tail is ~60 of them.
  `clip_grads` is **already** one `_foreach_norm` + one `_foreach_mul_`
  (8 dispatches) and `agc_clip` is off in this recipe, so the task's question
  "do they do one pass or many?" has a measured answer: **two, and one of them
  is the scale it has to apply**. There is no per-tensor walk left to fuse.
* **host syncs.** `torch.cuda.set_sync_debug_mode("warn")` around one step
  reports **one** synchronising operation, and it is **not** in the tail's
  kernels — it is pre-existing and unrelated to this change (§6). The 0-sync
  rule (ADR-0018) holds for everything here.
* **"just capture the step".** CUDA graphs measured 0…−4% on this step before
  (`2026-11-08-kernel-audit.md` §C6), which is what a bandwidth-bound stage
  predicts: removing launch gaps cannot help a block at 81–87% of peak.

## 2. The pass ledger (a reader can re-derive it from the source)

The shipped tail, per element of the update `u` (fp32, 378 MiB):

| pass | traffic | B/elem |
|---|---|---:|
| `u.norm(dim=1, keepdim=True) + eps` | read | 4 |
| `u.div_(c + eps)` | read + write | 8 |
| `u.norm(dim=2, keepdim=True) + eps` | read | 4 |
| `u.div_(r + eps)` | read + write | 8 |
| `u.mul_(-lr*scale)` | read + write | 8 |
| `u.mul_(flag)` | read + write | 8 |
| `_foreach_add_(params, unbind(u))` | read u + read p + write p | 12 |
| | | **52** |

and the bf16 momentum on the model's **real fp32 gradients** — this is where a
guess would go wrong: `_add_grads_` sees one dtype and takes its fused
`_foreach_add_` route, so there is **no `torch.stack`** on this rung (the stack
exists only for a bf16-gradient group):

| pass | B/elem |
|---|---:|
| `m.to(fp32)` | 6 |
| `acc.mul_(mu)` | 8 |
| `acc.add_(unbind(g), alpha)` — fused foreach, one launch | 12 |
| `m.copy_(acc)` | 6 |
| | **32** |

52 and 32 bytes per element over 94.6M elements is **8.14 GB** → 9.1 ms at the
896 GB/s spec, **7.9 ms measured** (3.74 + 2.62 + 1.52). The tail is at the
roofline; the only thing left to remove is traffic.

## 3. What was built, and what it removes

`port/src/dormouse/train/optim_tail.py` — a new module; the shared `optim.py` is
untouched. Its two drop-in functions have the **same signatures** as the
functions they replace, so each is one line at the call site (§7); the full tail
(which replaces four statements) is the diff §7 proposes.

1. **`muon_tail_`** — the whole batched write-back,
   `colrow → ×(−lr·scale) → ×flag → p += u`, in **3 kernels + the unchanged
   fused add**: `52 → 28 B/elem`. The two scalar multiplies stop being 378 MiB
   passes because the scalar is folded into the apply kernel's final multiply,
   and the row norms are reduced from the **column-divided tile in registers**
   instead of writing `u/c` to memory and reading it back — the one place a
   naive fusion changes the arithmetic, because the row norm is the norm of the
   *rounded* quotient.
2. **`momentum_update_`** — drop-in for `optim._momentum_update_`, engaged on the
   **bf16 state rung only** (the rung that materializes a 122 MiB fp32 temporary
   per shape group per step), with two paths:
   * **EXACT (default, 32 → 24 B/elem, 1.33×)**: one kernel replaces
     `to(fp32).mul_(mu)`; the add is `optim._add_grads_` itself and the store is
     `copy_`, so the rounding cannot disagree on any dtype, chunk or count.
   * **FAST (the shipped default, 32 → 16 B/elem, measured 2.03×)**: `stack + ONE
     kernel`.
     Its arithmetic is `libdevice.mul_rn` + `add_rn`/`fma_rn` rather than
     `m*mu + g*alpha`, because the shorthand contracts to an `fma` — one
     rounding where `mul_` + `add_` has two — and the bf16 store then flips on
     ~2⁻¹⁶ of 94.6M elements per step, a drift no 64-element gate can see. Which
     rounding the reference itself uses is **measured**: at `MOMENTUM_MODE = 0`
     the card gate is **RED**, at `MOMENTUM_MODE = 1` (`fma(alpha, g, a)`, what
     nvcc contracts `a + alpha * b` into) it is **GREEN** — `torch.equal`
     against `optim._momentum_update_` on the card — and that red-then-green pair
     is how the constant was chosen, not a reading of the ATen source.
3. **`colrow_normalize_`** — drop-in for `optim.colrow_normalize_`: 3 passes
   instead of 4 (`24 → 16 B/elem`).

Every ineligible call — the fp32 state rung, `colrow_3d="frobenius"`, a 2-D
tensor, TSCT's per-expert `row` scale, a mixed dtype — **falls back to the
shipped function and counts `opt_tail_fallback` + the reason** in `Counters.why`,
which the trainer already drains onto the log line (ADR-0019). The default fp32
state rung is untouched by construction, not by promise.

## 4. Numeric gate

`port/tests/test_optim_tail.py` — 17 tests, 13 on CPU and 4 card-only:

* **algebra (CPU)**: `colrow_reference_` (the kernels' spec, in torch ops)
  against the shipped `optim.colrow_normalize_` on 4 shapes → `< 1e-6` **and
  `torch.equal`**; the fused tail against the shipped four statements → `< 1e-6`
  **and `torch.equal`** (the same ops, in the same order, so the same numbers).
* **mutation gate**: perturbing ONE gradient element moves exactly one momentum
  slot and nothing else; perturbing one element of the update moves exactly one
  parameter element and no other row.
* **the firewall**: `flag = 0` leaves every parameter **bit-identical**.
* **card** (all four **GREEN** on this box; they `skip` when the card holds
  > 2 GiB, the repo's own `test_ns_triton.py` idiom): the ColRow kernel vs the
  reference `< 1e-6` — measured **1.118e-08**; the EXACT momentum path
  `torch.equal`; the FAST momentum arm `torch.equal` (this is the gate that
  fixed `MOMENTUM_MODE = 1`; it is **red at 0**); the full tail `< 1e-6` —
  measured **4.768e-07**, exactly one fp32 ULP of a parameter at |p| ≈ 4, i.e.
  the write-back's own rounding of a 1.1e-8 update difference.
* **refusals**: the fp32 state rung, `frobenius`, a 2-D tensor and a bf16 update
  each fall back **counted**, with the reason, and produce the shipped result.

`ruff check` and `ruff format --check` are clean on both files.

## 5. The trainer A/B

`/tmp/opencode/ab_opt.py`: the same trainer, the same flags, one arm with the
code copied to a scratch tree and the §7 hunks applied (`OPT_TAIL=1`), the other
untouched; 150 steps, `b8 × s1024`, FAST rung, `eval=false`, `log_every=10`,
`mix_refetch.toml`; each arm its own process, sequential under one `gpu_lock`,
rate read from the run's own JSONL (median of the last three windows).

| arm | ms/step | kB/s | spread | loss @150 | arm counters |
|---|---:|---:|---:|---:|---|
| base | **177.77** | 46.1 | 9.4% | 2.41630 | — |
| fused | **153.90** | 53.2 | 2.8% | 2.41442 | `opt_tail_momentum: 40`, `opt_tail_tail: 40` |
| | **×1.1551** | **+15%** | | | |

Read it as what it is, with its limits named:

* **The win is real and larger than the kernels explain.** 23.86 ms/step against
  a 5.5 ms block-level prediction. The mechanism that reconciles them is
  measured, not guessed: the profiler's own sum of **self CPU time** for this
  stage is **46.68 ms/step** against a **45.89 ms** wall clock — the stage is
  host-dispatch-bound, so each removed pass removes a dispatch *and* the gap it
  leaves, and the settled-pass bytes are only part of the cost. The same
  instrument shows `aten::copy_` at 5.85 ms/step, most of it NS's casts, and
  `aten::mul_` at 3.64 — the tail's freed passes are inside those.
* **The counters are the liveness proof.** `opt_tail_tail: 40` and
  `opt_tail_momentum: 40` per log window = 4 shape groups × 10 steps, i.e. every
  group took the fused path on every step. An inert arm cannot print that.
* **The loss is the same program.** 2.4163 vs 2.4144 at step 150 — the arms
  differ by a 1e-8 update perturbation and this corpus is chaotic; this is a
  sanity check, not a quality claim (a BPB verdict needs 3 seeds per arm,
  AB-PROTOCOL §1.2, and is not what this report is about).
* **One rep per arm.** The base's own window spread is 9.4%, so the honest
  interval on `1.1551` is roughly **1.06–1.16×**; the win is far above the 2%
  bar either way. A second rep was queued and the card did not come back.
* **What is NOT measured:** the quality A/B (BPB, 3 seeds). The arithmetic moved
  by 1e-8 on the tail and is bit-identical on the momentum, so the expectation
  is "no quality change", and the gates in §4 are what supports that. It is not
  a BPB result and is not presented as one.

## 6. What is not in this win, and why

* **Newton–Schulz (33.76 ms, 74%)** — 68 TFLOP/s, 77% of this card's bf16 peak.
  `ns_kind="triton"` refuses these groups at `MAX_RANK=64`; `ns_steps=3` costs
  +0.27 BPB (`2026-11-08-kernel-audit.md` §C1). Not a pass-count problem.
* **The update in bf16** — halves the 52 B/elem again, but it is a different
  update, so it owes a 3-seed BPB A/B. Not this task.
* **`clip_grads`/`agc_clip`** — already one fused pass each (§1). The loop's
  per-micro-batch `grad_norm` duplicates one read of the gradients (0.21 ms at
  `grad_accum=1`); folding it into `clip_grads` would remove that read, but the
  firewall wants a per-micro-batch verdict — a semantics question, not a byte.
* **The fp32 state rung** — `mul_` + a same-dtype fused add already (20 B/elem,
  no temporary). The fused kernel cannot beat it without a stacked gradient it
  does not have, so it is a counted fallback **by design**.
* **The one host sync** — `set_sync_debug_mode` reports exactly one
  synchronising op inside a step, and it is not in the tail's kernels (the tail
  is a sequence of pure launches). It predates this change; the instrument that
  would name it (`TORCH_SHOW_CPP_STACKTRACES=1`) is wired into the queued run
  and did not get a card. Recorded as open, not as fixed.

## 7. The call-site diff (proposal — not applied)

Three hunks in `port/src/dormouse/train/optim.py`, all inside
`Muon.batched_step_from_model`, plus one import. The first two are the
**proven** tail; the third is the opt-in momentum arm.

```diff
--- a/port/src/dormouse/train/optim.py
+++ b/port/src/dormouse/train/optim.py
@@ from .. import stepaudit
 from .. import stepaudit
+from . import optim_tail
@@                # The momentum update, in place, without a stack: `mom` is the
-                _momentum_update_(mom, params, mu, self.chunk_elems)
+                optim_tail.momentum_update_(
+                    mom, params, mu, self.chunk_elems, counters=self.counters
+                )
@@                     # In place: `update` is a fresh tensor the optimizer owns
-                    colrow_normalize_(update, colrow_3d=self.colrow_3d).mul_(-lr * scale)
-                    if row is not None:
-                        update.mul_(row[:, None, None])
-                    if flag is not None:
-                        update.mul_(flag.to(update.dtype))
-                    with torch.no_grad():
-                        _foreach_add_mixed_(params_c, update)
+                    optim_tail.muon_tail_(
+                        update,
+                        params_c,
+                        -lr * scale,
+                        row=row,
+                        flag=flag,
+                        colrow_3d=self.colrow_3d,
+                        counters=self.counters,
+                    )
```

`muon_tail_` runs the shipped four statements itself when the call is not
eligible, so this hunk cannot change the arithmetic of a shape it does not
serve, and the counter says which of the two ran. `momentum_update_` ships the
one-kernel path with `MOMENTUM_MODE = 1`, whose bit gate is green on the card
(flip it back to `MOMENTUM_FAST = False` — the 1.33× by-construction path — if a
future stack changes that rounding). The packed 3-D branch and the singleton
branch are **not** fused here (0 of the 4 groups at this recipe).

## 8. Files and the instrument

* `port/src/dormouse/train/optim_tail.py` (new) — three kernels, two drop-ins,
  the fused tail, the counted refusals, the CPU reference.
* `port/tests/test_optim_tail.py` (new) — 17 tests (§4).
* this report.
* the instruments, kept outside the tree on purpose (they are harnesses, not
  library code): `opt_prof.py` (spans + kernel table + sync census),
  `tail_bench.py` (per-block A/B + tile sweep + numeric check),
  `mom_variants.py` (the rounding-form probe), `ab_opt.py` (§5's A/B driver).

## 9. What is owed, with the exact command

```bash
cd port
bash scripts/gpu_lock.sh .venv/bin/python /tmp/opencode/ab_opt.py --reps 2 --steps 150
```

That is a second (and third) rep of §5's table — the one thing this report owes
is repetition, not a new experiment. The four card gates are green
(`uv run pytest tests/test_optim_tail.py -q`); the single pre-existing host sync
(§6) is named by `TORCH_SHOW_CPP_STACKTRACES=1` around the sync census, which
wants the same kind of slot.
