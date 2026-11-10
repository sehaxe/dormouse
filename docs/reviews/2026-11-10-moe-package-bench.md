# 2026-11-10 — the MoE package's bill: a 299M-total / 100M-active bank does not fit the control's batch

Lane `port/`, plan `.bulba/plan.md` §T4. One question, numbers only: does a
299M-total / 100M-active routed bank (`ffn_form="bank"`, `n_experts=8`,
`moe_topk=2`, `moe_capacity=2.0`) cost the same step time and VRAM as the 95M
dense control, or does it force the batch down?

**Verdict (2026-11-10, after §7 — READ THAT FIRST). The original verdict below
measured OUR layout, not MoE's physics: the padded bank ran 2× the rows the
router assigned and materialised two `[·, f]` buffers per layer just to scale
rows (624 MiB/layer at b24). Both are fixed in `SparseMoE._bank_dispatch` with
identical math (fp32 2.4e-7 relative, bf16 4.7e-3 relative and 0.92× the
reference loop's OWN fp64 error), the bench defaults to the dropless arm, and the
`b24 does not fit` claim is WITHDRAWN pending the re-measure §7.4 — which could
not run because another lane's 3-seed A/B held `/tmp/dormouse-gpu.lock` for the
whole window.**

The numbers below stand as the measurement of the PRE-FIX arm (padded,
`capacity=2.0`, fp32 router-weight product): at 299.15M total / 99.67M active it
OOMs at b24 and b16 — the batches the control runs at 10.80 GiB and 7.53 GiB —
and its largest working batch was **12**, where it paid **372.6 ms/step against
148.4 ms for the same-block dense control (2.51×) and 12.60 GiB against 6.15
(2.05×)** for the same 12,288 bytes.

Marks: **[M]** measured here, command named · **[C]** arithmetic over [M] inputs ·
**[S]** inference with the check that settles it.

---

## 1. What was run

```bash
# from port/ — the script re-execs itself through scripts/gpu_lock.sh
uv run --python .venv/bin/python python scripts/bench_moe_package.py \
    --batch-sizes 24,16 --steps 20 --warm 3
uv run --python .venv/bin/python python scripts/bench_moe_package.py \
    --arms bank,control-kda --batch-sizes 12,11,10 --steps 20 --warm 3
uv run --python .venv/bin/python python scripts/bench_moe_package.py \
    --arms bank --batch-sizes 15,14 --steps 20 --warm 3      # b15 OOM, b14 OOM in inductor
uv run --python .venv/bin/python python scripts/bench_moe_package.py \
    --arms bank,control-kda --batch-sizes 13 --steps 20 --warm 3
uv run --python .venv/bin/python python scripts/bench_moe_package.py --check
uv run --python .venv/bin/python python scripts/bench_moe_package.py --rank-scan
```

Every line took `/tmp/dormouse-gpu.lock` through `scripts/gpu_lock.sh` (flock +
the idle-card check + `timeout`); no 75/124 exit is quoted below. The card is a
**RTX 5070 Ti, 15.56 GiB**, holding **1.21 GiB of a non-dormouse service
(`/usr/bin/hex`)** for every arm — the same 13,845 MiB free at every start, so it
is a constant of the instrument, not an arm.

The recipe is the live one (`scripts/train_budget.sh`), one variable changed:

| ingredient | value |
|---|---|
| preset | `configs/flagship3_dense.toml` (d_model 832, 12 layers, vocab 256) |
| shape | seq 1024, `grad_accum=1`, b as named |
| data | `../data_local/best/fwedu_train.bin`, flat `ByteStream` (real bytes; **not** `mix_refetch` — timing does not read the mixture) |
| schedule | `wsd_lr`, `warmup_tokens=19e6` (the recipe's 1% of the 1.9 GB budget), `lr=3e-4`, `min_lr_frac` from the preset |
| optimizer | `opt=mix`, `muon_lr_mult=12`, `ns_dtype=bf16`, `muon_state_dtype=bf16` |
| compile | `torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")` — the trainer's own wrapper (`train/loop.py`) |
| per-step seam | `invalidate_banks_all` + `invalidate_qat_cache_all` after `opt.step`, exactly `train/loop.py` |
| protocol | 3 untimed warm steps, then 20 timed steps; per-step CUDA events (no per-step host sync); `peak = torch.cuda.max_memory_allocated` |
| seed | 1 |

`--check` (no GPU) rebuilds all three arms on the meta device and asserts the
parameter counts this document quotes: control 95,035,200 · control-kda
95,119,692 · bank 299,152,716 total / 99,672,396 active.

## 2. The table

`ms/step` is the median and the minimum over the 20 timed steps. `B/s` is
`batch·1024 / median`. `steady` is `memory_allocated` after the timed steps —
weights + grads + optimizer state, the floor a batch size barely moves (it shifts
only with the allocator's cache, which is why it reads 2.92-3.25 GiB for the
bank).

| arm | batch | ms/step (median) | ms/step (min) | peak GiB | steady GiB | B/step | B/s | params total/active |
|---|---|---|---|---|---|---|---|---|
| control | 24 | 281.9 | 281.2 | 10.80 | 0.98 | 24,576 | 87,178 | 95.04M / 95.04M |
| control-kda | 24 | 252.3 | 251.0 | 11.15 | 1.05 | 24,576 | 97,410 | 95.12M / 95.12M |
| bank | 24 | **OOM** | — | — | — | 24,576 | — | 299.15M / 99.67M |
| control | 16 | 202.8 | 202.2 | 7.53 | 0.98 | 16,384 | 80,786 | 95.04M / 95.04M |
| control-kda | 16 | 183.5 | 182.0 | 7.76 | 1.02 | 16,384 | 89,293 | 95.12M / 95.12M |
| bank | 16 | **OOM** | — | — | — | 16,384 | — | 299.15M / 99.67M |
| bank | 15 | **OOM** | — | — | — | 15,360 | — | 299.15M / 99.67M |
| bank | 14 | **OOM** (in `torch.compile`) | — | — | — | 14,336 | — | 299.15M / 99.67M |
| bank | 13 | **OOM** | — | — | — | 13,312 | — | 299.15M / 99.67M |
| control-kda | 13 | 157.5 | 156.8 | 6.57 | 1.01 | 13,312 | 84,521 | 95.12M / 95.12M |
| bank | 12 | 372.6 | 372.1 | 12.60 | 2.92 | 12,288 | 32,975 | 299.15M / 99.67M |
| control-kda | 12 | 148.4 | 147.5 | 6.15 | 1.01 | 12,288 | 82,812 | 95.12M / 95.12M |
| bank | 11 | 354.8 | 354.3 | 12.21 | 3.25 | 11,264 | 31,750 | 299.15M / 99.67M |
| control-kda | 11 | 139.8 | 138.9 | 5.71 | 1.00 | 11,264 | 80,568 | 95.12M / 95.12M |
| bank | 10 | 341.1 | 340.2 | 11.45 | 3.23 | 10,240 | 30,019 | 299.15M / 99.67M |
| control-kda | 10 | 129.7 | 129.2 | 5.28 | 1.00 | 10,240 | 78,938 | 95.12M / 95.12M |

The bank's counter line says which arm ran, so a silent fallback cannot pass as a
bank: `{'moe_padded_bmm': 276, 'moe_bank_builds': 276, 'moe_shared_forwards':
3391488, 'depth': 276, 'iter': 276, 'rezero_write': 276, 'forward': 23}` at b12.

## 3. The sizing: `rank`, not `d_ffn` — and what the bank cannot express

The arm is `BANK` in the script (config knobs only, no model code touched):
`block_form="kda"`, `loop_blocks=12`, `max_iter=12`, `n_layers=0`, d_model 832,
`ffn_form="bank"`, `n_experts=8`, `moe_topk=2`, `moe_capacity=2.0`,
**`rank=832`, `d_ffn=1664`**, `shared_rank=192` → **299,152,716 total /
99,672,396 active** [M, `--check`].

The bank's expert is a rank-`rank` dual-latent GLU plus ONE shared `down [f, d]`
that every expert writes through, and a shared expert that is always on. Total
is therefore `9` expert copies per layer against `3` active ones (top-2 + the
shared expert), and the ratio is only `3.0` when the copies dominate the shared
down. Sizing the SAME bank to ~299M total over the latent width:

| `rank` | `d_ffn` landing ~299M | total | active | total/active |
|---|---|---|---|---|
| 64 (the preset's) | 14,976 | 299.0M | 222.3M | 1.34 |
| 128 | 10,816 | 299.0M | 184.0M | 1.63 |
| 256 | 6,656 | 299.0M | 145.7M | 2.05 |
| 384 | 4,544 | 297.5M | 125.8M | 2.36 |
| 512 | 3,328 | 299.1M | 115.0M | 2.60 |
| 640 | 2,496 | 299.1M | 107.3M | 2.79 |
| **832** | **1,664** | 299.2M | **99.7M** | **3.00** |

[M, `--rank-scan` (meta device, CPU, no GPU)]

So "≈299M total, ≈100M active" **is expressible with knobs, but only at
`rank = d_model`**: at the preset's `rank=64` a 299M-total bank is **222.3M
active**, and no `d_ffn` fixes that. `rank=832` means each expert's latent is as
wide as the model — a full-width GLU expert behind a router, not a bottlenecked
one. That is the arm measured here and it should be read as "the current bank,
widened to the only latent width that reaches the package", not as "the bank as
every preset runs it".

## 4. Why the package costs more than its active count

* **Memory is the total, not the active count** [M]. Steady state (weights +
  grads + optimizer) is **2.92 GiB vs 1.01 GiB** at b12 — 2.9×, tracking the
  3.15× parameter ratio. On top of it: at b12 the peak is **12.60 GiB against
  6.15 GiB**, so the activation bill is **0.81 GiB per batch element against
  0.43** (1.9×). The padded arm's bank is `[E, cap, d]` with
  `cap = ceil(capacity·tokens·k/E)`, i.e. **twice the rows the router actually
  assigns** at `capacity=2.0` [C]. b16 then dies with the process already holding
  **~15.3 GiB of the 15.56 GiB card** (the allocator's own numbers: 265,748,480 B
  free when 220,200,960 B were asked for), and b24 at ~15.4 GiB: the package's
  peak there is past the card, exactly as the two factors above predict.
* **Step time is 2.51× at equal bytes** [M]: b12 bank 372.6 ms vs control-kda
  148.4 ms. Part of that is the same over-provisioning (`E·cap = 2·topk·tokens`
  rows through the expert GEMMs) — that part is [C], not decomposed here; part is
  the 9-copy factor traffic per forward. The bank is also slower per BYTE than
  the control at every batch it can run: 33.0 kB/s at b12 against 82.8.
* **The block form is not the explanation** [M]. The bank needs
  `block_form="kda"` (the only block that owns a `SparseMoE`); the control preset
  is `block_form="transformer"`. The `control-kda` arm isolates it: the KDA block
  is **faster** than the transformer block at this shape (b24 252.3 vs 281.9 ms,
  b16 183.5 vs 202.8 ms), so the confound works against the bank's cost, not for
  it.

## 5. What could skew this

* **`moe_capacity=2.0` is a memory knob.** At capacity 1.0 the padded bank's rows
  halve and b24 might pass. Not measured on purpose: it changes which assignments
  the model drops, so it is a different model, not a cheaper one.
* **b14 is OOM inside `torch.compile`'s autotune**, not at step time — recorded as
  OOM, and the harness now absorbs the `InductorError` spelling so one arm cannot
  kill the table. b13, b14, b15, b16 and b24 all fail; b12 fits at 12.60 GiB.
  **The largest batch this instrument measures as working is 12** — and because a
  new batch is a new shape (a fresh inductor compile), a batch the *model* might
  hold can still be refused by the *compiler*'s workspace. "Working" here means
  the whole recipe (compile on, 20 steps) ran.
* **The batch drop is recoverable with `grad_accum`** [S]: two micro-batches of
  12 hold b24's bytes at b12's VRAM. Not measured here (one micro-batch per
  step), so this document prices the bank, not the escape hatch; the escape
  hatch does not remove the 2.51× per-byte penalty.
* **No quality claim**: 20 steps, one seed, no eval. This is a throughput/VRAM
  probe; a BPB comparison between these arms is a different run with 3 seeds.
* **The card carries 1.21 GiB of someone else's service**; it is identical for
  every arm and every restart, so it shifts the absolute ceiling, not the ratio.
* **`peak` is torch's `max_memory_allocated`**, not the card's used memory: the
  two differ by the CUDA context and that service.

## 6. Artifacts

* `port/scripts/bench_moe_package.py` — the harness. `--check` (parameter counts)
  and `--rank-scan` (the §3 table) run on CPU with no GPU and no lock; no other
  file changed.
* Logs of the four runs: `/tmp/opencode/bench_moe_package*.log` (ephemeral).
* No model code, no preset and no checkpoint was modified: the arms are config
  overrides built inside the script.

---

## 7. REVISED 2026-11-10 (owner's challenge): two layout wastes, fixed

The numbers in §2 are the arm AS CONFIGURED AND CODED, and both of its costs were
ours, not the package's:

1. **`moe_capacity=2.0` sized the `[E, cap, ·]` bank to 2× the mean group.** At
   b24 that is `E·cap = 98,304` rows for `R = n·k = 49,152` assignments, so every
   expert computed twice the rows the router assigned — by construction
   (`cap = ceil(capacity·tokens·k/E)`), not by routing skew.
2. **The router weight was applied to the `[R, f]` output in fp32**, so the pass
   materialised a `4·R·f` byte buffer per layer purely to scale rows (312 MiB at
   b24), on top of a bf16 zero-row `cat` of the same size that existed only to
   make the dump slot index-legal. The code's own `ponytail:` note had already
   named the fix.

### 7.1 What changed (code, `SparseMoE._bank_dispatch`)

* The weight rides the BANK layout, before `@ v`: `w_bank [E·cap+1]` (exactly 0
  at the dump slot) → `h * w_bank` in the GEMM's dtype → ONE `bmm(h, v)`. The
  `[R, f]` fp32 product never exists.
* The `[E·cap+1, f]` zero-row `cat` is gone: dumped rows gather a clamped bank row
  and are scattered to token row `n`, which the return slices off. Their value is
  discarded and their gradient is exactly 0 (the slice's backward zeroes that
  row), so no zero row is needed and no wide mask multiply is added.
* The one fp32 buffer that stays is the accumulation target `[n+1, f]` — the
  exact k-way sum requires it.
* COUNTED (ADR-0011): `moe_w_in_bank` once per dispatch, and `moe_bank_rows` =
  `E·cap` — the capacity waste is now a number on the log line.
* Nothing reads the host: `zeros`/`index_put_`/`clamp`/`where`/`index_select`/
  `index_add_` are all shape-stable ops the file already uses, and the
  `keep`-as-mask rule (§2's comment block) is intact — rows are still never
  compressed, so `t[mask]` never enters the graph.

### 7.2 Parity (measured; `tests/test_moe_batched.py`, CPU)

| gate | fixture | measured | bound |
|---|---|---|---|
| bank vs naive, fp32 | 64 rows, d 64, f 128, E 4, k 2, cap 0 & 1.0 | max\|Δ\| **1.431e-06**, rel **2.392e-07** | 1e-5 |
| bank vs naive, bf16 | same | max\|Δ\| 2.843e-02, rel **4.730e-03** | 1e-2 |
| bank vs float64, bf16 | same | **1.992e-02 vs the loop's own 2.159e-02 = 0.92×** | ≤ 1× |
| wide `[R, f]` fp32 `mul` in the dispatch | 48 rows, cap 1.0 | **0 hits**, rel 1.612e-07 | 0 hits |
| `moe_w_in_bank` per dispatch | 32 rows, dropless | **1** | 1 |

Row 3 is the one that answers "is the bf16 delta a defect": a bf16-vs-bf16
comparison cannot separate a layout change from bf16's own rounding, so it
measures each arm against float64. The new bank is 0.81-0.92× the loop's own
error across two fixtures, i.e. the 4.7e-3 is noise the loop pays too.

The bf16 bound moved 2e-3 → 1e-2 in exactly the gates that compare a
pre-GEMM-weight arm with a post-GEMM-weight one (`test_moe_batched`,
`test_moe_sonic` ×2, `test_experts_gpu`, `test_moe_kernel`'s triton-vs-padded;
`triton`'s own bound stays 2e-3 because its weight is still post-GEMM). Two more
pins moved and are named here rather than buried:

* `test_moe_buckets`'s one-bucket **bit-equality** became the same dtype bound,
  and its fixture weight moved off `0.5` — an exact bf16 scaling that would have
  hidden the difference while `bucketed` still carries the old tail.
* `test_tsct_packed`'s OFF digest was re-captured (grad moved; **state and loss
  did not**: no init and no logits bit moved).

### 7.3 The per-layer bill (exact shapes; b24: R 49,152, E·cap 98,304, f 1,664, rank 832, bf16)

| buffer, per layer | before | after |
|---|---|---|
| fp32 router-weight product `[R, f]` | 312 MiB | **0** |
| bf16 zero-row `cat` `[E·cap+1, f]` | 312 MiB | **0** |
| bf16 `h·w` `[E, cap, rank]` | 0 | 156 MiB (kept: `w`'s gradient) |
| fp32 scatter source `[R, f]` | 312 MiB | 312 MiB |
| bank tensors `xg, g, u, h, y` | 936 MiB at cap 12,288 | 936·C/12,288 MiB |
| fp32 accumulate `[n, f]` | 156 MiB | 156 MiB |

**−468 MiB per layer that no batch size moves** (−234 MiB at b12), plus the 2.0×
cap's rows. This is a buffer census, not a peak: only part of twelve layers'
transients is live at once (the measured b12 peak of 12.60 GiB against ~1.6 GiB
of buffers per layer is the proof), which is why the number that settles the
question has to come from the run.

### 7.4 The re-measure: BLOCKED by the guard, not skipped

At 12:13-13:00 on 2026-11-10 another lane's 3-seed A/B held the card:
`scripts/arch_sweep.sh` with `ARMS="gated_silu gated_situ"`, `BATCH=24`,
`BYTES=120MB` (4,882 steps per arm), wrapped in ONE `scripts/gpu_lock.sh` that
holds `/tmp/dormouse-gpu.lock` across all six 23-minute arms (~14:30); the card
sat at 14.1 GiB, PID 897686. The bench was queued behind that lock (relaunched
twice on this side: once for a code edit after the first queue, once because the
other lane's driver had queued ahead) and was still waiting when this was
written; the guard's own `flock -w 1800` would have exited **75 — nothing was
measured** at ~13:14. Nothing was killed and no guard was bypassed.

Ready to run, in this order (each re-execs itself through `gpu_lock.sh`):

```bash
uv run --python .venv/bin/python python scripts/bench_moe_package.py \
    --arms control,bank --batch-sizes 24,16,20 --steps 20 --warm 3
uv run --python .venv/bin/python python scripts/bench_moe_package.py \
    --arms bank --batch-sizes 24 --moe-capacities 1.0,1.25,2.0 --steps 20 --warm 3
```

What it decides: whether the b24 peak now lands under the control's 10.80 GiB
(the question §0's original verdict answered "no" for the pre-fix arm), and what
the capacity rule costs per row (`moe_bank_rows` prints the cap actually run).

### 7.5 Follow-ups this lane did not take

* **`_bucketed_dispatch` still has the old tail** (`[n·k, f]` fp32 product and the
  zero-row `cat`). Same layout, different arm; the OOM was the padded/batched
  arm, so it is out of this fix's scope and its gate now records the difference.
* **`_triton_dispatch`'s non-fused route** applies the weight in fp32 after the
  GEMM through `moe_kernels.combine`; moving it into rank space is a kernel-side
  change with its own parity gate.
* The CUDA halves of the adjusted gates (`test_experts_gpu`,
  `test_moe_kernel`'s triton pair, `test_moe_sonic`'s card gate) were **not
  re-run** — the card was held — so their new bounds are the same measured
  4.7e-3 argument, not a local observation.
* The fp32 scatter source `[R, f]` (312 MiB/layer at b24) is now the largest
  buffer in the pass. A chunked `index_add_` (static row blocks, ~6× fewer bytes)
  would cut it further; it is not taken here because the two named wastes were
  the fix and the peak is unmeasured, so there is no evidence it is needed.

---

## 8. The reimplementation (2026-11-10, same day): the other two kernel arms

§7.5's follow-ups were taken as the task, with one constraint from the owner that
was respected rather than improved: **per-step shapes are frozen under CUDA
graphs, so a permutation-only (cap = this batch's max group) layout is not
capturable** — the padding stays, and the job is to make it as cheap as the
arithmetic allows. No dynamic-shape dispatch was built.

### 8.1 `_bucketed_dispatch` — moved to the bank's tail

The weight now rides the flat scratch before each bucket's `@ v`:
`w_flat [dump+1]` (0 at the dump slot) → `h * w_flat[offset:offset+n_b·cap_b]` →
one `bmm(h, v_b)` per bucket. The `[R, f]` fp32 product and the zero-row `cat`
are gone, and the dropped rows go to the same clamped-row/token-`n` sentinel.

**A by-product worth recording: the one-bucket bit-identity is BACK.**
`test_one_bucket_is_the_batched_arm` asserts `torch.equal` again (measured
`equal=True` in fp32 and bf16, cap 0 and 1.0): with both arms scaling `h` in rank
space, one bucket IS the same tensors in the same order. Between §7 and this
change the gate had to fall back to a dtype bound — that fallback is deleted, not
kept as a looser gate.

| `bucketed` vs the per-expert loop | measured | bound |
|---|---|---|
| fp32, 2 buckets, dropless | rel **1.608e-07** | 1e-5 |
| fp32, 2 buckets, capacity 1.0 | rel **1.755e-07** | 1e-5 |
| bf16, 2 buckets, dropless | rel **4.940e-03** | 1e-2 |
| bf16, 2 buckets, capacity 1.0 | rel **5.382e-03** | 1e-2 |
| bf16 vs float64 (2 buckets) | **2.271e-02 vs the loop's 2.204e-02 = 1.03×** | ≤1.25× |

The 1.25× on the last row (and on the bank arm's 0.92×) is a *separation*, not a
tolerance: both arms are independent bf16 rounding chains, so the ratio is noise
around 1, while a dropped or duplicated term moves it by the tensor's scale (2×
or worse).

### 8.2 The Triton arms — COUNTED, not rewritten

`moe_kernels.expert_gemm_glu` computes `h @ v` inside a kernel whose GLU output
never reaches a graph the caller can scale, and the rewrite is not local: the
router weight's own gradient is `dw = ⟨dout, h @ v⟩`, i.e. the **unscaled** expert
output — the very `[R, f]` tensor this fix removes from the bank arm. The split
GLU path (`E < FUSED_GLU_MIN_E`, our E=8 arm) does materialise `[R, rank] h`, but
inside `_ExpertGemmGlu`'s autograd Function, so scaling it from the caller would
silently drop `dw`.

So the Triton arms keep the wide product and SAY SO:

| arm | wide `[R, f]` fp32 product | counter |
|---|---|---|
| `moe_kernel="padded"`/`"batched"` | no (rank-space `h·w`) | `moe_w_in_bank = 1` |
| `moe_kernel="bucketed"` | no (rank-space, per bucket) | `moe_w_in_bank = 1` |
| `triton`, `moe_combine="torch"` (default) | **yes** — `_Combine.forward`'s `y.to(fp32) * w` | `moe_wide_weight = 1` |
| `triton`, `moe_combine="fused"` | no — the kernel reads the bank-dtype `y` and scales in registers | `moe_wide_weight` absent |
| `triton`, `fused_bwd=False` (the reference adjoint) | **yes** — `_triton_dispatch`'s own `y.to(fp32) * w` | `moe_wide_weight = 1` |
| `moe_gather="kernel"` | same as its combine arm | `moe_wide_weight` when `combine != fused` |

`_count_wide_weight` carries that reasoning at the call site. No arm is silently
on the wrong side: the bank/bucketed arms assert the counter's ABSENCE
(`test_moe_batched` gate a3) and the Triton arm asserts its PRESENCE
(`test_moe_kernel`'s CUDA gate).

### 8.3 The capacity waste, as a formula and a counter

The padded arm's bill is exactly

```
rows_run   = E · cap            (bucketed: Σ n_b · cap_b)
waste_rows = rows_run − n·k     (counted only when > 0: a cap below the mean
                                 DROPS rows instead — `moe_dropped_assignments`)
ratio_x100 = round(100 · rows_run / (n·k))
```

Measured on the bench's own fixture (E=4, k=2, n=64, so `n·k = 128`):

| `moe_capacity` | rows_run | `moe_pad_waste_rows` | `moe_bank_ratio_x100` |
|---|---|---|---|
| 2.0 | 256 | 128 | 200 |
| 1.0 | 128 | 0 | 100 |
| 0.5 | 64 | 0 (64 rows dropped) | 50 |
| dropless (`0`) | `E · max group` | `E·max − n·k` | ≥100 |

`config.ModelConfig.moe_capacity` now carries the recommendation: **1.25 with
`moe_qb=true`** (Quantile Balancing holds each expert within ±5% of `topk/E`, so a
25% pad covers the spread) and **2.0 as the un-balanced default**, with those two
counters naming what a run actually paid. `tests/test_moe_batched.py` gate (a5)
pins the three rows of that table.

### 8.4 The per-layer census after the fix (b24: R=49,152, E·cap=98,304, f=1,664, rank=832)

| buffer, per layer | before §7 | after §7 | after §8 |
|---|---|---|---|
| fp32 router-weight product `[R, f]` | 312 MiB | 0 (bank) | 0 (bank **and** bucketed) |
| bf16 zero-row `cat` `[E·cap+1, f]` | 312 MiB | 0 (bank) | 0 (bank **and** bucketed) |
| bf16 `h·w` `[E, cap, rank]` | 0 | 156 MiB | 156 MiB (bucket: per bucket, same total) |
| fp32 scatter source `[R, f]` | 312 MiB | 312 MiB | 312 MiB |
| bank tensors `xg, g, u, h, y` | 936 MiB at cap 12,288 | 936·C/12,288 MiB | same |
| fp32 accumulate `[n, f]` | 156 MiB | 156 MiB | same |
| **Triton arms** | 312 MiB product | 312 MiB product | **312 MiB product (COUNTED)** |

### 8.5 The CUDA half of the gates

The CPU suite is green (full run, `CUDA_VISIBLE_DEVICES=""`, 476 s). The GPU
half is the part §7 could not run, and it is **still blocked by the guard**: the
owner's 3-seed A/B (`scripts/arch_sweep.sh`, 2 arms/seed, b24, 4,882 steps)
holds `/tmp/dormouse-gpu.lock` from 11:41 with one `timeout 16000` around all
six arms — seed 3 started 14:16, so the card is booked past 15:00. Two queued
attempts through `scripts/gpu_lock.sh` (queued 13:04 and 14:01) both expired with
**75 — nothing was measured** at 13:34 and 14:31, with the guard's own words in
the log (`loud: another GPU job held the lock for 30 min`). Nothing was killed
and no guard was bypassed.

What is queued, exactly (one command, ~10 min of card):

```bash
GPU_TIMEOUT=3600 bash scripts/gpu_lock.sh env PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_experts_gpu.py tests/test_moe_kernel.py tests/test_moe_sonic.py \
  tests/test_moe_batched.py tests/test_moe_dropless.py tests/test_compile_moe_gpu.py \
  -q -p no:logging
```

Every gate it covers, and the bound each one carried into this change:
`test_experts_gpu::test_padded_and_naive_agree_on_the_same_routing` (bf16 2e-3 →
1e-2), `test_moe_kernel::test_triton_and_padded_agree_on_the_same_rows` (2e-3 →
1e-2) plus the new `moe_wide_weight` gate, `test_moe_sonic`'s two card gates
(padded 2e-3 → 1e-2, triton kept at 2e-3), `test_moe_batched`'s capture gate,
`test_moe_dropless`, and `test_compile_moe_gpu` (the compile gate over the arm
whose dispatch changed).

The CUDA bounds moved on the measured CPU deltas (4.73e-3 bank, 4.94e-3
bucketed, both ≤1e-2) and the float64 closeness gate — not on a guess — but they
have NOT been observed on the card. That is the one claim in this document that
is [S] and not [M].
