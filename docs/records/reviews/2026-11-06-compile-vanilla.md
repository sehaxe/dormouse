# 2026-11-06 — the compile/fusion lane for the vanilla transformer block

**Status: code + CPU gates landed; every GPU number is UNMEASURED.** The card
held a control run for the whole lane, so this document contains (a) the
measured graph-break inventory (dynamo's break machinery is backend-independent,
and the CPU trace is the same trace the card runs), (b) the pinned arithmetic,
(c) the gates, and (d) the exact commands whose output is the arms table. Any
number not marked **[M]** below is **[D]** derived from the constants in §1.

Changed: `port/scripts/bench_compile.py` (new), `port/tests/test_bench_compile.py`
(new), `port/src/dormouse/model/experts.py` (one decorator + its docstring).

---

## 1. The arithmetic, pinned so a wrong denominator fails

The card's measured bf16 peak is **47.7 TFLOP/s**; the flagship's own bill is
**137.42 MFLOP/byte** (FlopCounterMode on `checkpoints/flagship2/flagship2.pt`,
b1 × s1024 fwd+bwd, `docs/records/reviews/2026-11-06-control-transformer.md` §1). The
one shape the brief pins is b8 × s1024 × grad_accum 3.

| quantity | value | how |
|---|---:|---|
| bytes per optimizer step | 24 576 | 8 × 1024 × 3 |
| FLOPs per optimizer step | 3.377 TFLOP | 137.42e6 × 24 576 |
| the recorded throughput | 8.5 TFLOP/s = **17.8 % MFU** (the brief rounds to 17.9) | given |
| B/s at that MFU | 61.9 kB/s | 8.5e12 / 137.42e6 |
| 1 % of peak, in B/s | **3.47 kB/s** | 0.477e12 / 137.42e6 |
| upper bound for removing ALL host dispatch | **1.60×** → 28.6 % MFU | 320 ms wall / 200 ms GPU busy, kernel profile 2026-11-05 **[M]** |

`tests/test_bench_compile.py::test_mfu_arithmetic_is_pinned` pins 47.7, 137.42e6
and 24 576 and checks the 8.5 TFLOP/s ↔ 17.8 % conversion; 45.7 (the 5060 Ti's
number) and 88.7 (the 5070 Ti's spec FP16/FP16) both fall outside the tolerance.
The bench's `--check-flops` re-measures the model's bill with FlopCounterMode and
prints LOUD if it drifts more than 10 % from the pinned constant, so a config
change cannot silently invalidate the denominator.

**Read the upper bound as the lane's ceiling, not a forecast.** Compile can only
remove host work; 1.60× is what a perfect collapse of the measured 120 ms host
gap would buy on the live m400b step. Anything above that has to come from the
kernels themselves.

## 2. Graph breaks — measured, by config and kernel

`torch._dynamo.explain` on the eager model, CPU (`CUDA_VISIBLE_DEVICES=""`),
one fwd+bwd. `graph_break_count` counts break EVENTS (dynamo also counts the
resumes and skips that produce no `break_reasons` entry); "reasons" is the
distinct-site count; "graphs" is how many graph segments the forward becomes.

| config | kernel | break events | distinct reasons | graphs | site (file:line) | reason |
|---|---:|---:|---:|---|---|---|
| flagship2, b1 s64 | `naive` | 27 | 13 | 28 | `model/experts.py:3114 _naive_dispatch` | Dynamic shape operator (`aten.nonzero`) |
| flagship2, b1 s64 | `batched` | 31 | 17 | 32 | `model/experts.py:2760 _batched_cap` | `Tensor.item()` |
| flagship2, b1 s64 | `bucketed` | 31 | 16 | 32 | `model/experts.py:2798 _bucket_plan` | Data dependent operator |
| flagship2, b1 s64 | `triton` | 27 | 13 | 28 | same as `naive` | triton is CUDA-only → CPU takes the COUNTED naive fallback |
| **`block_form="transformer"`, 13.5M, b2 s64** | — | **0** | **0** | **1** | — | **none: the vanilla forward is ONE graph** |
| `block_form="transformer"`, tiny, bf16 and fp32 | — | 0 | 0 | 1 | — | none |

The vanilla form is measured on the parallel lane's tree (uncommitted
`model/transformer.py` + `configs/mirror_control.toml`): embed → 12 pre-LN
blocks → RMSNorm → TSCT head → `_byte_ce`, **0 breaks, 1 graph** — measured on
the tiny form in fp32 AND bf16, and on the 13.5M `mirror_control` preset at
b2 × s64. That is the answer to "what blocks compilation" for this lane:
nothing does.

The flagship reads are the **CPU fallback**, and that matters: `moe_kernel=triton`
is CUDA-only, so on the card the routed MoE never enters `_naive_dispatch` and
the inventory must be regenerated there. The bench's explain pass does that
on the card before the first timed step and prints `file:line func` + the source
line, so the GPU number is one command away and is NOT inferred here.

### The top break sites, by the only cost measurable without the card

Per-break GPU milliseconds are not measurable on a busy card, so the cost column
is the one I could measure: how many graph segments one dispatch splits into
(fewer segments = fewer resumes and a smaller guard set), plus the reach.

| # | site | reason | measured cost | reach |
|---|---|---|---|---|
| 1 | `experts.py:3114 _naive_dispatch` | `aten.nonzero` on `unique().tolist()` | 4 breaks/dispatch → **1 after the fix** (5/6 → 2/3 with the fix) | only the non-triton fallback (CPU, and any GPU run without triton) |
| 2 | `experts.py:2760 _batched_cap` | `counts.max().item()` | 17 reasons / 32 graphs for the whole model | `moe_kernel=batched` only |
| 3 | `experts.py:2798 _bucket_plan` | `counts.tolist()` | 16 / 32 | `moe_kernel=bucketed` only |
| 4 | `experts.py:3314` call site | the disable skip itself (after the fix) | 1 per dispatch | same as #1 |
| 5 | — | — | the vanilla form has none | — |

Rows 2 and 3 are **documented** host reads (the arms' own docstrings name them
as the sync an allocation shape costs); they are COUNTED by the bench, not
silent, and neither is on the live recipe (`moe_kernel=triton`, dropless).

## 3. What was fixed

`model/experts.py::_naive_dispatch` now carries `@torch.compiler.disable` and a
docstring naming it as a COUNTED break. Measured on one
`SparseMoE(tiny_cfg())` dispatch:

| program | break events | graphs | reason |
|---|---:|---:|---|
| before | 5 | 6 | Dynamic shape operator (`torch.unique`) |
| after | **2** | **3** | Skip inlining `torch.compiler.disable()`d function |

and on the whole tiny model (2 iterations): 20 events / 21 graphs → **17 / 18**.
Eager is bit-identical (the decorator only affects tracing) and that is gated.
Nothing else in the vanilla path needed fixing: the block, the model wrapper and
the CE compile to one graph already.

## 4. The bench

`port/scripts/bench_compile.py` (from `port/`: `scripts/bench_compile.py`).

Arms: `eager`, `default` (`mode="default"`), `max-autotune`
(`mode="max-autotune-no-cudagraphs"`), `trainer` (`torch.compile(model,
dynamic=True)` — `train/loop.py:1383`'s own wrapper), `strict`
(`fullgraph=True`, opt-in: a break becomes an error). Per arm: a fresh model
reloaded from the same CPU `state_dict`, `--warm 3` untimed steps, `--reps 5`
timed runs, MIN ms/step. Every row reports ms, B/s, TFLOP/s, MFU, peak VRAM,
breaks, graphs, recompile-limit fallbacks, `opt.step` ms and its share.

Parity is hard: logits/loss ≤ 1e-4 relative, one optimizer step ≤ 1e-6 relative
on every parameter, on the same seed and batch — a failure is exit 4.

Batch scaling keeps `bytes/step` at 24 576 where the arithmetic allows it. Only
b=8 can (g=3); 24 576 / 16 384 = 1.5 micro-batches does not exist, so b=16 lands
on g=2 and b=32 on g=1, i.e. 32 768 B/step, and every row prints its own B/step
so the B/s column is read against it. The next batch is projected as
`steady + (peak − steady) × b / b0` (weights + optimizer state are the steady
floor, activations are linear in the micro-batch); above 15 GiB it prints the
projection, the floor and the slope and exits 3 without allocating.

## 5. Optimizer step

`opt.step` is timed separately (after the timed loop, on the same grads, so the
measurement is not serialised by a synchronisation inside a timed step). When it
exceeds 15 % of the step the bench runs the two routes on identical weights and
grads — the port's fused writer (`MixOptimizer.step` →
`_foreach_adamw_step`) versus torch's own AdamW (the foreach route) beside the
batched Muon — prints both wall times and refuses the fused route if the 20-step
bpb trajectory moves by more than 1e-4.

Measured on the vanilla form (13.5M, b2 × s64, **CPU**, so the milliseconds are
a structural comparison, not a card number):

| route | opt.step | bpb after 20 steps |
|---|---:|---:|
| fused (`opt.step`) | 2 778.75 ms | 4.536300 |
| torch AdamW + batched Muon | 3 481.30 ms | 4.536300 |

The fused writer is 20.2 % faster on CPU with **0.00e+00** bpb drift over 20
steps, i.e. the same optimizer to the printed precision. The GPU share is
whatever the arms table says; the comparison runs automatically above 15 %.

## 6. Gates (five, each shown red on a scratch revert)

`port/tests/test_bench_compile.py` (10 tests, ~28 s, CPU):

1. **MFU arithmetic pinned** — 47.7 / 137.42e6 / 24 576, the 8.5 TFLOP/s ↔
   17.8 % conversion, and the default denominator IS the pinned one.
   RED: `PEAK_TFLOPS = 45.7`.
2. **The compile fallback is COUNTED, never silent** — every compiled row's
   `breaks` comes from the runtime `graph_break` counter and the explain table
   names the site. RED: removing `@torch.compiler.disable`.
3. **A busy card is exit 2 with the pid named** — real `nvidia-smi` parsing with
   a stub binary on `PATH`, then the exit code. RED: `EXIT_BUSY = 0`.
4. **The VRAM projection refuses** — projection arithmetic, and an end-to-end
   run that must stop at the second batch with exit 3. RED: `EXIT_VRAM = 0`.
5. **Parity is a hard error** — the three bounds and the failure list.
   RED: `parity_failures` returning `[]`.

Plus: the eager arm is bit-identical to the trainer's own
`zero_grad → backward → step` (same seed, same tensors), and `--form
transformer` SKIPs loudly on a tree without `block_form` instead of guessing a
second spelling of the flag.

## 7. The single most promising next lever

**Raise the micro-batch, hold the bytes: b=32 / g=1 at 32 768 B/step.** At this
shape every GEMM's M is `b × 1024` tokens, and the step is host-bound (a
measured 120 ms of host gap over 320 ms of wall), so the two levers multiply:
compile removes dispatch, the batch fills the tensor cores. Size **[D]**: the
1 % MFU = 3.47 kB/s conversion above means every MFU point is worth ~3.5 kB/s,
and the compile ceiling on the live step is 1.60× (28.6 % MFU); the batch half
is the one number this lane could not measure — `--batch-scale` prints it and
refuses above 15 GiB before it can OOM.

Not the lever: `mode="max-autotune"` and CUDA graphs. `max-autotune` only
re-picks the same GEMM tiles at these shapes (K=304/1024, N=256 is 91 % of the
big-shape rate per the 2026-11-05 memo — the cliff is K=64, which we are not
near), and the graph arm is refused with `torch_compile`
(`train/loop.py:1369`).

## 8. The exact GPU commands

Run from `port/` (the worktree once merged; `scripts/bench_compile.py` is
port-relative). One GPU process at a time — the bench itself exits 2 if the card
is held.

```bash
# 1. THE ARMS at the flagship shape: the table the brief asks for.
#    Compile/autotune time is the unknown; max-autotune is the slow arm.
cd /home/sehaxe/dormouse/port
uv run --extra gpu python scripts/bench_compile.py \
  --snapshot /home/sehaxe/dormouse/checkpoints/flagship2/flagship2.config.toml \
  --batch 8 --seq 1024 --grad-accum 3 \
  --arms eager,default,max-autotune,trainer \
  --warm 3 --reps 5 --steps 1 --flops-per-byte 137.42e6 \
  --json-out /tmp/opencode/compile-flagship2.json

# 2. Batch scaling: b=8/16/32, grad_accum adjusted, VRAM projected and refused.
uv run --extra gpu python scripts/bench_compile.py \
  --snapshot /home/sehaxe/dormouse/checkpoints/flagship2/flagship2.config.toml \
  --batch 8 --seq 1024 --grad-accum 3 --arms default,trainer \
  --batch-scale --batches 8,16,32 \
  --json-out /tmp/opencode/compile-batch.json

# 3. The vanilla form (after the parallel lane lands): 0 breaks expected.
uv run --extra gpu python scripts/bench_compile.py \
  --preset mirror_control --form transformer \
  --batch 8 --seq 1024 --grad-accum 3 --flops-per-byte 135.36e6 \
  --arms eager,default,max-autotune,trainer --batch-scale

# 4. The one-graph proof: fails with the break named if the shape is not whole.
uv run --extra gpu python scripts/bench_compile.py \
  --preset mirror_control --form transformer --arms strict --no-parity

# 5. The optimizer arm on demand (it also runs by itself above 15%).
... --opt-arms on --opt-steps 20
```

## 9. Full CPU suite

`pytest tests -q` in the worktree (`CUDA_VISIBLE_DEVICES=""`), 1 938 collected:
**one failure, `tests/test_sft_v2.py::test_the_manifest_accounts_for_every_row`**
— `code_fable: manifest vs files disagree`, i.e. a worktree-data artefact (the
manifest is tracked, the `sft_pairs` corpus rows are not). It fails **identically
on clean main `e2a0542`**, so it is not this lane's. The four failures the brief
lists (`test_export`, `test_fullattn::gate_7`, `test_loop_blocks`,
`test_prelude`) do not reproduce in this worktree — all 46 tests in those four
files pass — and nothing new is red because of this change. `ruff check` is
clean on the three touched Python files.

## 10. What is NOT measured here

* Every millisecond, B/s, TFLOP/s, MFU and GiB in the arms table — the card was
  busy. The CPU smoke of the same program passes both correctness gates
  (`0.00e+00` parity on the eager/compiled pair with `--backend eager`, the
  fused-vs-torch bpb drift in §5), so what is missing is card time, not
  plumbing.
* The GPU break inventory for the live recipe (`moe_kernel=triton`): the CPU
  trace cannot enter a CUDA kernel and takes the counted naive fallback instead.
  the bench's explain pass is what answers it.
* Per-break GPU cost. The method is a scratch A/B of the `torch.compiler.disable`
  region (the same measurement that sizes §3 on CPU); it needs the card.
