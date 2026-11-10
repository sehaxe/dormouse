# 2026-11-10 — the MoE package's bill: a 299M-total / 100M-active bank does not fit the control's batch

Lane `port/`, plan `.bulba/plan.md` §T4. One question, numbers only: does a
299M-total / 100M-active routed bank (`ffn_form="bank"`, `n_experts=8`,
`moe_topk=2`, `moe_capacity=2.0`) cost the same step time and VRAM as the 95M
dense control, or does it force the batch down?

**Verdict. It forces the batch down.** At 299.15M total / 99.67M active the bank
**OOMs at both b24 and b16** — the two batches the control runs at 10.80 GiB and
7.53 GiB — and its largest measured working batch is **12** (b13, b14, b15 and
b16 OOM); at b12 the bank pays
**372.6 ms/step against 148.4 ms for the same-block dense control (2.51×) and
12.60 GiB against 6.15 GiB (2.05×)** for the same 12,288 bytes. The active count
buys compute, not memory: the bank's 100M active weights still need 3.15× the
checkpoint state (2.92 vs 1.01 GiB) and 1.9× the activation bytes per batch
element (0.81 vs 0.43 GiB), for 2.05× the peak at equal batch.

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
