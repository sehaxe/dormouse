# Benchmarking the engine: one shape, one ledger, one number

`port/scripts/bench_engine.sh` is the reproducible throughput benchmark for the
byte-level trainer. It measures **bytes/s, ms/step, peak GiB and an MFU
estimate** at a list of `batch × seq_len` shapes, using the real training
recipe — not a forward-only microbenchmark. Output is JSONL plus a markdown
table; both carry the shape, the FLOP ledger and the peak denominator that
produced them.

The rule this file exists to enforce:

> **A speed claim names its shape AND its ledger.** "50 kB/s" is not a claim.
> "50.2 kB/s at b8×s1024, `flagship3_dense`, `mod_ledger` bill 691.37
> MFLOP/byte, 47.7 TFLOP/s denominator" is.

## Running it

```bash
cd port

bash scripts/bench_engine.sh --quick            # 3 shapes x 50 steps
bash scripts/bench_engine.sh --full             # 7 shapes x 200 steps
bash scripts/bench_engine.sh --quick --dry-run  # commands + bills, CPU only, no card
```

`--dry-run` is the smoke test for the benchmark itself: it resolves every
command and bill and prints one JSON object, without touching the GPU. Run it
after changing a config or a flag.

Both measured modes **refuse loudly** (exit 2, no result file) when there is no
CUDA device or the card is not free. That is deliberate: `_lane.preflight`
allows one GPU process at a time (AGENTS §1.5), and a benchmark that skips
silently reports "nothing" as if it were a number.

The default recipe is `scripts/recipe_ladder.sh`'s: `configs/flagship3_dense.toml`,
`data_local/best/fwedu_train.bin`, `opt=mix`, `muon_lr_mult=12`, `lr=3e-4`,
`configs/mix_v7.toml`, `doc_masking=false`, seed 1, checkpoints off. `--fast`
adds the ladder's `FAST=1` rung (`ns_dtype=bf16`, `muon_state_dtype=bf16`).
`--set KEY=VALUE` appends any extra override (model keys are priced in the
ledger, train keys are passed through). Two departures from the ladder are
named because they change the measurement:

* `eval=false` — eval wall time enters the next telemetry window and would be
  read as a slower step. Quality is the ladder's job; this file measures rate.
* `log_every=10`, and the steady state is the **median of the last three log
  windows**, not a single window.

Defaults can be overridden with `--config`, `--data`, `--eval`, `--mix`,
`--seed`, `--shapes 8x1024,4x2048`, `--peak-tflops` and `--out-dir` (default
`~/logs/bench`). Output files are `<UTC-stamp>.jsonl`, `<UTC-stamp>.md`, one
`<stamp>.<shape>.log` per run, and `_bench_peak.py` (the generated wrapper
described under *peak GiB*).

### The shapes

| mode | steps | shapes (batch × seq_len) | what it isolates |
|---|---:|---|---|
| `--quick` | 50 | 8×1024, 4×2048, 2×4096 | the live shape vs longer windows at the same 8 192 B/step |
| `--full` | 200 | 8×512, 8×1024, 8×2048, 16×1024, 4×2048, 4×4096, 2×8192 | batch at s1024 (8/16), longer windows at b8/b4/b2, 4K–16K B/step |

Shape is not cosmetic: the attention core is `4·L·T·d`, so the FLOP bill grows
with `T` at constant bytes/step (691.37 → 814.06 → 1 059.42 MFLOP/byte for the
three quick shapes). A bytes/s number at one window does not transfer to
another.

## What the columns mean

| column | definition |
|---|---|
| `shape` | `batch × seq_len` for the run; `B/step` is the trainer's own `batch·seq_len·grad_accum` |
| `steps` | optimizer steps in the run; the first windows are warmup by construction |
| `ms/step` | `1 000 · B/step ÷ bytes_per_s`, from the trainer's telemetry |
| `B/s` | median `tokens_per_s` of the last three log windows (byte-level lane: 1 byte = 1 token) |
| `peak GiB` | `torch.cuda.max_memory_allocated()` over the run, in GiB — see below |
| `MFLOP/B` | training FLOPs per byte from `scripts/mod_ledger.py` at this config and this `T` |
| `TFLOP/s` | `FLOP/B · B/step ÷ (ms/step · 1e-3) ÷ 1e12` |
| `MFU` | `TFLOP/s ÷ --peak-tflops` (default 47.7), expressed as a percentage |
| `±3 win` | `(max − min) / median` of the three rate windows — the noise the number must beat |
| `status` | `OK`, `OOM` (counted, not skipped) or `ERROR` (the run stopped) |

`peak GiB` comes from a generated wrapper, `_bench_peak.py`. It invokes the real
`dormouse.cli` entry point with the same argv (so `LoudError` and exit codes are
the trainer's own), then reads the allocator's high-water mark and prints
`[bench] peak_gib=…`. It is **not** `nvidia-smi` residency and **not**
`max_memory_reserved`; the two are different numbers and are not mixed here.
`scripts/vram_ledger.py` is the memory ledger and uses the same convention for
its one measured anchor (`max_memory_allocated`, 7.2 GiB at b8×s1024).

`OOM` rows keep the shape, the bill and the peak reached before the failure —
an arm that does not fit is a result, and dropping it would silently flatter
the matrix. Nothing else is swallowed: a non-OOM failure stops the run with
exit 1 and the last error line.

## The ledger, and why there are three of them

This repo has **three different MFU ledgers in circulation**, and they disagree
by up to 3.9×. From `docs/research/2026-11-05-small-model-mfu.md` §0.2:

| ledger | what it counts | MAC/token | training MFLOP/B (×6) | 100 %-MFU floor at 45.7 TF |
|---|---|---:|---:|---:|
| **A — attention-only** | KDA projections + `out_proj`, latent rank 256 | 6.49 M | 38.9 | 1.175 MB/s |
| **A′ — attention-only, GQA 4** | same, `n_kv_heads=4` | 4.52 M | 27.1 | 1.686 MB/s |
| **B — full `macs_per_token`** | `iters·(attn + out_proj + d·f)`, f=4096 | 25.36 M | 152.2 | 300 kB/s |
| **B′ — full, GQA 4** | same, `n_kv_heads=4` | 23.40 M | 140.4 | 326 kB/s |
| **C — MoE dense-down** | MoE only, 114.9 GMAC/step | 14.0 M | 84.1 | 543 kB/s |

The `1.7 %` figure the brief quotes is ledger **A**'s MFU at 22 kB/s — under
ledger **B** the same run is 6.8 %, and with the real bf16 tensor peak it is
3.8 % (§0.2–0.3). That is bigger than every speedup the memo discusses.

The denominator is contested too (§0.3): **45.7 TFLOP/s** is a measured RTX
5060 Ti number, **47.7** is what `scripts/bench_compile.py` pins for this box,
and **~88.7** is the 5070 Ti's spec FP16-with-FP32-accumulate rate. A different
denominator is a different claim, so `--peak-tflops` is explicit and recorded
in every row.

This benchmark uses the **B-class full bill**: `mod_ledger.ledger(cfg,
capacity=cfg.mod_capacity, seq_len=T).total` (×3 for forward → forward+backward,
the counter's own factor). For `flagship3_dense` at s1024 it reproduces the
measured `691.37 MFLOP/byte` (`docs/reviews/2026-11-06-control-transformer.md`
§1, FlopCounterMode), which is the strongest anchor the repo has. At other
windows and configs it is the repo's analytic bill, not a counter run.

Do not use `dormouse info`'s **active MAC/B** column for MFU. It prices one
loop body (`max_iter` passes); for `block_form="transformer"` it does not
multiply by `n_layers`, and it omits the `T`-dependent attention core and the
head. It is a useful sizing table, not a training bill. `mod_ledger.py` is the
module that includes depth, window and head — that is why the benchmark calls
it.

## Reproducing a claim

1. Run the benchmark at the shape you are claiming, on a free card, from a
   clean tree (`git rev-parse` and a dirty flag are in the JSONL header).
2. Quote the row: `bytes/s`, `ms/step`, `peak GiB`, `MFLOP/B`, `MFU`,
   `±3 win`, and the shape. The markdown table already carries all of them.
3. Name the ledger and the peak denominator. If you quote an MFU, say which
   one; if you quote bytes/s, say which shape.
4. A single run is a point, not a law. The three-window spread is the minimum
   noise estimate; a claim that moves less than it needs another run.

CI/CPU gate for the script itself: `uv run pytest tests/test_bench_engine.py`
(the bill, the MFU arithmetic, the steady-state selection, the recipe argv, the
no-GPU refusal and `--dry-run`).
