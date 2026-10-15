# dormouse — the byte-level 1B framework

dormouse is a self-contained trainer for the biggest smart model that fits a
single RTX 5060 Ti 16 GB + 64 GB RAM: **byte-level, tokenizer-free**, sparse
MoE with a hashed n-gram memory, and a weight-shared loop — the whole stack
runs the quant ladder down to 1.28 bits/weight (LittleBit).

The PyTorch stack in `src/dormouse/` is the present and the future; the Rust
burn stack (`crates/`, `vendor/`) is the inspiration and measuring history
that produced every number this project trusts (see `docs/`).

## Quickstart (CLI)

```bash
cd port
uv sync --extra gpu          # torch 2.13.0+cu129 + FLA 0.5.2 (the measured pair)
uv run dormouse info         # resolved config + parameter budget (meta device)
uv run dormouse train small --data /path/corpus.bin --eval /path/tail.bin \
    --device cuda --json-log run.jsonl --set steps=2000 --set moe_lb_coef=0.01
uv run dormouse eval  --out-dir checkpoints --ckpt-name small
uv run dormouse decode --out-dir checkpoints --ckpt-name small --prompt "Привет" --n 200
uv run dormouse plot run.jsonl      # loss / held-out / tok-s panels to run.png
```

CPU-only: `uv sync` (no extra) — the fused arms fall back COUNTED, CUDA gates
skip, every behavior pin stays assertable.

## The model (defaults from `configs/b1.toml`)

| | |
|---|---|
| objective | next-byte CE, honest unweighted, fp32 logits under bf16 compute |
| attention | KDA (gated delta), in-kernel q/k L2 norm; FLA dependency, SDPA fallback COUNTED |
| MoE | topk-2 over 1024 **dual-latent GLU (TSCT)** experts + shared rank-192; padding-free cap-padded bmm; Load-Free balancing bias |
| memory | Engram host n-gram tables (FNV orders 2/3/4) behind a `min(w_mem, lam)` floor |
| loop | weight-shared `max_iter` iterations, AttnRes (paper form) or ReZero write |
| optimizer | Muon (Jordan NS coefficients, factor-space) on 2-D maps + AdamW elsewhere, WSD schedule |
| precision | bf16 factors + fp32 masters; storage ladder `bf16 → fp8 → NVFP4 → LittleBit` (gates pre-registered) |

The b1 preset is **1,066,354,113 params (57M active)** — `tests/test_param_count.py`
recomputes it against the module tree, so no preset comment can rot.

## Invariants (what makes it trustworthy)

From the lessons this project paid for (all in `../docs/decisions/2026-10-15-1b-readiness-plan.md`):

- **Loud failures, no silent fallback** — every degradation is LOUD (error
  naming cause + escape) or COUNTED (a counter says so); SILENT is a defect
  class, not a mode.
- **Counters as the reader's check** — `kda=fla,n=…`, `bf_channel=…`,
  `moe_dropped_assignments=…`; if an arm did not run, the log line says so.
- **Resume refuses a drifted config** — the checkpoint's `.config.toml`
  snapshot is authority; progress keys exempt (extending a run is legal).
- **Eval window is bytes** — printed on every eval line; BPB numbers are
  only comparable inside one window.
- **Math-level tests** — formula goldens (ByteFlow oracles 2e-7), recurrent
  anchors (KDA 1.10e-3 fp32), invariants per mechanism, CLI round-trip.

## Layout

```
src/dormouse/
  config.py        strict flat-TOML presets; every key, every refusal
  instrument.py    LOUD/COUNTED/SILENT + counters + step timer
  model/           tsct · experts(MoE) · kda · engram · loopblock · model · byteflow · decode
  train/           data (mmap byte streams) · optim (Muon/AdamW) · lr (WSD) · loop
  cli.py           info | train | eval | decode | plot
tests/             117 tests, CPU-true by default, CUDA gates opt-in
scripts/           bench_moe · smoke_gpu · smoke_byteflow
configs/           small (9,476,993) · b1 (1,066,354,113)
```

**Rust stays where it is already the best code** (data pipeline, Engram host
tables, ByteFlow patcher) — as a PyO3 extension (`ffi/`), wired when a
profile says a hot loop is CPU-bound. GPU-side Rust buys nothing here.

## What lives in `docs/`

`docs/decisions/2026-10-15-1b-readiness-plan.md` is the live plan (b1
arithmetic, substrate verdict, the port lane §7). `docs/research/` carries
the verified technology bets (Muon, ByteFlow, LittleBit, data strategy,
agentic post-training — every claim with a source). The burn crates in
`../vendor/dormouse-fused/` are the formula source, not the spec.
