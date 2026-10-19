# Reproducing the numbers

The README quotes a handful of figures. This page says, for each, **the command,
the log it came from and the commit** — and what a stranger cannot reproduce on
their own box without the corpora.

The rule the project works by: a number without its config, window and commit is
not a number (AGENTS §1.4). The window is `eval_batches × batch × seq_len` bytes
and it is printed on every eval line.

## The headline table

| number | how it was produced | log | notes |
|---|---|---|---|
| **BPB 1.936** (`mix1`, 60k steps) | `dormouse train medium --data <filtered web> --eval <eval_mix> --set seq_len=1024 --set batch=24 --set mix_plan=configs/mix_big.toml --set engram_ram=true --set moe_capacity=0 --set moe_kernel=triton` | `~/logs/mix1_long.log`, eval lines | window 196 608 B (`8 × 24 × 1024`) |
| **BPB 1.917** (`mix1`, 80k steps) | the same run, continued | same log, `eval step=80000` | the best base on disk |
| **BPB 1.915** (`mix1`, 85k, after the resume) | resume with `--set steps=170000 --set moe_lb_coef=0` | `~/logs/mix1_long2.jsonl`, `event=eval` | the `.lfb` break-glass is COUNTED on the load line |
| **40.9k tok/s** (medium) | the same command; the rate is on every log line and in the JSONL's `tokens_per_s` | `mix1_long.jsonl` | b24 × s1024, 12 GiB, one RTX 5060 Ti |
| **BPB 2.073 / CE 1.437** (`sft12`) | `dormouse sft medium --data <sft12.bin> --build-sft --chat-data … --init-from checkpoints/mix1/mix1.pt --set steps=4000 --set eval=true` | `~/logs/sft12.log` | window 32 768 B; `first=` is the first-byte agreement, not accuracy |
| **HellaSwag 30.5 / ARC-E 27.2** (zero-shot, n=1000) | `dormouse bench --ckpt-name mix1 --out-dir checkpoints/mix1 --tasks arc_easy,hellaswag --limit 1000` | `~/logs/*bench*.jsonl` | byte-level `acc_mean`; the protocol differs from lm-eval's token-level `acc_norm` (see the scorecard) |
| **1.07B total / 57M active** | `dormouse info b1` | `tests/test_param_count.py` pins it | 1 066 358 209 at `max_iter=4`; the host-table run removes the 402M Engram rows from the model half (88.1M on `medium` with `engram_ram`) |
| **968 tests** | `uv run pytest` | CI | `CUDA_VISIBLE_DEVICES=""` on a box whose card is busy |

## What makes a run trustworthy here

- **The eval window is printed** and the eval rewinds its stream (`rewind()`), so
  two evals of one checkpoint score the same bytes.
- **The init is seeded** (`torch.manual_seed(run.train.seed)` before the model is
  built), so two processes with one seed produce the same weights — a gate
  (`tests/test_seed_init.py`) builds it twice and compares hashes.
- **A resume refuses a drifted config** (ADR-0021); only the progress keys
  (`steps`, `log_every`, `ckpt_every`, `eval`, `eval_every`, `json_log`) may
  change.
- **Counters separate the arms**: the eval line prints `engram=<rows>/<arms>`,
  `periter=`, and `depths=` when asked; a fallback prints `fallback: <reason>`,
  and the JSON row carries `why`.

## What a stranger cannot reproduce

1. **The corpora.** 19.7 GB of filtered web, the 230 GB four-source mixture and
   the 196 KB/500 MB eval carves live outside the repository (an external
   drive). `docs/guides/data.md` has the smallest honest recipe for building
   your own; the numbers above will differ on your text.
2. **The GPU.** sm_120-specific: FLA's KDA kernel, Triton MoE, `tf32` and the
   bf16 attention path were measured on one RTX 5060 Ti. On sm_75/86 the
   capability layer picks fp16/fp32 and the Triton arms refuse; on CPU the
   attention arm is the SDPA fallback, which is a *different algorithm* —
   a BPB taken there is not this model's.
3. **The checkpoint pairs.** `mix1.pt` travels with `mix1.config.toml` and
   `mix1.ngram` (host tables); a load refuses a missing half rather than
   starting from fresh memory.

## The artifacts on disk

| path | what |
|---|---|
| `checkpoints/mix1/` | the best base: `.pt` + `.config.toml` + `.ngram` (402 MB host tables) |
| `checkpoints/sft/sft12.*` | SFT on that base — the checkpoint behind the night report's chat transcript |
| `checkpoints/sw1/` | `small_wide` (67.7M stored, E=256) at 35k steps, stopped for the long run |
| `checkpoints/b1night/`, `medium_big2/`, `sft/sft9/` | **do not use**: invalidated by the expert-bank cache defect (AGENTS.md's retraction section) |
