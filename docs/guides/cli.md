# Every flag on `train`

**Source of truth: `./target/release/train --help`**, transcribed on
2026-10-01 at commit `32616c9` from the release binary built the same day. The
transcription is grouped by what a flag *decides*, not by the order clap prints
it; the flag set itself is unchanged. Where the binary's own help text is wrong
or advertises a flag that does not exist, that is called out rather than
smoothed over.

Train-layer defaults live in exactly one place, `TrainCfg::default`
(`crates/dormouse-train/src/lib.rs:166-185`), and model-schema defaults live in
the serde defaults on `DormouseConfig` (`crates/dormouse-core/src/config/schema.rs`),
whose schema defaults **are** the `small` preset (asserted by
`default_equals_small_preset`). A flag that is `Option`-typed is a *layer
override*: absent means "keep the preset's value", which is why `--steps` has no
default in `--help` but is 100000 in practice.

## Data and config

| flag | default | what it decides |
|---|---|---|
| `--data <dir>` | **required** | the streamed byte corpus. A directory; every readable file under it is streamed. A missing path or empty directory is a hard error by design |
| `--eval <dir>` | none | held-out corpus for the eval cadence. Without it, no BPB is ever printed |
| `--preset <name\|path>` | `small` | a name in `configs/` or a path to a `.toml` |
| `--config <path>` | none | an alias for `--preset` when given a path |
| `--set k=v` | — | override any schema key; repeatable. Resolution order is preset → `--set` → typed flags (`resolve`) |

## Schedule and cadence

| flag | default | what it decides |
|---|---|---|
| `--steps <n>` | 100000 | step budget |
| `--seq-len <n>` | 512 | context length in **bytes** — the stream is bytes, one position is one byte |
| `--batch <n>` | 3 | sequences per step. **It also sets the eval window** (see below) |
| `--log-every <n>` | 100 | loss/aux line cadence |
| `--ckpt-every <n>` | 1000 | checkpoint cadence; 0 = off, and the final model is always saved |
| `--eval-every <n>` | 0 (off) | held-out eval cadence |
| `--ckpt-name <name>` | `latest` | `<name>.bin` inside `--ckpt-dir`; **resume reuses the same name** |
| `--ckpt-dir <dir>` | `checkpoints` | where checkpoints and the `.config.toml` snapshot land |

Resuming reuses `--ckpt-name`. A resume whose resolved config differs from the
`<ckpt_name>.config.toml` snapshot is a hard error (ADR-0005 / ADR-0021), except
for `steps`, `log_every`, `ckpt_every`, `eval`, `eval_every` — extending a run is
a legal resume, changing the objective is not.

## Optimizer

| flag | default | what it decides |
|---|---|---|
| `--lr <f>` | 1e-4 | |
| `--wd <f>` | 0.01 | weight decay |
| `--grad-clip <f>` | 1.0 | |
| `--opt mix\|mix-adan\|adan\|adamw\|muon` | `mix` | Muon+ mixed optimizer. The live policy is the path-marker implementation in `crates/dormouse-train/src/optim.rs`; the id-based declaration in `crates/dormouse-core/src/routing.rs` is exercised by tests only — say which one you mean |
| `--factors-fallback` | off | A/B: drop expert TSCT u/v factors from the Muon+ group to the fallback optimizer |
| `--seed <n>` | 1 | seeds the device RNG **before any parameter exists**, so it governs model init as well as the JEPA span mask; both are pure functions of `(seed, step)`. Part of the config snapshot |

`--seed` was verified cross-process on 2026-09-30 (`57237c3`, `58281c8`..`8da23bb`):
same seed gives 7 951 694 of 7 951 694 non-TSCT slots bit-identical across 9
processes; different seeds separate by relFro ≈ 1.414. The TSCT factors move
≤1.04e-06 relative, entering at `qr_householder`
(`vendor/dormouse-fused/crates/burn-spectral/src/lib.rs:792`; the `:689` line
AGENTS.md §3.7 quotes predates the `burn-fused` → `dormouse-fused` rename).

## Precision and quantization

| flag | default | what it decides |
|---|---|---|
| `--bf16[=true\|false]` | preset | bf16 storage; compute is fp32. Bare `--bf16` = true, `--bf16 false` disables, absent keeps the preset. **On this box every bf16 run is SLOWER than fp32** — the LLVM dialect has no bf16 type, so there is no bf16 matmul (ADR-0016) |
| `--quant fp32\|bf16\|fp16\|fp8\|fp4` | auto | force the TSCT **factor** format. Auto = bf16 mode → `Bf16`, sm ≥ 120 → `Fp8`, else `Fp32`. The forward path only; masters stay fp32, so checkpoints are format-agnostic |
| `--act-quant int4\|int8\|fp4` | off | BitNet a4.8-style **activation** quantization (STE, f32 graph) |
| `--act-group <n>` | 0 | activation-quant scale group size; 0 = per-token |

`ActFormat::attn()` upgrades the attention path unconditionally (`fp4 → int8`,
`int(b) → int(max(b, 8))`), so **`--act-quant fp4` has never run 4-bit
attention**, and `--act-quant fp4` numbers taken before `9b343d3` measured a
3-level quantizer, not e2m1. `4` and `8` are the pinned paths.

## Model arms and depth

| flag | default | what it decides |
|---|---|---|
| `--max-iter <n>` | preset | loop depth `T`. Fixed depth, no learned halt head (ADR-0013) |
| `--no-kda` | off | disable the attention arm (bisect / A-B) |
| `--no-engram` | off | disable the hashed-memory arm |
| `--rand-depth` | off | sample `T` in 1..=max_iter per step, a pure function of the step index. Mutually exclusive with `use_mor`; `resolve` refuses the pair loudly |

There is deliberately **no `--no-gr` and no `--no-mor`**: those are config, not
A/B flags (`--set use_gr=false`, `--set use_mor=true`).

## TSCT maintenance

| flag | default | what it decides |
|---|---|---|
| `--retract-every <n>` | 1 | TSCT polar retraction cadence in steps |
| `--retract-iters <k>` | 3 | Newton-Schulz iterations per retraction |
| `--retract-batched` | off | retract the TSCT masters **grouped by shape** — one sync-free batched Newton-Schulz per group instead of one per factor. Same numbers, fewer launches |

The retraction is a **fixed per-step cost that does not amortise**: measured
52.8 / 53.3 / 64.6 ms at batch 8 / 16 / 32 against step times growing 244 →
826 ms (2026-09-29), i.e. 22 % of a step at batch 8. `--retract-every 1000`
gives `retr=0.0` and a 188 ms step.

`--retract-batched` is **in the config snapshot** even though it changes no
number, so a resume cannot silently switch arms (ADR-0019: the two paths produce
the same loss, so nothing else in the log would say which ran). The step line
prints `retr_arm=batched:<n>/factor:<n>` either way.

Above `max_ortho` 1e-3 **per entry**, all TSCT factors irreversibly switch to
fp32, the checks stop, and the latch is persisted in the checkpoint. Masters are
always fp32, so `--quant` picks the forward path only and a `--quant fp32` run
after a fallback reproduces the old behaviour exactly.

## Auxiliary objectives

| flag | default | what it decides |
|---|---|---|
| `--jepa-weight <f>` | preset (`small`: 0.05) | JEPA aux: EMA-teacher masked latent prediction + KoLeo. 0 = off |
| `--dspark-weight <f>` | preset (`small`: 0.0) | DSpark draft head. 0 = off |
| `--dspark-k <n>` | preset (4) | draft depth K |
| `--jepa-targets <file>` | none | offline precomputed teacher latents: no per-step second forward, no EMA advance |
| `--jepa-precompute <n>` | none | precompute N steps of JEPA targets into `--jepa-targets`, then exit |

**There is no `--jepa-k`.** The draft depth K is `--dspark-k`.
`dspark_stride` (anchor spacing in byte positions, default 16) is config only
(`--set dspark_stride=…`); `0` is refused loudly.

The JEPA + KoLeo combination is the **only auxiliary arm with an A/B verdict**:
it beat pure CE 3 seeds out of 3 at 2k steps (6.343 vs 6.425 mean held-out BPB,
`d8062d1`, 2026-10-01).

## Memory offload

| flag | default | what it decides |
|---|---|---|
| `--engram-ram` | off | host-RAM n-gram tables: millions of rows in host memory, CPU Nesterov+Sinkhorn update, only the batch's rows copied to the device. **The flag value is per order** — 8M slots = 24M rows |
| `--engram-slots <n>` | 1 000 000 | slot count **per order** |
| `--host-adam-every <n>` | 1 | host-table update cadence in steps; 0 = off. The flag name says Adam; the update is Nesterov + Sinkhorn (glossary) |

Footprint is `slots × 3 orders × 32 dims × 4 B × 2 buffers`; the same formula as
the AGENTS.md §2.4-3 derivation, which quotes 12.3 GB at 48M slots. On this path
the in-model Engram table is squeezed to one row per order
(`crates/dormouse-train/src/cfg.rs`), because it is never read.

## Measurement

| flag | default | what it decides |
|---|---|---|
| `--eval-batches <n>` | 20 | batches averaged per held-out eval. **The window is `n × batch × seq_len` bytes** — 102 400 B at batch 10, 20 480 B at batch 2. Not a fixed size, and the byte count on the eval line is the authority |
| `--eval-depths` | off | also print held-out BPB at depths 1..=max_iter at every eval. No training |
| `--timers` | off | per-step GPU/CPU split — printed on `step % 50 == 0`, **not** on `--log-every` |
| `--memlog` | off | cubecl pool stats at log cadence |
| `--quant-check` | off | one-off quant-fidelity probe on the first step |
| `--autotune <level>` | runtime | cubecl autotune level (`minimal`/`medium`/`full`), passed to the runtime through the process environment |

A step-time reading with no step index is not a measurement of a step: the
cubecl autotune cache is cold for the first steps and the step was **23× slower
at step 0** than warm (5 549 ms vs ~245 ms, 2026-09-29).

`--eval-batches`'s own help string says "20 = 100 KB", which is the same
fixed-window claim AGENTS.md §2.6 retracts — read the byte count on the eval
line instead.

## Stability and process

| flag | default | what it decides |
|---|---|---|
| `--stress` | off | stress protocol: constant LR, loss-spike counter, p99.9 pre-clip grad norm |
| `--stress-lr <x>` | 1.0 | the multiplier (2× / 4× the base LR) |
| `--stress-every <n>` | 50 | stress report cadence |
| `--log <file>` | none | append stdout+stderr to this file. Logs belong in `/home/sehaxe/logs/`, not `/tmp` (a 32 GB tmpfs that ate a 33.5 GB sidecar once) |
| `--detach` | off | daemonize: ignore SIGHUP, fork to background |

There is no process-level auto-restart: a failure exits non-zero and the
operator resumes with the same `--ckpt-name`. The `--guard` wrapper that
re-exec'd from a pinned image was removed 2026-10-02 — its detached parent
exited 0, so orchestrators read a crashed run as a finished one
(`docs/reviews/unguard-2026-10-02.md`). The in-loop NaN firewall (loss masked,
gradients sanitized, the step skipped) is unconditional and was never behind a
flag (ADR-0011).

`DM_QUANT_DEBUG=1` is the one remaining environment variable, debug-only, and
prints every `LinearLike` quant format. Everything else is a typed flag — with
one train-layer field deliberately having no flag at all: `warmup` (two fwd+bwd
steps before the loop, which raises the cubecl pool high-water early) keeps its
schema default of `true` for long runs (`crates/dormouse-cli/src/bin/train.rs:244`).

## The other two binaries

`generate` and `serve` take **`--export <file.dmexp>` and nothing else**:

| binary | flags |
|---|---|
| `generate` | `--export` (required) · `--prompt` (default `hello`) · `--steps` (32) · `--temp` (0.8) |
| `serve` | `--export` (required) · `--port` (8000) |
| `export run` | `--ckpt-dir` (`checkpoints`) · `--ckpt-name` (`latest`) · `--dtype bf16\|f16\|f32` (bf16) · `--out` · `--preset` (`small`) · `--config` · `--set k=v` |
| `export info` | `<path>` — prints the header and verifies the CRC without loading a model |

Handed a training checkpoint they say what the file is and print the command
that converts it. That refusal is the point (ADR-0011): a 34 GB read to obtain
30 MB of weights is the cardinal sin wearing a plausible face. See
[the `.dmexp` guide](dmexp.md).

## Known-bad help strings — RESOLVED 2026-10-01 (`2d05cea`): the four strings below were fixed; kept as the record of what was wrong

- `--rand-depth`'s help text names **`--gen-max-iter`**, which does not exist.
  The CALM-lite confidence exit from ADR-0013 is not implemented either.
- `--host-adam-every`'s name says Adam; the update is Nesterov + Sinkhorn.
- `--preset`'s help lists five names and omits `nano-fused`, `mor` and `p150`.
  All eight files in `configs/` load; the list in the help string is stale.
- `--jepa-weight` and `--dspark-weight` say "default: preset (0.05)" / "(0.1)".
  The **schema** default is 0.05 and 0.0 (`configs/small.toml` sets
  `dspark_weight = 0.0`), so the 0.1 in that string is stale.

These are code-vs-document disagreements of the class AGENTS.md §1.7 says to
report rather than resolve with a third name. The fixes live in
`crates/dormouse-cli/src/bin/train.rs`; they are reported here, not applied,
because that file belongs to another lane.