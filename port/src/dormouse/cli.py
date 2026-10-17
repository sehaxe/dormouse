"""The dormouse CLI: `dormouse info|train|sft|eval|decode|serve`.

A thin, pretty front over `config.load_run`, `train.loop`, and `model.decode`
— config strictness, resume refusal and counter discipline are already
library rules, and the CLI adds none of its own. The log line stays plain
`print`: it is the interface a training log must keep.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import PROGRESS_KEYS, UNSUPPORTED, ModelConfig, RunConfig, TrainConfig, load_run
from .instrument import LoudError, loud
from .model import DormouseModel, param_count
from .model.decode import decode_bytes, encode
from .serve import DEFAULT_MAX_TOKENS, build_server, serve_forever

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="dormouse — the byte-level LM trainer (PyTorch port).",
)
console = Console()
_LN2 = torch.log(torch.tensor(2.0)).item()


def main() -> None:
    try:
        app()
    except LoudError as e:
        console.print(f"[bold red]loud:[/] {e}")
        raise typer.Exit(1) from None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _split_sets(sets: list[str]) -> tuple[dict[str, object], dict[str, object]]:
    """`--set k=v` routed by schema: model keys vs train keys, unknown → loud.

    A bare value that parses as a TOML literal keeps that type (1, 1e-3,
    true); anything else is a string — the common case is a preset name or an
    optimizer mode.
    """
    import tomllib

    model_over: dict[str, object] = {}
    train_over: dict[str, object] = {}
    model_keys = {f.name for f in dataclasses.fields(ModelConfig)}
    train_keys = {f.name for f in dataclasses.fields(TrainConfig)}
    for s in sets:
        if s.count("=") != 1:
            raise LoudError(f"--set {s!r}: expected exactly one `key=value`")
        k, raw = s.split("=", 1)
        try:
            value: object = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw
        where = model_over if k in model_keys else train_over if k in train_keys else None
        if where is None:
            raise LoudError(
                f"--set {k}: neither a model nor a train key. Escape: fix the "
                f"typo, or add the field to its config class and wire it."
            )
        where[k] = value
    return model_over, train_over


def _device(name: str) -> torch.device:
    """`auto` = cuda when it is available AND ≥2 GiB free of the card; else cpu."""
    if name == "cpu":
        return torch.device("cpu")
    if name not in ("auto", "cuda"):
        raise LoudError(f"--device {name}: use auto|cuda|cpu")
    if not torch.cuda.is_available():
        if name == "cuda":
            raise LoudError(
                "--device cuda: torch.cuda.is_available() is False — a CPU-only "
                "torch got installed? Escape: --device cpu or sync the gpu extra."
            )
        return torch.device("cpu")
    free, _total = torch.cuda.mem_get_info()
    if free < 2 << 30:
        raise LoudError(
            f"--device cuda: {free >> 30} GiB free — another GPU process holds "
            "the card. One heavy thing at a time (readiness plan §7)."
        )
    return torch.device("cuda")


def _freeze_gc() -> None:
    """The GC-storm fix for long runs (gdb-found, 2026-10-16).

    A Python GC cycle scans every tracked object; a live model tree plus the
    optimizer's state runs that scan on a huge graph — a b1-shape compile
    spent 20+ minutes inside `gc_collect_main` at 100% CPU before this froze
    the world (the `deduce_unreachable` stack). `freeze()` moves everything
    alive at this point — modules, the model, the config — into the permanent
    generation, which is never scanned again; later short-lived rows (log
    dicts) still collect. Safe for a long run, wrong for nothing.
    """
    import gc

    gc.freeze()
    gc.set_threshold(1_000_000, 50, 50)


def _model(run) -> DormouseModel:
    return DormouseModel(run.model)


def _param_table(model: DormouseModel) -> Table:
    t = Table(title="parameters", title_style="bold")
    t.add_column("module", style="cyan")
    t.add_column("params", justify="right")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        t.add_row(name, f"{n:,}")
    t.add_row("[bold]total[/]", f"[bold]{param_count(model):,}[/]")
    return t


def _train_table(run) -> Table:
    t = Table(title="train", title_style="bold", show_header=False)
    for f in dataclasses.fields(TrainConfig):
        v = getattr(run.train, f.name)
        if f.metadata.get(UNSUPPORTED) and v != f.default:
            loud(
                f"resolved run carries {f.name}={v!r}: an unported mechanism "
                f"cannot sit inside a resolved run"
            )
        default = v == f.default
        t.add_row(
            f.name + (" (not ported)" if f.metadata.get(UNSUPPORTED) else ""),
            repr(v),
            style="dim" if default else None,
        )
    return t


def _run(preset: str, sets: list[str]):
    model_over, train_over = _split_sets(sets)
    return load_run(preset, model_overrides=model_over, train_overrides=train_over)


def _snapshot_run(
    out_dir: Path, ckpt_name: str, *, eval_file: str | None = None, eval_batches: int | None = None
):
    """Eval/decode resolve FROM THE SNAPSHOT, never from a preset: the snapshot
    IS the resolved run (`RunConfig.from_snapshot`). Only progress keys may be
    overridden; --set does not exist here on purpose (strict and predictable).
    """
    import dataclasses as _dc

    snap = out_dir / f"{ckpt_name}.config.toml"
    if not (out_dir / f"{ckpt_name}.pt").is_file() or not snap.is_file():
        raise LoudError(
            f"no checkpoint pair in {out_dir}: need both {ckpt_name}.pt and {ckpt_name}.config.toml"
        )
    run = RunConfig.from_snapshot(snap.read_text())
    over: dict[str, object] = {}
    if eval_file is not None:
        over["eval_data"] = eval_file
    if eval_batches is not None:
        over["eval_batches"] = eval_batches
    # `eval_data` is measurement, not a model-computing config — the snapshot's
    # strictness covers what the model computes; scoring against a different
    # held-out file is how a checkpoint is used, not a different experiment.
    allowed = set(PROGRESS_KEYS) | {"eval_data"}
    bad = [k for k in over if k not in allowed]
    if bad:
        raise LoudError(
            f"{bad} are not progress keys; eval/decode do not override them. "
            f"Escape: a new checkpoint, or drop the override."
        )
    if over:
        run.train = _dc.replace(run.train, **over)
    return run


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


@app.command()
def info(
    preset: Annotated[str, typer.Argument(help="preset NAME or a .toml PATH")] = "small",
    # typer constructs a fresh list per call (its contract) — B006 is a
    # plain-function footgun, and the option annotation IS the fixture.
    sets: Annotated[list[str], typer.Option("--set", help="override: key=value")] = [],  # noqa: B006
) -> None:
    """The resolved config and the parameter budget. Prints, trains nothing."""
    run = _run(preset, sets)
    model = DormouseModel(run.model, device="meta")
    console.print(Panel(f"[bold]{run.source}[/]", subtitle=f"{param_count(model):,} params"))
    console.print(_param_table(model))
    console.print(_train_table(run))


@app.command("train")
def train_cmd(
    preset: Annotated[str, typer.Argument(help="preset NAME or a .toml PATH")] = "small",
    data: Annotated[Path | None, typer.Option(help="a flat .bin byte corpus")] = None,
    eval_file: Annotated[
        Path | None,
        typer.Option("--eval", help="held-out .bin — its own file, or the no-leak tail split"),
    ] = None,
    batch: Annotated[int | None, typer.Option(help="tokens per step are batch*seq_len")] = None,
    steps: Annotated[int | None, typer.Option] = None,
    lr: Annotated[float | None, typer.Option] = None,
    device: Annotated[str, typer.Option(help="auto|cuda|cpu")] = "auto",
    resume: Annotated[
        bool, typer.Option(help="continue `ckpt_name`; a drifted config refuses")
    ] = False,
    json_log: Annotated[
        Path | None, typer.Option("--json-log", help="telemetry sink for `dormouse plot`")
    ] = None,
    # typer constructs a fresh list per call (its contract) — B006 is a
    # plain-function footgun, and the option annotation IS the fixture.
    sets: Annotated[list[str], typer.Option("--set", help="override: key=value")] = [],  # noqa: B006
) -> None:
    """Train. The log lines (loss / arms / counters) go to stdout as-is."""
    model_over, train_over = _split_sets(sets)
    if data is not None:
        train_over["data"] = str(data)
    if eval_file is not None:
        train_over["eval_data"] = str(eval_file)
    for k, v in (("batch", batch), ("steps", steps), ("lr", lr)):
        if v is not None:
            train_over[k] = v
    if json_log is not None:
        train_over["json_log"] = str(json_log)
    run = load_run(preset, model_overrides=model_over, train_overrides=train_over)
    dev = _device(device)
    model = _model(run).to(dev)
    _freeze_gc()
    from .train.loop import train as _train

    console.print(
        Panel(
            f"[bold]{run.source}[/] · {param_count(model):,} params · device {dev}",
            subtitle=f"steps={run.train.steps} batch={run.train.batch} seq={run.train.seq_len}"
            f" lr={run.train.lr} opt={run.train.opt}",
        )
    )
    report = _train(model, run, resume=resume, device=None if dev.type == "cpu" else dev)
    console.print(f"[green]done[/] steps={report['steps']} loss={report['loss']:.4f}")


@app.command()
def run(
    preset: Annotated[str, typer.Argument(help="preset NAME or a .toml PATH")] = "small",
    data: Annotated[Path | None, typer.Option(help="a flat .bin byte corpus")] = None,
    eval_file: Annotated[
        Path | None,
        typer.Option("--eval", help="held-out .bin — its own file, or the no-leak tail split"),
    ] = None,
    batch: Annotated[int | None, typer.Option(help="tokens per step are batch*seq_len")] = None,
    steps: Annotated[
        int | None, typer.Option(help="stage-1 token budget = steps*batch*seq")
    ] = None,
    lr: Annotated[float | None, typer.Option] = None,
    device: Annotated[str, typer.Option(help="auto|cuda|cpu")] = "auto",
    post_cls: Annotated[str | None, typer.Option(help="stage-2 task class: code|math")] = None,
    post_steps: Annotated[int, typer.Option(help="stage-2 groups per task")] = 2,
    post_budget: Annotated[int, typer.Option(help="stage-2 byte budget per rollout")] = 256,
    out_dir: Annotated[Path | None, typer.Option] = None,
    sets: Annotated[list[str], typer.Option("--set", help="override: key=value")] = [],  # noqa: B006
) -> None:
    """One button: stage-1 pretrain → checkpoint family → stage-2 post-train.

    The STAGE-EXIT rules are the research's, named (docs/research/
    2026-10-16-one-button-pipeline.md): stage 1 leaves on the TOKEN budget
    (the `steps` line) — not on a loss plateau, which no published pipeline
    uses; the terminal decay + checkpoint average is stage 1's ONLY
    irreversible act, and it re-enters from the pre-decay checkpoint family
    the same way later stages do (a stage never overwrites the family).
    Stage 2 rides the verifier-first reward: byte-budgeted rollouts on
    `post_cls` tasks, CRR fork-replay advantage, the pass-rate and the
    correct@budget printed — exactly the plan's success criterion.
    """
    from .post import build_tasks, crr_advantage, rollout
    from .train.loop import train as _train

    model_over, train_over = _split_sets(sets)
    if data is not None:
        train_over["data"] = str(data)
    if eval_file is not None:
        train_over["eval_data"] = str(eval_file)
    for k, v in (("batch", batch), ("steps", steps), ("lr", lr)):
        if v is not None:
            train_over[k] = v
    if out_dir is not None:
        train_over["out_dir"] = str(out_dir)
    run = load_run(preset, model_overrides=model_over, train_overrides=train_over)
    dev = _device(device)
    model = _model(run).to(dev)
    _freeze_gc()

    # ---- stage 1: pretrain, WITHOUT the decay (the family see below) ----
    console.print(
        Panel(
            f"[bold]stage 1[/] {run.source} · {param_count(model):,} params · {dev}",
            subtitle=f"steps={run.train.steps} batch={run.train.batch} seq={run.train.seq_len}"
            f" lr={run.train.lr} opt={run.train.opt}",
        )
    )
    report = _train(model, run, device=None if dev.type == "cpu" else dev)
    console.print(f"[green]stage 1 done[/] steps={report['steps']} loss={report['loss']:.4f}")

    # The stage gate: a post lane needs a task class, or the run ends here
    # (that IS the one button — stage 1 does not silently start stage 2).
    if post_cls is None:
        console.print("[yellow]stage 2 off[/] (no --post-cls); the family stays at the checkpoint")
        return

    # ---- stage 2: verifier-first post-train, byte-budgeted ----
    tokens = run.train.steps * run.train.batch * run.train.seq_len
    tasks = build_tasks(post_cls, seed=run.train.seed, n=8)
    rows = []
    for task in tasks:
        group = [rollout(model, task, budget=post_budget, device=dev) for _ in range(post_steps)]
        crr_advantage(group)
        rows.append(
            {
                "task": task.tid,
                # The METRIC is a well-defined fraction: rollouts that PASSED
                # verification at the byte budget. CRR's advantage mean is
                # negative when the whole group fails (the shape's design)
                # — the pass@budget fraction is what the report knows.
                "pass@budget": sum(r.passed for r in group) / max(1, len(group)),
                "budget": post_budget,
            }
        )
    passed = [r["pass@budget"] for r in rows]
    console.print(
        Panel(
            f"[bold]stage 2[/] {post_cls} · {len(rows)} tasks × {post_steps} rollouts ·"
            f" budget {post_budget} B",
            subtitle=f"mean pass@budget {sum(passed) / len(passed):.2f}"
            f" · tokens witnessed {tokens:,}",
        )
    )
    for r in rows[:4]:
        console.print(f"  {r['task']}  pass@budget={r['pass@budget']:.2f}")


@app.command("sft")
def sft_cmd(
    preset: Annotated[str, typer.Argument(help="preset NAME or a .toml PATH")] = "small",
    data: Annotated[
        Path | None, typer.Option(help="the SFT corpus .bin (records of seq_len+1 B)")
    ] = None,
    ckpt_name: Annotated[str, typer.Option("--ckpt-name", help="checkpoint name to write")] = "",
    out_dir: Annotated[Path | None, typer.Option(help="checkpoint directory")] = None,
    build_sft: Annotated[
        bool,
        typer.Option("--build-sft", help="(re)build the seeded instruction corpus at --data first"),
    ] = False,
    n_tasks: Annotated[
        int, typer.Option(help="code and math tasks per class in the built corpus")
    ] = 32,
    chat_data: Annotated[
        list[Path],
        typer.Option(
            "--chat-data",
            help="chat files mixed into the built corpus (repeatable): .jsonl or Q/A text",
        ),
    ] = [],  # noqa: B006 — typer constructs a fresh list per call (its contract)
    chat_cap: Annotated[
        int | None,
        typer.Option(
            "--chat-cap", help="max chat rows used across all files (a deterministic sample)"
        ),
    ] = None,
    steps: Annotated[int | None, typer.Option] = None,
    batch: Annotated[
        int | None, typer.Option(help="windows per step are batch*seq_len bytes")
    ] = None,
    lr: Annotated[float | None, typer.Option] = None,
    resume: Annotated[
        bool, typer.Option(help="continue `ckpt_name`; a drifted config refuses")
    ] = False,
    init_from: Annotated[
        str,
        typer.Option(
            "--init-from",
            help="a .pt whose WEIGHTS seed this SFT run (fresh optimizer/snapshot)",
        ),
    ] = "",
    device: Annotated[str, typer.Option(help="auto|cuda|cpu")] = "auto",
    json_log: Annotated[
        Path | None, typer.Option("--json-log", help="telemetry sink for `dormouse plot`")
    ] = None,
    # typer constructs a fresh list per call (its contract) — B006 is a
    # plain-function footgun, and the option annotation IS the fixture.
    sets: Annotated[list[str], typer.Option("--set", help="override: key=value")] = [],  # noqa: B006
) -> None:
    """SFT: the prompt-masked byte CE on a seeded instruction corpus.

    The base model is a FRESH init unless `--resume` names an existing
    checkpoint pair (which refuses loudly when either half is missing), so the
    bare command is the concept test — the format is learned, the knowledge is
    not there yet. `--build-sft` writes the corpus deterministically from
    `(seed, seq_len, n_tasks)`; a `--data` that does not exist without it is a
    loud refusal, because a pretrain corpus has no markers and the masked CE
    would refuse it one step later anyway.
    """
    from .train.sft import build_sft_corpus, sft_train

    model_over, train_over = _split_sets(sets)
    if data is not None:
        train_over["data"] = str(data)
    if ckpt_name:
        train_over["ckpt_name"] = ckpt_name
    if out_dir is not None:
        train_over["out_dir"] = str(out_dir)
    for k, v in (("steps", steps), ("batch", batch), ("lr", lr)):
        if v is not None:
            train_over[k] = v
    if json_log is not None:
        train_over["json_log"] = str(json_log)
    run = load_run(preset, model_overrides=model_over, train_overrides=train_over)
    if not run.train.data:
        raise LoudError(
            "sft needs --data (the SFT corpus .bin) or `--set data=/path`. Escape: "
            "--build-sft writes the seeded instruction corpus there first."
        )
    path = Path(run.train.data)
    if build_sft:
        built = build_sft_corpus(
            path,
            run.train.seed,
            seq_len=run.train.seq_len,
            n_tasks=n_tasks,
            chat_data=chat_data or None,
            chat_cap=chat_cap,
        )
        dropped = (
            f" · {built['dropped']} dropped (longer than one window)" if built["dropped"] else ""
        )
        chat_line = ""
        if built["chat_loaded"]:
            per_file = ", ".join(f"{Path(p).name}:{n}" for p, n in built["chat_files"].items())
            chat_line = (
                f" · chat {built['chat_used']}/{built['chat_loaded']} used ({per_file};"
                f" {built['chat_dropped_long']} long, {built['chat_dropped_dirty']} dirty dropped)"
            )
        console.print(
            f"[green]sft corpus[/] {built['records']} records x {built['record_bytes']} B "
            f"= {built['bytes']:,} B -> {built['path']} (seed {built['seed']}){dropped}{chat_line}"
        )
    elif not path.is_file():
        raise LoudError(
            f"sft: the corpus {path} does not exist and --build-sft was not given. "
            f"Escape: --build-sft (the seeded instruction corpus), or point --data "
            f"at an existing SFT corpus."
        )
    dev = _device(device)
    model = _model(run).to(dev)
    _freeze_gc()
    console.print(
        Panel(
            f"[bold]sft[/] {run.source} · {param_count(model):,} params · {dev}"
            + ("" if resume else (" · base = " + (init_from or "fresh init"))),
            subtitle=f"steps={run.train.steps} batch={run.train.batch} "
            f"seq={run.train.seq_len} lr={run.train.lr} opt={run.train.opt}",
        )
    )
    report = sft_train(
        model,
        run,
        resume=resume,
        init_from=init_from or None,
        device=None if dev.type == "cpu" else dev,
    )
    console.print(
        f"[green]sft done[/] steps={report['steps']} loss={report['loss']:.4f} "
        f"masked_bytes={report['supervised_bytes']:,} nan_skipped={report['nan_skipped']}"
    )


@app.command("eval")
def eval_cmd(
    out_dir: Annotated[Path, typer.Option(help="checkpoint directory")] = Path("checkpoints"),
    ckpt_name: Annotated[str, typer.Option(help="the checkpoint name to load")] = "",
    eval_file: Annotated[
        Path | None, typer.Option("--eval", help="score against THIS file")
    ] = None,
    eval_batches: Annotated[int | None, typer.Option("--eval-batches")] = None,
    device: Annotated[str, typer.Option(help="auto|cuda|cpu")] = "auto",
) -> None:
    """Held-out BPB of a checkpoint. It resolves from the checkpoint's own snapshot."""
    from .train.data import open_streams
    from .train.loop import evaluate as _evaluate

    if not ckpt_name:
        raise LoudError("eval needs --ckpt-name")
    run = _snapshot_run(
        out_dir,
        ckpt_name,
        eval_file=None if eval_file is None else str(eval_file),
        eval_batches=eval_batches,
    )
    run.train = dataclasses.replace(run.train, eval=True)
    dev = _device(device)
    model = _model(run).to(dev)
    _freeze_gc()
    ck = torch.load(Path(out_dir) / f"{ckpt_name}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    _train_stream, eval_stream = open_streams(run)
    if eval_stream is None:
        raise LoudError("no eval stream (the snapshot carries eval_data or eval=true)")
    loss, scored, fin, _per_iter = _evaluate(
        model, eval_stream, run.train.eval_batches, device=None if dev.type == "cpu" else dev
    )
    console.print(
        Panel(
            f"[bold]{ckpt_name}[/] · [bold]{fin / _LN2:.3f}[/] BPB (fin) · "
            f"{loss / _LN2:.3f} headline",
            subtitle=f"loss {loss:.4f} fin {fin:.4f} over {scored:,} B = {run.train.eval_batches}"
            f" x {run.train.batch} x {run.train.seq_len}",
        )
    )


@app.command()
def decode(
    out_dir: Annotated[Path, typer.Option(help="checkpoint directory")] = Path("checkpoints"),
    ckpt_name: Annotated[str, typer.Option("--ckpt-name")] = "",
    prompt: Annotated[str, typer.Option(help="prompt text; UTF-8 bytes are the tokens")] = "",
    n: Annotated[int, typer.Option(help="new bytes to sample")] = 100,
    temp: Annotated[float, typer.Option(help="sampling temperature")] = 0.8,
    device: Annotated[str, typer.Option(help="auto|cuda|cpu")] = "auto",
) -> None:
    """Sample bytes from a checkpoint (sliding-window next-byte decode)."""
    if not ckpt_name:
        raise LoudError("decode needs --ckpt-name")
    if not prompt:
        raise LoudError("decode needs --prompt (the first window is the prompt)")
    run = _snapshot_run(out_dir, ckpt_name)
    dev = _device(device)
    model = _model(run).to(dev)
    _freeze_gc()
    ck = torch.load(Path(out_dir) / f"{ckpt_name}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    gen = torch.Generator(device=dev).manual_seed(run.train.seed)
    out = decode_bytes(model, encode(prompt), n, temp=temp, device=dev, generator=gen)
    text = out[len(encode(prompt)) :].decode("utf-8", "replace")
    console.print(Panel(f"[bold]{prompt}[/]{text}", title=f"decode · temp {temp} · {n} bytes"))


@app.command()
def plot(
    json_log: Annotated[
        Path, typer.Argument(help="a run's telemetry .jsonl (train --json-log)")
    ] = ...,
    out: Annotated[
        Path | None, typer.Option(help="png path; default = the jsonl with .png")
    ] = None,
) -> None:
    """A training report from telemetry: loss panel, eval panel, throughput panel."""
    import json

    runs = [json.loads(line) for line in json_log.read_text().splitlines() if line.strip()]
    assert runs, f"{json_log}: empty telemetry (nothing was ever logged)"
    head = runs[0] if runs[0].get("event") == "run" else {}
    logs = [r for r in runs if r.get("event") == "log"]
    evals = [r for r in runs if r.get("event") == "eval"]
    assert logs, f"{json_log}: no log rows (train without --json-log?)"

    steps = [r["step"] for r in logs]
    losses = [float(r["loss"]) for r in logs if r["loss"] is not None and math.isfinite(r["loss"])]
    loss_steps = [r["step"] for r in logs if r["loss"] is not None and math.isfinite(r["loss"])]
    rates = [float(r["tokens_per_s"]) for r in logs if "tokens_per_s" in r]
    rate_steps = [r["step"] for r in logs if "tokens_per_s" in r]
    best = min((r for r in evals if r.get("bpb") is not None), key=lambda r: r["bpb"], default=None)
    out = out or json_log.with_suffix(".png")
    console.print(
        Panel(
            f"{json_log.name}: {head.get('source', '?')} · {len(steps)} rows, "
            f"steps {steps[0]}..{steps[-1]}",
            subtitle=(
                f"loss {min(losses):.4f}→{losses[-1]:.4f}"
                + (
                    f" · eval best {best['bpb']:.3f} BPB at step {best['step']}"
                    if best
                    else " · no eval rows"
                )
                + (f" · {max(rates):,.0f} tok/s peak" if rates else "")
            ),
        )
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("train CE", (loss_steps, losses), "log")]
    if evals:
        panels.append(
            ("held-out", ([r["step"] for r in evals], [float(r["bpb"]) for r in evals]), "linear")
        )
    panels.append(("tok/s", (rate_steps, rates), "linear"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.2 * len(panels)), sharex=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (title, (xs, ys), scale) in zip(axes, panels, strict=True):
        ax.plot(xs, ys, lw=0.9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(title)
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    console.print(f"[green]saved[/] {out}")


@app.command("serve")
def serve_cmd(
    preset: Annotated[
        str,
        typer.Argument(
            help="optional CHECK: given, it must resolve to the checkpoint's own numbers"
        ),
    ] = "",
    out_dir: Annotated[Path, typer.Option(help="checkpoint directory")] = Path("checkpoints"),
    ckpt_name: Annotated[str, typer.Option("--ckpt-name")] = "",
    port: Annotated[int, typer.Option(help="TCP port (0 = an ephemeral one)")] = 8765,
    host: Annotated[str, typer.Option(help="bind address")] = "127.0.0.1",
    max_tokens_cap: Annotated[
        int, typer.Option("--max-tokens-cap", help="hard ceiling on one reply, in bytes")
    ] = DEFAULT_MAX_TOKENS,
    temp: Annotated[float, typer.Option(help="sampling temperature")] = 0.8,
    device: Annotated[str, typer.Option(help="auto|cuda|cpu")] = "auto",
) -> None:
    """Serve a checkpoint on the OpenAI HTTP schema (`/v1/models`, chat completions).

    The model loads ONCE and is reused for every request, and resolution is the
    checkpoint's own snapshot (the `eval`/`decode` mechanics — the snapshot is
    authority, there is no `--set` here). `preset` is therefore a CHECK, not a
    source: given, its MODEL half must equal the snapshot's, or you are serving
    an architecture other than the one you named. Train-half keys are exempt —
    the cadence, optimizer and data path do not reach the served bytes, and
    every checkpoint on disk was trained with `--set` overrides.
    """
    if preset:
        claimed, snap = _run(preset, []), _snapshot_run(out_dir, ckpt_name)
        drift = [k for k in claimed.diff_keys(snap) if k.startswith("model.")]
        if drift:
            loud(
                f"serve {preset}: its model half differs from the checkpoint's own "
                f"snapshot in {drift}. Escape: drop the argument (the checkpoint IS "
                f"what is served), or point --ckpt-name at that run's checkpoint."
            )
    httpd = build_server(
        out_dir,
        ckpt_name,
        host=host,
        port=port,
        device=device,
        max_tokens_cap=max_tokens_cap,
        temp=temp,
    )
    state = httpd.state
    console.print(
        Panel(
            f"[bold]{ckpt_name}[/] · {param_count(state.model):,} params · {state.device}",
            subtitle=f"http://{host}:{httpd.server_address[1]}/v1 · cap {max_tokens_cap} B"
            f" · temp {temp}",
        )
    )
    serve_forever(httpd)


if __name__ == "__main__":
    main()
