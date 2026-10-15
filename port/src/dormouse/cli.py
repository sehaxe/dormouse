"""The dormouse CLI: `dormouse info|train|eval|decode`.

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
    tasks = build_tasks(post_cls, seed=run.model.seed, n=8)
    rows = []
    for task in tasks:
        group = [rollout(model, task, budget=post_budget, device=dev) for _ in range(post_steps)]
        advs = crr_advantage(group)
        rows.append(
            {
                "task": task["id"],
                "pass@budget": sum(a.artifact.passed for a in advs) / len(advs),
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
    ck = torch.load(Path(out_dir) / f"{ckpt_name}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    _train_stream, eval_stream = open_streams(run)
    if eval_stream is None:
        raise LoudError("no eval stream (the snapshot carries eval_data or eval=true)")
    loss, scored = _evaluate(
        model, eval_stream, run.train.eval_batches, device=None if dev.type == "cpu" else dev
    )
    console.print(
        Panel(
            f"[bold]{ckpt_name}[/] · [bold]{loss / _LN2:.3f}[/] BPB",
            subtitle=f"loss {loss:.4f} over {scored:,} B = {run.train.eval_batches}"
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
    ck = torch.load(Path(out_dir) / f"{ckpt_name}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    gen = torch.Generator().manual_seed(run.train.seed)
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


if __name__ == "__main__":
    main()
