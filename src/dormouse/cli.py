"""The dormouse CLI: `dormouse info|train|sft|rl|eval|bench|decode|serve`.

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

from .config import PROGRESS_KEYS, ModelConfig, RunConfig, TrainConfig, load_run
from .instrument import LoudError
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
    """Construct the model from `run` — seeded by the run, never by the caller.

    `train.seed` is recorded in every snapshot and every `.dormouse` header, so
    it must own the weights: `DormouseModel.__init__` draws from the global
    torch RNG, and before this call nothing in the train path seeded it (audit
    C1: two `seed=1` processes built `39cb6df0…` vs `c70cb36b…`). Set the seed
    HERE, immediately before the constructor, because this is the one place
    `train`/`run`/`sft`/`eval`/`bench`/`export`/`decode`/`serve` build a model
    (`serve.py` imports this helper). The RL lane builds its own
    (`post/rl_train.py`), so `rl_cmd` seeds immediately before that call.
    """
    torch.manual_seed(run.train.seed)
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
        default = v == f.default
        t.add_row(f.name, repr(v), style="dim" if default else None)
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


def _preset_check(preset: str, snap, *, what: str) -> None:
    """`preset` is a CHECK, not a source: given, its MODEL half must match the snapshot's.

    Train-half keys are exempt — the cadence, optimizer and data path do not
    reach the bytes a checkpoint produces, and every checkpoint on disk was
    trained with `--set` overrides. The shared rule for `serve` and `bench`
    (one definition, so the two cannot drift).
    """
    claimed = _run(preset, [])
    drift = [k for k in claimed.diff_keys(snap) if k.startswith("model.")]
    if drift:
        raise LoudError(
            f"{what} {preset}: its model half differs from the checkpoint's own "
            f"snapshot in {drift}. Escape: drop the argument (the checkpoint IS "
            f"what is {what}), or point --ckpt-name at that run's checkpoint."
        )


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
            "--chat-cap", help="max rows sampled from EACH chat file (a deterministic sample)"
        ),
    ] = None,
    holdout: Annotated[
        int,
        typer.Option("--holdout", help="records moved to a .holdout sibling for the in-loop eval"),
    ] = 0,
    eval_data: Annotated[
        Path | None,
        typer.Option(
            "--eval-data",
            help="holdout corpus for the in-loop SFT eval (masked CE + first-byte agreement)",
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
    `(seed, seq_len)`; a `--data` that does not exist without it is a
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
    if eval_data is not None:
        train_over["eval_data"] = str(eval_data)
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
            chat_data=chat_data or None,
            chat_cap=chat_cap,
            holdout=holdout,
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


@app.command("rl")
def rl_cmd(
    preset: Annotated[str, typer.Argument(help="preset NAME or a .toml PATH")] = "small",
    ckpt_name: Annotated[str, typer.Option("--ckpt-name", help="checkpoint name to write")] = "",
    out_dir: Annotated[Path | None, typer.Option(help="checkpoint directory")] = None,
    steps: Annotated[int | None, typer.Option] = None,
    lr: Annotated[float | None, typer.Option] = None,
    resume: Annotated[
        bool, typer.Option(help="continue `ckpt_name`; a drifted config refuses")
    ] = False,
    init_from: Annotated[
        str,
        typer.Option(
            "--init-from",
            help="a .pt whose WEIGHTS seed this RL run (fresh optimizer/snapshot)",
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
    """RL: the MiMo-style reward and a policy-gradient step on GENERATED tasks.

    There is no corpus to point at — a step draws its task from
    `build_tasks(cfg.rl_task_cls, seed, n=cfg.rl_tasks)` — so `--data`/`--eval`
    do not exist here. The lane's own knobs go through `--set`: `rl_task_cls`,
    `rl_tasks`, `rl_group`, `rl_budget`, `rl_adv`, and the reward's X/B/A/δ/s/γ.
    """
    from .post.rl_train import rl_train

    model_over, train_over = _split_sets(sets)
    if ckpt_name:
        train_over["ckpt_name"] = ckpt_name
    if out_dir is not None:
        train_over["out_dir"] = str(out_dir)
    for k, v in (("steps", steps), ("lr", lr)):
        if v is not None:
            train_over[k] = v
    if json_log is not None:
        train_over["json_log"] = str(json_log)
    run = load_run(preset, model_overrides=model_over, train_overrides=train_over)
    dev = _device(device)
    _freeze_gc()
    console.print(
        Panel(
            f"[bold]rl[/] {run.source} · {run.train.rl_task_cls} · {run.train.rl_tasks} "
            f"generated tasks · group {run.train.rl_group} · budget "
            f"{run.train.rl_budget} B · {dev}"
            + ("" if resume else (" · base = " + (init_from or "fresh init"))),
            subtitle=f"steps={run.train.steps} lr={run.train.lr} opt={run.train.opt} "
            f"adv={run.train.rl_adv}",
        )
    )
    # C1: `rl_train` constructs its own DormouseModel (`post/rl_train.py`), so
    # this is the seed call immediately before that construction. A direct
    # library caller of `rl_train` still bypasses it; the durable fix is the
    # same line inside `rl_train` (not this lane's file).
    torch.manual_seed(run.train.seed)
    report = rl_train(
        run,
        resume=resume,
        init_from=init_from or None,
        device=None if dev.type == "cpu" else dev,
    )
    per_solved = (
        f"{report['bytes_per_solved']:.1f}" if report["bytes_per_solved"] is not None else "-"
    )
    console.print(
        f"[green]rl done[/] steps={report['steps']} loss={report['loss']:.4f} "
        f"pass={report['pass_rate']:.2f} bytes/solved={per_solved} "
        f"rl_samples={report['rl_samples']:,} nan_skipped={report['nan_skipped']}"
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
    from .train.loop import load_model_half, load_model_state

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
    # The ONE loader (audit MAJOR 6 + the host-table path): the model half is
    # unpacked from its storage rung, and an `engram_ram` checkpoint switches
    # the model to host tables and restores its `.ngram` sibling — a raw
    # `load_state_dict` died on both (`Missing key(s): "loop.engram.tables"`).
    load_model_state(model, run.train, load_model_half(Path(out_dir) / f"{ckpt_name}.pt"))
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


@app.command("bench")
def bench_cmd(
    preset: Annotated[
        str,
        typer.Argument(
            help="optional CHECK: given, it must resolve to the checkpoint's own numbers"
        ),
    ] = "",
    out_dir: Annotated[Path, typer.Option(help="checkpoint directory")] = Path("checkpoints"),
    ckpt_name: Annotated[str, typer.Option("--ckpt-name", help="the checkpoint to score")] = "",
    tasks: Annotated[
        str,
        typer.Option(
            help="comma list: synth_math|arc_easy|hellaswag (default: synth_math,arc_easy; "
            "with --download: every task)",
        ),
    ] = "",
    data_dir: Annotated[Path, typer.Option("--data-dir", help="where the task files live")] = Path(
        "data_local/bench"
    ),
    limit: Annotated[int, typer.Option(help="items per task (0 = all)")] = 0,
    download: Annotated[
        bool, typer.Option("--download", help="fetch the named tasks into --data-dir and exit")
    ] = False,
    json_log: Annotated[
        Path | None, typer.Option("--json-log", help="one JSONL row per task (appended)")
    ] = None,
    device: Annotated[str, typer.Option(help="auto|cuda|cpu")] = "auto",
) -> None:
    """Downstream MC eval: ARC-Easy / HellaSwag / a synthetic arithmetic task.

    The score is the byte-level protocol (`bench/score.py`): every choice is
    teacher-forced in ONE forward over `context + choice` and scored by the sum
    of its bytes' log-probs. The printed headline is the per-byte MEAN and the
    raw sum rides next to it, because the two disagree when choices differ in
    length — a 1-byte choice and a 2-byte choice are not comparable as sums.

    The window is the model's own `max_seq_len`; an item that does not fit is
    DROPPED and counted (`bench_dropped`), never truncated. `preset` is a CHECK,
    not a source (the `serve` rule): the snapshot is authority, and a preset
    whose model half differs refuses. `--download` fetches the real tasks from
    HuggingFace's public JSON-rows endpoint into `--data-dir` and exits; a task
    that cannot be fetched refuses LOUDLY with its URL while the others land.
    """
    from .bench import TASKS, run_bench
    from .bench.score import model_logits
    from .bench.tasks import download as download_task
    from .instrument import Counters
    from .train.loop import load_model_half, load_model_state

    names = [t.strip() for t in tasks.split(",") if t.strip()]
    if not names:
        names = list(TASKS) if download else ["synth_math", "arc_easy"]
    if download:
        failed: list[str] = []
        for name in names:
            try:
                download_task(name, data_dir)
            except LoudError as e:
                failed.append(name)
                console.print(f"[bold red]loud:[/] {e}")
        if failed:
            raise LoudError(
                f"bench --download: {failed} did not land (the URLs above). The other tasks "
                f"are on disk; the harness refuses a missing task by name rather than "
                f"scoring one it did not read."
            )
        return

    if not ckpt_name:
        raise LoudError("bench needs --ckpt-name")
    run = _snapshot_run(out_dir, ckpt_name)
    if preset:
        _preset_check(preset, run, what="scored")
    dev = _device(device)
    if dev.type == "cuda" and run.model.use_kda:
        # The arm is checked BEFORE the run: `fla_status` is a tiny warmup call,
        # and a bench that discovers an hour later that it measured the SDPA
        # fallback has already spent the hour.
        from .model.kda import fla_status

        ok, why = fla_status(dev, recheck=True)
        if not ok:
            raise LoudError(
                f"bench: the FLA KDA kernel is not live on {dev} ({why}). The SDPA "
                f"fallback is a different algorithm, so its accuracy is not this "
                f"checkpoint's number. Escape: `uv sync --extra gpu`, or run --device cpu "
                f"and read the number as the COUNTED fallback's."
            )
    model = _model(run).to(dev)
    _freeze_gc()
    load_model_state(model, run.train, load_model_half(Path(out_dir) / f"{ckpt_name}.pt"))
    model.eval()
    # One counter set for the whole bench: the model's own arm counters (`kda_fla`
    # vs `kda_sdpa`, `engram_rows`), which each JSONL row carries.
    arm = Counters()

    def logits_fn(ids):
        return model_logits(model, ids.to(dev), arm)

    reports = run_bench(
        logits_fn,
        names,
        data_dir=data_dir,
        max_seq_len=run.model.max_seq_len,
        limit=limit,
        json_log=json_log,
        arm=arm,
    )
    macro = sum(r.acc_mean for r in reports) / len(reports)
    console.print(
        Panel(
            f"[bold]{ckpt_name}[/] · {len(reports)} tasks · {sum(r.n for r in reports):,} items ·"
            f" macro acc_mean {macro:.4f}",
            subtitle=f"device {dev} · max_seq_len {run.model.max_seq_len}",
        )
    )
    # The arm line goes OUTSIDE the panel: a Panel clips its subtitle at the
    # console width, and a clipped counter line is a counter nobody can read
    # (`kda_sdpa` was the field that got cut).
    console.print(f"bench arm: {arm.line()}")
    if dev.type == "cuda" and run.model.use_kda and not arm.get("kda_fla", 0):
        raise LoudError(
            "bench: on CUDA the KDA arm never ran the FLA kernel (kda_fla=0) although the "
            "pre-run check found FLA live — the model did not route through it, and the "
            "SDPA fallback is a different algorithm, so these accuracies are not this "
            "checkpoint's numbers. Escape: check the config's `use_kda`/dispatch seam, or "
            "run --device cpu and read the number as the COUNTED fallback's."
        )
    if dev.type == "cpu":
        console.print(
            f"[yellow]counted[/] CPU: the attention arm is the SDPA fallback "
            f"({arm.get('kda_sdpa', 0)} calls), not the FLA KDA kernel — the same counted "
            f"degradation the training forward takes on a CPU box."
        )


@app.command("export")
def export_cmd(
    preset: Annotated[
        str,
        typer.Argument(help="optional CHECK: given, it must resolve to the checkpoint's numbers"),
    ] = "",
    ckpt_name: Annotated[str, typer.Option("--ckpt-name", help="the checkpoint to export")] = "",
    out_dir: Annotated[Path, typer.Option(help="checkpoint directory")] = Path("checkpoints"),
    out: Annotated[
        str,
        typer.Option("--out", help="artifact path (default <out-dir>/<ckpt-name>.dormouse)"),
    ] = "",  # str, not Path: `Path()` is `.` and would read as "given"
    mode: Annotated[str, typer.Option(help="factor rung: ternary|mxfp4")] = "ternary",
    dense_dtype: Annotated[
        str, typer.Option("--dense-dtype", help="dense tensors: fp16 (default) | fp32")
    ] = "fp16",
    engram_dtype: Annotated[
        str, typer.Option("--engram-dtype", help="n-gram table: fp32 (default) | fp16")
    ] = "fp32",
    verify: Annotated[
        bool,
        typer.Option("--verify", help="read the file back and compare a forward to the source"),
    ] = False,
) -> None:
    """Write the one-file `.dormouse` artifact: config + weights for every reader.

    The readers are `dormouse decode/serve --export` (Python), the browser
    engine (`docs-site/public/demo/engine.js`) and distribution — one format,
    one writer. The snapshot decides the config (this is `eval`'s resolution
    rule); `preset` is therefore a CHECK, and a model half that disagrees with
    it is refused rather than exported under the wrong name.

    `--verify` is the gate that makes the file trustworthy: it reads the
    artifact back, checks that every record's bytes are where the header says,
    and forwards one batch through the source model (projection on) and the
    rebuilt one, printing the max |Δlogits|.
    """
    from .model.export import engine_refusal, load_export, read_export, verify_export, write_export
    from .train.loop import load_model_state, unpack_state_dict

    if not ckpt_name:
        raise LoudError("export needs --ckpt-name (the checkpoint to export)")
    run = _snapshot_run(out_dir, ckpt_name)
    if preset:
        _preset_check(preset, run, what="exported")
    weights = Path(out_dir) / f"{ckpt_name}.pt"
    model = _model(run)
    # The ONE loader `eval`/`decode`/`serve` share (`load_model_half` is exactly
    # these two lines): a storage rung's packed records unpack, and an
    # `engram_ram` checkpoint switches the model to host tables and restores
    # its `.ngram` sibling — which is what lets this export fold the memory IN
    # instead of writing a memory-less model under the run's name. The
    # checkpoint dict is read here rather than through `load_model_half` for the
    # `step` it also carries: the artifact's provenance, and a second full
    # `torch.load` (optimizer state included) is not worth a metadata field.
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    load_model_state(model, run.train, unpack_state_dict(ck["model"], ck.get("model_quant_rung")))
    step = ck["step"] if isinstance(ck.get("step"), int) else None
    commit = _git_commit()
    path = Path(out) if out else Path(out_dir) / f"{ckpt_name}.dormouse"
    report = write_export(
        model,
        path,
        mode=mode,
        preset=preset or run.source,
        seed=run.train.seed,
        step=step,
        commit=commit,
        dense_dtype=dense_dtype,
        engram_dtype=engram_dtype,
    )
    console.print(report.table())
    why = engine_refusal(run.model)
    engine_line = (
        "[green]runnable[/] (small-byte-lm)"
        if why is None
        else f"[yellow]NOT runnable[/] by docs-site/public/demo/engine.js — {why}"
    )
    console.print(f"browser engine: {engine_line}")
    if verify:
        torch.manual_seed(0)
        seq = min(64, run.model.max_seq_len)
        x = torch.randint(0, run.model.vocab, (2, seq))
        diff = verify_export(model, path, (x,))
        f = read_export(path)
        loaded = load_export(f)
        console.print(
            f"verify: max |Δlogits| = {diff:.3e} over {tuple(x.shape)} "
            f"(reference = the projection on the fp32 masters, loaded = the file) · "
            f"{sum(p.numel() for p in loaded.parameters()):,} params rebuilt"
        )
        if not math.isfinite(diff):
            raise LoudError("verify: non-finite logits — the rebuilt model is not the exported one")


def _git_commit() -> str | None:
    """`HEAD`'s short hash, or None — the artifact's provenance, never a guess.

    A non-git tree (a source tarball) leaves the header's `commit` null, which
    is the honest value; this is metadata, so it does not stop an export.
    """
    import subprocess

    try:
        got = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return got.stdout.strip() if got.returncode == 0 and got.stdout.strip() else None


@app.command()
def decode(
    out_dir: Annotated[Path, typer.Option(help="checkpoint directory")] = Path("checkpoints"),
    ckpt_name: Annotated[str, typer.Option("--ckpt-name")] = "",
    export_file: Annotated[
        str,
        typer.Option(
            "--export",
            help="read the model from a .dormouse artifact instead of a checkpoint "
            "(the browser demo reads the same bytes)",
        ),
    ] = "",  # str, not Path: `Path()` is `.` and would read as "given"
    prompt: Annotated[str, typer.Option(help="prompt text; UTF-8 bytes are the tokens")] = "",
    n: Annotated[int, typer.Option(help="new bytes to sample")] = 100,
    temp: Annotated[float, typer.Option(help="sampling temperature")] = 0.8,
    spec_k: Annotated[
        int,
        typer.Option(
            "--spec-k",
            help="speculative decode: bytes the draft proposes per round (0 = off, "
            "plain decode); needs a checkpoint trained with dspark_weight > 0",
        ),
    ] = 0,
    recurrent: Annotated[
        bool,
        typer.Option(
            "--recurrent",
            help="state-carry decode: carry the per-iteration KDA state across bytes "
            "(O(1)/byte, unbounded context) instead of re-forwarding the window "
            "(O(window)/byte). Off by default until the GPU gate has run.",
        ),
    ] = False,
    device: Annotated[str, typer.Option(help="auto|cuda|cpu")] = "auto",
) -> None:
    """Sample bytes from a checkpoint (sliding-window next-byte decode).

    `--spec-k N` is the speculative path (`model/specdecode.py`): the DSpark
    draft proposes N bytes, ONE target forward verifies them, and the accepted
    prefix is emitted — the same bytes the plain decoder emits at the same temp,
    for ~1 forward per N when the draft is right. The default 0 is exactly
    today's `decode_bytes`.

    `--recurrent` is the state-carry path (`model/decode.py`): one byte per step
    with the KDA state carried, `O(1)` per byte. Its bytes are NOT guaranteed to
    be the plain decoder's at the same seed — the step kernel and the chunked
    pass are the same recurrence with different summation orders (the `kda.py`
    numerics contract), which is why it is off by default and why the subtitle
    prints the counters of the arm that ran.
    """
    from .model.specdecode import spec_decode_bytes
    from .train.loop import load_model_half, load_model_state

    if not ckpt_name and not export_file:
        raise LoudError("decode needs --ckpt-name or --export")
    if ckpt_name and export_file:
        raise LoudError(
            "decode --ckpt-name and --export are two sources for one model. "
            "Escape: pass one of them (--export reads the artifact, --ckpt-name "
            "resolves through the checkpoint's snapshot)."
        )
    if not prompt:
        raise LoudError("decode needs --prompt (the first window is the prompt)")
    if spec_k and recurrent:
        raise LoudError(
            "decode --spec-k and --recurrent are two different decoders (a verified "
            "draft block vs a carried state), and neither is the other's control. "
            "Escape: pass one of them."
        )
    dev = _device(device)
    if export_file:
        from .model.export import load_export, read_export

        artifact = read_export(export_file)
        model = load_export(artifact, device=dev)
        seed = int(artifact.header.get("seed") or 1)
        source = str(export_file)
    else:
        run = _snapshot_run(out_dir, ckpt_name)
        model = _model(run).to(dev)
        seed = run.train.seed
        source = f"{Path(out_dir) / ckpt_name}.pt"
        # Same ONE loader as `eval`, `serve` and `load_ckpt`: the storage rung's
        # packed records are unpacked, and an `engram_ram` checkpoint switches
        # the model to host tables and restores its `.ngram` sibling.
        load_model_state(model, run.train, load_model_half(Path(out_dir) / f"{ckpt_name}.pt"))
    _freeze_gc()
    gen = torch.Generator(device=dev).manual_seed(seed)
    body = encode(prompt)
    if recurrent:
        from .instrument import Counters
        from .model.decode import recurrent_decode

        arm = Counters()
        out = recurrent_decode(model, body, n, temp=temp, device=dev, generator=gen, counters=arm)
        subtitle = f"recurrent · {arm.line()}"
    elif spec_k:
        out, stats = spec_decode_bytes(
            model, body, n, spec_k=spec_k, temp=temp, device=dev, generator=gen
        )
        subtitle = (
            f"spec k={spec_k} · accepted {stats.accepted}/{stats.drafted} "
            f"({stats.acceptance_rate:.0%}) · {stats.per_byte_forwards:.2f} forwards/byte"
            f" · {stats.counters.line()}"
        )
    else:
        out = decode_bytes(model, body, n, temp=temp, device=dev, generator=gen)
        subtitle = f"{n} bytes"
    text = out[len(body) :].decode("utf-8", "replace")
    console.print(
        Panel(
            f"[bold]{prompt}[/]{text}",
            title=f"decode · temp {temp}" + (f" · spec k={spec_k}" if spec_k else ""),
            subtitle=f"{subtitle} · {source}",
        )
    )


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
    export_file: Annotated[
        str,
        typer.Option(
            "--export",
            help="serve a .dormouse artifact instead of a checkpoint (the browser "
            "demo reads the same bytes)",
        ),
    ] = "",  # str, not Path: `Path()` is `.` and would read as "given"
    port: Annotated[int, typer.Option(help="TCP port (0 = an ephemeral one)")] = 8765,
    host: Annotated[str, typer.Option(help="bind address")] = "127.0.0.1",
    max_tokens_cap: Annotated[
        int, typer.Option("--max-tokens-cap", help="hard ceiling on one reply, in bytes")
    ] = DEFAULT_MAX_TOKENS,
    temp: Annotated[float, typer.Option(help="sampling temperature")] = 0.8,
    recurrent: Annotated[
        bool,
        typer.Option(
            "--recurrent/--no-recurrent",
            help="state-carry decode (O(1)/byte) when the model supports it — the "
            "default; --no-recurrent forces decode_bytes (O(window)/byte). A model "
            "that cannot run it falls back COUNTED (recurrent_fallback).",
        ),
    ] = True,
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

    Decode is the state-carry arm by default (`recurrent_decode`, `O(1)`/byte);
    `--no-recurrent` forces the window re-forward, and a model that cannot run
    recurrent (byteflow, MSA) falls back COUNTED with the reason on the startup
    line.
    """
    if not ckpt_name and not export_file:
        raise LoudError("serve needs --ckpt-name or --export")
    if ckpt_name and export_file:
        raise LoudError(
            "serve --ckpt-name and --export are two sources for one model. Escape: "
            "pass one of them (--export reads the artifact, --ckpt-name resolves "
            "through the checkpoint's snapshot)."
        )
    if preset and export_file:
        raise LoudError(
            "serve --preset with --export: the preset check compares against the "
            "checkpoint's snapshot, and an artifact carries its own config. Escape: "
            "drop --preset (the file IS what it is)."
        )
    if preset:
        _preset_check(preset, _snapshot_run(out_dir, ckpt_name), what="served")
    httpd = build_server(
        out_dir,
        ckpt_name,
        host=host,
        port=port,
        device=device,
        max_tokens_cap=max_tokens_cap,
        temp=temp,
        recurrent=recurrent,
        export=export_file or None,
    )
    state = httpd.state
    console.print(
        Panel(
            f"[bold]{state.ckpt_name}[/] · {param_count(state.model):,} params · {state.device}",
            subtitle=f"http://{host}:{httpd.server_address[1]}/v1 · cap {max_tokens_cap} B"
            f" · temp {temp} · {'recurrent' if state.recurrent else 'plain'} arm",
        )
    )
    serve_forever(httpd)


if __name__ == "__main__":
    main()
