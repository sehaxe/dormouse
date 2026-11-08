"""The run registry: one `run.json` per training invocation.

The trainer's JSONL telemetry answers "how did the loss move". It cannot answer
"which run produced this number, with what recipe, on what bytes": a log can be
appended across resumes (see `runlog.py`), a checkpoint directory can be moved,
and a preset NAME is not a recipe because the file behind it can be edited. The
registry is that answer, one small record per invocation:

    <runs-dir>/<run_id>/run.json

`<runs-dir>` is the run's `out_dir`/`runs` for a library caller and `logs/runs`
for the CLI (the CLI default). The record is written twice and only twice: once
before the first step (`status = "running"`) and once when the loop returns or
raises (`done` / `failed` + the error string). It never touches a training step.

The module imports only stdlib and pulls torch in `host_info` lazily, so
`dormouse runs list|show` reads JSON and formats a table without a tensor. The
schema is documented in `port/docs/guides/runs.md`; `runs.py` is the writer and
the reader, one definition.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the config module pulls torch through `instrument`
    from .config import RunConfig

#: Bumped when a field's meaning changes. Readers refuse a newer record rather
#: than silently dropping a field they do not know.
SCHEMA = 1

#: Where `dormouse train` writes when `--runs-dir` is not given.
DEFAULT_RUNS_DIR = "logs/runs"

#: `configs/` beside this checkout (`port/src/dormouse/runs.py`), and the wheel's
#: own copy. The search order is `config.load_preset`'s: cwd, checkout, wheel.
_CHECKOUT_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_PACKAGE_CONFIGS = Path(__file__).resolve().parent / "configs"


def _loud(msg: str) -> None:
    """Raise the repo's `LoudError` without importing `instrument` (and torch)."""
    from .instrument import loud

    loud(msg)


def _iso(epoch: float | None = None) -> str:
    when = time.time() if epoch is None else epoch
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))


def git_sha() -> str | None:
    """`HEAD` at write time, or None off a git checkout — provenance, not a gate."""
    root = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def _slug(text: str) -> str:
    """A filesystem-safe piece of a run id: letters, digits, dot, dash."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text or "run").strip("-")[:40] or "run"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preset_path(source: str) -> Path | None:
    """The file behind `source` (a preset NAME or a path), or None if unknown.

    A hand-built `RunConfig` may carry a label that is not a preset file; that
    is recorded as `path: null`, never invented.
    """
    if not source:
        return None
    if "/" in source or source.endswith(".toml"):
        path = Path(source).expanduser()
        return path if path.is_file() else None
    name = source.removesuffix(".toml")
    for path in (
        Path(f"configs/{name}.toml"),
        _CHECKOUT_CONFIGS / f"{name}.toml",
        _PACKAGE_CONFIGS / f"{name}.toml",
    ):
        if path.is_file():
            return path
    return None


def preset_identity(source: str) -> dict[str, Any]:
    """`{source, path, sha256}` — the exact recipe bytes the run resolved from.

    The hash is the point: `--preset small` names a file that can be edited
    between runs, so the name alone cannot prove two runs used one recipe.
    """
    path = preset_path(source)
    if path is None:
        return {"source": source, "path": None, "sha256": None}
    return {"source": source, "path": str(path), "sha256": _sha256(path)}


def _tree_bytes(path: Path) -> int | None:
    """A file's size, or a directory's total file bytes; None if unreadable."""
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return None
    return None


def data_sources(cfg) -> list[dict[str, Any]]:
    """Every byte source the run names, with its size (a missing one is `null`).

    `mix_plan` is itself a source of the mix (the weights and stage boundaries
    decide which bytes are read when), so the plan file is a row and each
    `[sources]` entry is another. A directory source is summed; its per-shard
    identity is the mixture manifest's job, not this index's.
    """
    rows: list[dict[str, Any]] = []

    def add(role: str, raw: str) -> None:
        if not raw:
            return
        path = Path(raw).expanduser()
        rows.append({"role": role, "path": str(path), "bytes": _tree_bytes(path)})

    if cfg.mix_plan:
        add("mix_plan", cfg.mix_plan)
        try:
            plan = tomllib.loads(Path(cfg.mix_plan).expanduser().read_text())
            base = Path(cfg.mix_plan).expanduser().parent
            for name in sorted(plan.get("sources") or {}):
                path = Path(plan["sources"][name]).expanduser()
                if not path.is_absolute():
                    path = base / path
                rows.append({"role": f"mix:{name}", "path": str(path), "bytes": _tree_bytes(path)})
        except (OSError, tomllib.TOMLDecodeError) as e:
            rows[0]["error"] = f"{type(e).__name__}: {e}"
    add("train", cfg.data)
    add("eval", cfg.eval_data)
    return rows


def host_info(device=None) -> dict[str, Any]:
    """The host that produced the number: GPU name, torch and Python versions.

    `device=None` is the trainer's CPU path, so a CPU run never asks CUDA for a
    device name. The import is here, not at module import: `runs list|show` is
    a JSON read and must not pay torch.
    """
    import platform

    import torch

    dev = "cpu" if device is None else str(device)
    gpu = "cpu"
    if device is not None:
        want = torch.device(device)
        if want.type == "cuda" and torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(want.index or 0)
    return {
        "gpu": gpu,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": dev,
    }


def _new_run_id(root: Path, ckpt_name: str, now: float) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
    base = f"{stamp}-{_slug(ckpt_name)}"
    for n in range(1, 101):
        run_id = base if n == 1 else f"{base}-{n}"
        try:
            (root / run_id).mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return run_id
    _loud(
        f"{root}: 100 runs share the id prefix {base}. Escape: move the old run "
        f"directories, or wait a second (the id is a UTC timestamp)."
    )
    raise AssertionError("unreachable")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@dataclass
class RunRecord:
    """The live record held by `train()`: one file, two writes."""

    path: Path
    data: dict[str, Any]
    started: float

    def _write(self) -> None:
        _atomic_write(
            self.path, json.dumps(self.data, indent=2, sort_keys=True, default=str) + "\n"
        )

    def finish(
        self,
        status: str,
        *,
        eval_line: str | None = None,
        final_bpb: float | None = None,
        error: str | None = None,
    ) -> None:
        """Rewrite the record as `done` or `failed`, always with a duration."""
        assert status in ("done", "failed"), status
        now = time.time()
        self.data.update(
            status=status,
            finished=_iso(now),
            duration_s=round(now - self.started, 3),
            eval_line=eval_line,
            final_bpb=final_bpb,
            error=error,
        )
        self._write()


def start(
    run: RunConfig,
    *,
    model=None,
    runs_dir: str | Path | None = None,
    device=None,
) -> RunRecord:
    """Create `<root>/<run_id>/run.json` with `status = "running"` and return it.

    `runs_dir=None` puts the registry under the run's own `out_dir` (the
    artifacts keep their provenance when they move); the CLI passes its
    `--runs-dir` (default `logs/runs`) so many runs share one index.
    """
    cfg = run.train
    root = Path(runs_dir).expanduser() if runs_dir is not None else Path(cfg.out_dir) / "runs"
    root.mkdir(parents=True, exist_ok=True)
    now = time.time()
    run_id = _new_run_id(root, cfg.ckpt_name, now)
    data = data_sources(cfg)
    params = None if model is None else sum(int(p.numel()) for p in model.parameters())
    record = {
        "schema": SCHEMA,
        "run_id": run_id,
        "status": "running",
        "timestamp": _iso(now),
        "finished": None,
        "duration_s": None,
        "git_sha": git_sha(),
        "preset": preset_identity(run.source),
        "config": {
            "model": dataclasses.asdict(run.model),
            "train": dataclasses.asdict(cfg),
        },
        "params": params,
        # The byte BUDGET the run asked for (`steps x batch x seq_len x
        # grad_accum`), not the corpus size: two runs on the same corpus with
        # different step counts saw different bytes and are different numbers.
        "train_bytes": cfg.steps * cfg.batch * cfg.seq_len * cfg.grad_accum,
        "data": data,
        "data_bytes": sum(r["bytes"] or 0 for r in data if r["role"] != "mix_plan"),
        "host": host_info(device),
        "eval_line": None,
        "final_bpb": None,
        "error": None,
    }
    record_obj = RunRecord(root / run_id / "run.json", record, now)
    record_obj._write()
    return record_obj


def _when(record: dict[str, Any]) -> str:
    """The record's start time (`started` is the pre-rename spelling)."""
    return str(record.get("timestamp") or record.get("started") or "")


def _load_record(path: Path) -> dict[str, Any]:
    """One run.json, validated enough that a table cannot silently misread it."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _loud(f"{path}: not a readable run record ({e}). Escape: fix or remove the file.")
    if not isinstance(record, dict) or not record.get("run_id"):
        _loud(f"{path}: not a run record (no `run_id`). Escape: fix or remove the file.")
    try:
        schema = int(record.get("schema", SCHEMA))
    except (TypeError, ValueError):
        _loud(f"{path}: schema {record.get('schema')!r} is not a number. Escape: fix the file.")
    if schema > SCHEMA:
        _loud(
            f"{path}: schema {schema} is newer than this build ({SCHEMA}). "
            f"Escape: run the build that wrote it, or upgrade."
        )
    return record


def read_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Every readable `run.json` under `runs_dir`, newest first.

    A missing root is an EMPTY registry, not an error: a fresh checkout has no
    runs, and `dormouse runs list` should say so. A malformed record is LOUD —
    a table that silently drops one run is the defect this registry exists to
    prevent.
    """
    root = Path(runs_dir).expanduser()
    if not root.is_dir():
        return []
    records = [_load_record(path) for path in sorted(root.glob("*/run.json"))]
    records.sort(key=lambda r: (_when(r), str(r.get("run_id") or "")), reverse=True)
    return records


def read_run(runs_dir: str | Path, run_id: str) -> dict[str, Any]:
    """One record by id. A traversal-shaped id is refused, not resolved."""
    if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        _loud(f"run id {run_id!r}: an id is one directory name, not a path.")
    path = Path(runs_dir).expanduser() / run_id / "run.json"
    if not path.is_file():
        _loud(
            f"no run {run_id!r} under {Path(runs_dir)}. Escape: `dormouse runs "
            f"list --runs-dir {Path(runs_dir)}` for the ids that exist."
        )
    return _load_record(path)


def table_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The `runs list` columns, one dict per record (the CLI only formats)."""
    rows = []
    for r in records:
        rows.append(
            {
                "id": r.get("run_id", "?"),
                "date": _when(r)[:19].replace("T", " "),
                "params": r.get("params"),
                "bytes": r.get("train_bytes"),
                "bpb": r.get("final_bpb"),
                "status": r.get("status") or "?",
                "duration": r.get("duration_s"),
            }
        )
    return rows


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "-"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
