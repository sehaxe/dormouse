"""MC task loaders: two real tasks from HuggingFace, one built in code.

Each task is a prompt TEMPLATE plus a file of raw rows. The template is part of
the number (a different context is a different question), so every template is
pinned here, in one place, and the module docstring is the record of it:

* `arc_easy` — `Question: {question}\nAnswer:` + the choice text, one leading
  space per choice. ARC-Easy test, 2 376 items, 4 choices each.
* `hellaswag` — the dataset's own `ctx` (it ends mid-sentence on purpose) + the
  ending appended directly; the canonical continuation protocol. Validation,
  10 042 items, 4 endings each.
* `synth_math` — `Q: {a} {op} {b} = ?\nA:` + ` {n}`, four distinct integers
  around the answer. Built in code, deterministic from `(seed, index)`, no
  network: the harness must be runnable and testable offline, and a model that
  cannot do two-digit arithmetic at 15 bytes of context is not "smart" by any
  reading.

**Downloads** (`download`, the CLI's `bench --download`) go through HuggingFace's
JSON-rows endpoint, `https://datasets-server.huggingface.co/rows`, in pages of
100. That endpoint is public for both datasets (checked 2026-10-18: ARC-Easy
test 2 376 rows, HellaSwag validation 10 042 rows, both HTTP 200 with no token).
The parquet files on the same repos are public too but need `pyarrow`, which
this repo does not depend on and which one 346 KB file does not justify.

A row that cannot be mapped is LOUD, and a missing file is LOUD with the URL and
the escape — never a silent skip. A silently skipped task reads as "the model
failed it", which is the one lie this harness exists to avoid.
"""

from __future__ import annotations

import json
import random
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..instrument import loud

__all__ = ["SYNTH_N", "TASKS", "MCItem", "TaskSpec", "download", "load_task", "task_names"]

#: Where the task files land (the CLI's default `--data-dir`).
DATA_DIR = Path("data_local/bench")

#: The public HF JSON-rows endpoint the downloader pages through.
ROWS_URL = "https://datasets-server.huggingface.co/rows"

#: Rows per request — the endpoint's own cap is 100.
PAGE = 100

#: Seconds between pages. The endpoint rate-limits by BURST (measured 2026-10-18:
#: ~39 pages back to back answer 200, the 40th answers 429; one request per
#: second answers 200 indefinitely), so the downloader paces itself instead of
#: relying on the retry backoff alone.
PAGE_DELAY = 1.0

#: Items the built-in synthetic task has; `--limit` caps them further.
SYNTH_N = 256

#: The offset pool a synthetic item's three distractors are drawn from. Distinct
#: offsets give distinct choices for every answer, including the negative ones
#: (`-` can reach -19), so the pool needs no rejection loop and cannot spin.
SYNTH_OFFSETS = (-2, -1, 1, 2, 3, 10, -10, -3)


@dataclass(frozen=True)
class MCItem:
    """One multiple-choice item: a context, its choices in file order, the label.

    Text, not bytes: the loader is where a task's bytes are decided, the scorer
    encodes with UTF-8 (`str.encode`), and the JSONL row a human reads stays
    readable.
    """

    tid: str
    context: str
    choices: tuple[str, ...]
    label: int

    def __post_init__(self) -> None:
        assert self.choices, f"{self.tid}: an item needs choices"
        assert 0 <= self.label < len(self.choices), (
            f"{self.tid}: label {self.label} outside {len(self.choices)} choices"
        )


@dataclass(frozen=True)
class TaskSpec:
    """A task: its template (in `loader`), its download, and its file name.

    `dataset` empty means the task is built in code and has nothing to fetch.
    """

    name: str
    file: str
    url: str
    dataset: str = ""
    config: str = ""
    split: str = ""
    loader: Callable[[list[dict]], list[MCItem]] | None = None

    @property
    def downloadable(self) -> bool:
        return bool(self.dataset)


# ---------------------------------------------------------------------------
# the two real tasks
# ---------------------------------------------------------------------------


def _arc_items(rows: list[dict]) -> list[MCItem]:
    """ARC rows → items. `answerKey` is a LABEL letter ('A'), not an index."""
    out = []
    for r in rows:
        labels = [str(x) for x in r["choices"]["label"]]
        key = str(r["answerKey"])
        if key not in labels:
            loud(
                f"arc_easy: item {r.get('id')!r} has answerKey={key!r} which is not in its "
                f"choice labels {labels} — the item cannot be scored and is not dropped "
                f"silently. Escape: re-download the task (bench --download --tasks arc_easy)."
            )
        out.append(
            MCItem(
                tid=str(r["id"]),
                context=f"Question: {r['question']}\nAnswer:",
                choices=tuple(" " + str(t) for t in r["choices"]["text"]),
                label=labels.index(key),
            )
        )
    return out


def _hellaswag_items(rows: list[dict]) -> list[MCItem]:
    """HellaSwag rows → items. `ctx` is the full context; endings are appended raw.

    Both the HF rows and the authors' original `hellaswag_val.jsonl` carry the
    same keys (`ctx`, `endings`, `label`), so a manual fetch of either loads
    here. `label` is a string on HF and an int in the original file.
    """
    return [
        MCItem(
            tid=str(r.get("ind", r.get("source_id", ""))),
            context=str(r["ctx"]),
            choices=tuple(str(e) for e in r["endings"]),
            label=int(r["label"]),
        )
        for r in rows
    ]


def synth_math(n: int = SYNTH_N, seed: int = 0) -> list[MCItem]:
    """`n` deterministic arithmetic items — a pure function of `(seed, index)`.

    The distractors are the answer plus three distinct offsets from
    [`SYNTH_OFFSETS`], shuffled per item by the item's own seeded RNG, so the
    correct choice is not always in the same slot and two runs of one seed
    produce byte-identical items (the repo's determinism rule, ADR-0021).
    """
    if n < 0:
        loud(f"synth_math n={n}: a task count cannot be negative")
    items = []
    for i in range(n):
        rng = random.Random(seed * 1_000_003 + i)
        op = "+-*"[i % 3]
        if op == "*":
            a, b = rng.randint(2, 12), rng.randint(2, 12)
        else:
            a, b = rng.randint(0, 19), rng.randint(0, 19)
        answer = {"+": a + b, "-": a - b, "*": a * b}[op]
        offsets = list(SYNTH_OFFSETS)
        rng.shuffle(offsets)
        values = [answer, *(answer + o for o in offsets[:3])]
        rng.shuffle(values)
        items.append(
            MCItem(
                tid=f"synth-{i}",
                context=f"Q: {a} {op} {b} = ?\nA:",
                choices=tuple(f" {v}" for v in values),
                label=values.index(answer),
            )
        )
    return items


#: The registry. `--download` iterates it; `--tasks` names its keys.
TASKS: dict[str, TaskSpec] = {
    "arc_easy": TaskSpec(
        name="arc_easy",
        file="arc_easy_test.jsonl",
        url="https://huggingface.co/datasets/allenai/ai2_arc",
        dataset="allenai/ai2_arc",
        config="ARC-Easy",
        split="test",
        loader=_arc_items,
    ),
    "hellaswag": TaskSpec(
        name="hellaswag",
        file="hellaswag_val.jsonl",
        url="https://huggingface.co/datasets/Rowan/hellaswag",
        dataset="Rowan/hellaswag",
        config="default",
        split="validation",
        loader=_hellaswag_items,
    ),
    "synth_math": TaskSpec(
        name="synth_math",
        file="",
        url="(built in code: dormouse/bench/tasks.py synth_math)",
    ),
}


def task_names() -> list[str]:
    """The registry's keys, in registry order."""
    return list(TASKS)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def load_task(name: str, data_dir: Path | str = DATA_DIR, *, limit: int = 0) -> list[MCItem]:
    """The items of `name`, capped at `limit` (0 = all). LOUD when it cannot.

    A missing file names the URL and the escape: the harness never reports an
    accuracy for a task whose items it did not actually read.
    """
    spec = TASKS.get(name)
    if spec is None:
        loud(f"unknown task {name!r} (known: {', '.join(TASKS)})")
    if not spec.downloadable:
        return synth_math(limit or SYNTH_N)
    path = Path(data_dir) / spec.file
    if not path.is_file():
        loud(
            f"task {name}: {path} is missing. The file is not in the repo — it is a "
            f"download. Escape: `dormouse bench --download --tasks {name}` (source: "
            f"{spec.url}), or point --data-dir at a directory that holds {spec.file}."
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        loud(f"task {name}: {path} is empty — an accuracy over 0 items is not a number.")
    items = spec.loader(rows)
    assert items, f"task {name}: {len(rows)} rows mapped to 0 items"
    return items[:limit] if limit else items


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def _fetch_rows(spec: TaskSpec, *, timeout: float = 60.0, attempts: int = 6) -> list[dict]:
    """Every row of the split, paged through the public JSON-rows endpoint.

    The endpoint rate-limits (a burst of 100-row pages answers HTTP 429), so a
    page is retried with a doubling backoff and the refusal below is only
    reached when the retries are exhausted — a quota is not a task that is
    unavailable, and a benchmark harness that gives up on the first 429 reports
    a network hiccup as a missing task.
    """
    rows: list[dict] = []
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": spec.dataset,
                "config": spec.config,
                "split": spec.split,
                "offset": len(rows),
                "length": PAGE,
            }
        )
        url = f"{ROWS_URL}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "dormouse-bench/0.1"})
        page = None
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https
                    page = json.load(resp)
                break
            except Exception as e:  # noqa: BLE001 - the refusal below is the contract
                if attempt == attempts - 1:
                    loud(
                        f"download {spec.name}: {url} failed after {attempts} attempts "
                        f"({type(e).__name__}: {e}). The task is NOT silently skipped. "
                        f"Escape: fetch it by hand from {spec.url} (the "
                        f"'{spec.config}/{spec.split}' split), convert the rows to JSONL with "
                        f"the keys this loader reads, and write it to {spec.file}."
                    )
                time.sleep(min(2.0 * 2**attempt, 30.0))
        assert page is not None
        got = page.get("rows", [])
        if not got:
            break
        rows.extend(r["row"] for r in got)
        if len(rows) >= page.get("num_rows_total", 0):
            break
        time.sleep(PAGE_DELAY)
    return rows


def download(name: str, data_dir: Path | str = DATA_DIR) -> tuple[Path, int]:
    """Fetch `name`'s rows into `data_dir` as JSONL → `(path, rows)`.

    The raw rows are written as they arrive, so the mapping (`loader`) can change
    without a re-download. A built-in task has nothing to fetch and says so.
    """
    spec = TASKS.get(name)
    if spec is None:
        loud(f"unknown task {name!r} (known: {', '.join(TASKS)})")
    if not spec.downloadable:
        print(f"bench {name}: built in code, nothing to download")
        return Path(data_dir) / spec.file, 0
    rows = _fetch_rows(spec)
    if not rows:
        loud(
            f"download {name}: the endpoint returned 0 rows for "
            f"{spec.dataset}/{spec.config}/{spec.split}. An empty task file is not a "
            f"result. Escape: fetch it by hand from {spec.url}."
        )
    # The loader is the contract check: a fetch that cannot be mapped fails HERE,
    # not an hour later inside the eval loop with a task-shaped hole in it.
    items = spec.loader(rows)
    path = Path(data_dir) / spec.file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    size = path.stat().st_size
    print(f"bench {name}: {len(rows)} rows, {len(items)} items → {path} ({size:,} B)")
    return path, len(rows)
