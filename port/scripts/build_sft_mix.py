"""Build one chat corpus from the downloaded SFT sources.

Reads `data_local/sft_raw/*` and writes `{"prompt", "response"}` jsonl that
`dormouse sft --chat-data` accepts. The filters are the ones the research note
argues for (docs/records/research/2026-11-14-sft-corpora.md):

  * short turns only (a 60M byte-level model can learn a reply of a few dozen
    words; a 2000-word essay is noise for it),
  * Cyrillic-first (the target is a Russian assistant),
  * no code, tables or markup (the model is not getting a code corpus here),
  * one example per (prompt, response) and per prompt, because duplicates in SFT
    are a silent weight on whatever the duplicate says,
  * a cap per source, so no single dataset becomes the model's whole personality.

Usage:
    python scripts/build_sft_mix.py            # writes data_local/chat/ru_sft_mix.jsonl
    python scripts/build_sft_mix.py --stats    # per-source survival, no write
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data_local" / "sft_raw"
OUT = Path(__file__).resolve().parents[1] / "data_local" / "chat" / "ru_sft_mix.jsonl"

MAX_PROMPT = 512  # bytes
MAX_RESPONSE = 400  # bytes
MAX_RESPONSE_EN = 512  # English is one byte per char; the same budget is more text
MIN_RESPONSE = 2

#: Things a chat model at this size should not be fed from these sources.
BAD_RESPONSE = re.compile(
    r"```|<html|<table|\{\s*\"|^\s*\|.*\|\s*$|\bdef \w+\(|\bclass \w+\(", re.M
)
#: The Russian lane requires Cyrillic in the answer (a prompt in Latin script
#: still passes: names, code questions). English sources are their own lane.
CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")

#: Per-source cap. 6k each keeps variety while stopping any one set from
#: dominating; the mixture is ~30k rows, which at 1024-byte windows is a few
#: thousand SFT steps for a couple of epochs.
CAP = 6000
#: English gets a smaller share and its own volume: the model is used in Russian
#: first, but a chat model that cannot answer an English question is not a chat
#: model. ~40% of rows.
CAP_EN = 9000


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if path.suffix == ".zst":
        proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        return proc.stdout
    return open(path, encoding="utf-8")


def _jsonl(path: Path) -> Iterator[dict]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _parquet(path: Path) -> Iterator[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    for batch in table.to_batches(max_chunksize=4096):
        yield from batch.to_pylist()


def src_saiga() -> Iterator[tuple[str, str]]:
    for row in _jsonl(RAW / "ru_turbo_saiga" / "ru_turbo_saiga.jsonl.zst"):
        msgs = row.get("messages") or []
        user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        bot = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "bot"), "")
        yield user, bot


def src_gpt4() -> Iterator[tuple[str, str]]:
    for row in _jsonl(RAW / "ru_instruct_gpt4" / "ru_instruct_gpt4.jsonl"):
        prompt = row.get("instruction", "")
        if row.get("input"):
            prompt = f"{prompt}\n{row['input']}"
        yield prompt, row.get("output", "")


def src_instr2() -> Iterator[tuple[str, str]]:
    for row in _jsonl(RAW / "russian_instructions_2" / "dataset.jsonl"):
        yield row.get("question", ""), row.get("answer", "")


def src_dialogues2() -> Iterator[tuple[str, str]]:
    for row in _jsonl(RAW / "russian_dialogues_2" / "dataset.jsonl.gz"):
        turns = [t for t in (row.get("sample") or []) if isinstance(t, str)]
        yield from zip(turns, turns[1:], strict=False)


def src_alpaca_ru() -> Iterator[tuple[str, str]]:
    for path in sorted((RAW / "alpaca-cleaned-ru" / "data").glob("*.parquet")):
        for row in _parquet(path):
            prompt = row.get("instruction", "")
            if row.get("input"):
                prompt = f"{prompt}\n{row['input']}"
            yield prompt, row.get("output", "")


def src_smoltalk() -> Iterator[tuple[str, str]]:
    """SmolTalk: built for sub-1B models, short turns — the English counterpart."""
    for path in sorted((RAW / "smol-smoltalk" / "data").glob("*.parquet")):
        for row in _parquet(path):
            msgs = [m for m in (row.get("messages") or []) if isinstance(m, dict)]
            user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            bot = next(
                (m.get("content", "") for m in reversed(msgs) if m.get("role") == "assistant"), ""
            )
            yield user, bot


def src_oasst2_en() -> Iterator[tuple[str, str]]:
    """The English half of the same human tree, so the two lanes share a voice."""
    path = RAW / "oasst2" / "2023-11-05_oasst2_ready.messages.jsonl.gz"
    texts: dict[str, tuple[str, str]] = {}
    rows = []
    for row in _jsonl(path):
        rows.append(row)
        texts[row["message_id"]] = (row.get("role", ""), row.get("text", ""))
    for row in rows:
        if row.get("role") != "assistant" or row.get("lang") != "en":
            continue
        parent = texts.get(row.get("parent_id") or "")
        if not parent or parent[0] != "prompter":
            continue
        yield parent[1], row["text"]


def src_oasst2() -> Iterator[tuple[str, str]]:
    """Assistant replies whose language is Russian, paired with their parent."""
    path = RAW / "oasst2" / "2023-11-05_oasst2_ready.messages.jsonl.gz"
    texts: dict[str, tuple[str, str]] = {}
    rows = []
    for row in _jsonl(path):
        rows.append(row)
        texts[row["message_id"]] = (row.get("role", ""), row.get("text", ""))
    for row in rows:
        if row.get("role") != "assistant" or row.get("lang") != "ru":
            continue
        parent = texts.get(row.get("parent_id") or "")
        if not parent or parent[0] != "prompter":
            continue
        yield parent[1], row["text"]


#: (name, reader, needs_cyrillic, cap)
SOURCES = [
    ("oasst2_ru", src_oasst2, True, CAP),
    ("saiga", src_saiga, True, CAP),
    ("ru_instruct_gpt4", src_gpt4, True, CAP),
    ("russian_instructions_2", src_instr2, True, CAP),
    ("russian_dialogues_2", src_dialogues2, True, CAP),
    ("alpaca_cleaned_ru", src_alpaca_ru, True, CAP),
    ("smoltalk_en", src_smoltalk, False, CAP_EN),
    ("oasst2_en", src_oasst2_en, False, CAP_EN),
]


def keep(prompt: str, response: str, needs_cyrillic: bool = True) -> bool:
    limit = MAX_RESPONSE if needs_cyrillic else MAX_RESPONSE_EN
    if not (MIN_RESPONSE <= len(response.encode()) <= limit):
        return False
    if not (2 <= len(prompt.encode()) <= MAX_PROMPT):
        return False
    if needs_cyrillic and not CYRILLIC.search(response):
        return False
    if BAD_RESPONSE.search(response):
        return False
    if response.strip() == prompt.strip():
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--stats", action="store_true", help="report survival, write nothing")
    args = ap.parse_args()

    seen_pair: set[str] = set()
    seen_prompt: set[str] = set()
    rows: list[dict] = []
    report: list[tuple[str, int, int]] = []

    for name, fn, needs_cyrillic, cap in SOURCES:
        kept = seen = 0
        try:
            for prompt, response in itertools.islice(fn(), cap * 60):
                seen += 1
                prompt = " ".join((prompt or "").split())
                response = " ".join((response or "").split())
                if not keep(prompt, response, needs_cyrillic):
                    continue
                h = hashlib.blake2b(
                    (prompt + "\x00" + response).encode(), digest_size=16
                ).hexdigest()
                if h in seen_pair:
                    continue
                ph = hashlib.blake2b(prompt.lower().encode(), digest_size=16).hexdigest()
                if ph in seen_prompt:  # one answer per question
                    continue
                seen_pair.add(h)
                seen_prompt.add(ph)
                rows.append({"prompt": prompt, "response": response, "source": name})
                kept += 1
                if kept >= cap:
                    break
        except FileNotFoundError as exc:
            print(f"{name}: missing ({exc.filename})", file=sys.stderr)
        report.append((name, seen, kept))

    print(f"{'source':22s} {'scanned':>8s} {'kept':>6s}")
    for name, seen, kept in report:
        print(f"{name:22s} {seen:8d} {kept:6d}")
    total_bytes = sum(len(r["prompt"].encode()) + len(r["response"].encode()) for r in rows)
    print(f"{'TOTAL':22s} {'':8s} {len(rows):6d}  ({total_bytes / 1e6:.1f} MB of text)")

    if args.stats:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
