#!/usr/bin/env python3
"""Build the dormouse SFT chat corpus from the local pretrain mixture.

The SFT loader (`src/dormouse/train/sft.py`) accepts chat data as `.jsonl` rows
of `{"instruction": ..., "response": ...}` and applies its own rule: drop a row
whose instruction+response+4 exceeds `seq_len`, drop a row carrying a sentinel
byte (0x00-0x03 are the template's markers), dedup nothing, and never truncate.
This script turns the three chat-shaped sources of the local mixture into such
files, filtered BEFORE the loader ever sees them:

* `aria_ru.jsonl`     — `rudialog/*.txt`: Russian dialogs, several templates
  (persona-prompted single turns, bare Q/A, FLAN-style task/input/output).
* `aria_en.jsonl`     — `agentic/*.txt` (the Claude-Mythos instruction/response
  traces) + `agentic/*.jsonl` (instruction/response exports).
* `aria_reason.jsonl` — `reasoning/*.txt`: the task, a `think`-tagged thought
  and the final answer; the thought rides along only when it fits the budget.

The hard filters (the SFT record is `seq_len + 1` = 513 bytes for seq_len 512;
the template's markers eat 4):

* instruction + response <= 450 bytes combined — dropped and counted otherwise;
* bytes 0x00-0x03 and `<|...|>` special-token strings are stripped from both
  fields (the loader refuses sentinels LOUD; a stripped row is a kept row);
* dedup by sha1 of `(instruction, response)`;
* either half shorter than 2 bytes is dropped as dirty.

Determinism: the file order is a pure function of `--seed`, files are streamed
(a 6 GB jsonl is never loaded), and each output stops at `--cap` kept pairs.
`--max-seconds` is a safety net, not the bound — if it fires, the run is
reported as truncated.

    uv run python port/scripts/parse_aria_sft.py
    uv run python port/scripts/parse_aria_sft.py --source en --cap 5000
    uv run python port/scripts/parse_aria_sft.py --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

#: The mixture root the sources live under (beside the pretrain corpus).
MIX_DEFAULT = Path("/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/mix")

#: Where the three outputs go — `port/sft_pairs/`, beside the hand-written `.qa`.
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "sft_pairs"

#: instruction + response, in bytes, must stay under this. The loader's own
#: bound is `seq_len` (508 at seq_len 512); 450 leaves the template room and is
#: the task's hard number, so every row written here loads without a drop.
LIMIT = 450

SEED_DEFAULT = 42
CAP_DEFAULT = 200_000
SECONDS_DEFAULT = 1500  # 25 min: the task's 30 min budget minus slack

_CTRL = {ord(c): None for c in "\x00\x01\x02\x03"}
_SPECIAL = re.compile(r"<\|[^|\n]{0,80}\|>")
_ROLE_WORDS = {"system", "user", "assistant", "human", "ai"}
#: The export's error stub, e.g. `[HATA: Connection error.]` — never an answer.
_JUNK_RESP = re.compile(r"^\[HATA[: ].*\]$", re.I)
#: The Mythos traces mark every response with this opener.
MYTHOS = "Drawing from the autonomous"
#: The reasoning traces bracket their thought with this tag pair. Spelled with
#: escapes so no editor or shell can silently rewrite the angle brackets.
THINK_OPEN = "\x3cthink\x3e"
THINK_CLOSE = "\x3c/think\x3e"
#: The second reasoning template: explicit thought/solution section markers.
BOT = "\x3c|begin_of_thought|\x3e"
EOT = "\x3c|end_of_thought|\x3e"
BOS = "\x3c|begin_of_solution|\x3e"
EOS = "\x3c|end_of_solution|\x3e"


def clean(s: str) -> str:
    """Normalize and de-contaminate one field for the SFT template.

    Newlines normalized, BOM dropped, bytes 0x00-0x03 deleted (they are the
    template's sentinels — the loader LOUD-refuses them), `<|...|>`
    special-token strings deleted, whitespace trimmed. Everything else is
    kept verbatim.
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    s = s.translate(_CTRL)
    s = _SPECIAL.sub("", s)
    return s.strip()


# ---------------------------------------------------------------------------
# rudialog: paragraph-pair scan
# ---------------------------------------------------------------------------

#: Imperative task verbs (prefixes, lowercase). A paragraph opening with one is
#: a user turn even without a `?` — the FLAN-style templates rely on it.
_IMPER = (
    "напиш", "созда", "сгенериру", "объясн", "расскаж", "перевед", "прочит",
    "определ", "выдел", "посчита", "состав", "исправ", "провер", "найд",
    "придума", "продолж", "законч", "ответь", "помог", "покаж", "оцен", "назов",
    "перечисли", "докаж", "реши", "вычисл", "подскаж", "дай ", "задай", "опиши",
    "сравни", "уточни", "выбери", "сформулируй", "перефразируй", "сократи",
    "раздели", "заполни", "перепиши", "законспектируй", "порекомендуй",
    "предложи", "сочини", "замени", "измени", "добавь", "вставь", "удали",
    "нарисуй", "спланируй", "проанализируй", "классифицируй", "поясни",
    "разъясни", "оформи", "выпиши", "отредактируй", "начни",
)
#: Words that mark a paragraph as a persona/system prompt rather than a turn.
_PERSONA_KW = (
    "ассистент", "помощник", "языковая модель", "нейросет", "gpt", "claude",
    "сайга", "коуч", "отвеча", "помога", "должен ответить", "должна ответить",
    "эксперт", "специалист", "консультант", "тренер", "бот",
)


def _paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [p.strip() for p in re.split(r"\n{2,}", text)]


def _is_persona(p: str) -> bool:
    low = p.lower()
    if not low.startswith(("ты", "вы")) or p.endswith("?") or len(p) > 400:
        return False
    return any(k in low for k in _PERSONA_KW)


def _is_question(p: str) -> bool:
    if len(p) > 2000:
        return False
    return p.endswith("?") or p.lower().startswith(_IMPER)


def parse_rudialog(text: str) -> Iterator[tuple[str, str]]:
    """Yield (question, first-answer-paragraph) pairs from one dialog file.

    The walk keeps one pending question; a non-question paragraph closes it.
    A persona paragraph forces the NEXT paragraph to be a user turn (the
    ShareGPT-shaped files carry statements, not questions, there). A question
    stem ending in `:` absorbs the next paragraph as its input (the FLAN
    template's task/input split) and answers with the paragraph after it.
    """
    paras = _paragraphs(text)
    n = len(paras)
    q: str | None = None
    force = False
    i = 0
    while i < n:
        p = paras[i]
        if not p:
            i += 1
            continue
        if _is_persona(p):
            q = None
            force = True
            i += 1
            continue
        candidate = force or _is_question(p)
        force = False
        if candidate:
            if p.endswith(":") and i + 2 < n and paras[i + 1] and paras[i + 2]:
                yield p + "\n" + paras[i + 1], paras[i + 2]
                q = None
                i += 3
                continue
            q = p
            i += 1
            continue
        if q is not None:
            yield q, p
            q = None
        i += 1


# ---------------------------------------------------------------------------
# agentic: Mythos .txt traces + instruction/response .jsonl exports
# ---------------------------------------------------------------------------


def parse_mythos(text: str) -> Iterator[tuple[str, str]]:
    """Yield (instruction, response) from one Claude-Mythos trace file.

    Every response opens with [`MYTHOS`]; the instruction is the LAST LINE of
    the text between two openers (the file head for the first record). The
    response is everything between its opener and the next instruction.
    """
    pos = [m.start() for m in re.finditer(re.escape(MYTHOS), text)]
    if not pos:
        return
    instrs: list[str] = []
    prev = 0
    for k, p in enumerate(pos):
        chunk = text[prev:p]
        prev = p + len(MYTHOS)
        instrs.append(chunk.strip() if k == 0 else chunk.rstrip("\n").split("\n")[-1].strip())
    for k, p in enumerate(pos):
        if k + 1 < len(pos):
            end = pos[k + 1] - 2 - len(instrs[k + 1])
            resp = text[p:end] if end > p else ""
        else:
            resp = text[p:]
        yield instrs[k], resp.strip()


def parse_jsonl_obj(obj: object) -> tuple[str, str] | None:
    """One jsonl row -> pair, or None for a schema this corpus does not carry.

    Two shapes: the export rows (`instruction`/`response`) and the
    conversation rows (`user`/`assistant`). Tool-call `messages` rows and the
    graph/incident rows are not pair-shaped and are skipped (counted).
    """
    if not isinstance(obj, dict):
        return None
    if "instruction" in obj and "response" in obj:
        i, a = obj["instruction"], obj["response"]
    elif "user" in obj and "assistant" in obj:
        i, a = obj["user"], obj["assistant"]
    else:
        return None
    if isinstance(i, str) and isinstance(a, str):
        return i, a
    return None


def jsonl_schema_ok(first_line: str) -> bool:
    """Does this jsonl file's first row carry a pair-shaped schema at all?"""
    try:
        obj = json.loads(first_line)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    return ("instruction" in obj and "response" in obj) or (
        "user" in obj and "assistant" in obj
    )


# ---------------------------------------------------------------------------
# reasoning: instruction + thought + final answer
# ---------------------------------------------------------------------------


def parse_reasoning_think(text: str) -> Iterator[tuple[str, str, str]]:
    """The tag-pair variant: `instr`, think-tagged thought, then the answer."""
    opens = [m.start() for m in re.finditer(re.escape(THINK_OPEN), text)]
    closes = [m.start() for m in re.finditer(re.escape(THINK_CLOSE), text)]
    n = min(len(opens), len(closes))
    if n == 0:
        return
    for k in range(n):  # the two marker streams must interleave strictly
        if opens[k] > closes[k] or (k > 0 and closes[k - 1] > opens[k]):
            return
    instrs: list[str] = []
    thoughts: list[str] = []
    answers: list[str] = []
    prev = 0
    for k in range(n):
        seg = text[prev:opens[k]].strip("\n")
        if k == 0:
            instrs.append(seg)
        else:
            ans, instr = seg.rsplit("\n", 1) if "\n" in seg else ("", seg)
            answers.append(ans.strip())
            instrs.append(instr.strip())
        thoughts.append(text[opens[k] + len(THINK_OPEN):closes[k]].strip())
        prev = closes[k] + len(THINK_CLOSE)
    answers.append(text[prev:].strip("\n"))
    for instr, thought, ans in zip(instrs, thoughts, answers, strict=False):
        yield instr, thought, ans


def parse_reasoning_bot(text: str) -> Iterator[tuple[str, str, str]]:
    """The section-marker variant: instruction, thought, solution sections.

    Layout: `instr`, `<|begin_of_thought|>`, thought, `<|end_of_thought|>`,
    `<|begin_of_solution|>`, answer, `<|end_of_solution|>`, next instruction.
    """
    b = [m.start() for m in re.finditer(re.escape(BOT), text)]
    e = [m.start() for m in re.finditer(re.escape(EOT), text)]
    s = [m.start() for m in re.finditer(re.escape(BOS), text)]
    z = [m.start() for m in re.finditer(re.escape(EOS), text)]
    n = min(len(b), len(e), len(s), len(z))
    if n == 0:
        return
    prev = 0
    for k in range(n):
        if not b[k] < e[k] < s[k] < z[k] or (k > 0 and z[k - 1] > b[k]):
            return
        yield (
            text[prev:b[k]].strip(),
            text[b[k] + len(BOT):e[k]].strip(),
            text[s[k] + len(BOS):z[k]].strip(),
        )
        prev = z[k] + len(EOS)


def parse_reasoning(text: str) -> Iterator[tuple[str, str, str]]:
    """Yield (instruction, thought, final_answer) from one reasoning file.

    Two templates live in the source: the tag-pair one and the section-marker
    one. Both return the thought separately so the sink can drop it when it
    does not fit the budget.
    """
    if BOT in text:
        yield from parse_reasoning_bot(text)
    else:
        yield from parse_reasoning_think(text)


# ---------------------------------------------------------------------------
# the sink: filters, dedup, counters, samples
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    scanned: int = 0        # raw pairs the source parser produced
    kept: int = 0
    dropped_long: int = 0   # instruction + response over LIMIT bytes
    dropped_dirty: int = 0  # a half under 2 bytes / role-word / repr-junk
    duplicates: int = 0     # same sha1(instruction, response) seen before
    cleaned: int = 0        # rows that had sentinels / <|...|> removed
    files_read: int = 0
    files_skipped: int = 0
    lines: int = 0
    seconds: float = 0.0


class Sink:
    """One output file: the hard filters, the dedup set, the counters."""

    def __init__(self, path: Path, cap: int, limit: int = LIMIT) -> None:
        self.path = path
        self.cap = cap
        self.limit = limit
        self.stats = Stats()
        self.seen: set[str] = set()
        self.samples: list[tuple[str, str]] = []
        self.fh = path.open("w", encoding="utf-8")

    def full(self) -> bool:
        return self.stats.kept >= self.cap

    def offer(self, instruction: str, answer: str) -> None:
        s = self.stats
        s.scanned += 1
        i, a = clean(instruction), clean(answer)
        if i != instruction or a != answer:
            s.cleaned += 1
        ib, ab = i.encode(), a.encode()
        if (
            len(ib) < 2
            or len(ab) < 2
            or i.lower() in _ROLE_WORDS
            or i.startswith(("{'messages'", '{"messages"'))
            or _JUNK_RESP.match(a)  # the export's error stub, e.g. [HATA: ...]
        ):
            s.dropped_dirty += 1
            return
        if len(ib) + len(ab) > self.limit:
            s.dropped_long += 1
            return
        h = hashlib.sha1(ib + b"\x00" + ab).hexdigest()
        if h in self.seen:
            s.duplicates += 1
            return
        self.seen.add(h)
        s.kept += 1
        if len(self.samples) < 3:
            self.samples.append((i, a))
        self.fh.write(json.dumps({"instruction": i, "response": a}, ensure_ascii=False))
        self.fh.write("\n")

    def close(self) -> int:
        self.fh.close()
        return self.path.stat().st_size


# ---------------------------------------------------------------------------
# file order: seeded shuffle, then round-robin across families
# ---------------------------------------------------------------------------


def _family(path: Path) -> str:
    name = re.sub(r"_\d{4,}\.", ".", path.name)      # agentic_00001.txt -> agentic.txt
    return re.sub(r"_shard-\d+", "", name)           # ..._shard-00007.jsonl


def order_files(files: list[Path], seed: int) -> list[Path]:
    """Deterministic, corpus-spanning read order.

    Sorted -> seeded shuffle (sampling within a family) -> round-robin across
    families (the scan touches every family before it touches any family
    twice), so a cap hit early still yields a mixed corpus.
    """
    files = sorted(files)
    random.Random(seed).shuffle(files)
    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(_family(f), []).append(f)
    out: list[Path] = []
    k = 0
    while True:
        added = 0
        for fam in sorted(groups):
            lst = groups[fam]
            if k < len(lst):
                out.append(lst[k])
                added += 1
        if added == 0:
            return out
        k += 1


# ---------------------------------------------------------------------------
# the sources and their runners
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _has_mythos(path: Path) -> bool:
    """Cheap probe: the Mythos opener sits in the first record of a trace file."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        return MYTHOS in fh.read(4096)


def run_txt_source(paths: list[Path], parse: Callable[[str], Iterator], sink: Sink,
                   max_seconds: float, probe: Callable[[Path], bool] | None = None) -> None:
    t0 = time.time()
    for path in paths:
        if sink.full() or time.time() - t0 > max_seconds:
            break
        if probe is not None and not probe(path):
            sink.stats.files_skipped += 1
            continue
        for pair in parse(_read_text(path)):
            sink.offer(*pair)
            if sink.full():
                break
        sink.stats.files_read += 1
    sink.stats.seconds = time.time() - t0


def run_jsonl_source(paths: list[Path], sink: Sink, max_seconds: float) -> None:
    t0 = time.time()
    for path in paths:
        if sink.full() or time.time() - t0 > max_seconds:
            break
        with path.open(encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
            if not first:
                continue
            if not jsonl_schema_ok(first):
                sink.stats.files_skipped += 1
                continue
            for line in fh:
                sink.stats.lines += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pair = parse_jsonl_obj(obj)
                if pair is not None:
                    sink.offer(*pair)
                if sink.full():
                    break
                if sink.stats.lines % 50_000 == 0 and time.time() - t0 > max_seconds:
                    break
        sink.stats.files_read += 1
    sink.stats.seconds = time.time() - t0


def run_reasoning_source(paths: list[Path], sink: Sink, max_seconds: float) -> None:
    """Like `run_txt_source`, but the thought rides along only when it fits."""
    t0 = time.time()
    for path in paths:
        if sink.full() or time.time() - t0 > max_seconds:
            break
        for instr, thought, ans in parse_reasoning(_read_text(path)):
            # The char count is a lower bound on the byte count, so it filters
            # out the (usually huge) thoughts before the expensive clean().
            if thought and len(instr) + len(thought) + len(ans) + 2 <= sink.limit:
                if len(clean(instr).encode()) + len(clean(thought).encode()) \
                        + len(clean(ans).encode()) + 2 <= sink.limit:
                    sink.offer(instr, thought + "\n\n" + ans)
                    if sink.full():
                        break
                    continue
            sink.offer(instr, ans)
            if sink.full():
                break
        sink.stats.files_read += 1
    sink.stats.seconds = time.time() - t0


# ---------------------------------------------------------------------------
# reporting and verification
# ---------------------------------------------------------------------------


def report(name: str, sink: Sink, size: int) -> None:
    s = sink.stats
    print(f"\n=== {sink.path.name}  (source: {name}) ===")
    print(f"  files read {s.files_read}  skipped {s.files_skipped}  jsonl lines {s.lines}"
          f"  ({s.seconds:.1f} s)")
    print(f"  scanned {s.scanned}  kept {s.kept}  dropped_long {s.dropped_long}"
          f"  dropped_dirty {s.dropped_dirty}  duplicates {s.duplicates}"
          f"  cleaned {s.cleaned}")
    for n, (i, a) in enumerate(sink.samples, 1):
        print(f"  sample {n}: instruction={i[:200]!r}")
        print(f"            response={a[:200]!r}")
    print(f"  bytes {size}")


def verify_output(path: Path) -> int:
    """Re-read one output as the SFT loader would: every row must pass.

    Same checks as `sft._chat_pairs`: valid JSON, both halves non-empty,
    `len(i)+len(a)+4 <= 512` (seq_len 512), no sentinel byte in either half.
    """
    rows = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            row = json.loads(line)
            i, a = row["instruction"], row["response"]
            raw_i, raw_a = i.encode(), a.encode()
            assert len(raw_i) + len(raw_a) + 4 <= 512, f"{path}:{lineno}: too long"
            assert all(b not in raw_i for b in (0, 1, 2, 3)), \
                f"{path}:{lineno}: sentinel in instruction"
            assert all(b not in raw_a for b in (0, 1, 2, 3)), \
                f"{path}:{lineno}: sentinel in response"
            assert len(raw_i) >= 2 and len(raw_a) >= 2, f"{path}:{lineno}: too short"
            rows += 1
    return rows


# ---------------------------------------------------------------------------
# selftest: the one runnable check for the parsers and the sink
# ---------------------------------------------------------------------------


def selftest() -> None:
    assert clean("a\x00b\x01c\x02d\x03e") == "abcde"
    assert clean("x<|im_start|>system<|im_end|>y") == "xsystemy"
    assert clean("\ufeffhi\r\nthere ") == "hi\nthere"

    ru = (
        "Ты — Claude 3.5 Sonnet, последняя языковая модель.\n\n"
        "Как дела?\n\nХорошо, спасибо!\n\n"
        "Вы - ассистент искусственного интеллекта.\n\n"
        "Напишите письмо по теме:\n\nАктивность\n\nДорогая команда, привет!\n"
    )
    pairs = list(parse_rudialog(ru))
    assert pairs[0] == ("Как дела?", "Хорошо, спасибо!"), pairs
    assert pairs[1] == ("Напишите письмо по теме:\nАктивность", "Дорогая команда, привет!"), pairs

    mythos = (
        "First task.\n\nDrawing from the autonomous, x\nFirst answer.*\n"
        "Second task.\n\nDrawing from the autonomous, x\nSecond answer."
    )
    assert list(parse_mythos(mythos)) == [
        ("First task.", "Drawing from the autonomous, x\nFirst answer.*"),
        ("Second task.", "Drawing from the autonomous, x\nSecond answer."),
    ]

    reason = (
        "Task one?\n\n" + THINK_OPEN + "\nthink one\n" + THINK_CLOSE + "\n\nAnswer one.\n"
        "Task two?\n\n" + THINK_OPEN + "\nthink two\n" + THINK_CLOSE + "\n\nAnswer two."
    )
    got = list(parse_reasoning(reason))
    assert got == [
        ("Task one?", "think one", "Answer one."),
        ("Task two?", "think two", "Answer two."),
    ], got

    bot = (
        "Task three?\n\n" + BOT + "\nthink three\n" + EOT + "\n\n" + BOS
        + "\nAnswer three.\n" + EOS + "\nTask four?\n\n" + BOT + "\nthink four\n"
        + EOT + "\n\n" + BOS + "\nAnswer four.\n" + EOS
    )
    got = list(parse_reasoning(bot))
    assert got == [
        ("Task three?", "think three", "Answer three."),
        ("Task four?", "think four", "Answer four."),
    ], got

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        sink = Sink(Path(td) / "t.jsonl", cap=10)
        sink.offer("Q1", "A1")
        sink.offer("Q1", "A1")                      # duplicate
        sink.offer("Q2", "x" * 500)                 # long
        sink.offer("Q3", "")                        # dirty
        sink.offer("Q\x00\x01x", "A<|endoftext|>2")  # cleaned, kept
        sink.close()
        s = sink.stats
        assert (s.kept, s.duplicates, s.dropped_long, s.dropped_dirty, s.cleaned) == \
               (2, 1, 1, 1, 1), s
        assert verify_output(Path(td) / "t.jsonl") == 2
    print("selftest ok")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mix-root", type=Path, default=MIX_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--source", action="append", choices=["ru", "en", "reason"],
                    help="build only these outputs (default: all three)")
    ap.add_argument("--cap", type=int, default=CAP_DEFAULT,
                    help=f"kept pairs per output (default {CAP_DEFAULT})")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--max-seconds", type=float, default=SECONDS_DEFAULT,
                    help="safety net per source (default 1500 s)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0
    if not args.mix_root.is_dir():
        print(f"mix root not found: {args.mix_root}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = args.source or ["ru", "en", "reason"]
    t0 = time.time()

    if "ru" in wanted:
        files = sorted((args.mix_root / "rudialog").glob("rudialog_*.txt"))
        sink = Sink(args.out_dir / "aria_ru.jsonl", args.cap)
        run_txt_source(order_files(files, args.seed), parse_rudialog, sink, args.max_seconds)
        size = sink.close()
        report("rudialog", sink, size)
        print(f"  verified rows: {verify_output(sink.path)}")

    if "en" in wanted:
        d = args.mix_root / "agentic"
        txts = sorted(d.glob("agentic_*.txt"))
        jsonls = sorted(d.glob("*.jsonl"))
        sink = Sink(args.out_dir / "aria_en.jsonl", args.cap)
        run_txt_source(order_files(txts, args.seed), parse_mythos, sink, args.max_seconds,
                       probe=_has_mythos)
        print(f"  [agentic txt phase] files read {sink.stats.files_read}"
              f" skipped {sink.stats.files_skipped} scanned {sink.stats.scanned}"
              f" kept {sink.stats.kept}")
        run_jsonl_source(order_files(jsonls, args.seed), sink, args.max_seconds)
        size = sink.close()
        report("agentic", sink, size)
        print(f"  verified rows: {verify_output(sink.path)}")

    if "reason" in wanted:
        files = sorted((args.mix_root / "reasoning").glob("reasoning_*.txt"))
        sink = Sink(args.out_dir / "aria_reason.jsonl", args.cap)
        run_reasoning_source(order_files(files, args.seed), sink, args.max_seconds)
        size = sink.close()
        report("reasoning", sink, size)
        print(f"  verified rows: {verify_output(sink.path)}")

    print(f"\n--- done in {time.time() - t0:.1f} s ---")
    for name in ("aria_ru.jsonl", "aria_en.jsonl", "aria_reason.jsonl"):
        p = args.out_dir / name
        if p.exists():
            print(f"  {name:18s} {p.stat().st_size:>12,} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
