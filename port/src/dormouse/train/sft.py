"""The SFT minimum: a seeded instruction corpus in bytes, prompt-masked CE, a loop.

Pretraining teaches the byte statistics of a corpus; it does not teach the model
that a request deserves an answer. This module is the smallest honest thing that
does: a corpus of `(instruction, answer)` pairs rendered in ONE canonical byte
template, and a loop whose cross-entropy counts only the answer's bytes.

**The template, and why the markers are 0x01/0x02/0x03.** The vocabulary is the
256 bytes, so a special token has to be a byte the task text cannot contain::

    \\x01 {instruction} \\n \\x02 {answer} \\x03 \\x00 \\x00 ...

`0x01` opens the user turn, `0x02` the assistant turn, `0x03` closes the answer,
`0x00` pads the record to the window. Every byte of every instruction and answer
is asserted marker-free at render time ([`render`]) — and the assert can only
pass because the sources are generated text (the seeded tasks of `post/tasks.py`,
the hand-written `GENERIC_QA`, the generated `_ru_pairs` bank, and the distill
JSONL rows, which are pre-filtered for sentinels and for one-window fits before
they enter the build), never arbitrary user bytes. `0x03` and
`0x00` exist because a fixed-size window needs an answer END: without them the
only way to keep a window single-example is to train on padding.

**One window, one example, and how that is guaranteed.** Every record is written
as exactly `seq_len + 1` bytes (the rendered example, then `0x00` padding), and
the file is a concatenation of records. `ByteStream` reads `batch x (seq_len+1)`
bytes and splits them into rows, so every row starts on a record boundary — as
long as `seq_len` is the one the corpus was BUILT with. That is checked at every
step, not assumed: [`sft_mask`] refuses a window with zero or two `0x02` markers
and a window whose tail after `0x03` is not padding, which is what a `seq_len`
mismatch looks like.

**What is reused vs copied.** Reused by import: `ByteStream` (the byte stream
contract), `wsd_lr` (the schedule — WSD, the same one `train()` runs, rather than
a second schedule to keep honest), `warmup_steps`, `MixOptimizer` + `clip_grads`,
`save_ckpt`/`load_ckpt`/`load_model_state`, `kda_field` and `Counters`. **Copied
verbatim from `train()` (`loop.py`):** the NaN firewall (finite check → skip →
count → `nan_skipped`, and the two-consecutive-all-non-finite-windows `LoudError`)
and the log/JSON cadence. It is copied, not shared, because `loop.py` belongs to
another lane this round; a `firewall()` helper both loops call is the follow-up,
and the two copies must not drift before it lands.

**The host-table path (`engram_ram`), and why SFT drives it.** A pretrain
checkpoint from a `--engram-ram` run stores its memory OUTSIDE the model half
(the sibling `<ckpt>.ngram`; `Engram.to_host` deletes the `engram.tables`
parameter), so `--init-from` such a checkpoint has to go through
`load_model_state` — and that loader leaves the model ON host tables. This loop
therefore owns that substrate the way `train()` does: enable before the
optimizer exists, one `host_step_all` per training step, the same counters, and
`drop_pending` on a NaN-skipped step. Without it the memory would be read every
forward and never trained — a SILENT degradation, and a `_pending` list that
grows one step at a time. `cfg.engram_ram` must AGREE with the init
checkpoint's substrate (`_init_cfg` refuses a mismatch loudly): the flag is what
`save_ckpt` keys the `.ngram` write off, so a host model under an `engram_ram=false`
run would lose its memory at the first checkpoint.

**The objective.** The model is called with `targets=None` (the decode forward:
no in-loop `L_Rec`), and the CE is taken on the logits it returns, masked. That
is deliberate: those are the logits `decode_bytes` samples from and the ones the
eval line's `fin` scores, so SFT trains exactly the readout generation reads.
"""

from __future__ import annotations

import dataclasses
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import RunConfig, TrainConfig
from ..instrument import Counters, LoudError, StepTimer, loud
from ..model.engram import Engram, enable_host_tables, host_engrams, host_step_all
from ..post.tasks import build_tasks
from .data import ByteStream, warmup_steps
from .loop import kda_field, load_ckpt, load_model_state, save_ckpt, unpack_state_dict
from .lr import wsd_lr
from .optim import MixOptimizer, clip_grads

#: The template's sentinels. All four are asserted absent from every rendered
#: instruction and answer ([`_assert_clean`]), which is what makes them markers:
#: a byte that can appear in the text is not a delimiter.
USER = 0x01
ASSIST = 0x02
END = 0x03
PAD = 0x00
SENTINELS = (USER, ASSIST, END, PAD)

#: The generic half of the corpus: the model has to learn the FORMAT, not just
#: code. Hand-written, both languages the corpus speaks, answers kept short so
#: the pair fits a 64-byte window (the tests' tiny preset).
GENERIC_QA: tuple[tuple[str, str], ...] = (
    ("Say hello.", "Hello!"),
    ("What is 2 + 2?", "answer: 4"),
    ("What is the capital of France?", "answer: Paris"),
    ("What color is the sky on a clear day?", "answer: blue"),
    ("How many days are in a week?", "answer: 7"),
    ("What is 10 - 3?", "answer: 7"),
    ("Is 17 a prime number?", "answer: yes"),
    ("What is the plural of 'mouse'?", "answer: mice"),
    ("What does CPU stand for?", "answer: central processing unit"),
    ("Next number: 2, 4, 6, 8, ...", "answer: 10"),
    ("What is 3 times 4?", "answer: 12"),
    ("Say goodbye politely.", "Goodbye, and have a good day."),
    ("Which month starts the year?", "answer: January"),
    ("Translate 'cat' into Russian.", "answer: кошка"),
    ("Give the reverse of 'abc'.", "answer: cba"),
    ("Поздоровайся.", "Привет!"),
    ("Сколько будет 3 + 4?", "answer: 7"),
    ("Столица России?", "answer: Москва"),
    ("Сколько дней в году?", "answer: 365"),
    ("Какого цвета трава?", "answer: зелёная"),
    ("Переведи слово 'dog' на русский.", "answer: собака"),
    ("Назови первый день недели.", "answer: понедельник"),
    ("Сколько будет 9 * 9?", "answer: 81"),
    ("Что больше: 5 или 8?", "answer: 8"),
    ("Скажи 'до свидания' вежливо.", "До свидания, хорошего дня!"),
    ("Сколько сторон у квадрата?", "answer: 4"),
    ("Продолжи: 1, 2, 3, ...", "answer: 4"),
    ("Как называется наша планета?", "answer: Земля"),
    ("Сколько букв в слове 'кот'?", "answer: 3"),
    ("Что такое вода?", "answer: жидкость"),
    # --- the second block: identity, arithmetic drills, terse style (owner:
    # "честная, критическая, немногословная" — answers stay short on purpose)
    ("Who are you?", "I am dormouse, a small byte-level language model."),
    ("What is your name?", "answer: dormouse"),
    ("Are you a human?", "answer: no"),
    ("Are you sure?", "answer: only as sure as my training data."),
    ("What can you do?", "answer: answer short questions, write small functions, do arithmetic."),
    ("What can you not do?", "answer: I am small: long reasoning and open-ended chat are weak."),
    ("Say something honest.", "I may be wrong; verify my answers."),
    ("Do you know everything?", "answer: no"),
    ("What is 5 + 7?", "answer: 12"),
    ("What is 9 - 4?", "answer: 5"),
    ("What is 6 times 6?", "answer: 36"),
    ("What is 12 / 4?", "answer: 3"),
    ("What is 2 + 2?", "answer: 4"),
    ("What is 100 - 1?", "answer: 99"),
    ("Which is bigger: 9 or 4?", "answer: 9"),
    ("Is 4 an even number?", "answer: yes"),
    ("Is 7 even?", "answer: no"),
    ("Count from 1 to 5.", "answer: 1, 2, 3, 4, 5"),
    ("What comes after 9?", "answer: 10"),
    ("What comes before 5?", "answer: 4"),
    ("Сколько будет 2 + 2?", "answer: 4"),
    ("Сколько будет 10 - 6?", "answer: 4"),
    ("Сколько будет 5 * 5?", "answer: 25"),
    ("Ты человек?", "answer: нет"),
    ("Как тебя зовут?", "answer: dormouse"),
    ("Что ты умеешь?", "answer: короткие ответы, простые функции, арифметика."),
    ("Что ты не умеешь?", "answer: я маленькая модель: длинные рассуждения слабые."),
    ("Скажи что-нибудь честное.", "Я могу ошибаться — проверяй ответы."),
    ("Ты уверен?", "answer: настолько, насколько позволяют мои данные."),
    ("Сколько будет 7 + 8?", "answer: 15"),
    ("Назови чётное число.", "answer: 2"),
    ("Что больше: 3 или 10?", "answer: 10"),
    ("Посчитай от 1 до 5.", "answer: 1, 2, 3, 4, 5"),
)


def _ru_pairs() -> list[tuple[str, str]]:
    """The Russian half: generated small talk, arithmetic drills and simple facts.

    `GENERIC_QA` carries ~35 RU pairs; a chat that only ever hears English plus
    a handful of Russian sentences drifts to English on a Russian prompt. This
    bank is a pure function (no RNG) so the corpus stays deterministic, and
    every answer stays short — the template renders both halves into one
    `seq_len` window.
    """
    out: list[tuple[str, str]] = []
    hello_q = [
        "Привет",
        "Привет!",
        "Здравствуй",
        "Здравствуйте",
        "Добрый день",
        "Добрый вечер",
        "Доброе утро",
        "Хай",
    ]
    hello_a = ["Привет!", "Здравствуй!", "Привет, чем помочь?", "Добрый день!"]
    for i, q in enumerate(hello_q):
        out.append((q, hello_a[i % len(hello_a)]))
    out += [
        ("Как дела?", "Хорошо, спасибо! А у тебя?"),
        ("Как у тебя дела?", "Нормально, спасибо."),
        ("Как тебя зовут?", "Меня зовут dormouse."),
        ("Как твоё имя?", "answer: dormouse"),
        ("Ты человек?", "answer: нет"),
        ("Ты робот?", "answer: да, я программа."),
        ("Ты живой?", "answer: нет, я языковая модель."),
        ("Что ты такое?", "Я маленькая байтовая языковая модель."),
        ("Кто ты?", "Я dormouse — маленькая языковая модель."),
        ("Что ты умеешь?", "answer: короткие ответы, арифметика, простые функции."),
        ("Что ты не умеешь?", "answer: длинные рассуждения и открытый чат — слабые."),
        ("Ты умный?", "Я маленькая модель: на простом отвечаю, на сложном ошибаюсь."),
        ("Ты уверен?", "answer: настолько, насколько позволяют мои данные."),
        ("Скажи что-нибудь честное.", "Я могу ошибаться — проверяй мои ответы."),
        ("Извинись.", "Извини, пожалуйста."),
        ("Скажи спасибо.", "Спасибо!"),
        ("Как сказать 'hello' по-русски?", "answer: привет"),
        ("Переведи 'cat' на русский.", "answer: кошка"),
        ("Переведи 'dog' на русский.", "answer: собака"),
        ("Переведи 'house' на русский.", "answer: дом"),
        ("Переведи 'water' на русский.", "answer: вода"),
        ("Сколько дней в неделе?", "answer: 7"),
        ("Сколько дней в году?", "answer: 365"),
        ("Сколько месяцев в году?", "answer: 12"),
        ("Сколько часов в сутках?", "answer: 24"),
        ("Сколько сторон у квадрата?", "answer: 4"),
        ("Сколько сторон у треугольника?", "answer: 3"),
        ("Какого цвета небо днём?", "answer: голубое"),
        ("Какого цвета трава?", "answer: зелёная"),
        ("Какого цвета снег?", "answer: белый"),
        ("Какого цвета солнце?", "answer: жёлтое"),
        ("Какого цвета море?", "answer: синее"),
        ("Столица России?", "answer: Москва"),
        ("Столица Франции?", "answer: Париж"),
        ("Столица Германии?", "answer: Берлин"),
        ("Столица Японии?", "answer: Токио"),
        ("Столица Италии?", "answer: Рим"),
        ("Как называется наша планета?", "answer: Земля"),
        ("Назови первый день недели.", "answer: понедельник"),
        ("Что больше: 5 или 8?", "answer: 8"),
        ("Что больше: 3 или 10?", "answer: 10"),
        ("Число 4 чётное?", "answer: да"),
        ("Число 7 чётное?", "answer: нет"),
        ("Посчитай от 1 до 5.", "answer: 1, 2, 3, 4, 5"),
        ("Продолжи: 2, 4, 6, ...", "answer: 8"),
        ("Продолжи: 1, 2, 3, ...", "answer: 4"),
        ("Что идёт после 9?", "answer: 10"),
        ("Что идёт перед 5?", "answer: 4"),
        ("Скажи 'до свидания' вежливо.", "До свидания, хорошего дня!"),
        ("Пока!", "Пока! Хорошего дня."),
    ]
    # The drills: every table a small model should hold without thinking.
    for a in range(1, 13):
        for b in range(1, 13):
            out.append((f"Сколько будет {a} + {b}?", f"answer: {a + b}"))
    for a in range(2, 13):
        for b in range(1, a):
            out.append((f"Сколько будет {a} - {b}?", f"answer: {a - b}"))
    for a in range(2, 10):
        for b in range(2, 10):
            out.append((f"Сколько будет {a} * {b}?", f"answer: {a * b}"))
    return out


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------


def _read_chat_file(path: Path) -> list[tuple[str, str]]:
    """Parse ONE chat file into raw pairs, refusing malformed input loudly.

    Two formats, dispatched by suffix:

    * `.jsonl` — one `{"instruction": ..., "response": ...}` object per line
      (the distill exports use that pair).
    * anything else (`.qa`, `.txt`) — the hand-editable Q/A text format:

          # comments and blank lines are skipped
          Q: What is your name?
          A: dormouse.

          Q: Привет!
          A: Привет! Чем помочь?

      The Q/A form exists because personal examples are hand-written: one line
      per half, `#` to comment, blank lines free. Every malformed line is a
      LOUD error naming `file:line`, never a silent skip.
    """
    import json

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        pairs: list[tuple[str, str]] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                loud(
                    f"sft --chat-data {path}:{lineno}: not valid JSON ({exc.msg}). "
                    f'Escape: one {{"instruction", "response"}} object per line.'
                )
            pairs.append(
                (str(row.get("instruction", "")).strip(), str(row.get("response", "")).strip())
            )
        return pairs
    pairs = []
    question: str | None = None
    q_line = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("Q:"):
            if question is not None:
                loud(
                    f"sft --chat-data {path}:{lineno}: a new `Q:` before the question "
                    f"at line {q_line} got its `A:`. Escape: give every question an "
                    f"answer, or separate pairs with a blank line."
                )
            question = line[2:].strip()
            q_line = lineno
            if not question:
                loud(f"sft --chat-data {path}:{lineno}: empty question after `Q:`.")
        elif line.startswith("A:"):
            if question is None:
                loud(
                    f"sft --chat-data {path}:{lineno}: an `A:` without a preceding "
                    f"`Q:`. Escape: write the question first, one line per half."
                )
            answer = line[2:].strip()
            if not answer:
                loud(f"sft --chat-data {path}:{lineno}: empty answer after `A:`.")
            pairs.append((question, answer))
            question = None
        else:
            loud(
                f"sft --chat-data {path}:{lineno}: neither `Q: ...`, `A: ...` nor a "
                f"`#` comment: {line[:48]!r}. Escape: the Q/A text format is one "
                f"line per half (or use a .jsonl file)."
            )
    if question is not None:
        loud(
            f"sft --chat-data {path}:{q_line}: the question never got an `A:` line. "
            f"Escape: add one."
        )
    return pairs


def _chat_pairs(
    paths: str | Path | list[str | Path],
    seq_len: int,
    cap_per_file: int | None = None,
    seed: int = 0,
) -> tuple[list[tuple[str, str]], dict]:
    """Load chat files (one path or a list) and keep the one-window fits.

    The build's own rule is applied here, early: an example longer than
    `seq_len` B of content can never be rendered into a record, so it is dropped
    with a count rather than in the build loop (where a `cap` would otherwise
    spend slots on rows that cannot land). Sentinel-carrying rows are dropped
    the same way — the marker license requires it, and real web text can carry
    a stray 0x00.

    The cap is PER FILE (`cap_per_file`), not on the merged pool: a 500-row
    hand-written file and a 100k-row export are both fully useful, and a global
    cap would dilute the small curated file to a rounding error. Each file's
    sample is `random.Random(seed + 1)`, so it is a pure function of the seed.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    pairs: list[tuple[str, str]] = []
    files: dict[str, int] = {}
    dropped_long = dropped_dirty = 0
    fit_total = 0
    for path in paths:
        p = Path(path)
        if not p.is_file():
            loud(
                f"sft --chat-data: {p} does not exist. Escape: a .jsonl export or "
                f"a hand-written Q/A text file (see sft_pairs/mine.qa)."
            )
        raw_pairs = _read_chat_file(p)
        files[str(p)] = len(raw_pairs)
        kept: list[tuple[str, str]] = []
        for instruction, answer in raw_pairs:
            if not instruction or not answer:
                dropped_dirty += 1
                continue
            raw_i, raw_a = instruction.encode(), answer.encode()
            if len(raw_i) + len(raw_a) + 4 > seq_len:
                dropped_long += 1
                continue
            if any(b in raw_i for b in SENTINELS) or any(b in raw_a for b in SENTINELS):
                dropped_dirty += 1
                continue
            kept.append((instruction, answer))
        fit_total += len(kept)
        if cap_per_file is not None and len(kept) > cap_per_file:
            random.Random(seed + 1).shuffle(kept)
            kept = kept[:cap_per_file]
        pairs.extend(kept)
    return pairs, {
        "chat_files": files,
        "chat_fit": fit_total,
        "chat_dropped_long": dropped_long,
        "chat_dropped_dirty": dropped_dirty,
    }


def _assert_clean(text: str) -> None:
    """No sentinel byte may appear in an instruction or an answer.

    The assert is the whole license for a 256-byte vocabulary's delimiters: if a
    marker can occur in the text, the mask's marker positions are not the
    example's boundaries and the loss silently supervises the wrong bytes.
    """
    raw = text.encode("utf-8")
    for b in SENTINELS:
        assert b not in raw, (
            f"sentinel 0x{b:02x} appears in the corpus text {text[:40]!r}: the "
            f"markers must be bytes the generated text cannot contain"
        )


def render(instruction: str, answer: str) -> bytes:
    """One example in the canonical template, markers in, no padding."""
    _assert_clean(instruction)
    _assert_clean(answer)
    return (
        bytes([USER])
        + instruction.encode("utf-8")
        + b"\n"
        + bytes([ASSIST])
        + answer.encode("utf-8")
        + bytes([END])
    )


def _examples(
    seed: int, n_tasks: int, chat: list[tuple[str, str]] | None = None
) -> list[tuple[str, str]]:
    """The pairs: generic QA + the RU bank + seeded tasks (REUSED) + chat rows."""
    out = list(GENERIC_QA) + _ru_pairs()
    for task in build_tasks("code", seed, n=n_tasks):
        # The task's prompt is the instruction (signature + docstring); the
        # hidden reference implementation is the answer.
        out.append((task.prompt.rstrip("\n"), task.solution.rstrip("\n")))
    for task in build_tasks("math", seed, n=n_tasks):
        # The math prompt asks for a final `answer: <value>` line, so the answer
        # teaches that format too (it is what `verifiers._final_answer` reads).
        out.append((task.prompt, f"answer: {task.solution}"))
    if chat:
        out.extend(chat)
    return out


def build_sft_corpus(
    out_path: str | Path,
    seed: int,
    *,
    seq_len: int,
    n_tasks: int = 32,
    chat_data: str | Path | list[str | Path] | None = None,
    chat_cap: int | None = None,
    holdout: int = 0,
) -> dict:
    """Write the instruction corpus: fixed `seq_len + 1`-byte records, deterministic.

    A pure function of `(seed, seq_len, n_tasks, chat_data, chat_cap)`: the same
    arguments produce the same bytes in any process (the shuffles are
    `random.Random(seed)` / `Random(seed + 1)`), which is what makes an SFT run
    replayable (ADR-0021's determinism discipline).

    `chat_data` mixes external rows (the distill `.jsonl` exports, hand-written
    Q/A text files — see [`_read_chat_file`]) into the corpus: they are
    pre-filtered to one-window fits and the sample from EACH file is cut to
    `chat_cap` rows (per file, not on the merged pool: a 500-row hand-written
    file must not be diluted by a 100k-row export). Without it the corpus is
    the small seeded set, which is enough to learn the FORMAT and nothing else.

    An example that does not fit in ONE window is DROPPED, not truncated (a
    truncated answer teaches the model to stop mid-function) and not wrapped (a
    window must never span two examples). The drop is counted in the returned
    report, and a corpus in which nothing fits is a LOUD refusal rather than an
    empty file.
    """
    if seq_len < 8:
        loud(f"build_sft_corpus seq_len={seq_len}: a window must hold a marker pair")
    if n_tasks < 0:
        loud(f"build_sft_corpus n_tasks={n_tasks}: a task count cannot be negative")
    if chat_cap is not None and chat_cap < 0:
        loud(f"build_sft_corpus chat_cap={chat_cap}: a cap cannot be negative")
    chat: list[tuple[str, str]] = []
    chat_stats: dict = {
        "chat_files": {},
        "chat_loaded": 0,
        "chat_used": 0,
        "chat_dropped_long": 0,
        "chat_dropped_dirty": 0,
    }
    if chat_data is not None:
        chat, dropped = _chat_pairs(chat_data, seq_len, cap_per_file=chat_cap, seed=seed)
        chat_stats.update(dropped)
        chat_stats["chat_loaded"] = dropped.get("chat_fit", len(chat))
        chat_stats["chat_used"] = len(chat)
    examples = _examples(seed, n_tasks, chat)
    random.Random(seed).shuffle(examples)
    record = seq_len + 1
    records: list[bytes] = []
    dropped = 0
    for instruction, answer in examples:
        raw = render(instruction, answer)
        # `len(raw) <= seq_len` (not `record`): the END marker must land INSIDE
        # the input window for the mask to see it, so at least one pad byte
        # follows every example.
        if len(raw) > seq_len:
            dropped += 1
            continue
        records.append(raw + bytes([PAD]) * (record - len(raw)))
    if not records:
        loud(
            f"build_sft_corpus: all {len(examples)} examples are longer than one "
            f"window ({seq_len} B of content + 1 pad). Escape: raise seq_len, or "
            f"shrink the examples."
        )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    held = 0
    if holdout:
        if holdout >= len(records):
            loud(
                f"build_sft_corpus holdout={holdout}: the corpus has only "
                f"{len(records)} records. Escape: a holdout smaller than the corpus."
            )
        held = holdout
        hold_path = path.with_name(path.name + ".holdout")
        hold_path.write_bytes(b"".join(records[-held:]))
        records = records[:-held]
    path.write_bytes(b"".join(records))
    return {
        "path": str(path),
        "records": len(records),
        "dropped": dropped,
        "record_bytes": record,
        "bytes": record * len(records),
        "seq_len": seq_len,
        "seed": seed,
        "n_tasks": n_tasks,
        "holdout": held,
        **chat_stats,
    }


# ---------------------------------------------------------------------------
# the mask and the objective
# ---------------------------------------------------------------------------


def sft_mask(windows: torch.Tensor) -> torch.Tensor:
    """`[b, t]` bool over TARGET positions: True from the assistant to the end marker.

    `windows` is the model INPUT `x [b, t]` and target position `j` predicts the
    record's byte `j + 1`, so the mask's first True is `assist - 1`: the `0x02`
    byte itself is supervised (the model has to learn to EMIT the marker — that
    is the format) and every prompt byte before it is not. The last True is
    `end - 1`, so `0x03` is supervised and the padding after it is not; a mask
    that ran to the end of the window would train the model on `0x00`.

    The three refusals are the single-example guarantee, checked per window
    rather than assumed from the build: no marker (a corpus built for another
    `seq_len`, or a pretrain corpus), two markers (a window that spans two
    examples), or non-padding after `0x03` (the same, caught from the other end).
    """
    assert windows.ndim == 2, f"windows must be [b, t], got {tuple(windows.shape)}"
    b, t = windows.shape
    assert t >= 2, f"a {t}-byte window cannot carry a marker pair"
    is_a = windows == ASSIST
    is_e = windows == END
    if not (bool(is_a.any(dim=1).all()) and bool(is_e.any(dim=1).all())):
        loud(
            "sft_mask: a window carries no assistant (0x02) or end (0x03) marker. "
            "This is not an SFT corpus window: it was built for another seq_len, "
            "or it is a pretrain corpus. Escape: rebuild with --build-sft at this "
            "run's seq_len."
        )
    if not (bool((is_a.sum(dim=1) == 1).all()) and bool((is_e.sum(dim=1) == 1).all())):
        loud(
            "sft_mask: a window carries more than one assistant/end marker, i.e. it "
            "spans two examples. The corpus records are `seq_len + 1` bytes and the "
            "run's seq_len must be the one they were built with. Escape: rebuild "
            "with --build-sft."
        )
    assist = is_a.to(torch.int8).argmax(dim=1)
    end = is_e.to(torch.int8).argmax(dim=1)
    if not bool((assist >= 1).all()) or not bool((end > assist).all()):
        loud(
            "sft_mask: a window's markers are out of order (the assistant marker at "
            "position 0, or the end marker before it). Escape: rebuild with "
            "--build-sft."
        )
    for i in range(b):
        if not bool((windows[i, int(end[i]) + 1 :] == PAD).all()):
            loud(
                "sft_mask: a window carries content after its end marker — the bytes "
                "after 0x03 must be 0x00 padding. The corpus was not built for this "
                "seq_len. Escape: rebuild with --build-sft."
            )
    pos = torch.arange(t, device=windows.device)
    return (pos[None, :] + 1 >= assist[:, None]) & (pos[None, :] + 1 <= end[:, None])


def sft_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked next-byte CE: the mean over the SUPERVISED positions only.

    `logits` is `[b*t, vocab]` (the model's own shape), `targets` `[b, t]`, `mask`
    `[b, t]` bool. Every prompt byte and every pad byte is dropped from BOTH the
    numerator and the denominator — an unmasked prompt byte is not "cheap", it is
    a byte the model was asked to predict that the task never wanted.
    """
    flat = mask.reshape(-1)
    if not bool(flat.any()):
        loud(
            "sft_loss: the mask supervises no byte in this window. Escape: the "
            "window's markers are wrong (see sft_mask)."
        )
    assert logits.shape[0] == flat.numel(), "logits rows != window positions"
    return F.cross_entropy(logits[flat], targets.reshape(-1)[flat])


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def _init_cfg(cfg: TrainConfig, path: Path, sd: dict, model: torch.nn.Module) -> TrainConfig:
    """The cfg that describes the INIT FILE: its substrate and its sidecar path.

    `load_model_state` reads the substrate from `cfg.engram_ram` and finds the
    `.ngram` sibling through `cfg.out_dir`/`cfg.ckpt_name`, so an init load
    cannot use the SFT run's own cfg — that names the sidecar THIS run will
    write, not the init file's. The substrate is read from the FILE, not from a
    flag: a host-table checkpoint's model half carries no `engram.tables`
    (`Engram.to_host` deletes the parameter), which is the file's own statement
    of where its memory lives.

    The run's `engram_ram` must AGREE with the file's, and a disagreement is
    LOUD, because the flag decides three things a mismatch would corrupt: the
    host switch happens BEFORE the optimizer exists (a switch after it leaves
    the tables in a parameter group that then steps a deleted tensor), the loop
    steps the tables only when the flag is on, and `save_ckpt` writes the
    `.ngram` only when it is on.
    """
    tables_in_file = any(k.endswith("engram.tables") for k in sd)
    has_engram = any(isinstance(m, Engram) for m in model.modules())
    host = has_engram and not tables_in_file
    if host and not cfg.engram_ram:
        loud(
            f"sft --init-from: {path}'s model half carries no `engram.tables`: it "
            f"is a host-table run (its rows are the sibling `{path.stem}.ngram`), "
            f"or it is not this model's checkpoint at all. This run is configured "
            f"engram_ram=false, so the memory could not be loaded: a host-table "
            f"load switches the model to host tables after the optimizer exists, "
            f"and the run would read a memory it never steps and never saves. "
            f"Escape: --set engram_ram=true (the pretrain's substrate), or an init "
            f"checkpoint whose model half carries the tables."
        )
    if tables_in_file and cfg.engram_ram:
        loud(
            f"sft --init-from: {path} carries its Engram in the model half "
            f"(`engram.tables`) but this run is configured engram_ram=true. "
            f"Switching to host tables would DROP the checkpoint's memory — the "
            f"host path starts from the model's own rows and this run's `.ngram` "
            f"sibling does not exist yet. Escape: --set engram_ram=false, or "
            f"re-save the init as a host-table run (with its `.ngram` sibling)."
        )
    return dataclasses.replace(cfg, engram_ram=host, out_dir=str(path.parent), ckpt_name=path.stem)


def sft_train(
    model: torch.nn.Module,
    run: RunConfig,
    *,
    stream: ByteStream | None = None,
    resume: bool = False,
    init_from: str | None = None,
    log=print,
    device: torch.device | str | None = None,
) -> dict:
    """Supervised fine-tuning on the prompt-masked CE. Returns a report dict.

    The schedule is WSD — `wsd_lr` with the run's `warmup_tokens`/`decay_frac`,
    i.e. the SAME schedule `train()` runs, not a second one to keep honest.

    The NaN firewall and the log/JSON cadence are COPIED from `train()`
    (`loop.py`, the finite-check/skip/count block and the window-drain block), so
    this loop carries the same two rules: a non-finite step touches no weight and
    is COUNTED as `nan_skipped`, and two consecutive windows in which every step
    was non-finite stop the run with a `LoudError`. They are copies because
    `loop.py` is another lane's file this round.

    `device` is the batch seam, as in `train()`: the stream hands out CPU
    tensors and the caller names where the model lives.

    `init_from` loads WEIGHTS through `load_model_state` (the ONE checkpoint
    loader), so an `engram_ram` pretrain checkpoint — whose model half carries
    no `engram.tables` — loads too: the helper switches the model to host tables
    and restores the sibling `.ngram`, and this loop then owns that substrate
    (see the module docstring). The run's `engram_ram` must match the
    checkpoint's; `_init_cfg` refuses a mismatch loudly.
    """
    cfg = run.train
    torch.set_float32_matmul_precision("high" if cfg.tf32 else "highest")
    device = None if device is None else torch.device(device)
    if stream is None:
        stream = ByteStream(cfg.data, cfg.batch, cfg.seq_len)
    eval_stream = None
    if cfg.eval and cfg.eval_data:
        # The holdout: the same record format, never read by the training
        # stream (the build writes it to a sibling file). The metric that
        # matters is `first` — the top-1 agreement on the FIRST answer byte,
        # which is the byte that requires READING the instruction; the CE can
        # look healthy while the model only predicts answer marginals.
        eval_stream = ByteStream(cfg.eval_data, cfg.batch, cfg.seq_len)
    model.train()
    if cfg.engram_ram:
        # The host path's substrate, switched BEFORE the optimizer exists — the
        # rule `train()` carries (`loop.py:495-500`): `Engram.to_host` DELETES
        # the table parameter, so a group built first would hold a dead tensor
        # and step it every step for nothing (402M params at b1).
        # `enable_host_tables`, not `host_tables`: the CUDA-only refusal is a
        # training VRAM-budget rule, and on a CPU model the host path's NUMERICS
        # are device-agnostic (the tables ARE CPU memory) — the same call
        # `load_model_state` makes, for the same reason.
        enable_host_tables(model)
        if not host_engrams(model):
            loud(
                "sft engram_ram=true but the model has no Engram to host "
                "(use_engram=false?): the flag would host nothing while the run "
                "reported it on. Escape: use_engram=true, or engram_ram=false."
            )
        log(
            f"engram_ram: {len(host_engrams(model))} table set(s) in host RAM, VRAM "
            f"holds the batch's unique rows"
        )
    opt = MixOptimizer(
        model,
        lr=cfg.lr,
        wd=cfg.wd,
        opt=cfg.opt,
        ns_steps=cfg.ns_steps,
        ns_kind=cfg.ns_kind,
        colrow_3d=cfg.muon_colrow_3d,
    )
    if cfg.optim_8bit:
        opt.enable_bnb8bit()
    start = 0
    if init_from is not None:
        # Fine-tune init: WEIGHTS from a foreign checkpoint (e.g. the pretrain
        # run), no config gate and no optimizer state. That is the deliberate
        # difference from `resume`: a resume continues THIS run and must prove
        # the config did not move (ADR-0021); an init knowingly changes the
        # objective (pretrain -> instruction) and starts a fresh snapshot, a
        # fresh optimizer and a fresh byte stream. LOUD because the weights
        # leave their recorded context.
        if resume:
            loud(
                "sft: --init-from and --resume together name two different "
                "sources of weights. Escape: --init-from to start a new SFT run "
                "from a pretrain checkpoint, or --resume to continue an SFT run."
            )
        path = Path(init_from)
        if not path.is_file():
            loud(f"sft --init-from: {path} does not exist. Escape: the pretrain run's .pt path.")
        foreign = torch.load(path, map_location="cpu", weights_only=False)
        sd = unpack_state_dict(foreign.get("model", foreign), foreign.get("model_quant_rung"))
        # The ONE loader (`load_model_state`), which is what makes a host-table
        # init possible at all: such a checkpoint's model half carries no
        # `engram.tables`, so the raw `load_state_dict` this replaced refused
        # every b1/medium checkpoint with `Missing key(s)`. The helper switches
        # the model to host tables BEFORE the load and restores the
        # `<ckpt>.ngram` sibling; a missing sibling is its LOUD refusal, never
        # freshly seeded rows. STRICT: a pretrain -> SFT transfer is the same
        # architecture by construction, so the substrate is the only legal
        # difference — and `_init_cfg` refuses a substrate mismatch before the
        # load, which is why strict can stay strict.
        load_model_state(model, _init_cfg(cfg, path, sd, model), sd, strict=True)
        log(f"sft init: weights from {path} (step {foreign.get('step', '?')}), fresh optimizer")
    if resume:
        ck, _stored = load_ckpt(run)
        # The same ONE loader as the init path (and as `train()`'s resume): a
        # host-table run's model half has no `engram.tables`, and `load_ckpt`
        # already refused a missing `.ngram` (`require_ngram`).
        load_model_state(model, cfg, ck["model"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"]) + 1
        if start:
            # One step consumes `batch x (seq_len + 1)` bytes, so a resume
            # fast-forwards exactly as far as the run had read (the same
            # positioning rule as `train()`).
            stream.skip_bytes(start * (cfg.seq_len + 1) * cfg.batch)
        log(f"sft resume: step={start} opt_steps={opt.steps}")

    warm = warmup_steps(cfg.warmup_tokens, cfg.batch, cfg.seq_len)
    decay = int(round(cfg.decay_frac * cfg.steps))
    tokens_per_step = cfg.batch * cfg.seq_len
    total = Counters()
    window = Counters()
    nan_skipped = 0
    drowning = 0
    window_finite: list[bool] = []
    timer = StepTimer()
    last_loss = float("nan")
    supervised = 0
    sink = None
    if cfg.json_log:
        sink = open(cfg.json_log, "a", encoding="utf-8")

        def _row(**kw) -> None:
            import json

            sink.write(
                json.dumps({k: v for k, v in kw.items() if v is not None}, default=str) + "\n"
            )
            sink.flush()

        _row(
            event="sft_run",
            source=run.source,
            steps=cfg.steps,
            batch=cfg.batch,
            seq_len=cfg.seq_len,
            lr=cfg.lr,
            opt=cfg.opt,
            resumed=start,
        )
    for step in range(start, cfg.steps):
        lr = wsd_lr(
            step, total_steps=cfg.steps, peak_lr=cfg.lr, warmup_steps=warm, decay_steps=decay
        )
        for group in opt.adamw.param_groups:
            group["lr"] = lr
        if opt.muon is not None:
            for group in opt.muon.param_groups:
                group["lr"] = lr

        x, y = stream.next_batch()
        if device is not None:
            x, y = x.to(device), y.to(device)
        mask = sft_mask(x)
        # `targets=None`: the decode forward. Its logits are the ones generation
        # samples and the eval's `fin` scores, so SFT trains that readout — the
        # in-loop `L_Rec` is a pretraining signal and is not built here.
        logits, _none, step_counters = model(x)
        assert logits.shape[-1] == run.model.vocab, "logits width != vocab"
        loss = sft_loss(logits, y, mask)
        n_mask = int(mask.sum())
        total.count("sft_bytes", n_mask)
        window.count("sft_bytes", n_mask)
        supervised += n_mask
        total.merge(step_counters)
        window.merge(step_counters)
        timer.tick()

        # ONE host read per step (audit MAJOR 4), same as the train loop.
        lf = float(loss.detach())
        finite = math.isfinite(lf)
        window_finite.append(finite)
        if finite:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                clip_grads(model.parameters(), cfg.grad_clip)
            opt.step()
            if cfg.engram_ram:
                # The host update runs ONCE per TRAINING step, after the
                # optimizer — the same rule and the same counters as `train()`
                # (`loop.py:761-772`): the loop reads the memory once per
                # ITERATION and `host_step` sums those into one update per row,
                # and a NaN-skipped step (below) updates nothing.
                updates, rows = host_step_all(model, lr)
                total.count("engram_host_updates", updates)
                window.count("engram_host_updates", updates)
                total.count("engram_host_update_rows", rows)
                window.count("engram_host_update_rows", rows)
            last_loss = lf
        else:
            nan_skipped += 1
            total.count("nan_skipped")
            window.count("nan_skipped")
            opt.zero_grad(set_to_none=True)
            # A skipped step never reaches `host_step_all`, so the host-path
            # leaves it built would sit in `Engram._pending` until the next REAL
            # step (`loop.py:786-793`); the in-VRAM path never appends, so this
            # is a no-op there.
            for m in host_engrams(model):
                m.drop_pending()

        if (step + 1) % cfg.log_every == 0:
            window.drain()
            total.drain()
            if window_finite and not any(window_finite):
                drowning += 1
            elif window_finite:
                drowning = 0
            window_finite = []
            parts = [
                f"sft step={step + 1}",
                f"loss={last_loss:.4f}",
                f"lr={lr:.3e}",
                kda_field(window),
                window.line(),
            ]
            # The interval read must precede any `report()` (it resets the
            # mark) — same order fix as `train/loop.py` (2026-10-17).
            interval_s = timer.window_s()
            if cfg.timers:
                parts.append(timer.report(cfg.log_every * tokens_per_step))
            timer.mark = time.perf_counter()
            rate = (cfg.log_every * tokens_per_step) / interval_s if interval_s > 0 else None
            log(" ".join(p for p in parts if p))
            if sink is not None:
                _row(
                    event="sft_log",
                    step=step + 1,
                    loss=last_loss,
                    lr=lr,
                    bytes_per_s=rate,
                    counters=dict(window),
                    nan_skipped=nan_skipped,
                )
            window = Counters()
            if drowning >= 2:
                raise LoudError(
                    f"two consecutive windows of {cfg.log_every} SFT steps were "
                    f"entirely non-finite (steps {step + 1 - 2 * cfg.log_every + 1}"
                    f"..{step + 1}); nan_skipped={nan_skipped}. The same rule "
                    f"`train()` carries (readiness plan §3 Phase 1 item 1): a "
                    f"drowning run must stop for a human, not exit 0. Escape: "
                    f"lower lr, check the corpus, or bisect the arms."
                )

        if eval_stream is not None and (step + 1) % cfg.eval_every == 0:
            model.eval()
            ce_total = 0.0
            batches = 0
            first_ok = 0
            first_n = 0
            with torch.no_grad():
                for _ in range(cfg.eval_batches):
                    ex, ey = eval_stream.next_batch()
                    if device is not None:
                        ex, ey = ex.to(device), ey.to(device)
                    emask = sft_mask(ex)
                    elogits, _en, _ec = model(ex)
                    flat = emask.reshape(-1)
                    ce_total += float(F.cross_entropy(elogits[flat], ey.reshape(-1)[flat]))
                    batches += 1
                    # The conditioning metric: at the assistant marker's
                    # position the model predicts the FIRST answer byte — the
                    # one byte that cannot be guessed from the answer alone.
                    is_a = ex == ASSIST
                    assist = is_a.to(torch.int8).argmax(dim=1)
                    rows = torch.arange(ex.shape[0], device=ex.device)
                    per_row = elogits.reshape(ex.shape[0], ex.shape[1], -1)
                    pred = per_row[rows, assist].argmax(dim=-1)
                    true = ey[rows, assist]
                    first_ok += int((pred == true).sum())
                    first_n += int(ex.shape[0])
            model.train()
            ce = ce_total / max(batches, 1)
            frac = first_ok / max(first_n, 1)
            log(
                f"sft eval step={step + 1} ce={ce:.4f} bpb={ce / math.log(2):.4f} "
                f"first={100 * frac:.1f}% over {first_n} records"
            )
            if sink is not None:
                _row(event="sft_eval", step=step + 1, ce=ce, first=frac, records=first_n)

        if (step + 1) % cfg.ckpt_every == 0:
            weights, snapshot = save_ckpt(run, model, opt, step)
            log(f"ckpt step={step + 1} -> {weights} ({snapshot})")

    if sink is not None:
        sink.close()
    return {
        "steps": cfg.steps,
        "loss": last_loss,
        "nan_skipped": nan_skipped,
        "supervised_bytes": supervised,
        "counters": total,
        "opt_steps": opt.steps,
    }
