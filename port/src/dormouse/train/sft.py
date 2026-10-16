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
pass because the sources are generated text (the seeded tasks of `post/tasks.py`
and the hand-written `GENERIC_QA`), never arbitrary user bytes. `0x03` and
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
`save_ckpt`/`load_ckpt`, `kda_field` and `Counters`. **Copied verbatim from
`train()` (`loop.py`):** the NaN firewall (finite check → skip → count →
`nan_skipped`, and the two-consecutive-all-non-finite-windows `LoudError`) and the
log/JSON cadence. It is copied, not shared, because `loop.py` belongs to another
lane this round; a `firewall()` helper both loops call is the follow-up, and the
two copies must not drift before it lands.

**The objective.** The model is called with `targets=None` (the decode forward:
no in-loop `L_Rec`), and the CE is taken on the logits it returns, masked. That
is deliberate: those are the logits `decode_bytes` samples from and the ones the
eval line's `fin` scores, so SFT trains exactly the readout generation reads.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import RunConfig
from ..instrument import Counters, LoudError, StepTimer, loud
from ..post.tasks import build_tasks
from .data import ByteStream, warmup_steps
from .loop import kda_field, load_ckpt, save_ckpt
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
)


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------


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


def _examples(seed: int, n_tasks: int) -> list[tuple[str, str]]:
    """The pairs: seeded code + math tasks (REUSED, not re-invented) + generic QA."""
    out = list(GENERIC_QA)
    for task in build_tasks("code", seed, n=n_tasks):
        # The task's prompt is the instruction (signature + docstring); the
        # hidden reference implementation is the answer.
        out.append((task.prompt.rstrip("\n"), task.solution.rstrip("\n")))
    for task in build_tasks("math", seed, n=n_tasks):
        # The math prompt asks for a final `answer: <value>` line, so the answer
        # teaches that format too (it is what `verifiers._final_answer` reads).
        out.append((task.prompt, f"answer: {task.solution}"))
    return out


def build_sft_corpus(out_path: str | Path, seed: int, *, seq_len: int, n_tasks: int = 32) -> dict:
    """Write the instruction corpus: fixed `seq_len + 1`-byte records, deterministic.

    A pure function of `(seed, seq_len, n_tasks)`: the same arguments produce the
    same bytes in any process (the shuffle is a `random.Random(seed)`), which is
    what makes an SFT run replayable (ADR-0021's determinism discipline).

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
    examples = _examples(seed, n_tasks)
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


def sft_train(
    model: torch.nn.Module,
    run: RunConfig,
    *,
    stream: ByteStream | None = None,
    resume: bool = False,
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
    """
    cfg = run.train
    if cfg.engram_ram:
        # LOUD, not a fallback: the host path's state is a `.ngram` sidecar that
        # `train()` writes and steps and this loop does neither, so the run would
        # report host tables while the memory trained in VRAM.
        loud(
            "sft with engram_ram=true: the host-table path's state is a `.ngram` "
            "sidecar that this loop does not enable, step or save (`train()` does "
            "all three). Escape: engram_ram=false, or port the host step into "
            "sft_train."
        )
    torch.set_float32_matmul_precision("high" if cfg.tf32 else "highest")
    device = None if device is None else torch.device(device)
    if stream is None:
        stream = ByteStream(cfg.data, cfg.batch, cfg.seq_len)
    model.train()
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
    if resume:
        ck, _stored = load_ckpt(run)
        model.load_state_dict(ck["model"])
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

        finite = bool(torch.isfinite(loss))
        window_finite.append(finite)
        if finite:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                clip_grads(model.parameters(), cfg.grad_clip)
            opt.step()
            last_loss = float(loss.detach())
        else:
            nan_skipped += 1
            total.count("nan_skipped")
            window.count("nan_skipped")
            opt.zero_grad(set_to_none=True)

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
            if cfg.timers:
                parts.append(timer.report(cfg.log_every * tokens_per_step))
            interval_s = timer.window_s()
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
