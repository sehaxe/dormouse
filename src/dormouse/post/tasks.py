"""The seedable synthetic task builder — the little-data multiplier.

`docs/research/2026-10-15-agentic-post-training.md` §2.2: verifiable tasks are
*generated* with their own verification targets; §5 rung (a) says every other
rung consumes `(prompt, response, verifier-tag)` records, so the builder emits
exactly that plus the hidden reference.

Two classes, each 128 seedable tasks:

- `code` — a function completion from a docstring: the prompt gives the
  signature and doc, the model must return the whole function, the reference
  implementation is hidden and drives the I/O the verifier asserts.
- `math` — arithmetic / linear-solve / fraction prompts with an exact final
  answer (sympy-equivalence, not string equality).

A third, `noop`, exists only so the do-not-run refusal of `verifiers` is
testable; it carries nothing to solve.

Determinism is the contract: a task is a pure function of `(cls, seed, index)`,
so the same seed is the same bytes and an eval can be replayed (ADR-0021's
determinism discipline). The builder never touches the network, the clock or the
global RNG.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ..instrument import loud

#: Tasks a single seed produces per class.
TASKS_PER_CLASS = 128

#: The seed offset that keeps the GENERATED RL tasks off the SFT corpus.
#:
#: `train.sft._examples` builds its task pairs from `build_tasks(code|math, seed)`
#: with the RUN's seed; an RL run with the same seed would draw the same prompts.
#: The builder has no offset parameter because SFT's own tasks must stay where
#: they are (a checkpoint's corpus and a gate's fixtures are pinned to them), so
#: the RL caller adds this instead. It is one constant, not a second convention:
#: `rl_train` is the only caller that adds it.
SFT_TASK_OFFSET = 10_000

#: Byte budgets the two live classes are scored under (`correct@budget`).
_CODE_BUDGET = 256
_MATH_BUDGET = 64

__all__ = ["SFT_TASK_OFFSET", "TASKS_PER_CLASS", "Task", "build_tasks", "load_rl_tasks"]


@dataclass(frozen=True)
class Task:
    """One verifiable task. `solution` is the hidden reference: [`as_dict`] drops
    it, so the verifier structurally cannot read the answer it judges."""

    cls: str
    tid: str
    prompt: str
    expected: str
    solution: str
    tests: tuple[tuple[str, str], ...]
    match: str
    budget: int

    def as_dict(self) -> dict:
        """The `(prompt, verifier-tag)` record a rollout and the verifier see."""
        return {
            "cls": self.cls,
            "tid": self.tid,
            "prompt": self.prompt,
            "expected": self.expected,
            "tests": self.tests,
            "match": self.match,
            "budget": self.budget,
        }


def build_tasks(cls: str, seed: int, n: int = TASKS_PER_CLASS) -> list[Task]:
    """`n` deterministic tasks of `cls`, a pure function of `seed`."""
    if n < 0:
        loud(f"build_tasks n={n}: a task count cannot be negative")
    if cls == "code":
        return [_code_task(seed, i) for i in range(n)]
    if cls == "math":
        return [_math_task(seed, i) for i in range(n)]
    if cls == "noop":
        return [_noop_task(seed, i) for i in range(n)]
    loud(f"build_tasks: unknown class {cls!r} (known: code, math, noop)")


def _code_task(seed: int, i: int) -> Task:
    rng = random.Random(seed * 1_000_003 + i)
    kind = i % 3
    if kind == 0:  # add
        a, b = rng.randint(-20, 20), rng.randint(-20, 20)
        name, params, doc, body = "add", "a, b", "Return the sum of a and b.", "return a + b"
        calls = (f"add({a}, {b})", f"add({b}, {a})")
        value = str(a + b)
    elif kind == 1:  # square_sum
        k = rng.randint(1, 12)
        name, params, doc = "square_sum", "n", "Return 1**2 + 2**2 + ... + n**2."
        body = "return sum(j * j for j in range(1, n + 1))"
        calls = (f"square_sum({k})",)
        value = str(sum(j * j for j in range(1, k + 1)))
    else:  # count_vowels
        word = "".join(rng.choice("aeioubcdfg") for _ in range(rng.randint(1, 8)))
        name, params, doc = "count_vowels", "s", "Return the number of vowels (aeiou) in s."
        body = 'return sum(ch in "aeiou" for ch in s.lower())'
        calls = (f"count_vowels({word!r})",)
        value = str(sum(ch in "aeiou" for ch in word.lower()))

    solution = f'def {name}({params}):\n    """{doc}"""\n    {body}\n'
    tests = tuple((call, value) for call in calls)
    prompt = (
        "Complete this Python function. Reply with the whole function, nothing else.\n\n"
        f'def {name}({params}):\n    """{doc}"""\n'
    )
    return Task(
        cls="code",
        tid=f"code-{seed}-{i}",
        prompt=prompt,
        expected="\n".join(expected for _, expected in tests),
        solution=solution,
        tests=tests,
        match="exact",
        budget=_CODE_BUDGET,
    )


def _math_task(seed: int, i: int) -> Task:
    rng = random.Random(seed * 2_000_003 + i)
    kind = i % 3
    if kind == 0:  # arithmetic
        a, b, c = rng.randint(2, 20), rng.randint(2, 20), rng.randint(1, 40)
        expr, answer = f"{a} * {b} + {c}", a * b + c
    elif kind == 1:  # linear solve
        a, x, b = rng.randint(2, 9), rng.randint(-9, 9), rng.randint(-20, 20)
        expr, answer = f"{a} * x + {b} = {a * x + b}", x
    else:  # fraction sum
        p, q, r, s = rng.randint(1, 9), rng.randint(2, 9), rng.randint(1, 9), rng.randint(2, 9)
        expr, answer = f"{p}/{q} + {r}/{s}", Fraction(p, q) + Fraction(r, s)

    expected = str(answer)
    prompt = "Solve. Put the final value on its own last line as 'answer: <value>'.\n\n" + expr
    return Task(
        cls="math",
        tid=f"math-{seed}-{i}",
        prompt=prompt,
        expected=expected,
        solution=expected,
        tests=(),
        match="exact",
        budget=_MATH_BUDGET,
    )


def _noop_task(seed: int, i: int) -> Task:
    return Task(
        cls="noop",
        tid=f"noop-{seed}-{i}",
        prompt="(do-not-run class: there is nothing to solve)",
        expected="",
        solution="",
        tests=(),
        match="exact",
        budget=0,
    )


def load_rl_tasks(path: str | Path) -> list[Task]:
    """Read a `scripts/build_rl_tasks.py` JSONL into the loop's [`Task`]s.

    The record schema is `{id, kind, prompt, answer, verifier, difficulty_hint}`.
    `verifier` is `"math"` (the answer is the exact rational/integer) or an
    object `{"cls": "code", "tests": [[call, repr], ...], "match": "exact"}`
    (the answer is the hidden reference function). The loader refuses:

    - a missing field (the six names are the contract),
    - a verifier it does not know (a task nobody can grade is not a task),
    - a code record with no tests (the sandbox would have nothing to run),
    - a DUPLICATE rendered prompt: two records with the same prompt and
      different tests are one question the model cannot answer deterministically,
      and a prompt is the only identity a rollout sees.

    The prompt hash is checked here, not the id: ids are generated by the same
    builder and would agree with themselves even after the prompt drifted.
    """
    source = Path(path)
    if not source.is_file():
        loud(
            f"load_rl_tasks: {source} does not exist. Escape: build it with "
            f"`uv run port/scripts/build_rl_tasks.py --out {source}`, or leave "
            f"`rl_corpus` empty to use `build_tasks(seed + SFT_TASK_OFFSET)`."
        )
    tasks: list[Task] = []
    seen: dict[str, int] = {}
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            loud(f"load_rl_tasks: {source}:{lineno}: not JSON ({exc}). Escape: rebuild the corpus.")
        if not isinstance(record, dict):
            loud(f"load_rl_tasks: {source}:{lineno}: a record must be a JSON object.")
        missing = [
            k
            for k in ("id", "kind", "prompt", "answer", "verifier", "difficulty_hint")
            if k not in record
        ]
        if missing:
            loud(
                f"load_rl_tasks: {source}:{lineno}: missing {missing}. Escape: the six "
                f"fields are id, kind, prompt, answer, verifier, difficulty_hint."
            )
        prompt = str(record["prompt"])
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if digest in seen:
            loud(
                f"load_rl_tasks: {source}:{lineno}: the rendered prompt is a duplicate "
                f"of line {seen[digest]} ({record['id']!r} vs that record's id). Two ids "
                f"do not make two tasks. Escape: rebuild with a different --seed."
            )
        seen[digest] = lineno
        verifier = record["verifier"]
        if verifier == "math":
            cls, tests, match, budget = "math", (), "exact", _MATH_BUDGET
            expected = str(record["answer"])
            # `S_sol` prices the RESPONSE the model must emit, not the value the
            # verifier extracts: a correct reply is `answer: <value>`, so the
            # hidden reference is that whole line. (The generated `build_tasks`
            # math tasks predate this and carry the bare value; a corpus task is
            # a new task, and its thrift reference is the one it is scored on.)
            solution = f"answer: {expected}"
        elif isinstance(verifier, dict) and verifier.get("cls") == "code":
            tests = tuple(tuple(str(x) for x in pair) for pair in verifier.get("tests") or ())
            if not tests:
                loud(
                    f"load_rl_tasks: {source}:{lineno}: code task {record['id']!r} carries "
                    f"no tests, so its verifier has nothing to run. Escape: rebuild it."
                )
            cls, match, budget = "code", str(verifier.get("match", "exact")), _CODE_BUDGET
            expected = "\n".join(value for _call, value in tests)
            solution = str(record["answer"])
        else:
            loud(
                f"load_rl_tasks: {source}:{lineno}: verifier {verifier!r} is not known. "
                f"Known: 'math', or {{'cls': 'code', 'tests': [...]}}. Escape: rebuild."
            )
        tasks.append(
            Task(
                cls=cls,
                tid=str(record["id"]),
                prompt=prompt,
                expected=expected,
                solution=solution,
                tests=tests,
                match=match,
                budget=budget,
            )
        )
    if not tasks:
        loud(f"load_rl_tasks: {source} has no records. Escape: build the corpus first.")
    return tasks
