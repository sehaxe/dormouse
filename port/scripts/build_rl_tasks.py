#!/usr/bin/env -S uv run --script
"""Build the RLVR task corpus: a JSONL of verifiable tasks, deterministic from a seed.

Why this exists: `post/tasks.py` ships 128 tasks per class for a smoke test, and
the RL loop draws one prompt per step from that pool. That is not a corpus — a
128-task pool repeats every 128 steps, and the SFT corpus is built from the SAME
seed, so an RL run could train on the exact prompts SFT was taught. This builder
emits 2 500 tasks (500 per kind, `--per-kind` to change) with a difficulty
spread, and it PROVES the disjointness by hashing the rendered prompt, not the
id: the project has already been bitten by an id-only check (`test_sft.py`'s
corpus gate is the one that stays green while two corpora share a prompt).

Kinds (five, 500 each by default):

| kind | verifier | what the model must produce |
|---|---|---|
| `arith_chain` | `math` (sympy) | a nested integer expression's exact value |
| `algebra_rational` | `math` (sympy) | `x` in `(p/q)x + r/s = t/u`, exact rational |
| `unit_conversion` | `math` (sympy) | an exact integer/fraction in the target unit |
| `string_list` | `code` (sandbox) | a whole Python function, tested on literal I/O |
| `algorithmic` | `code` (sandbox) | a whole Python function, tested on literal I/O |

Every record is `{id, kind, prompt, answer, verifier, difficulty_hint}`.
`verifier` is `"math"` for the three math kinds and an object
`{"cls": "code", "tests": [[call, repr], ...], "match": "exact"}` for the code
kinds — the six named fields are the schema; the object is how a code task
carries the tests the sandbox needs. `difficulty_hint` is a float in [0, 1]
derived from the generation parameters (chain depth, operand size, fraction
denominators, test count), not read back off a model: the trainer's difficulty
band uses the MEASURED pass rate, and the hint is for reporting and for a
curriculum that does not have to pay for rollouts to sort the pool.

Determinism: every task is `random.Random(seed * 1_000_003 + kind_index * 10_000 + i)`,
the file is written in generation order, and `json.dumps(..., sort_keys=True)`.
Same seed → same bytes → the same sha256 printed at the end. The builder never
touches the network, the clock or the global RNG.

    uv run port/scripts/build_rl_tasks.py                       # 2 500 tasks
    uv run port/scripts/build_rl_tasks.py --per-kind 50 --out /tmp/t.jsonl
    uv run port/scripts/build_rl_tasks.py --no-check-sft        # skip the proof

The SFT proof imports `train.sft._examples` — the ONE list the SFT corpus is
built from (generic QA + the RU bank + `build_tasks(code|math, seed)`). If the
import is unavailable the builder refuses rather than printing a proof it did
not compute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

KINDS = ("arith_chain", "algebra_rational", "unit_conversion", "string_list", "algorithmic")

#: The math prompt's final-answer contract (`verifiers._final_answer` reads it).
MATH_SUFFIX = "Solve. Put the final value on its own last line as 'answer: <value>'.\n\n"

#: The code prompt's contract (the same header `post.tasks` uses).
CODE_HEADER = "Complete this Python function. Reply with the whole function, nothing else.\n\n"

#: Unit pairs the conversion kind draws from: (source, target, factor).
_UNITS = (
    ("km", "m", Fraction(1000)),
    ("m", "cm", Fraction(100)),
    ("cm", "mm", Fraction(10)),
    ("m", "mm", Fraction(1000)),
    ("h", "s", Fraction(3600)),
    ("min", "s", Fraction(60)),
    ("day", "h", Fraction(24)),
    ("week", "day", Fraction(7)),
    ("kg", "g", Fraction(1000)),
    ("g", "mg", Fraction(1000)),
    ("L", "mL", Fraction(1000)),
    ("ft", "in", Fraction(12)),
    ("yd", "ft", Fraction(3)),
    ("mile", "ft", Fraction(5280)),
    # The inverses matter for the level-0 space: 14 pairs x 20 values is fewer
    # prompts than the 167 easy unit tasks 500-per-kind asks for.
    ("m", "km", Fraction(1, 1000)),
    ("cm", "m", Fraction(1, 100)),
    ("mm", "cm", Fraction(1, 10)),
    ("s", "h", Fraction(1, 3600)),
    ("s", "min", Fraction(1, 60)),
    ("h", "day", Fraction(1, 24)),
    ("day", "week", Fraction(1, 7)),
    ("g", "kg", Fraction(1, 1000)),
    ("mg", "g", Fraction(1, 1000)),
    ("mL", "L", Fraction(1, 1000)),
    ("in", "ft", Fraction(1, 12)),
    ("ft", "yd", Fraction(1, 3)),
    ("ft", "mile", Fraction(1, 5280)),
)


def _difficulty(level: int, rng: random.Random) -> float:
    """A [0, 1] hint from the three generation levels (easy / mid / hard)."""
    lo, span = ((0.12, 0.12), (0.42, 0.16), (0.76, 0.19))[level]
    return round(lo + rng.random() * span, 3)


def _math_record(kind: str, seed: int, i: int, prompt: str, answer: str, hint: float) -> dict:
    return {
        "id": f"{kind}-{seed}-{i:04d}",
        "kind": kind,
        "prompt": prompt,
        "answer": str(answer),
        "verifier": "math",
        "difficulty_hint": hint,
    }


def _arith_chain(seed: int, i: int, level: int, salt: int = 0) -> dict:
    rng = random.Random(seed * 1_000_003 + 0 * 10_000 + i + salt * 7_000_003)
    if level == 0:
        a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
        expr = f"({a} + {b}) * {c}"
        value = (a + b) * c
    elif level == 1:
        a, b, c, d = (rng.randint(2, 20) for _ in range(4))
        expr = f"(({a} + {b}) * {c}) - {d}"
        value = (a + b) * c - d
    else:
        a, b, c, d, e, f = (rng.randint(2, 40) for _ in range(6))
        divisor = rng.randint(2, 9)
        expr = f"(((({a} + {b}) * {c}) - {d}) * {e}) // {divisor}"
        value = (((a + b) * c - d) * e) // divisor
    return _math_record(
        "arith_chain", seed, i, MATH_SUFFIX + expr, str(value), _difficulty(level, rng)
    )


def _algebra_rational(seed: int, i: int, level: int, salt: int = 0) -> dict:
    rng = random.Random(seed * 1_000_003 + 1 * 10_000 + i + salt * 7_000_003)
    if level == 0:
        a, x, b = rng.randint(2, 9), rng.randint(-9, 9), rng.randint(-20, 20)
        # No spaces around the coefficient: the SFT corpus's linear-solve kind
        # already renders `{a} * x + {b} = ...`, and a shared prompt is leakage.
        expr = f"{a}x + {b} = {a * x + b}"
        answer = str(x)
    else:
        # (p/q)x + r/s = t/u, with x an exact rational. The level sets the
        # denominators: level 1 stays on halves/thirds, level 2 goes to ninths
        # and a negative coefficient.
        p, q = rng.randint(1, 9), rng.randint(2, 4 if level == 1 else 9)
        r, s = rng.randint(-9, 9), rng.randint(2, 4 if level == 1 else 9)
        num, den = rng.randint(-9, 9), rng.randint(2, 4 if level == 1 else 9)
        if level == 2 and rng.random() < 0.5:
            p = -p
        coeff, const, x = Fraction(p, q), Fraction(r, s), Fraction(num, den)
        rhs = coeff * x + const
        left = f"({p}/{q})x" if p > 0 else f"({p}/{q})x"
        expr = f"{left} + {r}/{s} = {rhs}"
        answer = str(x)
    return _math_record(
        "algebra_rational", seed, i, MATH_SUFFIX + expr, answer, _difficulty(level, rng)
    )


def _unit_conversion(seed: int, i: int, level: int, salt: int = 0) -> dict:
    rng = random.Random(seed * 1_000_003 + 2 * 10_000 + i + salt * 7_000_003)
    src, dst, factor = rng.choice(_UNITS)
    if level == 0:
        value = Fraction(rng.randint(1, 20))
    elif level == 1:
        value = Fraction(rng.randint(1, 12), rng.randint(2, 6))
    else:
        value = Fraction(rng.randint(1, 40), rng.randint(2, 12))
    answer = value * factor
    prompt = (
        f"Convert {value} {src} to {dst}. Give the exact value as an integer or fraction."
    )
    # The math verifier's contract still applies: the final line must be the answer.
    return _math_record("unit_conversion", seed, i, prompt, str(answer), _difficulty(level, rng))


# --------------------------------------------------------------------------- code


def _code_record(seed: int, kind: str, i: int, spec: dict) -> dict:
    """Render one function spec into a `Task`-shaped record, tests included.

    The expected value of each test is computed by EXECUTING the reference
    solution, so the tests and the answer cannot drift into two implementations.
    The verifier then runs the same reference in its sandbox on a sample (and
    every task in the tests), which is the check that the two agree.
    """
    name = spec["name"]
    params = spec["params"]
    doc = spec["doc"]
    body = spec["body"]
    solution = f'def {name}({params}):\n    """{doc}"""\n{body}\n'
    namespace: dict = {}
    exec(solution, namespace)  # noqa: S102 — the builder's own generated source
    tests = []
    for args in spec["cases"]:
        call = f"{name}({', '.join(repr(a) for a in args)})"
        tests.append([call, repr(eval(call, namespace))])  # noqa: S307 — builder-owned
    prompt = (
        f"{CODE_HEADER}def {name}({params}):\n    \"\"\"{doc}\"\"\"\n\n"
        f"Example: {tests[0][0]} -> {tests[0][1]}\n"
    )
    hint = _difficulty(spec["level"], random.Random(seed * 7 + i))
    return {
        "id": f"{kind}-{seed}-{i:04d}",
        "kind": kind,
        "prompt": prompt,
        "answer": solution,
        "verifier": {"cls": "code", "tests": tests, "match": "exact"},
        "difficulty_hint": hint,
    }


def _string_specs(rng: random.Random, level: int) -> dict:
    """One string/list function per id; the variant cycles with the index."""
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    n = 2 + level * 2
    picked = rng.sample(words, min(n, len(words)))
    numbers = [rng.randint(-5, 20) for _ in range(3 + level * 3)]
    nested = [[rng.randint(0, 9) for _ in range(1 + level)] for _ in range(2 + level)]
    variants = [
        {
            "name": "reverse_words",
            "params": "s",
            "doc": "Return the words of s in reverse order, joined by one space.",
            "body": '    return " ".join(reversed(s.split()))',
            "cases": [(" ".join(picked),), ("",)],
        },
        {
            "name": "dedupe",
            "params": "xs",
            "doc": "Return xs with duplicates removed, keeping first-occurrence order.",
            "body": "    return list(dict.fromkeys(xs))",
            "cases": [(numbers,)],
        },
        {
            "name": "rotate_right",
            "params": "xs, k",
            "doc": "Return xs rotated right by k steps; an empty list stays empty.",
            "body": (
                "    if not xs:\n"
                "        return []\n"
                "    k %= len(xs)\n"
                "    return xs[-k:] + xs[:-k]"
            ),
            "cases": [(numbers, 1 + level), ([], 3)],
        },
        {
            "name": "count_char",
            "params": "s, ch",
            "doc": "Return how many times ch occurs in s.",
            "body": "    return s.count(ch)",
            "cases": [(" ".join(picked), picked[0][0]), ("", "a")],
        },
        {
            "name": "capitalize_words",
            "params": "s",
            "doc": "Capitalize the first letter of every word in s, joined by one space.",
            "body": '    return " ".join(w.capitalize() for w in s.split())',
            "cases": [(" ".join(picked),), ("",)],
        },
        {
            "name": "flatten",
            "params": "nested",
            "doc": "Return the elements of the inner lists, in order.",
            "body": "    return [x for sub in nested for x in sub]",
            "cases": [(nested,), ([],)],
        },
        {
            "name": "longest_word",
            "params": "s",
            "doc": "Return the longest word in s (the first one when tied).",
            "body": "    return max(s.split(), key=len)",
            "cases": [(" ".join(w + "x" * j for j, w in enumerate(picked)),)],
        },
        {
            "name": "sum_even",
            "params": "xs",
            "doc": "Return the sum of the even numbers in xs.",
            "body": "    return sum(x for x in xs if x % 2 == 0)",
            "cases": [(numbers,)],
        },
    ]
    return variants


def _algorithmic_specs(rng: random.Random, level: int) -> dict:
    """One simple algorithmic function per id; the variant cycles with the index."""
    k = 1 + level * 3
    variants = [
        {
            "name": "fib",
            "params": "n",
            "doc": "Return the n-th Fibonacci number, with fib(0) = 0 and fib(1) = 1.",
            "body": (
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a"
            ),
            "cases": [(rng.randint(0, 100 * k),), (0,), (1,)],
        },
        {
            "name": "sum_digits",
            "params": "n",
            "doc": "Return the sum of the decimal digits of abs(n).",
            "body": "    return sum(int(ch) for ch in str(abs(n)))",
            "cases": [(rng.randint(1, 10 ** (k + 2)),), (0,)],
        },
        {
            "name": "is_prime",
            "params": "n",
            "doc": "Return True iff n is a prime number.",
            "body": (
                "    if n < 2:\n"
                "        return False\n"
                "    for d in range(2, int(n ** 0.5) + 1):\n"
                "        if n % d == 0:\n"
                "            return False\n"
                "    return True"
            ),
            "cases": [(rng.randint(3, 10 ** (k + 1)),), (1,), (2,)],
        },
        {
            "name": "gcd",
            "params": "a, b",
            "doc": "Return the greatest common divisor of a and b.",
            "body": (
                "    while b:\n"
                "        a, b = b, a % b\n"
                "    return abs(a)"
            ),
            "cases": [
                (rng.randint(1, 10 ** (k + 1)), rng.randint(1, 10 ** (k + 1))),
                (0, rng.randint(1, 10)),
            ],
        },
        {
            "name": "collatz_steps",
            "params": "n",
            "doc": "Return how many steps n takes to reach 1 under the Collatz rule.",
            "body": (
                "    steps = 0\n"
                "    while n != 1:\n"
                "        n = n // 2 if n % 2 == 0 else 3 * n + 1\n"
                "        steps += 1\n"
                "    return steps"
            ),
            "cases": [(rng.randint(2, 10 ** (k + 1)),), (1,)],
        },
        {
            "name": "max_subarray",
            "params": "xs",
            "doc": "Return the largest sum of a non-empty contiguous slice of xs.",
            "body": (
                "    best = cur = xs[0]\n"
                "    for x in xs[1:]:\n"
                "        cur = max(x, cur + x)\n"
                "        best = max(best, cur)\n"
                "    return best"
            ),
            "cases": [([rng.randint(-9, 9) for _ in range(3 + k)],)],
        },
    ]
    return variants


def _code_task(seed: int, kind: str, i: int, level: int, salt: int = 0) -> dict:
    rng = random.Random(
        seed * 1_000_003 + (3 if kind == "string_list" else 4) * 10_000 + i + salt * 7_000_003
    )
    specs = _string_specs(rng, level) if kind == "string_list" else _algorithmic_specs(rng, level)
    spec = dict(specs[i % len(specs)])
    spec["level"] = level
    return _code_record(seed, kind, i, spec)


# --------------------------------------------------------------------------- corpus


def _make(kind: str, seed: int, i: int, level: int, salt: int) -> dict:
    if kind == "arith_chain":
        return _arith_chain(seed, i, level, salt)
    if kind == "algebra_rational":
        return _algebra_rational(seed, i, level, salt)
    if kind == "unit_conversion":
        return _unit_conversion(seed, i, level, salt)
    return _code_task(seed, kind, i, level, salt)


def build_corpus(seed: int, per_kind: int = 500) -> list[dict]:
    """`per_kind` tasks of each of the five kinds, deterministic from `seed`.

    A generated prompt can repeat (a small operand space); a repeated prompt is
    a second TASK wearing the same question, so the retry loop re-draws with a
    salt until the RENDERED prompt is new. The retry is deterministic, so the
    corpus is still a pure function of `(seed, per_kind)`.
    """
    if per_kind < 0:
        raise SystemExit(f"per-kind must be >= 0, got {per_kind}")
    out: list[dict] = []
    seen: set[str] = set()
    for kind in KINDS:
        for i in range(per_kind):
            level = i % 3
            for salt in range(10_000):
                record = _make(kind, seed, i, level, salt)
                digest = hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    out.append(record)
                    break
            else:  # pragma: no cover — the spaces are far larger than the counts
                raise SystemExit(
                    f"could not draw a unique prompt for {kind} i={i} in 10 000 salts; "
                    f"lower --per-kind or widen the generator"
                )
    return out


def prompt_hashes(prompts) -> set[str]:
    """SHA-256 of each RENDERED prompt — the only disjointness key that holds."""
    return {hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts}


def sft_prompts(seed: int, n_tasks: int) -> list[str]:
    """The instructions `build_sft_corpus` renders for this seed (reused, not copied)."""
    try:
        from dormouse.train.sft import _examples
    except ImportError as exc:  # pragma: no cover — a broken install is not a proof
        raise SystemExit(
            "cannot import dormouse.train.sft._examples, so the SFT disjointness "
            "proof cannot be computed. Escape: run inside the port's environment "
            "(uv run), or --no-check-sft to skip the proof (NOT recommended)."
        ) from exc
    return [instruction for instruction, _answer in _examples(seed, n_tasks)]


def assert_disjoint(records: list[dict], sft_seed: int, sft_tasks: int) -> dict:
    """Refuse a corpus whose rendered prompts collide — with itself or with SFT."""
    rl = prompt_hashes(rec["prompt"] for rec in records)
    if len(rl) != len(records):
        dupes = len(records) - len(rl)
        raise SystemExit(
            f"corpus has {dupes} duplicate rendered prompt(s) — ids are not unique tasks"
        )
    sft = prompt_hashes(sft_prompts(sft_seed, sft_tasks))
    overlap = rl & sft
    if overlap:
        example = next(
            rec["prompt"]
            for rec in records
            if prompt_hashes([rec["prompt"]]).pop() in overlap
        )
        raise SystemExit(
            f"{len(overlap)} RL prompt(s) are also in the SFT corpus (seed {sft_seed}). "
            f"Escape: change --seed; first collision: {example[:120]!r}"
        )
    return {"rl_prompts": len(rl), "sft_prompts": len(sft), "overlap": 0}


def _verify_reference(record: dict) -> None:
    """The reference answer must pass the verifier it will be scored by."""
    from dormouse.post.verifiers import verify

    verifier = record["verifier"]
    if verifier == "math":
        task = {"cls": "math", "expected": record["answer"], "budget": 0}
        verdict = verify(task, f"answer: {record['answer']}")
    else:
        task = {
            "cls": "code",
            "tid": record["id"],
            "tests": tuple(tuple(t) for t in verifier["tests"]),
            "match": verifier["match"],
            "budget": 0,
        }
        verdict = verify(task, record["answer"])
    if not verdict.passed:
        raise SystemExit(f"reference answer for {record['id']} does not verify: {verdict.reason}")


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20261023, help="the corpus seed")
    parser.add_argument("--per-kind", type=int, default=500, help="tasks per kind (5 kinds)")
    parser.add_argument(
        "--out", type=Path, default=root / "data_local" / "rl_tasks.jsonl", help="JSONL path"
    )
    parser.add_argument("--sft-seed", type=int, default=1, help="the SFT run's seed for the proof")
    parser.add_argument("--sft-tasks", type=int, default=32, help="its n_tasks")
    parser.add_argument(
        "--no-check-sft", action="store_true", help="skip the SFT disjointness proof"
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=5,
        help="reference answers per code kind to run through the sandbox (0 = none)",
    )
    parser.add_argument("--verify-all", action="store_true", help="verify every reference answer")
    args = parser.parse_args(argv)

    records = build_corpus(args.seed, args.per_kind)
    if args.no_check_sft:
        stats = {
            "rl_prompts": len(prompt_hashes(r["prompt"] for r in records)),
            "sft_prompts": 0,
            "overlap": 0,
        }
    else:
        stats = assert_disjoint(records, args.sft_seed, args.sft_tasks)

    # The reference answers are built from the same source they are tested
    # against; running a sample through the verifier is the check that the
    # render/exec round trip did not drift.
    if args.verify_all:
        to_verify = records
    else:
        # Both halves of the corpus, not just code: a math reference that does
        # not parse would make `pass@1` a constant zero just as surely.
        math_sample = [r for r in records if r["verifier"] == "math"][: args.verify_sample]
        code_sample = [r for r in records if r["verifier"] != "math"][: 2 * args.verify_sample]
        to_verify = math_sample + code_sample
    for record in to_verify:
        _verify_reference(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    args.out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    kinds = {kind: sum(1 for r in records if r["kind"] == kind) for kind in KINDS}
    easy = sum(1 for r in records if r["difficulty_hint"] < 0.34)
    mid = sum(1 for r in records if 0.34 <= r["difficulty_hint"] <= 0.66)
    hard = sum(1 for r in records if r["difficulty_hint"] > 0.66)
    print(f"wrote {len(records)} tasks -> {args.out}")
    print("kinds: " + " ".join(f"{k}={v}" for k, v in kinds.items()))
    print(f"difficulty: easy={easy} mid={mid} hard={hard}")
    print(
        f"prompt hashes: rl={stats['rl_prompts']} sft={stats['sft_prompts']} "
        f"overlap={stats['overlap']} (sha256 of the RENDERED prompt)"
    )
    print(f"verified reference answers: {len(to_verify)}/{len(records)}")
    print(f"corpus sha256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
