"""Deterministic verifiers keyed by task class — the reward, and nothing else.

Read from `docs/research/2026-10-15-agentic-post-training.md` §2.1
("verifier-first, everywhere"; code execution needs "a process jail, rlimits" —
2602.11210's container-free shape) and §2.3 (a learned/LLM judge at this scale is
the failure mode, not the signal — that is the plan's measured-later node), plus
`docs/architecture/post-training.md`'s "Reward shape for token thrift" node: the
default grader at our scale is the deterministic verifier.

| class | signal | shape |
|---|---|---|
| `code` | subprocess exec, 5 s, rlimits, tempdir cwd, no-network preamble | per-test I/O |
| `math` | whitelisted parse; sympy equivalence of the FINAL answer | algebra, not a string |
| `noop` | **LOUD refusal of every task** (the do-not-run class) | raises [`LoudError`] |

**The model's text is hostile input in both classes.** The math answer used to go
through `sympy.sympify`, which *evaluates* the string in this process: an answer
like `__import__('os').system('...')` was arbitrary code execution in the
trainer, and `9**9**9` was a hang. [`_safe_expr`] now parses with a whitelist —
numbers, `+ - * / ** ( )`, a fixed function set, `pi`/`E` — and bounds the work
([`MAX_EXPR_CHARS`], [`MAX_EXPR_NODES`], [`MAX_EXPONENT`], [`MAX_MAGNITUDE`]);
everything else is a refusal whose reason names the offending token. The code
class is screened statically by an AST **allowlist** (imports limited to `math`,
calls limited to the standard names an algorithmic solution needs, no dunder
attributes, no `open`/`eval`/`exec`/`compile`/`__import__`/`getattr` and no
`os`/`posix`/`pathlib`/`io`/`subprocess`/`shutil`/`socket`/`ctypes`/`importlib`/
`sys`), and the child process keeps a second line: the forbidden modules are
removed from `sys.modules` and refused by an import hook, `open` refuses all
file I/O, and the submission executes with an allowlist of builtins.

[`verify`] is pure and deterministic: the same `(task, response)` always yields
the same `Verdict`, and the `reason` never carries the raw response — a byte
budget and a log line read both, and an unlogged verifier is the ADR-0019
failure mode. A refusal names the token, never the response text.

It is also **TOTAL**: any byte string is a verdict, never an exception. One
23-byte answer (`(2**1000)**1000`) used to raise out of the old magnitude check
— Python 3.12 refuses `str()` above 4 300 digits — and one rollout answer used
to spend 5.45 s in `simplify`. Every numeric node is now bounded OUTSIDE before
it is computed ([`MAX_RESULT_BITS`]), every exception in the numeric path is a
COUNTED refusal ([`COUNTERS`]), and a wall-clock deadline ([`MATH_DEADLINE_S`])
is the second line. The rollout caller keeps a third one (`rollout_verify_error`
in `post/rollouts.py`): even a verifier bug fails one sample, not the batch.
"""

from __future__ import annotations

import ast
import math
import re
import time
from dataclasses import dataclass

import sympy

from ..instrument import Counters, LoudError, loud
from .sandbox import TIMEOUT_S, run_code

#: The math parser's work bounds. These are refusals, not truncations: an answer
#: over a limit is REJECTED and the reason says which limit it crossed.
#: `9**9**9` is `9**(9**9)`: its exponent is 387 420 489 and the result would be
#: a 369-million-digit integer, so the trainer would hang inside sympy.
MAX_EXPR_CHARS = 256
MAX_EXPR_NODES = 64
MAX_EXPONENT = 1000
MAX_MAGNITUDE = 1e100

#: The result of every intermediate node, in bits. `333 = ceil(log2(1e100))`:
#: `10**100` passes, `2**1000` does not. The check runs on the POWER'S OPERANDS
#: before the power is computed (`_check_pow_size`), so `(2**1000)**1000` is
#: refused at the inner `2**1000` in microseconds. The old path let the inner
#: value form, then asked sympy for `2**10^6` (301 030 digits), and a 3-deep
#: form asked for `2**10^9` — measured 5.45 s, and the 4-deep form is a ~125 GB
#: integer. With this bound the worst allowed expression measured 0.31 ms on
#: this box (2026-10-19: `sin(10**99)`, `sin(1)**999`, `pi**100`, `10**100`,
#: all at the bound). That is the bound stated in the module contract; the
#: deadline below is a regression alarm, not the bound.
MAX_RESULT_BITS = 333

#: One answer's wall-clock budget. Checked before and after every node
#: evaluation and after the numeric difference; a parser/sympy regression that
#: slips past the operand bounds still returns a refusal promptly.
MATH_DEADLINE_S = 0.05

#: COUNTED numeric outcomes: `verify_refused` is every hostile/over-bound final
#: answer (a failed rollout, by design), `verify_error` is an exception the
#: verifier itself had to absorb (a bug, counted so a log reader sees it).
COUNTERS = Counters()

__all__ = [
    "COUNTERS",
    "MATH_DEADLINE_S",
    "MAX_EXPR_CHARS",
    "MAX_EXPR_NODES",
    "MAX_EXPONENT",
    "MAX_MAGNITUDE",
    "MAX_RESULT_BITS",
    "TIMEOUT_S",
    "Verdict",
    "sympy_equivalent",
    "verify",
]


@dataclass(frozen=True)
class Verdict:
    """One verifier decision. `budget_used` is the fraction of the task's byte
    budget the response spent (the `correct@budget` metric's input), clamped to
    `[0, 1]`; `reason` is task-side only and never the raw response."""

    passed: bool
    reason: str
    budget_used: float


class _Refused(Exception):
    """A final answer outside the numeric whitelist, or over a work bound.

    `.token` is the shortest spelling of the cause the reason can carry without
    echoing the response; `why` names the rule. The exception never escapes
    [`verify`] — a hostile answer is a failed rollout, not a dead trainer.
    """

    def __init__(self, token: str, why: str) -> None:
        super().__init__(f"{token}: {why}")
        self.token = token
        self.why = why


def sympy_equivalent(expected: str, got: str) -> bool:
    """True iff `expected − got` is exactly 0 (`2/4` == `0.5` == `1/2`).

    `got` is model text: it is parsed by the whitelist in [`_safe_expr`] and a
    refusal returns False. `expected` is the task's own reference, so a bad one
    is a task defect and stops the run LOUDLY rather than scoring every rollout
    wrong.
    """
    deadline = time.monotonic() + MATH_DEADLINE_S
    try:
        b = _safe_expr(got, deadline)
    except _Refused:
        COUNTERS.count("verify_refused")
        return False
    except Exception:  # model text must never raise out of the reward path
        COUNTERS.count("verify_error")
        return False
    a = _expected_expr(expected, deadline)
    try:
        return _equivalent(a, b, deadline)
    except _Refused:
        COUNTERS.count("verify_refused")
        return False


def verify(task: dict, response: str | bytes) -> Verdict:
    """Dispatch on `task["cls"]`; see the module docstring for the three classes."""
    if isinstance(response, bytes):
        response = response.decode("utf-8", "replace")
    cls = task.get("cls")
    used = _budget_used(response, int(task.get("budget", 0) or 0))
    if cls == "noop" or task.get("kind") == "noop":
        loud(
            f"noop verifier refuses task {task.get('tid', '?')!r}: this class is "
            "do-not-run, so there is no reward to compute"
        )
    if cls == "math":
        return _verify_math(task, response, used)
    if cls == "code":
        return _verify_code(task, response, used)
    loud(f"verify: unknown task class {cls!r} (known: code, math, noop)")


def _budget_used(response: str, budget: int) -> float:
    if budget <= 0:
        return 0.0
    return min(len(response.encode("utf-8")) / budget, 1.0)


# --------------------------------------------------------------------------- math

#: The last `answer: X` / `answer = X` line or the last `\boxed{X}` is the final
#: answer; else the last non-empty line, so a bare number still scores
#: (2606.30420's exact-match shape).
_ANSWER = re.compile(r"answer\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")

#: The whole arithmetic surface a numeric final answer needs. Anything not here
#: — a name, an attribute, a string, a comprehension, a lambda — is refused.
_FUNCS = {
    "sqrt": sympy.sqrt,
    "abs": sympy.Abs,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "log": sympy.log,
    "exp": sympy.exp,
}
_CONSTS = {"pi": sympy.pi, "E": sympy.E}
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)


def _final_answer(response: str) -> str | None:
    hits = [(m.start(), m.group(1)) for m in _ANSWER.finditer(response)]
    hits += [(m.start(), m.group(1)) for m in _BOXED.finditer(response)]
    if hits:
        return max(hits)[1].strip().rstrip(".")
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    return lines[-1].rstrip(".") if lines else None


def _short(value: object, n: int = 32) -> str:
    """A bounded spelling of `value`; TOTAL, because it formats hostile values.

    Python 3.12 raises `ValueError` on `str(int)` above 4 300 digits — that was
    the R3-1 crash: the magnitude branch formatted the very integer it had just
    refused. A value too big to spell is named by its size instead.
    """
    try:
        text = str(value)
    except (ValueError, OverflowError):
        if isinstance(value, (int, sympy.Integer)):
            text = f"<{int(value).bit_length()}-bit integer>"
        else:
            text = f"<{type(value).__name__}>"
    return text if len(text) <= n else text[: n - 3] + "..."


def _token(node: ast.AST) -> str:
    try:
        return _short(ast.unparse(node))
    except Exception:  # parser internals on a hostile tree; the class name is enough
        return type(node).__name__


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise _Refused(
            f">{MATH_DEADLINE_S * 1000:.0f} ms",
            "the arithmetic exceeded the wall-clock work deadline",
        )


def _bounded(value: sympy.Expr) -> None:
    """Refuse a node OUTSIDE the cheap numeric band, before it can grow.

    Integer/rational magnitudes are measured in bits (no `float()` on a huge
    int, no `str()` at all); everything else is evalf'd through `complex()`.
    Totality is the point: a conversion failure is a refusal naming the size or
    the type, never an exception out of the verifier.
    """
    if isinstance(value, sympy.Integer):
        if int(value).bit_length() > MAX_RESULT_BITS:
            raise _Refused(
                _short(value), f"the integer exceeds the {MAX_RESULT_BITS}-bit work band"
            )
        return
    if isinstance(value, sympy.Rational):
        if max(int(value.p).bit_length(), int(value.q).bit_length()) > MAX_RESULT_BITS:
            raise _Refused(
                _short(value), f"the fraction exceeds the {MAX_RESULT_BITS}-bit work band"
            )
        return
    if isinstance(value, sympy.Float):
        magnitude = abs(float(value))
    else:
        try:
            magnitude = abs(complex(value))
        except (TypeError, ValueError, OverflowError):
            magnitude = math.inf
    if not math.isfinite(magnitude):
        raise _Refused(_short(value), "is not a finite number")
    if magnitude > MAX_MAGNITUDE:
        raise _Refused(_short(value), f"magnitude exceeds the {MAX_MAGNITUDE:g} work limit")


def _check_pow_size(base: sympy.Expr, exponent: sympy.Expr) -> None:
    """Refuse a power whose RESULT would leave the band, BEFORE computing it.

    The exponent bound is per operation, so it cannot see composition: a 4-deep
    pow with every exponent = 1000 asks for `2**10^12`. The result's size is the
    base's log2 times the exponent — arithmetic on floats, no big integer is
    formed. `|log2|` also covers a base below 1, whose RESULT is tiny but whose
    exact fraction would grow a huge denominator (`0.5**-1000` = 2**1000).
    """
    try:
        base_f = abs(float(base))
    except (TypeError, ValueError, OverflowError):
        return  # a symbolic base is bounded by _bounded's evalf; exponent <= 1000
    if base_f == 0.0 or not math.isfinite(base_f):
        return
    try:
        bits = abs(math.log2(base_f)) * abs(float(exponent))
    except (ValueError, OverflowError):
        bits = math.inf
    if bits > MAX_RESULT_BITS:
        raise _Refused(
            _short(exponent), f"the power would exceed the {MAX_RESULT_BITS}-bit work band"
        )


def _safe_expr(text: str, deadline: float | None = None) -> sympy.Expr:
    """Parse ONE numeric expression under the math whitelist, or raise [`_Refused`]."""
    text = text.strip()
    if not text:
        raise _Refused("''", "the final answer is empty")
    if len(text) > MAX_EXPR_CHARS:
        raise _Refused(f"{len(text)} characters", f"exceeds the {MAX_EXPR_CHARS}-character limit")
    if deadline is None:
        deadline = time.monotonic() + MATH_DEADLINE_S
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as exc:
        # `ValueError` is the null-byte lexical error: binary garbage is data,
        # not a crash (`ast.parse` raises it, not `SyntaxError`).
        msg = getattr(exc, "msg", str(exc))
        raise _Refused(_short(text), f"is not a valid expression ({msg})") from exc
    nodes = sum(1 for _ in ast.walk(tree))
    if nodes > MAX_EXPR_NODES:
        raise _Refused(f"{nodes} nodes", f"exceeds the {MAX_EXPR_NODES}-node limit")
    return _eval_node(tree.body, deadline)


def _eval_node(node: ast.AST, deadline: float) -> sympy.Expr:
    """Evaluate one node, bounding its value and the clock on the way out."""
    _check_deadline(deadline)
    value = _eval_node_raw(node, deadline)
    _bounded(value)
    _check_deadline(deadline)
    return value


def _eval_node_raw(node: ast.AST, deadline: float) -> sympy.Expr:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _Refused(_token(node), f"constant {type(value).__name__} is not a number")
        if isinstance(value, float) and not math.isfinite(value):
            raise _Refused(_token(node), "a constant must be finite")
        return sympy.Integer(value) if isinstance(value, int) else sympy.Float(repr(value))
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise _Refused(node.id, "is not a name in the math whitelist")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_node(node.operand, deadline)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        return _eval_binop(node, deadline)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise _Refused(_token(node.func), "is not a function in the math whitelist")
        if node.keywords or len(node.args) != 1 or isinstance(node.args[0], ast.Starred):
            raise _Refused(node.func.id, "takes exactly one positional argument")
        return _FUNCS[node.func.id](_eval_node(node.args[0], deadline))
    raise _Refused(_token(node), f"{type(node).__name__} is not allowed in a final answer")


def _eval_binop(node: ast.BinOp, deadline: float) -> sympy.Expr:
    if not isinstance(node.op, _BINOPS):
        raise _Refused(type(node.op).__name__, "is not an operator in the math whitelist")
    left = _eval_node(node.left, deadline)
    right = _eval_node(node.right, deadline)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        if right == 0:
            raise _Refused("/", "division by zero")
        return left / right
    return _eval_pow(left, right, deadline)


def _eval_pow(base: sympy.Expr, exponent: sympy.Expr, deadline: float) -> sympy.Expr:
    if exponent.is_real is False:
        raise _Refused("**", "the exponent must be a real number")
    try:
        magnitude = abs(float(exponent))
    except (TypeError, ValueError, OverflowError):
        raise _Refused(_short(exponent), "the exponent is not a finite number") from None
    if not math.isfinite(magnitude) or magnitude > MAX_EXPONENT:
        raise _Refused(_short(exponent), f"exceeds the {MAX_EXPONENT} exponent limit")
    _check_pow_size(base, exponent)
    value = base**exponent
    _check_deadline(deadline)
    return value


def _equivalent(a: sympy.Expr, b: sympy.Expr, deadline: float) -> bool:
    """Exact equivalence, with a cheap numeric exit BEFORE `simplify`.

    `simplify` on a symbolic power is the measured 5.45 s path
    (`sin(1)**999` alone spent > 90 s in it). A numeric difference that is not
    exactly zero already proves the answers differ, so `simplify` runs only on
    the numerically-zero case.
    """
    diff = a - b
    try:
        magnitude = abs(complex(diff))
    except (TypeError, ValueError, OverflowError):
        raise _Refused(_short(diff), "the difference is not a finite number") from None
    _check_deadline(deadline)
    if not math.isfinite(magnitude):
        raise _Refused(_short(diff), "the difference is not finite")
    if magnitude:
        return False
    try:
        return bool(sympy.simplify(diff) == 0)
    except Exception:  # a sympy bug is a "not equivalent", never a dead rollout
        COUNTERS.count("verify_error")
        return False


def _expected_expr(expected: object, deadline: float | None = None) -> sympy.Expr:
    try:
        return _safe_expr(str(expected), deadline)
    except _Refused as exc:
        loud(
            f"task reference {_short(expected)!r} is outside the verifier's math "
            f"whitelist ({exc.token}: {exc.why}). Escape: fix the task's `expected` "
            "to a numeric expression; the verifier will not guess."
        )


def _verify_math(task: dict, response: str, used: float) -> Verdict:
    got = _final_answer(response)
    if got is None or not got:
        return Verdict(False, "no final answer found in the response", used)
    deadline = time.monotonic() + MATH_DEADLINE_S
    try:
        b = _safe_expr(got, deadline)
        a = _expected_expr(task["expected"], deadline)
        ok = _equivalent(a, b, deadline)
    except _Refused as exc:
        COUNTERS.count("verify_refused")
        return Verdict(False, f"refused: {exc.token!r} {exc.why}", used)
    except LoudError:
        raise  # a bad task reference is task data: LOUD, not a failed rollout
    except Exception as exc:  # the numeric path's last line: model text never raises
        COUNTERS.count("verify_error")
        return Verdict(
            False,
            f"refused: {_short(got)!r} raised {type(exc).__name__} in the numeric path",
            used,
        )
    reason = "sympy-equivalent" if ok else "not equivalent to the expected answer"
    return Verdict(bool(ok), reason, used)


# --------------------------------------------------------------------------- code


def _verify_code(task: dict, response: str, used: float) -> Verdict:
    """The jail lives in `post/sandbox.py` (static allowlist + runtime guard)."""
    tests = tuple(task.get("tests") or ())
    if not tests:
        return Verdict(False, "refused: the task carries no tests to run", used)
    reason = run_code(response, tests, str(task.get("match", "exact")))
    if reason is None:
        return Verdict(True, f"all {len(tests)} tests passed", used)
    return Verdict(False, reason, used)
