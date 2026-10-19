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
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass

import sympy

from ..instrument import loud
from .sandbox import TIMEOUT_S, run_code

#: The math parser's work bounds. These are refusals, not truncations: an answer
#: over a limit is REJECTED and the reason says which limit it crossed.
#: `9**9**9` is `9**(9**9)`: its exponent is 387 420 489 and the result would be
#: a 369-million-digit integer, so the trainer would hang inside sympy.
MAX_EXPR_CHARS = 256
MAX_EXPR_NODES = 64
MAX_EXPONENT = 1000
MAX_MAGNITUDE = 1e100

__all__ = [
    "MAX_EXPR_CHARS",
    "MAX_EXPR_NODES",
    "MAX_EXPONENT",
    "MAX_MAGNITUDE",
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
    """True iff `expected − got` simplifies to 0 (`2/4` == `0.5` == `1/2`).

    `got` is model text: it is parsed by the whitelist in [`_safe_expr`] and a
    refusal returns False. `expected` is the task's own reference, so a bad one
    is a task defect and stops the run LOUDLY rather than scoring every rollout
    wrong.
    """
    try:
        b = _safe_expr(got)
    except _Refused:
        return False
    a = _expected_expr(expected)
    return bool(sympy.simplify(a - b) == 0)


def verify(task: dict, response: str) -> Verdict:
    """Dispatch on `task["cls"]`; see the module docstring for the three classes."""
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
    text = str(value)
    return text if len(text) <= n else text[: n - 3] + "..."


def _token(node: ast.AST) -> str:
    try:
        return _short(ast.unparse(node))
    except Exception:  # parser internals on a hostile tree; the class name is enough
        return type(node).__name__


def _safe_expr(text: str) -> sympy.Expr:
    """Parse ONE numeric expression under the math whitelist, or raise [`_Refused`]."""
    text = text.strip()
    if not text:
        raise _Refused("''", "the final answer is empty")
    if len(text) > MAX_EXPR_CHARS:
        raise _Refused(f"{len(text)} characters", f"exceeds the {MAX_EXPR_CHARS}-character limit")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise _Refused(_short(text), f"is not a valid expression ({exc.msg})") from exc
    nodes = sum(1 for _ in ast.walk(tree))
    if nodes > MAX_EXPR_NODES:
        raise _Refused(f"{nodes} nodes", f"exceeds the {MAX_EXPR_NODES}-node limit")
    value = _eval_node(tree.body)
    _check_number(value)
    return value


def _eval_node(node: ast.AST) -> sympy.Expr:
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
        operand = _eval_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        return _eval_binop(node)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise _Refused(_token(node.func), "is not a function in the math whitelist")
        if node.keywords or len(node.args) != 1 or isinstance(node.args[0], ast.Starred):
            raise _Refused(node.func.id, "takes exactly one positional argument")
        return _FUNCS[node.func.id](_eval_node(node.args[0]))
    raise _Refused(_token(node), f"{type(node).__name__} is not allowed in a final answer")


def _eval_binop(node: ast.BinOp) -> sympy.Expr:
    if not isinstance(node.op, _BINOPS):
        raise _Refused(type(node.op).__name__, "is not an operator in the math whitelist")
    left = _eval_node(node.left)
    right = _eval_node(node.right)
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
    return _eval_pow(left, right)


def _eval_pow(base: sympy.Expr, exponent: sympy.Expr) -> sympy.Expr:
    if exponent.is_real is False:
        raise _Refused("**", "the exponent must be a real number")
    try:
        magnitude = abs(float(exponent))
    except (TypeError, ValueError, OverflowError):
        raise _Refused(_short(exponent), "the exponent is not a finite number") from None
    if not math.isfinite(magnitude) or magnitude > MAX_EXPONENT:
        raise _Refused(_short(exponent), f"exceeds the {MAX_EXPONENT} exponent limit")
    return base**exponent


def _check_number(value: sympy.Expr) -> None:
    if value.is_real is False:
        raise _Refused(_short(value), "the final answer is not a real number")
    try:
        magnitude = abs(complex(value))
    except (TypeError, ValueError, OverflowError):
        raise _Refused(_short(value), "the final answer is not a finite number") from None
    if not math.isfinite(magnitude) or magnitude > MAX_MAGNITUDE:
        raise _Refused(_short(value), f"magnitude exceeds the {MAX_MAGNITUDE:g} work limit")


def _expected_expr(expected: object) -> sympy.Expr:
    try:
        return _safe_expr(str(expected))
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
    try:
        b = _safe_expr(got)
    except _Refused as exc:
        return Verdict(False, f"refused: {exc.token!r} {exc.why}", used)
    a = _expected_expr(task["expected"])
    ok = bool(sympy.simplify(a - b) == 0)
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
