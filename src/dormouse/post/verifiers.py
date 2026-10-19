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
| `math` | sympy equivalence of the extracted FINAL answer | algebra, not string match |
| `noop` | **LOUD refusal of every task** (the do-not-run class) | raises [`LoudError`] |

[`verify`] is pure and deterministic: the same `(task, response)` always yields
the same `Verdict`, and the `reason` never carries the raw response — a byte
budget and a log line read both, and an unlogged verifier is the ADR-0019
failure mode.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import sympy

from ..instrument import loud

#: The exec jail's wall clock and CPU limit, in seconds (2602.11210's shape).
TIMEOUT_S = 5.0

__all__ = ["TIMEOUT_S", "Verdict", "sympy_equivalent", "verify"]


@dataclass(frozen=True)
class Verdict:
    """One verifier decision. `budget_used` is the fraction of the task's byte
    budget the response spent (the `correct@budget` metric's input), clamped to
    `[0, 1]`; `reason` is task-side only and never the raw response."""

    passed: bool
    reason: str
    budget_used: float


def sympy_equivalent(expected: str, got: str) -> bool:
    """True iff `expected − got` simplifies to 0 (`2/4` == `0.5` == `1/2`)."""
    a = sympy.sympify(expected)
    b = sympy.sympify(got)
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

#: The last `answer: X` / `answer = X` line is the final answer; else the last
#: non-empty line, so a bare number still scores (2606.30420's exact-match shape).
_ANSWER = re.compile(r"answer\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_PARSE_ERRORS = (sympy.SympifyError, SyntaxError, TypeError, ValueError)


def _final_answer(response: str) -> str | None:
    hits = _ANSWER.findall(response)
    if hits:
        return hits[-1].strip().rstrip(".")
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    return lines[-1].rstrip(".") if lines else None


def _verify_math(task: dict, response: str, used: float) -> Verdict:
    got = _final_answer(response)
    if got is None or not got:
        return Verdict(False, "no final answer found in the response", used)
    try:
        ok = sympy_equivalent(str(task["expected"]), got)
    except _PARSE_ERRORS:
        return Verdict(False, "final answer is not a parseable expression", used)
    reason = "sympy-equivalent" if ok else "not equivalent to the expected answer"
    return Verdict(bool(ok), reason, used)


# --------------------------------------------------------------------------- code

#: The static anti-reward-hack screen (2601.20103's class). A response that names
#: any of these is refused BEFORE it runs — the sandbox below is the second layer,
#: not the first.
_FORBIDDEN: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsocket\b"), "network access"),
    (
        re.compile(r"\b(?:urllib|requests|httpx|ftplib|smtplib|telnetlib|asyncio)\b"),
        "network access",
    ),
    (re.compile(r"\bsubprocess\b|\bos\.(?:system|popen|spawn\w*|exec\w*|fork)\b"), "process spawn"),
    (re.compile(r"\bctypes\b"), "native code"),
    (re.compile(r"__import__|\bimportlib\b"), "dynamic import"),
    (re.compile(r"/etc/passwd"), "sensitive path"),
    (re.compile(r"open\s*\(\s*[\"']/(?!tmp)"), "absolute-path read"),
)

#: Child-side jail: a new session, no CPU/address-space blowups, no forks, and a
#: preamble that unimports the network and refuses absolute-path reads. The
#: static screen catches the plain cases; this catches a string-built ones.
_SANDBOX_PREAMBLE = """\
import sys as _sys
for _m in ("socket", "ssl", "http", "urllib", "ftplib", "smtplib", "asyncio",
           "ctypes", "subprocess"):
    _sys.modules[_m] = None
import builtins as _builtins
_open = _builtins.open
def _sandbox_open(file, *args, **kwargs):
    path = str(file)
    if path.startswith("/") and not path.startswith("/tmp/"):
        raise PermissionError("sandbox: absolute-path reads outside the workdir are refused")
    return _open(file, *args, **kwargs)
_builtins.open = _sandbox_open
"""


class _Timeout(Exception):
    """The jailed program exceeded [`TIMEOUT_S`]."""


class _SandboxError(Exception):
    """The jailed program exited non-zero; carries the code, never the stderr."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _sandbox_limits() -> None:  # runs in the child, before exec
    os.setsid()
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (int(TIMEOUT_S), int(TIMEOUT_S)))
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def _extract_code(response: str) -> str:
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    return fence.group(1) if fence else response


def _harness(tests) -> str:
    lines = ["_results = []"]
    for call, _ in tests:
        lines.append(f"_results.append(repr({call}))")
    lines.append("print('\\n'.join(_results))")
    return "\n".join(lines) + "\n"


def _run_sandbox(program: str, *, timeout: float) -> str:
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", "-"],
                input=program,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": "/usr/bin:/bin", "HOME": workdir},
                preexec_fn=_sandbox_limits,
            )
        except subprocess.TimeoutExpired as exc:
            raise _Timeout from exc
    if proc.returncode != 0:
        raise _SandboxError(proc.returncode)
    return proc.stdout


def _screen_code(response: str) -> str | None:
    for pat, label in _FORBIDDEN:
        if pat.search(response):
            return label
    return None


def _verify_code(task: dict, response: str, used: float) -> Verdict:
    bad = _screen_code(response)
    if bad is not None:
        return Verdict(False, f"refused: {bad} is not allowed in a sandboxed verifier", used)
    tests = tuple(task.get("tests") or ())
    if not tests:
        return Verdict(False, "refused: the task carries no tests to run", used)
    program = _SANDBOX_PREAMBLE + "\n" + _extract_code(response) + "\n" + _harness(tests)
    try:
        out = _run_sandbox(program, timeout=TIMEOUT_S)
    except _Timeout:
        return Verdict(False, f"timed out after {TIMEOUT_S:g}s in the sandbox", used)
    except _SandboxError as exc:
        return Verdict(False, f"program exited with code {exc.code} under the sandbox", used)
    lines = out.strip().splitlines() if out.strip() else []
    if len(lines) != len(tests):
        return Verdict(False, f"expected {len(tests)} result lines, got {len(lines)}", used)
    match = str(task.get("match", "exact"))
    for (call, expected), got in zip(tests, lines, strict=True):
        if not _answer_matches(str(expected), got, match):
            return Verdict(False, f"test {call!r} failed (expected {expected!r})", used)
    return Verdict(True, f"all {len(tests)} tests passed", used)


def _answer_matches(expected: str, got: str, match: str) -> bool:
    if match == "regex":
        return re.search(expected, got) is not None
    return got.strip() == expected.strip()
