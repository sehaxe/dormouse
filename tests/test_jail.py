"""Gates for the code jail: the AST allowlist, and the runtime guard behind it.

The old jail was a regex blacklist plus a preamble that unimported a handful of
modules. `os.open`, `pathlib` and `getattr(__builtins__, 'open')` walked through
it — the first two import a module the blacklist never named, the third reaches a
builtin the string screen cannot see. The screen is now an AST ALLOWLIST: a Name
must be a standard name an algorithmic solution needs or a name the submission
binds itself; calls, attributes and imports are checked the same way. The child
process keeps a second line (import hook, `open` denial, builtin allowlist), and
the last test here drives that line DIRECTLY, past the static screen, so it is
proven rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dormouse.post import sandbox as S
from dormouse.post.verifiers import verify


def _task(tests) -> dict:
    return {
        "cls": "code",
        "tid": "jail",
        "prompt": "",
        "expected": "\n".join(expected for _, expected in tests),
        "tests": tuple(tests),
        "match": "exact",
        "budget": 256,
    }


@pytest.mark.parametrize(
    ("escape", "name"),
    [
        ("import os", "os"),
        ("import posix", "posix"),
        ("import pathlib", "pathlib"),
        ("from pathlib import Path", "pathlib"),
        ("import io", "io"),
        ("import subprocess", "subprocess"),
        ("import shutil", "shutil"),
        ("import socket", "socket"),
        ("import ctypes", "ctypes"),
        ("import importlib", "importlib"),
        ("import sys", "sys"),
        ("__import__('os')", "__import__"),
        ("eval('1')", "eval"),
        ("exec('1')", "exec"),
        ("compile('1', '<s>', 'eval')", "compile"),
        ("open('/tmp/jail_escape', 'w')", "open"),
        ("pathlib.Path('/tmp').read_text()", "pathlib"),
        ("os.open('/tmp/jail_escape', 0)", "os"),
        ("().__class__.__bases__", "__class__"),
        ("getattr(__builtins__, 'open')", "getattr"),
        ("__builtins__['open']('/tmp/jail_escape', 'w')", "__builtins__"),
    ],
)
def test_escape_attempts_are_refused_with_the_offending_name(escape: str, name: str):
    response = f"def add(a, b):\n    return a + b\n{escape}\n"
    verdict = verify(_task((("add(2, 3)", "5"),)), response)
    assert verdict.passed is False
    assert verdict.reason.startswith("refused:"), verdict.reason
    assert name in verdict.reason, (name, verdict.reason)
    assert not Path("/tmp/jail_escape").exists()


@pytest.mark.parametrize(
    ("solution", "tests"),
    [
        # Loops and accumulation.
        (
            "def total(n):\n    s = 0\n    for i in range(n):\n        s += i\n    return s",
            (("total(5)", "10"),),
        ),
        # List/dict comprehensions.
        (
            "def double(xs):\n    return [x * 2 for x in xs]",
            (("double([1, 2, 3])", "[2, 4, 6]"),),
        ),
        (
            "def index(xs):\n    return {i: x for i, x in enumerate(xs)}",
            (("index(['a', 'b'])", "{0: 'a', 1: 'b'}"),),
        ),
        # Slicing.
        ("def rev(s):\n    return s[::-1]", (("rev('abc')", "'cba'"),)),
        # sorted / len / range / sum.
        (
            "def f(xs):\n    return len(xs) + sum(sorted(xs)[:2])",
            (("f([3, 1, 2])", "6"),),
        ),
        # Generators (the built tasks' own reference uses one).
        (
            "def sq(n):\n    return sum(j * j for j in range(1, n + 1))",
            (("sq(3)", "14"),),
        ),
        # Recursion.
        (
            "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)",
            (("fact(5)", "120"),),
        ),
        # A lambda as a sort key.
        (
            "def by_second(xs):\n    return sorted(xs, key=lambda p: p[1])",
            (("by_second([(1, 3), (2, 1)])", "[(2, 1), (1, 3)]"),),
        ),
        # Class definitions, methods and instance attributes.
        (
            "class Point:\n"
            "    def __init__(self, x):\n        self.x = x\n"
            "    def double(self):\n        return self.x * 2\n\n"
            "def f(n):\n    return Point(n).double()",
            (("f(4)", "8"),),
        ),
        # The one importable module.
        (
            "import math\n\ndef hyp(a, b):\n    return math.isqrt(a * a + b * b)",
            (("hyp(3, 4)", "5"),),
        ),
    ],
)
def test_legitimate_solutions_still_run_and_pass(solution: str, tests):
    verdict = verify(_task(tests), solution)
    assert verdict.passed is True, verdict.reason


def _program(code: str, tests=()) -> str:
    return S._SANDBOX_PREAMBLE % (
        tuple(sorted(S._FORBIDDEN_NAMES)),
        tuple(sorted(S._ALLOWED_MODULES)),
        tuple(sorted(S._ALLOWED_BUILTINS)),
        code,
        S._harness(tests),
    )


@pytest.mark.parametrize(
    "code",
    [
        "open('/tmp/jail_runtime', 'w')",
        "import os",
        "__import__('os')",
        "getattr(__builtins__, 'open')",
    ],
)
def test_the_runtime_guard_is_the_second_line(code: str):
    """Drive the child preamble DIRECTLY, past `_static_screen`: the guard holds.

    Without this test the runtime guard could be deleted and every static gate
    would stay green — the second line would be a claim, not a gate.
    """
    with pytest.raises(S._SandboxError):
        S._run_sandbox(_program(code), timeout=S.TIMEOUT_S)
    assert not Path("/tmp/jail_runtime").exists()


def test_the_runtime_allowlist_still_imports_math():
    """The second line is an allowlist, not a blanket import ban."""
    out = S._run_sandbox(_program("import math\nprint(math.sqrt(9))"), timeout=S.TIMEOUT_S)
    assert out.splitlines()[0] == "3.0"
