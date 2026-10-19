"""Gates for the math verifier: a hostile answer is data, never code.

The verifier used to `sympify` the model's final answer in the trainer's process,
so an answer could execute arbitrary code (`__import__('os').system(...)`,
`open(...)`) or hang the run (`9**9**9`). The table below is the attack surface:
every entry must be REFUSED with a reason naming the offending token, the pwn
files must not exist afterwards, and the arithmetic the verifier exists for must
still pass.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from dormouse.post.verifiers import (
    COUNTERS,
    MATH_DEADLINE_S,
    MAX_EXPR_CHARS,
    Verdict,
    sympy_equivalent,
    verify,
)

PWN_1 = Path("/tmp/pwn_1")
PWN_2 = Path("/tmp/pwn_2")


def _math(expected: str) -> dict:
    return {
        "cls": "math",
        "tid": "m",
        "prompt": "",
        "expected": expected,
        "tests": (),
        "budget": 64,
    }


@pytest.mark.parametrize(
    ("response", "token"),
    [
        ("__import__('os').system('touch /tmp/pwn_1')", "__import__"),
        ("open('/tmp/pwn_2', 'w')", "open"),
        ("().__class__.__bases__", "__class__"),
        ('eval("1")', "eval"),
        ("exec('1')", "exec"),
        ("compile('1', '<s>', 'eval')", "compile"),
        ("getattr(__builtins__, 'open')", "getattr"),
        ("9**9**9", "387420489"),
        ("1 + " * (MAX_EXPR_CHARS // 4) + "1", "characters"),
    ],
)
def test_hostile_math_answers_are_refused_with_a_reason(response: str, token: str):
    """Every escape is a Verdict(False) whose reason NAMES the cause."""
    PWN_1.unlink(missing_ok=True)
    PWN_2.unlink(missing_ok=True)
    verdict = verify(_math("12"), response)
    assert verdict.passed is False
    assert verdict.reason.startswith("refused:"), verdict.reason
    assert token in verdict.reason, (token, verdict.reason)
    # The two file-creating payloads above must not have run in the trainer.
    assert not PWN_1.exists(), "the math verifier executed `__import__('os').system(...)`"
    assert not PWN_2.exists(), "the math verifier executed `open(...)`"


def test_a_hostile_answer_is_a_refusal_not_a_crash():
    """`verify` is total on model text: no exception, no traceback, same verdict."""
    task = _math("12")
    hostile = "__import__('os').system('touch /tmp/pwn_1')"
    first, second = verify(task, hostile), verify(task, hostile)
    assert first == second == Verdict(False, first.reason, first.budget_used)


#: R3-1, measured by review round 3: the 23-byte answer that raised
#: `ValueError` out of `_short`, the 5.45 s 3-deep pow, and the shapes around
#: them. `expected_token` is None for the two payloads that are a legitimate
#: (bounded) expression or no answer at all — they must still be a verdict.
R3_ABUSIVE = [
    ("answer: (2**1000)**1000", "power"),
    ("answer: 9**9**9**9", "exponent"),
    ("answer: " + "2" + "**2" * 85, "nodes"),  # exactly 256 chars of `**`
    ("answer: sin(10**99)", None),
    ("answer: " + "9" * 200_000, "characters"),
    ("", None),
    (b"answer: \x00\xff\xfe binary \x00", "valid expression"),
    ("answer: " + "(" * 200 + "1" + ")" * 200, "characters"),
    ("answer: exp(10**99)", "finite"),
    ("answer: ((2**1000)**1000)**1000", "power"),
]


def test_the_r3_crash_table_is_total_and_bounded():
    """R3-1 gate: 10 abusive answers, each a Verdict inside the work deadline.

    A refusal must NAME its cause (`power`/`exponent`/`nodes`/`characters`/
    `finite`/`valid expression`), the replayed `(2**1000)**1000` must not raise,
    and the module counter must tally exactly the refusals. The 250 ms wall is
    5x the deadline: the numeric path is sub-millisecond by construction (the
    measured worst allowed expression is < 3 ms), and the bound only gets that
    slack for scheduler jitter.
    """
    before = COUNTERS.get("verify_refused", 0)
    refusals = 0
    for response, token in R3_ABUSIVE:
        start = time.monotonic()
        verdict = verify(_math("2"), response)
        elapsed = time.monotonic() - start
        assert elapsed < MATH_DEADLINE_S * 5, (response, elapsed)
        assert verdict.passed is False, response
        if token is None:
            continue
        refusals += 1
        assert verdict.reason.startswith("refused:"), (response, verdict.reason)
        assert token in verdict.reason, (token, verdict.reason)
    assert COUNTERS.get("verify_refused", 0) - before == refusals


def test_a_numeric_path_exception_is_a_counted_refusal(monkeypatch):
    """The R3-1 guarantee independent of the bounds: an exception is a Verdict.

    The measured crash was a `ValueError` from `str()` on a big integer; the
    bound now prevents that value from forming, but a future sympy/parser bug
    must still fail the sample. `verify_error` is the counter a log reader sees.
    """
    before = COUNTERS.get("verify_error", 0)

    def boom(*_args, **_kwargs):
        raise ValueError("Exceeds the limit (4300 digits) for integer string conversion")

    monkeypatch.setattr("dormouse.post.verifiers._safe_expr", boom)
    verdict = verify(_math("2"), "answer: 2")
    assert verdict.passed is False
    assert "ValueError" in verdict.reason and verdict.reason.startswith("refused:")
    assert COUNTERS.get("verify_error", 0) == before + 1


def test_a_legal_answer_still_verifies_after_the_bound():
    """The R3-1 fix must not refuse the arithmetic the verifier exists for."""
    assert verify(_math("7"), r"\boxed{7}") == Verdict(True, "sympy-equivalent", 0.140625)
    assert verify(_math("10**100"), "answer: 10**100").passed is True
    assert verify(_math("1/3"), "answer: 1/3").passed is True
    assert verify(_math("2**10"), "answer: 2**10").passed is True


@pytest.mark.parametrize(
    ("expected", "response"),
    [
        ("12", r"\boxed{12}"),
        ("12", "answer: 12"),
        ("3/4", "answer: 3/4"),
        ("-1500", "answer: -1.5e3"),
        ("1/2", "answer: 2/4"),
        ("1/2", "answer: 0.5"),
        ("2.5", "answer: 5/2"),
    ],
)
def test_safe_answers_still_verify(expected: str, response: str):
    assert verify(_math(expected), response).passed is True


def test_a_wrong_answer_is_still_wrong():
    assert verify(_math("1/2"), "answer: 3/4").passed is False
    assert verify(_math("12"), "I cannot solve this").passed is False
    assert verify(_math("12"), "answer: 12 // 5").passed is False


def test_the_reference_comparison_is_unchanged():
    """The known-good case that existed before the whitelist: sympy equivalence."""
    assert sympy_equivalent("1/2", "2/4") is True
    assert sympy_equivalent("1/2", "0.5") is True
    assert sympy_equivalent("-1500", "-1.5e3") is True
    assert sympy_equivalent("1/2", "0.5000001") is False
    # A refused `got` is False, never an exception in a reward path.
    assert sympy_equivalent("1/2", "__import__('os')") is False
    assert sympy_equivalent("1/2", "9**9**9") is False


def test_a_bad_task_reference_is_loud():
    """The reference is task data, not model text: a bad one stops the run."""
    from dormouse.instrument import LoudError

    with pytest.raises(LoudError, match="outside the verifier's math whitelist"):
        verify(_math("__import__('os')"), "answer: 1")
