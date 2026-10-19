"""Gates for the math verifier: a hostile answer is data, never code.

The verifier used to `sympify` the model's final answer in the trainer's process,
so an answer could execute arbitrary code (`__import__('os').system(...)`,
`open(...)`) or hang the run (`9**9**9`). The table below is the attack surface:
every entry must be REFUSED with a reason naming the offending token, the pwn
files must not exist afterwards, and the arithmetic the verifier exists for must
still pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dormouse.post.verifiers import (
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
