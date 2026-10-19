"""Gates for the post-training lane state 1 (verifier · tasks · rollouts · CRR).

Every gate here is a thing the plan says must not lie: the builder is seedable,
the verifier refuses the do-not-run class and the reward-hack class, the exec
jail cannot reach the network or `/etc/passwd`, the byte budget truncates
exactly, and the CRR advantage matches a hand-computed table.
"""

from __future__ import annotations

import pytest
from conftest import tiny_model

from dormouse.instrument import LoudError
from dormouse.post.rollouts import Rollout, rollout
from dormouse.post.tasks import TASKS_PER_CLASS, build_tasks
from dormouse.post.train_steps import crr_advantage
from dormouse.post.verifiers import Verdict, verify


def _add_task() -> dict:
    return {
        "cls": "code",
        "tid": "add",
        "prompt": "",
        "expected": "5\n0",
        "tests": (("add(2, 3)", "5"), ("add(-1, 1)", "0")),
        "match": "exact",
        "budget": 128,
    }


def test_builder_is_seed_deterministic():
    a = build_tasks("code", 7, 16)
    b = build_tasks("code", 7, 16)
    assert a == b
    assert [t.as_dict() for t in a] == [t.as_dict() for t in b]
    assert a != build_tasks("code", 8, 16)  # a different seed is a different set
    math = build_tasks("math", 3)
    assert len(math) == TASKS_PER_CLASS
    assert math == build_tasks("math", 3)


def test_noop_verifier_refuses_loudly():
    task = build_tasks("noop", 1, 1)[0]
    with pytest.raises(LoudError):
        verify(task.as_dict(), "answer: 42")


def test_code_sandbox_refuses_network_and_passwd():
    task = _add_task()
    net = "import socket\ndef add(a, b):\n    return a + b"
    verdict = verify(task, net)
    assert verdict.passed is False
    assert verdict.reason.startswith("refused:") and "socket" in verdict.reason
    leak = 'def add(a, b):\n    return open("/etc/passwd").read()'
    verdict = verify(task, leak)
    assert verdict.passed is False
    assert verdict.reason.startswith("refused:") and "open" in verdict.reason


def test_code_sandbox_blocks_string_built_reads_by_name():
    # The path is built at runtime, but the CALL is a name: the allowlist sees it.
    task = _add_task()
    sneaky = 'def add(a, b):\n    open(chr(47) + "etc/passwd", "rb").read()\n    return a + b'
    verdict = verify(task, sneaky)
    assert verdict.passed is False
    assert verdict.reason.startswith("refused:") and "open" in verdict.reason


def test_code_verifier_accepts_correct_and_rejects_wrong():
    task = _add_task()
    good = "def add(a, b):\n    return a + b"
    assert verify(task, good).passed is True
    bad = "def add(a, b):\n    return a - b"
    assert verify(task, bad).passed is False


def test_built_code_tasks_are_solvable_by_their_reference():
    for task in build_tasks("code", 11, 8):
        assert verify(task.as_dict(), task.solution).passed is True, task.tid


def test_math_verifier_is_sympy_equivalence():
    task = {"cls": "math", "tid": "m", "prompt": "", "expected": "1/2", "tests": (), "budget": 64}
    assert verify(task, "answer: 2/4").passed is True
    assert verify(task, "answer: 0.5").passed is True
    assert verify(task, "answer: 3/4").passed is False
    assert verify(task, "I cannot solve this").passed is False


def test_byte_budget_truncates_the_rollout():
    task = build_tasks("math", 1, 1)[0]
    n = len(task.prompt.encode("utf-8"))
    model = tiny_model()
    none = rollout(model, task, n, greedy=True)
    assert none.trace == [] and none.n_bytes == n  # no room even for one byte
    small = rollout(model, task, n + 6, greedy=True)
    assert len(small.trace) == 2 and small.n_bytes == n + 6 == n + 3 * 2
    big = rollout(model, task, n + 60, greedy=True)
    assert len(big.trace) == 20 and big.n_bytes == n + 60
    assert small.n_bytes <= n + 6 <= n + 60 and small.verdict.budget_used <= 1.0


def test_crr_advantage_matches_a_hand_computed_table():
    def r(trace, passed):
        return Rollout(trace=list(trace), n_bytes=0, verdict=Verdict(passed, "x", 0.0))

    group = [
        r(["a", "b", "c", "d"], True),  # 0 pass
        r(["a", "b", "x"], False),  # 1 fail, 3 strokes
        r([], False),  # 2 empty trajectory
        r(["a", "b", "c", "d", "e"], True),  # 3 pass
        r(["a", "z"], False),  # 4 fail, 2 strokes
        r(["a", "b", "c", "d"], False),  # 5 fail, 4 strokes — the BEST failed
    ]
    adv = crr_advantage(group)
    # The best failed rollout is index 5 (4 strokes): its fork point is 4.
    assert [a.value for a in adv] == [1.0, -1.0, 0.0, 1.0, -1.0, -1.0]
    assert [a.source for a in adv] == [
        "pass",
        "fork_history",
        "empty",
        "pass",
        "fork_history",
        "fork_history",
    ]
    assert [a.fork_depth for a in adv] == [4, 4, 4, 4, 4, 4]
    assert [a.index for a in adv] == [0, 1, 2, 3, 4, 5]


def test_verify_is_pure_and_never_logs_the_raw_response():
    task = build_tasks("code", 5, 1)[0].as_dict()
    response = task["prompt"] + "def add(a, b):\n    return a + b\n# sentinel-9f3a\n"
    first, second = verify(task, response), verify(task, response)
    assert first == second
    assert "sentinel-9f3a" not in first.reason
