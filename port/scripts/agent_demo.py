"""A two-turn agent rollout on CPU, with a stub model — the loop, made visible.

    uv run python scripts/agent_demo.py

No GPU, no checkpoint, no training: `ScriptedModel` emits a fixed byte script
(`post/agent.py` says why that is legitimate), the loop parses the call, runs
`calc` through the verifier's AST path, appends the observation, and lets the
model answer. What a human gets out of it is the one thing a transcript does not
show on its own: WHICH bytes the SFT loss would count. The mask line prints one
character per transcript byte, `#` supervised, `.` not — the observations are
never `#`, which is the whole reason the template has a separate `0x07`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dormouse.post.agent import ScriptedModel, agent_rollout, calc_tool  # noqa: E402
from dormouse.post.tasks import Task  # noqa: E402
from dormouse.train.sft import scan_ranges  # noqa: E402

#: Turn 1 calls calc; turn 2 reads the observation and answers. The `0x06` after
#: the args is where the harness stops generation and takes over.
SCRIPT = (
    b'Let me compute.\n\x04calc\x05{"expr": "2 + 2"}\x06'
    b"answer: 4\x03"
)

TASK = Task(
    cls="math",
    tid="demo-add",
    prompt="What is 2 + 2? Please end with an `answer:` line.",
    expected="4",
    solution="4",
    tests=(),
    match="exact",
    budget=0,
)


def main() -> int:
    model = ScriptedModel(SCRIPT, max_seq_len=64)
    trajectory = agent_rollout(
        model,
        TASK,
        [calc_tool()],
        budget=512,
        max_turns=3,
        temp=0.8,
    )
    print(f"transcript ({trajectory.n_bytes} B):")
    print(f"  {trajectory.transcript!r}")
    ranges = scan_ranges(trajectory.transcript)
    mark = ["."] * trajectory.n_bytes
    for start, end in ranges:
        for j in range(start - 1, end):  # position j predicts byte j+1
            if 0 <= j < len(mark):
                mark[j] = "#"
    print(f"mask ({len(mark)} positions, '#' = supervised):")
    print("  " + "".join(mark))
    print("supervised spans (inclusive bytes):", ranges)
    for turn in trajectory.turns:
        what = f"call {turn.call.name} {turn.call.args}" if turn.call else "no call"
        print(f"turn {turn.index}: bytes [{turn.start}, {turn.end}) {what}")
        print(f"  supervised {list(turn.supervised)}")
        if turn.observation is not None:
            print(f"  observation {turn.observation!r} (never supervised)")
    print(f"verdict: passed={trajectory.passed} reason={trajectory.verdict.reason!r}")
    print(f"stop_reason: {trajectory.stop_reason}   calls: {len(trajectory.calls)}")
    assert trajectory.passed, "the demo's script answers the task"
    assert len(trajectory.calls) == 1 and trajectory.calls[0].name == "calc"
    assert trajectory.turns[0].observation == b"4", "calc ran and the observation is in the span"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
