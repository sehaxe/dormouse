"""The agent lane: the tool template + its mask, the loop, the jail, the credit.

Every gate is CPU. What has to be true here is the SEMANTICS — which bytes the
loss supervises, which malformations are refused loudly, where a turn's weight
comes from, and that `serve` without `tools` is the server it was before — and
none of that needs a GPU, a checkpoint or a real model. The stub
(`post.agent.ScriptedModel`) emits a fixed byte script; it is the same stub the
demo prints and it is not a language model.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg, tiny_model

from dormouse.config import RunConfig, TrainConfig
from dormouse.instrument import LoudError
from dormouse.post.agent import (
    ScriptedModel,
    Tool,
    ToolCall,
    Trajectory,
    TurnSpan,
    agent_rollout,
    calc_tool,
    parse_turn,
)
from dormouse.post.agent_rl import trajectory_advantages
from dormouse.post.rl import RLConfig
from dormouse.post.tasks import Task
from dormouse.post.verifiers import Verdict
from dormouse.serve import ServeServer, ServeState, build_server, make_handler
from dormouse.train.data import ByteStream
from dormouse.train.sft import (
    ASSIST,
    OBSERVATION,
    PAD,
    USER,
    Segment,
    build_sft_corpus,
    conversation_mask,
    render,
    render_conversation,
    scan_ranges,
    sft_mask,
    supervised_ranges,
)

ROOT = Path(__file__).resolve().parents[1]

#: The sha256 of `build_sft_corpus(seed=7, seq_len=64, n_tasks=4)`, taken on main
#: BEFORE the tool template existed. The extension must not move a byte of it.
PLAIN_CORPUS_SHA = "4689606b376a3b546acc92447f2eff5521bb3fd86826622db95bbb7c5d6b9050"

THREE_TURN = [
    Segment("user", "What is 2 + 2?"),
    Segment("assistant", "Let me compute.\n"),
    Segment("call", name="calc", args='{"expr": "2 + 2"}'),
    Segment("observation", "4"),
    Segment("assistant", "answer: 4"),
    Segment("end"),
]

#: The three-turn example's supervised byte spans, by hand (inclusive).
THREE_TURN_SPANS = [(16, 32), (33, 56), (59, 68), (69, 69)]
#: ... and the observation's two bytes (`0x07`, "4") — in NO span above.
OBSERVATION_BYTES = (57, 58)

CALL_CALC = b'Let me compute.\n\x04calc\x05{"expr": "2 + 2"}\x06'
ANSWER_4 = b"answer: 4\x03"

ECHO_SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
EMPTY_SCHEMA = {"type": "object", "properties": {}}
PATH_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
}


# --- module-level tools: a sandboxed tool's SOURCE is what the jail runs -------


def _echo(args):
    return str(args["x"] * 2)


def _cwd(args):
    import os as _os

    return _os.getcwd()


def _escape_process(args):
    import subprocess

    subprocess.run(["touch", args["path"]], check=True)
    return "wrote it"


def _escape_passwd(args):
    return open("/etc/passwd").read()


def _escape_net(args):
    import socket

    return socket.gethostname()


def _task(expected: str = "4") -> Task:
    return Task(
        cls="math",
        tid="gate",
        prompt="What is 2 + 2?",
        expected=expected,
        solution=expected,
        tests=(),
        match="exact",
        budget=0,
    )


def _trajectory(spec: bytes, expected: str = "4", **kw) -> Trajectory:
    return agent_rollout(
        ScriptedModel(spec),
        _task(expected),
        kw.pop("tools", [calc_tool()]),
        kw.pop("budget", 512),
        kw.pop("max_turns", 4),
        kw.pop("temp", 0.8),
        **kw,
    )


# --------------------------------------------------------------------------- (1) the template


def test_the_plain_corpus_is_bit_identical(tmp_path: Path):
    """The off-by-default gate: no tool markers, the same 23 725 bytes as main."""
    corpus = tmp_path / "sft.bin"
    report = build_sft_corpus(corpus, 7, seq_len=64, n_tasks=4)
    assert report["records"] == 365
    assert hashlib.sha256(corpus.read_bytes()).hexdigest() == PLAIN_CORPUS_SHA
    assert render("What is 2 + 2?", "answer: 4") == b"\x01What is 2 + 2?\n\x02answer: 4\x03"
    plain = [Segment("user", "What is 2 + 2?"), Segment("assistant", "answer: 4"), Segment("end")]
    assert render_conversation(plain) == render("What is 2 + 2?", "answer: 4")


def test_a_tool_free_window_masks_exactly_like_sft_mask(tmp_path: Path):
    """The tool mask is a GENERALIZATION: on today's windows it is sft_mask."""
    corpus = tmp_path / "sft.bin"
    build_sft_corpus(corpus, 7, seq_len=64, n_tasks=4)
    x, _y = ByteStream(corpus, 4, 64).next_batch()
    assert torch.equal(conversation_mask(x), sft_mask(x))


def test_three_turn_mask_is_exact(capsys: pytest.CaptureFixture[str]):
    """Print the mask of a 3-turn example: assistant and call bytes, never the tool."""
    raw = render_conversation(THREE_TURN)
    ranges = supervised_ranges(THREE_TURN)
    mask = conversation_mask(torch.tensor([list(raw)]))[0]
    supervised = [j for j in range(len(raw)) if bool(mask[j])]
    with capsys.disabled():
        # The dump a human reads in the gate's output (or in agent_demo.py).
        print(f"\n3-turn transcript ({len(raw)} B): {raw!r}")
        print(f"supervised spans (inclusive): {ranges}")
        print("mask (one char per byte, '#' supervised):")
        print("  " + "".join("#" if bool(mask[j]) else "." for j in range(len(raw))))
        print(f"supervised positions: {supervised}")
    assert ranges == THREE_TURN_SPANS, ranges
    # "The observation is unsupervised" is a statement about TARGETS: position j
    # predicts byte j + 1, so no byte of the observation may be a target.
    for target in OBSERVATION_BYTES:  # 0x07 and the "4"
        assert not bool(mask[target - 1]), f"observation byte {target} must not be a target"
    assert not any(s <= OBSERVATION_BYTES[0] <= e for s, e in ranges)
    # The byte AFTER the observation is the next turn's 0x02, and that IS a target:
    # the model must learn to continue after an observation, never to write one.
    assert bool(mask[OBSERVATION_BYTES[1]])
    assert bool(mask[THREE_TURN_SPANS[0][0] - 1])  # 0x02 is predicted by the byte before it
    assert int(mask.sum()) == 52  # 41 in turn 0 + 11 in turn 1
    # The last supervised position predicts the end marker; the position after it
    # would predict padding, and is never supervised.
    assert supervised[-1] == len(raw) - 2
    assert not any(j >= len(raw) - 1 for j in supervised)


def test_markers_inside_the_text_are_refused():
    with pytest.raises(LoudError, match="marker 0x04"):
        render_conversation([Segment("user", b"hi\x04there"), Segment("end")])
    with pytest.raises(LoudError, match="marker 0x07"):
        render_conversation([Segment("user", "q"), Segment("assistant", b"a\x07"), Segment("end")])


def test_a_call_outside_an_assistant_turn_is_refused():
    with pytest.raises(LoudError, match="must follow an assistant"):
        render_conversation([Segment("user", "q"), Segment("call", name="calc", args="{}")])
    with pytest.raises(LoudError, match="observation follows a call"):
        render_conversation([Segment("user", "q"), Segment("observation", "4")])


def test_malformed_call_json_is_refused_loudly():
    bad = [
        Segment("user", "q"),
        Segment("assistant", ""),
        Segment("call", name="calc", args="{not json"),
        Segment("observation", "x"),
        Segment("end"),
    ]
    with pytest.raises(LoudError, match="not valid JSON"):
        render_conversation(bad)
    with pytest.raises(LoudError, match="not valid JSON"):
        parse_turn(b"\x04calc\x05{oops}\x06", {"calc": calc_tool().schema})
    with pytest.raises(LoudError, match="not an object"):
        parse_turn(b"\x04calc\x05[1, 2]\x06", {"calc": calc_tool().schema})


def test_scan_ranges_refuses_a_window_that_is_not_a_conversation():
    with pytest.raises(LoudError, match="no assistant/call/end marker"):
        scan_ranges(list(bytes([USER]) + b"q\n" + bytes([PAD]) * 8))


# --------------------------------------------------------------------------- (2) the loop


def test_two_turns_call_calc_and_only_the_model_bytes_are_supervised():
    traj = _trajectory(CALL_CALC + ANSWER_4)
    assert traj.stop_reason == "end" and traj.passed
    assert [t.index for t in traj.turns] == [0, 1]
    assert len(traj.calls) == 1 and traj.calls[0].name == "calc"
    assert traj.calls[0].args == {"expr": "2 + 2"}
    assert traj.turns[0].observation == b"4"
    assert traj.answer == "Let me compute.\nanswer: 4"
    # The call's bytes are supervised; the observation's are in no span.
    call = traj.calls[0]
    spans = [s for t in traj.turns for s in t.supervised]
    assert any(s <= call.start and call.end <= e for s, e in spans)
    obs = traj.transcript.index(bytes([OBSERVATION]))
    assert not any(s <= obs <= e for s, e in spans)
    assert not any(s <= obs + 1 <= e for s, e in spans)


def test_a_turn_with_no_call_ends_the_rollout():
    traj = _trajectory(b"answer: 4\x03")
    assert len(traj.turns) == 1 and traj.stop_reason == "end" and traj.passed
    assert traj.calls == ()
    traj = _trajectory(b"answer: 4")  # no 0x03: the model emits the record's 0x00 pad
    assert traj.stop_reason == "pad" and traj.passed and traj.answer == "answer: 4"


def test_the_loop_terminates_at_max_turns_and_reports_the_count():
    model = ScriptedModel(CALL_CALC, repeat=True)
    with pytest.raises(LoudError, match=r"max_turns=3") as exc:
        agent_rollout(model, _task(), [calc_tool()], budget=600, max_turns=3)
    assert "Turns: 3" in str(exc.value), str(exc.value)
    assert "never terminated" in str(exc.value)


def test_max_turns_below_one_is_loud():
    with pytest.raises(LoudError, match="max_turns=0"):
        agent_rollout(ScriptedModel(b""), _task(), [], budget=64, max_turns=0)


def test_unknown_tool_is_loud():
    with pytest.raises(LoudError, match="unknown tool 'nope'"):
        _trajectory(b"\x04nope\x05{}\x06" + ANSWER_4)


def test_args_that_miss_the_schema_are_loud():
    with pytest.raises(LoudError, match="expected string"):
        _trajectory(b'\x04calc\x05{"expr": 3}\x06' + ANSWER_4)
    with pytest.raises(LoudError, match="missing required arg 'expr'"):
        _trajectory(b"\x04calc\x05{}\x06" + ANSWER_4)
    with pytest.raises(LoudError, match="unknown arg"):
        _trajectory(
            b'\x04calc\x05{"expr": "1", "z": 2}\x06' + ANSWER_4,
            tools=[
                Tool(
                    "calc", {**calc_tool().schema, "additionalProperties": False}, _echo, pure=True
                )
            ],
        )


def test_a_call_cut_by_the_byte_room_is_loud():
    with pytest.raises(LoudError, match="never closed by 0x06"):
        _trajectory(b'\x04calc\x05{"expr": "2 + 2"}', budget=200, max_turns=2)


def test_a_call_whose_observation_cannot_fit_the_budget_is_loud():
    spam = Tool("spam", EMPTY_SCHEMA, lambda _args: "x" * 200, pure=True)
    with pytest.raises(LoudError, match="against a 80 B budget"):
        _trajectory(b"\x04spam\x05{}\x06" + ANSWER_4, tools=[spam], budget=80, max_turns=2)


def test_a_budget_that_leaves_no_turn_room_is_loud():
    with pytest.raises(LoudError, match="leaves 0 B per turn"):
        agent_rollout(ScriptedModel(b""), _task(), [], budget=2, max_turns=3)


# --------------------------------------------------------------------------- (4) the jail


def test_a_sandboxed_tool_runs_in_the_verifier_jail():
    traj = _trajectory(
        b'\x04echo\x05{"x": 21}\x06answer: 42\x03',
        expected="42",
        tools=[Tool("echo", ECHO_SCHEMA, _echo)],
    )
    assert traj.turns[0].observation == b"42"
    # And the child ran somewhere that is NOT this checkout (the cwd is a tempdir).
    cwd = _trajectory(
        b"\x04cwd\x05{}\x06answer: 0\x03",
        expected="0",
        tools=[Tool("cwd", EMPTY_SCHEMA, _cwd)],
    )
    assert cwd.turns[0].observation is not None
    assert cwd.turns[0].observation.decode() != os.getcwd()


def test_an_escaping_tool_is_refused_by_screen_code_and_never_runs(tmp_path: Path):
    marker = tmp_path / "escaped"
    spec = b"\x04escape\x05" + json.dumps({"path": str(marker)}).encode() + b"\x06answer: 0\x03"
    with pytest.raises(LoudError, match="refused by the sandbox screen") as exc:
        _trajectory(spec, expected="0", tools=[Tool("escape", PATH_SCHEMA, _escape_process)])
    assert "process spawn" in str(exc.value)
    assert not marker.exists(), "the screen refused before the child could run"
    # The two payload classes the verifier's own jail gate uses.
    with pytest.raises(LoudError, match="sensitive path"):
        _trajectory(b"\x04t\x05{}\x06", tools=[Tool("t", EMPTY_SCHEMA, _escape_passwd)])
    with pytest.raises(LoudError, match="network access"):
        _trajectory(b"\x04t\x05{}\x06", tools=[Tool("t", EMPTY_SCHEMA, _escape_net)])


def test_pure_is_the_only_unsourced_callable():
    mul = Tool(
        "mul",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        lambda args: str(args["a"] * args["b"]),
    )
    with pytest.raises(LoudError, match="must be a plain module-level function"):
        _trajectory(b'\x04mul\x05{"a": 2, "b": 3}\x06answer: 6\x03', expected="6", tools=[mul])
    pure = Tool(mul.name, mul.schema, mul.fn, pure=True)
    traj = _trajectory(b'\x04mul\x05{"a": 2, "b": 3}\x06answer: 6\x03', expected="6", tools=[pure])
    assert traj.turns[0].observation == b"6"


# --------------------------------------------------------------------------- (3) the credit


def test_trajectory_advantages_match_hand_computed_values(capsys):
    traj = _trajectory(CALL_CALC + ANSWER_4)  # one call turn (0), one answer turn (1)
    adv = trajectory_advantages(traj, {0: [True, False]})
    with capsys.disabled():
        print(
            "\nhand fixture: actual passed, turn 0 replays [True, False] -> 1 - 0.5 = 0.5;"
            f" turn 1 made no call\n  {[(a.turn, a.span, a.weight, a.source) for a in adv]}"
        )
    assert [a.turn for a in adv] == [0, 0, 1, 1]
    assert [a.weight for a in adv] == [0.5, 0.5, 0.0, 0.0]
    assert [a.source for a in adv] == ["fork", "fork", "no_call", "no_call"]
    assert adv[0].span[0] < adv[1].span[0], "the assistant span precedes the call span"
    assert [a.span for a in adv[2:]] == list(traj.turns[1].supervised)

    # A failing actual: 0 - mean([True, True, True, False]) = -0.75.
    failing = _trajectory(CALL_CALC + ANSWER_4, expected="5")
    adv = trajectory_advantages(failing, {0: [True, True, True, False]})
    assert [a.weight for a in adv[:2]] == [-0.75, -0.75]
    assert traj.passed is True and failing.passed is False


def test_a_turn_that_is_not_a_fork_has_zero_weight(capsys):
    """Only turn 2 matters: turn 1 made no call, so it is not a fork and weighs 0."""
    no_call = TurnSpan(
        index=0,
        start=0,
        end=6,
        generated=b"hello",
        call=None,
        observation=None,
        supervised=((1, 5),),
    )
    call = ToolCall(name="calc", args={"expr": "1"}, raw_args='{"expr": "1"}', start=8, end=20)
    called = TurnSpan(
        index=1,
        start=6,
        end=24,
        generated=b"x",
        call=call,
        observation=b"1",
        supervised=((7, 20), (21, 23)),
    )
    traj = Trajectory(
        transcript=b"hello!" + b"x" * 18,
        turns=(no_call, called),
        calls=(call,),
        verdict=Verdict(True, "sympy-equivalent", 0.0),
        answer="1",
        stop_reason="end",
    )
    adv = trajectory_advantages(traj, {1: [True, False]})
    with capsys.disabled():
        print(f"\nonly turn 2 matters -> {[(a.turn, a.weight, a.source) for a in adv]}")
    assert [(a.turn, a.weight, a.source) for a in adv] == [
        (0, 0.0, "no_call"),
        (1, 0.5, "fork"),
        (1, 0.5, "fork"),
    ]


def test_a_fork_without_replays_is_loud():
    traj = _trajectory(CALL_CALC + ANSWER_4)
    with pytest.raises(LoudError, match="0 replayed continuation"):
        trajectory_advantages(traj, {})
    with pytest.raises(LoudError, match="min_replays=2"):
        trajectory_advantages(traj, {0: [True]}, RLConfig(min_replays=2))
    with pytest.raises(LoudError, match="turn 9"):
        trajectory_advantages(traj, {0: [True], 9: [True]})
    with pytest.raises(LoudError, match="made no tool call"):
        trajectory_advantages(traj, {1: [True]})


# --------------------------------------------------------------------------- (4) serve


CALC_DECL = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": "arithmetic",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    },
}


def _write_ckpt(d: Path, cfg, model) -> None:
    run = RunConfig(source="tiny", model=cfg, train=TrainConfig(ckpt_name="tiny", out_dir=str(d)))
    (d / "tiny.config.toml").write_text(run.snapshot_toml())
    torch.save({"model": model.state_dict(), "opt": {}, "step": 0}, d / "tiny.pt")


def _serve(httpd: ServeServer) -> Iterator[str]:
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=10)


@pytest.fixture(scope="module")
def served_tight(tmp_path_factory: pytest.TempPathFactory):
    """A SEEDED tiny checkpoint: the same weights in every process, so a pin holds."""
    d = tmp_path_factory.mktemp("ck_seeded")
    torch.manual_seed(0)
    _write_ckpt(d, tiny_cfg(), tiny_model())
    httpd = build_server(d, "tiny", port=0, device="cpu", max_tokens_cap=8, log=lambda *_: None)
    yield from _serve(httpd)


def _chat(url: str, **body) -> dict:
    payload = {"messages": [{"role": "user", "content": "Hello"}], **body}
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_serve_without_tools_is_byte_identical(served_tight):
    body = _chat(served_tight, max_tokens=8)
    assert "tool_calls" not in body["choices"][0]["message"]
    body.pop("id")
    body.pop("created")
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    # Pinned from MAIN (`37ce15eb…` measured on both trees with the same seeded
    # checkpoint and request, in separate processes): the tool branch must not
    # move the no-tools response by a byte.
    assert digest == "37ce15eb9a4e85674b10f462f83fbda472db008c88c35ccce4059fe3f5c14bf2"


def test_serve_returns_tool_calls_when_the_model_emits_one():
    model = ScriptedModel(b'\x04calc\x05{"expr": "2 + 2"}\x06', max_seq_len=256)
    run = RunConfig(source="stub", model=tiny_cfg(), train=TrainConfig(seed=1))
    state = ServeState(model, run, torch.device("cpu"), "stub", 64, 0.8, log=lambda *_: None)
    httpd = ServeServer(("127.0.0.1", 0), make_handler(state))
    httpd.state = state
    for url in _serve(httpd):
        body = _chat(url, tools=[CALC_DECL], max_tokens=32)
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    (call,) = choice["message"]["tool_calls"]
    assert call["type"] == "function"
    assert call["function"] == {"name": "calc", "arguments": '{"expr": "2 + 2"}'}
    assert call["id"].startswith("call_")
    assert body["usage"]["completion_tokens"] == len(b'\x04calc\x05{"expr": "2 + 2"}\x06')


def test_serve_accepts_a_tool_role_message_as_an_observation():
    model = ScriptedModel(b"answer: 4\x03", max_seq_len=256)
    run = RunConfig(source="stub", model=tiny_cfg(), train=TrainConfig(seed=1))
    state = ServeState(model, run, torch.device("cpu"), "stub", 64, 0.8, log=lambda *_: None)
    httpd = ServeServer(("127.0.0.1", 0), make_handler(state))
    httpd.state = state
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"expr": "2 + 2"}'},
                }
            ],
        },
        {"role": "tool", "content": "4"},
    ]
    for url in _serve(httpd):
        body = _chat(url, messages=messages, tools=[CALC_DECL], max_tokens=16)
    assert body["choices"][0]["finish_reason"] == "stop"
    prompt = model.seen[0]
    assert b"\x04calc\x05" in prompt, "the history's call is rendered"
    assert b"\x074" in prompt, "the tool message is rendered as an observation"
    assert prompt.endswith(bytes([ASSIST])), "the prompt opens a fresh assistant turn"


def test_serve_refuses_streaming_with_tools(served_tight):
    with pytest.raises(urllib.error.HTTPError) as err:
        _chat(served_tight, tools=[CALC_DECL], stream=True, max_tokens=8)
    assert err.value.code == 400
    assert "stream=true with `tools`" in json.loads(err.value.read())["error"]["message"]


def test_serve_still_refuses_a_tool_role_without_tools(served_tight):
    with pytest.raises(urllib.error.HTTPError) as err:
        _chat(served_tight, messages=[{"role": "tool", "content": "4"}], max_tokens=8)
    assert err.value.code == 400


# --------------------------------------------------------------------------- the demo


def test_agent_demo_runs_on_cpu_without_a_model():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "agent_demo.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert out.returncode == 0, out.stderr
    assert "mask (105 positions" in out.stdout
    assert "observation b'4' (never supervised)" in out.stdout
    assert "verdict: passed=True" in out.stdout
