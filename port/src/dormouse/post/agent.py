"""The multi-turn agent loop: generate → parse a call → dispatch → observe → repeat.

`post/rollouts.py` is ONE stroke of this: prompt, generate, verify. An agent is
the same loop with a fork in it — the model may ask for a tool, the harness runs
it and appends the result to the transcript, and the model continues from the
observation. Four things make that trainable and all four are here:

* **The transcript is the SFT template, extended** (`train/sft.py`): a call is
  `0x04 name 0x05 json 0x06`, an observation is `0x07 text`. The model writes
  the call bytes; the harness writes the observation bytes and the mask never
  supervises them (`scan_ranges`).
* **A tool runs in the EXISTING sandbox** (`post/verifiers.py`): the static
  screen plus the child preamble, rlimits and tempdir cwd. There is no second
  jail in this package. A tool declared `pure=True` runs in-process — that is
  the one exemption, and it is a declaration by the caller that the callable
  does no I/O (the calc tool's arithmetic, checked by a character allowlist,
  is the example).
* **A turn is byte-budgeted and the transcript is byte-capped.** `budget` is
  the whole rollout's B; each turn may generate `budget // max_turns`; a call
  whose observation would not fit is a LOUD refusal, not a truncation.
* **Every refusal is loud** (`instrument.loud`): `max_turns < 1`, an unknown
  tool, malformed JSON, args that miss the tool's own schema, a call marker
  without its `0x06`, and a transcript that still asks for a tool on the last
  turn (reporting the turn count — never an infinite loop, never a silent
  truncation).

Credit assignment is the other half and lives in `post/agent_rl.py`: the fork
for a turn is the transcript prefix, so the advantage is
`pass(actual) - mean(pass(replayed continuations))` per call-bearing turn.
"""

from __future__ import annotations

import inspect
import json
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import sympy
import torch

from ..instrument import loud
from ..model.decode import DEFAULT_TEMP, decode_bytes
from ..train.sft import (
    ASSIST,
    END,
    PAD,
    TOOL_ARGS,
    TOOL_CALL,
    TOOL_END,
    TOOL_MARKERS,
    Segment,
    render_conversation,
    scan_ranges,
    segment_bytes,
)
from .rollouts import GREEDY_TEMP
from .tasks import Task
from .verifiers import (
    SANDBOX_PREAMBLE,
    TIMEOUT_S,
    SandboxError,
    SandboxTimeout,
    Verdict,
    run_sandbox,
    screen_code,
    verify,
)

__all__ = [
    "CALC_SCHEMA",
    "ScriptedModel",
    "Tool",
    "ToolCall",
    "Trajectory",
    "TurnSpan",
    "agent_rollout",
    "calc_tool",
    "check_args",
    "parse_turn",
]


# ---------------------------------------------------------------------------
# the transcript's spans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One call the model emitted: the decoded args, and where the bytes are.

    `start`/`end` are INCLUSIVE byte offsets into [`Trajectory.transcript`]
    (the `0x04` through the `0x06`), so a trainer can rebuild the supervised
    span without re-parsing the transcript.
    """

    name: str
    args: dict
    raw_args: str
    start: int
    end: int


@dataclass(frozen=True)
class TurnSpan:
    """One assistant turn: what it generated, what it called, what came back."""

    index: int
    start: int  # first byte of the turn's `0x02`
    end: int  # one past the turn's last rendered byte
    generated: bytes  # exactly the bytes the model emitted this turn
    call: ToolCall | None
    observation: bytes | None
    #: The turn's supervised byte spans (inclusive), from `sft.scan_ranges`.
    supervised: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Trajectory:
    """A finished rollout: the whole transcript and the pieces a trainer reads."""

    transcript: bytes
    turns: tuple[TurnSpan, ...]
    calls: tuple[ToolCall, ...]
    verdict: Verdict
    answer: str
    #: "end" (the model emitted 0x03), "pad" (it emitted the record's 0x00), or
    #: "no_call" (it simply stopped asking for tools). A run that exhausts
    #: `max_turns` is a LOUD refusal, so it has no value here.
    stop_reason: str

    @property
    def passed(self) -> bool:
        return self.verdict.passed

    @property
    def n_bytes(self) -> int:
        return len(self.transcript)


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """`(name, json_schema, callable)` — the callable runs in the verifier's jail.

    `pure=True` is the ONE exemption from the sandbox: the callable runs
    in-process and must do no I/O (no file, no socket, no process, no clock
    that the reward depends on). It exists because a calculator cannot be
    usefully screened — its whole body is arithmetic — and because a lambda has
    no source for `inspect.getsource`, so a sandboxed tool has to be a plain
    module-level function. The default is the jail.
    """

    name: str
    schema: dict
    fn: Callable[[dict], str]
    pure: bool = False


#: The `calc` tool's schema. One string: the expression, which keeps the
#: sandboxed path out of it entirely (the allowlist below is the screen).
CALC_SCHEMA = {
    "type": "object",
    "properties": {"expr": {"type": "string"}},
    "required": ["expr"],
}

#: Arithmetic only: digits, `e`/`E` exponents, the four operators, parens, dots,
#: spaces. `**` is spelled with `*`, so exponentiation is not reachable — that is
#: a deliberate ceiling, not an oversight (it keeps `2**99999999` out too).
_CALC_ONLY = re.compile(r"^[0-9eE+\-*/(). ]+$")


def _calc(args: dict) -> str:
    expr = args.get("expr")
    if not isinstance(expr, str) or not _CALC_ONLY.match(expr):
        loud(
            f"calc: {expr!r} is not an arithmetic expression. Escape: digits and "
            f"+ - * / ( ) . only (no names, no calls, no **)."
        )
    return str(sympy.simplify(sympy.sympify(expr)))


def calc_tool() -> Tool:
    """The one `pure=True` example tool: arithmetic through sympy, no I/O."""
    return Tool("calc", CALC_SCHEMA, _calc, pure=True)


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
}


def check_args(name: str, args: dict, schema: dict) -> dict:
    """Refuse a call whose args do not match the tool's own JSON schema.

    The validator is the subset a tool's schema actually uses: `type: object`,
    `properties`, `required`, and `additionalProperties: false`. An unknown
    keyword in a schema is a LOUD error rather than a silently unchecked field —
    a schema that promises more than the validator checks is the ADR-0019 shape.
    """
    if not isinstance(schema, dict) or schema.get("type") != "object":
        loud(f"tool {name!r}: its schema must be an object schema, got {schema!r}")
    if not isinstance(args, dict):
        loud(f"tool {name!r}: the call's args are a JSON {type(args).__name__}, not an object")
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        loud(f"tool {name!r}: `properties` must be an object")
    for key in schema.get("required", ()):
        if key not in args:
            loud(
                f"tool {name!r}: missing required arg {key!r} (required: "
                f"{list(schema.get('required', ()))}). Escape: the model must emit all "
                f"required fields."
            )
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            if schema.get("additionalProperties", True) is False:
                loud(f"tool {name!r}: unknown arg {key!r} (the schema declares {sorted(props)})")
            continue
        want = spec.get("type") if isinstance(spec, dict) else None
        allowed = _JSON_TYPES.get(want or "")
        if allowed is None:
            loud(
                f"tool {name!r}: arg {key!r} declares type {want!r}, which this "
                f"validator does not check. Escape: one of {sorted(_JSON_TYPES)}."
            )
        # A JSON boolean is a Python int; the two types are distinguishable and
        # the schema is what decides, so check `bool` before the numeric arms.
        if isinstance(value, bool) and want != "boolean":
            loud(f"tool {name!r}: arg {key!r} is a boolean, expected {want}")
        if not isinstance(value, allowed) or (isinstance(value, bool) and want != "boolean"):
            loud(
                f"tool {name!r}: arg {key!r} is a {type(value).__name__}, expected "
                f"{want} (value {value!r}). Escape: the call's args must match the "
                f"tool's schema."
            )
    return args


def _parse_args(name: str, raw: str) -> dict:
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        loud(
            f"tool {name!r}: the call's args are not valid JSON ({exc.msg} at col "
            f"{exc.colno}). Escape: a JSON object between 0x05 and 0x06."
        )
    if not isinstance(args, dict):
        loud(f"tool {name!r}: the call's args are a JSON {type(args).__name__}, not an object")
    return args


def parse_turn(
    raw: bytes, schemas: Mapping[str, dict], *, offset: int = 0
) -> tuple[bytes, ToolCall | None]:
    """Split ONE turn's generated bytes into `(text before the call, call | None)`.

    This is the single parser `agent_rollout` and `serve` share: a turn is
    `text` or `text 0x04 name 0x05 args 0x06`, one call at most (the loop's
    granularity is one call per turn — a second call marker is refused). Every
    malformation is LOUD with the byte offset, so a log line names where.

    `offset` is the TRANSCRIPT position of `raw[0]` — not the record's start.
    The agent's generation begins after the turn's `0x02` marker, so it passes
    `turn_start + 1`; the server's prompt already ends with that marker.
    """
    marker = bytes([TOOL_CALL])
    i = raw.find(marker)
    if i < 0:
        return raw, None
    text = raw[:i]
    rest = raw[i + 1 :]
    args_marker = bytes([TOOL_ARGS])
    j = rest.find(args_marker)
    if j < 0:
        loud(
            f"parse_turn: the call marker 0x04 at byte {offset + i} has no 0x05 args "
            f"marker. Escape: a call is `0x04 name 0x05 json 0x06`."
        )
    name_raw = rest[:j]
    if not name_raw:
        loud(f"parse_turn: the call at byte {offset + i} carries an empty tool name")
    if any(b in TOOL_MARKERS for b in name_raw):
        loud(f"parse_turn: the tool name at byte {offset + i} carries a template marker")
    try:
        name = name_raw.decode("utf-8")
    except UnicodeDecodeError:
        loud(f"parse_turn: the tool name at byte {offset + i} is not UTF-8")
    end_marker = bytes([TOOL_END])
    k = rest.find(end_marker, j + 1)
    if k < 0:
        loud(
            f"parse_turn: the call to {name!r} at byte {offset + i} is never closed by "
            f"0x06 — the byte room ran out mid-call. Escape: a larger per-turn budget."
        )
    args_raw = rest[j + 1 : k].decode("utf-8", "replace")
    tail = rest[k + 1 :]
    if tail:
        loud(
            f"parse_turn: {len(tail)} byte(s) after the call's 0x06 at byte "
            f"{offset + i + k}: the harness appends the observation, so the model "
            f"must stop at the call's end."
        )
    args = _parse_args(name, args_raw)
    schema = schemas.get(name)
    if schema is None:
        loud(
            f"parse_turn: unknown tool {name!r}. This rollout declares "
            f"{sorted(schemas) or 'no tools'}. Escape: add the tool, or fix the name."
        )
    check_args(name, args, schema)
    return text, ToolCall(
        name=name,
        args=args,
        raw_args=args_raw,
        start=offset + i,
        end=offset + i + len(rest),
    )


# ---------------------------------------------------------------------------
# the executor: the verifier's jail, never a second one
# ---------------------------------------------------------------------------


def _run_tool(tool: Tool, args: dict) -> str:
    """Run `tool` on `args` and return its observation.

    A sandboxed tool's SOURCE is screened and then executed by the verifier's
    own child process (`verifiers.run_sandbox`: `-I -B`, rlimits, tempdir cwd,
    no network/process imports). A `pure=True` tool runs in-process, which is
    the caller's declaration and nothing here can check it.
    """
    if tool.pure:
        out = tool.fn(dict(args))
        if not isinstance(out, str):
            loud(f"tool {tool.name!r}: a pure tool must return str, got {type(out).__name__}")
        return out
    if not inspect.isfunction(tool.fn) or tool.fn.__name__ == "<lambda>":
        loud(
            f"tool {tool.name!r}: a sandboxed tool must be a plain module-level "
            f"function (its source is what the jail runs); a lambda has no source. "
            f"Escape: a def, or declare pure=True for an I/O-free callable."
        )
    try:
        src = textwrap.dedent(inspect.getsource(tool.fn))
    except (OSError, TypeError) as exc:
        loud(
            f"tool {tool.name!r}: its source is not inspectable ({exc}). Escape: a "
            f"module-level function, or pure=True for an I/O-free callable."
        )
    bad = screen_code(src)
    if bad is not None:
        loud(
            f"tool {tool.name!r}: refused by the sandbox screen — {bad} is not allowed "
            f"in the jail (the same screen `verifiers.verify` runs). Escape: remove "
            f"the I/O, or declare pure=True only if the callable really is I/O-free."
        )
    program = (
        SANDBOX_PREAMBLE
        + "\nimport json as _json\n"
        + src
        + "\n"
        + f"_args = _json.loads({json.dumps(json.dumps(args))})\n"
        + f"_out = {tool.fn.__name__}(_args)\n"
        + "if not isinstance(_out, str):\n"
        + "    raise TypeError('a tool must return str')\n"
        + "print(_out)\n"
    )
    try:
        out = run_sandbox(program, timeout=TIMEOUT_S)
    except SandboxTimeout:
        loud(f"tool {tool.name!r}: timed out after {TIMEOUT_S:g}s in the sandbox")
    except SandboxError as exc:
        loud(
            f"tool {tool.name!r}: the callable exited {exc.code} under the sandbox "
            f"(a raise, a refused import, or a non-str return)"
        )
    return out.rstrip("\n")


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def _schemas(tools: Sequence[Tool]) -> dict[str, dict]:
    schemas: dict[str, dict] = {}
    for t in tools:
        if not isinstance(t, Tool):
            loud(f"agent_rollout: {t!r} is not a Tool(name, schema, fn)")
        if t.name in schemas:
            loud(f"agent_rollout: two tools are named {t.name!r} — the call could not be routed")
        schemas[t.name] = t.schema
    return schemas


def _text(raw: bytes) -> str:
    """The bytes a verifier reads, with decode errors named rather than hidden."""
    return raw.decode("utf-8", "replace")


def _turn_spans(transcript: bytes, start: int) -> tuple[tuple[int, int], ...]:
    """The supervised spans of the turn that begins at `start`."""
    return tuple(r for r in scan_ranges(transcript) if start <= r[0] < len(transcript))


def agent_rollout(
    model,
    task: Task,
    tools: Sequence[Tool],
    budget: int,
    max_turns: int,
    temp: float = DEFAULT_TEMP,
    *,
    greedy: bool = False,
    device=None,
    generator=None,
) -> Trajectory:
    """Roll one task out for at most `max_turns` turns under `budget` bytes.

    Each turn generates up to `budget // max_turns` bytes and stops at `0x03`
    (the answer ends) or `0x06` (a call ends). A turn with no call ends the
    rollout: the model has stopped asking for tools. A rollout that asks for a
    tool on its LAST turn is refused loudly — there is no final answer to score,
    and looping forever is not an option.
    """
    if max_turns < 1:
        loud(f"agent_rollout max_turns={max_turns}: a rollout needs at least one turn")
    if budget <= 0:
        loud(f"agent_rollout budget={budget}: a rollout needs a positive byte budget")
    per_turn = budget // max_turns
    if per_turn < 1:
        loud(
            f"agent_rollout: budget={budget} over max_turns={max_turns} leaves "
            f"{per_turn} B per turn — no turn can generate a byte. Escape: raise the "
            f"budget or lower max_turns."
        )
    tools = tuple(tools)
    schemas = _schemas(tools)
    by_name = {t.name: t for t in tools}
    segments = [Segment("user", task.prompt)]
    transcript = render_conversation(segments)
    if len(transcript) > budget:
        loud(
            f"agent_rollout: the prompt is {len(transcript)} B against a {budget} B "
            f"budget. Escape: raise the budget or shorten the prompt."
        )
    effective_temp = GREEDY_TEMP if greedy else temp
    turns: list[TurnSpan] = []
    calls: list[ToolCall] = []
    answer = bytearray()
    stop_reason: str | None = None
    for index in range(max_turns):
        turn_start = len(transcript)
        # The model continues AFTER the turn's 0x02 — the same byte `serve.assemble`
        # ends its prompt with. The harness writes the marker; the model's first
        # generated byte is the first TEXT byte, which is why `parse_turn`'s offset
        # is `len(window)`, not `turn_start`.
        window = transcript + bytes([ASSIST])
        room = min(per_turn, budget - len(window))
        if room < 1:
            loud(
                f"agent_rollout: the transcript is {len(transcript)} B of a {budget} B "
                f"budget at turn {index} — no room for the turn's 0x02 marker and a "
                f"generated byte. Escape: a bigger budget."
            )
        out = decode_bytes(
            model,
            window,
            room,
            temp=effective_temp,
            device=device,
            generator=generator,
            # 0x00 ends a record in this template, so it ends a turn too (serve's
            # plain arm has made the same call since the first chat smoke).
            stop=frozenset({END, TOOL_END, PAD}),
        )
        assert out.startswith(window), "decode_bytes must return prompt + new bytes"
        generated = out[len(window) :]
        text, call = parse_turn(generated, schemas, offset=len(window))
        if call is None:
            if END in text:
                text = text[: text.index(END)]
                stop_reason = "end"
            elif PAD in text:
                text = text[: text.index(PAD)]
                stop_reason = "pad"
            else:
                stop_reason = "no_call"
            answer += text
            segments.append(Segment("assistant", text))
            if stop_reason == "end":
                segments.append(Segment("end"))
            transcript = render_conversation(segments)
            turns.append(
                TurnSpan(
                    index=index,
                    start=turn_start,
                    end=len(transcript),
                    generated=generated,
                    call=None,
                    observation=None,
                    supervised=_turn_spans(transcript, turn_start),
                )
            )
            break
        answer += text
        observation = _run_tool(by_name[call.name], call.args)
        call_seg = Segment("call", name=call.name, args=call.raw_args)
        # The rendered lengths, not `len(text)`: the markers count, and the
        # budget is a transcript-byte budget (the mask's own unit).
        head = len(segment_bytes(Segment("assistant", text)))
        call_bytes = len(segment_bytes(call_seg))
        obs_bytes = observation.encode("utf-8")
        need = len(transcript) + head + call_bytes + 1 + len(obs_bytes)
        if need > budget:
            loud(
                f"agent_rollout: the call to {call.name!r} at turn {index} needs "
                f"{need} B (transcript {len(transcript)} + assistant {head} + call "
                f"{call_bytes} + observation {len(obs_bytes)}) against a {budget} B "
                f"budget. Escape: a bigger budget, or a tool that answers in fewer bytes."
            )
        segments.append(Segment("assistant", text))
        segments.append(call_seg)
        segments.append(Segment("observation", observation))
        transcript = render_conversation(segments)
        calls.append(call)
        turns.append(
            TurnSpan(
                index=index,
                start=turn_start,
                end=len(transcript),
                generated=generated,
                call=call,
                observation=obs_bytes,
                supervised=_turn_spans(transcript, turn_start),
            )
        )
    if stop_reason is None:
        loud(
            f"agent_rollout: max_turns={max_turns} turns were ALL tool calls — the "
            f"transcript never terminated, so there is no final answer to verify. "
            f"Turns: {len(turns)}, calls: {len(calls)}. Escape: a higher max_turns, or "
            f"a model/prompt that emits 0x03."
        )
    verdict = verify(task.as_dict(), _text(bytes(answer)))
    return Trajectory(
        transcript=transcript,
        turns=tuple(turns),
        calls=tuple(calls),
        verdict=verdict,
        answer=_text(bytes(answer)),
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# the stub: a script, not a language model
# ---------------------------------------------------------------------------


class ScriptedModel:
    """A stand-in model for the CPU demo and the gates: emits a fixed byte script.

    Each forward is a one-hot distribution over the next scripted byte, so
    `decode_bytes` produces exactly `script`. It is NOT a language model and
    must never be used to measure quality — it exists so the loop's arithmetic
    (turn boundaries, masks, budgets, terminations) can be exercised without a
    GPU or a checkpoint.
    """

    def __init__(self, script: bytes, *, max_seq_len: int = 64, repeat: bool = False) -> None:
        self.script = script
        self.repeat = repeat
        self.pos = 0
        self.training = False
        self.cfg = SimpleNamespace(max_seq_len=max_seq_len)
        #: Every window the loop asked about — the serve gate reads this.
        self.seen: list[bytes] = []

    def __call__(self, ids: torch.Tensor):
        self.seen.append(bytes(ids[0].tolist()))
        if self.pos >= len(self.script) and self.repeat and self.script:
            self.pos %= len(self.script)
        byte = self.script[self.pos] if self.pos < len(self.script) else 0x00
        self.pos = min(self.pos + 1, len(self.script))
        logits = torch.full((ids.shape[0], ids.shape[1], 256), -100.0, device=ids.device)
        logits[:, -1, byte] = 100.0
        return (logits,)

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode
