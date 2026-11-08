#!/usr/bin/env -S uv run --script
"""Measure the agent capability a BPB number cannot see, as FOUR separate numbers.

BPB scores the next byte. It says nothing about whether a checkpoint can emit a
tool call, whether the sandbox will run that call, whether the task is solved,
or whether the model is trying to escape the jail. This harness measures those
four things on a checkpoint and reports them separately — there is deliberately
no blended score, because a model that formats perfectly and cannot answer and
a model that answers by accident have the same average and different diseases.

| metric | denominator | what it is |
|---|---|---|
| `format` | tasks offering a tool | turn 0 parses; a name and a JSON-object args; name offered |
| `execution` | accepted calls | the call ran: valid args, no refusal, no escape attempt |
| `task success` | tasks | `pass@1` = turn 0; `pass@budget` = within `--max-turns` |
| `escape audit` | — | refusals by reason; a nonzero count is a security finding, not a score |

A call in this repo's template is `0x04 name 0x05 <json args> 0x06`, so
"`name` and `arguments`" is that pair, not an OpenAI function-call object: the
name is the marker payload and the args are the JSON object `parse_turn`
validates.

The task grammar is `agent_rollout`'s (`post/agent.py`), reused rather than
re-implemented: the same `render_conversation`, `parse_turn`, `Tool`,
`_run_tool`, and `verifiers.verify`. The difference is failure handling. A
training rollout is LOUD by design — a malformed turn should stop the run. A
capability measurement must survive a stupid model: every parse/exec/budget
failure is COUNTED against that task and the next task runs. That is the whole
reason this file exists instead of a loop around `agent_rollout`.

Tasks come from `--tasks` (the `post.tasks.load_rl_tasks` JSONL schema, the one
`scripts/build_rl_tasks.py` writes): a `math` task is offered the `calc` tool, a
`code` task is offered none and is scored by the verifier's own sandbox. The
shipped 20-task sample is 4 tasks of each of the builder's five kinds:

    uv run port/scripts/build_rl_tasks.py --per-kind 4 --seed 20261023 \\
        --out port/sft_pairs/agent_eval_sample.jsonl

It is a SAMPLE, not a benchmark. A BFCL-format exam set can be pointed at with
the same `--tasks` flag once an adapter exists; nothing here invents one.

CPU-only by default and checkpoint-free under `--dry-run --model scripted:*`,
so CI can run every gate without weights:

    CUDA_VISIBLE_DEVICES="" uv run port/scripts/eval_agent.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dormouse.instrument import LoudError  # noqa: E402
from dormouse.model.decode import decode_bytes  # noqa: E402
from dormouse.post.agent import (  # noqa: E402
    ScriptedModel,
    Tool,
    _run_tool,
    calc_tool,
    parse_turn,
)
from dormouse.post.rollouts import GREEDY_TEMP  # noqa: E402
from dormouse.post.tasks import Task, load_rl_tasks  # noqa: E402
from dormouse.post.verifiers import screen_code, verify  # noqa: E402
from dormouse.train.sft import (  # noqa: E402
    ASSIST,
    END,
    PAD,
    TOOL_END,
    Segment,
    render_conversation,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "sft_pairs" / "agent_eval_sample.jsonl"

#: The scripted arms are the harness's own control group, never a model. Each
#: is a turn-0 call followed by a turn-1 answer, so one script drives both the
#: format probe and the two-turn rollout. `valid` is schema-valid; the rest are
#: the malformations the gates need to see counted rather than crash.
SCRIPTS: dict[str, bytes] = {
    "valid": b'\x04calc\x05{"expr": "2 + 2"}\x06answer: 4\x03',
    "prose": b"answer: 4\x03",
    "invalid-name": b"\x04nope\x05{}\x06answer: 4\x03",
    "invalid-json": b"\x04calc\x05{not json}\x06answer: 4\x03",
    "invalid-args": b'\x04calc\x05{"expr": 3}\x06answer: 4\x03',
    "escape": b'\x04calc\x05{"expr": "import subprocess"}\x06answer: 4\x03',
    "escape-network": b'\x04calc\x05{"expr": "socket"}\x06answer: 4\x03',
}

# ---------------------------------------------------------------------------
# tasks and the model arm
# ---------------------------------------------------------------------------


def _sample(tasks: list[Task], seed: int, limit: int) -> list[Task]:
    """Shuffle by `seed` and take `limit`; the same seed is the same sample.

    Shuffle-then-truncate is deliberate: a smaller `--limit` is a prefix of a
    larger one, so two runs differ by the tasks they actually added, not by a
    second sampling rule. `--limit 0` keeps every task.
    """
    order = list(tasks)
    random.Random(seed).shuffle(order)
    return order[:limit] if limit > 0 else order


def _tools_for(task: Task) -> tuple[Tool, ...]:
    """The offered set: `calc` for a math task, none for a code task.

    A code task is checked by `verifiers.verify`'s own sandbox (the filesystem /
    rlimit jail), so it does not need a tool to be checkable. A math task is
    where the tool plumbing is measured.
    """
    return (calc_tool(),) if task.cls == "math" else ()


def _load_arm(spec: str, *, dry_run: bool, out_dir: Path, ckpt_name: str, device: str):
    """Resolve `scripted:<name>` or a checkpoint; return `(arm, state, error)`.

    `scripted` arms need no checkpoint, which is what makes `--dry-run` a real
    CI path. A checkpoint resolves through the snapshot exactly as `eval` does
    (`serve.load_checkpoint` → `cli._snapshot_run`), so this harness cannot
    grow a second checkpoint convention. A missing checkpoint is a STATE, not
    a traceback: the report still carries every metric key.
    """
    if spec.startswith("scripted:"):
        name = spec.split(":", 1)[1]
        if name not in SCRIPTS:
            return None, "error", f"unknown scripted model {name!r} (known: {sorted(SCRIPTS)})"
        if device not in ("cpu", "auto"):
            return None, "error", f"scripted arms are CPU-only (got --device {device})"
        return {"scripted": name, "model": None, "device": torch.device("cpu")}, "scripted", None
    if dry_run:
        return None, "error", f"--dry-run needs a scripted model, got {spec!r}"
    if ckpt_name:
        base, name = Path(out_dir), ckpt_name
    elif spec:
        path = Path(spec)
        base, name = path.parent, path.name
    else:
        return None, "error", "no model: pass --model scripted:<name> or a checkpoint prefix"
    pair = (base / f"{name}.pt", base / f"{name}.config.toml")
    missing = [str(p) for p in pair if not p.is_file()]
    if missing:
        return None, "no-model", f"missing checkpoint files: {', '.join(missing)}"
    from dormouse.serve import load_checkpoint

    try:
        model, _run, dev = load_checkpoint(base, name, device)
    except LoudError as exc:
        return None, "load-error", f"checkpoint did not load: {exc}"
    except Exception as exc:  # a corrupt artifact is a state, not a traceback
        return None, "load-error", f"checkpoint did not load: {type(exc).__name__}: {exc}"
    return {"scripted": None, "model": model, "device": dev}, "ok", None


def _model_for_task(arm: dict):
    """A fresh scripted model per task; a loaded model is shared and stateless."""
    if arm["scripted"] is not None:
        return ScriptedModel(SCRIPTS[arm["scripted"]], max_seq_len=64)
    return arm["model"]


# ---------------------------------------------------------------------------
# one rollout, with failures counted instead of raised
# ---------------------------------------------------------------------------


def _format_label(message: str) -> str:
    """Bucket a `parse_turn` refusal by what the plumbing got wrong."""
    if "unknown tool" in message:
        return "invalid_name"
    if "not valid JSON" in message or "not an object" in message:
        return "invalid_json"
    if any(
        s in message for s in ("missing required arg", "unknown arg", "declares type", "expected ")
    ):
        return "invalid_args"
    return "malformed"


def _screen_label(message: str) -> str:
    """The reason out of `_run_tool`'s sandbox-screen refusal, else a stable tag."""
    if "refused by the sandbox screen" in message:
        tail = message.split("—", 1)[-1]
        label = tail.split(" is not allowed", 1)[0].strip()
        return label or "sandbox-screen"
    return "sandbox-screen"


def _execute(tools: dict[str, Tool], call) -> tuple[str, str, str | None]:
    """Run one accepted call: `(status, reason, observation)`.

    `refused` is the escape path. The static screen (`verifiers.screen_code`,
    the jail's first layer) reads the model's own ARGS before the callable is
    touched, so `{"expr": "import subprocess"}` is a process-spawn finding even
    though the tool is a calculator. `error` is an ordinary failure (the tool
    refused its input, the child exited, a timeout) and is not a security count.
    """
    reason = screen_code(call.raw_args)
    if reason is not None:
        return "refused", reason, None
    try:
        observation = _run_tool(tools[call.name], call.args)
    except LoudError as exc:
        message = str(exc)
        if "refused by the sandbox screen" in message:
            return "refused", _screen_label(message), None
        if "timed out" in message:
            return "error", "timeout", None
        if "exited" in message:
            return "error", "sandbox-exit", None
        return "error", "tool-refused", None
    except Exception as exc:  # model output is the trust boundary; never crash the run
        return "error", f"{type(exc).__name__}", None
    return "ran", "", observation


def _generate(model, window: bytes, room: int, *, temp: float, device, generator) -> bytes:
    out = decode_bytes(
        model,
        window,
        room,
        temp=temp,
        device=device,
        generator=generator,
        stop=frozenset({END, TOOL_END, PAD}),
    )
    assert out.startswith(window), "decode_bytes must return prompt + new bytes"
    return out[len(window) :]


def _rollout(
    task: Task,
    tools: tuple[Tool, ...],
    model,
    *,
    device,
    budget: int,
    max_turns: int,
    temp: float,
    generator,
) -> dict:
    """One task, up to `max_turns`, with every failure attached to the record.

    `format` is measured on TURN 0 only: one prompt, one produced turn, exactly
    what "can the model emit a call" means. `execution` counts every accepted
    call. `pass@1` is true only when the first turn alone verified (no tool
    call, no second turn) — a one-turn rollout. `pass@budget` is the final
    verdict within `max_turns`.
    """
    record = {
        "id": task.tid,
        "cls": task.cls,
        "tools": [t.name for t in tools],
        "turns": 0,
        "calls": 0,
        "format": None,
        "executions": [],
        "passed": False,
        "pass@1": False,
        "verify": "",
        "failure": "",
    }
    schemas = {t.name: t.schema for t in tools}
    by_name = {t.name: t for t in tools}
    segments = [Segment("user", task.prompt)]
    try:
        transcript = render_conversation(segments)
    except LoudError as exc:
        record["failure"] = f"prompt: {exc}"
        return record
    if len(transcript) > budget:
        record["failure"] = f"prompt {len(transcript)} B over a {budget} B budget"
        return record
    per_turn = budget // max_turns
    answer = bytearray()
    stopped = False
    for turn in range(max_turns):
        window = transcript + bytes([ASSIST])
        room = min(per_turn, budget - len(window))
        if room < 1:
            record["failure"] = f"no turn room at turn {turn} ({len(window)} B of {budget})"
            break
        try:
            generated = _generate(
                model, window, room, temp=temp, device=device, generator=generator
            )
        except Exception as exc:
            record["failure"] = f"decode:{type(exc).__name__}: {exc}"
            break
        record["turns"] = turn + 1
        try:
            text, call = parse_turn(generated, schemas, offset=len(window))
            parse_error = None
        except LoudError as exc:
            text, call, parse_error = b"", None, str(exc)
        if turn == 0 and tools:
            if parse_error is not None:
                record["format"] = _format_label(parse_error)
            else:
                record["format"] = "accepted" if call is not None else "no_call"
        if parse_error is not None:
            record["failure"] = f"parse: {parse_error}"
            break
        if call is None:
            stop_reason = "no_call"
            if END in text:
                text = text[: text.index(END)]
                stop_reason = "end"
            elif PAD in text:
                text = text[: text.index(PAD)]
                stop_reason = "pad"
            answer += text
            segments.append(Segment("assistant", text))
            if stop_reason == "end":
                segments.append(Segment("end"))
            try:
                transcript = render_conversation(segments)
            except LoudError as exc:
                record["failure"] = f"render: {exc}"
                break
            if len(transcript) > budget:
                record["failure"] = f"transcript {len(transcript)} B over a {budget} B budget"
                break
            stopped = True
            break
        record["calls"] += 1
        status, reason, observation = _execute(by_name, call)
        record["executions"].append({"call": call.name, "status": status, "reason": reason})
        if status != "ran":
            record["failure"] = f"execution:{reason}"
            break
        answer += text
        trial = [
            *segments,
            Segment("assistant", text),
            Segment("call", name=call.name, args=call.raw_args),
            Segment("observation", observation),
        ]
        try:
            transcript = render_conversation(trial)
        except LoudError as exc:
            record["failure"] = f"observation: {exc}"
            break
        if len(transcript) > budget:
            record["failure"] = f"transcript {len(transcript)} B over a {budget} B budget"
            break
        segments = trial
    else:
        record["failure"] = f"max_turns={max_turns} were all tool calls"

    if not stopped:
        return record
    try:
        verdict = verify(task.as_dict(), bytes(answer).decode("utf-8", "replace"))
    except LoudError as exc:
        record["verify"] = f"refused: {exc}"
        return record
    except Exception as exc:  # the verifier is the trust boundary for model output too
        record["verify"] = f"error: {type(exc).__name__}: {exc}"
        return record
    record["passed"] = bool(verdict.passed)
    record["verify"] = verdict.reason
    record["pass@1"] = bool(record["passed"] and record["turns"] == 1 and record["calls"] == 0)
    return record


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def _empty_metrics() -> tuple[dict, dict, dict, dict]:
    """Every metric present, with an empty denominator, so `no-model` is a state."""
    fmt = {
        "prompts": 0,
        "accepted": 0,
        "rate": None,
        "invalid_name": 0,
        "invalid_json": 0,
        "invalid_args": 0,
        "no_call": 0,
        "malformed": 0,
    }
    execution = {"accepted": 0, "ran": 0, "rate": None, "refused": 0, "errors": 0}
    success = {
        key: {"tasks": 0, "passed": 0, "first": 0, "pass@1": None, "pass@budget": None}
        for key in ("math", "code", "all")
    }
    escape = {"attempts": 0, "by_reason": {}, "findings": False}
    return fmt, execution, success, escape


def build_report(
    tasks: list[Task],
    per_task: list[dict],
    *,
    state: str,
    model: str,
    tasks_path: str,
    limit: int,
    seed: int,
    budget: int,
    max_turns: int,
    temp: float,
    dry_run: bool,
    error: str | None = None,
) -> dict:
    """Aggregate per-task records into the four metrics; never blend them."""
    fmt, execution, success, escape = _empty_metrics()
    format_counts: Counter[str] = Counter()
    execution_counts: Counter[str] = Counter()
    escape_counts: Counter[str] = Counter()
    stalled = 0
    for rec in per_task:
        if rec["tools"]:
            format_counts["prompts"] += 1
            format_counts[rec["format"] or "malformed"] += 1
        for call in rec["executions"]:
            execution_counts["accepted"] += 1
            execution_counts[call["status"]] += 1
            if call["status"] == "refused":
                escape_counts[call["reason"]] += 1
        key = rec["cls"] if rec["cls"] in ("math", "code") else None
        for bucket in ([key] if key else []) + ["all"]:
            success[bucket]["tasks"] += 1
            success[bucket]["passed"] += int(rec["passed"])
            success[bucket]["first"] += int(rec["pass@1"])
        if rec["failure"]:
            stalled += 1
    fmt["prompts"] = format_counts["prompts"]
    for key in ("accepted", "invalid_name", "invalid_json", "invalid_args", "no_call"):
        fmt[key] = format_counts[key]
    fmt["malformed"] = format_counts["prompts"] - sum(
        fmt[k] for k in ("accepted", "invalid_name", "invalid_json", "invalid_args", "no_call")
    )
    fmt["rate"] = fmt["accepted"] / fmt["prompts"] if fmt["prompts"] else None
    execution["accepted"] = execution_counts["accepted"]
    execution["ran"] = execution_counts["ran"]
    execution["refused"] = execution_counts["refused"]
    execution["errors"] = execution_counts["error"]
    execution["rate"] = execution["ran"] / execution["accepted"] if execution["accepted"] else None
    for key in ("math", "code", "all"):
        n = success[key]["tasks"]
        first = success[key].pop("first")
        success[key]["pass@1"] = first / n if n else None
        success[key]["pass@budget"] = success[key]["passed"] / n if n else None
    escape["attempts"] = sum(escape_counts.values())
    escape["by_reason"] = dict(sorted(escape_counts.items()))
    escape["findings"] = escape["attempts"] > 0
    report = {
        "state": state,
        "model": model,
        "tasks": tasks_path,
        "n_tasks": len(tasks),
        "limit": limit,
        "seed": seed,
        "budget": budget,
        "max_turns": max_turns,
        "temp": temp,
        "dry_run": dry_run,
        "format": fmt,
        "format_invalid_name": fmt["invalid_name"],
        "execution": execution,
        "task_success": success,
        "escape_audit": escape,
        "per_task": per_task,
    }
    if stalled:
        report["failed_tasks"] = stalled
    if error is not None:
        report["error"] = error
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Four separate agent capabilities on a checkpoint (or a scripted arm)."
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help="task JSONL")
    parser.add_argument("--limit", type=int, default=0, help="task count after the seeded shuffle")
    parser.add_argument(
        "--seed", type=int, default=0, help="shuffle seed (same seed = same sample)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="`scripted:<name>` or a checkpoint prefix; default scripted:valid without --ckpt-name",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--ckpt-name", default="")
    parser.add_argument("--device", default="cpu", help="cpu|cuda|auto (default cpu)")
    parser.add_argument("--budget", type=int, default=1024, help="transcript bytes for one rollout")
    parser.add_argument("--max-turns", type=int, default=4, help="the pass@budget denominator")
    parser.add_argument("--temp", type=float, default=GREEDY_TEMP)
    parser.add_argument("--dry-run", action="store_true", help="require a scripted arm, no weights")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit < 0:
        print("eval_agent: --limit must be >= 0", file=sys.stderr)
        return 2
    if args.max_turns < 1:
        print("eval_agent: --max-turns must be >= 1", file=sys.stderr)
        return 2
    if args.budget < 1:
        print("eval_agent: --budget must be >= 1", file=sys.stderr)
        return 2
    if args.temp <= 0:
        print("eval_agent: --temp must be > 0", file=sys.stderr)
        return 2
    try:
        tasks = load_rl_tasks(args.tasks)
    except LoudError as exc:
        print(f"eval_agent: {exc}", file=sys.stderr)
        return 2
    sampled = _sample(tasks, args.seed, args.limit)
    model_spec = args.model
    if model_spec is None:
        model_spec = "" if args.ckpt_name else "scripted:valid"
    arm, state, error = _load_arm(
        model_spec,
        dry_run=args.dry_run,
        out_dir=args.out_dir,
        ckpt_name=args.ckpt_name,
        device=args.device,
    )
    if state == "error":
        print(f"eval_agent: {error}", file=sys.stderr)
        return 2
    per_task: list[dict] = []
    if arm is not None:
        generator = torch.Generator(device=arm["device"])
        generator.manual_seed(args.seed)
        for task in sampled:
            per_task.append(
                _rollout(
                    task,
                    _tools_for(task),
                    _model_for_task(arm),
                    device=arm["device"],
                    budget=args.budget,
                    max_turns=args.max_turns,
                    temp=args.temp,
                    generator=generator,
                )
            )
    report = build_report(
        sampled,
        per_task,
        state=state,
        model=model_spec or str(args.out_dir / args.ckpt_name),
        tasks_path=str(args.tasks),
        limit=args.limit,
        seed=args.seed,
        budget=args.budget,
        max_turns=args.max_turns,
        temp=args.temp,
        dry_run=args.dry_run,
        error=error,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 1 if state == "load-error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
