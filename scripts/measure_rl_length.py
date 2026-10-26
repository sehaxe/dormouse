#!/usr/bin/env -S uv run --script
"""Measure correct-vs-incorrect rollout lengths, and justify `rl_len_penalty`.

The length penalty's ceiling X is a number, so it gets a measurement instead of
a taste. This script takes `--tasks` (default 200) from the RL corpus, trains the
same tiny model `tests/conftest.py` builds for `--train-steps` (default 250) on
a DISJOINT slice of the corpus, and measures the response length of every
rollout the verifier PASSES vs every one it FAILS.

At d=128 on CPU a random-init byte model does not solve arithmetic, so the
"correct" set is usually empty. The script does not hide that: it prints
`model_passes=0` and measures the correct set from the REFERENCE answers
rendered in the task's own final-line format, each put through the real
verifier (so they are true passing rollouts, just from an oracle policy). The
incorrect set is the model's own failing rollouts. Both numbers are labelled;
neither is inferred.

The decision rule printed at the end, and the one the shipped default follows:

    the longest correct answer sits `max/median` above the median correct
    length; at `s = 0.5` its ramp value is `(max/median - 1) / 0.5`; the
    penalty it pays is `X * ramp`. Choose the largest X in {0.1, 0.2, 0.3}
    whose longest-answer penalty is at most 10 % of a passing reward.

A passing reward is 1.0 (`R_test = 1`, and `S_sol = 1` when the response is the
reference itself), so 10 % is a stated budget, not a fit. If the measured
penalty exceeds it even at X=0.1, the script prints `X=0.0` — the term is not
worth shipping on that distribution.

    uv run port/scripts/measure_rl_length.py
    uv run port/scripts/measure_rl_length.py --tasks 200 --train-steps 250 --json out.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data_local" / "rl_tasks.jsonl"
SEQ = 128
TRAIN_SLICE = 300
EVAL_BUDGET = 200


def _load_corpus(path: Path, build: bool) -> list[dict]:
    if path.is_file():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not build:
        raise SystemExit(f"{path} does not exist. Run scripts/build_rl_tasks.py, or --build.")
    spec = importlib.util.spec_from_file_location(
        "build_rl_tasks", Path(__file__).with_name("build_rl_tasks.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_corpus(20261023, 500)


def _tiny_model():
    from dormouse.config import ModelConfig
    from dormouse.model import DormouseModel

    cfg = ModelConfig(
        d_model=128,
        n_heads=4,
        head_dim=32,
        d_ffn=256,
        max_seq_len=SEQ,
        max_iter=2,
        rank=8,
        n_experts=4,
        moe_topk=2,
        engram_rows=256,
        engram_dim=8,
        bf16=False,
        shared_rank=16,
    )
    return DormouseModel(cfg)


def _window(prompt: str, answer: str) -> bytes | None:
    from dormouse.train.sft import PAD, render

    raw = render(prompt.rstrip("\n"), answer)
    return None if len(raw) > SEQ else raw + bytes([PAD]) * (SEQ + 1 - len(raw))


def _train(model, windows: list[bytes], steps: int) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = 8
    model.train()
    for step in range(steps):
        rows = [windows[(step * batch + j) % len(windows)] for j in range(batch)]
        x = torch.tensor([list(w[:-1]) for w in rows], dtype=torch.long)
        targets = torch.tensor([list(w[1:]) for w in rows], dtype=torch.long)
        _logits, loss, _counters = model(x, targets=targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()


def _rollout_len(model, task, budget: int) -> tuple[bool, int]:
    from dormouse.post.rollouts import rollout

    record = rollout(model, task, budget, greedy=True)
    return record.passed, len(record.trace)


def _summary(lengths: list[int]) -> dict:
    if not lengths:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    ordered = sorted(lengths)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": statistics.median(ordered),
        "p90": p90,
        "max": ordered[-1],
    }


def _oracle_lengths(tasks) -> list[int]:
    """The reference answer rendered in the task's format, verified as a pass."""
    from dormouse.post.verifiers import verify

    lengths = []
    for task in tasks:
        if task.cls == "math":
            response = f"answer: {task.expected}"
        else:
            response = task.solution
        verdict = verify(task.as_dict(), response)
        if not verdict.passed:
            raise SystemExit(f"oracle answer for {task.tid} does not verify: {verdict.reason}")
        lengths.append(len(response.encode("utf-8")))
    return lengths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--build", action="store_true", help="build the corpus in memory if absent")
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="also write the measurement here")
    args = parser.parse_args(argv)

    from dormouse.post.tasks import Task

    records = _load_corpus(args.corpus, args.build)
    # Only the math kinds: a d=128 CPU model never emits a whole Python function
    # under a 200 B budget, so a code rollout would measure truncation, not the
    # answer's own length. The corpus's code half is still buildable and graded.
    math_records = [r for r in records if r["verifier"] == "math"]
    if len(math_records) < args.tasks + TRAIN_SLICE:
        raise SystemExit(f"need {args.tasks + TRAIN_SLICE} math tasks, got {len(math_records)}")
    order = list(range(len(math_records)))
    random.Random(args.seed).shuffle(order)
    eval_recs = [math_records[i] for i in order[: args.tasks]]
    train_recs = [math_records[i] for i in order[args.tasks : args.tasks + TRAIN_SLICE]]
    windows = [
        w
        for r in train_recs
        if (w := _window(r["prompt"], "answer: " + str(r["answer"]))) is not None
    ]
    if not windows:
        raise SystemExit("no training window fits; lower SEQ or raise --tasks? (train slice empty)")

    torch.manual_seed(args.seed)
    model = _tiny_model()
    print(f"training d=128 on {len(windows)} windows for {args.train_steps} steps (CPU)...")
    _train(model, windows, args.train_steps)

    tasks = [
        Task(
            cls="math",
            tid=r["id"],
            prompt=r["prompt"],
            expected=str(r["answer"]),
            solution=f"answer: {r['answer']}",
            tests=(),
            match="exact",
            budget=64,
        )
        for r in eval_recs
    ]
    passes: list[int] = []
    fails: list[int] = []
    for i, task in enumerate(tasks):
        passed, length = _rollout_len(model, task, EVAL_BUDGET)
        (passes if passed else fails).append(length)
        if i and i % 50 == 0:
            print(f"  rollouts: {i}/{len(tasks)} (model passes so far: {len(passes)})")

    model_correct = len(passes)
    correct = passes if passes else _oracle_lengths(tasks)
    correct_source = "model" if passes else "oracle-reference (model produced no passing rollout)"
    print(f"tasks={len(tasks)} model_passes={model_correct} model_failures={len(fails)}")
    print(f"correct source: {correct_source}")

    c = _summary(correct)
    f = _summary(fails)
    print(
        f"correct:   n={c['n']} mean={c['mean']:.2f} median={c['median']:.1f} "
        f"p90={c['p90']} max={c['max']}"
    )
    print(
        f"incorrect: n={f['n']} mean={f['mean']:.2f} median={f['median']:.1f} "
        f"p90={f['p90']} max={f['max']}"
    )
    ratio = c["max"] / c["median"] if c["median"] else float("inf")
    ramp = max(0.0, (ratio - 1.0) / 0.5)  # s = 0.5, delta = 0
    chosen = 0.0
    for candidate in (0.3, 0.2, 0.1):
        if candidate * ramp <= 0.10:
            chosen = candidate
            break
    print(
        f"correct max/median={ratio:.3f} -> ramp at s=0.5 is {ramp:.3f}; "
        f"X={chosen:.1f} costs the longest correct answer {chosen * ramp:.3f} "
        f"({100 * chosen * ramp:.1f} % of a passing reward; budget 10 %)"
    )
    print(f"chosen rl_len_penalty (X) = {chosen:.1f}")
    if args.json:
        payload = {
            "tasks": len(tasks),
            "model_passes": model_correct,
            "correct_source": correct_source,
            "correct": c,
            "incorrect": f,
            "max_over_median": ratio,
            "ramp_at_s_0_5": ramp,
            "chosen_x": chosen,
            "budget_fraction": 0.10,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
