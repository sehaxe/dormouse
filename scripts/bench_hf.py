#!/usr/bin/env python3
"""Score a HuggingFace causal LM on THIS repo's MC tasks, in the repo's protocol.

Why: our benchmark numbers (`dormouse bench`) are byte-level, continuation-scored,
choice-normalised (`acc_mean`). A published number from a token-level harness
(acc_norm / acc_per_char) is a different instrument, and comparing the two is the
exact class of error this project has already paid for. This script runs a foreign
model through OUR loader and OUR scoring rule, so the only remaining difference is
the model.

The model is asked for its next-token distribution; for a choice of `m` BYTES the
score is `Σ_j log p(byte_j)` under the model's own tokenisation of the same
`context + choice` string. Because bytes are the unit on both sides (the choice
text is encoded to bytes, then decoded back to the same string for the tokenizer),
the comparison is byte-for-byte.

Usage:
    python scripts/bench_hf.py <model> --tasks arc_easy hellaswag --limit 200
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from dormouse.bench.tasks import load_task


def token_logprobs_bpb(model, tok, text: str) -> tuple[float, int]:
    """`(total bits, n_bytes)` for `text` under the model, teacher-forced."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    n_bytes = len(text.encode("utf-8"))
    if ids.shape[1] < 2:
        return (0.0, n_bytes)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logprobs = torch.log_softmax(logits[:-1], dim=-1)
    tgt = ids[0, 1:]
    lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (float(-lp.sum()) / math.log(2), n_bytes)


def score_choice(model, tok, context: str, choice: str) -> float:
    """Byte-normalised log-prob of the choice as a continuation of the context.

    `bits(context+choice) - bits(context)` is exactly the code length the choice
    costs, so dividing by the choice's byte count makes choices of different
    lengths comparable — the same normalisation our `acc_mean` uses.
    """
    full_bits, _ = token_logprobs_bpb(model, tok, context + choice)
    ctx_bits, _ = token_logprobs_bpb(model, tok, context)
    m = len(choice.encode("utf-8"))
    return -(full_bits - ctx_bits) / m


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tasks", nargs="+", default=["arc_easy"])
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    for task in args.tasks:
        items = load_task(task, limit=args.limit)
        if not items:
            print(f"{task}: no items (run `dormouse bench --download {task}`)")
            continue
        correct = 0
        for i, item in enumerate(items):
            scores = [score_choice(model, tok, item.context, c) for c in item.choices]
            best = max(range(len(scores)), key=lambda j: scores[j])
            correct += int(best == item.label)
            if (i + 1) % 20 == 0:
                print(f"  {task}: {i + 1}/{len(items)} acc={correct / (i + 1):.3f}", flush=True)
        print(f"{task}: acc_mean={correct / len(items):.4f} (n={len(items)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
