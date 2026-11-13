---
title: What is next
description: Three steps, each with a budget and a way to fail.
---

The recipe is small on purpose; the plan is to spend the budget it saves.

## 1. Capacity without a slower step

Spectral factors plus ternary storage: store more parameters, compute on the same
number. The trade is measured, not assumed — at `r/f ≈ 9%` the loss penalty is
around 5% for ~2.7× less arithmetic, and the recipe only takes it when the freed
budget buys a model that wins against the smaller dense one.

**How it fails:** the larger factored model loses to the smaller dense model at
equal bytes. Then the rank is too low and the answer is a different rank, not a
different story.

## 2. Bytes per token

Decode is bandwidth-bound: a token costs one read of every active weight. Fewer
bits per weight is fewer bytes per token, which is the only lever that makes a
bigger model *faster* rather than slower.

**How it fails:** the unpack cost lands on the critical path and the kernel ends
up slower than the dense one at the same size.

## 3. A specialist

A model trained on execution-verified code traces, measured on our own test
suite rather than a public benchmark: hold out functions from this repository,
hide the body, ask the model to restore it, run the tests.

**How it fails:** pass@1 stays low enough that the loop cannot bootstrap, in
which case the honest conclusion is that the budget is too small for the task —
not that the method was wrong.

## What none of it needs

A cluster. Every step above fits the same one-card budget the project has always
had; what it needs is data, and data is the constraint that actually binds.

The dated queue, including what is blocked and why, is in
[`docs/archive/`](/dormouse/docs/archive/) and in `AGENTS.md`.
