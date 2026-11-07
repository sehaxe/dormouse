# Code style

Two lanes, one arbiter per lane. If a rule here disagrees with the tool, the tool
wins and this file is wrong — fix it in the same PR.

## Python (`port/`)

```bash
cd port
.venv/bin/ruff check src scripts tests        # must pass, zero findings
.venv/bin/ruff format src scripts tests       # must leave nothing to change
```

- **Line length 100.** Enforced by ruff (`E501`).
- **Rules:** `E, F, I, W, UP, B` today. New rules are added with a per-file
  ignore list and a count in the PR, never all at once: a 300-finding diff
  hides the one that matters.
- **Type hints** on every public function and every dataclass field. Internal
  helpers may skip them when the signature is obvious.
- **No bare `except`.** Catch the exception you can name; `except Exception`
  requires a comment saying what it protects against.
- **No silent fallbacks.** If a fast path is unavailable, the code either says so
  with a counter (`*_fallback`) or refuses with `Escape:`. The project's worst
  bugs were silent fallbacks that made a run look like a different run.
- **Errors:** `loud(...)` raises with an `Escape:` clause that names the fix.
  Every refusal must be actionable from its message alone.
- **Names carry the unit:** `bytes_seen`, `ms_per_step`, `bpb_fin`. A number
  without a unit in its name is a bug waiting for a misinterpretation.
- **Comments explain why, never what.** A comment that restates the code is
  deleted. A comment that records a measurement (`measured 2026-11-05: 0.462
  s/step at b8×1024`) stays, with the date.
- **No commented-out code, no `TODO` without an owner and a gate.** `ponytail:`
  marks a deliberate shortcut with its ceiling and upgrade path.

## Rust (`crates/`)

`cargo fmt` and `cargo clippy -- -D warnings` are the arbiters. Documentation on
every `pub` item (the crate denies `missing_docs` in places). Comments in the
same spirit as Python: why, with the measurement.

## Tests

- One file per behaviour area, named for the invariant it protects, not the
  function it calls: `test_no_future_bytes.rs`, `test_muon_reference.py`.
- Every test that guards a fix must fail on the un-fixed code. Show it red once
  in the PR body.
- Numerical tests state their tolerance and where it came from. `1e-6` without a
  reason is a guess; `1e-6 (bf16 ulp at 1.0)` is a fact.
- Timing tests do not exist. Anything that depends on wall-clock is a benchmark
  (`bench_*.py`) or a structural test (count calls, assert the shape).

## Docs

- `docs/reviews/` — what we measured. Numbers with config, date, commit, log.
- `docs/research/` — what the literature says. arXiv id, version, date, table.
- `docs/adr/` — what we decided and why, with the number that decided it.
- `docs/guides/` — how to do a thing.
- Every path mentioned in a comment or doc must exist:
  `python3 tools/check_doc_refs.py`.
