# External oracle research for `burn-spectral`'s Newton–Schulz orthogonalisation

**Read `docs/papers/spectral-reference.md` first.** This directory is the
evidence it cites.

## What is here

- `upstream/` — third-party sources, **pinned byte-for-byte**, so a future gate
  needs no network. Provenance (repo, tag/SHA, fetch date, sha256) is the table
  in §5 of the document. **Never hand-edit these**; they are transcripts of what
  actually ran.
- `*.py` — the scripts that produced `transcripts/`. Every number in the
  document came from one of these.
- `transcripts/` — the captured output, 2026-09-30. All six scripts re-run
  byte-identically.

## Re-running

```
/tmp/opencode/oracle-venv/bin/python <script>.py     # from THIS directory
```

`python3` alone will not do: the pinned Polar Express reference imports `torch`
and the first run of this work died on `ModuleNotFoundError: No module named
'torch'`. The venv is `python 3.12.14 / torch 2.14.0+cpu / numpy 2.5.2`, created
by the `burn-rmsnorm` oracle lane and used read-only. See `transcripts/
interpreter.txt`.

If that venv is gone, recreate it with the RMSNorm recipe
(`vendor/burn-fused/crates/burn-rmsnorm/tests/oracle/gen_rmsnorm_oracle.py`
documents the exact `uv` invocation). Per the RMSNorm precedent, torch's
arithmetic is COMPILED C++ and not quotable — the wheel and version are what
this oracle is pinned to, not any number claimed to come out of it.

## What this directory is NOT

It contains **no Rust test and no fixture**. Per the task, the crate's tests
belong to another lane. What the other lane needs is in §4.3 of the document:
the two findings a LAPACK comparison produced, and the one comment whose stated
justification the measurement contradicts.

The scripts label every transcription as a transcription. Nothing here is
promoted to tier (a) on the strength of being written by us.
