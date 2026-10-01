# Security policy

## Reporting a vulnerability

**Email the maintainer directly** — the address in the git log of any commit
(`git log -1 --format='%ae'`) — rather than opening a public issue. Do not open
a `.github/ISSUE_TEMPLATE/` issue for a suspected vulnerability: those are
public the moment they are filed.

What to include, and why each field changes the response:

- **the commit** you tested (`git rev-parse HEAD`). This repository moves fast
  and a report without it cannot be reproduced.
- **the configuration**, if the report involves a run: the flags, or
  `checkpoints/<name>.config.toml`. AGENTS.md §1.4 makes the same demand of our
  own numbers, and it applies to a bug report for the same reason — several
  retracted results here were the right arithmetic measured on the wrong
  configuration.
- **the exact command**, and whether it needs the GPU. The CUDA gates and the
  CPU gates test different things; backend-specific defects (AGENTS.md §2.6)
  cannot reproduce on the CPU backend at all.

## What is in scope

The model, the trainer, the CLI, and the vendored library under `vendor/`.

**The training loop is the interesting target and the hardest to attack from
outside**: it is a trainer for one owner's models on one owner's GPU, it
exposes no network service, and the one always-on listener (`serve`) binds
locally. Expect slow triage on anything requiring a GPU reproduction, and say
so in the report if that is a blocker for you.

## What is out of scope

- **The vendored crates' own upstream defects.** `vendor/cubecl-fix` and
  `vendor/cubek-fix` are patched forks of CubeCL; report those upstream at
  <https://github.com/tracel-ai/cubecl> so the fix reaches everyone, and
  mention it here if the patch we carry is what exposed it.
- **Dependabot alerts on transitive dependencies** without a demonstrated
  impact in this tree. A version bump that changes no behaviour is not a
  vulnerability report.
- **A model that produces bad output, or a run that goes NaN.** Those are
  correctness and stability questions about the research, not security, and
  AGENTS.md §2.3 plus the ADR records are where that class lives.
- **Model weights and checkpoints** in `checkpoints/` — they are gitignored and
  not published.

## Non-production software

This is research software with no release process, no published artifacts and no
support commitment. There is no CVE programme here, and a report will be
handled as an ordinary fix or declined with a reason, in the open, on a
best-effort basis.