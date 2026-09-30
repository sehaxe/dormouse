# Cross-process reproducibility of model init, by measurement

Lane: `wt/determinism3` (3rd agent; #1 and #2 died). 2026-09-30.
Gates ADR-0002's "3 seeds per arm", i.e. the whole A/B queue.

**STATUS: IN PROGRESS.** Written before the first number of my own, so a death
costs one section. Sections below marked (inherited) were produced by the dead
agents' harness in `/home/sehaxe/cache/determinism/`; everything marked
(measured here) is this agent's.

## Instrument note that changes the question (inherited, from `dmck.py`)

The brief's starting numbers were compared over the **whole 73 592 112-byte
checkpoint file** = 18 398 028 f32 slots. The `small` model section is
**9 197 390** slots. So the brief's denominator includes the **optimizer and
EMA-teacher** records, which are not parameters.

Further: `dmck.py` parses the burnpack record into named tensors with solved
byte offsets and self-validates the solve. At the correct base it reads
**0 NaN and 0 values >= 1e28 across all 9 197 390 slots of six runs**. The
brief's "0-4 NaN slots and ~475-501 slots at |v| >= 1e30" are therefore an
artifact of a **mis-aligned read window**, not of the file. To be confirmed
here.

## Deliverable 1: re-derive 0.167 % / 12.50 %   (pending)
## Deliverable 2: discriminate the cause          (pending)
## Deliverable 3: is "3 seeds per arm" implementable? (pending)
