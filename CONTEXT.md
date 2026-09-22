# dormouse

A byte-level language model trained for maximum capability per gigabyte on one consumer GPU plus host RAM. This file is the glossary: the words the code, docs, and ADRs share.

## Language

### Model

**Core**: the accelerator-resident part of the model, everything except Engram tables. Small on purpose; capacity lives in memory.
_Avoid_: trunk (reserved for the patched byte stream), backbone

**LoopBlock**: the weight-shared recurrent block; each iteration runs controller, attention, memory, experts. Depth comes from iterations, not layers.
_Avoid_: layer stack, recursion unit

**Iteration**: one pass of the loop over the sequence. Distinct from a training step.
_Avoid_: cycle, ponder step

**KDA**: the gated-delta linear attention arm. Holds sequence state in constant memory; earns its place at long context.
_Avoid_: linear attention (ambiguous), GDN

**MSA**: the sparse attention arm with a learned block indexer. Judged by the long gate, not by short-context A/Bs.
_Avoid_: sparse attention (generic)

**TSCT**: the spectral low-rank expert bodies inside the MoE. An unproven swap; must beat plain small experts at matched active params or die by A/B.

**GR**: Gated Residual. Four-branch residual with gated read and scalar writes. Off by default.

**Teacher**: the EMA copy of the model used by JEPA, advanced after every optimizer step.
_Avoid_: reference model (RL meaning), evaluator

**DSpark**: the next-K draft head, a byte-level multi-token prediction aux.

**Patching**: grouping bytes into patches so core steps cover more bytes per step while prediction stays byte-level. Pending its A/B.
_Avoid_: tokenization (it is not tokenization)

### Memory

**Engram**: hashed n-gram embedding tables living in host RAM, trained by CPU Adam, rows fetched per batch. Input side only; growing it is the main capacity lever.
_Avoid_: lookup table (generic), n-gram cache

**Host rows**: the per-position Engram rows a batch needs, prepared on CPU before the step.

### Training

**Recipe**: a named config plus its validated hyperparameters. The current best is real18.
_Avoid_: run, experiment

**Preset**: a built-in config skeleton (nano, small, base, swift50, one_b). All presets share one architecture; they differ in size only.

**Muon+**: the matrix optimizer with update-RMS matched to AdamW so learning rates transfer. Matrices only; embeddings, heads, tables stay on the Adam family.
_Avoid_: Muon (the upstream optimizer)

**Guard**: the in-process NaN/panic re-exec that resumes from the last checkpoint with a fresh CUDA context.

### Measurement

**BPB**: bits per byte on the eval tail at a fixed step budget and memory envelope. The one score.
_Avoid_: loss (raw CE is not comparable across configs), perplexity

**Eval tail**: the held-out byte carve of the real corpus. Grows with the context ladder so long gates keep enough windows.
_Avoid_: test set, validation split

**Smoke**: a 200-500 step run screening NaN, speed, early slope. A filter, never a verdict.

**Confirm**: a 2k+ step run producing a BPB verdict for a smoke survivor.

**Long gate**: BPB at distance on the eval tail, bytes at 512k-1M positions against 1-4k, after each context extension.

**A/B or death**: the policy that every mechanism beats its own removal on BPB or is deleted. A tie deletes the mechanism.

**Drift check**: the comparison of a resumed run's resolved config against its stored snapshot; mismatch fails loudly.

**Context ladder**: the staged extension 512, 4k, 32k, 256k, 1M. Most tokens at the bottom; each rung passes the long gate.

**Flagship**: the target large run the small-preset A/Bs are voting on.
