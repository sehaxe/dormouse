# `burn-kda` tier-(a) oracle — provenance

Written 2026-09-30 at `wt/kda-formula`, off `3234ecc`. Every sha, path, line and
number below was read out of the environment on that date; the commands are in
`transcript.txt`.

## What this oracle is compared against

| | |
|---|---|
| **repo** | <https://github.com/fla-org/flash-linear-attention> |
| **commit** | `9f38d24980c46d46bd38614e743cdacd21906578` (2026-09-29, HEAD of `main` at fetch) |
| **why this is the authors' code** | arXiv:2510.26692 (Kimi Linear) names `fla/ops/kda` as the official KDA implementation in its own footnote 1. `fla/ops/kda/gate.py:8` carries the line *"This file is modified and supported by the Moonshot AI Team"* — the decay gate is **Moonshot's own file**, not a third-party reimplementation. |
| **files pinned** | `fla/ops/kda/gate.py`, `fla/ops/kda/naive.py` |
| **functions used** | `naive_kda_gate`, `naive_kda_lowerbound_gate` (gate.py) and `naive_recurrent_kda`, `naive_chunk_kda` (naive.py) |
| **fetched and run** | 2026-09-30, CPU only, `torch==2.14.0+cpu` + `einops==0.8.2` |

The four functions are FLA's **pure-PyTorch reference implementations** of the
same math — the correctness oracle FLA's own tests compare its Triton kernels
against (`tests/ops/test_kda.py:20` imports `naive_chunk_kda,
naive_recurrent_kda`; the `naive_*_gate` twins live in the same file as the
`fused_kda_gate` they check).

**Why those and not the shipped Triton kernels.** FLA's shipping KDA is Triton
and Triton has no CPU backend, so the *fast* path can never be tier (a) on this
box. The reference path runs anywhere, on CPU torch, with no triton import at
all. This is the cheapest tier-(a) row available in the whole library and
nobody had taken it.

**The pin is byte-identical and machine-checked.** `gen_kda_oracle.py` asserts
the sha256 of both files before it imports anything and exits 2 with `PIN
ROTED` on a mismatch. No line of upstream source was edited, not even to add
this provenance header — which is why `docs/ORACLE-TIERS.tsv` carries a written
waiver on these two rows instead of a `github.com` URL in the file: the URL is
here, and the sha256 is enforced at generation time.

```
7ea46d62149b8e28e0827e070fee327576de783cb981e3b855182f2b8198c16e  fla_ops_kda_gate.py
60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016  fla_ops_kda_naive.py
```

They are byte-identical to
`https://codeload.github.com/fla-org/flash-linear-attention/tar.gz/9f38d24980c46d46bd38614e743cdacd21906578`,
re-verified at fetch time with `gen_kda_oracle.py --fla <clone> --verify-clone`.

## No function is re-typed, and why that is the whole point

Each of the four references is **extracted from the pinned file's bytes** and
`exec`'d (`extract_def`, scanning on column-0 boundaries). The code that
produces the fixture is upstream's text, not a transcription of it.

This is not fastidiousness. `tier-a-references.md` §7 records the failure mode
this repo has already paid for: a reference generator typed from memory is tier
(b) no matter how confident it looks, and it produced **6 false mismatches** in
one lane and **7** in another. The extraction also had to get the boundary
right — the first version stopped on the `) -> torch.Tensor:` line that closes
the parameter list and produced a `SyntaxError`, which is the cheapest possible
warning that a boundary scan is a parser and not a `grep`.

## The three vacuity guards, and all three fired

A golden file whose own gate would call it vacuous is the defect tier (a) exists
to remove, and the cheapest place to catch that is before it is written. The
generator exits 3 and writes nothing unless all three pass:

1. **Fidelity of the extraction.** Upstream's own recurrent and chunk
   references must agree with each other at `scale = 1.0` — two independent
   transcriptions of one recurrence inside one file, which is the check on the
   extraction before it reaches the fixture.
2. **Discriminating power.** The softplus cases must separate the two placements
   of `exp(A)` by more than the gate tolerance, or the fixture is blind to the
   defect it exists to catch. Computed in the generator as three lines of
   arithmetic and labelled as such — it is *not* a transcription of
   `src/lib.rs:268`.
3. **The published derivation.** FLA's *executed* reference must give
   `alpha = 0.0771` at our own init (`A = -3`, `b = +1`) — the number
   `src/lib.rs:58` publishes and derives in the "open discrepancy" section.
   **It does.** That derivation was ours; it no longer rests on our arithmetic
   alone.

**Guard 1's first version was wrong and is the most instructive thing here.** It
asserted that FLA's two gate references agree at `A = 0`. They do not:
`naive_kda_gate` gives `-softplus(z)` and `naive_kda_lowerbound_gate` gives
`-5·sigmoid(z)`. Those are two different *mechanisms* — Kimi Linear's unbounded
form and K3's bounded one — not two spellings of one formula, and the fixture's
`A_zero_control` case exists precisely because the two forms coincide at `A = 0`
*within* the softplus family. The guard was wrong; the extraction was fine.

## What it found, and what it did not

Two divergences from the source, both carried as **red tests on purpose** and
neither fixed, because a numerical change to a shipped model is the owner's
call:

| | upstream | ours | where |
|---|---|---|---|
| the read scale | `scale = K**-0.5` (`chunk.py:474`, `fused_recurrent.py:261`), and `fla/layers/kda.py:262` passes no `scale` so the official layer runs at that | `1.0` (`src/lib.rs:646,653`), no scale on the scan at all | §3.2 of the audit |
| Kimi-Linear decay | `-exp(A)·softplus(z)`, in the naive reference (`gate.py:50`) **and** the triton twin (`gate.py:167`) | `-softplus(exp(A)·z)` (`src/lib.rs:268`) | §3.1 of the audit |

And four agreements, three of them the first time anyone has executed FLA's KDA
code in this project: the **running** K3 decay form is character-for-character
`naive_kda_lowerbound_gate` with `g_min = -5`; Eq 1 is term-for-term
`naive_recurrent_kda` including the carried state; the chunked WY construction
is `naive_chunk_kda` on **both** of gdn2's chunk arms; and the output gate is
**sigmoid**, which is what K3 Eq 6 says and what `fla/layers/kda.py:191`
constructs.

`falsify.sh` runs 7 mutants. The valuable one is B1: it moves `exp(A)` outside
the softplus — the candidate fix for the softplus red — and that red goes
**green** with every other test's state unchanged. That is the statement that the
red is pinned to FLA's actual rule rather than to "not what we happen to have
written".

## Added 2026-10-01: the RoPE pin (`fla/modules/rotary.py`)

The `wt/rope-kda2` lane added a third pinned file, at **the same commit** and
under the same discipline:

```
6e9d1051751257370fd5f4f95ed4798adab642e1c59a9c0a3540adab440c5115  fla_modules_rotary.py
```

Extracted and `exec`'d exactly as the other two: `rotate_half`
(`fla/modules/rotary.py:21`) and `rotary_embedding_ref` (`:30`) are the
pure-PyTorch twin of the Triton `rotary_embedding_kernel` in the same file, and
`gen_kda_oracle.py` asserts the sha256 before importing anything. The three
`rope` blocks it adds to `fixtures/kda_oracle.txt` come from FLA's own rotation
of the raw q/k followed by the already-pinned `naive_chunk_kda` on the rotated
tensors.

**Two things this oracle is NOT, said here because the RoPE row is the easiest
one to overstate.**

1. **`RotaryEmbedding`'s cos/sin is TRANSCRIBED, not extracted.**
   `_compute_inv_freq` (`:410-414`) and `_update_cos_sin_cache` (`:419-447`) are
   methods of a class that cannot be `exec`'d without triton, so
   `gen_kda_oracle.py`'s `rope_cos_sin` writes them out as arithmetic. That is
   the **only** transcription in this generator, and the tier-(a) claim for the
   rope rests on a guard rather than on extraction: **VACUITY GUARD 4a/4b**.
   4a requires the rotation to move q/k by >= 1e-2 (measured 4.7-5.5); 4b
   requires the OUTPUT to move by >= 1e-2 relative to the un-rotated answer
   (measured 0.24 / 0.69 / 0.43 on the three cases). A wrong frequency — a
   dropped factor of 2, a `dim` that should be `dim/2` — fails both. If
   someone later finds a way to exec the class, replace the transcription and
   the guards become redundant rather than wrong.
2. **FLA's official KDA layer has no RoPE, so there is no upstream KDA answer to
   be faithful to.** `fla/layers/kda.py` at this commit contains zero
   `rotary`/`rope` occurrences, and GatedDeltaNet has no `use_rope` either. What
   is pinned here is **FLA's rotary**, used on a layer that upstream leaves
   without it. That makes the arm a **cross-family transplant** and it is why
   the placement question has an arithmetic answer rather than a citation one:
   RoPE preserves the L2 norm, so at a full-head rotation
   `l2(rope(x)) == rope(l2(x))` exactly and the order relative to the norm
   cannot matter. A *partial* rotation is a different function and is not in
   this fixture. The reasoning, with every line number:
   `research/reviews/rope-2026-09-30.md`.

## The coverage limit this oracle does NOT cover

Stated here because a tier-(a) row that overstates itself is worse than none.

**The `a_log` clamp.** `src/lib.rs:293` clamps `A_h` to `[-10, 20]`; no source
has a clamp. So a fixture case with `A` outside that range would make our
(deliberate, documented) clamped answer differ from FLA's unclamped one and turn
the *green* red. The fixture therefore contains no such case, and the
consequence is real: **a mutant that widens the clamp is invisible to every arm
of this oracle.** `falsify.sh`'s A3 narrows the clamp instead, which the greens
do see, and says why in place.

**The recurrent path has nowhere to put the scale.** `kda_step` and
`forward_recurrent` take no `scale` argument at all, so unlike the chunked call
site there is no single literal that would fix them. The chunked reds
deliberately drive `chunk_wy_forward` with an explicit scale so they compare the
*mechanism* against FLA, and the green twin
(`chunked_wy_honours_the_read_scale_when_asked`) is what proves the mechanism is
already right. A mutant that appeared to "fix" the scale reds would be measuring
the test rather than the code.

**The rope blocks carry no GVA.** All three cases have `HV == H`. A rotation
applied to the *value*-head axis, or to a repeated q, would be invisible here —
`rope_t38` is two heads precisely because it is the case that separates
"rotate within each head" from "rotate across the head boundary", and it does
not separate "rotate before the GVA repeat" from "after", because there is no
repeat. The flag's placement in `project` is above the repeat, which is correct
(q/k are on the key-head axis there, `lib.rs`'s "GVA: the decay is
parameterised on the KEY-head axis"), but nothing in the fixture checks it.

**The rope arm is a forward-only claim.** `tests/kda_rope.rs` compares forward
outputs. It says nothing about gradients: this project's fused chunked backward
is a separate and still-open question, and a rotation that is not
differentiated through is a *different* defect from a rotation that is wrong
(AGENTS.md §3.2, the `8fa5d4c` retraction).

