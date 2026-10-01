# The output gate: SiLU or sigmoid — two authorities, one unrecorded choice

**Status: UNRESOLVED. A named A/B arm, not a bug and not a transcription error.**
Found 2026-09-29 by `research/reviews/gdn-fwd-review.md`.

## What the two upstreams say

Both fetched 2026-09-29 at `main`. **Neither is pinned to a SHA, so neither
citation is reproducible from this document alone** — that is a property of
upstream, stated here rather than papered over.

| source | line | code | gate |
|---|---|---|---|
| `NVlabs/GatedDeltaNet-2` | `lit_gpt/gdn2.py:212` | `self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)` | **SiLU** |
| `fla-org/flash-linear-attention` | `fla/layers/gdn2.py:197` | `self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)` | **sigmoid** |

Both files also agree on everything adjacent, which is what makes this a real
disagreement rather than two forks of different architectures: `q/k/v` are
`F.silu(proj(x))` in both (`nv_gdn2.py:304-306`, `fla_gdn2.py:252-254`), and
`b`/`w` are `sigmoid` in both (`nv_gdn2.py:319-320`, `fla_gdn2.py:263-264`).
The divergence is **only** the output gate.

## What we do, and why it is not verified

`vendor/burn-fused/crates/burn-gdn2/src/module.rs` applies `silu(gate)` — the
**NVlabs** choice, i.e. the original GDN-2 reference.

**The reason this is a finding and not a footnote: our own oracle cannot see
it.** `tools/gen_reference_f64.py` transcribes the FLA file, which selects the
*other* branch, while the kernel implements NVlabs. So the f64 fixture and the
kernel would agree on this line even if the kernel were wrong — a
transcription agreeing with a shared misunderstanding, which is exactly tier
(c) in `docs/ORACLE.md` §3. Every other column of `ref_f64.bin` is unaffected;
this one is structurally blind.

## The neighbouring arm already picks the other side

`burn-kda` applies `RMSNorm(o) ⊙ sigmoid(W_g x) ⊙ w_norm` per arXiv:2607.24653
§2.1.1 Eq. 6 — `gdn-kda.md` row 30, "verified against the paper". So the two
crates in this repository currently disagree with each other on the output
gate, deliberately and on the record. That is not necessarily wrong: KDA and
GDN-2 are different operators, and Kimi's paper is the authority for KDA.

## Why it is not a detail

`silu` has range `(-∞, 1)`; `sigmoid` has range `(0, 1)`. Swapping them
**removes the gate's entire negative half** — after the change the output gate
is strictly positive, where before it could suppress as well as excite. That
is an architectural difference in what the arm can express, not a numerical
one.

## What would settle it

ADR-0002 / `docs/AB-PROTOCOL.md`: this is a named arm, held-out BPB, 3 seeds,
one batch size so the eval window matches. **Not run.** No number is quoted
here because none has been measured — the effect on BPB is unknown to this
project, and the honest statement is that it has never been A/B'd.

## Also worth noting

The `b`/`w` gates are sigmoid in BOTH upstreams, and so are ours
(`module.rs:530-531`). Only the output gate is in question.
