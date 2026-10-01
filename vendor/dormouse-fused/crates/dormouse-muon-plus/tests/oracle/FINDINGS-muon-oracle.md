# Findings — the Muon+ oracle lane, 2026-09-30

`wt/muon-oracle`, off `c3314e9`. Every number here is the output of a command in
`tests/oracle/transcript.txt` or in the probe runs quoted inline. Verdicts say
what was done: **fixed**, **confirm** (a claim in the tree, now backed by a
tier-(a) source), or **report** (found, not acted on, with `file:line`).

---

## 1. The paper's `√(m/n)` vs Jordan's `max(1, m/n)` — the justification is falsified. **REPORT.**

`vendor/dormouse-fused/crates/dormouse-muon-plus/src/lib.rs:456-479` (the comment above
`let lr_scaled = lr * (m / n).max(1.0).sqrt();`) says, in substance: *the
`max(1, ·)` is Jordan et al.'s; §3.1 adopts "the same configuration as in Jordan
et al." for the **polar operator**; Bernstein (2025) is the paper's own source
for `√(m/n)`; the two documents disagree and **this follows the implementation**.*

The first half is right. The last clause names the wrong implementation.
Measured on the Muon+ authors' own code, `K1seki221/MuonPlus@8a9ace12`:

- `utils/optim/muon_plus.py:245-252`, `adjust_lr_for_muon`:
  `scale *= math.sqrt(fan_out / fan_in)` — **no `max`**, and `rms_scaling=True`
  is the default (`muon_plus.py:152`).
- v3 §2.3 Eq. (4) and App. C Algorithm 1 line 10: `lr * (m / n) ** 0.5` — also
  no `max`.

So the Muon+ **paper** and the Muon+ **implementation** agree with each other
and both disagree with `KellerJordan/Muon`. The `max(1, ·)` we run is Jordan's
form, and the comment credits Jordan correctly while claiming the Muon+ side of
the disagreement is a document rather than the code. It is the code.

**No numeric change is implied.** The same comment's second claim — that the
`max` is inert on every shape the trainer routes here (TSCT factors `[in, k]` /
`[out, k]` are tall so `m/n > 1`; attention Q/K `[n_heads·head_dim, d]` are
square so `m/n = 1`) — is arithmetically correct, and `max(1, ·)` is 1.0 for
every `m ≤ n`. Reported, not changed: the constant is the trainer's and the
brief for this lane says do not change a training hyperparameter. **What should
change is the sentence that says which side of the disagreement is the code.**

## 2. The PolarExpress table in App. D.3 is not reproducible from the PolarExpress authors' own repository. **REPORT.**

`src/lib.rs:186-188` cites App. D.3's PolarExpress schedule and states it
"terminates at `(1.875, -1.25, 0.375)`". That is **true of the paper** (v3
App. D.3, verbatim) and **true of the Muon+ authors' copy**
(`muon_plus.py` → `polar_express.py:14-23`, the literal, verified byte-level and
now gated by `polarexpress_schedule_ends_where_our_doc_comment_says`).

It is not derivable from `NoahAmsel/PolarExpress`:

- `polar_express.py:80` at HEAD `71cc37943d99cae780024c1d198977f2f8795407` is
  `coeffs_list = optimal_composition(l=1e-3, num_iters=10, degree=5,
  safety_factor_eps=1e-2, cushion=0.02)` — **computed at import, not
  hardcoded.**
- Running it (torch 2.14.0+cpu, this box, 2026-09-30) gives a **10-entry** list
  whose 8th is `(1.8564, -1.2132, 0.3568)`, against App. D.3's
  `(1.875, -1.25, 0.375)`. Largest single-component difference on entry 1:
  **6.198e-1**.
- Sweeping the three parameters the API exposes does not recover the table:
  `cushion ∈ {0.02, 0.1} × safety_factor_eps ∈ {1e-2, 1e-3}` gives a worst
  deviation over the eight entries of **4.6e+0** at best. The generator named in
  the Muon+ authors' own comment (`OursFixedL(l=1e-3, cushion=1e-1, …)`) no
  longer exists in that file.
- None of the repository's **four** revisions of `polar_express.py`
  (`5454910920ca`, `df0fad249040`, `6f1bb73a2e1b`, `86c3c5e50e22`) has ever
  contained the literal. All four compute it.

Consequence: the D.3 table has **one** independent runnable witness (the Muon+
 authors' copy), not two. The crate's citation is not weakened — it cites v3 App.
D.3, which does print the table — but "verifiable against the algorithm's own
authors' code" would be false, and nothing in the tree claimed that. Recorded
here and in `tests/muon_oracle.rs` so the next reader does not try it.

(Independent corroboration: a parallel lane reached the same conclusion from the
same pinned file and additionally found `optimal_quintic` returns
`(1.875, -1.25, 0.375)` exactly. The terminal triple is safe; the eight-entry
schedule as a whole is not.)

## 3. Two authors' artefacts that disagree — worth knowing, no action. **REPORT.**

The Muon+ **paper** (Eq. 4, App. C line 3) says `M_t = μ·M_{t-1} + (1-μ)·G_t`
and **no** Nesterov lerp. The Muon+ **code** (`muon_plus.py:282-286`) says
`buf.mul_(momentum).add_(g)` — i.e. `μ·M_{t-1} + G_t`, no `(1-μ)` — and
`nesterov=True` by default (`muon_plus.py:151`), applying
`g = g.add(buf, alpha=momentum)`. Same file, `src/lib.rs:413-425` already says
"this file is the one quoting 2602.21545" and explains why. **Confirmed
correct**, now with the authors' code as the witness rather than the paper
alone. No change.

The `μ·M + (1-μ)G` vs `μ·M + G` difference is exactly a global constant:
`M_code_t = (1-μ)⁻¹·M_paper_t`, so it cancels in `orthogonalize`'s Frobenius
normalization. `src/lib.rs:421-425` says this; re-derived here, holds.

## 4. The authors' `norm_eps` parameter is dead code, and the effective epsilon is 1e-7. **REPORT — it changes what our oracle can be.**

`K1seki221/MuonPlus@8a9ace12` `utils/optim/muon_plus.py:87`:

```python
def apply_post_polar_norm(u, norm_mode: str, eps: float = 1e-8) -> torch.Tensor:
    """...
    eps is placed *inside* the sqrt: sqrt(sum_sq + 1e-7)
    """
    ...
    u = u / torch.sqrt((u * u).sum(dim=-2, keepdim=True) + 1e-7)   # line 105
```

`eps` is accepted and **never read**; lines 105, 116 and 118 hardcode `1e-7`.
The signature says `1e-8`, the docstring says `1e-7`, the paper's Algorithm 1
says `eps=1e-8`, and what the code runs is `1e-7`. Three different values in one
20-line function.

This is why the crate's `clamp_min(1e-7)` (`src/lib.rs:360-380`) is **not**
gated by the numeric oracle, and it is worth being precise about which value we
match: we match what they **run** (`1e-7`), not what their signature advertises
(`1e-8`). It also means the epsilon is out of the authors' reach as a
specification — there is no single upstream value to compare against.

## 5. v3 dropped the code link from its abstract. **REPORT.**

`aba296f` correctly moved this crate's citation to **v3**. v3 is the only
version with App. D, and also the only one whose abstract no longer names the
repository: v1 and v2 both end with *"We provide our code here:
https://github.com/K1seki221/MuonPlus."*; **v3 has no `github.com` string
anywhere in its text.** The tier-(a) provenance therefore lives in
`tests/oracle/PROVENANCE.md` and in `gen_oracle.py`'s header, not only in the
paper. Measured: `pdftotext -layout` over all three PDFs, `grep -c` — v1 2 hits,
v2 2 hits, v3 0.

## 6. The epsilon floor is invisible to this oracle. **STATED, not fixed.**

`falsify.sh` step C2: moving `clamp_min(1e-7)` to `clamp_min(1e-4)` leaves the
suite **green**. That is correct and expected — every axis norm in the fixture is
≥ 2.0e-3, so the floor never binds and any floor in that range gives an
identical output. It is also not reachable: where the floor binds, the two rules
(`sqrt(v²+1e-7)` inside the root, `max(v,1e-7)` outside it) are different
functions — at `v = 1e-6` they differ by **5e4** relative — so no single bar can
be both tight enough to catch a transposed axis (**measured 5.149e-1**, see
`transcript.txt` §4 step C) and loose enough to admit tiny-norm inputs.

Tiny-axis-norm inputs are therefore **excluded** from the numeric gate. That is
a stated limitation with a measurement attached, not a silent hole. A test for
the floor's own value would have to assert something the authors' code does not
determine, so it would be a test of our choice, not of our fidelity.

## 7. `src/lib.rs:27-30` and `:185-186` describe v2 wrongly. **REPORT.**

Both say v1 has appendices A–C and **"v2/v3 have A–G"**. Measured: v2 has
appendices A, B and C only — the same set as v1, and `diff v1.txt v2.txt` is
five hunks of typo fixes and table reflow. There is no App. D in v2, which is
also why `3.4445` and `1.875` each occur **0** times in v2.

The load-bearing half of both comments — cite v3, because v3 is where App. D
lives — is **correct**, and `aba296f`'s fix stands. The v2 claim is a leftover
from a reading of the version list rather than of the appendices. The test
`ns_coeffs_are_the_authors_literal` does not depend on it; the version claim is
now pinned by measurement in `tests/oracle/PROVENANCE.md` instead.

---

## What the oracle added, in one line

Before this lane, `dormouse-muon-plus` had three test files, all tier (d) — every
one comparing our code to our code. It now has a tier-(a) layer that **runs** the
Muon+ authors' own implementation (`K1seki221/MuonPlus@8a9ace12`, byte-pinned,
sha256-enforced), covering the constant at zero tolerance and the four
normalization directions at a 3e-5 bar with a **8 474x** margin over the defect
it exists to catch. The Newton-Schulz **iteration** has no numeric gate and
cannot have one against that source: their bf16 carries 2500x the noise of the
signal (`transcript.txt` §3). That is a real ceiling, stated with numbers, not
a gap left for later.
