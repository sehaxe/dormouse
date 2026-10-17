"""The optimizer group router: Muon on 2-D params, AdamW everywhere else.

`burn-muon-plus`'s live policy, with the port's membership rule stated where a
reader can check it:

* a parameter goes to **Muon** iff it is 2-D with BOTH dims > 1 and is not a
  hashed-memory table. That is exactly the TSCT factor form `[d, r]` / `[r, f]`
  and the loop's projections;
* everything else — 1-D vectors (the `s` singular scales, norms, the iteration
  embeddings, `residual_scale`, the AttnRes queries) and the Engram tables —
  goes to **AdamW**;
* `opt = "adamw"` routes EVERYTHING to AdamW. That is the A/B control, and it
  is the same code with the routing switched off rather than a second trainer.

The update is `burn-muon-plus`'s, not a re-derivation:

* Newton-Schulz with the **Jordan** triple `(3.4445, -4.7750, 2.0315)`
  (`NS_COEFFS`, v3 App. D.1 "Jordan Coefficients") — the same polynomial, not
  the PolarExpress triple `(1.875, -1.25, 0.375)` that `burn-spectral`'s
  retraction uses. Two different quantities with two correct citations
  (AGENTS §3.7).
* Two Newton-Schulz **arms**, `NS_KINDS`: `classic` (the recurrence above, the
  default) and `gram` — Gram Newton-Schulz, the SAME polynomial iterated on the
  `[r, r]` Gram matrix instead of on `X`, which removes the two rectangular
  matmuls per iteration (arXiv:2608.11612 §4 + `Dao-AILab/gram-newton-schulz`,
  transcribed in `_gram_via_ns`). Equivalent to `classic` up to fp32 round-off
  order, and gated as such in `tests/test_muon_gram.py`.
* `M_t = μ·M_{t-1} + (1-μ)·G_t` with **no Nesterov lerp** (the paper's Eq. 4;
  the crate's manifest was the defect, not the code).
* `O_t = Norm_d(Ortho(M_t))` with ColRow normalisation, and the step
  `W -= η·max(1, m/n)^0.5·O_t` — `max(1, ·)` is Jordan's, which is what both
  the crate and this port follow.
* `orthogonalize` divides by the Frobenius norm, and the NS loop runs in the
  TRANSPOSED orientation when the matrix is tall, i.e. in the `[r, r]` Gram
  space of the smaller side — the burn economics that keeps `[768, 768]` out
  of fp32 NS entirely.
* Two precision **rungs**, both `fp32` by default so a live run's semantics and
  its resume do not change:
  - `muon_state_dtype` (`fp32` | `bf16`) stores the momentum buffers in bf16
    while the ACCUMULATION stays fp32. Source: arXiv:2509.23106 — a Muon variant
    with a low-precision state (bf16/8-bit storage with stochastic rounding,
    confirmed at the 8-bit rung); the citation comes from the task that
    specified this rung, **the paper itself was not read here**. This port
    implements the bf16 rung with a round-to-nearest store; **stochastic
    rounding is the NEXT rung and is not implemented here**, so the rung is
    approximate by design (gate: `tests/test_muon_fast.py`).
  - `ns_dtype` (`fp32` | `bf16`) runs the Newton–Schulz chain in bf16 with
    explicit fp32 in/out casts — nanogpt's practice, row 4 of
    `docs/research/2026-10-16-muon-orthogonalization-cost.md` (predicted
    −100…−200 ms at b1, NOT measured here: this file prices nothing).
* Two **direction / schedule arms**, both opt-in and both `off` by default, so
  a live run's semantics and its resume do not change:
  - `variant` (`MUON_VARIANTS`): `muon` (the default and every number on
    record) or `normuon` — NorMuon's row-wise EMA second moment of the
    post-polar update, applied the AUTHORS' repo way (`zichongli5/NorMuon`
    `normuon.py:48-56`: row-scale, restore the pre-normalization Frobenius norm,
    then `× max(1, m/n)^0.5`). That is **the repo Muon's magnitude with NorMuon's
    direction**, and it REPLACES our ColRow tail rather than composing with it:
    composed, the arm is INERT, because NorMuon's row scaling IS ColRow's row
    half (measured, in `normuon_repo_update_`'s docstring). Against our ColRow
    control the magnitude is therefore not preserved either (0.71x on a wide
    factor, 0.13x on a tall one, measured), so this arm is a REPLACE and not a
    direction-only knob — the numbers are in the lane report.
  - `sf` (schedule-free): the SF-NorMuon wrapper (arXiv:2605.23061, JPMorgan)
    around the same direction — the params hold `Y = (1−β1)Z + β1 X` during the
    forward, `X` is the returned/eval iterate, and the Muon-group LR is
    constant after warmup. Algorithm 1 + Appendix F, transcribed in
    `_sf_batched` / `sf_switch`; the docstring there names the two deliberate
    deviations (our ColRow stays composed; the AdamW half keeps WSD, so this is
    a HYBRID SF-Muon + WSD-AdamW, not the paper's full-model SF).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from ..instrument import Counters, loud

#: The paper's Jordan triple, `burn-muon-plus::NS_COEFFS`, verbatim.
JORDAN_NS_COEFFS = (3.4445, -4.7750, 2.0315)

#: The Newton–Schulz arms `zeropower_via_ns(kind=...)` runs. One tuple, so a
#: name that is not an arm is refused at the door instead of silently meaning
#: something (`config.NS_KINDS` mirrors this list for the config layer).
#: `triton` is the fused chain in `train/ns_kernels.py`: ONE kernel launch runs
#: the whole Jordan chain (the Gram lives in registers), against ~52 dispatches
#: for a `[2048, 64]` factor (measured, `scripts/bench_ns_triton.py`) — the one
#: arm whose mechanism is *fewer launches* on a launch-bound step
#: (`docs/research/2026-10-17-muon-variants.md` §2 #1). It is COUNTED and falls
#: back to `classic` with its reason when it cannot serve a call.
NS_KINDS = ("classic", "gram", "triton")

#: The 3-D `colrow_normalize` arms (`config.MUON_COLROW_3D` mirrors this list):
#: `frobenius` is the per-slice Frobenius normalisation the batched Muon step ran
#: before the 2026-10-17 flip (`9ee82c5`) — NOT the ColRow the port documents —
#: and `colrow` is the 2-D definition applied per slice, i.e. the fix and the
#: default since that flip. The defect, its exact size
#: and the A/B it owed are in `colrow_normalize`'s docstring and
#: `tests/test_muon_fast.py`.
COLROW_3D = ("frobenius", "colrow")

#: The rungs of BOTH precision knobs (`muon_state_dtype`, `ns_dtype`):
#: `fp32` is the default and every number on record; `bf16` is the storage /
#: chain rung below it. A name that is not here is refused at the door, at the
#: config layer (`config.OPT_DTYPES` mirrors this tuple) and at the constructor.
OPT_DTYPES = ("fp32", "bf16")

#: The `muon_variant` literals — the DIRECTION arms (`Muon.variant`, where they
#: execute; `config.MUON_VARIANTS` mirrors this list for the config layer).
#: `muon` is the update every number on record ran: momentum → optional ColRow →
#: NS polar, step `-lr·max(1, m/n)^0.5`. `normuon` keeps that pipeline and
#: replaces the tail with NorMuon's row-wise EMA second moment, applied the way
#: the AUTHORS' repo applies it (`normuon.py:48-56`: row-scale, then restore the
#: pre-normalization Frobenius norm, then `× max(1, m/n)^0.5`) — i.e. **Muon's
#: magnitude, NorMuon's direction**, which is what makes the arm a controlled
#: direction-only change against our Muon+ColRow control. A name that is not
#: here is refused at the door, at the config layer and at the constructor.
MUON_VARIANTS = ("muon", "normuon")

#: NorMuon's row second-moment EMA rate — `beta2 = 0.95` in BOTH references
#: (`zichongli5/NorMuon` `normuon.py:35`'s `beta2=0.95` default; `2605.23061v1`
#: Appendix F's `betas=(0.9, 0.95)`, i.e. `beta2 = betas[1]`).
NOR_BETA2 = 0.95

#: The `muon_variant="normuon"` arm's ε — the reference REPO's own `1e-10`,
#: spelled twice in `normuon.py`: `:51` `step_size = 1 / second_momentum.sqrt()
#: .add_(1e-10)` and `:54` `vnorm / (vnorm_new.add_(1e-10))`.
NOR_REPO_EPS = 1e-10

#: The schedule-free arm's ε — `2605.23061v1` Appendix F's `eps=1e-8`
#: (`Phat = P / (v.sqrt() + eps).unsqueeze(1)`). Deliberately NOT the repo's
#: `1e-10`: the `sf` arm is the SF paper's algorithm and the two references
#: differ by 100x, so one constant could not serve both without lying about one.
NOR_SF_EPS = 1e-8

#: The `0.2` in the SF arm's `eta_hat = eta_scale·eta·√(mn)/‖P̂‖_F`
#: (`2605.23061v1` Alg 1 line 13 + Appendix F's `eta_scale=0.2`, the "match
#: Adam's RMS scaling" factor). Used by the `sf` arm ONLY: the `normuon` arm
#: takes its magnitude from the repo's Frobenius rescale instead, which is what
#: makes that arm a DIRECTION-only change against the control.
NOR_ETA_SCALE = 0.2

#: The schedule-free interpolation parameter β1 (`Y = (1−β1)Z + β1 X`):
#: `betas=(0.9, 0.95)` in the reference, so `β1 = betas[0] = 0.9` (Appendix F:23).
SF_BETA1 = 0.9

#: Where Gram-NS restarts: 0-based iteration indices, exactly the reference's
#: `gram_newton_schulz_reset_iterations` default (`[2]`, i.e. a restart before
#: the third step) — "for 5 steps of Newton-Schulz, we recommend num-restarts =
#: 1 for maximum speed while maintaining numerical stability". Source: the
#: pinned upstream README,
#: `vendor/dormouse-fused/crates/burn-spectral/tests/oracle/upstream/
#: gram_newton_schulz_README.md` (tag v0.1.6 = `e45d0aca7083cb275c9a303220c05c4
#: abecd9187`, sha256 `c8701727…bf99f`, both recorded in
#: `docs/papers/spectral-reference.md`). A restart cannot move the math: it
#: materializes `Q X` and rebuilds the Gram from it, which is exactly what the
#: classic arm does at every step — but it does change the fp32 error, and
#: leaving it out costs accuracy (`tests/test_muon_gram.py`).
GRAM_NS_RESTARTS = (2,)

#: The batched Muon step's chunk cap, in fp32 ELEMENTS per stacked buffer
#: (`1 << 25` = 33.5M elements = 134 MiB): `[1024, 64, 5461]` fp32 batched NS
#: needs ~1.33 GiB per intermediate and the 5-iteration chain keeps ~4 GiB
#: live — the 16-GB card OOMs WITH the model resident (2026-10-15 b1 bench).
#: Chunks cap the batch's fp32 footprint; NS still runs on whole chunks. A
#: named constant so a test can shrink it to exercise the chunk boundary and
#: the b1 bench can sweep it against VRAM. The value is the one the step has
#: always used (`(1 << 27) // (d0 * d1 * 4)` params).
MUON_CHUNK_ELEMS = 1 << 25


def _check_ns_kind(kind: str) -> None:
    """A name that is not an arm is LOUD, naming every arm and the escape."""
    if kind not in NS_KINDS:
        loud(
            f"ns_kind={kind!r} is not an arm. Known: {list(NS_KINDS)} — `classic` "
            f"forms `X @ X.T` at every iteration (the default, byte-identical to "
            f"this port's history), `gram` is Gram Newton–Schulz (arXiv 2608.11612 "
            f"§4): the same polynomial iterated in the `[r, r]` Gram space, "
            f"`triton` is the same chain FUSED into one kernel launch "
            f"(`train/ns_kernels.py`, COUNTED fallback to `classic`). Escape: "
            f"name one of those, or land the arm in `optim.NS_KINDS` first."
        )


def _check_colrow_3d(arm: str) -> None:
    """A name that is not an arm is LOUD, naming both arms and the escape."""
    if arm not in COLROW_3D:
        loud(
            f"muon_colrow_3d={arm!r} is not an arm. Known: {list(COLROW_3D)} — "
            f"`frobenius` is the per-slice Frobenius normalisation the batched "
            f"Muon step ran before the flip (the 1/sqrt(d0)-of-ColRow defect, "
            f"kept as the A/B control), `colrow` is the 2-D definition applied per "
            f"slice (the fix, bit-identical to the per-param path, the default "
            f"since `9ee82c5`). Escape: name one of "
            f"those, or land the arm in `optim.COLROW_3D` first."
        )


def _check_dtype(field: str, rung: str) -> None:
    """A rung that is not implemented is LOUD, naming both rungs and the escape."""
    if rung not in OPT_DTYPES:
        loud(
            f"{field}={rung!r} is not a rung. Known: {list(OPT_DTYPES)} — `fp32` "
            f"is the default and every number on record, `bf16` is the low-precision "
            f"rung (Muon state: bf16 storage with fp32 accumulation, "
            f"arXiv:2509.23106; NS chain: bf16 polynomial with fp32 in/out casts, "
            f"nanogpt's practice). Escape: name one of those, or land the rung in "
            f"`optim.OPT_DTYPES` with its gate BEFORE the config can lie about it."
        )


def _check_muon_variant(arm: str) -> None:
    """A name that is not an arm is LOUD, naming both arms and the escape."""
    if arm not in MUON_VARIANTS:
        loud(
            f"muon_variant={arm!r} is not an arm. Known: {list(MUON_VARIANTS)} — "
            f"`muon` is the update every number on record ran (momentum -> optional "
            f"ColRow -> NS polar, step `-lr*max(1, m/n)^0.5`), `normuon` keeps the "
            f"momentum and the NS polar and REPLACES the ColRow tail with NorMuon's "
            f"row-wise EMA second moment applied the authors' repo way "
            f"(`normuon.py:48-56`: row-scale, restore the pre-normalization "
            f"Frobenius norm, then `max(1, m/n)^0.5`). Escape: name one of those, or "
            f"land the arm in `optim.MUON_VARIANTS` with its gate BEFORE the config "
            f"can lie about it."
        )


def _torch_dtype(rung: str) -> torch.dtype:
    """The torch dtype a checked rung stores in (`fp32` maps to `torch.float32`)."""
    return torch.bfloat16 if rung == "bf16" else torch.float32


def zeropower_via_ns(
    g: torch.Tensor,
    steps: int = 5,
    kind: str = "classic",
    gram_restarts: tuple[int, ...] = GRAM_NS_RESTARTS,
    ns_dtype: str = "fp32",
    *,
    counters=None,
) -> torch.Tensor:
    """The polar factor's polynomial approximation, matrix-only, fp32 by default.

    `kind` selects the arm (`NS_KINDS`): `classic` is the recurrence this port
    has always run; `gram` is Gram Newton–Schulz — the same polynomial,
    iterated on the `[r, r]` Gram and applied once at the end, so the two
    rectangular matmuls per iteration are gone (`_gram_via_ns`); `triton` is the
    SAME chain fused into ONE kernel launch (`train/ns_kernels.py`), which is the
    only arm here whose mechanism is a dispatch count — it is COUNTED and falls
    back to `classic` with its reason when it cannot serve the call, so `counters`
    is what a caller passes to see which chain ran.

    `ns_dtype` (`OPT_DTYPES`) is the chain's own dtype: `fp32` is the default
    and every number on record; `bf16` runs the polynomial in bf16 with
    **explicit fp32 casts at the door and at the exit** — the input is upcast
    and the Frobenius prescale runs in fp32 (it is one pass and it is what the
    polynomial's conditioning rests on), then the chain runs in bf16 and the
    result is returned in fp32. nanogpt's practice (row 4 of
    `docs/research/2026-10-16-muon-orthogonalization-cost.md`); the rung is
    approximate by design and the orthogonality gate is in
    `tests/test_muon_fast.py`.

    The batched form accepts `[N, d0, d1]`: the polynomial is ELEMENTWISE in
    the leading batch dim (a Gram per slice), and `x @ x.T` broadcasts
    unchanged, so the same math runs for N slices at once — that is the
    whole point of the batched Muon pass (version-branched off the same
    body, no second implementation exists here).
    """
    _check_ns_kind(kind)
    _check_dtype("ns_dtype", ns_dtype)
    if kind == "triton":
        # The fused chain, with its COUNTED fallback to `classic` — the whole arm
        # (including the reason it could not run) lives in `ns_kernels.py`, so the
        # fallback logic is testable without a GPU. `counters` is what makes the
        # degradation visible in the training log (ADR-0019); the trainer drains
        # it (`Muon.counters` -> `train/loop.py`).
        from .ns_kernels import zeropower_triton_or_classic

        return zeropower_triton_or_classic(g, steps, counters=counters, ns_dtype=ns_dtype)
    if kind == "gram":
        return _gram_via_ns(g, steps=steps, restarts=gram_restarts, ns_dtype=ns_dtype)
    if g.ndim == 2:
        a, b, c = JORDAN_NS_COEFFS
        x = g.to(torch.float32)
        x = x / (x.norm() + 1e-7)
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.T.contiguous()
        x = x.to(_torch_dtype(ns_dtype))
        for _ in range(steps):
            gram = x @ x.T
            poly = b * gram + c * (gram @ gram)
            x = a * x + poly @ x
        out = x.T.contiguous() if transposed else x
        return out.to(torch.float32)
    assert g.ndim == 3, f"NS is matrix-only or [N,d0,d1] batched, got shape {tuple(g.shape)}"
    a, b, c = JORDAN_NS_COEFFS
    x = g.to(torch.float32)
    x = x / (x.norm(dim=(1, 2), keepdim=True) + 1e-7)
    transposed = x.shape[1] > x.shape[2]
    if transposed:
        x = x.transpose(1, 2).contiguous()
    x = x.to(_torch_dtype(ns_dtype))
    for _ in range(steps):
        gram = x @ x.transpose(1, 2)
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x
    out = x.transpose(1, 2).contiguous() if transposed else x
    return out.to(torch.float32)


def _gram_via_ns(
    g: torch.Tensor,
    *,
    steps: int,
    restarts: tuple[int, ...] = GRAM_NS_RESTARTS,
    ns_dtype: str = "fp32",
) -> torch.Tensor:
    """Gram Newton–Schulz: the classic recurrence, regrouped on the Gram.

    Transcribed from the authors' code — `Dao-AILab/gram-newton-schulz`,
    `gram_newton_schulz/gram_newton_schulz.py::_gram_newton_schulz`, tag
    **v0.1.6 = `e45d0aca7083cb275c9a303220c05c4abecd9187`** (fetched 2026-10-16;
    byte-identical to `main` that day) — and from the pseudo-code in the same
    tag's README, pinned in-tree at
    `vendor/dormouse-fused/crates/burn-spectral/tests/oracle/upstream/
    gram_newton_schulz_README.md` (arXiv:2608.11612 §4 is the paper behind it).
    The port's orientation is kept — the small side first, `[r, d]` with
    `r <= d`, so the Gram is `[r, r]`, which is the space this port's classic
    arm already works in (`zeropower_via_ns` transposes when `rows > cols`).
    `restarts` are 0-based iteration indices at which to re-materialize `X` (the
    reference's `gram_newton_schulz_reset_iterations`, default `(2,)`).

    `ns_dtype` is the chain's own dtype (see `zeropower_via_ns`): the Gram, `Q`
    and the eye all follow it, and the fp32 in/out casts are explicit.

    The recurrence, with `P(R) = a·I + b·R + c·R²`:

        X0 = X / (‖X‖_F + 1e-7)
        R  = X0 @ X0ᵀ                 # the [r, r] Gram, formed ONCE
        Q  = I                        # the invariant is X_t == Q_t @ X0
        for t in 0..steps-1:
            if t in restarts and t != 0:      # restart, then rebuild both
                X = Q @ X;  R = X @ Xᵀ;  Q = I
            Z  = b·R + c·R²                   # 1 small matmul
            Q  = Z + a·I   (or  Q @ Z + a·Q when Q is live)     # 1 small matmul
            R  = Z·RZ + a·RZ,  RZ = R·Z + a·R # 2 small matmuls, skipped on the
                                              # last step / before a restart
        return Q @ X

    **Why it is the same polynomial**, which is the whole claim: the classic
    step is `X ← (a·I + b·R + c·R²)·X = P(R)·X`, so after `steps` of them
    `X = P(R_{steps-1}) … P(R_0)·X0`. Every `P(R_t)` is itself a polynomial in
    `R_0` (because `R_t = P(R_{t-1})²·R_{t-1}`), and polynomials of one matrix
    commute — so the whole product collapses to ONE `[r, r]` factor `Q` with
    `X = Q·X0`, and `R` can be carried alongside it in `[r, r]`:
    `R_t = P(R_{t-1})²·R_{t-1}`. What the classic arm spends per iteration is
    two RECTANGULAR matmuls (`X @ Xᵀ` and `P @ X`, both `O(r·d·r)`, `d` =
    2048/5461 at the b1 shapes); this arm spends four `[r, r]` ones and pays
    exactly TWO rectangular matmuls in total (`X0 @ X0ᵀ`, `Q @ X`). Same
    polynomial, different fp32 round-off order — no training A/B is licensed by
    the speed, only by the equivalence gate (`tests/test_muon_gram.py`).

    NOT transcribed, deliberately: the reference's step 2, `X → float16(X)`.
    This port runs `ns_dtype` (`fp32` by default; `bf16` is the same cast with
    the port's own dtype rung — row 4 of the cost note). Its CuTeDSL
    symmetric-GEMM kernels are Hopper/Blackwell-only and are not
    ported either — this is the ALGORITHM, in torch.
    """
    a, b, c = JORDAN_NS_COEFFS
    _check_dtype("ns_dtype", ns_dtype)
    assert g.ndim in (2, 3), f"NS is matrix-only or [N,d0,d1] batched, got shape {tuple(g.shape)}"
    x = g.to(torch.float32)
    x = x / (x.norm(dim=(-1, -2), keepdim=True) + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-1, -2).contiguous()
    x = x.to(_torch_dtype(ns_dtype))
    eye = torch.eye(x.shape[-2], dtype=x.dtype, device=x.device)
    restart = set(restarts)
    q: torch.Tensor | None = None
    gram = x @ x.transpose(-1, -2)
    for t in range(steps):
        if t in restart and t != 0:
            x = q @ x
            gram = x @ x.transpose(-1, -2)
            q = None
        z = b * gram + c * (gram @ gram)
        q = (z + a * eye) if q is None else (q @ z + a * q)
        if t + 1 < steps and (t + 1) not in restart:
            rz = gram @ z + a * gram
            gram = z @ rz + a * rz
    out = x if q is None else q @ x  # `q is None` only at steps == 0
    out = out.transpose(-1, -2).contiguous() if transposed else out
    return out.to(torch.float32)


def colrow_normalize(x: torch.Tensor, eps: float = 1e-7, *, colrow_3d: str = "colrow"):
    """`Norm_d` for `NormDir::ColRow` (the trainer's choice): columns, then rows.

    **The 2-D form is the definition and is untouched by `colrow_3d`.** The 3-D
    form has two arms (`COLROW_3D`), because the batched step's original 3-D
    branch was NOT ColRow — found 2026-10-16 by the equivalence gate in
    `tests/test_muon_fast.py`:

    * `"frobenius"` (**the historical default until the 2026-10-17 flip, and what
      every number on record ran**):
      collapses dims `(1, 2)` twice, i.e. normalises each slice by its Frobenius
      norm. The batched Muon step therefore applies exactly `1/sqrt(d0)` of the
      per-param update (8x at `[64, 5461]`/`[64, 2048]`, 45x at `[2048, 2048]`).
      This is the DEFECT, kept as the control so the A/B is apples-to-apples.
    * `"colrow"` (**the fix, and the default since that flip — `9ee82c5`**):
      dims 1 then 2 per slice — the 2-D definition
      applied to each matrix, which `tests/test_muon_fast.py` shows is
      bit-identical to running `colrow_normalize` on the slices one at a time.

    The arm is a config key (`train.muon_colrow_3d`) and an optimizer attribute,
    never a param-group key: the group is what `load_state_dict` REPLACES with
    the checkpoint's copy, so a key added today would KeyError on every
    checkpoint written before it (the same rule `ns_kind` follows). The switch
    moves the update scale 8-45x, so it owed the standing 3-seed A/B (AGENTS
    §1.2) and that A/B ran: **-0.0428 BPB against `"frobenius"`, 8.7x the
    control's own seed spread, zero seed overlap** — which is why `"colrow"` is
    the default and `"frobenius"` is kept as the control, not the other way
    round.
    """
    if x.ndim == 2:
        x = x / (x.norm(dim=0, keepdim=True) + eps)
        return x / (x.norm(dim=1, keepdim=True) + eps)
    _check_colrow_3d(colrow_3d)
    x = x / (x.norm(dim=(1, 2) if colrow_3d == "frobenius" else 1, keepdim=True) + eps)
    return x / (x.norm(dim=(1, 2) if colrow_3d == "frobenius" else 2, keepdim=True) + eps)


def colrow_normalize_(x: torch.Tensor, eps: float = 1e-7, *, colrow_3d: str = "colrow"):
    """`colrow_normalize` in place — the SAME arithmetic, no new tensor.

    `x.div_(n + eps)` and `x / (n + eps)` are the same operation on the same
    operands, so this is bit-identical to the out-of-place form; it exists
    because the batched step's `x` is a fresh tensor the optimizer owns (at b1
    the two temporaries are 2 x 1.33 GiB of the expert-write group per step).
    Takes the same `colrow_3d` arm, with the same semantics.
    """
    if x.ndim == 2:
        x.div_(x.norm(dim=0, keepdim=True) + eps)
        return x.div_(x.norm(dim=1, keepdim=True) + eps)
    _check_colrow_3d(colrow_3d)
    x.div_(x.norm(dim=(1, 2) if colrow_3d == "frobenius" else 1, keepdim=True) + eps)
    return x.div_(x.norm(dim=(1, 2) if colrow_3d == "frobenius" else 2, keepdim=True) + eps)


def normuon_second_moment_(o: torch.Tensor, v: torch.Tensor, beta2: float = NOR_BETA2) -> None:
    """`v_t ← β2·v_{t-1} + (1−β2)·mean_cols(O ⊙ O)` — the step both forms share.

    `o` must be the POST-polar direction and must be read BEFORE it is
    normalized in place. Transcribed from `zichongli5/NorMuon`
    `normuon.py:49-50` (`v_mean = torch.mean(update * update, dim=-1,
    keepdim=True)`, `second_momentum.lerp_(v_mean, 1 - beta2)`) — the `lerp_`
    form, not a `mul_`/`add_` rewrite — and from `2605.23061v1` Algorithm 1 line
    8. The repo keeps `v` as `[..., m, 1]` (`zeros_like(p[..., 0:1])`, `:89`);
    this port keeps `[m]` / `[N, m]` (its state layout, which already
    round-trips) — the same values, one squeeze apart.
    """
    v_mean = (o * o).mean(dim=-1)
    v.lerp_(v_mean, 1.0 - beta2)


def normuon_normalize_(
    o: torch.Tensor, v: torch.Tensor, beta2: float = NOR_BETA2, eps: float = NOR_SF_EPS
) -> torch.Tensor:
    """`P̂_t = O_t ⊘ (√V_t + ε)` — the **SF paper's** Alg-1 form, used by `sf`.

    Source: `2605.23061v1` Appendix F's reference code
    (`row_ms = (P*P).mean(dim=1); v.mul_(beta2).add_(row_ms, alpha=1-beta2);
    Phat = P / (v.sqrt() + eps).unsqueeze(1)`) and Algorithm 1 lines 8-10 —
    Microsoft's NorMuon row normalization as the SF paper incorporates it, with
    the SF paper's own ε = 1e-8. **This is NOT the `muon_variant="normuon"`
    arm's tail** (`normuon_repo_update_` is): the SF arm needs a `P̂` whose
    Frobenius norm its `eta_hat = 0.2·η·√(mn)/‖P̂‖_F` can normalize, while the
    `normuon` arm needs the authors' repo's magnitude-preserving rescale.

    `o` is normalized IN PLACE — the contract `colrow_normalize_` already has
    (`o` is a fresh tensor the optimizer owns). `v` is one `m`-vector per matrix
    (`[m]` 2-D, `[N, m]` batched), fp32 as in the SF reference
    (`torch.zeros(p.shape[0], dtype=torch.float32)`, Appendix F:68): `m` scalars
    per matrix is negligible against the `m·n` momentum, so `v` does not ride
    the `state_dtype` rung. (The Microsoft repo instead makes it the PARAM's
    dtype, `zeros_like(p[..., 0:1])` at `normuon.py:89` — a storage difference,
    not an arithmetic one, and it is recorded here rather than adopted, because
    a bf16 EMA of a squared row mean is a precision loss with no memory win.)
    """
    normuon_second_moment_(o, v, beta2)
    return o.div_((v.sqrt() + eps).unsqueeze(-1))


def normuon_repo_update_(
    o: torch.Tensor,
    v: torch.Tensor,
    shape: tuple[int, int],
    beta2: float = NOR_BETA2,
    eps: float = NOR_REPO_EPS,
) -> torch.Tensor:
    """The `muon_variant="normuon"` tail: **the authors' repo, transcribed**.

    `zichongli5/NorMuon` `normuon.py::normuon_update`, lines 48-56 verbatim:

        :48  vnorm = update.norm(dim=(-2,-1), keepdim=True)
        :49  v_mean = torch.mean(update * update, dim=-1, keepdim=True)
        :50  second_momentum.lerp_(v_mean, 1 - beta2)
        :51  step_size = 1 / second_momentum.sqrt().add_(1e-10)
        :52  update.mul_(step_size)
        :53  vnorm_new = update.norm(dim=(-2,-1), keepdim=True)
        :54  update.mul_(vnorm / (vnorm_new.add_(1e-10)))
        :56  update *= max(1, grad.size(-2) / grad.size(-1))**0.5

    so `‖update‖_F` is restored to its pre-normalization value (`:54`'s comment
    in the repo says exactly that) and only then is the control's own
    `max(1, m/n)^0.5` applied. The repo applies `-lr * update` afterwards
    (`:92-93`: decay, then `p.add_(update.reshape(p.shape), alpha=-group["lr"])`)
    with NO further scale, which is why this arm's step magnitude equals the
    `muon` arm's: **Muon's magnitude, NorMuon's direction** — the design intent
    the A/B relies on (gated in `tests/test_muon_variants.py`).

    `o` is the BARE polar (`zeropower_via_ns(mom)`), exactly as the repo feeds
    its own tail (`normuon.py:43-47`): the arm REPLACES our ColRow tail rather
    than composing with it. **Why it is not composed, measured on the real polar**
    (`tests/test_muon_variants.py` pins the numbers): NorMuon's row scaling
    `1/√(EMA of mean_cols(O⊙O))` IS ColRow's row half — a per-row scaling by the
    row norm, up to the constant `√n` — and `colrow_normalize` makes every row
    unit-norm, so feeding the tail a ColRow'd matrix makes its statistic CONSTANT
    across rows (`mean_cols(O⊙O) = 1/n` for every row; measured spread `1.8e-7`,
    i.e. the eps). The composed arm then reproduces the control's update with the
    same norm and `cos = 1.000000`, i.e. it is INERT. Composing was the first
    plan; the measurement killed it.

    **The magnitude, stated because the A/B rests on it:** the invariant this form
    is designed for — `‖ΔW‖_F` identical to Muon's at equal η (`:54`'s own comment
    says exactly that) — holds against the REPO's Muon
    (`-η·max(1, m/n)^0.5·polar`, no ColRow) and is gated that way. Against OUR
    ColRow control it does NOT hold: ColRow's row normalization gives
    `‖ColRow(O)‖_F = √m` while the polar gives `‖O‖_F ≈ 0.7·√min(m,n)` (the Jordan
    quintic does not reach the exact polar — its singular values are spread), so
    the measured ratios `‖ΔW_arm‖/‖ΔW_control‖` are **0.7066 at `[64, 5461]`,
    0.8701 at `[64, 64]`, 0.1293 at `[2048, 64]`** (the tall factor carries
    `√(n/m)` on top). The arm is therefore a REPLACE of our ColRow tail — the row
    half is NorMuon's EMA instead of instantaneous — not a direction-only change
    against that control, and the tall-factor gap is reported, not papered over.

    **The rejected alternative, named here so nobody re-derives it:** the SF
    paper's Alg-1 form (`normuon_normalize_` + `nor_eta_hat`,
    `eta_hat = 0.2·η·√(mn)/‖P̂‖_F`) FIXES the step's Frobenius norm at
    `0.2·η·√(mn)` whatever the matrix is, so it moves the magnitude as well as
    the direction, and it also disagrees with itself on the factor (`√(mn)` in
    Alg 1 line 13 and the code, `√max(m,n)` in the prose and Lemma 3.1 — 41%
    apart at `[2, 3]`). The repo form is what produced the published +11.31% at
    1.1B, so the repo form is the arm.
    """
    m, n = shape
    vnorm = o.norm(dim=(-1, -2), keepdim=True)
    normuon_second_moment_(o, v, beta2)
    o.mul_(1.0 / (v.sqrt() + eps).unsqueeze(-1))
    vnorm_new = o.norm(dim=(-1, -2), keepdim=True)
    o.mul_(vnorm / (vnorm_new + eps))
    o.mul_(max(1.0, m / n) ** 0.5)
    return o


def nor_eta_hat(lr: float, shape: tuple[int, int], p_hat: torch.Tensor) -> torch.Tensor:
    """`η̂ = 0.2·η·√(mn)/‖P̂‖_F` — the **`sf` arm's** RMS-matching step scale.

    Source: arXiv:2605.23061v1 **Appendix F** line 77, verbatim —
    `eta_hat = eta_scale * lr * math.sqrt(m * n) / max(1e-12, Phat.float().norm())`
    — and Algorithm 1 line 13 (`η̂_t ← 0.2 η_t √(mn)/‖P̂_t‖_F`). The `sf` arm's
    step magnitude is set HERE on both variants (`_sf_direction`), which is what
    makes `sf` and `normuon` composable; the non-SF `normuon` arm does not use
    this function at all (`normuon_repo_update_`'s rescale is its magnitude).

    **The paper disagrees with itself and the CODE wins here:** §2.2's "learning
    rate scaling" paragraph and Lemma 3.1 both write `√max(m,n)`, while
    Algorithm 1 line 13 and Appendix F's code write `√(mn)` (41% apart at a
    `[2, 3]` factor). `√(mn)` is what this port runs, so a run is the algorithm
    the authors implemented, and the disagreement is recorded in this docstring
    instead of being silently picked.

    Per SLICE for the batched step: each `[m, n]` slice is its own matrix, so
    the norm is taken over the last two dims with `keepdim` — the reference's
    `norm()` over a whole stacked buffer would tie every slice's step to the
    batch's largest. `1e-12` floor as in the reference.
    """
    m, n = shape
    norm = p_hat.norm() if p_hat.ndim == 2 else p_hat.norm(dim=(-1, -2), keepdim=True)
    return NOR_ETA_SCALE * lr * math.sqrt(m * n) / norm.clamp_min(1e-12)


class _MomGroup:
    """One shape group's canonical STACKED momentum buffer.

    `mom` is `[N, d0, d1]` and it IS the state: `state[p]["momentum"]` is
    a view of row `i`, so the step never stacks the group in and copies it back
    (at b1 that was 3 x 1.33 GiB of fp32 traffic per step for the `[64, 5461]`
    expert writes alone, plus the allocator churn of a fresh 1.33 GiB block
    every step). The checkpoint format is unchanged — `Optimizer.state_dict`
    still emits one `momentum` tensor per parameter, and `load_state_dict`
    rebuilds the stack from them once. `mom.dtype` is the group's
    `state_dtype` rung (`fp32` by default).

    The two opt-in arms add stacked buffers the same way, each present only when
    its arm is on (`None` otherwise), so a `muon` run's memory and dispatches do
    not move:

    * `v` — NorMuon's row second moment, `[N, m]` fp32 (`variant="normuon"`);
    * `x`, `z` — the schedule-free iterates, `[N, d0, d1]` in the state dtype
      (`sf=True`). `x` is the returned/eval iterate and `z` the fast one; the
      PARAMS are `Y = (1-β1)Z + β1 X` (`sf_switch`).
    """

    __slots__ = ("gi", "shape", "params", "mom", "v", "x", "z")

    def __init__(
        self,
        gi: int,
        shape: tuple[int, int],
        params: list,
        mom: torch.Tensor,
        v: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> None:
        self.gi = gi
        self.shape = shape
        self.params = params
        self.mom = mom
        self.v = v
        self.x = x
        self.z = z


def _momentum_update_(mom: torch.Tensor, params: list, mu: float) -> None:
    """`m ← mu·m + (1-mu)·g`, in place over a stacked buffer, one fused pass.

    `fp32` (the default): `mom.mul_(mu)` + ONE `_foreach_add_` over the rows —
    the path this port has always run, untouched.

    `bf16`: the ACCUMULATION is fp32 and only the STORE rounds to bf16
    (arXiv:2509.23106's rung, cited in the module docstring). One fp32 temporary
    per CHUNK carries the arithmetic (`MUON_CHUNK_ELEMS` caps it at 134 MiB) and
    the store is one cast-copy per chunk: an unchunked fp32 temporary of the b1
    `[1024, 64, 5461]` group alone is 1.33 GiB, which is the size that OOM'd the
    warm attempts (audit MAJOR 7, `~/logs/b1warm.log`), so the chunking is what
    makes the bf16 rung a memory win instead of a memory-neutral one.
    """
    if mom.dtype != torch.bfloat16:
        mom.mul_(mu)
        torch._foreach_add_(torch.unbind(mom, 0), [p.grad for p in params], alpha=1.0 - mu)
        return
    n = len(params)
    n_chunk = max(1, min(n, MUON_CHUNK_ELEMS // mom[0].numel()))
    for start in range(0, n, n_chunk):
        chunk = params[start : start + n_chunk]
        acc = mom[start : start + n_chunk].to(torch.float32)
        acc.mul_(mu)
        torch._foreach_add_(torch.unbind(acc, 0), [p.grad for p in chunk], alpha=1.0 - mu)
        mom[start : start + n_chunk].copy_(acc)


class Muon(Optimizer):
    """Momentum + polar orthogonalization on 2-D parameters.

    Both precision knobs (`OPT_DTYPES`) default to `fp32`, which is the
    behaviour every number on record ran; `state_dtype="bf16"` halves the
    momentum state, `ns_dtype="bf16"` runs the polynomial in bf16.

    `variant` (`MUON_VARIANTS`) picks the direction arm and `sf` wraps it in the
    schedule-free averaging — both `off`/`muon` by default. `wd` is the Muon
    group's OWN weight decay (the papers' `-η·λ·W`), which this port carries at
    `0.0` on every run: the trainer's `wd` reaches the AdamW half only, so the
    default path is byte-identical to its history (the SF research doc's S2 gap).
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        ns_kind: str = "classic",
        gram_restarts: tuple[int, ...] = GRAM_NS_RESTARTS,
        colrow_3d: str = "colrow",
        state_dtype: str = "fp32",
        ns_dtype: str = "fp32",
        variant: str = "muon",
        beta2: float = NOR_BETA2,
        wd: float = 0.0,
        sf: bool = False,
        sf_beta: float = SF_BETA1,
    ) -> None:
        _check_ns_kind(ns_kind)
        _check_colrow_3d(colrow_3d)
        _check_dtype("state_dtype", state_dtype)
        _check_dtype("ns_dtype", ns_dtype)
        _check_muon_variant(variant)
        #: The arm and its restart positions live on the OPTIMIZER, not in the
        #: param group: the group is what `Optimizer.load_state_dict` REPLACES
        #: with the checkpoint's copy, so a key added today would KeyError on
        #: every checkpoint written before it. The arm is already pinned across
        #: a resume by the config snapshot (`train.ns_kind` is not a progress
        #: key, so `RunConfig::diff_keys` refuses a flip) — one place, not two.
        #: `colrow_3d` follows the same rule (`train.muon_colrow_3d`), and so do
        #: the two dtype rungs (`train.muon_state_dtype` / `train.ns_dtype`),
        #: `variant` (`train.muon_variant`) and `sf`/`sf_beta` (`train.muon_sf`).
        self.ns_kind = ns_kind
        self.gram_restarts = gram_restarts
        self.colrow_3d = colrow_3d
        self.state_dtype = state_dtype
        self.ns_dtype = ns_dtype
        self.variant = variant
        self.beta2 = beta2
        self.wd = wd
        self.sf = sf
        self.sf_beta = sf_beta
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))
        #: One stacked momentum buffer per (param group, shape) — the canonical
        #: state, built lazily by `_build_groups` and rebuilt after a load.
        self._groups: list[_MomGroup] = []
        #: The arm counters (ADR-0019). Only the `triton` arm writes here —
        #: `ns_triton` when the fused chain ran, `ns_triton_fallback` + the
        #: reason when it could not — and `train/loop.py` merges + clears it on
        #: every step that runs the optimizer, so the training log's counter line
        #: is where a reader learns which chain ran. Empty for every other arm,
        #: so the default path's log line does not move.
        self.counters = Counters()

    def _build_groups(self) -> None:
        """One stacked momentum buffer per unique shape; per-param state views it.

        Idempotent and cheap, but it STACKS whatever is in
        `state[p]["momentum"]`, so it is called (a) lazily by the batched step
        and (b) after `load_state_dict` — where the loader has replaced the
        views with the checkpoint's own per-param tensors, which is exactly the
        historical format and stays readable.

        The two arms' buffers (`v`, `x`, `z`) are stacked and viewed the same
        way, and are `None` when their arm is off — so the default path's state
        and memory are exactly what they were. `x`/`z` default to the PARAM's
        value when the checkpoint carries none, which is Algorithm 1's own
        initialization (`X_1 = Z_1`, and the params hold `Y_1 = X_1` at build
        time: `sf_switch` cannot have run, there is no state yet).
        """
        self._groups = []
        for gi, group in enumerate(self.param_groups):
            by_shape: dict[tuple[int, int], list] = {}
            for p in group["params"]:
                if p.ndim == 2:
                    by_shape.setdefault((p.shape[0], p.shape[1]), []).append(p)
            for shape, params in by_shape.items():
                n = len(params)
                dtype = _torch_dtype(self.state_dtype)
                mom = torch.zeros(n, *shape, dtype=dtype, device=params[0].device)
                v = (
                    torch.zeros(n, shape[0], dtype=torch.float32, device=params[0].device)
                    if self.variant == "normuon"
                    else None
                )
                x = z = None
                if self.sf:
                    x = torch.zeros(n, *shape, dtype=dtype, device=params[0].device)
                    z = torch.zeros(n, *shape, dtype=dtype, device=params[0].device)
                for i, p in enumerate(params):
                    prev = self.state[p].get("momentum")
                    if prev is not None:
                        mom[i].copy_(prev)
                    self.state[p]["momentum"] = mom[i]
                    if v is not None:
                        pv = self.state[p].get("v")
                        if pv is not None:
                            v[i].copy_(pv)
                        self.state[p]["v"] = v[i]
                    for key, buf in (("x", x), ("z", z)):
                        if buf is None:
                            continue
                        pv = self.state[p].get(key)
                        buf[i].copy_(pv if pv is not None else p.detach())
                        self.state[p][key] = buf[i]
                self._groups.append(_MomGroup(gi, shape, params, mom, v, x, z))

    def load_state_dict(self, state_dict) -> None:  # noqa: D102 - torch's contract
        super().load_state_dict(state_dict)
        # The loader installed the checkpoint's per-param `momentum` tensors;
        # stacking them once here is what makes an old checkpoint (per-param)
        # and a new one (views of one storage) both load correctly.
        self._build_groups()

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102 - torch's Optimizer contract
        """The per-param step: `_step_one` per parameter, nothing else.

        `MixOptimizer` runs the batched path; this is the Optimizer contract
        and the definition, in one place so the two dtype rungs cannot drift
        between a batched and a per-param implementation.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, mu, ns = group["lr"], group["momentum"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    loud(f"Muon got a rank-{g.ndim} gradient: the router must not send it here")
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                self._step_one(p, lr, mu, ns, scale)
        return loss

    def _orthogonalize(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """The NS call, with the rung forwarded only when it is NOT the default.

        The `fp32` rung therefore passes the four arguments every gate, spy and
        pin in this tree was written against (`tests/test_muon_gram.py`'s
        `test_the_requested_arm_is_the_arm_that_runs` transcribes exactly that
        signature), and the default path's call shape does not move. The `triton`
        arm is the same rule one step further: its counters are forwarded (the
        arm must be able to show which chain ran) and nothing else about the call
        changes for `classic`/`gram`.
        """
        if self.ns_kind == "triton":
            return zeropower_via_ns(
                x, steps, self.ns_kind, self.gram_restarts, self.ns_dtype, counters=self.counters
            )
        if self.ns_dtype == "fp32":
            return zeropower_via_ns(x, steps, self.ns_kind, self.gram_restarts)
        return zeropower_via_ns(x, steps, self.ns_kind, self.gram_restarts, self.ns_dtype)

    def _step_one(self, p: torch.Tensor, lr: float, mu: float, ns: int, scale: float) -> None:
        """The per-param update — the reference the batched path must match.

        Kept because it is the definition, and used by the batched step for the
        rare group where a 2-D parameter carries no gradient (a shape with one
        live member is not worth a batch). It reads and writes
        `state[p]["momentum"]`, which on a built group is a VIEW of the stacked
        buffer, so both paths share one state. On the `bf16` rung the view is
        bf16 and the update goes through an fp32 temporary (`_momentum_update_`'s
        arithmetic, one parameter at a time).

        The two arms branch here too, with the SAME helpers the batched path
        calls (`_nor_apply` / `_sf_one`), so the per-param and batched arms
        cannot drift — the file's own rule for the dtype rungs.
        """
        state = self.state[p]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p.grad, dtype=_torch_dtype(self.state_dtype))
        buf = state["momentum"]
        if buf.dtype == torch.bfloat16:
            buf.copy_(buf.float().mul_(mu).add_(p.grad, alpha=1.0 - mu))
        else:
            buf.mul_(mu).add_(p.grad, alpha=1.0 - mu)
        update = self._orthogonalize(buf, ns)
        if self.sf:
            self._sf_one(p, update, lr)
            return
        if self.variant == "normuon":
            self._nor_apply(p, update, lr)
            return
        update = colrow_normalize(update)
        with torch.no_grad():
            if torch.is_tensor(lr):
                # The CUDA-graph arm (`train/graph.py`): `lr` is a 0-dim DEVICE
                # tensor the captured ops re-read on every replay, and `add_`'s
                # `alpha` cannot take one without a host read (the dispatcher
                # converts a 0-dim tensor to a Scalar, which is a D2H sync — and a
                # sync inside a capture window aborts it). Fold it into the update
                # on device and keep `alpha` a host float; `-scale` is the same
                # factor, one elementwise op earlier. The batched path
                # (`batched_step_from_model`) needs nothing: `-lr * scale` is a
                # device scalar op when `lr` is a tensor.
                p.add_(update.to(p.dtype).mul_(lr), alpha=-scale)
            else:
                p.add_(update.to(p.dtype), alpha=-lr * scale)

    def _nor_apply(self, p: torch.Tensor, o: torch.Tensor, lr: float) -> None:
        """The `normuon` tail for ONE parameter: `W -= η·λ·W + η·update`.

        `update` is the repo's tail (`normuon_repo_update_`) applied to the BARE
        polar, and the apply is the repo's own (`normuon.py:92-93`: decay first,
        then `p.add_(update, alpha=-lr)`) — no extra scale here, because
        `max(1, m/n)^0.5` is already inside `update`. `λ` is this optimizer's own
        `wd` (`0.0` on every run: the trainer's `wd` reaches the AdamW half
        only), so the decay term is skipped entirely there.
        """
        state = self.state[p]
        v = state.setdefault("v", torch.zeros(p.shape[0], dtype=torch.float32, device=p.device))
        update = normuon_repo_update_(o, v, p.shape, self.beta2)
        with torch.no_grad():
            if self.wd:
                p.mul_(1.0 - lr * self.wd)
            p.add_(update.to(p.dtype), alpha=-lr)

    def _sf_one(self, p: torch.Tensor, o: torch.Tensor, lr: float) -> None:
        """The schedule-free tail for ONE parameter (the batched twin is
        `_sf_batched`; both use `_sf_direction` and `_sf_ckp`)."""
        state = self.state[p]
        if "z" not in state:
            dtype = _torch_dtype(self.state_dtype)
            state["z"] = p.detach().to(dtype).clone()
            state["x"] = p.detach().to(dtype).clone()
        z, x = state["z"], state["x"]
        v = state.get("v")
        if self.variant == "normuon" and v is None:
            v = state["v"] = torch.zeros(p.shape[0], dtype=torch.float32, device=p.device)
        p_hat, eta = self._sf_direction(o, lr, p.shape, v)
        s_new, c = self._sf_ckp(lr, float(state.get("sf_s", 0.0)))
        state["sf_s"] = s_new
        with torch.no_grad():
            if self.wd:
                z.mul_(1.0 - lr * self.wd)
            z.add_(p_hat, alpha=-eta)
            x.mul_(1.0 - c).add_(z, alpha=c)
            p.copy_(z).lerp_(x.to(p.dtype), self.sf_beta)

    def _sf_direction(
        self, o: torch.Tensor, lr: float, shape: tuple[int, int], v: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(P̂, η̂)` for one slice: the arm's normalization, then `nor_eta_hat`.

        Algorithm 1 line 13 applies the `0.2·η·√(mn)/‖P̂‖_F` scale whichever
        normalization produced `P̂`, so the SF arm's step magnitude does not
        depend on `variant` — that is what makes `sf` and `normuon` composable
        instead of two different step sizes wearing one name. `v` is the
        NorMuon row statistic (the group's slice, or the parameter's own) and is
        `None` on the `muon` variant.
        """
        if self.variant == "normuon":
            assert v is not None, "variant=normuon needs its row second moment"
            p_hat = normuon_normalize_(o, v, self.beta2)
        else:
            p_hat = colrow_normalize(o, colrow_3d=self.colrow_3d)
        return p_hat, nor_eta_hat(lr, shape, p_hat)

    @staticmethod
    def _sf_ckp(lr: float, s_prev: float) -> tuple[float, float]:
        """`s_t = s_{t-1} + η_t²`, `c_{t+1} = η_t²/s_t` (Alg 1 lines 14-15).

        A HOST float on purpose: `lr` is a host float the trainer sets, so the
        whole recursion is host arithmetic — no device read, no sync (the port's
        rule 1.3). Stored per parameter as `sf_s`, so torch's own `state_dict`
        carries it (a float, not a tensor: nothing to sync and nothing to cast).
        """
        s = s_prev + lr * lr
        return s, lr * lr / s

    @torch.no_grad()
    def sf_switch(self, mode: str) -> None:
        """Move the params between the `Y` (train) and `X` (eval) iterates.

        `Y_t = (1-β1)·Z_t + β1·X_t` is where the gradient is evaluated
        (arXiv:2605.23061v1 Alg 1 line 3; Appendix F's `train()`), `X_t` is the
        returned/eval iterate (line 18; `eval()`). BOTH writes are ABSOLUTE
        functions of the optimizer's own `x`/`z` state, so they are idempotent —
        the reference needs a `train_mode` flag to make its lerps safe
        (Appendix F:34/43), this port does not.

        A parameter with no `z` yet has never stepped, so it IS `X = Z`
        (Alg 1's `X_1 = Z_1`) and both switches are no-ops for it — which is why
        the trainer's post-construction `sf_switch("train")` is harmless.
        """
        if not self.sf:
            loud(
                f"sf_switch({mode!r}) on an optimizer with sf=false: the "
                f"schedule-free iterate state does not exist, so there is nothing "
                f"to switch. Escape: construct with sf=true (train.muon_sf=true)."
            )
        if mode not in ("train", "eval"):
            loud(
                f"sf_switch({mode!r}): known modes are 'train' (params <- Y = "
                f"(1-β1)Z + β1 X, where the gradient is evaluated) and 'eval' "
                f"(params <- X, the returned iterate). Escape: name one of those."
            )
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                if "z" not in st:
                    continue
                if mode == "eval":
                    p.copy_(st["x"])
                else:
                    p.copy_(st["z"]).lerp_(st["x"].to(p.dtype), self.sf_beta)

    def batched_step_from_model(self, model) -> int:
        """The shape-batched Muon pass (a no-op when the batches are trivially small).

        The per-param loop above re-derives identical-shape Newton-Schulz
        computations for EACH expert separately — 46,341 `aten::mm` launches
        for ONE step of b1 (the 2026-10-15 profiler probe at b8·s512: 6.9 s
        of a 7.2 s step, 84% of the CPU on per-tensor dispatches). The fix,
        one regex that mirrors the padded-MoE banked-bmm lesson: per UNIQUE
        parameter shape, stack the momentum buffers (and the grads) via view
        [N, d0, d1]; the SAME NS math inside runs unchanged; then scatter
        back. The scaled `max(1, d0/d1)^0.5` case is the SAME for the whole
        shape, so it stays a scalar.

        Since 2026-10-16 the stack is not per-step work at all: the stacked
        buffer is the STATE (`_build_groups`), the momentum update is
        `mom.mul_(mu)` + ONE fused `_foreach_add_` over its own rows (the
        gradient stack is gone with it), and the writes are in place. What
        remains per step is the NS chain and the final `_foreach_add_`. On the
        `bf16` state rung the same update runs with an fp32 accumulate and a
        bf16 store, one chunk temporary at a time (`_momentum_update_`).

        The group's `lr` may be a 0-dim DEVICE tensor instead of a host float
        (`train/graph.py`): `-lr * scale` is then a device scalar op the captured
        ops re-read on every replay, which is what lets a CUDA-graph replay track
        the WSD schedule. Nothing in this method branches on it — the singleton /
        missing-grad path it delegates to (`_step_one`) is where the difference
        has to be handled.
        """
        if not self._groups:
            self._build_groups()
        n_batches = 0
        for grp in self._groups:
            group = self.param_groups[grp.gi]
            lr, mu, ns = group["lr"], group["momentum"], group["ns_steps"]
            scale = max(1.0, grp.shape[0] / grp.shape[1]) ** 0.5
            params = grp.params
            if len(params) < 2 or any(p.grad is None for p in params):
                # A shape with one member, or a 2-D parameter with no gradient:
                # fall back to the definition rather than inventing a "zero
                # gradient" row. This is not just caution — the 3-D branch of
                # `colrow_normalize` is a KNOWN divergence from the per-param
                # ColRow (see its docstring), so routing a singleton through the
                # batch would MOVE that parameter's update scale by 1/sqrt(d0).
                for p in params:
                    if p.grad is not None:
                        self._step_one(p, lr, mu, ns, scale)
                continue
            mom = grp.mom
            # The momentum update, in place, without a stack: `mom` is the
            # buffer and `torch.unbind` hands out its rows as views
            # (`_momentum_update_` carries the two dtype rungs' arithmetic).
            _momentum_update_(mom, params, mu)
            n = len(params)
            if self.sf:
                # `s`/`c` are ONE scalar per group (Alg 1 lines 14-15): read the
                # group's accumulator once and write it back once, rather than
                # per chunk, so a chunk boundary cannot move the average.
                s_new, c = self._sf_ckp(lr, float(self.state[params[0]].get("sf_s", 0.0)))
                for p in params:
                    self.state[p]["sf_s"] = s_new
            # Chunk the shape stack: the cap is fp32 ELEMENTS per stacked
            # buffer (`MUON_CHUNK_ELEMS`), so NS never materializes the whole
            # shape group's chain at once.
            n_chunk = max(1, min(n, MUON_CHUNK_ELEMS // (grp.shape[0] * grp.shape[1])))
            for start in range(0, n, n_chunk):
                params_c = params[start : start + n_chunk]
                update = self._orthogonalize(mom[start : start + n_chunk], ns)
                if self.sf:
                    self._sf_batched(grp, start, len(params_c), params_c, update, lr, c)
                    continue
                if self.variant == "normuon":
                    self._nor_batched(grp, start, len(params_c), params_c, update, lr)
                    continue
                # In place: `update` is a fresh tensor the optimizer owns (NS
                # always allocates on its first `x / (norm + eps)`), and the
                # write-back is FOREACH — per-param copies are what kept 5221
                # add_ + 3267 copy_ launches on the post-batched profile
                # (55 + 57 ms/step). The size check guarantees identical
                # shapes so the sockets align 1:1.
                colrow_normalize_(update, colrow_3d=self.colrow_3d).mul_(-lr * scale)
                with torch.no_grad():
                    torch._foreach_add_([p.detach() for p in params_c], torch.unbind(update))
            n_batches += 1
        return n_batches

    def _nor_batched(
        self, grp: _MomGroup, start: int, k: int, params_c: list, o: torch.Tensor, lr: float
    ) -> None:
        """The `normuon` tail for one CHUNK: `W -= η·λ·W + η·update`.

        The batched twin of `_nor_apply` — same helpers, so the two paths cannot
        drift. `λ` is this optimizer's own `wd` (0.0 on every run).
        """
        update = normuon_repo_update_(
            o,
            grp.v[start : start + k],
            grp.shape,
            self.beta2,
        )
        with torch.no_grad():
            if self.wd:
                torch._foreach_mul_([p.detach() for p in params_c], 1.0 - lr * self.wd)
            torch._foreach_add_([p.detach() for p in params_c], torch.unbind(update), alpha=-lr)

    def _sf_batched(
        self,
        grp: _MomGroup,
        start: int,
        k: int,
        params_c: list,
        o: torch.Tensor,
        lr: float,
        c: float,
    ) -> None:
        """The schedule-free tail for one CHUNK (the batched twin of `_sf_one`).

        Alg 1 lines 17-18 and the `Y` write of Appendix F:81-82, with the two
        deliberate deviations the module docstring names: our ColRow stays
        composed (`_sf_direction` applies it on the `muon` variant and NorMuon's
        row statistic on the `normuon` one), and the LR the trainer hands this
        group is constant after warmup.
        """
        z = grp.z[start : start + k]
        x = grp.x[start : start + k]
        v = None if grp.v is None else grp.v[start : start + k]
        p_hat, eta = self._sf_direction(o, lr, grp.shape, v)
        p_hat.mul_(-eta)
        with torch.no_grad():
            if self.wd:
                z.mul_(1.0 - lr * self.wd)
            z.add_(p_hat)  # Z_{t+1} = Z_t - ηλZ_t - η̂P̂
            x.mul_(1.0 - c).add_(z, alpha=c)  # X_{t+1} = (1-c)X_t + c Z_{t+1}
            heads = [p.detach() for p in params_c]
            torch._foreach_copy_(heads, torch.unbind(z))
            torch._foreach_mul_(heads, 1.0 - self.sf_beta)  # params <- Y
            torch._foreach_add_(heads, torch.unbind(x), alpha=self.sf_beta)


def is_muon_param(name: str, p: torch.Tensor) -> bool:
    """The membership rule, in one place, checked by a test.

    burn's policy (AGENTS §3.5 item 3, `core/src/routing.rs`): 2-D linear maps
    go to Muon; **embeddings, the output head, the controller, the MoE router
    and the Engram tables go to AdamW** — orthogonalization hurts or ignores
    them. The literal "everything-2-D" rule is the defect the router agent
    flagged; this is the honest mirror.
    """
    if name.endswith("tables"):
        return False  # the hashed-memory table: Adam, orthogonalization is meaningless
    if name.startswith("fb."):
        # The future-byte head is a readout like `lm_head` (`fb.py`): burn
        # routes it to `Group::Rest`, and a polar update is wrong for a
        # classifier's last layer at the same lr as the body.
        return False
    if any(t in name for t in ("embed", "lm_head", "controller", "router")):
        return False  # orthogonalization hurts them (burn's measured policy)
    return p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1


class MixOptimizer:
    """Muon (2-D) + AdamW (the rest), or AdamW for everything."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float,
        wd: float,
        opt: str = "mix",
        momentum: float = 0.95,
        ns_steps: int = 5,
        ns_kind: str = "classic",
        gram_restarts: tuple[int, ...] = GRAM_NS_RESTARTS,
        colrow_3d: str = "colrow",
        muon_state_dtype: str = "fp32",
        ns_dtype: str = "fp32",
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        muon_variant: str = "muon",
        muon_sf: bool = False,
    ) -> None:
        if opt not in ("mix", "adamw"):
            loud(f"opt={opt!r}: known optimizers are 'mix' (default) and 'adamw'")
        _check_ns_kind(ns_kind)
        _check_colrow_3d(colrow_3d)
        _check_dtype("muon_state_dtype", muon_state_dtype)
        _check_dtype("ns_dtype", ns_dtype)
        _check_muon_variant(muon_variant)
        muon_params: list[nn.Parameter] = []
        adamw_params: list[nn.Parameter] = []
        self.routing: dict[str, str] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            group = "muon" if (opt == "mix" and is_muon_param(name, p)) else "adamw"
            self.routing[name] = group
            (muon_params if group == "muon" else adamw_params).append(p)
        if muon_sf and not muon_params:
            loud(
                f"muon_sf=true with no Muon parameter: `opt={opt!r}` routes every "
                f"2-D parameter to AdamW (or the model has none), so the "
                f"schedule-free wrapper has no group to wrap and the flag would be "
                f"inert while the run reported it on. Escape: opt='mix', or "
                f"muon_sf=false. (Full schedule-free AdamW is the named follow-up, "
                f"not this arm.)"
            )
        self.muon = (
            Muon(
                muon_params,
                lr=lr,
                momentum=momentum,
                ns_steps=ns_steps,
                ns_kind=ns_kind,
                gram_restarts=gram_restarts,
                colrow_3d=colrow_3d,
                state_dtype=muon_state_dtype,
                ns_dtype=ns_dtype,
                variant=muon_variant,
                # The Muon group's OWN weight decay (the papers' `-ηλW`): the
                # `muon` control carries none (today's behaviour, byte-identical),
                # the SF arm carries the trainer's `wd` because arXiv:2605.23061's
                # Lemma 3.1 makes decay at Z necessary for long-horizon stability
                # (the `normuon` arm runs λ=0, the Microsoft repo's own default,
                # so it stays a one-factor arm against the control).
                wd=wd if muon_sf else 0.0,
                sf=muon_sf,
            )
            if muon_params
            else None
        )
        self.adamw = AdamW(adamw_params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
        #: `bnb8 = True` swaps AdamW for bitsandbytes' 8-bit AdamW on the SAME
        #: params — the b1 enabler (the spike's 6.4-7.9 GiB regime rode it;
        #: eager fp32 state at 1.07B params OOMs a 16-GB card). Import is
        #: LOCAL and loud-when-missing, so CPU tests never touch bnb.
        self.bnb8bit = False
        #: Let a caller (the trainer, the smoke test) keep looking at the groups.
        self.n_muon, self.n_adamw = len(muon_params), len(adamw_params)
        #: The Muon arms' counters, re-exported so the trainer has ONE place to
        #: drain (`train/loop.py` merges + clears this after `opt.step()`): the
        #: `triton` NS arm's `ns_triton` / `ns_triton_fallback` + reason reach the
        #: log line through here (ADR-0019). Empty on every other arm.
        self.counters = self.muon.counters if self.muon is not None else Counters()
        #: The SHAPE answer: identical-shape parameters update as ONE stack.
        #: The per-param loop cost 84% of b1's step in per-tensor dispatches.
        self.muon_params = muon_params
        self.model = model
        self.steps = 0
        self.steps_batched = 0

    def enable_bnb8bit(self) -> None:
        """Swap the plain AdamW for bitsandbytes' 8-bit state, LOUD on absence.

        The swap is pairwise per group: same lr/wd/betas/eps, same params,
        only the state dtype drops to int8-quantized — so checkpoint stats
        ride the same optimizer API and a resume reloads it at the same
        shapes. Adequate for the b1 recipe; NOT the default (a 9.2M run
        saves nothing at 16 GB and loses the plain AdamW's exactness).
        """
        if self.bnb8bit:
            return
        try:
            import bitsandbytes as bnb
        except ImportError as e:  # pragma: no cover - the gpu extra pins bnb
            loud(f"bnb8bit: bitsandbytes is not installed ({e}); add the gpu extra")
        new = bnb.optim.AdamW8bit(
            self.adamw.param_groups[0]["params"],
            lr=self.adamw.param_groups[0]["lr"],
            weight_decay=self.adamw.param_groups[0].get("weight_decay", 0.01),
            betas=self.adamw.param_groups[0].get("betas", (0.9, 0.95)),
            eps=self.adamw.param_groups[0].get("eps", 1e-8),
        )
        self.adamw = new
        self.bnb8bit = True

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def sf_switch(self, mode: str) -> None:
        """`Muon.sf_switch` for the Muon group; LOUD when the arm is not on.

        The trainer's seam (`train/loop.py`): the params hold `Y` while a step
        runs and `X` at every eval / checkpoint site. Delegated rather than
        duplicated so the two iterates have one definition.
        """
        if self.muon is None or not self.muon.sf:
            loud(
                f"sf_switch({mode!r}) on an optimizer whose muon_sf is off: there is "
                f"no schedule-free iterate state to switch between. Escape: "
                f"muon_sf=true (train.muon_sf=true), or drop the call."
            )
        self.muon.sf_switch(mode)

    def step(self) -> None:
        """The mixed step: the batched Muon per shape ( Platzhalter-kill), AdamW per group."""
        if self.muon is not None:
            n_batched = self.muon.batched_step_from_model(self.model)
            self.steps_batched = n_batched
        self.adamw.step()
        self.steps += 1

    def state_dict(self) -> dict:
        return {
            "steps": self.steps,
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if self.muon is not None:
            assert state["muon"] is not None, "checkpoint has no Muon state"
            self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        self.steps = int(state["steps"])


def clip_grads(params, max_norm: float) -> torch.Tensor:
    """The fused replacement for `torch.nn.utils.clip_grad_norm_`.

    The stock routine walks every parameter and calls `linalg_vector_norm`
    per tensor — profiled at **252 ms/step** on small (2920 norm calls in a
    40-step run, the single largest eater found by the 2026-10-15 cProfile
    probe). This computes every per-tensor norm in ONE `_foreach_norm`
    launch, reduces once, and applies the coefficient in ONE fused
    `_foreach_mul_` in place. Same math (‖·‖₂ over concatenated gradients),
    no per-tensor kernels, and the returned norm stays a device tensor —
    nothing forces a host read.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    norms = torch._foreach_norm(grads, 2)
    total = torch.linalg.vector_norm(
        torch.stack([n.float() for n in norms])
    )  # l2 over l2s = the norm of the concatenation
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(grads, [coef] * len(grads))
    return total
