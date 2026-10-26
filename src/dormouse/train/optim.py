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
  default) and `triton` — the SAME chain fused into ONE kernel launch
  (`train/ns_kernels.py`), the only arm whose mechanism is a dispatch count.
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
    a HYBRID SF-Muon + WSD-AdamW, not the paper's full-model SF). Its weight
    decay is `muon_sf_wd`, applied at `Z` (Alg 1 line 17) and REQUIRED > 0
    (`lambda=0` leaves Lemma 3.1's bounded-diameter lift unmet); the per-step
    TSCT retraction is `muon_sf_retract` and `true` is refused (S3: the
    averaged iterate is a convex combination of retracted factors, and the
    projection step needs a convex set).
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
NS_KINDS = ("classic", "triton")

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
            f"this port's history), `triton` is the same chain FUSED into one "
            f"kernel launch (`train/ns_kernels.py`, COUNTED fallback to `classic`). "
            f"Escape: name one of those, or land the arm in `optim.NS_KINDS` first."
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
    ns_dtype: str = "fp32",
    *,
    counters=None,
) -> torch.Tensor:
    """The polar factor's polynomial approximation, matrix-only, fp32 by default.

    `kind` selects the arm (`NS_KINDS`): `classic` is the recurrence this port
    has always run; `triton` is the SAME chain fused into ONE kernel launch
    (`train/ns_kernels.py`), the only arm here whose mechanism is a dispatch
    count — it is COUNTED and falls back to `classic` with its reason when it
    cannot serve the call, so `counters` is what a caller passes to see which
    chain ran.

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
        # `torch.bmm` is the SAME kernel `x @ y` selects for 3-D operands, with
        # the same operands: `matmul` only adds view/expand metadata dispatches
        # around it (7 aten ops vs 2 per product, measured). The port's step is
        # CPU-dispatch-bound, so the metadata is the cost; the values are
        # bit-identical (`tests/test_muon_launch.py` pins it).
        gram = torch.bmm(x, x.transpose(1, 2))
        poly = b * gram + c * torch.bmm(gram, gram)
        x = a * x + torch.bmm(poly, x)
    out = x.transpose(1, 2).contiguous() if transposed else x
    return out.to(torch.float32)


def _gram(x: torch.Tensor) -> torch.Tensor:
    """`x @ xᵀ` for one 2-D matrix or one `[N, d0, d1]` batch.

    `torch.mm`/`torch.bmm` are the SAME kernels `x @ ...` selects; `matmul`
    only adds view/expand metadata around them. Values are bit-identical.
    """
    return torch.bmm(x, x.transpose(1, 2)) if x.ndim == 3 else torch.mm(x, x.T)


def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.bmm(x, y) if x.ndim == 3 else torch.mm(x, y)


def _foreach_is_cheaper(numels: list[int]) -> bool:
    """Would `_foreach_*` enqueue fewer kernels than the per-tensor loop?

    `multi_tensor_apply` splits at 320 blocks of 65536 elements, so ONE 33.5M
    tensor costs two launches through `_foreach_mul` and one through `x * b`.
    The batch exists for many SMALL matrices, where the foreach packs them; it
    must not turn a single large chunk into two launches (`medium`'s 513-tensor
    group is one such chunk). Pure arithmetic, gated without running a chain.
    """
    n = len(numels)
    if n == 0:
        return False
    blocks = sum(-(-x // 65536) for x in numels)
    return max(-(-blocks // 320), -(-n // 64)) < n


def _classic_ns_batch(tensors: list[torch.Tensor], steps: int, ns_dtype: str) -> list[torch.Tensor]:
    """One lockstep pass over a list that fits the memory cap (see `_classic_ns_many`)."""
    a, b, c = JORDAN_NS_COEFFS
    pre: list[torch.Tensor] = []
    transposed: list[bool] = []
    for t in tensors:
        x = t.to(torch.float32)
        if x.ndim == 2:
            x = x / (x.norm() + 1e-7)
        else:
            assert x.ndim == 3, f"NS is matrix-only or [N,d0,d1] batched, got {tuple(x.shape)}"
            x = x / (x.norm(dim=(1, 2), keepdim=True) + 1e-7)
        tall = x.shape[-2] > x.shape[-1]
        transposed.append(tall)
        if tall:
            x = x.transpose(-2, -1).contiguous()
        pre.append(x.to(_torch_dtype(ns_dtype)))
    # `_foreach_*` is NOT free: its fast route splits at 320 `multi_tensor_apply`
    # blocks (65536 elements each), so one 33.5M-element tensor costs TWO
    # launches through it and ONE through `x * b`. Use it only where its launch
    # count beats the per-tensor loop — many small matrices, which is the case
    # the batch exists for. The arithmetic is the same op either way.
    fused = _foreach_is_cheaper([t.numel() for t in tensors])

    def scale(xs: list[torch.Tensor], s: float) -> list[torch.Tensor]:
        return torch._foreach_mul(xs, s) if fused else [x * s for x in xs]

    def add(xs: list[torch.Tensor], ys: list[torch.Tensor]) -> list[torch.Tensor]:
        return torch._foreach_add(xs, ys) if fused else [x + y for x, y in zip(xs, ys, strict=True)]

    work = pre
    for _ in range(steps):
        grams = [_gram(x) for x in work]
        # `gram @ gram`, NOT `_gram(gram)`: the Gram is symmetric in exact
        # arithmetic, but a bmm's two triangles are independently rounded, so
        # `gram @ gramᵀ` moves the last bits. The reference has no transpose here.
        squares = [_matmul(g, g) for g in grams]
        # `b * gram` and `c * (gram @ gram)` are separate roundings in the
        # reference; keeping two scales preserves that (an `add(alpha=c)` fuse
        # would round once and move the bf16 chain).
        bg = scale(grams, b)
        cg = scale(squares, c)
        polys = add(bg, cg)
        px = [_matmul(p, x) for p, x in zip(polys, work, strict=True)]
        ax = scale(work, a)
        work = add(ax, px)
    out: list[torch.Tensor] = []
    for x, tall in zip(work, transposed, strict=True):
        if tall:
            x = x.transpose(-2, -1).contiguous()
        out.append(x.to(torch.float32))
    return out


def _classic_ns_many(
    tensors: list[torch.Tensor], steps: int = 5, ns_dtype: str = "fp32"
) -> list[torch.Tensor]:
    """`zeropower_via_ns(kind="classic")` for MANY matrices, one elementwise pass.

    The contraction (`x @ xᵀ`, `gram @ gram`, `poly @ x`) cannot be merged
    across different shapes, so each matrix keeps its own `mm`/`bmm`. The FIVE
    elementwise ops per iteration can be merged: they are the same scalar op on
    every matrix, so `torch._foreach_*` issues one launch for the whole list
    where the per-matrix loop issued one each (the port's step is
    host-dispatch-bound — `AGENTS.md` §2.1).

    The arithmetic is the SAME operation in the SAME dtype, in the same order:
    `b·gram`, then `c·(gram@gram)`, then their add; `a·x`, then `poly@x`, then
    their add. `_foreach_mul`/`_foreach_add` apply exactly those elementwise
    kernels; the per-element values are bit-identical to running
    `zeropower_via_ns` on each tensor in turn, which `tests/test_muon_launch.py`
    pins against the pre-change step.

    The list is split at `MUON_CHUNK_ELEMS` ELEMENTS, the same cap a single
    chunk already obeys, and the biggest tensors are placed first so a chunk
    that fills the cap gets its own batch while the many small groups share one.
    The lockstep pass keeps ~6 fp32 temporaries per element alive for the WHOLE
    batch (the per-chunk loop freed them between chunks), so an unbounded pass
    would turn b1's ~400M NS elements into ~10 GiB of transients. The cap keeps
    the peak equal to main's one-chunk peak and only costs one extra elementwise
    launch per sub-batch.

    The prescale/postscale stay per tensor: the Frobenius norm is per SLICE for
    a 3-D chunk (a `_foreach_norm` would collapse the batch) and the division
    by that norm has a different shape per matrix, which is `_foreach_*`'s slow
    route. Those are 3 launches per matrix, not 25.
    """
    _check_dtype("ns_dtype", ns_dtype)
    out: list[torch.Tensor | None] = [None] * len(tensors)
    # Biggest first: a chunk that fills the cap gets its own batch, and the many
    # small groups share the remaining ones. The greedy split is otherwise
    # order-dependent (a big chunk sandwiched between small ones costs an extra
    # batch). The contraction is per matrix, so the order cannot move a value.
    order = sorted(range(len(tensors)), key=lambda i: -tensors[i].numel())
    batch: list[torch.Tensor] = []
    batch_idx: list[int] = []
    elems = 0
    for i in order:
        t = tensors[i]
        n = t.numel()
        if batch and elems + n > MUON_CHUNK_ELEMS:
            for j, u in zip(batch_idx, _classic_ns_batch(batch, steps, ns_dtype), strict=True):
                out[j] = u
            batch, batch_idx, elems = [], [], 0
        batch.append(t)
        batch_idx.append(i)
        elems += n
    if batch:
        for j, u in zip(batch_idx, _classic_ns_batch(batch, steps, ns_dtype), strict=True):
            out[j] = u
    return out  # type: ignore[return-value]


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


def _stack_grads(params: list) -> torch.Tensor:
    """The chunk's gradients as ONE stacked tensor — the launch-saving copy.

    `_foreach_add_`'s CUDA fast route requires the same dtype across its lists
    (`ForeachUtils.h`'s `_check_tensors_share_device_and_dtype`), so an fp32
    accumulator plus bf16 gradients takes the SLOW route: one launch per tensor
    (513 for the medium `[1024, 64]` group, 256 for `[64, 2730]`). `torch.stack`
    is one copy kernel and the add that follows is one kernel, so the mixed
    dtype costs 2 launches per chunk instead of `len(chunk)`.

    The arithmetic does not move: `stack` copies values exactly, bf16→fp32 is
    exact, and `acc.add_(stacked, alpha)` promotes to fp32 exactly as the
    per-row `add_` the slow route ran (TensorIterator's common dtype). Gated
    bit-for-bit against the pre-change step in `tests/test_muon_launch.py`.
    """
    dtypes = {p.grad.dtype for p in params}
    if len(dtypes) != 1:
        loud(
            f"a Muon shape group's gradients carry mixed dtypes {sorted(map(str, dtypes))}: "
            f"stacking them would promote silently and move the accumulation's dtype. "
            f"Escape: give the group one dtype, or keep those parameters in separate groups."
        )
    return torch.stack([p.grad for p in params])


def _add_grads_(acc: torch.Tensor, params: list, alpha: float) -> None:
    """`acc[i].add_(grad_i, alpha)` — the cheaper of the two launch shapes.

    The per-row `_foreach_add_` takes the CUDA fast route only when the
    accumulator and the gradients share a dtype; otherwise it is one launch per
    tensor. The stacked `add_` is two launches whatever the dtypes are. Pick by
    the launch counts, not by taste: a fast foreach fuses 64 tensors per launch,
    so the stack only wins past two launches (and always wins on a mixed dtype
    with more than two tensors).
    """
    n = len(params)
    dtypes = {p.grad.dtype for p in params}
    if len(dtypes) != 1:
        loud(
            f"a Muon shape group's gradients carry mixed dtypes {sorted(map(str, dtypes))}: "
            f"an elementwise add cannot take them without a silent promotion. "
            f"Escape: one dtype per group."
        )
    fast_launches = -(-n // 64)
    if (dtypes.pop() != acc.dtype and n > 2) or fast_launches > 2:
        acc.add_(_stack_grads(params), alpha=alpha)
        return
    torch._foreach_add_(torch.unbind(acc, 0), [p.grad for p in params], alpha=alpha)


def _foreach_add_mixed_(params: list, update: torch.Tensor, alpha: float = 1.0) -> None:
    """`p.add_(u, alpha)` for every row in a handful of launches, not one each.

    The mixed-dtype add (bf16 parameter + fp32 update) is computed in fp32 and
    stored in the parameter's dtype — TensorIterator's common dtype — and a
    per-tensor `_foreach_add_` over that mix takes `ForeachUtils.h`'s SLOW
    route, i.e. one launch per parameter (781 for the medium Muon groups).
    Stacking the parameters, adding the stacked update ONCE, and copying the
    stack back is the same elementwise operation: the stack/copy are exact and
    the add is the same fp32-compute/bf16-store op. The copy back is a
    same-dtype `_foreach_copy_`, which DOES take the fast route (up to 64
    tensors per launch).

    A same-dtype update skips the stack: `_foreach_add_` is already fast there.
    A dtype this helper cannot reproduce LOUDLY refuses rather than silently
    changing the store's rounding.
    """
    dtype = params[0].dtype
    if any(p.dtype != dtype for p in params):
        loud(
            f"a Muon shape group mixes parameter dtypes "
            f"{sorted({str(p.dtype) for p in params})}: the stacked writeback would "
            f"promote them through one common dtype. Escape: one dtype per group."
        )
    if update.dtype == dtype:
        torch._foreach_add_([p.detach() for p in params], torch.unbind(update), alpha=alpha)
        return
    if update.dtype != torch.float32:
        loud(
            f"the Muon update is {update.dtype} against {dtype} parameters: this "
            f"writeback only knows the fp32-update/param-dtype store (the NS chain's "
            f"own contract) and a same-dtype add. Escape: cast the update at its "
            f"source, or extend the helper with a gated rung."
        )
    if len(params) <= 3:
        # The stack is 3 launches; a handful of per-tensor mixed adds is fewer.
        for p, u in zip(params, torch.unbind(update), strict=True):
            p.detach().add_(u, alpha=alpha)
        return
    stacked = torch.stack([p.detach() for p in params])
    stacked.add_(update, alpha=alpha)
    torch._foreach_copy_([p.detach() for p in params], torch.unbind(stacked))


def _momentum_update_(mom: torch.Tensor, params: list, mu: float) -> None:
    """`m ← mu·m + (1-mu)·g`, in place over a stacked buffer, a few launches.

    `fp32` (the default): `mom.mul_(mu)` + `_add_grads_` per CHUNK. The chunk is
    what keeps the add on the fast route (or on a 2-launch stack) instead of a
    per-tensor loop.

    `bf16`: the ACCUMULATION is fp32 and only the STORE rounds to bf16
    (arXiv:2509.23106's rung, cited in the module docstring). One fp32 temporary
    per CHUNK carries the arithmetic (`MUON_CHUNK_ELEMS` caps it at 134 MiB) and
    the store is one cast-copy per chunk: an unchunked fp32 temporary of the b1
    `[1024, 64, 5461]` group alone is 1.33 GiB, which is the size that OOM'd the
    warm attempts (audit MAJOR 7, `~/logs/b1warm.log`), so the chunking is what
    makes the bf16 rung a memory win instead of a memory-neutral one.
    """
    n = len(params)
    n_chunk = max(1, min(n, MUON_CHUNK_ELEMS // mom[0].numel()))
    if mom.dtype != torch.bfloat16:
        mom.mul_(mu)
        for start in range(0, n, n_chunk):
            _add_grads_(mom[start : start + n_chunk], params[start : start + n_chunk], 1.0 - mu)
        return
    for start in range(0, n, n_chunk):
        chunk = params[start : start + n_chunk]
        acc = mom[start : start + n_chunk].to(torch.float32)
        acc.mul_(mu)
        _add_grads_(acc, chunk, 1.0 - mu)
        mom[start : start + n_chunk].copy_(acc)


class Muon(Optimizer):
    """Momentum + polar orthogonalization on 2-D parameters.

    Both precision knobs (`OPT_DTYPES`) default to `fp32`, which is the
    behaviour every number on record ran; `state_dtype="bf16"` halves the
    momentum state, `ns_dtype="bf16"` runs the polynomial in bf16.

    `variant` (`MUON_VARIANTS`) picks the direction arm and `sf` wraps it in the
    schedule-free averaging — both `off`/`muon` by default. `wd` is the Muon
    group's OWN weight decay (the papers' `-η·λ·W`). The trainer sets it from
    `muon_sf_wd` **only under `sf`**, where it is applied at the fast iterate
    `Z` (Alg 1 line 17) and must be > 0; `MixOptimizer` refuses the arm when it
    is not (Lemma 3.1). With `sf=false` the optimizer is constructed with
    `wd=0.0` on every run, so the control is byte-identical to its history (the
    SF research doc's S2 gap is a property of the arm, not of the control).
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        ns_kind: str = "classic",
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

        The `fp32` rung therefore passes the three arguments every gate, spy and
        pin in this tree was written against, and the default path's call shape
        does not move. The `triton` arm is the same rule one step further: its
        counters are forwarded (the arm must be able to show which chain ran) and
        nothing else about the call changes for `classic`.
        """
        if self.ns_kind == "triton":
            return zeropower_via_ns(x, steps, self.ns_kind, self.ns_dtype, counters=self.counters)
        if self.ns_dtype == "fp32":
            return zeropower_via_ns(x, steps, self.ns_kind)
        return zeropower_via_ns(x, steps, self.ns_kind, self.ns_dtype)

    def _step_one(
        self,
        p: torch.Tensor,
        lr: float,
        mu: float,
        ns: int,
        scale: float,
        flag: torch.Tensor | None = None,
    ) -> None:
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
            self._nor_apply(p, update, lr, flag)
            return
        update = colrow_normalize(update)
        if flag is not None:
            # The device firewall's gate (0/1, on the device): `*1.0` is exact
            # on a healthy step, so this cannot move a health-run number.
            update = update.mul(flag.to(update.dtype))
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

    def _nor_apply(
        self, p: torch.Tensor, o: torch.Tensor, lr: float, flag: torch.Tensor | None = None
    ) -> None:
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
        if flag is not None:
            update = update.mul(flag.to(update.dtype))
        with torch.no_grad():
            if self.wd:
                p.mul_(_gate_identity(flag, 1.0 - lr * self.wd, p))
            p.add_(update.to(p.dtype), alpha=-lr)

    def _sf_one(self, p: torch.Tensor, o: torch.Tensor, lr: float) -> None:
        """The schedule-free tail for ONE parameter (the batched twin is
        `_sf_batched`; both use `_sf_direction` and `_sf_ckp`).

        The SF arm is the ONE host-gated path left, and it is named rather than
        silent: its `s`/`c` recursion is HOST state (`_sf_ckp`), so the firewall
        reads the flag on the host for this arm alone and skips the whole Muon
        step (`MixOptimizer.step`). `opt_graph` already refuses SF, so the
        capture goal — the reason the rest of the step is device-resident — is
        not touched by it.
        """
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
        This is why the SF arm is host-gated in `MixOptimizer.step`: its `s` is
        host state the device flag cannot reach without a read, and `opt_graph`
        refuses SF for the same reason.
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

    def batched_step_from_model(self, model, flag: torch.Tensor | None = None) -> int:
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
        buffer is the STATE (`_build_groups`), and the momentum update is
        `mom.mul_(mu)` + `_add_grads_` over its own rows (the gradient stack is
        gone with it). What remains per step is the NS chain and the final
        write-back. On the `bf16` state rung the same update runs with an fp32
        accumulate and a bf16 store, one chunk temporary at a time
        (`_momentum_update_`).

        The NS calls are QUEUED and run in ONE lockstep pass: the classic
        chain's five elementwise ops per iteration are the same scalar op on
        every queued matrix, so `_classic_ns_many` issues them once for the
        whole list instead of once per shape group. A singleton shape group
        still gets the per-parameter DEFINITION (`_step_one`'s 2-D ColRow and
        cast-then-add), only its NS joins the queue — the 3-D branch's scale is
        a different program.

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
        # The default `muon` arm queues its Newton–Schulz calls and runs the
        # classic chain in ONE lockstep pass (`_classic_ns_many`), so the five
        # elementwise ops per iteration are issued once for every shape group
        # instead of once per group. A job is
        # `(params, tensor_to_orthogonalize, lr, scale, ns_steps, singleton)`.
        jobs: list[tuple[list, torch.Tensor, float, float, int, bool]] = []
        for grp in self._groups:
            group = self.param_groups[grp.gi]
            lr, mu, ns = group["lr"], group["momentum"], group["ns_steps"]
            scale = max(1.0, grp.shape[0] / grp.shape[1]) ** 0.5
            params = grp.params
            if len(params) < 2 or any(p.grad is None for p in params):
                # A 2-D parameter with no gradient falls back to the definition
                # (inventing a "zero gradient" row would move the momentum).
                if any(p.grad is None for p in params):
                    for p in params:
                        if p.grad is not None:
                            self._step_one(p, lr, mu, ns, scale, flag)
                    continue
                if self.sf or self.variant == "normuon":
                    # The SF/NorMuon tails are per-parameter (`_sf_one` /
                    # `_nor_apply`); the singleton path stays the definition.
                    self._step_one(params[0], lr, mu, ns, scale, flag)
                    continue
                # A shape with ONE member, default arm: the momentum rung is
                # `_step_one`'s (same branch, same buffer), and the NS is
                # queued so it shares the lockstep chain. The tail below is
                # `_step_one`'s own 2-D ColRow + add, so this is the same
                # program — NOT the 3-D branch (whose scale is a known
                # divergence, see `colrow_normalize`'s docstring).
                p = params[0]
                buf = self.state[p].get("momentum")
                if buf is None:
                    buf = self.state[p]["momentum"] = torch.zeros_like(
                        p.grad, dtype=_torch_dtype(self.state_dtype)
                    )
                if buf.dtype == torch.bfloat16:
                    buf.copy_(buf.float().mul_(mu).add_(p.grad, alpha=1.0 - mu))
                else:
                    buf.mul_(mu).add_(p.grad, alpha=1.0 - mu)
                jobs.append(([p], buf, lr, scale, ns, True))
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
                pieces = mom[start : start + n_chunk]
                if self.sf:
                    update = self._orthogonalize(pieces, ns)
                    self._sf_batched(grp, start, len(params_c), params_c, update, lr, c)
                    continue
                if self.variant == "normuon":
                    update = self._orthogonalize(pieces, ns)
                    self._nor_batched(grp, start, len(params_c), params_c, update, lr, flag)
                    continue
                jobs.append((params_c, pieces, lr, scale, ns, False))
            n_batches += 1

        if jobs:
            # Group by the step count (every group normally shares `ns_steps`,
            # but a param group may override it): one lockstep chain per count.
            by_steps: dict[int, list[int]] = {}
            for i, job in enumerate(jobs):
                by_steps.setdefault(job[4], []).append(i)
            updates: list[torch.Tensor | None] = [None] * len(jobs)
            for steps, idxs in by_steps.items():
                outs = self._orthogonalize_many([jobs[i][1] for i in idxs], steps)
                for i, update in zip(idxs, outs, strict=True):
                    updates[i] = update
            for job, update in zip(jobs, updates, strict=True):
                params_c, _, lr, scale, _, singleton = job
                assert update is not None
                if singleton:
                    # `_step_one`'s tail, verbatim: out-of-place 2-D ColRow,
                    # the device gate, then the cast-then-add the per-parameter
                    # definition runs (the batched 3-D tail folds the scale into
                    # the update instead; that difference is the definition's).
                    update = colrow_normalize(update, colrow_3d=self.colrow_3d)
                    if flag is not None:
                        update = update.mul(flag.to(update.dtype))
                    with torch.no_grad():
                        p = params_c[0]
                        if torch.is_tensor(lr):
                            p.add_(update.to(p.dtype).mul_(lr), alpha=-scale)
                        else:
                            p.add_(update.to(p.dtype), alpha=-lr * scale)
                    continue
                # In place: `update` is a fresh tensor the optimizer owns (NS
                # always allocates on its first `x / (norm + eps)`). The
                # write-back is the same-dtype fast `_foreach_add_` (or the
                # 3-launch stacked form on the mixed dtype rung).
                colrow_normalize_(update, colrow_3d=self.colrow_3d).mul_(-lr * scale)
                if flag is not None:
                    # The device firewall's gate: `*1.0` on a healthy step is
                    # bit-exact, `*0` on a skipped one makes the write below a
                    # zero add. Nothing here branches on the flag.
                    update.mul_(flag.to(update.dtype))
                with torch.no_grad():
                    _foreach_add_mixed_(params_c, update)
        return n_batches

    def _orthogonalize_many(self, tensors: list[torch.Tensor], steps: int) -> list[torch.Tensor]:
        """`_orthogonalize` for a LIST: fused kernels where they fit, ONE classic pass.

        The `triton` arm is per-call by construction (one fused launch covers a
        whole `[N, d0, d1]` chunk), so eligible chunks still call it one by one.
        Everything the arm refuses — the `ns_dtype` rung, `MAX_RANK`, a non-CUDA
        box — is COUNTED with its reason and then run through the SAME lockstep
        classic chain, so a fallback no longer costs 25 elementwise launches per
        iteration per group.
        """
        if self.ns_kind != "triton":
            return _classic_ns_many(tensors, steps, self.ns_dtype)
        from .ns_kernels import LAUNCH_REFUSALS, path_reason, zeropower_triton

        out: list[torch.Tensor | None] = [None] * len(tensors)
        classic: list[int] = []
        for i, g in enumerate(tensors):
            why = path_reason(g, self.ns_dtype)
            if why is None:
                try:
                    out[i] = zeropower_triton(g, steps)
                except LAUNCH_REFUSALS as e:
                    why = f"the launch was refused by the device ({type(e).__name__}: {e})"
                else:
                    self.counters.count("ns_triton")
                    continue
            classic.append(i)
            self.counters.count("ns_triton_fallback")
            self.counters.why = f"triton: {why}"  # type: ignore[attr-defined]
        if classic:
            outs = _classic_ns_many([tensors[i] for i in classic], steps, self.ns_dtype)
            for i, update in zip(classic, outs, strict=True):
                out[i] = update
        return out  # type: ignore[return-value]

    def _nor_batched(
        self,
        grp: _MomGroup,
        start: int,
        k: int,
        params_c: list,
        o: torch.Tensor,
        lr: float,
        flag: torch.Tensor | None = None,
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
        if flag is not None:
            update = update.mul(flag.to(update.dtype))
        with torch.no_grad():
            if self.wd:
                torch._foreach_mul_(
                    [p.detach() for p in params_c],
                    [_gate_identity(flag, 1.0 - lr * self.wd, p) for p in params_c],
                )
            _foreach_add_mixed_(params_c, update, alpha=-lr)

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
        colrow_3d: str = "colrow",
        muon_state_dtype: str = "fp32",
        ns_dtype: str = "fp32",
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        muon_variant: str = "muon",
        muon_sf: bool = False,
        muon_sf_wd: float | None = None,
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
        # The SF arm's decay at Z is a DIFFERENT quantity from the AdamW half's
        # `wd` (the SF paper's Lemma 3.1, `docs/research/2026-10-16-sf-muon.md`:
        # S2), so it has its own value. `None` keeps the pre-key behaviour
        # (inherit `wd`) for a caller that predates the key; the trainer passes
        # the config field explicitly.
        sf_wd = wd if muon_sf_wd is None else muon_sf_wd
        if muon_sf and sf_wd <= 0:
            loud(
                f"muon_sf=true with muon_sf_wd={sf_wd}: arXiv:2605.23061v1 "
                f"Lemma 3.1 lifts the SF-Spectral bound's bounded-diameter "
                f"assumption only when the fast iterate Z carries a POSITIVE "
                f"decay (Alg 1 line 17: `Z <- (1 - lr*wd)*Z - eta_hat*P_hat`). "
                f"At zero the arm runs outside the published assumption set. "
                f"Escape: muon_sf_wd>0 (the arm's 0.01), or muon_sf=false."
            )
        self.muon = (
            Muon(
                muon_params,
                lr=lr,
                momentum=momentum,
                ns_steps=ns_steps,
                ns_kind=ns_kind,
                colrow_3d=colrow_3d,
                state_dtype=muon_state_dtype,
                ns_dtype=ns_dtype,
                variant=muon_variant,
                # The Muon group's OWN weight decay (the papers' `-ηλW`): the
                # `muon` control carries none (today's behaviour, byte-identical),
                # the SF arm carries `muon_sf_wd` at `Z` because
                # arXiv:2605.23061's Lemma 3.1 makes decay there necessary for
                # long-horizon stability (the `normuon` arm outside SF still runs
                # λ=0, the Microsoft repo's own default, so it stays a one-factor
                # arm against the control).
                wd=sf_wd if muon_sf else 0.0,
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

    def step(self, flag: torch.Tensor | None = None) -> None:
        """The mixed step: the batched Muon per shape ( Platzhalter-kill), AdamW per group.

        `flag` is the trainer's DEVICE firewall: a 0-dim 1/0 tensor that says
        whether this step's loss and gradients were finite. It stays on the
        device — nothing here reads it on the host — and it multiplies every
        WRITE the optimizer makes, so a non-finite step becomes a no-op update
        without a `bool(tensor)` anywhere (ADR-0018 rule 2, `mask_fill` /
        multiply-by-the-flag). `None` (every other caller) is the old path
        unchanged.

        Two arms cannot follow a device flag and are refused at setup rather
        than silently ungated: bitsandbytes' `AdamW8bit` (it has no hook
        between its state and the parameter write) — `train()` refuses
        `optim_8bit` with this firewall.

        AdamW's own decoupled weight decay is lifted out of `torch.optim` here
        for ONE reason: `p.mul_(1 - lr*wd)` moves a parameter even when its
        gradient is zero, so a gated (zeroed) first moment alone is not enough.
        The factor is applied by `_gated_adamw_step` instead, from the same
        `1 - lr*wd` value torch would have used (a 0-dim tensor of the SAME
        float32 value, which ATen's scalar path also casts to), and is selected
        to exactly 1.0 on a skipped step. On a healthy step the arithmetic is
        bit-identical to the stock optimizer — `tests/test_nan_firewall.py`
        pins that against a HEAD golden, and the conversion was checked on CPU.
        """
        if self.muon is not None:
            if self.muon.sf and flag is not None:
                # The SF arm's `s`/`c` are HOST state (`_sf_ckp`), so its skip
                # has to be decided on the host: ONE declared read per step,
                # exactly the old skip branch (the step is not called at all, so
                # `z`, `x` and `s` cannot advance). `opt_graph` refuses SF, so
                # this never sits inside a capture; every other arm is
                # device-resident.
                if float(flag) > 0.0:
                    self.steps_batched = self.muon.batched_step_from_model(self.model)
            else:
                n_batched = self.muon.batched_step_from_model(self.model, flag)
                self.steps_batched = n_batched
        if flag is None:
            self.adamw.step()
        else:
            self._gated_adamw_step(flag)
        self.steps += 1

    @torch.no_grad()
    def _gated_adamw_step(self, flag: torch.Tensor) -> None:
        """AdamW with every parameter WRITE scaled by the device flag.

        Three writes happen in a stock AdamW step, and each is gated:

        * `p.mul_(1 - lr*wd)` — replaced by our own multiply with a factor
          selected to 1.0 on a skipped step (see `step`'s docstring);
        * the moment update `exp_avg.lerp_(grad, 1-beta1)` — its `grad` was
          sanitized to zero by the trainer, so the only thing to stop is the
          carried first moment reaching the `addcdiv_`; it is multiplied by the
          flag before the call, so the update term is exactly zero. The cost is
          a semantic one, named here rather than discovered: a skipped step
          ZEROES the first moment (`exp_avg`), i.e. the step after a NaN restarts
          the momentum from the current gradient. The second moment keeps
          decaying (`grad = 0`), stays finite, and the old skip left both
          untouched — this is the price of not branching on the device value;
        * `p.addcdiv_(exp_avg, denom, value=-step_size)` — zero numerator is a
          zero update, and on the healthy path `exp_avg * 1.0` is bit-exact.

        The weight decay is written back after `adamw.step()` so the optimizer's
        `param_groups` — and therefore every checkpoint's `opt` half — keep the
        config's value; only the step's own view is zero.
        """
        dtype = next(p for g in self.adamw.param_groups for p in g["params"]).dtype
        device = flag.device
        scale = flag.to(dtype)
        keep = flag.to(torch.bool)
        one = torch.ones_like(scale)
        zeroed: list[tuple[dict, float]] = []
        for group in self.adamw.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr = group["lr"]
            wd = group.get("weight_decay", 0.0)
            if wd:
                # The graph arm's LR is a device tensor the capture re-reads; a
                # host lr is the eager path's float. Both produce the SAME
                # float32 scalar torch's `p.mul_(1 - lr*wd)` would have used.
                factor = (
                    1 - lr * wd
                    if torch.is_tensor(lr)
                    else torch.as_tensor(1.0 - lr * wd, dtype=params[0].dtype, device=device)
                )
                factor = torch.where(keep, factor, one)
                # ONE scalar operand, not a list of per-parameter expand views:
                # a list of 815 differently-shaped tensors makes `_foreach_mul_`
                # fall back to per-tensor launches (measured: 4618 kernels per
                # optimizer step vs 2400, ~35 ms of CPU dispatch on a step that
                # only moves 105M parameters).
                torch._foreach_mul_(params, factor)
                zeroed.append((group, wd))
                group["weight_decay"] = 0.0
            moments = [
                self.adamw.state[p]["exp_avg"]
                for p in params
                if "exp_avg" in self.adamw.state.get(p, {})
            ]
            if moments:
                torch._foreach_mul_(moments, scale)
        self.adamw.step()
        for group, wd in zeroed:
            group["weight_decay"] = wd

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


def _gate_identity(flag: torch.Tensor | None, value: float, like: torch.Tensor):
    """`value` on a finite step, the multiply identity `1.0` on a skipped one.

    The device firewall's multiply-by-the-flag pattern (ADR-0018): no
    `bool(tensor)` anywhere, and `* 1.0` is bit-exact, so the healthy path
    cannot move. Returns a host float when `flag is None` (the old path).
    """
    if flag is None:
        return value
    one = torch.ones((), dtype=like.dtype, device=like.device)
    kept = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return torch.where(flag.to(torch.bool), kept, one).expand_as(like)


def _grad_vector_norm(grads: list[torch.Tensor]) -> torch.Tensor:
    """`‖g‖₂` over a list of GRADIENT tensors as ONE device scalar."""
    if not grads:
        return torch.zeros(())
    norms = torch._foreach_norm(grads, 2)
    return torch.linalg.vector_norm(torch.stack([n.float() for n in norms]))


def grad_norm(params) -> torch.Tensor:
    """`‖g‖₂` over every parameter gradient as ONE device scalar.

    The device firewall's gradient test: a vector norm is non-finite iff any
    element is, so one reduction answers "did any gradient go NaN/inf" without
    a host read. `clip_grads` uses the same reduction for its coefficient, so
    both call sites measure the same quantity.
    """
    return _grad_vector_norm([p.grad for p in params if p.grad is not None])


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
    total = _grad_vector_norm(grads)
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(grads, [coef] * len(grads))
    return total
