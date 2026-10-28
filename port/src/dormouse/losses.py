"""The two softmax numerical fixes of arXiv:2610.31642, as autograd Functions.

**Softmax Collapse.** `F.log_softmax` builds the denominator
`1 + sum_{j != t} exp(z_j - z_t)` — it adds `e^{-m}` to 1.0, where `m` is the
paper's margin `z_t - LSE_{j != t} z_j`. Once `m` passes the precision cliff of
the logits dtype (`CLIFF` below: 5.5 nats in bf16, 16.6 in fp32) that sum
rounds to exactly 1.0, so the target probability rounds to 1, the loss to 0 and
its gradient `q_t - 1` to **exactly 0**, while the non-target gradients (small
numbers with their own exponent) stay positive: the gradient no longer sums to
zero, and an adaptive optimizer whose second moment decayed while the gradient
was zero turns the restored gradient into the paper's "slingshot" spike (§4).
The fix keeps the target term OUT of the sum: `m` is the max over `j != t`,
`s` the sum of their exponentials, `loss = log1p(s·exp(m - t))`, all in fp32,
and the target gradient is **defined** as `-(sum of the others)` (Eq. 7) rather
than computed as `q_t - 1`, which is the one identity autodiff cannot keep in
low precision.

**Softmax Cancellation.** Soft-target distillation computes
`q = softmax(z_S)` and `p = softmax(z_T)` separately and subtracts them. The
forward's `p` and the backward's `exp(log_softmax(z_S))` are different
operation sequences, so `z_S == z_T` still leaves `q - p != 0` — the student
drifts away from the teacher on pure noise. The fix (§3) builds `q - p` from
`delta = z_S - z_T` with `expm1`/`log1p`, so identical logits give exactly 0.

**Which dtype the loss sees.** fp32 on every path: the final norm + head run
in fp32 (`model.py`: "fp32 logits — the burn rule that fixes `bf16 logits
NaN`"), the loop's per-iteration head casts to fp32 the same way, and the
byteflow decoder widens before returning. So the LIVE cliff is fp32's 16.6; the
bf16 5.5 cliff is what a bf16 head would hit, and the fixed path upcasts
whatever it is handed to fp32 for its own arithmetic, so only the *returned*
gradient carries the caller's dtype.
"""

from __future__ import annotations

import torch

from .instrument import Counters, loud

#: The precision cliff per logits dtype: `-log(2**b)` with `2**b` the half-ulp
#: of 1.0 in that dtype, i.e. the margin at which `e^{-m}` stops changing
#: `1 + e^{-m}` (paper §2: 5.5 bf16, 16.6 fp32, 36.7 fp64).
CLIFF: dict[torch.dtype, float] = {
    torch.bfloat16: 5.5,
    torch.float16: 7.6,
    torch.float32: 16.6,
    torch.float64: 36.7,
}

#: `exp`'s safe argument for `s * exp(u)` in fp32 with `s` at most a vocab:
#: `exp(80)` is 5.5e34, and 2**16 * 5.5e34 stays finite. Outside the band the
#: loss takes the algebraically equal branch (`u + log(s)` above it, 0 below
#: the fp32 subnormal floor) and the clamp is COUNTED, never silent.
U_MAX = 80.0
U_MIN = -103.0


def cliff(dtype: torch.dtype) -> float:
    """The margin cliff of `dtype`, LOUD for a dtype this module never priced."""
    if dtype not in CLIFF:
        loud(
            f"cross-entropy cliff for dtype {dtype}: the cliff is `-log` of half an "
            f"ulp of 1.0, and only {sorted(str(d) for d in CLIFF)} have been priced. "
            f"Escape: compute the loss from fp32 logits (what every arm in this repo "
            f"does), or add the dtype here with its half-ulp."
        )
    return CLIFF[dtype]


def ce_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The paper's margin `z_t - LSE_{j != t} z_j` per row, in fp32.

    Positive means "confidently correct" — the regime where the standard path
    collapses. This is the quantity `CLIFF` is compared against and what
    `scripts/ce_cliff_report.py` measures on real data; the fixed loss's own
    counter computes it from the terms it already has, so this twin exists for
    callers that only want to LOOK (a report, a test), not to compute a loss.
    """
    z = logits.float()
    idx = targets.reshape(-1, 1)
    zt = z.gather(1, idx)
    # The target's slot is -inf so the LSE is over j != t: the whole point is
    # that the target is NOT in the denominator's sum.
    return (zt - z.scatter(1, idx, float("-inf")).logsumexp(dim=1, keepdim=True)).reshape(-1)


def _reduce(rows: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return rows.mean()
    if reduction == "sum":
        return rows.sum()
    return rows


def _row_scale(rows: torch.Tensor, reduction: str) -> float:
    return 1.0 / rows.numel() if reduction == "mean" else 1.0


class _FixedCE(torch.autograd.Function):
    """`fixed_cross_entropy`'s body: forward is Eq. 6, backward is Eq. 7."""

    @staticmethod
    def forward(ctx, logits, targets, reduction, counters):
        z = logits.float()  # the loss's arithmetic island: always fp32
        idx = targets.reshape(-1, 1)
        zt = z.gather(1, idx)  # [N, 1] target logit
        masked = z.scatter(1, idx, float("-inf"))  # j != t only
        m = masked.max(dim=1, keepdim=True).values  # [N, 1] max over j != t
        s = torch.exp(masked - m).sum(dim=1, keepdim=True)  # >= 1, fp32
        u = m - zt  # [N, 1]; u = -margin
        # loss = log1p(s * exp(u)) — the target term lives in `u`, never added
        # to 1.0. The clamp only bites outside fp32's `exp` range, where the
        # equal branch below is exact; it is counted as `ce_clamp_hits`.
        loss = torch.log1p(s * torch.exp(u.clamp(U_MIN, U_MAX)))
        loss = torch.where(u > U_MAX, u + torch.log(s), loss)
        loss = torch.where(u < U_MIN, torch.zeros_like(loss), loss)
        # `shift = u - loss` is the log of the non-target denominator's share;
        # written this way (`-log(s) - softplus(-lw)` = the same number) it has
        # no `1e4 - 1e4` cancellation when the target is far BELOW the max.
        lw = u + torch.log(s)
        shift = -torch.log(s) - torch.nn.functional.softplus(-lw)
        p = torch.exp((z - m).scatter(1, idx, 0.0) + shift)  # q_j for j != t
        p.scatter_(1, idx, 0.0)
        # THE FIX (paper Eq. 7): the target entry is DEFINED as the negative
        # sum of the others. `q_t - 1` is the same number in exact arithmetic
        # and exactly 0 past the cliff, which is the bug.
        p.scatter_(1, idx, -p.sum(dim=1, keepdim=True))
        ctx.save_for_backward(p * _row_scale(loss, reduction))
        ctx.dtype = logits.dtype
        if counters is not None:
            margin = -(u + torch.log(s))  # == zt - LSE_{j != t}
            counters.count_device("ce_cliff_tokens", (margin > cliff(logits.dtype)).sum())
            counters.count_device("ce_clamp_hits", ((u > U_MAX) | (u < U_MIN)).sum())
        return _reduce(loss.reshape(-1), reduction)

    @staticmethod
    def backward(ctx, grad_out):
        (g,) = ctx.saved_tensors
        return (g * grad_out.reshape(-1, 1)).to(ctx.dtype), None, None, None


def fixed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    *,
    counters: Counters | None = None,
) -> torch.Tensor:
    """Hard-target CE with the Softmax Collapse fix (paper §2, Eq. 6-7).

    `logits [N, V]`, `targets [N]` → `[N]` (`reduction="none"`) or a scalar.
    The forward is `log1p(s·exp(m - t))` in fp32 with the target masked out of
    `m`/`s`; the backward is `q - onehot` with `q_t - 1` set to exactly
    `-(sum of the others)`, so `sum_c dL/dlogit_c == 0` at every margin.

    `counters` (optional) receives two DEVICE-side counters — `ce_cliff_tokens`
    (rows whose margin passes `cliff(logits.dtype)`) and `ce_clamp_hits` (rows
    whose exponent left fp32's `exp` band) — so a real run reports the
    prevalence without a host read.
    """
    if reduction not in ("mean", "sum", "none"):
        loud(
            f"fixed_cross_entropy reduction={reduction!r}: known values are 'mean', "
            f"'sum' and 'none'. Escape: one of those three."
        )
    if logits.ndim != 2:
        loud(
            f"fixed_cross_entropy logits {tuple(logits.shape)}: the function is the "
            f"2-D `[N, V]` head form (what the model returns). Escape: flatten the "
            f"leading positions, as `_byte_ce` does."
        )
    flat = targets.reshape(-1)
    if flat.numel() != logits.shape[0]:
        loud(
            f"fixed_cross_entropy: {flat.numel()} targets for {logits.shape[0]} rows. "
            f"Escape: one target per row."
        )
    if not logits.is_floating_point():
        loud(f"fixed_cross_entropy logits dtype {logits.dtype}: logits must be floating.")
    return _FixedCE.apply(logits, flat, reduction, counters)


class _FixedSoftCE(torch.autograd.Function):
    """`fixed_soft_ce`'s body: forward is the KL, backward is paper §3's `g`."""

    @staticmethod
    def forward(ctx, student, teacher, temperature, reduction, counters):
        zs = student.float() / temperature
        zt = teacher.float() / temperature
        p = torch.softmax(zt, dim=1)  # the teacher's distribution: a constant
        q = torch.softmax(zs, dim=1)  # the student's, for the fallback + the sum
        delta = zs - zt
        # d centres delta under p, s = log(q/p) via expm1/log1p: no `1 +`
        # and no `- 1`, so z_S == z_T gives d == s == 0 exactly (Eq. 9).
        d = delta - (p * delta).sum(dim=1, keepdim=True)
        s = d - torch.log1p((p * torch.expm1(d)).sum(dim=1, keepdim=True))
        small = s.abs() < 1.0
        u = torch.where(small, p * torch.expm1(s), q - p)  # q - p, cancellation-free
        g = u - q * u.sum(dim=1, keepdim=True)  # paper §3: sum(g) = 0
        kl = -(p * s).sum(dim=1)  # KL(p || q); 0 exactly when the logits match
        ctx.save_for_backward(g * (_row_scale(kl, reduction) / temperature))
        ctx.dtype = student.dtype
        if counters is not None:
            # The `q - p` fallback is a counted precision degradation (the one
            # branch where cancellation can still happen), never silent.
            counters.count_device("ce_softce_fallback", (~small).sum())
        return _reduce(kl, reduction)

    @staticmethod
    def backward(ctx, grad_out):
        (g,) = ctx.saved_tensors
        return (g * grad_out.reshape(-1, 1)).to(ctx.dtype), None, None, None, None


def fixed_soft_ce(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
    *,
    counters: Counters | None = None,
) -> torch.Tensor:
    """Soft-target KL with the Softmax Cancellation fix (paper §3).

    `KL(softmax(z_T/T) || softmax(z_S/T))` per row, in fp32. `temperature` is
    applied inside; the value is the temperature-scaled nats, so a distillation
    loss that wants Hinton's gradient-preserving scale multiplies by `T**2`
    itself. `student == teacher` gives a value of exactly 0 and a gradient of
    exactly 0 (`torch.equal`), which the standard `softmax(z_S) - softmax(z_T)`
    path cannot. `counters` receives `ce_softce_fallback`: the rows that left
    the `|s| < 1` band and fell back to the subtract-the-probabilities form.
    """
    if reduction not in ("mean", "sum", "none"):
        loud(
            f"fixed_soft_ce reduction={reduction!r}: known values are 'mean', 'sum' "
            f"and 'none'. Escape: one of those three."
        )
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        loud(
            f"fixed_soft_ce: student {tuple(student_logits.shape)} vs teacher "
            f"{tuple(teacher_logits.shape)} must both be `[N, V]`. Escape: one row "
            f"per example on each side."
        )
    if temperature <= 0.0:
        loud(
            f"fixed_soft_ce temperature={temperature}: T divides the logits, so 0 is "
            f"a division by zero and a negative T reflects them. Escape: T > 0."
        )
    return _FixedSoftCE.apply(
        student_logits, teacher_logits, float(temperature), reduction, counters
    )
