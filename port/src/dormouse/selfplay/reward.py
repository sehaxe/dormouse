"""The generator reward — **Eq. 2** of arXiv:2609.30063 (paper §2.2).

The generator is rewarded for programs whose learner gradient aligns with the
learner's recent movement, in the inner product the AdamW preconditioner
defines::

    p(e) = floor(e / 2)                      δθ_e = θ_{p(e)} − θ_e
    r_i  = |⟨∇_θ L(y_i; θ_e), P_e ⊙ δθ_e⟩|   P_e = lr / (sqrt(v̂_e) + ε)

where `L(y; θ)` is the per-sequence mean cross-entropy over the output's content
tokens, `θ_{p(e)}` is the learner checkpoint at the growing lookback horizon
`⌈e/2⌉`, and `P_e` is the diagonal AdamW step operator read out of the learner's
optimizer state. The reward is what makes the curriculum adaptive: a mastered
program induces a near-zero gradient and an unrelated one induces a gradient
that does not align with `δθ`, so **both score low**, and only programs that are
not yet mastered *and* congruent with recent progress score high.

**Forward mode, as in the paper.** The paper computes Eq. 2 with forward-mode
automatic differentiation "to avoid materializing full gradients" (footnote 1
points at a fused `jvp` kernel). This module does the same with
`torch.func.jvp`, one pass for the whole batch — the JVP of a batched per-sequence
loss is the vector of directional derivatives, which is exactly `⟨∇L(y_i), u⟩`
for every `i` at once.

Two deviations, both named rather than hidden:

1. **No fused kernel.** The paper's footnote cites `jvp_flash_attention`; we run
   the plain torch path, so the reward is a small eager graph, not a fused one.
   At the port's scale the JVP is ~20-40 ms on a 100k-parameter model, so the
   kernel is an optimisation to revisit, not a blocker.
2. **The MATH SDPA backend.** `torch.func.jvp` through the model's attention arm
   raises `NotImplementedError: Trying to use forward AD with
   _scaled_dot_product_flash_attention_for_cpu` — torch has no forward-AD rule
   for the CPU flash kernel. `_jvp_rewards` therefore runs the forward inside
   `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])`, which is a *dispatch*
   choice made here and nowhere else (no model file is touched). The arithmetic
   of the composite kernel differs from flash's by fp32 round-off, so a reward
   read under MATH is not bit-identical to the same forward under flash.

`method="vjp"` is the reverse-mode fallback, and it is genuinely more expensive:
reverse mode has no batched directional derivative, so it needs **one backward
pass per program** where forward mode needs one pass for the whole pool. It
exists because a future model arm may not be forward-AD-differentiable, and it
is tested to agree with the JVP path to fp32 round-off.

Nothing here modifies the model or the trainer: the function reads parameters,
the caller's `θ_prev` snapshot and the caller's optimizer state, and returns
floats.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..instrument import Counters, loud

__all__ = [
    "OUT_PREFIX",
    "PROGRAM_PREFIX",
    "adamw_preconditioner",
    "gradient_alignment_rewards",
    "lookback",
    "output_batch",
    "sequence_ce",
]

#: §2.3: "Programs and outputs are prefixed by the bytes `S` and `O`".
PROGRAM_PREFIX = ord("S")
OUT_PREFIX = ord("O")

#: AdamW's default ε; the paper writes ε without a value.
DEFAULT_EPS = 1e-8


#: Eq. 2's lookback `p(e) = floor(e / 2)` — `⌈e/2⌉` rounds back from `e`.
def lookback(e: int) -> int:
    """Eq. 2's `p(e) = ⌊e/2⌋`: the round whose checkpoint θ_{p(e)} is compared."""
    assert e >= 0, f"round index must be >= 0, got {e}"
    return e // 2


def output_batch(outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`[N, T]` output bytes → `(inputs, targets)` with the `O` prefix (§2.3).

    The prefix is input-only: the model reads `O` and predicts the first content
    byte, so the loss is over the T content tokens and the marker is never
    asked for — the same convention the paper's `O`-prefixed outputs encode.
    """
    assert outputs.ndim == 2, f"outputs must be [N, T], got {tuple(outputs.shape)}"
    n, t = outputs.shape
    assert t >= 1, "an output of length 0 has no content token to predict"
    prefix = torch.full((n, 1), OUT_PREFIX, dtype=outputs.dtype, device=outputs.device)
    return torch.cat([prefix, outputs[:, :-1]], dim=1), outputs


def sequence_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean CE over the content tokens → `[N]` (Eq. 1's `L(y; θ)`).

    Eq. 1 averages `L(y_i; θ)` over the pool, and Eq. 2 differentiates one
    `L(y_i; θ)`, so the per-sequence form is the primitive: `mean` over the T
    positions, not a sum, so a long output is not rewarded more for its length.
    """
    assert logits.ndim == 3, f"logits must be [N, T, vocab], got {tuple(logits.shape)}"
    assert logits.shape[:2] == targets.shape, (
        f"logits {tuple(logits.shape)} do not match targets {tuple(targets.shape)}"
    )
    lp = F.log_softmax(logits.float(), dim=-1)
    return -lp.gather(-1, targets[..., None]).squeeze(-1).mean(dim=1)


def _bias_corrected_v(
    optimizer: torch.optim.Optimizer, p: torch.nn.Parameter
) -> torch.Tensor | None:
    """AdamW's `v̂_e` for `p`, or `None` when the optimizer has never stepped it.

    `None` is not an error: AdamW creates state only for a parameter that
    received a gradient, and a parameter that never received one was never
    updated, so `θ_{p(e)}` and `θ_e` are the same tensor and `δθ_e = 0` for it.
    [`_direction`] is what turns that argument into a checked invariant.
    """
    state = optimizer.state.get(p)
    if not state or "exp_avg_sq" not in state:
        return None
    beta2 = float(_group_for(optimizer, p)["betas"][1])
    step = int(state["step"])
    assert step >= 1, f"AdamW state exists with step {step}: nothing was stepped"
    return state["exp_avg_sq"] / (1.0 - beta2**step)


def _group_for(optimizer: torch.optim.Optimizer, p: torch.nn.Parameter) -> dict[str, object]:
    for g in optimizer.param_groups:
        if any(gp is p for gp in g["params"]):
            return g  # type: ignore[return-value]
    return optimizer.param_groups[0]  # type: ignore[return-value]


def adamw_preconditioner(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    *,
    lr: float | None = None,
    eps: float = DEFAULT_EPS,
) -> dict[str, torch.Tensor]:
    """`P_e = lr / (sqrt(v̂_e) + ε)` per parameter, from the optimizer's state.

    `lr` defaults to the optimizer group's own learning rate — the paper's `lr`
    is "the learner's", so the optimizer is the authority unless the caller
    overrides it. A parameter the optimizer has no state for gets `P_e = 0`:
    it has never been stepped, so `δθ_e = 0` and the product is 0 either way
    ([`_direction`] refuses to let that assumption go unchecked).
    """
    out: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        group_lr = float(_group_for(optimizer, p)["lr"])
        vhat = _bias_corrected_v(optimizer, p)
        if vhat is None:
            out[name] = torch.zeros_like(p)
        else:
            out[name] = (lr if lr is not None else group_lr) / (vhat.sqrt() + eps)
    return out


def _direction(
    model: nn.Module,
    theta_prev: Mapping[str, torch.Tensor],
    precond: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """`u = P_e ⊙ δθ_e`, with `δθ_e = θ_{p(e)} − θ_e` (Eq. 2).

    The host read on a stateless parameter is the guard for the `P_e = 0`
    convention in [`adamw_preconditioner`]: a zero preconditioner silently zeroes
    a term, and it is only *correct* because such a parameter never moved. The
    round is already a host-read boundary (it returns floats), so this is not a
    per-step sync — but it is one read per untouched parameter, and if that ever
    shows up in a profile the fix is to make `adamw_preconditioner` return the
    stateless names instead of probing the values.
    """
    params = dict(model.named_parameters())
    missing = sorted(set(params) - set(theta_prev))
    if missing:
        loud(
            f"theta_prev is missing {len(missing)} learner parameter(s), first "
            f"{missing[0]!r}: δθ_e = θ_{{p(e)}} − θ_e must be a full parameter "
            f"snapshot or the reward is a partial dot product. Escape: save the "
            f"learner's state_dict at every round and pass round e//2's."
        )
    missing_p = sorted(set(params) - set(precond))
    if missing_p:
        loud(f"precond is missing {len(missing_p)} parameter(s), first {missing_p[0]!r}")
    u: dict[str, torch.Tensor] = {}
    for name, p in params.items():
        delta = theta_prev[name].to(device=p.device, dtype=p.dtype) - p.detach()
        pre = precond[name].to(device=p.device, dtype=p.dtype)
        if not bool(pre.any()) and bool(delta.any()):
            loud(
                f"{name!r} has a zero preconditioner (the learner's optimizer has "
                f"no state for it) but a NON-zero δθ_e. Escape: the θ_prev snapshot "
                f"and the optimizer belong to different lineages — pass the "
                f"snapshot taken immediately before the step this optimizer state "
                f"came from."
            )
        u[name] = (pre * delta).detach()
    return u


def _jvp_rewards(
    model: nn.Module,
    outputs: torch.Tensor,
    u: Mapping[str, torch.Tensor],
) -> list[float]:
    """One forward-mode pass for the whole batch — the paper's mechanism."""
    inputs, targets = output_batch(outputs)
    params = dict(model.named_parameters())

    def per_seq_loss(p: Mapping[str, torch.Tensor]) -> torch.Tensor:
        logits = functional_call(model, p, (inputs,), {"targets": None})[0]
        n, t = inputs.shape
        return sequence_ce(logits.reshape(n, t, -1), targets)

    # MATH, not flash: torch has no forward-AD rule for the CPU flash SDPA
    # kernel (see the module docstring). Dispatch only; no model file changes.
    with sdpa_kernel([SDPBackend.MATH]):
        _, directional = jvp(per_seq_loss, (params,), (dict(u),))
    return directional.abs().tolist()


def _vjp_rewards(
    model: nn.Module,
    outputs: torch.Tensor,
    u: Mapping[str, torch.Tensor],
) -> list[float]:
    """Reverse-mode fallback: `∇L(y_i; θ)` per program, dotted with `u`.

    Reverse mode cannot batch the directional derivative the way forward mode
    can, so this is one backward pass per program (plus one shared forward). It
    exists for an arm that is not forward-AD-differentiable, and it is the path
    the paper's "avoid materializing full gradients" argument is an argument
    *against*.
    """
    inputs, targets = output_batch(outputs)
    logits = model(inputs, None)[0]
    n, t = inputs.shape
    per_seq = sequence_ce(logits.reshape(n, t, -1), targets)
    names = [name for name, _ in model.named_parameters()]
    ps = [p for _, p in model.named_parameters()]
    dots: list[torch.Tensor] = []
    for i in range(per_seq.shape[0]):
        grads = torch.autograd.grad(
            per_seq[i], ps, retain_graph=(i + 1 < per_seq.shape[0]), allow_unused=True
        )
        acc = per_seq.new_zeros(())
        for name, g in zip(names, grads, strict=True):
            if g is not None:
                acc = acc + (g.detach() * u[name]).sum()
        dots.append(acc.detach())
    return torch.stack(dots).abs().tolist()


def gradient_alignment_rewards(
    model: nn.Module,
    outputs: torch.Tensor,
    *,
    theta_prev: Mapping[str, torch.Tensor],
    precond: Mapping[str, torch.Tensor],
    method: str = "jvp",
    counters: Counters | None = None,
) -> list[float]:
    """Eq. 2 for every output in the batch, aligned with `outputs` by index.

    `outputs` is `[N, T]` byte ids (what `utm.run(..., out_len=T)` returns);
    `theta_prev` is `θ_{p(e)}`; `precond` is `P_e` from
    [`adamw_preconditioner`]. `method` is `"jvp"` (the paper's forward mode),
    `"vjp"` (the reverse-mode fallback) or `"auto"` — which tries JVP and falls
    back to VJP with the counter `reward_vjp_fallback` set, so a reader of the
    log sees that the paper's mechanism was not the one that ran.
    """
    assert method in ("jvp", "vjp", "auto"), f"unknown reward method {method!r}"
    assert outputs.numel() > 0, "no outputs: there is nothing to reward"
    u = _direction(model, theta_prev, precond)
    if method == "vjp":
        return _vjp_rewards(model, outputs, u)
    try:
        rewards = _jvp_rewards(model, outputs, u)
    except Exception as e:  # noqa: BLE001 - the COUNTED fallback's whole purpose
        if method == "jvp":
            raise
        warnings.warn(
            f"forward-mode reward unavailable ({type(e).__name__}: {e}); falling "
            f"back to one reverse-mode pass per program — COUNTED as "
            f"reward_vjp_fallback",
            stacklevel=2,
        )
        if counters is not None:
            counters.count("reward_vjp_fallback")
        return _vjp_rewards(model, outputs, u)
    if counters is not None:
        counters.count("reward_jvp", len(rewards))
    return rewards
