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
* `M_t = μ·M_{t-1} + (1-μ)·G_t` with **no Nesterov lerp** (the paper's Eq. 4;
  the crate's manifest was the defect, not the code).
* `O_t = Norm_d(Ortho(M_t))` with ColRow normalisation, and the step
  `W -= η·max(1, m/n)^0.5·O_t` — `max(1, ·)` is Jordan's, which is what both
  the crate and this port follow.
* `orthogonalize` divides by the Frobenius norm, and the NS loop runs in the
  TRANSPOSED orientation when the matrix is tall, i.e. in the `[r, r]` Gram
  space of the smaller side — the burn economics that keeps `[768, 768]` out
  of fp32 NS entirely.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from ..instrument import loud

#: The paper's Jordan triple, `burn-muon-plus::NS_COEFFS`, verbatim.
JORDAN_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_ns(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """The polar factor's polynomial approximation, matrix-only, fp32.

    The batched form accepts `[N, d0, d1]`: the polynomial is ELEMENTWISE in
    the leading batch dim (a Gram per slice), and `x @ x.T` broadcasts
    unchanged, so the same math runs for N slices at once — that is the
    whole point of the batched Muon pass (version-branched off the same
    body, no second implementation exists here).
    """
    if g.ndim == 2:
        a, b, c = JORDAN_NS_COEFFS
        x = g.to(torch.float32)
        x = x / (x.norm() + 1e-7)
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.T.contiguous()
        for _ in range(steps):
            gram = x @ x.T
            poly = b * gram + c * (gram @ gram)
            x = a * x + poly @ x
        return x.T.contiguous() if transposed else x
    assert g.ndim == 3, f"NS is matrix-only or [N,d0,d1] batched, got shape {tuple(g.shape)}"
    a, b, c = JORDAN_NS_COEFFS
    x = g.to(torch.float32)
    x = x / (x.norm(dim=(1, 2), keepdim=True) + 1e-7)
    transposed = x.shape[1] > x.shape[2]
    if transposed:
        x = x.transpose(1, 2).contiguous()
    for _ in range(steps):
        gram = x @ x.transpose(1, 2)
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x
    return x.transpose(1, 2).contiguous() if transposed else x


def colrow_normalize(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """`Norm_d` for `NormDir::ColRow` (the trainer's choice): columns, then rows."""
    dim = (1, 2) if x.ndim == 3 else 0
    x = x / (x.norm(dim=dim, keepdim=True) + eps)
    return x / (x.norm(dim=(1, 2) if x.ndim == 3 else 1, keepdim=True) + eps)


class Muon(Optimizer):
    """Momentum + polar orthogonalization on 2-D parameters, fp32."""

    def __init__(
        self,
        params,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102 - torch's Optimizer contract
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, mu, ns = group["lr"], group["momentum"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    loud(f"Muon got a rank-{g.ndim} gradient: the router must not send it here")
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)
                buf = state["momentum"]
                buf.mul_(mu).add_(g, alpha=1.0 - mu)
                update = colrow_normalize(zeropower_via_ns(buf, ns))
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(update.to(p.dtype), alpha=-lr * scale)
        return loss

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
        """
        by_shape: dict[tuple[int, int], list] = {}
        for group in self.param_groups:
            lr, mu, ns = group["lr"], group["momentum"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None or p.grad.ndim != 2:
                    continue
                by_shape.setdefault((p.shape[0], p.shape[1]), []).append(p)
        n_batches = 0
        for shape, params in by_shape.items():
            for p in params:
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
            if len(params) < 2:
                # Match the per-param math exactly: one computation per tensor.
                for p in params:
                    g = p.grad
                    state = self.state[p]
                    buf = state["momentum"]
                    buf.mul_(mu).add_(g, alpha=1.0 - mu)
                    update = colrow_normalize(zeropower_via_ns(buf, ns))
                    scale = max(1.0, shape[0] / shape[1]) ** 0.5
                    with torch.no_grad():
                        p.add_(update.to(p.dtype), alpha=-lr * scale)
                continue
            n = len(params)
            bufs = torch.stack([self.state[p]["momentum"] for p in params])
            grads = torch.stack([p.grad for p in params])
            bufs.mul_(mu).add_(grads.detach(), alpha=1.0 - mu)
            update = colrow_normalize(zeropower_via_ns(bufs, ns))
            scale = max(1.0, shape[0] / shape[1]) ** 0.5
            updates = update.mul(-lr * scale).detach()
            for i, p in enumerate(params):
                with torch.no_grad():
                    p.add_(updates[i].to(p.dtype))
            for i, p in enumerate(params):
                with torch.no_grad():
                    self.state[p]["momentum"].copy_(bufs[i])
            n_batches += 1
        return n_batches


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
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        if opt not in ("mix", "adamw"):
            loud(f"opt={opt!r}: known optimizers are 'mix' (default) and 'adamw'")
        muon_params: list[nn.Parameter] = []
        adamw_params: list[nn.Parameter] = []
        self.routing: dict[str, str] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            group = "muon" if (opt == "mix" and is_muon_param(name, p)) else "adamw"
            self.routing[name] = group
            (muon_params if group == "muon" else adamw_params).append(p)
        self.muon = (
            Muon(muon_params, lr=lr, momentum=momentum, ns_steps=ns_steps) if muon_params else None
        )
        self.adamw = AdamW(adamw_params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
        #: Let a caller (the trainer, the smoke test) keep looking at the groups.
        self.n_muon, self.n_adamw = len(muon_params), len(adamw_params)
        self.steps = 0
        #: The SHAPE answer: identical-shape parameters update as ONE stack.
        #: The per-param loop cost 84% of b1's step in per-tensor dispatches.
        self.muon_params = muon_params
        self.model = model
        self.steps_batched = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

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
