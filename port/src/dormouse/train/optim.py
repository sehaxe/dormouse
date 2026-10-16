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
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from ..instrument import loud

#: The paper's Jordan triple, `burn-muon-plus::NS_COEFFS`, verbatim.
JORDAN_NS_COEFFS = (3.4445, -4.7750, 2.0315)

#: The Newton–Schulz arms `zeropower_via_ns(kind=...)` runs. One tuple, so a
#: name that is not an arm is refused at the door instead of silently meaning
#: something (`config.NS_KINDS` mirrors this list for the config layer).
NS_KINDS = ("classic", "gram")

#: The 3-D `colrow_normalize` arms (`config.MUON_COLROW_3D` mirrors this list):
#: `frobenius` is the per-slice Frobenius normalisation the batched Muon step
#: has always run — NOT the ColRow the port documents — and `colrow` is the
#: 2-D definition applied per slice, i.e. the fix. The defect, its exact size
#: and the A/B it owes are in `colrow_normalize`'s docstring and
#: `tests/test_muon_fast.py`.
COLROW_3D = ("frobenius", "colrow")

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
    """A name that is not an arm is LOUD, naming both arms and the escape."""
    if kind not in NS_KINDS:
        loud(
            f"ns_kind={kind!r} is not an arm. Known: {list(NS_KINDS)} — `classic` "
            f"forms `X @ X.T` at every iteration (the default, byte-identical to "
            f"this port's history), `gram` is Gram Newton–Schulz (arXiv 2608.11612 "
            f"§4): the same polynomial iterated in the `[r, r]` Gram space. Escape: "
            f"name one of those, or land the arm in `optim.NS_KINDS` first."
        )


def _check_colrow_3d(arm: str) -> None:
    """A name that is not an arm is LOUD, naming both arms and the escape."""
    if arm not in COLROW_3D:
        loud(
            f"muon_colrow_3d={arm!r} is not an arm. Known: {list(COLROW_3D)} — "
            f"`frobenius` is the per-slice Frobenius normalisation the batched "
            f"Muon step has always run (the 1/sqrt(d0)-of-ColRow defect, kept as "
            f"the A/B control), `colrow` is the 2-D definition applied per slice "
            f"(the fix, bit-identical to the per-param path). Escape: name one of "
            f"those, or land the arm in `optim.COLROW_3D` first."
        )


def zeropower_via_ns(
    g: torch.Tensor,
    steps: int = 5,
    kind: str = "classic",
    gram_restarts: tuple[int, ...] = GRAM_NS_RESTARTS,
) -> torch.Tensor:
    """The polar factor's polynomial approximation, matrix-only, fp32.

    `kind` selects the arm (`NS_KINDS`): `classic` is the recurrence this port
    has always run; `gram` is Gram Newton–Schulz — the same polynomial,
    iterated on the `[r, r]` Gram and applied once at the end, so the two
    rectangular matmuls per iteration are gone (`_gram_via_ns`).

    The batched form accepts `[N, d0, d1]`: the polynomial is ELEMENTWISE in
    the leading batch dim (a Gram per slice), and `x @ x.T` broadcasts
    unchanged, so the same math runs for N slices at once — that is the
    whole point of the batched Muon pass (version-branched off the same
    body, no second implementation exists here).
    """
    _check_ns_kind(kind)
    if kind == "gram":
        return _gram_via_ns(g, steps=steps, restarts=gram_restarts)
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


def _gram_via_ns(
    g: torch.Tensor, *, steps: int, restarts: tuple[int, ...] = GRAM_NS_RESTARTS
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
    This port's NS is fp32; a bf16 NS chain is a separate, unmeasured arm
    (row 4 of `docs/research/2026-10-16-muon-orthogonalization-cost.md`). Its
    CuTeDSL symmetric-GEMM kernels are Hopper/Blackwell-only and are not
    ported either — this is the ALGORITHM, in torch.
    """
    a, b, c = JORDAN_NS_COEFFS
    assert g.ndim in (2, 3), f"NS is matrix-only or [N,d0,d1] batched, got shape {tuple(g.shape)}"
    x = g.to(torch.float32)
    x = x / (x.norm(dim=(-1, -2), keepdim=True) + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-1, -2).contiguous()
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
    return out.transpose(-1, -2).contiguous() if transposed else out


def colrow_normalize(x: torch.Tensor, eps: float = 1e-7, *, colrow_3d: str = "colrow"):
    """`Norm_d` for `NormDir::ColRow` (the trainer's choice): columns, then rows.

    **The 2-D form is the definition and is untouched by `colrow_3d`.** The 3-D
    form has two arms (`COLROW_3D`), because the batched step's original 3-D
    branch was NOT ColRow — found 2026-10-16 by the equivalence gate in
    `tests/test_muon_fast.py`:

    * `"frobenius"` (**the default, and what every number on record ran**):
      collapses dims `(1, 2)` twice, i.e. normalises each slice by its Frobenius
      norm. The batched Muon step therefore applies exactly `1/sqrt(d0)` of the
      per-param update (8x at `[64, 5461]`/`[64, 2048]`, 45x at `[2048, 2048]`).
      This is the DEFECT, kept as the control so the A/B is apples-to-apples.
    * `"colrow"` (**the fix**): dims 1 then 2 per slice — the 2-D definition
      applied to each matrix, which `tests/test_muon_fast.py` shows is
      bit-identical to running `colrow_normalize` on the slices one at a time.

    The arm is a config key (`train.muon_colrow_3d`) and an optimizer attribute,
    never a param-group key: the group is what `load_state_dict` REPLACES with
    the checkpoint's copy, so a key added today would KeyError on every
    checkpoint written before it (the same rule `ns_kind` follows). A switch
    whose `"colrow"` arm moves the update scale 8-45x needs the standing 3-seed
    A/B (AGENTS §1.2) before it can become the default.
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


class _MomGroup:
    """One shape group's canonical STACKED momentum buffer.

    `mom` is `[N, d0, d1]` fp32 and it IS the state: `state[p]["momentum"]` is
    a view of row `i`, so the step never stacks the group in and copies it back
    (at b1 that was 3 x 1.33 GiB of fp32 traffic per step for the `[64, 5461]`
    expert writes alone, plus the allocator churn of a fresh 1.33 GiB block
    every step). The checkpoint format is unchanged — `Optimizer.state_dict`
    still emits one `momentum` tensor per parameter, and `load_state_dict`
    rebuilds the stack from them once.
    """

    __slots__ = ("gi", "shape", "params", "mom")

    def __init__(self, gi: int, shape: tuple[int, int], params: list, mom: torch.Tensor) -> None:
        self.gi = gi
        self.shape = shape
        self.params = params
        self.mom = mom


class Muon(Optimizer):
    """Momentum + polar orthogonalization on 2-D parameters, fp32."""

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
    ) -> None:
        _check_ns_kind(ns_kind)
        _check_colrow_3d(colrow_3d)
        #: The arm and its restart positions live on the OPTIMIZER, not in the
        #: param group: the group is what `Optimizer.load_state_dict` REPLACES
        #: with the checkpoint's copy, so a key added today would KeyError on
        #: every checkpoint written before it. The arm is already pinned across
        #: a resume by the config snapshot (`train.ns_kind` is not a progress
        #: key, so `RunConfig::diff_keys` refuses a flip) — one place, not two.
        #: `colrow_3d` follows the same rule (`train.muon_colrow_3d`).
        self.ns_kind = ns_kind
        self.gram_restarts = gram_restarts
        self.colrow_3d = colrow_3d
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))
        #: One stacked momentum buffer per (param group, shape) — the canonical
        #: state, built lazily by `_build_groups` and rebuilt after a load.
        self._groups: list[_MomGroup] = []

    def _build_groups(self) -> None:
        """One stacked momentum buffer per unique shape; per-param state views it.

        Idempotent and cheap, but it STACKS whatever is in
        `state[p]["momentum"]`, so it is called (a) lazily by the batched step
        and (b) after `load_state_dict` — where the loader has replaced the
        views with the checkpoint's own per-param tensors, which is exactly the
        historical format and stays readable.
        """
        self._groups = []
        for gi, group in enumerate(self.param_groups):
            by_shape: dict[tuple[int, int], list] = {}
            for p in group["params"]:
                if p.ndim == 2:
                    by_shape.setdefault((p.shape[0], p.shape[1]), []).append(p)
            for shape, params in by_shape.items():
                mom = torch.zeros(len(params), *shape, dtype=torch.float32, device=params[0].device)
                for i, p in enumerate(params):
                    prev = self.state[p].get("momentum")
                    if prev is not None:
                        mom[i].copy_(prev)
                    self.state[p]["momentum"] = mom[i]
                self._groups.append(_MomGroup(gi, shape, params, mom))

    def load_state_dict(self, state_dict) -> None:  # noqa: D102 - torch's contract
        super().load_state_dict(state_dict)
        # The loader installed the checkpoint's per-param `momentum` tensors;
        # stacking them once here is what makes an old checkpoint (per-param)
        # and a new one (views of one storage) both load correctly.
        self._build_groups()

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
                update = colrow_normalize(
                    zeropower_via_ns(buf, ns, self.ns_kind, self.gram_restarts)
                )
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(update.to(p.dtype), alpha=-lr * scale)
        return loss

    def _step_one(self, p: torch.Tensor, lr: float, mu: float, ns: int, scale: float) -> None:
        """The per-param update — the reference the batched path must match.

        Kept because it is the definition, and used by the batched step for the
        rare group where a 2-D parameter carries no gradient (a shape with one
        live member is not worth a batch). It reads and writes
        `state[p]["momentum"]`, which on a built group is a VIEW of the stacked
        buffer, so both paths share one state.
        """
        state = self.state[p]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p.grad)
        buf = state["momentum"]
        buf.mul_(mu).add_(p.grad, alpha=1.0 - mu)
        update = colrow_normalize(zeropower_via_ns(buf, ns, self.ns_kind, self.gram_restarts))
        with torch.no_grad():
            p.add_(update.to(p.dtype), alpha=-lr * scale)

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
        remains per step is the NS chain and the final `_foreach_add_`.
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
            # buffer and `torch.unbind` hands out its rows as views.
            mom.mul_(mu)
            torch._foreach_add_(torch.unbind(mom, 0), [p.grad for p in params], alpha=1.0 - mu)
            n = len(params)
            # Chunk the shape stack: the cap is fp32 ELEMENTS per stacked
            # buffer (`MUON_CHUNK_ELEMS`), so NS never materializes the whole
            # shape group's chain at once.
            n_chunk = max(1, min(n, MUON_CHUNK_ELEMS // (grp.shape[0] * grp.shape[1])))
            for start in range(0, n, n_chunk):
                params_c = params[start : start + n_chunk]
                update = zeropower_via_ns(
                    mom[start : start + n_chunk], ns, self.ns_kind, self.gram_restarts
                )
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
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        if opt not in ("mix", "adamw"):
            loud(f"opt={opt!r}: known optimizers are 'mix' (default) and 'adamw'")
        _check_ns_kind(ns_kind)
        _check_colrow_3d(colrow_3d)
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
            Muon(
                muon_params,
                lr=lr,
                momentum=momentum,
                ns_steps=ns_steps,
                ns_kind=ns_kind,
                gram_restarts=gram_restarts,
                colrow_3d=colrow_3d,
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
