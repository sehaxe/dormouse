"""Manifold-Constrained Hyper-Connections (mHC, arXiv:2512.24880) — Eq. 3/7/8/9.

Ported from `vendor/dormouse-fused/crates/burn-mhc/src/{lib,block,sinkhorn}.rs`;
each formula carries a `burn: <file>:<line>` citation to the line it transcribes.

The operator, for `n` residual lanes of width `d` (`x_l` flattened to
`vec(x_l) ∈ R^{n·d}`):

* **Eq. 7** — `x̃' = RMSNorm(vec(x_l))`, then the three first-order
  hyper-network outputs are linear projections with a small gate and a static
  bias: `H̃ = α·(x̃'φ) + b` (`block.rs:105-195`).
* **Eq. 8** — `H_pre = σ(H̃_pre)`, `H_post = 2σ(H̃_post)`, `H_res =
  Sinkhorn-Knopp(H̃_res)` (`block.rs:189-194`).
* **Eq. 9** — the Sinkhorn-Knopp projection onto the Birkhoff polytope, 20
  iterations of alternating row/column normalization (`sinkhorn.rs:22-71`,
  `t_max = 20` at `:6`). **Log domain**: each normalization is
  `m -= logsumexp(m)`, shift-invariant and overflow-free, because the
  hyper-network output is unbounded (`sinkhorn.rs:8-21`).
* **Eq. 3** — `x_{l+1} = H_res x_l + H_postᵀ F(H_pre x_l)` (`block.rs:197-227`).

Two readings of the paper are decided the way the crate decided them, both in
favour of the PRINTED equations (`lib.rs:28-54`):

* **Eq. 7 has no `tanh`.** HC's Eq. 5 squashes its projections; mHC's Eq. 7
  (and its kernel section's Eq. 14-16) is the plain linear map above. The
  consequence is named: `φ_*` are unbounded, which is why the Sinkhorn input is
  unbounded and `sinkhorn_knopp` must run in the log domain.
* **`H_pre` is applied by the CALLER, not inside `forward`.** Eq. 3's
  `H_pre x_l` is `[1,n]·[n,C] = [1,C]`: it feeds the layer's own `F`, which
  lives outside this module. [`MHCMix.pre_mix`] is that read-out; `forward`
  implements only Eq. 3's `H_res x_l + H_postᵀ F` terms.

Static-bias init: `b_pre = 10`, `b_post = 0`, `b_res = 0` (`block.rs:5-25`,
`:86-90`). `b_res = 0` is the 2026-10-02 **fidelity fix**: the old `10·I` put
the diagonal at `e^10` against off-diagonals at `e^-10`, measured
`|grad α_res| = 7.3e-9` on the exponential's flat top
(`docs/reviews/fidelity-audit-2026-10-02.md` F-F4); at 0 the Sinkhorn starts at
uniform mixing and the projection can leave it. α init 0.01 and the RMSNorm
ε = 1e-20 are App. A.1 Table 5's (`lib.rs:66-72`).

**Layout — a named departure from the crate.** The crate's `h` is one
`[B,T,D]` tensor sliced into `n` lanes of width `C = D/n`, so `H_pre` collapses
the stream to `C` and the crate's caller (a `D`-wide body) cannot apply it
(`block.rs:211-217`). The loop here carries `n` lanes of the FULL residual
width (`C = D`, readout = their sum — 2409.19606 §2.1's readout, the base
paper's, `docs/reviews/mhc-2026-09-30.md` §1.2): every lane is a complete
residual state, `H_res` mixes them, `H_pre` is a sigmoid-weighted mix down to
the body's `D`-wide input, and `H_post_j` weights the same branch output onto
lane `j`. This is the only layout that composes a `D`-wide body with a
`D`-wide stream, and it makes the 11 % of the paper's effect the crate left
unwired (`H_pre`, `mhc-2026-09-30.md` §3.3) reachable.
"""

from __future__ import annotations

import torch
from torch import nn

from .. import stepaudit

__all__ = [
    "ALPHA_INIT",
    "EPS",
    "MHCMix",
    "SINKHORN_IMPLS",
    "SINKHORN_ITERS",
    "STATIC_INIT",
    "sinkhorn_knopp",
]

#: Sinkhorn-Knopp iterations (App. A.1: `t_max = 20`) — `sinkhorn.rs:5-6`.
SINKHORN_ITERS = 20
#: Eq. 9's normaliser. `"logsumexp"` is the shipped arithmetic, bit for bit:
#: `m -= logsumexp(m)`. `"log_softmax"` is the SAME expression as one ATen op
#: (`log_softmax` returns `m - logsumexp(m)`), so the value moves only by fp32
#: roundoff (≤2.4e-7 absolute on probabilities, ≤1.8e-7 on the gradient,
#: measured in `tests/test_mhc_write_launches.py`) while the launch family that
#: the kernel profile priced — the per-normalisation `sub` that CUDA lowers to
#: `CUDAFunctor_add` — disappears: 120 per Sinkhorn call, 881 ATen dispatches
#: down to 81. The fused kernel (`fused_residual._write_kernel`) does not read
#: this field: its 20 iterations are already in registers.
SINKHORN_IMPLS = ("logsumexp", "log_softmax")
#: Gating-factor init (App. A.1: `alpha = 0.01`) — `block.rs:5-6`.
ALPHA_INIT = 0.01
#: Static-bias magnitude: `sigmoid(10) ~ 1`, the full read-in gate —
#: `block.rs:7-9`. `b_pre` starts here; `b_post`/`b_res` start at 0.
STATIC_INIT = 10.0
#: RMSNorm's ε (App. A.1 Table 5, "Layer Norm eps = 1e-20") — `block.rs:109-118`.
EPS = 1e-20
# burn: block.rs:5-9, :86-90; sinkhorn.rs:6 — the four crate constants above.


def sinkhorn_knopp(
    logits: torch.Tensor,
    iters: int = SINKHORN_ITERS,
    impl: str = "logsumexp",
) -> torch.Tensor:
    """Eq. 9: the log-domain Sinkhorn-Knopp projection onto the Birkhoff polytope.

    `logits [..., n, n]` → `M⁽⁰⁾ = exp(logits)`, then `iters` alternating
    row/column normalizations (`T_r(T_c(·))`). Each normalization is
    `m -= logsumexp(m)`, which is shift-invariant and never overflows — the
    naive exp-then-divide form breaks on the hyper-network's unbounded output
    (`sinkhorn.rs:8-21`). Burn: `sinkhorn.rs:22-71`.

    `impl="log_softmax"` replaces each `m - logsumexp(m)` with the single
    `torch.log_softmax` op that is the same expression: one dispatch (≈3 CUDA
    kernels) instead of a reduction's ~6 kernels plus the separate subtract.
    The value is the same to fp32 roundoff and NOT bit-identical, which is why
    it is a named arm with `logsumexp` as the default.
    """
    assert impl in SINKHORN_IMPLS, (
        f"sinkhorn_knopp(impl={impl!r}) is not a normaliser. Escape: impl in {SINKHORN_IMPLS}."
    )
    m = logits
    if impl == "log_softmax":
        for _ in stepaudit.iterations("mhc.sinkhorn", range(iters)):
            # `log_softmax(dim)` IS `m - logsumexp(m, dim, keepdim=True)`.
            m = torch.log_softmax(m, dim=-1)
            m = torch.log_softmax(m, dim=-2)
        return m.exp()
    for _ in stepaudit.iterations("mhc.sinkhorn", range(iters)):
        # burn: sinkhorn.rs:48-68 — row then column, both `m -= logsumexp(m)`.
        m = m - torch.logsumexp(m, dim=-1, keepdim=True)
        m = m - torch.logsumexp(m, dim=-2, keepdim=True)
    return m.exp()


class MHCMix(nn.Module):
    """Eq. 3's residual mixing + Eq. 7's hyper-network, for `n` full-width lanes.

    `x_streams [b, t, n, d]` is the residual stream (`d = C` here — see the
    module docstring's layout note); `x_in [b, t, d]` is the caller's branch
    output `F(H_pre x_l)`. [`forward`] returns the next lane set; the caller
    reads it out by summing the lanes (2409.19606 §2.1).
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int = 2,
        device: torch.device | str = "cpu",
        sinkhorn: str = "logsumexp",
    ) -> None:
        super().__init__()
        assert n_streams >= 1, "n_streams (the expansion rate n) must be >= 1"
        assert sinkhorn in SINKHORN_IMPLS, (
            f"sinkhorn={sinkhorn!r} is not a normaliser. Escape: {SINKHORN_IMPLS}."
        )
        d, n = int(d_model), int(n_streams)
        self.d_model = d
        self.n_streams = n
        self.sinkhorn = sinkhorn
        total = n * d
        # burn: block.rs:81-97 — φ ~ N(0, 0.01), α = 0.01.
        # Eq. 7's projections: φ_pre/φ_post [nC, n], φ_res [nC, n²], drawn
        # N(0, 0.01) — `block.rs:81-94`. The paper states no init; the crate's
        # N(0, 0.01) is named as OURS (`mhc-2026-09-30.md` §2.1 last row).
        self.phi_pre = nn.Parameter(torch.randn(total, n, device=device) * 0.01)
        self.phi_post = nn.Parameter(torch.randn(total, n, device=device) * 0.01)
        self.phi_res = nn.Parameter(torch.randn(total, n * n, device=device) * 0.01)
        # Eq. 7's gating factors, one scalar each, init 0.01 — `block.rs:95-97`.
        self.alpha_pre = nn.Parameter(torch.full((1,), ALPHA_INIT, device=device))
        self.alpha_post = nn.Parameter(torch.full((1,), ALPHA_INIT, device=device))
        self.alpha_res = nn.Parameter(torch.full((1,), ALPHA_INIT, device=device))
        # Eq. 7's static mappings — `block.rs:86-90`. `b_res` starts at 0 (the
        # fidelity fix); `b_pre = 10` is σ⁻¹(0.5 + ~1/2) and `b_post = 0` is
        # σ⁻¹(0.5), i.e. `H_post ≈ 1` at init.
        self.b_pre = nn.Parameter(torch.full((n,), STATIC_INIT, device=device))
        self.b_post = nn.Parameter(torch.zeros(n, device=device))
        self.b_res = nn.Parameter(torch.zeros(n, n, device=device))

    def _x_norm(self, x_streams: torch.Tensor) -> torch.Tensor:
        """Eq. 7's `x̃' = RMSNorm(vec(x_l))` over the FULL lane vector — `block.rs:109-118`."""
        # burn: block.rs:109-118 — h / sqrt(mean(h²) + 1e-20), ε from App. A.1.
        b, t = x_streams.shape[:2]
        flat = x_streams.reshape(b, t, self.n_streams * self.d_model)
        # The lane norm is a REDUCTION: fp32 even when the lanes are stored
        # bf16 (`residual_dtype="bf16"`); `.float()` is the identity on the
        # fp32 path, so the OFF bits do not move. The result returns in the
        # module's OWN weight dtype (fp32 here; a bf16 test module keeps bf16)
        # because the hyper-network's next op is a matmul with those weights.
        flat = flat.float()
        return (flat / torch.sqrt(flat.pow(2).mean(-1, keepdim=True) + EPS)).to(self.phi_pre.dtype)

    def _pre_tilde(self, x_streams: torch.Tensor) -> torch.Tensor:
        """Eq. 7's `H̃_pre`, unconstrained — `block.rs:158-168`."""
        # burn: block.rs:158-168 — α_pre·(x̃'φ_pre) + b_pre.
        xn = self._x_norm(x_streams)
        return self.alpha_pre * (xn @ self.phi_pre) + self.b_pre

    def _post_res_tilde(self, x_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eq. 7's `H̃_post` and `H̃_res`, unconstrained — `block.rs:123-145`."""
        # burn: block.rs:123-145 — α·(x̃'φ) + b, the res row-major [n²] → [n,n].
        xn = self._x_norm(x_streams)
        post = self.alpha_post * (xn @ self.phi_post) + self.b_post
        n = self.n_streams
        res = self.alpha_res * (xn @ self.phi_res) + self.b_res.reshape(n * n)
        b, t = x_streams.shape[:2]
        return post, res.reshape(b, t, n, n)

    def hyper_mappings(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eq. 8's constrained mappings: `(H_pre, H_post, H_res)`.

        Shapes `[b, t, 1, n]`, `[b, t, 1, n]`, `[b, t, n, n]` — the crate's
        `hyper_mappings` shape (`block.rs:152-195`). `H_pre` is returned for the
        caller's body (`pre_mix` is the applied form); `forward` does not apply it.
        """
        pre = self._pre_tilde(x_streams)
        post, res = self._post_res_tilde(x_streams)
        # burn: block.rs:189-194 — σ / 2σ / Sinkhorn-Knopp (Eq. 8).
        return (
            torch.sigmoid(pre).unsqueeze(-2),
            2.0 * torch.sigmoid(post).unsqueeze(-2),
            sinkhorn_knopp(res, impl=self.sinkhorn),
        )

    def pre_mix(
        self,
        x_streams: torch.Tensor,
        *,
        convex: bool = False,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """`H_pre x_l`: Eq. 3's layer input, applied by the CALLER — `block.rs:189`.

        `H_pre` is `[1, n]`, so this aggregates the `n` full-width lanes into
        the body's `d`-wide input. Not part of [`forward`] on purpose
        (`lib.rs:43-48`). The two keyword arguments are the TRANSFORMER arm's
        identity lever; the loop calls the method with neither, so its bits do
        not move:

        * `convex=True` divides the gate by its own sum, making the read a
          convex combination of the lanes. On the transformer arm the K lanes
          are full-width replicas, so `Σ H_pre` would otherwise scale the
          block's input by K (the same "K copies re-sum" correction the
          readout applies); the normalised read reads the lane value itself.
        * `scale` multiplies the result. That arm stores `1/K` of the state in
          each lane for the mHC-ONLY readout (whose raw sum must be the state),
          so its read multiplies the convex mix back by K. `1.0` skips the
          multiply, keeping the fused/eager loop paths byte-for-byte.
        """
        # burn: block.rs:189 — H_pre = σ(H̃_pre), [B,T,1,n].
        h_pre = torch.sigmoid(self._pre_tilde(x_streams)).unsqueeze(-2)  # [b,t,1,n]
        if convex:
            h_pre = h_pre / h_pre.sum(-1, keepdim=True)
        out = (h_pre.transpose(-1, -2) * x_streams).sum(-2)
        return out if scale == 1.0 else out * scale

    def forward(
        self, x_streams: torch.Tensor, x_in: torch.Tensor, *, post_sum: float | None = None
    ) -> torch.Tensor:
        """Eq. 3: `x_{l+1} = H_res x_l + H_postᵀ F(H_pre x_l)` — `block.rs:197-227`.

        `x_streams [b, t, n, d]` are the lanes `x_l`; `x_in [b, t, d]` is the
        caller's branch output `F(H_pre x_l, W)`. Lane `j` receives
        `Σ_i H_res[j,i]·x_l[i] + H_post[j]·x_in`.

        `post_sum` is the TRANSFORMER arm's "K full-width deposits re-sum"
        correction: the K `H_post` weights are rescaled so they SUM to this
        value (its `<target>/Σ H_post`) per token. The loop passes nothing, so
        its Eq. 8 expression is byte-for-byte. Without it a random `φ_post`
        splits the replicated lanes on the first write and the arm's read no
        longer returns the state at step 0; the rescale keeps the deposit's
        total exact (its per-lane split — the learnable signal — still moves).
        """
        post, res = self._post_res_tilde(x_streams)
        h_post = (2.0 * torch.sigmoid(post)).unsqueeze(-2)  # [b, t, 1, n]
        if post_sum is not None:
            h_post = h_post * (post_sum / h_post.sum(-1, keepdim=True))
        h_res = sinkhorn_knopp(res, impl=self.sinkhorn)
        # burn: block.rs:213-226 — H_res x_l, then H_postᵀ F added per lane.
        # The MIX is a reduction against the fp32 Sinkhorn matrix, so the lanes
        # and the branch output are widened here; `.float()` is the identity on
        # the fp32 path (`residual_dtype="fp32"`), so its bits do not move.
        pdt = self.phi_pre.dtype
        xs = x_streams if x_streams.dtype is pdt else x_streams.to(pdt)
        xi = x_in if x_in.dtype is pdt else x_in.to(pdt)
        mixed = torch.einsum("btij,btjd->btid", h_res, xs)
        return mixed + h_post.transpose(-1, -2) * xi.unsqueeze(-2)
