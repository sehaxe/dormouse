"""Manifold-Constrained Hyper-Connections (mHC, arXiv:2512.24880) — Eq. 3/7/8/9.

The operator, for `n` residual lanes of width `d` (`x_l` flattened to
`vec(x_l) ∈ R^{n·d}`):

* **Eq. 7** — `x̃' = RMSNorm(vec(x_l))`, then the three first-order
  hyper-network outputs are linear projections with a small gate and a static
  bias: `H̃ = α·(x̃'φ) + b`.
* **Eq. 8** — `H_pre = σ(H̃_pre)`, `H_post = 2σ(H̃_post)`, `H_res =
  Sinkhorn-Knopp(H̃_res)`.
* **Eq. 9** — the Sinkhorn-Knopp projection onto the Birkhoff polytope, 20
  iterations of alternating row/column normalization (`t_max = 20`, App. A.1
  Table 5). **Log domain**: each normalization is `m -= logsumexp(m)`,
  shift-invariant and overflow-free, because the hyper-network output is
  unbounded.
* **Eq. 3** — `x_{l+1} = H_res x_l + H_postᵀ F(H_pre x_l)`.

Two readings of the paper are decided in favour of the PRINTED equations:

* **Eq. 7 has no `tanh`.** HC's Eq. 5 squashes its projections; mHC's Eq. 7
  (and its kernel section's Eq. 14-16) is the plain linear map above. The
  consequence is named: `φ_*` are unbounded, which is why the Sinkhorn input is
  unbounded and `sinkhorn_knopp` must run in the log domain.
* **`H_pre` feeds the layer's own `F`.** Eq. 3's `H_pre x_l` is
  `[1,n]·[n,C] = [1,C]`; [`MHCMix.pre_mix`] is that read-out and the loop's
  `forward` applies it (see "What is ours" below), not an external caller.

## Initialisation — derived, not chosen

The mHC paper states no bias init: App. A.1 Table 5 gives only `α = 0.01`,
`t_max = 20` and `ε = 1e-20`. The base Hyper-Connections paper
(arXiv:2409.19606) does state one, Eq. 14 (§2.3): the dynamic projections
`W_β, W_m, W_r` are initialised to **zero**, and the static maps to

    (0_{1×1}  B^k)   (0        1_{1×n})
    (A_m^k    A_r^k) = (e_{k mod n}  e_{n×n}),

so at step 0 the block **is** the Pre-Norm residual connection. We derive the
three biases of Eq. 7 from those maps in this module's own
sigmoid / 2-sigmoid / log-domain forms, for `K = n_streams`:

* `H_pre = σ(α_pre·(x̃'φ_pre) + b_pre)`. §2.3 sets `φ_pre = 0`, so
  `σ(b_pre) = 1/K` and **`b_pre = logit(1/K) = ln(1/(K-1))`** (`K=2 → 0`).
  The base paper's `A_m = e_{k mod n}` is a per-layer rotation; a
  weight-shared loop has no per-layer `k` (one block runs every iteration), so
  the rotation-averaged map `(1/K)Σ_k e_k = 1/K` is the depth-independent
  member of the family. On the paper's replicated lanes any convex combination
  returns the same state; `1/K` is the one that is also correct for a single
  shared block.
* `H_post = 2σ(α_post·(x̃'φ_post) + b_post)`. `φ_post = 0` and
  `B = 1_{1×K}` give `2σ(b_post) = 1`, so **`b_post = 0`**.
* `H_res = Sinkhorn(exp(α_res·mat(x̃'φ_res) + b_res))`. `φ_res = 0`. The
  printed `A_r = I` is the Sinkhorn fixed point `Sinkhorn(I) = I`, but in the
  log-domain form its input is `b_res = log I`: `0` on the diagonal and
  `-inf` off it. `exp(-inf) = 0` makes every off-diagonal logit — and its
  gradient — identically zero, which is the burn `10·I` frozen-arm failure
  (`|grad α_res| = 7.3e-9`, `docs/reviews/fidelity-audit-2026-10-02.md` F-F4)
  taken to its limit. The **finite** static map on the state the paper's init
  creates (`H^0` is `K` replicas of `h`, 2409.19606 §2.1) is the uniform
  matrix `J/K`: `Sinkhorn(exp(log(J/K))) = J/K`, and
  `(J/K)·(h,…,h) = (h,…,h)` exactly. **`b_res = log(J/K) = -ln K` is
  therefore the log of the finite fixed point**, not burn's chosen `0` (the
  Sinkhorn is shift-invariant, so `0` and `-ln K` are the same map; the
  parameter carries the derived value). The stiff-diagonal approximation
  (`10·I`) is rejected because it is the thing that froze the arm. The test
  `test_the_derived_biases_are_the_paper_maps` pins `σ(b_pre) = 1/K`,
  `2σ(b_post) = 1`, `H_res = J/K` and the bitwise identity on replicated lanes.
  The exact `I` is not reachable at finite bias; it is reported, not
  approximated.

`α = 0.01` **is** the mHC paper's (App. A.1 Table 5) and stays. `φ_* = 0` is
the base paper's §2.3, not burn's `N(0, 0.01)`; it also means the module's
construction draws no RNG, so `use_mhc=true` and `use_mhc=false` built under
one seed share every shared weight bit-for-bit and the A/B measures the
mechanism, not a different initialisation.

## What is ours, not the crate's

* **Full-width lanes, sum readout.** The crate's `h` is one `[B,T,D]` tensor
  sliced into `n` lanes of width `C = D/n`, and its caller (a `D`-wide body)
  cannot apply `H_pre`. The loop here carries `n` lanes of the FULL residual
  width (`C = D`), the paper's readout is their **sum** (2409.19606 §2.1), and
  the loop scales it by `1/n` because the paper's `n` width-`C` lanes re-sum to
  `C` while ours are `n` width-`D` copies: the same sum, normalised to the
  state width the body expects. Every lane is a complete residual state,
  `H_res` mixes them, `H_pre` is a sigmoid-weighted mix down to the body's
  `D`-wide input, and `H_post_j` weights the same branch output onto lane `j`.
  This is the only layout that composes a `D`-wide body with a `D`-wide
  stream.
* **`H_pre` is applied inside the loop's forward**, by
  [`MHCMix.pre_mix`], not left to an external caller. The crate allocated
  `phi_pre`/`alpha_pre`/`b_pre` and never read them; `H_pre` is 11 % of the
  measured effect in 2409.19606 Tab. 1 (`H_res` 81 %, `H_post` 7 % —
  `docs/reviews/mhc-2026-09-30.md` §3.3: *"`H_pre` adds 11 %"*), so wiring it
  is the point of the layout change.

## Gradient caveat — measured, not hidden

At this init every lane is an exact copy and the readout weights every lane
equally, so `∂loss/∂H_res` is row-constant; the Sinkhorn's row normalisation
annihilates exactly that direction. `alpha_res` and `phi_res` therefore sit at
the fp-noise floor on a full-model step (the gradient gate prints them), while
`phi_pre`, `phi_post`, `b_pre` and `b_post` train. The residual mixer **is**
differentiable — a lane-specific loss gives `phi_res > 1e-8` at this init
(`test_the_residual_mixer_is_live_when_the_loss_distinguishes_lanes`) — but the
model's symmetric readout does not. Breaking that symmetry needs a
lane-specific readout, which is a mechanism decision for the owner, not a bias
value; the paper's sum readout is kept here.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["ALPHA_INIT", "B_RES_INIT", "EPS", "MHCMix", "SINKHORN_ITERS", "sinkhorn_knopp"]

#: Sinkhorn-Knopp iterations (App. A.1: `t_max = 20`) — 2512.24880.
SINKHORN_ITERS = 20
#: Gating-factor init (App. A.1 Table 5: `alpha = 0.01`) — 2512.24880.
ALPHA_INIT = 0.01
#: The finite static residual map: `log(J/K) = -ln K` on every entry, whose
#: Sinkhorn is `J/K` — the identity on the paper's replicated lanes. The
#: literal `log I` is `-inf` off-diagonal and has zero gradient (see the module
#: docstring). Stated for the shipped `K = 2`; `MHCMix` uses `-ln n_streams`.
B_RES_INIT = -math.log(2.0)
#: RMSNorm's ε (App. A.1 Table 5, "Layer Norm eps = 1e-20") — 2512.24880.
EPS = 1e-20


def sinkhorn_knopp(logits: torch.Tensor, iters: int = SINKHORN_ITERS) -> torch.Tensor:
    """Eq. 9: the log-domain Sinkhorn-Knopp projection onto the Birkhoff polytope.

    `logits [..., n, n]` → `M⁽⁰⁾ = exp(logits)`, then `iters` alternating
    row/column normalizations (`T_r(T_c(·))`). Each normalization is
    `m -= logsumexp(m)`, which is shift-invariant and never overflows — the
    naive exp-then-divide form breaks on the hyper-network's unbounded output.
    The fixed point `Sinkhorn(I) = I` is why the base paper's `A_r = I` is the
    identity init; `logits = 0` gives the finite fixed point `J/n` (uniform).
    """
    m = logits
    for _ in range(iters):
        # Row then column, both `m -= logsumexp(m)`; the last pass is the
        # column one, so columns are the exact-to-fp direction.
        m = m - torch.logsumexp(m, dim=-1, keepdim=True)
        m = m - torch.logsumexp(m, dim=-2, keepdim=True)
    return m.exp()


class MHCMix(nn.Module):
    """Eq. 3's residual mixing + Eq. 7's hyper-network, for `n` full-width lanes.

    `x_streams [b, t, n, d]` is the residual stream (`d = C` here — see the
    module docstring's layout note); `x_in [b, t, d]` is the caller's branch
    output. [`forward`] implements Eq. 3's `H_res x_l + H_postᵀ F` terms;
    [`pre_mix`] is Eq. 3's `H_pre x_l` and the loop applies it before the body.
    The loop reads the lanes out by summing them (2409.19606 §2.1).
    """

    def __init__(
        self, d_model: int, n_streams: int = 2, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__()
        assert n_streams >= 2, (
            "n_streams (the expansion rate n) must be >= 2: with one lane there is "
            "no width to mix, and 2409.19606 Tab. 1 measures n=1 below the residual "
            "baseline it is meant to improve"
        )
        d, n = int(d_model), int(n_streams)
        self.d_model = d
        self.n_streams = n
        total = n * d
        # Eq. 7's projections: base paper §2.3 (arXiv:2409.19606) initialises
        # the dynamic W_β/W_m/W_r to ZERO, so the dynamic half is off at step 0
        # and the static biases alone define the maps. Burn's N(0, 0.01) is not
        # used; zero-init also consumes no RNG (see the module docstring).
        self.phi_pre = nn.Parameter(torch.zeros(total, n, device=device))
        self.phi_post = nn.Parameter(torch.zeros(total, n, device=device))
        self.phi_res = nn.Parameter(torch.zeros(total, n * n, device=device))
        # Eq. 7's gating factors, init 0.01 — 2512.24880 App. A.1 Table 5.
        self.alpha_pre = nn.Parameter(torch.full((1,), ALPHA_INIT, device=device))
        self.alpha_post = nn.Parameter(torch.full((1,), ALPHA_INIT, device=device))
        self.alpha_res = nn.Parameter(torch.full((1,), ALPHA_INIT, device=device))
        # Eq. 7's static biases, derived from 2409.19606 Eq. 14 (§2.3):
        #   σ(b_pre) = 1/K            -> b_pre = ln(1/(K-1))  (K=2 -> 0)
        #   2σ(b_post) = 1            -> b_post = 0
        #   Sinkhorn(exp(b_res)) = I  -> b_res = log I; its finite fixed point on
        #                                the replicated lanes is J/K, i.e. 0.
        self.b_pre = nn.Parameter(torch.full((n,), math.log(1.0 / (n - 1)), device=device))
        self.b_post = nn.Parameter(torch.zeros(n, device=device))
        # `log(J/K) = -ln K`; the Sinkhorn is invariant to a constant shift, so
        # this and 0 are the same map — the derived value is pinned instead of
        # burn's chosen 0, and `H_res = J/K` is the finite identity on copies.
        self.b_res = nn.Parameter(torch.full((n, n), -math.log(n), device=device))

    def _x_norm(self, x_streams: torch.Tensor) -> torch.Tensor:
        """Eq. 7's `x̃' = RMSNorm(vec(x_l))` over the FULL lane vector."""
        b, t = x_streams.shape[:2]
        flat = x_streams.reshape(b, t, self.n_streams * self.d_model)
        return flat / torch.sqrt(flat.pow(2).mean(-1, keepdim=True) + EPS)

    def _pre_tilde(self, x_streams: torch.Tensor) -> torch.Tensor:
        """Eq. 7's `H̃_pre`, unconstrained."""
        xn = self._x_norm(x_streams)
        return self.alpha_pre * (xn @ self.phi_pre) + self.b_pre

    def _post_res_tilde(self, x_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eq. 7's `H̃_post` and `H̃_res`, unconstrained."""
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

        Shapes `[b, t, 1, n]`, `[b, t, 1, n]`, `[b, t, n, n]`. `H_pre` is
        returned for inspection; the applied form is [`pre_mix`], which the
        loop's forward calls.
        """
        pre = self._pre_tilde(x_streams)
        post, res = self._post_res_tilde(x_streams)
        return (
            torch.sigmoid(pre).unsqueeze(-2),
            2.0 * torch.sigmoid(post).unsqueeze(-2),
            sinkhorn_knopp(res),
        )

    def pre_mix(self, x_streams: torch.Tensor) -> torch.Tensor:
        """`H_pre x_l`: Eq. 3's layer input, applied inside the loop's forward.

        `H_pre` is `[1, n]`, so this aggregates the `n` full-width lanes into
        the body's `d`-wide input. The crate allocated `phi_pre` but left
        `H_pre` to its caller; it is 11 % of the measured effect
        (`docs/reviews/mhc-2026-09-30.md` §3.3).
        """
        h_pre = torch.sigmoid(self._pre_tilde(x_streams)).unsqueeze(-2)  # [b,t,1,n]
        return (h_pre.transpose(-1, -2) * x_streams).sum(-2)

    def forward(self, x_streams: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
        """Eq. 3: `x_{l+1} = H_res x_l + H_postᵀ F(H_pre x_l)`.

        `x_streams [b, t, n, d]` are the lanes `x_l`; `x_in [b, t, d]` is the
        caller's branch output `F(H_pre x_l, W)` (the caller applies
        [`pre_mix`]). Lane `j` receives
        `Σ_i H_res[j,i]·x_l[i] + H_post[j]·x_in`.
        """
        post, res = self._post_res_tilde(x_streams)
        h_post = (2.0 * torch.sigmoid(post)).unsqueeze(-2)  # [b, t, 1, n]
        h_res = sinkhorn_knopp(res)
        mixed = torch.einsum("btij,btjd->btid", h_res, x_streams)
        return mixed + h_post.transpose(-1, -2) * x_in.unsqueeze(-2)
