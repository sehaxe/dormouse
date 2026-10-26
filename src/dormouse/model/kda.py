"""KDA (gated-delta) attention, with FLA as an optional dependency.

The port's rule for attention is the readiness plan §7's: **FLA is a
dependency, and its absence is COUNTED, not silent.**

* The real arm is `fla.ops.kda.chunk_kda` (FLA 0.5.2) — the chunked
  **per-channel** gated-delta kernel, the op that takes the port's own
  `g [b, t, h, head_dim]` log-decay. `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
  is the SCALAR-decay variant of the same recurrence, and a 4-D `g` handed to it
  runs and returns wrong numbers with no error (measured: 4.6e1 × the output RMS
  off the recurrence, while FLA's own naive reference matched at 1.5e-8). That is
  why this file names the op, not just "fla".
* Presence is proven by a **tiny warmup call at first use**, not by `import fla`
  succeeding: an installed-but-broken kernel (wrong arch, missing extension,
  sm_120 refusal) must fall back the same way a missing one does. The proof is
  per device TYPE — a CPU tensor on a CUDA box gets the fallback too, because
  FLA's kernels are CUDA-only and the CPU test path is the SDPA arm.
* The fallback is causal `scaled_dot_product_attention`, fp32-legal on CPU, and
  it increments a counter, so the log line reads `kda=sdpa,n=<calls>`.

WHAT RUNS OUTSIDE THE FUSED ARM, and why (COUNTED, not hidden):

* Our decay and write strength: `g = logsigmoid(g_proj(x))` (per channel) and
  `beta = sigmoid(beta_proj(x))` (per head) are computed in torch and passed in.
  FLA's in-kernel gate (`use_gate_in_kernel=True`) wants `A_log [H]` +
  `dt_bias [H·K]` parameters the port's set does not carry — the preset's count
  is pinned at `q,k,v,g = 4·d²` plus `beta = d·h` by `tests/test_param_count.py`,
  and what happened to be installed must not change the model's parameter count.
* The q/k L2 norm IS inside the kernel (`use_qk_l2norm_in_kernel=True`), which is
  where burn-kda has it too (K3: `q,k = L2Norm(ShortConv(Linear(x)))`). The port's
  projections are raw, and without that norm the gated-delta recurrence is
  numerically explosive (measured: 1e17 by t=64 at unit-variance q/k). **The
  SDPA fallback applies the same `F.normalize` explicitly**, so both arms
  implement K3's normalization and neither is anchored on inputs the other
  never saw.
* NOT in this model at all, so in neither arm: burn-kda's short convolutions, its
  output `Sigmoid(W_g x) ⊙ RMSNorm(o)` gate and its `o_proj`. Those are
  parameters the port does not have (the loop block norms BEFORE the layer and
  owns `out_proj` as its readout) — absent from the set, not inexpressible in FLA.

NUMERIC CONTRACT (the plan's rule): **the chunked kernel is NOT
tensor-comparable to burn-kda's KDA arm, and no cross-substrate bitwise claim exists
or can exist.** At the port's test shape (b1, h8, d_head 64, seq 256) it sits
1.3e-2 (fp32) / 4.9e-2 (bf16) of the output RMS away from the exact recurrence;
the spike measured 0.3 % / 1.8 % mean-relative at its own shape. The anchor is a
naive recurrent reference (`tests/test_kda_gpu.py`), which FLA's own
`naive_recurrent_kda` agrees with to 1.5e-8 in fp32. The bound asserted on it is
a WIRING gate (a transposed head axis is O(1) off), not a numerics certificate.

THE CARRIED STATE (`forward_state`) — read from the INSTALLED FLA source, not
from a doc page (2026-10-17, `.venv/.../fla/ops/kda/`):

* **`chunk_kda` itself takes `initial_state` and returns `final_state`** —
  `chunk.py:185` / `chunk.py:186`, `[N, HV, K, V]` (`K, V` = head_dim) and
  **fp32 on both sides**: the forward allocates it fp32
  (`common/chunk_delta_h.py:707`) and the entry point asserts fp32 on the way
  back in (`chunk.py:383`). The `fused_recurrent` variant has `h0`/`ht` too, but
  the chunked one — the arm this model runs — is not the poorer relation here,
  so the port uses the op it already calls.
* **The gradient rides the same two arguments**: `ChunkKDAFunction.backward`
  takes `dht` (`chunk.py:127`) and returns `dh0` (`chunk.py:135`, `:172`), and
  the boundary state is `save_for_backward`ed (`chunk.py:106`). So a segment's
  entering state is a real graph edge in BOTH directions — StateFlow's
  boundary state + gradient propagation (arXiv:2608.06838) — and nothing is
  detached or truncated.
* The state is `O(1)` in the sequence length (2·b·H·head_dim² fp32 = 393 KB at
  the shipped `small`), which is the point of carrying it rather than a KV cache.
* **The SDPA fallback has no recurrent state to carry** — it is not a
  recurrence (its own docstring says the delta rule and `beta` do not run
  there). Its carry is the EXACT causal-attention context (the K/V of the
  preceding segments, `_sdpa_path`), so a segmented forward on the fallback is
  still the same function as the one-shot one — which is what makes the CPU
  gates in `tests/test_longctx.py` mean something. It is linear in the
  sequence length, and the arm is already COUNTED (`kda=sdpa`) as a
  degradation, so nothing here pretends to be the fast path.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters, loud
from .tsct import dense_qat_kind, qat_linear

#: Cached warmup verdict per device TYPE: `{"cuda": (ok, why), "cpu": (ok, why)}`.
_FLA: dict[str, tuple[bool, str]] = {}

#: FLA's `disable_recompute` (chunk.py:50, documented at :258): when True the
#: kernel SAVES the backward intermediates instead of recomputing them —
#: "beneficial for training small models at the cost of increased memory
#: usage" (their docstring). One process-wide flag, set from
#: `TrainConfig.kda_disable_recompute` beside the tf32 knob; a fresh process is
#: False = FLA's own default = the behaviour every number on record ran with.
_DISABLE_RECOMPUTE = False


def set_disable_recompute(flag: bool) -> None:
    """`TrainConfig.kda_disable_recompute`'s ONE entry point (`train/loop.py`)."""
    global _DISABLE_RECOMPUTE
    _DISABLE_RECOMPUTE = bool(flag)


def disable_recompute() -> bool:
    """The live flag (for a gate or a bench)."""
    return _DISABLE_RECOMPUTE


def fla_status(device: torch.device | str = "cuda", recheck: bool = False) -> tuple[bool, str]:
    """Is FLA usable on THIS device type? Validated by a tiny call, once per process."""
    key = torch.device(device).type
    if key in _FLA and not recheck:
        return _FLA[key]
    if key != "cuda" or not torch.cuda.is_available():
        # Legal, not a defect: the CPU test path is the SDPA fallback.
        verdict = (False, f"device {key}: FLA kernels are CUDA-only")
    else:
        try:
            from fla.ops.kda import chunk_kda  # noqa: F401
        except Exception as e:  # ImportError, but also a broken extension
            verdict = (False, f"import fla failed: {type(e).__name__}: {e}")
        else:
            try:
                _warmup()
            except Exception as e:
                verdict = (False, f"warmup call failed: {type(e).__name__}: {e}")
            else:
                verdict = (True, "warmup ok")
    _FLA[key] = verdict
    return verdict


def _warmup() -> None:
    """One tiny chunked-KDA call at (b=1, t=8, h=1, d=16) — the presence proof.

    Same op, same flags and same input layout as the production path, so what it
    proves is the arm, not the import.
    """
    from fla.ops.kda import chunk_kda

    dev = torch.device("cuda")
    q = torch.randn(1, 8, 1, 16, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 1, 16, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 8, 1, 16, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, 8, 1, 16, device=dev, dtype=torch.float32)
    beta = torch.rand(1, 8, 1, device=dev, dtype=torch.bfloat16)
    out, _ = chunk_kda(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    assert out.shape == q.shape, f"warmup shape {tuple(out.shape)} != {tuple(q.shape)}"
    if not torch.isfinite(out).all():
        raise RuntimeError("warmup produced non-finite output")


def _recurrent_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ONE token of KDA Eq 1 in torch, fp32 — the CPU decode step and the kernel's anchor.

    `S = Diag(exp(g))·S + beta·k·(v − kᵀ·Diag(exp(g))·S)`, `o = scale·qᵀ·S` — the
    same recurrence `tests/test_kda_gpu.py::_recurrent` walks a whole sequence
    with, one position at a time. The q/k L2 norm is applied here the way the
    fused arm applies it IN the kernel (`q / √(Σq² + 1e-6)`,
    `use_qk_l2norm_in_kernel=True`): K3 normalizes q/k and raw q/k explode (module
    docstring), so the two arms must not be anchored on different inputs.

    `q,k,v,g [b, 1, h, hd]`, `beta [b, 1, h]`, `state [b, h, hd, hd] | None` (None
    = position 0) → (`o [b, 1, h, hd]`, the state leaving this token).
    """
    b, _, h, hd = q.shape
    qf, kf, vf = q.float(), k.float(), v.float()
    qf = qf * torch.rsqrt(qf.pow(2).sum(-1, keepdim=True) + 1e-6)
    kf = kf * torch.rsqrt(kf.pow(2).sum(-1, keepdim=True) + 1e-6)
    s = torch.zeros(b, h, hd, vf.shape[-1], dtype=torch.float32, device=q.device)
    if state is not None:
        s = s + state.float()
    s = s * g[:, 0].float().exp()[..., None]
    ks = torch.einsum("bhk,bhkv->bhv", kf[:, 0], s)
    s = s + kf[:, 0].unsqueeze(-1) * (beta[:, 0].float()[..., None] * (vf[:, 0] - ks)).unsqueeze(-2)
    o = torch.einsum("bhk,bhkv->bhv", qf[:, 0], s) * scale
    return o.unsqueeze(1), s


class KDAAttention(nn.Module):
    """`d → d` gated-delta attention: `q,k,v,g` projections plus per-head `beta`.

    The parameter set is the burn side's (KDA's own shape, `schema.rs`'s
    "q,k,v,g per-channel + beta" = `4·d² + d·n_heads`), and it is built the
    same way whatever device or library is present — the preset's parameter
    count must not depend on what happened to be installed.

    Both arms return `[b, t, d]` in the input's dtype; the fused arm runs its
    matmuls at `compute_dtype` (the port's bf16 plan: bf16 inside the arm, fp32
    masters and fp32 state between arms), the SDPA fallback runs at whatever the
    projections produced.
    """

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        assert h * cfg.head_dim == d, "validate() pins n_heads*head_dim == d_model"
        self.n_heads = h
        self.head_dim = cfg.head_dim
        #: GQA (`n_kv_heads`): the k/v PROJECTION width, expanded back to `h`
        #: immediately before the kernel. `0` resolves to `h` = MHA, so the OFF
        #: path is the four `d×d` projections every number on record ran.
        self.kv_heads = cfg.n_kv_heads or h
        if self.kv_heads < 0 or self.kv_heads > h:
            loud(
                f"n_kv_heads={cfg.n_kv_heads}: GQA needs 0 (off/MHA) or a count "
                f"in 1..n_heads={h}; there cannot be more KV heads than queries."
            )
        if h % self.kv_heads:
            loud(
                f"n_kv_heads={self.kv_heads} does not divide n_heads={h}: the "
                f"expansion is `repeat_interleave(n_heads // n_kv_heads)`, so a "
                f"ragged group has no valid mapping. Escape: a divisor of n_heads."
            )
        self.kv_width = self.kv_heads * cfg.head_dim
        #: MLA-style shared latent (`attn_latent_rank`): ONE `d → r` projection
        #: feeds the four `r → out` projections. `0` = off, four `d×d` linears.
        self.attn_latent_rank = cfg.attn_latent_rank
        if self.attn_latent_rank < 0:
            loud(f"attn_latent_rank={self.attn_latent_rank}: 0 is OFF, so a rank is >= 1.")
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        #: The DENSE path's switch (`tsct.dense_qat_project`, the SAME projector
        #: as the factor rung): `q/k/v/g/beta` are plain `nn.Linear`s, so their
        #: weights are read through the annealed STE. `dense_qat_alpha` is the
        #: SAME alpha the factor ramp sets (`set_factor_qat_alpha`). All five
        #: sites are ATTENTION, so the default `dense_qat_skip_attn` keeps them
        #: bf16 (the BitNet rule; `tsct.DENSE_QAT_ATTN_SITES` owns the list).
        #: The latent down/up are the same kind of dense site in the same arm,
        #: so they inherit the same kind.
        self.dense_qat = dense_qat_kind(
            cfg,
            "loop.attn.q_proj",
            "loop.attn.k_proj",
            "loop.attn.v_proj",
            "loop.attn.g_proj",
            "loop.attn.beta_proj",
        )
        self.dense_qat_alpha = cfg.factor_qat_alpha
        #: `None` when the latent is off — no module, no RNG draw, no state key.
        self.latent_down = (
            nn.Linear(d, self.attn_latent_rank, bias=False, device=device)
            if self.attn_latent_rank
            else None
        )
        # The four projections read `c` (the latent) when it is on, `x` when it
        # is off. Order (q, k, v, g, beta) is the old `setattr` loop's draw
        # order, so the OFF path's init is bit-identical.
        in_features = self.attn_latent_rank or d
        self.q_proj = nn.Linear(in_features, d, bias=False, device=device)
        self.k_proj = nn.Linear(in_features, self.kv_width, bias=False, device=device)
        self.v_proj = nn.Linear(in_features, self.kv_width, bias=False, device=device)
        self.g_proj = nn.Linear(in_features, d, bias=False, device=device)
        self.beta_proj = nn.Linear(d, h, bias=False, device=device)

    def _latent_base(self, x: torch.Tensor, dq: tuple[str, float]) -> torch.Tensor:
        """`c = x @ W_down` (`[b,t,d] → [b,t,r]`), or `x` itself when off.

        The OFF branch returns `x` unmodified — no matmul, no cast, no copy — so
        the dense path is the call it was.
        """
        if self.latent_down is None:
            return x
        return qat_linear(self.latent_down, x, *dq)

    def _kv(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        """`[b,t,kv_heads,hd]` → `[b,t,n_heads,hd]`, immediately before the kernel.

        `repeat_interleave` is GQA's group broadcast (Ainslie et al. 2023): each
        KV head serves `n_heads // n_kv_heads` consecutive query heads. When the
        group is 1 (MHA / `n_kv_heads = 0`) the tensor is returned as-is, so the
        OFF path allocates nothing and the bits do not move.
        """
        y = x.reshape(b, t, self.kv_heads, self.head_dim)
        if self.kv_heads == self.n_heads:
            return y
        return y.repeat_interleave(self.n_heads // self.kv_heads, dim=2)

    def projections(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x [b, t, d]` → `q, k, v [b, t, h, hd]`, `g [b, t, h, hd]` log decay, `beta [b, t, h]`.

        The `[B, T, H, ·]` layout is FLA 0.5.2's (`head_first` was removed from
        these ops) and it is also the port's head split, so nothing is
        transposed on the way into the kernel. `g` is log-space and fp32, which
        is what both the kernel and the recurrence reference read.

        `attn_latent_rank` computes `c` ONCE and reads the four projections off
        it; `n_kv_heads` leaves k/v at the KV width and expands them here, so
        every consumer — this arm's kernel, the SDPA fallback, the decode step —
        still sees `[b, t, h, hd]` for all four. Both OFF by default and both
        branches collapse to the dense call.
        """
        b, t, _ = x.shape
        h, hd = self.n_heads, self.head_dim
        dq = (self.dense_qat, self.dense_qat_alpha)
        c = self._latent_base(x, dq)
        q = qat_linear(self.q_proj, c, *dq).reshape(b, t, h, hd)
        k = self._kv(qat_linear(self.k_proj, c, *dq), b, t)
        v = self._kv(qat_linear(self.v_proj, c, *dq), b, t)
        g = F.logsigmoid(qat_linear(self.g_proj, c, *dq).float()).reshape(b, t, h, hd)
        beta = torch.sigmoid(qat_linear(self.beta_proj, x, *dq).float())  # [b, t, h]
        return q, k, v, g, beta

    def _heads(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        return x.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [b, t, d]` (block-body input) → `[b, t, d]`.

        The ONE-SHOT contract: no state in, none out. Every number on record ran
        through this call; the segmented path uses `forward_state` instead.
        """
        return self.forward_state(x, None, counters=counters)[0]

    def forward_state(
        self,
        x: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        counters: Counters | None = None,
        *,
        carry: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None]:
        """`x [b,t,d]` + the state entering this window → (`[b,t,d]`, the state leaving it).

        `carry=False` (the default) is `forward`'s call and returns `None` for
        the state; `carry=True` is the segmented training path's call and
        returns the state the NEXT window starts from. `state=None` means "this
        window starts at position 0" — segment 0 of a segmented run, and the
        one-shot call, take the same branch, which is why a single full-length
        segment reproduces the one-shot numbers bit for bit
        (`tests/test_longctx.py`).

        The state's SHAPE is per arm and both are documented at `_fla_path` /
        `_sdpa_path`: the fused arm carries the `[b, H, head_dim, head_dim]`
        fp32 gated-delta state, the SDPA fallback carries its K/V context.
        """
        b, t, d = x.shape
        counters = Counters() if counters is None else counters
        if state is not None and not carry:
            loud(
                "kda.forward_state got a non-None `state` with `carry=False`: the "
                "one-shot call drops the entering state, so the answer would be for a "
                "different sequence. Escape: carry=True (the segmented path) or "
                "state=None (this window starts at position 0)."
            )
        ok, why = fla_status(x.device)
        if ok:
            counters.count("kda_fla")
            return self._fla_path(x, state, carry=carry)
        counters.count("kda_sdpa")
        # The reason is attached to the counter object, not printed per forward:
        # `log line` prints one `kda=` field, and the reason belongs there.
        counters.why = why  # type: ignore[attr-defined]
        if not carry:
            return self._sdpa_path(x, b, t, d), None
        return self._sdpa_path_carry(x, b, t, d, state)

    def step_state(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        counters: Counters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`x [b, t, d]` + the state entering the block → (`[b, t, d]`, the state leaving it).

        The DECODE step: the same gated-delta recurrence `forward_state` computes
        over a window, one position at a time, with the state CARRIED instead of
        recomputed — `O(1)` per byte instead of `O(window)`. `t = 1` is the
        per-byte decode step; `t > 1` is the block verify (`decode.recurrent_decode`
        `decode_multibyte`), and the two MUST agree per token: the recurrence is
        sequential (`S_p` reads `S_{p-1}`), so a block is the same arithmetic in
        the same order, not a different reduction. (The projections are still ONE
        batched GEMM for the block, so the block and the one-byte loop differ in
        the last bits; `decode.py`'s docstring states that and its GPU gate pins
        the greedy bytes.)

        * FLA live on this device (a CUDA box whose warmup passed): the vLLM-derived
          decode kernel `fused_recurrent_kda` (FLA 0.5.2, `fused_recurrent.py:336`)
          — `initial_state` in, `final_state` out, `[b, H, head_dim, head_dim]`
          fp32 on both sides, i.e. the same state `chunk_kda` returns from
          `forward_state`. **It ships no backward**, so it is taken only where
          none is needed (the decode path runs under `no_grad`).
        * No FLA (CPU, or a card whose warmup fails): the exact torch recurrence
          (`_recurrent_step`), fp32, COUNTED `kda_step_torch` — a python loop over
          head_dim² is a real degradation and the counter is how the log says so.
          It is the same recurrence the fused kernel computes, so the decode arm
          does not change between the two; only its speed does.

        The SDPA fallback is deliberately NOT this path: its carried state is a
        `(k, v)` context that grows with the sequence (`_sdpa_path_carry`), which
        is `O(n)` per byte and would defeat the point of a recurrent decoder.
        `state=None` means position 0, exactly as `forward_state`.
        """
        b, t, d = x.shape
        assert t >= 1, f"step_state needs at least one position, got t={t}"
        counters = Counters() if counters is None else counters
        q, k, v, g, beta = self.projections(x)
        ok, why = fla_status(x.device)
        if ok and not torch.is_grad_enabled():
            from fla.ops.kda import fused_recurrent_kda

            dt = self.compute_dtype
            if dt != torch.float32:
                q, k, v, beta = q.to(dt), k.to(dt), v.to(dt), beta.to(dt)
            # ONE position per kernel call, even for a block: `fused_recurrent_kda`
            # with `T > 1` measured 1e-4/1e-3 from the same tokens stepped one at a
            # time (fp32 state, this card), and the block verify's whole contract is
            # "the same bytes the one-byte loop would emit". The projections above
            # are still shared by the block; only the recurrence is unrolled.
            outs: list[torch.Tensor] = []
            ht = state
            for p in range(t):
                o, ht = fused_recurrent_kda(
                    q[:, p : p + 1],
                    k[:, p : p + 1],
                    v[:, p : p + 1],
                    g=g[:, p : p + 1],
                    beta=beta[:, p : p + 1],
                    scale=self.scale,
                    initial_state=ht,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                outs.append(o)
            counters.count("kda_step", t)
            out = outs[0] if t == 1 else torch.cat(outs, dim=1)
            assert out.shape == (b, t, self.n_heads, self.head_dim), f"step {tuple(out.shape)}"
            assert ht is not None and ht.dtype == torch.float32, "the step state is fp32"
            want = (b, self.n_heads, self.head_dim, self.head_dim)
            assert ht.shape == want, f"step state {tuple(ht.shape)} != {want}"
            return out.reshape(b, t, d), ht
        counters.count("kda_step_torch", t)
        counters.why = (  # type: ignore[attr-defined]
            "grad enabled: fused_recurrent_kda ships no backward, so the torch "
            "recurrence ran"
            if ok
            else why
        )
        # The reference recurrence is written for ONE position; a block is the
        # same loop unrolled, which is what keeps `t=1` and `t=k` identical.
        # The multi-position form is what the decode graph steps.
        outs: list[torch.Tensor] = []
        ht = state
        for p in range(t):
            o, ht = _recurrent_step(
                q[:, p : p + 1],
                k[:, p : p + 1],
                v[:, p : p + 1],
                g[:, p : p + 1],
                beta[:, p : p + 1],
                ht,
                self.scale,
            )
            outs.append(o)
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return o.reshape(b, t, d), ht

    def _fla_path(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        carry: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The real arm: FLA 0.5.2's chunked per-channel gated-delta kernel.

        `carry=False` is the call that produced every number on record — not one
        argument of it differs. `carry=True` adds `initial_state` /
        `output_final_state` (both real arguments of `chunk_kda`, `chunk.py:185-186`;
        see the module docstring for the gradient side), so segment s's state is
        the graph edge into segment s+1.
        """
        if state is not None and not carry:
            loud(
                "kda._fla_path got a non-None `state` with `carry=False`: the "
                "one-shot arm neither passes `initial_state` nor returns a boundary "
                "state, so the state would be silently dropped. Escape: carry=True, "
                "or state=None."
            )
        from fla.ops.kda import chunk_kda

        b, t, d = x.shape
        dt = self.compute_dtype
        q, k, v, g, beta = self.projections(x)
        if dt != torch.float32:
            q, k, v, beta = q.to(dt), k.to(dt), v.to(dt), beta.to(dt)
        if not carry:
            out, _ = chunk_kda(
                q,
                k,
                v,
                g=g,
                beta=beta,
                scale=self.scale,
                use_qk_l2norm_in_kernel=True,  # burn-kda K3: L2Norm on q/k, in kernel
                disable_recompute=_DISABLE_RECOMPUTE,
            )
            assert out.shape == (b, t, self.n_heads, self.head_dim), f"chunk_kda {tuple(out.shape)}"
            return out.reshape(b, t, d), None
        out, ht = chunk_kda(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=self.scale,
            initial_state=state,  # None = position 0 (segment 0)
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            disable_recompute=_DISABLE_RECOMPUTE,
        )
        assert out.shape == (b, t, self.n_heads, self.head_dim), f"chunk_kda {tuple(out.shape)}"
        assert ht is not None and ht.dtype == torch.float32, "the boundary state is fp32"
        want = (b, self.n_heads, self.head_dim, self.head_dim)
        assert ht.shape == want, f"boundary state {tuple(ht.shape)} != {want}"
        return out.reshape(b, t, d), ht

    def _sdpa_path(self, x: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        """COUNTED fallback: causal SDPA over L2-normalized q/k, g as an output gate.

        The ONE-SHOT call — signature and return value exactly as they were, so
        the arm's own gates and every recorded number keep their call. The
        segmented path's call is `_sdpa_path_carry`, which is the same function
        with the context of the preceding windows (see there).
        """
        return self._sdpa_path_carry(x, b, t, d, None)[0]

    def _sdpa_path_carry(
        self,
        x: torch.Tensor,
        b: int,
        t: int,
        d: int,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """`_sdpa_path` + the state: (`[b,t,d]`, the `(k, v)` context leaving this window).

        Counted because it is a DEGRADATION, not a different name for the same
        function: the delta-rule recurrence and `beta` do not run here. The
        counter on the log line is how a reader tells which arm produced a loss.

        q/k are L2-normalized BEFORE the attention, mirroring the fused arm's
        `use_qk_l2norm_in_kernel=True`: burn-kda K3 normalizes q/k, and an
        un-normalized fallback would make the two arms two different models
        (raw q/k are 1e17-explosive, see the module docs).

        The CARRIED state is the exact causal-attention context `(k, v)` of the
        windows before this one, in the `[b, h, t, hd]` head layout — this arm
        has no recurrent state to carry (it is not a recurrence), and the
        context is what makes the segmented forward the SAME function as the
        one-shot one. It is linear in the sequence length, unlike the fused
        arm's `O(1)` state, and the arm is already COUNTED as the degradation it
        is. `state=None` (position 0) takes the `is_causal` branch, so segment 0
        is bit-identical to the one-shot call's own kernel.
        """
        dq = (self.dense_qat, self.dense_qat_alpha)
        c = self._latent_base(x, dq)
        q = F.normalize(self._heads(qat_linear(self.q_proj, c, *dq), b, t), dim=-1)
        k = F.normalize(self._heads(self._kv(qat_linear(self.k_proj, c, *dq), b, t), b, t), dim=-1)
        v = self._heads(self._kv(qat_linear(self.v_proj, c, *dq), b, t), b, t)
        if state is None:
            kk, vv = k, v
            y = F.scaled_dot_product_attention(q * self.scale, kk, vv, is_causal=True)
        else:
            kp, vp = state
            tp = int(kp.shape[2])
            assert tp > 0, "an empty context is `state=None`, not a zero-width cache"
            kk, vv = torch.cat([kp, k], dim=2), torch.cat([vp, v], dim=2)
            # Query i of this window is global position tp+i, so it reads keys
            # 0..tp+i. `is_causal=True` cannot express that: with L_q != L_k torch
            # aligns the mask to the top-left, which would read keys 0..i.
            cols = torch.arange(tp + t, device=x.device).view(1, -1)
            rows = torch.arange(t, device=x.device).view(-1, 1) + tp
            y = F.scaled_dot_product_attention(q * self.scale, kk, vv, attn_mask=cols <= rows)
        gate = torch.sigmoid(qat_linear(self.g_proj, c, *dq))  # [b, t, d]
        y = y.transpose(1, 2).reshape(b, t, d) * gate
        # The state returned is the ACCUMULATED context, not this window's own
        # keys: a third segment would otherwise read only the second one's keys
        # and lose the first (the gate in `test_segment_zero_moves_segment_one`
        # runs two segments and cannot see that, `test_segmented_logits…[16]`
        # runs four and does).
        return y, (kk, vv)
