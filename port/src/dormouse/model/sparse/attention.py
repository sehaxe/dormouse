"""`BlockSparseAttention` — the MSA arm: a sparse second attention arm.

The KDA linear arm REMAINS THE DEFAULT (`use_msa = false`): this is a new arm
behind a config flag, not a replacement. The flag is read off the config object
(`getattr(cfg, "use_msa", False)`) so the arm lands as new files only; adding
the `ModelConfig` field is a follow-up for the commit that wires it into
`LoopBlock`. When the flag is off the module builds
NO parameters and its forward returns `None` with a COUNTED `msa_off` — the
model's bytes are the same as before the arm existed.

**KERNEL CHOICE** (the plan's own open verdict, "no published rank/k sweep"):
the block MASK is a pure-torch product (`FlexSparseSelector`), and the
attention over the gathered keys is the fused Triton gather
(`sparse/msa_kernels.py`) — the owner's rule that kernel interfaces stay pure
torch in / torch out. The FLA alternatives were checked in the installed tree
and rejected on THEIR constraints, not on speed:
  * `fla.ops.nsa.parallel_nsa` (native sparse attention, has a backward) requires
    `HQ/H` to be a power of two and **>= 16** (`nsa/parallel.py:890`) and pulls
    in `flash_attn`; the port's head counts (12, 4) cannot satisfy it.
  * `fla.ops.attn.parallel_attn` (dense, has a backward) takes `window_size`
    and `cu_seqlens` but **no arbitrary mask head**, so it cannot express the
    indexer's selected-block set.
Both are recorded as the FLA drop-ins for this seam; neither is used — the arm
runs its own gather kernel, which is what the measured budget needed (see the
routes below).

**AGGREGATION**: gather the `r·K` key positions named by the selected blocks,
apply token-level causality, and take one softmax over the gathered keys — the
"softmax-normalized block weights" of the recipe. That is EXACTLY the dense
attention restricted to the selected blocks (the union of the blocks IS the
mask), which is the equivalence `tests/test_msa.py` replays.

**TWO ROUTES, ONE RESULT** (`MSA_KERNEL`, and both routes are gated against the
same naive per-query reference):

* `triton` — the fused gather of `sparse/msa_kernels.py`: one program per
  (batch, head, query-tile) reads the gathered key/value rows straight out of
  `k`/`v` and never materializes them. This is the route a CUDA run takes, and
  it is the arm's default (`auto`). MEASURED 2026-10-16 on the `small` preset
  b8 s512: the materialized route's arm cost 615 ms/step over a 91 ms/step
  MSA-off run (7.7x), one call of it is 95-100 ms fwd+bwd at 1.08 GiB, and no
  chunk size in the 2..32 sweep beats ~101 ms — the 1.6 GB gather per call IS
  the cost. The fused call is 7.6 ms at 0.19 GiB, and the same 2000-step run
  (seed 2, lr 1e-3, eval window 81 920 B) finishes in **325 s against 1442 s**,
  a 1.6x overhead over the arm-off run instead of 7.0x, with the loss curve
  inside seed spread of the torch route's (1.5347/1.8650/1.5432 train at
  500/1000/2000 against 1.5391/1.8673/1.5392; eval BPB 3.0400 against 3.0433).
* `torch` — the reference, kept for CPU and for the counted fallback: SLOT
  CHUNKS of `MSA_SLOT_CHUNK` selected-block columns with a conditional
  checkpoint region (`attend_selected_blocks`), so the largest tensor it builds
  is `[b,t,chunk*r,h,hd]` and not `[b,t,K*r,h,hd]`. Measured on the `small`
  preset (fp32 at this seam, d 768) at b8 s512: 13.59 GiB peak and an OOM
  before, 3.15 GiB after; the whole-m gather alone was 1.61 GB ×2 (k and v) ×4
  loop iterations = 12.9 GB of autograd-retained copies.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...instrument import Counters, loud
from .sparse import MSA_BLOCK, MSA_TOPK, FlexSparseSelector

try:  # the fused gather is an ABILITY: without triton the module is not importable
    from . import msa_kernels
except ImportError:  # pragma: no cover - the CPU-only wheel
    msa_kernels = None  # type: ignore[assignment]

__all__ = [
    "MSA_BLOCK",
    "MSA_TOPK",
    "MSA_SLOT_CHUNK",
    "MSA_CKPT_BYTES",
    "MSA_KERNEL",
    "BlockSparseAttention",
]

#: Selected-block columns gathered per chunk in `attend_selected_blocks`, i.e.
#: `MSA_SLOT_CHUNK * block` key positions per gather. A module constant and not
#: a config field: nothing sweeps it, and the value that matters is the ONE
#: measured bound it sets (the chunk gather is the largest tensor the torch
#: route builds). Only the FALLBACK route reads it — a CUDA run takes the fused
#: gather, which has no chunk (`MSA_KERNEL`).
#:
#: MEASURED 2026-10-16, `small` preset b8 s512, 6 steps each, peak is
#: `torch.cuda.max_memory_allocated` of the whole step
#: (`/tmp/opencode/chunk_sweep.py`): 1 slot 882 ms / 5.06 GiB · 2 slots
#: 850 ms / 3.56 GiB · **4 slots 661 ms / 3.15 GiB** · 8 slots 633 ms /
#: 3.51 GiB · all 32 slots in one region 632 ms / 7.73 GiB. RE-SWEPT the same
#: day after the conditional region landed, one arm call fwd+bwd in isolation
#: (`tests/test_msa_speed.py -- torch`): chunk 2 111.4 ms / 0.90 GiB ·
#: **chunk 4 101.4 ms / 1.08 GiB** · 8 109.6 / 1.74 · 16 120.7 / 3.20 ·
#: 32 125.4 / 6.19. 4 is the lowest peak of the five on the whole-step
#: instrument and the fastest of the five on the isolated one, so it stands.
MSA_SLOT_CHUNK = 4

#: The gathered bytes (`k` and `v`, one call) above which a chunk is wrapped in
#: a checkpoint region. The quantity is the WHOLE-m gather — `b·t·(m·r)·h·hd`
#: elements per tensor — because that is exactly what an uncheckpointed loop
#: retains for the backward: one live copy per chunk, i.e. all `m` of them.
#: MEASURED OOM POINT (2026-10-16, `small` b8 s512, 4 loop iterations): the
#: uncheckpointed loop retains 1.61 GB ×2 ×4 = 12.9 GB and OOMs; the same shape
#: with the region is 3.15 GiB peak. 256 MiB caps the retained set at ≤ 2 GiB
#: over `max_iter` 4 — 6x under the measured OOM — while a small shape (the
#: tiny test config's 4 MB whole-m gather) skips the recompute entirely. The
#: checkpoint is not free: it re-runs the whole chunk forward in the backward.
MSA_CKPT_BYTES = 256 << 20

#: Which aggregation route runs: `auto` (the fused gather when the ability and
#: the shapes allow, else the counted torch fallback), `torch`, or `triton`.
#: A module constant and not a `ModelConfig` field because the field is a
#: follow-up commit (`sparse/` is the arm's whole surface today; the same rule
#: `use_msa` was landed under). An unknown literal is LOUD at import.
MSA_KERNEL = "auto"


def _check_route(name: str) -> str:
    """The `MSA_KERNEL` literal, or a LOUD refusal naming the escape."""
    if name not in ("auto", "torch", "triton"):
        loud(f"MSA_KERNEL={name!r} is not a route. Escape: name one of auto|torch|triton")
    return name


_check_route(MSA_KERNEL)

#: Why the fused gather is not available at all, or None. Set once at import.
_TRITON_WHY: str | None = None
if msa_kernels is None:
    _TRITON_WHY = "triton is not installed (the fused gather is the CUDA arm)"


class BlockSparseAttention(nn.Module):
    """`x [b,t,d]` → `[b,t,d]`, or `None` when `use_msa` is off (COUNTED).

    The parameter set is KDA's own `q,k,v` projections (`3·d²`, the `g`/`beta`
    of the gated-delta arm are KDA-specific and are not carried here) plus the
    indexer — and ALL of it exists only when the arm is on, so the arm's absence
    is byte-for-byte the KDA-only model.
    """

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        assert h * cfg.head_dim == d, "validate() pins n_heads*head_dim == d_model"
        self.use_msa = bool(getattr(cfg, "use_msa", False))
        self.n_heads = h
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.topk = int(getattr(cfg, "msa_topk", MSA_TOPK))
        self.block = int(getattr(cfg, "msa_block", MSA_BLOCK))
        assert self.block >= 1 and self.topk >= 1, "msa_block/topk must be >= 1"
        self.selector: FlexSparseSelector | None = None
        self.q_proj: nn.Linear | None = None
        self.k_proj: nn.Linear | None = None
        self.v_proj: nn.Linear | None = None
        if self.use_msa:
            # Built only when the arm runs: `use_msa=False` must leave the whole
            # model's parameter count untouched (test_msa.py, case d).
            self.selector = FlexSparseSelector(cfg, device=device)
            self.q_proj = nn.Linear(d, d, bias=False, device=device)
            self.k_proj = nn.Linear(d, d, bias=False, device=device)
            self.v_proj = nn.Linear(d, d, bias=False, device=device)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor | None:
        """`x [b,t,d]` (block-body input) → `[b,t,d]`, or `None` when off."""
        counters = Counters() if counters is None else counters
        if not self.use_msa:
            counters.count("msa_off")
            return None
        assert self.selector is not None and self.q_proj is not None
        b, t, d = x.shape
        return self._forward(x, counters, k_override=None)

    def forward_k(self, x: torch.Tensor, counters: Counters, k_share: float) -> torch.Tensor | None:
        """The ADAPTIVE entry: K per query from `k_share` × candidate blocks.

        `k_share` is the per-ITERATION share (the adaptive schedule's ratio for
        this slot); 0 → MSA runs with no blocks this iteration (COUNTED
        `msa_iter_skip`, returns None so the caller adds nothing), the model
        spends its budget elsewhere. The share's mean over iterations is the
        arm's `msa_capacity_ratio` (owner's 1/4), the placement is learned.
        """
        assert self.use_msa and self.selector is not None
        b, t, _ = x.shape
        n_blocks = (t + self.block - 1) // self.block
        k = int(round(k_share * n_blocks))
        if k <= 0:
            counters.count("msa_iter_skip")
            return None
        return self._forward(x, counters, k_override=k)

    def _forward(self, x, counters, k_override) -> torch.Tensor:
        b, t, d = x.shape
        blocks, keep = self.selector(x, counters, k=k_override)
        counters.count("msa_blocks", b * t * blocks.shape[-1])
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, t, self.n_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, t, self.n_heads, self.head_dim)
        out = attend_selected_blocks(q, k, v, blocks, keep, self.block, self.scale, counters)
        assert out.shape == (b, t, self.n_heads, self.head_dim)
        return out.reshape(b, t, d)


def _gather_plan(
    blocks: torch.Tensor, keep: torch.Tensor, t: int, r: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`blocks [b,t,K]` (sentinel `n_blocks`) + `keep [b,t,K]` → the gather plan.

    `safe [b,t,M]` (M = K·r) is the absolute key position of every selected-block
    column, clamped into range so a sentinel slot reads a real row; `valid [b,t,M]`
    is the token-level causality AND block-keep mask. Both routes consume this one
    plan, so "which keys" cannot differ between them.
    """
    b, _, m = blocks.shape
    offsets = torch.arange(r, device=blocks.device)
    pos = (blocks.unsqueeze(-1) * r + offsets).reshape(b, t, m * r)  # [b,t,M]
    safe = pos.clamp(max=max(t - 1, 0))  # padded/sentinel slots point in range
    qpos = torch.arange(t, device=blocks.device).view(1, t, 1)
    valid = (pos < t) & (pos <= qpos) & keep.repeat_interleave(r, dim=-1)  # [b,t,M]
    assert valid.shape == (b, t, m * r)
    return safe, valid


def _triton_why(q: torch.Tensor, safe: torch.Tensor) -> str | None:
    """`None` if the fused gather can run THIS call, else why not.

    Every reason is COUNTED by the caller (ADR-0019): the fallback is a real
    degradation of the arm's speed and a reader of the log has to see it.
    """
    if _TRITON_WHY is not None:
        return _TRITON_WHY
    if not q.is_cuda:
        return f"input is on {q.device.type}: the fused gather is the CUDA arm"
    hd = q.shape[-1]
    if not (16 <= hd <= msa_kernels.MAX_HEAD_DIM and (hd & (hd - 1)) == 0):
        return (
            f"head_dim={hd} is not tileable by the fused gather "
            f"(16..{msa_kernels.MAX_HEAD_DIM}, power of two)"
        )
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return f"dtype {q.dtype} has no gather-attention tile"
    if not (q.is_contiguous() and safe.is_contiguous()):
        return "q/safe are not contiguous"
    return None


def _ckpt_needed(b: int, t: int, m: int, r: int, h: int, hd: int, itemsize: int) -> bool:
    """Is the WHOLE-m gather (what an uncheckpointed loop retains) over budget?

    See `MSA_CKPT_BYTES` for the measurement this threshold is set from.
    """
    return b * t * m * r * h * hd * itemsize > MSA_CKPT_BYTES


def _pos_pack(safe: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """`(safe, valid)` → the fused gather's `pos [b,t,C] int32` (-1 = invalid).

    One index load in the kernel instead of an index and a mask. -1 is distinct
    from every real position, so a padded or non-causal column can never be read
    as key 0 (which a `safe * valid` fold would do).
    """
    pos = torch.where(valid, safe, safe.new_full((), -1)).to(torch.int32)
    assert pos.shape == safe.shape
    return pos


def attend_selected_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    blocks: torch.Tensor,
    keep: torch.Tensor,
    block: int,
    scale: float,
    counters: Counters | None = None,
) -> torch.Tensor:
    """Gather the selected blocks' keys and attend, exactly.

    `q,k,v [b,t,h,hd]`; `blocks [b,t,K]` (sentinel `n_blocks` for padded slots),
    `keep [b,t,K]`. Returns `[b,t,h,hd]`.

    A query keeps key `p` iff `p` lies in a kept selected block AND `p <= i`
    (token-level causality). With no valid key the query would divide by a
    clamped denominator and read 0.0 — impossible here: every kept block's first
    token sits at or before the query (the selector's block-causal rule), so a
    query with any kept block always has at least one valid key. That invariant
    is what guards the online-softmax denominator below, and the assert is the
    gate on it (the fused route reads the same plan and clamps the same way).

    ROUTE (2026-10-16): the FUSED gather by default (`MSA_KERNEL`, see the module
    docstring for the measurement that put it there), the slot-chunked torch
    route as the reference and the counted fallback. The torch route walks the
    `m` selected columns in groups of `MSA_SLOT_CHUNK`, each group gathered,
    scored and folded into a running `(max, den, num)` softmax accumulator with
    the standard rescale — mathematically the one softmax over all kept keys,
    bit-for-bit close to it (`tests/test_msa.py` case e pins the parity).

    A chunk is a `torch.utils.checkpoint` region when `MSA_CKPT_BYTES` says the
    whole-m gather is too big to retain (and only outside `no_grad`), and that
    half is load-bearing rather than hygiene: the chunk's gathered
    `[b,t,chunk*r,h,hd]` is what the *backward* needs, so a plain loop would
    leave one retained copy per chunk — all `m` of them, the full tensor again.
    Checkpointing recomputes the gather in the backward instead, which is what
    makes the peak O(b·chunk·r·h·hd) rather than O(b·m·r·h·hd). Measured on the
    `small` preset at b8 s512 (2026-10-16, `/tmp/opencode/chunk_sweep.py`, 6
    steps, `torch.cuda.max_memory_allocated`): 13.59 GiB and an OOM with no
    chunk and no region, 7.73 GiB with the region but one whole-`m` chunk,
    3.15 GiB at `MSA_SLOT_CHUNK` 4. The same gather at b1's shape
    (b16 s1024, d 2048, fp32) is 2.1 GB per chunk instead of 17 GB whole.
    """
    b, t, h, hd = q.shape
    r = block
    m = blocks.shape[-1]
    safe, valid = _gather_plan(blocks, keep, t, r)
    if MSA_KERNEL != "torch":
        why = _triton_why(q, safe)
        if why is None:
            if counters is not None:
                counters.count("msa_triton")
            # -1 marks a column the query must not attend (`valid` false), so the
            # kernel needs one index load instead of an index and a mask.
            pos = _pos_pack(safe, valid)
            return msa_kernels.gather_attention(q, k, v, pos, scale)
        if counters is not None:
            counters.count("msa_triton_fallback")
            counters.why = f"triton: {why}"  # type: ignore[attr-defined]

    region = _ckpt_needed(b, t, m, r, h, hd, q.element_size())
    if region and counters is not None:
        counters.count("msa_ckpt")
    mx = torch.full((b, t, h), float("-inf"), dtype=q.dtype, device=q.device)
    den = torch.zeros((b, t, h), dtype=q.dtype, device=q.device)
    num = torch.zeros((b, t, h, hd), dtype=q.dtype, device=q.device)
    for s0 in range(0, m, MSA_SLOT_CHUNK):
        sl = slice(s0 * r, min(s0 + MSA_SLOT_CHUNK, m) * r)
        mxc, denc, numc = _maybe_checkpoint(
            _chunk, q, k, v, safe[:, :, sl], valid[:, :, sl], scale, region=region
        )
        mx_new = torch.maximum(mx, mxc)
        # Nothing valid anywhere yet (impossible for a query with a kept block,
        # but reachable while the first chunks are all empty): 0 keeps the
        # rescale finite and leaves den at 0, so the assert below fires.
        mx_new = torch.where(torch.isfinite(mx_new), mx_new, torch.zeros_like(mx_new))
        alpha = torch.exp(mx - mx_new)
        beta = torch.exp(mxc - mx_new)
        mx = mx_new
        den = den * alpha + denc * beta
        num = num * alpha.unsqueeze(-1) + numc * beta.unsqueeze(-1)

    den = den.clamp_min(torch.finfo(den.dtype).tiny)
    assert torch.isfinite(den).all(), "every query must have a valid key"
    return num / den.unsqueeze(-1)


def _chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    safe: torch.Tensor,
    valid: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One slot chunk → `(chunk max, chunk denominator, chunk numerator)`.

    `safe/valid [b,t,c]` is a column slice of the gather plan; the returned
    tensors are chunk-normalized (`p = exp(scores - chunk max)`) and the caller
    rescales them into its running accumulator. The chunk max may be `-inf`
    (every slot in this chunk masked) — the caller's rescale zeroes that
    contribution, and the `finite or zero` rule is applied HERE so `p` is a
    clean 0 row rather than a NaN.
    """
    b, t, h, hd = q.shape
    c = safe.shape[-1]
    b_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, t, c)
    kk = k.reshape(b, t, h * hd)[b_idx, safe].reshape(b, t, c, h, hd)
    vv = v.reshape(b, t, h * hd)[b_idx, safe].reshape(b, t, c, h, hd)
    scores = torch.einsum("bthd,btchd->bthc", q, kk) * scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    mxc = scores.amax(-1)  # [b,t,h]
    mx_used = torch.where(torch.isfinite(mxc), mxc, torch.zeros_like(mxc))
    p = torch.exp(scores - mx_used[..., None])  # exp(-inf) = 0 for masked slots
    return mxc, p.sum(-1), torch.einsum("bthc,btchd->bthd", p, vv)


def _maybe_checkpoint(fn, *args, region: bool = True):
    """`fn(*args)` — inside a checkpoint region when a gradient is being built.

    The region is what keeps the gather from being retained for the backward
    (see the route note above); outside `no_grad` it would only buy the
    recompute cost, so inference runs the plain call. `region` is the caller's
    own `MSA_CKPT_BYTES` verdict: a shape whose whole-m gather is small enough to
    retain skips the recompute, and the policy is COUNTED (`msa_ckpt`) when it
    fires.
    """
    if (
        region
        and torch.is_grad_enabled()
        and any(isinstance(a, torch.Tensor) and a.requires_grad for a in args)
    ):
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)
