"""A plain pre-LN transformer block — `scripts/control_transformer.py`'s Block.

WHY IT IS IN THE MODEL. The 2026-11-06 control
(`docs/reviews/2026-11-06-control-transformer.md`) trained a boring 13.5M
decoder on the trainer's own byte stream and beat the KDA/CSA/mHC/Engram block
by 0.126 BPB on all eight eval windows (0.13-0.37 by the report's range). An A/B
that answers "is the block form or the pipeline" therefore has to run BOTH forms
through this trainer: the embedding, data stream, eval window, optimizer,
checkpoint and resume are shared, and only the block differs.

THE BLOCK IS THE CONTROL'S, TRANSCRIBED. Pre-LayerNorm, no biases, packed
`qkv` (`3·d²`), RoPE on `head_dim`, `F.scaled_dot_product_attention(is_causal=True)`,
GELU FFN (`d_ffn = 4·d_model` in the control; an arm sets `d_ffn` and
`flagship3_dense` runs 3.69×, so the width is the config's, not this file's).
`ffn_gated=true` swaps that FFN for a gated GLU — `gate`/`up` into `fc2` as the
`down`, activation `expert_act` (`silu` = SwiGLU, `situ` = K3's capped GLU),
width `8·d/3` for the same bill — and is refused on every other block form
(`config.validate_ffn`). `ffn_form="tsct"` factors those THREE matrices
through `tsct.TSCTLinear` (`U[d,r]·diag(s)·V[r,f]`, ranks `ffn_rank_gate` /
`ffn_rank_down`, `0` = `rank`) — the one `ffn_form` value wired on this block;
every other value is refused (`config.validate_block_form`), because before
this wiring `ffn_form` reached no code here and an `arch_tsct275`-style config
built the dense control while its log line said `tsct`.
`attn_form="tsct"` factorises the ATTENTION side with the same module: the
packed `qkv [3d, d]` as ONE `[d, r]·diag(s)·[r, 3d]` (the dense arm packs all
three projections into that one matrix, so the factorisation keeps the packing)
and `proj` as `[d, r]·diag(s)·[r, d]`, rank `attn_rank` (`0` = `rank`). That is
the largest FLOP lever the FFN factorisation leaves inside the block: at
d=1152, f=2848, t=1024 the attention projections are 27% of the block's MAC
(5.31M of 19.87M per token) and SDPA's 4.72M is quadratic in the sequence, so
no rank cut shrinks it. Refused with a block form that builds no `qkv`/`proj`,
with `attn_latent_rank`/`fused_latent_proj` (a second attention factorisation
would double-apply) and with `act_qat`/`dense_qat` (their seams read plain
`nn.Linear` weights).
GPT-2 residual scaling on the two output projections
(`0.02/sqrt(2·n_layers)`). Parameters per block:
`4·d²` attention + `2·d·d_ffn` FFN + `4·d` of LayerNorm, which at
`d=304, ffn=1216` is 1 110 208 — the control's own count.

THE PORT'S DTYPE RULE, NOT A SECOND ONE. The residual stream stays fp32 (the
burn "mixed-dtype ops NaN" rule): every projection goes through
[`tsct.mm_mixed`] (fp32 masters, bf16 compute when `bf16=true`), the output is
widened back to the residual dtype, and SDPA runs in the compute dtype like
`FullAttention`. `bf16=false` (the CPU gates) is plain fp32 bit for bit.

MIXTURE-OF-DEPTHS (`mod_capacity < 1.0`, arXiv:2404.02258v1 §3.4). A per-block
router `r_i = w^T x_i` (one bias-free linear projection, `w ∈ R^d`) picks the
top-`C = round(capacity·T)` tokens of each sequence; those `C` tokens are
gathered, run through the block, scaled by their own `r_i` (Eq. 1's
`r_i·f(X̃) + x_i`, which puts the router on the gradient path) and scattered
back; the rest bypass through the residual. The gather is in POSITION order (a
scatter through the slots the selection occupies), so `is_causal=True` over the
compacted sequence is the same causal mask the full sequence had, and RoPE still
uses each token's ORIGINAL position. One `torch.topk` + gather/scatter per
block, static shapes, no `.item()`/`nonzero`/boolean-mask indexing — CUDA-graph
capturable by construction. `mod_capacity=1.0` builds no router at all: the
forward is the plain block, bit for bit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import check_gated_ffn_bill
from ..instrument import Counters, loud
from .experts import expert_glu
from .packed_acts import check_store, packed_act_mm, packed_glu
from .tsct import TSCTLinear, dense_qat_kind, dense_qat_weight, mm_dtype, mm_mixed

__all__ = [
    "TransformerBlock",
    "apply_rope",
    "mod_capacity_tokens",
    "mod_route",
    "rope_tables",
]


def rope_tables(
    t: int, head_dim: int, device: torch.device | str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """`cos, sin [1, 1, t, head_dim/2]` — the control's RoPE spectrum.

    `inv = 1 / 10000^(j/half)` for `j = 0..half-1`, `ang[p, j] = p·inv[j]`.
    Half the head rotates against the other half, so `head_dim` must be even
    (the `proj`/`qkv` split below is by `head_dim`, and an odd one would leave
    a channel unrotated).
    """
    assert head_dim % 2 == 0, f"RoPE needs an even head_dim, got {head_dim}"
    half = head_dim // 2
    inv = 1.0 / (10_000 ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    ang = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv)
    return ang.cos().to(dtype)[None, None], ang.sin().to(dtype)[None, None]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The control's pair rotation, `[..., hd] -> [..., hd]`, broadcast over heads."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def mod_capacity_tokens(capacity: float, top_k: int, t: int) -> int:
    """`C` — the tokens one block processes: `top_k`, else `round(capacity·T)`.

    Pure Python integers (`t` is a shape), so it costs no host read. `top_k`
    wins when nonzero; otherwise the capacity is rounded to the nearest integer
    and clamped into `1..t` (a static shape either way).
    """
    assert t >= 1, t
    if top_k:
        if top_k > t:
            loud(
                f"mod_top_k={top_k} > the sequence's {t} positions: the router "
                f"cannot select more tokens than the forward holds. Escape: "
                f"mod_top_k <= {t}, or 0 to derive it from mod_capacity."
            )
        return int(top_k)
    return min(max(int(round(capacity * t)), 1), t)


def mod_route(
    scores: torch.Tensor, capacity: float, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-`C` selection as ONE `topk` + scatter/cumsum — no host read.

    `scores [b, t]` are the router logits. Returns

    * `idx [b, C]` — the selected token positions (score order, as `topk`
      returns them);
    * `sel [b, t]` int64 — the selection mask (1 where selected), built by a
      scatter (never a boolean mask or `nonzero`);
    * `slots [b, C]` — each selected token's rank in POSITION order, built from
      `cumsum` of the mask. A `scatter` through `slots` compacts the selection
      into original-position order without a second sort, so `is_causal=True`
      on the compacted sequence is the same causal mask the full sequence has.

    Every op is device-side and static-shape; the only host values are `C`
    (from Python ints) and the buffer sizes. This is the capture-safe form the
    CUDA-graph probe in `scripts/probe_capture.py` looks for.
    """
    assert scores.dim() == 2, scores.shape
    b, t = scores.shape
    c = mod_capacity_tokens(capacity, top_k, t)
    idx = torch.topk(scores, c, dim=1).indices
    sel = torch.zeros((b, t), dtype=torch.long, device=scores.device)
    sel.scatter_(1, idx, 1)
    slots = (sel.cumsum(1) - 1).gather(1, idx)
    return idx, sel, slots


class TransformerBlock(nn.Module):
    """One vanilla decoder block: `x + proj(SDPA(RoPE(qkv(norm(x)))))`, then FFN.

    With `mod_capacity < 1.0` (or `mod_top_k > 0`) the same block also owns a
    Mixture-of-Depths router; its `forward` then routes `C` tokens through
    `_block` and passes the rest through unchanged (see the module docstring).
    """

    def __init__(
        self, cfg, *, n_layers: int, layer_idx: int = 0, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__()
        d, ffn, heads, hd = cfg.d_model, cfg.d_ffn, cfg.n_heads, cfg.head_dim
        assert heads * hd == d, "validate() pins n_heads*head_dim == d_model"
        assert d % heads == 0, f"d_model={d} must tile {heads} heads"
        assert hd % 2 == 0, f"RoPE needs an even head_dim, got {hd}"
        assert n_layers >= 1, n_layers
        self.d = d
        self.n_heads = heads
        self.head_dim = hd
        self.scale = 1.0 / math.sqrt(hd)
        self.compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        #: `nope_layers` = the LAST N blocks of the stack skip RoPE. The block
        #: is built with its own index so the rule is per-instance, not a
        #: global flag the forward would have to re-derive.
        self.nope = bool(cfg.nope_layers) and layer_idx >= n_layers - cfg.nope_layers
        #: The carry is a property of the STACK, not of α: block 0 has no α but
        #: is the source, so `last_v` is set whenever the arm is on.
        self.value_residual = bool(cfg.value_residual)
        self.ffn_clip = float(cfg.ffn_clip)
        #: `nn.LayerNorm`'s OWN default eps (1e-5), deliberately: the dense
        #: form's blocks have never read `cfg.norm_eps`, and wiring it now would
        #: move every recorded dense number (the knob's default is 1e-3 for the
        #: KDA form's RMSNorms). The knob is NOT dead on this form — the FINAL
        #: RMSNorm in `model.py` reads it — so an A/B on `norm_eps` tests the
        #: final norm only; `tests/test_block_form.py` pins both halves.
        self.ln1 = nn.LayerNorm(d, device=device)
        #: The dense arm's init contract (`nn.init.normal_` below): every
        #: projection N(0, 0.02), then the two writes into the residual stream
        #: shrunk by sqrt(2·L). The `TSCTLinear`s (attention under
        #: `attn_form="tsct"`, the FFN under `ffn_form="tsct"`) read the SAME
        #: numbers as their composed `U·diag(s)·V` entry std, so a fresh
        #: factorised block writes the dense block's distribution — without that
        #: the A/B would measure init scale (13.3x at the live shape for the
        #: FFN), not the factorisation.
        std = 0.02
        #: `attn_form="tsct"`: the packed `qkv [3d, d]` and the readout
        #: `proj [d, d]` are `TSCTLinear`s at rank `attn_rank` (`0` = `rank`).
        #: `config.validate_block_form` owns the loader-path refusals; the three
        #: checks below are the runtime seam a hand-built config reaches.
        self.attn_tsct = getattr(cfg, "attn_form", "dense") == "tsct"
        self.r_attn = 0
        if self.attn_tsct:
            if cfg.block_form != "transformer" or cfg.block_stack != "uniform":
                loud(
                    f"attn_form='tsct' on block_form={cfg.block_form!r}/"
                    f"block_stack={cfg.block_stack!r}: only the UNIQUE transformer "
                    f"block builds `qkv`/`proj`, so on this stack the flag would "
                    f"reach no code. Escape: block_form='transformer' with "
                    f"block_stack='uniform', or attn_form='dense'."
                )
            if cfg.attn_latent_rank or cfg.fused_latent_proj:
                loud(
                    f"attn_form='tsct' with attn_latent_rank="
                    f"{cfg.attn_latent_rank}/fused_latent_proj="
                    f"{cfg.fused_latent_proj}: those are a DIFFERENT attention "
                    f"factorisation, and pairing them with a factored `qkv` would "
                    f"apply two rank cuts to one projection. Escape: both off, or "
                    f"attn_form='dense'."
                )
            if cfg.act_qat != "none" or cfg.dense_qat != "none":
                loud(
                    f"attn_form='tsct' with act_qat={cfg.act_qat!r}/dense_qat="
                    f"{cfg.dense_qat!r}: both seams read a plain `nn.Linear` "
                    f"weight and this form's are `U`/`s`/`V` — the arm would be "
                    f"silently absent. Escape: both 'none', or attn_form='dense'."
                )
            self.r_attn = cfg.attn_rank or cfg.rank
            attn_kw = dict(
                dtype=self.compute_dtype,
                factor_qat=cfg.factor_qat,
                factor_qat_alpha=cfg.factor_qat_alpha,
                device=device,
            )
            self.qkv = TSCTLinear(d, 3 * d, self.r_attn, weight_std=std, **attn_kw)
            self.proj = TSCTLinear(
                d, d, self.r_attn, weight_std=std / math.sqrt(2 * n_layers), **attn_kw
            )
        else:
            self.qkv = nn.Linear(d, 3 * d, bias=False, device=device)
            self.proj = nn.Linear(d, d, bias=False, device=device)
        self.ln2 = nn.LayerNorm(d, device=device)
        #: `ffn_gated` (`config.validate_ffn` owns the refusals): the plain
        #: `gelu(fc(x))@fc2^T` becomes `fc2(expert_glu(act)(gate(x), up(x)))`.
        #: `fc2` stays the GLU's `down` on purpose — same `[d, ffn]` shape and
        #: the same state-dict name as the GELU block's write, so an export path
        #: that names `fc2` reads a write in both arms.
        self.ffn_gated = bool(getattr(cfg, "ffn_gated", False))
        self.expert_act = cfg.expert_act
        #: `ffn_form="tsct"`: the gated FFN's three matrices are
        #: `TSCTLinear`s. Only built on the gated path (the factorised form has
        #: no ungated arm), and only this form reaches code here —
        #: `config.validate_block_form` refuses every other non-default value,
        #: so a config cannot name a form this block never builds.
        self.ffn_tsct = cfg.ffn_form == "tsct"
        self.beta1 = float(cfg.expert_beta1)
        self.beta2 = float(cfg.expert_beta2)
        #: The dense-QAT seam (`dense_qat`, `DENSE_QATS`): the SAME projector the
        #: loop form's dense path runs, on this block's six `nn.Linear` weights.
        #: `dense_qat_skip_attn` (the BitNet b1.58 rule) keeps `qkv`/`proj` on
        #: their masters; the site names are the ones `DENSE_QAT_ATTN_SITES`
        #: lists, so the skip rule and this class cannot disagree. `alpha` rides
        #: `factor_qat_alpha` — `DormouseModel.set_factor_qat_alpha` walks every
        #: module carrying `dense_qat_alpha`, so ONE schedule drives both forms.
        self.dense_qat_alpha = float(cfg.factor_qat_alpha)
        self._qat_attn = dense_qat_kind(cfg, "transformer.qkv", "transformer.proj")
        self._qat_ffn = dense_qat_kind(
            cfg,
            "transformer.fc",
            "transformer.fc2",
            "transformer.gate_proj",
            "transformer.up_proj",
        )
        #: The 4-bit ACTIVATION arm (`act_qat`, `ACT_QATS`): the FFN GEMM inputs
        #: AND the GLU operands are fake-quantized to e2m1 with one e4m3 scale per
        #: 16 columns — the NVFP4 layout (`model/packed_acts.py`) — and SAVED in
        #: that packed form, which is what drops the step's VRAM. Attention stays
        #: at >= 8 bits (a4.8's rule), and the packed path is skipped entirely
        #: when the flag is off, so the off forward is the same calls it has
        #: always made.
        self.act_qat = cfg.act_qat
        #: WHERE the graph keeps this block's saved activations (`act_store`,
        #: `ACT_STORES`): "bf16" (the historical graph, bit for bit) or "nvfp4"
        #: (each GEMM input / GLU operand is SAVED packed, `model/packed_acts.py`'s
        #: store-only lane). A TrainConfig memory arm, so it is set once by
        #: `DormouseModel.set_act_store` — a model forwarded without that call
        #: keeps "bf16" and takes no branch and allocates no pack.
        self.act_store = "bf16"
        self.r_gate = self.r_down = 0
        if self.ffn_gated:
            # The runtime seam a hand-built config reaches, the same rule
            # `config.validate_ffn` applies on the loader path: the gated bill
            # must be the ungated block's, or the flag is a bigger model.
            check_gated_ffn_bill(d, ffn, cfg.d_ffn_gelu)
            if self.ffn_tsct:
                # The runtime seam `config.validate_block_form` owns: a
                # hand-built config can reach the block without the loader.
                if cfg.n_experts != 1 or cfg.moe_topk:
                    loud(
                        f"ffn_form='tsct' with n_experts={cfg.n_experts}/"
                        f"moe_topk={cfg.moe_topk} on the transformer block: this "
                        f"block has one FFN and no router, so the bank of tsct "
                        f"experts would never be built. Escape: n_experts=1 and "
                        f"moe_topk=0."
                    )
                self.r_gate = cfg.ffn_rank_gate or cfg.rank
                self.r_down = cfg.ffn_rank_down or cfg.rank
                if self.r_gate < 1 or self.r_down < 1:
                    loud(
                        f"ffn_rank_gate={self.r_gate} / ffn_rank_down="
                        f"{self.r_down}: a factor rank must be >= 1. Escape: a "
                        f"positive rank, or ffn_form='bank'."
                    )
                kw = dict(
                    dtype=self.compute_dtype,
                    factor_qat=cfg.factor_qat,
                    factor_qat_alpha=cfg.factor_qat_alpha,
                    device=device,
                )
                self.gate_proj = TSCTLinear(d, ffn, self.r_gate, weight_std=std, **kw)
                self.up_proj = TSCTLinear(d, ffn, self.r_gate, weight_std=std, **kw)
                self.fc2 = TSCTLinear(
                    ffn, d, self.r_down, weight_std=std / math.sqrt(2 * n_layers), **kw
                )
            else:
                self.gate_proj = nn.Linear(d, ffn, bias=False, device=device)
                self.up_proj = nn.Linear(d, ffn, bias=False, device=device)
        else:
            if self.ffn_tsct:
                loud(
                    "ffn_form='tsct' with ffn_gated=false: the factorised form "
                    "replaces the GATED FFN's gate/up/down, and the plain "
                    "`gelu(fc)` block has no arm. Escape: ffn_gated=true, or "
                    "ffn_form='bank'."
                )
            self.fc = nn.Linear(d, ffn, bias=False, device=device)
        if not self.ffn_tsct:
            self.fc2 = nn.Linear(ffn, d, bias=False, device=device)
        #: Value residuals (`value_residual`): a per-head scale on block 0's
        #: value, ZERO-initialised so the arm starts bit-equal to off. Block 0
        #: IS the source (`v_first`), so it gets no α — an α there would be a
        #: parameter with no reader, the dead-capacity defect this port audits.
        self.v_res_alpha: nn.Parameter | None = None
        if cfg.value_residual and layer_idx > 0:
            self.v_res_alpha = nn.Parameter(torch.zeros(1, heads, 1, 1, device=device))
        #: Sigmoid attention gate (`attn_gate`): one logit per head from `ln1`.
        #: bias=+4.0 keeps sigmoid(≈0.982) near the ungated write at step 0 —
        #: a zero-bias gate would halve the attention output before it learns.
        self.gate: nn.Linear | None = None
        if cfg.attn_gate:
            self.gate = nn.Linear(d, heads, device=device)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 4.0)
        #: Each `nn.Linear` child DECLARES the kind its forward reads, so
        #: `model/export.py` packs exactly the weights this block quantized —
        #: and never a neighbour — from the model itself rather than a second,
        #: hand-kept site list. `gate` (the attention gate) is in no site list,
        #: so it says `none` out loud instead of being guessed at.
        for _site, _kind in (
            ("qkv", self._qat_attn),
            ("proj", self._qat_attn),
            ("fc", self._qat_ffn),
            ("fc2", self._qat_ffn),
            ("gate_proj", self._qat_ffn),
            ("up_proj", self._qat_ffn),
            ("gate", "none"),
        ):
            _mod = getattr(self, _site, None)
            if isinstance(_mod, nn.Linear):
                _mod._qat_kind = _kind
        #: Block 0's value for the NEXT block's `value_residual` (None until the
        #: first forward; the model threads it, the block never looks ahead).
        self.last_v: torch.Tensor | None = None
        # The control's init (`Control._init` + its residual scaling): every
        # projection N(0, 0.02), then the two writes into the residual stream
        # shrunk by sqrt(2·L) so the stream does not grow with depth. LayerNorm
        # keeps torch's (weight=1, bias=0), and the embedding is the port's
        # shared one, not re-drawn.
        if not self.attn_tsct:
            nn.init.normal_(self.qkv.weight, std=std)
        if self.ffn_gated and not self.ffn_tsct:
            nn.init.normal_(self.gate_proj.weight, std=std)
            nn.init.normal_(self.up_proj.weight, std=std)
        elif not self.ffn_gated:
            nn.init.normal_(self.fc.weight, std=std)
        if not self.attn_tsct:
            nn.init.normal_(self.proj.weight, std=std / math.sqrt(2 * n_layers))
        if not self.ffn_tsct:
            nn.init.normal_(self.fc2.weight, std=std / math.sqrt(2 * n_layers))
        # The MoD router (`r_i = w^T x_i`), built IFF routing is on. A
        # CONSTANT init, not `nn.Linear`'s: `nn.Linear` would draw from the
        # global RNG, shifting every later block's init and turning the arm
        # into a different seed. `0.02` gives `r ≈ 0.02·Σx ≈ O(1)` on the
        # embedding stream, so the scaled block output starts at full strength
        # and every token is selectable. At `mod_capacity=1.0` there is no
        # router and `forward` is the plain block, bit for bit.
        self.mod_capacity = float(cfg.mod_capacity)
        self.mod_top_k = int(cfg.mod_top_k)
        self.router: nn.Parameter | None = None
        if self.mod_capacity < 1.0 or self.mod_top_k:
            self.router = nn.Parameter(torch.full((1, d), 0.02, device=device))
        #: The last forward's BCE aux value, DETACHED (log-cadence read only).
        self.last_aux: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        counters: Counters | None = None,
        v_first: torch.Tensor | None = None,
        *,
        attnres=None,
        layer_idx: int = 0,
        updates: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """`x [b, t, d]` in the residual dtype → `[b, t, d]` in the same dtype.

        `v_first` is block 0's value when `value_residual` is on (the caller
        threads it through `last_v`); `None` on the first block and on every
        default run.

        `attnres` is the model's `FullDepthAttnRes` when `use_attnres` is on
        (`model/attnres.py`): the block computes its body as usual, takes the
        residual delta `plain − x` as this layer's update, and writes the
        mixture instead of the delta. `updates` is the history the model threads
        across layers (oldest first). All three are `None`/unused on the default
        path, so the flag-off forward is the same calls it has always made.
        """
        if self.router is None:
            plain = self._block(x, counters, v_first=v_first)
        else:
            plain = self._forward_mod(x, counters, v_first=v_first)
        if attnres is None:
            return plain
        assert updates is not None, "AttnRes needs the model's update history"
        current = plain - x.to(plain.dtype)
        sources = [current] + [updates[k] for k in range(len(updates) - 1, -1, -1)]
        a = attnres(sources, layer_idx)
        updates.append(current)
        return x.to(plain.dtype) + a

    def set_act_store(self, mode: str) -> None:
        """The STORE-ONLY arm's switch (`act_store`): "bf16" | "nvfp4".

        Called once by `DormouseModel.set_act_store` (the trainer's
        `configure_actmem`), never per step: it changes what the graph RETAINS,
        not what the forward computes. This is the runtime seam a hand-built
        config reaches; `config.validate_model_train` owns the same refusals.
        """
        check_store(mode)
        if mode != "bf16":
            if self.act_qat != "none":
                loud(
                    f"act_store='nvfp4' with act_qat={self.act_qat!r} on this "
                    f"block: `act_qat` already saves the FFN operands packed, so "
                    f"one save set would carry two names with different forward "
                    f"numerics. Escape: act_qat='none', or act_store='bf16'."
                )
            if self.attn_tsct or self.ffn_tsct:
                loud(
                    "act_store='nvfp4' with a `TSCTLinear` projection "
                    "(attn_form/ffn_form='tsct'): the packed store wraps a dense "
                    "GEMM input and the factorised form has no such site. "
                    "Escape: dense/bank forms, or act_store='bf16'."
                )
            if mm_dtype() == "fp8":
                loud(
                    "act_store='nvfp4' with mm_dtype='fp8': the store-only lane's "
                    "forward runs `mm_mixed`, which enters the fp8 GEMM lane and "
                    "changes the forward's numerics — the arm would no longer be "
                    "memory-only. Escape: mm_dtype='bf16', or act_store='bf16'."
                )
        self.act_store = mode

    def _block(
        self,
        x: torch.Tensor,
        counters: Counters | None = None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        v_first: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The plain block. `rope` is the pre-built `(cos, sin)` table of a
        compacted MoD call (per-token ORIGINAL positions); `None` is today's
        table over positions `0..t-1`, and `self.nope` skips it entirely."""
        if not self.attn_tsct:
            # The `TSCTLinear` widens its own activation (and returns fp32); a
            # dense block casts the residual onto its master dtype first, as it
            # always has.
            pdt = self.qkv.weight.dtype
            if x.dtype is not pdt:
                x = x.to(pdt)
        b, t, _ = x.shape
        d = self.d
        if self.act_qat != "none" and counters is not None:
            counters.count("act_qat_arms")
        if self.attn_tsct and counters is not None:
            counters.count("attn_tsct", b * t)
        h1 = self.ln1(x)
        qkv = self._attn_proj(h1, self.qkv, counters)
        q, k, v = qkv.split(d, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if self.value_residual:
            if v_first is None:
                v_first = v
            else:
                v = v + self.v_res_alpha.to(v.dtype) * v_first
            self.last_v = v_first
        if not self.nope:
            if rope is None:
                cos, sin = rope_tables(t, self.head_dim, x.device, q.dtype)
            else:
                cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        if self.gate is not None:
            g = torch.sigmoid(F.linear(h1, self.gate.weight, self.gate.bias))
            y = y * g.transpose(1, 2).unsqueeze(-1).to(y.dtype)
        y = y.transpose(1, 2).reshape(b, t, d)
        # Every projection runs in the compute dtype; the residual add is the
        # fp32 stream (mm_mixed's contract, widened back here).
        x = x + self._attn_proj(y, self.proj, counters).to(x.dtype)
        h2 = self.ln2(x)
        return x + self._ffn(h2, counters).to(x.dtype)

    def _attn_proj(
        self, x: torch.Tensor, module: nn.Module, counters: Counters | None
    ) -> torch.Tensor:
        """One of the two attention projections, whichever form built it.

        `attn_form="tsct"` calls the `TSCTLinear` itself (its forward owns the
        factor matmuls and `factor_qat`; both `dense_qat` and `act_qat` are
        refused with that form, so no second numerics path can apply). Its
        result is widened back to the compute dtype, exactly what the dense
        arm's `mm_mixed` already returns: SDPA is quadratic in the sequence and
        must stay on the tensor-core dtype (`bf16=false` is fp32 either way).
        """
        if self.attn_tsct:
            return module(x).to(self.compute_dtype)
        if self.act_store == "nvfp4":
            return self._store_mm(x, module, self._qat_attn, counters)
        return self._qmm(x, module, self._qat_attn, counters)

    def _ffn(self, h2: torch.Tensor, counters: Counters | None) -> torch.Tensor:
        """The FFN body: `ln2(x) [b,t,d]` in, the FFN's write `[b,t,d]` out.

        Three arms, one seam: the factorised GLU (`ffn_tsct`, COUNTED as
        `ffn_tsct`), the dense gated GLU (`ffn_gated`) and the plain GELU
        block. The last two are the calls this method has always made; the
        tsct arm bypasses the packed/dense-QAT seams (both read an `nn.Linear`
        weight, and this arm's weights are `U`/`s`/`V`), which is why those
        combinations are refused at config load and at `__init__`.
        """
        packed = self.act_qat != "none"
        if self.ffn_tsct:
            if counters is not None:
                counters.count("ffn_tsct", h2.shape[0] * h2.shape[1])
            g = self.gate_proj(h2)
            u = self.up_proj(h2)
            h = expert_glu(self.expert_act, g, u, beta1=self.beta1, beta2=self.beta2)
        elif self.ffn_gated:
            if packed:
                # ONE pack feeds gate and up; `packed_glu` then owns the GLU so
                # its three `[T, f]` saves become two packed `[T, f/2]`.
                g, u = packed_act_mm(
                    h2,
                    self.compute_dtype,
                    self._qat_weight(self.gate_proj, self._qat_ffn),
                    self._qat_weight(self.up_proj, self._qat_ffn),
                    counters=counters,
                )
                h = packed_glu(
                    self.expert_act,
                    g,
                    u,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    counters=counters,
                )
            else:
                if self.act_store == "nvfp4":
                    g, u = packed_act_mm(
                        h2,
                        self.compute_dtype,
                        self._qat_weight(self.gate_proj, self._qat_ffn),
                        self._qat_weight(self.up_proj, self._qat_ffn),
                        counters=counters,
                        store_only=True,
                    )
                    h = packed_glu(
                        self.expert_act,
                        g,
                        u,
                        beta1=self.beta1,
                        beta2=self.beta2,
                        counters=counters,
                        store_only=True,
                    )
                else:
                    g = self._qmm(h2, self.gate_proj, self._qat_ffn, counters)
                    u = self._qmm(h2, self.up_proj, self._qat_ffn, counters)
                    h = expert_glu(self.expert_act, g, u, beta1=self.beta1, beta2=self.beta2)
        elif packed:
            h = F.gelu(
                packed_act_mm(
                    h2,
                    self.compute_dtype,
                    self._qat_weight(self.fc, self._qat_ffn),
                    counters=counters,
                )[0]
            )
        else:
            h = F.gelu(self._store_mm(h2, self.fc, self._qat_ffn, counters))
        if self.ffn_clip > 0.0:
            h = h.clamp(max=self.ffn_clip)
        if self.ffn_tsct:
            h = self.fc2(h)
        elif packed:
            h = packed_act_mm(
                h,
                self.compute_dtype,
                self._qat_weight(self.fc2, self._qat_ffn),
                counters=counters,
            )[0]
        else:
            h = self._store_mm(h, self.fc2, self._qat_ffn, counters)
        return h

    def _qat_weight(self, linear: nn.Linear, kind: str) -> torch.Tensor:
        """`dense_qat` projected onto `linear`'s weight — the master when off.

        ONE call shape for the six projections: `dense_qat_weight` short-circuits
        `kind == "none"` (no projector call at all, the gate
        `tests/test_dense_qat.py` requires), and the per-step `QuantCache` lives
        on the `nn.Linear` itself (`attach_qat_caches`).
        """
        return dense_qat_weight(
            linear.weight,
            kind,
            self.dense_qat_alpha,
            cache=getattr(linear, "_qat_cache", None),
        )

    def _qmm(
        self, x: torch.Tensor, linear: nn.Linear, kind: str, counters: Counters | None
    ) -> torch.Tensor:
        """`mm_mixed(x, W)` in the compute dtype — the bf16 (unpacked) lane."""
        return mm_mixed(x, self._qat_weight(linear, kind), self.compute_dtype, counters)

    def _store_mm(
        self, x: torch.Tensor, linear: nn.Linear, kind: str, counters: Counters | None
    ) -> torch.Tensor:
        """`_qmm`, or the SAME GEMM with `x` saved as NVFP4 bytes (`act_store`).

        The forward is `mm_mixed`'s bf16 line, bit for bit — the packed copy is
        storage only. The backward's `grad_W` reads the 4-bit rounding of `x`;
        `grad_x` is exact (`grad @ W`). `packed_act_mm` pads a width the codec's
        16-column scale grid cannot tile (the live `d=890`) and slices the pad
        off on unpack, so the store never changes the arithmetic.
        """
        if self.act_store == "nvfp4":
            return packed_act_mm(
                x,
                self.compute_dtype,
                self._qat_weight(linear, kind),
                counters=counters,
                store_only=True,
            )[0]
        return self._qmm(x, linear, kind, counters)

    def _forward_mod(
        self,
        x: torch.Tensor,
        counters: Counters | None,
        v_first: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route `C = mod_capacity_tokens(...)` tokens through the block.

        One `topk` on the router score, gather into fixed buffers in POSITION
        order, the plain block on those `C` tokens, then `r_i·f_i + x_i` (Eq. 1)
        scattered back. No `.item()`, `nonzero`, boolean-mask indexing or host
        sync anywhere in the path — the buffers are sized from the static
        shape, so a CUDA graph can hold it.
        """
        b, t, d = x.shape
        scores = F.linear(x, self.router).squeeze(-1)  # [b, t], fp32 residual
        idx, sel, slots = mod_route(scores, self.mod_capacity, self.mod_top_k)
        c = idx.shape[1]
        g = idx.unsqueeze(-1).expand(-1, -1, d)
        sg = slots.unsqueeze(-1).expand(-1, -1, d)
        # Compact into position order through `slots` (a scatter, not a sort).
        buf = x.new_zeros((b, c, d)).scatter_(1, sg, x.gather(1, g))
        rbuf = scores.new_zeros((b, c)).scatter_(1, slots, scores.gather(1, idx))
        pos = torch.arange(t, device=x.device).expand(b, t).gather(1, idx)
        pbuf = torch.zeros((b, c), dtype=torch.long, device=x.device).scatter_(1, slots, pos)
        cos_tab, sin_tab = rope_tables(t, self.head_dim, x.device, self.compute_dtype)
        cos = cos_tab[0, 0][pbuf].unsqueeze(1)
        sin = sin_tab[0, 0][pbuf].unsqueeze(1)
        y = self._block(buf, counters, rope=(cos, sin), v_first=v_first)
        # Eq. 1: `r_i · f_i(X̃) + x_i`. The router weight is ON the gradient
        # path, which is the only gradient the router gets from the LM term
        # (the top-k identity itself is not differentiable).
        delta = (y - buf) * rbuf.unsqueeze(-1)
        out = x + x.new_zeros((b, t, d)).scatter(1, g, delta.gather(1, sg))
        # The paper's §3.5 aux: BCE(logits=r_i, target=top-k indicator). One
        # value per block, detached for the log line; the caller weights it.
        self.last_aux = F.binary_cross_entropy_with_logits(scores, sel.to(scores.dtype))
        if counters is not None:
            counters.count("mod_block")
            counters.count("mod_tokens", b * c)
        return out
