"""The loop block — `max_iter` iterations over `loop_blocks` distinct bodies.

`loop_blocks=1` (the default) is the weight-shared loop every number on record
ran: ONE parameter set, `max_iter` iterations. `loop_blocks=N` (config.py)
cycles the SAME loop state through `N` DISTINCT bodies — iteration `i` runs
`blocks[i % N]` — so the loop gains depth from a new transformation, not only
from more passes of the same one. Each body is a full block (controller, KDA,
Engram, its own MoE bank, readout, residual operator); the per-ITERATION slots
(`iter_embed`, `attnres`, `msa_gate`, `csa_gate`, `loom_gate`) belong to the
driver once, because they name which pass is running, not which body runs it.
`DormouseModel` builds body 0 as `loop` and the rest as `loop_extra`
(`body_only`), and points `loop._blocks` at the tuple, so every path —
one-shot, segmented, recurrent decode — selects the same body per iteration.

Mirrors `crates/dormouse-core/src/loop_block.rs`. One iteration, in order:

1. the **iteration embedding** `e_i` is added to the state (this is the only
   thing that makes the `T` passes distinguishable, since the body is shared);
2. a single **RMSNorm** (shared, like everything else here) pre-norms it;
3. the **controller** reads `[h_ctx ; h0]` and emits `sigmoid(w_attn)`,
   `sigmoid(w_mem)`, `sigmoid(w_ffn)` plus the expert blend logits;
4. the **attention** (KDA, or its counted SDPA fallback) and the **experts**
   run, each scaled by its gate;
5. the **Engram** memory joins as a convex mixture with a hard floor:
   `lam = min(w_mem, lam_max)`;
6. the body is deposited by **ReZero** (`h = h_ctx + α·y`, α init 1.0 — NOT 0:
   at 0 the body gets no gradient at all), by **mHC**'s K-lane Sinkhorn mix,
   by **AttnRes**' pseudo-query mixture, or by the two TOGETHER (mHC carries
   the lanes, AttnRes chooses the update `H_post` deposits — "THE
   COMPOSITION" below);
7. the readout is `out_proj(h)`, and the final state is the AVERAGE over
   iterations. `L_Rec` — the per-iteration CE through the same `lm_head` — is
   accumulated inside the loop (deep supervision, burn's own shape), which is
   also why it is never materialized as `[T, b, t, d]`.

Depth is fixed by default and the per-iteration CE is an unweighted mean
(ADR-0013). The learned HALT HEAD (`halt.py`, built IFF `halt_weight > 0`)
reads the same per-iteration readout at INFERENCE and stops when its stop
probability passes 0.5; its target is the per-iteration CE's own improvement,
never the label sequence. The ADAPTIVE DEPTH arm
passes `depth=T` to run `T` iterations instead, and the readout divides by the
`T` actually run so different depths are scale-comparable.

**LOOM** (`2610.01153`, 2026-10-01; scanned in
`docs/research/2026-10-16-new-tech-scan.md` §2.1) is the arm behind
`cfg.loom`. The paper measures a looped MoE stalling at **two** iterations —
hidden-state variance growth plus expert-selection collapse — and reports 9-12
stable loops at 100M-1.7B, i.e. exactly the band our `max_iter 4` / depth 2
sits in. Its fix is a bundle (bounded residual updates, input re-injection per
loop, **per-loop routers**, a looping residual); what is transcribed here is the
MINIMAL faithful half of that summary and nothing more: **one learned per-slot
gate on the RMS-normalized write** — per-loop router conditioning, the only
thing that can distinguish the `T` passes of a weight-shared body, plus the
variance control the normalized write carries (it cannot grow with depth). The
gate is zero-initialised, so `loom = true` at init IS the `loom = false`
forward; the arm must earn its depth by training, and the A/B against a
fixed-depth control is the arbiter of the 8-12-loop claim. **CAVEAT, and it is
the scan's own convention: the arm is a transcription of the scan's summary;
the paper's full recipe may differ — re-derive before any multi-loop default.**

SEGMENTED TRAINING (`segment_len`, off at 0) — the enabler for training on
windows longer than one forward fits in VRAM. `_forward_segmented` cuts the
window into `t // segment_len` segments and runs the SAME `_window` body on
each, carrying the KDA state across the cuts **per iteration slot** (the loop
calls the attention arm once per iteration, so the state is per iteration, not
one state for the whole block) as a real graph edge, each segment inside
`torch.utils.checkpoint(..., use_reentrant=False)`. That is StateFlow's
boundary state AND gradient propagation (arXiv:2608.06838) rather than the
truncated BPTT the burn side runs (`forward_train_state`), and it is a MEMORY
decision only: the segmented logits are the one-shot logits to fp32 round-off
(`tests/test_longctx.py`), so the trainer's loss code, the eval window and
every number on record keep their meaning. `segment_len=0` does not enter the
branch at all — the one-shot path is `_window(carry=False)`, the same code with
the same calls it has always made.

CSA (`model/csa.py`) is the **compressed, selectable KV memory**: one shared
`k`/`v` per `csa_chunk` byte positions and exact attention over the top
`csa_topk` of the STRICTLY EARLIER chunks. It is ADDITIVE beside KDA — `attn =
attn_kda + csa_out · w_attn · sigmoid(csa_gate[i])`, the `msa_gate` pattern — and
built only when `cfg.use_csa`, so the off default is bit-identical. The decode
step carries a second, per-iteration state (the chunk ring) beside KDA's; the
`states` list holds `(kda_state, csa_state)` only on that arm.

mHC (`use_mhc`, arXiv:2512.24880) is the fourth residual arm, wired at the
LOOP BOUNDARY by the φ study's placement (2604.21106 §5.2: K=2 lanes *across
loops*, `docs/reviews/mhc-2026-09-30.md` §3.2): the loop carries `K =
mhc_streams` FULL-width residual lanes across iterations, `H_res` (doubly
stochastic, Eq. 9's Sinkhorn-Knopp) mixes them, `H_pre` aggregates them into
the body's input and `H_post` weights the body's output back onto each lane
(Eq. 3/7/8 — `mhc.py` transcribes the formulas with the burn line citations).
The readout is the SUM of the lanes (2409.19606 §2.1; the crate's concat is
its slicing layout, not ours). It replaces the SAME statement as ReZero and
LOOM; it COMPOSES with AttnRes (see "THE COMPOSITION"), so `config.validate`
refuses only the LOOM/ByteFlow pairs. `streams` are
replicated from `h0` and `h_ctx` = lanes + `iter_embed[i]` is the crate's
`x_l` (`mhc-2026-09-30.md` §3.4), so the A/B against ReZero differs in the
residual operator and nothing else.

THE COMPOSITION (`use_mhc=true, use_attnres=true`, legal since 2026-10-22).
The two arms act on the SAME statement — the loop-boundary residual write — but
on different axes: mHC owns the K-lane state (`H_res` mixes it, `H_pre` reads
it, `H_post` deposits into it), AttnRes owns WHICH UPDATE the write carries.
The order, one iteration:

1. `x_l = x + e_i` (the crate's `h_ctx`);
2. `ctx = H_pre x_l` — the K lanes mix down to the body's D-wide input;
3. the body computes `y = Attn + Engram + MoE` (unchanged);
4. AttnRes mixes the WRITE UPDATES `u_j` and writes exactly that mixture,
   `a = Σ_j w_j u_j`, with `w = S·softmax(q_i · RMSNorm(u_j))` (Eq. 1/2/3);
5. `H_post` deposits it: `x_{l+1} = H_res x_l + H_postᵀ a` (Eq. 3);
6. the readout is `Σ x_{l+1} / K` (2409.19606 §2.1).

The update set is `S = 2` sources: `u_0 = 0` — the token embedding's update,
because `H_res x_l` ALREADY carries the embedding (writing it again is the
double count a residual arm exists to avoid) — and `u_1 = y`, this iteration's
body. The pseudo-query scores the same tensors that are written (`RMSNorm(u_j)`
is the key, `u_j` the value), never a pre-mixed state.

**Why `S·softmax` and not Eq. 4's `Σw = 1`.** Eq. 4 is a convex mixture over
STATES; the composition can only deposit an UPDATE (the state accumulator is
`H_res`), and over updates the identity member of the family is the SUM. At
the zero-query init `softmax = 1/S` on every source, so `S·softmax = 1` on
every source and `a = 0 + y` — exactly the mHC-only write — for ANY number of
iterations. The departure is only the common scale `S`: the RELATIVE weights
are the paper's logits, and the softmax still redistributes write mass between
the two sources as the query trains. The identity is asserted bit-for-bit in
`tests/test_mhc_attnres.py`, and the two SINGLE-arm paths keep their own
pinned `state_dict` hashes so the composition cannot perturb them.

The readout above divides by `K` because our lanes are full-width copies of a
state the paper's `n` lanes re-sum (2409.19606 §2.1; `mhc.py`'s layout note).
The mHC-ONLY arm keeps its historical unnormalised sum, pinned bit-for-bit by
`test_mhc.py`; only the composition applies the paper's `1/K`.

RECURRENT DECODE (`recurrent_step`, the inference twin of the segmented path) —
`decode.recurrent_decode`'s per-byte step: the same body, over ONE position,
with the same per-iteration KDA state carried across BYTES. It is what makes
generation `O(1)` per byte instead of `O(window)`, and it is a reordering of the
one-shot forward rather than a second implementation of it (see the method).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..instrument import Counters, loud
from .csa import CompressedSelectiveAttention
from .engram import Engram, memory_floor_mix
from .experts import SparseMoE
from .fullattn import FullAttention
from .kda import KDAAttention
from .mhc import MHCMix
from .qat_cache import QuantCache
from .sparse import BlockSparseAttention
from .tsct import DenseLowRank, dense_qat_weight, make_projection, mm_mixed

__all__ = ["AttnResSlot", "LoopBlock", "depth_attend", "memory_floor_mix"]


def _rms_no_gain(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def masked_ce_mean(ce: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """`ce [N,1]`'s mean over the positions `loss_mask [b,t]` selects.

    The `doc_masking` arm's reduction: a mean over the SELECTED positions, so
    a padded position contributes neither loss nor gradient. The mask is
    guaranteed non-empty by the packed reader (every row holds a document or a
    document chunk), so this is two device ops and no host read — the 0-sync
    rule holds on the training path.
    """
    w = loss_mask.reshape(-1, 1).to(ce.dtype)
    return (ce * w).sum() / w.sum()


class AttnResSlot(nn.Module):
    """One pseudo-query per loop-iteration slot (arXiv:2603.15031 §5).

    Zero-initialised query × gain of ones: the first forward is the equal-weight
    average, so the arm starts where ReZero starts and the A/B is not a
    head start.
    """

    def __init__(self, d: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d, device=device))
        self.gain = nn.Parameter(torch.ones(d, device=device))


def depth_attend(
    sources: list[torch.Tensor], slot: AttnResSlot, eps: float, scale: float = 1.0
) -> torch.Tensor:
    """The AttnRes mixture: `scale · softmax_i(q_eff · RMSNorm(k_i))` over the sources.

    The score convention is the PAPER's — `q·RMSNorm(k)` with **no `1/√d`**
    (`burn_attnres::ScoreForm::Paper`, the default). The burn side shipped
    `d^-0.5` plus an L2 norm, together a `d`-fold logit compression, both
    silent; the fold `q_eff = γ ⊙ w_l` is fidelity fix F-F7 (the paper's
    affine norm, with its gain).

    `scale` is the composition's identity lever (`loopblock`'s module
    docstring, "THE COMPOSITION"): the untouched AttnRes arm calls it at its
    default `1.0`, which is Eq. 4's `Σα = 1` and is SKIPPED so that arm's
    bits do not move; the mHC+AttnRes write passes `scale = len(sources)`,
    which at the zero-query init makes every weight exactly `1` and therefore
    the mixture exactly the residual SUM of the same sources it scored.
    """
    q_eff = slot.gain * slot.query
    keys = torch.stack([_rms_no_gain(s, eps) for s in sources], dim=-2)  # [b,t,S,d]
    scores = (keys * q_eff).sum(-1)  # [b,t,S]
    w = F.softmax(scores, dim=-1)
    if scale != 1.0:  # not `w * 1.0`: the single-arm path must stay bit-identical
        w = w * scale
    return (w.unsqueeze(-1) * torch.stack(sources, dim=-2)).sum(-2)


class LoopBlock(nn.Module):
    def __init__(
        self,
        cfg,
        *,
        device: torch.device | str = "cpu",
        body_only: bool = False,
    ) -> None:
        """One loop body. `body_only` drops the LOOP-LEVEL, per-iteration slots.

        `DormouseModel(loop_blocks=N)` builds ONE full block (`loop`, the
        driver: it owns `iter_embed`/`attnres`/`*_gate` and the loop state) and
        `N-1` `body_only` blocks (`loop_extra`). A body owns everything its own
        iteration computes — norm, controller, attention arm, Engram, MoE bank,
        `mem_dense`, `out_proj`, `residual_scale`, and its own mHC mixer — but
        not the per-iteration slots: those are indexed by the ITERATION number
        (`iter_embed[i]`, `attnres[i]`, `msa_gate[i]`, ...), not by the block,
        so one set on the driver is the whole allocation. Building a second set
        on every body would allocate `max_iter` slots of which only those with
        `i % N == block` ever run — the config-lies defect this project refuses.
        """
        super().__init__()
        self.body_only = body_only
        d = cfg.d_model
        self.d = d
        self.max_iter = cfg.max_iter
        self.eps = cfg.norm_eps
        self.lam_max = cfg.engram_lam_max
        self.use_engram = cfg.use_engram
        self.use_attnres = cfg.use_attnres
        #: The compute dtype the bf16 plan names for the block's small mixed
        #: projections (controller, mem_dense) — the experts' own choice.
        self.compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        #: The DENSE path's switch (`tsct.dense_qat_project`, the SAME projector
        #: as the factor rung): every plain `nn.Linear` weight the block owns —
        #: the controller, `mem_dense`, and `out_proj` when it is a
        #: `DenseLinear` — is read through the annealed STE. `dense_qat_alpha`
        #: is the SAME alpha the factor ramp sets (`set_factor_qat_alpha`).
        self.dense_qat = cfg.dense_qat
        self.dense_qat_alpha = cfg.factor_qat_alpha
        #: The once-per-step projection cache for the two DENSE weights this
        #: block projects directly (controller, `mem_dense`); the linears go
        #: through `qat_linear`'s own attached cache.
        self._qat_cache = QuantCache() if cfg.dense_qat != "none" else None
        self.norm = nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
        # The three gates only. burn's controller ALSO emits the expert-blend
        # logits and its MoE "adds NO parameters"; the port's MoE owns a
        # dedicated router linear (the baked decision for this port), so the
        # controller carries three columns and no dead ones — a parameter with
        # no gradient is the defect class this project's counters exist for.
        self.controller = nn.Linear(2 * d, 3, bias=False, device=device)
        #: Which attention arm `use_kda` turns on (`attn_kind`: `kda` the
        #: compressed-state arm every number on record ran, `full` exact causal
        #: attention). `use_kda=false` keeps its own meaning — NO attention,
        #: the zero branch — so the off switch cannot silently become the full
        #: arm. The `full` arm is built only through this ONE line, and it is
        #: reached through the same `_attn` seam (`LoopBlock._attn`).
        self.attn_kind = cfg.attn_kind
        if cfg.attn_kind not in ("kda", "full"):
            loud(
                f"attn_kind={cfg.attn_kind!r} is not an attention arm. Known: 'kda' "
                f"(compressed state) and 'full' (exact causal attention). Escape: "
                f"name one of those."
            )
        if not cfg.use_kda and cfg.attn_kind != "kda":
            loud(
                f"attn_kind={cfg.attn_kind!r} with use_kda=false: the hard off switch "
                f"makes the attention output zeros, so this arm would never run. "
                f"Escape: use_kda=true, or attn_kind='kda'."
            )
        if not cfg.use_kda:
            self.attn = None
        elif cfg.attn_kind == "full":
            self.attn = FullAttention(cfg, device=device)
        else:
            self.attn = KDAAttention(cfg, device=device)
        # The second attention arm (owner's 1M-context directive, 2026-10-16):
        # MSA rides an ADDITIVE branch beside KDA, off by default; the ADAPTIVE
        # schedule (`adaptive_msa`) gives it a per-ITERATION share whose mean
        # is `msa_capacity_ratio` (the owner's 1/4), decided BY the model: one
        # learnable logit per iteration slot, softmax-normalized across the
        # slots and scaled to the mean — so the model picks WHERE the sparse
        # attention's compute is spent, the total budget is fixed.
        self.adaptive_msa = bool(getattr(cfg, "adaptive_msa", False))
        self.msa = BlockSparseAttention(cfg, device=device) if cfg.use_msa else None
        self.msa_capacity_ratio = float(getattr(cfg, "msa_capacity_ratio", 0.25))
        if self.msa is not None:
            assert 0.0 < self.msa_capacity_ratio <= 1.0, "msa_capacity_ratio in (0, 1]"
            if not body_only:
                self.msa_gate = nn.Parameter(torch.zeros(cfg.max_iter, device=device))
        # CSA (`model/csa.py`): the compressed, SELECTABLE KV memory, also an
        # ADDITIVE branch beside KDA and off by default. One learned gate per
        # iteration slot (the `msa_gate` pattern), shared by the fine top-k and
        # coarse dense levels, so the arm's contribution is per-pass and can be
        # switched off by training. Built only when the flag is on:
        # `use_csa=false` keeps the parameter tree — and every pinned
        # `state_dict` hash — exactly as it was.
        self.csa = CompressedSelectiveAttention(cfg, device=device) if cfg.use_csa else None
        if self.csa is not None:
            if not body_only:
                self.csa_gate = nn.Parameter(torch.zeros(cfg.max_iter, device=device))
        self.moe = SparseMoE(cfg, device=device)
        self.engram = Engram(cfg, device=device) if cfg.use_engram else None
        # The dense half of the memory branch's mixture — the `(1-g)·v_t` term.
        # Built only WITH the branch it feeds: `use_engram=false` never reads
        # it, so an unconditional build is 1 049 600 dead parameters at d=1024
        # (no `.grad`, but in `parameters()`, `param_count` and the optimizer).
        self.mem_dense = nn.Linear(d, d, device=device) if cfg.use_engram else None
        #: The readout (`out_proj_rank`): TSCT at `cfg.rank` by default, or at
        #: the flag's rank when on; the dense rung's full `d×d` becomes a dense
        #: two-stage `d → r → d`. `0` is the projection exactly as before.
        if cfg.out_proj_rank and not cfg.use_tsct:
            self.out_proj = DenseLowRank(
                d,
                cfg.out_proj_rank,
                d,
                dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
                dense_qat=cfg.dense_qat,
                dense_qat_alpha=cfg.factor_qat_alpha,
                device=device,
            )
        else:
            self.out_proj = make_projection(
                d, d, cfg, rank=cfg.out_proj_rank or cfg.rank, device=device
            )
        self.iter_embed = (
            None if body_only else nn.Parameter(torch.zeros(cfg.max_iter, d, device=device))
        )
        self.residual_scale = nn.Parameter(torch.ones(1, device=device))
        self.attnres = (
            nn.ModuleList([AttnResSlot(d, device) for _ in range(cfg.max_iter)])
            if cfg.use_attnres and not body_only
            else None
        )
        # LOOM (`2610.01153`): one learned scalar per iteration slot on the
        # RMS-normalized write, ZERO-INIT — at init the arm is the identity and
        # adds an exact zero, so pure-CE parity holds and the gate has to be
        # MOVED by gradient before the arm is anything (module docstring). Built
        # only when the flag is on, so a `loom = false` model's parameter tree —
        # and every preset count pinned off it — is exactly today's.
        self.loom = bool(getattr(cfg, "loom", False))
        self.loom_gate = (
            nn.Parameter(torch.zeros(cfg.max_iter, device=device))
            if self.loom and not body_only
            else None
        )
        # mHC (`2512.24880`, `mhc.py`): K full-width residual lanes across the
        # loop, K=2 only (the A/B'd configuration — App. A.1 ships 4, our φ
        # evidence and 2409.19606 Tab. 1's knee are 2). Built only when the flag
        # is on, so `use_mhc=false` keeps today's parameter tree exactly. It IS
        # per block (each body owns the residual operator that deposits its
        # write); the lane STATE is not — it lives in `_window` across the loop.
        self.mhc = (
            MHCMix(d, cfg.mhc_streams, device=device) if getattr(cfg, "use_mhc", False) else None
        )
        #: The DISTINCT bodies this driver cycles over (`DormouseModel` sets it
        #: to `(loop, *loop_extra)`); `None` means this block is the whole loop.
        #: Read by `_window`, so EVERY caller — one-shot, segmented, recurrent
        #: decode and prefill — resolves the same body per iteration.
        self._blocks = None

    def forward(
        self,
        h0: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        lm_head: nn.Module | None = None,
        counters: Counters | None = None,
        depth: int | None = None,
        segment_len: int = 0,
        halt: nn.Module | None = None,
        halt_min_iter: int = 1,
        collect_hidden: bool = False,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Counters]:
        """`h0 [b,t,d]` → (`mean readout [b,t,d]`, SUM of per-iteration mean CEs, counters).

        The CE is a SUM over iterations, not a mean: the model divides by the
        number of terms once it has added the final readout's own CE (see
        `model.DormouseModel.forward` for why that term exists).

        `depth=None` runs `max_iter` iterations (ADR-0013's fixed depth,
        unchanged); `depth=T` runs exactly `T` and the readout mean divides by
        that `T`, never by `max_iter` — the ADAPTIVE DEPTH arm's contract.
        `counters["depth"]` accumulates the depth each forward actually ran.

        `segment_len=0` (the default; the only value any number on record ran)
        is the ONE-SHOT path. `0 < segment_len <= t` is the SEGMENTED path
        (`_forward_segmented`), which is a MEMORY decision and not a model
        change: the two agree to fp32 round-off (`tests/test_longctx.py`), and
        `segment_len = t` — one segment — is bit-identical to the one-shot path
        (that is the pin the default rests on).
        """
        counters = Counters() if counters is None else counters
        b, t, d = h0.shape
        assert d == self.d, f"loop block width {self.d} != input {d}"
        T = self.max_iter if depth is None else depth
        assert 1 <= T <= self.max_iter, f"depth {T} outside 1..={self.max_iter}"
        counters.count("depth", T)
        if segment_len > t:
            loud(
                f"segment_len={segment_len} > the window {t}: a segment longer than "
                f"the sequence is the sequence, so the flag would buy nothing while "
                f"the log line reported it on. Escape: a segment_len <= {t} (and "
                f"< seq_len, or the config layer refuses it as pointless)."
            )
        if segment_len > 0 and t % segment_len != 0:
            loud(
                f"segment_len={segment_len} does not divide the window {t}: a shorter "
                f"last segment would weight its positions more than the others (the "
                f"per-iteration CE is a MEAN per segment), so the loss would stop "
                f"being the one-shot loss — and segmentation is allowed to change the "
                f"memory, never the objective. Escape: a divisor of {t}."
            )
        if segment_len > 0:
            if loss_mask is not None:
                loud(
                    f"loss_mask with segment_len={segment_len}: the segmented forward "
                    f"carries the KDA state across its cuts, so a document boundary "
                    f"inside the window is not a segment boundary. Escape: "
                    f"segment_len=0."
                )
            return self._forward_segmented(
                h0, input_ids, targets, lm_head, counters, T, segment_len
            )
        out, rec, per_iter, _ = self._window(
            h0,
            input_ids,
            targets,
            lm_head,
            counters,
            T,
            carry=False,
            halt=halt,
            halt_min_iter=halt_min_iter,
            collect_hidden=collect_hidden,
            loss_mask=loss_mask,
        )
        # The per-iteration CEs as a LIST on the module (host floats ONLY on an
        # eval-style read; see `model.DormouseModel.forward(breakdown=True)`) —
        # the depth-confounded headline metric (ΣCE + CE_final)/(T+1) averages
        # a DIFFERENT number of terms per depth, so cross-depth comparisons from
        # the headline alone are arithmetic artifacts (found 2026-10-16: every
        # trained arm read "deeper is worse" because the average gains another
        # mid-quality term with each iteration).  No sync here: tensors stay.
        self._per_iter_ce = per_iter
        return out, (None if targets is None else rec), counters

    def _moe_branch(self, normed: torch.Tensor, counters: Counters) -> torch.Tensor:
        """The MoE write at the SEAM: cast `normed` to the bank dtype ONCE, cast back to fp32.

        The residual stream is fp32 (the fp32-masters policy) and the MoE's
        INTERIOR is the bank dtype already (the router, the experts and the
        kernels all compute in `dt`). Handing the fp32 rows straight to
        `SparseMoE` charged every routed arm a fused-read chain it did not need:
        an fp32 `index_select` (4·R·d), the cast's fp32 read (4·R·d), the bf16
        write (2·R·d) and the kernel's bf16 read (2·R·d) — 12·R·d in total.
        ONE cast here (2·R·d write) makes the gather `x[owner]` alone (2·R·d
        write + 2·R·d kernel read): 4·R·d = 201 326 592 B (192 MiB) — 8·R·d
        saved per routed pass, 402 653 184 B (384 MiB) at
        `medium-b24·s1024`, where R·d = 49 152·1 024 and the old chain was
        603 979 776 B (576 MiB). Those three numbers are the delta of the
        CHAIN, not a fraction of its 16·R·d total (which includes the source
        read both seams pay).

        `dt` is `SparseMoE._bank_dtype()` — `bf16` when `cfg.bf16`, fp32
        otherwise — and the SHARED expert is covered by the same call:
        `SparseMoE.forward` hands this `x` to `self.shared` as well. The cast
        back is the block's contract (the residual write and `w_ffn` are
        fp32), so callers above see today's dtype. With `bf16=false` both
        casts are no-ops on the same tensors: the fp32 seam is byte-for-byte
        the old forward.
        """
        dt = self.moe._bank_dtype()
        x = normed.reshape(-1, self.d)
        return self.moe(x.to(dt), counters=counters).to(torch.float32)

    def _window(
        self,
        h0: torch.Tensor,
        input_ids: torch.Tensor | None,
        targets: torch.Tensor | None,
        lm_head: nn.Module | None,
        counters: Counters,
        T: int,
        *,
        carry: bool,
        states: list | None = None,
        drop: int = 0,
        recurrent: bool = False,
        halt: nn.Module | None = None,
        halt_min_iter: int = 1,
        collect_hidden: bool = False,
        loss_mask: torch.Tensor | None = None,
        blocks: tuple[LoopBlock, ...] | list[LoopBlock] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list]:
        """`T` iterations over ONE window — a segment, or the whole sequence.

        The body of the loop, in one place, called by both paths: `carry=False`
        is the one-shot call (`states=None`, `drop=0`, the attention arm called
        through its one-shot `forward`), `carry=True` is the segmented one
        (`states` = the per-iteration state entering this window, `drop` = the
        positions of `input_ids` that belong to the window BEFORE this one).

        `blocks` is the DISTINCT body per iteration (`loop_blocks`): iteration
        `i` runs `blocks[i % len(blocks)]`. `None` resolves `self._blocks` (set
        by `DormouseModel`), then `(self,)` — so a lone `LoopBlock` keeps
        today's one-body loop byte for byte. The loop STATE (`h`, the mHC
        lanes, the AttnRes history) is shared across bodies; only the body's
        weights change.

        `h0`/`input_ids`/`targets` are this window's slice; `input_ids` carries
        `drop` extra LEADING positions when `drop > 0` (see `_memory`).
        Returns `(mean readout, Σ per-iteration mean CE, per-iteration CEs, the
        per-iteration states leaving this window)`.
        """
        if blocks is None:
            blocks = self._blocks if self._blocks is not None else (self,)
        bodies: tuple[LoopBlock, ...] = tuple(blocks)
        b, t, d = h0.shape
        h = h0
        h0_flat = h0.reshape(b * t, d)
        #: The COMPOSITION (`use_mhc` + `use_attnres`, module docstring): mHC
        #: owns the K-lane state, AttnRes owns the update `H_post` deposits.
        #: A local, not a cached field, so `loop.mhc = None` (the test's way of
        #: removing one mechanism) cannot leave a stale dispatch behind.
        compose = self.use_attnres and self.mhc is not None
        if compose:
            # Once per window: the spy that says the pair ran at all. The
            # per-iteration counters below say it ran every iteration.
            counters.count("mhc_attnres_forward")
        # mHC's K lanes (`mhc.py`): replicated from h0 (2409.19606 §2.1), each
        # the FULL residual width. `None` is the flag-off path, byte-for-byte.
        streams = (
            h0.unsqueeze(-2).expand(b, t, self.mhc.n_streams, d) if self.mhc is not None else None
        )
        out_acc = torch.zeros_like(h0)
        rec = torch.zeros((), dtype=torch.float32, device=h0.device)
        per_iter: list[torch.Tensor] = []
        #: The per-iteration states the halt head reads (`model.py`'s loss);
        #: collected only for a training forward with a head. The references
        #: are graph nodes `out_acc` already keeps, so no extra storage, and
        #: the list is replaced on every forward.
        hidden: list[torch.Tensor] = []
        ran = 0
        # The AttnRes-ONLY history (`[h0]` + every body output). Unused by the
        # composition, which re-scores two write updates per iteration instead
        # of the whole depth history: mHC's lanes already hold that history.
        sources: list[torch.Tensor] = [h0] if self.use_attnres and not compose else []
        new_states: list = []
        for i in range(T):
            counters.count("iter")
            ran = i + 1
            #: The DISTINCT body for this iteration (`loop_blocks`): the
            #: weights are per body; the state (`h`, `streams`, `sources`) and
            #: the per-iteration slots on the driver are the loop's.
            blk = bodies[i % len(bodies)]
            if streams is not None:
                # x_l = h_ctx: the iteration embedding joins the LANES, and
                # H_pre (applied by the caller, `mhc.py`) aggregates them into
                # the body's input — the crate's `h_ctx` (`mhc-2026-09-30.md`
                # §3.4), with Eq. 3's H_pre moved out of `forward`.
                streams = streams + self.iter_embed[i].view(1, 1, 1, d)
                ctx = blk.mhc.pre_mix(streams)
                counters.count("mhc")
            else:
                ctx = h + self.iter_embed[i].view(1, 1, d)
            normed = blk.norm(ctx)
            ctrl = torch.cat([ctx.reshape(b * t, d), h0_flat], dim=-1)
            # MAGMA fix: fp32 masters with fp32 activations would run the slow
            # fp32 lane; `mm_mixed` casts to the compute dtype per forward.
            # `dense_qat_project` is the identity when the arm is off (same
            # parameter object), so the default path is byte-for-byte today's.
            raw = mm_mixed(
                ctrl,
                dense_qat_weight(
                    blk.controller.weight,
                    blk.dense_qat,
                    blk.dense_qat_alpha,
                    cache=blk._qat_cache,
                ),
                blk.compute_dtype,
                counters,
            )
            w_attn = torch.sigmoid(raw[:, 0:1])
            w_mem = torch.sigmoid(raw[:, 1:2])
            w_ffn = torch.sigmoid(raw[:, 2:3])

            if blk.attn is not None:
                # A CSA run carries `(kda_state, csa_state)` per iteration in
                # `states`; the KDA arm sees only its half.
                attn_state = None if states is None else states[i]
                if blk.csa is not None and isinstance(attn_state, tuple):
                    attn_state = attn_state[0]
                attn_out, st = blk._attn(
                    normed,
                    counters,
                    carry,
                    attn_state,
                    recurrent=recurrent,
                )
                attn = attn_out.reshape(b * t, d) * w_attn
            else:
                attn = torch.zeros(b * t, d, dtype=torch.float32, device=h0.device)
            if blk.csa is not None:
                prev = None if states is None else states[i]
                csa_state = prev[1] if isinstance(prev, tuple) else None
                if recurrent:
                    csa_out, csa_st = blk.csa.step_state(normed, csa_state, counters=counters)
                elif carry:
                    csa_out, csa_st = blk.csa.forward_state(normed, csa_state, counters=counters)
                else:
                    csa_out, csa_st = blk.csa(normed, counters=counters), None
                # The `msa_gate` pattern: one learned scalar per iteration slot,
                # times the controller's attention gate, on top of KDA's output.
                attn = attn + csa_out.reshape(b * t, d) * w_attn * torch.sigmoid(self.csa_gate[i])
                if carry or recurrent:
                    assert blk.attn is not None, "CSA is additive beside KDA (config refuses)"
                    st = (st, csa_st)
            if blk.attn is not None:
                new_states.append(st)
            if blk.msa is not None:
                # The ADAPTIVE share: softmax over the iteration gates × max_iter
                # = a per-slot share whose MEAN over slots is 1; × the arm's
                # capacity ratio (the owner's 1/4) it becomes the slot's budget.
                # The ROUNDED share picks the block count (compute follows), and
                # the SOFT share multiplies the branch's output below, so the
                # gates receive gradient — the model learns WHERE to spend the
                # 25%, and a zero share can switch a slot off entirely.
                if self.adaptive_msa and self.max_iter > 1:
                    share = torch.softmax(self.msa_gate, dim=0) * self.max_iter
                    counters.count("msa_iter_gate")
                    msa_out = blk.msa.forward_k(
                        normed, counters, float(share[i].detach()) * self.msa_capacity_ratio
                    )
                    if msa_out is not None:
                        attn = attn + msa_out.reshape(b * t, d) * w_attn * share[i]
                else:
                    msa_out = blk.msa(normed, counters=counters)
                    if msa_out is not None:
                        attn = attn + msa_out.reshape(b * t, d) * w_attn

            if blk.engram is not None and input_ids is not None:
                mem = blk._memory(normed, input_ids, drop, counters).reshape(b * t, d)
                dense = (
                    (
                        mm_mixed(
                            normed.reshape(b * t, d),
                            dense_qat_weight(
                                blk.mem_dense.weight,
                                blk.dense_qat,
                                blk.dense_qat_alpha,
                                cache=blk._qat_cache,
                            ),
                            blk.compute_dtype,
                            counters,
                        )
                        + blk.mem_dense.bias.to(blk.compute_dtype)
                    )
                    .to(torch.float32)
                    .reshape(b * t, d)
                )
                branch = memory_floor_mix(mem, dense, w_mem, self.lam_max)
            else:
                branch = torch.zeros(b * t, d, dtype=torch.float32, device=h0.device)

            ffn = blk._moe_branch(normed, counters) * w_ffn
            y = (attn + branch + ffn).reshape(b, t, d)

            if self.loom:
                # LOOM (2610.01153): per-loop router conditioning + variance
                # control in ONE place — the RMS-normalized write is the arm's
                # own signal (unit scale, so it cannot grow with depth) and the
                # per-slot gate decides how much of it this iteration writes.
                # `gate[i] == 0` adds an exact zero: the at-init forward IS the
                # `loom = false` forward, bit for bit.
                counters.count("loom")
                y = y + _rms_no_gain(y, self.eps) * self.loom_gate[i].view(1, 1, 1)

            if compose:
                # THE COMPOSITION (module docstring): AttnRes mixes the write
                # UPDATES — `u_0 = 0` (the embedding's update: H_res already
                # carries it) and `u_1 = y` (this iteration's body) — and
                # `H_post` deposits EXACTLY that mixture. The query scores the
                # same tensors that are written. `scale = 2` is the identity
                # lever: at the zero-query init every weight is 1, so the
                # update is `y` and the composition's write is the mHC-only
                # write bit-for-bit.
                assert blk.mhc is not None, "compose => mhc"
                counters.count("attnres_write")
                counters.count("mhc_attnres_write")
                updates = [torch.zeros_like(y), y]
                update = depth_attend(updates, self.attnres[i], self.eps, scale=float(len(updates)))
                streams = blk.mhc(streams, update)
                # 2409.19606 §2.1: our K full-width lanes re-sum to K times the
                # state, where the paper's K width-C lanes re-sum to C. The
                # mHC-ONLY arm keeps its historical unnormalised sum (pinned);
                # the composition applies the paper's 1/K.
                h = streams.sum(dim=-2) / blk.mhc.n_streams
            elif self.use_attnres:
                counters.count("attnres_write")
                sources.append(y)
                h = depth_attend(sources, self.attnres[i], self.eps)
            elif blk.mhc is not None:
                # Eq. 3's residual: H_res x_l + H_postᵀ F(H_pre x_l). The
                # readout is the SUM of the lanes (2409.19606 §2.1); the crate's
                # concat is its D/n-slicing layout, not this one (`mhc.py`).
                streams = blk.mhc(streams, y)
                h = streams.sum(dim=-2)
            else:
                counters.count("rezero_write")
                h = ctx + y * blk.residual_scale

            step_out = blk.out_proj(h.reshape(b * t, d)).reshape(b, t, d)
            out_acc = out_acc + step_out
            if collect_hidden:
                hidden.append(step_out)
            if targets is not None:
                assert lm_head is not None, "per-iteration CE needs the lm_head"
                logits = lm_head(step_out.reshape(b * t, d).to(torch.float32))
                ce = -F.log_softmax(logits, dim=-1).gather(1, targets.reshape(-1, 1))
                # `doc_masking`: the mean over the SELECTED positions only (the
                # `None` default is the historical `.mean()`, bit for bit).
                term = ce.mean() if loss_mask is None else masked_ce_mean(ce, loss_mask)
                rec = rec + term
                per_iter.append(term)
            if halt is not None and ran >= halt_min_iter:
                # The HALT DECISION: one scalar from this iteration's readout,
                # read to the host and compared to 0.5. Inference only — the
                # trainer never passes `halt`, so a training step adds no host
                # read (ADR-0018 rule 2). Reading here is what makes the skip
                # REAL: the iterations below are never launched.
                if float(torch.sigmoid(halt(step_out))) > 0.5:
                    counters.count("halt_stop")
                    break
        if halt is not None and ran < T:
            counters.count("halt_skipped", T - ran)
        self._depth_ran = ran
        self._per_iter_hidden = hidden
        return out_acc / ran, rec, per_iter, new_states

    def recurrent_step(
        self,
        h0: torch.Tensor,
        input_ids: torch.Tensor,
        states: list,
        *,
        drop: int = 0,
        counters: Counters | None = None,
    ) -> tuple[torch.Tensor, list]:
        """ONE position through the `T` iterations, KDA state carried — the decode step.

        `h0 [b, 1, d]` is this position's embedding, `input_ids [b, drop+1]` the
        byte window its causal n-grams read (`_memory`'s left context), `states`
        the per-iteration state leaving the PREVIOUS position. Returns
        `(readout [b, 1, d], the per-iteration state leaving this one)`.

        The same body as `_window` — one call site per arm, `self._attn` — with
        the attention arm's single-token step (`KDAAttention.step_state`) in place
        of the chunked pass. Only the ITERATION ORDER differs (position-major here,
        iteration-major in `_window`), and that is not a model change: iteration
        `i` at position `p` reads `h` from iteration `i-1` at the SAME position
        and the state from iteration `i` at `p-1`, both already produced in the
        position-major order; every other arm in the body is per position. The
        one place it can differ is a capacity-dropping MoE, whose drop set is a
        function of the window's composition (`decode.recurrent_decode` counts it).
        """
        counters = Counters() if counters is None else counters
        out, _, _, new_states = self._window(
            h0,
            input_ids,
            None,
            None,
            counters,
            self.max_iter,
            carry=True,
            recurrent=True,
            states=states,
            drop=drop,
        )
        return out, new_states

    def _attn(
        self,
        normed: torch.Tensor,
        counters: Counters,
        carry: bool,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        *,
        recurrent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None]:
        """One iteration's attention call → (`[b,t,d]`, the state leaving it).

        One call site for both paths, so the segmented arm cannot drift from the
        one-shot one: `carry=False` is the arm's `forward` (the call every
        number on record made) and returns no state; `carry=True` is
        `forward_state`, whose state is the edge into the next segment;
        `recurrent=True` (the decode step, `recurrent_step`) is the arm's
        `step_state`, which carries the state across BYTES rather than segments.

        The state is PER ITERATION SLOT: the loop body is shared, but each of its
        `T` passes is a separate pass over the sequence, so each needs its own
        boundary state — one state for the whole block would mix the passes,
        which is the same defect as sharing one hidden state across them.
        """
        assert self.attn is not None
        assert not (self.attn_kind == "full" and (recurrent or carry)), (
            "attn_kind='full' with a state-carrying attention call "
            f"(recurrent={recurrent}, carry={carry}): exact attention has no O(1) "
            "boundary state — segment_len would have to carry every preceding "
            "segment's K/V, and the recurrent decode would have to carry a growing "
            "cache, which is the O(window) memory the compressed arm exists to "
            "avoid. Escape: attn_kind='kda', or segment_len=0 and the window "
            "forward (--no-recurrent decode)."
        )
        if recurrent:
            assert carry, "the recurrent step carries the state by construction"
            return self.attn.step_state(normed, state, counters=counters)
        if not carry:
            return self.attn(normed, counters=counters), None
        return self.attn.forward_state(normed, state, counters=counters, carry=True)

    def _memory(
        self, normed: torch.Tensor, input_ids: torch.Tensor, drop: int, counters: Counters
    ) -> torch.Tensor:
        """The Engram read for one window, with the left context its n-grams need.

        `drop=0` (the one-shot path) is the call the arm has always made.
        `drop>0` is the segmented path: the memory's keys are CAUSAL n-grams
        (`engram.raw_keys`: position `e` reads bytes `max(0, e-order)..e`), so a
        segment that starts at byte `p` needs the `drop = min(max_order-1, p)`
        bytes BEFORE it or the first positions of every segment would read a
        different key than the one-shot forward — a silent model change, which
        is exactly what the parity gate exists to catch.

        The extra positions' OUTPUTS are discarded, and that is sound because
        the arm's gate math is elementwise per position (`engram._read`: the
        only axis it mixes is the n-gram order). Their `hidden` rows are
        therefore padding and never read — `F.pad` puts zeros there, and
        `_read` reads them only to produce values this function slices off.

        `engram_rows` on the counter line counts those context keys too (they
        ARE read from the table): at orders [2,3,4] that is ≤ 3 extra keys per
        segment per order, i.e. ≤ 3/segment_len of the rows.
        """
        if drop == 0:
            return self.engram(normed, input_ids, counters)  # type: ignore[misc]
        assert input_ids.shape[1] == normed.shape[1] + drop, "the window slice is misaligned"
        hidden = F.pad(normed, (0, 0, drop, 0))
        return self.engram(hidden, input_ids, counters)[:, drop:]  # type: ignore[misc]

    def _forward_segmented(
        self,
        h0: torch.Tensor,
        input_ids: torch.Tensor | None,
        targets: torch.Tensor | None,
        lm_head: nn.Module | None,
        counters: Counters,
        T: int,
        segment_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Counters]:
        """The segmented training forward: `t // segment_len` checkpointed windows.

        What is carried and what is not:

        * **The KDA state is carried, per iteration slot** (`_attn`), as a real
          graph edge — segment `s`'s state is an output of segment `s-1`'s
          checkpointed body, so the backward recomputes segment `s-1` and the
          gradient reaches its parameters. That is StateFlow's boundary state +
          gradient propagation (arXiv:2608.06838), and it is the reason this is
          not truncated BPTT: nothing is detached and no window is a fresh
          sequence.
        * **`h` is NOT carried and must not be**: it is a per-position residual
          stream, and iteration `i` of segment `s` reads iteration `i-1` of the
          SAME positions, which this window produced itself. Only the
          cross-position state (attention) crosses the cut.
        * **The objective is unchanged.** Each segment returns its own
          `out_acc / T` (concatenated) and its own per-iteration MEAN CE; with
          equal segments the mean of those segment means is the window's own
          mean, so the driver divides their sum by the segment count and
          `(Σ CE_i + CE_final)/(T+1)` is the one-shot loss — the trainer's loss
          code is untouched. (Equal segments are not an assumption: a
          `segment_len` that does not divide the window is refused, which is
          exactly what the divisor rests on.)

        `checkpoint(use_reentrant=False)` per segment is the memory win: the
        segment's activations are recomputed in the backward instead of stored,
        and the boundary state is what stays alive. Two consequences that are
        not optional here:

        * the body's counters are gathered in a per-segment `Counters` that is
          merged once, because the recomputation runs the body a second time
          (`checkpoint` calls the function again in the backward) — counting
          into the run's own object would report `2·S·T` iterations on the log
          line. The recomputation's copy is discarded by checkpoint.
        """
        assert self.attn_kind != "full", (
            "segment_len with attn_kind='full': exact attention has no O(1) boundary "
            "state, so a segment would have to carry every preceding segment's K/V "
            "(O(window) per iteration slot) — the memory this flag exists to save. "
            "Escape: segment_len=0, or attn_kind='kda'."
        )
        assert self.msa is None, (
            "segment_len with use_msa: the block-sparse arm selects its blocks "
            "INSIDE one window, so segmenting it would change the attention "
            "pattern itself (a boundary block can no longer be selected) - that is "
            "a model change wearing a memory flag. Escape: use_msa=false, or "
            "segment_len=0."
        )
        assert self.csa is None, (
            "segment_len with use_csa: the CSA state is a chunk ring carried across "
            "BYTES, and the boundary rule for a half-full chunk at a segment cut is "
            "undecided (carrying it is a model change, dropping it silently forgets "
            "the segment). Escape: use_csa=false, or segment_len=0. The config layer "
            "refuses this pair too."
        )
        assert self.moe.topk == 0 or self.moe.capacity == 0, (
            "segment_len with moe_topk>0 at moe_capacity>0: the capacity-padded sparse "
            "dispatch drops the rows past `ceil(moe_capacity · tokens · k / E)` per "
            "expert, so which tokens get an expert depends on the OTHER tokens in the "
            "same forward - cutting the window changes that set (measured 0.39 on the "
            "readout at moe_capacity=1.0, 2026-10-17) and the forward stops being the "
            "one-shot forward. Escape: moe_capacity=0 (DROPLESS: every assignment "
            "reaches its expert and the routing is per position), moe_topk=0 (the "
            "dense blend), or segment_len=0."
        )
        b, t, d = h0.shape
        assert t % segment_len == 0, f"segment_len={segment_len} does not divide {t}"
        segments = t // segment_len
        counters.count("loop_segments", segments)
        #: The left context the Engram's causal n-grams need (orders [2,3,4] -> 3).
        ctx = 0 if self.engram is None else max(self.engram.orders) - 1
        states: list = [None] * T
        outs: list[torch.Tensor] = []
        rec_sum = torch.zeros((), dtype=torch.float32, device=h0.device)
        per_iter_sum: list[torch.Tensor] | None = None
        for s in range(segments):
            lo, hi = s * segment_len, (s + 1) * segment_len
            drop = min(ctx, lo)
            ids = None if input_ids is None else input_ids[:, lo - drop : hi]
            tgts = None if targets is None else targets[:, lo:hi]
            h_seg = h0[:, lo:hi]

            def body(h_seg, ids, tgts, *states, _drop=drop):
                """The checkpointed unit. Its counters are its own (see above)."""
                seg_counters = Counters()
                out, rec, per_iter, states_out = self._window(
                    h_seg,
                    ids,
                    tgts,
                    lm_head,
                    seg_counters,
                    T,
                    carry=True,
                    states=list(states),
                    drop=_drop,
                )
                return out, rec, per_iter, seg_counters, *states_out

            out_seg, rec_seg, per_iter, seg_counters, *states = checkpoint(
                body, h_seg, ids, tgts, *states, use_reentrant=False
            )
            counters.merge(seg_counters)
            if hasattr(seg_counters, "why"):
                counters.why = seg_counters.why  # type: ignore[attr-defined]
            outs.append(out_seg)
            rec_sum = rec_sum + rec_seg
            if per_iter:
                if per_iter_sum is None:
                    per_iter_sum = list(per_iter)
                else:
                    per_iter_sum = [a + c for a, c in zip(per_iter_sum, per_iter, strict=True)]
        assert len(outs) == segments, f"{len(outs)} segments != {segments}"
        out = torch.cat(outs, dim=1) if segments > 1 else outs[0]
        # The per-segment CEs are MEANS over each segment, so their sum is
        # `segments` × the window's mean — the divisor the parity gate caught
        # (the logits matched at 2 segments while the loss was exactly 2×; a
        # segment-local mean is not a share of the window's mean).
        rec = None if targets is None else rec_sum / segments
        self._per_iter_ce = [] if per_iter_sum is None else [v / segments for v in per_iter_sum]
        return out, rec, counters
