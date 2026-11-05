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

from .. import stepaudit
from ..instrument import Counters, loud
from ..losses import fixed_cross_entropy, logit_softcap
from . import fused_kda, fused_residual
from .csa import CompressedSelectiveAttention
from .engram import Engram, memory_floor_mix
from .experts import SparseMoE
from .fullattn import FullAttention
from .kda import KDAAttention
from .layerrope import LayerRoPEGains, LayerRoPENorm, LayerRoPEPaperGains
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
    q_eff = (slot.gain * slot.query).float()
    # `.float()` is the identity on fp32 sources (the OFF path's bits), and it
    # is the FP32 REDUCTION the census requires when the residual is stored
    # bf16: the key norm and the softmax are not allowed to run in bf16.
    keys = torch.stack([_rms_no_gain(s.float(), eps) for s in sources], dim=-2)  # [b,t,S,d]
    scores = (keys * q_eff).sum(-1)  # [b,t,S]
    w = F.softmax(scores, dim=-1)
    if scale != 1.0:  # not `w * 1.0`: the single-arm path must stay bit-identical
        w = w * scale
    return (w.unsqueeze(-1) * torch.stack([s.float() for s in sources], dim=-2)).sum(-2)


class LoopBlock(nn.Module):
    def __init__(
        self,
        cfg,
        *,
        device: torch.device | str = "cpu",
        body_only: bool = False,
        layerrope: LayerRoPEGains | None = None,
        layerrope_depth: int = 0,
        layerrope_site: int = 0,
        layerrope_paper: LayerRoPEPaperGains | None = None,
        paper_block: int = 0,
        paper_position: int = 0,
    ) -> None:
        """One loop body. `body_only` drops the LOOP-LEVEL, per-iteration slots.

        `layerrope` is v1's shared `(gamma_shared, amp, theta)` table;
        `layerrope_paper` is the paper form's four sites for THIS body
        (`paper_block`) at the network position `paper_position` (read only
        when `layerrope_index="position"`). Exactly one of them may be set:
        v1 REPLACES the block's norm gain, the paper form replaces the norm
        itself (its read gain is the norm's affine) and adds two write gains.
        Both `None` builds the plain `nn.RMSNorm` bit for bit.

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
        #: `iter_checkpoint` (TrainConfig): wrap EACH iteration's body in
        #: `checkpoint(use_reentrant=False)` so only the iteration's INPUT state
        #: survives the forward. Off is the historical graph, bit for bit; the
        #: trainer sets it once (`DormouseModel.set_iter_checkpoint`) and a
        #: hand-built model defaults to off.
        self.iter_checkpoint = False
        #: `activation_recompute` (`TrainConfig`): "none" | "full" | "selective".
        #: `full` is `iter_checkpoint`'s whole-iteration checkpoint; `selective`
        #: keeps the read half (q/k/v/g projections, KDA/CSA boundary state) and
        #: recomputes only the write half (MoE + out_proj). Set once by
        #: `DormouseModel.set_recompute`; "none" with `iter_checkpoint` still on
        #: is the historical bool, mapped in `_window`.
        self.recompute = "none"
        self.eps = cfg.norm_eps
        self.lam_max = cfg.engram_lam_max
        self.use_engram = cfg.use_engram
        self.use_attnres = cfg.use_attnres
        #: The compute dtype the bf16 plan names for the block's small mixed
        #: projections (controller, mem_dense) — the experts' own choice.
        self.compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        #: The STORAGE dtype of the residual stream (`cfg.residual_dtype`).
        #: fp32 is the default and OFF is the identity: `_res` returns its
        #: argument unchanged, so the pre-flag forward is byte-for-byte today's.
        #: bf16 stores `h`, the mHC lanes, `ctx`, the normed reads, the branch
        #: writes and `y` at half the bytes; every arm that needs fp32 widens
        #: its own input and the reductions/state stay fp32 (see
        #: `scripts/precision_census.py`, which is the change list's source).
        res_name = getattr(cfg, "residual_dtype", "fp32")
        if res_name not in ("fp32", "bf16"):
            loud(
                f"residual_dtype={res_name!r} is not a dtype. Known: fp32, bf16. "
                f"Escape: one of those (the value names the loop's residual STORAGE "
                f"dtype, not the compute dtype — `bf16` is that switch)."
            )
        self.residual_dtype = torch.bfloat16 if res_name == "bf16" else torch.float32
        if self.residual_dtype is torch.bfloat16:
            # The runtime seam a hand-built model reaches (`config.validate`
            # owns the same refusals; ADR-0019 says both must hold). Name the
            # fp32-only kernel, never fall back to it silently.
            if getattr(cfg, "fused_residual", False):
                loud(
                    "residual_dtype='bf16' with fused_residual=true: the "
                    "fused_residual Triton chain (_Pre/_Write + the two VJPs) is "
                    "fp32-only. Escape: fused_residual=false, or residual_dtype='fp32'."
                )
            if getattr(cfg, "fused_epilogue", False):
                loud(
                    "residual_dtype='bf16' with fused_epilogue=true: the "
                    "fused_epilogue Triton kernel (epilogue_read) is fp32-only. "
                    "Escape: fused_epilogue=false, or residual_dtype='fp32'."
                )
            if getattr(cfg, "fused_kda_tail", False):
                loud(
                    "residual_dtype='bf16' with fused_kda_tail=true: the "
                    "fused_kda_tail combine kernel is fp32-only. Escape: "
                    "fused_kda_tail=false, or residual_dtype='fp32'."
                )
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
        #: LayerRoPE (`model/layerrope.py`): the block's ONE pre-norm gain is
        #: `gamma_{P+i}` — the same site the attention and the MoE read, which is
        #: this repo's "attention pre-norm + FFN pre-norm" already. `None` builds
        #: the plain `nn.RMSNorm` bit for bit; otherwise `_layerrope_depth` is the
        #: PRELUDE offset `P`, and `_window` adds the ACTUAL iteration number.
        self._layerrope_depth = None if layerrope is None else int(layerrope_depth)
        if layerrope is not None and layerrope_paper is not None:
            loud(
                "loop block built with BOTH layerrope forms (v1 gains and the paper "
                "sites): exactly one may own the norm. Escape: pass one, or neither."
            )
        #: The paper form's four sites for this body (`layerrope.py`). Held with
        #: `object.__setattr__` so the shared module is NOT re-registered under
        #: every block (one set of `layerrope.sites.*` keys, as v1 does for its
        #: gains via `LayerRoPENorm`).
        object.__setattr__(self, "_paper", layerrope_paper)
        self._paper_block = int(paper_block)
        self._paper_position = int(paper_position)
        self._paper_index = cfg.layerrope_index
        if layerrope_paper is not None:
            #: The read gains ARE the norm's affine (`F.rms_norm(x, gain)`), so
            #: a separate gain parameter here would be a second, dead vector.
            self.norm = None
        elif layerrope is None:
            self.norm = nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
        else:
            self.norm = LayerRoPENorm(
                d, cfg.norm_eps, layerrope, depth=layerrope_depth, site=layerrope_site
            )
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
        # The EXACT LOCAL WINDOW (`model/fullattn.py`, `attn_window`): a third
        # ADDITIVE path beside KDA/CSA, off by default, so `attn_window=0` keeps
        # the parameter tree — and every pinned `state_dict` hash — exactly as
        # it was. It carries no state (a `W`-position band is `O(W)` of keys per
        # forward), so the state-carrying calls are refused at the seam below.
        if cfg.attn_window:
            assert cfg.attn_kind != "full", (
                f"attn_window={cfg.attn_window} with attn_kind='full': the exact arm "
                f"already reads every earlier position, so the band would be a strict "
                f"subset computed a second time. Escape: attn_kind='kda', or "
                f"attn_window=0."
            )
        self.window_attn = (
            FullAttention(
                cfg,
                device=device,
                window=cfg.attn_window,
                query_chunk=cfg.attn_window_query_chunk,
            )
            if cfg.attn_window
            else None
        )
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
        #: Gemma-2's final-logit cap (`config.ModelConfig.logit_softcap`):
        #: applied to EVERY per-iteration head output before its CE, so the
        #: deep-supervision terms and the final CE are the same bounded
        #: distribution. `0.0` skips the branch; the value is a plain float, so
        #: a CUDA capture re-reads it as a constant (the cap is not scheduled).
        self.logit_softcap = float(getattr(cfg, "logit_softcap", 0.0))
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
        #: The residual chain as fused Triton kernels (`model/fused_residual.py`):
        #: `H_pre`'s lane mix, the AttnRes write, `H_res`/`H_post` and the readout
        #: in two forward launches and two backward launches per iteration instead
        #: of ~780. OFF is the eager path below, byte for byte — the flag only
        #: chooses a route, it never changes a formula. `config.validate` refuses
        #: it without `use_mhc` (there would be no lanes to fuse); a host without
        #: CUDA/Triton is a COUNTED fallback (`_fused_ok`).
        self.fused_residual = bool(getattr(cfg, "fused_residual", False))
        #: The NORM CHAIN as one fused read kernel (`epilogue_read`): the lane
        #: norm, the eight hyper-network dots, `ctx`, and the body's
        #: `RMSNorm(ctx)·LayerRoPE-gain` in the same pass. Requires `use_mhc`
        #: (the read is over the lanes) and fp32 lanes; a CPU host is a COUNTED
        #: fallback (`fused_epilogue_fallback`). If `fused_residual` is ALSO on,
        #: the write half is the existing fused kernel, so one iteration is two
        #: launches across the body instead of the eager chain's ~dozens.
        self.fused_epilogue = bool(getattr(cfg, "fused_epilogue", False))
        #: The WRITE-ASSEMBLY half of `fused_kda_tail` (`model/fused_kda.py`):
        #: `y = attn·attn_write + branch·attn_write + ffn·mlp_write` in ONE
        #: launch over the three branch outputs instead of five eager passes.
        #: The KDA projection tail (`logsigmoid(g)`, `sigmoid(beta)`, the four
        #: casts) is the same flag's other kernel, in `kda._fla_path`. OFF is
        #: the eager lines, bit for bit; a CPU host or a non-fp32 branch is a
        #: COUNTED `fused_kda_tail_fallback`.
        self.fused_kda_tail = bool(getattr(cfg, "fused_kda_tail", False))
        self._fused_fell_back = False
        #: The DISTINCT bodies this driver cycles over (`DormouseModel` sets it
        #: to `(loop, *loop_extra)`); `None` means this block is the whole loop.
        #: Read by `_window`, so EVERY caller — one-shot, segmented, recurrent
        #: decode and prefill — resolves the same body per iteration.
        self._blocks = None
        #: Cached constant ZEROS, one per (device, dtype, shape). The absent-arm
        #: branches below used to allocate a fresh `[b*t, d]` zero EVERY
        #: iteration; at `b8·s512·d1536` that is 25 MB of memset per branch per
        #: iteration for a tensor that never changes. `requires_grad=False`, and
        #: nothing writes through it — it is added, not mutated. Keyed by device
        #: too, so a `.to(cuda)` cannot serve a stale CPU buffer.
        self._zero_cache: dict[tuple, torch.Tensor] = {}

    def _res(self, x: torch.Tensor) -> torch.Tensor:
        """Store a residual-stream tensor in `residual_dtype`; OFF is identity.

        The whole flag lives in this one cast: with `residual_dtype="fp32"` it
        returns `x` untouched (so `bf16=true` keeps storing its branch outputs
        in bf16 exactly as today), and with `"bf16"` it is the half-bytes
        write. No consumer is allowed to infer precision from this tensor — it
        widens at its own seam.
        """
        if self.residual_dtype is torch.float32:
            return x
        return x.to(self.residual_dtype)

    def _zero(
        self, like: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """A constant zero on `like`'s device, allocated once per (dtype, shape)."""
        dt = like.dtype if dtype is None else dtype
        key = (like.device, dt, shape)
        z = self._zero_cache.get(key)
        if z is None:
            z = torch.zeros(shape, dtype=dt, device=like.device)
            self._zero_cache[key] = z
        return z

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
        ce_fixed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Counters]:
        """`h0 [b,t,d]` → (`mean readout [b,t,d]`, SUM of per-iteration mean CEs, counters).

        The CE is a SUM over iterations, not a mean: the model divides by the
        number of terms once it has added the final readout's own CE (see
        `model.DormouseModel.forward` for why that term exists).

        `ce_fixed` swaps the per-iteration CE for `losses.fixed_cross_entropy`
        (arXiv:2610.31642 §2). `False` is the historical `log_softmax` line, bit
        for bit.

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
                h0, input_ids, targets, lm_head, counters, T, segment_len, ce_fixed
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
            ce_fixed=ce_fixed,
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

    def _fused_ok(self, xl: torch.Tensor, counters: Counters) -> str | None:
        """Which fused route runs on this tensor: `"epilogue"`, `"residual"`, or
        `None` (eager). A host check is COUNTED, a shape/dtype mismatch is LOUD.

        The distinction is the repo's one mark per degradation: "this host has
        no CUDA" is a legal place to run the model (the CPU suite is), so it must
        not stop the run — but a reader has to be able to tell the arm did not
        run, so it increments `fused_epilogue_fallback` / `fused_residual_fallback`
        **once** per window. A `[b,t,K,d]` the kernel cannot hold, or a non-fp32
        lane, is a config that lies about the arm, so it dies with the escape.
        """
        if self._fused_fell_back:
            return None
        if self.fused_epilogue and self.mhc is not None:
            if fused_residual.host_reason(xl.device.type) is not None:
                counters.count("fused_epilogue_fallback")
                self._fused_fell_back = True
                return None
            why = fused_residual.unsupported_reason(
                xl.dtype,
                self.mhc.n_streams,
                xl.shape[-1] * self.mhc.n_streams,
                xl.device.type,
                kernel="fused_epilogue",
            )
            if why is None:
                counters.count("fused_epilogue")
                return "epilogue"
            loud(why)
        if not self.fused_residual or self.mhc is None:
            return None
        if fused_residual.host_reason(xl.device.type) is not None:
            counters.count("fused_residual_fallback")
            self._fused_fell_back = True
            return None
        why = fused_residual.unsupported_reason(
            xl.dtype, self.mhc.n_streams, xl.shape[-1] * self.mhc.n_streams, xl.device.type
        )
        if why is None:
            counters.count("fused_residual")
            return "residual"
        loud(why)
        return None  # unreachable; `loud` raises

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
        out = self.moe(x.to(dt), counters=counters)
        # The write's storage dtype: fp32 today (the block's contract), or the
        # residual dtype under `residual_dtype="bf16"` (half the bytes, and the
        # MoE's own bank dtype is bf16 already, so the cast is a no-op).
        if self.residual_dtype is torch.bfloat16:
            return out.to(torch.bfloat16)
        return out.to(torch.float32)

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
        ce_fixed: bool = False,
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

        `iter_checkpoint` (off by default) wraps each iteration in
        `checkpoint(use_reentrant=False)`, so the deepest activation set kept
        for the backward is ONE iteration's rather than `T` of them. It is a
        memory decision only: the same ops run on the same inputs, and the
        recomputation is exact (the gate is a `1e-6` relative match on logits
        and gradients). What survives the forward is the state that crosses
        iterations — `h`, the mHC lanes, the AttnRes history and the carried
        attention state — which the body takes as explicit arguments for that
        reason. `rand_depth` and `loop_blocks` are resolved per iteration before
        the call, so the count is the ACTUAL depth, never a fixed `max_iter`.
        """
        if blocks is None:
            blocks = self._blocks if self._blocks is not None else (self,)
        bodies: tuple[LoopBlock, ...] = tuple(blocks)
        b, t, d = h0.shape
        #: The residual stream's storage dtype (`residual_dtype`). OFF returns
        #: `h0` itself, so the pre-flag forward is unchanged; ON casts once at
        #: the loop boundary and the readout accumulator below stays fp32.
        h = self._res(h0)
        h0_flat = h.reshape(b * t, d)
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
            h.unsqueeze(-2).expand(b, t, self.mhc.n_streams, d) if self.mhc is not None else None
        )
        #: `None` until the first pre-mix decides it (`_fused_ok`); stays
        #: `None` for a block with no lanes, where neither fused chain exists.
        fuse: str | None = None
        #: The READOUT accumulator is fp32 by contract (`precision_census.py`),
        #: even when the residual is stored bf16. Under fp32 this is exactly
        #: `zeros_like(h0)` (today's allocation).
        out_acc = (
            torch.zeros_like(h0)
            if self.residual_dtype is torch.float32
            else torch.zeros(h0.shape, dtype=torch.float32, device=h0.device)
        )
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
        sources: list[torch.Tensor] = [h] if self.use_attnres and not compose else []
        new_states: list = []
        if self._paper is not None:
            # The paper form's gain table is a function of the parameters ONLY
            # (input-independent), so ONE build per window serves every
            # iteration — the reference computes the whole `[L, d]` table per
            # forward, and `for_iteration` slices it (`layerrope.py`).
            self._paper.begin_forward()
        #: `slots_for(input_ids)` is a pure function of the window's ids, and
        #: the loop calls the memory once per ITERATION over those same ids.
        #: ONE key pass per body per window (was `max_iter` passes: 3 orders ×
        #: up to 3 lag folds each, every iteration). `None` bodies key normally.
        slots_by_engram: dict[int, torch.Tensor] = {}
        for i in stepaudit.iterations("loopblock.iterations", range(T), modules=bodies):
            counters.count("iter")
            ran = i + 1
            blk = bodies[i % len(bodies)]
            mem_slots = None
            if blk.engram is not None and input_ids is not None:
                mem_slots = slots_by_engram.get(id(blk.engram))
                if mem_slots is None:
                    mem_slots = blk.engram.slots_for(input_ids)
                    slots_by_engram[id(blk.engram)] = mem_slots
            # The state slot is read ONLY for a block that has an attention arm:
            # a block without one (`attn_kind` off, the recurrent test's fixture)
            # is legal with a shorter `states`, and reading it unconditionally is
            # the out-of-range the original guarded with the same `if`.
            attn_state = states[i] if (blk.attn is not None and states is not None) else None
            #: The TRAIN-half recompute arm. `iter_checkpoint` is the historical
            #: boolean (already in every snapshot); `activation_recompute` names
            #: the three arms the bench compares. A run that sets both to
            #: different modes is refused at the config seam.
            mode = self.recompute if self.recompute != "none" else (
                "full" if self.iter_checkpoint else "none"
            )
            if mode == "full" and torch.is_grad_enabled():

                def body(h_in, streams_in, sources_in, attn_in, fuse_in, blk_in, i_in, slots_in):
                    """The checkpointed unit: its counters are its own (see above)."""
                    cnt = Counters()
                    out = self._step(
                        h_in,
                        streams_in,
                        sources_in,
                        blk_in,
                        i_in,
                        h0_flat,
                        attn_in,
                        fuse_in,
                        cnt,
                        input_ids,
                        carry=carry,
                        recurrent=recurrent,
                        drop=drop,
                        compose=compose,
                        mem_slots=slots_in,
                    )
                    return (*out, cnt)

                #: `use_reentrant=False` + the ACTUAL body for THIS iteration:
                #: `rand_depth` varies `T`, `loop_blocks` varies the weights per
                #: index, and both are resolved before the call, so no fixed
                #: count is assumed. What survives the forward is this
                #: iteration's INPUT state (`h`, the mHC lanes, the AttnRes
                #: history, the carried attention state), not its body.
                h, streams, y, step_out, st, fuse, cnt = checkpoint(
                    body,
                    h,
                    streams,
                    tuple(sources),
                    attn_state,
                    fuse,
                    blk,
                    i,
                    mem_slots,
                    use_reentrant=False,
                    # The forward is deterministic: no dropout, no sampled
                    # noise, no RNG-consuming op. Replaying the RNG state on
                    # the recompute is pure overhead here, and `False` is the
                    # documented opt-out.
                    preserve_rng_state=False,
                )
                counters.merge(cnt)
                #: The number of iterations the backward will run AGAIN, one
                #: per window. Counted OUTSIDE the body so a recompute cannot
                #: count itself (checkpoint discards the recompute's `cnt`).
                counters.count("iter_checkpoint_saved")
                counters.count("recompute_full")
            else:
                h, streams, y, step_out, st, fuse = self._step(
                    h,
                    streams,
                    tuple(sources),
                    blk,
                    i,
                    h0_flat,
                    attn_state,
                    fuse,
                    counters,
                    input_ids,
                    carry=carry,
                    recurrent=recurrent,
                    drop=drop,
                    compose=compose,
                    mem_slots=mem_slots,
                    post_checkpoint=mode == "selective",
                )
            term = self._iter_ce(step_out, targets, lm_head, counters, loss_mask, ce_fixed)
            if self.use_attnres and not compose:
                sources.append(y)
            if blk.attn is not None:
                new_states.append(st)
            out_acc = out_acc + step_out
            if collect_hidden:
                hidden.append(step_out)
            if term is not None:
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

    def _step(
        self,
        h: torch.Tensor,
        streams: torch.Tensor | None,
        sources: tuple[torch.Tensor, ...],
        blk: LoopBlock,
        i: int,
        h0_flat: torch.Tensor,
        attn_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        fuse: str | None,
        counters: Counters,
        input_ids: torch.Tensor | None,
        *,
        carry: bool,
        recurrent: bool,
        drop: int,
        compose: bool,
        mem_slots: torch.Tensor | None = None,
        post_checkpoint: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, ...] | None,
        str | None,
    ]:
        """ONE iteration of the loop — the body `_window` runs `T` times.

        Split out so the iteration can be wrapped in
        `checkpoint(use_reentrant=False)` (`iter_checkpoint`): the checkpointed
        function must be a real callable over the iteration's INPUT state, and
        the four things that live across iterations (`h`, the mHC lanes, the
        AttnRes history, the carried attention state) are explicit arguments.
        Everything that is fixed for the window — `h0_flat`, the ids, the head,
        the drop, `carry`/`recurrent` — is closed over or passed as a keyword, so
        a recompute reads the same values. Returns `(h, streams, y, step_out,
        attn_state_out, fuse)`; the per-iteration CE is the caller's (`_iter_ce`,
        outside every checkpoint), and the accumulators and the counter merge
        stay with the caller exactly as the segmented path does.
        """
        b, t, d = h.shape
        mhc_raw = None
        st = None
        gains = None
        #: The body's read gain(s), resolved ONLY on the fused route so the
        #: eager path keeps its one `gain` call (and the arm's application
        #: counter) exactly as before.
        read_gain = None
        read_gain_mlp = None
        if blk._paper is not None:
            # PAPER FORM (`layerrope.py`): the two READ gains are the norm's
            # affine — `normalize(ctx) · attn_read` folded into one `F.rms_norm`
            # call, which with `read == ones` is the OFF path's bits. The depth
            # index is the ACTUAL iteration (v1's choice) or the body's network
            # POSITION, independent of the iteration — the paper's looped-model
            # reading.
            gains = blk._paper.for_iteration(
                blk._paper_block,
                i if blk._paper_index == "iteration" else blk._paper_position,
            )
            read_gain, read_gain_mlp = gains[0], gains[2]
        if streams is not None:
            # x_l = h_ctx: the iteration embedding joins the LANES, and
            # H_pre (applied by the caller, `mhc.py`) aggregates them into
            # the body's input — the crate's `h_ctx` (`mhc-2026-09-30.md`
            # §3.4), with Eq. 3's H_pre moved out of `forward`.
            streams = streams + self._res(self.iter_embed[i].view(1, 1, 1, d))
            if fuse is None:
                # `streams` is the post-add tensor the kernels see, so the
                # dtype/shape check runs here and not on `h0`.
                fuse = self._fused_ok(streams, counters)
            if fuse == "epilogue":
                if read_gain is None:
                    if blk._layerrope_depth is None:
                        read_gain = blk.norm.weight
                    else:
                        # The ACTUAL iteration number (`i`), never the run's
                        # compacted position: a shorter rand-depth run uses a
                        # PREFIX of the depth table and the coda/final depths
                        # stay anchored to `max_iter` (`layerrope.py`). ONE
                        # `gain` call also keeps the application counter at its
                        # eager value.
                        read_gain = blk._gains.gain(blk._layerrope_depth + i, blk.norm.site)
                ctx, mhc_raw, normed, normed_mlp = fused_residual.epilogue_read(
                    blk.mhc, streams, read_gain, read_gain_mlp, blk.eps, counters
                )
                normed_mlp = normed if normed_mlp is None else normed_mlp
            elif fuse == "residual":
                ctx, mhc_raw = fused_residual.pre(blk.mhc, streams, counters)
            else:
                # `pre_mix` widens the bf16 lanes internally (the lane norm is
                # a reduction); the body's input is stored back in the residual
                # dtype, which is a no-op under fp32.
                ctx = self._res(blk.mhc.pre_mix(streams))
            counters.count("mhc")
        else:
            ctx = self._res(h + self._res(self.iter_embed[i].view(1, 1, d)))
        # `gains` was resolved ONCE at the top of this iteration (it is the same
        # `for_iteration` call the fused route already paid for); resolving it
        # again here cost one extra call per iteration — measured 2026-11-04 as
        # a counter reading 24 where 12 is right.
        if fuse != "epilogue":
            # The fused read kernel already emitted `normed`/`normed_mlp` from
            # the same pass that mixed the lanes; computing them again here
            # would be the very double pass the fusion removes.
            # The RMSNorm mean is a reduction: it runs in fp32 even when the
            # body's input is stored bf16, and the result goes back to the
            # residual dtype (`_res` is the identity under fp32).
            norm_in = ctx if self.residual_dtype is torch.float32 else ctx.float()
            if gains is not None:
                normed = self._res(F.rms_norm(norm_in, (d,), gains[0], blk.eps))
                normed_mlp = self._res(F.rms_norm(norm_in, (d,), gains[2], blk.eps))
            elif blk._layerrope_depth is None:
                normed = self._res(blk.norm(norm_in))
                normed_mlp = normed
            else:
                # The ACTUAL iteration number (`i`), never the run's compacted
                # position: a shorter rand-depth run uses a PREFIX of the depth
                # table and the coda/final depths stay anchored to `max_iter`.
                normed = self._res(blk.norm(norm_in, depth=blk._layerrope_depth + i))
                normed_mlp = normed
        #: The controller reads `[ctx; h0]` as ONE GEMM. It is a fresh allocation
        #: per iteration and it is KEPT: splitting it into `ctx@W1ᵀ + h0@W2ᵀ` would
        #: change the accumulation order and therefore the bits. Escape: a model
        #: change with its own pin, not a step optimisation.
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
            prev = attn_state
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
            attn = self._zero(h, (b * t, d), self.residual_dtype)
        if blk.csa is not None:
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
        if blk.window_attn is not None:
            # The exact local band: an ADDITIVE path, the same
            # `w_attn`-scaled sum the MSA branch below joins. It has no
            # O(1) boundary state, so a state-carrying call would silently
            # read a NARROWER band than the one-shot forward (the bytes
            # before a segment cut / the earlier decode positions).
            assert not (carry or recurrent), (
                f"attn_window={self.window_attn.window} with a state-carrying "
                f"attention call (recurrent={recurrent}, carry={carry}): the "
                f"local band has no boundary state, so a segment cut or a "
                f"one-byte step would read fewer than the window's positions — "
                f"a model change wearing a memory flag. Escape: attn_window=0, "
                f"or segment_len=0 and the one-shot forward."
            )
            counters.count("attn_window_iter")
            wout = blk.window_attn(normed, counters=counters)
            attn = attn + wout.reshape(b * t, d) * w_attn
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
        #: One store for every attention arm's write: the branch is in the
        #: residual dtype from here on (`_res` is the identity under fp32).
        attn = self._res(attn)

        if blk.engram is not None and input_ids is not None:
            # The Engram read widens internally (fp32 tables + the order
            # reduction); its OUTPUT is residual storage, so it is stored at
            # the stream's dtype. `_res` is the identity under fp32.
            mem = self._res(
                blk._memory(normed, input_ids, drop, counters, slots=mem_slots).reshape(b * t, d)
            )
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
                .to(self.residual_dtype if self.residual_dtype is torch.bfloat16 else torch.float32)
                .reshape(b * t, d)
            )
            branch = self._res(memory_floor_mix(mem, dense, w_mem, self.lam_max))
        else:
            branch = self._zero(h, (b * t, d), self.residual_dtype)

        post = (
            h,
            streams,
            attn,
            branch,
            normed_mlp,
            w_ffn,
            gains,
            mhc_raw,
            ctx,
            fuse,
            tuple(sources),
            blk,
            i,
        )
        if post_checkpoint and torch.is_grad_enabled():
            # SELECTIVE (`activation_recompute="selective"`): the READ side —
            # norm, controller, attention and Engram — stays in the graph, so
            # the q/k/v/g projection outputs and the KDA/CSA boundary state are
            # KEPT. Only the write side — the MoE pass, the residual write and
            # `out_proj` — is recomputed. The checkpointed unit has its own
            # counters and gets no CE: the per-iteration head is cheap and runs
            # in the caller (`_iter_ce`), so the backward never repeats it.
            cnt = Counters()
            h, streams, y = checkpoint(
                self._step_post,
                *post,
                cnt,
                compose,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            counters.merge(cnt)
            counters.count("iter_checkpoint_saved")
            counters.count("recompute_selective")
        else:
            h, streams, y = self._step_post(*post, counters, compose)
        # `out_proj` runs OUTSIDE the selective checkpoint: its input and output
        # are two [b,t,d] residuals (the cheap end of the chain), so keeping
        # them costs less than recomputing a d×d matmul per iteration. `full`
        # still checkpoints it as part of the whole body.
        b, t, _ = h.shape
        step_out = self._res(blk.out_proj(h.reshape(b * t, d)).reshape(b, t, d))
        return h, streams, y, step_out, st, fuse

    def _step_post(
        self,
        h: torch.Tensor,
        streams: torch.Tensor | None,
        attn: torch.Tensor,
        branch: torch.Tensor,
        normed_mlp: torch.Tensor,
        w_ffn: torch.Tensor,
        gains: tuple | None,
        mhc_raw: torch.Tensor | None,
        ctx: torch.Tensor,
        fuse: bool | None,
        sources: tuple[torch.Tensor, ...],
        blk: LoopBlock,
        i: int,
        counters: Counters,
        compose: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """The WRITE half of one iteration — the unit `selective` recomputes.

        `ffn` (the MoE pass) and the residual write. `out_proj` runs in the
        caller, outside the checkpoint, because its two residual-sized tensors
        are the cheap end of the chain. Everything this half reads from the
        READ half is an explicit argument, so a recompute reads the same tensors
        the forward did. Returns `(h, streams, y)`; the attention state and
        `fuse` stay with the caller (the read side's outputs, never recomputed).
        """
        b, t, d = h.shape
        ffn = self._res(blk._moe_branch(normed_mlp, counters) * w_ffn)
        # PAPER FORM's WRITE gains (`layerrope.py`): each branch's OUTPUT is
        # scaled where it enters `y`, the residual UPDATE. `y` is exactly the
        # `F(H_pre·x_l)` the residual operator deposits (mHC `H_post`, AttnRes'
        # write, ReZero's `α`), and all three are LINEAR in `y`, so this is the
        # paper's `h + write·F(...)` for every residual arm without touching a
        # mixer's internals. `branch` (the Engram memory read) belongs to the
        # attention side: same read of the same normalised state, same residual
        # update (module docstring states the mapping).
        gain_attn = gains[1] if gains is not None else None
        gain_mlp = gains[3] if gains is not None else None
        fused_y = self.fused_kda_tail and fused_kda.combine_supported(attn, branch, ffn) and (
            gain_attn is None or fused_kda.combine_supported(gain_attn, gain_mlp)
        )
        if fused_y:
            y = fused_kda.combine(
                attn.reshape(b, t, d),
                branch.reshape(b, t, d),
                ffn.reshape(b, t, d),
                gain_attn,
                gain_mlp,
                counters,
            )
        else:
            if self.fused_kda_tail:
                counters.count("fused_kda_tail_fallback")
            if gains is not None:
                # The write gains are [d] vectors; cast them once so the three
                # products stay in the residual dtype (`_res` is identity under
                # fp32, where this is byte-for-byte today's expression).
                gain_a, gain_m = self._res(gains[1]), self._res(gains[3])
                attn = attn * gain_a
                branch = branch * gain_a
                ffn = ffn * gain_m
            y = self._res((attn + branch + ffn).reshape(b, t, d))

        if self.loom:
            # LOOM (2610.01153): per-loop router conditioning + variance
            # control in ONE place — the RMS-normalized write is the arm's
            # own signal (unit scale, so it cannot grow with depth) and the
            # per-slot gate decides how much of it this iteration writes.
            # `gate[i] == 0` adds an exact zero: the at-init forward IS the
            # `loom = false` forward, bit for bit.
            counters.count("loom")
            # The normalized write's norm is a reduction: fp32 under a bf16
            # residual (`_res` is the identity under fp32, today's bits).
            y32 = y if self.residual_dtype is torch.float32 else y.float()
            y = self._res(y32 + _rms_no_gain(y32, self.eps) * self.loom_gate[i].view(1, 1, 1))

        #: The WRITE half runs the fused kernel when the residual arm is on
        #: (`fuse == "residual"`) or when the epilogue arm is on AND the write
        #: half is enabled too (`fused_residual`). `epilogue_read`'s `raw` is
        #: the same 8-column hand-off, so the two halves compose.
        fused_write = fuse == "residual" or (fuse == "epilogue" and self.fused_residual)
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
            if fused_write:
                streams, h = fused_residual.write(
                    blk.mhc, self.attnres[i], streams, y, mhc_raw, True, self.eps, counters
                )
            else:
                updates = [self._zero(y, tuple(y.shape)), y]
                update = depth_attend(updates, self.attnres[i], self.eps, scale=float(len(updates)))
                streams = self._res(blk.mhc(streams, update))
                # 2409.19606 §2.1: our K full-width lanes re-sum to K times the
                # state, where the paper's K width-C lanes re-sum to C. The
                # mHC-ONLY arm keeps its historical unnormalised sum (pinned);
                # the composition applies the paper's 1/K. The readout is a
                # reduction over lanes: fp32, then stored in the residual dtype.
                h = self._res(streams.float().sum(dim=-2) / blk.mhc.n_streams)
        elif self.use_attnres:
            counters.count("attnres_write")
            h = self._res(depth_attend((*sources, y), self.attnres[i], self.eps))
        elif blk.mhc is not None:
            # Eq. 3's residual: H_res x_l + H_postᵀ F(H_pre x_l). The
            # readout is the SUM of the lanes (2409.19606 §2.1); the crate's
            # concat is its D/n-slicing layout, not this one (`mhc.py`).
            if fused_write:
                streams, h = fused_residual.write(
                    blk.mhc,
                    self.attnres[i] if self.attnres is not None else None,
                    streams,
                    y,
                    mhc_raw,
                    False,
                    self.eps,
                    counters,
                )
            else:
                streams = self._res(blk.mhc(streams, y))
                h = self._res(streams.float().sum(dim=-2))
        else:
            counters.count("rezero_write")
            h = self._res(ctx + y * blk.residual_scale)

        return h, streams, y

    def _iter_ce(
        self,
        step_out: torch.Tensor,
        targets: torch.Tensor | None,
        lm_head: nn.Module | None,
        counters: Counters,
        loss_mask: torch.Tensor | None,
        ce_fixed: bool,
    ) -> torch.Tensor | None:
        """The per-iteration deep-supervision CE — OUTSIDE every checkpoint.

        The head is cheap (byte vocab) and its output is `[t, vocab]`; keeping
        it in the forward graph costs little, while recomputing it in the
        backward is a full head matmul + softmax per iteration. Running it
        here, on the checkpoint's output `step_out`, makes that recompute
        impossible by construction.
        """
        if targets is None:
            return None
        assert lm_head is not None, "per-iteration CE needs the lm_head"
        b, t, d = step_out.shape
        logits = lm_head(step_out.reshape(b * t, d).to(torch.float32))
        if self.logit_softcap > 0.0:
            # The same Gemma-2 cap the final exit applies; the loop's
            # per-iteration deep-supervision CE must score the same
            # bounded distribution (`losses.logit_softcap`).
            logits = logit_softcap(logits, self.logit_softcap)
        if ce_fixed:
            ce = fixed_cross_entropy(
                logits, targets, reduction="none", counters=counters
            ).reshape(-1, 1)
        else:
            ce = -F.log_softmax(logits, dim=-1).gather(1, targets.reshape(-1, 1))
        # `doc_masking`: the mean over the SELECTED positions only (the
        # `None` default is the historical `.mean()`, bit for bit).
        return ce.mean() if loss_mask is None else masked_ce_mean(ce, loss_mask)

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
        self,
        normed: torch.Tensor,
        input_ids: torch.Tensor,
        drop: int,
        counters: Counters,
        *,
        slots: torch.Tensor | None = None,
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
            return self.engram(normed, input_ids, counters, slots=slots)  # type: ignore[misc]
        assert input_ids.shape[1] == normed.shape[1] + drop, "the window slice is misaligned"
        hidden = F.pad(normed, (0, 0, drop, 0))
        return self.engram(hidden, input_ids, counters, slots=slots)[:, drop:]  # type: ignore[misc]

    def _forward_segmented(
        self,
        h0: torch.Tensor,
        input_ids: torch.Tensor | None,
        targets: torch.Tensor | None,
        lm_head: nn.Module | None,
        counters: Counters,
        T: int,
        segment_len: int,
        ce_fixed: bool = False,
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
        for s in stepaudit.iterations("loopblock.segments", range(segments)):
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
                    ce_fixed=ce_fixed,
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
