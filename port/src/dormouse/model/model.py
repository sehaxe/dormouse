"""`DormouseModel`: embedding → prelude → loop block(s) → coda → RMSNorm → lm_head.

`crates/dormouse-core/src/model.rs`'s tree, at the port's day-one fidelity:
the residual stream and the logits are fp32 (the bf16 plan casts at the ARM
boundary: bf16 matmuls inside a projection, fp32 accumulation and fp32 state
between arms — burn's "mixed-dtype ops NaN on this stack" rule, kept as a rule
even where torch tolerates the mix).

The loss is the honest unweighted CE: `L_Rec`, the mean CE over the loop's
iterations through the same head (deep supervision, burn's shape), AND the CE
of the final `norm`+`lm_head` logits the forward returns — one unweighted mean
over `max_iter + 1` terms. See `DormouseModel.forward` for why the final term
is not optional. There is no KL term and no `p_dist`: they were deleted with
PonderNet (ADR-0013).

The port also carries one REGULARISER, SIGReg (`sigreg.py`, default weight 0.0):
`CE + sigreg_weight · sigreg_loss(z)` over the strided sample of the same
`norm(out)` state — teacher-free, predictor-free, mask-free, 0 parameters, no
second forward (`docs/research/2026-10-16-jepa-into-dormouse.md` §2/§3, the ONE
arm to run first).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .. import stepaudit
from ..config import ModelConfig, block_kinds, hybrid_ratio, validate_block_form
from ..instrument import Counters, loud
from ..losses import fixed_cross_entropy, logit_softcap
from .act_mem import ActivationOffload
from .attnres import FullDepthAttnRes
from .byteflow import (
    BfChannel,
    ByteFlowNet,
    bf_for_model,
    byteflow_config,
    compat_armed,
    count_channel,
)
from .convcodec import ConvCodec, codec_kind
from .dspark import DSparkHead, dspark_aux_loss
from .energy import EnergyHead, energy_contrastive_loss
from .halt import HaltHead, halt_loss
from .jepa import JepaHead
from .layerrope import LayerRoPEGains, LayerRoPENorm, LayerRoPEPaperGains
from .loopblock import LoopBlock, masked_ce_mean
from .mhc import MHCMix
from .mtp import MTPHead
from .prelude import UniqueLayer
from .qat_cache import QuantCache, attach_qat_caches, take_qat_refresh_counts
from .sigreg import sigreg_loss, stride_sample
from .transformer import TransformerBlock
from .tsct import make_projection, set_factor_qat_seed


def mhc_readout(streams: torch.Tensor, compose: bool) -> torch.Tensor:
    """The mHC lane readout — the loop form's two expressions, verbatim.

    `streams [b, t, K, d]` → `[b, t, d]`. The mHC-ONLY arm reads the
    UNNORMALISED sum (the loop's historical convention, pinned by
    `test_mhc.py`); the composition applies the paper's `1/K` (2409.19606
    §2.1). On `block_form="transformer"` the lane scale `c` is chosen per arm
    (`_forward_transformer`) so BOTH of these are the plain residual state.
    """
    h = streams.float().sum(dim=-2)
    return h / streams.shape[-2] if compose else h


class DormouseModel(nn.Module):
    def __init__(self, cfg: ModelConfig, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        # The runtime seam a hand-built/mutated config reaches; `__post_init__`
        # and `config.validate` own the same refusals (ADR-0019).
        validate_block_form(cfg)
        # Same two refusals `config.validate` owns (bad name; `conv` with
        # `use_byteflow=false`), held at the seam a hand-built model reaches.
        codec_kind(cfg)
        if cfg.layerrope:
            # The same two refusals `config.validate` owns, held at the runtime
            # seam because a hand-built model skips `validate` (ADR-0019).
            if cfg.use_byteflow:
                loud(
                    "layerrope=true with use_byteflow=true: both byteflow shapes "
                    "leave this model's byte-level loop (Option A builds no loop, "
                    "Option B runs it over K patch latents), so the LayerRoPE norm "
                    "tree would never be built. Escape: use_byteflow=false, or "
                    "layerrope=false."
                )
            if cfg.d_model % 2:
                loud(
                    f"layerrope=true with d_model={cfg.d_model}: the rotation acts on "
                    f"adjacent channel PAIRS, so the width must be even. Escape: an "
                    f"even d_model, or layerrope=false."
                )
            if cfg.layerrope_form not in ("v1", "paper"):
                loud(
                    f"layerrope_form={cfg.layerrope_form!r} is not a LayerRoPE form. "
                    f"Known: 'v1' and 'paper' (`model/layerrope.py`). Escape: name "
                    f"one of those, or layerrope=false."
                )
            if cfg.layerrope_form == "paper":
                if cfg.layerrope_index not in ("iteration", "position"):
                    loud(
                        f"layerrope_index={cfg.layerrope_index!r} is not a depth "
                        f"index. Known: 'iteration' and 'position'. Escape: name one "
                        f"of those, or layerrope_form='v1'."
                    )
                if cfg.layerrope_init not in ("paper", "identity"):
                    loud(
                        f"layerrope_init={cfg.layerrope_init!r} is not a LayerRoPE "
                        f"init. Known: 'paper' and 'identity'. Escape: name one of "
                        f"those, or layerrope_form='v1'."
                    )
                if cfg.layerrope_base <= 0.0:
                    loud(
                        f"layerrope_base={cfg.layerrope_base}: the RoPE spectrum is "
                        f"base^(-2j/d), so the base must be positive. Escape: base "
                        f"> 0 (100.0 is the paper's)."
                    )
        if cfg.loop_blocks > 1 and cfg.use_byteflow:
            # The config layer's refusal, held here too because a hand-built
            # model skips `validate` (the same rule the prelude seam follows).
            loud(
                f"loop_blocks={cfg.loop_blocks} with use_byteflow=true: Option A "
                f"returns before the loop is built (blocks 1+ would never run), "
                f"and Option B runs this loop over K patch latents — a different "
                f"sequence shape whose depth has not been measured with distinct "
                f"bodies. Escape: use_byteflow=false, or loop_blocks=1."
            )
        if cfg.use_byteflow and (cfg.prelude_layers or cfg.coda_layers):
            # The runtime seam's refusal (the same one `config.validate` holds):
            # Option A builds no loop and returns before the tree below, and
            # Option B runs the loop over K patch latents — the patcher changes
            # the sequence mid-model, so a byte-level prelude/coda would either
            # never run (A) or run on patch states (B). `ModelConfig` is not
            # validated on the hand-built path, so the check lives here too.
            loud(
                f"prelude_layers={cfg.prelude_layers}/coda_layers={cfg.coda_layers} "
                f"with use_byteflow=true: the byteflow shapes run no byte-level loop "
                f"to wrap — Option A is a standalone net with no loop at all, and "
                f"Option B runs this loop over K patch LATENTS, so the sequence "
                f"length changes mid-model and the unique layers would act on a "
                f"different sequence than the one they are specified on. Escape: "
                f"use_byteflow=false, or prelude_layers=0 and coda_layers=0."
            )
        # JEPA's two runtime refusals (the same shape every aux arm carries):
        # `validate` owns the CLI path, but a hand-built model skips it, and a
        # term with nothing to act on must not be silently dropped (ADR-0019).
        if cfg.jepa_weight > 0.0 and cfg.use_byteflow:
            loud(
                f"use_byteflow with jepa_weight={cfg.jepa_weight}: the byteflow "
                f"shapes run the loop over K patch latents (or not at all), so "
                f"there are no byte positions for the horizon `h[p+h]` to shift "
                f"(model/jepa.py). Escape: jepa_weight=0.0, or drop use_byteflow."
            )
        if cfg.jepa_weight > 0.0 and cfg.jepa_horizon < 1:
            loud(
                f"jepa_horizon={cfg.jepa_horizon} with jepa_weight={cfg.jepa_weight}: "
                f"the horizon is a positive count of positions to look ahead "
                f"(1 = the next position). Escape: jepa_horizon >= 1."
            )
        # The two SEAM-level regularisers this file owns: both are defined on
        # this model's `lm_head`, and both byteflow shapes leave that head
        # behind, so a hand-built config must hear the same refusal `validate`
        # gives a preset (ADR-0019: a term with nothing to act on).
        if cfg.tie_embeddings and cfg.use_byteflow:
            loud(
                "tie_embeddings=true with use_byteflow=true: both byteflow shapes "
                "leave this model's `lm_head` behind, so there is no head to tie. "
                "Escape: tie_embeddings=false, or use_byteflow=false."
            )
        if cfg.z_loss > 0.0 and cfg.use_byteflow:
            loud(
                f"z_loss={cfg.z_loss} with use_byteflow=true: the z-term is defined "
                f"on the `norm`+`lm_head` logits and both byteflow shapes return "
                f"`_byte_ce` instead. Escape: z_loss=0.0, or use_byteflow=false."
            )
        if cfg.logit_softcap > 0.0 and cfg.use_byteflow:
            loud(
                f"logit_softcap={cfg.logit_softcap} with use_byteflow=true: both "
                f"byteflow shapes return `_byte_ce` from their own patch-latent head "
                f"and never reach this model's `norm`+`lm_head`, so the cap would be "
                f"inert. Escape: logit_softcap=0.0, or use_byteflow=false."
            )
        if cfg.logit_softcap > 0.0 and cfg.mtp_weight > 0.0:
            loud(
                f"logit_softcap={cfg.logit_softcap} with mtp_weight={cfg.mtp_weight}: "
                f"MTP's CE calls the shared `lm_head` directly and would read uncapped "
                f"logits while the model's two CE sites read capped ones. Escape: "
                f"mtp_weight=0.0, or logit_softcap=0.0."
            )
        # `factor_qat_seed`'s ONE consumer (`tsct.set_factor_qat_seed`): the S3T
        # mask stream is a process-local generator, and a model build is its
        # boundary — the trainer builds one model per process, so the config
        # field reaches the arm here and nowhere else (a seed with no build
        # behind it would be the config-field-that-lies defect).
        set_factor_qat_seed(cfg.factor_qat_seed)
        #: The byteflow arm, under ONE flag with two shapes (`byteflow.py`'s
        #: Option A/B): `bf_net` is the standalone net (no loop arm is on, so
        #: the loop is not built at all — no dormant parameter tree), `bf` is
        #: the channel that runs THIS model's loop over K patch latents. Exactly
        #: one of them is ever non-None, and which one is COUNTED per forward.
        self.bf_net: ByteFlowNet | ConvCodec | None = None
        self.bf: BfChannel | None = None
        #: The MTP arm (`mtp.py`) and the scalar its own log line reads. The
        #: tensor is DETACHED by the forward and read only on the log cadence
        #: (`train/loop.py`'s `mtp_field`) — never per step (ADR-0018 rule 2).
        self.mtp: MTPHead | None = None
        self.last_mtp: torch.Tensor | None = None
        #: The MoD router's BCE aux value (`model/transformer.py`), DETACHED by
        #: the forward and read only on the log cadence (`train/loop.py`'s
        #: `mod_field`) — never per step (ADR-0018 rule 2). `None` when the arm
        #: is off or the forward ran no routed block.
        self.last_mod_aux: torch.Tensor | None = None
        #: The ENERGY head (`energy.py`, EBT arXiv:2507.02092): inference
        #: reranker and an OPTIONAL contrastive training term, built IFF
        #: `energy_weight > 0`. Declared before the byteflow early return so a
        #: channel model has the attribute (None) rather than no attribute.
        self.energy: EnergyHead | None = None
        #: JEPA (`jepa.py`): declared before the byteflow early return so a
        #: channel model has the attribute (None) rather than no attribute.
        self.jepa: JepaHead | None = None
        #: LayerRoPE (`layerrope.py`, arXiv:2610.09179): v1's ONE shared
        #: `gamma_shared` plus the per-depth `(amp, theta)` table, or — under
        #: `layerrope_form="paper"` — the four per-site gains of the block.
        #: Built only with the flag on; declared before the byteflow early
        #: return so a channel model has the attribute (None) rather than no
        #: attribute.
        self.layerrope: LayerRoPEGains | LayerRoPEPaperGains | None = None
        #: The v1 gains only (None under the paper form): the prelude/coda/final
        #: norms and the loop body take `layerrope=` from this, so the paper
        #: form cannot half-wire them.
        v1_gains: LayerRoPEGains | None = None
        paper_gains: LayerRoPEPaperGains | None = None
        #: Every QAT cache in the tree (`qat_cache.QuantCache`), collected after
        #: the build below. `forward` drains their hit counts into `qat_reuse`;
        #: the trainer invalidates them once per optimizer step.
        self._qat_caches: list[QuantCache] = []
        if cfg.use_byteflow and not compat_armed(cfg):
            if codec_kind(cfg) == "conv":
                bfc = byteflow_config(cfg)
                self.bf_net = ConvCodec(
                    bfc.d_global,
                    patch_bytes=cfg.byteflow_patch_bytes,
                    dtype=bfc.dtype,
                    k_tokens=bfc.k_tokens,
                    device=device,
                )
            else:
                self.bf_net = ByteFlowNet(byteflow_config(cfg), device=device)
            return
        d = cfg.d_model
        if cfg.block_form == "transformer" and cfg.byte_patch:
            # THE PATCH FORM: the same UNIQUE block stack as the control, run
            # over the conv codec's patch positions. The codec is BOTH ends
            # (bytes -> latents before, byte logits after), so `embed` and
            # `lm_head` are not built — they would be a parameter tree with no
            # reader and no gradient (the dead-capacity defect this port's
            # audits exist for). The final `norm` stays: the control applies
            # it before its head, and the channel applies it before `out_proj`.
            n_layers = cfg.n_layers or cfg.loop_blocks * cfg.max_iter
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlock(cfg, n_layers=n_layers, layer_idx=i, device=device)
                    for i in range(n_layers)
                ]
            )
            self.norm = nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
            # `byte_patch` is the dense-stack arm of `compat_armed`; this is
            # the only channel shape that keeps `transformer_blocks`.
            self.bf = bf_for_model(cfg, device=device)
            self._qat_caches = attach_qat_caches(self, dense=False)
            # The attributes `DormouseModel.forward` reads before it dispatches,
            # and the ones the trainer's eval/checkpoint readers expect.
            self._act_offload = None
            self.halt = None
            self.dspark = None
            self.last_halt = None
            self.last_breakdown = {}
            return
        self.embed = nn.Embedding(cfg.vocab, d, device=device)
        if cfg.block_stack != "uniform":
            # THE HYBRID STACK (`block_stack`): the ordered mix `[K,K,K,A]×`,
            # built from the SAME two block modules the pure forms use — a
            # `LoopBlock` from the KDA form and a `TransformerBlock` from the
            # control — so the A/B changes the block order, not the block.
            # Each KDA body runs ONCE (`LoopBlock.forward(depth=1)`), which is
            # why `max_iter=1` is a config refusal; the objective is the
            # transformer form's single final CE, so `block_stack="uniform"`
            # and every loop-only loss weight stay out of this path.
            ratio = hybrid_ratio(cfg)
            assert ratio is not None
            self.hybrid_blocks = nn.ModuleList(
                [
                    LoopBlock(cfg, device=device)
                    if kind == "kda"
                    else TransformerBlock(cfg, n_layers=cfg.n_layers, device=device)
                    for kind in block_kinds(cfg.n_layers, ratio)
                ]
            )
            self.norm = nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
            if cfg.tie_embeddings:
                self.lm_head = TiedHead(self.embed)
            else:
                head_rank = min(cfg.rank, d, cfg.vocab)
                self.lm_head = make_projection(d, cfg.vocab, cfg, rank=head_rank, device=device)
            # `dense_qat`/`factor_qat` are refused on this stack, so only a
            # `use_tsct` head (itself refused here) could own a cache.
            self._qat_caches = attach_qat_caches(self, dense=False)
            self.bf = None
            self._act_offload = None
            self.halt = None
            self.dspark = None
            self.last_halt = None
            self.last_breakdown = {}
            self.last_mod_aux = None
            return
        if cfg.block_form == "transformer":
            # THE VANILLA FORM (`block_form`, `model/transformer.py`): the same
            # embedding and the same final norm/head/CE, a stack of UNIQUE
            # pre-LN blocks instead of the loop. The early return skips every
            # loop-only module (KDA/CSA/mHC/Engram/controller/readout,
            # prelude/coda, LayerRoPE, halt/aux heads, host tables), so the tree
            # and the parameter count are the control's — not the loop's with a
            # dormant copy beside it.
            n_layers = cfg.n_layers or cfg.loop_blocks * cfg.max_iter
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlock(cfg, n_layers=n_layers, layer_idx=i, device=device)
                    for i in range(n_layers)
                ]
            )
            # Full depth AttnRes (`use_attnres`, `model/attnres.py`): one shared
            # module, one pseudo-query per layer + the per-distance log-prior.
            # Built only when the flag is on, so the off tree — parameter names,
            # count, RNG order — is the control's. `byte_patch` is refused
            # alongside the flag (`validate_block_form`): that channel runs the
            # blocks over patches and does not thread the history.
            if cfg.use_attnres:
                self.attnres_full = FullDepthAttnRes(
                    d, n_layers, eps=cfg.norm_eps, device=device
                )
            self.norm = nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
            if cfg.tie_embeddings:
                self.lm_head = TiedHead(self.embed)
            else:
                head_rank = min(cfg.rank, d, cfg.vocab)
                self.lm_head = make_projection(d, cfg.vocab, cfg, rank=head_rank, device=device)
            # `dense_qat` is refused on this form (`validate_block_form`), so the
            # only cache to collect is a `use_tsct` head's own.
            self._qat_caches = attach_qat_caches(self, dense=False)
            # mHC (`use_mhc`, `model/mhc.py`): the K-lane residual across the
            # LAYERS. ONE mixer for the stack — the lane state belongs to the
            # stack, not to a block, exactly as `LoopBlock._window` carries ONE
            # per body across its iterations. A mixer PER BLOCK would leave the
            # first block's `H_res` mixing the replicated lanes the stack starts
            # with (a row-stochastic mix of identical lanes is the identity), a
            # permanently dead parameter; one shared operator always sees lanes
            # a previous layer has already split. Built LAST so every base
            # draw (blocks, norm, head) is the OFF model's, RNG order included.
            if cfg.use_mhc:
                self.mhc_full = MHCMix(
                    d, cfg.mhc_streams, device=device, sinkhorn=cfg.mhc_sinkhorn
                )
                # Lane representation scale `c` (see `_forward_transformer`):
                # the composition's readout divides by K, so its lanes carry
                # the state (`c = 1`); the mHC-ONLY readout is the raw sum, so
                # its lanes carry `1/K` of it and the sum IS the state
                # (`c = 1/K`). Either way the value handed to `_readout` is the
                # plain residual state at step 0 — the identity the arm owes.
                # `H_post`'s K weights are rescaled to sum to `K·c` per token
                # (`MHCMix.forward(post_sum=...)`): a random `φ_post` would
                # otherwise split the replicated lanes on the first write and
                # move the read-out's increment off `y`. The crate's
                # `N(0, 0.01)` draws are kept. `b_pre` is the ONE init this arm
                # changes: `σ(10)` saturates the read gate (~1 per lane), which
                # makes the convex combination constant to ~1e-7 and leaves the
                # gate's gradient on σ's flat top; the uniform read `1/K` puts
                # it in the responsive region, so the arm can learn WHICH lane
                # to read (the loop's pinned `b_pre = 10` is untouched).
                self._mhc_lane_scale = 1.0 if cfg.use_attnres else 1.0 / cfg.mhc_streams
                with torch.no_grad():
                    self.mhc_full.b_pre.fill_(math.log(1.0 / (cfg.mhc_streams - 1.0)))
            # The attributes `DormouseModel.forward` reads before it dispatches,
            # and the ones the trainer's eval/checkpoint readers expect.
            self.bf = None
            self._act_offload = None
            self.halt = None
            self.dspark = None
            self.last_halt = None
            self.last_breakdown = {}
            return
        if cfg.layerrope:
            if cfg.layerrope_form == "paper":
                #: The paper form (`layerrope.py`): FOUR sites per loop body,
                #: `d + 4` parameters each. The table's depth axis is the
                #: network's: `P + max_iter + C` covers both index modes
                #: (iteration `i < max_iter`, position `P + b`).
                paper_gains = LayerRoPEPaperGains(
                    d,
                    max(cfg.prelude_layers + cfg.max_iter + cfg.coda_layers, 1),
                    n_blocks=cfg.loop_blocks,
                    base=cfg.layerrope_base,
                    beta_init=cfg.layerrope_beta_init,
                    identity=cfg.layerrope_init == "identity",
                    device=device,
                )
                self.layerrope = paper_gains
            else:
                #: ONE shared vector + one scalar pair per depth (module
                #: docstring): with the terminal final norm the table is
                #: `n_depths = P + max_iter + C + 1`; the per-site variant adds
                #: the site axis only, never a second `gamma_shared`.
                n_depths = cfg.prelude_layers + cfg.max_iter + cfg.coda_layers + 1
                n_sites = cfg.prelude_layers + cfg.loop_blocks + cfg.coda_layers + 1
                v1_gains = LayerRoPEGains(
                    d,
                    n_depths,
                    n_sites=n_sites,
                    per_site=cfg.layerrope_per_site,
                    amp_init=cfg.layerrope_amp_init,
                    theta_init=cfg.layerrope_theta_init,
                    learn_theta=cfg.layerrope_learn_theta,
                    device=device,
                )
                self.layerrope = v1_gains
        #: The UNIQUE layers outside the shared loop (`prelude.py`, arXiv
        #: 2610.23036): `N` run once before it, `M` once after it. Empty at the
        #: defaults (0/0), so no parameter, no RNG draw and no state dict key —
        #: the off path is the `c8a2ff9` tree bit for bit (`tests/test_prelude.py`).
        self.prelude = nn.ModuleList(
            [
                UniqueLayer(
                    cfg,
                    device=device,
                    layerrope=v1_gains,
                    layerrope_depth=j,
                    layerrope_site=j,
                )
                for j in range(cfg.prelude_layers)
            ]
        )
        self.loop = LoopBlock(
            cfg,
            device=device,
            layerrope=v1_gains,
            layerrope_depth=cfg.prelude_layers,
            layerrope_site=cfg.prelude_layers,
            layerrope_paper=paper_gains,
            paper_block=0,
            paper_position=cfg.prelude_layers,
        )
        #: The DISTINCT bodies inside the loop (`loop_blocks`, `config.py`):
        #: `self.loop` is body 0 and owns the per-iteration slots; each extra is
        #: a `body_only` LoopBlock (its own controller/KDA/Engram/MoE bank/
        #: readout/mHC mixer, no second `iter_embed`/`attnres`/`*_gate` set).
        #: At `loop_blocks=1` the ModuleList is EMPTY: no parameter, no RNG draw
        #: and no state-dict key, so the off path is main's tree bit for bit
        #: (`tests/test_loop_blocks.py` pins the sha256).
        self.loop_extra = nn.ModuleList(
            [
                LoopBlock(
                    cfg,
                    device=device,
                    body_only=True,
                    layerrope=v1_gains,
                    layerrope_depth=cfg.prelude_layers,
                    layerrope_site=cfg.prelude_layers + b,
                    layerrope_paper=paper_gains,
                    paper_block=b,
                    paper_position=cfg.prelude_layers + b,
                )
                for b in range(1, cfg.loop_blocks)
            ]
        )
        #: Set on the DRIVER, not passed by every caller: the one-shot forward,
        #: the segmented path, `recurrent_step` and decode's `_window` all
        #: resolve `loop._blocks`, so no path can silently run block 0 alone.
        self.loop._blocks = (self.loop, *self.loop_extra)
        self.coda = nn.ModuleList(
            [
                UniqueLayer(
                    cfg,
                    device=device,
                    layerrope=v1_gains,
                    layerrope_depth=cfg.prelude_layers + cfg.max_iter + j,
                    layerrope_site=cfg.prelude_layers + cfg.loop_blocks + j,
                )
                for j in range(cfg.coda_layers)
            ]
        )
        #: The final norm is the TERMINAL depth (`P + max_iter + C`): it is not
        #: a coda layer, and a shorter rand-depth run must not shift it onto the
        #: last iteration's depth (see `layerrope.py`).
        self.norm = (
            nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
            if v1_gains is None
            else LayerRoPENorm(
                d,
                cfg.norm_eps,
                v1_gains,
                depth=cfg.prelude_layers + cfg.max_iter + cfg.coda_layers,
                site=cfg.prelude_layers + cfg.loop_blocks + cfg.coda_layers,
            )
        )
        # burn clamps the HEAD's rank to min(rank, d_model, vocab) explicitly
        # ("a rank above either dimension is not a factorization"); the port
        # does the same, here, and nowhere else.
        if cfg.tie_embeddings:
            # `lm_head` IS `embed.weight`, transposed, with no parameter of its
            # own (the `d·vocab` the dense head used to cost). On a TSCT body
            # this REPLACES the low-rank head — a second difference the config
            # docstring names; on a dense body it is the exact head matrix and
            # the saving is exactly `d·vocab`.
            self.lm_head = TiedHead(self.embed)
        else:
            head_rank = min(cfg.rank, d, cfg.vocab)
            self.lm_head = make_projection(d, cfg.vocab, cfg, rank=head_rank, device=device)
        self.bf = bf_for_model(cfg, device=device)
        # The HALT HEAD (`halt.py`): built only when its weight is on, so
        # `halt_weight = 0` (every preset) adds no parameter and no compute.
        # `halt_mode` without a head is refused here too, because the model
        # constructor deliberately does not call `validate` (a hand-built model
        # must not run a mode it has no head for).
        self.halt: HaltHead | None = HaltHead(d, device=device) if cfg.halt_weight > 0.0 else None
        #: The detached halt term the log line reads (`train/loop.py:halt_field`).
        self.last_halt: torch.Tensor | None = None
        # The JEPA head (`jepa.py`): ONE mechanism — a causal latent predictor
        # and its own SIGReg anti-collapse term. Built only when the weight is
        # on, so `jepa_weight = 0` adds no parameter and no compute (the off
        # path is bit-identical, pinned in `tests/test_jepa.py`).
        self.jepa = (
            JepaHead(
                d,
                horizon=cfg.jepa_horizon,
                reg_weight=cfg.jepa_reg_weight,
                device=device,
            )
            if cfg.jepa_weight > 0.0
            else None
        )
        if cfg.halt_mode and self.halt is None:
            loud(
                f"halt_mode=true with halt_weight={cfg.halt_weight}: the "
                f"checkpoint has no halt head (the head is built IFF the weight "
                f"is on), so there is nothing to decide the stop. Escape: train "
                f"with halt_weight>0 and resume into halt_mode, or halt_mode=false."
            )
        if cfg.halt_mode and self.halt is not None and cfg.halt_max_iter > cfg.max_iter:
            loud(
                f"halt_max_iter={cfg.halt_max_iter} > max_iter={cfg.max_iter}: the "
                f"inference cap exceeds the depth the loop was built for, so the "
                f"halted forward would ask for an iteration with no slot. Escape: "
                f"max_iter >= {cfg.halt_max_iter} (the depth-8 arm), or "
                f"halt_max_iter <= {cfg.max_iter}."
            )
        # The DSpark draft head (2607.05147): built only when the weight is on
        # — at 0.0 the parameter set and every preset count match today's.
        self.dspark: DSparkHead | None = (
            DSparkHead(cfg, device=device) if cfg.dspark_weight > 0.0 else None
        )
        # The MTP modules (DeepSeek-V3 §3.2, `mtp.py`): built only when the
        # weight is on, so `mtp_k = mtp_weight = 0` (every preset) leaves the
        # tree — and the parameter count — exactly as it was. A weight with
        # `mtp_k == 0` is LOUD in `MTPHead` even without `config.validate`.
        if cfg.mtp_soft and cfg.mtp_weight <= 0.0:
            # The same refusal `validate` owns, held at the runtime seam a
            # hand-built ModelConfig reaches without passing through it.
            loud(
                f"mtp_soft=true with mtp_weight={cfg.mtp_weight}: soft-MTP routes "
                f"the NTP gradient into an MTP hidden, and the MTP head is built "
                f"IFF the weight is on (mtp.py). Escape: mtp_weight > 0 with "
                f"mtp_k >= 1, or mtp_soft=false."
            )
        if cfg.mtp_soft and not cfg.mtp_head_norm:
            # The seam is what isolates each head's hidden; without it the
            # identity is last-head-only (see `mtp.py`). This is the runtime
            # seam for a config mutated after construction.
            loud(
                "mtp_soft=true with mtp_head_norm=false: the KL recovery identity "
                "needs the reference's per-head final RMSNorm (each head reads "
                "its OWN hidden). Without it the later heads' coupling flows back "
                "through the sequential projection. Escape: mtp_head_norm=true, "
                "or mtp_soft=false."
            )
        self.mtp = (
            MTPHead(
                cfg.d_model,
                cfg.mtp_k,
                eps=cfg.norm_eps,
                head_norm=cfg.mtp_head_norm,
                device=device,
            )
            if (cfg.mtp_weight > 0.0)
            else None
        )
        # The ENERGY head (EBT arXiv:2507.02092, `energy.py`): built IFF the
        # weight is on, so `energy_weight = 0` adds no parameter and no compute.
        # The constructor deliberately does not call `validate`, so the two
        # refusals the config owns are held here too (the same rule the halt
        # head follows).
        if cfg.energy_weight > 0.0:
            if cfg.use_byteflow:
                loud(
                    f"energy_weight={cfg.energy_weight} with use_byteflow=true: the "
                    f"channel runs the loop over K patch latents, so there is no "
                    f"per-byte state for the contrastive head to score. Escape: "
                    f"energy_weight=0.0, or use_byteflow=false."
                )
            if cfg.dspark_weight > 0.0:
                loud(
                    f"energy_weight={cfg.energy_weight} with dspark_weight="
                    f"{cfg.dspark_weight}: both aux objectives read the same "
                    f"`norm(out)` state, and their interaction is not measured. "
                    f"Escape: one weight per run."
                )
            self.energy = EnergyHead(d, cfg.energy_rank, device=device)
        # The QAT cost fix's wiring seam: modules whose switch is on built their
        # own `QuantCache`; every plain `nn.Linear` gets one here when
        # `dense_qat` is on (`qat_linear` reads it). A model built with both
        # switches off collects nothing — no cache, every number on record.
        self._qat_caches = attach_qat_caches(self, dense=cfg.dense_qat != "none")
        #: The host-offload mechanism, installed by the TRAINER
        #: (`configure_actmem`): a model built and forwarded without it keeps
        #: `None`, and `forward` takes no branch and installs no hooks.
        self._act_offload: ActivationOffload | None = None

    def set_iter_checkpoint(self, on: bool) -> int:
        """Enable per-iteration checkpointing on every loop block; return the count.

        The flag is a TRAIN-half field (`TrainConfig.iter_checkpoint`) because
        it changes what the backward recomputes, not what the model computes.
        One call at the top of a run is enough: nothing reads it per step on the
        host. A model with no loop block (`use_byteflow` Option A builds none)
        would run the flag inert, so it refuses with the escape.
        """
        if on and self.cfg.block_stack != "uniform":
            loud(
                f"iter_checkpoint=true with block_stack={self.cfg.block_stack!r}: "
                f"the checkpoint wrapper is applied to each loop ITERATION, and the "
                f"hybrid stack's attention blocks have no loop iteration to wrap — "
                f"the arm would cover only the KDA half while the log reported it "
                f"on. Escape: iter_checkpoint=false."
            )
        n = 0
        for module in self.modules():
            if isinstance(module, LoopBlock):
                module.iter_checkpoint = bool(on)
                n += 1
        if on and n == 0:
            loud(
                "iter_checkpoint=true but this model builds no loop block "
                "(use_byteflow Option A returns before the loop), so the flag would "
                "recompute nothing while the log line reported it on. Escape: "
                "use_byteflow=false, or iter_checkpoint=false."
            )
        return n

    def set_recompute(self, mode: str) -> int:
        """Enable `activation_recompute` on every loop block; return the count.

        `none` is the historical graph; `full` is `iter_checkpoint`'s whole-
        iteration checkpoint; `selective` keeps the read half (q/k/v/g outputs
        and the KDA/CSA boundary state) and recomputes only the write half.
        A model with no loop block refuses the two recompute modes instead of
        running them inert.
        """
        from .recompute import check_mode

        check_mode(mode)
        if mode != "none" and self.cfg.block_stack != "uniform":
            loud(
                f"activation_recompute={mode!r} with block_stack="
                f"{self.cfg.block_stack!r}: the recompute arms are wired on the KDA "
                f"loop block only, so the attention half would keep its full save "
                f"set while the log reported the arm on. Escape: "
                f"activation_recompute='none'."
            )
        n = 0
        for module in self.modules():
            if isinstance(module, LoopBlock):
                module.recompute = mode
                n += 1
        if mode != "none" and n == 0:
            loud(
                f"activation_recompute={mode!r} but this model builds no loop block "
                f"(use_byteflow Option A returns before the loop), so the arm would "
                f"recompute nothing while the log line reported it on. Escape: "
                f"use_byteflow=false, or activation_recompute='none'."
            )
        return n

    def configure_actmem(
        self,
        *,
        iter_checkpoint: bool,
        activation_offload: float,
        activation_recompute: str = "none",
    ) -> None:
        """Install the TrainConfig activation-memory arms in one call.

        Called once by the trainer. `activation_offload=0.0` leaves the hook pair
        uninstalled (`self._act_offload is None`), which is the bit-identical
        default; a positive budget refuses the byteflow shapes because their
        chunked head's save set is not measured (see `model/act_mem.py`).

        `activation_recompute` is the named arm. The historical
        `iter_checkpoint=True` means `full` when the enum is `none`; naming two
        different modes at once is refused by `config.validate_train` (the same
        rule reaches a hand-built call here).
        """
        if iter_checkpoint and activation_recompute not in ("none", "full"):
            loud(
                f"iter_checkpoint=true with activation_recompute="
                f"{activation_recompute!r}: two names for one recompute arm, and they "
                f"do not name the same one. Escape: iter_checkpoint=false and the "
                f"enum, or activation_recompute='full'."
            )
        mode = activation_recompute
        if mode == "none" and iter_checkpoint:
            mode = "full"
        self.set_recompute(mode)
        if activation_offload > 0.0 and self.cfg.block_stack != "uniform":
            loud(
                f"activation_offload={activation_offload} with block_stack="
                f"{self.cfg.block_stack!r}: the saved-tensor hook pair is priced on "
                f"the loop block's save set, and the hybrid stack's attention half "
                f"saves a different set — the budget would park a graph nobody "
                f"measured. Escape: activation_offload=0.0."
            )
        if activation_offload == 0.0:
            self._act_offload = None
            return
        if not self.cfg.bf16:
            loud(
                f"activation_offload={activation_offload} with bf16=false: the arm "
                f"parks fp32 activations, i.e. twice the pinned bytes and twice the "
                f"PCIe traffic of the measured bf16 arm, and the budget's arithmetic "
                f"would describe a different tensor set. Escape: bf16=true, or "
                f"activation_offload=0.0."
            )
        mech = ActivationOffload(activation_offload)
        if self.bf is not None or self.bf_net is not None:
            loud(
                f"activation_offload={activation_offload} with use_byteflow=true: "
                f"both byteflow shapes build a chunked head over patch latents "
                f"(`byteflow.encode_chunks`/`decode_chunks`) and re-read activations "
                f"across chunk boundaries, so a global saved-tensor hook would park a "
                f"graph whose save set nobody has measured — and the chunked head's "
                f"own re-forward already conflicts with a second hook pair. Escape: "
                f"use_byteflow=false, or activation_offload=0.0."
            )
        mech.bind(self)
        self._act_offload = mech

    def set_factor_qat_alpha(self, alpha: float) -> None:
        """Set the annealing alpha on EVERY QAT consumer (one call per step).

        The consumers are the `TSCTLinear`s (loop block + head), the
        `DualLatentGLU`s (routed + shared experts) and `SparseMoE._banks`'
        cached stacks — every module carrying `factor_qat_alpha` — PLUS every
        dense path module carrying `dense_qat_alpha` (`LoopBlock`, `SparseMoE`,
        `KDAAttention`, `Engram`, the MSA pair, `DSparkHead`, `DenseLinear`):
        ONE alpha drives both paths, so a run cannot anneal them on two
        schedules. The bank cache keys on `factor_qat_alpha` (see `_banks`), so
        a changed alpha rebuilds the quantized stacks on the next forward; a
        constant alpha (the default) rebuilds nothing extra.
        """
        assert 0.0 <= alpha <= 1.0, f"anneal alpha {alpha} outside 0..=1"
        for m in self.modules():
            if hasattr(m, "factor_qat_alpha"):
                m.factor_qat_alpha = alpha
            if hasattr(m, "dense_qat_alpha"):
                m.dense_qat_alpha = alpha

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        counters: Counters | None = None,
        depth: int | None = None,
        pure: bool = False,
        breakdown: bool = False,
        segment_len: int = 0,
        loss_mask: torch.Tensor | None = None,
        ce_fixed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Counters]:
        """`input_ids [b,t]` (+ `targets [b,t]`) → (logits `[b*t,vocab]`, loss, counters).

        `depth` is the ADAPTIVE DEPTH arm's per-step loop depth (`None` =
        `max_iter`, ADR-0013's fixed depth). It reaches the loop and the loss
        divides by the depth ACTUALLY run, so steps of different `T` are
        scale-comparable.

        `segment_len` is the SEGMENTED TRAINING path (`loopblock`, off at 0):
        the trainer passes `train.segment_len`, which cuts the window into
        segments and carries the KDA state across them so a window longer than
        one forward fits in VRAM can be trained. With `use_csa` the CSA ring is
        carried too, including the partial chunk a cut splits (it is completed
        by the next segment from its carried `sum`/`count`; nothing is dropped,
        nothing is padded), so the pair is legal. It is a memory decision, not a
        model change (the logits agree with the one-shot forward to fp32
        round-off), and the returned logits keep the `[b*t, vocab]` shape, so
        the trainer's loss code is untouched. The EVAL path deliberately does
        not pass it: an eval saves no activations (`no_grad`), and leaving it
        one-shot keeps the held-out number comparable with the record.

        `loss_mask [b,t]` is the `doc_masking` arm's position weight (1/0):
        the per-iteration CEs and the final CE are means over the SELECTED
        positions only, so padding and cross-document targets contribute
        neither a loss nor a gradient. `None` (the default, every number on
        record) is the unchanged `.mean()` path, bit for bit.

        `ce_fixed` routes the per-iteration CE and the final CE through
        `losses.fixed_cross_entropy` (arXiv:2610.31642 §2: the target logit's
        gradient does not collapse past the dtype's precision cliff, and the
        gradient sums to zero). `False` (the default, every number on record)
        is the historical `log_softmax` path, bit for bit — the flag is the
        TRAIN half's (`TrainConfig.ce_fixed`), which is why it is threaded here
        rather than stored on `ModelConfig`: a checkpoint's snapshot pairs a
        model with the objective it was trained on.
        """
        counters = Counters() if counters is None else counters
        assert input_ids.dtype in (torch.long, torch.int64), "input_ids must be integer ids"
        assert input_ids.ndim == 2, f"input_ids must be [b, t], got {tuple(input_ids.shape)}"
        if loss_mask is not None:
            assert loss_mask.shape == input_ids.shape, (
                f"loss_mask {tuple(loss_mask.shape)} != input_ids {tuple(input_ids.shape)}"
            )
            if segment_len:
                loud(
                    f"loss_mask with segment_len={segment_len}: the segmented forward "
                    f"carries state across its cuts, so a document boundary inside the "
                    f"window is not a segment boundary. Escape: segment_len=0."
                )
            if self.bf_net is not None or self.bf is not None:
                loud(
                    "loss_mask with use_byteflow: the byteflow shapes score patch "
                    "latents, not byte positions, so a [b,t] byte mask addresses a "
                    "different sequence. Escape: use_byteflow=false, or loss_mask=None."
                )
        if self.cfg.dense_qat != "none":
            # The dense path ran (the per-site proof is the coverage gate,
            # tests/test_dense_qat.py: every nn.Linear/DenseLinear weight in the
            # forward must be seen by `dense_qat_project`). One per forward, on
            # the existing counter line — a reader of the log sees the arm.
            counters.count("dense_qat_arms")
        if self.bf_net is not None:
            self._refuse_segmented(segment_len)
            logits, loss = self._forward_net(input_ids, targets, counters, ce_fixed)
        elif self.bf is not None:
            self._refuse_segmented(segment_len)
            logits, loss = self._forward_channel(
                input_ids, targets, counters, depth, ce_fixed, breakdown
            )
        else:
            # The host-offload arm is a TRAINING decision: `no_grad` saves
            # nothing, so an eval/decode call never installs the hooks even with
            # the flag on. `_act_offload is None` is the off path's only branch.
            off = self._act_offload
            if off is not None and torch.is_grad_enabled():
                with off.region(counters):
                    logits, loss = self._forward_byte_lm(
                        input_ids,
                        targets,
                        counters,
                        depth,
                        pure,
                        breakdown,
                        segment_len,
                        loss_mask,
                        ce_fixed,
                    )
            else:
                logits, loss = self._forward_byte_lm(
                    input_ids,
                    targets,
                    counters,
                    depth,
                    pure,
                    breakdown,
                    segment_len,
                    loss_mask,
                    ce_fixed,
                )
        if self._qat_caches:
            # Cache hits in THIS forward, on the existing counter line: the
            # proof the once-per-step path ran (a cache that silently missed on
            # every call would leave `qat_reuse=0` while still training).
            reuse = sum(c.take_hits() for c in self._qat_caches)
            if reuse:
                counters.count("qat_reuse", reuse)
            # The refresh LAUNCHES of this forward: `calls` is how many quantizer
            # chains ran, `tensors` how many weight tensors they covered. One
            # stacked bank call is one call and E tensors, so a regression to
            # per-expert shows up as calls == tensors instead of calls ≪ tensors.
            calls, tensors = take_qat_refresh_counts()
            if calls:
                counters.count("qat_refresh_calls", calls)
                counters.count("qat_refresh_tensors", tensors)
        if self.layerrope is not None:
            # ONE device scalar per forward, drained with the window's other
            # counters at the log cadence (`Counters.count_device`, no host
            # read): the proof the per-depth gain ran, and how many sites it
            # covered. Eval/decode calls drain nothing on their own.
            counters.count_device("layerrope_forward", self.layerrope.take_applications())
        counters.count("forward")
        return logits, loss, counters

    @staticmethod
    def _refuse_segmented(segment_len: int) -> None:
        """`segment_len` on a path that has no byte window to cut — LOUD.

        The byteflow shapes run the loop over K patch latents (Option B) or not
        at all (Option A), so a segment_len would either be silently inert or
        cut a sequence that is not the byte stream the carry was designed for.
        `validate` refuses the pair for a resolved run; this is the same
        refusal at the runtime seam, where a hand-built model arrives.
        """
        if segment_len:
            loud(
                f"segment_len={segment_len} with use_byteflow: the byteflow shapes run "
                f"the loop over K patch latents, not over the byte window, so there is "
                f"no byte sequence to cut into segments. Escape: use_byteflow=false, or "
                f"segment_len=0."
            )

    def _forward_net(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        counters: Counters,
        ce_fixed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Option A: the byteflow net alone — its byte logits ARE the model's."""
        counters.count("bf_net")
        assert self.bf_net is not None
        b, t = input_ids.shape
        flat = self.bf_net(input_ids).reshape(b * t, self.cfg.vocab)
        return flat, _byte_ce(flat, targets, fixed=ce_fixed, counters=counters)

    def _forward_channel(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        counters: Counters,
        depth: int | None = None,
        ce_fixed: bool = False,
        breakdown: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Option B: front stage → this model's loop OR dense stack → back stage.

        The objective is the byte CE from the DECODER, over all T positions —
        not the loop's in-loop `L_Rec`. Two reasons, both load-bearing: the
        decoder is the model's only exit, so a loss taken at the K patch starts
        would leave `decode_chunks`' blocks and the byte head with no gradient;
        and the held-out eval scores `[b,t,256]` logits against all T bytes, so
        anything else would be a train CE that is not the quantity on the eval
        line. `sel` is a discrete argtopk product: the SELECTION has no
        gradient, the gathered rows and everything after `z` train normally.

        TWO STACKS, ONE CODE. With `byte_patch=false` (the default and every
        number on record) the shared `self.loop` runs over K latents. With
        `byte_patch=true` the SAME channel runs `self.transformer_blocks`
        over EVERY patch of this window. Exact shapes for that form, with
        `s = byteflow_patch_bytes` and `P = t // s` (`t = 1024, s = 8` →
        `P = 128`):

        * `input_ids [b, t]` — bytes;
        * `encode_chunks(input_ids, P)` → `h [b, t, width]` (the per-byte
          stream the decoder's skip reads; `width` is the codec's own),
          `z [b, P, d_global]`, `sel [b, P]` (every patch start);
        * `in_proj: d_global → d_model` → `x [b, P, d_model]`;
        * `n_layers` unique blocks: `[b, P, d_model] → [b, P, d_model]`,
          causal over patch positions (`TransformerBlock`'s SDPA);
        * final RMSNorm: `[b, P, d_model] → [b, P, d_model]` (the control's
          own terminal norm);
        * `out_proj: d_model → d_global` → `[b, P, d_global]`;
        * `decode_chunks` → byte logits `[b, t, 256]`, fp32;
        * `[b·t, 256]` + targets → `_byte_ce`, the mean over BYTES — so BPB is
          the same quantity every other arm reports, not a per-patch mean.
        """
        assert self.bf is not None
        # The aux heads are refused with `use_byteflow` by `validate`, and the
        # model constructors deliberately do NOT call `validate` — so the same
        # refusal is held here, at the runtime seam (ADR-0019's "a term with
        # nothing to act on").
        assert (
            self.cfg.dspark_weight == 0.0
            and self.cfg.sigreg_weight == 0.0
            and self.cfg.mtp_weight == 0.0
            and self.cfg.energy_weight == 0.0
        ), (
            "the byteflow channel carries no aux term: validate refuses the aux weights "
            "(dspark/sigreg/mtp/energy) with use_byteflow - set them to 0"
        )
        count_channel(counters)
        assert self.cfg.halt_weight == 0.0, (
            "the byteflow channel carries no halt head: its loop runs over K patch "
            "latents with no per-byte readout, and the halt term needs the byte "
            "loop's per-iteration CE - set halt_weight to 0"
        )
        bf = self.bf
        # `byte_patch` runs EVERY patch of THIS window (K = P): with K < P the
        # codec's global top-k could let a future byte change the SELECTION and
        # move a past logit, so the strict prefix-causality gate only holds at
        # K = P. The byte-level channel keeps its configured K.
        k = input_ids.shape[1] // self.cfg.byteflow_patch_bytes if self.cfg.byte_patch else bf.k
        h, z, sel = bf.net.encode_chunks(input_ids, k)
        x = bf.in_proj(z)  # [b, k, d_model]
        assert x.shape[1] == k, (x.shape, k)
        if self.cfg.byte_patch:
            # The dense stack, unchanged, over PATCH positions. No byte ids
            # (the blocks read none) and no loop: this is `block_form`'s own
            # stack with the codec's sequence around it.
            assert self.cfg.block_form == "transformer"
            for block in stepaudit.iterations(
                "forward.transformer_blocks", self.transformer_blocks
            ):
                x = block(x, counters)
                counters.count("transformer_block")
            # The dense control's final RMSNorm, before the codec's `out_proj`
            # (its `lm_head` analogue). Without it the patch stream would enter
            # the projection at the raw residual scale.
            out_acc = self.norm(x)
        else:
            # The loop, unchanged, on K patch latents. No byte ids: the memory
            # arm is refused with use_byteflow (validate), so nothing here
            # needs them.
            out_acc, _, _ = self.loop(x, None, None, None, counters=counters, depth=depth)
        logits = bf.net.decode_chunks(bf.out_proj(out_acc), h, sel)  # [b, t, vocab]
        flat = logits.reshape(-1, self.cfg.vocab)
        loss = _byte_ce(flat, targets, fixed=ce_fixed, counters=counters)
        # The SAME breakdown contract `_readout` keeps: the eval loop reads
        # `last_breakdown["fin"]` and this path never wrote it, so every
        # byte_patch eval died with KeyError: 'fin' (found by E1, 2026-11-08 —
        # the arm had never been evaluated end to end).
        if breakdown and loss is not None and targets is not None:
            self.last_breakdown = {"fin": float(loss.detach()), "per_iter": []}
        return flat, loss

    def _forward_byte_lm(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        counters: Counters,
        depth: int | None = None,
        pure: bool = False,
        breakdown: bool = False,
        segment_len: int = 0,
        loss_mask: torch.Tensor | None = None,
        ce_fixed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The plain byte-level stack: embedding → prelude → loop → coda → RMSNorm → lm_head."""
        if self.cfg.block_stack != "uniform":
            # The ordered KDA/attention stack (`block_stack`). Its own tree, so
            # it returns before the loop's prelude/coda/MoE/aux seam.
            return self._forward_hybrid(
                input_ids,
                targets,
                counters,
                loss_mask,
                ce_fixed,
                breakdown,
                segment_len,
                depth,
            )
        if self.cfg.block_form == "transformer":
            # The dispatch `block_form` exists for: the SAME embedding, final norm,
            # head and CE, a different block stack (never a second model class).
            return self._forward_transformer(
                input_ids, targets, counters, loss_mask, ce_fixed, breakdown
            )
        h0 = self.embed(input_ids)
        # The prelude: unique layers, once, before the shared loop. The loop's
        # contract is `[b, t, d_model]` in the embedding's dtype — asserted at
        # the boundary, not assumed (a prelude that changed the width or the
        # residual dtype would otherwise fail deep inside the loop's first
        # projection, far from the flag).
        for layer in self.prelude:
            h0 = layer(h0)
            counters.count("prelude")
        b_in, t_in = input_ids.shape
        assert h0.shape == (b_in, t_in, self.cfg.d_model), (
            f"the loop's input contract is [b, t, d_model] = "
            f"{(b_in, t_in, self.cfg.d_model)}, got {tuple(h0.shape)}"
        )
        assert h0.dtype == self.embed.weight.dtype, (
            f"the loop's input dtype must be the residual dtype "
            f"{self.embed.weight.dtype}, got {h0.dtype}"
        )
        # The HALT HEAD is an INFERENCE mechanism (`halt.py`): a training step
        # runs the scheduled depth, because the head's BCE needs every iteration
        # that step ran. At inference (eval/decode, `model.eval()`) it caps the
        # loop at `halt_max_iter` and stops at the first iteration whose stop
        # probability exceeds 0.5. The decision costs ONE host read per forward
        # — never in a training step (ADR-0018 rule 2); the adaptivity research
        # doc prices a per-iteration read at <=0.1% of an iteration, and making
        # the skip REAL (the iterations below are never launched) is what buys
        # the compute back.
        halt = (
            self.halt
            if (self.halt is not None and self.cfg.halt_mode and not self.training)
            else None
        )
        if halt is not None and depth is not None:
            # An EXPLICIT depth is a fixed-depth call (the eval curve passes one
            # per depth): the halt policy would make `depths=1..8` mean "the
            # policy capped at d", a different quantity wearing the same label.
            # The policy runs where the depth is the model's own choice.
            halt = None
        if halt is not None:
            depth = self.cfg.halt_max_iter
        if self.halt is not None and segment_len:
            loud(
                f"segment_len={segment_len} with halt_weight={self.cfg.halt_weight}: "
                f"the segmented forward runs the loop per SEGMENT, so the halt "
                f"head's states would be concatenations of segment windows and "
                f"its target a per-segment mean — two unvalidated axes in one run. "
                f"Escape: segment_len=0, or halt_weight=0."
            )
        if self.jepa is not None and segment_len:
            loud(
                f"segment_len={segment_len} with jepa_weight={self.cfg.jepa_weight}: "
                f"JEPA predicts `h[p+h]` from `h[p]`, and the segmented forward "
                f"cuts the window, so the horizon would cross a segment boundary "
                f"— a state from a different window is a different objective. "
                f"Escape: segment_len=0, or jepa_weight=0.0."
            )
        # The routing controller's PER-FORWARD state: the `aux` arm accumulates
        # one term per routed forward (the loop's T iterations) and the loss
        # below reads the sum. Cleared here, before the loop, so a recompute's
        # copy (backward) can never leak into the next step.
        self.loop.moe.begin_step()
        out, rec_sum, counters = self.loop(
            h0,
            input_ids,
            targets,
            self.lm_head,
            counters=counters,
            depth=depth,
            segment_len=segment_len,
            halt=halt,
            halt_min_iter=self.cfg.halt_min_iter,
            collect_hidden=self.halt is not None and targets is not None and not pure,
            loss_mask=loss_mask,
            ce_fixed=ce_fixed,
        )
        # The coda: unique layers, once, after the loop's mean readout. They sit
        # BEFORE the final norm/head, which stay exactly where they are today
        # (`embed → prelude → loop ×T → coda → RMSNorm → lm_head`).
        for layer in self.coda:
            out = layer(out)
            counters.count("coda")
        # fp32 logits — the burn rule that fixes `bf16 logits NaN`: the final
        # norm and the head run in fp32 whatever the body computed in. The
        # cast is the identity on the fp32 path (`residual_dtype="fp32"`), and
        # it is what keeps the readout's storage dtype from reaching the head.
        hn = self.norm(out.to(torch.float32))
        logits = self.lm_head(hn.reshape(-1, self.cfg.d_model).to(torch.float32))
        if self.cfg.logit_softcap > 0.0:
            # Gemma-2's final-logit cap (`losses.logit_softcap`): applied to
            # the head output BEFORE the final CE, z-loss and the returned
            # logits, so a training step and the held-out objective score the
            # same bounded distribution. `0.0` enters no branch.
            logits = logit_softcap(logits, self.cfg.logit_softcap)
        loss = None
        # Set only on the soft-MTP arm; the hard path leaves it None and keeps
        # its MTP call exactly where it was, so `mtp_soft=false` is the old
        # program (gate 1 pins the state and logits).
        mtp_terms: tuple[torch.Tensor, list[torch.Tensor], torch.Tensor] | None = None
        if targets is not None:
            assert rec_sum is not None, "targets given but the loop accumulated no L_Rec"
            # (Σ_i CE_i + CE_final) / (max_iter + 1), unweighted.
            #
            # DEVIATION, deliberate and documented: burn's byte path trains on
            # the loop's L_Rec ALONE — `model.rs` returns `rec` and never puts
            # its own `norm`+`lm_head` logits into a loss, so its final norm
            # gain receives NO gradient while the loss curve looks perfectly
            # healthy. That is the `8fa5d4c` shape (an arm that runs forever
            # and trains nothing) on a parameter nobody counted. The port
            # keeps L_Rec — deep supervision over every iteration, which is
            # what a looped block needs (readiness plan §5) — and adds the one
            # term that makes the logits it RETURNS the logits it TRAINS.
            #
            # (Σ_i CE_i + CE_final) / (T + 1), unweighted, with T the depth this
            # forward ACTUALLY ran. burn's code divided by `max_iter + 1`
            # unconditionally; under the ADAPTIVE DEPTH arm that divisor is a
            # constant that no longer counts the terms summed, so a T=1 step
            # would read as the loss of a max_iter-deep model (and the arm's
            # whole point — scale-comparable depths — would be invisible in
            # the curve). One divisor, the count of terms it divides. It is
            # computed HERE, before the CE, because soft-MTP's scale needs it.
            T = self.cfg.max_iter if depth is None else depth
            if halt is not None:
                # The depth the halted loop ACTUALLY ran: the loss divides by
                # the terms it summed, exactly as a scheduled depth does.
                T = getattr(self.loop, "_depth_ran", T)
            if not pure and self.mtp is not None and self.cfg.mtp_soft:
                # SOFT-MTP (arXiv:2610.38331): the MTP modules compute their
                # own CE and return the NTP hidden with the straight-through
                # coupling added. The CE below then reads THAT hidden; the
                # returned `logits` and the aux terms keep the original one
                # (the coupling is a forward no-op — gate 3).
                assert loss_mask is None, (
                    "mtp_soft with loss_mask: MTP has no ignore_index, so the "
                    "coupled slice and the masked NTP mean would address "
                    "different position sets; config.validate refuses "
                    "doc_masking with mtp_weight>0"
                )
                assert self.cfg.logit_softcap == 0.0, (
                    "mtp_soft with logit_softcap: the MTP head reads the shared "
                    "head uncapped and the NTP CE would read a capped "
                    "distribution; ModelConfig.__post_init__ refuses the pair"
                )
                mtp_terms = self.mtp.forward_soft(
                    out,
                    input_ids,
                    targets,
                    embed=self.embed,
                    head=self.lm_head,
                    weight=self.cfg.mtp_weight,
                    ntp_hidden=hn,
                    # The NTP CE's factor in the total: `(rec_sum + ce)/ (T+1)`.
                    ntp_loss_scale=1.0 / (T + 1),
                )
                # A DEVICE counter: the arm's "it ran" proof, drained with the
                # window's other counters at the log cadence (never read here).
                counters.count_device(
                    "mtp_soft",
                    torch.ones((), dtype=torch.int64, device=hn.device),
                )
                logits_ntp = self.lm_head(
                    mtp_terms[2].reshape(-1, self.cfg.d_model).to(torch.float32)
                )
            else:
                logits_ntp = logits
            flat = targets.reshape(-1, 1)
            if ce_fixed:
                ce_final = fixed_cross_entropy(
                    logits_ntp, targets, reduction="none", counters=counters
                ).reshape(-1, 1)
            else:
                ce_final = -F.log_softmax(logits_ntp, dim=-1).gather(1, flat)
            if loss_mask is not None:
                # doc_masking: padding and cross-document targets carry no loss
                # and no gradient — the same selected mean the loop used.
                ce_final = masked_ce_mean(ce_final, loss_mask)
            else:
                ce_final = ce_final.mean()
            if breakdown:
                # Eval-only host reads (called under `torch.no_grad()` by
                # `evaluate(..., breakdown=True)`; NEVER in the training path —
                # the 0-sync rule). `fin` is the depth-comparable number: the CE
                # of the mean readout at the depth actually run.
                # With the halt policy on, the number of iterations varies per
                # batch, so a fixed-depth `periter=` column would be a misreport;
                # `fin` (the readout at the depth actually run) is still exact.
                per = [] if halt is not None else (getattr(self.loop, "_per_iter_ce", None) or [])
                self.last_breakdown = {
                    "fin": float(ce_final.detach()),
                    "per_iter": [float(t.detach()) for t in per],
                }
            # (Σ_i CE_i + CE_final) / (T + 1), unweighted, with T the depth this
            # forward ACTUALLY ran (computed above, before the CE expression).
            loss = (rec_sum + ce_final) / (T + 1)
            if not pure and self.cfg.moe_lb_mode == "aux":
                # The Switch/GShard load-balance term (`experts.switch_aux_loss`,
                # weighted by `moe_aux_coef` in the MoE): a ROUTING signal, so it
                # is added to the objective exactly like z-loss, and `pure` (the
                # held-out number) skips it. The counter is bumped where the term
                # enters the loss, not where it is computed (ADR-0019): a `pure`
                # eval must not report an arm it never added.
                moe_aux = self.loop.moe.aux_loss()
                if moe_aux is not None:
                    loss = loss + moe_aux
                    counters.count("moe_aux")
            if not pure and self.cfg.z_loss > 0.0:
                # z-loss (PaLM §5 / Gemma-2 §9): keep the pre-softmax logits
                # from drifting tall. It is added AFTER the CE mean that is the
                # objective, over the FINAL logits only (the same ones the
                # forward returns), and `pure` skips it for the held-out number
                # exactly as it skips every other training signal. The counter
                # is the void check: a term only in the config must not look
                # alive in a log.
                counters.count("z_loss")
                loss = loss + self.cfg.z_loss * torch.logsumexp(logits, dim=-1).pow(2).mean()
            if not pure and self.halt is not None:
                # The HALT TERM (`halt.py`): BCE over the iterations this step
                # actually ran, against a target read from the model's OWN
                # per-iteration CE (the seam `periter` prints) and never from the
                # label sequence. `halt_loss` detaches its input, so the head
                # learns from the states without bending them.
                hidden = getattr(self.loop, "_per_iter_hidden", None)
                per = getattr(self.loop, "_per_iter_ce", None)
                assert hidden is not None and per is not None and len(hidden) == len(per), (
                    "halt_weight>0 but the loop exposed no per-iteration states/CE: "
                    "the head would train on nothing (collect_hidden was not set)"
                )
                term = halt_loss(self.halt, hidden, per, self.cfg.halt_eps, self.cfg.max_iter)
                if term is not None:
                    loss = loss + self.cfg.halt_weight * term
                    self.last_halt = term.detach()
                    if breakdown:
                        self.last_breakdown["halt"] = float(term.detach())
            # `pure=True` reads the PREDICTIVE CE alone (the aux terms are
            # training signals, not part of the model's held-out quality).
            # Without it the A/B compares two DIFFERENT quantities: an aux
            # arm's eval loss carries +w*aux_term and reads ~0.3 BPB worse by
            # construction (measured on the first A/B run, 2026-10-16 —
            # control 3.045 vs fb 3.469 while the arm's own counters showed
            # the head was fine).
            if not pure and self.cfg.sigreg_weight > 0.0:
                # SIGReg (arXiv:2511.08544, `sigreg.py`): ONE term on the state
                # `lm_head` already reads, no teacher, no predictor, no mask, no
                # second forward, no parameters, and NO detach — the gradient
                # reaches the loop (LeJEPA's "no stop-gradient"). The counter is
                # bumped exactly here, where the term is added: the research
                # doc's metric 2 is a term that was never added looking alive
                # (`docs/research/2026-10-16-jepa-into-dormouse.md` §3).
                counters.count("sigreg")
                loss = loss + self.cfg.sigreg_weight * sigreg_loss(
                    stride_sample(hn.reshape(-1, self.cfg.d_model).to(torch.float32))
                )
            if not pure and self.jepa is not None:
                # JEPA (`jepa.py`) as ONE term: the causal latent prediction
                # (`1 - cos(pred(h[p]), sg(h[p+h])))`) PLUS its own SIGReg
                # anti-collapse term on the predicted latents, scaled by
                # `jepa_reg_weight` INSIDE the head. `jepa_weight` scales the
                # sum, so the mechanism cannot be split into two arms by a
                # config. The counter is the seam's void check, as sigreg's is.
                counters.count("jepa")
                loss = loss + self.cfg.jepa_weight * self.jepa(out.to(torch.float32))
            if not pure and self.energy is not None:
                # The ENERGY term (EBT arXiv:2507.02092, `energy.py`): the target
                # byte's energy must beat `energy_negs` negatives sampled from
                # the `lm_head` softmax. The scorer reads `(h, candidate)` only;
                # targets pick the positive and are never an input to the score.
                # The counter is the A/B's void check (as sigreg's is).
                counters.count("energy")
                loss = loss + self.cfg.energy_weight * energy_contrastive_loss(
                    self.energy,
                    hn.reshape(-1, self.cfg.d_model).to(torch.float32),
                    self.embed.weight.to(torch.float32),
                    targets.reshape(-1),
                    logits,
                    n_negs=self.cfg.energy_negs,
                    temp=self.cfg.energy_temp,
                )
            if not pure and self.dspark is not None:
                # The DSpark draft head (`dspark.py`, standalone module; this
                # hook is its integration commit): γ is the config weight (D2 —
                # the loss returns γ-free), the window reads the CONSUMED
                # sequence and the byte ids the lm_head just scored (D1), and
                # `dspark=<windows>/<calls>` on the log line is the void check
                # (0/<n> = the length budget allowed no window, counted).
                # `logits` is [b*t, vocab] here; the head's contract is
                # [b, t, vocab] (its own assert caught exactly this on the
                # first A/B run — counted as a FAIL in the driver, fixed here).
                b_tok, t_tok = input_ids.shape
                loss = loss + self.cfg.dspark_weight * dspark_aux_loss(
                    self.dspark,
                    hn.to(torch.float32),
                    input_ids,
                    logits.reshape(b_tok, t_tok, -1),
                    counters=counters,
                )
            if not pure and self.mtp is not None:
                # Multi-Token Prediction (DeepSeek-V3 §3.2, `mtp.py`): K
                # future-byte modules on the loop's own final readout, sharing
                # THIS model's `embed` and `lm_head`. The term is a training
                # signal like DSpark's, so `pure` (the held-out number) skips it.
                # The soft arm computed its terms above (the CE had to read the
                # coupled hidden); the hard arm computes them HERE, unchanged.
                if mtp_terms is None:
                    mtp_loss, mtp_per_k = self.mtp(
                        out,
                        input_ids,
                        targets,
                        embed=self.embed,
                        head=self.lm_head,
                        weight=self.cfg.mtp_weight,
                    )
                else:
                    mtp_loss, mtp_per_k, _coupled = mtp_terms
                loss = loss + mtp_loss
                # The value `train/loop.py`'s `mtp_field` reads on the LOG
                # cadence: detached, and NOT read here — a `.item()` per step
                # is the host sync ADR-0018 rule 2 bans.
                self.last_mtp = mtp_loss.detach()
                if breakdown:
                    self.last_breakdown["mtp"] = float(mtp_loss.detach())
                    self.last_breakdown["mtp_per_k"] = [float(t.detach()) for t in mtp_per_k]
            # No finiteness assert here: `bool(tensor)` is a host read on the
            # loss scalar, and the trainer's DEVICE firewall (`train/loop.py`)
            # owns the decision — it masks a non-finite loss and gates the
            # update on the device (ADR-0018 rule 2). The assert used to fire
            # before the firewall could see the value.
        return logits, loss

    def _forward_transformer(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        counters: Counters,
        loss_mask: torch.Tensor | None,
        ce_fixed: bool,
        breakdown: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`block_form="transformer"`: embed → unique blocks → norm → head → CE.

        The objective is the SAME final-CE term `_forward_byte_lm` adds (the
        control's own `F.cross_entropy`, here through `_byte_ce`), and `fin`
        is the number `breakdown` reports for a looped model — one readout, so
        `per_iter` is empty. No aux term, no L_Rec, no per-iteration CE: those
        are loop-only and `validate_block_form` refuses their weights.

        THE mHC LANE SEAM (`use_mhc`). The loop carries its K lanes across
        ITERATIONS (`loopblock._window`); this form carries the same K
        full-width lanes across LAYERS, one write per block:

            x_l    = the K lanes (the stack state, `c·P` initialised that way)
            ctx    = H_pre x_l                 # Eq. 3's read, convex / c
            y      = block(ctx) - ctx          # the block's own write
            a      = y                         # mHC only
                   | AttnRes([0, y])           # the composition: WHICH update
            x_l'   = H_res x_l + H_postᵀ a     # Eq. 3, unchanged

        READOUT, copied from the loop form exactly: the mHC-ONLY arm reads the
        unnormalised SUM of the lanes; the composition applies the paper's
        `1/K` (2409.19606 §2.1). The loop's lanes carry the state and its sum
        is `K×`, which the composition's `1/K` undoes; this form's mHC-only
        lanes carry `1/K` so its raw sum IS the state, and the composition's
        lanes carry the state so its `1/K` read is the state. Same two
        expressions, and both arms hand `_readout` the plain residual state at
        step 0 — the identity `test_mhc_transformer.py` asserts (a `K×` state
        would fail the final RMSNorm's `1e-3` eps).

        The composition is `loopblock.py`'s: mHC owns the lane STATE, AttnRes
        owns WHICH UPDATE the write carries. The update set is `S = 2` —
        `u_0 = 0` (the embedding is already in the lanes) and `u_1 = y` — and
        `FullDepthAttnRes`'s per-layer zero query makes it exactly `y` at init.
        """
        h = self.embed(input_ids)
        mod_aux = None
        attnres = getattr(self, "attnres_full", None)
        mhc = getattr(self, "mhc_full", None)
        if mhc is not None:
            h, mod_aux = self._forward_transformer_mhc(h, counters, attnres)
        else:
            v_first = None
            # `use_attnres` only: the update history the blocks mix. `None` keeps
            # the plain call signature and allocation-free path of the control.
            updates: list[torch.Tensor] | None = [] if attnres is not None else None
            for block in stepaudit.iterations(
                "forward.transformer_blocks", self.transformer_blocks
            ):
                if updates is None:
                    h = block(h, counters, v_first=v_first)
                else:
                    # `len(updates)` IS this block's layer index: one entry per
                    # completed block, so block 0 sees an empty history.
                    h = block(
                        h,
                        counters,
                        v_first=v_first,
                        attnres=attnres,
                        layer_idx=len(updates),
                        updates=updates,
                    )
                v_first = block.last_v
                counters.count("transformer_block")
                aux = getattr(block, "last_aux", None)
                if aux is not None:
                    mod_aux = aux if mod_aux is None else mod_aux + aux
        self.last_mod_aux = None
        if mod_aux is not None:
            # The arm's BCE value is ONE number per forward: the mean over the
            # routed blocks (the paper's per-block term, not per-token). The
            # trainer weights it by `mod_aux_weight` and logs it as `mod_aux=`.
            mod_aux = mod_aux / len(self.transformer_blocks)
            self.last_mod_aux = mod_aux.detach()
        logits, loss = self._readout(h, targets, counters, loss_mask, ce_fixed, breakdown)
        if loss is not None and mod_aux is not None and self.cfg.mod_aux_weight:
            # The paper's §3.5 aux, weighted: `CE + α·BCE(router, top-k)`.
            # `mod_aux` stays attached so the router learns from it.
            loss = loss + self.cfg.mod_aux_weight * mod_aux
        return logits, loss

    def _forward_transformer_mhc(
        self,
        h: torch.Tensor,
        counters: Counters,
        attnres: FullDepthAttnRes | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The K-lane stack (`use_mhc`, `_forward_transformer`'s docstring).

        `h` is the embedding; returns the read-out residual state and the MoD
        aux value (or `None`). Every lane write is ONE `MHCMix.forward`, the
        same Eq. 3 statement the loop runs, and the readout is the loop's two
        expressions (sum, or `/K` in the composition).
        """
        mhc = self.mhc_full
        k = mhc.n_streams
        # `c`: the state each lane carries (`1` composing, `1/K` mHC-only).
        c = self._mhc_lane_scale
        compose = attnres is not None
        b, t, d = h.shape
        # Replicated from the embedding, as the loop does (`2409.19606 §2.1`).
        streams = (h * c).unsqueeze(-2).expand(b, t, k, d)
        v_first = None
        mod_aux = None
        layer = 0
        for block in stepaudit.iterations(
            "forward.transformer_blocks", self.transformer_blocks
        ):
            # Eq. 3's `H_pre x_l`, applied by the caller (`mhc.py`): a convex
            # mix over the K lanes, divided by `c` so the block reads the state.
            ctx = mhc.pre_mix(streams, convex=True, scale=1.0 / c)
            plain = block(ctx, counters, v_first=v_first)
            # The block's write is its residual DELTA — the same `F(H_pre x_l)`
            # the loop's body emits (its two residual adds live inside `_block`).
            y = plain - ctx.to(plain.dtype)
            # The composition scores the write UPDATES `[u_0 = 0, u_1 = y]`
            # with this layer's pseudo-query; at the zero-query init every
            # weight is 1, so `a` is `y` bit for bit.
            a = attnres([y], layer) if compose else y
            # `post_sum = K·c`: the K deposits re-sum to the lane scale, so the
            # write adds exactly `y` to the read-out state (see `MHCMix.forward`).
            streams = mhc(streams, a, post_sum=float(k) * c)
            # The loop's readout, expression for expression.
            h = mhc_readout(streams, compose)
            v_first = block.last_v
            counters.count("transformer_block")
            counters.count("mhc")
            if compose:
                counters.count("mhc_attnres_write")
            aux = getattr(block, "last_aux", None)
            if aux is not None:
                mod_aux = aux if mod_aux is None else mod_aux + aux
            layer += 1
        return h, mod_aux

    def _readout(
        self,
        h: torch.Tensor,
        targets: torch.Tensor | None,
        counters: Counters,
        loss_mask: torch.Tensor | None,
        ce_fixed: bool,
        breakdown: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`norm` + `lm_head` + the single final CE, shared by both STACK forms.

        ONE readout for `block_form="transformer"` and `block_stack="hybrid"`:
        the two forms differ only in the blocks, so a second copy of the head or
        the masked/fixed CE would be a second set of rules for the same number.
        `breakdown` is the eval-only host read (never in a training step).
        """
        hn = self.norm(h.to(torch.float32))
        logits = self.lm_head(hn.reshape(-1, self.cfg.d_model).to(torch.float32))
        if self.cfg.logit_softcap > 0.0:
            logits = logit_softcap(logits, self.cfg.logit_softcap)
        loss = None
        if targets is not None:
            if loss_mask is None:
                loss = _byte_ce(logits, targets, fixed=ce_fixed, counters=counters)
                assert loss is not None
            else:
                if ce_fixed:
                    ce = fixed_cross_entropy(
                        logits, targets, reduction="none", counters=counters
                    ).reshape(-1, 1)
                else:
                    ce = -F.log_softmax(logits, dim=-1).gather(1, targets.reshape(-1, 1))
                loss = masked_ce_mean(ce, loss_mask)
            if breakdown:
                self.last_breakdown = {"fin": float(loss.detach()), "per_iter": []}
        return logits, loss

    def _forward_hybrid(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        counters: Counters,
        loss_mask: torch.Tensor | None,
        ce_fixed: bool,
        breakdown: bool,
        segment_len: int = 0,
        depth: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`block_stack="K:A"`: embed → ordered KDA/attention blocks → readout.

        The KDA bodies are the SAME `LoopBlock` the kda form builds, run ONCE
        (`depth=1`) each: no weight sharing, no L_Rec, so the objective is the
        transformer form's single final CE and the two controls are one number
        apart. The attention bodies are the control's `TransformerBlock`, init'd
        with the residual scaling of the FULL stack depth (`n_layers`). A
        `LoopBlock` gets `targets=None`, which is also why the deep-supervision
        weights are refused on this stack: the blocks receive no per-pass CE.
        """
        if segment_len:
            loud(
                f"segment_len={segment_len} with block_stack="
                f"{self.cfg.block_stack!r}: each KDA block runs once and the "
                f"attention blocks carry no state, so there is no single "
                f"recurrence whose state crosses a cut. Escape: segment_len=0."
            )
        if depth is not None:
            loud(
                f"depth={depth} with block_stack={self.cfg.block_stack!r}: the "
                f"hybrid stack's depth is fixed by n_layers="
                f"{self.cfg.n_layers}; an adaptive depth moves a LOOP's iteration "
                f"count, and this stack has no loop. Escape: depth=None."
            )
        h = self.embed(input_ids)
        for block in stepaudit.iterations("forward.hybrid_blocks", self.hybrid_blocks):
            if isinstance(block, LoopBlock):
                h, _, _ = block(h, input_ids, None, None, counters=counters, depth=1)
                counters.count("hybrid_kda")
            else:
                h = block(h, counters)
                counters.count("hybrid_attn")
            counters.count("hybrid_block")
        return self._readout(h, targets, counters, loss_mask, ce_fixed, breakdown)


class TiedHead(nn.Module):
    """`lm_head` as `embed.weight`, transposed — ZERO parameters of its own.

    The embedding is held through a LIST (`self._owner`): assigning the Module
    as an attribute would register it as a child, and `state_dict()` would then
    carry a second key (`lm_head.embed.weight`) for the one matrix. The list
    keeps the reference out of `_modules`, so a tied model's state dict has the
    single `embed.weight` key the checkpoint format expects.

    The product runs in fp32 (`x @ W.T`), the burn rule that the final norm and
    head stay fp32 even in bf16 mode (`bf16 logits NaN`). The shared
    parameter's gradient is the SUM of the embedding-lookup path and every head
    path — the loop's per-iteration deep-supervision reads (`L_Rec`) and this
    forward's final logits both accumulate into `embed.weight` — which is the
    whole of what "tied" means and what `tests/test_train4.py` asserts against
    a hand-built two-path model.
    """

    def __init__(self, embed: nn.Embedding) -> None:
        super().__init__()
        self._owner = [embed]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(torch.float32), self._owner[0].weight.to(torch.float32))


def _byte_ce(
    flat_logits: torch.Tensor,
    targets: torch.Tensor | None,
    *,
    fixed: bool = False,
    counters: Counters | None = None,
) -> torch.Tensor | None:
    """The byte-lm CE: mean over positions and batch of the target log-prob.

    Shared by both byteflow shapes (A and B) so the two cannot disagree about
    what the objective is; `None` on a decode forward ("no labels" is not
    "loss 0"). `fixed` selects `losses.fixed_cross_entropy`
    (arXiv:2610.31642 §2); `False` is the historical line, bit for bit.
    """
    if targets is None:
        return None
    if fixed:
        return fixed_cross_entropy(flat_logits, targets, counters=counters)
    loss = -F.log_softmax(flat_logits, dim=-1).gather(1, targets.reshape(-1, 1)).mean()
    # No finiteness assert: see `_forward_byte_lm`'s note — the device
    # firewall owns the NaN decision and cannot be preempted by a host read.
    return loss


def param_count(model: nn.Module) -> int:
    """Named parameters actually built — the number a preset is priced by.

    The burn side's lesson (`preset_exec.rs`): a closed-form cost that drifts
    from the module tree is exactly the kind of number that reads true and
    lies, so the port counts the instantiated model — on the `meta` device for
    a b1-sized config, where there is no storage to pay for.
    """
    return sum(p.numel() for p in model.parameters())
