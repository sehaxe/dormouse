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

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from ..instrument import Counters, loud
from .byteflow import (
    BfChannel,
    ByteFlowNet,
    bf_for_model,
    byteflow_config,
    compat_armed,
    count_channel,
)
from .dspark import DSparkHead, dspark_aux_loss
from .energy import EnergyHead, energy_contrastive_loss
from .halt import HaltHead, halt_loss
from .jepa import JepaHead
from .loopblock import LoopBlock, masked_ce_mean
from .mtp import MTPHead
from .prelude import UniqueLayer
from .qat_cache import QuantCache, attach_qat_caches, take_qat_refresh_counts
from .sigreg import sigreg_loss, stride_sample
from .tsct import make_projection, set_factor_qat_seed


class DormouseModel(nn.Module):
    def __init__(self, cfg: ModelConfig, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
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
        self.bf_net: ByteFlowNet | None = None
        self.bf: BfChannel | None = None
        #: The MTP arm (`mtp.py`) and the scalar its own log line reads. The
        #: tensor is DETACHED by the forward and read only on the log cadence
        #: (`train/loop.py`'s `mtp_field`) — never per step (ADR-0018 rule 2).
        self.mtp: MTPHead | None = None
        self.last_mtp: torch.Tensor | None = None
        #: The ENERGY head (`energy.py`, EBT arXiv:2507.02092): inference
        #: reranker and an OPTIONAL contrastive training term, built IFF
        #: `energy_weight > 0`. Declared before the byteflow early return so a
        #: channel model has the attribute (None) rather than no attribute.
        self.energy: EnergyHead | None = None
        #: JEPA (`jepa.py`): declared before the byteflow early return so a
        #: channel model has the attribute (None) rather than no attribute.
        self.jepa: JepaHead | None = None
        #: Every QAT cache in the tree (`qat_cache.QuantCache`), collected after
        #: the build below. `forward` drains their hit counts into `qat_reuse`;
        #: the trainer invalidates them once per optimizer step.
        self._qat_caches: list[QuantCache] = []
        if cfg.use_byteflow and not compat_armed(cfg):
            self.bf_net = ByteFlowNet(byteflow_config(cfg), device=device)
            return
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab, d, device=device)
        #: The UNIQUE layers outside the shared loop (`prelude.py`, arXiv
        #: 2610.23036): `N` run once before it, `M` once after it. Empty at the
        #: defaults (0/0), so no parameter, no RNG draw and no state dict key —
        #: the off path is the `c8a2ff9` tree bit for bit (`tests/test_prelude.py`).
        self.prelude = nn.ModuleList(
            [UniqueLayer(cfg, device=device) for _ in range(cfg.prelude_layers)]
        )
        self.loop = LoopBlock(cfg, device=device)
        #: The DISTINCT bodies inside the loop (`loop_blocks`, `config.py`):
        #: `self.loop` is body 0 and owns the per-iteration slots; each extra is
        #: a `body_only` LoopBlock (its own controller/KDA/Engram/MoE bank/
        #: readout/mHC mixer, no second `iter_embed`/`attnres`/`*_gate` set).
        #: At `loop_blocks=1` the ModuleList is EMPTY: no parameter, no RNG draw
        #: and no state-dict key, so the off path is main's tree bit for bit
        #: (`tests/test_loop_blocks.py` pins the sha256).
        self.loop_extra = nn.ModuleList(
            [LoopBlock(cfg, device=device, body_only=True) for _ in range(cfg.loop_blocks - 1)]
        )
        #: Set on the DRIVER, not passed by every caller: the one-shot forward,
        #: the segmented path, `recurrent_step` and decode's `_window` all
        #: resolve `loop._blocks`, so no path can silently run block 0 alone.
        self.loop._blocks = (self.loop, *self.loop_extra)
        self.coda = nn.ModuleList([UniqueLayer(cfg, device=device) for _ in range(cfg.coda_layers)])
        self.norm = nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
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
        self.mtp = (
            MTPHead(cfg.d_model, cfg.mtp_k, eps=cfg.norm_eps, device=device)
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, Counters]:
        """`input_ids [b,t]` (+ `targets [b,t]`) → (logits `[b*t,vocab]`, loss, counters).

        `depth` is the ADAPTIVE DEPTH arm's per-step loop depth (`None` =
        `max_iter`, ADR-0013's fixed depth). It reaches the loop and the loss
        divides by the depth ACTUALLY run, so steps of different `T` are
        scale-comparable.

        `segment_len` is the SEGMENTED TRAINING path (`loopblock`, off at 0):
        the trainer passes `train.segment_len`, which cuts the window into
        segments and carries the KDA state across them so a window longer than
        one forward fits in VRAM can be trained. It is a memory decision, not a
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
            logits, loss = self._forward_net(input_ids, targets, counters)
        elif self.bf is not None:
            self._refuse_segmented(segment_len)
            logits, loss = self._forward_channel(input_ids, targets, counters, depth)
        else:
            logits, loss = self._forward_byte_lm(
                input_ids, targets, counters, depth, pure, breakdown, segment_len, loss_mask
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Option A: the byteflow net alone — its byte logits ARE the model's."""
        counters.count("bf_net")
        assert self.bf_net is not None
        b, t = input_ids.shape
        flat = self.bf_net(input_ids).reshape(b * t, self.cfg.vocab)
        return flat, _byte_ce(flat, targets)

    def _forward_channel(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        counters: Counters,
        depth: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Option B: front stage → this model's loop on K latents → back stage.

        The objective is the byte CE from the DECODER, over all T positions —
        not the loop's in-loop `L_Rec`. Two reasons, both load-bearing: the
        decoder is the model's only exit, so a loss taken at the K patch starts
        would leave `decode_chunks`' blocks and the byte head with no gradient;
        and the held-out eval scores `[b,t,256]` logits against all T bytes, so
        anything else would be a train CE that is not the quantity on the eval
        line. `sel` is a discrete argtopk product: the SELECTION has no
        gradient, the gathered rows and everything after `z` train normally.
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
        h, z, sel = bf.net.encode_chunks(input_ids, bf.k)
        x = bf.in_proj(z)  # [b, k, d_model]
        # The loop, unchanged, on K patch latents. No byte ids: the memory arm
        # is refused with use_byteflow (validate), so nothing here needs them.
        out_acc, _, _ = self.loop(x, None, None, None, counters=counters, depth=depth)
        logits = bf.net.decode_chunks(bf.out_proj(out_acc), h, sel)  # [b, t, vocab]
        flat = logits.reshape(-1, self.cfg.vocab)
        return flat, _byte_ce(flat, targets)

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The plain byte-level stack: embedding → prelude → loop → coda → RMSNorm → lm_head."""
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
        )
        # The coda: unique layers, once, after the loop's mean readout. They sit
        # BEFORE the final norm/head, which stay exactly where they are today
        # (`embed → prelude → loop ×T → coda → RMSNorm → lm_head`).
        for layer in self.coda:
            out = layer(out)
            counters.count("coda")
        # fp32 logits — the burn rule that fixes `bf16 logits NaN`: the final
        # norm and the head run in fp32 whatever the body computed in.
        hn = self.norm(out)
        logits = self.lm_head(hn.reshape(-1, self.cfg.d_model).to(torch.float32))
        loss = None
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
            flat = targets.reshape(-1, 1)
            ce_final = -F.log_softmax(logits, dim=-1).gather(1, flat)
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
            # forward ACTUALLY ran. burn's code divided by `max_iter + 1`
            # unconditionally; under the ADAPTIVE DEPTH arm that divisor is a
            # constant that no longer counts the terms summed, so a T=1 step
            # would read as the loss of a max_iter-deep model (and the arm's
            # whole point — scale-comparable depths — would be invisible in
            # the curve). One divisor, the count of terms it divides.
            T = self.cfg.max_iter if depth is None else depth
            if halt is not None:
                # The depth the halted loop ACTUALLY ran: the loss divides by
                # the terms it summed, exactly as a scheduled depth does.
                T = getattr(self.loop, "_depth_ran", T)
            loss = (rec_sum + ce_final) / (T + 1)
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
                mtp_loss, mtp_per_k = self.mtp(
                    out,
                    input_ids,
                    targets,
                    embed=self.embed,
                    head=self.lm_head,
                    weight=self.cfg.mtp_weight,
                )
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


def _byte_ce(flat_logits: torch.Tensor, targets: torch.Tensor | None) -> torch.Tensor | None:
    """The byte-lm CE: mean over positions and batch of the target log-prob.

    Shared by both byteflow shapes (A and B) so the two cannot disagree about
    what the objective is; `None` on a decode forward ("no labels" is not
    "loss 0").
    """
    if targets is None:
        return None
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
