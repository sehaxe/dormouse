"""`DormouseModel`: embedding → loop block → RMSNorm → lm_head.

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

The one aux term the port carries is the FUTURE-BYTE head (arXiv:2404.19737,
`fb.py`), off at its default weight 0.0: a dense untied `d_model -> vocab`
readout over the same `norm(out)` `lm_head` reads, one plain CE on the label
rolled by -1. It is AB-PROTOCOL row 1b's pre-registered competitor to JEPA at
**our 2.1% weighting claim, verified in the research doc**
(`docs/research/2026-10-16-jepa-dspark-alternatives.md` §1.3/§4.1: 196 864
params at our d768 = 2.14% of the burn `small`, no extra forward).

The port also carries one REGULARISER, SIGReg (`sigreg.py`, default weight 0.0):
`CE + sigreg_weight · sigreg_loss(z)` over the strided sample of the same
`norm(out)` state — teacher-free, predictor-free, mask-free, 0 parameters, no
second forward (`docs/research/2026-10-16-jepa-into-dormouse.md` §2/§3, the ONE
arm to run first). A regulariser and a head are different classes, so `validate`
allows it beside `aux_fb_weight`; the one-signal-per-run rule is about two HEADS
in one run.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from ..instrument import Counters
from .byteflow import (
    BfChannel,
    ByteFlowNet,
    bf_for_model,
    byteflow_config,
    compat_armed,
    count_channel,
)
from .fb import FutureByteHead, future_byte_loss
from .loopblock import LoopBlock
from .sigreg import sigreg_loss, stride_sample
from .tsct import make_projection


class DormouseModel(nn.Module):
    def __init__(self, cfg: ModelConfig, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        #: The byteflow arm, under ONE flag with two shapes (`byteflow.py`'s
        #: Option A/B): `bf_net` is the standalone net (no loop arm is on, so
        #: the loop is not built at all — no dormant parameter tree), `bf` is
        #: the channel that runs THIS model's loop over K patch latents. Exactly
        #: one of them is ever non-None, and which one is COUNTED per forward.
        self.bf_net: ByteFlowNet | None = None
        self.bf: BfChannel | None = None
        if cfg.use_byteflow and not compat_armed(cfg):
            self.bf_net = ByteFlowNet(byteflow_config(cfg), device=device)
            return
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab, d, device=device)
        self.loop = LoopBlock(cfg, device=device)
        self.norm = nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
        # burn clamps the HEAD's rank to min(rank, d_model, vocab) explicitly
        # ("a rank above either dimension is not a factorization"); the port
        # does the same, here, and nowhere else.
        head_rank = min(cfg.rank, d, cfg.vocab)
        self.lm_head = make_projection(d, cfg.vocab, cfg, rank=head_rank, device=device)
        self.bf = bf_for_model(cfg, device=device)
        # The future-byte head exists IFF its weight is non-zero (`fb.py`,
        # burn's `future_byte.rs:182`): at 0.0 the parameter set, the checkpoint
        # and every measured preset count are exactly today's, which is what
        # makes pure CE a valid control without a re-baseline.
        self.fb: FutureByteHead | None = (
            FutureByteHead(d, cfg.vocab, device=device) if cfg.aux_fb_weight > 0.0 else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        counters: Counters | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Counters]:
        """`input_ids [b,t]` (+ `targets [b,t]`) → (logits `[b*t,vocab]`, loss, counters)."""
        counters = Counters() if counters is None else counters
        assert input_ids.dtype in (torch.long, torch.int64), "input_ids must be integer ids"
        assert input_ids.ndim == 2, f"input_ids must be [b, t], got {tuple(input_ids.shape)}"
        if self.bf_net is not None:
            logits, loss = self._forward_net(input_ids, targets, counters)
        elif self.bf is not None:
            logits, loss = self._forward_channel(input_ids, targets, counters)
        else:
            logits, loss = self._forward_byte_lm(input_ids, targets, counters)
        counters.count("forward")
        return logits, loss, counters

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
            self.cfg.jepa_weight == 0.0
            and self.cfg.dspark_weight == 0.0
            and self.cfg.aux_fb_weight == 0.0
            and self.cfg.sigreg_weight == 0.0
        ), (
            "the byteflow channel carries no aux term: validate refuses the aux weights "
            "and sigreg_weight with use_byteflow - set them to 0"
        )
        count_channel(counters)
        bf = self.bf
        h, z, sel = bf.net.encode_chunks(input_ids, bf.k)
        x = bf.in_proj(z)  # [b, k, d_model]
        # The loop, unchanged, on K patch latents. No byte ids: the memory arm
        # is refused with use_byteflow (validate), so nothing here needs them.
        out_acc, _, _ = self.loop(x, None, None, None, counters=counters)
        logits = bf.net.decode_chunks(bf.out_proj(out_acc), h, sel)  # [b, t, vocab]
        flat = logits.reshape(-1, self.cfg.vocab)
        return flat, _byte_ce(flat, targets)

    def _forward_byte_lm(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        counters: Counters,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The plain byte-level stack: embedding → loop → RMSNorm → lm_head."""
        h0 = self.embed(input_ids)
        out, rec_sum, counters = self.loop(h0, input_ids, targets, self.lm_head, counters=counters)
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
            ce_final = -F.log_softmax(logits, dim=-1).gather(1, flat).mean()
            loss = (rec_sum + ce_final) / (self.cfg.max_iter + 1)
            if self.fb is not None:
                # The same readout `lm_head` reads, the same graph (`fb.py`):
                # no teacher forward, no detach, no second pass over the loop —
                # the A/B arm is this term and nothing else.
                loss = loss + self.cfg.aux_fb_weight * future_byte_loss(
                    self.fb, hn.to(torch.float32), targets, counters
                )
            if self.cfg.sigreg_weight > 0.0:
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
            assert torch.isfinite(loss), "L_Rec + final CE must be finite"
        return logits, loss


def _byte_ce(flat_logits: torch.Tensor, targets: torch.Tensor | None) -> torch.Tensor | None:
    """The byte-lm CE: mean over positions and batch of the target log-prob.

    Shared by both byteflow shapes (A and B) so the two cannot disagree about
    what the objective is; `None` on a decode forward ("no labels" is not
    "loss 0").
    """
    if targets is None:
        return None
    loss = -F.log_softmax(flat_logits, dim=-1).gather(1, targets.reshape(-1, 1)).mean()
    assert torch.isfinite(loss), "the byte CE must be finite"
    return loss


def param_count(model: nn.Module) -> int:
    """Named parameters actually built — the number a preset is priced by.

    The burn side's lesson (`preset_exec.rs`): a closed-form cost that drifts
    from the module tree is exactly the kind of number that reads true and
    lies, so the port counts the instantiated model — on the `meta` device for
    a b1-sized config, where there is no storage to pay for.
    """
    return sum(p.numel() for p in model.parameters())
