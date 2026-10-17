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
from .fb import FutureByteHead, future_byte_loss
from .loopblock import LoopBlock
from .sigreg import sigreg_loss, stride_sample
from .tsct import make_projection, set_factor_qat_seed


class DormouseModel(nn.Module):
    def __init__(self, cfg: ModelConfig, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
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
        # The DSpark draft head (2607.05147): built only when the weight is on
        # — at 0.0 the parameter set and every preset count match today's.
        self.dspark: DSparkHead | None = (
            DSparkHead(cfg, device=device) if cfg.dspark_weight > 0.0 else None
        )

    def set_factor_qat_alpha(self, alpha: float) -> None:
        """Set the annealing alpha on EVERY factor_qat consumer (one call per step).

        The consumers are the `TSCTLinear`s (loop block + head),
        `DualLatentGLU`s (routed + shared experts) and `SparseMoE._banks`'
        cached stacks — the bank module reads its SAME-named attributes, so
        walking `modules()` and setting the attribute is the complete set.
        The bank cache keys on `factor_qat_alpha` (see `_banks`), so a changed
        alpha rebuilds the quantized stacks on the next forward; a constant
        alpha (the default) rebuilds nothing extra.
        """
        assert 0.0 <= alpha <= 1.0, f"anneal alpha {alpha} outside 0..=1"
        for m in self.modules():
            if hasattr(m, "factor_qat_alpha"):
                m.factor_qat_alpha = alpha

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        counters: Counters | None = None,
        depth: int | None = None,
        pure: bool = False,
        breakdown: bool = False,
        segment_len: int = 0,
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
        """
        counters = Counters() if counters is None else counters
        assert input_ids.dtype in (torch.long, torch.int64), "input_ids must be integer ids"
        assert input_ids.ndim == 2, f"input_ids must be [b, t], got {tuple(input_ids.shape)}"
        if self.bf_net is not None:
            self._refuse_segmented(segment_len)
            logits, loss = self._forward_net(input_ids, targets, counters)
        elif self.bf is not None:
            self._refuse_segmented(segment_len)
            logits, loss = self._forward_channel(input_ids, targets, counters, depth)
        else:
            logits, loss = self._forward_byte_lm(
                input_ids, targets, counters, depth, pure, breakdown, segment_len
            )
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The plain byte-level stack: embedding → loop → RMSNorm → lm_head."""
        h0 = self.embed(input_ids)
        out, rec_sum, counters = self.loop(
            h0,
            input_ids,
            targets,
            self.lm_head,
            counters=counters,
            depth=depth,
            segment_len=segment_len,
        )
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
            if breakdown:
                # Eval-only host reads (called under `torch.no_grad()` by
                # `evaluate(..., breakdown=True)`; NEVER in the training path —
                # the 0-sync rule). `fin` is the depth-comparable number: the CE
                # of the mean readout at the depth actually run.
                per = getattr(self.loop, "_per_iter_ce", None) or []
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
            loss = (rec_sum + ce_final) / (T + 1)
            if pure:
                # `pure=True` reads the PREDICTIVE CE alone (the aux terms are
                # training signals, not part of the model's held-out quality).
                # Without this the A/B compares two DIFFERENT quantities: the
                # fb/dspark arms' eval loss carries +w*aux_term and reads ~0.3
                # BPB worse by construction (measured on the first A/B run,
                # 2026-10-16 — control 3.045 vs fb 3.469 while the fb ARM's
                # own counters showed the head was fine).
                pass
            elif self.fb is not None:
                # The same readout `lm_head` reads, the same graph (`fb.py`):
                # no teacher forward, no detach, no second pass over the loop —
                # the A/B arm is this term and nothing else.
                loss = loss + self.cfg.aux_fb_weight * future_byte_loss(
                    self.fb, hn.to(torch.float32), targets, counters
                )
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
