"""The weight-shared loop block — one parameter set, `max_iter` iterations.

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
   at 0 the body gets no gradient at all) or by **AttnRes**' pseudo-query
   mixture (mutually exclusive by construction);
7. the readout is `out_proj(h)`, and the final state is the AVERAGE over
   iterations. `L_Rec` — the per-iteration CE through the same `lm_head` — is
   accumulated inside the loop (deep supervision, burn's own shape), which is
   also why it is never materialized as `[T, b, t, d]`.

There is no halt head and no PonderNet: depth is fixed by default and the
per-iteration CE is an unweighted mean (ADR-0013). The ADAPTIVE DEPTH arm
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
from .engram import Engram, memory_floor_mix
from .experts import SparseMoE
from .kda import KDAAttention
from .sparse import BlockSparseAttention
from .tsct import dense_qat_project, make_projection, mm_mixed

__all__ = ["AttnResSlot", "LoopBlock", "depth_attend", "memory_floor_mix"]


def _rms_no_gain(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


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


def depth_attend(sources: list[torch.Tensor], slot: AttnResSlot, eps: float) -> torch.Tensor:
    """The AttnRes mixture: `softmax_i(q_eff · RMSNorm(k_i))` over the sources.

    The score convention is the PAPER's — `q·RMSNorm(k)` with **no `1/√d`**
    (`burn_attnres::ScoreForm::Paper`, the default). The burn side shipped
    `d^-0.5` plus an L2 norm, together a `d`-fold logit compression, both
    silent; the fold `q_eff = γ ⊙ w_l` is fidelity fix F-F7 (the paper's
    affine norm, with its gain).
    """
    q_eff = slot.gain * slot.query
    keys = torch.stack([_rms_no_gain(s, eps) for s in sources], dim=-2)  # [b,t,S,d]
    scores = (keys * q_eff).sum(-1)  # [b,t,S]
    w = F.softmax(scores, dim=-1)
    return (w.unsqueeze(-1) * torch.stack(sources, dim=-2)).sum(-2)


class LoopBlock(nn.Module):
    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
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
        self.norm = nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
        # The three gates only. burn's controller ALSO emits the expert-blend
        # logits and its MoE "adds NO parameters"; the port's MoE owns a
        # dedicated router linear (the baked decision for this port), so the
        # controller carries three columns and no dead ones — a parameter with
        # no gradient is the defect class this project's counters exist for.
        self.controller = nn.Linear(2 * d, 3, bias=False, device=device)
        self.attn = KDAAttention(cfg, device=device) if cfg.use_kda else None
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
            self.msa_gate = nn.Parameter(torch.zeros(cfg.max_iter, device=device))
        self.moe = SparseMoE(cfg, device=device)
        self.engram = Engram(cfg, device=device) if cfg.use_engram else None
        # The dense half of the memory branch's mixture — the `(1-g)·v_t` term.
        self.mem_dense = nn.Linear(d, d, device=device)
        self.out_proj = make_projection(d, d, cfg, rank=cfg.rank, device=device)
        self.iter_embed = nn.Parameter(torch.zeros(cfg.max_iter, d, device=device))
        self.residual_scale = nn.Parameter(torch.ones(1, device=device))
        self.attnres = (
            nn.ModuleList([AttnResSlot(d, device) for _ in range(cfg.max_iter)])
            if cfg.use_attnres
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
            nn.Parameter(torch.zeros(cfg.max_iter, device=device)) if self.loom else None
        )

    def forward(
        self,
        h0: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        lm_head: nn.Module | None = None,
        counters: Counters | None = None,
        depth: int | None = None,
        segment_len: int = 0,
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
            return self._forward_segmented(
                h0, input_ids, targets, lm_head, counters, T, segment_len
            )
        out, rec, per_iter, _ = self._window(
            h0, input_ids, targets, lm_head, counters, T, carry=False
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
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list]:
        """`T` iterations over ONE window — a segment, or the whole sequence.

        The body of the loop, in one place, called by both paths: `carry=False`
        is the one-shot call (`states=None`, `drop=0`, the attention arm called
        through its one-shot `forward`), `carry=True` is the segmented one
        (`states` = the per-iteration state entering this window, `drop` = the
        positions of `input_ids` that belong to the window BEFORE this one).

        `h0`/`input_ids`/`targets` are this window's slice; `input_ids` carries
        `drop` extra LEADING positions when `drop > 0` (see `_memory`).
        Returns `(mean readout, Σ per-iteration mean CE, per-iteration CEs, the
        per-iteration states leaving this window)`.
        """
        b, t, d = h0.shape
        h = h0
        h0_flat = h0.reshape(b * t, d)
        out_acc = torch.zeros_like(h0)
        rec = torch.zeros((), dtype=torch.float32, device=h0.device)
        per_iter: list[torch.Tensor] = []
        sources: list[torch.Tensor] = [h0] if self.use_attnres else []
        new_states: list = []
        for i in range(T):
            counters.count("iter")
            ctx = h + self.iter_embed[i].view(1, 1, d)
            normed = self.norm(ctx)
            ctrl = torch.cat([ctx.reshape(b * t, d), h0_flat], dim=-1)
            # MAGMA fix: fp32 masters with fp32 activations would run the slow
            # fp32 lane; `mm_mixed` casts to the compute dtype per forward.
            # `dense_qat_project` is the identity when the arm is off (same
            # parameter object), so the default path is byte-for-byte today's.
            raw = mm_mixed(
                ctrl,
                dense_qat_project(self.controller.weight, self.dense_qat, self.dense_qat_alpha),
                self.compute_dtype,
                counters,
            )
            w_attn = torch.sigmoid(raw[:, 0:1])
            w_mem = torch.sigmoid(raw[:, 1:2])
            w_ffn = torch.sigmoid(raw[:, 2:3])

            if self.attn is not None:
                attn_out, st = self._attn(
                    normed,
                    counters,
                    carry,
                    None if states is None else states[i],
                    recurrent=recurrent,
                )
                new_states.append(st)
                attn = attn_out.reshape(b * t, d) * w_attn
            else:
                attn = torch.zeros(b * t, d, dtype=torch.float32, device=h0.device)
            if self.msa is not None:
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
                    msa_out = self.msa.forward_k(
                        normed, counters, float(share[i].detach()) * self.msa_capacity_ratio
                    )
                    if msa_out is not None:
                        attn = attn + msa_out.reshape(b * t, d) * w_attn * share[i]
                else:
                    msa_out = self.msa(normed, counters=counters)
                    if msa_out is not None:
                        attn = attn + msa_out.reshape(b * t, d) * w_attn

            if self.engram is not None and input_ids is not None:
                mem = self._memory(normed, input_ids, drop, counters).reshape(b * t, d)
                dense = (
                    (
                        mm_mixed(
                            normed.reshape(b * t, d),
                            dense_qat_project(
                                self.mem_dense.weight, self.dense_qat, self.dense_qat_alpha
                            ),
                            self.compute_dtype,
                            counters,
                        )
                        + self.mem_dense.bias.to(self.compute_dtype)
                    )
                    .to(torch.float32)
                    .reshape(b * t, d)
                )
                branch = memory_floor_mix(mem, dense, w_mem, self.lam_max)
            else:
                branch = torch.zeros(b * t, d, dtype=torch.float32, device=h0.device)

            ffn = self.moe(normed.reshape(b * t, d), counters=counters) * w_ffn
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

            if self.use_attnres:
                counters.count("attnres_write")
                sources.append(y)
                h = depth_attend(sources, self.attnres[i], self.eps)
            else:
                counters.count("rezero_write")
                h = ctx + y * self.residual_scale

            step_out = self.out_proj(h.reshape(b * t, d)).reshape(b, t, d)
            out_acc = out_acc + step_out
            if targets is not None:
                assert lm_head is not None, "per-iteration CE needs the lm_head"
                logits = lm_head(step_out.reshape(b * t, d).to(torch.float32))
                ce = -F.log_softmax(logits, dim=-1).gather(1, targets.reshape(-1, 1))
                rec = rec + ce.mean()
                per_iter.append(ce.mean())
        return out_acc / T, rec, per_iter, new_states

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
        assert self.msa is None, (
            "segment_len with use_msa: the block-sparse arm selects its blocks "
            "INSIDE one window, so segmenting it would change the attention "
            "pattern itself (a boundary block can no longer be selected) - that is "
            "a model change wearing a memory flag. Escape: use_msa=false, or "
            "segment_len=0."
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
