#!/usr/bin/env python3
"""The VRAM ledger: what fits on one 16 GB card, at what batch, and what each lever buys.

Every parameter count below comes from the REAL `ModelConfig` tree instantiated on
the `meta` device (`DormouseModel`, `param_count`), and the optimizer split is the
optimizer's own router (`optim.is_muon_param`), not a second copy of the rule. The
activation model is arithmetic: each component is `b * s * d * bytes * n` with the
multiplicity `n` counted from the autograd save census of this model (one save per
`n` copies of that shape). The one measured anchor is
`port/src/dormouse/model/act_mem.py:4`: at `b8 s1024`, `max_iter 3` the measured
step peak was **7.2 GiB**; `tests/test_vram_ledger.py` asserts this model is within
10 % of it.

WHAT IS MEASURED AND WHAT IS COMPUTED
-------------------------------------
[M] `act_mem.py:4` / `port/scripts/ab_actmem.py:60` — the 7.2 GiB b8/s1024 peak.
[M] `dormouse info` — every parameter count and the `ffn_form`/MAC tables.
[M] `docs/research/2026-11-05-small-model-mfu.md` §0.1/§0.3 — 45.7 TFLOP/s bf16
    measured on this card class; the brief's m400b 14.2 GiB reserved figure (not
    `max_memory_allocated`, and not the anchor).
[M] `act_ckpt.py` — the MXFP4 codec's 17 B per 32 elements and its gradient bound.
[M] `ternary.py` — 2-bit packing = 4 weights/byte + one fp32 scale per row.
[E] everything else is arithmetic over those numbers, with the formula printed.

No GPU command runs unless `--measure` is passed, and that path refuses while a
compute process holds the card (`port/scripts/_lane.py`).

USAGE
-----
    python scripts/vram_ledger.py                 # tables, CPU only
    python scripts/vram_ledger.py --json out.json # same, machine-readable
    python scripts/vram_ledger.py --measure --target spa1b --batch 6 --seq 1024 \
        --muon-dtype bf16
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "port" / "src"))

GIB = 2**30
CARD_GIB = 15.0
PEAK_BF16_TFLOPS = 45.7  # [M] the purchase report's measured bf16 ceiling
MFU = 0.18
CTX_GIB = 1.0  # [M] the fixed CUDA-context + allocator floor in act_mem.py:4-5
MEASURED_PEAK_GIB = 7.2  # [M] act_mem.py:4, b8 x s1024, max_iter 3
REFERENCE_B, REFERENCE_S = 8, 1024

#: The MX codec's exact bytes (`mxfp.bytes_per_elements_mxfp4`): 32 elements =
#: 16 B of e2m1 codes + 1 B e8m0 scale, and a partial tail block is padded whole.
MX_BLOCK = 32
MX_BYTES_PER_BLOCK = 17
#: `QuantSavedHooks`' default gate: a saved tensor below this is left exact.
MX_MIN_NUMEL = 4096

#: The Muon chain keeps ~6 fp32 temporaries per element for the whole chunk
#: (`config.muon_chunk_elems`' docstring: "~6 temporaries per element").
NS_TEMPORARIES = 6


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One model the ledger prices. `cfg` is a real, validated `ModelConfig`."""

    name: str
    cfg: object
    note: str
    ns_chunk: int = 1 << 26


def _base():
    from dormouse.config import load_preset

    return load_preset("flagship_base")


def build_targets() -> dict[str, Target]:
    """The four targets + the measured reference recipe.

    The 400M/1B geometries are chosen so the INSTANTIATED count hits the label
    (not a hand sum): 401.9M, 1012.1M, 1017.7M. The 1B pair is `max_iter=1`
    (the task's 242 MFLOP/B only closes at one pass: 6 x 40.3M = 242); the
    looped 100M/400M keep the project's `max_iter=3`.
    """
    from dormouse.config import validate

    base = _base()
    targets = {
        "flagship": Target(
            "100M flagship_base",
            base,
            "today's 102.68M looped MoE (Engram + CSA + mHC + AttnRes)",
            ns_chunk=1 << 25,
        ),
        "m400": Target(
            "400M MoE",
            dataclasses.replace(
                base,
                d_model=1536,
                d_ffn=4096,
                n_heads=24,
                n_experts=1280,
                moe_topk=2,
                rank=64,
                shared_rank=192,
                share_u=True,
                share_u_mode="plane",
                max_iter=3,
                max_seq_len=8192,
                moe_capacity=0.0,
                engram_rows=524288,
            ),
            "the next ladder step: 401.88M, top-2 of 1280, `max_iter=3`",
        ),
        "dense1b": Target(
            "1B dense",
            dataclasses.replace(
                base,
                ffn_form="dense",
                n_experts=1,
                moe_topk=0,
                share_u=False,
                share_u_mode="off",
                use_tsct=False,
                use_engram=False,
                use_csa=False,
                use_mhc=False,
                use_attnres=False,
                layerrope=False,
                layerrope_form="v1",
                d_model=10496,
                d_ffn=27989,
                n_heads=164,
                max_iter=1,
                max_seq_len=8192,
                attn_latent_rank=256,
                moe_lb_coef=0.0,
                csa_coarse_chunk=0,
                csa_mid_chunk=0,
            ),
            "1012.15M, every parameter active, one plain GLU block",
        ),
        "spa1b": Target(
            "1B sparse (40M active)",
            dataclasses.replace(
                base,
                ffn_form="bank",
                d_model=2048,
                d_ffn=5461,
                n_heads=32,
                n_experts=172,
                moe_topk=2,
                rank=1055,
                shared_rank=192,
                share_u=True,
                share_u_mode="plane",
                use_engram=False,
                layerrope=False,
                layerrope_form="v1",
                use_csa=False,
                use_mhc=False,
                use_attnres=False,
                max_iter=1,
                max_seq_len=8192,
                attn_latent_rank=256,
                moe_capacity=0.0,
                csa_coarse_chunk=0,
                csa_mid_chunk=0,
            ),
            "1017.72M stored, ~40.3M active, top-2 of 172 experts, bank resident",
        ),
    }
    for target in targets.values():
        validate(target.cfg)
    return targets


def reference_target() -> Target:
    """The config `act_mem.py:4` measured: `medium` at `max_iter=3`, the shipped
    recipe's flags, `moe_capacity=0` (the `ab_actmem.py` common set)."""
    from dormouse.config import load_preset, validate

    cfg = load_preset("medium")
    cfg = dataclasses.replace(
        cfg,
        max_iter=3,
        max_seq_len=1024,
        moe_capacity=0.0,
        use_csa=True,
        use_mhc=True,
        use_attnres=True,
        share_u=True,
        share_u_mode="plane",
        use_engram=True,
    )
    validate(cfg)
    return Target("reference (measured)", cfg, "the 7.2 GiB anchor", ns_chunk=1 << 25)


# ---------------------------------------------------------------------------
# parameters and optimizer state
# ---------------------------------------------------------------------------


def _group(name: str, form: str = "bank") -> str:
    """The ledger's parameter groups. Every name lands in exactly one."""
    if name.startswith(("embed", "lm_head")):
        return "embeddings + head"
    if ".engram." in name or ".mem_dense." in name:
        return "Engram memory"
    if (
        ".attn." in name
        or ".csa." in name
        or ".msa." in name
        or ".window_attn." in name
    ):
        return "attention (KDA/CSA)"
    if "out_proj" in name:
        return "attention (KDA/CSA)"
    if ".moe.experts." in name:
        return "FFN (dense)" if form == "dense" else "expert bank (routed)"
    if any(
        t in name for t in (".moe.down.", ".moe.shared.", ".moe.gate_u", ".moe.up_u")
    ):
        return "shared FFN (bank basis)"
    if ".moe.router." in name:
        return "router"
    return "controller + norms"


@dataclass
class ParamLedger:
    total: int
    groups: dict[str, int]
    muon: int
    adamw: int

    @property
    def groups_pct(self) -> dict[str, float]:
        return {k: 100.0 * v / self.total for k, v in self.groups.items()}


def param_ledger(cfg) -> ParamLedger:
    """Instantiate the model on `meta` and split its parameters with the
    optimizer's own `is_muon_param` rule (one source, no second copy)."""
    from dormouse.model import DormouseModel, param_count
    from dormouse.train.optim import is_muon_param

    model = DormouseModel(cfg, device="meta")
    groups: dict[str, int] = {}
    muon = adamw = 0
    for name, p in model.named_parameters():
        n = p.numel()
        groups[_group(name, cfg.ffn_form)] = (
            groups.get(_group(name, cfg.ffn_form), 0) + n
        )
        if is_muon_param(name, p):
            muon += n
        else:
            adamw += n
    total = param_count(model)
    assert sum(groups.values()) == total == muon + adamw, "param groups must partition"
    return ParamLedger(total=total, groups=groups, muon=muon, adamw=adamw)


def optimizer_bytes(
    ledger: ParamLedger, *, muon_dtype: str = "fp32"
) -> tuple[int, int]:
    """`(muon momentum, AdamW exp_avg + exp_avg_sq)` in bytes.

    Muon stores ONE buffer per parameter (`optim.Muon`'s `momentum`), in
    `muon_state_dtype`; AdamW creates `exp_avg`/`exp_avg_sq` with
    `torch.zeros_like(p)` (`_foreach_adamw_step`), i.e. the parameter's fp32
    dtype, two buffers. No fp32 master exists: the parameters themselves are
    fp32 (`model.py`'s mixed-precision rule casts at arm boundaries only).
    """
    assert muon_dtype in ("fp32", "bf16")
    muon = ledger.muon * (2 if muon_dtype == "bf16" else 4)
    adamw = ledger.adamw * 8
    return muon, adamw


def ns_workspace(cfg, ledger: ParamLedger, chunk: int) -> int:
    """The Muon-Newton-Schulz peak: ~6 fp32 temporaries over one chunk."""
    return NS_TEMPORARIES * min(ledger.muon, chunk) * 4


# ---------------------------------------------------------------------------
# activation memory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """One named allocation in the forward/backward graph.

    `elements` is what the autograd graph keeps; `elem_bytes` the saved dtype.
    `packable` is False for the bank cache (a module-owned stack, not an autograd
    save) and for the CUDA context — neither is touched by `saved_tensors_hooks`.
    """

    name: str
    elements: int
    elem_bytes: int
    packable: bool = True
    note: str = ""

    @property
    def nbytes(self) -> int:
        return self.elements * self.elem_bytes


def _bt(cfg, batch: int, seq: int) -> int:
    return batch * seq


def _act_bytes(cfg) -> int:
    """2 for the bf16 compute dtype, 4 for fp32 (`ModelConfig.bf16`)."""
    return 2 if cfg.bf16 else 4


def _res_bytes(cfg) -> int:
    return 2 if getattr(cfg, "residual_dtype", "fp32") == "bf16" else 4


def bank_cache_elements(cfg) -> int:
    """The expanded expert bank, `[E, d, r] + [E, r, f]` (the census's
    `(1024, 64)` and `(64, 2730)` bf16 saves at `medium`)."""
    if cfg.ffn_form != "bank":
        return 0
    return cfg.n_experts * (cfg.d_model * cfg.rank + cfg.rank * cfg.d_ffn)


def bank_ternary_bytes(cfg) -> int:
    """2-bit bank: 4 weights/byte + one fp32 scale per row (`ternary.py`'s own
    worked example, `[E, r]` rows). 0 for the dense form."""
    if cfg.ffn_form != "bank":
        return 0
    weights = cfg.n_experts * cfg.rank * cfg.d_ffn
    return math.ceil(weights / 4) + cfg.n_experts * cfg.rank * 4


def activation_components(
    cfg,
    batch: int,
    seq: int,
    *,
    recompute: str = "none",
) -> list[Component]:
    """The forward/backward activation set, by named component.

    The multiplicities are the autograd save census of this model, not a standard
    transformer's: the loop is KDA (linear attention, no `s^2` scores), mHC keeps
    K lanes, and the MoE bank is stacked once per optimizer step. The census was
    taken with `saved_tensors_hooks` on CPU at `b1/s128` and `b2/s256`; the
    dominant integers below are the ones that came back (16 `b*s*d` fp32 copies
    per iteration, 5 `b*s*2d`, 6 `b*s*d` bf16, one `b*s*f`; see the module
    docstring's provenance). `recompute` shrinks the per-iteration body to the
    checkpoint's surviving input state.
    """
    b, s, d = batch, seq, cfg.d_model
    f, r, k = cfg.d_ffn, cfg.rank, cfg.moe_topk
    bt = _bt(cfg, b, s)
    ba, br = _act_bytes(cfg), _res_bytes(cfg)
    iters = cfg.max_iter
    comps: list[Component] = []

    # Per-window (outside the loop): the embedding read and the final readout.
    comps.append(Component("embedding output", bt * d, br))
    comps.append(Component("final norm + readout", 2 * bt * d, br))
    comps.append(Component("logits (final + per iteration)", bt * cfg.vocab, 4))
    comps.append(Component("CE masks/scalars", bt, 4, packable=False))

    if recompute in ("full", "selective"):
        # The checkpoint node keeps the ITERATION INPUT state: h, the mHC lanes
        # and the carried attention state. The same arithmetic as `_window`'s
        # `checkpoint(body, h, streams, sources, attn_state, ...)`.
        floor = iters * (bt * d * br)
        if getattr(cfg, "use_mhc", False):
            floor += iters * bt * getattr(cfg, "mhc_streams", 2) * d * br
        if getattr(cfg, "use_kda", False):
            floor += iters * bt * d * br  # one carried KDA state [b, t, h*hd]
        if getattr(cfg, "use_csa", False):
            floor += iters * bt * d * br
        comps.append(Component("checkpoint input state", floor, 1, note="kept whole"))
        if recompute == "selective":
            # The READ half is kept: norm/controller reads, q/k/v/g outputs,
            # the KDA/CSA boundary state and the Engram read. The WRITE half
            # (the MoE pass, the residual write, out_proj) is recomputed.
            read = 4 * bt * d * br * iters
            if getattr(cfg, "use_kda", False):
                read += 8 * bt * d * br * iters
            read += 6 * bt * d * ba * iters
            read += 5 * bt * 2 * d * br * iters
            if getattr(cfg, "use_engram", False):
                read += bt * d * ba * iters
            comps.append(Component("selective read half", read, 1, note="kept whole"))
        comps.append(Component("recomputed body", 0, 1, note="excluded from peak"))
        return comps

    # Body, per iteration. Each line is the measured save multiplicity x the
    # tensor's elements.
    comps.append(
        Component("residual stream", 4 * bt * d * iters, br, note="4 copies/iter")
    )
    comps.append(
        Component("norms / normalized reads", 4 * bt * d * iters, br, note="4/iter")
    )
    if getattr(cfg, "use_kda", False):
        comps.append(
            Component("KDA boundary state", 8 * bt * d * iters, br, note="8/iter")
        )
        comps.append(
            Component(
                "attention projections q/k/v/g", 6 * bt * d * iters, ba, note="6/iter"
            )
        )
        comps.append(
            Component(
                "widened projection reads", 5 * bt * 2 * d * iters, br, note="5/iter"
            )
        )
    if getattr(cfg, "use_csa", False):
        # Fine CSA: `[b, h, s, chunk]` scores + the selected-block gather.
        comps.append(
            Component(
                "CSA fine scores",
                bt * cfg.n_heads * getattr(cfg, "csa_chunk", 8) * iters,
                4,
                note="b*h*s*chunk",
            )
        )
        comps.append(
            Component(
                "CSA selected-block gather",
                bt * getattr(cfg, "csa_topk", 32) * cfg.head_dim * iters,
                4,
                note="b*s*topk*hd",
            )
        )
    if getattr(cfg, "use_mhc", False):
        comps.append(
            Component(
                "mHC lanes",
                bt * getattr(cfg, "mhc_streams", 2) * d * iters,
                br,
                note="K lanes/iter",
            )
        )
    comps.append(Component("branch writes", 6 * bt * d * iters, ba, note="6/iter"))
    comps.append(
        Component("widened branch writes", bt * 2 * d * iters, ba, note="1/iter")
    )
    if getattr(cfg, "use_engram", False):
        comps.append(Component("Engram read", bt * d * iters, ba, note="1/iter"))
        comps.append(
            Component(
                "Engram slots",
                bt * len(cfg.engram_orders) * iters,
                8,
                packable=False,
                note="int64 hashes",
            )
        )

    if cfg.ffn_form == "dense":
        comps.append(Component("MLP gate + up", 2 * bt * f * iters, ba))
        comps.append(Component("MLP activation", bt * f * iters, ba))
        comps.append(Component("MLP down input", bt * f * iters, ba))
        comps.append(Component("MLP output", bt * d * iters, ba))
    else:
        # Routed bank: 2 latents per assignment, one write row, plus the shared
        # expert and the shared `f -> d` down (whole token set).
        comps.append(Component("expert gate/up latents", 2 * bt * k * r * iters, ba))
        comps.append(Component("expert write rows", bt * k * f * iters, ba))
        comps.append(
            Component(
                "shared expert",
                bt * (cfg.shared_rank * 2 + cfg.shared_rank) * iters,
                ba,
            )
        )
        comps.append(Component("shared down input", bt * f * iters, ba))
        # The expanded bank cache is a module-owned stack, NOT an autograd save;
        # `peak_bytes` adds it once (and applies the ternary lever there).
    return comps


def activation_bytes(
    cfg,
    batch: int,
    seq: int,
    *,
    recompute: str = "none",
    act_pack: str = "none",
) -> int:
    if recompute == "none" and act_pack == "none":
        return sum(c.nbytes for c in activation_components(cfg, batch, seq))
    comps = activation_components(cfg, batch, seq, recompute=recompute)
    total = 0
    for c in comps:
        if act_pack == "mxfp4" and c.packable and c.elements >= MX_MIN_NUMEL:
            total += math.ceil(c.elements / MX_BLOCK) * MX_BYTES_PER_BLOCK
        else:
            total += c.nbytes
    return total


# ---------------------------------------------------------------------------
# levers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Levers:
    recompute: str = "none"  # none | full | selective
    muon_dtype: str = "fp32"  # fp32 | bf16
    act_pack: str = "none"  # none | mxfp4
    offload_state: bool = False
    zero_cards: int = 1  # 1 | 2 (ZeRO-1 state shard)
    bank_dtype: str = "bf16"  # bf16 | ternary
    grad_accum: int = 1
    segment_len: int = 0

    def label(self) -> str:
        parts = []
        if self.recompute != "none":
            parts.append(f"ckpt={self.recompute}")
        if self.muon_dtype == "bf16":
            parts.append("muon=bf16")
        if self.act_pack != "none":
            parts.append(f"act={self.act_pack}")
        if self.offload_state:
            parts.append("offload-state")
        if self.zero_cards > 1:
            parts.append(f"zero{self.zero_cards}")
        if self.bank_dtype != "bf16":
            parts.append(f"bank={self.bank_dtype}")
        if self.grad_accum > 1:
            parts.append(f"accum={self.grad_accum}")
        if self.segment_len:
            parts.append(f"segment={self.segment_len}")
        return "+".join(parts) or "none"


@dataclass
class Peak:
    params: int
    grads: int
    muon_state: int
    adamw_state: int
    ns_workspace: int
    bank_cache: int
    activations: int
    context: int
    fwd_peak: int
    opt_peak: int

    @property
    def peak(self) -> int:
        return max(self.fwd_peak, self.opt_peak)

    @property
    def peak_gib(self) -> float:
        return self.peak / GIB


def peak_bytes(
    cfg, ledger: ParamLedger, batch: int, seq: int, lv: Levers, ns_chunk: int
) -> Peak:
    muon, adamw = optimizer_bytes(ledger, muon_dtype=lv.muon_dtype)
    state = muon + adamw
    if lv.zero_cards > 1:
        state //= lv.zero_cards
    if lv.offload_state:
        state = 0
    params = ledger.total * 4
    grads = ledger.total * 4
    ns = ns_workspace(cfg, ledger, ns_chunk)
    bbank = bank_cache_elements(cfg) * _act_bytes(cfg)
    if cfg.ffn_form == "bank" and lv.bank_dtype == "ternary":
        bbank = bank_ternary_bytes(cfg)
    act = activation_bytes(
        cfg,
        batch,
        seq,
        recompute=lv.recompute,
        act_pack=lv.act_pack,
    )
    ctx = int(CTX_GIB * GIB)
    fwd = params + grads + state + bbank + act + ctx
    opt = params + grads + state + ns + ctx
    return Peak(
        params=params,
        grads=grads,
        muon_state=muon if not (lv.offload_state or lv.zero_cards > 1) else state,
        adamw_state=adamw,
        ns_workspace=ns,
        bank_cache=bbank,
        activations=act,
        context=ctx,
        fwd_peak=fwd,
        opt_peak=opt,
    )


def _flops_per_byte(cfg) -> float:
    from dormouse.width import active_macs_per_token

    macs = active_macs_per_token(
        cfg.d_model,
        iters=cfg.max_iter,
        n_heads=cfg.n_heads,
        attn_latent_rank=cfg.attn_latent_rank,
        n_kv_heads=cfg.n_kv_heads,
        out_proj_rank=cfg.out_proj_rank,
        f=cfg.d_ffn,
        ffn_form=cfg.ffn_form,
        ffn_rank=cfg.rank,
        n_experts=cfg.n_experts,
        topk=cfg.moe_topk,
        shared_rank=cfg.shared_rank,
    )
    return 6.0 * macs  # training = fwd 2 + bwd 4


def active_params(cfg) -> int:
    """Parameters the forward actually touches per token.

    Dense: every parameter. Bank: share the plane + the shared expert + the
    shared down, and count `topk` of the routed experts. Engram tables are a
    lookup (3 x engram_dim rows per token), not a full read.
    """
    from dormouse.model import DormouseModel

    model = DormouseModel(cfg, device="meta")
    if cfg.ffn_form == "dense":
        return sum(p.numel() for p in model.parameters())
    total = 0
    routed = 0
    for name, p in model.named_parameters():
        n = p.numel()
        if ".moe.experts." in name:
            routed += n
        elif ".engram.tables" in name:
            total += len(cfg.engram_orders) * cfg.engram_dim
        else:
            total += n
    if cfg.n_experts:
        total += routed // cfg.n_experts * cfg.moe_topk
    return total


def max_batch(
    cfg, ledger: ParamLedger, seq: int, lv: Levers, ns_chunk: int, cap: int = 512
) -> int:
    for b in range(1, cap + 1):
        if peak_bytes(cfg, ledger, b, seq, lv, ns_chunk).peak > CARD_GIB * GIB:
            return b - 1
    return cap


def bps_at_mfu(cfg, mfu: float = MFU) -> float:
    return mfu * PEAK_BF16_TFLOPS * 1e12 / _flops_per_byte(cfg)


def step_seconds(cfg, batch: int, seq: int, mfu: float = MFU) -> float:
    return batch * seq / bps_at_mfu(cfg, mfu)


#: The lever rows. A `None` saving means "this lever does not change VRAM on
#: this target" and is printed as such; every other number is a difference of
#: `peak_bytes`/`activation_bytes`, never a hand estimate.
def lever_rows(
    cfg,
    ledger: ParamLedger,
    batch: int,
    seq: int,
    ns_chunk: int,
    *,
    pcie_gbs: float = 6.5,
    ddr_gbs: float = 68.0,
) -> list[dict]:
    base = peak_bytes(cfg, ledger, batch, seq, Levers(), ns_chunk)
    rows = []
    full = peak_bytes(cfg, ledger, batch, seq, Levers(recompute="full"), ns_chunk)
    sel = peak_bytes(cfg, ledger, batch, seq, Levers(recompute="selective"), ns_chunk)
    rows.append(
        {
            "lever": "activation checkpointing (iter_checkpoint/full)",
            "gib": (base.peak - full.peak) / GIB,
            "cost": "+1 forward / iteration (step = 3 fwd units -> 4: 33.3% slower)",
            "numerics": "bit-identical on CPU; exact recompute, GPU gate 1e-6 relative",
        }
    )
    rows.append(
        {
            "lever": "selective recompute (write half)",
            "gib": (base.peak - sel.peak) / GIB,
            "cost": (
                f"recompute {_write_mac_fraction(cfg):.0%} of the active MACs "
                "(the MoE pass is most of them)"
            ),
            "numerics": "same ops, same values; 1e-6 relative gate",
        }
    )
    packed = peak_bytes(cfg, ledger, batch, seq, Levers(act_pack="mxfp4"), ns_chunk)
    rows.append(
        {
            "lever": "mxfp4 saved tensors (act_ckpt int4)",
            "gib": (base.peak - packed.peak) / GIB,
            "cost": "codec compute; CPU 2.5-3x the step, GPU unmeasured (act_ckpt.py)",
            "numerics": "loss bit-identical; grads lossy, worst 2.4e-1 relative on the tiny graph",
        }
    )
    bank = peak_bytes(cfg, ledger, batch, seq, Levers(bank_dtype="ternary"), ns_chunk)
    if cfg.ffn_form == "bank":
        rows.append(
            {
                "lever": "expert bank ternary (2-bit, not wired to the bank)",
                "gib": (base.peak - bank.peak) / GIB,
                "cost": "same MACs and launches; 2-bit unpack + one fp32 scale per row",
                "numerics": "weights rounded to {-1,0,+1}*scale: lossy, not bit-identical",
            }
        )
    mu = peak_bytes(cfg, ledger, batch, seq, Levers(muon_dtype="bf16"), ns_chunk)
    rows.append(
        {
            "lever": "optimizer state bf16 (muon_state_dtype)",
            "gib": (base.peak - mu.peak) / GIB,
            "cost": "free: same launches, half the store bandwidth",
            "numerics": "accumulation stays fp32, store rounds; 1e-2 relative gate",
        }
    )
    zero = peak_bytes(cfg, ledger, batch, seq, Levers(zero_cards=2), ns_chunk)
    state_gib = (ledger.muon * 4 + ledger.adamw * 8) / GIB
    rows.append(
        {
            "lever": "ZeRO-1 shard-if-two-cards (state only)",
            "gib": (base.peak - zero.peak) / GIB,
            "cost": (
                f"{state_gib / 2:.2f} GiB/shard; one all-gather of "
                f"{ledger.total * 4 / GIB:.2f} GiB per step at {pcie_gbs} GB/s = "
                f"{ledger.total * 4 / (pcie_gbs * 1e9):.3f} s"
            ),
            "numerics": "bit-identical (fp32 copies)",
        }
    )
    off = peak_bytes(cfg, ledger, batch, seq, Levers(offload_state=True), ns_chunk)
    st = (ledger.muon * 4 + ledger.adamw * 8) / GIB
    rows.append(
        {
            "lever": "host offload of optimizer state",
            "gib": (base.peak - off.peak) / GIB,
            "cost": (
                f"2 x {st:.2f} GiB / {pcie_gbs} GB/s = {2 * st * GIB / (pcie_gbs * 1e9):.3f} s "
                f"per step (PCIe is the wall; the 68 GB/s DDR4 side is "
                f"{2 * st * GIB / (ddr_gbs * 1e9):.3f} s)"
            ),
            "numerics": "bit-identical (same state values, moved)",
        }
    )
    rows.append(
        {
            "lever": "gradient accumulation",
            "gib": 0.0,
            "cost": "free VRAM: peak is one micro-batch; same total FLOPs",
            "numerics": "sum order changes: drift bounded by fp32 adds, not bit-identical",
        }
    )
    seg = 4096 if seq >= 4096 else seq // 2
    scr = _score_bytes(cfg, batch, seq)
    seg_saving = scr * (1.0 - seg / seq) / GIB if seq else 0.0
    rows.append(
        {
            "lever": f"sequence chunking (segment_len={seg})",
            "gib": seg_saving,
            "cost": f"{math.ceil(seq / seg)} segments: more launches; +2% per extra (assumption, unmeasured)",
            "numerics": f"exact; only the attention KDA state boundary changes (b*s*s scores: {scr / GIB:.3f} GiB)",
        }
    )
    return rows


def _write_mac_fraction(cfg) -> float:
    from dormouse.width import active_macs_per_token, attn_macs

    attn = attn_macs(cfg.d_model, cfg.n_heads, cfg.attn_latent_rank, cfg.n_kv_heads)
    total = active_macs_per_token(
        cfg.d_model,
        iters=1,
        n_heads=cfg.n_heads,
        attn_latent_rank=cfg.attn_latent_rank,
        n_kv_heads=cfg.n_kv_heads,
        out_proj_rank=cfg.out_proj_rank,
        f=cfg.d_ffn,
        ffn_form=cfg.ffn_form,
        ffn_rank=cfg.rank,
        n_experts=cfg.n_experts,
        topk=cfg.moe_topk,
        shared_rank=cfg.shared_rank,
    )
    return max(0.0, min(1.0, (total - attn) / total))


def _score_bytes(cfg, batch: int, seq: int) -> int:
    """The full-attention score matrix, the only O(s^2) term. 0 for KDA/CSA."""
    if getattr(cfg, "attn_kind", "kda") != "full" and not getattr(
        cfg, "use_msa", False
    ):
        return 0
    return _bt(cfg, batch, seq) * cfg.n_heads * seq * _act_bytes(cfg)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _fmt_gib(n: float) -> str:
    return f"{n / GIB:8.3f}"


def print_param_table(target: Target, ledger: ParamLedger) -> None:
    print(f"\n== {target.name} ==  {ledger.total:,} params  ({target.note})")
    print(f"   {'group':28s} {'params':>15s} {'GiB fp32':>10s} {'of total':>9s}")
    for name, n in sorted(ledger.groups.items(), key=lambda kv: -kv[1]):
        print(
            f"   {name:28s} {n:15,d} {n * 4 / GIB:10.3f} {100 * n / ledger.total:8.1f}%"
        )
    print(f"   {'Muon group':28s} {ledger.muon:15,d} {ledger.muon * 4 / GIB:10.3f}")
    print(f"   {'AdamW group':28s} {ledger.adamw:15,d} {ledger.adamw * 4 / GIB:10.3f}")


def print_component_table(
    target: Target, ledger: ParamLedger, batch: int, seq: int
) -> None:
    p = peak_bytes(target.cfg, ledger, batch, seq, Levers(), target.ns_chunk)
    total = p.fwd_peak
    rows = [
        ("parameters (fp32)", p.params),
        ("gradients (fp32)", p.grads),
        ("Muon momentum (fp32)", p.muon_state),
        ("AdamW exp_avg+exp_avg_sq", p.adamw_state),
        ("expert bank cache (bf16)", p.bank_cache),
        (f"activations @ b{batch}/s{seq}", p.activations),
        ("CUDA context/allocator", p.context),
    ]
    print(
        f"\n   component ledger @ b{batch} s{seq} (fwd/bwd peak; "
        f"max_iter={target.cfg.max_iter})"
    )
    print(f"   {'component':34s} {'GiB':>8s} {'% of peak':>10s} {'bytes':>16s}")
    for name, n in rows:
        print(f"   {name:34s} {_fmt_gib(n)} {100 * n / total:9.1f}% {n:16,d}")
    over = (
        ""
        if total <= CARD_GIB * GIB
        else f"  OVER by {(total - CARD_GIB * GIB) / GIB:.3f} GiB"
    )
    print(
        f"   {'TOTAL fwd/bwd peak':34s} {_fmt_gib(total)} {100.0:9.1f}% "
        f"{total:16,d}{over}"
    )
    print(
        f"   optimizer-stage peak: {p.ns_workspace / GIB:.3f} GiB NS workspace "
        f"(6 x {min(ledger.muon, target.ns_chunk):,} elements x 4 B) + params/grads "
        f"+ state + context = {p.opt_peak / GIB:.3f} GiB"
    )
    print(f"   TRAIN PEAK = max(fwd/bwd, optimizer) = {p.peak_gib:.3f} GiB")


def print_max_batch(target: Target, ledger: ParamLedger) -> None:
    for seq in (1024, 8192):
        b = max_batch(target.cfg, ledger, seq, Levers(), target.ns_chunk)
        if b == 0:
            one = peak_bytes(target.cfg, ledger, 1, seq, Levers(), target.ns_chunk)
            print(
                f"   max batch s={seq:<5d}: 0  (b1 alone is {one.peak_gib:.3f} GiB "
                f"> {CARD_GIB:.1f} GiB by {(one.peak_gib - CARD_GIB):.3f} GiB)"
            )
        else:
            pk = peak_bytes(target.cfg, ledger, b, seq, Levers(), target.ns_chunk)
            print(
                f"   max batch s={seq:<5d}: {b:4d}  ({pk.peak_gib:.3f} GiB, ceiling {CARD_GIB})"
            )


def print_levers(
    target: Target, ledger: ParamLedger, batch: int = REFERENCE_B, seq: int = 1024
):
    print(
        f"\n   levers @ b{batch} s{seq} (saving is a difference of the ledger, not an estimate)"
    )
    print(f"   {'lever':46s} {'GiB saved':>10s}  cost / numerics")
    for row in lever_rows(target.cfg, ledger, batch, seq, target.ns_chunk):
        print(
            f"   {row['lever']:46s} {row['gib']:10.3f}  {row['cost']}\n"
            f"   {'':46s} {'':10s}  {row['numerics']}"
        )


def decision_rows(
    targets: dict[str, Target], ledgers: dict[str, ParamLedger]
) -> list[dict]:
    arms = {
        "none": Levers(),
        "muon16": Levers(muon_dtype="bf16"),
        "ckpt": Levers(recompute="full"),
        "ckpt+muon16": Levers(recompute="full", muon_dtype="bf16"),
        "ckpt+muon16+mxfp4": Levers(
            recompute="full", muon_dtype="bf16", act_pack="mxfp4"
        ),
        "ckpt+muon16+offload": Levers(
            recompute="full", muon_dtype="bf16", offload_state=True
        ),
        "ckpt+muon16+zero2": Levers(recompute="full", muon_dtype="bf16", zero_cards=2),
        "ckpt+muon16+bank-ternary": Levers(
            recompute="full", muon_dtype="bf16", bank_dtype="ternary"
        ),
    }
    out = []
    for key, target in targets.items():
        ledger = ledgers[key]
        flops = _flops_per_byte(target.cfg)
        for arm, lv in arms.items():
            if "bank" in arm and target.cfg.ffn_form != "bank":
                continue
            b = max_batch(target.cfg, ledger, 1024, lv, target.ns_chunk)
            if b >= 1:
                pk = peak_bytes(
                    target.cfg, ledger, b, 1024, lv, target.ns_chunk
                ).peak_gib
            else:
                pk = peak_bytes(
                    target.cfg, ledger, 1, 1024, lv, target.ns_chunk
                ).peak_gib
            bps = bps_at_mfu(target.cfg)
            if b == 0:
                verdict = f"IMPOSSIBLE: b1 = {pk:.2f} GiB, over by {pk - CARD_GIB:.2f}"
            elif b >= 8:
                verdict = "ACHIEVABLE at b8"
            else:
                verdict = f"ACHIEVABLE only at b{b}"
            out.append(
                {
                    "target": target.name,
                    "levers": lv.label(),
                    "max_batch_s1024": b,
                    "peak_gib": round(pk, 3),
                    "bps": round(bps, 1),
                    "mflops_per_byte": round(flops / 1e6, 1),
                    "verdict": verdict,
                }
            )
    return out


def print_decision(rows: list[dict]) -> None:
    print("\n== decision table (s=1024, ceiling 15 GiB, MFU 18% of 45.7 TFLOP/s) ==")
    print(
        f"   {'target':22s} {'levers':30s} {'maxb':>4s} {'GiB':>7s} {'B/s':>9s}  verdict"
    )
    for r in rows:
        print(
            f"   {r['target']:22s} {r['levers']:30s} {r['max_batch_s1024']:4d} "
            f"{r['peak_gib']:7.3f} {r['bps']:9,.0f}  {r['verdict']}"
        )


def print_sparse_section(
    targets: dict[str, Target], ledgers: dict[str, ParamLedger]
) -> None:
    key = "spa1b"
    target, ledger = targets[key], ledgers[key]
    cfg = target.cfg
    routed = ledger.groups.get("expert bank (routed)", 0)
    expanded = bank_cache_elements(cfg)
    bf16 = expanded * _act_bytes(cfg)
    tern = bank_ternary_bytes(cfg)
    act = active_params(cfg)
    routed_active = routed // cfg.n_experts * cfg.moe_topk if cfg.n_experts else 0
    ternary_active = routed_active * 0.25 + (act - routed_active) * 2
    flops = _flops_per_byte(cfg)
    macs = flops / 6.0
    t = step_seconds(cfg, 8, 1024)
    bnk8 = 8 * 1024
    print("\n== sparse-1B: bank, active bytes, arithmetic intensity ==")
    print(f"   stored {ledger.total:,} params; active {act:,} params/token")
    print(
        f"   routed expert params {routed:,} ({routed * 2 / GIB:.3f} GiB raw bf16); "
        f"expanded bank [E,d,r]+[E,r,f] {expanded:,}"
    )
    print(f"   bank bf16        : {bf16 / GIB:.3f} GiB")
    print(
        f"   bank ternary 2bit: {tern / GIB:.3f} GiB (4 weights/byte + {cfg.n_experts * cfg.rank * 4 / 2**20:.2f} MiB scales)"
    )
    print(
        f"   per-token active bytes: bf16 {act * 2 / 2**20:.1f} MiB; ternary bank "
        f"{ternary_active / 2**20:.1f} MiB (only the {routed_active:,} routed weights shrink)"
    )
    print(
        f"   train FLOPs/byte: {flops / 1e6:.1f} MFLOP/B (6 x {macs / 1e6:.2f} M active MAC/B; "
        f"6 x {act / 1e6:.2f} M active params)"
    )
    print(
        f"   at 62 kB/s -> {flops * 62_000 / 1e12:.2f} TFLOP/s = {flops * 62_000 / (PEAK_BF16_TFLOPS * 1e12):.1%} MFU"
    )
    print(
        f"   at 18% MFU ({bps_at_mfu(cfg, 0.18) / 1000:.1f} kB/s) a b8/s1024 step is {t:.3f} s"
    )
    print(
        f"   weights streamed once per step: {bf16 / GIB:.3f} GiB / {t:.3f} s = "
        f"{bf16 / t / 1e9:.1f} GB/s (bf16) vs ~1900 GB/s HBM -> fits"
    )
    print(
        f"   batch-1 per-token streaming: {act * 2 / 2**20:.1f} MiB x 62 kB/s = "
        f"{act * 2 * 62_000 / 1e12:.1f} TB/s -> does NOT fit; ternary bank "
        f"{ternary_active * 62_000 / 1e12:.2f} TB/s; the machine balance is "
        f"{PEAK_BF16_TFLOPS * 1e12 / 1900e9:.1f} FLOP/B"
    )
    print(
        f"   at b8/s1024 each expert sees ~{bnk8 * cfg.moe_topk / cfg.n_experts:.0f} tokens: "
        f"AI = {3 * bnk8 * cfg.moe_topk / cfg.n_experts:.0f} FLOP/B, compute-bound"
    )


# ---------------------------------------------------------------------------
# the GPU driver (refuses while the card is held)
# ---------------------------------------------------------------------------


def measure(
    target: Target, ledger: ParamLedger, batch: int, seq: int, lv: Levers
) -> int:
    """One real step on the card: peak GiB, seconds, B/s, implied MFU.

    The preflight is FIRST and touches no torch CUDA call: `_lane.preflight`
    reads `nvidia-smi` only, because `torch.cuda.is_available()` would create a
    context in the parent (`_lane.py`'s own history). A busy card prints the
    SKIP banner and exits 0.
    """
    sys.path.append(str(REPO / "port" / "scripts"))
    from _lane import preflight, skip

    why = preflight("cuda")
    if why:
        return skip(why)

    import time

    import torch
    from dormouse.model import DormouseModel, param_count
    from dormouse.train.optim import MixOptimizer

    cfg = dataclasses.replace(target.cfg, max_seq_len=max(seq, target.cfg.max_seq_len))
    torch.manual_seed(1)
    model = DormouseModel(cfg).to("cuda").train()
    assert param_count(model) == ledger.total, (
        "the measured model is not the ledger's model"
    )
    if lv.recompute != "none":
        model.configure_actmem(
            iter_checkpoint=lv.recompute == "full",
            activation_offload=0.0,
            activation_recompute=lv.recompute,
        )
    opt = MixOptimizer(
        model,
        lr=1e-3,
        wd=0.01,
        muon_state_dtype=lv.muon_dtype,
        muon_chunk_elems=target.ns_chunk,
    )
    ids = torch.randint(0, 256, (batch, seq), device="cuda")
    # One warmup step allocates the optimizer state and the first-call kernels;
    # the measured step is then the steady-state peak, not a lazy-init spike.
    _, loss, _ = model(ids, targets=ids)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    _, loss, _ = model(ids, targets=ids)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / GIB
    bps = batch * seq / dt
    mfu = _flops_per_byte(cfg) * bps / (PEAK_BF16_TFLOPS * 1e12)
    print(
        json.dumps(
            {
                "target": target.name,
                "batch": batch,
                "seq": seq,
                "levers": lv.label(),
                "peak_gib": round(peak, 3),
                "step_s": round(dt, 4),
                "bytes_per_s": round(bps, 1),
                "mfu": round(mfu, 4),
            }
        )
    )
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write every row here")
    ap.add_argument(
        "--measure", action="store_true", help="run one real step (needs a free card)"
    )
    ap.add_argument(
        "--target", default="spa1b", choices=["flagship", "m400", "dense1b", "spa1b"]
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument(
        "--recompute", default="none", choices=["none", "full", "selective"]
    )
    ap.add_argument("--muon-dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--no-levers", action="store_true")
    args = ap.parse_args(argv)

    targets = build_targets()
    if args.measure:
        target = targets[args.target]
        ledger = param_ledger(target.cfg)
        lv = Levers(recompute=args.recompute, muon_dtype=args.muon_dtype)
        return measure(target, ledger, args.batch, args.seq, lv)

    ledgers = {k: param_ledger(t.cfg) for k, t in targets.items()}
    ref = reference_target()
    ref_ledger = param_ledger(ref.cfg)

    for key, target in targets.items():
        print_param_table(target, ledgers[key])
        print_component_table(target, ledgers[key], REFERENCE_B, REFERENCE_S)
        print_max_batch(target, ledgers[key])
        if not args.no_levers:
            print_levers(target, ledgers[key])
    print_sparse_section(targets, ledgers)
    rows = decision_rows(targets, ledgers)
    print_decision(rows)

    rp = peak_bytes(
        ref.cfg, ref_ledger, REFERENCE_B, REFERENCE_S, Levers(), ref.ns_chunk
    )
    print(
        f"\n== gate: reference (medium, max_iter=3, b8/s1024) = {rp.peak_gib:.3f} GiB "
        f"vs measured {MEASURED_PEAK_GIB} GiB (act_mem.py:4) -> "
        f"{100 * (rp.peak_gib - MEASURED_PEAK_GIB) / MEASURED_PEAK_GIB:+.1f}%"
    )
    assert abs(rp.peak_gib - MEASURED_PEAK_GIB) / MEASURED_PEAK_GIB <= 0.10, (
        "the activation model drifted past 10% of the one measured peak"
    )

    if args.json:
        payload = {
            "targets": {
                k: {
                    "params": ledgers[k].total,
                    "groups": ledgers[k].groups,
                    "muon": ledgers[k].muon,
                    "adamw": ledgers[k].adamw,
                    "max_batch_s1024": max_batch(
                        t.cfg, ledgers[k], 1024, Levers(), t.ns_chunk
                    ),
                    "max_batch_s8192": max_batch(
                        t.cfg, ledgers[k], 8192, Levers(), t.ns_chunk
                    ),
                    "active_params": active_params(t.cfg),
                    "flops_per_byte": _flops_per_byte(t.cfg),
                }
                for k, t in targets.items()
            },
            "reference_peak_gib": rp.peak_gib,
            "measured_peak_gib": MEASURED_PEAK_GIB,
            "decision": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
