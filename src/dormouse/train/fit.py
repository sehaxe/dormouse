"""The scrap-metal path: one `--fit <vram_gb>` knob chooses the memory set.

The port already owns four memory mechanisms, each with its own gate and its own
counter. What it did not have is a way to say "this card has 4 GB; make the run
fit" without a reader re-deriving which combination is needed. This module is
that resolver — it estimates the peak, engages mechanisms in order of rising
cost to the run, and PRINTS the set it chose (a mechanism that ran and was not
announced is the COUNTED failure the repo's rules forbid).

The mechanisms, cheapest-first:

1. **`moe_offload`** (model) — the routed expert pass's saved activations go to
   host RAM; gradients are EXACT (`tests/test_offload_experts.py`). Engaged only
   when `moe_topk > 0` (a dense blend has no routed bank to offload and the
   switch is COUNTED inert).
2. **`engram_ram`** (train) — the n-gram tables and their momentum leave VRAM
   for host RAM; the table parameters stop existing. Not available on a fleet
   (`distributed.refuse_incompatible`: per-rank host state does not reduce).
3. **`muon_state_dtype="bf16"`** (train) — Muon's momentum halves; fp32
   accumulation, approximate by the rung's own design.
4. **`optim_8bit`** (train) — bitsandbytes' 8-bit AdamW state for everything
   AdamW routes (the b1 enabler; the VRAM lever, not a speed one).

`act_ckpt` (`model/act_ckpt.py`, quantized saved tensors) is a real mechanism
with a real bench, but it has **no config seam yet** — it is not in this plan,
and the doc says so rather than pretending the list is complete.

The estimate is a MODEL, not a measurement: the weights are the port's fp32
masters, the gradients and states follow the routing `MixOptimizer` actually
computes (on a `meta` build, so the split is the real policy and not a guess),
and the activation constant is derived from the b1 spike's recorded
decomposition (`docs/research/2026-10-15-torch-spike.md`: ≈2.8 GiB of
activations at 16 384 tokens and d=2048, i.e. 64 B per token·dim). The doc
quotes the b1 cross-check and says which number is measured and which is
assumed. `plan_fit` refuses loudly when no set fits.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from ..config import ModelConfig, RunConfig, TrainConfig
from ..instrument import loud

#: The port's storage: fp32 masters, fp32 gradients, AdamW m+v (8 B), Muon
#: momentum (4 B). `optim_8bit` replaces the AdamW pair with bitsandbytes'
#: 8-bit block state (~2 B/param including its quantizer constant);
#: `muon_state_dtype="bf16"` halves Muon's momentum (2 B).
WEIGHT_BYTES = 4
GRAD_BYTES = 4
ADAMW_STATE_BYTES = 8
ADAMW_8BIT_STATE_BYTES = 2
MUON_STATE_BYTES = 4
MUON_STATE_BF16_BYTES = 2

#: Activation + autograd + workspace bytes per (token × d_model), derived from
#: the b1 spike's measured decomposition: ≈2.8 GiB at 16 384 tokens × d 2048
#: (`docs/research/2026-10-15-torch-spike.md` §"memory budget"). A MODEL — the
#: KDA scratch makes it shape-dependent — stated so the plan errs visibly.
ACT_BYTES_PER_TOKEN_DIM = 64

#: The share of the routed expert pass's activations that `moe_offload` moves to
#: host RAM. An assumption: the mechanism offloads the routed bank's saved
#: tensors, not the shared expert's or the attention's. The doc names it.
MOE_OFFLOAD_SHARE = 0.35

#: The plan must fit in this share of the stated budget. The knob names the
#: CARD's size; the caching allocator, cuBLAS workspaces, the eval pass and
#: fragmentation live in the rest. 10% is a starting headroom, not a measured
#: optimum — the doc says to raise it if a fit plan still OOMs.
FIT_HEADROOM = 0.9


@dataclass(frozen=True)
class FitPlan:
    """The chosen set and the arithmetic that chose it — the line is the record."""

    vram_gb: float
    peak_gb: float
    engaged: tuple[str, ...]
    model_over: dict
    train_over: dict
    components: dict

    def line(self) -> str:
        parts = " ".join(f"{k}={v / 2**30:.2f}GiB" for k, v in self.components.items())
        eng = ",".join(self.engaged) if self.engaged else "none"
        return f"fit={self.vram_gb:.1f}GB: est_peak={self.peak_gb:.2f}GiB engaged={eng} ({parts})"

    def overrides(self) -> tuple[dict, dict]:
        """`(model_over, train_over)` — the shape `load_run` and `replace` take."""
        return self.model_over, self.train_over


def _table_params(cfg: ModelConfig) -> int:
    """The Engram table parameters — the ones `engram_ram` removes from VRAM."""
    if not cfg.use_engram:
        return 0
    return cfg.engram_rows * len(cfg.engram_orders) * cfg.engram_dim


def _routing(model_cfg: ModelConfig) -> tuple[int, int, int]:
    """`(total, muon, adamw)` parameter counts from the REAL optimizer routing.

    A `meta` build costs no memory and no forward: `MixOptimizer` reads the
    module tree's names and shapes exactly as it will in the run, so the split
    is the policy rather than a reproduction of it. The optimizer itself is
    discarded — constructing it allocates no state (that happens at the first
    step).
    """
    from ..model import DormouseModel
    from .optim import MixOptimizer

    model = DormouseModel(model_cfg, device="meta")
    opt = MixOptimizer(model, lr=1e-3, wd=0.0)
    muon = sum(p.numel() for p in opt.muon_params)
    total = sum(p.numel() for p in model.parameters())
    return total, muon, total - muon


def estimate(
    model_cfg: ModelConfig,
    *,
    batch: int,
    seq_len: int,
    engram_ram: bool = False,
    optim_8bit: bool = False,
    muon_state_dtype: str = "fp32",
    moe_offload: bool = False,
    routing: tuple[int, int, int] | None = None,
) -> dict[str, int]:
    """The peak estimate with a given set engaged, split by component."""
    total, muon, adamw = routing if routing is not None else _routing(model_cfg)
    table = _table_params(model_cfg) if engram_ram else 0
    # The host tables leave VRAM entirely: no masters, no grads, no state. The
    # in-VRAM residue is one row per order, which the estimate ignores (bytes).
    p_total, p_muon, p_adamw = total - table, muon, adamw - table
    assert p_total >= 0 and p_muon >= 0 and p_adamw >= 0, "the table count exceeds a group"
    weights = p_total * WEIGHT_BYTES
    grads = p_total * GRAD_BYTES
    muon_s = p_muon * (MUON_STATE_BF16_BYTES if muon_state_dtype == "bf16" else MUON_STATE_BYTES)
    adamw_s = p_adamw * (ADAMW_8BIT_STATE_BYTES if optim_8bit else ADAMW_STATE_BYTES)
    act = batch * seq_len * model_cfg.d_model * ACT_BYTES_PER_TOKEN_DIM
    if moe_offload and model_cfg.moe_topk > 0:
        act = int(act * (1.0 - MOE_OFFLOAD_SHARE))
    return {
        "weights": weights,
        "grads": grads,
        "muon_state": muon_s,
        "adamw_state": adamw_s,
        "activations": act,
    }


def plan_fit(
    run: RunConfig,
    vram_gb: float,
    *,
    batch: int | None = None,
    seq_len: int | None = None,
    allow_engram_ram: bool = True,
) -> FitPlan:
    """Pick the cheapest set that fits `vram_gb`, or refuse naming the arithmetic.

    `batch`/`seq_len` default to the run's; a fleet passes its capacity
    profile's largest entry (the plan must fit the rank with the biggest
    micro-batch, not the average).
    """
    if vram_gb <= 0:
        loud(
            f"fit_vram_gb={vram_gb}: 0 means OFF (the layer is never called). "
            f"Escape: a positive GB number, or drop --fit."
        )
    cfg = run.model
    b = run.train.batch if batch is None else batch
    t = run.train.seq_len if seq_len is None else seq_len
    budget = int(vram_gb * 1e9 * FIT_HEADROOM)  # decimal GB, the unit the knob names
    # The mechanisms in rising order of cost to the run (see the module doc).
    if run.train.dist:
        allow_engram_ram = False
    ladder: list[tuple[str, dict, dict]] = []
    if cfg.moe_topk > 0:
        ladder.append(("moe_offload", {"moe_offload": True}, {}))
    if allow_engram_ram and cfg.use_engram:
        ladder.append(("engram_ram", {}, {"engram_ram": True}))
    routing = _routing(cfg)
    if routing[1] > 0:
        ladder.append(("muon_state_dtype=bf16", {}, {"muon_state_dtype": "bf16"}))
    ladder.append(("optim_8bit", {}, {"optim_8bit": True}))

    state = {
        "engram_ram": False,
        "optim_8bit": False,
        "muon_state_dtype": "fp32",
        "moe_offload": False,
    }
    engaged: list[str] = []
    model_over: dict = {}
    train_over: dict = {}
    comp = estimate(cfg, batch=b, seq_len=t, routing=routing, **state)
    for name, mo, to in ladder:
        if sum(comp.values()) <= budget:
            break
        state.update(mo)
        state.update(to)
        model_over.update(mo)
        train_over.update(to)
        engaged.append(name)
        comp = estimate(cfg, batch=b, seq_len=t, routing=routing, **state)
    peak = sum(comp.values())
    if peak > budget:
        hints = (
            ""
            if allow_engram_ram
            else (
                " The host-table mechanism is unavailable on a fleet (per-rank host state "
                "does not reduce — distributed.refuse_incompatible);"
            )
        )
        breakdown = ", ".join(f"{k}={v / 2**30:.2f}" for k, v in comp.items())
        loud(
            f"fit={vram_gb:.1f}GB cannot hold {run.source}: the estimate after every "
            f"mechanism is {peak / 2**30:.2f} GiB ({breakdown}). {hints} Escape: a "
            f"smaller preset, more cards (the model then shards or replicates), or a "
            f"host offload engine this repo does not have yet (optimizer state to RAM "
            f"is the named follow-up)."
        )
    return FitPlan(
        vram_gb=vram_gb,
        peak_gb=peak / 2**30,
        engaged=tuple(engaged),
        model_over=model_over,
        train_over=train_over,
        components=comp,
    )


def apply_fit(run: RunConfig, plan: FitPlan) -> RunConfig:
    """The resolved set onto the run — the snapshot then records the MECHANISMS.

    Not the knob: `--fit` is a resolver like `--set`, so a checkpoint's snapshot
    says `engram_ram=true`, `optim_8bit=true`, … and a resume re-derives the
    same set from the same knob (or refuses the drift). A resume must pass
    `--fit` again — the same rule as every other override.
    """
    model_over, train_over = plan.overrides()
    known_model = {f.name for f in fields(ModelConfig)}
    known_train = {f.name for f in fields(TrainConfig)}
    bad = [k for k in model_over if k not in known_model] + [
        k for k in train_over if k not in known_train
    ]
    assert not bad, f"fit plan names non-config keys {bad}"
    return replace(
        run,
        model=replace(run.model, **model_over) if model_over else run.model,
        train=replace(run.train, **train_over) if train_over else run.train,
    )
