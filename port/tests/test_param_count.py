"""The presets' parameter counts, recomputed INDEPENDENTLY of the model tree.

Two checks, and they are different checks:

1. the hand arithmetic below (written out here, not read from any module) must
   equal the count of the model the config actually BUILDS. The burn side's own
   lesson (`preset_exec.rs`): "a formula that drifts from the module tree is
   exactly the kind of number that reads true and lies";
2. the totals written in `configs/small.toml` and `configs/b1.toml` must be
   those numbers, so the preset's own comment cannot rot.

b1 is 1.07G parameters: it is instantiated on the `meta` device, which builds
every module and every shape with no storage behind them.
"""

from __future__ import annotations

from dormouse.config import load_preset
from dormouse.model import DormouseModel, param_count, table_slots

#: The totals the two presets' comments carry.
SMALL_PORT_TOTAL = 9_476_993
B1_PORT_TOTAL = 1_066_358_209  # max_iter 4; MSA off (tie verdict 2026-10-17)


def arithmetic(cfg) -> dict[str, int]:
    """The port's module tree, priced by hand, component by component."""
    d, f, r, v = cfg.d_model, cfg.d_ffn, cfg.rank, cfg.vocab
    e, sr, dim = cfg.n_experts, cfg.shared_rank, cfg.engram_dim
    n_ord = len(cfg.engram_orders)
    slots = table_slots(cfg.engram_rows)
    head_rank = min(r, d, v)
    # TSCT factors: `U [in, r]`, `s [r]`, `V [r, out]` = r*(in + out + 1).
    tsct = lambda i, o, k: k * (i + o + 1)  # noqa: E731
    # TSCT2 §14's expert: two latents (gate, up) at rank r, the GLU at the
    # latent, one write `V [r, f]`. The down projection is SHARED (`down`).
    routed_expert = 2 * (d * r + r) + r * f
    return {
        "embed": v * d,
        "lm_head": tsct(d, v, head_rank),
        "norms": 2 * d,  # loop pre-norm + the final norm
        "controller": 2 * d * 3,  # three sigmoid gates, no bias
        "kda": 4 * d * d + d * cfg.n_heads,  # q,k,v,g + per-head beta
        "experts": e * routed_expert,
        "shared_expert": 2 * (d * sr + sr) + sr * f,
        "down": f * d,  # the single shared write back to the residual width
        "router": d * e,
        "engram_tables": n_ord * slots * dim,
        "engram_projs": d * dim + dim * dim + dim * d,  # query, key, value
        "mem_dense": d * d + d,
        "out_proj": tsct(d, d, r),
        "iter_embed": cfg.max_iter * d,
        "residual_scale": 1,
        "msa": _msa_params(cfg),
    }


def _msa_params(cfg) -> int:
    """The MSA arm's parameters when `use_msa` (else 0 — the arm's absence is
    byte-for-byte the KDA-only model, `tests/test_msa.py` case d).

    Three `d x d` projections (q, k, v — the g/beta of gated-delta are
    KDA-specific and not carried), the indexer's MQA pair
    (`q_idx [d, q_heads*head_dim]`, `k_idx [d, head_dim]`, q_heads =
    `sparse.sparse.MSA_Q_HEADS`) and the per-slot adaptive gate `[max_iter]`.
    """
    if not getattr(cfg, "use_msa", False):
        return 0
    from dormouse.model.sparse.sparse import MSA_Q_HEADS

    return (
        3 * cfg.d_model * cfg.d_model
        + cfg.d_model * MSA_Q_HEADS * cfg.head_dim
        + cfg.d_model * cfg.head_dim
        + cfg.max_iter
    )


def built(model) -> dict[str, int]:
    """The same components, counted off the instantiated model."""
    out: dict[str, int] = {}
    for name, p in model.named_parameters():
        if name.startswith("loop.moe.experts"):
            key = "experts"
        elif name.startswith("loop.moe.shared"):
            key = "shared_expert"
        elif name.startswith("loop.moe.down"):
            key = "down"
        elif name.startswith("loop.moe.router"):
            key = "router"
        elif name.startswith("loop.engram.tables"):
            key = "engram_tables"
        elif name.startswith("loop.engram"):
            key = "engram_projs"
        elif name.startswith("loop.attn"):
            key = "kda"
        elif name.startswith("loop.msa"):
            key = "msa"  # covers loop.msa.* and the loop.msa_gate parameter
        elif name.startswith("loop.mem_dense"):
            key = "mem_dense"
        elif name.startswith("loop.out_proj"):
            key = "out_proj"
        elif name.startswith("loop.iter_embed"):
            key = "iter_embed"
        elif name.startswith("loop.residual_scale"):
            key = "residual_scale"
        elif name.startswith("loop.controller"):
            key = "controller"
        elif name.startswith("loop.norm") or name.startswith("norm"):
            key = "norms"
        elif name.startswith("embed"):
            key = "embed"
        elif name.startswith("lm_head"):
            key = "lm_head"
        else:  # pragma: no cover - a new parameter must land in a bucket
            raise AssertionError(f"unbucketed parameter {name}")
        out[key] = out.get(key, 0) + p.numel()
    if "msa" not in out:
        # The arm is off: `built` must carry the key so the dict compare holds
        # (arithmetic always emits it, 0 when off).
        out["msa"] = 0
    return out


def test_b1_arithmetic_matches_the_model_and_the_preset():
    """The b1 arithmetic, written out. §2.2's table priced the expert bank as
    `1024 x (2048*64 + 64*5461 + s)` with NO down-projection; this is the
    corrected shape (configs/b1.toml carries the full table):

        per routed expert = 2*(d*r + r) + r*f
                          = 2*(2048*64 + 64) + 64*5461
                          = 262 272 + 349 504 = 611 776
        routed experts    = 1024 * 611 776            = 626 458 624
        shared expert     = 2*(2048*192 + 192) + 192*5461 =   1 835 328
        shared write      = 5461 * 2048               =  11 184 128
    """
    cfg = load_preset("b1")
    want = arithmetic(cfg)
    assert want["experts"] == 1024 * 611_776 == 626_458_624
    model = DormouseModel(cfg, device="meta")
    assert built(model) == want, {k: (want.get(k), built(model).get(k)) for k in want}
    assert sum(want.values()) == B1_PORT_TOTAL == param_count(model)

    # The delta against plan §2.2's table, computed from the table's OWN terms:
    #   1024*(2048*64 + 64*5461 + 64) = 492 175 360  (one latent, no down-proj)
    plan_experts = 1024 * (2048 * 64 + 64 * 5461 + 64)
    assert plan_experts == 492_175_360
    second_latent = 1024 * (2048 * 64 + 64)  # TSCT2 §13.3: two latents, not one
    assert second_latent == 134_283_264
    assert want["experts"] - plan_experts == second_latent
    assert want["experts"] + want["down"] + want["shared_expert"] - plan_experts == (147_302_720)


def test_small_arithmetic_matches_the_model_and_the_preset():
    cfg = load_preset("small")
    model = DormouseModel(cfg, device="meta")
    want = arithmetic(cfg)
    assert built(model) == want
    assert sum(want.values()) == SMALL_PORT_TOTAL == param_count(model)
    # The burn side's `small`, measured on ITS instantiated model (2026-10-01):
    BURN_SMALL_TOTAL = 9_197_390
    assert SMALL_PORT_TOTAL - BURN_SMALL_TOTAL == 279_603, (
        "the delta is structural (port pays a shared write + shared expert it "
        "does not owe the burn tree, and carries no aux heads); if it moves, "
        "either the port's tree changed or burn's did - say which"
    )
    # The Engram share, the number the memory budget is argued from.
    rows = want["engram_tables"]
    assert round(100 * rows / SMALL_PORT_TOTAL, 1) == 33.2  # burn measured 34.2%


MEDIUM_PORT_TOTAL = 141_910_405  # the recipe-validation preset (2026-10-16)


def test_medium_arithmetic_matches_the_model_and_the_preset():
    cfg = load_preset("medium")
    model = DormouseModel(cfg, device="meta")
    want = arithmetic(cfg)
    assert built(model) == want
    assert sum(want.values()) == MEDIUM_PORT_TOTAL == param_count(model)
    assert want["engram_tables"] == 3 * 524288 * 32 == 50_331_648
    # The b1-bet shapes survive the halving: rank-64 experts, topk-2, depth 4.
    assert cfg.rank == 64 and cfg.moe_topk == 2 and cfg.max_iter == 4
