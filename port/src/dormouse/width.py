"""Pure arithmetic for the three cheap-width flags — no torch, no model build.

The bill this module prices is the one the scaling recipe names: the dense
per-token cost of the shared body,

    iters · (attention projections + out_proj + the MoE's shared `down`)

with `4·d²` for KDA's `q,k,v,g`, `d²` for a DENSE readout, and `d·f` for the
shared write back to the residual width. `beta` (`d·n_heads`) and the router
(`d·E`) are below the noise this table is about and are omitted from MAC — the
parameter deltas include `beta` only as "unchanged".

`dormouse model.width` prints the grid; `tests/test_cheapwidth.py` checks its
parameter deltas against `dormouse info`'s instantiated count, so a formula that
drifts from the module tree fails a gate instead of reading true and lying.

The default `f` is `8·d//3`, the `medium` preset's own ratio (`1024 → 2730`);
the table only varies `d`, so it needs one stated rule for `d_ffn`.
"""

from __future__ import annotations

HEAD_DIM = 64  # the port's presets all ship head_dim = 64
#: The `ffn_form` literals (`config.FFN_FORMS`), duplicated here so this module
#: stays torch-free and importable by the pure-arithmetic gate.
FFN_FORMS = ("bank", "dense", "tsct")


def heads(d_model: int, head_dim: int = HEAD_DIM) -> int:
    """The preset's head count at a width (all our presets tile `d = h·head_dim`)."""
    if d_model % head_dim:
        raise ValueError(f"d_model={d_model} is not a multiple of head_dim={head_dim}")
    return d_model // head_dim


def d_ffn(d_model: int) -> int:
    """`medium`'s ratio: `2730/1024 = 8/3` (the table's stated f rule)."""
    return 8 * d_model // 3


def kv_width(d_model: int, n_kv_heads: int, n_heads: int) -> int:
    """The k/v projection's output width: `n_heads·hd` when GQA is off, else `h_kv·hd`."""
    if n_kv_heads and n_heads % n_kv_heads:
        raise ValueError(f"n_kv_heads={n_kv_heads} does not divide n_heads={n_heads}")
    if d_model % n_heads:
        raise ValueError(f"d_model={d_model} does not tile {n_heads} heads")
    return (n_kv_heads or n_heads) * (d_model // n_heads)


def attn_macs(d_model: int, n_heads: int, attn_latent_rank: int, n_kv_heads: int) -> int:
    """MAC/token for KDA's four projections (`q,k,v,g`), one loop iteration."""
    kvw = kv_width(d_model, n_kv_heads, n_heads)
    if attn_latent_rank:
        # one d→r, then q:r→d, g:r→d, k/v:r→kvw each
        return d_model * attn_latent_rank + attn_latent_rank * (2 * d_model + 2 * kvw)
    # q,g: d→d; k,v: d→kvw
    return 2 * d_model * d_model + 2 * d_model * kvw


def out_proj_macs(d_model: int, out_proj_rank: int) -> int:
    """MAC/token for the readout. `0` is the DENSE `d×d` baseline this table prices."""
    return 2 * d_model * out_proj_rank if out_proj_rank else d_model * d_model


def _ffn_ranks(rank: int, rank_gate: int, rank_down: int) -> tuple[int, int]:
    """`(r_gate, r_down)`; 0 means `rank`."""
    return (rank if not rank_gate else rank_gate, rank if not rank_down else rank_down)


def ffn_params(
    form: str,
    d_model: int,
    f: int,
    *,
    rank: int = 64,
    rank_gate: int = 0,
    rank_down: int = 0,
    n_experts: int = 1,
    shared_rank: int = 64,
) -> int:
    """The expert slot's parameter count for one `ffn_form`.

    `dense` is `3·d·f + f + d` (gate bias `f`, down bias `d`; the up projection
    is bias-free). `tsct` is `(2·r_g + r_d)·(d + f + 1)` per expert plus the
    router `d·E` when `E > 1`. `bank` is today's tree: `E` dual-latent experts,
    the always-on shared expert, the shared `down [f, d]` and the router.

    `tests/test_ffn_form.py` checks every value against an instantiated
    `DormouseModel` on CPU; `tests/test_cheapwidth.py` keeps the bank path's
    old numbers pinned.
    """
    rg, rd = _ffn_ranks(rank, rank_gate, rank_down)
    if form == "dense":
        return 3 * d_model * f + f + d_model
    if form == "tsct":
        per = (2 * rg + rd) * (d_model + f + 1) + f + d_model
        return n_experts * per + (d_model * n_experts if n_experts > 1 else 0)
    if form == "bank":
        per = 2 * d_model * rank + 2 * rank + rank * f
        shared = 2 * d_model * shared_rank + 2 * shared_rank + shared_rank * f
        return n_experts * per + shared + d_model * f + d_model * n_experts
    raise ValueError(f"ffn_form={form!r} is not one of {FFN_FORMS}")


def ffn_macs_per_token(
    form: str,
    d_model: int,
    f: int,
    *,
    rank: int = 64,
    rank_gate: int = 0,
    rank_down: int = 0,
    n_experts: int = 1,
    topk: int = 0,
    shared_rank: int = 64,
) -> int:
    """One FFN forward's MAC for one token, by form.

    `dense` reads gate + up + down once: `3·d·f + f + d` (the biases are
    counted, they are the same order). `tsct` reads `2·r_g·(d + f) +
    r_d·(f + d)` per assignment plus the biases, and the router `d·E` when
    `E > 1`. `bank` is the dual-latent bill `2·d·rank + rank·f` per routed
    assignment, the always-on shared expert, the shared `down` and the router.

    `topk = 0` is the dense blend: every expert runs. The BANK ledger's own
    convention counts the routed/shared terms once per token (the same
    convention `capacity_math` has always printed); `macs_per_token` keeps that
    for `bank` and prices dense/tsct per loop iteration. The info table labels
    the unit so the two are not subtracted from one another by accident.
    """
    if form == "dense":
        return 3 * d_model * f + f + d_model
    rg, rd = _ffn_ranks(rank, rank_gate, rank_down)
    assignments = n_experts if topk == 0 else topk
    router = d_model * n_experts if n_experts > 1 else 0
    if form == "tsct":
        per = 2 * rg * (d_model + f) + rd * (f + d_model) + (2 * rg + rd) + f + d_model
        return assignments * per + router
    if form == "bank":
        routed = 2 * d_model * rank + rank * f
        shared = 2 * d_model * shared_rank + shared_rank * f
        # The bank always builds its router, even at E=1: this is the historical
        # `capacity_math` ledger, not the non-bank forms' "no router" rule.
        return assignments * routed + shared + d_model * f + d_model * n_experts
    raise ValueError(f"ffn_form={form!r} is not one of {FFN_FORMS}")


def macs_per_token(
    d_model: int,
    *,
    iters: int,
    n_heads: int,
    attn_latent_rank: int = 0,
    n_kv_heads: int = 0,
    out_proj_rank: int = 0,
    f: int | None = None,
    ffn_form: str = "bank",
    ffn_rank: int = 64,
    ffn_rank_gate: int = 0,
    ffn_rank_down: int = 0,
    n_experts: int = 1,
    topk: int = 0,
    shared_rank: int = 64,
) -> int:
    """The dense bill per token: `iters · (attn + out_proj + FFN)`.

    `bank` keeps the historical `iters · d·f` shared-write term (the routed
    bank's terms are added by `capacity_math`, the ledger that owns them).
    `dense`/`tsct` price the WHOLE expert slot here, since there is no separate
    routed/shared term to add.
    """
    f = d_ffn(d_model) if f is None else f
    core = iters * (
        attn_macs(d_model, n_heads, attn_latent_rank, n_kv_heads)
        + out_proj_macs(d_model, out_proj_rank)
    )
    if ffn_form == "bank":
        return core + iters * d_model * f
    return core + iters * ffn_macs_per_token(
        ffn_form,
        d_model,
        f,
        rank=ffn_rank,
        rank_gate=ffn_rank_gate,
        rank_down=ffn_rank_down,
        n_experts=n_experts,
        topk=topk,
        shared_rank=shared_rank,
    )


def active_macs_per_token(
    d_model: int,
    *,
    iters: int,
    n_heads: int,
    attn_latent_rank: int = 0,
    n_kv_heads: int = 0,
    out_proj_rank: int = 0,
    f: int | None = None,
    ffn_form: str = "bank",
    ffn_rank: int = 64,
    ffn_rank_gate: int = 0,
    ffn_rank_down: int = 0,
    n_experts: int = 1,
    topk: int = 0,
    shared_rank: int = 64,
) -> int:
    """The full active bill: body + the expert slot, whatever its form.

    The `bank` arm subtracts the `d·f` already inside `macs_per_token` from
    `ffn_macs_per_token("bank")`, so the result is exactly the historical
    `capacity_math` ledger. Dense/tsct are already complete inside
    `macs_per_token`.
    """
    f = d_ffn(d_model) if f is None else f
    body = macs_per_token(
        d_model,
        iters=iters,
        n_heads=n_heads,
        attn_latent_rank=attn_latent_rank,
        n_kv_heads=n_kv_heads,
        out_proj_rank=out_proj_rank,
        f=f,
        ffn_form=ffn_form,
        ffn_rank=ffn_rank,
        ffn_rank_gate=ffn_rank_gate,
        ffn_rank_down=ffn_rank_down,
        n_experts=n_experts,
        topk=topk,
        shared_rank=shared_rank,
    )
    if ffn_form == "bank":
        return body + ffn_macs_per_token(
            "bank",
            d_model,
            f,
            rank=ffn_rank,
            shared_rank=shared_rank,
            n_experts=n_experts,
            topk=topk,
        ) - d_model * f
    return body


def attn_param_delta(
    d_model: int, n_heads: int, attn_latent_rank: int, n_kv_heads: int
) -> int:
    """Parameters added by the latent + GQA versus the four `d×d` projections.

    `beta_proj [n_heads, d]` is unchanged in every arm, so it cancels.
    """
    kvw = kv_width(d_model, n_kv_heads, n_heads)
    if attn_latent_rank:
        new = d_model * attn_latent_rank + attn_latent_rank * (2 * d_model + 2 * kvw)
    else:
        new = 2 * d_model * d_model + 2 * d_model * kvw
    return new - 4 * d_model * d_model


def out_proj_param_delta(d_model: int, out_proj_rank: int, *, baseline_rank: int = 0) -> int:
    """Parameters added by the low-rank readout.

    `baseline_rank = 0` prices the DENSE readout (`d²`); a positive
    `baseline_rank` prices the TSCT rung (`rank·(2d + 1)`, its `U`,`s`,`V`).
    """
    if baseline_rank:
        return (out_proj_rank or baseline_rank) * (2 * d_model + 1) - baseline_rank * (
            2 * d_model + 1
        )
    new = 2 * d_model * out_proj_rank if out_proj_rank else d_model * d_model
    return new - d_model * d_model


def param_delta(
    d_model: int,
    *,
    n_heads: int,
    attn_latent_rank: int = 0,
    n_kv_heads: int = 0,
    out_proj_rank: int = 0,
    out_proj_baseline_rank: int = 0,
) -> int:
    """The two new-parameter deltas together (the `down` write is untouched)."""
    return attn_param_delta(d_model, n_heads, attn_latent_rank, n_kv_heads) + (
        out_proj_param_delta(d_model, out_proj_rank, baseline_rank=out_proj_baseline_rank)
    )


def main() -> None:
    """The grid: `d ∈ {1024..4096}`, iters=3, the three flags' crossings."""
    widths = (1024, 1536, 2048, 3072, 4096)
    print(
        "dense MAC/token, iters=3, f=8d/3 (medium's ratio), head_dim=64; "
        "out_proj priced DENSE (d²) at out_proj_rank=0"
    )
    print(
        f"{'d':>5s} {'f':>6s} {'r_attn':>7s} {'kv':>4s} {'r_out':>6s} "
        f"{'MAC/token':>12s} {'dense':>12s} {'cut':>7s} {'Δparams':>12s}"
    )
    for d in widths:
        h = heads(d)
        f = d_ffn(d)
        base = macs_per_token(d, iters=3, n_heads=h, f=f)
        for r in (0, d // 4, 256):
            for kv in (0, 2):
                for out_r in (0, d // 4):
                    now = macs_per_token(
                        d,
                        iters=3,
                        n_heads=h,
                        attn_latent_rank=r,
                        n_kv_heads=kv,
                        out_proj_rank=out_r,
                        f=f,
                    )
                    dp = param_delta(
                        d,
                        n_heads=h,
                        attn_latent_rank=r,
                        n_kv_heads=kv,
                        out_proj_rank=out_r,
                    )
                    print(
                        f"{d:5d} {f:6d} {r:7d} {kv:4d} {out_r:6d} "
                        f"{now:12,d} {base:12,d} {1 - now / base:7.1%} {dp:+12,d}"
                    )
        print()


if __name__ == "__main__":
    main()
