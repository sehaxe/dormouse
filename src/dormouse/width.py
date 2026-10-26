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


def macs_per_token(
    d_model: int,
    *,
    iters: int,
    n_heads: int,
    attn_latent_rank: int = 0,
    n_kv_heads: int = 0,
    out_proj_rank: int = 0,
    f: int | None = None,
) -> int:
    """The dense bill per token: `iters · (attn + out_proj + d·f)`."""
    f = d_ffn(d_model) if f is None else f
    return iters * (
        attn_macs(d_model, n_heads, attn_latent_rank, n_kv_heads)
        + out_proj_macs(d_model, out_proj_rank)
        + d_model * f
    )


def attn_param_delta(d_model: int, n_heads: int, attn_latent_rank: int, n_kv_heads: int) -> int:
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
