#!/usr/bin/env python3
"""Static capacity arithmetic for one config: params, active MAC/byte, bank share.

    python scripts/capacity_math.py m400b
    python scripts/capacity_math.py configs/m400b.toml --set n_experts=128 --set d_ffn=4216
    python scripts/capacity_math.py small --batch 8 --seq 1024 --live 8

WHY THIS FILE EXISTS. Every capacity decision on record was argued from a
parameter COUNT, and a parameter count cannot say whether the parameter is
touched by a token. The 199M m400b model has a 134M expert bank; top-2 of 512
touches 2 x 262k = 0.4% of it per token (the exact disappearance of the bank
from the active ledger is why `width.macs_per_token` has never priced it). The
same question in the other direction: at 22.2 GB of corpus, how many bytes reach
one expert? This script prints both, so "we cannot train the bank" and "the bank
is not trainable at this rank" stop being the same sentence.

THE LEDGER, named because an MFU number without one is not a number
(`docs/research/2026-11-05-small-model-mfu.md` §0.2):

  body   = `width.macs_per_token` — `iters x (attn + out_proj + d x f)`, the
           project's own dense bill. `out_proj` is priced at its real rank
           (`out_proj_rank or rank`) under TSCT, dense `d^2` under
           `use_tsct=false`. The output head, norms, CSA/mHC/Engram/LayerRoPE
           are identical across the capacity arms and are omitted here.
  routed = assignments x (`2 x d x rank + rank x f`) — the dual-latent GLU:
           two `d -> rank` projections and one `rank -> f` write per
           assignment. `assignments` is `moe_topk` on the sparse route and
           `n_experts` on the dense blend (`moe_topk = 0`).
  shared = `2 x d x shared_rank + shared_rank x f` — the always-on expert.
  router = `d x n_experts`.

One byte is one token in this model (vocab 256, flat byte stream), so MAC/token
IS MAC/byte. Called by `scripts/ab_capacity.py` to refuse an unequal-compute
comparison; `tests/test_ab_capacity.py` pins the two configs below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PORT = Path(__file__).resolve().parents[1]
if str(PORT / "src") not in sys.path:
    sys.path.insert(0, str(PORT / "src"))

from dormouse.config import ModelConfig, RunConfig, load_run  # noqa: E402
from dormouse.instrument import LoudError  # noqa: E402
from dormouse.model import DormouseModel  # noqa: E402

#: The A/B's one shape: b8 x s1024 = 8 192 bytes/step, the m400b recipe.
BATCH = 8
SEQ = 1024
#: The measured corpus on disk (`configs/mix_v7.toml` header: 22.2 GB).
BUDGET_GB = 22.2
#: The floor the question is stated in: one million tokens per expert.
TOKENS_PER_EXPERT = 1_000_000


def as_model(run_or_cfg: RunConfig | ModelConfig) -> ModelConfig:
    """Accept the config half or a whole run; `RunConfig` is only a wrapper."""
    return run_or_cfg.model if isinstance(run_or_cfg, RunConfig) else run_or_cfg


def assignments_per_token(cfg: ModelConfig) -> int:
    """How many experts one token runs: top-k, or every expert on the dense blend."""
    return cfg.n_experts if cfg.moe_topk == 0 else cfg.moe_topk


def active_macs_per_token(cfg: ModelConfig) -> int:
    """The active bill of one byte: the `width` ledger plus the MoE terms.

    The `width.macs_per_token` call is deliberately INSIDE this function (not a
    module-level import alias) so a test can monkeypatch the project's function
    and prove this path really calls it rather than a private copy.
    """
    from dormouse.width import active_macs_per_token as width_active

    c = as_model(cfg)
    d, r = c.d_model, c.rank
    out_rank = c.out_proj_rank or (r if c.use_tsct else 0)
    return width_active(
        d,
        iters=c.max_iter,
        n_heads=c.n_heads,
        attn_latent_rank=c.attn_latent_rank,
        n_kv_heads=c.n_kv_heads,
        out_proj_rank=out_rank,
        f=c.d_ffn,
        ffn_form=c.ffn_form,
        ffn_rank=r,
        ffn_rank_gate=c.ffn_rank_gate,
        ffn_rank_down=c.ffn_rank_down,
        n_experts=c.n_experts,
        topk=c.moe_topk,
        shared_rank=c.shared_rank,
    )


def routed_bank_params(model: DormouseModel) -> int:
    """Parameters the routed bank owns: the experts plus the shared input basis.

    `loop.moe.experts.*` is `[gate_u, up_u, gate_s, up_s, v]` per routed expert;
    `loop.moe.gate_u`/`up_u` is the one basis of `share_u_mode='plane'`; the
    `dict` mode's `loop.moe.u_dict` is the same idea. The always-on shared
    expert is NOT counted: it runs in every arm and is not the bank whose size
    is the question.
    """
    if model.cfg.ffn_form == "dense":
        # One active FFN, not a bank: counting it as "routed capacity" would
        # price the baseline as if it carried the thing it replaced.
        return 0
    total = 0
    for name, p in model.named_parameters():
        if name.startswith("loop.moe.experts.") or name in (
            "loop.moe.gate_u",
            "loop.moe.up_u",
            "loop.moe.u_dict",
        ):
            total += p.numel()
    assert total >= 0, "a negative bank is impossible; the name filter is wrong"
    return total


def _ffn_slot(cfg: ModelConfig) -> int:
    """The expert slot's parameter bill, by form (`width.ffn_params`)."""
    from dormouse.width import ffn_params

    return ffn_params(
        cfg.ffn_form,
        cfg.d_model,
        cfg.d_ffn,
        rank=cfg.rank,
        rank_gate=cfg.ffn_rank_gate,
        rank_down=cfg.ffn_rank_down,
        n_experts=cfg.n_experts,
        shared_rank=cfg.shared_rank,
    )


def measure(
    run_or_cfg: RunConfig | ModelConfig,
    *,
    batch: int = BATCH,
    seq: int = SEQ,
    budget_gb: float = BUDGET_GB,
    live: int | None = None,
) -> dict[str, float | int | bool | None]:
    """The capacity row for one config. Builds the model on `meta` (no storage).

    `live` is `route_hist`'s live-expert count. With it, the honest reading is
    the EFFECTIVE tokens per LIVE expert: the uniform numbers below are an upper
    bound, and the m400 measurement was 8/512 live (`route_hist` 2026-11-04).
    """
    cfg = as_model(run_or_cfg)
    model = DormouseModel(cfg, device="meta")
    params = sum(p.numel() for p in model.parameters())
    bank = routed_bank_params(model)
    macs = active_macs_per_token(cfg)
    tokens = batch * seq
    width = assignments_per_token(cfg)
    assignments = tokens * width
    per_expert = assignments / cfg.n_experts
    budget_bytes = int(budget_gb * 1_000_000_000)
    bytes_one_million = TOKENS_PER_EXPERT * cfg.n_experts / width
    tokens_at_budget = budget_bytes * width / cfg.n_experts
    live_tokens = budget_bytes * width / live if live and live > 0 else None
    return {
        "params": params,
        "bank": bank,
        "bank_share": bank / params if params else 0.0,
        "active_macs": macs,
        "batch": batch,
        "seq": seq,
        "tokens_per_step": tokens,
        "assignments_per_token": width,
        "tokens_per_expert_step": per_expert,
        "bytes_for_1m_per_expert": bytes_one_million,
        "tokens_per_expert_at_budget": tokens_at_budget,
        "live": live,
        "tokens_per_live_expert_at_budget": live_tokens,
        "trainable": bytes_one_million <= budget_bytes,
        "d_model": cfg.d_model,
        "d_ffn": cfg.d_ffn,
        "n_experts": cfg.n_experts,
        "rank": cfg.rank,
        "moe_topk": cfg.moe_topk,
        "use_tsct": cfg.use_tsct,
        "share_u_mode": cfg.share_u_mode,
        "ffn_form": cfg.ffn_form,
        "ffn_rank_gate": cfg.ffn_rank_gate,
        "ffn_rank_down": cfg.ffn_rank_down,
        "ffn_slot": _ffn_slot(cfg),
    }


def print_row(row: dict, *, budget_gb: float = BUDGET_GB) -> None:
    """The human table. Every line names its unit, because the disagreement here
    has twice been a unit disagreement (MAC vs FLOP, byte vs token)."""
    print(f"config                {row.get('source', '<resolved>')}")
    print(
        f"geometry              d={row['d_model']} f={row['d_ffn']} "
        f"E={row['n_experts']} rank={row['rank']} topk={row['moe_topk']} "
        f"tsct={row['use_tsct']} share_u={row['share_u_mode']} "
        f"ffn={row['ffn_form']}"
    )
    print(f"params                {row['params']:,}")
    print(f"bank                  {row['bank']:,}  ({100 * row['bank_share']:.1f}% of params)")
    print(f"active MAC/byte       {row['active_macs']:,}  (1 byte = 1 token)")
    print(f"tokens/step           {row['batch']} x {row['seq']} = {row['tokens_per_step']:,} bytes")
    print(
        f"assignments/token     {row['assignments_per_token']}  ->  "
        f"{row['tokens_per_expert_step']:.1f} tokens/expert/step"
    )
    need = row["bytes_for_1m_per_expert"]
    print(
        f"1M tokens/expert      {need:,.0f} B = {need / 1e9:.3f} GB  "
        f"({'inside' if row['trainable'] else 'ABOVE'} the {budget_gb:g} GB corpus)"
    )
    print(
        f"at {budget_gb:g} GB         {row['tokens_per_expert_at_budget']:,.0f} "
        f"tokens/expert if routing is uniform"
    )
    if row["live"]:
        print(
            f"at {budget_gb:g} GB, live={row['live']}  "
            f"{row['tokens_per_live_expert_at_budget']:,.0f} tokens/live expert"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("preset", nargs="?", default="m400b", help="preset NAME or a .toml PATH")
    ap.add_argument(
        "--set", action="append", default=[], metavar="key=value", help="model override"
    )
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--budget-gb", type=float, default=BUDGET_GB)
    ap.add_argument("--live", type=int, default=None, help="route_hist live-expert count")
    a = ap.parse_args(argv)

    model_over: dict[str, object] = {}
    for item in a.set:
        if item.count("=") != 1:
            raise LoudError(f"--set {item!r}: expected exactly one `key=value`")
        k, raw = item.split("=", 1)
        if not raw:
            raise LoudError(f"--set {item!r}: empty value")
        import tomllib

        try:
            value: object = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw
        model_over[k] = value

    run = load_run(a.preset, model_overrides=model_over)
    row = measure(run, batch=a.batch, seq=a.seq, budget_gb=a.budget_gb, live=a.live)
    row["source"] = run.source
    print_row(row, budget_gb=a.budget_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
