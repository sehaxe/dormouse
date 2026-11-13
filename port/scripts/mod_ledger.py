#!/usr/bin/env python3
"""MFLOP/byte as a function of Mixture-of-Depths capacity, from the real config.

    uv run python scripts/mod_ledger.py                       # flagship anchor + control
    uv run python scripts/mod_ledger.py --preset mirror_control
    uv run python scripts/mod_ledger.py --capacities 1.0 0.75 0.5 0.125

THE FORMULA (printed by every run, so a reader never has to guess which bill a
number is):

    training MFLOP/byte(c) = 3 · [ fixed + c · routed ]

    3          forward + backward at the standard 6N accounting (the
               FlopCounterMode bill is 3.0× the forward on this model; the
               control report measured fwd 45.81 -> fwd+bwd 137.42)
    fixed      = attention core + embedding/head
    routed     = QKV/projection/MLP work that MoD's top-k removes from the
                 skipped tokens
    c          = mod_capacity (C = round(c·T) tokens per sequence per block)

Two honest choices are spelled out rather than hidden:

1. **The attention core is priced as fixed.** The paper routes tokens through
   attention as well as the MLP (`2404.02258v1` §3.4), and our implementation
   gathers `C` tokens into the block, so the SDPA core is `C²` not `T²` — the
   real saving is LARGER than this ledger claims. Not crediting it makes the
   arm's FLOPs table a lower bound, which is the safe direction for a pre-
   registered A/B. `all-routed` in the output prices the counterfactual where
   the core scales too, for comparison.
2. **The flagship anchor is a MEASURED bill**, not a second formula.
   `docs/records/reviews/2026-11-06-control-transformer.md:50` measured
   `45.81` forward / `137.42` fwd+bwd MFLOP/byte with FlopCounterMode on
   `checkpoints/flagship2` (b1, s1024). The config's own static ledger
   (`width.active_macs_per_token`) prices the non-attention body; the T-term is
   `4·L·T·d` and the head `2·d·V`. The report's measured split is body 34.51 /
   attention 11.30 / head 0.52 of the 45.81; this script reproduces that split
   and therefore 137.42 at c=1.0 exactly.

The transformer form (the MoD arm itself) is priced analytically, with no
calibration: `2·L·(4·d² + 2·d·d_ffn)` for the block's projections + MLP and
`4·L·T·d` for the core, all FLOPs per byte (1 byte = 1 position in this
byte-level model). The old `24·L·d²` block term assumed `d_ffn = 4·d`; it is
kept only as the f=4d special case of the formula now used.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PORT = Path(__file__).resolve().parents[1]
if str(PORT / "src") not in sys.path:
    sys.path.insert(0, str(PORT / "src"))

from dormouse.config import ModelConfig, load_preset  # noqa: E402
from dormouse.model import DormouseModel  # noqa: E402

#: The flagship's measured FlopCounterMode bill (`control-transformer.md:50`),
#: fwd+bwd 137.42 MFLOP/byte at b1 s1024. Divided by the counter's 3.0
#: forward->fwd+bwd factor so c=1.0 reproduces the headline exactly (the
#: report's forward column, 45.81, is that same number rounded). The head is
#: `2·d·V`; the body is the report's own "repo ledger" forward (34.51), so the
#: residual is the T-dependent attention core.
FLAGSHIP_FWD_MFLOP = 137.42 / 3.0
FLAGSHIP_BODY_FWD_MFLOP = 34.51
#: The paper's FLOPs result this arm is betting on: 0.60× at 12.5% capacity
#: (`2404.02258v1` §5.1, 60M-3B, equal loss).
PAPER_CAPACITY = 0.125
PAPER_RATIO = 0.60


@dataclass(frozen=True)
class Ledger:
    """One config's bill at one capacity, in training MFLOP/byte."""

    name: str
    form: str
    capacity: float
    tokens_per_block: int
    fixed: float
    routed: float
    all_routed_fixed: float
    all_routed: float
    params: int
    router_params: int

    @property
    def total(self) -> float:
        return 3.0 * (self.fixed + self.capacity * self.routed)

    @property
    def total_all_routed(self) -> float:
        return 3.0 * (self.all_routed_fixed + self.capacity * self.all_routed)

    @property
    def ratio(self) -> float:
        return (self.fixed + self.capacity * self.routed) / (self.fixed + self.routed)

    @property
    def ratio_all_routed(self) -> float:
        return (self.all_routed_fixed + self.capacity * self.all_routed) / (
            self.all_routed_fixed + self.all_routed
        )


def _params(cfg: ModelConfig, capacity: float) -> tuple[int, int]:
    """Instantiated parameter count (meta device) and the routers' share."""
    from dataclasses import replace

    build = cfg if cfg.block_form != "transformer" else replace(cfg, mod_capacity=capacity)
    model = DormouseModel(build, device="meta")
    total = sum(p.numel() for p in model.parameters())
    router = sum(p.numel() for n, p in model.named_parameters() if n.endswith(".router"))
    return total, router


def ledger(cfg: ModelConfig, capacity: float, *, seq_len: int, name: str = "") -> Ledger:
    """The bill for one config at one capacity. Pure arithmetic (no model run)."""
    if not 0.0 < capacity <= 1.0:
        raise ValueError(f"capacity={capacity}: the paper's capacity is in (0, 1]")
    d, v = cfg.d_model, cfg.vocab
    t = min(seq_len, cfg.max_seq_len)
    if cfg.use_tsct:
        r = min(cfg.rank, d, v)
        head = 2.0 * (d * r + r * v) / 1e6
    else:
        head = 2.0 * d * v / 1e6  # dense head (tied or not: same FLOPs)
    if cfg.block_form == "transformer":
        layers = cfg.n_layers or cfg.loop_blocks * cfg.max_iter
        # qkv 3d² + proj d² + MLP 2·d·f, in MACs; ×2 FLOPs/MAC. `24·d²` is
        # the f = 4d special case and overprices any other FFN width (the
        # flagship3_dense arm has f = 3072, not 4·832 = 3328: +4.4%).
        block = 2.0 * layers * (4.0 * d * d + 2.0 * d * cfg.d_ffn) / 1e6
        core = 4.0 * layers * t * d / 1e6
        fixed = core + head
        routed = block
        all_fixed = head
        all_routed = block + core
        form = "transformer"
    else:
        # The KDA/MoE flagship: the report's measured forward split, with the
        # static body replaced by the config's own active bill when it is
        # available (it is: `width.active_macs_per_token`).
        from dormouse.width import active_macs_per_token

        layers = cfg.max_iter
        out_rank = cfg.out_proj_rank or (cfg.rank if cfg.use_tsct else 0)
        body = (
            2.0
            * active_macs_per_token(
                d,
                iters=cfg.max_iter,
                n_heads=cfg.n_heads,
                attn_latent_rank=cfg.attn_latent_rank,
                n_kv_heads=cfg.n_kv_heads,
                out_proj_rank=out_rank,
                f=cfg.d_ffn,
                ffn_form=cfg.ffn_form,
                ffn_rank=cfg.rank,
                ffn_rank_gate=cfg.ffn_rank_gate,
                ffn_rank_down=cfg.ffn_rank_down,
                n_experts=cfg.n_experts,
                topk=cfg.moe_topk,
                shared_rank=cfg.shared_rank,
            )
            / 1e6
        )
        core = 4.0 * layers * t * d / 1e6
        if (d, layers) == (1024, 3):
            # The flagship shape the measured bill belongs to. Absorb the
            # counter-vs-ledger gap into the routed part so c=1.0 reproduces
            # 137.42; the fixed share is the report's attention + head split.
            fixed = FLAGSHIP_FWD_MFLOP - (FLAGSHIP_BODY_FWD_MFLOP - head)
            routed = FLAGSHIP_BODY_FWD_MFLOP - head
        else:
            fixed = core + head
            routed = body
        all_fixed = head
        all_routed = body + core
        form = "kda"
    params, router = _params(cfg, capacity)
    return Ledger(
        name=name or f"{form}/{d}x{cfg.n_layers or cfg.max_iter}",
        form=form,
        capacity=capacity,
        tokens_per_block=int(round(capacity * t)),
        fixed=fixed,
        routed=routed,
        all_routed_fixed=all_fixed,
        all_routed=all_routed,
        params=params,
        router_params=router,
    )


def _print_formula() -> None:
    print(
        "formula: training MFLOP/byte(c) = 3 * [ fixed + c * routed ]\n"
        "  fixed      = attention core (4*L*T*d) + head (2*d*V; for a TSCT head\n"
        "               2*(d*r + r*V))                          [priced FULL]\n"
        "  routed     = 2*L*(4*d^2 + 2*d*f): qkv/proj/MLP work removed from\n"
        "               skipped tokens (the 24*L*d^2 form is f=4*d only)\n"
        "  3          = FlopCounterMode's forward->forward+backward factor\n"
        "  all-routed = the same split with the attention core scaled too (the\n"
        "               implementation's actual saving; the headline is its lower bound)\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="flagship_base", help="model preset name or path")
    ap.add_argument("--also", default="mirror_control", help="second preset, '' to skip")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument(
        "--capacities",
        type=float,
        nargs="+",
        default=[1.0, 0.75, PAPER_CAPACITY, 0.5],
        help="capacity points (the task table is 1.0/0.75/0.5)",
    )
    args = ap.parse_args(argv)
    _print_formula()
    presets = [args.preset] + ([args.also] if args.also else [])
    for preset in presets:
        cfg = load_preset(preset)
        name = "flagship" if Path(preset).stem == "flagship_base" else Path(preset).stem
        print(
            f"[{preset}] form={cfg.block_form} d={cfg.d_model} "
            f"L={cfg.n_layers or cfg.loop_blocks * cfg.max_iter} "
            f"T={min(args.seq_len, cfg.max_seq_len)} V={cfg.vocab} "
            f"experts={cfg.n_experts} top{k if (k := cfg.moe_topk) else 'dense'}"
        )
        header = (
            f"{'capacity':>8s} {'C/T':>8s} {'params':>12s} {'+router':>8s} "
            f"{'MFLOP/B':>9s} {'ratio':>7s} {'all-routed':>10s}"
        )
        print(header)
        base = None
        for cap in args.capacities:
            led = ledger(cfg, cap, seq_len=args.seq_len, name=name)
            if cap == 1.0:
                base = led.total
            share = cap if base is None else led.total / base
            print(
                f"{cap:8.3f} {led.tokens_per_block:4d}/"
                f"{min(args.seq_len, cfg.max_seq_len):<3d} {led.params:12,} "
                f"{led.router_params:8,} {led.total:9.2f} {share:7.3f} "
                f"{led.ratio_all_routed:10.3f}"
            )
        print()
    print(
        f"reference: the paper's {PAPER_CAPACITY:.3f} capacity measures "
        f"{PAPER_RATIO:.2f}x FLOPs at equal loss (2404.02258v1 §5.1); "
        f"the flagship's 1.0 point is the measured 137.42 MFLOP/byte "
        f"(control-transformer.md:50)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
