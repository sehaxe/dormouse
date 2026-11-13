"""The fresh-run restart recipe, as a command and a token budget.

    python scripts/plan_restart.py [config] [--n-experts E] [--route-json PATH]

Why: `route_hist.py` measured 80 of 512 experts alive on real bytes
(`docs/records/reviews/2026-11-04-routing-collapse-CONFIRMED.md`). The restart is a
decision — the balance arm, and whether E is worth paying for — so it prints the
exact command, both token budgets and the warning that the bank is oversized for
the routing that actually happens. No model load, no GPU, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

PORT = Path(__file__).resolve().parents[1]
REPO = PORT.parent
sys.path.insert(0, str(PORT / "src"))

DEFAULT_CONFIG = "configs/m400.toml"
#: Chinchilla's 20 bytes (tokens, byte-level) per parameter.
BYTES_PER_PARAM = 20
#: One pass of `configs/mix_v7.toml` (the measured 22.2 GB on disk).
MIX_V7_BYTES = 22_200_000_000
#: The measured throughput the ETA is quoted at.
MEASURED_BPS = 22_000.0

#: The owner's settled recipe, in command order. `n_experts`, `moe_kernel` and
#: `moe_lb_mode` are substituted per run; everything else is fixed.
RECIPE: tuple[tuple[str, str], ...] = (
    ("batch", "8"),
    ("seq_len", "1024"),
    ("grad_accum", "1"),
    ("lr", "5e-4"),
    ("decay_frac", "0.4"),
    ("min_lr_frac", "0.01"),
    ("use_msa", "false"),
    ("use_csa", "true"),
    ("csa_chunk", "8"),
    ("csa_coarse_chunk", "256"),
    ("use_mhc", "true"),
    ("use_attnres", "true"),
    ("share_u", "true"),
    ("use_engram", "true"),
    ("engram_ram", "false"),
    ("attn_latent_rank", "256"),
    ("moe_kernel", "triton"),
    ("doc_masking", "false"),
    ("layerrope", "true"),
    ("layerrope_form", "paper"),
    ("layerrope_index", "iteration"),
    ("layerrope_init", "paper"),
    ("rank", "64"),
    ("n_experts", "512"),
    ("seed", "1"),
    ("opt_graph", "false"),
    ("iter_checkpoint", "false"),
    ("moe_lb_mode", "prop"),
)


def build_command(
    config: str,
    n_experts: int,
    kernel: str = "triton",
    balance: str = "prop",
) -> str:
    """The fresh-run command: the recipe, `--resume` is never emitted."""
    flags = dict(RECIPE)
    flags["moe_kernel"] = kernel
    flags["n_experts"] = str(int(n_experts))
    flags["moe_lb_mode"] = balance
    sets = " ".join(f"--set {k}={v}" for k, v in flags.items())
    return f".venv/bin/dormouse train {config} {sets}"


def budget(
    params: int,
    *,
    bytes_per_param: int = BYTES_PER_PARAM,
    one_pass: int = MIX_V7_BYTES,
    bps: float = MEASURED_BPS,
) -> dict:
    """Both budgets and their ETAs at the measured rate (bytes are tokens)."""
    chinchilla = int(params) * int(bytes_per_param)
    return {
        "params": int(params),
        "chinchilla_tokens": chinchilla,
        "chinchilla_gb": chinchilla / 1e9,
        "one_pass_tokens": int(one_pass),
        "one_pass_gb": one_pass / 1e9,
        "passes": one_pass / max(1, chinchilla),
        "eta_s": chinchilla / max(1.0, bps),
        "one_pass_eta_s": one_pass / max(1.0, bps),
    }


def oversize_warning(n_experts: int, live: int | None) -> str | None:
    """The next decision: E above 4x the experts routing actually uses."""
    if not live:
        return None
    if n_experts <= 4 * live:
        return None
    return (
        f"WARNING: E={n_experts} is {n_experts / live:.1f}x the {live} live experts "
        f"route_hist measured — the bank is oversized for the routing that actually "
        f"happens. Decision: shrink E to <= {4 * live}, or prove the balance arm "
        f"revives more before paying for the dead weight."
    )


def balance_mode() -> str:
    """Whichever balance arm the tree landed; `prop` while none is there."""
    try:
        from dormouse import config as cfg
    except Exception:
        return "prop"
    for name in ("MOE_LB_MODES", "BALANCE_MODES"):
        modes = getattr(cfg, name, None)
        if modes:
            return "prop" if "prop" in modes else str(modes[0])
    return "prop"


def resolve_config(config: str) -> Path:
    p = Path(config)
    candidates = [p]
    if not p.suffix:
        candidates.append(PORT / "configs" / f"{config}.toml")
    candidates += [PORT / config, REPO / config]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"config not found: {config} (tried {tried})")


def display_config(path: Path) -> str:
    try:
        return str(path.relative_to(PORT))
    except ValueError:
        return str(path)


def default_kernel(path: Path) -> str:
    """The config's explicit kernel, else the measured-best `triton`."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    kernel = raw.get("moe_kernel")
    return kernel if isinstance(kernel, str) and kernel else "triton"


def _typed(flags: dict[str, str]) -> dict:
    out: dict = {}
    for k, v in flags.items():
        if v in ("true", "false"):
            out[k] = v == "true"
            continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


MODEL_KEYS = (
    "use_msa",
    "use_csa",
    "csa_chunk",
    "csa_coarse_chunk",
    "use_mhc",
    "use_attnres",
    "share_u",
    "use_engram",
    "attn_latent_rank",
    "moe_kernel",
    "layerrope",
    "layerrope_form",
    "layerrope_index",
    "layerrope_init",
    "rank",
    "n_experts",
)
TRAIN_KEYS = (
    "batch",
    "seq_len",
    "grad_accum",
    "lr",
    "decay_frac",
    "min_lr_frac",
    "engram_ram",
    "doc_masking",
    "seed",
    "opt_graph",
    "iter_checkpoint",
)


def params_for(path: Path, n_experts: int, kernel: str) -> int:
    """Instantiate the model on `meta`: exact parameter count, zero storage.

    The overrides are the COMMAND's own `RECIPE`, typed — one list, so the
    budget cannot describe a different model than the command trains.
    """
    from dormouse.config import load_run
    from dormouse.model import DormouseModel, param_count

    typed = _typed(dict(RECIPE))
    typed["n_experts"] = int(n_experts)
    typed["moe_kernel"] = kernel
    model_over = {k: typed[k] for k in MODEL_KEYS}
    train_over = {k: typed[k] for k in TRAIN_KEYS}
    run = load_run(str(path), model_overrides=model_over, train_overrides=train_over)
    return param_count(DormouseModel(run.model, device="meta"))


def find_route_json(explicit: str | None, stem: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"--route-json: {explicit} is not a file")
        return p
    for p in (
        REPO / "checkpoints" / stem / f"{stem}.route.json",
        Path.home() / "logs" / f"{stem}_route.json",
    ):
        if p.is_file():
            return p
    return None


def read_live(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    live = d.get("live_experts")
    if live is None and d.get("counts"):
        n = int(d.get("n_experts") or len(d["counts"]))
        live = sum(1 for v in d["counts"][:n] if v)
    return int(live) if live is not None else None


def plan(
    config: str = DEFAULT_CONFIG,
    *,
    n_experts: int | None = None,
    kernel: str | None = None,
    balance: str | None = None,
    route_json: str | None = None,
) -> dict:
    path = resolve_config(config)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    e = int(n_experts or raw.get("n_experts") or 512)
    k = kernel or default_kernel(path)
    b = balance or balance_mode()
    shown = display_config(path)
    route = find_route_json(route_json, path.stem)
    live = read_live(route)
    params = params_for(path, e, k)
    return {
        "config": shown,
        "resolved": str(path),
        "n_experts": e,
        "d_model": raw.get("d_model"),
        "kernel": k,
        "balance": b,
        "command": build_command(shown, e, k, b),
        "params": params,
        "budget": budget(params),
        "route_json": str(route) if route else None,
        "live_experts": live,
        "warning": oversize_warning(e, live),
    }


def _days(s: float) -> str:
    return f"{s / 86400:.2f} d"


def format_plan(p: dict) -> str:
    b = p["budget"]
    lines = [
        f"config  {p['config']}",
        "# run from port/ — FRESH run, no --resume",
        p["command"],
        "",
        f"params  {p['params']:,}  (d={p['d_model']} · E={p['n_experts']}"
        f" · kernel {p['kernel']} · balance {p['balance']})",
        f"budget  20 B/param Chinchilla: {b['chinchilla_gb']:.2f} GB"
        f" ({b['chinchilla_tokens']:,} tokens)",
        f"        one pass of mix_v7:    {b['one_pass_gb']:.1f} GB ({b['passes']:.1f}x Chinchilla)",
        f"eta     at {MEASURED_BPS:,.0f} B/s: {_days(b['eta_s'])} Chinchilla"
        f" · {_days(b['one_pass_eta_s'])} one pass",
    ]
    if p["live_experts"] is None:
        lines.append(
            "routing unmeasured — run `scripts/route_hist.py --json "
            "checkpoints/<run>/<run>.route.json`; no size warning today"
        )
    else:
        live = p["live_experts"]
        frac = 100.0 * live / max(1, p["n_experts"])
        lines.append(f"routing {live}/{p['n_experts']} live ({frac:.1f}%) from {p['route_json']}")
    if p["warning"]:
        lines.append(p["warning"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    ap.add_argument("--n-experts", "--experts", type=int, dest="n_experts")
    ap.add_argument("--kernel", help="moe_kernel arm (default: config, else triton)")
    ap.add_argument("--balance", help="balance arm (default: whichever landed, else prop)")
    ap.add_argument("--route-json", metavar="PATH", help="route_hist.py --json output")
    a = ap.parse_args(argv)
    print(
        format_plan(
            plan(
                a.config,
                n_experts=a.n_experts,
                kernel=a.kernel,
                balance=a.balance,
                route_json=a.route_json,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
