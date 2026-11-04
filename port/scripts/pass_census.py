"""Static census of the POINTWISE passes in ONE loop iteration — arithmetic, no card.

One `LoopBlock._step` + `_step_post` + the per-iteration CE, at the resolved
config and `b x s`. For every elementwise op the iteration performs, this
prints the tensor shape, the dtype it reads and writes, the bytes moved
(read + written, the traffic a pass actually costs) and one launch. The rows
are hand-mapped to the code (each row names the file and the call), because a
pointwise kernel count is a property of the *code path*, not of the config:
the config only supplies the shapes and chooses which arm runs.

Why this exists: a fusion's ceiling is the traffic it removes plus the launches
it removes. Both are countable here before a kernel is written. The reference
unit is the owner's: one `[b·s, d_model]` bf16 tensor is 25.2 MB at the m400
shape and 13.3 us at 1.90 TB/s; "passes" below are measured in that unit.

    uv run python scripts/pass_census.py                       # the m400b 199M run
    uv run python scripts/pass_census.py --set fused_residual=true --set fused_epilogue=true
    uv run python scripts/pass_census.py --set fused_kda_tail=true
    uv run python scripts/pass_census.py --config configs/medium.toml --batch 8 --seq 512

`--json PATH` writes every row for a diff (`--diff A.json B.json` is the same
delta the `fused_kda_tail` section of the report quotes).

WHAT IS NOT COUNTED, stated so the total is not read as a whole step:

* GEMMs and the FLA KDA kernel. The census is the elementwise seam between
  them; the two dense GEMM chains are compute, not traffic.
* The MoE kernel's interior. With `moe_kernel="triton"` the GLU (silu and the
  gate multiply) runs INSIDE one launch over the routed rows, so it is one
  launch, not N passes; its bytes are the kernel's own tiles, not a torch pass.
* The Engram gather (`tables[order_idx, slots]`): random 32-float rows, not a
  dense pass. Its VALUE projection and the gate reduction ARE counted.
* Another attention arm. The KDA rows assume `use_kda=true` (the measured
  run); `attn_kind="full"` / `attn_window` / `use_msa` are not modeled, and a
  census of one of those arms is a new row list, not a flag.
* The window-level tail (`out_acc = zeros_like`, `/ ran`): once per window, not
  once per iteration; printed as a footnote when non-zero.
* The backward. A training step pays it too, and a fused forward usually fuses
  its VJP as well; this census is the forward's arithmetic, the part the
  fusion's own parity gate can check on a CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dormouse.config import ModelConfig, validate

#: `<repo>/port/scripts/pass_census.py` -> `<repo>`; the saved run snapshots
#: live under `checkpoints/`, outside the `port/` package but inside the repo.
ROOT = Path(__file__).resolve().parents[2]
#: The m400b run (the 199M model): d=1536, E=512, top-2, KDA + CSA + Engram +
#: mHC/AttnRes + LayerRoPE-paper, b8 x s1024, max_iter 3.
DEFAULT_CONFIG = str(ROOT / "checkpoints/m400b/m400b.config.toml")
DEFAULT_BATCH, DEFAULT_SEQ = 8, 1024
#: The card the owner measured on (RTX 5070 Ti, ~1.9 TB/s of HBM).
DEFAULT_BW = 1.9e12

#: Element size per dtype name; the census never imports torch (it must run
#: beside a busy card, and a config is arithmetic, not a graph).
ITEMSIZE = {"f32": 4, "bf16": 2, "f16": 2, "int64": 8, "int32": 4}


def nbytes(shape: tuple[int, ...], dtype: str) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * ITEMSIZE[dtype]


@dataclass(frozen=True)
class Row:
    """One elementwise op: the reason the census can be audited op by op."""

    group: str
    op: str
    site: str
    shape: tuple[int, ...]
    reads: tuple[tuple[tuple[int, ...], str], ...]
    writes: tuple[tuple[tuple[int, ...], str], ...]
    launches: int = 1

    @property
    def nbytes(self) -> int:
        return sum(nbytes(s, d) for s, d in self.reads) + sum(nbytes(s, d) for s, d in self.writes)


@dataclass(frozen=True)
class Ctx:
    """The shapes one row needs, resolved once from the config."""

    b: int
    s: int
    d: int
    h: int
    hd: int
    kv: int  # the k/v PROJECTION width (kv_heads * head_dim)
    f: int
    vocab: int
    k: int  # mHC lanes
    orders: int
    engram_dim: int
    chunk: int
    coarse: int
    kv_csa: int
    bf16: bool

    @property
    def n(self) -> int:
        return self.b * self.s


def ctx_for(cfg: ModelConfig, b: int, s: int) -> Ctx:
    return Ctx(
        b=b,
        s=s,
        d=cfg.d_model,
        h=cfg.n_heads,
        hd=cfg.head_dim,
        kv=(cfg.n_kv_heads or cfg.n_heads) * cfg.head_dim,
        f=cfg.d_ffn,
        vocab=cfg.vocab,
        k=cfg.mhc_streams if getattr(cfg, "use_mhc", False) else 1,
        orders=len(cfg.engram_orders) if cfg.use_engram else 0,
        engram_dim=cfg.engram_dim,
        chunk=max(int(cfg.csa_chunk), 1),
        coarse=int(cfg.csa_coarse_chunk),
        kv_csa=int(cfg.csa_kv_dim),
        bf16=bool(cfg.bf16),
    )


def _rows(cfg: ModelConfig, c: Ctx) -> list[Row]:
    """The iteration's elementwise ops, in execution order, from the code.

    Sites are `file:function`; each row is one torch kernel on the eager path
    (a `view`/`reshape`/`transpose` allocates nothing and is not a row).
    """
    n, d, k = c.n, c.d, c.k
    p = ((n, d), "f32")  # one [N, d] fp32 tensor
    p16 = ((n, d), "bf16")
    kd = ((n, k, d), "f32")
    fused_kda = bool(getattr(cfg, "fused_kda_tail", False))
    rows: list[Row] = []

    # ---- loopblock._window / _step: the residual bookkeeping ---------------
    rows.append(
        Row(
            "loop",
            "streams += iter_embed[i]",
            "loopblock._step",
            (n, k, d),
            (kd,),
            (kd,),
        )
    )
    rows.append(
        Row(
            "loop",
            "ctrl = cat([ctx, h0_flat], -1)",
            "loopblock._step",
            (n, 2 * d),
            (p, p),
            (((n, 2 * d), "f32"),),
        )
    )
    rows.append(
        Row(
            "loop",
            "out_acc += step_out",
            "loopblock._window",
            (n, d),
            (p, p),
            (p,),
        )
    )

    # ---- the read half: mHC lanes -> ctx + the body's normed inputs --------
    fused_epilogue = bool(getattr(cfg, "fused_epilogue", False)) and cfg.use_mhc
    if fused_epilogue:
        reads = [kd]
        if cfg.layerrope and cfg.layerrope_form == "paper":
            writes = [p, p, p]  # ctx, normed(attn), normed(mlp): two sites
        else:
            writes = [p, p]  # ctx, normed
        rows.append(
            Row(
                "mhc-read",
                "epilogue_read (lane norm + 8 dots + ctx + RMSNorm)",
                "fused_residual.epilogue_read",
                (n, d),
                tuple(reads),
                tuple(writes),
            )
        )
    else:
        # MHCMix.pre_mix: _x_norm = pow, mean, sqrt, div; then the sigmoid mix.
        rows.append(Row("mhc-read", "_x_norm: x.pow(2)", "mhc._x_norm", (n, k, d), (kd,), (kd,)))
        rows.append(
            Row("mhc-read", "_x_norm: .mean(-1)", "mhc._x_norm", (n,), (kd,), (((n, 1), "f32"),))
        )
        rows.append(
            Row(
                "mhc-read",
                "_x_norm: rsqrt(mean+eps)",
                "mhc._x_norm",
                (n, 1),
                (((n, 1), "f32"),),
                (((n, 1), "f32"),),
            )
        )
        rows.append(Row("mhc-read", "_x_norm: flat * inv", "mhc._x_norm", (n, k, d), (kd,), (kd,)))
        rows.append(
            Row("mhc-read", "sigmoid(pre) * lanes (pre_mix)", "mhc.pre_mix", (n, d), (kd,), (kd,))
        )
        rows.append(Row("mhc-read", "pre_mix: .sum(-2)", "mhc.pre_mix", (n, d), (kd,), (p,)))
        if cfg.layerrope and cfg.layerrope_form == "paper":
            rows.append(
                Row(
                    "mhc-read",
                    "normed = RMSNorm(ctx)*attn_read",
                    "loopblock._step",
                    (n, d),
                    (p,),
                    (p,),
                )
            )
            rows.append(
                Row(
                    "mhc-read",
                    "normed_mlp = RMSNorm(ctx)*mlp_read",
                    "loopblock._step",
                    (n, d),
                    (p,),
                    (p,),
                )
            )
        else:
            rows.append(
                Row("mhc-read", "normed = RMSNorm(ctx)", "loopblock._step", (n, d), (p,), (p,))
            )

    # ---- KDA: the projection tail, then the output scale -------------------
    pv = ((n, c.kv), "f32")
    pv16 = ((n, c.kv), "bf16")
    hf, h16 = ((n, c.h), "f32"), ((n, c.h), "bf16")
    if fused_kda:
        # `q/k/v` casts, `logsigmoid(g)` and `sigmoid(beta)` in one row-program.
        rows.append(
            Row(
                "kda",
                "fused_kda.tail (casts + logsigmoid + sigmoid)",
                "fused_kda.tail",
                (n, d),
                (p, pv, pv, p, hf),
                (p16, pv16, pv16, p, h16),
            )
        )
    else:
        rows.append(Row("kda", "g = logsigmoid(g_proj(x))", "kda.projections", (n, d), (p,), (p,)))
        rows.append(
            Row("kda", "beta = sigmoid(beta_proj(x))", "kda.projections", (n, c.h), (hf,), (hf,))
        )
        if cfg.use_kda and c.bf16:
            rows.append(Row("kda", "q.to(bf16)", "kda._fla_path", (n, d), (p,), (p16,)))
            rows.append(Row("kda", "k.to(bf16)", "kda._fla_path", (n, c.kv), (pv,), (pv16,)))
            rows.append(Row("kda", "v.to(bf16)", "kda._fla_path", (n, c.kv), (pv,), (pv16,)))
            rows.append(Row("kda", "beta.to(bf16)", "kda._fla_path", (n, c.h), (hf,), (h16,)))
    rows.append(
        Row(
            "kda",
            "attn = attn_out * w_attn",
            "loopblock._step",
            (n, d),
            (((n, d), "bf16"), ((n, 1), "f32")),
            (p,),
        )
    )

    # ---- CSA: the additive branch's output scale joins attn ----------------
    if cfg.use_csa:
        dbg = ((n, d), "f32")
        pool_w = ((n, 2 * c.kv_csa), "f32")
        rows.append(
            Row(
                "csa",
                "_pool(x) fine + coarse",
                "csa._pool",
                (n, c.kv_csa),
                (dbg,),
                (pool_w,),
                launches=2,
            )
        )
        rows.append(Row("csa", "out = fine + coarse", "csa.forward", (n, d), (dbg, dbg), (dbg,)))
        rows.append(
            Row(
                "csa", "csa_out * w_attn", "loopblock._step", (n, d), (dbg, ((n, 1), "f32")), (dbg,)
            )
        )
        rows.append(Row("csa", "* sigmoid(csa_gate[i])", "loopblock._step", (n, d), (dbg,), (dbg,)))
        rows.append(Row("csa", "attn = attn + csa", "loopblock._step", (n, d), (dbg, dbg), (dbg,)))

    # ---- Engram: the value gate reduction and the convex mix ---------------
    if cfg.use_engram:
        # value_proj reads the [N, orders, dim] memory and writes [N, orders, d];
        # the gate (sign·sqrt) is the [N, orders] tensor beside it. The reduce
        # is `(value * gate.unsqueeze(-1)).sum(2)` in engram._read.
        vo = ((n, c.orders, d), "f32")
        rows.append(
            Row(
                "engram",
                "value * gate (over orders)",
                "engram._read",
                (n, c.orders, d),
                (vo,),
                (vo,),
            )
        )
        rows.append(Row("engram", ".sum(orders) -> mem", "engram._read", (n, d), (vo,), (p,)))
        rows.append(
            Row(
                "engram",
                "mem_dense: + bias (bank dtype)",
                "loopblock._step",
                (n, d),
                (p16,),
                (p16,),
            )
        )
        rows.append(Row("engram", "mem_dense: .to(fp32)", "loopblock._step", (n, d), (p16,), (p,)))
        rows.append(Row("engram", "memory * lam", "engram.memory_floor_mix", (n, d), (p,), (p,)))
        rows.append(
            Row("engram", "dense * (1 - lam)", "engram.memory_floor_mix", (n, d), (p,), (p,))
        )
        rows.append(
            Row("engram", "memory_floor_mix: add", "engram.memory_floor_mix", (n, d), (p, p), (p,))
        )

    # ---- the MoE seam: two casts and the write gate ------------------------
    rows.append(
        Row("moe", "normed_mlp.to(bank dtype)", "loopblock._moe_branch", (n, d), (p,), (p16,))
    )
    rows.append(Row("moe", "moe_out.to(fp32)", "loopblock._moe_branch", (n, d), (p16,), (p,)))
    rows.append(Row("moe", "ffn = ffn * w_ffn", "loopblock._step_post", (n, d), (p,), (p,)))

    # ---- the write half: y = attn*g + branch*g + ffn*g ---------------------
    paper = bool(cfg.layerrope and cfg.layerrope_form == "paper")
    if fused_kda:
        g = ((d,), "f32")
        reads = [p, p, p] + ([g, g] if paper else [])
        rows.append(
            Row("write", "fused_kda.combine -> y", "fused_kda.combine", (n, d), tuple(reads), (p,))
        )
    else:
        if paper:
            rows.append(
                Row("write", "attn = attn * attn_write", "loopblock._step_post", (n, d), (p,), (p,))
            )
            rows.append(
                Row(
                    "write",
                    "branch = branch * attn_write",
                    "loopblock._step_post",
                    (n, d),
                    (p,),
                    (p,),
                )
            )
            rows.append(
                Row("write", "ffn = ffn * mlp_write", "loopblock._step_post", (n, d), (p,), (p,))
            )
        rows.append(
            Row(
                "write",
                "y = attn + branch (first add)",
                "loopblock._step_post",
                (n, d),
                (p, p),
                (p,),
            )
        )
        rows.append(
            Row("write", "y += ffn (second add)", "loopblock._step_post", (n, d), (p, p), (p,))
        )

    # ---- the write half: the residual operator -----------------------------
    fused_residual = bool(getattr(cfg, "fused_residual", False)) and cfg.use_mhc
    if fused_residual:
        # `write` reads the lanes and y, stores the next lanes and h.
        rows.append(
            Row(
                "mhc-write",
                "fused_residual.write (AttnRes + Sinkhorn + H_res)",
                "fused_residual.write",
                (n, k, d),
                (kd, p),
                (kd, p),
            )
        )
    elif cfg.use_mhc:
        rows.append(
            Row("mhc-write", "_x_norm: pow", "mhc._post_res_tilde", (n, k, d), (kd,), (kd,))
        )
        rows.append(
            Row(
                "mhc-write", "_x_norm: mean", "mhc._post_res_tilde", (n,), (kd,), (((n, 1), "f32"),)
            )
        )
        rows.append(
            Row(
                "mhc-write",
                "_x_norm: rsqrt",
                "mhc._post_res_tilde",
                (n, 1),
                (((n, 1), "f32"),),
                (((n, 1), "f32"),),
            )
        )
        rows.append(
            Row("mhc-write", "_x_norm: * inv", "mhc._post_res_tilde", (n, k, d), (kd,), (kd,))
        )
        rows.append(Row("mhc-write", "H_res x_l (einsum)", "mhc.forward", (n, k, d), (kd,), (kd,)))
        rows.append(Row("mhc-write", "H_post^T * x_in", "mhc.forward", (n, k, d), (kd, p), (kd,)))
        rows.append(
            Row("mhc-write", "mixed + H_post term", "mhc.forward", (n, k, d), (kd, kd), (kd,))
        )
        rows.append(
            Row("mhc-write", "h = lanes.sum(-2)", "loopblock._step_post", (n, d), (kd,), (p,))
        )
    else:
        rows.append(
            Row(
                "mhc-write",
                "h = ctx + y * residual_scale",
                "loopblock._step_post",
                (n, d),
                (p, p),
                (p,),
            )
        )

    # ---- the readout and the per-iteration CE -----------------------------
    rows.append(
        Row("readout", "out_proj: x.to(bf16)", "tsct.TSCTLinear.forward", (n, d), (p,), (p16,))
    )
    rows.append(
        Row("readout", "out_proj: y.to(fp32)", "tsct.TSCTLinear.forward", (n, d), (p16,), (p,))
    )
    v32 = ((n, c.vocab), "f32")
    v16 = ((n, c.vocab), "bf16")
    rows.append(Row("ce", "lm_head: x.to(bf16)", "tsct.TSCTLinear.forward", (n, d), (p,), (p16,)))
    rows.append(Row("ce", "logits.to(fp32)", "loopblock._iter_ce", (n, c.vocab), (v16,), (v32,)))
    rows.append(Row("ce", "log_softmax", "loopblock._iter_ce", (n, c.vocab), (v32,), (v32,)))
    rows.append(
        Row("ce", "gather(targets)", "loopblock._iter_ce", (n,), (v32,), (((n, 1), "f32"),))
    )
    rows.append(
        Row("ce", "ce.mean()", "loopblock._iter_ce", (), (((n, 1), "f32"),), (((), "f32"),))
    )
    return rows


def census(cfg: ModelConfig, b: int, s: int) -> tuple[Ctx, list[Row]]:
    c = ctx_for(cfg, b, s)
    return c, _rows(cfg, c)


def report(cfg: ModelConfig, rows: list[Row], c: Ctx, bw: float, top: int = 0) -> dict:
    n, d = c.n, c.d
    ref = nbytes((n, d), "bf16")
    total = sum(r.nbytes for r in rows)
    launches = sum(r.launches for r in rows)
    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(r.group, []).append(r)
    by_group = sorted(
        ((g, sum(x.nbytes for x in rs), sum(x.launches for x in rs)) for g, rs in groups.items()),
        key=lambda t: -t[1],
    )
    print(
        f"\n== pointwise census: b{c.b} x s{c.s}  d_model={d}  bf16={c.bf16}  "
        f"max_iter={cfg.max_iter}  (ONE iteration, forward)"
    )
    print(
        f"   reference pass = one [{n}, {d}] bf16 tensor = {ref / 1e6:.2f} MB "
        f"= {ref / bw * 1e6:.1f} us at {bw / 1e12:.2f} TB/s"
    )
    print(f"   {'group':<11} {'op':<52} {'shape':<18} {'dtype':<9} {'MB r+w':>9} {'launch':>6}")
    for g in groups:
        for r in groups[g]:
            dt = "/".join(sorted({d for _, d in r.reads} | {d for _, d in r.writes}))
            shape = "x".join(str(x) for x in r.shape)
            print(
                f"   {r.group:<11} {r.op[:52]:<52} {shape:<18} {dt:<9} "
                f"{r.nbytes / 1e6:9.2f} {r.launches:6d}"
            )
    print("\n   -- groups by traffic --")
    for g, nb, nl in by_group:
        print(
            f"   {g:<11} {nb / 1e9:7.3f} GB  {nb / total * 100:5.1f} %  "
            f"{nb / bw * 1e6:8.1f} us  {nl:3d} launches"
        )
    print(
        f"   {'TOTAL':<11} {total / 1e9:7.3f} GB  100.0 %  "
        f"{total / bw * 1e6:8.1f} us  {launches:3d} launches"
    )
    print(
        f"   {total / ref:.1f} bf16-reference passes; "
        f"{total / nbytes((n, d), 'f32'):.1f} fp32 passes"
    )
    if top:
        for r in sorted(rows, key=lambda r: -r.nbytes)[:top]:
            print(f"   top: {r.nbytes / 1e6:7.1f} MB  {r.group:<11} {r.op}")
    out = {
        "config": {
            "b": c.b,
            "s": c.s,
            "d": d,
            "bf16": c.bf16,
            "max_iter": int(cfg.max_iter),
            "fused_residual": bool(getattr(cfg, "fused_residual", False)),
            "fused_epilogue": bool(getattr(cfg, "fused_epilogue", False)),
            "fused_kda_tail": bool(getattr(cfg, "fused_kda_tail", False)),
        },
        "reference_bytes": ref,
        "total_bytes": total,
        "total_us": total / bw * 1e6,
        "launches": launches,
        "groups": [
            {"group": g, "bytes": nb, "launches": nl, "us": nb / bw * 1e6} for g, nb, nl in by_group
        ],
        "rows": [
            {
                "group": r.group,
                "op": r.op,
                "site": r.site,
                "shape": list(r.shape),
                "bytes": r.nbytes,
                "launches": r.launches,
            }
            for r in rows
        ],
    }
    return out


def _load(path: str) -> ModelConfig:
    """A flat preset OR a run snapshot (`[model]`) -> ModelConfig.

    `load_preset` resolves names against the repo's own `configs/`; a census
    of the RUN THAT IS ON THE CARD reads its saved `ckpt.config.toml`, which
    lives under `checkpoints/` and is not a preset. Both are TOML with the same
    keys, so one reader serves both and no path resolution is hidden.
    """
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"pass_census: {p} does not exist. Run snapshots live under "
            "<repo>/checkpoints/ (gitignored), so a fresh worktree has none: "
            "pass --config /path/to/<ckpt>.config.toml, or the tracked preset "
            "`--config configs/m400.toml` (same geometry, fewer arms)."
        )
    data = tomllib.loads(p.read_text())
    model = data.get("model", data)
    return ModelConfig(**{k: v for k, v in model.items()})


def _diff(a: dict, b: dict) -> None:
    print(f"\n== diff  A={Path(a['path']).name}  B={Path(b['path']).name}")
    for g in sorted({x["group"] for x in a["groups"]} | {x["group"] for x in b["groups"]}):
        ga = next((x for x in a["groups"] if x["group"] == g), {"bytes": 0, "launches": 0})
        gb = next((x for x in b["groups"] if x["group"] == g), {"bytes": 0, "launches": 0})
        db = (gb["bytes"] - ga["bytes"]) / 1e6
        dl = gb["launches"] - ga["launches"]
        if db or dl:
            print(f"   {g:<11} {db:+8.1f} MB  {dl:+3d} launches")
    print(
        f"   {'TOTAL':<11} {(b['total_bytes'] - a['total_bytes']) / 1e6:+8.1f} MB  "
        f"{b['launches'] - a['launches']:+3d} launches  "
        f"{(b['total_us'] - a['total_us']):+.1f} us"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    ap.add_argument("--bw", type=float, default=DEFAULT_BW / 1e12, help="TB/s")
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    ap.add_argument("--json", default=None)
    ap.add_argument("--diff", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args(argv)
    if args.diff:
        a = json.loads(Path(args.diff[0]).read_text())
        b = json.loads(Path(args.diff[1]).read_text())
        a["path"], b["path"] = args.diff
        _diff(a, b)
        return 0
    over: dict = {}
    for kv in args.set:
        key, _, val = kv.partition("=")
        low = val.lower()
        if low in ("1", "0", "true", "false", "yes", "no"):
            over[key] = low in ("1", "true", "yes")
        else:
            try:
                over[key] = int(val)
            except ValueError:
                try:
                    over[key] = float(val)
                except ValueError:
                    over[key] = val
    cfg = _load(args.config)
    cfg = dataclasses.replace(cfg, **over) if over else cfg
    validate(cfg)
    c, rows = census(cfg, args.batch, args.seq)
    out = report(cfg, rows, c, args.bw * 1e12, top=args.top)
    if args.json:
        out["path"] = args.json
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\n   wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
