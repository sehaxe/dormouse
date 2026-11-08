"""Static census of the PRECISION of every activation in ONE loop iteration.

The sibling `pass_census.py` prices the elementwise OPS. This one prices the
TENSORS: for each `[b·s, d_model]`-sized tensor the loop materialises, it
prints the shape, the dtype it is stored in today, the bytes ONE pass over it
costs, and — the column the change rides on — whether fp32 is **required** by
a downstream op (a reduction, a softmax/logsumexp, a state update, a kernel
that is fp32-only), or whether the tensor is merely *carried* in fp32.

The unit is the owner's: one `[8192, 1536]` tensor is 25.2 MB stored bf16 and
50.3 MB stored fp32. The census exists so the `residual_dtype="bf16"` change
list is derived from the code, not guessed: a tensor that lands in the
`required` column is a tensor that stays fp32.

    uv run python scripts/precision_census.py --config configs/m400.toml
    uv run python scripts/precision_census.py --config /path/<run>.config.toml \\
        --set residual_dtype=bf16
    uv run python scripts/precision_census.py --json /tmp/before.json
    uv run python scripts/precision_census.py --diff /tmp/before.json /tmp/after.json

WHAT IS COUNTED, and what is not:

* Every ACTIVATION tensor one iteration stores on the forward path, including
  the transients (`xn`, the eight hyper-network outputs, the softmax weights).
  `bytes` is ONE pass: the tensor's own elements × itemsize, not read+write.
* The state-carrying tensors the task names (`kda.S`, the CSA ring and its
  accumulator, the Sinkhorn logits) are rows too — they are how the census
  says WHY each stays fp32.
* Optimizer masters are a NOTE, not a row: they are parameter state, not
  iteration traffic, and their count needs the instantiated parameter tree
  (the census never imports torch). `--state` prints the list.

WHY THIS IS STATIC. The census must run beside a busy card (the multi-day run
owns it); a config is arithmetic. It also means the per-iteration total can be
asserted in `tests/test_residual_bf16.py` against a RECOMPUTED `shape·itemsize`
sum, so a table edit that does not match the arithmetic goes red.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = str(ROOT / "checkpoints/m400b/m400b.config.toml")
DEFAULT_BATCH, DEFAULT_SEQ = 8, 1024

#: Element size per dtype name. Keep in one place: every byte in this script is
#: `prod(shape) * ITEMSIZE[dtype]`, and the test recomputes exactly that.
ITEMSIZE = {"f32": 4, "bf16": 2, "f16": 2, "int64": 8, "int32": 4}


def nbytes(shape: tuple[int, ...], dtype: str) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * ITEMSIZE[dtype]


@dataclass(frozen=True)
class Tensor:
    """One stored tensor: what it is, where it lives, and whether fp32 is needed."""

    group: str
    name: str
    site: str
    shape: tuple[int, ...]
    dtype: str
    required: bool
    why: str
    #: The dtype under `residual_dtype="bf16"` when it is NOT the default
    #: policy (`required` ? keep : bf16). A tensor that is not residual
    #: storage — the MoE's bank dtype, the KDA kernel's q/k/v — keeps its
    #: dtype: the residual flag has no authority over the compute arm.
    after_dtype: str | None = None

    @property
    def elements(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n

    @property
    def nbytes(self) -> int:
        return nbytes(self.shape, self.dtype)

    @property
    def after(self) -> str:
        """The dtype under `residual_dtype="bf16"`.

        An explicit `after_dtype` wins (the compute arm keeps its own dtype);
        otherwise fp32 where the downstream op requires it, bf16 elsewhere.
        """
        if self.after_dtype is not None:
            return self.after_dtype
        if self.required:
            return self.dtype
        return "bf16" if self.dtype == "f32" else self.dtype


@dataclass(frozen=True)
class Ctx:
    b: int
    s: int
    d: int
    h: int
    hd: int
    kv: int
    f: int
    vocab: int
    k: int
    orders: int
    engram_dim: int
    chunk: int
    kv_csa: int
    bf16: bool

    @property
    def n(self) -> int:
        return self.b * self.s


def ctx_for(cfg, b: int, s: int) -> Ctx:
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
        kv_csa=int(cfg.csa_kv_dim),
        bf16=bool(cfg.bf16),
    )


def _row(
    group: str,
    name: str,
    site: str,
    shape: tuple[int, ...],
    dtype: str,
    required: bool,
    why: str,
    after: str | None = None,
) -> Tensor:
    return Tensor(group, name, site, shape, dtype, required, why, after)


def tensors(cfg, b: int, s: int) -> list[Tensor]:
    """The iteration's tensors in execution order, from the code.

    Shape and dtype are properties of the code path (the config only chooses
    which arm runs); `site` is the function that materialises the tensor.
    """
    c = ctx_for(cfg, b, s)
    n, d, k = c.n, c.d, c.k
    p = (n, d)
    kd = (n, k, d)
    compute = "bf16" if c.bf16 else "f32"
    paper = bool(cfg.layerrope and cfg.layerrope_form == "paper")
    rows: list[Tensor] = []

    # ---- the residual stream ------------------------------------------------
    rows.append(
        _row(
            "residual",
            "h",
            "loopblock._step",
            p,
            "f32",
            False,
            "storage only: the update is written by the residual operator and every "
            "consumer widens its own arm",
        )
    )
    if cfg.use_mhc:
        rows.append(
            _row(
                "residual",
                "streams (K lanes)",
                "loopblock._window",
                kd,
                "f32",
                False,
                "storage only: mHC widens the lanes for H_pre/H_res and writes back",
            )
        )
        rows.append(
            _row(
                "residual",
                "h0_flat (controller input)",
                "loopblock._step",
                p,
                "f32",
                False,
                "controller GEMM input; mm_mixed casts to the compute dtype",
            )
        )
    rows.append(
        _row(
            "residual",
            "ctx = H_pre x_l (or h + e_i)",
            "loopblock._step",
            p,
            "f32",
            False,
            "pointwise mix output; the norm is the next reduction",
        )
    )
    rows.append(
        _row(
            "residual",
            "normed (body input)",
            "loopblock._step",
            p,
            "f32",
            False,
            "no reduction on the STORED tensor: the RMSNorm mean is fp32 and every "
            "consumer that needs fp32 widens at its own seam",
        )
    )
    if paper:
        rows.append(
            _row(
                "residual",
                "normed_mlp (paper second read)",
                "loopblock._step",
                p,
                "f32",
                False,
                "same as normed: consumed by the MoE, which already casts to bank dtype",
            )
        )
    rows.append(
        _row(
            "residual",
            "ctrl = cat([ctx, h0])",
            "loopblock._step",
            (n, 2 * d),
            "f32",
            False,
            "GEMM input; mm_mixed casts it to the compute dtype",
        )
    )
    for gate in ("w_attn", "w_mem", "w_ffn"):
        rows.append(
            _row(
                "residual",
                gate,
                "loopblock._step",
                (n, 1),
                "f32",
                False,
                "sigmoid output stored as a per-row scale; fp32 sigmoid, bf16 store",
            )
        )
    rows.append(
        _row(
            "residual",
            "attn (KDA/CSA write)",
            "loopblock._step",
            p,
            "f32",
            False,
            "branch output; the fp32 reductions are inside the attention arm",
        )
    )
    rows.append(
        _row(
            "residual",
            "branch (memory floor mix)",
            "loopblock._step",
            p,
            "f32",
            False,
            "pointwise convex mix of two branch outputs",
        )
    )
    rows.append(
        _row(
            "residual",
            "ffn (MoE write)",
            "loopblock._step_post",
            p,
            "f32",
            False,
            "pointwise gate on the MoE output",
        )
    )
    rows.append(
        _row(
            "residual",
            "y = attn + branch + ffn",
            "loopblock._step_post",
            p,
            "f32",
            False,
            "the residual update; the loss reads the fp32 readout, not this tensor",
        )
    )
    rows.append(
        _row(
            "residual",
            "step_out (out_proj)",
            "loopblock._step",
            p,
            "f32",
            False,
            "readout storage; the per-iteration CE widens it to fp32 for the head",
        )
    )
    rows.append(
        _row(
            "readout",
            "out_acc (Σ readout)",
            "loopblock._window",
            p,
            "f32",
            True,
            "ACCUMULATION over T iterations; the final norm + head are the loss path",
        )
    )
    rows.append(
        _row(
            "readout",
            "hn (final RMSNorm)",
            "model._forward_byte_lm",
            p,
            "f32",
            True,
            "the final norm and head must be fp32: bf16 logits NaN on this stack (AGENTS §2.2)",
        )
    )
    rows.append(
        _row(
            "readout",
            "logits",
            "model._forward_byte_lm",
            (n, c.vocab),
            "f32",
            True,
            "log_softmax / CE: the reduction's accumulator and its output are fp32",
        )
    )
    rows.append(
        _row(
            "readout",
            "ce (loss term)",
            "loopblock._iter_ce",
            (),
            "f32",
            True,
            "the scalar the optimizer differentiates; fp32",
        )
    )

    # ---- mHC: the lane norm, the hyper-network, the Sinkhorn ----------------
    if cfg.use_mhc:
        rows.append(
            _row(
                "mhc",
                "xn (lane RMSNorm)",
                "mhc._x_norm",
                (n, k * d),
                "f32",
                True,
                "mean(x²) reduction; the lane norm is the hyper-network's input",
            )
        )
        rows.append(
            _row(
                "mhc",
                "h_pre",
                "mhc.pre_mix",
                (n, k),
                "f32",
                True,
                "sigmoid gate that weights the lane mix (a reduction over lanes)",
            )
        )
        rows.append(
            _row(
                "mhc",
                "h_post",
                "mhc.forward",
                (n, k),
                "f32",
                True,
                "2σ gate that deposits the branch into each lane",
            )
        )
        rows.append(
            _row(
                "mhc",
                "h_res logits",
                "mhc._post_res_tilde",
                (n, k, k),
                "f32",
                True,
                "Sinkhorn-Knopp runs in the LOG domain (logsumexp × 40); fp32",
            )
        )
        rows.append(
            _row(
                "mhc",
                "h_res probabilities",
                "mhc.sinkhorn_knopp",
                (n, k, k),
                "f32",
                True,
                "exp() of the log-domain projection; the lane mix's coefficients",
            )
        )
        rows.append(
            _row(
                "mhc",
                "mixed lanes (H_res x_l)",
                "mhc.forward",
                kd,
                "f32",
                True,
                "reduction over K lanes against the fp32 Sinkhorn matrix",
            )
        )
    if cfg.use_attnres and not (cfg.use_mhc):
        s_src = int(cfg.max_iter) + 1
        rows.append(
            _row(
                "attnres",
                "keys (RMSNorm sources)",
                "loopblock.depth_attend",
                (n, s_src, d),
                "f32",
                True,
                "keys for the softmax score; the norm is a reduction",
            )
        )
        rows.append(
            _row(
                "attnres",
                "weights (softmax)",
                "loopblock.depth_attend",
                (n, s_src),
                "f32",
                True,
                "softmax over sources; the mixture's coefficients",
            )
        )

    # ---- KDA ----------------------------------------------------------------
    if cfg.use_kda and cfg.attn_kind == "kda":
        rows.append(
            _row(
                "kda",
                "q / k / v (projection outputs)",
                "kda._fla_path",
                p,
                compute,
                False,
                "the FLA kernel accepts the compute dtype; under bf16=true these are "
                "already bf16 storage — the residual flag does not own them",
                after=compute,
            )
        )
        rows.append(
            _row(
                "kda",
                "beta (write strength)",
                "kda.projections",
                (n, c.h),
                compute,
                False,
                "sigmoid output; the kernel accepts the compute dtype",
                after=compute,
            )
        )
        rows.append(
            _row(
                "kda",
                "g (log-decay)",
                "kda.projections",
                p,
                "f32",
                True,
                "log-domain decay: logsigmoid and the kernel's exp(); fp32",
            )
        )
        rows.append(
            _row(
                "state",
                "kda.S (gated-delta state)",
                "kda._fla_path",
                (c.b, c.h, c.hd, c.hd),
                "f32",
                True,
                "the recurrence's carried state; FLA allocates and asserts fp32",
            )
        )
    elif cfg.use_kda and cfg.attn_kind == "full":
        rows.append(
            _row(
                "kda",
                "q / k / v (full attention)",
                "fullattn.projections",
                p,
                compute,
                False,
                "projection outputs; the arm casts them to the compute dtype",
                after=compute,
            )
        )
        rows.append(
            _row(
                "kda",
                "attention scores",
                "fullattn._sdpa",
                (c.b, c.h, c.s, c.s),
                "f32",
                True,
                "softmax over the full causal matrix",
            )
        )

    # ---- CSA ----------------------------------------------------------------
    if cfg.use_csa:
        chunks = max(1, (c.s + c.chunk - 1) // c.chunk)
        rows.append(
            _row(
                "state",
                "csa ring k/v",
                "csa.CSAState",
                (chunks, c.b, c.kv_csa),
                "bf16",
                False,
                "the ring is ALREADY bf16 (2 B/elem, the decode state's documented dtype)",
            )
        )
        rows.append(
            _row(
                "state",
                "csa ring sum",
                "csa.CSAState",
                (c.b, c.d),
                "f32",
                True,
                "running chunk accumulator; fp32 by the state contract",
            )
        )
        rows.append(
            _row(
                "csa",
                "top-k scores",
                "csa._level_scores",
                (c.b, c.s, chunks),
                "f32",
                True,
                "softmax/top-k over the chunk scores; fp32",
            )
        )
        rows.append(
            _row(
                "csa",
                "csa_out",
                "csa.forward",
                p,
                compute,
                False,
                "weighted attention sum; the softmax is inside the arm and fp32",
            )
        )

    # ---- Engram -------------------------------------------------------------
    if cfg.use_engram:
        rows.append(
            _row(
                "engram",
                "emb (gathered rows)",
                "engram._read",
                (n, c.orders, c.engram_dim),
                "f32",
                True,
                "the tables are fp32 end to end; the Engram kernels are f32-only on "
                "the burn stack (ILLEGAL_ADDRESS on bf16)",
            )
        )
        rows.append(
            _row(
                "engram",
                "value * gate reduction",
                "engram._read",
                p,
                "f32",
                True,
                "sum over n-gram orders; the gate is sign·sqrt in fp32",
            )
        )
        rows.append(
            _row(
                "engram",
                "mem (memory read)",
                "engram._read",
                p,
                "f32",
                False,
                "storage after the reduction; the convex mix is pointwise",
            )
        )
        rows.append(
            _row(
                "engram",
                "dense (mem_dense + bias)",
                "loopblock._step",
                p,
                compute,
                False,
                "mm_mixed's compute-dtype output, already cast at the seam",
            )
        )

    # ---- MoE ----------------------------------------------------------------
    rows.append(
        _row(
            "moe",
            "moe input (normed_mlp, bank dtype)",
            "loopblock._moe_branch",
            p,
            compute,
            False,
            "the bank computes in its own dtype; a bf16 residual makes the seam cast a no-op",
            after=compute,
        )
    )
    rows.append(
        _row(
            "moe",
            "moe output",
            "loopblock._moe_branch",
            p,
            compute,
            False,
            "the bank's output dtype; the loop widens it for the residual write today",
            after=compute,
        )
    )

    # ---- the CE head --------------------------------------------------------
    rows.append(
        _row(
            "readout",
            "lm_head input",
            "tsct.TSCTLinear.forward",
            p,
            "f32",
            True,
            "the head casts to fp32 and its logits/cross-entropy are fp32",
        )
    )
    return rows


def total(rows: list[Tensor], *, after: bool = False) -> int:
    """Bytes of ONE pass over every tensor; `after` applies the bf16 policy."""
    return sum(nbytes(r.shape, r.after if after else r.dtype) for r in rows)


def state_notes(cfg) -> list[str]:
    """The state-carrying paths and the reason each stays fp32 (no bytes)."""
    notes = [
        "kda.S — [b,H,hd,hd] fp32: the gated-delta recurrence's carried state; FLA "
        "allocates it fp32 and asserts fp32 on the way back in (chunk.py:383).",
        "csa ring — bf16 storage (already), its `sum` accumulator fp32: the ring is "
        "the decode state, the sum is the running chunk total.",
        "Sinkhorn-Knopp — fp32, log domain: 40 logsumexp normalisations on [n,K,K]; "
        "a bf16 log-domain projection underflows/overflows the exponent range.",
        "CE head — fp32: bf16 logits NaN on this stack (AGENTS §2.2) and the "
        "log_softmax reduction is the loss.",
        "optimizer masters — fp32: parameter state, never cast by the forward; "
        "`--quant`/`dense_qat` choose the forward path only.",
    ]
    if not cfg.use_kda or cfg.attn_kind != "kda":
        notes = [n for n in notes if not n.startswith("kda.S")]
    if not cfg.use_csa:
        notes = [n for n in notes if not n.startswith("csa ring")]
    return notes


def report(cfg, rows: list[Tensor], c: Ctx, top: int = 0) -> dict:
    now = total(rows)
    after = total(rows, after=True)
    ref32 = nbytes((c.n, c.d), "f32")
    ref16 = nbytes((c.n, c.d), "bf16")
    print(
        f"\n== precision census: b{c.b} x s{c.s}  d_model={c.d}  bf16={c.bf16}  "
        f"max_iter={cfg.max_iter}  (ONE iteration)"
    )
    print(f"   one [n, d_model] pass = {ref32 / 1e6:.2f} MB fp32 / {ref16 / 1e6:.2f} MB bf16")
    print(
        f"   {'group':<9} {'tensor':<34} {'shape':<16} {'dtype':<6} "
        f"{'MB':>8} {'req':>4} {'after':<6} site"
    )
    groups: dict[str, list[Tensor]] = {}
    for r in rows:
        groups.setdefault(r.group, []).append(r)
    for g, rs in groups.items():
        for r in rs:
            shape = "x".join(str(x) for x in r.shape)
            print(
                f"   {g:<9} {r.name[:34]:<34} {shape:<16} {r.dtype:<6} "
                f"{r.nbytes / 1e6:8.2f} {'YES' if r.required else '-':>4} "
                f"{r.after:<6} {r.site}"
            )
    print("\n   -- REQUIRED fp32, and why --")
    for r in rows:
        if r.required:
            print(f"   {r.group:<9} {r.name:<34} {r.why}")
    print("\n   -- totals, one iteration (one pass per tensor) --")
    print(f"   today (fp32 residual)          {now / 1e9:8.3f} GB")
    print(
        f"   residual_dtype=bf16            {after / 1e9:8.3f} GB   "
        f"{(after - now) / now * 100:+.1f} %"
    )
    print(
        f"   delta                          {(after - now) / 1e6:+8.1f} MB   "
        f"{(now - after) / ref16:5.1f} bf16 passes saved"
    )
    n_req = sum(1 for r in rows if r.required)
    print(f"   {len(rows)} tensors, {n_req} fp32-required, {len(rows) - n_req} carried/free")
    by_group: dict[str, list[int]] = {}
    for r in rows:
        g = by_group.setdefault(r.group, [0, 0])
        g[0] += r.nbytes
        g[1] += nbytes(r.shape, r.after)
    print("\n   -- by group (the stream halves; the reductions/state do not) --")
    for g, (a, b) in sorted(by_group.items(), key=lambda kv: -kv[1][0]):
        print(f"   {g:<10} {a / 1e9:7.3f} -> {b / 1e9:7.3f} GB  {(b - a) / a * 100:+6.1f} %")
    print(
        f"   {'TOTAL':<10} {now / 1e9:7.3f} -> {after / 1e9:7.3f} GB  "
        f"{(after - now) / now * 100:+6.1f} %"
    )
    print("\n   -- state, kept fp32 (no bytes counted) --")
    for note in state_notes(cfg):
        print(f"   * {note}")
    if top:
        for r in sorted(rows, key=lambda x: -x.nbytes)[:top]:
            print(f"   top: {r.nbytes / 1e6:7.1f} MB  {r.group:<9} {r.name}")
    return {
        "config": {"b": c.b, "s": c.s, "d": c.d, "max_iter": int(cfg.max_iter)},
        "total_bytes": now,
        "after_bytes": after,
        "reference_fp32": ref32,
        "reference_bf16": ref16,
        "rows": [
            {
                "group": r.group,
                "name": r.name,
                "site": r.site,
                "shape": list(r.shape),
                "dtype": r.dtype,
                "after": r.after,
                "required": r.required,
                "bytes": r.nbytes,
                "why": r.why,
            }
            for r in rows
        ],
    }


def _load(path: str):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"precision_census: {p} does not exist. Run snapshots live under "
            "<repo>/checkpoints/ (gitignored), so a fresh worktree has none: pass "
            "--config /path/to/<ckpt>.config.toml, or the tracked preset "
            "`--config configs/m400.toml` (same geometry, fewer arms)."
        )
    data = tomllib.loads(p.read_text())
    from dormouse.config import ModelConfig

    model = data.get("model", data)
    return ModelConfig(**{k: v for k, v in model.items()})


def _diff(a: dict, b: dict) -> None:
    print(f"\n== diff  A={Path(a['path']).name}  B={Path(b['path']).name}")
    amap = {(r["group"], r["name"]): r for r in a["rows"]}
    bmap = {(r["group"], r["name"]): r for r in b["rows"]}
    for key in sorted(set(amap) | set(bmap)):
        ra, rb = amap.get(key), bmap.get(key)
        ba = ra["bytes"] if ra else 0
        bb = rb["bytes"] if rb else 0
        if ba != bb or (ra or {}).get("dtype") != (rb or {}).get("dtype"):
            print(
                f"   {key[0]:<9} {key[1]:<34} "
                f"{(bb - ba) / 1e6:+8.2f} MB  "
                f"{ra['dtype'] if ra else '-':<6} -> {rb['dtype'] if rb else '-':<6}"
            )
    print(f"   {'TOTAL':<9} {'':<34} {(b['total_bytes'] - a['total_bytes']) / 1e6:+8.2f} MB")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--seq", type=int, default=DEFAULT_SEQ)
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
    import dataclasses

    cfg = dataclasses.replace(_load(args.config), **over) if over else _load(args.config)
    c = ctx_for(cfg, args.batch, args.seq)
    rows = tensors(cfg, args.batch, args.seq)
    out = report(cfg, rows, c, top=args.top)
    out["config"]["residual_dtype"] = getattr(cfg, "residual_dtype", "fp32")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\n   wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
