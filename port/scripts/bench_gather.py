"""The gather arm (`moe_gather = "kernel"`) against the shipped triton pass.

    uv run --extra gpu python scripts/bench_gather.py

WRITTEN 2026-10-21 AND **NOT RUN** (the card was held). No speed number in this
file, in `model/moe_sonic.py` or in a report is a measurement until this prints
one: the module's own estimate is `<=1% of the step, most likely <0.5%`, and it
is an estimate.

What it asks, per shape (`b1-b8-s1024`, `medium-b24-s1024` — the two shapes
`moe_sonic.SHAPES` prices from the presets' own fields):

* **bytes moved**, from the two dispatches' own layout, not from the paper's:
  `moe_sonic.wired_pass_bytes` is asserted line by line in
  `tests/test_moe_sonic.py`, so a stale price fails the CPU suite. Three
  quantities per arm: the FORWARD, what the forward leaves CACHED for the
  backward, and the whole routed PASS (forward + adjoint);
* **ms fwd** — one `no_grad` pass. This is the arm's real claim (the kernel
  reads `x`'s rows through the map instead of a materialised `[R, d]` copy);
* **ms fwd+bwd** — the pass a TRAINING step runs. For the gather arm that
  contains the recompute `_GatherGlu` makes (`_gather_glu_kernel` saves no
  `gpre`/`upre`/`h`, so the shipped grouped adjoint is re-run on the same rows):
  **two forwards per backward**. The byte table says so; this column prices it;
* **peak GiB** per arm (`max_memory_allocated` after a reset), which is where the
  dropped `xg`/gate caches should show up.

Parity is printed too (forward and `dx`, gather vs triton, rel): a bench that
measures a wrong function is worse than no bench.

Preflight (AGENTS §1.5): no `dormouse` process alive and the card holding
< 2 GiB. Otherwise it prints a SKIP banner naming what it found and exits 0 —
nothing is measured, so there is no number to quote. The byte arithmetic is NOT
a measurement and is printed under its own banner so it cannot be mistaken for
one.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F

from dormouse.config import ModelConfig
from dormouse.instrument import Counters
from dormouse.model import moe_sonic as ms
from dormouse.model.experts import SparseMoE

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

DEV = "cuda"
ROOT = Path(__file__).resolve().parents[1]
REPS = 5
#: The preset each `SHAPES` entry is read back from (the same contract
#: `tests/test_moe_sonic.py::test_the_shape_table_is_the_configs_own` pins).
PRESETS = {"medium": "medium", "b1": "b1"}


def mib(n: int) -> str:
    return f"{n / 2**20:,.0f} MiB"


def bytes_table(shape: ms.Shape) -> dict[str, int]:
    """Print one shape's arithmetic and return it — no device call inside."""
    w = ms.wired_pass_bytes(shape)  # the asserts live in the module
    b = ms.pass_bytes(shape)
    print(
        f"\n[bytes] {shape.name}: T={shape.tokens:,} R={shape.rows:,} "
        f"E={shape.experts} d={shape.d} f={shape.f} r={shape.rank} k={shape.topk}"
    )
    for label, shipped, gather in (
        ("forward", w["shipped_forward"], w["gather_forward"]),
        ("cached ", w["shipped_cache"], w["gather_cache"]),
        ("pass   ", w["shipped_pass"], w["gather_pass"]),
    ):
        delta = gather - shipped
        print(
            f"  {label}  triton {mib(shipped):>10}  gather {mib(gather):>10}  "
            f"{delta / 2**20:+9,.0f} MiB  ({100.0 * delta / shipped:+.1f}%)"
        )
    print(
        f"  the gather forward removes {mib(b['forward_gather_saving'])} of xg traffic and "
        f"{mib(b['gates_write'])} of gates; its backward adds a RECOMPUTE of "
        f"{mib(w['recompute'])} (the shipped adjoint is re-run on the same rows), and at "
        f"k={shape.topk} the saved fp32 x ({mib(b['x_read'] // shape.topk)}) is the SAME "
        f"bytes as the bf16 xg it replaces ({mib(b['xg_dt_write'])})"
    )
    return w


def _timed(fn, reps: int = REPS) -> tuple[float, float, float]:
    """`(median ms, min ms, peak GiB)` over `reps` calls, one warm-up excluded."""
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return (
        statistics.median(times) * 1e3,
        min(times) * 1e3,
        torch.cuda.max_memory_allocated() / 2**30,
    )


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float(), b.detach().float()
    scale = float(b.abs().max())
    return float((a - b).abs().max()) / scale if scale > 0 else 0.0


def _cfg(shape: ms.Shape, preset: dict, *, gather: bool) -> ModelConfig:
    """The preset's own geometry, field for field, with the arm switched."""
    return ModelConfig(
        d_model=preset["d_model"],
        n_heads=4,
        head_dim=preset["d_model"] // 4,
        d_ffn=preset["d_ffn"],
        rank=preset["rank"],
        n_experts=preset["n_experts"],
        moe_topk=preset["moe_topk"],
        moe_capacity=preset["moe_capacity"],
        shared_rank=preset.get("shared_rank", 192),
        bf16=bool(preset["bf16"]),
        moe_kernel="triton",
        moe_gather="kernel" if gather else "off",
    )


def measure(shape: ms.Shape) -> None:
    preset = tomllib.loads(
        (ROOT / "configs" / f"{PRESETS[shape.name.split('-')[0]]}.toml").read_text()
    )
    assert preset["d_model"] == shape.d and preset["d_ffn"] == shape.f, "the shape drifted"
    assert preset["n_experts"] == shape.experts and preset["moe_topk"] == shape.topk
    assert preset["rank"] == shape.rank, "the shape drifted from the preset"
    dropless = preset["moe_capacity"] == 0.0
    run = _cfg(shape, preset, gather=False)
    arm = _cfg(shape, preset, gather=True)
    torch.manual_seed(7)
    moe = SparseMoE(run, device=DEV)
    moe_gather = SparseMoE(arm, device=DEV)
    moe_gather.load_state_dict(moe.state_dict())
    x = torch.randn(shape.tokens, shape.d, device=DEV)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(shape.topk, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
        order, keep, dropped = moe._permute(index)
    rows = int(keep.sum())
    print(
        f"\n[shape] {shape.name}: E={shape.experts} d={shape.d} f={shape.f} "
        f"r={shape.rank} tokens={shape.tokens} rows={rows} dropped={int(dropped)} "
        f"capacity={preset['moe_capacity']} ({'dropless' if dropless else 'capped'})",
        flush=True,
    )

    def fwd_t():
        with torch.no_grad():
            moe._triton_dispatch(x, weight, index, order, keep, dropless=dropless)

    def fwd_g():
        with torch.no_grad():
            moe_gather._gather_dispatch(x, weight, index, order, keep, Counters())

    xt = x.detach().clone().requires_grad_(True)
    xg = x.detach().clone().requires_grad_(True)

    def both_t():
        moe.zero_grad(set_to_none=True)
        xt.grad = None
        out = moe._triton_dispatch(xt, weight, index, order, keep, dropless=dropless)
        out.sum().backward()

    def both_g():
        moe_gather.zero_grad(set_to_none=True)
        xg.grad = None
        out = moe_gather._gather_dispatch(xg, weight, index, order, keep, Counters())
        out.sum().backward()

    with torch.no_grad():
        ref = moe._triton_dispatch(x, weight, index, order, keep, dropless=dropless)
        got = moe_gather._gather_dispatch(x, weight, index, order, keep, Counters())
    rel_fwd = _rel(got, ref)
    del ref, got
    xg.grad = None
    xt.grad = None
    moe.zero_grad(set_to_none=True)
    moe._triton_dispatch(xt, weight, index, order, keep, dropless=dropless).sum().backward()
    moe_gather.zero_grad(set_to_none=True)
    moe_gather._gather_dispatch(xg, weight, index, order, keep, Counters()).sum().backward()
    rel_dx = _rel(xg.grad, xt.grad)
    rel_bank = max(
        _rel(b.grad, a.grad)
        for a, b in zip(moe.experts, moe_gather.experts, strict=True)
        if a.gate_u.grad is not None and b.gate_u.grad is not None
    )
    moe.zero_grad(set_to_none=True)
    moe_gather.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    print(
        f"[parity] gather vs triton: fwd={rel_fwd:.2e} dx={rel_dx:.2e} gate_u grads={rel_bank:.2e}",
        flush=True,
    )

    arms = {
        "triton fwd": fwd_t,
        "gather fwd": fwd_g,
        "triton fwd+bwd": both_t,
        "gather fwd+bwd": both_g,
    }
    acc: dict[str, list] = {k: [] for k in arms}
    # Interleaved: this card's dispatch timings drift a few ms between runs of
    # one binary, so one-arm-then-the-other attributes the drift to the arm.
    for _ in range(REPS):
        for name, fn in arms.items():
            acc[name].append(_timed(fn, reps=1))
    print(f"  {'arm':<18} {'median':>9} {'min':>9} {'peak':>8}", flush=True)
    for name, runs in acc.items():
        med = statistics.median(r[0] for r in runs)
        low = min(r[1] for r in runs)
        peak = max(r[2] for r in runs)
        print(f"  {name:<18} {med:8.2f}ms {low:8.2f}ms {peak:7.3f}G", flush=True)


def main() -> int:
    busy = subprocess.run(
        ["pgrep", "-af", "dormouse (trai[n]|bench)|[/]target/release/train"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    why = preflight("cuda") or (f"a dormouse process is alive:\n{busy}" if busy else None)
    if why is not None:
        rc = skip(why)
        print("=== ARITHMETIC ONLY (no GPU call was made) ===")
        for shape in reversed(ms.SHAPES):
            bytes_table(shape)
        print(
            "\n=== The byte arithmetic above is a MODEL OF THE CODE, not a timing. The\n"
            "=== forward is a byte win; the pass is a byte LOSS because the backward\n"
            "=== recomputes the grouped forward. Only the run says what those bytes cost.\n",
            flush=True,
        )
        return rc
    for shape in reversed(ms.SHAPES):  # b1 first: it is the shape b1 runs
        bytes_table(shape)
        measure(shape)
    print(
        "\n=== Read the two timing columns, not the bytes alone: `gather fwd` is the\n"
        "=== kernel's claim and `gather fwd+bwd` is what a step pays while the arm\n"
        "=== recomputes its forward for the shipped adjoint. A win in the first and a\n"
        "=== loss in the second means the rung is a gather-fused ADJOINT (it needs\n"
        "=== gpre/upre/h, which _gather_glu_kernel does not emit), not this wiring.\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
