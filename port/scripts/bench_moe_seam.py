"""The MoE seam's byte arithmetic (both seams) and, on a free card, its ms/pass.

    uv run --extra gpu python scripts/bench_moe_seam.py

The seam is `loopblock.LoopBlock._moe_branch`: cast `normed` to
`SparseMoE._bank_dtype()` once on the way in, cast the MoE output back to fp32
before `* w_ffn`. Its whole claim is arithmetic, and `moe_sonic.pass_bytes`
prices it on CPU with the transport pinned by `tests/test_moe_sonic.py`:

* **fp32 seam** (`pass_bytes(..., x_bytes=4)`): the gather chain is 12·R·d —
  fp32 `index_select` copy (4), the cast's fp32 read (4), the bf16 write (2),
  the kernel's bf16 read (2);
* **bf16 seam** (`pass_bytes`, the shipped route since 2026-10-21): 4·R·d —
  the bf16 copy (2) + the kernel read (2).

At the two shapes `moe_sonic.SHAPES` prices from the presets' own fields the
move is 192 MiB/pass at `medium-b24-s1024` and 128 MiB/pass at `b1-b8-s1024`
(8·R·d each). Those are MODEL numbers. The second half of this file measures
what the bytes cost:

* `ms fwd` and `ms fwd+bwd` for the routed pass at each seam, one warm-up
  excluded, median of `REPS`. The bf16 arm's timed body INCLUDES
  `x.to(dt)` — that cast is the seam's own cost and leaving it out would price
  a mechanism nobody ships;
* peak GiB per arm.

Preflight (AGENTS §1.5): no `dormouse` process alive and the card holding
< 2 GiB. Otherwise it prints a SKIP banner and the arithmetic, and exits 0 —
**no time exists until the free-card run happens**, and a number copied out of
a busy card is not a measurement.
"""

from __future__ import annotations

import statistics
import sys
import time
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F

from dormouse.config import ModelConfig
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
PRESETS = {"medium": "medium", "b1": "b1"}


def mib(n: int) -> str:
    return f"{n / 2**20:,.0f} MiB"


def bytes_table(shape: ms.Shape) -> None:
    """Both seams' forward/cached/pass arithmetic, from the module's own model."""
    bf16 = ms.pass_bytes(shape)
    fp32 = ms.pass_bytes(shape, x_bytes=4)
    print(
        f"\n[bytes] {shape.name}: T={shape.tokens:,} R={shape.rows:,} E={shape.experts} "
        f"d={shape.d} f={shape.f} r={shape.rank} k={shape.topk}"
    )
    print(
        f"  gather chain   fp32 {mib(fp32['forward_gather_saving']):>10}  "
        f"bf16 {mib(bf16['forward_gather_saving']):>10}  "
        f"{-(fp32['forward_gather_saving'] - bf16['forward_gather_saving']) / 2**20:+,.0f} MiB"
    )
    for key in ("x_read", "xg_write", "xg_cast_read", "xg_dt_write", "xg_read"):
        print(f"  {key:<14} fp32 {mib(fp32[key]):>10}  bf16 {mib(bf16[key]):>10}")
    print(
        f"  forward total  fp32 {mib(fp32['forward_total']):>10}  "
        f"bf16 {mib(bf16['forward_total']):>10}"
    )
    # The shipped pass's own total (forward + adjoint), the denominator a step
    # is priced against. This is the routed pass, not a whole training step.
    print(
        f"  shipped pass   fp32 "
        f"{mib(fp32['forward_total'] + fp32['backward_adjoint_current']):>10}  bf16 "
        f"{mib(bf16['forward_total'] + bf16['backward_adjoint_current']):>10}  "
        f"(the routed pass, not a step)"
    )


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


def _cfg(shape: ms.Shape, preset: dict) -> ModelConfig:
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
    )


def measure(shape: ms.Shape) -> None:
    preset = tomllib.loads(
        (ROOT / "configs" / f"{PRESETS[shape.name.split('-')[0]]}.toml").read_text()
    )
    assert preset["d_model"] == shape.d and preset["d_ffn"] == shape.f, "the shape drifted"
    assert preset["n_experts"] == shape.experts and preset["moe_topk"] == shape.topk
    dropless = preset["moe_capacity"] == 0.0
    torch.manual_seed(7)
    moe = SparseMoE(_cfg(shape, preset), device=DEV)
    dt = moe._bank_dtype()
    x_fp32 = torch.randn(shape.tokens, shape.d, device=DEV)
    x_seam = x_fp32.to(dt)
    with torch.no_grad():
        probs = F.softmax(moe.router(x_seam), dim=-1)
        weight, index = probs.topk(shape.topk, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
        order, keep, _ = moe._permute(index)
    print(
        f"\n[shape] {shape.name}: rows={int(keep.sum())} "
        f"capacity={preset['moe_capacity']} ({'dropless' if dropless else 'capped'})",
        flush=True,
    )

    def fwd(x):
        with torch.no_grad():
            moe._triton_dispatch(x, weight, index, order, keep, dropless=dropless)

    def both(x):
        moe.zero_grad(set_to_none=True)
        xt = x.detach().clone().requires_grad_(True)
        out = moe._triton_dispatch(xt, weight, index, order, keep, dropless=dropless)
        out.sum().backward()

    for label, arm in (("fp32", lambda: fwd(x_fp32)), ("bf16", lambda: fwd(x_seam))):
        med, mn, peak = _timed(arm)
        print(f"  fwd     {label} seam: {med:8.2f} ms median ({mn:.2f} min) {peak:.2f} GiB")
    for label, x in (("fp32", x_fp32), ("bf16", x_seam)):
        med, mn, peak = _timed(lambda x=x: both(x))
        print(f"  fwd+bwd {label} seam: {med:8.2f} ms median ({mn:.2f} min) {peak:.2f} GiB")
    print(
        "  (the bf16 arm's timed body includes `x.to(dt)`; the fp32 arm passes "
        "fp32 into the same dispatch and pays the gather's cast instead)"
    )


def main() -> int:
    why = preflight("cuda")
    if why is not None:
        rc = skip(why)
        print("=== ARITHMETIC ONLY (no GPU call was made) ===")
        for shape in reversed(ms.SHAPES):
            bytes_table(shape)
        print(
            "\n=== The arithmetic above is a MODEL OF THE CODE, not a timing. The bf16\n"
            "=== seam removes 8·R·d of the gather chain (12·R·d -> 4·R·d); only a free\n"
            "=== card says what those bytes cost. No ms/step exists until then.\n",
            flush=True,
        )
        return rc
    for shape in reversed(ms.SHAPES):
        bytes_table(shape)
        measure(shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
