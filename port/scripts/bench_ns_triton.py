"""The bench for the fused Triton Newton–Schulz arm (`ns_kind="triton"`).

    uv run python scripts/bench_ns_triton.py          # preflight + everything
    uv run python scripts/bench_ns_triton.py cpu      # the dispatch counts only

What it answers, in the order the arm's claim is made
(`docs/research/2026-10-17-muon-variants.md` §1 row 1, §2 #1):

1. **how many dispatches each chain issues** — the classic arm's aten count is
   MEASURED with a `TorchDispatchMode` on CPU (device-independent: it is the same
   graph the GPU arm enqueues), and the triton arm's KERNEL count is measured with
   a launch proxy (`test_ns_triton._fake_kernel` — the op's `_ns_chain_kernel[(B,)]`
   call is the only launch it makes, and the proxy sees it). Both numbers are
   measured; neither is the research note's estimate.
2. **ms per chain at the b1 shapes** (GPU, only when the card is free): wall clock
   with `cuda.synchronize`, plus `torch.profiler`'s own kernel count and CUDA time.
3. **the optimizer step time on `medium`** (GPU, only when the card is free): the
   Muon group's `batched_step_from_model` with classic vs triton, on the same
   parameters and gradients, so the comparison is of two programs and not of two
   inputs.

PREFLIGHT: refuses to run anything on the card while a training run is live
(`pgrep -f "dormouse trai[n]"`) or the card holds memory — one GPU thing at a time
(AGENTS §1.5). The CPU section is not gated: it allocates nothing and measures the
part of the claim that is device-independent.

Read it with the equivalence gate (`tests/test_ns_triton.py`), never instead of it:
a tie in quality plus a fall in step time is the arm's whole claim, and this file
measures only the second half.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_muon_gram import gauss  # noqa: E402
from test_ns_triton import B1_SHAPES, _fake_kernel  # noqa: E402

from dormouse.instrument import Counters  # noqa: E402
from dormouse.train import ns_kernels as nk  # noqa: E402
from dormouse.train.optim import Muon, zeropower_via_ns  # noqa: E402

#: The medium preset (141.9M params, 90.9M in the Muon group, 12 shapes) — the
#: scale the research note's A/B protocol runs at.
MEDIUM = "medium"


def preflight() -> str | None:
    """`None` when the card may be used, else why not (AGENTS §1.5)."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        busy = subprocess.run(
            ["pgrep", "-f", "dormouse trai[n]"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a bench's precondition, not a run's
        return f"cannot check the card ({e})"
    if busy:
        return f"a training run is live (pids {busy.replace(chr(10), ' ')})"
    # `nvidia-smi --format=csv,noheader` prints `694 MiB, 3 %`: the NUMBER is the
    # first whitespace token. Splitting on the comma leaves `694 MiB` and `int()`
    # dies on it (the battery's `ValueError: invalid literal for int()`), which
    # turned a free card into a refused bench.
    mib = int(used.split()[0]) if used else -1
    if mib >= 1024:
        return f"the card already holds {mib} MiB ({used})"
    return None


def dispatch_counts(fn) -> tuple[int, int]:
    """(matmul dispatches, all aten dispatches) for one call — the classic cost."""
    counted: list[str] = []

    class _Count(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            counted.append(str(func))
            return func(*args, **(kwargs or {}))

    with _Count():
        fn()
    n_mm = sum(1 for k in counted if k.startswith(("aten.mm", "aten.bmm", "aten.matmul")))
    return n_mm, len(counted)


def launch_count(x: torch.Tensor, steps: int = 5) -> int:
    """The triton arm's KERNEL LAUNCHES for one chain — measured, no GPU needed.

    The op's only launch is `_ns_chain_kernel[(B,)](...)`; the proxy records it.
    `_check` is stubbed because the measurement is of the launch structure, and
    the tensor here is a CPU one (the launch never happens).
    """
    seen: list = []
    real_check, real_kernel = nk._check, nk._ns_chain_kernel
    nk._check = lambda t: None
    nk._ns_chain_kernel = _fake_kernel(seen)
    try:
        nk.zeropower_triton(x, steps)
    finally:
        nk._check, nk._ns_chain_kernel = real_check, real_kernel
    return len(seen)


def bench_cpu() -> None:
    """The dispatch delta — the arm's entire mechanism, measured without a card."""
    print("=== the dispatch delta (CPU, device-independent) ===")
    print(f"torch {torch.__version__} · triton {nk.TRITON_AVAILABLE} · MAX_RANK {nk.MAX_RANK}")
    for shape in B1_SHAPES:
        g = gauss(shape)
        n_mm, n_all = dispatch_counts(lambda g=g: zeropower_via_ns(g, 5, "classic"))
        if min(shape[-2:]) > nk.MAX_RANK:
            print(
                f"  {str(shape):18s} classic {n_all:4d} aten  ->  "
                f"COUNTED FALLBACK (rank > MAX_RANK)"
            )
            continue
        n_launch = launch_count(g)
        print(
            f"  {str(shape):18s} classic {n_mm:3d} matmul / {n_all:4d} aten  ->  "
            f"triton {n_launch} launch  (-{n_all - n_launch})"
        )
    print(
        "  read it: the triton column is a KERNEL count and the classic column an\n"
        "  ATEN count — one triton launch is one cuBLAS-equivalent program, while the\n"
        "  classic chain pays one dispatch per matmul and per elementwise op. The\n"
        "  b1 step is launch-bound (GPU idle 87% of a warm step), so this column is\n"
        "  the mechanism; the wall clock below is the verdict."
    )


def _median_ms(fn, reps: int = 20, warm: int = 3) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    torch.cuda.synchronize()
    return statistics.median(ts) * 1e3


def _profile(fn) -> tuple[int, float]:
    """(kernel launches, CUDA ms) from `torch.profiler` for one call."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False
    ) as prof:
        fn()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    kernels = [k for k in ka if k.self_device_time_total > 0]
    return sum(k.count for k in kernels), sum(k.self_device_time_total for k in kernels) / 1e3


def bench_gpu() -> None:
    """ms per chain + profiler kernel counts, at the b1 shapes."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")  # tf32 would make CLASSIC the loose arm
    print("\n=== ms per chain at the b1 shapes (GPU, fp32/ieee) ===")
    try:
        for shape in B1_SHAPES:
            g = gauss(shape).cuda()
            print(f"\n[{', '.join(map(str, shape))}]  r={min(shape[-2:])}")
            for arm in ("classic", "triton"):
                if arm == "triton":
                    c = Counters()
                    fn = lambda g=g, c=c: nk.zeropower_triton_or_classic(g, 5, counters=c)  # noqa: E731
                    fn()
                    if c.get("ns_triton_fallback"):
                        print(f"  triton   COUNTED FALLBACK: {c.why}")
                        continue
                else:
                    fn = lambda g=g: zeropower_via_ns(g, 5, "classic")  # noqa: E731
                ms = _median_ms(fn)
                n_k, cuda_ms = _profile(fn)
                print(f"  {arm:8s} {ms:8.3f} ms/chain  {n_k:5d} kernels  {cuda_ms:7.3f} ms CUDA")
    finally:
        torch.set_float32_matmul_precision(prev)


def muon_shapes(preset: str) -> dict[tuple[int, int], int]:
    """A preset's Muon shape groups, read off the model on `meta` (allocates nothing)."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import optim

    with torch.device("meta"):
        model = DormouseModel(load_run(preset).model)
    shapes: dict[tuple[int, int], int] = {}
    for name, p in model.named_parameters():
        if optim.is_muon_param(name, p):
            shapes[tuple(p.shape)] = shapes.get(tuple(p.shape), 0) + 1
    return shapes


def bench_opt_step(preset: str = MEDIUM, reps: int = 3, warm: int = 1) -> None:
    """The Muon group's batched step at a preset's OWN shape groups, classic vs triton."""
    from torch import nn

    shapes = muon_shapes(preset)
    n_params = sum(shapes.values())
    gib = sum(c * s[0] * s[1] * 4 for s, c in shapes.items()) / 2**30
    print(f"\n=== the Muon step, preset `{preset}` ({len(shapes)} shapes, {n_params} params) ===")
    for s, c in sorted(shapes.items(), key=lambda kv: -kv[1] * kv[0][0] * kv[0][1]):
        print(f"  {str(s):>14s} x{c:<5d} {c * s[0] * s[1] * 4 / 2**20:9.1f} MiB")
    print(f"  fp32 state per arm: {gib:.2f} GiB")
    gen = torch.Generator().manual_seed(0)
    params = []
    for shape, count in shapes.items():
        for _ in range(count):
            p = nn.Parameter((torch.randn(*shape, generator=gen) * 0.05).cuda())
            p.grad = (torch.randn(*shape, generator=gen) * 0.1).cuda()
            params.append(p)
    init = [p.detach().clone() for p in params]
    torch.cuda.reset_peak_memory_stats()
    for kind in ("classic", "triton"):
        muon = Muon(params, lr=1e-3, ns_kind=kind)
        for i, p in enumerate(params):
            p.detach().copy_(init[i])
        ms = _median_ms(lambda muon=muon: muon.batched_step_from_model(None), reps=reps, warm=warm)
        n_k, cuda_ms = _profile(lambda muon=muon: muon.batched_step_from_model(None))
        print(
            f"  {kind:8s} {ms:8.3f} ms/step  {n_k:5d} kernels  {cuda_ms:7.3f} ms CUDA  "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        )
    print(
        "  read it: this is the Muon group's step ALONE (no forward, no backward, no\n"
        "  AdamW). The b1 profile prices the whole Muon pass at 530-850 ms of a b8 s1024\n"
        "  step, 44-53% of it — the arm's claim is a fall HERE, and the BPB is the gate."
    )


if __name__ == "__main__":
    only_cpu = "cpu" in sys.argv[1:]
    bench_cpu()
    if only_cpu:
        raise SystemExit(0)
    why = preflight()
    if why is not None:
        print(f"\n=== GPU sections REFUSED: {why} ===")
        print("  one GPU thing at a time (AGENTS §1.5). Nothing was allocated. Re-run when idle.")
        print("  what would run: ms/chain + profiler kernel counts at the b1 shapes, and the")
        print(f"  Muon step time on `{MEDIUM}` — the arm's pre-registered verdict is a speed rule.")
        raise SystemExit(0)
    print("\ncard free — running the GPU sections")
    bench_gpu()
    bench_opt_step()
