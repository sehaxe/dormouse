"""Per-op capture probe: which of the expressions `_dispatch` runs can a graph hold?

Each op is captured ALONE (own graph), so a failure is attributed to that one op
and not to a stream that a previous op invalidated.
"""

from __future__ import annotations

import torch

N, K, E = 32, 2, 4
dev = "cuda"


def cap(name, fn):
    holder = {}

    def run():
        holder["out"] = fn()

    # one warm-up outside the graph, so any lazy init lands there
    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            run()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        print(f"  OK    {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name}: {type(e).__name__}: {str(e).splitlines()[0]}")
        torch.cuda.synchronize()
        return False


def main() -> int:
    idx = torch.randint(0, E, (N, K), device=dev)
    mask = torch.rand(N, K, device=dev) > 0.3
    order = torch.argsort(idx.reshape(-1), stable=True)
    counts = torch.zeros(E, dtype=torch.int64, device=dev)
    x = torch.randn(N, 64, device=dev)
    xg = torch.zeros(E * 4, 64, device=dev)
    big = torch.randn(E, 4, 64, device=dev)
    w = torch.randn(E, 64, 32, device=dev)

    checks = [
        ("bool-mask index  t[mask]", lambda: order[mask.reshape(-1)]),
        ("int index        t[i]", lambda: order[idx.reshape(-1)]),
        ("argsort stable", lambda: torch.argsort(idx.reshape(-1), stable=True)),
        ("argsort unstable", lambda: torch.argsort(idx.reshape(-1))),
        ("sort values", lambda: torch.sort(idx.reshape(-1)).values),
        ("cumsum", lambda: torch.cumsum(counts, 0)),
        ("arange", lambda: torch.arange(N * K, device=dev)),
        ("index_put", lambda: xg.index_put((torch.arange(4, device=dev),), x[:4])),
        ("index_add_", lambda: x.index_add(0, torch.arange(N, device=dev), x)),
        ("bmm", lambda: torch.bmm(big, w)),
        ("masked_fill", lambda: idx.masked_fill(~mask, E)),
        (
            "ones_like+scatter_add",
            lambda: torch.ones_like(counts).scatter_add_(
                0, idx.reshape(-1), torch.ones_like(idx.reshape(-1))
            ),
        ),
        ("zeros", lambda: torch.zeros(8, 8, device=dev)),
        ("unique", lambda: torch.unique(idx.reshape(-1))),
        ("nonzero", lambda: torch.nonzero(idx.reshape(-1) > 1)),
        ("tolist", lambda: idx.reshape(-1).tolist()),
        ("item", lambda: idx.sum().item()),
        ("bool(tensor)", lambda: bool(idx.sum() > 0)),
        ("python if on device", lambda: 1 if idx.max() > 0 else 0),
        ("repeat_interleave(tensor)", lambda: x.repeat_interleave(idx.reshape(-1)[:N] % 3, dim=0)),
        ("empty_like", lambda: torch.empty_like(x)),
        ("cat", lambda: torch.cat([x, x], 0)),
    ]
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    fails = []
    for name, fn in checks:
        if not cap(name, fn):
            fails.append(name)
    print(f"\n{len(fails)} non-capturable: {fails}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
