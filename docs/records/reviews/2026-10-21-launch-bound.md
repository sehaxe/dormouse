# Launch-bound at the small shape: what to count, and what a full-step graph costs
**Date** 2026-10-21 · **lane** Python port (`port/`) · **card** RTX 5060 Ti 16 GB
**Status** analysis only: the profile, compile retry and shape table are written and **not run** (card busy); no number below is new.
## 1. The measured problem, with its arithmetic
| shape | B/step | ms/step | B/s | us/B | us/KiB |
|---|---|---|---|---|---|
| b8 s512 bf16 | 4 096 | 355 | **11 538** | **86.7** | 88 750 |
| b24 s1024 bf16 | 24 576 | 600 | **40 960** | **24.4** | 25 000 |
| b8 s512 ternary | 4 096 | 392 | **10 449** | **95.7** | 98 000 |

- 4 096 B / 0.355 s = **11 538 B/s**; 355 000 us / 4 096 B = **86.7 us/B**.
- 24 576 B / 0.600 s = **40 960 B/s**; 600 000 us / 24 576 B = **24.4 us/B**.
- 86.7 / 24.4 = **3.55x**. The b24 step does 6.0x the bytes for 1.69x the wall.
  The brief calls the per-byte column "us/KB"; the division is per byte, and per
  KiB is 88 750 vs 25 000 — the same 3.55x. Ternary is 392 / 355 = **+10.4%**
  over bf16; the stacked refresh took the bank to 3 calls / 768 tensors, so the
  remaining QAT launches are per-site chains.

The signature is launch-bound, not FLOP-bound: the per-step kernel count is a
property of the graph (loop iterations, experts, optimizer), and the batch does
not change it. `scripts/launch_profile.py` turns that inference into a count:
`cudaLaunchKernel` per step, the executed-kernel phase split, and the union of
kernel time against the wall of the same step. The idle share is the decision
input; top-15-by-count names the seats to attack first.

## 2. `torch.compile` retry — the legal settings, gated
The 0.20x rejection (2939 ms/step vs 573 eager) was `_banks`' per-step Python
cache key, not inductor. `_banks` and `QuantCache.refresh_stacked` now carry
`@torch.compiler.disable`; `scripts/compile_retry.py` measures `dynamic=False`
and `dynamic=True` at medium b8 s512, 20 steps each, first step (compile) and
steady median, peak GiB, and dynamo's `graph_break` / recompile counters. Gate:
a graph break is legal; **>20 captured graphs or any `recompile_limit` fallback
is ILLEGAL** (exit 1). torch 2.13 has no `recompile` counter —
`stats["unique_graphs"]` is the proxy, printed beside the real keys.

## 3. Full-step CUDA graph — what blocks it, stated honestly
A whole-step capture needs static shapes at every launch and no host read
between them. Two independent blockers:
1. **Dropless MoE is ragged by construction.** `moe_capacity=0` keeps all
   `b*s*k = 4096*2 = 8 192` assignments; per-expert counts depend on the router's
   output, so `offs` changes value every step. The grouped-GEMM grid, the gather
   and the combine read it, and a graph records shapes/pointers, not values that
   select a different launch. A dropless step has no legal capture.
2. **The step still reads the host.** `float(loss)` is a D2H per step and the
   finite/NaN branch is a Python `if`; the Muon path moves Python state per step
   (counters, cache invalidation). A capture would freeze those decisions.

The legal capture is the **padded arm with a static cap**, and its price is
routing quality, not just padding work:
- medium b8 s512, E=256, k=2, capacity 2.0: cap = max(2, ceil(2.0·4096·2/256))
  = **64** per expert; padded rows = 256·64 = 16 384 vs 8 192 real = 2.0x.
- b1 step 1, E=1024, 6144 tokens, k=2: cap = ceil(2.0·6144·2/1024) = **24**;
  the recorded line dropped **38 026 of 49 152** assignments = **77.4%** — a
  skewed init router, not a capacity arithmetic error.
- cap 64 + LFB at the same shape: **3.8% dropped** (this task's measurement; it
  is not yet in a committed log, and the experiment re-measures it).

Trade, explicitly: the graph is legal only on the padded route; the padded route
is quality-legal only with LFB and a cap high enough that the drop is small; and
even then it is not dropless. **A 2-3x step-time win does not settle a 3.8%
assignment loss** — only a BPB A/B at the same window does.

## 4. The experiment this analysis proposes (not implemented)
Same data, same 81 920 B window (20×8×512), 3 seeds, 2000 steps, medium b8 s512:
**A** dropless + eager (control), **B** padded cap 64 + LFB + eager, **C** padded
cap 64 + LFB + full-step graph. B exists so C's ms/step delta is the graph's, not
the padding's. Report BPB mean/spread and median ms/step. Pre-registered
expectation: if the profile shows the GPU busy share at X%, C's ceiling is 1/X on
the launch gaps; with the optimizer's real GPU work unchanged, expect **1.5-2.5x**,
while A and B stay in the 11-12k B/s band. C is a win only if its BPB is within
A's spread; a larger gap loses regardless of speed.

## 5. Prerequisites for the full-step graph, in landing order
1. **Static cap** — padded dispatch, fixed `[E, cap, ·]`, no ragged `offs`.
2. **Device-side NaN firewall and counters out of the hot path** — a sticky
   device flag read at the log cadence replaces `float(loss)` + the Python
   branch; MoE counters are already device-side, so audit the remaining
   `.item()`/`.tolist()` calls and keep `drain()` at the log cadence only.
3. **Stable buffers and a capture-safe step** — `zero_grad(set_to_none=False)`
   and the optimizer graph's device-tensor LR (both already shipped); the
   per-step `invalidate_banks_all` / `invalidate_qat_cache_all` stay outside the
   capture, because captured tensors must not move address.
4. **Warmup, then capture the whole step** — 5 eager steps so optimizer state,
   cuBLAS workspaces and allocator blocks exist; then fwd + bwd + clip +
   optimizer as one graph, replayed per step, with `torch.compile` refused
   alongside (a recompile aborts a capture).
5. **Measure** — replay vs eager on fixed weights, bit-identity on one batch,
   and peak VRAM including the capture pool.

Not done here: the graph, the NaN firewall, and any run. The three scripts are the instruments the decision needs on a free card.
