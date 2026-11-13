# Long-context attention: exact state vs compressed sparse, at equal budget

Written 2026-11-11 while the three arms run. This note exists because the first
draft of the long-context plan compared KDA against **quadratic SDPA** — and our
product does not use SDPA at long context. It uses CSA (compressed sparse
attention, the DeepSeek-V4 lineage). That mistake would have produced a flattering
and wrong "2.2× for KDA" headline.

## The cost of attention at 8192 bytes (d=890, MAC per byte)

| mechanism | cost | vs SDPA | what it cannot do |
|---|---:|---:|---|
| SDPA (quadratic) | **29.2M** | 1× | scale past a few k |
| **CSA** (chunk 8, kv 64, topk 32) | **~0.08M** | **346× cheaper** | *retrieve reliably* — measured, see below |
| KDA (linear recurrent) | 1.58M | 18× cheaper | nothing structural; it is exact state |

CSA's figure is attention-side + projection: `mac_per_byte(8192, kv=64, chunk=8,
topk=32)` = 0.07M, plus `2·d·kv/chunk` = 0.014M per byte of K/V projection.

**So KDA is 20× *more expensive* than CSA at 8k.** The speed argument for KDA
evaporates the moment the baseline is our actual mechanism rather than SDPA.

## What KDA still has, and why it is the axis that matters

CSA compresses every 8 bytes into one KV and lets each query read only the top 32
chunks. That is where its 346× comes from — and it is exactly where it broke for
us: in the first long-context gate the selector ranked a needle at **chance**
(119/128, with the needle in the first ~50 bytes). The mechanism is cheap because
it *chooses*, and the choice failed.

KDA has no choice to fail: its state is an exact linear recurrence. So the honest
comparison at long context is not "which is faster" but **"how much quality does
the 346× cost"**, and that is measurable end-to-end.

## The three arms now running

| arm | attention | params | what it establishes |
|---|---|---:|---|
| `arch_dense890_8k` | quadratic SDPA | 109.9M | the reference (not a product candidate) |
| `arch_hybrid31_8k` | 3 KDA : 1 full | 109.9M | the mix, at the length where a mix is supposed to pay |
| `arch_csa8k` | CSA, 12 loop blocks | 120.9M | our existing mechanism, its real cost |

Protocol: 60 MB, b4 (T = 32,768 bytes/step — activation-comparable to b24×1024),
8192-byte windows, 2 seeds, one GPU lock. Reported: **bytes/s** (not FLOPs — the
4-bit work already showed FLOPs do not convert) and BPB on the same eval window.

## What would make KDA win

If CSA's BPB at 8k is worse than the 3:1 mix's by more than the mix's throughput
handicap, then the 346× was buying a broken mechanism and KDA is the right answer.
If CSA holds quality, KDA has no case at long context and belongs only where its
exact state is required (streaming decode, O(1) per byte).

Either way the answer is a number from these three arms, not an argument from the
cost table.
