# The cheapest smart architecture — cost is arithmetic, not taste

**Date:** 2026-11-07. Every number computed from the recipe's own shape; the
formula is `FLOPs/byte = 6·N · passes / (bytes per position) · active fraction`.

## What our current recipe actually costs

| recipe | MFLOP/byte | vs ours | kB/s at 16 TFLOP/s | hours per GB |
|---|---:|---:|---:|---:|
| **ours: 1 byte/pos, 3 iterations** | **1710** | 1.00× | 9.4 | 29 688 |
| dense, 1 byte/pos, 1 pass | 570 | 3.00× cheaper | 28.1 | 9 896 |
| patch 8, 1 pass | 71 | 24× cheaper | 225 | 1 237 |
| patch 8 + MoD 0.63 | 45 | 38× cheaper | 356 | 779 |
| patch 8 + MoD + MoE 40% active | 18 | 95× cheaper | 891 | 312 |

**We are the most expensive recipe in the table.** The cause is one line of the
config: `max_iter=3`. It triples the cost of every byte for a capacity gain the
literature prices at φ=0.46 per recurrence — `3^0.46 = 1.66` blocks' worth of
capacity for 3× the compute, i.e. **45% of the spend buys nothing**.

This is the answer to "the architecture doesn't look cheap": it isn't. It was
built when the hypothesis was "depth without parameters"; the measured φ killed
that hypothesis and the config did not follow.

## The cheap recipe, in order of what the evidence supports

1. **Single pass** (`max_iter=1`) with unique layers — 3× cheaper, and the
   2026 depth-width law says our width wants ~14 layers (`D*∝C^0.12`).
2. **Patch unit at 8 bytes** — 8× fewer positions per byte; measured 7.4×
   wall-clock at b=32 on this card, 3.8× less VRAM. Quality at equal bytes is
   being measured now (E1).
3. **MoD at capacity 0.5** — 0.63× FLOPs, published at 60M-3B (2404.02258),
   merged behind a flag, A/B not run.
4. **Fine-grained MoE with a low active fraction** — the only mechanism that
   cuts FLOPs per unit of capacity with equal-quality evidence (2B+: 0.175×),
   but at 22 GB of data the published evidence does not reach our scale.

Items 1-3 are supported at or near our scale. Item 4 is a bet whose closest
measurement is 13.7× above us.

## What this means for the 1T goal

Cost of a 1T run = `6 · N_active · D`. The architecture does not change the
formula; it changes the constants:

| design | N_active | tokens | GPU-hours | $ at $2/h |
|---|---:|---:|---:|---:|
| 1T dense | 1000B | 20T | 208M | $416M |
| 1T MoE 5% active | 50B | 20T | 10.4M | $20.8M |
| 1T MoE 2% active | 20B | 20T | 4.2M | $8.3M |
| + patch 8 (÷8 per byte) | 20B | 2.5T-position-equivalent | 0.52M | **$1.0M** |
| + MoD 0.63 | 20B | | 0.33M | **$0.66M** |

So the honest claim we can defend: **the same 1T-quality run is ~12× cheaper with
the patch unit and MoD than without them, if (and only if) the quality per byte
holds.** That "if" is E1, and it is the only thing standing between this table
and a paper.

## The gate

E1 (running): bytes vs patch-8 at equal bytes seen, 2 seeds. Pre-registered:
patch wins if BPB ≤ bytes + 0.02; loses if > bytes + 0.05. Until it lands, the
cheap recipe is a hypothesis with arithmetic behind it, not a result.
