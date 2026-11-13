# The `train --init-from` explosion (2026-10-22, open)

**Status: OPEN. Do not use `train --init-from` on a converged host-table
checkpoint until this is fixed.** SFT's `--init-from` is unaffected (see
"why SFT survives").

## What happens

Resuming training from a converged checkpoint through the new `--init-from`
seam degrades the model monotonically and then explodes, with **no
hyper-parameter making it better**:

| run | flags that differ | loss trace |
|---|---|---|
| v1 (QAT) | `factor_qat=ternary dense_qat=ternary`, lr 1e-4, `opt_graph` | 1.384 → **963.8** → 6.98 → 1.89, eval BPB **2.8497** at step 1000 |
| v2 (factors only) | `factor_qat=ternary`, lr 3e-5, no `opt_graph` | 1.272 → 1.403 → **971 332** → 323 198 → 9 816 |
| diag 3 (**no quantiser at all**) | `factor_qat=none`, lr 3e-5 | 1.272 → 1.407 → **1 003 269** → 15 633 418 |
| diag 4 (**weights frozen**) | `factor_qat=none`, **lr 1e-12** | 1.275 → 1.500 → **11 425 785** → 132 933 208 |

Reference: mix1 at step 170 000 reads `loss 1.3052, bpb 1.883` on the same window
(`8 x 24 x 1024 = 196 608 B`), and the first 100-200 steps of every arm above
start healthy (1.27-1.50).

## What that rules out

- **Not the quantiser**: `factor_qat=none` explodes identically. The ternary
  ladder's +0.057 (see `2026-10-22-ternary-ladder.md`) is not implicated.
- **Not the learning rate**: at `lr=1e-12` the parameters cannot move, and the
  loss still explodes at the same step (~300). So the corruption lives in state
  that is **not a parameter**.
- **Not the CUDA graph**: diag 3 ran with `opt_graph=false`.
- **Not the data**: all four arms draw the same `mix_big` stream from byte 0, and
  the first 200 steps are healthy.

## What it implicates

The only non-parameter state stepped every training step on this path is the
**Engram host table set** (`host_step_all(model, lr)` at `train/loop.py:945`).
Its update runs unconditionally when `engram_ram=true`, and it writes into
`HostTables` (CPU fp32 tables + one momentum buffer) that were **loaded from a
foreign `.ngram` sidecar** by `load_model_state` → `load_host_state`.

`host_step_all` also applies a **Sinkhorn** normalisation, which is
lr-independent — the reason `lr=1e-12` does not protect the tables: even a
frozen-lr step renormalises them.

Two candidate mechanisms, not yet separated:

1. the loaded momentum buffer is not in the state `host_step_all` expects (a
   format/offset mismatch between `save_ckpt`'s sidecar and `load_host_state`),
   so the Nesterov term is wrong-sized from the first step and accumulates; or
2. the Sinkhorn pass is not idempotent on a converged table set and drifts it,
   with the drift amplified by the table read path (`host_step` sums per
   iteration).

## Why SFT survives

`train/sft.py` never calls `host_step_all` — its host tables are read but not
stepped. sft12/sft13 trained 4000 steps on this same checkpoint (holdout CE
1.437 → 0.898) with no explosion, which is the control that isolates the
corruption to the host-table **update**, not the load, not the forward.

## The fix direction

A gate first, then the fix. The cheapest reproducer: load a small host-table
checkpoint, run 3 steps with `lr=1e-12`, and assert the table tensors are
bit-identical between step 0 and step 3 (a frozen lr must not move them; if the
Sinkhorn is intentional, it must be shown to be a no-op on an already-normalised
table). Then compare the sidecar round-trip: `save_ckpt` → `load_host_state` →
`host_step_all` on the same tables must reproduce the momentum the writer had.

## Consequence for tonight's plan

The QAT-finish of mix1 (bf16 → ternary fine-tune) is **blocked**. The release
model must come from the from-scratch ternary run instead — which the ladder
measured as stable (`q32w`: 3.065 vs the bf16 control's 3.080 over 2000 steps,
3 seeds) — or the second of these, whichever comes first:

- fix the host-table step and re-run the QAT-finish (2 h of GPU); or
- a from-scratch `q32w`-geometry run at a real budget (the recipe is frozen and
  measured, and it does not depend on this bug).

## RESOLVED (2026-10-22): the LFB routing bias, not the host tables

The implicated state was the wrong one. The Engram host tables are exonerated
by direct measurement, and the corruption is the **Loss-Free-Balancing routing
bias** — also non-parameter state, but stepped in the *forward* (3× per step,
once per loop iteration), which is why no learning rate freezes it.

**The host tables are innocent** (CPU probe over the real
`checkpoints/mix1/mix1.ngram`, 3 steps of `host_step_all` at `lr=1e-12`):

- rows **bit-identical** after 3 frozen-lr steps (`torch.equal` of the int32
  views); 300 frozen steps move 3 rows by 4.4e-16 (fp32 rounding only);
- `host_state` → `torch.save` → `load_host_state` round-trips rows, momentum
  and `step` **bit-exactly** — there is no offset/format mismatch;
- `sinkhorn_l1` is not bit-idempotent but convergent: applying it twice moves
  the block by 3.4e-8, a fourth time by 1.2e-10 (it normalises the *update
  block*, not the table, and the table only moves by `lr·block`).

**The mechanism.** `SparseMoE.forward` adds `lfb_bias` to the router logits
unconditionally (`model/experts.py:850`); `_lfb_step` moves it by
`±moe_lb_coef` per routed forward from the **cumulative** counts
(`model/experts.py:912-913`). `lfb_bias` is a non-persistent buffer
(`experts.py:759`) — not a parameter, not in the model half — so the optimizer
and `lr` never touch it. `--init-from` deliberately has no config-drift gate,
so `train medium`'s preset coefficient (`configs/medium.toml:44`, 0.01)
silently switched the controller ON over mix1, which trained with it OFF
(`checkpoints/mix1/mix1.config.toml:37`, 0). A cold integrator over a converged
router has no feedback: the top-k is pinned by weights the bias does not reach
until it is several logits tall, the sign never flips, and the bias ramps
linearly. Measured on the diag-4 run: bias ±4.8 after ~480 forwards, routing
moving onto experts the model never selects, loss 1.275 → 1.500 → 11 425 785 by
step 300 with the weights frozen (lr=1e-12).

**The fifth diagnostic** separates the forward state from the data and from
every training-path update: the diag-4 stream, run **forward-only** (no
backward, no optimizer, no host update, same weights/data), explodes at step
173 with `moe_lb_coef=0.01` and stays at 1.21–1.49 for all 400 steps with
`moe_lb_coef=0`. The only difference between the two arms is the controller.

**The fix** (`train/loop.py::init_controller`, called from the `init_from`
branch): the init file's snapshot is the authority on the controller — the
seam carries `moe_lb_coef` from `<ckpt>.config.toml`, LOUD, and refuses
outright when a checkpoint has no snapshot but the run asks for a live
controller. A fresh run never enters the branch (a gate pins that, and a 5-step
fresh trace is bit-identical with the fix removed).

Acceptance (`train medium --init-from mix1 --set lr=3e-5 ..., 400 steps`):
`step=300 loss=1.3053`, `step=400 loss=1.3136` — flat, against 1 003 269 at
step 300 before.
