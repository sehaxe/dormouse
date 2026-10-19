"""WSD: warmup → stable → decay, with the warmup measured in TOKENS.

"Warmup-stable-decay" is the recipe the burn side runs (the post-training and
byte-LM research both land on it: warm up, hold the peak, then decay to ~0.1×
peak). Two decisions the port makes explicit:

* **warmup is a token count, not a step count.** A schedule in steps silently
  changes when the batch size changes, and the burn side has three separate
  step-time lessons about numbers that moved with a knob nobody expected.
* **the decay is a fraction of the total budget**, so extending a run
  (`--steps`) moves the decay window rather than truncating it — the same rule
  the resume seam uses for progress keys.
"""

from __future__ import annotations

from ..instrument import loud


def wsd_lr(
    step: int,
    *,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int = 0,
    decay_steps: int = 0,
    final_frac: float = 0.1,
) -> float:
    """The learning rate at `step` (0-based), in `[final_frac·peak, peak]`."""
    if total_steps < 1:
        loud(f"total_steps={total_steps} must be >= 1")
    if peak_lr <= 0.0:
        loud(f"peak_lr={peak_lr} must be > 0")
    if not 0.0 < final_frac <= 1.0:
        loud(f"final_frac={final_frac} must be in (0, 1]")
    if warmup_steps < 0 or decay_steps < 0:
        loud(f"warmup_steps={warmup_steps} decay_steps={decay_steps} must be >= 0")
    if warmup_steps == 0:
        lr = peak_lr
    elif step < warmup_steps:
        return peak_lr * float(step + 1) / float(warmup_steps)
    else:
        lr = peak_lr
    decay_start = max(total_steps - decay_steps, warmup_steps)
    if step < decay_start or decay_steps == 0:
        return lr
    frac = min(1.0, float(step - decay_start + 1) / float(decay_steps))
    return peak_lr * (1.0 - frac * (1.0 - final_frac))
