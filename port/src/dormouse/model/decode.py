"""Byte-level decode for the port: sliding-window autoregressive next-byte.

The context is always the last `max_seq_len` bytes: every step re-forwards the
whole window, so the KDA state is rebuilt from exactly the bytes a reader of
the log would attribute to it — no incremental-kernel assumption hides inside
generation. Slow by design at large windows; correctness first, an incremental
decoder is a profile-then-build item.
"""

from __future__ import annotations

import torch

from ..instrument import loud

#: The pseudo-count temperature keeps sampling near-greedy but never zerolock.
DEFAULT_TEMP = 0.8


def encode(text: str) -> bytes:
    """UTF-8 — the corpus is raw UTF-8 bytes, so a prompt is too."""
    return text.encode("utf-8")


def decode_bytes(
    model,
    prompt: bytes,
    n_new: int,
    *,
    temp: float = DEFAULT_TEMP,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> bytes:
    """Return `prompt + n_new` sampled bytes, temperature-scaled logits."""
    if not prompt:
        loud("decode needs a non-empty prompt (the first window is the prompt)")
    if temp <= 0:
        loud(f"temp={temp}: use a positive temperature (0 means greedy, not this flag)")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    was_training = model.training
    model.eval()
    ctx = list(prompt)
    window = model.cfg.max_seq_len
    with torch.no_grad():
        for _ in range(n_new):
            ids = torch.tensor([ctx[-window:]], dtype=torch.long, device=dev)
            logits = model(ids)[0][-1]
            probs = torch.softmax(logits / temp, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator))
            ctx.append(nxt)
    model.train(was_training)
    return bytes(ctx)
