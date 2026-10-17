"""Byte-level decode for the port: sliding-window autoregressive next-byte.

The context is always the last `max_seq_len` bytes: every step re-forwards the
whole window, so the KDA state is rebuilt from exactly the bytes a reader of
the log would attribute to it — no incremental-kernel assumption hides inside
generation. Slow by design at large windows; correctness first, an incremental
decoder is a profile-then-build item.

`window_forward` is the ONE model call this module and `specdecode.py` share:
spec decode is this same forward over a longer sequence, never a second
implementation of it.
"""

from __future__ import annotations

import torch

from ..instrument import loud

#: The pseudo-count temperature keeps sampling near-greedy but never zerolock.
DEFAULT_TEMP = 0.8


def encode(text: str) -> bytes:
    """UTF-8 — the corpus is raw UTF-8 bytes, so a prompt is too."""
    return text.encode("utf-8")


def window_forward(
    model, ids: torch.Tensor, *, hidden: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """ONE `model(ids)` call: `(logits [t, vocab], hidden [t, d] | None)`.

    The one model call the port's decode paths share. `decode_bytes` takes the
    last row per byte; `specdecode` verifies a whole draft block with one call
    over a longer sequence — the same function, so a speculative verify cannot
    become a second implementation of the forward. Batch 1 by construction.

    `hidden` returns the tensor the DSpark draft head was TRAINED on
    (`model.py`: `hn = self.norm(out)`, the first argument of
    `dspark_aux_loss`), captured with a forward hook on `model.norm` because
    `forward` does not return it — an observation of the one call, not a
    re-implementation of it.
    """
    assert ids.ndim == 2 and ids.shape[0] == 1, f"decode is batch 1, got {tuple(ids.shape)}"
    t = ids.shape[1]
    if not hidden:
        return model(ids)[0].reshape(t, -1), None
    if not hasattr(model, "norm"):
        loud(
            "hidden=True needs the byte-lm path (`model.norm`): the byteflow shapes "
            "have no norm, and the DSpark head is refused with use_byteflow. "
            "Escape: decode with use_byteflow=false."
        )
    captured: dict[str, torch.Tensor] = {}
    handle = model.norm.register_forward_hook(lambda _m, _i, out: captured.__setitem__("h", out))
    try:
        logits = model(ids)[0].reshape(t, -1)
    finally:
        handle.remove()
    if "h" not in captured:
        loud(
            "the forward never reached `model.norm`: the draft head's hidden state is "
            "undefined on this path. Escape: spec decode needs the plain byte-lm "
            "forward (no use_byteflow)."
        )
    return logits, captured["h"].reshape(t, -1)


def decode_bytes(
    model,
    prompt: bytes,
    n_new: int,
    *,
    temp: float = DEFAULT_TEMP,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    stop: frozenset[int] | None = None,
) -> bytes:
    """Return `prompt + n_new` sampled bytes, temperature-scaled logits.

    `stop` names bytes that END the reply (the SFT template's `0x03`): a
    generated stop byte is appended and generation returns early, so the
    caller never has to trim the model's padding habits out of a reply. The
    default (`None`) keeps the historical contract — exactly `n_new` bytes.
    """
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
            logits, _ = window_forward(model, ids)
            probs = torch.softmax(logits[-1] / temp, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator))
            ctx.append(nxt)
            if stop is not None and nxt in stop:
                break
    model.train(was_training)
    return bytes(ctx)
