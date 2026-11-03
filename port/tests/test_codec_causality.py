"""Causal-invariance and anti-shortcut gates for BOTH byteflow codecs.

The conv codec merged in `95daa24` patchified each patch with a strided conv
over its OWN bytes and ran a symmetric (``padding=1``) residual conv over the
patch axis, so latent `i` contained bytes `>= is` — the bytes it then
predicted. Measured on the 5070 Ti (314 M, b8 s1024, patch_bytes=4, K=256=P):
eval BPB 3.6532 @200 and **0.4992 @400** while train loss was 0.7851. The
attention codec's encoder is causal (`h[:, p]` sees bytes `<= p`); its own A/B
(3.2427 vs 3.6152) is a plausible improvement, not this shape of collapse.

Gate 1 (generic, not codec-specific): for a causal model, changing
`bytes[:, q:]` must leave `logits[:, :q]` BIT-IDENTICAL for every `q`. It runs
at every `q` in the window (inside a patch and on every patch boundary for the
conv codec), fp32 and bf16, for both codecs. It FAILS on the pre-fix conv
codec (measured max |Δ| = 1.83e-1 at q=1) and passes on the attention codec.

Gate 2 (the attention arm of the same test): K=P (all patches/bytes selected).
At K<P BOTH chunkers run a global top-k over the whole window; the attention
codec's selection can therefore change when future bytes change, and the
strict invariance fails at q in {7, 8, 10-13} (characterised below as a
strict xfail). That is a future-dependent CHOICE among PAST latents — a
causality violation, not an information leak — and it predates this task.

Gate 3 (end to end): train codec-only Option A for 50 steps on the committed
real-text fixture and assert the held-out BPB is not below a collapse floor.
The floor is 1.0 BPB, picked from the measured attention arm: same 50 steps,
same split, CPU, seed 1234 gives 4.8329 BPB (fixed conv: 4.5367). 1.0 is
~0.2x that arm, so it cannot fire on a legitimate 50-step run, while a
label/target leak does: the `y = x` scratch reads 0.4402. The 95daa24 leak
reads 2.9094 on this tiny fixture at 50 steps (it needs ~400 GPU steps to
reach 0.5), so gate 1 — not this one — is what catches that defect; gate 3 is
the canary for a LATER leak that preserves forward causality but collapses the
score by another route.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from dormouse.config import ModelConfig
from dormouse.model import DormouseModel

PIN = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=32,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=0,
    engram_rows=64,
    engram_dim=8,
    shared_rank=16,
    use_byteflow=True,
    use_engram=False,
    use_kda=False,
    use_attnres=False,
    loom=False,
    byteflow_d_local=16,
    byteflow_d_global=16,
    byteflow_heads_local=2,
    byteflow_heads_global=2,
    byteflow_e_layers=1,
    byteflow_w_local=8,
    byteflow_d_ff_local=32,
    byteflow_bins=4,
    byteflow_max_bytes=32,
)

#: Window and batch of the invariance gate. 16 bytes = 4 conv patches at 4 B.
T, B = 16, 2
CONV_PATCH = 4


def cfg(**over) -> ModelConfig:
    base = dict(PIN)
    base.update(over)
    return ModelConfig(**base)


def _model(codec: str, bf16: bool) -> DormouseModel:
    """Option A (no loop arm): the codec's byte logits are the model's."""
    if codec == "conv":
        over = dict(
            byteflow_codec="conv",
            byteflow_patch_bytes=CONV_PATCH,
            byteflow_k_tokens=T // CONV_PATCH,
        )
    else:
        over = dict(byteflow_codec="attention", byteflow_k_tokens=T)
    torch.manual_seed(1234)
    return DormouseModel(cfg(bf16=bf16, **over))


def _logits(model: DormouseModel, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        flat, _, counters = model(x)
    assert counters.get("bf_net") == 1, counters
    return flat.reshape(x.shape[0], x.shape[1], 256)


def _x() -> torch.Tensor:
    return torch.arange(B * T, dtype=torch.long).reshape(B, T) % 251


def _q_ids() -> list[int]:
    # Every q proves "for EVERY q"; the list is the whole window plus both ends.
    return list(range(T + 1))


@pytest.mark.parametrize("codec", ["conv", "attention"])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_causal_invariance_at_every_q(codec: str, bf16: bool):
    """`bytes[:, q:]` must not move `logits[:, :q]` for ANY q (bit-exact)."""
    model = _model(codec, bf16)
    x = _x()
    base = _logits(model, x)
    assert torch.isfinite(base).all()
    for q in _q_ids():
        gen = torch.Generator().manual_seed(7 + q)
        changed = x.clone()
        changed[:, q:] = torch.randint(0, 256, changed[:, q:].shape, generator=gen)
        out = _logits(model, changed)
        assert torch.equal(base[:, :q], out[:, :q]), (
            f"{codec}/{('bf16' if bf16 else 'fp32')}: changing bytes[:, {q}:] "
            f"moved logits[:, :{q}] by max |Δ|="
            f"{(base[:, :q] - out[:, :q]).abs().max().item():.3e}. A causal codec "
            f"must not read the future; a leak here is the 95daa24 defect "
            f"(BPB 0.4992 @400) in another form."
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SECOND FINDING, pre-existing: at K<P the attention codec's "
        "`select_positions` is a GLOBAL top-k over the whole window, so a "
        "future byte can displace a past boundary and change which PAST latent "
        "a prefix position reads. The selected latents are all causal (no "
        "future BYTES are read into a value), but the strict prefix-invariance "
        "fails at q in {7,8,10,11,12,13}. K=P — the shape the 95daa24 leak was "
        "measured at — passes. Fixing this is a chunker design change, out of "
        "this task's scope; strict xfail makes the finding loud and forces its "
        "removal if it is ever fixed."
    ),
)
def test_attention_chunker_is_prefix_causal_at_k_lt_p():
    """Characterises the K<P global-top-k finding above (expected to fail)."""
    torch.manual_seed(1234)
    model = DormouseModel(cfg(byteflow_k_tokens=T // 2))
    x = _x()
    base = _logits(model, x)
    for q in _q_ids():
        gen = torch.Generator().manual_seed(7 + q)
        changed = x.clone()
        changed[:, q:] = torch.randint(0, 256, changed[:, q:].shape, generator=gen)
        out = _logits(model, changed)
        assert torch.equal(base[:, :q], out[:, :q]), f"prefix moved at q={q}"


#: Floor for the 50-step Option A gate. Chosen from the attention arm (the
#: leak-free reference measured on the same fixture, steps, batch and seed:
#: 4.8329 BPB; the fixed conv arm is 4.5367). 1.0 is ~0.2x the reference, so
#: no legitimate 50-step run can hit it, while a target/label leak does
#: (the y=x scratch reads 0.4402). This is a collapse canary, not a quality bar.
COLLAPSE_FLOOR_BPB = 1.0

_FIXTURE = Path(__file__).parent / "fixtures" / "byteflow_real_32k.txt"
_STREAM_T, _STREAM_B, _STREAM_STEPS = 128, 4, 50
_TRAIN_BYTES = 24576  # the fixture is 32 KiB; the tail 8 KiB is held out

STREAM_CFG = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=_STREAM_T,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=0,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
    use_byteflow=True,
    use_engram=False,
    use_kda=False,
    use_attnres=False,
    loom=False,
    byteflow_d_local=16,
    byteflow_d_global=64,
    byteflow_heads_local=2,
    byteflow_heads_global=2,
    byteflow_e_layers=1,
    byteflow_w_local=8,
    byteflow_d_ff_local=32,
    byteflow_bins=4,
    byteflow_k_tokens=_STREAM_T // CONV_PATCH,
    byteflow_max_bytes=_STREAM_T,
    byteflow_codec="conv",
    byteflow_patch_bytes=CONV_PATCH,
)


def test_option_a_does_not_collapse_bpb_in_50_steps():
    """The end-to-end anti-shortcut canary (gate 3 in the module docstring)."""
    raw = torch.tensor(list(_FIXTURE.read_bytes()), dtype=torch.long)
    train, held = raw[:_TRAIN_BYTES], raw[_TRAIN_BYTES:]
    torch.manual_seed(1234)
    model = DormouseModel(ModelConfig(**STREAM_CFG))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    for _ in range(_STREAM_STEPS):
        starts = torch.randint(0, len(train) - _STREAM_T - 1, (_STREAM_B,), generator=gen)
        x = torch.stack([train[s : s + _STREAM_T] for s in starts])
        y = torch.stack([train[s + 1 : s + 1 + _STREAM_T] for s in starts])
        model.train()
        _, loss, _ = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    total, windows = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(held) - _STREAM_T, _STREAM_T):
            x = held[s : s + _STREAM_T].unsqueeze(0)
            y = held[s + 1 : s + 1 + _STREAM_T].unsqueeze(0)
            _, loss, _ = model(x, y)
            total += float(loss)
            windows += 1
    bpb = total / windows / math.log(2)
    assert windows > 0 and math.isfinite(bpb)
    assert bpb > COLLAPSE_FLOOR_BPB, (
        f"Option A reached {bpb:.4f} BPB in {_STREAM_STEPS} steps on real text, "
        f"below the {COLLAPSE_FLOOR_BPB} collapse floor. The leak-free arms on "
        f"this fixture read 4.54 (conv) / 4.83 (attention); a score this low means "
        f"the objective or the data split leaks the answer, not that the codec "
        f"learned it. Check the target shift (y = x + 1) and the train/held-out "
        f"split before blaming the codec."
    )
