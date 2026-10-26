"""The full-attention arm (`attn_kind`) — seven gates on one flag.

WHAT THIS ADDS. `LoopBlock` had one attention arm (`KDAAttention`, a LINEAR
recurrence over a compressed state) and a zeros branch (`use_kda=false`). The
model it is compared against runs exact causal attention at 1 024 tokens, and
`4·t·d = 4.2M` MAC/token is affordable there — so `ModelConfig.attn_kind` now
selects `kda` (default, unchanged) or `full` (`model/fullattn.py`, SDPA with
`is_causal=True`), and both arms reach the body through the SAME
`LoopBlock._attn` seam.

Gates, each of which fails on a revert:

1. `attn_kind="kda"` (the default) is BIT-IDENTICAL to main: the sha256 of
   `state_dict`+logits on a fixed tiny fixture, captured from a clean
   `29510db` checkout before the edit (`beb23f76…`, and `15d1d7df…` with the
   engram off, the fixture gate 2 runs on).
2. `full` is CAUSAL BOTH WAYS on the whole model with the engram off (so
   attention is the only cross-position op): changing a LATER byte leaves
   every earlier logit bit-identical; changing byte 0 moves the LAST
   position's logits (a longer prefix is read).
3. gradients reach all four projections (`q,k,v,o`), finite and non-zero.
4. the op is exact: at `t=16`, fp32, `FullAttention(x)` equals a hand-written
   `softmax(qkᵀ/√d + causal_mask)·v` (then `o_proj`) to 1e-5.
5. every declared-unsupported combination is LOUD: an unknown `attn_kind`, the
   hard off switch + `full`, `segment_len>0`, the recurrent decode/prefill —
   while the combinations with no real conflict (`use_mhc`, `use_byteflow`)
   still BUILD AND RUN (evidence, not a promise).
6. no host read in the forward (`.item/.tolist/.bool/.float` patched to raise).
7. the parameter delta at `medium` is exactly `-d·n_heads = -16 384`
   (`dormouse info`: 141 910 405 → 141 894 021).

The KDA goldens were captured from commit `29510db` (main) in a clean
worktree, BEFORE `attn_kind` existed; the constants are not regenerated from
this branch.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch
from conftest import tiny_cfg
from test_model import BYTEFLOW

from dormouse.config import ModelConfig, TrainConfig, _validate_segment, load_preset, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.decode import recurrent_refusal
from dormouse.model.fullattn import FullAttention

BATCH, SEQ = 2, 64
SEED = 1234

#: sha256(state_dict keys+bytes | logits bytes), fixture below, seed 1234,
#: captured from a clean main checkout (`29510db`) before this arm existed.
KDA_SHA = "beb23f76773c5f63a8b5ed5a549fb3c451174e9f4622c2549246cc63b4d0d318"
KDA_ENGRAM_OFF_SHA = "ef1c14cc7058d4a9fe10cae1e3efad6822f3fa4aab21cdf564ed23604f90d813"
#: Moved 2026-10-26: `use_engram=false` no longer builds the dead `mem_dense`.
#: The Engram-ON `KDA_SHA` above is unchanged.

MEDIUM_KDA_PARAMS = 141_910_405
MEDIUM_FULL_PARAMS = 141_894_021


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    return ids, (ids + 1) % 251


def _sha(model: torch.nn.Module, logits: torch.Tensor) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    h.update(b"|logits|")
    h.update(logits.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _pinned(over: dict) -> str:
    torch.manual_seed(SEED)
    model = DormouseModel(ModelConfig(**{**dataclasses.asdict(tiny_cfg()), **over}))
    with torch.no_grad():
        logits, _, _ = model(_batch()[0])
    return _sha(model, logits)


# --- 1. the default arm is bit-identical to main ---------------------------


def test_gate_1_kda_default_is_bit_identical_to_main():
    got = _pinned({})
    print(f"kda sha {got}")
    assert got == KDA_SHA, "attn_kind='kda' (the default) moved the model"
    got = _pinned({"use_engram": False})
    assert got == KDA_ENGRAM_OFF_SHA, "the engram-off fixture gate 2 uses moved"


# --- 2. exact attention is causal in both directions -----------------------


def test_gate_2_full_attention_is_causal_both_ways():
    """Engram OFF: attention is the only op that can read another position."""
    torch.manual_seed(0)
    # `moe_topk=0` (the dense blend): a CAPACITY-DROPPING MoE is not prefix-causal
    # by design (which expert drops what depends on the other tokens in the
    # forward, `test_decode_recurrent.py`), and this gate is about attention.
    model = DormouseModel(tiny_cfg(attn_kind="full", use_engram=False, moe_topk=0))
    assert isinstance(model.loop.attn, FullAttention)
    ids, _ = _batch()
    with torch.no_grad():
        base = model(ids)[0].view(BATCH, SEQ, -1)
        # (a) a LATER byte cannot move an EARLIER position.
        later = ids.clone()
        later[:, 1] = (later[:, 1] + 1) % 251
        got = model(later)[0].view(BATCH, SEQ, -1)
        assert torch.equal(base[:, 0], got[:, 0]), "position 0 read position 1"
        last = ids.clone()
        last[:, -1] = (last[:, -1] + 1) % 251
        got = model(last)[0].view(BATCH, SEQ, -1)
        assert torch.equal(base[:, :-1], got[:, :-1]), "a position read a future byte"
        # (b) the LAST position reads the whole prefix: byte 0 moves it.
        first = ids.clone()
        first[:, 0] = (first[:, 0] + 7) % 251
        moved = model(first)[0].view(BATCH, SEQ, -1)
        assert not torch.allclose(base[:, -1], moved[:, -1]), (
            "the last position ignored the first byte — attention is not reading its prefix"
        )
        print(f"last-row max delta {float((base[:, -1] - moved[:, -1]).abs().max()):.3e}")


# --- 3. the gradients reach q/k/v/o ----------------------------------------


def test_gate_3_gradients_reach_q_k_v_o():
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg(attn_kind="full"))
    ids, targets = _batch()
    _, loss, _ = model(ids, targets)
    assert loss is not None
    loss.backward()
    grads = {
        n.removeprefix("loop.attn."): p.grad
        for n, p in model.named_parameters()
        if n.startswith("loop.attn.")
    }
    assert set(grads) == {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"}, (
        sorted(grads)
    )
    for name, grad in sorted(grads.items()):
        assert grad is not None and torch.isfinite(grad).all(), f"{name}: no finite gradient"
        mag = float(grad.abs().sum())
        assert mag > 0.0, f"{name}: zero gradient — the arm runs and trains nothing"
        print(f"{name:14s} |grad|_1 = {mag:.4f}")


# --- 4. the op is exact ----------------------------------------------------


def test_gate_4_full_attention_equals_the_hand_written_softmax():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    attn = FullAttention(cfg)
    x = torch.randn(BATCH, 16, cfg.d_model)
    q, k, v = attn.projections(x)  # [b, t, h, hd]
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scores = (qh @ kh.transpose(-1, -2)) * (1.0 / (cfg.head_dim**0.5))
    mask = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    ref = (torch.softmax(scores, dim=-1) @ vh).transpose(1, 2).reshape(BATCH, 16, cfg.d_model)
    ref = torch.nn.functional.linear(ref, attn.o_proj.weight)  # o_proj has no bias
    out = attn(x)
    err = float((out - ref).detach().abs().max())
    print(f"max |FullAttention - softmax| = {err:.3e}")
    assert err <= 1e-5, err


# --- 5. the refusals are LOUD, the legal pairs run -------------------------


def test_gate_5_unsupported_combinations_are_loud():
    with pytest.raises(LoudError, match="attn_kind"):
        validate(tiny_cfg(attn_kind="flash"))
    with pytest.raises(LoudError, match="use_kda=false"):
        validate(tiny_cfg(attn_kind="full", use_kda=False))
    with pytest.raises(LoudError, match="attn_kind='full'"):
        _validate_segment(tiny_cfg(attn_kind="full"), TrainConfig(segment_len=32, seq_len=SEQ))
    # The runtime seam: a hand-built model never calls `validate`.
    model = DormouseModel(tiny_cfg(attn_kind="full"))
    ids, targets = _batch()
    with pytest.raises(AssertionError, match="attn_kind='full'"):
        model(ids, targets, segment_len=32)
    why = recurrent_refusal(model)
    assert why is not None and "attn_kind='full'" in why, why
    with pytest.raises(AssertionError, match="attn_kind='full'"):
        model.loop.recurrent_step(
            model.embed(ids[:, :1]),
            ids[:, :2],
            [None] * model.cfg.max_iter,
            drop=1,
            counters=Counters(),
        )
    with pytest.raises(LoudError, match="use_kda=false"):
        DormouseModel(tiny_cfg(attn_kind="full", use_kda=False))


def test_gate_5b_the_legal_pairs_run():
    """mHC is residual-lane logic and byteflow attends over patch latents: no conflict."""
    ids, targets = _batch()
    # mHC: independent of which attention arm runs (residual lanes, loop boundary).
    mhc = DormouseModel(tiny_cfg(attn_kind="full", use_mhc=True))
    logits, loss, c = mhc(ids, targets)
    assert loss is not None and torch.isfinite(loss)
    assert c["attn_full_forward"] == mhc.cfg.max_iter and c["mhc"] == mhc.cfg.max_iter
    # byteflow Option B: causal attention over the K patch latents the channel
    # already feeds to KDA — the same sequence, uncompressed.
    bf = DormouseModel(tiny_cfg(attn_kind="full", **BYTEFLOW))
    logits, loss, c = bf(ids[:, :32], targets[:, :32])
    assert loss is not None and torch.isfinite(loss)
    assert c["bf_channel"] == 1 and c["attn_full_forward"] == bf.cfg.max_iter
    assert c.get("kda_sdpa", 0) == 0 and c.get("kda_fla", 0) == 0, "the KDA arm also ran"


# --- 6. no host read in the full forward -----------------------------------


def test_gate_6_the_full_forward_touches_no_host(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("host sync inside the full-attention forward (ADR-0018 rule 2)")

    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "tolist", boom)
    monkeypatch.setattr(torch.Tensor, "bool", boom)
    monkeypatch.setattr(torch.Tensor, "float", boom)
    monkeypatch.setattr(torch.Tensor, "__bool__", boom)
    monkeypatch.setattr(torch.Tensor, "__float__", boom)
    # The ARM's own forward, not the whole model: the CPU MoE's naive dispatch
    # reads `unique(...).tolist()` BY DESIGN (its counted slow path), so a
    # model-wide gate would fail on an arm this one does not own.
    torch.manual_seed(0)
    cfg = tiny_cfg(attn_kind="full")
    attn = FullAttention(cfg)
    x = torch.randn(BATCH, 16, cfg.d_model)
    counters = Counters()
    out = attn(x, counters=counters)
    assert torch.equal(torch.isfinite(out), torch.ones_like(out, dtype=torch.bool))
    assert counters["attn_full_forward"] == 1


# --- 7. the parameter delta at medium, documented --------------------------


def test_gate_7_the_medium_parameter_delta():
    base = load_preset("medium")

    def count(kind: str) -> int:
        model = DormouseModel(dataclasses.replace(base, attn_kind=kind), device="meta")
        return sum(p.numel() for p in model.parameters())

    kda, full = count("kda"), count("full")
    print(f"medium kda={kda:,} full={full:,} delta={full - kda:+,}")
    assert (kda, full) == (MEDIUM_KDA_PARAMS, MEDIUM_FULL_PARAMS)
    assert full - kda == -base.d_model * base.n_heads == -16_384
