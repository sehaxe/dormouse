"""The compressed, selectable KV memory (CSA) — the ten gates.

WHAT THIS ADDS. `LoopBlock` gains an ADDITIVE branch beside KDA: one shared
`k`/`v` per `csa_chunk` byte positions (MQA, `csa_kv_dim` wide), exact
attention over the top `csa_topk` **strictly earlier** chunks, added behind a
learned per-iteration gate. `use_csa=false` builds nothing.

Gates, each of which fails on a revert:

1. `use_csa=false` is BIT-IDENTICAL to main — sha256 of `state_dict`+logits on
   fixed tiny fixtures, captured from a clean `main` worktree at `d80888d`
   before the arm existed.
2. CAUSALITY: with the engram off (attention the only cross-position read), a
   byte changed at `t` leaves every output at `< t` bit-identical and moves the
   output at `t`.
3. PARITY: `csa_chunk=1`, `csa_topk >= t` equals a hand-written STRICT causal
   softmax attention to 1e-5 (fp32).
4. gradients reach `W_k`, `W_v` and `csa_gate` (finite, non-zero, named).
5. SELECTION IS LIVE: top-1 vs all chunks changes the branch's output.
6. LOUD refusals for the pairs the arm is not specified on, and `csa_topk > t`
   is legal at runtime.
7. no host read in the forward (`.item/.tolist/.bool/.float` patched).
8. the memory table is the ACTUAL ring size: 32k at chunk 8 / kv 64 = 1.0 MiB.
9. the decode ring appends per chunk and refuses loudly past its cap.
10. the recurrent path carries the CSA state beside KDA's.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import ModelConfig, TrainConfig, _validate_segment, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.csa import (
    CompressedSelectiveAttention,
    CSAState,
    bytes_per_byte,
    memory_table,
    state_bytes,
)

BATCH, SEQ = 2, 64
SEED = 1234

#: sha256(state_dict keys+bytes | logits bytes), fixture below, seed 1234,
#: captured from a clean main worktree (`d80888d`) BEFORE this branch existed.
CSA_OFF_SHA = "beb23f76773c5f63a8b5ed5a549fb3c451174e9f4622c2549246cc63b4d0d318"
CSA_OFF_ENGRAM_OFF_SHA = "ef1c14cc7058d4a9fe10cae1e3efad6822f3fa4aab21cdf564ed23604f90d813"
#: The Engram-OFF pins moved on 2026-10-26: the dead `mem_dense` (`d²+d`,
#: 4160 here at d=64) is no longer built with `use_engram=false`. The
#: Engram-ON pin above did not move.
#: The causality fixture (engram off, dense MoE) — the same model gate 2 mutates.
CSA_OFF_CAUSAL_SHA = "2b3139030d8e46e62ef8ec5dfc5b517cec2ca002abb9db4814d0e19c610f4fb7"


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


# --- 1. the off switch is bit-identical to main ----------------------------


def test_gate_1_csa_off_is_bit_identical_to_main():
    got = _pinned({})
    print(f"csa=off sha {got}")
    assert got == CSA_OFF_SHA, "use_csa=false moved the default model"
    assert _pinned({"use_engram": False}) == CSA_OFF_ENGRAM_OFF_SHA
    assert _pinned({"use_engram": False, "moe_topk": 0}) == CSA_OFF_CAUSAL_SHA


# --- 2. strictly earlier chunks, both directions ---------------------------


def test_gate_2_csa_is_causal_both_ways():
    """Engram OFF: attention is the only op that can read another position.

    A changed byte at `t` may move the pooled key of `t`'s own chunk, but that
    key is read only by LATER chunks; every output at `< t` must be
    bit-identical. The query at `t` is built from the byte at `t`, so a `t`
    that has at least one earlier chunk (position 8 at chunk 8) must move.
    """
    torch.manual_seed(0)
    model = DormouseModel(
        tiny_cfg(use_csa=True, use_engram=False, moe_topk=0, csa_chunk=8, csa_topk=32)
    )
    ids, _ = _batch()
    with torch.no_grad():
        base = model(ids)[0].view(BATCH, SEQ, -1)
        # (a) a later byte cannot move an earlier position.
        later = ids.clone()
        later[:, 8] = (later[:, 8] + 1) % 251
        got = model(later)[0].view(BATCH, SEQ, -1)
        assert torch.equal(base[:, :8], got[:, :8]), "a position read a future byte"
        last = ids.clone()
        last[:, -1] = (last[:, -1] + 7) % 251
        assert torch.equal(base[:, :-1], model(last)[0].view(BATCH, SEQ, -1)[:, :-1])
        # (b) the mutated position itself moves (it has earlier chunks to read).
        assert not torch.allclose(base[:, 8], got[:, 8]), (
            "position 8 ignored its own byte — the CSA query is not live"
        )
        print(f"pos-8 max delta {float((base[:, 8] - got[:, 8]).abs().max()):.3e}")


# --- 3. parity with hand-written exact strict-causal attention -------------


def test_gate_3_chunk_1_topk_all_equals_exact_attention():
    torch.manual_seed(0)
    cfg = tiny_cfg(csa_chunk=1, csa_topk=SEQ)
    attn = CompressedSelectiveAttention(cfg)
    x = torch.randn(BATCH, 16, cfg.d_model)
    out = attn(x)
    q = attn.W_q(x)
    k = attn.W_k(x)
    v = attn.W_v(x)
    ref = torch.zeros_like(x)
    for p in range(1, 16):  # strictly earlier positions: 0..p-1 (no diagonal)
        s = (q[:, p : p + 1] @ k[:, :p].transpose(1, 2)).squeeze(1) / math.sqrt(cfg.csa_kv_dim)
        w = torch.softmax(s, dim=-1)
        ref[:, p] = (w.unsqueeze(-1) * v[:, :p]).sum(1)
    ref = attn.W_o(ref)
    err = float((out - ref).detach().abs().max())
    print(f"max |CSA - strict causal softmax| = {err:.3e}")
    assert err <= 1e-5, err


# --- 4. gradients reach W_k, W_v, the gate --------------------------------


def test_gate_4_gradients_reach_the_kv_and_the_gate():
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg(use_csa=True, use_engram=False, moe_topk=0))
    ids, targets = _batch()
    _, loss, _ = model(ids, targets)
    assert loss is not None
    loss.backward()
    wanted = {
        "loop.csa.W_k.weight": model.loop.csa.W_k.weight.grad,
        "loop.csa.W_v.weight": model.loop.csa.W_v.weight.grad,
        "loop.csa_gate": model.loop.csa_gate.grad,
    }
    for name, grad in wanted.items():
        assert grad is not None and torch.isfinite(grad).all(), f"{name}: no finite gradient"
        mag = float(grad.abs().sum())
        assert mag > 0.0, f"{name}: zero gradient — the branch runs and trains nothing"
        print(f"{name:22s} |grad|_1 = {mag:.6f}")


# --- 5. the selection is live ---------------------------------------------


def test_gate_5_topk_changes_the_output():
    torch.manual_seed(0)
    cfg = tiny_cfg(csa_chunk=4, csa_topk=1)
    attn = CompressedSelectiveAttention(cfg)
    x = torch.randn(BATCH, 32, cfg.d_model)
    with torch.no_grad():
        few = attn(x)
        attn.topk = 32  # every earlier chunk
        all_chunks = attn(x)
    delta = float((few - all_chunks).abs().max())
    print(f"top-1 vs all-chunks max delta {delta:.3e}")
    assert not torch.allclose(few, all_chunks), "top-k selected nothing different"
    assert delta > 0.0


# --- 6. the refusals are LOUD, topk>t is runtime-legal ---------------------


def test_gate_6_unsupported_combinations_are_loud():
    with pytest.raises(LoudError, match="csa_chunk"):
        validate(tiny_cfg(use_csa=True, csa_chunk=6))
    with pytest.raises(LoudError, match="csa_kv_dim"):
        validate(tiny_cfg(use_csa=True, csa_kv_dim=32))
    with pytest.raises(LoudError, match="csa_topk"):
        validate(tiny_cfg(use_csa=True, csa_topk=0))
    with pytest.raises(LoudError, match="use_msa"):
        validate(tiny_cfg(use_csa=True, use_msa=True))
    with pytest.raises(LoudError, match="use_byteflow"):
        validate(tiny_cfg(use_csa=True, use_byteflow=True))
    with pytest.raises(LoudError, match="use_kda=false"):
        validate(tiny_cfg(use_csa=True, use_kda=False))
    with pytest.raises(LoudError, match="attn_kind"):
        validate(tiny_cfg(use_csa=True, attn_kind="full"))
    with pytest.raises(LoudError, match="use_csa"):
        _validate_segment(tiny_cfg(use_csa=True), TrainConfig(segment_len=32, seq_len=SEQ))
    # The runtime seam: a hand-built model never calls `validate`.
    model = DormouseModel(tiny_cfg(use_csa=True))
    ids, targets = _batch()
    with pytest.raises(AssertionError, match="use_csa"):
        model(ids, targets, segment_len=32)
    # topk > t is fine at runtime: the selection is min(topk, available).
    wide = DormouseModel(tiny_cfg(use_csa=True, csa_topk=1000, use_engram=False, moe_topk=0))
    logits, loss, _ = wide(ids, targets)
    assert loss is not None and torch.isfinite(loss)
    print(
        "loud: csa_chunk/csa_kv_dim/csa_topk, use_msa, use_byteflow, use_kda=false, "
        "attn_kind='full', segment_len>0; topk=1000 ran"
    )


# --- 7. no host read in the forward ---------------------------------------


def test_gate_7_the_forward_touches_no_host(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("host sync inside the CSA forward (ADR-0018 rule 2)")

    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "tolist", boom)
    monkeypatch.setattr(torch.Tensor, "bool", boom)
    monkeypatch.setattr(torch.Tensor, "float", boom)
    monkeypatch.setattr(torch.Tensor, "__bool__", boom)
    monkeypatch.setattr(torch.Tensor, "__float__", boom)
    torch.manual_seed(0)
    cfg = tiny_cfg(csa_chunk=8)
    attn = CompressedSelectiveAttention(cfg)
    x = torch.randn(BATCH, 32, cfg.d_model)
    counters = Counters()
    out = attn(x, counters=counters)
    assert torch.equal(torch.isfinite(out), torch.ones_like(out, dtype=torch.bool))
    assert counters["csa_forward"] == 1
    _, state = attn.step_state(x[:, :1], None, counters=counters)
    assert state.n == 0 and counters["csa_step"] == 1
    print(f"no host read: csa_forward={counters['csa_forward']} csa_step={counters['csa_step']}")


# --- 8. the memory table is the actual ring size --------------------------


def test_gate_8_memory_table_matches_the_state_tensors():
    table = memory_table(csa_kv_dim=64, csa_chunk=8)
    assert bytes_per_byte(64, 8) == 32.0
    assert table == {1024: 32768, 32768: 1_048_576, 1_048_576: 33_554_432}
    # 32k context → 4096 chunks → 1.0 MiB per layer, and it is the tensor size.
    st = CSAState(64, 64, 8, batch=1)
    st.reserve(32768 // 8)
    assert st.k.element_size() == 2 and st.v.element_size() == 2
    assert st.bytes() == state_bytes(32768, 64, 8) == 1_048_576
    print(f"32k ring: {st.bytes()} B = {st.bytes() / 2**20:.3f} MiB")
    # The 1k and 1M rows of the same table, against real allocations.
    small = CSAState(64, 64, 8, batch=1)
    small.reserve(1024 // 8)
    assert small.bytes() == table[1024] == state_bytes(1024, 64, 8)
    large = CSAState(64, 64, 8, batch=1)
    large.reserve(1_048_576 // 8)
    assert large.bytes() == table[1_048_576] == state_bytes(1_048_576, 64, 8)
    print(
        f"1k ring {small.bytes()} B, 1M ring {large.bytes()} B = "
        f"{large.bytes() / 2**20:.1f} MiB (both are the helper's rows)"
    )


# --- 9. the decode ring appends and refuses to forget ---------------------


def test_gate_9_decode_ring_appends_and_refuses_past_the_cap():
    cfg = tiny_cfg(csa_chunk=4)
    attn = CompressedSelectiveAttention(cfg)
    state = CSAState(cfg.d_model, cfg.csa_kv_dim, cfg.csa_chunk, batch=1)
    with torch.no_grad():
        for _p in range(8):
            _, state = attn.step_state(torch.randn(1, 1, cfg.d_model), state)
    assert (state.n, state.count) == (2, 0), (state.n, state.count)
    tight = CSAState(cfg.d_model, cfg.csa_kv_dim, cfg.csa_chunk, batch=1, max_chunks=2)
    k = torch.zeros(1, cfg.csa_kv_dim)
    tight.append(k, k)
    tight.append(k, k)
    with pytest.raises(LoudError, match="cannot grow"):
        tight.append(k, k)
    print(f"ring: {state.n} chunks / {state.count} pooled; cap 2 -> LoudError on the 3rd")


# --- 10. the recurrent path carries CSA beside KDA ------------------------


def test_gate_10_recurrent_step_carries_the_chunk_ring():
    torch.manual_seed(0)
    cfg = tiny_cfg(use_csa=True, use_engram=False, moe_topk=0, csa_chunk=4)
    model = DormouseModel(cfg)
    ids, _ = _batch()
    states: list = [None] * cfg.max_iter
    with torch.no_grad():
        for p in range(8):
            out, states = model.loop.recurrent_step(
                model.embed(ids[:, p : p + 1]), ids[:, : p + 1], states, counters=Counters()
            )
    assert out.shape == (BATCH, 1, cfg.d_model) and torch.isfinite(out).all()
    for slot, st in enumerate(states):
        assert isinstance(st, tuple) and len(st) == 2, (slot, type(st))
        assert st[1].n == 2 and st[1].count == 0, (slot, st[1].n, st[1].count)
        assert st[0] is not None  # the KDA half still rides the same list
    print("recurrent: " + ", ".join(f"slot{s}: csa_n={st[1].n}" for s, st in enumerate(states)))
