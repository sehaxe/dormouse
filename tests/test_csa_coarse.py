"""The COARSE, DENSE second level (HCA) of CSA — the gates.

WHAT THIS ADDS. `csa_coarse_chunk > 0` makes the CSA arm two levels: the
existing fine top-k level and a coarse level that mean-pools
`csa_coarse_chunk` bytes into one entry and attends DENSELY over all strictly
earlier coarse entries. Their outputs are summed and pass through the ONE
existing `csa_gate` per iteration slot.

Gates, each of which fails on a revert:

1. `csa_coarse_chunk=0` is BIT-IDENTICAL to `main` (sha256 of state_dict +
   logits, captured from a clean `main` worktree at `962d831` BEFORE this
   branch existed) — both with `use_csa=false` and with the fine level alone.
2. CAUSALITY: mutating byte `t` leaves every output at `< t` bit-identical and
   moves the output at `t`, for each level and for the pair.
3. PARITY: the coarse level alone (`csa_fine=false`, chunk=1, coarse_chunk=1)
   equals a hand-written STRICT causal softmax attention to 1e-5; with both
   levels exact the output is the SUM of the two exact attentions, stated
   honestly because that double-counts the past.
4. gradients reach `W_k`/`W_v` of BOTH levels and the shared `csa_gate`
   (finite, non-zero, named).
5. LEVEL IS LIVE: turning the coarse level on changes the output, and zeroing
   its output projection changes it again (back to the fine-only value).
6. LOUD refusals for every pair the arm is not specified on.
7. no host read in the forward (`.item/.tolist/.bool/.float` patched).
8. the memory table prints and asserts BOTH levels and their sum against real
   tensor allocations at 1k/32k/1M.
9. the 1M decode bill is printed for {fine only, coarse only, both} and the
   coarse level is asserted under 3M MAC/byte — the whole point of the arm.
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
    decode_cost_table,
    mac_per_byte,
    memory_table,
    total_state_bytes,
)

BATCH, SEQ = 2, 64
SEED = 1234

#: Captured from a clean `main` worktree (962d831) BEFORE this branch existed,
#: seed 1234, fixture below.  Coarse is OFF, so these must never move.
CSA_OFF_SHA = "beb23f76773c5f63a8b5ed5a549fb3c451174e9f4622c2549246cc63b4d0d318"
CSA_ON_FINE_ONLY_SHA = "defb2afec16ed17637447ad5c495fb105e4c382dd7c64e9aacfbaf0d0ec986b6"
CSA_ON_FINE_ONLY_CAUSAL_SHA = "5c679b42c30f67d786b66e9d1f4e6f17f25789f60c61e65382dac322d8473e0a"
#: Moved 2026-10-26: the fixture is `use_engram=false`, which no longer builds
#: the dead `mem_dense`. `CSA_OFF_SHA`/`CSA_ON_FINE_ONLY_SHA` (Engram on) are
#: unchanged.


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


def test_gate_1_coarse_off_is_bit_identical_to_main():
    got = _pinned({})
    print(f"coarse=0, csa=off sha {got}")
    assert got == CSA_OFF_SHA, "csa_coarse_chunk=0 moved the default model"
    fine = _pinned({"use_csa": True})
    print(f"coarse=0, csa=fine-only sha {fine}")
    assert fine == CSA_ON_FINE_ONLY_SHA, "the fine level moved when coarse is off"
    causal = _pinned({"use_csa": True, "use_engram": False, "moe_topk": 0})
    assert causal == CSA_ON_FINE_ONLY_CAUSAL_SHA
    # Explicit off spelling is the default spelling.
    assert _pinned({"use_csa": True, "csa_coarse_chunk": 0, "csa_fine": True}) == fine


# --- 2. causality, each level and the pair ---------------------------------


def test_gate_2_coarse_is_causal_both_ways():
    """Engram OFF: attention is the only op that can read another position.

    A mutated byte may change its own chunk's pooled entry, but that entry is
    readable only by LATER positions; every output at `< t` must be
    bit-identical. The same mutation must move the output AT `t`, because the
    query at `t` is built from that byte.
    """
    torch.manual_seed(0)
    ids, _ = _batch()
    for name, over in (
        ("fine-only", {"csa_fine": True, "csa_coarse_chunk": 0}),
        ("coarse-only", {"csa_fine": False, "csa_coarse_chunk": 8}),
        ("both", {"csa_fine": True, "csa_coarse_chunk": 8}),
    ):
        model = DormouseModel(
            tiny_cfg(
                use_csa=True,
                use_engram=False,
                moe_topk=0,
                csa_chunk=4,
                csa_topk=8,
                **over,
            )
        )
        with torch.no_grad():
            base = model(ids)[0].view(BATCH, SEQ, -1)
            later = ids.clone()
            later[:, 8] = (later[:, 8] + 1) % 251
            got = model(later)[0].view(BATCH, SEQ, -1)
            assert torch.equal(base[:, :8], got[:, :8]), f"{name}: a position read a future byte"
            assert not torch.allclose(base[:, 8], got[:, 8]), (
                f"{name}: position 8 ignored its own byte — the query is not live"
            )
            last = ids.clone()
            last[:, -1] = (last[:, -1] + 7) % 251
            moved = model(last)[0].view(BATCH, SEQ, -1)
            assert torch.equal(base[:, :-1], moved[:, :-1]), f"{name}: tail leaked back"
            # A byte INSIDE the same fine/coarse chunk as position 8 still must
            # not move position 8: that chunk is not readable until a later chunk.
            mid = ids.clone()
            mid[:, 9] = (mid[:, 9] + 5) % 251
            m_got = model(mid)[0].view(BATCH, SEQ, -1)
            assert torch.equal(base[:, :9], m_got[:, :9]), (
                f"{name}: a byte in the same chunk leaked into an earlier position"
            )
            assert not torch.allclose(base[:, 9], m_got[:, 9]), (
                f"{name}: position 9 ignored its own byte"
            )
            print(f"{name}: pos-8 max delta {float((base[:, 8] - got[:, 8]).abs().max()):.3e}")
        # A mutation at a COARSE boundary (16, chunk 8) must not move earlier
        # positions either; that is the ring's append-after-read rule.
        if over["csa_coarse_chunk"]:
            with torch.no_grad():
                boundary = ids.clone()
                boundary[:, 16] = (boundary[:, 16] + 3) % 251
                b_got = model(boundary)[0].view(BATCH, SEQ, -1)
            assert torch.equal(base[:, :16], b_got[:, :16]), f"{name}: coarse boundary"


# --- 3. parity: coarse alone is exact strict-causal attention -------------


def _exact_ref(q, k, v, out_proj, n: int) -> torch.Tensor:
    ref = torch.zeros_like(q)
    for p in range(1, n):  # strictly earlier positions only (no diagonal)
        s = (q[:, p : p + 1] @ k[:, :p].transpose(1, 2)).squeeze(1) / math.sqrt(q.shape[-1])
        w = torch.softmax(s, dim=-1)
        ref[:, p] = (w.unsqueeze(-1) * v[:, :p]).sum(1)
    return out_proj(ref)


def test_gate_3_coarse_alone_equals_exact_attention():
    torch.manual_seed(0)
    cfg = tiny_cfg(
        use_csa=True,
        csa_chunk=1,
        csa_coarse_chunk=1,
        csa_fine=False,
        csa_topk=1,
    )
    attn = CompressedSelectiveAttention(cfg)
    x = torch.randn(BATCH, 16, cfg.d_model)
    out = attn(x)
    q = attn.W_q_c(x)
    k = attn.W_k_c(x)
    v = attn.W_v_c(x)
    ref = _exact_ref(q, k, v, attn.W_o_c, 16)
    err = float((out - ref).detach().abs().max())
    print(f"max |coarse - strict causal softmax| = {err:.3e}")
    assert err <= 1e-5, err

    # And state the sum honestly: with BOTH levels exact the branch returns
    # W_o(fine) + W_o_c(coarse), two independent exact attentions.
    cfg2 = dataclasses.replace(cfg, csa_fine=True, csa_topk=16)
    both = CompressedSelectiveAttention(cfg2)
    got = both(x)
    fine_ref = _exact_ref(both.W_q(x), both.W_k(x), both.W_v(x), both.W_o, 16)
    coarse_ref = _exact_ref(both.W_q_c(x), both.W_k_c(x), both.W_v_c(x), both.W_o_c, 16)
    sum_err = float((got - (fine_ref + coarse_ref)).detach().abs().max())
    print(f"max |both - (W_o·fine + W_o_c·coarse)| = {sum_err:.3e}")
    assert sum_err <= 1e-5, sum_err


# --- 4. gradients reach W_k/W_v of BOTH levels and the shared gate --------


def test_gate_4_gradients_reach_both_levels_and_the_gate():
    torch.manual_seed(0)
    model = DormouseModel(
        tiny_cfg(
            use_csa=True,
            use_engram=False,
            moe_topk=0,
            csa_chunk=4,
            csa_topk=8,
            csa_coarse_chunk=8,
            csa_coarse_kv_dim=64,
        )
    )
    ids, targets = _batch()
    _, loss, _ = model(ids, targets)
    assert loss is not None
    loss.backward()
    wanted = {
        "loop.csa.W_k.weight": model.loop.csa.W_k.weight.grad,
        "loop.csa.W_v.weight": model.loop.csa.W_v.weight.grad,
        "loop.csa.W_k_c.weight": model.loop.csa.W_k_c.weight.grad,
        "loop.csa.W_v_c.weight": model.loop.csa.W_v_c.weight.grad,
        "loop.csa.W_q_c.weight": model.loop.csa.W_q_c.weight.grad,
        "loop.csa_gate": model.loop.csa_gate.grad,
    }
    for name, grad in wanted.items():
        assert grad is not None and torch.isfinite(grad).all(), f"{name}: no finite gradient"
        mag = float(grad.abs().sum())
        assert mag > 0.0, f"{name}: zero gradient — the level runs and trains nothing"
        print(f"{name:24s} |grad|_1 = {mag:.6f}")


# --- 5. the coarse level is live ------------------------------------------


def test_gate_5_turning_the_coarse_level_on_changes_the_output():
    torch.manual_seed(0)
    cfg = tiny_cfg(use_csa=True, csa_chunk=4, csa_topk=8, csa_coarse_chunk=8)
    attn = CompressedSelectiveAttention(cfg)
    x = torch.randn(BATCH, 32, cfg.d_model)
    with torch.no_grad():
        both = attn(x)
        attn.coarse_enabled = False
        fine = attn(x)
        assert not torch.allclose(both, fine), "the coarse level added nothing"
        attn.coarse_enabled = True
        attn.W_o_c.weight.zero_()
        zeroed = attn(x)
    delta = float((both - fine).abs().max())
    again = float((both - zeroed).abs().max())
    print(f"coarse on vs off max delta {delta:.3e}; zeroed W_o_c {again:.3e}")
    assert not torch.allclose(both, zeroed), "zeroing the coarse weights changed nothing"
    assert torch.allclose(zeroed, fine, atol=0.0, rtol=0.0), (
        "zeroed W_o_c must leave exactly the fine level's output"
    )


# --- 6. the refusals are LOUD ---------------------------------------------


def test_gate_6_unsupported_combinations_are_loud():
    with pytest.raises(LoudError, match="csa_coarse_chunk"):
        validate(tiny_cfg(use_csa=True, csa_coarse_chunk=100))
    with pytest.raises(LoudError, match="multiple"):
        # 100 is not a multiple of 8; checked before the allowed-set message so
        # the divisibility rule is reachable through the config seam.
        validate(tiny_cfg(use_csa=True, csa_chunk=8, csa_coarse_chunk=100))
    with pytest.raises(LoudError, match="csa_coarse_kv_dim"):
        validate(tiny_cfg(use_csa=True, csa_coarse_chunk=256, csa_coarse_kv_dim=32))
    with pytest.raises(LoudError, match="use_csa"):
        validate(tiny_cfg(use_csa=False, csa_coarse_chunk=256))
    with pytest.raises(LoudError, match="use_msa"):
        validate(tiny_cfg(use_csa=True, use_msa=True, csa_coarse_chunk=256))
    with pytest.raises(LoudError, match="csa_fine"):
        validate(tiny_cfg(use_csa=True, csa_fine=False, csa_coarse_chunk=0))
    with pytest.raises(LoudError, match="csa_coarse_chunk"):
        _validate_segment(
            tiny_cfg(use_csa=True, csa_coarse_chunk=256),
            TrainConfig(segment_len=32, seq_len=SEQ),
        )
    # The runtime seam: a hand-built module never calls `validate`.
    with pytest.raises(AssertionError, match="multiple"):
        CompressedSelectiveAttention(tiny_cfg(csa_chunk=8, csa_coarse_chunk=100))
    # A legal coarse run passes validation and the fine-off switch alone does too.
    validate(tiny_cfg(use_csa=True, csa_coarse_chunk=256))
    validate(tiny_cfg(use_csa=True, csa_fine=False, csa_coarse_chunk=256))
    print(
        "loud: non-multiple, non-allowed chunk, coarse width, coarse without "
        "use_csa, use_msa, csa_fine without coarse, segment_len>0, runtime multiple"
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
    cfg = tiny_cfg(csa_chunk=4, csa_coarse_chunk=8)
    attn = CompressedSelectiveAttention(cfg)
    x = torch.randn(BATCH, 32, cfg.d_model)
    counters = Counters()
    out = attn(x, counters=counters)
    assert torch.equal(torch.isfinite(out), torch.ones_like(out, dtype=torch.bool))
    assert counters["csa_forward"] == 1
    assert counters["csa_coarse_forward"] == 1
    _, st = attn.forward_state(x, None, counters=counters)
    assert isinstance(st, tuple) and st[0].n == 8 and st[1].n == 4
    _, st = attn.step_state(x[:, :1], st, counters=counters)
    assert st[1].count == 1
    print(
        f"no host read: csa_forward={counters['csa_forward']} "
        f"csa_coarse_forward={counters['csa_coarse_forward']} "
        f"csa_step={counters['csa_step']}"
    )


# --- 8. the memory table prints both levels and their sum -----------------


def test_gate_8_memory_table_is_both_rings_against_real_tensors():
    table = memory_table(csa_kv_dim=64, csa_chunk=8, csa_coarse_kv_dim=64, csa_coarse_chunk=256)
    expected = {
        1024: 32768 + 1024,
        32768: 1_048_576 + 32768,
        1_048_576: 33_554_432 + 1_048_576,
    }
    assert table == expected, table
    for context in (1024, 32768, 1_048_576):
        fine = CSAState(64, 64, 8, batch=1)
        fine.reserve(-(-context // 8))
        coarse = CSAState(64, 64, 256, batch=1)
        coarse.reserve(-(-context // 256))
        real = fine.bytes() + coarse.bytes()
        assert fine.k.element_size() == coarse.k.element_size() == 2
        assert (
            real
            == table[context]
            == total_state_bytes(context, 64, 8, csa_coarse_kv_dim=64, csa_coarse_chunk=256)
        ), (context, real, table[context])
    print(
        f"1M: fine {expected[1_048_576] - 1_048_576} B + coarse 1_048_576 B "
        f"= {expected[1_048_576] / 2**20:.1f} MiB (real tensors agree)"
    )
    assert expected[1_048_576] == 34_603_008


# --- 9. the 1M decode bill is printed and the coarse level is cheap -------


def test_gate_9_coarse_decode_is_under_budget_at_1m():
    bill = decode_cost_table()
    assert bill["coarse"] < 3_000_000, bill
    assert bill["coarse"] < bill["fine"], bill
    # Every allowed coarse chunk at the default width is under the budget.
    for chunk in (64, 128, 256, 512):
        got = mac_per_byte(1_048_576, csa_coarse_chunk=chunk, csa_coarse_kv_dim=64)
        assert got["coarse"] < 3_000_000, (chunk, got)
    dense = 2 * 1_048_576 * 64  # exact attention, one head, d=64
    print(
        f"dense exact 1-head attention at 1M (d=64): {dense:,} MAC/byte; "
        f"coarse/fine = {bill['coarse'] / bill['fine']:.1%} of the fine bill, "
        f"{bill['coarse'] / dense:.2%} of dense"
    )
    assert bill["coarse"] < dense
