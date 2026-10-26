"""Fast decode: the CUDA-graph step and the MTP multibyte draft, gated.

Three claims, each with the check that fails when it is reverted:

1. **Flags off is the old decoder, bit for bit.** `test_flags_off_pins_the_transcript`
   pins a sha256 of the full byte string for a fixed model seed, prompt and
   sampling seed. The pin was taken on `main` before this arm landed; a change
   to the eager path shows up as a different hash, not as a review comment.
2. **The graph is the same step.** `test_graph_replays_consecutive_steps` runs
   the captured step for 12 bytes and compares to eager, and asserts the replay
   counter (so a fallback that silently went eager cannot pass it).
3. **The block verify is real.** `test_multibyte_matches_the_one_byte_loop` (k in
   {2,4}) compares against the one-byte loop at temp=1e-6, and
   `test_a_broken_draft_still_emits_the_correct_bytes` makes the draft propose
   garbage — the acceptance rule must reject it and the bytes must stay the
   reference's. That is the gate that falsifies "accept everything".

The graph tests need CUDA (a hash of two CPU runs proves nothing about a
capture); they skip on a CPU-only box like the rest of the port's CUDA gates.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from conftest import tiny_model

from dormouse.instrument import Counters, LoudError
from dormouse.model.decode import (
    _GraphStep,
    _state_refusal,
    _step_logits,
    decode_graph_refusal,
    recurrent_decode,
)

#: `main` @ 8733cc0, tiny model seeded 1234, prompt b"pin this transcript",
#: 12 new bytes, temp 0.8, generator seeded 7. Computed on the BASELINE worktree.
TRANSCRIPT_SHA256 = "0734d296c948f213920bc4aea113e8cc75ce4897c6650938caf62f0d11c455d9"
PROMPT = b"pin this transcript"


def _tiny_mtp(**over):
    """A tiny model with the MTP head built (it exists iff mtp_weight > 0)."""
    return tiny_model(mtp_weight=0.1, mtp_k=3, **over)


def _one_byte(model, prompt: bytes, n: int, **kw) -> bytes:
    dev = next(model.parameters()).device
    return recurrent_decode(
        model,
        prompt,
        n,
        temp=1e-6,
        device=dev,
        generator=torch.Generator(device=dev).manual_seed(7),
        **kw,
    )


# ------------------------------------------------------------ (1) flags off


def test_flags_off_pins_the_transcript():
    """The default decoder's bytes, pinned; the two fast arms are opt-in."""
    torch.manual_seed(1234)
    model = tiny_model()
    model.eval()
    counters = Counters()
    out = recurrent_decode(
        model,
        PROMPT,
        12,
        temp=0.8,
        generator=torch.Generator().manual_seed(7),
        counters=counters,
    )
    assert hashlib.sha256(out).hexdigest() == TRANSCRIPT_SHA256, (
        "the flags-off transcript moved: the eager decode path is not the one main ran"
    )
    # No fast-arm counter may appear when the flags are off: a silent arm would
    # show up as a replay/draft on an untouched call.
    assert not [k for k in counters if k.startswith(("decode_graph", "decode_multibyte"))]


# ----------------------------------------------------- (2) the multibyte gate


@pytest.mark.parametrize("k", [2, 4])
def test_multibyte_matches_the_one_byte_loop(k):
    """Greedy, a verified block emits exactly the one-byte loop's bytes."""
    torch.manual_seed(11)
    model = _tiny_mtp(moe_topk=0)  # dropless: the verify block IS prefix-causal
    model.eval()
    ref = _one_byte(model, b"hello once", 12)
    counters = Counters()
    out = recurrent_decode(
        model,
        b"hello once",
        12,
        temp=1e-6,
        generator=torch.Generator().manual_seed(7),
        counters=counters,
        decode_multibyte=k,
    )
    assert out == ref, f"k={k} changed the bytes:\n got {out!r}\n ref {ref!r}"
    assert counters["decode_multibyte"] == k
    # A round that ends on the byte budget never reaches the carry (no full, no
    # repair) — so reject is the round count or one past the repairs.
    assert counters["decode_multibyte_round"] > 0
    assert counters["decode_multibyte_reject"] - counters["decode_multibyte_repair"] in (0, 1)
    assert (
        counters.get("decode_multibyte_full", 0) + counters["decode_multibyte_repair"]
        <= counters["decode_multibyte_round"]
    )


def test_multibyte_honours_a_stop_byte():
    """The stop contract survives the block walk: both decoders end at the same byte."""
    torch.manual_seed(14)
    model = _tiny_mtp(moe_topk=0)
    model.eval()
    full = _one_byte(model, b"stop me", 12)
    stop_byte = full[-3]  # a byte the one-byte loop really emits
    stop = frozenset({stop_byte})
    ref = recurrent_decode(
        model,
        b"stop me",
        12,
        temp=1e-6,
        generator=torch.Generator().manual_seed(7),
        stop=stop,
    )
    out = recurrent_decode(
        model,
        b"stop me",
        12,
        temp=1e-6,
        generator=torch.Generator().manual_seed(7),
        stop=stop,
        decode_multibyte=4,
    )
    assert out == ref
    assert out[-1] == stop_byte


def test_a_broken_draft_still_emits_the_correct_bytes(monkeypatch):
    """The gate that falsifies "accept everything": a wrong draft is REJECTED.

    The draft is monkeypatched to propose a byte that is never the target's
    argmax. The acceptance walk must reject at the first position, emit the
    target's byte, and the full string must still equal the one-byte loop — so a
    verify that merely copied the draft (or accepted by construction) fails here.
    """
    torch.manual_seed(12)
    model = _tiny_mtp(moe_topk=0)
    model.eval()
    ref = _one_byte(model, b"broken draft", 12)

    def wrong(model, head, h_last, first, n, counters):
        counters.count("decode_multibyte_draft", n)
        return [((first + 131) % 256)] * n

    monkeypatch.setattr("dormouse.model.decode._mtp_draft", wrong)
    counters = Counters()
    out = recurrent_decode(
        model,
        b"broken draft",
        12,
        temp=1e-6,
        generator=torch.Generator().manual_seed(7),
        counters=counters,
        decode_multibyte=4,
    )
    assert out == ref, f"a rejected draft changed the bytes:\n got {out!r}\n ref {ref!r}"
    assert counters["decode_multibyte_round"] > 0
    assert counters["decode_multibyte_reject"] == counters["decode_multibyte_round"]
    assert counters["decode_multibyte_repair"] in (
        counters["decode_multibyte_round"],
        counters["decode_multibyte_round"] - 1,
    )
    assert "decode_multibyte_full" not in counters


def test_a_one_token_dispatch_is_the_same_with_and_without_the_keep_mask():
    """The graph shortcut (`keep=None`) is exact: at n=1 the mask is all-True."""
    torch.manual_seed(31)
    model = tiny_model(moe_topk=2, moe_kernel="padded")
    moe = model.loop.moe
    x = torch.randn(1, model.cfg.d_model)
    index = torch.tensor([[0, 2]])
    weight = torch.tensor([[0.7, 0.3]])
    order, keep, dropped = moe._permute(index, None)
    assert bool(keep.all()) and int(dropped) == 0
    masked = moe._padded_dispatch(x, weight, index, order, keep)
    static = moe._padded_dispatch(x, weight, index, order, None)
    assert torch.equal(masked, static)


# --------------------------------------------------------------- refusals


def test_multibyte_refuses_sampling():
    model = _tiny_mtp(moe_topk=0)
    with pytest.raises(LoudError, match="GREEDY ONLY"):
        recurrent_decode(model, b"x", 2, temp=0.8, decode_multibyte=4)


def test_multibyte_refuses_one_byte():
    model = _tiny_mtp(moe_topk=0)
    with pytest.raises(LoudError, match="one byte per forward"):
        recurrent_decode(model, b"x", 2, temp=1e-6, decode_multibyte=1)


def test_multibyte_refuses_a_capacity_dropping_moe():
    """A block's drop set is not prefix-causal — the verify cannot be the loop."""
    model = _tiny_mtp(moe_topk=2)  # capacity 2.0 by default
    with pytest.raises(LoudError, match="DROP SET"):
        recurrent_decode(model, b"x", 2, temp=1e-6, decode_multibyte=4)


def test_multibyte_without_a_head_falls_back_counted():
    model = tiny_model(moe_topk=0)  # no MTP head
    counters = Counters()
    out = recurrent_decode(model, b"no head", 4, temp=1e-6, counters=counters, decode_multibyte=4)
    assert counters["decode_multibyte_fallback"] == 1
    assert "MTP head" in counters.why
    assert out == _one_byte(model, b"no head", 4)


def test_graph_refuses_batch_two():
    why = decode_graph_refusal(tiny_model(), batch=2)
    assert why is not None and "batch-1" in why


def test_graph_refuses_a_full_attention_arm():
    why = decode_graph_refusal(tiny_model(attn_kind="full", use_msa=False))
    assert why is not None and "attn_kind='full'" in why


def test_graph_refuses_a_state_that_is_not_a_tensor():
    why = _state_refusal([("k", "v")], torch.device("cpu"))
    assert why is not None and "fixed-size tensor" in why
    assert _state_refusal([], torch.device("cpu")) is None


def test_graph_refuses_a_cpu_device():
    model = tiny_model()
    with pytest.raises(LoudError, match="CUDA device"):
        recurrent_decode(model, b"cpu graph", 2, temp=1e-6, decode_graph=True)


# --------------------------------------------------------------- CUDA gates

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
def test_graph_replays_consecutive_steps():
    """At least 3 CONSECUTIVE replayed steps, same bytes as eager."""
    torch.manual_seed(21)
    model = tiny_model().to("cuda")
    model.eval()
    ref = _one_byte(model, b"graph steps", 12)
    counters = Counters()
    out = recurrent_decode(
        model, b"graph steps", 12, temp=1e-6, device="cuda", counters=counters, decode_graph=True
    )
    assert out == ref
    assert counters["decode_graph_replay"] >= 3
    assert "decode_graph_fallback" not in counters


@needs_cuda
def test_graph_advances_the_state_in_place_on_the_torch_step(monkeypatch):
    """The state writeback is what advances the replay; force the non-aliasing arm.

    The FLA step happens to hand back its `initial_state` buffer, so a graph
    without the in-place writeback can still look right on this box. The torch
    step returns a NEW tensor, so this test is the one that fails when the
    writeback is removed.
    """
    monkeypatch.setattr(
        "dormouse.model.kda.fla_status", lambda *a, **k: (False, "test: force the torch step")
    )
    torch.manual_seed(24)
    model = tiny_model().to("cuda")
    model.eval()
    # `prefill="byte"` too: the chunked prefill's carry arm is the SDPA fallback
    # (a growing (k, v) tuple), not the torch recurrence, when FLA is off.
    ref = _one_byte(model, b"torch graph", 8, prefill="byte")
    counters = Counters()
    out = recurrent_decode(
        model,
        b"torch graph",
        8,
        temp=1e-6,
        device="cuda",
        counters=counters,
        prefill="byte",
        decode_graph=True,
    )
    assert out == ref
    assert counters["decode_graph_replay"] >= 3


@needs_cuda
def test_graph_writes_the_state_back_in_place(monkeypatch):
    """The replay must ADVANCE its own state buffers, not re-read the input state.

    The torch step is forced (the FLA kernel happens to hand back its input
    buffer on this card, which hides a missing writeback). Comparing the graph's
    state after two replays to the eager state after the same two bytes fails on
    a graph that only copies its inputs in.
    """
    monkeypatch.setattr(
        "dormouse.model.kda.fla_status", lambda *a, **k: (False, "test: force the torch step")
    )
    torch.manual_seed(25)
    model = tiny_model().to("cuda")
    model.eval()
    dev = torch.device("cuda")
    drop = 3
    prompt = list(b"state writeback")
    states: list = [None] * model.cfg.max_iter
    for i in range(len(prompt)):
        logits, states = _step_logits(model, prompt[: i + 1], states, dev, drop, Counters())
    ctx = prompt + [int(logits.argmax())]
    lg1, st1 = _step_logits(model, ctx, [s.clone() for s in states], dev, drop, Counters())
    ctx2 = ctx + [int(lg1.argmax())]
    lg2, st2 = _step_logits(model, ctx2, [s.clone() for s in st1], dev, drop, Counters())

    graph = _GraphStep(model, dev, drop, 1, Counters())
    glogits, _hidden, gstates = graph.step(ctx, [s.clone() for s in states])
    assert torch.allclose(glogits[-1], lg1, atol=1e-5)
    glogits, _hidden, gstates = graph.step(ctx2, gstates)
    assert torch.allclose(glogits[-1], lg2, atol=1e-5)
    for i, (a, b) in enumerate(zip(gstates, st2, strict=True)):
        assert torch.equal(a, b), f"state slot {i} was not advanced in place by the replay"


@needs_cuda
def test_multibyte_graph_matches_the_one_byte_loop():
    torch.manual_seed(22)
    model = _tiny_mtp(moe_topk=0).to("cuda")
    model.eval()
    ref = _one_byte(model, b"block graph", 12)
    counters = Counters()
    out = recurrent_decode(
        model,
        b"block graph",
        12,
        temp=1e-6,
        device="cuda",
        counters=counters,
        decode_multibyte=4,
        decode_graph=True,
    )
    assert out == ref
    assert counters["decode_graph_replay"] >= 3
    assert "decode_graph_fallback" not in counters


@needs_cuda
def test_a_capture_failure_is_counted_and_stays_eager(monkeypatch):
    """A capture that dies must not kill the decode: COUNTED, then eager bytes."""
    torch.manual_seed(23)
    model = tiny_model().to("cuda")
    model.eval()

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("capture refused by the test")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Boom)
    counters = Counters()
    out = recurrent_decode(
        model, b"capture boom", 6, temp=1e-6, device="cuda", counters=counters, decode_graph=True
    )
    monkeypatch.undo()
    assert counters["decode_graph_fallback"] == 1
    assert "capture refused by the test" in counters.why
    assert out == _one_byte(model, b"capture boom", 6)
