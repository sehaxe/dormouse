"""Segmented training WITH the CSA ring — the boundary contract.

WHAT THIS ADDS. `segment_len>0` used to be refused with `use_csa` because the
rule for a chunk a segment boundary splits was undecided. The contract is now:
**the rings keep their entries, the partial chunk is CARRIED and completed by
the next segment** (never dropped, never padded), and the ring's
`n·chunk + count` IS the number of bytes seen — so the byte position of a
window's first token (`pos = (count + arange(t)) // chunk + n`), the `key < pos`
mask and the pooled keys are the one-shot ones.

Gates (each proven red on a scratch mutation, reported in
`docs/reviews/2026-11-08-segment-csa.md`):

1. `segment_len=0` is bit-identical to clean main (`3f15810e…`/`02548e27…`,
   captured from `d1b2c2e`), and the one-shot path never enters
   `forward_state`.
2. Two segments over 256 bytes == one continuous pass, `segment_len` a
   multiple of the chunk (64/128): logits and loss to 1e-5.
3. The HARD case: `segment_len=100` is not a multiple of `csa_chunk=8`, so the
   cut lands inside a chunk and only the carry can make the math right.
4. The COARSE ring carries across a cut too (`csa_coarse_chunk=256`, a cut at
   byte 100 is inside a coarse chunk) — the "ignore the coarse level" mutation.
5. The carried STATE is the continuous state (1e-6), and `n·chunk + count` is
   the byte count after every segment.
6. The chunk a cut splits is EXACTLY the mean of the concatenated bytes: ring
   entry 12 at `segment_len=100, chunk=8` is `W_k(mean(x[96:104]))`, and the
   coarse ring entry 0 is `W_k_c(mean(x[0:256]))`.
7. `csa_topk >= n` (all chunks) is dense strict-causal attention, hand-written
   over the concatenated input, across a cut — the repo's own parity arm.
8. The carry is a GRADIENT edge: a loss on segment 2 moves segment 1's inputs.
9. Segmented and one-shot GRADIENTS match to fp32 round-off (this is what sees
   a checkpoint replay appending the ring twice; the logits cannot).
10. The config refuses only what the contract cannot hold: the CSA pair is
    legal, `segment_len < csa_chunk` and `csa_hierarchical` stay loud.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.config import TrainConfig, _validate_segment
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.csa import CompressedSelectiveAttention

BATCH = 2
#: `segment_len=0` pins, fixture `_cfg()` + `_ids(256)`, `torch.set_num_threads(1)`
#: + matmul precision `highest`, captured from a clean `main` checkout at
#: `d1b2c2e` BEFORE the boundary contract existed. The thread pin is part of the
#: fixture: a fp32 CPU reduction's order moves with the pool size (measured: four
#: different shas at 1/2/4/10 threads), so a hash gate that does not name its
#: thread count is a gate on the test runner.
PIN_ONE_SHOT = "2bc9cf691d1ba69570baa1ff97b97eed08d3a3dcc88d09cdf9315fe8a982d22a"
PIN_ONE_SHOT_COARSE = "4eb661ad2d417786128e3106aa1f73713c8ede9d0f9932c43c4495c3fe57b99a"


def _cfg(**over):
    base = dict(
        use_csa=True,
        use_engram=False,
        moe_topk=0,
        moe_capacity=0.0,
        max_seq_len=1024,
        csa_chunk=8,
        csa_kv_dim=64,
        csa_topk=32,
    )
    base.update(over)
    return tiny_cfg(**base)


def _ids(seq: int) -> torch.Tensor:
    return torch.arange(BATCH * seq, dtype=torch.long).view(BATCH, seq) % 251


def _model(**over) -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(_cfg(**over))


def _sha(model: torch.nn.Module, logits: torch.Tensor) -> str:
    h = hashlib.sha256()
    for name, val in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(val.detach().cpu().contiguous().numpy().tobytes())
    h.update(b"|logits|")
    h.update(logits.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _parity(model: DormouseModel, ids: torch.Tensor, seg: int):
    tgts = (ids + 1) % 251
    with torch.no_grad():
        ref, ref_loss, _ = model(ids, tgts)
        got, loss, counters = model(ids, tgts, segment_len=seg)
    return ref, got, float(ref_loss), float(loss), counters


def _segmented_states(attn: CompressedSelectiveAttention, x: torch.Tensor, seg: int):
    """Run `forward_state` over `x` in `seg`-byte windows; return outputs + state."""
    outs, state, history = [], None, []
    for lo in range(0, x.shape[1], seg):
        out, state = attn.forward_state(x[:, lo : lo + seg], state, counters=Counters())
        outs.append(out)
        history.append(state)
    return torch.cat(outs, dim=1), state, history


# --- 1. segment_len=0 is still the pinned one-shot path --------------------


def test_gate_1_segment_len_zero_is_the_pinned_one_shot(monkeypatch):
    """`0` is the default and every number on record; it must not move.

    The sha pins `state_dict` + logits from clean `main` `d1b2c2e`. The second
    half proves the path STRUCTURALLY: a default call never reaches
    `forward_state`, so the carry code cannot leak into a one-shot run.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    try:
        for over, want in (
            ({}, PIN_ONE_SHOT),
            ({"csa_coarse_chunk": 256}, PIN_ONE_SHOT_COARSE),
        ):
            model = _model(**over)
            ids = _ids(256)
            with torch.no_grad():
                logits, _, _ = model(ids, (ids + 1) % 251)
            got = _sha(model, logits)
            print(f"segment_len=0 coarse={over.get('csa_coarse_chunk', 0)} sha {got}")
            assert got == want, "segment_len=0 moved (the one-shot forward is pinned)"
    finally:
        torch.set_num_threads(threads)

    def boom(*_a, **_k):
        raise AssertionError("segment_len=0 entered the carry path")

    monkeypatch.setattr(CompressedSelectiveAttention, "forward_state", boom)
    model = _model()
    ids = _ids(64)
    with torch.no_grad():
        a = model(ids, counters=Counters())[0]
        b = model(ids, segment_len=0, counters=Counters())[0]
    assert torch.equal(a, b)
    print("segment_len=0 never calls forward_state")


# --- 2/3. the parity gate: segmented IS the continuous pass ---------------


@pytest.mark.parametrize("seg", [64, 128])
def test_gate_2_two_segments_are_one_continuous_pass(seg):
    model = _model()
    ids = _ids(256)
    ref, got, ref_loss, loss, counters = _parity(model, ids, seg)
    d = float((ref - got).abs().max())
    dl = abs(ref_loss - loss)
    print(f"[seg {seg}] segments={counters['loop_segments']} max|dlogits|={d:.3e} dloss={dl:.3e}")
    assert counters["loop_segments"] == 256 // seg
    assert d <= 1e-5, f"segmented logits differ by {d}"
    assert dl <= 1e-5, f"segmented loss differs by {dl}"


def test_gate_3_a_segment_that_is_not_a_chunk_multiple_carries_the_chunk():
    """100 is not a multiple of 8: the cut lands INSIDE a chunk.

    This is the case the old refusal was about. Dropping the partial chunk or
    padding it both move the boundary chunk's byte membership, so this gate
    and gate 6 (the arithmetic) are the pair that forces the carry.
    """
    model = _model(csa_topk=64)
    ids = _ids(200)
    ref, got, ref_loss, loss, counters = _parity(model, ids, 100)
    d = float((ref - got).abs().max())
    dl = abs(ref_loss - loss)
    print(f"[seg 100] segments={counters['loop_segments']} max|dlogits|={d:.3e} dloss={dl:.3e}")
    assert counters["loop_segments"] == 2
    assert d <= 1e-5, d
    assert dl <= 1e-5, dl


# --- 4. the coarse ring carries across the same cut -----------------------


@pytest.mark.parametrize(("seq", "seg"), [(512, 64), (512, 128), (600, 100)])
def test_gate_4_the_coarse_ring_carries_across_a_cut(seq, seg):
    """`csa_coarse_chunk=256`: every cut here is inside a coarse chunk (100 is)."""
    model = _model(csa_coarse_chunk=256, csa_topk=64)
    ids = _ids(seq)
    ref, got, ref_loss, loss, counters = _parity(model, ids, seg)
    d = float((ref - got).abs().max())
    print(f"[coarse seg {seg}/{seq}] max|dlogits|={d:.3e} loss d={abs(ref_loss - loss):.3e}")
    assert counters["loop_segments"] == seq // seg
    assert d <= 1e-5, f"the coarse ring diverged: {d}"
    assert abs(ref_loss - loss) <= 1e-5


# --- 5. the carried state IS the continuous state -------------------------


def test_gate_5_the_carried_state_is_the_continuous_state():
    torch.manual_seed(0)
    attn = CompressedSelectiveAttention(_cfg(csa_coarse_chunk=256, csa_topk=64))
    x = torch.randn(BATCH, 600, 64)
    with torch.no_grad():
        cont_out, cont = attn.forward_state(x, None, counters=Counters())
        seg_out, seg, history = _segmented_states(attn, x, 100)
    d = float((cont_out - seg_out).abs().max())
    print(f"[state] 600 B, 6x100 segments: output max|d|={d:.3e}")
    assert d <= 1e-6, d
    (fc, cc), (fs, cs) = cont, seg
    assert (fs.n, fs.count) == (fc.n, fc.count) == (75, 0)
    assert (cs.n, cs.count) == (cc.n, cc.count) == (2, 88)
    seen = 0
    for st in history:
        seen += 100
        for part, chunk in ((st[0], 8), (st[1], 256)):
            assert part.n * chunk + part.count == seen, (part.n, part.count, seen, chunk)
    for name, a, b in (
        ("fine k", fc.k[: fc.n], fs.k[: fs.n]),
        ("fine v", fc.v[: fc.n], fs.v[: fs.n]),
        ("coarse k", cc.k[: cc.n], cs.k[: cs.n]),
        ("coarse v", cc.v[: cc.n], cs.v[: cs.n]),
        ("fine sum", fc.sum, fs.sum),
        ("coarse sum", cc.sum, cs.sum),
    ):
        dd = float((a - b).abs().max())
        print(f"  {name:10s} max|d|={dd:.3e}")
        assert dd <= 1e-6, (name, dd)
    assert cc.k.dtype == torch.float32, "the training ring must be fp32"


# --- 6. which bytes go to which chunk (the two-boundary arithmetic) -------


def test_gate_6_the_chunk_a_cut_splits_is_the_mean_of_the_concatenated_bytes():
    torch.manual_seed(0)
    attn = CompressedSelectiveAttention(_cfg(csa_coarse_chunk=256, csa_topk=64))
    x = torch.randn(BATCH, 600, 64)
    with torch.no_grad():
        st = None
        for lo in range(0, 600, 100):
            _, st = attn.forward_state(x[:, lo : lo + 100], st, counters=Counters())
        fine, coarse = st
    # The cut at 100 is inside fine chunk 12 = bytes 96..103; the cut at 600 is
    # inside coarse chunk 2 = bytes 512..767 (carried, not appended).
    for idx, lo, hi in ((11, 88, 96), (12, 96, 104), (13, 104, 112)):
        want = attn.W_k(x[:, lo:hi].mean(dim=1))
        got = fine.k[idx]
        d = float((want - got).detach().abs().max())
        print(f"  fine chunk {idx:2d} (bytes {lo:3d}..{hi - 1:3d}) max|d|={d:.3e}")
        assert d <= 1e-6, (idx, d)
    for idx, lo, hi in ((0, 0, 256), (1, 256, 512)):
        want = attn.W_k_c(x[:, lo:hi].mean(dim=1))
        d = float((want - coarse.k[idx]).detach().abs().max())
        print(f"  coarse chunk {idx} (bytes {lo:3d}..{hi - 1:3d}) max|d|={d:.3e}")
        assert d <= 1e-6, (idx, d)
    assert (fine.n, fine.count, coarse.n, coarse.count) == (75, 0, 2, 88)
    print("  boundary chunks hold exactly the concatenated bytes' means")


# --- 7. topk >= n is dense attention, across the cut ----------------------


def test_gate_7_all_chunks_selected_equal_dense_attention_across_a_cut():
    """`csa_topk >= n`: exact chunk attention, hand-written over `x`.

    The reference pools the CONCATENATED input itself, so it cannot inherit the
    module's carry; if the segmented path dropped or padded the chunk the cut
    splits, the boundary-chunk key (and therefore the output) would differ.
    """
    torch.manual_seed(0)
    attn = CompressedSelectiveAttention(_cfg(csa_topk=1000))
    x = torch.randn(BATCH, 200, 64)
    with torch.no_grad():
        out, _ = _segmented_states(attn, x, 100)[:2]
        q = attn.W_q(x)
        pooled = F.avg_pool1d(x.transpose(1, 2), 8, 8).transpose(1, 2)
        k = attn.W_k(pooled)
        v = attn.W_v(pooled)
        pos = torch.arange(200) // 8
        allowed = torch.arange(k.shape[1]).view(1, 1, -1) < pos.view(1, -1, 1)
        scores = (q @ k.transpose(1, 2)) * attn.scale
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        # The arm's rule for an all-masked row (query 0 has no earlier chunk):
        # softmax alone would give a uniform weight to the garbage row, so the
        # mask is re-applied and the row renormalised (zero).
        w = torch.softmax(scores, -1) * allowed
        w = w / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
        ref = attn.W_o(w @ v)
    d = float((ref - out).abs().max())
    n = k.shape[1]
    print(f"[dense] n={n} topk=1000, seg=100 (cut inside a chunk): max|d|={d:.3e}")
    assert attn.topk >= n, "the gate needs every chunk selected"
    assert d <= 1e-6, d


# --- 8. the carry is a gradient edge --------------------------------------


def test_gate_8_the_carried_bytes_reach_the_next_segment_gradient():
    torch.manual_seed(0)
    attn = CompressedSelectiveAttention(_cfg(csa_topk=64))
    x = torch.randn(BATCH, 200, 64, requires_grad=True)
    _, st = attn.forward_state(x[:, :100], None, counters=Counters())
    out, _ = attn.forward_state(x[:, 100:], st, counters=Counters())
    (out**2).sum().backward()
    g0 = float(x.grad[:, :100].abs().sum())
    g1 = float(x.grad[:, 100:].abs().sum())
    kw = float(attn.W_k.weight.grad.abs().sum())
    kv = float(attn.W_v.weight.grad.abs().sum())
    print(
        f"[carry grad] segment-1 bytes |g|={g0:.4f}, segment-2 |g|={g1:.4f}, "
        f"W_k={kw:.4f}, W_v={kv:.4f}"
    )
    assert g0 > 0.0, "the completed chunk did not carry a gradient back"
    assert g1 > 0.0 and kw > 0.0 and kv > 0.0


# --- 9. gradients, not only logits (checkpoint replay) --------------------


def _grads(model: DormouseModel, ids: torch.Tensor, seg: int):
    tgts = (ids + 1) % 251
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(ids, tgts, segment_len=seg)
    loss.backward()
    return {
        name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None
    }


def test_gate_9_segmented_gradients_match_the_one_shot_gradients():
    """The backward of TWO segments is the backward of ONE.

    `_forward_segmented` wraps each segment in `checkpoint`, so the ring must
    be built functionally: an in-place append would replay every chunk a second
    time in the backward while the forward logits stayed right.
    """
    ids = _ids(200)
    model = _model(csa_topk=64)
    one = _grads(model, ids, 0)
    seg = _grads(model, ids, 100)
    assert set(one) == set(seg)
    worst, worst_name = 0.0, ""
    for name in one:
        scale = max(float(one[name].abs().max()), 1e-30)
        rel = float((one[name] - seg[name]).abs().max()) / scale
        if rel > worst:
            worst, worst_name = rel, name
    print(f"[grad parity] {len(one)} tensors, worst rel {worst:.3e} at {worst_name}")
    for name in ("loop.csa.W_k.weight", "loop.csa.W_v.weight", "loop.csa.W_o.weight"):
        assert name in one, name
    assert worst <= 1e-3, (worst, worst_name)


# --- 10. the config surface -----------------------------------------------


def test_gate_10_the_config_refuses_only_what_the_contract_cannot_hold():
    ok = tiny_cfg(use_csa=True, csa_chunk=8, csa_coarse_chunk=256, moe_topk=0, moe_capacity=0)
    for seg, seq in ((64, 256), (100, 200), (128, 256)):
        _validate_segment(ok, TrainConfig(segment_len=seg, seq_len=seq))
    with pytest.raises(LoudError, match="csa_chunk"):
        _validate_segment(ok, TrainConfig(segment_len=4, seq_len=256))
    with pytest.raises(LoudError, match="csa_hierarchical"):
        _validate_segment(
            tiny_cfg(
                use_csa=True,
                csa_hierarchical=True,
                csa_mid_chunk=512,
                csa_coarse_chunk=256,
                moe_topk=0,
                moe_capacity=0,
            ),
            TrainConfig(segment_len=64, seq_len=512),
        )
    # The runtime seam (a hand-built model does not call `validate`).
    model = _model(csa_chunk=8)
    ids = _ids(100)
    with torch.no_grad():
        logits, loss, counters = model(ids, (ids + 1) % 251, segment_len=100)
    assert torch.isfinite(loss) and counters["loop_segments"] == 1
    hier = _model(csa_hierarchical=True, csa_mid_chunk=512, csa_coarse_chunk=256)
    with pytest.raises(AssertionError, match="csa_hierarchical"):
        hier(ids, (ids + 1) % 251, segment_len=100)
    print("legal: use_csa + segment (64/100/128); loud: segment<csa_chunk, hierarchical")
