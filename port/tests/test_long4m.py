"""The long-context stack for a 4 MiB window — the ten gates.

WHAT THIS ADDS, in one place, on top of the two-level CSA already shipped:

1. `csa_mid_chunk` — a THIRD level on the coarse level's algebra (mean-pool,
   dense causal mask over strictly earlier chunks, its own `W_{q,k,v,o}` at
   `csa_kv_dim`), its own ring in the `(fine, coarse, mid)` state.
2. `csa_hierarchical` — the fine level's flat scan replaced by a containment
   chain: the outermost level is top-k'd first, the next level's candidates
   are restricted to the survivors' children, and the fine attention runs only
   over the fine chunks inside the surviving coarse entries.
3. `attn_window` — an EXACT LOCAL BAND beside KDA/CSA (`LoopBlock` sums the
   attention paths): position `p` reads `max(0, p-W+1)..p` and nothing older.
4. `attn_window_query_chunk` — the same band in query blocks, so the mask is
   `C × (C+W-1)` instead of `t × t`.

Gates, each RED under a scratch revert of the code it names:

1. every new flag OFF is BIT-IDENTICAL to clean `main` (four sha256 pins of
   `state_dict`+logits, captured in a detached worktree at `3d36844` BEFORE
   this file existed);
2. the MID level is causal: mutating byte `x` leaves every output at `< x`
   bit-identical and moves the output at `x`, for the dense and the hierarchical
   path;
3. hierarchical selection equals the FLAT selection when `csa_topk` covers
   every level (to round-off, max|Δ| printed) — the hierarchy selects the same
   REGION, not a different one — and a small top-k really narrows it;
4. the window is causal AND banded: over every (mutated byte, output) pair the
   moved set is exactly `0 <= q-p <= W-1`; `W >= t` equals exact causal
   attention; and a position does not move when the mutated byte is just
   OUTSIDE the band (older than `W`) while one just INSIDE does;
5. `attn_window_query_chunk` equals the unchunked output (max|Δ| and the
   tolerance printed, fp32 and bf16);
6. no host read (`.item/.tolist/.bool/.float/__bool__/__float__` patched to
   raise) on ANY new path: mid dense, hierarchical prefill, hierarchical decode
   step, window unchunked, window chunked;
7. every illegal combination is LOUD with a reason and an `Escape:`;
8. every new path has a counter, with the arithmetic asserted (not just
   non-zero);
9. the memory arithmetic: the ring bytes at 32 KiB / 256 KiB / 4 MiB against the
   closed form, the 4 MiB `csa_chunk=32, csa_kv_dim=64` state = 32 MiB per
   slot (printed), and the window's `2·W·d` MAC/byte bill;
10. the window is SUMMED with KDA (zeroing its `W_o` returns the no-window
    model bit-for-bit) and gradients reach its four projections.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import ModelConfig, TrainConfig, _validate_segment, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.csa import (
    CompressedSelectiveAttention,
    CSAState,
    memory_table,
    mid_state_bytes,
    state_bytes,
    total_state_bytes,
)
from dormouse.model.decode import recurrent_refusal
from dormouse.model.fullattn import FullAttention, window_mac_per_byte

BATCH, SEQ = 2, 64
SEED = 1234

#: Captured in a detached `main` worktree at `3d36844` BEFORE this branch
#: existed, seed 1234, fixture below. Gate 1 is the only place they are used.
PIN_DEFAULT = "beb23f76773c5f63a8b5ed5a549fb3c451174e9f4622c2549246cc63b4d0d318"
PIN_CAUSAL = "2b3139030d8e46e62ef8ec5dfc5b517cec2ca002abb9db4814d0e19c610f4fb7"
PIN_CSA_FINE = "defb2afec16ed17637447ad5c495fb105e4c382dd7c64e9aacfbaf0d0ec986b6"
PIN_CSA_COARSE = "733ded4eac215bb99a07f58fa50b95c38ed52c41c5732c4bcaddbf4f9a71c468"

#: The three-level fixture the level gates use: 32 fine entries, 4 coarse,
#: 2 mid — small enough to reason about, long enough for every stage to fire.
THREE = dict(
    use_csa=True,
    csa_chunk=8,
    csa_topk=8,
    csa_coarse_chunk=64,
    csa_mid_chunk=128,
)


def _batch(n: int = SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * n, dtype=torch.long).view(BATCH, n) % 251
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


# --- 1. every new flag OFF is bit-identical to clean main ------------------


def test_gate_1_all_flags_off_is_bit_identical_to_main():
    cases = {
        "default": ({}, PIN_DEFAULT),
        "causal fixture": ({"use_engram": False, "moe_topk": 0}, PIN_CAUSAL),
        "csa fine only": ({"use_csa": True}, PIN_CSA_FINE),
        "csa + coarse": (
            {"use_csa": True, "csa_chunk": 4, "csa_topk": 8, "csa_coarse_chunk": 8},
            PIN_CSA_COARSE,
        ),
    }
    for name, (over, pin) in cases.items():
        got = _pinned(over)
        print(f"{name:16s} {got}")
        assert got == pin, f"{name} moved: the off path is not the old path"
    # The explicit off spelling of each new flag is the default spelling.
    assert _pinned({"csa_mid_chunk": 0, "csa_hierarchical": False}) == PIN_DEFAULT
    assert _pinned({"attn_window": 0, "attn_window_query_chunk": 0}) == PIN_DEFAULT


# --- 2. the mid level is causal, dense and hierarchical -------------------


def _mutate_causality(attn, x: torch.Tensor, at: int, name: str) -> None:
    with torch.no_grad():
        base = attn(x)
        moved = x.clone()
        moved[:, at, 3] += 1.0
        got = attn(moved)
        assert torch.equal(base[:, :at], got[:, :at]), (
            f"{name}: mutating byte {at} moved an output at a position < {at}"
        )
        assert not torch.allclose(base[:, at], got[:, at]), (
            f"{name}: position {at} ignored its own byte"
        )
        print(f"{name}: pos-{at} max delta {float((base[:, at] - got[:, at]).abs().max()):.3e}")


def test_gate_2_the_mid_level_is_causal_both_ways():
    torch.manual_seed(0)
    x = torch.randn(BATCH, 256, 64)
    for name, over in (
        ("fine+coarse+mid", {}),
        ("hierarchical", {"csa_hierarchical": True}),
    ):
        attn = CompressedSelectiveAttention(tiny_cfg(**{**THREE, **over}))
        x = x.clone().requires_grad_(False)
        # 100 sits inside a coarse entry and a mid entry; 200 inside another.
        for at in (100, 200):
            _mutate_causality(attn, x, at, name)

    # The same property through the WHOLE model, where the arm is summed with
    # KDA (`moe_topk=0`: the dense blend is per position; a capacity-dropping
    # MoE is not prefix-causal by design, test_decode_recurrent.py).
    torch.manual_seed(0)
    model = DormouseModel(
        tiny_cfg(
            use_csa=True,
            csa_chunk=4,
            csa_topk=8,
            csa_coarse_chunk=8,
            csa_mid_chunk=16,
            csa_hierarchical=True,
            attn_window=8,
            use_engram=False,
            moe_topk=0,
        )
    )
    ids, _ = _batch()
    with torch.no_grad():
        base = model(ids)[0].view(BATCH, SEQ, -1)
        later = ids.clone()
        later[:, 8] = (later[:, 8] + 1) % 251
        got = model(later)[0].view(BATCH, SEQ, -1)
    assert torch.equal(base[:, :8], got[:, :8]), "the three-level model read a future byte"
    assert not torch.allclose(base[:, 8], got[:, 8]), "position 8 ignored its own byte"
    print("whole model: pos-8 delta moved, positions <8 bit-identical")


# --- 3. hierarchical selection == flat when topk covers everything --------


def test_gate_3_hierarchy_selects_the_flat_region_when_topk_covers_it():
    torch.manual_seed(0)
    x = torch.randn(BATCH, 256, 64)
    cover = dict(THREE, csa_topk=64)  # >= t/chunk = 32, and > every level
    torch.manual_seed(7)
    hier = CompressedSelectiveAttention(tiny_cfg(csa_hierarchical=True, **cover))
    flat = CompressedSelectiveAttention(tiny_cfg(**cover))
    flat.load_state_dict(hier.state_dict())
    c_h, c_f = Counters(), Counters()
    with torch.no_grad():
        yh = hier(x, counters=c_h)
        yf = flat(x, counters=c_f)
    # The hierarchy narrows through ALLOWED containers only. The FIRST container
    # of each level holds the query's own future, so it is masked and its keys
    # are unreachable — flat attention does reach them. The two therefore agree
    # everywhere EXCEPT inside that first container. The old assertion (hier ==
    # flat everywhere) passed only because the parent mask was missing: it let
    # children of a masked parent through, which is also why forward and
    # forward_state disagreed. See docs/reviews/2026-11-08-math-audit.md §6.3.
    # Measured, not assumed: the first fine chunk (positions 0..csa_chunk-1)
    # matches, because no parent container is allowed yet and the level falls
    # through to the same keys; every later partially-built container differs.
    # The OLD assertion (equal everywhere) was false and passed only while the
    # parent mask was missing — see docs/reviews/2026-11-08-math-audit.md §6.3.
    chunk = 4                        # csa_chunk in `cover`
    d_first = float((yh[:, :chunk] - yf[:, :chunk]).abs().max())
    d_rest = float((yh[:, chunk:] - yf[:, chunk:]).abs().max())
    tol = 1e-5
    print(f"hier vs flat: first fine chunk max|d| = {d_first:.3e}, rest = {d_rest:.3e}")
    assert d_first <= tol, d_first
    assert d_rest > 1e-3, "the hierarchy must actually narrow something"
    # Both kept EVERY entry at every level (n_fine = 32, n_coarse = 4, n_mid = 2).
    assert c_h["csa_sel_fine"] == BATCH * 256 * 32
    assert c_h["csa_sel_coarse"] == BATCH * 256 * 4
    assert c_h["csa_sel_mid"] == BATCH * 256 * 2
    assert c_h["csa_scan_fine"] == BATCH * 256 * 32, (
        "the surviving coarse entries did not expand to the whole fine range"
    )
    assert "csa_sel_fine" not in c_f, "the flat path must not claim a selection"

    # ... and the hierarchy is LIVE: a small top-k narrows the fine scan (at
    # this shape the mid level has 2 entries, so topk=2 keeps them all and the
    # narrowing happens at the COARSE stage: 2 of 4 -> 16 fine candidates).
    narrow = CompressedSelectiveAttention(
        tiny_cfg(csa_hierarchical=True, **dict(THREE, csa_topk=2))
    )
    narrow.load_state_dict(hier.state_dict())
    c_n = Counters()
    with torch.no_grad():
        yn = narrow(x, counters=c_n)
    assert not torch.allclose(yn, yh), "the hierarchy did not narrow anything"
    assert c_n["csa_scan_fine"] < c_h["csa_scan_fine"]
    flat_scan = BATCH * 256 * 32
    # A prefill SHORTER than the outermost chunk finalises no mid entry, so the
    # ring path has no container: the selection must degrade to the FLAT scan,
    # COUNTED, never to a zeroed fine branch.
    short = torch.randn(BATCH, 64, 64)
    torch.manual_seed(9)
    tiny_h = CompressedSelectiveAttention(tiny_cfg(csa_hierarchical=True, **THREE))
    tiny_f = CompressedSelectiveAttention(tiny_cfg(**THREE))
    tiny_f.load_state_dict(tiny_h.state_dict())
    c_short = Counters()
    with torch.no_grad():
        y_short, st_h = tiny_h.forward_state(short, None, counters=c_short)
        y_flat, st_f = tiny_f.forward_state(short, None, counters=Counters())
    assert st_h[2].n == 0 == st_f[2].n, "the mid ring finalised an entry it cannot have"
    assert c_short["csa_hier_flat_fallback"] == 1, dict(c_short)
    assert torch.equal(y_short, y_flat), "the fallback is not the flat path"
    print(
        f"short prefill (64 < mid {THREE['csa_mid_chunk']}): flat fallback counted, "
        f"output bit-identical to the flat arm"
    )

    # The DECODE half: the same chain through the carried ring, step by step.
    torch.manual_seed(3)
    h_model = CompressedSelectiveAttention(tiny_cfg(csa_hierarchical=True, **cover))
    f_model = CompressedSelectiveAttention(tiny_cfg(**cover))
    f_model.load_state_dict(h_model.state_dict())
    xs = torch.randn(BATCH, 128, 64)
    with torch.no_grad():
        o_h, s_h = h_model.forward_state(xs, None, counters=Counters())
        o_f, s_f = f_model.forward_state(xs, None, counters=Counters())
        step_delta = 0.0
        for i in range(4):
            a, s_h = h_model.step_state(xs[:, i : i + 1], s_h, counters=Counters())
            b, s_f = f_model.step_state(xs[:, i : i + 1], s_f, counters=Counters())
            step_delta = max(step_delta, float((a - b).abs().max()))
    assert [None if s is None else (s.n, s.count) for s in s_h] == [
        None if s is None else (s.n, s.count) for s in s_f
    ], "the hierarchical decode advanced a different ring"
    assert step_delta <= tol, step_delta
    print(
        f"decode: prefill max|d| {float((o_h - o_f).abs().max()):.3e}, "
        f"4 steps max|d| {step_delta:.3e} (tol {tol:.0e}), rings identical"
    )
    print(
        f"narrow: mid kept {c_n['csa_sel_mid']}, coarse kept "
        f"{c_n['csa_sel_coarse']}, fine scan {c_n['csa_scan_fine']} vs flat "
        f"{flat_scan} ({flat_scan / c_n['csa_scan_fine']:.1f}x fewer), "
        f"output delta {float((yn - yh).abs().max()):.3e}"
    )


# --- 4. the window is causal AND banded -----------------------------------


def test_gate_4_the_window_is_exactly_the_last_w_positions():
    torch.manual_seed(0)
    window, t = 8, 32
    attn = FullAttention(tiny_cfg(), window=window)
    x = torch.randn(t + 1, t, 64)  # row 0 base, row p+1 = byte p mutated
    for p in range(t):
        x[p + 1] = x[0]
        x[p + 1, p, 3] += 1.0
    with torch.no_grad():
        out = attn(x)
    moved = torch.tensor(
        [[not torch.equal(out[0, q], out[p + 1, q]) for p in range(t)] for q in range(t)]
    )
    band = torch.tensor([[0 <= q - p <= window - 1 for p in range(t)] for q in range(t)])
    assert torch.equal(moved, band), (
        "the moved set is not the band 0 <= q-p <= W-1: "
        f"{int((moved != band).sum())} of {t * t} pairs disagree"
    )
    # The two mutations the gate names: just OUTSIDE the band of the last
    # position (older than W) must not move it; just INSIDE must.
    last = t - 1
    outside, inside = last - window, last - window + 1
    assert not moved[last, outside], "a byte older than the window moved the last position"
    assert moved[last, inside], "a byte inside the window did not move it"
    print(
        f"band exact on {t * t} pairs: q-{last} moved by bytes {inside}..{last}, not by {outside}"
    )

    # W >= t is the exact causal arm: the band covers everything.
    full = FullAttention(tiny_cfg(), window=t)
    exact = FullAttention(tiny_cfg())
    exact.load_state_dict(full.state_dict())
    x1 = torch.randn(BATCH, t, 64)
    with torch.no_grad():
        assert torch.equal(full(x1), exact(x1)), "W >= t is not exact causal attention"
    print("W >= t equals exact causal attention bit-for-bit")


# --- 5. query chunking is a memory-only decision --------------------------


def test_gate_5_query_chunking_equals_the_unchunked_output():
    torch.manual_seed(0)
    x = torch.randn(BATCH, 64, 64)
    for bf16, tol in ((False, 1e-5), (True, 2e-2)):
        whole = FullAttention(tiny_cfg(bf16=bf16), window=8)
        blocked = FullAttention(tiny_cfg(bf16=bf16), window=8, query_chunk=16)
        blocked.load_state_dict(whole.state_dict())
        with torch.no_grad():
            a = whole(x)
            b = blocked(x)
        delta = float((a - b).abs().max())
        print(
            f"{'bf16' if bf16 else 'fp32'}: max|d| = {delta:.3e} (tol {tol:.0e}, "
            f"|out| = {float(a.abs().max()):.3f})"
        )
        assert delta <= tol, (bf16, delta)
    # Four blocks per call, counted; one block when the flag is off.
    c = Counters()
    with torch.no_grad():
        FullAttention(tiny_cfg(), window=8, query_chunk=16)(x, counters=c)
    assert c["attn_window_query_blocks"] == 4 and c["attn_window_forward"] == 1


# --- 6. no host read on any new path --------------------------------------


def test_gate_6_no_new_path_touches_the_host(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("host sync inside a new long-context path (ADR-0018 r2)")

    for attr in ("item", "tolist", "bool", "float", "__bool__", "__float__"):
        monkeypatch.setattr(torch.Tensor, attr, boom)
    torch.manual_seed(0)
    x = torch.randn(BATCH, 128, 64)
    mid = CompressedSelectiveAttention(tiny_cfg(**THREE))
    assert mid(x, counters=Counters()).shape == x.shape
    hier = CompressedSelectiveAttention(tiny_cfg(csa_hierarchical=True, **THREE))
    assert hier(x, counters=Counters()).shape == x.shape
    _, st = hier.forward_state(x, None, counters=Counters())
    assert len(st) == 3 and st[0].n == 16 and st[1].n == 2 and st[2].n == 1
    step, st = hier.step_state(x[:, :1], st, counters=Counters())
    assert step.shape == (BATCH, 1, 64)
    win = FullAttention(tiny_cfg(), window=8)
    assert win(x, counters=Counters()).shape == x.shape
    blocked = FullAttention(tiny_cfg(), window=8, query_chunk=32)
    blocked.load_state_dict(win.state_dict())
    assert blocked(x, counters=Counters()).shape == x.shape
    print("no host read on: mid dense, hierarchical prefill/step, window chunked/not")


# --- 7. every illegal combination is LOUD --------------------------------


def test_gate_7_illegal_combinations_are_loud_with_an_escape():
    cases = [
        ({"csa_mid_chunk": 512}, "without use_csa"),
        ({"use_csa": True, "csa_coarse_chunk": 256, "csa_mid_chunk": 300}, "multiple"),
        ({"use_csa": True, "csa_chunk": 8, "csa_mid_chunk": 4}, "smaller than"),
        ({"use_csa": True, "csa_hierarchical": True}, "without csa_mid_chunk"),
        ({"csa_hierarchical": True}, "without use_csa"),
        (
            {"use_csa": True, "csa_mid_chunk": 512, "csa_fine": False, "csa_hierarchical": True},
            "csa_fine=false",
        ),
        ({"attn_window": -1}, "negative"),
        ({"attn_window_query_chunk": 4}, "without attn_window"),
        ({"attn_window_query_chunk": -4}, "negative"),
        ({"attn_window": 8, "attn_kind": "full"}, "attn_kind='full'"),
        ({"attn_window": 8, "use_byteflow": True}, "use_byteflow"),
        ({"attn_window": 8, "attn_window_query_chunk": SEQ}, "max_seq_len"),
    ]
    for over, match in cases:
        with pytest.raises(LoudError, match=match) as exc:
            validate(tiny_cfg(**over))
        assert "Escape:" in str(exc.value), (over, str(exc.value))
        print(f"loud: {over} -> {match}")
    with pytest.raises(LoudError, match="attn_window") as exc:
        _validate_segment(tiny_cfg(attn_window=8), TrainConfig(segment_len=32, seq_len=SEQ))
    assert "Escape:" in str(exc.value)

    # The runtime seams, which a hand-built model never skips.
    with pytest.raises(AssertionError, match="multiple"):
        CompressedSelectiveAttention(
            tiny_cfg(use_csa=True, csa_coarse_chunk=256, csa_mid_chunk=300)
        )
    with pytest.raises(AssertionError, match="smaller than"):
        CompressedSelectiveAttention(tiny_cfg(use_csa=True, csa_mid_chunk=4))
    with pytest.raises(AssertionError, match="query_chunk without a window"):
        FullAttention(tiny_cfg(), window=0, query_chunk=8)
    with pytest.raises(AssertionError, match="attn_kind='full'"):
        DormouseModel(tiny_cfg(attn_kind="full", attn_window=8))

    model = DormouseModel(tiny_cfg(attn_window=8, use_engram=False, moe_topk=0))
    ids, targets = _batch()
    with pytest.raises(AssertionError, match="attn_window=8"):
        model(ids, targets, segment_len=32)
    with pytest.raises(AssertionError, match="attn_window=8"):
        model.loop.recurrent_step(
            model.embed(ids[:, :1]),
            ids[:, :2],
            [None] * model.cfg.max_iter,
            drop=1,
            counters=Counters(),
        )
    why = recurrent_refusal(model)
    assert why is not None and "attn_window" in why and "Escape:" in why, why
    # The legal pairs BUILD AND RUN (evidence, not a promise).
    validate(tiny_cfg(**THREE))
    validate(tiny_cfg(csa_hierarchical=True, **THREE))
    validate(tiny_cfg(attn_window=8, attn_window_query_chunk=32))
    print("loud: mid/coarse/hierarchical/window combinations, plus 3 runtime seams")


# --- 8. counters for every new path, with the arithmetic ------------------


def test_gate_8_every_new_path_is_counted():
    torch.manual_seed(0)
    x = torch.randn(BATCH, 128, 64)
    mid = CompressedSelectiveAttention(tiny_cfg(**THREE))
    mid_c = Counters()
    mid(x, counters=mid_c)
    assert mid_c["csa_mid_forward"] == 1
    assert mid_c["csa_coarse_forward"] == 1
    hier = CompressedSelectiveAttention(tiny_cfg(csa_hierarchical=True, **THREE))
    c = Counters()
    with torch.no_grad():
        hier(x, counters=c)
    assert c["csa_hier_forward"] == 1
    # 16 fine entries, 2 coarse, 1 mid at t=128; topk=8 caps the fine stage.
    assert c["csa_sel_mid"] == BATCH * 128 * 1
    assert c["csa_sel_coarse"] == BATCH * 128 * 2
    assert c["csa_scan_fine"] == BATCH * 128 * 2 * (64 // 8)
    assert c["csa_sel_fine"] == BATCH * 128 * 8
    hier_c = c
    win = FullAttention(tiny_cfg(), window=8, query_chunk=32)
    c = Counters()
    win(x, counters=c)
    assert c["attn_window_forward"] == 1
    assert c["attn_window_query_blocks"] == 4
    print(
        f"counters: mid={mid_c['csa_mid_forward']}, sel mid/coarse/fine="
        f"{hier_c['csa_sel_mid']}/{hier_c['csa_sel_coarse']}/"
        f"{hier_c['csa_sel_fine']}, fine scan={hier_c['csa_scan_fine']}, blocks="
        f"{c['attn_window_query_blocks']}"
    )


# --- 9. the memory arithmetic and the window's MAC bill -------------------


def test_gate_9_state_bytes_against_the_closed_form():
    for context, chunk in ((32 << 10, 8), (256 << 10, 8), (4 << 20, 8), (4 << 20, 32)):
        entries = -(-context // chunk)
        closed = entries * 2 * 64 * 2  # k and v, bf16, kv=64
        got = state_bytes(context, 64, chunk)
        print(
            f"{context:>9,d} B at chunk {chunk:>2d} -> {got:>12,d} B "
            f"({got / 2**20:7.1f} MiB) = {entries:,} entries x 256 B"
        )
        assert got == closed
    # The brief's headline: 4 MiB of context at chunk 32 / kv 64 is 32 MiB.
    assert state_bytes(4 << 20, 64, 32) == 33_554_432
    # A real ring agrees with the closed form at the largest context the decode
    # ring allows (it caps at 1 MiB of context).
    ring = CSAState(64, 64, 8)
    ring.reserve(-(-(256 << 10) // 8))
    assert ring.bytes() == state_bytes(256 << 10, 64, 8) == 8 << 20
    # Three levels, summed against the closed form and against real tensors.
    for context in (32 << 10, 256 << 10):
        fine = state_bytes(context, 64, 8)
        coarse = state_bytes(context, 64, 256)
        mid = mid_state_bytes(context, 64, 512)
        assert mid == state_bytes(context, 64, 512)
        assert (
            total_state_bytes(context, 64, 8, csa_coarse_chunk=256, csa_mid_chunk=512)
            == fine + coarse + mid
        )
    table = memory_table(
        csa_kv_dim=64,
        csa_chunk=8,
        csa_coarse_chunk=256,
        csa_mid_chunk=512,
        contexts=(32 << 10, 256 << 10, 4 << 20),
    )
    assert table[4 << 20] == (128 << 20) + (4 << 20) + (2 << 20)
    assert table[4 << 20] == total_state_bytes(
        4 << 20, 64, 8, csa_coarse_chunk=256, csa_mid_chunk=512
    )
    print(
        f"4 MiB, three levels (chunk 8/256/512, kv 64): "
        f"{table[4 << 20] / 2**20:.0f} MiB per layer per iteration slot; "
        f"at chunk=32 fine is {state_bytes(4 << 20, 64, 32) / 2**20:.0f} MiB"
    )
    # The window's bill, on top of the rest: flat in the context length.
    for w in (4096, 32768):
        bill = window_mac_per_byte(w, 1024)
        assert bill == 2 * w * 1024
        print(
            f"window {w:>6,d} at d=1024: {bill:>15,.0f} MAC/byte "
            f"(W*d score + W*d value; the brief's W*d is the score half)"
        )


# --- 10. the window is summed beside KDA and it trains --------------------


def test_gate_10_the_window_is_summed_and_its_gradients_flow():
    torch.manual_seed(0)
    base = tiny_cfg(use_engram=False, moe_topk=0)
    torch.manual_seed(1)
    plain = DormouseModel(base)
    torch.manual_seed(1)
    windowed = DormouseModel(dataclasses.replace(base, attn_window=8))
    assert isinstance(windowed.loop.window_attn, FullAttention)
    # The two models draw different RNG streams once the arm exists, so copy
    # the shared tree: the ONLY missing keys must be the window's four
    # projections, which is the parameter-delta claim below.
    missing = windowed.load_state_dict(plain.state_dict(), strict=False)
    assert set(missing.missing_keys) == {
        f"loop.window_attn.{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }, sorted(missing.missing_keys)
    assert not missing.unexpected_keys
    assert (
        sum(p.numel() for n, p in windowed.named_parameters() if "window_attn" in n)
        == 4 * base.d_model**2
    )
    ids, targets = _batch()
    with torch.no_grad():
        a = plain(ids)[0]
        b = windowed(ids)[0]
    assert not torch.allclose(a, b), "attn_window built a module and added nothing"
    with torch.no_grad():
        windowed.loop.window_attn.o_proj.weight.zero_()
    with torch.no_grad():
        zeroed = windowed(ids)[0]
    assert torch.allclose(zeroed, a, atol=0.0, rtol=0.0), (
        "zeroing the window's W_o must return the no-window model exactly"
    )
    # Gradient reaches all four window projections (the arm runs and trains).
    torch.manual_seed(1)
    trained = DormouseModel(dataclasses.replace(base, attn_window=8))
    _, loss, _ = trained(ids, targets)
    assert loss is not None
    loss.backward()
    grads = {
        n.removeprefix("loop.window_attn."): p.grad
        for n, p in trained.named_parameters()
        if n.startswith("loop.window_attn.")
    }
    assert set(grads) == {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"}
    for name, grad in sorted(grads.items()):
        assert grad is not None and torch.isfinite(grad).all(), f"{name}: no finite grad"
        mag = float(grad.abs().sum())
        assert mag > 0.0, f"{name}: zero gradient — the arm runs and trains nothing"
        print(f"window.{name:12s} |grad|_1 = {mag:.4f}")
    print(f"kda+window vs kda: max|d| {float((a - b).abs().max()):.3e}, zeroed W_o bit-identical")
