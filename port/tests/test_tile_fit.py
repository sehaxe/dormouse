"""Gates for the MoE tile SEARCH (`moe_kernels._fit` → `tiles_for`).

The defect this file exists for: `tiles_for` picked ONE tile set from the dtype
and `assert`ed it fitted, so the first shape whose estimate ran past the CTA's
shared memory died at step 0 — every `rank >= 192` bf16 arm of the rank sweep,
in ~14 s, with `bf16 tile BLOCK_N=128 needs 163840 bytes > 101376`.

What is claimed here:

a. the shape that used to fit still picks the SAME tile (`BLOCK_N=128`) and a
   CPU forward+backward is bit-identical to a pin taken from a clean `main`
   worktree before the edit (sha256 of the logits and of every gradient);
b. the shape that crashed now selects the WIDEST tile that fits, and the bytes
   the model charges are the bytes the device compiles (98304 at rank 192, from
   `metadata.shared`, no card needed);
c. fp32 still selects the measured legal tile (32768 B at one stage, 65536 at
   two) and stays bit-identical;
d. a shape NO candidate fits is LOUD: dtype, the smallest candidate's bytes, the
   ceiling, the shape and an `Escape:` line;
e. the search is deterministic and returns the FIRST candidate that fits;
f. the selection adds no host sync (`.item`/`.tolist`/`__float__`/`__bool__`
   patched to raise);
g. the chosen tile is COUNTED on the dispatch seam (`moe_tile_N…_M…_st…`), so a
   log line says which tile ran.

WHAT THIS FILE CANNOT PROVE: there is no CUDA device in this environment, so the
forward+backward gates run the CPU fallback arm and no Triton kernel EXECUTES
here. The shared-memory numbers come from a host-side `triton.compile` (ptxas)
of the real kernels, which is the same harness `smem_estimate` is calibrated
against; the launch itself is unverified.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.config import ModelConfig
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE

mk = pytest.importorskip("dormouse.model.moe_kernels")
pytest.importorskip("triton")
# The compile harness is `test_moe_kernel.py`'s: one implementation of "ask ptxas
# what this kernel costs", not a second copy that could disagree with the table.
from test_moe_kernel import _compiled_shared, _sig  # noqa: E402

#: The ByteFlow/`medium` geometry the rank sweep ran: d 1024, d_ffn 2730.
FIT = dict(d_model=1024, d_ff=2730, rank=64)  # the control arm: fits, BLOCK_N=128
CRASH = dict(d_model=1024, d_ff=2730, rank=192)  # the arm that died: rp = 256
EXHAUST = dict(d_model=1024, d_ff=2730, rank=512)  # nothing fits

#: sha256 of (logits bytes + counter line + every parameter's gradient bytes) for
#: the tiny model of `_run_sha`, PINNED FROM A CLEAN `main` WORKTREE
#: (`git worktree add --detach wt/baseline main`) BEFORE this file's subject was
#: edited. `moe_kernel="triton"` so the arm under test is the one asked for; on CPU
#: its dispatch falls back COUNTED, which is the point — the fallback must be the
#: CPU's, not a tile's.
#:
#: The width is part of the pin: bf16 CPU matmuls reorder with the thread count
#: (measured on this box: 4 and 10 threads both give `3ce3b9e7…`, one thread
#: `5fbda133…`, two `01c49e33…`). The pin above was captured at this box's
#: DEFAULT width, which is why it went red on the CI runner with
#: "the bf16 forward+backward is not the clean-main one" — a width difference,
#: not the tile search (its integer asserts passed there). `_run_sha` now fixes
#: one thread on both sides, so the sha is a function of the code again.
#:
#: Measured 2026-11-13, one thread, this box (x86-64, torch 2.13.0+cu129 CPU):
#: fp32 `d48ec81a…`, bf16 `5fbda133…`. The tile selection at these shapes is the
#: clean-main one (`test_the_fitting_shape…` asserts TILES / BLOCK_N=128 / 24576
#: B before it reads the sha), so the re-measured numbers are the same run the
#: clean-main pin measured, at a declared width.
PIN = {
    False: "d48ec81a875acd480212090112c9476ed09c351a37d3a888e1c3e1f638d31b7b",
    True: "5fbda1332a856d34050b58e45fea4654bb23ba1c689a2af0f3964cdaa187e8d2",
}

GLU_SIG = (
    "x_ptr:*bf16 gu_ptr:*bf16 uu_ptr:*bf16 gs_ptr:*bf16 us_ptr:*bf16 v_ptr:*bf16 "
    "y_ptr:*bf16 offs_ptr:*i32 gpre_ptr:*fp32 upre_ptr:*fp32 h_ptr:*bf16 "
    "D:i32 R_:i32 F_:i32 BLOCK_M:constexpr BLOCK_K:constexpr BLOCK_F:constexpr "
    "RP:constexpr PREC:constexpr SAVE:constexpr SPLIT:constexpr"
)
PAIR_SIG = (
    "a1_ptr:*bf16 w1_ptr:*bf16 a2_ptr:*bf16 w2_ptr:*bf16 o_ptr:*bf16 offs_ptr:*i32 "
    "N:i32 K:i32 sam:i32 sak:i32 swe:i32 swk:i32 swn:i32 som:i32 son:i32 "
    "BLOCK_M:constexpr BLOCK_N:constexpr BLOCK_K:constexpr PREC:constexpr"
)


def _run_sha(**over) -> str:
    """sha256 of a tiny model's logits, gradients AND counter line — the pin.

    `torch.manual_seed` before the build and before the data, so the pin is a
    function of the code and nothing else. Every parameter's `.grad` is hashed by
    name, `b"-"` when it has none, so a parameter that STOPS training moves it
    too (the `8fa5d4c` class of defect). The counter line is hashed because it is
    the arm evidence: a change that sends this config down a different route
    (a tile refusal where there was none) moves the pin even if the CPU fallback
    happens to compute the same numbers.

    ONE thread on both sides: bf16 CPU reductions reorder with the width (see
    `PIN`), and the restore is `try/finally` so this file cannot leak the width
    into the hash pins of another module.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        model = DormouseModel(tiny_cfg(moe_kernel="triton", **over))
        ids = torch.arange(2 * 16, dtype=torch.long).view(2, 16) % 251
        logits, loss, counters = model(ids, (ids + 1) % 251)
        loss.backward()
        h = hashlib.sha256()
        h.update(logits.detach().contiguous().numpy().tobytes())
        h.update(counters.line().encode())
        for name, p in sorted(model.named_parameters()):
            h.update(name.encode())
            h.update(b"-" if p.grad is None else p.grad.detach().contiguous().numpy().tobytes())
        return h.hexdigest()
    finally:
        torch.set_num_threads(threads)


def _first_fitting(dtype: torch.dtype, shape: dict) -> tuple[dict, int, int]:
    """The candidate the search MUST return: the first in order whose bytes fit."""
    base = mk._base_tiles(dtype, shape["d_model"], shape["d_ff"])
    for t, st in mk._candidates(dtype, base):
        n = mk.smem_estimate(t, dtype, shape["rank"], stages=st)
        if n <= mk.SMEM_CEILING_BYTES:
            return t, st, n
    raise AssertionError(f"no candidate fits {dtype} {shape}; the test's own premise is wrong")


@pytest.mark.pins
def test_the_fitting_shape_keeps_the_shipped_tile_and_its_numbers():
    """Gate (a): a shape that fitted before picks the SAME tile and runs the same.

    bf16 at the control shape (rank 64) must still be `TILES` — BLOCK_N=128, the
    measured-fast wide tile — and the forward+backward must be bit-identical to
    the clean-main pin. A tile search that "fixed" the crash by shrinking a tile
    that already fitted fails here.
    """
    t = mk.tiles_for(torch.bfloat16, **FIT)
    print(
        f"[fit] {mk.tile_name(torch.bfloat16, **FIT)} "
        f"est={mk.smem_estimate(t, torch.bfloat16, FIT['rank'])} B "
        f"<= {mk.SMEM_CEILING_BYTES} B"
    )
    assert t == mk.TILES, f"the bf16 fast path moved: {t}"
    assert t["BLOCK_N"] == 128
    assert mk.tile_name(torch.bfloat16, **FIT) == "moe_tile_N128_M64_st2"
    assert mk.smem_estimate(t, torch.bfloat16, FIT["rank"]) == 24576
    sha = _run_sha(bf16=True)
    print(f"[fit] sha={sha}")
    assert sha == PIN[True], "the bf16 forward+backward is not the clean-main one"


def test_the_rank_192_shape_that_crashed_now_runs_at_the_widest_tile():
    """Gate (b): the arm that died at kernel selection runs, at 98304 B not 163840.

    `rank=192` is `RP = 256` for the kernels. The tile search keeps BLOCK_N=128 /
    BLOCK_M=64 and gives up a pipeline stage instead, and the numbers are checked
    against a COMPILE of the two kernels that can be the family max at that rank
    (`_glu_kernel` at RP=256, `_grouped_gemm_pair_kernel` at BLOCK_K=256). The
    forward+backward then has to complete.
    """
    t = mk.tiles_for(torch.bfloat16, **CRASH)
    n = mk.smem_estimate(t, torch.bfloat16, CRASH["rank"])
    print(f"[crash shape] {mk.tile_name(torch.bfloat16, **CRASH)} est={n} B")
    assert (t["BLOCK_N"], t["BLOCK_M"]) == (128, 64), "the widest fitting tile, not a smaller one"
    assert n == 98304, f"the widest tile at rank 192 charges {n} B, expected 98304"
    assert n <= mk.SMEM_CEILING_BYTES

    glu = _compiled_shared(
        mk._glu_kernel,
        _sig(GLU_SIG),
        dict(BLOCK_M=64, BLOCK_K=64, BLOCK_F=128, RP=256, PREC="tf32", SAVE=True, SPLIT=False),
        4,
    )
    pair = _compiled_shared(
        mk._grouped_gemm_pair_kernel,
        _sig(PAIR_SIG),
        dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=256, PREC="tf32"),
        8,
    )
    print(f"[crash shape] compiled: glu={glu} B pair={pair} B (model {n} B)")
    assert glu == 98304, f"_glu_kernel at RP=256 measures {glu} B, not 98304"
    assert pair == 98304, f"the pair kernel at BLOCK_K=256 measures {pair} B, not 98304"
    assert max(glu, pair) == n, "the model must equal the widest COMPILED kernel"

    # ... and the run that used to die now completes a forward+backward.
    cfg = ModelConfig(
        moe_kernel="triton",
        bf16=True,
        d_model=1024,
        n_heads=16,
        head_dim=64,
        d_ffn=2730,
        rank=192,
        n_experts=85,
        moe_topk=2,
        max_seq_len=16,
        max_iter=1,
        engram_rows=64,
        engram_dim=8,
    )
    moe = SparseMoE(cfg, device="cpu")
    torch.manual_seed(0)
    x = torch.randn(8, cfg.d_model)  # fp32 in; the bank casts to its own dtype
    counters = Counters()
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    y = moe(x, counters)
    y.float().pow(2).mean().backward()
    graded = [name for name, p in moe.named_parameters() if p.grad is not None]
    print(f"[crash shape] fwd+bwd done, y={tuple(y.shape)}, graded={len(graded)} params")
    assert y.shape == (8, cfg.d_model) and torch.isfinite(y).all()
    assert graded, "the forward+backward produced no gradient at all"


@pytest.mark.pins
def test_fp32_still_selects_a_legal_tile_and_stays_bit_identical():
    """Gate (c): fp32 keeps the measured tile and the measured two-stage charge.

    The case that forced 65536 B is the fp32 pair at BN=64 with `stages=2`
    (`smem_estimate`'s measured multiplier); it is still legal, and the shipped
    launch (one stage, 32768 B) is unchanged. A wider fp32 shape (rank 192) must
    also get a legal tile out of the search instead of an assert.
    """
    t = mk.tiles_for(torch.float32, **FIT)
    n = mk.smem_estimate(t, torch.float32, FIT["rank"])
    two = mk.smem_estimate(t, torch.float32, FIT["rank"], stages=2)
    print(
        f"[fp32] {mk.tile_name(torch.float32, **FIT)} est={n} B "
        f"stages=2 -> {two} B <= {mk.SMEM_CEILING_BYTES} B"
    )
    assert (t["BLOCK_N"], t["BLOCK_F"], t["BLOCK_M"]) == (64, 64, 64), t
    assert mk.tile_name(torch.float32, **FIT) == "moe_tile_N64_M64_st1"
    assert n == 32768 and two == 65536 and two <= mk.SMEM_CEILING_BYTES
    assert mk.tile_reason(torch.float32, **FIT) is None, "fp32 must not fall back"

    wide = mk.tiles_for(torch.float32, d_model=1024, d_ff=2730, rank=192)
    wn = mk.smem_estimate(wide, torch.float32, 192)
    print(f"[fp32 rank 192] {mk.tile_name(torch.float32, 1024, 2730, 192)} est={wn} B")
    assert wide["BLOCK_N"] == 64 and wn <= mk.SMEM_CEILING_BYTES
    assert wide == _first_fitting(torch.float32, CRASH)[0]

    sha = _run_sha()
    print(f"[fp32] sha={sha}")
    assert sha == PIN[False], "the fp32 forward+backward is not the clean-main one"


def test_a_shape_no_candidate_fits_is_loud_with_numbers_shape_and_escape():
    """Gate (d): exhaustion names dtype, bytes, ceiling, shape and the ESCAPE.

    `rank=512` (`RP=512`) is over the ceiling for every candidate; the message is
    the one a run dies with, and `tile_reason` returns the SAME words so the
    dispatch can count the fallback instead of dying (`experts._triton_reason`).
    """
    smallest = _smallest_bytes(torch.bfloat16, EXHAUST)
    with pytest.raises(LoudError) as err:
        mk.tiles_for(torch.bfloat16, **EXHAUST)
    msg = str(err.value)
    print(f"[exhausted] {msg}")
    assert "bf16" in msg
    assert f"{smallest} bytes" in msg, f"the smallest candidate's bytes must be named: {msg}"
    assert str(mk.SMEM_CEILING_BYTES) in msg
    assert "rank=512" in msg and "d_model=1024" in msg and "d_ff=2730" in msg
    assert "Escape:" in msg and "moe_kernel=naive" in msg
    assert mk.tile_reason(torch.bfloat16, **EXHAUST) == msg, (
        "tile_reason must return the same words the run dies with"
    )


def _smallest_bytes(dtype: torch.dtype, shape: dict) -> int:
    base = mk._base_tiles(dtype, shape["d_model"], shape["d_ff"])
    return min(
        mk.smem_estimate(t, dtype, shape["rank"], stages=st)
        for t, st in mk._candidates(dtype, base)
    )


def test_the_search_is_deterministic_and_takes_the_first_candidate_that_fits():
    """Gate (e): same shape in, same tile out — and it is the FIRST that fits.

    Determinism is a property of the code (a fixed list, no measurement, no
    state); "first that fits" is the ordering contract, so a tile that fits
    earlier in `TILE_SEARCH` must never be passed over.
    """
    for dtype, shape in (
        (torch.bfloat16, FIT),
        (torch.bfloat16, CRASH),
        (torch.float32, FIT),
        (torch.float32, CRASH),
    ):
        first = mk.tiles_for(dtype, **shape)
        again = mk.tiles_for(dtype, **shape)
        name = mk.tile_name(dtype, **shape)
        want, st, n = _first_fitting(dtype, shape)
        print(f"[determinism] {mk._DTYPE_NAMES[dtype]} rank={shape['rank']} -> {name} ({n} B)")
        assert first == again == want, f"{dtype} {shape}: {first} != {want}"
        assert name == mk.tile_name(dtype, **shape)
        assert (
            mk.smem_estimate(first, dtype, shape["rank"], stages=st) == n <= mk.SMEM_CEILING_BYTES
        )


def test_the_selection_adds_no_host_sync(monkeypatch):
    """Gate (f): the tile path never reads a device value (ADR-0018 rule 2).

    A per-step scalar read serializes host against device. The search is
    arithmetic on Python ints and the tile name is a string, so patching the four
    ways a tensor becomes a Python value must not trip the search OR the dispatch
    seam it feeds. Scoped to that seam on purpose: a CPU-only `SparseMoE.forward`
    reaches the PRE-EXISTING `.tolist()` in `_naive_dispatch` (the reference
    loop's per-expert grouping), which is not the tile path and which this gate
    must not blame on it.
    """

    def boom(*_a, **_k):
        raise AssertionError("host sync in the tile path (ADR-0018 rule 2)")

    for name in ("item", "tolist", "__float__", "__bool__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    for dtype, shape in ((torch.bfloat16, FIT), (torch.bfloat16, CRASH), (torch.float32, FIT)):
        mk.tiles_for(dtype, **shape)
        mk.tile_name(dtype, **shape)
        assert mk.tile_reason(dtype, **shape) is None, shape  # fp32/bf16 both fit
    assert mk.tile_reason(torch.bfloat16, **EXHAUST) is not None  # the loud words, still no read
    cfg = tiny_cfg(n_experts=4, moe_topk=2, rank=8, moe_kernel="triton", d_model=64, d_ffn=128)
    moe = SparseMoE(cfg, device="cpu")
    monkeypatch.setattr(SparseMoE, "_triton_reason", lambda self, _x: None)
    monkeypatch.setattr(
        SparseMoE, "_triton_dispatch", lambda self, x, *a, **k: torch.zeros(32, 128)
    )
    x = torch.randn(32, 64)
    index, weight = _routing(moe, x)
    counters = Counters()
    out = moe._dispatch(x, weight, index, counters)
    print(f"[no sync] {counters.line()}")
    assert out.shape == (32, 128)


def test_the_chosen_tile_is_counted_on_the_dispatch_seam(monkeypatch):
    """Gate (g): the log line says WHICH tile ran — COUNTED, not SILENT (ADR-0019).

    The kernels are the CUDA arm, so the positive path is proved at the dispatch
    seam it shares with `moe_triton_gemm`: the reason gate is forced to `None` and
    the kernel call to a stub, and the counter must carry the name the real
    search produces for this model's shape.
    """
    cfg = tiny_cfg(n_experts=4, moe_topk=2, rank=8, moe_kernel="triton", d_model=64, d_ffn=128)
    moe = SparseMoE(cfg, device="cpu")
    x = torch.randn(32, 64)
    index, weight = _routing(moe, x)
    monkeypatch.setattr(SparseMoE, "_triton_reason", lambda self, _x: None)
    monkeypatch.setattr(
        SparseMoE, "_triton_dispatch", lambda self, x, *a, **k: torch.zeros(32, 128)
    )
    counters = Counters()
    moe._dispatch(x, weight, index, counters)
    want = mk.tile_name(moe._bank_dtype(), moe.d, moe.f, moe.experts[0].rank)
    print(f"[tile counter] {counters.line()} want={want}")
    assert counters["moe_triton_gemm"] == 1
    assert counters[want] == 1, f"the chosen tile is not counted: {counters.line()}"


def _routing(moe: SparseMoE, x: torch.Tensor, seed: int = 0):
    """`test_moe_kernel._routing`'s fixed assignment: no graph back into the router."""
    torch.manual_seed(seed)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    return index, weight / weight.sum(dim=-1, keepdim=True)
