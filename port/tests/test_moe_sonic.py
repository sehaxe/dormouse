"""Gates for the SonicMoE analysis, the prototype, and the arm it is wired as.

**The module is a shipped, config-gated arm since 2026-10-21**
(`moe_gather = "kernel"`, `experts.SparseMoE._gather_dispatch`). The split here
is now by WHAT each half can prove, stated per gate.

**Proven on CPU** (every one of these runs in the plain suite):

1. **the shape table is the configs' own** — `SHAPES`' fields are read back out
   of `configs/medium.toml` / `configs/b1.toml` with `tomllib`, so the byte table
   cannot drift from the geometry it prices;
2. **the traffic table is hand-computable** — every literal in the module
   docstring's table is asserted, plus the two closed forms the fusion turns on
   (`forward_gather_saving == 4·R·d` at the bf16 seam, `12·R·d` for the fp32
   control, `backward_gather_saving == −2·R·f`);
3. **the premise of the byte table is measured** — the MoE input arrives in
   the bank dtype at the loop's seam (bf16 when `bf16=true`, fp32 otherwise),
   through a real `DormouseModel` forward; the fp32 seam remains the priced
   control (`pass_bytes(x_bytes=4)`, 12·R·d), and the bf16 seam is 4·R·d. The
   seam's own gates: the bf16 seam is a parity move (measured `max|Δlogit|`
   printed against one bf16 `eps` × logit scale, with a dtype spy that fails
   if the cast is removed), `bf16=false` is `torch.equal` to the old branch,
   the cast happens ONCE (`Tensor.to` meter), the gather casts ZERO times on
   a bf16 `x` (and asserts a third dtype), and the branch adds no host sync;
4. **the ABI is the permute the pass already has** — `gather_abi`'s `toks` is
   bit-identical to the `owner` the shipped path gathers `x` with, and its
   `offs` is the same ragged partition the kernels take;
5. **the fused math is the module's own reference** — `fused_reference` +
   the weight scatter equals `SparseMoE._naive_dispatch` at fp32 and bf16;
6. **the fused load cannot read outside `x`** — a CPU emulation of the kernel's
   grid and masks: every address is in bounds (masked lanes point at row 0),
   exactly `R` token rows are read, and they are exactly the gathered rows;
7. **the kernel builds for sm_120** — `triton.compile` with an explicit
   `GPUTarget` lowers to cubin on the HOST (ptxas, no device; the precedent is
   `tests/test_moe_kernel.py::test_the_fused_kernels_compile_for_this_arch`);
8. **the WIRING is the module's own reference loop** — `_gather_dispatch` against
   `_naive_dispatch` on the same routing, forward AND gradients, at fp32 with
   both Triton calls swapped for `fused_reference` and `combine` for its own
   torch forward (`_wired_arm`). The **CPU half can only test the index
   arithmetic** and the graph: the kernel, the shipped adjoint and `combine`'s
   fused backward are all Triton launches, and a CPU gate that swapped them for
   torch is measuring the proxies it swapped in;
9. **the branch, its counters and its refusals** — `moe_gather_forward` /
   `moe_gather_rows` are counted when the arm runs, a call it cannot serve is
   COUNTED with the reason (`moe_gather_fallback`) and falls through to
   `moe_kernel`, and `config.validate` refuses the four combinations the arm has
   no evidence for (`bf16=false`, `use_byteflow=true`, `moe_topk_mode != "off"`,
   `factor_qat != "none"`) while `off` stays the default and draws nothing.

**Needs the card** (written, `skipif`'d on the repo's preflight with its reason
printed, so a reader of a skip line knows what was not measured): the kernel's
NUMBERS — the fused kernel against `moe_kernels.expert_gemm_glu` on the same
rows and the same banks, at fp32 and bf16, on both the tight capacity bound and
the dropless static one — and the WIRED arm against the shipped triton arm
(logits and gradients, 1e-5, fp32 and bf16). On 2026-10-18 the preflight found
the card IDLE (849 MiB, no `dormouse train` process) and the prototype's gates
RAN and PASSED; the wired arm's gate has been written and NOT run (the card was
held on 2026-10-21), so no number from it exists yet. No timing gate: a one-shot
parity call says nothing about speed, and the perf question is
`scripts/bench_gather.py`'s (written, NOT run).
"""

from __future__ import annotations

import dataclasses
import subprocess
import tomllib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg
from torch.overrides import TorchFunctionMode

from dormouse.config import ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import moe_sonic as ms
from dormouse.model.experts import SparseMoE

try:  # the shipped kernels: the GPU parity gate's other side
    from dormouse.model import moe_kernels as mk
except ImportError:  # pragma: no cover - the CPU-only wheel
    mk = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """`max|a-b| / max|b|` — the metric `tests/test_moe_kernel.py` reports."""
    a, b = a.detach().float(), b.detach().float()
    scale = float(b.abs().max())
    d = float((a - b).abs().max())
    return d / scale if scale > 0 else d


# --------------------------------------------------------------------------
# 1-2. the table: the configs' geometry, and the bytes it implies
# --------------------------------------------------------------------------


def test_the_shape_table_is_the_configs_own():
    """`SHAPES` is not a second copy of the geometry — it is read back.

    A config edit that moves `d_ffn`, `rank` or `n_experts` changes every byte
    count in the module docstring, so it must fail HERE rather than leave a
    stale table standing as evidence.
    """
    fields = {"d_model": "d", "d_ffn": "f", "rank": "rank", "n_experts": "experts"}
    for shape in ms.SHAPES:
        cfg = tomllib.loads((ROOT / "configs" / f"{shape.name.split('-')[0]}.toml").read_text())
        for key, attr in fields.items():
            assert getattr(shape, attr) == cfg[key], (
                f"{shape.name}: {attr}={getattr(shape, attr)} but configs/... {key}={cfg[key]}"
            )
        assert shape.topk == cfg["moe_topk"]
        print(f"[shapes] {shape.name}: the config's own fields; R={shape.rows}")


def test_the_traffic_table_is_hand_computable():
    """Every byte the module docstring quotes, asserted as a literal.

    The two closed forms are the point of the table: at the bf16 seam the
    forward saving is `4·R·d` — the `index_select` copy (2·R·d) plus the
    kernel's read of it (2·R·d) — and the backward saving is exactly
    `−2·R·f`, NEGATIVE, because a gathered `grad_out` is fp32 and would be
    read twice where the shipped pass writes a bf16 `dy` once and reads it
    twice. The fp32 control (`x_bytes=4`) is asserted in the same test: it
    still prices `12·R·d`, so a re-introduced seam cast moves the bf16 number
    and fails the `4 * r * d` line instead of drifting the whole table.
    """
    medium, b1 = ms.SHAPES
    m = ms.pass_bytes(medium)
    assert medium.rows == 49_152
    assert m["x_read"] == 100_663_296  # one [R,d] bf16 read (the source rows)
    assert m["xg_write"] == 100_663_296  # the bf16 `index_select` result
    assert m["xg_cast_read"] == 0, "no cast at the bf16 seam"
    assert m["xg_dt_write"] == 0, "no bf16 copy at the bf16 seam"
    assert m["xg_read"] == 100_663_296  # `_glu_kernel`'s A load
    assert m["forward_gather_saving"] == 201_326_592
    assert m["forward_total"] == 1_295_122_432
    assert m["y_write"] == m["y_read"] == 268_369_920
    assert m["out_write"] == 268_369_920  # [T,f] fp32
    assert m["banks_read"] == 156_565_504
    assert m["xg_copy"] == 100_663_296  # the gathered bf16 copy the pass caches
    assert m["cached_activation"] == 400_490_496
    assert m["backward_adjoint_current"] == 1_073_479_680
    assert m["backward_adjoint_fused"] == 1_341_849_600
    assert m["backward_gather_saving"] == -268_369_920

    b = ms.pass_bytes(b1)
    assert b1.rows == 16_384
    assert b["x_read"] == 67_108_864
    assert b["xg_cast_read"] == b["xg_dt_write"] == 0
    assert b["forward_gather_saving"] == 134_217_728
    assert b["forward_total"] == 2_001_305_600
    assert b["banks_read"] == 1_252_655_104  # 63% of the forward: the fine-grained bill
    assert b["cached_activation"] == 256_540_672
    assert b["backward_gather_saving"] == -178_946_048

    for shape, table in ((medium, m), (b1, b)):
        r, d, f = shape.rows, shape.d, shape.f
        assert table["forward_gather_saving"] == 4 * r * d, (
            "bf16 seam: the copy + the kernel's read"
        )
        assert table["backward_gather_saving"] == -2 * r * f, "8Rf gathered vs 6Rf copied+read"
        assert table["forward_total"] == sum(
            table[k]
            for k in (
                "x_read",
                "xg_write",
                "xg_cast_read",
                "xg_dt_write",
                "xg_read",
                "y_write",
                "y_read",
                "out_write",
                "banks_read",
                "gates_write",
            )
        )
        print(
            f"[traffic] {shape.name}: R={r} forward={table['forward_total'] / 2**20:.0f} MiB "
            f"saving={table['forward_gather_saving'] / 2**20:.0f} MiB "
            f"({100 * table['forward_gather_saving'] / table['forward_total']:.0f}%) "
            f"backward saving={table['backward_gather_saving'] / 2**20:+.0f} MiB "
            f"cache={table['cached_activation'] / 2**20:.0f} MiB"
        )

    # The CONTROL: the fp32 seam still prices the old four-transfer chain.
    # Reverting the seam cast (`X_BYTES = 4`, or a `.to(dt)` back in the
    # gather) moves the bf16 assertion above to 12·R·d and turns it red.
    fp32 = ms.pass_bytes(medium, x_bytes=4)
    assert fp32["xg_cast_read"] == 201_326_592
    assert fp32["xg_dt_write"] == 100_663_296
    assert fp32["forward_gather_saving"] == 12 * medium.rows * medium.d == 603_979_776
    assert fp32["cached_activation"] == m["cached_activation"], (
        "the gathered bf16 copy is cached under either seam"
    )
    print(
        f"[seam control] fp32 saving={fp32['forward_gather_saving'] / 2**20:.0f} MiB "
        f"-> bf16 {m['forward_gather_saving'] / 2**20:.0f} MiB "
        f"(-{8 * medium.rows * medium.d / 2**20:.0f} MiB = 8·R·d)"
    )


def test_the_wired_arms_traffic_is_its_own():
    """What `moe_gather="kernel"` really moves — asserted, not inferred.

    The forward keeps the kernel's saving (`4·R·d` at the bf16 seam + the
    gates), the BACKWARD pays the recompute back, and the CACHE is `x`, `toks`,
    `y`: at k=2 the cached bf16 `x` (`T·d·2 = R·d`) is half the shipped arm's
    gathered `xg` (`R·d·2`), so the cache saving is the gates plus `R·d` — the
    paper's 4.9x belongs to the FUSED design, which needs an adjoint that reads
    `dout` through the map and does not exist here.

    `gather_pass > shipped_pass` is asserted ON PURPOSE: as wired, the arm is a
    forward byte win and a PASS byte loss, and a future gather-fused adjoint
    must change this test when it changes that fact. `scripts/bench_gather.py`
    times both; this is only arithmetic.
    """
    for shape in ms.SHAPES:
        w = ms.wired_pass_bytes(shape)
        b = ms.pass_bytes(shape)
        rows, d = shape.rows, shape.d
        assert w["gather_forward"] == (
            w["shipped_forward"] - 4 * rows * d - b["gates_write"] + 4 * rows
        )
        assert w["recompute"] == w["shipped_forward"] - b["y_read"] - b["out_write"]
        assert w["gather_backward"] == w["shipped_backward"] + w["recompute"]
        assert w["gather_cache"] == b["x_read"] // shape.topk + 4 * rows + b["y_write"]
        assert w["gather_cache"] == (
            w["shipped_cache"]
            - b["xg_copy"]
            - b["gates_write"]
            + b["x_read"] // shape.topk
            + 4 * rows
        )
        assert w["gather_pass"] == w["gather_forward"] + w["gather_backward"]
        assert w["gather_forward"] < w["shipped_forward"], "the forward IS the win"
        assert w["gather_cache"] < w["shipped_cache"], "the cache falls by the gates"
        assert w["gather_pass"] > w["shipped_pass"], "the recompute is a whole forward"
        print(
            f"[wired traffic] {shape.name}: fwd {w['shipped_forward'] / 2**20:.0f} -> "
            f"{w['gather_forward'] / 2**20:.0f} MiB, cache "
            f"{w['shipped_cache'] / 2**20:.0f} -> {w['gather_cache'] / 2**20:.0f} MiB, "
            f"pass {w['shipped_pass'] / 2**20:.0f} -> {w['gather_pass'] / 2**20:.0f} MiB"
        )


def test_tile_work_is_the_tile_quantization_b1_pays():
    """`Σ_e cdiv(n_e, BLOCK_M)·BLOCK_M` — the 25% at b1 is arithmetic, not a guess.

    b1's dropless routing gives `R/E = 16 384/1 024 = 16` rows per expert against
    a 64-row tile, so the kernel runs 65 536 row-slots for 16 384 useful ones.
    `medium` divides evenly (192 = 3·64) and shows what a routing that fits the
    tile would look like; the real routing sits between the two, which is why
    the docstring quotes the uniform case as the boundary and not a measurement.
    """
    medium, b1 = ms.SHAPES
    assert ms.tile_work(b1, [16] * 1024) == 65_536
    assert ms.tile_work(medium, [192] * 256) == 49_152
    assert ms.tile_work(b1, [16] * 1024) == 4 * b1.rows, "16 rows in a 64-row tile"
    assert ms.tile_work(medium, [192] * 256) == medium.rows, "192 = 3 x 64: no waste"
    # A group that straddles the tile boundary pays the same as one that fills it.
    assert ms.tile_work(b1, [65]) == 128 and ms.tile_work(b1, [64]) == 64
    with pytest.raises(AssertionError):
        ms.tile_work(b1, [])
    print(f"[tile work] b1 uniform 4x; medium uniform 1x (BLOCK_M={ms.TILES['BLOCK_M']})")


def test_the_moe_input_arrives_in_the_bank_dtype_at_the_seam():
    """The premise the whole byte table rests on, MEASURED through a real forward.

    `loopblock.LoopBlock._moe_branch` casts `normed` to `SparseMoE._bank_dtype()`
    ONCE, so `x[owner]` is the gather alone: bf16 at `bf16=true` (one copy +
    the kernel read = 4·R·d), fp32 at `bf16=false` (the no-op seam, unchanged
    `torch.equal` forward). If the seam cast is removed, the bf16 branch sees
    fp32 here and both this test and the `4·R·d` traffic assertion go red —
    which is why the dtype is measured at the module that receives it, not
    inferred from the config.
    """
    from dormouse.model import DormouseModel

    for bf16 in (False, True):
        seen: dict = {}
        real = SparseMoE.forward

        def spy(self, x, counters=None, _real=real, _seen=seen):
            _seen["dtype"] = x.dtype
            return _real(self, x, counters)

        SparseMoE.forward = spy
        try:
            model = DormouseModel(tiny_cfg(bf16=bf16, n_experts=4, moe_topk=2, moe_kernel="naive"))
            ids = torch.arange(2 * 64, dtype=torch.long).view(2, 64) % 251
            with torch.no_grad():
                model(ids, None)
        finally:
            SparseMoE.forward = real
        dt = model.loop.moe._bank_dtype()
        want = torch.bfloat16 if bf16 else torch.float32
        assert seen["dtype"] == want, (
            f"bf16={bf16}: the MoE input is {seen['dtype']}, not the bank dtype {want}"
        )
        print(
            f"[input dtype] bf16={bf16}: compute {dt}, input {seen['dtype']} "
            f"-> X_BYTES={2 if bf16 else 4}"
        )


# --------------------------------------------------------------------------
# 4-6. the ABI and the reference math, on CPU
# --------------------------------------------------------------------------


def _cpu_setup(*, bf16: bool = False, capacity: float = 0.0, tokens: int = 32):
    """A tiny CPU `SparseMoE` with a FIXED routing, exactly as the gates use one."""
    moe = SparseMoE(
        tiny_cfg(
            n_experts=4, moe_topk=2, rank=8, d_model=64, d_ffn=128, bf16=bf16, moe_capacity=capacity
        ),
        device="cpu",
    )
    torch.manual_seed(0)
    x = torch.randn(tokens, moe.d)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    weight = weight / weight.sum(dim=-1, keepdim=True)
    order, keep, dropped = moe._permute(index)
    return moe, x, weight, index, order, keep, dropped


# --------------------------------------------------------------------------
# 3b. the SEAM: one bf16 cast in, one fp32 cast out, and its byte evidence
# --------------------------------------------------------------------------


class _ToCalls(TorchFunctionMode):
    """Record every `Tensor.to` call as `(source, dtype)` — the seam's cast meter."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.dtype | None]] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.Tensor.to:
            self.calls.append((args[0], args[1] if len(args) > 1 else kwargs.get("dtype")))
        return func(*args, **kwargs)


class _SyncCalls(TorchFunctionMode):
    """Name every host-sync torch function (`item`/`cpu`/`tolist`/`numpy`/`bool`)."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in ("item", "cpu", "tolist", "numpy", "__bool__", "__float__"):
            self.names.append(name)
        return func(*args, **(kwargs or {}))


def _old_seam(self, normed: torch.Tensor, counters: Counters) -> torch.Tensor:
    """The pre-2026-10-21 branch: hand the fp32 `normed` straight to the MoE."""
    return self.moe(normed.reshape(-1, self.d), counters=counters)


def test_the_bf16_seam_is_a_parity_move_not_a_rewrite(monkeypatch):
    """Gate (a): the bf16 seam against the fp32 seam on ONE fixed fixture.

    The measured `max|Δ|` is printed, and the bound is one bf16 `eps` times the
    logit scale — the largest move ONE round-to-nearest can cause, so this is a
    parity gate, not a bit-identity one. The reason the number comes out at
    exactly 0 is worth stating: the fp32 seam's first consumer (`mm_mixed` for
    the router, `DualLatentGLU`/`_gather_rows` for the experts) cast the SAME
    rows to the SAME bf16, and `round(round(x)) == round(x)`; the seam moved
    when the rounding happens, not what it rounds to. A future consumer that
    reads `x` in fp32 (e.g. an activation quantizer before the cast) would
    still be bounded by `eps · scale` here.

    The loss is measured the same way: the delta is printed and asserted under
    a bound first measured at 0.0 and set to 1e-5 (two orders above it). The
    dtype spy is what makes the test red if the seam cast is removed: the
    comparison would then be fp32-seam vs fp32-seam and would trivially pass.
    """
    from dormouse.model import DormouseModel
    from dormouse.model.loopblock import LoopBlock

    torch.manual_seed(11)
    model = DormouseModel(tiny_cfg(bf16=True, n_experts=4, moe_topk=2, moe_kernel="naive"))
    ids = torch.arange(2 * 64, dtype=torch.long).view(2, 64) % 251
    targets = (ids + 1) % 256

    seen: dict = {}
    real = SparseMoE.forward

    def spy(self, x, counters=None, _real=real, _seen=seen):
        _seen["dtype"] = x.dtype
        return _real(self, x, counters)

    monkeypatch.setattr(SparseMoE, "forward", spy)
    with torch.no_grad():
        logits_seam, loss_seam, _ = model(ids, targets)
    assert seen["dtype"] == torch.bfloat16, "the seam ran without its bank-dtype cast"

    monkeypatch.setattr(LoopBlock, "_moe_branch", _old_seam)
    with torch.no_grad():
        logits_fp32, loss_fp32, _ = model(ids, targets)

    delta = float((logits_seam - logits_fp32).abs().max())
    scale = float(logits_fp32.abs().max())
    eps = torch.finfo(torch.bfloat16).eps
    loss_delta = abs(float(loss_seam) - float(loss_fp32))
    print(
        f"[seam parity] max|Δlogit|={delta:.3e} bound={eps * scale:.3e} "
        f"(eps={eps:.3e} x scale={scale:.3f}), |Δloss|={loss_delta:.3e} bound=1e-5"
    )
    assert delta <= eps * scale, f"{delta:.3e} exceeds one bf16 step at this scale"
    assert loss_delta <= 1e-5, f"the loss moved by {loss_delta:.3e}"


def test_the_fp32_seam_is_bit_identical(monkeypatch):
    """Gate (d): `bf16=false` runs the SAME forward the fp32 seam ran.

    Both `_moe_branch` casts are no-ops on existing tensors when `dt` is fp32,
    so the new branch and `_old_seam` must be `torch.equal` on logits AND loss.
    Reverting the change by making the seam cast unconditionally to bf16 turns
    this red (the logits would move and the dtype spy in gate (a) would fail).
    """
    from dormouse.model import DormouseModel
    from dormouse.model.loopblock import LoopBlock

    torch.manual_seed(5)
    model = DormouseModel(tiny_cfg(bf16=False, n_experts=4, moe_topk=2, moe_kernel="naive"))
    ids = torch.arange(2 * 64, dtype=torch.long).view(2, 64) % 251
    targets = (ids + 1) % 256
    with torch.no_grad():
        logits_new, loss_new, _ = model(ids, targets)
    monkeypatch.setattr(LoopBlock, "_moe_branch", _old_seam)
    with torch.no_grad():
        logits_old, loss_old, _ = model(ids, targets)
    assert torch.equal(logits_new, logits_old), "the fp32 seam moved the logits"
    assert torch.equal(loss_new, loss_old), "the fp32 seam moved the loss"
    print("[fp32 seam] torch.equal on logits and loss; bf16=false is unchanged")


def test_the_seam_casts_once_and_the_gather_does_not():
    """Gate (e): ONE cast in at the seam, and the routed gather is cast-free.

    The `Tensor.to` meter sees every call, including a no-op `.to(dt)` — which
    is exactly what a silently re-introduced cast would be on a bf16 `x`. The
    direct-API control proves the meter can see a cast at all (fp32 in -> one
    `.to(dt)`), and a third dtype is an ASSERT, not a coercion.
    """
    from dormouse.model.loopblock import LoopBlock

    moe, x, _w, _i, order, keep, _d = _cpu_setup(bf16=True)
    dt = moe._bank_dtype()
    owner = order[keep] // 2
    xb = x.to(dt)

    with _ToCalls() as spy:
        xg = moe._gather_rows(xb, owner, dt)
    assert xg.dtype == dt
    assert spy.calls == [], f"the gather cast a bf16 x: {spy.calls}"

    with _ToCalls() as spy:
        xg = moe._gather_rows(x, owner, dt)  # direct API caller: fp32 in
    assert len(spy.calls) == 1 and spy.calls[0][1] == dt, spy.calls
    with pytest.raises(AssertionError):
        moe._gather_rows(x.half(), owner, dt)

    block = LoopBlock(tiny_cfg(bf16=True, n_experts=4, moe_topk=2, moe_kernel="naive"))
    normed = torch.randn(2, 64, block.d)
    with _ToCalls() as spy:
        out = block._moe_branch(normed, Counters())
    seam = [c for c in spy.calls if c[0].shape == (2 * 64, block.d) and c[1] == dt]
    assert len(seam) == 1, f"the seam cast {len(seam)} times, not once: {seam}"
    assert out.dtype == torch.float32, "the MoE output must be widened before `* w_ffn`"
    print(f"[seam cast] 1 x .to({dt}) on the [128, {block.d}] input; gather casts 0; out fp32")


def test_the_bf16_seam_adds_no_host_sync():
    """Gate (c): the new branch and the gather call no host-sync torch function.

    A `TorchFunctionMode` names every `item`/`cpu`/`tolist`/`numpy`/`bool`/
    `__float__` call made inside them; the counters object is device-side and
    the seam's only Python is a dtype check. (This is a CPU run, so it cannot
    catch a CUDA implicit sync — the diff is the evidence for that; this gate
    pins that no EXPLICIT read was added.)
    """
    from dormouse.model.loopblock import LoopBlock

    moe, x, _w, index, order, keep, _d = _cpu_setup(bf16=True, capacity=1.0)
    dt = moe._bank_dtype()
    xb = x.to(dt)
    with _SyncCalls() as spy:
        moe._gather_rows(xb, order[keep] // 2, dt)
    assert spy.names == [], f"the gather reads the host: {spy.names}"

    block = LoopBlock(tiny_cfg(bf16=True, n_experts=4, moe_topk=2, moe_kernel="naive"))
    # The naive reference arm lists its kept expert ids on the host
    # (`torch.unique(...).tolist()`) — pre-existing and not this seam's. Stub
    # the dispatch so what this gate measures is the seam + routing + shared
    # expert, not the reference loop's own known sync.
    block.moe._dispatch = lambda x, w, i, c: torch.zeros(
        x.shape[0], block.moe.f, dtype=torch.float32
    )
    normed = torch.randn(2, 64, block.d)
    with _SyncCalls() as spy:
        block._moe_branch(normed, Counters())
    assert spy.names == [], f"the seam reads the host: {spy.names}"
    print("[no sync] 0 host-sync torch calls in the seam and the gather")


def test_the_bf16_seam_arms_agree():
    """Gate (f): padded vs naive on the same bf16 rows and banks.

    `_triton_dispatch` is CUDA-only (the card gate below covers it); on CPU the
    two arms that can run must agree to the file's bf16 bound (2e-3, two
    rounding orders: batched `bmm` vs per-expert matmul). The max is printed.
    """
    moe, x, weight, index, order, keep, _d = _cpu_setup(bf16=True, capacity=1.0)
    dt = moe._bank_dtype()
    xb = x.to(dt)
    ref = moe._naive_dispatch(xb, weight, index, order, keep, Counters())
    pad = moe._padded_dispatch(xb, weight, index, order, keep)
    rel = _rel(pad, ref)
    print(f"[bf16 arms] padded vs naive rows={int(keep.sum())} max_rel={rel:.3e} bound=2e-3")
    assert rel <= 2e-3, f"{rel:.3e} exceeds the bf16 bound"


def test_the_fused_abi_is_the_gather_the_pass_already_does():
    """`toks`/`offs` are the shipped permute's own outputs, bit for bit.

    The fused kernel's A operand must be EXACTLY the tensor `_triton_dispatch`
    materializes today (`xg = x[owner].to(dt)`, `experts.py:990`), or the
    prototype is a different function and the parity gates measure nothing.
    `torch.equal` on the fp32 rows: an equality here is a statement about the
    map, not about a rounding.
    """
    moe, x, _w, index, order, keep, _dropped = _cpu_setup()
    toks, offs = ms.gather_abi(index, order, keep, moe.n_experts)
    # the shipped path's own three lines, verbatim from experts.py:982-990
    rows = order[keep]
    owner = rows // 2
    kept = index.reshape(-1)[rows]
    xg = x[owner]

    assert toks.dtype == torch.int32 and toks.is_contiguous()
    assert toks.shape == (rows.numel(),) and offs.shape == (moe.n_experts + 1,)
    assert torch.equal(toks.to(torch.int64), owner), "toks is not the gather's row map"
    assert torch.equal(x[toks.to(torch.int64)], xg), "the fused load's A != the gathered copy"
    counts = torch.bincount(kept, minlength=moe.n_experts).to(torch.int32)
    assert torch.equal(offs[1:] - offs[:-1], counts), "offs is not the routing's partition"
    assert int(offs[0]) == 0 and int(offs[-1]) == rows.numel()
    print(
        f"[abi] E={moe.n_experts} rows={rows.numel()} toks={toks.dtype} "
        f"offs={offs.tolist()} x[toks] == x[owner]"
    )


@pytest.mark.parametrize("bf16,bound", [(False, 1e-6), (True, 2e-3)])
def test_the_fused_reference_is_the_modules_own_reference_loop(bf16, bound):
    """`fused_reference` + the weight scatter == `_naive_dispatch`, on CPU.

    The strongest gate this lane can run without a card: it goes through the
    MODULE's own reference arm (`experts._naive_dispatch` — the day-one loop the
    padded and Triton arms are both measured against), so a wrong row map, a
    misaligned `offs`, a swapped bank or a reordered GLU chain all fail here.
    At bf16 the bound is the file's own 2e-3: the reference mirrors the KERNEL's
    rounding chain (fp32 accumulate, `round -> scale -> round`, silu on the
    widened gate) while the loop rounds through bf16 matmuls, so this compares
    two rounding orders, not two implementations.
    """
    moe, x, weight, index, order, keep, _dropped = _cpu_setup(bf16=bf16)
    toks, offs = ms.gather_abi(index, order, keep, moe.n_experts)
    gate_u, up_u, v, gate_s, up_s = moe._banks(moe._bank_dtype())
    y = ms.fused_reference(x, toks, gate_u, up_u, gate_s, up_s, v, offs)

    rows = order[keep]
    owner = rows // 2
    out = torch.zeros(x.shape[0], moe.f, dtype=torch.float32)
    out.index_add_(0, owner, y.to(torch.float32) * weight.reshape(-1)[rows].unsqueeze(-1))
    ref = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    rel = _rel(out, ref)
    print(
        f"[fused reference] bf16={bf16} rows={rows.numel()} E={moe.n_experts} "
        f"max_rel={rel:.3e} bound={bound}"
    )
    assert rel <= bound, f"{rel:.3e} exceeds {bound}"


def test_the_fused_load_reads_only_rows_inside_x():
    """The address audit: the kernel's grid, its masks, and what they can touch.

    `_gather_glu_kernel` adds one address the shipped kernel does not have:
    `toks[row]·D`. A masked lane loads `toks` with `other=0`, so it points at ROW
    0 — in bounds by construction, and the A load's own `m_mask` is what keeps
    that value out of the tile. Both halves are checked here, in Python, over a
    real routing and the DROPLESS grid (where most blocks exit on the first
    branch): every address the pointer arithmetic produces is inside `x`,
    exactly `R` token rows are read, and they are exactly the gathered ones.
    """
    moe, x, _w, index, order, keep, _d = _cpu_setup(capacity=0.0, tokens=64)
    toks, offs = ms.gather_abi(index, order, keep, moe.n_experts)
    block_m = ms.TILES["BLOCK_M"]
    cap = int(toks.numel())  # `_dropless_cap`: R rows in one group, no offs read-back
    e, t, d = moe.n_experts, x.shape[0], x.shape[1]

    read_rows: list[int] = []
    addresses: list[int] = []
    blocks_with_work = 0
    for expert in range(e):
        start, end = int(offs[expert]), int(offs[expert + 1])
        for b in range(-(-cap // block_m)):
            if start + b * block_m >= end:
                continue  # the kernel's early exit: no `toks`, no A, no store
            blocks_with_work += 1
            for j in range(block_m):
                row = start + b * block_m + j
                masked = row >= end
                tok = 0 if masked else int(toks[row])
                addresses.append(tok * d + d - 1)  # the tile's LAST element
                if not masked:
                    read_rows.append(tok)

    assert min(addresses) >= 0 and max(addresses) < t * d, "an address leaves `x`"
    assert len(read_rows) == toks.numel(), "the load must read every gathered row once"
    owner = (order[keep] // 2).tolist()
    assert sorted(read_rows) == sorted(owner), "the rows read are not the gathered rows"
    grid_blocks = e * (-(-cap // block_m))
    counts = torch.bincount(index.reshape(-1)[order[keep]], minlength=e)
    assert blocks_with_work == sum(-(-int(c) // block_m) for c in counts)
    assert grid_blocks > blocks_with_work, "the fixture must exercise the early exit"
    print(
        f"[address audit] grid={grid_blocks} programs, {blocks_with_work} with work, "
        f"{len(read_rows)} rows read, all inside [0, {t * d})"
    )


def test_the_prototype_refuses_a_cpu_call_loudly():
    """A CPU call is an ERROR that names the reason and the escape, not a fallback.

    The module is not wired into `_dispatch`, so it has no COUNTED fallback of
    its own — the honest shape for a prototype is a hard assert naming the arm
    that WOULD run (`SparseMoE._dispatch`'s padded/naive chain).
    """
    moe, x, _w, index, order, keep, _d = _cpu_setup()
    toks, offs = ms.gather_abi(index, order, keep, moe.n_experts)
    gate_u, up_u, v, gate_s, up_s = moe._banks(moe._bank_dtype())
    with pytest.raises(AssertionError) as e:
        ms.fused_expert_gemm_glu(x, toks, gate_u, up_u, gate_s, up_s, v, offs)
    print(f"[cpu refusal] {e.value}")
    assert "contiguous CUDA" in str(e.value)


def test_the_prototype_tiles_are_the_shipped_ones():
    """The comparison to `_glu_kernel` is only fair at the same tiles."""
    if mk is None:
        pytest.skip("no triton in this build")
    assert ms.TILES == mk.TILES, "the prototype's tiles drifted from moe_kernels.TILES"
    assert ms._gather_glu_kernel is not None


def test_the_gather_kernel_compiles_for_this_arch():
    """A COMPILE gate that needs no card — the only way to fail a kernel here.

    `triton.compile` with an explicit `GPUTarget` lowers to PTX/cubin on the
    HOST (ptxas; no device, no launch), which catches every frontend failure a
    new kernel can have — a bad broadcast, an `int32` offset the backend
    rejects, a mask of the wrong rank, a `.to()` on a loaded tile. It is NOT a
    parity gate (the CUDA gate below is) and does not replace one; it is what
    makes a kernel that cannot even be built fail HERE instead of on the first
    training step. Both compute dtypes are built: the fp32 arm is the one whose
    `PREC` is `ieee`, and the bf16 arm is the one that exercises the cast.
    """
    if mk is None or ms._gather_glu_kernel is None:
        pytest.skip("no triton in this build")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as tcompile

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = major * 10 + minor
    else:
        arch = 120  # this box's card; a compile-only gate does not need the device
    target = GPUTarget("cuda", arch, 32)
    t = ms.TILES
    built = []
    for dt, prec in (("bf16", "tf32"), ("fp32", "ieee")):
        sig = {
            "x_ptr": "*fp32",
            "toks_ptr": "*i32",
            "gu_ptr": f"*{dt}",
            "uu_ptr": f"*{dt}",
            "gs_ptr": f"*{dt}",
            "us_ptr": f"*{dt}",
            "v_ptr": f"*{dt}",
            "y_ptr": f"*{dt}",
            "offs_ptr": "*i32",
            "D": "i32",
            "R_": "i32",
            "F_": "i32",
            "BLOCK_M": "constexpr",
            "BLOCK_K": "constexpr",
            "BLOCK_F": "constexpr",
            "RP": "constexpr",
            "PREC": "constexpr",
        }
        consts = {
            "BLOCK_M": t["BLOCK_M"],
            "BLOCK_K": t["BLOCK_K"],
            "BLOCK_F": t["BLOCK_F"],
            "RP": 64,
            "PREC": prec,
        }
        src = ASTSource(ms._gather_glu_kernel, signature=sig, constexprs=consts)
        k = tcompile(src, target=target, options={"num_warps": 4})
        assert "cubin" in k.asm, f"{dt}: no cubin"
        built.append(f"{dt}={len(k.asm['cubin'])}B")
    print(f"[kernel compile] arch=sm_{arch} " + " ".join(built))


# --------------------------------------------------------------------------
# 7. the GPU gates: written, skipped, reason printed
# --------------------------------------------------------------------------


def _card_reason() -> str | None:
    """`None` when a GPU gate may run, else why not — the repo's preflight.

    `test_ns_triton.py`'s pattern (`pgrep` + `nvidia-smi`, no CUDA context on a
    card that is in use). On 2026-10-18 this returned the live pretrain's reason
    and the two gates below did NOT run: the kernel's numbers are written and
    unmeasured, and this docstring is where that is said out loud.
    """
    if not torch.cuda.is_available():
        return "no CUDA device"
    if mk is None or ms._gather_glu_kernel is None:
        return "no triton (the fused kernel is the CUDA arm)"
    try:
        busy = subprocess.run(
            ["pgrep", "-f", "dormouse trai[n]"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a gate's precondition, not a run's
        return f"cannot check the card ({e})"
    if busy:
        return f"a live training run holds the card (pids {busy.replace(chr(10), ' ')})"
    mib = int(used.split()[0]) if used else -1
    if mib >= 1024:
        return f"the card already holds {mib} MiB"
    return None


_GPU_REASON = _card_reason()
CUDA = pytest.mark.skipif(_GPU_REASON is not None, reason=_GPU_REASON or "GPU gate")

DEV = "cuda"


def _cuda_setup(*, bf16: bool, capacity: float):
    moe = SparseMoE(
        tiny_cfg(
            n_experts=4, moe_topk=2, rank=8, d_model=64, d_ffn=128, bf16=bf16, moe_capacity=capacity
        ),
        device=DEV,
    )
    torch.manual_seed(0)
    x = torch.randn(64, moe.d, device=DEV)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    weight = weight / weight.sum(dim=-1, keepdim=True)
    order, keep, dropped = moe._permute(index)
    return moe, x, weight, index, order, keep, dropped


@CUDA
@pytest.mark.parametrize("bf16,bound", [(False, 1e-5), (True, 2e-3)])
def test_the_fused_kernel_is_the_shipped_kernel_on_the_same_rows(bf16, bound):
    """The kernel against `expert_gemm_glu` on identical rows, banks and `offs`.

    The bounds are `tests/test_moe_kernel.py`'s own (1e-5 fp32, 2e-3 bf16). The
    ONLY difference between the two calls is where the A operand comes from —
    `x[toks]` read inside the kernel vs `xg = x[owner].to(dt)` read from a
    permuted copy — so a failure here is a wrong map or a broken mask, never a
    tile or a precision question.
    """
    moe, x, _w, index, order, keep, _d = _cuda_setup(bf16=bf16, capacity=0.0)
    dt = moe._bank_dtype()
    toks, offs = ms.gather_abi(index, order, keep, moe.n_experts)
    gate_u, up_u, v, gate_s, up_s = moe._banks(dt)
    rows = order[keep]
    xg = x[rows // 2].to(dt)
    cap = int(toks.numel())
    got = ms.fused_expert_gemm_glu(x, toks, gate_u, up_u, gate_s, up_s, v, offs, cap=cap)
    ref = mk.expert_gemm_glu(xg, gate_u, up_u, gate_s, up_s, v, offs, cap=cap)
    rel = _rel(got, ref)
    print(f"[fused kernel] bf16={bf16} rows={rows.numel()} max_rel={rel:.3e} bound={bound}")
    assert rel <= bound, f"{rel:.3e} exceeds {bound}"


@CUDA
def test_the_fused_kernel_takes_the_dropless_static_bound():
    """`cap=None` is the dropless route: the bound is `R`, read from a SHAPE.

    The fixture drops rows on purpose (capacity 0.25) so some experts own none,
    which is what the early exit exists for — the same rows the shipped kernel
    serves, with a grid far wider than the work.
    """
    moe, x, _w, index, order, keep, dropped = _cuda_setup(bf16=True, capacity=0.25)
    dt = moe._bank_dtype()
    toks, offs = ms.gather_abi(index, order, keep, moe.n_experts)
    gate_u, up_u, v, gate_s, up_s = moe._banks(dt)
    rows = order[keep]
    xg = x[rows // 2].to(dt)
    tight = ms.fused_expert_gemm_glu(
        x, toks, gate_u, up_u, gate_s, up_s, v, offs, cap=moe._capacity(*index.shape)
    )
    wide = ms.fused_expert_gemm_glu(x, toks, gate_u, up_u, gate_s, up_s, v, offs, cap=None)
    ref = mk.expert_gemm_glu(xg, gate_u, up_u, gate_s, up_s, v, offs, cap=int(toks.numel()))
    print(
        f"[dropless bound] dropped={int(dropped)}/{index.numel()} rows={rows.numel()} "
        f"empty groups={int((offs[1:] == offs[:-1]).sum())} "
        f"tight-vs-ref={_rel(tight, ref):.3e} wide-vs-ref={_rel(wide, ref):.3e}"
    )
    assert torch.equal(tight, wide), "the bound must not change the answer"
    assert _rel(wide, ref) <= 2e-3


@CUDA
def test_the_bf16_seam_arms_agree_on_the_card():
    """Gate (f) on the card: triton vs padded vs naive on the same bf16 rows.

    One routing, one bank, one `x.to(dt)` — so the only difference is the arm.
    `moe_capacity = 0.25` gives all three a capacity to work with (the padded
    arm refuses dropless by design); the DROPS are the same `keep` for all
    three, which is what makes the comparison a comparison of the arithmetic.
    `bf16` is the arm this gate exists for: it is the seam's dtype
    (`moe_gather`/the triton path run on the bank's dtype), and the fp32 arm is
    `test_experts_gpu.py`'s.
    """
    bf16 = True
    moe, x, weight, index, order, keep, _d = _cuda_setup(bf16=bf16, capacity=0.25)
    dt = moe._bank_dtype()
    xb = x.to(dt)
    ref = moe._naive_dispatch(xb, weight, index, order, keep, Counters())
    pad = moe._padded_dispatch(xb, weight, index, order, keep)
    tri = moe._triton_dispatch(xb, weight, index, order, keep)
    rel_pad, rel_tri = _rel(pad, ref), _rel(tri, ref)
    print(
        f"[bf16 seam on the card] bf16={bf16} rows={int(keep.sum())} "
        f"padded={rel_pad:.3e} triton={rel_tri:.3e} bound=2e-3"
    )
    assert rel_pad <= 2e-3 and rel_tri <= 2e-3


# --------------------------------------------------------------------------
# 8-9. the WIRED arm (`moe_gather = "kernel"`): the branch, counters, refusals
# --------------------------------------------------------------------------


def _torch_combine(y, w, owner, rowmap, n, *, ws=None, tiles=None):
    """`moe_kernels.combine`'s forward in torch — the CPU proxy for its scatter.

    `combine` is a Triton kernel, so a card-less gate substitutes the FORWARD it
    performs (the same three lines `_Combine.forward` runs) and lets autograd
    build the previous adjoint. Its fused backward (the `rowmap` path) stays
    unmeasured here, and `_wired_arm`'s docstring says so.
    """
    out = torch.zeros(n, y.shape[1], dtype=torch.float32)
    out.index_add_(0, owner, y.to(torch.float32) * w.unsqueeze(-1))
    return out


def _wired_arm(monkeypatch) -> None:
    """Point the arm's three Triton seams at their torch twins, for a CPU gate.

    `_kernel_forward` IS the kernel's forward, `_shipped_forward` IS the
    grouped adjoint's forward, and `combine` IS the scatter. Substituting all
    three means what this gate proves is the INDEX ARITHMETIC and the wiring —
    which rows `toks` names, that `dx` lands on those token rows, that the
    weights and `offs` line up — and NOT any of the three kernels. The CUDA
    gate below is the half that measures those.
    """
    monkeypatch.setattr(
        ms,
        "_kernel_forward",
        lambda x, toks, gu, uu, gs, us, v, offs, cap=None: ms.fused_reference(
            x, toks, gu, uu, gs, us, v, offs
        ),
    )
    monkeypatch.setattr(
        ms,
        "_shipped_forward",
        lambda xg, gu, uu, gs, us, v, offs, *, cap, seg, ws: ms.fused_reference(
            xg, torch.arange(xg.shape[0], dtype=torch.int32), gu, uu, gs, us, v, offs
        ),
    )
    monkeypatch.setattr(mk, "combine", _torch_combine)


def test_the_gather_arm_is_the_naive_pass_on_cpu(monkeypatch):
    """`_gather_dispatch` == `_naive_dispatch`: forward, `dx`, every expert grad.

    One routing feeds both arms, so the only difference is where the A operand
    comes from. fp32 has no rounding between them, and 1e-5 is two orders above
    the measured disagreement (1.5e-7) and far BELOW any wiring error: a
    permuted `toks` moves `dx` by O(1), so this gate cannot pass on a wrong map.
    It is also the only gradient evidence for the arm that a CPU can produce —
    `_GatherGlu`'s backward is the shipped adjoint, so its MATH is not what is
    under test here; the row map is.
    """
    if mk is None:
        pytest.skip("no triton in this build: the shipped grouped GEMM is the adjoint")
    _wired_arm(monkeypatch)
    moe, x, weight, index, order, keep, _d = _cpu_setup(bf16=False)

    xg = x.clone().requires_grad_(True)
    out = moe._gather_dispatch(xg, weight, index, order, keep, Counters())
    out.float().pow(2).sum().backward()
    dx = xg.grad.detach().clone()
    grads = {n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None}
    moe.zero_grad(set_to_none=True)

    xn = x.clone().requires_grad_(True)
    ref = moe._naive_dispatch(xn, weight, index, order, keep, Counters())
    ref.float().pow(2).sum().backward()
    ref_grads = {
        n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None
    }

    rel_fwd = _rel(out, ref)
    rel_dx = _rel(dx, xn.grad)
    assert set(grads) == set(ref_grads) and len(grads) == 20, sorted(grads)
    rel_max = max(_rel(g, ref_grads[n]) for n, g in grads.items())
    print(
        f"[wired arm] rows={int(keep.sum())} E={moe.n_experts} fwd={rel_fwd:.1e} "
        f"dx={rel_dx:.1e} grads({len(grads)})={rel_max:.1e} bound=1e-5"
    )
    assert rel_fwd <= 1e-5, f"forward {rel_fwd:.3e}"
    assert rel_dx <= 1e-5, f"dx {rel_dx:.3e}"
    assert rel_max <= 1e-5, f"parameter gradients {rel_max:.3e}"


def test_the_gather_counters_prove_the_path_ran(monkeypatch):
    """`moe_gather_forward` + `moe_gather_rows` are the arm's evidence line.

    ADR-0019's rule for a new arm: a run labelled with it must show that it
    ran. `_gather_reason` is forced to `None` because this fixture is a CPU one
    and that gate is the device's — the object under test here is `_dispatch`'s
    branch (the counters and the routing), not the device check, which the next
    test pins from the other side.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    _wired_arm(monkeypatch)
    moe, x, weight, index, order, keep, _d = _cpu_setup(bf16=False)
    assert moe.moe_gather == "off", "the fixture must start from the default"
    moe.moe_gather = "kernel"
    monkeypatch.setattr(moe, "_gather_reason", lambda x: None)
    # The counter must be the branch's OWN evidence: a run that counts the arm
    # and then returns the shipped arm's value is the defect this pins.
    calls: list[int] = []
    real = moe._gather_dispatch

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(moe, "_gather_dispatch", spy)
    counters = Counters()
    out = moe._dispatch(x, weight, index, counters)
    rows = int(keep.sum())
    assert calls, "the branch counted the arm but never called it"
    assert counters["moe_gather_forward"] == 1, dict(counters)
    assert counters.drain()["moe_gather_rows"] == rows
    assert "moe_gather_fallback" not in counters
    rel = _rel(out, moe._naive_dispatch(x, weight, index, order, keep, Counters()))
    print(f"[counters] forward=1 rows={rows} (R={index.numel()}) rel={rel:.1e}")
    assert rel <= 1e-5


def test_the_gather_arm_is_counted_when_the_kernel_cannot_run():
    """A CPU call is the COUNTED fallback, never silence and never the arm.

    `moe_gather="kernel"` on a device without the kernel: `moe_gather_fallback`
    is counted, `counters.why` names the reason, `moe_gather_forward` stays
    absent, and the run is the arm `moe_kernel` names — which is what makes the
    counter a statement about THIS call rather than a config echo.
    """
    moe, x, weight, index, order, keep, _d = _cpu_setup(bf16=False)
    moe.moe_gather = "kernel"
    counters = Counters()
    out = moe._dispatch(x, weight, index, counters)
    why = getattr(counters, "why", "")
    ref = moe._naive_dispatch(x, weight, index, order, keep, Counters())
    print(f"[counted fallback] fallback={counters['moe_gather_fallback']} why={why!r}")
    assert counters["moe_gather_fallback"] == 1
    assert "moe_gather_forward" not in counters
    assert "cpu" in why and "gather" in why
    assert torch.equal(out, ref), "the fallback must be the shipped arm's own numbers"


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"bf16": False}, "bf16=false"),
        ({"use_byteflow": True}, "use_byteflow=true"),
        ({"moe_topk_mode": "entropy"}, "moe_topk_mode='entropy'"),
        ({"factor_qat": "ternary"}, "factor_qat='ternary'"),
    ],
)
def test_the_gather_route_is_refused_without_its_evidence(over, needle):
    """The four combinations the arm has no evidence for are LOUD, with escapes.

    Each is a REFUSAL and not a clamp: the arm's forward is gated on the bf16
    loop-stack route (its launcher's tiles/stages), on the constant width (the
    ABI is the permute's own `keep`) and on an unquantized bank (the kernel
    reads `gate_u`/`up_u`/`v` after `factor_qat_project`). The message must
    name the escape, or it is a wall rather than a gate.
    """
    cfg = {"bf16": True}  # the route the arm is gated on; `over` moves ONE thing off it
    cfg.update(over)
    with pytest.raises(LoudError) as e:
        validate(tiny_cfg(moe_gather="kernel", **cfg))
    msg = str(e.value)
    print(f"[refusal] {needle}: {msg[:120]}...")
    assert needle in msg
    assert "moe_gather='off'" in msg


def test_the_gather_arm_is_off_by_default_and_names_only_known_arms():
    """`off` draws nothing and validates on every tiny fixture; a typo is LOUD.

    The default matters twice: `off` is every number on record, and a field that
    is on by default would have to be argued from a measurement that does not
    exist yet (`scripts/bench_gather.py` has not run).
    """
    assert ModelConfig().moe_gather == "off"
    validate(tiny_cfg())
    for over in ({"moe_gather": "kernel"}, {"moe_gather": "off"}):
        validate(tiny_cfg(bf16=True, **over))
    with pytest.raises(LoudError) as e:
        validate(tiny_cfg(moe_gather="sonic"))
    assert "moe_gather='sonic'" in str(e.value)
    print("[default] moe_gather='off'; 'kernel' validates only on the bf16 route")


@CUDA
@pytest.mark.parametrize("bf16", [False, True])
def test_the_wired_arm_is_the_shipped_triton_arm(bf16):
    """The arm vs `moe_kernel="triton"`: logits AND gradients, rel <= 1e-5.

    One seed, two modules, one `state_dict` — so the router draws the same
    `index` and the only difference is the arm. `moe_gather_forward` on the
    gather side is the evidence the branch ran; without it the comparison would
    be triton against itself and would pass no matter what.

    At bf16 the only expected difference is WHERE the cast happens:
    `x[toks].to(dt)` in torch against `.to(y_ptr.dtype)` on the loaded tile,
    both round-to-nearest. So 1e-5 is what "the same numbers" means here, and a
    red bf16 line is that cast disagreeing — a finding, not noise.
    """
    base = tiny_cfg(
        n_experts=8,
        moe_topk=2,
        rank=8,
        d_model=64,
        d_ffn=128,
        bf16=bf16,
        moe_kernel="triton",
        moe_capacity=0.0,
    )
    torch.manual_seed(3)
    shipped = SparseMoE(base, device=DEV)
    torch.manual_seed(3)
    gather = SparseMoE(dataclasses.replace(base, moe_gather="kernel"), device=DEV)
    gather.load_state_dict(shipped.state_dict())
    x = torch.randn(64, shipped.d, device=DEV)

    def run(moe):
        xg = x.clone().requires_grad_(True)
        counters = Counters()
        out = moe(xg, counters)
        out.float().pow(2).sum().backward()
        grads = {
            n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None
        }
        return out.detach(), xg.grad.detach(), grads, counters

    ref_out, ref_dx, ref_grads, ref_c = run(shipped)
    got_out, got_dx, got_grads, got_c = run(gather)
    assert "moe_gather_forward" not in ref_c
    assert got_c["moe_gather_forward"] == 1, dict(got_c)
    assert set(got_grads) == set(ref_grads)
    rel_out = _rel(got_out, ref_out)
    rel_dx = _rel(got_dx, ref_dx)
    rel_max = max(_rel(got_grads[n], ref_grads[n]) for n in ref_grads)
    print(
        f"[wired arm on the card] bf16={bf16} out={rel_out:.1e} dx={rel_dx:.1e} "
        f"grads={rel_max:.1e} rows={int(got_c.drain()['moe_gather_rows'])} bound=1e-5"
    )
    assert rel_out <= 1e-5, f"logits {rel_out:.3e}"
    assert rel_dx <= 1e-5, f"dx {rel_dx:.3e}"
    assert rel_max <= 1e-5, f"parameter gradients {rel_max:.3e}"
