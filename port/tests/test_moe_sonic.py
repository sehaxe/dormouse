"""Gates for the SonicMoE analysis and the gather-fused prototype (`model/moe_sonic.py`).

The module is ANALYSIS + a prototype, not a shipped arm: nothing in `experts.py`
imports it, and the card is held by the live pretrain, so the split here is
deliberate and stated per gate.

**Proven on CPU** (every one of these runs in the plain suite):

1. **the shape table is the configs' own** — `SHAPES`' fields are read back out
   of `configs/medium.toml` / `configs/b1.toml` with `tomllib`, so the byte table
   cannot drift from the geometry it prices;
2. **the traffic table is hand-computable** — every literal in the module
   docstring's table is asserted, plus the two closed forms the fusion turns on
   (`forward_gather_saving == 12·R·d`, `backward_gather_saving == −2·R·f`);
3. **the premise of the two-copies claim is measured** — the MoE input really
   arrives fp32 while the compute dtype is bf16, through a real `DormouseModel`
   forward (not through a comment);
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
   `tests/test_moe_kernel.py::test_the_fused_kernels_compile_for_this_arch`).

**Needs the card** (written, `skipif`'d on the repo's preflight with its reason
printed, so a reader of a skip line knows what was not measured): the kernel's
NUMBERS — the fused kernel against `moe_kernels.expert_gemm_glu` on the same
rows and the same banks, at fp32 and bf16, on both the tight capacity bound and
the dropless static one. On 2026-10-18 the preflight found the card IDLE (849
MiB, no `dormouse train` process) and those gates RAN and PASSED rather than
skipping — `model/moe_sonic.py`'s docstring records the same. No timing gate: a
one-shot parity call says nothing about speed, and the honest perf question at
this shape is the tile-utilisation one the module docstring prices
(`tile_work`), which needs a bench, not a gate.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.instrument import Counters
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

    The two closed forms are the point of the table: the forward saving is
    `12·R·d` (four [R,d] transfers of which the fusion keeps one 4-byte read)
    and the backward saving is exactly `−2·R·f` — NEGATIVE, because a gathered
    `grad_out` is fp32 and would be read twice where the shipped pass writes a
    bf16 `dy` once and reads it twice. A future change that makes the backward
    a win has to change this test, which is what makes the claim falsifiable.
    """
    medium, b1 = ms.SHAPES
    m = ms.pass_bytes(medium)
    assert medium.rows == 49_152
    assert m["x_read"] == 201_326_592  # one [R,d] fp32 read (the source rows)
    assert m["xg_write"] == 201_326_592  # the fp32 `index_select` result
    assert m["xg_cast_read"] == 201_326_592  # `.to(dt)`'s read
    assert m["xg_dt_write"] == 100_663_296  # the bf16 copy
    assert m["xg_read"] == 100_663_296  # `_glu_kernel`'s A load
    assert m["forward_gather_saving"] == 603_979_776
    assert m["forward_total"] == 1_798_438_912
    assert m["y_write"] == m["y_read"] == 268_369_920
    assert m["out_write"] == 268_369_920  # [T,f] fp32
    assert m["banks_read"] == 156_565_504
    assert m["cached_activation"] == 400_490_496
    assert m["backward_adjoint_current"] == 1_073_479_680
    assert m["backward_adjoint_fused"] == 1_341_849_600
    assert m["backward_gather_saving"] == -268_369_920

    b = ms.pass_bytes(b1)
    assert b1.rows == 16_384
    assert b["x_read"] == 134_217_728
    assert b["xg_dt_write"] == 67_108_864
    assert b["forward_gather_saving"] == 402_653_184
    assert b["forward_total"] == 2_336_849_920
    assert b["banks_read"] == 1_252_655_104  # 54% of the forward: the fine-grained bill
    assert b["cached_activation"] == 256_540_672
    assert b["backward_gather_saving"] == -178_946_048

    for shape, table in ((medium, m), (b1, b)):
        r, d, f = shape.rows, shape.d, shape.f
        assert table["forward_gather_saving"] == 12 * r * d, "4 transfers, 3 fp32 + 2 bf16, keep 1"
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


def test_the_moe_input_is_fp32_so_the_gather_is_two_copies():
    """The premise the whole byte table rests on, MEASURED through a real forward.

    `xg = x[owner].to(dt)` (`experts.py:990`) is two materializations only if
    `x` is not already `dt`. It is not: the MoE input is `nn.RMSNorm`'s output
    over the fp32 loop state (`loopblock.py:278,130`), in BOTH precision modes.
    If that ever changes, `X_BYTES` and every number above it change with it —
    which is why this is a gate and not a sentence in a docstring.
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
        assert seen["dtype"] == torch.float32, (
            f"bf16={bf16}: the MoE input moved to {seen['dtype']}"
        )
        assert (dt == torch.bfloat16) == bf16
        print(f"[input dtype] bf16={bf16}: compute {dt}, input {seen['dtype']} -> X_BYTES=4")


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
