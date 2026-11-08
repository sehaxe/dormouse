"""The combine's one-pass forward, and whether a bf16 store is allowed.

The census prices the combine at 1 476 MB and 4 launches per forwarded pass —
**81 % of the routing cluster's bytes**, the largest routing row in the region
(`scripts/dispatch_census.py`). Two of its three forward passes exist only
because torch needs a materialized `[n·k, f]` fp32 operand:

    out = zeros(n, f)                     134 MB, write
    out.index_add_(owner, y.to(f32) * w)  402 + 537 MB, read/write
    (then `weighted.to(dt)` in the caller  134 + 67 MB)

`moe_combine="fused"` replaces the three with ONE token-major kernel that reads
each token's k rows of `y` and writes the token's output, accumulating in fp32
registers:

    torch/fp32   1 476 526 080 B / 4 launches
    fused/fp32     671 350 784 B / 2 launches   (-805 MB, -2)
    fused/bf16     537 133 056 B / 2 launches   (-939 MB, -2)
      and the caller's `weighted.to(dt)` disappears (-201 MB) — see the census.

**fp32 is bit-identical**, and that is a property, not a measurement: `y` is
bf16, `w` is fp32, the product is fp32 in both arms, and with k=2 fp32 addition
is commutative, so `index_add_`'s row order cannot change the answer. The gate
asserts `torch.equal` on the logits AND every parameter gradient.

**bf16 is NOT bit-identical and does NOT meet the 1e-3 logits target.** It rounds
the routed activation once (2^-9 = 1.95e-3 relative per element) and, because the
MoE then adds the shared expert in bf16, that rounding happens one step earlier
than today's. Measured on this fixture over three seeds: **logits 4.1e-3 …
6.2e-3 relative, gradients 3.6e-3 … 4.9e-3 relative**. The gate pins 1e-2 for
both (the measured worst case plus margin) and the docstring says plainly that
the 1e-3 logits target is NOT met — which is why the default stays fp32.

The CPU gates substitute the two Triton calls with their torch references
(`combine_reference` is the kernel's math line for line); the kernel's
transcription is the GPU parity gate's job and is written and skipped.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest
import torch

from dormouse.config import ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import moe_kernels as mk
from dormouse.model.experts import SparseMoE
from dormouse.model.model import DormouseModel

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**over) -> dict:
    base = dict(
        d_model=64,
        n_heads=4,
        head_dim=16,
        d_ffn=128,
        vocab=256,
        max_seq_len=64,
        max_iter=2,
        rank=8,
        n_experts=4,
        moe_topk=2,
        engram_rows=64,
        engram_dim=8,
        bf16=True,
        shared_rank=16,
    )
    base.update(over)
    return base


def _routing(moe: SparseMoE, x: torch.Tensor):
    logits = x @ moe.router.weight.t()
    probs = torch.softmax(logits.float(), dim=-1)
    index = probs.topk(moe.topk, dim=-1).indices
    weight = torch.gather(probs, 1, index)
    return index, weight / weight.sum(dim=-1, keepdim=True)


# --------------------------------------------------------------------------- #
# A. defaults and refusals
# --------------------------------------------------------------------------- #
def test_the_combine_flags_are_off_by_default() -> None:
    cfg = ModelConfig()
    assert cfg.moe_combine == "torch" and cfg.moe_combine_dtype == "fp32"
    print("[defaults] moe_combine='torch' moe_combine_dtype='fp32'")


def test_the_refusals_carry_the_escape() -> None:
    cases = [
        dict(moe_combine="bogus"),
        dict(moe_combine_dtype="fp16"),
        # bf16 storage is the one-pass kernel's; the torch arm cannot express it.
        dict(moe_combine="torch", moe_combine_dtype="bf16"),
        dict(moe_combine="fused", moe_combine_dtype="bf16", bf16=False),
        # the padded/batched banks scatter themselves: the arm would be inert
        dict(moe_combine="fused", moe_kernel="batched", moe_capacity=1.0),
    ]
    for over in cases:
        with pytest.raises(LoudError) as e:
            validate(ModelConfig(**{**_cfg(), **over}))
        msg = str(e.value)
        assert "Escape:" in msg, msg
        print(f"[refusal] {over}: {msg.splitlines()[-1][:100]}")


# --------------------------------------------------------------------------- #
# B. the reference IS the torch form (fp32) and one rounding (bf16)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3])
def test_combine_reference_is_the_torch_scatter(k) -> None:
    torch.manual_seed(0)
    n, f, R = 37, 32, 37 * k
    y = torch.randn(R, f).to(torch.bfloat16)
    w = torch.rand(R, dtype=torch.float32)
    # Any distinct row->(token, slot) map: the rows are expert-major in
    # production, so a random permutation is the honest fixture.
    rowmap = torch.randperm(R).to(torch.int32).view(n, k)
    owner = torch.empty(R, dtype=torch.int64)
    owner[rowmap.reshape(-1)] = torch.arange(n).repeat_interleave(k)

    want = torch.zeros(n, f, dtype=torch.float32)
    want.index_add_(0, owner, y.to(torch.float32) * w.unsqueeze(-1))
    got = mk.combine_reference(y, w, rowmap, n, out_dtype=torch.float32)
    if k <= 2:
        assert torch.equal(got, want), f"k={k}: fp32 must be bit-identical"
    else:
        d = float((got - want).abs().max())
        assert torch.allclose(got, want, atol=1e-6, rtol=0), f"k={k}: {d}"
    # bf16: the SAME fp32 sum, rounded ONCE on the store.
    got16 = mk.combine_reference(y, w, rowmap, n, out_dtype=torch.bfloat16)
    assert torch.equal(got16, want.to(torch.bfloat16)), (
        "the bf16 store must round the fp32 sum once"
    )
    print(f"[reference] k={k}: fp32 matches the torch scatter; bf16 == its single rounding")


def test_a_dropped_row_contributes_nothing() -> None:
    y = torch.tensor([[1.0], [2.0], [4.0], [8.0]], dtype=torch.bfloat16)
    w = torch.tensor([1.0, 1.0, 1.0, 1.0])
    rowmap = torch.tensor([[0, 1], [-1, 2], [3, -1]], dtype=torch.int32)
    got = mk.combine_reference(y, w, rowmap, 3)
    assert got.flatten().tolist() == [3.0, 4.0, 8.0]
    print("[reference] -1 rows are masked, not wrapped to the last row")


# --------------------------------------------------------------------------- #
# C. the wiring: `_triton_dispatch` hands the arm its own switches
# --------------------------------------------------------------------------- #
def test_the_dispatch_passes_the_combine_switches(monkeypatch) -> None:
    moe = SparseMoE(
        ModelConfig(**_cfg(moe_capacity=0.0, moe_combine="fused", moe_combine_dtype="bf16")),
        device="cpu",
    )
    x = torch.randn(8, moe.d)
    index, weight = _routing(moe, x)
    order, keep, _ = moe._permute(index)
    seen: list[dict] = []

    def glu_stub(xg, gu, uu, gs, us, v, offs, **kw):
        return mk._glu_torch(xg, gu, uu, gs, us, v, offs)

    def combine_stub(y, w, owner, rowmap, n, *, ws=None, tiles=None, counters=None, **kw):
        seen.append(kw)
        return torch.zeros(n, y.shape[1], dtype=torch.float32)

    monkeypatch.setattr(mk, "expert_gemm_glu", glu_stub)
    monkeypatch.setattr(mk, "combine", combine_stub)
    moe._triton_dispatch(x, weight, index, order, keep, dropless=True, counters=Counters())
    assert seen and seen[0]["fused"] is True and seen[0]["dtype"] == "bf16", seen
    print(f"[wiring] _triton_dispatch -> combine({seen[0]})")


# --------------------------------------------------------------------------- #
# D. model-level parity: fp32 bit-identical, bf16 measured
# --------------------------------------------------------------------------- #
def _model(combine: str, dtype: str, seed: int = 0) -> DormouseModel:
    cfg = ModelConfig(**_cfg(moe_capacity=0.0, moe_combine=combine, moe_combine_dtype=dtype))
    torch.manual_seed(seed)
    return DormouseModel(cfg)


def _forward_backward(model, ids):
    logits, *_ = model(ids)
    model.zero_grad()
    logits.float().pow(2).mean().backward()
    grads = torch.cat(
        [p.grad.reshape(-1) for _, p in sorted(model.named_parameters()) if p.grad is not None]
    )
    return logits.detach().float(), grads


@pytest.fixture()
def grouped_cpu(monkeypatch):
    """Drive `_triton_dispatch` on CPU with both Triton seams as torch references.

    `_glu_torch` is the shipped launch-refusal reference for the expert pass and
    `combine_reference` is the forward kernel's math; both are differentiable, so
    the proxy measures the COMBINE's dtype question and says the kernels
    themselves are the skipped GPU gate's.
    """

    def glu_stub(xg, gu, uu, gs, us, v, offs, **kw):
        return mk._glu_torch(xg, gu, uu, gs, us, v, offs)

    def combine_proxy(
        y, w, owner, rowmap, n, *, ws=None, tiles=None, counters=None, fused=False, dtype="fp32"
    ):
        if not fused:  # the `torch` arm: `_Combine.forward`'s three passes
            out = torch.zeros(n, y.shape[1], dtype=torch.float32)
            out.index_add_(0, owner, y.to(torch.float32) * w.unsqueeze(-1))
            return out
        return mk.combine_reference(
            y, w, rowmap, n, out_dtype=torch.float32 if dtype == "fp32" else y.dtype
        )

    def via_triton(self, x, weight, index, order, keep, keep_rows, counters):
        return self._triton_dispatch(
            x,
            weight,
            index,
            order,
            keep_rows,
            dropless=True,
            fused_bwd=True,
            counters=counters,
        )

    monkeypatch.setattr(mk, "expert_gemm_glu", glu_stub)
    monkeypatch.setattr(mk, "combine", combine_proxy)
    monkeypatch.setattr(SparseMoE, "_dropless_dispatch", via_triton)


def test_the_fused_fp32_combine_is_bit_identical_on_the_model(grouped_cpu) -> None:
    ids = torch.randint(0, 256, (2, 16))
    base, gbase = _forward_backward(_model("torch", "fp32"), ids)
    fused, gfused = _forward_backward(_model("fused", "fp32"), ids)
    assert torch.equal(base, fused), "fused fp32 moved the logits"
    assert torch.equal(gbase, gfused), "fused fp32 moved the gradients"
    print("[model] fused/fp32: logits and every parameter grad are bit-identical")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_bf16_combine_store_parity_is_measured(grouped_cpu, seed) -> None:
    """Measured, not claimed: ~4-6e-3 on logits, ~4-5e-3 on gradients.

    The 1e-3 logits target is NOT met and the default stays fp32; this gate pins
    the actual trade so a future claim has to move a number, not rewrite prose.
    """
    ids = torch.randint(0, 256, (2, 16))
    base, gbase = _forward_backward(_model("torch", "fp32", seed), ids)
    out16, gout16 = _forward_backward(_model("fused", "bf16", seed), ids)

    def rel(a, b):
        return float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))

    rl, rg = rel(base, out16), rel(gbase, gout16)
    assert rl <= 1e-2, f"logits moved {rl:.2e} > 1e-2"
    assert rg <= 1e-2, f"gradients moved {rg:.2e} > 1e-2"
    print(f"[model] fused/bf16 seed={seed}: logits rel {rl:.2e}, grads rel {rg:.2e} (1e-3 not met)")


# --------------------------------------------------------------------------- #
# E. the census: the bytes each arm buys
# --------------------------------------------------------------------------- #
def _census():
    spec = importlib.util.spec_from_file_location(
        "dispatch_census", ROOT / "scripts" / "dispatch_census.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _m400b() -> ModelConfig:
    preset = tomllib.loads((ROOT / "configs" / "m400.toml").read_text())
    return dataclasses.replace(
        ModelConfig(**preset),
        moe_capacity=0.0,
        share_u=True,
        share_u_mode="plane",
        moe_lb_mode="prop",
        moe_lb_gamma=0.002,
        moe_lb_clamp=4.0,
    )


def test_the_census_prices_the_combine_arms() -> None:
    dc = _census()

    def group(over, name):
        _, rows = dc.census(dataclasses.replace(_m400b(), **over), 8, 1024)
        return sum(r.nbytes for r in rows if r.group == name), sum(
            r.launches for r in rows if r.group == name
        )

    torch_b, torch_l = group({}, "combine")
    fp32_b, fp32_l = group(dict(moe_combine="fused"), "combine")
    bf16_b, bf16_l = group(dict(moe_combine="fused", moe_combine_dtype="bf16"), "combine")

    assert torch_b == 1_476_526_080 and torch_l == 4
    assert fp32_l == bf16_l == 2
    assert torch_b - fp32_b == 805_175_296, torch_b - fp32_b
    assert torch_b - bf16_b == 939_393_024, torch_b - bf16_b
    # bf16 also deletes the caller's `weighted.to(dt)` (down group).
    down_t, _ = group({}, "down")
    down_16, _ = group(dict(moe_combine="fused", moe_combine_dtype="bf16"), "down")
    assert down_t - down_16 == 201_326_592
    print(
        f"[census] combine {torch_b / 1e6:.1f} MB/4 -> fp32 {fp32_b / 1e6:.1f} MB/2 "
        f"(-{(torch_b - fp32_b) / 1e6:.1f}) -> bf16 {bf16_b / 1e6:.1f} MB/2 "
        f"(plus -{201_326_592 / 1e6:.1f} MB in down)"
    )


def test_the_bank_epoch_census_is_host_time_not_bytes() -> None:
    dc = _census()
    _, rows = dc.census(_m400b(), 8, 1024)
    host_off = sum(r.host_us for r in rows)
    bank_off = sum(r.nbytes for r in rows if r.group == "bank")
    _, rows = dc.census(dataclasses.replace(_m400b(), moe_bank_epoch=True), 8, 1024)
    host_on = sum(r.host_us for r in rows)
    bank_on = sum(r.nbytes for r in rows if r.group == "bank")
    assert abs(host_off - 512 * dc.KEY_WALK_US_PER_EXPERT) < 0.1, host_off
    assert host_on < 1.0, host_on
    assert bank_on == bank_off, "the epoch must not move bank bytes or launches"
    print(
        f"[census] bank key: {host_off:.0f} us/pass -> {host_on:.1f} "
        f"(x3 passes = {(host_off - host_on) * 3 / 1000:.1f} ms/step); bytes unchanged"
    )


# --------------------------------------------------------------------------- #
# F. the kernel builds; its numbers need the card
# --------------------------------------------------------------------------- #
def test_the_combine_kernel_builds_for_sm_120() -> None:
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as tcompile

    arch = 120
    if torch.cuda.is_available():
        mj, mn = torch.cuda.get_device_capability()
        arch = mj * 10 + mn
    target = GPUTarget("cuda", arch, 32)
    built = []
    for dt in ("bf16", "fp32"):
        src = ASTSource(
            fn=mk._combine_fwd_kernel,
            signature={
                "y_ptr": f"*{dt}",
                "w_ptr": "*fp32",
                "rowmap_ptr": "*i32",
                "out_ptr": f"*{dt}",
                "N": "i32",
                "F_": "i32",
                "K": "constexpr",
                "TOP_K": "constexpr",
                "BLOCK_T": "constexpr",
                "BLOCK_F": "constexpr",
            },
            constexprs={"K": 2, "TOP_K": 2, "BLOCK_T": 16, "BLOCK_F": 128},
        )
        k = tcompile(src, target=target, options={"num_warps": 4})
        assert "cubin" in k.asm
        built.append(f"{dt}={len(k.asm['cubin'])}B")
    # The backward with the bf16 STORE (its grad_out is bf16 then) — the pair
    # `moe_combine_dtype="bf16"` creates.
    bwd = ASTSource(
        fn=mk._combine_bwd_kernel,
        signature={
            "dout_ptr": "*bf16",
            "y_ptr": "*bf16",
            "w_ptr": "*fp32",
            "rowmap_ptr": "*i32",
            "dy_ptr": "*bf16",
            "dw_ptr": "*fp32",
            "N": "i32",
            "F_": "i32",
            "K": "constexpr",
            "TOP_K": "constexpr",
            "BLOCK_T": "constexpr",
            "BLOCK_F": "constexpr",
        },
        constexprs={"K": 2, "TOP_K": 2, "BLOCK_T": 16, "BLOCK_F": 128},
    )
    kb = tcompile(bwd, target=target, options={"num_warps": 4})
    assert "cubin" in kb.asm
    built.append(f"bwd_bf16={len(kb.asm['cubin'])}B")
    print(f"[compile] sm_{arch}: _combine_fwd_kernel " + " ".join(built))


def test_the_fused_combine_counter_is_counted_by_the_kernel() -> None:
    """The kernel counts `moe_combine_fused` after it LAUNCHES (ADR-0019).

    Written and skipped: `combine` asserts CUDA, and the card is on a multi-day
    run. The CPU wiring gate above asserts the branch hands it `fused=True`;
    this is the arm's own "it ran" evidence for a log reader.

    Two arms, because the counter is dtype-dependent: `combine` stores fp32 by
    default, and only a NON-fp32 store counts `moe_combine_bf16`
    (`moe_kernels.py:1604`). The old single call used the default and asserted
    the bf16 counter — a test that could never pass.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs the card: the combine asserts CUDA")
    y = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
    w = torch.rand(16, dtype=torch.float32, device="cuda")
    rowmap = torch.arange(16, dtype=torch.int32, device="cuda").view(8, 2)
    c = Counters()
    mk.combine(y, w, rowmap.reshape(-1), rowmap, 8, counters=c, fused=True)
    assert c["moe_combine_fused"] == 1 and "moe_combine_bf16" not in c, c
    c2 = Counters()
    mk.combine(y, w, rowmap.reshape(-1), rowmap, 8, counters=c2, fused=True, dtype="bf16")
    assert c2["moe_combine_fused"] == 1 and c2["moe_combine_bf16"] == 1, c2
