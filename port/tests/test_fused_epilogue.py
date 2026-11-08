"""The fused epilogue's gates: clean-main pins, parity, refusals, counters.

Three arms are judged here, each against its OWN named rule:

1. **The read chain** (`fused_residual.epilogue_read`, `fused_epilogue=true`):
   one Triton pass over the lanes that does the lane RMSNorm, the eight
   hyper-network dots, `ctx`, and the body's `RMSNorm(ctx)·LayerRoPE-gain`.
   The CPU gate compares its torch mirror ([`epilogue_reference`]) against the
   shipped eager modules (`MHCMix.pre_mix` + `F.rms_norm`) on outputs AND
   gradients at 1e-6 relative; the CUDA gate runs the real kernel against the
   same eager calls. Two scratch mutations (the gain the kernel applies, and
   the normalisation itself) must turn the comparison red.
2. **The latent projection** (`fused_latent_proj=true`): `q/k/v/g` become ONE
   GEMM on `cat([W_q,W_k,W_v,W_g])` and a view-split. NOT bit-identical to the
   four `nn.Linear` calls in general (a GEMM may reorder the K reduction), so
   the rule is 1e-6 fp32 / 1e-2 bf16, and the test says so instead of claiming
   equality. The view (no copy) and the single-matmul count are asserted.
3. **OFF is clean main**: the sha256 pins below were captured on a detached
   clean worktree of `main` (`3e736ae`) BEFORE this branch edited anything, at
   `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)`, with the fused flags at
   their default (OFF, where they add no parameter, no RNG draw and no op).
"""

from __future__ import annotations

import contextlib
import hashlib

import pytest
import torch

from dormouse.config import ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import fused_residual as fr

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the card")

#: `main 3e736ae`, captured in a clean detached worktree before any edit:
#: mHC + AttnRes + LayerRoPE-paper, both fused flags absent (= OFF).
PIN_A = "ee4433a46eefa5b9ac6e379e6faf7359464af63786c184810712c17e77bc151f"
#: The same, `attn_latent_rank=16` (the four projections read one latent).
PIN_B = "20a63ddeb57aa7a6bed66808a07ede59415a0e9f69075075cb5b6df8830789a8"

BASE = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=32,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=2,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
    use_kda=True,
    use_engram=False,
    use_msa=False,
    use_csa=False,
)


def _cfg(**over) -> ModelConfig:
    return ModelConfig(**{**BASE, **over})


def _digest(cfg: ModelConfig, ids: torch.Tensor) -> str:
    from dormouse.model import DormouseModel

    torch.manual_seed(0)
    m = DormouseModel(cfg, device="cpu").to("cpu").train()
    m.zero_grad(set_to_none=True)
    logits, loss, _ = m(ids, targets=ids, counters=Counters())
    loss.backward()
    h = hashlib.sha256()
    for t in (logits, loss.reshape(1)):
        h.update(t.detach().contiguous().numpy().tobytes())
    for name, p in sorted(m.named_parameters()):
        if p.grad is not None:
            h.update(name.encode())
            h.update(p.grad.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def _pinned(fn):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return fn()
    finally:
        torch.set_num_threads(threads)


# --------------------------------------------------------------------------- #
# 0. OFF is clean main — the pins were taken before the edit
# --------------------------------------------------------------------------- #
def test_off_is_bit_identical_to_clean_main() -> None:
    a = _cfg(
        use_mhc=True,
        mhc_streams=2,
        use_attnres=True,
        layerrope=True,
        layerrope_form="paper",
        layerrope_index="iteration",
        layerrope_init="paper",
        fused_residual=False,
        fused_epilogue=False,
    )
    ids = torch.arange(96, dtype=torch.long).reshape(3, 32) % 251
    got = _pinned(lambda: _digest(a, ids))
    assert got == PIN_A, (
        f"the epilogue arm's OFF path moved: the flag must be a route, not a formula. got {got}"
    )


def test_off_latent_is_bit_identical_to_clean_main() -> None:
    b = _cfg(attn_latent_rank=16, fused_latent_proj=False)
    ids = torch.arange(64, dtype=torch.long).reshape(2, 32) % 251
    got = _pinned(lambda: _digest(b, ids))
    assert got == PIN_B, (
        "the latent arm's OFF path moved: the fused projection must be a "
        f"reroute, not a recalculation. got {got}"
    )


# --------------------------------------------------------------------------- #
# 1. the read chain: parity + the gate BITES
# --------------------------------------------------------------------------- #
def _assert_read_parity(rep: dict, rtol: float = 1e-6) -> None:
    scale = rep["scale"][1]
    assert scale > 0, "the reference gradient is identically zero — nothing measured"
    for name, (d, m) in rep.items():
        if name == "scale":
            continue
        tol = rtol * max(m, 1e-30)
        if d > tol:
            raise AssertionError(
                f"epilogue read parity: {name} moved by {d:.3e} (reference {m:.3e}, "
                f"tolerance {tol:.3e}). The fused chain no longer reproduces eager."
            )


@pytest.mark.parametrize("separate", [False, True], ids=["one-gain", "paper-two-gains"])
def test_read_parity_is_fp32_roundoff(separate: bool, capsys) -> None:
    rep = fr.parity_read(b=3, t=8, d=64, separate=separate, seed=0)
    _assert_read_parity(rep)
    worst = max((d / max(m, 1e-30), n, d) for n, (d, m) in rep.items() if n != "scale")
    with capsys.disabled():
        print(
            f"\n  epilogue read parity (separate={separate}): worst {worst[1]} "
            f"rel={worst[0]:.2e} |Δ|={worst[2]:.2e}"
        )


def test_gate_reddens_when_the_rotation_gain_moves(monkeypatch) -> None:
    """A scratch 1e-3 nudge of the gain the fused arm APPLIES — the rotation's wiring."""
    _assert_read_parity(fr.parity_read(b=3, t=8, d=64, separate=True))
    real = fr.epilogue_reference

    def rotated(mhc, xl, gain, gain_mlp, eps):
        return real(mhc, xl, gain * 1.001, gain_mlp, eps)

    monkeypatch.setattr(fr, "epilogue_reference", rotated)
    rep = fr.parity_read(b=3, t=8, d=64, separate=True)
    print(f"\n  gain × 1.001: max|Δnormed| {rep['normed'][0]:.3e}")
    with pytest.raises(AssertionError):
        _assert_read_parity(rep)


def test_gate_reddens_when_the_normalisation_is_dropped(monkeypatch) -> None:
    """`ctx` un-normalised is the pre-norm value — the one thing the pass adds."""
    _assert_read_parity(fr.parity_read(b=3, t=8, d=64, separate=False))
    real = fr.epilogue_reference

    def unnormalised(mhc, xl, gain, gain_mlp, eps):
        ctx, _, _ = real(mhc, xl, gain, None, eps)
        return ctx, ctx * gain, None

    monkeypatch.setattr(fr, "epilogue_reference", unnormalised)
    rep = fr.parity_read(b=3, t=8, d=64, separate=False)
    print(f"\n  no RMSNorm: max|Δnormed| {rep['normed'][0]:.3e}")
    with pytest.raises(AssertionError):
        _assert_read_parity(rep)


# --------------------------------------------------------------------------- #
# 2. the latent projection: ONE GEMM, view-split, 1e-6 fp32 / 1e-2 bf16
# --------------------------------------------------------------------------- #
def _latent_pair(**over):
    from dormouse.model.kda import KDAAttention

    torch.manual_seed(0)
    off = KDAAttention(_cfg(attn_latent_rank=16, **over), device="cpu")
    torch.manual_seed(0)
    on = KDAAttention(_cfg(attn_latent_rank=16, fused_latent_proj=True, **over), device="cpu")
    on.load_state_dict(off.state_dict())
    return off, on


def _proj_diff(off, on, x):
    worst = (0.0, 0.0, "")
    for name, a, b in zip("qkvgb", off.projections(x), on.projections(x), strict=True):
        d = (a - b).abs().max().item()
        m = a.abs().max().item()
        if d > worst[0]:
            worst = (d, m, name)
    return worst


def test_one_latent_gemm_is_fp32_roundoff_and_splits_by_view(capsys) -> None:
    off, on = _latent_pair()
    x = torch.randn(4, 16, 64)
    d, m, name = _proj_diff(off, on, x)
    with capsys.disabled():
        print(
            f"\n  latent fp32: worst {name} |Δ|={d:.3e} scale={m:.3e} rel={d / max(m, 1e-30):.2e}"
        )
    assert d <= 1e-6 * max(m, 1e-30), f"{name} moved by {d:.3e} on scale {m:.3e}"

    # the split is a VIEW: the four slices share the single GEMM's storage.
    dq = (on.dense_qat, on.dense_qat_alpha)
    c = on._latent_base(x, dq)
    w = torch.cat([lin.weight for lin in (on.q_proj, on.k_proj, on.v_proj, on.g_proj)])
    out = torch.nn.functional.linear(c, w)
    d0 = on.n_heads * on.head_dim
    kv = on.kv_width
    slices = (
        out[..., :d0],
        out[..., d0 : d0 + kv],
        out[..., d0 + kv : d0 + 2 * kv],
        out[..., d0 + 2 * kv :],
    )
    for s in slices:
        assert s.untyped_storage().data_ptr() == out.untyped_storage().data_ptr(), (
            "the projection split copied: the point is one GEMM and four views"
        )


def test_the_fused_projection_issues_one_matmul(capsys) -> None:
    """Profiler truth, not a comment: `aten::mm` count on the two arms.

    With the shared latent, OFF is `latent_down + q + k + v + g + beta = 6`
    GEMMs and ON is `latent_down + (qkvg concatenated) + beta = 3`.
    """
    from torch.profiler import ProfilerActivity, profile

    off, on = _latent_pair()
    x = torch.randn(4, 16, 64)

    def count(m) -> int:
        m.projections(x)  # warm
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            m.projections(x)
        return sum(
            int(e.count) for e in prof.key_averages() if e.key in ("aten::mm", "aten::addmm")
        )

    n_off, n_on = count(off), count(on)
    with capsys.disabled():
        print(f"\n  projections GEMMs: four-linears {n_off}  concatenated {n_on}")
    assert n_off == 6 and n_on == 3, (
        f"the concatenated arm ran {n_on} GEMMs against the reference's {n_off}: "
        "q/k/v/g are no longer ONE matmul"
    )


def test_one_latent_gemm_is_bf16_1e2(capsys) -> None:
    off, on = _latent_pair()
    x = torch.randn(4, 16, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        d, m, name = _proj_diff(off, on, x)
    with capsys.disabled():
        print(
            f"\n  latent bf16: worst {name} |Δ|={d:.3e} scale={m:.3e} rel={d / max(m, 1e-30):.2e}"
        )
    assert d <= 1e-2 * max(m, 1e-30), f"bf16 {name} moved by {d:.3e} on scale {m:.3e}"


def test_the_fused_projection_keeps_the_qat_arm() -> None:
    """`dense_qat` projects each master FIRST; the concatenation is after it."""
    from dormouse.model.kda import KDAAttention

    over = dict(dense_qat="ternary", dense_qat_skip_attn=False)
    torch.manual_seed(0)
    off = KDAAttention(_cfg(attn_latent_rank=16, **over), device="cpu")
    torch.manual_seed(0)
    on = KDAAttention(_cfg(attn_latent_rank=16, fused_latent_proj=True, **over), device="cpu")
    on.load_state_dict(off.state_dict())
    x = torch.randn(4, 16, 64)
    d, m, name = _proj_diff(off, on, x)
    assert d <= 1e-6 * max(m, 1e-30), f"QAT {name} moved by {d:.3e} on scale {m:.3e}"


def _model_pair(**over):
    from dormouse.model import DormouseModel

    torch.manual_seed(0)
    off = DormouseModel(_cfg(attn_latent_rank=16, **over), device="cpu").train()
    torch.manual_seed(0)
    on = DormouseModel(
        _cfg(attn_latent_rank=16, fused_latent_proj=True, **over), device="cpu"
    ).train()
    on.load_state_dict(off.state_dict())
    return off, on


def _fwd_bwd(m, ids, bf16: bool):
    m.zero_grad(set_to_none=True)
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
    with ctx:
        logits, loss, _ = m(ids, targets=ids, counters=Counters())
    loss.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return logits.detach(), grads


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_model_logits_and_gradients_match_the_four_gemms(bf16: bool, capsys) -> None:
    """The rule is 1e-6 fp32 / 1e-2 bf16 — stated, because ONE GEMM is not the
    same bit pattern as four when the reduction order differs."""
    rtol = 1e-2 if bf16 else 1e-6
    off, on = _model_pair(bf16=bf16)
    ids = torch.arange(64, dtype=torch.long).reshape(2, 32) % 251
    torch.manual_seed(3)
    lo, go = _fwd_bwd(off, ids, bf16)
    torch.manual_seed(3)
    ln, gn = _fwd_bwd(on, ids, bf16)

    d = (lo.float() - ln.float()).abs().max().item()
    m = lo.float().abs().max().item()
    worst = max(
        (
            ((p - go[n]).float().abs().max().item(), p.float().abs().max().item(), n)
            for n, p in gn.items()
        ),
        key=lambda t: t[0] / max(t[1], 1e-30),
    )
    with capsys.disabled():
        print(
            f"\n  model {rtol:g} gate: logits |Δ|={d:.3e} scale={m:.3e} rel={d / m:.2e}; "
            f"worst grad {worst[2]} rel={worst[0] / max(worst[1], 1e-30):.2e}"
        )
    assert d <= rtol * max(m, 1e-30), f"logits moved by {d:.3e} on scale {m:.3e}"
    assert worst[0] <= rtol * max(worst[1], 1e-30), (
        f"{worst[2]} gradient moved by {worst[0]:.3e} on scale {worst[1]:.3e}"
    )


# --------------------------------------------------------------------------- #
# 3. refusals, the counted CPU fallback, and the OFF contract
# --------------------------------------------------------------------------- #
def test_the_flags_are_off_by_default() -> None:
    cfg = ModelConfig()
    assert cfg.fused_epilogue is False
    assert cfg.fused_latent_proj is False


def test_fused_epilogue_without_mhc_is_refused() -> None:
    with pytest.raises(LoudError) as err:
        validate(_cfg(fused_epilogue=True, use_mhc=False))
    msg = str(err.value)
    assert "fused_epilogue=true with use_mhc=false" in msg and "Escape:" in msg
    validate(_cfg(fused_epilogue=True, use_mhc=True))  # the legal pair is silent


def test_fused_latent_proj_without_kda_is_refused() -> None:
    for over in ({"use_kda": False}, {"attn_kind": "full"}):
        with pytest.raises(LoudError) as err:
            validate(_cfg(fused_latent_proj=True, **over))
        msg = str(err.value)
        assert "fused_latent_proj=true without the KDA arm" in msg and "Escape:" in msg
    validate(_cfg(fused_latent_proj=True, use_kda=True))


def test_unsupported_reason_names_cause_and_escape() -> None:
    assert fr.unsupported_reason(torch.float32, 2, 2 * 1536) is None or not fr.fused_available()
    for dtype, streams, kd in (
        (torch.bfloat16, 2, 2048),  # the chain is fp32-only
        (torch.float32, 4, 4096),  # not a 2×2 Sinkhorn
        (torch.float32, 2, 8192),  # wider than the register tile
        (torch.float32, 2, 2 * 769),  # odd d_model: no pair rotation
    ):
        why = fr.unsupported_reason(dtype, streams, kd)
        assert why is not None and "Escape:" in why, f"{dtype}/{streams}/{kd} was accepted"


def test_a_cpu_host_falls_back_counted_and_bit_identically() -> None:
    """The fallback runs the SAME eager lines, so its values cannot move."""
    from dormouse.model import DormouseModel

    def build(epilogue: bool) -> DormouseModel:
        torch.manual_seed(0)
        return DormouseModel(
            _cfg(
                use_mhc=True,
                mhc_streams=2,
                use_attnres=True,
                layerrope=True,
                layerrope_form="paper",
                layerrope_init="paper",
                fused_epilogue=epilogue,
                fused_residual=True,
            ),
            device="cpu",
        ).train()

    off, on = build(False), build(True)
    on.load_state_dict(off.state_dict())
    ids = torch.randint(0, 256, (2, 16))
    c = Counters()
    with torch.no_grad():
        a = off(ids, targets=ids, counters=Counters())[0]
        b = on(ids, targets=ids, counters=c)[0]
    assert torch.equal(a, b), "the CPU fallback moved a value: it is not the eager path"
    assert c["fused_epilogue_fallback"] == 1 and c.get("fused_epilogue", 0) == 0


# --------------------------------------------------------------------------- #
# 4. no host sync in the fused paths (ADR-0018 rule 2)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def no_host_reads():
    saved = {}
    names = ("item", "tolist", "__float__", "__bool__")

    def boom(_self, *_a, **_k):
        raise AssertionError("host-device sync in a fused path (ADR-0018 rule 2)")

    for name in names:
        saved[name] = getattr(torch.Tensor, name)
        setattr(torch.Tensor, name, boom)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.Tensor, name, fn)


def test_the_fused_reads_never_touch_the_host() -> None:
    from dormouse.model.mhc import MHCMix

    mhc = MHCMix(64, 2, device="cpu")
    xl = (torch.randn(2, 8, 2, 64) * 0.5).requires_grad_(True)
    gain = torch.randn(64, requires_grad=True)
    with no_host_reads():
        _, n1, n2 = fr.epilogue_reference(mhc, xl, gain, gain, 1e-3)
        (n1.sum() + n2.sum()).backward()
    assert xl.grad is not None and gain.grad is not None


def test_the_fused_latent_projection_never_touches_the_host() -> None:
    _, on = _latent_pair()
    x = (torch.randn(2, 8, 64)).requires_grad_(True)
    with no_host_reads():
        out = on.projections(x)
        sum(t.sum() for t in out).backward()
    assert x.grad is not None


# --------------------------------------------------------------------------- #
# 5. the card: the real kernel, the launch counter, the mutation
# --------------------------------------------------------------------------- #
@CUDA
@pytest.mark.parametrize("separate", [False, True], ids=["one-gain", "paper-two-gains"])
def test_the_kernel_is_fp32_roundoff(separate: bool) -> None:
    _assert_read_parity(fr.parity_read(b=2, t=8, d=64, separate=separate, use_kernel=True))
    fr.assert_parity(fr.parity(b=2, t=8, d=64, compose=True, epilogue=True, separate=separate))


@CUDA
def test_1536_is_legal_and_the_kernel_matches_there() -> None:
    """The m400 width: `d=1536` is NOT a power of two and must not refuse."""
    assert fr.unsupported_reason(torch.float32, 2, 2 * 1536) is None
    fr.assert_parity(fr.parity(b=1, t=4, d=1536, compose=True, epilogue=True, separate=True))


@CUDA
def test_the_fused_kernel_issues_far_fewer_launches(capsys) -> None:
    from torch.profiler import ProfilerActivity, profile

    from dormouse.model.loopblock import AttnResSlot, depth_attend
    from dormouse.model.mhc import MHCMix

    torch.manual_seed(0)
    mhc = MHCMix(64, 2, device="cuda")
    slot = AttnResSlot(64, "cuda")
    xl = torch.randn(4, 16, 2, 64, device="cuda") * 0.5
    y = torch.randn(4, 16, 64, device="cuda") * 0.5
    gain = torch.randn(64, device="cuda")

    def eager():
        ctx = mhc.pre_mix(xl)
        normed = torch.nn.functional.rms_norm(ctx, (64,), gain, 1e-3)
        upd = depth_attend([torch.zeros_like(y), y], slot, 1e-3, 2.0)
        (ctx.sum() + normed.sum() + mhc(xl, upd).sum()).backward()

    def fused():
        c = Counters()
        ctx, raw, normed, _ = fr.epilogue_read(mhc, xl, gain, None, 1e-3, c)
        sn, h = fr.write(mhc, slot, xl, y, raw, True, 1e-3, c)
        (ctx.sum() + normed.sum() + (sn.sum() + h.sum())).backward()
        return c

    def launches(fn):
        fn()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
        return sum(1 for e in prof.events() if e.name == "cudaLaunchKernel")

    n_eager, n_fused = launches(eager), launches(fused)
    c = fused()
    drained = c.drain()
    with capsys.disabled():
        print(
            f"\n  read chain launches: eager {n_eager} fused {n_fused}; "
            f"counter {drained.get('fused_epilogue_launches')}"
        )
    assert n_fused * 5 <= n_eager, (
        f"the fused read issued {n_fused} launches against eager's {n_eager}: "
        "the norm chain stopped collapsing"
    )
    # each side's counter is its own `cnt` tensor: one read kernel and one write
    # kernel per iteration (the drain is the only host read, at the log cadence).
    assert drained["fused_epilogue_launches"] == 1, drained
    assert drained["fused_residual_launches"] == 1, drained


@CUDA
def test_the_fused_boundary_never_reads_the_device() -> None:
    from dormouse.model.loopblock import AttnResSlot
    from dormouse.model.mhc import MHCMix

    mhc = MHCMix(64, 2, device="cuda")
    slot = AttnResSlot(64, "cuda")
    xl = (torch.randn(4, 16, 2, 64, device="cuda") * 0.5).requires_grad_(True)
    y = (torch.randn(4, 16, 64, device="cuda") * 0.5).requires_grad_(True)
    gain = torch.randn(64, device="cuda", requires_grad=True)
    with no_host_reads():
        c = Counters()
        ctx, raw, normed, _ = fr.epilogue_read(mhc, xl, gain, None, 1e-3, c)
        sn, h = fr.write(mhc, slot, xl, y, raw, True, 1e-3, c)
        ((ctx.sum() + normed.sum() + (sn.sum() + h.sum())) * 1e-3).backward()
    assert ctx.shape == (4, 16, 64)


@CUDA
def test_the_kernel_gate_reddens_when_the_gain_moves(monkeypatch) -> None:
    fr.assert_parity(fr.parity(b=2, t=8, d=64, compose=True, epilogue=True))
    real = fr.epilogue_read

    def rotated(mhc, xl, gain, gain_mlp, eps, counters):
        return real(mhc, xl, gain * 1.001, gain_mlp, eps, counters)

    monkeypatch.setattr(fr, "epilogue_read", rotated)
    rep = fr.parity(b=2, t=8, d=64, compose=True, epilogue=True)
    print(f"\n  kernel gain × 1.001: max|Δnormed| {rep['normed'][0]:.3e}")
    with pytest.raises(AssertionError):
        fr.assert_parity(rep)
