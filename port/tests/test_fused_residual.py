"""The fused residual chain's gates: parity, launch count, 0-sync, refusals.

Four things are asserted here, and each one is the *reason* the arm exists:

1. **Parity** — `fused_residual.parity()` runs the shipped eager modules and the
   fused kernels on the same tensors and compares every output and every
   gradient. The tolerance is the rule in `assert_parity`: 1e-5 relative on the
   outputs, 5e-5 on a gradient, and "both are tiny" for the tensors the
   20-iteration Sinkhorn makes degenerate (its Jacobian is ~1e-6 against
   gradients of ~1e2, so fp32 noise is the whole signal there; the VJP itself is
   verified exact in float64 — 4.9e-17 — in `parity`'s docstring).
2. **The gate BITES** — four perturbations of the chain are applied and each one
   must push the comparison past the tolerance. A parity gate that cannot fail
   is not a gate (AGENTS §3.3, the AttnRes history), so this is the test that
   makes (1) mean something. It has already earned its keep once: an
   `H_pre` gradient that ignored the `dctx` path was invisible to the forward
   and showed up here at 8.1e-4.
3. **Launch count** — the same chain profiled on both paths; the fused count
   must be at least 5× lower. Both numbers are printed, always.
4. **No host sync** — `.item/.tolist/.__float__/.__bool__` are replaced by
   raisers while the fused forward AND backward run, so a D2H read anywhere in
   the path fails the test (ADR-0018 rule 2).

Plus the two refusals: a CPU host is a COUNTED fallback whose values are
bit-identical to the eager path, and `fused_residual` without `use_mhc` is LOUD.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from dormouse.config import ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import fused_residual as fr

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the card")
TOL = 1e-5


def _cuda() -> str:
    return "cuda"


# --------------------------------------------------------------------------- #
# 1. parity
# --------------------------------------------------------------------------- #
@CUDA
@pytest.mark.parametrize("compose", [True, False], ids=["compose", "mhc-only"])
def test_parity_is_fp32_roundoff(compose: bool) -> None:
    rep = fr.parity(b=3, t=8, d=64, compose=compose)
    fr.assert_parity(rep)  # raises LOUD naming the tensor that moved
    # the forward bound is what a reader would quote, so pin it, not just "under tol"
    for name in ("ctx", "streams", "h"):
        d, m = rep[name]
        assert d <= TOL * m, f"{name}: |Δ|={d:.3e} on scale {m:.3e}"


# --------------------------------------------------------------------------- #
# 2. the gate reddens
# --------------------------------------------------------------------------- #
def _clean_passes() -> None:
    """The unperturbed chain must pass first, or a red mutation means nothing."""
    fr.assert_parity(fr.parity(b=3, t=8, d=64, compose=True))


@CUDA
def test_gate_reddens_when_the_sinkhorn_normalisation_is_dropped(monkeypatch) -> None:
    """`iters=0` is `exp(logits)` — no Birkhoff projection at all."""
    _clean_passes()
    real = fr.write
    monkeypatch.setattr(fr, "write", lambda *a, **k: real(*a, **{**k, "iters": 0}))
    rep = fr.parity(b=3, t=8, d=64, compose=True)
    print(f"\n  drops sinkhorn: max|Δstreams| {rep['streams'][0]:.3e}")
    with pytest.raises(AssertionError):
        fr.assert_parity(rep)


@CUDA
def test_gate_reddens_when_the_hyper_network_is_stored_in_bf16(monkeypatch) -> None:
    """The task's second named mutation: bf16 stores where eager stores fp32."""
    _clean_passes()
    real = fr.write

    def bf16_raw(*a, **k):
        args = list(a)
        args[4] = args[4].bfloat16().float()  # `raw`: H̃_pre/H̃_post/H̃_res
        return real(*args, **k)

    monkeypatch.setattr(fr, "write", bf16_raw)
    rep = fr.parity(b=3, t=8, d=64, compose=True)
    print(f"\n  bf16 hyper-network store: max|Δh| {rep['h'][0]:.3e}")
    with pytest.raises(AssertionError):
        fr.assert_parity(rep)


@CUDA
def test_gate_reddens_when_the_attnres_query_is_zeroed(monkeypatch) -> None:
    """`q_eff = 0` makes the write `y` instead of `2σ(s)·y` — the arm's identity lever."""
    _clean_passes()
    real = fr.write

    def no_query(mhc, slot, xl, y, *a, **k):
        class _Zero:
            gain = slot.gain * 0.0
            query = slot.query * 0.0

        return real(mhc, _Zero(), xl, y, *a, **k)

    monkeypatch.setattr(fr, "write", no_query)
    rep = fr.parity(b=3, t=8, d=64, compose=True)
    print(f"\n  zeroed query: max|Δh| {rep['h'][0]:.3e}")
    with pytest.raises(AssertionError):
        fr.assert_parity(rep)


@CUDA
def test_gate_reddens_when_the_pre_bias_moves(monkeypatch) -> None:
    """A 1e-4 nudge of `b_pre` — the pre kernel's own wiring, not a re-derivation."""
    _clean_passes()
    real = fr.epilogue_read

    def nudged(mhc, xl, gain, gain_mlp, eps_norm, counters):
        before = mhc.b_pre.data.clone()
        with torch.no_grad():
            mhc.b_pre.add_(1e-4)
        try:
            return real(mhc, xl, gain, gain_mlp, eps_norm, counters)
        finally:
            with torch.no_grad():
                mhc.b_pre.copy_(before)

    monkeypatch.setattr(fr, "epilogue_read", nudged)
    rep = fr.parity(b=3, t=8, d=64, compose=True)
    print(f"\n  b_pre + 1e-4: max|Δctx| {rep['ctx'][0]:.3e}")
    with pytest.raises(AssertionError):
        fr.assert_parity(rep)


# --------------------------------------------------------------------------- #
# 3. launch count (the metric this arm is judged by)
# --------------------------------------------------------------------------- #
def _launches(fn) -> int:
    from torch.profiler import ProfilerActivity, profile

    fn()  # warm: the kernels must be compiled before the counted run
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    # THREE names: torch's ops launch through the runtime API
    # (`cudaLaunchKernel`), a Triton kernel through the driver API
    # (`cuLaunchKernel` / `cuLaunchKernelEx` measured on this box, 2026-11-08).
    # Counting only the first made the fused arm read as ZERO launches, so the
    # "drop" it asserted was the arithmetic 11-0.
    launch_names = ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx")
    return sum(1 for e in prof.events() if e.name in launch_names)


@CUDA
def test_the_fused_path_issues_far_fewer_launches(capsys) -> None:
    from dormouse.model.loopblock import AttnResSlot, depth_attend
    from dormouse.model.mhc import MHCMix

    torch.manual_seed(0)
    mhc = MHCMix(64, 2, device=_cuda())
    slot = AttnResSlot(64, _cuda())
    xl = (torch.randn(4, 16, 2, 64, device=_cuda()) * 0.5).requires_grad_(True)
    y = (torch.randn(4, 16, 64, device=_cuda()) * 0.5).requires_grad_(True)
    gh = torch.randn(4, 16, 64, device=_cuda())

    def eager():
        ctx = mhc.pre_mix(xl)
        upd = depth_attend([torch.zeros_like(y), y], slot, 1e-3, 2.0)
        h = mhc(xl, upd).sum(dim=-2) / 2
        ((ctx.sum() + (h * gh).sum()) * 1e-3).backward()

    def fused():
        c = Counters()
        ctx, raw = fr.pre(mhc, xl, c)
        sn, h = fr.write(mhc, slot, xl, y, raw, True, 1e-3, c)
        ((ctx.sum() + (h * gh).sum()) * 1e-3).backward()

    n_eager = _launches(eager)
    n_fused = _launches(fused)
    if n_fused == 0 and _launches(eager) == 0:
        # both arms empty: CUPTI captured nothing (busy device), not a result
        pytest.skip("CUPTI captured no kernel launches in this session (busy device?)")
    assert n_fused > 0, "the profiler saw no kernel launch from the fused arm — nothing measured"
    print(
        f"\n  launches per residual chain: eager {n_eager}  fused {n_fused}  "
        f"({n_eager / max(n_fused, 1):.1f}x)"
    )
    assert n_fused * 5 <= n_eager, (
        f"fused_residual issued {n_fused} launches against eager's {n_eager}: the "
        "kernel collapse stopped happening"
    )


@CUDA
def test_the_device_counter_matches_the_profiler() -> None:
    """`fused_residual_launches` is what the log line shows; it must be true."""
    from dormouse.model.loopblock import AttnResSlot
    from dormouse.model.mhc import MHCMix

    torch.manual_seed(0)
    mhc = MHCMix(64, 2, device=_cuda())
    slot = AttnResSlot(64, _cuda())
    xl = torch.randn(4, 16, 2, 64, device=_cuda()) * 0.5
    y = torch.randn(4, 16, 64, device=_cuda()) * 0.5
    c = Counters()
    for _ in range(2):
        _, raw = fr.pre(mhc, xl, c)
        fr.write(mhc, slot, xl, y, raw, True, 1e-3, c)
    drained = c.drain()
    # 2 forward kernels per call × 2 iterations
    assert drained["fused_residual_launches"] == 4, drained


# --------------------------------------------------------------------------- #
# 4. no host sync
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def no_host_reads():
    """Every D2H read on a tensor raises — ADR-0018 rule 2, as a context."""
    saved = {}
    names = ("item", "tolist", "__float__", "__bool__")

    def boom(_self, *_a, **_k):
        raise AssertionError("host-device sync in the fused path (ADR-0018 rule 2)")

    for name in names:
        saved[name] = getattr(torch.Tensor, name)
        setattr(torch.Tensor, name, boom)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.Tensor, name, fn)


@CUDA
def test_the_fused_path_never_reads_the_device() -> None:
    from dormouse.model.loopblock import AttnResSlot
    from dormouse.model.mhc import MHCMix

    torch.manual_seed(0)
    mhc = MHCMix(64, 2, device=_cuda())
    slot = AttnResSlot(64, _cuda())
    xl = (torch.randn(4, 16, 2, 64, device=_cuda()) * 0.5).requires_grad_(True)
    y = (torch.randn(4, 16, 64, device=_cuda()) * 0.5).requires_grad_(True)
    with no_host_reads():
        c = Counters()
        ctx, raw = fr.pre(mhc, xl, c)
        sn, h = fr.write(mhc, slot, xl, y, raw, True, 1e-3, c)
        (h.sum() * 1e-3).backward()
    assert ctx.shape == (4, 16, 64) and sn.shape == (4, 16, 2, 64)


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
@pytest.mark.pins
def test_the_off_path_is_bit_identical_to_clean_main(on: bool) -> None:
    """Pinned from a DETACHED clean-main worktree at `3d36844`.

    `1ada26f9…` is the sha256 of the logits, the loss and every parameter
    gradient of the tiny mHC+AttnRes model below, taken on
    `/home/sehaxe/dormouse-wt/cleanmain` (detached at `3d36844`, the same commit
    this branch was cut from) with `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)`,
    and reproduced twice. `on=False` must keep it: the flag adds a route and
    changes no formula. `on=True` on a CPU host falls back to the same eager
    lines, so the *values* must match too — only the counter may differ.
    """

    # The pin is thread-count sensitive (reductions reorder), and this file must
    # not leave the process at one thread: two `test_muon_variants` hash pins
    # measure with the default count and went red when it leaked once.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _off_path_hash(on)
    finally:
        torch.set_num_threads(threads)


def _off_path_hash(on: bool) -> None:
    import hashlib

    from dormouse.model import DormouseModel

    cfg = ModelConfig(
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
        use_mhc=True,
        mhc_streams=2,
        use_attnres=True,
        use_engram=False,
        use_msa=False,
        use_csa=False,
        fused_residual=on,
    )
    torch.manual_seed(0)
    m = DormouseModel(cfg, device="cpu").to("cpu").train()
    ids = torch.arange(96, dtype=torch.long).reshape(3, 32) % 251
    logits, loss, _ = m(ids, targets=ids, counters=Counters())
    loss.backward()
    h = hashlib.sha256()
    for t in (logits, loss.reshape(1)):
        h.update(t.detach().contiguous().numpy().tobytes())
    for name, p_ in sorted(m.named_parameters()):
        if p_.grad is not None:
            h.update(name.encode())
            h.update(p_.grad.detach().contiguous().numpy().tobytes())
    assert h.hexdigest() == "1ada26f9c6095a6fcd25748010b479084a9f242beb02fb23f6d06c627d8b4d63", (
        "the OFF path moved: this branch no longer reproduces clean main's "
        f"logits+grads (got {h.hexdigest()}). The flag must be a route, not a formula."
    )


# --------------------------------------------------------------------------- #
# 5. the flag: off by default, counted fallback on a host, LOUD when a lie
# --------------------------------------------------------------------------- #
def test_the_flag_is_off_by_default() -> None:
    assert ModelConfig().fused_residual is False


def test_fused_residual_without_mhc_is_refused() -> None:
    cfg = ModelConfig(fused_residual=True, use_mhc=False)
    with pytest.raises(LoudError) as err:
        validate(cfg)
    assert "fused_residual=true with use_mhc=false" in str(err.value)
    assert "Escape:" in str(err.value)
    validate(ModelConfig(fused_residual=True, use_mhc=True))  # the legal pair is silent
    # the AttnRes-ONLY write is not a local chain either (a growing source list)
    with pytest.raises(LoudError) as err:
        validate(ModelConfig(fused_residual=True, use_attnres=True, use_mhc=False))
    assert "GROWING source list" in str(err.value) and "Escape:" in str(err.value)


def test_a_cpu_host_falls_back_counted_and_bit_identically() -> None:
    """The fallback runs the SAME eager code, so its values cannot move."""
    from dormouse.model import DormouseModel

    def build(fused: bool) -> DormouseModel:
        cfg = ModelConfig(
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
            use_mhc=True,
            mhc_streams=2,
            use_attnres=True,
            use_engram=False,
            use_msa=False,
            use_csa=False,
            fused_residual=fused,
        )
        torch.manual_seed(0)
        m = DormouseModel(cfg, device="cpu").to("cpu").train()
        return m

    off, on = build(False), build(True)
    on.load_state_dict(off.state_dict())
    ids = torch.randint(0, 256, (2, 16))
    c = Counters()
    with torch.no_grad():
        a = off(ids, targets=ids, counters=Counters())[0]
        b = on(ids, targets=ids, counters=c)[0]
    assert torch.equal(a, b), "the CPU fallback moved a value: it is not the eager path"
    assert c["fused_residual_fallback"] == 1 and c.get("fused_residual", 0) == 0


def test_unsupported_reason_names_cause_and_escape() -> None:
    """Each refusal carries its own Escape, so a reader knows the way out."""
    assert fr.unsupported_reason(torch.float32, 2, 2048) is None or not fr.fused_available()
    for dtype, streams, kd in (
        (torch.bfloat16, 2, 2048),  # not fp32
        (torch.float32, 4, 4096),  # not a 2x2 Sinkhorn
        (torch.float32, 2, 8192),  # wider than the register tile
        (torch.float32, 2, 2 * 769),  # odd d_model: no adjacent-pair rotation
    ):
        why = fr.unsupported_reason(dtype, streams, kd)
        if not fr.fused_available():
            assert why and "Escape:" in why
            continue
        assert why is not None and "Escape:" in why, f"{dtype}/{streams}/{kd} was accepted"
