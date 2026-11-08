"""`fused_kda_tail`'s gates: clean-main pins, parity, the biting mutation, refusals.

The flag moves two elementwise chains into Triton (`model/fused_kda.py`):

1. **`tail`** — `q/k/v` cast + `logsigmoid(g)` + `sigmoid(beta)`, six eager
   launches -> one. Its arithmetic is checked by `fused_kda.parity` on the CPU
   mirror (`tail_reference`), outputs AND every gradient, at 1e-5 relative.
2. **`combine`** — `y = attn·attn_write + branch·attn_write + ffn·mlp_write`,
   five eager passes -> one. Checked on the CPU mirror (`combine_reference`) and
   through the WHOLE MODEL: with the mirror monkeypatched in, a model running
   `fused_kda_tail=True` must match the OFF model's logits and every parameter
   gradient at 1e-5.

The pins below were captured on a detached clean `main` (`689d4ac`) BEFORE this
branch edited anything, at `OMP_NUM_THREADS=1` + `torch.set_num_threads(1)`:

* `PIN_A` — mHC(2) + AttnRes + LayerROPE-paper, fp32. It also equals
  `test_fused_epilogue.PIN_A`, so two independent captures agree on the tree.
* `PIN_B` — the same, `bf16=True` (the arm the measured run uses).
* `PIN_C` — `attn_latent_rank=16`, fp32 (the projection split the tail hangs off);
  it equals `test_fused_epilogue.PIN_B`, a second independent capture.

The CUDA gates (the real kernels, the profiler's launch count) are written and
SKIPPED on a CPU host: this branch was cut while a training job owned the card.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from dormouse.config import ModelConfig
from dormouse.instrument import Counters
from dormouse.model import fused_kda as fk

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the card")

#: clean `main` 689d4ac, captured before the flag existed.
PIN_A = "ee4433a46eefa5b9ac6e379e6faf7359464af63786c184810712c17e77bc151f"
PIN_B = "e63ee314a613b12ed3c333a9ea87c56becce9452181fc1c18a60d3ed85d08098"
PIN_C = "20a63ddeb57aa7a6bed66808a07ede59415a0e9f69075075cb5b6df8830789a8"

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


def _digest_of(m, ids: torch.Tensor, counters: Counters | None = None) -> str:
    m.zero_grad(set_to_none=True)
    logits, loss, _ = m(ids, targets=ids, counters=Counters() if counters is None else counters)
    loss.backward()
    h = hashlib.sha256()
    for t in (logits, loss.reshape(1)):
        h.update(t.detach().contiguous().numpy().tobytes())
    for name, p in sorted(m.named_parameters()):
        if p.grad is not None:
            h.update(name.encode())
            h.update(p.grad.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def _digest(cfg: ModelConfig, ids: torch.Tensor) -> str:
    from dormouse.model import DormouseModel

    torch.manual_seed(0)
    m = DormouseModel(cfg, device="cpu").to("cpu").train()
    return _digest_of(m, ids)


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
def _full_stack(**over) -> ModelConfig:
    base = dict(
        use_mhc=True,
        mhc_streams=2,
        use_attnres=True,
        layerrope=True,
        layerrope_form="paper",
        layerrope_index="iteration",
        layerrope_init="paper",
        fused_kda_tail=False,
    )
    base.update(over)
    return _cfg(**base)


def test_off_is_bit_identical_to_clean_main() -> None:
    ids = torch.arange(96, dtype=torch.long).reshape(3, 32) % 251
    got = _pinned(lambda: _digest(_full_stack(), ids))
    assert got == PIN_A, f"the flag's OFF path moved: it must be a route, not a formula. got {got}"


def test_off_bf16_is_bit_identical_to_clean_main() -> None:
    ids = torch.arange(96, dtype=torch.long).reshape(3, 32) % 251
    got = _pinned(lambda: _digest(_full_stack(bf16=True), ids))
    assert got == PIN_B, f"the bf16 OFF path moved: got {got}"


def test_off_latent_is_bit_identical_to_clean_main() -> None:
    ids = torch.arange(64, dtype=torch.long).reshape(2, 32) % 251
    got = _pinned(lambda: _digest(_cfg(attn_latent_rank=16), ids))
    assert got == PIN_C, f"the latent OFF path moved: got {got}"


# --------------------------------------------------------------------------- #
# 1. parity of the two plans, outputs AND every gradient
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gains", [True, False], ids=["paper-gains", "plain-sum"])
def test_parity_is_fp32_roundoff(gains: bool, capsys) -> None:
    rep = fk.parity(b=3, t=8, d=64, h=4, seed=0, gains=gains)
    fk.assert_parity(rep)
    worst = max((d / max(m, 1e-30), n, d) for n, (d, m) in rep.items() if n != "scale")
    with capsys.disabled():
        print(
            f"\n  fused_kda parity (gains={gains}): worst {worst[1]} "
            f"rel={worst[0]:.2e} |Δ|={worst[2]:.2e}"
        )


def test_gate_reddens_when_a_write_gate_is_dropped(monkeypatch) -> None:
    """`ffn·mlp_write` loses its gain — the mutation the flag must never make."""
    fk.assert_parity(fk.parity(b=3, t=8, d=64, h=4, gains=True))
    real = fk.combine_reference

    def dropped_gate(attn, branch, ffn, gain_attn, gain_mlp):
        del gain_mlp
        return (attn + branch) * gain_attn + ffn

    assert real is not dropped_gate
    monkeypatch.setattr(fk, "combine_reference", dropped_gate)
    rep = fk.parity(b=3, t=8, d=64, h=4, gains=True)
    print(f"\n  dropped mlp_write: max|Δy| {rep['y'][0]:.3e}")
    with pytest.raises(AssertionError, match="d_ffn|d_gain_mlp|y"):
        fk.assert_parity(rep)


def test_gate_reddens_when_logsigmoid_is_dropped(monkeypatch) -> None:
    """`g` un-logsigmoided is the raw projection — the one function the tail adds."""
    fk.assert_parity(fk.parity(b=3, t=8, d=64, h=4, gains=False))
    monkeypatch.setattr(fk, "_logsigmoid_ref", lambda x: x * 1.0)
    rep = fk.parity(b=3, t=8, d=64, h=4, gains=False)
    print(f"\n  no logsigmoid: max|Δg| {rep['g'][0]:.3e}")
    with pytest.raises(AssertionError, match="g"):
        fk.assert_parity(rep)


# --------------------------------------------------------------------------- #
# 2. the WHOLE MODEL: ON (mirror) vs OFF, logits and every parameter gradient
# --------------------------------------------------------------------------- #
def test_on_mirror_matches_off_logits_and_every_gradient(monkeypatch) -> None:
    """The flag's route, wired, checked against OFF at 1e-5 on the whole model.

    On a CPU host the kernels cannot run, so the mirror ([`tail_reference`] /
    [`combine_reference`]) stands in for them: the point is that the CALL the
    loop makes carries the same three branches, the same two gains and the same
    order the kernel does. The differences left are the two formulas'
    last-ulp round-off (`logsigmoid` via `log(1+exp)`, and `(a+b)·g` instead of
    `a·g + b·g`).
    """
    from dormouse.model import DormouseModel

    ids = torch.arange(96, dtype=torch.long).reshape(3, 32) % 251
    # Engram ON: without it `branch` is the shared zero tensor and
    # `(attn+0)·g` is bit-identical to `attn·g + 0·g`, i.e. the comparison
    # would pass before measuring anything. A nonzero branch is the point.
    off_cfg = _full_stack(use_engram=True)
    torch.manual_seed(0)
    off = DormouseModel(off_cfg, device="cpu").to("cpu").train()
    on_cfg = _full_stack(fused_kda_tail=True, use_engram=True)
    torch.manual_seed(1)  # a different draw on purpose: the state dict is copied
    on = DormouseModel(on_cfg, device="cpu").to("cpu").train()
    on.load_state_dict(off.state_dict())

    monkeypatch.setattr(fk, "unsupported_reason", lambda dtype, device: None)
    monkeypatch.setattr(fk, "combine_supported", lambda *tensors: True)
    monkeypatch.setattr(
        fk,
        "tail",
        lambda q, k, v, g, beta, dt, counters=None: fk.tail_reference(q, k, v, g, beta, dt),
    )
    monkeypatch.setattr(
        fk,
        "combine",
        lambda a, b, c, g1, g3, counters=None: fk.combine_reference(a, b, c, g1, g3),
    )

    def run(m):
        m.zero_grad(set_to_none=True)
        logits, loss, _ = m(ids, targets=ids, counters=Counters())
        loss.backward()
        grads = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
        return logits.detach(), grads

    lo, go = run(off)
    ln, gn = run(on)
    assert set(go) == set(gn), "the ON model lost a parameter's gradient path"
    # The 20-iteration Sinkhorn's Jacobian is ~1e-6 of the graph's largest
    # gradient (fused_residual.DEGENERATE's note): those tensors are noise in
    # BOTH arms, so the rule is "both are tiny" there, not a relative match.
    scale = max(m.abs().max().item() for m in go.values())
    worst = (0.0, "", 0.0)
    for name in sorted(go):
        d = (go[name] - gn[name]).abs().max().item()
        m = go[name].abs().max().item()
        tol = 1e-5 * max(m, 1e-30)
        if m < 1e-3 * scale:
            tol = 1e-3 * scale
        if d > worst[0]:
            worst = (d, name, m)
        assert d <= tol, (
            f"parameter {name}: |Δgrad|={d:.3e} on scale {m:.3e} (tol {tol:.3e}) "
            "— the fused write route lost a branch"
        )
    dl = (lo - ln).abs().max().item()
    ml = lo.abs().max().item()
    print(f"\n  ON vs OFF: logits |Δ|={dl:.3e} scale={ml:.3e}; worst grad {worst[1]}")
    assert dl <= 1e-5 * max(ml, 1e-30)
    assert worst[2] > 0


# --------------------------------------------------------------------------- #
# 3. the CPU fallback: COUNTED and bit-identical to OFF
# --------------------------------------------------------------------------- #
def test_cpu_fallback_is_counted_and_bit_identical() -> None:
    from dormouse.model import DormouseModel

    ids = torch.arange(96, dtype=torch.long).reshape(3, 32) % 251
    off = _full_stack()
    on = _full_stack(fused_kda_tail=True)
    torch.manual_seed(0)
    m_off = DormouseModel(off, device="cpu").to("cpu").train()
    got_off = _pinned(lambda: _digest_of(m_off, ids))
    torch.manual_seed(0)
    m_on = DormouseModel(on, device="cpu").to("cpu").train()
    counters = Counters()
    got_on = _pinned(lambda: _digest_of(m_on, ids, counters))
    assert got_on == got_off, "a fallback must be the eager call, bit for bit"
    assert counters.get("fused_kda_tail_fallback", 0) == on.max_iter, (
        "the ON flag on a CPU host must COUNT one fallback per iteration, not "
        f"run silently: {dict(counters)}"
    )
    assert counters.get("fused_kda_tail_launches", 0) == 0


@CUDA
def test_no_host_sync() -> None:
    """No `.item/.tolist/__float__/__bool__` in either fused forward or backward.

    Scoped to the two chains, not to the whole model: the MoE's naive dispatch
    calls `.tolist()` on its router's ids (a counted CPU-path cost, unrelated
    to this flag), so a whole-model catcher would redden on someone else's
    line. The kernel path itself must never cross the bus.
    """
    dev = "cuda"
    b, t, d, h = 2, 8, 64, 4
    q = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    k = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    v = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    g = (torch.randn(b, t, d, device=dev) * 2.0).requires_grad_(True)
    beta = (torch.randn(b, t, h, device=dev) * 2.0).requires_grad_(True)
    a = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    bb = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    c = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    g1 = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    g3 = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    saved = {
        name: getattr(torch.Tensor, name) for name in ("item", "tolist", "__float__", "__bool__")
    }

    def boom(*_a, **_k):
        raise AssertionError("host-device sync in the fused path (ADR-0018 rule 2)")

    try:
        for name in saved:
            setattr(torch.Tensor, name, boom)
        outs = fk.tail(q, k, v, g, beta, torch.bfloat16)
        y = fk.combine(a, bb, c, g1, g3)
        (outs[0].float().sum() + outs[3].sum() + y.sum()).backward()
    finally:
        for name, fn in saved.items():
            setattr(torch.Tensor, name, fn)


# --------------------------------------------------------------------------- #
# 4. refusals carry `Escape:`
# --------------------------------------------------------------------------- #
def test_refusals_carry_escape() -> None:
    cpu = fk.unsupported_reason(torch.float32, "cpu")
    assert cpu is not None and "Escape:" in cpu, cpu
    any_host = fk.unsupported_reason(torch.bfloat16, "cpu")
    assert any_host is not None and "Escape:" in any_host, any_host
    if not torch.cuda.is_available():
        # The host reason wins on a CPU box; the dtype refusal is still a
        # sentence with an escape, checked above through the host branch.
        assert "Triton" in (cpu or "") or "CUDA" in (cpu or ""), cpu


def test_tail_refuses_a_bf16_input() -> None:
    """The kernel is fp32-input only; a bf16 projection says so instead of running."""
    q = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    why = fk.unsupported_reason(q.dtype, "cpu")
    assert why is not None and "Escape:" in why
    with pytest.raises(AssertionError, match="Escape:"):
        fk.tail(q, q, q, q, torch.zeros(1, 2, 2), torch.bfloat16)


# --------------------------------------------------------------------------- #
# 5. the census: the static arithmetic this flag is justified by
# --------------------------------------------------------------------------- #
def _census():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "pass_census", root / "scripts" / "pass_census.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # `dataclass` resolves string annotations through `sys.modules[cls.__module__]`;
    # a spec-loaded module must be registered before exec or every Row fails.
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M400B = dict(
    d_model=1536,
    n_heads=24,
    head_dim=64,
    d_ffn=4096,
    vocab=256,
    max_seq_len=1024,
    max_iter=3,
    rank=64,
    n_experts=512,
    moe_topk=2,
    engram_rows=524288,
    engram_orders=[2, 3, 4],
    engram_dim=32,
    bf16=True,
    use_kda=True,
    use_engram=True,
    use_attnres=True,
    use_mhc=True,
    mhc_streams=2,
    layerrope=True,
    layerrope_form="paper",
    layerrope_index="iteration",
    use_csa=True,
    csa_chunk=8,
    csa_kv_dim=64,
    csa_topk=32,
    csa_coarse_chunk=256,
    csa_fine=True,
    share_u=True,
    share_u_mode="plane",
    fused_residual=True,
    fused_epilogue=True,
    fused_kda_tail=False,
)


def test_census_prices_the_flag_before_it_runs() -> None:
    """OFF vs ON at the m400b shape: -404 MB and -9 launches per iteration.

    This is the arithmetic the module docstring quotes; if a code path changes,
    this table changes and the test says by how much. The write assembly is the
    byte win (5 launches -> 1); the KDA tail is the launch win (6 -> 1) and
    saves only the double read of `beta`.
    """
    pc = _census()
    off = ModelConfig(**M400B)
    on = dataclasses.replace(off, fused_kda_tail=True)
    c = pc.ctx_for(off, 8, 1024)
    rows_off = pc._rows(off, c)
    rows_on = pc._rows(on, c)
    b_off = sum(r.nbytes for r in rows_off)
    b_on = sum(r.nbytes for r in rows_on)
    l_off = sum(r.launches for r in rows_off)
    l_on = sum(r.launches for r in rows_on)
    print(
        f"\n  census: {b_off / 1e9:.3f} GB / {l_off} launches -> "
        f"{b_on / 1e9:.3f} GB / {l_on} launches  "
        f"(Δ {b_on - b_off:+,} B, {l_on - l_off:+d} launches)"
    )
    assert l_off - l_on == 9, "6 tail launches + 4 write passes -> 2 kernels"
    assert b_off - b_on == pytest.approx(402.6e6, rel=0.01)
    # The census's own ranking: the Engram read is #1, the write assembly #2,
    # the KDA tail #5. The flag is named for the arm it fuses, not the ranking.
    groups = {}
    for r in rows_off:
        groups[r.group] = groups.get(r.group, 0) + r.nbytes
    ranked = sorted(groups, key=groups.get, reverse=True)
    assert ranked[0] == "engram", ranked
    assert "write" in ranked[1:3], ranked


# --------------------------------------------------------------------------- #
# 6. the card: the real kernels and the launch count (written, not run here)
# --------------------------------------------------------------------------- #
@CUDA
def test_kernel_parity_is_fp32_roundoff() -> None:
    rep = fk.parity(b=3, t=8, d=64, h=4, seed=0, use_kernel=True, gains=True)
    fk.assert_parity(rep)


def _launches(fn) -> int:
    """`cudaLaunchKernel` events for one call — the eager path's launch cost."""
    from torch.profiler import ProfilerActivity, profile

    fn()  # warm: the kernels must be compiled before the counted run
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    return sum(1 for e in prof.events() if e.name == "cudaLaunchKernel")


@CUDA
def test_kernel_launch_counter_drops(monkeypatch) -> None:
    """One tail launch + one combine launch where the eager arms make ELEVEN."""
    dev = "cuda"
    b, t, d, h = 2, 8, 64, 4
    q = torch.randn(b, t, d, device=dev)
    k = torch.randn(b, t, d, device=dev)
    v = torch.randn(b, t, d, device=dev)
    g = torch.randn(b, t, d, device=dev)
    beta = torch.randn(b, t, h, device=dev)
    a = torch.randn(b, t, d, device=dev)
    bb = torch.randn(b, t, d, device=dev)
    c = torch.randn(b, t, d, device=dev)
    g1 = torch.randn(d, device=dev)
    g3 = torch.randn(d, device=dev)

    def fused():
        out = fk.tail(q, k, v, g, beta, torch.bfloat16)
        out = out + (fk.combine(a, bb, c, g1, g3),)
        del out

    def eager():
        outs = (
            q.half(),
            k.half(),
            v.half(),
            torch.nn.functional.logsigmoid(g),
            torch.sigmoid(beta).half(),
            (a * g1 + bb * g1) + c * g3,
        )
        del outs

    n_fused = _launches(fused)
    n_eager = _launches(eager)
    print(f"\n  launches/iteration: fused {n_fused}, eager {n_eager}")
    # A zero from the profiler would make the DROP trivially true; the count that
    # matters is only a measurement if the fused arm launched something.
    assert n_fused > 0, "the profiler saw no kernel launch from the fused arm — nothing measured"
    assert n_fused <= 3, f"two fused chains plus the tuple add must be <= 3, saw {n_fused}"
    assert n_eager - n_fused >= 8, f"expected a large drop, saw {n_eager} -> {n_fused}"

    counters = Counters()
    fk.tail(q, k, v, g, beta, torch.bfloat16, counters)
    fk.combine(a, bb, c, g1, g3, counters)
    drained = counters.drain()
    assert drained.get("fused_kda_tail_launches", 0) == 2, drained
