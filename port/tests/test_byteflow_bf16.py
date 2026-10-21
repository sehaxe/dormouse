"""The bf16 ByteFlow channel: the dtype map, parity, the fp32 islands, no sync.

`byteflow.py`'s precision rule is the whole design: every matmul runs in bf16
with fp32 masters (`mm_mixed`, the loop's policy), the fp32 residual stream is
untouched, and three named islands stay fp32. This file gates each half of
that, at a fixed seed on a 32 KiB slice of the real corpus
(`fixtures/byteflow_real_32k.txt`, bytes 0..32 KiB of
`/mnt/.../pretrain/real/corpus.bin`, sha256 9f8233ff… — the same text the
module docstring's real-corpus measurement used).

The numbers, all measured before the thresholds were written:

* **parity** (`gate a`): max|Δ logit| = 1.66–1.87·bf16-epsilon·max|logit| on
  this fixture's windows (1.43 at the smoke geometry on 256 real rows, 1.77 at
  the medium geometry on 64). bf16's epsilon is 2⁻⁸, so a gate at 4·eps·scale
  is the dtype's own noise, not a tolerance pulled from the air.
* **selection** (`gate b`): the fp32 and bf16 nets select the same K positions
  on 98.44 % of rows (99.77 % of individual positions); on the real corpus at
  the smoke geometry it is 96.88 % of 256 rows and 100 % of 64 medium rows.
  The threshold is 95 %, below the worst measured arm.
* **tie rate**: top-1 vs top-2 gains are within bf16 epsilon on 0.39 % of rows
  here (0.00 % on 256 real-corpus smoke rows) — but the k-th vs (k+1)-th
  gains, the pair that actually decides the Top-K boundary, tie on 72.27 %
  of rows here and 100 % of the real-corpus rows. That is why the gains and
  the top-k stay fp32: the boundary is decided among numerically equal gains.

CPU-only; the memory half of `gate d` is the CUDA arm (skipped here) because
PyTorch exposes no CPU allocator peak.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import dormouse.model.byteflow as bf_mod
from dormouse.config import ModelConfig, validate
from dormouse.instrument import LoudError
from dormouse.model.byteflow import (
    ByteFlowConfig,
    ByteFlowNet,
    RateMode,
    marginal_gains_l2,
    select_positions,
)

FIXTURE = Path(__file__).parent / "fixtures" / "byteflow_real_32k.txt"
B, T, K = 2, 128, 32
EPS_BF16 = 2.0**-8
#: gate a's bar: 4 bf16 ulps of the logit scale (measured 1.87, gate a).
PARITY_EPS = 4.0
#: gate b's bar: below every measured arm (98.44 % fixture, 96.88 % corpus).
SELECT_BAR = 0.95


def net_config(**over) -> ByteFlowConfig:
    """A geometry small enough for a CPU test and wide enough to be a net."""
    base = dict(
        d_local=64,
        d_global=96,
        k_tokens=K,
        e_layers=2,
        g_layers=0,
        n_heads_local=4,
        n_heads_global=4,
        w_local=128,
        d_ff_local=128,
        d_ff_global=160,
        bins=8,
        eps2=0.5,
        max_bytes=T,
    )
    base.update(over)
    return ByteFlowConfig(**base)


def windows() -> torch.Tensor:
    raw = FIXTURE.read_bytes()
    assert len(raw) == 32 * 1024, "the fixture is pinned by sha256, not by its length"
    ids = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.long)
    return ids.reshape(-1, B, T)


def net_pair() -> tuple[ByteFlowNet, ByteFlowNet]:
    """Same seed → the SAME fp32 masters; only the compute dtype differs."""
    torch.manual_seed(0)
    n32 = ByteFlowNet(net_config())
    torch.manual_seed(0)
    n16 = ByteFlowNet(dataclasses.replace(net_config(), dtype=torch.bfloat16))
    s32, s16 = n32.state_dict(), n16.state_dict()
    for (name, a), (name16, b) in zip(s32.items(), s16.items(), strict=True):
        assert name == name16 and torch.equal(a, b), f"{name}: the dtype moved a value"
    return n32, n16


# ---------------------------------------------------------------------------
# gate a — bf16 vs fp32 logits
# ---------------------------------------------------------------------------


def test_gate_a_bf16_logits_are_within_the_dtypes_own_noise():
    """Parity, not identity: max|Δ| ≤ 4·eps·max|logit|, measured printed."""
    n32, n16 = net_pair()
    x = windows()[0]
    with torch.no_grad():
        l32, l16 = n32(x), n16(x)
    assert l32.dtype == torch.float32 and l16.dtype == torch.float32, "fp32 arm boundary"
    delta = (l32 - l16).abs().max().item()
    scale = l32.abs().max().item()
    ratio = delta / (EPS_BF16 * scale)
    print(f"[bf16] max|Δlogit|={delta:.6g} scale={scale:.6g} = {ratio:.2f}·bf16-eps")
    assert delta <= PARITY_EPS * EPS_BF16 * scale, f"{ratio:.2f} eps, bar {PARITY_EPS}"


# ---------------------------------------------------------------------------
# gate b — the chunker's selection
# ---------------------------------------------------------------------------


def test_gate_b_selection_agrees_between_dtypes():
    """The Top-K boundary is the same set at both dtypes, at the measured rate."""
    n32, n16 = net_pair()
    rows = exact = positions = 0
    for batch in windows():
        with torch.no_grad():
            _, _, s32 = n32.encode_chunks(batch, K)
            _, _, s16 = n16.encode_chunks(batch, K)
        rows += batch.shape[0]
        exact += int((s32 == s16).all(dim=1).sum())
        positions += int((s32 == s16).sum())
    rate = exact / rows
    print(
        f"[bf16] selection: exact-set {rate:.2%} of {rows} rows, "
        f"positions {positions / (rows * K):.2%}"
    )
    assert rate >= SELECT_BAR, f"{rate:.2%} < {SELECT_BAR:.0%}, the measured floor"


def _raw_bf16_gains(h: torch.Tensor) -> torch.Tensor:
    """`marginal_gains_l2` with its fp32 widening removed — the defect, kept as
    a probe: bf16 `cumsum`/`sqrt` over the squared norms."""
    hb = h.to(torch.bfloat16)
    cum = hb.pow(2).sum(-1).cumsum(1).sqrt()
    zero = cum[:, :1] * 0.0
    return cum - torch.cat([zero, cum[:, : cum.shape[1] - 1]], dim=1)


def test_gate_b_gains_and_topk_are_the_fp32_island():
    """A bf16-STORED `h` scores the SAME gains as its fp32 widening.

    The island is the widening at `marginal_gains_l2`'s entry. Without it the
    gains (and every tie in them) would be computed in bf16: on this fixture
    the raw bf16 computation selects the same positions as fp32 on 0.00 % of
    rows (measured), which is the discrete flip with no gradient the rule
    exists to prevent.
    """
    _, n16 = net_pair()
    h = n16.embedding(windows()[0])
    for blk in n16.encoder:
        h = blk(h)
    assert h.dtype == torch.float32
    stored = h.to(torch.bfloat16)
    widened = marginal_gains_l2(stored)
    assert widened.dtype == torch.float32, "the gains were scored in bf16"
    assert torch.equal(widened, marginal_gains_l2(stored.to(torch.float32)))
    assert torch.equal(select_positions(widened, K), select_positions(widened.to(torch.float32), K))

    rows = exact = 0
    for batch in windows():
        with torch.no_grad():
            hb = n16.embedding(batch)
            for blk in n16.encoder:
                hb = blk(hb)
        ref = select_positions(marginal_gains_l2(hb), K)
        raw = select_positions(_raw_bf16_gains(hb), K)
        rows += batch.shape[0]
        exact += int((ref == raw).all(dim=1).sum())
    print(f"[bf16] gains without the fp32 widening: {exact / rows:.2%} of {rows} rows agree")
    assert exact / rows < 0.5, "the widening is not load-bearing on this fixture"


# ---------------------------------------------------------------------------
# gate c — no host sync in the forward
# ---------------------------------------------------------------------------


def test_gate_c_the_forward_touches_no_host(monkeypatch):
    """`.item()`, `.tolist()` and the scalar dunders are refused mid-forward."""

    def boom(*_a, **_k):
        raise AssertionError("host sync in the ByteFlow forward (ADR-0018 rule 2)")

    n16 = net_pair()[1]
    for name in ("item", "tolist", "__int__", "__float__", "__bool__"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=True)
    with torch.no_grad():
        logits = n16(windows()[0])
    # `torch.equal` returns a Python bool (C++ side), so the assertion itself
    # does not go through the patched `__bool__`.
    assert torch.equal(torch.isfinite(logits).all(), torch.tensor(True))


# ---------------------------------------------------------------------------
# gate d — the flag moves the dtype, not the model
# ---------------------------------------------------------------------------


def test_gate_d_params_and_state_are_unchanged_by_the_flag():
    """Same parameter inventory, same fp32 masters, same persistent state.

    The only tensors the flag may move are the derived RoPE tables (non-
    persistent buffers: not in the checkpoint, not optimizer state).
    """
    n32, n16 = net_pair()
    p32 = {n: (tuple(p.shape), p.dtype, p.numel()) for n, p in n32.named_parameters()}
    p16 = {n: (tuple(p.shape), p.dtype, p.numel()) for n, p in n16.named_parameters()}
    assert p32 == p16, "the flag changed the parameter set"
    assert {d for _, d, _ in p16.values()} == {torch.float32}, "a master left fp32"
    assert torch.equal(
        torch.cat([p.reshape(-1) for _, p in n32.named_parameters()]),
        torch.cat([p.reshape(-1) for _, p in n16.named_parameters()]),
    )
    assert n32.state_dict().keys() == n16.state_dict().keys()
    for name, t in n16.state_dict().items():
        assert t.dtype == n32.state_dict()[name].dtype, f"{name}: checkpoint dtype moved"
    moved = {n for n, b in n16.named_buffers() if b.dtype != dict(n32.named_buffers())[n].dtype}
    assert moved and all("rope." in n for n in moved), moved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CPU allocator peak exists")
def test_gate_d_peak_memory_is_not_larger():
    """CUDA: the bf16 forward's peak allocation is ≤ the fp32 one's."""
    torch.manual_seed(0)
    n32 = ByteFlowNet(net_config(), device="cuda")
    x = windows()[0].cuda()
    with torch.no_grad():
        n32(x)
        torch.cuda.synchronize()
        peak32 = torch.cuda.max_memory_allocated()
    del n32
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    n16 = ByteFlowNet(dataclasses.replace(net_config(), dtype=torch.bfloat16), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        n16(x)
        torch.cuda.synchronize()
        peak16 = torch.cuda.max_memory_allocated()
    print(f"[bf16] peak fp32={peak32 / 2**20:.1f} MiB bf16={peak16 / 2**20:.1f} MiB")
    assert peak16 <= peak32


# ---------------------------------------------------------------------------
# gate e — bf16 + the logdet chunker is LOUD
# ---------------------------------------------------------------------------


def _run(**over) -> ModelConfig:
    base = dict(
        use_byteflow=True,
        bf16=True,
        use_engram=False,
        byteflow_k_tokens=8,
        dspark_weight=0.0,
        sigreg_weight=0.0,
        mtp_weight=0.0,
    )
    base.update(over)
    return ModelConfig(**base)


def test_gate_e_bf16_and_the_logdet_chunker_are_refused():
    """A bf16 Cholesky is unstable, so the pair dies at load, naming both fields."""
    with pytest.raises(LoudError) as e:
        validate(_run(byteflow_logdet=True))
    assert "bf16 + byteflow_logdet" in str(e.value)
    # ... and the same refusal at the runtime seam, where a hand-built net
    # arrives without `validate` (the model constructors never call it).
    with pytest.raises(LoudError, match="bf16 \\+ byteflow_logdet"):
        ByteFlowNet(net_config(rate_mode=RateMode.LOGDET, dtype=torch.bfloat16))
    # ... and the two legal neighbours still validate: bf16 on the L2 path ...
    validate(_run())
    # ... and the exact chunker at fp32.
    validate(_run(bf16=False, byteflow_logdet=True))


# ---------------------------------------------------------------------------
# gate f — the fp32 path IS the pre-bf16 function
# ---------------------------------------------------------------------------


def test_gate_f_fp32_logits_are_the_pre_bf16_expression(monkeypatch):
    """`mm_mixed` at fp32 is `F.linear`, bit for bit — the old op, not a similar one.

    `nn.Linear` calls `F.linear`; patching the module's `mm_mixed` to it makes
    the net run exactly the pre-bf16 expression through every projection
    (`_MixedLinear`, the head is `self.out` at fp32, `upsample`'s `@`).
    """
    n32, _ = net_pair()
    x = windows()[0]
    with torch.no_grad():
        before = n32(x)
    monkeypatch.setattr(bf_mod, "mm_mixed", lambda a, w, dt: F.linear(a, w))
    with torch.no_grad():
        after = n32(x)
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# the channel end to end (Option B, `model.py`) — bf16 is not a dormant flag
# ---------------------------------------------------------------------------


def test_the_channel_runs_bf16_and_trains_its_masters():
    """`DormouseModel` with `use_byteflow` + `bf16`: logits finite, grads flow.

    The dtype knob is at the net, so this is the wiring gate: the loop's fp32
    state and the net's bf16 matmuls meet at `in_proj`/`out_proj`, and every
    channel parameter (fp32 masters) receives a finite non-zero gradient.
    """
    from dormouse.model import DormouseModel

    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        head_dim=16,
        d_ffn=128,
        vocab=256,
        max_seq_len=64,
        max_iter=2,
        rank=8,
        n_experts=2,
        moe_topk=1,
        use_kda=True,
        use_engram=False,
        bf16=True,
        use_byteflow=True,
        byteflow_k_tokens=16,
        byteflow_d_local=32,
        byteflow_d_global=32,
        byteflow_heads_local=4,
        byteflow_heads_global=4,
        byteflow_e_layers=1,
        byteflow_w_local=64,
        byteflow_d_ff_local=64,
        byteflow_bins=4,
        byteflow_max_bytes=64,
    )
    torch.manual_seed(0)
    model = DormouseModel(cfg)
    assert model.bf is not None and model.bf_net is None, "Option B must be armed"
    ids = windows()[0][:, :64]
    logits, loss, counters = model(ids, targets=ids)
    assert tuple(logits.shape) == (B * 64, 256)
    assert torch.isfinite(logits).all()
    assert loss is not None and bool(torch.isfinite(loss).detach())
    assert counters.get("bf_channel") == 1
    loss.backward()
    assert model.bf is not None
    checked = 0
    for name, p in model.bf.named_parameters():
        assert p.dtype == torch.float32, f"{name}: a master left fp32"
        assert p.grad is not None, f"bf.{name}: no gradient"
        assert bool(torch.isfinite(p.grad).all()), f"bf.{name}: non-finite gradient"
        assert float(p.grad.abs().max()) > 0.0, f"bf.{name}: gradient is exactly zero"
        checked += 1
    assert checked > 10, checked
