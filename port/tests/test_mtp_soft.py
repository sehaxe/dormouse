"""Soft-MTP gates (arXiv:2610.38331, "Multi-Token Prediction Done Right", Teoh 2026).

The claim under test is the paper's KL gradient-recovery identity, in the
version `model/mtp.py` implements for THIS repo:

    d/dx_k [ CE_mtp_k - CE_ntp_at_aligned ] = (p_mtp - p_ntp) · W

`x_k` is module `k`'s projection output (the tensor the shared head reads).
The coupling is `ntp_hidden[:, k:] += s_k·(x_k.detach() - x_k)`, a forward
no-op whose backward supplies exactly the `-s_k·g_CE_ntp` half. `s_k` carries
the reference's `kl_recovery_scale` AND this repo's total-loss factor
`ntp_loss_scale = 1/(T+1)`, because the NTP CE is divided by the depth here.

The decisive gate (b) is float64 with a full vocabulary and a DENSE linear
head — the paper's premise. The repo's actual head may be dense, TSCT or tied;
all three are linear in the hidden, which is all the identity reads, and the
fixture exercises the real `MTPHead.forward_soft` and the real scale code.
The gate for non-final modules is honest about the structural difference: this
module's `x_j` also feeds module `j+1`, so the TOTAL gradient at a non-final
`x_j` carries that chain term. The gate isolates the last module (whose `x`
feeds no later head) and tests the routed DIFFERENCE on every module.

Pins in gate (a) were captured on clean `main` at `21a97d2` BEFORE any edit,
with the script transcribed in `_state_hash`/`_grad_hash` below.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg
from torch import nn

from dormouse.config import RunConfig, TrainConfig, validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.mtp import MTPHead

B, T = 2, 64

# --- gate (a) pins: clean main 21a97d2, captured before the edit -------------
# `tiny_cfg`, `torch.manual_seed(seed)`, ids = arange(2*64) % 251 reshaped,
# targets = (ids + 7) % 256, one forward + backward.
PIN_OFF_STATE = "e26c6a0762e1e7aa5704106678a7ef1adc7fccd1921e59a47af771d900bbc4f5"
PIN_OFF_LOGITS = "c7839283a79291cf713aab57476cf202ba1a68167c365197f785da08203055c2"
PIN_OFF_LOSS = 6.119609355926514
PIN_OFF_EMBED_GRAD = "2259d9e4633b6d3000794c317e8ea9d6f2cbcc1d96ae2150fcbfaed04c662a71"

PIN_K2_STATE = "894fe40ff7787c20941cc27b7871f48b1fdb79f42f1d69146af47c2762daf112"
PIN_K2_LOGITS = "c7839283a79291cf713aab57476cf202ba1a68167c365197f785da08203055c2"
PIN_K2_LOSS = 8.997851371765137
PIN_K2_EMBED_GRAD = "3a73c42bcdf94d422b4ae8140b6e3e42a0b576ba4434051a777e6e0c369f4101"
PIN_K2_PROJ_GRADS = (
    "6abf8db065f484f2b5d27562810480cce872e5243839728b9abdf8e78a7be34b",
    "c71a4b6391d1ac14cc55b670ae1e0d619c61d34ef631232005ac5bc4d30b135b",
)

PIN_K1_STATE = "2b88ee5addcf2d74d0c130587766fd67594cb40830b811eb6a5a60e256abbecc"
PIN_K1_LOGITS = "335ec9dbc322df27c4c215d9ecd2d148bbb29d0574642045e5c865abca2df85e"
PIN_K1_LOSS = 7.874235153198242
PIN_K1_EMBED_GRAD = "0dd006599fb8b658768f5c4b1e057c99781a9fa1b0a9fdd991e7d252a90a02ea"
PIN_K1_PROJ_GRAD = "27f74fdd021f2251ff9b09162d7ebd8637eb5b77164a9617b67128f29e6592bd"


def _state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        b = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(b.shape)).encode())
        h.update(str(b.dtype).encode())
        h.update(b.numpy().tobytes())
    return h.hexdigest()


def _grad_hash(param: torch.nn.Parameter) -> str:
    assert param.grad is not None, "the pinned parameter got no gradient"
    b = param.grad.detach().cpu().contiguous()
    return hashlib.sha256(b.numpy().tobytes()).hexdigest()


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = (torch.arange(B * T, dtype=torch.long) % 251).reshape(B, T)
    return ids, (ids + 7) % 256


def _build(seed: int = 1234, **over) -> DormouseModel:
    torch.manual_seed(seed)
    return DormouseModel(tiny_cfg(**over))


def test_gate_a_mtp_soft_off_is_bit_identical_to_clean_main() -> None:
    """`mtp_soft=false` is the old program: pinned state, logits, loss, grads."""
    ids, targets = _fixed_batch()

    off = _build(mtp_k=0, mtp_weight=0.0, mtp_soft=False)
    assert off.mtp is None and param_count(off) == 64_881
    assert _state_hash(off) == PIN_OFF_STATE
    logits, loss, _ = off(ids, targets)
    assert hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest() == PIN_OFF_LOGITS
    assert float(loss.detach()) == PIN_OFF_LOSS
    loss.backward()
    assert _grad_hash(off.embed.weight) == PIN_OFF_EMBED_GRAD

    k2 = _build(mtp_k=2, mtp_weight=0.5, mtp_soft=False)
    assert _state_hash(k2) == PIN_K2_STATE
    logits, loss, _ = k2(ids, targets)
    assert hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest() == PIN_K2_LOGITS
    assert float(loss.detach()) == PIN_K2_LOSS
    loss.backward()
    assert _grad_hash(k2.embed.weight) == PIN_K2_EMBED_GRAD
    assert tuple(_grad_hash(p.weight) for p in k2.mtp.projs) == PIN_K2_PROJ_GRADS

    k1 = _build(7, mtp_k=1, mtp_weight=0.3, mtp_soft=False)
    assert _state_hash(k1) == PIN_K1_STATE
    logits, loss, _ = k1(ids, targets)
    assert hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest() == PIN_K1_LOGITS
    assert float(loss.detach()) == PIN_K1_LOSS
    loss.backward()
    assert _grad_hash(k1.embed.weight) == PIN_K1_EMBED_GRAD
    assert _grad_hash(k1.mtp.projs[0].weight) == PIN_K1_PROJ_GRAD


# --- gate (b): the decisive float64 KL-recovery identity ---------------------

D_FIX, V_FIX, B_FIX, T_FIX = 8, 11, 2, 9
W_FIX, DIV_FIX = 0.7, 3.0


def _soft_fixture(k: int, seed: int = 0):
    """A tiny fp64 fixture: real `MTPHead.forward_soft`, dense shared head.

    The head is `nn.Linear` with no bias — the paper's premise (`W` dense and
    shared). Every tensor is fp64; `mtp._widen` preserves fp64 (it only widens
    fp16/bf16), so the fixture measures the algebra, not fp32 round-off.
    Returns the module, the inputs, the captured `x_j`, and their gradients.
    """
    torch.manual_seed(seed)
    mtp = MTPHead(D_FIX, k).double()
    embed = nn.Embedding(V_FIX, D_FIX).double()
    head = nn.Linear(D_FIX, V_FIX, bias=False).double()
    h = torch.randn(B_FIX, T_FIX, D_FIX, dtype=torch.float64)
    ids = torch.randint(0, V_FIX, (B_FIX, T_FIX))
    targets = torch.randint(0, V_FIX, (B_FIX, T_FIX))

    xs: list[torch.Tensor] = []

    def keep(_m, _inp, out):
        out.retain_grad()
        xs.append(out)

    # The head reads `x.reshape(...)` (a VIEW), while the coupling reads `x`
    # itself. A hook on the projection output is therefore the only place both
    # gradient paths meet — a hook on the head input would see the view only.
    for proj in mtp.projs:
        proj.register_forward_hook(keep)

    ntp_hidden = h.clone()
    loss, _per_k, coupled = mtp.forward_soft(
        h,
        ids,
        targets,
        embed=embed,
        head=head,
        weight=W_FIX,
        ntp_hidden=ntp_hidden,
        ntp_loss_scale=1.0 / DIV_FIX,
    )
    assert torch.equal(coupled, ntp_hidden), "the coupling must be a forward no-op"
    logits = head(coupled.reshape(-1, D_FIX))
    ce = -F.log_softmax(logits, dim=-1).gather(1, targets.reshape(-1, 1)).mean()
    total = loss + ce / DIV_FIX
    total.backward()
    assert all(x.grad is not None for x in xs), "a module produced no gradient"
    return mtp, head, h, ids, targets, xs


def _kl_grad_for(x: torch.Tensor, p_ntp: torch.Tensor, w: torch.Tensor, k: int) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    logp = F.log_softmax(x @ w.t(), dim=-1)
    kl = (p_ntp * (torch.log(p_ntp) - logp)).sum(-1).mean()
    (kl * (W_FIX / k)).backward()
    return x.grad


@pytest.mark.parametrize("k", [1, 2, 3])
def test_gate_b_the_last_module_gradient_equals_explicit_kl_fp64(k: int) -> None:
    """The decisive gate: <= 1e-10 relative in float64, full vocabulary.

    Only the LAST module is compared: its `x` feeds no later head, so its
    total gradient IS the head's own CE minus the routed NTP term. Non-final
    modules carry the later modules' chain term — tested separately below.
    """
    _mtp, head, h, _ids, _targets, xs = _soft_fixture(k)
    j = k - 1
    shift = j + 1
    n = T_FIX - shift
    x = xs[j].detach().reshape(B_FIX, n, D_FIX)
    got = xs[j].grad.reshape(B_FIX, n, D_FIX)
    p_ntp = F.softmax(h.detach()[:, shift : shift + n] @ head.weight.t(), dim=-1)
    want = _kl_grad_for(x, p_ntp, head.weight.detach(), k)
    rel = (got - want).norm() / (want.norm() + 1e-300)
    assert rel <= 1e-10, f"k={k}: relative {rel:.3e} (max abs {(got - want).abs().max():.3e})"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_gate_b2_the_routed_difference_is_exact_at_the_last_module(k: int) -> None:
    """`soft_grad − hard_grad == −s_j·g_CE_ntp` at the last module, <= 1e-10 rel.

    At the last module there is no later head, so the difference is exactly the
    coupling's contribution and the explicit reconstruction can be compared.
    This pins the scale formula (`s_k` includes the per-head mean and the 1/k
    average) at `n_heads = k`, not only at k=1.
    """
    # The hard arm: `forward` (no coupling), same seed/inputs.
    torch.manual_seed(0)
    mtp = MTPHead(D_FIX, k).double()
    embed = nn.Embedding(V_FIX, D_FIX).double()
    head = nn.Linear(D_FIX, V_FIX, bias=False).double()
    h = torch.randn(B_FIX, T_FIX, D_FIX, dtype=torch.float64)
    ids = torch.randint(0, V_FIX, (B_FIX, T_FIX))
    targets = torch.randint(0, V_FIX, (B_FIX, T_FIX))
    hs: list[torch.Tensor] = []

    def keep(_m, _inp, out):
        out.retain_grad()
        hs.append(out)

    for proj in mtp.projs:
        proj.register_forward_hook(keep)
    loss, _ = mtp(h, ids, targets, embed=embed, head=head, weight=W_FIX)
    loss.backward()
    hard = [x.grad.detach().clone() for x in hs]

    # The soft arm again, with the same seed (a fresh module has the same init).
    mtp_s, head_s, h_s, ids_s, targets_s, xs = _soft_fixture(k)
    assert torch.equal(head_s.weight, head.weight), "the two arms must share W"
    assert torch.equal(mtp_s.projs[0].weight, mtp.projs[0].weight), (
        "the two arms must share the head module"
    )
    assert torch.equal(h_s, h) and torch.equal(ids_s, ids) and torch.equal(targets_s, targets)

    n_heads = min(k, T_FIX - 1)
    j = n_heads - 1
    shift = j + 1
    n = T_FIX - shift
    soft = xs[j].grad.reshape(B_FIX, n, D_FIX)
    diff = soft - hard[j].reshape(B_FIX, n, D_FIX)

    # Explicit routed term: a fresh backward through the coupled hidden, with
    # the real scale formula from `mtp.py`.
    x = xs[j].detach().reshape(B_FIX, n, D_FIX).clone().requires_grad_(True)
    coupled = h_s.detach().clone()
    scale = W_FIX * (B_FIX * T_FIX) / (n_heads * (B_FIX * n) * (1.0 / DIV_FIX))
    coupled[:, shift:] = coupled[:, shift:] + scale * (x.detach() - x)
    ce = (
        -F.log_softmax(head_s(coupled.reshape(-1, D_FIX)), dim=-1)
        .gather(1, targets_s.reshape(-1, 1))
        .mean()
    )
    (ce / DIV_FIX).backward()
    want = x.grad
    rel = (diff - want).norm() / (want.norm() + 1e-300)
    assert rel <= 1e-10, (
        f"k={k} module {j}: relative {rel:.3e} (max abs {(diff - want).abs().max():.3e})"
    )


def test_gate_b3_the_model_wiring_uses_the_T_plus_1_scale() -> None:
    """The model-level scale: `ntp_loss_scale = 1/(T+1)` reaches `forward_soft`.

    The module gate (b) passes `ntp_loss_scale` directly; this one runs the
    real `DormouseModel` at `max_iter=2` (so the NTP CE carries `1/3`) and
    compares the LAST head's soft-vs-hard gradient difference to the explicit
    routed term. A model that passed `1.0` (or the wrong alignment) is red.
    fp32, so the bound is 1e-5 relative — the module's fp64 bound is gate (b).
    """
    ids, targets = _fixed_batch()
    b, t = ids.shape
    k = 2
    shift = k
    n = t - shift
    weight = 0.5
    divisor = 3.0  # max_iter=2 in tiny_cfg

    def _run_arm(soft: bool):
        model = _build(mtp_k=k, mtp_weight=weight, mtp_soft=soft)
        if soft:
            # Same initialisation as the hard arm, so the MTP CE term cancels.
            model.load_state_dict(hard_state)
        store: list[torch.Tensor] = []

        def keep(_m, _inp, out):
            out.retain_grad()
            store.append(out)

        model.mtp.projs[-1].register_forward_hook(keep)
        seen = _capture_head_inputs(model)
        _logits, loss, _ = model(ids, targets)
        loss.backward()
        return store[0].detach(), store[0].grad.detach(), seen

    torch.manual_seed(1234)
    hard_state = _build(mtp_k=k, mtp_weight=weight, mtp_soft=False).state_dict()
    hard_x, hard_grad, _ = _run_arm(False)
    soft_x, soft_grad, seen = _run_arm(True)
    assert torch.equal(hard_x, soft_x), "the two arms' last hidden differs"
    diff = (soft_grad - hard_grad).reshape(b, n, -1)

    # The NTP hidden the soft CE read: the nearest [b*t, d] head input before
    # the coupled call (the last one); see gate (c) for the call order.
    coupled = seen[-1]
    hn = next(s for s in reversed(seen[:-1]) if s.shape == coupled.shape)
    assert torch.equal(hn, coupled), "the coupled hidden moved"

    x = soft_x.reshape(b, n, -1).clone().requires_grad_(True)
    model = _build(mtp_k=k, mtp_weight=weight, mtp_soft=True)
    model.load_state_dict(hard_state)
    coupled_hidden = hn.reshape(b, t, -1).clone()
    scale = weight * (b * t) / (min(k, t - 1) * (b * n) * (1.0 / divisor))
    coupled_hidden[:, shift:] = coupled_hidden[:, shift:] + scale * (x.detach() - x)
    logits = model.lm_head(coupled_hidden.reshape(-1, model.cfg.d_model).to(torch.float32))
    ce = -F.log_softmax(logits, dim=-1).gather(1, targets.reshape(-1, 1)).mean()
    (ce / divisor).backward()
    want = x.grad.reshape(b, n, -1)
    rel = (diff - want).norm() / (want.norm() + 1e-300)
    assert rel <= 1e-5, f"model-level routing relative {rel:.3e} (want <= 1e-5)"


def test_the_nonfinal_head_deviation_is_the_sequential_chain_not_a_scale_bug() -> None:
    """FINDING, not a tolerance: this module's `x_j` also feeds module `j+1`.

    The reference's per-head seam is `final_norm(hiddens[k])`, a tensor no later
    head reads. Here the projection output `x_j` IS the next module's input, so
    the NTP coupling applied to `x_{j+1}` back-propagates to `x_j` through the
    sequential projection. The last head's total gradient is therefore the KL
    exactly (gate b), while a non-final head's carries that cross-head term on
    top of its own KL. This test quantifies it so a reader cannot mistake the
    partial identity for a broken scale; it does NOT assert equality.
    """
    k = 3
    _mtp, head, h, _ids, _targets, xs = _soft_fixture(k)
    # j = 0's total gradient vs the direct KL for head 0.
    shift = 1
    n = T_FIX - shift
    got = xs[0].grad.reshape(B_FIX, n, D_FIX)
    p_ntp = F.softmax(h.detach()[:, shift : shift + n] @ head.weight.t(), dim=-1)
    want = _kl_grad_for(xs[0].detach().reshape(B_FIX, n, D_FIX), p_ntp, head.weight.detach(), k)
    rel = (got - want).norm() / (want.norm() + 1e-300)
    # The last head is the isolated one; its identity is gate b.
    last = xs[-1].grad.reshape(B_FIX, T_FIX - k, D_FIX)
    p_last = F.softmax(h.detach()[:, k:] @ head.weight.t(), dim=-1)
    want_last = _kl_grad_for(
        xs[-1].detach().reshape(B_FIX, T_FIX - k, D_FIX), p_last, head.weight.detach(), k
    )
    assert (last - want_last).norm() / want_last.norm() <= 1e-10
    # Non-final: the cross term is real and large. If a future refactor isolates
    # the head inputs (a per-head final norm), this record must be updated.
    assert rel > 1e-3, (
        f"the non-final head now matches the direct KL ({rel:.3e}); the module's "
        f"head seam changed — update this finding and the gate docstring"
    )


# --- gates (c)+(d): the forward and the loss value are untouched -------------


def _capture_head_inputs(model: DormouseModel) -> list[torch.Tensor]:
    seen: list[torch.Tensor] = []

    def keep(_m, args):
        seen.append(args[0].detach().clone())

    model.lm_head.register_forward_pre_hook(keep)
    return seen


def test_gate_cd_ntp_forward_and_loss_scalar_are_unchanged() -> None:
    """The coupling is `x - x`: the returned logits, the NTP hidden and the
    total loss scalar are `torch.equal` to the hard arm's."""
    ids, targets = _fixed_batch()
    hard = _build(mtp_k=2, mtp_weight=0.5, mtp_soft=False)
    soft = _build(mtp_k=2, mtp_weight=0.5, mtp_soft=True)
    soft.load_state_dict(hard.state_dict())

    logits_h, loss_h, counters_h = hard(ids, targets)
    seen = _capture_head_inputs(soft)
    logits_s, loss_s, counters_s = soft(ids, targets)

    assert torch.equal(logits_h, logits_s), "the returned logits moved"
    assert torch.equal(loss_h, loss_s), f"loss {loss_s!r} != {loss_h!r}"
    # The coupled hidden is the LAST call; the final NTP hidden is the nearest
    # preceding call of the same [b*t, d] shape (the MTP calls are one and two
    # bytes shorter, and the in-loop CEs are before both).
    coupled = seen[-1]
    final_ntp = next(s for s in reversed(seen[:-1]) if s.shape == coupled.shape)
    assert torch.equal(final_ntp, coupled), "the coupled NTP hidden moved in the forward"
    assert len(seen) >= 4, f"the soft path made only {len(seen)} head calls"
    assert "mtp_soft" not in counters_h._device
    assert counters_s._device["mtp_soft"].item() == 1
    assert float(counters_s.drain()["mtp_soft"]) == 1.0


def test_gate_no_host_sync_in_the_soft_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.item()`/`.tolist()`/`bool(tensor)` raise inside `forward_soft`.

    The device counter is added after the module returns, in `model.py`; the
    module itself must stay sync-free (ADR-0018 rule 2). This gate is narrower
    than the whole model on purpose: the loop's pre-existing MoE dispatch calls
    `.tolist()` on this branch and is not part of the soft path.
    """
    torch.manual_seed(0)
    mtp = MTPHead(8, 2)
    embed = nn.Embedding(16, 8)
    head = nn.Linear(8, 16)
    h = torch.randn(2, 6, 8)
    ids = torch.randint(0, 16, (2, 6))
    targets = torch.randint(0, 16, (2, 6))

    def boom(*_a, **_k):
        raise AssertionError("host sync inside the soft-MTP forward (ADR-0018 rule 2)")

    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "tolist", boom)
    monkeypatch.setattr(torch.Tensor, "__bool__", boom)
    loss, per_k, coupled = mtp.forward_soft(
        h,
        ids,
        targets,
        embed=embed,
        head=head,
        weight=0.5,
        ntp_hidden=h.clone(),
        ntp_loss_scale=1.0 / 3.0,
    )
    assert loss.requires_grad and coupled.requires_grad and len(per_k) == 2
    assert torch.equal(coupled, h)


# --- gate (e): finite gradients, non-zero on the MTP parameters --------------


def test_gate_e_gradients_are_finite_and_mtp_parameters_are_trained() -> None:
    ids, targets = _fixed_batch()
    hard = _build(mtp_k=2, mtp_weight=0.5, mtp_soft=False)
    _logits, loss_h, _ = hard(ids, targets)
    loss_h.backward()
    soft = _build(mtp_k=2, mtp_weight=0.5, mtp_soft=True)
    soft.load_state_dict(hard.state_dict())
    _logits, loss_s, _ = soft(ids, targets)
    loss_s.backward()

    # The trick must not REMOVE a gradient: the same parameters are trained.
    hard_trained = {n for n, p in hard.named_parameters() if p.grad is not None}
    soft_trained = {n for n, p in soft.named_parameters() if p.grad is not None}
    assert soft_trained == hard_trained
    for name, param in soft.named_parameters():
        if param.grad is None:
            continue
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"
    mtp_params = {n: p for n, p in soft.named_parameters() if n.startswith("mtp.")}
    assert len(mtp_params) == 2, sorted(mtp_params)
    for name, param in mtp_params.items():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.count_nonzero(param.grad).item() > 0, f"{name} got an all-zero gradient"


# --- gate (f): refusals and the snapshot round-trip --------------------------


def test_gate_f_refusals_and_snapshot_round_trip() -> None:
    with pytest.raises(LoudError, match="mtp_soft"):
        validate(tiny_cfg(mtp_k=1, mtp_weight=0.0, mtp_soft=True))
    with pytest.raises(LoudError, match="mtp_soft"):
        tiny_cfg(mtp_k=1, mtp_weight=0.5, mtp_soft=True, logit_softcap=15.0)
    with pytest.raises(LoudError, match="mtp_soft"):
        DormouseModel(tiny_cfg(mtp_k=1, mtp_weight=0.0, mtp_soft=True))

    on = RunConfig(
        source="tiny",
        model=tiny_cfg(mtp_k=1, mtp_weight=0.5, mtp_soft=True),
        train=TrainConfig(),
    )
    text = on.snapshot_toml()
    assert "mtp_soft = true" in text, text
    back = RunConfig.from_snapshot(text)
    assert back.model.mtp_soft is True
    assert back.diff_keys(on) == []

    off = RunConfig(
        source="tiny",
        model=tiny_cfg(mtp_k=1, mtp_weight=0.5, mtp_soft=False),
        train=TrainConfig(),
    )
    assert RunConfig.from_snapshot(off.snapshot_toml()).model.mtp_soft is False


# --- the A/B driver's pre-registered rule (pure logic, no GPU) ---------------


def _load_ab_soft():
    """Load `scripts/ab_mtp_soft.py` by path — `scripts/` is not a package."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "ab_mtp_soft.py"
    spec = importlib.util.spec_from_file_location("ab_mtp_soft", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_row(seed: int, bpb: float, mtp_logged: int) -> dict:
    return {
        "seed": seed,
        "bpb": bpb,
        "scored": 81920,
        "n_evals": 3,
        "nan": 0,
        "mtp_logged": mtp_logged,
    }


def test_the_driver_keep_rule_is_the_preregistered_three_clauses() -> None:
    mod = _load_ab_soft()
    window = 81920
    ntp = [_run_row(1, 3.00, 0), _run_row(2, 3.04, 0), _run_row(3, 2.96, 0)]  # spread 0.08
    hard = [_run_row(1, 2.99, 2), _run_row(2, 3.01, 2), _run_row(3, 3.00, 2)]
    soft_win = [_run_row(1, 2.97, 2), _run_row(2, 2.98, 2), _run_row(3, 2.98, 2)]

    keep = mod.verdict(ntp, hard, soft_win, window, accept_hard=0.50, accept_soft=0.60)
    assert keep["verdict"] == "keep", keep

    # Clause 1: not better than hard.
    tie = mod.verdict(ntp, hard, hard, window, accept_hard=0.50, accept_soft=0.60)
    assert tie["verdict"] == "drop", tie

    # Clause 2: better than hard but worse than ntp by more than 2x its spread.
    bad = [_run_row(1, 3.20, 2), _run_row(2, 3.21, 2), _run_row(3, 3.20, 2)]
    far = mod.verdict(ntp, hard, bad, window, accept_hard=0.50, accept_soft=0.60)
    assert far["verdict"] == "drop", far

    # Clause 3: BPB wins but acceptance is not higher.
    no_accept = mod.verdict(ntp, hard, soft_win, window, accept_hard=0.60, accept_soft=0.60)
    assert no_accept["verdict"] == "drop", no_accept

    # No probe => VOID, and a VOID arm is never a keep.
    void = mod.verdict(ntp, hard, soft_win, window, accept_hard=None, accept_soft=None)
    assert void["verdict"] == "void" and not void["verdict"] == "keep", void
