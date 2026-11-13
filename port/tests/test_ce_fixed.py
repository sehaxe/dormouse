"""The two softmax numerical fixes (arXiv:2610.31642) — every gate, proven RED.

Gates, in the brief's order:

a. `ce_fixed=false` is BIT-IDENTICAL to `main` (`3f83788`): the sha256 below
   was captured on main BEFORE the module existed (model: 5 AdamW steps, hash
   over `state_dict` + the five forwards' logits; trainer: 5 `train()` steps,
   hash over `state_dict` + the step-5 loss). The flag-off call is also the
   same CALL: the kwarg is absent when off.
b. The collapse: margin 8 in bf16 / 25 in fp32, the standard path's target
   gradient is exactly 0 and the gradient no longer sums to zero; the fixed
   path's target entry is `-(sum of the others)` and the sum is 0.
c. Below the cliff the fixed loss matches the fp64 closed form to 1e-6
   (relative) and the gradient to 1e-5 — where the standard path has already
   lost those digits (printed).
d. Symmetric: target = argmax and target = argmin, margins to 1e4, finite and
   equal to the fp64 closed form; no inf/nan.
e. Distillation: `student == teacher` in bf16 gives a gradient of EXACTLY 0
   and a KL of EXACTLY 0; the standard path's gradient norm is non-zero.
f. `ce_cliff_tokens` fires on exactly the tokens above the cliff, stays on the
   device (`count_device`), and the whole forward+backward touches no host.
g. Refusals + the counted clamp: a bad reduction/shape/temperature is LOUD,
   and a clamp that fires lands in `ce_clamp_hits`.
i. The suite is the last gate (its own run).

The numbers are semantic, CPU and tiny on purpose — the card is not the
question here.
"""

from __future__ import annotations

import dataclasses as dc
import hashlib

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg, write_corpus

from dormouse.config import RunConfig, TrainConfig
from dormouse.instrument import Counters, LoudError
from dormouse.losses import (
    CLIFF,
    U_MAX,
    ce_margin,
    cliff,
    fixed_cross_entropy,
    fixed_soft_ce,
)
from dormouse.model import DormouseModel
from dormouse.train.loop import train

SEQ, BATCH = 64, 2

#: Captured on `main` (`3f83788`), BEFORE `losses.py` existed:
#:   * `_model_pin()` with the flag absent: sha256 over `model.state_dict()`
#:     (sorted keys, raw bytes) after 5 AdamW steps of `lr=1e-3` on
#:     `ids = arange(b*t) % 251`, `targets = (ids + 1) % 251`,
#:     `torch.manual_seed(0)`; sha256 over the 5 forwards' concatenated logits;
#:   * `_train_pin()`: 5 `train()` steps of `tiny_run`-shaped config
#:     (`steps=5, lr=0.01, log_every=5`) — sha256 over the state dict, plus the
#:     step-5 loss field verbatim.
MAIN_MODEL_SD = "b8e610d715ec86d59b229e61c173998813ac40a34867dc82ccc90dc1b1682694"
MAIN_MODEL_LOGITS = "642106fbb7cbc5c354c45d8edd5490ca1d73201b9a3bc1a9d128a50c90354128"
#: Re-pinned 2026-11-11 (Muon's small-side Newton–Schulz): only the TRAIN pin
#: moved — `_model_pin` is AdamW-only. The step-5 loss is unchanged (`2.7103`).
MAIN_TRAIN_SD = "62cf3843ac570a67c089ef8a7a0f19529ac5e45cbd9e7ef0f71c1bf98a6bf988"
MAIN_TRAIN_LOSS = "2.7103"


def _sha_state(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _model_pin(**kw) -> tuple[str, str, float]:
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    x, y = ids, (ids + 1) % 251
    seen = []
    loss = None
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        logits, loss, _c = model(x, y, **kw)
        loss.backward()
        opt.step()
        seen.append(logits.detach().clone())
    return _sha_state(model.state_dict()), _sha_tensor(torch.cat(seen)), float(loss.detach())


def _train_pin(tmp_path, **kw) -> tuple[str, str, list[str]]:
    torch.manual_seed(0)
    cfg = TrainConfig(
        steps=5,
        batch=BATCH,
        seq_len=SEQ,
        lr=0.01,
        log_every=5,
        ckpt_every=1000,
        ckpt_name="p",
        out_dir=str(tmp_path / "ck"),
        data=str(write_corpus(tmp_path / "corpus.bin")),
        **kw,
    )
    run = RunConfig(source="pin", model=tiny_cfg(), train=cfg)
    lines: list[str] = []
    model = DormouseModel(run.model)
    report = train(model, run, log=lines.append)
    loss_field = f"{report['loss']:.4f}"
    return _sha_state(model.state_dict()), loss_field, lines


# --------------------------------------------------------------------------- a


@pytest.mark.pins
def test_a_flag_off_is_bit_identical_to_main(tmp_path):
    """The pin AND the call: absent is `false`, and `false` moves no bit."""
    sd, logits, _ = _model_pin()
    assert (sd, logits) == (MAIN_MODEL_SD, MAIN_MODEL_LOGITS)
    assert _model_pin(ce_fixed=False)[:2] == (MAIN_MODEL_SD, MAIN_MODEL_LOGITS)
    tsd, tloss, lines = _train_pin(tmp_path)
    assert (tsd, tloss) == (MAIN_TRAIN_SD, MAIN_TRAIN_LOSS)
    assert "ce_cliff" not in lines[-1], "a control run's log line must not grow a field"


def _byteflow_tiny():
    return tiny_cfg(
        use_byteflow=True,
        bf16=True,
        use_engram=False,
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


def test_a_flag_off_never_reaches_the_fixed_loss_at_any_site(tmp_path):
    """The branch, not only the bits: OFF is not a "call and discard".

    Every call site is spied at its own module attribute (the names are
    imported, so patching `dormouse.losses` alone would prove nothing): the
    loop model's per-iteration + final CE, the byteflow shapes' `_byte_ce`, SFT
    and RL. Then the same seams are run with the flag ON and the spies must
    fire — a flag a `--set` accepts but a loop ignores is the SILENT class.
    """
    import dataclasses as dc

    import dormouse.model.loopblock as lb_mod
    import dormouse.model.model as model_mod
    import dormouse.post.rl as rl_mod
    import dormouse.train.sft as sft_mod
    from dormouse.train.sft import build_sft_corpus, sft_train

    fired: list[str] = []

    def spy(*a, **k):
        fired.append("hit")
        raise AssertionError("fixed_cross_entropy ran with ce_fixed=false")

    names = [
        (model_mod, "fixed_cross_entropy"),
        (lb_mod, "fixed_cross_entropy"),
        (sft_mod, "fixed_cross_entropy"),
        (rl_mod, "fixed_cross_entropy"),
    ]
    real = {(m, n): getattr(m, n) for m, n in names}
    real_any = real[names[0]]
    for m, n in names:
        setattr(m, n, spy)

    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    t = (ids + 1) % 251
    try:
        # 1. the loop model: per-iteration CE + the final CE
        m = DormouseModel(tiny_cfg())
        _, loss, _c = m(ids, t)
        loss.backward()
        # 2. the byteflow shapes (`_byte_ce`)
        bf = DormouseModel(_byteflow_tiny())
        _, bf_loss, _c = bf(ids, t)
        assert bf_loss is not None
        # 3. SFT
        data = tmp_path / "sft.bin"
        build_sft_corpus(data, seed=1, seq_len=SEQ, n_tasks=32)
        cfg = TrainConfig(
            steps=1,
            batch=BATCH,
            seq_len=SEQ,
            lr=0.01,
            log_every=1,
            ckpt_every=1000,
            ckpt_name="s",
            out_dir=str(tmp_path / "ck"),
            data=str(data),
        )
        run = RunConfig(source="spy", model=tiny_cfg(), train=cfg)
        sft_train(DormouseModel(run.model), run, log=lambda *_: None)
        # 4. RL's log-probability seam
        rl_mod.response_logp(torch.randn(2 * 8, 256), torch.arange(16).view(2, 8) % 256, 3)
        assert not fired

        # ... and ON: each seam calls it (the spy records, then gets out of the
        # way by delegating to the real function).
        for m, n in names:
            setattr(m, n, lambda *a, **k: (fired.append("hit"), real_any(*a, **k))[1])
        DormouseModel(tiny_cfg())(ids, t, ce_fixed=True)
        assert fired, "the loop model ignored ce_fixed"
        fired.clear()
        DormouseModel(_byteflow_tiny())(ids, t, ce_fixed=True)
        assert fired, "the byteflow shapes ignored ce_fixed"
        fired.clear()
        sft_train(
            DormouseModel(run.model),
            dc.replace(run, train=dc.replace(cfg, ce_fixed=True)),
            log=lambda *_: None,
        )
        assert fired, "sft_train ignored ce_fixed"
        fired.clear()
        rl_mod.response_logp(torch.randn(16, 256), torch.arange(16).view(2, 8) % 256, 3, fixed=True)
        assert fired, "response_logp ignored fixed"
    finally:
        for m, n in names:
            setattr(m, n, real[(m, n)])


# --------------------------------------------------------------------------- b


def _cliff_logits(dtype, margin, k=8) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.full((1, k), -margin / 2, dtype=dtype)
    z[0, 0] = margin / 2  # the target IS the argmax: "confidently correct"
    return z, torch.zeros(1, dtype=torch.long)


def _grad_pair(z, fixed: bool, z32: torch.Tensor | None = None):
    t = torch.zeros(1, dtype=torch.long)
    if fixed:
        x = z.clone().requires_grad_(True)
        fixed_cross_entropy(x, t).backward()
        return x.grad, None
    x = z.clone().requires_grad_(True)
    F.cross_entropy(x, t).backward()
    # The fixed path's own arithmetic (fp32) on the SAME values, for the rows
    # whose storage dtype cannot hold the identity (see the table's note).
    x32 = (z.float() if z32 is None else z32).clone().requires_grad_(True)
    fixed_cross_entropy(x32, t).backward()
    return x.grad, x32.grad


def test_b_collapse_and_fix(capsys):
    """The two rows of the brief: (bf16, 8) and (fp32, 25)."""
    rows = []
    for dtype, margin in ((torch.bfloat16, 8.0), (torch.float32, 25.0)):
        z, t = _cliff_logits(dtype, margin)
        g_std, g_fix32 = _grad_pair(z, fixed=False)
        g_fix, _ = _grad_pair(z, fixed=True)
        assert g_std[0, 0].item() == 0.0, "the standard path must zero the target entry"
        assert abs(float(g_std.sum())) / float(g_std.abs().max()) > 1e-2, (
            "and the gradient must fail to sum to zero by a visible margin"
        )
        assert g_fix[0, 0].item() != 0.0, "the fix restores the entry the cliff zeroed"
        others = float(g_fix32[0, 1:].sum())
        assert abs(float(g_fix32[0, 0]) + others) <= 1e-6 * float(g_fix32.abs().max())
        assert abs(float(g_fix32.sum())) <= 1e-6
        for name, g in (("standard", g_std), ("fixed", g_fix), ("fixed fp32", g_fix32)):
            mx = float(g.abs().max())
            rows.append(
                f"{str(dtype):15s} m={margin:<5g} {name:11s} g_t={float(g[0, 0]):+12.3e} "
                f"sum(others)={float(g[0, 1:].sum()):+12.3e} "
                f"g_t+others={float(g[0, 0]) + float(g[0, 1:].sum()):+9.2e} "
                f"sum={float(g.sum()):+10.2e} |sum|/max={abs(float(g.sum())) / mx if mx else 0:.2e}"
            )
    print("\n".join(rows))
    # The bf16 row's note is printed with the table: the RETURNED gradient is
    # bf16 (autograd casts the fp32 fix to the input dtype), so its zero-sum
    # property is bounded by bf16's ulp, not by the fix — the fp32 interior of
    # the same values is the line that holds to 1e-6.
    assert "fixed fp32" in "\n".join(rows)


# --------------------------------------------------------------------------- c


def _fp64_ce(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    z64 = z.double()
    return -torch.log_softmax(z64, -1).gather(1, t.reshape(-1, 1)).reshape(-1)


def test_c_below_the_cliff_agrees_with_fp64(capsys):
    """Margins 0, 1, 5, 15: the fix is the SAME objective, to fp64 round-off.

    The standard path is compared too, and it is the one that drifts: at margin
    15 its `1 + e^{-m}` quantises the loss to fp32's ulp of 1.0 (2e-3 relative),
    which is why the brief's "fixed ≈ standard to 1e-6" cannot be a gate at 15 —
    the reference that means something is fp64, and both are held to it.
    """
    rows, bad = [], []
    for margin in (0.0, 1.0, 5.0, 15.0):
        z, t = _cliff_logits(torch.float32, margin)
        ref = _fp64_ce(z, t)
        zf = z.clone().requires_grad_(True)
        lf = fixed_cross_entropy(zf, t)
        lf.backward()
        zs = z.clone().requires_grad_(True)
        ls = F.cross_entropy(zs, t)
        ls.backward()
        gref = (torch.softmax(z.double(), -1) - F.one_hot(t, z.shape[-1]).double())[0]
        l_rel_fix = abs(float(lf.detach()) - float(ref)) / abs(float(ref))
        l_rel_std = abs(float(ls.detach()) - float(ref)) / abs(float(ref))
        g_rel_fix = float((zf.grad[0].double() - gref).norm() / gref.norm())
        g_rel_std = float((zs.grad[0].double() - gref).norm() / gref.norm())
        rows.append(
            f"m={margin:<4g} loss fp64={float(ref):.6e} fixed rel={l_rel_fix:.2e} "
            f"standard rel={l_rel_std:.2e} | grad fixed rel={g_rel_fix:.2e} "
            f"standard rel={g_rel_std:.2e} | fixed==standard loss rel="
            f"{abs(float(lf.detach()) - float(ls.detach())) / abs(float(ref)):.2e}"
        )
        if l_rel_fix > 1e-6 or g_rel_fix > 1e-5:
            bad.append((margin, l_rel_fix, g_rel_fix))
    print("\n".join(rows))
    assert not bad, f"the fix is not fp64-accurate at {bad}"
    # Below the point where the standard path's own 1+x rounding bites, the two
    # are the same number: this is the "exact-class-compatible" half.
    for margin in (0.0, 1.0):
        z, t = _cliff_logits(torch.float32, margin)
        a = fixed_cross_entropy(z, t)
        b = F.cross_entropy(z, t)
        assert abs(float(a) - float(b)) <= 1e-6 * abs(float(b))


# --------------------------------------------------------------------------- d


def test_d_symmetric_loss_never_overflows(capsys):
    """argmax-target and argmin-target, margins to 1e4: fp64 closed form, no inf."""
    rows = []
    for margin in (0.0, 1.0, 10.0, 100.0, 1e3, 1e4):
        for sign in (+1, -1):
            # The target is the argmax (sign +1: loss -> 0) or the argmin
            # (sign -1: loss -> margin + log s): the SAME code path both ways.
            z = torch.full((1, 8), 0.0, dtype=torch.float32)
            z[0, 1:] = torch.linspace(-3.0, 3.0, 7)
            z[0, 0] = sign * margin
            t = torch.zeros(1, dtype=torch.long)
            f = z.clone().requires_grad_(True)
            loss = fixed_cross_entropy(f, t)
            loss.backward()
            ref = float(_fp64_ce(z, t))
            got = float(loss.detach())
            ok = abs(got - ref) <= 1e-6 * max(1.0, abs(ref))
            finite = torch.isfinite(f.grad).all().item() and torch.isfinite(loss).item()
            rows.append(
                f"m={sign * margin:<9g} fixed={got:.9g} fp64={ref:.9g} ok={ok} "
                f"finite={finite} sum(g)={float(f.grad.sum()):.3g}"
            )
            assert finite, rows[-1]
            assert ok, rows[-1]
            assert abs(float(f.grad.sum())) <= 1e-6 * max(1e-30, float(f.grad.abs().max()))
    print("\n".join(rows))


# --------------------------------------------------------------------------- e


def test_e_distillation_identical_logits_give_exactly_zero(capsys):
    """`student == teacher`: KL 0 and gradient 0 EXACTLY; the standard path drifts."""
    torch.manual_seed(0)
    teacher = torch.randn(16, 32, dtype=torch.bfloat16)
    student = teacher.clone().requires_grad_(True)
    counters = Counters()
    kl = fixed_soft_ce(student, teacher, 1.0, counters=counters)
    kl.backward()
    assert kl.item() == 0.0, "KL at z_S == z_T must be exactly 0"
    assert torch.equal(student.grad, torch.zeros_like(student.grad)), "and so must g"
    # The standard path: two separately-rounded softmaxes subtracted.
    std = teacher.clone().requires_grad_(True)
    ref = -(torch.softmax(teacher.float(), -1) * F.log_softmax(std, -1)).sum(-1).mean()
    ref.backward()
    grad = std.grad
    print(
        f"fixed: kl={kl.item():.6g} |g|={float(student.grad.norm()):.6g} "
        f"exact_zero={torch.equal(student.grad, torch.zeros_like(student.grad))} | "
        f"standard: kl={float(ref.detach()):.6g} |g|={float(grad.norm()):.6g} "
        f"max={float(grad.abs().max()):.6g}"
    )
    assert float(grad.norm()) > 0.0, "the cancellation floor must be visible"
    # Near-identical fp32 logits: the gradient matches fp64's `q - p` where the
    # standard path cannot (printed), and the loss is positive.
    torch.manual_seed(1)
    t64 = torch.randn(8, 16, dtype=torch.float64)
    s32 = (t64 + 1e-3 * torch.randn(8, 16, dtype=torch.float64)).float()
    ref64 = torch.softmax(s32.double(), -1) - torch.softmax(t64.float().double(), -1)
    z = s32.clone().requires_grad_(True)
    loss = fixed_soft_ce(z, t64.float(), reduction="none").sum()
    loss.backward()
    rel = float((z.grad.double() - ref64).norm() / ref64.norm())
    stdg = s32.clone().requires_grad_(True)
    slog = -(torch.softmax(t64.float(), -1) * F.log_softmax(stdg, -1)).sum()
    slog.backward()
    rel_std = float((stdg.grad.double() - ref64).norm() / ref64.norm())
    print(f"near-identical fp32: fixed rel={rel:.2e} standard rel={rel_std:.2e}")
    assert rel < 1e-5 and rel_std > rel


# --------------------------------------------------------------------------- f


def test_f_the_cliff_counter_counts_exactly_the_cliff_tokens():
    """A known set of rows above/below the cliff: the counter is that set."""
    torch.manual_seed(0)
    z = torch.randn(12, 64, dtype=torch.float32)
    t = torch.randint(0, 64, (12,), generator=torch.Generator().manual_seed(1))
    # Craft: rows 0-4 confident (target 30 above its own row's max), 5-11 not.
    for r in range(5):
        z[r] = z[r] - z[r].max()
        z[r, t[r]] = 30.0
    margin = ce_margin(z, t)
    expected = int((margin > cliff(z.dtype)).sum())
    assert expected == 5, margin.tolist()
    c = Counters()
    fixed_cross_entropy(z, t, counters=c)
    assert "ce_cliff_tokens" not in c, "the counter must stay on the device lane"
    assert "ce_cliff_tokens" in c._device, "`count_device`, never a host int"
    drained = c.drain()
    assert c["ce_cliff_tokens"] == expected
    assert drained["ce_clamp_hits"] == 0
    # bf16 logits count against the bf16 cliff (the standard path's cliff).
    assert cliff(torch.bfloat16) == CLIFF[torch.bfloat16] == 5.5
    assert cliff(torch.float32) == 16.6 and CLIFF[torch.float64] == 36.7


def test_f_the_fixed_forward_backward_touches_no_host(monkeypatch):
    """No `.item()`/`.tolist()`/`bool()`/`int()`/`float()` in fwd OR bwd."""

    def boom(*_a, **_k):
        raise AssertionError("host sync inside the fixed CE (ADR-0018 rule 2)")

    # `moe_topk=0`: the CPU MoE's `_naive_dispatch` `.tolist()`s its expert ids
    # BY DESIGN (it is the slow correctness path, `experts.py:1433`), which is
    # not this path's read — the same reason `test_adaptive_topk.py` forces the
    # padded arm for its gate.
    model = DormouseModel(tiny_cfg(moe_topk=0))
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    c = Counters()
    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "tolist", boom)
    monkeypatch.setattr(torch.Tensor, "__bool__", boom)
    monkeypatch.setattr(torch.Tensor, "__int__", boom)
    monkeypatch.setattr(torch.Tensor, "__float__", boom)
    logits, loss, c = model(ids, (ids + 1) % 251, counters=c, ce_fixed=True)
    # `torch.equal` returns a Python bool: it does not route through the
    # patched `Tensor.__bool__` (the same trick `test_adaptive_topk.py` uses).
    assert torch.equal(torch.isfinite(loss), torch.ones((), dtype=torch.bool))
    loss.backward()
    assert torch.equal(torch.isfinite(logits), torch.ones_like(logits, dtype=torch.bool))
    assert "ce_cliff_tokens" in c._device and "ce_cliff_tokens" not in c.line()


# --------------------------------------------------------------------------- g


def test_g_refusals_are_loud():
    z = torch.zeros(4, 8)
    t = torch.zeros(4, dtype=torch.long)
    with pytest.raises(LoudError, match="reduction"):
        fixed_cross_entropy(z, t, "median")
    with pytest.raises(LoudError, match="2-D"):
        fixed_cross_entropy(z[0], t[:1])
    with pytest.raises(LoudError, match="targets for"):
        fixed_cross_entropy(z, t[:3])
    with pytest.raises(LoudError, match="temperature"):
        fixed_soft_ce(z, z, 0.0)
    with pytest.raises(LoudError, match="both be"):
        fixed_soft_ce(z, z[:2])
    with pytest.raises(LoudError, match="cliff"):
        cliff(torch.uint8)


def test_g_a_clamp_that_fires_is_counted():
    """A target 1e4 BELOW the max: the exponent leaves fp32's `exp` band."""
    z = torch.tensor([[-1e4, 0.0, -1.0]])
    t = torch.zeros(1, dtype=torch.long)
    c = Counters()
    loss = fixed_cross_entropy(z, t, counters=c)
    drained = c.drain()
    assert drained["ce_clamp_hits"] == 1, "the clamp must never be silent"
    assert drained["ce_cliff_tokens"] == 0
    # u = 1e4 > U_MAX: the loss is the equal branch `u + log s`, to fp64.
    assert abs(float(loss) - float(_fp64_ce(z, t))) <= 1e-6 * float(_fp64_ce(z, t))
    assert U_MAX == 80.0
    # The other direction (confidently correct): u underflows; the loss is 0.
    z2 = torch.tensor([[1e4, 0.0, -1.0]])
    c2 = Counters()
    assert float(fixed_cross_entropy(z2, t, counters=c2)) == 0.0
    c2.drain()
    assert c2["ce_clamp_hits"] == 1


def test_g_fp8_is_the_gemm_mode_not_the_logits_dtype():
    """Decision: no refusal. `mm_dtype` names the 2-D GEMM's compute mode.

    The loss reads the HEAD's logits, and the head runs fp32 on every path
    (`model.py`: "fp32 logits — the burn rule that fixes `bf16 logits NaN`"),
    so an fp8 GEMM lane cannot move the dtype the cliff is read from. Pinned by
    construction: the pair resolves and the cliff stays fp32's.
    """
    run = RunConfig(
        source="fp8",
        model=tiny_cfg(),
        train=TrainConfig(mm_dtype="fp8", ce_fixed=True),
    )
    assert run.train.ce_fixed and run.train.mm_dtype == "fp8"
    assert cliff(torch.float32) == 16.6


# --------------------------------------------------------------------------- wiring


def test_the_rl_train_loop_passes_the_flag_to_its_step(monkeypatch, tmp_path):
    """`rl_train`'s step gets `fixed=True` iff the flag is set (its own seam)."""
    import sys

    rl_train_mod = sys.modules["dormouse.post.rl_train"]

    got: dict = {}

    def fake_step(model, opt, samples, advantages, cfg, *, counters=None, **kw):
        got.update(kw)
        counters = Counters() if counters is None else counters
        counters.count("rl_steps")
        loss = sum(p.float().sum() for p in model.parameters()) * 0.0
        loss = loss.clone().requires_grad_(True)
        return loss, counters

    monkeypatch.setattr(rl_train_mod, "rl_step", fake_step)
    cfg = TrainConfig(
        steps=1,
        batch=2,
        seq_len=64,
        lr=0.01,
        log_every=1,
        ckpt_every=1000,
        ckpt_name="r",
        out_dir=str(tmp_path / "ck"),
        rl_task_cls="math",
        rl_tasks=2,
        rl_group=2,
        rl_budget=96,
        rl_holdout=1,
    )
    run = RunConfig(source="rl", model=tiny_cfg(max_seq_len=128), train=cfg)
    rl_train_mod.rl_train(run, log=lambda *_: None)
    assert got == {}, f"ce_fixed=false must pass no kwarg, got {got}"
    got.clear()
    run = RunConfig(
        source="rl", model=tiny_cfg(max_seq_len=128), train=dc.replace(cfg, ce_fixed=True)
    )
    rl_train_mod.rl_train(run, log=lambda *_: None)
    assert got == {"fixed": True}, f"rl_train dropped ce_fixed: {got}"


def test_the_rl_log_probability_seam_matches_and_sums_to_zero():
    """`response_logp(fixed=True)` is the same VALUE as the standard gather.

    RL is a policy gradient, so the number that matters is the log-probability
    of the SAMPLED byte; the fix may not move it, and its gradient must be
    `q - onehot` with the sampled entry exactly `-(sum of the others)`.
    """
    from dormouse.instrument import Counters
    from dormouse.post.rl import response_logp

    torch.manual_seed(0)
    logits = torch.randn(2 * 16, 256)
    ids = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(1))
    lp = response_logp(logits, ids, 4)
    lpf = response_logp(logits, ids, 4, fixed=True)
    assert torch.allclose(lp, lpf, atol=1e-6, rtol=1e-6), (lp - lpf).abs().max()
    # The gradient, with the sampled byte's row summed so each row's grad is
    # a distribution minus one: sum over v must be 0.
    z = logits.clone().requires_grad_(True)
    response_logp(z, ids, 4, fixed=True).sum().backward()
    per_row = z.grad.reshape(2, 16, 256)[:, 1:]
    c = Counters()
    response_logp(z.detach(), ids, 4, fixed=True, counters=c)
    drained = c.drain()
    assert drained["ce_cliff_tokens"] == 0 and drained["ce_clamp_hits"] == 0
    assert float(per_row.sum(-1).abs().max()) <= 1e-6


def test_the_firewalls_do_not_swallow_a_non_finite_logit():
    """inf must reach the trainer's NaN firewall, not read as a perfect prediction.

    Two blind spots found by the numerics audit (2026-11-08):
    * `fixed_cross_entropy` returned exactly 0.0 for a +inf TARGET logit
      (`u = -inf` takes the `u < U_MIN` branch);
    * `logit_softcap` mapped any inf to a finite `+cap` with a zero gradient.
    Both are one-line checks away, and both were dormant — which is why nobody
    noticed. This gate is what makes them not dormant.
    """
    import torch

    from dormouse.losses import fixed_cross_entropy, logit_softcap

    # softcap: a finite logit is unchanged, an inf survives
    z = torch.tensor([[1.0, -2.0, 0.5]])
    assert torch.allclose(logit_softcap(z, 15.0), 15.0 * torch.tanh(z / 15.0))
    bad = torch.tensor([[1.0, float("inf"), 0.5]])
    out = logit_softcap(bad, 15.0)
    assert not torch.isfinite(out[0, 1]), out
    assert torch.equal(out[0, [0, 2]], logit_softcap(z, 15.0)[0, [0, 2]])

    # fixed CE: a +inf target logit must not come back as 0.0
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    targets = torch.tensor([2])
    ok = fixed_cross_entropy(logits, targets, reduction="none")
    assert torch.isfinite(ok).all()
    logits[0, 2] = float("inf")
    bad_loss = fixed_cross_entropy(logits, targets, reduction="none")
    assert not torch.isfinite(bad_loss).all(), bad_loss
