"""JEPA gates: the exact loss, the detach, no host sync, the collapse gate, refusals.

`model/jepa.py` is ONE mechanism — a causal latent predictor (`h[p] -> h[p+h]`,
stop-grad target, cosine) plus SIGReg on the PREDICTED latents, with its own
weight inside the term. These gates pin each half and each seam:

1. `jepa_weight = 0` is the pre-change model bit for bit (the MTP goldens, which
   were measured before either head existed).
2. The term is `(1 − cos(pred, sg(target))) + jepa_reg_weight · sigreg(pred)`
   recomputed line by line, with the horizon pinned to the right shift.
3. Every predictor parameter gets a finite non-zero gradient; the target is
   DETACHED (the target positions receive no gradient; attaching the target's
   graph leaves the predictor's input gradient unchanged).
4. The forward touches no host (`.item()` / `.tolist()` / `float(tensor)` raise).
5. THE COLLAPSE GATE: 200 steps, tiny model, with and without the internal reg.
   Measured, not assumed — see the test's docstring for both numbers and for the
   one place the measured failure mode differs from "collapses to a constant".
6. Every refusal: the switch/weight split, negative weights, a bad horizon, the
   horizon against `seq_len`, `segment_len`, and `use_byteflow`.

The last test is the A/B driver's pure verdict logic (`scripts/ab_jepa.py`),
which needs no GPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg, tiny_run

from dormouse.config import ModelConfig, load_preset, load_run, parse_preset, validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.jepa import JepaHead
from dormouse.model.sigreg import sigreg_loss, stride_sample
from dormouse.train import train

ROOT = Path(__file__).resolve().parents[1]

#: Gate 1's literals: `tests/test_mtp.py` measured them on the tree BEFORE the
#: JEPA arm existed (seed 1234, `tiny_cfg()`, the batch below). An off arm that
#: moves any of them is not bit-identical.
GOLDEN_PARAMS = 64_881
GOLDEN_LOGITS_SHA256 = "c7839283a79291cf713aab57476cf202ba1a68167c365197f785da08203055c2"
GOLDEN_LOSS = 6.119609355926514

B, T = 2, 64


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = (torch.arange(B * T, dtype=torch.long) % 251).reshape(B, T)
    return ids, (ids + 7) % 256


@pytest.mark.local_pin
def test_jepa_off_is_the_pre_change_model_bit_for_bit() -> None:
    """Gate 1: `jepa_weight = 0` builds no head and moves no number."""
    torch.manual_seed(1234)
    plain = DormouseModel(tiny_cfg())
    torch.manual_seed(1234)
    off = DormouseModel(tiny_cfg(jepa=False, jepa_weight=0.0))
    assert plain.jepa is None and off.jepa is None
    assert param_count(off) == GOLDEN_PARAMS
    ids, targets = _fixed_batch()
    logits, loss, _ = off(ids, targets)
    assert hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest() == GOLDEN_LOGITS_SHA256
    assert float(loss.detach()) == pytest.approx(GOLDEN_LOSS, abs=1e-6)
    logits2, loss2, _ = plain(ids, targets)
    assert torch.equal(logits, logits2) and torch.equal(loss, loss2)


def test_the_term_is_cosine_plus_the_internal_weighted_sigreg() -> None:
    """Gate 2: both terms, their weights, and the horizon's shift, recomputed.

    The hand formula is the module docstring's, with `d = 8`, `horizon = 2`,
    `reg_weight = 0.25`: `pred = up(silu(down(h[:, :-2])))`,
    `target = h[:, 2:].detach()`, term = mean(1 − cos) + 0.25 · SIGReg(pred).
    """
    torch.manual_seed(0)
    head = JepaHead(8, horizon=2, reg_weight=0.25)
    h = torch.randn(3, 10, 8)
    term = head(h)
    pred = head.up(F.silu(head.down(h[:, :-2])))
    want = (1.0 - F.cosine_similarity(pred, h[:, 2:].detach(), dim=-1)).mean()
    want = want + 0.25 * sigreg_loss(stride_sample(pred.reshape(-1, 8)))
    assert float(term.detach()) == pytest.approx(float(want), abs=1e-6)
    # The shift is h[p+h], not h[p+1] or h[p]: a horizon-1 recomputation is a
    # different number on this fixture (the gate is the shift, not the shape).
    wrong = (1.0 - F.cosine_similarity(pred, h[:, 1:-1].detach(), dim=-1)).mean()
    wrong = wrong + 0.25 * sigreg_loss(stride_sample(pred.reshape(-1, 8)))
    assert abs(float(term) - float(wrong)) > 1e-4
    # The internal weight is real: reg_weight=0 drops exactly the second term.
    zero = JepaHead(8, horizon=2, reg_weight=0.0)
    with torch.no_grad():
        zero.down.weight.copy_(head.down.weight)
        zero.down.bias.copy_(head.down.bias)
        zero.up.weight.copy_(head.up.weight)
        zero.up.bias.copy_(head.up.bias)
        cosine_only = (1.0 - F.cosine_similarity(pred, h[:, 2:].detach(), dim=-1)).mean()
    assert float(zero(h)) == pytest.approx(float(cosine_only), abs=1e-6)


def test_predictor_grads_are_nonzero_and_the_target_is_detached() -> None:
    """Gate 3: finite non-zero grads on every predictor param; target stop-grad.

    Two readings of the detach. (a) The module's own forward sends ZERO gradient
    to the target positions `h[:, h:]` — a missing `.detach()` makes that sum
    non-zero. (b) Recomputing the SAME term with the target's graph ATTACHED
    leaves the predictor's input gradient `h[:, :-h]` unchanged, while the target
    positions now receive gradient: that is exactly what `detach()` buys (the
    target values steer the prediction; they never train the backbone).
    """
    torch.manual_seed(0)
    # A gap-wide horizon (3 of 6) makes the predictor's input `h[:, :3]` and
    # the target `h[:, 3:]` DISJOINT, so "no gradient at the target positions"
    # is a clean reading: at horizon 1 the slices overlap and every interior
    # position is legitimately both an input and a target.
    head = JepaHead(8, horizon=3, reg_weight=0.1)
    h = torch.randn(2, 6, 8, requires_grad=True)
    head(h).backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert torch.count_nonzero(p.grad).item() > 0, f"{name} got an all-zero gradient"
    assert torch.count_nonzero(h.grad[:, 3:]).item() == 0, "gradient reached the target"

    # (b) The same forward with a NON-detached target, on a fresh graph and the
    # SAME predictor (so the two readings differ only in the target's graph).
    h2 = h.detach().clone().requires_grad_(True)
    pred = head.up(F.silu(head.down(h2[:, :3])))
    attached = (1.0 - F.cosine_similarity(pred, h2[:, 3:], dim=-1)).mean()
    attached = attached + 0.1 * sigreg_loss(stride_sample(pred.reshape(-1, 8)))
    attached.backward()
    # The predictor's input gradient is the same as the detached module's...
    assert torch.equal(h.grad[:, :3], h2.grad[:, :3])
    # ...and the target is the only place the attach ADDS gradient.
    assert torch.count_nonzero(h2.grad[:, 3:]).item() > 0


def test_the_forward_touches_no_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Gate 4: no `.item()` / `.tolist()` / `float(tensor)` in the head or seam."""
    torch.manual_seed(0)
    head = JepaHead(8, horizon=1, reg_weight=0.1)
    h = torch.randn(2, 6, 8)

    def boom(*_a, **_k):
        raise AssertionError("host sync inside the JEPA forward (ADR-0018 rule 2)")

    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "tolist", boom)
    monkeypatch.setattr(torch.Tensor, "__float__", boom)
    term = head(h)
    assert term.requires_grad and term.ndim == 0

    # The model seam adds no read of its own. The dense blend (`moe_topk=0`)
    # keeps the unrelated CPU expert fallback — whose `unique(...).tolist()` is
    # not JEPA's — out of the forward, so the same three guards apply.
    run = tiny_run(tmp_path)
    cfg = dataclasses.replace(run.model, jepa=True, jepa_weight=0.1, moe_topk=0)
    model = DormouseModel(cfg)
    ids, targets = _fixed_batch()
    _, loss, counters = model(ids, targets)
    assert counters.get("jepa") == 1 and loss is not None


def _train_jepa(
    tmp_path: Path, reg_weight: float, *, steps: int = 200
) -> tuple[DormouseModel, dict, float, float]:
    """One 200-step tiny run at a given internal reg weight, seed 1."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    run = tiny_run(tmp_path, steps=steps, lr=0.01, log_every=steps)
    model_cfg = dataclasses.replace(
        run.model, jepa=True, jepa_weight=0.1, jepa_reg_weight=reg_weight, jepa_horizon=1
    )
    run = dataclasses.replace(run, model=model_cfg)
    torch.manual_seed(1)
    model = DormouseModel(run.model)
    report = train(model, run, log=lambda *_: None)
    z = model.jepa.last_pred.reshape(-1, run.model.d_model).detach().float()
    raw = float(z.std(dim=0).mean())
    unit = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return model, report, raw, float(unit.std(dim=0).mean())


def test_collapse_gate_the_internal_reg_is_load_bearing(tmp_path: Path) -> None:
    """Gate 5: 200 steps, tiny model, reg 0.1 vs 0.0 — the measured numbers.

    **Measured on this tree (seed 1, 200 steps, `jepa_weight=0.1`):**

    * `jepa_reg_weight = 0.1` — mean per-dim std of the predicted latents
      **0.0612** (seed 2: 0.0708), final train loss 0.0157, 0 NaN steps.
    * `jepa_reg_weight = 0.0` — mean per-dim std **29 350.7** (seed 2: 140 413),
      final train loss 0.0014: the scale-free cosine objective drives the
      latent's SCALE to infinity while the loss goes to ~0.

    One honest deviation from the brief, because the measurement says so: a
    cosine loss is invariant to the prediction's scale, so the failure mode of
    the no-reg arm is not a collapse to a POINT — its per-dim std (normalized
    rows) stays ~0.118 in both arms. It is a collapse into a DEGENERATE,
    unbounded latent, which is the documented JEPA collapse chain's
    "target-variance rise" step (`docs/research/...-jepa-into-dormouse.md` §3's
    2607.23531 order), not the final cosine-collapse step. The gate therefore
    has a TWO-SIDED band: above the floor (not a constant) and below the
    ceiling (not diverged). The no-reg arm misses the ceiling by 4-5 orders of
    magnitude, which is the falsifier — delete `reg_weight` from the term and
    this test goes red.
    """
    _m, reg_report, reg_std, reg_unit_std = _train_jepa(tmp_path / "reg", 0.1)
    _m2, bad_report, bad_std, bad_unit_std = _train_jepa(tmp_path / "noreg", 0.0)
    print(
        f"JEPA collapse gate: reg=0.1 std={reg_std:.6f} unit_std={reg_unit_std:.5f} "
        f"loss={reg_report['loss']:.5f} | reg=0.0 std={bad_std:.2f} "
        f"unit_std={bad_unit_std:.5f} loss={bad_report['loss']:.5f}"
    )
    assert reg_report["nan_skipped"] == 0 and bad_report["nan_skipped"] == 0
    # The arm under test stays spread...
    assert reg_std > 0.02, f"the reg arm collapsed: std={reg_std}"
    assert reg_std < 1.0, f"the reg arm diverged: std={reg_std}"
    assert reg_unit_std > 0.05, "the predicted directions are a point mass"
    # ...and removing the internal reg collapses it into the degenerate mode.
    assert bad_std > 100.0, f"without the reg the latent stayed healthy: std={bad_std}"


def test_the_switch_the_weight_and_the_horizon_are_refused_loudly() -> None:
    """Gate 6: every misconfiguration names the key and the escape."""
    with pytest.raises(LoudError, match="jepa"):
        ModelConfig(jepa=True, jepa_weight=0.0)
    with pytest.raises(LoudError, match="jepa_weight"):
        ModelConfig(jepa=False, jepa_weight=0.1)
    with pytest.raises(LoudError, match="jepa_reg_weight"):
        validate(ModelConfig(jepa=True, jepa_weight=0.1, jepa_reg_weight=-0.1))
    with pytest.raises(LoudError, match="jepa_horizon"):
        validate(ModelConfig(jepa=True, jepa_weight=0.1, jepa_horizon=0))
    with pytest.raises(LoudError, match="jepa_horizon"):
        load_run(
            "small",
            model_overrides={"jepa": True, "jepa_weight": 0.1, "jepa_horizon": 65},
            train_overrides={"seq_len": 512},
        )
    with pytest.raises(LoudError, match="jepa_horizon"):
        parse_preset(
            "d_model = 64\nn_heads = 4\nhead_dim = 16\njepa = true\n"
            "jepa_weight = 0.1\njepa_horizon = 0\n"
        )
    # The segment refusal is the horizon crossing a cut; 256 divides 512, so
    # only the JEPA rule can fire here.
    with pytest.raises(LoudError, match="segment_len"):
        load_run(
            "small",
            model_overrides={"jepa": True, "jepa_weight": 0.1},
            train_overrides={"segment_len": 256},
        )
    # The byteflow channel has no byte positions for the horizon to shift.
    with pytest.raises(LoudError, match="use_byteflow"):
        parse_preset(
            "d_model = 64\nn_heads = 4\nhead_dim = 16\nuse_byteflow = true\n"
            "byteflow_k_tokens = 8\nbf16 = false\nuse_engram = false\n"
            "jepa = true\njepa_weight = 0.1\n"
        )
    # The model constructor is the runtime seam: `validate` is not called there.
    with pytest.raises(LoudError, match="use_byteflow"):
        DormouseModel(
            tiny_cfg(
                use_byteflow=True,
                use_engram=False,
                bf16=False,
                byteflow_k_tokens=16,
                jepa=True,
                jepa_weight=0.1,
            )
        )
    with pytest.raises(LoudError, match="jepa_horizon"):
        DormouseModel(tiny_cfg(jepa=True, jepa_weight=0.1, jepa_horizon=0))
    assert load_preset("small").jepa_weight == 0.0


# --- the A/B driver's pure logic (no GPU) ----------------------------------


def _load_ab():
    spec = importlib.util.spec_from_file_location("ab_jepa", ROOT / "scripts" / "ab_jepa.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_driver_divergence_flag_and_verdict_are_pure() -> None:
    ab = _load_ab()
    assert ab.diverged([3.100, 3.110, 3.105]) == [False, False, False]
    flags = ab.diverged([3.100, 3.110, 4.200])
    assert flags[2] and not flags[0] and not flags[1]
    win = ab.verdict([3.10, 3.12, 3.11], [3.08, 3.07, 3.09])
    assert win["verdict"] == "win" and win["delta"] > win["spread"]
    tie = ab.verdict([3.10, 3.12, 3.11], [3.10, 3.11, 3.09])
    assert tie["verdict"] == "tie"
    assert ab.verdict([3.10, 3.12], [3.08])["verdict"] == "void"
