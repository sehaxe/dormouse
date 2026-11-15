"""The AUX SEAM's gates: bit-identity off, the budget, grads, stop-grad, causality.

`model/aux.py` is ONE seam over two auxiliary objectives, both OFF by default.
These gates pin, in order:

1. `aux_objective = "none"` is the pre-change model bit for bit (the MTP
   goldens, measured before either head existed).
2. The parameter gate: a LightMTP head over 1% of the model is refused, and the
   built head's count is PRINTED so a log names it.
3. Each arm is the paper's term, recomputed line by line — LightMTP's cosine
   against `h[p+h]` (several horizons), FSP's bag-of-words summary (`t+2` start,
   membership by independent Python sets) under the balanced BCE.
4. Gradient direction: every head parameter gets a non-zero gradient, and the
   trunk receives a non-zero contribution FROM the aux term (zero the weight and
   a sampled trunk gradient differs; at weight 0 the head gets nothing).
5. Stop-gradient: the target carries no graph and the target positions of the
   input receive no gradient (the horizon slices are made disjoint to read it).
6. Causality: corrupting `h[p+h]` changes the LOSS but not the prediction at
   `p` — position `p` reads `h[p]`, never the future.
7. The token loss is untouched: the logits and the CE computed from them are
   `torch.equal` to the aux-off run, and the aux term rides beside them.
8. A 5-step CPU run per objective: finite, `last_aux` set, `aux=` logged
   SEPARATELY from `loss=`.
9. Every refusal: the literal, the weight/objective agreement, both ranges, the
   window cap, the segmented path, byteflow, and the transformer block form.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg, tiny_run

from dormouse.config import AUX_OBJECTIVES, ModelConfig, load_run, validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.aux import FSPHead, LightMTPHead, auto_dim, build_aux
from dormouse.train import train
from dormouse.train.loop import aux_field

#: Gate 1's literals: measured by `tests/test_mtp.py` on the tree BEFORE any
#: aux head existed (seed 1234, `tiny_cfg()`, the batch below).
GOLDEN_PARAMS = 64_881
GOLDEN_LOGITS_SHA256 = "c7839283a79291cf713aab57476cf202ba1a68167c365197f785da08203055c2"
GOLDEN_LOSS = 6.119609355926514

B, T = 2, 64


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = (torch.arange(B * T, dtype=torch.long) % 251).reshape(B, T)
    return ids, (ids + 7) % 256


def _aux_cfg(objective: str, **over) -> ModelConfig:
    base = dict(aux_weight=0.1, aux_dim=4)
    base.update(over)
    return tiny_cfg(aux_objective=objective, **base)


# --- gate 1: off is bit-identical ------------------------------------------


@pytest.mark.pins
def test_aux_off_is_the_pre_change_model_bit_for_bit() -> None:
    torch.manual_seed(1234)
    plain = DormouseModel(tiny_cfg())
    torch.manual_seed(1234)
    off = DormouseModel(tiny_cfg(aux_objective="none", aux_weight=0.0))
    assert plain.aux_head is None and off.aux_head is None
    assert off.last_aux is None
    assert param_count(off) == GOLDEN_PARAMS
    ids, targets = _fixed_batch()
    logits, loss, counters = off(ids, targets)
    assert hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest() == GOLDEN_LOGITS_SHA256
    assert float(loss.detach()) == pytest.approx(GOLDEN_LOSS, abs=1e-6)
    logits2, loss2, _ = plain(ids, targets)
    assert torch.equal(logits, logits2) and torch.equal(loss, loss2)
    assert not counters.get("aux"), "the off arm bumped the aux counter"


# --- gate 2: the parameter budget ------------------------------------------


def test_the_head_is_capped_at_one_percent_and_the_count_is_printed(capsys) -> None:
    """LightMTP's published budget is <= 1% (arXiv:2610.06031 §1)."""
    capsys.readouterr()
    model = DormouseModel(_aux_cfg("lightmtp", aux_horizon=2))
    n_head = sum(p.numel() for p in model.aux_head.parameters())
    n_model = param_count(model)
    assert n_head <= 0.01 * n_model, f"head {n_head} over 1% of {n_model}"
    out = capsys.readouterr().out
    assert f"aux=lightmtp params={n_head}" in out, out
    assert f"{100.0 * n_head / n_model:.3f}%" in out
    # A head that breaks the budget is refused, naming the knob and the escape.
    with pytest.raises(AssertionError, match="1%"):
        DormouseModel(_aux_cfg("lightmtp", aux_dim=64))
    # The auto width is the one that keeps the ratio scale-free.
    assert auto_dim(768) * 2 * 768 <= 0.01 * 9_200_000
    assert DormouseModel(_aux_cfg("lightmtp", aux_dim=0)).aux_head is not None


# --- gate 3: each arm is the paper's term ----------------------------------


def test_lightmtp_is_the_paper_cosine_against_the_models_own_future() -> None:
    """`1 − cos(g(h[p]), sg(h[p+h]))`, mean over horizons and positions."""
    torch.manual_seed(0)
    head = LightMTPHead(8, horizons=(1, 2, 4), dim=3)
    h = torch.randn(2, 12, 8)
    term = head(h)
    z = F.gelu(head.down(h))
    want = torch.stack(
        [
            (
                1.0
                - F.cosine_similarity(head.up[j](z[:, : 12 - step]), h[:, step:].detach(), dim=-1)
            ).mean()
            for j, step in enumerate((1, 2, 4))
        ]
    ).mean()
    assert float(term.detach()) == pytest.approx(float(want.detach()), abs=1e-6)
    # COSINE, not MSE: the paper's Eq. 1, and a squared error on the same
    # (unnormalised) vectors is a different number — the scale the target's
    # stop-grad lets the trunk drift would otherwise BE the objective.
    mse = torch.stack(
        [
            F.mse_loss(head.up[j](z[:, : 12 - step]), h[:, step:].detach())
            for j, step in enumerate((1, 2, 4))
        ]
    ).mean()
    assert abs(float(term.detach()) - float(mse.detach())) > 1e-3
    # The horizon is the shift, not a shape: h=1 and h=3 differ on this fixture.
    one = LightMTPHead(8, horizons=(1,), dim=3)
    with torch.no_grad():
        one.down.load_state_dict(head.down.state_dict())
        one.up[0].load_state_dict(head.up[0].state_dict())
    assert abs(float(one(h).detach()) - float(term.detach())) > 1e-4


def test_fsp_predicts_the_bag_of_words_of_the_future_window() -> None:
    """The summary is membership of `x[t+2 .. t+τ]` — verified with Python sets."""
    torch.manual_seed(0)
    head = FSPHead(8, 5, horizon=4, dim=3)
    ids = torch.tensor([[1, 2, 3, 4, 1, 4, 0, 2]])
    bow = head.summary(ids)
    assert bow.shape == (1, 8, 5) and not bow.requires_grad
    for p in range(8):
        span = {int(x) for x in ids[0, p + 2 : p + 5]}
        got = {i for i in range(5) if float(bow[0, p, i]) > 0}
        assert got == span, f"position {p}: {got} != {span}"
    # `t+2`, not `t+1`: the next byte is the NTP target and must not be counted.
    assert float(bow[0, 0, 2]) == 0.0, "the summary counted x[t+1]"
    # The loss is the balanced BCE: half the positive mean, half the negative,
    # over the positions that still HAVE a window (the last two do not).
    h = torch.randn(1, 8, 8)
    scored = head.summary(ids)[:, :6]
    logits = head.out(F.gelu(head.down(h))).float()[:, :6]
    per = F.binary_cross_entropy_with_logits(logits, scored, reduction="none")
    n_pos = scored.sum()
    want = 0.5 * (per * scored).sum() / n_pos + 0.5 * (per * (1.0 - scored)).sum() / (
        scored.numel() - n_pos
    )
    assert float(head(h, ids).detach()) == pytest.approx(float(want.detach()), abs=1e-6)
    # An empty-window position is not scored at all: its row carries no positive.
    assert float(head.summary(ids)[0, -1].sum()) == 0.0 and float(scored[0, -1].sum()) > 0.0


# --- gates 4 and 5: gradients reach both sides, the target carries none ----


def test_the_head_trains_and_the_trunk_hears_it() -> None:
    torch.manual_seed(0)
    model = DormouseModel(_aux_cfg("lightmtp", aux_horizon=2))
    ids, targets = _fixed_batch()
    model.cfg.aux_weight = 0.1
    _, loss, counters = model(ids, targets)
    assert counters.get("aux") == 1
    loss.backward()
    for name, p in model.aux_head.named_parameters():
        assert p.grad is not None, f"aux.{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"aux.{name} has a non-finite gradient"
        assert torch.count_nonzero(p.grad).item() > 0, f"aux.{name} got an all-zero gradient"
    on = model.embed.weight.grad.clone()
    term = float(model.last_aux)
    loss_on = float(loss.detach())
    model.zero_grad()
    model.cfg.aux_weight = 0.0
    _, loss_off, _ = model(ids, targets)
    loss_off.backward()
    off = model.embed.weight.grad.clone()
    # The trunk's gradient differs, and by exactly the aux term's contribution.
    assert not torch.equal(on, off), "the aux term never reached the trunk"
    assert loss_on - float(loss_off.detach()) == pytest.approx(0.1 * term, abs=1e-5)
    # At weight 0 the head is reached but contributes nothing.
    for p in model.aux_head.parameters():
        assert torch.count_nonzero(p.grad).item() == 0


def test_the_target_carries_no_gradient() -> None:
    """Stop-grad, read on a DISJOINT horizon: no gradient at `h[:, h:]`."""
    torch.manual_seed(0)
    head = LightMTPHead(8, horizons=(3,), dim=3)
    h = torch.randn(2, 6, 8, requires_grad=True)
    target = h[:, 3:].detach()
    assert target.requires_grad is False
    head(h).backward()
    assert torch.count_nonzero(h.grad[:, 3:]).item() == 0, "gradient reached the target"
    assert torch.count_nonzero(h.grad[:, :3]).item() > 0
    # The same forward with the target ATTACHED puts gradient on the target
    # positions and leaves the input half unchanged — that is what detach buys.
    h2 = h.detach().clone().requires_grad_(True)
    z = F.gelu(head.down(h2))
    attached = (1.0 - F.cosine_similarity(head.up[0](z[:, :3]), h2[:, 3:], dim=-1)).mean()
    attached.backward()
    assert torch.equal(h.grad[:, :3], h2.grad[:, :3])
    assert torch.count_nonzero(h2.grad[:, 3:]).item() > 0
    # FSP's summary is built from ids: no graph to detach, and nothing to leak.
    fsp = FSPHead(8, 5, horizon=4, dim=3)
    assert fsp.summary(torch.zeros(1, 8, dtype=torch.long)).requires_grad is False


# --- gate 6: causality ------------------------------------------------------


def test_position_p_never_sees_p_plus_h() -> None:
    """Corrupting the target at `p+h` moves the LOSS, never the prediction."""
    torch.manual_seed(0)
    head = LightMTPHead(8, horizons=(2,), dim=3)
    h = torch.randn(1, 8, 8)
    z = F.gelu(head.down(h))
    pred = head.up[0](z[:, :6]).detach().clone()
    clean = float(head(h).detach())
    hurt = h.clone()
    hurt[:, 6:] += 5.0  # the target positions of the LAST predictions only
    z2 = F.gelu(head.down(hurt))
    assert torch.equal(pred, head.up[0](z2[:, :6]).detach())
    assert abs(float(head(hurt).detach()) - clean) > 1e-4
    # The input path's gradient is the target's function, not its content: the
    # gradient at `h[:, :6]` is identical, only the loss value moved.
    h.grad = None
    h.requires_grad_(True)
    head(h).backward()
    g_clean = h.grad.clone()
    hurt.grad = None
    hurt.requires_grad_(True)
    head(hurt).backward()
    assert not torch.equal(g_clean, hurt.grad), "the target content never reached the head"


# --- gate 7: the token loss is untouched -----------------------------------


def test_the_token_loss_and_the_logits_are_unchanged() -> None:
    ids, targets = _fixed_batch()
    torch.manual_seed(7)
    off = DormouseModel(tiny_cfg())
    torch.manual_seed(7)
    on = DormouseModel(_aux_cfg("fsp", aux_horizon=4))
    logits_off, _, _ = off(ids, targets)
    logits_on, loss_on, counters = on(ids, targets)
    assert torch.equal(logits_off, logits_on), "the aux head changed the logits"
    ce = F.cross_entropy(logits_on, targets.reshape(-1))
    assert torch.isfinite(ce) and counters.get("aux") == 1
    # readout: the aux term is a SEPARATE scalar, added on top, not folded in.
    assert on.last_aux is not None and float(on.last_aux) > 0.0
    # `pure` (the held-out number) skips every aux term, this one included.
    on.last_aux = None
    _, _, counters_pure = on(ids, targets, pure=True)
    assert not counters_pure.get("aux") and on.last_aux is None


# --- gate 8: 5 CPU steps per objective -------------------------------------


@pytest.mark.parametrize("objective", ["lightmtp", "fsp"])
def test_five_steps_train_with_the_objective_on(tmp_path: Path, objective: str) -> None:
    lines: list[str] = []
    run = tiny_run(tmp_path, steps=5, log_every=5)
    model_cfg = dataclasses.replace(
        run.model,
        aux_objective=objective,
        aux_weight=0.1,
        aux_dim=4,
        aux_horizon=2 if objective == "lightmtp" else 4,
    )
    run = dataclasses.replace(run, model=model_cfg)
    torch.manual_seed(1)
    model = DormouseModel(run.model)
    report = train(model, run, log=lines.append)
    assert report["nan_skipped"] == 0
    assert torch.isfinite(torch.tensor(report["loss"])), report["loss"]
    assert model.last_aux is not None
    field = aux_field(model)
    assert field.startswith("aux_loss="), field
    line = [x for x in lines if "step=" in x][-1]
    assert "loss=" in line and field in line
    print(f"{objective}: {line}")


# --- gate 9: every refusal --------------------------------------------------


def test_the_objective_the_weight_and_the_ranges_are_refused_loudly() -> None:
    with pytest.raises(LoudError, match="aux_objective"):
        ModelConfig(aux_objective="lightmtp2", aux_weight=0.1)
    with pytest.raises(LoudError, match="aux_weight"):
        ModelConfig(aux_objective="none", aux_weight=0.1)
    with pytest.raises(LoudError, match="aux_weight"):
        ModelConfig(aux_objective="lightmtp", aux_weight=0.0)
    with pytest.raises(LoudError, match="aux_weight"):
        validate(ModelConfig(aux_objective="lightmtp", aux_weight=-0.1))
    with pytest.raises(LoudError, match="aux_dim"):
        validate(ModelConfig(aux_objective="lightmtp", aux_weight=0.1, aux_dim=-1))
    with pytest.raises(LoudError, match="aux_horizon"):
        validate(ModelConfig(aux_objective="fsp", aux_weight=0.1, aux_horizon=1))
    with pytest.raises(LoudError, match="aux_horizon"):
        validate(ModelConfig(aux_objective="lightmtp", aux_weight=0.1, aux_horizon=0))
    # The window cap is the load path's (seq_len // 8).
    with pytest.raises(LoudError, match="aux_horizon"):
        load_run(
            "small",
            model_overrides={"aux_objective": "fsp", "aux_weight": 0.1, "aux_horizon": 65},
            train_overrides={"seq_len": 512},
        )
    # The segmented forward cuts the window both targets read.
    with pytest.raises(LoudError, match="segment_len"):
        load_run(
            "small",
            model_overrides={"aux_objective": "lightmtp", "aux_weight": 0.1},
            train_overrides={"segment_len": 256},
        )
    # The doc-masking arm does not reach either head's target positions.
    with pytest.raises(LoudError, match="aux_weight"):
        load_run(
            "small",
            model_overrides={"aux_objective": "lightmtp", "aux_weight": 0.1},
            train_overrides={"doc_masking": True},
        )
    # Byteflow runs no byte-level loop to read positions from.
    with pytest.raises(LoudError, match="use_byteflow"):
        DormouseModel(
            tiny_cfg(
                use_byteflow=True,
                use_engram=False,
                bf16=False,
                byteflow_k_tokens=16,
                aux_objective="lightmtp",
                aux_weight=0.1,
            )
        )
    # The transformer block form has no loop readout for the head to sit on.
    with pytest.raises(LoudError, match="aux_weight"):
        validate(
            ModelConfig(
                aux_objective="lightmtp",
                aux_weight=0.1,
                block_form="transformer",
                max_iter=1,
                use_kda=False,
                use_engram=False,
            )
        )
    # The literal list is the single source both halves read.
    assert AUX_OBJECTIVES == ("none", "lightmtp", "fsp")
    assert build_aux("none", d_model=8, vocab=5, horizon=1) is None
    with pytest.raises(LoudError, match="aux_objective"):
        build_aux("nope", d_model=8, vocab=5, horizon=1)
