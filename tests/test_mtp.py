"""The six MTP gates (each one fails if its property breaks) + driver logic.

1. `mtp_k=0` is TODAY's model, bit-for-bit: the parameter count and the logits
   of a fixed-seed tiny build are pinned as literals measured on the commit
   before `model/mtp.py` existed (our own transcription — no external reference
   exists for a weight hash).
2. With `mtp_k=2, mtp_weight=1.0` every MTP parameter gets a finite non-zero
   gradient.
3. The MTP input embedding is built from `input_ids` only: mutating `targets`
   moves the CE but not the module's hidden state (the DSpark D1 lesson).
4. The shift is `targets[t+1]` at k=1, pinned on a hand fixture whose right and
   wrong labels give different CEs.
5. No host sync in the MTP forward: `.item()` / `.tolist()` (and `int(tensor)`,
   which routes through `.item()`) raise if the forward touches them.
6. `mtp_k > 8` and `mtp_weight > 0 with mtp_k == 0` are LOUD refusals.

The last two tests are the A/B driver's pure logic (`scripts/ab_mtp.py`): the
pre-registered verdict rule and the JSONL reader, which needs no GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg, tiny_run
from torch import nn

from dormouse.config import RunConfig, validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.mtp import MAX_K, MTPHead
from dormouse.train import train

ROOT = Path(__file__).resolve().parents[1]

#: Gate 1's literals, measured on the tree BEFORE this module existed:
#: `torch.manual_seed(1234)`, `tiny_cfg()`, ids = arange(2*64) % 251 reshaped,
#: targets = (ids + 7) % 256. Params 64 881; loss 6.119609355926514.
GOLDEN_PARAMS = 64_881
GOLDEN_LOGITS_SHA256 = "c7839283a79291cf713aab57476cf202ba1a68167c365197f785da08203055c2"
GOLDEN_LOSS = 6.119609355926514

B, T = 2, 64


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = (torch.arange(B * T, dtype=torch.long) % 251).reshape(B, T)
    return ids, (ids + 7) % 256


def _build(seed: int = 1234, **over) -> DormouseModel:
    torch.manual_seed(seed)
    return DormouseModel(tiny_cfg(**over))


def _mtp_params(model: nn.Module) -> dict[str, nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if n.startswith("mtp.")}


@pytest.mark.local_pin
def test_gate1_mtp_off_reproduces_the_recorded_model_bit_for_bit() -> None:
    model = _build()
    assert model.mtp is None and model.last_mtp is None
    assert param_count(model) == GOLDEN_PARAMS
    ids, targets = _fixed_batch()
    logits, loss, _ = model(ids, targets)
    assert hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest() == GOLDEN_LOGITS_SHA256
    assert float(loss.detach()) == pytest.approx(GOLDEN_LOSS, abs=1e-6)
    # The explicit off values are the SAME tree and the same forward, not a
    # second path that happens to look identical.
    explicit = _build(mtp_k=0, mtp_weight=0.0)
    assert param_count(explicit) == GOLDEN_PARAMS
    logits2, loss2, _ = explicit(ids, targets)
    assert torch.equal(logits, logits2)
    assert torch.equal(loss, loss2)


def test_gate2_every_mtp_parameter_gets_a_finite_non_zero_gradient() -> None:
    model = _build(seed=0, mtp_k=2, mtp_weight=1.0)
    params = _mtp_params(model)
    # The only new parameters are the two K projections — gain-free norms,
    # no bias: 2 x (2d * d) at d=64.
    assert len(params) == 2, sorted(params)
    assert param_count(model) - GOLDEN_PARAMS == 2 * (2 * 64 * 64)
    ids, targets = _fixed_batch()
    _, loss, _ = model(ids, targets)
    loss.backward()
    for name, p in params.items():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert torch.count_nonzero(p.grad).item() > 0, f"{name} got an all-zero gradient"


def test_gate3_the_mtp_input_never_reads_the_label_sequence() -> None:
    torch.manual_seed(0)
    head = MTPHead(8, 2)
    embed = nn.Embedding(16, 8)
    lm_head = nn.Linear(8, 16)
    seen: list[torch.Tensor] = []
    lm_head.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
    g = torch.Generator().manual_seed(7)
    h = torch.randn(2, 6, 8, generator=g)
    ids = torch.randint(0, 16, (2, 6), generator=g)
    targets = torch.randint(0, 16, (2, 6), generator=g)
    changed = targets.clone()
    # Position 1 is the first label any module reads (shift j+1); position 0 is
    # never a label (the trunk scored it), so mutating it would prove nothing.
    changed[0, 1] = (changed[0, 1] + 1) % 16

    loss_a, _ = head(h, ids, targets, embed=embed, head=lm_head, weight=1.0)
    first = list(seen)
    seen.clear()
    loss_b, _ = head(h, ids, changed, embed=embed, head=lm_head, weight=1.0)
    second = list(seen)

    assert len(first) == len(second) == 2, "one head call per MTP module"
    for a, b in zip(first, second, strict=True):
        # `x_j` is the module's hidden state: the tensor the shared head reads.
        assert torch.equal(a, b), "the MTP hidden state moved when only targets moved"
    assert not torch.equal(loss_a, loss_b), "the CE must move when its label moves"


def test_gate4_k1_scores_targets_shifted_by_one() -> None:
    head = MTPHead(2, 1, eps=0.0)
    embed = nn.Embedding(3, 2)
    readout = nn.Linear(2, 3)
    with torch.no_grad():
        embed.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        head.projs[0].weight.copy_(torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]))
        readout.weight.zero_()
        readout.bias.copy_(torch.tensor([0.0, 0.0, 2.0]))
    h = torch.ones(1, 2, 2)
    ids = torch.tensor([[2, 1]])
    # Right shift: targets[1] = 0. Wrong shift: targets[0] = 2.
    targets = torch.tensor([[2, 0]])
    loss, per_k = head(h, ids, targets, embed=embed, head=readout, weight=1.0)
    # logits are the bias [0, 0, 2] at every position; CE(0) = log(2 + e^2).
    right = math.log(2.0 + math.exp(2.0))
    wrong = math.log(1.0 + 2.0 * math.exp(-2.0))  # log(2 + e^2) - 2
    assert float(per_k[0].detach()) == pytest.approx(right, abs=1e-6)
    assert float(loss.detach()) == pytest.approx(right, abs=1e-6)
    assert abs(right - wrong) > 1.0, "the fixture must separate the two shifts"


def test_gate5_the_mtp_forward_touches_no_host(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    head = MTPHead(8, 2)
    embed = nn.Embedding(16, 8)
    lm_head = nn.Linear(8, 16)
    g = torch.Generator().manual_seed(7)
    h = torch.randn(2, 6, 8, generator=g)
    ids = torch.randint(0, 16, (2, 6), generator=g)
    targets = torch.randint(0, 16, (2, 6), generator=g)

    def boom(*_a, **_k):
        raise AssertionError("host sync inside the MTP forward (ADR-0018 rule 2)")

    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "tolist", boom)
    loss, per_k = head(h, ids, targets, embed=embed, head=lm_head, weight=1.0)
    assert loss.requires_grad and len(per_k) == 2


def test_gate6_the_two_misconfigurations_are_loud() -> None:
    with pytest.raises(LoudError, match="mtp_k=9"):
        validate(tiny_cfg(mtp_k=9))
    with pytest.raises(LoudError, match="mtp_weight"):
        validate(tiny_cfg(mtp_k=0, mtp_weight=0.1))
    with pytest.raises(LoudError, match="mtp k=9"):
        MTPHead(8, 9)
    # The model constructor refuses WITHOUT passing through validate (the
    # runtime seam: a hand-built ModelConfig reaches MTPHead directly).
    with pytest.raises(LoudError, match="mtp k=0"):
        DormouseModel(tiny_cfg(mtp_k=0, mtp_weight=0.5))
    assert MAX_K == 8


def test_the_byteflow_channel_refuses_mtp_loudly() -> None:
    cfg = tiny_cfg(
        use_byteflow=True,
        use_engram=False,
        bf16=False,
        byteflow_k_tokens=16,
        mtp_k=1,
        mtp_weight=0.5,
    )
    with pytest.raises(LoudError, match="mtp_weight"):
        validate(cfg)


def test_the_trainers_log_line_carries_mtp_only_when_the_arm_is_on(tmp_path: Path) -> None:
    """The loop.py seam: one `mtp=<value>` on the counter line, read on cadence."""
    off = tiny_run(tmp_path, steps=1, log_every=1)
    off_lines: list[str] = []
    train(DormouseModel(off.model), off, log=off_lines.append)
    assert "mtp=" not in off_lines[0], off_lines[0]

    base = tiny_run(tmp_path, steps=1, log_every=1)
    on = RunConfig(source="tiny", model=tiny_cfg(mtp_k=1, mtp_weight=0.5), train=base.train)
    model = DormouseModel(on.model)
    on_lines: list[str] = []
    train(model, on, log=on_lines.append)
    assert "mtp=" in on_lines[0], on_lines[0]
    assert model.last_mtp is not None and torch.isfinite(model.last_mtp)


# --- the A/B driver's pure logic -------------------------------------------


def _load_ab():
    """Load `scripts/ab_mtp.py` by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("ab_mtp", ROOT / "scripts" / "ab_mtp.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ab = _load_ab()
WINDOW = 20 * 8 * 512  # eval_batches x batch x seq_len


def _run(seed: int, bpb: float, *, mtp: int = 1, nan: int = 0, scored: int = WINDOW) -> dict:
    return {"seed": seed, "bpb": bpb, "scored": scored, "nan": nan, "mtp_logged": mtp}


CONTROL = [_run(1, 5.0, mtp=0), _run(2, 5.2, mtp=0), _run(3, 5.1, mtp=0)]  # mean 5.1, spread .2


def test_ab_verdict_win_must_beat_the_controls_own_spread() -> None:
    arm = [_run(1, 4.5), _run(2, 4.6), _run(3, 4.55)]
    v = ab.verdict(CONTROL, arm, WINDOW)
    assert v["verdict"] == "win" and v["control_spread"] == pytest.approx(0.2)


def test_ab_verdict_inside_the_spread_is_a_tie_and_a_void_arm_is_not_a_number() -> None:
    tie = [_run(1, 5.0), _run(2, 5.1), _run(3, 5.05)]
    assert ab.verdict(CONTROL, tie, WINDOW)["verdict"] == "tie"
    silent = [_run(1, 3.0, mtp=0), _run(2, 3.0, mtp=0), _run(3, 3.0, mtp=0)]
    v = ab.verdict(CONTROL, silent, WINDOW)
    assert v["verdict"] == "void" and "mtp=" in v["reason"]
    assert ab.verdict(CONTROL, [_run(1, 3.0, nan=1)], WINDOW)["verdict"] == "void"


def test_ab_read_run_takes_the_last_eval_and_counts_the_mtp_field(tmp_path: Path) -> None:
    rows = [
        {"event": "run", "steps": 2000, "batch": 8, "seq_len": 512},
        {"event": "log", "step": 250, "nan_skipped": 0},
        {"event": "log", "step": 500, "nan_skipped": 1},
        {"event": "eval", "step": 500, "bpb": 4.9, "scored": WINDOW},
        {"event": "eval", "step": 2000, "bpb": 4.4, "scored": WINDOW},
    ]
    jsonl = tmp_path / "k2_s1.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    log = tmp_path / "k2_s1.log"
    log.write_text("step=250 loss=4.1 lr=1e-3 kda=fla,n=1000 mtp=0.8123 forward=200\n")
    r = ab.read_run(jsonl, log)
    assert r["bpb"] == 4.4 and r["scored"] == WINDOW and r["n_evals"] == 2
    assert r["nan"] == 1 and r["mtp_logged"] == 1
    empty = tmp_path / "empty.jsonl"
    empty.write_text(json.dumps({"event": "run", "steps": 2000}) + "\n")
    with pytest.raises(AssertionError, match="no eval row"):
        ab.read_run(empty, log)


def test_ab_preflight_refuses_a_missing_data_file_without_touching_the_card(tmp_path: Path) -> None:
    why = ab.preflight("cpu", str(tmp_path / "nope.bin"), str(tmp_path / "nope_eval.bin"))
    assert why is not None and "no data file" in why
