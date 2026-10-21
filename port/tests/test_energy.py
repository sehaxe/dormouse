"""The energy head's gates: off-arm identity, calibration, purity, grads, no
host sync, and the reranker's selection.

The calibration gate (b) is the falsifiable one: an UNTRAINED head is measured
first and must fail it, then the same synthetic task trains the head and the
correct-byte argmin rate is printed (run with `-s` to see both numbers). The
scratch revert that turns it red is deleting the positive/negative split: the
loss then cannot prefer the target and the trained rate collapses toward 1/V.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg, tiny_model

from dormouse.config import validate
from dormouse.instrument import LoudError
from dormouse.model.energy import EnergyHead, energy_contrastive_loss, sample_negatives
from dormouse.model.model import DormouseModel, param_count

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Load `scripts/<name>.py` by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rerank = _load("rerank")


def _ids_batch(seed: int = 0, b: int = 2, t: int = 16):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, t), generator=g), torch.randint(0, 256, (b, t), generator=g)


# ---------------------------------------------------------------------------
# gate a — weight 0 builds no head and is bit-identical
# ---------------------------------------------------------------------------


def test_gate_a_weight_zero_builds_nothing_and_is_bit_identical():
    """Hostile inert values (`energy_negs=7`, `energy_rank=3`) must move no bit.

    Both models are built under the same seed, so any RNG draw the head made
    would shift the backbone and break `torch.equal` — the gate is the digest,
    not just the parameter count.
    """
    torch.manual_seed(5)
    off = tiny_model()
    torch.manual_seed(5)
    inert = tiny_model(energy_negs=7, energy_temp=0.05, energy_rank=3)
    assert off.energy is None and inert.energy is None
    assert param_count(off) == param_count(inert)
    assert list(off.state_dict()) == list(inert.state_dict())
    ids, targets = _ids_batch()
    assert torch.equal(off(ids, targets)[0], inert(ids, targets)[0])
    assert torch.equal(off(ids, targets)[1], inert(ids, targets)[1])

    torch.manual_seed(5)
    on = tiny_model(energy_weight=0.05, energy_rank=16)
    assert on.energy is not None
    head_params = sum(p.numel() for p in on.energy.parameters())
    assert head_params > 0
    assert param_count(on) - param_count(off) == head_params
    # The term is ADDED, not replacing the CE: the backbone is the same seed.
    base = off(ids, targets)[1]
    armed = on(ids, targets)[1]
    assert armed > base, "the energy term did not reach the loss"


# ---------------------------------------------------------------------------
# gate b — calibration against a known-correct byte, with the random control
# ---------------------------------------------------------------------------


def test_gate_b_the_head_calibrates_and_a_random_init_head_fails():
    """Synthetic task: `h = embed[target] + noise`, so the correct byte is known.

    The head sees `(h, candidate)` only. Training it with the real loss must make
    the correct candidate the argmin for ~every position; the SAME architecture
    at init is the control and sits at chance (`1/256`). Both numbers are
    printed, so the threshold is a measurement, not a claim.
    """
    d, rank, steps = 32, 32, 300
    g = torch.Generator().manual_seed(3)
    torch.manual_seed(3)
    table = torch.nn.Embedding(256, d)
    table.weight.requires_grad_(False)
    targets = torch.randint(0, 256, (512,), generator=g)
    h = table.weight[targets] + 0.05 * torch.randn(512, d, generator=g)
    logits = torch.randn(512, 256, generator=g)
    head = EnergyHead(d, rank)

    def accuracy(model: EnergyHead) -> float:
        picked = model.scores(h, table.weight).argmin(dim=-1)
        return float((picked == targets).float().mean())

    before = accuracy(head)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = energy_contrastive_loss(
            head, h, table.weight, targets, logits, n_negs=64, temp=0.5, generator=g
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = accuracy(head)
    print(
        f"\ncalibration: random-init {before:.4f} -> trained {after:.4f} "
        f"(chance {1 / 256:.4f}, threshold 0.99)"
    )
    assert before < 0.05, "the control is not at chance — the fixture leaks the answer"
    assert after >= 0.99, f"calibrated only to {after:.4f} of 1.0"
    # The same arithmetic in the two APIs: `scores` over all V IS `energy` per
    # candidate, so the reranker and the loss cannot disagree about the score.
    pair = head.energy(h, table.weight[targets])
    gathered = head.scores(h, table.weight).gather(1, targets.reshape(-1, 1)).squeeze(1)
    assert torch.allclose(pair, gathered, atol=1e-6)


def test_sample_negatives_never_draws_the_positive():
    g = torch.Generator().manual_seed(4)
    targets = torch.randint(0, 256, (64,), generator=g)
    logits = torch.randn(64, 256, generator=g)
    # A distribution that would draw the target almost always if it were not
    # removed: logit 30 at the target, 0 elsewhere.
    logits.scatter_(1, targets.reshape(-1, 1), 30.0)
    neg = sample_negatives(logits, targets, 32, generator=g)
    assert neg.shape == (64, 32)
    assert not (neg == targets.reshape(-1, 1)).any(), "the sampler drew the positive"


# ---------------------------------------------------------------------------
# gate c — the score is a pure function of (h, candidate), never of targets
# ---------------------------------------------------------------------------


def test_gate_c_the_score_reads_only_h_and_candidate():
    """Signatures AND runtime inputs: targets never reach the scorer (DSpark D1).

    The hook on the head captures the exact tensors the loss feeds it. Two
    forwards over the same ids with DIFFERENT targets must produce identical
    logits (targets are labels, not input), and every captured call must be
    `(h, candidate_vectors)` — a byte-id tensor of shape `[P]` would be the
    leak this gate exists to catch.
    """
    assert "targets" not in inspect.signature(EnergyHead.energy).parameters
    assert "targets" not in inspect.signature(EnergyHead.scores).parameters
    assert "targets" not in inspect.signature(EnergyHead.forward).parameters

    torch.manual_seed(2)
    model = DormouseModel(tiny_cfg(energy_weight=0.05, energy_rank=16, energy_negs=8))
    seen: list[tuple[tuple, torch.Tensor]] = []
    handle = model.energy.register_forward_hook(lambda _m, args, out: seen.append((args, out)))
    ids, t1 = _ids_batch(seed=7)
    t2 = (t1 + 7) % 256
    logits1 = model(ids, t1)[0]
    first = list(seen)
    seen.clear()
    logits2 = model(ids, t2)[0]
    second = list(seen)
    handle.remove()

    assert torch.equal(logits1, logits2), "targets changed the forward, not just the loss"
    assert len(first) == len(second) == 2, "one call for the positive, one for the negatives"
    for args, out in (*first, *second):
        assert len(args) == 2
        h_arg, cand_arg = args
        assert h_arg.shape[-1] == model.cfg.d_model
        assert cand_arg.shape[-1] == model.cfg.d_model
        assert cand_arg.ndim == h_arg.ndim, "candidate and state must broadcast"
        # The head's output is recomputed from exactly these two inputs.
        assert torch.equal(model.energy(*args), out)


# ---------------------------------------------------------------------------
# gate d — every energy parameter gets a finite, non-zero gradient
# ---------------------------------------------------------------------------


def test_gate_d_every_energy_parameter_gets_a_finite_nonzero_grad():
    torch.manual_seed(9)
    model = tiny_model(energy_weight=0.05, energy_rank=16, energy_negs=16)
    ids, targets = _ids_batch(seed=11)
    _, loss, _ = model(ids, targets)
    loss.backward()
    names = []
    for name, p in model.energy.named_parameters():
        names.append(name)
        assert p.grad is not None, f"{name}: no gradient — the term does not train it"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient"
        assert p.grad.abs().sum() > 0.0, f"{name}: zero gradient"
    assert names == ["h_proj.weight", "h_proj.bias", "y_proj.weight", "out.weight"]


# ---------------------------------------------------------------------------
# gate e — no host sync in the contrastive path
# ---------------------------------------------------------------------------


def test_gate_e_the_contrastive_path_touches_no_host(monkeypatch):
    """Patch the six host reads; the loss and `scores` must run through them.

    The gate is scoped to the NEW path (`energy_contrastive_loss` + `scores`):
    `_forward_byte_lm`'s own `assert torch.isfinite(loss)` is a pre-existing
    host bool on the loss scalar, outside this mechanism.
    """

    def boom(*_a, **_k):
        raise AssertionError("host sync inside the energy path (ADR-0018 rule 2)")

    for name in ("item", "tolist", "__bool__", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=True)
    g = torch.Generator().manual_seed(6)
    head = EnergyHead(16, 8)
    h = torch.randn(4, 16, generator=g)
    cand = torch.randn(256, 16, generator=g)
    targets = torch.randint(0, 256, (4,), generator=g)
    logits = torch.randn(4, 256, generator=g)
    loss = energy_contrastive_loss(head, h, cand, targets, logits, n_negs=6, generator=g)
    scores = head.scores(h, cand)
    assert loss.ndim == 0
    assert torch.equal(torch.isfinite(loss), torch.ones((), dtype=torch.bool))
    assert torch.equal(torch.isfinite(scores), torch.ones_like(scores, dtype=torch.bool))
    loss.backward()


# ---------------------------------------------------------------------------
# gate f — the reranker's selection is deterministic and lowest-energy
# ---------------------------------------------------------------------------


class _StubHead:
    """A head whose scores are fixed: candidate 1 is the lowest for every state."""

    def scores(self, h: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        out = torch.full((h.shape[0], candidates.shape[0]), 5.0)
        out[:, 1] = -2.0
        return out


def test_gate_f_the_reranker_picks_the_lowest_energy_deterministically():
    energies = torch.tensor([0.5, -1.0, 2.0])
    assert rerank.pick(energies) == 1
    assert rerank.pick(energies.clone()) == 1  # pure, so a fixed seed repeats it
    assert rerank.pick(torch.tensor([-1.0, -1.0, 0.0])) == 0, "ties must take the earliest"
    h = torch.zeros(2, 4)
    cand = torch.zeros(3, 4)
    # next byte ids 0 and 1 -> picked energies 5.0 and -2.0 -> mean 1.5
    assert float(rerank.sequence_energy(_StubHead(), h, cand, torch.tensor([0, 1]))) == 1.5
    # The same seed is the same rollout stream (the reproduction contract).
    a = rerank.rollout_generator(3, 7, 2, device="cpu")
    b = rerank.rollout_generator(3, 7, 2, device="cpu")
    pa = torch.rand(4, generator=a)
    pb = torch.rand(4, generator=b)
    assert torch.equal(pa, pb)


# ---------------------------------------------------------------------------
# the config refusals
# ---------------------------------------------------------------------------


def test_the_two_interactions_and_the_scales_are_loud():
    with pytest.raises(LoudError, match="use_byteflow"):
        validate(
            tiny_cfg(
                use_byteflow=True,
                use_engram=False,
                byteflow_k_tokens=64,
                energy_weight=0.05,
            )
        )
    with pytest.raises(LoudError, match="dspark_weight"):
        validate(tiny_cfg(energy_weight=0.05, dspark_weight=0.05))
    with pytest.raises(LoudError, match="energy_weight"):
        validate(tiny_cfg(energy_weight=-0.1))
    with pytest.raises(LoudError, match="energy_negs"):
        validate(tiny_cfg(energy_negs=0))
    with pytest.raises(LoudError, match="energy_temp"):
        validate(tiny_cfg(energy_temp=0.0))
    with pytest.raises(LoudError, match="energy_rank"):
        validate(tiny_cfg(energy_rank=0))
    validate(tiny_cfg(energy_weight=0.05, energy_negs=64, energy_temp=0.5, energy_rank=16))
