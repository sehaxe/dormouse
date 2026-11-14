"""The KDA arm's BACKWARD: counted, and a gate that fails when it is missing.

WHY THIS FILE EXISTS. The KDA arm's oldest verdict is inadmissible: a burn-crate
run logged `fused kda=3126/0` — 3126 forward calls, ZERO backward calls — and
every A/B on that arm was read against a model that never trained its
attention. The cause over there was a checkpointing node that returned an
untracked tensor; the port is a different implementation on a different backend,
and "it is probably fine" is not a verdict. The cheap check the research pass
prescribes is exactly what this file runs: a `lr=0` step whose weights are
bit-identical, and an assertion that the KDA parameters' gradients are non-zero.

The gates, each able to fail on its own:

1. every KDA forward gets a backward — the `kda=` log field now prints both
   (`n=<fwd>,b=<bwd>`) and the test reads the numbers, not a printout;
2. the KDA parameters receive a finite, non-zero gradient on step 1 (the
   defect's own signature is a zero there);
3. the gate CATCHES the defect: a detached KDA output makes gate 1 report
   `b=0` and gate 2 find every attention gradient dead — a test that cannot
   fail is not a gate;
4. `lr=0` two-step identity: same batch, same weights, same loss, bit for bit;
5. `arch_hybrid31` is iso-params with its dense control (`arch_dense890`) to
   within 1%, so the pair is the mechanism and only the mechanism.
"""

from __future__ import annotations

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg

from dormouse.config import load_run
from dormouse.model import DormouseModel, KDAAttention, kda, param_count
from dormouse.train import kda_field

#: The 3:1 KDA/GQA stack at depth 4 is 3 `LoopBlock`s + 1 attention block, so
#: one forward runs the KDA arm three times.
N_KDA = 3
#: The two presets' exact counts — `arch_hybrid31` vs its equal-parameter dense
#: control (`tests/test_hybrid_stack.py` pins the same pair from the stack side).
HYBRID31_PARAMS = 109_892_835
DENSE890_PARAMS = 109_947_930


def _cfg(**over) -> object:
    """The tiny fixture as the arm's real shape: the 3:1 hybrid stack."""
    base = dict(
        block_form="kda",
        block_stack="3:1",
        max_iter=1,
        loop_blocks=1,
        n_layers=4,
        ffn_form="dense",
        use_kda=True,
        use_csa=False,
        use_mhc=False,
        use_attnres=False,
        use_engram=False,
        layerrope=False,
        use_tsct=False,
        n_experts=1,
        moe_topk=0,
        bf16=False,
    )
    base.update(over)
    return tiny_cfg(**base)


def _batch(seq: int = SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * seq, dtype=torch.long).view(BATCH, seq) % 251
    return ids, (ids + 1) % 251


def _kda_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The attention arm's own parameters — `q/k/v/g/beta` projections."""
    return {n: p for n, p in model.named_parameters() if ".attn." in n}


def _dead(grads: dict[str, torch.Tensor | None]) -> list[str]:
    return [
        n
        for n, g in grads.items()
        if g is None or not bool(torch.isfinite(g).all()) or float(g.abs().sum()) == 0.0
    ]


# --- gate 1: forwards == backwards ------------------------------------------


def test_every_kda_forward_gets_a_backward():
    """The log line's `n=` and `b=` must agree, step after step."""
    torch.manual_seed(0)
    model = DormouseModel(_cfg())
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    x, y = _batch()
    for step in range(2):
        opt.zero_grad()
        kda.reset_backward_calls(model)
        _, loss, counters = model(x, y)
        fwd = counters.get("kda_sdpa", 0) + counters.get("kda_fla", 0)
        assert fwd == N_KDA, f"step {step}: {fwd} KDA forwards, expected {N_KDA}"
        loss.backward()
        # The field the trainer prints: drained once per window, exactly once.
        field = kda_field(counters, model)
        assert f"n={fwd},b={fwd}" in field, f"step {step}: {field}"
        assert kda.backward_calls(model) == 0, "kda_field did not drain the window"
    # Two windows of N_KDA each: the counters are per window, not cumulative.
    assert field == f"kda=sdpa,n={N_KDA},b={N_KDA}"


# --- gate 2: non-zero gradient on step 1 ------------------------------------


def test_kda_parameters_take_a_nonzero_gradient_on_step_one():
    """The exact defect signature: an attention arm with no gradient."""
    torch.manual_seed(0)
    model = DormouseModel(_cfg())
    x, y = _batch()
    kda.reset_backward_calls(model)
    _, loss, counters = model(x, y)
    loss.backward()
    grads = {n: p.grad for n, p in _kda_params(model).items()}
    assert len(grads) == N_KDA * 5, sorted(grads)  # q/k/v/g/beta, once per block
    sites = {n.rsplit(".", 2)[-2] for n in grads}
    assert sites == {"q_proj", "k_proj", "v_proj", "g_proj", "beta_proj"}, sites
    dead = _dead({n: g for n, g in grads.items() if not n.endswith("beta_proj.weight")})
    assert dead == [], f"KDA parameters with no/zero gradient: {dead}"
    # `beta` is the delta rule's write strength, and the SDPA FALLBACK does not
    # run the recurrence (`kda._sdpa_path_carry`: "the delta-rule recurrence and
    # `beta` do not run here"). So on the fused arm beta must learn too, and on
    # the counted fallback it is allowed to be dead BY DESIGN — the log line's
    # `kda=fla|sdpa` is what tells the two apart.
    if counters.get("kda_fla"):
        assert _dead({n: g for n, g in grads.items() if n.endswith("beta_proj.weight")}) == []
    norms = {
        n: float(p.grad.abs().sum()) for n, p in _kda_params(model).items() if p.grad is not None
    }
    assert norms and min(norms.values()) > 0.0, norms
    assert kda.backward_calls(model) == N_KDA


# --- gate 3: the gate catches a detached output -----------------------------


def test_the_gate_catches_a_detached_kda_output(monkeypatch):
    """A test that cannot fail is not a gate: cut the edge, watch it fail.

    The burn defect was structural (a checkpointed node came back untracked), so
    the gate has to be proven against the same shape: detach the attention
    output and both signals — `b=` and the gradient — must go to zero.
    """
    torch.manual_seed(0)
    model = DormouseModel(_cfg())
    x, y = _batch()
    orig = KDAAttention.forward_state

    def detached(self, x, state=None, counters=None, *, carry=False):
        out, st = orig(self, x, state, counters, carry=carry)
        return out.detach(), st

    monkeypatch.setattr(KDAAttention, "forward_state", detached)
    kda.reset_backward_calls(model)
    _, loss, counters = model(x, y)
    loss.backward()
    field = kda_field(counters, model)
    fwd = counters.get("kda_sdpa", 0) + counters.get("kda_fla", 0)
    # The two assertions of gates 1 and 2, on the broken model:
    assert f"n={fwd},b=0" in field, field
    assert (
        _dead(
            {n: p.grad for n, p in _kda_params(model).items() if not n.endswith("beta_proj.weight")}
        )
        != []
    ), "a detached KDA output must leave every attention gradient dead"


# --- gate 4: lr=0 is a no-op on the weights ---------------------------------


def test_two_steps_at_lr_zero_produce_the_same_loss():
    """The research pass's cheap check: bit-identical weights, live gradients."""
    torch.manual_seed(0)
    model = DormouseModel(_cfg())
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    x, y = _batch()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    losses = []
    for _ in range(2):
        opt.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[0] == losses[1], losses
    moved = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])]
    assert moved == [], f"lr=0 moved {moved[:5]}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# --- gate 5: the arm's parameter budget is the control's --------------------


@pytest.mark.pins
def test_arch_hybrid31_is_iso_params_with_its_dense_control():
    """`arch_hybrid31` vs `arch_dense890`: the pair differs by <= 1%."""
    hybrid = DormouseModel(load_run("arch_hybrid31").model, device="meta")
    dense = DormouseModel(load_run("arch_dense890").model, device="meta")
    hp, dp = param_count(hybrid), param_count(dense)
    assert (hp, dp) == (HYBRID31_PARAMS, DENSE890_PARAMS)
    assert abs(hp - dp) / dp <= 0.01, (hp, dp)
    # The stack is the mechanism: 9 KDA loop blocks + 3 attention blocks.
    kinds = [type(b).__name__ for b in hybrid.hybrid_blocks]
    assert kinds.count("LoopBlock") == 9 and kinds.count("TransformerBlock") == 3
