"""Gates for the two activation-memory arms (`iter_checkpoint`, `activation_offload`).

What is claimed, and what is deliberately NOT:

1. **OFF is bit-identical to main.** A fixed 5-step tiny run (SGD, one thread,
   seed 0) is pinned by the sha256 of its final `state_dict` and its final
   logits, captured from a clean detached worktree at `5857394` BEFORE the
   mechanisms were written. A default run must reproduce both, exactly.
2. **`iter_checkpoint` is exact AND actually recomputes.** Forward+backward
   with the flag on matches the off run to 1e-6 relative on logits and every
   gradient (CPU measures bit-identity); a spy on `LoopBlock._step` shows the
   body runs `2·depth` times (once per iteration in the forward, once in the
   backward) and `iter_checkpoint_saved` equals the ACTUAL depth, including a
   `depth=1` call whose count is 1 — the fixed-count defect this gate exists
   for. The 5-step off run is the harness for the pinned shas, so the two arms
   are compared on one model definition.
3. **`activation_offload` is exact and leak-free.** A 0.01 GiB budget (the tiny
   model's saves are bytes, so the arm has real work) produces the same 5-step
   state and logits as the off run; the first forward is the profile (no bytes
   parked), later forwards park tensors and count them; after EVERY backward
   the number of checked-out buffers is 0, and the pool stops growing (it is
   one reusable buffer per size class, not one per step). A two-micro-batch
   step with `grad_accum` also returns every buffer before the next forward.
4. **the refusals**: `full_graph` / `opt_graph` refuse both arms (a capture
   cannot see a recompute or a host copy) and `use_byteflow` refuses the
   offload (its chunked head's save set is not measured) — each with the escape.

**VRAM is NOT claimed here.** CPU cannot show a device peak and the card belongs
to a live LR-ladder run; what CPU can show is the semantics (the hooks fire, the
recompute happens, the buffers come back, the values are exact). Whether the
offload overlaps compute on this PCIe link is a GPU measurement
(`scripts/ab_actmem.py`, written and not run).
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn

from dormouse.config import (
    ModelConfig,
    RunConfig,
    TrainConfig,
    validate_model_train,
    validate_train,
)
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.loopblock import LoopBlock

#: The pinned run's shape: the tiny preset's body with `max_iter=3` so the loop
#: has three iterations to checkpoint (the preset's own 2 would make a
#: fixed-count bug in the arm harder to see than the actual count).
CFG = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=64,
    max_iter=3,
    rank=8,
    n_experts=4,
    moe_topk=2,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
)
BATCH, SEQ, STEPS, LR = 2, 64, 5, 0.01

#: Captured at `5857394` in a clean detached worktree (`/tmp/opencode/pin_actmem.py`,
#: `CUDA_VISIBLE_DEVICES=""`, one thread, seed 0, SGD lr=0.01, five steps).
PIN_STATE = "39d951a1859fe937b8080ab1883cdc49d33a79081903c9b43c77549562ddc384"
PIN_LOGITS = "129539b0b7eb99e8f11b409cb82f25e11bc5883d526e8983f1dcc4ac0e8c2872"


def _ids() -> tuple[torch.Tensor, torch.Tensor]:
    ids = (torch.arange(BATCH * SEQ).reshape(BATCH, SEQ) % 251).long()
    return ids, (ids + 1) % 251


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _state_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


#: `activation_offload` refuses `bf16=false` (fp32 doubles the pinned traffic),
#: so its mechanism gates run this fixture in bf16. The pinned OFF run above
#: stays fp32: it is the historical graph, not an offload run.
BF16_CFG = {**CFG, "bf16": True}


def _model(configure=None, over: dict | None = None) -> DormouseModel:
    torch.manual_seed(0)
    model = DormouseModel(ModelConfig(**{**CFG, **(over or {})}))
    if configure is not None:
        configure(model)
    return model


def _five_steps(
    configure=None, over: dict | None = None
) -> tuple[str, str, DormouseModel, list[dict]]:
    """The pinned run: five SGD steps, returning the shas and per-step records."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _model(configure, over)
        opt = torch.optim.SGD(model.parameters(), lr=LR)
        ids, targets = _ids()
        records: list[dict] = []
        logits = None
        for _ in range(STEPS):
            counters = Counters()
            logits, loss, counters = model(ids, targets, counters)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            stats = model._act_offload.stats() if model._act_offload is not None else {}
            records.append({"counters": dict(counters), "stats": stats})
        assert logits is not None
        return _state_digest(model), _digest(logits), model, records
    finally:
        torch.set_num_threads(old)


def _one_step(
    configure=None, over: dict | None = None
) -> tuple[torch.Tensor, torch.Tensor, Counters, DormouseModel]:
    """One forward+backward on a fresh identical model (the parity fixture)."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _model(configure, over)
        ids, targets = _ids()
        counters = Counters()
        logits, loss, counters = model(ids, targets, counters)
        loss.backward()
        return logits.detach().clone(), loss.detach().clone(), counters, model
    finally:
        torch.set_num_threads(old)


def _grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm()) / max(float(b.norm()), 1e-12)


# --------------------------------------------------------------- OFF: the pin


def test_off_is_bit_identical_to_the_pinned_main_run():
    """The default run is the historical graph: both shas must match exactly.

    This is the gate that fails if a branch for either mechanism leaks into the
    off path (a new view, a re-ordered accumulation, a stray hook). It was
    RED on the first version of the refactor, by a 4.7e-10 move on
    `embed.weight`, because the controller's `h0.reshape` view was rebuilt per
    iteration instead of shared — the fix is the shared `h0_flat`.
    """
    state, logits, _, _ = _five_steps()
    assert state == PIN_STATE, f"state_dict moved: {state} != {PIN_STATE}"
    assert logits == PIN_LOGITS, f"logits moved: {logits} != {PIN_LOGITS}"
    print(f"[actmem off] state={state[:16]} logits={logits[:16]} (pinned)")


# ------------------------------------------------------- iter_checkpoint (on)


def test_iter_checkpoint_is_exact_and_recomputes_the_actual_depth():
    """Same bits, and the body runs TWICE per iteration — proved by a spy.

    The spy is the answer to "the flag is on but silently does nothing": with
    no checkpoint the body runs `depth` times; with it, `2·depth` (forward +
    backward recompute), and `iter_checkpoint_saved` equals the depth the call
    actually ran, not `max_iter`.
    """
    base_logits, _, _, base_model = _one_step()
    base_grads = _grads(base_model)

    calls: list[int] = []
    real = LoopBlock._step

    def spy(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    LoopBlock._step = spy
    try:
        logits, _, counters, model = _one_step(
            lambda m: m.configure_actmem(iter_checkpoint=True, activation_offload=0.0)
        )
    finally:
        LoopBlock._step = real

    assert len(calls) == 2 * CFG["max_iter"], (
        f"the body ran {len(calls)} times: {CFG['max_iter']} means "
        f"`iter_checkpoint` did nothing (forward only); 2x means forward + the "
        f"backward recompute"
    )
    assert counters.get("iter_checkpoint_saved") == CFG["max_iter"], counters.line()
    assert _rel(logits, base_logits) <= 1e-6, "checkpointing must be exact on logits"
    got = _grads(model)
    assert set(got) == set(base_grads)
    worst = max(_rel(got[k], base_grads[k]) for k in got)
    assert worst <= 1e-6, f"the worst relative gradient move is {worst:.3e}"
    print(
        f"[actmem iter] body calls fwd+bwd={len(calls)} saved={CFG['max_iter']} "
        f"logits rel={_rel(logits, base_logits):.1e} grads rel={worst:.1e}"
    )

    # The ACTUAL depth, not `max_iter`: a `depth=1` call checkpoints ONE body.
    torch.manual_seed(0)
    model1 = DormouseModel(ModelConfig(**CFG))
    model1.configure_actmem(iter_checkpoint=True, activation_offload=0.0)
    ids, targets = _ids()
    calls.clear()
    LoopBlock._step = spy
    try:
        _, _, c1 = model1(ids, targets, Counters(), depth=1)
        assert len(calls) == 1, f"depth=1 ran {len(calls)} bodies in the forward"
        assert c1.get("iter_checkpoint_saved") == 1, c1.line()
    finally:
        LoopBlock._step = real


# ---------------------------------------------------- activation_offload (on)


def test_activation_offload_is_exact_and_returns_every_buffer():
    """Five steps: same shas as the off run, zero checked-out buffers each step.

    The leak gate is `live == 0` after every backward (a buffer the unpack hook
    never returned) plus a pool that STOPS growing: one reusable buffer per
    exact size class, not one allocation per step. The first forward is the
    profile, so it parks nothing by design (`offload_profile_forward`).
    """
    base_state, base_logits, base_model, _ = _five_steps(over=BF16_CFG)

    state, logits, model, records = _five_steps(
        lambda m: m.configure_actmem(iter_checkpoint=False, activation_offload=0.01),
        over=BF16_CFG,
    )
    # bf16 CPU is not bit-reproducible across *storage alignments*: the parked
    # buffer the unpack returns is a fresh allocation, so a vectorised
    # reduction can take a different lane order. The arm's contract is 1e-6
    # relative (the fp32 run is exact and pinned above).
    move = max(
        _rel(v, base_model.state_dict()[k])
        for k, v in model.state_dict().items()
        if v.is_floating_point()
    )
    assert move <= 1e-6, f"offload moved a parameter by {move:.3e}"
    # The forward itself is exact: one step's logits are `torch.equal` on both
    # arms (the hook fires when autograd SAVES, never when it computes). The
    # five-step curves diverge only through the 1e-9 parameter move above.
    assert _rel(_one_step(over=BF16_CFG)[0], _one_step(
        lambda m: m.configure_actmem(iter_checkpoint=False, activation_offload=0.01),
        over=BF16_CFG,
    )[0]) <= 1e-6

    mech = model._act_offload
    assert mech is not None
    assert records[0]["counters"].get("offload_profile_forward") == 1
    assert not records[0]["counters"].get("offload_tensors"), "the profile must not park"
    parked = [r["counters"].get("offload_tensors", 0) for r in records[1:]]
    assert all(n > 0 for n in parked), f"the arm never fired after the profile: {parked}"
    for step, record in enumerate(records):
        assert record["stats"]["live"] == 0, (
            f"step {step}: {record['stats']['live']} pinned buffer(s) were not "
            f"returned by the backward (the unpack hook leaked)"
        )
    pools = [r["stats"]["pool_tensors"] for r in records[1:]]
    # A capacity-dropping MoE makes the gathered-row count router-dependent, so
    # a NEW size class can appear on a later step; the pool is bounded by the
    # number of distinct sizes, not by the step count. What must hold is that
    # every allocation is still checked in (`allocations == pool`) and that
    # later forwards reuse buffers rather than allocating fresh ones.
    assert pools[-1] <= 2 * max(parked), f"the pool is not one-per-size-class: {pools}"
    assert mech.stats()["allocations"] == pools[-1], (
        f"allocations {mech.stats()['allocations']} != pooled {pools[-1]}: a buffer "
        f"was allocated and lost"
    )
    for record in records[1:]:
        parked_bytes = record["counters"].get("offload_forward_bytes", 0)
        assert parked_bytes == record["stats"]["last_forward_bytes"], (
            f"the counter and the mechanism disagree: {parked_bytes} != "
            f"{record['stats']['last_forward_bytes']}"
        )
        assert parked_bytes <= mech.budget, (
            f"parked {parked_bytes} B > budget {mech.budget} B"
        )
    # The pool is provably REUSED, not just stable: after the profile every
    # later forward checks out a buffer a previous backward returned.
    assert mech.stats()["reuses"] > 0, mech.stats()
    assert mech.stats()["allocations"] == pools[-1], (
        f"allocations {mech.stats()['allocations']} != pooled {pools[-1]}: the pool "
        f"is not one buffer per size class"
    )
    assert mech.stats()["live"] == 0
    print(
        f"[actmem offload] parked={parked} bytes={records[-1]['stats']['last_forward_bytes']} "
        f"pool={pools} alloc={mech.stats()['allocations']} reuse={mech.stats()['reuses']}"
    )


def _two_micro(
    configure=None, over: dict | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Counters, DormouseModel]:
    """Two micro-batches, gradients accumulated before any optimizer stage."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _model(configure, over)
        model.zero_grad(set_to_none=True)
        ids, targets = _ids()
        counters = Counters()
        logits = None
        for _ in range(2):
            counters = Counters()
            logits, loss, counters = model(ids, targets, counters)
            (loss / 2).backward()
        assert logits is not None
        return logits.detach().clone(), _grads(model), counters, model
    finally:
        torch.set_num_threads(old)


def test_activation_offload_composes_with_grad_accum():
    """Two micro-batches before one optimizer step: exact and leak-free.

    `grad_accum`'s contract is forward → backward → forward → backward before
    the optimizer stage. Each micro-batch's parked tensors must be consumed by
    ITS backward; a buffer left checked out would surface on the second
    forward (`live != 0`) or make the accumulated gradient wrong. The baseline
    is the SAME two-micro sequence with no hooks, so the `/2` division and the
    accumulation order match.
    """
    base_logits, base_grads, _, _ = _two_micro(over=BF16_CFG)
    logits, got, counters, model = _two_micro(
        lambda m: m.configure_actmem(iter_checkpoint=False, activation_offload=0.01),
        over=BF16_CFG,
    )
    assert counters.get("offload_tensors", 0) > 0, counters.line()
    assert _rel(logits, base_logits) <= 1e-6
    worst = max(_rel(got[k], base_grads[k]) for k in got)
    assert worst <= 1e-6, f"accumulated gradient moved by {worst:.3e}"
    assert model._act_offload.stats()["live"] == 0
    print(f"[actmem grad_accum] worst gradient rel={worst:.1e} live=0")


def test_both_arms_together_stay_exact():
    """`iter_checkpoint` + `activation_offload`: the composition keeps the bits.

    With checkpointing on, the offload pair sees only the checkpoint nodes'
    input saves (torch keeps the innermost hook pair; `act_ckpt.py` measured the
    same rule), so the two mechanisms do not double-park the body. The gate is
    the same 1e-6 parity and a nonzero `iter_checkpoint_saved`.
    """
    base_logits, _, _, base_model = _one_step(over=BF16_CFG)
    base_grads = _grads(base_model)
    logits, _, counters, model = _one_step(
        lambda m: m.configure_actmem(iter_checkpoint=True, activation_offload=0.01),
        over=BF16_CFG,
    )
    assert counters.get("iter_checkpoint_saved") == CFG["max_iter"], counters.line()
    assert _rel(logits, base_logits) <= 1e-6
    got = _grads(model)
    worst = max(_rel(got[k], base_grads[k]) for k in got)
    assert worst <= 1e-6, f"the composition moved a gradient by {worst:.3e}"
    print(f"[actmem both] logits/grads exact, saved={CFG['max_iter']}, worst={worst:.1e}")


def test_iter_checkpoint_composes_with_the_csa_ring_and_with_segment_len():
    """The two state-carrying paths: a CSA ring input and a checkpointed segment.

    CSA's ring is CARRIED across iterations (`attn_state`), so the iteration
    checkpoint must treat it as an input and recompute the body around it; a
    `segment_len` window is itself checkpointed, so this also proves the nested
    `checkpoint(use_reentrant=False)` is exact. Both are compared to the off run
    on the same seed, and the CSA fixture is `tiny_cfg`'s own (CPU-capable).
    """
    from conftest import tiny_cfg

    ids, targets = _ids()
    ids, targets = ids[:, :64], targets[:, :64]

    def parity(cfg, **forward):
        torch.manual_seed(0)
        base = DormouseModel(cfg)
        torch.manual_seed(0)
        on = DormouseModel(cfg)
        on.configure_actmem(iter_checkpoint=True, activation_offload=0.0)
        _, lb, _ = base(ids, targets, Counters(), **forward)
        lb.backward()
        _, lo, counters = on(ids, targets, Counters(), **forward)
        lo.backward()
        assert _rel(lo.detach(), lb.detach()) <= 1e-6
        gb, go = _grads(base), _grads(on)
        worst = max(_rel(go[k], gb[k]) for k in gb)
        assert worst <= 1e-6, f"the nested/carried checkpoint moved a gradient by {worst:.3e}"
        return counters, worst

    csa = tiny_cfg(use_csa=True, use_engram=False, moe_topk=0, csa_chunk=8, csa_topk=32)
    counters, worst = parity(csa)
    assert counters.get("iter_checkpoint_saved") == csa.max_iter, counters.line()
    print(f"[actmem csa] saved={csa.max_iter} worst grad rel={worst:.1e}")

    seg = tiny_cfg(moe_topk=0, use_msa=False)
    counters, worst = parity(seg, segment_len=32)
    assert counters.get("iter_checkpoint_saved") == seg.max_iter * 2, counters.line()
    assert counters.get("loop_segments") == 2, counters.line()
    print(f"[actmem segment] saved={counters['iter_checkpoint_saved']} worst grad rel={worst:.1e}")


# ------------------------------------------------------------- the refusals


def test_capture_arms_are_refused_with_the_reason_and_the_escape():
    """A graph capture cannot see a recompute or a Python hook: LOUD, not inert."""
    for flag in ("full_graph", "opt_graph"):
        for arm, value in (
            ("iter_checkpoint", True),
            ("activation_offload", 4.0),
            ("activation_recompute", "selective"),
        ):
            cfg = TrainConfig(**{flag: True, arm: value})
            try:
                validate_train(cfg)
            except LoudError as e:
                message = str(e)
                assert f"{arm}=" in message and flag in message, message
                assert "Escape:" in message, message
            else:  # pragma: no cover - the assertion below is the failure path
                raise AssertionError(f"{arm} + {flag} was not refused")


def test_byteflow_and_bad_budgets_are_refused():
    """`use_byteflow` has an unmeasured chunked save set; a bad budget is a typo."""
    model = ModelConfig(**CFG, use_byteflow=True)
    try:
        validate_model_train(model, TrainConfig(activation_offload=4.0))
    except LoudError as e:
        assert "use_byteflow" in str(e) and "Escape:" in str(e), str(e)
    else:  # pragma: no cover
        raise AssertionError("activation_offload + use_byteflow was not refused")
    for bad in (-1.0, float("nan"), float("inf")):
        try:
            validate_train(TrainConfig(activation_offload=bad))
        except LoudError as e:
            assert "Escape:" in str(e), str(e)
        else:  # pragma: no cover
            raise AssertionError(f"activation_offload={bad} was not refused")
    # fp32 doubles the pinned traffic; the arm's budget arithmetic is bf16.
    try:
        validate_model_train(ModelConfig(**CFG), TrainConfig(activation_offload=4.0))
    except LoudError as e:
        assert "bf16=false" in str(e) and "Escape:" in str(e), str(e)
    else:  # pragma: no cover
        raise AssertionError("activation_offload on an fp32 model was not refused")


def test_the_trainer_wires_both_flags(tmp_path):
    """`train()` installs both arms from the TrainConfig fields, once.

    The mechanism gates above call `configure_actmem` directly; this is the
    wiring gate — a `TrainConfig` flip must reach the model's loop and forward
    without a second switch. Two steps are enough: the first forward profiles
    the offload, the second parks tensors, and both steps count the recompute.
    """
    import dataclasses

    from conftest import tiny_cfg, tiny_run

    from dormouse.train import train

    run = tiny_run(
        tmp_path,
        steps=2,
        log_every=2,
        iter_checkpoint=False,
        activation_offload=0.01,
        activation_recompute="selective",
    )
    run = dataclasses.replace(run, model=tiny_cfg(bf16=True))
    model = DormouseModel(run.model)
    lines: list[str] = []
    report = train(model, run, log=lines.append)
    counters = report["counters"]
    assert counters.get("iter_checkpoint_saved") == 2 * run.model.max_iter, counters.line()
    assert counters.get("offload_tensors", 0) > 0, counters.line()
    assert model._act_offload is not None and model._act_offload.stats()["live"] == 0
    assert any("iter_checkpoint" in line for line in lines), lines
    print(
        f"[actmem train] iter_saved={counters['iter_checkpoint_saved']} "
        f"offloaded={counters['offload_tensors']}"
    )


def test_the_two_fields_are_snapshot_checked():
    """Both fields change behaviour, so a resume must refuse a flip.

    `RunConfig.snapshot_toml` writes every dataclass field and `diff_keys`
    compares every non-progress key; this is the gate that says the fields are
    in that set (a progress-key exemption would let a resume silently drop the
    arm).
    """
    a = RunConfig(source="x", model=ModelConfig(**CFG), train=TrainConfig())
    text = a.snapshot_toml()
    assert "iter_checkpoint" in text and "activation_offload" in text, text
    assert "activation_recompute" in text, text
    assert RunConfig.from_snapshot(text).train == a.train
    b = RunConfig(
        source="x",
        model=a.model,
        train=TrainConfig(
            iter_checkpoint=True, activation_offload=4.0, activation_recompute="selective"
        ),
    )
    moved = a.diff_keys(b)
    for key in ("train.iter_checkpoint", "train.activation_offload", "train.activation_recompute"):
        assert key in moved, (key, moved)
