"""Gates for the VRAM arms: recompute modes, offload accounting, optimizer launches.

What is claimed, and how each claim is made CPU-checkable:

1. **OFF is bit-identical to clean main.** Five SGD steps on a fixed tiny model,
   pinned by the sha256 of the final `state_dict` and logits, captured at the
   branch point (`3e736ae`) BEFORE the arm was written.
2. **The three recompute arms are exact.** `none` / `full` / `selective` share
   logits to 1e-6 and every gradient to 1e-5 (CPU measures bit-identity), and a
   spy proves the bodies run where each arm says: `full` re-runs the whole
   iteration, `selective` re-runs only its write half.
3. **The memory ledger is arithmetic, not a device guess.** A counting
   `saved_tensors_hooks` pair sees exactly what each arm holds for the backward;
   the per-arm tensor count and byte total are PINNED, so adding a saved tensor
   without updating the table turns the gate red. `selective` keeps more than
   `full` (the read half) and less than `none`.
4. **The offload budget is exact and refuses.** The parked bytes equal the
   counter and stay under the budget, the pool's reuse is counted, and an
   over-budget forward is LOUD instead of silently partial.
5. **The optimizer launch count drops.** A dispatch meter shows the fused
   AdamW path issues far fewer ops than torch's walk, the fast path is
   bit-identical to it, `clip_grads` stays at one `_foreach_norm` + one
   `_foreach_mul_`, and the changed steps add no host sync.

The GPU side (peak GiB, ms/step, B/s per arm) is `scripts/bench_vram.py`; no
GPU number is claimed here.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from dormouse.config import (
    ModelConfig,
    TrainConfig,
    validate_model_train,
    validate_train,
)
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.act_mem import GIB, ActivationOffload
from dormouse.model.loopblock import LoopBlock
from dormouse.model.recompute import check_mode, projection_bytes, saved_table
from dormouse.train.optim import MixOptimizer, clip_grads

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

#: Captured from clean main (`3e736ae`) before this branch touched a line.
PIN_STATE = "39d951a1859fe937b8080ab1883cdc49d33a79081903c9b43c77549562ddc384"
PIN_LOGITS = "129539b0b7eb99e8f11b409cb82f25e11bc5883d526e8983f1dcc4ac0e8c2872"

#: The per-arm resident saved-tensor ledger for the tiny fixture, measured from
#: the code's own `saved_tensors_hooks` (not a hand count). A new save moves
#: these numbers; the pin is the reason a silent memory regression cannot land.
SAVE_TABLE = {
    "none": (512, 7_240_076),
    "full": (41, 1_451_392),
    "selective": (260, 5_394_400),
}


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _state_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _ids() -> tuple[torch.Tensor, torch.Tensor]:
    ids = (torch.arange(BATCH * SEQ).reshape(BATCH, SEQ) % 251).long()
    return ids, (ids + 1) % 251


def _model(mode: str = "none", over: dict | None = None) -> DormouseModel:
    torch.manual_seed(0)
    model = DormouseModel(ModelConfig(**{**CFG, **(over or {})}))
    if mode != "none":
        model.configure_actmem(
            iter_checkpoint=False, activation_offload=0.0, activation_recompute=mode
        )
    return model


def _one_step(mode: str = "none", *, grad: bool = True, over: dict | None = None):
    model = _model(mode, over)
    ids, targets = _ids()
    counters = Counters()
    logits, loss, counters = model(ids, targets, counters)
    if grad:
        loss.backward()
    return logits.detach().clone(), loss.detach().clone(), counters, model


def _grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm()) / max(float(b.norm()), 1e-12)


# --------------------------------------------------------------- 1. the OFF pin


@pytest.mark.pins
def test_off_is_bit_identical_to_main():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _model()
        opt = torch.optim.SGD(model.parameters(), lr=LR)
        ids, targets = _ids()
        logits = None
        for _ in range(STEPS):
            logits, loss, _ = model(ids, targets, Counters())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        assert logits is not None
        assert _state_digest(model) == PIN_STATE, "the off path moved a parameter"
        assert _digest(logits) == PIN_LOGITS, "the off path moved the logits"
    finally:
        torch.set_num_threads(old)


# --------------------------------------------------- 2. the three arms are exact


def test_selective_and_full_match_no_checkpoint():
    base_logits, _, _, base_model = _one_step("none")
    base_grads = _grads(base_model)
    for mode in ("full", "selective"):
        logits, _, counters, model = _one_step(mode)
        assert _rel(logits, base_logits) <= 1e-6, f"{mode}: logits moved"
        got = _grads(model)
        assert set(got) == set(base_grads)
        worst = max(_rel(got[k], base_grads[k]) for k in got)
        assert worst <= 1e-5, f"{mode}: the worst gradient move is {worst:.3e}"
        assert counters.get("iter_checkpoint_saved") == CFG["max_iter"], counters.line()
        print(f"[vram {mode}] logits rel={_rel(logits, base_logits):.1e} grads={worst:.1e}")


def test_the_body_runs_where_each_arm_says():
    """`full` re-runs `_step`; `selective` re-runs only `_step_post`."""
    real_step, real_post = LoopBlock._step, LoopBlock._step_post
    calls = {"step": 0, "post": 0}

    def step_spy(self, *a, **k):
        calls["step"] += 1
        return real_step(self, *a, **k)

    def post_spy(self, *a, **k):
        calls["post"] += 1
        return real_post(self, *a, **k)

    LoopBlock._step, LoopBlock._step_post = step_spy, post_spy
    try:
        for mode in ("full", "selective"):
            calls.update(step=0, post=0)
            _one_step(mode)
            depth = CFG["max_iter"]
            assert calls["post"] == 2 * depth, (mode, calls)
            want_step = 2 * depth if mode == "full" else depth
            assert calls["step"] == want_step, (mode, calls)
            print(f"[vram {mode}] _step={calls['step']} _step_post={calls['post']}")
    finally:
        LoopBlock._step, LoopBlock._step_post = real_step, real_post


# ------------------------------------------------------- 3. the memory ledger


def test_the_saved_tensor_table_is_pinned_per_arm():
    ck = Counters()
    for mode in ("none", "full", "selective"):
        model = _model(mode)
        ids, targets = _ids()
        table, logits = saved_table(lambda m=model, i=ids, t=targets: m(i, t, ck))
        assert table.tensors and table.bytes
        want = SAVE_TABLE[mode]
        assert (table.tensors, table.bytes) == want, (
            f"{mode}: saved {table.tensors} tensors / {table.bytes} B, pinned {want}. "
            f"A new save must update the table (and the report's ledger)."
        )
        print(f"[vram table {mode}] {table.tensors} tensors, {table.bytes} B")


def test_selective_keeps_the_read_half_and_drops_the_write_half():
    """The kept set is the read half: q/k/v/g outputs + boundary state."""
    ids, targets = _ids()
    table, _ = saved_table(lambda: _model("selective")(ids, targets, Counters()))
    kept = projection_bytes([_model("selective").loop], BATCH * SEQ, 4)
    assert kept > 0, "the fixture's attention owns no projection"
    assert table.bytes > SAVE_TABLE["full"][1], "selective kept no more than full"
    assert table.bytes < SAVE_TABLE["none"][1], "selective kept everything"
    print(f"[vram selective] projections={kept} B, table={table.bytes} B")


# ------------------------------------------------------------ 4. the refusals


def _loud(fn, *contains: str) -> str:
    try:
        fn()
    except LoudError as e:
        message = str(e)
        for needle in contains:
            assert needle in message, (needle, message)
        assert "Escape:" in message, message
        return message
    raise AssertionError(f"expected LOUD({contains}), got no refusal")


def test_every_new_refusal_fires():
    _loud(lambda: check_mode("half"), "activation_recompute", "selective")
    _loud(
        lambda: validate_train(TrainConfig(activation_recompute="full", full_graph=True)),
        "activation_recompute",
        "full_graph",
    )
    _loud(
        lambda: validate_train(TrainConfig(activation_recompute="selective", opt_graph=True)),
        "activation_recompute",
        "opt_graph",
    )
    _loud(
        lambda: validate_train(TrainConfig(iter_checkpoint=True, activation_recompute="selective")),
        "iter_checkpoint",
        "selective",
    )
    model = ModelConfig(**CFG, use_byteflow=True)
    _loud(
        lambda: validate_model_train(model, TrainConfig(activation_recompute="selective")),
        "use_byteflow",
    )
    _loud(
        lambda: validate_model_train(ModelConfig(**CFG), TrainConfig(activation_offload=4.0)),
        "bf16=false",
    )
    # A hand-built model reaches the same refusals at the runtime seam.
    hand = DormouseModel(ModelConfig(**CFG))
    _loud(
        lambda: hand.configure_actmem(
            iter_checkpoint=False, activation_offload=4.0, activation_recompute="none"
        ),
        "bf16=false",
    )
    _loud(
        lambda: hand.configure_actmem(
            iter_checkpoint=True, activation_offload=0.0, activation_recompute="selective"
        ),
        "iter_checkpoint",
    )


def test_the_offload_budget_refuses_an_overflow():
    """A later forward whose selected set overflows the budget is LOUD."""
    mech = ActivationOffload(64 / GIB)
    mech.threshold = 1
    mech._profiled = True
    try:
        with mech.region(Counters()):
            x = torch.randn(64, requires_grad=True)
            (x * x).sum()  # `mul` saves BOTH inputs: 2 × 256 B > 64 B
    except LoudError as e:
        assert "exceeded the budget" in str(e) and "Escape:" in str(e), str(e)
    else:  # pragma: no cover
        raise AssertionError("an over-budget forward was silently partial")
    print("[vram offload] overflow refused")


def test_the_offload_bytes_are_exact_and_reused():
    """The counter, the mechanism and the budget agree; the pool is reused."""
    torch.manual_seed(0)
    model = DormouseModel(ModelConfig(**{**CFG, "bf16": True}))
    model.configure_actmem(iter_checkpoint=False, activation_offload=0.05)
    ids, targets = _ids()
    records = []
    for _ in range(4):
        counters = Counters()
        _, loss, counters = model(ids, targets, counters)
        loss.backward()
        model.zero_grad(set_to_none=True)
        records.append((dict(counters), model._act_offload.stats()))
    mech = model._act_offload
    assert mech is not None
    parked = [c.get("offload_forward_bytes", 0) for c, _ in records[1:]]
    assert all(n > 0 for n in parked), parked
    for (counters, stats), nbytes in zip(records[1:], parked, strict=True):
        assert nbytes == stats["last_forward_bytes"], (nbytes, stats)
        assert nbytes == counters["offload_bytes"], (nbytes, counters)
        assert nbytes <= stats["budget_bytes"], (nbytes, stats)
        assert stats["live"] == 0, stats
    assert mech.stats()["reuses"] > 0, mech.stats()
    assert mech.stats()["allocations"] == mech.stats()["pool_tensors"], mech.stats()
    print(
        f"[vram offload] parked={parked} reuse={mech.stats()['reuses']} "
        f"peak={mech.stats()['peak_forward_bytes']}"
    )


# ---------------------------------------------------- 5. the optimizer launches


class _TinyModel(nn.Module):
    """A mixed Muon/AdamW tree with the shape groups the medium model has."""

    def __init__(self) -> None:
        super().__init__()
        for i in range(4):
            self.register_parameter(f"w{i}", nn.Parameter(torch.randn(8, 8)))
        for i in range(4):
            self.register_parameter(f"b{i}", nn.Parameter(torch.randn(8)))


class _Meter(TorchDispatchMode):
    def __init__(self) -> None:
        self.count = 0
        self.local_reads = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1
        if func is torch.ops.aten._local_scalar_dense.default:
            self.local_reads += 1
        return func(*args, **(kwargs or {}))


def _seed_grads(model: nn.Module, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=gen)


def _two_models() -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(0)
    a, b = _TinyModel(), _TinyModel()
    b.load_state_dict(a.state_dict())
    return a, b


def test_the_fused_adamw_is_bit_identical_and_cheaper():
    """The fast path vs torch's walk: same bits, measured fewer dispatches."""
    a, b = _two_models()
    fast = MixOptimizer(a, lr=1e-3, wd=0.01, opt="mix")
    slow = MixOptimizer(b, lr=1e-3, wd=0.01, opt="mix")
    slow._fast_adamw_ok = lambda: False  # type: ignore[method-assign]
    for step in range(4):
        _seed_grads(a, step)
        _seed_grads(b, step)
        flag = torch.tensor(0, dtype=torch.int32) if step == 2 else None
        if step == 2:
            # The trainer sanitizes the gradients before the gated step; do the
            # same so both paths see exactly the gated contract.
            for p in a.parameters():
                p.grad.zero_()
            for p in b.parameters():
                p.grad.zero_()
        fast.step(flag)
        slow.step(flag)
        for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
            assert torch.equal(pa, pb), f"step {step} moved a parameter"
    for name, pa in a.named_parameters():
        sa = fast.adamw.state.get(pa)
        pb = dict(b.named_parameters())[name]
        sb = slow.adamw.state.get(pb)
        if sa is None:
            assert sb is None
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(sa[key], sb[key]), (name, key)
        assert float(sa["step"]) == float(sb["step"]), name

    # The measured drop: the same step through the two paths.
    a, b = _two_models()
    fast = MixOptimizer(a, lr=1e-3, wd=0.01, opt="mix")
    slow = MixOptimizer(b, lr=1e-3, wd=0.01, opt="mix")
    slow._fast_adamw_ok = lambda: False  # type: ignore[method-assign]
    fast.step()
    slow.step()
    _seed_grads(a, 7)
    _seed_grads(b, 7)
    m_fast, m_slow = _Meter(), _Meter()
    with m_fast:
        fast.step()
    with m_slow:
        slow.step()
    assert m_fast.count < m_slow.count, (m_fast.count, m_slow.count)
    print(f"[vram optim] dispatches fast={m_fast.count} torch={m_slow.count}")
    assert m_fast.local_reads == 0, f"{m_fast.local_reads} host step reads"


def test_clip_grads_is_one_fused_norm_and_one_fused_scale():
    """The bf16 path must not cast one norm per tensor."""
    torch.manual_seed(0)
    params = [nn.Parameter(torch.randn(8, dtype=torch.bfloat16)) for _ in range(24)]
    for p in params:
        p.grad = torch.randn_like(p)
    meter = _Meter()
    with meter:
        clip_grads(params, 1.0)
    assert meter.count <= 10, f"{meter.count} dispatches for 24 bf16 gradients"
    assert meter.local_reads == 0, "clip_grads read a scalar to the host"
    print(f"[vram clip] {meter.count} dispatches for 24 bf16 gradients")


def test_the_recompute_and_optimizer_steps_do_not_sync():
    """No `item`/`tolist`/`bool`/`float` in the new forward or optimizer paths."""
    names = ("item", "tolist", "bool", "__float__")
    saved = {n: getattr(torch.Tensor, n) for n in names}

    def boom(self, *a, **k):  # noqa: ANN001
        raise AssertionError("host sync in a VRAM arm")

    try:
        for n in names:
            setattr(torch.Tensor, n, boom)
        for mode in ("full", "selective"):
            # `moe_topk=0`: the CPU MoE's `_naive_dispatch` has a pre-existing
            # `.tolist()` (reported as a follow-up); this gate is scoped to the
            # paths this branch changed, as `test_cheapwidth`'s gate 7 is.
            _one_step(mode, over={**CFG, "moe_topk": 0})
        a, _ = _two_models()
        opt = MixOptimizer(a, lr=1e-3, wd=0.01, opt="mix")
        _seed_grads(a)
        opt.step()
        clip_grads(a.parameters(), 1.0)
    finally:
        for n, f in saved.items():
            setattr(torch.Tensor, n, f)
    print("[vram sync] no host read in the changed paths")
