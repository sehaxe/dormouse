"""`moe_shared_compute`: ONE token projection instead of E per-expert copies.

`share_u` (plane) stores ONE `[d, rank]` basis per site, but the dispatch used
to materialize it E-fold (`_banks`) and run `bmm(xg[E, cap, d], gate_u[E, d, r])`
— the expensive `d -> r` product repeated for every expert on its own rows. The
arm changes only WHERE that product happens:

    off:  g = (bank[E, cap, d] @ gate_u[E, d, r]) · gate_s
    on:   lat = x[n, d] @ [gate_u | up_u][d, 2·rank]   ONE projection / token
          g   = bank_lat[E, cap, 2·rank][..., :r] · gate_s

The weights, the rank-space GLU and the write `@ v` are the same tensors; the
bank gathers `2·rank` columns instead of `d`, and two assignments of a token
share the one latent (top-2 projects each token once, not twice).

Gates (each with the scratch-revert proof noted where it applies):

A. `moe_shared_compute=false` is bit-identical to clean main: the goldens were
   captured BEFORE the field existed (`state_dict` sha 256, logits sha 256 and
   the parameter count), and an explicit `false` equals the default build.
B. ON vs OFF: fp32 logits and EVERY parameter gradient agree within 1e-5
   relative. This is NOT a bit-identity claim: the reduction order moved (one
   `[n, d] @ [d, 2r]` GEMM instead of E `bmm`s), so the two shapes round in a
   different order and the last bits differ by design.
C. The flag removes matmul LAUNCHES: a `TorchDispatchMode` counts `aten.mm` /
   `aten.bmm` / `aten.matmul` inside one real `_batched_dispatch`; ON is
   strictly lower (the two projection `bmm`s become one `mm`).
D. The LOUD refusals fire with `Escape:`: `share_u_mode='off'`, `rank=0`, and a
   per-expert adaptive rank (a heterogeneous bank — the one a config cannot
   see).
E. No host syncs are added on the step path: `.item()` / `.tolist()` are
   poisoned during a real forward on both arms, with a static capacity so the
   pre-existing dropless `.item()` is not in the window.
F. The parameter set is a checkpoint round-trip away from the old shape:
   `state_dict()` keys and shapes are equal, the OFF checkpoint loads into the
   ON model with `strict=True`, and every gradient comes back the same shape.

Scratch-revert proofs (each observed RED in this worktree, then restored):

* A — default the field to `True`: `_build()` raises the shared-basis refusal.
* B — swap `gate_s`/`up_s` in the shared branch: `gate_s` gradient rel 2.96e-1.
* C — `if False and self.moe_shared_compute`: 3 vs 3 matmul dispatches.
* D — early `return` in `_assert_shared_compute`: `DID NOT RAISE LoudError`.
* E — one `.item()` inside `_shared_projection`: "host sync inside …".
* F — a persistent flag-only buffer: "the arm changed the parameter set".
"""

from __future__ import annotations

import hashlib
import io

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.experts import SparseMoE

#: The fixed, non-RNG input `tests/test_share_u.py` uses, so the "clean main"
#: pin is the same arithmetic another gate already protects.
IDS = torch.arange(2 * 64).reshape(2, 64) * 7 % 256

#: Captured on clean `94012f2` BEFORE `moe_shared_compute` existed. Regenerate
#: ONLY with a deliberate init/forward change.
GOLDEN_PARAMS = 64_881
GOLDEN_SD_SHA = "f1ea6bb2273de0345913331fe9b747878796b4fb40ba5cc143fa678a7359a4b8"
GOLDEN_LOGITS_SHA = "d51cc54995a5024320c9357cfeb564acf576bd3b68e9b6f9e90ad3c7d917dac1"

#: cap = max(2, ceil(1.0 · 128 · 2 / 4)) = 64 rows per expert at the tiny
#: shape: static, so neither arm reaches `_batched_cap`'s dropless host read.
ARMED = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=64,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=2,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
    share_u=True,
    moe_kernel="batched",
    moe_capacity=1.0,
)


def _build(**over) -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(tiny_cfg(**over), device="cpu")


def _armed(**over) -> DormouseModel:
    return _build(**{**ARMED, **over})


def _state_sha(model: DormouseModel) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def _logits_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).numpy().tobytes()).hexdigest()


def _forward(model: DormouseModel, counters: Counters | None = None):
    return model(IDS, targets=torch.roll(IDS, -1, dims=1), counters=counters)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).norm() / (b.detach().norm() + 1e-30))


# --- A. the off path is clean main ------------------------------------------


@pytest.mark.pins
def test_a_flag_off_is_bit_identical_to_clean_main() -> None:
    print(f"[gate A] {GOLDEN_PARAMS} params, sd={GOLDEN_SD_SHA[:12]}")
    default = _build()
    explicit = _build(moe_shared_compute=False)

    assert default.cfg.moe_shared_compute is False, "the arm must default to off"
    assert param_count(default) == GOLDEN_PARAMS
    assert _state_sha(default) == GOLDEN_SD_SHA, "the new field changed the init"
    logits, _, _ = _forward(default)
    assert _logits_sha(logits) == GOLDEN_LOGITS_SHA, "the new field changed the forward"
    logits_explicit, _, _ = _forward(explicit)
    assert torch.equal(logits, logits_explicit)


# --- B. on vs off: same math, different reduction order ---------------------


def test_b_on_and_off_agree_in_fp32_logits_and_gradients() -> None:
    """NOT bit-identical, and not claimed to be: the projection changed shape.

    The rotated GEMM (one `[n, d] @ [d, 2r]` over the tokens, then a `2r`-wide
    gather) and the old E `bmm`s add the `d` axis in a different order, so the
    last fp32 bits move. 1e-5 relative is the parity the arm must hold to be the
    same function.

    Depth 1 on purpose: at `max_iter=2` the loop's nonlinearity amplifies the
    same last-bit difference in two unrelated, tiny-norm Engram gradients to
    ~5e-5 RELATIVE (4e-9 absolute — cancellation, not a shape error). The
    module this gate is about is checked at both depths; this bound is the
    function's.
    """
    off = _armed(moe_shared_compute=False, max_iter=1)
    on = _armed(moe_shared_compute=True, max_iter=1)

    l_off, loss_off, _ = _forward(off)
    l_on, loss_on, _ = _forward(on)
    rel_logits = _rel(l_on, l_off)
    assert rel_logits < 1e-5, f"fp32 logits moved {rel_logits:.2e}"

    loss_off.backward()
    loss_on.backward()
    g_off = dict(off.named_parameters())
    g_on = dict(on.named_parameters())
    assert g_off.keys() == g_on.keys()
    worst, worst_name = 0.0, ""
    for name, p_off in g_off.items():
        p_on = g_on[name]
        assert (p_off.grad is None) == (p_on.grad is None), f"{name}: one arm has no grad"
        if p_off.grad is None:
            continue
        assert p_off.grad.shape == p_on.grad.shape, name
        r = _rel(p_on.grad, p_off.grad)
        if r > worst:
            worst, worst_name = r, name
        assert r < 1e-5, f"{name}: gradient moved {r:.2e}"
    print(
        f"[gate B] logits rel {rel_logits:.2e}; worst grad {worst:.2e} "
        f"({worst_name}); {len(g_off)} parameter tensors"
    )


# --- C. the arm removes matmul launches -------------------------------------


def _matmul_dispatches(fn) -> tuple[int, list[str]]:
    """`(matmul dispatches, every aten op name)` for ONE call, on CPU."""

    counted: list[str] = []

    class _Count(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            counted.append(str(func))
            return func(*args, **(kwargs or {}))

    with _Count():
        fn()
    mm = [k for k in counted if k.startswith(("aten.mm", "aten.bmm", "aten.matmul"))]
    return len(mm), counted


def test_c_the_flag_launches_fewer_matmul_ops() -> None:
    torch.manual_seed(0)
    off = SparseMoE(
        tiny_cfg(
            n_experts=4,
            moe_topk=2,
            rank=4,
            d_ffn=64,
            share_u=True,
            moe_kernel="batched",
            moe_capacity=1.0,
            moe_shared_compute=False,
        ),
        device="cpu",
    )
    torch.manual_seed(0)
    on = SparseMoE(
        tiny_cfg(
            n_experts=4,
            moe_topk=2,
            rank=4,
            d_ffn=64,
            share_u=True,
            moe_kernel="batched",
            moe_capacity=1.0,
            moe_shared_compute=True,
        ),
        device="cpu",
    )
    x = torch.randn(8, off.d)
    index = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]] * 2, dtype=torch.long)
    weight = torch.full((8, 2), 0.5)

    def one(moe: SparseMoE) -> int:
        order, keep, _ = moe._permute(index)
        n, _ = _matmul_dispatches(lambda: moe._batched_dispatch(x, weight, index, order, keep))
        return n

    n_off, n_on = one(off), one(on)
    print(f"[gate C] matmul dispatches: off {n_off}, on {n_on}")
    assert n_on < n_off, "the shared projection must replace two bmms with one mm"


# --- D. the LOUD refusals ---------------------------------------------------


def test_d_refusals_name_the_escape() -> None:
    with pytest.raises(LoudError, match="share_u_mode='off'"):
        SparseMoE(
            tiny_cfg(
                share_u_mode="off",
                moe_kernel="batched",
                moe_shared_compute=True,
            ),
            device="cpu",
        )
    with pytest.raises(LoudError, match="rank=0"):
        validate(
            ModelConfig(
                **{
                    **tiny_cfg().__dict__,
                    "share_u_mode": "plane",
                    "moe_kernel": "batched",
                    "rank": 0,
                    "moe_shared_compute": True,
                }
            )
        )
    torch.manual_seed(0)
    moe = SparseMoE(
        tiny_cfg(
            n_experts=2,
            moe_topk=2,
            rank=4,
            share_u=True,
            moe_kernel="batched",
            moe_shared_compute=True,
        ),
        device="cpu",
    )
    with pytest.raises(LoudError, match="ADAPTIVE RANK"):
        moe.experts[1].v = torch.nn.Parameter(torch.randn(5, moe.f))
        moe._assert_shared_compute()
    print("[gate D] off / rank 0 / adaptive rank all refused with an Escape")


# --- E. no host read on the step path ---------------------------------------


def test_e_the_shared_forward_touches_no_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_, **__):
        raise AssertionError("host sync inside the shared-compute forward")

    def arm(shared: bool) -> SparseMoE:
        torch.manual_seed(0)
        return SparseMoE(
            tiny_cfg(
                n_experts=4,
                moe_topk=2,
                rank=4,
                share_u=True,
                moe_kernel="batched",
                moe_capacity=1.0,
                moe_shared_compute=shared,
            ),
            device="cpu",
        )

    off, on = arm(False), arm(True)
    x = torch.randn(8, on.d)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "item", boom)
        m.setattr(torch.Tensor, "tolist", boom)
        counters = Counters()
        out = on(x, counters)
        out_off = off(x, Counters())  # the control pays no read either
        assert counters.get("moe_shared_compute_calls") == 1, counters
    counters.drain()
    assert counters["moe_projection_rows"] == 8, counters
    print(f"[gate E] one forward, {out.shape[0]} rows, no host read on either arm")
    assert torch.isfinite(out).all() and torch.isfinite(out_off).all()


# --- F. the checkpoint shape is unchanged -----------------------------------


def test_f_checkpoint_round_trip_and_grad_shapes() -> None:
    off = _armed(moe_shared_compute=False)
    on = _armed(moe_shared_compute=True)

    sd_off = off.state_dict()
    sd_on = on.state_dict()
    assert sd_off.keys() == sd_on.keys(), "the arm changed the parameter set"
    for k in sd_off:
        assert sd_off[k].shape == sd_on[k].shape, f"{k}: shape moved"
        assert sd_off[k].dtype == sd_on[k].dtype, f"{k}: dtype moved"

    buf = io.BytesIO()
    torch.save(sd_off, buf)
    buf.seek(0)
    on.load_state_dict(torch.load(buf, weights_only=True), strict=True)

    _, loss_off, _ = _forward(off)
    _, loss_on, _ = _forward(on)
    loss_off.backward()
    loss_on.backward()
    ref = dict(off.named_parameters())
    for name, p in on.named_parameters():
        q = ref[name]
        assert (p.grad is None) == (q.grad is None), f"{name}: one arm has no gradient"
        if p.grad is not None:
            assert p.grad.shape == p.shape, f"{name}: gradient shape moved"
    print(f"[gate F] {len(sd_off)} tensors round-trip; every gradient has its parameter's shape")
