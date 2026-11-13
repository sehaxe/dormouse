"""`tsct_packed`: ONE buffer per expert site, bit-identical to the E parameters.

The 2026-11-05 census priced the bank rebuild (`SparseMoE._banks`, once per
optimizer step) at 519 launches: 512 experts are 512 separate `nn.Parameter`s,
so the stack is 512 `.to(dt)` casts plus a `stack`, and the m400b recipe pays it
three times per step (one per loop iteration; the cache serves the rest). This
arm stores the same numbers as ONE `[E, rank, d_ffn]` master for the write `v`
(and one `[E, d_model, rank]` per input site when `share_u` is off) plus one
`[E, 2, rank]` scale buffer, and makes `_banks` cast the whole buffer once.

Gates, each with the scratch-revert proof where it applies:

A. `tsct_packed=false` is BIT-IDENTICAL to clean main: `state_dict` sha256,
   logits/loss and every parameter gradient were pinned BEFORE the edit
   (`torch.set_num_threads(1)`, `torch.manual_seed(0)`).
B. Packed vs unpacked is `torch.equal` on the BANK (`_banks`'s five tensors, in
   fp32 and bf16), on the LOGITS/loss and on EVERY parameter gradient. The
   packed model is loaded from the unpacked one first, so the comparison is the
   same weights, not the same seed.
C. The checkpoint is readable both ways: an old per-expert state dict gathers
   into the packed buffers, a packed state dict expands into the per-expert
   ones, and a third hop reproduces the original values. `state_dict(packed_form=)`
   writes either form on demand.
D. The rebuild drops >=100x in launches: E=512 tiny shapes, one `_banks` build
   per arm, `scripts/launch_profile.DispatchCounter` (the repo's CPU instrument).
E. Muon-on-3D is explicit: a marked `[E, rows, cols]` parameter is routed to
   Muon and stepped per slice (bit-equal to E 2-D parameters on this fixture);
   an unmarked 3-D tensor stays AdamW; the per-parameter `Muon.step` and the
   SF/NorMuon tails REFUSE the packed shape loudly.
F. The starvation driver's pre-registered rule rejects a tie (and the dry run
   prints the resolved arms / tokens-per-expert / compute table).

Scratch-revert proofs (each observed RED in this worktree, then restored):

* A — default `tsct_packed=True`: the pinned digests move (the state keys change).
* B — serve `packed_s[:, 0]` to the gate slot: bank/`logits` equality reddens.
* C — translate only `packed_v` (drop the scales): the old->packed load raises
  `Missing key(s)`.
* D — remove the packed branch in `_one` (always stack): 521 vs 521 launches.
* E — unmark `_muon_batched`: the packed buffer routes to AdamW and the update
  gate reddens.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.config import ModelConfig
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.experts import SparseMoE
from dormouse.train.optim import Muon, is_muon_param

ROOT = Path(__file__).resolve().parents[1]

#: The fixed tiny recipe the pins were captured on. `share_u="plane"` and
#: `moe_kernel="batched"` are the m400b arm set, shrunk to d=64.
BASE = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=32,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=2,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
    use_kda=True,
    use_engram=False,
    use_msa=False,
    use_csa=False,
    share_u=True,
    moe_kernel="batched",
    moe_bank_epoch=True,
)
IDS = torch.arange(2 * 64).reshape(2, 64) * 7 % 256

#: Captured on clean `907b3b5` BEFORE the field existed (the first pin script in
#: this worktree): params, sha256(logits+loss+every grad), sha256(state dict).
#:
#: RE-CAPTURED 2026-11-10 for `SparseMoE._bank_dispatch`'s router-weight layout
#: (the weight is applied in rank space before `@ v` instead of scaling a
#: `[n·k, f]` fp32 product). That is a deliberate change to the OFF forward's
#: rounding, so this alarm MOVED — and it moved in the one place the change
#: touches: the logits differ (review measured max|Δ| 7.15e-07, rel 1.67e-07,
#: 27,670 of 32,768 elements) while `state` (no init bit) and `loss` (rounded at
#: its own scale) did not.
#:
#: THIS DIGEST IS A DRIFT ALARM, NOT PROOF. A digest captured from our own run
#: cannot say the layout is right; it says nothing moved since yesterday.
#: The PROOF is `test_the_moe_slot_matches_an_external_float64_implementation`
#: (logits AND dL/dweight against a second, float64 implementation of the same
#: module) and `test_the_hand_computed_two_token_case` (two tokens, two rank-1
#: experts, `w = [0.75, 0.25]`, expected output written on paper).
GOLDEN_PARAMS = 55_025
GOLDEN_GRAD = "256f0dde88fc62d086cf906a6268e30fa73756586ac3d8822451dc05a3df9227"
GOLDEN_STATE = "537126066f1b70c4f40b05773be323a27221690e74a6cfd1040408493a18460b"
GOLDEN_LOSS = 5.939486026763916


def _build(**over) -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(ModelConfig(**{**BASE, **over}), device="cpu").to("cpu").train()


def _forward(model: DormouseModel, counters: Counters | None = None):
    return model(IDS, targets=torch.roll(IDS, -1, dims=1), counters=counters or Counters())


def _digest(cfg: ModelConfig) -> tuple[str, str, float]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _digest_at_one_thread(cfg)
    finally:
        torch.set_num_threads(threads)


def _digest_at_one_thread(cfg: ModelConfig) -> tuple[str, str, float]:
    torch.manual_seed(0)
    model = DormouseModel(cfg, device="cpu").to("cpu").train()
    model.zero_grad(set_to_none=True)
    logits, loss, _ = _forward(model)
    loss.backward()
    grad = hashlib.sha256()
    for t in (logits, loss.reshape(1)):
        grad.update(t.detach().contiguous().numpy().tobytes())
    for name, p in sorted(model.named_parameters()):
        if p.grad is not None:
            grad.update(name.encode())
            grad.update(p.grad.detach().contiguous().numpy().tobytes())
    state = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        state.update(name.encode())
        state.update(t.detach().contiguous().numpy().tobytes())
    return grad.hexdigest(), state.hexdigest(), float(loss.detach())


# --------------------------------------------------------------------------- #
# A. OFF is clean main — the pins were taken before the edit
# --------------------------------------------------------------------------- #
@pytest.mark.pins
def test_off_is_bit_identical_to_clean_main() -> None:
    """Gate A: the OFF arm matches its 2026-11-10 re-captured alarm (DRIFT only).

    Read the constant's comment: the digest moved with the router-weight layout,
    so this test is a regression alarm. What proves that move is CORRECT is the
    float64 anchor and the hand-computed case below, not this equality.
    """
    model = _build()
    assert model.cfg.tsct_packed is False, "the arm must default to off"
    assert param_count(model) == GOLDEN_PARAMS
    grad, state, loss = _digest(ModelConfig(**BASE))
    print(f"[gate A] {GOLDEN_PARAMS:,} params, grad={grad[:12]} state={state[:12]}")
    assert grad == GOLDEN_GRAD, f"the OFF gradient/logits path moved: got {grad}"
    assert state == GOLDEN_STATE, f"the OFF state dict moved: got {state}"
    assert loss == GOLDEN_LOSS


# --------------------------------------------------------------------------- #
# ------------------------------------------------- A2. the external anchors
#
# The digest above is an alarm, so the layout needs an authority OUTSIDE our own
# run. Two, at different granularity: (i) a second, float64 implementation of the
# whole `SparseMoE` module (router → top-k → routed GLU → shared expert → down),
# compared on logits AND `dL/dweight`; (ii) a two-token fixture whose expected
# output is arithmetic on paper.
#
# NOT re-derived: the model's VOCAB logits in float64. That needs a second
# implementation of KDA/attention/engram, which is a different project; the
# module that changed is the MoE slot, and these anchors cover it at its own
# boundary (`SparseMoE.forward`'s output, and the gradient reaching its `weight`).


def _glu64(expert, x64: torch.Tensor) -> torch.Tensor:
    """One `DualLatentGLU` forward in float64, straight from the parameter names."""
    g = (x64 @ expert.gate_u.detach().double()) * expert.gate_s.detach().double()
    u = (x64 @ expert.up_u.detach().double()) * expert.up_s.detach().double()
    return (F.silu(g) * u) @ expert.v.detach().double()


def _moe64(moe: SparseMoE, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`SparseMoE.forward` in float64: `(output, weight-for-grad, index)`.

    Deliberately a SECOND transcription (autograd included) of the module: the
    router, the LFB-raised top-k, the per-assignment weighted GLU, the always-on
    shared expert and the `down` projection. If the bank's layout is wrong, this
    and the module disagree — a digest cannot say that.
    """
    x64 = x.detach().double()
    logits = x64 @ moe.router.weight.detach().double().T
    probs = torch.softmax(logits, -1)
    index = (logits + moe.lfb_bias.detach().double()).topk(moe.topk, dim=-1).indices
    w = torch.gather(probs, 1, index)
    w = (w / w.sum(-1, keepdim=True).clamp_min(1e-9)).requires_grad_()
    out = torch.zeros(x64.shape[0], moe.f, dtype=torch.float64)
    for e in range(moe.n_experts):
        for slot in range(moe.topk):
            sel = torch.nonzero(index[:, slot] == e).reshape(-1)
            if sel.numel() == 0:
                continue
            y = _glu64(moe.experts[e], x64[sel]) * w[sel, slot].unsqueeze(-1)
            out = out.index_add(0, sel, y)
    out = out + _glu64(moe.shared, x64)
    return out @ moe.down.weight.detach().double().T, w, index


def test_the_moe_slot_matches_an_external_float64_implementation() -> None:
    """Anchor A2(i): the MoE module's logits AND `dL/dweight` vs float64.

    The pin's digest was re-captured because `SparseMoE._bank_dispatch` moved the
    router weight into rank space (2026-11-10). This is what says that move is
    the same function: the module and an independent float64 transcription agree
    on its output to the fp32 bound, and on the gradient reaching the routed
    weight (`dL/dweight`) to the same bound. Measured: output rel 2.7e-07,
    `dL/dweight` rel 1.2e-07, routing indices identical.
    """
    torch.manual_seed(4)
    moe = SparseMoE(
        tiny_cfg(
            n_experts=4,
            moe_topk=2,
            d_model=32,
            d_ffn=48,
            head_dim=8,
            rank=4,
            bf16=False,
            moe_capacity=0.0,
            moe_kernel="batched",
        ),
        device="cpu",
    ).train()
    x = torch.randn(19, moe.d)
    # Capture the NORMALIZED weight the changed code receives, with its gradient.
    seen: dict[str, torch.Tensor] = {}
    real = moe._dispatch

    def spy(xin, w, idx, counters):
        w.retain_grad()
        seen["w"] = w
        seen["i"] = idx
        return real(xin, w, idx, counters)

    moe._dispatch = spy
    out = moe(x)
    cot = torch.randn_like(out)
    (out * cot).sum().backward()
    assert "w" in seen, "the spy never saw the routed weight"

    ref, w64, idx64 = _moe64(moe, x)
    (ref * cot.double()).sum().backward()
    d_out = float((out.detach().double() - ref.detach()).abs().max())
    rel_out = d_out / float(ref.detach().abs().max())
    d_w = float((seen["w"].grad.double() - w64.grad).abs().max())
    rel_w = d_w / float(w64.grad.abs().max())
    print(
        f"[fp64 anchor] out max|Δ|={d_out:.3e} rel={rel_out:.3e} "
        f"dL/dweight max|Δ|={d_w:.3e} rel={rel_w:.3e} bound=1e-5 "
        f"routing_identical={torch.equal(idx64, seen['i'])}"
    )
    assert torch.equal(idx64, seen["i"]), "the two implementations routed differently"
    assert rel_out <= 1e-5, f"logits moved: {rel_out:.3e}"
    assert rel_w <= 1e-5, f"dL/dweight moved: {rel_w:.3e}"


def test_the_hand_computed_two_token_case() -> None:
    """Anchor A2(ii): two tokens, two rank-1 experts, `w = [0.75, 0.25]` — ON PAPER.

    Fixture: d=2, rank=1, f=2. Expert 0 reads `h = silu(x0)·x1` with `v = [1,1]`;
    expert 1 reads `h = silu(x1)·x0` with `v = [0,2]`; both scales are 1.

    `x = [[1,1], [2,3]]`, both tokens route to both experts, weights 0.75/0.25.

        token 0: h0 = silu(1)·1 = 0.7310585786, h1 = silu(1)·1 = same
                 o = 0.75·[h,h] + 0.25·[0,2h] = [0.75h, 1.25h]
                   = [0.5482939339, 0.9138232233]
        token 1: h0 = silu(2)·3 = 5.2847824679, h1 = silu(3)·2 = 5.7154447609
                 o = 0.75·[h0,h0] + 0.25·[0,2h1] = [0.75h0, 0.75h0+0.5h1]
                   = [3.9635868509, 6.8213092314]

    (`silu(z) = z/(1+e^-z)`: 0.7310585786, 1.7615941560, 2.8577223805.)
    Measured max|Δ| against those literals: 0.0 — the same seven digits.
    """
    torch.manual_seed(0)
    moe = SparseMoE(
        tiny_cfg(
            n_experts=2,
            moe_topk=2,
            d_model=2,
            d_ffn=2,
            head_dim=1,
            rank=1,
            bf16=False,
            moe_capacity=0.0,
            moe_kernel="batched",
        ),
        device="cpu",
    )
    with torch.no_grad():
        e0, e1 = moe.experts
        e0.gate_u.copy_(torch.tensor([[1.0], [0.0]]))
        e0.up_u.copy_(torch.tensor([[0.0], [1.0]]))
        e0.v.copy_(torch.tensor([[1.0, 1.0]]))
        e1.gate_u.copy_(torch.tensor([[0.0], [1.0]]))
        e1.up_u.copy_(torch.tensor([[1.0], [0.0]]))
        e1.v.copy_(torch.tensor([[0.0, 2.0]]))
        for e in (e0, e1):
            e.gate_s.fill_(1.0)
            e.up_s.fill_(1.0)
    x = torch.tensor([[1.0, 1.0], [2.0, 3.0]])
    index = torch.tensor([[0, 1], [0, 1]])
    weight = torch.tensor([[0.75, 0.25], [0.75, 0.25]])
    got = moe._dispatch(x, weight, index, Counters())
    want = torch.tensor(
        [
            [0.5482939339725037, 0.9138232232875061],
            [3.9635868509004706, 6.8213092313677710],
        ]
    )
    delta = float((got.detach() - want).abs().max())
    print(f"[hand case] got={got.tolist()} max|Δ|={delta:.3e} atol=1e-6")
    assert delta <= 1e-6, f"{delta:.3e} — the paper arithmetic and the bank disagree"


# B. packed == unpacked, bit for bit
# --------------------------------------------------------------------------- #
def _pair(share: str) -> tuple[DormouseModel, DormouseModel]:
    """An unpacked model and a packed TWIN (loaded from it, so same weights)."""
    over = {"share_u": False, "share_u_mode": share} if share != "plane" else {}
    unpacked = _build(**over)
    packed = _build(tsct_packed=True, **over)
    packed.load_state_dict(unpacked.state_dict(), strict=True)
    return unpacked, packed


@pytest.mark.parametrize("share", ["plane", "off"])
@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_the_bank_is_torch_equal(share: str, dt: torch.dtype) -> None:
    unpacked, packed = _pair(share)
    mu, mp = unpacked.loop.moe, packed.loop.moe
    # Distinct per-site scales, so a swapped gate_s/up_s cannot hide behind the
    # all-ones initialisation.
    with torch.no_grad():
        for e, ex in enumerate(mu.experts):
            ex.gate_s.fill_(1.0 + e)
            ex.up_s.fill_(0.5 - e)
    packed.load_state_dict(unpacked.state_dict(), strict=True)
    mu._bank_cache = None
    mp._bank_cache = None
    a = mu._banks(dt)
    b = mp._banks(dt)
    assert len(a) == len(b) == 5, "gate_u, up_u, v, gate_s, up_s"
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        assert isinstance(y, torch.Tensor)
        assert torch.equal(x, y), f"bank tensor {i} moved on the packed arm ({dt})"
    # The packed scales are views of ONE `[E, 2, rank]` buffer, not two arrays.
    assert mp.packed_s.shape == (mp.n_experts, 2, mp.experts[0].rank)
    assert torch.equal(mp.packed_s[:, 0], a[3])
    assert torch.equal(mp.packed_s[:, 1], a[4])


@pytest.mark.parametrize("share", ["plane", "off"])
def test_logits_and_every_gradient_are_torch_equal(share: str) -> None:
    unpacked, packed = _pair(share)
    unpacked.zero_grad(set_to_none=True)
    packed.zero_grad(set_to_none=True)
    lu, lossu, _ = _forward(unpacked)
    lp, lossp, _ = _forward(packed)
    assert torch.equal(lu, lp), "the packed forward moved the logits"
    assert torch.equal(lossu.detach(), lossp.detach())
    (lossu * 1.0).backward()
    lossp.backward()

    mu, mp = unpacked.loop.moe, packed.loop.moe
    for e in range(mu.n_experts):
        pairs = [
            (mu.experts[e].v, mp.packed_v, (e,)),
            (mu.experts[e].gate_s, mp.packed_s, (e, 0)),
            (mu.experts[e].up_s, mp.packed_s, (e, 1)),
        ]
        if hasattr(mp, "packed_gate_u"):
            pairs += [
                (mu.experts[e].gate_u, mp.packed_gate_u, (e,)),
                (mu.experts[e].up_u, mp.packed_up_u, (e,)),
            ]
        for src, base, idx in pairs:
            dst = None if base.grad is None else base.grad[idx]
            assert (src.grad is None) == (dst is None), (
                f"{tuple(src.shape)}: one arm has a gradient and the other does not"
            )
            if src.grad is not None:
                assert torch.equal(src.grad, dst), f"gradient moved for {tuple(src.shape)}"

    # Every non-bank parameter, by name.
    g_u = {n: p.grad for n, p in unpacked.named_parameters() if "moe.experts" not in n}
    g_p = {n: p.grad for n, p in packed.named_parameters() if "moe.packed" not in n}
    assert set(g_u) == set(g_p)
    for name, grad in g_u.items():
        assert (grad is None) == (g_p[name] is None), name
        if grad is not None:
            assert torch.equal(grad, g_p[name]), f"gradient moved at {name}"


# --------------------------------------------------------------------------- #
# C. the checkpoint round-trips through BOTH formats
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("share", ["plane", "off"])
def test_checkpoint_round_trip_both_ways(tmp_path: Path, share: str) -> None:
    over = {"share_u": False, "share_u_mode": share} if share != "plane" else {}
    old = _build(**over)
    old_sd = old.state_dict()
    assert f"loop.moe.experts.{old.loop.moe.n_experts - 1}.v" in old_sd

    packed = _build(tsct_packed=True, **over)
    packed.load_state_dict(old_sd, strict=True)
    moe = packed.loop.moe
    for e in range(moe.n_experts):
        assert torch.equal(moe.packed_v[e], old_sd[f"loop.moe.experts.{e}.v"])
        assert torch.equal(moe.packed_s[e, 0], old_sd[f"loop.moe.experts.{e}.gate_s"])
        assert torch.equal(moe.packed_s[e, 1], old_sd[f"loop.moe.experts.{e}.up_s"])
        if hasattr(moe, "packed_gate_u"):
            assert torch.equal(moe.packed_gate_u[e], old_sd[f"loop.moe.experts.{e}.gate_u"])
            assert torch.equal(moe.packed_up_u[e], old_sd[f"loop.moe.experts.{e}.up_u"])

    packed_sd = packed.state_dict()
    assert "loop.moe.packed_v" in packed_sd
    assert "loop.moe.experts.0.v" not in packed_sd

    # packed checkpoint -> a fresh per-expert model -> the ORIGINAL values.
    back = _build(**over)
    back.load_state_dict(packed_sd, strict=True)
    back_sd = back.state_dict()
    assert set(back_sd) == set(old_sd)
    for key, value in old_sd.items():
        assert torch.equal(back_sd[key], value), f"{key} did not survive packed -> old"

    # A real file, not just an in-memory dict.
    buf = io.BytesIO()
    torch.save(packed_sd, buf)
    buf.seek(0)
    disk = torch.load(buf, map_location="cpu", weights_only=True)
    fresh = _build(**over)
    fresh.load_state_dict(disk, strict=True)
    assert torch.equal(fresh.loop.moe.experts[0].v, old_sd["loop.moe.experts.0.v"])

    # `state_dict(packed_form=)` writes the OTHER form on demand. The switch is
    # the MoE module's (the recursive `Module.state_dict` call cannot forward a
    # custom kwarg from the root): `moe.state_dict()` keys are local to the bank.
    assert "packed_v" in old.loop.moe.state_dict(packed_form=True)
    exported = packed.loop.moe.state_dict(packed_form=False)
    assert f"experts.{moe.n_experts - 1}.v" in exported
    assert "packed_v" not in exported
    third = _build(**over)
    third.loop.moe.load_state_dict(exported, strict=True)
    assert torch.equal(third.state_dict()["loop.moe.experts.0.v"], old_sd["loop.moe.experts.0.v"])

    # Two formats in ONE dict is a wiring error, never a merge.
    with pytest.raises(LoudError, match="BOTH"):
        back.load_state_dict({**old_sd, **packed_sd}, strict=False)


# --------------------------------------------------------------------------- #
# D. the rebuild drops >= 100x in launches
# --------------------------------------------------------------------------- #
def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _count_build(**over) -> int:
    """Launches of ONE `_banks` build at E=512, tiny per-expert shapes."""
    prof = _load_script("launch_profile")
    cfg = ModelConfig(
        d_model=32,
        n_heads=2,
        head_dim=16,
        d_ffn=8,
        vocab=32,
        max_seq_len=8,
        max_iter=1,
        rank=4,
        n_experts=512,
        moe_topk=2,
        engram_rows=8,
        engram_dim=4,
        bf16=False,
        shared_rank=8,
        use_kda=False,
        use_engram=False,
        use_msa=False,
        use_csa=False,
        moe_kernel="batched",
        moe_bank_epoch=True,
        **over,
    )
    torch.manual_seed(0)
    from dormouse.model.experts import SparseMoE

    moe = SparseMoE(cfg, device="cpu")
    moe._bank_cache = None
    counter = prof.DispatchCounter()
    with counter.observe(moe):
        moe._banks(torch.bfloat16)
    return sum(counter.launches.values())


def test_the_rebuild_drops_at_least_100x_launches() -> None:
    for share in ("plane", "off"):
        unpacked = _count_build(share_u_mode=share, tsct_packed=False)
        packed = _count_build(share_u_mode=share, tsct_packed=True)
        print(f"[gate D] {share}: {unpacked} -> {packed} launches ({unpacked / packed:.1f}x)")
        assert packed <= 5, f"{share}: the packed build should be v + 2 u + 2 scales"
        assert unpacked >= 100 * packed, (
            f"{share}: only {unpacked / packed:.1f}x, the gate needs >=100x"
        )


# --------------------------------------------------------------------------- #
# E. Muon on a 3-D batch: routed, per-slice, or LOUD
# --------------------------------------------------------------------------- #
def test_muon_routes_a_marked_batch_and_refuses_elsewhere() -> None:
    packed = torch.randn(4, 8, 16, requires_grad=True)
    packed._muon_batched = True
    plain = packed.detach().clone().requires_grad_(True)
    assert is_muon_param("loop.moe.packed_v", packed)
    assert not is_muon_param("loop.moe.packed_v", plain), (
        "an unmarked 3-D tensor must stay AdamW (the pre-arm behaviour)"
    )

    # The MODEL must set the marker: the routing table is the contract.
    from dormouse.train.optim import MixOptimizer

    model = _build(tsct_packed=True)
    moe = model.loop.moe
    assert all(getattr(p, "_muon_batched", False) for p in (moe.packed_v,)), (
        "the packed write lost its Muon marker"
    )
    assert not getattr(moe.packed_s, "_muon_batched", False), "scales are not matrices"
    routing = MixOptimizer(model, lr=1e-3, wd=0.0).routing
    assert routing["loop.moe.packed_v"] == "muon"
    assert routing["loop.moe.packed_s"] == "adamw"

    grad = torch.randn_like(packed)
    with torch.no_grad():
        packed.copy_(grad)
    packed.grad = grad.clone()
    opt = Muon([packed], lr=0.05, ns_steps=3, momentum=0.9)
    opt.batched_step_from_model(None)
    grp = opt._groups[0]
    assert grp.packed and grp.rows == 4
    assert grp.mom.shape == packed.shape
    assert opt.state[packed]["momentum"] is grp.mom

    # The per-slice definition: the same update as four independent 2-D params.
    rows = [torch.nn.Parameter(grad[i].clone()) for i in range(4)]
    for i, row in enumerate(rows):
        row.grad = grad[i].clone()
    ref = Muon(rows, lr=0.05, ns_steps=3, momentum=0.9)
    ref.batched_step_from_model(None)
    for i, row in enumerate(rows):
        assert torch.equal(packed.detach()[i], row.detach()), (
            f"packed slice {i} differs from the per-expert Muon step"
        )

    # The per-parameter contract refuses the rank-3 write LOUDLY.
    packed.grad = grad.clone()
    with pytest.raises(LoudError, match="rank-3"):
        opt.step()

    # SF / NorMuon tails are per-parameter loops; the packed shape is refused.
    with pytest.raises(LoudError, match="packed"):
        Muon([packed], lr=0.05, sf=True)
    with pytest.raises(LoudError, match="packed"):
        Muon([packed], lr=0.05, variant="normuon")


# --------------------------------------------------------------------------- #
# F. the starvation rule rejects a tie (and the dry run prints the table)
# --------------------------------------------------------------------------- #
def test_the_starvation_rule_rejects_a_tie() -> None:
    drv = _load_script("ab_tsct_starvation")
    tie = drv.starvation_verdict(0.05, 0.05, 0.01)
    assert tie["verdict"] == "NOT STARVATION", tie
    assert "2x control-spread" in tie["reason"]

    half = drv.starvation_verdict(0.05, 0.024, 0.01)
    assert half["verdict"] == "STARVATION", half

    between = drv.starvation_verdict(0.05, 0.04, 0.001)
    assert between["verdict"] == "INCONCLUSIVE", between

    no_gap = drv.starvation_verdict(-0.01, -0.02, 0.01)
    assert no_gap["verdict"] == "NO GAP TO SHRINK", no_gap


def test_the_dry_run_prints_the_resolved_arms_and_the_rule(capsys) -> None:
    drv = _load_script("ab_tsct_starvation")
    assert drv.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "RESOLVED ARMS" in out
    assert "PRE-REGISTERED DECISION RULE" in out
    for arm in ("A", "B", "C", "D", "E"):
        assert f"\n  {arm} " in out, out
    assert "tok/expert" in out
    # The 32-token control and the 3x arm are both in the table.
    assert "32.0" in out
    assert ("48.0" in out) or ("96.0" in out), "the high tier did not resolve"
    assert "FALLBACK" in out or "b24 x s1024" in out
    assert "STARVATION is TRUE" in out
    # `--seq 1024` is the primary arm and says so instead of falling back.
    assert drv.main(["--dry-run", "--seq", "1024", "--vram-gib", "24"]) == 0
    out = capsys.readouterr().out
    assert "96.0" in out


# --------------------------------------------------------------------------- #
# refusals: the combinations the packed form cannot keep honest
# --------------------------------------------------------------------------- #
def test_refusals_name_the_escape() -> None:
    for over, needle in (
        ({"tsct_packed": True, "quant": "fp8"}, "quant"),
        ({"tsct_packed": True, "use_byteflow": True}, "use_byteflow"),
    ):
        with pytest.raises(LoudError, match="Escape") as exc:
            ModelConfig(**{**BASE, **over})
        assert needle in str(exc.value), f"the refusal does not name {needle}"
    # A hand-built SparseMoE skips the config layer: the pack still works.
    from dormouse.model.experts import SparseMoE

    moe = SparseMoE(ModelConfig(**{**BASE, "tsct_packed": True}), device="cpu")
    assert moe.tsct_packed and moe.packed_v.shape == (4, 8, 128)
