"""The pre-sized wgrad buffer: parity, reuse, and no aliasing under grad_accum.

The pilot OOM'd on a fresh `torch.empty(E, r, f)` = 682.5 MiB in `_wgrad` at
step ~2900. The fused adjoint now writes its three wgrads into pre-sized,
caller-owned `BwdWorkspace` slots. This module gates three things:

* the gradients are BIT-IDENTICAL to the reference adjoint (same fp64 CPU
  stand-ins for the three GEMMs the existing parity gate uses);
* the slot is reserved once and reused (no fresh allocation per backward);
* autograd never aliases a reused slot into a non-leaf bank's leaf.grad —
  two micro-batches must sum to g1+g2, not 2*g2 (the measured corruption).
"""

import torch

from dormouse.model import moe_kernels as mk


def _fake_nt(a, w, offs, cap, n, k, tiles, out_dtype, out=None):
    res = torch.zeros(a.shape[0], n, dtype=out_dtype)
    for i, (s, t) in enumerate(zip(offs[:-1].tolist(), offs[1:].tolist(), strict=False)):
        if t > s:
            res[s:t] = (a[s:t].double() @ w[i].double().T).to(out_dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def _fake_wgrad(a, b, offs, e, cap, k, n, tiles, out_dtype, out=None):
    res = torch.zeros(e, k, n, dtype=out_dtype)
    for i in range(e):
        s, t = int(offs[i]), int(offs[i + 1])
        if t > s:
            res[i] = (a[s:t].double().T @ b[s:t].double()).to(out_dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def _fake_pair(a1, w1, a2, w2, offs, cap, n, k, tiles, out_dtype):
    return _fake_nt(a1, w1, offs, cap, n, k, tiles, out_dtype) + _fake_nt(
        a2, w2, offs, cap, n, k, tiles, out_dtype
    )


def _patch(monkeypatch):
    monkeypatch.setattr(mk, "_grouped_nt", _fake_nt)
    monkeypatch.setattr(mk, "_wgrad", _fake_wgrad)
    monkeypatch.setattr(mk, "_grouped_nt_pair", _fake_pair)


def _inputs(e=5, d=24, r=8, f=32, seed=3):
    torch.manual_seed(seed)
    counts = torch.tensor([0, 3, 7, 2, 6][:e], dtype=torch.int32)
    rows = int(counts.sum())
    offs = torch.zeros(e + 1, dtype=torch.int32)
    offs[1:] = torch.cumsum(counts, 0)
    seg = torch.repeat_interleave(torch.arange(e, dtype=torch.int32), counts.to(torch.int64))
    xg = torch.randn(rows, d)
    gu = torch.randn(e, d, r) / d**0.5
    uu = torch.randn(e, d, r) / d**0.5
    gs = torch.randn(e, r) * 0.1 + 1.0
    us = torch.randn(e, r) * 0.1 + 1.0
    v = torch.randn(e, r, f) / r**0.5
    gpre, upre = torch.randn(rows, r), torch.randn(rows, r)
    h = torch.randn(rows, r)
    dy = torch.randn(rows, f)
    args = (xg, gu, uu, gs, us, v, offs, int(counts.max()), gpre, upre, h, dy, mk.TILES)
    return args, seg


NAMES = ("dx", "dgate_u", "dup_u", "dgate_s", "dup_s", "dv")


def test_fused_wgrad_parity_and_slot_reuse(monkeypatch):
    """The pre-sized buffers carry bit-identical grads and are reserved once."""
    _patch(monkeypatch)
    args, seg = _inputs()
    ref = mk._glu_backward(*args)  # fresh _wgrad (out=None) — the reference
    ws = mk.BwdWorkspace()
    got = mk._glu_backward_fused(*args, seg, ws)
    for name, a, b in zip(NAMES, got, ref, strict=True):
        assert torch.equal(a, b), f"{name}: pre-sized wgrad changed the gradient"

    e, r, f = args[5].shape
    dv_slot = ws._flat["wg_dv"]
    assert dv_slot.numel() == e * r * f
    # The wide wgrad at b1 is [1024, 64, 5461] bf16 = 682.5 MiB; the slot's
    # theoretical size is the same number, allocated ONCE not per backward.
    assert dv_slot.numel() * dv_slot.element_size() == e * r * f * dv_slot.element_size()

    ptr = dv_slot.data_ptr()
    got2 = mk._glu_backward_fused(*args, seg, ws)
    assert ws._flat["wg_dv"].data_ptr() == ptr, "the wgrad slot must be reused, not reallocated"
    assert got2[5].data_ptr() == ptr, "the returned dv must be the pre-sized slot"


def test_grad_accumulation_does_not_alias_the_reused_wgrad(monkeypatch):
    """Two micro-batches sum to g1+g2 even though the wgrad slot is reused."""
    _patch(monkeypatch)

    def run(fused):
        torch.manual_seed(11)
        e, d, r, f = 4, 12, 4, 8
        counts = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        rows = int(counts.sum())
        offs = torch.zeros(e + 1, dtype=torch.int32)
        offs[1:] = torch.cumsum(counts, 0)
        seg = torch.repeat_interleave(torch.arange(e, dtype=torch.int32), counts.to(torch.int64))
        gu_leaves = [torch.nn.Parameter(torch.randn(d, r) / d**0.5) for _ in range(e)]
        uu_leaves = [torch.nn.Parameter(torch.randn(d, r) / d**0.5) for _ in range(e)]
        v_leaves = [torch.nn.Parameter(torch.randn(r, f) / r**0.5) for _ in range(e)]
        gu = torch.stack(gu_leaves)
        uu = torch.stack(uu_leaves)
        v = torch.stack(v_leaves)
        gs = torch.randn(e, r) * 0.1 + 1.0
        us = torch.randn(e, r) * 0.1 + 1.0
        xg = torch.randn(rows, d)
        gpre, upre = torch.randn(rows, r), torch.randn(rows, r)
        h = torch.randn(rows, r)
        ws = mk.BwdWorkspace()

        # The forward's Triton kernels cannot launch on CPU. Only the backward
        # is under test, so stub the forward to a differentiable zero of the
        # right shape while returning the saved activations we prepared.
        def fake_head(_xg, _gu, _uu, _gs, _us, _v, *_rest, **_kw):
            return _xg.new_zeros(_xg.shape[0], _v.shape[2]), (gpre, upre, h)

        monkeypatch.setattr(mk, "_glu_fused_or_head", fake_head)
        monkeypatch.setattr(
            mk, "expert_gemm", lambda hh, vv, oo, **kw: hh.new_zeros(hh.shape[0], vv.shape[2])
        )
        for seed in (1, 2):
            out = mk._ExpertGemmGlu.apply(
                xg,
                gu,
                uu,
                gs,
                us,
                v,
                offs,
                int(counts.max()),
                mk.TILES,
                fused,
                seg,
                ws,
                None,
            )
            # A DIFFERENT grad_output per micro-batch: with equal dy the two
            # contributions are identical and the aliasing bug (2*g2 instead of
            # g1+g2) would pass. Weighting the sum makes the gate sharp.
            w = torch.full_like(out, float(seed))
            (out * w).sum().backward()
        return [p.grad.clone() for p in gu_leaves + uu_leaves + v_leaves]

    ref = run(False)  # reference adjoint: fresh grads, packs g1+g2
    got = run(True)  # fused adjoint: pre-sized slots + step-boundary clone
    for i, (a, b) in enumerate(zip(got, ref, strict=True)):
        assert torch.equal(a, b), f"leaf {i}: grad accumulation corrupted by slot reuse"


def test_old_size_and_new_fresh_allocation(monkeypatch):
    """State the allocation change numerically: 684 MiB per call -> 0 reused."""
    e, r, f, dt = 1024, 64, 5461, torch.bfloat16
    old = e * r * f * torch.empty((), dtype=dt).element_size()
    assert old == 715_784_192  # 682.5 MiB, the exact OOM allocation
    assert int(old / 2**20) == 682
    # After the fix, a second backward into the same workspace performs zero
    # fresh wgrad allocations: the three slots already exist at full size.
    _patch(monkeypatch)
    args, seg = _inputs()
    ws = mk.BwdWorkspace()
    mk._glu_backward_fused(*args, seg, ws)
    before = {k: t.data_ptr() for k, t in ws._flat.items()}
    mk._glu_backward_fused(*args, seg, ws)
    after = {k: t.data_ptr() for k, t in ws._flat.items()}
    assert before == after, "every workspace slot must be reused on the 2nd backward"
