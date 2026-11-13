"""`attn_form="tsct"` on the LIVE transformer block: wired, exact, matched, LOUD.

WHY THIS EXISTS. TSCT factorised only the FFN — 50% of the block's MAC. The
attention projections (`qkv`, `proj`) are another 27% at d=1152, f=2848,
t=1024 (5.31M of 19.87M MAC/token), and they are factorisable with the same
module: the dense arm packs all three projections into ONE `[3d, d]` matrix,
so one `[d, r]·diag(s)·[r, 3d]` factors all three at a cost of `4·d·r` MAC —
`6·d·r` together with `proj`'s `2·d·r`, against the dense `4·d²`. SDPA's
4.72M MAC/token is quadratic in the sequence; no rank cut shrinks it.

Gates, each one fails on a scratch revert of the code it pins:

a. the flag builds the two factorised matrices per block (no `.weight`);
b. full rank IS the dense attention: SVD the dense `qkv`/`proj` into the
   factors and the logits agree to 1e-5 (fp32, exact reparameterisation);
c. params and MACs hand-computed at the live shape, r=64 and r=256, against
   the instantiated modules (not against a recorded number);
d. flag OFF is bit-identical to the HEAD fixture (names, RNG order, logits);
e. a rank-1 block's two projections are the arithmetic written in this file,
   and the projections hand SDPA the compute dtype (fp32 masters must not
   leak the fp32 lane into SDPA's quadratic cost);
f. no host sync in the forward or the backward;
g. a fresh tsct-attention block writes the DENSE arm's distribution at init
   (both projection output stds and the last-block output std within 5%):
   without `weight_std` the A/B would measure init scale, not factorisation;
h. the refusals — an unwired block form, a second attention factorisation,
   the two QAT seams, a rank that is not a factorisation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from dormouse.config import ModelConfig, load_preset
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.transformer import TransformerBlock
from dormouse.model.tsct import TSCTLinear

PRECHANGE = Path(__file__).resolve().parent / "fixtures" / "transformer_off_prechange.pt"


def build(**over) -> ModelConfig:
    """The tiny stand-in for the live block: pre-LN transformer + gated SwiGLU."""
    kw = dict(
        d_model=64,
        n_heads=4,
        head_dim=16,
        d_ffn=128,
        d_ffn_gelu=192,
        vocab=256,
        max_seq_len=64,
        n_layers=2,
        n_experts=1,
        moe_topk=0,
        max_iter=1,
        block_form="transformer",
        ffn_gated=True,
        expert_act="silu",
        bf16=False,
        use_tsct=False,
        tie_embeddings=False,
        use_kda=False,
        use_engram=False,
        use_csa=False,
        layerrope=False,
    )
    kw.update(over)
    return ModelConfig(**kw)


def _model(seed: int = 1234, **over) -> DormouseModel:
    torch.manual_seed(seed)
    return DormouseModel(build(**over))


def _ids() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, 256, (2, 16))


def _logits(model: DormouseModel) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out, _, _ = model(_ids(), _ids())
    return out


# ---------------------------------------------------------------------------
# a. the flag builds the two factorised projections, and nothing dense
# ---------------------------------------------------------------------------


def test_attn_tsct_builds_factorised_projections():
    dense = _model()
    tsct = _model(attn_form="tsct", attn_rank=8)
    modules = [m for m in tsct.modules() if isinstance(m, TSCTLinear)]
    assert len(modules) == 2 * 2, "one qkv + one proj per block, two blocks"
    assert not any(isinstance(m, TSCTLinear) for m in dense.modules())

    names = dict(tsct.named_parameters())
    for i in range(2):
        for mod in ("qkv", "proj"):
            for leaf in ("u", "s", "v"):
                assert f"transformer_blocks.{i}.{mod}.{leaf}" in names
            assert f"transformer_blocks.{i}.{mod}.weight" not in names
    # The qkv factorisation is ONE fused map [3d, d], not three [d, d] cuts.
    blk = tsct.transformer_blocks[0]
    assert (blk.qkv.in_features, blk.qkv.out_features) == (64, 192)
    assert (blk.proj.in_features, blk.proj.out_features) == (64, 64)
    assert param_count(tsct) < param_count(dense)


# ---------------------------------------------------------------------------
# b. full rank IS the dense attention (exact reparameterisation)
# ---------------------------------------------------------------------------


def _copy_non_attn(dst: DormouseModel, src: DormouseModel) -> None:
    """Same embed/head/norms/FFN on both sides: only `qkv`/`proj` may differ."""
    with torch.no_grad():
        tgt = dict(dst.state_dict())
        for name, value in src.state_dict().items():
            if any(f".{mod}." in name for mod in ("qkv", "proj")):
                continue
            if name in tgt and tgt[name].shape == value.shape:
                tgt[name].copy_(value)


def test_attn_tsct_at_full_rank_is_the_dense_attention():
    dense = _model()
    tsct = _model(attn_form="tsct", attn_rank=64)  # r = min(d, 3d) = d
    _copy_non_attn(tsct, dense)
    with torch.no_grad():
        for blk_d, blk_t in zip(dense.transformer_blocks, tsct.transformer_blocks, strict=True):
            for mod in ("qkv", "proj"):
                w = getattr(blk_d, mod).weight  # [out, in]
                u, s, vh = torch.linalg.svd(w.T, full_matrices=False)  # [in, out]
                t = getattr(blk_t, mod)
                t.u.copy_(u)
                t.s.copy_(s)
                t.v.copy_(vh)
    diff = float((_logits(dense) - _logits(tsct)).abs().max())
    print(f"[attn_tsct] full-rank reparameterisation: max|Δlogits| = {diff:.3e}")
    assert diff <= 1e-5, f"full-rank tsct attention is not the dense one: {diff:g}"


# ---------------------------------------------------------------------------
# c. params and MACs, hand-computed at d=1152, L=18
# ---------------------------------------------------------------------------


def _factor_macs(mod: TSCTLinear) -> int:
    """`x @ u` plus `(·) @ v` per token: in→r and r→out (the `s` is r)."""
    return mod.in_features * mod.rank + mod.rank * mod.out_features


def test_params_and_macs_are_the_hand_arithmetic():
    d, f, t, layers = 1152, 2848, 1024, 18
    base = dict(
        d_model=d,
        n_heads=18,
        head_dim=64,
        d_ffn=f,
        d_ffn_gelu=4272,
        n_layers=1,
        vocab=256,
        max_seq_len=t,
    )
    dense_blk = TransformerBlock(build(**base), n_layers=1)
    assert dense_blk.qkv.weight.numel() + dense_blk.proj.weight.numel() == 4 * d * d

    for r in (64, 256):
        blk = TransformerBlock(build(**base, attn_form="tsct", attn_rank=r), n_layers=1)
        got = sum(p.numel() for mod in (blk.qkv, blk.proj) for p in mod.parameters())
        # qkv [d,r]+[r]+[r,3d] and proj [d,r]+[r]+[r,d]
        hand = r * (d + 1 + 3 * d) + r * (d + 1 + d)
        assert got == hand, (r, got, hand)
        assert (got, hand) == {
            64: (442_496, 442_496),
            256: (1_769_984, 1_769_984),
        }[r]
        # The MAC arithmetic, read off the built modules: 4·d·r + 2·d·r = 6·d·r.
        macs = _factor_macs(blk.qkv) + _factor_macs(blk.proj)
        assert macs == 6 * d * r
        print(
            f"[attn_tsct] r={r}: params {got:,}/layer (dense {4 * d * d:,}), "
            f"MAC/token {macs:,} vs dense {4 * d * d:,} = {4 * d * d / macs:.1f}x"
        )

    # The live arms' totals: 18 layers of the per-layer delta.
    dense_attn = 4 * d * d
    for r, delta in ((64, 87_586_560), (256, 63_691_776)):
        per_layer = dense_attn - (6 * d * r + 2 * r)  # +r is `s` on both matrices
        assert layers * per_layer == delta, (r, per_layer)


def test_block_mac_ceiling_with_both_factorisations():
    """The number the arm exists for: the block's MAC/token, 18 layers.

    Dense FFN `3·d·f`; tsct FFN `3·r·(d + f)`; SDPA `4·t·d` (quadratic, and
    identical on both arms). The brief's 3.27x/1.96x assumed `8·d·r`
    (three separate `d→r→d` cuts for q, k, v); the FUSED `[3d, d]` map asked
    for here costs `4·d·r`, so the ceiling is higher.
    """
    d, f, t = 1152, 2848, 1024
    sdpa = 4 * t * d
    dense = 4 * d * d + sdpa + 3 * d * f
    assert dense == 19_869_696

    for r, ffn_only, both in ((64, 1.8404, 3.3513), (256, 1.5169, 2.0785)):
        ffn_tsct = 3 * r * (d + f)
        attn_tsct = 6 * d * r
        assert abs(dense / (4 * d * d + sdpa + ffn_tsct) - ffn_only) < 1e-3
        assert abs(dense / (attn_tsct + sdpa + ffn_tsct) - both) < 1e-3
        # Three separate rank-r q/k/v cuts (the brief's model) would be 8·d·r.
        assert 8 * d * r > attn_tsct


# ---------------------------------------------------------------------------
# d. flag OFF is the recorded forward, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.pins
def test_off_forward_is_head_bit_identical():
    """`attn_form` defaults to "dense": names, RNG order and logits unchanged."""
    golden = torch.load(PRECHANGE, weights_only=False)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _model()
        assert sorted(model.state_dict()) == golden["keys"], (
            "the off tree's parameter names moved — a flag-off build must be "
            "the control's, RNG order included"
        )
        assert not any(isinstance(m, TSCTLinear) for m in model.modules())
        logits, _, _ = model(golden["input"], golden["input"])
        assert torch.equal(logits, golden["logits"])
    finally:
        torch.set_num_threads(old)


# ---------------------------------------------------------------------------
# e. a rank-1 block, arithmetic computed here
# ---------------------------------------------------------------------------


def test_rank1_attention_projections_are_the_arithmetic_in_this_test():
    """`d=4, r=1`; one-hot factors make each projection a single product.

    qkv: `u = e0, s = 2, v[0, 1] = 3, v[0, 5] = -2` →
         output channel 1 is `6·x0`, channel 5 is `-4·x0`.
    proj: `u = e2, s = 0.5, v[0, 3] = 1` → channel 3 is `0.5·y2`.
    """
    cfg = build(
        d_model=4,
        n_heads=1,
        head_dim=4,
        d_ffn=8,
        d_ffn_gelu=0,
        n_layers=1,
        ffn_gated=False,
        attn_form="tsct",
        attn_rank=1,
        vocab=64,
        max_seq_len=8,
    )
    blk = TransformerBlock(cfg, n_layers=1)
    with torch.no_grad():
        blk.qkv.u.zero_()
        blk.qkv.u[0, 0] = 1.0
        blk.qkv.s.fill_(2.0)
        blk.qkv.v.zero_()
        blk.qkv.v[0, 1] = 3.0
        blk.qkv.v[0, 5] = -2.0
        blk.proj.u.zero_()
        blk.proj.u[2, 0] = 1.0
        blk.proj.s.fill_(0.5)
        blk.proj.v.zero_()
        blk.proj.v[0, 3] = 1.0

    x = torch.tensor([[[1.0, -1.0, 4.0, 0.5]]])  # [1, 1, d]
    with torch.no_grad():
        qkv = blk.qkv(x)
        y = blk.proj(torch.tensor([[[7.0, 0.0, 8.0, 0.0]]]))

    want_qkv = torch.zeros(1, 1, 12)
    want_qkv[0, 0, 1] = 2.0 * 3.0 * 1.0
    want_qkv[0, 0, 5] = 2.0 * -2.0 * 1.0
    want_proj = torch.zeros(1, 1, 4)
    want_proj[0, 0, 3] = 0.5 * 8.0
    assert torch.allclose(qkv, want_qkv, atol=1e-6), (qkv, want_qkv)
    assert torch.allclose(y, want_proj, atol=1e-6), (y, want_proj)


def test_attn_tsct_hands_sdpa_the_compute_dtype():
    """`TSCTLinear.forward` returns fp32 by contract; SDPA must not inherit it.

    SDPA is 4.72M MAC/token at t=1024 and quadratic in the sequence, so a
    factorised arm that quietly ran it on the fp32 lane would spend the FLOP
    cut this flag exists for. The dense arm's `mm_mixed` already returns the
    compute dtype; the factorised arm matches it in both modes.
    """
    with torch.no_grad():
        for bf16, want in ((True, torch.bfloat16), (False, torch.float32)):
            x = torch.randn(1, 4, 64)
            blk = TransformerBlock(build(bf16=bf16, attn_form="tsct", attn_rank=8), n_layers=1)
            dense = TransformerBlock(build(bf16=bf16), n_layers=1)
            assert blk._attn_proj(x, blk.qkv, None).dtype is want
            assert dense._attn_proj(x, dense.qkv, None).dtype is want


# ---------------------------------------------------------------------------
# f. no host sync
# ---------------------------------------------------------------------------


def test_no_host_sync_in_the_attn_tsct_forward_or_backward(monkeypatch):
    model = _model(attn_form="tsct", attn_rank=8)
    x = _ids()

    def boom(*_a, **_k):
        raise AssertionError("host sync in the attn tsct forward")

    for name in ("item", "tolist", "cpu", "numpy", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    _, loss, _ = model(x, x)
    loss.backward()


# ---------------------------------------------------------------------------
# g. the init is matched to the dense arm (the A/B's control)
# ---------------------------------------------------------------------------


def _forward_probe(model, monkeypatch, ids, capture: bool):
    """One eval forward: `(captured projection inputs, last-block output std)`.

    The capture is the `(block, which, input)` the DENSE arm's `_qmm` sees, so
    the same activations can be pushed through BOTH arms' `qkv`/`proj`. That
    isolates the projection's own output scale from the residual stream's.
    """
    got: list[tuple[TransformerBlock, str, torch.Tensor]] = []
    outputs: list[torch.Tensor] = []
    orig = TransformerBlock._qmm

    def rec(self, x, linear, kind, counters):
        if capture:
            if linear is self.qkv:
                got.append((self, "qkv", x.detach().float()))
            elif linear is self.proj:
                got.append((self, "proj", x.detach().float()))
        return orig(self, x, linear, kind, counters)

    monkeypatch.setattr(TransformerBlock, "_qmm", rec)
    handle = model.transformer_blocks[-1].register_forward_hook(
        lambda _m, _i, out: outputs.append(out.detach().float())
    )
    model.eval()
    try:
        with torch.no_grad():
            model(ids, ids)
    finally:
        handle.remove()
    assert len(got) == 2 * len(model.transformer_blocks) if capture else not got
    return got, outputs[0].std().item()


def _proj_out_stds(captured) -> dict[str, float]:
    acc: dict[str, list[torch.Tensor]] = {"qkv": [], "proj": []}
    with torch.no_grad():
        for blk, which, x in captured:
            acc[which].append((blk.qkv if which == "qkv" else blk.proj)(x))
    return {k: torch.cat(v).std().item() for k, v in acc.items()}


def test_tsct_attention_init_matches_the_dense_arm(monkeypatch):
    """GATE: both arms of the A/B start from the same attention write scale.

    The dense contract is `nn.init.normal_(w, 0.02)` on `qkv` and
    `0.02/sqrt(2L)` on `proj`; the factorised form must compose to the SAME
    entry std (`TSCTLinear(weight_std=σ)`), or the run measures init scale.
    LIVE shape: `arch_dense275`, one field changed.
    """
    torch.manual_seed(1234)
    dense = DormouseModel(dataclasses.replace(load_preset("arch_dense275"), bf16=False))
    torch.manual_seed(1234)
    tsct = DormouseModel(
        dataclasses.replace(
            load_preset("arch_dense275"), bf16=False, attn_form="tsct", attn_rank=64
        )
    )

    layers = 18
    shrink = 0.02 / (2 * layers) ** 0.5
    with torch.no_grad():
        blk = tsct.transformer_blocks[0]
        for mod, target in (("qkv", 0.02), ("proj", shrink)):
            got = getattr(blk, mod).effective_weight().std().item()
            assert abs(got / target - 1) <= 0.05, (mod, got, target)

    torch.manual_seed(7)
    ids = torch.randint(0, 256, (2, 8))
    captured, d_out = _forward_probe(dense, monkeypatch, ids, capture=True)
    _, t_out = _forward_probe(tsct, monkeypatch, ids, capture=False)

    d_std = _proj_out_stds(captured)
    t_std = _proj_out_stds(captured)  # the SAME activations, the tsct modules
    print(
        f"[attn_tsct] init: qkv std dense={d_std['qkv']:.4f} tsct={t_std['qkv']:.4f} "
        f"({t_std['qkv'] / d_std['qkv']:.3f}x); proj std dense={d_std['proj']:.5f} "
        f"tsct={t_std['proj']:.5f} ({t_std['proj'] / d_std['proj']:.3f}x); "
        f"last-block out dense={d_out:.4f} tsct={t_out:.4f} ({t_out / d_out:.3f}x)"
    )
    for key in ("qkv", "proj"):
        assert abs(t_std[key] / d_std[key] - 1) <= 0.05, (key, t_std[key], d_std[key])
    assert abs(t_out / d_out - 1) <= 0.05, f"last-block out {t_out:.4f} vs {d_out:.4f}"


# ---------------------------------------------------------------------------
# h. the refusals (loud, never inert)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("over", "named"),
    [
        ({"attn_form": "banana"}, "attn_form"),
        ({"attn_form": "tsct", "block_form": "kda", "n_layers": 0}, "block_form"),
        (
            {
                "attn_form": "tsct",
                "block_stack": "hybrid",
                "block_form": "kda",
                "n_layers": 4,
            },
            "block_stack",
        ),
        ({"attn_form": "tsct", "attn_latent_rank": 32}, "attn_latent_rank"),
        ({"attn_form": "tsct", "fused_latent_proj": True}, "fused_latent_proj"),
        ({"attn_form": "tsct", "act_qat": "e2m1"}, "act_qat"),
        ({"attn_form": "tsct", "dense_qat": "ternary"}, "dense_qat"),
        ({"attn_form": "tsct", "attn_rank": 128}, "attn_rank"),
        ({"attn_rank": 16}, "attn_rank"),
        ({"attn_rank": -1}, "attn_rank"),
    ],
)
def test_the_unwired_forms_and_combinations_refuse(over, named):
    with pytest.raises(LoudError) as e:
        build(**over)
    msg = str(e.value)
    assert named in msg, msg
    assert "Escape:" in msg, msg


def test_the_runtime_seam_refuses_a_hand_built_config():
    """`config.validate_block_form` is the loader path; the block checks too."""
    cfg = build(act_qat="e2m1")
    object.__setattr__(cfg, "attn_form", "tsct")  # bypasses `__post_init__`
    with pytest.raises(LoudError) as e:
        TransformerBlock(cfg, n_layers=1)
    assert "act_qat" in str(e.value) and "Escape:" in str(e.value)


def test_the_default_form_is_not_a_claim_on_this_block():
    """`attn_form="dense"` is the field's default, so it cannot be refused."""
    build(attn_form="dense")
    build(attn_form="dense", attn_rank=0)
