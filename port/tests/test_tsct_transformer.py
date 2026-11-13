"""`ffn_form="tsct"` on the LIVE transformer block: wired, exact, and LOUD.

The defect this file pins: `configs/arch_tsct275.toml` set `ffn_form="tsct"`
with `block_form="transformer"`, loaded, and priced 273,393,792 parameters —
identical to the dense control — with ZERO `TSCTLinear` modules. The form only
reached code on the KDA loop, so the arm would have measured a dense model
while the log line said `tsct`.

Gates, each one fails on a scratch revert of the code it pins:

a. the form builds the three factorised matrices and is strictly smaller than
   the dense control at the same shape (both counts printed);
b. full rank IS the dense gated FFN: SVD the dense weights into the factors
   and the logits agree to 1e-5 (fp32, the exact-reparameterisation gate);
c. rank 64 is `3·(d·r + r + r·f)` per FFN, hand-computed, at the live 275M
   shape (`d=1152, f=2848`);
d. flag OFF is bit-identical to the HEAD fixture (`names`, RNG order, logits);
e. a rank-1 FFN whose output is arithmetic written in this file;
f. no host sync in the forward, and every factor name is one `stiefel_qr`
   retracts (`tsct.is_factor_key`), so the trainer's retraction composes;
h. a fresh tsct block writes the DENSE gated block's distribution at the live
   shape (FFN write std and last-block output std within 5%): without it the
   running A/B measures init scale (13.3x), not rank-64 factorisation.
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
from dormouse.model.tsct import TSCTLinear, is_factor_key

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
# a. the defect: factors exist, and the model is strictly smaller
# ---------------------------------------------------------------------------


def test_tsct_builds_factorised_matrices_and_is_smaller_than_dense():
    dense = _model()
    tsct = _model(ffn_form="tsct", ffn_rank_gate=8, ffn_rank_down=8)
    n_dense, n_tsct = param_count(dense), param_count(tsct)
    print(f"[tsct] transformer FFN @ d=64 f=128 L=2: dense {n_dense:,} vs tsct {n_tsct:,}")

    modules = [m for m in tsct.modules() if isinstance(m, TSCTLinear)]
    assert len(modules) == 3 * 2, "one gate/up/down per block, two blocks"
    assert not any(isinstance(m, TSCTLinear) for m in dense.modules())
    assert n_tsct < n_dense, "a rank-8 factorisation must be smaller than dense"

    names = dict(tsct.named_parameters())
    for i in range(2):
        for mod in ("gate_proj", "up_proj", "fc2"):
            for leaf in ("u", "s", "v"):
                assert f"transformer_blocks.{i}.{mod}.{leaf}" in names
            assert f"transformer_blocks.{i}.{mod}.weight" not in names


def test_the_live_arch_tsct275_config_actually_builds_the_arm():
    """The exact config that found the defect: it must build factors, not dense."""
    cfg = load_preset("arch_tsct275")
    model = DormouseModel(cfg)
    assert cfg.ffn_form == "tsct" and cfg.block_form == "transformer"
    n = sum(1 for m in model.modules() if isinstance(m, TSCTLinear))
    assert n == 3 * cfg.n_layers, f"{n} factorised matrices, expected {3 * cfg.n_layers}"


# ---------------------------------------------------------------------------
# b. full rank IS the dense GLU (exact reparameterisation)
# ---------------------------------------------------------------------------


def _copy_non_ffn(dst: DormouseModel, src: DormouseModel) -> None:
    """Same attention/embed/head on both sides: only the FFN may differ."""
    with torch.no_grad():
        tgt = dict(dst.state_dict())
        for name, value in src.state_dict().items():
            if any(f".{mod}." in name for mod in ("gate_proj", "up_proj", "fc2")):
                continue
            if name in tgt and tgt[name].shape == value.shape:
                tgt[name].copy_(value)


def test_tsct_at_full_rank_is_the_dense_gated_ffn():
    dense = _model()
    tsct = _model(block_form="transformer", ffn_form="tsct", ffn_rank_gate=64, ffn_rank_down=64)
    _copy_non_ffn(tsct, dense)
    with torch.no_grad():
        for blk_d, blk_t in zip(dense.transformer_blocks, tsct.transformer_blocks, strict=True):
            for mod in ("gate_proj", "up_proj", "fc2"):
                w = getattr(blk_d, mod).weight  # [out, in]
                u, s, vh = torch.linalg.svd(w.T, full_matrices=False)  # [in, out]
                t = getattr(blk_t, mod)
                t.u.copy_(u)
                t.s.copy_(s)
                t.v.copy_(vh)
    diff = float((_logits(dense) - _logits(tsct)).abs().max())
    print(f"[tsct] full-rank reparameterisation: max|Δlogits| = {diff:.3e}")
    assert diff <= 1e-5, f"full-rank tsct is not the dense GLU: max|Δlogits| = {diff:g}"


# ---------------------------------------------------------------------------
# c. rank 64 is 3·(d·r + r + r·f) per FFN, at the live shape
# ---------------------------------------------------------------------------


def test_rank64_ffn_params_match_the_hand_formula():
    d, f, r = 1152, 2848, 64
    cfg = build(
        d_model=d,
        n_heads=18,
        head_dim=64,
        d_ffn=f,
        d_ffn_gelu=4272,
        n_layers=1,
        ffn_form="tsct",
        rank=r,
    )
    blk = TransformerBlock(cfg, n_layers=1)
    ffn = sum(p.numel() for mod in (blk.gate_proj, blk.up_proj, blk.fc2) for p in mod.parameters())
    hand = 3 * (d * r + r + r * f)  # u [in,r] + s [r] + v [r,out], three matrices
    print(f"[tsct] rank-64 FFN @ d={d} f={f}: {ffn:,} params (hand {hand:,})")
    assert ffn == hand == 768_192

    dense = TransformerBlock(dataclasses.replace(cfg, ffn_form="bank"), n_layers=1)
    dense_ffn = (
        dense.gate_proj.weight.numel() + dense.up_proj.weight.numel() + dense.fc2.weight.numel()
    )
    assert dense_ffn == 3 * d * f == 9_842_688
    # 18 layers of that delta is the whole 275M arm's saving.
    assert 18 * (dense_ffn - ffn) == 163_340_928


# ---------------------------------------------------------------------------
# d. flag OFF is the recorded forward, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.pins
def test_off_forward_is_head_bit_identical():
    """The golden was captured from HEAD (b7c2f72) with this exact fixture."""
    golden = torch.load(PRECHANGE, weights_only=False)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _model()
        assert sorted(model.state_dict()) == golden["keys"], (
            "the off tree's parameter names moved — a flag-off build must be the "
            "control's, RNG order included"
        )
        assert not any(isinstance(m, TSCTLinear) for m in model.modules())
        logits, _, _ = model(golden["input"], golden["input"])
        assert torch.equal(logits, golden["logits"])
    finally:
        torch.set_num_threads(old)


# ---------------------------------------------------------------------------
# e. a rank-1 FFN, arithmetic computed here
# ---------------------------------------------------------------------------


def test_rank1_ffn_output_is_the_arithmetic_in_this_test():
    """`d=6, f=16, r=1`; one-hot factors make the FFN a single product.

    gate: `u = e0, s = 2, v = 1` → `g_j = 2·x0` for every hidden unit.
    up:   `u = e1, s = 3, v = e5` → `u_j = 3·x1` at unit 5, 0 elsewhere.
    down: `u = e5, s = 1, v = e2` → output channel 2 reads hidden unit 5.

    so `FFN(x) = 3·x1·silu(2·x0)` on channel 2, 0 elsewhere (silu = SwiGLU).
    """
    cfg = build(
        d_model=6,
        n_heads=1,
        head_dim=6,
        d_ffn=16,
        d_ffn_gelu=0,
        n_layers=1,
        ffn_form="tsct",
        rank=1,
        vocab=64,
        max_seq_len=8,
    )
    blk = TransformerBlock(cfg, n_layers=1)
    with torch.no_grad():
        blk.gate_proj.u.zero_()
        blk.gate_proj.u[0, 0] = 1.0
        blk.gate_proj.s.fill_(2.0)
        blk.gate_proj.v.fill_(1.0)
        blk.up_proj.u.zero_()
        blk.up_proj.u[1, 0] = 1.0
        blk.up_proj.s.fill_(3.0)
        blk.up_proj.v.zero_()
        blk.up_proj.v[0, 5] = 1.0
        blk.fc2.u.zero_()
        blk.fc2.u[5, 0] = 1.0
        blk.fc2.s.fill_(1.0)
        blk.fc2.v.zero_()
        blk.fc2.v[0, 2] = 1.0

    x = torch.tensor([[[1.0, 2.0, 0.0, 0.0, 0.0, 0.0]]])  # [1, 1, d]
    with torch.no_grad():
        out = blk._ffn(x, None)
    expected = torch.zeros_like(out)
    expected[0, 0, 2] = 3.0 * 2.0 * torch.nn.functional.silu(torch.tensor(2.0 * 1.0))
    assert torch.allclose(out, expected, atol=1e-6), (out, expected)


# ---------------------------------------------------------------------------
# f. no host sync, and `stiefel_qr` sees every factor
# ---------------------------------------------------------------------------


def test_every_ffn_factor_is_a_stiefel_key():
    model = _model(ffn_form="tsct", ffn_rank_gate=8, ffn_rank_down=8)
    ffn = [
        n
        for n, _ in model.named_parameters()
        if ".gate_proj." in n or ".up_proj." in n or ".fc2." in n
    ]
    assert len(ffn) == 3 * 2 * 3
    factors = [n for n in ffn if is_factor_key(n)]
    assert len(factors) == 3 * 2 * 2, factors  # u and v; `s` is the scale
    assert not any(n.endswith(".s") for n in factors)


def test_no_host_sync_in_the_tsct_forward_or_backward(monkeypatch):
    model = _model(ffn_form="tsct", ffn_rank_gate=8, ffn_rank_down=8)
    x = _ids()

    def boom(*_a, **_k):
        raise AssertionError("host sync in the tsct transformer forward")

    for name in ("item", "tolist", "cpu", "numpy", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    _, loss, _ = model(x, x)
    loss.backward()


# ---------------------------------------------------------------------------
# h. the init is matched to the dense gated arm (the A/B's control)
# ---------------------------------------------------------------------------


def _init_stds(model: DormouseModel, ids: torch.Tensor, monkeypatch) -> tuple[float, float]:
    """(last block's FFN write std, last block's output std) of one FRESH model.

    The write is `_ffn`'s return, captured by wrapping the block method the
    real forward calls (hooking `fc2` would miss the tag: `_qmm` calls
    `mm_mixed`, not the module). The write std is POOLED over the whole stack:
    one block's `64×2848` factor draw alone swings ±8% on sampling noise, while
    the pooled estimate is the block's scale (three seeds: 0.999–1.024x).
    """
    writes: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    orig = TransformerBlock._ffn

    def rec(self, h2, counters=None):
        out = orig(self, h2, counters)
        writes.append(out.detach().float())
        return out

    monkeypatch.setattr(TransformerBlock, "_ffn", rec)
    handle = model.transformer_blocks[-1].register_forward_hook(
        lambda _m, _i, out: outputs.append(out.detach().float())
    )
    model.eval()
    try:
        with torch.no_grad():
            model(ids, ids)
    finally:
        handle.remove()
    return torch.cat(writes).std().item(), outputs[0].std().item()


def test_tsct_init_matches_the_dense_gated_arm(monkeypatch):
    """GATE: both arms of the 275M A/B start from the same FFN write scale.

    The dense arm's contract is `nn.init.normal_(w, 0.02)` on gate/up and
    `0.02/sqrt(2L)` on the write. The tsct arm must produce the SAME composed
    `U·diag(s)·V` entry std — it does that through `weight_std`, since `u` and
    `s` keep their shapes/scheme. LIVE shape: the two presets as they ship.

    Both stds are computed from two freshly built models, never recorded.
    """
    torch.manual_seed(1234)
    dense = DormouseModel(dataclasses.replace(load_preset("arch_dense275"), bf16=False))
    torch.manual_seed(1234)
    tsct = DormouseModel(dataclasses.replace(load_preset("arch_tsct275"), bf16=False))

    # The arithmetic `weight_std` relies on: the composed entry std IS the target.
    with torch.no_grad():
        blk = tsct.transformer_blocks[0]
        shrink = 0.02 / (2 * tsct.cfg.n_layers) ** 0.5
        for mod, target in (("gate_proj", 0.02), ("up_proj", 0.02), ("fc2", shrink)):
            got = getattr(blk, mod).effective_weight().std().item()
            assert abs(got / target - 1) <= 0.05, (mod, got, target)

    torch.manual_seed(7)
    ids = torch.randint(0, 256, (2, 16))
    dw, do = _init_stds(dense, ids, monkeypatch)
    tw, to = _init_stds(tsct, ids, monkeypatch)
    print(
        f"[tsct] init: FFN write std dense={dw:.4f} tsct={tw:.4f} ({tw / dw:.2f}x); "
        f"last-block out std dense={do:.4f} tsct={to:.4f} ({to / do:.2f}x)"
    )
    assert abs(tw / dw - 1) <= 0.05, f"tsct FFN write {tw:.4f} vs dense {dw:.4f}"
    assert abs(to / do - 1) <= 0.05, f"tsct last-block out {to:.4f} vs dense {do:.4f}"


# ---------------------------------------------------------------------------
# g. the refusals (step 1 of the defect: loud, never inert)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("over", "named"),
    [
        ({"ffn_form": "dense"}, "ffn_form"),
        ({"ffn_form": "tsct", "ffn_gated": False}, "ffn_gated"),
        ({"ffn_form": "tsct", "n_experts": 2}, "n_experts"),
        ({"ffn_form": "tsct", "act_qat": "e2m1"}, "act_qat"),
        ({"ffn_form": "tsct", "dense_qat": "ternary"}, "dense_qat"),
        (
            {
                "ffn_form": "tsct",
                "block_stack": "hybrid",
                "block_form": "kda",
                "n_layers": 4,
            },
            "ffn_form",
        ),
    ],
)
def test_the_unwired_forms_and_combinations_refuse(over, named):
    with pytest.raises(LoudError) as e:
        build(**over)
    msg = str(e.value)
    assert named in msg, msg
    assert "Escape:" in msg, msg


def test_the_default_form_is_not_a_claim_on_this_block():
    """`ffn_form="bank"` is the field's default, so it cannot be refused here."""
    build(ffn_form="bank")  # every transformer preset on disk omits the key
