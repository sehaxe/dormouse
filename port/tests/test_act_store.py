"""Gates for `act_store` — the STORE-ONLY packed-activation arm.

THE CLAIM THIS FILE PINS. `act_store="nvfp4"` moves VRAM without moving the
forward: the block's GEMM inputs and GLU operands are multiplied at bf16 exactly
as the `bf16` arm does, and only `save_for_backward` holds the 4-bit copy
(`model/packed_acts.py`, the store-only lane of the SAME codec `act_qat` uses).
So the logits are `torch.equal`, not "close"; the gradients differ by exactly
the storage rounding of the saved operand — `grad_x` not at all, `grad_W` by
`gradᵀ @ (xq − x)` — and that difference is compared here against the bound
`||grad||_F · ||E||_F` derived from the codec, not captured from a run.

What is proven, in order:

1. the default arm takes NO branch: the packed helpers are never entered;
2. `nvfp4` logits are bit-identical and the save set is smaller, with no wide
   bf16 save left at the packed sites;
3. the operand bound holds for one site, and `grad_x` is bit-identical;
4. the model-level gradient deltas are reported and held inside what the
   existing 4-bit activation arm (`act_qat`) accepts on the same fixture;
5. a width that does not tile the codec's 16-column grid (`d=890` shape class)
   is padded on the STORED copy, not refused, and stays exact in the forward;
6. every refusal the arm owns is LOUD (block form, hybrid stack, `act_qat`,
   `torch_compile`, `mm_dtype='fp8'`, a `TSCTLinear` form, a bad literal).
"""

from __future__ import annotations

import pytest
import torch
from conftest import write_corpus
from torch.autograd.graph import saved_tensors_hooks

from dormouse.config import (
    ModelConfig,
    RunConfig,
    TrainConfig,
    validate_model_train,
    validate_train,
)
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, packed_acts, tsct
from dormouse.model.nvfp4 import dequantize_nvfp4, quantize_nvfp4
from dormouse.model.packed_acts import packed_act_mm

BASE = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    d_ffn_gelu=192,
    vocab=256,
    max_seq_len=64,
    n_layers=2,
    max_iter=1,
    block_form="transformer",
    ffn_gated=True,
    expert_act="silu",
    bf16=True,
    use_tsct=False,
    tie_embeddings=False,
    use_kda=False,
    use_engram=False,
    use_csa=False,
    layerrope=False,
)

#: `d=40` does NOT tile the codec's 16-column grid (40 % 16 = 8) — the live
#: `d=890` is the same class (890 % 16 = 10), and the store must pad, not refuse.
ODD = dict(d_model=40, n_heads=5, head_dim=8, d_ffn=40, d_ffn_gelu=60, vocab=64)


def _model(over: dict | None = None) -> DormouseModel:
    torch.manual_seed(1234)
    return DormouseModel(ModelConfig(**{**BASE, **(over or {})}))


def _inputs(b: int = 2, t: int = 32, vocab: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    ids = torch.randint(0, vocab, (b, t))
    return ids, (ids + 1) % vocab


def _saves(fn):
    """Every tensor `save_for_backward` holds while `fn` runs (pass-through spy)."""
    seen: list[tuple[tuple[int, ...], torch.dtype, int]] = []

    def pack(t):
        seen.append((tuple(t.shape), t.dtype, t.numel() * t.element_size()))
        return t

    with saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return seen, out


def _step(model: DormouseModel, *, store: str | None = None, vocab: int = 256):
    if store is not None:
        model.set_act_store(store)
    ids, targets = _inputs(vocab=vocab)
    counters = Counters()
    logits, loss, counters = model(ids, targets, counters)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach().clone(), grads, counters


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# ---------------------------------------------------------------------------
# (1) the default arm takes no branch and packs nothing
# ---------------------------------------------------------------------------


def test_the_default_arm_never_enters_the_packed_lane(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("the packed lane ran with act_store='bf16'")

    monkeypatch.setattr(packed_acts, "packed_act_mm", boom)
    monkeypatch.setattr(packed_acts, "packed_glu", boom)
    monkeypatch.setattr(packed_acts._PackedActMM, "apply", boom)
    monkeypatch.setattr(packed_acts._PackedGLU, "apply", boom)

    model = _model()
    assert model.transformer_blocks[0].act_store == "bf16"
    logits, grads, counters = _step(model)
    assert float(logits.abs().sum()) > 0.0
    assert counters.get("act_store_mm", 0) == 0
    assert counters.get("packed_act_mm", 0) == 0
    assert all(torch.isfinite(g).all() for g in grads.values())


# ---------------------------------------------------------------------------
# (2) exact forward, smaller save set, no wide bf16 save at the packed sites
# ---------------------------------------------------------------------------


def test_the_store_arm_is_the_bf16_forward_bit_for_bit():
    off_logits, off_grads, _ = _step(_model(), store="bf16")
    on_logits, on_grads, counters = _step(_model(), store="nvfp4")
    assert torch.equal(off_logits, on_logits), "the memory arm moved the forward"
    assert set(off_grads) == set(on_grads)
    # The counter names the sites that actually packed (4 GEMMs + 1 GLU per
    # layer), so an arm that reaches nothing cannot report itself on.
    layers = BASE["n_layers"]
    assert counters.get("act_store_mm") == 4 * layers
    assert counters.get("act_store_glu") == layers


#: `b=3, t=32` → `T = 96`, chosen so `T` is neither `d` nor `f`: a bf16
#: activation save of `[T, f]` is then unambiguous against a weight cast.
T_ROWS = 3 * 32


def _saved_bytes(store: str) -> tuple[int, list[tuple[tuple[int, ...], torch.dtype, int]]]:
    model = _model()
    model.set_act_store(store)
    ids, targets = _inputs(3, 32)
    seen, _ = _saves(lambda: model(ids, targets)[1])
    return sum(nbytes for _, _, nbytes in seen), seen


def test_the_saved_set_shrinks_and_no_wide_ffn_save_survives():
    off_bytes, off = _saved_bytes("bf16")
    on_bytes, on = _saved_bytes("nvfp4")
    f = BASE["d_ffn"]
    assert on_bytes < off_bytes, f"act_store='nvfp4' kept {on_bytes} B against bf16's {off_bytes} B"
    # The GLU's `[T, f]` operands were the widest FFN saves; none is bf16 now
    # (weights are not `T` rows, so the filter is the existing one).
    assert not any(
        dt is torch.bfloat16 and len(sh) == 2 and sh[0] == T_ROWS and sh[-1] == f
        for sh, dt, _ in on
    ), "a wide bf16 GLU save survived the packed store"
    assert any(dt is torch.uint8 for _, dt, _ in on), "nothing was packed"
    assert any(dt is torch.float8_e4m3fn for _, dt, _ in on), "no e4m3 scale grid"


# ---------------------------------------------------------------------------
# (3) the operand bound: output and grad_x exact, grad_W within ||grad||·||E||
# ---------------------------------------------------------------------------


def test_the_grad_w_delta_is_bounded_by_the_codec_rounding_it_saves():
    torch.manual_seed(3)
    x = torch.randn(96, 64, dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(64, 64, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    gx = torch.randn(96, 64, dtype=torch.bfloat16)

    # The bf16 line is `mm_mixed`: `x` and `w` are already the compute dtype.
    y_ref = x @ w.t()
    y = packed_act_mm(x, torch.bfloat16, w, store_only=True)[0]
    assert torch.equal(y, y_ref), "the store-only lane changed the forward product"
    y_ref.backward(gx)
    w_ref, x_ref = w.grad.clone(), x.grad.clone()
    w.grad = None
    x.grad = None
    y = packed_act_mm(x, torch.bfloat16, w, store_only=True)[0]
    y.backward(gx)
    assert torch.equal(x.grad, x_ref), "grad_x must not read the stored operand"

    packed, scales, g = quantize_nvfp4(x)
    err = (dequantize_nvfp4(packed, scales, x.dtype, global_scale=g) - x).detach()
    bound = float(gx.norm().detach() * err.norm())
    measured = float((w.grad - w_ref).norm())
    assert measured <= bound * (1.0 + 1e-6), (
        f"grad_W moved {measured:.3e} against the codec bound {bound:.3e}"
    )
    assert measured > 0.0, "the stored operand is bf16 exactly — the arm packed nothing"


# ---------------------------------------------------------------------------
# (4) the model-level deltas, inside what the 4-bit activation arm accepts
# ---------------------------------------------------------------------------


def test_gradient_deltas_stay_inside_the_existing_4bit_arm():
    """Same fixture, two 4-bit arms, per site.

    The FFN matrices are the sites `act_qat` already rounds: `grad_W` there
    reads the SAME packed `xq`, so `act_qat`'s own relative L2 delta is the
    bound this arm is held to. The attention GEMM inputs (`qkv`, `proj`) have
    NO `act_qat` counterpart — that arm keeps attention at >= 8 bits (a4.8's
    rule) — so this arm's attention cost is measured here and reported, not
    compared: it is the ONE thing the existing bound does not cover.
    """
    _, base, _ = _step(_model())
    _, store, _ = _step(_model(), store="nvfp4")
    torch.manual_seed(1234)
    qat_model = DormouseModel(ModelConfig(**{**BASE, "act_qat": "e2m1"}))
    ids, targets = _inputs()
    _, loss, _ = qat_model(ids, targets)
    loss.backward()
    qat = {
        n: p.grad.detach().clone() for n, p in qat_model.named_parameters() if p.grad is not None
    }

    ffn = ("fc.", "fc2.", "gate_proj.", "up_proj.")
    attn = ("qkv.", "proj.")
    rows = []
    for name in base:
        if "transformer_blocks" not in name or not any(k in name for k in ffn + attn):
            continue
        r_store = _rel(store[name], base[name])
        r_qat = _rel(qat[name], base[name])
        rows.append((name, r_store, r_qat))
        if any(k in name for k in ffn):
            assert r_store <= r_qat + 0.02, f"{name}: store {r_store:.3f} > qat {r_qat:.3f} + 0.02"
        else:
            # 0.2 is the codec's coarse ceiling on an LN output: the block scale
            # is set by the 16-column amax, so a typical element sits at ~1/8
            # of it. Measured 0.09 here, and 0.16 for `act_qat` on the FFN.
            assert r_store <= 0.2, f"{name}: the attention input moved by {r_store:.3f}"
    print("\nsite                                   store    act_qat")
    for name, r_store, r_qat in rows:
        print(f"{name:<38} {r_store:.4f}   {r_qat:.4f}")
    assert rows, "no block matrix was compared"


# ---------------------------------------------------------------------------
# (5) a width the codec's grid does not tile is padded, not refused
# ---------------------------------------------------------------------------


def test_a_width_that_does_not_tile_16_is_padded_on_the_stored_copy():
    odd = {**BASE, **ODD}
    off_logits, off_grads, _ = _step(_model(odd), store="bf16", vocab=ODD["vocab"])
    on_logits, on_grads, counters = _step(_model(odd), store="nvfp4", vocab=ODD["vocab"])
    assert torch.equal(off_logits, on_logits)
    assert counters.get("act_store_mm", 0) > 0, "the padded width reached no site"
    for name in off_grads:
        assert torch.isfinite(on_grads[name]).all(), name
    # The pad is zeros in the stored tail, so the real columns keep their grid:
    # the deltas stay in the same family as the tiling fixture.
    worst = max(_rel(on_grads[n], off_grads[n]) for n in off_grads if off_grads[n].norm() > 0)
    assert worst <= 0.25, f"the padded store moved a parameter by {worst:.3f}"


# ---------------------------------------------------------------------------
# (6) the refusals
# ---------------------------------------------------------------------------


def _odd_model(**over) -> ModelConfig:
    return ModelConfig(**{**BASE, **over})


def test_the_literal_is_refused_by_the_train_half():
    with pytest.raises(LoudError, match="not a saved-activation store"):
        validate_train(TrainConfig(act_store="e2m1"))


@pytest.mark.parametrize(
    ("over", "train", "match"),
    [
        ({"block_form": "kda", "n_layers": 0, "ffn_gated": False}, {}, "block_form='kda'"),
        ({"block_stack": "hybrid"}, {}, "block_stack='hybrid'"),
        ({"act_qat": "e2m1"}, {}, "act_qat='e2m1'"),
        ({"attn_form": "tsct", "attn_rank": 0}, {}, "attn_form='tsct'"),
        ({"ffn_form": "tsct"}, {}, "ffn_form='tsct'"),
        ({}, {"torch_compile": True}, "torch_compile=true"),
        ({}, {"mm_dtype": "fp8"}, "mm_dtype='fp8'"),
    ],
)
def test_every_unwired_shape_refuses(over, train, match):
    with pytest.raises(LoudError, match=match):
        validate_model_train(_odd_model(**over), TrainConfig(act_store="nvfp4", **train))


def test_the_runtime_seam_refuses_a_hand_built_block(monkeypatch):
    model = _model({"act_qat": "e2m1"})
    with pytest.raises(LoudError, match="act_qat"):
        model.set_act_store("nvfp4")
    with pytest.raises(LoudError, match="not a saved-activation store"):
        model.set_act_store("int4")

    monkeypatch.setattr(tsct, "_MM_DTYPE", "fp8")
    with pytest.raises(LoudError, match="mm_dtype='fp8'"):
        _model().set_act_store("nvfp4")


def test_the_model_seam_counts_transformer_blocks_only():
    model = _model()
    assert model.set_act_store("nvfp4") == BASE["n_layers"]
    assert all(b.act_store == "nvfp4" for b in model.transformer_blocks)

    kda = _model({"block_form": "kda", "n_layers": 0, "ffn_gated": False, "d_ffn_gelu": 0})
    with pytest.raises(LoudError, match="no transformer block"):
        kda.set_act_store("nvfp4")


def test_configure_actmem_installs_the_arm_beside_the_off_defaults():
    model = _model()
    model.configure_actmem(iter_checkpoint=False, activation_offload=0.0, act_store="nvfp4")
    assert all(b.act_store == "nvfp4" for b in model.transformer_blocks)
    assert model._act_offload is None, "the offload arm must stay uninstalled"


def test_a_hand_built_hook_free_default_still_reaches_saved_table():
    """`recompute.saved_table` is the same spy the memory ledger uses."""
    from dormouse.model.recompute import saved_table

    model = _model()
    ids, targets = _inputs()
    table, _ = saved_table(lambda: model(ids, targets)[1])
    assert table.tensors > 0 and table.bytes > 0
    model.set_act_store("nvfp4")
    packed, _ = saved_table(lambda: model(ids, targets)[1])
    assert packed.bytes < table.bytes


def test_the_trainer_installs_the_arm_once(tmp_path):
    """The seam is `train/loop.py`: an arm the trainer never installs is inert."""
    from dormouse.train import train

    cfg = TrainConfig(
        steps=2,
        batch=2,
        seq_len=32,
        lr=0.01,
        log_every=10,
        ckpt_every=100,
        ckpt_name="t",
        out_dir=str(tmp_path / "ckpts"),
        data=str(write_corpus(tmp_path / "corpus.bin")),
        act_store="nvfp4",
    )
    run = RunConfig(source="tiny", model=ModelConfig(**BASE), train=cfg)
    model = DormouseModel(run.model)
    lines: list[str] = []
    train(model, run, log=lines.append)
    assert all(b.act_store == "nvfp4" for b in model.transformer_blocks), (
        "the trainer ran a step without installing the arm"
    )
    assert any("act_store=nvfp4" in line for line in lines), lines
