"""Model-internals audit: the dense block against its own documentation.

Companion to `docs/reviews/2026-11-07-model-internals.md`. Every test is a
NUMERIC claim about `model/transformer.py` + the dense branch of `model.py`,
not a shape check, and each one is able to fail on its own:

* the RoPE convention (base, pairing, sign, norm preservation) is compared to
  an independent f64 reference built in the test — no library rotation;
* the two norms are identified (LayerNorm vs RMSNorm, gain/bias, eps location)
  against their reference formulas at 1e-6;
* the GPT-2 residual scaling is measured per tensor against `0.02/sqrt(2L)`;
* head tying is proven by parameter identity and by the checkpoint payload;
* the attention `scale`, the packed `qkv` and the causal-only mask are read off
  the live SDPA call and compared to a hand-rolled masked softmax;
* the positional extent is measured past `max_seq_len`, where RoPE has no table;
* the counters are counted through a real training window;
* every init std is compared with its documented family, and `--seed` is shown
  to reach the weights through the CLI's own `_model` path.

The claims are taken from `model/transformer.py`'s module docstring and
`scripts/control_transformer.py` (the control this form transcribes).
"""

from __future__ import annotations

import dataclasses
import math
import re

import pytest
import torch
import torch.nn.functional as F
from conftest import SEQ, tiny_cfg, tiny_run

from dormouse.config import load_run
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.transformer import apply_rope, rope_tables
from dormouse.model.tsct import DenseLinear
from dormouse.train import load_ckpt, train
from dormouse.train.loop import ckpt_paths, load_model_state

D_MODEL = 64
HEAD_DIM = 16


def _dense_cfg(**over):
    """The tiny fixture as a legal vanilla transformer (`max_iter=1`, no arms)."""
    base = dict(
        block_form="transformer",
        max_iter=1,
        loop_blocks=1,
        n_layers=2,
        use_kda=False,
        use_engram=False,
        use_tsct=False,
        moe_topk=0,
        n_experts=1,
        bf16=False,
        tie_embeddings=False,
    )
    base.update(over)
    return tiny_cfg(**base)


@pytest.fixture(scope="module")
def flagship():
    """The 95M arm, built once; the global RNG is restored afterwards."""
    run = load_run("flagship3_dense")
    state = torch.get_rng_state()
    torch.manual_seed(0)
    try:
        yield DormouseModel(run.model), run.model
    finally:
        torch.set_rng_state(state)


def _ids(t: int, batch: int = 2) -> torch.Tensor:
    return torch.arange(batch * t, dtype=torch.long).view(batch, t) % 251


# --- 1. RoPE ---------------------------------------------------------------


def _angles(t: int, head_dim: int, device="cpu") -> torch.Tensor:
    """`ang[p, j] = p · 10000^(-2j/head_dim)`, in f64, built here not imported."""
    half = head_dim // 2
    inv = 10_000.0 ** (-2.0 * torch.arange(half, device=device, dtype=torch.float64) / head_dim)
    return torch.outer(torch.arange(t, device=device, dtype=torch.float64), inv)


def _reference_split_half(x: torch.Tensor) -> torch.Tensor:
    """Complex rotation of `(x[..., :half], x[..., half:])` — the NeoX pairing."""
    half = x.shape[-1] // 2
    ang = _angles(x.shape[-2], x.shape[-1])
    z = torch.complex(x[..., :half].double(), x[..., half:].double())
    rot = z * torch.polar(torch.ones_like(ang), ang)
    return torch.cat((rot.real, rot.imag), dim=-1)


def _reference_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Adjacent pairs `(x[2k], x[2k+1])` with the SAME spectrum — the other
    convention. A swap of the pairing still trains, so it is worth separating."""
    half = x.shape[-1] // 2
    ang = _angles(x.shape[-2], x.shape[-1])
    pairs = x.reshape(*x.shape[:-1], half, 2)
    z = torch.complex(pairs[..., 0].double(), pairs[..., 1].double())
    rot = z * torch.polar(torch.ones_like(ang), ang)
    return torch.stack((rot.real, rot.imag), dim=-1).reshape(x.shape)


def test_rope_matches_the_split_half_reference_not_the_interleaved_one():
    torch.manual_seed(11)
    x = torch.randn(2, 3, 9, HEAD_DIM)
    cos, sin = rope_tables(9, HEAD_DIM, "cpu", torch.float32)
    y = apply_rope(x, cos, sin)
    split = (y.double() - _reference_split_half(x)).abs().max().item()
    inter = (y.double() - _reference_interleaved(x)).abs().max().item()
    assert split < 2e-6, f"split-half reference disagrees: {split:.3e}"
    assert inter > 0.05, f"interleaved reference agrees too well: {inter:.3e}"


def test_rope_base_is_10000_and_the_rotation_sign_is_positive():
    half = HEAD_DIM // 2
    cos, sin = rope_tables(2, HEAD_DIM, "cpu", torch.float32)
    # Position 1, channel 0: the frequency IS 1 rad/position iff the base is 1e4
    # and the exponent is 2j/head_dim. Channel 1 pins the base itself.
    inv1 = 10_000.0 ** (-1.0 / half)
    assert cos[0, 0, 1, 0].item() == pytest.approx(math.cos(1.0), abs=1e-6)
    assert sin[0, 0, 1, 0].item() == pytest.approx(math.sin(1.0), abs=1e-6)
    assert cos[0, 0, 1, 1].item() == pytest.approx(math.cos(inv1), abs=1e-6)
    # A +1 in the first half of the pair at position 1 must appear as +sin in
    # the second half: that is the sign of `R(+p)`, not `R(-p)`.
    x = torch.zeros(1, 1, 2, HEAD_DIM)
    x[0, 0, 1, 0] = 1.0
    y = apply_rope(x, cos, sin)
    assert y[0, 0, 1, 0].item() == pytest.approx(math.cos(1.0), abs=1e-6)
    assert y[0, 0, 1, half].item() == pytest.approx(math.sin(1.0), abs=1e-6)


def test_rope_preserves_the_norm_and_dot_products_in_fp32():
    torch.manual_seed(12)
    x = torch.randn(2, 3, 8, 64)  # the flagship's head_dim
    cos, sin = rope_tables(8, 64, "cpu", torch.float32)
    y = apply_rope(x, cos, sin)
    rel_norm = ((y.norm(dim=-1) - x.norm(dim=-1)).abs() / x.norm(dim=-1)).max().item()
    a, b = x[0, 0], x[0, 1]
    ya, yb = y[0, 0], y[0, 1]
    dots = (ya * yb).sum(-1) - (a * b).sum(-1)
    rel_dot = (dots.abs() / (a.norm(dim=-1) * b.norm(dim=-1))).max().item()
    assert rel_norm <= 5e-7, f"RoPE moves the norm by {rel_norm:.3e} (fp32 rounding is ~1e-7)"
    assert rel_dot <= 1e-6, f"RoPE moves a dot product by {rel_dot:.3e}"


# --- 2. RMSNorm / LayerNorm ------------------------------------------------


def test_block_norm_is_layernorm_with_gain_and_bias_at_torch_epsilon():
    """The block's norm is the CONTROL's: LayerNorm, affine, eps=1e-5.

    `norm_eps` (1e-3 in every preset) reaches the final norm only; the two
    per-block norms keep torch's default. Pinned here, proposal in the report —
    changing it changes the numerics of every dense checkpoint.
    """
    torch.manual_seed(5)
    model = DormouseModel(_dense_cfg(norm_eps=1e-3))
    blk = model.transformer_blocks[0]
    assert isinstance(blk.ln1, torch.nn.LayerNorm)
    assert blk.ln1.elementwise_affine and blk.ln1.bias is not None
    assert torch.equal(blk.ln1.weight, torch.ones(D_MODEL))
    assert torch.equal(blk.ln1.bias, torch.zeros(D_MODEL))
    assert blk.ln1.eps == 1e-5
    x = torch.randn(2, 7, D_MODEL) * 3.0 + 0.7

    def reference(eps: float) -> torch.Tensor:
        xd = x.double()
        mu = xd.mean(dim=-1, keepdim=True)
        var = xd.var(dim=-1, unbiased=False, keepdim=True)
        normed = (xd - mu) / torch.sqrt(var + eps)
        return (normed * blk.ln1.weight.double() + blk.ln1.bias.double()).float()

    out = blk.ln1(x)
    assert (out - reference(1e-5)).abs().max().item() < 1e-6
    assert (out - reference(1e-3)).abs().max().item() > 1e-5, "cfg.norm_eps reached the block norm"


def test_final_norm_is_rmsnorm_no_bias_and_epsilon_is_inside_sqrt():
    torch.manual_seed(6)
    model = DormouseModel(_dense_cfg(norm_eps=1e-3))
    norm = model.norm
    assert isinstance(norm, torch.nn.RMSNorm) and not hasattr(norm, "bias")
    assert torch.equal(norm.weight, torch.ones(D_MODEL))
    x = torch.randn(2, 5, D_MODEL) * 3.0 + 0.3
    out = norm(x.float())
    inside = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-3)
    outside = x / (torch.sqrt(x.pow(2).mean(-1, keepdim=True)) + 1e-3)
    assert (out - inside).abs().max().item() < 1e-6
    assert (out - outside).abs().max().item() > 1e-4, "epsilon is outside the sqrt"


# --- 3. residual scaling ---------------------------------------------------


def test_residual_scaling_is_on_exactly_the_two_writes_at_12_layers():
    torch.manual_seed(4)
    model = DormouseModel(_dense_cfg(n_layers=12))
    want_scaled = 0.02 / math.sqrt(2 * 12)
    for i, blk in enumerate(model.transformer_blocks):
        for name, tensor, want in (
            ("qkv", blk.qkv.weight, 0.02),
            ("fc", blk.fc.weight, 0.02),
            ("proj", blk.proj.weight, want_scaled),
            ("fc2", blk.fc2.weight, want_scaled),
        ):
            got = tensor.detach().double().std(unbiased=False).item()
            assert got == pytest.approx(want, rel=0.06), (
                f"block {i} {name}: {got:.5g} != {want:.5g}"
            )
        assert blk.qkv.bias is None and blk.proj.bias is None
        assert blk.fc.bias is None and blk.fc2.bias is None
        assert torch.all(blk.ln1.weight == 1) and torch.all(blk.ln1.bias == 0)
        assert torch.all(blk.ln2.weight == 1) and torch.all(blk.ln2.bias == 0)
    scaled = {n for n, _ in model.named_parameters() if n.endswith((".proj.weight", ".fc2.weight"))}
    assert len(scaled) == 2 * 12, "residual scaling must touch exactly 2 projections per layer"


def test_residual_scaling_reads_the_unique_layer_count():
    got = {}
    for layers in (2, 12):
        torch.manual_seed(4)
        model = DormouseModel(_dense_cfg(n_layers=layers))
        got[layers] = model.transformer_blocks[0].proj.weight.detach().double().std().item()
        assert got[layers] == pytest.approx(0.02 / math.sqrt(2 * layers), rel=0.06)
    assert got[2] / got[12] == pytest.approx(math.sqrt(6.0), rel=0.10)


# --- 4. head tying ---------------------------------------------------------


def test_the_flagship_dense_head_is_untied_and_the_count_is_the_documented_one(flagship):
    model, cfg = flagship
    assert cfg.tie_embeddings is False
    assert isinstance(model.lm_head, DenseLinear)
    assert "lm_head.weight" in model.state_dict()
    assert model.lm_head.weight is not model.embed.weight
    assert model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr()
    per_block = 4 * 832**2 + 2 * 832 * 3072 + 4 * 832
    assert per_block == 7_884_032
    assert sum(p.numel() for p in model.transformer_blocks.parameters()) == 12 * per_block
    assert param_count(model) == 95_035_200
    assert model.embed.weight.numel() + model.lm_head.weight.numel() == 2 * 256 * 832


def test_a_tied_head_is_the_embedding_by_identity_and_carries_no_key():
    torch.manual_seed(2)
    untied = DormouseModel(_dense_cfg(n_layers=2, tie_embeddings=False))
    tied = DormouseModel(_dense_cfg(n_layers=2, tie_embeddings=True))
    assert tied.lm_head._owner[0].weight is tied.embed.weight
    assert not [n for n, _ in tied.named_parameters() if n.startswith("lm_head")]
    assert "lm_head.weight" not in tied.state_dict()
    assert param_count(untied) - param_count(tied) == 256 * D_MODEL


def _one_step_run(tmp_path, name: str, *, tie: bool):
    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    run = tiny_run(out, steps=1, ckpt_every=1, log_every=100)
    return dataclasses.replace(run, model=_dense_cfg(n_layers=2, tie_embeddings=tie))


def test_the_checkpoint_stores_the_head_form_and_a_switch_refuses(tmp_path):
    payload = {}
    for tie in (False, True):
        run = _one_step_run(tmp_path, "tied" if tie else "untied", tie=tie)
        torch.manual_seed(3)
        train(DormouseModel(run.model), run, log=lambda *_: None)
        ckpt, stored = load_ckpt(run)
        assert stored.model.tie_embeddings is tie
        weights, snapshot = ckpt_paths(run.train)
        assert weights.is_file()
        assert f"tie_embeddings = {str(tie).lower()}" in snapshot.read_text()
        assert ("lm_head.weight" in ckpt["model"]) is not tie
        fresh = DormouseModel(run.model)
        result = load_model_state(fresh, run.train, ckpt["model"], strict=True)
        assert not result.missing_keys and not result.unexpected_keys, result
        payload[tie] = ckpt["model"]
    with pytest.raises(RuntimeError):
        DormouseModel(_dense_cfg(n_layers=2, tie_embeddings=True)).load_state_dict(
            payload[False], strict=True
        )
    with pytest.raises(RuntimeError):
        DormouseModel(_dense_cfg(n_layers=2, tie_embeddings=False)).load_state_dict(
            payload[True], strict=True
        )
    a = _one_step_run(tmp_path, "a", tie=False)
    b = dataclasses.replace(a, model=_dense_cfg(n_layers=2, tie_embeddings=True))
    assert "model.tie_embeddings" in a.diff_keys(b)


# --- 5. attention details --------------------------------------------------


def test_attention_scale_is_inverse_sqrt_head_dim_and_the_mask_is_causal_only(monkeypatch):
    torch.manual_seed(7)
    model = DormouseModel(_dense_cfg(n_layers=1))
    blk = model.transformer_blocks[0]
    assert blk.scale == 1.0 / math.sqrt(blk.head_dim)
    seen: dict[str, object] = {}
    original = F.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        seen.update(kw)
        seen["q"] = q.detach().clone()
        seen["k"] = k.detach().clone()
        seen["v"] = v.detach().clone()
        out = original(q, k, v, **kw)
        seen["out"] = out.detach().clone()
        return out

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    ids = _ids(SEQ)
    with torch.no_grad():
        model(ids)
    assert seen["is_causal"] is True
    assert "attn_mask" not in seen, "the dense block has no padding mask (no padding exists)"
    assert seen["scale"] == blk.scale
    # The captured call, reproduced by hand: softmax(QKᵀ/√hd + causal) V.
    q, k, v = (seen[n][0, 0].float() for n in ("q", "k", "v"))  # [t, hd]
    scores = (q @ k.T) * blk.scale
    future = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    ref = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1) @ v
    assert (seen["out"][0, 0].float() - ref).abs().max().item() < 2e-5


def test_qkv_is_one_packed_projection_of_three_d_squared(monkeypatch):
    torch.manual_seed(8)
    model = DormouseModel(_dense_cfg(n_layers=2))
    blk = model.transformer_blocks[0]
    assert blk.qkv.weight.shape == (3 * D_MODEL, D_MODEL)
    assert blk.qkv.bias is None
    assert not any(hasattr(blk, n) for n in ("q", "k", "v"))
    x = torch.randn(2, 5, D_MODEL)
    q, k, v = blk.qkv(x).split(D_MODEL, dim=-1)
    w = blk.qkv.weight
    assert torch.allclose(q, x @ w[:D_MODEL].t(), atol=1e-6)
    assert torch.allclose(k, x @ w[D_MODEL : 2 * D_MODEL].t(), atol=1e-6)
    assert torch.allclose(v, x @ w[2 * D_MODEL :].t(), atol=1e-6)
    # `_block` never calls `self.qkv(x)`: it multiplies the weight through
    # `mm_mixed`, so a module hook cannot see it. Count the GEMMs instead.
    from dormouse.model import transformer as tf

    original = tf.mm_mixed
    shapes: list[tuple[int, ...]] = []

    def spy(x, weight, dt, counters=None):
        shapes.append(tuple(weight.shape))
        return original(x, weight, dt, counters)

    monkeypatch.setattr(tf, "mm_mixed", spy)
    with torch.no_grad():
        model(_ids(SEQ))
    packed = [s for s in shapes if s == (3 * D_MODEL, D_MODEL)]
    assert len(packed) == len(model.transformer_blocks), "one packed qkv GEMM per block"
    assert len(shapes) == 4 * len(model.transformer_blocks), "four projections per block"


def test_attention_is_causal_and_no_position_reads_a_future_token():
    torch.manual_seed(9)
    model = DormouseModel(_dense_cfg(n_layers=2))
    ids = _ids(8)
    changed = ids.clone()
    changed[:, 6:] = 7
    with torch.no_grad():
        before, _, _ = model(ids)
        after, _, _ = model(changed)
    delta = (before - after).abs().reshape(2, 8, 256)
    assert delta[:, :6].max().item() == 0.0
    assert delta[:, 6:].max().item() > 0.0


# --- 6. positional extent --------------------------------------------------


def test_forward_extrapolates_past_max_seq_len_because_rope_has_no_table():
    torch.manual_seed(10)
    model = DormouseModel(_dense_cfg(n_layers=2, max_seq_len=16))
    assert not [n for n, _ in model.named_buffers() if "rope" in n.lower()]
    long_ids = _ids(48)
    with torch.no_grad():
        logits, loss, _ = model(long_ids, long_ids)
    assert logits.shape == (2 * 48, 256)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    cos48, sin48 = rope_tables(48, HEAD_DIM, "cpu", torch.float32)
    cos16, sin16 = rope_tables(16, HEAD_DIM, "cpu", torch.float32)
    assert torch.equal(cos48[0, 0, :16], cos16[0, 0])
    assert torch.equal(sin48[0, 0, :16], sin16[0, 0])


def test_load_run_refuses_seq_len_above_max_seq_len():
    with pytest.raises(LoudError, match="exceeds max_seq_len"):
        load_run("flagship3_dense", train_overrides={"seq_len": 2048})
    run = load_run("flagship3_dense", train_overrides={"seq_len": 1024})
    assert run.train.seq_len == run.model.max_seq_len == 1024


# --- 7. counters -----------------------------------------------------------


def test_transformer_block_counter_is_one_per_block_per_forward():
    torch.manual_seed(13)
    model = DormouseModel(_dense_cfg(n_layers=12))
    ids = _ids(SEQ)
    counters = Counters()
    for _ in range(3):
        with torch.no_grad():
            model(ids, ids, counters=counters)
    assert counters["transformer_block"] == 12 * 3
    assert counters["forward"] == 3


def test_a_window_count_is_layers_x_forwards_and_grad_accum_multiplies(tmp_path):
    """`12 x steps` is only the block count at grad_accum=1; the log line's
    `forward` is already `steps x grad_accum`, and blocks follow it."""
    out = tmp_path / "window"
    out.mkdir(parents=True, exist_ok=True)
    run = tiny_run(out, steps=1, log_every=1, grad_accum=3)
    run = dataclasses.replace(run, model=_dense_cfg(n_layers=12))
    lines: list[str] = []
    torch.manual_seed(14)
    train(DormouseModel(run.model), run, log=lines.append)
    line = next(line for line in lines if line.startswith("step=1 "))
    blocks = int(re.search(r"transformer_block=(\d+)", line).group(1))
    forwards = int(re.search(r"\bforward=(\d+)", line).group(1))
    assert forwards == 3, line
    assert blocks == 12 * forwards == 36, line


# --- 8. init vs config -----------------------------------------------------


def test_a_negative_layer_count_is_refused_instead_of_building_no_blocks():
    """`range(-1)` is empty: today a negative count built embed -> norm -> head
    and the snapshot still reported the count, i.e. a model with no blocks."""
    with pytest.raises(LoudError, match="n_layers"):
        _dense_cfg(n_layers=-1)


def test_parameter_stds_match_the_documented_init(flagship):
    model, _ = flagship
    scaled = 0.02 / math.sqrt(2 * 12)
    print("\nper-tensor std (flagship3_dense, 95M):")
    for name, param in model.named_parameters():
        std = param.detach().double().std(unbiased=False).item()
        print(f"  {name:44s} {tuple(param.shape)!s:18s} std={std:.6g}")
    for i, blk in enumerate(model.transformer_blocks):
        for tensor_name, tensor, want in (
            ("qkv", blk.qkv.weight, 0.02),
            ("fc", blk.fc.weight, 0.02),
            ("proj", blk.proj.weight, scaled),
            ("fc2", blk.fc2.weight, scaled),
            ("ln1.weight", blk.ln1.weight, 0.0),
            ("ln1.bias", blk.ln1.bias, 0.0),
        ):
            std = tensor.detach().double().std(unbiased=False).item()
            if want == 0.0:
                assert std == 0.0, f"block {i} {tensor_name} is not constant"
            else:
                assert std == pytest.approx(want, rel=0.06), f"block {i} {tensor_name}"
    # The embedding and the untied head are OUTSIDE the block's init claim, and
    # neither is the control's N(0, 0.02): torch's N(0, 1) embedding and
    # DenseLinear's 1/sqrt(in) head. The report calls this out.
    assert model.embed.weight.detach().double().std().item() == pytest.approx(1.0, rel=0.03)
    assert model.lm_head.weight.detach().double().std().item() == pytest.approx(
        1.0 / math.sqrt(832), rel=0.03
    )


def test_seed_reaches_every_parameter_of_the_init(tmp_path):
    from dormouse.cli import _model

    out = tmp_path / "seed"
    out.mkdir(parents=True, exist_ok=True)
    run = tiny_run(out, steps=1)
    run = dataclasses.replace(run, model=_dense_cfg(n_layers=2))
    first = _model(run)
    torch.rand(4096)  # consume the stream: _model must rewind, not continue it
    second = _model(run)
    pairs = zip(first.named_parameters(), second.named_parameters(), strict=True)
    for (name, a), (_, b) in pairs:
        assert torch.equal(a, b), f"{name} moved between two same-seed builds"
    other = dataclasses.replace(run, train=dataclasses.replace(run.train, seed=run.train.seed + 1))
    third = _model(other)
    assert not torch.equal(first.embed.weight, third.embed.weight)
    states = zip(first.state_dict().values(), third.state_dict().values(), strict=True)
    assert any(not torch.equal(a, b) for a, b in states)
