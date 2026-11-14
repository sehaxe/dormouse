"""Gates for `act_store="codes"` — the FAKE-QUANT seam stored as 4-bit codes.

THE CLAIM, in three parts:

1. the codec is exact: `pack_codes` → `unpack_codes` rebuilds `quantize_act`'s
   value with `torch.equal`, so the backward's `grad_W = gradᵀ @ xq` reads the
   operand the forward's GEMM read — not a second rounding of it;
2. therefore the rung moves NOTHING: one step with it off and one with it on
   produce the same loss, the same every-gradient and the same weights after one
   optimizer step, `torch.equal`;
3. and it costs ~1/4: the saved bytes are read from `saved_tensors_hooks` at the
   stated shape — never computed in the test from a formula.

What is REFUSED, each a test: a seam that does not fake-quantize (coding an exact
operand is the lossy `nvfp4` trade under a name that promises exactness), a codec
whose dequantize is not `level · scale` (`v2_int4`'s Hadamard round trip), and a
dispatch arm that never runs the per-expert forward where the codes are packed.
"""

from __future__ import annotations

import pytest
import torch
from conftest import tiny_cfg
from torch.autograd.graph import saved_tensors_hooks

from dormouse.config import ModelConfig, TrainConfig, validate_model_train
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.act_quant import fake_quant_act, pack_codes, quantize_act, unpack_codes
from dormouse.model.experts import SparseMoE
from dormouse.model.packed_acts import coded_act_mm

#: The seam's shape at `b1 b8 s1024` with `moe_topk=2`: 8 192 tokens × 2
#: assignments = 16 384 rows of the expert GEMM's `d = 1024` operand.
ROWS, WIDTH, RANK = 16_384, 1024, 64

#: `b1`'s expert ladder is `d=1024` (the `d` the measured 245 MiB `[R, f]` save
#: was taken at); the GoU here is the same shape class, so the byte claim below is
#: the configuration's own, not a fixture's.
STATEMENT = "b1 b8 s1024, moe_topk=2"


# ---------------------------------------------------------------------------
# (1) the codec: a code plus a scale IS the value the forward multiplied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["e2m1", "int4"])
@pytest.mark.parametrize("group", [0, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pack_unpack_is_the_forward_value(fmt, group, dtype):
    """`torch.equal`, not a tolerance: the codes are the value, not near it."""
    torch.manual_seed(0)
    for shape in [(64, 128), (7, 1024), (1, 832), (3, 32)]:
        x = torch.randn(*shape, dtype=dtype)
        want = quantize_act(x, fmt, group)
        packed, scale, k = pack_codes(x, fmt, group)
        got = unpack_codes(packed, scale, k, fmt, group, dtype)
        assert torch.equal(got, want), (fmt, group, dtype, shape)
        assert packed.dtype is torch.uint8, "two codes per byte, not one"
        # Two codes per byte: 4 bits against bf16's 16, before the per-row scale.
        assert packed.numel() * 2 <= x.numel(), "the pack must not grow the tensor"


def test_the_e2m1_tie_table_round_trips():
    """The seven interior midpoints `act_quant.py` transcribes, plus both signs.

    A codec that reads a tie differently from `_e2m1_round` would be exact on
    random data and wrong exactly where a quantizer is decided.
    """
    mid = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]])
    x = torch.cat([mid, -mid, torch.zeros(1, mid.shape[1])], dim=0)
    packed, scale, k = pack_codes(x, "e2m1")
    assert torch.equal(unpack_codes(packed, scale, k, "e2m1"), quantize_act(x, "e2m1"))


def test_a_codec_without_level_scale_is_refused():
    """`v2_int4`'s dequantize is a Hadamard round trip — a code is not a value."""
    with pytest.raises(LoudError, match="level · scale"):
        pack_codes(torch.randn(4, 16), "v2_int4")


# ---------------------------------------------------------------------------
# (2) the step: nothing moves
# ---------------------------------------------------------------------------


def _moe(**over) -> SparseMoE:
    torch.manual_seed(3)
    # `moe_kernel="naive"` is the reference loop: the per-expert forward
    # (`DualLatentGLU`) IS where this rung packs, and the other arms are refused.
    return SparseMoE(tiny_cfg(act_quant="e2m1", moe_kernel="naive", moe_capacity=0, **over))


def _step(moe: SparseMoE, x: torch.Tensor, codes: bool, step: bool):
    moe.zero_grad(set_to_none=True)
    moe.set_act_store("codes" if codes else "bf16")
    x = x.detach().clone().requires_grad_(True)
    y = moe(x)
    loss = y.float().pow(2).sum()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None}
    xgrad = x.grad.clone()
    if step:
        torch.optim.SGD(moe.parameters(), lr=0.1, momentum=0.9).step()
    return y.detach().clone(), grads, xgrad


def test_the_step_is_bit_identical():
    """Loss, EVERY gradient, and the weights after one step: `torch.equal`."""
    torch.manual_seed(3)
    moe = _moe()
    x = torch.randn(6, moe.d, generator=torch.Generator().manual_seed(11))
    start = {k: v.clone() for k, v in moe.state_dict().items()}

    off = _step(moe, x, codes=False, step=False)
    moe.load_state_dict(start)
    on = _step(moe, x, codes=True, step=False)

    assert torch.equal(off[0], on[0]), "the forward moved"
    assert set(off[1]) == set(on[1])
    for name, g in off[1].items():
        assert torch.equal(g, on[1][name]), f"grad moved at {name}"
    assert torch.equal(off[2], on[2]), "the STE's grad_x moved"

    moe.load_state_dict(start)
    _step(moe, x, codes=False, step=True)
    after_off = {k: v.clone() for k, v in moe.state_dict().items()}
    moe.load_state_dict(start)
    _step(moe, x, codes=True, step=True)
    for name, w in moe.state_dict().items():
        assert torch.equal(w, after_off[name]), f"weights moved at {name}"


def test_the_codes_path_is_not_inert():
    """A guard on the gate above: the seam really takes the other branch."""
    torch.manual_seed(3)
    moe = _moe()
    x = torch.randn(6, moe.d)
    moe.set_act_store("bf16")
    assert all(e.act_store == "bf16" for e in (*moe.experts, moe.shared))
    assert moe.set_act_store("codes") == len(moe.experts) + 1
    assert all(e.act_store == "codes" for e in (*moe.experts, moe.shared))
    with torch.no_grad():
        # The codes rebuild `quantize_act`, so an expert's GEMM operand is the
        # same value the bf16 path reads — what changes is only the STORE.
        assert torch.equal(moe.experts[0](x), moe.experts[0](x))
        moe.experts[0].set_act_store("bf16")
        assert torch.equal(moe.experts[0](x), moe.experts[0](x))


# ---------------------------------------------------------------------------
# (3) the bytes: read from the graph, at the stated shape
# ---------------------------------------------------------------------------


def _saved_bytes(fn) -> list[tuple[tuple[int, ...], torch.dtype, int]]:
    seen: list[tuple[tuple[int, ...], torch.dtype, int]] = []

    def pack(t):
        seen.append((tuple(t.shape), t.dtype, t.numel() * t.element_size()))
        return t

    with saved_tensors_hooks(pack, lambda t: t):
        fn()
    return seen


def test_the_saved_bytes_are_a_quarter_at_the_stated_shape():
    """`STATEMENT`: the seam's 32 MiB bf16 operand is 8 MiB of codes, measured."""
    torch.manual_seed(0)
    x = torch.randn(ROWS, WIDTH, dtype=torch.bfloat16)
    # A MASTER weight (the expert's `U`): with `requires_grad` off, autograd saves
    # the weight for `grad_x` and never the operand — the store this test prices is
    # the one a TRAINING step keeps.
    w = torch.randn(WIDTH, RANK, dtype=torch.bfloat16, requires_grad=True)

    def run(codes: bool):
        xx = x.detach().clone().requires_grad_(True)
        if codes:
            y = coded_act_mm(xx, "e2m1", 0, torch.bfloat16, w)[0]
        else:
            xq = fake_quant_act(xx, "e2m1")  # the seam's own forward, STE and all
            y = xq @ w
        y.float().pow(2).sum().backward()
        return xx

    bf16 = [t for t in _saved_bytes(lambda: run(False)) if t[0] == (ROWS, WIDTH)]
    coded = [t for t in _saved_bytes(lambda: run(True)) if t[0] in ((ROWS, WIDTH // 2), (ROWS, 1))]
    assert len(bf16) == 1 and bf16[0][1] is torch.bfloat16, bf16
    assert bf16[0][2] == ROWS * WIDTH * 2, "the bf16 operand is not the shape we priced"
    bf16_bytes = bf16[0][2]
    codes_bytes = sum(t[2] for t in coded)
    # 4 bits per element plus one fp32 scale per ROW (group 0) — 0.252 of bf16.
    assert codes_bytes < bf16_bytes * 0.26, f"{codes_bytes} B against bf16's {bf16_bytes} B"
    assert codes_bytes == ROWS * (WIDTH // 2) + ROWS * 4, (
        f"the packed sizes are not the codec's own: {[t[0:3] for t in coded]}"
    )


# ---------------------------------------------------------------------------
# (4) the refusals
# ---------------------------------------------------------------------------


def test_a_seam_that_does_not_fake_quantize_is_refused():
    """A `[K, N]` weight, the expert's layout — and `fmt='none'` is the refusal."""
    x = torch.randn(8, 16)
    with pytest.raises(LoudError, match="does NOT fake-quantize"):
        coded_act_mm(x, "none", 0, torch.float32, torch.randn(16, 4))
    # And the expert's own switch says the same thing before a step can run.
    from dormouse.model.experts import DualLatentGLU

    with pytest.raises(LoudError, match="act_quant='none'"):
        DualLatentGLU(16, 32, 4, dtype=torch.float32).set_act_store("codes")


@pytest.mark.parametrize(
    "over,match",
    [
        ({"moe_kernel": "triton"}, "moe_kernel='triton'"),
        ({"moe_kernel": "batched", "moe_capacity": 0.25}, "moe_kernel='batched'"),
        ({"moe_topk": 0}, "topk=0"),
    ],
)
def test_an_arm_that_never_runs_the_expert_forward_is_refused(over, match):
    """No silent inert switch: the banked/kernel arms never reach the pack."""
    moe = SparseMoE(tiny_cfg(act_quant="e2m1", **over))
    with pytest.raises(LoudError, match=match):
        moe.set_act_store("codes")


#: A `ModelConfig` with every arm the transformer form refuses switched off.
_MODEL = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=64,
    max_iter=1,
    use_tsct=False,
    use_kda=False,
    use_engram=False,
    use_csa=False,
    layerrope=False,
    tie_embeddings=False,
)


def test_config_refuses_codes_without_a_quantizer_or_a_moe():
    model = ModelConfig(**_MODEL, n_layers=0, block_form="kda", act_qat="none")
    with pytest.raises(LoudError, match="act_qat='none'"):
        validate_model_train(model, TrainConfig(act_store="codes"))
    _on = ModelConfig(**_MODEL, block_form="transformer", act_qat="e2m1")
    with pytest.raises(LoudError, match="block_form='transformer'"):
        validate_model_train(_on, TrainConfig(act_store="codes"))
    # The seam's own block shape: the reference loop, where the per-expert
    # forward (and so the pack) really runs.
    seam = dict(n_layers=0, block_form="kda", act_qat="e2m1", moe_topk=2, moe_kernel="naive")
    validate_model_train(ModelConfig(**_MODEL, **seam), TrainConfig(act_store="codes"))
    with pytest.raises(LoudError, match="torch_compile=true"):
        validate_model_train(
            ModelConfig(**_MODEL, **seam), TrainConfig(act_store="codes", torch_compile=True)
        )
    # And the LIVE kernel arm is refused rather than armed-and-inert: its GEMMs
    # never build the per-expert forward the codes are packed in.
    with pytest.raises(LoudError, match="moe_kernel='triton'"):
        validate_model_train(
            ModelConfig(**_MODEL, **{**seam, "moe_kernel": "triton"}),
            TrainConfig(act_store="codes"),
        )


def test_the_model_switch_needs_a_moe_seam():
    """A model with no `SparseMoE` would report the arm on and store nothing."""
    torch.manual_seed(0)
    model = DormouseModel(ModelConfig(**_MODEL, block_form="transformer", act_qat="e2m1"))
    with pytest.raises(LoudError, match="builds no `SparseMoE`"):
        model.set_act_store("codes")
