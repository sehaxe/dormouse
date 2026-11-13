"""`byte_patch`: the dense transformer stack over the conv codec's PATCHES.

Gate map (the task's letters):

* (a) `test_byte_patch_off_is_bit_identical` — params and logits sha256 taken
  on clean main `550bc43` BEFORE the flag existed, for the transformer control
  and for the existing KDA+conv channel. `byte_patch` absent and `false` must
  both reproduce them.
* (b) `test_causal_invariance_at_every_byte_offset` — `bytes[:, q:]` must not
  move `logits[:, :q]` (and therefore the loss at every position `<= q`) for
  every q in the window, fp32 and bf16. Runs inside patches and across patch
  boundaries (q = 1..T at 8 bytes/patch covers offsets 1..15).
* (c) `test_the_stack_sees_patches_and_twelve_unique_bodies` — a forward hook
  pins the first block's input shape to `[b, T/patch_bytes, d_model]`, and 12
  blocks are 12 distinct parameter sets, not one repeated.
* (d) `test_the_shipped_config_count_and_cli` — `configs/arch_dense_byteflow8.toml`
  builds 97,584,896 parameters, and `dormouse.cli info` prints that number.
  The CLI output is pasted in the test below.
* (e) `test_decode_is_per_byte` — byte logits have shape `[b*t, 256]` and the
  returned loss is the mean CE over BYTES (the same expression a byte model
  uses), so BPB stays comparable.
* (f) `test_the_impossible_combinations_refuse_loudly` — every rejected shape.

The 2026-11-03 causal fix this form inherits: `ConvCodec._patches` (two
left-padded depthwise Conv1d, 5 bytes of left context, and a patchify shifted
by one patch) makes patch `i` a function of bytes `<= i·s - 1`, and
`decode_chunks` gathers chunk `patch(j)`. Gate (b) is the end-to-end proof.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.config import ModelConfig, load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count

runner = CliRunner()

#: sha256 of the transformer control's params/logits on clean main `550bc43`
#: (no `byte_patch` field in the source), seed 1234, b=2 T=16.
TRANSFORMER_PARAMS_SHA256 = "c7fd09f5fb3571c7d9a9a61150b44494c78b8cea6209f141170dabe8a949d35b"
TRANSFORMER_LOGITS_SHA256 = "e4347bc896fc26f7e18ad91bfe3fece7a9881545bcd491c163b0b04325b8e8c3"

#: The same pin for the existing KDA+conv channel (Option B), seed 1234.
CHANNEL_PARAMS_SHA256 = "4f03ae9ce8cf6ce8ca5447802eee338868e9c3442ccd046cded2fe3bf57d7c85"
CHANNEL_LOGITS_SHA256 = "39ce579e2ebdf928966e9a800124c0fb79e2dda1f86cef81817e5e5bfc83e727"

#: One dense block's worth of config, tiny: 2 unique blocks over 16 bytes.
TRANSFORMER = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=32,
    max_iter=1,
    loop_blocks=1,
    n_layers=2,
    block_form="transformer",
    use_kda=False,
    use_csa=False,
    use_mhc=False,
    use_attnres=False,
    use_engram=False,
    layerrope=False,
    n_experts=1,
    moe_topk=0,
    use_tsct=False,
    bf16=False,
)

#: The byte_patch form: 8 bytes/patch, so 32 bytes -> 4 positions.
BYTE_PATCH = dict(
    TRANSFORMER,
    max_seq_len=32,
    n_layers=3,
    use_byteflow=True,
    byte_patch=True,
    byteflow_codec="conv",
    byteflow_patch_bytes=8,
    byteflow_k_tokens=4,  # 32 / 8: K = P, the only causal shape
    byteflow_max_bytes=32,
    byteflow_d_local=16,
    byteflow_heads_local=2,
    byteflow_d_global=16,
    byteflow_heads_global=2,
)

#: The existing Option B channel (KDA loop + conv codec), 4 bytes/patch.
CHANNEL = dict(
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
    shared_rank=16,
    bf16=False,
    use_byteflow=True,
    use_engram=False,
    use_kda=True,
    use_csa=False,
    use_mhc=False,
    use_attnres=False,
    loom=False,
    byteflow_d_local=16,
    byteflow_d_global=16,
    byteflow_heads_local=2,
    byteflow_heads_global=2,
    byteflow_e_layers=1,
    byteflow_w_local=8,
    byteflow_d_ff_local=32,
    byteflow_bins=4,
    byteflow_k_tokens=4,
    byteflow_max_bytes=32,
    byteflow_codec="conv",
    byteflow_patch_bytes=4,
)

T, B = 32, 2


def cfg(**over: object) -> ModelConfig:
    base = dict(BYTE_PATCH)
    base.update(over)
    return ModelConfig(**base)


def ids(t: int = T, b: int = B) -> torch.Tensor:
    return torch.arange(b * t, dtype=torch.long).reshape(b, t) % 251


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().to(torch.float32).numpy().tobytes()).hexdigest()


def params_sha(model: DormouseModel) -> str:
    return hashlib.sha256(
        b"".join(sha(p).encode() for _, p in model.named_parameters())
    ).hexdigest()


# ---------------------------------------------------------------------------
# (a) the flag off is bit-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,base,pin_params,pin_logits",
    [
        ("transformer", TRANSFORMER, TRANSFORMER_PARAMS_SHA256, TRANSFORMER_LOGITS_SHA256),
        ("channel_conv", CHANNEL, CHANNEL_PARAMS_SHA256, CHANNEL_LOGITS_SHA256),
    ],
)
@pytest.mark.parametrize("explicit_false", [False, True], ids=["absent", "false"])
@pytest.mark.pins
def test_byte_patch_off_is_bit_identical(name, base, pin_params, pin_logits, explicit_false):
    """`byte_patch=false` (the default) is today's model, bit for bit."""
    over = dict(base)
    if explicit_false:
        over["byte_patch"] = False
    torch.manual_seed(1234)
    model = DormouseModel(ModelConfig(**over))
    assert params_sha(model) == pin_params, f"{name}: params moved off the 550bc43 pin"
    x = ids(t=16)
    y = (x + 1) % 251
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x, y)
    assert sha(logits) == pin_logits, f"{name}: logits moved off the 550bc43 pin"


# ---------------------------------------------------------------------------
# (b) causality: the 2026-11-03 codec guarantee, end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_causal_invariance_at_every_byte_offset(bf16: bool):
    """Changing byte `q` onward must not move any logit/loss at position `< q`."""
    torch.manual_seed(1234)
    model = DormouseModel(cfg(bf16=bf16))
    x = ids()
    model.eval()
    with torch.no_grad():
        base, _, counters = model(x)
    assert counters.get("bf_channel") == 1, counters
    assert counters.get("bf_net") is None, counters
    base = base.reshape(B, T, 256)

    for q in range(1, T + 1):
        gen = torch.Generator().manual_seed(7 + q)
        changed = x.clone()
        changed[:, q:] = torch.randint(0, 256, changed[:, q:].shape, generator=gen)
        with torch.no_grad():
            out, _, _ = model(changed)
        out = out.reshape(B, T, 256)
        assert torch.equal(base[:, :q], out[:, :q]), (
            f"{'bf16' if bf16 else 'fp32'}: changing bytes[:, {q}:] moved "
            f"logits[:, :{q}] by max |Δ|={(base[:, :q] - out[:, :q]).abs().max().item():.3e}"
        )
        ce_base = -torch.log_softmax(base[:, :q], dim=-1).gather(2, y_all()[:, :q].unsqueeze(-1))
        ce_out = -torch.log_softmax(out[:, :q], dim=-1).gather(2, y_all()[:, :q].unsqueeze(-1))
        assert torch.equal(ce_base, ce_out), f"loss at positions <= {q} moved"


def y_all() -> torch.Tensor:
    x = ids()
    return (x + 1) % 251


# ---------------------------------------------------------------------------
# (c) the stack really sees patches, with 12 unique bodies
# ---------------------------------------------------------------------------


def test_the_stack_sees_patches_and_twelve_unique_bodies():
    torch.manual_seed(1234)
    model = DormouseModel(cfg(n_layers=12))
    assert model.bf is not None and model.bf_net is None
    assert not hasattr(model, "embed") and not hasattr(model, "lm_head"), (
        "the codec is both ends; a byte embedding/lm_head here would be dead params"
    )
    blocks = list(model.transformer_blocks)
    assert len(blocks) == 12
    seen: list[tuple[int, ...]] = []
    handle = blocks[0].register_forward_pre_hook(lambda _m, args: seen.append(tuple(args[0].shape)))
    model.eval()
    with torch.no_grad():
        _, _, counters = model(ids())
    handle.remove()
    assert seen == [(B, T // 8, 64)], seen
    assert counters.get("transformer_block") == 12, counters
    # 12 unique bodies: no weight tensor is shared/repeated.
    assert len({b.qkv.weight.data_ptr() for b in blocks}) == 12
    assert len({b.fc.weight.data_ptr() for b in blocks}) == 12
    # Every parameter receives a gradient: the codec is both ends, so a
    # disconnected piece (or a dormant embed/head) would fail here.
    model.train()
    _, loss, _ = model(ids(), y_all())
    loss.backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


# ---------------------------------------------------------------------------
# (d) the shipped config: exact count, and the CLI the run uses
# ---------------------------------------------------------------------------


def test_the_shipped_config_count_and_cli():
    """`dormouse.cli info arch_dense_byteflow8` on the shipped .toml:

    ╭──────────────────────────────────────────────────────────────╮
    │ arch_dense_byteflow8                                         │
    ╰───────────────────────────── 97,584,896 params ──────────────╯
    ┌────────────────────┬────────────┐
    │ module             │     params │
    ├────────────────────┼────────────┤
    │ transformer_blocks │ 94,608,384 │
    │ norm               │        832 │
    │ bf                 │  2,975,680 │
    │ total              │ 97,584,896 │
    └────────────────────┴────────────┘
    """
    run = load_run("arch_dense_byteflow8")
    model = DormouseModel(run.model, device="meta")
    assert param_count(model) == 97_584_896
    assert sum(p.numel() for p in model.transformer_blocks.parameters()) == 94_608_384
    assert sum(p.numel() for p in model.bf.parameters()) == 2_975_680
    r = runner.invoke(app, ["info", "arch_dense_byteflow8"])
    assert r.exit_code == 0, r.output
    assert "97,584,896" in r.output


# ---------------------------------------------------------------------------
# (e) the decode path: per BYTE logits and per BYTE loss
# ---------------------------------------------------------------------------


def test_decode_is_per_byte():
    torch.manual_seed(1234)
    model = DormouseModel(cfg())
    x = ids()
    y = (x + 1) % 251
    model.eval()
    with torch.no_grad():
        logits, loss, counters = model(x, y)
    assert tuple(logits.shape) == (B * T, 256), logits.shape  # per BYTE, not per patch
    expected = -torch.log_softmax(logits, dim=-1).gather(1, y.reshape(-1, 1)).mean()
    assert torch.equal(loss, expected), (float(loss), float(expected))
    assert counters.get("bf_channel") == 1 and counters.get("transformer_block") == 3


# ---------------------------------------------------------------------------
# (f) every combination that cannot work is LOUD
# ---------------------------------------------------------------------------


def test_the_impossible_combinations_refuse_loudly():
    cases = [
        (dict(use_byteflow=False), "use_byteflow=false"),
        (dict(block_form="kda", n_layers=0, use_kda=True), "block_form='kda'"),
        (dict(byteflow_codec="attention"), "byteflow_codec='attention'"),
        (dict(byteflow_k_tokens=3), "byteflow_k_tokens=3"),
        (dict(max_seq_len=30), "not a multiple"),
        (dict(mod_capacity=0.5), "mod_capacity=0.5"),
        (dict(use_tsct=True), "use_tsct=true"),
    ]
    for over, match in cases:
        with pytest.raises(LoudError, match=match):
            cfg(**over)
    # The old transformer refusal is still there, now naming the escape.
    with pytest.raises(LoudError, match="byte_patch=true"):
        ModelConfig(**dict(TRANSFORMER, use_byteflow=True))
    # The SEQUENCE half: a window the strided patchify cannot tile is refused
    # at load, not at the first step.
    with pytest.raises(LoudError, match="seq_len=1001"):
        load_run("arch_dense_byteflow8", train_overrides={"seq_len": 1001})


def test_the_patch_form_publishes_the_breakdown_the_eval_loop_reads():
    """A byte_patch forward with `breakdown=True` must write `last_breakdown["fin"]`.

    The held-out eval reads exactly that key (`train/loop.py:_eval_loss`), and
    this path used to leave `last_breakdown` as `{}` — so every patch eval died
    with `KeyError: 'fin'`. Found by E1 on 2026-11-08: the arm had never been
    evaluated end to end, which is why the bug survived nine gates.
    """
    cfg = ModelConfig(**BYTE_PATCH)
    model = DormouseModel(cfg)
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 32))
    _, loss, _ = model(ids, ids.roll(-1, dims=1), breakdown=True)
    assert loss is not None
    bd = model.last_breakdown
    assert "fin" in bd, f"the eval loop would raise KeyError: 'fin' (got {sorted(bd)})"
    assert "per_iter" in bd and bd["per_iter"] == []
    assert abs(bd["fin"] - float(loss.detach())) < 1e-6
