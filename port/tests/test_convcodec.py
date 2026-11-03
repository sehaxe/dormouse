"""ConvCodec: the `byteflow_codec = "conv"` arm, and the attention default's pin.

Gate map (the task's letters):

* (a) `test_default_attention_codec_stays_bit_identical` — the sha256 was taken
  on clean main (`21a97d2`) BEFORE `convcodec.py` existed, with
  `/tmp/opencode/pin_attention.py`'s exact config and seed. The default and an
  explicit `"attention"` must both reproduce it.
* (b) `test_conv_shapes_are_the_contract`,
  `test_conv_every_parameter_gets_a_gradient`,
  `test_conv_forward_does_not_host_sync`.
* (c) `test_codec_name_and_patch_arms_are_loud`,
  `test_conv_requires_use_byteflow`, `test_conv_with_segment_len_is_loud`,
  `test_conv_geometry_is_loud`.
* (d) `test_conv_stays_under_the_cost_ceilings` — parameter count and MACs per
  patch against the ceilings `convcodec.py` states, so an edit cannot quietly
  grow the codec back to attention cost.
* (e) is the full suite, not this file.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from dormouse.config import ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import ConvCodec, DormouseModel
from dormouse.model.convcodec import CODEC_MAC_CEILING, CODEC_PARAM_CEILING

#: sha256 of the default (attention) codec's `[32, 256]` fp32 logits on clean
#: main `21a97d2`, 2026-10-30, from `/tmp/opencode/pin_attention.py`.
ATTENTION_LOGITS_SHA256 = "8abbcc64f0611f813b7de1d90ab9e953de8240379204d8f426b48e0c68714bbd"

#: The pin's config (tiny, CPU, one byteflow encoder layer, K=8 of T=16 bytes).
PIN_CFG = dict(
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
    use_byteflow=True,
    use_engram=False,
    use_kda=False,
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
    byteflow_k_tokens=8,
    byteflow_max_bytes=32,
)


def cfg(**over) -> ModelConfig:
    base = dict(PIN_CFG)
    base.update(over)
    return ModelConfig(**base)


def ids_targets(b: int = 2, t: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(b * t, dtype=torch.long).reshape(b, t) % 251
    return x, (x + 1) % 251


def codec(**over) -> ConvCodec:
    args = dict(patch_bytes=4, width=8, rank=4)
    args.update(over)
    return ConvCodec(16, **args)


# ---------------------------------------------------------------------------
# (a) the attention default is the pin
# ---------------------------------------------------------------------------


def _logit_sha(model: DormouseModel, x: torch.Tensor, y: torch.Tensor) -> str:
    model.eval()
    with torch.no_grad():
        logits, loss, counters = model(x, y)
    assert tuple(logits.shape) == (x.numel(), 256)
    assert counters.get("bf_channel") == 1, counters
    return hashlib.sha256(logits.contiguous().to(torch.float32).numpy().tobytes()).hexdigest()


def test_default_attention_codec_stays_bit_identical():
    """`byteflow_codec` defaults to today's path, bit for bit (`21a97d2` pin)."""
    torch.manual_seed(1234)
    default = DormouseModel(cfg())
    torch.manual_seed(1234)
    explicit = DormouseModel(cfg(byteflow_codec="attention"))
    x, y = ids_targets()
    assert _logit_sha(default, x, y) == ATTENTION_LOGITS_SHA256
    assert _logit_sha(explicit, x, y) == ATTENTION_LOGITS_SHA256


# ---------------------------------------------------------------------------
# (b) the contract, gradients, no host sync
# ---------------------------------------------------------------------------


def test_conv_shapes_are_the_contract():
    cc = codec()
    x, _ = ids_targets()
    b, t = x.shape
    h, z, sel = cc.encode_chunks(x, 4)
    assert tuple(h.shape) == (b, t, cc.width), "h stays the per-byte stream"
    assert tuple(z.shape) == (b, 4, cc.d_global)
    assert tuple(sel.shape) == (b, 4)
    assert sel.dtype == torch.long
    # `sel` is byte positions of patch starts, ascending, position 0 present.
    assert sel.tolist() == [[0, 4, 8, 12], [0, 4, 8, 12]]
    out = cc.decode_chunks(z, h, sel)
    assert tuple(out.shape) == (b, t, 256)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    # K < P selects a subset with the shared L2/top-k rule; K > P is refused.
    _, z2, sel2 = cc.encode_chunks(x, 2)
    assert tuple(z2.shape) == (b, 2, cc.d_global)
    assert tuple(sel2.shape) == (b, 2)
    assert (sel2[:, 0] == 0).all()
    assert tuple(cc.decode_chunks(z2, h, sel2).shape) == (b, t, 256)
    with pytest.raises(LoudError, match="must be in 1..="):
        cc.encode_chunks(x, 5)
    bad = torch.zeros(2, 15, dtype=torch.long)
    with pytest.raises(LoudError, match="not a multiple"):
        cc.encode_chunks(bad, 2)


def test_conv_every_parameter_gets_a_gradient():
    cc = codec()
    x, _ = ids_targets()
    h, z, sel = cc.encode_chunks(x, 4)
    cc.decode_chunks(z, h, sel).sum().backward()
    missing = [n for n, p in cc.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"
    dead = [n for n, p in cc.named_parameters() if not bool(p.grad.abs().sum())]
    assert not dead, f"zero gradient reached {dead}"
    assert all(bool(torch.isfinite(p.grad).all()) for p in cc.parameters())


def test_conv_forward_does_not_host_sync(monkeypatch):
    """No `.item`/`.tolist`/truthiness on a tensor anywhere in encode+decode."""

    def boom(*_a, **_k):
        raise AssertionError("host sync in the conv codec's forward")

    cc = codec()
    x, _ = ids_targets()
    counters = Counters()
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "item", boom)
        m.setattr(torch.Tensor, "tolist", boom)
        m.setattr(torch.Tensor, "__bool__", boom)
        h, z, sel = cc.encode_chunks(x, 4)
        out = cc.decode_chunks(z, h, sel)
        out.sum().backward()
        # A device counter stays a device tensor until the log cadence: no
        # host read rides the forward (ADR-0018 rule 2).
        counters.count_device("conv_runs", torch.tensor(1, dtype=torch.int64))
        assert "conv_runs" not in counters
    assert counters.drain() == {"conv_runs": 1}


def test_conv_channel_end_to_end_and_option_a():
    """DormouseModel Option B (channel) and Option A both build and train."""
    model = DormouseModel(cfg(byteflow_codec="conv", byteflow_patch_bytes=4, byteflow_k_tokens=4))
    assert model.bf is not None and isinstance(model.bf.net, ConvCodec)
    x, y = ids_targets()
    logits, loss, counters = model(x, y)
    assert tuple(logits.shape) == (x.numel(), 256)
    assert loss is not None and torch.isfinite(loss) and loss.requires_grad
    assert counters.get("bf_channel") == 1, counters
    loss.backward()
    missing = [n for n, p in model.bf.net.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"

    # Option A: no loop arm, the codec alone returns the byte logits.
    solo = DormouseModel(
        cfg(
            byteflow_codec="conv",
            byteflow_patch_bytes=4,
            byteflow_k_tokens=4,
            moe_topk=0,
        )
    )
    assert solo.bf is None and isinstance(solo.bf_net, ConvCodec)
    logits, loss, counters = solo(x, y)
    assert tuple(logits.shape) == (x.numel(), 256)
    assert counters.get("bf_net") == 1, counters


# ---------------------------------------------------------------------------
# (c) refusals
# ---------------------------------------------------------------------------


def test_codec_name_and_patch_arms_are_loud():
    with pytest.raises(LoudError, match="byteflow_codec"):
        validate(cfg(byteflow_codec="attn"))
    with pytest.raises(LoudError, match="byteflow_codec"):
        DormouseModel(cfg(byteflow_codec="attn"))  # the seam a hand build reaches
    # patch_bytes is the conv patchify's stride: inert under attention.
    with pytest.raises(LoudError, match="byteflow_patch_bytes"):
        validate(cfg(byteflow_patch_bytes=4))


def test_conv_requires_use_byteflow():
    off = cfg(byteflow_codec="conv", use_byteflow=False)
    with pytest.raises(LoudError, match="use_byteflow=false"):
        validate(off)
    with pytest.raises(LoudError, match="use_byteflow=false"):
        DormouseModel(off)


def test_conv_with_segment_len_is_loud():
    """A segment cut would put patch boundaries where the codec's `sel` is not."""
    model = DormouseModel(cfg(byteflow_codec="conv", byteflow_patch_bytes=4, byteflow_k_tokens=4))
    x, y = ids_targets()
    with pytest.raises(LoudError, match="segment_len"):
        model(x, y, segment_len=8)


def test_conv_geometry_is_loud():
    with pytest.raises(LoudError, match="not a multiple"):
        validate(cfg(byteflow_codec="conv", byteflow_patch_bytes=5))
    with pytest.raises(LoudError, match="patch count"):
        validate(cfg(byteflow_codec="conv", byteflow_patch_bytes=4, byteflow_k_tokens=9))
    with pytest.raises(LoudError, match="byteflow_logdet"):
        validate(cfg(byteflow_codec="conv", byteflow_patch_bytes=4, byteflow_logdet=True))
    with pytest.raises(LoudError, match="layers"):
        codec(layers=3)


# ---------------------------------------------------------------------------
# (d) the cost ceilings
# ---------------------------------------------------------------------------


def test_conv_stays_under_the_cost_ceilings():
    """The profile shape (`bench_bf_dtype.py`): d_global=1024, 8 bytes/patch.

    States the two ceilings `convcodec.py` documents: <= 2 % of the measured
    loop's 24.1 M MAC/latent per patch, and <= 1 M parameters. The attention
    codec is >400 % and ~94 M at the same shape, so this is the line that keeps
    a future edit from growing the codec back into it.
    """
    for bpp in (4, 8):
        cc = ConvCodec(1024, patch_bytes=bpp, width=64, rank=64)
        assert cc.mac_per_patch() <= CODEC_MAC_CEILING, (
            f"{bpp}B/patch: {cc.mac_per_patch()} MAC > ceiling {CODEC_MAC_CEILING}"
        )
        assert cc.num_params() <= CODEC_PARAM_CEILING, (
            f"{bpp}B/patch: {cc.num_params()} params > ceiling {CODEC_PARAM_CEILING}"
        )
        # The documented estimates, so a change to either constant is visible.
        want = 459_776 if bpp == 8 else 307_712
        assert cc.mac_per_patch() == want
        assert 1.0 <= 100.0 * cc.mac_per_patch() / 24_100_000 <= 2.0
