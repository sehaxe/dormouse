"""The factorized token embedding (`embed_rank`): price, wiring, refusals.

A 32k BPE vocabulary at `d=832` is a 27.3M-parameter table that ONE first matmul
reads. `E = U @ Vᵀ` (`U [vocab, r]`, `V [d, r]`) is 8.6M at `r=256`. These gates
pin the arithmetic, the init (same variance, so an A/B is not an init change),
the full-rank identity, both factors' gradients, and that `0` is the old model.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_run

from dormouse.instrument import LoudError
from dormouse.model import (
    DormouseModel,
    FactorizedEmbedding,
    param_count,
    refuse_vocab_mismatch,
)
from dormouse.model.model import embedding_rows, tokenizer_vocab
from dormouse.train import train

#: The migration's shape: a 32k BPE vocabulary at `d=832`, rank 256.
V, D, R = 32768, 832, 256


def _migration_cfg(**over):
    return tiny_cfg(d_model=D, n_heads=13, head_dim=64, d_ffn=2 * D, vocab=V, **over)


def _ids(vocab: int = SEQ) -> torch.Tensor:
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % vocab
    return ids, (ids + 1) % vocab


# --- (1) the price -------------------------------------------------------


def test_parameter_count_is_pinned_against_the_config():
    """`32768×256 + 256×832` = 8 601 600, and the plain table is 27 262 976."""
    want = V * R + R * D
    print(f"factorized: {V}×{R} + {R}×{D} = {want}; plain: {V * D}")
    assert want == 8_601_600
    assert V * D == 27_262_976

    fact = DormouseModel(_migration_cfg(embed_rank=R), device="meta")
    plain = DormouseModel(_migration_cfg(embed_rank=0), device="meta")
    fact_embed = sum(p.numel() for n, p in fact.named_parameters() if n.startswith("embed."))
    plain_embed = sum(p.numel() for n, p in plain.named_parameters() if n.startswith("embed."))
    assert fact_embed == want
    assert plain_embed == V * D
    assert param_count(plain) - param_count(fact) == V * D - want
    assert sorted(n for n, _ in fact.named_parameters() if n.startswith("embed.")) == [
        "embed.u.weight",
        "embed.v.weight",
    ]


def test_init_matches_the_plain_tables_variance():
    """`U ~ N(0,1)`, `V ~ N(0,1/r)` → a row's variance is `r·1·(1/r) = 1`."""
    torch.manual_seed(0)
    plain = DormouseModel(tiny_cfg())
    torch.manual_seed(0)
    fact = DormouseModel(tiny_cfg(embed_rank=16))
    plain_std = float(plain.embed.weight.detach().std())
    fact_std = float(fact.embed.weight.detach().std())  # U @ V, materialised
    assert abs(plain_std - 1.0) < 0.02, plain_std
    assert abs(fact_std - 1.0) < 0.05, fact_std


# --- (2) wiring: full rank is the plain table, and the head is factored ---


def test_full_rank_equals_the_plain_embedding_on_the_logits():
    """At `r = d_model`, `U @ Iᵀ` IS the table — `torch.equal`, not allclose.

    The identity is the one full-rank factorization whose product is exact in
    fp32, so equality here means the lookup and the linear are wired in the
    order the arithmetic says, not merely shaped right (the rank-16 gate below
    proves `V` is read at all).
    """
    torch.manual_seed(0)
    plain = DormouseModel(tiny_cfg())
    fact = DormouseModel(tiny_cfg(embed_rank=plain.cfg.d_model))
    assert isinstance(fact.embed, FactorizedEmbedding)
    with torch.no_grad():
        fact.embed.u.weight.copy_(plain.embed.weight)
        fact.embed.v.weight.copy_(torch.eye(plain.cfg.d_model))
    shared = {k: v for k, v in plain.state_dict().items() if not k.startswith("embed.")}
    fact.load_state_dict({**fact.state_dict(), **shared})

    ids, _ = _ids()
    assert torch.equal(plain.embed(ids), fact.embed(ids))
    a, _, _ = plain(ids)
    b, _, _ = fact(ids)
    assert torch.equal(a, b)


def test_a_low_rank_factorization_reads_both_factors():
    """`r < d_model`: the write is `V`'s, not the table's (`forward != u(ids)`)."""
    torch.manual_seed(0)
    m = DormouseModel(tiny_cfg(embed_rank=16))
    ids, _ = _ids()
    with torch.no_grad():
        assert not torch.equal(m.embed(ids), m.embed.u(ids))
        # and the tied head is the transpose of the SAME product
        x = torch.randn(3, m.cfg.d_model)
        assert torch.allclose(m.embed.head(x), x @ m.embed.weight.T, atol=1e-5)


def test_tied_head_reads_the_factorization_and_owns_no_parameter():
    torch.manual_seed(0)
    m = DormouseModel(tiny_cfg(embed_rank=16, tie_embeddings=True))
    assert sum(p.numel() for p in m.lm_head.parameters()) == 0
    x = torch.randn(4, m.cfg.d_model)
    assert torch.allclose(m.lm_head(x), x @ m.embed.weight.T, atol=1e-5)
    ids, targets = _ids(16)
    _, loss, _ = m(ids, targets=targets)
    loss.backward()
    grads = dict(m.named_parameters())
    for name in ("embed.u.weight", "embed.v.weight"):
        g = grads[name].grad
        assert g is not None and float(g.abs().sum()) > 0.0, name


def test_both_factors_get_a_nonzero_gradient_on_step_one():
    torch.manual_seed(0)
    m = DormouseModel(tiny_cfg(embed_rank=16))
    ids, targets = _ids(16)
    _, loss, _ = m(ids, targets=targets)
    loss.backward()
    grads = dict(m.named_parameters())
    for name in ("embed.u.weight", "embed.v.weight"):
        g = grads[name].grad
        assert g is not None and float(g.abs().sum()) > 0.0, name


# --- (3) `embed_rank = 0` is today's model --------------------------------


def test_embed_rank_zero_is_bit_identical():
    """Same seed, same weights, same draws: `embed.weight` and the logits."""
    torch.manual_seed(7)
    explicit = DormouseModel(tiny_cfg(embed_rank=0))
    torch.manual_seed(7)
    default = DormouseModel(tiny_cfg())
    assert default.cfg.embed_rank == 0
    assert set(explicit.state_dict()) == set(default.state_dict())
    assert explicit.embed.weight.shape == (default.cfg.vocab, default.cfg.d_model)
    ids, _ = _ids()
    assert torch.equal(explicit(ids)[0], default(ids)[0])
    assert torch.equal(explicit.embed.weight, default.embed.weight)


# --- (4) the arm actually trains on CPU -----------------------------------


def test_three_step_cpu_train_with_embed_rank_256_finishes_finite(tmp_path):
    cfg = tiny_cfg(d_model=512, n_heads=8, head_dim=64, d_ffn=1024, vocab=1024, embed_rank=256)
    run = dataclasses.replace(tiny_run(tmp_path, steps=3, ckpt_every=100), model=cfg)
    report = train(DormouseModel(cfg), run, log=lambda *_: None)
    assert math.isfinite(report["loss"]), report


# --- (5) the refusals -----------------------------------------------------


def test_embed_rank_above_d_model_is_refused():
    with pytest.raises(LoudError) as e:
        tiny_cfg(embed_rank=65)
    assert "embed_rank=65" in str(e.value)
    with pytest.raises(LoudError) as e:
        tiny_cfg(embed_rank=-1)
    assert "embed_rank=-1" in str(e.value)


def test_embed_rank_with_byteflow_is_refused():
    """The byteflow shapes build their own table and never `self.embed`."""
    with pytest.raises(LoudError) as e:
        tiny_cfg(embed_rank=8, use_byteflow=True)
    assert "embed_rank" in str(e.value) and "use_byteflow" in str(e.value)


def test_checkpoint_vocabulary_mismatch_is_loud_never_padded():
    m = DormouseModel(tiny_cfg(embed_rank=8))
    sd = dict(m.state_dict())
    sd["embed.u.weight"] = torch.zeros(100, 8)
    with pytest.raises(LoudError) as e:
        m.load_state_dict(sd)
    assert "100 rows" in str(e.value) and "vocab=256" in str(e.value)

    plain = DormouseModel(tiny_cfg())
    sd = dict(plain.state_dict())
    sd["embed.weight"] = torch.zeros(100, plain.cfg.d_model)
    with pytest.raises(LoudError) as e:
        plain.load_state_dict(sd)
    assert "100 rows" in str(e.value)
    # a matching table loads
    plain.load_state_dict(dict(plain.state_dict()))


def test_tokenizer_model_mismatch_is_refused(tmp_path):
    tok = tmp_path / "tok.json"
    tok.write_text(
        json.dumps(
            {
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}},
                "added_tokens": [{"id": 2, "content": "<s>"}],
            }
        )
    )
    assert tokenizer_vocab(tok) == 3
    with pytest.raises(LoudError) as e:
        refuse_vocab_mismatch(vocab=256, tokenizer=tok)
    assert "3 ids" in str(e.value) and "vocab=256" in str(e.value)
    refuse_vocab_mismatch(vocab=3, tokenizer=tok)  # matching is silent
    assert embedding_rows({"embed.u.weight": torch.zeros(3, 8)}) == 3
    assert embedding_rows({"loop.q.weight": torch.zeros(3, 8)}) is None
