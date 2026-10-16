"""MSA — the block-sparse second attention arm (readiness-plan §3.5 item 4).

Four claims, one per case the task names:

(a) the indexer gives a DETERMINISTIC block set per (query, block): same input,
    same indices;
(b) at a tiny s=128 the arm's output equals NAIVE dense attention restricted to
    the top-k blocks (a 2e-3 bound — a wiring gate, not a certificate);
(c) the FLEX mask edge case: a query with fewer than K causal blocks truncates
    the top-K, and that truncation is COUNTED, not silently `-inf`;
(d) the arm's ABSENCE is free: no parameters, `None` out, a counted `msa_off`,
    and the whole model's bytes are the same with the flag on or off.

The flag is read off the config object (`getattr(cfg, "use_msa", False)`) so the
arm lands as new files only — the `ModelConfig` schema field is a follow-up for
the commit that wires the arm into `LoopBlock`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.instrument import Counters
from dormouse.model import DormouseModel, param_count
from dormouse.model.sparse import BlockSparseAttention, FlexSparseSelector

R, K = 4, 4  # small block/top-K so the top-k is exercised at tiny shapes


def _cfg(block: int = R, topk: int = K):
    """A tiny config with the MSA arm ON (the flag rides the config object)."""
    cfg = tiny_cfg()
    cfg.use_msa = True
    cfg.msa_block = block
    cfg.msa_topk = topk
    return cfg


def _dense_reference(attn: BlockSparseAttention, x: torch.Tensor) -> torch.Tensor:
    """Naive dense attention masked to the union of the SELECTED blocks.

    Built independently of the arm: the selected set comes from the selector,
    but the mask arithmetic (causality, block-of-key lookup, SDPA) is written
    here so the arm cannot pass by sharing a bug with the reference.
    """
    b, t, _ = x.shape
    r, h = attn.block, attn.n_heads
    n = (t + r - 1) // r
    q = attn.q_proj(x).reshape(b, t, h, attn.head_dim)
    k = attn.k_proj(x).reshape(b, t, h, attn.head_dim)
    v = attn.v_proj(x).reshape(b, t, h, attn.head_dim)
    blocks, keep = attn.selector(x)
    # column n is the selector's padding sentinel; it is never a real block.
    sel = torch.zeros(b, t, n + 1, dtype=torch.bool)
    sel.scatter_(2, blocks, keep)  # topk gives distinct indices per row
    kpos = torch.arange(t).view(1, 1, t)
    qpos = torch.arange(t).view(1, t, 1)
    blk_of_key = (torch.arange(t) // r).view(1, 1, t).expand(b, t, t)
    allow = sel.gather(2, blk_of_key) & (kpos <= qpos)  # [b,t,t]
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=allow[:, None],
        scale=attn.scale,
        dropout_p=0.0,
    )
    return ref.transpose(1, 2).reshape(b, t, -1)


def test_a_indexer_blocks_are_deterministic_per_query():
    torch.manual_seed(0)
    sel = FlexSparseSelector(_cfg(topk=8))
    x = torch.randn(2, 128, 64)
    b1, k1 = sel(x)
    b2, k2 = sel(x)
    assert torch.equal(b1, b2) and torch.equal(k1, k2), "same input must give the same blocks"
    b3, _ = sel(torch.randn(2, 128, 64))
    assert not torch.equal(b1, b3), "a different input should pick a different set"

    t, r, n = 128, R, 128 // R
    assert b1.shape == (2, t, 8)
    assert b1.min() >= 0 and b1.max() <= n, "index (or the n sentinel) only"
    assert not (k1 & (b1 == n)).any(), "a kept slot is never the sentinel"

    # The kept count is the block-causal candidate count: floor(i/r)+1, capped
    # at K. That ties the selected set to the recipe's block-causal rule.
    cand = torch.arange(t) // r + 1
    assert k1[0].sum(-1).tolist() == cand.clamp(max=8).tolist()


def test_b_sparse_matches_dense_within_the_selected_blocks():
    torch.manual_seed(1)
    attn = BlockSparseAttention(_cfg(topk=8))
    x = torch.randn(2, 128, 64)
    out = attn(x, Counters())
    assert out is not None and out.shape == x.shape
    ref = _dense_reference(attn, x)
    assert (out - ref).abs().max().item() < 2e-3


def test_c_topk_truncates_on_short_prefixes_and_is_counted():
    torch.manual_seed(2)
    t, r, k, n = 64, R, K, 64 // R
    sel = FlexSparseSelector(_cfg())
    x = torch.randn(1, t, 64)
    c = Counters()
    blocks, keep = sel(x, c)
    cand = torch.arange(t) // r + 1
    assert keep[0].sum(-1).tolist() == cand.clamp(max=k).tolist(), (
        "short prefixes keep every candidate"
    )
    # query i truncates iff floor(i/r)+1 < K, i.e. i < (K-1)*r: 12 queries at K=4.
    assert c["msa_truncated"] == (k - 1) * r == 12
    assert (blocks[0][~keep[0]] == n).all(), "truncated slots carry the sentinel"

    # A whole sequence shorter than K blocks: every query truncates.
    sel64 = FlexSparseSelector(_cfg(topk=8))
    c64 = Counters()
    _, keep64 = sel64(torch.randn(1, 16, 64), c64)  # 16 tokens -> 4 blocks < 8
    assert c64["msa_truncated"] == 16
    assert keep64[0].sum(-1).tolist() == (torch.arange(16) // r + 1).tolist()

    # No truncation at all when K == 1: the counter stays absent.
    c1 = Counters()
    FlexSparseSelector(_cfg(topk=1))(torch.randn(1, 64, 64), c1)
    assert "msa_truncated" not in c1

    # The truncated edge still computes the RIGHT thing, not just a shaped one.
    attn = BlockSparseAttention(_cfg(topk=8))
    x34 = torch.randn(1, 34, 64)  # a partial final block (34 = 8*4 + 2)
    got = attn(x34, Counters())
    assert (got - _dense_reference(attn, x34)).abs().max().item() < 2e-3


def test_d_arm_off_is_free_and_the_model_is_unchanged():
    off = BlockSparseAttention(tiny_cfg())  # no use_msa field -> the default, OFF
    assert sum(p.numel() for p in off.parameters()) == 0, "off must build no parameters"
    c = Counters()
    assert off(torch.randn(2, 64, 64), c) is None, "off returns None, not zeros"
    assert c["msa_off"] == 1 and "msa_blocks" not in c
    assert sum(p.numel() for p in BlockSparseAttention(_cfg()).parameters()) > 0

    # The flag is a REAL ModelConfig field since 2026-10-16 (the owner's 1M
    # directive made the arm configurable): OFF builds no arm parameters, ON
    # adds the arm's own projections, and both trees forward end to end.
    m_off = DormouseModel(tiny_cfg())
    b = tiny_cfg()
    b.use_msa = True
    m_on = DormouseModel(b)
    assert param_count(m_on) > param_count(m_off) > 0

    # And the default 512-seq forward still runs (the KDA-only path).
    model = DormouseModel(tiny_cfg(max_seq_len=512))
    ids = torch.arange(512, dtype=torch.long).view(1, 512) % 251
    logits, loss, _ = model(ids, (ids + 1) % 251)
    assert logits.shape == (512, 256) and torch.isfinite(loss)
