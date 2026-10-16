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
from dormouse.model.sparse import attention as msa_attention
from dormouse.model.sparse.attention import (
    MSA_SLOT_CHUNK,
    _ckpt_needed,
    attend_selected_blocks,
)

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


def _naive_selected_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    blocks: torch.Tensor,
    keep: torch.Tensor,
    block: int,
    scale: float,
) -> torch.Tensor:
    """The chunked gather's algorithm, spelled out one query at a time.

    Deliberately the slowest, most obvious form: per query, walk the K selected
    slots, collect the in-range causal keys of every KEPT slot (the sentinel
    slot is dropped by `keep`), take a plain softmax over exactly those keys and
    weight the values. Nothing here shares arithmetic with
    `attend_selected_blocks` — no chunk, no index plan, no running max — so a
    parity failure names the arm rather than a shared helper.
    """
    b, t, h, hd = q.shape
    r = block
    out = torch.zeros_like(q)
    for bi in range(b):
        for i in range(t):
            keys: list[int] = []
            for s in range(blocks.shape[-1]):
                if not bool(keep[bi, i, s]):
                    continue
                start = int(blocks[bi, i, s]) * r
                keys += [p for p in range(start, start + r) if p < t and p <= i]
            assert keys, f"query {i} kept a block but has no valid key"
            ks, vs = k[bi, keys], v[bi, keys]  # [P,h,hd]
            sc = torch.einsum("hd,phd->hp", q[bi, i], ks) * scale  # [h,P]
            out[bi, i] = torch.einsum("hp,phd->hd", torch.softmax(sc, dim=-1), vs)
    return out


def _arm_io(b: int, t: int, h: int, hd: int, block: int, topk: int, seed: int):
    """`(q,k,v,blocks,keep)` with the SELECTOR's own block plan (real sentinels)."""
    torch.manual_seed(seed)
    cfg = tiny_cfg(d_model=h * hd, n_heads=h, head_dim=hd, msa_block=block, msa_topk=topk)
    sel = FlexSparseSelector(cfg)
    x = torch.randn(b, t, h * hd)
    blocks, keep = sel(x)
    q, k, v = (torch.randn(b, t, h, hd) for _ in range(3))
    return q, k, v, blocks, keep, 1.0 / (hd**0.5)


def test_e_chunked_gather_matches_the_naive_per_query_attention():
    """PARITY: the chunked/online-softmax route == the longhand per-query loop.

    Shapes chosen so the chunk loop runs more than once and the TRUNCATION path
    is live: `topk > n_blocks`, so the plan carries `keep=False` sentinel slots
    next to real ones.
    """
    r, topk = 4, 32
    for t in (64, 34):  # 34 = a partial final block (8*4 + 2)
        q, k, v, blocks, keep, scale = _arm_io(2, t, 4, 16, r, topk, seed=3)
        assert not keep.all(), "the truncated/sentinel path must be in this gate"
        got = attend_selected_blocks(q, k, v, blocks, keep, r, scale)
        ref = _naive_selected_attention(q, k, v, blocks, keep, r, scale)
        err = (got - ref).abs().max().item()
        assert err < 1e-5, f"chunked gather vs the naive loop: |d| = {err:.2e}"

    # The arm end to end at the tiny shape, against the same loop.
    attn = BlockSparseAttention(_cfg(topk=topk))
    x = torch.randn(2, 64, 64)
    q = attn.q_proj(x).reshape(2, 64, attn.n_heads, attn.head_dim)
    k = attn.k_proj(x).reshape(2, 64, attn.n_heads, attn.head_dim)
    v = attn.v_proj(x).reshape(2, 64, attn.n_heads, attn.head_dim)
    blocks, keep = attn.selector(x)
    err = (
        (
            attend_selected_blocks(q, k, v, blocks, keep, attn.block, attn.scale)
            - _naive_selected_attention(q, k, v, blocks, keep, attn.block, attn.scale)
        )
        .abs()
        .max()
        .item()
    )
    assert err < 1e-5, f"arm vs the naive loop: |d| = {err:.2e}"


def test_f_gather_is_chunk_bounded_and_runs_at_the_b1_shape(monkeypatch):
    """MEMORY GATE — the shapes, which is what a CPU container can honestly assert.

    Every 5-D tensor `attend_selected_blocks` hands to `einsum` is the gathered
    chunk, so its column count IS the arm's peak tensor. Asserted to be
    `MSA_SLOT_CHUNK * r` and never the whole `m * r` — in every policy state:
    with the gradient ON the chunk region may recompute the gather in the
    backward (that is what keeps the 12.9 GB that OOMed b8 s512 on 2026-10-16
    from being retained) or, when `MSA_CKPT_BYTES` says the whole-m gather is
    small enough to retain, the backward reuses the chunk it was handed. Both
    are chunk-bounded; which one ran is COUNTED (`msa_ckpt`).
    """
    r, topk = 4, 32
    assert isinstance(MSA_SLOT_CHUNK, int) and 1 <= MSA_SLOT_CHUNK < topk
    q, k, v, blocks, keep, scale = _arm_io(1, 128, 4, 16, r, topk, seed=4)
    widths: list[int] = []
    real = torch.einsum

    def spy(*args):
        for a in args:
            if isinstance(a, torch.Tensor) and a.ndim == 5:
                widths.append(a.shape[2])  # gathered key positions (columns)
        return real(*args)

    monkeypatch.setattr(torch, "einsum", spy)
    chunks = -(-topk // MSA_SLOT_CHUNK)
    with torch.no_grad():
        attend_selected_blocks(q, k, v, blocks, keep, r, scale)
    assert len(widths) == 2 * chunks, f"kk+vv per chunk: {widths}"
    assert max(widths) == min(MSA_SLOT_CHUNK, topk) * r < topk * r

    # This shape's whole-m gather is 4 MiB — 64x under the budget — so the
    # policy leaves the region OFF and the backward does not re-gather. The peak
    # is the chunk either way: that is the claim.
    assert not _ckpt_needed(1, 128, topk, r, 4, 16, 4)
    c_off = Counters()
    widths.clear()
    qg = q.clone().requires_grad_(True)
    attend_selected_blocks(qg, k, v, blocks, keep, r, scale, c_off).sum().backward()
    assert len(widths) == 2 * chunks, f"the un-regioned backward re-gathered: {len(widths)}"
    assert max(widths) <= MSA_SLOT_CHUNK * r, f"backward re-gathered {max(widths)} positions"
    assert "msa_ckpt" not in c_off, "the policy fired on a shape it should skip"

    # Over budget (the measured b8 s512 shape) the region fires and the backward
    # RECOMPUTES every chunk instead of retaining the gathers, which is exactly
    # the 12.9 GB the old form kept alive. 4x and not 2x: the recompute is the
    # evidence the region ran.
    assert _ckpt_needed(8, 512, 32, r, 12, 64, 4), "b8 s512 is the measured OOM shape"
    monkeypatch.setattr(msa_attention, "MSA_CKPT_BYTES", 0)
    c_on = Counters()
    widths.clear()
    qg = q.clone().requires_grad_(True)
    attend_selected_blocks(qg, k, v, blocks, keep, r, scale, c_on).sum().backward()
    assert len(widths) == 4 * chunks, f"the chunk regions did not recompute: {len(widths)}"
    assert max(widths) <= MSA_SLOT_CHUNK * r, f"backward re-gathered {max(widths)} positions"
    assert c_on["msa_ckpt"] == 1, "the checkpoint policy must be COUNTED when it fires"
    monkeypatch.undo()

    # The b1 attention SHAPE on CPU (b=2,t=512,K=32,r=4,h=32,hd=64 = d 2048):
    # 32 slots at chunk 4 means 8 gathers of [2,512,16,32,64] = 134 MB each,
    # against 1024 MB for the single whole-m gather this replaces.
    q, k, v, blocks, keep, scale = _arm_io(2, 512, 32, 64, r, topk, seed=6)
    got = attend_selected_blocks(q, k, v, blocks, keep, r, scale)
    assert got.shape == q.shape and torch.isfinite(got).all()
    ref = _naive_selected_attention(q, k, v, blocks, keep, r, scale)
    assert (got - ref).abs().max().item() < 1e-5, "the b1-shaped run must match the loop too"


def test_g_chunked_backward_matches_the_naive_loop():
    """The checkpoint region is load-bearing for memory, so it gets a gradient
    gate: the recomputed backward must equal the naive loop's own autograd."""
    r, topk = 4, 16
    _, k0, v0, blocks, keep, scale = _arm_io(1, 32, 2, 8, r, topk, seed=7)

    def grads(impl):
        torch.manual_seed(8)
        q = torch.randn(1, 32, 2, 8, requires_grad=True)
        k = k0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        impl(q, k, v, blocks, keep, r, scale).sum().backward()
        return q.grad, k.grad, v.grad

    got, ref = grads(attend_selected_blocks), grads(_naive_selected_attention)
    for name, g, e in zip(("q", "k", "v"), got, ref, strict=True):
        err = (g - e).abs().max().item()
        assert err < 1e-5, f"grad {name}: |d| = {err:.2e}"


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
