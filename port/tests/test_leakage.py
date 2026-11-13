"""Leakage gates for the CURRENT training path.

The defect class this file exists for: a byte-level LM that can see the byte it
must predict gets a loss floor it did not earn, and every A/B after it compares
two wrong numbers. Each test below fails on a real leak and names the mechanism
that would carry it.

Arms covered (CPU; the KDA arm runs its COUNTED causal-SDPA fallback because
FLA is CUDA-only — `kda=` on the log line says which one ran on a real run):

* `block_form="transformer"` (the vanilla control, one GELU FFN);
* `block_form="kda"` with the dense FFN (`n_experts=1`) and the 4-expert bank
  (`moe_topk=2`), which is the arm `configs/arch_moe8.toml` names.

The gates:

1. `_windows` is the next-byte shift, and the model's logits/CE change ONLY at
   the position whose target moved (targets never enter the forward);
2. a change at byte `p` leaves every logit at `< p` alone — zeroed or garbage
   tail, so a windowed statistic that peeks is red;
3. the attention mask is lower-triangular (`is_causal` pin + an explicit-mask
   recompute), and padding cannot reach a real position or the loss;
4. `doc_masking` never scores a cross-document target and a document cannot see
   the previous one (the real `code.bin.docs.json` sidecar included);
5. the eval window and the train stream are disjoint by BYTE RANGE (inode), and
   `train()` refuses a config where they overlap;
6. the sampler is a pure function of `(seed, step)`; the flat/shard readers are
   periodic per epoch, which is DOCUMENTED behaviour and pinned as such.

Mutation evidence for every gate is in `docs/reviews/2026-11-07-leakage.md`:
reverse the shift, unmask the triangle, drop the boundary mask, remove the eval
carve, and make the sampler draw from a global RNG — each turns its gate RED.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from conftest import BATCH, SEQ, tiny_cfg, tiny_run

from dormouse.config import ModelConfig, TrainConfig
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.data import (
    ByteStream,
    DocPackedStream,
    MaskedByteStream,
    ShardStream,
    _windows,
    doc_sidecar_path,
    eval_train_overlap,
    load_doc_starts,
    open_streams,
    shard_files,
    shard_order,
)
from dormouse.train.mixture import (
    MixPlan,
    MixStage,
    MixtureStream,
    dumps_mix_plan,
    loads_mix_plan,
    source_for_step,
)

PORT = Path(__file__).resolve().parents[1]
#: The corpus lives outside git (`data_local/`), in whichever worktree mounted it;
#: a worktree without the data skips the real-sidecar/real-plan gates.
DATA = next(
    (
        p
        for p in (
            Path(os.environ["DORMOUSE_DATA"]) if os.environ.get("DORMOUSE_DATA") else None,
            PORT.parent / "data_local",
            Path("/home/sehaxe/dormouse/data_local"),
        )
        if p is not None and p.is_dir()
    ),
    PORT.parent / "data_local",
)
CODE = DATA / "mix4" / "code.bin"
CODE_SIDECAR = doc_sidecar_path(CODE)


# --- fixtures ---------------------------------------------------------------


def _transformer_cfg(**over) -> ModelConfig:
    base = dict(
        block_form="transformer",
        max_iter=1,
        loop_blocks=1,
        n_layers=2,
        use_kda=False,
        use_engram=False,
        use_tsct=False,
        n_experts=1,
        moe_topk=0,
        bf16=False,
    )
    base.update(over)
    return tiny_cfg(**base)


def _kda_cfg(*, moe: bool, **over) -> ModelConfig:
    base = dict(
        block_form="kda",
        max_iter=2,
        use_engram=True,
        n_experts=4 if moe else 1,
        moe_topk=2 if moe else 0,
        moe_capacity=1.0 if moe else 0.0,
    )
    base.update(over)
    return tiny_cfg(**base)


#: `(form, ffn)` arms. Dense is the shipped KDA config's one-expert bank; MoE is
#: the 8-expert arm's shape at the tiny fixture's size.
ARMS = [
    ("transformer", lambda: DormouseModel(_transformer_cfg())),
    ("kda-dense", lambda: DormouseModel(_kda_cfg(moe=False))),
    ("kda-moe", lambda: DormouseModel(_kda_cfg(moe=True))),
]


def _build(kind: str) -> DormouseModel:
    torch.manual_seed(1234)
    for name, make in ARMS:
        if name == kind:
            model = make()
            model.eval()
            return model
    raise AssertionError(kind)


def _batch(n: int = BATCH, t: int = SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    ids = (torch.arange(n * t, dtype=torch.long).view(n, t) * 7 + 3) % 251
    return ids, (ids + 5) % 251


def _per_position_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    flat = logits.reshape(-1, logits.shape[-1])
    return -F.log_softmax(flat, dim=-1).gather(1, targets.reshape(-1, 1)).reshape(targets.shape)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max())


def _docs_file(tmp_path: Path, docs: list[bytes], name: str = "corpus.bin") -> Path:
    path = tmp_path / name
    path.write_bytes(b"".join(docs))
    starts, at = [], 0
    for d in docs:
        starts.append(at)
        at += len(d)
    Path(str(path) + ".docs.json").write_text(
        json.dumps({"version": 1, "size": at, "starts": starts})
    )
    return path


# --- 1 + 2: shift, targets never enter the forward, causality ---------------


def test_windows_is_the_next_byte_shift():
    """The ONE shift definition: `x = raw[:-1]`, `y = raw[1:]` — per row."""
    raw = bytes((i * 11) % 251 for i in range(BATCH * (SEQ + 1)))
    x, y = _windows(raw, BATCH, SEQ)
    rows = [raw[r * (SEQ + 1) : (r + 1) * (SEQ + 1)] for r in range(BATCH)]
    assert torch.equal(x, torch.tensor([list(r[:-1]) for r in rows]))
    assert torch.equal(y, torch.tensor([list(r[1:]) for r in rows]))
    # The invariant a REVERSED shift breaks: every target is the next input byte,
    # never the byte at the same position.
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert not torch.equal(y[:, :-1], x[:, :-1])


@pytest.mark.parametrize("kind", [a[0] for a in ARMS])
def test_a_target_byte_never_enters_the_forward(kind):
    """Change ONE target far from the end: logits are BIT-IDENTICAL everywhere.

    This is the shift gate. A model that consumes `targets` (the classic
    teacher-forcing leak) or whose CE is computed one position off moves here.
    The loss may move, and only through position `p`'s own prediction.
    """
    model = _build(kind)
    x, y = _batch()
    p = SEQ // 2
    y2 = y.clone()
    y2[0, p] = (y2[0, p] + 1) % 251
    with torch.no_grad():
        logits, loss, _ = model(x, y)
        logits2, loss2, _ = model(x, y2)
    assert torch.equal(logits, logits2), "a target byte reached the forward"
    assert loss is not None and loss2 is not None

    ce = _per_position_ce(logits, y)
    ce2 = _per_position_ce(logits, y2)
    # Per-position CE differs at exactly `(0, p)` (and the batch mean follows).
    diff = ce2 - ce
    assert _max_abs(diff[:, :p], torch.zeros_like(diff[:, :p])) == 0.0
    assert _max_abs(diff[:, p + 1 :], torch.zeros_like(diff[:, p + 1 :])) == 0.0
    assert float(diff[0, p]) != 0.0
    if kind == "transformer":
        # The control's headline IS the final CE mean, so its delta is exact.
        n = y.numel()
        assert math.isclose(
            float((loss2 - loss).detach()),
            float(diff.sum().detach()) / n,
            rel_tol=0,
            abs_tol=1e-6,
        ), "the mean loss moved by more than the one target's CE"
    else:
        # The loop's headline is (Σ per-iteration CE + final CE)/(T+1); every
        # term is a mean over the SAME positions, so only p moves in each.
        assert float(loss2) != float(loss)


@pytest.mark.parametrize("kind", [a[0] for a in ARMS])
def test_logits_at_p_depend_only_on_bytes_up_to_p(kind):
    """Zero the tail, then replace it with garbage — position `p` cannot move.

    A leak here is any windowed statistic (a mean, a max, the MoE capacity rank,
    a non-causal mask) that reads `x[p+1:]`. The MoE bank's routed gather has a
    shape that varies with the tail, so its float reduction order moves in the
    last bits; the bound is 1e-5, against O(0.1) for an unmasked triangle.
    """
    model = _build(kind)
    x, y = _batch(1, SEQ)
    p = SEQ - 8
    variants = {
        "zeros": torch.cat([x[:, : p + 1], torch.zeros_like(x[:, p + 1 :])], dim=1),
        "garbage": torch.cat(
            [x[:, : p + 1], (torch.arange(SEQ - p - 1) * 37 + 11).view(1, -1) % 251], dim=1
        ),
    }
    with torch.no_grad():
        base, _, _ = model(x, y)
    atol = 1e-5 if kind.endswith("moe") else 0.0
    for name, xv in variants.items():
        with torch.no_grad():
            other, _, _ = model(xv, y)
        a = base.reshape(1, SEQ, -1)[:, p]
        b = other.reshape(1, SEQ, -1)[:, p]
        if atol:
            assert _max_abs(a, b) <= atol, f"{kind}/{name}: logits at p moved"
        else:
            assert torch.equal(a, b), f"{kind}/{name}: logits at p moved"


@pytest.mark.parametrize("kind", [a[0] for a in ARMS])
def test_changing_a_consumed_byte_leaves_every_earlier_logit_alone(kind):
    """`x[p]` moves: every logit at `< p` is unchanged, at every p.

    Built through `_windows` on purpose: a reversed shift makes `raw[p]` land in
    `x[p-1]`, so the logit at `p-1` moves and this fails.
    """
    model = _build(kind)
    raw = bytes((i * 13 + 7) % 251 for i in range(SEQ + 1))
    x, y = _windows(raw, 1, SEQ)
    p = SEQ - 9
    raw2 = raw[:p] + bytes([(raw[p] + 41) % 251]) + raw[p + 1 :]
    x2, _y2 = _windows(raw2, 1, SEQ)
    assert not torch.equal(x, x2), "the fixture did not move an input byte"
    with torch.no_grad():
        base, _, _ = model(x, y)
        other, _, _ = model(x2, y)
    a = base.reshape(1, SEQ, -1)[:, :p]
    b = other.reshape(1, SEQ, -1)[:, :p]
    atol = 1e-5 if kind.endswith("moe") else 0.0
    if atol:
        assert _max_abs(a, b) <= atol
    else:
        assert torch.equal(a, b), f"{kind}: byte p={p} reached a logit before it"


# --- 3: the attention mask, padding, and a numeric lower-triangular check ----


@contextlib.contextmanager
def _capture_sdpa():
    """Record every `scaled_dot_product_attention` call and its output."""
    calls: list[dict] = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        out = real(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        calls.append(
            dict(
                q=q.detach(),
                k=k.detach(),
                v=v.detach(),
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=scale,
                out=out.detach(),
            )
        )
        return out

    with mock.patch.object(F, "scaled_dot_product_attention", spy):
        yield calls


def _manual_sdpa(call: dict) -> torch.Tensor:
    """The same attention with an EXPLICIT lower-triangular mask, fp32."""
    q, k, v = call["q"].float(), call["k"].float(), call["v"].float()
    scale = call["scale"] if call["scale"] is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale
    if call["attn_mask"] is not None:
        keep = call["attn_mask"]
    elif call["is_causal"]:
        tq, tk = q.shape[-2], k.shape[-2]
        keep = torch.ones(tq, tk, dtype=torch.bool).tril(diagonal=tk - tq)
    else:
        keep = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool)
    scores = scores.masked_fill(~keep, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _padded_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows 0/1 are documents of 20 and 64 bytes, tail-padded; mask drops padding."""
    b, t = 2, SEQ
    x = torch.zeros(b, t, dtype=torch.long)
    y = torch.zeros(b, t, dtype=torch.long)
    mask = torch.zeros(b, t, dtype=torch.bool)
    for r, n in enumerate((20, t)):
        x[r, :n] = (torch.arange(n) * 5 + r * 3) % 251
        y[r, : n - 1] = x[r, 1:n]
        mask[r, : n - 1] = True
    return x, y, mask


@pytest.mark.parametrize("kind", ["transformer", "kda-dense"])
def test_the_attention_mask_is_lower_triangular(kind):
    """The model's own attention call is causal, and recomputing it with an
    explicit `tril` mask reproduces it — on a padded batch, padding included."""
    model = _build(kind)
    x, y, mask = _padded_batch()
    with _capture_sdpa() as calls:
        with torch.no_grad():
            model(x, y, loss_mask=mask)
    assert calls, "no SDPA call reached the capture"
    strided = [c for c in calls if c["is_causal"] and c["attn_mask"] is None]
    assert strided, "no call used `is_causal=True`: the causal pin is gone"
    for c in strided:
        assert c["q"].shape[-2] == c["k"].shape[-2] == SEQ, "padding is not in the mask"
        assert _max_abs(c["out"], _manual_sdpa(c)) <= 1e-5
    masked = [c for c in calls if c["attn_mask"] is not None]
    for c in masked:
        assert torch.equal(
            c["attn_mask"],
            torch.ones(c["attn_mask"].shape, dtype=torch.bool).tril(
                diagonal=c["attn_mask"].shape[1] - c["attn_mask"].shape[0]
            ),
        ), "the explicit mask is not lower-triangular"
        assert _max_abs(c["out"], _manual_sdpa(c)) <= 1e-5


def test_the_kda_segmented_mask_is_lower_triangular():
    """The `cols <= rows` arm (KDA's carried state) keeps the diagonal at `tp`."""
    model = _build("kda-dense")
    x, y = _batch(1, SEQ)
    with _capture_sdpa() as calls:
        with torch.no_grad():
            model(x, y, segment_len=SEQ // 4)
    explicit = [c for c in calls if c["attn_mask"] is not None]
    assert explicit, "the segmented path did not take its explicit-mask branch"
    for c in explicit:
        tq, tk = c["attn_mask"].shape
        want = torch.ones(tq, tk, dtype=torch.bool).tril(diagonal=tk - tq)
        assert torch.equal(c["attn_mask"], want)
        assert _max_abs(c["out"], _manual_sdpa(c)) <= 1e-5


@pytest.mark.parametrize("kind", ["transformer", "kda-dense"])
def test_padding_cannot_reach_a_real_position_or_the_loss(kind):
    """The padded region gets the answer; masked positions and real logits do not move.

    `oracle` writes the last scored position's target into the first padding
    slot. If the mask leaked, the model could read it at a later (padding) query
    or the loss could score it; neither may happen.
    """
    model = _build(kind)
    x, y, mask = _padded_batch()
    last = 20 - 2  # last SCORED position of the short row
    # The padded region gets the answer; masked positions and real logits do not move.
    x_oracle = x.clone()
    x_oracle[0, 20:] = torch.tensor([int(y[0, last])] + [7] * (SEQ - 21))
    y_pad = y.clone()
    y_pad[0, 20:] = 255  # padding targets, all masked
    with torch.no_grad():
        logits, loss, _ = model(x, y, loss_mask=mask)
        logits_o, loss_o, _ = model(x_oracle, y_pad, loss_mask=mask)
    # A leak moves a logit by O(1); masked softmax weights are exactly 0 in
    # exact arithmetic but not bit-identical across BLAS/kernel choices, so the
    # bar is a tolerance and not `torch.equal` (it flipped on the CI runner).
    assert torch.allclose(
        logits.reshape(2, SEQ, -1)[:, :20], logits_o.reshape(2, SEQ, -1)[:, :20], rtol=0, atol=1e-5
    ), "padding changed a real position's logit"
    assert torch.allclose(loss, loss_o, rtol=0, atol=1e-7), "padding changed the masked loss"
    assert not bool(mask[0, 20:].any()) and int(mask[0].sum()) == 19, (
        "the padded region must be masked out and the short row must score 19 positions"
    )

    # The reduction is the mean over SELECTED positions, exactly. Only the
    # transformer's loss is the final CE alone; the loop also sums per-iteration
    # CEs (deep supervision), so its headline is a mean of means there.
    if kind == "transformer":
        ce = _per_position_ce(logits, y)
        want = ce[mask].mean()
        # RELATIVE: a wrong reduction (mean-of-means vs mean-over-selected) is
        # off by ~1e-2; an absolute 1e-7 bar is summation-order noise and flipped
        # on the runner.
        assert torch.allclose(loss, want, rtol=1e-6, atol=1e-9), (
            f"the loss is not the mean over selected positions: "
            f"{float(loss):.6f} vs {float(want):.6f}"
        )


def test_masked_ce_mean_divides_by_selected_positions_only():
    from dormouse.model.loopblock import masked_ce_mean

    ce = torch.arange(12, dtype=torch.float32).reshape(-1, 1)
    mask = torch.tensor(
        [[True, True, False, False, False, False], [True, True, True, True, False, False]]
    )
    assert torch.allclose(masked_ce_mean(ce, mask), ce[mask.reshape(-1)].mean(), rtol=0, atol=0)


# --- 4: doc masking ---------------------------------------------------------


def _packed_rows(stream: DocPackedStream, n: int):
    """`n` rows with the (a, b) document range and the cursor that produced each."""
    out = []
    for _ in range(n):
        pos = stream.position
        a, b = stream._doc_at(pos)
        x, y, mask, scored = stream._row()
        out.append((pos, a, b, x, y, mask, scored))
    return out


@pytest.mark.skipif(not CODE.is_file() or not CODE_SIDECAR.is_file(), reason="local mix4 absent")
def test_real_sidecar_no_target_crosses_a_document_boundary():
    """The REAL `code.bin` sidecar: every scored target is inside its document.

    Samples 120 consecutive documents from the middle of the corpus, which
    includes a document longer than `seq_len` (the chunk branch) and short ones
    (the whole-document, padded branch).
    """
    starts = load_doc_starts(CODE)
    i = next(k for k in range(1000, len(starts) - 130) if starts[k + 1] - starts[k] > SEQ + 8)
    lo, hi = starts[i], starts[i + 130]
    src = ByteStream(CODE, 2, SEQ, lo=lo, hi=hi, check_min=False)
    packed = DocPackedStream(src, starts)
    rows = _packed_rows(packed, 40)
    starts_set = set(starts)
    saw_long = saw_short = False
    for pos, a, b, _x, y, mask, scored in rows:
        assert lo <= a <= pos < b <= hi
        remaining = b - pos
        saw_long |= remaining > SEQ
        saw_short |= remaining < SEQ
        for j in range(SEQ):
            if not bool(mask[j]):
                continue
            target = pos + j + 1
            assert target < b, "a target crossed the document end"
            assert target not in starts_set, "a scored target is a document start"
            assert int(y[j]) == int(src.read_at(target, 1)[0])
        # The last byte of a whole-document row has no in-document successor: masked.
        if remaining <= SEQ:
            assert not bool(mask[remaining - 1])
        assert int(mask.sum()) == scored
    assert saw_long and saw_short, "the sample did not cover both packing branches"


def test_previous_document_bytes_do_not_change_the_next_document(tmp_path):
    """Two documents, `doc_masking=true`: change doc 1, doc 2's logits are identical."""
    doc1 = bytes(range(16)) * 4  # 64 B, exactly one whole row
    doc2 = bytes((i * 9 + 2) % 251 for i in range(23))
    path = _docs_file(tmp_path, [doc1, doc2])
    model = _build("kda-dense")
    starts = load_doc_starts(path)

    def doc2_logits(p: Path):
        stream = DocPackedStream(ByteStream(p, 1, SEQ, check_min=False), starts)
        stream._row()  # doc1
        x, y, mask, _ = stream._row()  # doc2
        assert torch.equal(x[: len(doc2)], torch.tensor(list(doc2)))
        assert not bool(mask[len(doc2) - 1 :].any())
        with torch.no_grad():
            logits, _, _ = model(x[None], y[None], loss_mask=mask[None])
        return logits

    before = doc2_logits(path)
    path.write_bytes(bytes([7]) * len(doc1) + doc2)  # doc1 rewritten, doc2 untouched
    after = doc2_logits(path)
    assert torch.equal(before, after), "the previous document reached the new one"

    # And the same doc as the FIRST row of a file scores bit-identically: no
    # reader state, no cross-row context.
    first_path = _docs_file(tmp_path, [doc2], name="solo.bin")
    solo = DocPackedStream(ByteStream(first_path, 1, SEQ, check_min=False), [0])
    x, y, mask, _ = solo._row()
    with torch.no_grad():
        logits, _, _ = model(x[None], y[None], loss_mask=mask[None])
    assert torch.equal(before, logits)


def test_masked_eval_excludes_every_document_start(tmp_path):
    docs = [b"a" * 17, b"b" * 31, b"c" * 9]
    path = _docs_file(tmp_path, docs, name="eval.bin")
    starts = load_doc_starts(path)
    ev = MaskedByteStream(ByteStream(path, 1, 24), starts)
    x, y, mask = ev.next_batch_masked()
    assert x.shape == y.shape == mask.shape == (1, 24)
    starts_set = set(starts)
    assert 17 in starts_set
    assert not bool(mask[0, 16]), "the target at the doc start was not masked"
    for j in range(24):
        target = j + 1
        crosses = target >= ev.hi or target in starts_set
        assert bool(mask[0, j]) != crosses, f"position {j}: boundary mask is wrong"


# --- 5: the eval window vs the training stream ------------------------------


def _flagship_cfg(tmp_path: Path, **over) -> TrainConfig:
    base = dict(
        steps=1,
        batch=BATCH,
        seq_len=SEQ,
        lr=0.01,
        log_every=10,
        ckpt_every=100,
        ckpt_name="t",
        out_dir=str(tmp_path / "ck"),
        data=str(tmp_path / "train.bin"),
    )
    base.update(over)
    return TrainConfig(**base)


def test_the_same_file_split_is_disjoint_by_byte_range(tmp_path):
    size = 1 << 20
    p = tmp_path / "train.bin"
    p.write_bytes(bytes(range(256)) * (size // 256))
    run = tiny_run(tmp_path, data=str(p), eval=True, eval_bytes=65_536, eval_batches=None)
    train_stream, ev = open_streams(run)
    assert train_stream is not None and ev is not None
    assert train_stream.hi == ev.lo, "the split must touch, never overlap"
    assert train_stream.lo == 0 and ev.hi == size
    assert eval_train_overlap(run.train, train_stream, ev) == (0, None)
    assert train_stream.hi - train_stream.lo > 0 and ev.hi - ev.lo > 0
    assert train_stream.hi <= ev.lo and ev.lo < ev.hi


def test_eval_data_naming_a_training_file_is_refused(tmp_path):
    """The hole the split cannot see: `--eval` pointed at `--data`."""
    size = 1 << 20
    p = tmp_path / "big_train.bin"
    p.write_bytes(bytes(range(256)) * (size // 256))
    run = tiny_run(
        tmp_path,
        data=str(p),
        eval=True,
        eval_data=str(p),
        eval_bytes=65_536,
        eval_batches=None,
    )
    train_stream, ev = open_streams(run)
    leaked, where = eval_train_overlap(run.train, train_stream, ev)
    assert leaked == 512 * 2 * 65, "the whole eval read span is inside the train file"
    assert where is not None and str(p) in where
    with pytest.raises(LoudError, match="eval/train overlap"):
        train(DormouseModel(run.model), run, log=lambda *_: None)


def test_a_mix_plan_source_cannot_be_the_eval_file(tmp_path):
    """A 3-source plan where the eval file IS the second source: exact bytes named."""
    src_a = tmp_path / "a.bin"
    src_b = tmp_path / "b.bin"
    src_a.write_bytes(b"A" * 8192)
    src_b.write_bytes(b"B" * 8192)
    plan = MixPlan((MixStage(1.0, {"a": 0.5, "b": 0.5}),), {"a": src_a, "b": src_b})
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(dumps_mix_plan(plan))
    cfg = _flagship_cfg(
        tmp_path,
        mix_plan=str(plan_path),
        eval=True,
        eval_data=str(src_b),
        eval_batches=4,
        eval_offset=0,
        batch=2,
        seq_len=64,
    )
    run = dataclasses.replace(tiny_run(tmp_path), train=cfg)
    mix = MixtureStream(loads_mix_plan(plan_path.read_text()), 2, 64, steps=8)
    _t, ev = open_streams(run, eval_only=True)
    leaked, where = eval_train_overlap(cfg, mix, ev)
    assert leaked == 4 * 2 * 65, "the eval read span is batches x batch x (seq+1)"
    assert str(src_b) in where
    with pytest.raises(LoudError, match="eval/train overlap"):
        train(DormouseModel(run.model), run, log=lambda *_: None)


def test_a_mix_plan_with_an_eval_tail_of_its_own_source_is_refused(tmp_path):
    """The latent hole: `mix_plan` + `eval=true` with NO `eval_data`.

    `open_streams` carves the eval tail out of `train.data`; the mixture reads
    every planned source in full, so if `train.data` is one of them the tail is
    training data. The gate names the byte count; there is no silent run.
    """
    src = tmp_path / "a.bin"
    src.write_bytes(bytes((i * 17) % 251 for i in range(8192)))
    plan = MixPlan((MixStage(1.0, {"a": 1.0}),), {"a": src})
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(dumps_mix_plan(plan))
    cfg = _flagship_cfg(
        tmp_path,
        data=str(src),
        mix_plan=str(plan_path),
        eval=True,
        eval_batches=4,
        eval_offset=0,
        batch=2,
        seq_len=64,
    )
    run = dataclasses.replace(tiny_run(tmp_path), train=cfg)
    mix = MixtureStream(loads_mix_plan(plan_path.read_text()), 2, 64, steps=8)
    _t, ev = open_streams(run, eval_only=True)
    leaked, where = eval_train_overlap(cfg, mix, ev)
    assert leaked == 4 * 2 * 64, "the carved tail is the full eval window"
    assert where is not None and str(src) in where
    with pytest.raises(LoudError, match="eval/train overlap"):
        train(DormouseModel(run.model), run, log=lambda *_: None)


def _real_mix_plan() -> Path:
    return PORT / "configs" / "mix_v7.toml"


@pytest.mark.skipif(
    not _real_mix_plan().is_file()
    or not (DATA / "mix4" / "web_eval.bin").is_file()
    or not all(
        p.is_file() or p.is_dir()
        for p in loads_mix_plan(_real_mix_plan().read_text()).sources.values()
    ),
    reason="local mix_v7 data absent",
)
def test_the_flagship_eval_file_is_disjoint_from_every_mix_source(tmp_path):
    """The REAL night config: `web_eval.bin` vs all 12 planned sources, by inode/range."""
    plan = loads_mix_plan(_real_mix_plan().read_text())
    eval_path = DATA / "mix4" / "web_eval.bin"
    cfg = _flagship_cfg(
        tmp_path,
        mix_plan=str(_real_mix_plan()),
        eval=True,
        eval_data=str(eval_path),
        eval_bytes=196_608,
        eval_offset=0,
        batch=8,
        seq_len=1024,
    )
    run = dataclasses.replace(tiny_run(tmp_path), train=cfg)
    mix = MixtureStream(plan, 8, 1024, steps=8)
    _t, ev = open_streams(run, eval_only=True)
    leaked, where = eval_train_overlap(cfg, mix, ev)
    assert leaked == 0 and where is None, f"eval window overlaps {where}"

    eval_id = (eval_path.stat().st_dev, eval_path.stat().st_ino)
    for name, path in plan.sources.items():
        files = shard_files(path) if path.is_dir() else [path]
        for f in files:
            assert (f.stat().st_dev, f.stat().st_ino) != eval_id, f"{name} -> {f} is the eval file"

    # The read span is the documented window: batches x batch x (seq_len+1).
    assert ev.lo == 0 and ev.hi == eval_path.stat().st_size
    assert cfg.eval_bytes == 24 * 8 * 1024
    span = 24 * 8 * (1024 + 1)
    assert span == 196_800 and span <= ev.hi


# --- 6: the sampler ---------------------------------------------------------


def _flat_file(tmp_path: Path, cycles: int = 8) -> Path:
    window = BATCH * (SEQ + 1)
    p = tmp_path / "flat.bin"
    p.write_bytes(bytes((i * 31 + 5) % 251 for i in range(window * cycles)))
    return p


def test_two_runs_one_seed_read_the_same_flat_batches(tmp_path):
    p = _flat_file(tmp_path)
    a = ByteStream(p, BATCH, SEQ)
    b = ByteStream(p, BATCH, SEQ)
    for _ in range(20):
        xa, ya = a.next_batch()
        xb, yb = b.next_batch()
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_flat_epochs_are_periodic_and_that_is_documented(tmp_path):
    """`ByteStream` is a RING: epoch 2 is epoch 1 byte for byte (class docstring)."""
    p = _flat_file(tmp_path, cycles=5)
    stream = ByteStream(p, BATCH, SEQ)
    first = [stream.next_batch() for _ in range(5)]  # 5 x window == the range
    second = [stream.next_batch() for _ in range(5)]
    for (xa, ya), (xb, yb) in zip(first, second, strict=True):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert (stream.hi - stream.lo) == 5 * BATCH * (SEQ + 1)


def test_shard_order_is_seeded_and_the_epoch_repeats(tmp_path):
    """A shard directory: the permutation is `(seed)`; the epoch order is fixed."""
    d = tmp_path / "shards"
    d.mkdir()
    for i in range(8):
        (d / f"{i:02d}.txt").write_bytes(bytes([65 + i]) * (BATCH * (SEQ + 1)))
    files = shard_files(d)
    assert shard_order(files, 1) == shard_order(files, 1)
    assert shard_order(files, 1) != shard_order(files, 2)

    stream = ShardStream(d, BATCH, SEQ, seed=3)
    a = [stream.next_batch() for _ in range(len(files))]
    b = [stream.next_batch() for _ in range(len(files))]
    for (xa, ya), (xb, yb) in zip(a, b, strict=True):
        assert torch.equal(xa, xb) and torch.equal(ya, yb), "epoch 2 re-read a different order"


def test_the_mixture_draw_is_a_pure_function_of_seed_and_step(tmp_path):
    names = ("a", "b", "c")
    sources = {}
    for name in names:
        p = tmp_path / f"{name}.bin"
        p.write_bytes(bytes([65 + names.index(name)]) * 4096)
        sources[name] = p
    plan = MixPlan(
        (
            MixStage(0.5, {"a": 0.5, "b": 0.3, "c": 0.2}),
            MixStage(1.0, {"a": 0.2, "b": 0.3, "c": 0.5}),
        ),
        sources,
    )
    steps = 60
    one = MixPlan(plan.stages, plan.sources, seed=11)
    two = MixPlan(plan.stages, plan.sources, seed=11)
    seq1 = [source_for_step(one, k, steps) for k in range(steps)]
    seq2 = [source_for_step(two, k, steps) for k in range(steps)]
    assert seq1 == seq2
    other = [
        source_for_step(MixPlan(plan.stages, plan.sources, seed=12), k, steps) for k in range(steps)
    ]
    assert seq1 != other, "the seed does not reach the source draw"

    s1 = MixtureStream(one, 1, 64, steps=steps)
    s2 = MixtureStream(two, 1, 64, steps=steps)
    for _ in range(steps):
        x1, y1 = s1.next_batch()
        x2, y2 = s2.next_batch()
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert s1.step == s2.step == steps


def test_the_mixture_does_not_replay_its_source_draw_next_pass(tmp_path):
    """Two passes over the step budget: the SOURCE draw is not periodic.

    The per-source BYTES are (each `ByteStream` is a ring), but the mixture's
    global sequence is `(seed, step)`; a second pass draws a different source
    order. That is the documented behaviour, and it is why `max_epochs` exists.
    """
    sources = {}
    for name in ("a", "b"):
        p = tmp_path / f"{name}.bin"
        p.write_bytes(bytes([65 + len(name)]) * 4096)
        sources[name] = p
    plan = MixPlan((MixStage(1.0, {"a": 0.5, "b": 0.5}),), sources, seed=5)
    steps = 40
    draws = [source_for_step(plan, k, steps) for k in range(steps)]
    # A second "epoch" is the same step index range again only if the loop keeps
    # counting; the stream counts past `steps`, and `source_for_step` keeps
    # drawing from the hash. Steps `[steps, 2*steps)` are NOT `[0, steps)`.
    second = [source_for_step(plan, k, steps) for k in range(steps, 2 * steps)]
    assert draws != second, "the source draw repeated exactly: it is not step-pure"


def test_the_unit_draw_has_no_rng_state(tmp_path):
    import random

    from dormouse.train.mixture import _unit

    before = [_unit(7, k) for k in range(5)]
    for _ in range(100):
        random.random()
    torch.rand(100)
    after = [_unit(7, k) for k in range(5)]
    assert before == after
    assert 0.0 <= min(after) < max(after) < 1.0


# --- the fix seam: a resume is exact ---------------------------------------


def test_a_doc_masked_resume_replays_the_same_packing(tmp_path):
    """`skip_bytes` on a packed mixture must land on the same rows (lengths only)."""
    docs = [b"d" * n for n in (7, 90, 13, 130, 5, 64, 31)]
    a = _docs_file(tmp_path, docs, name="a.bin")
    plan = MixPlan((MixStage(1.0, {"a": 1.0}),), {"a": a})
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(dumps_mix_plan(plan))
    whole = MixtureStream(plan, 2, 64, steps=12, doc_masking=True)
    resumed = MixtureStream(plan, 2, 64, steps=12, doc_masking=True)
    for _ in range(4):
        whole.next_batch_masked()
        resumed.next_batch_masked()
    resumed.skip_bytes(4 * 2 * (64 + 1))
    assert resumed.step == 4
    for _ in range(6):
        xa, ya, ma = whole.next_batch_masked()
        xb, yb, mb = resumed.next_batch_masked()
        assert torch.equal(xa, xb) and torch.equal(ya, yb) and torch.equal(ma, mb)
