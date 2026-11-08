"""Eval correctness (2026-11-07): `bpb_fin` IS `-log2 p(target)` over the bytes.

The audit's check is written from scratch, not by calling the trainer's eval:
load the checkpoint, build the model, read the SAME window, run the forward,
sum the target nats with `-log_softmax(...).gather` and divide by
`ln2 x scored bytes`. It must equal the reported `fin` to 1e-5.

That check found one defect: a doc-masked eval averaged the per-batch means,
so a batch of 4 scored positions weighed as much as a batch of 116, and the
reported number (9.051990 BPB) was 0.0062 away from the same window's bytes
(9.058218). `_eval_loss` now weights every mean by the batch's scored bytes;
the flat path is unchanged because its weights are constant.

The window/offset are part of the measurement: the bytes are pinned by
`eval_data` + `eval_offset`, printed by `off=`, and the independent read below
starts at exactly that offset.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import RunConfig, TrainConfig
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.data import eval_batches_for, open_streams
from dormouse.train.loop import _eval_line, load_ckpt, load_model_state

TFORM = dict(
    block_form="transformer",
    n_layers=2,
    max_iter=1,
    loop_blocks=1,
    use_kda=False,
    use_csa=False,
    use_mhc=False,
    use_attnres=False,
    use_engram=False,
    layerrope=False,
    n_experts=1,
    moe_topk=0,
    use_tsct=False,
)
KDAFORM = dict(block_form="kda", max_iter=2, moe_topk=2)
LN2 = math.log(2)
SEQ, BATCH, BATCHES = 64, 2, 2


def _doc_file(path: Path, docs: list[bytes]) -> None:
    data = b"".join(docs)
    path.write_bytes(data)
    starts = [0]
    for d in docs[:-1]:
        starts.append(starts[-1] + len(d))
    Path(str(path) + ".docs.json").write_text(
        json.dumps({"version": 1, "size": len(data), "starts": starts})
    )


def _fixture(tmp_path: Path, form: dict, *, doc_masking: bool) -> dict:
    """Train two steps, write a checkpoint, reload it, and take the eval line."""
    corpus = tmp_path / "corpus.bin"
    corpus.write_bytes(bytes(range(16)) * 256)
    if doc_masking:
        # Train needs a sidecar too; it never enters the eval number below.
        Path(str(corpus) + ".docs.json").write_text(
            json.dumps({"version": 1, "size": 4096, "starts": [0, 1000, 3000]})
        )
    # Very uneven documents: the two eval batches score DIFFERENT byte counts,
    # which is the fixture that separates a byte mean from a mean of means.
    docs = [bytes([65 + i]) * (i * 7 + 1) for i in range(12)]
    ev_path = tmp_path / "eval.bin"
    _doc_file(ev_path, docs)

    cfg = TrainConfig(
        steps=2,
        batch=BATCH,
        seq_len=SEQ,
        lr=0.01,
        log_every=10,
        ckpt_every=2,
        ckpt_name="t",
        out_dir=str(tmp_path / "ck"),
        data=str(corpus),
        eval=True,
        eval_every=2,
        eval_data=str(ev_path),
        eval_batches=BATCHES,
        eval_offset=3,
        doc_masking=doc_masking,
    )
    run = RunConfig(source="eval", model=tiny_cfg(**form), train=cfg)
    torch.manual_seed(7)
    train(DormouseModel(run.model), run, log=lambda *_: None)

    ck, stored = load_ckpt(run)
    model = DormouseModel(stored.model)
    load_model_state(model, stored.train, ck["model"])
    _, ev = open_streams(stored)
    line, ev_mean, scored, fin = _eval_line(model, ev, stored.train, run.model.max_iter, None, 2)
    return {
        "run": stored,
        "ck": ck,
        "model": model,
        "line": line,
        "ev": ev_mean,
        "scored": scored,
        "fin": fin,
        "ev_path": ev_path,
    }


def _independent(model: DormouseModel, run: RunConfig) -> tuple[float, int, list[float]]:
    """`(-log2 p(target))` over the eval window, from the bytes up.

    Returns `(bpb, scored_bytes, per_batch_bpb)`; no trainer eval helper is
    called, and the model is the checkpoint loaded by `_fixture`.
    """
    _, stream = open_streams(run)
    assert stream.lo == run.train.eval_offset
    model.eval()
    nats = 0.0
    count = 0
    per_batch: list[float] = []
    masked = hasattr(stream, "next_batch_masked")
    with torch.no_grad():
        for _ in range(eval_batches_for(run.train)):
            if masked:
                x, y, mask = stream.next_batch_masked()
            else:
                x, y = stream.next_batch()
                mask = torch.ones_like(y, dtype=torch.bool)
            logits, _, _ = model(x, y, pure=True)
            ce = -torch.log_softmax(logits, dim=-1).gather(1, y.reshape(-1, 1))
            ce = ce.reshape_as(mask)
            nats += float(ce[mask].sum())
            n = int(mask.sum())
            count += n
            per_batch.append(float(ce[mask].mean()) / LN2)
    assert count > 0
    return nats / (LN2 * count), count, per_batch


@pytest.mark.parametrize("form", [TFORM, KDAFORM], ids=["transformer", "kda"])
def test_reported_bpb_fin_is_the_scored_byte_mean(tmp_path, form):
    f = _fixture(tmp_path, form, doc_masking=False)
    bpb, count, per_batch = _independent(f["model"], f["run"])
    assert count == f["scored"] == BATCHES * BATCH * SEQ
    assert bpb == pytest.approx(f["fin"] / LN2, abs=1e-5), (
        f"reported {f['fin'] / LN2} vs independent {bpb}"
    )
    assert len(set(per_batch)) > 1, "the fixture must have distinct batch means"
    assert f"bpb_fin={f['fin'] / LN2:.3f}" in f["line"]
    assert f"over {count} B" in f["line"]
    assert "off=3" in f["line"], "the eval offset must be printed"
    assert "eval_docmask=flat (doc_masking=false)" in f["line"]


@pytest.mark.parametrize("form", [TFORM, KDAFORM], ids=["transformer", "kda"])
def test_a_doc_masked_eval_is_weighted_by_scored_bytes(tmp_path, form):
    """The 2026-11-07 defect: mean-of-means vs mean over the same bytes.

    The fixture's batches score different numbers of positions, so the two
    definitions separate by far more than the 1e-5 gate. The reported number
    must be the byte mean; the mean-of-means is recomputed here to prove the
    fixture would catch a revert.
    """
    f = _fixture(tmp_path, form, doc_masking=True)
    bpb, count, per_batch = _independent(f["model"], f["run"])
    mean_of_means = sum(per_batch) / len(per_batch)
    assert count == f["scored"]
    assert count < BATCHES * BATCH * SEQ, "the fixture must mask something"
    assert len(set(per_batch)) > 1, "the fixture must score unequally"
    assert abs(mean_of_means - bpb) > 1e-4, (
        f"the fixture does not separate the definitions: {mean_of_means} vs {bpb}"
    )
    assert bpb == pytest.approx(f["fin"] / LN2, abs=1e-5), (
        f"reported {f['fin'] / LN2} vs byte mean {bpb} "
        f"(mean-of-means {mean_of_means}); cause: per-batch scored counts"
    )
    assert "eval_docmask=sidecar" in f["line"]
    assert f"over {count} B" in f["line"]


def test_a_different_offset_is_a_different_window(tmp_path):
    """`eval_offset` is the byte the window starts at, not decoration."""
    f = _fixture(tmp_path, TFORM, doc_masking=False)
    bpb0, count0, _ = _independent(f["model"], f["run"])
    run = f["run"]
    moved = RunConfig(
        source=run.source,
        model=run.model,
        train=dataclasses.replace(run.train, eval_offset=0),
    )
    bpb1, count1, _ = _independent(f["model"], moved)
    assert count1 == count0
    assert bpb0 != bpb1, "moving the offset did not move the scored bytes"
    assert (
        "off=3"
        not in _eval_line(
            f["model"], open_streams(moved)[1], moved.train, run.model.max_iter, None, 2
        )[0]
    )
