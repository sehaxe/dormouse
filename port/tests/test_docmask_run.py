"""The docmask-run gates: audit, builder arithmetic, eval policy, boundary loss.

Each test fails on a revert of the thing it names:

a. `docmask_audit.py` reports REAL and the EXACT document count for a synthetic
   `.bin` + sidecar (and BOGUS/REFUSED for the two failure shapes);
b. the builder refuses records that do not sum to the `.bin`, naming the offset;
c. the trainer refuses a sidecar whose `size` is not the file's (`Escape:`);
d. `doc_masking=true` changes the loss ONLY at a document boundary: a boundary
   target is excluded from the masked mean while every mid-document per-token
   loss is bit-identical;
e. `doc_masking=false` keeps the flat reader and the flat eval policy (the
   bit-identical training path is pinned by `test_docmask.py`'s golden).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train.data import (
    ByteStream,
    DocPackedStream,
    MaskedByteStream,
    doc_sidecar_path,
    load_doc_starts,
    open_streams,
)
from dormouse.train.loop import _eval_loss
from dormouse.train.mixture import load_mix_plan

PORT = Path(__file__).resolve().parents[1]
AUDIT = PORT / "scripts" / "docmask_audit.py"
BUILDER = PORT / "scripts" / "build_doc_sidecar.py"

TINY = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=64,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=2,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(PORT / "src")
    return env


def _docs_file(tmp_path: Path, docs: list[bytes], name: str = "d.bin") -> Path:
    data = b"".join(docs)
    p = tmp_path / name
    p.write_bytes(data)
    starts = [0]
    for d in docs[:-1]:
        starts.append(starts[-1] + len(d))
    doc_sidecar_path(p).write_text(
        json.dumps({"version": 1, "size": len(data), "starts": starts})
    )
    return p


def _run(tmp_path: Path, data: Path, **over) -> RunConfig:
    import dataclasses

    cfg = TrainConfig(
        steps=1,
        batch=2,
        seq_len=16,
        lr=0.01,
        log_every=1,
        ckpt_every=10**9,
        ckpt_name="g",
        out_dir=str(tmp_path),
        data=str(data),
    )
    return RunConfig(
        source="docmask-run",
        model=ModelConfig(**TINY),
        train=dataclasses.replace(cfg, **over) if over else cfg,
    )


# --- gate a: the audit ------------------------------------------------------


def test_audit_reports_real_and_the_exact_count(tmp_path):
    docs = [b"A" * 100, b"B" * 150, b"C" * 200]
    path = _docs_file(tmp_path, docs)
    report = tmp_path / "audit.json"
    out = subprocess.run(
        [sys.executable, str(AUDIT), "--bin", str(path), "--json", str(report)],
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )
    assert out.returncode == 0, out.stderr
    row = json.loads(report.read_text())["sources"][0]
    assert row["verdict"] == "REAL", row
    assert row["documents"] == len(docs) == 3
    assert row["bytes"] == 450
    assert row["mean_doc_bytes"] == pytest.approx(150.0)


def test_audit_rejects_a_file_seam_sidecar_and_names_a_builder_refusal(tmp_path):
    # One "document" per 30 KB: the audit's own BOGUS rule must fire.
    path = tmp_path / "bogus.bin"
    path.write_bytes(b"x" * 30_000)
    doc_sidecar_path(path).write_text(
        json.dumps({"version": 1, "size": 30_000, "starts": [0]})
    )
    report = tmp_path / "audit.json"
    out = subprocess.run(
        [sys.executable, str(AUDIT), "--bin", str(path), "--json", str(report)],
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )
    assert out.returncode == 0, out.stderr
    row = json.loads(report.read_text())["sources"][0]
    assert row["verdict"] == "BOGUS"
    assert "mean document" in row["reason"]

    # No sidecar + a builder refusal is REFUSED-BY-BUILDER, not MISSING.
    refused = tmp_path / "flat.bin"
    refused.write_bytes(b"y" * 400)
    Path(str(refused) + ".docs.refused.json").write_text(
        json.dumps({"version": 1, "size": 400, "reason": "no record token"})
    )
    out = subprocess.run(
        [sys.executable, str(AUDIT), "--bin", str(refused)],
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert '"verdict": "REFUSED-BY-BUILDER"' in out.stdout
    assert "no record token" in out.stdout


# --- gate b: the builder's sum check ---------------------------------------


def test_builder_refuses_records_that_do_not_sum_to_the_bin(tmp_path):
    src = tmp_path / "shards"
    src.mkdir()
    a, b = b"A" * 100, b"B" * 200
    (src / "a.txt").write_bytes(a)
    (src / "b.txt").write_bytes(b)
    bin_path = tmp_path / "domain.bin"
    bin_path.write_bytes(a + b + b"X")  # one byte too many
    out = subprocess.run(
        [sys.executable, str(BUILDER), "--bin", str(bin_path), "--source-dir", str(src)],
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )
    assert out.returncode != 0
    assert "records end at 300 B" in out.stderr
    assert "off by 1 B" in out.stderr
    assert "offset 300" in out.stderr
    assert not doc_sidecar_path(bin_path).exists()

    # And the short side names the cursor too.
    bin_path.write_bytes(a + b[:-1])
    out = subprocess.run(
        [sys.executable, str(BUILDER), "--bin", str(bin_path), "--source-dir", str(src)],
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )
    assert out.returncode != 0
    assert "offset 100" in out.stderr
    assert not doc_sidecar_path(bin_path).exists()

    # The honest one writes exactly the three starts.
    bin_path.write_bytes(a + b)
    out = subprocess.run(
        [sys.executable, str(BUILDER), "--bin", str(bin_path), "--source-dir", str(src)],
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert load_doc_starts(bin_path) == [0, 100]


# --- gate c: a sidecar that lies about the file ----------------------------


def _builder_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_doc_sidecar", BUILDER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_builder_records_core_refuses_an_off_by_one(tmp_path):
    """The core the CLI and every domain replay use: sum(records) must equal size."""
    mod = _builder_module()
    a, b = b"A" * 100, b"B" * 200
    path = tmp_path / "core.bin"
    path.write_bytes(a + b + b"X")
    with pytest.raises(SystemExit, match="off by 1 B at offset 300"):
        mod.starts_from_records(path, [a, b])
    path.write_bytes(a + b)
    assert mod.starts_from_records(path, [a, b]) == [0, 100]


def test_trainer_refuses_a_sidecar_with_the_wrong_total_length(tmp_path):
    path = tmp_path / "corpus.bin"
    path.write_bytes(bytes(range(256)) * 16)
    doc_sidecar_path(path).write_text(
        json.dumps({"version": 1, "size": 4095, "starts": [0, 100]})
    )
    run = _run(tmp_path, path, doc_masking=True)
    with pytest.raises(LoudError, match="Escape:"):
        open_streams(run)
    with pytest.raises(LoudError, match="describes 4095 B"):
        load_doc_starts(path)


# --- gate d: the mask changes the loss only at a boundary ------------------


def test_masked_loss_moves_only_at_the_document_boundary(tmp_path):
    """Two 64-byte documents, `seq=64`: the last target is doc2's first byte.

    The flat reader scores it; the packed reader's mask drops it. Every
    mid-document per-token CE is identical between the two forwards (the model
    and the inputs are the same; only the reduction changes).
    """
    doc1 = b"A" * 64
    doc2 = b"A" * 64  # its first byte equals doc1's last: predictable by copy
    path = _docs_file(tmp_path, [doc1, doc2])
    flat = ByteStream(path, 1, 64)
    fx, fy = flat.next_batch()
    assert int(fy[0, -1]) == int(doc2[0]) == int(fx[0, -1])

    packed = DocPackedStream(ByteStream(path, 1, 64, check_min=False), load_doc_starts(path))
    px, py, mask = packed.next_batch_masked()
    assert torch.equal(px, fx), "the packed row must hold doc1's 64 bytes"
    assert int(mask.sum()) == 63, "the document's last byte is the one masked"
    assert not bool(mask[0, -1])
    assert torch.equal(py[0, :-1], fy[0, :-1])
    assert int(py[0, -1]) == 0  # padding, never doc2's byte

    torch.manual_seed(0)
    model = DormouseModel(ModelConfig(**{**TINY, "max_seq_len": 64}))
    model.eval()
    with torch.no_grad():
        logits_u, unmasked, _ = model(fx, fy)
        logits_m, masked, _ = model(fx, fy, loss_mask=mask)
        ce_u = -torch.log_softmax(logits_u, dim=-1).gather(1, fy.reshape(-1, 1))
        ce_m = -torch.log_softmax(logits_m, dim=-1).gather(1, fy.reshape(-1, 1))
    assert unmasked is not None and masked is not None
    assert not torch.equal(masked, unmasked), "the boundary target did not change the mean"
    keep = mask.reshape(-1)
    assert bool(keep.any()) and not bool(keep.all())
    # The mask is a reduction, not a different forward: EVERY per-token loss is
    # identical, including the mid-document ones. The mean moves because the
    # boundary target is excluded and the denominator is 63, not 64.
    assert torch.equal(logits_u, logits_m)
    assert torch.equal(ce_u, ce_m)
    # The excluded position is a real, finite target (doc2's first byte), not padding.
    assert torch.isfinite(ce_u[63])
    print(
        f"masked={float(masked):.9f} unmasked={float(unmasked):.9f} "
        f"ce_boundary={float(ce_u[63]):.6f} ce_mid={float(ce_u[0]):.6f}"
    )


# --- gate e + the eval wiring ----------------------------------------------


def test_eval_uses_the_same_sidecar_when_it_has_one(tmp_path):
    docs = [b"A" * 64, b"B" * 64, b"C" * 64]
    data = _docs_file(tmp_path, docs, name="train.bin")
    eval_path = _docs_file(tmp_path, docs, name="eval.bin")
    run = _run(
        tmp_path,
        data,
        doc_masking=True,
        eval=True,
        eval_data=str(eval_path),
        eval_batches=1,
    )
    tr, ev = open_streams(run)
    assert isinstance(tr, DocPackedStream)
    assert isinstance(ev, MaskedByteStream)
    assert ev.doc_boundary_policy == "sidecar"
    x, y, mask = ev.next_batch_masked()
    assert x.shape == (2, 16) and mask.shape == (2, 16)
    # The first window is inside doc0; the mask is not vacuous at a seam later.
    assert bool(mask.any())

    # Evaluate through the same function the trainer uses: the scored count is
    # the mask's, not batch*seq.
    torch.manual_seed(0)
    model = DormouseModel(ModelConfig(**TINY))
    loss, scored, fin, _ = _eval_loss(model, ev, 1, None, None)
    assert loss > 0 and fin > 0
    assert scored == int(mask.sum())


def test_eval_stays_flat_and_names_it_when_the_sidecar_is_absent(tmp_path):
    data = _docs_file(tmp_path, [b"A" * 64, b"B" * 64], name="train.bin")
    eval_path = tmp_path / "eval.bin"
    eval_path.write_bytes(b"A" * 64 + b"B" * 64)  # no eval.bin.docs.json
    run = _run(
        tmp_path,
        data,
        doc_masking=True,
        eval=True,
        eval_data=str(eval_path),
        eval_batches=1,
    )
    _tr, ev = open_streams(run)
    assert isinstance(ev, ByteStream)
    assert ev.doc_boundary_policy == "flat (no sidecar)"

    torch.manual_seed(0)
    model = DormouseModel(ModelConfig(**TINY))
    loss, scored, fin, _ = _eval_loss(model, ev, 1, None, None)
    assert loss > 0 and fin > 0
    assert scored == 2 * 16


def test_off_keeps_the_flat_eval_and_no_mask(tmp_path):
    data = _docs_file(tmp_path, [b"A" * 256, b"B" * 256], name="train.bin")
    run = _run(tmp_path, data, eval=True, eval_batches=3)
    tr, ev = open_streams(run)
    assert isinstance(tr, ByteStream) and isinstance(ev, ByteStream)
    assert ev.doc_boundary_policy == "flat (doc_masking=false)"
    assert not isinstance(ev, MaskedByteStream)


def test_the_docmask_plan_names_only_sources_with_real_sidecars():
    """mix_v7 minus the three REFUSED-BY-BUILDER sources, weights renormalized."""
    path = PORT / "configs" / "mix_v7_docmask.toml"
    plan = load_mix_plan(path)
    assert set(plan.sources) == {
        "fwedu",
        "cosmo",
        "ru",
        "math",
        "finemath",
        "science",
        "code",
        "wiki_ru",
        "instruct",
    }
    assert all(abs(sum(s.weights.values()) - 1.0) < 1e-6 for s in plan.stages)
    missing = [
        str(p)
        for p in plan.sources.values()
        if not p.is_file() or not doc_sidecar_path(p).is_file()
    ]
    if missing:
        pytest.skip(f"local mix data/sidecars absent: {missing[:2]}")
    for name, p in plan.sources.items():
        assert load_doc_starts(p), name
