"""Document masking: the OFF parity, the packed window, the masked loss, refusals.

Every gate here fails on a revert:

1. `doc_masking=false` is `main`, byte for byte — the state_dict, logits and
   batch-tensor sha256s were captured from a CLEAN `main` checkout
   (`scripts/docmask_goldens.py`, run BEFORE the arm existed);
2. a synthetic 3-document file produces rows that never hold two documents
   (asserted on the batch tensors, boundaries printed);
3. the loss ignores every position outside a document (the masked count is
   asserted, padding bytes are mutated, and the loss is bit-identical) while a
   change to a SCORED target still moves it;
4. the packed single-document window and the equivalent hand-built
   `batch=1, seq=doc_len` flat window produce the same loss;
5. a 4096-byte window of a REAL mix4 shard runs forward+backward with the
   boundary counters non-zero (skipped when the local data is absent);
6. every refusal is LOUD and names the escape;
7. the sidecar tool recovers a synthetic shard-built `.bin`'s offsets exactly.

The comparison is against `DocPackedStream` (one document per row), not
whole-document packing: the KDA recurrence has no mid-window state reset, so a
row holding two documents would leak document A into document B — see the
class docstring and `TrainConfig.doc_masking`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from dormouse.config import ModelConfig, RunConfig, TrainConfig, _validate_doc
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import DocPackedStream, train
from dormouse.train.data import (
    ByteStream,
    load_doc_starts,
    open_streams,
)
from dormouse.train.mixture import MixtureStream, loads_mix_plan

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
SEQ, BATCH, STEPS = 16, 2, 3
GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "docmask_golden.json").read_text())
DOCS = [b"A" * 5, b"B" * 7, b"C" * 9]  # 5 + 7 + 9 = 21 bytes, distinct letters


# --- helpers ----------------------------------------------------------------


def _hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _state_hash(model: DormouseModel) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _corpus(path: Path) -> Path:
    path.write_bytes(bytes(range(16)) * 256)
    return path


def _doc_file(tmp_path: Path, docs: list[bytes], name: str = "d.bin") -> Path:
    p = tmp_path / name
    data = b"".join(docs)
    p.write_bytes(data)
    starts = [0]
    for d in docs[:-1]:
        starts.append(starts[-1] + len(d))
    (tmp_path / (name + ".docs.json")).write_text(
        json.dumps({"version": 1, "size": len(data), "starts": starts})
    )
    return p


def _doc_stream(path: Path, batch: int, seq: int) -> DocPackedStream:
    return DocPackedStream(ByteStream(path, batch, seq, check_min=False), load_doc_starts(path))


def _run(tmp_path: Path, corpus: Path | None = None, **over) -> RunConfig:
    cfg = TrainConfig(
        steps=STEPS,
        batch=BATCH,
        seq_len=SEQ,
        lr=0.01,
        log_every=1,
        ckpt_every=10**9,
        ckpt_name="g",
        out_dir=str(tmp_path),
        data=str(corpus or _corpus(tmp_path / "corpus.bin")),
    )
    import dataclasses

    return RunConfig(
        source="docmask-test",
        model=ModelConfig(**TINY),
        train=dataclasses.replace(cfg, **over) if over else cfg,
    )


# --- gate 1: OFF is main ----------------------------------------------------


@pytest.mark.local_pin
def test_off_is_bit_identical_to_main(tmp_path):
    """`doc_masking=false` moves no tensor: the state, the logits, the batches."""
    assert GOLDEN["fixture"] == {"corpus": "bytes(range(16))*256", "batch": BATCH, "seq": SEQ}
    run = _run(tmp_path)
    assert run.train.doc_masking is False
    torch.manual_seed(0)
    model = DormouseModel(run.model)
    assert _state_hash(model) == GOLDEN["init_state"], "the OFF state_dict moved"
    stream = ByteStream(Path(run.train.data), BATCH, SEQ)
    batches = [stream.next_batch() for _ in range(3)]
    for i, (x, y) in enumerate(batches):
        assert [_hash(x), _hash(y)] == GOLDEN["batches"][i], f"batch {i} moved"

    torch.manual_seed(7)
    ids = (torch.arange(2 * 16, dtype=torch.long) % 251).reshape(2, 16)
    model.eval()
    with torch.no_grad():
        logits, loss, _ = model(ids, ids + 1)
    assert loss is not None
    assert _hash(logits) == GOLDEN["logits"], "the OFF logits moved"
    assert repr(float(loss)) == GOLDEN["loss"]
    model.train()

    losses: list[str] = []

    def cap(line: str) -> None:
        m = re.search(r"loss=([0-9.eE+-]+)", line)
        if m and "step=" in line:
            losses.append(m.group(1))

    train(model, run, stream=stream, log=cap)
    assert losses == GOLDEN["train_losses"], f"the OFF losses moved: {losses}"
    assert _state_hash(model) == GOLDEN["train_state"], "the OFF training state moved"


# --- gate 2: the packed window holds no straddling document -----------------


def test_windows_never_straddle_a_document(tmp_path):
    """Three documents, `seq=6`: every row is one document (or one doc chunk)."""
    path = _doc_file(tmp_path, DOCS)
    stream = _doc_stream(path, batch=2, seq=6)
    expected = [
        (0, 0, 5, 4),  # doc 0, 5 B: 4 scored targets, then padding
        (1, 0, 6, 6),  # doc 1, 7 B > 6: 6 input bytes, the 7th is the last target
        (2, 0, 6, 6),  # doc 2, first chunk (7 bytes consumed: 6 + 1 target)
        (2, 7, 2, 1),  # doc 2, remaining 2 B: 1 target, then padding
    ]
    row = 0
    for _ in range(2):  # 2 windows x batch 2
        x, y, mask = stream.next_batch_masked()
        for b in range(2):
            got = x[b].tolist()
            real = [v for v in got if v != 0]
            assert len(set(real)) == 1, f"row {row} glues documents: {bytes(got)!r}"
            doc, off, n, scored = expected[row]
            want = list(DOCS[doc][off : off + n]) + [0] * (6 - n)
            assert got == want, f"row {row}: {bytes(got)!r} != {bytes(want)!r}"
            assert int(mask[b].sum()) == scored, f"row {row}: masked count"
            print(
                f"row={row} doc={doc} off={off} real={n} scored={scored} "
                f"x={bytes(got)!r} mask={mask[b].tolist()}"
            )
            row += 1
    counts = stream.doc_counts()
    assert counts["docmask_docs"] == 4
    # The exact masked count is recomputed from the layout, not trusted:
    want_masked = sum(6 - scored for _doc, _off, _n, scored in expected)
    assert counts["docmask_masked"] == want_masked
    assert counts["docmask_doc_ends"] == 2  # doc 0 ends in row 0; doc 2 in row 3


def test_open_streams_wraps_the_train_side_only(tmp_path):
    """`doc_masking=true` → the train stream is packed, the eval stream is flat."""
    corpus = _corpus(tmp_path / "corpus.bin")
    # Three documents over the 4096-byte fixture, then the run asks for eval.
    starts = [0, 1000, 3000]
    (tmp_path / "corpus.bin.docs.json").write_text(
        json.dumps({"version": 1, "size": 4096, "starts": starts})
    )
    run = _run(tmp_path, corpus, doc_masking=True, eval=True)
    tr, ev = open_streams(run)
    assert isinstance(tr, DocPackedStream)
    assert isinstance(ev, ByteStream)
    x, y, mask = tr.next_batch_masked()
    assert x.shape == (BATCH, SEQ) == y.shape == mask.shape
    assert mask.sum() > 0
    ids, _ = ev.next_batch()
    assert ids.shape == (BATCH, SEQ)


# --- gate 3: the loss ignores positions outside documents -------------------


def test_masked_loss_ignores_padding_and_boundary_targets(tmp_path):
    """Mutating every masked position leaves the loss bit-identical."""
    path = _doc_file(tmp_path, DOCS)
    stream = _doc_stream(path, batch=2, seq=6)
    x, y, mask = stream.next_batch_masked()
    assert int(mask.sum()) == 4 + 6  # one window: doc 0 (whole) + doc 1 (chunk)
    torch.manual_seed(0)
    model = DormouseModel(ModelConfig(**{**TINY, "moe_capacity": 0.0}))
    _, loss_a, _ = model(x, y, loss_mask=mask)
    assert loss_a is not None

    # Every masked position (padding + a document's last byte) is mutated to a
    # different value with the SAME mask: the loss must not move.
    g = torch.Generator().manual_seed(3)
    x2 = x.clone()
    y2 = y.clone()
    x2[~mask] = torch.randint(0, 256, (int((~mask).sum()),), generator=g)
    y2[~mask] = torch.randint(0, 256, (int((~mask).sum()),), generator=g)
    _, loss_b, _ = model(x2, y2, loss_mask=mask)
    assert torch.equal(loss_a, loss_b), f"masked positions moved the loss: {loss_a} vs {loss_b}"

    # And the mask is not vacuous: moving a SCORED target does move it.
    y3 = y.clone()
    j = int(mask[0].nonzero()[0])
    y3[0, j] = (int(y3[0, j]) + 1) % 256
    _, loss_c, _ = model(x, y3, loss_mask=mask)
    assert not torch.equal(loss_a, loss_c), "a scored target did not move the loss"


# --- gate 4: packed single-doc == hand-built flat window --------------------


def test_packed_document_equals_handbuilt_flat_window(tmp_path):
    """One 300-byte document at `seq=64` IS the flat `batch=1, seq=64` window."""
    doc = bytes((i * 7 + 3) % 256 for i in range(300))
    path = tmp_path / "long.bin"
    path.write_bytes(doc)
    (tmp_path / "long.bin.docs.json").write_text(
        json.dumps({"version": 1, "size": len(doc), "starts": [0]})
    )
    packed = _doc_stream(path, batch=1, seq=64)
    x, y, mask = packed.next_batch_masked()
    flat = ByteStream(path, 1, 64).next_batch()
    assert torch.equal(x, flat[0]) and torch.equal(y, flat[1])
    assert bool(mask.all()), "a chunk inside one document must score every position"

    torch.manual_seed(0)
    model = DormouseModel(ModelConfig(**TINY))
    _, l_flat, _ = model(x, flat[1])
    _, l_packed, _ = model(x, y, loss_mask=mask)
    assert l_flat is not None and l_packed is not None
    assert abs(float(l_flat.detach()) - float(l_packed.detach())) <= 1e-6, (
        f"packed {float(l_packed.detach())} vs flat {float(l_flat.detach())}"
    )
    print(f"packed={float(l_packed.detach()):.9f} flat={float(l_flat.detach()):.9f}")


# --- the mixture stream -----------------------------------------------------


def test_mixture_doc_masking_resume_and_mask(tmp_path):
    """Two `.bin` sources: packed windows, rewind, and an exact step resume."""
    a = _doc_file(tmp_path, [b"a" * 9, b"b" * 4, b"c" * 11], name="a.bin")
    b = _doc_file(tmp_path, [b"x" * 6, b"y" * 8, b"z" * 5], name="b.bin")
    plan_path = tmp_path / "mix.toml"
    plan_path.write_text(
        f"max_epochs = 0\nseed = 1\n[[stage]]\nuntil_frac = 1.0\n"
        f"[stage.weights]\na = 0.5\nb = 0.5\n[sources]\na = {json.dumps(str(a))}\n"
        f"b = {json.dumps(str(b))}\n"
    )
    loaded = loads_mix_plan(plan_path.read_text())
    window = 2 * (8 + 1)
    one = MixtureStream(loaded, 2, 8, 6, seed=1, doc_masking=True)
    first = [one.next_batch_masked() for _ in range(3)]
    assert one.doc_counts()["docmask_windows"] == 6
    with pytest.raises(LoudError, match="next_batch"):
        one.next_batch()
    one.rewind()
    again = [one.next_batch_masked() for _ in range(3)]
    for (x1, y1, m1), (x2, y2, m2) in zip(first, again, strict=True):
        assert torch.equal(x1, x2) and torch.equal(y1, y2) and torch.equal(m1, m2)

    two = MixtureStream(loaded, 2, 8, 6, seed=1, doc_masking=True)
    two.skip_bytes(2 * window)  # a resume: the trainer's formula, 2 steps of this batch
    assert torch.equal(two.next_batch_masked()[0], first[2][0])
    assert two.doc_counts()["docmask_docs"] > 0


def test_a_directory_source_is_refused_with_doc_masking(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    (d / "x.txt").write_bytes(b"doc " * 32)  # 128 B: past the shard floor, refused for shape
    plan = (
        "max_epochs = 0\n[[stage]]\nuntil_frac = 1.0\n"
        f"[stage.weights]\nx = 1.0\n[sources]\nx = {json.dumps(str(d))}\n"
    )
    loaded = loads_mix_plan(plan)
    with pytest.raises(LoudError, match="shard DIRECTORY"):
        MixtureStream(loaded, 1, 4, 2, doc_masking=True)


# --- gate 5: the long-context smoke on a real mix4 shard --------------------


REAL = Path("/home/sehaxe/dormouse/data_local/mix4/reasoning.bin")


def test_long_context_window_on_a_real_mix4_shard():
    """`seq_len=4096`, `segment_len=0`: one forward+backward, counters non-zero."""
    if not REAL.is_file() or not Path(str(REAL) + ".docs.json").is_file():
        pytest.skip(f"no real mix4 shard + sidecar at {REAL}")
    stream = DocPackedStream(ByteStream(REAL, 1, 4096, check_min=False), load_doc_starts(REAL))
    x, y, mask = stream.next_batch_masked()
    assert x.shape == (1, 4096)
    assert bool(mask.any())
    torch.manual_seed(0)
    model = DormouseModel(ModelConfig(**{**TINY, "max_seq_len": 4096}))
    _, loss, _ = model(x, y, loss_mask=mask)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    counts = stream.doc_counts()
    assert counts["docmask_windows"] == 1
    assert counts["docmask_docs"] >= 1
    assert counts["docmask_scored"] > 0
    print(f"longctx counts={counts} loss={float(loss.detach()):.6f}")


# --- gate 6: refusals -------------------------------------------------------


def test_refuses_segment_len():
    with pytest.raises(LoudError, match="segment_len"):
        _validate_doc(ModelConfig(**TINY), TrainConfig(doc_masking=True, segment_len=32))


def test_refuses_byteflow():
    with pytest.raises(LoudError, match="use_byteflow"):
        _validate_doc(ModelConfig(**{**TINY, "use_byteflow": True}), TrainConfig(doc_masking=True))


def test_refuses_a_target_bearing_aux_term():
    with pytest.raises(LoudError, match="dspark_weight"):
        _validate_doc(ModelConfig(**{**TINY, "dspark_weight": 0.1}), TrainConfig(doc_masking=True))


def test_refuses_capacity_dispatch():
    """Padding rows may not move another row's expert assignment."""
    with pytest.raises(LoudError, match="moe_capacity"):
        _validate_doc(
            ModelConfig(**{**TINY, "moe_capacity": 2.0, "moe_topk": 2}),
            TrainConfig(doc_masking=True),
        )


def test_refuses_a_data_file_without_boundaries(tmp_path):
    corpus = _corpus(tmp_path / "no_sidecar.bin")
    run = _run(tmp_path, corpus, doc_masking=True)
    with pytest.raises(LoudError, match="sidecar"):
        open_streams(run)
    with pytest.raises(LoudError, match="sidecar"):
        load_doc_starts(corpus)


def test_refuses_a_stale_sidecar(tmp_path):
    corpus = _corpus(tmp_path / "stale.bin")
    (tmp_path / "stale.bin.docs.json").write_text(
        json.dumps({"version": 1, "size": 123, "starts": [0, 10]})
    )
    with pytest.raises(LoudError, match="stale"):
        load_doc_starts(corpus)


def test_train_step_runs_with_the_mask(tmp_path):
    """End-to-end: the trainer asks for the masked window and reports the counters."""
    import dataclasses

    corpus = _doc_file(tmp_path, DOCS, name="corpus.bin")
    run = _run(tmp_path, corpus, doc_masking=True)
    run = dataclasses.replace(run, model=dataclasses.replace(run.model, moe_capacity=0.0))
    torch.manual_seed(0)
    model = DormouseModel(run.model)
    stream = _doc_stream(corpus, BATCH, SEQ)
    report = train(model, run, stream=stream, log=lambda _line: None)
    assert report["doc_counts"]["docmask_windows"] > 0
    assert report["doc_counts"]["docmask_docs"] > 0
    assert report["nan_skipped"] == 0


def test_load_run_wires_the_flag(tmp_path):
    """The CLI seam sees the field, and the doc arm's own refusals are on it."""
    from dormouse.config import load_run

    run = load_run(
        "tiny",
        model_overrides={"moe_capacity": 0.0},
        train_overrides={"doc_masking": True, "seq_len": 16},
    )
    assert run.train.doc_masking is True
    with pytest.raises(LoudError, match="segment_len"):
        load_run(
            "tiny",
            model_overrides={"moe_capacity": 0.0},
            train_overrides={"doc_masking": True, "segment_len": 8, "seq_len": 16},
        )


# --- gate 7: the sidecar tool ----------------------------------------------


def test_sidecar_tool_recovers_shard_offsets(tmp_path):
    """A shard-built `.bin` is a concatenation of whole files, in any order."""
    src = tmp_path / "shards"
    src.mkdir()
    docs = [bytes([i]) * (100 + i) for i in range(1, 6)]
    for i, d in enumerate(docs):
        (src / f"s{i}.txt").write_bytes(d)
    order = [3, 0, 4, 1, 2]
    bin_path = tmp_path / "domain.bin"
    bin_path.write_bytes(b"".join(docs[i] for i in order))
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_doc_sidecar.py"
    out = subprocess.run(
        [sys.executable, str(script), "--bin", str(bin_path), "--source-dir", str(src)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    starts = load_doc_starts(bin_path)
    want = [0]
    for i in order[:-1]:
        want.append(want[-1] + len(docs[i]))
    assert starts == want


def test_sidecar_tool_refuses_a_foreign_directory(tmp_path):
    src = tmp_path / "shards"
    src.mkdir()
    (src / "s0.txt").write_bytes(b"Z" * 200)
    bin_path = tmp_path / "domain.bin"
    bin_path.write_bytes(b"Y" * 200)
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_doc_sidecar.py"
    out = subprocess.run(
        [sys.executable, str(script), "--bin", str(bin_path), "--source-dir", str(src)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode != 0
    assert "no source shard starts" in out.stderr
    assert not Path(str(bin_path) + ".docs.json").exists()
