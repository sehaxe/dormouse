"""Data invariants (2026-11-07): document boundaries, skipped counts, flat parity.

Three claims, each checked against an independent (test-side) reference rather
than the reader that produced the data:

1. **`doc_masking=true` never scores a cross-document target.** Every packed
   row is one document (or one contiguous chunk of it); every supervised
   position's TARGET byte is inside that document and is not a document start.
2. **The skipped count is the sidecar's.** For each row, the sidecar fixes the
   document length and the cursor, so the expected number of dropped positions
   (padding + the boundary byte) is `seq - (remaining-1)` for a whole document
   and `0` for a chunk; the sum over rows must equal the reader's own
   `docmask_masked` counter and `(1-mask).sum()` on the tensors.
3. **`doc_masking=false` is the flat stream byte for byte.** The first windows
   are pinned by a sha256 recorded on clean HEAD (`e954e8b`) and reproduced
   from the file bytes by hand.

`eval_docmask=sidecar` is checked against the SAME predicate: a flat window
whose target is a document start is masked, exactly as the packed trainer
masks a document's last byte. No sidecar is not masked, and says so.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import torch

from dormouse.train.data import (
    ByteStream,
    DocPackedStream,
    MaskedByteStream,
    _mask_eval,
    load_doc_starts,
)

SEQ, BATCH = 16, 2
DOCS = [b"A" * 5, b"B" * 300, b"C" * 7, b"D" * 64, b"E" * 1]
#: `bytes(range(16)) * 256`, batch 2, seq 64, first three windows, clean HEAD
#: `e954e8b` (the `doc_masking=false` path must never move).
FLAT_HEAD = [
    ("115d76d1227d8144", "23f2e30dcc2c2d3f"),
    ("6e8206bae4e75469", "428f6957536cfc0e"),
    ("3678dcdb35789434", "3b19ad2cedf78c96"),
]


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:16]


def _doc_file(tmp_path: Path, docs: list[bytes], name: str = "d.bin") -> tuple[Path, list[int]]:
    path = tmp_path / name
    data = b"".join(docs)
    path.write_bytes(data)
    starts = [0]
    for d in docs[:-1]:
        starts.append(starts[-1] + len(d))
    Path(str(path) + ".docs.json").write_text(
        json.dumps({"version": 1, "size": len(data), "starts": starts})
    )
    return path, starts


def _pack_reference(
    data: bytes, starts: list[int], lo: int, hi: int, batch: int, seq: int, rows: int
) -> list[dict]:
    """The packing rule transcribed from the module docstring, independent code.

    One document per row, or a `seq+1` byte chunk of one document when it is
    longer; tail-padded; a whole document's last byte has no in-document target.
    """
    docs = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else hi
        a, b = max(s, lo), min(end, hi)
        if a < b:
            docs.append((a, b))
    doc_starts = [a for a, _ in docs]

    def at(pos: int) -> tuple[int, int]:
        i = bisect.bisect_right(doc_starts, pos) - 1
        assert i >= 0, pos
        return docs[i]

    pos = lo
    out = []
    for _ in range(rows):
        a, b = at(pos)
        remaining = b - pos
        x = torch.zeros(seq, dtype=torch.long)
        y = torch.zeros(seq, dtype=torch.long)
        mask = torch.zeros(seq, dtype=torch.bool)
        if remaining <= seq:
            chunk = data[pos:b]
            scored = max(remaining - 1, 0)
            x[:remaining] = torch.tensor(list(chunk), dtype=torch.long)
            if remaining > 1:
                y[: remaining - 1] = torch.tensor(list(chunk[1:]), dtype=torch.long)
                mask[: remaining - 1] = True
            nxt = b
            kind = "whole"
        else:
            chunk = data[pos : pos + seq + 1]
            x = torch.tensor(list(chunk[:seq]), dtype=torch.long)
            y = torch.tensor(list(chunk[1:]), dtype=torch.long)
            mask[:] = True
            scored = seq
            nxt = pos + seq + 1
            kind = "chunk"
        out.append(
            {
                "start": pos,
                "doc": (a, b),
                "kind": kind,
                "x": x,
                "y": y,
                "mask": mask,
                "scored": scored,
                "skipped": seq - scored,
            }
        )
        pos = lo if nxt >= hi else nxt
    return out


def _packed(tmp_path: Path, docs: list[bytes], batch: int = BATCH, seq: int = SEQ):
    path, starts = _doc_file(tmp_path, docs)
    stream = DocPackedStream(ByteStream(path, batch, seq, check_min=False), load_doc_starts(path))
    return path, starts, stream


def test_no_packed_target_crosses_a_document_boundary(tmp_path):
    path, starts, stream = _packed(tmp_path, DOCS)
    data = path.read_bytes()
    rows = stream.next_batch_masked()
    ref = _pack_reference(data, starts, 0, len(data), BATCH, SEQ, BATCH)
    for b, expected in enumerate(ref):
        x, y, mask = rows[0][b], rows[1][b], rows[2][b]
        assert torch.equal(x, expected["x"]), f"row {b} inputs"
        assert torch.equal(y, expected["y"]), f"row {b} targets"
        assert torch.equal(mask, expected["mask"]), f"row {b} mask"
        a, end = expected["doc"]
        for j in range(SEQ):
            target = expected["start"] + j + 1
            if not bool(mask[j]):
                continue
            # The target byte is inside THIS document and is not a new one.
            assert a < target < end, f"row {b} pos {j}: target {target} leaves {a}:{end}"
            assert target not in starts, f"row {b} pos {j}: target {target} is a document start"
            assert int(y[j]) == data[target], f"row {b} pos {j}: target byte"


def test_skipped_targets_equal_the_sidecar_derivation(tmp_path):
    path, starts, stream = _packed(tmp_path, DOCS)
    data = path.read_bytes()
    total_masked = 0
    rows = 7
    seen = []
    for _ in range((rows + BATCH - 1) // BATCH):
        x, y, mask = stream.next_batch_masked()
        for b in range(BATCH):
            seen.append((x[b], y[b], mask[b]))
    ref = _pack_reference(data, starts, 0, len(data), BATCH, SEQ, len(seen))
    expected_skipped = 0
    for (x, y, mask), r in zip(seen, ref, strict=True):
        assert torch.equal(x, r["x"]) and torch.equal(y, r["y"]) and torch.equal(mask, r["mask"])
        assert int((~mask).sum()) == r["skipped"]
        assert int(mask.sum()) == r["scored"]
        total_masked += r["skipped"]
    counts = stream.doc_counts()
    assert counts["docmask_masked"] == total_masked == sum(int((~m).sum()) for _, _, m in seen)
    assert counts["docmask_scored"] == sum(int(m.sum()) for _, _, m in seen)
    assert counts["docmask_doc_ends"] == sum(1 for r in ref if r["kind"] == "whole")
    # And every skipped position is padding or a document's LAST byte, never a
    # mid-document target (the `skipped == seq - scored` arithmetic above).
    expected_skipped = 0
    for r in ref:
        a, end = r["doc"]
        remaining = end - r["start"]
        expected_skipped += 0 if remaining > SEQ else max(SEQ - remaining + 1, 0)
    assert expected_skipped == total_masked


def test_eval_docmask_sidecar_uses_the_train_rule(tmp_path):
    path, starts = _doc_file(tmp_path, DOCS)
    train_stream = DocPackedStream(
        ByteStream(path, BATCH, SEQ, check_min=False), load_doc_starts(path)
    )
    data = path.read_bytes()
    ref = _pack_reference(data, starts, 0, len(data), BATCH, SEQ, BATCH)
    x, y, mask = train_stream.next_batch_masked()
    for b, r in enumerate(ref):
        assert torch.equal(mask[b], r["mask"])
        for j in range(SEQ):
            if bool(mask[b, j]):
                # The train rule, stated once: supervise a position iff its
                # TARGET is not a document start (and exists).
                assert (r["start"] + j + 1) not in starts

    ev = MaskedByteStream(ByteStream(path, BATCH, SEQ), starts)
    assert ev.doc_boundary_policy == "sidecar"
    window = BATCH * (SEQ + 1)
    for k in range(2):
        ex, ey, emask = ev.next_batch_masked()
        base = k * window
        for b in range(BATCH):
            row_base = base + b * (SEQ + 1)
            for j in range(SEQ):
                target = row_base + j + 1
                want = target < len(data) and target not in starts
                assert bool(emask[b, j]) == want, f"eval window {k} row {b} pos {j}"
                if want:
                    assert int(ey[b, j]) == data[target]


def test_no_eval_sidecar_is_flat_and_says_so(tmp_path):
    path = tmp_path / "flat.bin"
    path.write_bytes(bytes(range(16)) * 64)
    ev = ByteStream(path, BATCH, SEQ)
    got = _mask_eval(ev, path, enabled=True)
    assert got is ev and got.doc_boundary_policy == "flat (no sidecar)"
    x, y = got.next_batch()
    assert x.shape == (BATCH, SEQ) == y.shape


def test_doc_masking_off_is_the_pinned_flat_stream(tmp_path):
    """`doc_masking=false`: the bytes are clean HEAD's, window for window."""
    path = tmp_path / "corpus.bin"
    path.write_bytes(bytes(range(16)) * 256)
    stream = ByteStream(path, 2, 64)
    raw = path.read_bytes()
    cursor = 0
    for k, (want_x, want_y) in enumerate(FLAT_HEAD):
        x, y = stream.next_batch()
        assert (_sha(x), _sha(y)) == (want_x, want_y), f"window {k} moved from HEAD"
        # ...and from first principles: `[b, sq+1]` bytes -> inputs, next bytes.
        window = raw[cursor : cursor + 2 * 65]
        assert len(window) == 130
        assert torch.equal(x, torch.tensor(list(window)).view(2, 65)[:, :-1])
        assert torch.equal(y, torch.tensor(list(window)).view(2, 65)[:, 1:])
        cursor += 130
