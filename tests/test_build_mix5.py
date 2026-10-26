"""Gates for the near-duplicate + quality-band stage (`build_mix4.py` mix5 path).

Run without torch/pyarrow (the stage is the byte path only):

    uv run --with pytest --with numpy python -m pytest --noconftest \
        port/tests/test_build_mix5.py -q -s

Each test is a revert gate: turning the stage off, moving the eval carve after
the train loop, or letting the index keep text makes one of these fail.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_mix4.py"
spec = importlib.util.spec_from_file_location("build_mix4", SCRIPT)
mix = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mix)


# --- fixtures ---------------------------------------------------------------


def _text(n_words: int, seed: int, vocab: int = 600) -> bytes:
    """Prose-shaped: ~20% stopwords and lines that end in punctuation."""
    rng = random.Random(seed)
    words = [
        b"the" if i % 5 == 0 else f"w{rng.randrange(vocab):04d}".encode()
        for i in range(n_words)
    ]
    lines = [b" ".join(words[i : i + 12]) + b"." for i in range(0, len(words), 12)]
    return b"\n".join(lines) + b"\n"


def _variant(base: bytes, step: int) -> bytes:
    """Same-length words every `step`-th position: byte-aligned, Jaccard ~0.8-0.9."""
    out_lines = []
    k = 0
    for line in base.split(b"\n"):
        words = line.split()
        out = []
        for j, w in enumerate(words):
            if j and j < len(words) - 1 and k % step == 0:
                out.append(b"x%04d" % (k % 997))
            else:
                out.append(w)
            k += 1
        out_lines.append(b" ".join(out))
    return b"\n".join(out_lines)


def _shards(tmp_path: Path, docs: dict[str, bytes]) -> Path:
    shards = tmp_path / "shards"
    shards.mkdir()
    for i, (_name, body) in enumerate(docs.items()):
        (shards / f"web_{i:05d}.txt").write_bytes(body)
    return shards


def _no_shuffle(monkeypatch) -> None:
    """Keep sorted order so the eval carve picks a shard the test can name."""

    class _R:
        def __init__(self, *a, **k) -> None:
            pass

        def shuffle(self, x) -> None:
            pass

    monkeypatch.setattr(mix.random, "Random", _R)


# --- 1. the near-dup actually fires on planted duplicates -------------------


def test_planted_duplicates_drop_and_distinct_pair_survives() -> None:
    base = _text(3000, 7)
    docs = {
        "exact": base,
        "j90": _variant(base, 92),
        "j80": _variant(base, 38),
        "similar": _variant(base, 13),  # ~0.5 Jaccard: similar, not a duplicate
        "distinct": b" ".join(f"z{i:04d}".encode() for i in range(3000)) + b".",
    }
    base_set = mix.shingle_set(base)
    base_sig = mix.minhash_signature(mix.chunk_features(base))
    idx = mix.NearDupIndex()
    assert idx.add(mix.chunk_features(base), "synthetic", 0) == (None, None, 0.0)
    reasons = {}
    for name, doc in docs.items():
        feat = mix.chunk_features(doc)
        sig = mix.minhash_signature(feat)
        exact = mix.jaccard(base_set, mix.shingle_set(doc))
        est = float((base_sig == sig).mean())
        print(f"{name:9s} exact_jaccard={exact:.3f} signature_est={est:.3f}")
        reason, _rep, _j = idx.add(feat, "synthetic", 0)
        reasons[name] = reason

    assert reasons["exact"] == "near_dup"
    assert reasons["j90"] == "near_dup"
    assert reasons["j80"] == "near_dup"
    assert reasons["similar"] is None
    assert reasons["distinct"] is None
    # the exact one is dropped at J=1.0; the 90%/80% pairs are above the 0.8
    # banding target and above the 0.75 verification floor.
    assert idx.cluster_count() == 3  # base+3 duplicates, the ~0.5 similar, the distinct
    assert idx.signature_nbytes() == 6 * mix.NEAR_DUP_PERMS * 4


def test_banding_probability_at_the_threshold() -> None:
    p = 1.0 - (1.0 - mix.NEAR_DUP_THRESHOLD**mix.NEAR_DUP_ROWS) ** mix.NEAR_DUP_BANDS
    print(f"P(collide | J={mix.NEAR_DUP_THRESHOLD}) = {p:.4f}")
    assert p > 0.90


# --- 2. memory is bounded by signatures x documents -------------------------


def test_signature_array_size_is_the_stated_bound() -> None:
    idx = mix.NearDupIndex()
    n = 200
    for i in range(n):
        idx.add(mix.chunk_features(_text(60, seed=i)), "s", 0)
    assert idx.signature_nbytes() == n * mix.NEAR_DUP_PERMS * 4
    assert idx.sig.dtype.name == "uint32"
    assert idx.sig.shape[1] == mix.NEAR_DUP_PERMS
    # doubling capacity, never a per-byte buffer
    assert n <= len(idx.sig) <= 2 * max(1024, n)
    assert len(idx.ranges) == n  # (path, offset, len) only


# --- 3. eval carve first; a near-dup of eval is never trained ---------------


def test_near_dup_of_eval_is_not_in_the_train_output(tmp_path: Path, monkeypatch) -> None:
    a = _text(22_000, 11)
    a90 = _variant(a, 92)
    distinct_b = _text(22_000, 12)
    distinct_c = _text(22_000, 13)
    # sorted order; eval carve starts at len//2 == 2, so web_00002 (a) is eval.
    shards = _shards(
        tmp_path,
        {"a90": a90, "c": distinct_c, "a": a, "b": distinct_b},
    )
    _no_shuffle(monkeypatch)
    monkeypatch.setitem(mix.SHARD_DIRS, "web", shards)
    out = tmp_path / "out"

    d = mix.Domain("web", 700 << 10, out, eval_target=64 << 10, near_dup=True)
    assert d.start()
    mix.build_shards_quality(d)
    entry = d.finish()

    train = (out / "web.bin").read_bytes()
    ev = (out / "web_eval.bin").read_bytes()
    assert d.state["eval_source"] == "web_00002.txt"
    assert ev == a  # the carve is the whole shard, first
    assert a90 not in train  # the near-duplicate variant never made it
    assert distinct_b[:200] in train and distinct_c[:200] in train
    nd = entry["near_dup"]
    assert nd["eval_collisions"] >= 1
    assert entry["eval"]["exclusion"]["near_dup"]["pass"] is True
    assert entry["eval"]["exclusion"]["near_dup"]["eval_chunks_indexed"] > 0
    print(
        f"eval chunks={nd['eval_chunks']} collisions={nd['eval_collisions']} "
        f"train_bytes={entry['train_bytes']}"
    )


def test_finish_refuses_when_the_eval_index_is_empty(tmp_path: Path, monkeypatch) -> None:
    """The no-leak proof must not be vacuous: no eval seeds -> no publish."""
    shards = _shards(tmp_path, {"a": _text(22_000, 1), "b": _text(22_000, 2)})
    _no_shuffle(monkeypatch)
    monkeypatch.setitem(mix.SHARD_DIRS, "web", shards)
    d = mix.Domain("web", 300 << 10, tmp_path / "out", eval_target=64 << 10, near_dup=True)
    assert d.start()
    mix.build_shards_quality(d)
    d.state["near_dup"]["eval_chunks"] = 0
    with pytest.raises(RuntimeError, match="no eval chunks"):
        d.finish()


# --- 4. determinism: two runs, identical bytes and manifest -----------------


def test_two_runs_are_byte_identical(tmp_path: Path, monkeypatch) -> None:
    a = _text(22_000, 21)
    shards = _shards(
        tmp_path,
        {
            "a": a,
            "a90": _variant(a, 92),
            "b": _text(22_000, 22),
            "c": _text(22_000, 23),
            "d": _text(22_000, 24),
        },
    )
    _no_shuffle(monkeypatch)
    monkeypatch.setitem(mix.SHARD_DIRS, "web", shards)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"

    m1 = mix.run(
        ("web",),
        out_dir=out1,
        near_dup=True,
        quality_band=True,
        max_source_bytes=1 << 20,
    )
    m2 = mix.run(
        ("web",),
        out_dir=out2,
        near_dup=True,
        quality_band=True,
        max_source_bytes=1 << 20,
    )
    for name in ("web.bin", "web_eval.bin"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
    b1 = (out1 / "manifest.json").read_bytes()
    b2 = (out2 / "manifest.json").read_bytes()
    assert b1 == b2
    assert hashlib.sha256(b1).hexdigest() == hashlib.sha256(b2).hexdigest()
    assert m1["domains"]["web"]["train_sha256"] == m2["domains"]["web"]["train_sha256"]
    assert m1["pipeline"]["near_dup"] and m1["pipeline"]["quality_band"]
    assert m1["domains"]["web"]["max_source_bytes"] == 1 << 20
    print("manifest sha256", hashlib.sha256(b1).hexdigest())


def test_the_two_flags_are_independent(tmp_path: Path, monkeypatch) -> None:
    a = _text(22_000, 31)
    junk = _text(22_000, 32) + b" cookie cookie cookie."
    shards = _shards(tmp_path, {"a": a, "a90": _variant(a, 92), "junk": junk})
    _no_shuffle(monkeypatch)
    monkeypatch.setitem(mix.SHARD_DIRS, "web", shards)

    # quality only: the 90%-similar shard is NOT deduplicated
    q_out = tmp_path / "q"
    d = mix.Domain("web", 700 << 10, q_out, eval_target=64 << 10, quality_band=True)
    assert d.start()
    mix.build_shards_quality(d)
    e = d.finish()
    assert "near_dup" not in e
    assert _variant(a, 92)[:200] in (q_out / "web.bin").read_bytes()

    # near-dup only: the boilerplate shard is NOT quality-dropped
    n_out = tmp_path / "n"
    d = mix.Domain("web", 700 << 10, n_out, eval_target=64 << 10, near_dup=True)
    assert d.start()
    mix.build_shards_quality(d)
    e = d.finish()
    assert "quality_band" not in e
    assert junk[:200] in (n_out / "web.bin").read_bytes()


# --- 5. the quality band fires, one filter per defect -----------------------


def _q(doc: bytes, lang: str = "en"):
    return mix.quality_band(doc, mix.chunk_features(doc), lang)


def test_quality_band_drops_each_defect_for_its_own_reason() -> None:
    filler = _text(900, 3)  # passes everything on its own
    ok, reason, metrics, checked = _q(filler)
    assert ok, (reason, metrics)
    assert len(checked) == len(mix.QUALITY_BANDS)
    assert metrics["score"] == 1.0

    prose = b" ".join(
        (b"the" if i % 5 == 0 else b"w%04d" % (i % 600)) for i in range(900)
    )
    cases = {
        "words": b"too short.",
        "mean_word_len": b" ".join([b"a" * 24] * 900) + b".",
        "stopword_frac": b" ".join(f"w{i:04d}".encode() for i in range(900)) + b".",
        "line_punct_frac": (prose + b"\n") * 2,
        "list_line_frac": (b"- " + prose[:600] + b".\n") * 20,
        "repetition": b"the quick brown fox jumps. " * 120,
        "symbol_alpha": b" ".join(
            b"the" if i % 5 == 0 else f"a####{i % 600:03d}".encode() for i in range(900)
        )
        + b".",
        "uppercase_frac": b" ".join(
            b"the" if i % 3 == 0 else f"THE W{i % 600:03d} QUICK BROWN FOX JUMPS".encode()
            for i in range(400)
        )
        + b".",
        "boilerplate_hits": filler + b" cookie cookie cookie.",
    }
    for want, doc in cases.items():
        ok, reason, _m, _checked = _q(doc)
        assert not ok, want
        assert reason == want, (want, reason)

    ru = b" ".join(
        b"\xd0\xb8" if i % 5 == 0 else f"\u0441\u043b\u043e\u0432\u043e{i % 600}".encode()
        for i in range(600)
    ) + b"."
    ok, reason, metrics, _ = _q(ru, "ru")
    assert ok, (reason, metrics)


# --- 6. flags are off by default and refuse non-crawl domains ---------------


def test_flags_default_off_and_are_refused_off_the_crawl_domains(tmp_path: Path) -> None:
    d = mix.Domain("web", 1000, tmp_path)
    assert d.near_dup is False and d.quality_band is False
    d2 = mix.Domain("web", 1000, tmp_path, near_dup=True)
    assert d2._key() != d._key()
    with pytest.raises(SystemExit, match="web, ruweb"):
        mix.run(("math",), out_dir=tmp_path, near_dup=True)


def test_default_build_key_is_still_the_mix4_key(tmp_path: Path, monkeypatch) -> None:
    """The flags must not invalidate a finished mix4 domain on a rerun."""
    monkeypatch.setattr(mix, "source_fingerprint", lambda name: {"dir": "x"})
    d = mix.Domain("web", 1000, tmp_path)
    old = {
        "v": mix.FILTER_VERSION,
        "name": "web",
        "target": 1000,
        "eval_target": mix.EVAL_TARGET,
        "seed": mix.SEED,
        "source": {"dir": "x"},
    }
    assert d._key() == mix.sha1(json.dumps(old, sort_keys=True).encode())


def test_iter_chunks_is_lossless_and_line_aligned() -> None:
    raw = b"".join(b"line %d of some prose\n" % i for i in range(1000)) + b"tail"
    chunks = list(mix.iter_chunks(raw))
    assert b"".join(c for _, c in chunks) == raw
    assert all(len(c) <= mix.CHUNK_BYTES for _, c in chunks)
    for off, chunk in chunks[:-1]:
        assert chunk.endswith(b"\n")
        assert raw[off : off + len(chunk)] == chunk
    # a single line longer than the chunk budget is split, never lost
    long_line = b"x" * (mix.CHUNK_BYTES + 100) + b"\n"
    assert b"".join(c for _, c in mix.iter_chunks(long_line)) == long_line
