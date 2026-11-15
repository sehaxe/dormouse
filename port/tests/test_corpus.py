"""The corpus builder's gates: filters, the hash split, the mix, and a real
end-to-end run on a toy corpus.

CPU only, no network: everything the build touches is redirected into `tmp_path`
by `sandbox`, and the tokenizer is trained in the test on the same toy text.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "port" / "scripts"
for p in (_SCRIPTS, _ROOT / "port" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import corpus  # noqa: E402

#: Lexically varied on purpose: `word_repeat` fires below a 0.3 unique-word
#: ratio, so a fixture that repeats one sentence would be dropped as boilerplate
#: and every filter gate below would pass for the wrong reason.
SYL_EN = "ka lo mi tren val gor sten bel dor far".split()
SYL_RU = "ка ло ми трен вал гор стен бел дор фар".split()
#: 400 distinct tokens each, so a 25-sentence sample keeps a unique-word ratio
#: well above the 0.3 boilerplate floor.
WORDS_EN = [a + b for a in SYL_EN for b in SYL_EN][:400]
WORDS_RU = [a + b for a in SYL_RU for b in SYL_RU][:400]


def _prose(words: list[str], seed: int, sentences: int, stops: tuple[str, ...]) -> str:
    rng = random.Random(seed)
    out = []
    for _ in range(sentences):
        ws = [rng.choice(words) for _ in range(11)]
        for stop in stops:
            ws.insert(rng.randrange(len(ws)), stop)
        out.append(" ".join(ws).capitalize() + ".")
    return " ".join(out)


PROSE = _prose(WORDS_EN, 1, 25, ("the", "of", "and"))
PROSE_RU = _prose(WORDS_RU, 2, 25, ("и", "в", "не"))

# ------------------------------------------------------------------ filters


def test_short_doc_is_dropped():
    assert corpus.keep_text(b"hello", "en") == "short"
    assert corpus.keep_text(PROSE.encode(), "en") is None


def test_english_needs_letters_and_a_stopword():
    assert corpus.keep_text(b"0123456789 " * 40, "en") == "low_letters"
    # long enough, latin, letters -- but no English function word anywhere
    no_stops = "Pascal Lisp Haskell Scala Kotlin Rust Swift Zig Nim" * 8
    assert corpus.keep_text(no_stops.encode(), "en") == "lang_no_en_stopword"


def test_russian_is_kept_and_english_is_not_mistaken_for_it():
    assert corpus.keep_text(PROSE_RU.encode(), "ru") is None
    assert corpus.keep_text(PROSE.encode(), "ru") == "lang_not_ru"
    # the English source refuses Cyrillic text: no English edu classifier was
    # applied to RU, so the two must not bleed into each other
    assert corpus.keep_text(PROSE_RU.encode(), "en") == "lang_not_latin"


def test_boilerplate_and_repetition_are_dropped():
    nav = "Home\n" * 40 + "The article body is here, and it has words.\n"
    assert corpus.keep_text(nav.encode(), "en") == "line_repeat"
    slop = "".join(f"the {i} line\n" for i in range(40))  # distinct, but all short
    assert corpus.keep_text(slop.encode(), "en") == "short_lines"
    loop = "the same three words " * 60
    assert corpus.keep_text(loop.encode(), "sci") == "word_repeat"


def test_code_source_keeps_code_that_english_rules_would_reject():
    code = "public class Student {\n\tprivate String name;\n" + "".join(
        f"\tvoid f{i}(int x) {{ y(x); }}\n" for i in range(30)
    )
    assert corpus.keep_text(code.encode(), "code") is None
    assert corpus.keep_text(code.encode(), "en") in ("lang_no_en_stopword", "short_lines")
    repeated = "}\n" * 120  # bracket soup, still long enough to reach the line check
    assert corpus.keep_text(repeated.encode(), "code") == "line_repeat"


def test_score_filter_never_removes_more_than_half():
    web = corpus.BY_NAME["fwedu"]
    assert corpus.score_keep(3, web, dropped=0, seen=10)
    assert not corpus.score_keep(2, web, dropped=0, seen=10)
    # 60 % of the pool already dropped -> the cap forces the rest to be kept
    assert corpus.score_keep(2, web, dropped=6, seen=10)
    # a source with no score column is not filtered at all
    assert corpus.score_keep(None, corpus.BY_NAME["ru"], dropped=99, seen=100)


# ------------------------------------------------------------- dedup / split


def test_holdout_predicate_is_a_pure_function_of_the_hash():
    docs = [f"doc {i}".encode() for i in range(4000)]
    hs = [corpus.digest(d) for d in docs]
    assert hs == [corpus.digest(d) for d in docs]  # stable across calls
    share = sum(corpus.is_holdout(h) for h in hs) / len(hs)
    assert 0.5 / corpus.HOLDOUT_MOD < share < 1.5 / corpus.HOLDOUT_MOD
    # one hash, one bucket: a doc can never land on both sides
    assert all(corpus.is_holdout(h) == (h % corpus.HOLDOUT_MOD == 0) for h in hs)


def test_minhash_bands_catch_a_near_duplicate_and_ignore_an_unrelated_doc():
    a = PROSE.replace(" ", " ", 1).encode()
    b = PROSE.encode().replace(b" the ", b" a ", 1)  # one word in ~330
    c = ("Разговор о погоде и о том, как хорошо гулять по набережной вечером. " * 12).encode()

    bands_a = set(corpus.band_keys(corpus.sketch(a)))
    bands_b = set(corpus.band_keys(corpus.sketch(b)))
    bands_c = set(corpus.band_keys(corpus.sketch(c)))
    assert bands_a & bands_b, "a one-word edit must collide (it is a near-dup)"
    assert not bands_a & bands_c, "unrelated text must not collide"


# ---------------------------------------------------------------- doc streams


def test_text_docs_merge_blocks_and_resume_deterministically(tmp_path):
    path = tmp_path / "t.bin"
    body = b"one\n\ntwo\n\n" + PROSE.encode() + b"\n\n" + PROSE_RU.encode() + b"\n\n"
    path.write_bytes(body * 3)

    docs = list(corpus.docs_from_text(path, 0))
    assert all(len(d) >= corpus.MIN_DOC for d, *_ in docs)
    assert b"".join(d for d, *_ in docs).count(b"\n\n") > 0
    offs = [o for *_, o in docs]
    assert offs == sorted(offs) and offs[-1] == path.stat().st_size

    # resuming from the second offset must yield exactly the same tail
    tail = list(corpus.docs_from_text(path, offs[1]))
    assert [d for d, *_ in tail] == [d for d, *_ in docs[2:]]


def test_parquet_docs_carry_score_and_licence(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "p.parquet"
    table = pa.table(
        {
            "text": [PROSE, PROSE * 2, PROSE_RU, PROSE],
            "int_score": [5, 2, 4, 4],
            "license": ["mit", "apache-2.0", "mit", "gpl-3.0"],
        }
    )
    pq.write_table(table, path)

    code = corpus.Source(
        "x",
        "d",
        "p",
        "text",
        "MIT",
        "en",
        1.0,
        score_min=3,
        licence_col="license",
        licence_ok=("mit", "apache-2.0"),
    )
    got = list(corpus.docs_from_parquet(path, code))
    assert [row[1] for row in got] == [5, 2, 4, 4]
    assert [row[2] for row in got] == ["mit", "apache-2.0", "mit", "gpl-3.0"]
    # the licence gate is what removes row 3, never an assumption about the shard
    kept = [row[2] in code.licence_ok for row in got]
    assert kept == [True, True, True, False]
    assert corpus.score_keep(got[1][1], code, 0, 10) is False


# ---------------------------------------------------------------- the mix


def test_interleave_hits_the_weights_and_reuses_no_chunk():
    shares = {"a": 0.75, "b": 0.25}
    avail = {"a": corpus.CHUNK * 8, "b": corpus.CHUNK * 8}
    pick = corpus.interleave(shares, avail, corpus.CHUNK * 8)
    count = {"a": 0, "b": 0}
    for name, _, _ in pick:
        count[name] += 1
    assert count["a"] == 6 and count["b"] == 2
    assert len({(n, c) for n, c, _ in pick}) == len(pick), "a chunk was copied twice"
    assert pick == corpus.interleave(shares, avail, corpus.CHUNK * 8), "not deterministic"
    # a pool smaller than one chunk is still used (its last chunk is short)
    short = {"a": corpus.CHUNK * 4, "b": corpus.CHUNK // 2}
    got = corpus.interleave(shares, short, corpus.CHUNK * 4)
    assert {n for n, _, _ in got} == {"a", "b"}
    assert sum(t for n, _, t in got if n == "b") == corpus.CHUNK // 2


def test_interleaved_output_is_the_weighted_subset_of_the_pools(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(np.full(corpus.CHUNK * 4, 11, np.uint16).tobytes())
    b.write_bytes(np.full(corpus.CHUNK * 4, 22, np.uint16).tobytes())
    out = tmp_path / "mix.bin"
    counts = corpus._write_interleaved(
        out,
        {"a": (a, corpus.CHUNK * 4), "b": (b, corpus.CHUNK * 4)},
        {"a": 0.5, "b": 0.5},
        corpus.CHUNK * 4,
    )
    assert counts == {"a": corpus.CHUNK * 2, "b": corpus.CHUNK * 2}
    got = np.fromfile(out, dtype=np.uint16)
    assert got.size == corpus.CHUNK * 4
    assert (got == 11).sum() == corpus.CHUNK * 2 and (got == 22).sum() == corpus.CHUNK * 2
    # interleaved, not concatenated: the two pools alternate
    assert got[0] != got[corpus.CHUNK]


# ------------------------------------------------------------ end to end


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Point the module's paths at tmp_path and give it a one-source table."""
    for name, value in (
        ("PRE", tmp_path / "pre"),
        ("TEXT", tmp_path / "pre" / "text"),
        ("RAW", tmp_path / "raw"),
        ("HASHES", tmp_path / "pre" / "hashes"),
        ("STATE", tmp_path / "pre" / "state.json"),
        ("SOURCES_JSON", tmp_path / "pre" / "sources.json"),
        ("STATS", tmp_path / "pre" / "stats.json"),
    ):
        monkeypatch.setattr(corpus, name, value)
    (tmp_path / "pre").mkdir(parents=True)
    (tmp_path / "pre" / "text").mkdir()
    src = corpus.Source("toy", "toy/ds", "toy", "text", "MIT", "en", 0.001, hub=False)
    monkeypatch.setattr(corpus, "SOURCES", (src,))
    monkeypatch.setattr(corpus, "BY_NAME", {"toy": src})
    monkeypatch.setattr(corpus, "BUCKETS", (("all", 1.0, ("toy",)),))
    monkeypatch.setattr(corpus, "HOLDOUT_MOD", 2)  # a toy corpus needs a 1-in-2 split
    monkeypatch.setattr(corpus, "MIN_HOLDOUT_MB", 0)
    # the near-dup gate has its own test; here it would only eat synthetic prose
    monkeypatch.setattr(corpus, "band_keys", lambda sig: [])

    # a tokenizer small enough to train inside the test, on the same text
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(
        [PROSE, PROSE_RU, PROSE * 3],
        trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tok_path = tmp_path / "pre" / "tokenizer.json"
    tok.save(str(tok_path))
    return {"dir": tmp_path, "tokenizer": tok_path, "tok": tok}


def _build(sandbox, **kw):
    args = argparse.Namespace(
        tokenizer=str(sandbox["tokenizer"]), batch=8, only=None, max_shards=0, **kw
    )
    return corpus.build(args)


def _toy_text(n: int = 40) -> bytes:
    """`n` distinct docs plus 3 exact duplicates -- enough kept docs that a
    1-in-2 hold-out split leaves both streams non-empty."""
    docs = [_prose(WORDS_EN, 100 + i, 6, ("the", "of", "and")) for i in range(n)]
    body = "\n\n".join(docs) + "\n\n" + "\n\n".join(docs[:3])
    return (body + "\n\n").encode()


def test_the_target_stops_a_pool_and_a_rerun_does_not_redo_it(sandbox, monkeypatch):
    """The budget is a SHARE, so it must stop inside the stream, not after it.

    `_toy_text` is ~60 KB; a 20 KB target must stop it early, and the state that
    made it stop must survive a restart — a rewind to the pool's baseline would
    redo the same work forever and never reach the target.
    """
    import dataclasses

    text = _toy_text(200)
    (sandbox["dir"] / "pre" / "text" / "toy.bin").write_bytes(text)
    src = dataclasses.replace(corpus.SOURCES[0], target_gb=20_000 / 1e9)
    monkeypatch.setattr(corpus, "SOURCES", (src,))
    monkeypatch.setattr(corpus, "BY_NAME", {"toy": src})

    assert _build(sandbox) == 0
    first = corpus.Tally.load(json.loads(corpus.STATE.read_text())["sources"]["toy"])
    assert 0 < first.kept_bytes, "the pool kept nothing"
    assert first.kept_bytes < len(text), "the budget did not bind: the whole file was read"
    assert first.inflight is None or not str(first.inflight.get("what", "")).startswith("shard")
    assert (corpus.PRE / "toy.bin").stat().st_size == first.tokens * 2

    assert _build(sandbox) == 0
    again = corpus.Tally.load(json.loads(corpus.STATE.read_text())["sources"]["toy"])
    assert again.kept_bytes == first.kept_bytes
    assert again.tokens == first.tokens


def test_end_to_end_build_then_assemble_then_verify(sandbox):
    (sandbox["dir"] / "pre" / "text" / "toy.bin").write_bytes(_toy_text())

    assert _build(sandbox) == 0
    state = json.loads(corpus.STATE.read_text())
    toy = corpus.Tally.load(state["sources"]["toy"])
    assert toy.docs > toy.kept_docs > 0
    # 40 distinct docs + 3 exact repeats: the repeats are the only loss
    assert toy.kept_docs == 40
    assert toy.drop.get("dup", 0) == 3
    assert toy.kept_bytes < toy.in_bytes

    stream = np.fromfile(corpus.PRE / "toy.bin", dtype=np.uint16)
    assert stream.size == toy.tokens > 0
    assert len(sandbox["tok"].decode([int(t) for t in stream[:200]])) > 100

    args = argparse.Namespace(target_gb=0.0)
    assert corpus.assemble(args) == 0
    mix = json.loads((corpus.PRE / "mix.bin.json").read_text())
    assert mix["dtype"] == "uint16" and mix["tokens"] > 0
    assert (corpus.PRE / "mix.bin").stat().st_size == mix["tokens"] * 2
    # one pool at weight 1.0 IS that pool's stream, chunk for chunk
    n = mix["tokens"]
    assert np.array_equal(np.fromfile(corpus.PRE / "mix.bin", dtype=np.uint16)[:n], stream[:n])
    assert (corpus.PRE / "toy.bin.json").read_text()
    assert corpus.verify() == 0


def test_verify_refuses_a_holdout_doc_that_also_sits_in_train(sandbox):
    (sandbox["dir"] / "pre" / "text" / "toy.bin").write_bytes(_toy_text())
    _build(sandbox)
    corpus.assemble(argparse.Namespace(target_gb=0.0))

    if not (corpus.HASHES / "toy.holdout.u64").exists():
        pytest.skip("the toy corpus drew no hold-out doc")

    held = np.fromfile(corpus.HASHES / "toy.holdout.u64", dtype=np.uint64)
    with (corpus.HASHES / "toy.train.u64").open("ab") as fh:
        fh.write(held[0].tobytes() * 8)
    assert corpus.verify() == 1
    assert corpus.verify() == 1  # still refuses: the leak is in the record, not the output


def test_cli_dispatch_reaches_every_subcommand(monkeypatch):
    """`main` is what the runner calls; a dict-of-functions dispatch once
    dropped the subcommand's Namespace, and a real corpus build starts if this
    test forgets to stub the command out -- so everything is stubbed."""
    seen: list[str] = []
    for name in ("build", "assemble"):
        monkeypatch.setattr(corpus, name, lambda args, n=name: seen.append(n) or 0, raising=False)
    for name in ("plan", "layout", "print_stats", "verify"):
        monkeypatch.setattr(corpus, name, lambda n=name: seen.append(n) or 0, raising=False)
    for cmd in ("build", "assemble", "stats", "verify", "plan", "layout"):
        assert corpus.main([cmd]) == 0
    assert seen == ["build", "assemble", "print_stats", "verify", "plan", "layout"]
    with pytest.raises(SystemExit):
        corpus.main([])


def test_rewind_keeps_a_committed_text_window_but_redoes_a_shard():
    zero = {k: 0 for k in corpus.Tally.RESUME}
    t = corpus.Tally(in_bytes=100, tokens=10, text_offset=500, kept_bytes=50)
    t.inflight = {"what": "text:fwedu.bin@123", "base": dict(zero), "drop": {}, "shards": {}}
    t.rewind()
    # the files were truncated to these counts, so the offset must stay with them
    assert (t.tokens, t.text_offset, t.kept_bytes) == (10, 500, 50)

    t = corpus.Tally(in_bytes=100, tokens=10, text_offset=500, kept_bytes=50,
                     drop={"dup": 3})
    t.inflight = {"what": "shard:000_00002.parquet", "base": dict(zero),
                  "drop": {"dup": 0}, "shards": {}}
    t.rewind()
    assert (t.tokens, t.text_offset, t.kept_bytes) == (0, 0, 0)
    assert t.drop == {"dup": 0}, "the shard's drops are rolled back with it"


def test_resume_after_a_killed_shard_repeats_no_tokens(sandbox):
    (sandbox["dir"] / "pre" / "text" / "toy.bin").write_bytes(_toy_text())
    _build(sandbox)
    after_first = corpus.Tally.load(json.loads(corpus.STATE.read_text())["sources"]["toy"])
    first = np.fromfile(corpus.PRE / "toy.bin", dtype=np.uint16)

    # pretend the run died mid-window: token + hash files hold bytes the state
    # never committed. The restart must truncate them and not emit a double.
    with (corpus.PRE / "toy.bin").open("ab") as fh:
        fh.write(np.zeros(64, np.uint16).tobytes())
    with (corpus.HASHES / "toy.train.u64").open("ab") as fh:
        fh.write(b"\x01" * 8)
    _build(sandbox)
    second = np.fromfile(corpus.PRE / "toy.bin", dtype=np.uint16)
    assert np.array_equal(first, second), "resume duplicated or lost tokens"
    again = corpus.Tally.load(json.loads(corpus.STATE.read_text())["sources"]["toy"])
    assert again.tokens == after_first.tokens and again.kept_bytes == after_first.kept_bytes
