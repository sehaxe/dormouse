"""Speculative decode's gates: the output must not change, the stats must not lie.

The load-bearing gate is (a): at `temp -> 0` both decoders are deterministic, so
spec decode must emit the SAME byte string `decode_bytes` emits — a spec decoder
that changes the output is a different model, whatever its acceptance rate. The
rest is the instrument around it: the acceptance rule's exactness at temp 1, the
counters, the stop byte, and the two hand-built drafts (one perfect, one
garbage) that separate "the draft was right" from "the target won anyway".

Everything is CPU and tiny (`conftest.TINY`, d=64) and every gate runs in the
regime where the exactness argument is unconditional: `len(prompt) + n <=
max_seq_len`, so no sliding-window truncation is in play at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_model, write_corpus
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.instrument import Counters, LoudError
from dormouse.model import specdecode
from dormouse.model.decode import decode_bytes, window_forward
from dormouse.model.specdecode import accept_or_residual, spec_decode_bytes

runner = CliRunner()

#: The SFT template's prompt shape — what `chat.py`/`serve.py` decode.
PROMPT = b"\x01 hi \n\x02 "
N = 22  # not a multiple of K: the last block is cut by the byte budget
K = 4
TEMP0 = 1e-6  # the deterministic corner (softmax is one-hot, so temp > 0 is legal)


@pytest.fixture
def model():
    """A tiny model WITH the draft head — the arm spec decode needs.

    `moe_capacity=0.0` (dropless) on purpose: capacity dropping makes the
    forward non-prefix-causal (see the causality gate below), and the
    load-bearing equivalence gate needs the regime where ONE forward over a
    block IS `spec_k` forwards over growing prefixes. It is also the form
    `configs/b1.toml` ships.
    """
    return tiny_model(dspark_weight=0.1, dspark_markov_rank=16, moe_capacity=0.0)


@pytest.fixture
def plain(model):
    """`decode_bytes` at the deterministic corner: the byte string to match."""
    return decode_bytes(model, PROMPT, N, temp=TEMP0, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def perfect_draft(monkeypatch, model, plain):
    """A draft that is RIGHT: it proposes the bytes plain decode would emit.

    Built from plain decode's own output, so "the draft is right" is a fact of
    the fixture and not a hope about the head's random weights. Its `q` is
    one-hot on the proposal, which is what a confident drafter's distribution
    looks like and passes `min(1, p/q) = 1` exactly.
    """
    gold = plain[len(PROMPT) :]

    def perfect(head, ctx, h_last, base_last, k, temp, generator):
        done = len(ctx) - len(PROMPT)
        ids = [gold[done + j] for j in range(k) if done + j < len(gold)]
        probs = [F.one_hot(torch.tensor(b), model.cfg.vocab).float() for b in ids]
        return ids, probs

    monkeypatch.setattr(specdecode, "draft_block", perfect)
    return perfect


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_spec_and_plain_are_the_same_bytes_at_temp_0(model, plain):
    """(a) THE gate: same bytes, or the whole arm is wrong.

    Also the byte budget (22 is not 4·k: the last block must be cut) and the
    stats' basic honesty (`accepted <= drafted`, the counter, the arithmetic).
    """
    assert len(PROMPT) + N <= model.cfg.max_seq_len, "run the gate inside the window"
    out, stats = spec_decode_bytes(model, PROMPT, N, spec_k=K, temp=TEMP0, generator=_gen(1))
    assert out == plain
    assert len(out) == len(PROMPT) + N  # the budget is the plain decoder's
    assert stats.emitted == N
    assert stats.accepted <= stats.drafted
    assert stats.rounds == stats.drafted / K  # every round drafts a full block
    assert stats.per_byte_forwards == stats.forwards / N
    assert "spec_over_window" not in stats.counters  # no truncation in this regime


def test_a_wrong_draft_still_emits_the_target_bytes(model, plain, monkeypatch):
    """(b) the draft is deliberately wrong: the target's bytes come out anyway.

    The proposal is always one byte off plain decode's own next byte, so every
    block is rejected at its first position by construction — and the emitted
    stream is still exactly the target's.
    """
    gold = plain[len(PROMPT) :]

    def garbage(head, ctx, h_last, base_last, k, temp, generator):
        done = len(ctx) - len(PROMPT)
        ids = [(gold[(done + j) % len(gold)] + 1) % 256 for j in range(k)]
        q = torch.full((model.cfg.vocab,), 1.0 / model.cfg.vocab)
        return ids, [q] * k

    monkeypatch.setattr(specdecode, "draft_block", garbage)
    out, stats = spec_decode_bytes(model, PROMPT, N, spec_k=K, temp=TEMP0, generator=_gen(2))
    assert out == plain
    assert stats.accepted == 0  # every proposal rejected...
    assert stats.counters.get("spec_accept", 0) == 0  # ...and the counter says so
    assert stats.drafted == K * stats.rounds
    assert stats.forwards == 2 * stats.rounds  # a rejected block carries no state


def test_a_right_draft_costs_less_than_one_forward_per_byte(model, plain, perfect_draft):
    """(b) with the draft right, the verification pays: `forwards < drafted`.

    And the emitted stream is still the target's own — a faster path to the same
    bytes, not a different one.
    """
    out, stats = spec_decode_bytes(model, PROMPT, N, spec_k=K, temp=TEMP0, generator=_gen(3))
    assert out == plain
    assert stats.accepted == stats.drafted  # nothing to reject
    assert stats.forwards < stats.drafted
    assert stats.per_byte_forwards < 1.0  # the plain decoder's rate is exactly 1.0


def test_spec_k_1_is_one_draft_and_one_verify(model, plain, perfect_draft):
    """(c) k=1 degenerates: one proposal, one verification, no free lunch.

    With the draft right, a round is 1 verify forward (the first round also pays
    the state forward), i.e. it can never beat plain decode's 1 forward/byte.
    """
    out, stats = spec_decode_bytes(model, PROMPT, N, spec_k=1, temp=TEMP0, generator=_gen(4))
    assert out == plain
    assert stats.drafted == stats.rounds == stats.emitted == N
    assert stats.forwards == stats.rounds + 1


def test_a_stop_byte_ends_the_round_like_plain_decode(model, plain, perfect_draft):
    """(d) the stop byte ends the loop exactly like `decode_bytes` — mid-block.

    `gen_bytes[2]` first occurs at index <= 2, so with k=4 the stop lands INSIDE
    the first block: the block's later bytes are accepted by the target and must
    still be dropped (that is why `accepted` can exceed `emitted`).
    """
    gen_bytes = plain[len(PROMPT) :]
    stop_byte = gen_bytes[2]
    first = gen_bytes.index(stop_byte)
    assert first <= 2 < K
    stop = frozenset({stop_byte})
    plain_stop = decode_bytes(model, PROMPT, N, temp=TEMP0, generator=_gen(0), stop=stop)
    out, stats = spec_decode_bytes(
        model, PROMPT, N, spec_k=K, temp=TEMP0, generator=_gen(5), stop=stop
    )
    assert out == plain_stop == plain[: len(PROMPT) + first + 1]
    assert out[-1] == stop_byte
    assert stats.emitted == first + 1
    assert stats.accepted >= stats.emitted  # the walk had accepted more than it emitted


def test_the_counters_are_wired_to_the_callers_counters(model, perfect_draft):
    """(e) `spec_draft`/`spec_accept` land on the caller's `Counters` (the log line)."""
    cs = Counters()
    _, stats = spec_decode_bytes(
        model, PROMPT, N, spec_k=K, temp=TEMP0, generator=_gen(6), counters=cs
    )
    assert stats.counters is cs
    assert cs["spec_draft"] == stats.drafted == N
    assert cs["spec_accept"] == stats.accepted == N
    assert stats.acceptance_rate == 1.0


def test_at_temp_1_the_loop_still_emits_exactly_the_budget(model):
    """The sampling corner: rejections and residuals, still the plain contract."""
    out, stats = spec_decode_bytes(model, PROMPT, 16, spec_k=2, temp=1.0, generator=_gen(9))
    assert len(out) == len(PROMPT) + 16
    assert stats.emitted == 16
    assert 0.0 <= stats.acceptance_rate <= 1.0


def test_the_rejection_rule_reproduces_the_target_distribution():
    """The temp>0 claim, measured: `x ~ q` + `accept_or_residual` is `~ p`.

    Greedy verify would be exact only as temp -> 0; this rule is the standard
    speculative-sampling test and is exact at ANY temp — 20k draws land within
    0.02 of p, which is the one place the claim lives in code.
    """
    p = torch.tensor([0.7, 0.2, 0.1])
    q = torch.tensor([0.2, 0.5, 0.3])
    gen = _gen(7)
    n = 20_000
    counts = torch.zeros(3)
    for _ in range(n):
        x = int(torch.multinomial(q, 1, generator=gen))
        counts[accept_or_residual(p, q, x, gen)] += 1
    got = counts / n
    assert (got - p).abs().max() < 0.02, f"emitted {got.tolist()} for p={p.tolist()}"


def test_the_hidden_is_the_tensor_the_draft_head_reads(model):
    """`window_forward(hidden=True)` hands back `norm(out)` — not a lookalike.

    `model.py` feeds `dspark_aux_loss` exactly `hn = self.norm(out)`, and the
    returned logits are `lm_head` of it; the identity pins the capture to the
    tensor the head was trained on.
    """
    ids = torch.tensor([list(PROMPT)], dtype=torch.long)
    logits, hidden = window_forward(model, ids, hidden=True)
    assert hidden.shape == (len(PROMPT), model.cfg.d_model)
    assert torch.allclose(logits, model.lm_head(hidden.to(torch.float32)), rtol=1e-6, atol=1e-6)


def test_a_capacity_limited_moe_is_not_prefix_causal():
    """The property spec decode's exactness rests on, measured both ways.

    The logits at position `i` must not move when bytes AFTER `i` are appended —
    that is what makes one forward over `ctx + block` equal to `spec_k` forwards
    over growing prefixes. `moe_capacity=0` (dropless) agrees to fp noise;
    `moe_capacity=1.0` does not, because which assignments survive capacity
    depends on the routing statistics of the whole window. Hence the dropless
    fixture above, and hence the decoder counts this case instead of assuming
    it away.
    """
    torch.manual_seed(0)
    short = torch.tensor([list(PROMPT)], dtype=torch.long)
    longer = torch.tensor([list(PROMPT) + [7, 7, 7, 7]], dtype=torch.long)
    for cap, limit in ((0.0, 1e-5), (1.0, 1e-3)):
        model = tiny_model(dspark_weight=0.1, dspark_markov_rank=16, moe_capacity=cap)
        with torch.no_grad():
            a, _ = window_forward(model, short)
            b, _ = window_forward(model, longer)
        drift = float((a[-1] - b[len(PROMPT) - 1]).abs().max())
        assert (drift < limit) == (cap == 0.0), f"moe_capacity={cap}: drift {drift:.3e}"


def test_a_dropping_moe_is_counted_not_silent(model):
    """(e) the one deviation spec decode cannot see from inside is on the log line."""
    dropping = tiny_model(dspark_weight=0.1, dspark_markov_rank=16, moe_capacity=1.0)
    cs = Counters()
    spec_decode_bytes(dropping, PROMPT, 4, spec_k=2, temp=TEMP0, generator=_gen(8), counters=cs)
    assert cs["spec_non_causal"] == 1
    cs = Counters()
    spec_decode_bytes(model, PROMPT, 4, spec_k=2, temp=TEMP0, generator=_gen(8), counters=cs)
    assert "spec_non_causal" not in cs


def test_the_refusals_are_loud():
    """Every way to ask for spec decode without the arm is a LOUD stop."""
    headless = tiny_model()  # dspark_weight = 0.0: no draft head built
    with pytest.raises(LoudError, match="draft head"):
        spec_decode_bytes(headless, PROMPT, 4)
    model = tiny_model(dspark_weight=0.1, dspark_markov_rank=16)
    with pytest.raises(LoudError, match="spec_k"):
        spec_decode_bytes(model, PROMPT, 4, spec_k=0)
    with pytest.raises(LoudError, match="temperature"):
        spec_decode_bytes(model, PROMPT, 4, temp=0.0)
    with pytest.raises(LoudError, match="prompt"):
        spec_decode_bytes(model, b"", 4)


# ---------------------------------------------------------------------------
# the CLI wiring: `--spec-k` on `dormouse decode`
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec_ckpt(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """A checkpoint WITH the draft head, written by the real `train` command.

    The wiring gate needs the head on disk, and the head is built only when
    `dspark_weight > 0` — every preset ships 0.0, so the arm is turned on
    explicitly here (a fixture that inherited the preset default would test
    nothing, which is the burn lesson in `.bulba/memory.md`). Dropless on
    purpose: the causal form (see the causality gate above).
    """
    d = tmp_path_factory.mktemp("spec_ck")
    corpus = write_corpus(d / "corpus.bin", n_bytes=4096)
    preset = d / "tiny.toml"
    preset.write_text(
        "\n".join(
            [
                "d_model = 64",
                "n_heads = 4",
                "head_dim = 16",
                "d_ffn = 128",
                "max_iter = 2",
                "rank = 8",
                "shared_rank = 16",
                "moe_topk = 2",
                "moe_capacity = 0.0",
                "engram_orders = [2, 3]",
                "engram_rows = 64",
                "engram_dim = 8",
                "max_seq_len = 32",
            ]
        )
    )
    r = runner.invoke(
        app,
        [
            "train",
            str(preset),
            "--data",
            str(corpus),
            "--set",
            f"out_dir={d / 'ck'}",
            "--set",
            "ckpt_name=spec",
            "--set",
            "batch=2",
            "--set",
            "seq_len=32",
            "--set",
            "steps=4",
            "--set",
            "log_every=2",
            "--set",
            "ckpt_every=4",
            "--set",
            "eval=false",
            "--set",
            "dspark_weight=0.1",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    return d / "ck", "spec"


def _decode(spec_k: str, ckpt_dir: tuple[Path, str]) -> str:
    ck, name = ckpt_dir
    args = [
        "decode",
        "--out-dir",
        str(ck),
        "--ckpt-name",
        name,
        "--prompt",
        "abcdefgh",
        "--n",
        "8",
        "--temp",
        "0.7",
        "--device",
        "cpu",
    ]
    if spec_k:
        args += ["--spec-k", spec_k]
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output
    return r.output


def test_the_decode_command_carries_spec_k(spec_ckpt):
    """(e) the flag reaches the decoder and the stats reach the user's terminal."""
    out = _decode("4", spec_ckpt)
    assert "spec k=4" in out and "accept" in out and "forwards/byte" in out
    assert "spec_accept" in out  # the counters line, not just the rate
    assert "spec_non_causal" not in out  # this checkpoint is dropless/causal


def test_the_decode_command_is_plain_by_default(spec_ckpt):
    """Default OFF is today's behavior: no draft, no counters, no spec line."""
    out = _decode("", spec_ckpt)
    assert "spec k=" not in out and "spec_draft" not in out
