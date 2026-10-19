"""The SFT minimum: the corpus, the markers, the mask, the masked CE, the loop, the CLI.

Every gate here is CPU and tiny: what has to be true of SFT is the SEMANTICS
(which bytes the loss counts, that a window is one example, that a NaN step
touches no weight), and none of it needs a GPU.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.engram import enable_host_tables, host_engrams
from dormouse.train.data import ByteStream
from dormouse.train.sft import (
    ASSIST,
    END,
    GENERIC_QA,
    PAD,
    USER,
    build_sft_corpus,
    render,
    sft_loss,
    sft_mask,
    sft_train,
)

runner = CliRunner()
SEQ, BATCH = 64, 2


def tiny_model_cfg(**over) -> ModelConfig:
    base = dict(
        d_model=64,
        n_heads=4,
        head_dim=16,
        d_ffn=128,
        vocab=256,
        max_seq_len=SEQ,
        max_iter=2,
        rank=8,
        n_experts=4,
        moe_topk=2,
        engram_rows=64,
        engram_dim=8,
        bf16=False,
        shared_rank=16,
    )
    base.update(over)
    return ModelConfig(**base)


def sft_run(tmp_path: Path, *, corpus: Path | None = None, **train_over) -> RunConfig:
    data = corpus if corpus is not None else tmp_path / "sft.bin"
    if corpus is None:
        build_sft_corpus(data, seed=1, seq_len=SEQ, n_tasks=32)
    cfg = TrainConfig(
        steps=30,
        batch=BATCH,
        seq_len=SEQ,
        lr=0.02,
        log_every=1,
        ckpt_every=1000,
        ckpt_name="s",
        out_dir=str(tmp_path / "ck"),
        data=str(data),
    )
    return RunConfig(
        source="sft-tiny", model=tiny_model_cfg(), train=dataclasses.replace(cfg, **train_over)
    )


def window(instr: str = "Hi", ans: str = "Hello", t: int = SEQ) -> tuple[bytes, torch.Tensor]:
    """A hand-built single-example window: the record and the model's `x` input."""
    raw = render(instr, ans)
    assert len(raw) <= t
    rec = raw + bytes([PAD]) * (t + 1 - len(raw))
    return rec, torch.tensor(list(rec[:-1]), dtype=torch.long).view(1, t)


def _losses(lines: list[str]) -> list[float]:
    return [float(ln.split("loss=")[1].split()[0]) for ln in lines if ln.startswith("sft step=")]


# --------------------------------------------------------------------------- corpus


def test_corpus_is_deterministic_and_seed_dependent(tmp_path: Path):
    """Same seed = same bytes, in any process; a different seed is a different corpus."""
    a, b, c = tmp_path / "a.bin", tmp_path / "b.bin", tmp_path / "c.bin"
    ra = build_sft_corpus(a, 7, seq_len=SEQ, n_tasks=4)
    rb = build_sft_corpus(b, 7, seq_len=SEQ, n_tasks=4)
    build_sft_corpus(c, 8, seq_len=SEQ, n_tasks=4)
    assert a.read_bytes() == b.read_bytes()
    assert (ra["records"], ra["dropped"]) == (rb["records"], rb["dropped"])
    assert a.read_bytes() != c.read_bytes()


def test_markers_never_collide_with_the_content(tmp_path: Path):
    """Every record: one user marker at 0, one assistant marker, one end marker, pad tail."""
    path = tmp_path / "s.bin"
    rep = build_sft_corpus(path, 3, seq_len=128, n_tasks=8)
    raw = path.read_bytes()
    assert len(raw) == rep["bytes"] and len(raw) % rep["record_bytes"] == 0
    for i in range(rep["records"]):
        rec = raw[i * rep["record_bytes"] : (i + 1) * rep["record_bytes"]]
        assert rec.count(bytes([USER])) == 1 and rec.count(bytes([ASSIST])) == 1
        assert rec.count(bytes([END])) == 1
        assert rec.index(USER) == 0 < rec.index(ASSIST) < rec.index(END)
        assert set(rec[rec.index(END) + 1 :]) <= {PAD}, "only padding may follow the end marker"
    # The assert that licenses all of the above: a marker in the text is refused.
    with pytest.raises(AssertionError):
        render("boom\x02", "no")
    with pytest.raises(AssertionError):
        render("ok", "no\x00")


def test_the_corpus_carries_the_seeded_tasks_and_the_generic_pairs(tmp_path: Path):
    """The sources are REUSED (post/tasks.py), not re-invented: both classes show up."""
    path = tmp_path / "s.bin"
    rep = build_sft_corpus(path, 1, seq_len=256, n_tasks=4)
    text = path.read_bytes()
    assert rep["records"] >= len(GENERIC_QA)
    assert b"def add(" in text or b"def square_sum(" in text  # a code task's reference
    assert b"answer: " in text  # the math prompt's own format, taught in the answer
    assert "Привет!".encode() in text  # the RU half of the generic set


def test_every_window_is_a_single_example(tmp_path: Path):
    """The stream's own windows pass the single-example gate, and never leak forward."""
    run = sft_run(tmp_path)
    stream = ByteStream(run.train.data, BATCH, SEQ)
    for _ in range(10):
        x, _y = stream.next_batch()
        mask = sft_mask(x)  # LOUD on a window with 0 or 2 assistant markers
        assert bool(mask.any())
        # The last target predicts the NEXT record's first byte: never supervised.
        assert not bool(mask[:, -1].any())


# --------------------------------------------------------------------------- the mask


def test_mask_matches_the_known_marker_positions():
    rec, x = window("Hi", "Hello")
    a, e = rec.index(ASSIST), rec.index(END)
    expected = torch.tensor([[a - 1 <= j <= e - 1 for j in range(SEQ)]])
    mask = sft_mask(x)
    assert torch.equal(mask, expected)
    # The `0x02` and `0x03` bytes themselves ARE supervised (the model must learn
    # to emit them); the count is the whole answer region.
    assert int(mask.sum()) == e - a + 1
    assert not bool(mask[0, : a - 1].any()), "no prompt byte may be supervised"


class _PosWise(nn.Module):
    """`logits[j]` depends on `x[j]` ALONE — no context mixing.

    Under a causal attention stack a prompt-byte flip DOES move the answer
    positions' logits, so this fixture is what isolates the mask itself; the
    real model's causal analogue is the pad-byte test below.
    """

    def __init__(self, vocab: int = 256) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.table = nn.Parameter(torch.randn(vocab, vocab, generator=g) * 0.5)

    def forward(self, x, targets=None, counters=None):
        return self.table[x.reshape(-1)], None, Counters()


def test_masked_loss_ignores_a_prompt_byte():
    rec, x = window("Hi", "Hello")
    model = _PosWise()
    l0 = sft_loss(model(x)[0], x.roll(-1, dims=1), sft_mask(x))
    flipped = x.clone()
    flipped[0, 2] = ord("X")  # inside "Hi", before the assistant marker
    assert 2 < rec.index(ASSIST)
    assert torch.equal(l0, sft_loss(model(flipped)[0], flipped.roll(-1, dims=1), sft_mask(flipped)))


def test_masked_loss_moves_when_an_answer_byte_changes():
    rec, x = window("Hi", "Hello")
    model = _PosWise()
    l0 = sft_loss(model(x)[0], x.roll(-1, dims=1), sft_mask(x))
    flipped = x.clone()
    flipped[0, rec.index(ASSIST) + 2] = ord("Z")  # inside the answer
    assert not torch.equal(
        l0, sft_loss(model(flipped)[0], flipped.roll(-1, dims=1), sft_mask(flipped))
    )


def test_a_pad_byte_outside_the_mask_cannot_move_the_loss():
    """The real model: causal attention means a later byte cannot move an earlier CE."""
    torch.manual_seed(0)  # the fixture's weights must not depend on suite order
    rec, x = window("Hi", "Hello")
    y = x.roll(-1, dims=1)
    mask = sft_mask(x)
    model = DormouseModel(tiny_model_cfg())
    l0 = sft_loss(model(x)[0], y, mask)
    corrupted = x.clone()
    corrupted[0, rec.index(END) + 1] = ord("A")  # padding, outside the mask
    l1 = sft_loss(model(corrupted)[0], y, mask)
    # SEMANTIC equality, not bitwise: the claim is "a pad byte cannot move the
    # loss", and a causal forward is deterministic per weight set — but the two
    # calls may land on different autotuned kernels, so the bitwise form was a
    # suite-order-dependent flake (found 2026-10-17).
    assert torch.allclose(l0, l1, rtol=0, atol=1e-6), (float(l0), float(l1))


# --------------------------------------------------------------------------- the loop


def test_sft_train_runs_thirty_steps_and_the_loss_falls(tmp_path: Path):
    run = sft_run(tmp_path)
    lines: list[str] = []
    report = sft_train(DormouseModel(run.model), run, log=lines.append)
    losses = _losses(lines)
    assert len(losses) == 30 and all(math.isfinite(v) for v in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5, losses
    assert report["nan_skipped"] == 0
    assert report["loss"] == pytest.approx(losses[-1], abs=1e-4)
    # The counter line is the run's own count of SUPERVISED bytes, per window.
    assert report["supervised_bytes"] == sum(
        int(ln.split("sft_bytes=")[1]) for ln in lines if ln.startswith("sft step=")
    )
    assert report["counters"]["iter"] == 30 * run.model.max_iter


class _NaNSftModel(nn.Module):
    """The SFT firewall's fixture: the logits are never finite, so the CE is NaN."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x, targets=None, counters=None):
        logits, _loss, counters = self.inner(x, targets, counters)
        return logits * float("nan"), None, counters


def test_sft_nan_step_is_skipped_and_counted_not_learned(tmp_path: Path):
    run = sft_run(tmp_path, steps=1)
    model = _NaNSftModel(DormouseModel(run.model))
    before = [p.detach().clone() for p in model.parameters()]
    report = sft_train(model, run, log=lambda *_: None)
    assert report["nan_skipped"] == 1 and report["counters"]["nan_skipped"] == 1
    for p, b in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, b), "a non-finite SFT step must not touch a weight"


def test_sft_two_all_nan_windows_stop_the_run(tmp_path: Path):
    run = sft_run(tmp_path, steps=4, log_every=1)
    with pytest.raises(LoudError) as e:
        sft_train(_NaNSftModel(DormouseModel(run.model)), run, log=lambda *_: None)
    assert "entirely non-finite" in str(e.value)


def test_sft_refuses_a_window_that_spans_two_examples(tmp_path: Path):
    """The single-example guarantee, checked per window: 10-byte records, seq_len 32."""
    path = tmp_path / "packed.bin"
    path.write_bytes((render("a", "b") + bytes([PAD]) * 4) * 40)
    run = sft_run(tmp_path, corpus=path, seq_len=32, steps=1)
    with pytest.raises(LoudError) as e:
        sft_train(DormouseModel(tiny_model_cfg(max_seq_len=32)), run, log=lambda *_: None)
    assert "spans two examples" in str(e.value) and "build-sft" in str(e.value)


def test_sft_engram_ram_trains_the_host_tables_and_saves_their_sidecar(tmp_path: Path):
    """`engram_ram=true` is the host path, not a refusal (it was a LOUD refusal
    until the `--init-from` loader landed: tonight's pretrain writes such a
    checkpoint, and a handoff from it is impossible while the loop cannot step
    host tables). The tables leave VRAM, the loop STEPS them once per training
    step (COUNTED `engram_host_updates`/`..._rows`), and `save_ckpt` writes the
    `.ngram` sibling this run's own resume needs."""
    run = sft_run(tmp_path, steps=3, log_every=1, ckpt_every=3, engram_ram=True)
    model = DormouseModel(run.model)
    before = [
        p.detach().clone() for k, p in model.state_dict().items() if k.endswith("engram.tables")
    ]
    assert before, "the fixture must start with an in-VRAM table to host"
    lines: list[str] = []
    report = sft_train(model, run, log=lines.append)
    hosts = host_engrams(model)
    assert len(hosts) == 1
    assert not any(k.endswith("engram.tables") for k in model.state_dict())
    assert report["counters"]["engram_host_updates"] == 3
    assert report["counters"]["engram_host_update_rows"] > 0
    assert not torch.equal(hosts[0].host.rows, before[0]), "the host tables must train"
    assert (Path(run.train.out_dir) / "s.ngram").is_file()
    step3 = [ln for ln in lines if ln.startswith("sft step=3")]
    assert step3 and "engram_host_updates=1" in step3[0]  # the WINDOW's count: one per step
    assert "table set(s) in host RAM" in lines[0]


def test_sft_engram_ram_resume_restores_the_sidecar(tmp_path: Path):
    """A host-table SFT run resumes through the same loader: `load_ckpt` refuses
    a missing `.ngram`, and the tables continue from the checkpoint's rows
    (`lr=1e-6`, so the resumed step leaves them where the load put them)."""
    run = sft_run(tmp_path, steps=2, log_every=1, ckpt_every=2, lr=1e-6, engram_ram=True)
    sft_train(DormouseModel(run.model), run, log=lambda *_: None)
    ng = Path(run.train.out_dir) / "s.ngram"
    assert (Path(run.train.out_dir) / "s.pt").is_file() and ng.is_file()
    saved = torch.load(ng, map_location="cpu", weights_only=True)["engrams"][0]["rows"]

    model = DormouseModel(run.model)
    longer = dataclasses.replace(run, train=dataclasses.replace(run.train, steps=4))
    report = sft_train(model, longer, resume=True, log=lambda *_: None)
    hosts = host_engrams(model)
    assert hosts, "the resumed run must be on host tables"
    assert float((hosts[0].host.rows - saved).abs().max()) < 1e-3, (
        "the resume must restore the sidecar"
    )
    # `save_ckpt` stamps the 0-BASED index of the completed step, so the resume
    # starts at step 2 and runs 2..3 = two steps.
    assert report["counters"]["engram_host_updates"] == 2


# --------------------------------------------------------------------------- the CLI

_TINY_PRESET = "\n".join(
    [
        "d_model = 64",
        "n_heads = 4",
        "head_dim = 16",
        "d_ffn = 128",
        "max_seq_len = 64",
        "max_iter = 2",
        "rank = 8",
        "shared_rank = 16",
        "moe_topk = 2",
        "moe_capacity = 1.0",
        "engram_orders = [2, 3]",
        "engram_rows = 64",
        "engram_dim = 8",
        "bf16 = false",
    ]
)


def _preset(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.toml"
    p.write_text(_TINY_PRESET)
    return p


def test_cli_sft_builds_the_corpus_and_trains(tmp_path: Path):
    corpus = tmp_path / "sft.bin"
    r = runner.invoke(
        app,
        [
            "sft",
            str(_preset(tmp_path)),
            "--data",
            str(corpus),
            "--build-sft",
            "--n-tasks",
            "8",
            "--ckpt-name",
            "s",
            "--set",
            f"out_dir={tmp_path / 'ck'}",
            "--set",
            "batch=2",
            "--set",
            "seq_len=64",
            "--set",
            "steps=5",
            "--set",
            "log_every=1",
            "--set",
            "ckpt_every=5",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    assert corpus.is_file() and "sft corpus" in r.output
    assert (tmp_path / "ck" / "s.pt").is_file()
    assert (tmp_path / "ck" / "s.config.toml").is_file()
    assert "sft done" in r.output and "masked_bytes=" in r.output


def test_cli_sft_refuses_a_missing_corpus(tmp_path: Path):
    r = runner.invoke(
        app,
        [
            "sft",
            str(_preset(tmp_path)),
            "--data",
            str(tmp_path / "nope.bin"),
            "--set",
            "seq_len=64",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code != 0
    assert "build-sft" in str(r.exception)


def test_cli_sft_refuses_a_resume_without_a_checkpoint(tmp_path: Path):
    r = runner.invoke(
        app,
        [
            "sft",
            str(_preset(tmp_path)),
            "--data",
            str(tmp_path / "sft.bin"),
            "--build-sft",
            "--n-tasks",
            "4",
            "--ckpt-name",
            "absent",
            "--set",
            f"out_dir={tmp_path / 'ck2'}",
            "--set",
            "seq_len=64",
            "--set",
            "steps=2",
            "--resume",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code != 0
    assert "resume" in str(r.exception)


def test_cli_sft_init_from_an_engram_ram_checkpoint(tmp_path: Path):
    """The CLI shape of the pretrain handoff: `sft --init-from <engram_ram
    checkpoint>` with `--set engram_ram=true` (the substrate the checkpoint
    records — a mismatch is a LOUD refusal, see the library gates).

    The checkpoint is built from the CLI's OWN resolved run, so the strict load
    has nothing to forgive; the run writes its own `.ngram` at the checkpoint
    cadence, which is what makes the SFT run resumable."""
    from dormouse.config import load_run
    from dormouse.train.loop import save_ckpt
    from dormouse.train.optim import MixOptimizer

    preset = _preset(tmp_path)
    pre = load_run(
        str(preset),
        train_overrides={
            "engram_ram": True,
            "out_dir": str(tmp_path / "pre"),
            "ckpt_name": "pre",
            "seq_len": 64,
            "batch": 2,
        },
    )
    model = DormouseModel(pre.model)
    assert enable_host_tables(model) == 1
    ids = torch.arange(2 * 64, dtype=torch.long).view(2, 64) % 251
    model(ids)[0].sum().backward()
    opt = MixOptimizer(model, lr=pre.train.lr, wd=pre.train.wd, opt=pre.train.opt)
    save_ckpt(pre, model, opt, 0)
    ckpt = tmp_path / "pre" / "pre.pt"
    assert ckpt.is_file() and ckpt.with_suffix(".ngram").is_file()

    r = runner.invoke(
        app,
        [
            "sft",
            str(preset),
            "--data",
            str(tmp_path / "sft.bin"),
            "--build-sft",
            "--n-tasks",
            "4",
            "--init-from",
            str(ckpt),
            "--ckpt-name",
            "s",
            "--set",
            f"out_dir={tmp_path / 'ck'}",
            "--set",
            "engram_ram=true",
            "--set",
            "batch=2",
            "--set",
            "seq_len=64",
            "--set",
            "steps=2",
            "--set",
            "log_every=1",
            "--set",
            "ckpt_every=2",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "table set(s) in host RAM" in r.output
    assert "sft init: weights from" in r.output
    assert (tmp_path / "ck" / "s.ngram").is_file(), "the SFT run saves its own sidecar"


def test_init_from_seeds_weights_without_the_config_gate(tmp_path):
    """`--init-from` loads WEIGHTS from a foreign checkpoint: no config gate
    (that is `resume`'s job), no optimizer state, and a LOUD refusal for a
    missing file or an architecture mismatch."""
    from dormouse.instrument import LoudError
    from dormouse.train.sft import sft_train

    run = sft_run(tmp_path, steps=2, log_every=2)

    # a foreign checkpoint: the model's own state at a different init
    foreign = tmp_path / "foreign.pt"
    other = DormouseModel(run.model)
    torch.save({"model": other.state_dict(), "step": 7}, foreign)

    model = DormouseModel(run.model)
    before = [p.detach().clone() for p in model.parameters()]
    report = sft_train(model, run, init_from=str(foreign), log=lambda *_: None)
    assert math.isfinite(report["loss"])
    # the weights ARE the foreign ones at load time (not the fresh init)
    for p, b in zip(model.parameters(), before, strict=True):
        if not torch.equal(p, b):
            break
    else:  # pragma: no cover - the step must move weights
        raise AssertionError("the SFT step did not move any weight")

    with pytest.raises(LoudError, match="does not exist"):
        sft_train(
            DormouseModel(run.model), run, init_from=str(tmp_path / "no.pt"), log=lambda *_: None
        )
    bad = tmp_path / "bad.pt"
    torch.save({"model": {"nope": torch.zeros(1)}}, bad)
    # A model half with no `engram.tables` is EITHER a host-table checkpoint or
    # the wrong architecture; this run is in-VRAM, so both are refused with the
    # substrate message (the keys torch would name are the same mismatch, caught
    # one step earlier — see `_init_cfg`).
    with pytest.raises(LoudError, match="carries no `engram.tables`"):
        sft_train(DormouseModel(run.model), run, init_from=str(bad), log=lambda *_: None)


# --- the engram_ram handoff: one loader, and the substrate it leaves behind ---


def _host_ckpt(run: RunConfig, fill: float = 0.5) -> Path:
    """A host-table checkpoint pair (model half + `.ngram` sibling), CPU-only.

    Built the way a `--engram-ram` pretrain run saves: `enable_host_tables`
    (the tables leave the parameter list) then `save_ckpt`, which writes the
    sibling from `host_state`. The rows are FILLED so the transfer is checkable
    — a fresh init's rows are ~N(0, 0.02), never 0.5.
    """
    from dormouse.train.loop import save_ckpt
    from dormouse.train.optim import MixOptimizer

    model = DormouseModel(run.model)
    assert enable_host_tables(model) == 1
    for m in host_engrams(model):
        m.host.rows.fill_(fill)
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    model(ids)[0].sum().backward()
    opt = MixOptimizer(model, lr=run.train.lr, wd=run.train.wd, opt=run.train.opt)
    save_ckpt(run, model, opt, 0)
    return Path(run.train.out_dir) / f"{run.train.ckpt_name}.pt"


def test_init_from_an_engram_ram_checkpoint_loads_and_sft_trains(tmp_path: Path):
    """Gate (a): `--init-from` a host-table checkpoint (tonight's pretrain shape)
    loads through `load_model_state` and SFT trains on it.

    The rows ARE the checkpoint's — `lr=1e-6` keeps them where the load put them
    (0.5) while the model half still takes a step — and the memory is STEPPED
    (the host counters fire). A raw `load_state_dict` refused this checkpoint
    outright (`Missing key(s): "loop.engram.tables"`).
    """
    pre = sft_run(
        tmp_path, steps=1, ckpt_name="pretrain", out_dir=str(tmp_path / "pre"), engram_ram=True
    )
    ckpt = _host_ckpt(pre)
    run = sft_run(tmp_path, steps=1, log_every=1, lr=1e-6, engram_ram=True)
    model = DormouseModel(run.model)
    lines: list[str] = []
    report = sft_train(model, run, init_from=str(ckpt), log=lines.append)
    assert math.isfinite(report["loss"])
    hosts = host_engrams(model)
    assert len(hosts) == 1
    assert float((hosts[0].host.rows - 0.5).abs().max()) < 1e-3, "the checkpoint's rows must arrive"
    assert report["counters"]["engram_host_updates"] == 1
    assert report["counters"]["engram_host_update_rows"] > 0
    assert any(ln.startswith("sft init: weights from") for ln in lines)


def test_init_from_a_host_checkpoint_without_its_sidecar_is_loud(tmp_path: Path):
    """Gate (b): on the host path the sibling `.ngram` IS the memory's state, so
    a missing one is the loader's LOUD refusal — never freshly seeded rows."""
    pre = sft_run(
        tmp_path, steps=1, ckpt_name="pretrain", out_dir=str(tmp_path / "pre"), engram_ram=True
    )
    ckpt = _host_ckpt(pre)
    assert ckpt.with_suffix(".ngram").is_file()
    ckpt.with_suffix(".ngram").unlink()
    run = sft_run(tmp_path, steps=1, engram_ram=True)
    with pytest.raises(LoudError, match=r"Escape: restore the \.ngram"):
        sft_train(DormouseModel(run.model), run, init_from=str(ckpt), log=lambda *_: None)


def test_init_from_an_in_vram_checkpoint_still_loads(tmp_path: Path):
    """Gate (c): the old path is unbroken — a checkpoint whose tables ARE in the
    model half loads into an in-VRAM run, with no host switch and no `.ngram`."""
    run = sft_run(tmp_path, steps=1, log_every=1, lr=1e-6)
    src = DormouseModel(run.model)
    with torch.no_grad():
        for key, p in src.state_dict().items():
            if key.endswith("engram.tables"):
                p.fill_(0.25)
    foreign = tmp_path / "vram.pt"
    torch.save({"model": src.state_dict(), "step": 3}, foreign)

    model = DormouseModel(run.model)
    report = sft_train(model, run, init_from=str(foreign), log=lambda *_: None)
    assert math.isfinite(report["loss"])
    assert not host_engrams(model)
    tables = [p for k, p in model.state_dict().items() if k.endswith("engram.tables")]
    assert len(tables) == 1
    assert float((tables[0] - 0.25).abs().max()) < 1e-3
    assert not foreign.with_suffix(".ngram").exists()


def test_init_from_refuses_a_substrate_mismatch(tmp_path: Path):
    """The flag decides the host switch (BEFORE the optimizer), the loop's host
    step and the `.ngram` write — so a run whose `engram_ram` disagrees with the
    init checkpoint's substrate is refused, not silently degraded."""
    host_ckpt = _host_ckpt(
        sft_run(tmp_path, steps=1, ckpt_name="pre", out_dir=str(tmp_path / "pre"), engram_ram=True)
    )
    vram = tmp_path / "vram.pt"
    torch.save({"model": DormouseModel(sft_run(tmp_path, steps=1).model).state_dict()}, vram)

    # a host-table init into an in-VRAM run: the tables would be read, never
    # stepped, never saved
    run = sft_run(tmp_path, steps=1)
    with pytest.raises(LoudError, match="--set engram_ram=true"):
        sft_train(DormouseModel(run.model), run, init_from=str(host_ckpt), log=lambda *_: None)

    # an in-VRAM init into a host-table run: the switch would drop its memory
    host_run = sft_run(tmp_path, steps=1, engram_ram=True)
    with pytest.raises(LoudError, match="--set engram_ram=false"):
        sft_train(DormouseModel(host_run.model), host_run, init_from=str(vram), log=lambda *_: None)


def test_chat_jsonl_rows_are_filtered_capped_and_deterministic(tmp_path):
    """The chat rows go through the build's own window rule: long rows and
    sentinel-carrying rows are dropped with counts, the cap keeps a
    deterministic sample, and two builds of one (seed, args) are bit-equal."""
    import json

    rows = [
        {"instruction": "Hi there", "response": "Hello!"},
        {"instruction": "How are you?", "response": "Fine, thanks."},
        {"instruction": "Too long?", "response": "x" * 600},
        {"instruction": "Bad\x00byte", "response": "no"},
        {"instruction": "", "response": "empty instruction"},
    ]
    src = tmp_path / "chat.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    a = build_sft_corpus(tmp_path / "a.bin", 7, seq_len=64, n_tasks=0, chat_data=src, chat_cap=1)
    build_sft_corpus(tmp_path / "b.bin", 7, seq_len=64, n_tasks=0, chat_data=src, chat_cap=1)
    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()
    assert a["chat_loaded"] == 2 and a["chat_used"] == 1
    assert a["chat_dropped_long"] == 1 and a["chat_dropped_dirty"] == 2
    assert a["chat_files"] == {str(src): 5}

    with pytest.raises(LoudError, match="does not exist"):
        build_sft_corpus(
            tmp_path / "c.bin", 1, seq_len=64, n_tasks=0, chat_data=tmp_path / "no.jsonl"
        )


def test_qa_text_format_parses_and_refuses_malformed_lines(tmp_path):
    """The hand-editable Q/A format: `#` comments and blank lines are free,
    pairs are positional, and every malformed line is a LOUD error naming
    file:line — a personal file deserves a message, not a silent skip."""
    src = tmp_path / "mine.qa"
    src.write_text(
        "# my own examples\n"
        "\n"
        "Q: What is your name?\n"
        "A: dormouse.\n"
        "\n"
        "Q: Привет!\n"
        "A: Привет! Чем помочь?\n",
        encoding="utf-8",
    )
    built = build_sft_corpus(tmp_path / "a.bin", 3, seq_len=128, n_tasks=0, chat_data=src)
    assert built["chat_files"] == {str(src): 2}
    assert built["chat_loaded"] == 2 and built["chat_used"] == 2

    bad = tmp_path / "bad.qa"
    bad.write_text("A: answer first?\n", encoding="utf-8")
    with pytest.raises(LoudError, match=r"bad\.qa:1"):
        build_sft_corpus(tmp_path / "b.bin", 3, seq_len=128, n_tasks=0, chat_data=bad)
    bad.write_text("Q: q1\nQ: q2\nA: a\n", encoding="utf-8")
    with pytest.raises(LoudError, match="before the question"):
        build_sft_corpus(tmp_path / "b.bin", 3, seq_len=128, n_tasks=0, chat_data=bad)
    bad.write_text("Q: unfinished question\n", encoding="utf-8")
    with pytest.raises(LoudError, match="never got"):
        build_sft_corpus(tmp_path / "b.bin", 3, seq_len=128, n_tasks=0, chat_data=bad)


def test_chat_cap_is_per_file_so_small_files_stay_whole(tmp_path):
    """The cap applies to EACH file: a small hand-written file is fully kept
    even when a large export is sampled, and the sample is a pure function of
    the seed."""
    big = tmp_path / "big.qa"
    big.write_text(
        "Q: q1\nA: a1\nQ: q2\nA: a2\nQ: q3\nA: a3\nQ: q4\nA: a4\nQ: q5\nA: a5\n",
        encoding="utf-8",
    )
    small = tmp_path / "small.qa"
    small.write_text("Q: mine\nA: kept\n", encoding="utf-8")

    a = build_sft_corpus(
        tmp_path / "a.bin", 11, seq_len=64, n_tasks=0, chat_data=[big, small], chat_cap=2
    )
    build_sft_corpus(
        tmp_path / "b.bin", 11, seq_len=64, n_tasks=0, chat_data=[big, small], chat_cap=2
    )
    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()
    assert a["chat_fit"] == 6 and a["chat_used"] == 3  # 2 sampled from big + the whole small
    assert a["chat_files"] == {str(big): 5, str(small): 1}


def test_holdout_moves_the_tail_to_a_sibling(tmp_path):
    """`holdout=N` splits the LAST N records into `<name>.holdout` (the same
    fixed record size), so the training stream never reads them and the eval
    stream reads nothing else. Deterministic like the rest of the build."""
    a = build_sft_corpus(tmp_path / "a.bin", 5, seq_len=64, n_tasks=0, holdout=7)
    build_sft_corpus(tmp_path / "b.bin", 5, seq_len=64, n_tasks=0, holdout=7)
    hold = tmp_path / "a.bin.holdout"
    assert hold.is_file()
    assert hold.read_bytes() == (tmp_path / "b.bin.holdout").read_bytes()
    assert a["holdout"] == 7 and a["records"] > 7
    assert len(hold.read_bytes()) == 7 * a["record_bytes"]
    with pytest.raises(LoudError, match="holdout"):
        build_sft_corpus(tmp_path / "c.bin", 5, seq_len=64, n_tasks=0, holdout=10_000)


def test_sft_eval_reports_first_byte_agreement(tmp_path):
    """The in-loop eval (masked CE + first-answer-byte agreement) runs on the
    holdout file and lands in the log — the metric that catches a model which
    only predicts answer marginals (the sft5 failure)."""
    corpus = tmp_path / "sft.bin"
    built = build_sft_corpus(corpus, seed=1, seq_len=SEQ, n_tasks=32, holdout=4)
    hold = tmp_path / "sft.bin.holdout"
    run = sft_run(
        tmp_path,
        corpus=corpus,
        steps=4,
        log_every=2,
        eval=True,
        eval_every=2,
        eval_batches=2,
        eval_data=str(hold),
    )
    lines: list[str] = []
    sft_train(DormouseModel(run.model), run, log=lines.append)
    evals = [ln for ln in lines if ln.startswith("sft eval")]
    assert evals, lines
    assert "first=" in evals[-1] and "over " in evals[-1]
    assert built["holdout"] == 4


def test_ru_bank_is_broad_and_fits_one_window():
    """The RU bank must stay deterministic, wide (>= 200 pairs) and every pair
    must render into a 512-byte window — it is the fix for an English-drift
    chat, and it is generated, not sampled."""
    from dormouse.train.sft import _ru_pairs

    ru = _ru_pairs()
    assert len(ru) >= 200
    assert len({q for q, _ in ru}) == len(ru)  # no duplicate prompts
    assert all(len(render(q, a)) <= 512 for q, a in ru)
    assert any("Сколько будет" in q for q, _ in ru)  # the drills are there
