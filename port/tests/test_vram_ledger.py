"""Gates for `scripts/vram_ledger.py`: the four claims it makes.

1. **The param ledger is the model tree, not a hand sum.** The instantiated
   `meta` model's count for the current flagship equals `dormouse info`'s
   banner exactly (102 678 943), and the committed live recipe (m400b,
   199 275 551) too. If the ledger's grouping drops a parameter, this fails.
2. **The activation formula reproduces the one measured peak.** The anchor is
   `port/src/dormouse/model/act_mem.py:4`: `b8 s1024`, `max_iter 3`, **7.2 GiB**.
   The test asserts the citation still says that and the formula is within 10 %.
3. **Every lever that claims a saving is arithmetic.** Each row is recomputed
   here from `peak_bytes`/`activation_bytes`; a removed formula (a zero return)
   fails the equality. The two levers that honestly claim zero (`grad_accum`,
   `segment_len` on the KDA arm) are pinned at zero.
4. **The driver refuses a busy card.** `measure()` routes through
   `_lane.preflight` before importing/using CUDA; a busy preflight prints the
   SKIP banner and returns 0 without building a model.

No GPU is touched: every path here is the CPU ledger, and the driver test
monkeypatches the preflight before the CUDA branch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dormouse.cli import app

REPO = Path(__file__).resolve().parents[2]  # port/tests -> repo root
LEDGER_PATH = REPO / "scripts" / "vram_ledger.py"
FLAGSHIP_PARAMS = 102_678_943
M400B_PARAMS = 199_275_551


def _load_ledger():
    spec = importlib.util.spec_from_file_location("vram_ledger", LEDGER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["vram_ledger"] = mod
    spec.loader.exec_module(mod)
    return mod


LEDGER = _load_ledger()


@pytest.fixture(scope="module")
def built():
    """The four targets and their ledgers, built once for the module."""
    targets = LEDGER.build_targets()
    ledgers = {k: LEDGER.param_ledger(t.cfg) for k, t in targets.items()}
    return targets, ledgers


@pytest.fixture(scope="module")
def reference():
    target = LEDGER.reference_target()
    return target, LEDGER.param_ledger(target.cfg)


def _cli_params(preset: str) -> int:
    result = CliRunner().invoke(app, ["info", preset])
    assert result.exit_code == 0, result.output
    import re

    m = re.search(r"([\d,]+) params", result.output)
    assert m, f"no params banner in `dormouse info {preset}`:\n{result.output}"
    return int(m.group(1).replace(",", ""))


def test_param_counts_match_dormouse_info(built):
    from dormouse.config import load_preset, validate

    targets, ledgers = built
    flagship = LEDGER.Target("flagship", load_preset("flagship_base"), "")
    validate(flagship.cfg)
    got = LEDGER.param_ledger(flagship.cfg).total
    assert got == FLAGSHIP_PARAMS
    assert got == _cli_params("flagship_base")

    m400b = LEDGER.Target("m400b", load_preset("m400b"), "")
    validate(m400b.cfg)
    live = LEDGER.param_ledger(m400b.cfg).total
    assert live == M400B_PARAMS
    assert live == _cli_params("m400b")

    # The ledger's groups partition the same tree.
    for key in targets:
        ledger = ledgers[key]
        assert sum(ledger.groups.values()) == ledger.total
        assert ledger.muon + ledger.adamw == ledger.total


def test_activation_formula_reproduces_the_measured_peak(reference):
    """7.2 GiB is the only measured b8/s1024 peak on record (act_mem.py:4)."""
    target, ledger = reference
    cited = (REPO / "port" / "src" / "dormouse" / "model" / "act_mem.py").read_text()
    assert "measured peak is 7.2 GiB" in cited.splitlines()[3], "the citation moved"

    peak = LEDGER.peak_bytes(target.cfg, ledger, 8, 1024, LEDGER.Levers(), target.ns_chunk)
    rel = abs(peak.peak_gib - 7.2) / 7.2
    assert rel <= 0.10, (
        f"activation model {peak.peak_gib:.3f} GiB is {rel:.1%} off the measured "
        f"7.2 GiB (act_mem.py:4)"
    )
    print(f"reference peak {peak.peak_gib:.3f} GiB, measured 7.2 GiB, {rel:.2%}")


def test_every_claiming_lever_is_exact_arithmetic(built):
    targets, ledgers = built
    target, ledger = targets["flagship"], ledgers["flagship"]
    rows = {r["lever"]: r for r in LEDGER.lever_rows(target.cfg, ledger, 8, 1024, target.ns_chunk)}
    base = LEDGER.peak_bytes(target.cfg, ledger, 8, 1024, LEDGER.Levers(), target.ns_chunk)

    def diff(**kw) -> float:
        pk = LEDGER.peak_bytes(target.cfg, ledger, 8, 1024, LEDGER.Levers(**kw), target.ns_chunk)
        return (base.peak - pk.peak) / LEDGER.GIB

    assert rows["activation checkpointing (iter_checkpoint/full)"]["gib"] == pytest.approx(
        diff(recompute="full")
    )
    assert rows["selective recompute (write half)"]["gib"] == pytest.approx(
        diff(recompute="selective")
    )
    assert rows["mxfp4 saved tensors (act_ckpt int4)"]["gib"] == pytest.approx(
        diff(act_pack="mxfp4")
    )
    assert rows["optimizer state bf16 (muon_state_dtype)"]["gib"] == pytest.approx(
        diff(muon_dtype="bf16")
    )
    assert rows["ZeRO-1 shard-if-two-cards (state only)"]["gib"] == pytest.approx(
        diff(zero_cards=2)
    )
    assert rows["host offload of optimizer state"]["gib"] == pytest.approx(diff(offload_state=True))
    # The two that must claim NOTHING on this KDA/bank target.
    assert rows["gradient accumulation"]["gib"] == 0.0
    seg = next(r for k, r in rows.items() if k.startswith("sequence chunking"))
    assert seg["gib"] == 0.0, "KDA has no s^2 score matrix for segmenting to shrink"
    # A removed formula (the only alternative to a true saving) turns these red.
    for name in (
        "activation checkpointing (iter_checkpoint/full)",
        "selective recompute (write half)",
        "mxfp4 saved tensors (act_ckpt int4)",
        "optimizer state bf16 (muon_state_dtype)",
        "ZeRO-1 shard-if-two-cards (state only)",
        "host offload of optimizer state",
    ):
        assert rows[name]["gib"] > 0.0, name


def test_max_batch_is_the_largest_that_fits(built):
    targets, ledgers = built
    for key, target in targets.items():
        ledger = ledgers[key]
        for seq in (1024, 8192):
            b = LEDGER.max_batch(target.cfg, ledger, seq, LEDGER.Levers(), target.ns_chunk)
            if b == 0:
                one = LEDGER.peak_bytes(
                    target.cfg, ledger, 1, seq, LEDGER.Levers(), target.ns_chunk
                )
                assert one.peak > LEDGER.CARD_GIB * LEDGER.GIB
                continue
            fits = LEDGER.peak_bytes(target.cfg, ledger, b, seq, LEDGER.Levers(), target.ns_chunk)
            assert fits.peak <= LEDGER.CARD_GIB * LEDGER.GIB
            if b < 512:
                nxt = LEDGER.peak_bytes(
                    target.cfg, ledger, b + 1, seq, LEDGER.Levers(), target.ns_chunk
                )
                assert nxt.peak > LEDGER.CARD_GIB * LEDGER.GIB, (key, seq, b)


def test_sparse_bank_bytes_and_flops_per_byte(built):
    targets, ledgers = built
    target = targets["spa1b"]
    cfg = target.cfg
    expanded = LEDGER.bank_cache_elements(cfg)
    assert expanded == cfg.n_experts * (cfg.d_model * cfg.rank + cfg.rank * cfg.d_ffn)
    bf16 = expanded * 2
    ternary = LEDGER.bank_ternary_bytes(cfg)
    assert ternary == (cfg.n_experts * cfg.rank * cfg.d_ffn + 3) // 4 + cfg.n_experts * cfg.rank * 4
    assert 2.5 < bf16 / LEDGER.GIB < 2.6
    assert 0.22 < ternary / LEDGER.GIB < 0.24
    assert ternary * 7 < bf16  # the 2-bit packer is >7x smaller incl. scales

    active = LEDGER.active_params(cfg)
    assert 35_000_000 < active < 42_000_000
    flops = LEDGER._flops_per_byte(cfg)
    assert 235e6 < flops < 250e6, f"{flops / 1e6:.1f} MFLOP/B is not the ~242 target"
    assert 62_000 * flops / 1e12 == pytest.approx(15.0, rel=0.05)


def test_decision_marks_impossible_rows_with_the_arithmetic(built):
    targets, ledgers = built
    rows = LEDGER.decision_rows(targets, ledgers)
    sparse_none = next(
        r for r in rows if r["target"].startswith("1B sparse") and r["levers"] == "none"
    )
    assert sparse_none["max_batch_s1024"] == 0
    assert "over by" in sparse_none["verdict"]
    dense_ckpt = next(
        r for r in rows if r["target"] == "1B dense" and r["levers"].startswith("ckpt=full+muon")
    )
    assert dense_ckpt["max_batch_s1024"] >= 8
    assert dense_ckpt["verdict"] == "ACHIEVABLE at b8"


def test_driver_refuses_a_busy_card(built, monkeypatch, capsys):
    sys.path.append(str(REPO / "port" / "scripts"))
    import _lane

    targets, ledgers = built
    target, ledger = targets["flagship"], ledgers["flagship"]

    def busy(_device: str, _ceiling_gib: float = 2.0) -> str:
        return "a compute process holds the card:\n12345, python, 14000"

    monkeypatch.setattr(_lane, "preflight", busy)
    rc = LEDGER.measure(target, ledger, 1, 128, LEDGER.Levers())
    out = capsys.readouterr().out
    assert rc == 0
    assert "SKIP" in out and "compute process holds the card" in out
    assert "peak_gib" not in out, "the driver measured after the refusal"


def test_driver_refuses_a_real_busy_preflight(built, monkeypatch, capsys):
    """The full path: driver -> `_lane.preflight` -> `nvidia-smi` busy row."""
    sys.path.append(str(REPO / "port" / "scripts"))
    import _lane

    class Busy:
        stdout = "12345, /usr/bin/python3, 14000\n"

    monkeypatch.setattr(_lane.subprocess, "run", lambda *a, **k: Busy())
    targets, ledgers = built
    rc = LEDGER.measure(targets["flagship"], ledgers["flagship"], 1, 128, LEDGER.Levers())
    out = capsys.readouterr().out
    assert rc == 0
    assert "SKIP" in out and "compute process holds the card" in out
    assert "peak_gib" not in out


def test_lane_preflight_reads_a_busy_row(monkeypatch):
    sys.path.append(str(REPO / "port" / "scripts"))
    import _lane

    assert _lane._compute_app_gib("12345, /usr/bin/python3, 14000") == pytest.approx(13.671875)
    assert _lane._compute_app_gib("not a row") == 0.0
