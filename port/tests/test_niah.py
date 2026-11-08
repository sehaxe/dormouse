"""The S-NIAH-8 runner's four load-bearing checks — CPU only, one tiny model.

WHY THESE FOUR. `scripts/niah.py` implements the long-context protocol's gate D
and the gate C rank probe, and its numbers are only as good as the things those
gates assume:

1. the item generator puts the key where it says it does (a misplaced needle
   would read as a retrieval failure, or as a pass on the wrong bytes);
2. the segmented forward IS the 1-pass forward on the tiny model — the same
   comparison the script's gate B makes, at the report's fp32 tolerance (1e-5).
   If this fails, a 1M number is a different function;
3. the artifact refusal fires for `use_csa=false` and names the escape. That
   refusal IS gate A: the on-disk 95M is a 1K dense transformer, so a 1M
   retrieval number on it would measure a different network;
4. `--dry-run` exits 0 with the item count and the §4.5 arithmetic, without a
   checkpoint — it is the one path that can run before a CSA artifact exists.

The real gate D (100 1M items) cannot run here: it needs a trained CSA
checkpoint that is not on this disk. This file does not pretend otherwise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from conftest import tiny_cfg

from dormouse.config import RunConfig, TrainConfig
from dormouse.model import DormouseModel

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("niah", ROOT / "scripts" / "niah.py")
niah = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = niah  # a dataclass looks itself up in sys.modules at creation
_spec.loader.exec_module(niah)


# --- 1. the generator's placement arithmetic -------------------------------


def test_item_places_the_key_at_the_stated_distance():
    item = niah.build_item(seed=7, context=2048, align=8)
    assert len(item.scored) == 2048
    assert len(item.prompt) == 2048 - niah.KEY_LEN
    assert item.prompt.endswith(niah.CUE), "the question must end the prompt"
    assert item.key_at % 8 == 0, "the key must be one aligned fine chunk"
    assert item.prompt[item.key_at : item.key_at + niah.KEY_LEN] == item.key
    query_at = item.key_at + niah.KEY_LEN + item.distance
    assert item.prompt[query_at:] == niah.CUE, "distance is not key-end -> question"
    assert item.prompt.count(niah.CUE) == 2
    assert item.prompt.count(item.key) == 1, "the key must be unique in the haystack"
    # The scored target is exactly the question's key, not a copy of it.
    assert item.scored[item.key_at : item.key_at + niah.KEY_LEN] == item.key
    assert item.scored[-niah.KEY_LEN :] == item.key

    placed = niah.build_item(seed=7, context=2048, distance=1000, align=8)
    assert 1000 <= placed.distance < 1008, "distance is honoured to the alignment"
    assert placed.prompt[placed.key_at + niah.KEY_LEN + placed.distance :] == niah.CUE
    assert placed.prompt.count(niah.CUE) == 2


def test_no_needle_keeps_the_drawn_key_but_not_the_text():
    base = niah.build_item(seed=11, context=512)
    none = niah.build_item(seed=11, context=512, needle=False)
    assert none.key == base.key, "the removed key must be the one a hallucination emits"
    assert none.needle is False
    assert none.prompt.count(niah.CUE) == 1
    assert none.prompt.count(base.key) == 0
    assert len(none.scored) == 512


# --- 2. the carry gate: 2 segments == 1 pass -------------------------------


def _tiny() -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(
        tiny_cfg(
            use_csa=True,
            csa_chunk=8,
            csa_kv_dim=64,
            csa_topk=8,
            csa_coarse_chunk=0,
            use_engram=False,
            moe_topk=0,
            moe_capacity=0.0,
            max_seq_len=64,
        )
    )


def test_two_segments_are_the_one_pass_at_fp32_tolerance():
    model = _tiny()
    ids = (torch.arange(64, dtype=torch.long).view(1, -1) % 251).clone()
    with torch.no_grad():
        ref, _, _ = model(ids)
        got, _, counters = model(ids, segment_len=32)
    assert counters["loop_segments"] == 2, "the segmented path did not run twice"
    assert ref.shape == got.shape
    delta = float((ref[-8:] - got[-8:]).abs().max())
    print(f"[carry] 64 B = 1 pass vs 2x32: last-8 max|dlogits|={delta:.3e} (tol 1e-5 fp32)")
    assert delta <= 1e-5, f"the 2-segment pass is not the 1-pass forward: {delta:.3e}"


# --- 3. gate A refuses a model that cannot run this test -------------------


def test_artifact_refusal_fires_on_dense_and_names_the_escape():
    why = niah.artifact_refusal(tiny_cfg(use_csa=False), 1024)
    assert why is not None
    assert "use_csa=false" in why
    assert "train a CSA model at >=8K" in why
    assert ".bulba/plan.md" in why

    short = niah.artifact_refusal(tiny_cfg(use_csa=True, max_seq_len=1024), 1 << 20)
    assert short is not None and "max_seq_len=1024" in short
    assert "train a CSA model at >=8K" in short

    ok = niah.artifact_refusal(tiny_cfg(use_csa=True, max_seq_len=1024), 1024)
    assert ok is None, "a legal CSA config must not be refused"


# --- 4. the dry run is the path that needs no checkpoint -------------------


def test_dry_run_exits_zero_and_prints_the_arithmetic(capsys):
    rc = niah.main(
        ["--dry-run", "--context", "4096", "--segment", "512", "--items", "3", "--seed", "7"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "seed=7" in out
    assert "context=4096" in out
    assert "items=3" in out
    assert "segment=512" in out
    assert "segments/item=8" in out
    assert "FLOPs/pass=" in out
    assert "state (report" in out
    assert "command: PYTHONPATH=port/src" in out


# --- an end-to-end real path on a tiny snapshot ----------------------------


def _tiny_snapshot(tmp_path: Path) -> Path:
    """A checkpoint PAIR the real path can load — tiny, CPU, no training."""
    cfg = tiny_cfg(
        use_csa=True,
        csa_chunk=8,
        csa_kv_dim=64,
        csa_topk=8,
        csa_coarse_chunk=0,
        use_engram=False,
        moe_topk=0,
        moe_capacity=0.0,
        max_seq_len=1024,
    )
    torch.manual_seed(0)
    model = DormouseModel(cfg)
    run = RunConfig(
        source="test",
        model=cfg,
        train=TrainConfig(ckpt_name="tiny", out_dir=str(tmp_path), seed=0),
    )
    (tmp_path / "tiny.config.toml").write_text(run.snapshot_toml())
    torch.save({"model": model.state_dict()}, tmp_path / "tiny.pt")
    return tmp_path


def test_real_path_gates_and_controls_on_a_tiny_snapshot(tmp_path, capsys):
    """Every real-path stage the 1M run will use, on 256-byte items.

    The rank probe is OFF here (`--rank-items 0`) because an untrained model's
    selector is chance and the protocol correctly stops before gate D on it; the
    next test drives the rank probe itself.
    """
    ckpt = _tiny_snapshot(tmp_path)
    rc = niah.main(
        [
            "--ckpt-dir",
            str(ckpt),
            "--ckpt-name",
            "tiny",
            "--context",
            "256",
            "--segment",
            "128",
            "--items",
            "2",
            "--rank-items",
            "0",
            "--device",
            "cpu",
            "--controls",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "gate A: PASS" in out
    assert "gate B:" in out and "max|dlogits|" in out
    assert "gate C: NOT MEASURED" in out
    assert "gate D tiny: exact=" in out
    assert "control no-needle" in out
    assert "control shuffled-key" in out
    assert "control csa-off" in out
    assert "unused" in out, "the off control must COUNT the CSA tensors it drops"
    assert "control positive 1024 B" in out


def test_rank_probe_runs_on_a_tiny_snapshot(tmp_path, capsys):
    ckpt = _tiny_snapshot(tmp_path)
    rc = niah.main(
        [
            "--ckpt-dir",
            str(ckpt),
            "--ckpt-name",
            "tiny",
            "--context",
            "256",
            "--segment",
            "128",
            "--items",
            "1",
            "--rank-items",
            "1",
            "--device",
            "cpu",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "gate C: flat best-layer rank" in out
    print(out)


def test_the_window_refusal_is_a_caveat_when_it_is_a_generalization_test():
    """`max_seq_len < context` refuses by default and is legal with the flag.

    The loop form has no absolute position table and carries KDA/CSA state, so a
    pass past the trained window is mechanically expressible — and whether it
    generalizes is the experiment. The flag names that, and the `use_csa=false`
    refusal must survive it (a lenient flag cannot wave through a different
    network).
    """
    from dormouse.config import ModelConfig

    artifact_refusal = niah.artifact_refusal
    dense = ModelConfig(block_form="transformer", max_iter=1, n_layers=2,
                        use_kda=False, use_engram=False, engram_rows=0)
    assert artifact_refusal(dense, 1024) is not None
    assert artifact_refusal(dense, 1024, allow_beyond_window=True) is not None, (
        "the use_csa refusal must not be waived by the window flag"
    )

    csa = ModelConfig(
        block_form="kda", use_csa=True, csa_chunk=8, csa_coarse_chunk=256,
        max_seq_len=8192, max_iter=2, loop_blocks=1,
    )
    assert artifact_refusal(csa, 1_048_576) is not None
    assert artifact_refusal(csa, 1_048_576, allow_beyond_window=True) is None
