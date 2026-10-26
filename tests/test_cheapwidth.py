"""Cheap width: the three flags, their OFF parity, their bills, and their gates.

Every gate here fails on a revert:

1. the OFF path's state_dict and logits are the sha256s captured from a clean
   `main` worktree BEFORE the flags existed — a changed dtype, draw order or
   extra module moves them;
2. the MAC helper and the parameter deltas are checked against the INSTANTIATED
   model through `dormouse info`, so a formula that drifts from the module tree
   fails instead of reading true;
3. every new parameter is named in a gradient gate (finite, non-zero);
4. each flag ON moves the logits (a flag that never reaches the forward fails);
5. the tensors handed to `chunk_kda` keep FLA's `[b,t,h,hd]` contract under all
   eight flag combinations (stub kernel, CPU);
6. the refusals are LOUD and name the escape;
7. the changed forward does no host sync (`item`/`tolist`/`bool`/`float`).

The FLA-shape gate uses a STUB, not the real kernel: FLA is CUDA-only and this
suite is CPU. The stub records the exact argument shapes and returns a
zero output with the contract's own state, so `_fla_path`'s asserts run — it
proves the call site's contract, which is what the flag could break.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from dormouse import width
from dormouse.cli import app
from dormouse.config import ModelConfig, load_preset, validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.model.kda import KDAAttention
from dormouse.model.tsct import DenseLowRank

#: Captured from `/tmp/opencode/cheapwidth-main` (detached at `main`, 4f88102)
#: before any flag existed. `tiny` = TSCT rung; `tiny_dense` = use_tsct=false.
GOLDEN = {
    "tiny_tsct": (
        "d3360f3178dc1be2f3bfc6614f52927f762fc76054187098de8faa1bdb6d2d27",
        "57d0c0ee6c5b0240e81d95eaea18681d06ef2c9ca1400292c3f97948e79837ee",
        64_881,
    ),
    "tiny_dense": (
        "5452e0155674c0a4804a2d58f59c103b2fb3edb094fc02a36e85f1a53a0315c2",
        "d0b4f9ea389bfea7d5c287415353c1314d8148f0acc2bdddecde10bb3f450f78",
        81_761,
    ),
}


def _cfg(**over) -> ModelConfig:
    cfg = load_preset("tiny")
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _state_hash(model) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _logits_hash(model, seed: int = 7) -> str:
    torch.manual_seed(seed)
    ids = (torch.arange(2 * 16, dtype=torch.long) % model.cfg.vocab).reshape(2, 16)
    model.eval()
    with torch.no_grad():
        logits, loss, _ = model(ids, ids)
    assert loss is not None and torch.isfinite(loss)
    h = hashlib.sha256()
    h.update(logits.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _built(label: str, **over):
    torch.manual_seed(0)
    return DormouseModel(_cfg(**over))


@pytest.mark.parametrize("label,over", [("tiny_tsct", {}), ("tiny_dense", {"use_tsct": False})])
@pytest.mark.local_pin
def test_off_path_is_bit_identical_to_main(label, over):
    """Gate 1: the flags are off by default and off is main's tree, byte for byte."""
    want_state, want_logits, want_params = GOLDEN[label]
    model = _built(label, **over)
    assert param_count(model) == want_params
    assert _state_hash(model) == want_state, f"{label}: the OFF state_dict moved"
    assert _logits_hash(model) == want_logits, f"{label}: the OFF logits moved"


def test_mac_helper_prints_the_grid():
    """The pure-arithmetic helper runs with no torch at all (no GPU, no model)."""
    import importlib

    mod = importlib.import_module("dormouse.width")
    assert not hasattr(mod, "torch")
    mod.main()


def _info_total(*sets: str) -> int:
    """The instantiated total from the CLI's own `info` command."""
    out = CliRunner().invoke(app, ["info", "medium", *[f"--set={s}" for s in sets]])
    assert out.exit_code == 0, out.output
    import re

    m = re.search(r"([\d,]+) params", out.output)
    assert m, out.output
    return int(m.group(1).replace(",", ""))


def test_mac_and_params_match_dormouse_info():
    """Gate 2: the predicted deltas ARE the instantiated model's deltas.

    Dense rung (`use_tsct=false`) because that is where the readout is the `d²`
    the MAC table prices — on the TSCT rung the shipped `rank=64` readout is
    already `2·d·rank`, and `out_proj_rank` sets that rank instead.
    """
    d, h = 1024, 16  # medium
    base = _info_total("use_tsct=false")
    cases = {
        "latent": dict(attn_latent_rank=d // 4),
        "gqa": dict(n_kv_heads=2),
        "out_proj": dict(out_proj_rank=d // 4),
        "all": dict(attn_latent_rank=d // 4, n_kv_heads=2, out_proj_rank=d // 4),
    }
    for name, flags in cases.items():
        sets = ["use_tsct=false"] + [f"{k}={v}" for k, v in flags.items()]
        got = _info_total(*sets) - base
        want = width.param_delta(
            d,
            n_heads=h,
            attn_latent_rank=flags.get("attn_latent_rank", 0),
            n_kv_heads=flags.get("n_kv_heads", 0),
            out_proj_rank=flags.get("out_proj_rank", 0),
        )
        assert got == want, f"{name}: model delta {got:+,d} != formula {want:+,d}"
        if name == "all":
            assert got < 0, "the three flags together must SHRINK the model"
    # The table's own claim: at every width, `all` is cheaper per token.
    for d in (1024, 1536, 2048, 3072, 4096):
        hh = width.heads(d)
        base_mac = width.macs_per_token(d, iters=3, n_heads=hh)
        all_mac = width.macs_per_token(
            d, iters=3, n_heads=hh, attn_latent_rank=d // 4, n_kv_heads=2, out_proj_rank=d // 4
        )
        assert all_mac < base_mac, (d, base_mac, all_mac)


def test_every_new_parameter_gets_a_finite_nonzero_gradient():
    """Gate 3: the new tensors are in the graph, named if they are not."""
    tiny = 16  # d/4 in the tiny preset (d=64)
    cases = {
        "latent": (
            {"attn_latent_rank": tiny},
            {"loop.attn.latent_down.weight", "loop.attn.q_proj.weight"},
        ),
        "gqa": ({"n_kv_heads": 2}, {"loop.attn.k_proj.weight", "loop.attn.v_proj.weight"}),
        "out_proj": (
            {"use_tsct": False, "out_proj_rank": tiny},
            {"loop.out_proj.down.weight", "loop.out_proj.up.weight"},
        ),
        "all": (
            {"attn_latent_rank": tiny, "n_kv_heads": 2, "use_tsct": False, "out_proj_rank": tiny},
            {
                "loop.attn.latent_down.weight",
                "loop.attn.q_proj.weight",
                "loop.attn.k_proj.weight",
                "loop.attn.v_proj.weight",
                "loop.attn.g_proj.weight",
                "loop.out_proj.down.weight",
                "loop.out_proj.up.weight",
            },
        ),
    }
    ids = (torch.arange(2 * 16, dtype=torch.long) % 256).reshape(2, 16)
    for name, (over, want) in cases.items():
        torch.manual_seed(0)
        model = DormouseModel(_cfg(**over))
        _, loss, _ = model(ids, ids)
        loss.backward()
        for pname in sorted(want):
            p = dict(model.named_parameters())[pname]
            assert p.grad is not None, f"{name}: {pname} has no gradient"
            assert torch.isfinite(p.grad).all(), f"{name}: {pname} has non-finite gradient"
            assert p.grad.abs().sum() > 0, f"{name}: {pname} has an all-zero gradient"


def _logits(model, ids) -> torch.Tensor:
    with torch.no_grad():
        logits, _, _ = model(ids)
    return logits


@pytest.mark.parametrize(
    "name,on,off",
    [
        ("latent", {"attn_latent_rank": 16}, {}),
        ("gqa", {"n_kv_heads": 2}, {}),
        ("out_proj", {"use_tsct": False, "out_proj_rank": 16}, {"use_tsct": False}),
    ],
)
def test_each_flag_changes_the_logits(name, on, off):
    """Gate 4: ON must move the number; a flag that never reaches the forward fails."""
    ids = (torch.arange(2 * 16, dtype=torch.long) % 256).reshape(2, 16)
    torch.manual_seed(0)
    control = DormouseModel(_cfg(**off))
    torch.manual_seed(0)
    arm = DormouseModel(_cfg(**on))
    delta = (_logits(arm, ids) - _logits(control, ids)).abs().max()
    assert delta > 1e-5, f"{name}: logits did not move (max |Δ| {delta:.2e})"


# ---------------------------------------------------------------------------
# Gate 5: FLA's shape contract, under all eight flag combinations
# ---------------------------------------------------------------------------


def _stub_fla(monkeypatch) -> list[dict]:
    """Install a `fla.ops.kda.chunk_kda` stub; return the list it records into."""
    calls: list[dict] = []
    mod = types.ModuleType("fla.ops.kda")

    def chunk_kda(q, k, v, **kw):
        calls.append(
            {
                "q": tuple(q.shape),
                "k": tuple(k.shape),
                "v": tuple(v.shape),
                "g": tuple(kw["g"].shape),
                "beta": tuple(kw["beta"].shape),
            }
        )
        b, t, h, hd = q.shape
        out = torch.zeros_like(q)
        ht = torch.zeros(b, h, hd, hd, dtype=torch.float32, device=q.device)
        return out, ht

    mod.chunk_kda = chunk_kda
    for name in ("fla", "fla.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "fla.ops.kda", mod)
    monkeypatch.setattr(
        "dormouse.model.kda.fla_status", lambda device, recheck=False: (True, "stub")
    )
    return calls


@pytest.mark.parametrize("latent_rank", [0, 16])
@pytest.mark.parametrize("n_kv_heads", [0, 2])
@pytest.mark.parametrize("out_proj_rank", [0, 16])
def test_fla_call_keeps_the_bhhd_contract(monkeypatch, latent_rank, n_kv_heads, out_proj_rank):
    """Every combo hands `chunk_kda` `[b,t,n_heads,head_dim]` for all four."""
    calls = _stub_fla(monkeypatch)
    cfg = _cfg(
        attn_latent_rank=latent_rank,
        n_kv_heads=n_kv_heads,
        use_tsct=False,
        out_proj_rank=out_proj_rank,
    )
    attn = KDAAttention(cfg)
    x = torch.randn(2, 7, cfg.d_model)
    out, state = attn.forward_state(x, carry=True)
    assert out.shape == x.shape
    assert state is not None and state.dtype == torch.float32
    assert len(calls) == 1, calls
    want = (2, 7, cfg.n_heads, cfg.head_dim)
    call = calls[0]
    for name in ("q", "k", "v", "g"):
        assert call[name] == want, f"chunk_kda {name} {call[name]} != {want}"
    assert call["beta"] == (2, 7, cfg.n_heads)


# ---------------------------------------------------------------------------
# Gate 6: LOUD refusals (config validator and the hand-built module)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over,match",
    [
        ({"attn_latent_rank": -1}, "attn_latent_rank"),
        ({"attn_latent_rank": 8, "use_kda": False}, "use_kda=false"),
        ({"n_kv_heads": -1}, "n_kv_heads"),
        ({"n_kv_heads": 8}, "> n_heads"),
        ({"n_kv_heads": 3}, "does not divide"),
        ({"n_kv_heads": 2, "use_kda": False}, "use_kda=false"),
        ({"out_proj_rank": -1}, "out_proj_rank"),
        ({"out_proj_rank": 65}, "outside"),
    ],
)
def test_config_refusals_are_loud(over, match):
    cfg = _cfg(**over)
    with pytest.raises(LoudError, match=match):
        validate(cfg)


@pytest.mark.parametrize(
    "over,match",
    [
        ({"attn_latent_rank": -1}, "attn_latent_rank"),
        ({"n_kv_heads": 3}, "does not divide"),
        ({"n_kv_heads": 8}, "n_kv_heads"),
    ],
)
def test_hand_built_module_refusals_are_loud(over, match):
    """`DormouseModel(cfg)` skips `validate`; the module must refuse too."""
    cfg = _cfg(**over)
    with pytest.raises(LoudError, match=match):
        KDAAttention(cfg)


def test_latent_with_factor_qat_is_allowed_and_explained():
    """`factor_qat` reaches TSCT factors only; the latent is a DENSE site.

    The pair is therefore disjoint, not ambiguous: `factor_qat=ternary` must
    still build and run with `attn_latent_rank > 0`. The gate also proves the
    latent is not silently left on the factor path (no TSCT pair appears in the
    attention subtree).
    """
    model = DormouseModel(_cfg(attn_latent_rank=16, factor_qat="ternary"))
    ids = (torch.arange(2 * 16, dtype=torch.long) % 256).reshape(2, 16)
    _, loss, _ = model(ids, ids)
    assert torch.isfinite(loss)
    assert model.loop.attn.latent_down is not None
    assert not [n for n in model.loop.attn.state_dict() if n.endswith((".u", ".v"))]


def test_out_proj_rank_on_the_tsct_rung_sets_that_readout_s_rank():
    """On TSCT, `out_proj_rank` is the one matrix's rank (not a second factor)."""
    model = DormouseModel(_cfg(out_proj_rank=16))
    assert model.loop.out_proj.rank == 16
    base = DormouseModel(_cfg())
    assert base.loop.out_proj.rank == base.cfg.rank
    assert isinstance(
        DormouseModel(_cfg(use_tsct=False, out_proj_rank=16)).loop.out_proj, DenseLowRank
    )


# ---------------------------------------------------------------------------
# Gate 7: no host sync in the changed forward
# ---------------------------------------------------------------------------


def test_the_changed_forward_does_not_sync():
    """`item`/`tolist`/`bool`/`float` raise while the new arms run.

    Scoped to the two changed modules: the CPU MoE's `_naive_dispatch` has a
    pre-existing `.tolist()` (`experts.py:1376`, outside this task) and is
    reported as a follow-up, not silently blessed here.
    """
    cfg = _cfg(attn_latent_rank=16, n_kv_heads=2, use_tsct=False, out_proj_rank=16)
    torch.manual_seed(0)
    model = DormouseModel(cfg)
    attn = model.loop.attn
    x = torch.randn(2, 8, cfg.d_model)
    readout = model.loop.out_proj

    names = ("item", "tolist", "bool", "__float__")
    saved = {n: getattr(torch.Tensor, n) for n in names}

    def boom(self, *a, **k):  # noqa: ANN001
        raise AssertionError("host sync in the cheap-width forward")

    try:
        for n in names:
            setattr(torch.Tensor, n, boom)
        y = attn(x)
        z = readout(y.reshape(-1, cfg.d_model))
    finally:
        for n, f in saved.items():
            setattr(torch.Tensor, n, f)
    assert y.shape == x.shape and z.shape == (2 * 8, cfg.d_model)


# ---------------------------------------------------------------------------
# The A/B driver: its arms, its COMMON, and its pre-registered verdict
# ---------------------------------------------------------------------------


def _load_script(name: str):
    root = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, root)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ab_cheapwidth_is_preregistered():
    ab = _load_script("ab_cheapwidth")
    assert [a for a, _ in ab.ARMS] == ["dense", "latent", "gqa", "latent+gqa", "all"]
    assert ab.STEPS == 600 and ab.SEEDS == (1, 2, 3) and ab.PRESET == "medium"
    assert ab.DATA.endswith("/data_local/mix4/web.bin")
    assert ab.EVAL.endswith("/data_local/mix4/web_eval.bin")
    assert ab.COMMON_TRAIN["mix_plan"] == "configs/mix_big.toml"
    assert ab.COMMON_MODEL["moe_kernel"] == "triton"
    # The arms differ ONLY in the three width keys.
    keys = {"attn_latent_rank", "n_kv_heads", "out_proj_rank"}
    dense = dict(ab.ARMS)["dense"]
    allf = dict(ab.ARMS)["all"]
    for arm, flags in ab.ARMS:
        assert set(flags) <= keys, (arm, flags)
        assert set(flags) == keys, (arm, flags)
    assert dense == {"attn_latent_rank": 0, "n_kv_heads": 0, "out_proj_rank": 0}
    assert allf["attn_latent_rank"] == 256 and allf["out_proj_rank"] == 256
    assert allf["n_kv_heads"] == 2


def test_ab_cheapwidth_verdict_is_the_preregistered_rule():
    ab = _load_script("ab_cheapwidth")
    good = [
        {
            "seed": s,
            "bpb": 3.0 + 0.01 * s,
            "scored": ab.WINDOW,
            "nan": 0,
            "ms_per_step": 100.0,
            "counters": {},
        }
        for s in (1, 2, 3)
    ]
    arm = [
        {
            "seed": s,
            "bpb": 3.005 + 0.01 * s,
            "scored": ab.WINDOW,
            "nan": 0,
            "ms_per_step": 90.0,
            "counters": {},
        }
        for s in (1, 2, 3)
    ]
    v = ab.verdict(good, arm, ab.WINDOW, ab.MAC_CUT["latent"])
    assert v["verdict"] == "adopt" and v["mac_cut"] > 0.2, v
    bad = [dict(r, bpb=r["bpb"] + 0.5) for r in arm]
    assert ab.verdict(good, bad, ab.WINDOW, ab.MAC_CUT["latent"])["verdict"] == "loss"
    broken = [dict(r, nan=1) for r in arm]
    assert ab.verdict(good, broken, ab.WINDOW, ab.MAC_CUT["latent"])["verdict"] == "void"
    # The cost-target half of the rule: no cut, no adoption.
    assert ab.verdict(good, arm, ab.WINDOW, 0.05)["verdict"] == "tie"
