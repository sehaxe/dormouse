"""`ffn_form`: dense vs tsct vs bank, their bills, their checkpoints, their lies.

The gates, each one fails on a scratch revert of the code it pins:

a. `ffn_form="bank"` is bit-identical to clean `main` (the hashes below were
   captured from the worktree at `1ab5020` BEFORE this file existed). The
   default must not move.
b. `"dense"` is exactly `3·d·f + f + d` and carries no rank-shaped parameter.
c. `"tsct"` at `r_g = r_d = min(d, f)` is the dense GLU REPARAMETERISED: set the
   factors to the dense weights and the logits agree to 1e-5.
d. `"tsct"` at a small rank has fewer parameters and is NOT output-equivalent,
   even when the factors are the best rank-`r` SVD approximation of dense.
e. every illegal `ffn_form` combination refuses with `Escape:`.
f. a checkpoint of one form loaded into another refuses and names the tensors;
   the same form round-trips through a file.
g. `dormouse info`'s params/MAC table for all three forms equals an independent
   hand computation at d=1536, f=4096.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re

import pytest
import torch
from typer.testing import CliRunner

from dormouse import width
from dormouse.cli import app
from dormouse.config import load_preset, validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count

#: (state_dict sha256, logits sha256, total params) captured at `1ab5020`
#: before the FFN field existed. `tiny_bank` is the default tree.
BANK_GOLDEN = (
    "d3360f3178dc1be2f3bfc6614f52927f762fc76054187098de8faa1bdb6d2d27",
    "57d0c0ee6c5b0240e81d95eaea18681d06ef2c9ca1400292c3f97948e79837ee",
    64_881,
)

FORM_OVER = {
    "bank": {},
    "dense": {"ffn_form": "dense", "n_experts": 1, "moe_topk": 0},
    "tsct": {
        "ffn_form": "tsct",
        "n_experts": 1,
        "moe_topk": 0,
        "ffn_rank_gate": 16,
        "ffn_rank_down": 16,
    },
}


def _cfg(**over):
    return dataclasses.replace(load_preset("tiny"), **over)


def _built(over: dict, seed: int = 0) -> DormouseModel:
    torch.manual_seed(seed)
    return DormouseModel(_cfg(**over))


def _state_hash(model: DormouseModel) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _logits_hash(model: DormouseModel, seed: int = 7) -> str:
    torch.manual_seed(seed)
    ids = (torch.arange(2 * 16, dtype=torch.long) % model.cfg.vocab).reshape(2, 16)
    model.eval()
    with torch.no_grad():
        logits, loss, _ = model(ids, ids)
    assert loss is not None and torch.isfinite(loss)
    h = hashlib.sha256()
    h.update(logits.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _ids(n: int = 2, t: int = 16) -> torch.Tensor:
    return (torch.arange(n * t, dtype=torch.long) % 256).reshape(n, t)


def _logits(model: DormouseModel) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out, _, _ = model(_ids(), _ids())
    return out


def _copy_non_ffn(dst: DormouseModel, src: DormouseModel) -> None:
    """Give both forms the same attention/engram/head so only the FFN differs."""
    with torch.no_grad():
        for name, value in src.state_dict().items():
            if name.startswith("loop.moe."):
                continue
            target = dict(dst.state_dict()).get(name)
            if target is not None and target.shape == value.shape:
                target.copy_(value)


def _wire_full_rank(dense: DormouseModel, tsct: DormouseModel) -> None:
    """`U = I`, `V = Wᵀ`, `s = 1`: an exact full-rank reparameterisation."""
    with torch.no_grad():
        dm = dense.loop.moe.experts[0]
        tm = tsct.loop.moe.experts[0]
        d = dm.d
        eye = torch.eye(d)
        for name in ("gate", "up"):
            getattr(tm, f"{name}_u").copy_(eye)
            getattr(tm, f"{name}_s").copy_(torch.ones(d))
            getattr(tm, f"{name}_v").copy_(getattr(dm, name).weight.T)
        tm.gate_b.copy_(dm.gate.bias)
        tm.down_u.copy_(dm.down.weight.T)
        tm.down_s.copy_(torch.ones(d))
        tm.down_v.copy_(eye)
        tm.down_b.copy_(dm.down.bias)


def _wire_best_rank(dense: DormouseModel, tsct: DormouseModel, rank: int) -> None:
    """The best rank-`rank` SVD approximation of each dense matrix."""
    with torch.no_grad():
        dm = dense.loop.moe.experts[0]
        tm = tsct.loop.moe.experts[0]
        for name in ("gate", "up"):
            u, s, vh = torch.linalg.svd(getattr(dm, name).weight.T, full_matrices=False)
            getattr(tm, f"{name}_u").copy_(u[:, :rank])
            getattr(tm, f"{name}_s").copy_(s[:rank])
            getattr(tm, f"{name}_v").copy_(vh[:rank])
        u, s, vh = torch.linalg.svd(dm.down.weight.T, full_matrices=False)
        tm.down_u.copy_(u[:, :rank])
        tm.down_s.copy_(s[:rank])
        tm.down_v.copy_(vh[:rank])
        tm.gate_b.copy_(dm.gate.bias)
        tm.down_b.copy_(dm.down.bias)


# ---------------------------------------------------------------------------
# a. the default is main's bank, bit for bit
# ---------------------------------------------------------------------------


def test_bank_default_is_bit_identical_to_clean_main():
    assert _cfg().ffn_form == "bank"
    model = _built({})
    state, logits, params = BANK_GOLDEN
    assert param_count(model) == params
    assert _state_hash(model) == state
    assert _logits_hash(model) == logits


# ---------------------------------------------------------------------------
# b. dense is really dense
# ---------------------------------------------------------------------------


def test_dense_has_exactly_three_plain_matmuls_and_no_rank():
    cfg = _cfg(**FORM_OVER["dense"])
    model = _built(FORM_OVER["dense"])
    d, f = cfg.d_model, cfg.d_ffn
    slot = model.loop.moe.experts[0]
    assert sum(p.numel() for p in slot.parameters()) == 3 * d * f + f + d
    shapes = {name: tuple(p.shape) for name, p in slot.named_parameters()}
    assert shapes == {
        "gate.weight": (f, d),
        "gate.bias": (f,),
        "up.weight": (f, d),
        "down.weight": (d, f),
        "down.bias": (d,),
    }
    assert not any(re.search(r"(_u|_v|_s)$|rank", name) for name in shapes), shapes
    # The whole model's delta against the bank is exactly the slot's delta:
    # no router, no shared expert and no shared down survive the form.
    bank = _built({})
    assert param_count(model) - param_count(bank) == (3 * d * f + f + d) - 20_832


# ---------------------------------------------------------------------------
# c. full-rank tsct IS dense
# ---------------------------------------------------------------------------


def test_tsct_at_full_rank_is_output_equivalent_to_dense():
    cfg_d = _cfg(**FORM_OVER["dense"])
    lo = min(cfg_d.d_model, cfg_d.d_ffn)
    cfg_t = dataclasses.replace(cfg_d, ffn_form="tsct", ffn_rank_gate=lo, ffn_rank_down=lo)
    dense = _built(FORM_OVER["dense"])
    tsct = _built(
        {
            **FORM_OVER["dense"],
            "ffn_form": "tsct",
            "ffn_rank_gate": lo,
            "ffn_rank_down": lo,
        }
    )
    _copy_non_ffn(tsct, dense)
    _wire_full_rank(dense, tsct)
    a, b = _logits(dense), _logits(tsct)
    max_diff = float((a - b).abs().max())
    assert max_diff <= 1e-5, f"full-rank tsct != dense: max|Δ|={max_diff:.3e}"
    # Not a parameter tie: the reparameterisation carries extra factors.
    assert width.ffn_params(
        "tsct", cfg_t.d_model, cfg_t.d_ffn, rank=cfg_t.rank, rank_gate=lo, rank_down=lo
    ) > width.ffn_params("dense", cfg_t.d_model, cfg_t.d_ffn)


# ---------------------------------------------------------------------------
# d. small-rank tsct is compression, not a free lunch
# ---------------------------------------------------------------------------


def test_small_rank_tsct_has_fewer_params_and_is_not_equivalent():
    dense = _built(FORM_OVER["dense"])
    small = _built(
        {
            **FORM_OVER["dense"],
            "ffn_form": "tsct",
            "ffn_rank_gate": 8,
            "ffn_rank_down": 8,
        },
        seed=0,
    )
    _copy_non_ffn(small, dense)
    _wire_best_rank(dense, small, 8)
    assert width.ffn_params("tsct", 64, 128, rank=8, rank_gate=8, rank_down=8) < (
        width.ffn_params("dense", 64, 128)
    )
    assert sum(p.numel() for p in small.loop.moe.parameters()) < sum(
        p.numel() for p in dense.loop.moe.parameters()
    )
    max_diff = float((_logits(dense) - _logits(small)).abs().max())
    assert max_diff > 1e-2, f"rank-8 tsct matched dense (max|Δ|={max_diff:.3e})"


# ---------------------------------------------------------------------------
# e. every illegal combination refuses with Escape:
# ---------------------------------------------------------------------------


def _refuses(cfg, needle: str) -> None:
    with pytest.raises(LoudError, match="Escape:"):
        validate(cfg)
    with pytest.raises(LoudError, match=needle):
        validate(cfg)


def test_refusals_are_loud_and_name_the_escape():
    _refuses(_cfg(ffn_form="dense", n_experts=4), "no bank")
    _refuses(
        _cfg(ffn_form="dense", n_experts=1, moe_topk=0, share_u_mode="plane"),
        "no factor basis",
    )
    _refuses(_cfg(ffn_form="tsct", n_experts=1, moe_topk=0, rank=65), "not a factorisation")
    packed = _cfg(ffn_form="tsct", n_experts=1, moe_topk=0)
    # `tsct_packed` lives on the unmerged packed-bank branch; the refusal is
    # written with `getattr` so the two branches cannot silently diverge.
    packed.tsct_packed = True
    with pytest.raises(LoudError, match="Escape:"):
        validate(packed)
    with pytest.raises(LoudError, match="BANK feature"):
        validate(packed)
    _refuses(_cfg(ffn_form="bank", ffn_rank_gate=8), "only the `tsct` form")


# ---------------------------------------------------------------------------
# f. checkpoints: same form round-trips, wrong form names the tensors
# ---------------------------------------------------------------------------


def test_checkpoints_round_trip_per_form_and_wrong_form_names_the_tensors(tmp_path):
    models: dict[str, DormouseModel] = {}
    paths = {}
    for form, over in FORM_OVER.items():
        model = _built(over)
        sd = model.state_dict()
        # Same form, fresh weights: strict load and identical logits.
        fresh = _built(over, seed=3)
        fresh.load_state_dict(sd)
        torch.testing.assert_close(_logits(model), _logits(fresh))
        path = tmp_path / f"{form}.pt"
        torch.save({"model": sd}, path)
        models[form] = _built(over, seed=5)
        paths[form] = path

    for src_form, path in paths.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        for dst_form, model in models.items():
            if dst_form == src_form:
                model.load_state_dict(ck["model"])
                continue
            want = set(model.state_dict())
            missing = sorted(set(ck["model"]) - want)
            assert missing, f"{src_form} -> {dst_form}: no key differs"
            with pytest.raises(RuntimeError) as err:
                model.load_state_dict(ck["model"])
            msg = str(err.value)
            assert any(name in msg for name in missing), (
                f"{src_form} -> {dst_form}: refusal did not name a missing tensor; "
                f"missing={missing[:3]}, error={msg[:300]}"
            )


# ---------------------------------------------------------------------------
# g. info's table is the independently computed bill
# ---------------------------------------------------------------------------


def _info_row(output: str, form: str) -> tuple[int, int]:
    line = next(x for x in output.splitlines() if "│" in x and x.lstrip("│ ").startswith(form))
    nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", line)]
    return nums[0], nums[1]


def test_info_prints_the_independently_computed_table():
    out = CliRunner().invoke(
        app,
        [
            "info",
            "m400",
            "--set",
            "n_experts=1",
            "--set",
            "moe_topk=0",
        ],
    )
    assert out.exit_code == 0, out.output
    d, f, iters, rank, shared = 1536, 4096, 3, 64, 192
    # Independent hand sums: the body is `iters·(4·d² + 2·d·rank)` because
    # `use_tsct=true` prices the readout at `rank`; 1 byte = 1 token.
    body = iters * (4 * d * d + 2 * d * rank)
    dense_params = 3 * d * f + f + d
    tsct_params = (2 * rank + rank) * (d + f + 1) + f + d
    bank_params = (
        (2 * d * rank + 2 * rank + rank * f)
        + (2 * d * shared + 2 * shared + shared * f)
        + d * f
        + d
    )
    want = {
        "bank": (
            bank_params,
            body + iters * d * f + (2 * d * rank + rank * f) + (2 * d * shared + shared * f) + d,
        ),
        "dense": (dense_params, body + iters * dense_params),
        "tsct": (tsct_params, body + iters * tsct_params),
    }
    for form, (params, macs) in want.items():
        got_params, got_macs = _info_row(out.output, form)
        assert (got_params, got_macs) == (params, macs), (
            f"{form}: info {got_params:,}/{got_macs:,} != table {params:,}/{macs:,}"
        )
    core = iters * (4 * d * d + 2 * d * rank)
    assert (
        width.macs_per_token(
            d,
            iters=iters,
            n_heads=24,
            out_proj_rank=rank,
            f=f,
            ffn_form="dense",
        )
        == core + iters * dense_params
    )
    assert (
        width.macs_per_token(
            d,
            iters=iters,
            n_heads=24,
            out_proj_rank=rank,
            f=f,
            ffn_form="tsct",
            ffn_rank=rank,
        )
        == core + iters * tsct_params
    )
    # And the formula is the module tree at tiny shape, not just arithmetic.
    for form, over in FORM_OVER.items():
        model = _built(over)
        cfg = model.cfg
        slot = sum(p.numel() for p in model.loop.moe.parameters())
        assert slot == width.ffn_params(
            form,
            cfg.d_model,
            cfg.d_ffn,
            rank=cfg.rank,
            rank_gate=cfg.ffn_rank_gate,
            rank_down=cfg.ffn_rank_down,
            n_experts=cfg.n_experts,
            shared_rank=cfg.shared_rank,
        )
